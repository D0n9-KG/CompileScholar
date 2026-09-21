# Perspective

# Dielectric Metasurfaces Enabling Advanced Optical Biosensors

Ming Lun Tseng, Yasaman Jahani, Aleksandrs Leitis, and Hatice Altug

ACS Photonics, Just Accepted Manuscript • Publication Date (Web): 14 Sep 2020

Downloaded from pubs.acs.org on September 14, 2020

# Just Accepted

"Just Accepted" manuscripts have been peer-reviewed and accepted for publication. They are posted online prior to technical editing, formatting for publication and author proofing. The American Chemical Society provides "Just Accepted" as a service to the research community to expedite the dissemination of scientific material as soon as possible after acceptance. "Just Accepted" manuscripts appear in full in PDF format accompanied by an HTML abstract. "Just Accepted" manuscripts have been fully peer reviewed, but should not be considered the official version of record. They are citable by the Digital Object Identifier (DOI®). "Just Accepted" is an optional service offered to authors. Therefore, the "Just Accepted" Web site may not include all articles that will be published in the journal. After a manuscript is technically edited and formatted, it will be removed from the "Just Accepted" Web site and published as an ASAP article. Note that technical editing may introduce minor changes to the manuscript text and/or graphics which could affect content, and all legal disclaimers and ethical guidelines that apply to the journal pertain. ACS cannot be held responsible for errors or consequences arising from the use of information contained in these "Just Accepted" manuscripts.

# Dielectric Metasurfaces Enabling Advanced Optical Biosensors

Ming Lun Tseng $^{1*}$ , Yasaman Jahani $^{1*}$ , Aleksandrs Leitis $^{1*}$ , and Hatice Altug $^{1†}$

$^{1}$ Institute of BioEngineering, École Polytechnique Fédérale de Lausanne (EPFL), 1015 Lausanne, Switzerland.

\*Equally contributed authors

$^{\dagger}$ Corresponding Author: Hatice Altug, E-mail: hatice.altug@epfl.ch

# ABSTRACT

Dielectric metasurfaces have emerged as a powerful platform for novel optical biosensors. Mainly due to their low optical loss and strong light-matter interaction, they demonstrate several exotic optical properties, including sharp resonances, strong near-field enhancements, and the compelling capability to support magnetic modes. They also show advantages such as CMOS-compatible fabrication processes and lower resonance-induced heating compared to their plasmonic counterparts. These unique characteristics are enabling the advancement of cutting-edge sensing techniques for new applications. In this perspective, we review the recent progress of dielectric metasurface sensors. First, the working mechanisms and properties of dielectric metasurfaces are briefly introduced by highlighting several state-of-art examples. Next, we describe the application of dielectric metasurfaces for label-free sensing in three different detection schemes, namely refractometric sensing, surface-enhanced spectroscopy through Raman scattering and infrared absorption, and chiral sensing. Finally, we provide a perspective for the future directions of this exciting research topic.

KEYWORDS: dielectric metasurface, biosensing, bound state in the continuum, chiral sensing, surface-enhanced infrared absorption, surface enhanced Raman sensing

# TOC art

![](images/8310353ad8eb0ae3d16d2b305f462ed1f6db392e145138153b05ff9e3fb52e1b.jpg)

<details>
<summary>natural_image</summary>

Illustration of a microfluidic cell culture with bacterial colonies and DNA helix structures (no text or symbols)
</details>

Optical sensing techniques are essential in modern biomedical, material, and environmental monitoring applications as they provide fast and robust ways to detect and distinguish target objects from a variety of samples. In recent years, nanophotonic metasurfaces have been emerging as a powerful platform to advance sensing technologies. Metasurfaces are artificial photonic devices composed of subwavelength nanoresonators (called meta-atoms), and their functionalities can be engineered by rationally tailoring their geometric parameters $^{1-4}$ . Metasurfaces have been used as wavefront control devices, $^{5-8}$ nanoscale light sources, $^{9, 10}$ optical signal modulators, $^{11, 12}$ and optical quantum devices. $^{13, 14}$ For optical sensing, their capabilities to harvest light into nanoscale electromagnetic hotspots have been utilized to enhance the sensitivity of various detection techniques $^{15-17}$ For instance, plasmonic metasurfaces have enabled the detection of monolayers and their real-time interactions with biomolecules in the mid-infrared (mid-IR) regime, $^{18-22}$ which is challenging by using conventional infrared absorption spectroscopy methods. So far, metasurfaces consisting of metallic nanostructures (e.g., Au, Ag, and Al) have been exploited dominantly, and numerous biosensing devices have been demonstrated. $^{21, 23-25}$ Although plasmonic metasurfaces provide several useful properties, e.g., strong electric field enhancement, for various sensing techniques, the intrinsic losses of the constituent metals usually impose limitations. An emerging direction to circumvent such restraints is to replace the constituent materials of metasurfaces with dielectrics that have low optical loss and high refractive index. Thanks to their striking optical properties dielectric metasurfaces bring a new horizon for nanophotonic sensing schemes. In this perspective, we provide an overview of the state of the art for this rapidly developing research topic as well as an outlook for the possible future directions. This paper is organized as follows. In the beginning, we present a brief introduction of dielectric metasurfaces and their resonance mechanisms. Next, we review three major label-free sensing approaches based on refractometric detection, surface-enhanced spectroscopy for molecular specific detection, as well as chiral molecular sensing. Finally, we summarize current challenges and emerging directions.

Dielectric metasurface. The physics of optical resonances in dielectric metasurfaces is mostly associated with Mie scattering of subwavelength resonators. $^{7, 26}$ As the effective wavelength of the incident light is close to the feature size of a high-index nanoparticle, optical resonances arising from the oscillating displacement currents can be excited. Notably, individual subwavelength dielectric resonators can show a strong response to both electric and magnetic fields of light. This property is in contrast to the plasmonic counterparts that respond weakly to the magnetic field. As a result, electrical and magnetic dipoles, as the first and second Mie-type modes (Figure 1a), are usually observed in the spectra of dielectric resonators. A transmission spectrum and the resonance modes of a spherical Si nanoparticle with a diameter of 150 nm were simulated using commercial software CST Microwave Studio (Figure 1b). In the transmission spectrum, two dips, associated with the electric (at \~490 nm) and the magnetic (at \~600 nm) modes, can be observed, respectively. As an experimental demonstration of the resonances in dielectric nanoparticles, a dark-field image of Si nanoparticle made by laser ablation of a Si substrate is presented in Figure 1c $^{27}$ . The spherical nanoparticles show vibrant colors, ranging from violet to red. As highlighted by the red and green dashed boxes in the images, the red and the green spots in the image are associated to Si nanoparticles with different diameters. The dependence of the resonance wavelength and the dimension of a dielectric nanoparticle can be described as: $g(\lambda/n)=2\pi r$ , where $g$ is a positive integer, $\lambda$ is the resonance wavelength, and $r$ is the radius, respectively. $^{28, 29}$ Therefore, for different sensing techniques, the resonance wavelengths of dielectric metasurfaces can be precisely tuned from the visible, $^{27, 30}$ to infrared (IR) $^{31, 32}$ by the choice of the feature sizes of the meta-atoms and the materials. This useful property enables the realization of various sensing metasurfaces.

In the design of sensing metasurfaces, several parameters should be carefully considered to achieve the targeted functionalities, including the working wavelength, resonance bandwidth, field enhancement, and the resonance quality. Especially, the resonance quality and the near-field enhancement are important parameters for most of the sensing mechanisms. The quality factor of an optical resonator (Q) is defined as the ratio of the resonance wavelength versus the resonance bandwidth. For instance, in refractometric

biosensing the Q factor influences the resolution to measure refractive index changes $^{33, 34}$ . Additionally, Q factor is closely associated with the field enhancement given by the resonators. Their relationship can be described as $P^{2}=\gamma_{\mathrm{rad}}(Q^{2}/V)$ , where P is the field enhancement, $\gamma_{rad}$ is the radiative decay ratio, and V is the mode volume of the resonator. $^{35, 36}$ Therefore, metasurfaces with a high-quality factor (high-Q) resonance are also useful for different techniques, such as surface-enhanced spectroscopy. $^{18-22}$ On the other hand, a higher Q-factor implies a narrower bandwidth, which can bring certain limitations for techniques such as SEIRA and vibrational circular dichroism spectroscopy $^{37}$ . It is because a narrower resonance implies a reduced spectral bandwidth, and it might not be able to cover well the molecular absorption bands for chemical specific detection. Therefore, it is important to be aware of such trade-offs and consider device architectures that can benefit from high-Q while still having a broad spectral coverage. Several examples will be discussed in the following sections related to this point.

a   
![](images/8a1babdb1896ec67aaa093755ad9a31200146441bc9d9ef0b0abe415729bb70e.jpg)

<details>
<summary>text_image</summary>

Electric dipole
EⓍ→HⓍ
kⓍ
Magnetic dipole
HⓍ→EⓍ
kⓍ
</details>

b   
![](images/9ef5d3dd8668aea7729dd54af7f2f7bc580c74007ea4029976ab32705078868f.jpg)

<details>
<summary>heatmap</summary>

| Wavelength (nm) | Electric dipole (E-field a.u.) | Magnetic dipole (M-field a.u.) |
| --------------- | ------------------------------ | ------------------------------ |
| 450             | 1.0                            | 1.0                            |
| 500             | 0.1                            | 0.1                            |
| 550             | 0.6                            | 0.6                            |
| 600             | 0.0                            | 0.0                            |
| 650             | 1.0                            | 1.0                            |
</details>

C   
![](images/923b91da0d092d89e22ac819aba3e7acfca110dbc656eb0ccc49b36ea1ed2372.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing two gray square clusters with white circular spots, surrounded by scattered bright spots and a 3 μm scale bar (no text or symbols beyond scale)
</details>

Figure 1. Mie resonances in high-index dielectric nanostructures. (a) Schematics of electric dipole and magnetic dipole induced in a spherical dielectric nanoparticle. The red and blue arrow indicate the direction of the induced magnetic and electric filed inside the nanoparticle, respectively. (b) Transmission spectrum and the resonance modes of a 150-nm Si spherical nanoparticle. (c) Dark-field microscopic image of Si nanoparticles on a Si substrate produced by using femtosecond laser ablation $^{27}$ . Insert: SEM

image of the Si nanoparticles highlighted in the dashed squares. Scale bar of the inserts: 200 nm. (c) is taken with permission from ref 27. Copyright 2012 Nature Publishing Group.

As a brief overview of the current progress on high-Q dielectric metasurfaces, several state-of-art works are presented in Figure 2. Yang et al uses the co-existence of strong magnetic and electric dipole resonances in dielectric nanoresonators to achieve a Fano-type resonance in a metasurface $^{38}$ . A scanning electron microscopic (SEM) image of the metasurface is presented in Figure 2a. It has been designed for operation at the telecommunication wavelength band and Si was selected as the constituent material due to its low loss property in the IR regime.

The meta-atom consists of two nanoresonators, a nanobar, and a nanodisk. The electric dipole mode of the nanobar resonator serves as the bright mode while the out-of-plane magnetic dipole mode of the nanodisk resonator is optically dark in nature. The coupling of the bright and dark modes suppresses the far-field radiation and results in a sharp spectral feature. As a result, the transmission spectrum of the metasurface reveals a sharp resonance peak at 1371 nm with a Q factor of 483. Another useful approach for designing high-Q metasurfaces is to introduce a mode interference inside the individual resonators. Semmlinger et al. demonstrate the excitation of an anapole resonance $^{39-41}$ in a dielectric metasurface. Anapole resonance results from destructive interference between the electric and toroidal dipoles excited in the meta-atoms, and it is useful for boosting light-matter interactions. (Figure 2b). The meta-atom of the metasurface is a nanodisk, as can be seen in the left panel of Figure 2b. To achieve a proper resonance property in the anapole metasurface, amorphous $TiO_{2}$ is used as the constituent material because of its low loss in the visible regime. The anapole resonance produced a sharp dip in the transmission spectrum, as can be observed in the right panel of Figure 2b. Notably, the simulation results for both of the metasurfaces (the inserts in Figure 2a and 2b) illustrate that electromagnetic energy is confined on the surface of meta-atoms at the resonance condition. This characteristic is beneficial to boost the light-matter interactions at the nanoscale for many applications, including several biosensing techniques.

More recently, a new approach for achieving high-Q metasurfaces based on the quasi bound state in the continuum (quasi-BIC) has attracted significant attention. $^{42-44}$ BICs are the optical modes that remain ideally confined even though they are compatible with free-space radiation. $^{43, 44}$ A true BIC is a mathematical object with an infinite Q factor and a vanishing resonance. Within the concept of metasurfaces, at the BIC condition, the resonance feature in the transmission spectrum collapses due to the transformation of the resonance mode to be optically dark. Symmetry protected BIC can become quasi-BIC by breaking the in-plane symmetry of the unit cell with a controllable asymmetry parameter. This property provides so-called supercavity modes $^{45}$ with finite but high-Q factors. Absorption influences the total Q factor of a quasi-BIC resonance, yet the radiative Q factor for all quasi-BIC supporting metasurfaces has an inverse quadratic relation with the asymmetry parameter of the unit cell. This quadratic correlation, along with the linear scalability of Maxwell's equations, can be employed to engineer metasurfaces with desired properties. BIC modes have been previously observed in plasmonic metasurfaces by Lovera et al $^{46}$ . They proposed a link between Fano resonances and BIC modes in their work. Later, the theory of quasi-BIC has been developed and systematically explained by Koshelev et al. within the context of dielectric metasurfaces. $^{42, 45}$ An example of dielectric metasurface based on the physics of BIC is presented in Figure 2c $^{42}$ . The metasurface consists of pairs of Si nanoellipses. The middle panel of Figure 2c represents that when the nanoellipses are parallel, a pure BIC mode appears with an infinite Q factor in the spectra and spectrally vanishes. By breaking the in-plane symmetry with a gradual tilting of the ellipses, a radiation channel is created, and the BIC is turned into a quasi-BIC mode having a finite and tunable $Q$ factor. A resonance dip can be seen in the spectra, and their quality factor varies with the rotation angle. Simulation results of the electric field distribution for the symmetric and asymmetric nanoellipse pair at the resonance wavelength are presented in the bottom of Figure 2c. It displays similar field patterns, a characteristic of the transition between the pure BIC mode to the quasi-BIC mode. With such an exceptional type of optical resonance, a Si metasurface with a record-high-Q factor has been recently realized (Figure 2d). $^{36}$ Similarly, the authors introduce an in-plane inversion

symmetry breaking by cutting out part of the edges. The metasurface provides an experimentally measured Q factor up to 18511 for operation in the telecommunication wavelength range. These recent works show that the type of optical resonances, functionalities, and the working wavelengths can be effectively tuned by changing the material, geometry, and arrangement of the unit cell. In the following sections, we will review implementation of dielectric metasurfaces for sensing.

![](images/21891540b835ea9daf53319539e98f8831af879eab1c43035f65c081cc4aaee3.jpg)

<details>
<summary>text_image</summary>

a
Silicon
x-y plane
Quartz
1 µm
Transmission
1300 1320 1340 1360 1380 1400
Wavelength (nm)
</details>

![](images/f138f73d78aa15e1a3e5d59c07e2ce27963dce1dc0114f4ab3c8959471299f3b.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Transmission (θ=10°) | Transmission (θ=8°) | Transmission (θ=6°) | Transmission (θ=5°) | Transmission (θ=4°) | Transmission (θ=3°) | Transmission (θ=2°) | Transmission (θ=1°) | Transmission (θ=0.5°) | Transmission (θ=0°) | Transmission (BIC) |
| --------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | ---------------------- | --------------------- | ------------------- |
| 1563            | ~0.9                  | ~0.8                  | ~0.7                  | ~0.6                  | ~0.5                  | ~0.4                  | ~0.3                  | ~0.2                  | ~0.1                   | ~0.05                 | ~0.0                |
| 1565            | ~0.8                  | ~0.7                  | ~0.6                  | ~0.5                  | ~0.4                  | ~0.3                  | ~0.2                  | ~0.1                  | ~0.05                  | ~0.0                  | ~0.0                |
| 1567            | ~0.7                  | ~0.6                  | ~0.5                  | ~0.4                  | ~0.3                  | ~0.2                  | ~0.1                  | ~0.05                 | ~0.0                   | ~0.0                  | ~0.0                |
| 1569            | ~0.6                  | ~0.5                  | ~0.4                  | ~0.3                  | ~0.2                  | ~0.1                  | ~0.05                 | ~0.0                   | ~0.0                   | ~0.0                  | ~0.0                |
| 1571            | ~0.5                  | ~0.4                  | ~0.3                  | ~0.2                  | ~0.1                  | ~0.05                 | ~0.0                   | ~0.0                   | ~0.0                   | ~0.0                  | ~0.0                |
</details>

![](images/5e1235428b81250a9151dbac3c9e4b29bf1196d983b70901cc6eca2765b321fb.jpg)

<details>
<summary>text_image</summary>

b
500nm
Transmission
450 500 550 600 650 700
Wavelength (nm)
</details>

![](images/e1bca4852b3cd0796a98618f06bbd0708e7a5f5998fb3108238ffbe80d77b94b.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Transmission |
| --------------- | ------------ |
| 1487            | ~1.0         |
| 1509            | ~1.0         |
| 1512            | ~1.0         |
| 1536            | ~1.0         |
| 1557            | ~1.0         |
</details>

![](images/c5d69a35cb0b644405c8c025b9769a7c3181f6bdebd087f108a732fdfcee53a0.jpg)

<details>
<summary>text_image</summary>

BIC
quasi-BIC
0
1
|E|
</details>

Figure 2. Examples of high-Q dielectric metasurfaces. (a) Fano-resonant metasurface $^{38}$ . Left panel: SEM image of the metasurface consisting of Si bars and disks. Insert: the electric field pattern at the resonance wavelength. Right panel: simulation (purple curve) and experiment (red) transmission spectrum of the metasurface. (b) Anapole metasurface $^{41}$ . Left panel: SEM image of the metasurface consisting of $TiO_{2}$ disks. Insert: the electric field pattern at the resonance wavelength. Right panel: Transmission spectrum

of the metasurface. (c) Quasi-BIC metasurfaces with different symmetry breakings $^{42}$ . Top panel: Schematic of the metasurface consisting of asymmetric Si asymmetric nanoellipse pairs. Middle panel: Dependence of the transmission spectra versus the rotational angle $\theta$ (indicated in the top panel). Bottom panel: Comparison of the electric field patterns for the BIC and quasi-BIC mode, respectively. (d) Ultrahigh-Q factor quasi-BIC metasurface $^{47}$ . Left panel: SEM image of the metasurface consisting of Si nanoresonators. Insert on the top: the electric field pattern at the resonance wavelength. Right panel: Transmission spectra of the metasurfaces with the different cut-out size S. The Q factor of the metasurface is highly sensitive to S. The polarization of the excitation for the metasurfaces shown in this figure is along the x-axis shown in (d). (a) is taken with permission from ref ${}^{38}$ . Copyright 2015 ACS. (b) is taken with permission from ref ${}^{41}$ . Copyright 2019 ACS. (c) is taken with permission from ref ${}^{42}$ . Copyright 2018 APS. (d) is taken with permission from ref ${}^{47}$ . Copyright 2017 APS.

Refractometric biosensing. The strong Mie-type resonances in dielectric metasurfaces provide near-field hotspots giving rise to pronounced light-matter interactions with the analytes in the vicinity of the resonators. By carefully tailoring the metasurface design, the resonance can be sensitive and responsive to the dielectric properties of the surrounding media. Analyte binding on the resonators creates local changes in the refractive index, which can be detected and quantified by tracking the resonance wavelength or the induced changes in the transmission or reflection intensity. Sensitivity is a measure of the extent of this resonance shifts for a given change in the refractive index, which depends on the material, geometry, and arrangement of the nanostructures. A comprehensive parameter to appraise the performance of refractometric biosensors is figure-of-merit (FOM), defined as the ratio of the sensitivity divided to the full-width at half-maximum (FWHM) of the resonance, thus depends on the Q factor.

One of the initial works of dielectric metasurfaces for biosensing was demonstrated by Si nanodisks for end-point measurements $^{48}$ . The metasurface was fabricated by electron beam lithography (EBL) and operated in the telecommunication wavelength regime where Si is nearly lossless. The authors used a spectrometry system to track the resonance wavelength shift. The biosensor shows a resonance at around

1488 nm with a Q factor of 25. It is capable of detecting selectively down to 5 ng.ml $^{-1}$ of streptavidin when the surface is functionalized with Biotin molecules. Later, EBL fabricated Si nanodisk structures, optimized for shorter near-IR wavelengths, were integrated with microfluidics to perform bulk refractive index sensing as well as prostate-specific antigen (PSA) detection in a sandwich assay leveraging single-point spectroscopy. $^{49}$ Figure 3a shows the microfluidic structure, a close view of 8 independent sensing channels, and an SEM image of the Si nanodisks. The setup was equipped with a stage scanning system to acquire information from multiple spatial points on the sensing chip sequentially. The nanodisks, manifesting Mie-type resonances at 844 nm with a Q factor of 20, bulk sensitivity as high as 227 nm.RIU $^{-1}$ and in turn FOM of 5.4 RIU $^{-1}$ , were reported to have a limit of detection (LOD) down to 0.69 ng.mL $^{-1}$ of PSA when a second antibody was used to amplify the signal (Figure 3b). The extinction spectra of the nanodisks in contact with various concentrations of glucose solutions are shown in Figure 3c. The sensing performance of the Si nanodisks was compared with the gold nanorod arrays under the same conditions except for the surface chemistry. The normalized calibration curves for both platforms detecting PSA in a sandwich assay are shown in Figure 3d. The dielectric metasurface offers a better LOD compared to the plasmonic counterpart, while the plasmonic nanorods afford a wider dynamic range.

In another effort, semi-random Si nanocylinder arrays were fabricated using colloidal lithography as an effective technique to fulfill the need for low-cost and scalable metasurface-based biosensors that are suitable for lab-on-a-chip systems. $^{50}$ The structures exhibit a Mie-type resonance at 900 nm in water with a Q factor of slightly lower than the EBL fabricated nanodisks mentioned above and a bulk sensitivity of 86 nm.RIU $^{-1}$ . The metasurface was assembled to a microfluidic-based spectroscopic platform equipped with spatial scanning arrangements to detect PSA in a sandwich assay. The experimental data are represented both in terms of resonance centroid shift in spectroscopic interrogation, and extinction reduction in ensemble-averaged intensity interrogation. They observed that the latter approach provides a wider dynamic range and a lower LOD. The calibration curve for the resonance centroid shift, shown in Figure 3e, characterizes the sensor with an LOD and a dynamic range of 1.55 ng.mL $^{-1}$ and 2.35-9.79

ng.mL $^{-1}$ , respectively, while the intensity interrogation achieved an LOD and a dynamic range of 0.83 ng.mL $^{-1}$ and 1.87-30 ng.mL $^{-1}$ , correspondingly. The results suggest that further investigations of the intensity interrogation methods are essential towards realizing high-performance spectrometer-less biosensing platforms for point-of-care (POC) devices. In the intensity interrogation methods, the spectral location of the resonance also gains special attention as the working wavelength range under 1000 nm allows the use of standard low-cost Si cameras for compact and affordable integrated lab-on-chip systems.

High-Q factor metasurfaces are favorable for upsurging the sensitivity of refractometric biosensors and for bringing in new functionalities. As an example, a high-Q factor Fano-resonant metasurface $^{51}$ consists of symmetry-broken Si nanoresonators, working in the telecommunication wavelength range, was proposed to exhibit a bulk sensitivity as high as 289 nm.RIU $^{-1}$ and an FOM of 103. Moreover, metasurfaces supporting quasi-BIC offer high-Q factor resonances manifested as supercavity modes. Recently, BIC-type dielectric metasurfaces were used in combination with hyperspectral imaging to perform ultrasensitive biomolecule detection. $^{52}$ The unit cell comprises of EBL fabricated pairs of Si nanoellipses that are tilted around their y-axis in mirror symmetry (Figure 3g). The metasurface exhibited a resonance dip around 855 nm for measurements in aqueous media, a Q factor as high as 144, and a bulk sensitivity of 263 nm.RIU $^{-1}$ . The hyperspectral imaging platform shown in Figure 3f consists of a supercontinuum laser source coupled to a narrowband tunable filter for high spectral resolution ( $\Delta\lambda=0.1$ nm) acquisition. The resonance wavelength range of the metasurface allows the use of a large-area CMOS camera to image simultaneously multiple sensors units on the chip. The transmission spectrum of each sensor is extracted from the data cube with high spatial resolution and processed to create a resonance map, which provides the resonance wavelength at each pixel. The spatially resolved resonance shifts caused by the analyte binding on the resonators are extracted from the resonance shift maps that are computed by subtracting the resonance maps of the metasurface with and without the analyte. For highly diluted analytes, due to the stochastic nature of the binding events, the low number of molecules does not induce measurable spectral shifts, while analyte presence can be confirmed visually from the extracted

resonance shift maps (Figure 3i). To exploit this feature for high-performance sensing, a thresholding method called receiver operating characteristic (ROC) curve, was implemented, and the results (Figure 3h) shows the ability to detect IgG molecules as low as 3 molecules per $\mu \mathrm{m}^2$ . The work illustrates the superior sensing performance of the imaging techniques in comparison with the ensemble averaging methods commonly used in conventional spectrometer-based measurements. Moreover, it highlights the unexplored potentials of data processing techniques for advanced biosensing. Extracting spectral information from spatial data was also utilized with chirped guided-mode resonant gratings.[53] These structures offering a bulk sensitivity of $137~\mathrm{nm.RIU^{-1}}$ were used for the detection of IgG molecules from a $500~\mathrm{ng.mL^{-1}}$ solution. Overall, imaging-based schemes hold a great promise for ultrasensitive spectrometer-less biosensing towards the realization of the POC devices as an enabling key for personalized medicine and global healthcare monitoring.

![](images/5b8a4104aa213cd2b5740c852ecdde7b3e3bfa224bd9878c00d0493a52ea64aa.jpg)

![](images/78873b3680e0b33b5fac11baefebe07872ee5f425b073d0057145945c4996866.jpg)

<details>
<summary>line</summary>

| [PSA] (ng/ml) | Centroid shift (nm) |
| ------------- | ------------------- |
| 1             | 0.1                 |
| 10            | 0.2                 |
| 100           | 0.4                 |
| 1000          | 0.6                 |
</details>

![](images/e9acf3eb2fdd0500ed593359b12f42e3b1f92c4f8a290178e19caee7882cb0cd.jpg)

<details>
<summary>scatter</summary>

| average number of IgG molecules (molecules/µm²) | AUC |
| --- | --- |
| 10^0 | 0.55 |
| 10^1 | 0.65 |
| 10^2 | 0.75 |
| 10^3 | 0.85 |
| 10^4 | 0.95 |
</details>

![](images/3a581fe19db638d6d463b4c6d2fc2b8ed8460d2efc1379694e27910e276c2dd9.jpg)

<details>
<summary>text_image</summary>

Control
2.84 molecules/µm²
55.3 molecules/µm²
2.79 x 10³ molecules/µm²
50 µm
0
Δλ_res(nm)
1
0
Δλ_res(nm)
3+
</details>

Figure 3. Refractometric biosensing metasurfaces. (a) Picture of a multichannel microfluidic chip (left) with a close-up picture of the eight flow channels showing multiple sensors (middle) and an SEM image of a Si nanodisk metasurface (right). $^{49}$ (b) Schematic of a sandwich bioassay on a Si nanodisks. $^{49}$ (c) Extinction spectra a nanodisk array exposed to six different glucose water mixtures. (d) Normalized calibration curve of the centroid shifts for PSA detection with Si nanodisks and gold nanorods using a sandwich immunoassay for signal amplification. Error bars represent the replicas of the measurement on

the same chip. The inset illustrates the extinction spectra of the Si nanodisk arrays and gold nanorod arrays. $^{49}$ . (e) Calibration curve of the centroid shifts for PSA detection with semi-random Si nanocylinders. $^{50}$ (f) Principle of hyperspectral imaging-based biosensing using tilted Si nanoellipses. $^{52}$ (g) SEM images of the tilted nanoellipses metasurface (top) and its unit cell (bottom). $^{52}$ (h) The area under the curve (AUC) values of all receiver operating characteristic (ROC) curves show the detection of biomolecules at low concentrations. $^{52}$ (i) The resonance shift map for a control measurement, detection of two low concentrations of IgG molecules, and a high concentration of the biomolecule from left to right, respectively $^{52}$ . (a)-(d) are taken with permission from ref 49. Copyright 2017 ACS. (e) is taken with permission from ref 50. Copyright 2019 ACS. (f)-(i) are taken with permission from ref 52. Copyright 2019 Nature Publishing Group.

Surface-enhanced spectroscopy. An alternative method to refractometric detection for biomolecule sensing is vibrational spectroscopy, which allows not only to detect the surface-bound molecules, but also provides chemically specific information about each analyte. Spectroscopy techniques, like Raman spectroscopy and IR absorption spectroscopy, enable label-free sensing and monitoring of chemical reactions and interaction kinetics between various analytes. Raman spectroscopy probes molecular vibrations via inelastic scattering mechanism of the photons, while the IR spectroscopy access the chemical information via light absorption measurements. Although these techniques have been well established for chemical analysis, they are fundamentally limited by the low sensitivity, which arises from weak light-matter interaction in IR spectral region and low molecule cross-section for Raman scattering. Therefore, the study of low-concentration analytes is challenging when using conventional spectroscopy methods. This problem can be addressed by employing nanophotonic enhancement from well-designed metasurfaces through surface-enhanced Raman spectroscopy (SERS) $^{16, 54}$ and surface-enhanced infrared absorption spectroscopy (SEIRAS) $^{15, 21, 55}$ .

Heretofore, SERS has been realized mainly with metallic nanostructures $^{24, 56-59}$ and metasurfaces $^{60-62}$ , which provide strong near-field enhancement. Although their surface enhancement

effectively improves the signals, still high laser intensity and tight light focusing on the metasurface are needed. This causes intense heating in metal-based metasurfaces, which can cause damage to the biomolecules $^{63, 64}$ , change in the local refractive index $^{65}$ , or even deform the metasurface itself $^{66}$ . Therefore, SERS measurements with metal-based metasurfaces are prone to repeatability issues. $^{54}$ Recently dielectric metasurfaces have emerged as an alternative to plasmonic-based metasurfaces for Raman spectroscopy. The near-zero absorption losses in the constituent materials of dielectric metasurfaces allow reducing the generated heat, while the high refractive index enables profound light confinement. An example is presented by Caldarola et al., using a metasurface consisting of a Si dimer array on a Si-on-insulator substrate (Figure 4a) $^{67}$ . It was successfully implemented for SERS to detect a thin layer of polymethyl-methacrylate (PMMA) coated on the metasurface. A similar Si dimer metasurface was fabricated on a sapphire substrate to detect a self-assembled monolayer of $\beta$ -carotenal molecules (Figure 4b) $^{68}$ . Here, the chemically specific $\beta$ -carotenal signal appears as two pronounced Raman peaks, located at 1150 cm $^{-1}$ and 1520cm $^{-1}$ . To further improve the nanophotonic enhancement, a metasurface that sustains quasi-BIC (Q>10 $^{3}$ ) modes was exploited $^{69}$ . It consists of an array of nanoholes in a Si $_{3}$ N $_{4}$ slab and shows Raman enhancement factors larger than 10 $^{3}$ for crystal violet molecules (Figure 4c). Although there have been initial efforts toward dielectric metasurfaces for SERS, metal-based antennas still provide higher field enhancements and improved SERS signals. Nevertheless, dielectric metasurfaces could be further improved, for instance, by optimizing the designs of the meta-atoms that sustain high-Q resonances and support the highest near-field enhancements outside the resonator volume. In this way, the near-fields can be accessed by the biomolecules and eventually results in higher signal values. Furthermore, the wide variety of accessible dielectric materials enable diverse surface functionalization methods for analyte bonding $^{70}$ and increased chemical enhancement $^{64}$ .

Similar to SERS, SEIRA measurements have been performed mostly by using metal-based metasurfaces. The plasmonic antennas are well suited for signal enhancement, however, even in the IR spectral region, they are fundamentally limited by the Ohmic losses. This significantly broadens their

resonances and results in higher full width at FWHMs when compared to the molecular absorption bands.

Therefore to be able to detect infrared absorption bands and obtain the chemical information, bulky spectroscopic equipment is usually required $^{71}$ . Dielectric materials empower the realization of higher Q-factor resonances in the mid-IR, where the FWHM of the resonance is sharper than the molecular absorption bands of the bioanalytes. This property allows to target the molecular absorption bands with better spectral selectivity while maintaining high sensitivity. $^{72}$ High-Q factor metasurfaces for SEIRA and their implementation in a large-area imaging configuration was first demonstrated by Tittl et al. In order to maintain a broad bandwidth, they utilized a two-dimensional (2D) array of pixelated metasurfaces over a large-area such that each ‘metapixel’ of the array is tuned to a specific resonant frequency due its high-Q factor and spectrally clean optical response. This feature provided a one-to-one mapping between the spectral and the spatial information and enabled to access the molecular signals across an extended wavelength range at high sensitivity (Figure 4d). $^{73, 74}$ Interestingly, with this 2D pixelated metasurface method, the absorbance of the surface-bound analytes (e.g., monolayered protein, see the bottom panels of Figure 4d) can be extracted directly from the total reflectance signals, thus eliminating the need for expensive spectroscopy equipment or tunable light sources. Another approach based on high-Q dielectric metasurfaces was demonstrated by Leitis et al. (Figure 4e), where the resonance of a single metasurface was tuned by light incidence angle and polarization to retrieve molecular fingerprint information over a broad spectrum ranging from 1100 to 1800 cm $^{-1}$ (i.e., \~5 to \~9 μm in wavelength). $^{75}$ The specific resonator shape provides the maximum near-field enhancement at the tips of the nanoellipses, where the bioanalytes can interact with light. Therefore, it enables chemical fingerprint detection with high sensitivity only using broadband light sources and detectors. This method allows combining the device-level simplicity of the state-of-the-art angle-scanning refractometric sensors with the chemical specificity of infrared spectroscopy. An example of using this metasurface-equipped system for detecting polylysine, single stranded DNA and human odontogenic ameloblast-associated proteins (ODAM) is presented in the

bottom panel of the Figure 4e. The high-Q dielectric metasurfaces hold great potential for miniaturizing IR sensors, which could be well suited for point-of-care devices.

![](images/6384f394d3ba36e06ed8bc2c0cb531808b8d1c3a793acf57f5f9cf54be4d81a0.jpg)

<details>
<summary>scatter</summary>

| Material | Heating laser intensity (mW/μm²) | Tgap (°C) |
|----------|-----------------------------------|-----------|
| Au       | 0                                 | 20        |
| Au       | 2                                 | 30        |
| Au       | 4                                 | 60        |
| Au       | 6                                 | 100       |
| Au       | 8                                 | 110       |
| Au_th    | 0                                 | 20        |
| Au_th    | 2                                 | 30        |
| Au_th    | 4                                 | 60        |
| Au_th    | 6                                 | 100       |
| Au_th    | 8                                 | 110       |
| Si       | 0                                 | 20        |
| Si       | 2                                 | 30        |
| Si       | 4                                 | 60        |
| Si       | 6                                 | 100       |
| Si       | 8                                 | 110       |
| Si_th    | 0                                 | 20        |
| Si_th    | 2                                 | 30        |
| Si_th    | 4                                 | 60        |
| Si_th    | 6                                 | 100       |
| Si_th    | 8                                 | 110       |
</details>

![](images/a1a6923bc7498b44b6eaf66ebb95007729b8dd94ab662f66fee5a7525eeeb275.jpg)

<details>
<summary>line</summary>

| Wavenumber (cm⁻¹) | Normalised Reflectance |
| ----------------- | ---------------------- |
| 1400              | ~0.9                   |
| 1600              | ~0.8                   |
| 1800              | ~0.9                   |
</details>

![](images/43e78d74962623e5caaf58395720f542e39bfc2e1011fd5ee25ba8740ff67cec.jpg)

<details>
<summary>line</summary>

| Raman shift (cm⁻¹) | Raman signal (a.u.) |
| ------------------ | ------------------- |
| 1100               | 0.6                 |
| 1300               | 0.2                 |
| 1500               | 0.7                 |
</details>

![](images/b6df4a294d8337ae83f84d06f08be41f626f1a7e9b58729e90726d6e9d68a748.jpg)

<details>
<summary>text_image</summary>

e
TMx
θ
Ge
CaF₂
z y x
TMx
θ=55°
θ=13°
TEy
θ=55°
PMMA
Wavenumber (cm⁻¹)
1200
1400
1600
1800
Normalised reflectance
0.0
0.5
1.0
z y x
</details>

![](images/bdd4ab987859dc27661e87d91032ab7cb86df57c0d8c7cd1169d853adace98b0.jpg)

<details>
<summary>line</summary>

| Raman shift (cm⁻¹) | Intensity (counts/s) ×10³ |
| ------------------ | ------------------------ |
| 800                | ~0.4                     |
| 1200               | ~0.6                     |
| 1600               | ~0.7                     |
</details>

![](images/d49adf8a98f58bbc6e72c2a5527bc5f71ba6446b223ba22e5bfdb8d57626091c.jpg)

<details>
<summary>line</summary>

| Wavenumber (cm⁻¹) | Normalized reflectance (polylysine) | Normalized reflectance (ODAM protein) | AFR signal (a.u.) |
| ----------------- | ----------------------------------- | ------------------------------------- | ----------------- |
| 1180              | ~0.0                                | ~0.0                                  | ~0.0              |
| 1200              | ~0.5                                | ~0.5                                  | ~0.5              |
| 1220              | ~0.0                                | ~0.0                                  | ~0.0              |

| Wavenumber (cm⁻¹) | Normalized reflectance (polylysine) | Normalized reflectance (ODAM protein) | AFR signal (a.u.) |
| ----------------- | ----------------------------------- | ------------------------------------- | ------------------ |
| 1180              | ~0.0                                | ~0.0                                  | ~0.0              |
| 1200              | ~0.5                                | ~0.5                                  | ~0.5              |
| 1220              | ~0.0                                | ~0.0                                  | ~0.0              |

| Wavenumber (cm⁻¹) | Normalized reflectance (polylysine) | Standard Model (ssDNA) | AFR signal (a.u.) |
| ----------------- | ----------------------------------- | ------------------------ | ------------------ |
| 1180              | ~0.0                                | ~0.0                                  | ~0.0              |
| 1200              | ~0.5                                | ~0.5                                  | ~0.5              |
| 1220              | ~0.0                                | ~0.0                                  | ~0.0              |

| Wavenumber (cm⁻¹) | Normalized reflectance (polylysene) | Normalized reflectance (ODAM protein) | AFR signal (a.u.) |
| ----------------- | ----------------------------------- | -------------------------------------- | ------------------ |
| 1180              | ~0.0                                | ~0.0                                  | ~0.0              |
| 1200              | ~0.5                                | ~0.5                                  | ~0.5              |
| 1220              | ~0.0                                | ~0.0                                  | ~0.0              |

| Wavenumber (cm⁻¹) | Normalized reflectance (polylysne) | Normalized reflectance (ODAM protein) | AFR signal (a.u.) |
| ----------------- | ----------------------------------- | -------------------------------------- | ------------------ |
| 1180              | ~0.0                                | ~0.0                                  | ~0.0              |
| 1200              | ~0.5                                | ~0.5                                  | ~0.5              |
| 1220              | ~0.0                                | (~0.0)                                 | ~0.0              |

| Wavenumber (cm⁻¹) | Normalized reflectance (polylysine) | Normalized reflectance (ODAM protein) | AFR signal (a.u.) |
| ----------------- | ----------------------------------- | -------------------------------------- | ------------------ |
| 1180              | ~0.0                                | ~0.0                                  | ~0.0              |
| 1200              | ~0.5                                | ~0.5                                  | ~0.5              |
| 1220              | ~0.0                                | ~(0.0)                                 | ~0.0              |

| Wavenumber (cm⁻¹) | Normalized reflectance (polylysine) | Normalized reflectance (ODAM protein) | AFR signal (a.u.) |
| ----------------- | ----------------------------------- | -------------------------------------- | ------------------ |
| 1180              | ~0.0                                | ~0.0                                  | ~0.0              |
| 1200              | ~1.0                                | ~1.0                                   | ~1.5               |
| 1220              | ~0.5                                | ~0.5                                   | ~1.5               |
| 1600              | ~0.5                                | ~1.5                                   | ~1.5               |
| 1800              | ~0.5                                | ~1.5                                   | ~1.5               |
</details>

Figure 4. Dielectric metasurfaces for surface-enhanced spectroscopy. (a) Left panel: Electric field enhancement in a Si dimer structure and the temperature dependence on the laser intensity $^{67}$ . Right panel: SEM images of Si dimers where the scale bar on the bottom micrographs is 100nm. (b) SERS signal of $\beta$ -carotenal monolayer using Si dimer metasurfaces $^{68}$ . (c) SERS enhancement of crystal violet molecules using high-Q factor nanohole metasurfaces $^{69}$ . (d) Top panel: Illustration of pixelated metasurface and unit cell coated with protein molecules. Bottom panel: Measurements of high-Q factor metasurface response after protein monolayer deposition and imaging-based molecular fingerprint detection $^{73,74}$ . (e) Top panel: unit cell of a Ge metasurface and the spectral response for each light incidence angle. Bottom panel:

spectral integration signal that emulates the total reflectance signal and the measured absorbance signals from polylysine, ssDNA aptamers, and ODAM proteins $^{75}$ . (a) is taken with permission from ref 67. Copyright 2015 Nature Publishing Group. (b) is taken with permission from ref 68. Copyright 2018 ACS. (c) is taken with permission from ref 69. Copyright 2018 ACS. (d) is taken with permission from refs 73, 74. Copyright 2018 AAAS and 2019 OSA publishing. (e) is taken with permission from ref 75. Copyright 2019 AAAS.

Chiral sensing. Chirality, a geometric symmetry property in objects, gives rise to pivotal physical and chemical properties and plays a significant role in protein function, cell communication, and in general human physiology. Many naturally occurring biomolecules and chemically synthesized drugs can be present in both left- and right-handed forms, i.e., enantiomers. They can show significant divergences in their metabolic uptake, treatment potential, and toxicologic effects. It has been demonstrated that distortions in the chiral-molecular structure participate in the appearance of neurodegenerative and neuropsychiatric disorders such as Alzheimer's and Parkinson's diseases. $^{76-78}$

Chiral molecules manifest their chiral attributes when interacting with the chiral state of light, i.e., circularly polarized light (CPL). As a result, discrimination of enantiomers can be performed observing their differential absorption of left and right CPL, conventionally done with circular dichroism (CD) spectroscopy. However, CD spectrometers are sophisticated, bulky, and expensive; hence their veritable competent employment is hindered by the intrinsically weak chiroptical signals of most small molecules.

Metasurfaces can reinforce light-matter interaction in chiral sensing by creating fields with large concentrations of optical chirality, C, defined as:

$$
C \{E, H \} = \frac {- k _ {0}}{2 c _ {0}} \operatorname{Im} \left\{E \cdot H ^ {*} \right\} \tag {1}
$$

Here $k_{0}$ and $c_{0}$ stand for the wavenumber and speed of light in free space, respectively, while E and H represent the complex electric and magnetic fields. $^{79}$ Mohammadi et al. formulated an analytical model

to explain the interplay of the characteristics and the performance of an arbitrary metasurface to elucidate the key parameters in their design and unravel the involved trade-offs. The analytical model defines the criteria of the best performing chirality detection by the nanophotonic platform as (i) Removing the background signal emanated from the sensing device and taking into account the attenuation role of its chiral absorption. (ii) Equal intensity enhancement factors for the right and left CPL to remove the permittivity dependence of the total output signal. (iii) High optical chirality to maximize the output signal.

Metasurfaces can be designed to offer strong, uniform, and accessible electromagnetic fields for efficient C intensification. Since planar metallic nanostructures usually show weak responses to the magnetic field of the incident light, complex three-dimensional chiral nanostructures were studied as an alternative for chiral enhancement. $^{80-82}$ However, the experimental realization of such structures is challenging in the context of nanofabrication. Moreover, such chiral nanoresonators are prone to introduce background signals emanated from the chiral nanostructures, which can obscure the CD signal of the analyte.

In contrast to plasmonics, the strong response of dielectric nanoresonators to both of the electric and magnetic fields of light makes dielectric metasurface an attractive choice for chirality enhancement. Mohammadi et al. proposed a planar achiral metasurface made from Si nanodisks to successfully enhance the minuscule tail of CD signal in the technology-friendly range of the spectrum, i.e., visible and near IR below 1000 nm. $^{79}$ The metasurface provides extremely high CD enhancement of the order of 30. Later, Si nanodisks were used to experimentally differentiate L- and D- phenylalanine from a 200 nm thick molecular thermal evaporated layer yielding a dense film (Figures 5a,5b). $^{83}$ The spectrometer-based recorded CD signal showed a strong dependence on the electric dipole.

Further investigations by Mohammadi et al. led to the tailoring of the spectrally and specially overlapped electric and magnetic resonances with a $\frac{\pi}{2}$ phase-difference to fulfill the C enhancement

conditions by exploiting the Kerker effect based on the duality principle in electromagnetics (Figure 5c). $^{84}$ The authors proposed a Si-based metasurface consists of Kerker-inspired disks with a nanohole at the center (Figure 5d) $^{85}$ to provide uniformly distributed and accessible super chiral near-fields to enhance the differential absorption (also transmission) by a factor of 24.

The maximum chiroptical signature of most small molecules takes place in the UV and IR spectrum termed electronic and vibrational CD (VCD), respectively. Hence, exploiting the capabilities of the nanophotonic platforms in these ranges enlightens the enhancement mechanisms and performances of such devices. A biperiodic diamond disk metasurface in Figure 5e was proposed theoretically to enhance the averaged C for more than two orders of magnitude in the UV while the local C enhancement reached 1100. $^{84}$ The Q factor of a resonance correlates with the C enhancement due to its contribution to concentrating the electromagnetic fields. In another theoretical work, a metasurface made of PbTe nanodimers (Figure 5f) was conceived to support quasi-BIC modes in mid-IR, $^{86}$ exhibiting a resonance with a Q factor of 800 and a local C enhancement of 270 for VCD measurements. Dielectric metasurfaces offer exciting prospects toward developing high-performance sensing devices for easy, fast, and economical stereoisomers differentiation, which is of great importance for the pharmaceutical industry.

![](images/6848c63e4198212392591a04a10f6493a1674d0ae110d97682154661aa291fbc.jpg)

<details>
<summary>text_image</summary>

a
-10°
-160 nm
-160 nm
300 nm
-200 nm
phenylalanine
Si
1 µm
</details>

![](images/89c4ee6230ed7c5e470640a7ffee103f587fae1b32b3454ad7568d3e1f749273.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | CD (mdeg) - Sensor | CD (mdeg) - Phe |
| --------------- | ------------------ | --------------- |
| 600             | ~0                 | ~0              |
| 700             | ~250               | ~-150           |
| 800             | ~0                 | ~0              |
</details>

![](images/870dd181cd6de5984a6dd9613fb1794b754c4828a68ab3bfb375e1fa52291777.jpg)

<details>
<summary>chemical</summary>

Diagram illustrating dual E and H fields for maximizing C, showing ED and MD interactions with labeled components
</details>

![](images/738009c18603aed4757c6b787c6ae04021be68a60e56951021e5def50515ff73.jpg)

<details>
<summary>text_image</summary>

d
m_y
p_x
m_x
py
y
x
</details>

![](images/163ef04f74ff88a8828a7955533e6ca653ceeb3d8cf05063c818ffbd1cdd1f8a.jpg)

<details>
<summary>line</summary>

| wavelength (nm) | simulation | Analytical model |
|---|---|---|
| 750 | 2.0 | 2.0 |
| 800 | 10.0 | 8.0 |
| 850 | 24.0 | 16.0 |
| 900 | 3.0 | 3.0 |
| 950 | 1.0 | 1.0 |
</details>

![](images/9b451436970b53d5dcf59d9a861c60b78d8679420020ae17455a02663c7ea675.jpg)

<details>
<summary>text_image</summary>

e
d-Δ/2
← d+Δ/2
← a
h
</details>

![](images/a9341194ddae194b242beebf6d52446e31a12675b946ac4e018c057b4cfa958f.jpg)

<details>
<summary>text_image</summary>

f
LCP
RCP
</details>

![](images/dd88c0903cbefaa39d18b3d6ed0ac5e981c20484b2b4c4304f784dc06807786b.jpg)

<details>
<summary>line</summary>

| Wavelength, nm | Chirality enhancement C/C₀ |
| -------------- | -------------------------- |
| 3985           | ~1                         |
| 3995           | ~50                        |
| 4005           | ~280                       |
| 4015           | ~50                        |
| 4025           | ~10                        |
</details>

![](images/7dc99e242a355c710da4b782c9e65fc350d2d567a7faac077325c0194496c1b6.jpg)

<details>
<summary>line</summary>

| z(nm) | C/C CPL (black line) | C/C avg (green line) |
|-------|----------------------|----------------------|
| 0     | ~1000                | ~500                 |
| 20    | ~700                 | ~400                 |
| 40    | ~500                 | ~300                 |
| 60    | ~300                 | ~200                 |
| 80    | ~150                 | ~100                 |
| 100   | ~50                  | ~50                  |
| 120   | ~20                  | ~20                  |
| 140   | ~10                  | ~10                  |
| 160   | ~5                   | ~5                   |
| 180   | ~2                   | ~2                   |
| 200   | ~1                   | ~1                   |
</details>

Figure 5. Chiral sensing using dielectric metasurfaces. (a) A side-view SEM image of Si nanodisks covered with a dense 200 nm of phenylalanine. $^{83}$ (b) CD signals differentiating molecular enantiomers of phenylalanine using Si nanodisks. $^{83}$ (c) Schematic of generating near-field enhancement using electromagnetic duality principle. $^{85}$ (d) Schematic of a Si disk nanostructure with a hole for Kerker-inspired optical chirality enhancement. $^{85}$ The plot on the right shows normalized differential transmission from numerical and analytical studies. $^{85}$ (e) A dielectric metasurface for chiral field enhancement, based on the asymmetric diamond disk array, capable of boosting the near-field in the UV spectrum. $^{84}$ The bottom plot shows its optical chirality enhancement. (f) Asymmetric PbTe bars proposed for VCD measurements by optical chirality intensification. $^{86}$ The plot on the right depicts the local optical chirality enhancement. (a)-(b) are taken with permission from ref 83. Copyright 2020 ACS. (c)-(d) are taken with permission from ref 85. Copyright 2019 ACS. (e) is taken with permission from ref 84. Copyright 2019 ACS. (f) is taken with permission from ref 86. Copyright 2019 OSA Publishing.

Outlook. The highlighted recent demonstrations of dielectric metasurfaces illustrate their immense potential for biosensing. We anticipate that dielectric metasurfaces will have a substantial impact to design on-demand functionalities due to the additional degrees of freedom in their optical response and their unique capabilities to boost the light-matter interactions. Although they have shown great promises, several open challenges still need to be addressed. One of them is to develop a universal method for large-scale fabrication of different types of dielectric metasurface sensors. Despite some works for efficient production of metasurfaces have been reported lately, including nano-imprinting $^{87, 88}$ , laser direct write $^{89, 90}$ , and hole-mask lithography $^{91-93}$ , methods that are compatible with standard semiconductor processes are yet to be developed. Another direction for future studies is to improve the sensitivities of dielectric metasurfaces. Even though the current studies show that dielectric metasurfaces can provide sufficient sensitivities, their overall performance has not reached the levels of the state-of-art plasmonic counterparts. Especially for SERS, there is room to further boost the device sensitivity.

Furthermore, progress in material science is beneficial for the advancement of dielectric metasurfaces and their device implementation. In Figure 6, we provide an overview of low-loss dielectric materials (k<0.001), which are suitable candidates for metasurface sensors in the UV-Visible and IR regimes. Some of the suggested materials have already been implemented, such as $TiO_{2}$ in the visible regime and Si in the IR regime. On the other hand, the utilizations of many others for metasurfaces are yet to be explored, and they can be investigated to open up new territories in dielectric metasurfaces. For example, as shown in Figure 6a, by using high bandgap materials, such as diamond, AlN, and $HfO_{2}$ , the working wavelength of dielectric metasurface sensors can be extended into the UV regime. It will be useful for sensing technologies such as resonance Raman spectroscopy $^{94}$ and UV circular dichroism spectroscopy $^{95}$ . By using active materials, such as GaAs and GaN, multifunctional metasurfaces capable of not only generating near-field enhancements but also working as on-chip light sources can be realized for ultra-compact sensor integration. In addition, some unconventional materials, including high-index chalcogenides $^{96, 97}$ , multiple quantum wells $^{98, 99}$ , phase-change media $^{100, 101}$ , and perovskites $^{102, 103}$ have

been recently used for metasurfaces to realize new optical properties. Further investigation of the unconventional materials for metasurfaces could lead to high-performance optical biosensors.

![](images/e9c6e5cdb67144b3b92f59bccd8c2c9be0fab72c6e135c2d2050bb6baf7ecb1c.jpg)

<details>
<summary>heatmap</summary>

| Material          | Wavelength (nm) | Refractive index n |
| ----------------- | --------------- | ------------------ |
| Si                | 3.7             | ~3.6               |
| GaP               | 3.5             | ~3.1               |
| GaN               | 2.5             | ~2.3               |
| TiO₂              | 2.3             | ~2.2               |
| Si₃N₄             | ~2              | ~2                 |
| AlN               | ~2              | ~2                 |
| Diamond           | ~2.5            | ~2.5               |
</details>

![](images/84a54f9122ee63a63bfbe6500d136be4bf2a4377068488b5bae3d4dea8750c30.jpg)

<details>
<summary>line</summary>

| Material | Wavelength (μm) | Refractive index n |
| -------- | --------------- | ------------------ |
| PbTe     | ~5.9            | 6.0                |
| Te       | 4.9-4.7         | 5.0                |
| Ge       | ~4.0            | 4.0                |
| Si       | ~3.5            | 3.5                |
| GaAs     | 3.5-3.2         | 3.5                |
| GaP      | ~3.1            | 3.0                |
</details>

Figure 6. Refractive index (n) of low-loss dielectric materials for metasurfaces at (a) the UV-Visible regime and (b) the IR regime. The double arrow lines indicate the wavelength range where the related materials have near-zero extinction coefficient. The height of the double-headed arrows corresponds to the average value of the refractive index in the wavelength ranges. Possible sensing techniques of dielectric metasurfaces at different regimes are labeled on the top of the figure.

Concurrently, miniaturization of the current bioanalytical and clinical devices can be addressed by metasurface-based sensing technologies. Performing rapid, efficient, and accurate diagnosis is of great importance for the timely treatment of life-threatening diseases. As we have witnessed during the COVID-19 pandemic, devices enabling point-of-care testing and continuous health monitoring are becoming critical for saving lives and improving life quality $^{104, 105}$ . While the integration of plasmonic metasurfaces into systems such as CMOS-based chips $^{23, 106, 107}$ , microfluidic devices $^{21, 22, 108}$ , and lab-on-a-chip systems $^{109}$ has been more established, integrating dielectric metasurfaces into them is in its infancy and

expected to leverage accumulated knowledge for achieving new functionalities. For instance, light-emitting metausurfaces $^{8}$ , reconfigurable metasurfaces $^{11, 100, 110, 111}$ , and pixelated metasurfaces $^{73}$ can enable compact sensors by incorporating on-chip light generation, signal processing, and surface enhancement.

Finally, artificial intelligence can be considered as another competent tool to extend the capacity of metasurface sensors. Novel approaches for designing metasurfaces and advanced data processing are the new directions to investigate. Several works have shown the unprecedented capability of machine learning for inverse designs of metasurfaces with the assigned optical properties $^{112, 113}$ . It can be useful to realize metasurface sensors with properties that require strict criteria, such as super chiral field generation and spectrally clean sharp resonances. By combining machine learning and novel dielectric metasurfaces, optical sensors capable of performing molecular pattern recognition, distinguishing molecules in complex environments, and the observation of the dynamic molecular interactions can be achieved.

Conclusion. In this Perspective, we have covered the current progress of dielectric metasurfaces for biosensing applications. Their unique properties have enabled the realization of new sensing schemes including imaging-based biomolecular detections, and surface-enhanced chiral sensing and spectroscopy. We have also discussed future directions as well as the key challenges for this emerging research topic. We expect that dielectric metasurfaces will have impact on photonic sensing due to their compatibility to the standard semiconductor fabrication processes, design flexibility and accessibility to a wider choice of materials.

# AUTHOR INFORMATION

# Corresponding Author

\*(H.A.) E-mail: hatice.altug@epfl.ch

# ORCID:

Ming Lun Tseng: 0000-0003-0418-8162

Aleksandrs Leitis: 0000-0003-4855-5876

Yasaman Jahani: 0000-0003-2345-2264

Hatice Altug: 0000-0001-5522-1343

ACKNOWLEDGMENTS The authors gratefully acknowledge funding from the European Research Council (ERC) under grant agreement no.682167 VIBRANT-BIO and the European Union Horizon 2020 Framework Programme for Research and Innovation under Grant Agreements No.FETOPEN-737071 (ULTRA-CHIRAL Project), no. 777714 (NOCTURNO project), no. 875672 (POCSEL project). The authors also acknowledge the comments from Deepthy Kavungal and Dr. Kseniia Boriachek, EPFL.

Notes The authors declare no competing financial interest.

# REFERENCES

1. Koshelev, K.; Kivshar, Y. Dielectric Resonant Metaphotonics. ACS Photon. 2020, in press.   
2. Meinzer, N.; Barnes, W. L.; Hooper, I. R. Plasmonic meta-atoms and metasurfaces. Nat. Photon. 2014, 8, 889-898.   
3. Monticone, F.; Alu, A. Metamaterial, plasmonic and nanophotonic devices. Rep. Prog. Phys. 2017, 80, 036401.   
4. Kildishev, A. V.; Boltasseva, A.; Shalaev, V. M. Planar Photonics with Metasurfaces. Science 2013, 339, 1289.   
5. Yu, N.; Capasso, F. Flat optics with designer metasurfaces. Nat. Mater. 2014, 13, 139-150.   
6. Tseng, M. L.; Hsiao, H.-H.; Chu, C. H.; Chen, M. K.; Sun, G.; Liu, A.-Q.; Tsai, D. P. Metalenses: Advances and Applications. Adv. Opt. Mater. 2018, 6, 1800554.   
7. Kruk, S.; Kivshar, Y. Functional Meta-Optics and Nanophotonics Governed by Mie Resonances. ACS Photon. 2017, 4, 2638-2649.   
8. Colburn, S.; Zhan, A.; Majumdar, A. Varifocal zoom imaging with large area focal length adjustable metalenses. Optica 2018, 5, 825-831.   
9. Wang, D.; Wang, W.; Knudson, M. P.; Schatz, G. C.; Odom, T. W. Structural Engineering in Plasmon Nanolasers. Chem. Rev. 2018, 118, 2865-2881.   
10. Staude, I.; Pertsch, T.; Kivshar, Y. S. All-Dielectric Resonant Meta-Optics Lightens up. ACS Photon. 2019, 6, 802-814.   
11. Ou, J.-Y.; Plum, E.; Zhang, J.; Zheludev, N. I. An electromechanically reconfigurable plasmonic metamaterial operating in the near-infrared. Nat. Nanotechnol. 2013, 8, 252-255.   
12. Fang, X.; Lun Tseng, M.; Ou, J.-Y.; MacDonald, K. F.; Ping Tsai, D.; Zheludev, N. I. Ultrafast all-optical switching via coherent modulation of metamaterial absorption. Appl. Phys. Lett. 2014, 104, 141102.

13. Wang, K.; Titchener, J. G.; Kruk, S. S.; Xu, L.; Chung, H.-P.; Parry, M.; Kravchenko, I. I.; Chen, Y.-H.; Solntsev, A. S.; Kivshar, Y. S.; Neshev, D. N.; Sukhorukov, A. A. Quantum metasurface for multiphoton interference and state reconstruction. Science 2018, 361, 1104-1108.   
14. Li, L.; Liu, Z.; Ren, X.; Wang, S.; Su, V.-C.; Chen, M.-K.; Chu, C. H.; Kuo, H. Y.; Liu, B.; Zang, W.; Guo, G.; Zhang, L.; Wang, Z.; Zhu, S.; Tsai, D. P. Metalens-array-based high-dimensional and multiphoton quantum source. Science 2020, 368, 1487-1490.   
15. Neubrech, F.; Huck, C.; Weber, K.; Pucci, A.; Giessen, H. Surface-Enhanced Infrared Spectroscopy Using Resonant Nanoantennas. Chem. Rev. 2017, 117, 5110.   
16. Wang, P.; Nasir, M. E.; Krasavin, A. V.; Dickson, W.; Jiang, Y.; Zayats, A. V. Plasmonic Metamaterials for Nanochemistry and Sensing. Acc. Chem. Res. 2019, 52, 3018-3028.   
17. Halas, N. J.; Lal, S.; Chang, W.-S.; Link, S.; Nordlander, P. Plasmons in Strongly Coupled Metallic Nanostructures. Chem. Rev. 2011, 111, 3913-3961.   
18. Wu, C.; Khanikaev, A. B.; Adato, R.; Arju, N.; Yanik, A. A.; Altug, H.; Shvets, G. Fano-resonant asymmetric metamaterials for ultrasensitive spectroscopy and identification of molecular monolayers. Nat. Mater. 2012, 11, 69-75.   
19. Dong, L.; Yang, X.; Zhang, C.; Cerjan, B.; Zhou, L.; Tseng, M. L.; Zhang, Y.; Alabastri, A.; Nordlander, P.; Halas, N. J. Nanogapped Au Antennas for Ultrasensitive Surface-Enhanced Infrared Absorption Spectroscopy. Nano Lett. 2017, 17, 5768-5774.   
20. Rodrigo, D.; Limaj, O.; Janner, D.; Etezadi, D.; García de Abajo, F. J.; Pruneri, V.; Altug, H. Mid-infrared plasmonic biosensing with graphene. Science 2015, 349, 165-168.   
21. Rodrigo, D.; Tittl, A.; Ait-Bouziad, N.; John-Herpin, A.; Limaj, O.; Kelly, C.; Yoo, D.; Wittenberg, N. J.; Oh, S. H.; Lashuel, H. A.; Altug, H. Resolving molecule-specific information in dynamic lipid membrane processes with multi-resonant infrared metasurfaces. Nat. Commun. 2018, 9, 2160.   
22. John-Herpin, A.; Tittl, A.; Altug, H. Quantifying the Limits of Detection of Surface-Enhanced Infrared Spectroscopy with Grating Order-Coupled Nanogap Antennas. ACS Photon. 2018, 5, 4117-4124.   
23. Cerjan, B.; Halas, N. J. Toward a Nanophotonic Nose: A Compressive Sensing-Enhanced, Optoelectronic Mid-Infrared Spectrometer. ACS Photon. 2018, 6, 79-86.   
24. Gwo, S.; Wang, C.-Y.; Chen, H.-Y.; Lin, M.-H.; Sun, L.; Li, X.; Chen, W.-L.; Chang, Y.-M.; Ahn, H. Plasmonic Metasurfaces for Nonlinear Optics and Quantitative SERS. ACS Photon. 2016, 3, 1371-1384.   
25. Lee, J.; Park, J.; Lee, J. Y.; Yeo, J. S. Contact Transfer Printing of Side Edge Prefunctionalized Nanoplasmonic Arrays for Flexible microRNA Biosensor. Adv. Sci. (Weinh) 2015, 2, 1500121.   
26. Jahani, S.; Jacob, Z. All-dielectric metamaterials. Nat. Nanotechnol. 2016, 11, 23-36.   
27. Kuznetsov, A. I.; Miroshnichenko, A. E.; Fu, Y. H.; Zhang, J.; Luk'yanchuk, B. Magnetic light. Sci. Rep. 2012, 2, 492.   
28. Bohren, C. F.; Huffman, D. R., Absorption and scattering of light by small particles. John Wiley & Sons: 2008.   
29. Cao, L.; Fan, P.; Barnard, E. S.; Brown, A. M.; Brongersma, M. L. Tuning the Color of Silicon Nanostructures. Nano Lett. 201, 10, 2649-2654.   
30. Semmlinger, M.; Tseng, M. L.; Yang, J.; Zhang, M.; Zhang, C.; Tsai, W.-Y.; Tsai, D. P.; Nordlander, P.; Halas, N. J. Vacuum Ultraviolet Light-Generating Metasurface. Nano Lett. 2018, 18, 5738.   
31. Ginn, J. C.; Brener, I.; Peters, D. W.; Wendt, J. R.; Stevens, J. O.; Hines, P. F.; Basilio, L. I.; Warne, L. K.; Ihlefeld, J. F.; Clem, P. G.; Sinclair, M. B. Realizing optical magnetism from dielectric metamaterials. Phys. Rev. Lett. 2012, 108, 097402.   
32. Shcherbakov, M. R.; Neshev, D. N.; Hopkins, B.; Shorokhov, A. S.; Staude, I.; Melik-Gaykazyan, E. V.; Decker, M.; Ezhov, A. A.; Miroshnichenko, A. E.; Brener, I.; Fedyanin, A. A.; Kivshar, Y. S. Enhanced Third-Harmonic Generation in Silicon Nanoparticles Driven by Magnetic Response. Nano Lett. 2014, 14, 6488-6492.

33. Anker, J. N.; Hall, W. P.; Lyandres, O.; Shah, N. C.; Zhao, J.; Van Duyne, R. P. Biosensing with plasmonic nanosensors. Nat. Mater. 2008, 7, 442-53.   
34. Homola, J. Surface Plasmon Resonance Sensors for Detection of Chemical and Biological Species. Chem. Rev. 2008, 108, 462-493.   
35. Seok, T. J.; Jamshidi, A.; Kim, M.; Dhuey, S.; Lakhani, A.; Choo, H.; Schuck, P. J.; Cabrini, S.; Schwartzberg, A. M.; Bokor, J.; Yablonovitch, E.; Wu, M. C. Radiation engineering of optical antennas for maximum field enhancement. Nano Lett. 2011, 11, 2606-10.   
36. Baranov, D. G.; Zuev, D. A.; Lepeshov, S. I.; Kotov, O. V.; Krasnok, A. E.; Evlyukhin, A. B.; Chichkov, B. N. All-dielectric nanophotonics: the quest for better materials and fabrication techniques. Optica 2017, 4, 814-825.   
37. Multidisciplinary Vazquez-Guardado, A.; Chanda, D. Superchiral Light Generation on Degenerate Achiral Surfaces. Phys. Rev. Lett. 2018, 120, 137601.   
38. Yang, Y.; Wang, W.; Boulesbaa, A.; Kravchenko, II; Briggs, D. P.; Puretzky, A.; Geohegan, D.; Valentine, J. Nonlinear Fano-Resonant Dielectric Metasurfaces. Nano Lett. 2015, 15, 7388-93.   
39. Wu, P. C.; Liao, C. Y.; Savinov, V.; Chung, T. L.; Chen, W. T.; Huang, Y. W.; Wu, P. R.; Chen, Y. H.; Liu, A. Q.; Zheludev, N. I.; Tsai, D. P. Optical Anapole Metamaterial. ACS Nano 2018, 12, 1920-1927.   
40. Miroshnichenko, A. E.; Evlyukhin, A. B.; Yu, Y. F.; Bakker, R. M.; Chipouline, A.; Kuznetsov, A. I.; Luk/'yanchuk, B.; Chichkov, B. N.; Kivshar, Y. S. Nonradiating anapole modes in dielectric nanoparticles. Nat. Commun. 2015, 6, 8069.   
41. Semmlinger, M.; Zhang, M.; Tseng, M. L.; Huang, T. T.; Yang, J.; Tsai, D. P.; Nordlander, P.; Halas, N. J. Generating Third Harmonic Vacuum Ultraviolet Light with a TiO2 Metasurface. Nano Lett. 2019, 19, 8972-8978.   
42. Koshelev, K.; Lepeshov, S.; Liu, M.; Bogdanov, A.; Kivshar, Y. Asymmetric Metasurfaces with High-Q Resonances Governed by Bound States in the Continuum. Phys. Rev. Lett. 2018, 121, 193903.   
43. Koshelev, K.; Kivshar, Y. Light trapping gets a boost. Nature 2019, 574, 491-492.   
44. Hsu, C. W.; Zhen, B.; Stone, A. D.; Joannopoulos, J. D.; Soljačić, M. Bound states in the continuum. Nat. Rev. Mat. 2016, 1, 16048.   
45. Rybin, M. V.; Koshelev, K. L.; Sadrieva, Z. F.; Samusev, K. B.; Bogdanov, A. A.; Limonov, M. F.; Kivshar, Y. S. High-Q Supercavity Modes in Subwavelength Dielectric Resonators. Phys. Rev. Lett. 2017, 119, 243901.   
46. Lovera, A.; Gallinet, B.; Nordlander, P.; Martin, O. J. F. Mechanisms of Fano Resonances in Coupled Plasmonic Systems. ACS Nano 2013, 7, 4527-4536.   
47. Liu, Z.; Xu, Y.; Lin, Y.; Xiang, J.; Feng, T.; Cao, Q.; Li, J.; Lan, S.; Liu, J. High-Q Quasibound States in the Continuum for Nonlinear Metasurfaces. Phys. Rev. Lett. 2019, 123, 253901.   
48. Bontempi, N.; Chong, K. E.; Orton, H. W.; Staude, I.; Choi, D.-Y.; Alessandri, I.; Kivshar, Y. S.; Neshev, D. N. Highly sensitive biosensors based on all-dielectric nanoresonators. Nanoscale 2017, 9, 4972-4980.   
49. Yavas, O.; Svedendahl, M.; Dobosz, P.; Sanz, V.; Quidant, R. On-a-chip Biosensing Based on All-Dielectric Nanoresonators. Nano Lett. 2017, 17, 4421-4426.   
50. Yavas, O.; Svedendahl, M.; Quidant, R. Unravelling the Role of Electric and Magnetic Dipoles in Biosensing with Si Nanoresonators. ACS Nano 2019, 13, 4582-4588.   
51. Yang, Y.; Kravchenko, I. I.; Briggs, D. P.; Valentine, J. All-dielectric metasurface analogue of electromagnetically induced transparency. Nat. Commun. 2014, 5, 5753.   
52. Yesilkoy, F.; Arvelo, E. R.; Jahani, Y.; Liu, M.; Tittl, A.; Cevher, V.; Kivshar, Y.; Altug, H. Ultrasensitive hyperspectral imaging and biodetection enabled by dielectric metasurfaces. Nat. Photonics 2019, 13, 390-396.   
53. Triggs, G. J.; Wang, Y.; Reardon, C. P.; Fischer, M.; Evans, G. J. O.; Krauss, T. F. Chirped guided-mode resonance biosensor. Optica 2017, 4, 229-234.

54. Zong, C.; Xu, M.; Xu, L.-J.; Wei, T.; Ma, X.; Zheng, X.-S.; Hu, R.; Ren, B. Surface-Enhanced Raman Spectroscopy for Bioanalysis: Reliability and Challenges. Chem. Rev. 2018, 118, 4946-4980.

55. Dieringer, J. A.; Lettan, R. B.; Scheidt, K. A.; Van Duyne, R. P. A Frequency Domain Existence Proof of Single-Molecule Surface-Enhanced Raman Spectroscopy. J. Am. Chem. Soc. 2007, 129, 16249-16256.

56. Mao, P.; Liu, C.; Favraud, G.; Chen, Q.; Han, M.; Fratalocchi, A.; Zhang, S. Broadband single molecule SERS detection designed by warped optical spaces. Nat. Commun. 2018, 9, 5428.

57. Tseng, M. L.; Huang, Y.-W.; Hsiao, M.-K.; Huang, H. W.; Chen, H. M.; Chen, Y. L.; Chu, C. H.; Chu, N.-N.; He, Y. J.; Chang, C. M.; Lin, W. C.; Huang, D.-W.; Chiang, H.-P.; Liu, R.-S.; Sun, G.; Tsai, D. P. Fast Fabrication of a Ag Nanostructure Substrate Using the Femtosecond Laser for Broad-Band and Tunable Plasmonic Enhancement. ACS Nano 2012, 6, 5190-5197.

58. Jha, S. K.; Ahmed, Z.; Agio, M.; Ekinci, Y.; Löffler, J. F. Deep-UV Surface-Enhanced Resonance Raman Scattering of Adenine on Aluminum Nanoparticle Arrays. J. Am. Chem. Soc. 2012, 134, 1966-1969.

59. Cardinal, M. F.; Vander Ende, E.; Hackler, R. A.; McAnally, M. O.; Stair, P. C.; Schatz, G. C.; Van Duyne, R. P. Expanding applications of SERS through versatile nanomaterials engineering. Chem. Soc. Rev. 2017, 46, 3886-3903.

60. Zhang, J.; Cao, C.; Xu, X.; Liow, C.; Li, S.; Tan, P.; Xiong, Q. Tailoring Alphabetical Metamaterials in Optical Frequency: Plasmonic Coupling, Dispersion, and Sensing. ACS Nano 2014, 8, 3796-3806.

61. Xu, X.; Peng, B.; Li, D.; Zhang, J.; Wong, L. M.; Zhang, Q.; Wang, S.; Xiong, Q. Flexible Visible-Infrared Metamaterials and Their Applications in Highly Sensitive Chemical and Biological Sensing. Nano Lett. 2011, 11, 3232-3238.

62. Lee, Y.; Kim, S. J.; Park, H.; Lee, B. Metamaterials and Metasurfaces for Sensor Applications. Sensors (Basel) 2017, 17, 1726.

63. Mahmoudi, M.; Lohse, S. E.; Murphy, C. J.; Fathizadeh, A.; Montazeri, A.; Suslick, K. S. Variation of Protein Corona Composition of Gold Nanoparticles Following Plasmonic Heating. Nano Lett. 2014, 14, 6-12.

64. Alessandri, I.; Lombardi, J. R. Enhanced Raman Scattering with Dielectrics. Chem. Rev. 2016, 116, 14921-14981.

65. Ioffe, Z.; Shamai, T.; Ophir, A.; Noy, G.; Yutsis, I.; Kfir, K.; Cheshnovsky, O.; Selzer, Y. Detection of heating in current-carrying molecular junctions by Raman scattering. Nat. Nanotechnol. 2008, 3, 727-32.

66. Kuhlicke, A.; Schietinger, S.; Matyssek, C.; Busch, K.; Benson, O. In situ observation of plasmon tuning in a single gold nanoparticle during controlled melting. Nano Lett. 2013, 13, 2041-6.

67. Caldarola, M.; Albella, P.; Cortes, E.; Rahmani, M.; Roschuk, T.; Grinblat, G.; Oulton, R. F.; Bragas, A. V.; Maier, S. A. Non-plasmonic nanoantennas for surface enhanced spectroscopies with ultra-low heat conversion. Nat. Commun. 2015, 6, 7915.

68. Cambiasso, J.; König, M.; Cortés, E.; Schlücker, S.; Maier, S. A. Surface-Enhanced Spectroscopies of a Molecular Monolayer in an All-Dielectric Nanoantenna. ACS Photon. 2018, 5, 1546-1557.

69. Romano, S.; Zito, G.; Managò, S.; Calafiore, G.; Penzo, E.; Cabrini, S.; De Luca, A. C.; Mocella, V. Surface-Enhanced Raman and Fluorescence Spectroscopy with an All-Dielectric Metasurface. The J. Phys. Chem. C 2018, 122, 19738-19745.

70. Sapsford, K. E.; Algar, W. R.; Berti, L.; Gemmill, K. B.; Casey, B. J.; Oh, E.; Stewart, M. H.; Medintz, I. L. Functionalizing nanoparticles with biological molecules: developing chemistries that facilitate nanotechnology. Chem. Rev. 2013, 113, 1904-2074.

71. Tittl, A.; John-Herpin, A.; Leitis, A.; Arvelo, E. R.; Altug, H. Metasurface-Based Molecular Biosensing Aided by Artificial Intelligence. Angew. Chem. Int. Ed. Engl. 2019, 58, 14810-14822.

72. Wu, C.; Arju, N.; Kelp, G.; Fan, J. A.; Dominguez, J.; Gonzales, E.; Tutuc, E.; Brener, I.; Shvets, G. Spectrally selective chiral silicon metasurfaces based on infrared Fano resonances. Nat. Commun. 2014, 5, 3892.   
73. Tittl, A.; Leitis, A.; Liu, M.; Yesilkoy, F.; Choi, D.-Y.; Neshev, D. N.; Kivshar, Y. S.; Altug, H. Imaging-based molecular barcoding with pixelated dielectric metasurfaces. Science 2018, 360, 1105-1109.   
74. Leitis, A.; Tittl, A.; Liu, M.; Yesilkoy, F.; Choi, D.-Y.; Neshev, D. N.; Kivshar, Y. S.; Altug, H. In All-dielectric Metasurfaces for Infrared Absorption Spectroscopy Applications, Conference on Lasers and Electro-Optics, San Jose, California, 2019/05/05, 2019; Optical Society of America: San Jose, California, p SF3J.1.   
75. Leitis, A.; Tittl, A.; Liu, M.; Lee, B. H.; Gu, M. B.; Kivshar, Y. S.; Altug, H. Angle-multiplexed all-dielectric metasurfaces for broadband molecular fingerprint retrieval. Sci. Adv. 2019, 5, eaaw2871.   
76. Kumar, J.; Eraña, H.; López-Martínez, E.; Claes, N.; Martín, V. F.; Solís, D. M.; Bals, S.; Cortajarena, A. L.; Castilla, J.; Liz-Marzán, L. M. Detection of amyloid fibrils in Parkinson's disease using plasmonic chirality. Proc. Natl. Acad. Sci. U.S.A 2018, 115, 3225-3230.   
77. Li, G.; DeLaney, K.; Li, L. Molecular basis for chirality-regulated Aβ self-assembly and receptor recognition revealed by ion mobility-mass spectrometry. Nat. Commun. 2019, 10, 5038.   
78. Li, M.; Howson, S. E.; Dong, K.; Gao, N.; Ren, J.; Scott, P.; Qu, X. Chiral Metallohelical Complexes Enantioselectively Target Amyloid $\beta$ for Treating Alzheimer's Disease. J. Am. Chem. Soc. 2014, 136, 11655-11663.   
79. Mohammadi, E.; Tsakmakidis, K. L.; Askarpour, A. N.; Dehkhoda, P.; Tavakoli, A.; Altug, H. Nanophotonic Platforms for Enhanced Chiral Sensing. ACS Photon. 2018, 5, 2669-2675.   
80. Schäferling, M.; Dregely, D.; Hentschel, M.; Giessen, H. Tailoring Enhanced Optical Chirality: Design Principles for Chiral Plasmonic Nanostructures. Phys. Rev. X 2012, 2, 031010.   
81. Schäferling, M.; Yin, X.; Engheta, N.; Giessen, H. Helical Plasmonic Nanostructures as Prototypical Chiral Near-Field Sources. ACS Photon. 2014, 1, 530-537.   
82. Tseng, M. L.; Lin, Z. H.; Kuo, H. Y.; Huang, T. T.; Huang, Y. T.; Chung, T. L.; Chu, C. H.; Huang, J. S.; Tsai, D. P. Stress-Induced 3D Chiral Fractal Metasurface for Enhanced and Stabilized Broadband Near-Field Optical Chirality. Adv. Opt. Mater. 2019, 7, 1900617.   
83. Garcia-Guirado, J.; Svedendahl, M.; Puigdollers, J.; Quidant, R. Enhanced Chiral Sensing with Dielectric Nanoresonators. Nano Lett. 2020, 20, 585-591.   
84. Hu, J.; Lawrence, M.; Dionne, J. A. High Quality Factor Dielectric Metasurfaces for Ultraviolet Circular Dichroism Spectroscopy. ACS Photon. 2019, 7, 36.   
85. Mohammadi, E.; Tavakoli, A.; Dehkhoda, P.; Jahani, Y.; Tsakmakidis, K. L.; Tittl, A.; Altug, H. Accessible Superchiral Near-Fields Driven by Tailored Electric and Magnetic Resonances in All-Dielectric Nanostructures. ACS Photon. 2019, 6, 1939-1946.   
86. Koshelev, K.; Jahani, Y.; Tittl, A.; Altug, H.; Kivshar, Y. In Enhanced Circular Dichroism and Chiral Sensing with Bound States in the Continuum, Conference on Lasers and Electro-Optics, San Jose, California, 2019/05/05, 2019; Optical Society of America: San Jose, California, p FTh4C.6.   
87. Das Gupta, T.; Martin-Monier, L.; Yan, W.; Le Bris, A.; Nguyen-Dang, T.; Page, A. G.; Ho, K. T.; Yesilkoy, F.; Altug, H.; Qu, Y.; Sorin, F. Self-assembly of nanostructured glass metasurfaces via templated fluid instabilities. Nat. Nanotechnol. 2019, 14, 320-327.   
88. Chanda, D.; Shigeta, K.; Gupta, S.; Cain, T.; Carlson, A.; Mihi, A.; Baca, A. J.; Bogart, G. R.; Braun, P.; Rogers, J. A. Large-area flexible 3D optical negative index metamaterial formed by nanotransfer printing. Nat. Nanotechnol. 2011, 6, 402-7.   
89. Tseng, M. L.; Wu, P. C.; Sun, S.; Chang, C. M.; Chen, W. T.; Chu, C. H.; Chen, P. L.; Zhou, L.; Huang, D. W.; Yen, T. J.; Tsai, D. P. Fabrication of multilayer metamaterials by femtosecond laser-induced forward-transfer technique. Laser Photonics Rev. 2012, 6, 702-707.

90. Gansel, J. K.; Thiel, M.; Rill, M. S.; Decker, M.; Bade, K.; Saile, V.; von Freymann, G.; Linden, S.; Wegener, M. Gold Helix Photonic Metamaterial as Broadband Circular Polarizer. Science 2009, 325, 1513-1515.   
91. Cataldo, S.; Zhao, J.; Neubrech, F.; Frank, B.; Zhang, C.; Braun, P. V.; Giessen, H. Hole-Mask Colloidal Nanolithography for Large-Area Low-Cost Metamaterials and Antenna-Assisted Surface-Enhanced Infrared Absorption Substrates. ACS Nano 2012, 6, 979-985.   
92. Yuan, L.; Zhang, C.; Zhang, X.; Lou, M.; Ye, F.; Jacobson, C. R.; Dong, L.; Zhou, L.; Lou, M.; Cheng, Z.; Ajayan, P. M.; Nordlander, P.; Halas, N. J. Photocatalytic Hydrogenation of Graphene Using Pd Nanocones. Nano Lett. 2019, 19, 4413-4419.   
93. Zhang, C.; Zhao, H.; Zhou, L.; Schlather, A. E.; Dong, L.; McClain, M. J.; Swearer, D. F.; Nordlander, P.; Halas, N. J. Al-Pd Nanodisk Heterodimers as Antenna-Reactor Photocatalysts. Nano Lett. 2016, 16, 6677-6682.   
94. Sigle, D. O.; Perkins, E.; Baumberg, J. J.; Mahajan, S. Reproducible Deep-UV SERRS on Aluminum Nanovoids. J. Phys. Chem. Lett. 2013, 4, 1449-52.   
95. Meyring, M.; Mühlbacher, J.; Messer, K.; Kastner-Pustet, N.; Bringmann, G.; Mannschreck, A.; Blaschke, G. In Vitro Biotransformation of (R)- and (S)-Thalidomide: Application of Circular Dichroism Spectroscopy to the Stereochemical Characterization of the Hydroxylated Metabolites. Anal. Chem. 2002, 74, 3726-3735.   
96. Krishnamoorthy, H. N. S.; Adamo, G.; Yin, J.; Savinov, V.; Zheludev, N. I.; Soci, C. Infrared dielectric metamaterials from high refractive index chalcogenides. Nat. Commun. 2020, 11, 1692.   
97. Verre, R.; Baranov, D. G.; Munkhbat, B.; Cuadra, J.; Kall, M.; Shegai, T. Transition metal dichalcogenide nanodisks as high-index dielectric Mie nanoresonators. Nat. Nanotechnol. 2019, 14, 679-683.   
98. Wu, P. C.; Pala, R. A.; Kafaie Shirmanesh, G.; Cheng, W. H.; Sokhoyan, R.; Grajower, M.; Alam, M. Z.; Lee, D.; Atwater, H. A. Dynamic beam steering with all-dielectric electro-optic III-V multiple-quantum-well metasurfaces. Nat. Commun. 2019, 10, 3654.   
99. Shen, K.-C.; Huang, Y.-T.; Chung, T. L.; Tseng, M. L.; Tsai, W.-Y.; Sun, G.; Tsai, D. P. Giant Efficiency of Visible Second-Harmonic Light by an All-Dielectric Multiple-Quantum-Well Metasurface. Phys. Rev. Appl. 2019, 12, 064056.   
100. Leitis, A.; Heßler, A.; Wahl, S.; Wuttig, M.; Taubner, T.; Tittl, A.; Altug, H. All-Dielectric Programmable Huygens' Metasurfaces. Adv. Funct. Mater. 2020, 30, 1910259.   
101. Chu, C. H.; Tseng, M. L.; Chen, J.; Wu, P. C.; Chen, Y.-H.; Wang, H.-C.; Chen, T.-Y.; Hsieh, W. T.; Wu, H. J.; Sun, G.; Tsai, D. P. Active dielectric metasurface based on phase-change medium. Laser Photonics Rev. 2016, 10, 986-994.   
102. Gholipour, B.; Adamo, G.; Cortecchia, D.; Krishnamoorthy, H. N.; Birowosuto, M. D.; Zheludev, N. I.; Soci, C. Organometallic Perovskite Metasurfaces. Adv. Mater. 2017, 29, 1604268.   
103. Gao, Y.; Huang, C.; Hao, C.; Sun, S.; Zhang, L.; Zhang, C.; Duan, Z.; Wang, K.; Jin, Z.; Zhang, N.; Kildishev, A. V.; Qiu, C. W.; Song, Q.; Xiao, S. Lead Halide Perovskite Nanostructures for Dynamic Color Display. ACS Nano 2018, 12, 8847-8854.   
104. Soler, M.; Estevez, M. C.; Cardenosa-Rubio, M.; Astua, A.; Lechuga, L. M. How Nanophotonic Label-Free Biosensors Can Contribute to Rapid and Massive Diagnostics of Respiratory Virus Infections: COVID-19 Case. ACS Sens. 2020.   
105. Yang, T.; Wang, Y. C.; Shen, C. F.; Cheng, C. M. Point-of-Care RNA-Based Diagnostic Device for COVID-19. Diagnostics (Basel) 2020, 10.   
106. Hong, L.; Li, H.; Yang, H.; Sengupta, K. Fully Integrated Fluorescence Biosensors On-Chip Employing Multi-Functional Nanoplasmonic Optical Structures in CMOS. IEEE Journal of Solid-State Circuits 2017, 52, 2388-2406.   
107. Hong, L.; Li, H.; Yang, H.; Sengupta, K. Integrated Angle-Insensitive Nanoplasmonic Filters for Ultraminiaturized Fluorescence Microarray in a 65 nm Digital CMOS Process. ACS Photon. 2018, 5, 4312-4322.

108. Chen, C.; Li, Y.; Kerman, S.; Neutens, P.; Willems, K.; Cornelissen, S.; Lagae, L.; Stakenborg, T.; Van Dorpe, P. High spatial resolution nanoslit SERS for single-molecule nucleobase sensing. Nat. Commun. 2018, 9, 1733.   
109. Belushkin, A.; Yesilkoy, F.; Gonzalez-Lopez, J. J.; Ruiz-Rodriguez, J. C.; Ferrer, R.; Fabrega, A.; Altug, H. Rapid and Digital Detection of Inflammatory Biomarkers Enabled by a Novel Portable Nanoplasmonic Imager. Small 2020, 16, e1906108.   
110. Tseng, M. L.; Yang, J.; Semmlinger, M.; Zhang, C.; Nordlander, P.; Halas, N. J. Two-Dimensional Active Tuning of an Aluminum Plasmonic Array for Full-Spectrum Response. Nano Lett. 2017, 17, 6034-6039.   
111. Zheludev, N. I.; Kivshar, Y. S. From metamaterials to metadevices. Nat. Mater. 2012, 11, 917-924.   
112. Malkiel, I.; Mrejen, M.; Nagler, A.; Arieli, U.; Wolf, L.; Suchowski, H. Plasmonic nanostructure design and characterization via Deep Learning. Light Sci. Appl. 2018, 7, 60.   
113. Liu, Z.; Zhu, D.; Rodrigues, S. P.; Lee, K. T.; Cai, W. Generative Model for the Inverse Design of Metasurfaces. Nano Lett. 2018, 18, 6570-6576.