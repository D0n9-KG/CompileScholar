# Review

# Sensitive optical biosensors for unlabeled targets: A review

Xudong Fan*, Ian M. White, Siyka I. Shopova, Hongying Zhu, Jonathan D. Suter, Yuze Sun

Department of Biological Engineering, University of Missouri, 240D, Bond Life Sciences Center, 1201 E. Rollins Street, Columbia, Missouri 65211, USA

# ARTICLE INFO

Article history:

Received 2 March 2008

Received in revised form

10 May 2008

Accepted 12 May 2008

Published on line 18 May 2008

Keywords:

Optical biosensor

Photonic biosensor

Label-free

Refractive index

# ABSTRACT

This article reviews the recent progress in optical biosensors that use the label-free detection protocol, in which biomolecules are unlabeled or unmodified, and are detected in their natural forms. In particular, it will focus on the optical biosensors that utilize the refractive index change as the sensing transduction signal. Various optical label-free biosensing platforms will be introduced, including, but not limited to, surface plasmon resonance, interferometers, waveguides, fiber gratings, ring resonators, and photonic crystals.

Emphasis will be given to the description of optical structures and their respective sensing mechanisms. Examples of detecting various types of biomolecules will be presented. Wherever possible, the sensing performance of each optical structure will be evaluated and compared in terms of sensitivity and detection limit.

© 2008 Elsevier B.V. All rights reserved.

# Contents

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

ELSEVIER

available at www.sciencedirect.com

ScienceDirect

journal homepage: www.elsevier.com/locate/aca

ANALYTICA CHMCA ACTA

* Corresponding author. Tel.: +1 573 884 2543; fax: +1 573 884 9676.

E-mail address: fanxud@missouri.edu (X. Fan).

0003-2670/$ – see front matter © 2008 Elsevier B.V. All rights reserved.

doi:10.1016/j.aca.2008.05.022

2.5. Optical fiber based biosensors 12

2.6. Photonic crystal based sensors 14

3.Outlook 15

Acknowledgments 16

References 16

# 1. Introduction

Optical biosensors are a powerful detection and analysis tool that has vast applications in biomedical research, healthcare, pharmaceuticals, environmental monitoring, homeland security, and the battlefield [1]. They are immune to electromagnetic interference, capable of performing remote sensing, and can provide multiplexed detection within a single device. Generally, there are two detection protocols that can be implemented in optical biosensing: fluorescence-based detection and label-free detection.

In fluorescence-based detection, either target molecules or biorecognition molecules are labeled with fluorescent tags, such as dyes; the intensity of the fluorescence indicates the presence of the target molecules and the interaction strength between target and biorecognition molecules. While fluorescence-based detection is extremely sensitive, with the detection limit down to a single molecule [2], it suffers from laborious labeling processes that may also interfere with the function of a biomolecule.

Quantitative analysis is challenging due to the fluorescence signal bias, as the number of fluorophores on each molecule cannot be precisely controlled [3]. In contrast, in label-free detection, target molecules are not labeled or altered, and are detected in their natural forms. This type of detection is relatively easy and cheap to perform, and allows for quantitative and kinetic measurement of molecular interaction.

Additionally, as discussed later, some label-free detection mechanisms measure refractive index (RI) change induced by molecular interactions, which is related to the sample concentration or surface density, instead of total sample mass. As a result, the detection signal does not scale down with the sample volume.

This characteristic is particularly attractive when ultrasmall (femtoliter to nanoliter) detection volume is involved and is advantageous over fluorescence-based detection whose signal usually depends on the total number of analytes in the detection volume or on the detection surface.

Despite all these differences between fluorescence-based and label-free detection, both protocols are being widely used in optical sensors and provide vital and complementary information regarding interactions among biomolecules, which makes optical sensors more versatile than other types of sensing technologies, such as surface acoustic wave and quartz crystal microbalance where only label-free detection can be implemented.

In this article, we will discuss label-free optical detection. Under this category, there exist a number of detection methods, including RI detection, optical absorption detection, and Raman spectroscopic detection. RI and absorption are the real and imaginary part of the more general complex RI constant and are related to each other via the Kronig-Kramers relations [4]. Raman detection is unique in that, on one hand, like RI and

absorption detection, target molecules are not labeled; on the other hand, emitted Raman light is used for sensing, similar to fluorescence-based detection. Here, we will further limit the scope of our article to RI-based label-free detection. Optical absorption and Raman-based biosensing can be found elsewhere [5,6]. Note that some of the RI-based sensors can detect both the RI of bulk solution (refractometer) and the RI change induced by molecule binding (biosensor) whereas others can be used only for bulk solution RI detection. We will only discuss those that have been demonstrated for biosensing or have potential capability of biosensing.

For biosensor development, a number of aspects need to be considered, such as transduction signal generation (increase of signal, decrease of noise, etc.), fluidics design (sample injection and drainage, reduction of sample consumption, increase of analyte transport, reduction in detection time, etc.), surface immobilization chemistry (analyte capture efficiency, elimination of non-specific binding, etc.), detection format (direct binding, sandwich-type binding, competitive binding, etc.), and data analysis (extraction of information regarding analyte concentration, binding kinetics, etc.).

In this article, we will focus mainly on the transduction part with the emphasis on optical structures and their interactions with the analyte. Fig. 1 shows the generic configuration of a RI-based optical label-free biosensor (label-free sensor, for short). Biorecognition molecules, such as antibodies, oligonucleotides, aptamers, or phages, are immobilized on the sensor surface. Initially, the sensor chamber is filled with the buffer solution.

When target analytes bind to the biorecognition molecules, they replace buffer solution molecules within a few nanometers to a few thousands of nanometers from the surface. The target analytes have different RI than that of the buffer solution (for example, the RI for protein is 1.5 vs. 1.33 for buffer solution), resulting in a RI change near the sensor surface, which can be detected optically as the sensing transduction signal. In most label-free optical sensors, the sensing light is con

![](dt=2026-05-11/ht=18/2e609fddd179674ef16dd2f54715decf3b1aacf3e427bad0a24db30fb3aadfe7.jpg)

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

9

centrated near the sensor surface with an evanescent field exponentially decaying into the bulk solution with a characteristic decay length of a few tens to a few hundreds of nanometers. Therefore, it is capable of detecting the RI change induced by the analyte binding that is within the decay length. However, for those analytes that are not captured by the biorecognition molecules, and therefore, are far away from the sensing surface, the sensing signal decreases significantly. As a consequence, this evanescent sensing provides a means to distinguish actual target molecules from those that do not bind to the sensing surface but may still be present in
bulk solution.

In sensor development, sensitivity is an important parameter to evaluate the sensor performance. Sensitivity is the magnitude of sensor transduction signal change in response to the change in analyte. Fundamentally, sensitivity is determined by the strength of light-matter interaction [7-9]. In most evanescent wave based sensors, it is determined by the fraction of light in solution or the light intensity at the sensor surface. Sensor detection limit (DL) is another important parameter to characterize the sensor performance.

The DL can be deduced by taking into account the noise in the transduction signal, $\sigma$ , i.e., the minimum resolvable signal: $\mathrm{DL} = \sigma / S$ , where $S$ is the sensitivity. For an optical RI-based label-free sensor, typically there are three ways to specify the DL. First, since all these sensors are sensitive to the RI change in bulk solution, DL in units of refractive index units (RIU) is naturally used to quantify the sensor performance, which enables a rough comparison of the sensing capability among different optical technologies and structures.

The second way is to use surface mass density (or total mass) in units of pg mm $^{-2}$ (or pg), which is what a biosensor actually measures. While experimentally surface mass density is difficult to determine accurately, it reflects the intrinsic detection capability of a sensor and can be used to evaluate or compare the sensor performance. The third way is to use sample concentration (in units of ng mL $^{-1}$ or molarity).

Sample concentration is quite useful and easy to determine from an experimental point of view, as no detailed information regarding the mass density on the surface is needed. However, the DL defined in this manner depends on the target molecule and its affinity to the biorecognition molecule on the surface, and therefore needs to be specified for each type of a biomolecule. These three DLs are correlated and the detailed relationship among them needs to be studied for each individual optical biosensor.

Improvement in the DL can be accomplished by increasing the sensitivity or reducing the noise level. Sensitivity can be enhanced by increasing the light-matter interaction. In most cases, this amounts to increasing the fraction of light near the sensing surface (examples of which will be shown later). For a label-free optical biosensor, the most commonly seen noise is from temperature fluctuations, which results in a thermo-optic effect (i.e., temperature-dependent RI changes) and a thermo-mechanic effect (e.g., thermal expansion) in both sensor substrate and buffer solution.

A general approach to reduce the thermally induced noise is to implement a temperature control, such as a thermoelectric cooler to stabilize temperature. Another method is to balance the thermo-optic and thermo-mechanic effects. For example, most solvents have a negative thermo-optic coefficient, which can be explored

to counteract the thermo-optic and thermo-mechanic effects of the solid substrate that have a positive coefficient. With an appropriate design, the temperature-induced noise can be significantly reduced [10-12]. A third method is to employ a reference channel, which is either built into the same sensor or placed on a different sensor nearby, to reduce so-called common-mode noise, such as temperature related noise, bulk RI change, and non-specific binding. Through these methods, the sensor performance can be improved significantly.

In the past decade, optical sensor development has been a fascinating and fast-paced area. Various optical structures have been investigated for sensitive label-free detection. This article will review the recent advancement in optical label-free biosensors that are based on RI detection.

We will select the following optical structures that we believe represent the majority of research activities in optical label-free sensor development: (1) surface plasmon resonance based biosensors; (2) interferometer-based biosensors; (3) optical waveguide based biosensors; (4) optical ring resonator based biosensors; (5) optical fiber based biosensors; and (6) photonic crystal based biosensors. Note that it is common that an optical label-free sensor involves two or more optical structures mentioned above to enhance its sensing performance.

The above selection and categorization are solely the authors' choice. Emphasis will be given to the description of these optical structures and their respective sensing mechanisms. In particular, we will focus on the work done after 2000. Examples of detecting various types of biomolecules will be presented. Wherever possible, the sensing performance of each optical structure will be evaluated and compared in terms of sensitivity and DL (see Table 1).

# 2. Optical label-free biosensor structures

# 2.1. Surface plasmon resonance based biosensors

Surface plasmon resonance (SPR) biosensor was first demonstrated for biosensing in 1983 by Liedberg et al. [13]. Since then it has been extensively explored and has gradually become a very powerful label-free tool to study the interactions between the target and biorecognition molecules. The principle, development, and applications of SPR biosensors have been well described in several excellent review papers [14-17].

A surface plasmon wave (SPW) is a charge density oscillation that occurs at the interface of two media with dielectric constants of opposite signs, such as a metal (gold or silver) and a dielectric. There are four basic methods to excite the SPR, as shown in Fig. 2: prism coupling [18], waveguide coupling [19], fiber optic coupling [20], and grating coupling [21,22]. In the prism coupling configuration (Fig. 2(A)), the incident light is totally reflected at the prism-metal interface and generates an evanescent field penetrating into the metal layer. At the resonant angle or resonant wavelength, the propagation constant of the evanescent field matches that of the SPW as described in Eq. (1), and as a result, the photon will be coupled into the SPW.

$$
\frac {2 \pi}{\lambda} n _ {\mathrm {p}} \sin \theta = \beta_ {\mathrm {s p}}, \tag {1}
$$

10

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

![](dt=2026-05-11/ht=18/0f5fe38fcd1b31c7cdb734f4f4e9326797e554beb3f007a25bcdfd7e65237836.jpg)

<table><tr><td colspan="5">Table 1 - Detection limits of various optical biosensors</td></tr><tr><td>Technology platform</td><td>Optical structures</td><td>Analyte</td><td>Detection limit</td><td>Ref.</td></tr><tr><td rowspan="15">Surface plasmon resonance</td><td>Surface plasmon resonance</td><td>Bulk solution</td><td>10-5-10-8RIU</td><td>[36-41]</td></tr><tr><td>Long range SPR</td><td>Bulk solution</td><td>10-7-10-8RIU</td><td>[43,45]</td></tr><tr><td>Surface plasmon resonance imaging (SPRI)</td><td>Bulk solution</td><td>10-5-10-7RIU</td><td>[53-56]</td></tr><tr><td>SPRI</td><td>Protein</td><td>1 nM</td><td>[58]</td></tr><tr><td>Optical heterodyne SPR</td><td>Protein</td><td>0.2 nM</td><td>[59]</td></tr><tr><td>Phase sensitive SPR</td><td>Protein</td><td>1.3 nM</td><td>[60]</td></tr><tr><td>Wavelength modulated SPR</td><td>DNA</td><td>10 pM (Note 1)</td><td>[63]</td></tr><tr><td>SPRI</td><td>DNA and RNA</td><td>10 nM</td><td>[64]</td></tr><tr><td>Flow injection SPR</td><td>DNA</td><td>54 fM, 1.38 fM (Note 1)</td><td>[65,66]</td></tr><tr><td>Angle modulated SPR</td><td>Protein (PSA)</td><td>0.15 ng mL-1 (Note 1)</td><td>[67]</td></tr><tr><td>SPRI</td><td>Protein (CA19-9)</td><td>66.7 unit mL-1</td><td>[69]</td></tr><tr><td>SPRI</td><td>Protein (α-fetoprotein)</td><td>50 ng mL-1</td><td>[70]</td></tr><tr><td rowspan="2">Prism-based SPR</td><td>Bacteria (E. coli)</td><td>106cfu mL-1</td><td>[74]</td></tr><tr><td>Salmonella yphimurium</td><td>100cfu mL-1</td><td>[75]</td></tr><tr><td>BICore 2000 SPR</td><td>Bacteria</td><td>25 cfu mL-1</td><td>[76]</td></tr><tr><td rowspan="1
4">Interferometer</td><td rowspan="2">Mach-Zehnder interferometer</td><td>Bulk solution</td><td>10-7RIU</td><td>[83]</td></tr><tr><td>Protein</td><td>20 pg mm-2</td><td>[82]</td></tr><tr><td rowspan="2">Young's interferometer</td><td>Bulk solution</td><td>~1 × 10-7RIU</td><td>[89,90]</td></tr><tr><td>Virus</td><td>1000 particles mL-1</td><td>[91]</td></tr><tr><td rowspan="4">Hartman interferometer</td><td>DNA</td><td>4 ng mL-1</td><td>[95]</td></tr><tr><td>Protein</td><td>5 ng mL-1, 0.1 ng mL-1 (Note 1)</td><td>[95,96]</td></tr><tr><td>Virus</td><td>107pfu mL-1</td><td>[95]</td></tr><tr><td>Bacteria</td><td>5 × 108cfu mL-1, 5 × 105cfu mL-1 (Note 1)</td><td>[95]</td></tr><tr><td>Backscattering interferometer</td><td></td><td></td><td></td></tr><tr><td>Porous silicon</td><td>DNA</td><td>2 pM</td><td>[98]</td></tr><tr><td>BioCD</td><td>Protein</td><td>0.1 ng mL-1</td><td>[100]</td></tr><tr><td rowspan="3">Microchannel backscattering</td><td>Bulk solution</td><td>7 × 10-9RIU</td><td>[105]</td></tr><tr><td>Protein</td><td>femto-mole (Note 2)</td><td>[103]</td></tr><tr><td>Protein</td><td>picomolar (Note 3)</td><td>[106]</td></tr><tr><td rowspan="6">Waveguide</td><td rowspan="2">Resonant mirror</td><td>Protein</td><td>~0.1 pg mm-2</td><td>[113]</td></tr><tr><td>Cell</td><td>106cells mL-1</td><td>[108]</td></tr><tr><td rowspan="2">Metal-clad waveguide</td><td>Bacterial spore</td><td>~105spores mL-1</td><td>[119]</td></tr><tr><td>Cell</td><td>~10 cells mm-2 (Note 2)</td><td>[121]</td></tr><tr><td>Reverse symmetry waveguide</td><td>Cell</td><td>60 cells mm-2 (Note 2)</td><td>[123]</td></tr><tr><td>Symmetrical metal-clad waveguide</td><td>Bulk solution</td><td>2 × 10-7RIU</td><td>[129]</td></tr><tr><td rowspan="9">Ring resonator</td><td rowspan="4">Ring on a chip</td><td>Bulk solution</td><td>10-4to 10-7RIU</td><td>[168,171,185,189]</td></tr><tr><td>DNA</td><td>~100 nM</td><td>[186]</td></tr><tr><td>Protein</td><td>20-250 pg mm-2, 0.1 nM</td><td>[168,173,190]</td></tr><tr><td>Bacteria</td><td>105cfu mL-1</td><td>[186]</td></tr><tr><td rowspan="5">Dielectric microsphere</td><td>Bulk solution</td><td>10-7RIU</td><td>[144]</td></tr><tr><td>DNA</td><td>1 pg mm-2</td><td>[134]</td></tr><tr><td>Protein</td><td>10 pg mL-1 (trypsin), 1 unit mL-1 (thrombin)</td><td>[145,151]</td></tr><tr><td>Virus</td><td>~1 pg mm-2</td><td>[143]</td></tr><tr><td>Bacteria</td><td>100cfu mm-2 (Note 2)</td><td>[140]</td></tr><tr><td colspan="5">Table 1 (Continued)</td></tr><tr><td>Technology platform</td><td>Optical structures</td><td>Analyte</td><td>Detection limit</td><td>Ref.</td></tr><tr><td></td><td>Capillary opto-fluidic ring</td><td>Bulk solution</td><td>10-6-10-7RIU</td><td>[152]</td></tr><tr><td></td><td></td><td>DNA</td><td>4 pgmm-2, 10 pM</td><td>[163]</td></tr><tr><td></td><td></td><td>Protein</td><td>~1 pgmm-2, 3 pM</td><td>[157]</td></tr><tr><td></td><td></td><td></td><td>(Note 2)</td><td></td></tr><tr><td></td><td></td><td>Virus</td><td>1000 particles mL-1</td><td>[164]</td></tr><tr><td>Optical fiber</td><td>Fiber Bragg grating</td><td>Bulk solution</td><td>~10-6RIU</td><td>[204-209]</td></tr><tr><td></td><td></td><td>DNA</td><td>0.7 μg mL-1, 0.1 μM</td><td>[209]</td></tr><tr><td></td><td>Long-period grating</td><td>Bulk solution</td><td>10-4RIU</td><td>[219]</td></tr><tr><td></td><td></td><td>Protein</td><td>2 μg mL-1</td><td>[218]</td></tr><tr><td></td><td>Nanofiber</td><td>Bulk solution</td><td>10-7RIU</td><td>[223]</td></tr><tr><td></td><td>Fiber coupler</td><td>Bulk solution</td><td>4 × 10-6RIU</td><td>[227]</td></tr><tr><td></td><td></td><td>Protein</td><td>0.5 μg mL-1</td><td>[227]</td></tr><tr><td></td><td>Fiber Fabry-Perot cavity</td><td>Bulk solution</td><td>10-5RIU</td><td>[207,231]</td></tr><tr><td></td><td></td><td>DNA</td><td>76 μM, 1.7 ng</td><td>[229]</td></tr><tr><td></td><td></td><td>Protein</td><td>25 μg mL-1</td><td>[228]</td></tr><tr><td>Photonic crystal</td><td>2-D PC</td><td>Bulk solution</td><td>10-5RIU</td><td>[240,241]</td></tr><tr><td></td><td></td><td>Protein</td><td>0.4 pgmm-2 (Notes 2 and 4)</td><td>[240,241]</td></tr><tr><td></td><td>2-D PC microcavity</td><td>Bulk solution</td><td>~10-3RIU</td><td>[234,235]</td></tr><tr><td></td><td></td><td>Protein</td><td>1 fg</td><td>[237]</td></tr><tr><td></td><td>PC waveguide</td><td>Protein</td><td>0.15 μM</td><td>[239]</td></tr><tr><td></td><td>1-D PC microcavity array</td><td>Bulk solution</td><td>7 × 10-5RIU</td><td>[246]</td></tr><tr><td></td><td>PCF</td><td>Bulk solution</td><td>10-4RIU</td><td>[219]</td></tr></table>

Notes: (1) With gold nanoparticle amplification. (2) Adsorption result. (3) Binding takes place in free solution rather than on a solid substrate. (4) Results obtained after solution containing protein is dried.

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

11

![](image)

where $\lambda$ is the incident wavelength, $n_{\mathfrak{p}}$ the prism refractive index, $\theta$ the incident angel and $\beta_{\mathrm{sp}}$ is the propagation constant of the SPW. Prism coupling is the most convenient SPR configuration and generally has the best sensing DL; however, the prism is bulky and it is difficult to integrate. Waveguide coupling offers a good alternative to the prism; it is robust and easy to integrate with other optical and electrical components. The light propagates in a waveguide through total internal reflection and generates an evanescent field at the

waveguide-metal interface, which excites the SPW in the same way as in the prism configuration (Fig. 2(B)). Optical fibers are also used to excite the SPW in an easy and flexible way. As shown in Fig. 2(C), a small portion of the fiber is removed and then coated with a layer of metal film. Several different fiber optic SPR biosensor configurations have been developed based on side-polished single-mode fiber (Fig. 2(D)) [23,24], side-polished multimode fiber [25], polarization maintaining fiber [26], and D-shaped fiber [27]. Additionally, the SPW can

![](dt=2026-05-11/ht=18/c388c67afdda5d9113987735e7d3e1c708955a8d3999b503b33fde4c2edc53c2.jpg)

![](dt=2026-05-11/ht=18/e0983d7b14f5d6e6ec2b5d9e69c91f208ef584165063f3df49eb1b4659b221a0.jpg)

![](dt=2026-05-11/ht=18/63fe3fa21dd4d17eb30ac888153c4c1bf613d399c2ba3852349f8ac4a9e3292b.jpg)

![](dt=2026-05-11/ht=18/eeb488526417a15a089abc1727228f2a5470e1c8b923abc1e7c6e018fedff416.jpg)

![](dt=2026-05-11/ht=18/8fb7253fbc63cffc9da2040599fd8c49e327ed577e40ae109099a2a020d1c8cf.jpg)

![](dt=2026-05-11/ht=18/1316262c581f8ef641f6432be2f7fd07a548a51fc68d2fbf336337d9385d3d6b.jpg)

12

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

be excited by grating coupling (Fig. 2(E)), which can be mass produced with low lost. In this way, a light wave is incident on a metallic grating. If the momentum of diffracted light parallel to the grating surface is the same as the propagation constant of the surface plasmon, the light will be coupled into the SPW as described in Eq. (2):

$$
\frac {2 \pi}{\lambda} n _ {\mathrm {d}} \sin \theta + m \frac {2 \pi}{\Lambda} = \pm \beta_ {\mathrm {s p}}, \tag {2}
$$

where $n_{\mathrm{d}}$ is the dielectric medium RI, $m$ an integer representing the diffraction order and $\Lambda$ is the grating period. However, with the grating coupling, the incident light directly illuminates the metal surface, therefore it requires the sample fluid and microfluidics to be optically transparent. The detection of the SPR change induced by the binding of target molecules to the sensing surface can be accomplished by monitoring
the resonant angle [28], resonant wavelength [29] or resonant intensity change [30]. Recently, several groups also explored the detection of the changes in phase [31-33] or polarization [34].

The commercialized SPR sensors, as exemplified by the instrument from Biacore [35], are typically based on prism coupling and have a DL between $1 \times 10^{-6}$ to $1 \times 10^{-7}$ RIU and a mass surface density DL around $1\mathrm{pgmm}^{-2}$ . An order of magnitude improvement in DL, i.e., $10^{-8}$ RIU, was achieved using a bicell photodetector instead of an array of photodetectors that is commonly employed in an SPR sensor [36]. For optical fiber coupled SPR biosensor, typically the DL is around $10^{-5}$ to $10^{-6}$ RIU [27,37].

A better DL of $5 - 7 \times 10^{-7}$ RIU has been achieved by multiple peak SPR fiber optic sensors [38] and by spectral interrogation in a side-polished single-mode fiber using depolarized light [24]. In waveguide-based SPR biosensors, the DL is usually from $10^{-5}$ to $10^{-6}$ RIU [29,39]. For the grating coupling configuration, the DL ranges from $10^{-5}$ to $10^{-6}$ RIU [40,41]. Besides the optical structures, detection methods also influence the detection limit. Phase detection has the best DL over other detection methods and it is typically about $10^{-7}$ RIU [33,42].

Although these basic SPR structures have good DL that can satisfy most research requirements, there still exist two potential problems that will limit their applications in some fields. First, the evanescent field in those basic SPR structures only penetrates into the surrounding medium for about $100\mathrm{nm}$ , and thus it is very difficult to detect the large target molecules like cells and bacteria. Second, there is only one SPW to detect the RI change. It is impossible to differentiate the surface RI change and the bulk solution RI change.

As a result, the sensing performance is deteriorated when detecting the target molecules in a complex solution, such as blood samples. To overcome those two problems, a new optical structure has been developed, as illustrated in Fig. 2(F), where the metal layer is sandwiched by two dielectric layers with a similar RI [43-47]. Two new surface plasmon modes called long range surface plasmon (LRSP) and short range surface plasmon (SRSP) form, which are bound to both metal-dielectric interfaces.

Using this dual mode of the LRSP and SRSP, the sensor is able to differentiate the background RI change and surface bound RI change [46]. Furthermore, the LRSP has a longer penetration length in the surrounding medium, mak

ing it suitable for cell and bacterium detection. The LRSP also exhibits an excellent DL as low as $2 \times 10^{-7}$ to $2 \times 10^{-8}$ RIU [43,45].

There are many ultra sensitive SPR biosensors but most of them only can detect one analyte. For practical applications, there is an urgent need for the development of SPR biosensors capable of multi-analyte detection. SPR sensors based on multi-channel structures [48,49] could provide a method for detecting multiple analytes simultaneously, but they are not suitable for high throughput detection. The SPR imaging (SPRI) technique is so far the most promising tool for high throughput detection with a DL of approximately $10^{-5}$ RIU [50-53].

The typical SPRI sensor is based on prism coupling, in which monochromatic incident light is expanded, passes through a prism and strikes the interface of the thin film and a prism at the coupling angle, exciting a broad area of the sensing surface. The reflected light is imaged on a CCD camera. Similar to the micro-array technique, in SPRI the sensing chip is divided into many small sensing spots for detecting different analytes.

A few other measurement methodologies were also developed for SPRI, such as dual wavelength SPRI system [54] and polarization contrast imaging [55] with DLs of $2 \times 10^{-6}$ RIU, respectively. Recently, $5 \times 10^{-7}$ RIU was achieved with a compact SPRI system [56].

In the past few years, SPR and SPRI have been applied in molecular biology for the detection of proteins with a DL ranging from picomolar to nanomolar [56-61], and detection of DNA molecules [62-65] with the DL as low as 54fM with a flow injection device [65] and as low as 1.38fM when aided by signal amplification through gold nanoparticles [66]. SPR also provides sensitive and fast detection of cancer biomarkers [67-70] and other biomarkers [71-73] with the DL at $\mathrm{ngmL}^{-1}$ level for medical diagnosis. Additionally, SPR biosensors have been used for bacterial detection for environmental monitoring, food safety and homeland security with a DL ranging from 25 to $10^{6}$ cfu mL $^{-1}$ [74-76].

# 2.2. Interferometer-based biosensors

# 2.2.1. Mach-Zehnder interferometer

The operation principle of the integrated Mach-Zehnder interferometer (MZI) is illustrated in Fig. 3(A). Coherent, single frequency, single polarization light from a laser enters the single-mode input waveguide and is split equally at a Y-junction. One branch has a window over the top of it allowing the evanescent field of that branch to interact with the sample while the reference arm is protected from the sample with a thick cladding layer. The two branches recombine at the output, resulting in interference, and a photodetector measures the intensity.

Generally, the waveguide structure must be single polarization and single mode so that multimodal and cross-polarization interference do not appear at the output. A change in the RI at the surface of the sensor arm results in an optical phase change on the sensing arm and a subsequent change in the light intensity measured at the photodetector, as described by: $I(\Delta n_{\mathrm{eff}}) \propto \cos (\Delta n_{\mathrm{eff}} k_0 L)$ , where $\Delta n_{\mathrm{eff}}$ , $k_0$ , and $L$ are the RI change, the amplitude of wave vector, and the length of the sensing region, respectively.

Typically, increasing the length of the sensing window increases the sensing signal. However, note that because of the cosine-dependent intensity

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

13

![](dt=2026-05-11/ht=18/75c462d8757e40346f01092b681b86f3500135930a30fa63ff785258347272ad.jpg)

![](dt=2026-05-11/ht=18/f523e8e3a7bd0d500e4ceebe7fd20a9f225d6c172414f6425f0aa9857bec0fda.jpg)

![](dt=2026-05-11/ht=18/a49c521f11de3158d375dacc9527b96f281d503ca50b27529124f85aa1597c96.jpg)

function, the signal change is not easily resolvable near the maximum and minimum of the cosine function. This is one penalty of using the MZI device as a sensor as compared to sensors that have a linear intensity response.

The first biosensing demonstration using an integrated MZI was performed by Heideman et al. [77]. The authors created a MZI on a silicon substrate with a $\mathrm{Si}_3\mathrm{N}_4$ waveguide and etched gratings for input/output coupling. The RI DL was experimentally determined to be around $5\times 10^{-6}$ RIU. An antibody for human chorionic gonadotropin (hCG) was adsorbed onto one channel, and hCG was then specifically detected. The experimental DL was $50\mathrm{pM}$ hCG. Nearly at the same time, Ingenhoff et al., used in integrated MZI for the detection of monolayer formations of adsorbed biomolecules [78], and shortly thereafter, the same group made incremental improvements by optimizing the waveguide materials [79].

Waveguide technologies improved in the years that followed these pioneering demonstrations. Brosinger et al. [80] and Schipper et al. [81] utilized a SiON or $\mathrm{Si}_3\mathrm{N}_4$ waveguide with a ribbed structure on a silicon substrate. The measured RI DL was around $10^{-5}$ RIU, and the authors demonstrated sensitive detection of an IgG molec
ular monolayer [80]. Additionally, this work demonstrated that in principle, the reference arm of the MZI can be used to cancel out common mode noise terms, such as temperature modulations, RI changes, and non-specific adsorption.

A fluidic system was constructed to allow different surface treatments on each arm so that the sensing arm was prepared for specific detection while the reference arm was prepared to cancel nonspecific binding. However, it should be noted that because the surfaces of the sensing and reference arms are biologically different, it is expected that the non-specific adsorption onto each surface will be different. A follow-up to this work was later published by Weisser et al.

[82], in which the authors claim detection of $1\%$ of a streptavidin monolayer on the sensing surface using the same MZI design and a biotin biorecognition layer. This translates to a protein DL of $20\mathrm{pgmm}^{-2}$ .

A number of engineering enhancements for integrated MZI sensors were incorporated in the work by Heideman and Lambeck [83]. As the authors demonstrated, applying a properly designed constant ac signal to the phase modulator reveals the quadrature points of the MZI's sinusoidal output intensity response through a simple demodulation circuit. Recall that one drawback of the MZI structure is that the sinusoidal-based response results in a sensitivity loss at the maxima and minima of the cosine term. However, the quadrature points

move continuously linearly during a sensing event, and thus tracking the quadrature points instead of the output intensity eliminates the problem of decreasing sensitivity at the maximum and minimum of the MZI sinusoidal response. With this and other engineering improvements, the authors showed a RI DL of $10^{-7}$ RIU. Despite the excellent bulk RI detection capability, to date, no actual biosensing has been demonstrated yet.

Another important design improvement is to enable larger waveguide core dimensions without sacrificing the single-mode behavior or the sensitivity. The previously discussed waveguide core dimensions are on the order of $100\mathrm{nm}$ , which causes high loss at the input coupling interface. One well-studied waveguide architecture that can achieve single-mode behavior with micrometer dimensions is the antiresonant reflective optical waveguide (ARROW).

Furthermore, this structure uses a reflective cladding of alternating layers of high and low refractive indices to push more light above the surface of the waveguide, and thus the sensitivity is not sacrificed with the larger dimensions. The ARROW was first proposed in an integrated MZI by Jimenez et al. [84], and was later implemented for biosensing by the same group [85]. Also, a unique ARROW-based MZI structure has been proposed by Hsu and Huang that utilizes a vertically stacked dual-core waveguide structure [86].

Further advancements to the integrated MZI functionality have been incorporated by Blanco et al. [87]. In this work, integrated MZI devices were fabricated on a wafer. Then, a microfluidic network was constructed on top of the wafer using CMOS compatible multilevel polymer assembly technology. This network effectively interfaces with external tubing and precisely delivers the sample to the sensing area. While this was the first demonstration of the integration of microfluidics onto the wafer, it is expected that this is ultimately required for the effective development of waveguide-based interferometry biosensors.

# 2.2.2. Young's interferometer

In parallel to the development of integrated MZI sensors, another interferometric technique based on an integrated Young's interferometer (YI) has been used in a similar way for biosensing. In a YI, the coherent, single mode, single polarization light at the input is split into a reference arm and a sensing arm. Instead of recombining the arms as in the MZI, the optical output of the two arms combines to form interference fringes on a detector screen, such as a CCD. The spatial intensity distribution, $I(x)$ , along the detector screen, $x$ , is given

14

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

by [88]:

$$
I (x) = 1 + \cos \left(\frac {2 \pi n d}{\lambda_ {0} f} x - \delta\right), \tag {3}
$$

where $\lambda_0$ is the wavelength in vacuum, $n$ the effective RI, $\delta$ represents a phase shift due to a RI change, and $d$ and $f$ are the distance between the two arms and the distance between the YI output and the screen, respectively. The equation shows that a phase change on the reference arm causes a shift in the position of the interference fringes, which can be analyzed quickly with a fast Fourier transform of the spatial intensity [88].

The first demonstration of an integrated YI for sensing was demonstrated by Brandenburg and Henninger [88]. A follow-up to this work using the same design concept yielded a RI DL of $10^{-7}$ RIU [89]. This is equivalent to the highest performance from the MZI, as expected, as the sensing mechanisms are equivalent in the two interferometer designs. An enhancement to the basic design of the integrated YI was proposed by Ymeti et al. a few years later to enable multiplexed sensing [90]. As shown in Fig. 3(B), the design uses four waveguide branches, three for sensing and one for reference.

The spacing between each sensing arm and the reference arm is unique and does not share any harmonics. It can be deduced from Eq. (3) that the frequency of the sinusoidal spatial intensity is dependent upon the spacing between the reference and sensing channel, and thus three unique spatial frequencies exist that can be resolved with Fourier analysis [90]. Within this work, the authors also optimized the design geometry and the signal processing techniques in an effort to minimize phase noise and inter-channel cross-talk.

Ultimately, it was concluded that the RI DL of this device is slightly better than $10^{-7}$ RIU. These authors later demonstrated biosensing with this YI chip design [91]. Specific detection of herpes simplex virus type 1 (HSV-1) was performed using an anti-HSV-1 antibody immobilized on the interferometer surface, showing that $10^{5}$ particles $\mathrm{mL}^{-1}$ in serum could be detected within only $30\mathrm{min}$ and that a DL of $10^{3}$ particles $\mathrm{mL}^{-1}$ in phosphate buffered saline could be achieved.

A commercial product is currently available based on the integrated YI technique from Fairfield Scientific [92]. The product, called AnaLight®, uses a polarization controller so that the two polarizations of light can sequentially probe the sample. This additional information enables the simultaneous measurement of analyte deposition quantification and RI measurement. Experimental demonstrations of the analysis of biotin/streptavidin binding utilizing the AnaLight have been published [93,94].

# 2.2.3. Hartman interferometer

A third and very similar interferometry configuration that has been used for biosensing is the Hartman interferometer (HI). In this configuration, functionalization molecules are patterned in strips on top of a planar waveguide, as shown in Fig. 3(C) [95]. Light is coupled into and out of the planar waveguide using gratings. Integrated optics is placed near the output of the chip to create interference between pairs of functionalized strips. In the first biosensing demonstration with the device, human chorionic gonadotropin (hCG) was detected

down to concentrations of $5\mathrm{ngmL}^{-1}$ [95]. Also, detection down to $4\mathrm{ngmL}^{-1}$ of a 44-base-pair oligonucleotide with a 19-base-pair target sequence was demonstrated. The authors further demonstrated detection of Influenza A and Salmonella. In follow-up work, the authors performed detection of hCG, first in serum [96], and then in whole blood [97].

To reduce the non-specific adsorption signal, the authors used the reference channel for subtraction (an antibody that is not specific to hCG was immobilized onto the reference arm) and using
biotin-(poly)ethylene glycol (PEG) as a blocking agent. Also, the authors were able to demonstrate detection of $0.1\mathrm{ngmL}^{-1}$ by using an amplification method. After the sample was run, antibodies targeting hCG labeled with metal nanoparticles were allowed to bind with the captured hCG [96,97].

This caused an additional phase shift at the output that is larger than the phase shift due to the hCG molecules alone.

# 2.2.4. Backscattering interferometry

Although the three interferometry techniques described thus far have differences in their design, the principle of detection is quite the same. A guided wave undergoes a phase change as its evanescent field interacts with the sample. Thus, all three designs ultimately have similar performance, as well as similar design constraints. Most notably, to produce a sensitive device, a long interaction length between guided wave and sample is necessary. As a result, the size of these interferometric structures is often measured in centimeters.

A separate class of interferometry-based biosensors exist called backscattering interferometry (BI). Some (but not all) BI implementations have achieved comparable performance as the previously described interferometry techniques while using a much smaller sensing area. Typically, a BI system consists of a coherent single wavelength laser focused onto a small sensing area and a detector to analyze the reflected intensity. Sub-wavelength structures on the sensing surface result in interference at the detector. Capture of proteins onto a selected part of the illuminated structure change the intensity on the detector due to a phase change of the reflection from this surface. This produces a sensing transduction signal.

This technique was used by Lin et al., who demonstrated one of the first implementations of this concept using a porous silicon sensor surface [98]. The top and bottom surfaces of the porous sensor serve as the reflecting surfaces. When analytes are specifically captured by biorecognition molecules immobilized inside the pores, a shift in the interference signal is observed. The authors show detection of 2pM target DNA using this sensor.

A second use of the BI technique is the biological compact disk (BioCD) immunoassay [99,100]. In this work, a reflective disk is patterned with biorecognition molecules, such as antibodies. The disk is then "read" by spinning it while a focused laser beam reflects off the disk surface and the antibody spots. The edges of the biomolecular spots result in an interference signal. The disk is then incubated with the sample. If the biorecognition spots capture proteins, the interference signal changes when the disk is read again after drying. The authors have demonstrated detection of $100\mathrm{pgmL}^{-1}$ of protein [100]. This technique may provide the capability for ultra-fast readout of thousands of sensors on the CD. The BioCD has been commercialized by QuadraSpec [101].

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

15

![](dt=2026-05-11/ht=18/f7e962d18ccce2fa4a9bb90eef477bc4d4f4de1a93ef1db9bc91168b62ea2605.jpg)

In a related technique, Chang et al. performed BI detection by patterning a reflective surface with strips of biorecognition molecules in a grating array which then capture polystyrene beads coated with the ligand [102]. In this work, the authors created strips of biotinylated BSA and then incubated the sensor surface in a solution with streptavidin coated beads. The resulting interference signal has the potential to provide a quantitative biological assay. The use of beads may serve as an excellent amplification mechanism.

In another use of backscattering interferometry, Markov et al. utilized the interference fringe pattern obtained from illuminating a simple, rectangular microfluidic channel from the top with a laser [103]. The walls of the microfluidic channel can be patterned with biorecognition molecules and then used to capture analytes from the sample. The authors showed that as target analytes were captured by the channel, the interference signal measured by an avalanche photodiode produced a sensing signal.

As a biodetection example, the binding of IgG molecules by immobilized Protein A was demonstrated. Importantly, this design enables simple microfluidics with small sample volume. This group also demonstrated similar biosensing results using a capillary microfluidic channel [104]. This system had previously been shown to have a RI DL of $10^{-9}$ RIU when used in a dual capillary configuration with a reference channel [105].

Later, returning to the rectangular microfluidic channel concept, this group demonstrated the use of BI for detection of biomolecular interactions in free solution, as opposed to surface-based interactions only [106]. While the picomolar DL is similar to other surface-sensing techniques, this is the first known report of label-free detection of biomolecular interactions in free solution.

# 2.3. Optical waveguide based biosensors

In the previous section, we have discussed interferometer-based biosensors, some of which employ waveguide structures as the sensing element. In this section, we will introduce a few new waveguide structures that have unique designs and distinctive characteristics in comparison with the conventional waveguides.

The resonant mirror (RM) design has been used as a biosensor sensor for over a decade [107-112]. In a resonant mirror,

the mode is leaky at the waveguide-substrate boundary, in contrast to a conventional waveguide in which total internal reflection occurs at both waveguide-solution and waveguidesubstrate boundaries. Fig. 4(B) shows the structure of the RM, where the high RI waveguide and the high RI substrate (or prism) are separated by a metal layer or a low loss and low RI dielectric layer. When the incident light is at the resonant angle, the light can be coupled strongly into the high-index waveguide layer and has a strong reflection at the output side of the RM.

This resonant light travels along in the waveguide and has the evanescent field outside the waveguide. As a result, the resonant angle is sensitive to the RI change near the waveguide, which can be detected at the output side of the RM. Note that, unlike SPR, which operates only at the TM mode, RM supports both TE and TM modes, which have different resonant angles [107]. The RM-based biosensor is commercialized by NeoSensors with a DL on the order of $0.1\mathrm{pgmm}^{-2}$ [113].

Metal clad waveguide (MCWG) or metal clad leaky waveguide (MCLW) is another optical structure that involves a metal layer [114-121]. The structure of the MCWG is schematically shown in Fig. 4(C). Note that in the MCWG, the light is guided in a low RI layer and the metal layer works as a spacer to separate the low RI waveguide and the high RI substrate, and to push more light into the solution layer to increase the light-matter interaction and hence the sensitivity. There are two types of MCWGs, dip type and peak type [118].

When the imaginary part of the metal dielectric constant is small and the metal thickness is on the order of tens of nanometers, the MCWG is operated as a dip type, in which the reflection features a characteristic resonant dip. On the other hand, when the imaginary part of metal dielectric constant is high and the metal thickness is thin (only a few nanometers), the MCWG is operated as a peak type and a characteristic peak emerges in the reflection. Both the dip and peak can be used as a sensing signal in which the response is an angular shift due to changes of the RI of the solution layer.

Although both dip and peak types of the MCWG have a lower sensitivity than SPR, they have a much narrower dip or peak, making them a competitive sensing technology to SPR [118].

The MCWG has much longer penetration depth than RM, SPR, and regular waveguide [115]. A comparison between RM and the MCWG s
hows the MCWG is three times more sensitive

16

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

to the bulk RI change in the solution, indicating a much larger fraction of light in the evanescent field [116,120]. Therefore, the MCWG is very suitable for detection of cells that are typically a few microns in size [115,116,119,121]. Non-specific detection of cells has been carried out on both dip and peak type MCWG with a DL of 8-9 cells $\mathrm{mm}^{-2}$ [121]. A DL of $8\times 10^{4}$ spores $\mathrm{mL}^{-1}$ for Bacillus subtilis var. niger bacterial spores was also demonstrated in a peak-type waveguide sensor [119]. Lipid bilayer and interactions between protein and a lipid bilayer were studied with the dip type MCWG [114].

Reverse symmetry waveguides bear the similar design motivation, which is to push more light into solution to increase the light interaction with large biomolecules, such as cells [122-125]. In the conventional waveguide geometry, the lower cladding layer has a RI higher than that of solution, which usually consists of water and has a RI of 1.33. Therefore, the waveguide mode has a light intensity more concentrated toward the lower cladding layer, thus leaving less light to interact with the analytes near to sensing surface. As shown in Fig.

4(D), in a reverse symmetry waveguide, nanoporous silica is used as a lower cladding layer. Nanoporous silica has a RI of 1.193, much lower than that of water. As a result, more light can be concentrated near the sensing surface to enhance the sensor sensitivity. Non-specific detection of Escherichia coli was demonstrated with reverse symmetry waveguides. The DL is estimated to be 60 cells $\mathrm{mm}^{-2}$ [123], a few hundreds times better than that in the RM [108].

Recently, a symmetrical metal-clad waveguide was also investigated theoretically and experimentally [126-129] (Fig. 4(E)). It is shown that ultrahigh-order modes can be excited through free space coupling. Those modes are sen

sitive to the RI and thickness change in the waveguide and can potentially be used for biosensing. Additionally, a large Goos-Hänchen shift was observed for the light incident upon the upper metal clad, which was recently employed for RI detection when the waveguide was replaced by a sample cell [129]. A DL of $2 \times 10^{-7}$ RIU was reported. For the future work, it will be interesting to see its applications in biosensing.

# 2.4. Optical ring resonator based biosensors

Optical microring resonators are an emerging sensing technology that has recently been under intensive investigation [7,10,12,130-190]. In a ring resonator, the light propagates in the form of whispering gallery modes (WGMs) or circulating waveguide modes [191-193], which result from total internal reflection of light along the curved boundary between the high and low refractive index (RI) media. Since the WGM and circulating waveguide mode can be calculated using Mie theory [191-193], for simplicity, we will use the term "WGM" to describe both types of modes.

The WGM has the evanescent field present at the ring resonator surface and responds to the binding of biomolecules. In contrast to the straight waveguide, the effective light-analyte interaction length of a ring resonator sensor is no longer determined by the sensor's physical size, but rather by the number of revolutions of the light supported by the resonator, which is characterized by the resonator quality factor, or the $Q$ -factor.

The effective length, $L_{\mathrm{eff}}$ , is related to the $Q$ -factor by [194]: $L_{\mathrm{eff}} = Q\lambda / 2\pi n$ , where $\lambda$ is the resonant wavelength and $n$ is the RI of the ring resonator. For example, for a ring resonator with a $Q$ -factor of $10^6$ , $n = 1.45$ , and $\lambda = 1550\mathrm{nm}$ , $L_{\mathrm{eff}}$ can be as long as

![](dt=2026-05-11/ht=18/f64aaa6e5edd55a805926c9db053c037733122de8b84747e71bd17f99151dda1.jpg)

![](dt=2026-05-11/ht=18/30e7a672bd164b0deafd29e0c1bce0eb807e54175cfde13ec676e478ddf0cd0c.jpg)

![](dt=2026-05-11/ht=18/64201a00346eb62e54501b456c37ea6e7f92b746faf9f4c43c1ca8ffaa663107.jpg)

![](dt=2026-05-11/ht=18/f09430a05ec37bff93528b139ac7767f114e162fb0af88b010c2707b410f7876.jpg)

![](dt=2026-05-11/ht=18/72c4196655672ebf090a28c013055bd7e0eded95c366266de3d504ba8073ee1c.jpg)

![](dt=2026-05-11/ht=18/ff2a158d26c65fe5fd3f870c6f96e4f7436e878337a833ba5c7d022001a4814c.jpg)

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

17

17 cm. Consequently, despite its small physical size, a ring resonator can deliver sensing performance similar or superior to a waveguide while using orders of magnitude less surface area and sample volume. Furthermore, due to the small size of ring resonators, high density sensor integration becomes possible.

The WGM spectral position, i.e., resonant wavelength, $\lambda$ , is related to the RI through the resonant condition [191]: $\lambda = 2\pi r n_{\mathrm{eff}} / m$ , where $r$ is the ring outer radius, $n_{\mathrm{eff}}$ the effective RI experienced by the WGM, and $m$ is an integer. $n_{\mathrm{eff}}$ changes when the RI near the ring resonator surface is modified due to the capture of target molecules on the surface, which in turn leads to a shift in the WGM spectral position. Thus, by directly or indirectly monitoring the WGM spectral shift, it is possible to obtain both quantitative and kinetic information about the binding of molecules near the surface.

Ring resonators biosensors have so far been implemented in three major configurations (see Fig. 5): (1) microfabricated ring-, disk-, or microtoroid-shaped resonators on a chip (Fig. 5(A)-(D)) [12,165-168,171-173,175-181,183-186,189,190]; (2) stand-alone dielectric microspheres [130-147,149,151,154, 169,174,182] (Fig. 5(E)); and (3) capillary-based opto-fluidic ring resonators (OFRR) (Fig. 5(F)) [7,10,148,150,152-164,170,187]. The advantages of the chip-based ring resonator include potential of mass production and optoelectronic integration capabilities.

However, except for a microtoroid, which has Q-factors larger than $10^{8}$ after it undergoes thermal reflow to smoothen the resonator surface, they usually have relatively low Q-factors (typically $10^{4}$ in water) due mainly to surface roughness [168,171,185]. Moreover, the fluidics needs to be fabricated separately and then mounted onto the ring resonator sensors through multiple complicated steps [185,195], which may present many design challenges.

In contrast, microsphere ring resonators possess much higher Q-factors ( $>10^{6}$ ) owing to the geometry and the extremely low surface roughness [194]. Microspheres of a few tens to a few hundred micrometers in diameter can easily be made by heating the end of an optical fiber by either an acetylene/hydrogen-oxygen torch or a $\mathrm{CO}_{2}$ laser. However, the major disadvantage of microsphere ring resonators is lack of mass production capability and difficulties in integration with fluidics, which requires a design of an external structure of channels and chambers [143].

The third design, the OFRR, has been recently introduced; it is based on a thin-walled capillary with a diameter of 50-200 μm and a wall thickness of a few micrometers [148,153] and can be made by pulling a glass preform under heat [148]. The OFRR can support high Q-factor ( $\sim 10^{6}$ ) WGMs in its circular cross-section that have the evanescent field inside the core to interact with the analytes flowing through the capillary, thus achieving dual use of the capillar
y as the sensor and the microfluidic channel.

# 2.4.1. Chip-based ring resonator sensors

Theoretically, in the limit of shot noise, the DL for RI is estimated to be on the order of $10^{-9}$ RIU [177]. However, the experimental results are still a few orders of magnitude away from it. A bulk RI sensitivity of $140\mathrm{nm}$ $\mathrm{RIU}^{-1}$ was achieved with a glass-based ring resonator of $60~{\mu\mathrm{m}}$ in radius with $Q\sim 12,000$ [171]. The DL was calculated to be $1.8\times 10^{-5}$ RIU [171]. Similarly, the DL of a polymer ring of $45~{\mu\mathrm{m}}$ in radius with $Q = 20,000$ was estimated to be $5\times 10^{-5}$ RIU [168]. However, a better DL

of $10^{-7}$ RIU was achieved in the same system by monitoring the intensity variation [168]. The DL for protein molecules is estimated to be $250\mathrm{pgmm}^{-2}$ [168].

In another work, a silicon-on-insulator (SOI) ring resonator on an area below $100\mu \mathrm{m}^2$ showed capability of detecting bulk RI change at a sensitivity of $70\mathrm{nmRIU^{-1}}$ with a DL of $7\times 10^{-5}$ RIU [185]. A DL of $10\mathrm{ngmL^{-1}}$ of avidin was reported on this device, which corresponds to only 1 fg molecular mass on the sensor due to the small size of the resonator [185]. Similar DL (0.1 nM, $6.8\mathrm{ngmL^{-1}}$ for avidin) was also achieved with a 2-mm diameter $\mathrm{Si_xN_y / SiO_2}$ ring resonator [173]. DNA and bacteria detection were also demonstrated with the DL for E. coli of $10^{5}$ cfu mL $^{-1}$ [186].

Recently, the slot-waveguide based ring resonator was developed, which has a larger light-analyte interaction and hence a higher sensitivity, as more light is present in the slot through which biomolecules flow [189,190]. A slot-waveguide ring resonator showed a RI sensitivity of over $200\mathrm{nmRIU}^{-1}$ and a DL of $2\times 10^{-4}$ RIU [189], and an estimated protein DL of approximately $20\mathrm{pgmm}^{-2}$ [190].

# 2.4.2. Stand-alone microsphere based ring resonator sensors

While microsphere-based ring resonators have a lower RI sensitivity $(20 - 50\mathrm{nm}\mathrm{RIU}^{-1})$ due to their relatively large size $(\sim 100\mu \mathrm{m}$ in diameter) [144], they benefit from a better spectral resolution resulting from much higher Q-factors $(>10^{6})$ [131,134,144]. As a consequence, the DL on the order of $10^{-7}$ RIU can be achieved by directly monitoring the WGM spectral shift [144].

Streptavidin detection has been performed with microspheres. Binding stoichiometry of 1:1 between biotin-BSA and streptavidin was estimated from the magnitude of the WGM shift. Due to high Q-factors, small molecule ( $\sim 50$ Da) detection was predicted and experimentally demonstrated [131,146]. Additionally, protease (such as trypsin) and thrombin detection were also carried out with microsphere sensors [145,151]. DNA detection with microspheres was first demonstrated by Vollmer et al. [134] with a DL of $1\mathrm{pgmm}^{-2}$ .

The authors further showed the detection of single nucleotide polymorphisms (SNP) using two microspheres. As the ring resonator supports multiple resonant wavelengths and both TE and TM modes, two wavelengths were used in a microsphere sensor to independently determine both the thickness and the optical dielectric constant [136]. Furthermore, simultaneous use of TE and TM modes was employed to determine the orientation of biomolecules on the surface [142].

A comprehensive work on viral detection with a microsphere sensor that includes theoretical discussion as well as experimental results was published recently [143]. It predicts that the microsphere is capable of detecting virions down to a single particle. Experimental demonstration was performed with MS2 at a concentration of $10^{9}$ pfu mL $^{-1}$ . A DL for surface density of $2\mathrm{pgmm}^{-2}$ was deduced.

Capabilities of microspheres of detecting larger bioparticles, like bacteria, were theoretically studied and experimentally demonstrated through simple adsorption [140,141]. The DL was estimated to be $\sim 1.2 \times 10^{2} \mathrm{~mm}^{-2}$ E. coli, which corresponds to $\sim 44$ bacteria bound to the sphere. With the average

18

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

dry weight of an E. coli cell $\sim 0.28\mathrm{pg}$ , this corresponds to a DL for bacterial dry-mass loading of $\sim 34\mathrm{pgmm}^{-2}$ [140]. Very recently, Brownian particles of a few tens to a few hundreds of nanometers were detected with a microsphere using the autocorrelation of resonance wavelength fluctuations [141].

# 2.4.3. Capillary-based opto-fluidic ring resonator sensors

A capillary-based opto-fluidic ring resonator (OFRR) with an outer radius of $50\mu \mathrm{m}$ and a wall thickness of $2 - 3\mu \mathrm{m}$ typically has a RI sensitivity of around $50\mathrm{nmRIU^{-1}}$ , a Q-factor over $10^{6}$ , and a DL on the order of $10^{-7}$ RIU, which are similar to those in microspheres. Protein adsorption measurement was recently accomplished with the OFRR. The DL for BSA adsorption is approximately $3\mathrm{pM}$ , corresponding to a surface density of $0.5\mathrm{pgmm}^{-2}$ and a total mass of a few femto-grams [157].

The OFRR was also used for actual breast cancer biomarker (CA15-3) detection with a DL on the order of 1 unit $\mathrm{mL}^{-1}$ , which is sufficiently sensitive to differentiate healthy women from breast cancer patients [196]. Quantitative detection of DNA sequences with base-mismatch studies have also been carried out using the OFRR. DNA concentration down to $10\mathrm{pM}$ was experimentally detected and the estimated mass density DL is $4\mathrm{pgmm}^{-2}$ [163].

The OFRR can also be employed for sensitive, rapid, on-line virus detection [164]. In this work, M13 filamentous bacteriophage was used as a safe model system. M13 is about $900\mathrm{nm}$ long and $10\mathrm{nm}$ wide and has similar molecular weight to serious human pathogens. The OFRR had a detection dynamic range of seven orders of magnitude $(2\times 10^{3} - 2\times 10^{10}\mathrm{pfu}\mathrm{mL}^{-1})$ . The experimental DL was on the order of $1000\mathrm{pfu}\mathrm{mL}^{-1}$ , which resulted in a WGM shift of a few pico-meters, well above the sensor noise level.

Additionally, the circular nature of the capillary enables the efficient capture and rapid detection of the analyte. Detection can be completed within a few minutes, even at very low virus concentration. Detection of a few viruses may be achievable with this device.

# 2.5. Optical fiber based biosensors

# 2.5.1. Fiber Bragg grating and long-period grating-based biosensors

Fibers are a convenient material for optical sensor design because they can be inexpensive and provide easy and efficient signal delivery [197,198]. Fiber Bragg gratings (FBGs), while developed as a tool for the telecommunications industry, have flourished as a versatile sensor with a wide breadth of applications. They are currently among the most popular of all fiber-based optical sensors for analyzing load, strain, temperature, vibration, and RI [199].

Illuminating the fiber with two focused intersecting laser beams, RI perturbations can be written into the fiber core that have periodicities $(\Lambda)$ on the order of the wavelength [200]. The resulting structure functions as a band rejection filter, reflecting a narrow band of light at the Bragg wavelength $(\lambda_{\mathrm{B}})$ according to the following relationship [201]:

$$
\lambda_ {\mathrm {B}} = 2 n _ {\text {e f f}} \Lambda , \tag {4}
$$

where $n_{\mathrm{eff}}$ is the effective RI encountered by the fiber core mode. By monitoring $\lambda_{\mathrm{B}}$ , the system functions as a RI sen

sor, which serves as the foundation for biochemical sensing functionality.

In order to expose the evanescent field from the FBG, several strategies have been pursued. One approach is to create a surface grating on the side of the fiber [202-204]. This usually requires bending and polishing until the fiber's core is
exposed, and then a grating is physically patterned on that surface. This design has yielded devices with a RI sensitivity as high as $340\mathrm{nmRIU^{-1}}$ and a DL down to $2\times 10^{-6}$ [204]. More recently biosensing was also carried out with a D-shaped fiber patterned with a FBG (Fig. 6(A)) which shows a DL of $10^{-5}$ RIU [205,206].

Another way of exposing the evanescent field involves chemically etching the optical fiber down to its core (see Fig. 6(B)) [207-210]. Chryssis et al. demonstrated a RI sensitivity of $1394\mathrm{nmRIU^{-1}}$ with a DL of $7\times 10^{-6}$ when the fiber was etched down to $3.4\mu \mathrm{m}$ in diameter [209]. Using the same sensor, $0.7\mu \mathrm{g}\mathrm{mL}^{-1}$ target DNA was detected with 20 base strands, which corresponds roughly to $0.1\mu \mathrm{M}$ [209]. This is still below par for label-free DNA detection, but it appears to be only the first proof of concept on this work.

In a fourth study, light-matter interaction was increased by inscribing FBGs into a microstructured fiber [211]. This configuration allows passage of sample solution through the holes where light intensity is high, allowing detection of $6\times 10^{-6}$ RIU.

Long-period gratings (LPGs), have attracted a great deal of attention in recent years for biochemical sensing applications although the advantages of LPGs have been known since the early 1990's [212,213]. These devices are manufactured with periodicity on the order of $100\mu \mathrm{m}$ to $1\mathrm{mm}$ , three or four orders of magnitude larger than traditional FBGs. In an LPG, core modes couple into the cladding modes of the fiber [214]. The characteristic resonance condition is expressed differently for LPGs [215]:

$$
\lambda = \left(n _ {\text {e f f - c o r e}} - n _ {\text {e f f - c l a d d i n g}}\right) \Lambda , \tag {5}
$$

where $n_{\mathrm{eff - core}}$ is the effective RI encountered by core modes, and $n_{\mathrm{eff - cladding}}$ is the effective RI encountered by cladding modes, which is affected by external RI. Because of this increased field intensity in the cladding of the fiber, the spectral position of the reflection peak becomes more sensitive to the RI change in the surrounding medium [216]. They therefore have the advantage of being able to sense the RI changes around the cladding without etching or complex structural design. Due to the large grating pitch, they are also far easier to manufacture.

LPGs can also be customized by chemically etching the cladding material down in order to enhance the sensitivity [214,217]. For example, in ref. [217], a sensitivity of $172\mathrm{nm}\mathrm{RIU}^{-1}$ was reached.

The earliest demonstration of biomolecule detection with LPGs was in 2000, when etched LPGs were used to detect antibodies [218], where goat anti-human IgG antibodies were linked to the fiber surface with glutaraldehyde and were then used to detect aqueous human IgG. These results showed repeatable detection down to $2\mu \mathrm{g}\mathrm{mL}^{-1}$ with a dynamic range of $2 - 100\mu \mathrm{g}\mathrm{mL}^{-1}$ . In 2007, Chen et al. were able to use etched LPGs for detection of hemoglobin [213]. In this work, the hemoglobin was allowed to adsorb non-specifically, yielding an estimated DL of around $0.005\%$ for hemoglobin in water.

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

19

![](dt=2026-05-11/ht=18/8df63461d3b1c8515c32cbf6cd9cd445763963a9fbb36f9d796f7121cbc11c77.jpg)

![](dt=2026-05-11/ht=18/dc84a544a25b76fb2b53640b7eb6c02635536b76acb768aa125ae4cd68be4144.jpg)

![](dt=2026-05-11/ht=18/05bb8f9fda3d451aef65264ca7e09149b87c0fb45ed0571d1d7663406269d3d4.jpg)

![](dt=2026-05-11/ht=18/f0cecd1686016bc8ddb77d9a63051385f3774d8e7c7e577ddf7c7be4a2935b94.jpg)

![](dt=2026-05-11/ht=18/2dd37567c6c26330b33a156fe06cfc63f4d55b9e3480a1ff3e5f4c52dd77bfa8.jpg)

![](dt=2026-05-11/ht=18/532a51c1c4c0ab36a85c558534deab8587fb5213b00ba93f10ea4db3f75c7df8.jpg)

This particular sensor utilized a dual-resonance measurement technique where at smaller grating periods two reflection maxima appear and may be measured relative to each other in order to achieve a better DL.

Photonic crystal fibers have been inscribed with LPG for detection of DNA and bulk RI very recently [9,219]. The fibers used were $30\mathrm{cm}$ in length with a grating pitch of $700\mu \mathrm{m}$ . These experiments demonstrated oligonucleotide deposition inside the holes using poly-L-lysine with a DL of $10^{-4}$ RIU. The authors also observed high sensitivity to biomolecule deposition, with $1.4\mathrm{nm}$ of spectral shift per nanometer of DNA monolayer deposition. This configuration presents the greatest potential for microfluidic efficiency of any fiber grating technology to date.

Overall, FBG and LPG sensors are very attractive due to their simple design, low manufacturing costs, and high compatibility with standard optical fibers. Most applications, however, focus on physical and basic RI sensing, because they lend themselves so readily to these purposes. For biochemical detection, these sensors cannot compete with other types of optical sensors in terms of their DLs even though their sensitivities can exceed $100\mathrm{nmRIU^{-1}}$ . Virtually all of the published work focused on FBG and LPG biosensing has come out within the last decade, and much of it has been within the last five years. It is therefore reasonable to expect that there will be substantial advancements made in this field in the very near future.

# 2.5.2. Other optical fiber based label-free biosensors

Aside from gratings, several other fiber-based technologies have been under development very recently that take advantage of optical fiber's cost effectiveness and its waveguiding capabilities. One technology is the nanofiber shown in Fig. 6(C), a silica fiber that is pulled under heat to a diameter less than $1\mu \mathrm{m}$ [220,221]. The nanofiber has a very large evanescent field outside of the fiber due to its small size, and therefore, is sensitive to the RI change.

There are a few optical structures developed based on the nanofibers, such as ring resonators, coils, and Mach-Zehnder interferometers [221-226]. It has been estimated that the DL for these sensors can approach $10^{-7}$ RIU [223] and the sensitivity can reach $700\mathrm{nm}$ RIU $^{-1}$ with fiber radii of $300\mathrm{nm}$ [222,224]. When used as the sensing element in a Mach-Zehnder interferometer, the nanofiber with a diameter of $400\mathrm{nm}$ and a sensing length of $1\mathrm{mm}$ can reach the DL of $4.2\times 10^{-4}$ monolayers of analytes [225].

Coupled optical fibers or fiber couplers, shown in Fig. 6(D), are another fiber-based biochemical sensor platform. In this case, two identical optical fibers are fused together and tapered to a diameter of $9\mu \mathrm{m}$ . The transmission spectrum is sinusoidal and shows a RI sensitivity close to $70\mathrm{nmRIU}^{-1}$ with a DL of $4\times 10^{-6}$ RIU [227]. Tazawa et al. demonstrated this sensor's ability to detect streptavidin with concentrations between 0.5 and $2\mu \mathrm{g}\mathrm{mL}^{-1}$ using covalent surface chemistries.

A third structure is fiber Fabry-Perot cavity based sensors formed by a short piece of hollow fiber, a slot sandwiched

20

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

between two pieces of optical fiber, or a pair of FBGs (Fig. 6(E)) [228-232]. It can also be formed by a piece of tapered fiber between a pair of FBGs (Fig. 6(F)) [207]. The spectral reflectance of these devi
ces is sensitive to the RI near the fiber tip or the fiber taper. A RI sensitivity of $71.2\mathrm{nmRIU}^{-1}$ and a DL of $1.4\times 10^{-5}$ RIU were demonstrated with a $5\mathrm{mm}$ long tapered fiber cavity [207].

This type of a device was also used to detect pig/anti-pig IgG with a concentration of $25\mu \mathrm{g}\mathrm{mL}^{-1}$ [228] and 26-base $\sim 76\mu \mathrm{M}$ oligonucleotide with a total surface mass of $1.7\mathrm{ng}$ [229]. Very recently, a slot fiber Fabry-Perot cavity showed a RI sensitivity in excess of $1000\mathrm{nmRIU}^{-1}$ [231,232]. However, its applications in biosensing have yet to be demonstrated.

# 2.6. Photonic crystal based sensors

Photonic crystal (PC) biosensors are a novel type of label-free optical biosensing platform, which has been presented in the recent literature [8,9,219,233-247]. A PC has a periodic dielectric structure with a periodicity on the order of a wavelength, from which forms the photonic bandgap. Therefore, the incident light whose wavelength lies within the photonic bandgap cannot propagate through the PC and a wide bandgap emerges on the transmission (or reflection) spectrum.

However, a photonic "defect" within the bandgap can be introduced by locally disturbing the PC's periodic structure, leading to the formation of the defect mode. The light resonant with the defect mode can propagate in the PC. As a result, on the transmission (or reflection) spectrum, the defect mode appears as a relatively sharp peak within the bandgap. Since spectral position of the defect mode is highly sensitive to the change in the

local environment around the "defect", it can be used as the sensing transduction signal when the RI changes as a result of the binding of the molecules to the defect. Due to the strong light confinement provided by the photonic bandgap and the flexibility of adjusting the defect mode wavelength across the photonic bandgap by finely tuning the structural parameters, PC biosensors have received tremendous attention since the first demonstration of detecting RI change in both waveguide [233] and microcavity laser [234] configurations.

A two-dimensional photonic crystal microcavity biosensor, as illustrated in Fig. 7(A), is formed by introducing a point defect into the periodic microstructure, either by increasing or decreasing the central hole. The presence of molecules inside the central hole and the first layer of surrounding holes will cause a local RI change, which can be monitored as the spectral shift of the resonant wavelength of the PC microcavity. In 2004, Chow et al. demonstrated the detection of 0.002 ambient RI changes with a passive PC microcavity with a Q-factor around 400 [235].

The DL could be improved to 0.001 RIU with the optimized PC microcavity laser structure [234]. A polymer-coated PC microcavity was also demonstrated for the detection of micromolar concentrations of specific ions in liquid solutions [236]. Later in 2007, Lee and Fauchet for the first time demonstrated a PC microcavity biosensor that is capable of monitoring protein binding on the walls of the defect hole and quantitatively measuring the protein diameter [237]. The minimum amount of protein that can be detected by the device was approximately $2.5\mathrm{fg}$ .

Shortly after, the same authors further improved the sensitivity down to $\sim 1\mathrm{fg}$ by precisely delivering the targets to the defect area only. Detection of a single latex sphere with a diameter of $370\mathrm{nm}$ was

![](dt=2026-05-11/ht=18/e3c5977a63d6f9375bee2c95e972a5b674016d904caef406ce132e3d2f0766d2.jpg)

![](dt=2026-05-11/ht=18/eb39ba37d8e7330fc45eba4d3b2538864337b714cfc33e24ee913691a3005f59.jpg)

![](dt=2026-05-11/ht=18/dd7c89caf15d96f5de7c02cf2f7bc0826ff90107b02e36e90fedd73a6f333789.jpg)

![](dt=2026-05-11/ht=18/f20be78ce6fa9156016ea9c302c1c2f48c0bf521c8ee62cb5385ba95cc25c92c.jpg)

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

21

achieved [238]. The PC microcavity had a sensing area of $\sim 40\mu \mathrm{m}^2$ , with an active sensing volume of $\sim 0.15\mu \mathrm{m}^3$ .

An optical biosensor based on light propagation through a PC waveguide (Fig. 7(B)) was presented by Skivesen et al. [239], where the cutoff wavelength of the PC waveguide was used as the indicator for surrounding RI changes. Concentrations of around $0.15\mu \mathrm{M}$ of BSA-solution were measured with excellent signal to noise ratio. However, compared to PC microcavity based biosensors, this PC waveguide biosensor suffers from relatively larger sensing area, which requires a larger sample volume.

A third class of PC biosensor is based on photonic crystal fiber (PCF), which is a fiber optic cable with air holes in the cross-section (Fig. 7(C)). PCF is a promising sensing platform for two reasons. First, the air holes in the fiber can act as a simple fluidic channel to deliver the biological samples. Second, the unique light guiding mechanism of photonic crystal fiber ensures a strong light-matter interaction because of the large light-sample overlap. The array of air holes may hold a sample volume of a few nL per cm of the fiber, which is a desirable advantage for biosensing applications. Rindorf et al. recently demonstrated the detection of biomolecules using a long-period grating in a PCF, which has been discussed in the previous section [9,219].

Cunningham et al. developed a unique type of PC biosensors that behaves as a narrowband wavelength reflectance filter, where the sensor reflects nearly $100\%$ of the incident light at the resonant wavelength [240]. The sensor operates by measuring changes in the wavelength of reflected light as biochemical or cellular binding events take place on the surface. The sensor is able to detect a molecular surface density of $0.4\mathrm{pgmm}^{-2}$ (after solution containing the protein is dried) [241]. There are a number of advantages for this PC sensor structure.

First, the sensor has a relatively good sensitivity as well as a large dynamic range, which enables it to measure biochemical entities from small size, such as biomolecular interactions [242,243] to large size, such as live cells, cell membranes, virions and bacteria (also protein-cell interactions) [244]. Second, the sensor structure is compatible with standard microtiter plate-based and microarray-based infrastructures that are most widely used for high throughput biomolecular interaction analysis.

Third, the sensor can be measured either in "single point mode" or "imaging mode", which greatly increases the overall resolution and throughput of the system. Fourth, the sensors are inexpensively manufactured on plastic film.

This group also showed that when the PC sensor is operated in the near ultraviolet wavelength range, the surface-sensitivity to bulk-sensitivity ratio will be enhanced over 4.5 times compared to the infrared range, which enables enhanced detection resolution for biomolecules on the biosensor surface and is very useful for small molecule detection or ultralow concentration detection [245]. This platform has now been commercialized by SRU Biosystems [248].

Mandal and Erickson recently demonstrated a novel optofluidic sensor array based on the use of arrays of 1D photonic crystal resonators and a single bus waveguide (Fig. 7(D)), which can perform highly parallel, label-free detection of biomolecular interactions [246]. The beauty of this structure is that each PC resonator adjacent to the bus waveguide is independent

from all others, which greatly enhances the device's multiplex detection capability. The simulation results predict that the DL is on the order of tens of attograms, and detection o
f $7 \times 10^{-5}$ RIU was demonstrated experimentally.

In summary, while the development of PC-based label-free biosensor is still in its initial stage, based on the experimental demonstrations achieved to date, PC structures are a very promising biosensor platform. In PC biosensors, light can be concentrated or localized in a very small volume, thus achieving a large light-matter interaction with a volume down to femtoliters. Parallel detection is also possible. However, further investigations are still needed to improve the overall performance, such as surface chemistry, compatible microfluidics for efficient sample delivery, and detection procedures, such as rinsing to minimize non-specific binding.

# 3. Outlook

In this article, we have overviewed recent progress in various optical biosensor structures for label-free detection. To date, some of them, like SPR, waveguides, and fiber gratings, are relatively mature and have even been commercialized. Others, such as ring resonators and photonic crystals, although still in their infancy, possess unique and advantageous properties over the state-of-the-art and may have great potential for next generation label-free biosensors.

Eventually, the goals of label-free biosensors are to develop portable or hand-held devices that can provide laboratory grade sensing performance with low cost, low sample volume and total mass, and improved multiplexed detection capabilities. Towards this end, we envision that a few levels of integration may be needed. The first integration is on the optical sensor structure itself. For example, nano-fibers described in Section 2.5.2 can be coiled to make a ring resonator (Section 2.4) or to make an interferometer (Section 2.2). FBG or LPG structures (Sections 2.5.1 and 2.5.

2) can be inscribed onto a photonic crystal fiber (Section 2.6). Combination or integration of different optical structures onto the same sensor will leverage the advantages of each individual structure to achieve a better performance. In addition, different detection protocols like fluorescence and Raman spectroscopy can also be incorporated with the label-free detection to provide complementary sensing information. The second level of integration is between optical structures and microfluidics.

Microfluidics has been under investigation for over a decade for lab-on-chip or micro total analysis systems. A particular area called "opto-fluidics" has recently drawn tremendous attention, which specifically addresses the issues of how microfluidics interfaces with optical detection [249-251]. A few examples have been given in this review, including opto-fluidic ring resonators (Section 2.4) and photonic crystal based LPG (Section 2.5.1). The last integration is to incorporate light sources (such as lasers) and detectors directly onto the same sensor chip [252,253].

Currently, while miniaturization of sensors is underway, the light sources and detectors used for those sensors are still bulky. Integration of light sources, sensors, and detectors, as well as the corresponding on-board or on-chip optical and electronic controls (such as color filters and signal processors)

22

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

will be a critical step towards achieving a truly hand-held device.

# Acknowledgments

The authors gratefully thank the support from American Chemical Society Petroleum Research Fund (43879-G10), Wallace H. Coulter Early Career Award, NSF (ECCS-0729903), and NIH (5K25EB006011-02).

# REFERENCES

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

23

24

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

ANALYTICA CHIMICA ACTA 620 (2008) 8-26

25

26

ANALYTICA CHIMICA ACTA 620 (2008) 8-26