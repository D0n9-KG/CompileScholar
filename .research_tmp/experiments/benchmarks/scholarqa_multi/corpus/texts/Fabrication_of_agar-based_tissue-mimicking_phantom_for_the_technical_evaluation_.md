# Fabrication of agar-based tissue-mimicking phantom for the technical evaluation of biomedical optical imaging systems

![](dt=2026-03-14/ht=06/53749ebdaa80cc3c63b6adb0403efb69b711c2fea26e574b14a63f7a6814a729.jpg)

Mingyu Kim a, Seonghui Im a, Inyoung Park b, Donghyeok Kim b, Eun Su Kim a, James Joseph c,d, Jonghee Yoon a,*

# ARTICLEINFO

# Keywords:

Tissue-mimicking phantom

Light-tissue interaction

Optical system

Technical evaluation

Soft tissue

# ABSTRACT

The development process of the optical systems for various biomedical applications typically involve evaluations of technical performance. One popular evaluation method is to use a reference object such as a phantom that exhibits similar optical properties of tissue. Fabrication of a consistent phantom with known optical properties, such as scattering and absorption, is essential for accurate technical evaluation of the optical system.

This paper presents a protocol for fabricating an agar-based tissue-mimicking phantom, offering practical guidance to ensure consistent and reproducible phantom creation. In addition, optical setups that measure light information required for quantifying the optical properties via an inverse adding-doubling (IAD) method are discussed. We demonstrated the fabrication of phantoms with diverse scattering and absorption properties, and the IAD method successfully quantified the optical properties.

Moreover, we employed the phantom to assess the imaging depth limitation of a hyperspectral imaging system, demonstrating potential usage of phantoms for performing technical evaluation.

# 1. Introduction

Biomedical optics technologies exploit light-tissue interactions for disease diagnosis and treatment, which can be broadly classified into two categories: label-free and label-based methods [1]. Label-free optical methods harness intrinsic optical properties of biological tissues, comprising cells, blood vessels, extracellular matrix (e.g., collagen), and various biochemical molecules [2]. From an optical perspective, the structural complexity of tissues introduces uneven refractive index distribution, causing light reflection and refraction at refractive index discontinuities.

Biochemical molecule composition determines optical absorption properties, exhibiting distinct features wavelength dependent light absorption. Nonlinear phenomena such as multiphoton absorption, second harmonic generation, and Raman scattering, induced by biochemical molecule structures, further contribute to optical features [3]. Consequently, tissue structural and biochemical features can be estimated from the measured optical properties, enabling the identification of tissues with pathological alterations during disease progression.

For example, in the case of cancer, rapid cell proliferation

results in changes in cell density, angiogenesis, and inflammatory responses, modifying tissue absorption and scattering properties [4-6]. This alteration in optical properties facilitates cancer diagnosis. Accurate measurement of tissue optical properties enables the assessment of structure and biochemical constituents, making label-free optical techniques invaluable for disease diagnosis and treatment. In contrast, label-based techniques employ fluorescent dyes, nanoparticles, and external agents to enhance target contrast [7].

Utilizing antigen-antibody reactions or other affinity-based interactions, these techniques precisely observe specific targets. This capability is particularly advantageous for measuring complex biological tissues that are challenging to assess due to their complex composition.

To harness the diverse light-tissue interactions for biomedical applications, several label-free and label-based optical imaging technologies have been developed [8]. Each imaging technique is designed to capture specific light-tissue interactions, and approaches for the optical system development typically involves theoretical and experimental demonstration of their utility for biomedical applications. Recently, several promising optical technologies have emerged, including:

Current Applied Physics 61 (2024) 80-85

ELSEVIER

Contents lists available at ScienceDirect

Current Applied Physics

journal homepage: www.elsevier.com/locate/cap

Current Applied Physics Physics, Geometry and Material Science

* Corresponding author.

E-mail address: jyoon48@ajou.ac.kr (J. Yoon).

https://doi.org/10.1016/j.cap.2024.02.013

Received 3 December 2023; Received in revised form 22 February 2024; Accepted 22 February 2024

Available online 28 February 2024

1567-1739/© 2024 The Authors. Published by Elsevier B.V. on behalf of Korean Physical Society. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/).

hyperspectral imaging (HSI) method which measures the morphology and spectral information for mapping the structural and biochemical features from biological tissues [9,10]. HSI enables disease diagnosis not achievable through conventional optical imaging methods. Another modality called spatial-frequency domain imaging (SFDI) utilizes structured illumination to measure absorption and scattering coefficients on a large scale [11]. On the other hand, photoacoustic imaging measures ultrasound signals generated by the absorption of light energy [12].

It is particularly useful for vascular imaging and cancer diagnosis, providing detailed information about tissue composition and vasculature distribution. Laser speckle imaging focuses on the temporal dynamics of scattered light signals caused by variations in the medium [12]. This allows for the quantitative analysis of blood flow, providing insights into physiological processes [13].

Although a variety of optical imaging devices have been reported, the clinical translation of these technologies often comprises the usage of robust technical validation frameworks. The technical validation of optical imaging technologies primarily involves simulation and experimental methods [14]. Simulation methods rely on computational approaches to predict precise optical characteristics through theoretical analysis [15,16].

While simulations are powerful tools for anticipating optical behavior, they have limitations in accurately assessing complex errors that may arise in the actual optical system. In contrast, experimental validation methods utilize tissue-mimicking phantoms that are designed to have similar optical properties of biological tissues [14, 17-19]. These phantoms serve as invaluable tools for assessing optical systems under various conditions, providing a practical means of validating the technologies for comprehensive understanding of optical systems in real-world conditions.

Biological tissue-mimicking phantoms are artificially created to replicate the optical properties of actual biological tissues. They can be broadly categorized into water-based and non-water-based phantoms [14,17]. Water-based phantoms are relatively easy to fabricate and offer the advantage of using various substances to control their optical properties. However, they have the drawback of evaporation over time, limiting their stability and suitability for long-term experiments.

On the other hand, non-water-based phantoms involve a more complex manufacturing process, but they offer the advantage of long-term stability without significant changes in their properties. This makes them particularly useful for extended longitudinal experiments and studies. To control scattering in these phantoms, substances like lipid, titanium oxide, and polymer microspheres can be employed, while absorption coefficients can be adjusted using substances like natural tissue chromophores or synthetic chemical dyes [17].

Additionally, depending on specific experimental requirements, fluorescent materials can be incorporated into the phantoms.

Considering the diverse o
ptical properties exhibited by different biological tissues, the creation of tissue-mimicking phantoms with tailored optical properties becomes essential for precise validation of optical systems. These phantoms enable researchers to mimic specific experimental conditions relevant to their research objectives, ensuring accurate performance testing of optical systems under a variety of circumstances.

Therefore, creating a consistent tissue-mimicking phantom with known optical properties is essential for harnessing the utility of phantoms for various applications. This paper provides a detailed description of water-based phantoms, along with the necessary techniques and insights required for their fabrication. Additionally, the paper presents a protocol for the performing computations using the inverse adding-doubling (IAD) method, which enables the measurement of the optical properties of the created phantoms.

Although many protocols of tissue-mimicking phantom creation and IAD methods were proposed, the details and practical issues were usually omitted. In this context, this paper particularly elaborates on crucial considerations for manufacturing a safe and reliable phantom and reports the optical measurements performed for the IAD based calculations. Specifically, the proposed

protocol created a phantom with optical properties similar to those of the esophagus and stomach tissues (absorption coefficients of 0.1–2.5 $\mathrm{cm}^{-1}$ , reduced scattering coefficients of $0.5 - 20\mathrm{cm}^{-1}$ at $550~\mathrm{nm}$ ) [20, 21]. Furthermore, the study included applied research exploiting the created phantoms to investigate the depth limits of hyperspectral imaging technology. This finding contributed to a better understanding of the capabilities and limitations of hyperspectral imaging and its potential applications.

# 2. Material and methods

# 2.1. Protocols for tissue phantom creation

An agar-based tissue-mimicking phantom was created in this study. Agar powder (05039, Sigma Aldrich) was used as the base material of the phantom. Agar concentration determines the stiffness of the phantom; thus, concentration could be adjusted for the stiffness of the target tissue [22]. Intralipid $20\%$ emulsion (68890-65-3, Sigma Aldrich) and Nigrosin (8005-03-6, Sigma Aldrich) were used to adjust the scattering and absorption features of the phantom, respectively. Nigrosin stock solution was prepared by dissolving in distilled water to have a concentration of $0.5\mathrm{mg / mL}$ . Fig. 1 shows overall phantom creation procedures, and the details of the protocols are as follows.

In this work, the phantoms were prepared using $35\mathrm{mm}$ Petri dishes, and $5\mathrm{mL}$ of phantom solution was poured into the dishes to create a phantom having thickness of $5\mathrm{mm}$ . The shape and thickness of the phantom were determined based on the field-of-view of an imaging system and the target tissue (the target tissue is gastric tissue in this protocol, and its thickness is around $5\mathrm{mm}$ [23]), but those parameters could be freely adjusted based on the optical system and characteristics of the target.

# 2.2. Inverse adding-doubling method

The optical properties of the phantom should be carefully measured

M. Kim et al.

Current Applied Physics 61 (2024) 80-85

81

![](dt=2026-03-14/ht=06/198dcafcef87da873700145110d718fb0937520c1242e5de758342cb7edb3bdf.jpg)

![](dt=2026-03-14/ht=06/ebff5f670f2e80c520bdf0769d7eb4722b9cb59e45b530022b0b53ba000256ad.jpg)

![](dt=2026-03-14/ht=06/32eea804706290dce60f2191b15f6f72eb7adec4eae9f676a1b0e61b24ce3fd6.jpg)

![](dt=2026-03-14/ht=06/efaca4c60c9f6397132b6bf26f25d08a6d0f8badb1b392a6547268a140e77eb6.jpg)

![](dt=2026-03-14/ht=06/c9bd6e8725e7a362f2d2cc18ebae29efc7bde843e0ec2a6d0b686e633584f454.jpg)

![](dt=2026-03-14/ht=06/901c79bb162c8d282529acccc4e69ef484a0d21e5d0c25371d6978be45c0394c.jpg)

prior to exploiting the tissue-mimicking phantom for the technical evaluation of an optical system. The IAD method is a well-known reported method to estimate absorption and reduced scattering coefficients. IAD relies on using the transmitted, reflected, and unscattered light measured from the sample [24,25]. Monte Carlo simulation is further performed using these measurement data to predict the optical properties of the sample. Due to its simplicity and robustness, the IAD method has been widely used in estimating the optical properties of various materials, including phantom and biological tissues.

There are many optical systems available to measure optical signals required for the IAD method. This study exploits the usage of a single integrating sphere (3P-GPS-033-SL 3, Labsphere) and supercontinuum

laser source (SuperK, NKT photonics). The integrating sphere allows a homogenous distribution of the optical field across the internal surface of the sphere thereby enabling the precise measurement of the transmission and reflectance of the sample. As refractive index and absorption features vary as a function of wavelength, the supercontinuum source interfaced with an acoustic-optic tunable filter (AOTF, SuperK Varia, NKT photonics) was exploited.

This illumination system enables the production of light output ranging from $450~\mathrm{nm}$ to $800~\mathrm{nm}$ with a spectral bandwidth of $10~\mathrm{nm}$ . This study involved the usage of $550~\mathrm{nm}$ illumination with $10~\mathrm{nm}$ bandwidth for demonstrating the measurement protocol of the IAD, but the same protocol can be applied for obtaining optical properties at other wavelengths by the simply tunning center

![](dt=2026-03-14/ht=06/f28ae0879904c6910ece59a31f093fea37a1444659403436273ae3d05e88e636.jpg)

![](dt=2026-03-14/ht=06/52e452f6e9a55e366f481717313daf2a055ac41e1678fd7731c9513d2e2e59e0.jpg)

![](dt=2026-03-14/ht=06/30185ae5bb58dcdbd67871760891c528b0de811e94508f067f3edfe11ea94957.jpg)

![](dt=2026-03-14/ht=06/f4a5c12ab9741e10f5980b68da14a45117a5198823f1354cca59455b26d4ce11.jpg)

M. Kim et al.

Current Applied Physics 61 (2024) 80-85

82

wavelength of the supercontinuum source.

Fig. 2 shows the optical setup and methods used to measure transmission, reflection, and unscattered light from a phantom used in this study. The beam size of the collimated laser was decreased by half via a 4f system made of two plano-convex lenses $(\mathrm{f} = 150\mathrm{mm}$ and $\mathrm{f} = 75\mathrm{mm})$ . A 4-inch integrating sphere with three ports was aligned to make the laser pass through the two ports without light reflection at the internal surface of the integrating surface.

A fiber-coupled spectrometer was connected to the port located at the top to capture the reflected light inside the integrating sphere. First, the total power of the incoming laser signal was measured by blocking the exit port with a port plug coated with spectralon (Plug, Port, $1.0^{\prime \prime}$ , Spectralon, Labsphere), which allows the reflection of all incident laser power.

In this step, the exposure time of the spectrometer should be carefully adjusted to make the measured signal close to the saturated level, which provides a sufficient signal-to-noise ratio of transmission and
reflection signals at the phantom, even though there exists high light attenuation. Once the exposure time of the spectrometer was determined, transmission and reflection signals from the phantom were measured by placing the phantom at the entrance and exit ports, respectively (Fig. 2(b) and (c)).

After the acquisition of transmission and reflection, the unscattered signal was measured by modifying the optical setup as shown Fig. 2(d). The integrating sphere was removed, and a spectrometer was positioned. The collimated light is focused into the spectrometer by using an additional plano-convex lens $(\mathrm{f} = 35\mathrm{mm})$ . Then, the exposure time was adjusted to avoid the spectrometer saturation due to the high intensities experienced while performing the direct measurement of the focused laser beam.

The phantom was then placed in front of the first lens, which allowed the measurement of unscattered light signals. As the spectrometer was aligned to measure only the collimated laser, the scattered signals from the phantom could not reach the spectrometer. Dark signals for each exposure time were measured without laser illumination.

Based on measured transmission, reflection, and unscattered light, absorption and reduced scattering coefficients were estimated using the IAD method [25].

# 3. Results and discussion

# 3.1. Estimation of the optical properties of phantoms using the IAD method

To demonstrate the feasibility of the protocols of tissue mimicking phantom fabrication and the IAD method, two different sets of tissue phantoms were fabricated. The first set of phantoms were created by changing intralipid concentration while maintaining Nigrosin concentration, which adjusted scattering properties without altering the absorption properties. The other set of phantoms were fabricated with varying absorption properties but the same scattering property. Each set consists of four tissue phantoms, and the composition is summarized in Table 1.

Fig. 3 shows the images of each sample and their optical properties computed using the IAD method at $550~\mathrm{nm}$ . Phantoms with increasing concentrations of intralipid were indistinguishable in photos (Fig. 3(a)). But IAD measurement showed increased reduced scattering coefficients with an increase in the intralipid concentration. On the other hand, absorption properties were similar in these phantoms, indicating that intralipid does not change the absorption property of the phantom.

However, phantoms become darker with increasing Nigrosin concentration as shown in Fig. 3(b). Quantification of optical properties of the phantoms with varying Nigrosin concentration revealed that absorption coefficients were increased as the Nigrosin concentration increased. However, there was also a rapid decrease in the reduced scattering coefficients between the Nigrosin concentration of $3.1~\mu \mathrm{g / mL}$ and $4.7~\mu \mathrm{g}/$ mL.

These results indicate that Nigrosin can adjust the absorption property of a phantom, but high Nigrosin concentration might affect the quantification accuracy of the scattering property. Thus, further study is required to determine whether Nigrosin affects the quantification of the reduced scattering coefficient or decreased light intensity due to high absorption in the phantom affects the quantification accuracy of the scattering coefficient.

Although there was an issue in measuring consistent reduced scattering coefficients in the phantoms with the same intralipid concentration, the feasibility of tissue phantom creation with varying optical properties was demonstrated, and the IAD was proved to be versatile to estimate the optical properties of the phantom.

# 3.2. Application of a phantom for evaluating penetration depth limitation of HSI

To test the applicability of a tissue-mimicking phantom in the technical evaluation of biophotonics tools, we applied the phantom to assess the imaging depth of an HSI system. A drawback of HSI methods is the limited imaging depth, as optical penetration is affected by absorption and scattering in tissue [26,27]. Thus, current HSI methods applications mainly examine diseases in organs that can be easily accessed via optical imaging systems [28].

Knowing the depth limitation of an HSI system is significant as it is directly related to the disease diagnostic capability of lesions located under the surface [29]. Therefore, we applied phantoms with various thickness to assess the imaging depth of the HSI system. An HSI system was implemented via a spectral scanning method using a liquid crystal tunable filter (LCTF, KURI OS-VB1, Thorlabs) (Fig. 4(a)).

The LCTF, placed in front of a CMOS camera (Grasshopper3 GS3-U3-41C6M, FLIR), allows the electronic adjustment of the filter property to transmit a light signal at a specific wavelength, enabling the acquisition of spectral images. We performed HSI of a color chart by illuminating a broadband diffuse light source (LEDP260C, Godox), allowing the acquisition of spectral information at visible wavelength ranges from $450\mathrm{nm}$ to $700\mathrm{nm}$ .

Phantoms with three thicknesses (1.5 mm, $3.0\mathrm{mm}$ , and $7.0\mathrm{mm}$ ) were put on the top of the color chart to investigate the depth limitation of the HSI system. Phantoms were created with concentrations of intralipid $(0.83\%)$ and Nigrosin $(3.1\mu \mathrm{g} / \mathrm{mL})$ to have similar optical properties of the skin [30].

Fig. 4(b) shows the effect of phantom thickness on the target in HSI. The RGB image without the phantom clearly shows four colors (red, yellow, green, and blue), and they have a distinct reflectance at visible wavelength, ranging from 450 to $700\mathrm{nm}$ . However, RGB images became vague and white as the phantom's thickness increased from $1.5\mathrm{mm}$ to $7.0\mathrm{mm}$ . Moreover, the discrimination of each color based on reflectance is challenging due to indistinguishable spectral features under thick phantom conditions.

These results indicate that thick phantoms result in higher light extinction, which makes it difficult to measure the accurate spectral features of the underlying target. Interestingly, the reflectance of each color around $600 - 700\mathrm{nm}$ is still discernible even though the thickness of the phantom is $7\mathrm{mm}$ . This is because longer wavelengths

Table 1 Details of the composition of tissue-mimicking phantoms.

![](dt=2026-03-14/ht=06/caa472dea8245ee277d53e929a7dc1148c7338b72e62e6f682263c30feaa3411.jpg)

<table><tr><td></td><td colspan="4">Intralipid varying phantoms</td><td colspan="4">Nigrosin varying phantoms</td></tr><tr><td>Agar powder (g)</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.15</td></tr><tr><td>Distilled water (mL)</td><td>9.730</td><td>9.522</td><td>9.314</td><td>9.106</td><td>9.553</td><td>9.522</td><td>9.491</td><td>9.460</td></tr><tr><td>20% intralipid solution (mL)</td><td>0.208</td><td>0.416</td><td>0.624</td><td>0.832</td><td>0.416</td><td>0.416</td><td>0.416</td><td>0.416</td></tr><tr><td>5 mg/mL Nygrosin solution (mL)</td><td>0.062</td><td>0.062</td><td>0.062</td><td>0.062</td><td>0.031</td><td>0.062</td><td>0.093</td><td>0.124</td></tr></table>

M. Kim et al.

Current Applied Physics 61 (2024) 80-85

83

![](dt=2026-03-14/ht=06/229295f78300d279eeff4a02c0140d663379e82736a44bbb38f2bb9c21db1ffd.jpg)

![](dt=2026-03-14/ht=06/0d007992556685855d4ebf8e86d17bd87a6ab326bb32fd2f854f79848c9aacdf.jpg)

![](dt=2026-03-14/ht=06/4dbf470ea0a7eec83a0ef1f50278bdff3c453c5d170d382b56e97a9aeea8e0aa.jpg)

![](image)
cess/type=image/dt=2026-03-14/ht=06//ff9ce13e1a23ee54606c5d2eb0e49136b8e8db8b0a24e44f52aee9b171c228c8.jpg)

![](dt=2026-03-14/ht=06/595b12652559562b81c852a4516ad0f5567f0d47677e83b5f80a13ffc3d7c343.jpg)

![](dt=2026-03-14/ht=06/0cfc689735f4ae2d4c290a473d0446e15cb0c4efa8849c78c98dcb0c117ee14e.jpg)

![](dt=2026-03-14/ht=06/97eab07ad1675c2cdfa2de8f7a6a7f9e2f6e1df8eb6d8b4ce3e1b5a5fdfdac97.jpg)

![](dt=2026-03-14/ht=06/906a1203817935f9e284fb47595e2640aed28c32fb32e438cfcda975feea51b1.jpg)

![](dt=2026-03-14/ht=06/d0461f2bf0de41188f2d6e6a0d1d39fde1b827927b4ec93b929951ea74700409.jpg)

![](dt=2026-03-14/ht=06/f10b42ef90ae8789dd315688c2b30a68fc11b605838c5932e0c0df8b7f744c35.jpg)

![](dt=2026-03-14/ht=06/bc6cd5fcad6605e4c4319d39983546298e4ccf58cfcb410c69f094cfb4f9fb95.jpg)

![](dt=2026-03-14/ht=06/c0d45124165372de6599eaacc4f4d024e1970821272257608d7356d8d5967467.jpg)

![](dt=2026-03-14/ht=06/69e4f65e154d74e7998e9b3d9f563acef0e6a86b54e4290296d4f3ce81517f27.jpg)

![](dt=2026-03-14/ht=06/7898f634541edb1e95c40b15197cbd3490001abe5b26809862e8ae1ed9decde5.jpg)

![](dt=2026-03-14/ht=06/6b6f28179e03ebb6815b58fac2da4e4f17195843afd450e1383fc615821e90e0.jpg)

![](dt=2026-03-14/ht=06/5e53f30543cbd8c4860e43d2bff4906be8c4f81d43f066894d39f47c9bfac53c.jpg)

![](dt=2026-03-14/ht=06/d5ef4fc1f8007df28146ea44760e242733451a1a1f64a9e591716ad439b492d9.jpg)

are attenuated less when compared to shorter wavelengths in agreement with the lower Nigrosin absorption and scattering at longer wavelengths. These results show a consistent tendency with a previous report that HSI with longer wavelength enables the measurement of targets with enough contrast [31].

Although the scale of measured reflectance is decreased as the thickness of the phantom increases, the shapes of reflectance curves were maintained for image acquisition of the phantom with $3\mathrm{mm}$ thickness. This indicates that this HSI system can be used to detect a pathological condition located under $3\mathrm{mm}$ from the surface.

# 4. Conclusion

This paper reports a protocol for creating an agar-based tissue-mimicking phantom that enables the tuning of scattering and absorption properties by changing intralipid and Nigrosin concentrations, respectively. Moreover, practical guidelines were also provided for enabling consistent and reproducible phantom fabrication. Optical setups were proposed to measure transmission, reflection, and unscattered light information from a phantom, which is required for characterizing the optical properties of a phantom using the IAD method. We found that measuring unscattered light is significant to obtain accurate calculation of the optical properties.

As many optical systems have been developed to examine the gastrointestinal tract due to easy accessibility of useful optical tools, this study specifically targeted the fabrication of tissue-mimicking phantoms with absorption and scattering properties similar to the esophagus and gastric tissues. Various phantoms were created with different

concentrations of intralipid and Nigrosin, and the IAD method successfully quantified the optical properties of phantoms that are consistent with intralipid and Nigrosin concentrations. For the representative application of a tissue-mimicking phantom in the technical evaluation of biomedical optics system, we implemented the HSI system via the spectral scanning method. The phantoms with different thicknesses enabled the analysis of the depth limitation of the HSI system, which can be used for identifying the maximal depth of disease tissue that can be diagnosed via the HSI system.

When the proposed protocol is combined with a 3D printer technology, its applicability can significantly increase [32,33]. The 3D printer provides a mold that can mimic complex biological structures such as vessel networks, allowing the technical evaluation of optical systems under actual biological tissue conditions. In addition, combining 3D printer technology and fluidic channels enables the assessment of functional imaging capabilities such as blood flow and oxygen saturation monitoring [34,35]. Collectively, the tissue-mimicking phantom is an essential tool for evaluating biomedical optical systems, and it can also be used for the standardization of optical imaging techniques that can be used for a variety of biomedical applications.

# Declaration of competing interest

The authors have no competing financial interests to declare.

M. Kim et al.

Current Applied Physics 61 (2024) 80-85

84

# Author contributions

JY and MK conceived the study. JY, IP, DK, ESK, and SI designed and performed experiments and analyzed the data. JJ revised and reviewed the manuscript. All authors wrote the manuscript

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Acknowledgements

This research was supported by Ajou University and the National Research Foundation (NRF) of Korea (no. 2021R1C1C1011047, 2021R1A6A1A10044950). This research was also supported by Learning & Academic research institution for Master's-PhD students, and Postdocs (LAMP) Program of the National Research Foundation of Korea (NRF) grant funded by the Ministry of Education (No. RS-2023-00285390). JJ acknowledges the funding obtained through Academy of Medical Sciences Springboard Award (REF: SBF007\100007).

# References

M. Kim et al.

Current Applied Physics 61 (2024) 80-85

85