# Dynamic contrast in scanning microscopic OCT

Michael Münter $^{*1}$ , Malte vom Endt $^{2}$ , Mario Pieper $^{3,4}$ , Malte Casper $^{5}$ , Martin Ahrens $^{1,4}$ , Tabea Kohlfaerber $^{2}$ , Ramtin Rahmanzadeh $^{1}$ , Peter König $^{3,4}$ , Gereon Hüttmann $^{1,2,4}$ and Hinnerk Schulz-Hildebrandt $^{1,2,4}$

$^{1}$ Institute of Biomedical Optics, Universität zu Lübeck, 23552 Lübeck, Germany

$^{2}$ Medizinisches Laserzentrum Lübeck GmbH, 23552 Lübeck, Germany

$^{3}$ Institute of Anatomy, Universität zu Lübeck, 23552 Lübeck, Germany

$^{4}$ Airway Research Center North (ARCN), Member of the German Center of Lung Research (DZL), 35392 Gießen, Germany

$^{5}$ Laboratory for Functional Optical Imaging, Department of Biomedical Engineering, Columbia University, New York, NY 10027, USA

\*Corresponding author: mic.muenter@uni-luebeck.de

# ABSTRACT

While optical coherence tomography (OCT) provides a resolution down to 1 $\mu$ m it has difficulties to visualize cellular structures due to a lack of scattering contrast. By evaluating signal fluctuations, a significant contrast enhancement was demonstrated using time-domain full-field OCT (FF-OCT), which makes cellular and subcellular structures visible. The putative cause of the dynamic OCT signal is ATP-dependent motion of cellular structures in a sub-micrometer range, which provides histology-like contrast. Here we demonstrate dynamic contrast with a scanning frequency-domain OCT (FD-OCT). Given the inherent sectional imaging geometry, scanning FD-OCT provides depth-resolved images across tissue layers, a perspective known from histopathology, much faster and more efficiently than FF-OCT. Both, shorter acquisition times and tomographic depth-sectioning reduce the sensitivity of dynamic contrast for bulk tissue motion artifacts and simplify their correction in post-processing. The implementation of dynamic contrast makes microscopic FD-OCT a promising tool for histological analysis of unstained tissues.

# INTRODUCTION

OCT is an imaging technique, which is based on the interferometric measurement of backscattered light. Providing fast, high-resolution sectional images of biological tissue, OCT has become a valuable tool in different clinical fields like ophthalmology and dermatology $^{1}$ . One especially useful characteristic of OCT is the decoupling of the axial resolution and lateral resolution $^{2}$ . For resolving cells and even subcellular structures, it is necessary to increase the resolution of OCT from typically 10 - 15 $\mu$ m to 1 $\mu$ m. Optical coherence microscopy (OCM) $^{3}$ , micro-optical coherence tomography ( $\mu$ OCT) $^{4}$ , and microscopic optical coherence tomography (mOCT) $^{5}$ typically reach a micron resolution comparable to other microscopic techniques like confocal or nonlinear microscopy $^{6}$ . The use of ultra-broadband light from coupled SLDs, Ti-sapphire lasers or supercontinuum light sources increases systems axial resolution to 1-2 $\mu$ m. Using a high NA microscope objective, a lateral resolution of 1 $\mu$ m can be achieved. Loss of lateral resolution outside the focal plane can be compensated by Bessel beams $^{7,8}$ , introducing aberrations $^{9}$ or a numerical correction of the defocus in post-processing $^{10}$ . In OCT, coherent noise (speckle) reduces contrast $^{11,12}$ and often makes it impossible to differentiate individual cells based on their scattering properties. Recently, a novel approach, to analyze data of time domain full-field OCT (FF-OCT) was demonstrated and termed as dynamic OCT $^{13}$ . This method exploits the dynamic scattering changes of metabolically active cellular structures to enhance contrast. FF-OCT illuminates the object over a large area with incoherent light. A horizontal sectional (en-face) image is created by interference between the light scattered by the sample and the light reflected from the reference mirror. To create a tomographic image, it is necessary to move the microscope objective or the sample in axial direction. Dynamic contrast is obtained

by analyzing temporal intensity fluctuation in each voxel. Correlation functions or power spectra of the signal fluctuations are evaluated over a few seconds at a typical sampling frequency of $100 \, Hz^{13, 14}$ . This technique detects motion of cellular structures with nanometer sensitivity and $1 \, \mu m$ spatial resolution delivering images which appear similar to conventional histological techniques. Dynamic FF-OCT acquires en-face images of unfixed biological tissue from a depth of up to $100 \, \mu m$ with rich cellular details. Disadvantages of this technique are a low penetration depth in scattering tissues, the limited use of scattered photons from only a $1 \, \mu m$ thick sample plane, the high sensitivity to axial movements and the incapability to quickly acquire large volumes. In contrast, scanned frequency-domain OCT (FD-OCT) provides real tomographic imaging simultaneous over a certain depth range and uses all photons. Additionally, it has more degrees of freedom in defining the imaged region. However scanned OCT typically lacks lateral phase stability and suffers from a shallow depth of focus, when using high-NA objective lenses, e.g. at $1 \, \mu m$ focus diameter the confocal parameter is only $17 \, \mu m$ . Outside this range, lateral resolution decreases and dynamic contrast may be annihilated. This work demonstrates that scanned FD-OCT produces surprisingly rich dynamic contrasted B-scans and volumetric images of freshly excised murine tissue samples - tongue and liver, despite high NA.

# METHODS AND PROCEDURES

![](images/136a3b57ae2a232fb940b6f6c1fdb5e1501152de2539d466b50cb9f2a657d865.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    BS["BS"] --> FB["FB"]
    FB --> FC["FC"]
    FC --> C["C"]
    C --> A["A"]
    C --> DC["DC"]
    DC --> RR["RR"]
    PC["PC"] --> S["S"]
    S --> DAQ["DAQ"]
    DAQ --> G["G"]
    G --> TS["TS"]
    TS --> O["O"]
    O --> Z1["Z1"]
```
</details>

![](images/f50c1efdbac35f695a14d5868f244d11679b0b8fc53f41aacc74a05966931724.jpg)

<details>
<summary>line</summary>

| Frequency [Hz] | Intensity [a.u.] |
| -------------- | ---------------- |
| 0.5            | ~1.0             |
| 5              | ~0.1             |
| 10             | ~0.05            |
| 15             | ~0.03            |
| 20             | ~0.02            |
| 25             | ~0.01            |
</details>

Figure 1. (a) Schematic of the mOCT setup. LS: light source, FB: filter box, FC: 50/50 fiber coupler, C: collimators, G: xy galvanometer mirror scanner, TS: scan lenses, O: microscope objective, DC: dispersion compensation, A: aperture, RR: retroreflector, DAQ: data acquisition device, S: spectrometer; PC: computer for data acquisition and scanning control. Nanometer cellular motion is detected by intensity fluctuations (b) The dOCT image is generated by t color-coding of three frequency bands after Fourier transforming the temporal signal fluctuations in each pixel of the B-scan image stack

The mOCT setup illustrated in Fig. 1, uses a supercontinuum light source (SuperK EXTREME-EXW-OCT, NKT Holding, Denmark) in combination with spectrometer covering a spectral range from 550 to $950~\mathrm{nm}$ in order to achieve $1\mu \mathrm{m}$ axial resolution. Emission of the supercontinuum light source was bandpass filtered (SuperK SPLIT, NKT Holding, Denmark) and injected to a fiber-based Michelson interferometer. A broadband 50/50 fiber coupler (TW630R5A2, Thorlabs Inc., U.S.) splits the light into the sample and reference arm. Light in the sample arm is collimated (60FC-L-4-M25-02, Schäfter + Kirchhoff GmbH, Germany) and scanned via an x/y galvanometer scanner module (6210H, Cambridge Technology, U.S.) and an achromatic telescope (SL50-CLS2 and TL200-CLS2, Thorlabs Inc., U.S.) to the back focal plane of a $10\mathrm{x} / 0.3$ NA microscope objective (HCX APO L $10\mathrm{x} / 0.3$ WUVI, Leica Microsystems, Germany). Radiant flux at the sample was $40~\mathrm{mW}$ . Light in the reference arm was collimated (60FC-L-4-M25-02, Schäfter + Kirchhoff GmbH, Germany), attenuated by a variable neutral density filter (NDC-50C-2M, Thorlabs Inc., U.S.), and matched in dispersion to the light of the sample arm by a $14~\mathrm{mm}$ thick piece of SF57 glass substrate (Casix Inc., China). Reference light from a retroreflector (PS975M-B, Thorlabs Inc., U.S.) and sample light were brought to interference in the custom-designed spectrometer (Thorlabs GmbH, Germany). Two different cameras (SprintSpL4096 $140\mathrm{km}$ , Basler, Germany and OctoPlus CL, Teledyne, e2v, Canada) were used for achieving A-scan rates of up to $127\mathrm{kHz}$ and $248\mathrm{kHz}$ , respectively. Synchronization of the scanner, spectrometer and data acquisition software was realized using a USB data acquisition device (NI USB-6251, National Instruments, U.S.).

For volumetric imaging, at each line 100 B-scans with 500 A-scans each were taken at 30 kHz A-scan rate with the Sprint SpL4096 camera, which results in an effective B-scan rate of 46 Hz. Adjacent A-Scans were axial aligned using the reflection of the glass plate's surface to compensate for sample motion and image field curvature. Volumes were recorded by stacking a series of B-scans in y-direction. With the OctoPlus CL 150 B-scans with 500 A-scan, each were sampled at 100 kHz A-scan rate. With an effective B-scan rate of 108 Hz and a total recording time of 1.39 s, the frequency of signal fluctuations was measured between 0 to 54 Hz. Raw data from the spectrometer was Hann-windowed and Fourier transformed to obtain the OCT images. Residual dispersion was numerically compensated using a 5th order polynomial for the spectral phase error correction. Polynomial coefficients were determined by optimizing image quality, which was measured by the Shannon entropy $^{15, 16}$ . Spectral processing and dynamic contrasting of the data were performed in MATLAB (MATLAB R2019b, The MathWorks, Inc., U.S.). At each voxel, the temporal variations of the absolute value of the OCT signal was evaluated. Similar to the data evaluation in dynamic FF-OCT, the time series was Fourier transformed and the integral amplitude was calculated in three frequency bands (Fig. 1b). For each pixel the three resulting values were color-coded in an RGB image, representing different time scales of motion activity. Blue represents slow motion frequencies (0-0.5 Hz), green medium motion frequencies (0.5-5 Hz), and red fast motion (5-25 Hz). For the representation of the images, the color channels were scaled logarithmically. Image contrast was enhanced using contrast-limited adaptive histogram equalization $^{17}$ . Brightness outside the focus was adjusted to the peak intensity in the focus. Finally, a 3x3 median filter was applied to each channel. Reference measurements were acquired with a commercially available FF-OCT device (LightCT, LLTech Inc., France), which was designed for real-time optical biopsy using dynamic OCT contrast. LightCT uses a Linnik interferometer and a temporally and spatially incoherent light source $^{18}$ . Using a spectral width of 100 nm at a central wavelength of 565 nm and a 0.3 NA water immersion objective, a nearly 1 $\mu$ m isotropic resolution is achieved. Our frequency bands for color-coding were chosen directly to match those of the LightCT. For immobilizing of the sample, freshly excised tongue and liver tissue of C57BL/6 mice were placed in Ringer's solution in the specially designed sample holder of the LightCT. Tissues were imaged by slightly pressing the tissue surface against the quartz cover glass plate to which the objective was coupled using silicon immersion oil.

# RESULTS

Despite the sufficient resolution, mOCT and LightCT OCT images of liver tissue showed only weak contrast. The murine liver shown in Fig. 2 was imaged over a FOV of 550x270x285 $\mu$ m (zxy). Fine cellular details of the tissue are difficult to identify without dynamic contrasting in averaged en-face OCT image planes shown in Fig. 2a and c. Brighter details might indicate extracellular matrix structures and spherical voids cell nuclei. However, images with the dynamic contrast clearly show cellular structures (Fig. 2b and d). Cell nuclei and cell borders can be distinguished with high contrast. Cell nuclei appear with increased movement activity in red while the extracellular matrix in blue indicating slow movement. Dynamic mOCT and LightCT images depict the tissue structures in similar color, however, intensity and composition of color/motion components are slightly different. A volumetric mOCT image cropped in z recorded is given in Fig. 2e and Visualization 1 (available on request). Similar results were achieved imaging the bottom of murine tongue. The murine tongue shown in Fig. 3 was imaged over a FOV of 660x500 $\mu$ m (zx). This part of the tongue consists of a cornified stratified squamous epithelium and a subepithelial layer of connective tissue followed by skeletal muscle cells (Fig. 3a). The average of 150 mOCT B-Scans does only shows a faint image of the tissue structures (Fig. 3b). On the right side of the OCT image four tissue layers can only be guessed. On the left side, a layered structure is not visible. Using dynamic contrast, histologically relevant layers become clearly visible in the whole field of view (Fig. 3c). Below the glass plate, the cornified layer, the layer of the squamous epithelium with the transition to the granular and spinous layer is contrasted as a purple and a green-purple layer, respectively. Deeper, the basal layer, which is characterized by yellow-colored cell nuclei, is displayed. Since the optical focus was placed in this region the basal layer is imaged at the highest resolution and structures even within the nuclei became discernible. Below, the lamina propria is visible as a purple band again

![](images/000ece92198dffe2b3f7a6db96b6442889f56bb8687b6c13becdf9145d531945.jpg)

<details>
<summary>text_image</summary>

LightCT-OCT
LightCT-dynamic
mOCT
dmOCT
(c)
(d)
(e)
0-0.5 Hz
5-25 Hz
0.5-5 Hz
x
y
z
y
x
y
x
y
0-0.5 Hz
</details>

Figure 2. (a) LightCT en-face OCT image of murine liver; (b) LightCT dynamic OCT en-face image with corresponding regions marked with (\*) (c) Averaged en-face image from the scanned mOCT, which was acquired at the same region; (d) in the corresponding dynamic mOCT image hepatocytes become visible with nuclei (e) Cropped volume representation; size: 135x270x285 $\mu$ m (zxy) (Visualization available on request); scalebar: 100 $\mu$ m

followed by the green-colored skeletal muscle fibers. Shifting the focus below the basal layer increases the visibility of the muscle layer (Fig. 3d). Dynamic contrast mOCT images show an excellent match to structures in HE stained histological sections (Fig. 3a, c-d). The dynamic contrast images are completely free of speckle noise and have remarkably high contrast. Cellular structures are discernible even well beyond the focal plane, which are typically invisible in averaged OCT images even when exactly focused. The high contrast permits a reliable segmentation of layers and even individual basal cells.

![](images/9c2a35bfd42560271cb57e02590e1550942241a893e1ffae84d9d1ec3b6d9331.jpg)  
Figure 3. HE stained histology of the imaged sample at a different location (I) Cornified layer, (II) Granular & spinous layers, (III) Basal layer, (IV) Lamina propria, (V) muscle, (VI) Glass plate (b) OCT image of mouse tongue; lamina propria (IV) can be identified by brighter contrast (c) Corresponding dynamic contrast mOCT image with focus in basal cell layer; (I-V) and even cell nuclei (\*) are visible (d) Dynamic contrast mOCT image with focus in the lamina propria; cropped image with size of 460x500 µm (zx); scalebar: 100 µm

# DISCUSSION

These results demonstrate, that implementing dynamic contrast in a scanning mOCT is possible and yields dramatically increased contrast that allows identification of individual epithelial cells. This is somewhat surprising as dynamic contrast is inferred from motion on nanometer scale and scanning can lead to additional phase noise potentially overcasting phase fluctuation caused in the sample. While the concept of dynamic contrast was developed for FF-OCT, here we show, it can also be used with scanning mOCT. This offers several advantages. First, it provides inherently sectional images across the tissue layers, which show several tissue layers at a glance as known from conventional histological sections that are routinely used in pathology. Reconstruction of a B-scan from a series of en-face FF-OCT images is time-consuming and image quality can easily be degraded by small sample motions. Second, acquiring a sectional image plane reduces the sensitivity to axial sample motion, which has the most severe influence on dynamic contrast. While in FF-OCT systems axial sample motion is difficult to detect and correct $^{19}$ , in sectional scans each A-scan can be individually corrected by multiplying an appropriate phase factor. Third, the parallel measurement of the complete depth information in FD-OCT potentially increases speed for volumetric imaging. Dynamic contrast OCT needs to measure each pixel over a few seconds with a temporal resolution of about 10 ms. This limits the imaging speed. FF-OCT, which currently uses a 2 Megapixel camera is able to image about 1 million volume elements per second. At 1 μm resolution this converts into an imaging time of 1 s/mm2 for one en-face image. For volumetric imaging over 300 μm depth, time is increased to 300 s or 5 min per mm2. Scanned OCT with B-scan based evaluation of the signal fluctuation does not increase imaging speed for volume acquisition. However, FD-OCT with MHz A-scan rate could acquire a whole volume within 10 ms $^{20}$ and complete volumetric dynamic contrast OCT imaging within a few seconds. Currently, our imaging speed is limited by three components â€' galvo scanner, line scan camera and relative intensity noise (RIN) of supercontinuum light sources. No negative effects of RIN on image quality could be found yet and recently we were able to demonstrate mOCT imaging at 1 μm isotropic resolution with 600 kHz A-scan rate and nearly 12 volumes per second $^{21}$ . However, fast volumetric scanning is still a challenge. Linear, single direction scanning introduces considerably overhead, which can only be compensated by bidirectional or resonant scanning. The influence of fast scanning on dynamic contrast is currently under investigation. Forth, dynamic contrast with scanning OCT is easier to implement in an endoscopic system. Since technologies using scanning OCT imaging in rigid and flexible endoscopes have already been developed $^{22}$ , dynamic contrasting, as an imaging routine, could equip such devices with the additional contrast needed to facilitate non-invasive, in vivo optical histology. In conclusion, the analysis of the signal dynamics of cellular resolution scanned FD-OCT visualizes epithelial tissues comparable to conventional histology without the need for tissue processing. We found that scanning mirrors do not interfere with the dynamic signal extraction. By evaluating the fluctuation of the OCT signal, speckle-free B-scan and volumetric imaging with high contrast and resolution is achieved. Spectral analysis discriminates and visualizes important cellular components like nuclei and cytoplasm and allows the identification of different tissue components such as epithelium, connective tissue and muscle. We expect that scanning FD-OCT is more robust against sample motion than FF-OCT and can increase imaging speed by two orders of magnitude. Given these assets, dynamic contrast OCT might even fulfill the 20 years old promise of optical biopsies by OCT $^{2}$ .

# FUNDING

This research was funded by the European Union project within Interreg Deutschland-Denmark from the European Regional Development Fund (ERDF) in the project CELLTOM and German Ministry of Research, Innovation and Science, Helmholtz Center Munich of Health and Environment DZL-ARCN (82DZL001A2) and German Research Foundation (DFG, RA 1771/4-1, EXC 2167).

# REFERENCES

1. W. Drexler, U. Morgner, R. K. Ghanta, F. X. Kärtner, J. S. Schuman, and J. G. Fujimoto, "Ultrahigh-resolution ophthalmic optical coherence tomography," Nat. Med. 7, 502-506 (2001). https://doi.org/10.1038/86589.

2. J. G. Fujimoto, C. Pitris, S. A. Boppart, and M. E. Brezinski, "Optical coherence tomography: An emerging technology for biomedical imaging and optical biopsy," Neoplasia 2, 9-25 (2000). https://doi.org/10.1038/sj.neo.7900071.   
3. J. A. Izatt, E. A. Swanson, J. G. Fujimoto, M. R. Hee, and G. M. Owen, "Optical coherence microscopy in scattering media," Opt. Lett. 19, 590 (1994). https://doi.org/10.1364/ol.19.000590.   
4. L. Liu, J. A. Gardecki, S. K. Nadkarni, J. D. Toussaint, Y. Yagi, B. E. Bouma, and G. J. Tearney, "Imaging the subcellular structure of human coronary atherosclerosis using micro-optical coherence tomography," Nat. Med. 17, 1010-1014 (2011). https://doi.org/10.1038/nm.2409.   
5. M. Pieper, H. Schulz-Hildebrandt, M. A. Mall, G. Hüttmann, and P. König, "Intravital microscopic optical coherence tomography imaging to assess mucus mobilizing interventions for muco-obstructive lung disease in mice," Am. J. Physiol. Cell. Mol. Physiol. (2020). [in print] https://doi.org/10.1152/ajplung.00287.2019.   
6. G. Cox and C. J. R. Sheppard, "Practical Limits of Resolution in Confocal and Non-Linear Microscopy," Microsc. Res. Tech. (2004). https://doi.org/10.1002/jemt.10423   
7. R. A. Leitgeb, M. Villiger, A. H. Bachmann, L. Steinmann, and T. Lasser, "Extended focus depth for Fourier domain optical coherence microscopy," Opt. Lett. 31, 2450 (2006). https://doi.org/10.1364/ol.31.002450.   
8. A. Curatolo, P. R. T. Munro, D. Lorenser, P. Sreekumar, C. C. Singe, B. F. Kennedy, and D. D. Sampson, "Quantifying the influence of Bessel beams on image quality in optical coherence tomography," Sci. Rep. 6, (2016). https://doi.org/10.1038/srep23483   
9. H. Schulz-Hildebrandt, M. Pieper, C. Stehmar, M. Ahrens, C. Idel, B. Wollenberg, P. König, and G. Hüttmann, "Novel endoscope with increased depth of field for imaging human nasal tissue by microscopic optical coherence tomography," Biomed. Opt. Express 9, 636 (2018). https://doi.org/10.1364/boe.9.000636   
10. T. S. Ralston, D. L. Marks, P. S. Carney, and S. A. Boppart, "Interferometric synthetic aperture microscopy," Nat. Phys. 3, 129-134 (2007). https://doi.org/10.1007/978-3-319-06419-2\_31   
11. A. Curatolo, B. F. Kennedy, D. D. Sampson, and T. R. Hillman, "Speckle in optical coherence tomography," Adv. Biophotonics Tissue Opt. Sect. 4, 211-277 (2016). https://doi.org/10.1117/1.429925   
12. O. Liba, M. D. Lew, E. D. Sorelle, R. Dutta, D. Sen, D. M. Moshfeghi, S. Chu, and A. De La Zerda, "Speckle-modulating optical coherence tomography in living mice and humans," Nat. Commun. 8, 15845 (2017). https://doi.org/10.1038/ncomms15845   
13. C. Apelian, F. Harms, O. Thouvenin, and A. C. Boccara, "Dynamic full field optical coherence tomography: subcellular metabolic contrast revealed in tissues by interferometric signals temporal analysis," Biomed. Opt. Express 7, 1511 (2016). https://doi.org/10.1364/boe.7.001511   
14. O. Thouvenin, C. Apelian, A. Nahas, M. Fink, and C. Boccara, "Full-field optical coherence tomography as a diagnosis tool: Recent progress with multimodal imaging," Appl. Sci. 7, 236 (2017). https://doi.org/10.3390/app7030236   
15. M. Wojtkowski, V. J. Srinivasan, T. H. Ko, J. G. Fujimoto, A. Kowalczyk, and J. S. Duker, "Ultrahigh-resolution, high-speed, Fourier domain optical coherence tomography and methods for dispersion compensation," Opt. Express 12, 2404 (2004). https://doi.org/10.1364/opex.12.002404   
16. H. Schulz-Hildebrandt, M. Münter, M. Ahrens, H. Spahr, D. Hillmann, P. König and G. Hüttmann, "Coherence and diffraction limited resolution in microscopic OCT by a unified approach for the correction of dispersion and aberrations," in Proc. SPIE 10591 (2018), p. 48. https://doi.org/10.1117/12.2303755   
17. K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in Graphics Gems, P. S. Heckbert, ed. (Academic Press, 1994), pp. 474-485. https://doi.org/10.1016/B978-0-12-336156-1.50061-6   
18. L. Van Manen, P. L. Stegehuis, A. Faria-Sarasqueta, L. M. De Haan, J. Eggermont, B. A. Bonsing, H. Morreau, B. P. F. Lelieveldt, C. J. H. Van De Velde, A. L. Vahrmeijer, J. Dijkstra, and J. S. D. Mieog, "Validation of full-field optical coherence tomography in distinguishing malignant and benign tissue in resected pancreatic cancer specimens," PLoS One 12, 1-15 (2017). https://doi.org/10.1371/journal.pone.0175862

19. J. Scholler, "Motion artifact removal and signal enhancement to achieve in vivo dynamic full field OCT," Opt. Express 27, 19562 (2019). https://doi.org/10.1364/oe.27.019562   
20. W. Wieser, B. R. Biedermann, T. Klein, C. M. Eigenwillig, and R. Huber, "Multi-Megahertz OCT: High quality 3D imaging at 20 million A-scans and 45 GVoxels per second," Opt. Express 18, 14685 (2010). https://doi.org/10.1364/oe.18.014685   
21. M. Münter, H. Schulz-Hildebrandt, M. Pieper, P. König, and G. Hüttmann, "4D microscopic optical coherence tomography imaging of ex vivo mucus transport," Proc. SPIE. 11078 (2019), p. 36. https://doi.org/10.1117/12.2527138   
22. H. Schulz-Hildebrandt, T. Pfeiffer, T. Eixmann, S. Lohmann, M. Ahrens, J. Rehra, W. Draxinger, P. König, R. Huber, and G. Hüttmann, "High-speed fiber scanning endoscope for volumetric multi-megahertz optical coherence tomography," Opt. Lett. 43, 4386 (2018). https://doi.org/10.1364/ol.43.004386