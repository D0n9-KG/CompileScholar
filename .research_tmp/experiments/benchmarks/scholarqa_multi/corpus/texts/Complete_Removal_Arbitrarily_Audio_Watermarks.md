# Complete removal of arbitrarily strong and arbitrarily located auto correlation artifacts in spectral domain optical coherence tomography: Demonstration of an efficient and cost effective technique

Hari Nandakumar, Satya Prasanna Mallick and Shailesh Srivastava

Department of Physics, Sri Sathya Sai Institute of Higher Learning, Prasanthi Nilayam, AP, 515134 India

# Abstract

The proof-of-principle demonstration of a simple, yet effective method of autocorrelation artifact removal for optical coherence tomography (OCT) is presented using a custom-designed parallel spectral-domain OCT (SD-OCT) instrument. Our real-time method is based on time-averaged sampling of a sinusoidal phase modulation in the reference arm. Unlike other existing methods, our technique can completely eliminate arbitrarily located, arbitrarily strong autocorrelation artifacts.

Keywords: Optical coherence tomography, Imaging systems, Tomographic imaging

# 1. Introduction

2 Optical Coherence Tomography (OCT) is a non-destructive tomographic   
3 imaging modality using non-ionizing photons. OCT can be regarded as an op-

tical analog to Ultrasound scanning commonly used in medicine, but delivers micron-level depth resolution compared to centimeter-scale resolutions with Ultrasound [1]. OCT first became popular in ophthalmology in the 1990s [2, 3] and subsequently became an established eye care tool [4]. Advances in OCT technology have seen several generations of products with improved resolutions and shorter acquisition times. Initial time-domain systems (TD-OCT) were followed by Fourier-domain systems (FD-OCT) which could be implemented without a moving reference arm. FD-OCT devices can be further of two types, spectral-domain systems (SD-OCT) or swept-source devices (SS-OCT). OCT now sees applications in cardiology, dentistry, pathology and dermatology in addition to non-destructive materials testing, historical artifact analysis and many other fields [5].

Recently, our group demonstrated a fresh approach to background subtraction in TD-OCT, which we termed the $J_{0}$ null technique [6], briefly mentioning that the technique should also work for autocorrelation removal in FD-OCT. In this paper, we apply that technique to SD-OCT and convincingly demonstrate its utility in the removal of arbitrarily strong autocorrelation artifacts. Autocorrelation artifact removal is particularly important for OCT studies on samples which have multiple layers with strong reflectivities. The traditional method of ignoring autocorrelation by increasing the reference arm reflectivity [7] would not suffice in such cases. A variety of autocorrelation removal techniques have been suggested over the years, like averaging spectra over multiple points [8], using resonant acquisition [9], using an off-axis reference beam [10], dispersion encoding [11], various phase shifting methods [12, 13, 14] and various balanced detection methods

[15, 16]. Additionally, several computational autocorrelation removal techniques [17, 18, 19] have also been proposed. Comparing these with our $J_{0}$ null method, we find that the most important advantages of our method are: its application to removal of arbitrarily strong and arbitrarily placed autocorrelation artifacts, ease of implementation, and suitability for use with low-cost parallel OCT devices.

# 2. Theoretical background

OCT theory tells us that in the frequency domain, the intensity at a particular point on the image plane is represented by the spectral interferogram [20]

$$
\begin{array}{l} I (x, y, k) = S (k) \cdot r _ {R} ^ {2} \\ + 2 S (k) r _ {R} \int_ {- \infty} ^ {\infty} r _ {s} ^ {\prime} (x, y, l _ {s}) c o s (2 k (n _ {s} l _ {s} - l _ {R})) d l _ {s} \\ + S (k) \left| \int_ {- \infty} ^ {\infty} r _ {s} ^ {\prime} (x, y, l _ {s}) e x p [ i 2 k (n _ {s} l _ {s}) ] d l _ {s} \right| ^ {2} \tag {1} \\ \end{array}
$$

where $S(k)$ is the source power spectral density, $r_{R}$ is the reference arm amplitude reflectivity, $r_{s}^{\prime}(x,y,l_{s})$ is the sample arm amplitude reflectivity density located at a path length $l_{s}$ inside the sample and $n_{s}$ is the refractive index of the sample. The first term in the right-hand side of equation (1) is the DC or reference intensity term. The second term in equation (1) is the desirable one in FD-OCT, and is used to extract $r_{s}^{\prime}(x,y,l_{s})$ using the inverse Fourier transform. The third term is the self-interference or autocorrelation term, which is undesirable and causes artifacts in OCT reconstruction. As mentioned above, various methods have been used to minimize artifacts from

the autocorrelation term. We find that the $J_{0}$ null technique presents a simple and robust way of eliminating this term when phase noise due to vibration or sample motion causes difficulties in implementing phase shifting methods.

If the reference arm undergoes a sinusoidal phase modulation, the spectral interferogram in equation (1) becomes

$$
\begin{array}{l} I _ {J _ {0}} (x, y, k) = S (k) \cdot r _ {R} ^ {2} \\ + 2 S (k) r _ {R} \int_ {- \infty} ^ {\infty} r _ {s} ^ {\prime} (x, y, l _ {s}) c o s [ 2 k (n _ {s} l _ {s} - l _ {R}) - M s i n (\omega t + \theta) ] d l _ {s} \\ + S (k) \left| \int_ {- \infty} ^ {\infty} r _ {s} ^ {\prime} (x, y, l _ {s}) e x p [ i 2 k (n _ {s} l _ {s}) ] d l _ {s} \right| ^ {2} \tag {2} \\ \end{array}
$$

Here, the Bessel function of the first kind $J_{0}$ makes its appearance by the use of the series

$$
\cos [ x \sin (\theta) ] = J _ {0} (x) + 2 \sum_ {n = 1} ^ {\infty} J _ {2 n} (x) \cos (2 n \theta) \tag {3}
$$

and

$$
\sin [ x \sin (\theta) ] = 2 \sum_ {n = 1} ^ {\infty} J _ {2 n - 1} (x) \sin [ (2 n - 1) \theta ] \tag {4}
$$

The second term of equation (2) completely vanishes when the amplitude M equals the $J_{0}$ null amplitude, provided the acquisition time of the spectral interferogram is an integral multiple of the modulation time-period [21] or is long enough to average over several cycles of the phase modulation of the reference arm. Subtracting the spectral interferogram obtained with the $J_{0}$ null from the interferogram in Eq.. 1, we obtain a spectral interferogram free

from autocorrelation artifacts. This forms the basis for our autocorrelation artifact removal scheme, which we refer to as the $J_{0}$ null technique.

![](images/dadec0b8487141940aadbc37f7c0a25575935b12765fc0f827d25173fd7c2eef.jpg)

<details>
<summary>text_image</summary>

Collimated
SLED
Cylindrical lens
Beamsplitter
Sample
collimating lens
grating
reference mirror
Focusing lens
CMOS camera
(a)
</details>

![](images/8b236495e881e2ac8f3aa54b121ebfb761c871b5a5382f09df604ffc413b4817.jpg)

<details>
<summary>text_image</summary>

(b)
</details>

Figure 1: (a) Schematic and (b) photograph of the parallel FD-OCT instrument used in our study. Labelled components: 1, SLED source; 2, steering mirror; 3, cylindrical lens; 4, cube beamsplitter; 5, sample; 6, reference mirror; 7, collimating lens; 8, steering mirror; 9, grating; 10, imaging lens; 11, camera.

# 64 3. Experimental results and conclusions

Experimental validation of our technique was done using a lab-made parallel SD-OCT [22] instrument, also known as line-field OCT [23, 24] as shown

in figure 1. A cylindrical lens with $f = 15cm$ focused the collimated beam from an SLED source (Exalos EXS210022-03) onto a $50\mu m$ vertical line on the sample. A reflection grating with 1200 lines/mm (Newport 33067FL01-360R) and a CMOS camera (QHY5L-II M) made up the spectrometer of our SD-OCT. Light from each point on the illuminated vertical line on the sample was dispersed horizontally into a spectrum, thus filling the 2D camera surface. Since the camera acquires the spectra of all the points on the illuminated vertical line on the sample in a single exposure, this parallel SD-OCT setup delivers single-shot B-scans. At low resolutions, 320x240 frame rate captures resulted in 240x80 B-scans at 148 frames per second (fps) with our system, which is 35,520 A-scans per second. At full 1280x960 resolution, 2x2 binning and FFT resampling to 2096 points resulted in 10 fps 480x320 B-scans, ie. 4,800 A-scans per second, limited by our real-time computational speed. Lateral resolution was limited by our optics, and was experimentally found to be $40\mu m$ by clearly resolving Group 3 Element 5 of a USAF target, using a translation stage for repeated B-scans across its surface. Axial

![](images/3745444f73d365f79d76246370b39c5ab762f9e9bcf6689c7c373b30d35d0afa.jpg)

<details>
<summary>line</summary>

| Piezo drive voltage (Volts) | Experimental values | Bessel function |
| --------------------------- | ------------------- | --------------- |
| 0                           | 48.0                | 48.0            |
| 1                           | 47.5                | 47.5            |
| 2                           | 46.5                | 46.5            |
| 3                           | 45.0                | 45.0            |
| 4                           | 43.0                | 43.0            |
| 5                           | 40.0                | 40.0            |
| 6                           | 35.0                | 35.0            |
| 7                           | 20.0                | 20.0            |
| 8                           | 12.0                | 12.0            |
| 9                           | 30.0                | 30.0            |
| 10                          | 38.0                | 38.0            |
| 11                          | 40.0                | 40.0            |
| 12                          | 40.0                | 40.0            |
</details>

Figure 2: Calibration curve, showing intensity of OCT reflectance at a point as a function of piezo drive voltage. The $J_{0}$ function is traced out, with a null at 7.20 V for vibration at 2.5 kHz.

resolution was limited by our choice of spectrometer bandwidth. We chose to be able to image a larger depth, 6.4mm, at the cost of $20\mu m$ per pixel axial resolution. Data acquisition and real-time signal processing was done on a desktop computer (Intel i5, 8 GB RAM) running our open source software [25] which uses OpenCV[26]. The reference arm mirror modulation was done by a piezo actuator (Steminc SMPAK155510D10) driven by a function generator (Scientific SM5070). Representative calibration data relating the reference arm vibration and piezo drive voltage is presented figure 2, wherein the null of the $J_{0}$ curve is clearly defined.

Our OCT system had a sub-50 dB dynamic range due to the $23ke^{-}$ equivalent full-well-capacity of our camera [27]. The ease in which the $J_{0}$ null subtraction can be implemented helps us to use it for repeated subtraction, which yields higher Signal to Noise Ratio (SNR) than a single subtraction, due to the effect of averaging. Figure 3 shows a stack of glass cover-slips imaged with our OCT instrument. Since the cover-slips have reflectivities similar to our reference arm, strong autocorrelation artifacts are seen in figure 3 (a). Figure 3 (b) shows autocorrelation removal by a single $J_{0}$ null subtraction. Figure 3 (c) and (d) show around 40 dB of autocorrelation removal by repeated $J_{0}$ null subtraction. The process we followed for repeated null subtraction is explained below.

In order to bring out signal buried in our noisy acquired data, we repeated adding B-scans and subtracting $J_{0}$ null frames 10 times for 10 averaged frames. This is similar to the standard lock-in detection techniques commonly used with optical choppers, wherein the signal gets sequentially added while the noise gets cancelled due to being alternately added (+1)

and subtracted $(-1)$ on account of the square wave reference. For enhanced autocorrelation removal efficiency, since the autocorrelation signal may be riding on top of noise and hence may be higher in some B-scan frames than

![](images/fce986c1ced98c212bcbd426f1d093dd367635b27a0200b984ec4cc31a0446bb.jpg)

<details>
<summary>natural_image</summary>

Grayscale topographic or topographic image showing horizontal striations with a grayscale color scale bar (no text or symbols)
</details>

(a)

![](images/30f76e7d086b4037b2d00b4c3343e075a241964e9cd49b658d01ccf134e3a1bd.jpg)

<details>
<summary>text_image</summary>

Subtracted
28
26
24
22
20
18
16
14
12
</details>

(b)

![](images/bea9ff1096856423368a4c2600c1ddd946b9398f1e6f32ce53cedd49f5ac05d0.jpg)

<details>
<summary>text_image</summary>

Repeated 2, subtraction
-10
-5
0
10
20
</details>

(c)

![](images/05bb958ed20daa8a7c417c62f8b0faede3338b774f4a99bdca5a795da1603660.jpg)

<details>
<summary>line</summary>

| Depth (meters) | Averaged B-scan | B-scan with repeated J, 0 null subtraction |
| -------------- | --------------- | ------------------------------------------ |
| 0              | -20             | -20                                        |
| 1000           | 25              | 25                                         |
| 2000           | 15              | 15                                         |
| 3000           | 10              | 10                                         |
| 4000           | 5               | 5                                          |
| 5000           | 0               | 0                                          |
| 6000           | -5              | -5                                         |
| 7000           | -10             | -10                                        |
</details>

(d)   
Figure 3: (a) 10x10 averaged B-scan of a stack of coverslips with strong autocorrelation artifacts. Lateral extent (x axis) is 6.29 mm and axial extent (y axis) is 6.6 mm. (b) Autocorrelation artifacts removed by single $J_{0}$ null subtraction. (c) Autocorrelation artifacts removed more efficiently and dynamic range improved by repeated $J_{0}$ null subtraction. (d) A representative A-scan, with some autocorrelation artifacts labelled by arrows.

in the $J_{0}$ null frames, we multiplied the B-scan frames by a fudge-factor of 0.8 before the subtraction. After each subtraction, any negative numbers were thresholded to zero before repeating the process for the next set of frames. This thresholding process allows even weak signals which are just able to raise above the noise floor in any one signal frame to be added to the final averaged result. The fudge factor is finally divided out.

We note that the $J_{0}$ null technique is insensitive to phase noise due to vibration, due to the fact that no particular phase relationship needs to be maintained between the two spectral interferogram acquisitions. This is the advantage of our technique over multi-shot phase shifting [13] techniques. Single-shot phase shifting techniques [14] would also have vibration insensitive behaviour similar to our technique, but when used with 2-D sensors, single-shot phase shifting is generally cumbersome, either having to use multiple cameras [28] or results in lowering of available pixels in the sensor [29]. Our technique results in a subtraction similar to using an optical switch in the reference arm [30], but with the advantage of low-cost implementation for single-shot B-scan imaging. Other advantages of our $J_{0}$ null technique over the use of an optical switch are, enhanced DC removal along with subtraction of reference mirror blemishes. Computational autocorrelation removal comes with caveats on the location and strength of the autocorrelation artifact, while the $J_{0}$ null technique has no such limitations. Dispersion encoding and subsequent reconstruction of a full-range signal is an attractive technique, but requires more than double the computational effort [11] needed in our method. The efficiency of autocorrelation removal with the $J_{0}$ null technique does depend on the accuracy with which the null point was determined by

the initial calibration, but is thereafter only limited by the SNR of the measurements. In conclusion, we have demonstrated a new technique for removal of autocorrelation which can work with arbitrarily strong and arbitrarily located autocorrelation artifacts. Our technique is specially suited for parallel or line-field FD-OCT devices $[22, 6, 23, 24]$ using spectrometers with 2-D sensors (cameras) which result in single-shot B-scans.

# Funding

This work was not funded by any grant. Our department had infrastructure funding from the Department of Science and Technology, Government of India (DST) -

DST FIST 2012-2017 SR/FST/PSI-172/2012.

# Acknowledgments

We convey our gratitude to Bhagawan Sri Sathya Sai Baba, the founder Chancellor of our University, who guided and inspired us in our work.

# References

[1] C. A. Puliafito, J. S. Schuman, M. R. Hee, J. G. Fujimoto, Optical coherence tomography of ocular diseases, SLACK Inc., 1996.   
[2] A. F. Fercher, K. Mengedoht, W. Werner, Eye-length measurement by interferometry with partially coherent light, Optics Letters 13 (1988) 186–188.

[3] C. K. Hitzenberger, W. Drexler, A. F. Fercher, Measurement of corneal thickness by laser Doppler interferometry., Investigative ophthalmology & visual science 33 (1992) 98–103.   
[4] J. Fujimoto, E. Swanson, The development, commercialization, and impact of optical coherence tomography, Investigative ophthalmology & visual science 57 (2016) OCT1–OCT13.   
[5] W. Drexler, J. G. Fujimoto, Optical Coherence Tomography: Technology and Applications, Second Edition, Springer International Publishing Switzerland, 2015.   
[6] H. Nandakumar, S. Parameshwaran, R. Gamini, S. Srivastava, Artifact-free robust single-shot background subtraction for optical coherence tomography, OSA Continuum 2 (2019) 1556–1564.   
[7] A. F. Fercher, C. K. Hitzenberger, G. Kamp, S. Y. El-Zaiat, Measurement of intraocular distances by backscattering spectral interferometry, Optics communications 117 (1995) 43–48.   
[8] R. K. Wang, Z. Ma, A practical approach to eliminate autocorrelation artefacts for volume-rate spectral domain optical coherence tomography, Physics in Medicine & Biology 51 (2006) 3231.   
[9] M. Shalaby, S. S. Al-Sowayan, Autocorrelation noise free optical coherence tomography using the novel concept of resonant oct (roct), Journal of the European Optical Society-Rapid Publications 12 (2016) 10.   
[10] D. Hillmann, H. Spahr, H. Sudkamp, C. Hain, L. Hinkel, G. Franke,

G. Hüttmann, Off-axis reference beam for full-field swept-source oct and holoscopy, Optics Express 25 (2017) 27770–27784.

[11] F. Köttig, P. Cimalla, M. Gärtner, E. Koch, An advanced algorithm for dispersion encoded full range frequency domain optical coherence tomography, Opt. Express 20 (2012) 24925–24948.

[12] M. V. Sarunic, M. A. Choma, C. Yang, J. A. Izatt, Instantaneous complex conjugate resolved spectral domain and swept-source oct using 3x3 fiber couplers, Opt. Express 13 (2005) 957–967.

[13] M. Wojtkowski, A. Kowalczyk, R. Leitgeb, A. F. Fercher, Full range complex spectral optical coherence tomography technique in eye imaging, Opt. Lett. 27 (2002) 1415–1417.

[14] E. Bo, S. Chen, D. Cui, S. Chen, X. Yu, Y. Luo, L. Liu, Single-camera full-range high-resolution spectral domain optical coherence tomography, Appl. Opt. 56 (2017) 470–475.

[15] W.-C. Kuo, C.-M. Lai, Y.-S. Huang, C.-Y. Chang, Y.-M. Kuo, Balanced detection for spectral domain optical coherence tomography, Optics Express 21 (2013) 19280–19291.

[16] E. Bo, X. Liu, S. Chen, X. Yu, X. Wang, L. Liu, Spectral-domain optical coherence tomography with dual-balanced detection for auto-correlation artifacts reduction, Optics Express 23 (2015) 28050–28058.

[17] A. Ozcan, M. J. F. Digonnet, G. S. Kino, Minimum-phase-function-based processing in frequency-domain optical coherence tomography systems, J. Opt. Soc. Am. A 23 (2006) 1669–1677.

[18] B. J. Davis, T. S. Ralston, D. L. Marks, S. A. Boppart, P. S. Carney, Autocorrelation artifacts in optical coherence tomography and interferometric synthetic aperture microscopy, Opt. Lett. 32 (2007) 1441–1443.   
[19] Y. C. S. Hon Luen Seck, Ying Zhang, Autocorrelation noise removal for optical coherence tomography by sparse filter design, Journal of Biomedical Optics 17 (2012) 1 - 7 - 7.   
[20] L. V. Wang, H.-i. Wu, Biomedical optics: principles and imaging, John Wiley & Sons, 2012.   
[21] V. Sudarshanam, K. Srinivasan, Universal dynamic phase-calibration technique for fiber-optic interferometric sensors and phase modulators, Optics Letters 14 (1989) 1287–1289.   
[22] B. Grajciar, M. Pircher, A. F. Fercher, R. A. Leitgeb, Parallel fourier domain optical coherence tomography for in vivo measurement of the human eye, Optics Express 13 (2005) 1131–1137.   
[23] S. Lawman, Y. Dong, B. M. Williams, V. Romano, S. Kaye, S. P. Harding, C. Willoughby, Y.-C. Shen, Y. Zheng, High resolution corneal and single pulse imaging with line field spectral domain optical coherence tomography, Opt. Express 24 (2016) 12395–12405.   
[24] A. Kazaili, S. Lawman, B. Geraghty, A. Eliasy, Y. Zheng, Y. Shen, R. Akhtar, Line-field optical coherence tomography as a tool for in vitro characterization of corneal biomechanics under physiological pressures, Scientific reports 9 (2019) 6321.

[25] H. Nandakumar, FDOCT, https://github.com/hn-88/FDOCT, 2019.   
[26] G. Bradski, A. Kaehler, Opencv, Dr. Dobbs journal of software tools (2000).   
[27] H. Nandakumar, S. Srivastava, Data and code for analyzing performance of qhy cmos cameras, Preprints 2018 (2018) 2018100179.   
[28] Y. Zhu, W. Gao, Single-shot wavelength-independent phase-shifting method for full-field optical coherence tomography, Appl. Opt. 58 (2019) 806–813.   
[29] H. M. Subhash, Full-field and single-shot full-field optical coherence tomography: A novel technique for biomedical imaging applications, Advances in Optical Technologies vol. 2012 (2012) Article ID 435408.   
[30] J. Ai, L. V. Wang, Spectral-domain optical coherence tomography: Removal of autocorrelation using an optical switch, Applied Physics Letters 88 (2006) 111115.