Article

# A Generic and Effective System Dispersion Compensation Method: Development and Validation in Visible-Light OCT

Jiarui Wang, Chao Xu, Shaodi Zhu, Defu Chen, Haixia Qiu, Alexander K. N. Lam, Christopher K. S. Leung and Wu Yuan

Special Issue

Advanced Techniques in Biomedical Optical Imaging

Edited by

Dr. Haigang Ma, Dr. Yujiao Shi and Dr. Yue Zhao

hw

photonics

IMPACT FACTOR 2.4 CITESCORE 2.3

MDPI

https://doi.org/10.3390/photonics10080892

Article

# A Generic and Effective System Dispersion Compensation Method: Development and Validation in Visible-Light OCT

Jiarui Wang $^{1,\dagger,\ddagger}$ , Chao Xu $^{1,\ddagger}$ , Shaodi Zhu $^{1}$ , Defu Chen $^{2,*}$ , Haixia Qiu $^{3}$ , Alexander K. N. Lam $^{4}$ , Christopher K. S. Leung $^{4}$ and Wu Yuan $^{1,5,*}$

![](dt=2026-03-14/ht=06/62e1e418cc85fbd019ffdc354050765ba71107f734ccb5c0f9077853f8a2d0bb.jpg)

check for updates

Citation: Wang, J.; Xu, C.; Zhu, S.

Chen, D.; Qiu, H.; Lam, A.K.N.;

Leung, C.K.S.; Yuan, W. A Generic and Effective System Dispersion

Compensation Method:

Development and Validation in

Visible-Light OCT. Photonics 2023, 10,

892. https://doi.org/10.3390/

photonics10080892

Received: 26 April 2023

Revised: 11 July 2023

Accepted: 28 July 2023

Published: 2 August 2023

![](dt=2026-03-14/ht=06/a28b8aad4813b2c0d0e8cf65c11abed3b58de5fefe7113020f0818e44c866d98.jpg)

Copyright: © 2023 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).

Abstract: Compared with optical coherence tomography (OCT) in the near-infrared domain, the visible-light OCT (vis-OCT) system affords a higher axial resolution for discerning subtle pathological changes associated with early diseases. However, the significant material dispersion at the visible-light range leads to a severe problem for dispersion management in vis-OCT systems, which results in a compromised axial resolution.

While dispersion compensators (such as prism pairs) are commonly used, a digital method is still highly desirable and has been widely used to compensate for the residual dispersion imbalance between the reference and sample arms in an OCT system. In this paper, we develop a generic approach to effectively compensate for the system dispersion, especially the higher-order dispersion in the vis-OCT system, by using a single arbitrary measurement of the mirror-reflection (SAMMR) method and its resulting phase information.

Compared with the previous methods, including the method based on the Taylor series iterative fitting and differential method, the proposed method does not need to extract the dispersion coefficients or use the metric functions and affords a better performance for axial resolution and the signal-to-noise ratio in vis-OCT systems. Its effectiveness is further validated in an OCT system operating in the near-infrared domain.

Keywords: visible light; optical coherence tomography; axial resolution; dispersion compensation; material dispersion

# 1. Introduction

Optical coherence tomography (OCT) is an emerging technology capable of the high-resolution volumetric imaging of microanatomy in vivo [1]. At present, the spectral-domain OCT (SD-OCT) system has become mainstream because of its faster A-line scan rate of over $100\mathrm{kHz}$ [2,3] and higher sensitivity compared to traditional time-domain OCT (TD-OCT) systems [4]. The high imaging speed of the SD-OCT system makes it possible to provide 3D diagnostic information of tissues for clinical use.

A prime example of this is the widespread use of the SD-OCT system as a retinal imaging modality in clinical settings [5,6]. Additionally, the SD-OCT system has also shown great potential in assessing structural and microangiographic changes in skin lesions [7,8]. Furthermore, when combined with a miniature endoscope, the SD-OCT system allows for minimally invasive intravascular imaging in coronary arteries and intraluminal imaging in internal organs, such as the gastrointestinal tract and airways [9-13].

hV

photonics

MDPI

Photonics 2023, 10, 892. https://doi.org/10.3390/photonics10080892

https://www.mdpi.com/journal/photonics

To achieve a high resolution in SD-OCT systems, it is necessary to take into account several key factors, including the central wavelength and spectral bandwidth of the laser source, the source spectral shape, the additional dispersion induced by the tissue, and the mismatch of optical dispersions between two interferometric arms [14]. A shorter central wavelength and a broader effective spectrum help achieve a higher axial resolution [11,15-18]. For example, compared with near-infrared SD-OCT systems, the vis-OCT system provides a finer axial resolution of less than $2\mu \mathrm{m}$ .

The extremely high axial resolution of $2\mathrm{nm}$ with a laser plasma soft X-ray source has also been demonstrated [19]. Usually, a non-ideal Gaussian spectrum introduces side lobes in the point spread function (PSF) and degrades the axial resolution. Apodization or digital spectral shaping significantly suppresses the side lobes [20]. As for the additional depth-dependent dispersion induced by the tissue, it is usually negligible due to the shallow imaging depth in most SD-OCT systems when imaging superficial tissue surfaces.

However, it is worthwhile noting that depth-dependent dispersion compensation is still needed in some applications of long imaging depth, such as anterior segment imaging [21]. In contrast, the mismatch of material dispersions between the reference and sample arms leads to a severe problem in dispersion management in the SD-OCT system and results in the broadening and asymmetric pulse distortion of PSF, thus degrading the axial resolution and image quality [22]. This problem is even more severe in the vis-OCT system due to the significant material dispersion in the visible-light range [23].

While dispersion compensators, such as prism pairs, are widely used in vis-OCT systems to match the material dispersions between the reference and sample arms, a perfect match of dispersions with optics remains challenging. Therefore, a digital method is highly desirable to accurately compensate for the residual dispersion imbalance to achieve an optimal resolution in the vis-OCT system.

Several methods have been previously proposed for dispersion compensation in the SD-OCT system. One method is based on the iterative fitting of the Taylor series expansion of propagation constants [24,25] which is referred to as the Taylor series iterative fitting (TSIF) method. By maximizing the sharpness metric function of the PSF or sample image, the second- and third-order dispersion coefficients or even higher order-dispersions can be extracted and thus compensated [24,26]. Combined with short-time Fourier transform (STFT), the dispersion can also be compensated by minimizing the ridge variance (variance of the peak position of PSFs calculated from STFT at different center wavelengths) [27].

There are some other dispersion compensation methods not requiring the pre-measurement of PSF. For example, in a method proposed by Cense et al., the center of the fovea (fovealumbo) was treated as a reflector and the TSIF method was utilized to extract the dispersion coefficients through a polynomial fitting procedure to compensate for the dispersion in retinal images [25]. To compensate for the dispersion of the sample, Kho et al.

developed a method based on a sub-band, sub-image correlation algorithm to estimate the spatially dependent dispersion along the imaging depth and lateral direction without using the pre-measured PSF [23]. These methods were able to compensate for both the dispersion caused
by the system and the sample itself and required no pre-measurements. However, these methods practically do not consider the higher-order dispersion.

To compensate for the higher-order dispersion and reduce the computational complexity, phase-based methods that require the measurements of spectra at different optical path differences (OPDs) of the interferometer were proposed [28], though the measurements could also be used for correcting wavenumber non-linearity [29,30].

The symmetrical measurements of the mirror-reflection (TSMMR) method or dispersion compensation with the symmetric phase measurement (DCSPM) method requires measuring the PSFs of mirror reflections at two symmetrical locations relative to the zero OPD position in the interferometer [28,31]. The measured result can be used to calibrate k-linearization and dispersion simultaneously. However, the movement of the mirror in the reference arm might introduce extra misalignments.

Recently, various deep learning-based methods have been proposed for automated dispersion compensation in OCT systems [32,33].

Photonics 2023, 10, 892

2 of 14

In this paper, we propose a generic and robust phase-based method to adequately compensate for the residual material dispersion in a spectral-domain vis-OCT system with a broadband light source. The algorithm for the linearization of the wavenumber is not within the scope of this work. In this method, only a single arbitrary measurement of mirror reflection (SAMMR) is required to extract the phase delay caused by the material dispersion imbalance, involving no mirror movement and capable of the effective compensation of severe higher-order dispersions in the visible-light range.

Compared with the previous methods, i.e., TSIF and TSMMR, the proposed method demonstrates a better performance for axial resolution and signal-to-noise ratio (SNR) in the vis-OCT system. Its robustness is validated by accurately compensating for additional material and higher-order dispersions in the visible-light range. We also discuss the artificial peaks induced by the phase calibration and its elimination by our method. In addition, the effectiveness of our method is demonstrated in an $800\mathrm{nm}$ SD-OCT system.

However, the proposed method only compensates for dispersions caused by the instrument itself and requires the measurement of mirror reflection before taking the images.

# 2. Experiment and Theory

# 2.1. Vis-OCT System and Experiments

As illustrated in Figure 1, a vis-OCT system was built by adopting a spectral-domain design. In this system, a 90:10 fiber coupler was used to maximize the laser power backscattered from the sample entering the spectrometer. Fiber optics were used in our system to mimic the practical dispersion issues encountered in the widely used fiberized vis-OCT systems [34,35]. A supercontinuum laser (SuperK EXTREME, NKT photonics) was employed to provide broadband visible light with a full spectral bandwidth ranging from 493 to $705\mathrm{nm}$ .

To minimize the PMD effect in the vis-OCT system, a polarizer was utilized to increase the polarization linearity of the laser source and a pair of polarization controllers was used to optimize the polarization states of the laser in two interferometric arms. Reflective collimators were used in both arms of the interferometer. A matched glass block (LSM03DC-VIS, Thorlabs) for the scan lens (LSM03-VIS, Thorlabs) was used as a dispersion compensator to minimize the dispersion imbalance between the reference and sample arms.

The back-reflected laser power from the reference arm could be precisely tuned using an adjustable diaphragm. A commercialized spectrometer (CS550-600/200, Wasatch Photonics) was used with a calibrated wavenumber. The detection sensitivity of the system was measured to be $\sim 97.8$ dB with an incident power of $\sim 1.3$ mW for the sample.

![](dt=2026-03-14/ht=06/731f6543f630918923a4ab2b37584c43f4550bcf5002343dd4712ee63a0cef91.jpg)

The PSFs were measured by placing a mirror reflector in the sample arm and minimizing the power from the laser source to avoid the saturation of the spectrometer. A-lines were then acquired separately at two symmetrical locations relative to the zero OPD position by moving the mirror in the reference arm. Then, three dispersion compensation methods, including TSIF, TSMMR, and SAMMR, were used to optimize the PSFs. The

Photonics 2023, 10, 892

3 of 14

TSIF [24] and SAMMR methods only required one PSF measurement, while the TSMMR method [28] needed to use two PSFs measured at symmetric locations. To validate the robustness of the SAMMR method, a glass slide (SiO2, 1 mm in thickness) was inserted in the reference arm to introduce additional dispersion imbalance into the vis-OCT system. B-scan images of a human finger and onion were acquired to test the effectiveness of the dispersion compensation method.

# 2.2. Dispersion Compensation Method

In this section, the theories and mathematics of the TSIF method, TSMMRmethod, and our proposed SAMMR method for compensating material dispersions in the OCT were reviewed and explained. Both the TSMMR and SAMMR methods are phase-based methods and consider the higher-order (>3rd order) dispersion in the OCT system, while the TSIF method only considers up to the third-order dispersion in this paper.

In SD-OCT, A-lines are achieved by performing fast-Fourier transform (FFT) on the linear-wavenumber spectral interferograms acquired from a spectrometer, which can be expressed by the following Equation (1):

$$
\begin{array}{l} I _ {\mathrm {o u t}} (k) \\ = I _ {r} (k) + \sum I _ {m} (k) + 2 \sum_ {m} \sum_ {l \neq m} \sqrt {I _ {m} (k) \cdot I _ {l} (k)} \exp (2 \cdot k \cdot n (k) \cdot z _ {m l} \cdot i) \tag {1} \\ + 2 \sum \sqrt {I _ {r} (k) \cdot I _ {m} (k)} \exp (2 k \cdot n (k) \cdot z _ {m} \cdot i + \phi_ {D} \cdot i) \\ \end{array}
$$

where $I_{r}(k)$ is the laser power spectrum reflected from the reference arm, $I_{m}(k)$ and $I_{l}(k)$ represent the laser power spectra backscattered from the $m_{\mathrm{th}}$ and $l_{\mathrm{th}}$ layers of the sample, respectively. $z_{m}$ is the OPD between the $m_{\mathrm{th}}$ layer of the sample and the reference arm. $z_{ml}$ is the depth difference between $m_{\mathrm{th}}$ and $l_{\mathrm{th}}$ layers in the sample. $n(k)$ is the refractive index of the sample and $k$ stands for the wavenumber of light.

The first two terms in Equation (1) are known as the DC signal, which can be measured in advance. The third term is the auto-correlation component and can sometimes be ignored due to the low reflectivity of the sample and small value of $z_{ml}$ [36]. The last term is the interference signal that contains the depth information of the sample. Here, $\phi_{D}$ refers to the additional phase delay caused by the material dispersion imbalance between two interferometric arms. The three methods to compensate $\phi_{D}$ and the dispersion imbalance are introduced in the following section.

# 2.2.1. Taylor Series Iterative Fitting (TSIF) Method

The TSIF method is based on the Taylor series expansion of propagation constant $\beta (\omega)$ at the center angular frequency $\omega_0$ , as shown in Equation (2) [21]. The material dispersion-induced phase delay $\phi_{D}$ can be represented by the product of the propagation constant and the optical path length $D$ , as shown in Equation (3). The linear-wavenumber spectral interferogram is then multiplied by $\exp (-\phi_D)$ with the dispersion coefficients, i.e., $a_2$ and $a_3$ , as the unknown parameters to be optimized. Usually, the zeroth and first-order dispersion coefficients, i.e.

, $a_0$ and $a_1$ , are ignored since they only change the amplitude of PSF and its absolute location along the imaging depth. By maximizing a metric function of sharpness, one can deduce the second- and third-order dispersion coefficients, i.e
., $a_2$ and $a_3$ [24,26]. Thus, $\phi_D$ can be calculated using Equation (3) and compensated. However, this method requires the optimization of the metric function, and the use of higher dispersion orders significantly increases the calculation complexity.

$$
\beta (\omega) = \beta \left(\omega_ {0}\right) + \frac {d \beta}{d \omega} \left| _ {\omega_ {0}} \left(\omega - \omega_ {0}\right) + \frac {1}{2} \frac {d ^ {2} \beta}{d \omega^ {2}} \right| _ {\omega_ {0}} \left(\omega - \omega_ {0}\right) ^ {2} + \frac {1}{6} \frac {d ^ {3} \beta}{d \omega^ {3}} \left| _ {\omega_ {0}} \left(\omega - \omega_ {0}\right) ^ {3} + \dots \right. \tag {2}
$$

$$
\phi_ {D} (\omega) = \beta (\omega) \cdot D = a _ {0} + a _ {1} (\omega - \omega_ {0}) + a _ {2} (\omega - \omega_ {0}) ^ {2} + a _ {3} (\omega - \omega_ {0}) ^ {3} + \dots \tag {3}
$$

Photonics 2023, 10, 892

4 of 14

# 2.2.2. Two Symmetrical Measurements of the Mirror-Reflection (TSMMR) Method

To compensate for the higher-order dispersion, Singh et al. proposed a phase-based TSMMR method [28]. In order to extract the dispersion-induced phase delay $\phi_{D}$ , the spectral interferogram of PSF was acquired at two symmetrical locations relative to the zero OPD position by moving the mirror in the reference arm. Hilbert transform (HT) was then applied to derive the phase information of the two spectral interferograms after removing the DC signal (i.e., $\phi_{+}$ and $\phi_{-}$ ), which consisted of OPD-induced phase delays (i.e.

, $\phi_{\mathrm{z} + }$ and $\phi_{\mathrm{z}-}$ ) and a dispersion-associated phase delay $(\phi_{D})$ , as shown in Equations (4) and (5). $\phi_{\mathrm{z} + }$ and $\phi_{\mathrm{z}-}$ refer to the phase delays caused by two symmetric OPDs in the interferometer and are considered to have the same absolute value. As shown in Equation (6), $\phi_{D}$ can be conveniently calculated through $\phi_{+}$ and $\phi_{-}$ . This method simplifies the calculation, avoids the complicated optimization procedure in the TSIF method, and can compensate for the higher-order material dispersion.

However, this method causes a potential alignment issue in the reference arm.

$$
\phi_ {+} = \phi_ {\mathrm {z} +} + \phi_ {D} \tag {4}
$$

$$
\phi_ {-} = \phi_ {z -} - \phi_ {D} \tag {5}
$$

$$
\phi_ {D} = \frac {\phi_ {+} - \phi_ {-}}{2} \tag {6}
$$

# 2.2.3. Single Arbitrary Measurement of Mirror-Reflection (SAMMR) Method

To overcome the limitations of the TSIF and TSMMR methods, we proposed the SAMMR method to directly extract and compensate $\phi_{D}$ . As indicated by its name, this method requires only a single measurement of a mirror reflection with arbitrary OPD within the imaging depth. Figure 2 illustrates the post-processing procedures based on the SAMMR method for dispersion compensation in the vis-OCT system. First, the DC signals measured from the reference and sample arms were subtracted from the output spectrum $(I_{out}(\lambda))$ acquired with a spectrometer in the wavelength domain.

Then, a cubic spline interpolation on the remaining signal $(I_{rem}(\lambda))$ was performed to achieve a spectrum of linear wavenumber distribution $(I_{rem}(k))$ . The interference signal $I_{\mathrm{int}}(k)$ was calculated to further remove the residual DC signal $(I_{res}(k))$ , which was calculated by moving an average window over the spectrum of $I_{rem}(k)$ . This step helped avoid the potential artificial peaks in A-lines (see Section 3.4). Then, HT was conducted to derive the complex interference signal $\widetilde{I}(k)$ and phase delay $\phi$ , as shown in Equations (7) and (8).

The phase delay $\phi$ consisted of the phase induced by the OPD and material dispersion (Equation (9)). After determining $z_{\mathrm{OPD}}$ by locating the peak position of PSF $(A_{disp}(z))$ derived from FFT of $I_{int}(k)$ , $\phi_{D}$ could be conveniently calculated in Equation (9). As shown in Equation (10), the dispersion compensated spectral interferogram $(I_{comp}(k))$ is derived and the A-line $(A_{comp}(z))$ is then achieved through the FFT of $I_{comp}(k)$ .

$$
\widetilde {I} (k) = I (k) + i H [ I (k) ] \tag {7}
$$

$$
\phi = \operatorname {A n g l e} \left\{\frac {H [ I (k) ]}{I (k)} \right\} \tag {8}
$$

$$
\phi = 2 k z _ {O P D} + \phi_ {D} \tag {9}
$$

$$
I _ {c o m p} (k) = \widetilde {I} (k) \exp (- i \phi_ {D}) \tag {10}
$$

Photonics 2023, 10, 892

5 of 14

![](dt=2026-03-14/ht=06/4e7893d6f466aff782b88a37d7e250d671fc18d503999018cbcd9d3ddad01903.jpg)

# 3. Result

# 3.1. Performance Evaluation

# 3.1.1. Dispersion-Compensated PSFs and Their Symmetrical Properties

The original PSF measured in the vis-OCT system and dispersion-compensated PSFs using TSIF, TSMMR, and the proposed SAMMR methods are illustrated in Figure 3a. It was found that the original PSF had a broad and asymmetrical shape, and the TSIF method could barely improve the PSF due to the pronounced higher-order dispersion in the vis-OCT system. In contrast, both phase-based methods could sharpen and symmetrize the PSF, while the SAMMR method provided a PSF of better symmetry than that of the TSMMR method.

As shown in Figure 3b, the symmetry of PSFs is indicated using the amplitudes of two symmetric first side lobes of the PSF. A symmetry metric is defined in Supplementary Note 1 and also indicates the better symmetrical property of PSF calculated by the SAMMR method. It was also noted that the amplitude of the main peak of PSF increased from about $89\mathrm{dB}$ (original) to approximately $100\mathrm{dB}$ (SAMMR method). The linear amplitude of PSFs with different dispersion compensation methods is also presented in Figure 3c, demonstrating the best performance of the SAMMR method.

![](dt=2026-03-14/ht=06/091e13f5e4b764967ee9a14263e8ac7eb3635a8c9063afe2d68239972c0be579.jpg)

![](dt=2026-03-14/ht=06/de1f09c7bfecba08acef528b6a78d76a90d15ef593d744832ec4d29b55e30808.jpg)

![](dt=2026-03-14/ht=06/6d12be6b9fc2c18ab1324a1fab6958ff322f24ab4c3ec5a2c0eb39a01ad3211d.jpg)

# 3.1.2. Signal-To-Noise Ratio, Full Width at Half Maximum, and Contrast of PSFs

To compare the performance of different dispersion compensation methods, we evaluated the signal-to-noise ratio (SNR, also see Figure 3a) and the full width at half maximum (FWHM) of the PSFs. As shown in Figure 4a,b, it is found that all three methods can improve the SNR of PSF; however, the TSIF and TSMMR methods can barely improve the

Photonics 2023, 10, 892

6 of 14

FWHM of PSF, and the proposed SAMMR method provides both the highest SNR and the smallest FWHM of PSF. The underperformance of the TSIF method was mostly due to the significant higher-order dispersion in the vis-OCT system, while the performance of the TSMMR method was mainly manifested in the increased SNR and optimized sharpness and symmetry of PSF (also see Figure 3) with the FWHM not being improved by TSMMR.

![](dt=2026-03-14/ht=06/8ae14c845af9e1f29dad28dba7fc7b7b7c85abc21df7daa87f787b24650538f4.jpg)

![](dt=2026-03-14/ht=06/4691093f4730ef10902284a3d7a67524ab27ea23e2b0eb34c0e9c606f9f022a4.jpg)

![](dt=2026-03-14/ht=06/f301a30e9ac8b132bfba3f6c2447ac1f70a23c9712ee9b3af2370a8afb564730.jpg)

We further evaluated the contrast of PSF to demonstrate the performance of the three methods. The PSF contrast was defined by the difference between the main peak and the higher first side lobe of PSF (see Figure 3b). As shown in Figure 4c, the SAMMR method pr
ovides a PSF contrast of about $30~\mathrm{dB}$ , versus around $5\mathrm{dB}$ from the TSIF method and approximately $15\mathrm{dB}$ from the TSMMR method; though, the FWHM of PSF is not improved by TSMMR. The suboptimal PSF contrast may indicate that the material dispersion is not adequately compensated by the TSIF and TSMMR methods.

# 3.2. Improvement of Vis-OCT Images Using the SAMMR Method

To demonstrate the performance of the SAMMR method, vis-OCT images of a human fingertip and onion were tested. We followed the post-processing procedures illustrated in Figure 2 and used the SAMMR method to extract the material dispersion-induced phase delay, i.e., $\phi_{D}$ , in the vis-OCT system. Then, the derived $\phi_{D}$ was used to compensate for the dispersions in B-frames using Equation (10). As shown in Figure 5a-d, a significantly sharpened tissue structure with an improved contrast is observed in the dispersion-compensated fingertip image versus the original one.

In addition, the SAMMR and TSMMR methods offered greater improvements in the sharpness and SNR of images when compared with the TSIF method. The selected A-lines from original and dispersion-compensated B-frames were compared side-by-side and are illustrated in Figure 5e. The proposed method was further validated using the onion image. As seen in Figure 6a-d, the dispersion-compensated images clearly delineate finer cell wall structures of the onion compared to those in the original one. Additionally, the SAMMR and TSMMR methods show sharper images using the TSIF method.

The sharpened cell wall structures with improved SNR values are also indicated by the spiculate peaks along the imaging depth in a representative A-line (Figure 6e).

Photonics 2023, 10, 892

7 of 14

![](dt=2026-03-14/ht=06/38d1fb1a46498d0cdac7a0610f82fea8ac99f51c0100a99aeb98d9837ae1e73e.jpg)

![](dt=2026-03-14/ht=06/4aac3087021e826a171ebf15961cb34c71dbfd71fe79052ebc944093ea199a9f.jpg)

![](dt=2026-03-14/ht=06/8855a9d5d9d62eca39c2ac5032f75202cd6b21572963c03093d2d25f0ec7ca69.jpg)

# 3.3. Robustness of the SAMMR Method

In this section, the robustness of the SAMMR method to compensate for the additional material and higher-order dispersions in the vis-OCT system are discussed.

Photonics 2023, 10, 892

8 of 14

# 3.3.1. Additional Dispersion in the Vis-OCT System

A glass slide was inserted into the reference arm to introduce an additional dispersion mismatch in the vis-OCT system and the three dispersion-compensation methods were tested for compensation effects. As seen in Figure 7, the TSIF method can barely optimize the PSF and both phase-based methods considerably improve the sharpness and symmetry of the PSF, while the SAMMR method provides the PSF with the best symmetrical property.

![](dt=2026-03-14/ht=06/4fe9c21ea4686c303060d40380b497f9727619dfdcec103081bb3b3a69e92801.jpg)

![](dt=2026-03-14/ht=06/1fde7679aa358f4aff8480141ef6c4eb8e433fc3eb91b1435e340046b3e5b000.jpg)

Figure 8 further shows a detailed comparison of SNR, FWHM, and contrast of resulting PSFs from different methods with and without additional glass slides in the reference arm. It was found that the additional dispersion in the vis-OCT system led to an original PSF of a lower SNR (due to the light attenuation inside the glass slice), a wider FWHM, and a worse contrast (due to the increased dispersion mismatch in the interferometer) compared to the original PSF measured without a glass slide.

Nevertheless, the SAMMR method still provided the dispersion-compensated PSF with the highest SNR, the smallest FWHM, and the best contrast than those achieved with the other two methods. Notably, a PSF of comparable FWHM and contrast could be achieved with the SAMMR method in both cases with and without additional dispersions in the vis-OCT system.

![](dt=2026-03-14/ht=06/ffec4848172fad7d0d6079054aea0267573eb38f2f1f96d37807caa9369b69a6.jpg)

![](dt=2026-03-14/ht=06/5b75268895f5be51a4b9ff16c0145bf4387d5d602f4d73a24a50ba0dfe2b0adc.jpg)

![](dt=2026-03-14/ht=06/7904c8d45bc6144c6b6ce7f3397198843a36d2cae4281f1aff81dbb3653c0809.jpg)

# 3.3.2. Higher-Order Dispersion in the Vis-OCT System

Compared to other near-infrared OCT systems, the vis-OCT system suffers more from higher-order dispersion. A second-order dispersion can easily be compensated for by matching optical materials or prism pairs in sample and reference arms. Therefore, a method capable of compensating for a higher-order dispersion is important for the vis-OCT system. As shown in Figure 9a, using the laser spectrum measured from the reference and sample arms and assuming an OPD of $200\mu \mathrm{m}$ ( $z_{m} = 200\mu \mathrm{m}$ ), an ideal PSF without

Photonics 2023, 10, 892

9 of 14

any dispersion-induced phase delay $(\phi_D = 0)$ can be simulated by applying FFT to the interference term of Equation (1). Then, a PSF with a higher-order dispersion can be simulated by adding a phase delay $(\phi_D)$ into Equation (1). In our simulation, the phase delay $(\phi_D)$ was calculated by assigning the dispersion coefficients (in the wavenumber domain), such as $a_2 = 279$ , $a_3 = 121.5$ , $a_4 = 33.5$ , $a_5 = 1.7$ , of a $\mathrm{SiO}_2$ of $5\mathrm{mm}$ in length into Equation (3).

Then, the three dispersion compensation methods were tested on the PSF with a higher-order dispersion. It was found that all methods could optimize the sharpness and symmetry of the PSF, while only SAMMR and TSMMR methods could compensate for all the higher-order dispersions by providing a dispersion-compensated PSF similar to the ideal one with an FWHM of $1.36~{\mu\mathrm{m}}$ compared to a $8.35~{\mu\mathrm{m}}$ FWHM by using the TSIF method. However, in reality, the dispersion was not restricted to the fifth-order dispersion and was not solely caused by a single material.

All materials used in the system affected the dispersion, such as fiber and the lens.

![](dt=2026-03-14/ht=06/bce25302695c5838f770c0c77e811f1dcae05ec447bc31b61e1ad990fa74b59b.jpg)

![](dt=2026-03-14/ht=06/90dae315f98d07b2c2725d5006beb797771f6f4370c76e3b79259182d2c2c03c.jpg)

As for the TSMMR method, the movement of the mirror in the reference arm could easily result in the tilting of the optical alignment, as shown in Figure 9b. The tilting of the mirror alters the optical path in the dispersion compensator, leading to a different dispersion-induced phase delay in two symmetric mirror measurements, a biased estimation of $\phi_{\mathrm{D}}$ in the TSMMR method and potential misalignment. In contrast, the SAMMR method only required one measurement of the mirror reflection at any location within the imaging depth, which avoided the translation and tilting of the mirror in the reference arm and considerably improved its performance for the dispersion compensation.

# 3.4. Oscillation of Phase Delay and Artificial Peaks

It was noted that directly using the SAMMR method on a spectral interferogram caused artificial peaks in the resulting A-line. The interferogram signal after subtracting the prior measured DC signals can be expressed as below:

$$
I _ {r e m} (k) = 2 \sqrt {I _ {r} (k) I _ {s} (k)} \cos \left(2 k z _ {O P D} + \phi_ {D}\right) +
I _ {r e s} (k) \tag {11}
$$

where the first term is the interference signal, $I_{r}(k)$ and $I_{s}(k)$ are the reflected power spectra from the reference and sample arms, respectively, and $I_{res}(k)$ stands for the residual DC signal. If the residual DC signal is not sufficiently removed, the phase delays $\phi$ and $\phi_{D}$ calculated with the SAMMR method become inaccurate (since both the resulting real and imaginary parts of HT in Equation (7) are biased) and demonstrate a characteristic oscillation feature (see blue curve in Figure 10a), leading to an artificial peak in the resulting A-line (see blue curve in Figure 10b).

Photonics 2023, 10, 892

10 of 14

![](dt=2026-03-14/ht=06/cd01523953230a0e32f499c4682bb7d045553f8577f2b7ebad28036834444d28.jpg)

![](dt=2026-03-14/ht=06/1ac5872cd9cd7117a80e217ddc3c2f03e98ebbd64ae67fbaf772d3686a40240e.jpg)

![](dt=2026-03-14/ht=06/02a78148dcde71b2b75952a7220ec180b8b06aa6ddf4224aa51b036b94c4e4fd.jpg)

To overcome this problem, a smoothing window was used to achieve a smoothed phase signal (see red curve in Figure 10a) for compensating for the dispersion and avoiding the artificial peak (see red curve in Figure 10b). As previously discussed in Section 2.2.3, a step by subtracting the moving average of the signal could also be used to remove the residual DC signal $(I_{res}(k))$ in the spectral interferogram $(I_{rem}(k))$ . This approach allowed us to derive the interference signal $(I_{int}(k)$ , see purple curve in Figure 10c), minimize the oscillation in the phase signal (see yellow curve in Figure 10a), and eliminate the artificial peaks in the A-line (see yellow curve in Figure 10b).

Moreover, the low-frequency oscillations in the phase curve (see yellow, red, and blue curves in Figure 10a) were mainly due to the pronounced higher-order dispersion of the vis-OCT system. This issue made it difficult to fit perfectly with Equation (3), which led to the suboptimal performance of the TSIF method in improving the PSF. These observations further verified the effectiveness and robustness of the SAMMR method for dispersion compensation in the vis-OCT system.

# 3.5. Application of the SAMMR Method to the 800 nm SD-OCT System

The SAMMR method was further tested on the PSFs measured in a homemade $800\mathrm{nm}$ SD-OCT system. As shown in Figure 11a,b, the TSIF method provides a better dispersion compensation in the $800\mathrm{nm}$ domain compared to its performance in the visible-light domain (see Figures 3 and 7) in terms of the sharpness and symmetry of the resulting PSF. One reason for this improvement was mainly the relatively smaller higher-order dispersions in the near-infrared domain than those in the visible-light range.

However, since the higher-order dispersion was not adequately compensated for in the TSIF method, the resulting PSF was still asymmetric (see Figure 11b). In contrast, the symmetrical property, SNR, FWHM, and contrast of PSF could be further improved using phase-based TSMMR and SAMMR methods (see Figure 11a,b). It is important to note that the TSMMR method suffers from the above-mentioned limitations on the resolution of the translation stage and the tilting of the mirror in the reference arm.

Photonics 2023, 10, 892

11 of 14

![](dt=2026-03-14/ht=06/47c9291226e9db48e82afacafd874daa1565134f8b0413f0c1262ac9011c6681.jpg)

![](dt=2026-03-14/ht=06/f425114a44d771576ff595b37ee060b7206732981ca540bed25ce4ad4276115b.jpg)

# 4. Discussion and Conclusions

In this paper, we proposed a generic and robust method, i.e., SAMMR, to effectively manage the pronounced dispersion mismatch in the visible-light SD-OCT system. Specifically, the proposed method took advantage of a single arbitrary measurement of mirror reflection in the sample arm to accurately extract the material dispersion-induced phase delay for dispersion compensation. This approach eliminated the potential phase errors caused by the limited mechanical accuracy of the translational stage in the TSMMR method and the tilting of optical alignment in the reference arm.

Our method was able to efficiently compensate for the higher-order dispersions, and thus provided an ultrahigh axial resolution (about $1.5\mu \mathrm{m}$ ) in vis-OCT imaging. The robustness of the SAMMR method was also validated through its effective compensation of additional material and higher-order (more than third-order) dispersions in the vis-OCT system. The effectiveness and genericity of the SAMMR method were further demonstrated in an $800\mathrm{nm}$ SD-OCT system. This generic and robust method for dispersion compensation can help to acquire an image with an optimal resolution.

Moreover, the post-processing step, such as the step mentioned in Section 2.2.3, can be used to remove the DC signal and minimize artificial peaks. The calibrated high-resolution images can be further used for accurately tracking angiographic changes and visualizations of subtle structures in tissues.

Although the SAMMR method could effectively compensate for the residual dispersion imbalance existing between the reference and sample arms and help optimize the axial resolution and image quality in the vis-OCT system, there were still several limitations. First, the SAMMR method was only applicable in the OCT system, where measuring the mirror reflection in the sample arm was possible. Second, only the material dispersion in the OCT interferometer was compensated for with the SAMMR method.

This method could not compensate for the dispersion induced by the sample itself, for example, the dispersion induced by vitreous humor when imaging retinal layers. Third, limited by the single measurement of PSF with a normally incident laser beam on the mirror, the material dispersion was assumed to be the same in the full field of view (FOV) of the sample arm.

However, when imaging the sample at the corner of FOV, the light beam was tilted in the scan lens, leading to a longer optical path length in the scan lens and resulting in a different dispersion compared to the measurement in the middle of the FOV. To overcome these limitations, we aim to develop a digital dispersion compensation method not requiring the measurement of mirror reflection and capable of compensating for the dispersion of the sample and the spatially dependent dispersion in the sample arm in our future work.

Supplementary Materials: The following supporting information can be downloaded at: https://www.mdpi.com/article/10.3390/photonics10080892/s1, Figure S1: The calculation flowchart of TSIF method and TSMMR method; Note 1: The indicator for the symmetrical property.

Photonics 2023, 10, 892

12 of 14

Author Contributions: Conceptualization, J.W., C.X., D.C. and W.Y.; methodology, J.W. and C.X.; software, S.Z.; validation, J.W.; formal analysis, J.W.; investigation, J.W.; resources, J.W.; data curation, J.W.; writing—original draft preparation, J.W. and W.Y.; writing—review and editing, J.W., C.X., S.Z., D.C., H.Q., A.K.N.L., C.K.S.L. and W.Y.; visualization, J.W.; supervision, W.Y.; project administration, W.Y.; funding acquisition, D.C. and W.Y. All authors have read and agreed to the published version of the manuscript.

Funding: This work was supported by the Shun Hing Institute of Advanced Engineering (BME-p3-20/4720264) at the Chinese University of Hong Kong (CUHK), the Research Grants Council (RGC) of Hong Kong SAR (ECS24211020, GRF14203821, GRF14216222), the Innovation and Technology Fund (ITF) of Hong Kong SAR (ITS/240/21), the Science, Technology, and Innovation Commission (STIC) of Shenzhen Municipalit
y (SGDX20220530111005039), and the National Science Foundation Program of China (61835015).

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: Not applicable.

Conflicts of Interest: The authors declare no conflict of interest.

# References

Photonics 2023, 10, 892

13 of 14

Disclaimer/Publisher's Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.

Photonics 2023, 10, 892

14 of 14