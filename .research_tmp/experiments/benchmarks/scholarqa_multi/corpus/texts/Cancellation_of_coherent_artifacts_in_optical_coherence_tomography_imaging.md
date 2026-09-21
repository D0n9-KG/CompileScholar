# Cancellation of coherent artifacts in optical coherence tomography imaging

Daqing Piao, Quing Zhu, Niloy K. Dutta, Shikui Yan, and Linda L. Otis

Coherent artifacts in optical coherence tomography (OCT) images can severely degrade image quality by introducing false targets if no targets are present at the artifact locations. Coherent artifacts can also add constructively or destructively to the targets that are present at the artifact locations. This constructive or destructive interference will result in cancellation of the true targets or in display of incorrect echo amplitudes of the targets. We introduce the use of a nonlinear deconvolution algorithm, CLEAN, to cancel coherent artifacts in OCT images of extracted human teeth. The results show that CLEAN can reduce the coherent artifacts to the noise background, sharpen the air-enamel and enamel-dentin interfaces, and improve the image contrast. © 2001 Optical Society of America

OCIS codes: 170.4500, 170.3880, 170.3890, 170.1850.

# 1. Introduction

Optical coherence tomography (OCT) is a new imaging technique that has various medical applications.1-8 OCT exploits the short temporal coherence of the broadband light sources that are used to achieve optical scanning of scattering tissue in the depth dimension. Spatial scanning is accomplished by mechanical or optical scanning of the beam. At each spatial location, the OCT scanner output is the Fourier transform of the source spectrum convolved with the tissue reflectivity over a narrow scattering angle and with the transfer function of the detection system.

Assuming that the transfer function of the detection system is an ideal delta function, the imperfect Gaussian shape of the source spectrum will introduce coherent artifacts or sidelobes. If the source spectrum is ideal, however, the nonideal transfer function of the detection system may introduce coherent spikes as a result of reflections from optical components, and these coherent spikes will appear in OCT images as coherent artifacts. In a realistic OCT system, coher

ent artifacts can be introduced by both sources. Several authors have reported observing coherent artifacts or coherent sidelobes in their OCT systems. $^{1,5,9,10}$ These artifacts were due primarily to multiple reflections inside the source, which resulted in an imperfect source spectrum. The locations and strengths of these artifacts vary with the source and the source current level as well as with the configuration of the detection system. Coherent artifacts can severely degrade OCT image quality by introducing false targets if no targets are present at the artifact locations.

Artifacts can also add constructively or destructively to the targets that are present at the artifact locations. This constructive or destructive interference will result in cancellation of the true targets or in the display of incorrect echo amplitudes of the targets. In this paper we show that these artifacts in OCT images of extracted teeth can be effectively reduced to the background-noise level by use of an iterative nonlinear deconvolution procedure known as CLEAN.

CLEAN was invented in the middle 1970s for radio astronomy[11] and was modified later for use in microwave imaging.[12] In ultrasound, similar deconvolution procedures were used to reduce the refractive artifacts.[13] In OCT the use of CLEAN to reduce speckle noise caused by interference of nonresolvable scatterers of highly scattering tissue has been reported by Schmitt.[14] Schmitt implemented CLEAN by using a theoretical point-spread function with a Gaussian envelope. Therefore, coherent artifacts caused by an imperfect Gaussian source spectrum or nonideal detection optics have not been considered.

D. Piao, Q. Zhu (zhu@engr.uconn.edu), and S. Yan are with the Department of Electrical and Computer Engineering, University of Connecticut, Storrs, Connecticut 06269-2157. N. K. Dutta is with the Department of Physics, University of Connecticut, Storrs, Connecticut 06269-3046. L. L. Otis is with the University of Pennsylvania School of Dental Medicine, Philadelphia, Pennsylvania 19104-6003.

Received 18 December 2000; revised manuscript received 26 April 2001.

0003-6935/01/315124-08$15.00/0

© 2001 Optical Society of America

5124

APPLIED OPTICS / Vol. 40, No. 31 / 1 November 2001

![](dt=2026-06-02/ht=00/5e400c7d198fe75df140f543bb36dff351ede2c7c902ca6550d9bcfe4af41278.jpg)

In addition, that author used an unconventional OCT system with an array of sources and detectors, which enables CLEAN to be implemented in two dimensions.

In this paper we focus on the use of CLEAN, implemented in a conventional OCT system, to cancel coherent artifacts in OCT images of extracted teeth. A human tooth, as illustrated in Fig. 1, consists of a crown and a root. The junction between the crown and the root is called the cervical margin. The tooth's crown is covered by an acellular, highly mineralized tissue, i.e., enamel. Enamel is translucent and varies in thickness from a maximum of $2.5\mathrm{mm}$ to a feather edge at the cervical margin.

The bulk of the tooth comprises dentin, which is a hard, elastic, avascular tissue that is approximately $70\%$ mineralized. The interior of the tooth is called the pulp. The pulp consists of soft connective tissue that is innervated and highly vascular. We show that CLEAN reduces the artifacts associated with the air-enamel and enamel-dentin interferences to the noise floor, sharpens these interferences, and improves the contrast between the interfaces and the surrounding medium.

# 2. Basic Principle

# A. Point-Spread Function

The point-spread function (PSF) of an OCT scanner depends on the coherence length of the source and on the effective aperture defined by the pupil functions of the source and the detection optics. Therefore it is a two-dimensional function and is written as $h(x, z)$ , where $z$ and $x$ are the propagation and the lateral dimensions, respectively. In conventional OCT scanners, a single beam is scanned across the sample; therefore the spatial distribution of $h(x, z)$ cannot be measured. Figure 2 is a schematic of the OCT scanning process. A point scatterer is located in the me

![](dt=2026-06-02/ht=00/a132f241a8f594e1963386903c5ce50dafad325ef9a5ab95fb8cef897f8232bc.jpg)

dium, and it scatters the incident light. The light reflected within a narrow angle is received by the detection optics, and the off-axis scattered light falls outside the receiving aperture. If more detectors were located at the off-axis positions, the scattered light could have been received over a larger angle and $h(x,z)$ in the $x$ dimension could have been measured.

However, because the incident beam is highly focused and the energy received over the beam spot area (bsa in the equation below) far exceeds the energy of the off-axis scattered light, it is possible to use a one-dimensional PSF, $h(z)\approx \int_{\mathrm{bsa}}h(x,z)\mathrm{d}x$ , to estimate the $h(x,z)$ . We found that by using this approximate PSF in CLEAN we could effectively remove a large portion of the sidelobe energy.

# B. Conventional Deconvolution

Assume that the medium contains $N$ point scatterers, each with complex strength $C_j$ . The original image that we wish to reconstruct is

$$
O _ {\text {i m a g e}} (x, z) = \sum_ {j = 1} ^ {N} C _ {j} \delta \left(x - x _ {j}, z - z _ {j}\right), \tag {1}
$$

where $x_{j}$ and $z_{j}$ are the coordinates of the $j$ th scatterer. The so-called dirty image is

$$
D _ {\text {i m a g e}} (x, z) = O _ {\text {i m a g e}} (x, z) * h (x, z) + N (x, z), \tag {2}
$$

where $*$ denotes convolution and $N(x,z)$ represents system noise.

Conventional deconvolution procedures include a Fourier transform of the dirty image $D_{\mathrm{image}}(x,z)$ and PSF $h(x,z)$ to obtain their spat
ial frequency spectrum $D_{\mathrm{image}}(u,v)$ and $h(u,v)$ . The original image can be recovered from the inverse Fourier transform of

$$
\frac {D _ {\text {i m a g e}} (u , v)}{h (u , v)} = O _ {\text {i m a g e}} (u, v) + \frac {N (u , v)}{h (u , v)}, \tag {3}
$$

1 November 2001 / Vol. 40, No. 31 / APPLIED OPTICS

5125

where $O_{\mathrm{image}}(u,v)$ and $N(u,v)$ are the spatial frequency spectra of $O_{\mathrm{image}}(x,z)$ and $N(x,z)$ , respectively. Unfortunately, this procedure is highly sensitive to $h(x,z)$ and to system noise. In a practical OCT system, $h(u,v)$ approaches zero rapidly because of the limited bandwidth in the axial direction and the effective aperture in the lateral direction. Therefore this procedure is not robust, and the deconvolution results depend strongly on system noise.[14]

# C. CLEAN Procedure

CLEAN provides more-robust means to reconstruct the original image. CLEAN is performed in the spatial image domain, and it iteratively recovers the original image. The basic steps of the CLEAN algorithm are as follows: First, find the brightest pixel $[x^{(1)},z^{(1)}]$ in the dirty image and subtract a fraction of the deconvolution kernel

$$
\alpha h \left[ x - x ^ {(1)}, z - z ^ {(1)} \right] D _ {\text {i m a g e}} \left[ x ^ {(1)}, z ^ {(1)} \right] / h _ {\max } \tag {4}
$$

from the dirty image. Parameter $\alpha$ is called loop gain, which is less than unity, and it represents the fraction that is subtracted out by each iteration of CLEAN. Second, find the brightest pixel $[x^{(2)},z^{(2)}]$ from the residual of

$$
D _ {\mathrm {i m a g e}} - \alpha h [ x - x ^ {(1)}, z - z ^ {(1)} ] D _ {\mathrm {i m a g e}} [ x ^ {(1)}, z ^ {(1)} ] / h _ {\max} (5)
$$

and repeat the first step. The iteration continues until the maximum value in the dirty image reaches the noise floor. Assume that a total of $M$ targets of strengths $\alpha D_{\mathrm{image}}[x^{(i)},z^{(i)}]$ are found before iteration is stopped. The final CLEANed image is obtained by convolution of the set of delta functions of strengths $\alpha D_{\mathrm{image}}[x^{(i)},z^{(i)}]$ at locations $[x^{(i)},z^{(i)}]$ with the clean beam of the PSF that contains only the main lobe. Generally the residuals after CLEAN are added back to the CLEANed images to produce a realistic noise level in the final image.

# D. One-Dimensional CLEAN Procedure for Conventional OCT Imaging

As we discussed above, a two-dimensional PSF cannot be obtained from a conventional OCT scanner. Therefore we used the one-dimensional PSF measured from a mirror to approximate the two-dimensional PSF. In this modified procedure, we first find the brightest pixel $[x^{(1)},z^{(1)}]$ in the dirty B-scan image and subtract a fraction of the deconvolution kernel

$$
\alpha h \left[ z - z ^ {(1)} \right] D _ {\text {i m a g e}} \left[ x ^ {(1)}, z ^ {(1)} \right] / h _ {\max } \tag {6}
$$

from the dirty A-scan line. We then find the brightest pixel $[x^{(2)},z^{(2)}]$ from the residual of

$$
D _ {\text {i m a g e}} - \alpha h [ z - z ^ {(1)} ] D _ {\text {i m a g e}} [ x ^ {(1)}, z ^ {(1)} ] / h _ {\max } \tag {7}
$$

and repeat the first step. The final CLEANed image is obtained by convolution of the set of delta functions of strengths $\alpha D_{\mathrm{image}}[x^{(i)},z^{(i)}]$ at locations $[x^{(i)},z^{(i)}]$

![](dt=2026-06-02/ht=00/abbb0a2ec255274edaca9e27c2a97da92cdd72b4e2ce43ff7c28f37d18ac19e5.jpg)

with the clean beam of the 1-D PSF containing only the main lobe. The disadvantage of using the 1-D CLEAN is that the off-axis sidelobe energy is not removed. However, we found that the off-axis sidelobe energy is significantly smaller than the on-axis sidelobe energy, and the one-dimensional CLEAN effectively reduces the sidelobe energy to the noise floor in the images.

# 3. Methods

A schematic diagram of our OCT system is shown in Fig. 3. An erbium-doped fiber amplifier (EDFA) at a center wavelength of $1550~\mathrm{nm}$ is used as the lowcoherence source. The optical power output and the 3-dB coherence length of this EDFA at $40\mathrm{-mA}$ current are $0.43\mathrm{mW}$ and $25~{\mu\mathrm{m}}$ respectively. The light from the source enters the optical circulator and is split into a reference for the differential receiver input and a signal that passes through a 3-dB fiber coupler, where the outputs form a Michelson interferometer.

The reference arm of interferometer consists of a microscope objective (O1), a plane mirror, a speaker, and a linear dc motor. The mirror is glued onto the bowl of the speaker, which is mounted upon the linear motor. The speaker provides the Doppler shifting frequency and is driven by a sinusoidal wave of $100\mathrm{-Hz}$ frequency and 6.2-V peak-to-peak amplitude. The measured maximum Doppler shifting frequency is $17\mathrm{kHz}$ . The depth $(Z$ direction) scan is achieved by translation of the speaker and motor assembly.

The sample arm consists of a microscope objective (O2), a galvanometer scanner, a lens, and a sample. The alignment requirement for the sample arm is not only that the incident light be perpendicular to the rotation axis of the galvanometer mirror but also that the incident-light spot be located at the focal point of the lens. With this precise alignment, the spatial $(X$ direction) scanning is achieved by translation of the rotation of the galvanometer mirror

5126

APPLIED OPTICS / Vol. 40, No. 31 / 1 November 2001

![](dt=2026-06-02/ht=00/115d92f47f85172b6d232816b0f860faef0c62bfb0da7a52a52b35cd1d66a1a5.jpg)

![](dt=2026-06-02/ht=00/bd62f99ff81d6b8560d001b63f4404e9fae5ac5d92d6de80d56f86c6781e36ff.jpg)

![](dt=2026-06-02/ht=00/e80d3ff9cb1c3d26266574f6b3d1f7164eb2f9aa9837122d37436e6a49030c53.jpg)

to linear scanning upon the sample without the introduction of a change in optical path.

The data acquisition is accomplished with a dualbalanced optical receiver, a high-pass filter, and a PC. The interferometer output is received by the dualbalanced receiver, which is used to reduce the common-mode noise. The resultant signal is passed through a high-pass filter, which is used to preserve the wide frequency range of signals. We found that a high-pass filter tracks the peak Doppler frequency better than a bandpass filter. The filtered signals are sampled by an analog-to-digital converter of $400\mathrm{-kHz}$ sampling frequency, and the sampled data are Hilbert transformed to yield the envelope of the Doppler signals. An A-scan line is obtained from this envelope signal.

The galvanometer and the linear motor are controlled by a galvanometer driver and a dc motor driver, respectively. The rotation of the galvanometer and the translation of the motor are synchronized by the PC control software. A 10-Hz sine wave generated by the PC is used to drive the galvanometer controller, which in turn drives the scanning mirror. One set of B-scan data is acquired for 128 cycles of galvanometer rotation and $3.84\mathrm{-mm}$ in depth translation of the reference arm.

For the images presented, the transversal scanning range is set to 6 mm, and a total of 400 pixels are used. Thus the pixel size at the transversal $(X)$ direction is $15~{\mu\mathrm{m}}$ which is less than the 3-dB spot size $(\sim 30~\mu \mathrm{m})$ of the sample beam. The pixel size for the depth direction $(Z)$ is $15~{\mu\mathrm{m}}$ which is less than the 3-dB coherence length $(25~{\mu\mathrm{m}})$ of the EDFA source at a 40-mA driving current.

# 4. Results

The measured PSF of the system is shown in Fig. 4(a). The distance between the main lobe and the artifacts' positions is $1.335\mathrm{mm}$ , which is within the scan range of $3.84\mathrm{mm}$ . W
hen a planar mirror is imaged, the artifacts create two parallel lines, one on each side of the central image line [see Fig. 4(b)]. Under this ideal imaging condition, the artifacts can be easily identified. However, when the imaging medium is an extracted tooth, the artifacts appear as two complicated curves, one on each side of the air-enamel interface [see Fig. 4(c)]. These artifact curves can be confused with real interfaces in diagnosis. Furthermore, if the enamel-dentin interface were located in the neighborhood of the artifact curve, it would not be imaged with the correct echo amplitude.

Figure 5(a) shows the dirty image of the extracted tooth, and Fig. 5(b) shows the CLEANed image. Figure 5(c) is a microscopic picture of the sectioned tooth image; the scanned region contains a fissure defect in the enamel (marked by a black arrow). Diagnostically, it is important to determine whether such a defect extends beyond the enamel layer. Comparing OCT images before and after the CLEAN algorithm has been implemented, we can observe significant improvement in image detail in the fissure region [marked by the arrow in the upper part of Fig. 5(a)].

In addition, the two artifact curves on the sides of the air-enamel interface shown in Fig. 5(a) are reduced to background level after the CLEAN algorithm has been implemented. The contrast between the air and the enamel has been improved, and the interface is delineated well after application of

1 November 2001 / Vol. 40, No. 31 / APPLIED OPTICS

5127

![](dt=2026-06-02/ht=00/5ed72375825c6db14fbf908e12a9c9f7f31ab31ea7afae597a23d977036b3f69.jpg)

![](dt=2026-06-02/ht=00/92f057a4148de0829ac1bcbd46539129d237e2344f311c8a513f862ff867bdc0.jpg)

![](dt=2026-06-02/ht=00/2cc51e6267d91296b43d8683188496c162e55b325b67e145f74ca6eba896df3a.jpg)

CLEAN. Furthermore, the artifacts associated with the enamel-dentin interface [the arrow at the right at the bottom of Fig. 5(a)] are removed, and this interface [left arrow at the bottom of Fig. 5(a)] is enhanced after CLEAN has been used. To assess the reduction of sidelobe artifacts quantitatively, we show an A-scan line in Fig. 6(a), which was obtained from an area that includes the air-enamel and enamel-dentin interfaces as well as their coherent sidelobes. The measured peak artifacts associated with these interfaces are $-15$ dB and are reduced to the background level after implementation of CLEAN [Fig. 6(b)], and both interfaces are enhanced.

Another example of CLEAN is shown in Fig. 7; a tooth that contains a metallic restoration is imaged. Diagnostically, it is important to verify smooth integration of the restoration into the tooth surface and to detect structural defects of the tooth at the interface.

![](dt=2026-06-02/ht=00/3d3359e587be7307f89c6d035243c588b6abcb46869633bdb934864f1dc8fe3a.jpg)

![](dt=2026-06-02/ht=00/85b605a7afc739ac1a653c04991facb8d99c5b83c697d42de75e49b7bb3cd633.jpg)

A noticeable improvement after CLEAN has been used is the contrast between the restoration and the enamel. Because the metallic restoration has higher reflectivity than the tooth, the incident light is reflected at the air-restoration interface, and the interior of the restoration should appear dark. However, the contrast between the restoration and the enamel is poor in the dirty image because of significant artifact energy in the restoration. After application of CLEAN, the restoration-tooth interface is more clearly seen both at the surface and along the internal aspects of the restoration.

In addition, the portion of the air-enamel interface pointed to by the bigger arrow at the top of the CLEANed image is blurred and diffused in the background level in the dirty image and is clearly visible after CLEAN has

5128

APPLIED OPTICS / Vol. 40, No. 31 / 1 November 2001

![](dt=2026-06-02/ht=00/b705e7076c5605ba1e03df1cd20ccb8afc3e29bc8a00180ec131ff52f0a61720.jpg)

![](dt=2026-06-02/ht=00/2ffb81b2a9a74f6432d9bb644bea96b8af80a0622b4263b2b5868b5b3a624a3b.jpg)

![](dt=2026-06-02/ht=00/6087e98fa9f6dbfd1ac595a6e7a862f167ec241deff82bf083cd697b68bb1992.jpg)

been applied. Another noticeable improvement is in the boundary of a cavity at the tooth surface, pointed to by the smaller arrow at the bottom of the CLEANed image, which is delineated well after use of CLEAN.

The residual artifact level after CLEAN was applied is visible at several spatial locations. Because the reflectivity of the metallic restoration is strong, the signals that return from the air-metal interface are saturated at several spatial positions, and the extent of the saturation depends on the angle of incident light with the metallic restoration surface. As a result, the ratios of the peak artifact-to-main-lobe strength at these spatial positions are several decibels higher than the ratio obtained from the mirror. Therefore the artifact subtraction at these A-scan positions is not so nearly complete as that obtained at the nonsaturation positions, and it leaves larger residual artifact energy.

![](dt=2026-06-02/ht=00/152323e5026af69506e9727f754a0622d7052f1e8bd6305a1fa3ea65c8213d4d.jpg)

![](dt=2026-06-02/ht=00/87fce5e11984d047865fc20126a6dd89121f1bb637e227f32dd93529625e2778.jpg)

![](dt=2026-06-02/ht=00/8906fdb6fac8a8c89695d4ab0f94a219ebc828a8935f2c68737a3e31d4ab8be6.jpg)

# 5. Discussion

The strengths and positions of coherent artifacts caused by an imperfect source spectrum change relative to the sources and the source current levels used. In Ref. 9 it was reported that the coherent sidelobe level increases with the source current when an erbium-doped fiber amplifier is used as a source. In Ref. 10 the authors reported that, as the injection current increased, the gain of the diode increased and multiple internal reflections occurred, which resulted in an increase in sidelobe level. Figures 8(a) and 8(b) show the measured source spectrum of the EDFA used in our system.

As shown in the figures, the source spectrum has a periodic ripple that overlaps the main spectrum waveform. The interval between two ripple peaks is approximately $1\mathrm{nm}$ , which in the spatial domain corresponds to $1.29\mathrm{mm}$ . This distance is very close to the measured sidelobe position of $1.335\mathrm{mm}$ . Figure 8(c) is the Fourier transform of the source spectrum shown in Fig. 8(a), and it is the autocorrelation function of the source. The peak sidelobe level is approximately $-27\mathrm{dB}$ . Because the measured PSF from a planar mirror [Fig.

4(a)] is the convolution of the Fourier transform of the source spectrum with the transfer function of the detection system, the measured sidelobe positions and strengths are modified by the detection optics.

There are many system parameters of detection

1 November 2001 / Vol. 40, No. 31 / APPLIED OPTICS

5129

optics that can modify the PSF. We have found that the reflectivity of the mirror used in the reference arm has an important effect on artifact strength. The thickness of the mirror used in the reported experiments was $0.9\mathrm{mm}$ and a few-percent reflection from the back plate of the
mirror produced a coherent spike, which was located approximately at the source sidelobe position $[0.9\mathrm{mm} \times 1.33$ (reflective index) $= 1.2\mathrm{mm}]$ . As a result, the measured artifact strength of our system PSF was higher than the sidelobe strength calculated from the autocorrelation function of the source spectrum. However, the results that using CLEAN can reduce the artifact level to noise background and improve image contrast are independent of artifact strength.

In this paper, we describe the application of CLEAN to A-scan data obtained from a conventional OCT scanner. Therefore the CLEANed images still contain a small residual sidelobe energy caused by off-axis scattered light. Another factor that could affect the elimination of sidelobe artifacts is the nonlinearity introduced by the scanning optics. Inasmuch as the beam is scanned across the lens at the sample arm (see Fig. 3), the center beam alignment and the beam spot size vary with the galvanometer's rotation angle. Therefore a certain spatial nonlinearity is introduced into the image.

Because this nonlinearity also exists in the mirror image, we used the corresponding PSF measured at every spatial location to clean the images of the extracted teeth. However, we did not observe any visible changes in CLEANed images by using a single PSF and the corresponding PSF obtained at each spatial location.

We have described applying CLEAN to A-scan data, which constitute the envelope of the received A-scan signal. CLEAN can also be applied to complex data, and the algorithm will be more powerful in recovering targets distorted by constructive and destructive interference caused by coherent artifacts. This subject is one of our topics for further study.

The final image quality is affected by the loop gain and the stopping criteria. Because the signals cannot be distinguished if their levels are below the noise floor in the image, it is straightforward to choose the background noise level in the image as a stopping criterion when the CLEAN algorithm is used. The loop gain determines the final image quality.

Testing with loop gains of 1, 0.5, 0.25, 0.1, 0.05, 0.025, and 0.01 shows that there is a distinguishable image-quality change for loop gains of 1, 0.5, 0.25, and 0.1 but that, if loop gain is further reduced, no distinguishable image-quality change is observed. In terms of processing time, with a Pentium III 800-MHz CPU the computing times for loop gains of 1, 0.5, 0.25, 0.1, 0.05, 0.025, and 0.01 are all $\sim 11$ min. In our regular CLEAN algorithm, therefore, we choose 0.1 as the loop gain and the background noise level as the stopping criterion.

The power of the EDFA at 40-mA pumping current is $0.43\mathrm{mW}$ , which is lower than that of superlumi-

nestic light sources used by many research groups. We could increase the pumping current to deliver more power to the medium, but the source coherence length would also increase.[9] In Ref. 9, it was reported that the coherence length as well as the sidelobe level increases with the source current. When the pumping current was increased from 40 to 100 mA, the coherence length increased from 25 to $48\mu \mathrm{m}$ . As a trade-off between coherence length and source power, we chose to use a 40-mA pump current in our experiments. We have demonstrated that coherent artifacts can be removed and that image contrast is significantly improved by use of CLEAN and our source.

# 6. Summary

Coherent artifacts can severely degrade OCT image quality by introducing false targets if no targets are present at the artifacts' locations. These artifacts can add constructively or destructively to the targets that are present at the artifacts' locations. This constructive or destructive interference will result in cancellation of the true targets or in the display of incorrect echo amplitudes of the targets.

In this paper we have demonstrated that the nonlinear deconvolution algorithm CLEAN can be used to reduce coherent artifacts caused by the imperfect point spread function of the OCT system. We have modified CLEAN and adapted it to a conventional OCT system and have shown that the artifacts can be effectively reduced to the level of background noise. As a result of artifact reduction, the image contrast of the extracted tooth has been improved significantly.

The CLEAN algorithm also sharpens the air-enamel and enamel-dentin interfaces and improves the visibility of these interfaces, which will be beneficial for diagnostic applications.

We acknowledge funding support of this study by National Institutes of Health grant NIH 1R01 DE11154-03.

# References

5130

APPLIED OPTICS / Vol. 40, No. 31 / 1 November 2001

1 November 2001 / Vol. 40, No. 31 / APPLIED OPTICS

5131