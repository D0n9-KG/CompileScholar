# Multiple scattering suppression for in vivo optical coherence tomography measurement using B-scan-wise multi-focus averaging method

YIQIANG ZHU, $^{1}$ LIDA ZHU, $^{1}$ YIHENG LIM, $^{1}$ SHUICHI MAKITA, $^{1}$ YU GUO, $^{1}$ AND YOSHIAKI YASUNO $^{1,*}$

<sup>1</sup>Computational Optics Group, University of Tsukuba, Tsukuba, Ibaraki, Japan  
* yoshiaki,yasuno@cog-labs.org

Abstract: We demonstrate a method that reduces the noise caused by multi-scattering (MS) photons in an in vivo optical coherence tomography image. This method combines a specially designed image acquisition (i.e., optical coherence tomography scan) scheme and subsequent complex signal processing. For the acquisition, multiple cross-sectional images (frames) are sequentially acquired while the depth position of the focus is altered for each frame by an electrically tunable lens. In the signal processing, the frames are numerically defocus-corrected, and complex averaged.

Because of the inconsistency in the MS-photon trajectories among the different electrically tunable lens-induced defocus, this averaging reduces the MS signal. Unlike the previously demonstrated volume-wise multi-focus averaging method, our approach requires the sample to remain stable for only a brief period, approximately 70 ms, thus making it compatible with in vivo imaging. This method was validated using a scattering phantom and in vivo unanesthetized small fish samples, and was found to reduce MS noise even for unanesthetized in vivo measurement.

# 1. Introduction

Optical coherence tomography (OCT) is a noninvasive imaging technique with a resolution of a few to tens of micrometers, and has been used for clinical imaging in fields such as ophthalmology [1-4] and cardiology [5,6]. Because of its long imaging depth, OCT recently has been adopted for noninvasive and nondestructive microscopy [7]. OCT can visualize the deep tissue region at around several hundred of micrometers to a few millimeters from the sample surface, and it is significantly larger than that of conventional microscopy, which is only around a few tens to hundreds of micrometers.

Conventionally, the imaging depth of OCT was believed to be dominated by two factors, the depth of focus of the probe optics and the scattering of the sample. To overcome the former limit, several methods have been successful. For instance, the fusion of multiple images with several depth focus positions [8], computational refocusing methods [9-12], and the combination of complex signal processing and focus fusion such as Gabor-domain OCT [13, 14]. To overcome the tissue-scattering limit of the imaging depth, longer wavelength probe has been applied.

In general, a $1.3 - \mu \mathrm{m}$ wavelength probe has a better image penetration than a $830\mathrm{-nm}$ or visible OCT. For retinal imaging, a $1.0 - \mu \mathrm{m}$ probe was shown to have a higher penetration [15-18], and it has become a common probe wavelength band for clinical retinal imaging.

More recently, even longer wavelength, such as $1.7\mu \mathrm{m}$ , has been used to investigate samples with high scattering, such as cardiovascular tissues [19] and brain tissues [20], and have demonstrated higher penetration than OCT using shorter wavelengths, such as more than $3\mathrm{mm}$ in coronary arterial tissues or around 1.3 to $1.6\mathrm{mm}$ in brain tissues.

Because these two limiting factors of imaging depth have been overcome, the multiple scattering (MS) has gradually been recognized as an additional limiting factor. In general, OCT imaging theory is based on the assumption that most MS photons are rejected by a confocal pinhole (i.e., a single-mode fiber tip) [21] and the single-scattering (SS) photons govern the imaging.

arXiv:2404.01811v2 [physics optics] 12 Dec 2024

In practice, however, some MS photons are captured through the confocal pinhole and appear in the image. Because an MS photon has a longer optical path than that of the SS photons, a photon that undergoes multiple scattering at a specific depth appears in the image to be located at a deeper location. In contrast, the contribution of SS photons becomes less at the deeper depth in the image. Hence, deeper regions in an OCT image are more dominated by MS photons. This dominance of MS photon degrades the resolution and contrast of the OCT image in the deep regions [22]. In addition, it degrades the quantitative measurement capability of functional OCT, such as polarization sensitive OCT [23].

Several methods have been used to mitigate the MS effect. For example, Badon et al. proposed the smart OCT method, which modulates the pupil using a spatial light modulator (SLM) and pre-defined reflection matrix [24]. Borycki and associates used SLM-based pupil modulation and spatial correlation theory to reduce the MS effect (or equally, the coherent cross-talk) of full-field swept-source OCT [25-27]. Liu et al. proposed the aberration-diverse method for MS suppression [28].

In this method, intentional astigmatism was introduced using a deformable mirror, and multiple, typically twelve, OCT volumes were acquired with different astigmatism axes. After correcting the astigmatism using computational adaptive optics [29], the volumes were coherently averaged. The paths of the MS photons are not consistent across the different astigmatism axes, whereas those of the SS photons are consistent. Hence, the coherent average can reduce the MS-photon contributions.

Although the aforementioned modalities have successfully mitigated the MS effect, they require expensive wavefront manipulation devices, such as an SLM or deformable mirror. We previously proposed the multi-focus-averaging (MFA) method, which is based on an approach similar to that of the aberration diverse method, but we used a cost-effective electrical tunable lens (ETL) [30]. Multiple, typically seven, OCT volumes are acquired with different defocus positions, and the volumes are coherently averaged after correcting the defocus using computational defocus correction. This method was also applied to Jones-matrix based PS-OCT (JM-OCT), and mitigation of polarization artifacts was demonstrated [23].

Although the aberration diverse method and MFA perform well on static samples, such as static phantom and postmortem samples, its application to in vivo measurement still poses a great challenge. Because these methods rely on coherent (i.e., complex) averaging of multiple volumes, the phases of these volumes should be consistent. In other words, the sample should be highly stable during the multiple volumetric acquisition, which usually takes a few tens of seconds.

In this work, we propose a new version of the MFA method, referred to as the B-scan-wise-MFA (B-MFA) method for MS suppression in in vivo measurement. This new method sequentially acquires multiple cross-sectional OCT frames, instead of volumes, with different defocus. The defocus of each frame is corrected by applying a one-dimensional (1-D) version of computational refocusing, and all defocus-corrected frames are coherently averaged to reduce the MS signals.

Note that throughout the manuscript, a single cross-sectional scan is denoted as a frame, while a B-scan refers to a set of frames acquired at the same location. This method requires phase stability only during the acquisition time of a few frames (i.e., a B-scan), not of volumes, so the required stable time is typically less than $100\mathrm{ms}$ . Hence, this method is applicable to in vivo measurement. The performance of the B-MFA method was validated by measuring a scattering phantom and in vivo small fish.

We also discuss the optimization of parameters for the B-MFA method, the artifacts related to 1-D computational refocusing, and their impact on the final images.

# 2. Principle and implementation of B-MFA method

# 2.1. Data acquisition and signal-processing flow

Before describing the details of the B-MFA method, we first present an overview of this method. Th
e B-MFA method is based on the assumption that the trajectories of MS photons are not

![](dt=2026-05-29/ht=23/181ef5c6b1bdbf92a7a9950fe2624f4322744099dd5cdafbd8032adc88a17625.jpg)

![](dt=2026-05-29/ht=23/479d0a963232826d5d4a3c218a75b6a5784dd9bd68580116c2fd3c12680e18b5.jpg)

consistent if the depth positions of the focus are different [30], as depicted in Fig. 1. Hence, the first step of B-MFA is to acquire multiple frames with different focus positions using an ETL. The details of the data acquisition protocol are described in Section 2.2. In the second step, we correct the defocus using a 1-D phase-only spatial frequency filter (Section 2.3). Finally, the frames are complex averaged after the axial shifts and phase offsets of the frames have been corrected. Because we used a JM-OCT in our particular implementation, we additionally correct the bulk-phase offset in the four polarization channels of the JM-OCT. The details of the shift and phase corrections as well as the complex-averaging process are described in Section 2.4

Because MS-photon trajectories are different in frames with different focus depths, the randomized MS signal is reduced by the complex averaging, whereas the defocus corrected SS-signal is not. In this manuscript, we refer to this complex-averaged image as a "B-MFA image." For volumetric measurement, we sequentially acquire cross-sectional B-MFA images at several slow-scan positions.

# 2.2. Data acquisition protocol

The first step of the B-MFA imaging is to acquire multiple frames with different depth positions of focus. Here the depth positions of the focus are actively controlled by an ETL that is equipped in the sample arm of the OCT [30]. The implementation details of the OCT system are described later in Section 3.1.

The data acquisition protocol of B-MFA is summarized in a schematic time chart in Fig. 2(a). As shown in this diagram, multiple frames are sequentially acquired at each slow-scan (i.e., B-scan) position. For every frame acquisition, the ETL updates the depth position of the focus so that all frames are acquired using different focus positions. This measurement process is repeated for each slow-scan location to obtain a volumetric dataset.

In our typical implementation, a single frame acquisition time including the focus transition time of ETL is $13.4\mathrm{ms}$ and the number of frames at a single slow-scan location (i.e., the number of frames per B-scan) is five (see Section 3.2 for details). Hence, the typical acquisition time for a single slow-scan location is $67~\mathrm{ms}$ .

![](dt=2026-05-29/ht=23/6ab29d6a604f7bc7189069e2a884dbc05b6c07c4fd89be09f29438c22ecbb8a7.jpg)

![](dt=2026-05-29/ht=23/3ca2682af37955f40a3a3538c2848e410bc3c342a54dcac473947e9cc3596c8a.jpg)

# 2.3. Computational refocusing

# 2.3.1. Computational refocusing using a 1-D phase filter

After data acquisition, each frame is then processed for 1-D computational refocusing. Here, the 1-D lateral complex signal at each depth of the frame is processed by a 1-D phase-only spatial frequency filter designed based on the Fresnel-diffraction model [10]. For a defocus distance (i.e., the distance from the focus to the imaging depth) of $z_{d}$ , the phase only filter is

$$
H ^ {- 1} \left(f _ {x}; z _ {d}\right) = \exp \left(\frac {- i \pi \lambda_ {c} z _ {d} f _ {x} ^ {2}}{2}\right), \tag {1}
$$

where $f_{x}$ denotes the spatial frequencies corresponding to the fast-scan lateral position $x$ , and $\lambda_{c}$ is the center wavelength of the probe beam.

The refocused frame $S'(x;z)$ is obtained using this filter and two sequential 1-D Fourier transform operations as

$$
S ^ {\prime} (x; z) = \mathcal {F} _ {x} ^ {- 1} \left[ \mathcal {F} _ {x} [ S (x; z) ] H ^ {- 1} \left(f _ {x}; z _ {d} (z)\right) \right], \tag {2}
$$

where $S(x;z)$ is the original complex frame and $\mathcal{F}_x[\quad]$ and $\mathcal{F}_x^{-1}[\quad]$ are the 1-D Fourier transform and its inverse Fourier transform, respectively, along the fast-scan $(x)$ direction. Here, $z_d$ is considered to be a function of the depth in image $z$ . In our implementation, $z_d$ is estimated from the measured data, as described in detail in the next section (Section 2.3.2).

Note that this method corrects the defocus only along the fast-scan direction. The impact of this limitation is discussed in Section 5.3.

# 2.3.2. Estimation of the defocus distance

In our method, the defocus distance $z_{d}$ is estimated from the measured OCT images, where the information entropy of the OCT images is used as a sharpness metric. Note that here, we use the information entropy of an 2D en-face OCT image at each depth, although the refocusing was performed for each lateral ( $x$ -) line individually. This is because a 1-D lateral signal is not informative enough to compute a reliable sharpness metric.

The defocus-distance estimation is performed at each depth, and hence these initial estimates are obtained as a function of depth. We then extract the estimates from the depth region in which the estimated focus distances are linear to the depth, and use them to estimate the defocus distances over the range of all depths in the image. Specifically, the estimated defocus distance is linear-fit to the depth using an intensity-weighted linear regression. This linearly fitted line gives the final estimates of the defocus distance throughout the whole depth range.

# 2.4. Shift and phase-offset corrections and complex averaging in JM-OCT

In our implementation, we used JM-OCT, which acquires four OCT cross-sectional images at each slow-scan position [31] that corresponds to the four polarization entries of the Jones matrix. Here, we describe the methods to correct the mutual phase offsets in the four images in addition to the shift and phase-offset corrections in the multiple frame acquisitions because they are crucial to complex averaging the refocused OCT signals.

A Jones matrix cross-sectional image (JM cross-section) consists of four complex OCT images corresponding to the four polarization channels. For computational refocusing, we estimate the defocus distance using only one polarization channel with the method described in Section 2.3.2, and we apply it to all polarization channels. Namely, the defocuses of all the polarization channels are corrected with the same estimated defocus distance.

Because of the deformation of the ETL, there are non-negligible depth shifts among the images taken with different defocus. We corrected the depth shifts as summarized in the diagram presented in Fig. 3. We compute the shift using the cross-correlation function of linear-intensity OCT images (blue box in the diagram), where the linear intensity images are obtained from a single polarization channel and the intensity image of the first defocus value is used as the reference.

Before computing the cross-correlation function (Step 2 in the diagram), the images are up-sampled four times along the depth using Fourier-domain zero-padding to achieve sub-pixel accuracy (Step 1). The cross-correlation function is computed along the depth by a direct method (i.e., not the Fourier-domain method). The amount of shift is determined from the peak of the cross correlation function (Step 3), and the shifts of all polarization channels are corrected by using this estimated shift amount (Step 4).

After correcting the shift in the up-sampled images, the images are down-sampled to the original pixel resolution using Fourier-domain de-padding (Step 5). And finally, this process is repeated for all JM cross-sections of all defocused volumes.

After correcting the axial shifts, the mutual phase offset
s in the images with different defocus are estimated and corrected. Here, we consider the phase offsets in the JM cross-sections, where a JM cross-section is a set of four complex OCT images one for each of the four polarization channels (PCs). We first estimate the phase offsets of the images at each PC independently. For this estimation, we use only the depth region of 30 pixels from the sample surface (see Appendix A for details of the sample surface detection).

The first JM cross-section is used as a reference, and the phase offsets between the complex images of the reference JM cross-section and the images of the target JM cross-section are estimated. The complex image of each PC in the target JM cross-section is multiplied by the complex conjugate of the corresponding PC's complex image in the reference JM cross-section (Step 1 in Fig. 4), and then complex averaged along the depth (Step 2).

Because this operation is performed for each of four PCs, four 1-D complex arrays along the fast scan direction are given (the phase offset of each PC in the figure). By averaging these four 1-D complex arrays corresponding to the PCs and taking the phase, we obtain a 1-D

![](dt=2026-05-29/ht=23/13916d2fd1a3ae3373656421b6db56e28ecf7cab41fcb7d63cc2302e6dd2df95.jpg)

array of the phase offsets where each entry of the array represents the phase offset of each A-line. Finally, the mutual phase shift of the reference JM cross-section and the target JM cross-section is corrected for all depths by subtracting the estimated phase offsets from the complex images in the target JM cross-section, i.e., multiplying the conjugate phase in complex (Step 3). This operation is repeated for all frames (i.e., JM cross-sections) so that the phase offsets of all the frames are corrected (Step 4).

After the phase offset corrections, the OCT images of each PC are complex averaged over the frames to generate four MS-reduced complex OCT images, which forms a MS-reduced JM cross-section. The final B-MFA image is obtained by averaging the squared intensities of four complex images of the MS-reduced JM cross-section.

Note that the aforementioned operations are for one slow-scan position, i.e., B-scan position. We repeat these operations for all other slow-scan positions to obtain a B-MFA volume.

# 3. Validation design

# 3.1. JM-OCT setup

A custom-built passive-polarization-delay (PPD) based JM-OCT was used to evaluate the B-MFA method. A PPD module splits the probe beam into two orthogonal polarizations and

![](dt=2026-05-29/ht=23/8699a71e74d46f03edb5046ea0f5a65051be9c6e600100c0440a9bcdb8c01518.jpg)

applies different delays to them. In addition to this probe-beam polarization multiplexing, polarization-diversity detection is used. Hence, four complex OCT images corresponding to the four polarization channels (i.e., two multiplexed polarizations of the probe beam times two detection polarizations) are obtained.

The center wavelength and scanning bandwidth of the light source (AXP50124-8, AXSUN, MA, USA) are $1,310\mathrm{nm}$ and $106\mathrm{nm}$ , respectively. The effective focal length of the objective lens (LMS03, Thorlabs, NJ, USA) used in the system is $36\mathrm{mm}$ . The lateral and axial resolutions are $17\mu \mathrm{m}$ (in $1/e^2$ -width) and $14\mu \mathrm{m}$ (in full-width-half-maximum) in tissue, respectively. The depth-of-focus (DOF) without the ETL is $0.36\mathrm{mm}$ . The A-line rate of the system is 50,000 A-lines/s. More details of the JM-OCT principle [31] and the implementation of the particular JM-OCT system used in this study [32, 33] can be found in elsewhere.

An ETL (EL-10-30, CI-NIR-LD-MV, Optotune, Switzerland) is used in the sample arm to axially shift the focus position. The details of the sample arm equipped with the ETL are described in [30]. As we investigated in the previous study, the focus modulation of the ETL alters the lateral resolution only by a negligible amount, i.e., less than $0.5 \mu \mathrm{m}$ (Section 4.3.4 of Ref. [30]).

Note that, although we used a JM-OCT in this study, the B-MFA method can be applied to

standard (i.e., non-polarization-sensitive) OCT.

# 3.2. Measurement protocol

At each slow-scan (i.e., B-scan) position, five continuous cross-sectional frames were acquired as the defocus was incremented $0.18\mathrm{mm}$ at each acquisition. The defocus increment was equivalent to half the DOF, where the DOF is that without the ETL. The total focus shift of five frames was $0.72\mathrm{mm}$ . These parameters were determined by an optimization experiment that is described in details in Section 5.1. Each cross-sectional frame consists of 256 A-lines, and the acquisition time of a single frame was $5.12\mathrm{ms}$ .

By accounting for the defocus transition time of ETL (7.5 ms) and the pullback time of the galvanometer scanner (0.8 ms), the five continuous frames were acquired in $67.1\mathrm{ms}$ . The phase should be stable during this acquisition time. The acquisition was repeated for 256 slow-scan positions, and the total time to acquire a volume was $17.18\mathrm{s}$ . In summary, the volumetric acquisition time and required phase-stable duration were $17.18\mathrm{s}$ and $67.1\mathrm{ms}$ , respectively.

For reference, we acquired or generated three additional volumetric images. The first image is a single acquisition image, which was made by extracting only the third frame of the five sequential frames acquired for B-MFA. Although no averaging was performed, the defocus was corrected.

The second reference is the single frame averaging (SFA) image. Here, we acquired a volume following the B-MFA protocol but without shifting the focus. This volume was processed in a manner identical to that of the B-MFA, i.e., the defocus correction, shift and phase corrections, and complex averaging were all the same.

The third reference is a standard MFA image [30]. Here, five OCT volumes were sequentially acquired with different defocus, as shown in Fig. 2(b). The increments in defocus between two consecutive volume acquisitions were the same as that of the B-MFA. Each volume was acquired with a standard raster scan with $256 \times 256$ A-lines, and the total acquisition time of the sequential five volumes was $9.92\mathrm{s}$ . For MFA, the phase should be stable during the five volume acquisitions, and hence it was $9.92\mathrm{s}$ , which is 148 times longer than the required phase-stable time of B-MFA.

For all scan protocols, the lateral scanning range was $1.5\mathrm{mm}\times 1.5\mathrm{mm}$ . The lateral field was covered with $256\times 256$ lateral sampling points, which yielded a lateral pixel separation of 5.86 $\mu \mathrm{m}$ that is around $1 / 3$ of the lateral resolution.

# 3.3. Samples

A scattering phantom and ten in vivo medaka fish were measured to evaluate the proposed method. The scattering phantom consists of four parts, two cover glasses, a scattering layer, and a glass plate under the scattering layer that creates a space without a back scattering signal. The scattering layer is a mixture of $0.04\mathrm{-mL}$ polystyrene microparticles (diameter of $10~\mu \mathrm{m}$ , 72986-10ML-F, Sigma-Aldrich, MO, USA) and $0.5\mathrm{-mL}$ ultrasound gel (pro Jelly, Jex, Japan). This mixing ratio results in a particle concentration of $13.5\times 10^{3}$ particles/ $\mathrm{mm}^3$ . The schematic and photograph of the phantom are shown in Fig. 5(a) and (b).

The medaka, also known as the Japanese rice fish, is a small fish similar in size to a zebrafish and is widely used as a model animal in biological research. We measured ten in vivo medakas without anesthesia. A fish was placed in a 3-D printed container with a water-filled groove that was $5\mathrm{mm}\times 8\mathrm{mm}\times 31\mathrm{mm}$ (height $\times$ width $\times$ length) in s
ize. We placed a thin glass slip above the container to prevent the fish from accidentally jumping out during the measurement. Figure 5(c) shows a photograph of the container and a sample. In this figure, the red box roughly indicates the scan area.

The protocol of the fish experiment follows the animal experiment guidelines of the University of Tsukuba and is approved by the Institutional Animal Care and Use Committee of the University of Tsukuba.

![](dt=2026-05-29/ht=23/6d39978c271a6032f255032d4684ad3963d8b0ac0860c551faac20442485bd68.jpg)

![](dt=2026-05-29/ht=23/e8a558a1d523fbf51d819fae6a63bf32e0e4abf2a57d56729b1750cd5e093ef9.jpg)

![](dt=2026-05-29/ht=23/2c9f18e0471438a27cc81ea191801d24bd32a43fe22cd366a0c3596b21c9475c.jpg)

# 4. Result

# 4.1. Scattering phantom

Figure 6 shows the intensity cross-sectional images of the scattering phantom for the single acquisition, SFA, MFA, and B-MFA methods from left to right. The images in the second row are magnified views of the deep regions, indicated in the images in the first row. In the shallow regions of the sample, the standard MFA image has fewer particles than the other images [Fig. 6(c)]. This difference can be attributed to the fact that only the standard MFA uses 2-D computational refocusing, whereas the others use 1-D refocusing. This is discussed in detail in Section 5.3.

In the deep regions (indicated by the green boxes), the B-MFA and MFA images [Fig. 6(h) and (g), respectively] show particles with higher contrast than the single acquisition and SFA images [Fig. 6(e) and (f)], as indicated by the yellow arrows. Inside the glass plate region, the B-MFA and MFA images [Fig. 6(h) and (g), respectively] have the lowest noise intensity (indicated by the blue arrows), whereas the single acquisition and SFA images [Fig. 6(e) and (f)] have the highest and intermediate noise intensities, respectively.

This may indicate that standard OCT measurement noise has been mitigated in the SFA image because of the complex averaging, while both the measurement noise and MS signal are mitigated in the MFA and B-MFA images.

This noise and MS suppression are more clearly and quantitatively shown in the averaged intensity depth profiles in Fig. 6(i). Here, the central 216 A-lines [indicated by the orange brace

![](dt=2026-05-29/ht=23/3062cf903a3285a8ce31e08738c07c59765da8abfc52b5c0343270bc1b7c76ea.jpg)

in Fig. 6(a)] were averaged. The black arrow indicates the surface of the scattering layer. In the superficial region of the scattering layer, the single acquisition (blue), SFA (gray) and B-MFA (orange) curves are similar to each other. In the deeper regions (the purple-background region in the plot), the B-MFA and MFA curves show intensities that are lower than those of the single acquisition and SFA curves by around 2 to $3\mathrm{dB}$ . The glass plate region is indicated by light blue background in the plot.

At the surface of the glass plate (blue arrow), the single acquisition and SFA curves show higher intensity than those of MFA and B-MFA due to overlapping MS signals. Near the superficial depth in the glass (indicated by the blue brace), the single acquisition image (blue line) yields the highest intensity. The SFA (gray line) shows lower intensity than the single acquisition (2.02-dB reduction in average). This could be attributed to the reduction in measurement noise caused by the complex averaging.

The B-MFA (orange line) shows a further reduction (2.07 dB with respect to SFA and 4.09 dB with respect to the single acquisition in average). This may indicate additional suppression of the MS signal. The MFA (green line) shows the lowest signal values, i.e., the best reductions in measurement noise and MS-signal suppression (0.58 dB with respect to B-MFA and 4.66 dB with respect to single acquisition in average).

# 4.2. In vivo small fish sample

Figure 7 shows the intensity cross-sections of one of the ten fish samples. The images are, from top to bottom, single acquisition, SFA, MFA, and B-MFA images, respectively. Enlarged images of the blue and orange regions are shown to the right of the cross-sectional images. Several anatomic features, such as a layered hyperscattering structure in the muscle region (enlarged in the blue dashed boxes) and a dark region surrounded by hyperscattering, which may indicate a notochord (enlarged in orange dashed boxes) are visible in all images, but these features are most clearly visualized in the B-MFA image.

Figure 7(m–p) shows the en-face images at a deep depth indicated by the yellow dashed line in Fig. 7(a). The B-MFA image [Fig. 7(p)] exhibits a lower signal intensity but higher contrast than the single acquisition and SFA images [Fig. 7(m) and (n), respectively]. These findings may suggest that the B-MFA method reduces the MS signal and improves the image contrast. We also notice that the en-face MFA image [Fig. 7(o)] shows reduced signal intensity but does not show improved contrast. We suspect that this reduced intensity in the MFA image is not fully due to the MS reduction, but is also due to signal washout caused by the motion of the sample. Details are discussed in Section 5.4.

It might be noteworthy that, since lateral motion is not observed in the en-face images, we reasonably assume that the cross-sectional images in Fig. 7 were taken at the same anatomical position for all methods. In addition, we can consider that "refocus artifact," which is discussed in detail later in Section 5.3, is not remarkable according to these en-face images.

The intensity cross-sectional image of the other nine fish are presented in the Supplementary Material (Fig. S1-S3).

To quantitatively compare the sharpness of the images, we computed the information entropy of the en-face slab projections. The projections were computed at three depth regions indicated by the pink braces in Fig. 7(a). The computed information entropy values are summarized in Table 1, where the depths are relative to the zero-delay depth. At all three depth regions, the B-MFA shows the smallest information entropy. Namely, B-MFA provides the sharpest image of the four methods. For reference, the en-face projection images used to compute the information entropy are summarized in Fig. 8. These images may provide an intuitive understanding of the correspondence between the information entropy values and image sharpness.

We performed the same information entropy analysis on the other nine fishes as summarized in Fig. 9. Here, the three depth regions (i.e., upper, middle, and lower) of each fish were selected to cover the same structures as the case of the first fish. Since the information entropies may

![](dt=2026-05-29/ht=23/e6cbc37b05e91dbe7aeb67760e46e031aa41dddff97da38c38a4e227d32929f1.jpg)

![](dt=2026-05-29/ht=23/9b3415108291c4d7abdf3e324b5861b2b102b9c916342e17f12f2ccdb765cbd3.jpg)

![](dt=2026-05-29/ht=23/2d7baa44eb0bb57f92283fa021c0319e2b172864b755b6390a1d17219672fbe6.jpg)

![](dt=2026-05-29/ht=23/c1179623e1188b02c96333d0f2dbef9ed459eec23392759f044e3a0cdec3c9e1.jpg)

![](dt=2026-05-29/ht=23/66bacf292f5494c4637161a68140951de90f8e29fea288ee786ca982b07e4696.jpg)

Table 1. Information entropy of the en-face slab projections of the small fish. The depths of (1)-(3) corr
espond to the depths indicated in Fig. 7(a) (braces). For all depths, B-MFA shows the smallest entropy, which indicates the sharpest image.

![](dt=2026-05-29/ht=23/d459e85b7e10420bfa4c6cb758708fc795041e916fb7538e6824318cb05c2bd4.jpg)

<table><tr><td rowspan="2">Depth (mm)</td><td colspan="4">Information entropy</td></tr><tr><td>Single acquisition</td><td>SFA</td><td>Standard MFA</td><td>B-MFA</td></tr><tr><td>(1) 1.05-1.12</td><td>4.62</td><td>4.60</td><td>4.57</td><td>4.56</td></tr><tr><td>(2) 1.41-1.48</td><td>4.18</td><td>4.16</td><td>4.08</td><td>4.06</td></tr><tr><td>(3) 1.77-1.85</td><td>4.52</td><td>4.42</td><td>4.39</td><td>4.29</td></tr></table>

vary among the fishes due to the particular structures of each fish, we computed the information entropy difference from the single acquisition for each fish. In the figure, small circles indicate the means among the ten fishes, i.e., the additional nine fish and the first fish presented in Table 1, while the whiskers indicate the standard deviation among the ten fishes.

Although MFA shows large standard deviations, SFA and B-MFA show reasonably small standard deviations. In comparison to the mean information entropies of SFA and MFA, B-MFA showed smaller information entropies, indicating higher image sharpness. By considering the relatively small standard deviations of B-MFA to the mean information-entropy reductions, the reductions (i.e., the sharpness enhancements) are not trivial.

To quantitatively evaluate the image contrast, we computed the signal-to-signal ratio (SSR). SSR is defined as the mean intensity ratio between two manually selected ROIs, where one ROI was chosen to include a high-scattering-intensity structure [dark blue box in Fig. 7(a)] and the other was selected in a low scattering region [red box in Fig. 7(a)] in a cross-sectional image. The size of each ROI was 35 pixels $\times$ 16 pixels $(205.1\mu \mathrm{m}\times 115.8\mu \mathrm{m})$ . Examples of the SSRs are shown in Fig. 7(a-d).

The SSRs of the SFA, B-MFA, and MFA images were compared with that of the single acquisition image by computing the SSR enhancement (SSRE), which is defined as the difference in SSR from that of the single acquisition image. For the images presented in Fig. 7, the SSREs were $0.42\mathrm{dB}$ (SFA), $1.37\mathrm{dB}$ (B-MFA), and $0.36\mathrm{dB}$ (MFA). Namely, of the four images, the B-MFA image provides the largest SSRE.

The SSREs of all ten fish were computed using ROIs that include the same anatomical structures for all fish and plotted in Fig. 10. The B-MFA achieved the best mean SSRE of 1.82 dB, which is significantly larger than that of SFA (SSRE = 0.72 dB, p = 0.0008) and MFA (SSRE = -0.52 dB, p = 0.0009). The statistical comparison was done using paired t-tests. It is noteworthy that although the mean SSRE of B-MFA was only 1.82 dB (i.e., × 1.52), the images in Figs. 7 and S1 to S3 demonstrated evidently superior observational contrast of B-MFA images compared to the single acquisition images.

We note that the MFA method showed the largest interquartile range. This is because the MFA is more susceptible to sample motion, and the motions of the in vivo fish samples highly varied from case to case. This susceptibility of MFA to the sample motion indicates that the B-MFA might be the best method for in vivo small fish samples.

# 5. Discussion

# 5.1. Scan protocol optimization

To determine the optimal measurement protocol for B-MFA, five focus shifting steps $(\Delta z)$ and several values for the number of total frames per B-scan (N) were examined. The results for the

![](dt=2026-05-29/ht=23/992d32d861230f2e4311bac784af510764371d365433813e55334a6151bbbd7c.jpg)

Table 2. Summary of B-MFA measurement protocols used for the scan-protocol optimization. We used five focus shifting steps $(\Delta z)$ and several values for the number of frames to be averaged (N). The total focus shifting distance (D) was defined from $\Delta z$ and N.

![](dt=2026-05-29/ht=23/1213d71612fa6e739f151947822ee57cd9416313bc27f561db41c2e057d5c7b9.jpg)

<table><tr><td>Measurement protocol</td><td>Δz</td><td>N</td><td>D = (N - 1) × Δz</td></tr><tr><td>1</td><td>1 DOF (0.36 mm)</td><td>1, 2, 3, 4</td><td>0, 0.36, 0.72, 1.08</td></tr><tr><td>2</td><td>1/2 DOF (0.18 mm)</td><td>1, 2, 3, ..., 7</td><td>0, 0.18, 0.36, ..., 1.08</td></tr><tr><td>3</td><td>1/3 DOF (0.12 mm)</td><td>1, 2, 3, ..., 10</td><td>0, 0.12, 0.24, ..., 1.08</td></tr><tr><td>4</td><td>1/4 DOF (0.09 mm)</td><td>1, 2, 3, ..., 13</td><td>0, 0.09, 0.18, ..., 1.08</td></tr><tr><td>5</td><td>1/6 DOF (0.06 mm)</td><td>1, 2, 3, ..., 15</td><td>0, 0.06, 0.12, ..., 0.84</td></tr></table>

focus shifting step, total frame number, and total focus shifting distance $\mathrm{D} = (\mathrm{N} - 1) \times \Delta z$ are summarized in Table 2. A scattering phantom (Section 3.3) was measured using all protocols.

To evaluate the image contrast, the signal-to-background ratio (SBR) was computed. Here the "signal" was defined as the mean signal intensity of five manually selected particles in a scattering region of the phantom. The particles were selected at a depth 10 pixels below the top surface depth of the glass plate as schematically indicated by the pink dashed line in Fig. 11(a), and each of the five particles were selected from different cross-sectional image in a volume.

Each scatterer was cropped by a $3 \times 3$ pixels $(17.6 \times 21.7\mu \mathrm{m})$ window in a cross-sectional image, as indicated by the red boxes in Fig. 11(b), and the mean intensity of this window was used as the intensity of that scatterer. The background was defined as the mean signal intensity of an ROI located at the same depth as the particles but in the glass plate, i.e., a region without scattering. The ROI extends over 60 pixel $\times 3$ pixel $(351.5\mu \mathrm{m} \times 21.7\mu \mathrm{m}$ , lateral times depth), and is indicated by the dashed light blue box in Fig. 11(b).

Because there should be no scattering

![](dt=2026-05-29/ht=23/7c9a90ab4b711d0e884de9a1ceda9f6f3509cf5ba7c3d911a5ac196264ab4891.jpg)

in the glass, the background intensity is used as a measure of the MS signal.

The SBR of each protocol is shown in Fig. 12 where each symbol corresponds to each focus shifting step $\Delta z$ . The plot demonstrates that the SBR is highest at $D = 0.72 \mathrm{~mm}$ , which is twice the DOF. At this D, the SBRs are 28.74, 28.76, 28.63, 28.52, and 27.54 dB for $\Delta z = 1/6$ , 1/4, 1/3, 1/2, and $1 \times$ DOF, respectively. Here, the numbers of complex averaged frames (N) of each protocol are 13, 9, 7, 5, and 3, respectively. Among these five protocols, the first four give similarly high SBR, and they can be the candidates for the optimal protocol.

Because B-MFA is used for in vivo measurements, a shorter acquisition time is preferable. Therefore, we have used the protocol of $(\Delta z = 1/2$ DOF, $N = 5$ , $D = 2$ DOF) for the measurements shown in the Section 4.

Note that this protocol was selected to best suit small fish samples. For other types of samples, it could be worth optimizing the protocol again to adapt it for the new samples.

# 5.2. Phase stability requirements of MFA and B-MFA

The previously proposed MFA method used a 2-D computational refocusing [34], in which the complex en-face OCT signal at a depth is two-dimensionally Fourier-transformed and 2-D quadratic phase is applied in the spatial-frequency domain. Hence, the phase of the OCT signal should be stable over the volume or at least over several frames that cover an area larger than the lateral resolution.

In contrast, B-MFA uses 1-D computational refocusing as descri
bed in Section 2.3.1. Hence, the phase stability is required only within a frame.

Because the sample motion can cause significant phase error, the lower requirement for the phase stability of the B-MFA is an important advantage for in vivo measurement.

![](dt=2026-05-29/ht=23/10f70140aff9ad75e3ff7ee1a94d7f1f9a6c7fad4039947c54246559e6d9d50f.jpg)

# 5.3. Refocus artifact of B-MFA and conventional MFA in the phantom result

Comparing the B-MFA and MFA cross-sectional images [Fig. 6(d) and (c), respectively] at the superficial region scattering layer, we notice that the B-MFA image exhibits more scattering particles than the MFA image. This difference in the numbers of particle can be attributed to the difference in the 1-D and 2-D computational refocusing. Because 1-D refocusing refocuses the image only along the fast-scan direction, the refocused particle signal must be elongated along the slow-scan (vertical) direction, as shown in the en-face slice of the same data [Fig. 13(a)].

Hence, the images of the scatterers that are not really in a particular B-scan smear into the B-scan, and this causes an artifactual increase in the number of scatterers in that B-scan. On the other hand, the 2-D refocusing used in MFA isotropically refocuses the image as shown in the corresponding en-face slice [Fig. 13(b)]. Hence, the artifactual increase of the scatterer does not occur.

A similar artifact also can be seen in the small fish image shown in Fig. 13(e) and (f), although it is less evident than in the phantom case, because the tissue microstructure is aligned roughly along the slow-scan direction.

In the other depth of the phantom, which is around $1.5\mathrm{mm}$ from the surface, the elongation artifact is negligible [Fig. 13(c) and (d)] because the physical focus is located near this depth.

To solve this problem in the future, we may be able to adopt a computational refocusing method [35, 36] for the MFA or B-MFA method that is less susceptible to phase instability.

# 5.4. Signal reduction of the MFA in the in vivo result

In the in vivo fish measurements, the MFA image [the third row of Fig. 7] showed lower signal intensity than the B-MFA image [the fourth row of Fig. 7]. In addition, the SSRE of MFA is smaller than that of B-MFA for the in vivo measurement (ninth paragraph of Section 4.2). This can be attributed to the signal washout caused by the sample motion, which reduces not only the MS signal but also the SS signal. These findings also emphasize the advantage of the B-MFA method for in vivo imaging over the MFA method.

![](dt=2026-05-29/ht=23/4c818ef88576f2c3d29396034c10ab195ef503ea9a7c2416d9981472d0e64922.jpg)

# 5.5. Heartbeat and respiration of fish

The heart and respiratory rates of medaka fish are approximately $2.3\mathrm{Hz}$ and $4.9\mathrm{Hz}$ , respectively [37]. These correspond to heart and respiratory periods of $0.43\mathrm{s}$ and $0.20\mathrm{s}$ , respectively. On the other hand, the B-MFA acquisition of a single cross-section (i.e., acquisition time for five consecutive frames) is $67.1\mathrm{ms}$ (Section 3.2), which is 6.5 times and 3 times shorter than the heart and respiratory periods, respectively. This fact further supports the feasibility of in vivo application of B-MFA.

However, this comparison between the B-MFA acquisition time and the heart and respiratory periods also suggests that some B-MFA cross-sections may have been affected by sample motion, causing signal reduction. Overcoming this issue remains a future challenge.

# 5.6. Extensions of B-MFA-based OCT

# 5.6.1. Functional OCT and B-MFA

One limitation of B-MFA is its incompatibility with functional OCT imaging techniques based on fast sequential acquisition, including Doppler OCT [38-40], OCT angiography [41, 42], and certain types of dynamic OCT (DOCT) [43-45]. This limitation stems from B-MFA's requirement for the sequential acquisition of a few frames followed by complex averaging. As a result, the effective frame rate of B-MFA cross-sectional images is reduced, and the complex averaging process can wash out not only the MS signal components but also the dynamic signal components.

On the other hand, certain types of DOCT methods employ a relatively slow time sequence of OCT images. For instance, volumetric logarithmic intensity variance (LIV) and volumetric OCT correlation decay speed (OCDS) methods use a sequence of 32 OCT frames with a frame interval of $204.8\mathrm{ms}$ [46]. Given that the acquisition time of a cross-sectional B-MFA image (i.e., the acquisition time of a set of five frames) is only $67.1\mathrm{ms}$ (as discussed in Section 3.2), B-MFA is theoretically compatible with LIV- and OCDS-based DOCT. In this scenario, relatively fast dynamics may be washed out, but slow dynamics can still be detected to some extent. Therefore, experimental validation of B-MFA-based dynamic OCT could be a promising avenue for future research.

B-MFA may also be compatible with static functional OCT, such as polarization-sensitive OCT. Although we used the JM-OCT system in this study without exploiting its polarization sensitivity, a minor modification of the signal processing may enable polarization-sensitive B-MFA imaging. It is worth noting that Lida Zhu et al. have demonstrated MFA-based polarization ex vivo imaging and showed that MFA can reduce polarization artifacts [23]. Therefore, polarization-sensitive B-MFA may mitigate the polarization artifact in in vivo imaging.

# 5.6.2. Extension of measurement filed

For certain in vivo applications, wide-lateral-field imaging is of significant interest. However, B-MFA necessitates lateral oversampling due to its utilization of computational refocusing, and it may limit the field size. For instance, in the present study, the lateral B-scan area is $1.5\mathrm{mm}$ and is covered with 256 sampling points.

One potential solution is the adoption of high-speed OCT systems. While the current study employs an OCT system with an A-line rate of 50,000 A-lines/s, ultra-high-speed OCT systems capable of A-line rates such as 3,280,000 A-lines/s [47] have been demonstrated, recently. By utilizing such a system, for example, the lateral size of the B-scan could be extended by up to 65.6 times (i.e., up to $98.4\mathrm{mm}$ ) without compromising the sampling density and without increasing the measurement time in principle.

# 6. Conclusion

We proposed the B-MFA method, which suppresses MS signals in in vivo imaging. The method was validated using phantom and in vivo small fish measurements. The subjective observation of the images and objective evaluation of SSRE showed that the B-MFA method improves the image contrast by reducing the MS signals. In addition, the B-MFA showed superior performance than MFA for in vivo measurements.

Here, we conclude that the B-MFA method can reduce noise caused by the MS signal in OCT images and is a better option for in vivo measurement than our previously proposed MFA method.

# Appendix

# A. Sample-surface detection

The details of the sample surface detection used in Section 2.4 are as follows. We first manually select the rough depth region in which the sample surface is searched, which is typically around 90 pixels (around $652\mu \mathrm{m}$ in air). Then, we compute the first order derivative of linear OCT intensity along the depth, where the derivative is defined as the difference between two neighboring pixels. The surface is defined at the top-most depth where the derivative is larger than a predefined threshold. The threshold is 50 in our particular case, but this value is in arbitrary unit and may differ in various OCT systems.

# Funding

Core Research for Evolutional Science and Technology (JPMJCR2105); Japan Society for the Promotion of Science (21H01836, 22K04962, 22KF0058); China Scholarship Counci
l (201908130130).

# Acknowledgment

Please see https://optics.bk.tsukuba.ac.jp/COG/.

# Disclosures

Y. Zhu, L. Zhu, Lim, Makita, Guo, Yasuno: Sky Technology (F), Nikon (F), Kao Corp. (F), Topcon (F), Panasonic (F), Santec (F). L. Zhu is currently employed by Santec.

# Data Availability

Data underlying the results presented in this paper are not publicly available at this time but may be obtained from the authors upon reasonable request.

# Supplemental document

See Supplement 1 for supporting content.

# References

# Supplementary Material

Abstract: This file supplements Section 4.2 by showing the intensity cross-sectional images of the other medaka fish samples (samples 2 to 10) measured for validation of MS reduction by the B-scan-wise-multi-focus averaging (B-MFA) method. All of the results show that several anatomic features, such as a hyper-scattering layer structure and the boundary of hollow structure are better visible in the B-MFA image than single acquisition, SFA, and conventional MFA.

© 2024 Optica Publishing Group

![](dt=2026-05-29/ht=23/88c53c31046ffdac3ac5aa78de36363d742fd4da7bd45b2d5bd70ef696fb5da7.jpg)

![](dt=2026-05-29/ht=23/7318487b116f8249fe66db918632a6361132e9596ca02e58d4d50e72614a7c36.jpg)

![](dt=2026-05-29/ht=23/cca3b0120437e66cc860c8e8b0aed473f96c68809738424aa8f928772671b9fa.jpg)