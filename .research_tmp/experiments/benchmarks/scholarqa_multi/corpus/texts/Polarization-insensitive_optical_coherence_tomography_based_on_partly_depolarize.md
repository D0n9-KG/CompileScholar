# All fiber polarization insensitive detection for spectrometer based optical coherence tomography using optical switch

DAVID ODEKE OTUYA, $^{1,2}$ GARGI SHARMA, $^{3}$ GUILLERMO J. TEARNEY, $^{1,2}$ AND KANWARPAL SINGH $^{3,*}$

Abstract: Polarization dependent image artifacts are common in optical coherence tomography imaging. Polarization insensitive detection scheme for swept source based optical coherence tomography systems is well established but is yet to be demonstrated for all fiber spectrometer-based Fourier domain optical coherence tomography systems. In this work, we present an all fiber polarization insensitive detection scheme for spectrometer based optical coherence tomography systems. Images from chicken breast muscle tissue were acquired to demonstrate the effectiveness of this scheme for the conventional Fourier domain optical coherence tomography system.

© 2019 Optical Society of America under the terms of the OSA Open Access Publishing Agreement

# 1. Introduction

High resolution, real time, three-dimensional (3D) imaging of biological tissue is of great interest in both; to study and to diagnosis human diseases. Optical coherence tomography (OCT) provides such 3D tissue imaging capability with a micron level resolution at video rate [1]. In a typical OCT system [1], an optical signal from a broadband source is divided into a sample-arm and reference-arm signals using a beam splitter. The reflected reference signal and sample signal are combined and the interference signal is detected using a detector assembly.

A system that employs a wavelength-tuning optical source, where the interference signal is scanned temporally at the detector, is termed as a swept source OCT (SS-OCT) system [2]. Meanwhile, a system where a stationary broadband signal is dispersed spatially and detected using a spectrometer is referred as a spectrometer-based Fourier domain OCT (FD-OCT) system [3]. Both these systems suffer from changes in the polarization of the optical signal when the signal is transmitted through materials possessing anisotropic properties [4].

In particular, OCT systems that employ single mode optical fibers, experience considerable variation in the polarization state of the optical signal arising from stretching and bending of the optical fiber [4]. One condition necessary for interference to occur between the reference-arm and sample-arm signals is that the polarization states of both these signals should match. This ideal condition is hard to satisfy in the clinical setting, where the probe undergoes constant motion resulting in the polarization change for sample-arm signal.

A Polarization maintaining (PM) fiber could be used to overcome such challenges, however, such fibers introduce artifacts such as polarization crosstalk in the images [5,6]. Furthermore, the polarization state of the sample-arm signal may change due to the birefringence of the tissue being imaged and PM fibers may not be able to account for that. This problem has been mitigated in SS-OCT systems using a polarization insensitive detection unit (PIDU) [6,7].

This allows for detection of the orthogonally polarized interference signal, independently, which when combined remove the polarization dependent image artifacts. Employing a PIDU in spectrometer-based FD-OCT system is challenging, for it would require two separate identical spectrometers. The use of two spectrometers to achieve polarization artifact free image has been demonstrated in a special

Check for updates

Research Article

Vol. 2, No. 12/15 December 2019/OSA Continuum

3465

OSA CONTINUUM

377581

Journal © 2019

https://doi.org/10.1364/OSAC.2.003465

Received 10 Sep 2019; revised 19 Nov 2019; accepted 20 Nov 2019; published 3 Dec 2019

class of OCT systems called polarization sensitive OCT (PS-OCT), where the spectra from the two spectrometers were corrected to match each other through signal processing [8]. Single camera PS-OCT systems have also been demonstrated [9-13], however for all these systems, whether using a single camera or multiple cameras, were developed as bench-top systems, where the polarization state of the reference and sample arm fibers could easily be controlled precisely.

Previously, an optical switch based bench top OCT system was demonstrated using a single camera [13]. The system [13] demonstrated was based on free space optics and hence not compatible with endoscopic applications where sample fiber is in constant motion because of the operator's manipulation of the endoscope. To our knowledge, a polarization artifact free, spectrometer-based system, compatible with a constantly moving sample arm has not been demonstrated yet. In this work, we have developed a PIDU for spectrometer-based FD-OCT system and demonstrate that polarization associated artifacts in image can be minimized.

# 2. Experimental design

The schematic of the FD-OCT systems used in this work are shown in Fig. 1.

![](dt=2026-04-06/ht=12/c4ffc30f5752783e4de8572d38c09d248e4da6de6e303c63efee66f1da785ff8.jpg)

Light with optical power $3\mathrm{mW}$ from a broadband light source; super luminescent diode was amplified using a broad band optical amplifier to $25\mathrm{mW}$ optical power with a full width half maximum of $90\mathrm{nm}$ around central wavelength of $1310\mathrm{nm}$ . This signal was then coupled to a single mode fiber (mode field diameter $9\mu \mathrm{m}$ ) which was further connected to a circulator. The circulator made it possible to direct the signal from the SMF to a 50:50 beam splitter (Gould Optics, USA) and the returning signal from the beam splitter to a detection unit.

One port of the beam splitter formed the reference arm and other port was used as the sample arm. The signal in the reference arm was collimated using an optical collimator and directed towards a reference

Research Article

Vol. 2, No. 12/15 December 2019/OSA Continuum

3466

OSA CONTINUUM

mirror through a variable neutral density filter. The reference signal reflected from the mirror was coupled back to the reference arm. The sample arm signal from the fiber coupler was coupled to a tethered capsule. At the distal end of the tether, a focusing ball lens and a side reflecting prism mounted on a rotating motor (rotation speed $40\mathrm{Hz}$ ) were assembled within the capsule housing. This enabled scanning of the optical beam on to the sample circumferentially which made it possible to obtain cross sectional images of the tissue.

The light reflected from the sample and the reference mirror was coupled to port 3 of the circulator. To develop a polarization insensitive OCT imaging system, a fiber-based polarization beam splitter (AFW Technologies, Australia) was introduced after the circulator. Two arms of the polarization splitter were connected to an optical switch (Agiltron, USA) which was triggered in synchronous with the spectrometer camera.

Optical fibers in the reference arm were fixed and the polarization controller in the reference arm was adjusted in a way that both polarization arms of the polarizing beam splitter receive an equal amount of reference signal. Alternate spectra were acquired from the two arms of the polarization splitter representing two orthogonal polarization states of the interfered signals.

Spectrometer consisted of transmission grating (1200 lines per mm), focusing lens $(\mathrm{f} = 80$ mm) and InGaAs line scan camera (Sensors Unlimited, USA) with 2048 pixels. The data from the spectrometer was collected using a frame grabber (BitFlow, USA) at line scan speed of $100~\mathrm{kHz}$ . The collected data was processed using custom designed Labview (National Instruments, USA) based software. The spectra from the spectrometer were rescaled from wavelength space to wave number space and Fourier transformed to obtain the axial scan of the sample. The two orthogonal polarization axial scans were combi
ned using Eq. 1, to obtain an axial scan, free from polarization artifacts.

$$
I = \sqrt {I _ {H} ^ {2} + I _ {V} ^ {2}} \tag {1}
$$

where $I$ is the total intensity of the axial scan, $I_{H}$ is the signal intensity from horizontal polarization state and $I_{V}$ is the signal intensity from vertical polarization state.

# 3. Results

The developed system was tested for various parameters such as axial resolution, lateral resolution and imaging range which were found to be $10\mu \mathrm{m}$ , $30\mu \mathrm{m}$ , and $5\mathrm{mm}$ respectively. To measure the sensitivity of the system, we used a mirror as the sample. The sensitivity of the system which is a measure of the minimum sample signal detectable by the system was measured to be $105\mathrm{dB}$ at zero optical path difference between the sample and the reference mirror.

The sensitivity of the system dropped by $6\mathrm{dB}$ at an optical path difference of $2\mathrm{mm}$ between the sample and the reference mirror. To demonstrate the proof of principle in biological tissue we imaged chicken breast as sample. Chicken breast because of its high birefringence has been used numerous times to demonstrate polarization based measurements.

Chicken breast tissue was wrapped around the tethered capsule and imaged using the OCT system without PIDU and with PIDU. During the imaging, tissue and the tethered capsule assembly was held in hand and maneuvered constantly to mimic real clinical conditions. The image of the tissue acquired with the OCT system without PIDU is shown in Fig. 2 (left). Image artifacts appearing as bands of bright and dark intensities are clearly observed in the movie (Visualization 1) presented as supplementary information. Images were acquired and recorded for 10 seconds.

For the OCT system without PIDU (Visualization 1), it was observed that the bright and dark bands were constantly fluctuating. This movement in the intensity band can be attributed to the polarization dependent phase changes in the sample light because of the fiber movements while the tethered assembly was maneuvered.

Images acquired for the same tissue at the approximately same location under similar conditions with the OCT system with PIDU (Visualization 2) are shown in Fig. 2 (right). One can see that the image artifacts (black bands) are not noticeable in the images acquired with OCT system with

Research Article

Vol. 2, No. 12/15 December 2019/ OSA Continuum

3467

OSA CONTINUUM

![](dt=2026-04-06/ht=12/ae8c83b7be5be75c2042f72869d3ccf85611797e4f427f8886fb6fe5af3a238e.jpg)

![](dt=2026-04-06/ht=12/7093ed8a0052f6a175ffa1152b7bf2e04989d3e56888daec035ef9eeeb836fc5.jpg)

PIDU. On close examination, the reader will also notice from the movies of the two cases that it is not only the light from the tissue that changes in terms of intensity but also the light from the inner wall of the capsule which is not in tissue contact. This supports the idea that polarization artifacts need not come from the tissue but can also come from the system alone.

To quantify the improvement of the OCT system with PIDU over the OCT system without PIDU, we measured the change in light intensity from capsule wall for both cases. To do so, the

![](dt=2026-04-06/ht=12/ffdd171fdb01339088b83e281ed49e7727bbc4656b573468f9b57f7072efb859.jpg)

Research Article

Vol. 2, No. 12/15 December 2019/OSA Continuum

3468

OSA CONTINUUM

average intensity of $50~{\mu\mathrm{m}}$ X $50~{\mu\mathrm{m}}$ image area around the inner capsule wall was measured. Change in light intensity for the OCT system without PIDU and OCT system with PIDU is shown in Fig. 3. The standard deviation in intensity change for OCT system without PIDU was measured to be $2.66\mathrm{dB}$ and for OCT system with PIDU was measured to be $0.33\mathrm{dB}$ .

# 4. Conclusion

We present a simple design for spectrometer-based OCT systems that serve to minimize sample- and system- dependent polarization artifacts for endoscopic applications. Our design is particularly useful for systems used in clinical settings where the sample arm is constantly under motion during probe introduction and when subjected peristaltic motion. One drawback of the proposed design is the inevitable sacrifice in the imaging speed as the overall acquisition speed is reduced by half. However, the availability of faster cameras still makes it possible to achieve video rate imaging speeds.

Another limitation of the proposed design is the sequential acquisition of the orthogonal polarization A-scans that are combined to create a single A-scan. The two sequential A-scans are not acquired from the exact same location of the tissue and hence sudden changes in the signal phase from the system or changes within the tissue may introduce artifacts in the image at slower imaging speeds.

Since time interval and spatial distance between adjacent scans were $10\mathrm{ms}$ and $17~{\mu\mathrm{m}}$ respectively, such abrupt phase changes are minimal to degrade the polarization artifact free image. Nevertheless, if such artifacts are prominent, then one can introduce a delay line in one of the arms of the polarization splitter such that both signals reaching the spectrometer are essentially from the same location and experiencing the same system disturbances.

Such systems have been previously reported to remove random intensity noise from OCT images which occur because of the instability in the power level of the light source and can be minimized through balanced detection [14].

# Funding

Max-Planck-Gesellschaft; Massachusetts General Hospital.

# Disclosures

# References

Research Article

Vol. 2, No. 12/15 December 2019/ OSA Continuum

3469

OSA CONTINUUM

# OSA CONTINUUM

Research Article

Vol. 2, No. 12/15 December 2019/ OSA Continuum

3470

123.120.11.147 (5/2023-06-08 03:07:49)