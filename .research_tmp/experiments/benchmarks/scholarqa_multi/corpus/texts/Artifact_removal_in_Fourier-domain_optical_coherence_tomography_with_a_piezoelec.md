# Artifact removal in Fourier-domain optical coherence tomography with a piezoelectric fiber stretcher

Sebastien Vergnole, $^{1,*}$ Guy Lamouche, $^{1}$ and Marc L. Dufour

$^{1}$ Industrial Materials Institute, National Research Council Canada, 75 Boulevard de Mortagne, Boucherville, Quebec, J4B 6Y4, Canada

*Corresponding author: Sebastien.Vergnole@cnrc-nrc.gc.ca

Received September 26, 2007; revised February 15, 2008; accepted February 18, 2008;

posted March 4, 2008 (Doc. ID 87957); published March 31, 2008

We describe an artifact removal setup swept-source optical coherence tomography (OCT) system that enables high-speed full-range imaging. We implement a piezoelectric fiber stretcher to generate a periodic phase shift between successive A-scans, thus introducing a transverse modulation. The depth ambiguity is then resolved by performing a Fourier filtering in the transverse direction before processing the data in the axial direction. The dc artifact is also removed.

The key factor is that the piezoelectric fiber stretcher can be used to generate discrete phase shifts with a high repetition rate. The proposed experimental setup is a much improved version of the previously reported B-M mode scanning for spectral-domain OCT in that it does not generate additional artifacts. It is a simple and low-cost solution for artifact removal that can easily be applied. © 2008 Optical Society of America

OCIS codes: 110.4500, 170.4500, 100.5070.

In Fourier-domain optical coherence tomography (FD-OCT), the signal is collected as a function of the wavelength and the spatial information is recovered by Fourier transform. The main drawback of such a technique is that, since a real signal is acquired, the Fourier transform is symmetric around the origin. Therefore, one cannot distinguish the positive depth from the negative depth. Additionally, autocorrelation terms in the recorded signal lead to a spurious component at zero frequency (dc artifact).

In spectral-domain OCT (SD-OCT) [1], a broadband source is used to illuminate the sample and the wavelengths are separated upon detection. A common solution to remove artifacts is to introduce phase shifts between measurements [2-4]. In swept-source OCT (SS-OCT) [5], the wavelengths are separated upon generation with a wavelength sweeping source. Efficient setups have been proposed to remove artifacts by shifting the frequency using electro-optic [6] or acousto-optic modulators [7,8]. Although efficient, these approaches increase significantly the cost of the OCT system.

In SD-OCT, a solution that uses the B-M scanning method with a mirror mounted on a piezoelectric transducer has been proposed by Yasuno et al. [9]. In this approach, a phase shift (M-scan) is introduced between successive A-scans to generate a B-scan with transverse modulation that is used to remove both the depth degeneracy and the dc artifact. The main advantage of this approach is that it has a higher stability with respect to phase and amplitude fluctuations than for phase-shifting techniques that use successive spectra during continuous scanning for complex signal reconstruction.

A very similar approach has also been proposed by Wang [10,11] in SD-OCT. In [9], the experimental setup does not allow discrete increments in phase shift at a high speed. Consequently, a triangular ramp is used that generates other artifacts that require further data

processing to be removed. In [10,11], a sawtooth waveform is applied, and this technique does not remove the autocorrelation and cross-correlation artifacts. We present an alternative method applied to SS-OCT and using a piezoelectric fiber stretcher (PFS). The use of a fiber stretcher improves over previous papers by allowing the fast generation of discrete steps in the phase shift generated between A-scans. This avoids the introduction of additional artifacts and leads to a more efficient technique. We recently reported preliminary results for this approach [12], and we provide here a detailed description of the technique.

The signal processing approach has some similarity with phase-shifting interferometry [2]. We illustrate it by considering the trivial case of a single reflector. For a B-scan, a simplified version of the interferometric signal is $i(x,\nu) = k_{0} + \cos (k_{x}x + k_{\nu}\nu)$ , where $k_{0}$ is a constant that includes the autocorrelation terms, $k_{x}$ is linked with the phase shift introduced by the PFS, $x$ is the transverse position, $k_{\nu}$ is linked with the wavelength sweeping of the source, and $\nu$ is the optical frequency. We first compute the Fourier transform of $i(x,\nu)$ along the $x$ transverse direction:

$$
I (u, v) = k _ {0} \delta (u) + \frac {1}{2} \delta \left(u - k _ {x}\right) e ^ {- i k _ {v} v} + \frac {1}{2} \delta \left(u + k _ {x}\right) e ^ {i k _ {v} v}, \tag {1}
$$

where $u$ is the spatial frequency (Fourier conjugate of $x$ ), $k_{0}\delta(u)$ is the dc component, the second term on the right hand side is the OCT data, and the last term is the complex conjugate of the second one. A high-pass filtering with a rectangular window is then performed to keep only the data corresponding to the OCT signal to yield: $\hat{I}(u,\nu) = 1/2[\delta(u - k_x)]e^{-ik_\nu \nu}$ . Then, the inverse transverse Fourier transform is evaluated:

732

OPTICS LETTERS / Vol. 33, No. 7 / April 1, 2008

0146-9592/08/070732-3/$15.00

© 2008 Optical Society of America

$$
\hat {i} (x, \nu) = \mathcal {F} _ {x} ^ {- 1} [ \hat {I} (u, \nu) ] = \frac {1}{2} \left(e ^ {- i k _ {x} x}\right) e ^ {- i k _ {\nu} \nu}. \tag {2}
$$

Finally, as usually performed in FD-OCT, the axial inverse Fourier transform is evaluated:

$$
\hat {I} (x, z) = \mathcal {F} _ {z} ^ {- 1} [ \hat {i} (x, \nu) ] = \frac {1}{2} [ e ^ {- i k _ {x} x} ] [ \delta (z - k _ {\nu}) ], \qquad (3)
$$

where $z$ is the Fourier conjugate of $\nu$ and is proportional to the depth position. It must be noted that we display the OCT image by computing $20 \times \log (|\hat{I}(x,z)|)$ .

Figure 1 shows an experimental implementation of the proposed SS-OCT setup. It is a Mach-Zehnder fiber-based interferometer. The source is a Thorlabs swept source with a $1325\mathrm{nm}$ center wavelength and an $85~\mathrm{nm}$ FWHM. The theoretical axial resolution is $\delta z = 9.1\mu \mathrm{m}$ in air. The A-scan rate is $16\mathrm{kHz}$ when using both the backward and the forward wavelength scans. The setup is fitted with a PFS from Optiphase (PZ1-STD-FC/APC). This device consists of $10\mathrm{m}$ of fiber wound around a cylindrical piezoelectric transducer.

Figure 2 gives the shape of the low voltage applied to the PFS. The PFS drive signal is adjusted to achieve a $\pi /2$ phase shift between two successive forward wavelength scan interferograms. For the current work, the system operates at a reduced rate of $8\mathrm{kHz}$ since we use only the forward wavelength scans.

One of the key adjustments is to choose a transverse step small enough to ensure efficient artifact removal. Introducing a periodic phase shift between A-scans can be seen as introducing a carrier spatial frequency in the transverse direction. By taking the Fourier transform in the transverse direction, one obtains three components in the spectrum: a positive part centered around the carrier frequency, a zero component, and a negative part centered around the negative of the carrier frequency.

The width of the positive and negative parts is related to the spatial frequency content of the image in the transverse direction. The amplitude of the transverse Fourier spectrum from an OCT image of an onion is shown in Fig. 3 for two step sizes: 1.2 and $2.5\mu \mathrm{m}$ . Let $\delta x$ be the transverse step and $u_{\delta x} = 1 / [2(\delta x)]$ the Nyquist spatial frequency. The OCT data are associated in Fig. 3 to the positive part of the spectrum $[0:u_{\delta x}]$ , while the conjugate OCT data are linked to the negative

![](image)
-ha/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-02-21/ht=12//3e32418f920037a259e90ac66b526fa181962b0481c04178c6514b055d450768.jpg)

![](dt=2026-02-21/ht=12/97c0678f97fc4aa8cc693323321970bb16c7fcf214d92d76c2d6d213eb379cc2.jpg)

part $([-u_{\delta x}:0[$ ). For the larger transverse step, corresponding to a low carrier frequency, the negative part of the spectrum leaks into the positive part (Fig. 3, top left). This leads to the mixing of the OCT data and their conjugate and to an insufficient artifact removal (Fig. 3, bottom left). By selecting a small transverse step, thus a large carrier frequency, the positive and negative parts of the spectrum are well separated (Fig. 3, top right) and the artifact removal is very efficient (Fig. 3, bottom right).

Decreasing the transverse step size adversely increases the measurement time, so a trade-off must be selected. A convenient criterion is a step size that corresponds to a Nyquist frequency that is two times larger than the full width of the transverse spectrum of the OCT image $(\Delta u$ defined at $1 / e)$ : $1 / [(2\delta x)] > 2\Delta u$ . The width $\Delta u$ is related to the speckle size $s$ (full width at $1 / e$ ) by $\Delta u = 4 / (\pi s)$ . The speckle size is given by $s = (0.68 \times 4\lambda f) / (\sqrt{2}\pi d)$ , taking into account the illuminating

![](dt=2026-02-21/ht=12/aaa691d369c1b2b18a20f916fb24c5fc243533b0cd5d8113c8fb78a76f95ab21.jpg)

April 1, 2008 / Vol. 33, No. 7 / OPTICS LETTERS

733

and detecting optics along with contribution from a high density of scatterers [13]. The parameter $d$ is the diameter of the incoming light beam, and $f$ is the focal length of the illuminating and collecting optics. Therefore, this leads to the condition for the transverse step size:

$$
\delta x <   \frac {0 . 6 8}{4 \sqrt {2}} \frac {\lambda f}{d}. \tag {4}
$$

In the example of Fig. 3, we have $\lambda = 1.325\mu \mathrm{m}$ , $f = 14.5\mathrm{mm}$ , and $d = 2\mathrm{mm}$ , which gives a speckle size of $6\mu \mathrm{m}$ and thus to the condition: $\delta x < 1.2\mu \mathrm{m}$ . The results of Fig. 3 illustrate the validity of the criterion. The rightmost image meets the upper limit of the criterion, whereas the leftmost image was obtained with a twice larger transverse step size.

To validate our approach, we first evaluate the attenuation of the mirror artifact: a mirror was used as a sample and the ratio of the surface signal to its complex conjugate was measured to be greater than $32\mathrm{dB}$ in the range $[-0.5: + 0.5]\mathrm{mm}$ . Then, a cross section of a finger tip of a healthy human volunteer was acquired. Figure 4 shows the images without (left) and with (right) artifact removal processing. The imaged area is $3\mathrm{mm}$ wide and $3\mathrm{mm}$ deep with the zero path delay set in the middle of the depth scale.

As seen on the image on the right, the dc artifact is completely removed and the mirror image is greatly reduced with our postprocessing technique. The apparent structure below the skin of the finger that can hardly be seen on the left image is clearly resolved in the right one, providing a good appreciation of the efficiency of the proposed method. The optics used to acquire this image is different from that of Fig. 3. The transverse step size is half the value determined by the condition in Eq. (4).

To our knowledge, this is the first report of a transverse scanning method applied to SS-OCT and using a PFS. The implementation of a PFS solves the additional artifact problem encountered in [9]. Our postprocessing enables the removal of autocorrelation

![](dt=2026-02-21/ht=12/98c013ba64a01ee9505bb3e5a96895a469ce351980bf9bf6a0cd9d5d15d96c26.jpg)

![](dt=2026-02-21/ht=12/1abc3b4f60f8b4e55329499506c939483d65e4c70c3e06da594213c03949f46f.jpg)

noise, which is not the case in [10,11]. The main advantages of using such a technique are as follows. First, it is very easy to implement experimentally because the PFS is a fiber-based device with conventional connectors (type FC/APC) and thus easy to connect with other photonic devices. Second, it is very efficient in terms of transmission power since the only small losses are due to the fiber connections. Third, it is low cost compared with the techniques using electro-optic or acousto-optic modulators. Finally, it must be noted that the PFS does not affect the measurement nor request any changes in the setup when not in use. This technique can also be applied to SD-OCT setups.

It came to our attention that three articles [14-16] that used a similar approach to ours to achieve full-range complex FD-OCT imaging have been published during the reviewing process of this letter. However, these papers use the galvanometer scanning system to perform the transverse modulation and are applied to SD-OCT systems. Our proposed approach is of more general application since it does not depend on the transverse scanning technique.

We thank Bruno Gauthier for his technical contribution. We acknowledge the financial support of the Genomics and Health Initiative of the National Research Council Canada.

# References

734

OPTICS LETTERS / Vol. 33, No. 7 / April 1, 2008