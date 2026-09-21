# Dispersion compensation in Fourier domain optical coherence tomography using the fractional Fourier transform

Norman Lippok, $^{1,2*}$ Stéphane Coen, $^{1}$ Poul Nielsen, $^{2}$ and Frédérique Vanholsbeeck $^{1}$

<sup>1</sup> Physics Department, The University of Auckland, Private Bag 92019, Auckland, New Zealand

$^{2}$ Auckland Bioengineering Institute, The University of Auckland, Private Bag 92019,

Auckland, New Zealand

* nlip001@aucklanduni.ac.nz

Abstract: We address numerical dispersion compensation based on the use of the fractional Fourier transform (FrFT). The FrFT provides a new fundamental perspective on the nature and role of group-velocity dispersion in Fourier domain OCT. The dispersion induced by a $26\mathrm{mm}$ long water cell was compensated for a spectral bandwidth of $110~\mathrm{nm}$ , allowing the theoretical axial resolution in air of $3.6\mu \mathrm{m}$ to be recovered from the dispersion degraded point spread function. Additionally, we present a new approach for depth dependent dispersion compensation based on numerical simulations. Finally, we show how the optimized fractional Fourier transform order parameter can be used to extract the group velocity dispersion coefficient of a material.

© 2012 Optical Society of America

OCIS codes: (110.4500) Optical coherence tomography (Imaging systems); (070.2575) Fractional Fourier transforms; (170.3890) Medical optics instrumentation.

# References and links

#171869 - $15.00 USD

Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23398

#171869 - $15.00 USD

Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23399

# 1. Introduction

Developed in the early 1990's, optical coherence tomography (OCT) is an established and successful imaging technology that enables high resolution, cross-sectional imaging of biological tissues and materials [1, 2]. Nowadays, it is commonly implemented in the frequency domain with so-called Fourier-domain OCT (FD OCT), which has significantly improved sensitivity and imaging speed [3-5]. Frequency domain systems can be implemented either by using a low coherent light source and a spectrometer (SD OCT) [6-8], or by using a coherent swept-source laser (SS OCT) that is sampled by a photo diode [9-12].

OCT imaging has proven to be a powerful diagnostic tool in many medical fields. In some systems, OCT tomograms have image quality comparable to histology with resolutions down to the $1\mu \mathrm{m}$ level [13]. Obtaining high isotropic resolution over a large imaging depth range is however hindered by two factors. First, the high NA objectives traditionally used to obtain high lateral resolution limit the depth of field and hence the imaging depth. This can be circumvented by using axicons [14-16], binary-phase filters [17], or multi-beam OCT [18, 19].

The second problem arises from the broadband nature of the light source. Increasing the axial resolution requires larger optical bandwidths, which make OCT systems more sensitive to chromatic dispersion, especially for a long sensing range. For example, using a center wavelength around $800\mathrm{nm}$ , imaging the human macula with an axial resolution below $15\mu \mathrm{m}$ makes dispersion compensation necessary in order to account for the dispersion produced by the vitreous body of the human eye [20, 21].

In ophthalmology, it is therefore advantageous to operate close to the zero dispersion wavelength of water, $\lambda_0 = 1\mu \mathrm{m}$ [22].

With this in mind, it is no surprise that dispersion compensation methods have become increasingly important for OCT. Traditional methods rely on placing the right amount of dispersion balancing material in one interferometer arm of the OCT setup [23-25], but this is usually only practical for 2nd-order dispersion. Grating-based phase delay scanners [26] and dual optical fiber stretchers [27] can also be used for 2nd order dispersion compensation and present some degree of tunability.

However, these approaches require bulky components and the latter is difficult to adapt for depth-dependent compensation of sample dispersion. We note that a fiber-stretching-based dispersion compensator has been recently combined with a grating-based, scanning free, time domain OCT system to compensate for both 2nd and 3rd-order dispersion [28], but to the detriment of added complexity.

Because of these drawbacks, OCT systems increasingly rely on numerical dispersion compensation techniques, which offer continuous adjustment capabilities and, in theory, can be optimized for any amount of dispersion. This was first demonstrated in time-domain systems [21, 29-31]. It is however more readily implemented with FD OCT systems which have direct access to the phase information of the signal as was originally demonstrated almost simultaneously by two groups, Cense et al. [32] and Wojtkowski et al. [33]. Their method is based on the complex conjugate of the dispersive spectral phase. It compensates for 2nd and higher-order dispersion by adding an optimized phase term to the analytical expression of the meas

#171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23400

ured spectral fringe signal. The phase term is found using a sharpness function that searches for maximum signal magnitude in a given depth range. Note that FD OCT systems still require nearly dispersion-matched interferometer arms: too much dispersion spreads the point-spread-function (PSF) over a wide range of depths, leading to sensitivity decay and information loss, even after numerical dispersion compensation.

In our work, we revisit the problem of numerical dispersion compensation based on the use of the fractional Fourier transform (FrFT). In doing so, our aim is not to provide a replacement to the dispersive spectral phase compensation technique discussed above and which is rapidly becoming the de-facto standard of numerical dispersion compensation. Rather, we wish to illustrate how the FrFT provides a new fundamental perspective on the nature and role of group-velocity dispersion in Fourier domain OCT and how it highlights in a visual manner the physics behind dispersion compensation.

Our work also provides new general insights into advanced problems associated with dispersion compensation such as defining a sharpness function that is not dependent on the presence of an isolated single back-scatterer, the issue of depth-dependent dispersion, or the prospect of differentiating materials by dispersion mapping [34, 35]. In the following, we first briefly present the theory of the FrFT and then demonstrate numerical dispersion compensation using FrFT experimentally.

Furthermore, we show theoretically how we can extend our method, and use the FrFT for depth-dependent sample dispersion compensation. Finally, we show how we can extract the 2nd-order group-velocity dispersion coefficient $\beta_{2}$ of a sample from the order parameter of the fractional Fourier transform. All of our techniques are general and can be applied to SD as well as SS OCT, but also more generally to interferometry or optical coherence domain reflectometry.

# 2. Theory

The fractional Fourier transform (FrFT) is a generalization of the traditional Fourier transform (FT), with the addition of an order parameter $a$ that can be interpreted as the FT to the power of $a$ . The traditional FT and the associated frequency domain are special cases of the FrFT and its continuum of fractional Fourier domains, which are elegantly related to phase-space distributions as will be discussed below [36]. The FrFT was initially defined by Namias in the context of quantum mechanics [37] but has then been used in optics [38] and signal proc
essing [39].

For various problems in which the Fourier transform (FT) is used, there is a potential for generalization and improvement by using the FrFT. By replacing the traditional Fourier transform (FT) with the FrFT, we gain an additional degree of freedom to a problem within a phase-space distribution (time-frequency distribution).

The $a$ -th order FrFT is a linear transform. For the range $0 < |a| < 2$ it is defined as [40]

$$
f _ {a} \left(u _ {a}\right) = F ^ {a} [ f (u) ] = \int_ {- \infty} ^ {\infty} A _ {\phi} \exp \left[ i \pi \left[ u _ {a} ^ {2} \cot (\phi) - 2 u u _ {a} \csc (\phi) + u ^ {2} \cot (\phi) \right] \right] f (u) d u, \tag {1}
$$

where

$$
A _ {\phi} = \frac {\exp \left\{- i \pi^ {\frac {\operatorname {s i g n} [ \sin (\phi) ]}{4} + \frac {i \phi}{2}} \right\}}{| \sin (\phi) | ^ {1 / 2}} \quad \text {a n d} \quad \phi = \frac {a \pi}{2}. \tag {2}
$$

Here $u$ is the independent variable of the transform input function while $u_{a}$ is that of the transform output, with corresponding fractional transform order $a$ . These independent variables are assumed to be dimensionless (see below).

When $a = 1$ , $A_{\phi}$ becomes unity, the first and third terms in the argument of the exponential in Eq. (1) vanish, and the FrFT reduces to the traditional FT (i.e., the FrFT to the power of 1). Therefore, the $u_{1}$ and $u$ axes represent the (normalized) time $\tau$ and optical frequency $\nu$ axes, respectively (the normalization is such that $uu_{1} = \nu \tau$ ). Note that, in our work, these domains

#171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23401

are swapped compared to traditional terminology because we deal here with FD OCT, i.e., the measurements $f(u)$ are performed in the frequency domain.

For $a \to 0$ and $a \to \pm 2$ the integral kernel approaches $\delta(u_a - u)$ and $\delta(u_a + u)$ , sampling $f(u)$ as the identity and parity operators, respectively [40]. Only for $a = 1$ does the transform of a real function yield mirrored counterparts in the two halves of the Fourier space. Otherwise the transform yields different energy distributions in the two (fractional) Fourier half spaces. Except for the special cases $a = 0$ and $a = 1$ , the transform output lies in neither the traditional frequency nor time domain.

The FrFT is related to a more general time-frequency representation of the function $f(u)$ , namely its Wigner distribution $W_{f}(u,u_{1})$ [36]. Recall that time-frequency distributions aim to represent how the spectral content of a signal change with time, or, in other words, how the energy of a signal is distributed both in time and frequency. Compare this with the traditional FT, which provides no information as to when specific frequency components occur in the signal.

The Wigner distribution is one of the most commonly used time-frequency distributions and it is found that the Wigner distribution of $f_{a}$ is simply a rotated version of that of $f$ [40]. More specifically, $W_{f_a}$ is $W_{f}$ as seen in a reference frame $(u_{a},u_{a + 1})$ rotated by an angle $\phi = a\pi /2$ with respect to the time-frequency plane $(u,u_1)$ ,

$$
W _ {f a} \left(u _ {a}, u _ {a + 1}\right) = W _ {f} \left(u _ {a} \cos \phi - u _ {a + 1} \sin \phi , u _ {a} \sin \phi + u _ {a + 1} \cos \phi\right). \tag {3}
$$

From the integral projection properties of the Wigner distribution, $\int_{-\infty}^{\infty}W_{f}(u,u_{1})du_{1} = |f(u)|^{2}$ , it follows that the integral projection of the Wigner distribution $W_{f}$ onto an axis $u_{a}$ making an angle $\phi = a\pi /2$ with the $u$ axis, yields the squared amplitude of the fractional Fourier transform of order $a$ ,

$$
\int_ {- \infty} ^ {\infty} W _ {f _ {a}} \left(u _ {a}, u _ {a + 1}\right) d u _ {a + 1} = \left| f _ {a} \left(u _ {a}\right) \right| ^ {2}. \tag {4}
$$

A projection at $90^{\circ}$ $(a = 1)$ corresponds to the traditional FT, and is consistent with the other marginal of the Wigner distribution, $\int_{-\infty}^{\infty}W_{f}(u,u_{1})du = |f_{1}(u_{1})|^{2}$ .

To illustrate these concepts, we consider a simple linear chirp signal

$$
S (\omega) = \operatorname {R e} \left\{I (\omega) \exp \left[ j \left(\omega \tau + \beta_ {2} l \left(\omega - \omega_ {\mathrm {c}}\right) ^ {2}\right) \right] \right\}. \tag {5}
$$

In an FD OCT setup, such a spectral signal would be observed in the presence of residual chromatic dispersion from a single reflection occurring at a delay $\tau$ from the point of zero-path difference of the interferometer, with $\beta_{2}$ the group-velocity dispersion coefficient of the dispersive element of optical thickness $l$ . Here $\omega = 2\pi \nu$ is the angular optical frequency, and $\omega_{\mathrm{c}}$ the center frequency of the source spectrum $I(\omega)$ . After normalization, $u = \nu /\eta$ , $u_{1} = \eta \tau$ , Eq. (5) reads

$$
S (u) = \operatorname {R e} \left\{I (u) \exp \left[ j \left(2 \pi u u _ {1} + (2 \pi \eta) ^ {2} \beta_ {2} l \left(u - u _ {\mathrm {c}}\right) ^ {2}\right) \right] \right\} = \operatorname {R e} \left\{I (u) \exp [ j \varphi ] \right\}. \tag {6}
$$

For digital computation, the normalization factor $\eta$ is set as [40],

$$
\eta = \sqrt {\frac {\Delta v}{\Delta \tau}} = d v \sqrt {N}, \tag {7}
$$

where $\Delta \nu$ and $\Delta \tau$ are the width of the spectral and temporal intervals over which our signal is represented, respectively, $d\nu = \Delta \nu /N = 1 / \Delta \tau$ is the spectral resolution, and $N$ is the number of sampling points. With this scaling, the normalized length of both intervals are equal to the dimensionless quantity $\sqrt{\Delta\nu\Delta\tau} = \sqrt{N}$ , the Wigner distribution is confined to a circle, and the samples in both domains are spaced $1 / \sqrt{N}$ apart.

#171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23402

![](dt=2026-06-09/ht=11/7e3b8cc02504c19059eafb6ccc4bfaa52a2ca6b4f6b94e1ecd8d844653e03a86.jpg)

The Wigner distribution of the chirped signal $S(u)$ , Eq. (6), is shown schematically in Fig. 1, which can be understood as follows. The second-order dispersion term $(\beta_{2})$ leads to a linear frequency modulation of the detected spectral fringes $S(u)$ as a function of optical frequency, which can be related to the inclination of the Wigner distribution in the time-frequency plane. Note that, because we are dealing with spectral fringes, the instantaneous "frequency" of the fringes has unit of inverse optical frequency, i.e., time, hence appears as the $u_{1}$ (i.e., temporal) domain. This instantaneous frequency can be obtained by taking the derivative of the phase of the complex spectral fringes with respect to angular optical frequency, hence is given by

$$
\frac {1}{2 \pi} \frac {\partial \varphi}{\partial u} = u _ {1} + (2 \pi \eta) ^ {2} \frac {\beta_ {2} l}{\pi} (u - u _ {\mathrm {c}}), \tag {8}
$$

which highlights the linear frequency modulation. As explained above, Eq. (4), the squared amplitude of the $a$ -th order FrFT can be obtained as the integral projection of the Wigner distribution onto the $u_{a}$ axis making an angle $\phi = a\pi /2$ with the $u$ axis. In Fig. 1, we have chosen the order $a$ and the $u_{a}$ axis orientation in a way that the integral projection of our chirped signal leads to a maximally narrowed response (see green shaded area along the $u_{a}$ axis).

This illustrates that the FrFT, with an optimized order parameter $a$ , can effectively lead to dispersion compensation of an FD OCT signal. In contrast, projection onto the $u_{1}$ axis, i.e., the traditional FT of the signal, leads to a much broader response (red shaded area along the $u_{1}$ axis), because dispersi
on is uncompensated.

In essence, the FrFT order $a$ can be chosen to adjust the chirp rate of a dispersed OCT signal. This amounts to a rotation of the time-frequency plane. In the optimally rotated frame, the instantaneous frequency of the spectral fringes remains constant. The fractional Fourier transformed OCT signal only results from a different projection angle in the time-frequency distribution compared to the standard FT, while entirely preserving the energy of the original spectrum. With this in mind, one could state that group-velocity dispersion does not degrade the axial resolution in FD OCT but only causes one to observe an OCT depth signal from a perspective in the time-frequency plane that is not suitable for imaging. The FrFT allows one to correct for that perspective.

The optimized FrFT order parameter $a$ that leads to a dispersion-compensated OCT depth signal can be interpreted as an intuitive measure of the amount of chromatic dispersion present in an FD OCT system. If the optimized order is $a = 1$ , the OCT A-scan is dispersion free. Normal and anomalous dispersion can easily be distinguished by $a > 1$ and $a < 1$ , respectively.

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23403

# 3. Experimental setup

To experimentally demonstrate numerical dispersion compensation using the FrFT, we have used an SD OCT configuration as shown in Fig. 2. It is based on a Michelson interferometer built around a 50/50 fiber coupler made up of SM800 single-mode optical fiber (Thorlabs). The sample arm incorporates a galvanometric mirror for transverse scanning so that we can obtain 2D images. On the detector side, we use a custom-built spectrometer with a $30\mathrm{mm}$ focal length achromatic collimating lens at the input.

The light is then spectrally dispersed with a transmission volume phase holographic grating (1200 lines/mm, Wasatch Photonics Inc.) before being imaged onto a CMOS line scan camera (BASLER Sprint spl2048-70km) using another achromatic doublet lens with $75\mathrm{mm}$ focal length. The camera offers 2048 pixels, 12 bit resolution, and a maximum acquisition rate of $70\mathrm{kHz}$ . The spectrometer provides a spectral bandwidth of $220\mathrm{nm}$ with a spectral resolution of $0.1\mathrm{nm}$ . It has been carefully calibrated to avoid any coupling with dispersion compensation [41,42].

The light source of our OCT system is a superluminescent diode with a center wavelength of $843\mathrm{nm}$ and an optical bandwidth (full width at half maximum, FWHM) of $\Delta \lambda_{\mathrm{FWHM}} = 110\mathrm{nm}$ . The theoretical axial resolution, calculated by inverse Fourier transforming the source spectrum

![](dt=2026-06-09/ht=11/a71b36bfbc524672264644aba924c71ed82c97397b897d90b377078e19d47971.jpg)

![](dt=2026-06-09/ht=11/20337586b67d590746b2fa06881049837e3d0432bb5f4641ab011da9ccfe46e9.jpg)

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23404

(Wiener-Khinchin theorem), is $3.6\mu \mathrm{m}$ FWHM. Although care was taken to equalize the fiber lengths between the two arms of the interferometer, dispersion was not completely balanced in the initial setup. Coarse dispersion compensation was done physically by inserting two BK7 microscope slides into the reference arm, which led to an actual axial resolution of $3.8\mu \mathrm{m}$ that was close to the theoretical minimum. Dispersion may be matched more precisely e.g. by dispersion compensating prisms of the correct material.

The corresponding point-spread-function (PSF), obtained by using a mirror in place of the sample, is plotted as the dashed red curve in Fig. 3. Note that here the measured spectrum has been processed with the traditional FT to obtain the PSF. Figure 3 also shows the theoretical PSF (solid blue curve) for comparison. The difference can be explained by the fact that the two microscope slides do not exactly compensate the residual setup dispersion.

For all our measurements, the signal coming from the reference arm, measured by blocking the sample arm, and averaged over 100 spectra, is subtracted to obtain the spectrally modulated signal only. The modulated signal is then re-sampled to account for the hyperbolic dependence between wavelength and frequency. Finally, zero padding is performed in order to improve the digital sampling resolution in the transformed domain. The sensitivity of our OCT system was measured as $98\mathrm{dB}$ at $50~\mu \mathrm{s}$ exposure time and $100~{\mu\mathrm{m}}$ depth. A sensitivity fall off of $11.6\mathrm{dB}$ was measured at a depth of $1\mathrm{mm}$ .

# 4. Results

# 4.1. Point-spread function measurements

To test the efficiency of the FrFT for numerical dispersion compensation, the interferometer was first purposely dispersion mismatched by placing a $26\mathrm{mm}$ long water cell in the sample arm. Applying the traditional FT to the acquired spectrum resulted in a dispersed PSF $49~{\mu\mathrm{m}}$ wide (FWHM). This is shown as the solid blue curve in Fig. 4(a) and compared with the PSF obtained without the dispersive water cell (dotted red curve), which is the same as that plotted in Fig. 3.

Using the FrFT with an optimized order parameter $a_{\mathrm{opt}} = 1.0555$ , one obtains the OCT depth scan while simultaneously fully compensating for group velocity dispersion, as revealed by the dashed black curve in Fig. 4(a). Using the FrFT leads to an axial resolution of $3.65~{\mu\mathrm{m}}$ , closer to the theoretical minimum than that observed with coarse physical dispersion compensation before introducing the water cell $(3.8~{\mu\mathrm{m}})$ . This clearly highlights that the FrFT

![](dt=2026-06-09/ht=11/e896c36ea94d466a5352772fc8fece8ae646a20bbd40299f77f740b13ef843b5.jpg)

![](dt=2026-06-09/ht=11/2a391dc2103e5771f438c5116c3806716dc6955a96940f0f9b6af055d57fc634.jpg)

#171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23405

![](dt=2026-06-09/ht=11/f37ecce07bf9d84b929d1a1f4cc13bba89c3b7fed0c583cb541045c57e4f9d79.jpg)

![](dt=2026-06-09/ht=11/c626b3fa82ce5cb0c7def35789fd4da14e66e877e87d1e235ddc7a2a4bd3608c.jpg)

![](dt=2026-06-09/ht=11/a8915c50b54405c30c1825dfd1878bb05386e7fda561cd1ba9d4158244900e29.jpg)

![](dt=2026-06-09/ht=11/ef664e773df99f3a83cc1a93898a3cd3e6fa642c71e40a86fc82b4019b989993.jpg)

can efficiently compensate dispersion with continuous adjustment capabilities.

The optimized FrFT order $a_{\mathrm{opt}}$ was found by looking for the FrFT order $a$ for which the peak intensity of the PSF is maximized. Figure 4(b) presents a graph of the PSF peak intensity versus $a$ , which can be interpreted as a sharpness function. The peak intensity obtained with the optimized FrFT order is approximately 3.3 times higher than that obtained with the traditional Fourier transform (i.e., without dispersion compensation). This optimization only needs to be done occasionally and subsequent images can be analyzed using the same order value.

Note that during in-vivo imaging it can be challenging to find a refer
ence-interface that is suitable for optimization. For retinal imaging, it has been suggested to use the center of the fovea (fovealumbo) for this purpose [32]. In the next subsection, we will introduce another more systematic method.

To provide additional physical insights into FrFT-based dispersion compensation, Fig. 5(a) compares the measured unwrapped spectral phase and Fig. 5(b) the instantaneous spectral fringe frequency [i.e., 1st order derivative of the spectral phase, see Eq. (8)] without the water cell (dotted red), with the water cell (solid blue), and with the water cell plus FrFT dispersion compensation (dashed black). As can be seen, the presence of the water cell induces a strong spectral frequency modulation visible in the inclination of the solid blue curve in Fig.

5(b) compared with the dotted red curve, as was already discussed schematically in Fig. 1. After FrFT dispersion compensation, the curve becomes horizontal. It is even flatter than before the water cell is introduced, showing that the FrFT numerically compensates for both the dispersion of the water cell and the residual dispersion of the setup, i.e., what was left uncompensated by the physical introduction of the BK7 glass. Figure 5(c) and Fig. 5(d) show the Wigner distribution of the signal, respectively before and after FrFT dispersion compensation.

It is easily seen that the signal energy is entirely preserved and only rotated about the domains after FrFT.

For completeness, let us point out that the spectral phase of the FrFT dispersion compensated signal [dashed black curve in Fig. 5(a) and Fig. 5(b)] was obtained from the inverse Fourier transform of the FrFT-compensated complex spectrum (i.e. analytic signal),

$$
\varphi_ {\mathrm {F r F T}} = \arg \left\{F ^ {- 1} \left\{F ^ {a _ {\text {o p t}}} \left\{S (u) + i \hat {S} (u) \right\} \right\} \right\}. \tag {9}
$$

#171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23406

![](dt=2026-06-09/ht=11/f42c58b64245ecf2e5e5d09200f58aa1f3fcb625cc500c7e33732800e870950f.jpg)

Here $\hat{S}(u)$ is the Hilbert transform of the acquired spectrum $S(u)$ , and $F^{-1}$ represents the traditional inverse Fourier transform. To understand this expression, we need to recall that the Wigner transform of a real signal $S(u)$ is symmetric with respect to the transformed axis, $W_{S}(u,u_{1}) = W_{S}(u, -u_{1})$ . Upon projection on the optimal rotated $u_{a}$ axis [which amounts to perform the FrFT, see Eq.

(4)], out of the two mirror parts of the Wigner transform only one leads to a dispersion compensated response while the other leads to a broadened response in the other half of the transformed domain (not shown in Fig. 1 for simplicity). The broadened part of the response is in essence affected by twice the amount of dispersion present. It is the equivalent of the complex conjugate term encountered while using the traditional FT on a real signal.

We will refer to it as such, although strictly speaking it is not, in the general case ( $a \neq 1$ ), the mirror image of the other half of the fractional Fourier domain. This "complex conjugate" term needs to be eliminated or it will distort the retrieved spectral phase after inverse Fourier transform. We solve this problem by starting from the analytic signal.

Finally, the width of the PSF obtained with the water cell after FrFT numerical dispersion compensation was measured for different axial delays. These measurements are plotted in Fig. 6 and compared with that obtained without water cell (and using the traditional FT). The same FrFT order was used for all depths. The resolution observed in the two cases are in very good agreement. Higher order dispersion terms did not disturb our measurements.

However, we need to point out a slight increase $(7\%)$ of asymmetric side lobes of the PSF after numerical dispersion compensation, which we attribute to the third order dispersion of the water cell. A similar effect may also arise if higher bandwidths were used. In our current state of knowledge, we cannot readily compensate higher order dispersion using the FrFT but we do not preclude that this may be possible using transformations of the time-frequency plane more complex than rotations (i.e. linear projections).

# 4.2. Imaging

To further validate our method, we have performed FrFT numerical dispersion compensation on two particular samples. The first consists of two stacked $100\mu \mathrm{m}$ thick microscope cover slides. The poor surface flatness of the slides produced an air gap of approximately $20~{\mu\mathrm{m}}$ between them. Figure 7(a) was obtained using the traditional FT. The gap is blurred due to the poor axial resolution caused by the presence of the dispersive water cell. Using the optimized FrFT, the air gap between the two microscope slides can be resolved as seen in Fig. 7(b).

Similarly, Fig. 8(b) shows the cross sectional image of a grape using the traditional FT and Fig. 8(c) using the FrFT with optimized order, which offered an instantaneous dispersion compensated tomogram. Here, because of the complexity of the image, it was more challenging to use the same sharpness function as described in the previous Section to find the optimized

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23407

![](dt=2026-06-09/ht=11/ef272e66ee892873070238567cbd6f05d5d99c52e7a3f34e230b89312bc49ae2.jpg)

![](dt=2026-06-09/ht=11/b1674c8ca65c210ad896f16d4060d5d688f6625ffa096ed4aed47b7bb614ec0e.jpg)

![](dt=2026-06-09/ht=11/045e1210b1bac69f3e909ccbcbc53fd39e5cfc5c0f84b9f400e76413093edeb9.jpg)

order parameter $a_{\mathrm{opt}}$ . Instead, we have looked for the maximum of the Radon transform of the spectrogram of the full complex depth signal of one random A-scan. This Radon transform plot is shown in Fig. 8(a). Recall that the Radon transform of a two-dimensional distribution consists of a set of integral projections for various projection angles $\delta$ [43]. In essence, this procedure is therefore similar to taking the FrFT of the signal for a range of order parameter $a$ [which would correspond to the Radon transform of the Wigner distribution, see Eq.

(4)] where we can relate the projection angle $\delta$ and the order parameter $a$ through $\delta = a\pi /2$ . The important difference is that here we used the spectrogram of the signal rather than the Wigner distribution. The spectrogram is another time-frequency distribution that employs a windowed FT of the complex signal, i.e., the short-time FT of the complex signal. It has the advantage of not exhibiting cross-terms, in contrast to the Wigner distribution, because it is phase insensitive.

Consequently, the point of the Radon transform where the energy converges determines the optimized order parameter. For our grape image, a quick examination of Fig. 8(a) suggests an average optimized FrFT order $a_{\mathrm{opt}} = 1.04$ highlighted by the green line. It was used for all A-scans of the tomographic B-scan shown in Fig. 8(c). In contrast, the orange line in Fig. 8(a) represents the traditional FT which leads to Fig. 8(b).

Note that the grape was imaged from the bottom through a microscope cover slide and that the front interface of that slide was positioned ahead of the point of zero path difference of the interferometer in order to minimize the depth dependent sensitivity fall-off. The intensity maximum at approximately 2000 Radon bins and FrFT order $a = 0.96
$ therefore corresponds to the complex conjugate term of the front interface of the glass slide.

Its intensity maximum is at FrFT order $2 - a_{\mathrm{opt}}$ as it has opposite dispersion and therefore experiences twice the amount of dispersion after compensation compared to the traditional FT.

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23408

To complete this Section, we must note that the algorithm used for numerical computation of the FrFT, and which was introduced by Ozaktas et al. [40], is currently one order of magnitude slower than the Fast Fourier Transform (FFT) algorithm used to compute traditional FTs. With current computing technology, this would however not preclude the use of the FrFT for real time processing.

# 5. Depth-dependent sample dispersion compensation

An axial resolution below $3\mu \mathrm{m}$ is generally sensitive to sample dispersion, so that even thin sample layers cause a broadening of the PSF during imaging. In such situations, the depth-dependent dispersion of the sample is not negligible and must be taken into account in order to get the sharpest image at all depths. For a Gaussian source spectrum, the factor of axial resolution broadening $\sigma_{\mathrm{PSF}}$ due to a dispersive element of thickness $l$ and group velocity dispersion coefficient $\beta_{2}$ is given by

$$
\sigma_ {\mathrm {P S F}} = \sqrt {1 + \left(\frac {\pi^ {2} c ^ {2}}{2 \ln (2)} \frac {l \beta_ {2} \Delta \lambda_ {\mathrm {F W H M}} ^ {2}}{\lambda_ {c} ^ {4}}\right) ^ {2}}. \tag {10}
$$

Applying this relation to the parameters of the source used in our experiments (see Section 3) reveals that sample dispersion is negligible in our case: a $1\mathrm{mm}$ thick slide of BK7 glass $(\beta_{2} = 41~\mathrm{ps}^{2} / \mathrm{km})$ would only broaden the PSF by a factor of 1.2. Given the limitations of our equipment, we have therefore been unable to investigate experimentally depth-dependent dispersion compensation at this stage.

However, in order to demonstrate that the FrFT can be used to handle this problem, and for the completeness of this article, we present below a proof-of-principle demonstration of such capability based on numerical simulations. In all these simulations, we consider a source bandwidth $\Delta \lambda_{\mathrm{FWHM}} = 210\mathrm{nm}$ at $\lambda_0 = 710\mathrm{nm}$ center wavelength.

When the sample dispersion is not negligible, the optimal FrFT order required for dispersion compensation becomes a function of imaging depth. Figure 9 highlights this issue for a simulated sample made up of five identical $100\mu \mathrm{m}$ -thick microscope cover slides, stacked against each other, and for which we assume an average sample group velocity dispersion of $\beta_{2} = 54~\mathrm{ps}^{2} / \mathrm{km}$ . In Fig. 9(a), we have plotted the sharpness function [refer to Fig. 4(b)] at each interface, i.e., the intensity of the OCT signal at those depths, versus the FrFT order.

Because dispersion gradually increases across the sample, choosing the FrFT order to compensate dispersion and maximize the PSF at a certain depth means the signals simultaneously obtained at the other depths are not optimal. For completeness, Fig. 9(b) shows the optimum FrFT order required for dispersion compensation at each depth.

![](dt=2026-06-09/ht=11/22cb3d5d41495c64bf1b9aa81781f4ff41730912189d3ce39d1e91e6fad9e4ff.jpg)

![](dt=2026-06-09/ht=11/c1d46727332cbca9457736f4d54d46007a9ab6e1753ec535f9e0c720947b6f0f.jpg)

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23409

![](dt=2026-06-09/ht=11/85fba2a75686d711fead8db69416f70840ba47b76a16fadebda08be479fc1d0f.jpg)

![](dt=2026-06-09/ht=11/8c951992e6dd4524f0dd228a0df5b788f6202fa02af5c0254d35664cc54cfb35.jpg)

![](dt=2026-06-09/ht=11/4da95d55863be7d049019b3d3bc61c703f6f2206f894c290561b19d92cc75f32.jpg)

![](dt=2026-06-09/ht=11/73ec1531530a7da2b603b7350dd4348904ae0bf9df28261f5d906d6af6fae7e2.jpg)

![](dt=2026-06-09/ht=11/110fd16629faca9a246723e91aedee687bc0ad2631f57a3fadab32ba3cbb5d8e.jpg)

![](dt=2026-06-09/ht=11/3203931e00c8cc582ef38f0b3b60648cd39f43faffafa05a084bff97a28e3b4a.jpg)

![](dt=2026-06-09/ht=11/67a06c9462c6fbd66b73bdc969a1ed584a5fbe2e19fe11778ae8a73c3ff4527b.jpg)

To perform depth-dependent dispersion compensation, it is possible to generalize the approach used to sharpen the image of the grape (Fig. 8) and that was based on looking for a maximum of the Radon transform of a time-frequency distribution of the OCT signal. To illustrate this, we show in Fig. 10(a) a two-dimensional plot of the FrFT of the OCT signal of our simulated sample for a range of orders, which is equivalent to the Radon transform of the Wigner distribution of the signal.

We can immediately see that taking a cross-section of that plot along the diagonal dashed yellow line that links the maxima, and projecting it along the depth axis, leads to the depth-dependent dispersion compensated OCT depth scan shown in Fig. 10(d). In essence, this procedure can be interpreted as performing a "short time" FrFT of the signal and using a different optimal FrFT order for each depth. For comparison, we also show in Fig. 10(b) and Fig. 10(c) the depth scans obtained, respectively, when performing a

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23410

traditional FT (corresponding to the solid green line in the Radon plot) and when using the same FrFT order at all depths so as to compensate for the average dispersion (dotted red line in the Radon plot). In both cases, dispersion is compensated only for one optimized depth region, whereas other regions appear blurred.

Performing depth-dependent dispersion compensation by finding an optimized cross-section of the Radon plot of the signal can be generalized to more complex samples. In Fig. 11, we show similar plots to Fig. 10 but for a simulated sample which presents normal dispersion in a first $150~\mu \mathrm{m}$ -thick layer, with a group-velocity dispersion coefficient $\beta_{2}$ linearly increasing with depth from $54~\mathrm{ps}^2/\mathrm{km}$ to $97~\mathrm{ps}^2/\mathrm{km}$ , followed by a second layer with uniform anomalous dispersion, $\beta_{2} = -38~\mathrm{ps}^2/\mathrm{km}$ .

Here the optimized cross-section is made up of a quadratic and a linear section, which leads to the optimized A-scan plotted in Fig. 10(c). For comparison, Fig. 10(b) is the less satisfactory result obtained with the traditional FT.

OCT signals from real samples may be more challenging to process, but the problem of finding an optimal cross-section passing through maxima of the Radon plot as defined above seems amenable to an appropriate algorithm and may even be applicable to samples with no isolated scatterers. Indeed, a quick look at Fig. 8(a) obtained for our grape image shows that this procedure would have led to the horizontal dashed green line that corresponds to the optimal FrFT order, $a = 1.04$
.

Only one maximum, on the left, is excluded, but we have identified this as an artefact resulting from a spurious reflection in our OCT setup. It explains the thin horizontal line visible in both Fig. 8(b) and Fig. 8(c), which always appear at the same depth. We must also point out that third-order dispersion is too small to have an impact on depth dependent dispersion compensation and can therefore be neglected up to an axial resolution of $2\mu \mathrm{m}$ , for a center wavelength in the normal dispersion regime of water ( $\lambda_0 < 1\mu \mathrm{m}$ ) [44].

If water is the dominant medium in the sample, only if a better resolution is needed would the approach described here break down, as compensation of higher order terms may become necessary.

# 6. Group velocity dispersion measurement using FrFT

While dispersion causes broadening of the PSF and blurs images, it has the potential to provide additional functional information. For example, in tissues which exhibit regions with different group-velocity dispersion coefficients, dispersion information can be used for tissue differentiation. Liu et al. proposed to use material dispersion in order to differentiate water and lipid as a diagnosis of vulnerable plaques in the coronary arteries [34]. Obviously, this requires a mean to extract absolute or relative information about the group-velocity dispersion coefficient $\beta_{2}$ .

We show below that our FrFT-based dispersion compensation routine makes such information readily available. Note that the experimental estimation of the dispersion coefficient $\beta_{2}$ of a material is not only of interest for imaging, but has also applications in many other fields such as in fiber optics or in non-linear optics.

From the analysis of a chirp signal, Eq. (5), we have shown in Fig. 1 how the time-frequency representation provides a simple geometric relationship between the group-velocity dispersion coefficient $\beta_{2}$ and the FrFT order required for optimal dispersion compensation, $a_{\mathrm{opt}}$ . In particular, the modulation rate of the spectral fringes of the OCT signal [see Eq. (8)] can be readily extracted from Fig. 1 without the need of additional phase analysis:

$$
\tan (\pi - \phi) = \frac {\pi}{l \beta_ {2} (2 \pi \eta) ^ {2}}, \tag {11}
$$

where $\phi = a_{\mathrm{opt}}\pi /2$ defines the orientation of the optimal fractional Fourier domain. Solving for the group-velocity dispersion coefficient, and with $\eta = d\nu \sqrt{N}$ the normalization parameter

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23411

used in our digital implementation [Eq. (7)], yields

$$
\beta_ {2} = \frac {- \pi}{l N d \omega^ {2} \tan \left(a _ {\mathrm {o p t}} \pi / 2\right)}, \tag {12}
$$

where $d\omega = 2\pi d\nu$ is the angular frequency spacing between the samples of the OCT spectral signal. The above equation clearly shows that the absolute value of $\beta_{2}$ , including its sign, can be extracted once the optimal FrFT order for dispersion compensation has been obtained.

To demonstrate this capability, we have measured with this technique the group velocity dispersion of a short length of single mode fiber and of distilled water, both at a center wavelength of $843~\mathrm{nm}$ . The fiber sample was a $244~\mathrm{mm}$ long SM800 (Thorlabs) fiber, which has a mode field diameter of $5.6~{\mu\mathrm{m}}$ . The water sample was the same as used for the dispersion compensation experiments, i.e., $26~\mathrm{mm}$ thick. In both cases, we first determined the residual dispersion of the setup $(l\beta_{2})^{\mathrm{(setup)}}$ , using Eq.

(12) without the sample-under-test, by optimizing the point spread function with only a mirror in the sample arm. This was then subtracted from the corresponding measurement with the sample present $(l\beta_{2})^{\mathrm{(total)}}$ and normalized to the sample length to yield the dispersion coefficient $\beta_{2}$ of the sample itself,

$$
\beta_ {2} ^ {\text {(s a m p l e)}} = \frac {1}{l _ {\text {s a m p l e}}} \left[ (l \beta_ {2}) ^ {\text {(t o t a l)}} - (l \beta_ {2}) ^ {\text {(s e t u p)}} \right]. \tag {13}
$$

The results are shown in Table 1, and are in excellent agreement with reference values. The measured fiber dispersion agrees with a measurement that was obtained by traditional white light interferometry [27] while that of water is in good agreement with Van Engen et al., who measured a group velocity dispersion of water of approximately $22~\mathrm{ps}^2/\mathrm{km}$ at $840~\mathrm{nm}$ . The table also shows the optimal FrFT orders that were obtained first without then with the sample, the former value being reminiscent of the residual uncompensated dispersion of the setup.

Note how the residual setup dispersion is anomalous, $a_{\mathrm{opt}}^{\mathrm{(setup)}} < 1$ , for the fiber sample measurement but normal, $a_{\mathrm{opt}}^{\mathrm{(setup)}} > 1$ , for the water measurement. This is due to the empty water cell being included in the setup dispersion in the latter case.

Table 1. FrFT-based measurements of the group velocity dispersion coefficient $\beta_{2}$ of a single mode fiber and of distilled water. All the measurements have been done at $843~\mathrm{nm}$ but the reference value for water is for a $840~\mathrm{nm}$ wavelength. $N = 1001$ , $d\omega = 34.5\times 10^{10}$ rad/s.

![](dt=2026-06-09/ht=11/544e69a28172a0ca4970568f22a11ab8a9d3100ec0c6cdba9fe2e02f5fcb10f9.jpg)

<table><tr><td>Sample</td><td>l</td><td>\( a_{\text{opt}}^{\text{(setup)}} \)</td><td>\( a_{\text{opt}}^{\text{(total)}} \)</td><td>\( \beta_2^{\text{(sample)}} \)</td><td>Reference value</td></tr><tr><td>SM800</td><td>244 mm</td><td>0.9986</td><td>1.2161</td><td>38.3 ± 0.9 ps2/km</td><td>38 ps2/km [27]</td></tr><tr><td>H2O</td><td>26 mm</td><td>1.0059</td><td>1.0193</td><td>21.4 ± 0.4 ps2/km</td><td>22 ps2/km [22]</td></tr></table>

In OCT one generally has no information about the absolute thickness of depth layers in a cross sectional image as the refractive index of the material is unknown. However, using Eq. (12), one can readily obtain relative dispersion information by using the optimized FrFT order and the relative thickness of a layer. The relative thickness corresponds to $l = \bar{l} /n$ , where $n$ may be assumed as 1.33 and $\bar{l}$ is the layer thickness obtained from the OCT depth scan.

# 7. Conclusion

The fractional Fourier transform (FrFT) is a generalization of the traditional Fourier transform. The FrFT provides a new fundamental perspective on the nature and role of group-velocity dispersion in Fourier domain OCT and can be seen as a visual tool to highlight the physics behind

#171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23412

dispersion compensation. Using the FrFT one obtains depth information in FD OCT while simultaneously compensating for group velocity dispersion. Our theoretical axial resolution of $3.6\mu \mathrm{m}$ was recovered by optimizing the order parameter of the FrFT to compensate the group velocity dispersion induced by a $26~\mathrm{mm}$ long water cell for a source spectral bandwidth of $110~\mathrm{nm}$ . The technique was successfully demonstrated on a biological sample. Furthermore we provided new insights on the issue of depth-dependent dispersion.

Simulations have shown that both normal and anomalous sample dispersion can be addressed by dynamically adapting the order parameter as a function of depth. This method can be seen as analogous to a "short time" FrFT but is more efficient and intuitive and may even be applicable to samples without isolated scatterers. From the optimized FrFT order parameter one also readily obtains some qu
antitative information about the amount of dispersion in an OCT configuration and sample.

We have derived the relationship between the FrFT order parameter and group velocity dispersion and used it to successfully measure the group velocity dispersion coefficient of distilled water and some single mode fiber (at $840~\mathrm{nm}$ ).

# Acknowledgments

This work was supported by a NERF grant from the Foundation for Research Science and Technology from the New Zealand government.

171869 - $15.00 USD Received 3 Jul 2012; revised 6 Sep 2012; accepted 14 Sep 2012; published 26 Sep 2012

(C) 2012 OSA

8 October 2012 / Vol. 20, No. 21 / OPTICS EXPRESS 23413