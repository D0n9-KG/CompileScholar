Moving slice septa and pseudo three-dimensional reconstruction for multi-ring PET

This content has been downloaded from IOPscience. Please scroll down to see the full text.

1992 Phys. Med. Biol. 37 661

(http://iopscience.iop.org/0031-9155/37/3/012)

View the table of contents for this issue, or go to the journal homepage for more

Download details:

IP Address: 141.161.91.14

This content was downloaded on 26/08/2015 at 17:21

Please note that terms and conditions apply.

IOPscience

iopscience.iop.org

Home Search Collections Journals About Contact us My IOPscience

# Moving slice septa and pseudo three-dimensional reconstruction for multi-ring PET

E Tanaka†‡, S Morit, K Shimizu†, E Yoshikawa†, T Yamashita† and H Murayama‡

Received 21 October 1991

Abstract. This paper describes moving slice septa for reducing scattered and random events in multi-ring positron emission tomographs (PET) and a new simple method of three-dimensional (3D) image reconstruction which is useful when the maximum axial acceptance angle of coincidence is not too large. The moving septa considered are linear and sinusoidal piston motions of parallel septa, axis wobbling of parallel septa and rotation of spiral septa. The proposed reconstruction algorithm is basically a filtered(1D) backprojection(3D) method.

The low frequency image is reconstructed using direct (and cross) plane coincidence as in the conventional way, and the high frequency image is reconstructed using all projection data through high-pass filtering. The two images are superimposed in such a way that the final frequency response is normal. The axial cross talk is effectively eliminated by Gaussian high-pass filtering with negligible increase in statistical noise, and it is possible to include most of the oblique coincidence events in the reconstruction in a single pass.

Simulation studies with a maximum axial acceptance angle of $\pm 7.6$ degrees ( $\pm 10$ ring difference) show satisfactory results.

# 1. Introduction

Fully three-dimensional (3D) image reconstruction using a multi-ring positron emission tomograph (PET) is receiving increasing attention recently (Muehllehner et al 1988, Dahlbom et al 1989, Rogers et al 1989, Townsend et al 1989). To take full advantage of the 3D reconstruction, the slice septa are usually removed or retracted so that as many photons as possible are detected. Removal of the slice septa, however, results in appreciable increase in scattered events and random events (Thompson 1988, Thompson 1989).

On the contrary, the use of thin and short slice septa having a large axial acceptance angle is limited by septal penetration of photons. A possible compromise between with and without septa is moving coarse slice septa, which allows the use of sufficiently thick and long septa for negligible penetration with a moderate detection efficiency.

A number of 3D reconstruction algorithms have been reported. Defrise et al (1989) have recently summarized the linear shift-invariant formula for the 3D reconstruction. A practical difficulty in applying these algorithms to multi-ring PET systems is in the fact that the axial acceptance angle is not constant in the region of interest (Ra et al 1982, Cho et al 1983, Rogers et al 1987, Clack et al 1989, Defrise et al 1990). A solution to this problem is to divide an object space into a number of regions depending on the acceptance angles, and to reconstruct each of them with a shift-invariant formula

Phys. Med. Biol., 1992, Vol. 37, No 3, 661-672. Printed in the UK

0031-9155/92/030661+12$04.50 © 1992 IOP Publishing Ltd

661

(Ra et al 1982, Cho et al 1983) or to use iterative techniques in which the missing data for large oblique angles are estimated from the image reconstructed using small angle projections (Rogers et al 1987, Defrise et al 1990). This paper first presents the preliminary study on the performance of several types of the moving slice septa, and then describes a simple practical 3D reconstruction algorithm suitable for the PET systems in which a maximum axial acceptance angle is relatively small.

# 2. Moving slice septa

We consider the following four types of moving septa as shown in figure 1: linear piston motion and sinusoidal piston motion of parallel septa, axis wobbling of parallel septa and rotating spiral septa. As a typical example for numerical evaluation, we assume the following geometries: the detector ring diameter is $60\mathrm{cm}$ , the pitch of the detector rings, $6\mathrm{mm}$ , the pitch of the slice septa, $24\mathrm{mm}$ , the septal length, $10\mathrm{cm}$ , and the septal thickness, $3\mathrm{mm}$ . The septal material is assumed to be opaque to radiations.

The detection efficiencies of the four detector rings in a septal gap were calculated by a computer ray tracing technique as a function of the detector ring difference, $d$ , in coincidence ( $d = 0$ implies direct plane coincidence) as follows. On a plane passing through the detector ring axis, we generated a sufficiently large number of photon trajectories with uniform angular and spatial distribution, and determined the average fraction of the trajectories which did not intersect the septa at various septa positions.

In the above calculation, the effects of non-colinearity of photon pairs, positron range, and scattering were ignored.

![](dt=2026-06-04/ht=12/e7df3518f18e93d92d8895715b0a916e1190be0866d14de7eece4beac9d2fd61.jpg)

![](dt=2026-06-04/ht=12/6cebe517011a39f469be757abde42da7742dec999fff4536bdf6ddfd8fdd4f28.jpg)

![](dt=2026-06-04/ht=12/e2c0e60c7b3874af0d0714af61c7b929c9ba7ad532f251eb21ec0c6bca97b3ef.jpg)

Figure 2 shows the differential and the integral (cumulative) efficiencies. The four line styles indicate the efficiencies of the four detector rings in a septal gap (full curve, broken curve, dense dotted curve and sparse dotted curve, in order). The motion amplitude of the linear piston is assumed to be $24\mathrm{mm}$ (=septal pitch) so that the four detectors have the same performance. In the sinusoidal piston and the axis wobbling, the motion amplitude is determined in such a way that the four integral efficiency curves are as close as possible to each other. The optimal amplitude is $18\mathrm{mm}$ for the

662

E Tanaka et al

![](dt=2026-06-04/ht=12/3a7b6576e4a6d3ab63cc49c4a21c9b36d8ffb428f32e69e7e0872c6789566018.jpg)

sinusoidal piston and $22\mathrm{mm}$ for the axis wobbling. Since we expect that the differential efficiency for scattered events would be nearly constant for the ring difference, the results in figure 2 suggest that the maximum ring difference in the data collection should be limited to a certain number (around 10 in this particular case) in order to optimize the relationship between sensitivity and scatter fraction.

In the cases of the linear piston and the rotating spiral, the four efficiency curves are the same, and it is expected that the image reconstruction is performed satisfactorily with the pseudo 3D reconstruction algorithm described later, because an arbitrary point in an object is sampled with a constant efficiency for a given ring difference, $d$ . The

Moving septa and pseudo 3D reconstruction for PET

663

sinusoidal piston shows a similar performance to the linear piston with the advantage of an easier driving mechanism, i.e., smaller amplitude and smaller mechanical acceleration. However, the differential efficiencies of the axis wobbling are different from each other for the four detectors. This may cause artifacts in the reconstructed image unless suitable normalization for the efficiency is performed. In the practical application, however, the difference may be s
moothed out because the image is formed by a large number of coincidence lines with various ring differences, but the detailed performance has not yet been confirmed.

# 3. Pseudo 3D reconstruction algorithm

The proposed algorithm is basically a filtered backprojection algorithm. Observed 2D projections are first filtered one-dimensionally by a 'a modified convolution' along the transaxial direction ( $t$ -direction in figure 3), and the filtered projections are backprojected into an object space along the lines of coincidence. The modified convolution for oblique projections involves high (frequency)-pass filtering, as shown later, which eliminates the axial cross talk between slices due to the oblique backprojection.

In other words, the oblique projections are used only to reconstruct the high frequency components of images, and the low frequency components are reconstructed using projection data for $d = 0$ (or $d = 0$ , $\pm 1$ ) in the conventional way as a stack of 2D reconstructions. Accordingly, the slice images are reconstructed independently from each other. The acceptance angle should be constant in each slice, but it may differ from slice to slice.

Thus we can use a large acceptance angle for slices in the central region of the axial field of view, and a smaller angle for slices near the end of the field of view. The final frequency response of images is normalized after (or in the process of) the 3D backprojection.

The kernel of the modified convolution for a ring difference $d$ is given by

$$
g _ {d} (t) = g _ {0} (t) w _ {d} (t) ^ {*} \left[ \delta (t) - \frac {1}{\sqrt {2 \pi} \sigma_ {d}} \exp \left(- \frac {1}{2 \sigma_ {d} ^ {2}} t ^ {2}\right) \right] \quad (\sigma_ {d} = \infty \text {f o r} d = 0) \tag {1}
$$

where $g_0(t)$ is a convolution kernel for 2D reconstructions, $w_d(t)$ is the 'writing function' which is the density distribution deposited on an image slice by the backprojection of a unit beam, and $\delta(t)$ is the Dirac delta function. We use the Shepp-Logan filter as $g_0(t)$ throughout this work. The term in square brackets implies a Gaussian high-pass

![](dt=2026-06-04/ht=12/b86a46d16014f0aca32a40bb48440a64ca41182bbd4cbd4a6dd5f533e89d293e.jpg)

664

E Tanaka et al

filter, and the asterisk indicates convolution. The term $w_{d}(t)$ compensates for the effect of a finite length of the writing function as shown below.

In voxel driven backprojection with bi-linear interpolation, the writing function, $w_{d}(t)$ , is expressed by a triangle function:

$$
w _ {d} (t) = 1 - | t | / \mathrm {F W H M} _ {w} \quad \text {f o r} | t | <   \mathrm {F W H M} _ {w} \tag {2}
$$

$\mathbf{FWHM}_w =$ detector ring radius/d. $\mathbf{FWHM}_w$ is the full width at half maximum of the writing function. We assumed in the above that the width of the image slice is half the axial pitch of the detector rings. Neglecting high-pass filtering, the point spread function in the slice in which a point source exists is given by

$$
\operatorname {P S F} (r) = \sum_ {d} \left[ \varepsilon_ {d} \int_ {0} ^ {2 \pi} g _ {0} (r \sin \theta) w _ {d} (r \sin \theta) w _ {d} (r \cos \theta) d \theta \right] \tag {3}
$$

![](dt=2026-06-04/ht=12/e2d9c7a44a1e2d135092bd775fbd4f8af74fd1cf1a8619a62fea6cc33ca22602.jpg)

![](dt=2026-06-04/ht=12/f5471c58f0b5b74f14cf860f4008b3666a50d700dd3df7dcf821304f667dee06.jpg)

![](dt=2026-06-04/ht=12/2c908079dabe11638141be7d5cd2d2abda033dd3dc312465e541ccb359f9883a.jpg)

![](dt=2026-06-04/ht=12/29d25285bd67b6ab3447a0256ddf4305947c0bac832515886b020626ad5a803d.jpg)

![](dt=2026-06-04/ht=12/f635c7c72133372a13d2e50af4a4cb4158056f8e170278d6fc3ea55eebab522c.jpg)

![](dt=2026-06-04/ht=12/453de839b15ef0416b1b342069fc3430c9dff066f0d2bcf643c00ef168026fde.jpg)

![](dt=2026-06-04/ht=12/c2451f004e67be56ea3e0cda13dd99b6c53657c083c4b08bca5fb787a7f036e7.jpg)

![](dt=2026-06-04/ht=12/67f84f02860e65cfd92e5f823d02681ad0bbf4c8d078137c36d204a19b4f8c90.jpg)

![](dt=2026-06-04/ht=12/19f97494de511a18c964109a714a0c40eb25e073501d7b50db26b3b5406916ba.jpg)

![](dt=2026-06-04/ht=12/0eea65ce7cea1f60d6567c938d18021573835b70292bab55d1a4e0710c86181a.jpg)

![](dt=2026-06-04/ht=12/34a2c1c07f0d6f5dbd3bc4a97eae6e5d6cc818df7ee2c86bbf8ed2e3d80de2c8.jpg)

![](dt=2026-06-04/ht=12/e606f91fbddec455008fb1309fb9818b1e085afbc1f3ad0b8fac2cb7ec6c9b67.jpg)

Moving septa and pseudo 3D reconstruction for PET

665

Table 1. Noise increase, $R_{\mathrm{var}}$ , and cross talk of high-pass filters.

![](dt=2026-06-04/ht=12/9822ce0fc63622ba25f25f5f55063b838abb34e96b363a444122734ee24937b4.jpg)

<table><tr><td colspan="5">Variable filter</td><td colspan="5">Fixed filter</td></tr><tr><td rowspan="2">FWHMw</td><td rowspan="2">Rvara</td><td colspan="3">Cross talk (%)b</td><td rowspan="2">FWHMf(Pixels)</td><td rowspan="2">Rvara</td><td colspan="3">Cross talk (%)b</td></tr><tr><td>Slice 1</td><td>Slice 2</td><td>Slice 3</td><td>Slice 1</td><td>Slice 2</td><td>Slice 3</td></tr><tr><td>1</td><td>1.0025</td><td>+9.19</td><td>+1.92</td><td>+0.88</td><td>16</td><td>1.0022</td><td>+12.00</td><td>+3.94</td><td>+2.05</td></tr><tr><td></td><td>1.0083</td><td>-4.63</td><td>-0.33</td><td>-0.00</td><td></td><td>1.0075</td><td>-10.97</td><td>-2.13</td><td>-0.00</td></tr><tr><td>2</td><td>1.019</td><td>+3.71</td><td>+0.54</td><td>+0.25</td><td>8</td><td>1.017</td><td>+4.63</td><td>+1.04</td><td>+0.55</td></tr><tr><td></td><td>1.055</td><td>-2.15</td><td>-0.09</td><td>-0.00</td><td></td><td>1.056</td><td>-4.27</td><td>-0.57</td><td>-0.00</td></tr><tr><td>4</td><td>1.113</td><td>+1.44</td><td>+0.18</td><td>+0.08</td><td>4</td><td>1.124</td><td>+1.58</td><td>+0.27</td><td>+0.17</td></tr><tr><td></td><td>1.275</td><td>-1.01</td><td>-0.02</td><td>-0.00</td><td></td><td>1.354</td><td>-1.38</td><td>-0.10</td><td>-0.00</td></tr></table>

a Upper and lower values of $R_{\mathrm{var}}$ are for $\mathsf{FWHM_0} = 1.2$ pixels (4 mm) and 1.8 pixels (6 mm), respectively. b The values with plus (or minus) signs are total densities of positive (or negative) cross talk in the field of view having a diameter of 60 pixels (20 cm).

where $\varepsilon_{d}$ is the relative detection efficiency of the ring difference $d$ normalized at $d = 0$ . Since $g_{0}(r\sin \theta)$ has a large value only when $(r\sin \theta) \simeq 0$ where $w_{d}(r\sin \theta) \simeq 1$ and $w_{d}(r\cos \theta) \simeq w_{d}(r)$ , we can approximate equation (3) by

$$
\begin{array}{l} \operatorname {P S F} (r) \simeq \sum_ {d} \left[ \varepsilon_ {d} w _ {d} (r) \int_ {0} ^ {2 \pi} g _ {0} (r \sin \theta) d \theta \right] (4) \\ \simeq \sum_ {d} \varepsilon_ {d} \int_ {0} ^ {2 \pi} g _ {0} (r \sin \theta) d \theta . (5) \\ \end{array}
$$

This equation implies that the reconstruction is satisfactory since the integral on the right hand of equation (5) represents the point spread function of the conventional 2
D reconstruction. Note that equation (4) is accurate when $w_{d}(t)$ is Gaussian.

The Gaussian high-pass filter in equation (1) may or may not be constant for a different ring difference $d(\neq 0)$ . We consider two cases: one is the 'variable filters' case in which the FwHM, FwHMf = 2.355σd, of the Gaussian function is proportional to FwHMMw, and the other is the 'fixed filters' case, where FwHMMf has a fixed value for all $d(\neq 0)$ . The point spread function in the source slice and its cross talk to the nearby three slices were calculated assuming $d = -10 + 10$ and $\varepsilon_{d} = 1$ .

Figure 4 shows the point spread functions, (a) without filtering, (b) with a variable filter (FWHMMf = FwHMMw/2) and (c) with a fixed filter (FWHMMf = 8 pixels = 26.7 mm). The detector geometries are the same as those used in the simulation of the moving septa. The source is located at the centre of slice 0. The small undershoot in slice 0 of figure 4(b) and (c) will be removed by the following frequency normalization. It is seen that the cross talk is effectively eliminated by either of the high-pass filters. The cross talk with various filters is shown in table 1.

The values with plus (or minus) signs in the table are the total densities of the positive (or negative) components of the cross talk in the field of view (20 cm in diameter).

# 4. Analysis of the statistical noise in pseudo 3D reconstruction

A possible drawback of the pseudo 3D reconstruction may be the degradation of the signal to noise ratio which may occur by discarding the low frequency components

666

E Tanaka et al

for oblique projections. In this section, we evaluate the increase in the statistical noise. Neglecting the effect of $w_{d}(t)$ in equation (1), the frequency response of the backprojected image after the modified convolution is given by, in a polar coordinate $(f,\phi)$ :

$$
S (f, \phi) = \sum_ {d} \left\{\varepsilon_ {d} \left[ 1 - \exp \left(- 2 \pi^ {2} \sigma_ {d} ^ {2} f ^ {2}\right) \right] \exp \left(- 2 \pi^ {2} \sigma_ {0} ^ {2} f ^ {2}\right) \right\} \quad \left(\sigma_ {d} = \infty \text {f o r} d = 0\right) \tag {6}
$$

where the point spread function of the image reconstruction (not including detector resolution) is assumed to be a Gaussian function having a $\mathrm{FWHM_0} = 2.355\sigma_0$ . Consider a uniform cylindrical source placed coaxially with the detector rings for the noise evaluation. The frequency spectrum of the variance at the centre of the backprojected image (without frequency normalization) is given by

$$
V (f, \phi) = k N _ {0} f \sum_ {d} \left\{\varepsilon_ {d} \left[ 1 - \exp \left(- 2 \pi^ {2} \sigma_ {d} ^ {2} f ^ {2}\right) \right] \exp \left(- 2 \pi^ {2} \sigma_ {0} ^ {2} f ^ {2}\right) \right\} \tag {7}
$$

(Tanaka and Linuma 1975) where $N_0$ is the total counts in $d = 0$ and $k$ is a constant which depends on the diameter of the source (Tanaka and Murayama 1982). The frequency correction required for normalizing the backprojected image is given by

$$
C (f, \phi) = \exp \left(- 2 \pi^ {2} \sigma_ {0} ^ {2} f ^ {2}\right) / S (f, \phi). \tag {8}
$$

Then the variance of the final image after the normalization is given by

$$
\begin{array}{l} V _ {\text {p s e u d o}} = \int_ {0} ^ {\infty} \int_ {0} ^ {2 \pi} C ^ {2} (f, \phi) V (f, \phi) f \mathrm {d} \phi \mathrm {d} f (9) \\ = 2 \pi k N _ {0} \int_ {0} ^ {\infty} \frac {f ^ {2} \exp \left(- 4 \pi^ {2} \sigma_ {0} ^ {2} f ^ {2}\right)}{\sum_ {d} \left\{\varepsilon_ {d} \left[ 1 - \exp \left(- 2 \pi^ {2} \sigma_ {d} ^ {2} f ^ {2}\right) \right] \right\}} d f. (10) \\ \end{array}
$$

The denominator of the integrand of equation (10) denotes the effect of the high-pass filtering. The variance of an ideal reconstruction will then be obtained by putting $\sigma_{d} = \infty$ in equation (10):

$$
V _ {\text {i d e a l}} = \frac {k N _ {0}}{1 6 \pi \sqrt {\pi} \sigma_ {0} ^ {3}} \left(\sum \varepsilon_ {d}\right) ^ {- 1}. \tag {11}
$$

The relative increase of variance due to the high-pass filtering is then given by

$$
R _ {\text {v a r}} = \frac {V _ {\text {p s c u d o}}}{V _ {\text {i d e a l}}} = 3 2 \pi^ {2} \sqrt {\pi} \sigma_ {0} ^ {3} \sum_ {d} \varepsilon_ {d} \int_ {0} ^ {\infty} \frac {f ^ {2} \exp \left(- 4 \pi^ {2} \sigma_ {0} ^ {2} f ^ {2}\right)}{\Sigma_ {d} \left\{\varepsilon_ {d} [ 1 - \exp (- 2 \pi^ {2} \sigma_ {d} ^ {2} f ^ {2}) ] \right\}} d f. \tag {12}
$$

Since equation (12) includes $\sigma_0^3$ , the value of $R_{\mathrm{var}}$ depends strongly on the reconstruction resolution, $\mathrm{FWHM}_0 = 2.355\sigma_0$ . The values of $R_{\mathrm{var}}$ were evaluated for two $\mathrm{FWHM}_0$ values, 1.2 pixels ( $= 4\mathrm{mm}$ ) and 1.8 pixels ( $= 6\mathrm{mm}$ ). We assumed again $d = -10 + 10$ and $\varepsilon_d = 1$ . The results are summarized in table 1. It is seen that the increase of noise can be negligibly small (less than a few percent for $\mathrm{FWHM}_0 = 4\mathrm{mm}$ ) with a suitable choice of the filter parameters.

Figure 5 shows the frequency spectra of the $V_{\mathrm{pseudo}}$ (full curves), $V_{\mathrm{iddeal}}$ (broken curves) and the high-pass filter (dotted curves). It is shown that the increase of the noise with the variable filter is more widely distributed in the low frequency region than the fixed filter.

In the above calculations, we have assumed that the projections are fully sampled in the axial direction. In the stationary detector mode, however, the axial sampling is not sufficient, and the sampling is usually doubled by interleaving the average of the

Moving septa and pseudo 3D reconstruction for PET

667

![](dt=2026-06-04/ht=12/354c7ebe7169ea89137587b6ce676dd2d930d8510e180fb4547c3fc9b480cfdd.jpg)

![](dt=2026-06-04/ht=12/6e717842b633907c3646bff721a7c9195070168bd1897395e487ff7d73113ad7.jpg)

data in two adjacent azimuthal angles. In this case, the increase of variance in the 'cross planes' will be smaller than the values estimated above because the low frequency images are reconstructed using two ring differences, $d = \pm 1$ . It will also be possible to decrease the noise of the 'direct planes' by using the average of the three ring differences, $d = 0, \pm 2$ , in reconstructing the low frequency images with a little increase in the axial cross talk.

# 5. Phantom simulations with the pseudo 3D reconstruction algorithm

Simulation studies were made with two mathematical phantoms shown in figure 6. One is a uniform cylindrical phantom and the other consists of a rectangular block and a sphere. The detector ring diameter is $60\mathrm{cm}$ , the detector ring pitch is $8\mathrm{mm}$ , and

![](dt=2026-06-04/ht=12/97d36294dff8c42ed8dd3b6914e05b9c02a12314e81fb314095babdfa3bb02ce.jpg)

![](dt=2026-06-04/ht=12/a3a33e3b40517ee2a3213c4d05487089e365dd6bcbfd30f7efdb384d451a70a4.jpg)

668

E Tanaka et al

the imaging matrix is $64 \times 64 \times 32(\mathrm{H})$ with voxel size $4 \times 4 \times 4 \mathrm{~mm}^3$ . We used a fixed filter with $\mathrm{FWHM}_f = 8$ pixels, and assumed $d = -10 + 10$ and $\varepsilon_d = 1$ . Attenuation and scattering of photons were neglected and statistical noise was not included. The mathematical phantoms were projected along the lines of coincidence and the axial sampling was doubled by interleaving the average of two projections in adjacent azimuthal angles. The procedure of the reconstruction is as follows: the low frequency images are first reconstructed from projections for $d = 0$ by the conventional 2D reconstruction method except that the Shepp-Logan filter is smoothed by a Gaussian

![](image)
=image/dt=2026-06-04/ht=12//c8bd8832c49d7e7d68926f01752739607c559e459935b1a78348b16f17b57019.jpg)

![](dt=2026-06-04/ht=12/224d156e1309348471c7bc55c572a443e26dfbde0bfe73f23a66d78196e38af8.jpg)

![](dt=2026-06-04/ht=12/a6dfb21f8a23e510715ff2c64ef02209761b1fe7ea38a303d96b7a354bc61989.jpg)

![](dt=2026-06-04/ht=12/d56d3af4b9766ff3338b28f173618052432f1f9ae2a6c38525ce334d94cf91e8.jpg)

Moving septa and pseudo 3D reconstruction for PET

669

function having $\mathsf{FWHM}_f$ . Next, the high frequency images are reconstructed using all the projections ( $d = -10 \sim +10$ ). The high frequency images are superimposed on the low frequency images with a weight of $1 / \Sigma \varepsilon_d (= 1 / 21$ in this case) at the backprojection step, so that the overall frequency response is normalized.

Figures 7(a) and (b) show the cross-sections of the reconstructed image of the cylindrical phantom along the diameter and the axis, respectively, and (c) is the profile of the total density of slices. The total slice density at the centre is $2.1\%$ higher than the true value due to the cross talk. In figures 7(b) and (c), the difference in shape of the two edges is explained by the fact that the lower edge is a cross plane while the other is a direct plane.

Figure 8 shows the reconstructed images of the phantom shown in figure $6(b)$ . The images, from the upper left to the lower right, correspond to slices 10-21 in figure $6(b)$ . These results of the simulation studies indicate that the algorithm works adequately with negligible axial cross talk. The simulations were performed with a constant acceptance angle for all the slices, but it will be apparent that the algorithm works satisfactorily even when the acceptance angle differs from slice to slice, by the independency of the reconstruction of slice images.

# 6. Conclusion and discussions

The detection efficiencies for true coincidence of the four types of moving septa have been investigated. Although the integral efficiency curves are slowly varying functions of the ring difference, $d$ , the differential efficiency curves show large fluctuations as shown in figure 2. The differential detection efficiency curves for scattered coincidence, on the other hand, are expected to be smooth functions of $d$ . This suggests that the scatter fraction in the reconstructed image can be reduced by the image reconstruction in which larger weights are placed on the data with high efficiency ring difference than on the data with low efficiency. Further studies will be desirable on the possible improvements and on the detailed performance of the moving septa.

We have shown that the pseudo 3D reconstruction algorithm is a simple and practical method for PET systems as long as the axial acceptance angle is not too large. This requirement for the acceptance angle will not be serious even for systems having no slice septa, because the maximum acceptance angle is usually limited by detector end shields in order to prevent excess increase in scattered and random coincidence events.

The high-pass filtering for oblique projections effectively eliminates the axial cross talk with negligible increase in the statistical noise, although we have to compromise the two effects by the choice of the filter parameter. We also expect that the corrections for random coincidence and scattered coincidence may be skipped or simplified in the reconstruction of the high frequency images, because these events have little high frequency components except noise.

# Acknowledgments

We thank the other members of the PET Developing Group of HPK for their various and many contributions. We also thank Dr N Nohara and other colleagues in the National Institute of Radiological Sciences for their useful discussions on this work.

670

E Tanaka et al

# Résumé

Septa mobiles et pseudo-reconstruction tri-dimensionnelle pour le PET à multi-anneaux.

Ce travail décrit l'emploi de septa mobiles pour la réduction des événements diffusés et des événements aléatoires Presents dans les coupes en tomographie à émission de positons (PET), avec des apparciels multi-anneaux, ainsi qu'une nouvelle méthode simple de reconstruction d'images tri-dimensionnelles, utile quand l'angle maximal d'acceptance axiale des coincidences n'est pas trop grand. Le mouvement des septa peut être linéaire ou sinusoidal dans le sens longitudinal ou oscillant suivant leur axe, ou en rotation spiralée.

L'algorithmie de reconstruction proposé repose sur la méthode de rétroprojection (3D) filtrée (1D). L'image BASSE fréquence est reconstruite en utilisant une coincidence plane directe (et croisiée) comme la technique conventionnelle; l'image haute fréquence est reconstruite en utilisant toutes les données de projection à travers un filtre passé-haut. Les deux images sont superposées de telle sorte que la réponse spécifique的最后一 seule soit normale.

L'élimination efficace du recoupement axial (cross-talk) est réalisé par filtrage gaussian passé-haut entrainant une augmentation négligeable du bruit statistique. Il est possible que le 'cross talk' axial soit efficacement eliminé par un filtrage gaussian pase-haut avec une augmentation négligeable du bruit statistique, et il est possible d'inclure la plupart des événements en coincidence oblique dans la reconstruction en un seul passage.

Les études de simulation avec un angle maximal d'acceptance de $\pm 7.6^{\circ}$ (différence de $\pm 10$ anneaux) montrent que les résultats sont satisfaisants.

# Zusammenfassung

Bewegte Scheibenblenden und Pseudo-dreidimensionale Rekonstruktion bei der Multiring-PET

In der vorliegenden Arbeit werden bewegte Scheibenblenden zur Reduzierung von Streuereignissen und zufälligen Ereignissen bei Multiring-Positronenemissionstomographen (PET) beschreiben und eine neue, einfache Methode der dreidimensionalen (3D) Bildrekonstruktion, die nützlich ist, wenn der maximale axiale Aufnahmewinkel nicht zu groß ist. Bei den bewegten Blenden handelt es sich um lineare und sinusoidale Kolbenbewegungen von parallelen Blenden, Achsenverschiebungen paralleler Blenden und Rotation von spiralförmigen Blenden.

Der vorgeschlagene Rekonstruktionsalgorithmus ist im wesentlichen eine gefilterte (1D) Rückprojektions (3D) Methode. Das niedereffekte Bild wird reconstruiert durch Verwendung direkter (und schräger) Koinzidenzen in einer Ebene, wie bei konventionellen Verfahren. Das hochfrequente Bild wird reconstruiert unter Verwendung aller Rekonstruktionsdaten durch eine Hochpaßfilterung. Die beiden Bilder werden so überlagert, daß das endgültige Freqenzverhalten wieder normal ist.

Der axiale Kopiereffekt wird wirksam vermieden durch Gaußische Hochpaßfilterung mit vernachlässigbarem Anstieg des statistischen Rauschens. Außerden ist es möglich, die meisten der schrägen Koinzidenzereignisse bei der Rekonstruktion in einen einzigen Paß einschließlich. Simulationsuntersuchungen mit einem maximalen axialen Aufnahmewinkel von ±7.6 Grad (±10 Ringunterschiede) zeigen befriedigende Ergebnisse.

# References

Moving septa and pseudo 3D reconstruction for PET

671

672

E Tanaka et al