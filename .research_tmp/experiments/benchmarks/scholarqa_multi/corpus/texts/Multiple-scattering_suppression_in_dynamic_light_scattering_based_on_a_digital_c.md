# Multiple-scattering suppression in dynamic light scattering based on a digital camera detection scheme

Pavel Zakharov, Suresh Bhat, Peter Schurtenberger, and Frank Scheffold

We introduce a charge-coupled device (CCD) camera-based detection scheme in dynamic light scattering that provides information on the single-scattered autocorrelation function even for fairly turbid samples. It is based on the single focused laser beam geometry combined with the selective cross-correlation analysis of the scattered light intensity. Using a CCD camera as a multispeckle detector, we show how spatial correlations in the intensity pattern can be linked to single- and multiple-scattering processes.

Multiple-scattering suppression is then achieved by an efficient cross-correlation algorithm working in real time with a temporal resolution down to $0.02\mathrm{s}$ . Our approach allows access to the extensive range of systems that show low-order scattering by selective detection of the singly scattered light. Model experiments on slowly relaxing suspensions of titanium dioxide in glycerol were carried out to establish the validity range of our approach.

Successful application of the method is demonstrated up to a scattering coefficient of more than $\mu_{s} = 5\mathrm{cm}^{-1}$ for the sample size of $L = 1\mathrm{cm}$ . © 2006 Optical Society of America

OCIS codes: 290.0290, 290.4210, 030.6140, 040.1520.

# 1. Introduction

Dynamic light scattering (DLS) analyzes the intensity fluctuations of light scattered from a medium in the weak scattering limit. This is typically done by means of the normalized intensity autocorrelation function (ACF):

$$
g _ {2} (q, \tau) = \frac {\langle I (q , t) I (q , t + \tau) \rangle_ {t}}{\langle I (q , t) \rangle_ {t} ^ {2}}, \tag {1}
$$

where $\langle \cdot \cdot \rangle_t$ denotes time averaging, $\tau$ is the lag time and $q$ is the scattering wavenumber or momentum transfer defined by scattering angle $\theta$ , the wavelength in vacuum $\lambda$ , and the solvent refractive index $n$ : $q = 4\pi n \sin(\theta/2)/\lambda$ . The measurable quantity $g_2(q, \tau)$ can be linked to actual physical microscopic properties by the normalized field autocorrelation function $g_1(q, \tau)$ via the Siegert relation

$$
g _ {2} (q, \tau) = 1 + \beta | g _ {1} (q, \tau) | ^ {2}, \tag {2}
$$

where the coefficient $\beta$ depends on the detection optics.

Quite generally the field autocorrelation function $g_{1}(q, t)$ provides access to thermally driven local dynamic properties on length scales of the order of $1/q$ . A prominent example is the Brownian motion of colloidal particles in a solvent such as water. For this most simple case the normalized field correlation function can be written as<sup>1</sup>

$$
g _ {1} (q, \tau) = \exp (- D _ {0} q ^ {2} \tau) = \exp (- \tau / \tau_ {c}), \tag {3}
$$

where $\tau_{c} = 1 / Dq^{2}$ is the relaxation time and $D_0$ is the particle diffusion coefficient defined by the Stokes-Einstein relation

$$
D _ {0} = \frac {k T}{6 \pi \eta R}, \tag {4}
$$

where $\eta$ is the solvent viscosity, $T$ is the sample temperature in K, and $R$ is the particle radius. Equation (3) is widely used in dynamic light scattering for the sizing of small particles.

An essential condition for traditional dynamic light scattering to work is the absence of multiple scattering. As soon as higher-order scattering becomes considerable (typically if transmission in the line of sight is lower than $95\%$ ) the measured intensity correlation function starts to deviate from the theo

The authors are with the Department of Physics, University of Fribourg, CH-1700 Fribourg, Switzerland. P. Zakharov's e-mail address is Pavel.Zakharov@unifr.ch.

Received 18 May 2005; revised 28 July 2005; accepted 29 July 2005.

0003-6935/06/081756-09$15.00/0

© 2006 Optical Society of America

1756

APPLIED OPTICS / Vol. 45, No. 8 / 10 March 2006

retical expectations, leading to a faster decay. Furthermore, information on the scattering wavenumber and on the scattering angle $\theta$ is lost since the detected light is composed of several scattering events with unknown momentum transfer.

The influence of multiple scattering can be reduced by decreasing the concentration of the sample under study or the path inside the cell (e.g., using the corner of a square cuvette) or by refractive index matching. The latter is usually not possible without changing other sample properties. Limitations to the size of the container are set by the optical quality of the sample cells and by boundary effects. For cylindrical cells, minimal diameters of typically $1 - 3\mathrm{mm}$ are used, whereas in flat or rectangular containers even smaller photon path lengths are achieved.

To maximally reduce the photons' path lengths one can use fiber optic probes directly immersed in the (liquid) sample. This approach, known as fiber optic quasielastic light scattering (FOQELS), has been applied in a number of recent studies (see, e.g., Refs. 6-8). The application of FOQELS is, however, limited to backscattering angles of approximately $180^{\circ}$ , and moreover the interpretation of the data is often complicated because of the incomplete suppression of multiple scattering.

Quite a different way of actively dealing with multiple scattering has been put forward over the past two decades. The idea is to carry out two simultaneous DLS experiments with exactly the same scattering vectors in the same scattering volume and analyze the time cross-correlation function. It has been clearly shown that under proper conditions (see Refs. 9-11) the cross-correlation function equals the autocorrelation function for single scattering within the range of experimental resolution. Successful implementations of this scheme have been reported by several groups. The techniques are called two-color DLS (TCDLS) (Refs. 11 and 12) and three-dimensional DLS (3DDLS),[3,12] respectively, and the latter is available commercially.[13]

Another cross-correlation approach uses a single-beam two-detector configuration. [14,15] Suppression of multiple scattering is based on a consequence of the van Cittert-Zernike theorem, [16] which states that intensity correlations in an observation region are closely related to the Fourier transform of the intensity distribution across the source. This means that a small region of single scattering (e.g., the volume of a focused beam) will produce large correlated areas (speckles), whereas a comparably large halo of multiple scattered photons will give rise to small speckles [see Fig. 1(a)].

This is reflected in the well-known expression [16] for the speckle size $\delta x$ in the far-field geometry: $\delta x = \lambda z / S$ , where $z$ is the distance from the light emitting object to the detector and $S$ is the lateral extension of the object along one chosen direction. The consequences for a light scattering cross-correlation experiment are obvious. Spatially resolved detection of the scattered intensity will carry selective information about the spatial distribution of light in the scattering volume.

If we assume that the size of single-scattering vol

![](dt=2026-05-12/ht=04/5f836793302f0ddf992cb8b7d8ac7d86d8fb411c71b7a61bc78d45ee4596a484.jpg)

![](dt=2026-05-12/ht=04/79698b81d1a633eaf07113ec3f8816817e64dc7ecee16c994a0d0f9910b79d20.jpg)

ume $S_{1}$ is equal to an average radius of beam cross section of $S_{1} \approx 20\mu \mathrm{m}$ and the scattering mean free path to be $l \approx 1\mathrm{mm}$ , then the volume of double scattering extends over roughly a $25\times$ larger cross section. In practice the dimension of the detected scattering volume will determine $S_{2}$ both for weak and moderate multiple scattering.

Scattered intensities measured in two points separated by distance $\Delt
a x$ ,

$$
\frac {\lambda z}{S _ {1}} \geq \Delta x \gg \frac {\lambda z}{S _ {2}}, \tag {5}
$$

will thus be correlated only by single scattering as shown in Fig. 1(b) so the normalized intensity cross-correlation function (CCF),

10 March 2006 / Vol. 45, No. 8 / APPLIED OPTICS

1757

$$
g _ {2} ^ {\Delta x} (q, \tau) = \frac {\langle I (q , t , 0) I (q , t + \tau , \Delta x) \rangle_ {t}}{\langle I (q , t , 0) \rangle_ {t} \langle I (q , t , \Delta x) \rangle_ {t}}, \tag {6}
$$

will provide the proper estimate of the autocorrelation function of a singly scattered intensity. Such an approach has already been successfully demonstrated by Meyer et al.14 with the scheme based on the cross correlation of scattered intensities detected by two spatially separated fibers. While the underlying optical background (van Cittert-Zernike theorem) is highly plausible, it is more complicated to put forward a detailed theoretical description since this requires modeling of the low-order scattering processes. Such treatment has been derived by Lock17 for the case of double scattering.

He found the multiple-scattering suppression ratio to be approximately proportional to the speckle size ratio $S_{2} / S_{1}$ if the detectors are placed at the distance $\Delta x \approx \lambda z / S_{1}$ . The suppression ratio can be improved by choosing a larger separation $\Delta x$ , albeit at the cost of a decreased signal. Choosing a large distance, on the other hand, might prove unnecessary for small amounts of multiple scattering.

It is due to these practical difficulties that the technically simpler single-beam cross-correlation geometry is often considered inferior to the two-beam realization (where a sample-independent accurate theoretical description is available).

Here we propose an extension of the single-beam cross-correlation method that allows one to overcome this shortcoming. Using a charge-coupled device (CCD) camera as a detector, we can analyze speckle correlations and adapt $\Delta x$ , thus assuring single-scattering detection with high accuracy. As we will show, this flexibility, together with the intrinsically high statistical accuracy of multispeckle detection, leads to a much improved performance of the single-beam two-detector configuration while essentially preserving its technical simplicity.

# 2. Experimental Setup

The fluctuations of scattered light intensity are monitored in a traditional scattering geometry using a fixed scattering angle (Fig. 2). As a light source, the solid-state laser system (Verdi Coherent, Incorporated, Santa Clara, California) operating at the wavelength $\lambda = 532\mathrm{nm}$ is used. The beam is strongly focused by a lens with focal length $f = 50\mathrm{mm}$ inside the sample cell that results in an average beam radius of $20~\mu \mathrm{m}$ inside the scattering volume.

The focal spot was positioned in the center of a rectangular quartz cell (Hellma GmbH, Mullheim, Germany) with inner base dimensions of $10\times 10\mathrm{mm}$ and a height of $45\mathrm{mm}$ . The light scattered from the sample is recorded simultaneously by a single-mode fiber connected to a photon counting module (PerkinElmer, Canada, Quebec) and analyzed with a digital correlator (Correlator.com, Bridgewater, New Jersey) from one side and a CCD camera from the opposite side.

The CCD gray-scale camera Pixelfly produced by PCO Computer Optics GmbH, Bavaria, Germany, is configured to operate in videographics array (VGA)

![](dt=2026-05-12/ht=04/d681dcfe72cfac5af8135b083c20fbde707fe6c0c53f0488bd17fe400eab5da9.jpg)

mode $(640\times 480$ pixels) with 12 bit resolution and 50 frame/s speed with an exposure time of $\tau_0 = 20~\mathrm{ms}$ . This camera is based on a Sony ICX414AL image sensor with the pixel size of $9.9\times 9.9\mu \mathrm{m}$ . The dynamic range of analog-digital conversion of the CCD signal is $68.7\mathrm{dB}$ (according to manufacturer specifications). A diaphragm with an opening diameter of roughly $3.5\mathrm{mm}$ selects the range of scattering vectors seen by the CCD matrix and also determines the effective scattering volume. The distance from the beam center to the CCD matrix was $z = 11\mathrm{cm}$ . The minimum speckle size estimated for our system was approximately $14\mu \mathrm{m}$ or 1.4 pixels.

Scattering angles covered by the CCD chip are found in the range $\theta = 90^{\circ} \pm 3.15^{\circ}$ . The corresponding scattering vector is equal to $q = (24.59 \pm 0.67) \times 10^{6} \mathrm{~m}^{-1}$ with a deviation of the order of $2.75\%$ from the average $90^{\circ}$ angle. An improved angular accuracy could be achieved by placing the camera further away from the sample, however, at the cost of decreased statistical accuracy because of the smaller number of detected speckles. A better way to deal with this problem would be the processing of an angle-resolved digital image. However, this has not been realized in this study.

Simultaneously the intensity of the collimated beam transmitted through the cell is measured by a laser powermeter FieldMax (Coherent Incorporated). The temperature is monitored by a digital thermometer with the probe placed close to the cell surface.

As a model system we studied the Brownian motion of commercial $\mathrm{TiO_2}$ particles in pure $(99.5\%)$ glycerol. Solutions were sequentially passed through filters with pore diameters of 5 and $1.2~{\mu\mathrm{m}}$ .Measure

1758

APPLIED OPTICS / Vol. 45, No. 8 / 10 March 2006

ments on samples highly diluted with water reveal a mean hydrodynamic diameter for $\mathrm{TiO_2}$ particles of $293\pm 15~{\mu\mathrm{m}}$ , in qualitative agreement with electron micrographs. For all the measurements the temperature was in the range of $20^{\circ}\mathrm{C}\pm 0.5^{\circ}\mathrm{C}.$ The estimated solvent viscosity for current conditions is approximately 1.28 Pa s and corresponding variations are of the order of $\pm 5\%$ .

18 We select glycerol as the solvent to decrease the particle diffusion coefficient $D_{0}$ and thus enable real-time detection and processing of the scattered intensity fluctuations with a CCD camera. Glycerol is hygroscopic and it is thus difficult to know precisely the exact water content. As a consequence, a small uncertainty remains with respect to the solvent viscosity. We have ensured, however, that all the samples were prepared under identical conditions. Thus any systematic shift in the solvent viscosity will not affect the results of our study.

Typically $10^{4} - 10^{5}$ frames were collected corresponding to a measurement time of $15 - 30\mathrm{min}$ . The actual number of collected correlation coefficients depends on the processing scheme (see Section 3 for details) but usually is of the order of $10^{8}$ for the correlation coefficient of the smallest delay time.

The turbidity of the sample was characterized by means of the scattering coefficient $\mu_{s}$ . It is proportional to the particle density $\rho$ and the scattering cross section $\sigma_{s}:\mu_{s} = \rho \sigma_{s}$ . Because of the unknown amount of particles lost in the filtering process, however, no accurate density values are available. Since even fairly small particle densities lead to considerable multiple scattering, we can nevertheless safely assume that we are working in the highly dilute limit.

Experimentally $\mu_{s}$ can be estimated from the transmitted intensity $I_{t}$ according to the Lambert-Beer law for the case of nonabsorbing particles: $I_{t} / I_{0} = A\exp (-\mu_{s}L)$ , where $I_0$ is the intensity incident on the cell, $L$ is the cell thickness, and $A$ describes loss and deflection of intensity at the surface of the cell. The latter is independent of the particle density and was estimated from the transmission coefficient of a cell containing pure glycerol.

# 3. Processing Techniques

A key element of our study is the optimization o
f multispeckle detection and processing schemes. Our goal is to combine intelligent optical realizations with the power of modern parallel processing of a large amount of data acquired by a digital camera. Such optimized data analysis can be both achieved in the time and the space domain. As the first step we will discuss the application of the multi-tau scheme as developed by Schätzel to our multispeckle analysis. Second, we will show that a conceptually similar approach can be used in the space domain to improve both the multiple-scattering suppression efficiency and the processing speed of our CCD detection scheme.

# A. Multi-Tau Correlation Scheme

The multi-tau correlation scheme was originally proposed by Schätzel et al.19 to increase the accuracy of hardware correlators for large lag times. Recently, a number of software implementations have been proposed (see, e.g., Refs. 20 and 21). In most cases the linear spacing of lag times is not required to analyze single-scattering correlation functions. Instead, one can increase the distance between lag points in the correlation function for large lag times, thus saving valuable processing time. Such an approach can be easily and efficiently realized with a sequential doubling of the effective exposure time, which furthermore improves statistical accuracy.

Let us suppose that the camera provides data with an initial exposure time equal to $\tau_0$ with $1 / \tau_0\mathrm{Hz}$ frequency, which implies the absence of delays between the collection of sequential images. Then the doubling of the exposure time to $2\tau_{0}$ can be realized by integrating two sequential values of data obtained with exposure time $\tau_0$ . In the same way the effective exposure time can be increased to $4\tau_{0}$ and so on, as it is shown in Fig. 3(a). From the obtained time series of intensity fluctuations with different exposure times the correlation coefficients can be calculated with a simple linear scale processing.

In terms of software realization this is an ideal case for an object-oriented approach. The main object in this scheme is an elementary linear correlator shown in Fig. 3(b), which receives the sequential data point $I(i)$ as an input at each cycle $i$ , multiplies the new value with previous data points $I(i - k)$ for different linearly spaced integer delays $k = 0, 1, 2, \ldots$ , and updates the mean values of the corresponding products $\langle I(i)I(i - k) \rangle$ .

Every second cycle the correlator estimates the mean integrated value of intensity for two sequential time steps $[I(2n) + I(2n + 1)] / 2$ (where $n = 0, 1, 2, \ldots$ ) and sends it to the next linear correlator, thus forming a cascading line of correlators. Since the data rate on a sequential correlator is half of the data rate compared to the preceding one, the evaluation period doubles at every correlator. Extra effort has to be taken to optimize the cycles with respect to limitations in computational power.

The elementary correlator unit used in this study was designed to calculate the cross-correlation products together with the autocorrelation. ACF coefficients are calculated on every pixel individually to obtain local values of $\langle I(x,t)I(x,t + \tau)\rangle_t$ and $\langle I(x,t)\rangle_t$ . For the CCF with separation $\Delta x$ we obtain $\langle I(x,t)I(x + \Delta x,t + \tau)\rangle_t$ , $\langle I(x,t)\rangle_t$ , and $\langle I(x + \Delta x,t)\rangle_t$ . These values are accumulated in each correlator for further spatial averaging.

The scheme described here allows us to evaluate the data in real time and therefore does not set any limitations on the duration of the measurements.

# B. Binning Technique

The low dynamic range of CCD cameras in comparison to photodiodes or photomultipliers used in previ

10 March 2006 / Vol. 45, No. 8 / APPLIED OPTICS

1759

![](dt=2026-05-12/ht=04/376e74b80975de3e5a449efbc0aafcfbed471a0dde953803cfca91bf2d872174.jpg)

![](dt=2026-05-12/ht=04/b33419662043e99b3b1847a778d23a58811ecde2b379253fa69aca016d2ef625.jpg)

ous studies $^{11,14}$ is the main challenge for their utilization in a cross-correlation scheme that requires resolving the small signal from single scattering hidden by a dominating signal from multiple scattering. The problem can be partially solved with an original binning technique developed in the course of this study. If the size of a speckle from single scattering exceeds the area of several pixels, it can be approximated with integral values of these pixel intensities.

Let us call the bin, or metapixel, the area $S_{x} \times S_{y}$ pixels represented by a single intensity value obtained by integration (or floating-point averaging) of intensities of included pixels. Because of multiple sampling, the noise of measurements will be reduced by the factor $\sqrt{S_x S_y}$ (Ref. 22) and thus the dynamic range defined as $DR = 20 \log_{10} \mathrm{SNR} \mathrm{dB}$ , where SNR is a signal-to-noise ratio, will increase on $10 \log_{10} S_x S_y \mathrm{dB}$ .

For example, for a $10 \times 2$ window the dynamic range of our camera model is increased up to $81.7 \mathrm{~dB}$ or a $\mathrm{SNR} = 1.22 \times 10^4$ . The loss of spatial resolution reduces the statistical accuracy and intercept $\beta$ only if the binning area $\sqrt{S_x \times S_y}$ is comparable or larger than the coherence area

$\sqrt{S_1 \times S_2}$ . As a consequence, binning leads to partial suppression of multiple scattering by averaging out small speckles. Binning introduces a two-dimensional filtering with a boxlike kernel function that will reduce more efficiently the high-frequency spatial fluctuations (multiple-scattering speckles) than the lower-frequency fluctuations that are connected to single-scattering speckles. In other words, the binning technique in our cross-correlation approach can be considered a spatial analog of the multi-tau technique in the time domain described in Subsection 3.A.

# C. Multispeckle Averaging Technique

The capability of cameras to register a large number of independently fluctuating speckles simultaneously can be efficiently used to increase the statistical accuracy of a measurement by averaging the data along the bins. For the case of independent speckles different pixels or bins can be treated as separate photodetectors and thus their measurements can be processed altogether to estimate the correlation function. Actually two possible ways of dealing with these

1760

APPLIED OPTICS / Vol. 45, No. 8 / 10 March 2006

data exist. As mentioned in Subsection 3.A, correlators collect the time-averaged products $\langle I(x,t)I(x,t+\tau)\rangle_t$ , and the mean intensity $\langle I(x,t)\rangle_t$ for certain bins. Thus the further averaging followed by a normalization with the mean intensity can be performed with products and mean intensities as was proposed in the original multispeckle scheme[21,23,24]

$$
\begin{array}{l} g _ {2} ^ {a d} (\tau) = \frac {\langle \langle I (x , t) I (x , t + \tau) \rangle_ {t} \rangle_ {x}}{\langle \langle I (x , t) \rangle_ {t} \rangle_ {x} ^ {2}} \\ = \frac {\left\langle \left\langle I (x , t) I (x , t + \tau) \right\rangle_ {t} \right\rangle_ {x}}{\bar {I} ^ {2}}, \tag {7} \\ \end{array}
$$

where $\langle \dots \rangle_{x}$ denotes the averaging along the entire two-dimensional CCD matrix and $\bar{I}^2 = \langle \langle I(x,t)I(x,t + \tau)\rangle_t\rangle_x = \langle \langle I(x,t)I(x,t + \tau)\rangle_x\rangle_t$ is a mean value of intensity averaged over time and space. This "average-and-divide" sequence can be applied for the ideal case of a uniform illumination of the CCD matrix, i.e., when the mean intensity $\langle I(x,t)\rangle_t$ is not a function of $x$ . Because for $\tau \to \infty$ , one finds $I(x,t)$ , $I(x,t + \tau) \rightarrow \langle I(x,t)\rangle_t
$ , and $\langle I(x,t)I(x,t + \tau)\rangle_t \rightarrow \langle I(x,t)\rangle_t^2$ , then for this processing scheme

$$
\begin{array}{l} \lim  _ {\tau \rightarrow \infty} g _ {2} (\tau) = \frac {\langle \langle I (x , t) \rangle_ {t} {} ^ {2} \rangle_ {x}}{\bar {I} ^ {2}} \\ = \frac {\langle [ \langle I (x , t) \rangle_ {t} - \bar {I} ] ^ {2} \rangle_ {x}}{\bar {I} ^ {2}} + 1 \\ = K + 1, \tag {8} \\ \end{array}
$$

where $K = \langle [I(x,t) - \bar{I} ]^2\rangle_x / \bar{I}^2$ is the contrast of the picture that would be obtained with infinite exposure time. When illumination is uniform $[\langle l(x,t)\rangle_t = \bar{l}$ all along the detector matrix] contrast $K$ is equal to zero. In this case $g_{2}(\tau \rightarrow \infty)$ will approach 1 and thus the normalized field autocorrelation function estimated as $|g_{1}(\tau)|^{2} = [g_{2}(\tau) - 1] / \beta$ will decay to zero. But even for a slightly nonuniform illumination, when $K > 0$ , the nonzero additive component appears in the obtained $g_{1}^{2}(\tau)$ .

To overcome this problem the original multispeckle scheme was somewhat modified in our study by normalization of the locally estimated correlation coefficients and averaging only after that (divide-and-average sequence):

$$
g _ {2} ^ {d a} (\tau) = \left\langle \frac {\left\langle I (x , t) I (x , t + \tau) \right\rangle_ {t}}{\left\langle I (x , t) \right\rangle_ {t} {} ^ {2}} \right\rangle_ {x}. \tag {9}
$$

Still this requires the deviation of the mean intensity along the matrix to be small in comparison to the noise level. For a perfectly uniform illumination of the matrix both algorithms provide the same result. It is important to point out that the proposed normalization is not an inherent feature of our multiple-scattering suppression scheme but was used in order

to achieve a higher signal-to-noise ratio. If the optical setup is optimized for homogeneous illumination of the CCD detector, the scheme average and divide can be used without any problems. This is turn would provide access to arrested systems such as glasses and gels.

# 4. Results and Discussion

We have carried out a series of experiments using samples of different scattering strength, hence different amounts of multiple scattering, ranging from the dilute limit $\mu_{\mathrm{s}}L = 0.1$ to a regime $\mu_{\mathrm{s}}L = 5.92$ , where the transmitted beam is attenuated to only $0.27\%$ of its initial intensity.

# A. Spatial Intensity Correlations in the Speckle Pattern

As soon as multiple-scattering effects become considerable the detected intensity distribution will consist of a distribution of correlation lengths. The longest correlation length, $\delta x = \lambda z\approx 300~\mu \mathrm{m}\approx 30$ pixels, can be associated with single scattering, whereas higher-order scattering leads to smaller speckles. Figure 4 illustrates this by showing the intensity fluctuations along the CCD matrix columns. For the case of a single-scattering sample ( $\mu_{s} = 0.1\mathrm{cm}^{-1}$ , upper plot) the anisotropy of the speckle pattern can be clearly seen.

The reason for the anisotropy is the horizontal confinement of the incident focused beam that defines the scattering volume in this regime. The scale of intensity fluctuations along the $x$ dimension are of the order of tens of pixels so the correlated areas of intensity or speckles are large. With increasing scattering coefficient $\mu_{s}$ the fluctuations become more pronounced and the correlation length decreases. The horizontal speckle size is always given by the dimensions of the detected scattering volume both in the single- and multiple-scattering case.

For a quantitative estimation of the actual speckle size the normalized spatial autocorrelation function can be used

$$
g _ {2} ^ {s} (\Delta x) = \frac {\left\langle I (x) I (x + \Delta x) \right\rangle_ {x , t}}{\left\langle I (x) \right\rangle_ {x , t} \left\langle I (x + \Delta x) \right\rangle_ {x , t}}, \tag {10}
$$

where $\langle \cdot \cdot \cdot \rangle_{x,t}$ denotes the time and space averaging. In practice the averaging is performed through all pixel pairs separated vertically by a distance $\Delta x$ . A total of eight independent speckle images is analyzed with an exposure time of $20~\mathrm{ms}$ . Temporal fluctuations are much slower, typically of the order of $0.3~\mathrm{s}$ or more, and could thus be neglected.

Estimated values for the mean speckle size $\delta x$ , $\delta y$ can be defined by the condition $g_2^s (\Delta x,\Delta y\ll \delta y) - 1 = 1 / e$ , $g_2^s (\Delta x\ll \delta x,\Delta y) - 1 = 1 / e$ . Since speckles in the vertical direction are always small, the limit $\Delta y\ll \delta y$ cannot be reached in our setup and we have thus calculated the correlation length in the $\Delta x$ direction using the correlation function normalized with respect to the intercept $g_2^s (\Delta x = 0,\Delta y = 0)$ estimated for an essentially single-scattering sample.

This value is a good estimate of the $\beta$ factor for our setup. Spatial ACFs

10 March 2006 / Vol. 45, No. 8 / APPLIED OPTICS

1761

![](dt=2026-05-12/ht=04/aefe6071e7b974e795fc9dc261856bd269f80b2baabc20753acae5530ef0e9cd.jpg)

![](dt=2026-05-12/ht=04/81d371b41a4611e083293792363c98113f961eb5952fbffeae9c8f22670879fd.jpg)

![](dt=2026-05-12/ht=04/6a0cba653b6d07ff76d56fd33270745445d9607ccd25fbd99cf218ffcb15541a.jpg)

![](dt=2026-05-12/ht=04/422e4fa6b5d61058070f32b98c31c7e49d52e0d0cb743e96f8624070241cc2ac.jpg)

for samples of various turbidities are shown in Fig. 5, together with the estimated speckle sizes. In the vertical direction, with the increase of $\mu_{s}$ , the speckle size decreases rapidly. In the multiple-scattering limit, the speckles are almost isotropic with a size of approximately $1 \times 1$ pixels, which means that the initially focused beam is dispersed into a diffuse scattering cloud.

The spatial resolution in our experiments allows direct estimation of the relative weight of single-scattering contributions. For separations much lar

![](dt=2026-05-12/ht=04/b44e19fec40749b3edaa6cbd550ae7b3cc4de522d4e7a262dcb2ca9ef0ac54d7.jpg)

ger than 3 pixels, the cross-correlation signal will be dominated by single scattering. A separation comparable to the size of a single-scattering speckle, $\Delta x \approx 30$ pixels, is a good compromise value separating single from multiple scattering.

Note that by using a CCD camera we are, however, not restricted to fixed detector positions as in previous studies. Smaller separations $\Delta x$ could be used in the weak scattering limit whereas a gradual increase of $\Delta x$ in the strong scattering limit simultaneously optimizes multiple-scattering suppression and the signal-to-noise ratio.

# B. Pixel Binning

Instead of a pixel by pixel cross correlation, we can apply the superior binning approach described previously. Proper selection of the binning areas can increase the signal-to-noise ratio and also partially filter out the small multiple-scattering speckles. The latter effect is demonstrated in Fig. 6 for different numbers of binning pixels in the vertical direction. Since the horizontal speckle size is constant the binning in this dimension was selected to be 2 pixels.

Autocorrelation functions calculated from such larger areas are found increasingly closer to the single-scattering function as the binning area is increased. However, complete suppression is difficult to achieve since multiple-scattering suppression scales linearly with the number of pixels, whereas cross-correlation scales exponential
ly with pixel separation. Ideally the binning area is chosen somewhat smaller than the size of the single-scattering speckle to retain a high number of independent speckles. In our case the best choice in the $x$ direction is in the range of typically 5-20 pixels, as can be seen in Fig. 5.

1762

APPLIED OPTICS / Vol. 45, No. 8 / 10 March 2006

![](dt=2026-05-12/ht=04/3d89cfe2b56511bea3785952edb215a87519d2f0f73d18d0e772b5aeaa3644df.jpg)

# C. Intensity Correlation Functions

Figure 7 shows the correlation function for one of the most strongly scattering samples using different measurement schemes. With our optimized cross correlation $(\Delta x \approx 40$ pixels) and binning approach $(20 \times 2)$ , the measured correlation function perfectly agrees with the single-scattering result.

A set of measurements has been carried out and the measured particles' diameters are presented in Fig. 8 as a function of the scattering coefficient $\mu_{s}$ . For the case of dilute samples ( $\mu < 0.5\mathrm{cm}^{-1}$ ), all processing schemes yield the same results within the experimental error. Increasing $\mu_{s}$ results in smaller apparent

![](dt=2026-05-12/ht=04/a43deaf15ae0ac49c2ed93d2b41451c0877bb656356e81469b8c5d18cb5fa04b.jpg)

![](dt=2026-05-12/ht=04/cf8d14718ac1fb746cc2819918ec31023592dbb761fc6f66a7e7173cc7297eac.jpg)

values of the particle size for the case of autocorrelation measurements (ACF). At the same time the particle diameter obtained using our multiple-scattering suppression scheme (CCF processing) stays the same for all $\mu_{s} < 5.5\mathrm{cm}^{-1}$ corresponding to $\mu L = 5.5$ . Only for the most turbid sample, $\mu_{s} \approx 5.92\mathrm{cm}^{-1}$ , a noticeable difference is observed because of the loss of the single-scattering signal and the unavoidable increase of error.

Finally we would like to comment on the performance of the multispeckle averaging scheme implemented in our approach. Neglecting any angular dependence we are treating all $x$ columns equally. As we have seen, the size of a single-scattering speckle roughly amounts to $30 \times 2$ pixels. Since the full CCD chip has $640 \times 480$ pixels, we detect approximately 5000 independent speckles. This number can be increased only if the diameter of the incident beam is increased, but at the expense of inferior performance in multiple-scattering suppression.

# 5. Conclusion and Outlook

CCD-camera-based light scattering can be used to efficiently suppress the undesirable effects of multiple scattering. Our single-beam cross-correlation scheme combines, for the first time to our knowledge, the advantages of multiple-scattering suppression and multispeckle detection. The latter reduces the collection time for slowly evolving samples dramatically and provides the necessary statistical accuracy to detect even small cross-correlation signals deep in the multiple-scattering regime. Successful application of these combined processing schemes has been

10 March 2006 / Vol. 45, No. 8 / APPLIED OPTICS

1763

demonstrated for sample turbidities as large as $\mu_{\mathrm{s}}L$ $\approx 5$ .We think that our relatively simple approach, heavily relying on a flexible processing scheme, can be useful for a variety of applications for the study of complex fluids and soft materials. With the maximum frame rate available in our experiments, the most appropriate areas of application can be found in the field of slowly relaxing systems such as glasses and gels or dense surfactant solutions, to name a few.

At the current rate of development of image sensors and computer hardware, much smaller lag times seem to be feasible in the near future. Already now, with the use of modern fast complementary metaloxide semiconductor cameras with embedded programmable digital signal processors, the real-time on-camera calculation of the cross-correlation functions is theoretically possible, which, in turn, would improve the time resolution tremendously.

Part of this work was supported by KTI/CTI program TopNano21 and Nestlé Research Center, Lausanne. Support of the Swiss National Science Foundation is gratefully acknowledged.

# References

1764

APPLIED OPTICS / Vol. 45, No. 8 / 10 March 2006