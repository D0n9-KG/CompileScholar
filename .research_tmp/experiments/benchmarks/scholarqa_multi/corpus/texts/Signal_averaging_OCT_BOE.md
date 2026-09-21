# Signal averaging improves signal-to-noise in OCT images: But which approach works best, and when?

BERNHARD BAUMANN, $^{*}$ CONRAD W. MERKLE, RAINER A. LEITGEB, MARCO AUGUSTIN, ANDREAS WARTAK, $^{ID}$ MICHAEL PIRCHER, $^{ID}$ AND CHRISTOPH K. HITZENBERGER $^{ID}$

Center for Medical Physics and Biomedical Engineering, Medical University of Vienna, Währinger Gürtel 18-20, 4L, 1090 Vienna, Austria

\*bernhard.baumann@meduniwien.ac.at

Abstract: The high acquisition speed of state-of-the-art optical coherence tomography (OCT) enables massive signal-to-noise ratio (SNR) improvements by signal averaging. Here, we investigate the performance of two commonly used approaches for OCT signal averaging. We present the theoretical SNR performance of (a) computing the average of OCT magnitude data and (b) averaging the complex phasors, and substantiate our findings with simulations and experimentally acquired OCT data. We show that the achieved SNR performance strongly depends on both the SNR of the input signals and the number of averaged signals when the signal bias caused by the noise floor is not accounted for. Therefore we also explore the SNR for the two averaging approaches after correcting for the noise bias and, provided that the phases of the phasors are accurately aligned prior to averaging, then find that complex phasor averaging always leads to higher SNR than magnitude averaging.

Published by The Optical Society under the terms of the Creative Commons Attribution 4.0 License. Further distribution of this work must maintain attribution to the author(s) and the published article's title, journal citation, and DOI.

# 1. Introduction

Optical coherence tomography (OCT) [1] performs biomedical imaging at high speed and with high sensitivity. State-of-the-art OCT systems provide data rates of $\sim 100$ million pixels per second and frame rates of $\sim 100 - 200$ frames per second and beyond [2,3]. OCT provides a distinctively high sensitivity of typically $\sim 100$ dB, meaning that backscatter signals as weak as $10^{-10}$ of a mirror reflection can still be detected [4-6]. At the same time, OCT enables imaging with a very high dynamic range spanning several tens of decibels between the strongest and the weakest signal in the image.

The strong performance of OCT in detecting weak signals can be improved even more by image processing. Noise in OCT images limits the detection capabilities for weakly scattering structures, and thus a variety of different approaches for reducing OCT image noise have been proposed $[7–15]$ . The high acquisition speeds of modern OCT systems enable the improvement of the signal-to-noise ratio by averaging multiple signals, for instance by fusing OCT frames or even volumes quickly repeated at the same sample position. In recent years, several methods were proposed for improving the detection sensitivity of OCT by averaging the complex-valued OCT signals rather than just averaging their magnitudes $[16–20]$ . In 2013, Szkulmowski and Wojtkowski published a thorough analysis of signals and noise subject to different averaging approaches $[21]$ . In their analysis, the authors found a much stronger reduction of the noise floor by complex averaging as compared to magnitude averaging but also observed a heterogeneous outcome in terms of signal-to-noise performance for different imaging scenarios.

In this article, we set out to answer the question: Which signal averaging approach works best for improving signal-to-noise in OCT images, and when? We first introduce signals and noise in OCT (section 2.1) based on the analysis in Ref. [21]. We then analyze the signals and noise after magnitude and complex averaging, respectively, for a given pixel in an OCT image (section 2.2) and present simple expressions for the resulting signal-to-noise ratios. Next, we compare the somewhat surprising theoretical performance of the averaging schemes for different input signal levels and for different numbers of averaged signals and introduce a noise bias corrected signal-to-noise ratio (SNR) analysis (section 2.3). After substantiating our theoretical analysis with data from simulations (section 3.1) and experimentally acquired OCT data (section 3.2), we conclude the paper with a brief summary of the findings and their implications on actual OCT image processing (section 4). An overview of the terminology as well as a brief section on phase correction required for complex phasor averaging is provided in the appendix.

# 2. Analysis of signals and noise in OCT

# 2.1. OCT signals and noise

OCT signals $S_{OCT}$ can be described by complex phasors of the form

$$
S _ {O C T} (\mathbf {x}, t) = A (\mathbf {x}, t) \exp [ i \phi (\mathbf {x}, t) ] \tag {1}
$$

where $A(\mathbf{x}, t)$ denotes the amplitude and $\phi(\mathbf{x}, t)$ the phase of the OCT signal at location x and time t. Here the amplitude $A(\mathbf{x}, t)$ represents the length of the phasor and the phase $\phi(\mathbf{x}, t)$ corresponds to the polar angle in the complex plane. The noise $S_{noise}$ in OCT images can be specified by phasors too, namely by random phasors. In the complex plane, the real and imaginary parts of random phasors, $r_{noise}$ and $i_{noise}$ , can be described by normal distributions with zero mean and $\sigma^{2}$ variance [22]. Hence, the complex noise signal is characterized by a binormal (or Beckmann) distribution $p_{ri}$ in the complex plane [22,23]:

$$
p _ {r i} (r _ {n o i s e}, i _ {n o i s e}) = \frac {1}{2 \pi \sigma^ {2}} \exp \left[ - \frac {r _ {n o i s e} ^ {2} + i _ {n o i s e} ^ {2}}{2 \sigma^ {2}} \right]. \tag {2}
$$

This 2D Gaussian probability density function (PDF) describes the probability for a noise phasor of an image pixel to have the real part $r_{noise}$ and the imaginary part $i_{noise}$ . The probability density function $p_A$ of the noise amplitude $A_{noise}$ is represented by the Rayleigh distribution (which can be derived by transforming Eq. (2) to polar coordinates and integrating over all angles $\phi$ ) [22]

$$
p _ {A} (A _ {\text { noise }}) = \frac {A _ {\text { noise }}}{\sigma^ {2}} \exp \left[ - \frac {A _ {\text { noise }} ^ {2}}{2 \sigma^ {2}} \right]. \tag {3}
$$

Analogous to $p_{ri}$ , the PDF of the noise amplitude represents the probability of the noise amplitude to take specific values $A_{noise}$ . Figure 1(a-d) shows an example of the complex representation of noise phasors in an OCT image and the corresponding Beckmann and Rayleigh distributions. Note that the Beckmann distribution is symmetric and centered at the origin while the Rayleigh distribution is skewed and only takes positive values. The mean amplitude $\overline{u}$ and variance $\sigma_{U}^{2}$ of a general PDF $p_{U}(u)$ is given by $\overline{u} = \int_{-\infty}^{\infty} up_{U}(u) du$ and $\sigma_{U}^{2} = \int_{-\infty}^{\infty} (u - \overline{u})^{2} p_{U}(u) du$ , respectively [22]. The mean amplitude $\overline{A_{noise}}$ and the variance $\sigma_{noise}^{2}$ of $p_{A}$ can be computed as

$$
\overline {{{A _ {n o i s e}}}} = \sqrt {\frac {\pi}{2}} \sigma , \tag {4}
$$

$$
\sigma_ {n o i s e} ^ {2} = \left(2 - \frac {\pi}{2}\right) \sigma^ {2}. \tag {5}
$$

In every OCT image pixel, the detected signal is a combination of the actual OCT signal $S_{OCT}$ and the noise $S_{noise}$ . By choosing the OCT signal to have $\phi = 0$ and thus to point into the

direction of the positive real axis (Fig. 1(e)) similar to Goodman [22], the PDF of the measured amplitudes incorporating both signal and noise is represented by a Rice distribution

$$
p _ {A} (A, A _ {\text { noise }}) = \frac {A _ {\text { noise }}}{\sigma^ {2}} \exp \left[ - \frac {A _ {\text { noise }} ^ {2} + A ^ {2}}{2 \sigma^ {2}} \right] B _ {0} \left[ \frac {A _ {\text { noise }} A}{\sigma^ {2}} \right]. \tag {6}
$$

where $B_{0}$ is a modified Bessel function of the first kind and zero order. An example of the PDF of the signal amplitudes in presence of noise seen in an OCT image is shown in Fig. 1(g). Note that the signal gives rise to a somewhat spread probability accumulation at a high amplitude level. The mean amplitude $\overline{A}$ and variance $\sigma_{A}^{2}$ in an image pixel are observed as

$$
\overline {{A}} = \sqrt {\frac {\pi}{2}} \sigma L _ {1 / 2} \left[ - \frac {A ^ {2}}{2 \sigma^ {2}} \right], \tag {7}
$$

$$
\sigma_ {A} ^ {2} = A ^ {2} + 2 \sigma^ {2} + \frac {\pi}{2} \sigma L _ {1 / 2} ^ {2} \left[ - \frac {A ^ {2}}{2 \sigma^ {2}} \right], \tag {8}
$$

where $L_{1/2}(\cdot)$ denotes the Laguerre polynomial of degree 1/2. $L_{1/2}(-x)$ yields steadily increasing, positive values for increasingly negative arguments -x.

OCT amplitude data are usually squared in order to display a quantity proportional to sample reflectivity, $I = A^{2}$ , henceforth called intensity. Therefore it makes sense to investigate the behavior of the PDFs of the squared signal and noise amplitude data. For this purpose, a variable

![](images/d161f034063af3ccba2d5de941c9e967fc0eed9b69befb9855e573cf47cee7f0.jpg)

<details>
<summary>text_image</summary>

(a)
Im
Snoise
inose
rnoise
Re
noise
Probability
low high
</details>

![](images/2f4c91ca078deb8da5314beb7943ed3305f1d8dd82e82cffdb3dab66b7aa0234.jpg)

<details>
<summary>scatter</summary>

| Real part | Imaginary part |
| --------- | -------------- |
| -0.5      | 0.0            |
| 0.0       | 0.0            |
| 0.2       | 0.1            |
| 0.3       | -0.1           |
| 0.4       | 0.2            |
| 0.5       | -0.2           |
| 0.6       | 0.3            |
| 0.7       | -0.3           |
| 0.8       | 0.4            |
| 0.9       | -0.4           |
| 1.0       | 0.5            |
</details>

![](images/588790f6071c3409b44d3dedc4e8d0dea5f0996e224c652e9c996d0f4783d97b.jpg)

<details>
<summary>line</summary>

| Amplitude | Count / Probability density |
| --------- | ---------------------------- |
| 0.0       | 0                            |
| 0.1       | 45                           |
| 0.2       | 30                           |
| 0.3       | 15                           |
| 0.4       | 5                            |
| 0.5       | 2                            |
| 0.6       | 1                            |
| 0.7       | 0                            |
| 0.8       | 0                            |
| 0.9       | 0                            |
| 1.0       | 0                            |
</details>

![](images/88d40ae76a0c0a1f0ecc99c3c81bd8da4cdbb1759cd46cf8930b23a6d5a41cb5.jpg)

<details>
<summary>line</summary>

| Intensity | Count / Probability density |
| --------- | ---------------------------- |
| 0.0       | 70                           |
| 0.5       | 0                            |
| 1.0       | 0                            |
</details>

![](images/0291b64a2a95ab33cdf7d172886b319e592e21ffdeed89d484522c47ae42c16a.jpg)

<details>
<summary>heatmap</summary>

| Signal Type | Probability |
|-------------|-------------|
| Soct Re    | High        |
</details>

![](images/5ce4bc783032d8d23a752f611be2f5ed0209b9a7abde6ea2bb3e064238fb8bf8.jpg)

<details>
<summary>scatter</summary>

| Real part | Imaginary part |
| --------- | -------------- |
| -0.5      | -0.3           |
| -0.4      | -0.2           |
| -0.3      | -0.1           |
| -0.2      | 0.0            |
| -0.1      | 0.1            |
| 0.0       | 0.2            |
| 0.1       | 0.3            |
| 0.2       | 0.4            |
| 0.3       | 0.5            |
| 0.4       | 0.6            |
| 0.5       | 0.7            |
| 0.6       | 0.8            |
| 0.7       | 0.9            |
| 0.8       | 1.0            |
</details>

![](images/5b0edf728c11d8d9e809c98ac1d6eaff92230de7dac82371faf50f60db309286.jpg)

<details>
<summary>line</summary>

| Amplitude | Count / Probability density |
| --------- | ---------------------------- |
| 0.0       | 0                            |
| 0.5       | 10                           |
| 0.6       | 30                           |
| 0.7       | 45                           |
| 0.8       | 60                           |
| 0.9       | 40                           |
| 1.0       | 0                            |
</details>

![](images/cfbe437aa3b955d1aa1dba3152ba8522238ef2c5427100747e9bcdadd9f25c19.jpg)

<details>
<summary>line</summary>

| Intensity | Count / Probability density |
| --------- | ---------------------------- |
| 0.0       | 0                            |
| 0.1       | 5                            |
| 0.2       | 15                           |
| 0.3       | 30                           |
| 0.4       | 45                           |
| 0.5       | 50                           |
| 0.6       | 40                           |
| 0.7       | 25                           |
| 0.8       | 10                           |
| 0.9       | 5                            |
| 1.0       | 0                            |
</details>

Fig. 1. Complex phasor representation of noise and signals in OCT and their probability density functions. (a) Cartoon of Beckmann distribution of noise phasors around the origin of the complex plane. A representative phasor $S_{noise}$ with real part $r_{noise}$ and imaginary part $i_{noise}$ is shown in green. (b) Complex OCT signals of 100 repeated noise measurements in the same pixel from real-world OCT data. (c) Histogram of noise amplitudes (1000 repeats, gray line) and Rayleigh PDF (red line) computed from the standard deviation $\sigma$ of the Beckmann distribution in (b) by Eq. (3). (d) Histogram of the noise intensity (gray line) and PDF (red line) computed from $\sigma$ in (b) by Eq. (9). (e) Cartoon of an OCT signal affected by noise. The green arrow represents the signal phasor. (f) Complex OCT signals of 100 repeated measurements of a weak reflection in the same pixel. (g) Histogram of signal amplitudes (1000 repeats, gray line) and Rice distribution (blue line) computed from the mean signal amplitude and $\sigma$ using Eq. (6). (h) Histogram of the signal intensity (gray line) and PDF (blue line) computed from the mean intensity and $\sigma$ by Eq. (10). The + in (b) and (f) indicates the origin of the complex plane. The PDFs in (c,d,g,h) were scaled to match the count levels of the respective histograms.

transform of $A = \sqrt{I}$ is performed such that the PDFs are rendered into $p_{I}(I) = p_{A}(A = \sqrt{I}) \left| \frac{dA}{dI} \right|$ . The resulting PDFs for noise ( $p_{I}(I_{noise})$ ) and signal intensities ( $p_{I}(I, I_{noise})$ ) are

$$
p _ {I} (I _ {\text { noise }}) = \frac {1}{2 \sigma^ {2}} \exp \left[ - \frac {I _ {\text { noise }}}{2 \sigma^ {2}} \right], \tag {9}
$$

$$
p _ {I} (I, I _ {\text { noise }}) = \frac {1}{2 \sigma^ {2}} \exp \left[ - \frac {I _ {\text { noise }} + I}{2 \sigma^ {2}} \right] B _ {0} \left[ \frac {\sqrt {I _ {\text { noise }} I}}{\sigma^ {2}} \right], \tag {10}
$$

where $I_{noise} = A_{noise}^{2}$ . Note that $p_I(I_{noise})$ is a negative exponential probability function whereas $p_I(I, I_{noise})$ is a combination of a negative exponential and a monotonically increasing Bessel function. Examples of the above PDFs for noise and signal intensities are shown in Figs. 1(d) and (h), respectively. The mean values and variances for noise and signal intensities, respectively, are given by

$$
\overline {{{I _ {\text { noise } e}}}} = 2 \sigma^ {2}, \tag {11}
$$

$$
\sigma_ {I _ {\text { noise }}} ^ {2} = 4 \sigma^ {4}, \tag {12}
$$

$$
\bar {I} = I + 2 \sigma^ {2}, \tag {13}
$$

$$
\sigma_ {I} ^ {2} = 4 \sigma^ {2} (I + \sigma^ {2}). \tag {14}
$$

By comparing Eqs. (11) and (12), one can observe that the mean noise intensity equals the standard deviation $\sigma_{noise}$ . Therefore, for the case of both non-zero signal and noise, the observed mean intensity corresponds to $\overline{I}=I+\overline{I_{noise}}$ and the intensity variance to $\sigma_{I}^{2}=\overline{I_{noise}}(2I+\overline{I_{noise}})$ . Next, we will investigate the effect of averaging the magnitudes of the phasors as well as the effect of averaging the complex phasors on OCT intensity data.

# 2.2. Averaging magnitudes and phasors

For averaging OCT data, usually the absolute values (magnitudes) of the signals are used. For some applications, the complex signals have been exploited for adding or increasing image contrast in one way or another $[31–36]$ . Here, we systematically analyze the effect of averaging magnitude and complex signals in OCT images.

# 2.2.1. Magnitude averaging

Magnitude averaging uses the absolute values of N spatially and/or temporally separated signals (e.g., N pixels in a 2D or 3D kernel or N repeated measurements at the same spatial pixel location but at different time points):

$$
\langle I \rangle_ {M A G} = \frac {1}{N} \sum_ {j = 1} ^ {N} I _ {j}. \tag {15}
$$

The PDF representing the average of N data points can be computed numerically by convolving the PDF of a single data point $(N - 1)$ -times [21]. Unfortunately, no closed form expressions are available for the distribution describing an N-fold average of the above PDFs and only some approximations and bounds have been derived [37]. Nevertheless, the mean intensity values and intensity variances can be calculated for Rayleigh distributed noise and Rician signals affected by

random noise, respectively, as

$$
\overline {{\langle I _ {\text { noise }} \rangle_ {M A G}}} = 2 \sigma^ {2} = \overline {{I _ {\text { noise }}}}, \tag {16}
$$

$$
\langle \sigma_ {I _ {\text { noise }}} ^ {2} \rangle_ {M A G} = \frac {1}{N} 4 \sigma^ {4} = \frac {1}{N} \sigma_ {I _ {\text { noise }}} ^ {2}, \tag {17}
$$

$$
\overline {{\langle I \rangle_ {M A G}}} = I + 2 \sigma^ {2} = \bar {I}, \tag {18}
$$

$$
\langle \sigma_ {I} ^ {2} \rangle_ {M A G} = \frac {1}{N} 4 \sigma^ {2} (I + \sigma^ {2}) = \frac {1}{N} \sigma_ {I} ^ {2}. \tag {19}
$$

The above expressions show that the mean noise level as well as the observed mean signal level remain unchanged upon averaging and retain the mean intensity values calculated for single data points. However, the variances of the observed signal and noise, $\sigma_{I}^{2}$ and $\sigma_{I_{noise}}^{2}$ , are reduced by 1/N and thus reducing the standard deviations $\sigma_{I}$ and $\sigma_{I_{noise}}$ by a factor $1/\sqrt{N}$ .

# 2.2.2. Complex averaging

In contrast to magnitude averaging, complex averaging also includes the phase information into the averaging process. This inclusion exploits the fact that the noise phasors (in absence of a signal) randomly fluctuate about the origin according to a 2D Gaussian PDF with a standard deviation of $\sigma$ , see Eq. (2). By averaging $N$ spatially and/or temporally independent complex noise phasors, the standard deviation $\sigma$ of the Beckmann distribution in Eq. (2) is reduced by a factor $1/\sqrt{N}$ [22]. This reduction of the noise amplitude variance translates into a reduction of both the mean noise intensity as well as of its variance:

$$
\overline {{\langle I _ {\text { noise }} \rangle_ {C P X}}} = \frac {1}{N} 2 \sigma^ {2} = \frac {1}{N} \overline {{I _ {\text { noise }}}}, \tag {20}
$$

$$
\langle \sigma_ {I _ {\text { noise }}} ^ {2} \rangle_ {C P X} = \frac {1}{N ^ {2}} 4 \sigma^ {4} = \frac {1}{N ^ {2}} \sigma_ {I _ {\text { noise }}} ^ {2}. \tag {21}
$$

Compared to the noise performance after magnitude averaging (Eqs. (16) and (17)), which maintains the mean noise level and reduces the noise intensity variance by 1/N, complex averaging leads to a 1/N decrease of the noise floor intensity and to a $1/N^{2}$ smaller noise intensity variance.

Recalling that we initially assumed that the signal vector $S_{OCT}$ was pointing in the direction of the positive real axis (i.e. Fig. 1(e)), the precondition for averaging signals in a complex fashion is that the phase of $S_{OCT}$ remains constant at $\phi = 0$ . (Likewise, a different signal phase $\phi = \phi_{0}$ could be chosen, as long as the phases of all signals to be averaged are aligned at the same angle $\phi_{0}$ . The signal phase ( $\phi = 0$ ) is chosen here out of convenience to ensure a purely real signal which facilitates the calculations described above.) For such perfectly aligned signals in the presence of random noise, the following mean signal intensity and intensity variance will be observed:

$$
\overline {{\langle I \rangle_ {C P X}}} = I + \frac {1}{N} 2 \sigma^ {2} = \bar {I} - \frac {N - 1}{N} 2 \sigma^ {2}, \tag {22}
$$

$$
\langle \sigma_ {I} ^ {2} \rangle_ {C P X} = \frac {1}{N} 4 \sigma^ {2} (I + \frac {1}{N} \sigma^ {2}) = \frac {1}{N} \sigma_ {I} ^ {2} - \frac {N - 1}{N ^ {2}} \sigma^ {2}. \tag {23}
$$

Unlike magnitude averaging, which maintains the original mean signal intensity $\bar{I}$ , the mean signal intensity does not stay constant but is reduced by $2\sigma^{2}(N-1)/N$ after complex averaging. This signal reduction converges to an amount of $2\sigma^{2}$ for large N. At the same time, also the noise intensity variance is reduced by an amount $\sigma^{2}(N-1)/N^{2}$ compared to magnitude averaging, which converges to a reduction by $\sigma^{2}/N$ for large N.

# 2.3. Signal-to-noise performance

The signal-to-noise ratio is the measure of choice to describe the influence of noise on signal measurements. An overview of methods for determining the SNR and in particular the sensitivity of an OCT system has recently been published by Agrawal et al. [24]. The SNR in OCT is typically defined as the ratio of the average signal intensity to the standard deviation of the noise intensity, $SNR = \overline{I}/\sigma_{I_{noise}}$ [24–28] (or alternatively $\overline{\langle I\rangle}/\langle\sigma_{I_{noise}}\rangle$ for averaged signals). As shown in the previous sections and visualized in Fig. 2, the measured signal intensity $\overline{I}$ is the sum of the pure signal intensity I and the noise floor $\overline{I_{noise}}$ . Historically, the bias caused by the noise floor is not specifically factored out from the SNR analysis, however some investigations of OCT image data did exclude the background [29,30]. Because of the multiple approaches for OCT data processing, it makes sense to ask, "how important is noise bias correction when measuring SNR?". In the following section, we will investigate the SNR for the total measured signal (i.e. including the contribution of the noise floor) with and without averaging. Subsequently, we will account for the signal offset caused by the noise floor and evaluate the SNR with noise bias correction.

![](images/16b6a4bddd52570702b5200fb12c8f07228dba3f8f462f39eeb3922fdde7f26e.jpg)

<details>
<summary>line</summary>

| Depth | Intensity |
|-------|---------|
| 0     | 0       |
| 1     | ~0.2    |
| 2     | ~0.3    |
| 3     | ~0.4    |
| 4     | ~0.6    |
| 5     | ~1.0    |
| 6     | ~0.3    |
| 7     | ~0.2    |
| 8     | ~0.1    |
| 9     | ~0.1    |
| 10    | ~0.1    |
| 11    | ~0.1    |
| 12    | ~0.1    |
| 13    | ~0.1    |
| 14    | ~0.1    |
| 15    | ~0.1    |
| 16    | ~0.1    |
| 17    | ~0.1    |
| 18    | ~0.1    |
| 19    | ~0.1    |
| 20    | ~0.1    |
| 21    | ~0.1    |
| 22    | ~0.1    |
| 23    | ~0.1    |
| 24    | ~0.1    |
| 25    | ~0.1    |
| 26    | ~0.1    |
| 27    | ~0.1    |
| 28    | ~0.1    |
| 29    | ~0.1    |
| 30    | ~0.1    |
| 31    | ~0.1    |
| 32    | ~0.1    |
| 33    | ~0.1    |
| 34    | ~0.1    |
| 35    | ~0.1    |
| 36    | ~0.1    |
| 37    | ~0.1    |
| 38    | ~0.1    |
| 39    | ~0.1    |
| 40    | ~0.1    |
| 41    | ~0.1    |
| 42    | ~0.1    |
| 43    | ~0.1    |
| 44    | ~0.1    |
| 45    | ~0.1    |
| 46    | ~0.1    |
| 47    | ~0.1    |
| 48    | ~0.1    |
| 49    | ~0.1    |
| 50    | ~0.1    |
| 51    | ~0.1    |
| 52    | ~0.1    |
| 53    | ~0.1    |
| 54    | ~0.1    |
| 55    | ~0.1    |
| 56    | ~0.1    |
| 57    | ~0.1    |
| 58    | ~0.1    |
| 59    | ~0.1    |
| 60    | ~0.1    |
| 61    | ~0.1    |
| 62    | ~0.1    |
| 63    | ~0.1    |
| 64    | ~0.1    |
| 65    | ~0.1    |
| 66    | ~0.1    |
| 67    | ~0.1    |
| 68    | ~0.1    |
| 69    | ~0.1    |
| 70    | ~0.1    |
| 71    | ~0.1    |
| 72    | ~0.1    |
| 73    | ~0.1    |
| 74    | ~0.1    |
| 75    | ~0.1    |
| 76    | ~0.1    |
| 77    | ~0.1    |
| 78    | ~0.1    |
| 79    | ~0.1    |
| 80    | ~0.1    |
| 81    | ~0.1    |
| 82    | ~0.1    |
| 83    | ~0.1    |
| 84    | ~0.1    |
| 85    | ~0.1    |
| 86    | ~0.1    |
| 87    | ~0.1    |
| 88    | ~0.1    |
| 89    | ~0.1    |
| 90    | ~0.1    |
| 91    | ~0.1    |
| 92    | ~0.1    |
| 93    | ~0.1    |
| 94    | ~0.1    |
| 95    | ~0.1    |
| 96    | ~0.1    |
| 97    | ~0.1    |
| 98    | ~0.1    |
| 99    | ~0.1    |
| 100   | ~0.1    |
</details>

Fig. 2. Intensity signal and noise background in a schematic OCT depth profile. The noise floor is characterized by its average intensity $\overline{I_{noise}}$ and its variance $\sigma_{I_{noise}}^{2}$ . The measured OCT intensity signal $\overline{I}$ consists of the pure signal intensity I biased by the average noise level $\overline{I_{noise}}$ .

# 2.3.1. Signal-to-noise performance without noise bias correction

Using the signal and noise pairs in Eqs. (13) and (12), (18) and (17), and (22) and (21), and the relation $\overline{I_{noise}} = \sigma_{I_{noise}} = 2\sigma^{2}$ in Eqs. (11) and (12), we find the following SNRs for single signals, N magnitude averaged signals, and N complex averaged signals, respectively:

$$
\mathrm{SNR} _ {1} = \frac {\bar {I}}{\sigma_ {I _ {\text { noise }}}} = \frac {I + \overline {{I _ {\text { noise }}}}}{\sigma_ {I _ {\text { noise }}}}, \tag {24}
$$

$$
\mathrm{SNR} _ {M A G} = \frac {\overline {{\langle I \rangle_ {M A G}}}}{\left\langle \sigma_ {I _ {\text { noise }}} \right\rangle_ {M A G}} = \frac {I + \overline {{I _ {\text { noise }}}}}{\frac {1}{\sqrt {N}} \sigma_ {I _ {\text { noise }}}} = \sqrt {N} \cdot \mathrm{SNR} _ {1}, \tag {25}
$$

$$
\mathrm{SNR} _ {C P X} = \frac {\overline {{\langle I \rangle_ {C P X}}}}{\left\langle \sigma_ {I _ {\text {noise}}} \right\rangle_ {C P X}} = \frac {I + \frac {1}{N} \overline {{I _ {\text {noise}}}}}{\frac {1}{N} \sigma_ {I _ {\text {noise}}}} = N \cdot \mathrm{SNR} _ {1} - (N - 1). \tag {26}
$$

In short, averaging the magnitudes of N signals improves the SNR by a factor $\sqrt{N}$ whereas complex averaging of N signals changes the SNR by a factor N less $(N - 1)$ .

The relative SNR improvements $SNR_{MAG,CPX}/SNR_{1}$ and $SNR_{CPX}/SNR_{MAG}$ are plotted for N up to 100 and for typical OCT image dynamics with $SNR_{1}$ between 5 dB and 20 dB in Fig. 3.

In particular for small N, complex averaging provides a considerable SNR improvement over magnitude averaging. At the same time, the $SNR_{1}$ dependence of complex averaging has a considerable impact on its performance: While for strong signals with $SNR_{1} \gg 1$ , complex averaging outperforms magnitude averaging by a factor $\sqrt{N}$ , the signal-to-noise improvement becomes much less for weaker input signals.

(a)   
![](images/54a4ce6066d26bec27edbda690f0e9c8dfbadb4f9725f5b4156b95dd479be89f.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | SNR_CPX/SNR₁ | SNR_MAG/SNR₁ |
| ---------------------------- | ------------ | ------------ |
| 0                            | 0            | 0            |
| 20                           | ~12          | ~5           |
| 40                           | ~16          | ~7           |
| 60                           | ~18          | ~8           |
| 80                           | ~19          | ~9           |
| 100                          | ~20          | ~10          |
</details>

(b)   
![](images/e9656f6fcc793620b03ae6c9c7367929c553f0a5413cad09a163a4c61a8467de.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | 5 dB  | 10 dB | 15 dB | 20 dB |
| ---------------------------- | ----- | ----- | ----- | ----- |
| 0                            | 1.0   | 1.0   | 1.0   | 1.0   |
| 20                           | 3.0   | 4.0   | 5.0   | 6.0   |
| 40                           | 4.5   | 6.0   | 7.5   | 8.5   |
| 60                           | 5.5   | 7.5   | 9.0   | 9.5   |
| 80                           | 6.5   | 8.5   | 10.0  | 10.5  |
| 100                          | 7.0   | 9.0   | 10.5  | 11.0  |
</details>

Fig. 3. Relative SNR improvement by signal averaging for strong input signals (without noise bias correction). (a) The ratios $SNR_{CPX}/SNR_{1}$ and $SNR_{MAG}/SNR_{1}$ are shown for N from 1 to 100. $SNR_{MAG}/SNR_{1}$ is plotted as a dash-dotted line, whereas the $SNR_{1}$ -dependent ratio $SNR_{CPX}/SNR_{1}$ is plotted in rainbow colors for several $SNR_{1}$ values between 5 dB and 50 dB. Note that $SNR_{CPX}/SNR_{1}$ converges to an N-fold improvement. (b) The ratio $SNR_{CPX}/SNR_{MAG} = (N - (N - 1)/SNR_{1})/\sqrt{N}$ is plotted for the spectrum of $SNR_{1}$ values used in (a). Note that in particular for strong input signals with high $SNR_{1}$ , complex averaging outperforms magnitude averaging and converges to a $\sqrt{N}$ -fold better SNR performance. As the SNR profiles converge for large SNRs, the curves in (a) and (b) start to overlap for values greater than 15 dB.

In the next section, we are going to explore the SNR enhancement of the two averaging approaches particularly for very weak signals on the order of the noise background, however still without correcting for the signal bias caused by the noise floor. Will magnitude averaging or complex averaging be superior in terms of recovering such small signals?

# 2.3.2. Averaging domains and borderline SNR

When the signal bias caused by the noise is not accounted for, the SNR performance averaging depends on the approach taken. Equations (25) and (26) revealed a dependence on the number of averaged signals, N, and – for complex averaging – on the SNR of the input signals, $SNR_{1}$ . In this section, we are investigating this dependence for different numbers of averaged signals, N, and for a great range of signals – from much smaller than the noise variance to several orders of magnitude greater. Recalling the identities for the noise variance and the observed signal intensity in Eqs. (12) and (13), we choose the quantity $I/2\sigma^{2}$ , which is identical to $SNR_{1}-1$ , as the benchmark for the input signal.

Figure 4(a) shows three plots of the signal-to-noise ratios calculated for magnitude averaging (red) and complex averaging (blue) where relatively small signals $I/2\sigma^{2}=1$ , 0.5 and 0.1 were chosen as the respective inputs. Note that for $I=2\sigma^{2}$ (left plot), complex averaging always yields greater SNR than magnitude averaging for N>1. However, for $I/2\sigma^{2}<1$ and a small number of signals N, magnitude averaging provides a more effective SNR improvement. Figure 4(b) plots three pairs of SNR profiles for a fixed number of averaged signals (N=2, 10 and 100), this time as a function of the relative signal intensity $I/2\sigma^{2}$ . Again, for weak input signals, the superior performance of magnitude averaging can be observed, while complex averaging excels beyond a

crossover point of $I/2\sigma^{2}=1/\sqrt{N}$ . This relative borderline signal is plotted for N up to 100 in Fig. 4(c) alongside the borderline input $SNR_{1}$

$$
\mathrm{SNR} _ {1, \text { borderline }} = 1 + \frac {1}{\sqrt {N}} \tag {27}
$$

which is the input $\mathrm{SNR}_1$ for which magnitude averaging and complex averaging perform equally well.

(a)   
![](images/4815d31af55a5af2411f2d025e9c8e7bb00a68defcf1977dbcddfa6614941ce9.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N (log) | MAG [dB] | CPX [dB] |
|---|---|---|
| 1 | 3 | 3 |
| 100 | 13 | 20 |
N=1, I/2σ²=1
</details>

![](images/ec9137893b2f314315abff71fb14ba871161a49a1419947898cc0ac28e7172a0.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N [log] | MAG SNR [dB] | CPX SNR [dB] |
| ---------------------------------- | ------------ | ------------ |
| 1                                  | 2            | 2            |
| 10                                 | 8            | 10           |
| 100                                | 12           | 17           |
</details>

![](images/80807fa27c4cf3c79974b25478419d24a4a58a137f4e99910c44f4ec5f1a93b2.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N [log] | MAG SNR [dB] | CPX SNR [dB] |
| ---------------------------------- | ------------ | ------------ |
| 1                                  | 0            | 0            |
| 10                                 | 5            | 3            |
| 100                                | 10           | 10           |
</details>

(b)   
![](images/02a755d43b250ae603c0af11b1e31f8cc2ed3f6a035ab391b7afa76212eed0a0.jpg)

<details>
<summary>line</summary>

| Relative signal intensity I/(2σ²) [log] | MAG SNR [dB] | CPX SNR [dB] |
| -------------------------------------- | ------------ | ------------ |
| 0.01                                   | ~0           | ~0           |
| 0.1                                    | ~1           | ~1           |
| 1                                      | ~5           | ~5           |
| 10                                     | ~15          | ~15          |
</details>

![](images/45532e5a7aefa380a3d5d8eb5657bd9efa8c90fc65544e27a428db9161db65eb.jpg)

<details>
<summary>line</summary>

| Relative signal intensity I/(2σ²) [log] | MAG SNR [dB] | CPX SNR [dB] |
| -------------------------------------- | ------------ | ------------ |
| 0.01                                   | ~5           | ~0           |
| 0.1                                    | ~6           | ~5           |
| 1                                      | ~10          | ~15          |
| 10                                     | ~15          | ~20          |
</details>

![](images/bcd69f20805e3e016efe5e646a53d41929c9aefae9f3f04b3a98a97b10a6747e.jpg)

<details>
<summary>line</summary>

| Relative signal intensity I/(2σ²) [log] | MAG SNR [dB] | CPX SNR [dB] |
| -------------------------------------- | ------------ | ------------ |
| 0.01                                   | 10           | 5            |
| 0.1                                    | 10           | 10           |
| 1                                      | 15           | 20           |
| 10                                     | 20           | 30           |
</details>

(c)   
![](images/2e098b54308c5142017c5a487b2b85ce0f12937e17564fcaa3e2a24cac2eb7af.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N [log] | Borderline I/2σ² [dB] |
| ---------------------------------- | --------------------- |
| 1                                  | 0                     |
| 10                                 | -5                    |
| 100                                | -10                   |
</details>

![](images/8282e6b2b683021f48765e8ec411393682563a6361fb94990af6b28370e0573d.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N [log] | Borderline SNR₁ [dB] |
| ---------------------------------- | -------------------- |
| 1                                  | 3.0                  |
| 10                                 | 1.5                  |
| 100                                | 0.5                  |
</details>

Fig. 4. Influence of input signal level and number of averaged signals on the SNR (without noise bias correction). (a) $SNR_{MAG}$ and $SNR_{CPX}$ after magnitude and complex averaging of N signals plotted for relative signal levels of $I/2\sigma^{2}=1$ (left), $I/2\sigma^{2}=0.5$ (middle), and $I/2\sigma^{2}=0.1$ (right), respectively. (b) $SNR_{MAG}$ and $SNR_{CPX}$ after magnitude and complex averaging of signals $I/2\sigma^{2}$ ranging from 0.01 through 10, plotted for averages of N=2 (left), N=10 (middle), and N=100 signals (right), respectively. Green arrows in (a) and (b) indicate the intercepts of the SNR profiles, i.e. the borderline SNR where magnitude and complex averaging perform equally. (c) Borderline plots of $I/2\sigma^{2}=1/\sqrt{N}$ as well as $SNR_{1}$ as described in Eq. (27).

An overview of the relative SNR performance $SNR_{CPX}/SNR_{MAG}$ at various inputs $I/2\sigma^{2}$ and N is shown as a heat map in Fig. 5. At first glance, two domains can be differentiated: For stronger signals $I/2\sigma^{2}$ and larger N, complex averaging provides a greater SNR improvement than magnitude averaging (see the area in blue in Fig. 5). For signals comparable to or smaller than the noise level, magnitude averaging yields a better SNR enhancement (see the area in red).

In between these two domains, the borderline with equal SNR performance of the two approaches is indicated in white, $SNR_{CPX}/SNR_{MAG} = 1$ .

![](images/64957c341a101fa2b2aef7d3b63f6ab859c3e90252b4495972f1310e583c49c4.jpg)

<details>
<summary>heatmap</summary>

| Number of averaged signals N | I/2σ² [dB] | Ratio SNR_CPX / SNR_MAG [dB] |
| ---------------------------- | ---------- | --------------------------- |
| 1                            | 0          | +9                          |
| 20                           | -5         | +9                          |
| 40                           | -8         | +9                          |
| 60                           | -10        | +9                          |
| 80                           | -11        | +9                          |
| 100                          | -12        | +9                          |
</details>

Fig. 5. Relative SNR performance for magnitude and complex averaging (without noise bias correction). The heat map displays the ratio $SNR_{CPX}/SNR_{MAG}$ in decibels for relative input signals $I/2\sigma^{2}$ ranging from -20 dB to +40 dB and up to a number of averaged signals N = 100. For small input signals below the noise level, magnitude averaging yields a better SNR improvement (red range), while complex averaging performs better for greater N and stronger input signals (blue range). The borderline SNR where $SNR_{MAG} = SNR_{CPX}$ separates these two domains (white plot).

# 2.3.3. SNR in absence of a signal (noise floor only)

An interesting scenario is posed by the calculation of the SNR when the signal intensity I is zero. Then, using the relation $\overline{I_{noise}} = \sigma_{I_{noise}} = 2\sigma^{2}$ from Eqs. (11) and (12), the SNRs for a single signal, after magnitude averaging and after complex averaging, respectively, become:

$$
\mathrm{SNR} _ {1} (I = 0) = \frac {\overline {{I _ {\text { noise }}}}}{\sigma_ {I _ {\text { noise }}}} = 1, \tag {28}
$$

$$
\mathrm{SNR} _ {M A G} (I = 0) = \frac {\overline {{I _ {n o i s e}}}}{\frac {1}{\sqrt {N}} \sigma_ {I _ {n o i s e}}} = \sqrt {N}, \tag {29}
$$

$$
\mathrm{SNR} _ {C P X} (I = 0) = \frac {\frac {1}{N} \overline {{I _ {n o i s e}}}}{\frac {1}{N} \sigma_ {I _ {n o i s e}}} = 1. \tag {30}
$$

An obvious disunity of the SNRs calculated for pure noise signal can be observed. This calls for a revised definition of the SNR, this time accounting for the signal bias $\overline{I_{noise}}$ imposed by the noise floor.

# 2.3.4. SNR analysis with noise bias correction

In the limit of a very weak OCT signal, the small meaningful signal contribution sits on top of a comparatively large noise floor. This noise bias is evident as the term related to $\overline{I_{noise}}$ in Eqs. (24–26) and is also visualized in Fig. 2. Thus, it makes sense to actually consider the impact of this noise bias in the SNR analysis – in particular for weak signals. In this section, we introduce

and evaluate an SNR analysis with noise bias correction. A similar approach has for instance also been used by Makita et al. $[39]$ and recently been modified by our group $[40]$ to correct for noise contributions in PS-OCT images and improve the computation of the degree of polarization uniformity for weak signals. To some extent, this SNR with noise bias correction resembles the SNR definitions in Refs. $[29,30]$ and has formal similarities with previous definitions of the contrast-to-noise ratios in Refs. $[21,27]$ .

The noise bias of the average OCT signals can be removed by subtracting the average noise intensity $(\overline{I_{noise}}, \overline{\langle I_{noise} \rangle_{MAG}}$ , and $\overline{\langle I_{noise} \rangle_{CPX}}$ in Eqs. (11), (16) and (20), respectively) from the average signal intensity $(\overline{I}, \overline{\langle I \rangle_{MAG}}$ , and $\overline{\langle I \rangle_{CPX}}$ in Eqs. (13), (18) and (22), respectively). The noise bias corrected average signal intensities then read

$$
\bar {I} ^ {\prime} = \bar {I} - \overline {{I _ {n o i s e}}} = I + 2 \sigma^ {2} - 2 \sigma^ {2} = I, \tag {31}
$$

$$
\overline {{\langle I \rangle_ {M A G}}} ^ {\prime} = \overline {{\langle I \rangle_ {M A G}}} - \overline {{\langle I _ {n o i s e} \rangle_ {M A G}}} = I + 2 \sigma^ {2} - 2 \sigma^ {2} = I, \tag {32}
$$

$$
\overline {{\langle I \rangle_ {C P X}}} ^ {\prime} = \overline {{\langle I \rangle_ {C P X}}} - \overline {{\langle I _ {\text { noise }} \rangle_ {C P X}}} = I + \frac {1}{N} 2 \sigma^ {2} - \frac {1}{N} 2 \sigma^ {2} = I. \tag {33}
$$

Note that the three corrected average signal intensities now match the pure signal intensity I. Further, using the noise variances in Eqs. (12), (17) and (21), the respective noise bias corrected SNRs can be calculated as

$$
S N R _ {1} ^ {\prime} = \frac {\bar {I} ^ {\prime}}{\sigma_ {I _ {\text { noise }}}} = \frac {I}{2 \sigma^ {2}}, \tag {34}
$$

$$
S N R _ {M A G} ^ {\prime} = \frac {\overline {{\langle I \rangle_ {M A G}}} ^ {\prime}}{\left\langle \sigma_ {I _ {\text {noise}}} \right\rangle_ {M A G}} = \sqrt {N} \frac {I}{2 \sigma^ {2}} = \sqrt {N} \cdot S N R _ {1} ^ {\prime}, \tag {35}
$$

$$
S N R _ {C P X} ^ {\prime} = \frac {\overline {{\langle I \rangle_ {C P X}}} ^ {\prime}}{\left\langle \sigma_ {I _ {\text {noise}}} \right\rangle_ {C P X}} = N \frac {I}{2 \sigma^ {2}} = N \cdot S N R _ {1} ^ {\prime}. \tag {36}
$$

$SNR_{1}^{\prime}$ equates the quantity $SNR - 1 = I/2\sigma^{2}$ already known from the analyses in the previous sections. Even more strikingly, the noise bias corrected SNRs after magnitude and complex averaging now exhibit a simple proportionality of $\sqrt{N}$ and N with $SNR_{1}^{\prime}$ (Fig. 6), which also means that $SNR^{\prime}$ will be zero if there is no signal for Eqs. (34–36). Using the noise bias correction, complex averaging thus always yields a $\sqrt{N}$ -times better SNR than magnitude averaging, even for weak input signals and small N.

![](images/d1b3e52dc0f7b68bf3050b72f45452505c8ec821711614276f2ec3dfa9224c25.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG  | CPX  |
| ---------------------------- | ---- | ---- |
| 0                            | 0    | 0    |
| 50                           | ~5   | 60   |
| 100                          | ~10  | 100  |
</details>

![](images/3e4162e5dfd30ddf5af0956edf076679b8c89a47b42f9eff38da8cae3ab7befe.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG  | CPX  |
| ---------------------------- | ---- | ---- |
| 1                            | 1    | 1    |
| 10                           | ~3   | ~15  |
| 100                          | ~10  | 100  |
</details>

Fig. 6. Theoretical improvement of the signal-to-noise ratio SNR' after noise bias correction plotted on (a) linear scales and (b) log scales. A $\sqrt{N}$ - and $N$ -fold improvement of the SNR' of a single signal can be observed for magnitude and complex averaging, respectively. Note that, unlike for the noise-afflicted SNR calculations in Figs. 3 through 5, neither of the averaging approaches depend on the input signal strength.

# 3. Experimental validation

# 3.1. Numerical simulation

OCT signals and noise were simulated according to the probability density functions described in Eqs. (3) and (6), respectively. For the noise contribution, binormally distributed complex signals with similar standard deviations $\sigma$ along both the real axis and the imaginary axis were generated. For the signal contribution, the binormal distribution was generated around a real-valued signal $A$ such that the phasor distribution was shifted from the origin of the complex plane to $(A,0)$ (see also Fig. 1(e)). A total of $M = 100$ averaged signals were computed for every simulation run. For complex averaging, in every run $N = 1$ to $N = 100$ complex-valued phasors were averaged. For magnitude averaging, the absolute values were computed for every single noise and signal phasor prior to averaging. Finally, the respective SNRs and SNR's were calculated from the average signal intensity and the variance of the noise signals for every $N$ .

Results of the simulations performed in MATLAB (R2014a, MathWorks) are shown in Fig. 7. To showcase the effect of averaging for strong and weak signals, SNR simulations for two distinct relative signal levels of $I/2\sigma^{2}=10$ and $I/2\sigma^{2}=0.1$ are presented in Fig. 7(a) and (b), respectively. Unlike magnitude averaging, complex averaging markedly decreased the signal intensity but at the same time also reduced noise much more by averaging. For the standard

Signal-to-noise ratio   
![](images/8192adca2723898d562983ab3a965a0a7036b00de48c93166719c79547476339.jpg)  
Fig. 7. Simulation of the effect of averaging N signals with relative strength $I/2\sigma^{2}=10$ in (a) and $I/2\sigma^{2}=0.1$ in (b). Shown are the averaged signal-to-noise ratios calculated from N simulated phasors ( $\bullet$ ) alongside the corresponding theoretical plots (−). The SNRs without and with noise bias correction are plotted in the left and right panels, respectively. Without noise bias correction, the $SNR_{CPX}$ shows a better performance for the strong signal in (a), whereas $SNR_{MAG}$ dominates for N=1 to 100 both for theoretical calculation and simulation. With noise bias correction (rightmost column), complex averaging similarly provides an N-fold improvement of the respective $SNR^{\prime}=I/2\sigma^{2}$ whereas an $\sqrt{N}$ -fold improvement can be observed for magnitude averaging.

# Biomedical Optics EXPRESS

deviation of the noise fluctuations, a decrease by $1 / \sqrt{N}$ and $1 / N$ was observed for magnitude and complex averaging, respectively. The resulting SNRs are shown in the four panels in Fig. 7. Here, for averaging up to 100-fold, $\mathrm{SNR}_{CPX}$ reveals a better SNR improvement for the strong input signal (Fig. 7(a)) when the noise bias is not corrected for, while magnitude averaging appears more effective for the weaker input signal in Fig. 7(b). After noise bias correction, however, no more dependence upon the input signal level can be observed, and complex averaging provides an $N$ -fold increase in $\mathrm{SNR}'$ whereas the $\mathrm{SNR}'$ is only enhanced by a factor $\sqrt{N}$ after magnitude averaging.

# 3.2. Averaging experimental OCT data

For demonstrating the effect of the different averaging approaches on OCT images, we used a spectral domain (SD) OCT system in our lab $[38]$ . This polarization sensitive SD-OCT system was used for imaging a stationary eye phantom. This system is based on a multiplexed superluminescent diode ( $\lambda = 840$ nm, $\Delta\lambda = 100$ nm) as a light source, a free-space Michelson-type interferometer, and a polarization-sensitive detection unit including two spectrometers. For the investigations presented here, only the co-polarized detection channel was analyzed; thus the PS-OCT system was reduced to what would be considered a standard SD-OCT with high-resolution imaging capabilities - 3.6 $\mu$ m axial resolution (assuming a refractive index of 1.35), 83 kHz A-scan rate - similar to state-of-the-art commercial SD-OCT scanners for retinal imaging.

The eye phantom, composed of several layers of transparent nail polish on a glass bead to produce a laminar reflectance pattern $[38]$ , was imaged using B-scan and M-scan protocols at different levels of attenuation (from 0 dB to -40 dB) to observe the effects of averaging in low-signal conditions. The B-scan protocol scanned a 1 mm lateral range with 100 repeats to generate cross-sectional images of the phantom (Fig. 8(a-d)). Representative depth profiles are shown in Fig. 8(e).

The M-scan protocol was used to acquire 1000 repeated A-lines from a fixed position within the phantom for a precise comparison of averaging methods. The M-scan protocol was chosen to minimize phase differences between consecutive A-lines and ensure that averaging represents a best-case real-world scenario. The 1000 A-lines were split into 10 temporally independent bins of 100 consecutive A-lines each. Complex and magnitude averaging for N up to 100 was performed on each of the 10 bins and the resulting signal curves were averaged together before calculating SNR and SNR' in order to yield more stable SNR curves, particularly for lower numbers of averages. Here, the noise variances were computed from the noise pixel data of 100 consecutive A-lines, similar to the simulation above. Three characteristic SNR and SNR' curves from the magnitude-averaging dominant, borderline, and complex-averaging dominant domains are shown in Fig. 8(f) and (g). Theoretical SNR profiles based on $SNR_{1}$ as described in Eqs. (25) and (26) demonstrate good agreement with the experimental data in Fig. 8(f). These results show that the theoretical magnitude-averaging dominant domain is reproducible with real world OCT data when the noise floor induced bias is not accounted for. Similarly, the SNR' plots from the experimental data with noise bias correction follow the expected slopes of $\sqrt{N}$ and N (Fig. 8(g)).

![](images/9de7efae24b2256eeb3e04a94f500f716c0feaa63d8be11452d884578c240aaa.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing a textured surface with a green dashed line and 100 μm scale bar (no text or symbols beyond label)
</details>

![](images/2dde6f6761ba7b675135136b179c01c316269f7c81983ec68971d87cf74fe608.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing a textured surface with a 100 μm scale bar, no visible text or symbols.
</details>

![](images/4945508377d0a312f8a58f774e19fcb6154340562a53002a0cef9867716fe34d.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing a vertical red dashed line at 100 μm scale, with no visible text or symbols beyond the scale bar.
</details>

![](images/2c1d17aab167106c642bf3bbaf106b887675599304b741fd73df60a3e5ab8db2.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing layered structure with scale bar (100 μm) and labeled regions (a, b), no readable text or symbols beyond labels.
</details>

Relative Intensity [dB]

![](images/f496c6f2d9284d9d4b03988d8a9705834574f827a80aeee08636a4183d27e5d5.jpg)

<details>
<summary>line</summary>

| Depth [µm] | single | MAG  | CPX  | unattenuated |
| ---------- | ------ | ---- | ---- | ------------ |
| 0          | -10    | -20  | -30  | -60          |
| 50         | -10    | -20  | -30  | -60          |
| 100        | -10    | -20  | -30  | -60          |
| 150        | -10    | -20  | -30  | -60          |
| 200        | -10    | -20  | -30  | -60          |
| 250        | -10    | -20  | -30  | -60          |
| 300        | -10    | -20  | -30  | -60          |
| 350        | -10    | -20  | -30  | -60          |
| 400        | -10    | -20  | -30  | -60          |
| 450        | -10    | -20  | -30  | -60          |
</details>

![](images/e5d1f687d505e692d2ed3258e9d36546625ce8acef5bc8b8420b3d844871b3e5.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG theo | CPX theo | MAG exp | CPX exp |
| ---------------------------- | -------- | -------- | ------- | ------- |
| 0                            | 0.0      | 0.0      | 0.0     | 0.0     |
| 50                           | 6.0      | 2.0      | 4.0     | 1.0     |
| 100                          | 10.0     | 3.0      | 8.0     | 2.0     |
</details>

![](images/74ac4e10482a2a5a555913dc65138eeb45c15bb098035fee9d67a719319992a5.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG theo | CPX theo | MAG exp | CPX exp |
| ---------------------------- | -------- | -------- | ------- | ------- |
| 0                            | 0        | 0        | 0       | 0       |
| 50                           | 8        | 9        | 7       | 8       |
| 100                          | 14       | 15       | 13      | 14      |
</details>

![](images/7a7ef45bec595bd2ef07da6e40da05e9e2e3fd5c4ec872d4926cf0273ebc9a47.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG theo | CPX theo | MAG exp | CPX exp |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 50 | 25 | 150 | 10 | 15 |
| 100 | 50 | 300 | 20 | 30 |
</details>

![](images/fb1c8ccfa3242ce3fac1f641969e150aa24180187a9a7fe5a1b135d418ebceab.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG theo | CPX theo | MAG exp | CPX exp |
| ---------------------------- | -------- | -------- | ------- | ------- |
| 0                            | 0.0      | 0.0      | 0.0     | 0.0     |
| 50                           | 0.5      | 1.5      | 0.5     | 1.0     |
| 100                          | 1.0      | 2.5      | 1.0     | 2.0     |
</details>

![](images/1709ed69447bf3d29f07a82de1630436425874674ec82c69ca9472d0e1fc4271.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG theo | CPX theo | MAG exp | CPX exp |
| ---------------------------- | -------- | -------- | ------- | ------- |
| 0                            | 0        | 0        | 0       | 0       |
| 50                           | ~1       | ~8       | ~1      | ~8      |
| 100                          | ~2       | ~14      | ~2      | ~14     |
</details>

![](images/87ef167c425505145aad70613ef901ebce67eef250f9fcc62e4ca9757396ab54.jpg)

<details>
<summary>line</summary>

| Number of averaged signals N | MAG theo | CPX theo | MAG exp | CPX exp |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 50 | ~10 | ~200 | ~10 | ~200 |
| 100 | ~30 | ~300 | ~30 | ~300 |
</details>

Fig. 8. Experimental verification of SNR improvement by the different averaging approaches in a layered retina phantom. (a) Single B-scan image of the phantom (no attenuation). (b) Single B-scan image after attenuating the sample beam by 30 dB. (c) B-scan image after averaging the magnitudes of 100 repeated frames (with 30 dB attenuation). (d) B-scan image after averaging the phasors of 100 repeated frames (with 30 dB attenuation). Note that all B-scan images in (a-d) are displayed with identical dynamic ranges of 40 dB where 0 dB refers to the maximum signal intensity in the frame. (e) Depth profiles of a single A-scan before and after attenuation, 100 magnitude averaged, and 100 complex averaged A-scans at the locations indicated by the dotted lines in (a-d). Due to a beam offset caused by the ND filter, the scattering profile of the unattenuated case has a slightly different structure. Dynamic range as in (a-d). (f) SNR improvement without noise bias correction for three pixels with weak (left), borderline (middle) and strong signal strength $I / 2\sigma^2$ (right), respectively. Pixel locations are indicated by orange boxes numbered with 1-3 in panel (d). SNR curves are shown for magnitude and complex averaging of 1-100 repeated M-scan signals for the experimental data (●) alongside the corresponding theoretical plots (−). (g) SNR' improvement after noise bias correction for the data sets shown in panel (f). Note that the experimental data (●) slightly fluctuates around the theoretical profiles (−). SNR' data fluctuating below SNR' = 0 is not shown. The axes are scaled as in the respective plots in (f) in order to enable a direct comparison between the two SNR analyses.

# 4. Discussion and conclusion

Signal averaging is one of the most commonly used procedures for OCT image processing. Here, we investigated the SNR performance of magnitude and complex averaging in theory and backed our findings with experimental results from simulations and actual OCT image data. We also studied the effect of the background noise bias on the measured SNR and calculated a noise bias corrected SNR. In the following paragraphs, the main observations are summarized and discussed in order to provide a better understanding of the strengths and weaknesses of the two approaches.

# 4.1. Impact of averaging on noise and signals

Complex averaging reduces the noise variance by $1 / N$ , and thus provides a much stronger noise reduction than magnitude averaging, which only reduces the noise by $1 / \sqrt{N}$ . At the same time, complex averaging also reduces the signal - in contrast to magnitude averaging, which maintains the signal level and only reduces noise. The main cause of the average signal decrease by complex averaging is the reduction of the average noise level. Both signal and noise characteristics have to be considered to understand the impact of averaging on OCT data.

# 4.2. Leaving or removing the noise floor bias for the SNR calculation

The signal-to-noise ratio is a commonly used measure to assess the image quality and to describe the system performance of OCT machines. Usually, the ratio of the average intensity of a signal peak and the standard deviation of the noise intensity is used to calculate the SNR $[24,28]$ . While it is rather clear how to estimate the noise variance (e.g., by considering the noise intensity in a signal-free image area), the definition of the average intensity is not that obvious. As visualized in Fig. 2, taking the entire measured signal intensity from the zero line to the (average) peak intensity $\overline{I}$ includes two portions, namely the actual signal intensity term I and the background noise level $\overline{I_{noise}}$ .

For relatively strong signals, the measured signal is dominated by the intensity term I and $\overline{I} \approx I \gg \overline{I_{noise}}$ . In contrast, when the actual signal contribution to the measured signal is small (i.e. $I \ll \overline{I}$ ), the noise floor bias $\overline{I_{noise}}$ prevails. We hence investigated the signal-to-noise performance of magnitude and complex averaging for (a) leaving the noise floor bias as part of the measured signal and (b) correcting for the noise bias. As discussed in detail in the following sections, similar results were observed for both SNR analyses when strong signals were investigated. However, for rather weak signals, the impact of $\overline{I_{noise}}$ manifested in the observation of dissimilar SNR characteristics.

# 4.3. Noise-afflicted signal-to-noise ratios after averaging strong and weak signals

For sufficiently large signals $\bar{I}=I+\overline{I_{noise}}$ , the $N\cdot SNR_{1}$ term in Eq. (26) dominates such that complex averaging converges on an N-fold SNR improvement, while magnitude averaging only enhances the $SNR\sqrt{N}$ -fold. In contrast, for recovering weak signals comparable to or smaller than the noise, magnitude averaging appeared to outperform complex averaging – in particular for small N. When the input $SNR_{1}$ was just slightly larger than unity (i.e., $I/2\sigma^{2}$ just slightly greater than zero), the right-hand side of Eq. (26) was essentially neutralized such that the SNR improvement was far less than the $\sqrt{N}$ -fold enhancement yielded by magnitude averaging, especially in the limit of small N. Hence, magnitude averaging could be considered more effective for boosting small signals as they may be found in the outer nuclear layer in the retina when the noise floor induced signal bias is not accounted for. However, by taking a look at the SNR in absence of an actual signal (I=0), odd SNR characteristics were observed (see section 2.3.3) which underscored the importance of a noise bias correction.

# 4.4. SNR calculations with noise bias correction

When SNRs are calculated in the classical way described in section 2.3.1, the average noise floor level $\overline{I_{noise}}$ contributes to the intensity measured as "signal". This noise bias impacts the measured SNR, in particular for weak signals. Akin to noise offset removal in PS-OCT processing [39,40], we performed SNR calculations with noise bias correction in sections 2.3.4 and 3. By using this modified approach, a superior SNR performance was always observed for complex averaging, regardless of the input signal strength. Compared to the noise bias corrected SNR of a single signal prior to averaging, magnitude and complex averaging improved the SNR (after noise bias correction) by a factor of $\sqrt{N}$ and N, respectively. The theoretically predicted performance agreed well with the performance observed in simulated and experimental OCT data (Figs. 7 and 8). This finding suggests that noise bias correction definitely is an important processing step in SNR analyses – especially when it comes to averaging weak signals.

# 4.5. Implications for practical OCT image averaging

State-of-the-art swept source and spectral domain OCT devices deliver complex-valued signals (see Eq. (1)) right out of the box. Hence both magnitude and complex averaging can be easily implemented using the image data. For effective complex averaging, it is imperative that the phases of the signals are accurately aligned before averaging as described in more detail in Appendix B. While this means additional computational steps, it may be well worth the effort when the SNR is to be increased in images of scattering structures such as the retina where OCT has been frequently applied. Retinal OCT images may easily span a dynamic range of 40 dB with hyperscattering structures such as the nerve fiber layer and the pigment epithelium that can serve as landmarks for phasor alignment [18]. Other retinal tissues such as the ganglion cell layer and the outer nuclear layer [41], and also structures in the vitreous, which are usually much less reflective [42], may then be visualized by averaging multiple OCT images together. Complex averaging has also been found promising to detect signals in settings with multiple scattering [20].

Magnitude averaging on the other hand can be implemented at less computational expense and may be the averaging approach of choice for images containing mostly weak signals of interest, which would render phase alignment for complex averaging difficult or even impossible. Also in scenarios where images include a lot of varying motion, e.g. when visualizing weakly scattering structures such as flow in lymphatic vessels $[43]$ , albeit theoretically less effective by $\sqrt{N}$ in terms of SNR improvement, magnitude averaging may likely trump complex averaging.

# 4.6. Future perspectives

Finally, we would like to point out that, while the analysis presented here particularly focused on the SNR improvement for different averaging approaches, it may be interesting to also investigate other image metrics such as their contrast-to-noise characteristics and/or their efficiency in terms of speckle reduction. For this purpose and to explore scenarios with tissue-like scattering, recently described OCT signal models could be particularly interesting candidates $[28,44–46]$ . Additionally, OCT images are often displayed on a logarithmic scale in order to compress signals covering a large dynamic range. The logarithm pulls up weak signals but also the noise floor $[21]$ , such that the SNR performance for averaged logarithmic amplitude data will deviate from that for uncompressed OCT data discussed here. An in-depth analysis similar to the one performed for linear data in section 2.2 may provide more insight in the advantages and disadvantages of magnitude- and phasor-based averaging approaches for enhancing OCT image quality.

# Appendix

Appendix A – Glossary

The table below provides an overview of the variables and symbols used in this article.

Table 1. Overview of variables and abbreviations 

<table><tr><td>Symbol</td><td>Explanation</td><td>Equation</td></tr><tr><td> $A(\mathbf{x},t)$ </td><td>Amplitude of phasor representing the OCT signal</td><td>(1)</td></tr><tr><td> $\overline{A}$ </td><td>Mean signal amplitude</td><td>(7)</td></tr><tr><td> $A_{noise}$ </td><td>Amplitude of complex phasor representing the OCT noise signal</td><td>(3)</td></tr><tr><td> $\overline{A_{noise}}$ </td><td>Mean noise amplitude</td><td>(4)</td></tr><tr><td> $\Delta\phi$ </td><td>Phase difference</td><td>(39,42)</td></tr><tr><td> $I$ </td><td>Signal intensity,  $I = A^{2}$ </td><td></td></tr><tr><td> $\overline{I}$ </td><td>Mean signal intensity</td><td>(13)</td></tr><tr><td> $\overline{I}_{x}'$ </td><td> $\overline{I}_{x}$  with noise bias correction, (x = void, MAG, CPX)</td><td>(31–33)</td></tr><tr><td> $I_{noise}$ </td><td>Noise intensity,  $I_{noise} = A_{noise}^{2}$ </td><td></td></tr><tr><td> $\overline{I_{noise}}$ </td><td>Mean noise intensity</td><td>(11)</td></tr><tr><td> $i_{noise}$ </td><td>Imaginary part of complex phasor representing the OCT noise signal</td><td>(2)</td></tr><tr><td> $N$ </td><td>Number of averaged signals</td><td></td></tr><tr><td> $p_{A}(A_{noise})$ </td><td>PDF describing amplitudes of random phasors (Rayleigh distribution)</td><td>(3)</td></tr><tr><td> $p_{\Delta\phi}(\Delta\phi,A)$ </td><td>PDF describing distribution of phase differences</td><td>(43)</td></tr><tr><td> $p_{I}(I_{noise})$ </td><td>PDF describing intensities of random phasors</td><td>(9)</td></tr><tr><td> $p_{I}(I,I_{noise})$ </td><td>PDF describing intensities of signals in presence of noise</td><td>(10)</td></tr><tr><td> $p_{A}(A,A_{noise})$ </td><td>PDF describing amplitudes of signal phasors in presence of noise (Rice distribution)</td><td>(6)</td></tr><tr><td> $p_{ri}(r_{noise},i_{noise})$ </td><td>Binormal PDF describing random phasors</td><td>(2)</td></tr><tr><td> $P(N)$ </td><td>Penalty factor after averaging  $N$  signals</td><td>(37)</td></tr><tr><td>PDF</td><td>Probability density function</td><td>(2,3,6,9,10)</td></tr><tr><td> $\phi(\mathbf{x},t)$ </td><td>Phase of phasor representing the OCT signal</td><td>(1)</td></tr><tr><td> $r_{noise}$ </td><td>Real part of complex phasor representing the OCT noise signal</td><td>(2)</td></tr><tr><td> $S_{OCT}(\mathbf{x},t)$ </td><td>Complex-valued OCT signal</td><td>(1)</td></tr><tr><td> $S_{OCT}^{(moco)}$ </td><td>OCT signal after motion correction (moco)</td><td>(40,41,45)</td></tr><tr><td> $\sigma^{2}$ </td><td>Variance of real/imaginary part of  $p_{ri}$ </td><td>(1)</td></tr><tr><td> $\sigma_{A}^{2}$ </td><td>Variance of signal amplitudes for  $p_{A}(A,A_{noise})$ </td><td>(8)</td></tr><tr><td> $\sigma_{I}^{2}$ </td><td>Variance of signal intensity for  $p_{I}(I,I_{noise})$ </td><td>(14)</td></tr><tr><td> $\sigma_{I_{noise}}^{2}$ </td><td>Variance of noise intensity for  $p_{I}(I_{noise})$ </td><td>(12)</td></tr><tr><td> $\sigma_{noise}^{2}$ </td><td>Variance of noise amplitudes for  $p_{A}(A_{noise})$ </td><td>(5)</td></tr><tr><td> $SNR_{1}$ </td><td>SNR of non-averaged signal</td><td>(24)</td></tr><tr><td> $SNR_{1,borderline}$ </td><td> $SNR_{1}$  of non-averaged signal for  $SNRMAG = SNRCPX$ </td><td>(27)</td></tr><tr><td> $SNRMAG$ </td><td>SNR after magnitude averaging</td><td>(25)</td></tr><tr><td> $SNRCPX$ </td><td>SNR after complex phasor averaging</td><td>(26)</td></tr><tr><td> $SNR_{x}'$ </td><td> $SNR_{x}$  with noise bias correction, (x = 1, MAG, CPX)</td><td>(34–36)</td></tr><tr><td> $\langle\cdot\rangle_{MAG}$ </td><td>Quantity · after magnitude averaging</td><td>(16–19)</td></tr><tr><td> $\langle\cdot\rangle_{CPX}$ </td><td>Quantity · after complex averaging</td><td>(20–23)</td></tr></table>

Appendix B – Complex signal averaging in case of axial motion

Effect of axial motion on complex averaging Keeping the signal phase $\phi$ constant is essential for complex signal averaging. If $\mathrm{d}^{2}\phi(\mathbf{x},t)/\mathrm{d}\mathbf{x}\mathrm{d}t\neq0$ , the complex sum of the signal vectors will have a length smaller than the sum of their absolute values. In other words, if the phases of the complex signals are not matched prior to averaging, the length of the average signal vector will be reduced. The condition $\Delta\phi=0$ is fulfilled for simultaneously acquired signals at the same z-position within the same speckle. When averaging signals among different speckles acquired at the same time point, first the phase offset between different speckles has to be eliminated [48]. Similarly, when signals acquired at the same location x but at different time points t are averaged (e.g., signals from repeatedly acquired OCT images), the influence of axial motion has to be removed prior to averaging [18].

Axial motion introduces a proportional phase shift in the complex signals (Fig. 9(a)). This phase shift has been extensively exploited for velocity measurements in Doppler OCT, for displacement measurements in OCT elastography and, recently most prominently, for some OCT angiography approaches [32-36]. The phase shift of an OCT signal is proportional to $2k\Delta z$ where $k = 2\pi/\lambda$ is the central wavenumber and $\Delta z$ is the axial displacement between successive measurements. When $N$ complex signals with displacements $\Delta z_{j}$ relative to the first signal are averaged in a complex fashion, the resulting relative signal reduction corresponds to the penalty factor $P(N)$ :

$$
P (N) = \frac {1}{N} \left| \sum_ {j = 1} ^ {N} e ^ {i 2 \pi W _ {j}} \right| \tag {37}
$$

where $W_{j} = 2\Delta z_{j}/\lambda$ denotes the displacement of the j-th signal in terms of wavelengths. In case of constant axial motion, the displacement will increase by $(j - 1)\Delta z$ for the j-th signal compared to the first signal. The penalty factor $P(N)$ when averaging N signals with relative separation $W_{j} = W$ is

$$
P (N, W) = \frac {1}{N} \left| \sum_ {j = 1} ^ {N} e ^ {i 2 \pi W j} \right| = \frac {1}{N} \left| \frac {\sin [ \pi W (N + 1) ]}{\sin [ \pi W ]} e ^ {i \pi N W} - 1 \right|. \tag {38}
$$

Examples for the relative signal intensity decrease due to $P(N, W)$ are shown in Fig. 9(b). For large W, the upper limit of the rectified sinc-like decrease is given by 1/N.

Compensation of axial motion In order to remove the detrimental effect of axial motion and thus to fully exploit the SNR advantage of complex averaging, the phases $\phi_{j}$ of the signals to be averaged have to be aligned. Ju et al. proposed two approaches: one compensating phase offsets A-line-wise and another one compensating phase offsets in a pixel-wise fashion [18]. Assuming a constant "bulk" displacement of the simultaneously acquired signals within one A-scan, the bulk phase shift $\Delta\phi_{bulk}$ between the m-th A-line in a reference frame (ref) and the corresponding m-th A-scan in other frames to be averaged can be calculated by computing the angle of the weighted complex signal average along every A-scan:

$$
\begin{array}{l} \Delta \phi_ {b u l k} ^ {(m)} = \arg \left[ \sum_ {z = z _ {0}} ^ {z _ {\max}} \left(A _ {m} ^ {j} (z) e ^ {i \phi_ {m} ^ {j} (z)}\right) \left(A _ {m} ^ {\text {ref}} (z) e ^ {i \phi_ {m} ^ {\text {ref}} (z)}\right) ^ {*} \right] = \tag {39} \\ = \arg \left[ \sum_ {z = z _ {0}} ^ {z _ {\max}} A _ {m} ^ {j} (z) A _ {m} ^ {r e f} (z) e ^ {i [ \phi_ {m} ^ {j} (z) - \phi_ {m} ^ {r e f} (z) ]} \right] \\ \end{array}
$$

where m, j, and $*$ denote the A-scan index, the B-scan index and the conjugate complex, respectively, while $z_{0}$ and $z_{max}$ are the axial depth limits. By subtracting the bulk phase shift

![](images/b573086fcb50a4402475b1eeee1ec6e290484d77882abc598725da86b94749e2.jpg)

![](images/43996dfe91bdd6e5fb89620fbc5cc246b0e67bf0468989aa334e49745de2e209.jpg)

<details>
<summary>line</summary>

| Number of averaged frames N | Relative signal intensity [dB] (W = 0.005) | Relative signal intensity [dB] (W = 0.05) | Relative signal intensity [dB] (1/N) |
| --------------------------- | ------------------------------------------ | ----------------------------------------- | ----------------------------------- |
| 0                           | 0                                          | 0                                         | 0                                   |
| 20                          | -10                                        | -15                                       | -15                                 |
| 40                          | -15                                        | -20                                       | -20                                 |
| 60                          | -20                                        | -25                                       | -25                                 |
| 80                          | -25                                        | -30                                       | -30                                 |
| 100                         | -30                                        | -35                                       | -35                                 |
</details>

Fig. 9. Phasor rotation by axial motion and signal penalty after complex averaging of axially displaced signals. (a) Example of a signal phasor trajectory of 1000 repeated measurements of a weak reflector (an attenuated glass surface). Over 1/70 second, the phasor (blue dots) is markedly moving around the origin. Corresponding noise signals are shown as red dots. The main source of displacement in this measurement was air flow from a nearby air condition outlet. (b) Relative signal intensity decrease caused by complex averaging of N signals with relative displacements W of 0.5 (solid line), 0.05 (dashed line), and 0.005 (dotted line). For reference, 1/N is plotted as well. While relative displacements of $\lambda/200$ (W = 0.005) between consecutive signals almost do not affect the intensity of the averaged signal, greater displacements such as $\lambda/20$ and $\lambda/2$ cause a periodic, significant signal reduction. In unlucky cases, $P(N, W)$ reaches zero such that the averaged signal is completely annihilated.

vector $\Delta\phi_{bulk}^{(m)}$ from the phases of every transverse line of the OCT image, an axial motion corrected (moco) image is generated:

$$
S _ {O C T} ^ {(m o c o)} (x, z) = A _ {m} ^ {j} (x, z) \exp \left[ i (\phi_ {m} ^ {j} (x, z) - \Delta \phi_ {b u l k} ^ {(m)}) \right]. \tag {40}
$$

The assumption that displacements are constant along the depth profile does however not hold for all samples. In particular live biological samples are subject to dynamic deformations, which may for instance be caused by pulsating blood vessels. Hence, in another complex image averaging approach, all complex signals of one reference B-scan serve as a 2D baseline image $S_{OCT}^{(ref)}(x,z)$ [18]. The phasors in all pixel locations $(x,z)$ of the subsequent frames are then aligned with respect to those in the reference frame by multiplying them to the conjugate complex of $S_{OCT}^{(ref)}(x,z)$ in an amplitude weighted manner,

$$
S _ {O C T} ^ {(j, m o c o)} (x, z) = \sqrt {A _ {j} (x , z) A _ {r e f} (x , z)} \exp \left[ i \left(\phi_ {j} (x, z) - \phi_ {r e f} (x, z)\right) \right]. \tag {41}
$$

Since Eq. (41) does not only align phasors at signal locations but also all noise phasors, complex averaging of $S_{OCT}^{(j,moco)}(x,z)$ essentially corresponds to magnitude averaging. This detriment can be overcome by locally averaging the phase difference values $\Delta\phi_{j}(x,z)=\phi_{j}(x,z)-\phi_{ref}(x,z)$ prior to phase balancing and complex averaging. For this purpose, the phase differences $\Delta\phi_{j}(x,z)$ are first calculated between the j-th frame and the reference frame. In order to minimize the number of phase wraps in presence of strong axial motion differences, we choose the central frame as a reference. Second, the j-th phase difference B-scan is averaged using the respective phasors within a small kernel which may, for instance, be rectangular:

$$
\overline {{{{\Delta \phi_ {j}}}}} (x, z) = \arg \left[ \sum_ {x = x - \Delta x / 2} ^ {x + \Delta x / 2} \sum_ {z = z - \Delta z / 2} ^ {z + \Delta z / 2} A _ {j} (x, z) A _ {\text { ref }} (x, z) \exp [ i \Delta \phi_ {j} (x, z) ] \right]. \tag {42}
$$

# Biomedical Optics EXPRESS

Ideally, the kernel size $(\Delta x, \Delta z)$ is chosen large enough to include at least a few speckles but small enough to only cover a patch of similar axial motion. The PDF of the averaged phase difference $\overline{\Delta\phi_{j}}(x, z)$ centered at zero can be derived from Eq. (2) by performing a variable transform to polar coordinates and then integrating the PDF over the amplitude space [47], whereby $\sigma_{\Delta\phi} = \sqrt{2}\sigma$ is used for the standard deviation:

$$
p _ {\Delta \phi} (\Delta \phi , A) = \frac {1}{2 \pi} \exp \left[ - \frac {A ^ {2}}{4 \sigma_ {\Delta \phi} ^ {2}} \right] + \tag {43}
$$

$$
+ \frac {A}{\sqrt {4 \pi} \sigma_ {\Delta \phi}} \cos \Delta \phi \exp \left[ - \frac {A ^ {2} \sin^ {2} \Delta \phi}{4 \sigma_ {\Delta \phi} ^ {2}} \right] \operatorname{erf} \left[ \frac {A}{\sqrt {2} \sigma_ {\Delta \phi}} \cos \Delta \phi \right]
$$

where $\operatorname{erf}(u)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{u}\exp\left[-v^{2}/2\right]dv$ denotes the Gaussian error function. With increasing $A/\sigma_{\Delta\phi}$ , the distribution $p_{\Delta\phi}$ becomes more narrow and converges towards a $\delta$ -function at $\Delta\phi=0$ [22,47]. For $\Delta\phi\neq0$ , the maximum of the distribution is located at the respective expectancy value for the phase difference. In absence of a signal A, the somewhat complicated PDF $p_{\Delta\phi}$ reduces drastically and becomes a uniform distribution on $(-π,\pi)$ ,

$$
p _ {\Delta \phi} (\Delta \phi , A = 0) = \frac {1}{2 \pi}. \tag {44}
$$

Hence, the averaging process in Eq. (42) provides an estimate of the localized phase shift $\Delta\phi$ with improved precision in case signal pixels are included within the averaging kernel $(\Delta x, \Delta z)$ but delivers a random $\overline{\Delta\phi}$ if the kernel only includes noise.

If now, in a third step, the locally averaged phase difference map (Eq. (42)) is used to correct the phasors of the j-th frame with respect to the reference frame,

$$
S _ {O C T} ^ {(j, \overline {{m o c o}})} (x, z) = \sqrt {A _ {j} (x , z) A _ {r e f} (x , z)} \exp \left[ i \left(\phi_ {j} (x, z) - \overline {{\Delta \phi_ {j}}} (x, z)\right) \right], \tag {45}
$$

the phasors will only be aligned for signals but will retain their random character for noise regions. By averaging N such axial motion balanced frames, the full potential of complex averaging can be tapped without suffering from signal reduction due to local displacement differences.

# Funding

Austrian Science Fund (P25823-B24, P26553-N20); H2020 European Research Council (ERC Starting Grant 640396 OPTIMALZ).

# Acknowledgements

The authors would like to acknowledge fruitful discussions with Antonia Lichtenegger, Danielle J. Harper, Pablo Eugui, Laurin Ginner, and Florian Beer. We would like to express our particular gratitude to Stanislava Fialova for her contributions to the imaging system and eye phantom. We also acknowledge the thoughtful feedback and inputs provided by the anonymous reviewers.

# Disclosures

The authors declare that there are no conflicts of interest related to this article.

# References

1. D. Huang, E. A. Swanson, C. P. Lin, J. S. Schuman, W. G. Stinson, W. Chang, M. R. Hee, T. Flotte, K. Gregory, C. A. Puliafito, and J. Fujimoto, "Optical coherence tomography," Science 254(5035), 1178–1181 (1991).   
2. W. Drexler, M. Liu, A. Kumar, T. Kamali, A. Unterhuber, and R. A. Leitgeb, “Optical coherence tomography today: speed, contrast, and multimodality,” J. Biomed. Opt. 19(7), 071412 (2014).

# Biomedical Optics EXPRESS

3. T. Klein and R. Huber, “High-speed OCT light sources and systems [Invited],” Biomed. Opt. Express 8(2), 828–859 (2017).   
4. R. Leitgeb, C. K. Hitzenberger, and A. F. Fercher, "Performance of fourier domain vs. time domain optical coherence tomography," Opt. Express 11(8), 889-894 (2003).   
5. M. A. Choma, M. V. Sarunic, C. Yang, and J. A. Izatt, “Sensitivity advantage of swept source and Fourier domain optical coherence tomography,” Opt. Express 11(18), 2183–2189 (2003).   
6. J. F. de Boer, B. Cense, B. H. Park, M. C. Pierce, G. J. Tearney, and B. E. Bouma, “Improved signal-to-noise ratio in spectral-domain compared with time-domain optical coherence tomography,” Opt. Lett. 28(21), 2067–2069 (2003).   
7. A. Sakamoto, M. Hangai, and N. Yoshimura, “Spectral-Domain Optical Coherence Tomography with Multiple B-Scan Averaging for Enhanced Imaging of Retinal Diseases,” Ophthalmology 115(6), 1071–1078.e7 (2008).   
8. P. Puvanathasan and K. Bizheva, "Interval type-II fuzzy anisotropic diffusion algorithm for speckle noise reduction in optical coherence tomography images," Opt. Express 17(2), 733-746 (2009).   
9. W. Wu, O. Tan, R. R. Pappuru, H. Duan, and D. Huang, “Assessment of frame-averaging algorithms in OCT image analysis,” Ophthalmic Surg. Lasers Imag. Retin. 44(2), 168–175 (2013).   
10. C.-L. Chen, H. Ishikawa, G. Wollstein, R. A. Bilonick, L. Kagemann, and J. S. Schuman, “Virtual Averaging Making Nonframe-Averaged Optical Coherence Tomography Images Comparable to Frame-Averaged Images,” Trans. Vis. Sci. Tech. 5(1), 1 (2016).   
11. D. Xu, N. Vaswani, Y. Huang, and J. U. Kang, “Modified compressive sensing optical coherence tomography with noise reduction,” Opt. Lett. 37(20), 4209–4211 (2012).   
12. M. Sugita, S. Zotter, M. Pircher, T. Makihiro, K. Saito, N. Tomatsu, M. Sato, P. Roberts, U. Schmidt-Erfurth, and C. K. Hitzenberger, "Motion artifact and speckle noise reduction in polarization sensitive optical coherence tomography by retinal tracking," Biomed. Opt. Express 5(1), 106–122 (2014).   
13. H. Zhang, Z. Li, X. Wang, and X. Zhang, “Speckle reduction in optical coherence tomography by two-step image registration,” J. Biomed. Opt. 20(3), 036013 (2015).   
14. A. C. Chan, K. Kurokawa, S. Makita, M. Miura, and Y. Yasuno, "Maximum a posteriori estimator for high-contrast image composition of optical coherence tomography," Opt. Lett. 41(2), 321–324 (2016).   
15. B. Tan, A. Wong, and K. Bizheva, “Enhancement of morphological and vascular features in OCT images using a modified Bayesian residual transform,” Biomed. Opt. Express 9(5), 2394–2406 (2018).   
16. P. H. Tomlins and R. K. Wang, “Digital phase stabilization to improve detection sensitivity for optical coherence tomography,” Meas. Sci. Technol. 18(11), 3365–3372 (2007).   
17. J. W. Jacobs and S. J. Matcher, "Digital phase stabilization for improving sensitivity and degree of polarization accuracy in polarization sensitive optical coherence tomography," Proc. SPIE 7889, 788938 (2011).   
18. M. J. Ju, Y.-J. Hong, S. Makita, Y. Lim, K. Kurokawa, L. Duan, M. Miura, S. Tang, and Y. Yasuno, “Advanced multi-contrast Jones matrix optical coherence tomography for Doppler and polarization sensitive imaging,” Opt. Express 21(16), 19412–19436 (2013).   
19. T. Pfeiffer, W. Wieser, T. Klein, M. Petermann, J. P. Kolb, M. Eibl, and R. Huber, “Flexible A-scan rate MHz OCT: computational downscaling by coherent averaging,” Proc. SPIE 9697, 96970S (2016).   
20. L. Thrane, S. Gu, B. J. Blackburn, K. V. Damodaran, A. M. Rollins, and M. W. Jenkins, “Complex decorrelation averaging in optical coherence tomography: a way to reduce the effect of multiple scattering and improve image contrast in a dynamic scattering medium,” Opt. Lett. 42(14), 2738–2741 (2017).   
21. M. Szkulmowski and M. Wojtkowski, “Averaging techniques for OCT imaging,” Opt. Express 21(8), 9757–9773 (2013).   
22. J. W. Goodman, Statistical Optics (John Wiley & Sons, 2000).   
23. P. Beckmann, “Statistical distribution of the amplitude and phase of a multiply scattered field,” J. Res. Natl. Bur. Stand. (U. S.) 66D(3), 231–240 (1962).   
24. A. Agrawal, T. J. Pfefer, P. D. Woolliams, P. H. Tomlins, and G. Nehmetallah, “Methods to assess sensitivity of optical coherence tomography systems,” Biomed. Opt. Express 8(2), 902–917 (2017).   
25. X. Zhu, Y. Liang, Y. Mao, Y. Jia, Y. Liu, and G. Mu, “Analyses and calculations of noise in optical coherence tomography systems,” Front. Optoelectron. China 1(3-4), 247–257 (2008).   
26. J. Izatt and M. Choma, "Theory of optical coherence tomography," in Optical Coherence Tomography - Technology and Applications, (2008), pp. 47-72.   
27. M. Szkulmowski, I. Gorczynska, D. Szlag, M. Sylwestrzak, A. Kowalczyk, and M. Wojtkowski, “Efficient reduction of speckle noise in Optical Coherence Tomography,” Opt. Express 20(2), 1337–1359 (2012).   
28. M. Sugita, A. Weatherbee, K. Bizheva, I. Popov, and A. Vitkin, “Analysis of scattering statistics and governing distribution functions in optical coherence tomography,” Biomed. Opt. Express 7(7), 2551–2564 (2016).   
29. D. M. Stein, H. Ishikawa, R. Hariprasad, G. Wollstein, R. J. Noecker, J. G. Fujimoto, and J. S. Schuman, “A new quality assessment parameter for optical coherence tomography,” Br. J. Ophthalmol. 90(2), 186–190 (2006).   
30. A. Lozzi, A. Agrawal, A. Boretsky, C. G. Welle, and D. X. Hammer, “Image quality metrics for optical coherence angiography,” Biomed. Opt. Express 6(7), 2435–2447 (2015).   
31. M. Wojtkowski, A. Kowalczyk, R. Leitgeb, and A. F. Fercher, “Full range complex spectral optical coherence tomography technique in eye imaging,” Opt. Lett. 27(16), 1415–1417 (2002).   
32. S. Wang and K. V. Larin, “Optical coherence elastography for tissue characterization: a review,” J. Biophotonics 8(4), 279–302 (2015).

# Biomedical Optics EXPRESS

33. S. Makita, K. Kurokawa, Y.-J. Hong, M. Miura, and Y. Yasuno, “Noise-immune complex correlation for optical coherence angiography based on standard and Jones matrix optical coherence tomography,” Biomed. Opt. Express 7(4), 1525–1548 (2016).   
34. K. V. Larin and D. D. Sampson, “Optical coherence elastography - OCT at work in tissue biomechanics [Invited],” Biomed. Opt. Express 8(2), 1172–1202 (2017).   
35. J. Xu, S. Song, Y. Li, and R. K Wang, “Complex-based OCT angiography algorithm recovers microvascular information better than amplitude- or phase-based algorithms in phase-stable systems,” Phys. Med. Biol. 63(1), 015023 (2017).   
36. B. Braaf, S. Donner, A. S. Nam, B. E. Bouma, and B. J. Vakoc, “Complex differential variance angiography with noise-bias correction for optical coherence tomography of the retina,” Biomed. Opt. Express 9(2), 486–506 (2018).   
37. S. Nadarajah, “A review of results on sums of random variables,” Acta Appl. Math. 103(2), 131–140 (2008).   
38. S. Fialová, M. Augustin, M. Glösmann, T. Himmel, S. Rauscher, M. Gröger, M. Pircher, C. K. Hitzenberger, and B. Baumann, “Polarization properties of single layers in the posterior eyes of mice and rats investigated using high resolution polarization sensitive optical coherence tomography,” Biomed. Opt. Express 7(4), 1479–1495 (2016).   
39. S. Makita, Y.-J. Hong, M. Miura, and Y. Yasuno, "Degree of polarization uniformity with high noise immunity using polarization-sensitive optical coherence tomography," Opt. Lett. 39(24), 6783-6786 (2014).   
40. B. Baumann, M. Augustin, A. Lichtenegger, D. J. Harper, M. Muck, P. Eugui, A. Wartak, M. Pircher, and C. K. Hitzenberger, “Polarization-sensitive optical coherence tomography imaging of the anterior mouse eye,” J. Biomed. Opt. 23(08), 1 (2018).   
41. Z. Liu, K. Kurokawa, F. Zhang, J. J. Lee, and D. T. Miller, “Imaging ganglion cells in the living human retina,” Proc. Nat. Acad. Sci. 114(48), 12803–12808 (2017).   
42. J. J. Liu, A. J. Witkin, M. Adhi, I. Grulkowski, M. F. Kraus, A.-H. Dhalla, C. D. Lu, J. Hornegger, J. S. Duker, and J. G. Fujimoto, “Enhanced Vitreous Imaging in Healthy Eyes Using Swept Source Optical Coherence Tomography,” PLoS One 9(7), e102950 (2014).   
43. C. Blatter, E. F. J. Meijer, A. S. Nam, D. Jones, B. E. Bouma, T. P. Padera, and B. J. Vakoc, “In vivo label-free measurement of lymph flow velocity and volumetric flow rates using Doppler optical coherence tomography,” Sci. Rep. 6(1), 29035 (2016).   
44. M. Almasian, T. G. van Leeuwen, and D. J. Faber, “OCT amplitude and speckle statistics of discrete random media,” Sci. Rep. 7(1), 14873 (2017).   
45. T. B. Dubose, D. Cunefare, E. Cole, P. Milanfar, J. A. Izatt, and S. Farsiu, “Statistical models of signal and noise and fundamental limits of segmentation accuracy in retinal optical coherence tomography,” IEEE Trans. Med. Imaging 37(9), 1978–1988 (2018).   
46. M. Sugita, R. A. Brown, I. Popov, and A. Vitkin, “K-distribution three-dimensional mapping of biological tissues in optical coherence tomography,” J. Biophotonics 11(3), e201700055 (2018).   
47. W. C. van Etten, Introduction to Random Signals and Noise (John Wiley & Sons, 2005).   
48. Y. Lim, M. Yamanari, S. Fukuda, Y. Kaji, T. Kiuchi, M. Miura, T. Oshika, and Y. Yasuno, “Birefringence measurement of cornea and anterior segment by office-based polarization-sensitive optical coherence tomography,” Biomed. Opt. Express 2(8), 2392–2402 (2011).