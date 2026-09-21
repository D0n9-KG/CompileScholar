# On the performance quantification of resonant refractive index sensors

Ian M. White and Xudong Fan

Biological Engineering Department, University of Missouri - Columbia

240D Bond Life Sciences Center, 1201 E. Rollins Street, Columbia, MO 65211

fanxud@missouri.edu

http://web.missouri.edu/~fanxud

Abstract: Refractive index (RI) sensors based on optical resonance techniques are receiving a high degree of attention because of the need to develop simple, low-cost, high-throughput detection technologies for a number of applications. While the sensing mechanism of most of the reported RI sensors is similar, the construction is quite different from technique to technique. It is desirable to have a uniform mechanism for comparing the various RI sensing techniques, but to date there exists a degree of variation as to how the sensing performance is quantified.

Here we set forth a rigorous definition for the detection limit of resonant RI sensors that accounts for all parameters that affect the detection performance. Our work will enable a standard approach for quantifying and comparing the performance of optical resonance-based RI sensors. Additionally, it will lead to design strategies for performance improvement of RI sensors.

©2008 Optical Society of America

OCIS codes: (140.4780) Optical resonators; (230.3990) Micro-optical devices; (130.6010)

Sensors.

# References and links

#89212 - $15.00 USD

Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1020

# 1. Introduction

Optical refractive index (RI) sensors are widely researched for a number of applications and are prominent among the commercial landscape of current sensing technologies. This sensing paradigm promises real-time results and minimal sample preparation with no fluorescent labeling required. Examples of optical RI sensors include surface plasmon resonance (SPR) [1], 2-D photonic crystal structure [2, 3], long-period fiber grating (LPFG) [4, 5], and various forms of ring resonators [6-15].

In these sensors, an optical resonance can easily be observed; at least a fraction of the corresponding mode is interacting with the sample. A change in RI of the region probed by the resonant mode causes a corresponding frequency shift of the optical resonance of the sensor. The change in resonant frequency is converted to the sensing signal. For applications requiring the analysis of a liquid sample, the sensing signal can be used to determine the RI of the sample as compared to a reference sample.

For biomolecule detection applications, the specific capture of biomolecules at the sensor surface results in a local change in RI, producing a sensing signal that enables quantification of the biomolecules in the sample [3,6,11,12,14].

RI sensors are often quantitatively compared by their RI sensitivity. Here we define the RI sensitivity as the magnitude in shift of the resonant wavelength versus the change in RI of the sample. However, this measure alone is not sufficient for quantitatively characterizing the ability of the sensor to identify and to quantify the targeted material in the sample. To fully describe the performance of the sensor, the detection limit must be presented. The purpose of this correspondence is to put forth a definition for the detection limit for RI sensors for both refractometric and biomolecule sensing applications, and to comment on the factors that determine the detection limit, including RI sensitivity, system resolution, and a number of noise sources.

#89212 - $15.00 USD

Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1021

![](dt=2026-06-02/ht=06/b3dbf5ea3582b5e13aea570029f5925846e5eb59cf92135581c7e39cefe906ae.jpg)

![](dt=2026-06-02/ht=06/e168a87feec9e75b49eb379f73e45fc938435cf3c74e408c4b35205a4b1edf9f.jpg)

# 2. Detection limit

Consider an RI sensor in contact with a liquid sample, with a resonant mode as illustrated in Fig. 1(A). If the RI of the sample changes, the resonance condition for the device changes, and as a result, the resonant mode shifts to a new wavelength [Fig. 1(A)]. The magnitude of the wavelength shift divided by the change in RI is the sensitivity of the device. In other words, if the spectral shift is measured for several changes in RI, the sensitivity is the slope of the generated trend, as illustrated in Fig. 1(B). The sensitivity is given in units of nm/RIU, where the difference between a RI of 1.4 and 1.3 is 0.1 RIUs.

A RI sensor's sensitivity is governed by the fraction of the optical mode that interacts with the sample. For many resonant-based RI sensors, a change in sample RI, $\Delta n_{s}$ , results in a spectral shift $\Delta \lambda$ is given by

$$
\Delta \lambda = \eta \Delta n _ {s} \frac {\lambda}{n _ {\text {e f f}}} \tag {1}
$$

where $\eta$ is the fraction of optical intensity that exists in the liquid sample [5,14] and $n_{\mathrm{eff}}$ is the effective RI experienced by the resonant mode. Two resonant RI sensor configurations of note do not fit Eq. (1). For the first configuration, LPFG, the sensitivity is also related to the fraction of light interacting with the sample. The sensitivity increases as a function of $\eta_{\mathrm{c}} - \eta_{\mathrm{cl}}$ , where $\eta_{\mathrm{c}}$ and $\eta_{\mathrm{cl}}$ are the fraction of optical intensity in solution in the core mode and the cladding mode, respectively [4]. The expressions for the second one, SPR, sensitivity in a prism-coupled and in a grating-coupled configuration are given in Homola [16].

Demonstrated RI sensitivities for various RI sensors are listed in Table 1. In general, designs that enable high interaction between the resonant mode and the sample have high sensitivity. In contrast, in a conventional ring resonator, only around $1 - 3\%$ of the optical mode interacts with the sample [8,12], which leads to a relatively low sensitivity.

Table 1. Examples of demonstrated RI sensitivity for various RI sensing technologies.

![](dt=2026-06-02/ht=06/57f1e065b800f06c25eac466f7bde69cbed24ab65aadffa4623dbd845576f6af.jpg)

<table><tr><td>RI sensor configuration</td><td>Demonstrated RI sensitivity</td></tr><tr><td>SPR (prism coupled)</td><td>7120 nm/RIU [17]</td></tr><tr><td>SPR (grating coupled)</td><td>3365 nm/RIU [18]</td></tr><tr><td>LPFG</td><td>~ 6000 nm/RIU [4]</td></tr><tr><td>Capillary ring resonator</td><td>800 nm/RIU [19]</td></tr><tr><td>2-D Photonic crystal</td><td>200 nm/RIU [2]</td></tr><tr><td>Planar ring resonator</td><td>212 nm/RIU [20]</td></tr><tr><td>Microsphere ring resonator</td><td>30 nm/RIU [8]</td></tr></table>

#89212 - $15.00 USD Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1022

However, the magnitude of the spectral shift (i.e., the sensitivity) does not wholly articulate the capability of the device to detect and quantify the properties of the sample. Equally important is the ability to precisely and accurately measure the spectral shift that results from the sample. Here we introduce the concept of the sensor resolution, which characterizes the smallest possible spectral shift that can be accurately measured. As described later, this term takes into account the spectral resolution of the system and a number of noise parameters. The sensitivity (S) and the sensor resolution (R) combine to form the detection limit (DL) of the device:

$$
D L = \frac {R}{S}. \tag {2}
$$

For refractometric sensing, the DL reports the smallest sample RI change that can accurately be measured. For biomolecule
sensing, the DL describes the minimum amount of analyte that the RI sensor can accurately quantify.

Spectral resolution and system noise factor into the DL because of the methodology for measuring the spectral shift in response to a sample. In most resonant-based RI sensors, the resonant mode has a Lorentzian spectral profile. One simple method to monitor the spectral shift is to track the position of the extremum (i.e., minimum or maximum, depending on the measurement configuration). The total sensor response is the spectral difference between the final and initial extrema, as illustrated in Fig. 1(A).

If infinitely high spectral resolution were available and absolutely zero system noise were present, the sensor performance could be characterized by the sensitivity alone. However, spectral resolution and system noise detract from the precision and accuracy with which the true center of the resonant mode can be located. The minimum error in determining the actual mode spectral position leads to finite sensor resolution, and subsequently to the need for the sensor DL.

There are two classes of noise that contribute to errors in determining the positions of the resonant mode: amplitude variations and spectral variations. Amplitude noise refers to the cumulative noise added to the spectral mode profile. Noise sources include thermal and shot noise in the photodetector, laser relative intensity noise, and quantization error. Figure 2(A) shows a spectral mode trace with noise signals added to it. In a typical system, the extremum is located quickly and automatically by the system's processor, such as a computer.

With added amplitude noise, the actual extremum is not likely to be at the exact center frequency of the Lorentzian shape as it should be. Thus, the amplitude noise processes result in a random spectral deviation of the measured spectral location of the resonant mode.

It is important to analyze the random process of this deviation and its dependence on the magnitude of the noise and the linewidth of the mode. Although the noise processes are well characterized by standardized statistical distributions, the extremum operation prevents a direct analytical solution for the statistical distribution of the resulting spectral variation of the actual extremum. However, the system can be studied numerically using a Monte Carlo simulation. Here we use this technique to determine the statistical variance of the extremum of a Lorentzian-shaped mode when a random noise signal is added to the amplitude of the mode. For simplicity we draw each amplitude noise data point from a white Gaussian distribution (in reality, shot noise is produced by a Poisson process).

The Lorentzian modes pictured in Fig. 2(A) have a noise signal added to produce a signal to noise ratio (SNR) of $40\mathrm{dB}$ and $60\mathrm{dB}$ (60 dB is a good SNR for a typical photonic link), respectively, where the signal is defined as the height of the signal (power normalized to 1) and the noise is the variance of the noise distribution. In the Monte Carlo simulations, these signals are created repeatedly while the maximum is found for each resulting mode profile in each iteration.

The deviation of the spectral location of the maximum amplitude from the actual center wavelength of the mode is determined for each iteration and then the resulting data set is statistically analyzed to determine the standard deviation. Figure 2(B) plots the three standard deviations $(3\sigma)$ value of the determined maxima for SNRs ranging from 20 dB

#89212 - $15.00 USD Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1023

![](dt=2026-06-02/ht=06/b890f738a4400bcac417d291a3612c772bfc6ad1ec358d7ed622a4095d02a6d4.jpg)

![](dt=2026-06-02/ht=06/070ece0fed749cfcf15b02249e5ff612a4d54a299bafbe17015392baac4f54ea.jpg)

to $80~\mathrm{dB}$ , and for Q-factors between $10^{3}$ and $10^{8}$ . The inset shows a numerically generated probability density function (PDF) for the relative spectral position of the maximum value for a particular simulation run. The PDF is similar to a Gaussian distribution. The simulation results reveal that having a high Q-factor is advantageous in reducing the spectral noise of the sensor.

This phenomenon can be explained by the fact that the noise signal results in a finite probability for samples within the optical bandwidth of the resonant mode to be identified as the extremum [see Fig. 2(A)]. Resonant modes with narrower linewidth filter the spectral noise more effectively, which leads to lower spectral deviation from the actual center of the mode. As a result, Q-factor plays an important role in the DL of the sensor.

Although an analytical solution for the statistical variance of the location of the mode amplitude maximum value is not available, the Monte Carlo simulation results provide an opportunity to approximate the relationship between Q-factor, SNR, and the resulting spectral variation of the mode's maximum value. The simulation results reveal a linear dependence of the standard deviation on the linewidth and an exponential relationship with the SNR. It is found that within the ranges of parameters considered here, the numerical results can be approximated by:

$$
\sigma \approx \frac {\Delta \lambda}{4 . 5 \left(S N R ^ {0 . 2 5}\right)} \tag {3}
$$

where $\sigma$ is the standard deviation of the resulting spectral variation and $\Delta \lambda$ is the full-width-half-max of the mode amplitude and is related to the Q-factor by $Q = \lambda / \Delta \lambda$ . SNR is in linear units in this expression (e.g., $60~\mathrm{dB} = 10^{6}$ ). The results of this estimation are plotted in Fig. 2(B), and show good agreement with the Monte Carlo simulation results. Equation (3) is generally applicable to any RI sensor technology in which the measured extremum value is used to identify the spectral location of the resonant mode.

In addition to amplitude noise, there are spectral noise factors that also contribute to variation in the spectral location of the resonant mode and thus to the sensor resolution. First is the thermal variation of the system. All RI sensors are susceptible to thermal-induced fluctuations because the sensor material and the sample have non-zero thermo-optic coefficients and thermal expansion coefficients. Thus, slight changes in temperature cause the

#89212 - $15.00 USD Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1024

RI of the material and of the sample to change, which, as shown by Eq. (1), causes the spectral position of the resonant mode to shift. This results in another noise source contributing to error in determining the spectral shift produced by the analyte.

The spectral resolution of the system setup also may limit the precision with which the resonant mode position may be identified. Depending on the measurement setup, either the laser or the optical detection mechanism may limit the resolution. In one common setup that is used for very narrow resonant mode linewidth RI sensors, the laser is tuned over a very small spectral range – on the order of $\mathrm{pm}$ – while a photodetector measures the amplitude at the output of the system.

It is expected that the processing unit in the setup can sample the photodetector voltage at a very high rate compared to the laser tuning speed. The limitation in this case is due to the linewidth of the laser, which is typically less than $1\mathrm{MHz}$ . At $1550\mathrm{nm}$ , $1\mathrm{MHz}$ is equivalent to about 8 fm.

In a setup that is common for relatively large resonant mode linewidth RI sensors, a broadband optical source and an opt
ical spectrum analyzer or spectrometer are used to record the resonant mode. In this case, it is clear that the spectral resolution of the detection device provides the limitation, which may be on the order of $1\mathrm{pm}$ . The resulting error due to spectral limitations of the detection device can be modeled as quantization error. As an example, for a device with a spectral resolution of $1\mathrm{pm}$ , the error in determining the position of the resonant mode is uniformly distributed between $-0.5\mathrm{pm}$ and $0.5\mathrm{pm}$ , and has a resulting standard deviation of $\sigma_{spec-res} = 0.29\mathrm{pm}$ .

Once the statistical properties of all of the noise sources are understood, the sensor resolution can be determined. Here we use the typical convention of establishing the resolution as $3\sigma$ of the noise in the system. The total system noise variance can be approximated by summing all of the individual noise variances, i.e.

$$
3 \sigma = 3 \sqrt {\sigma_ {a m p l - n o i s e} ^ {2} + \sigma_ {t e m p - i n d u c e d} ^ {2} + \sigma_ {s p e c t - r e s} ^ {2}}.
$$

For illustration, we can compare the DL of two very different cases. Consider one RI sensor with a sensitivity of $1000\mathrm{nm / RIU}$ , a Q-factor of $10^{4}$ , a SNR of 60 dB, a spectral resolution of $1\mathrm{pm}$ , and temperature stabilization with standard deviation of $10\mathrm{fm}$ . Consider another RI sensor with a sensitivity of $25\mathrm{nm / RIU}$ , a Q-factor of $10^{7}$ , a SNR of 60 dB, a spectral resolution of $1\mathrm{fm}$ , and a temperature stabilization with standard deviation of $10\mathrm{fm}$ . Assuming an operating wavelength of $1550\mathrm{nm}$ and using Eq.

(3), we can compare the refractometric detection limit of these two RI sensors. The parameters and the resulting DLs are compared in Table 2. Despite the lower sensitivity, the high-Q sensor has slightly better performance because of the dramatically lower spectral variation due to amplitude noise. This result accents the need for a rigorous method of performance quantification for RI sensors instead of relying on the RI sensitivity value to indicate the sensor's performance.

This illustration is also designed to emphasize the fact that high-Q-factor RI sensors are typically limited in performance by temperature stabilization, while low-Q-factor RI sensors are typically limited by amplitude noise and spectral resolution. Sensor design should focus on the minimization of these issues in order to minimize the detection limit.

Table 2. Comparison of the detection limit of two different resonant modes with largely differing Q-factor and RI sensitivity.

![](dt=2026-06-02/ht=06/a648b35da634a52811b64028008eef504a7fbdd373d64f79968ae7ee853d5735.jpg)

<table><tr><td>Q-factor</td><td>Q = 10^4</td><td>Q = 10^7</td></tr><tr><td>RI sensitivity</td><td>1000 nm/RIU</td><td>25 nm/RIU</td></tr><tr><td>Spectral resolution</td><td>1 pm</td><td>1 fm</td></tr><tr><td>Temp stabilization (σ)</td><td>10 fm</td><td>10 fm</td></tr><tr><td>σ of extremum</td><td>1.1 pm</td><td>1.1 fm</td></tr><tr><td>Calculated sensor resolution R</td><td>3.4 pm</td><td>30.3 fm</td></tr><tr><td>Detection limit DL</td><td>3.4 × 10^-6 RIU</td><td>1.2 × 10^-6 RIU</td></tr></table>

#89212 - $15.00 USD Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1025

Our analysis can also be used to investigate the role of the optical mode fraction that interacts with the sample, $\boldsymbol{\eta}$ . Equation (1) shows that sensitivity increases linearly with $\boldsymbol{\eta}$ , and Eq. (2) states that DL decreases linearly with S. Therefore, it seems logical to conclude that a higher $\boldsymbol{\eta}$ results in a lower detection limit. However, closer examination reveals that this is not always the case.

Although sensitivity increases with increasing $\boldsymbol{\eta}$ , the absorption of photons by the sample medium also increases with increasing $\boldsymbol{\eta}$ . This decreases the Q-factor of the resonant mode, which is equivalent to an increase in the mode linewidth. This increasing linewidth serves to counteract the increasing sensitivity, and, depending on the absorption coefficient of the medium, may have an equalizing effect on the actual DL.

Consider an optical ring resonator sensor as an example. The ratio of the evanescent intensity in the sample to the total mode intensity is $\eta$ . The sample has an absorption given by $\alpha$ , where $\alpha$ is related to the Q-factor of the mode by:

$$
\eta \alpha = \frac {2 \pi n}{\lambda Q _ {\alpha}} \tag {4}
$$

where $Q_{\alpha}$ is the $Q$ factor due to the absorption of photons and the observed $Q$ -factor is defined as $1 / Q = 1 / Q_{\alpha} + 1 / Q_{0}$ . Figure 3(A) compares two Lorentzian modes from a ring resonator with $\eta = 0.05$ and $\eta = 0.01$ when the evanescent field is interacting with water, which has $\alpha \approx 1000 \, \mathrm{m}^{-1}$ at $1550 \, \mathrm{nm}$ [21]. As shown in the previous analysis, the broader linewidth leads to poorer sensor resolution and degrades the DL. Using Eqs.

(3) and (4) to calculate the noise term due to the mode linewidth, we can calculate the DL for various sample absorption $\alpha$ and mode fraction in the sample $\eta$ . Figure 3(B) plots the calculated detection limit for $\eta$ of 0.01, 0.05, and 0.1, assuming a wavelength of $1550 \, \mathrm{nm}$ , a mode amplitude SNR of $60 \, \mathrm{dB}$ , and a $Q_{0}$ of $10^{8}$ .

While higher $\eta$ results in a larger sensitivity, the calculations show that for $\alpha > 10 \, \mathrm{m}^{-1}$ , there is no advantage in terms of detection limit because the increasing linewidth counteracts the increasing sensitivity. Alternatively, for $\alpha < 1 \, \mathrm{m}^{-1}$ , an increasing $\eta$ results in a corresponding improvement to the detection limit, as intuitively expected.

The dashed lines in Fig. 3(B) show the detection limit for $\eta = 0.05$ and $\eta = 0.1$ if a temperature-induced spectral fluctuation of $\sigma = 10\mathrm{fm}$ is also factored into the calculation. The calculation indicates that for $\alpha < 100\mathrm{m}^{-1}$ , the temperature-induced fluctuation dominates,

![](dt=2026-06-02/ht=06/84f676d6e6f6c6b1e9b330e420593d10b937a05124df04ea0e8dfda117a9bf4e.jpg)

![](dt=2026-06-02/ht=06/4e624c9bf68b3f4ec6335553c9140981c554aa1b6fd6758ca06ffd87ede0e0dc.jpg)

#89212 - $15.00 USD Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1026

while for $\alpha > 100\mathrm{m}^{-1}$ , the effect on linewidth of the optical absorption due to the sample dominates the DL calculation. When the temperature-induced spectral fluctuation is dominating, an increase in $\eta$ results in a linear improvement in detection limit. A direct consequence of the above result is that under the same experimental condition, the detection limit is better for the solution of higher RI, as more light is pulled into the solution [19].

This analysis offers clues for sensor design parameters that can improve performance. For samples that have low absorption, increasing the mode fraction of light interacting with the sample improves the sensor detection limit. For samples with higher absorption, there is no advantage to increasing the interaction between the resonant mode and the sample. Furthermore, consider the case of water, which has two orders of magnitude lower absorption for visible light as compared to $1550\mathrm{nm}$ .

Figure 3(B) shows the significant advantage in sensor performance by operating at the lower wavelength and wit
h higher interaction between the resonant mode and the sample. Armani et al. recently reported significant reduction in solvent absorption at near infrared by replacing $\mathrm{H}_2\mathrm{O}$ with $\mathrm{D}_2\mathrm{O}$ [10]. This is another strategy to improve the RI sensor performance.

Altogether, the analyses presented here provide insight into how to optimize sensor design to improve the detection limit. In the case of high-Q sensors, controlling the temperature-induced spectral fluctuations is the key to an excellent detection limit. Design options include temperature control using a thermal electric cooler (TEC) [22], the use of materials with advantageous thermal properties [23], or utilizing a reference channel to subtract the common mode noise, which includes thermally-induced spectral fluctuations [24].

Also, optimizing the wavelength of operation and the absorption of the sample contribute to improved detection limit, and lead to the opportunity to further improve the performance by increasing the interaction between the resonant mode and the sample. For low Q-factor sensing devices, these parameters are not expected to be particularly important because the detection limit is dominated by amplitude noise added to the recorded mode spectrum and by spectral resolution.

Reducing the amplitude noise and applying signal processing techniques to reduce the recorded noise are expected to improve the detection performance of these devices. With these improvements, it is possible to break the RIU detection limit, which is currently on the order of $10^{-7}$ RIU [25, 26].

These analyses apply to resonance-based RI sensors operated in a peak-detection operation. Additionally, resonance-based RI sensors can be operated in an alternative detection scheme in which a narrow-bandwidth fixed-wavelength laser is used instead of a broadband or tunable laser [11, 12]. The system is configured so that the wavelength of the laser is near the spectral location of the half-max point of the resonant mode on either side of the extremum.

When the mode shifts due to RI changes, the amplitude of the optical signal at the sensor output increases or decreases according to the shape of the mode and the magnitude of the spectral shift of the mode. The analysis presented above does not directly apply to this configuration, and thus it remains as an open problem for future development. In this analysis, the sensitivity is a combination of the spectral shift of the resonant mode and of the slope of the spectral mode profile (and thus the Q-factor). Thus, again, the linewidth of the mode factors into the detection performance.

Additionally, amplitude noise may also play a significant role in the detection limit, as the measured intensity is the sensing signal in this case.

# 3. Biomolecule detection limit

A number of RI sensors, including SPR and ring resonator sensors, have been utilized for biomolecule detection. Essentially, the sensors are detecting the RI change resulting from the specific capture of biomolecules at the surface. The RI sensor is utilizing the evanescent field at the sensor surface, where biorecognition techniques can be used to ensure specificity in detection. Non-evanescent-detection-based sensors are not considered in this section.

#89212 - $15.00 USD Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1027

It is relatively straightforward to convert a sensor's RI sensitivity into a sensitivity for biomolecules because the capture of biomolecules at the sensor surface changes the RI in the region probed by the sensor's evanescent field. An analysis to convert RI sensitivity to biomolecule sensitivity has been presented and experimentally verified for generalized ring resonator sensors by Zhu, et al., in [14]. In that work, the authors began with Eq. (1) above. The analysis is generally applicable to all resonant-based RI sensors that also conform to Eq. (1). The analysis leads to an expression for the spectral shift of the resonant mode, $\delta \lambda$ , for biomolecule capture given an bulk RI sensitivity of $S$ :

$$
\frac {\delta \lambda}{\lambda} = \sigma_ {p} \alpha_ {e x} \frac {2 \pi \sqrt {n _ {m} ^ {2} - n _ {s} ^ {2}}}{\varepsilon_ {0} \lambda^ {2}} \frac {n _ {m}}{n _ {s} ^ {2}} S. \tag {5}
$$

In this expression, $\sigma_{\mathrm{p}}$ is the surface density of the captured biomolecules, $\alpha_{\mathrm{ex}}$ is the excess polarizability of the molecule, $n_m$ is the RI of the sensor material (e.g., glass in a typical ring resonator), and $n_s$ is the RI of the sample buffer. From this, we can define a sensitivity for biomolecule detection:

$$
S _ {B} = \frac {\delta \lambda}{\sigma_ {p}}. \tag {6}
$$

A detection limit analysis for biomolecules is performed in the same manner as the bulk RI analysis presented above. This provides a DL in terms of analyte molecules per area or mass of analyte per area on the sensor surface. This value can be used to compare sensors in terms of the ability to detect a low density of molecules on the sensor surface, which indeed is an important standard. Note, however, that this does not include the sensor's ability to capture molecules on the surface, which also plays an important part in how well a sensor can detect and quantify a low concentration of analyte molecules in a sample. Naturally, the experimentally measured detection limit is the optimal figure for comparison across different RI sensors.

# 4. Summary and discussion

We have proposed a method for characterizing the performance of RI sensors that utilize optical resonance. The method offers a measure for comparison based on the sensor's actual capability to detect and quantify small changes in the RI of the sample. Instead of simply relying on the RI sensitivity of the sensor, our method uses a DL that is based on the sensitivity as well as the sensor resolution, where the resolution depends on a number of noise sources.

We have also shown that the method can easily be applied to biomolecule detection by utilizing a relationship between the bulk RI sensitivity and the sensitivity to capture of biomolecules on the sensor surface. Our hope is that this proposed methodology contributes to a standardized comparison tool for optical RI sensors that accurately represents the utility of the sensor.

Additionally, in identifying the impact of all of the parameters involved in determining DL, we have revealed design strategies that are focused on improving the actual ability of the RI sensor to identify and quantify the target in the sample.

# Acknowledgments

This work is sponsored by the Wallace H. Coulter Foundation, 3M Non-Tenured Faculty Award, and the NIH (1K25 EB006011).

#89212 - $15.00 USD Received 31 Oct 2007; revised 3 Jan 2008; accepted 10 Jan 2008; published 11 Jan 2008

(C) 2008 OSA

21 January 2008 / Vol. 16, No. 2 / OPTICS EXPRESS 1028