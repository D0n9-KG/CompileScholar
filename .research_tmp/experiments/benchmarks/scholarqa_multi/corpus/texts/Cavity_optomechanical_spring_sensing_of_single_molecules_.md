ARTICLE

Received 12 Feb 2016 | Accepted 21 Jun 2016 | Published 27 Jul 2016

DOI: 10.1038/ncomms12311

OPEN

# Cavity optomechanical spring sensing of single molecules

Wenyan Yu $^{1,*}$ , Wei C. Jiang $^{2,*}$ , Qiang Lin $^{2,3}$ & Tao Lu $^{1}$

Label-free bio-sensing is a critical functionality underlying a variety of health- and security-related applications. Micro-/nano-photonic devices are well suited for this purpose and have emerged as promising platforms in recent years. Here we propose and demonstrate an approach that utilizes the optical spring effect in a high-Q coherent optomechanical oscillator to dramatically enhance the sensing resolution by orders of magnitude compared with conventional approaches, allowing us to detect single bovine serum albumin proteins with a molecular weight of 66 kDa at a signal-to-noise ratio of 16.8. The unique optical spring sensing approach opens up a distinctive avenue that not only enables biomolecule sensing and recognition at individual level, but is also of great promise for broad physical sensing applications that rely on sensitive detection of optical cavity resonance shift to probe external physical parameters.

Sensitive detection of a single nanoparticle/molecule is essential for many applications ranging from medical diagnostics, drug discovery, security screening and to environmental science. In the past decades, a variety of approaches have been developed to observe single particles down to molecular scale $^{1,2}$ , among which optical detection based on high-Q microcavities has shown significant advantages for its high sensitivity and label-free operation $^{3-12}$ . Binding of a particle to a high-Q optical microcavity perturbs the cavity mode at a resonance wavelength of $\lambda_{0}$ , resulting in a cavity resonance shift of $\delta\lambda$ which in turn changes the cavity transmission. This mechanism underlies the majority of current microcavity sensors, with a sensing resolution dependent critically on the optical quality factor (Q) $^{13}$ . To date, the highest resolution reported is a resonance shift of $(\delta\lambda/\lambda_{0})=3\times10^{-10}$ achieved with an optical Q of one hundred million at a visible wavelength in an aqueous environment $^{8}$ , which, however, is still larger than that induced by a single protein binding event $^{6}$ . Consequently, detection of a single protein molecule down to 1 kDa requires incorporating a plasmonic nanoantenna on the microcavity to enhance the resonance wavelength shift $^{14-17}$ , at the price of a significant reduction of the effective detection area.

On the other hand, the optical wave cycling inside the microcavity is able to produce a radiation pressure that interacts with the mechanical motion of the device. Such optomechanical coupling flourishes in profound physics that has been intensively explored in recent years, particularly in the context of quantum control of mesoscopic mechanical motion $^{18,19}$ . When the laser wavelength $\lambda_{1}$ is blue detuned to the cavity resonance, the optical wave can efficiently boost the mechanical motion above the threshold of regenerative oscillation $^{18,19}$ , resulting in highly coherent optomechanical oscillation (OMO) with a narrow mechanical linewidth. Of particular interest is that the optical wave inside the cavity is able to produce an effective mechanical rigidity $^{19}$ , leading to an OMO frequency $f_{m}$ depending sensitively on the laser-cavity detuning $\Delta_{\lambda} = \lambda_{1} - \lambda_{0}$ . Consequently, any tiny perturbation to the cavity resonance wavelength, $\delta\lambda$ , induced by particle/molecule binding would be readily transferred to the frequency shift, $\delta f_{m}$ , of the mechanical motion: $\delta f_{m} = -\frac{df_{m}}{d\Delta\lambda}\delta\lambda$ , thus enabling an efficient transduction mechanism to amplify the resonance wavelength sensing.

As the minimal detectable frequency shift of OMO is determined by its linewidth $\Delta f_{m}$ , the minimal detectable optical cavity resonance shift is thus given by $\left(\frac{\delta\lambda}{\lambda_{0}}\right)_{\min} = \frac{\Delta f_{m}/\lambda_{0}}{(-\frac{df_{m}}{d\Delta_{\lambda}})}$ . With a narrow linewidth of coherent OMO and a significant frequency tuning slope $\left|\frac{df_{m}}{d\Delta_{\lambda}}\right|$ (see below), the intriguing optical spring effect would thus offer an elegant approach for sensitive probing of cavity resonance variation, with a sensing resolution given by (Supplementary Note 1).

$$
\left(\frac {\delta \lambda}{\lambda_ {0}}\right) _ {\min} = \frac {1}{\eta_ {\mathrm{om}} Q _ {\mathrm{m}} ^ {\mathrm{eff}} Q _ {\mathrm{t}}}, \tag {1}
$$

where $Q_{m}^{eff}$ is the effective mechanical Q factor of OMO, defined as the ratio between the frequency $f_{m}$ and the linewidth $\Delta f_{m}$ of the OMO: $Q_{m}^{eff} \equiv \frac{f_{m}}{\Delta f_{m}}$ . $Q_{t}$ is the loaded optical Q and $\eta_{om}$ represents the optomechanical transduction factor for sensing whose magnitude depends on laser-cavity detuning, with a value in the order of $\eta_{om} \sim 1$ . Equation (1) shows clearly that the sensing resolution scales not only with the optical Q of the cavity as in conventional microcavity sensors, but also with the effective mechanical Q of OMO. Consequently, in principle, the proposed cavity optomechanical spring sensing is able to enhance the sensing resolution by about a factor of $Q_{m}^{eff}$ compared with conventional approaches. As we will show below, the effective mechanical Q of coherent OMO in our device can reach a value above $10^{6}$ , resulting in a sensing resolution enhanced by orders of magnitude that is sufficient for single-molecule detection.

To date, cavity optomechanics has been widely applied as a sensitive approach for probing mechanical displacement $^{18,19}$ . However, the magnitude of mechanical motion is not relevant here. Instead, the optically induced frequency shift of the OMO is employed as the information carrier to transduce and to amplify the molecule binding signal. In this sense, we call our approach optical spring sensing. Although the optical spring effect has been known for a decade $^{19}$ , we realize its potential for particle and molecule sensing.

As the optomechanical effect is intrinsic to a high-Q microcavity, the proposed approach does not rely on any specific external sensing element attached to the device (for example, a plasmonic nanoantenna $^{15-17}$ ) and is thus capable of fully utilizing the entire effective sensing area offered by a whispering-gallery microcavity which is more than five orders of magnitude larger than that of plasmonic devices. For the same reason, it does not rely on any gain medium and is therefore universal to different material platforms as long as the device has reasonably high optical Q. On the other hand, this approach is distinctive from the conventional micro-/nano-mechanical sensing $^{2,20-22}$ where particle detection is realized by monitoring the mechanical frequency shift directly induced by the mass change from a particle attaching, which exhibits a minimal detectable mass of $(\delta m)_{\mathrm{min}} = \frac{2m_{\mathrm{eff}}}{Q_{\mathrm{m}}^{\mathrm{eff}}}$ that relies critically on the motional mass $m_{eff}$ and the effective mechanical Q of the sensor (see Supplementary Note 2 for detailed discussion). This mechanical sensing principle underlies the optomechanical sensors developed recently $^{23-25}$ while in combination with optical actuation and readout. With significant intrinsic motional masses, these sensors, however, can only detect $\sim 1-\mu m$ -diameter silica beads with a sub-picogram resolution $^{23-25}$ . For nanomechanical sensing, achieving single molecule resolution requires extremely tiny mass of employed nanomechanical oscillators $^{2,20-22}$ .

Here we show that the optical spring sensing principle proposed above is able to dramatically enhance the sensing resolution by orders of magnitude compared with conventional approaches. It allows us to detect single silica nanobeads as small as 11.6 nm in radius and bovine serum albumin (BSA) proteins with a molecular weight of 66 kDa at a signal-to-noise ratio (SNR) of 16.8.

# Results

OMO of a silica microsphere in buffered solution. To verify the optical spring sensing principle proposed above, we carried out experiments in a silica microsphere with a diameter of about 100 $\mu$ m. The device exhibits an intrinsic optical Q as high as $4.8 \times 10^{6}$ at a wavelength $\sim$ 974 nm in the aqueous environment (Fig. 1c), which is close to the theoretical limit. With such a high optical Q, the optical wave inside the microsphere produces a strong radiation pressure that efficiently actuates the radial breathing mechanical motion of the microsphere (Fig. 1a). Consequently, by injecting an optical power of 3.0 mW into the cavity (Fig. 1d), we are able to boost the mechanical mode above the threshold even in the aqueous environment $^{26}$ , resulting in coherent OMO at a frequency of 262 kHz with a mechanical linewidth as narrow as 0.1 Hz (Fig. 1f), corresponding to an effective mechanical Q of $2.6 \times 10^{6}$ . As shown in Fig. 1e, the significant OMO leads to a harmonic comb on the power spectrum of the cavity transmission, a feature of coherent OMO resulting from the nonlinear transduction of the optical cavity $^{27,28}$ .

![](images/fd93a2030fa0319cf8210231418f763c131fcabacff954e2a0a0645056188b7c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Laser"] -->|Tapered fiber| B["Image of BSA molecule"]
    B --> C["Photo detector"]
    C --> D["Optical transmission"]
    D --> E["Time graph: δf_m, δλ, δf_m"]
    E --> F["Mechanical spectrum"]
    F --> G["Binding"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fcf,stroke:#333
    subgraph Time
        H["δf_m"] --> I["Time axis f_m"]
        J["δf_m"] --> K["Time axis f_m"]
        L["Time axis f_m"]
    end
```
</details>

b

![](images/b421a8eb9b843088d2417323210efc7515ed775a6744ac939253d7565f177aa2.jpg)

<details>
<summary>natural_image</summary>

Microscopic grayscale image of a spherical object with a 50.0 μm scale bar (no text or symbols on the object itself)
</details>

C   
![](images/821d74fdd713cda69009110ad11c2221509c9563104aa54786c1662cb996a545.jpg)

<details>
<summary>line</summary>

| Laser detuning Δλ' (pm) | Exp. | Fit |
| ------------------------ | ---- | --- |
| -0.4                     | 1.00 | 1.00 |
| -0.2                     | 0.99 | 0.99 |
| 0.0                      | 0.97 | 0.97 |
| 0.2                      | 0.99 | 0.99 |
| 0.4                      | 1.00 | 1.00 |
</details>

![](images/ae922bbc0968e3d09931cfce085c37acec2b87b6b9f8f2ab69b83512f9b1be90.jpg)

<details>
<summary>line</summary>

| Laser detuning Δλ' (pm) | Transsited power (mW) |
| ------------------------ | --------------------- |
| -20                      | 8.5                   |
| -15                      | 7.0                   |
| -10                      | 6.5                   |
| -5                       | 6.0                   |
| 0                        | 1.0                   |
| 5                        | 8.5                   |
</details>

![](images/3047014324ff12aa8e361bf90cc1c706c7b616ec12ba42ebb5cf84d95fad4f26.jpg)

<details>
<summary>line</summary>

| Frequency (MHz) | RF power (dBm) |
| --------------- | -------------- |
| 0.0             | -60            |
| 0.5             | -30            |
| 1.0             | -70            |
| 1.5             | -30            |
| 2.0             | -70            |
</details>

f   
![](images/70f4283d3b1e649e58fd7364a1e90192bbb1dccbab1efcd16218fe71a044b66e.jpg)

<details>
<summary>line</summary>

| Frequency (MHz) | Exp. (dBm) | Fit (dBm) |
| --------------- | ---------- | --------- |
| 0.25            | -75        | -75       |
| 0.26            | -40        | -40       |
| 0.27            | -75        | -75       |
| 0.28            | -80        | -80       |
</details>

Figure 1 | Experiment schematics and device characterization. (a) Schematic illustrating the sensing mechanism. A protein molecule bound to an optomechanically oscillating microsphere yields an optical resonance shift $\delta \lambda$ , which is transduced to a mechanical frequency shift $\delta f_{m}$ . The colour map on the microsphere shows the radial breathing mechanical mode simulated by the finite element method. (b) A scanning electron microscopic (SEM) image of a fabricated silica microsphere. (c) The optical transmission spectrum of the microsphere immersed in DPBS, at a probe laser wavelength of $974\mathrm{nm}$ , with experimental data in blue and theoretical fitting in red. The input power is maintained low enough to characterize the intrinsic optical property of the device, which exhibits an intrinsic optical $Q$ of $4.8\times 10^{6}$ . (d) The optical transmission spectrum at an input laser power of $8.5\mathrm{mW}$ . The coherent OMO was excited with a threshold power of $3.0\mathrm{mW}$ injected into the cavity. (e) An example of the power spectral density of the cavity transmission. The fundamental oscillation frequency is located at $262\mathrm{kHz}$ , with six high-order harmonics clearly visible on the spectrum. (f) The detailed spectrum of the fundamental oscillation tone, with experimental data in blue and theoretical fitting in red. The OMO exhibits a full-width at half maximum of $0.1\mathrm{Hz}$ , corresponding to an effective mechanical $Q$ of $2.6\times 10^{6}$ .

OMO versus laser-cavity detuning. In particular, the strong optical spring effect from the optomechanical coupling results in an OMO frequency sensitively dependent on the laser-cavity detuning (Fig. 2a). As shown in Fig. 2b, when the laser-cavity wavelength detuning $\Delta_{\lambda}$ decreases from $-150\mathrm{fm}$ , the OMO frequency increases from $247\mathrm{kHz}$ to a peak value of $267\mathrm{kHz}$ , and then decreases quickly to about $115\mathrm{kHz}$ when the laser wavelength is tuned close to the centre of the cavity resonance, with a tuning slope of $df_{\mathrm{m}} / d\Delta_{\lambda} \approx -1.5\mathrm{kHzfm}^{-1}$ at a laser-cavity detuning of $\Delta_{\lambda} \approx -70\mathrm{fm}$ . The observed optical spring follows closely the theoretical expectation (grey curve in Fig. 2b; see Supplementary Note 1). The slight discrepancy is likely because the coherent OMO in experiment exhibits a significant amplitude (Fig. 1d) beyond the linear perturbation regime used in the theoretical estimation.

Such an optical spring corresponds to a sensitive optical-to-mechanical frequency transduction, inferring that every 1-fm cavity resonance wavelength shift induced by a particle binding event can be transduced to an OMO frequency change of about 1.5 kHz that is about four orders of magnitude larger than the linewidth of OMO (Fig. 1f). A detailed characterization of the Allan deviation of the OMO frequency (Fig. 2c, see also Methods section) shows a minimum two-sample deviation of 9.5 Hz at the fundamental OMO frequency. This Allan deviation includes all noise in the microcavity, implying a detection resolution of $\delta\lambda/\lambda_{0}\approx6\times10^{-12}$ in the device. This resolution clearly shows the power of the demonstrated approach, which is more than $10^{4}$ times higher than a conventional microcavity sensor with the same optical quality factor $^{13}$ . It is even about 50 times higher than that achieved with an optical Q of $10^{8}$ at a visible wavelength $^{8}$ .

![](images/29e24c5d2919dd3b260bd26957665a99ea54d0190b8e31266d56ef219d03d0dd.jpg)

<details>
<summary>heatmap</summary>

| Frequency (kHz) | Laser detuning Δλ' (pm) |
| --------------- | ------------------------ |
| 0               | 0                        |
| 100             | -2                       |
| 200             | -6                       |
| 300             | -10                      |
| 400             | -8                       |
| 500             | -4                       |
</details>

![](images/a6865e83fea66ccad0c58796df233b50dfb83c343efe2ddc1809f2ba0ac8e442.jpg)

<details>
<summary>line</summary>

| Laser-cavity detuning Δλ (fm) | fm (kHz) |
| ----------------------------- | -------- |
| -160                          | 260      |
| -140                          | 265      |
| -120                          | 270      |
| -100                          | 275      |
| -80                           | 280      |
| -60                           | 275      |
| -40                           | 260      |
| -20                           | 230      |
| 0                             | 180      |
| 20                            | 140      |
| 40                            | 100      |
| 60                            | 60       |
| 80                            | 40       |
| 100                           | 20       |
| 120                           | 10       |
| 140                           | 5        |
| 160                           | 2        |
</details>

![](images/660e3fb7018f78534aa6e100dffec346a986f9d7313a577dd78368948e03972f.jpg)

<details>
<summary>line</summary>

| Time (s) | Fundamental | 2nd harmonic | 3rd harmonic |
| -------- | ----------- | ------------ | ------------ |
| 0.01     | 0.05        | 0.08         | 0.07         |
| 0.1      | 0.02        | 0.04         | 0.03         |
| 1        | 0.01        | 0.02         | 0.015        |
| 10       | 0.05        | 0.1          | 0.08         |
</details>

Figure 2 | OMO versus laser-cavity detuning. (a) Spectrogram of cavity transmitted signal as a function of laser wavelength detuning $\Delta'_{\lambda}$ (see Fig. 1d for the meaning of $\Delta'_{\lambda}$ ), showing the detuning-dependent mechanical frequency. The proportional frequency variations at the second and third harmonics are clearly visible. Every spectrum was averaged over five traces acquired continuously. (b) The OMO frequency as a function of laser-cavity wavelength detuning. The blue crosses show the experimental data and the grey curve shows the theory. The red curve is a polynomial fitting to the experimental data. The dashed circle indicates the operating regime for the particle and molecule sensing, with a frequency tuning slope of $df_{m}/d\Delta_{\lambda} = -1.5\mathrm{kHzfm}^{-1}$ at a laser-cavity detuning of $\Delta_{\lambda} = -70\mathrm{fm}$ . Inset: recorded optical power as a function of laser wavelength detuning. This curve was used to obtain the real laser-cavity wavelength detuning. (c) The two-sample Allan deviations of the fundamental, second and third harmonic tones measured in DPBS in the absence of sensing particle, showing a minimum deviation of $9.5\mathrm{Hz}$ at the fundamental oscillation tone.

Note that the harmonics of the OMO vary proportionally with the OMO fundamental frequency (Fig. 2a) and thus can also be applied for particle sensing. Although this does not improve the sensing resolution due to the same SNR of detection, in practice, the larger frequency shifts on the higher order harmonics significantly facilitate the mechanical spectrum analysis (by allowing to use a coarser resolution bandwidth), which reduces considerably the excessive detection noises from consecutive multiple particles skimming by the cavity surface (see discussions in Methods section).

Detection of silica nanobeads. To characterize the real sensing performance, we performed the sensing experiments on silica nanobeads with different diameters. We set the laser-cavity detuning at the operational point indicated within the dashed circle of Fig. 2b and delivered silica nanobeads diluted in Dulbecco's phosphate-buffered saline (DPBS) around the microsphere. A particle binding would introduce a sudden change of the OMO frequency. One example is clearly shown in the video clip provided in Supplementary Movie 1. The particle binding events were recorded by searching for the sudden changes of the oscillation frequency in the recorded spectrograms. Typical examples are shown in Fig. 3a–d for the nanobeads with radii of 11.6, 25, 50 and 85 nm, respectively. Here in the case of 11.6-nm beads, the frequency steps were recorded at the third harmonic of the oscillation frequency while all others were obtained at the fundamental. Note the noisy spectrum of the 50-nm sensing experiment is due to the carousal trap effect reported in ref. 29, which is also evident from the video clip provided in Supplementary Movie 1. As shown in Fig. 3a, a clear step of $1.3 \pm 0.1$ kHz (corresponding to $0.43 \pm 0.03$ kHz step at the fundamental oscillation tone) was observed at the time of 58 s with an SNR of 13. An increase of the oscillation frequency implies a red shift of the cavity resonance wavelength, which corresponds to a binding of a 11.6-nm silica bead on the surface of the microsphere. Figure 3b–d show frequency steps of $-1.7\pm0.3$ , $3.5\pm0.9$ and $6.8\pm0.4$ kHz, respectively, which correspond to the binding (positive frequency steps) or unbinding (negative steps) events of 25, 50 and 85 nm beads. Note that the direct contribution of particle binding to the mechanical inertia of the OMO is negligible since the masses of the nanobeads ( $\sim0.01-5$ fg depending on particle radius) are more than nine orders of magnitude smaller than the effective motional mass of the OMO ( $\sim1\mu g$ ). The observed OMO frequency shifts induced by particle bindings are purely transduced from the optical spring effect.

To obtain the statistical properties of the binding events, we recorded a total number of 500,121, 521,389, 1,335,415 and 758,728 spectra in sensing 11.6, 25, 50 and $85\mathrm{nm}$ beads, respectively, among which frequency steps of 1,690, 2,043, 2,685 and 2,558 were captured with the SNR exceeding unity. Figure 3e-h show the histograms of the normalized frequency steps, $\delta f_{\mathrm{m}} / f_{\mathrm{m}}$ , which indicate maximum OMO frequency shifts of $\delta f_{\mathrm{m}} / f_{\mathrm{m}} = (1.4 \pm 0.4) \times 10^{-3}$ , $(-7.8 \pm 1.5) \times 10^{-3}$ , $(1.3 \pm 0.3) \times 10^{-2}$ and $(-2.3 \pm 0.6) \times 10^{-2}$ , respectively, for beads with radii of 11.6, 25, 50 and $85\mathrm{nm}$ . We converted the recorded OMO frequency steps into the corresponding cavity resonance wavelength shifts $\delta \lambda$ , with the transduction rate of $df_{\mathrm{m}} / d\Delta_{\lambda} = -1.5$ kHz fm $^{-1}$ . The probability density function of their absolute values are plotted as colour bars in Fig. 3i, with maximum wavelength shifts of $|\delta \lambda| / \lambda_0 = 2.6 \times 10^{-10}$ , $1.2 \times 10^{-9}$ , $2.4 \times 10^{-9}$ and $4.6 \times 10^{-9}$ , respectively, for the fourbead sizes (red circles). For comparison, we also numerically estimated the expected maximum wavelength change as a function of bead radius (green dashed line) $^{30}$ . The experimental results agree with the theoretical predictions when the bead size is small (11.6 and $25\mathrm{nm}$ ). At larger bead radii (50 and $85\mathrm{nm}$ ), the experimental values are smaller than the theoretical predictions because the optical Q starts to degrade at a large bead size. As an estimate, binding of a $85\mathrm{-nm}$ bead to the equator of a $100\text{-}\mu \mathrm{m}$ microsphere would degrade the optical Q by $2.7 \times 10^{4}$ . As the optical spring depends on both the laser-cavity detuning and the optical Q, the

![](images/2ace555b79447c79e6bbf3cafe9feba236e23ff23605e93242b4ed50731f9cae.jpg)

<details>
<summary>line</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.914           | 58       |
</details>

![](images/48edf4ef121896692e18a56168f9b099bc44183ec1630eb3d5da00bc4d40c5e9.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.29            | 0        |
| 0.295           | 1        |
| 0.3             | 2        |
| 0.305           | 3        |
| 0.31            | 4        |
</details>

![](images/9cbadcc4df2a302fc66abed27b3922030fd535e1501ec1aaf63d4fa8ee73ba66.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.26            | 52       |
| 0.255           | 53       |
| 0.27            | 54       |
| 0.275           | 55       |
| 0.28            | 56       |
</details>

![](images/75980bd4df02da092d1fb4a2c884e394112a70b3f156fdfd2a7e699b08b9da98.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.35            | 6.8 ± 0.4 |
</details>

![](images/e71fede07208f832bd1667a39e89b617a99b4ca6705915fbb45c0884c2426b43.jpg)

<details>
<summary>histogram</summary>

| δfₘ/fₘ (×10⁻³) | Counts |
| -------------- | ------ |
| -1.5           | 0      |
| -1.0           | 10     |
| -0.5           | 60     |
| 0.0            | 400    |
| 0.5            | 250    |
| 1.0            | 30     |
| 1.5            | 0      |
</details>

![](images/8dffb22ddafcebe62c06d4692d388d44ac9d705b7bf55275595f94adbdc24d3a.jpg)

<details>
<summary>histogram</summary>

| δf_m/f_m | Counts |
| --------- | ------ |
| -0.01     | 0      |
| -0.005    | 50     |
| 0         | 420    |
| 0.005     | 100    |
| 0.01      | 0      |
</details>

![](images/2294ca34f5ae0b501a65e7291716820e31a92d466a22e0166e791743aef4443f.jpg)

<details>
<summary>histogram</summary>

| δf_m/f_m | Counts |
|---|---|
| -0.015 | 10 |
| -0.01 | 20 |
| -0.005 | 90 |
| 0 | 450 |
| 0.005 | 330 |
| 0.01 | 70 |
| 0.015 | 10 |
</details>

![](images/1f653b527fe588c66b484df01224a3f65f504975e3952a89b7d4d8d96d597312.jpg)

<details>
<summary>histogram</summary>

| δf_m/f_m | Counts |
| -------- | ------ |
| -0.03    | 0      |
| -0.02    | 0      |
| -0.01    | 100    |
| 0        | 800    |
| 0.01     | 150    |
| 0.02     | 0      |
| 0.03     | 0      |
</details>

![](images/d7e165c36690dc6c6205b7b06d8d63dcde14be48e212482c44ca9b12099da251.jpg)  
Figure 3 | Detection of silica nanobeads. (a-d) Typical mechanical spectrograms for the binding events of silica beads with average radii of 11.6, 25, 50 and $85\mathrm{nm}$ , where a shows that of third harmonic and b-d show those of the fundamental oscillation frequency. The blue solid lines are the peak frequency traces computed by a least square fitting of each spectrum to a Lorentzian function. (e-h) The histograms of the normalized frequency steps $\delta f_{\mathrm{m}} / f_{\mathrm{m}}$ . (i) The corresponding cavity resonance shifts induced by the particle binding as a function of bead radius. The colour bars show the probability density functions of the recorded cavity resonance wavelength shifts induced by particle binding, where the bar width indicates the s.d. of the bead size (provided by the manufacturer) and the colour map indicates the magnitude of probability density. The red circles indicate the recorded maximum wavelength shifts of the cavity resonance with the s.d. of the difference between the measured data and the corresponding least square fitted step function represented by error bars. The dashed curve shows the theoretical prediction[30].

impact from cavity Q change counteracts that from the cavity resonance wavelength shift, leading to a smaller shift of OMO frequency (see Supplementary Note 1 for more details). Further discussions on nanobead results are detailed in Supplementary Notes 3 and 4.

Detection of single protein molecules. The ultrahigh detection sensitivity demonstrated on silica nanobeads readily implies the superior capability of sensing single protein molecules. To do so, we injected BSA diluted in DPBS around the microsphere sensor, with the concentration gradually increased from 0–100 nM and plotted the typical spectrograms in Fig. 4a–f. In the protein sensing experiments, the excessive noises from the unwanted molecules were significantly reduced, resulting in a detection noise level close to the DPBS background noise obtained from the Allan deviation measurement. The third order harmonic of the

![](images/7e06bcb1dddc79190b67690011e95662d182ea674193ed347d4246d51774b3f2.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.772           | 0        |
| 0.774           | 0        |
| 0.776           | 0        |
| 0.778           | 0        |
| 0.780           | 0        |
| 0.782           | 0        |
</details>

![](images/5eec1d36904b82b1033e85dfb671a0874f26fd846e1f0aca6f3cd590c2127743.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.788           | 0        |
| 0.79            | 2        |
| 0.792           | 8        |
| 0.794           | 2        |
| 0.796           | 0        |
</details>

![](images/8761ad0eef2066a88e82ed9b5ed0c28a3ed815a51340003a270f39a2dc230c75.jpg)

<details>
<summary>histogram</summary>

| δfₘ/fₘ (×10⁻³) | Counts |
| -------------- | ------ |
| -1.0 to -0.9   | 0      |
| -0.9 to -0.8   | 0      |
| -0.8 to -0.7   | 0      |
| -0.7 to -0.6   | 0      |
| -0.6 to -0.5   | 0      |
| -0.5 to -0.4   | 10     |
| -0.4 to -0.3   | 20     |
| -0.3 to -0.2   | 50     |
| -0.2 to -0.1   | 140    |
| -0.1 to 0.0    | 370    |
| 0.0 to 0.1     | 260    |
| 0.1 to 0.2     | 410    |
| 0.2 to 0.3     | 190    |
| 0.3 to 0.4     | 70     |
| 0.4 to 0.5     | 20     |
| 0.5 to 0.6     | 5      |
| 0.6 to 0.7     | 0      |
| 0.7 to 0.8     | 0      |
| 0.8 to 0.9     | 0      |
| 0.9 to 1.0     | 0      |
</details>

![](images/d3d61973856d94e1821117ba6559fb54e22a9956f574fb4e625de1ac888ad701.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.77            | 15       |
| 0.77            | 10       |
</details>

![](images/4ed08dfddbdcb5a913180ee2dc9b26cad54f4d32448d2efb59b2fca7e1e6f510.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) |
| --------------- | -------- |
| 0.89            | 32       |
| 0.895           | 34       |
| 0.9             | 36       |
| 0.89            | 38       |
| 0.895           | 40       |
| 0.9             | 42       |
</details>

![](images/9469738fc2c1f71a3b876aab6ba9dc61a9fe1662e525b2bfc76b784a32a60286.jpg)

<details>
<summary>histogram</summary>

| δf_m/f_m (×10⁻³) | Counts |
| ---------------- | ------ |
| -1.0 to -0.5     | 0      |
| -0.5 to 0.0      | 5      |
| 0.0 to 0.5       | 17     |
| 0.5 to 1.0       | 12     |
</details>

![](images/0aeaf3e418fa205f005d9c9f67f84e2a9015690e24c93bbd2806085a93f55376.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) | BSA (100 nM) |
| --------------- | -------- | ------------ |
| 0.77            | 35       |              |
| 0.772           | 35       |              |
| 0.774           | 35       |              |
| 0.776           | 35       |              |
| 0.778           | 35       |              |
| 0.78            | 35       |              |
</details>

![](images/bfa15c5b248de8ddf5d906561c01c2498d2cac4862860d72bca3f85844bb292a.jpg)

<details>
<summary>heatmap</summary>

| Frequency (MHz) | Time (s) | BSA (100 nM) |
| --------------- | -------- | ------------ |
| 0.72            | 6        | 0.51±0.09    |
| 0.72            | 8        | -0.3±0.1     |
</details>

Figure 4 | Detection of single protein molecules. (a–f) Typical mechanical spectrograms recorded at the third harmonic of the oscillation tone with different BSA concentrations. (a) Bare DPBS environment where protein molecules were absent. (b) BSA molecules with 1 nM nominal concentration were gradually injected. These two spectrograms show no binding event. (c,d) BSA molecules with 10 nM nominal concentration were injected. The spectrograms capture the event of a BSA protein molecule binding at an off-equator site at 9.4 s with a clear frequency step of $180 \pm 6$ Hz (c), and that of a molecule detaching from the silica microsphere surface at 38 s with a clear frequency step of $-670 \pm 40$ Hz (d). (e,f) BSA molecules with 100 nM nominal concentration were injected. The spectrograms further show the corresponding binding events with increased binding frequency as well as background noises. (g) The histogram of the normalized frequency steps of BSA binding events. (h) The histogram of the normalized frequency steps in the bare DPBS environment where BSA molecules were absent.

oscillation frequency was employed to monitor the frequency steps. As shown in Fig. 4d, a maximum frequency step of $-0.67 \pm 0.04$ kHz was observed with an SNR of 16.8, which corresponds to a step of $-0.22 \pm 0.01$ kHz at the fundamental oscillation tone. In total, we recorded 145,407 spectra with nominal concentration up to $10\mathrm{nM}$ among which 1,785 frequency steps were captured. The histogram of the normalized frequency steps is plotted in Fig. 4g, with the maximum step of $\delta f_{\mathrm{m}} / f_{\mathrm{m}} = (-7.6 \pm 0.4) \times 10^{-4}$ which corresponds to a cavity resonance shift of $|\delta \lambda| / \lambda_0 = 1.5 \times 10^{-10}$ . As a comparison, we also collected spectra by immersing the microsphere in bare DPBS without any protein molecules. As shown in Fig. 4a, the spectrogram indicates the OMO is stable in the absence of BSA molecules. Further, the histogram of all 118 steps found by our programme displayed in Fig. 4h shows the baseline noise is well below the step signal detected in BSA experiments. In addition, the detailed analysis of expected frequency step size and noise level (Supplementary Note 5) further confirms that the observed signals were protein binding induced. This observation clearly proves the capability of sensing single BSA molecules with a molecular weight of $66\mathrm{kDa}$ . By assuming the resonance shift is proportional to the mass (or equivalently, to the volume) of the protein[30], we derive that our current set-up is capable of detecting proteins as small as 3.9 kDa with an SNR above unity. A discussion of the influence of protein concentration in sensing experiments and binding time are presented in Supplementary Notes 5 and 6.

# Discussion

The demonstrated single-molecule detection now paves the foundation of ultra-sensitive cavity optomechanical spring sensing. The sensing resolution can be further improved significantly in the future. For example, the minimal detectable OMO frequency shift in current devices is primarily limited by the laser frequency jitter in our experiment. With an OMO linewidth of only $\sim0.1$ Hz, we expect that the future adoption of a fine laser frequency locking circuitry can further improve the sensing resolution by $\sim100$ times to around $\delta\lambda/\lambda_{0}\sim10^{-14}$ . On the other hand, the optical Q can be increased to above $10^{8}$ if a visible laser is employed $^{8}$ , which would further improve the sensing resolution by more than one order of magnitude. Moreover, a plasmonic structure can also be incorporated to enhance the cavity resonance shift. These future improvements would enable detecting small molecules and atoms with a mass down to sub-Dalton level in cryogenic environment, with great potential for dramatically advancing the capability of sensing to an unprecedented level.

In particular, as the molecule binding occurs during the coherent mechanical motion of the sensor, controlling the motion pattern of the coherent OMO (amplitude, phase, time waveform and so on) may function as a unique paradigm to study and control the mechanical properties of molecule binding and unbinding. This, in combination with certain functionalizations of the sensor surface $^{5}$ and with implementation of potentially versatile optomechanical motions $^{19}$ , may offer a unique multifunctional biomolecule toolbox that is not only able to observe cellular machineries at work, but also to selectively manipulate single-molecule interactions. Moreover, although we focus here on particle and molecule sensing, the demonstrated optical spring sensing principle can be applied for other physical sensing applications $^{31}$ , such as inertial sensing $^{32,33}$ , electromagnetic field sensing $^{34}$ , gas sensing $^{35}$ and so on, which are based on sensitive detection of optical cavity resonance shifts induced by external physical perturbations. Therefore, we expect the demonstrated optical spring sensing to be of great promise for broad applications beyond particle and molecule sensing itself.

# Methods

Device fabrication and characterization. To fabricate the device, we first heated a single-mode optical fibre with a hydrogen torch and pulled it to form a tapered tip which was then reflowed into a spherical shape by a $CO_{2}$ laser. The optical wave was coupled into and out of the device via a tapered optical fibre, as illustrated in Fig. 1a. The intrinsic optical Q of the device was measured on the silica microsphere immersed in DPBS (c.f. Fig. 6), with the laser wavelength calibrated by a reference interferometer. The input power was maintained low enough to prevent any optomechanical, thermo-optic or nonlinear effect. By fitting the cavity transmission spectrum with a Lorentzian function (Fig. 1c), we obtained an intrinsic optical Q of $4.8 \times 10^{6}$ . Its close agreement with the numerically predicted value of $6 \times 10^{6}$ indicates a nearly complete elimination of excessive contaminants that could have otherwise degraded the optical Q. This also verifies that the potential impact of dangling bonds of silica, which might produce fracture on the

![](images/c1e99b3def87edd5954294bdc6474565575687ba586da5a786dc8d8c4df195a5.jpg)

<details>
<summary>line</summary>

| Δλ' (pm) | Q0 (×10⁶) |
| -------- | --------- |
| 0        | 4.8       |
| 5        | 4.5       |
| 10       | 4.2       |
| 15       | 4.3       |
| 20       | 4.4       |
| 25       | 4.5       |
</details>

Figure 5 | Cavity Q degradation test. Main plot: the intrinsic quality factor $(Q_{0})$ versus time; left inset: $Q_{0}=4.8\times10^{6}$ when the silica microsphere was just immersed in DPBS buffer and right inset: $Q_{0}=4.5\times10^{6}$ after 25 h. The blue trace in each inset is the normalized transmission spectrum averaged over 100 traces and the red curve is the least square fit of the transmission curve to a Lorentzian function.

![](images/ca0605f69b46b5b8223c9c33fd3c482c97fe96afc4ac93bf27984f8ed6e555f8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Probe laser"] --> B["Waveform generator"]
    B --> C["Polarization controller 199"]
    C --> D["Polarization controller 99"]
    D --> E["Optical powermeter"]
    E --> F["Oscilloscope"]
    E --> G["Real time spectrum analyser"]
    G --> H["Photo detector 50"]
    H --> I["Photo detector 50"]
    I --> J["Electrical cable"]
    I --> K["Single mode fiber"]
```
</details>

Figure 6 | Experiment set-up. Ninety-nine per cent of the laser light is delivered to the microcavity through a tapered optical fibre. The light escaped from the cavity is coupled back to the same fibre and captured by a photodetector. The captured signal is then converted to electrical signals and analysed by a spectrum analyser and an oscilloscope. Meanwhile 1% light is delivered to an optical power metre for monitoring purposes.

microsphere to introduce additional loss, is negligible on optical Q in the regime $Q < 10^{7}$ .

To further verify the long-term stability of optical Q of the devices after they are immersed in DPBS buffer, we monitored the intrinsic Q fluctuation of a silica microsphere over a day. As shown in Fig. 5, the device exhibits a high optical Q of $4.8 \times 10^{6}$ right after the device was immersed in DPBS. After 25 h, the optical Q remains nearly intact, as high as $4.5 \times 10^{6}$ . This clearly verifies the long-term stability of optical Q in our devices. Note that our sensing experiments typically last no longer than 10 h, we are confident that the cavity degradation during the sensing experiments is negligible.

To excite the coherent OMO, we adjusted the spacing between the microsphere and the fibre taper to a position close to critical coupling and increased the input power to 8.5 mW. As shown in Fig. 1d, the coherent OMO started at a threshold optical power of 3.0 mW. To characterize the properties of coherent OMO, we set the wavelength of the continuous-wave laser for stable OMO and recorded the power spectrum of the signal transmitted from the cavity. The linewidth of the OMO was obtained by fitting the fundamental oscillation tone with a Lorentzian function, as shown in Fig. 1f.

Allan deviation measurement of OMO frequency. To obtain the background noise and to determine the minimum sensing resolution of our experiment, we measured the Allan deviation (Supplementary Note 7) of the fundamental, second and third order harmonics of OMO on the device immersed in pure DPBS in the absence of particle and recorded the power spectrum of cavity transmission seamlessly. As shown in Fig. 2c, the Allan deviation reaches a minimum within a measurement interval of 256 ms with a minimum two-sample deviation of 9.5 Hz at the fundamental oscillation. This value of Allan deviation includes all background noises in the experiments, thus corresponding to the minimum detectable shift of fundamental OMO frequency with an SNR of unity. It was also observed that the Allan deviations of higher order harmonics increase almost proportionally to that of the fundamental oscillation tone, indicating that the SNR would not be improved on the higher order harmonics. However, in practice, multiple particles that skim by the cavity surface within a single spectrum acquisition time contribute to an excessive detection noise. This noise can be reduced by sensing at higher order harmonics which facilitates considerably the mechanical spectral analysis (by allowing to use a coarser resolution bandwidth and thus a faster data acquisition interval).

Calibration of laser-cavity wavelength detuning. We set the laser wavelength at a off-resonance blue-detuned position and increased it step by step towards the cavity resonance, with a fixed input power. Both the time waveform and power spectrum of the transmitted signal were recorded at each set wavelength. The inset of Fig. 2b shows the averaged optical power injected into the cavity $P_{d}(\Delta_{\lambda}^{\prime})$ as a function of laser wavelength detuning $\Delta_{\lambda}^{\prime}$ , which reflects the thermo-optic bistability. The actual laser-cavity detuning, $\Delta_{\lambda} = \lambda_{l} - \lambda_{0}$ where $\lambda_{l}$ is the laser wavelength, is obtained by the following equation

$$
\Delta_ {\lambda} = - \frac {\lambda_ {0}}{2 Q _ {\mathrm{t}}} \sqrt {\frac {P _ {d} (0) - P _ {d} (\Delta_ {\lambda} ^ {\prime})}{P _ {d} (\Delta_ {\lambda} ^ {\prime})}}, \tag {2}
$$

where $P_{d}(0)$ is the optical power transferred to the cavity when the probe laser is on-resonance.

Experiment set-up. The experimental set-up is illustrated in Fig. 6. Here a Newport TLB 6700 external cavity tunable laser was used as a light source. The wavelength of the laser was tuned by a waveform generator (Agilent 33210A). In our experiment, the laser output was connected to a 99/1 optical directional coupler so that 1% of the light was connected to an optical power metre (Newport 1830c) to monitor the power of the laser output. The 99% output branch was connected to a polarization controller before entering a tapered fibre. The light was then delivered to the silica microsphere through this tapered fibre while the light escaped from the microsphere was coupled to the same taper and delivered to a photodetector before converting into an electrical signal. The electrical signal was divided into two equal output ports using an electrical power splitter where one output signal from the splitter was recorded by an oscilloscope (Agilent DSO 90404A) for time domain measurements and the other output was delivered to a real time spectrum analyser (Tektronix RSA 3408B) for spectral analysis.

Data availability. The data that support the findings of this study are available from the corresponding author on request.

# References

1. Piliarik, M. & Sandoghdar, V. Direct optical sensing of single unlabelled proteins and super-resolution imaging of their binding sites. Nat. Commun. 5, 4495 (2013).   
2. Hanay, M. S. et al. Single-protein nanomechanical mass spectrometry in real time. Nat. Nanotechnol. 7, 602–608 (2012).

3. Serpengüzel, A., Griffel, G. & Arnold, S. Excitation of resonances of microspheres on an optical fiber. Opt. Lett. 20, 654–656 (1995).   
4. Vollmer, F. et al. Protein detection by optical shift of a resonant microcavity. Appl. Phys. Lett. 80, 4057–4059 (2002).   
5. Vollmer, F. & Arnold, S. Whispering-gallery-mode biosensing: label-free detection down to single molecules. Nat. Methods 5, 591–596 (2008).   
6. Fan, X. et al. Sensitive optical biosensors for unlabeled targets: A review. Anal. Chim. Acta 620, 8–26 (2008).   
7. Boyd, R. W. & Heebner, J. E. Sensitive disk resonator photonic biosensor. Appl. Opt. 40, 5742–5747 (2001).   
8. Lu, T. et al. High sensitivity nanoparticle detection using optical microcavities. Proc. Natl Acad. Sci. USA 108, 5976–5979 (2011).   
9. Knittel, J., Swaim, J. D., McAuslan, D. L., Brawley, G. A. & Bowen, W. P. Back-scatter based whispering gallery mode sensing. Sci. Rep. 3, 2974 (2013).   
10. Avino, S. et al. Direct sensing in liquids using whispering-gallery-mode droplet resonators. Adv. Opt. Mater. 2, 1155–1159 (2014).   
11. Lu, T., Su, T.-T. J., Vahala, K. J. & Fraser, S. Split frequency sensing methods and systems. US Patent US 8,593,638 B2 (2013). Preliminary filing in October (2008).   
12. Zhu, J. et al. On-chip single nanoparticle detection and sizing by mode splitting in an ultrahigh-Q microresonator. Nat. Photon. 4, 46–49 (2010).   
13. White, I. M. & Fan, X. On the performance quantification of resonant refractive index sensors. Opt. Exp. 16, 1020-1028 (2008).   
14. Min, B. et al. High-Q surface plasmon polariton whispering gallery microcavity. Nature 457, 455–458 (2009).   
15. Shopova, S. I., Rajmangal, R., Holler, S. & Arnold, S. Plasmonic enhancement of a whispering-gallery-mode biosensor for single nanoparticle detection. Appl. Phys. Lett. 98, 243104 1-3 (2011).   
16. Dantham, V. R. et al. Label-free detection of single protein using a nanoplasmonic-photonic hybrid microcavity. Nano Lett. 13, 3347–3351 (2013).   
17. Baaske, M. D., Foreman, M. R. & Vollmer, F. Single-molecule nucleic acid interactions monitored on a label-free microcavity biosensor platform. Nat. Nanotechnol. 9, 933–939 (2014).   
18. Kippenberg, T. J. & Vahala, K. J. Cavity optomechanics: back-action at the mesoscale. Science 321, 1172–1176 (2008).   
19. Aspelmeyer, M., Kippenberg, T. J. & Marquardt, F. Cavity optomechanics. Rev. Mod. Phys. 86, 1391–1452 (2014).   
20. Waggoner, P. S. & Craighead, H. G. Micro- and nanomechanical sensors for environmental, chemical, and biological detection. Lab Chip 7, 1238–1255 (2007).   
21. Arlett, J. L., Myers, E. B. & Roukes, M. L. Comparative advantages of mechanical biosensors. Nat. Nanotechnol. 6, 203–215 (2011).   
22. Ramos, D., Mertens, J., Calleja, M. & Tamayo, J. Phototermal self-excitation of nanomechanical resonators in liquids. Appl. Phys. Lett. 92, 173108 1-3 (2008).   
23. Liu, F. & Hossein-Zadeh, M. Mass sensing with optomechanical oscillation. Sensors J. IEEE 13, 146–147 (2013).   
24. Kim, K. H. & Fan, X. Surface sensitive microfluidic optomechanical ring resonator sensors. Appl. Phys. Lett. 105, 191101 1-4 (2014).   
25. Liu, F., Alaie, S., Leseman, Z. C. & Hossein-Zadeh, M. Sub-pg mass sensing and measurement with an optomechanical oscillator. Opt. Exp. 21, 19555–19567 (2013).   
26. Yu, W., Jiang, W. C., Lin, Q. & Lu, T. Coherent optomechanical oscillation of a silica microsphere in an aqueous environment. Opt. Exp. 22, 21421-21426 (2014).   
27. Hossein-Zadeh, M., Rokhsari, H., Hajimiri, A. & Vahala, K. J. Characterization of a radiation-pressure-driven micromechanical oscillator. Phys. Rev. A 74, 023813 1-15 (2006).   
28. Xiong, C., Sun, X., Fong, K. Y. & Tang, H. X. Integrated high frequency aluminum nitride optomechanical resonators. Appl. Phys. Lett. 100, 171111 (2012).   
29. Arnold, S. et al. Whispering gallery mode carousel—a photonic mechanism for enhanced nanoparticle detection in biosensing. Opt. Exp. 17, 6230–6238 (2009).   
30. Arnold, S., Khoshsima, M., Teraoka, I., Holler, S. & Vollmer, F. Shift of whispering-gallery modes in microspheres by protein absorption. Opt. Lett. 28, 272–274 (2003).   
31. Foreman, M. R., Swaim, J. D. & Vollmer, F. Whispering gallery mode sensors. Adv. Opt. Photon. 7, 168–240 (2015).   
32. Ciminelli, C., Dell'Olio, F., Campanella, C. E. & Armenise, M. N. Photonic technologies for angular velocity sensing. Adv. Opt. Photon. 2, 370–404 (2010).   
33. Krause, A. G., Winger, M., Blasius, T. D., Lin, Q. & Painter, O. A high-resolution microchip optomechanical accelerometer. Nat. Photon. 6, 768–772 (2012).   
34. Ioppolo, T., Ayaz, U. & Ötügen, M. V. Tuning of whispering gallery modes of spherical resonators using an external electric field. Opt. Exp. 17, 16465-16479 (2009).

35. Yebo, N. A. et al. Selective and reversible ammonia gas detection with nanoporous film functionalized silicon photonic micro-ring resonator. Opt. Exp. 20, 11855–11862 (2012).

# Acknowledgements

The related contents are pending for patent application. W.Y. and T.L. acknowledge the support from NSERC Discovery and CFI LOF funds. W.C.J. and Q.L. acknowledge the support from DARPA under the QuASAR programme and from NSF under grant no. ECCS-1610674.

# Author contributions

W.Y. fabricated the devices, prepared the samples and performed the experiments. W.C.J. and Q.L. discovered the sensing principle and mechanism, developed the theory, and conducted the numerical simulation. T.L. conceived and designed the experiments, developed software to control the experiments and processed the data. Q.L. and T.L. conceived the concept and supervised the project. W.Y., W.C.J., Q.L. and T.L. worked together on the result analysis/interpretation and manuscript preparation.

# Additional information

Supplementary Information accompanies this paper at http://www.nature.com/naturecommunications

Competing financial interests: The authors declare no competing financial interests.

Reprints and permission information is available online at http://npg.nature.com/reprintsandpermissions/

How to cite this article: Yu, W. et al. Cavity optomechanical spring sensing of single molecules. Nat. Commun. 7:12311 doi: 10.1038/ncomms12311 (2016).

![](images/66fe371d766904e777fe012fe3677994716a2da51bc2afd82471479edd5579b1.jpg)

This work is licensed under a Creative Commons Attribution 4.0 International License. The images or other third party material in this led in the article's Creative Commons license, unless indicated otherwise; if the material is not included under the Creative Commons license, to obtain permission from the license holder to reproduce the material. of this license, visit http://creativecommons.org/licenses/by/4.0/

© The Author(s) 2016