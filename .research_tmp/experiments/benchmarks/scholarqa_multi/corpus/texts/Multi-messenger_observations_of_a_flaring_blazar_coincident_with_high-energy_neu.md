# Detection of a flaring blazar coincident with an IceCube high-energy neutrino

Anna Franckowiak $^{1,*}$ for the Fermi-LAT and IceCube Collaborations

$^{1}$ DESY, Zeuthen

Abstract. In September 22, 2017, IceCube released a public alert announcing the detection of a 290 TeV neutrino track event with an angular uncertainty of one square degree (90% containment). A multi-messenger follow-up campaign was initiated resulting in the detection of a GeV gamma-ray flare by the Fermi Large Area Telescope positionally consistent with the location of the known Bl Lac object, TXS 0506+056, located only 0.1 degrees from the best-fit neutrino position. The probability of finding a GeV gamma-ray flare in coincidence with a high-energy neutrino event assuming a correlation of the neutrino flux with the gamma-ray energy flux in the energy band between 1 and 100 GeV was calculated to be $3\sigma$ (after trials correction). Following the detection of the flaring blazar the imaging air Cherenkov telescope MAGIC detected the source for the first time in the >100 GeV gamma-ray band. The activity of the source was confirmed in X-ray, optical and radio wavelength. Several groups have developed lepto-hadronic models which succeed to explain the multi-messenger spectral energy distribution.

# 1 The IceCube Realtime Program

After the detection of an extragalactic diffuse high-energy neutrino flux in 2013 [1], the most pressing question in the young field of neutrino astronomy is where those neutrinos originate. To help answer this question IceCube has implemented a target of opportunity program (ToO), where neutrino track-events with high probability of being of cosmic origin are selected in realtime at the South Pole [2] and forwarded to telescopes for immediate electromagnetic follow-up observations. Two public neutrino streams exist: the high-energy starting event (HESE $^{1}$ ) and extremely high-energy (EHE $^{2}$ ) stream, which are distributed by the Astrophysical Multi-messenger Observatory Network (AMON) via the Gamma-ray Coordinates Network (GCN $^{3}$ ). Approximately four HESE and four EHE alerts are expected per year with an astrophysical purity of roughly 25% and 50% respectively.

# 2 IceCube-170922A

On September 22, 2017 the EHE event, IceCube-170922A, was detected by IceCube. The muon track event has a best fit neutrino energy of 290 TeV with a 90% lower limit of 183 TeV.

![](images/c672b1b82916a025d13d44a971c2d5dee93c979e28fa60ba6fc26ff73ff6d6f5.jpg)

<details>
<summary>line</summary>

| Time [MJD] | Flux [10⁻⁷ phcm⁻² s⁻¹] |
| ---------- | ------------------------ |
| 55000      | ~0.1                     |
| 55500      | ~0.3                     |
| 56000      | ~0.2                     |
| 56500      | ~0.4                     |
| 57000      | ~0.1                     |
| 57500      | ~0.3                     |
| 58000      | ~3.0                     |
| 58500      | ~2.0                     |
</details>

Figure 1. Adaptive binned gamma-ray light curve in the energy range from 300 MeV to 1 TeV. The vertical orange line shows the arrival time of IceCube-170922A and the orange band shows the duration of the lower-energy neutrino flare (see Finley et al. in these proceedings for more details). Figure adapted from [7].

The direction of the neutrino was reconstructed to a declination (Dec) of $5.72^{+0.50}_{-0.30}$ degree and right ascension (RA) of $77.43^{+0.95}_{-0.65}$ degree (J2000 equinox). The fraction of neutrino events with this energy and arrival direction in the EHE alert stream that have an astrophysical origin is 56.5% [3]. The distribution of the event information to the public initiated a large multi-wavelength observation campaign.

# 3 The Blazar TXS 0506+056

# 3.1 GeV Gamma-ray Observations

The Large Area gamma-ray Telescope (LAT) on board of the Fermi spacecraft observed the neutrino error circle in the energy range from 100 MeV to > 100 GeV since the start of the mission in July 2008 thanks to its large field of view and observing profile which scans the entire sky every 3 hours. A known gamma-ray blazar, TXS 0506+056, was identified only 0.1 degrees from the best-fit neutrino position and well within the 50% containment region (see Fig. 2, left panel).

The source is listed in the third Fermi-LAT source catalog $[3FGL, 4]$ as well as the third hard source catalog $[3FHL, 5]$ as 3FGL J0509.4+0541 and 3FHL J0509.4+0541, respectively. While the 3FGL catalog is based on gamma-ray data in the energy range of 100 MeV to 300 GeV, the 3FHL catalog is focused on energies above 10 GeV. Furthermore, TXS 0506+056 is also listed in the 2FHL $[6]$ catalog based on gamma-ray data above 50 GeV identifying it as a potential target for very-high-energy gamma-ray emission. TXS 0506+056 is among the brightest 4.4% (5.9%) sources in 3FGL (3FHL) in terms of gamma-ray energy flux within the energy bounds of the corresponding catalog $[7]$ .

Moreover, the source was identified in flaring state during the arrival time of IceCube-170922A. The gamma-ray activity started several months before the neutrino arrival and lasted several months after (see Fig. 1). Substructure within the several-months long flare was identified with week-scale variability [7].

# 3.2 Very-high-energy Gamma-ray Observations

Triggered by the neutrino alert the source was observed by several Imaging Air Cherenkov Telescopes (IACTs). MAGIC announced the first detection of TXS 0506+056 in >100 GeV

![](images/60662813b11bcb75eb4845fbeecb236e1d467c831e9120660f052c08bd7497c9.jpg)

<details>
<summary>scatter</summary>

| Right Ascension | Declination | Fermi-LAT Counts/Pixel |
| --------------- | ----------- | ---------------------- |
| 77.6            | 5.8         | 0.15                   |
| 77.2            | 5.8         | 0.15                   |
| 76.8            | 5.0         | 0.15                   |
| 76.4            | 4.6         | 0.15                   |
</details>

![](images/f611290aefaadfa01aea9781ab35e0d59791321b4c4c764c2559d3183f57d9a4.jpg)

<details>
<summary>heatmap</summary>

| Right Ascension | Declination | MAGIC significance [σ] |
| --------------- | ----------- | ------------------------ |
| 77.6°           | 5.8°        | ~3                       |
| 77.2°           | 5.4°        | ~1                       |
| 76.8°           | 5.0°        | ~-1                      |
| 76.4°           | 4.6°        | ~-2                      |
</details>

Figure 2. Gamma-ray sky maps: The 50% and 90% containment neutrino error is shown in gray and red respectively. Left: Fermi-LAT counts map. Right: MAGIC significance map. Figures from [3].

gamma rays [8, 9] (see Fig. 2, right panel), which was later confirmed by VERITAS [10]. Fast variability on daily time-scale was discovered by MAGIC indicating to a compact emission region [9].

# 3.3 X-ray, Optical and Radio Observations

The source was identified in active state also in X-rays by Swift X-Ray Telescope (XRT) and NuSTAR, in optical by ASAS-SN and Kanata and in radio by OVRO and VLT [3]. The redshift of TXS 0506+056 was measured to be z = 0.336 by [11] and was later confirmed to $z = 0.3365 \pm 0.0010$ by [12].

# 4 Chance Coincidence

How likely is a chance alignment of IceCube-170922A with a flaring blazar? To answer this question a Monte Carlo sample representative of the IceCube realtime EHE stream was generated. Random events were drawn and checked for spatial coincidences with cataloged extra-galactic Fermi sources. Once a spatial coincidence was identified the gamma-ray energy flux in the monthly bin overlapping with the neutrino arrival time was obtained. This gamma-ray energy flux was used as a weight in a likelihood ratio, which yields a test statistic (TS) for each generated trial. For the simple case of one neutrino event TS is given by

$$
T S = 2 \log \frac {\mathcal {S}}{\mathcal {B}}, \tag {1}
$$

with the signal PDF, which is a sum over all extragalactic Fermi sources (2257 sources were considered)

$$
\mathcal {S} (\vec {x}, t) = \sum_ {s} \frac {1}{2 \pi \sigma^ {2}} e ^ {- | \vec {x _ {s}} - \vec {x} | ^ {2} / (2 \sigma^ {2})} w _ {s} (t) w _ {\mathrm{acc}} (\theta_ {s}) \tag {2}
$$

and the background PDF

$$
\mathcal {B} (\vec {x}, t) = \frac {P _ {\mathrm{BG}} (\theta)}{2 \pi}. \tag {3}
$$

$\vec{x}$ is the reconstructed neutrino direction with uncertainty $\sigma$ and $\vec{x}_{s}$ is the location of source s. $w_{\mathrm{acc}}(\theta_{s})$ describes the acceptance of the detector as a function of the zenith angle $\theta$ and $w_{s}(t)$

is a model dependent weight, e.g. the gamma-ray energy flux of the source $s$ at the neutrino arrival time $t$ . $P_{\mathrm{BG}}(\theta)$ is the probability density function describing the zenith distribution of the alert events.

The $TS$ thus depends on the angular separation of the neutrino and the source position (taking in to account the angular uncertainty estimate of the neutrino direction), the gamma-ray energy flux in the corresponding monthly time bin, and the relative acceptance of the detector to neutrinos from the source direction. The measured $TS$ for IceCube-170922A and TXS 0506+056 was then compared to the background $TS$ distribution to calculate the p-value. A pre-trial p-value of $2.1 \times 10^{-5}$ was found. This value was corrected for trials to account for similar experiments conducted for the previous 9 IceCube realtime alerts and additional 41 archival events, which would have generated realtime alerts but were recorded before the IceCube ToO program was activated. After trials correction a p-value of $1.1 \times 10^{-3}$ was obtained corresponding to a Gaussian equivalent one-sided significance of $3.0\sigma$ . We tested three models predicting a connection of the gamma-ray activity with the neutrino flux. The first assumes that the neutrino flux is linearly correlated with the gamma-ray energy flux in the time bin of the neutrino arrival. The second one assumes a linear correlation of the neutrino flux with the ratio of gamma-ray flux in the corresponding time bin and the average gamma-ray flux. The third one assumes a correlation of the 100 GeV to 1 TeV gamma-ray energy flux with the neutrino flux. The latter one uses an extrapolation of the energy flux measured by Fermi to TeV energies. All three models yield similar results for the significance of the correlation of IceCube-170922A and TXS 0506+056.

# 5 Spectral Energy Distribution

Figure 3 shows the broad-band multi-wavelength spectral energy distribution (SED) of TXS 0506+056 during the detection of IceCube-170922A. It shows the characteristic two bump structure known for blazars. The first bump can be explained by synchrotron radiation of electrons, while the second bump can have its origin in a) leptonic processes (inverse Compton radiation of the same electrons that up-scatter either the synchrotron photons or external photons) or b) hadronic processes (decay of neutral pions or proton synchroton emission). While leptonic models do not predict neutrino emission, high-energy protons in hadronic models can interact with ambient photon fields or matter and produce pions. Charged pions produce neutrinos in their decay chain, while neutral pions decay to two gamma rays. High-energy gamma rays interact in the source or during propagation and cascade down to lower energies. Figure 3 shows representative $\nu_{\mu} + \bar{\nu}_{\mu}$ neutrino flux upper limits that produce on average one detection like IceCube-170922A over a period of 0.5 (solid black line) and 7.5 years (dashed black line). Note that the corresponding neutrino flux most likely overestimates the true flux due to the Eddington bias [13].

Several groups managed to find a self-consistent model explaining both the electromagnetic and the possible neutrino emission from TXS0506+056 in 2017. While simple single-zone hadronic models overshoot the observed X-ray flux [14, 15], lepto-hadronic models with a radiatively-subdominant hadronic component succeed to give a consistent picture [9, 14, 15]. More details can be found in Fedynitch et al. in these proceedings.

# 6 Summary and Conclusion

The 290 TeV neutrino event IceCube-170922A was found in spatial coincidence with the flaring gamma-ray blazar TXS 0506+056. The significance of such a correlation is $3\sigma$ assuming a linear relation of the gamma-ray activity with the neutrino production. This finding triggered a search for lower-energy $(O(10\mathrm{TeV}))$ archival neutrinos from the position of

![](images/0e129bf26cdde557172b39c2d058e53cfb59149ac3a6766d7e489b37d4e56ca9.jpg)

<details>
<summary>scatter</summary>

| Instrument | Energy [eV] | E² dN/dE [erg cm⁻² s⁻¹] |
| ---------- | ----------- | ------------------------ |
| Archival   | ~10⁻⁶ to 10⁰ | ~10⁻¹⁴ to 10⁻¹⁰        |
| VLA        | ~10⁻⁶ to 10⁰ | ~10⁻¹⁴ to 10⁻¹⁰        |
| OVRO       | ~10⁻⁶ to 10⁰ | ~10⁻¹⁴ to 10⁻¹⁰        |
| Kanata/HONIR | ~10⁻⁶ to 10⁰ | ~10⁻¹⁴ to 10⁻¹⁰        |
| Kiso/KWFC  | ~10⁻⁶ to 10⁰ | ~10⁻¹⁴ to 10⁻¹⁰        |
| SARAA/UA   | ~10⁰ to 10⁶ | ~10⁻¹³ to 10⁻¹¹        |
| Swift UVOT | ~10⁰ to 10⁶ | ~10⁻¹³ to 10⁻¹¹        |
| ASAS-SN    | ~10⁰ to 10⁶ | ~10⁻¹³ to 10⁻¹¹        |
| Swift XRT  | ~10⁰ to 10⁶ | ~10⁻¹³ to 10⁻¹¹        |
| NuSTAR     | ~10⁰ to 10⁶ | ~10⁻¹³ to 10⁻¹¹        |
| Fermi-LAT  | ~10⁹ to 26   | ~10⁻¹⁰ to 10⁻¹⁰        |
| AGILE      | ~10⁹ to 26   | ~10⁻¹⁰ to 10⁻¹⁰        |
| MAGIC      | ~10⁹ to 26   | ~10⁻¹⁰ to 10⁻¹⁰        |
| H.E.S.S. (UL) | ~10⁹ to 26   | ~10⁻¹⁰ to 10⁻¹⁰        |
| INTEGRAL (UL) | ~10⁵ to 22   | ~10⁻¹⁰ to 10⁻¹⁰        |
| VERITAS (UL) | ~10⁵ to 22   | ~10⁻¹⁰ to 10⁻¹⁰        |
| HAWC (UL)  | ~10⁵ to 26   | ~10⁻¹⁰ to 10⁻¹⁰        |
| Neutrino - 0.5yr | ~28    | ~10⁻¹²                 |
| Neutrino - 7.5yr | ~28    | ~10⁻¹²                 |
</details>

Figure 3. Spectral energy distribution based on observations obtained within 14 days of the detection of IceCube-170922A. $95\%$ differential flux upper limits are shown as colored bands and indicated as "UL" in the legend. Significant detections are shown by colored markers. Archival observations are shown in gray to illustrate the historical flux level of the blazar. Representative $\nu_{\mu} + \bar{\nu}_{\mu}$ neutrino flux upper limits that produce on average one detected event like IceCube-170922A over a period of 0.5 (solid black line) and 7.5 years (dashed black line) are shown assuming an $E^{-2}$ spectrum at the most probable neutrino energy. Figure from [3].

TXS 0506+056, which revealed an excess of neutrinos in a time window of 158 days between September 2014 and March 2015 with a $3.5\sigma$ significance [16] (more details in Finley et al. in these proceedings) without a coincident gamma-ray flare. Simple models so far cannot explain both IceCube-170922A and the 2014/15 neutrino flare in one consistent picture [17–19]. Detailed analyses of archival IceCube and Fermi data combined with more detections of high-energy neutrinos connected to blazars in the future might shed more light on particle acceleration in blazar jets.

# References

[1] M.G. Aartsen, R. Abbasi, Y. Abdouand et al. (IceCube), Science 342, 1242856 (2013), 1311.5238   
[2] M.G. Aartsen, M. Ackermann, J. Adams, J.A. Aguilar, M. Ahlers, M. Ahrens, D. Altmann, K. Andeen, T. Anderson, I. Ansseau et al., Astroparticle Physics 92, 30 (2017), 1612.06028   
[3] M.G. Aartsen, M. Ackermann, J. Adams et al., Science 361, eaat1378 (2018)   
[4] F. Acero, M. Ackermann, M. Ajello, A. Albert, W.B. Atwood, M. Axelsson, L. Baldini, J. Ballet, G. Barbiellini, D. Bastieri et al., ApJS 218, 23 (2015), 1501.02003

[5] M. Ajello, W.B. Atwood, L. Baldini, J. Ballet, G. Barbiellini, D. Bastieri, R. Bellazzini, E. Bissaldi, R.D. Blandford, E.D. Bloom et al., ApJS 232, 18 (2017), 1702.00664   
[6] M. Ackermann, M. Ajello, W.B. Atwood, L. Baldini, J. Ballet, G. Barbiellini, D. Bastieri, J. Becerra Gonzalez, R. Bellazzini, E. Bissaldi et al., ApJS 222, 5 (2016), 1508.04449   
[7] S. Garrappa, S. Buson, A. Franckowiak et al., in preparation (2019)   
[8] R. Mirzoyan, The Astronomer's Telegram 10817 (2017)   
[9] S. Ansoldi, L.A. Antonelli, C. Arcaro, D. Baack, A. Babić, B. Banerjee, P. Bangale, U. Barres de Almeida, J.A. Barrio, J. Becerra González et al., ApJl 863, L10 (2018), 1807.04300   
[10] A.U. Abeysekara et al. (VERITAS), Astrophys. J. 861, L20 (2018), 1807. 04607   
[11] M. Ajello, R.W. Romani, D. Gasparrini, M.S. Shaw, J. Bolmer, G. Cotter, J. Finke, J. Greiner, S.E. Healey, O. King et al., ApJ 780, 73 (2014), 1310.0006   
[12] S. Paiano, R. Falomo, A. Treves, R. Scarpa, ApJL 854, L32 (2018), 1802.01939   
[13] N.L. Strotjohann, M. Kowalski, A. Franckowiak (2018), 1809.06865   
[14] S. Gao, A. Fedynitch, W. Winter, M. Pohl, Nature Astronomy p. 154 (2018)   
[15] A. Keivani, K. Murase, M. Petropoulou, D.B. Fox, S.B. Cenko, S. Chaty, A. Coleiro, J.J. DeLaunay, S. Dimitrakoudis, P.A. Evans et al., ApJ 864, 84 (2018), 1807.04537   
[16] M.G. Aartsen, M. Ackermann, J. Adams et al. (IceCube), Science 361, 147 (2018)   
[17] A. Reimer, M. Boettcher, S. Buson, arXiv e-prints (2018), 1812.05654   
[18] X. Rodrigues, S. Gao, A. Fedynitch, A. Palladino, W. Winter (2018), 1812.05939   
[19] F. Halzen, A. Kheirandish, T. Weisgarber, S.P. Wakely, arXiv e-prints (2018), 1811.07439