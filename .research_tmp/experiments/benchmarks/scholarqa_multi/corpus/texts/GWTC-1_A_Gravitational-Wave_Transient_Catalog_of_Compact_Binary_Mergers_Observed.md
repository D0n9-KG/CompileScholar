# GWTC-1: A Gravitational-Wave Transient Catalog of Compact Binary Mergers Observed by LIGO and Virgo during the First and Second Observing Runs

B.P. Abbott et al.\* (LIGO Scientific Collaboration and Virgo Collaboration)

![](dt=2026-04-19/ht=09/303775706f46f208586560c9e30c16c34e0d531a2671cf453b5d3c71523a6a24.jpg)

(Received 14 December 2018; revised manuscript received 27 March 2019; published 4 September 2019)

We present the results from three gravitational-wave searches for coalescing compact binaries with component masses above $1M_{\odot}$ during the first and second observing runs of the advanced gravitational-wave detector network. During the first observing run $(O1)$ , from September 12, 2015 to January 19, 2016, gravitational waves from three binary black hole mergers were detected.

The second observing run $(O2)$ , which ran from November 30, 2016 to August 25, 2017, saw the first detection of gravitational waves from a binary neutron star inspiral, in addition to the observation of gravitational waves from a total of seven binary black hole mergers, four of which we report here for the first time: GW170729, GW170809, GW170818, and GW170823. For all significant gravitational-wave events, we provide estimates of the source properties.

The detected binary black holes have total masses between $18.6_{-0.7}^{+3.2}M_{\odot}$ and $84.4_{-11.1}^{+15.8}M_{\odot}$ and range in distance between $320_{-110}^{+120}$ and $2840_{-1360}^{+1400}\mathrm{Mpc}$ . No neutron star-black hole mergers were detected. In addition to highly significant gravitational-wave events, we also provide a list of marginal event candidates with an estimated false-alarm rate less than 1 per 30 days.

From these results over the first two observing runs, which include approximately one gravitational-wave detection per 15 days of data searched, we infer merger rates at the $90\%$ confidence intervals of $110 - 3840\mathrm{Gpc^{-3}y^{-1}}$ for binary neutron stars and $9.7 - 101\mathrm{Gpc^{-3}y^{-1}}$ for binary black holes assuming fixed population distributions and determine a neutron star-black hole merger rate $90\%$ upper limit of $610\mathrm{Gpc^{-3}y^{-1}}$ .

DOI: 10.1103/PhysRevX.9.031040

Subject Areas: Astrophysics, Gravitation

# I. INTRODUCTION

The first observing run (O1) of Advanced LIGO, which took place from September 12, 2015 until January 19, 2016, saw the first detections of gravitational waves (GWs) from stellar-mass binary black holes (BBHs) [1-4]. After an upgrade and commissioning period, the second observing run (O2) of the Advanced LIGO detectors [5] commenced on November 30, 2016 and ended on August 25, 2017. On August 1, 2017, the Advanced Virgo detector [6] joined the observing run, enabling the first three-detector observations of GWs.

This network of ground-based interferometric detectors is sensitive to GWs from the inspiral, merger, and ringdown of compact binary coalescences (CBCs), covering a frequency range from about $15\mathrm{Hz}$ up to a few kilohertz (see Fig. 1). In this catalog, we report 11 confident detections of GWs from compact binary mergers as well as a selection of less significant triggers from both observing runs. The observations reported here and future GW detections will shed light on binary

formation channels, enable precision tests of general relativity (GR) in its strong-field regime, and open up new avenues of astronomy research.

The events presented here are obtained from a total of three searches: two matched-filter searches, PyCBC [7,8] and GSTLAL [9,10], using relativistic models of GWs from CBCs, as well as one unmodeled search for short-duration transient signals or bursts, coherent WaveBurst (cWB) [11].

The two matched-filter searches target GWs from compact binaries with a redshifted total mass $M(1 + z)$ of $2 - 500M_{\odot}$ for PyCBC and $2 - 400M_{\odot}$ for GSTLAL, where $z$ is the cosmological redshift of the source binary [12], and with maximal dimensionless spins of 0.998 for black holes (BHs) and 0.05 for neutron stars (NSs). The results of a matched-filter search for sub-solar-mass compact objects in $O1$ can be found in Ref. [13]; the results for $O2$ will be discussed elsewhere.

The burst search cWB does not use waveform models to compare against the data but instead identifies regions of excess power in the time-frequency representation of the gravitational strain. We report results from a cWB analysis that is optimized for the detection of compact binaries with a total mass less than $100M_{\odot}$ . A different tuning of the cWB analysis is used for a search for intermediate-mass BBHs with total masses greater than $100M_{\odot}$ ; the results of that analysis are discussed elsewhere.

The three searches reported here use different methodologies to identify GWs from compact binaries

PHYSICAL REVIEW X 9, 031040 (2019)

*Full author list given at the end of the article.

Published by the American Physical Society under the terms of the Creative Commons Attribution 4.0 International license. Further distribution of this work must maintain attribution to the author(s) and the published article's title, journal citation, and DOI.

2160-3308/19/9(3)/031040(49)

031040-1

Published by the American Physical Society

![](dt=2026-04-19/ht=09/59f4652fb5a53b8fdb23b68e42210575e250be3fb82ce701bfb9a6dccb313b68.jpg)

![](dt=2026-04-19/ht=09/cd14784200d08d210c62b03dfb75877c45bd7619fd76b9ecee828876746874a0.jpg)

in an overlapping but not identical search space, thus providing three largely independent analyses that allow for important cross-checks and yield consistent results. All searches have undergone improvements since $O1$ , making it scientifically valuable to reanalyze the $O1$ data in order to reevaluate the significance of previously identified GW events and to potentially discover new ones.

The searches identified a total of ten BBH mergers and one binary neutron star (BNS) signal. The GW events GW150914, GW151012 [14], GW151226, GW170104, GW170608, GW170814, and GW170817 have been reported previously [4,15-18]. In this catalog, we announce four previously unpublished BBH mergers observed during $O2$ : GW170729, GW170809, GW170818, and GW170823. We estimate the total mass of GW170729 to be $84.4_{-11.1}^{+15.8}M_{\odot}$ , making it the highest-mass BBH observed to date.

GW170818 is the second BBH observed in triple coincidence between the two LIGO observatories and Virgo after GW170814 [16]. As the sky location is primarily determined by the differences in the times of arrival of the GW signal at the different detector sites, LIGO-Virgo coincident events have a vastly improved sky localization, which is crucial for electromagnetic follow-up campaigns [19-22]. The reanalysis of the $O1$ data did not result in the discovery of any new GW events, but GW151012 is now detected with increased significance.

In addition, we list 14 GW candidate events that have an estimated false-alarm rate (FAR) less than 1 per 30 days in either of the two matched-filter analyses but whose astrophysical origin cannot be established nor excluded unambiguously (Sec. VII).

Gravitational waves from compact binaries carry information about the properties of the source such as the masses and spins. These can be extracted via Bayesian inference by using theoretical models of the GW signal that describe the inspiral, merger, and ringdown of the final object for BBH

[23-30] and the inspiral (and merger) for BNS [31-33]. Such models are built by combining post-Newtonian calculations [34-38], the effective-one-body formalism [39-44], and numerical relativity [45-50]. Based on a variety of theoretical models, we provide key source properties of all confident GW detections. For p
reviously reported detections, we provide updated parameter estimates which exploit refined instrumental calibration, noise subtraction (for $O2$ data) [51,52], and updated amplitude power spectral density estimates [53,54].

The observation of these GW events allows us to place constraints on the rates of stellar-mass BBH and BNS mergers in the Universe and probe their mass and spin distributions, putting them into astrophysical context. The nonobservation of GWs from a neutron star-black hole binary (NSBH) yields a stronger $90\%$ upper limit on the rate. The details of the astrophysical implications of our observations are discussed in Ref. [55].

This paper is organized as follows: In Sec. II, we provide an overview of the operating detectors during $O2$ , as well as the data used in the searches and parameter estimation. Section III briefly summarizes the three different searches, before we define the event selection criteria and present the results in Sec. IV. Tables I and II summarize some key search parameters for the clear GW detections and the marginal events. Details about the source properties of the GW events are given in Sec.

V, and the values of some important parameters obtained from Bayesian inference are listed in Table III. We do not provide parameter estimation results for marginal events. An independent consistency analysis between the waveform-based results and the data is performed in Sec. VI. In Sec. VII, we describe how the probability of astrophysical origin is calculated and give its value for each significant and marginal event in Table IV. We provide an updated estimate of binary merger rates in this

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-2

TABLE I. Search results for the 11 GW events. We report a false-alarm rate for each search that found a given event; otherwise, we display $\cdots$ . The network SNR for the two matched-filter searches is that of the template ranked highest by that search, which is not necessarily the template with the highest SNR. Moreover, the network SNR is the quadrature sum of the detectors coincident in the highest-ranked trigger; in some cases, only two detectors contribute, even if all three are operating nominally at the time of that event.

![](dt=2026-04-19/ht=09/0e81e75204475ab376cb0a240a37783428e1a38a56f18e9274bb8b18dd5c669c.jpg)

<table><tr><td rowspan="2">Event</td><td rowspan="2">UTC time</td><td colspan="3">FAR [y-1]</td><td colspan="3">Network SNR</td></tr><tr><td>PyCBC</td><td>GstLAL</td><td>cWB</td><td>PyCBC</td><td>GstLAL</td><td>cWB</td></tr><tr><td>GW150914</td><td>09:50:45.4</td><td>&lt;1.53 × 10-5</td><td>&lt;1.00 × 10-7</td><td>&lt;1.63 × 10-4</td><td>23.6</td><td>24.4</td><td>25.2</td></tr><tr><td>GW151012</td><td>09:54:43.4</td><td>0.17</td><td>7.92 × 10-3</td><td>...</td><td>9.5</td><td>10.0</td><td>...</td></tr><tr><td>GW151226</td><td>03:38:53.6</td><td>&lt;1.69 × 10-5</td><td>&lt;1.00 × 10-7</td><td>0.02</td><td>13.1</td><td>13.1</td><td>11.9</td></tr><tr><td>GW170104</td><td>10:11:58.6</td><td>&lt;1.37 × 10-5</td><td>&lt;1.00 × 10-7</td><td>2.91 × 10-4</td><td>13.0</td><td>13.0</td><td>13.0</td></tr><tr><td>GW170608</td><td>02:01:16.5</td><td>&lt;3.09 × 10-4</td><td>&lt;1.00 × 10-7</td><td>1.44 × 10-4</td><td>15.4</td><td>14.9</td><td>14.1</td></tr><tr><td>GW170729</td><td>18:56:29.3</td><td>1.36</td><td>0.18</td><td>0.02</td><td>9.8</td><td>10.8</td><td>10.2</td></tr><tr><td>GW170809</td><td>08:28:21.8</td><td>1.45 × 10-4</td><td>&lt;1.00 × 10-7</td><td>...</td><td>12.2</td><td>12.4</td><td>...</td></tr><tr><td>GW170814</td><td>10:30:43.5</td><td>&lt;1.25 × 10-5</td><td>&lt;1.00 × 10-7</td><td>&lt;2.08 × 10-4</td><td>16.3</td><td>15.9</td><td>17.2</td></tr><tr><td>GW170817</td><td>12:41:04.4</td><td>&lt;1.25 × 10-5</td><td>&lt;1.00 × 10-7</td><td>...</td><td>30.9</td><td>33.0</td><td>...</td></tr><tr><td>GW170818</td><td>02:25:09.1</td><td>...</td><td>4.20 × 10-5</td><td>...</td><td>...</td><td>11.3</td><td>...</td></tr><tr><td>GW170823</td><td>13:13:58.5</td><td>&lt;3.29 × 10-5</td><td>&lt;1.00 × 10-7</td><td>2.14 × 10-3</td><td>11.1</td><td>11.5</td><td>10.8</td></tr></table>

TABLE II. Marginal triggers from the two matched-filter CBC searches. To distinguish events occurring on the same UTC day, we extend the YYMMDD label by decimal fractions of a day as needed, always rounding down (truncating) the decimal. The search that identifies each trigger is given, and the false alarm and network SNR. This network SNR is the quadrature sum of the individual detector SNRs for all detectors involved in the reported trigger; that can be fewer than the number of nominally operational detectors at the time, depending on the ranking algorithm of each pipeline. The detector chirp mass reported is that of the most significant template of the search. The concentration of our marginal triggers at low chirp masses is consistent with expectations for noise triggers, because search template waveforms are much more densely packed at low masses. The final column indicates whether there are any detector characterization concerns with the trigger; for an explanation and more details, see the text.

![](dt=2026-04-19/ht=09/eee1538586ba3297a36afc4b6b978af9efa362d4aa3cf83f5cc657b111c8f1e9.jpg)

<table><tr><td>Date</td><td>UTC</td><td>Search</td><td>FAR [y-1]</td><td>Network SNR</td><td>Mdet[M☉]</td><td>Data quality</td></tr><tr><td>151008</td><td>14:09:17.5</td><td>PyCBC</td><td>10.17</td><td>8.8</td><td>5.12</td><td>No artifacts</td></tr><tr><td>151012.2</td><td>06:30:45.2</td><td>GstLAL</td><td>8.56</td><td>9.6</td><td>2.01</td><td>Artifacts present</td></tr><tr><td>151116</td><td>22:41:48.7</td><td>PyCBC</td><td>4.77</td><td>9.0</td><td>1.24</td><td>No artifacts</td></tr><tr><td>161202</td><td>03:53:44.9</td><td>GstLAL</td><td>6.00</td><td>10.5</td><td>1.54</td><td>Artifacts possibly caused</td></tr><tr><td>161217</td><td>07:16:24.4</td><td>GstLAL</td><td>10.12</td><td>10.7</td><td>7.86</td><td>Artifacts possibly caused</td></tr><tr><td>170208</td><td>10:39:25.8</td><td>GstLAL</td><td>11.18</td><td>10.0</td><td>7.39</td><td>Artifacts present</td></tr><tr><td>170219</td><td>14:04:09.0</td><td>GstLAL</td><td>6.26</td><td>9.6</td><td>1.53</td><td>No artifacts</td></tr><tr><td>170405</td><td>11:04:52.7</td><td>GstLAL</td><td>4.55</td><td>9.3</td><td>1.44</td><td>Artifacts present</td></tr><tr><td>170412</td><td>15:56:39.0</td><td>GstLAL</td><td>8.22</td><td>9.7</td><td>4.36</td><td>Artifacts possibly caused</td></tr><tr><td>170423</td><td>12:10:45.0</td><td>GstLAL</td><td>6.47</td><td>8.9</td><td>1.17</td><td>No artifacts</td></tr><tr><td>170616</td><td>19:47:20.8</td><td>PyCBC</td><td>1.94</td><td>9.1</td><td>2.75</td><td>Artifacts present</td></tr><tr><td>170630</td><td>16:17:07.8</td><td>GstLAL</td><td>10.46</td><td>9.7</td><td>0.90</td><td>Artifacts present</td></tr><tr><td>170705</td><td>08:45:16.3</td><td>GstLAL</td><td>10.97</td><td>9.3</td><td>3.40</td><td>No artifacts</td></tr><tr><td>170720</td><td>22:44:31.8</td><td>GstLAL</td><td>10.75</td><td>13.0</td><td>5.96</td><td>Artifacts possibly caused</td></tr></table>

section before concluding in Sec. VIII. We also provide the Appendixes containing additional technical details.

A variety of additional information on each event, data products including strain data and posterior samples, and postprocessing tools can be obtained from the accompanying data release [56] hosted by the Gravitational Wave Open Science Center [57].

# II. INSTRUMENTAL OVERVIEW AND DATA

# A. LIGO instruments

The Advanced LIGO detectors [58,59] began scientific operations in September, 2015 and almost immediately

detected the first gravitational waves from the BBH merger GW150914.

Between $O1$ and $O2$ , improvements were made to both LIGO instruments. At LIGO-Livingston (L
LO), a malfunctioning temperature sensor [60] was replaced immediately after $O1$ , contributing to an increase in the BNS range from approximately 60 Mpc to approximately 80 Mpc [61]. Other major changes included adding passive tuned mass dampers on the end test mass suspensions to reduce ringing up of mechanical modes, installing a new output Faraday isolator, adding a new in-vacuum array of photodiodes for stabilizing the laser intensity, installing

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-3

higher quantum-efficiency photodiodes at the output port, and replacing the compensation plate on the input test mass suspension for the $Y$ arm. An attempt to upgrade the LLO laser to provide higher input power was not successful. During $O2$ , improvements to the detector sensitivity continued, and sources of scattered light noise were mitigated. As a result, the sensitivity of the LLO instrument rose from a BNS range of $80\mathrm{Mpc}$ at the beginning of $O2$ to greater than $100\mathrm{Mpc}$ by the run's end.

The LIGO-Hanford (LHO) detector had a range of approximately $80\mathrm{Mpc}$ as $O1$ ended, and it was decided to concentrate on increasing the input laser power and forgo any incursions into the vacuum system. Increasing the input laser power to $50\mathrm{W}$ was successful, but since this increase did not result in an improvement in sensitivity, the LHO detector operated with $30\mathrm{W}$ input power during $O2$ .

It was eventually discovered that there was a point absorber on one of the input test mass optics, which we speculate led to increased coupling of input "jitter" noise from the laser table into the interferometer. By use of appropriate witness sensors, it was possible to perform an offline noise subtraction on the data, leading to an increase in the BNS range at LHO by an average of about $20\%$ over all of $O2$ [51,52].

On July 6, 2017, LHO was severely affected by a 5.8 magnitude earthquake in Montana. Postearthquake, the sensitivity of the detector dropped by approximately $10\mathrm{Mpc}$ and remained in this condition until the end of the run on August 25, 2017.

# B. Virgo instrument

Advanced Virgo [6] aims to increase the sensitivity of the Virgo interferometer by one order of magnitude, and several upgrades were performed after the decommissioning of the first-generation detector in 2011. The main modifications include a new optical design, heavier mirrors, and suspended optical benches, including photodiodes in a vacuum. Special care was also taken to improve the decoupling of the instrument from environmental disturbances. One of the main limiting noise sources below $100\mathrm{Hz}$ is the thermal Brownian excitation of the wires used for suspending the mirrors.

A first test performed on the Virgo configuration showed that silica fibers would reduce this contribution. A vacuum contamination issue, which has since been corrected, led to failures of these silica suspension fibers, so metal wires were used to avoid delaying Virgo's participation in $O2$ . Unlike the LIGO instruments, Virgo has not yet implemented signal recycling, which will be installed in a later upgrade of the instrument.

After several months of commissioning, Virgo joined $O2$ on August 1, 2017 with a BNS range of approximately 25 Mpc. The performance experienced a temporary degradation on August 11 and 12, when the microseismic activity on site was highly elevated and it was difficult to keep the interferometer in its low-noise operating mode.

# C. Data

Figure 1 shows the BNS ranges of the LIGO and Virgo instruments over the course of $O2$ and the representative amplitude spectral density plots of the total strain noise for each detector.

We subtract several independent contributions to the instrumental noise from the data at both LIGO detectors [51]. For all of $O2$ , the average increase in the BNS range from this noise subtraction process at LHO is approximately $20\%$ [51]. At LLO, the noise-subtraction process targeted narrow line features, resulting in a negligible increase in the BNS range.

Calibrated strain data from each interferometer are produced online for use in low-latency searches. Following the run, a final frequency-dependent calibration is generated for each interferometer.

For the LIGO instruments, this final calibration benefits from the use of postrun measurements and the removal of instrumental lines. The calibration uncertainties are $3.8\%$ in amplitude and $2.1^{\circ}$ in phase for LLO and $2.6\%$ in amplitude and $2.4^{\circ}$ in phase for LHO. The results cited in this paper use the full frequency-dependent calibration uncertainties described in Refs. [64,65]. The LIGO timing uncertainty of $< 1~\mu \mathrm{s}$ [66] is included in the phase correction factor.

The calibration of strain data produced online by Virgo has large uncertainties due to the short time available for measurements. The data are reprocessed to reduce the errors by taking into account better calibration models obtained from postrun measurements and the subtraction of frequency noise. The reprocessing includes a time dependence for the noise subtraction and for the determination of the finesse of the cavities. The final uncertainties are $5.1\%$ in amplitude and $2.3^{\circ}$ in phase [67]. The Virgo calibration has an additional uncertainty of $20~\mu \mathrm{s}$ originating from the time stamping of the data.

During $O2$ , the individual LIGO detectors had duty factors of approximately $60\%$ with a LIGO network duty factor of about $45\%$ . Times with significant instrumental disturbances are flagged and removed, resulting in about 118 days of data suitable for coincident analysis [68]. Of these data, about 15 days are collected in coincident operation with Virgo, which after joining $O2$ operated with a duty factor of about $80\%$ . Times with excess instrumental noise, which is not expected to render the data unusable, are also flagged [68]. Individual searches may then decide to include or not include such times in their final results.

# III. SEARCHES

The search results presented in the next section are obtained by two different, largely independent matched-filter searches, PyCBC and GSTLAL, and the burst search cWB. Because of the sensitivity imbalance between the Advanced Virgo detector as compared to the two Advanced LIGO detectors, neither PyCBC nor cWB elect to analyze

B.P.ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-4

data from Virgo. GstLAL, however, includes Virgo into its search during the month of August. The two matched-filter searches assume sources that can be modeled by general relativity and, in particular, quasicircular binaries whose spin angular momenta are either aligned or antialigned with their orbital angular momenta. They are still capable, however, of detecting many systems that exhibit precession [69].

In contrast, the cWB search relies on no specific physical models of the source waveform, though in results presented here it does impose a restriction that signals are "chirping" in the time-frequency plane. We therefore refer to it as weakly modeled. In the remainder of this section, we present a brief description of each of these searches, summarizing both the parameter space searched and improvements made since their use in $O1$ [4].

# A. The PyCBC search

A pipeline to search for GWs from CBCs is constructed using the PyCBC software package [7,8]. This analysis performs direct matched filtering of the data against a bank of template waveforms to calculate the signal-to-noise ratio (SNR) for each combination of detector, template waveform, and coalescence time [70]. Whenever the local maximum of this SNR time series is larger than a threshold of 5.5, the pipeline produces a single-detector trigger associated with the detector, the parameters of the template, and the coalescence time.

In order to suppress triggers caused by high-amplitude noise tr
ansients ("glitches"), two signal-based vetoes may be calculated [71,72]. Using the SNR, the results of these two vetoes, and a fitting and smoothing procedure designed to ensure that the rate of single-detector triggers is approximately constant across the search parameter space, a single-detector rank $\varrho$ is calculated for each single-detector trigger [73].

After generating triggers in the Hanford and Livingston detectors as described above, PyCBC finds two-detector coincidences by requiring a trigger from each detector associated with the same template and with coalescence times within $15\mathrm{ms}$ of each other. This time window accounts for the maximum light-travel time between LHO and LLO as well as the uncertainty in the inferred coalescence time at each detector.

Coincident triggers are assigned a ranking statistic that approximates the relative likelihood of obtaining the event's measured trigger parameters in the presence of a GW signal versus in the presence of noise alone [73]. The detailed construction of this network statistic, as well as the single-detector rank $\varrho$ , is improved from the corresponding statistics used in $O1$ , partially motivating the reanalysis of $O1$ by this pipeline.

Finally, the statistical significances of coincident triggers are quantified by their inverse false-alarm rate (IFAR). This rate is estimated by applying the same coincidence procedure after repeatedly time shifting the triggers from one detector and using the resulting coincidences as a background sample. Each foreground coincident trigger is

assigned a false-alarm rate (FAR) given by the number of background triggers with an equal or larger ranking, divided by the total time searched for time-shifted coincidences. For an event with a given IFAR observed in the data of duration $T$ , the probability of obtaining one or more equally highly ranked events due to noise is

$$
p = 1 - e ^ {- T / \text {I F A R}}. \tag {1}
$$

In the analysis of this paper, the data are divided into analysis periods that allow at least 5.2 days of coincident data between the two LIGO detectors [74]. Though previous publications performed time shifting across larger amounts of time [1,2,4], the results here consider only time shifts within a given analysis period, which is done because the noise characteristics of the detector vary significantly from the beginning of $O1$ through the end of $O2$ , so this restriction more accurately reflects the variation in detector performance.

This restriction means, however, that the minimum bound on the false-alarm rate of candidates that have a higher ranking statistic than any trigger in the background sample is larger than it would be if longer periods of data are used for the time-shift analysis.

For the PyCBC analysis presented here, the template bank described in Ref. [75] is used. This bank covers binary systems with a total mass between 2 and $500M_{\odot}$ and mass ratios down to $1 / 98$ . Components of the binary with a mass below $2M_{\odot}$ are assumed to be neutron stars and have a maximum spin magnitude of 0.05; otherwise, the maximum magnitude is 0.998. The high-mass boundary of the search space is determined by the requirement that the waveform duration be at least $0.15\mathrm{~s~}$ , which reduces the number of false-alarm triggers from short instrumental glitches. The waveform models used are a reduced-order-model (ROM) [29,76-78] of SEOBNRv4 [29] for systems with a total mass greater than $4M_{\odot}$ and TaylorF2 [38,80] otherwise.

# B. The GSTLAL search

A largely independent matched-filter pipeline based on the GSTLAL library [9,10] (henceforth GSTLAL) also performs a matched-filter search for CBC signals. GSTLAL produces triggers for each template waveform and each detector by maximizing the matched-filter SNR $\rho$ over one-second windows and requiring that it exceed a threshold of 4 for the two LIGO detectors and 3.5 for Virgo. The relatively lower Virgo SNR threshold is an ad hoc choice designed to improve the network sensitivity of the search given Virgo's smaller horizon distance.

For the search described here, candidates are formed by requiring a temporal coincidence between triggers from the same template but from different detectors, with the coincidence window set by the light-travel time between detectors plus 5 ms [81]. GSTLAL ranks candidates using the logarithm of the likelihood ratio, $\mathcal{L}$ , a measure of how likely it is to observe that candidate if a signal is present compared to if

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-5

only noise is present [9,82,83]. The noise model is constructed, in part, from single-detector triggers that are not found in coincidence, so as to minimize the possibility of contamination by real signals. In the search presented here, the likelihood ratio is a function of $\rho$ , a signal-consistency test, the differences in time and phase between the coincident triggers, the detectors that contribute triggers to the candidate, the sensitivity of the detectors to signals at the time of the candidate, and the rate of triggers in each of the detectors at the time of the candidate [9]. This function is an expansion of the parameters used to model the likelihood ratio in earlier versions of GstLAL and improves the sensitivity of the pipeline used for this search over that used in $O1$ .

The GSTLAL search uses Monte Carlo methods and the likelihood ratio's noise model to determine the probability of observing a candidate with a log likelihood ratio greater than or equal to $\log \mathcal{L}$ , $P(\log \mathcal{L}^* \geq \log \mathcal{L}|\text{noise})$ . The expected number of candidates from noise with log likelihood ratios at least as high as $\log \mathcal{L}$ is then $NP(\log \mathcal{L}^* \geq \log \mathcal{L}|\text{noise})$ , where $N$ is the number of observed candidates. The FAR is then the total number of expected candidates from noise divided by the live time of the experiment, $T$ , and the $p$ value is obtained by assuming the noise is a Poisson process:

$$
\operatorname {F A R} = \frac {N P (\log \mathcal {L} ^ {*} \geq \log \mathcal {L} | \text {n o i s e})}{T}, \tag {2}
$$

$$
p = 1 - e ^ {- N P \left(\log \mathcal {L} ^ {*} \geq \log \mathcal {L} | \text {n o i s e}\right)}. \tag {3}
$$

For the analysis in this paper, GSTLAL analyzes the same periods of data as PyCBC. However, FARs are assigned using the distribution of likelihood ratios in noise computed from marginalizing $P(\log \mathcal{L}^* \geq \log \mathcal{L} | \text{noise}, \text{period})$ over all analysis periods; thus, all of $O1$ and $O2$ are used to inform the noise model for FAR assignment. The only exception is GW170608. The analysis period used to estimate the significance of GW170608 is unique from the other ones [17], and thus its FAR is assigned using only its local background statistics.

For this search, GstLAL uses a bank of templates with a total mass between 2 and $400M_{\odot}$ and a mass ratio between 1/98 and 1. Components with a mass less than $2M_{\odot}$ have a maximum spin magnitude of 0.05 (as for PyCBC); otherwise, the spin magnitude is less than 0.999. The TaylorF2 waveform approximant is used to generate templates for systems with a chirp mass [see Eq. (5)] less than 1.73, and the reduced-order model of the SEOBNRv4 approximant is used elsewhere. More details on the bank construction can be found in Ref. [84].

# C. Coherent WaveBurst

Coherent WaveBurst (cWB) is an analysis algorithm used in searches for weakly modeled (or unmodeled)

transient signals with networks of GW detectors. Designed to operate without a specific waveform model, cWB identifies coincident excess power in the multiresolution time-frequency representations of the detector strain data [85], for signal frequencies up to $1\mathrm{kHz}$ and durations up to a few seconds. The search identifies events that are coherent in
multiple detectors and reconstructs the source sky location and signal waveforms by using the constrained maximum likelihood method [11].

The cWB detection statistic is based on the coherent energy $E_{c}$ obtained by cross-correlating the signal waveforms reconstructed in the two detectors. It is proportional to the coherent network SNR and used to rank each cWB candidate event. For an estimation of its statistical significance, each candidate event is ranked against a sample of background triggers obtained by repeating the analysis on time-shifted data, similar to the background estimation in the PyCBC search.

To exclude astrophysical events from the background sample, the time shifts are selected to be much larger than the expected signal delay between the detectors. Each cWB event is assigned a FAR given by the rate of background triggers with a larger coherent network SNR.

To increase robustness against nonstationary detector noise, cWB uses signal-independent vetoes, which reduce the high rate of the initial excess power triggers. The primary veto cut is on the network correlation coefficient $c_{c} = E_{c} / (E_{c} + E_{n})$ , where $E_{n}$ is the residual noise energy estimated after the reconstructed signal is subtracted from the data. Typically, for a GW signal $c_{c} \approx 1$ , and for instrumental glitches $c_{c} \ll 1$ . Therefore, candidate events with $c_{c} < 0.7$ are rejected as potential glitches.

Finally, to improve the detection efficiency for a specific class of stellar-mass BBH sources and further reduce the number of false alarms, cWB selects a subset of detected events for which the frequency is increasing with time, i.e., events with a chirping time-frequency pattern. Such a time-frequency pattern captures the phenomenological behavior of most CBC sources. This flexibility allows cWB to potentially identify CBC sources with features such as higher-order modes, high mass ratios, misaligned spins, and eccentric orbits; it complements the existing templated algorithms by searching for new and possibly unexpected CBC populations.

For events that passed the signal-independent vetoes and chirp cut, the detection significance is characterized by a FAR computed as described above; otherwise, cWB provides only the reconstructed waveforms (see Sec. VI).

# IV. SEARCH RESULTS

# A. Selection criteria

In this section, we motivate and describe the selection of gravitational-wave events for presentation in this paper. We include any candidate event that can be identified with a nontrivial probability of association to an astrophysical binary merger event, as opposed to instrumental noise [86].

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-6

The matched-filter and cWB search pipelines produce large numbers of candidate events, but the majority of these are of very low significance and have a correspondingly low probability of being of astrophysical origin.

We desire to identify all events that are confidently astrophysical in origin and additionally provide a manageable set of marginal triggers that may include some true signals but certainly also includes noise triggers. To do this identification, we establish an initial threshold on estimated FAR of 1 per 30 days (about 12.2 per year), excluding any event that does not have a FAR less than this threshold in at least one of the two matched-filter analyses (see Sec. III). The cWB search results are not used in the event selection process.

At this FAR threshold, if each pipeline produces independent noise events, we would expect on average two such noise events (false alarms) per month of analyzed coincident time. During these first two observing runs, we also empirically observe approximately two likely signal events per month of analyzed time. Thus, for $O1$ and $O2$ , any sample of events all of whose measured FARs are greater than 1 per 30 days is expected to consist of at least $50\%$ noise triggers. Individual triggers within such a sample are then considered to be of little astrophysical interest.

Since the number of triggers with a FAR less than 1 per 30 days is manageable, restricting our attention to triggers with lower FAR captures all confident detections while also probing noise triggers.

Within the sample of triggers with a FAR less than the ceiling of 1 per 30 days in at least one of the matched-filter searches, we assign the "GW" designation to any event for which the probability of astrophysical origin from either matched-filter search is greater than $50\%$ (for the exact

definition and calculation of the astrophysical probability, see Sec. VII). We list these events in Table I.

For the remaining events in the sample that pass the initial FAR threshold, neither matched-filter search finds a greater than $50\%$ probability of astrophysical origin. These are considered marginal events and are listed in Table II. The astrophysical probabilities of all events, confident and marginal, are given in Table IV.

# B. Gravitational-wave events

Results from the two matched-filter searches are shown in Fig. 2 and that of the unmodeled burst search in Fig. 3. In each plot, we show the observed distribution of events as a function of the inverse false-alarm rate, as well as the expected background for the analysis time, with Poisson uncertainty bands. The foreground distributions clearly stand out from the background, even though we show only rightward-pointing arrows for any event with a measured or bounded IFAR greater than $3000\mathrm{y}$ .

We present more quantitative details below on the 11 gravitational events, as selected by the criteria in Sec. IVA, in Table I. Of these 11 events, seven have been previously reported: the three gravitational-wave events from $O1$ [1-4] and, from $O2$ , the binary neutron star merger GW170817 [18] and the binary black hole events GW170104 [15], GW170608 [17], and GW170814 [16]. The updated results we report here supersede those previously published. Four new gravitational-wave events are reported here for the first time: GW170729, GW170809, GW170818, and GW170823. All four are binary black hole events.

As noted in Sec. III, data from $O1$ are reanalyzed because of improvements in the search pipelines and the

![](dt=2026-04-19/ht=09/c11d44e13ed7291e3345ad7d9a3ce71f98538e1657318dcdd2e664ebe202e796.jpg)

![](dt=2026-04-19/ht=09/cf4aeb7cd3f2ec84dcdbe6001bd719ed6b1f981a1c2b34c3c4b67c6f5348f928.jpg)

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-7

![](dt=2026-04-19/ht=09/70417465ee45f881b997ccbc6831f29f23691b02b1ca9afb65e9a617d84b423f.jpg)

expansion of the parameter space searched. For the $O2$ events already published, our reanalysis is motivated by updates to the data itself. The noise subtraction procedure [52] that is available for parameter estimation of three of the published $O2$ events was not initially applied to the entire $O2$ dataset and, therefore, could not be used by searches. Following the procedures of Ref. [51], this noise subtraction is applied to all of $O2$ and is reflected in Table I for the four previously published $O2$ GW events, as well as the four events presented here for the first time.

For both PyCBC and cWB, the time-shift method of background estimation may result in only an upper bound on the false-alarm rate, if an event has a larger value of the ranking statistic than any trigger in the time-shifted background; this result is indicated in Table I. For GW150914 and GW151226, the bound that PyCBC places on the FAR in these updated results is in fact higher than that previously published [1,2,4], because, as noted in Sec. III A, this search elects to use shorter periods of time shifting to better ca
pture the variation in the detectors' sensitivities.

For GstLAL, the FAR is reported in Table I as an upper bound of $1.00 \times 10^{-7}$ whenever a smaller number is obtained, which reflects a more conservative noise hypothesis within the GstLAL analysis and follows the procedures and motivations detailed in Sec. IV in Ref. [3].

Five of the GW events reported here occurred during August 2017, which comprises approximately $10\%$ of the total observation time. There are ten nonoverlapping periods of similar duration, with an average event rate of 1.1 per

period. The probability that a Poisson process would produce five events or more in at least one of those periods is $5.3\%$ . Thus, seeing five events in one month is statistically consistent with expectations. For more details, see Ref. [87].

For the remainder of this section, we briefly discuss each of the gravitational-wave events, highlighting interesting features from the perspectives of the three searches. A discussion of the properties of these sources may be found in Sec. V. Though the results presented are from the final, offline analysis of each search, for the four new GW events, we also indicate whether the event is found in a low-latency search and an alert sent to electromagnetic observing partners.

Where this process did occur, we mention in this paper only the low-latency versions of the three searches with offline results presented here; in some cases, additional low-latency pipelines also found events. A more thorough discussion of all of the low-latency analyses and the electromagnetic follow-up of $O2$ events may be found in Ref. [22].

# 1. GW150914, GW151012, and GW151226

During $O1$ , two confident detections of binary black holes were made: GW150914 [1] and GW151226 [2]. Additionally, a third trigger was noted in the $O1$ catalog of binary black holes [3,4] and labeled LVT151012. That label is a consequence of the higher FAR of that trigger, though detector characterization studies show no instrumental or environmental artifact, and the results of parameter estimation are consistent with an astrophysical BBH source. Even with the significance that is measured with the $O1$ search pipelines [4], this event meets the criteria of Sec. IVA for a gravitational-wave event, and we henceforth relabel this event as GW151012.

The improved $O2$ pipelines substantially reduce the FAR assigned to GW151012: It is now $0.17\mathrm{y}^{-1}$ in the PyCBC search (previously, $0.37\mathrm{y}^{-1}$ ) and $7.92\times 10^{-3}\mathrm{y}^{-1}$ in the GstLAL search (previously, $0.17\mathrm{y}^{-1}$ ). These improved FAR measurements for GW151012 are the most salient result of the reanalysis of $O1$ with the $O2$ pipelines; no new gravitational-wave events were discovered. The first binary black hole observation, GW150914, remains the highest SNR event in $O1$ and the second highest in the combined $O1$ and $O2$ datasets, behind only the binary neutron star inspiral GW170817.

Recently, Ref. [88] appeared. That catalog also presents search results from the PyCBC pipeline for $O1$ and also finds GW150914, GW151012, and GW151226 as the only confident gravitational-wave events in $O1$ , with identical bounds on FAR to the PyCBC results in Table I for GW150914 and GW151226. The measured FAR for GW151012 is not identical but is consistent with the results we present in Table I.

# 2. GW170104, GW170608, and GW170814

Three binary black hole events from $O2$ have already been published: GW170104 [15], GW170608 [17], and

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-8

GW170814 [16]. Updated search results for these events are presented in Table I. As noted in the original publication for GW170608 [17], the Hanford detector was undergoing a procedure to stabilize angular noise at the time of the event; the Livingston detector was operating in a nominal configuration. For this reason, a specialized analysis time when both LIGO detectors were operating in that same configuration is identified, between June 7, 2017 and June 9, 2017. This period that was used to analyze GW170608 in the initial publication is again used for the results in Table I, though with the noise subtraction applied.

In the reanalysis of $O2$ data, GW170814 is identified as a double-coincident event between LLO and LHO by GstLAL. This results from the noise subtraction in the LIGO data and updated calibration of the Virgo data. Because of the noise subtraction in the LIGO data, under GstLAL's ranking of multiple triggers [9], a new template generates the highest ranked trigger as double coincident, with a Hanford SNR of 9.1 (the previous highest ranked trigger, a triple, had 7.3).

Though this highest ranked event is a double-coincident trigger, the pipeline does identify other highly significant triggers, some double coincident and some triple coincident. As the search uses a discrete template bank, peaks from the SNR time series of the individual detectors, and clustering of several coincident triggers over the bank, it is difficult in this case to tell from the search results alone whether the event is truly a triple-coincident detection. For a definitive answer, we perform a fully Bayesian analysis with and without the Virgo data, similar to the results in Ref.

[16]. Comparing the evidence, this Bayesian analysis—which enforces coherence and therefore more fully exploits consistency among detected amplitudes, phases, and times of arrival than the search pipelines—finds that a triple-coincident detection is strongly favored over a double-coincident detection, by a factor of approximately 60. Thus, the updated results are consistent with those that were previously published.

# 3. GW170817

Across the entirety of $O1$ and $O2$ , the binary neutron star inspiral GW170817 remains the event with the highest network SNR and is accordingly assigned the most stringent possible bound on its FAR by PyCBC and the highest value of $\mathcal{L}$ (the logarithm of the likelihood ratio) of any event in the combined $O1$ and $O2$ dataset by GstLAL. As explained in detail in the original detection paper [18], a loud glitch occurs near the end of this signal in LLO. For the matched-filter searches, this glitch is excised via time-domain gating (and that gating is applied consistently to all such glitches throughout $O2$ ). Because the cWB pipeline is designed to detect short signals, it does not use that gating technique, and it rejects this event because of the glitch.

# 4. GW170729

We turn now to gravitational-wave events not previously announced. The first of these is GW170729, observed at 18:56:29.3 UTC on July 29, 2017. The PyCBC pipeline assigns it a FAR of $1.36\mathrm{y}^{-1}$ , the GSTLAL pipeline a FAR of $0.18\mathrm{y}^{-1}$ , and the cWB pipeline a FAR of $0.02\mathrm{y}^{-1}$ .

As it is identified with the highest significance among all three search pipelines by the weakly modeled pipeline, it is worth investigating whether this event is unusual in some way, exhibiting effects (for instance, precession or higher-order modes) not adequately modeled by the templates used in the matched-filter searches. As a relatively simple way of investigating this question, a comparison study is done between the PyCBC pipeline and cWB, using software injections with parameters drawn from the SEOBNRv4 ROM parameter estimation of this event.

That waveform does not incorporate precession or higher-order modes, but, by using these samples as inputs to both searches, we can probe how often we see comparable results. It is found that approximately $4\%$ of these SEOBNRv4 ROM samples are recovered by both the PyCBC and cWB pipelines with $\mathrm{FAR} \geq 1\mathrm{y}^{-1}$ and $\mathrm{FAR} \leq 0.02\mathrm{y}^{-1}$ , respectively.

Thus, the observed difference in FARs between the two pipelines is not exceptionally unlikely and is consistent with a noise fluctuation which happens to decrease the significance of the event as seen b
y PyCBC and increase it for cWB. The detailed CBC parameter estimation studies in Sec. V also indicate no significant evidence for observationally important precession or higher-order modes. This event was identified only in the offline analyses, so no alert was sent to electromagnetic partners.

# 5. GW170809

GW170809 was observed on August 9, 2017 at 08:28:21.8 UTC with a FAR of $1.45 \times 10^{-4} \mathrm{y}^{-1}$ by PyCBC and $< 1.00 \times 10^{-7} \mathrm{y}^{-1}$ by GSTLAL. This event was identified in low latency by both the GSTLAL and cWB pipelines, and an alert was sent to electromagnetic observing partners. In the final offline cWB analysis with updated calibration and noise subtracted from LIGO data, this event did not pass one of the signal-independent vetoes (Sec. III C) and was therefore not assigned a FAR.

# 6. GW170818

GW170818 was observed at 02:25:09.1 UTC on August 18, 2017, by GSTLAL with a FAR of $4.20 \times 10^{-5} \mathrm{y}^{-1}$ ; it was not observed by either the PyCBC or cWB pipelines. It is observed as a triple-coincident event by GSTLAL, with an SNR in Virgo of 4.2, a Hanford SNR of 4.1, and a Livingston SNR of 9.7. In the PyCBC search, a trigger is seen in the Livingston detector with a comparable SNR and is noted as a "chirplike" single-detector trigger. When the Hanford and Virgo data are analyzed with modified settings around the time of that event, there are triggers with a similar SNR to

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-9

those of GSTLAL and, therefore, well below the threshold of 5.5 needed for a single-detector trigger in the PyCBC search to be considered further for possible coincidence. This event is initially identified in low latency by the GSTLAL pipeline as a LLO-Virgo double-detector trigger. Online, the Virgo trigger is not included in significance estimation, and the LLO-only trigger does not pass the false-alarm threshold for the online search. Therefore, at that time no alerts was sent to electromagnetic observing partners.

# 7. GW170823

On August 23, 2017, GW170823 was observed at 13:13:58.5 UTC. Its FAR is $< 3.29 \times 10^{-5} \mathrm{y}^{-1}$ in the PyCBC pipeline, $< 1.00 \times 10^{-7} \mathrm{y}^{-1}$ in the GstLAL pipeline, and $2.14 \times 10^{-3} \mathrm{y}^{-1}$ in the cWB pipeline. The online versions of each of these pipelines detect this event in low latency, and an alert is sent to electromagnetic observing partners.

# C. Marginal triggers and instrumental artifacts

In Table II, we present the remaining 14 triggers from $O1$ and $O2$ that pass the initial threshold of a FAR less than one per 30 days in at least one of the two matched-filter searches but are not assigned a probability of astrophysical origin of more than $50\%$ by either pipeline (see Table IV). As noise triggers are generically a function of the details of the pipeline that identifies a trigger, we do not typically expect to see the same noise triggers in each pipeline. In Table II, we therefore present which of the two pipelines identified the trigger, as well as the FAR of that trigger, its SNR, and the chirp mass of the template generating the trigger; these chirp masses do not come from a detailed parameter estimation as is performed in Sec. V for the gravitational-wave events.

Before discussing the final column in Table II, we consider the reasonableness of the number of these triggers. The matched-filter pipelines analyze 0.46 y of coincident data, so, at a false-alarm threshold of once per 30 days, we would expect about six triggers purely from noise. We see from Table II that the PyCBC search observed three marginal triggers and the GSTLAL search observed 11.

Though the probability that 11 triggers could arise only from noise when six are expected is low, it is by itself not sufficiently low to confidently assert that some fraction of these triggers are astrophysical in origin. It is possible, however, that either search's marginal triggers could contain a population of real GW signals. In particular, the multicomponent population analysis [89,90] (see Sec. VII) explicitly considers the possibility of triggers (both confident detections and marginal triggers) arising from a combination of noise and distinct source populations.

For GSTLAL, the combined count of GWs and marginal triggers is 22, and the analysis in Sec. VII finds that to be within expectations at the $90\%$ level. Although it may be the case

that some of these marginal triggers are of astrophysical origin, we cannot then determine which ones.

Now we turn to a summary of the detector characterization information for each marginal trigger, briefly indicated in the final column in Table II. Following a subset of procedures used for previous gravitational-wave detections [91], we evaluate the possibility that artifacts from instrumental or environmental noise could have caused each of the marginal triggers.

Using auxiliary sensors at each detector, as well as the gravitational-wave strain data, we evaluate the state of the detectors at the time of each marginal trigger, identify and investigate any artifacts in the data due to noise, and test whether any identified artifacts might explain the excess SNR observed in the analysis. Of the marginal triggers presented in this catalog, nine have excess power from known sources of noise occurring during times when the matched-filter template that yields the trigger has a GW frequency within the sensitive band of the detectors.

For four of these cases, the observed instrumental artifact overlaps the signal region and possibly causes the marginal trigger.

Details on the physical couplings that create these instrumental artifacts and possible mitigation strategies useful for analysis of LIGO-Virgo data are discussed in Appendix A. For the remainder of this subsection, we describe how the different categories discussed in that Appendix apply to the marginal triggers in Table II.

To determine whether artifacts identified as noise "could account for" marginal triggers, we use two metrics: (i) whether the type of noise has been shown to produce an excess of triggers consistent with the properties of the trigger present and (ii) the noise artifact could account for the presence of the trigger as reported by that search, including SNR and time-frequency evolution, without the presence of an astrophysical signal.

In and of themselves, these classifications do not affect the probability that any particular marginal trigger is associated with a signal as measured by the searches but are statements about the evidence of transient noise in the detectors. It is expected that a substantial fraction of marginal events at the false-alarm rate values reported are caused by noise, given the estimated background of our searches and the expected rate of signals. See Sec. VII, Fig. 11, and Table IV below for a more detailed discussion of the probabilities of astrophysical or noise origin of such events.

# 1. No noise artifacts present: 151008, 151116, 170219, 170423, and 170705

Investigations into this set of marginal triggers have identified no instrumental artifacts in time coincidence with the triggers.

# 2. Light scattering possibly caused: 161217 and 170720

All marginal triggers in this class and the next are in time coincidence with artifacts from scattered light in one of the

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-10

detectors. Scattered light leads to excess power at low frequencies that appear in time-frequency spectrograms as archlike shapes. In some cases, the frequencies affected are above the minimum frequency used in the analysis. When this happens, scattered light transients can create significant triggers in matched-filter searches [68,92,93].

The two marginal triggers 161217 and 170720 occur during periods of scattered light affecting frequencies up to $80\mathrm{Hz}$ with high-amplitude arches. In both cases,
a significant overlap with the trigger template and the excess power from scattering is observed. Investigations into the status of the observatories at the times in question identified high-amplitude ground motion correlated with the scattering.

The marginal trigger 161217 occurs during a period of high-amplitude ground motion at Livingston caused by storm activity. During this storm activity, the Livingston detector is not able to maintain a stable interferometer for periods longer than $10\mathrm{min}$ . The presence of intense scattering artifacts contribute to the unstable state of the interferometer and could account for the SNR of the marginal trigger. Because of the short observing duration, this time period is not analyzed by the PyCBC search.

Within 20 s of trigger 170720, excess ground motion from earthquakes forced the Livingston detector to drop out of its nominal mode of operation. Before the detector dropped out of the observing state, the data are heavily polluted with scattering artifacts that could account for the SNR of the triggers. As the PyCBC search does not consider times near the edges of observing periods, this time period is also not analyzed by that search. Artifacts related to scattered light are also observed at Hanford at this time.

# 3. Light scattering present: 151012.2, 170208, and 170616

In the case of trigger 151012.2, light scattering does not introduce significant power above $30\mathrm{Hz}$ prior to the reported trigger time. Investigations into the relationship between the trigger and the scattered light find no power overlap, suggesting that the artifacts could not account for the observed marginal trigger.

Investigations into triggers 170208 and 170616 find similar results. In the case of these triggers, a slight overlap with excess power from scattering is observed. Multiple efforts, including BAYESWAVE [53] glitch subtraction and gating [8], are used to mitigate the scattered light artifacts. After subtraction of the noise artifacts, the data are reanalyzed to evaluate whether the excess power subtracted could have accounted for the trigger. In both cases, the marginal trigger remains with similar significance, suggesting that the observed scattering artifact could not have accounted for the SNR of the marginal trigger.

# 4. 60-200 Hz nonstationarity possibly caused: 161202 and 170412

This class of marginal triggers occurs during periods of noise referred to as "60-200 Hz nonstationarity." This

nonstationarity appears in time-frequency spectrograms as excess power with slowly varying frequencies over time periods of multiple minutes.

Previous work [68] has shown that periods of $60 - 200\mathrm{Hz}$ nonstationarity can cause significant triggers in the searches, both impacting the ability of searches to accurately measure the noise spectrum of the data and contributing excess noise to matched-filter searches. Triggers 161202 and 170412 demonstrate a significant overlap with excess power from the nonstationarity noise. BAYESWAVE [53] glitch subtraction is unable to completely mitigate the $60 - 200\mathrm{Hz}$ nonstationarity due to its long duration.

# 5. Short-duration, high-amplitude artifacts present: 170405 and 170630

The marginal triggers in this class occur in time coincidence with short-duration, high-amplitude noise transients that are removed in the data-conditioning step of the search pipelines [8]. The times surrounding these transients do not demonstrate an elevated trigger rate after the transient has been removed. Trigger 170405 is in coincidence with this class of transient at Hanford, and trigger 170630 is in coincidence with this class of transient at Livingston. As triggers 170405 and 170630 are identified as significant after the removal of the short-duration transients, the presence of noise artifacts cannot account for the SNR of these marginal triggers.

# V. SOURCE PROPERTIES

Here, we present inferred source properties of gravitational-wave signals observed by the LIGO and Virgo detectors under the assumption that they originate from compact binary coalescences described by general relativity. We analyze all GW events described in Sec. IV. Full parameter estimation (PE) results for $O1$ events are provided for GW150914 in Refs. [4,95,96], for GW151226 in Refs. [2,4], and for GW151012 in Refs. [3,4]. PE results for four $O2$ events are provided for GW170104 in Ref. [15], for GW170608 in Ref. [17], for GW170814 in Ref. [16], and for GW170817 in Refs. [18,97].

Data from the three-detector LIGO-Virgo network are used to obtain parameter estimates for GW170729, GW170809, GW170814, GW170817, and GW170818. Virgo data for GW170729, from the commissioning phase before it officially joined, are included for PE analyses, because the calibration of the data and the sensitivity of the instrument are comparable to that in August. For the remaining events, the analysis uses data from the two LIGO detectors.

We perform a reanalysis of the data for the entirety of $O1$ and $O2$ including the published events. As discussed in Sec. II C, the $O2$ data are recalibrated and cleaned [52]. These improvements increase the sensitivity of the detector network and motivate a reanalysis also for already-published events found in $O2$ data. While the $O1$ data

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-11

TABLE III. Selected source parameters of the 11 confident detections. We report median values with $90\%$ credible intervals that include statistical errors and systematic errors from averaging the results of two waveform models for BBHs. For GW170817, credible intervals and statistical errors are shown for IMRPhenomPv2NRT with a low spin prior, while the sky area is computed from TaylorF2 samples. The redshift for NGC 4993 from Ref. [94] and its associated uncertainties are used to calculate source-frame masses for GW170817.

For BBH events, the redshift is calculated from the luminosity distance and assumed cosmology as discussed in Appendix B. The columns show source-frame component masses $m_{i}$ and chirp mass $\mathcal{M}$ , dimensionless effective aligned spin $\chi_{\mathrm{eff}}$ , final source-frame mass $M_{f}$ , final spin $a_{f}$ , radiated energy $E_{\mathrm{rad}}$ , peak luminosity $l_{\mathrm{peak}}$ , luminosity distance $d_{L}$ , redshift $z$ , and sky localization $\Delta \Omega$ . The sky localization is the area of the $90\%$ credible region.

For GW170817, we give conservative bounds on parameters of the final remnant discussed in Sec. V E.

![](dt=2026-04-19/ht=09/d7fc247df1bc8666fa36b6d982ff2b53561fab9019d8622ebf5fa211bbec16fa.jpg)

<table><tr><td>Event</td><td>m1/M⊙</td><td>m2/M⊙</td><td>M/M⊙</td><td>χeff</td><td>Mf/M⊙</td><td>af</td><td>Erad/(M⊙c2)</td><td>ℓpeak/(erg s-1)</td><td>dL/Mpc</td><td>z</td><td>ΔΩ/deg2</td></tr><tr><td rowspan="2">GW150914</td><td>35.6+4.7</td><td>30.6+3.0</td><td>28.6+1.7</td><td>-0.01+0.12</td><td>63.1+3.4</td><td>0.69+0.05</td><td>3.1+0.4</td><td>3.6+0.4×1056</td><td>440+150</td><td>0.09+0.03</td><td>182</td></tr><tr><td>-3.1</td><td>-4.4</td><td>-1.5</td><td>-0.13</td><td>-3.0</td><td>-0.04</td><td>-0.4</td><td>-0.4</td><td>-1.70</td><td>0.09-0.03</td><td>1523</td></tr><tr><td rowspan="2">GW151012</td><td>23.2+14.9</td><td>13.6+4.1</td><td>15.2+2.1</td><td>0.05+0.31</td><td>35.6+10.8</td><td>0.67+0.13</td><td>1.6+0.6</td><td>3.2+0.8×1056</td><td>1080+550</td><td>0.21+0.09</td><td>921</td></tr><tr><td>-5.5</td><td>-4.8</td><td>-1.2</td><td>-0.20</td><td>-3.8</td><td>-0.11</td><td>-0.5</td><td>-1.7</td><td>-490</td><td>0.20+0.08</td><td>921</td></tr><tr><td rowspan="2">GW151226</td><td>13.7+8.8</td><td>7.7+2.2</td><td>8.9+0.3</td><td>0.18+0.20</td><td>20.5+6.4</td><td>0.74+0.07</td><td>1.0+0.1</td><td>3.4+0.7×1056</td><td>450+180</td><td>0.09+0.04</td><td>1033</td></tr><tr><td>-3.2</td><td>-2.5</td><td>-0.3</td><t
d>-0.12</td><td>-1.5</td><td>-0.05</td><td>-0.2</td><td>-1.7</td><td>-190</td><td>0.09-0.04</td><td>1033</td></tr><tr><td rowspan="2">GW170104</td><td>30.8+7.3</td><td>20.0+4.9</td><td>21.4+2.2</td><td>-0.04+0.17</td><td>48.9+5.1</td><td>0.66+0.08</td><td>2.2+0.5</td><td>3.3+0.6×1056</td><td>990+440</td><td>0.20+0.08</td><td>921</td></tr><tr><td>-5.6</td><td>-4.6</td><td>-1.8</td><td>-0.21</td><td>-4.0</td><td>-0.11</td><td>-0.5</td><td>-1.0</td><td>-430</td><td>0.20-0.08</td><td>921</td></tr><tr><td rowspan="2">GW170608</td><td>11.0+5.5</td><td>7.6+1.4</td><td>7.9+0.2</td><td>0.03+0.19</td><td>17.8+3.4</td><td>0.69+0.04</td><td>0.9+0.0</td><td>3.5+0.4×1056</td><td>320+120</td><td>0.07+0.02</td><td>392</td></tr><tr><td>-1.7</td><td>-2.2</td><td>-0.2</td><td>-0.07</td><td>-0.7</td><td>-0.04</td><td>-0.1</td><td>-1.3</td><td>-110</td><td>-0.02</td><td>392</td></tr><tr><td rowspan="2">GW170729</td><td>50.2+16.2</td><td>34.0+9.1</td><td>35.4+6.5</td><td>0.37+0.21</td><td>79.5+14.7</td><td>0.81+0.07</td><td>4.8+1.7</td><td>4.2+0.9×1056</td><td>2840+1400</td><td>0.49+0.19</td><td>1041</td></tr><tr><td>-10.2</td><td>-10.1</td><td>-4.8</td><td>-0.25</td><td>-10.2</td><td>-0.13</td><td>-1.7</td><td>-1.5</td><td>-1360</td><td>-0.21</td><td>1041</td></tr><tr><td rowspan="2">GW170809</td><td>35.0+8.3</td><td>23.8+5.1</td><td>24.9+2.1</td><td>0.08+0.17</td><td>56.3+5.2</td><td>0.70+0.08</td><td>2.7+0.6</td><td>3.5+0.6×1056</td><td>1030+320</td><td>0.20+0.05</td><td>308</td></tr><tr><td>-5.9</td><td>-5.2</td><td>-1.7</td><td>-0.17</td><td>-3.8</td><td>-0.09</td><td>-0.6</td><td>-0.9</td><td>-390</td><td>-0.07</td><td>308</td></tr><tr><td rowspan="2">GW170814</td><td>30.6+5.6</td><td>25.2+2.8</td><td>24.1+1.4</td><td>0.07+0.12</td><td>53.2+3.2</td><td>0.72+0.07</td><td>2.7+0.4</td><td>3.7+0.4×1056</td><td>600+150</td><td>0.12+0.03</td><td>87</td></tr><tr><td>-3.0</td><td>-4.0</td><td>-1.1</td><td>-0.12</td><td>-2.4</td><td>-0.05</td><td>-0.3</td><td>-0.5</td><td>-220</td><td>-0.04</td><td>87</td></tr><tr><td rowspan="2">GW170817</td><td>1.46+0.12</td><td>1.27+0.09</td><td>1.186+0.001</td><td>0.00+0.02</td><td>≤2.8</td><td>≤0.89</td><td>≥0.04</td><td>≥0.1×1056</td><td>40+7</td><td>0.01+0.00</td><td>16</td></tr><tr><td>-0.10</td><td>-0.09</td><td>-0.001</td><td>-0.01</td><td>-0.1</td><td>-0.89</td><td>≥0.04</td><td>≥0.1×1056</td><td>-15</td><td>-0.00</td><td>16</td></tr><tr><td rowspan="2">GW170818</td><td>35.4+7.5</td><td>26.7+4.3</td><td>26.5+2.1</td><td>-0.09+0.18</td><td>59.4+4.9</td><td>0.67+0.07</td><td>2.7+0.5</td><td>3.4+0.5×1056</td><td>1060+420</td><td>0.21+0.07</td><td>39</td></tr><tr><td>-4.7</td><td>-5.2</td><td>-1.7</td><td>-0.21</td><td>-3.8</td><td>-0.08</td><td>-0.5</td><td>-0.7</td><td>-380</td><td>-0.07</td><td>39</td></tr><tr><td rowspan="2">GW170823</td><td>39.5+11.2</td><td>29.0+6.7</td><td>29.2+4.6</td><td>0.09+0.22</td><td>65.4+10.1</td><td>0.72+0.09</td><td>3.3+1.0</td><td>3.6+0.7×1056</td><td>990+970</td><td>0.35+0.15</td><td>1666</td></tr><tr><td>-6.7</td><td>-7.8</td><td>-3.6</td><td>-0.26</td><td>-7.4</td><td>-0.12</td><td>-0.9</td><td>-1.1</td><td>-900</td><td>-0.15</td><td>1666</td></tr></table>

and calibration have not changed, a reanalysis is valuable for the following reasons: (i) Parameter estimation analyses use an improved method for estimating the power spectral density of the detector noise [53,54] and frequency-dependent calibration envelopes [98]; (ii) we use two waveform models that incorporate precession and combine their posteriors to mitigate model uncertainties.

Key source parameters for the ten BBHs and one BNS are shown in Table III. We quote the median and symmetric $90\%$ credible intervals for inferred quantities. For BBH coalescences, parameter uncertainties include statistical and systematic errors from averaging posterior probability distributions over the two waveform models, as well as calibration uncertainty. Apart from GW170817, all posterior distributions of GW events are consistent with originating from BBHs. Posterior distributions for all GW events are shown in Figs. 4-8.

Mass and tidal deformability posteriors for GW170817 are shown in Fig. 9. For BBH coalescences, we present combined posterior distributions from an effective precessing spin waveform model (IMRPhenomPv2) [25,26,49] and a fully precessing model (SEOBNRv3) [27,28,30]. For the analysis of GW170817, we present results for three frequency-domain models IMRPhenomPv2NRT [25,26,32,49,99], SEOBNRv4NRT [29,32,77,99], and TaylorF2 [35,36, 38,100-112] and two time-domain models SEOBNRv4T [31] and TEOBResumS [33,113].

Details on Bayesian parameter estimation methods, prior choices, and waveform models used for BBH and BNS systems are provided in Appendix B, B 1, and B 2, respectively. We discuss an

analysis including higher harmonics in the waveform in Appendix B 3 and find results broadly consistent with the analysis presented below. The impact of prior choices on selected results is discussed in Appendix C.

# A. Source parameters

The GW signal emitted from a BBH coalescence depends on intrinsic parameters that directly characterize the binary's dynamics and emitted waveform, and extrinsic parameters that encode the relation of the source to the detector network. In general relativity, an isolated BH is uniquely described by its mass, spin, and electric charge [114-118]. For astrophysical BHs, we assume the electric charge to be negligible.

A BBH undergoing quasicircular inspiral can be described by eight intrinsic parameters, the two masses $m_{i}$ , and the two three-dimensional spin vectors $\vec{S}_{i}$ of its component BHs defined at a reference frequency. Seven additional extrinsic parameters are needed to describe a BH binary: the sky location (right ascension $\alpha$ and declination $\delta$ ), luminosity distance $d_{L}$ , the orbital inclination $\iota$ and polarization angle $\psi$ , the time $t_{c}$ , and phase $\phi_{c}$ at coalescence.

Since the maximum spin a Kerr BH of mass $m$ can reach is $(Gm^2) / c$ , we define dimensionless spin vectors $\vec{\chi}_i = c\vec{S}_i / (Gm_i^2)$ and spin magnitudes $a_{i} = c|\vec{S}_{i}| / (Gm_{i}^{2})$ . If the spins have a component in the orbital plane, then the binary's orbital angular momentum $\vec{L}$ and its spin vectors precess [119,120] around the total angular momentum $\vec{J} = \vec{L} +\vec{S}_1 + \vec{S}_2$ .

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-12

![](dt=2026-04-19/ht=09/e8ae2d32bc5f8bd260159149cfba215e23d766c2af0db13af8b9fb75d8be3ff0.jpg)

![](dt=2026-04-19/ht=09/fdd6b72805413ca4fa9fb3cfea1d16a07e1d9087b2cef4659733b46592b72317.jpg)

![](dt=2026-04-19/ht=09/2bbff8f56a507dc40365b54f5ee81ea5757d227b0e75cb5f6198bef29bd0d4e6.jpg)

We describe the dominant spin effects by introducing effective parameters. The effective aligned spin is defined as a simple mass-weighted linear combination of the spins [23,24,121] projected onto the Newtonian angular momentum $\hat{L}_N$ , which is normal to the orbital plane ( $\hat{L} = \hat{L}_N$ for aligned-spin binaries)

$$
\chi_ {\text {e f f}} = \frac {\left(m _ {1} \vec {\chi} _ {1} + m _ {2} \vec {\chi} _ {2}\right) \cdot \hat {L} _ {N}}{M}, \tag {4}
$$

where $M = m_{1} + m_{2}$ is the total mass of the binary and $m_{1}$ is defined to be the mass of the larger component of the binary, such that $m_{1} \geq m_{2}$ . Different parameterizations of spin effects are possible and can be motivated from their appearance in the GW phase or dynamics [122-124]. $\chi_{\mathrm{eff}}$ is approximately conserved throughout the inspiral [121]. To assess whether a binary is precessing, we use a single effective precession spin parameter $\chi_{p}$ [125] (see Appendix C).

During the i
nspiral, the phase evolution depends at leading order on the chirp mass [34,126,127]

$$
\mathcal {M} = \frac {\left(m _ {1} m _ {2}\right) ^ {3 / 5}}{M ^ {1 / 5}}, \tag {5}
$$

which is also the best measured parameter for low-mass systems dominated by the inspiral [63,101,122,128]. The mass ratio

$$
q = \frac {m _ {2}}{m _ {1}} \leq 1 \tag {6}
$$

and effective aligned spin $\chi_{\mathrm{eff}}$ appear in the phasing at higher orders [101,121,123].

For precessing binaries, the orbital angular momentum vector $\vec{L}$ is not a stable direction, and it is preferable to describe the source inclination by the angle $\theta_{JN}$ between

the total angular momentum $\vec{J}$ (which typically is approximately constant throughout the inspiral) and the line-of-sight vector $\vec{N}$ instead of the orbital inclination angle $\iota$ between $\vec{L}$ and $\vec{N}$ [119,129]. We quote frequency-dependent quantities such as spin vectors and derived quantities as $\chi_{p}$ at a GW reference frequency $f_{\mathrm{ref}} = 20~\mathrm{Hz}$ .

Binary neutron stars have additional degrees of freedom (d.o.f.) related to their response to a tidal field. The dominant quadrupolar $(\ell = 2)$ tidal deformation is described by the dimensionless tidal deformability $\Lambda = (2/3)k_{2}[(c^{2}/G)(R/m)]^{5}$ of each neutron star (NS), where $k_{2}$ is the dimensionless $\ell = 2$ Love number and $R$ is the NS radius. The tidal deformabilities depend on the NS mass $m$ and the equation of state (EOS). The dominant tidal contribution to the GW phase evolution is encapsulated in an effective tidal deformability parameter [130,131]:

$$
\tilde {\Lambda} = \frac {1 6}{1 3} \frac {\left(m _ {1} + 1 2 m _ {2}\right) m _ {1} ^ {4} \Lambda_ {1} + \left(m _ {2} + 1 2 m _ {1}\right) m _ {2} ^ {4} \Lambda_ {2}}{M ^ {5}}. \tag {7}
$$

# B. Masses

In the left panel in Fig. 4, we show the inferred component masses of the binaries in the source frame as contours in the $m_{1} - m_{2}$ plane. Because of the mass prior, we consider only systems with $m_{1} \geq m_{2}$ and exclude the shaded region. The component masses of the detected BH binaries cover a wide range from about $5M_{\odot}$ to about $70M_{\odot}$ and lie within the range expected for stellar-mass BHs [132-134]. The posterior distribution of the heavier component in the heaviest BBH, GW170729, grazes the lower boundary of the possible mass gap expected from pulsational pair instability and pair instability supernovae at

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-13

![](dt=2026-04-19/ht=09/774f860b7e3823fb3608b97d160df11338f28001256b24d8d3b21cfd57e60426.jpg)

![](dt=2026-04-19/ht=09/67a55d1aff8881498163755d90eb3166d15321433bee5459bc62bb1c4b44f210.jpg)

![](dt=2026-04-19/ht=09/5f51a9e3d7c1054892236df24280d4d159f8ce6358b66568d23abdab830875a0.jpg)

![](dt=2026-04-19/ht=09/0c35f3f92da030b83af02aa067307b05cd85439a96083c091ee314bc5c0268c5.jpg)

![](dt=2026-04-19/ht=09/d5f65f676bf9a114a357fe0b06decd667aa42ac050fefd99513f250ebcbd2b50.jpg)

approximately $60 - 120M_{\odot}$ [135-138]. The lowest-mass BBH systems, GW151226 and GW170608, have $90\%$ credible lower bounds on $m_2$ of $5.6M_{\odot}$ and $5.9M_{\odot}$ , respectively, and therefore lie above the proposed BH mass gap region [139-142] of $2 - 5M_{\odot}$ . The component masses of the BBHs show a strong degeneracy with each other. Lower-mass systems are dominated by the inspiral of the binary, and the component mass contours trace out a line of constant chirp mass Eq. (5) which is the best measured parameter in the inspiral [34,63,122]. Since higher-mass systems merge at a lower GW frequency, their GW signal is dominated by the merger of the binary. For high-mass binaries, the total mass can be measured with an accuracy comparable to that of the chirp mass [143-146].

We show posteriors for the ratio of the component masses Eq. (6) in the top left in Fig. 5. This parameter is much harder to constrain than the chirp mass. The width of the posteriors depends mostly on the SNR, and so the mass ratio is best measured for the loudest events, GW170817, GW150914, and GW170814. Even

though GW170817 has the highest SNR of all events, its mass ratio is less well constrained, because the signal power comes predominantly from the inspiral, while the merger contributes little compared to the BBH [147]. GW151226 and GW151012 have posterior support for more unequal mass ratios than the other events, with lower bounds of 0.28 and 0.29, respectively, at $90\%$ credible level.

The final mass, radiated energy, final spin, and peak luminosity of the BH remnant from a BBH coalescence are computed using averages of fits to numerical relativity (NR) results [15,148-153]. Posteriors for the mass and spin of the BH remnant for BBH coalescences are shown in the right in Fig. 4. Only a fraction (0.02-0.07) of the binary's total mass is radiated away in GWs. The amount of radiated energy scales with its total mass. The heaviest remnant BH found is GW170729, at $79.5_{-10.2}^{+14.7}M_{\odot}$ while the lightest remnant BH is GW170608, at $17.8_{-0.7}^{+3.4}M_{\odot}$ .

GW mergers reach extraordinary values of peak luminosity which is independent of the total mass. While it

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-14

![](dt=2026-04-19/ht=09/6a55820227f3315b595fe42c3a29a8816b70ae3967dcfdd07a98893fdb0296c4.jpg)

![](dt=2026-04-19/ht=09/876e60a4d6b866b5ae8a32317e0cd6a1cf37c31a9a2ff9ac85d6e0bfe885422f.jpg)

![](dt=2026-04-19/ht=09/87137e57e31beecdb9be2e95965906d4cb0859b0c04946a1e1aafa95f2ba5ebc.jpg)

![](dt=2026-04-19/ht=09/2ec70e563ce89d04b47c36b793c40590568ac27aa942ef47328e18c3ba850ff2.jpg)

![](dt=2026-04-19/ht=09/9d2bb3d45d6ca793e0cb28708c8ab7b1010e7a666ba1a7308209c8539e5010bc.jpg)

![](dt=2026-04-19/ht=09/fe4e2cd514facc7635e106d43946fc4e6ea2d52dd34753bda018012f963541ed.jpg)

![](dt=2026-04-19/ht=09/5318a635f9a4c3ef10dacc49b665e6d9e8945c9665a75ddd4b6fb443888ee019.jpg)

![](dt=2026-04-19/ht=09/d25f6c5b6cb2504ada49f26190ea5f9d80b7077faa90f27c2518983306307a73.jpg)

![](dt=2026-04-19/ht=09/13036a69ce41f596e2279243f2e8db4da9e8a3b8c39c69985b67dc15b8c50352.jpg)

![](dt=2026-04-19/ht=09/d551734385300f2d33a9da29a1db4111cdfe081043eb544bfea8a814b03575ae.jpg)

![](dt=2026-04-19/ht=09/455c51a40dd1a728170fea13eb52bc10ed33611b555113c7a892c9661e5f8fef.jpg)

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-15

![](dt=2026-04-19/ht=09/e9cbbdcabbdd3113f5ea90e2dace2257879d46fd9e202c7411532fa1c78d9b01.jpg)

![](image)
b/mineru_full_text/v0/result=success/type=image/dt=2026-04-19/ht=09//e34d3ffe0e873b48e7c34d71f1e1176063251058bc9f50b678554a9d9c7a7cd3.jpg)

![](dt=2026-04-19/ht=09/4404f00d06d3816a14ae076c2b983d1a0c24a8967183b3796facdcd03e58d03a.jpg)

![](dt=2026-04-19/ht=09/5be6d446dbcf0c9c8db700a1b5418e588464bab8e760976fc858c867d41e75ac.jpg)

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-16

![](dt=2026-04-19/ht=09/df2e75abe1056ba6f74c091bbdf95abb5615ac95a86f8405ba805aa8d4602f4b.jpg)

![](dt=2026-04-19/ht=09/1ad405b978aa0c31243fe2f69b0ce88b29fd6d8c89ffdcd76ceb2c5b827aba46.jpg)

![](dt=2026-04-19/ht=09/7130a0696b88d700714d347c97a1b3c623b4277ad93ca0efe2dc099cb55e7d73.jpg)

![](dt=2026-04-19/ht=09/6d7e11390b1cb72d5ee059f88d17f184203f9a15b157f3827ddb52dfbfc73728.jpg)

depends on the mass ratio and spins, the posteriors overlap to a large degree for the observed BBH events. Because of its relatively high spin, GW170729 has the highest value of $\ell_{\mathrm{peak}} = 4.2_{-1.5}^{+0.9}\times 10^{56}$ erg s $^{-1}$ .

# C. Spins

The spin vectors of compact binaries can a priori point in any direction. Particular directions in the spin space are easier to constrain, and we focus on these first. An averaged projection of the spins parallel to the Newtonian orbital

angular momentum of the binary can be measured best. This effective aligned spin $\chi_{\mathrm{eff}}$ is defined by Eq. (4). Positive (negative) values of $\chi_{\mathrm{eff}}$ increase (decrease) the number of orbits from any given separation to merger with respect to a nonspinning binary [38,154]. We show posterior distributions for this quantity in the top right in Fig. 5. Most posteriors peak around zero. The posteriors for GW170729 and GW151226 exclude $\chi_{\mathrm{eff}} = 0$ at $>90\%$ confidence, but see Sec. V F. As can be seen from Table III, the $90\%$ intervals are 0.11-0.58 for GW170729 and 0.06-0.38 for GW151226.

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-17

As shown in the bottom left in Fig. 5, the mass ratio and effective aligned spin parameters can be degenerate [122,128,155], which makes them difficult to measure individually. For lower-mass binaries, most of the waveform is in the inspiral regime, and the posterior has a shape that curves upwards towards larger values of $\chi_{\mathrm{eff}}$ and lower values of $q$ , exhibiting a degeneracy between these parameters. This degeneracy is broken for high-mass binaries for which the signal is short and is dominated by the late inspiral and merger [147].

For all observed binaries, the posteriors reach up to the equal mass boundary ( $q = 1$ ). With current detector sensitivity, it is difficult to measure the individual BH's spins [147,156-158], and, in contrast to $\chi_{\mathrm{eff}}$ , the posteriors of an antisymmetric mass-weighted linear combination of $\chi_{1}$ and $\chi_{2}$ are rather wide.

The remaining spin d.o.f. are due to a misalignment of the spin vectors with the normal to the orbital plane and give rise to a precession of the orbital plane and spin vectors around the total angular momentum of the binary. The bottom right in Fig. 5 shows posterior and prior distributions for the quantity $\chi_{p}$ , which encapsulates the dominant effective precession spin. The prior distribution for $\chi_{p}$ is induced by the spin prior assumptions (see Appendix B 1 and C).

Since $\chi_{p}$ and $\chi_{\mathrm{eff}}$ are correlated, we show prior distributions conditioned on the $\chi_{\mathrm{eff}}$ posteriors. The $\chi_{p}$ posteriors are broad, covering the entire domain from 0 to 1, and are overall similar to the conditioned priors. A more detailed representation of the spin distributions is given in Fig. 6, showing the probability of the spin magnitudes and tilt angles relative to the Newtonian orbital angular momentum. Deviations from uniformity in the shading indicate the strength of precession effects.

Overall, it is easier to measure the spin of the heavier component in each binary [147,157]. None of the GW events exhibit clear precession. To quantify this result, we compute the Kullback-Leibler divergence $D_{\mathrm{KL}}$ [159] for the information gain from the $\chi_{p}$ prior to the posterior. Since $\chi_{p}$ and $\chi_{\mathrm{eff}}$ are correlated, we can condition the prior on the $\chi_{\mathrm{eff}}$ posterior before we compute $D_{\mathrm{KL}}^{\chi_p}$ . We give $D_{\mathrm{KL}}^{\chi_p}$ values with and without conditioning in Table V in Appendix B.

Among the BBH events, the highest values of $D_{\mathrm{KL}}^{\chi_p}$ are found for GW170814 ( $0.13_{-0.02}^{+0.03}$ bits) and GW151226 ( $0.12_{-0.02}^{+0.05}$ bits). In all cases, the information gain for $D_{\mathrm{KL}}^{\chi_p}$ is much less than a single bit. Conversely, we gain more than one bit of information in $\chi_{\mathrm{eff}}$ for several events (see Table V and also Table VI). For very well measured quantities such as the chirp mass for GW170817, we can gain approximately 10 bits of information and come close to the information entropy [160] in the posterior data.

A clear imprint of precession could also help break the degeneracy between the mass ratio and effective aligned spin [161-164]. We discuss the influence of the choice of priors for spin parameters (and distance) in Appendix C.

As a weighted average of the mass and aligned spin of the binary, $\chi_{\mathrm{eff}}$ provides a convenient tool to test models of

compact object binary spin properties via GW measurements [165-170]. Several authors suggest [171-179] how stellar binary evolutionary pathways leave imprints on the overall distribution of detected parameters such as masses and spins. By inferring the population properties of the events observed to date [55], we disfavor scenarios in which most black holes merge with large spins aligned with the binary's orbital angular momentum. With more detections, it will be possible to determine, for example, if the BH spin is preferentially aligned or isotropically distributed.

For comparable-mass binaries, the spin of a remnant black hole comes predominantly from the orbital angular momentum of the progenitor binary at merger. For nonspinning equal-mass binaries, the final spin of the remnant is expected to be approximately 0.7 [180-184]. The final spin posteriors are more precisely constrained than the component spins and also the effective aligned spin $\chi_{\mathrm{eff}}$ . Masses and spins of the final black holes are shown in Fig. 4. Except for GW170729 with its sizable positive $\chi_{\mathrm{eff}} = 0.37_{-0.25}^{+0.21}$ , the medians of all final spin distributions are around approximately 0.7. The remnant of GW170729 has a median final spin of $a_{f} = 0.81_{-0.13}^{+0.07}$ and is consistent with 0.7 at $90\%$ confidence.

# D. Distance, inclination, and sky location

The luminosity distance $d_{L}$ of a GW source is inversely proportional to the signal's amplitude. Six BBH events (GW170104, GW170809, GW170818, GW151012, GW170823, and GW170729) have median distances of about a Gpc or beyond, the most distant of which is GW170729 at $d_{L} = 2840_{-1360}^{+1400}$ Mpc, corresponding to a redshift of $0.49_{-0.21}^{+0.19}$ . The closest BBH is GW170608, at $d_{L} = 320_{-110}^{+120}$ Mpc, while the BNS GW170817 is found at $d_{L} = 42_{-13}^{+6}$ Mpc.

The significant uncertainty in the luminosity distance stems from the degeneracy between the distance and the binary's inclination, inferred fr
om the signal amplitude [128,186,187]. We show joint posteriors of the luminosity distance and inclination $\theta_{JN}$ in the left in Fig. 7. In general, the inclination angle is only weakly constrained, and for most events it has a bimodal distribution around $\theta_{JN} = 90^{\circ}$ with greatest support for the source being either face on or face off (angular momentum pointed parallel or antiparallel, respectively, to the line of sight).

These orientations produce the greatest gravitational-wave amplitude and so are consistent with the largest distance. For GW170817, the $\theta_{JN}$ distribution has a single mode. For GW170809, GW170818, and GW150914, the $90\%$ interval contains only a single mode so that the fifth percentile lies above $\theta_{JN} = 90^{\circ}$ .

Orientations of the total orbital angular momentum that are strongly misaligned with the line of sight are, in general, disfavored due to the weaker emitted GW signal compared to observing a binary face on $(\theta_{JN} = 0^{\circ})$ or face off $(\theta_{JN} = 180^{\circ})$ . For GW170818, the misalignment is more likely, with the probability that $45^{\circ} < \theta_{JN} < 135^{\circ}$ being 0.38.

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-18

This probability is less than 0.36 for all other events. An inclination close to $\theta_{JN} = 90^{\circ}$ would enhance subdominant modes in the GW signal but also result in a weaker emitted signal and, to compensate, a closer source. A more precise measurement of the inclination will be possible for strongly precessing binaries [162,188].

This analysis assumes that the emitted GW signal is not affected by gravitational lensing. Lensing would make GW mergers appear closer than they are and reduce their inferred, redshift-corrected source-frame masses, depending on the true distance and magnification factor of the lens. Motivated by the heavy BBHs observed by LIGO and Virgo, Ref. [189] claims that four of the published BBH observations are magnified by gravitational lensing. On the other hand, it has been pointed out that at LIGO's and Virgo's current sensitivities it is unlikely but not impossible that one of the GWs is multiply imaged. The analysis in Reference [190] concludes that lensing by massive galaxy clusters of one of our BBH GW detections can be rejected at the $4\sigma$ level.

In the right in Fig. 7, we show the joint posterior between the luminosity distance (or redshift) and source-frame chirp mass. We see that overall luminosity distance and chirp mass are positively correlated, as expected for unlensed BBHs observations.

An observed GW signal is registered with different arrival times at the detector sites. The observed time delays and amplitude and phase consistency of the signals at the sites allow us to localize the signal on the sky [191-193]. Two detectors can constrain the sky location to a broken annulus [194-197], and the presence of additional detectors in the network improves localization [19,198-200]. Figure 8 shows the sky localizations for all GW events. Both panels show posteriors in celestial coordinates which indicate the origin of the signal.

In general, the credible regions of sky position are made up of a collection of disconnected components determined by the pattern of sensitivity of the individual detectors. The top shows localizations for confidently detected $O2$ events that were communicated to EM observers and are discussed further in Ref. [22]. The results for the credible regions and sky areas are different from those shown in Ref. [22] because of updates in the data calibration and choice of waveform models. The bottom shows localizations for $O1$ events, along with $O2$ events not previously released to EM observers.

The sky area is expected to scale inversely with the square of the SNR [20,197]. This trend is followed for events detected by the two LIGO detectors. Several events (GW170729, GW170809, GW170814, GW170817, and GW170818) are observed with the two LIGO detectors and Virgo, which improves the sky localization [201]. The SNR contributed by Virgo can significantly shrink the area. We find the smallest $90\%$ sky localization areas for GW170817: $16\mathrm{deg}^2$ and GW170818: $39\mathrm{deg}^2$ .

# E. GW170817

We carry out a reanalysis of GW170817 using a set of waveform models including tidal effects described in detail in Appendix B 2. This analysis follows the one performed in Ref. [97], employs the same settings, but uses the recalibrated $O2$ data. We restrict the sky location to the known position of SSS17a/AT 2017gfo as determined by electromagnetic observations [21]. When computing the source-frame masses from the detector-frame masses, we use the redshift for NGC 4993 from Ref. [94] and its associated uncertainties.

Updated posteriors for masses and the effective tidal deformability parameter $\tilde{\Lambda}$ are shown in Fig. 9. For the results presented here, we allow the tidal parameters to vary independently rather than being determined by a common equation of state [202]. Results are consistent with those presented previously in Ref. [97] with slight differences in the derived tidal deformability, discussed below. Posterior distributions for SEOBNRv4T and TEOBResumS are obtained from RAPIDPE.

In contrast to the BBH events discussed above, GW170817 is completely dominated by the inspiral phase of the binary coalescence. The merger and postmerger happen at frequencies above $1\mathrm{kHz}$ , where LIGO and Virgo are less sensitive. The distributions of component masses are shown in the top in Fig. 9. With $90\%$ probability, the mass of the larger NS $m_{1}$ for the IMRPhenomPv2NRT model is contained in the range $[1.36, 1.84]M_{\odot}$ ([1.36, 1.58] $M_{\odot}$ ) and the smaller NS $m_{2}$ in $[1.03, 1.36]M_{\odot}$ ([1.18, 1.36] $M_{\odot}$ ) for the high-spin (low-spin) prior.

In Fig. 5, we show contours for the mass ratio and aligned effective spin posteriors for the IMRPhenomPv2NRT model assuming the high-spin prior. The results are consistent with those presented in Ref. [97]. The effective precession spin $\chi_{p}$ shown in the bottom right in Fig. 5 peaks at lower values than the prior, and the KL divergence $D_{\mathrm{KL}}^{\chi_p}$ between this prior and the posterior is $0.19_{-0.03}^{+0.04}$ bits.

When conditioning the prior on the measured $\chi_{\mathrm{eff}}$ , $D_{\mathrm{KL}}^{\chi_p}$ decreases to $0.07_{-0.02}^{+0.01}$ bits, providing very little evidence for precession. The strongly constrained $\chi_{\mathrm{eff}}$ restricts most of the spin d.o.f. into the orbital plane, and in-plane spins are large only when the binary's inclination angle approaches $180^\circ$ , where they have the least impact on the waveform.

We show marginal posteriors for the effective tidal parameter $\tilde{\Lambda}$ in the bottom in Fig. 9. The prior and posterior for $\tilde{\Lambda}$ go to zero as $\tilde{\Lambda} \to 0$ because of the flat prior on the component deformability parameters $\Lambda_{1}$ and $\Lambda_{2}$ . We reweight the posterior for $\tilde{\Lambda}$ by dividing by the prior used, effectively imposing a flat prior in $\tilde{\Lambda}$ . The reweighted posterior has nonzero support at $\tilde{\Lambda} = 0$ .

We find bounds on the effective tidal parameter that are about $10\%$ wider compared to the results presented in Ref. [97]. For the high-spin prior, the $90\%$ upper limit on the tidal parameter is 686 for IMRPhenomPv2NRT, compared to the value of 630 found in Ref. [97]. The upper limit for SEOBNRv4NRT is

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-19

very close, 664, and the value for TaylorF2 is higher at 816. For SEOBNRv4T and TEOBResumS, we find 843 and 841, respectively. For the low-spin prior, we quote the two-sided $90\%$ highest posterior density (HPD) credible interval on $\tilde{\Lambda}$ that does not contain $\tilde{\Lambda} = 0$ . This $90\%$ HPD interva
l is the smallest interval that contains $90\%$ of the probability. For IMRPhenomPv2NRT, we obtain $\tilde{\Lambda} = 330_{-251}^{+438}$ , which is slightly higher than the interval $300_{-230}^{+420}$ found in Ref. [97].

For SEOBNRv4NRT, we find $\tilde{\Lambda} = 305_{-241}^{+432}$ and for TaylorF2 $394_{-321}^{+557}$ . For SEOBNRv4T and TEOBResumS, we find $349_{-349}^{+394}$ and $405_{-375}^{+545}$ , respectively. The posteriors produced by these two models agree better for the low-spin prior. This result is consistent with the very good agreement between the models for small spins $|\chi_i| \leq 0.15$ shown in Ref. [33]. For reference, we also show contours for a representative subset of theoretical EOS models given by piecewise-polytrope fits from Ref. [203].

These fits are evaluated using the IMRPhenomPv2NRT component mass posteriors, and the sharp cutoff to the right of each EOS posterior corresponds to the equal mass ratio boundary. As found in Ref. [97], the EOSs MS1, MS1b, and H4 lie outside the $90\%$ credible upper limit and are therefore disfavored.

In Table III, we quote conservative estimates of key final-state parameters for GW170817 obtained from fits to NR simulations of quasicircular binary neutron star mergers [204-206]. We do not assume the type of final remnant and quote quantities at either the moment of merger or after the postmerger GW transient. Lower limits of radiated energy up to the merger and peak luminosity are given at $1\%$ credible level. The final mass is computed from the radiated energy including the postmerger transient as an upper limit at $99\%$ credible level. For the final angular momentum, we quote an upper bound computed from the radiated energy and using the phenomenological universal relation found in Ref. [204].

# F. Comparison against previously published results

We compare PE results between the original published $O1$ and $O2$ analyses for GW150914, GW151012, GW151226, GW170104, GW170608, and GW170814 and the reanalysis performed here. The values presented here supersede previously published results. For some events, we see differences in the overall posteriors that are due to a different choice of waveform models that have been combined. This difference is especially the case when comparing against previous results that combine samples between spin-aligned and effective precession models and mostly affects spin parameters. We first mention differences that are apparent when comparing results from the same waveform models in the original analysis and the reanalysis.

The source-frame total mass is consistent with the original analysis. For GW150914, we find an increase in

the median of about $1M_{\odot}$ in this reanalysis when comparing between the same precessing waveform models because of the improved method for computing the power spectral density of the detector noise and the use of frequency-dependent calibration envelopes. For GW170104, we find the median of the total mass to be $0.3M_{\odot}$ higher because of the recalibration of the data and the noise subtraction.

Similarly, we find an increase of about $0.2M_{\odot}$ in the total mass for GW151012 and a decrease in the total mass of about $0.3M_{\odot}$ for GW151226 and $0.2M_{\odot}$ for GW170608 in the reanalysis. The mass ratio and effective spin parameters are broadly consistent with the original analysis. GW170104 especially benefits from the noise subtraction. This subtraction increases the matched-filter SNR recovered by the parameter-estimation analysis from $13.3_{-0.3}^{+0.2}$ to $14.0_{-0.3}^{+0.2}$ . The increase in SNR results in reduced parameter uncertainties [128].

For the effective spin parameter, the tightening of the posterior results in the loss of the tail at low values. The inferred value changes from $-0.12_{-0.30}^{+0.21}$ to $-0.04_{-0.21}^{+0.17}$ ; the upper limit remains about the same, and there is still little support for large aligned spins. For GW151226, we find from using the fully precessing model that the inferred effective aligned spin is $0.15_{-0.11}^{+0.25}$ , and with the effective precession model it is $0.20_{-0.08}^{+0.18}$ .

The fully precessing model has some support at $\chi_{\mathrm{eff}} = 0$ ; the probability that $\chi_{\mathrm{eff}} < 0$ is, however, $< 0.01$ . We find with $99\%$ probability that at least one spin magnitude is greater than 0.28 compared to the value 0.2 in the $O1$ analysis. We discuss further differences between results obtained from the two BBH waveform models in Appendix B 2.

# VI. WAVEFORM RECONSTRUCTIONS

In the previous section, we present estimates of the source properties for each event based on different relativistic models of the emitted gravitational waveform. Such models, however, do not necessarily incorporate all physical effects. Here, we take an independent approach to determine the GW signal present in the data and assess the consistency with the waveform model-based analysis.

Figure 10 shows the time-frequency maps of the gravitational-wave strain data measured in the detector where the higher SNR is recorded [207], as well as three different types of waveform reconstructions for all GW events from BBHs. Two of those waveform reconstructions provide an independent estimate of the most probable signal: Instead of relying on waveform models, these algorithms exclusively use the coherent gravitational-wave energy measured by the detector network, requiring only weak assumptions on the form of the signal for the reconstruction.

The first method, BAYESWAVE, represents the waveform as a sum of sine-Gaussian wavelets $h(\vec{\lambda}; t) = \sum_{j=1}^{N} \Psi(\vec{\lambda}_j; t)$ , where the number of wavelets used in the reconstruction, $N$ , and the parameters describing each wavelet, $\vec{\lambda}_j$ , are

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-20

![](dt=2026-04-19/ht=09/3a33373e0ef34c1a4fd10040243b83c491cc0e015f39e7f9ca4a82c1638e4bef.jpg)

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-21

explored by a transdimensional Markov chain Monte Carlo algorithm [53] (blue bands in the top panels). In comparison, we also show the waveforms obtained from the Bayesian inference results in Sec. V with the precessing waveform model PhenomP (red bands). The $90\%$ credible regions are computed by selecting a discrete collection of times $t_k$ , and computing the waveform at these times from a large number of fair draws from the posterior distribution of the model parameters; they indicate the range of waveforms that are consistent with the data for a particular model.

The wavelet-based method trades sensitivity for flexibility and, therefore, is unable to discern the early inspiral or ringdown part of the CBC waveforms where the signals are weaker and/or spread out over time. The CBC waveform models use strict assumptions about the shape of the waveforms and, based on those assumptions, predict the shape of the signal in places where it is weak compared to the detector noise. Comparing results from the different methods enables a visual assessment of the reconstructed signals, both with and without physical assumptions about the source.

Regions where the $90\%$ credible bands are not overlapping do not necessarily imply any physical discrepancy, instead arising from differences in the models for the GW signals. Furthermore, as can be seen in several panels in Fig. 10, the $90\%$ credible intervals for the BAYESWAVE reconstructions sometimes include features that are not present in the template-based reconstructions. We verify that these outliers are absent from the BAYESWAVE $50\%$ credible intervals (not shown here), indicating that they have low significance.

Analogous features are seen when reconstructing simulated signals added to real data and should not be misinterpreted as evidence for disagreement w
ith the template-based reconstructions. Their origin is more mundane: They are caused by small random coherent features in the noise that are seen by BAYESWAVE; similar behavior is also seen for cWB, the second reconstruction method used (see below).

Quantitative comparisons of the template- and wavelet-based methods include computing overlaps between the reconstructions and reanalyzing the data with the wavelet-based analysis after the best-fit CBC waveform has been subtracted. Both comparisons agree within expectations from analysis of simulations, and we conclude that there are no detectable discrepancies between the wavelet- and template-based reconstructions, i.e., that the template-based methods agree with the data within the uncertainties.

The bottom panels show the waveform reconstructions obtained from the second model-independent method, cWB, which reconstructs the maximum likelihood signal waveforms $\pmb{h} = \{h_{H}(t), h_{L}(t)\}$ triggered in multiple detectors by a GW event. Since the reconstruction does not use any specific waveform model, the components of $\pmb{h}$ are effectively the denoised detector responses normalized by the noise power spectral amplitude. To test the consistency between the cWB reconstructed signals $\pmb{h}$ and the CBC

parameter estimation results in Sec. V, the following method is used: (a) Waveforms from the CBC parameter estimation posterior samples are generated and injected at random times in the data around the GW event, (b) the cWB pipeline is run on these data and $\tilde{h}$ —the best-fit waveforms expected for a given waveform model—are reconstructed, and (c) the waveforms $\tilde{h}$ are used to construct the confidence intervals. This method combines both the CBC and cWB reconstruction errors to produce confidence intervals for the denoised signal waveforms obtained with cWB.

In addition, the confidence interval is robust to non-Gaussian detector noise, which may affect both the CBC and cWB reconstructions. Figure 10 shows the maximum likelihood waveforms reconstructed by cWB: A comparison with the $90\%$ confidence interval indicates good agreement with the model-based parameter estimation results.

Further reconstructions are made available through the Gravitational Wave Open Science Center [57].

# VII. MERGER RATES OF COMPACT BINARY SYSTEMS

This section presents bounds on the astrophysical merger rates of BNS, NSBH, and BBH systems in the local Universe, derived from the search results presented in earlier sections. These bounds supersede earlier estimates and limits from previous LIGO-Virgo results [4,15,18,208]. Our merger rate estimates are derived by modeling the search results of each pipeline as a mixture of a set of astrophysical events and a set of background (noise) events of terrestrial origin [90]. Here, we describe how merger rates, as well as the probabilities that each candidate event is of astrophysical or terrestrial origin, are calculated.

Since we now have confident detections of different astrophysical event types—BNS and BBH—a more sophisticated treatment is necessary as compared to previous results. We define four categories: one terrestrial and three astrophysical categories (BNS, NSBH, and BBH). For each category, the distribution of the pipelines' ranking statistic values—generally denoted here as $x$ —is empirically determined.

Terrestrial quantities are denoted by the $T$ label—thus, the probability distribution of terrestrial event $x$ values is written $p(x|T)$ —while the set of astrophysical categories is denoted by $A_i$ , where $i$ labels the three types of binaries considered. For a given category $A_i$ and a population configuration $\{\theta\}$ , i.e., the assumed spin and mass distribution, the ranking statistic distribution is given as $p(x|A_i, \{\theta\})$ .

In practice, differing $\{\theta\}$ do not significantly affect the shape of $p(x|A_i, \{\theta\})$ over the range of ranking statistics considered here [19].

All four categories are assumed to contribute events according to a Poisson process with mean $\Lambda$ . In each astrophysical category, the mean can be further described as the product of the accessible spacetime volume $\langle VT\rangle$ for a

B.P.ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-22

source population $\{\theta\}$ and the astrophysical rate density $R_{i}$ . The terrestrial $\Lambda_{T}$ and astrophysical $\Lambda_{i}$ count parameters are determined by fitting the mixture of $\Lambda_{T}p(x|T)$ and a given $\Lambda_{i}p(x|A_{i},\{\theta\})$ . Since each model is computed from the outputs of a given search, for the purposes of computing quantities such as the probability of astrophysical origin, each pipeline is treated separately.

Figure 11 shows the resulting astrophysical foreground and terrestrial background models, as well as the observed number of events above a ranking statistic threshold: on the left, PyCBC results, restricted to events with masses compatible with a BBH, with chirp mass $>4.35M_{\odot}$ (so that BNS candidate events including GW170817 are not plotted); on the right, GstLAL results including all events, with the signal counts summed over the three astrophysical categories BNS, NSBH, and BBH.

In both searches, the background model falls exponentially with the detection statistic, with no non-Gaussian tails. The different detection statistic used in the PyCBC and GstLAL searches leads to differently shaped signal models. However, both searches show agreement between the search results and the sum of the foreground and terrestrial background models. In both panels in Fig. 11, we see three regions: At the high-ranking statistic threshold, the signal model dominates and the observed events are inconsistent with terrestrial noise; at the low-ranking statistic threshold, the noise model

dominates and the observed events are fully consistent with terrestrial noise; and a narrow intermediate region where both noise and signal models are comparable, and the observed events are consistent with the sum of the two models. The list of marginal events in Tables II and IV come from that narrow intermediate region.

The accessible spacetime volume $\langle VT\rangle$ is estimated by injecting synthesized signals with parameters drawn from $\{\theta\}$ and recovering them using the search pipeline. For all $\{\theta\}$ , the injections are assumed to be uniformly distributed in the comoving volume. Then the detection efficiency over redshift $f(z|\{\theta\})$ derived from the recovery campaign measures the fraction of the differential volume $dV/dz$ which is accessible to the network:

$$
\langle V T \rangle_ {\{\theta \}} = T _ {\mathrm {o b s}} \int_ {0} ^ {\infty} f (z | \{\theta \}) \frac {d V}{d z} \frac {1}{1 + z} d z. \tag {8}
$$

The total $\langle VT\rangle$ is then the product of the accessible volume for a given population with the observational time $T_{\mathrm{obs}}$ . The angle brackets indicate that the volume is averaged over members of the population drawn from $\{\theta\}$ . In the following, we suppress the $\{\theta\}$ dependence on $\langle VT\rangle$ and $p(x|A_i)$ and, instead, indicate specific populations where they are relevant. The factor of $1 + z$ arises from the

![](dt=2026-04-19/ht=09/8b25d71d26d87b9d13585f519ed9c77f1bedaca368d567431e39c90999fb06c1.jpg)

![](dt=2026-04-19/ht=09/0706140cf91f9edc2c1e480b7e5e8b4c70bfb374998b07afab25894cc1091029.jpg)

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-23

conversion of source-frame time to detector-frame time which is integrated to obtain $T_{\mathrm{obs}}$ . $\langle VT\rangle$ is estimated over smaller periods of observation time to better account for time-varying detector sensitivity changes. When summed over analysis pe
riods, the total observation time is 0.46 y.

Additional details on how $\langle VT\rangle$ is measured can be found in Refs. [4,209]. A more generalized approach [210] to obtaining $\langle VT\rangle$ is used by PyCBC.

Up to a normalization constant, the joint rate posterior has the form of

$$
p (\{R \}, \Lambda_ {T}, \{\langle V T \rangle \} | \{x \}) = p (\{R \}, \Lambda_ {T}, \{\langle V T \rangle \}) \prod_ {\mu} \left\{\Lambda_ {T} p (x _ {\mu} | T) + \sum_ {i} R _ {i} \langle V T \rangle_ {i} p (x _ {\mu} | A _ {i}) \right\} \exp \left(- \Lambda_ {T} - \sum_ {i} R _ {i} \langle V T \rangle_ {i}\right), \tag {9}
$$

where $p(\{R\}, \Lambda_T, \{\langle VT\rangle\})$ is the joint prior density and $\mu$ enumerates the event candidate ranking statistics for each search. We single out the terrestrial class, since we measure only its overall count $\Lambda_T$ and are otherwise uninterested in its properties as an event rate density. To account for statistical and calibration uncertainty in $\langle VT\rangle$ , all searches marginalize over an 18% relative uncertainty incorporated into the prior on $\langle VT\rangle$ .

To quantify our uncertainty on the mass and spin distributions (encoded in $\{\theta\}$ ) of the source populations, we examine different populations. As mentioned before, $f(z|\{\theta\})$ is mostly unaffected by these choices, but the $\langle VT\rangle$ estimated is strongly influenced by the assumed population. Thus, a separate rate posterior is obtained for each population tested, and we also quote the union of population-specific credible intervals as the overall rate interval.

Previous GW BBH and BNS event rate distributions treat each source category independently—event candidates are assigned a category $a$ priori based on the properties measured by a given search. In the following, PyCBC and cWB retain this approach. In addition to the rates derived in this way, we also present the results from an enhanced model jointly treating multiple astrophysical event categories [89]. This enhancement accounts for correlations between the source categories by measuring the response of the GstLAL search template banks to the astrophysical populations.

We expect that the methods should agree for two reasons: negligible correlation between the BBH and BNS category and no significant candidates in the intervening NSBH category. Since an inspection of the corresponding rate posteriors confirms this expectation, we present a single search-combined posterior for both BBH populations. In the BNS category, the rates are also compatible between searches, but, because they are derived from differing methods, we present the BNS rate posteriors separately.

We update the event rates for the BBH and BNS categories with the additional events and observing time. In addition to the two categories with confirmed detections, we also revisit the NSBH category. In contrast to the BNS and BBH categories, there are no confident detections in the NSBH spaces (see Tables I and II and the absence of significant candidates in Table IV). Hence, we update the

upper limits on the NSBH event rate from $O1$ . Instead of using $O1$ or earlier detections as a prior on the $O2$ measurement, we reanalyze $O1$ and $O2$ as a whole and use an uninformative prior on the result. As in Ref. [4], we use the Jeffreys prior for a Poisson rate parameter, proportional to $R_{i}^{-1/2}$ , for BNS and BBH, while for NSBH we use a prior uniform in $R_{i}$ which yields a conservative upper limit bound.

# A. Event classification

To determine the probability that a given candidate originated in one of the four categories, the models are marginalized over the counts with the ranking statistic distributions fixed at the value of the ranking statistic of the candidate. The distribution that is marginalized is the ratio of the category under consideration versus all categories (including terrestrial):

$$
\begin{array}{l} p _ {A _ {i}} \left(x _ {\mu} \mid \{x \}\right) = \int p \left(\left\{R \right\}, \Lambda_ {T}, \{\langle V T \rangle \} \mid \{x \}\right) \\ \times \frac {R _ {i} \langle V T \rangle_ {i} p \left(x _ {\mu} \mid A _ {i}\right)}{\Lambda_ {T} p \left(x _ {\mu} \mid T\right) + \sum_ {j} R _ {j} \langle V T \rangle_ {j} p \left(x _ {\mu} \mid A _ {j}\right)} \\ \times d \{R \} d \Lambda_ {T} d \{\langle V T \rangle \}. \tag {10} \\ \end{array}
$$

Thus, we obtain $p_{\text{terrestrial}}, p_{\text{BBH}}, p_{\text{BNS}}$ , and $p_{\text{NSBH}}$ , which are mutually exclusive categorizations. The overall probability of astrophysical origin sums the expression over all categories in $\{A\}$ .

We expect different values of $p_{A_i}$ to be assigned to any given event by different search pipelines. This assignment is due to differences in the averaged efficiency of various methods to discriminate signal from noise events and also to the effects of random noise fluctuations on the ranking statistics assigned to a specific event. We also expect systematic uncertainties in the quoted probabilities due to our lack of knowledge of the true event populations, for instance, the mass distribution of BNS and NSBH mergers.

Parameter estimation is not performed on all candidates used to obtain rate estimates, so only the search masses and rankings are used to derive the astrophysical probabilities. Table IV shows the per-pipeline assigned probability values for each of the relevant categories. The cWB search does

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-24

not have a specific event type corresponding to NSBH or BNS; thus, we treat all cWB search events as BBH candidates. The astrophysical probabilities from PyCBC are estimated by applying simple chirp mass cuts to the set of events with ranking statistic $\rho > 8$ : Events with $\mathcal{M} < 2.1$ are considered as candidate BNS, those with $\mathcal{M} > 4.35$ as candidate BBH, where the lower bound assumes two $5M_{\odot}$ BHs, and all remaining events as potential NSBH.

We note that the value of the boundary between NSBH and BBH is chosen somewhat arbitrarily, given the uncertainty as to the exact value in our current understanding due to, for example, the formation and environment of the source. The astrophysical probabilities from GstLAL in Table IV are estimated using the pipeline response to injected synthetic signals, where neutron stars are assumed to have masses in the range $1 - 3M_{\odot}$ and black holes are assumed to have masses of $3M_{\odot}$ or larger. The details can be found in Ref. [89].

We note that the different definitions used by these three pipelines in classifying events as BNS, NSBH, or BBH reflect current astrophysical uncertainties in such classifications. Other, yet different definitions are used in order to compute event rates in the following subsections.

# B. Binary black hole event rates

After the detection of GW170104, the event rate of BBH mergers had been measured to lie between 12 and $213\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ [15]. This measurement included the four events identified at that time. The $\langle VT\rangle$ , and hence the rates, are derived from a set of assumed BBH populations. In $O1$ two distributions of the primary massone uniform in the log and one a power law $p(m_{1})\propto m_{1}^{-\alpha}$ with an index of $\alpha = 2.3$ -are used as representative extremes. In both populations shown here, the mass distribution cuts off at a lower mass of $5M_{\odot}$ .

The mass distributions cut off at a maximum mass of $50M_{\odot}$ . The detector network is sensitive to binaries with a larger mass; however, the new cutoff is motivated by both more sophisticated modeling of the mass spectrum [55] preferring maximum BH masses much smaller than the previous limit of $100M_{\odot}$ , as well as astrophysical processes which are expected to truncate the distribution [136]. The BH spin distribution has magnitude uniform in [0, 1].

The PyCBC search uses a spin tilt
distribution which is isotropic over the unit sphere, and GsTLAL uses a distribution that aligns BH spins to the orbital angular momentum.

The posteriors on the rate distributions are shown in Fig. 12. Including all events, the event rate is now measured to be $R = 56_{-27}^{+44}\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ (GstLAL) and $R = 57_{-29}^{+47}\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ (PyCBC) for the power-law distribution. For the uniform in log distribution, we obtain $R = 18.1_{-8.7}^{+13.9}\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ (GstLAL) and $R = 19.5_{-9.7}^{+15.2}\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ (PyCBC). The difference in $\langle VT\rangle$ and rate distributions between the two spin populations is smaller than the

uncertainty from calibration. Therefore, we present in Fig. 12 the rate distribution for both assumed mass distributions, combined over searches as an averaging over the spin configurations. The union of the intervals combined over both populations lies in $9.7 - 101\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ . GW170608 is included in the estimation of $\Lambda$ for BBH, but, given difficulties in characterizing the amount of time in which it could have occurred, its analysis period is not included in the overall $\langle VT\rangle$ . We believe this exclusion introduces a bias that is no larger than the already accounted for calibration uncertainty.

A more detailed analysis [4] previously showed that both of the assumed populations used here are consistent with an inferred fit to the power-law index $\alpha$ as measured from the population of events known at the time. An update to this analysis using all current detections and examining a variety of plausible mass and spin distributions is explored in Ref. [55]. Allowing for a self-consistent fit to the event rate while varying a power-law model with a spectral index and maximum and minimum primary mass, the rate interval is found to be $53_{-28}^{+56}\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ .

This result is consistent with the intervals obtained from the fixed parameter populations used here. Within the same model, we obtain a $90\%$ interval of the distribution for the power-law index of $\alpha = 1.3_{-1.7}^{+1.4}$ . Compared with the earlier analysis [4], this result favors somewhat shallower power-law indices.

# C. Binary neutron star event rates

The discovery of GW170817 is the only unambiguous BNS candidate obtained in $O2$ . Regardless, it provides a means to independently measure the rate of binary neutron star mergers. Previous estimates [211-213] from observations are derived from the properties of neutron star binaries

![](dt=2026-04-19/ht=09/bd1f5accf7d259cc3910966e1e5a8627941d159b468d0a426f5a6cb48b47949f.jpg)

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-25

TABLE IV. This table lists the probability of each event candidate belonging to a given source category. Each category is further delineated by the probabilities derived from each search pipeline's output. Where candidate values are not indicated, the search did not calculate a probability for that category or did not have a candidate with a ranking statistic sufficient for interest. $p_{A_i}$ values below $10^{-3}$ are shown as zero. The astrophysical category is the sum over the BNS, NSBH, and BBH categories (where available), and the sum over astrophysical and terrestrial is unity.

![](dt=2026-04-19/ht=09/89c60e1cb7d20d0ec4504a218f9d02cc78028df04e7b5cd21dff48ac766d56b7.jpg)

<table><tr><td rowspan="2"></td><td colspan="5">GstLAL</td><td colspan="5">PyCBC</td><td colspan="2">cWB</td></tr><tr><td>Terrestrial</td><td>BNS</td><td>NSBH</td><td>BBH</td><td>Astrophysical</td><td>Terrestrial</td><td>BNS</td><td>NSBH</td><td>BBH</td><td>Astrophysical</td><td>Terrestrial</td><td>BBH</td></tr><tr><td>GW150914</td><td>0</td><td>0</td><td>0.0064</td><td>0.99</td><td>1</td><td>0</td><td>...</td><td>...</td><td>1</td><td>1</td><td>0</td><td>1</td></tr><tr><td>151008a</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>0.73</td><td>...</td><td>...</td><td>0.27</td><td>0.27</td><td>...</td><td>...</td></tr><tr><td>151012.2</td><td>0.98</td><td>0.022</td><td>0.0012</td><td>0</td><td>0.023</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>GW151012</td><td>0.001</td><td>0</td><td>0.031</td><td>0.97</td><td>1</td><td>0.04</td><td>...</td><td>...</td><td>0.96</td><td>0.96</td><td>...</td><td>...</td></tr><tr><td>151116b</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>~1</td><td>&lt;0.05</td><td>...</td><td>...</td><td>&lt;0.5</td><td>...</td><td>...</td></tr><tr><td>GW151226</td><td>0</td><td>0</td><td>0.12</td><td>0.88</td><td>1</td><td>0</td><td>...</td><td>...</td><td>1</td><td>1</td><td>0.05</td><td>0.95</td></tr><tr><td>161202</td><td>0.97</td><td>0.034</td><td>0</td><td>0</td><td>0.034</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>161217</td><td>0.98</td><td>0</td><td>0.011</td><td>0.0078</td><td>0.018</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>GW170104</td><td>0</td><td>0</td><td>0.0028</td><td>1</td><td>1</td><td>0</td><td>...</td><td>...</td><td>1</td><td>1</td><td>0</td><td>1</td></tr><tr><td>170208</td><td>0.98</td><td>0</td><td>0.011</td><td>0.0088</td><td>0.02</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>170219</td><td>0.98</td><td>0.019</td><td>0</td><td>0</td><td>0.02</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>170405</td><td>1</td><td>0.004</td><td>0</td><td>0</td><td>0.004</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>170412</td><td>0.94</td><td>0</td><td>0.029</td><td>0.032</td><td>0.06</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>170423</td><td>0.91</td><td>0.086</td><td>0</td><td>0</td><td>0.086</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>GW170608</td><td>0</td><td>0</td><td>0.084</td><td>0.92</td><td>1</td><td>0</td><td>...</td><td>...</td><td>1</td><td>1</td><td>0</td><td>1</td></tr><tr><td>170616b</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>~1</td><td>...</td><td>&lt;0.5</td><td>...</td><td>&lt;0.5</td><td>...</td><td>...</td></tr><tr><td>170630</td><td>0.98</td><td>0.02</td><td>0</td><td>0</td><td>0.02</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>170705</td><td>0.99</td><td>0</td><td>0.006</td><td>0.0061</td><td>0.012</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>170720</td><td>0.99</td><td>0</td><td>0.0077</td><td>0.002</td><td>0.0097</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>GW170729</td><td>0.018</td><td>0</td><td>0</td><td>0.98</td><td>0.98</td><td>0.48</td><td>...</td><td>...</td><td>0.52</td><td>0.52</td><td>0.057</td><td>0.94</td></tr><tr><td>GW170809</td><td>0</td><td>0</td><td>0.0064</td><td>0.99</td><td>1</td><td>0</td><td>...</td><td>...</td><td>1</td><td>1</td><td>...</td><td>...</td></tr><tr><td>GW170814</td><td>0</td><td>0</td><td>0.0024</td><td>1</td><td>1</td><td>0</td><td>...</td><td>...</td><td>1</td><td>1</td><td>0</td><td>1</td></tr><tr><td>GW170817</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>1</td><td>...</td><td>...</td><td>1</td><td>...</td><td>...</td></tr><tr><td>GW170818</td><td>0</td><td>0</td><td>0.0053</td><td>0.99</td><td>1</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</t
d></tr><tr><td>GW170823</td><td>0</td><td>0</td><td>0.0059</td><td>0.99</td><td>1</td><td>0</td><td>...</td><td>...</td><td>1</td><td>1</td><td>0.0043</td><td>1</td></tr></table>

<sup>a</sup>Calculated assuming that this event is a member of the BBH population for PyCBC, though its NSBH probability could also be nonnegligible.

The astrophysical probability for categories with few or zero detected events can be strongly influenced by the assumed prior on rates and physical property distributions. As such, we provide only upper bounds on these values.

with a pulsar component [214]. Earlier analyses [18,208] use a population model of binary neutron stars with uniform component masses in the $1 - 2M_{\odot}$ range and obtain an event rate interval of $320 - 4740\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ . In addition to updating this rate to account for all available data from $O1$ and $O2$ , we also introduce another fiducial population, serving two purposes. The first is to emulate a distribution assumed previously [208], which models both components as uncorrelated Gaussians.

The overall mass distribution is centered at $1.33M_{\odot}$ with a standard deviation of $0.09M_{\odot}$ . Second, this distribution can be considered as a bracket on the event rate from the upper end, since its $\langle VT\rangle$ over the population is smaller than the value obtained from the uniform set. To facilitate comparison and keep commensurate ranges of masses between the two distributions, we expand the uniform set to have component masses distributed between 0.8 and $2.3M_{\odot}$ .

The event rate distribution for each search and mass distribution is shown in Fig. 13. The differences in the distribution between the searches are a consequence of the

ranking statistic threshold applied to either. PyCBC measures a smaller $\langle VT\rangle$ , because its fiducial threshold is higher than GSTLAL. Despite the threshold difference, the two searches find similar values for $\Lambda_{\mathrm{BNS}}$ , and hence the rate for GSTLAL is lower than for PyCBC.

For the uniform mass set, we obtain an interval at $90\%$ confidence of $R = 800_{-680}^{+1970}$ Gpc $^{-3}$ y $^{-1}$ (PyCBC) and $R = 662_{-565}^{+1609}$ Gpc $^{-3}$ y $^{-1}$ (GstLAL), and for the Gaussian set we obtain $R = 1210_{-1040}^{+3230}$ Gpc $^{-3}$ y $^{-1}$ (PyCBC) and $R = 920_{-790}^{+2220}$ Gpc $^{-3}$ y $^{-1}$ (GstLAL). These values are consistent with previous observational values (both GW and radio pulsar) as well as more recent observationally driven investigations [215]. The union of the intervals combined over both populations lies in 110–3840 Gpc $^{-3}$ y $^{-1}$ .

# D. Neutron star black hole event rates

The NSBH space is a unique challenge both to model astrophysically and for which to produce accurate waveforms. Astrophysical models span a wide range of potential

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-26

![](dt=2026-04-19/ht=09/e038af07be74c9102f8406385e081bdd9bd71416c715334e8f6702e5db59b25f.jpg)

![](dt=2026-04-19/ht=09/f18c4bbb44ce7de9b4b8da4eaa0d666d5e2316ff640e9e14954a67f3abcea855.jpg)

mass ratios and spin configurations, and there are no electromagnetic observational examples. Hence, we take an approach similar to previous analyses [208] and examine specific points in the mass space while considering two component spin configurations: isotropic and orbital angular momentum aligned as described in Sec. VII B.

Since there are no confident detection candidates in the NSBH category, we update the upper limit at $90\%$ confidence in this category in Fig. 14. All upper limits are below $610\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ . Those results are obtained using a uniform prior over $R$ . The Jeffreys prior (which also appeared in Ref. [208]) suppresses larger $R$ values. This prior choice would obtain a less conservative upper limit. This limit is now stronger at all masses than the "high" rate prediction [216] ( $10^{3}\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ ) for NSBH sources.

# VIII. CONCLUSIONS

In GWTC-1, we have reported the results from three GW searches for compact mergers during the first and second observing runs by the advanced GW detector network. Advanced LIGO and Advanced Virgo have confidently detected gravitational waves from ten stellar-mass binary black hole mergers and one binary neutron star inspiral. The signals were discovered using three independent analyses: two matched-filter searches [8,9] and one weakly modeled burst search [11].

We have reported four previously unpublished BBH signals discovered during $O2$ , as well as updated FARs and parameter estimates for all previously reported GW detections. The reanalysis of $O1$ data did not reveal any new GW events, but improvements to the various detection pipelines have resulted in an increase of the significance of GW151012.

![](dt=2026-04-19/ht=09/72c202d6f42c2b31d85ee2943209c1a0405ad584db1b61c0029eccf5c07189bc.jpg)

Including these four new BBH mergers, the observed BBHs span a wide range of component masses, from $7.7_{-2.5}^{+2.2}M_{\odot}$ to $50.2_{-10.2}^{+16.2}M_{\odot}$ . One of the new events, GW170729, is found to be the highest-mass BBH observed to date, with GW170608 still being the lightest BBH [17]. The three other new events GW170809, GW170818, and GW170823 are all identified as heavy stellar-mass BBH mergers, ranging in total mass from $59.0_{-4.1}^{+5.4}M_{\odot}$ to $68.7_{-8.1}^{+10.8}M_{\odot}$ .

Similar to previous results, we find that the spins of the individual black holes are only weakly constrained, though for GW151226 and also for GW170729 we find that $\chi_{\mathrm{eff}}$ is positive and thus can rule out two nonspinning black holes as their constituents at greater than the $90\%$ credible level. On the other hand, $\chi_p$ is only weakly constrained in our measurements. Furthermore, we find that the higher mode content of the observed GW signals is weak enough that waveform models including them are not strongly preferred given our data (see Appendix B 3).

The binary mergers observed during $O1$ and $O2$ range in distance between $40_{-15}^{+8}$ Mpc for the binary neutron star inspiral GW170817 to $2840_{-1360}^{+1400}$ Mpc for GW170729, making it not only the heaviest BBH but also the most distant one observed to date. For the BNS merger GW170817, we have presented conservative upper limits on the properties of the remnant.

GW170818 is the second triple-coincident LIGO-Virgo GW event and is localized to an area of $39\mathrm{deg}^2$ , making it the best localized BBH to date. A similar impact of Virgo on the sky localization was already seen for GW170814 [16], reaffirming the importance of a global GW detector network for accurately localizing GW sources [200].

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-27

The properties of the observations reported in this catalog are based on general relativistic waveform models. Tests of the consistency of these observations with GR can be found in Refs. [15,217,218].

We have also presented a set of 14 marginal candidate events identified by the two matched-filter searches. The number of observed marginal events is consistent with our expectation given the chosen FAR threshold, but it is not possible to say whether or not a particular marginal trigger is a real GW signal.

Even with the set of ten BBH and one BNS, several outstanding questions remain regarding the origin and evolution of the detected binaries. To date, no binary components have been observed in either of the two putative mass gaps [139,140]—one between NSs and BHs and the other one due to pair instability supernovae [136,219]. Gravitational-wave measurement of BH spins favors
either small magnitudes or large misalignment with the orbital angular momentum. The latter favors a formation scenario where no spin alignment process is present, e.g., assembly in globular clusters [172,174].

Several studies [166–169,220–225] indicate that, with a few hundred detections, more detailed formation scenarios and evolutionary details can be parsed from the population. The BBH sample from $O1$ and $O2$ allows for new constraints on the primary mass power-law index $\alpha = 1.3_{-1.7}^{+1.4}$ [55].

Data products, including strain data and posterior samples, and postprocessing tools can be obtained from the accompanying data release [56].

The third observing run (O3) of Advanced LIGO and Virgo began on April 1, 2019. The inferred rate of BBH mergers is $9.7 - 101\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ and for BNS $110 - 3840\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ , and for NSBH binaries we obtain an improved $90\%$ upper limit of the merger rate of $610\mathrm{Gpc}^{-3}\mathrm{y}^{-1}$ ; in combination with further sensitivity upgrades to both LIGO and Virgo as well as the prospects of the Japanese GW detector KAGRA [226-228] joining the network possibly towards the end of O3 in 2019, many tens of binary observations are anticipated in the coming years [200], which will be presented in forthcoming catalogs.

# ACKNOWLEDGMENTS

The authors gratefully acknowledge the support of the United States National Science Foundation (NSF) for the construction and operation of the LIGO Laboratory and Advanced LIGO as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. The authors gratefully acknowledge the Italian Istituto

Nazionale di Fisica Nucleare (INFN), the French Centre National de la Recherche Scientifique (CNRS), and the Foundation for Fundamental Research on Matter supported by the Netherlands Organisation for Scientific Research, for the construction and operation of the Virgo detector and the creation and support of the EGO consortium.

The authors also gratefully acknowledge research support from these agencies as well as by the Council of Scientific and Industrial Research of India, the Department of Science and Technology, India, the Science and Engineering Research Board (SERB), India, the Ministry of Human Resource Development, India, the Spanish Agencia Estatal de Investigacion, the Vicepresidencia i Conselleria d'Innovacion, Recerca i Turisme and the Conselleria d'Educacion i Universitat del Govern de les Illes Balears, the Conselleria d'Educacion, Investigacion, Cultura i Esport de la Generalitat Valenciana, the

National Science Centre of Poland, the Swiss National Science Foundation (SNSF), the Russian Foundation for Basic Research, the Russian Science Foundation, the European Commission, the European Regional Development Funds (ERDF), the Royal Society, the Scottish Funding Council, the Scottish Universities Physics Alliance, the Hungarian Scientific Research Fund (OTKA), the Lyon Institute of Origins (LIO), the Paris Île-de-France Region, the National Research, Development and Innovation Office Hungary (NKFIH), the National Research Foundation of Korea, Industry Canada and the Province of Ontario

through the Ministry of Economic Development and Innovation, the Natural Science and Engineering Research Council Canada, the Canadian Institute for Advanced Research, the Brazilian Ministry of Science, Technology, Innovations, and Communications, the International Center for Theoretical Physics South American Institute for Fundamental Research (ICTP-SAIFR), the Research Grants Council of Hong Kong, the National Natural Science Foundation of China (NSFC), the Leverhulme Trust, the Research Corporation, the Ministry of Science and Technology (MOST), Taiwan, and the Kavli Foundation.

The authors gratefully acknowledge the support of the NSF, STFC, MPS, INFN, CNRS, and the State of Niedersachsen/Germany for provision of computational resources.

# APPENDIX A: CHARACTERIZATION OF TRANSIENT NOISE RELEVANT TO CATALOG TRIGGERS

The instrumental artifacts identified in time coincidence with triggers in this catalog can be split into four groups: scattered light, $60 - 200\mathrm{Hz}$ nonstationarity, short-duration transients, and blips. Time-frequency spectrograms of each glitch class can be seen in Fig. 15. We discuss the challenges of analyzing the times surrounding the instrumental artifacts and mitigation methods that can be used to address excess noise in the data.

B.P.ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-28

![](dt=2026-04-19/ht=09/1ba89ebd3f29c8e500f15682b9f0eab7bc043c8665917915c557d90080e46618.jpg)

![](dt=2026-04-19/ht=09/4674ecf5f797a76673dce64ccaa3de70d3e506478e82c6e04c39a39a735f6545.jpg)

![](dt=2026-04-19/ht=09/c0cc38e33da5a2b40ba59e76d18e09861b8e9c35876a989bad07c173e495c562.jpg)

![](dt=2026-04-19/ht=09/ea6df098640023273268423d07e3adee9342a18e4504576bedd9ed88bcdc416f.jpg)

# 1. Scattered light

Scattered light is a common source of noise in the LIGO and Virgo detectors. Stray light is reflected back into the main interferometer path, resulting in excess power in the data [91,229,230]. Motion of the reflective surface, such as optic mounts, phase shift the reflected light. When this motion is smaller than the wavelength of the main laser, $1064~\mathrm{nm}$ , the resultant artifacts are associated with stationary noise contributions to the interferometer spectrum. Larger motions result in archlike shapes in the time-frequency spectrograms.

It is these larger motions that impact transient searches and which, therefore, concern us here. This motion is observed by auxiliary sensors and is used to help identify periods of scattering [231]. Scattering light can be present in the data for stretches of multiple hours during periods of increased ground motion [91]. Scattered light is one of the most common sources of background triggers in searches for compact binary coalescence signals [68,92,93]. Variability in the time-frequency morphology of scattered light leads to noise triggers for a wide variety of template parameters.

The strain noise amplitude of scattered light instrumental artifacts correlates with the intensity of the corresponding ground motion. Additionally, higher velocities of optic motion lead to higher-frequency content in the scattered light [229].

Since scattered light typically impacts low frequencies, it is possible to mitigate the impact by using only data from frequencies above the impacted region. Because it is often

related to monitored optic motion, subtraction of the artifacts based on a nonlinear relationship with this optic motion may be possible.

# 2. 60-200 Hz nonstationarity

The $60 - 200\mathrm{Hz}$ nonstationarity appears in time-frequency spectrograms as excess power with slowly varying frequencies in clusters of multiple minutes [68]. The structure of the excess power appears similar to scattered light but at higher frequencies than predicted by available witnesses of optic motion. This type of nonstationarity occurs in both Hanford and Livingston at a rate of 1-2 times per day, making it unlikely for an astrophysical signal to be correlated by chance.

While correlations between excess seismic noise and the nonstationarity exist, no clear witnesses have been identified. The structure of this class of artifact changed after mitigation of the motion of baffles used to block s
tray light [232,233]. This change suggests that baffles may be involved in the production of the $60 - 200\mathrm{Hz}$ nonstationarity.

In previous observing runs, the rate of these artifacts caused multiple hours of data to be vetoed. In time that is not vetoed, these transients can create significant triggers in the background of matched-filter searches [68]. This instrumental artifact particularly affects signals that are in the sensitive band of the detector above $30\mathrm{Hz}$ for longer than 3 sec.

The long-duration, variable frequencies and lack of a clear witness make this nonstationarity a difficult target for

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-29

noise subtraction. Efforts to completely mitigate the periods of nonstationarity are ongoing.

# 3. Short-duration transient

Short-duration transients last less than one second and have a high amplitude. The highest-amplitude transients can cause overflows of the digital-to-analog converters used to control the positions of the test masses. These transients occur in both LIGO detectors at a rate of approximately 1 per hour, with their cause largely unknown.

The large amount of excess power due to these artifacts produces a large impulse response during the whitening process, affecting the ability of searches to optimally search the surrounding data [8]. For this reason, these artifacts are gated by the searches before analysis, using the procedure described in Ref. [8]. Notably, the instrumental artifact present in LIGO-Livingston during GW170817 [18] is of this class.

For systems like GW170817, gating is shown to remove short-duration transients without a significant bias to astrophysical parameter estimation [234]; full glitch subtraction with BAYESWAVE [53], as used for the GW170817 parameter estimation, produces more robust results.

# 4. Blips

Blip transients [91] are short, band-limited transients that occur in both LIGO detectors at a rate of roughly once per hour. Because of their subsecond duration and limited bandwidth, these transients often have a significant overlap with the shortest templates used in matched-filter searches. Templates that terminate between 50 and $100\mathrm{Hz}$ and have high ratios of component mass parameters have a similar morphology to these artifacts.

Blip transients are particularly problematic, as they typically do not couple into any witness sensors used to monitor the detector, which makes it difficult to systematically remove them from the analyses. As such, these transients are the limiting noise source to modeled searches for high-mass compact binary coalescences in $O1$ and $O2$ [68,72,91,235].

Investigations into blips have identified multiple causes [236], but the vast majority of blips remain unexplained. Although these transients cannot be removed from the analysis entirely, signal morphology tests in matched-filter searches are used to mitigate their effects [72].

# APPENDIX B:PARAMETER-ESTIMATION DESCRIPTION

We use coherent Bayesian inference methods to extract the posterior distribution $p(\vec{\vartheta}|\vec{d})$ for the parameters $\vec{\vartheta}$ that characterize a compact binary coalescence associated with a particular GW event. Following Bayes' theorem [237,238], the posterior is proportional to the product of the likelihood of the data given the parameters and the prior (assumed) distribution of the parameters. The likelihood function depends on a noise-weighted inner product

between the detector data $\vec{d}$ and a parametrized waveform model for the two GW polarizations $h_{+, \times}(\vec{\vartheta}; t)$ which is projected onto the response of each detector to obtain the strain [128]. By marginalizing the posterior distribution over all but one or two parameters, it is then possible to generate credible intervals or credible regions for those parameters.

We sample the posterior distribution with stochastic sampling algorithms using an implementation of Markov chain Monte Carlo [239,240] and nested sampling available in the LALINFERENCE package [241] as part of the LSC Algorithm Library (LAL) [242]. Additional posterior results for computationally expensive waveform models are obtained with an alternative parallelized parameter-estimation code RAPIDPE [243,244].

We estimate the power spectral density (PSD) that enters the inner product using BayesWave [53,54]. The PSD is modeled as a cubic spline for the broadband structure and a sum of Lorentzians for the line features. A median PSD is computed from the resulting posterior probability density function of PSDs, defined separately at each frequency. This PSD is expected to lead to more stable and reliable inference. The parameter estimation analyses in $O1$ assumed uniform priors for the calibration uncertainties in frequency.

For the analyses in this catalog, frequency-dependent spline calibration envelopes [98] are incorporated into the measured GW strain to factor in potential deviations from the true GW strain due to uncertainties in the detector calibration [64,65]. We marginalize over the additional calibration parameters. See Sec. II B in Ref. [97] for details.

The low-frequency cutoff for likelihood integration used for parameter-estimation analyses is set to $20\mathrm{Hz}$ with the following exceptions: For GW170817 the analysis starts at $23\mathrm{Hz}$ , for GW170818 at $16\mathrm{Hz}$ , and for the GW170608 Hanford data at $30\mathrm{Hz}$ .

Because of the expansion of the Universe, we measure redshifted masses from GW observations. We assume a standard flat $\Lambda$ CDM cosmology with Hubble parameter $H_0 = 67.9\mathrm{km s^{-1}Mpc^{-1}}$ and matter density parameter $\Omega_{m} = 0.306$ [245] and compute the redshift from the measured luminosity distance distribution. To obtain physical binary masses in the source frame, we divide the observed detector-frame masses by $(1 + z)$ [12,246]. For the events in this catalog, changing from the above cosmology to the updated Planck (TT, TE, EE + lowE + lensing + BAO) [247] cosmology results in a relative change in the redshift of $< 0.3\%$ and a relative change in the source-frame total mass of $< 0.03\%$ .

# 1. Priors used in individual event analysis

We assume a jointly uniform prior in the detector-frame component masses with bounds chosen such that the posterior has support only in the interior of the domain.

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-30

Spin vectors are assumed to be isotropic on the sphere and uniform in spin magnitude. This prior is also used for models that enforce spins to be aligned with the orbital angular momentum. We use an isotropic prior for the location of the source on the sky. The distance prior is proportional to the luminosity distance squared. The prior distribution for the inclination angle $\theta_{JN}$ is assumed to be uniform in its cosine. The priors for polarization angle time and phase of coalescence are uniform.

For the reanalysis of GW170817, we choose two different spin priors, consistent with previous analyses [18,97]: $a_{i} \leq 0.89$ and $a_{i} \leq 0.05$ . In addition to the BBH binary parameters, we also sample in the dimensionless tidal deformabilities $\Lambda_{i}$ of each NS. They are assumed to be jointly uniform within $0 \leq \Lambda_{i} \leq 5000$ .

# 2. Waveform models

For analyses of BBH systems in this catalog, we use two waveform models [248]: an effective precession model (IMRPhenomPv2) [25,26,49] using the effective precession parameter $\chi_{p}$ and a full precession model (SEOBNRv3) [27,28,30] which includes generic two-spin inspiral precession dynamics. For both models, only the nonprecessing spin sector is tuned to NR simulations. Analyses with the effective precession model are carried out with LALINFERENCE. Analyses with the full precession model also use LALINFERENCE, except for GW170814, where we use RAPIDPE.

For most BBH events, results from the two wav
eform models are consistent, and the data give us little reason to prefer one model over the other. Posteriors generated with LALINFERENCE and RAPIDPE for the two models also agree well for most of the events presented here. We point out notable differences in results between the BBH waveform models below.

To quantify the agreement between the models, we compute the Jensen-Shannon divergence (JSD) [249] between posterior distributions obtained with the two BBH waveform models. The JSD is a symmetrized and smoothed measure of distance between two probability distributions $p(x)$ and $q(x)$ defined as

$$
D _ {\mathrm {J S}} (p \mid q) = \frac {1}{2} \left[ D _ {\mathrm {K L}} (p \mid s) + D _ {\mathrm {K L}} (q \mid s) \right], \tag {B1}
$$

where $s = 1 / 2(p + q)$ and

$$
D _ {\mathrm {K L}} (p \mid q) = \int p (x) \log_ {2} \left(\frac {p (x)}{q (x)}\right) d x \tag {B2}
$$

is the Kullback-Leibler divergence (KLD) between the distributions $p$ and $q$ , measured in bits. The JSD fulfills the bound $0 \leq D_{\mathrm{JS}}(p|q) \leq 1$ when measured in bits. We compute results for the KLDs for the JSDs between BBH models and the KLDs between prior and posteriors shown in Tables V and VI from kernel density estimates of random

![](dt=2026-04-19/ht=09/ab63cfe722671baa0de3b954f7055f96994bb64b435a37ced7a0fe8680d4113b.jpg)

draws from the prior and posterior distributions and quoting the median and $90\%$ intervals.

We plot the JSD for all GW events for selected binary parameters in Fig. 16. The JSD values are, in general, smaller than approximately 0.05 bits, which indicates that the posteriors from the two BBH waveform models agree well. The largest value, $D_{\mathrm{JS}} = 0.14$ bits, is found for the $\chi_{\mathrm{eff}}$ distributions for GW151226. This result indicates the different $\chi_{\mathrm{eff}}$ probability density function (PDFs) measured by the two models for this event, as mentioned in Sec. VF.

Further notable differences, quoted for the two BBH waveform models (SEOBNRv3 and IMRPhenomPv2) are the detector-frame chirp mass $\mathcal{M}^{\mathrm{det}}$ for GW151226: $(9.68_{-0.08}^{+0.08}M_{\odot},9.70_{-0.07}^{+0.07}M_{\odot})$ , the effective aligned spin $\chi_{\mathrm{eff}}$ for GW150914: $(0.01_{-0.13}^{+0.12}$ $-0.03_{-0.12}^{+0.11})$ and GW170608: $(0.02_{-0.07}^{+0.19}$ $0.04_{-0.06}^{+0.19})$ , the effective precession spin $\chi_p$ for GW150914: $(0.30_{-0.20}^{+0.34}$ $0.39_{-0.31}^{+0.47})$ , GW170814: $(0.37_{-0.28}^{+0.47}$ $0.34_{-0.26}^{+0.43})$ and GW170818:

$(0.42_{-0.28}^{+0.38}$ $0.56_{-0.39}^{+0.33})$ , and the mass ratio for GW170729: $(0.72_{-0.28}^{+0.25}$ $0.63_{-0.26}^{+0.32})$

Because of the good overall agreement between waveforms, we present in our overall results posterior distributions for BBH coalescences that are averaged between the two models (using SEOBNR samples from LALINFERENCE and for GW170814 from RAPIDPE) and incorporate an equal number of samples from either model. These overall samples are used in the discussion of the source properties below. Waveforms from the full precession model need to be generated sufficiently far away from merger to enable cleanly attaching the merger-ringdown part to the inspiral-plunge part of the waveform.

For several events, this procedure requires generating the full precession model from a lower starting frequency than for the effective precession model. To quote frequency-dependent quantities at a consistent reference frequency, which is a

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-31

prerequisite for combining samples, we evolve the samples from the full precession model forward in time from $10\mathrm{Hz}$ to the fiducial reference frequency of $20\mathrm{Hz}$ .

We use marginalization over the arrival time and phase of the signal as an approximation for the fully precessing SEOBNRv3 model to make the analyses more computationally tractable. This approximation is valid if the $(\ell, m) = (2, \pm 1)$ observer-frame modes are subdominant compared to $(2, \pm 2)$ modes, as is the case for nearly face-on-face-off binaries. It is demonstrated that the impact of this approximation is negligible for GW150914 [96] and for GW170104 and GW151226 based on preliminary analyses.

Waveform models for BNS.—Unlike in the analysis of GW170817 in Ref. [97], we use three frequency-domain waveform models: the purely analytical TaylorF2 [35,36,38,100-112] and two point-particle models which add a fit to the phase evolution from tidal effects [32,99], SEOBNRv4NRT [29,32,77,99] and IMRPhenomPv2NRT [25,26,32,49,99,250]. These models are fast enough to be used as templates in LALINFERENCE. We supplement our results with two time-domain models (SEOBNRv4T [31] and TEOBResumS [33,113]), where the analysis is performed with RAPIDPE.

In addition to tidal effects, IMRPhenomPv2NRT also includes the spin-induced quadrupole moment that enters in the phasing up to 3PN order [251,252]. In contrast to the analysis in Ref. [97], the terms up to 2PN order are now also included in SEOBNRv4NRT and SEOBNRv4T. The EOS dependence of each NS's spin-induced quadrupole moment is included by relating it to the tidal parameter of each NS using the quasiconiversal relations of Ref. [253]. For the spin-aligned models used for GW170817, the phase at coalescence is analytically marginalized out [241].

# 3. Impact of higher harmonics in the waveform

Waveform models including higher modes beyond the leading-order quadrupole contribution that cover the entire parameter space of our analyses were not available at the time of writing of this paper. Here, we systematically compare all $O2$ BBH observations with NR simulations supplemented by NR surrogate waveforms [254] that cover mass ratios up to $q = 0.2$ and aligned effective spins up to $|\chi_{\mathrm{eff}}| \sim 0.8$ . These calculations focus on the impact of higher modes on the measurement of intrinsic parameters (i.e., masses and spins) using RAPIDPE [244,255,256] techniques.

We find no compelling evidence that higher-order modes substantially affect our measurement of mass or spin parameters for any event. Instead, we find that they only modestly influence the interpretation of any observation, i.e., at a level smaller than our current statistical measurement uncertainty. For instance, we find for GW170729 a Bayes factor of approximately 1.4 for higher modes versus a pure quadrupole model. Assuming that GR is correct and these modes are present, however, we infer a

modestly different mass ratio distribution with and without higher modes, with a mean (median) value of $q$ to be 0.61 (0.58) and 0.66(0.65), respectively, using the fiducial prior. Similarly, for 170809, we find a revised $\chi_{\mathrm{eff}}$ distribution which is symmetric about a median value of zero.

We conclude that the higher-mode content of the GW signals is weak enough that models including them are not strongly preferred given our data. This conclusion is consistent with the fact that the contribution from higher modes is highly suppressed for signals emitted by binaries with mass ratio $q \gtrsim 0.5$ , total masses $\lesssim 100 M_{\odot}$ , and weak support for an edge on inclination $\theta_{JN} = 90^{\circ}$ , as is the case for the observed BBHs [257,258]. Our results agree with those in Refs. [144,259], which find that in these cases higher modes mostly affect the estimation of the inclination angle and luminosity distance.

# APPENDIX C: IMPACT OF PRIORS ON BAYESIAN PARAMETER ESTIMATION

In Bayesian inference, certain parameters inferred from measurements can be sensitive to the choice of priors and, thus, affect the interpretation of the observed events, with GW observations being no exception [55,179,260-262]. In this Appendix, we illustrate the impact of prior choices on the inference of source properties for GW events. Specifically
, we choose the BBH GW170809, which lies in the bulk of GW observations to date (see Sec. V), as our explicit example.

# 1. Prior choices

Our default prior choice for the analysis of a GW event is uninformative. For GW170809, we choose the following default prior, henceforth referred to as P1:

Let us now consider two alternative prior choices:

P2: A volumetric spin prior in which the spin components are distributed uniformly inside the sphere $V = 4\pi (a_{\mathrm{max}}^3 - a_{\mathrm{min}}^3) / 3$ . This choice replaces assumptions (b) and (c) in the default prior P1.

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-32

TABLE V. KL divergences (in bits) between the prior and posterior for the effective aligned spin $\chi_{\mathrm{eff}}$ and the effective precession spin $\chi_p$ . For the computation of the KL divergence for $\chi_p$ , we quote the KL divergence with the prior conditioned on the $\chi_{\mathrm{eff}}$ posterior, $D_{\mathrm{KL}}^{\chi_p}(\chi_{\mathrm{eff}})$ , and without conditioning, $D_{\mathrm{KL}}^{\chi_p}$ . For GW170817, $D_{\mathrm{KL}}^{\chi_p}$ is given for the high spin prior. The median and 90% interval for the KL divergences is estimated by computing the statistic for repeated draws of a subset of the posterior and prior PDFs. Single-detector optimal SNRs from parameter-estimation analyses for Hanford (H), Livingston (L), and Virgo (V).

![](dt=2026-04-19/ht=09/9ef4d63a9c46eed056885f6d80655d3e23979c97fbbcde76930f7fc298d7efca.jpg)

<table><tr><td>Event</td><td>GW150914</td><td>GW151012</td><td>GW151226</td><td>GW170104</td><td>GW170608</td><td>GW170729</td><td>GW170809</td><td>GW170814</td><td>GW170817</td><td>GW170818</td><td>GW170823</td></tr><tr><td>\(D_{\text{KL}}^{\chi_{\text{eff}}}\)</td><td>0.71\(_{-0.03}\)</td><td>0.23\(_{-0.02}\)</td><td>1.32\(_{-0.06}\)</td><td>0.54\(_{-0.03}\)</td><td>0.97\(_{-0.05}\)</td><td>1.83\(_{-0.09}\)</td><td>0.71\(_{-0.03}\)</td><td>0.99\(_{-0.07}\)</td><td>2.32\(_{-0.10}\)</td><td>0.50\(_{-0.03}\)</td><td>0.32\(_{-0.03}\)</td></tr><tr><td>\(D_{\text{KL}}^{\chi_p}\)</td><td>0.16\(_{-0.02}\)</td><td>0.09\(_{-0.02}\)</td><td>0.17\(_{-0.04}\)</td><td>0.05\(_{-0.01}\)</td><td>0.07\(_{-0.02}\)</td><td>0.09\(_{-0.02}\)</td><td>0.05\(_{-0.01}\)</td><td>0.02\(_{-0.01}\)</td><td>0.19\(_{-0.03}\)</td><td>0.06\(_{-0.01}\)</td><td>0.03\(_{-0.01}\)</td></tr><tr><td>\(D_{\text{KL}}^{\chi_p}(\chi_{\text{eff}})\)</td><td>0.09\(_{-0.02}\)</td><td>0.08\(_{-0.01}\)</td><td>0.12\(_{-0.02}\)</td><td>0.07\(_{-0.01}\)</td><td>0.08\(_{-0.02}\)</td><td>0.03\(_{-0.01}\)</td><td>0.06\(_{-0.01}\)</td><td>0.13\(_{-0.02}\)</td><td>0.07\(_{-0.02}\)</td><td>0.09\(_{-0.01}\)</td><td>0.03\(_{-0.01}\)</td></tr><tr><td>H SNR</td><td>20.6\(_{-1.6}\)</td><td>6.4\(_{-1.3}\)</td><td>9.8\(_{-1.4}\)</td><td>9.5\(_{-1.6}\)</td><td>12.1\(_{-1.6}\)</td><td>5.9\(_{-1.1}\)</td><td>5.9\(_{-1.4}\)</td><td>9.3\(_{-1.2}\)</td><td>18.9\(_{-1.0}\)</td><td>4.6\(_{-0.8}\)</td><td>6.8\(_{-1.2}\)</td></tr><tr><td>L SNR</td><td>14.2\(_{-1.4}\)</td><td>5.8\(_{-1.2}\)</td><td>6.9\(_{-1.1}\)</td><td>9.9\(_{-1.3}\)</td><td>9.2\(_{-1.2}\)</td><td>8.3\(_{-1.4}\)</td><td>10.7\(_{-1.8}\)</td><td>14.3\(_{-1.4}\)</td><td>26.3\(_{-1.3}\)</td><td>9.7\(_{-1.5}\)</td><td>9.2\(_{-1.5}\)</td></tr><tr><td>V SNR</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>1.7\(_{-1.1}\)</td><td>1.1\(_{-0.8}\)</td><td>4.1\(_{-1.1}\)</td><td>3.0\(_{-0.2}\)</td><td>4.2\(_{-0.7}\)</td><td>...</td></tr></table>

P3: In addition to the volumetric spin prior, we also assume a uniform distribution in luminosity distance.

This choice replaces assumptions (b)-(d) in P1. Naturally, the priors for derived parameters, such as the effective aligned spin [23,121] given in Eq. (4) or the effective precession spin [125],

$$
\chi_ {p} = \frac {1}{B _ {1} m _ {1} ^ {2}} \max  \left(B _ {1} S _ {1 \perp}, B _ {2} S _ {2 \perp}\right) > 0, \tag {C1}
$$

where $B_{1} = 2 + 3q / 2$ and $B_{2} = 2 + 3 / (2q)$ , are coupled to the choice of priors for the mass ratio, spin magnitudes, and spin directions.

In current GW observations, small values of $\chi_{\mathrm{eff}}$ are preferred and $\chi_p$ is unconstrained, while spin measurements in x-ray binaries point to a range of spin magnitudes [263], including high spins. The volumetric spin prior P2 adds weight to higher spin values in comparison to the default spin prior, allowing us to test the robustness of the measurement, which can be important for understanding our inferences on the underlying binary population [55,262].

The additional assumption of a uniform in distance distribution in P3 may seem unnatural at first, but it provides a strong test on the robustness of the inference of the luminosity distance which is important for the cross-correlation with galaxy catalogs. Furthermore, it has computational advantages when low-significance events are considered in combination with nested sampling [264].

If the data are uninformative about a parameter, the choice of prior will play a dominant role in determining the shape of the posterior probability distribution. In the left column in Fig. 17, we show the different prior choices for a subset of physical parameters.

# 2. Comparison of posteriors under different prior assumptions

Here, we detail the posterior probability distributions obtained from a Bayesian parameter estimation on

TABLE VI. KL divergences (in bits) between the prior and posterior distribution for various parameters for GW170809 under the three different prior assumptions P1, P2, and P3. We note that the values for P1 are different from the ones in Table V, as we consider only the IMRPhenomPv2 model here.

![](dt=2026-04-19/ht=09/c0fea7c56f230abab3bad93b78f3753d935ceb68743e6ffcd471f1994230600d.jpg)

<table><tr><td>Prior</td><td>P1</td><td>P2</td><td>P3</td></tr><tr><td>DχeffKL</td><td>0.69+0.03-0.04</td><td>1.04+0.04-0.05</td><td>0.96+0.04-0.03</td></tr><tr><td>DχpKL</td><td>0.02+0.01-0.01</td><td>0.03+0.01-0.01</td><td>0.02+0.01-0.01</td></tr><tr><td>DχpKL(χeff)</td><td>0.04+0.01-0.01</td><td>0.05+0.01-0.01</td><td>0.06+0.01-0.01</td></tr><tr><td>DMdetKL</td><td>2.25+0.06-0.06</td><td>2.07+0.07-0.05</td><td>2.02+0.05-0.07</td></tr><tr><td>DMtotKL</td><td>2.20+0.06-0.08</td><td>1.84+0.06-0.05</td><td>1.77+0.05-0.05</td></tr><tr><td>DqKL</td><td>0.73+0.03-0.04</td><td>0.64+0.03-0.03</td><td>0.54+0.03-0.03</td></tr><tr><td>DdlKL</td><td>4.59+0.12-0.20</td><td>4.58+0.11-0.17</td><td>2.04+0.07-0.05</td></tr></table>

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-33

![](dt=2026-04-19/ht=09/0cece1b4b32cc32b5ffb91ed4c28aca49440b373649ec941f883319805bc5d04.jpg)

![](dt=2026-04-19/ht=09/67fc4f064d0231f8c0298be6ff03ec5f9874527147e6ae910adbd1ab37e83d50.jpg)

![](dt=2026-04-19/ht=09/deed43857c99694e5c47fd526d7e880dd7ee2fb4e690133e1f2a2ab6571e70ea.jpg)

![](dt=2026-04-19/ht=09/feffe20f72b382af2a11955a2cfd0c616d7449c8e75063b253e5e6d976556318.jpg)

![](dt=2026-04-19/ht=09/7db3b5cf52c38b11b6ad8b1c50deaea434b483ca516a43f6783bc90ae026929b.jpg)

![](dt=2026-04-19/ht=09/841b73d5e2166c8e8913183502106e0c156605bf33e6ae3872aa6aa1e65c5b2d.jpg)

![](dt=2026-04-19/ht=09/2e1d5a813585496642180c6c0d501ba4ea43fcd68099d40c6efe2784dbeef655.jpg)

![
](s3://lakehouse2/hive-ha/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-04-19/ht=09//57be90c1fa2c5aedf6028664b98c12f9578dd4ee7993448528e0e461c89a9a43.jpg)

GW170809 with the different prior choices P1, P2, and P3. The results are obtained using the nested sampling algorithm implemented in LALINFERENCE [241] and the precessing waveform model IMRPhenomPv2 [25,26,49]. Marginalized one-dimensional PDFs for various parameters under the three different prior assumptions are shown in the right column in Fig. 17: The posterior PDFs for well-measured parameters have similar shapes irrespective of the assumed prior (e.g., the chirp mass), whereas they are very

different, and hence prior dependent, for ill-measured parameters such as the effective precession spin.

To quantify the impact of the choice of prior on parameter estimation and hence our observations from Fig. 17, we use the Kullback-Leibler (KL) divergence $D_{\mathrm{KL}}$ [159] as defined in Eq. (B2). This divergence allows us to determine the information gain between the prior and the posterior distributions. The results are summarized in Table VI. A similar spread on parameters, where

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-34

applicable, is reported in Ref. [261], where $D_{\mathrm{KL}}^{\chi_{\mathrm{eff}}} \sim \mathcal{O}(1)$ and $D_{\mathrm{KL}}^{\chi_{\mathrm{p}}} \sim \mathcal{O}(10^{-2})$ . As expected, parameters that have a dominant impact on the binary phasing, for example, the chirp mass $\mathcal{M}$ and the effective aligned spin $\chi_{\mathrm{eff}}$ , are well measured and robust against different prior choices. Other parameters such as the effective precession spin $\chi_{p}$ , however, are relatively poorly constrained in current observations, and the KL divergence approaches zero, implying that we predominantly recover the priors.

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-35

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-36

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-37

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-38

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-39

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-40

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-41

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-42

B.P.Abbott1R.AbbottT.D.Abbott2S.Abraham3F.Acernese,4K. Ackley6C.Adams7R.X.Adhikari V.B.Adya89C.Affeldt89M.Agathos10K.Agatsuma11N.Aggarwal12O.D.Aguiar13L.Aiello14,15A.Ain3P.Ajith16 G.Allen,17A.Allocca18,19M.A.Aloy20P.A.Altin21A.Amato22A.Ananyeva1S.B.Anderson1W.G.Anderson2 S.V.Angelova24S.Antier25S.Appert1K.Arai1M.C.Araya1J.S.Areeda26M.Arene27N.Arnaud25,28K.G.Arun29 S. Ascenzi30,31G.Ashton6M.Aston7P.Astone32F.Aubin33P.Aufmuth9K.AultONeal34C.Austin2V.Avendano35 A.Avila-Alvarez26S.Babak36,27P.Bacon27F.Badaracco14,15M.K.M.Bader37S.Bae38P.T.Baker39F.Baldaccini40,41 G.Ballardin28S.W.Ballmer42S.

Banagiri43J.C.Barayoga1S.E.Barclay44B.C.Barish1D.Barker45K.Barkett46 S.Barnum12F.Barone45B.Barr44L.Barsotti12M.Barsuglia27D.Barta47J.Bartlett45I.Bartos48R.Bassiri49 A. Basti18,19M.Bawaj50,41J.C.Bayley44M.Bazzan51,52B.Bécsy53M.Bejger27,54I.Belahcene25A.S.Bell44 D.Beniwal55B.K.Berger49G.Bergmann89S.Bernuzzi56,57J.J.Bero58C.P.L.Berry59D.Bersanetti60A.Bertolini37 J. Betzwieser7R.Bhandare61J.Bidler26I.A.Bilenko62S.A.Bilgili39G.Billingsley1J. Birch7R. Birney24 O.Birnholtz58S.Biscans112S.Biscoveanu6A.Bisht9MBitossi28,19M.A.Bizouard25J.K.Blackburn1J.Blackman46 C.D.Blair7D.G.Blair63R.M.Blair45S.

Bloemen64N.Bode89M.Boer65Y.Boetzel66G.Bogaert65F.Bondu67 E. Bonilla49R. Bonnard33P.Booker89B.ABoom37C.D Booth68R. Bork1V. Boschi28S. Bose69K. Bossie7 V. Bossilkov63J. Bosveld63Y. Bouffanais27A.Bozzi28C.Bradaschia19P.R.Brady23A. Bramley7M.Branchesi14,15 J.E.Brau70T. Briant71J.H.Briggs44F.Brighenti72,73A.Brillet65M.Brinkmann89V. Brisson25†P. Brockill23 A.FBrooks1D.D.Brown55S.Brunett1A.Buikema12T. Bulik7H.J. Bulten75,37A. Buonanno36,76D. Buskulic3 M.J. Bustamante Rosell77C. Buy27R.L.Byer49M.Cabero89L.Cadonati78G.Cagnoli22,79C.Cahillane 1 J.Calderon Bustillo6T.A.CallisterE.Calloni80,5J.B.Camp8W.A.

Campbell6M.Canepa82,60K.C.Cannon8 H.Cao55J.Cao84E.Capocasa27F.Carbognani28S.Caride85M.F.Carney59G.Carullo18J.Casanueva Diaz19 Casetontini30,31S.Caudill37M.Cavaglia86F.Cavalier25R.Cavalieri28G.Cella19P.Cerdá-Durán20G.Cerretani18,19 E.Cesarini87,31O.Chaibi65K.Chakravarti3S.J.Chamberlin88MChan44S.Chao89P.Charlton90E.A.Chase59 E.Chassande-Mottin27D.Chatterjee23M.Chaturvedi61K.Chatzioannou91B.D.Cheeseboro39H.Y.Chen92 X.Chen63Y.Chen46H.-P.Cheng48C.K.Cheong93H.Y.Chia48A.Chincarini60A.Chiummo28G.Cho94H.S.Cho95 M.Cho76N.Christensen65,96Q.Chu63S.Chua7K.W.Chung93S.Chung6G.Ciani51,52A.A.Ciobanu55 R.

Ciolfi97,98F.Cipriano65A.Cirone82,60F.Clara45J.A.Clark78P.Clearwater99F.Cleva65C.Cocchieri86 E.Coccia14,15P-F.Cohadon7D.Cohen25R.Colgan100M.Colleoni100C.G.Collette102C.Collins11L.R.Cominsky103 M.Constancio Jr.,13L.Conti52S.J.Cooper11P.Corban7T.R.Corbitt2I.Cordero-Carrion104 K.R.Corley100 N.Cornish53A.Corsi85S.Cortese28C.A.Costa13R.Cotesta36M.W.Coughlin1S.B.Coughlin68,59J.-P.Coulon65 S.T. Countryman100P.Couvares1P.B.Covas101E.E.Cowan78D.M.Coward63M.J.Cowart7D.C.Coyne 1 R.Coyne105J.D.E.Creighton23T.D.Creighton106J.Cripe2M.Croquette7S.G.Crowder107T.J.Cullen2 A.Cumming44L.Cunningham44E.Cuoco28T.

Dal Canton8G.Dalya108S.L.Danilishin8,S.D'Antonio3 K.Danzmann9,A.Dasgupta109C.F.Da Silva Costa48L.E.H.Datrier4V.Dattilo28I.Dave6M.Davier25D.Davis42 E.J.Daw11D.DeBra49M.Deenadayalan3J.Degallaix22M.De Laurentis80,5S.Deleglise7W.Del Pozzo18,19 L.M.DeMarchi59N.Demos12T.Dent89,11R.De Pietri112,57J.Derby26R.De Rosa80,C.De Rossi22,28R.DeSalvo 1 0 de Varona8,S Dhurandhar,M.C.Diaz,106T.Dietrich7L.Di Fiore5M.Di Giovanni,114,98T.Di Girolamo80,5 A.Di Lieto,18,9B.Ding,102S.Di Pace,115,32I.Di Palma,115,32F.Di Renzo,18,19A.Dmittriev,1Z.Doctor92F.Doovan 1 0 K.L.Dooley68,86S.Doravari8,I.Dorrington68T.P.

Downes23M.Drago14,15J.C.Driggers45Z.Du8J.-G.Ducoin25

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-43

P. Dupej,\(^{44}\) S. E. Dwyer,\(^{45}\) P. J. Easter,\(^{6}\) T. B. Edo,\(^{110}\) M. C. Edwards,\(^{96}\) A. Effler,\(^{7}\) P. Ehrens,\(^{1}\) J. Eichholz,\(^{1}\) S. S. Eikenberry,\(^{48}\) M. Eisenmann,\(^{33}\) R. A. Eisenstein,\(^{12}\) R. C. Essick,\(^{92}\) H. Estelles,\(^{101}\) D. Estevez,\(^{33}\) Z. B. Etienne,\(^{39}\) T. Etzel,\(^{1}\) M. Evans,\(^{12}\) T. M. Evans,\(^{7}\) V. Fafone,\(^{30,31,14}\) H. Fair,\(^{42}\) S. Fairhurst,\(^{68}\) X. Fan,\(^{84}\) S. Farinon,\(^{60}\) B. Farr,\(^{70}\) W. M. Farr,\(^{11}\) E. J. Fauchon-Jones,\(^{68}\) M. Favata,\(^{35}\) M.

Fays,\(^{110}\) M. Fazio,\(^{116}\) C. Fee,\(^{117}\) J. Feicht,\(^{1}\) M. M. Fejer,\(^{49}\) F. Feng,\(^{27}\) A. Fernandez-Galiana,\(^{12}\) I. Ferrante,\(^{18,19}\) E. C. Ferreira,\(^{13}\) T. A. Ferreira,\(^{13}\) F. Ferrini,\(^{28}\) F. Fidecaro,\(^{18,19}\) I. Fiori,\(^{28}\) D. Fiorucci,\(^{27}\) M. Fishbach,\(^{92}\) R. P. Fisher,\(^{42,118}\) J. M. Fisher,\(^{12}\) M. Fitz-Axen,\(^{43}\) R. Flaminio,\(^{33,119}\) M. Fletcher,\(^{44}\) E. Flynn,\(^{26}\) H. Fong,\(^{91}\) J. A. Font,\(^{20,120}\) P. W. F. Forsyth,\(^{21}\) J.-D. Fournier,\(^{65}\) S. Frasca,\(^{115,32}\) F.

Frasconi,\(^{19}\) Z. Frei,\(^{108}\) A. Freise,\(^{11}\) R. Frey,\(^{70}\) V. Frey,\(^{25}\) P. Fritschel,\(^{12}\) V. V. Frolov,\(^{7}\) P. Fulda,\(^{48}\) M. Fyffe,\(^{7}\) H. A. Gabbard,\(^{44}\) B. U. Gadre,\(^{3}\) S.M.Gaebel,\(^{11}\) J.R.Gair,\(^{121}\) L.Gammaitoni,\(^{40}\) M.R.Ganija,\(^{55}\) S.G.Gaonkar,\(^{3}\) A.Garcia\(^{26}\) C.Garcia-Quiros,\(^{101}\) F.Garufi,\(^{80,5}\) B.Gateley,\(^{45}\) S.Gaudio,\(^{34}\) G.Gaur,\(^{122}\) V.Gayathri,\(^{123}\) G.Gemme,\(^{60}\) E.Genin,\(^{28}\) A.Gennai,\(^{19}\) D.George,\(^{71}\) J.George,\(^{61}\) L.Gergely,\(^{124
}\) V.

Germain,\(^{33}\) S.Ghong,\(^{78}\) Abhirup Ghosh,\(^{16}\) Archisman Ghosh,\(^{37}\) S.Ghosh,\(^{23}\) B.Giacomazzo,\(^{114,98}\) J.A.Giaime,\(^{2,7}\) K.D.Giardina,\(^\dagger\) A.Giazotto,\(^\dagger\) K.Gill,G.\(^\dagger\) Giordano, \(^\dagger\) L.Glover, \(^\dagger\) P.Godwin, \(^\dagger\) R.Goetz, \(^\dagger\) R.Goetz,B.Goncharov, \(^\dagger\) G.Gonzalez, \(^\dagger\) J.M.Gonzalez Castro, \(^\dagger\) A.Gopakumar, \(^\dagger\) M.L.Gorodetsky, \(^\dagger\) S.E.Gossan, \(^\dagger\) M.Gosselin, \(^\dagger\) R.Gouaty, \(^\dagger\) A.Grado, \(^\dagger\) 26C.Graef, \(^\dagger\) M.

Granata, \(^\dagger\) A/grant, \(^\dagger\) S.Grassia, \(^\dagger\) C.Gray, \(^\dagger\) R.Gray, \(^\dagger\) G.Greco, \(^\dagger\) A.C.Green, \(^\dagger\) 48R.Green, \(^\dagger\) E.Mretary,S.HildT.Hinderer\(^\dagger\) 29,37,13D.Hoak, \(^\ddagger\) S.Hochheim, \(^\ddagger\) D.Hofman, \(^\ddagger\) A.M.Holgado, \(^\ddagger\) N.A.Holland, \(^\ddagger\) K.Holt, \(^\ddagger\) D.E.Holz, \(^\ddagger\) P.Hopkins, \(^\ddagger\) 68C.Horst, \(^\ddagger\) J.Hough, \(^\ddagger\) E.J.Howell, \(^\ddagger\) C.G.Hoy, \(^\ddagger\) A.Hreibi, \(^\ddagger\) Y.Huang, \(^\ddagger\) 12,31E.A.Huerta, \(^\dagger\) D.

Huet, \(^\ddagger\) B.Hughey, \(^\ddagger\) M.Hulko,S.Husa, \(^\dagger\) 101S.H.Huttner, \(^\dagger\) T.Huynh-Dinh, \(^\ddagger\) B.Idzkowski, \(^\ddagger\) A.Isss, \(^\ddagger\) 30,C.Ingram,55R.Inta85G.Intini,, \(^{155}32\) B.Irwin, \(^{117}H.N.Isa\) , \(^{\dagger}J.-M.Isac\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}I.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.

Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^ {\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\) , \(^{\dagger}J.-M.Isa\)\( ^{\dagger}J.-M.

Isa\), \(\mathbb { 4 } _ { \text{一} }\mathbb { I }\mathbb { m }\mathbb { l }\mathbb { r }\mathbb { a }\mathbb { n }\mathbb { s }\mathbb { k }\mathbb { t }\mathbb { l }\mathbb { e }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb {n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n

}\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { n }\mathbb { u }\mathbb { t }\mathbb { c } \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( {}^{39} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\\( ^{\dagger} \)\( ^{\dagger} \)\(

^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dagger} \)\( ^{\dbegin{array}{c}{\tt 4}} \)\( ^{\tt 48} \)\( ^{\tt 49} \)\( ^{\tt 50} \)\( ^{\tt 51} \)\( ^{\tt 52} \)\( ^{\tt 53} \)\( ^{\tt 54} \)\( ^{\tt 55} \)\( ^{\tt 56} \)\( ^{\tt 57} \)\( ^{\tt 58} \)\( ^{\tt 59} \)\( ^{\tt 60} \)\( ^{\tt 61} \)\( ^{\tt 62} \)\( ^{\tt 63} \)\( ^{\tt 64} \)\( ^{\tt 65} \)\( ^{\tt 66} \)\( ^{\tt 67} \)\( ^{\tt 68} \)\( ^{\tt 69} \)\( ^{\tt 70} \)\( ^{\tt 71} \)\( ^{\tt 72} \)\( ^{\tt 73} \)\( ^{\tt 74} \)\(

^{\tt 75} \)\( ^{\tt 76} \)\( ^{\tt 77} \)\( ^{\tt 78} \)\( ^{\tt 79} \)\( ^{\tt 80} \)\( ^{\tt 81} \)\( ^{\tt 82} \)\( ^{\tt 83} \)\( ^{\tt 84} \)\( ^{\tt 85} \)\( ^{\tt 86} \)\( ^{\tt 87} \)\( ^{\tt 88} \)\( ^{\tt 89} \)\( ^{\tt 90} \)\( ^{\tt 91} \)\( ^{\tt 92} \)\( ^{\tt 93} \)

\( ^{(3)}\mathrm{T}_{\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot .\cdot ..
\]

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-44

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-45

# (LIGO Scientific Collaboration and Virgo Collaboration)

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-46

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-47

B. P. ABBOTT et al.

PHYS. REV. X 9, 031040 (2019)

031040-48

152INFN, Sezione di Roma Tre, I-00146 Roma, Italy

153 ESPCI, CNRS, F-75005 Paris, France

$^{154}$ OzGrav, Swinburne University of Technology, Hawthorn, Victoria 3122, Australia

<sup>155</sup>University of Portsmouth, Portsmouth, PO1 3FX, United Kingdom

$^{156}$ Southern University and A&M College, Baton Rouge, Louisiana 70813, USA

$^{157}$ College of William and Mary, Williamsburg, Virginia 23187, USA

158 Centre Scientifique de Monaco, 8 quai Antoine Ier, MC-98000, Monaco

159Indian Institute of Technology Madras, Chennai 600036, India

160INFN Sezione di Torino, Via P. Giuria 1, I-10125 Torino, Italy

161 Institut des Hautes Etudes Scientifiques, F-91440 Bures-sur-Yvette, France

162 IISER-Kolkata, Mohanpur, West Bengal 741252, India

163 Whitman College, 345 Boyer Avenue, Walla Walla, Washington 99362, USA

<sup>164</sup>Université de Lyon, F-69361 Lyon, France

165 Hobart and William Smith Colleges, Geneva, New York 14456, USA

$^{166}$ Dipartimento di Fisica, Università degli Studi di Milano Bicocca,

Piazza della Scienza 3, 20126 Milano, Italy

$^{167}$ Dipartimento di Fisica, Università degli Studi di Torino, I-10125 Torino, Italy

168 Janusz Gil Institute of Astronomy, University of Zielona Góra, 65-265 Zielona Góra, Poland

$^{169}$ University of Washington, Seattle, Washington 98195, USA

170SUPA, University of the West of Scotland, Paisley PA1 2BE, United Kingdom

$^{171}$ Indian Institute of Technology, Gandhinagar Ahmedabad Gujarat 382424, India

<sup>172</sup>Université de Montréal/Polytechnique, Montreal, Quebec H3T 1J4, Canada

$^{173}$ Indian Institute of Technology Hyderabad, Sangareddy, Khandi, Telangana 502285, India

$^{174}$ International Institute of Physics, Universidade Federal do Rio Grande do Norte, Natal RN 59078-970, Brazil

175Villanova University, 800 Lancaster Avenue, Villanova, Pennsylvania 19085, USA

176Andrews University, Berrien Springs, Michigan 49104, USA

177Max Planck Institute for Gravitationalphysik (Albert Einstein Institute),

D-14476 Potsdam-Golm, Germany

178 Università di Siena, I-53100 Siena, Italy

$^{179}$ Trinity University, San Antonio, Texas 78212, USA

<sup>180</sup>Van Swinderen Institute for Particle Physics and Gravity, University of Groningen,

Nijenborgh 4, 9747 AG Groningen, Netherlands

Deceased.

GWTC-1: A GRAVITATIONAL-WAVE TRANSIENT CATALOG ...

PHYS. REV. X 9, 031040 (2019)

031040-49