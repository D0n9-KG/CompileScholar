# An Early-warning System for Electromagnetic Follow-up of Gravitational-wave Events

Surabhi Sachdev $^{1,2}$ , Ryan Magee $^{1,2}$ , Chad Hanna $^{1,2,3,4}$ , Kipp Cannon $^{5}$ , Leo Singer $^{6}$ , Javed Rana SK $^{1,2}$ , Debnandini Mukherjee $^{1,2}$ , Sarah Caudill $^{7}$ , Chiwai Chan $^{5}$ , Jolien D. E. Creighton $^{8}$ , Becca Ewing $^{1,2}$ , Heather Fong $^{5}$ , Patrick Godwin $^{1,2}$ , Rachael Huxford $^{1,2}$ , Shasvath Kapadia $^{9}$ , Alvin K. Y. Li $^{10}$ , Rico Ka Lok Lo $^{10}$ , Duncan Meacher $^{8}$ , Cody Messick $^{11}$ , Siddharth R. Mohite $^{8}$ , Atsushi Nishizawa $^{5}$ , Hiroaki Ohta $^{5}$ , Alexander Pace $^{1,2}$ , Amit Reza $^{12}$ , B. S.

Sathyaprakash $^{1,2,3,13}$ , Minori Shikauchi $^{5}$ , Divya Singh $^{1,2}$ , Leo Tsukada $^{5}$ , Daichi Tsuna $^{5}$ , Takuya Tsutsui $^{5}$ , and Koh Ueno $^{5}$ $^{1}$ Department of Physics, The Pennsylvania State University, University Park, PA 16802, USA $^{2}$ Institute for Gravitation and the Cosmos, The Pennsylvania State University, University Park, PA 16802, USA $^{3}$ Department of Astronomy and Astrophysics, The Pennsylvania State University, University Park, PA 16802, USA $^{4}$ Institute for CyberScience, The Pennsylvania State University, University Park, PA 16802, USA

$^{5}$ The University of Tokyo, Hongo 7-3-1 Bunkyo-ku, Tokyo 113-0033, Japan $^{6}$ NASA/Goddard Space Flight Center, Greenbelt, MD 20771, USA $^{7}$ Nikhef, Science Park, 1098 XG Amsterdam, The Netherlands $^{8}$ Leonard E.

Parker Center for Gravitation, Cosmology, and Astrophysics, University of Wisconsin-Milwaukee, Milwaukee, WI 53201, USA $^{9}$ International Centre for Theoretical Sciences, Tata Institute of Fundamental Research, Bengaluru 560089, India $^{10}$ LIGO Laboratory, California Institute of Technology, MS 100-36, Pasadena, CA 91125, USA $^{11}$ Department of Physics, University of Texas, Austin, TX 78712, USA $^{12}$ Department of Physics, Indian Institute of Technology Gandhinagar, Gujarat 382355, India $^{13}$ School of Physics and Astronomy, Cardiff University, Cardiff CF24 3AA, UK

Received 2020 September 8; revised 2020 November 2; accepted 2020 November 4; published 2020 December 21

# Abstract

Binary neutron stars (BNSs) will spend $\simeq 10 - 15$ minutes in the band of Advanced Laser Interferometer Gravitational-Wave Observatory (LIGO) and Virgo detectors at design sensitivity. Matched-filtering of gravitational-wave (GW) data could in principle accumulate enough signal-to-noise ratio (S/N) to identify a forthcoming event tens of seconds before the companions collide and merge. Here we report on the design and testing of an early-warning GW detection pipeline.

Early-warning alerts can be produced for sources that are at low enough redshift so that a large enough S/N accumulates $\sim 10 - 60$ s before merger. We find that about $7\%$ (49%) of the total detectable BNS mergers will be detected $60~\mathrm{s}$ (10 s) before the merger. About $2\%$ of the total detectable BNS mergers will be detected before merger and localized to within $100\mathrm{deg}^2$ (90% credible interval). Coordinated observing by several wide-field telescopes could capture the event seconds before or after the merger.

LIGO-Virgo detectors at design sensitivity could facilitate observing at least one event at the onset of merger.

Unified Astronomy Thesaurus concepts: Neutron stars (1108); Gravitational waves (678)

# 1. Introduction

2017 August 17 saw the beginning of a new era in multimeterner astronomy with the joint detection of gravitational waves (GWs) by the Laser Interferometer Gravitational-Wave Observatory (LIGO) and Virgo interferometers and the gamma-ray burst (GRBs) by the Fermi Gamma-ray Burst Monitor (GBM) and INTEGRAL satellite from the BNS coalescence, GW170817 (Abbott et al. 2017a, 2017b). The detection was followed by observations of the electromagnetic (EM) counterpart and afterglow by gamma-ray, UV, optical, infrared, and radio telescopes.

These observations triggered several important science results: (a) they settled a long-standing question about the origin of short GRBs (Abbott et al. 2017b), (b) provided a new tool for measuring cosmological parameters (with the first measurement of the Hubble constant using standard sirens; Abbott et al. 2017c), (c) confirmed the production of heavy elements in the aftermath of the merger (Arcavi et al. 2017b; Chornock et al. 2017; Drout et al. 2017; Kasen et al. 2017; see Abbott et al.

2017d for a more detailed list of references), (d) triggered many questions about the central engine producing GRBs and afterglows (Abbott et al. 2017e), and (e) set limits on the difference in the speed of GWs and light helping rule out certain alternative theories of gravity (Abbott et al. 2017e, 2019; Baker et al. 2017).

The GW alert was sent out $\sim 40$ minutes (LIGO Scientific Collaboration 2017a) and the sky localization $\sim 4.5$ hr (LIGO Scientific Collaboration 2017b) after the signal arrived on Earth. Among the factors that contributed to the delay were a nonstationary glitch in the Livingston interferometer and issues with the transfer of data from the Virgo detector to analysis sites delaying the sky localization of the event. By the time EM telescopes participating in the follow-up program received the alerts the source was below the horizon for them.

Apart from the GRB, which was observed $\sim 2$ s after the merger event, the first manual follow-up observations took place $\sim 8$ hr after the epoch of merger (Abbott et al. 2017b). The One-Meter, Two-Hemisphere (1M2H) team was the first to discover and announce the optical counterpart (Coulter et al. 2017a, 2017b). Five other teams took images of the optical transient within the hour of the 1M2H image: the Dark Energy Camera (Allam et al. 2017), the Distance Less Than $40\mathrm{Mpc}$ survey (Yang et al. 2017), Las Cumbres Observatory (Arcavi et al.

2017a), the Visible and Infrared Survey Telescope for Astronomy (Tanvir & Levan et al. 2017), and MASTER (Lipunov et al. 2017).

For a fraction of BNS events it will be possible to issue alerts up to $\delta t \sim 60$ s before the epoch of merger. Premerger or early warning detections will facilitate EM observations of the prompt emission, which encodes the initial conditions of the outflow and

THE ASTROPHYSICAL JOURNAL LETTERS, 905:L25 (6pp), 2020 December 20

© 2020. The American Astronomical Society. All rights reserved.

https://doi.org/10.3847/2041-8213/abc753

#

CrossMark

1

the state of the merger remnant. Indeed, early optical and ultraviolet observations are necessary to further inform our understanding of $r$ -process nucleosynthesis (Nicholl et al. 2017) and shock-heated ejecta (Metzger 2017), while prompt X-ray emission would reveal the final state of the remnant (Metzger & Piro 2014; Ciolfi & Siegel 2015; Siegel & Ciolfi 2016). Early observations made in the radio band could indicate premerger magnetosphere interactions (Most & Philippov 2020), and would test models that propose BNS as a possible precursor of fast radio bursts (Totani 2013; Wang et al. 2016; Dokuchaev & Eroshenko 2017).

The GSTLAL-based inspiral pipeline (hereafter shortened to GstLAL; Messick et al. 2017; Sachdev et al. 2019; Hanna et al. 2020) is a low-latency matched-filtering pipeline used to detect gravitational waves from compact binary coalescences in LIGO-Virgo data. Other low-latency detection pipelines running on LIGO-Virgo data include PyCBCLive (Nitz et al. 2018), MBTAOnline (Adams et al. 2016), SPIIR (Chu 2017), and the unmodeled search—CWB (Klimenko et al. 2016). In a seminal paper, Cannon et al.

(2012) described a computationally practical filtering strategy for near-real-time matched-filtering of GW data that could produce early-warning triggers. This work describes the foundations of GstLAL, which has been detecting GWs in low-latency since the first observing run (O1) of the Advanced LIGO and Virgo detectors. Cannon et al. (2012
) also discussed the expected rates of BNS events that could be detectable before merger and prospects for their localizations based on theoretical signal-to-noise ratio $(\mathrm{S} / \mathrm{N})$ and Fisher estimates.

There have been other studies examining theoretical potentials of premerger BNS detections, such as Chu et al. (2016) and Akcay (2019). In this Letter, for the first time, we show the implementation of a search that can detect BNSs premerger and provide early warnings to other observatories in practice; we examine the performance of GstLAL in recovering BNS systems before merger by running it over a month of simulated data with added signals. Based on the median rate of BNS mergers deduced from GW170817 and GW190425 (Abbott et al.

2020), our studies suggest alerts could be issued $10\mathrm{s}$ $(60~\mathrm{s})$ before merger for 24 (3) BNS systems over the course of one year of observations of a three-detector Advanced network operating at design sensitivity. Our results broadly agree with the estimates of Cannon et al. (2012). In addition, we provide the distribution of realistic sky localizations (all-sky localizations quoted are $90\%$ credible intervals unless stated otherwise) for various times before merger, using a rapid Bayesian localization tool, BAYESTAR (Singer & Price 2016).

We find that based on current BNS merger rate estimates, a $\mathcal{O}(1)$ event will be both detected before merger and localized to $100\mathrm{deg}^2$ .

These results assume zero latency from data transfer, calibration, filtering, and follow-up processes. In real application, these latencies will need to be subtracted from the premerger times at which we can provide alerts. In the latest observing run of the Advanced LIGO-Virgo detectors (O3), these latencies accounted for $\sim 20$ s of delay in alerts, but eventually we hope to be able to reduce it to $\sim 7$ s for the early warning alerts. The remainder of the Letter is structured as follows: we discuss the pipeline and simulations used in Section 2, the prospects of rapid sky localization of premerger candidates in Section 3, and broader implications of coincident GW and EM observation in Section 4.

# 2. Simulation

We assess the prospects of premerger alerts with an Advanced LIGO-Virgo network at design sensitivity. For this study, we generate one month of stationary Gaussian data recolored to Advanced LIGO and Advanced Virgo design sensitivities. $^{14}$

We generate a population of 1,918,947 simulated BNS signals, henceforth referred to as injections, using the SpinTaylorT4 waveform model (Buonanno et al. 2004). Both source-frame component masses are drawn from a Gaussian distribution between $1.0M_{\odot} < m_{1}$ , $m_{2} < 2.0M_{\odot}$ with a mean mass of $1.33M_{\odot}$ and standard deviation of $0.09M_{\odot}$ , modeled after observations of galactic BNSs (Özel & Freire 2016). The neutron stars in the population are nonspinning, motivated by the low spins of BNSs expected to merge within a Hubble time (Burgay et al. 2003; Zhu et al.

2018). The signals are distributed uniformly in comoving volume up to a redshift of $z = 0.2$ . We reject 1,659,747 injections with LIGO-Hanford or LIGO-Livingston S/Ns below 3 to reduce the computational load and inject the remaining signals in the Gaussian data. We do not expect the search to recover signals with such small S/Ns, so no bias is introduced in rejecting these.

We use the offline configuration of the GSTLAL pipeline (Messick et al. 2017; Sachdev et al. 2019; Hanna et al. 2020) to recover the remaining 259,200 injected into the Gaussian data described above. GSTLAL has been successfully detecting compact binary coalescences in low-latency since O1 (Abbott et al. 2016) and is so far the only pipeline to detect a BNS in low latency (Abbott et al. 2017a, 2020).

# 2.1. GSTLAL Methods

Matched-filtering GW searches use a template bank (Owen & Sathyaprakash 1999) containing a set of GW waveforms covering the desired parameter space. GstLAL divides the template bank into several subbanks by grouping templates that respond to noise in a similar fashion based on their intrinsic parameters (Messick et al. 2017; Sachdev et al. 2019). It then uses the LLOID method (Cannon et al. 2012) to construct orthogonal basis filters from the subbanks by performing in order multibanding and singular value decomposition (SVD; Cannon et al. 2010) of each time-slice.

The data are cross-correlated with the basis filters to produce GW candidates. Candidates with S/Ns below 4.0 are discarded to reduce the volume of triggers. Candidates that survive this step are assigned a log-likelihood ratio, $\log \mathcal{L}$ . The log-likelihood ratio ranks candidate events by their S/N, the sensitivity of each detector at the time of the trigger, an autocorrelation-based signal consistency test $(\xi^2)$ , and (for coincident triggers) the time and phase delays between participating interferometers (Cannon et al. 2015; Messick et al. 2017; Sachdev et al.

2019; Hanna et al. 2020). A template-dependent factor, $\log P(\theta_k|\mathrm{signal})$ where $\theta$ denotes the template, is included in the log-likelihood ratio to account for the population mass model (Fong 2018) of signals. The distribution of log-likelihood ratio for noise triggers is created by sampling the

THE ASTROPHYSICAL JOURNAL LETTERS, 905:L25 (6pp), 2020 December 20

Sachdev et al.

14 We use the power spectral densities provided in https://dcc.ligo.org/LIGOT0900288/public and https://dcc.ligo.org/LIGO-P1200087/public for the Advanced LIGO and Advanced Virgo interferometers, respectively. We assume that LIGO-Hanford and LIGO-Livingston will reach the same design sensitivities.

2

noise distributions of the parameters it depends on, and all candidates are subsequently assigned a false-alarm rate (FAR) to describe how often a candidate with a log $\mathcal{L}$ at least as high as its own is expected to be produced from noise fluctuations.

# 2.2. Early-warning Methods

In this search, we used a stochastically generated template bank (Harry et al. 2009; Privitera et al. 2014) with nonspinning components between masses $0.95M_{\odot} < m_1$ , $m_2 < 2.4M_{\odot}$ ; bounds chosen to account for edge effects and redshift, and chirp mass $\in (0.9M_{\odot}, 1.7M_{\odot})$ ; and bounds chosen based on the Gaussian population described above. We model the GW emission from $10\mathrm{Hz}$ to merger using the TaylorF2 (Sathyaprakash & Dhurandhar 1991; Blanchet et al. 1995, 2005; Buonanno et al. 2009) waveform.

The resulting template bank has a minimum match of $98\%$ (Owen & Sathyaprakash 1999) and consists of 80,679 waveforms. The template-dependent factor to account for the population mass model used in the log-likelihood ratio is modeled as a Gaussian in chirp mass with a mean of $1.18M_{\odot}$ and a standard deviation of $0.055M_{\odot}$ . The mean chirp mass is derived from the Gaussian component mass distribution described in Section 2 at a redshift of $z = 0.02$ .

We repeat the search six times, using the same template bank and the same data set including injections, to determine the pipeline's performance at various times before merger. The searches begin filtering at $10\mathrm{Hz}$ , but complete filtering at different frequencies. In particular, we choose 29, 32, 38, 49, 56, and $1024\mathrm{Hz}$ to analyze signal recovery at (approximately) 58, 44, 28, 14, 10, and 0s before merger. We will refer to each ending frequency configurations as a different "run" in the discussion that follows.

In practice, these ending frequencies are only approximate, since we chose to align the waveforms that are grouped together before performing the SVD (see Section 2.1) such that the waveforms in each subbank provide the same premerger time. The times before merger quoted here are the median times for each run. In our simulation, this time ranged from $\sim 6$ to $99\mathrm{s}$ between the six runs. While performin
g multibanding, the waveforms belonging to a subbank are time sliced and each slice is sampled according to the highest Nyquist frequency in that subbank and time-slice.

However, for these runs we fixed the sample rate of the final time-slice at $2048\mathrm{Hz}$ so that the $\xi^2$ is calculated at the full frequency resolution. As the bandwidth of the search is decreased, the variance associated with the recovered end time, phase, and $\mathrm{S} / \mathrm{N}$ grows. We account for increased uncertainty in the signal end time by extending the time window in which we search for coincident signals to $10\mathrm{ms}$ plus light travel time. We repeat the procedure described in Hanna et al.

(2020) for each analysis to account for bandwidth-related changes to the covariance matrix and construct signal distributions for time and phase delays for each of the runs. In addition, we tuned the binning and sampling of the $\mathrm{S} / \mathrm{N}$ and $\xi^2$ histograms that are used to calculate the distribution of the log-likelihood ratio of noise triggers that defines the background model of the search. In the absence of any simulated signals in the Gaussian data, we expect the foreground of our runs to agree with the background model computed by the search.

We confirmed that for each of the six runs, on excluding the simulated signals, the distribution of the log-likelihood ratio of the candidates agreed with the background model computed by the search.

Table 1 Sensitive Spacetime Volume $(\langle VT\rangle)$ of the Six Runs and the Expected Number of Signals $(N_{\mathrm{signals}})$ per Year Based on the Median BNS Merger Rate

![](dt=2026-03-14/ht=04/c2990dc95b241fbdee4eaafc19aeb2c52082d2a12b1f8942c00c4d1253a4830d.jpg)

<table><tr><td>fhigh (Hz)</td><td>&lt;VT&gt;(Gpc3a)</td><td>Nsignalsa(a-1)</td><td>Nlow - Nhigh(a-1)</td></tr><tr><td>29</td><td>2.55 × 10-4</td><td>3.21</td><td>0.775–8.71</td></tr><tr><td>32</td><td>3.84 × 10-4</td><td>4.84</td><td>1.17–13.2</td></tr><tr><td>38</td><td>7.23 × 10-4</td><td>9.12</td><td>2.20–24.8</td></tr><tr><td>49</td><td>1.45 × 10-3</td><td>18.2</td><td>4.41–49.5</td></tr><tr><td>56</td><td>1.88 × 10-3</td><td>23.6</td><td>5.71–64.2</td></tr><tr><td>1024</td><td>3.86 × 10-3</td><td>48.7</td><td>11.8–132</td></tr></table>

Note. We also show the expected range of events based on the uncertainty in the BNS merger rate $(N_{\mathrm{low}} - N_{\mathrm{high}})$ .

# 2.3. Results

We consider any injection that is recovered with a $\mathrm{FAR} \leqslant 1/(30$ days) to be found by our pipeline in each of the six different runs. We can compute the expected number of signals for each run based on the sensitive spacetime volume of each run at our chosen FAR threshold and on the local BNS merger rate, $250 - 2810\mathrm{Gpc}^{-3}\mathrm{a}^{-1}$ (90% credible interval; Abbott et al. 2020). The sensitive spacetime volume of the search at a given FAR threshold is then estimated as

$$
\langle V T \rangle = \langle V T \rangle_ {\text {i n j e c t e d}} \frac {N _ {\text {r e c o v e r e d}}}{N _ {\text {t o t a l s i m s}}}, \tag {1}
$$

where $N_{\mathrm{recovered}}$ is the number of recovered injections at the given FAR. This assumes that the injections have not been restricted to space or time that the pipeline could have been sensitive to. For this signal distribution, the simulated signals probed a spacetime volume of $VT_{\mathrm{injected}} = 0.178\mathrm{Gpc}^3\mathrm{a}$ . The results are shown in Table 1. We expect 12-132 BNSs per year for a three-detector Advanced network at design sensitivity, about half of which will be detected 10 s before merger and 1-9 events will be detected 1 minute before merger.

At a FAR of $1/30$ days, based on the current median BNS merger rate $(1035\mathrm{Gpc}^{-3}\mathrm{a}^{-1})$ , average of the two median rates from Abbott et al. (2020), the contamination fraction from noise for the $29\mathrm{Hz}$ run (60 s) before merger is $79\%$ , going down to $20\%$ for the full bandwidth run. The contamination fraction is higher for runs that provide the earliest triggers. This suggests a natural method to vet early-warning triggers; triggers that are identified in an "early" band but not later bands are likely to be noise.

# 3. Sky Localization of Early-warning Alerts

The primary goal of providing premerger alerts for BNSs is to facilitate EM observations before and/or at merger. The earliest alerts we could provide will be $\sim 60$ s before merger; therefore, to achieve this goal it is crucial that we provide rapid and accurate sky localizations. LIGO-Virgo use BAYESTAR (Singer & Price 2016) to generate rapid localizations, which is a fast Bayesian algorithm that can reconstruct positions of GW transients using the output provided by the matched-filtering searches.

We generate the S/N time series of all injections that pass the FAR threshold for each run and provide these to BAYESTAR in order to localize the signals. The results are shown in Figure 1. We show the cumulative histograms of the $90\%$ credible interval of sky localizations of the injections that pass the FAR threshold in each run. The right vertical axis

THE ASTROPHYSICAL JOURNAL LETTERS, 905:L25 (6pp), 2020 December 20

Sachdev et al.

3

![](dt=2026-03-14/ht=04/58ace4ac9f48b57a1a0be102edea7dc56b6d42d356ffac769047bd471745532c.jpg)

shows the expected number of events per year as a function of the largest localization area based on the median merger rate; the left vertical axis shows this number as a fraction of the total injections recovered at the full bandwidth. In addition we show cumulative histograms of luminosity distances of events that pass the FAR threshold in Figure 2. These results can be easily reinterpreted with any update in the BNS merger rate.

# 4. Discussion

Ideally we want the signals to be well localized in sky given the small fields of view (FOVs) of optical telescopes. Figure 1 shows that at least one event per year will be both detected before merger and localized to within $100\mathrm{deg}^2$ . Furthermore, if we consider the "searched area," defined as the area searched in the sky according to the localization probability distribution before finding the true location of the event, about nine events per year ( $\sim 18\%$ of total detectable BNSs) will be both detected before merger and found before searching over $100\mathrm{deg}^2$ .

Additionally, the searched area can be reduced by using galaxy catalogs to inform imaging strategies (Hanna et al. 2014). Events we are able to provide early warnings for, especially the well-localized ones, will be the ones that are the closest to us further enabling better follow-up. At least 1 event per year ( $3.4\%$ of the total) detected 60 s before merger will be within $100\mathrm{Mpc}$ and about 13 events per year ( $28\%$ of the total) will both be detected before merger and lie within $200\mathrm{Mpc}$ (Figure 2).

Wide-field optical transient facilities such as the BlackGEM array $(0.65\mathrm{m} / 2.7\mathrm{deg}^2$ per telescope) with 3 telescopes planned in the first phase of operation eventually expanding to 15 telescopes (BlackGEM 2020), the Zwicky Transient Facility $(1.2\mathrm{m}/47\mathrm{deg}^2$ ; ZTF 2020), the Dark Energy Camera $(4\mathrm{m} / 3.8\mathrm{deg}^2$ ; Flaughter et al. 2015), the Rubin Observatory $(8.4\mathrm{m} / 9.6\mathrm{deg}^2$ ; LSST 2020), the Swope Telescope $(1\mathrm{m} / 7\mathrm{deg}^2$ ; SWOPE 2020), the Subaru Telescope $(8.2\mathrm{m} / 1.7\mathrm{deg}^2$ ; Subaru 2020), etc., operated in "target of opportunity" mode will be most fitting for the optical follow-up of well-localized events. Events with larger

![](image)
57adc2ebf8bf8a3d670fb493b2efb.jpg)

localization areas will be useful to alert space telescopes, such as the Fermi-GBM (all-sky) and the Swift Observatory (hosting the Burst Alert Telescope with an FOV of $\sim 10,000\mathrm{deg}^2$ and can localize events with an accuracy of $1^{\prime} - 4^{\prime}$ within 15 s, X-Ray Telescope, and Ultraviolet Optical Telescope). Premerger GW detection will be especially helpful to identify subthreshold GRBs for off-axis BNS mergers (Tohuvavohu et al. 2020).

Radio telescopes with large FOVs of hundreds of square degrees such as the Murchison Widefield Array (MWA 2020), the National Radio Astronomy Observatory (NRAO 2020) consisting of several telescope arrays, the Giant Metrewave Radio Telescope (GMRT 2020), the Owens Valley Long Wavelength Array (OVRO-LWA 2020), and (under construction) the Square Kilometre Array (SKA 2020), etc., can use even the poorly localized early-warning alerts. Callister et al.

(2019) have demonstrated a search that looks for radio signals coincident with GW alerts by buffering the data of OVRO-LWA, to look for signals coincident with the GW170104 (Abbott et al. 2017f). Early-warning alerts will enable buffering of the radio data at significantly higher time resolution. James et al. (2019) also describe using negative-latency BNS merger alerts to detect prompt radio bursts with MWA.

Early-warning alerts will also be useful for ground-based gamma-ray detector facilities such as the Cherenkov Telescope Array, which consists of several fast slewing telescopes (CTA 2020). They can slew within tens of seconds but must be pointing at the source at the time of merger, since there is no hypothesized afterglow at the high energies these telescopes can detect.

The premerger latencies described in the Letter are the median latencies from each run. The exact premerger latencies depend on the template masses, and range from 6 to $99\mathrm{s}$ . The premerger times quoted in this Letter also assume zero latency from data transfer, calibration, and the matched-filtering processes. In O3, this latency was about $\sim 20$ s; our goal is to bring this latency down to $\sim 7$ s for the smaller bandwidth (early-warning) configurations.

The low-latency GsTAL pipeline recently participated in a test of the LIGO-Virgo early-warning infrastructure and issued the first test alerts and retractions for premerger candidates

THE ASTROPHYSICAL JOURNAL LETTERS, 905:L25 (6pp), 2020 December 20

Sachdev et al.

4

(Guide 2020; LIGO Scientific Collaboration 2020). This test will be described in more detail in a future publication.

This work was supported by the NSF grant OAC-1841480. S.S. is supported by the Eberly Research Funds of Penn State, The Pennsylvania State University, University Park, Pennsylvania. D.M. acknowledges the support of NSF PHY-1454389, ACI-1642391, and OAC-1841480. S.R.M. thanks the LSSTC Data Science Fellowship Program, which is funded by LSSTC, NSF Cybertraining grant #1829740, the Brinson Foundation, and the Moore Foundation; his participation in the program has benefited this work. B.S.S. is supported in part by NSF grant No. PHY-1836779 and AST-1708146.

Computations for this research were performed on the Pennsylvania State University's Institute for Computational and Data Sciences Advanced CyberInfrastructure (ICDS-ACI). We also thank the LIGO Laboratory for use of its computing facility to make this work possible. The research leading to these results has also received funding from the European Union's Horizon 2020 Programme under the AHEAD2020 project (grant agreement No. 871158).

Data: The data that were used to infer the results in this Letter are described in https://gstlal.docs.ligo.org/ewgw-data-release (Sachdev et al. 2020).

Software: The analysis of the data and the detections of the simulation signals were made using the GSTLAL-based inspiral software pipeline (Cannon et al. 2012; Privitera et al. 2014; Messick et al. 2017; Sachdev et al. 2019; Hanna et al. 2020). These are built on the LALSuite software library (LIGO Scientific Collaboration 2018). The sky localizations made use of ligo.skymap<sup>15</sup>, which uses Astropy,<sup>16</sup> a community-developed core Python package for Astronomy (Astropy Collaboration et al. 2013; Price-Whelan et al. 2018). The plots were prepared using Matplotlib (Hunter 2007).

# ORCID iDs

# References

THE ASTROPHYSICAL JOURNAL LETTERS, 905:L25 (6pp), 2020 December 20

Sachdev et al.

15 https://Iscsoft.docs.ligo.org/ligo.skymap

16 http://www.astrophy.org

5

THE ASTROPHYSICAL JOURNAL LETTERS, 905:L25 (6pp), 2020 December 20

Sachdev et al.

6