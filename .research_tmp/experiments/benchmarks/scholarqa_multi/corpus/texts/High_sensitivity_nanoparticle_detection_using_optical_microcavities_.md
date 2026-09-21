# High sensitivity nanoparticle detection using optical microcavities

Tao Lu $^{a,b,1,2}$ , Hansuek Lee $^{a,1}$ , Tong Chen $^{a}$ , Steven Herchak $^{b}$ , Ji-Hun Kim $^{a}$ , Scott E. Fraser $^{a,c}$ , Richard C. Flagand, and Kerry Vahala $^{a,2}$

$^{a}$ Department of Applied Physics, MC 128-95, California Institute of Technology, Pasadena, CA 91125; $^{b}$ Department of Electrical and Computer Engineering, University of Victoria, Victoria, BC, Canada V8P 5C2; $^{c}$ Division of Biology, MC 139-74, California Institute of Technology, Pasadena, CA 91125; and $^{d}$ Division of Chemistry and Chemical Engineering, MC 210-41, California Institute of Technology, Pasadena, CA 91125

Edited* by Amnon Yariv, California Institute of Technology, Pasadena, CA, and approved February 25, 2011 (received for review December 3, 2010)

We demonstrate a highly sensitive nanoparticle and virus detection method by using a thermal-stabilized reference interferometer in conjunction with an ultrahigh-Q microcavity. Sensitivity is sufficient to resolve shifts caused by binding of individual nanobeads in solution down to a record radius of $12.5\mathrm{nm}$ , a size approaching that of single protein molecules. A histogram of wavelength shift versus nanoparticle radius shows that particle size can be inferred from shift maxima.

Additionally, the signal-to-noise ratio for detection of Influenza A virus is enhanced to 38:1 from the previously reported 3:1. The method does not use feedback stabilization of the probe laser. It is also observed that the conjunction of particle-induced backscatter and optical-path-induced shifts can be used to enhance detection signal-to-noise.

biosensor | nanodetector | optical microcavity

Label-free biosensing has been an active research area with applications to biomolecular interactions as well as early-stage disease diagnosis. Several techniques have been explored (1-6), and among these, microcavity sensors in the form of whispering gallery resonators have received considerable attention. In this method, a particle or molecule binding on the surface of the microcavity perturbs its optical properties, causing a resonant wavelength shift with magnitude that depends upon the particle's polarizability (proportional to volume for macroscopic particles).

Measurement of the shift enables observation of binding events in real time and can also be used to assess particle size. Silica microspheres (7) or silica microtoroids (8, 9) provide an easily functionalized detection surface that directly interacts with a high-quality-factor (Q) whispering gallery mode. The combination of small size and high Q endows these devices with excellent detection sensitivity.

Detection of an Influenza A (InfA) virion with an effective size around $100\mathrm{nm}$ in diameter has been reported by monitoring the resonance wavelength shift of a silica microsphere cavity (10) with a 3:1 signal-to-noise ratio (SNR).

Typically, measurements proceed by monitoring the transmission spectrum of a selected microcavity resonance. Wavelength shifts of the resonance, caused by particle binding, are measured by interpreting the scan voltage used to repetitively sweep the wavelength of a probe laser. Although the ultimate precision in measuring a shift depends upon the microcavity linewidth (and hence its Q factor), the conversion of the scan voltage to wavelength provides a significant source of error—even greater than the microcavity linewidth when Q factor is in the range of 10–100 million.

Indeed, the long-term frequency jitter associated with many commercial tunable lasers exceeds the linewidth of high-Q resonators such as microtoroids (8, 9). The stability of the mechanical scanning mechanism can also introduce error. Each of these sources of error are uncorrelated with the scan voltage. One approach to reduce such errors involves frequency-doubling a solid-state distributed feedback laser (11).

# Results and Discussion

In this work, we introduce a reference interferometer into the detection system to minimize the error contributions from frequency jitter and laser scan-voltage control. With this approach, a wavelength shift as small as several 10ths of a femtometer can be detected. No feedback control or stabilization of the laser system is provided (although this could be added). The experimental setup is shown in Fig. 1. A 680-nm tunable laser (Newport Velocity 6304) is driven by a voltage ramp signal, and the output of the laser is split into two branches by a directional coupler.

In one branch the laser power is used to monitor the resonance of a microtoroid immersed in an aqueous bath containing nanoparticles or biomolecules. Optical coupling to the microtoroid proceeds using a fiber taper (12, 13), before the signal is photodetected. Typically, the coupled power was kept in the range of $10\mu \mathrm{W}$ . In the other branch, the laser frequency is monitored using a reference interferometer.

The interferometer features two fiber optic paths of differing lengths, and is immersed in an ice-water bath within a $1\mathrm{ft}^3$ thermally and mechanically insulated styrofoam enclosure. The interferometer is allowed to thermally stabilize for several hours before use. The free spectral range (FSR) of the interferometer was $40.8\mathrm{MHz}$ (corresponding to a $4.9\mathrm{-m}$ differential length in the two fiber optic paths) for measurement of 12.5 and $25\mathrm{-nm}$ radius bead binding.

This FSR value was sufficiently wide so as to minimize the likelihood of the laser jittering beyond one FSR during typical measurement intervals. In $50~\mathrm{nm}$ radius bead measurements, an FSR of $235\mathrm{MHz}$ (equivalent $0.9\mathrm{-m}$ differential length) was used to ensure the maximum step size caused by bead binding events would be smaller than the FSR. The dual outputs of the interferometer were detected using a balanced homodyne detector (Thorlabs PDB120A, noise equivalent power $6\mathrm{pW / pHz}$ ) to reduce contributions from laser power noise.

Polarization was adjusted using a polarization controller (shown in Fig. 1). Because the photodetected output of the reference interferometer depends sinusoidally on the frequency of the laser (period set by interferometer FSR), the laser frequency (relative to an initial value) can be accurately measured (as opposed to being inferred by the scan voltage) at the moment when the microcavity resonance is excited in the other branch, which greatly reduces measurement noise, as demonstrated below.

In high-Q resonators, there is a subtlety to determination of resonance line center because resonances occur as doublets on

PNAS

Author contributions: T.L. and K.V. designed research; T.L., H.L., and K.V. performed research; J.-H.K. contributed new reagents/analytic tools; T.L., H.L., T.C., S.E.F., R.C.F., and K.V. contributed conceptual ideas; T.L., H.L., T.C., S.H., and K.V. analyzed data; and T.L., H.L., T.C., S.H., J.-H.K., S.E.F., R.C.F., and K.V. wrote the paper.

The authors declare no conflict of interest.

*This Direct Submission article had a prearranged editor.

Freely available online through the PNAS open access option.

T.L. and H.L. contributed equally to this work.

<sup>2</sup>To whom correspondence may be addressed. E-mail: taolu@ece.uvic.ca or vahala@caltech.edu.

5976-5979 | PNAS | April 12, 2011 | vol. 108 | no. 15

www.pnas.org/cgi/doi/10.1073/pnas.1017962108

![](dt=2026-06-09/ht=16/4dc9896f18adc9f1f117323995639e54b7385c91050a050cc75dd8526a5f05a4.jpg)

account of backscatter-induced splitting of the initially degenerate clockwise and counterclockwise whispering gallery modes (14, 15). Additionally, the splitting frequency itself can also be used to monitor protein molecules in an aqueous environment (16) and aerosol particles (17), thereby providing additional confirmation of particle binding.

To spectrally locat
e the split resonances for purposes of monitoring the binding-induced frequency shifts, and to also obtain information on any variation in splitting, the spectral sweeps were numerically fit to a theoretical split-resonance spectrum over a narrow range of frequencies (typically less than the interferometer FSR). The average location of the split resonances or the location of either resonance alone is then plotted versus time to monitor particle binding. We also observe a useful detection enhancement mechanism.

In some cases, the combined effect of the resonance shift and the induced-splitting can be used to increase signal-to-noise.

Interferometer Stability. To assess the stability of the interferometer, two independent interferometers, immersed in separate baths, (one having FSR of $40.8\mathrm{MHz}$ and a second having an FSR of $38.9\mathrm{MHz}$ ) were used to simultaneously monitor the scanning laser. The outputs of the interferometers were recorded at a sampling time interval of $10\mathrm{ms}$ determined by the laser scan rate over a span of $80\mathrm{s}$ as the tunable laser was scanned. The output from the scan generator was used to provide a trigger level for recording the output from the interferometers.

As an aside, in the actual experiment, the time trace of the interferometer transmission is recorded concurrently with the resonator transmission to provide accurate measurement of the resonance. The measured laser frequency from both interferometers is presented in Fig. 2, Inset (red trace for FSR of $40.8\mathrm{MHz}$ with a frequency fluctuation of $5\mathrm{MHz}$ ; blue trace for FSR of $38.9\mathrm{MHz}$ with a similar fluctuation).

The difference of these traces $(\Delta \nu = \nu_{1} - \nu_{2}$ on the left vertical axis; and $\Delta \lambda = \lambda_{2} - \lambda_{1}$ on the right vertical axis) is plotted in the main figure panel. The standard deviation of the frequency difference is $0.1\mathrm{fm}$ ( $60\mathrm{kHz}$ ). This number is further reduced to $0.06\mathrm{fm}$ ( $40\mathrm{kHz}$ ) with the adoption of a three-point moving average, displayed as the blue trace (this averaged trace is shown in the main figure panel, but shifted $1\mathrm{MHz}$ for clarity).

To compensate for the effect of frequency drift between the two interferometers, a moving standard deviation was first computed versus time over 10 consecutive points.

![](dt=2026-06-09/ht=16/21b6b9f7894a430230157827790bd3845bd501dad622e4190afa3dbec99d2d00.jpg)

The average of this time-dependent quantity over $80\mathrm{~s}$ is given in the lower right of the figure. The origin of the noise in this measurement will be studied elsewhere; however, it is also worth noting that the interferometers agree to within $2\mathrm{MHz}$ over the time interval of $80\mathrm{~s}$ , indicating a thermal stability better than $1\mathrm{mK}$ .

Polystyrene Nanobead Detection. To test the system, measurements were first performed using polystyrene beads at three distinct sizes. Beads in solution were obtained from Polyscience except for the $12.5\mathrm{-nm}$ beads which were obtained from Phosphorex. A syringe pump was used to inject a solution into a microaquarium containing a fiber-taper-coupled, microtoroid resonator.

After the solution containing beads was introduced into the aquarium, the syringe pump was turned off for the actual measurement to reduce fluctuations that are believed to be associated with hydrodynamic-induced instabilities in the taper-microtoroid coupling. Fig. 3A displays representative resonance wavelength versus time traces (sampled at a 10-ms interval) for solutions of polystyrene beads of radius $R = 50\mathrm{~nm}$ (red line), $R = 25\mathrm{~nm}$ (blue line), and $R = 12.5\mathrm{~nm}$ (green line).

The corresponding in-solution quality factors of the microtoroids used in these measurements were $8\times 10^{6}$ , $3\times 10^{7}$ , and $1\times 10^{8}$ , respectively. The higher values here should be taken as approximate. Even though coupled power levels were maintained low in the range of $10\mu \mathrm{W}$ to minimize thermal effects, some thermal effects were nonetheless present in the higher-Q devices. The 50- and $25\mathrm{-nm}$ nanobeads were diluted in Dulbecco's Phosphate Buffered Saline (DPBS) to achieve a concentration of $1\mathrm{pM}$ .

To enhance the probability of observing a maximum resonance shift, a $100\mathrm{-pM}$ solution was used for $R = 12.5\mathrm{-nm}$ bead detection, and the shift was measured with a three-point moving average to reduce noise.

In Fig. 3A, steps of 2.7, 6.4, and $12.3\mathrm{fm}$ at 31.4, 44.2, and $56.2\mathrm{s}$ , respectively, are present in the $R = 50\mathrm{-nm}$ scan (red trace), suggesting individual particle binding events. At $60.2\mathrm{s}$ , there is also a step down in the red trace suggesting that a bead has desorbed. The standard deviation for this measurement is $0.6\mathrm{fm}$ . For comparison purposes, the gray trace in the figure shows the same data scan except using the conventional scan-voltage method (i.e., without jitter compensation provided by the reference interferometer).

For the $R = 25\mathrm{-nm}$ beads, a step of $2.6 \pm 0.5$ fm is observed at $30.6~\mathrm{s}$ as displayed in the upper right inset, and steps of 1.5 and $2.2\mathrm{fm}$ are observed at 41.8 and $74.7\mathrm{s}$ , respectively. Finally, a step

PNAS

Lu et al.

PNAS April 12, 2011 vol. 108 no. 15 5977

SEDNEDS TVDSAHD

A50TO1B VANOLLYNDWOND NVAISDAHDOI

![](dt=2026-06-09/ht=16/ca8cdefbb748f5852423c2e5c4bfdb67b05be58baff8c534c9a142ad2479387b.jpg)

![](dt=2026-06-09/ht=16/fc7b17f25e8ff9acc79308850f729a8a90fadefaca49f0f7102de6b2f28c3de3.jpg)

of $0.4 \pm 0.2$ fm for the case of $R = 12.5$ -nm beads is observable in the green trace at $11\mathrm{s}$ with an SNR of 2. Here, the step amplitude can be enhanced by measuring on the longer-wavelength resonance (see Fig. 3A, Inset II) as both the average resonance wavelength and the split frequency increase upon binding. Indeed, this useful mechanism boosts the observed amplitude by over a factor of 2 ( $1.0 \pm 0.2$ fm). The step sizes appearing in the blue and green traces should be contrasted with the noise level of the gray trace.

Measured wavelength-shift step sizes from many measurements using different microtoroids and bead sizes are compiled in Fig. 3B. The data presented were compiled from 51, 11, and 15 distinct runs in which 10, 25, and 43 identifiable binding events were observed using $R = 12.5$ nm (green cross), $R = 25$ nm (blue cross), and $R = 50$ nm (red cross) beads, respectively. Various particle concentrations ranging from 0.1 to $100$ pM were also tested during the measurements.

For the smallest beads, femtomolar-range concentrations were insufficient to produce binding events of sufficient frequency and amplitude to observe. The amplitude of a binding/unbinding-induced wavelength step is determined by both the particle size and its proximity to the optical whispering gallery mode. Maximum shift occurs when the particle binds at the equator of the microtoroid where the optical field is maximum. This maximum value (computed using COM-SOL, ref. 18) is provided in Fig. 3.

For the largest particle size, we found it necessary to account for the actual field variation within

the particle. Error bars were obtained by computing the variance of 10 data points in the vicinity of the measured step. In addition to the measurement uncertainty, there is also a variation in toroid major diameter and bead diameter, both approximately $15\%$ .

Influenza A Virion Detection. To test the method for biosensing, binding of InfA virion diluted in DPBS was measured. Fig. 4A (trace A) provides a typical data scan i
n which virus binding is observed at a concentration of $1\mathrm{pM}$ . At $14.5\mathrm{s}$ , an $8.4 \pm 0.3$ -fm step is observed (also magnified in Fig. 4A, Inset I). Steps are also observed at $(23.6\mathrm{s}:6.0\mathrm{fm})$ , $(32.1\mathrm{s}:2.2\mathrm{fm})$ , $(33.5\mathrm{s}:2.3\mathrm{fm})$ , and $(38.1\mathrm{s}:2.4\mathrm{fm})$ , to indicate a few. Fig.

4B provides the histogram of binding counts and the corresponding step size over eight runs. In contrast to the bead experiments, unbinding events were less frequent, indicating a much stronger affinity force between InfA and the toroid surface. A maximum shift of $11.3 \pm 0.3\mathrm{fm}$ was observed in Fig. 4B, Inset, yielding an SNR of 38 and representing an improvement over the previously reported SNR of 3 (2). This result also agrees with the simulation.

# Conclusion

In conclusion, by employing a thermal-stabilized, reference interferometer to monitor wavelength shift in real time, we have achieved a record sensitivity for nanodetection using a microcavity biosensor. Sensing of $12.5\mathrm{-nm}$ radius polystyrene nanobeads was demonstrated; shift distribution data were measured for detection of beads having three different diameters and agreed well with theory. Greatly enhanced SNR detection of individual InfA virion was also demonstrated. Although our detection experiment was based on a silica microtoroid platform, the refer

![](dt=2026-06-09/ht=16/a01132518f37a267611ac33236403082b9d8847eea800e572ee51ee120118456.jpg)

![](dt=2026-06-09/ht=16/26cd7a398dea01bfef5d36d8633e3ee76861749af37eed91c45fe15b6cffc67a.jpg)

PNAS

PNAS

PNA

5978

Lu et al.

ence-interferometer method can be readily applied to other platforms such as microsphere and microdisk sensors and requires no feedback control of the probe laser source. Further improvement of detection sensitivity is possible by adopting a feedback control loop to stabilize the probe laser. Finally, we note that a mechanism in which both path length and backscatter effects enhance

step shift has been observed and is used to improve detection signal-to-noise.

ACKNOWLEDGMENTS. Fig. 1 was prepared by Benjamin Taylor and Xuan Du at Department of Electrical and Computer Engineering, University of Victoria. The work was supported in part by Defense Advanced Research Planning Agency and Natural Sciences and Engineering Research Council (Canada).

PNAS PNAS PNAS

SENEIDS TVISAHHDdDd

A09018 TANONIAYNNDNO 0N SISAHdOIB

Lu et al.

PNAS | April 12, 2011 | vol. 108 | no. 15 | 5979