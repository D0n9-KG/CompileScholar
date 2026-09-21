# BIOPHYSICS

# Label-free optical detection of single enzyme-reactant reactions and associated conformational changes

Eugene Kim,\* Martin D. Baaske,\* Isabel Schuldes,\* Peter S. Wilsch,\* Frank Vollmer\*

Monitoring the kinetics and conformational dynamics of single enzymes is crucial to better understand their biological functions because these motions and structural dynamics are usually unsynchronized among the molecules. However, detecting the enzyme-reactant interactions and associated conformational changes of the enzyme on a single-molecule basis remains as a challenge to established optical techniques because of the commonly required labeling of the reactants or the enzyme itself.

The labeling process is usually nontrivial, and the labels themselves might skew the physical properties of the enzyme. We demonstrate an optical, label-free method capable of observing enzymatic interactions and associated conformational changes on a single-molecule level. We monitor polymerase/DNA interactions via the strong near-field enhancement provided by plasmonic nanorods resonantly coupled to whispering gallery modes in microcavities.

Specifically, we use two different recognition schemes: one in which the kinetics of polymerase/DNA interactions are probed in the vicinity of DNA-functionalized nanorods, and the other in which these interactions are probed via the magnitude of conformational changes in the polymerase molecules immobilized on nanorods. In both approaches, we find that low and high polymerase activities can be clearly discerned through their characteristic signal amplitude and signal length distributions.

Furthermore, the thermodynamic study of the monitored interactions suggests the occurrence of DNA polymerization. This work constitutes a proof-of-concept study of enzymatic activities using plasmonically enhanced microcavities and establishes an alternative and label-free method capable of investigating structural changes in single molecules.

2017 © The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. Distributed under a Creative Commons Attribution NonCommercial License 4.0 (CC BY-NC).

# INTRODUCTION

Enzymes fulfill a plethora of metabolic functions in all living organisms. In many cases, enzymatic activity is closely connected to changes in the enzymes' conformation often involving the transition through multiple substates. One of the most important and, perhaps, most studied enzymes is DNA polymerase, which is present in all cells and responsible for replicating genetic information. Outside of the actual metabolisms, it is used for important biological applications, such as polymerase chain reaction (PCR) and DNA sequencing.

The enzymatic activity of DNA polymerase involves multiple steps, such as the binding of primer-hybridized template DNA, insertion of a deoxynucleoside triphosphate (dNTP), and incorporation of dNTP, thereby extending the strand by 1 nucleotide (nt). Each step of such a catalytic process is accompanied by conformational changes of the DNA polymerase. These transitions, together with the corresponding reaction pathways, have an intrinsically transient nature and have been vastly studied via single molecule-based techniques.

The most widely used method is perhaps single-molecule Förster resonance energy transfer (FRET), which resolves the dynamics of DNA/polymerase interactions and the associated structural changes by measuring distances between labels attached to specific polymerase domains or DNA strands (1-5). Despite its great contribution to the extension of knowledge on the reaction mechanisms of DNA polymerase, this method intrinsically requires chemical modification of the enzyme to attach labels; hence, it can cause the studied enzyme to deviate from its natural kinetics.

Inherent physical processes, such as photobleaching and large background signals, also limit the applicability of FRET (6). As a result, there is a great demand for

Max Planck Institute for the Science of Light, Staudtstrasse 2, 91058 Erlangen, Germany.  
*Corresponding author. Email: eugene.kim@mpl.mpg.de (E.K.); martin.baaske@mpl.mpg.de (M.D.B.); f.vollmer@exeter.ac.uk (F.V.)  
†These authors contributed equally to this work.  
‡Present address: Chair for Crystallography and Structural Physics, University of Erlangen-Nuremberg, Staudtstrasse 3, Erlangen, Germany.  
§Present address: Living Systems Institute, School of Physics, University of Exeter, Exeter EX4 4QD, U.K.

label-free methodologies that can potentially broaden the scope of single-enzyme studies and complement the information obtained using label-based methods.

Here, we establish a label-free optical sensor platform capable of monitoring the kinetics and conformational dynamics of single enzymes, thus extending our previous bulk-sensing approach (7). We show that the kinetics of single-molecule DNA/polymerase (sm-DNA/Pol) interactions and the related conformational transitions of the polymerase can be studied using plasmonically enhanced whispering gallery mode (WGM) microcavity sensors (8-15).

Our sensor recognizes conformational changes and the motion of single enzymes as shifts of the cavity's optical resonance wavelength induced by the perturbation of the highly localized electric field at the tips of nanorods (NRs). The magnitude and sign of these shifts are proportional to the change in the electric field intensity integrated over the volume of the molecule. For an increasing (decreasing) integrated intensity, the induced resonance shift is toward longer (shorter) wavelengths. This can be applied for the study of molecular kinetics as follows.

When a molecule enters the enhanced electric near field in the vicinity of the NRs, it causes a spectral red shift with an increasing magnitude as it moves toward the field's intensity maximum. This is followed by a blue shift with the same magnitude when the molecule moves away from the NRs and leaves the near field completely.

This concept also holds true when a molecule that is immobilized on the NRs changes its shape (that is, conformational state) in such a manner that the change in the volume-integrated field intensity is sufficient to cause recognizable spectral shifts of the WGM's position in either direction.

Here, we use these mechanisms to probe sm-DNA/Pol interactions with two different approaches: immobilization of DNA on the NRs for the study of sm-DNA/Pol interaction kinetics and immobilization of polymerase on the NRs for the observation of specific conformational transitions accompanied by its interaction with DNA molecules. In both cases, the statistical analysis of our sensor's signals allows us to discern the different activity levels of three polymerase species: the Klenow fragment

SCIENCE ADVANCES | RESEARCH ARTICLE

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

1 of 8

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

of Escherichia coli DNA polymerase I (KF) and DNA polymerases from Thermus aquaticus (Taq) and Pyrococcus furiosus (Pfu). We also study the sm-DNA/Pol interaction kinetics and associated conformational changes with respect to the type of DNA [primer/template DNA (ptDNA) and single-stranded DNA (ssDNA)], temperature, and the presence of dNTPs.

# RESULTS AND DISCUSSION

# Method for monitoring single-molecule DNA/polymerase interactions

The experimental setup used for monitoring interaction kinetics between polymerase and DNA is depicted in Fig. 1A. A fused silica microsphere with a diameter of $\sim 80$ to $100\mu \mathrm{m}$ serves as a WGM resonator and is placed inside a liquid sample cell made of polydimethylsiloxane. A Peltier element, which is attached to the wall of the sample cell, allows for temperature regulation of the liquid. WGMs are excited via frustrated total internal reflection of a wavelength-tunable laser beam ( $\lambda_{\mathrm{c}}\approx 642$ and $780~\mathrm{nm}$ ) focused on t
he surface of a prism. The resonance wavelengths of WGMs are then determined from the transmission spectra obtained by sweeping the laser wavelength with a frequency of $50\mathrm{Hz}$ using a modified centroid method (14).

Single-molecule sensitivity can then be achieved by using the plasmonic near-field enhancement provided by gold NRs, whose longitudinal surface plasmon resonances match the laser's wavelength (12-17). For this, the NRs are immobilized on the resonator (see Methods for the chemical protocols) in a directly monitored process, allowing one not only to count the number of deposited NRs but also to determine if

their long axes are aligned reasonably parallel to the polarization of the electric field via the binding-induced linewidth broadening and wavelength shifts of the monitored WGM (section S1). The local perturbations of the electric field near the NRs, as induced by sm-DNA/Pol interactions, can then be recognized as shifts $\Delta \lambda$ in the spectral position of WGMs (Fig. 1B).

As aforementioned, the magnitude and sign of these shifts are proportional to the changes in the electric field intensity integrated over the volume occupied by the molecule $\nu_{\mathrm{m}}(t)$ at the times $t_1$ and $t_2 = t_1 + \Delta t$ (where $\Delta t = 20$ ms is the time between two laser sweeps) and to the molecule's polarizability in excess to the medium $\alpha_{\mathrm{e}}$ (assuming a constant and isotropic molecular polarizability) (9, 10, 18)

$$
\begin{array}{l} \Delta \lambda^ {\infty} \alpha_ {e} \left(\int_ {v _ {m} (t _ {2})} | E (r) | ^ {2} d V - \int_ {v _ {m} (t _ {1})} | E (r) | ^ {2} d V\right) \\ = \alpha_ {\mathrm {e}} \left(I \left(t _ {2}\right) - I \left(t _ {1}\right)\right) = \alpha_ {\mathrm {e}} \Delta I \tag {1} \\ \end{array}
$$

where $E(r)$ denotes the unperturbed electric field in the absence of the polymerase. However, this does not consider that the process of sweeping the laser over the spectral range occupied by the resonant mode (indicated as $\lambda_{\mathrm{m}}$ in Fig. 1B) itself requires a certain time $\tau_{\mathrm{m}}$ . Consequently, each experimentally measured volume-integrated intensity $I_{\exp,k}$ for the $k$ th sweep of the laser originates from an averaging process

$$
I _ {\exp . k} = \bar {I} _ {k} = \left(\tau_ {\mathrm {m}}\right) ^ {- 1} \int_ {t _ {0}} ^ {t _ {0} + \tau_ {\mathrm {m}}} I (t) d t \tag {2}
$$

![](dt=2026-04-08/ht=14/aed1160fe860353e1c1b23daa56ea6c68fa09e3f22a544dbd2c2bba2153143ac.jpg)

![](dt=2026-04-08/ht=14/bf1c09219d9ecd556298203f6fe88cbf4d75f73a8b2765aea4d24e0df7f54141.jpg)

![](dt=2026-04-08/ht=14/49e6fea1baf82cdbe4e10e0398229b11e2792112095126128a1eef8d5b26f2e1.jpg)

![](dt=2026-04-08/ht=14/a285ed3640790d9b90b5b459a22f78feba9d93797f4a5102e2c28e4e128d1496.jpg)

![](dt=2026-04-08/ht=14/a0697db741e2abfdf91ef5bb38fc4e890ee81cc1474692b9780178fdb19a458c.jpg)

SCIENCE ADVANCES | RESEARCH ARTICLE

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

2 of 8

where $t_0$ is the time in which the excitation of the mode begins. As a result, the experimentally obtained shifts are

$$
\Delta \lambda_ {k} \propto \alpha_ {\mathrm {e}} (\bar {I} _ {k} - \bar {I} _ {k - 1}) = \alpha_ {\mathrm {e}} \Delta \bar {I} _ {k} \tag {3}
$$

Therefore, our sensor can only recognize molecular interactions that keep the analyte molecules confined temporally on the order of $\tau_{\mathrm{m}} \approx 200~\mu \mathrm{s}$ and spatially within the plasmonic hotspots (that is, near the NR tips).

Molecular processes shorter than $\tau_{\mathrm{m}}$ but occurring repeatedly during $\tau_{\mathrm{m}}$ can also be recognized but with reduced magnitudes, whereas one-time events shorter than $\tau_{\mathrm{m}}$ (for example, freely diffusing analyte molecules near the hotspots) are unlikely to be recognized as their $\bar{I}_k$ is significantly lower (section S9) (14). To study the sm-DNA/Pol interactions, we take two approaches: The first approach (Fig. 1C, top) is based on the polymerase interacting with DNA strands immobilized on the NRs (henceforth referred to as the immo-DNA scheme).

In this case, the shifts occur because of the changes in $\bar{I}$ as polymerase molecules are attached and detached from the DNA strands, consequently moving in and out of the areas with high field intensities (Fig. 2, A and D). For the second approach (Fig. 1C, bottom), the polymerase molecules are immobilized on the NRs (henceforth referred to as the immo-Pol scheme), and the observed shifts are caused by changes in $\bar{I}$ due to the polymerase changing its conformational states accompanied by its interaction with DNA strands (Fig. 2, B, C, and E).

Using both approaches, we obtain similar transient signal patterns (so-called "spikes") arising from the sm-DNA/Pol interactions (Fig. 1D). These are composed of an initial red shift of the resonance position as a molecular interaction starts and a consequent return to the unperturbed mode position (blue shift) as the interaction ceases. Each individual spike exceeding the wavelength noise $\sigma$ by at least three times is found and extracted using a spike detection algorithm (14).

This algorithm also removes background drifts and returns the average and maximum shifts ( $\overline{\Delta\lambda}$ and $\Delta\lambda_{\max}$ , respectively) and the spike duration ( $\Delta\tau$ ). In line with the proofs of the single analyte nature demonstrated in our previous studies (13-15), we have found that the detected spikes from the sm-DNA/Pol interactions originate from a Poisson process, and their detection rates scale linearly with the analyte concentrations (section S2).

However, we would like to note that the presented single-molecule proofs do not indicate that only one receptor molecule (the immobilized reactant) was monitored overall because the spikes can originate from an ensemble of receptor molecules immobilized on the NRs. Nonetheless, the statistical proof confirms that each individual spike originates from a single receptor interacting with a single analyte molecule, whereas previous or later spikes may originate from a different receptor.

To further elaborate on the sensor's response, we have performed finite element simulations to obtain the near-field intensity $I$ of the electric field in the proximity of NRs. For this, we used simplified geometry for the polymerase consisting of two moving arms and a stationary bottom with the size parameters obtained via x-ray crystallography (19). Specifically, we compare the changes in $I$ that are associated with the movement of the polymerase (Fig. 2, A and D) as a correspondence to the immo-DNA scheme.

For the immo-Pol scheme, the changes in $I$ depending on the variations of angular spread between the thumb and the finger domain of the polymerase were compared at two different immobilization positions (Fig. 2, B, C, and E). The near-field intensity exhibits a highly inhomogeneous distribution within the volume of the polymerase and rapidly decays with increasing distance from the NR's tip. Consequently, $I$ decreases significantly as the gap $\Delta d$ between the

![](dt=2026-04-08/ht=14/1a2eae27fcd7d728adf69128b48f342d6a7b8a3466f1b083770c321b1fe3d9aa.jpg)

![](dt=2026-04-08/ht=14/6431e89eb82506e58d9d7dfe986e8cf9a3cab238ca5a33fe7dee196d26585ba1.jpg)

![](image)
a/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-04-08/ht=14//9accfa8320ef8e4e296cb3ab485d99e5d7b0282dcb22a166285d2f8825551f56.jpg)

![](dt=2026-04-08/ht=14/34b68583b5bbdaf2431bbd985778f552dfdf3d68a92f21c003e1a7f4ba08261b.jpg)

![](dt=2026-04-08/ht=14/6ae8ac4f0c7eb0842e59d77cc1bbd8df4f80c28f827e9e7891347fbe818c6a8c.jpg)

![](dt=2026-04-08/ht=14/7a3955c251d3caf2f4399a29461d8b08bec78ef86d42f1f2f0fb3930b82dadbb.jpg)

NR and the polymerase increases on the scale of a few nanometers (Fig. 2D). Furthermore, $I$ generally increases as the angle $\theta$ between the two arms of the polymerase increases, whereas its absolute value and $\theta$ dependency largely vary for different immobilization locations (Fig. 2E). In addition, we observed that for a different bound position of the polymerase, an increment of the angle $\theta$ could also lead to a decrease in $I$ . This indicates that the WGM shift induced by the same conformational change of the polymerase (that is, the same $\theta$ ) can exhibit a different magnitude and a different sign of the shift. However, in the experiments, only the spikes toward longer wavelengths were observed (Fig. 1D).

# Interaction kinetics of different DNA polymerase species

By using the immo-DNA approach (Fig. 1C, top), we experimentally inspect the interaction kinetics of two different polymerase species, Taq and KF, at a temperature $(T)$ of $\approx 293\mathrm{K}$ . Both Taq and KF are

SCIENCE ADVANCES | RESEARCH ARTICLE

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

3 of 8

expected to show apparent differences in their kinetic behavior because their enzymatic activity is optimal at distinctively different temperatures ( $T_{\mathrm{opt}}$ ) of 348 to $353\mathrm{K}$ and $310\mathrm{K}$ , respectively. Representative resonance wavelength traces for both species are shown in Fig. 3A.

In the case of Taq, we do not observe any change in noise level associated with either the presence or the absence of Taq and ptDNA. However, as for KF, the noise level rises once KF is added and increases even further after DNA is immobilized on the NRs. The former noise increase may be attributed to the unspecific short and reversible attachment of KF to the NRs, whereas the latter might originate from KF interacting with ptDNA molecules bound to locations on the NRs with low field intensities.

Distinct spikes are observed only if ptDNA is immobilized on the NRs and if Taq or KF is present in solution, thus confirming the specificity of the monitored sm-DNA/ Pol interactions (section S3). Furthermore, the expected difference in the kinetic behavior of Taq and KF is directly evident by comparing the spike magnitude and duration distributions obtained for both species (Fig. 3, B and C). The spike amplitudes found for Taq/ptDNA/dNTP interactions exhibit an exponentially decaying distribution (Fig. 3B, top), with $53\%$ of the events populating the first bin above the $3\sigma$ limit.

In contrast, the spike amplitudes obtained for KF/ptDNA/dNTP interactions have a significantly broader distribution exhibiting a clear peak at $\overline{\Delta\lambda}_{\mathrm{c}}$ well in excess of $3\sigma$ (Fig. 3B, bottom). This stark difference in the distributions is, at first, surprising because one would expect higher spike amplitudes for Taq polymerase because of its larger molecular mass (98 kDa) compared to KF (68 kDa). However, the spike magnitude is the result of a temporal averaging process (Eqs.

2 and 3) and, therefore, should be seen in conjunction with the spike duration because events shorter than $\tau_{\mathrm{m}}$ are recognized with a reduced shift magnitude. We find the spike durations $\Delta \tau$ associated

with Taq/ptDNA interactions (Fig. 3C, left) to be significantly shorter than those originating from KF/ptDNA interactions (Fig. 3C, right). Correspondingly, both species also yield distinctively different off-rates $(k_{\mathrm{off}})$ of 47 and $23~\mathrm{s}^{-1}$ [extracted via fitting of $N(\Delta \tau)\sim e^{-k_{\mathrm{off}}\Delta \tau}$ to the respective distributions].

The fact that the experiments were performed at $293\mathrm{K}$ , a temperature rather close to the optimal temperature for KF but well below that for Taq, indicates that there is a correlation between our sensor's signal and the activity of the monitored enzyme. However, it is worthwhile to note that the off-rates, which are extracted from the $\Delta \tau$ distributions, require careful interpretation because they are not necessarily equivalent to the dissociation rates between ptDNA and the polymerase.

They rather reflect how long the polymerase resides within the plasmonic hotspots (that is, a period for which the value of $\bar{I}$ is high enough to be recognized) while it interacts with a DNA strand. This means that $\bar{I}$ can drop below the recognition threshold before the actual DNA/Pol interaction has ended because the polymerase moves along the overhanging template strand and away from the NR's surface.

In addition, the polymerase will start its activity at the end of the primer strand, which is $\approx 8\mathrm{nm}$ away from the NR's surface, thus not reaching the maximum possible field overlap (compare Fig. 2D). Consequently, our values for $k_{\mathrm{off}}$ exceed the dissociation rates reported in other studies (2, 20, 21) by at least one order of magnitude because we recognize only a small fraction of the actual reaction process along the entire template strand.

Thus, it is evident that by using the immo-DNA approach, the amount of information that is directly obtainable is limited, although it still allowed us to recognize differences in the interaction kinetics of KF associated with ssDNA and ptDNA/dNTP interactions (section S5). Hence, we use the approach of immobilizing the polymerase on the NRs (the immo-Pol scheme) in the following studies of sm-DNA/Pol interactions.

![](dt=2026-04-08/ht=14/9b19c578541ce5457eca091c00e2b591278e842400e3dd619552a69daf91684b.jpg)

SCIENCE ADVANCES | RESEARCH ARTICLE

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

4 of 8

# Conformational transitions of Pfu polymerase at various temperatures

The above results suggest a possible correlation between our sensor signal (namely, the spike amplitudes and durations) and the activity of the monitored enzymes. Nonetheless, it is challenging to precisely determine the physical process associated with the signals because conformational changes and the motion of the whole enzyme cannot be directly separated from the immo-DNA scheme. To exclude the polymerase motion's contribution to the signals, we have performed experiments with polymerase immobilized on the NRs (immo-Pol scheme as shown in Fig. 2B).

For this, we selectively use Pfu polymerase because it maintains its enzymatic activity even when immobilized on the NR surface. Experimental data confirming its activity and successful immobilization on the NRs are provided in sections S4 and S6.

While monitoring the sm-DNA/Pol interactions in this scheme, we recognize spike events similar to the ones observed using the previously discussed immo-DNA scheme. Furthermore, the rates at which these spikes occur exhibit a linear dependence on the ptDNA's concentration (fig. S2A), whereas no spikes were recognized in the absence of ptDNA (fig. S3A, bottom) and when the NRs were not modified with Pfu polymerase, thus confirming that the spikes originate from individual ptDNA/Pol interactions.

To further elaborate on the mec
hanism behind these spikes, we also monitored DNA/Pol interactions for ptDNA strands with different lengths (section S7), which yielded no significant difference in the observed spike amplitudes despite a 3.3-fold increase in the ptDNA's molecular mass. On the basis of this finding and together with the fact that the immobilized polymerase has a significantly larger volume (as compared to the ptDNA) and, in turn, occupies a larger fraction of the sensing volume (Fig. 2), we believe that the spike signals cannot solely originate from ptDNA molecules directly perturbing the field close to

the NRs but may rather be induced by conformational changes of the ptDNA/Pfu complex accompanied by the initial incorporation of ptDNA. However, we would like to note that relating the patterns of individual spikes with transitions between specific conformational states is difficult because, in the current sensing scheme, even a single point inside one spike is the result of an averaging process over a period $\tau_{\mathrm{m}}$ (Eq. 2), which may include multiple transitions between conformational substates.

The relation between the recognized spikes and the conformational changes of the Pfu polymerase is further supported by their temperature dependence. The conformational changes of the thermophilic Pfu polymerase are closely linked to its enzymatic activity, which, in turn, is dependent on the ambient temperature and reaches its maximum in the 345 to $348\mathrm{K}$ range. Thus, we study how the durations and magnitudes of the ptDNA triggered spike signals change as we stepwise increase the ambient temperature toward this range while dNTPs are present in solution (Fig. 4).

With respect to the spike amplitude distributions, we find that with increasing temperature, the center of the peak position $\overline{\Delta\lambda_{\mathrm{c}}}$ shifts toward higher amplitudes while their broadness also increases (Fig. 4A). On the one hand, this result is in line with our previous finding from the comparison of KF and Taq polymerase (Fig. 3) that the more active polymerase induces higher spike amplitudes (Fig. 4B).

On the other hand, it further supports the premise that the recognized signals are not influenced by the ptDNA strands in terms of the direct addition of molecular mass to the Pfu molecules through binding. Furthermore, the fact that the peak center consistently shifts toward higher amplitudes as the temperature is adjusted toward values that favor the polymerase's activity strongly suggests a relation between the spike amplitudes and conformational changes associated with the ptDNA/Pol interactions. These results specifically indicate that upon the incorporation of ptDNA,

![](dt=2026-04-08/ht=14/16f2b07841ab693b113f0ca2d0fd3367da2499f1e2b491aadb2e7e527e68394e.jpg)

![](dt=2026-04-08/ht=14/4581746ee94395d72f81d7a732995bc91efe10f29a0f5f3ea7dd8cbe2605138c.jpg)

SCIENCE ADVANCES | RESEARCH ARTICLE

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

5 of 8

the ptDNA/Pfu complex additionally undergoes a conformational change in such a manner that its time-averaged field overlap integral $\bar{I}$ increases as compared to the moment of the initial ptDNA/Pfu interaction. Toward higher temperatures (where the enzyme has higher activity), this conformational transition that the ptDNA/Pfu complex undergoes evolves in such a way that the $\overline{\Delta I}$ increases, which can be interpreted as an increase in the (time-averaged) magnitude of the conformational change.

Furthermore, the spike durations observed for the Pfu/ptDNA/dNTP interactions also become shorter as the temperature and, thus, the expected activity of the polymerase increase (Fig. 4C). This is in line with the expectation that the polymerase processes the ptDNA faster and consequently releases it earlier under more favorable ambient conditions.

We next compare the temperature-dependent Pfu/DNA interactions in the absence and presence of dNTPs (see Fig. 5). Independent of the presence of dNTPs, the peak centers of the spike amplitude distributions shift again toward higher amplitudes with increasing temperature, whereas the shifts are larger if dNTPs are present (Fig. 5A). Furthermore, we find that $\overline{\Delta\lambda}_{\mathrm{c}}\propto e^{A}$ (Fig. 4B), where the values for $A$ as determined from the corresponding Arrhenius plots (Fig.

5B) yield $-6.5\times 10^{3}\pm 0.9\times 10^{3}\mathrm{K}$ $(-1.7\times 10^{3}\pm 0.7\times 10^{3}\mathrm{K})$ in the presence (absence) of dNTPs. The precise numerical values for $\overline{\Delta\lambda}_{\mathrm{c}}$ are listed in table S1. Here, the overall higher spike amplitudes that were found in

the presence of dNTPs might be associated with the polymerase undergoing changes between additional conformational substates accompanying the extension of the primer strand by possibly multiple dNTPs (the sensor's time resolution is too low to resolve the incorporation of single dNTPs), yielding an increased $\overline{\Delta I}$ . The temperature dependence of the $k_{\mathrm{off}}$ values as extracted from spike duration ( $\Delta \tau$ ) distributions, however, displays a significant difference with respect to the presence of dNTPs.

Although we find that $k_{\mathrm{off}} \propto e^{\frac{B}{T}}$ for both cases, the sign of the values for $B$ as extracted from the corresponding Arrhenius plots (Fig. 5D) is different and $B$ values yield $-3.8 \times 10^{3} \pm 0.6 \times 10^{3} \mathrm{~K}$ ( $2 \times 10^{3} \pm 1 \times 10^{3} \mathrm{~K}$ ) in the presence (absence) of dNTPs. The result obtained in the presence of dNTPs is in line with what we had found before, namely, the increasing speed of the enzymatic process. However, in the absence of dNTPs, the spike durations increase with increasing temperature (Fig. 5C).

This indicates that after the ptDNA/Pfu complex is formed, it remains in a certain conformation for a duration that increases as the temperature increases, until the ptDNA is eventually released again. This, in turn, implies that the ptDNA/Pfu complex is more stable at higher temperatures. From an enzymatic point of view, such a property is certainly desirable because, in this state, the polymerase is waiting for the arrival of dNTPs and a premature release of the ptDNA would be a waste of energy.

These results, especially the distinctively different temporal behavior in the presence and absence of dNTPs, provide strong evidence that the

![](dt=2026-04-08/ht=14/f52babe7169e8c385193e0e7137cde96ac06fe0d3aba30614a39024125f36041.jpg)

![](dt=2026-04-08/ht=14/529253c66ef9cd20cfd3662823201ef193401e52d1d8d66757a0501c0e3be068.jpg)

![](dt=2026-04-08/ht=14/aa5df0d5653f2e93e6e66d9fbf3143e770a6e34b9e38c27e3efdab6df5925e7b.jpg)

![](dt=2026-04-08/ht=14/0912c11cb70fae7b8f4493aebbf387abf1139aef84afc59c08cc2796655cf141.jpg)

![](dt=2026-04-08/ht=14/82806e9003eb23d489bd6459861d2d39582286e8a14c22cbbee72de4cb4558b1.jpg)

SCIENCE ADVANCES | RESEARCH ARTICLE

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

6 of 8

monitored conformational changes accompanied by the Pfu/DNA complex formation involved the incorporation of nucleotides.

# CONCLUSIONS

We have demonstrated our sensor's capability to detect polymerase/DNA interactions linked to enzymatic act
ivity on a single-molecule level by monitoring the kinetics and conformational transitions of DNA polymerases. Our results exhibit a clear correlation between our sensor's signal and the enzymatic activity of three different types of polymerases at various ambient temperatures. Moreover, we have shown that the magnitude and duration of the conformational changes of the polymerase associated with DNA interactions vary with respect to the presence and absence of dNTPs.

In this context, we have found a distinct difference in the temperature dependence of the enzymes' kinetic behavior, providing strong evidence for the correlation of our sensor's signal with enzymatic activity in the form of nucleotide incorporation.

Our results are potentially significant because monitoring enzymatic activity in a multiplexed and high-throughput fashion is a crucial requirement for next-generation sequencing. The fact that our approach is label-free and the sensor's signals can be monitored in real time opens a new and direct way to determine conformational substates of various proteins without the necessity to mitigate label-associated background fluctuations.

However, our sensor is intrinsically limited by our signal amplification method because the plasmonically enhanced near field may only probe a fraction of the enzyme's volume. This limitation might become an advantage because it may, in turn, allow for the selective probing of certain protein subdomains. Furthermore, enzymatic kinetics can be easily tested with respect to diverse medium conditions, such as molecule concentrations, ionic strength, pH, and temperature.

In this context, combining our method with label-based techniques would be promising when linked to the extensive knowledge already established via label-based methods, thus diversifying quantitative analysis. For example, in combination with FRET, our sensor would allow for probing of conformational changes beyond FRET's spatial limitations yet take advantage of its selectivity.

Moreover, the temperature dependence of polymerase/DNA interactions can be further investigated by optically driven local heating of the NRs, which may allow for fast switching of the enzymes' activity and the corresponding interaction kinetics.

# METHODS

All solutions without NRs were filtered with $0.1\mathrm{-}\mu \mathrm{m}$ syringe filters (Merck Millipore) before usage. The power that was coupled to microspheres was $<  0.1~\mathrm{mW}$ overall, accounting for an average total energy of $<  20~\mathrm{nJ}$ being coupled to the resonators per wavelength sweep.

# The immo-DNA recognition scheme

Sensor assembly was conducted by adopting a modified version of the three-step wet-chemical procedure used in the study by Baaske et al. (13). First, cetyltrimethylammonium bromide-stabilized gold NRs with diameters of $10\mathrm{nm}$ and lengths of $35\mathrm{nm}$ (Nanopartz) were immobilized on the microresonator surface. For this, the NRs were injected into a sample cell holding about $0.5\mathrm{ml}$ of $100\mathrm{mMNaCl}$ solution at $\mathrm{pH}\approx$ 1.6. This process was directly monitored, and individual NR binding events were classified as discrete steps in the resonance position and linewidth traces. Second, thiol-modified ssDNA ([ThiC6]- $5^{\prime}$ -TTTCTCGTTGGGGTCTTTGCTC, Eurofins) were conjugated to the

adsorbed NRs. To cleave any disulfide bonds, we treated the thiolated DNA with $100\mathrm{mM}$ dithiothreitol and $100\mathrm{mMNaCl}$ for $30\mathrm{min}$ at room temperature before measurement (22). The NR-modified sphere was then immersed in a solution with $500\mathrm{mMNaCl}$ , $0.02\%$ (w/w) SDS at $\mathsf{pH}\approx 3$ and $1\mu \mathrm{M}$ DNA (13, 23).

The time spans required to produce a surface coverage that was sufficient to monitor sm-DNA/Pol interactions were in the range of 10 to $30\mathrm{min}$ , although longer reaction times may result in undesirably high DNA surface densities and hinder protein-DNA interactions. Last, in NEBuffer 2 (New England BioLabs Inc.), the initial $\mathsf{pH}$ of the solution was reduced to 6.7 by adding $2\mathrm{mM}$ HCl to maintain stable NR adsorption.

If required, ssDNA was hybridized to ptDNA by injection of the template strand DNA (60 nt, CCGACAACCACTACACCGGTCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGA), followed by the introduction of the desired amount of polymerase (that is, Taq and KF; New England BioLabs Inc.).

# The immo-Pol recognition scheme

Sensor assembly was performed according to the following steps. First, polymerase-gold NR conjugates were prepared by mixing $2\mu \mathrm{l}$ of Pfu DNA polymerase (BioVision) from a stock solution $(2.5\mathrm{U} / \mu \mathrm{l})$ with $5\mu \mathrm{l}$ of a solution containing citrate gold NRs (diameter, $25\mathrm{nm}$ ; length, $49\mathrm{nm}$ ; Nanopartz) at a concentration of $5.7\times 10^{11}$ nanoparticles/ $\mu \mathrm{l}$ .

Second, the surface of the freshly fabricated microsphere was functionalized with aminopropyltriethoxysilane $(\mathrm{C_9H_{23}NO_3Si}$ ; APTES) by immersing the microsphere in a $100 - \mu \mathrm{l}$ droplet of $2.5\%$ (v/v) APTES for about 1 to $2\mathrm{min}$ . The binding of Pfu-NR conjugates to the microsphere was then performed in a sample chamber filled with PCR buffer. The PCR buffer contained $10\mathrm{mM}$ tris-HCl, $50\mathrm{mM}$ KCl, $1.5\mathrm{mM}$ $\mathrm{MgCl}_2$ , and $0.001\%$ (w/v) gelatin (Sigma-Aldrich).

Last, the ptDNA used for the measurements was prepared by mixing the template and primer strands (1:1 molar ratio) in a solution containing $50\mathrm{mMNaCl}$ . This mixture was heated to $94^{\circ}\mathrm{C}$ in an Eppendorf incubator and cooled down to room temperature before use. The observation of ptDNA and immobilized Pfu polymerase was then undertaken in the PCR buffer.

# Numerical analysis

Numerical simulations were performed using COMSOL Multiphysics (frequency-domain module). The polymerase was modeled as three rectangular parallelepipeds $(6\mathrm{nm}\times 4\mathrm{nm}\times 5\mathrm{nm}$ for the moving arms and $2\mathrm{nm}\times 5\mathrm{nm}\times 5\mathrm{nm}$ for the stationary bottom) that are conjugated with two cylinders $(2\mathrm{nm}\times 5\mathrm{nm})$ to maintain a constant volume while increasing the angle between the two arms.

It was initially attached to the center of the gold NR's tip, and this gold NR was modeled as a prolate circular cylinder with hemispherical end caps with dimensions of $25\mathrm{nm}\times 49\mathrm{nm}$ . The refractive index of the surrounding medium (water) used for the simulation was 1.332, and the frequency-dependent refractive index values of gold were taken from the study by Johnson and Christy (24). The simulation domain was surrounded by a perfectly matched layer (PML) to absorb the outward-propagating radiation.

To calculate the local field enhancement, we illuminated the entire domain with a plane wave at a pump wavelength of $642\mathrm{nm}$ . The polarization state of the incoming field was parallel to the long axis of the NR. With regard to meshing, a swept mesh was used to discretize the external and PML domain. Note that the NR and polymerase domains were meshed using free tetrahedral discretization, whereas the finer meshes were used in the NR and polymerase domains.

SCIENCE ADVANCES | RESEARCH ARTICLE

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

7 of 8

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

# SUPPLEMENTARY MATERIALS

Acknowledgments: We thank C. Koch from the Biochemistry chair of Friedrich-Alexander University Erlangen-Nürnberg for providing access to the PCR instrument. E.K. and M.D.B. thank S. Vincent for his feedback on the manuscript. Funding: We acknowledge financial support from the Max Planck Society. Author contributions: E.K. and M.D.B. wrote the manuscript and performed the simulation and developed the experimental setup. M.D.B. developed data analysis tools and supervis
ed the experiment. P.S.W., I.S., and E.K. performed the experiment. F.V. supervised the entire project.

All authors commented on the manuscript. Competing interests: The authors declare that they have no competing interests. Data and materials availability: All data needed to evaluate the conclusions in the paper are present in the paper and/or the Supplementary Materials. Additional data related to this paper may be requested from the authors.

Submitted 5 December 2016

Accepted 9 March 2017

Published 29 March 2017

10.1126/sciadv.1603044

Citation: E. Kim, M. D. Baaske, I. Schuldes, P. S. Wilsch, F. Vollmer, Label-free optical detection of single enzyme-reactant reactions and associated conformational changes. Sci. Adv. 3, e1603044 (2017).

SCIENCE ADVANCES | RESEARCH ARTICLE

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

Kim et al., Sci. Adv. 2017;3:e1603044 29 March 2017

8 of 8

This article is publisher under a Creative Commons license. The specific license under which this article is published is noted on the first page.

For articles published under CC BY licenses, you may freely distribute, adapt, or reuse the article, including for commercial purposes, provided you give proper attribution.

For articles published under CC BY-NC licenses, you may distribute, adapt, or reuse the article for non-commercial purposes. Commercial use requires prior permission from the American Association for the Advancement of Science (AAAS). You may request permission by clicking here.

The following resources related to this article are available online at http://advances.sciencemag.org. (This information is current as of March 29, 2017):

Updated information and services, including high-resolution figures, can be found in the online version of this article at: http://advances.sciencemag.org/content/3/3/e1603044.full

Supporting Online Material can be found at: http://advances.sciencemag.org/content/suppl/2017/03/29/3.3.e1603044.DC1

This article cites 27 articles, 3 of which you can access for free at: http://advances.sciencemag.org/content/3/3/e1603044#BIBL

Science Advances

Label-free optical detection of single enzyme-reactant reactions and associated conformational changes  
Eugene Kim, Martin D. Baaske, Isabel Schuldes, Peter S. Wilsch and Frank Vollmer (March 29, 2017)  
Sci Adv 2017, 3:  
doi: 10.1126/sciadv.1603044

Downloaded from http://advances.sciencemag.org/ on March 29, 2017

Science Advances (ISSN 2375-2548) publishes new articles weekly. The journal is published by the American Association for the Advancement of Science (AAAS), 1200 New York Avenue NW, Washington, DC 20005. Copyright is held by the Authors unless stated otherwise. AAAS is the exclusive licensee. The title Science Advances is a registered trademark of AAAS