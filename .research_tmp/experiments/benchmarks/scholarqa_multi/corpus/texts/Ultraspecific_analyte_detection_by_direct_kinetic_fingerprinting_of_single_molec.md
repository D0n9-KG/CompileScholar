# Article

# Single-Molecule Kinetic Fingerprinting for the Ultrasensitive Detection of Small Molecules with Aptasensors

Rui Weng, Shengting Lou, Lidan Li, Yi Zhang, Jing Qiu, Xin Su, Yongzhong Qian, and Nils G Walter

Anal. Chem., Just Accepted Manuscript • DOI: 10.1021/acs.analchem.8b04145 • Publication Date (Web): 18 Dec 2018

Downloaded from http://pubs.acs.org on December 21, 2018

# Just Accepted

"Just Accepted" manuscripts have been peer-reviewed and accepted for publication. They are posted online prior to technical editing, formatting for publication and author proofing. The American Chemical Society provides "Just Accepted" as a service to the research community to expedite the dissemination of scientific material as soon as possible after acceptance. "Just Accepted" manuscripts appear in full in PDF format accompanied by an HTML abstract. "Just Accepted" manuscripts have been fully peer reviewed, but should not be considered the official version of record.

They are citable by the Digital Object Identifier (DOI®). "Just Accepted" is an optional service offered to authors. Therefore, the "Just Accepted" Web site may not include all articles that will be published in the journal. After a manuscript is technically edited and formatted, it will be removed from the "Just Accepted" Web site and published as an ASAP article. Note that technical editing may introduce minor changes to the manuscript text and/or graphics which could affect content, and all legal disclaimers and ethical guidelines that apply to the journal pertain.

ACS cannot be held responsible for errors or consequences arising from the use of information contained in these "Just Accepted" manuscripts.

analytical chemistry

Subscriber access provided by YORK UNIV

A C S

ACS Publications

is published by the American Chemical Society. 1155 Sixteenth Street N.W., Washington, DC 20036

Published by American Chemical Society. Copyright © American Chemical Society. However, no copyright claim is made to original U.S. Government works, or works

produced by employees of any Commonwealth realm Crown government in the course of their duties.

# Single-Molecule Kinetic Fingerprinting for the Ultrasensitive Detection of Small Molecules with Aptasensors

Rui Weng, $^{\dagger, \#}$ Shengting Lou, $^{\ddagger, \#}$ Lidan Li, $^{\ddagger}$ Yi Zhang, $^{\ddagger}$ Jing Qiu, $^{\dagger}$ Xin Su, $^{\ddagger, *}$ Yongzhong Qian, $^{\dagger, *}$ and Nils G. Walter $^{8, *}$

ABSTRACT: Aptamers have emerged as promising molecular tools for small-molecule analyte sensing. However, the performance of such aptasensors is generally limited by leakage since it has been difficult to completely suppress signal in the absence of analyte, resulting in a compromise between sensitivity and specificity. Here, we describe a methodology for the ultrasensitive detection of analytes combining aptasensors with single-molecule kinetic fingerprinting.

A short, fluorescently labeled DNA probe is utilized to detect the structural changes upon ligand binding to the designed hairpin-shaped aptasensor probe. The Poisson statistics of binding and dissociation events of the DNA probe to single surface-immobilized aptasensor molecules is monitored by total internal reflection fluorescence microscopy, permitting the high-accuracy discrimination of the ligand bound and ligand-free states, resulting in zero background.

The programmable dynamics of the hairpin enables fine-tuning of the hybridization kinetics of the fluorescent probe, rendering the acquisition time sufficiently flexible to optimize discrimination. Remarkable detection limits are achieved for a diverse set of analytes when spiked into chicken meat extract: the nucleotide adenosine (0.3 pM), the insecticide acetamiprid (0.35 pM), and the dioxin-like toxin PCB-77 (0.72 pM), which is superior to recently reported aptasensors.

Our generalizable method significantly improves the performance of aptasensors, with the potential to extend to other molecular biomarkers.

# INTRODUCTION

Aptamers are synthetic single-stranded oligonucleotides selected in vitro by systematic evolution of ligands using the exponential enrichment (SELEX) that recognize a variety of ligands including metal ions, small molecules, proteins, and even entire cells. An aptasensor is a particular class of biosensor in which the recognition capability is based on a DNA or RNA aptamer coupled to a molecular sensor domain.

With the growing need for rapid, low-cost, and high-confidence methods for environmental analysis and clinical diagnosis, aptasensors have been increasingly used due to their robustness, small size, and high binding specificity. The design strategies of an aptasensor can be mainly categorized into three classes: conformational change mode, split-combination mode, and competitive mode. In the conformational change mode, binding of an analyte causes a conformational change in the aptamer that can be monitored using, e.g., quenching or fluorescence resonance energy transfer (FRET) (Figure 1A).

In the split-combination mode, the presence of an analyte leads to the binding of two aptamer half-motifs, resulting in an optical or electrochemical signal change (Figure 1B). In the competitive mode, the binding of an analyte releases the aptamer from an auxiliary strand that is fully or partially complementary to the aptamer (Figure 1C). For practical reasons, the competitive mode is most common because a ligand-induced conformational change in some aptamers is small, and many ligands are not conducive to being split. Regardless, a common problem in all of these systems is that

sensing is typically "leaky", that is, it is difficult to completely suppress unwanted signal in the absence of analyte. For example, in Figure 1D the background signal from a competitive mode strategy is found not to be negligible, particularly when the auxiliary probe is short. Conversely, increasing the length of the auxiliary probe leads to poorer sensitivity. These observations illustrate a trade-off between sensitivity and specificity.

To date, much effort has been invested in improving the performance of aptasensors. A variety of nanomaterials such as gold nanoparticles upconversion nanoparticles, and graphene have been employed as alternatives to the auxiliary strand. Moreover, downstream amplification strategies such as hybridization chain reaction (HCR) and exponential amplification reaction (EXPAR) have been employed to enhance sensitivity. The implementation of these approaches has extended the detection limit to nanomolar, and even picomolar, levels.

However, severe background caused by spurious amplification and/or the nonspecific interaction of nanomaterials and analytes inevitably leads to false positives and a fundamental lower limit of detection. These limitations create challenges with regard to developing a robust aptasensor for practical use. Developing strategies to achieve an ultimate detection limit where the signal-to-noise ratio does not decrease with decreasing analyte concentration is expected to significantly improve aptamer-based detection.

High sensitivity and specificity can be achieved simultaneously if the ligand bound and ligand-free states can be discriminated with

Page 1 of 9

Analytical Chemistry

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

arbitrarily high accuracy. Single-molecule measurements have emerged as an ultimate-sensitivity toolkit to explore systems from small molecules to living cells. $^{12}$ These toolkits can reveal the subpopulations and dynamics of an inhomogeneous biochemical system that are otherwise hidden in ensemble me
asurements. $^{13}$ Total internal reflection fluorescence microscopy (TIRFM) emerges as a powerful platform for single-molecule detection with both high throughput and high signal-to-background ratio.

Aptasensors can be readily modified to be observable at the single-molecule level using TIRFM. For example, Landry et al. developed an aptasensor for detecting protein efflux from microorganisms by using the single-molecule fluorescence of single-walled carbon nanotubes. $^{14}$ We and others used single-molecule FRET (smFRET) to study the conformational changes in aptamer upon ligand binding. $^{15-17}$

![](dt=2026-05-04/ht=12/d937dbe67c25bb3ff0e455c09398ff971b6e4ec9bf5aa92feb2fcac0fd8e7997.jpg)

Herein, we demonstrate a single-molecule kinetic fingerprinting analysis via TIRFM for maximizing the sensitivity and specificity of virtually any aptasensor. A short, fluorescently labeled single-stranded DNA (ssDNA) probe, with a sequence complementary to a particular region of interest, is used to probe the structural change in an aptamer containing hairpin probe in a transient and repetitive manner.[17-19] The hybridization kinetics of the short ssDNA probe is highly sensitive to the number of base pairs[20] formed with, and thus the secondary structure of, the nucleic acid target,[17] which was utilized here to probe the structural change in the aptasensor

upon ligand binding. Our method allows for the absolute discrimination between the ligand bound and ligand-free states by discriminating the binding kinetics of the short probe. High sensitivity was achieved because the ligand-free state can be confidently screened out based on its distinct kinetics. We anticipate that this single-molecule kinetic fingerprinting analysis will find broad applications in detecting tightly binding single analyte molecules.

# EXPERIMENTAL SECTION

Materials. Oligonucleotides were synthesized and purified by HPLC (Sangon, Shanghai, China). The sequences of these oligonucleotides are listed in Table S1. The adenosine, PCB-77, acetamiprid, (3-Aminopropyl)triethoxysilane (APTES), 3,4-dihydroxybenzoate (PCA), protocatechuate dioxygenase (PCD), and Trolox were obtained from Sigma-Aldrich (St. Louis, MO). The mPEG-succinimidyl valerate (mPEG-SVA, MW, 5000), biotin-PEG-succinimidyl valerate (biotin-PEG-SVA, MW, 5000), and sulfo-disuccinimidyl tartarate (Sulfo-DST) were obtained from SeeBio Co. (Shanghai, China). All chemicals were used as received without additional purification. DNase/RNase-free deionized water from Tiangen Biotech Co. (Beijing, China) was used in all experiments.

Bulk fluorescence measurement for the competitive mode aptasensors. In a typical assay, in a $1.7\mathrm{-mL}$ centrifuge tube, 500 nM of double-stranded DNA and $200~\mathrm{nM}$ of adenosine were incubated for $30\mathrm{min}$ at room temperature in $1\times$ PBS. The fluorescence (Ex: $530~\mathrm{nm}$ , Em: $590~\mathrm{nm}$ ) was measured on a multilabel reader (EnVision, PerkinElmer, UK).

TIRFM setup and imaging surface preparation. The objective-type TIRFM was set up using a Nikon inverted microscope (ECLIPSE, Ti-U) equipped with a $100\times$ magnification, 1.49 numerical aperture (NA) TIRFM objective (Nikon). For TIRF illumination, a $520~\mathrm{nm}$ laser was coupled into a single-mode fiber (Solamere Technologies). The fiber optic cable that delivers laser light to the microscope was secured into a fiber launch fitted with an XY fiber holder mounted atop a micrometer-driven optical rail for Z adjustment (Thorlabs).

Images were captured using an electron multiplying (EMCCD) camera (Andor). The pixel size of this camera matches very well with the magnification offered by the $100\times$ TIRF objective, giving a final resolution of $0.15\mu \mathrm{m}$ per pixel. For TIRF microscopy measurements, sample cells were constructed by fixing a 1-cm length of a cut pipet tip (Eppendorf) to a coverslip using epoxy adhesive. The single-molecule imaging surface was coated with a 10:1 mixture of mPEG and biotin-PEG.[18]

Single-molecule kinetics analysis for target detection. All solutions were prepared in $1.7\mathrm{-ml}$ microcentrifuge tubes. The slide surface was briefly incubated with TE buffer ( $10~\mathrm{mM}$ Tris-HCl, 1 mM EDTA, $\mathrm{pH}8.0$ ) followed by $1\mathrm{mg / ml}$ streptavidin for $10\mathrm{min}$ Then, excess streptavidin was flushed with TE buffer. Next, $50~\mathrm{pM}$ of the biotinylated hairpin probe was annealed from $90^{\circ}\mathrm{C}$ to r.t.

in $1\times$ PBS buffer and added to the reaction cell for $10\mathrm{min}$ and the excess was flushed with $1\times$ PBS three times. The $1\times$ PBS buffer containing the target was introduced into the sample cell and incubated for $1\textrm{h}$ .The $1\times$ PBS buffer containing an oxygen scavenger system21 (OSS, $2.5\mathrm{mM}$ PCA, $25\mathrm{nM}$ PCD, $1\mathrm{mM}$ Trolox) and $20~\mathrm{nM}$ of the Cy3-labeled fluorescent probe was added to the sample cell.

The transient binding of the fluorescent probes to the hairpin probes was monitored under illumination by the $520~\mathrm{nm}$ laser light. Image acquisition was performed using the EMCCD camera $(100~\mathrm{ms}$ gain 20). It should be noted that for the singlemolecule assays, the room temperature was controlled at $25\pm 1^{\circ}\mathrm{C}$ Fluorescence time trajectories were extracted from acquired movies by using a custom MATLAB code. The trajectories were

Analytical Chemistry

Page 2 of 9

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

fitted with a hidden Markov model (HMM) using vbFRET 22 to identify the number of transitions and dwell times of the bound and unbound states for an individual molecule.

Characterization of the folding/unfolding kinetics of the hairpin probes. The dual-labeled hairpin probes were used for kinetics characterization. A total of $100~\mathrm{pM}$ of the biotinylated hairpin probe was annealed from $90^{\circ}\mathrm{C}$ to r.t. in $1\times$ PBS buffer and was added to the reaction cell for $10\mathrm{min}$ . The excess was flushed with $1\times$ PBS three times. The $1\times$ PBS buffer containing OSS was added. The dynamics were monitored under $520\mathrm{-nm}$ last light illumination. Image acquisition was performed using the EMCCD camera (10 ms, gain 80).

Detection of acetamiprid- and PCB-77-spiked chicken meat extracts. Pesticide-free chicken samples were homogenized and incubated with acetonitrile followed by centrifugation for $10\mathrm{min}$ at $8000~\mathrm{rpm}$ $(4^{\circ}\mathrm{C})$ . The supernatants were collected and stored at $-20^{\circ}\mathrm{C}$ for future use. The freshly thawed extract was diluted 100-fold with $1\times$ PBS containing either synthetic acetamiprid or PCB-77. The spike-in samples were allowed to bind to the detection surface for $1\mathrm{h}$ .

The excess solution was removed, and the surface was washed with $1\times$ PBS three times. Next, a single-molecule analysis was carried out for these samples, as described above. The measured concentration of the targets was calculated using the standard curve collected in buffer.

# RESULTS AND DISCUSSION

Principle of the aptasensor with single-molecule kinetics fingerprinting. As illustrated in Figure 2A, the aptasensor consists of two probes, a hairpin probe immobilized on the slide surface through the interaction of streptavidin and biotin, and a Cy3 labeled short ssDNA probe. The fluorescent probe is complementary to a particular region of the hairpin probe (green). The hairpin probe also contains the 27-nt aptamer

sequ
ence (orange) for adenosine (purple), which has been extensively utilized for adenosine detection.[23-26] The aptamer sequence is appended with the stem of the hairpin probe, which forms in the absence of analyte. All such DNA hairpins fluctuate in solution between a folded (closed) and a denatured (open) state, where only the latter allows for the binding of the fluorescent probe.

Due to the excitation geometry of TIRFM, only fluorescent molecules entering the evanescent field (extending $\sim 100$ nm above the surface) are illuminated, and only when they bind to the complementary hairpin sequence will they stay in place long enough for a signal to be detected using our 100-ms camera integration time. Accordingly, the stochastic switching of fluorescence signals between the bound (ON-) and unbound (OFF-)states is observed in TIRFM mode, which can be attributed to the binding and dissociation of the fluorescent probe, respectively (Figures 2B and C).

These single-molecule fluorescence-time trajectories in the presence or absence of adenosine ligand (Figures 2B and C) are distinct from those arising from nonspecific surface binding of probes in the absence of aptasensor, which most likely reflects photobleaching of the fluorescent probe (Figure S2). These results suggest that the fluorescent probe is not completely occluded from binding the hairpin probe in the absence of ligand (Figure 2B). Notably, when aptasensors are observed in bulk, such background in the absence of ligand is typically referred to as sensor leakage.

Upon the addition of adenosine, the binding of the ligand to its aptamer results in hairpin unfolding, enhancing the binding of the fluorescent probe (Figure 2C) as expected when the fluorescent probe binds increasingly to its complementary strand (Figure S3). Accordingly, in this system the change in hairpin accessibility directly reflects the binding of a single analyte molecule, provided that ligand dissociation is sufficiently slow.

![](dt=2026-05-04/ht=12/8054df21d8fdf32cef142f80235cada9b8cf986623ff111ca16a2c30db0248e1.jpg)

![](dt=2026-05-04/ht=12/1903656c0cce2912008185043368b5f4121eb36df08e81b7f95da598601381a9.jpg)

![](dt=2026-05-04/ht=12/ae421b6339779471413835b9bf4519038aced5bfd1839475bbf65af092ec09b0.jpg)

![](dt=2026-05-04/ht=12/6b5b14a4a44b9d48171edff67698d730b7394c0e1c79bb1302df677f4ba09328.jpg)

![](dt=2026-05-04/ht=12/e524d60af68408595b36ea6247eb601dcc8b981cf7f10066a5e283db4070e1a2.jpg)

![](dt=2026-05-04/ht=12/5bca06bd44d99bf03197bcfbfa8e1e3e84363c2c0fc5623e64184d308430c653.jpg)

Page 3 of 9

Analytical Chemistry

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

probe with/without adenosine. All of the distributions were from more than 300 molecules. The concentrations of the hairpin probe and the fluorescent probe are $50~\mathrm{pM}$ and $20~\mathrm{nM}$ , respectively. All single-molecule assays were carried out at room temperature.

The binding equilibrium of a bimolecular complex can usually be approximated as a two-state system whose kinetics are characterized by a bimolecular association rate constant. The transient binding of the fluorescent DNA probe to the immobilized hairpin is monitored at the single-molecule level by TIRFM, and the dwell times in the fluorescence-ON $(t_{\mathrm{on}})$ state and fluorescence-OFF $(t_{\mathrm{off}})$ state are both exponentially distributed.

[27,28] Accordingly, the resulting fluorescence-versus time trajectories were fitted with a two-state hidden Markov Model (HMM) to extract mean dwell times. Fitting of single-exponential distributions to the experimental dwell time distributions yields the time constants $\tau_{\mathrm{on}}$ and $\tau_{\mathrm{off}}$ , respectively. $\tau_{\mathrm{on}}$ does not change significantly upon the addition of ligand (Figures 2D and E). In contrast, $\tau_{\mathrm{off}}$ decreases by approximately six-fold (Figures 2F and G).

The hybridization kinetics of the fluorescent probe to the ligand-bound hairpin is in good agreement with that to just the isolated complementary strand (Figure S4). The change in accessibility of the hairpin probe is therefore reflected in the differential $\tau_{\mathrm{off}}$ of the fluorescent probe. However, while a difference in $\tau_{\mathrm{off}}$ was found, it is exponentially distributed that cannot be used for state identification.

Poisson statistics of binding and dissociation events of the fluorescent DNA probe. Because the number of the repetitive hybridization of probes to an immobilized target is characterized by Poisson statistics,[18] the standard deviation in the number of binding and dissociation events $(\mathrm{N}_{\mathrm{b + d}})$ per aptasensor molecule is expected to increase only as the square root of the number, implying that the signal acquisition time can be prolonged to distinguish states with distinct probe binding kinetics to arbitrarily high accuracy.

That is, as long as there is a slight difference in the kinetics of probe binding or dissociation between states, single-molecule kinetics analysis with a Poisson model is theoretically able to resolve them[18]. As expected, $\mathrm{N_{b + d}}$ of our fluorescent probe follows a Poisson process so that the two probability distribution peaks for the ligand bound and ligand-free states are gradually resolved with increasing acquisition time (Figure 3).

Interestingly, no background peak was observed in the presence of ligand implying that all aptasensors are occupied under the concentration of ligand we used. This renders a lower dissociation equilibrium constant than the previously reported.[23,29] One possible reason is that the concentrations of ligand close to the detection surface were underestimated. In addition, the dissociation rate constant of the ligand is likely slow, not leading to full equilibration.

[30] The diffusion-driven fluorescent probe binding provides resistance against photobleaching as the probe is continuously exchanged with bulk solution. This allows for a flexible, and essentially unlimited, acquisition time to separate the Poisson peaks. For a specific fluorescent probe, different hairpin structures lead to

distinct probe hybridization kinetics, and the hairpin with a higher stability requires a shorter acquisition time (Figure 3A) than that with a lower stability (Figure 3B) for good separation of the bound and ligand-free states. It is therefore possible and desirable to rationally tune the structure of the aptasensor hairpin to achieve a desirable timescale to resolve the detection and background Poisson peaks.

![](dt=2026-05-04/ht=12/f31e11c28591d1d2bffccb803eed426887a6a526a93ab2e8f6b1f0371d49372d.jpg)

![](dt=2026-05-04/ht=12/5fa6a0d11338e3d12c3ad10957f6f501d1400a9cc745c1dbd68ef07ec669e4af.jpg)

![](dt=2026-05-04/ht=12/b1175085d23770f91f14bb8cd70a775131f034a64635552027b4d34843ce0051.jpg)

![](image)
esult=success/type=image/dt=2026-05-04/ht=12//6de7f03b7c7963ca4e4c2c748426e2232d4ef1e7ed0fac7d0f0d714e42f7fe87.jpg)

![](dt=2026-05-04/ht=12/b68b300ea7cd0a88aa985c162e950d595a2b6e8053de4745492eda91b54b7a2b.jpg)

![](dt=2026-05-04/ht=12/569322eafa4997bddbbb17aefcaa7b46aa86f1e7a204c1116dd61c4f26c1fa28.jpg)

Theoretical consideration of the interactions of fluorescent probe and hairpin-shaped aptasensor. Fortunately, the hybridization kinetics of DNA can be tuned and predicted in a programmable manner.[31,32] DNA hairpins in particular serve as excellent model systems for fine-tuning hybridization.[33-35] To this end, it was necessary first to construct the relation of the hybridization kinetics of our fluorescent probe and the hairpin's opening and closing dynamics.

The rate constants of the fluorescent probe binding and dissociation in the ligand-free state are denoted as $k_{on,f}$ and $k_{off,f}$ (Figure 4A), and those in the ligand-bound state are $k_{on,b}$ and $k_{off,b}$ (Figure 4B), respectively. We used the two-state assumption of DNA hairpin formation to simplify the system, although a more complicated, multistate model with intermediates can also be used.[36,37] The rate constants for unfolding and folding of the hairpin are denoted as $k_{open}$ and $k_{close}$ , respectively.

The hairpin equilibrium constant is:

$$
K _ {e q, H} = \frac {k _ {\text {o p e n}}}{k _ {\text {c l o s e}}} = \frac {[ H _ {o} ]}{[ H _ {c} ]} \tag {1}
$$

where $[H_{o}]$ and $[H_{c}]$ are the concentrations of the open and closed states, respectively.

Analytical Chemistry

Page 4 of 9

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

![](dt=2026-05-04/ht=12/243efba7819edf115d0d2df06232b07422f39b7986557369e03627f46321ddc0.jpg)

![](dt=2026-05-04/ht=12/4d3cdaed8f5e1fede132c83b24c1d264caba990a38a883e4f5c0a0146901672a.jpg)

![](dt=2026-05-04/ht=12/26cdb71fb9715bc4357372fe910648fd9b8d39f25c25c16bcd67d52269b9a3e5.jpg)

![](dt=2026-05-04/ht=12/4c1ac7b64f224b1ef811d80f8c6cee5465c90e1ec2ba118c3d6a0cafdaac1ec2.jpg)

![](dt=2026-05-04/ht=12/e658fc5aa1ebc483084a1bf4f246c09832aea744c11c90878bb629581a9d1d08.jpg)

![](dt=2026-05-04/ht=12/747a2db67adb050ec50c25aea4ae42365ad42d24ae71ea612f966d6f8b933dfb.jpg)

![](dt=2026-05-04/ht=12/934d2f53c2893c3abb81d4a71aa33a86a401473fc150ab2e0e12f2b675ce48b1.jpg)

![](dt=2026-05-04/ht=12/4e7cca58d5879b8ed5a04f9300ebaad1b799fca890e26e0bc86e1f5c28dbc2af.jpg)

![](dt=2026-05-04/ht=12/ba813cdc8b8a698f91664aab3cbacd4fcfd6aff10d10eea38d7277401d0398d4.jpg)

![](dt=2026-05-04/ht=12/996a9612106098fde5d2374374ac55a883325883db4be0311d83c23d176f7c35.jpg)

![](dt=2026-05-04/ht=12/744219fb65c7e37799aafd7337992c9866fd8d605b788520cb2d7cfa67f5422c.jpg)

![](dt=2026-05-04/ht=12/1943c46b500f76cb24b3e47f27eb062b452522a9dc9d40daff6da9f433253f95.jpg)

![](dt=2026-05-04/ht=12/9269cd734a9d69a22219ecaf38aa44c1b63ab91f2efb85962e7b5e1e47a469af.jpg)

The reaction in Figure 4A consists of two steps, reversible hairpin unfolding followed by reversible fluorescent probe binding (Figure 4C). Single-molecule approaches have demonstrated that the folding/unfolding rate constants of DNA hairpins are significantly faster than those of the corresponding two-strand hybridization.[33,35] For our system, we characterized the folding and unfolding kinetics of hairpin structures at the single-molecule level by using fluorophore and quencher labeled hairpins (Figure S5A-C). The kinetics are closely correlated with the melting temperatures (Figure S6).

The As shown in Figure S7A-D, the rate constant of hairpin opening and closing is 100- to 800-fold faster than that of binding of the short fluorescent probe, $v_{1}^{+} >> v_{2}^{+}$ (Figure 4C). Therefore:

$$
v ^ {+} = v _ {2} ^ {+} \tag {2}
$$

$$
k _ {o n, f} \left(\left[ H _ {o} \right] + \left[ H _ {c} \right]\right) [ F ] = k _ {o n, b} \left[ H _ {o} \right] [ F ] \tag {3}
$$

where $[F]$ is the concentration of the short fluorescent probe. After substituting $K_{eq,H}[H_c]$ for $[H_o]$ , we obtain the ratio of $k_{on,f}$ and $k_{on,b}$ , which is a function of the equilibrium constant of the hairpin.

$$
\frac {k _ {\text {o n} , f}}{k _ {\text {o n} , b}} = \frac {K _ {\text {e q} , H}}{1 + K _ {\text {e q} , H}} \tag {4}
$$

Similarly, for the reverse reaction:

$$
v ^ {-} = v _ {2} ^ {-} \tag {5}
$$

We therefore have the relation of $k_{off,f}$ and $k_{off,b}$ as follows:

$$
k _ {\text {o f f}, f} = k _ {\text {o f f}, b} \tag {6}
$$

The expected $\mathrm{N_{b + d}}$ in a given time window (t) is:

$$
N _ {b + d} = \frac {t}{\tau_ {o n} + \tau_ {o f f}} \tag {7}
$$

The $\mathrm{N}_{\mathrm{b + d}}$ for the ligand bound and ligand-free states should be separated by three-fold of their standard deviations to be able to construct a high-confidence analytical method. Furthermore, according to Poisson statistics a minimal time for signal acquisition is required. This is given by the following inequality:

Page 5 of 9

Analytical Chemistry

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

$$
\frac {t}{\tau_ {o n , b} + \tau_ {o f f , b}} - \frac {t}{\tau_ {o n , f} + \tau_ {o f f , f}} \geq 3 \left(\sqrt {\frac {t}{\tau_ {o n , b} + \tau_ {o f f , b}}} + \sqrt {\frac {t}{\tau_ {o n , f} + \tau_ {o f f , f}}}\right) (8)
$$

where $\tau_{on,b}$ and $\tau_{off,b}$ are the dwell times of the fluorescent probe binding and dissociation in the presence of analyte, respectively, and $\tau_{on,f}$ and $\tau_{off,f}$ are the corresponding dwell times in the absence of analyte. For two-state binding and dissociation, the dwell times are a function of the rate constants:

$$
\tau_ {o n} = \frac {1}{k _ {o f f}} \tag {9}
$$

$$
\tau_ {o f f} = \frac {1}{k _ {o n} ^ {\prime}} \tag {10}
$$

where the pseudo-first-order rate constant $k_{on}^{\prime}$ equals $k_{on}[F]$ because the diffusing probe strand is in large access over the immobilized hairpin. By substituting the dwell times with the rate constants, we obtain:

$$
\begin{array}{l} t \left(\frac {k _ {\text {o n} , b} ^ {\prime} k _ {\text {o f f} , b}}{k _ {\text {o n} , b} ^ {\prime} + k _ {\text {o f f} , b}} - \frac {k _
{\text {o n} , f} ^ {\prime} k _ {\text {o f f} , f}}{k _ {\text {o n} , f} ^ {\prime} + k _ {\text {o f f} , f}}\right) \geq \\ 3 \sqrt {t} \left(\sqrt {\frac {k _ {\text {o n} , b} ^ {\prime} k _ {\text {o f f} , b}}{k _ {\text {o n} , b} ^ {\prime} + k _ {\text {o f f} , b}}} + \sqrt {\frac {k _ {\text {o n} , f} ^ {\prime} k _ {\text {o f f} , f}}{k _ {\text {o n} , f} ^ {\prime} + k _ {\text {o f f} , f}}}\right) \end{array} \tag {11}
$$

According to the hybridization of the fluorescent probe we used in this study, the $k_{off,b}$ is $\sim 10$ -fold larger than $k_{on,b}^{\prime}$ (Figure S3). Moreover, $k_{on,b}^{\prime}$ is larger than $k_{on,f}^{\prime}$ , and $k_{off,f}$ equals $k_{off,b}$ . For simplicity, equation 11 can be rearranged into:

$$
t \left(k _ {o n, b} ^ {\prime} - k _ {o n, f} ^ {\prime}\right) \geq 3 \sqrt {t} \left(\sqrt {k _ {o n , b} ^ {\prime}} + \sqrt {k _ {o n , f} ^ {\prime}}\right) \tag {12}
$$

Using equation 4, we obtain:

$$
t \geq \frac {9}{k _ {\text {o n} , b} ^ {\prime}} \left[ 1 + K _ {\text {e q}, H} + \sqrt {K _ {\text {e q} , H} \left(1 + K _ {\text {e q} , H}\right)} \right] ^ {2} \tag {13}
$$

For a certain short fluorescent probe, the minimal acquisition time is solely determined by the equilibrium constant of the hairpin. Various approaches have been used to study the hairpin dynamics, which is believed to mainly depend on the length of the stem.[33,37,38] For the dual-labeled hairpin, the fluorescence-ON time ( $\tau_{open}$ ) decreases with the length of the stem (Figure S6). In contrast, the fluorescence-OFF time ( $\tau_{close}$ ) exhibits the opposite trend.

Using the relation between the rate constant and dwell time, the rate constants and equilibrium constants were obtained (Figure S8). The hybridization kinetics of the fluorescent probe were tested by using different hairpin structures (Table S1). As a theoretical consideration, the ratio of $k_{on,b}$ and $k_{on,f}$ is a function of $K_{eq,H}$ (Figure 4D), whereas the ratio of $k_{off,b}$ and $k_{off,f}$ is close to one and independent of

$K_{eq,H}$ (Figure 4E). Furthermore, the empirical minimal time shows good agreement with the theoretical prediction. For all nucleic acid constructs, the ligand bound and ligand-free states can be discriminated within $20\mathrm{min}$ (Figure 4F). A hairpin with a longer stem yields limited accessibility for the fluorescent probe in the absence of analyte and a larger change in the hybridization kinetics of the fluorescent probe upon ligand binding (Table S1), rendering the time needed to resolve the Poisson peaks shorter.

Using this single-molecule kinetics fingerprinting, background signal due to sensor leakage is completely excluded. Hence, the probe structure can be optimized for acquisition time and sensitivity. The binding of aptamer and small molecule ligand is near irreversible with a slow dissociation rate<sup>30</sup> that permits the open conformation of the hairpin aptasensor within the detection timescale shown in Figure 4F. As shown in Figure S9, a hairpin with a shorter stem yields more single-molecule counts that more quickly resolve the Poisson peaks.

![](dt=2026-05-04/ht=12/31837ee43b1a1a178c12322b34fed9ec2f1d8e1159488717cf8379de04f50af4.jpg)

![](dt=2026-05-04/ht=12/8df4f26455502214a0aa408f6f39dbaf40633d4bf98c1db2693b68a4b5b60ae5.jpg)

![](dt=2026-05-04/ht=12/3cf52cb48ceb20593453de5227c885048687b139c5fd0c6526043e2791840413.jpg)

![](dt=2026-05-04/ht=12/7761d551d0b06e13c4f4987415889e72219cdc83d2a2e544ad348aebbf77b2ad.jpg)

![](dt=2026-05-04/ht=12/59774ef0638222b329d558a7400d65bb2b020d222c94a7550466a32b57d1d9bc.jpg)

![](dt=2026-05-04/ht=12/0b0f69be9d6bd30559d539167ddafc5bcc4edcb4810871fc51252747d5ea963c.jpg)

![](dt=2026-05-04/ht=12/96724dd673b32b1859003fe6ea549f54b3dcfa32545b567a22dd44cf55c9bf7c.jpg)

![](dt=2026-05-04/ht=12/72a081e50302bf5dc0bc957ade731499968556610a271c2bbed596a9ded222b6.jpg)

Analytical Chemistry

Page 6 of 9

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

chicken meat samples with proteinase K and SDS. The hairpin with a 6-nt stem for adenosine was used. Error bars represent s.e.m. from triplicate experiments. The probes for adenosine are the same as those in Figure 2. For other probes, see Figure S1D.

Ultrasensitive detection of different small molecule ligands. We chose a hairpin with a 6-nt stem as the probe for adenosine detection. A standard curve was constructed with the linear portion from $0.5\mathrm{pM}$ to $50~\mathrm{pM}$ yielding an LOD of 0.3 pM, calculated as three standard deviations above the blank (Figure 5A). The sensitivity is superior to some recently reported aptasensors.[39-41] Furthermore, the selectivity of an aptasensor is highly dependent on the differential binding affinity of the analyte compared its molecular analogs.

High sensitivity is usually helpful to achieve high selectivity because analogs at low concentration cannot bind the aptamer effectively. As shown in Figure 5D, no other nucleosides, including guanosine, cytidine, and thymidine, gave a significantly higher signal than the blank. To demonstrate the generality of this approach, additional analytes were tested. Acetamiprid is one of the most extensively used pesticides, and its accumulation in agricultural products may give rise to potential health risks due to its neurotoxicity.

[42] $3,3^{\prime},4,4^{\prime}$ -Tetrachlorobiphenyl (PCB-77) is a polychlorinated biphenyl that persists as an organic pollutant found to be associated with neuroreproductive toxicity, immunotoxicity, and tumor promotion.[43] A significant difference in the binding kinetics between the bound and ligand-free states was also found for these two analytes using similarly constructed aptasensors (Figure S10). Arbitrarily high discrimination was achieved within $20\mathrm{min}$ of kinetic fingerprinting (Figure S11).

The LODs of acetamiprid and PCB-77 were determined as $0.35\mathrm{pM}$ and $0.72\mathrm{pM}$ , respectively (Figure 5B and C), which is superior than recently reported biosensors.[44-47] High selectivity was also achieved for these two analytes (Figure 5E and F). The performance of our single-molecule aptasensors in practical applications was further evaluated by spiking acetamiprid and PCB-77 into chicken meat extracts as more biologically relevant samples.

The measured concentrations (calculated from the calibration curves) were strongly correlated with the nominal spiked-in concentrations (Figure 5G and H, $R > 0.98$ ).

# CONCLUSION

In summary, single-molecule kinetic fingerprinting provides a unique approach to probe the dynamic behavior of aptamer-ligand binding processes, allowing for high-accuracy discrimination between analyte bound and apo states of single aptasensor molecules. The leakage of the aptasensor was suppressed by sorting single-molecule time traces by their kinetic probing behavior. Three analytes were detected with remarkable sensitivity. The high sensitivity renders this aptasensor readout
suitable as an alternative approach to standard methods such as HPLC-MS.

Our generalizable design holds great potential to be extended to non-TIRFM based detection platforms and can potentially find broad application for nucleic acid based biosensors, ligand-initiated logic gate computing, and alternative single-molecule immunoassays.

# ASSOCIATED CONTENT

Supporting Information. The Supporting Information is available free of charge on the ACS Publications website.

The sequences of oligonucleotides used in this work are listed in Table S1. The binding and dissociation kinetics parameters of the fluorescent probe with the hairpin probe in the

presence/absence of target are shown in Table S2. The supporting figures are included.

# AUTHOR INFORMATION

# Corresponding Author

# Author Contributions

These authors contributed equally. Notes

The authors declare no competing financial interest.

# ACKNOWLEDGMENT

We thank A. Johnson-Buck from the University of Michigan for the MATLAB analysis code. This work was supported by NSFC (31600687), the Technical Innovation Project of the Chinese Academy of Agricultural Sciences, the National Risk Assessment Project of Agro-food Safety and Quality of the Ministry of Agriculture (GJFP201800704), Fundamental Research Funds for the Central Universities (12060090071, 12060046030), Beijing Young Scholar Funds (2016000020124G033), the 13th Five-Year Major Projects (2018ZX09721001), and NIH grant R21 CA204560.

# REFERENCES

Page 7 of 9

Analytical Chemistry

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

Analytical Chemistry

Page 8 of 9

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment

For TOC only

![](dt=2026-05-04/ht=12/eb3f17ba59afcf36ae93c41e07460be0cc3d85097f6b48075b2a3650da0b54ff.jpg)

Page 9 of 9

Analytical Chemistry

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20   
21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31   
32   
33   
34   
35   
36   
37   
38   
39   
40   
41   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60

ACS Paragon Plus Environment