# Article

# Quantitation of Site-Specific Glycosylation in Manufactured Recombinant Monoclonal Antibody Drugs

Nan Yang, Elisha Goonatilleke, Dayoung Park, Ting Song, Guorong Fan, and Carlito B. Lebrilla  
Anal. Chem., Just Accepted Manuscript • DOI: 10.1021/acs.analchem.6b00963 • Publication Date (Web): 16 Jun 2016  
Downloaded from http://pubs.acs.org on June 17, 2016

# Just Accepted

"Just Accepted" manuscripts have been peer-reviewed and accepted for publication. They are posted online prior to technical editing, formatting for publication and author proofing. The American Chemical Society provides "Just Accepted" as a free service to the research community to expedite the dissemination of scientific material as soon as possible after acceptance. "Just Accepted" manuscripts appear in full in PDF format accompanied by an HTML abstract. "Just Accepted" manuscripts have been fully peer reviewed, but should not be considered the official version of record.

They are accessible to all readers and citable by the Digital Object Identifier (DOI®). "Just Accepted" is an optional service offered to authors. Therefore, the "Just Accepted" Web site may not include all articles that will be published in the journal. After a manuscript is technically edited and formatted, it will be removed from the "Just Accepted" Web site and published as an ASAP article.

Note that technical editing may introduce minor changes to the manuscript text and/or graphics which could affect content, and all legal disclaimers and ethical guidelines that apply to the journal pertain. ACS cannot be held responsible for errors or consequences arising from the use of information contained in these "Just Accepted" manuscripts.

analytical chemistry

Subscriber access provided by UNIV OF CAMBRIDGE

A C S

ACS Publications

Analytical Chemistry is published by the American Chemical Society. 1155 Sixteenth Street N.W., Washington, DC 20036

Published by American Chemical Society. Copyright © American Chemical Society. However, no copyright claim is made to original U.S. Government works, or works produced by employees of any Commonwealth realm Crown government in the course of their duties.

# Quantitation of Site-Specific Glycosylation in Manufactured Recombinant

# Monoclonal Antibody Drugs

Nan Yang $^{1,3,4}$ , Elisha Goonatilleke $^{2,4}$ , Dayoung Park $^{2}$ , Ting Song $^{2}$ , Guorong Fan $^{1,3}$ , Carlito B. Lebrilla $^{2,*}$

# Corresponding Author

*E-mail: cblebrilla@ucdavis.edu. Phone: +1 530 752 6364. Fax: +1 530 752 8995.

Page 1 of 27

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

1

ACS Paragon Plus Environment

# ABSTRACT

During the development of recombinant monoclonal antibody (rMAb) drugs, glycosylation receives particular focus because changes in the attached glycans can have a significant impact on the antibody effector functions. The vast heterogeneity of structures that exist across glycosylation sites hinders the in-depth analysis of glycan changes specific to an individual protein within a complex mixture.

In this study, we established a sensitive and specific method for monitoring site-specific glycosylation in rMAbs using multiple reaction monitoring (MRM) on an ultra high performance liquid chromatography - triple quadrupole MS (UHPLC-QqQ-MS). Our results showed that irrespective of the IgG subclass expressed in the drugs, the N-glycopeptide profiles are nearly the same but differ in abundances. In all rMAb drugs, a single subclass of IgG comprised over $97\%$ of the total IgG content and showed over $97\%$ N-glycan site occupancy.

This study demonstrates the utility of an RMM-based method to rapidly characterize over 130 distinct glycopeptides and determine the extent of site occupancy within minutes. Such multi-level structural characterization is important for the successful development of therapeutic antibodies.

Analytical Chemistry

Page 2 of 27

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

2

ACS Paragon Plus Environment

# INTRODUCTION

Recombinant monoclonal antibody (rMAb) drugs have emerged as an effective biopharmaceutical for cancer and other chronic diseases $^{1-4}$ due to the specificity of these drugs toward target antigens. They function by activating the immune system to kill tumor cells, blocking the signal transduction of tumor cells to proliferate, or carrying drugs to tumor cells as radiation targets. $^{1}$ To date, more than 30 rMAb drugs have been approved and hundreds of new candidates are in development or under clinical trials.

$^{5}$ To date, all licensed rMAbs have been of the immunoglobulin G (IgG) class; however, the four subclasses of IgG (IgG1, IgG2, IgG3 & IgG4) also exhibit unique effector functions. $^{6}$ Therefore, it is important to select the appropriate IgG subclass and modulate the glycosylation to have the most potent activity for a given disease. Previous studies provide evidence that the physiochemical properties and functions of IgG are governed by its glycosylation, particularly for ligand binding and activation.

$^{7-10}$ These functions can vary depending on the type of N-glycan structures associated with the specific therapeutic drug. $^{10}$ Accordingly, in manufacturing rMAbs, the site-specific N-glycosylation and assessment of N-glycan site occupancy are of utmost importance. $^{11-14}$

Today, the focus on discovery and development of rMAb drugs continues to grow rapidly within the pharmaceutical industry, driven by a recognition of their significant advantages over traditional small molecule drugs.[15] Characterization during drug development and production presents continuing challenges in analysis and quality control.[16] Consequently, there is an urgent demand for developing high-performance analytical techniques for characterization of N-glycosylation and quantitation of the N-glycan site occupancy of

Page 3 of 27

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

3

ACS Paragon Plus Environment

rMAbs.

Quantitation of protein N-glycosylation is a challenging task. The lack of commercially available N-glycosylated standards precludes absolute quantitation, thus the quantitation of glycosylation is performed by relative comparisons of glycan/glycopeptide signals obtained using various detection systems. $^{17,18}$ Until now, capillary electrophoresis with laser-induced fluorescence detection has been applied for the analysis of released N-linked carbohydrate moieties from an IgG1 monoclonal antibody, rituximab.

$^{19}$ However, based on this technique, structural identification is obtained by chromatographic retention times, which cannot be used for characterizing unknown compounds. Furthermore, spectrophotometric detection of released glycans requires specific derivatization, which results in inter-laboratory variability due to incomplete derivatization. $^{20}$ An alternative to chromophoric labeling is the use of high-performance anion-exchange chromatography with pulsed amperometric detection (HPAEC-PAD) for the released native glycans.

$^{20,21}$ Alt
hough the technique is simpler, it is associated with less sensitivity and selectivity compared to LC with fluorescence detection. For these reasons, in recent glycan quantitation research, LC-MS has become the common technique in analyzing released N-glycans, using Q-TOF, $^{21,22}$ IT $^{23,24}$ or Orbitrap $^{25}$ MS. However, because the glycans are released from the proteins prior to analysis, information about the original protein and site of attachment are lost.

Site-specific characterization of glycosylation is a powerful, more informative analytical tool in evaluating which specific sites are susceptible to changes when monitoring quality control and establishing the impact of introducing new steps during rMAb expression and purification. Triple quadrupole (QqQ) mass spectrometry with multiple reaction monitoring (MRM) is valued for its potential

Analytical Chemistry

Page 4 of 27

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

4

ACS Paragon Plus Environment

towards the reliable quantitation of analytes of low abundance in complex mixtures. $^{26}$ We previously showed that MRM is a robust and sensitive technique for the characterization of immunoglobulin G and for site-specific quantitation relative to the protein content. $^{27}$

In this study, we have refined the MRM method to observe and quantify high and low abundant N-glycopeptides in both a protein- and site-specific manner directly from rMAbs without protein enrichment nor N-glycan release. Because the glycopeptide absolute ion abundances are greatly affected by protein concentration, we adopted a normalization method in which glycopeptide signals are normalized to the abundance of a distinguishing peptide belonging to the parent IgG subclass. Figure 1a shows the glycopeptide normalization method for IgG1, which was also applied for the other IgG subclasses.

Furthermore, to quantify N-glycan site occupancy of rMAbs, we utilized MRM to detect the conversion of asparagine to aspartic acid at the glycosylation site by releasing the N-glycans with PNGase F, which increases the peptide molecular weight by 0.984 Da. (Figure 1b). This module was developed for IgG1 and IgG2 subclasses as the six rMAbs that we analyzed were mainly composed of IgG1 and IgG2. The MRM quantitation methods employed in this study enable rapid analysis of multiple glycoforms simultaneously within a run time of 10 minutes per sample.

Here, we monitored over 130 glycopeptide transitions and determined the site-specific glycosylation and the site occupancy for each rMAb drug.

# EXPERIMENTAL PROCEDURES

# Chemicals and Reagents

The rMAb drugs used in this study, panitumumab, trastuzumab, cetuximab, bevacizumab,

Page 5 of 27

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

5

ACS Paragon Plus Environment

rituximab and infliximab, were obtained from the University of California Davis Medical Center. IgG1 and IgG2 peptide standards (EEQYNSTYR, EEQYDSTYR, EEQFNSTFR and EEQFDSTFR) were purchased from A&A Labs (San Diego, CA). Sequencing grade modified trypsin and dithiothreitol (DTT) were purchased from Promega (Madison, WI). Iodoacetamide (IAA) was purchased from Sigma-Aldrich (St. Louis, MO). Peptide N-glycosidase F (PNGase F) was obtained from New England Biolabs (Ipswich, MA). All reagents were of analytical or HPLC grade.

# Compositional Analysis and Quantitation of N-Glycopeptides in rMAbs

Samples were prepared by using $40~\mu \mathrm{g}$ of rMAbs reconstituted in $50~\mathrm{mM}$ $\mathrm{NH_4HCO_3}$ to a total volume of $100~\mu \mathrm{L}$ . Proteins were reduced using $2~\mu \mathrm{L}$ of $550~\mathrm{mM}$ DTT in a $60^{\circ}\mathrm{C}$ water bath for $50~\mathrm{min}$ , and alkylated using $4~\mu \mathrm{L}$ of $450~\mathrm{mM}$ IAA at room temperature in the dark for $30~\mathrm{min}$ .

Then, $1~\mu \mathrm{g}$ of trypsin in $10~\mu \mathrm{L}$ of $50~\mathrm{mM}$ $\mathrm{NH_4HCO_3}$ was added and proteins were digested in a $37^{\circ}\mathrm{C}$ incubator for $18~\mathrm{h}$ . When digestion was completed, the samples were kept at $-20^{\circ}\mathrm{C}$ for $1~\mathrm{h}$ to stop the reaction. The resulting peptide samples were used directly for mass spectrometric analysis without further sample cleanup or dilution.

# Determination of N-Glycan Site Occupancy of rMAbs

Accurate amounts of IgG1 and IgG2 peptide standards (EEQYNSTYR, EEQYDSTYR, EEQFNSTFR and EEQFDSTFR) were weighed using a XP26 microbalance (Mettler Toledo, Columbus, OH), and dissolved in $50~\mathrm{mM}$ $\mathrm{NH_4HCO_3}$ to make $4\mathrm{mg / mL}$ stock solutions. A 25 $\mu \mathrm{L}$ stock solution of IgG1 and IgG2 were combined to make a standard peptide mixture. The standard protein mixture was serially diluted in nanopure water to obtain calibration curves for quantitation. For the sample preparation, $40~\mu \mathrm{g}$ of rMAbs were reconstituted in $50~\mathrm{mM}$

Analytical Chemistry

Page 6 of 27

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

6

ACS Paragon Plus Environment

$\mathrm{NH_4HCO_3}$ to a total volume of $100~\mu \mathrm{L}$ . Proteins were reduced using $2\mu \mathrm{L}$ of $550~\mathrm{mM}$ DTT in a $60^{\circ}\mathrm{C}$ water bath for $50\mathrm{min}$ , and alkylated using $4\mu \mathrm{L}$ of $450~\mathrm{mM}$ IAA at room temperature in the dark for $30\mathrm{min}$ . Then, $1\mu \mathrm{g}$ of trypsin in $10~\mu \mathrm{L}$ of $50~\mathrm{mM}$ $\mathrm{NH_4HCO_3}$ was added and proteins were digested in a $37^{\circ}\mathrm{C}$ incubator for $18\mathrm{h}$ .

After digestion, the samples were kept at $-20^{\circ}\mathrm{C}$ for $1\mathrm{h}$ to stop the reaction and then the samples were thawed at room temperature before N-glycan release. To release the N-glycans, $2\mu \mathrm{L}$ of PNGase F was added to the samples, which were then incubated at $37^{\circ}\mathrm{C}$ in a microwave reactor (CEM Corporation, Matthews, NC) for $10\mathrm{min}$ at 20 watts. The samples were purified using solid phase extraction (SPE).

The C18 SPE cartridge was preconditioned with three column volumes of pure water in $0.1\%$ TFA, three volumes of $80\%$ acetonitrile (ACN), and three volumes of pure water in $0.1\%$ TFA. The samples were loaded on the column and washed with three volumes of pure water in $0.1\%$ TFA, prior to eluting with two volumes of $40\%$ ACN in $0.1\%$ TFA and two volumes of $80\%$ ACN in $0.1\%$ TFA, and dried completely. The samples were reconstituted with $100~\mu \mathrm{L}$ nanopure water prior to injection.

# Nano-LC-Chip-Quadrupole-Time-of-Flight (Q-TOF) MS/MS Analysis

Tandem MS data of peptides and glycopeptides were obtained by injecting $2\mu \mathrm{L}$ of sample into an Agilent 1200 series HPLC-Chip system coupled to an Agilent 6520 Q-TOF mass spectrometer (Agilent Technologies, Santa Clara, CA). The microfluidic chip consisted of C18 $(300\AA, 5\mu \mathrm{m})$ enrichment $(4\mathrm{mm}, 40\mathrm{nL})$ and separation $(43\mathrm{mm} \times 75\mu \mathrm{m})$ columns with a nanoelectrospray tip. LC separation was performed using a 60-min binary gradient at a
flow rate of $0.3~\mu \mathrm{L / min}$ . Solvent A consisted of $3\%$ acetonitrile and $0.1\%$ formic acid in nanopure water $(\mathrm{v / v})$ ; solvent B consisted of $90\%$ acetonitrile and $0.1\%$ formic acid in nanopure water

Page 7 of 27

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

7

ACS Paragon Plus Environment

(v/v). The mass spectrometer was operated in the positive mode. Collision energies $(\mathrm{V}_{\mathrm{collision}})$ were calculated on the basis of $m / z$ values using equation (1) for peptides and equation (2) for glycopeptides.

$$
V _ {\text {c o l l i s i o n}} = 3. 6 \mathrm {V} \left(\frac {m / z}{1 0 0 D a}\right) - 4. 8 \mathrm {V} \tag {1}
$$

$$
V _ {\text {c o l l i s i o n}} = 1. 8 \mathrm {V} \left(\frac {m / z}{1 0 0 D a}\right) - 2. 4 \mathrm {V} \tag {2}
$$

# Ultra High Performance Liquid Chromatography (UHPLC)-Triple Quadrupole (QqQ) MS Analysis

The MRM method was developed on an Agilent 1290 Infinity UHPLC system coupled to an Agilent 6490 triple quadrupole mass spectrometer (Agilent Technologies, Santa Clara, CA). An Agilent Eclipse Plus C18 column (RRHD $1.8\mu \mathrm{m}$ , $2.1\mathrm{mm} \times 100\mathrm{mm}$ ) was used for UHPLC separation.

For quantitation of peptides and glycopeptides, $2\mu \mathrm{L}$ of sample was injected and separated by using a 10-minute binary gradient with solvent A consisting of $3\%$ acetonitrile and $0.1\%$ formic acid; solvent B consisting of $90\%$ acetonitrile and $0.1\%$ formic acid in nanopure water (v/v) at a flow rate of $0.5~\mathrm{mL / min}$ . A 10-minute gradient was applied as follows: $0\mathrm{min}$ at $2\%$ B; $2.5\mathrm{min}$ at $5\%$ B; $7.0\mathrm{min}$ at $40\%$ B; the column was washed at $100\%$ B from $7.1\mathrm{min}$ to $8.6\mathrm{min}$ , and reequilibrated at $2.0\%$ B from $8.7\mathrm{min}$ to $10\mathrm{min}$ .

The quantitation of N-glycan site occupancy was done by using a 13-minute binary gradient as follows: $0 \sim 2$ min at $2\%$ B; $6 \mathrm{~min}$ at $6\%$ B; $6.1 \mathrm{~min}$ at $8\%$ B; $10 \sim 11$ min at $10\%$ B; the

Analytical Chemistry

Page 8 of 27

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

8

ACS Paragon Plus Environment

column was washed at $100\%$ B from 11.1 min to 12 min, and reequilibrated at $2\%$ B from 12.1 min to 13 min.

To reduce the cycle time, dynamic MRM mode was used at unit resolution. For this analysis, the cycle time was fixed at $500~\mathrm{ms}$ . The dwell time was varied depending on the number of concurrent transitions. Ionization was performed in the positive mode. Results were analyzed using MassHunter Quantitative Analysis B.06.00 (Agilent Technologies, Santa Clara, CA).

# RESULTS AND DISCUSSION

A system-wide glycoproteomic analytical platform based on multiple reaction monitoring (MRM) was developed for the characterization of therapeutic monoclonal antibodies at the site-specific level, enabling quantitation of distinct glycoforms without glycan release and protein enrichment steps. Using this approach, the glycosylation of IgG molecules in six rMAb drugs was mapped in the following way: Per given occupied glycosylation site, we determined the heterogeneity of attached glycans and the degree of site occupancy.

# Construction of the Dynamic MRM Method

# Tandem MS of Peptides and Glycopeptides in rMAbs

To build MRM transitions, the collision induced dissociation (CID) behavior of the selected surrogate glycopeptides and quantitating peptides was initially examined using Q-TOF-MS/MS. The tandem mass spectra of two glycopeptides (Hex $_3$ HexNAc $_4$ Fuc $_1$ -IgG1 and Hex $_3$ HexNAc $_4$ Fuc $_1$ -IgG2) are shown in Figure 2a and 2b, where the abundant ions are characteristic of glycan fragmentation. Thus, for glycopeptide identification, the most

Page 9 of 27

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

9

ACS Paragon Plus Environment

abundant and common carbohydrate oxonium ions, HexNAc $(m/z 204.08)$ and $\mathrm{Hex}_1\mathrm{HexNAc}_1$ $(m/z 366.14)$ ,[28-30] were used as diagnostic fragments (Figure S1). Glycopeptides were identified using a library of glycan structures released from the same rMAb drugs.[22] A partial list of the MRM transitions and their respective fragmentation voltages is shown in Table 1. The complete list is given in the Supporting Information (Table S1).

Each subclass of IgG was differentiated by a distinguishing peptide: FNWYVDGVEVHNAK (IgG1); CCVECPPCPAPPVAGPSVFLFPKKPK (IgG2); WYVDGVEVHNAK (IgG3); TTPPVLDSDGSFFLYSR (IgG4). For these peptides, the following transitions were determined to be optimal based on their fragmentation patterns: $([\mathrm{M} + 2\mathrm{H}]^{2+}$ $839.4 \rightarrow m/z$ 968.5 and $m/z$ 1067.6) for IgG1, $([\mathrm{M} + 3\mathrm{H}]^{3+}$ $970.1 \rightarrow m/z$ 1100.6 and $m/z$ 839.5) for IgG2, $([\mathrm{M} + 3\mathrm{H}]^{3+}$ $472.9 \rightarrow m/z$ 697.4 and $m/z$ 534.3) for IgG3 and $([\mathrm{M} + 3\mathrm{H}]^{3+}$ $635.0 \rightarrow m/z$ 1217.6 and $m/z$ 425.2) for IgG4.

For the absolute quantification of N-glycan site occupancy, the appropriate product ions were selected for MRM according to abundances as well as the sequence to include ions containing an asparagine (N). The corresponding peptide replaced by an aspartic acid (D) residue after N-glycan release was subsequently monitored (e.g., EEQYNSTYR/EEQYDSTYR, EEQFNSTFR/EEQFDSTFR). A representative fragmentation spectrum of the IgG1 (N) peptide backbone EEQYNSTYR is shown in Figure 2c.

For the quantitation of site occupancy in IgG1 and IgG2, the following transitions were determined to be optimal: $([\mathrm{M} + 2\mathrm{H}]^{2+}$ $595.25\rightarrow m / z$ 803.35 and $m / z$ 640.30) for the IgG1 (N) peptide EEQYNSTYR; $([\mathrm{M} + 2\mathrm{H}]^{2+}$ $595.75\rightarrow m / z$ 804.35 and $m / z$ 641.29) for the IgG1 (D) peptide EEQYDSTYR; $([\mathrm{M} + 2\mathrm{H}]^{2+}$ $579.27\rightarrow m / z$ 771.35 and $m / z$ 624.29) for the IgG2 (N) peptide

Analytical Chemistry

Page 10 of 27

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

10

ACS Paragon Plus Environment

EEQFNSTFR; and $\left[\mathrm{M} + 2\mathrm{H}\right]^{2+}579.76 \rightarrow m/z772.36$ and $m/z625.29)$ for the IgG2 (D) peptide EEQFDSTFR.

# Multiple Reaction Monitoring of IgG Subclasses and Their Glycoforms in rMAbs

From the six rMAb drugs, a total of four peptides, representing each of the IgG subclasses, and 138 unique glycopeptides were monitored. All peptides and glycopeptides were separated using C18 ultra high performance liquid chromatography (UHPLC). As demonstrated in Figure 3a, good separation of the IgG peptides was achieved within two minutes. These peptides showed high repeatability and were used for quantitation. The MRM chromatograms of the glyco
peptides from the four IgG subclasses are shown in Figure 3b.

Glycopeptides from IgG1 eluted at $2.6\mathrm{min}$ , followed by IgG3/4 glycopeptides at $3.7\mathrm{min}$ and IgG2 glycopeptides at $4.2\mathrm{min}$ . In general, the glycopeptides eluted earlier than the peptides, resulting in higher sensitivity and less charge competition. The representative chromatogram in Figure 3c shows the responses of the IgG1, de-glycosylated IgG1, IgG2 and de-glycosylated IgG2 peptides that were used for quantitation of site occupancy.

The peptide-centric separation of glycopeptides on the C18 stationary phase results in co-elution of glycoforms that share the same peptide backbone. However, performing multiple concurrent transitions will necessitate either a longer cycle time, which will lower the sampling efficiency, or a shorter dwell time, which will result in a poor signal-to-noise ratio. Therefore, in our experiments, dynamic MRM mode was applied wherein the transitions are performed only at a specific time segment, reserving the duty cycle for compounds with overlapping retention times.[27] The dynamic MRM transitions employed for all rMAbs are included in Table S1. In addition, to further reduce the number of concurrent

Page 11 of 27

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

11

ACS Paragon Plus Environment

transitions, only one transition was chosen for each glycopeptide according to abundance. We have previously shown that single transition monitoring in conjunction with dynamic MRM provides sufficient specificity as it both identifies the compound as a glycopeptide and enables quantitation.[256]

# Analysis of IgG Glycosylation in rMAbs

# Quantitation of Glycopeptides and Protein Glycosylation

Due to the lack of available standards, relative quantitation of protein glycosylation is provided by using absolute ion abundances. $^{17,31,32}$ However, because protein concentration greatly influences signal intensity, we adopted a normalization method to account for the differences in IgG1-4 content in the rMAb drugs. Each glycopeptide was normalized to the abundance of the corresponding IgG molecule as follows:

$$
\text {D e g r e e o f g l y c o s y l a t i o n} = \frac {\text {g l y c o p e t i d e i o n a b u n d a n c e}}{\text {p r o t e i n a b u n d a n c e (p e p t i d e i o n a b u n d a n c e)}}
$$

The quantifying peptides were selected based on the conditions that it must be unique only to the originating subclass, abundant and devoid of post-translational modifications (PTMs).

IgG1 and IgG2 yielded distinct glycopeptides that could be individually monitored and normalized to their related subclasses. However, glycopeptides from IgG3 (EEQYN*STFR) and IgG4 (EEQFN*STYR) contain the same amino acid residues and therefore could not be distinguished. Consequently, for IgG3 and IgG4, the abundances of the two contributing peptides were summed together for normalization.

Analytical Chemistry

Page 12 of 27

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

12

ACS Paragon Plus Environment

# N-Glycan and N-Glycopeptide Profiles of rMAb Drugs

Following IgG subclass quantification, we observed that all of the rMAb drugs in this study are predominantly IgG1 except panitumumab, which mainly consists of IgG2 (Figure S2). These results show that the rMAb drugs are not purely of one subclass but the main subclass comprises over $97\%$ of the total IgG.

For each rMAb drug, we classified the glycans by their originating glycopeptide and by abundance to compare glycoproteome quantitation data with released glycan analysis. For example, the N-glycan profile of IgG1 from bevacizumab is shown in Figure 4a. We have previously compiled an N-glycan library of over 70 structures with isomer and linkage specificity based on a group of rMAbs analyzed by nano-LC electrospray ionization quadrupole time-of-flight (nano-LC-ESI-Q-TOF) MS.

[22] According to the N-glycosylation analysis, glycopeptides bearing high mannose type and sialylated biantennary complex type glycans make up less than $1\%$ in relative abundances. The most common glycan structures in IgG possess zero, one, or two terminal galactose (G) residues and up to one fucose (F), and are defined as G0, G1, G0F, G1F and G2F.[33-35] In this context, the abundances of glycopeptides analyzed in this study were grouped by the presence of these glycan structures.

Figure 4b shows the distribution of these N-glycopeptides across the main IgG subclass of each antibody drug. All six rMAbs express the common glycan structures on either IgG1 or IgG2 but in different quantities. These results agree with the previous N-glycan study, which showed that most of the N-glycans between different antibodies are nearly the same but differ in abundances. It was also observed that the most abundant glycopeptides in the rMAb drugs were fucosylated.

Page 13 of 27

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

13

ACS Paragon Plus Environment

For the relative comparison of glycoforms using MRM, a major concern is whether the response is affected by the nature of ionization and fragmentation. Recent research has shown that the ionization efficiencies of different glycoforms with the same peptide moiety are similar in electrospray ionization.[36] To evaluate the contributing effects of different fragmentation efficiencies, we compared the N-glycan distribution profiles of panitumumab from Chip-Q-TOF-MS and QqQ-MS and observed that they were similar in both analyses (Figure 5). Therefore, it is possible to compare abundances of glycopeptides from MRM signals to study the distribution of N-glycans in rMAbs.

# N-Glycan Site Occupancy of rMAb Drugs

As PNGase F treatment of glycopeptides results in the deamidation of the asparagine (N) at the NxS/T site, the asparagine (N) to aspartic acid (D) conversion is used as a 'signature' for site occupancy. Accordingly, for site occupancy quantification of rMAb drugs, PNGase F treatment was performed to remove all N-linked glycans. The deglycosylated peptides and unoccupied peptides were then monitored simultaneously using MRM. The site occupancy was determined by the absolute concentration of deglycosylated peptides (D) and unoccupied peptides (N).

Absolute concentrations of peptides were calculated using calibration curves made by serial dilutions of peptide standards, as depicted for IgG1 and IgG2 in Figure 6. The response of each peptide was plotted against concentration (0.1, 0.5, 1, 2, 5, 10, 50, 100, 200 $\mu$ g/mL) and fit with good linearity. The percentage of N-glycan site occupancy for the six rMAb drugs is shown in Figure 7. All six drugs were highly glycosylated with over $97\%$ site occupancy.

# CONCLUSION

Analytical Chemistry

Page 14 of 27

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

14

ACS Paragon Plus Environment

In this study, we developed an MRM method that is not only rapid in profiling the N-glycome but that which also provides the glycoproteome details. The established methods are effective in simultaneously determining the N-glycan compositions, their sites of attachment and the site occupancy in commercial rMAb drugs. Using this approach, we determined that the six rMAb drugs analyzed in this study have similar glycopeptides but in different quantities and all of the drugs are highly glycosylated with the N-glycan site occupancy of over $97\%$ .

The FDA and other regulatory agencies require data on the analytical characterization of rMAbs. $^{37}$ The methods developed in this study are rapid and highly specific. They can be widely used to determine the protein-specific glycan profile in rMAbs and percent site occupancy during drug development and also in quality control. Further, these methods can be easily used to check batch-to-batch consistency, which is essential because the molecular heterogeneity of rMAbs will affect their stability and their potency. Understanding the distributions of N-glycan structures at the site-specific level can provide more information on the activities of rMAbs and is beneficial in optimizing their clinical outcomes for different diseases.

# SUPPORTING INFORMATION

Additional information as noted in text. This material is available free of charge via the Internet at http://pubs.acs.org.

# NOTES

Page 15 of 27

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

15

ACS Paragon Plus Environment

The authors declare no competing financial interest.

# ACKNOWLEDGMENTS

Funding provided by the National Institutes of Health (RO1AT008759, R01GM049077

AT007079 to C.B.L.) is gratefully acknowledged. This project was also sponsored by the

China Scholarship Council.

# REFFERENCES

Analytical Chemistry

Page 16 of 27

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

16

ACS Paragon Plus Environment

Page 17 of 27

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

17

ACS Paragon Plus Environment

# FIGURE CAPTIONS

Figure 1. Illustration of method development: (a) glycopeptide normalization method for IgG1, (b) MRM method for quantification of N-glycan site occupancy in IgG1 and IgG2.

Figure 2. Representative Q-TOF tandem mass spectra of glycopeptides and peptides: (a) MS/MS spectrum of glycopeptide $\mathrm{Hex}_3\mathrm{HexNAc}_4\mathrm{Fuc}_1\_ \mathrm{EEQYNSTYR}$ from IgG1, (b) MS/MS spectrum of glycopeptide $\mathrm{Hex}_3\mathrm{HexNAc}_4\mathrm{Fuc}_1\_ \mathrm{EEQFNSTFR}$ from IgG2, and (c) MS/MS spectrum of peptide EEQYNSTYR from IgG1. A blue diamond is drawn above the selected precursor ion.

Figure 3. Representative chromatograms of peptides and glycopeptides: (a) MRM chromatogram of the four IgG subclass peptides, (b) MRM chromatogram of glycopeptides in each of the four subclasses, and (c) MRM chromatogram of glycosylated and deglycosylated IgG1 and IgG2 peptides with asparagine and aspartic acid residues.

Figure 4. Representative profile of N-glycopeptides of rMAb drugs: (a) The N-glycan profile of IgG1 glycopeptides from bevacizumab, and (b) the N-glycopeptide distribution of six rMAb drugs.

Figure 5. Comparison of the panitumumab N-glycan distribution profiles between Chip-Q-TOF-MS and QqQ-MS.

Figure 6. Calibration curves used for quantitation of: (a) peptide EEQYNSTYR, which is indicative of unoccupied IgG1; (b) peptide EEQYDSTYR, which results from IgG1 after N-glycan release; (c) peptide EEQFNSTFR, which is indicative of unoccupied IgG2; (d) peptide EEQFDSTFR, which results from IgG2 after N-glycan release.

Figure 7. The percentage of N-glycan site occupancy for six rMAb drugs, where occupied sites possessed aspartic acid (D) residues and unoccupied sites possessed asparagine (N) residues after PNGase F treatment.

Analytical Chemistry

Page 18 of 27

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

18

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/722563757fb8a265c09db982e4cfa59b4011b5be8178bcff022dfca527dd68e6.jpg)

![](dt=2026-04-15/ht=14/e527eb6ff4743fe359bd6646fed5da1764a0cf219a2fd76f84b253c36ba5209c.jpg)

![](dt=2026-04-15/ht=14/6a4c7bb57045670683a6c9afec3e22f37f5c5934a612d5586505557bdac8339a.jpg)

Page 19 of 27

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

19

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/def5d3f08a5db78b5430198cc71a51a552f86c193da228b54921ce71b0aeae2f.jpg)

![](dt=2026-04-15/ht=14/7003167f6b96979b996fd792cf0d8c6665786113e428c67ef79768823ddcc639.jpg)

![](dt=2026-04-15/ht=14/51be81f6c235c1fa74b9f5aa0a2237f16bb08dbb121c6c453fe6e97885c32848.jpg)

Analytical Chemistry

Page 20 of 27

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

20

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/d1ba09911ca965712d3a8ba1834e537cb624e1ab3726be45029575420914b5ff.jpg)

![](dt=2026-04-15/ht=14/20de9b3651a29dc7da24461ac2c821aaf5025415b3db5451034e47e41e700774.jpg)

![](dt=2026-04-15/ht=14/972894e3fe93bd89a72445c0892ccda2e685cc4e344accf4ddd9366ea6b44ad0.jpg)

Page 21 of 27

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

21

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/64ae16af7fe71b3e14cc480939590ac77dd8cd5edb377f5af3fcabe8081758f4.jpg)

![](dt=2026-04-15/ht=14/5b979547af4155040fcc1107149f7bfd294df418288decf842a73bc02b53bc8b.jpg)

Analytical Chemistry

Page 22 of 27

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

22

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/b1f794d85734bf80c2283c4575e8f741c11312a7fc87ba2c83de0dac7b98e0ef.jpg)

Page 23 of 27

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

23

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/05c81c3be23530a776bfb6472ee5b355d055f11f706fb67f84da4a7731955c35.jpg)

![](dt=2026-04-15/ht=14/22e612e16ffdea77df47d3f2d71a528b670cde67305bce8d49656083fa9079a5.jpg)

![](dt=2026-04-15/ht=14/10fe62d793a0de1cf09e5de25c2a5fda88094d77b417ba141e9862175a0ee285.jpg)

![](dt=2026-04-15/ht=14/91e2483d2f17736061a0e104b5ef52084ee46e396122adf878f12c05290bdb08.jpg)

Analytical Chemistry

Page 24 of 27

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

24

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/ff794fb2b845ce9b722c9b9c102beff18e75cf4619b6e3848e8ac13f7964619f.jpg)

Page 25 of 27

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

25

ACS Paragon Plus Environment

Table 1. MRM Transitions Used to Monitor Glycopeptides

![](dt=2026-04-15/ht=14/a36b16e5f6b881fd34813fcc446cf316ca0f34d7e5466a835044eed0dac0eb38.jpg)

<table><tr><td>Compound Namea</td><td>Precursor Ion (m/z)</td><td>Product Ion (m/z)</td><td>Collision Energy (eV)</td><td>Retention Time (min)</td><td>Delta Retention Timeb (min)</td><td>Structures</td></tr><tr><td>3.2.0.0.0_IgG1</td><td>694.6</td><td>204.1</td><td>15</td><td>2.6</td><td>1</td><td>6α 4β 4γ 3α</td></tr><tr><td>3.2.1.0.0_IgG1</td><td>743.3</td><td>204.1</td><td>16</td><td>2.6</td><td>1</td><td>6α 4β 4γ 3α</td></tr><tr><td>3.3.0.0.0_IgG1</td><td>762.3</td><td>204.1</td><td>16</td><td>2.6</td><td>1</td><td>6α 4β 4γ 3α</td></tr><tr><td>3.3.1.0.0_IgG1</td><td>811.0</td><td>204.1</td><td>17</td><td>2.6</td><td>1</td><td>6α 4β 4γ 3α</td></tr><tr><td>3.4.0.0.0_IgG1</td><td>830.0</td><td>204.1</td><td>18</td><td>2.6</td><td>1</td><td>2β 6α 4β 4γ 2β 3α</td></tr></table>

${}^{a}$ 3.2.0.0.0_IgG1 indicates 3 Hexose; 2 HexNAc; 0 Fucose; 0 N-Acetylneuraminic acid; 0 N-Glycolylneuraminic acid from IgG1. ${}^{b}$ Dynamic MRM was used. Delta retention time is the retention time window for the target transition.

Analytical Chemistry

Page 26 of 27

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

26

ACS Paragon Plus Environment

![](dt=2026-04-15/ht=14/b4740e146c62f6d9b88aa3c0fe3de21c8bd007978015a27effee9ec684f46f2b.jpg)

Page 27 of 27

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

27

ACS Paragon Plus Environment