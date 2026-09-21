High-Content Analysis of Antibody Phage-Display Library Selection Outputs Identifies Tumor Selective Macropinocytosis-Dependent Rapidly Internalizing Antibodies

Kevin D. Ha, Scott M. Bidlingmaier, Yafeng Zhang, Yang Su, and Bin Liu*

Department of Anesthesia, UCSF Helen Diller Family Comprehensive Cancer Center, University of California at San Francisco, San Francisco, CA 94110-1305 USA

*Corresponding author

Department of Anesthesia

1001 Potrero Ave., Box 1305

San Francisco, CA 94110-1305

Email: liub@anesthesia.ucsf.edu

Key words: Phage antibody display; prostate cancer; high-content analysis; receptor-dependent macropinocytosis; rapidly internalizing antibodies, EphA2

MCP Papers in Press. Published on August 22, 2014 as Manuscript M114.039768 Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

1

Copyright 2014 by The American Society for Biochemistry and Molecular Biology, Inc.

# Abbreviations

HCA: High content analysis; ScFv: single chain variable fragment; PCC: Pearson's correlation coefficient; CFU: Colony forming unit; MFI: mean fluorescence intensity; EEA: early endosomal antigen; LAMP: lysosomal-associated membrane protein; IgG: immunoglobulin G; ND70-TR: Texas Red-conjugated neutral dextran 70 kDa; FBS: Fetal bovine serum; HEK: human embryonic kidney; LCM: Laser capture microdissection; EphA2: ephrin type-A receptor 2; HRP: horseradish peroxidase; EC50: half maximal effective concentration; MAbs: monoclonal antibodies

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

2

# Summary

Many forms of antibody-based targeted therapeutics, including antibody drug conjugates, utilize the internalizing function of the targeting antibody to gain intracellular entry into tumor cells. Ideal antibodies for developing such therapeutics should be capable of both tumor-selective binding and efficient endocytosis. The macropinocytosis pathway is capable of both rapid and bulk endocytosis, and recent studies have demonstrated that it is selectively upregulated by cancer cells.

We hypothesize that receptor-dependent macropinocytosis can be achieved using tumor-targeting antibodies that internalize via the macropinocytosis pathway, improving potency and selectivity of the antibody-based targeted therapeutic. While phage antibody display libraries have been utilized to find antibodies that bind and internalize to target cells, no methods have been described to screen for antibodies that internalize specifically via macropinocytosis. We hereby describe a novel screening strategy to identify phage antibodies that bind and rapidly enter tumor cells via macropinocytosis.

We utilized an automated microscopic imaging-based, High Content Analysis platform to identify novel internalizing phage antibodies that colocalize with macropinocytic markers from antibody libraries that we have generated previously by laser capture microdissection-based selection, which are enriched for internalizing antibodies binding to tumor cells in situ residing in their tissue microenvironment [1]. Full-length human IgG molecules derived from macropinocytosing phage antibodies retained the ability to internalize via macropinocytosis, validating our screening strategy.

The target antigen for a cross-species binding antibody with a highly active macropinocytosis activity was identified as ephrin type-A receptor 2. Antibody-toxin conjugates created using this macropinocytosing IgG were capable of potent and receptor-dependent killing of a panel of EphA2-positive tumor cell lines in vitro.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

3

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

These studies identify novel methods to screen for and validate antibodies capable of receptor-dependent macropinocytosis, allowing further exploration of this highly efficient and tumor-selective internalization pathway for targeted therapy development.

4

# INTRODUCTION

There is significant interest in the development of targeted therapeutics such as antibody drug conjugates that have the potential to improve the therapeutic window of cytotoxic drugs by delivering them specifically and intracellularly to cancer cells [2-6]. The pathway by which the targeted agent enters tumor cells can influence both the uptake efficiency and the intracellular fate of the internalized agent, both of which contribute to the cytotoxic potency [6, 7].

Among known endocytic routes, macropinocytosis provides an intriguing pathway for cellular entry as it is a form of bulk uptake and can therefore efficiently and rapidly internalize targeting agents. Macropinosomes comprise large, endocytic vesicles that range from $0.2\mu \mathrm{m}$ to $3\mu \mathrm{m}$ in size, which are up to 30-fold larger than the $0.1\mu \mathrm{m}$ average size of protein-coated, endocytic vesicles [8].

Additionally, studies have shown that macropinocytosis is selectively upregulated in Ra-transformed cancers (a common oncogenic mutation in human cancers) and plays an important role in tumor cell homeostasis by serving as an amino acid supply route [9], suggesting that targeted therapeutics based on antibodies that internalize via the macropinocytosis pathway may provide additional tumor-specificity against a wide variety of human cancers.

To therapeutically explore the utility of antibodies that gain entry into tumor cells via receptor-dependent macropinocytosis, a generally applicable method needs to be developed that readily identifies such antibodies. While phage antibody display libraries have been extensively used to select for antibodies that internalize into tumor cells, no methods have been developed to uncover antibodies capable of cellular entry through the macropinocytosis pathway. To address this need, we developed a high content analysis (HCA)-based screening strategy that employs automated image-based analysis to identify phage antibodies that colocalize with Texas Red-conjugated $70\mathrm{kDa}$ neutral dextran (ND70-TR), a macropinocytosis marker. We used the HCA

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

5

protocol to screen single chain variable fragment (scFv) phage antibody display library selection outputs that we have generated previously by laser capture microdissection (LCM)-based selection on live tumor cells and tissues [1], and identified antibodies that are capable of efficient internalization via macropinocytosis.

We performed kinetics and subcellular colocalization studies for phage antibodies as well as full-length immunoglobulin G (IgG) molecules derived from the parental scFvs and identified a highly active, macropinocytosing antibody that rapidly internalizes and colocalizes with early endosomal and lysosomal markers. We further identified the target antigen as EphA2 by immunoprecipitation and mass spectrometry.

To confirm internalization by an independent functional assay and to demonstrate therapeutic potential, we created an antibody-toxin conjugate and showed potent and specific cytotoxic activity against a panel of EphA2-positive tumor cell lines. To our knowledge, this is the first description of a generally applicable screening strategy to uncover macropinocytosing antibodies, enabling further exploration of this class of antibody-antigen pairs for the development of effective antibody-targeted therapeutics.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

6

# Materials and Methods

# Tissue culture

Prostate cancer cell lines DU145 and LNCaP, breast cancer cell line MDA-MB-231, lung cancer cell line A549, cervical cancer cell line HeLa, epidermoid carcinorma cell line A431, and human embryonic kidney (HEK) 293A cell line were purchased from the American Type Culture Collection (ATCC). Human foreskin normal fibroblast line Hs27 was purchased from UCSF Cell Culture Core Facility. Benign prostatic hyperplasia (BPH-1) cells w
ere originally obtained from Dr. Gerald Cunha's lab at UCSF [10] and maintained in the lab.

All cells were grown in high-glucose, L-glutamine, and sodium pyruvate-supplemented complete DMEM (Caisson Labs) with the addition of $10\%$ fetal bovine serum (Fisher Scientific) and penicillin-streptomycin solution (Axenia BioLogix). Cells were grown in $5\%$ $\mathrm{CO}_{2}$ at $37^{\circ}\mathrm{C}$ on tissue culture-treated flasks (BD Biosciences). Cells were passaged utilizing $0.25\%$ trypsin-EDTA (Life Technologies).

Preparation of phage antibody display library selection output for screening

Phage antibody library selection outputs generated previously by LCM-based selection on prostate tumor tissues [1] were streaked onto 2X YT agar plates containing $12.5\mu \mathrm{g} / \mathrm{ml}$ tetracycline to yield monoclonal phage antibodies. In this phage display system, the scFv was fused to pIII in the fd phage display vector that carries genes necessary for phage assembly [11, 12]. Individual colonies were inoculated in 2X YT containing $12.5\mu \mathrm{g} / \mathrm{ml}$ tetracycline and grown in deep 96-well plates (Fisher Scientific) at $37^{\circ}\mathrm{C}$ with 225 RPM shaking for $18\mathrm{h}$ . The plates

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

7

were centrifuged to pellet the bacteria and supernatants containing phage particles were transferred into a new 96-well plate for HCA experiments (see below). Positive clones from initial HCA screenings were re-tested using purified phage using polyethylene glycol (PEG8000) as previously described [1, 13-15]. Antibody sequences were determined using 96-well plate-based DNA sequencing (Functional Biosciences).

# Recombinant antibody cloning, expression, and purification

For IgG production, heavy and light chain variable fragments were subcloned into IgGAbVec (kindly provided by Dr. Patrick Wilson at University of Chicago) $\gamma$ and $\lambda$ mammalian expression vectors, as previously described, to produce secretable IgG1 antibodies [16]. For scFv-Fc fusion production, scFv was subcloned from phage into pFUSE-hIgG1 Fc2 (InvivoGen).

Mammalian transfection complexes containing antibody expression DNA and polyethylenimine (Sigma-Aldrich) in Opti-MEM (Life Technologies) were added to HEK 293A cells in the presence of serum-free DMEM containing Nutridoma-SP (Roche) and penicillin-streptomycin. Antibody-containing media were harvested after 4 days and affinity-purified using protein A agarose (Pierce/Fisher). Antibody concentrations were determined using the Blitz® Bio-Layer interferometry System (ForteBio).

# HCA screening

Supernatants from 96-well bacterial culture plates (see above) were used for initial HCA screening. DU145 cells were seeded in 96-well plates (BD Biosciences) overnight, and incubated

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

8

with phage-containing supernatants and $50~\mu \mathrm{g / ml}$ TexasRed-conjugated 70-kDa neutral dextran (ND70-TR, Life Technologies) in DMEM/10% FBS at $37^{\circ}C$ with $5\%$ $\mathrm{CO}_{2}$ overnight. Cells were washed 3x with PBS, fixed with $4 \%$ paraformaldehyde (Santa Cruz Biotechnology) in PBS for $10\mathrm{min}$ , washed 3x in PBS, and then permeabilized in PBS containing $1 \%$ fraction V bovine serum albumin (Fisher Scientific) and $0.1\%$ TritonX-100 (Sigma).

Phage were detected with 3.5 $\mu \mathrm{g / ml}$ biotin-conjugated, rabbit anti-fd bacteriophage (Sigma-Aldrich) for 1h at RT followed by $1\mu \mathrm{g / ml}$ Alexa Fluor® 488-conjugated streptavidin (Jackson ImmunoResearch) for $15\mathrm{min}$ at RT. Hoechst 33342 (Thermo Scientific) at $1\mu \mathrm{g / ml}$ for $30\mathrm{min}$ at RT was used to detect nuclei. The 96-well plates were imaged on a CellInsightTM NXT HCS platform (Thermo Scientific) with a semi-aprochromat 20X LUCPLFLN objective (Olympus) utilizing $>6$ fields per well with a minimum of 300 cells per well.

Pearson's correlation coefficient analysis between ND70-TR and phage particles were conducted using Thermo Scientific HCS Studio software suite on all imaged fields and averaged per well.

# Confocal analysis:

DU145 cells were seeded in 8-well Lab-Tek II chambered coverglass (Thermo Scientific) overnight for confocal microscopy studies. Cells were incubated with antibodies (IgGs at 10 $\mu$ g/ml or purified phage at $10^{9}$ cfu/ml) and 50 $\mu$ g/ml ND70-TR in DMEM/10% FBS at $37^{\circ}$ C with $5\%$ $\mathrm{CO}_{2}$ for indicated periods (see text), washed, fixed and permeabilized as described above. To label subcellular structures, rabbit antibodies against early endosomes, lysosomes, caveolin-2, and clathrin heavy chain (Cell Signaling) were added to permeabilized cells at 1:100 dilutions for 3h at RT. Cell-associated human IgGs were detected with 1 $\mu$ g/ml Alexa Fluor®

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

9

647-conjugated goat anti-human IgG (Jackson ImmunoResearch) for $30\mathrm{min}$ at RT. Cell-associated phage were detected with $3.5~\mu \mathrm{g / ml}$ biotin-conjugated, rabbit anti-fd bacteriophage for 1h at RT followed by $1\mu \mathrm{g / ml}$ Alexa Fluor® 488-conjugated streptavidin (Jackson ImmunoResearch) for $15\mathrm{min}$ at RT. Antibodies against organelles were detected with Alexa Fluor® 488- or phycoerythrin-conjugated goat anti-rabbit for $30\mathrm{min}$ at RT. Hoechst 33342 at $1\mu \mathrm{g / ml}$ for $30\mathrm{min}$ at RT was used to detect nuclei.

Cells in 8-well glass chambered coverglass were then imaged on the FluoView® FV10i laser confocal microscope (Olympus) equipped with two galvanometer scanning mirrors. Confocal images were taken with an Olympus 60X phase contrast water-immersion objective with NA 1.2. Image analyses including Pearson's and Mander's correlation coefficients, Z-projection, Z-projection dissection, and 3D renderings were performed with the included Olympus confocal software suite.

# Internalization kinetics assay

DU145 cells seeded in 8-well chambered coverglass were pulsed with antibodies at $10\mu \mathrm{g} / \mathrm{ml}$ in complete DMEM/FBS for $30\mathrm{min}$ at $4^{\circ}\mathrm{C}$ , followed by a chase in $37^{\circ}\mathrm{C}$ warmed, complete DMEM/FBS and incubated at $37^{\circ}\mathrm{C}$ with $5\%$ $\mathrm{CO}_{2}$ . Individual wells at varying time points were then washed in PBS and fixed in $4\%$ paraformaldehyde before undergoing immunofluorescence as described above.

For flow cytometry-based internalization kinetics assay, DU145 cells were seeded in 6-well plate, then treated with antibodies at $10\mu \mathrm{g} / \mathrm{ml}$ for varying amounts of time. Then cells were trypsinized, probed with anti-human secondary antibody, and analyzed on a flow cytometer. Cytochalasin D (Sigma) was resuspended in

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

10

DMSO and cells were pulsed with $50~\mu \mathrm{g / ml}$ of the drug in serum-free DMEM at $37^{\circ}C,$ followed by a chase in complete DMEM/FBS containing the drug and antibodies.

# Immunoprecipitation of the target antigen

Purified HCA-F1 scFv-Fc fusions were first chemically cross-linked to protein A agarose beads. Briefly, antibodies were affinity-bound onto protein A agarose (Life Technologies) in a tube. Beads were then spun down and washed with $0.2\mathrm{M}$ sodium borate, $\mathrm{pH}9.0$ . Dry dimethyl pimelimidate (DMP, Sigma) was added to the beads in the presence of sodium borate to yield a final concentration of $13\mathrm{mg / ml}$ and incubated at RT for $30\mathrm{min}$ . Beads were washed with sodium borate and DMP crosslinking was repeated a second time.

Chemical crosslinking was terminated through washes with $0.2\mathrm{M}$ ethanolamine, $\mathrm{pH}8.0$ , for $2\mathrm{h}$ at RT. Finally, unconjugated antibodies were eluted from beads using $0.1\mathrm{M}$ glycine, $\mathrm{pH}2.8$ , followed by washe
s with PBS. Exposed surface proteins on DU145 cells were biotinylated using EZ-Link Sulfo-NHS-LC-Biotin (Thermo Pierce) according to manufacturer's recommendations and then lysed using standard RIPA buffer ( $50\mathrm{mM}$ Tris, $\mathrm{pH}7 - 8$ , $150\mathrm{mMNaCl}$ , $0.1\%$ SDS, $0.5\%$ sodium deoxycholate, $1\%$ NP-40).

Immunoprecipitation was performed as described previously [17, 18]. Briefly, $5\mathrm{mg}$ of biotinylated lysates were first pre-cleared against protein A agarose for $1\mathrm{h}$ at RT and then incubated with scFv-Fc-conjugated protein A beads overnight at $4^{\circ}\mathrm{C}$ . Beads were then washed with $500\mathrm{mMNaCl}$ in PBS, spun down, and boiled in SDS sample buffer to be run on two $4 - 12\%$ Tris-glycine SDS-PAGE gels (Life Technologies). One gel was GelCode-stained (Thermo) and the other gel was used for Western blotting using standard procedures.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

11

Horseradish peroxidase-conjugated streptavidin was used in the Western Blot to assess which protein band to extract from the GelCode-stained gel.

# Antigen identification by mass spectrometry analysis

Extracted gel bands were trypsin-digested and analyzed via tandem mass spectrometry (MS/MS, University of Minnesota). Charge state deconvolution and deisotoping were not performed. All MS/MS samples were analyzed using Sequest (Thermo Fisher Scientific; version 27, rev. 12). Sequest was set up to search the rs_human9606_031313_cRAP123 database (unknown version, 36010 entries) assuming the digestion enzyme trypsin. Sequest was searched with a fragment ion mass tolerance of $0.80\mathrm{Da}$ and a parent ion tolerance of $0.079\mathrm{Da}$ and 0.32 Da.

Iodoacetamide derivative of cysteine and oxidation of methionine were specified as fixed and variable modifications, respectively, in Sequest. Scaffold (version 4.0.5, Proteome Software Inc.) was used to validate protein identifications to create peak lists. Peptide identifications were accepted if they could be established at greater than $95.0\%$ probability by the Peptide Prophet algorithm [19]. Protein identifications were accepted if they could be established at greater than $90.0\%$ probability and contained at least 2 identified peptides.

Protein probabilities were assigned by the Protein Prophet algorithm [20]. Peptide and protein false discovery rates, as determined by Protein Prophet algorithm, are $0.4\%$ and $0.1\%$ , respectively. Proteins that contained similar peptides and could not be differentiated based on MS/MS analysis alone were grouped to satisfy the principles of parsimony.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

12

# Antibody-toxin cytotoxicity assay

The human IgG HCA-F1 was biotinylated with EZ-Link Sulfo-NHS-LC-Biotin (Thermo Pierce) according to manufacturer's recommendations. A panel of tumor and non-tumorigenic cell lines (see text) were seeded in 96-well plates at a density of 1,000-2,000 cells per well and grown for 16h at $37^{\circ}\mathrm{C}$ in $5\%$ $\mathrm{CO}_{2}$ .

Biotinylated IgG HCA-F1 was then incubated with streptavidin-ZAP (saporin conjugated with streptavidin, Advanced Targeting Systems) at a molar ratio of 1:1 and incubated on ice for 30 min to form the antibody-toxin (saporin) conjugate, which was then added to cells and incubated for 96 h at $37^{\circ}\mathrm{C}$ in $5\%$ $\mathrm{CO}_{2}$ . Cell viability was then determined by CCK-8 assay (Dojindo) according to manufacturer's recommendations using the Synergy HT microtiter plate reader (Bio-Tek).

The half maximal effective concentration (EC50) values were determined by curve fit using Prism (GraphPad Software).

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

13

# RESULTS

# HCA-based screening strategy

The HCA-based strategy that we used to identify antibodies capable of internalizing into tumor cells via macropinocytosis is outlined in Figure 1A. The key feature is the development of an HCA platform that allows quantitative measurement of colocalization between phage antibodies and a macropinocytic marker, ND70-TR. To identify clinically relevant macropinocytosing antibodies, we screened phage antibody library selection outputs that we have generated previously by laser capture microdissection (LCM)-based selection, which are highly enriched for internalizing antibodies that bind to prostate tumor cells in situ residing in the tumor tissue microenvironment [1].

Analysis of phage antibody binding patterns by automated fluorescence microscopy

Phage infected bacteria were arrayed into 96-well plates and phage-containing supernatants were incubated with prostate cancer DU145 cells in 96-well plates in the presence of complete DMEM/10% FBS for 24h at $37^{\circ}\mathrm{C}$ . Phage antibody binding patterns were analyzed by automated fluorescent microscopy (Figure 1B). A broad range of patterns of cell-associated phages was observed but internalization could not be clearly determined (Supplemental Figure 1).

Image-based quantitation of phage binding was performed to generate a mean fluorescence intensity (MFI) value for each phage antibody (Figure 1C). We selected the top 25% (MFI > 250,000) or 360 phage clones for more detailed analysis of internalizing properties (Figure 1C). FACS analysis of a fraction of these phage clones on DU145 cells yielded MFI values consistent with the microscopic imaging-based analysis (Supplemental Figure 2).

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

14

# HCA identifies phage antibodies that internalize via macropinocytosis

Previous methods to select and screen for internalizing phage antibodies have utilized low pH, high salt washes in an attempt to strip surface-bound phage antibodies. While this approach can be successful, strong binding, high-affinity phage antibodies may be resistant to even these harsh conditions. We tested several of the strongest binding phage antibody clones on fixed cells, which are incapable of internalization, and found that binding was resistant to low pH, high salt washes (Supplemental Figure 3). Thus, new methods are required to identify high affinity internalizing phage antibodies.

To screen for phage antibody clones that internalize into DU145 cells via macropinocytosis, we performed HCA on the strongest binding clones (top $25\%$ , or 360) utilizing ND70-TR as a fluid-phase macropinocytic marker [21, 22]. Previous studies have established that fluorescent high molecular weight dextrans can be used to label macropinosomes [21]. Phage antibody-containing supernatants were co-incubated with ND70-TR over DU145 cells in culture media for 24h at $37^{\circ}\mathrm{C}$ .

Following washing, fixing and permeabilization, cell-associated phage were detected by anti-phage antibody, and subjected to HCA to assess colocalization with ND70-TR (Figure 2D). An initial image analysis revealed that some phage antibodies internalized into cells and colocalized with ND70-TR, primarily in juxtanuclear structures, while other clones exhibited poor colocalization with ND70-TR (Figure 2A, B). Next, we performed a quantitative analysis by measuring the Pearson's correlation coefficient (PCC) between immunolabeled phage and ND70-TR fluorescence.

High PCC values identified phage antibodies that exhibited strong colocalization with ND70-TR, while low PCC values

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

15

identified phage antibodies that exhibited poor colocalization with ND70-TR (Figure 2C).

About $10\%$ , or 36 clones, possessed greater than 2-fold PCC values when compared to controls

(Figure 2E). Following sequencing, 14 unique antibody sequences were identified from the 36 clones.

To rule out that the observed differences in internalization patterns are due to different scFv display levels on phage, we performed Western blot analysis to determ
ine the status of scFv-pIII fusion for six cell-binding phage antibodies with varying PCC values and found similar display levels (Supplemental Figure 4), which are consistent with the multivalent fd phage display format. It thus appears that the different internalization behavior exhibited by various phage antibodies reflects an intrinsic property of the scFv component.

Endocytosed phages macropinocytose en route to lysosomes in DU145 cells

We further characterized three phage antibody clones, named HCA-F1, HCA-M1, and HCA-S1 (sequences shown in Supplemental Table 1), two of which possessed high (HCA-F1 and HCA-M1, $>2$ -fold PCC values over control) and one with low (HCA-S1, $<2$ -fold PCC value over control) correlation between immunolabeled phages and ND70-TR. Using fluorescent confocal microscopy, we investigated whether these clones could internalize into juxtanuclear structures coinciding with lysosomal markers.

After 24h incubation with DU145 cells, phage antibodies colocalized with lysosomal-associated membrane protein 1 (LAMP1). Phages HCA-F1 and HCA-M1 were visible as compact, vesicular structures present in a juxtanuclear area while phage HCA-S1 exhibited poor internalization (Figure 3A). Computed 3D tomography also demonstrated that endocytosed phage HCA-F1 colocalized with internalized ND70-TR

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

16

(Supplemental Movie 1). We also examined whether phages could be visualized within early endosomes during early stages of endocytosis, however, phages did not colocalize with the endosomal marker, early endosomal antigen 1 (EEA1) (data not shown), suggesting that either the phages transited quickly through early endosomes or bypassed the early endosomes en route to lysosomes.

# Phage macropinocytose into DU145 cells with varying kinetics

We next examined whether phage antibodies HCA-F1, HCA-M1, and HCA-S1 can exhibit distinguishable internalization kinetics. Whereas two phage antibodies HCA-F1 and HCA-M1 displayed a similar internalization pattern after a 24h incubation, only phage HCA-F1 was capable of internalizing into DU145 cells after an 8h incubation (Figure 3B). PCC analysis between fluorescently immunolabeled, internalized phages and ND70-TR after an 8h incubation showed significant differences between the three phage antibodies (Figure 3C). Mander's correlation coefficient analysis, which is similar to PCC analysis but places weight on fluorescent intensity, also corroborated these differences (data not shown).

# Internalization of IgGs derived from scFvs

We cloned scFv from phages HCA-F1, HCA-M1, and HCA-S1 into full-length human IgG1 expression constructs and purified IgGs from transiently transfected human embryonic kidney (HEK) 293A cell supernatants. The purified IgGs HCA-F1, HCA-M1, and HCA-S1 demonstrated binding to DU145 cells via flow cytometry (Supplemental Figure 5) and

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

17

colocalized with internalized ND70-TR in DU145 cells in a similar fashion to their parental phage antibodies (Figure 4A). 3D computed tomography showed that IgG HCA-F1 possesses the most robust internalization properties, internalizing almost immediately upon incubation with cells and yielding very low amounts of detectable IgG on the surface of the cell after 90 minutes of incubation (Supplemental Movie 2). Similar to the data from the phage experiments, the PCC value between immunolabeled IgG and ND70-TR was significantly higher for IgG HCA-F1 than either IgG HCA-M1 or IgG HCA-S1 (Figure 4B).

We next utilized immunolabeling against the endocytic markers EEA1 and LAMP1 to examine the colocalization of IgGs HCA-F1, HCA-M1, and HCA-S1 with early endosomes and lysosomes over varying time intervals. All of the IgGs bound to the surface of DU145 cells almost immediately after administration (Figure 5A-B). IgG HCA-F1 fluorescence increased in intensity over time in punctate-like structures at the expense of cell surface fluorescence (Figure 5A-B). IgG HCA-F1 addition also led to increased numbers of EEA1-labeled punctate structures when compared to either IgGs HCA-M1 or -S1 (Figure 5A).

Antibody colocalization with both organelles was quantitated via PCC analysis across all time points. IgG HCA-F1 possessed significantly higher PCC values at earlier time points for both EEA1 and LAMP1 when compared to either IgG HCA-M1 or HCA-S1 (Figure C-D). IgG HCA-F1 did not significantly colocalize with caveolin-2 or clathrin heavy chain, especially at earlier time points (Supplemental Figure 6). Furthermore, the HCA-F1 scFv-Fc fusion also bound, internalized, and colocalized with both EEA1 and LAMP1 within DU145 cells in the same fashion as its IgG counterpart (Supplemental Figure 7).

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

18

# IgG HCA-F1 internalizes via macropinocytosis

To confirm antibody internalization via macropinocytosis, we studied antibody internalization with and without inhibitors of macropinocytosis. Previous studies have demonstrated that cytochalasin D and ethylisopropylamiloride (EIPA) both inhibit macropinocytosis [9, 23-25]. DU145 cells pre-treated with cytochalasin D, EIPA, or DMSO for $30\mathrm{min}$ were chased with IgG HCA-F1 in the presence of drug or DMSO. Both cytochalasin D and EIPA significantly inhibited IgG HCA-F1 internalization into DU145 cells (Figure 6A). Measurements of internalized, immunolabeled IgG HCA-F1 fluorescence showed that both cytochalasin D and EIPA decreased endocytosed IgG HCA-F1 by $>50\%$ when compared to DMSO control (Figure 6B).

# EphA2 identified as antigen target for macropinocytosing IgG HCA-F1

We next sought to determine the target antigen bound by the rapidly internalizing macropinocytosing IgG HCA-F1. We surface-biotinylated DU145 cells, prepared cell lysates and performed immunoprecipitation with HCA-F1 scFv-Fc immobilized to agarose beads. Immunoprecipitation products underwent parallel SDS-PAGE and immunoblotting. Immunoblotting results with streptavidin-conjugated horseradish peroxidase (HRP) showed a dominant band at $\sim 110$ kDa (Figure 7A).

After excising the corresponding band from the Coomassie-stained gel, the extracted protein gel slice underwent trypsin-digestion and analysis via tandem mass spectrometry. The results identified a transmembrane protein, ephrin type-A receptor 2 (EphA2), as the target antigen (Supplemental Table 2. Peptide identification is shown in Supplemental Table 3). For an independent verification, we ectopically expressed

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

19

human EphA2 cDNA in Chinese hamster ovary (CHO) cells and found that IgG HCA-F1 bound strongly to these cells but not CHO cells transfected with a control cDNA (Figure 7B-C).

Neither IgG HCA-M1 nor -S1 binds to EphA2 as they did not bind to cells expressing EphA2 such as benign prostatic hyperplasia (BPH-1) cells that IgG HCA-F1 bound (Supplemental Figure 8). We screened our current cell-binding phage antibody collection and found no additional EphA2 binders within this pool (Supplemental Figure 9). Nonetheless additional anti-EphA2 antibodies could still exist in the LCM selection output, which could be uncovered by expanding the scope of the screening in the future.

Receptor-dependent macropinocytosis of the anti-EphA2 IgG

As EphA2 is widely overexpressed by cancer cells [26, 27], we next examined whether IgG HCA-F1 is capable of binding to other cancer cell lines and internalizing via macropinocytosis. We analyzed the binding of IgG HCA-F1 to five human cancer cell lines (prostate cancer DU145, breast cancer MDA-MB-231, lung cancer A549, cervical cancer HeLa, epidermoid carcinoma A431) and two non-cancer cell lines (Hs27 and BPH-1) by FACS. IgG HCA-F1 binding was higher for all five cancer cell lines when compared to the non-cancer cell lines (Supplemental Figure 10).
IgG HCA-F1 did not bind to the LNCaP line that does not express EphA2 (Figure 8A), demonstrating the receptor-dependent nature of this type of cell entry. To assess binding to cross-species epitopes, we also performed FACS analysis of IgG HCA-F1 on a mouse melanoma cell line B16F10 and observed binding, which suggests that IgG HCA-F1 bind to an EphA2 epitope conserved across species (data not shown). To investigate the specificity of internalization, we co-incubated IgG HCA-F1 and ND70-TR, over the

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

20

aforementioned panel of both cancer and non-cancer cell lines. Confocal imaging using equal exposure times confirmed that IgG HCA-F1 bound strongly to cancer cell lines when compared to non-cancer cell lines (Supplemental Figure 11A). Internalized IgG HCA-F1 was quantified by measuring mean fluorescent intensities of IgG HCA-F1 within individual, confocal slices of cytosolic areas of cells. Quantitation of internalized IgG HCA-F1 across all cell lines revealed that cancer cell lines possess greater amounts of internalized IgG HCA-F1 when compared to non-cancer cell lines (Supplemental Figure 11B).

# Antibody-toxin conjugate exhibits potent cytotoxicity in vitro

To obtain functional evidence for internalization, we investigated whether an IgG HCA-F1-based antibody-toxin conjugate could lead to targeted killing of tumor cells. We created an IgG HCA-F1-toxin conjugate by first modifying IgG HCA-F1 with amine-reactive biotin, followed by attachment of streptavidin-conjugated saporin, a highly potent ribosome-inactivating protein toxin. Saporin lacks a chain required for cell insertion and is thus non-toxic by itself.

We incubated the antibody-toxin conjugates at varying concentrations with both DU145 (EphA2 positive) and LNCAP (EphA2-negative) cells and examined cell viability after 4 days. The IgG HCA-F1-toxin conjugate exhibited potent cytotoxicity against DU145 cells $(\mathrm{EC}_{50}$ about $19~\mathrm{pM})$ but not on control LNCaP cells (Figure 8B), demonstrating functionally a receptor-dependent internalization mechanism. Toxin conjugated to a control non-binding human IgG did not kill tumor cells, neither did toxin alone nor naked HCA-F1 IgG.

These studies provide functional evidence for rapid internalization by our anti-EphA2 antibody IgG HCA-F1 and demonstrate potential for the development of targeted therapeutics against EphA2-positive tumors.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

21

# DISCUSSION

Recent studies suggest that macropinocytosis is a rapid and efficient cellular internalization pathway that is upregulated selectively by tumor cells [9, 28]. Exploring this pathway for targeted therapy development has the potential of improving potency and selectivity for tumor targeting agents. While studies have been done previously to identify internalizing antibodies from phage antibody display libraries [13-15, 29, 30], no method has been developed to identify macropinocytosing antibodies.

In this study, we developed an HCA-based high throughput method to identify macropinocytosing antibodies from phage antibody display libraries. Following conversion into full-length human IgG1s, we determined by confocal microscopy that one of the antibodies, IgG HCA-F1, rapidly internalizes via macropinocytosis and colocalizes with early endosome and lysosome markers. The microscopic internalization studies were confirmed by functional internalization assays based on the plant toxin saporin that lacks an internalization mechanism on its own.

The rapid internalization of the HCA-F1 IgG resulted in potent cytotoxicity of antibody-toxin conjugate against a broad panel of tumor cells expressing the target antigen, demonstrating functionally that this antibody is efficiently internalized by target cells.

Previous methods to select and screen for internalizing phage antibodies have utilized low pH, high salt wash buffers in an attempt to strip away surface-bound phage antibodies [14, 15, 29]. While this approach has been at least partially successful, strong binding high affinity phage antibodies may be resistant to even these harsh conditions. Indeed, when we tested strong binding phage antibodies on fixed cells which are incapable of internalization, we found that binding was resistant to low pH, high salt washes. In addition, we found that analysis of patterns of cell-associated phage that were generated by non-confocal HCA instruments was not

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

22

sufficient to determine if the phage is internalized. Many heterogeneous patterns were observed, and it was difficult to reliably associate any of the patterns with internalization, let alone macropinocytosis. Thus, our new methods based on multi-marker microscopic HCA establish an effective means for the identification of internalizing and furthermore macropinocytosing antibodies from phage display libraries.

Our studies showed that there are major differences in internalization kinetics between an antibody in soluble form and on phage, which must be taken into consideration for screening design. For example, when tested in full-length IgG or scFv-Fc fusion forms, the highly active macropinocytosing antibody HCA-F1 starts internalization almost immediately and completes the process in 40-80 min, while the same antibody in phage format does so in 8h.

The large size of the phage particle may have slowed down the internalization process considerably regardless of how rapidly the antibody internalizes in soluble forms. In addition, although in soluble forms different antibodies utilized disparate internalization pathways, in phage forms they seem to converge into the macropinocytosis pathway. This is not entirely surprising considering the size of the phage particle.

Nonetheless, despite the generally retarded rate and the near uniform route of phage particle internalization (phage macropinocytosis), the kinetic differences in phage antibody internalization are a function of the underlying scFv, with HCA-F1-like phage internalizing in 8h, HCA-M1-like in 24h and HCA-S1-like $>24\mathrm{h}$ . These kinetics differences allowed us to develop screening schemes to uncover rapidly internalizing antibodies such as the macropinocytosing antibody HCA-F1.

In this context, we would like to re-emphasize that HCA screening using phage directly is convenient and compatible with the high throughput format, but the result must be verified using antibodies in soluble forms. Either scFv-Fc fusions or full IgGs can be used for secondary screening or validation study as we have not found significant

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

23

differences between these two forms of soluble antibodies with regard to internalization patterns. We normally produce and test the scFv-Fc fusion molecules first due to ease in cloning and production, followed by conversion of scFvs into human IgG molecules for bioassays.

Another peculiar feature of phage internalization is revealed by our organelle-labeling experiment. While phage antibodies are seen to colocalize to lysosomes, they could not be readily seen entering cells via the early endosomal pathway. We reconciled this observation by hypothesizing that large, macropinocytosed phage particles may be trafficking via endosomes distinct from traditional coated vesicle-formed endosomes, which has been previously observed [8]. For IgG however, we were able to observe that the phage-derived IgG could internalize via macropinocytosis towards the lysosomal organelles via the endosomal pathway.

EphA2 is known to be expressed by various tumor cells and play roles in tumor invasion and metastasis [26]. EphA2 also interacts with Kaposi's sarcoma-associated herpesvirus and regulates viral entry into endothelial cells by macropinocytosis [31]. Several groups have
developed anti-EphA2 antibodies [32-34], and it appears that different epitopes mediate different rates of internalization [32, 35]. No phage antibody library selection scheme has been developed previously that allows for selection of macropinocytosing antibodies binding to EphA2 or other antigens. Our unbiased screening has uncovered an antibody that binds to EphA2 and is rapidly internalized by the macropinocytosis pathway, thereby creating novel agents against this receptor.

EphA2 has been the target for many forms of cancer therapy development. Nanoparticles conjugated with anti-EphA2 antibodies have been used for siRNA delivery [36, 37]. In addition, an anti-EphA2 antibody auristatin E conjugate was tested in a phase I trial for solid tumor treatment [33, 38]. This particular anti-EphA2 antibody-auristatin E-conjugate showed

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

24

unacceptably high toxicity at sub-therapeutic doses [38]. Given that different EphA2 epitopes distinctly influence the kinetics and pathway of internalization, it is possible that the aforementioned setback with the anti-EphA2 antibody-auristatin E-conjugate is an isolated phenomenon relating to the particular antibody used. In any event, anti-EphA2 antibodies can be utilized to deliver payloads other than auristatin. As such there still could be further development of an anti-EphA2 antibody-based therapeutic in the future.

Our anti-EphA2 antibody is internalized by the tumor selective macropinocytosis pathway. It remains to be determined if our antibody has a different potency/toxicity profile than those previously reported anti-EphA2 antibodies. Given that the macropinocytosing epitope bound by our HCA-F1 antibody is conserved across species, any targeted therapeutics developed from this antibody can be tested in small rodents to obtain meaningful toxicology profiles. Finally, we have only screened a limited fraction of our LCM-selection output and identified one anti-EphA2 binder.

Work is underway to expand the scope of the screening to identify additional macropinocytosing anti-EphA2 antibodies.

We have previously developed an LCM-based selection strategy to enrich for phage antibodies binding to tumor cells in situ residing in their tissue microenvironment as opposed to cell line artifacts [1]. In this report we further screened the LCM selection output using our HCA-based method and identified novel macropinocytosing human antibodies targeting clinically relevant tumor antigens.

Integrating LCM and HCA into phage antibody display library selection thus allows identification of novel antibodies that target true tumor antigens expressed by tumor cells residing in their tissue microenvironment and enter target cells via tumor selective pathways such as macropinocytosis. Targeted therapeutics based on these novel antibodies have the potential to improve potency in tumor killing and reduce toxicity on normal

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

25

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

tissues, thus widening the therapeutic window and improving effectiveness of such antibody-targeted therapeutics.

26

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

# Acknowledgement:

We thank Drs. Christopher R. Behrens, NamKyung Lee and Daniel Sherbenou for helpful discussions. We thank the National Institutes of Health for financial support (R01 CA118919, R01 CA129491 and R01 CA171315) and the Center for Mass Spectrometry and Proteomics at the University of Minnesota for mass spectrometry analysis.

27

# FIGURE legends

FIGURE 1 Outline of screening strategy and data from the first step of the screening, i.e., phage binding to DU145 cells. A) Schematic of HCA screening to identify macropinocytosis-dependent antibodies. HCA instruments allow automated high throughput detection of antibody colocalization with a macropinocytosis marker. The starting materials for the screening are sublibraries generated previously by us from LCM-based phage antibody library selection [1] that are enriched for internalizing phage antibodies binding to tumor cells in situ.

B) DU145 cells were incubated in 96-well plates with phage-containing supernatants for 24 hours at $37^{\circ}\mathrm{C}$ in complete DMEM/10% FBS. Nuclei were stained with Hoechst 33342. Bound phages were immunolabeled with anti-fd antibodies (green). Zoomed insert portrays software-based, automated cell analysis, measuring mean fluorescence intensities (MFI) of immunolabeled phages. Over 300 cells were quantified for each phage clone. C) Plot of MFI values of immunolabeled phage binding to cell for 1,439 phage clones.

Red horizontal line represents MFI of $\sim 250,000$ , the threshold for prioritizing clones for further internalization analysis.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

28

Figure 2 Colocalization of phage antibodies with the macropinocytosis marker ND70-TR. A) Epifluorescent images of DU145 cells that were incubated with phage-containing supernatants and $50~\mu \mathrm{g / ml}$ ND70-TR (red) for $24\mathrm{h}$ at $37^{\circ}\mathrm{C}$ . Cell-associated phage were then detected by biotin-labeled anti-fd antibody followed by streptavidin-AlexaFluor 488 (green). Colocalization results in color overlap (yellow).

B) To analyze colocalization, arbitrary lines were drawn across cells and fluorescent intensities along the drawn line were plotted for phages (green) and ND70-TR fluorescence (red). Co-variation of line intensity indicates colocalization. Representative images of two different phage antibodies with differing colocalization patterns are shown. C) Pearson's correlation coefficient (PCC) was quantified and averaged from $>30$ cells per phage conditions.

Error bars denote SEM for $n = 3$ ; * and ** indicate P-values of $< 0.05$ and $< 0.01$ , respectively, using two-tailed student's T-tests assuming unequal variance. Scale bar denotes 20 $\mu$ m. D) Colocalization screening. DU145 cells were plated onto 96-well plates and incubated with phages and ND70-TR (red) for $24\mathrm{h}$ at $37^{\circ}\mathrm{C}$ . Cells were immunolabeled against bacteriophages (green) and nuclei were stained with Hoechst 33342 (blue). E) Mean PCC between immunolabeled phage and ND70-TR of 360 phage clones, quantified from minimum of 300 cells per phage clone.

PCC values were normalized to control phage clones that exhibited poor internalization. Green horizontal line represents $200\%$ of control, a threshold for further analysis.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

29

Figure 3 Confocal analysis of phage antibody internalization by DU145 cells. Confocal Z-slices of DU145 cells incubated with purified phage for A) $24\mathrm{h}$ at $37^{\circ}\mathrm{C}$ or B) $8\mathrm{h}$ at $37^{\circ}\mathrm{C}$ in the presence of ND70-TR. Cells were immunolabeled against phages (green), lysosomes (LAMP1, red), and nuclei (Hoechst 33342, blue). Scale bar: $20~\mu \mathrm{m}$ . C) Mean PCC of internalized phages and ND70-TR. Over 30 cells were analyzed per phage antibody. ** denotes two-tailed t-test P-values of $<0.01$ . Error bars represent SEM for $n = 3$ .

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

30

Figure 4 Internalization and colocalization analysis of IgGs derived from scFvs. A) DU145 cells co-incubated with three IgGs with different internalization properties at $10~\mu \mathrm{g / ml}$ and 50 $\mu \mathrm{g / ml}$ ND70-TR (red) for $90\mathrm{min}$ at $37^{\circ}C$ . Cells were immunolabeled against IgG using anti-human Fc (green). Nuclei were stained with Hoechst 33342 (blue). Single confocal Z-slice images are shown. Scale bar: $20~\mu \mathrm{m}$ . B) PCC analysis of colocalization of IgGs HCA-F1, HCA-M1, and HCA-S1 with ND70-TR using Z-slices crossing the entire cell, quantit
ating a minimum of 10 cells. ** and *** denote two-tailed t-test P-values of $< 0.01$ and $< 0.001$ respectively. Error bars represent SEM for $\mathfrak{n} = 3$

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

31

Figure 5 Kinetics of antibody internalization and subcellular localization. DU145 cells were incubated with three different IgGs (HCA-F1, HCA-M1, or HCA-S1) at $10~\mu \mathrm{g / ml}$ for $15\mathrm{min}$ at $4^{\circ}\mathrm{C}$ and then chased with complete DMEM/ $10\%$ FBS for indicated time periods. Cells were then fixed, permeabilized and immunolabeled against human IgG (green) and A) early endosomes (EEA1, red) or B) lysosomes (LAMP1, red). Nuclei were stained with Hoechst 33342 (blue). Scale bar: $20~\mu \mathrm{m}$ . Pearson's correlation coefficients between immunolabeled C) EEA1 or D) LAMP1 and immunolabeled IgG were averaged from a minimum of 30 cells. Error bars denote SEM of $n = 3$ .

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

32

Figure 6 Macropinocytosis inhibitors prevent internalization of IgG HCA-F1. DU145 cells were pre-treated with $50~\mu \mathrm{g / ml}$ cytochalasin D, $7.5~\mu \mathrm{g / ml}$ EIPA, or DMSO (control) for $30\mathrm{min}$ at $37^{\circ}C$ followed by co-incubation with $10~\mu \mathrm{g / ml}$ IgG HCA-F1 and ND70-TR (red) in the presence of cytochalasin D, EIPA, or DMSO in complete DMEM/ $10\%$ FBS for $40\mathrm{min}$ at $37^{\circ}C$ . Cells were then immunolabeled for human IgG (green). Nuclei were stained with Hoechst 33342 (blue). A) Individual confocal Z-slices of representative cells.

CytoD: cytochalasin D. Scale bar: $20~\mu \mathrm{m}$ . B) The percentage of internalized IgG HCA-F1 was quantitated by measuring the ratio of internalized, cytosolic IgG HCA-F1 fluorescence over total cell IgG HCA-F1 fluorescence, analyzing $>15$ cells over 3 independent experiments. CytoD: cytochalasin D. *** indicates P-value of $<0.001$ using two-tailed student's T-test assuming unequal variance. Error bars represent SEM with $n = 3$ .

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

33

Figure 7 EphA2 identified as target antigen bound by macropinocytosing antibody IgG HCA-F1. A) Immunoprecipitation of the target antigen from surface-biotinylated Du145 whole cell lysates using scFv HCA-F1-Fc fusion immobilized onto a solid matrix. The immunoprecipitation product was run on SDS-PAGE and subjected to Western blot analysis using streptavidin-HRP to locate the position of membrane proteins. The dominant band, denoted by “*”, represents the approximate region from which the corresponding SDS-PAGE gel was extracted for mass spectrometry analysis.

B) Binding to ectopically expressed EphA2. Chinese hamster ovarian (CHO) cells were co-transfected with pEGFP-N2 (to label transfected cells) and pCMV6 expression constructs bearing either human EphA2 or Lgr5 (control). Cells were then incubated with IgG HCA-F1, followed by immunolabeling using anti-human Fc AlexaFluor® 647. Cells were gated for GFP expression and plotted for AlexaFluor® 647 fluorescence (FL4). C) Plot of MFI values as analyzed by FACS. IgG HCA-F1 binds specifically to ectopically expressed EphA2, confirming the target identification.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

34

Figure 8 Functional internalization assay using IgG HCA-F1-toxin conjugates. A) FACS analysis showing EphA2-positive (DU145) and EphA2-negative (LNCaP, control) cells. IgG HCA-F1 was incubated with the cells and binding was detected with anti-human Fc. MFI values are shown in the far right panel. B) IgG HCA-F1 was conjugated to saporin and incubated with target (DU145) and control (LNCaP) cells. Controls: toxin only and IgG HCA-F1 only. Cell viability was measured 4 days later using the CCK-8 assay.

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

35

# References:

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

36

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

37

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

38

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

39

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

40

Running title: HCA screen identifies tumor-targeting macropinocytosing mAbs

41

![](dt=2026-05-07/ht=10/b7a5d81336493668287f4f9b45de2505d378434ca66df1fa72ed3343bbe0c313.jpg)

![](dt=2026-05-07/ht=10/63b2f92bd01dcd41f7db062424e8b3a98bddc7d644c7874ce0a3e6b589f5e083.jpg)

![](dt=2026-05-07/ht=10/0704eb446f2ee2c611d09dfb42b727f5a9236b58c0f938bc864fda7aeb27ffa1.jpg)

![](dt=2026-05-07/ht=10/1e3063b8de817caba1be9a69e8ad98363b06b896af7e7f1b0a17b245a71f3134.jpg)

![](dt=2026-05-07/ht=10/91cb8ed3b96911588197f5b3dc23dae37697af6517a58e5ef09697ce8295b9ef.jpg)

![](dt=2026-05-07/ht=10/a293cd87ce6a9606e2ae22340e69cbd37d1a92f8bcf4d2f19c7eb9ff8cbecd62.jpg)

![](dt=2026-05-07/ht=10/ad2263281aae7d37b69661c3e6d8d223b4e6ebf48875858705596afc3bf7c22a.jpg)

![](dt=2026-05-07/ht=10/81d906afb392c57786f40f367e95c5dc45ce3083d18521e806353a50fba3e969.jpg)

![](dt=2026-05-07/ht=10/c144a9eeb57230de21fc7752ad598313233bb69c0b11eea47640975b5067639a.jpg)

![](dt=2026-05-07/ht=10/63c66f3dea5de376ec9764203806552faffb29b8aefc93275702d8a221aed6af.jpg)

![](dt=2026-05-07/ht=10/78813f0390e95aeb021e7854d733d00306526d5e8a8a4541f4c0026a37a9b128.jpg)

![](dt=2026-05-07/ht=10/b49426672c92d5e7f718483d93d084e28cf7201cc5534db1e01ef046ce9059a2.jpg)

![](dt=2026-05-07/ht=10/5980f1de625da4ca0d20ba8375e1af9026ec702d8d762126fcc04236c8d8a616.jpg)

![](dt=2026-05-07/ht=10/9d642dcb5ac0f118c2dfdad5a4fbbbe168ff1211277a63dd2b6b93014328c789.jpg)

![](dt=2026-05-07/ht=10/acc2d14dfeb4bf542929052d9a5677a31b883996c00ba0528a6fd86959f20857.jpg)

![](dt=2026-05-07/ht=10/c64dc8bfb0cca47aeb4a9f8c5b1f1f1753507af500d67e3543e67fd567f2eb3b.jpg)

![](dt=2026-05-07/ht=10/f80d981ae19134cd5a5a622e2509dca1824a99a746f7ef8f79a871f61589280c.jpg)

![](dt=2026-05-07/ht=10/3f166720d3fd7a538d677c8ded1ead9a99b9873f23ebc3b1873ceedc9f320aeb.jpg)

![](dt=2026-05-07/ht=10/f387ca0a6541ef84ca97554b058bc6e93f6c2733207590c52a447787dfab1e92.jpg)

![](dt=2026-05-07/ht=10/d600770f16dce9456ca81240a4d9485f1d2f02462dabf48cbe37ece1d005ca08.jpg)

![](image)
://lakehouse2/hive-ha/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-05-07/ht=10//5e3d689551714944348d31d22f3d9fbb6832ad10baf767afff78d556da8cb597.jpg)

![](dt=2026-05-07/ht=10/bfa7d3c22ce83ac65a6a935c4313d17ff9c66a4290b4c3da41e3b08d98d2a327.jpg)

![](dt=2026-05-07/ht=10/f533cf33a29502795837f740383b608c9976faf6eb8acc6b83081b7f06d92b3f.jpg)

![](dt=2026-05-07/ht=10/6e1899ecfdff2d107d6c40e70c610f1ff13e9de326386bc157615d6bea55e3bf.jpg)

![](dt=2026-05-07/ht=10/16000b20d4463c3d1d0c257279896f71ed3ba9879aafc0e91502b23c967b3e24.jpg)

![](dt=2026-05-07/ht=10/58b0e13929fde0bc6f85ae72a15b3ad960b9426c59339f7f8438ab1748d6b90d.jpg)

![](dt=2026-05-07/ht=10/e777571ee3492a43ef08e6a9efe6751900930752b09d12e8c0f4156bbb33de91.jpg)

![](dt=2026-05-07/ht=10/cdb7bbe0742317fb7ed406fdb0421b6bbebdc2c3b8946c7ea3347336274b7ad6.jpg)

![](dt=2026-05-07/ht=10/28af2b65684820f663b06af9ea4b99a63fe6f4efc5fedd3746331ea4f75dc877.jpg)

![](dt=2026-05-07/ht=10/5ce8fa3d20869736525943e8c9dde21eff8ae1c6bb29846d492036112780abb1.jpg)

![](dt=2026-05-07/ht=10/332a0bbc05c49577526c237a03710b6e2382babf1007876fff28cd50814a5521.jpg)

![](dt=2026-05-07/ht=10/35f112fec2085dd86e5863c966093ecfdbc16fee59dc8f6f42db57c408fba565.jpg)