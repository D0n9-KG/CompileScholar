# Insightful directed evolution of Escherichia coli quorum sensing promoter region of the IsrACDBFG operon: a tool for synthetic biology systems and protein expression

Pricila Hauk $^{1,2}$ , Kristina Stephens $^{1,2}$ , Ryan Mckay $^{1,2}$ , Chelsea Ryan Virgile $^{1,2}$ , Hana Ueda $^{3}$ , Marc Ostermeier $^{4}$ , Kyoung-Seok Ryu $^{5}$ , Herman O. Sintim $^{6}$ and William E. Bentley $^{1,2,*}$

$^{1}$ Institute for Bioscience and Biotechnology Research, College Park, MD, USA, $^{2}$ Fischell Department of Bioengineering, University of Maryland, College Park, MD, USA, $^{3}$ Department of Mathematics, University of Maryland, College Park, MD 20742, USA, $^{4}$ Department of Chemical and Biomolecular Engineering, The Johns Hopkins University, 3400 North Charles Street, Baltimore, MD 21218, USA, $^{5}$ Protein Structure Group, Korea Basic Science Institute, 162 Yeongudangi-Ro, Ochang-Eup, Cheongju-Si, Chungcheongbuk-Do 363-883, South Korea and $^{6}$ Department of Chemistry and Biochemistry, University of Maryland, College Park, Maryland 20742, USA

Received August 11, 2016; Revised October 10, 2016; Editorial Decision October 11, 2016; Accepted October 18, 2016

# ABSTRACT

Quorum sensing (QS) regulates many natural phenotypes (e.q. virulence, biofilm formation, antibiotic resistance), and its components, when incorporated into synthetic genetic circuits, enable user-directed phenotypes. We created a library of Escherichia coli lsr operon promoters using error-prone PCR (ePCR) and selected for promoters that provided E. coli with higher tetracycline resistance over the native promoter when placed upstream of the tet(C) gene. Among the fourteen clones identified, we found several mutations in the binding sites of QS repressor, LsrR.

Using site-directed mutagenesis we restored all p-IsrR-box sites to the native sequence in order to maintain LsrR repression of the promoter, preserving the other mutations for analysis. Two promoter variants, EP01rec and EP14rec, were discovered exhibiting enhanced protein expression. In turn, these variants retained their ability to exhibit the LsrR-mediated QS switching activity. Their sequences suggest regulatory linkage between CytR (CRP repressor) and LsrR. These promoters improve upon the native system and exhibit advantages over synthetic QS promoters previously reported.

Incorporation of these promoters will facilitate future applications of QS-regulation in synthetic biology and metabolic engineering.

# INTRODUCTION

Quorum sensing is a process of cell-cell communication that allows bacteria to enumerate their cell density and modify their behaviors (such as virulence, biofilm formation and antibiotic resistance) in a collective manner. To do this, bacteria 'talk' to each other using small molecules called autoinducers (1). Autoinducer-2 (AI-2) has attracted significant attention because its terminal synthase (LuxS) and its signal transduction cascade (Lsr regulon) are widely conserved among Eubacteria (2-4).

For example, the Enterobacteriaceae, Escherichia coli and Salmonella enterica serovar Typhimurium share similarities involving detection and production of AI-2, as they possess homologous machinery to control quorum sensing based on AI-2 levels. The bidirectional quorum-sensing $E.$ coli lsr regulon, $lslRK$ , $lslrACDBFG$ is responsible for controlling the expression of genes involved in the perception and degradation of AI-2 (5). The lsr operon (lserACDBFG) coordinates the expression of genes involved in the transport and degradation of AI-2.

The four genes $lsrACDB$ produce the ABC transporter components, which are responsible for AI-2 uptake (5-7); the $lsrFG$ genes are responsible for the degradation of its active form, phospho-AI-2 (8,9), and its subsequent assimilation into central carbon metabolism (8-10). In Salmonella Typhimurium, there is an additional $lsrE$ gene, which encodes a putative sugar epimerase (11). In both $E.$ coli and Salmonella, $lsrK$ and $lsrR$ serve to coordinate the induction and repression of the $lsr$ operon (8,11).

LsrR represses transcription of the operon and itself by directly binding to two LsrR binding 'boxes' within the $lsr$ promoter region (4). LsrR is released in the presence of

Published online 24 October 2016

Nucleic Acids Research, 2016, Vol. 44, No. 21 10515-10525

doi: 10.1093/nar/gkw981

*To whom correspondence should be addressed. Tel: +1 301 405 4321; Fax: +1 301 405 9953; Email: bentley@umd.edu

© The Author(s) 2016. Published by Oxford University Press on behalf of Nucleic Acids Research.

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact journals.permissions@oup.com

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

phospho-AI-2, which, in turn, is the phosphorylated product of the LsrK kinase (4,8) and AI-2.

Because they mediate communication among various bacteria and their genetic circuitry is relatively well understood, QS-based circuits have been engineered for use in widely varied application areas: biochemicals production, sensor development, infectious disease, tissue engineering, and mixed-species fermentations (12-21). Also, both AHL (N-acyl-homoserine lactones) and AI-2 based species communication systems have been developed as tools for exogenously controlling bacterial phenotype and protein expression (22,23).

For example, the luxCDABE operon of the bioluminescent bacterium Photorhabdus luminescens, based on AHL, has proven to be an exceptional transcriptional reporter for expression in high-GC bacteria (12). AHL-actuated promoters were incorporated into many synthetic biology strategies for guiding cell behaviors (24-33), including an elaborate circuit that enables the sensing, tracking, and targeting of Pseudomonas aeruginosa, an important human pathogen (20,34).

The native $lsr$ quorum sensing regulon, on the other hand, has received comparatively less attention, even though it is the native system of $E.$ coli, which could to be viewed as one of the 'workhorses' of microbial synthetic biology. The $lsr$ promoter operon was rewired by Tsao and colleagues (2010) (13) to act as an autonomous inducer for the expression of recombinant proteins. In their two-plasmid system, the $E.

$ coli $lsrACDBFG$ promoter region served as a trigger of T7 RNA polymerase expression, which, in turn 'amplified' target protein expression from commercially available pET vectors (13). Due to the fact that the $lsrACDBFG$ operon promoter in $E.$ coli [-307 to +92 relative to the start codon of $lsrA$ ] is known to be very weak (13), we created a library of $lsr$ operon promoters through directed evolution using the error-prone PCR (ePCR). Our objective was to discover promoter sequences that were superior to the native system that also required no signal amplification (via T7 polymerase).

For this, we constructed a plasmid (pLSR) for the expression of two gene reporters, gfp and tet(C) under $lsr$ regulon control (in the direction of the $lsrACDBFG$ operon as opposed to $lsrRK$ ). Our mutagenic library yielded two mutant $lsr$ promoters, EP01rec and EP14rec, that demonstrated greater strength than the wild type $lsr$ promoter. Sequencing and subsequent expression analyses revealed mutations responsible for the increase in promoter strength. We also identified what is believed to be a CytR binding site within the $lsr$ operon promoter sequence.

Importantly, evolved $lsr$ promoters (EP01rec and EP14rec) retain the same properties of the wild type $lsr$ promoter: induction via AI-2 and repression by LsrR.

# MATERIALS AND METHODS

# Strains and media

NEB turbo electrocompetent $E.$ coli, $\mathrm{DH5\alpha}$ (NEB), LW6, and LW7 (6) strains (Supplementary T
able S1) were grown in Luria-Bertani (LB) medium at $37^{\circ}\mathrm{C}$ for DNA manipulation or expression experiments. Media were supplemented with chloramphenicol $(34~\mathrm{ug / ml})$ to maintain the pLSR plasmid and/or tetracycline (2.5, 5, 10 or $20~\mathrm{ug / ml})$ to select

the $lsr$ operon promoter mutant candidates from the library. Miller assay experiments were performed using $50~\mathrm{ug / ml}$ ampicillin to maintain the LW7 (6) strain transformed with pLW11 (6), pPH01 or pPH14 (Supplementary Table S1).

# Plasmid and library creation

pTS40 is a plasmid that carries the CloDF13 replication origin (20-40 copies/cell) and under the control of the ampC promoter expresses bicistronic, gfp_mut2 and tet (C). This plasmid confers chloramphenicol antibiotic resistance from pTS1 (35). In order to remove the ampC promoter and ampR gene from this plasmid and replace with the E. coli lsrACDBFG operon promoter region, pTS40 was digested using the restriction site PvuI present in the ampR sequence to linearize the plasmid.

Primers pTS40delampR_F and pTS40delampR_R (Supplementary Table S2) were used in a PCR to exclude a sequence fragment containing both the ampR gene and ampC promoter and also to insert at the restriction sites PvuI and SpeI. The 399-bp E. coli lsrACDBFG operon promoter region $[-307$ to $+92$ relative to the start codon of $lSrA]$ (6) was amplified with the primers lsrP_PvuI and lsrP_R_SpeI (Supplementary Table S2) and cloned into the pTS40 backbone containing the same restriction sites inserted through PCR.

This final plasmid (pLSR) was used as a template to generate the $lSr$ operon promoter mutant library (Supplementary Table S1).

Error-Prone PCR (ePCR) was employed to obtain the mutant library containing a vast diversity of $lsr$ promoter mutants. For this, the pLSR plasmid harboring the wild type $lsr$ operon promoter was used as template, and also two oligonucleotides LsrEP_F and LsrEP_R (Supplementary Table S2) flanked by PvuI and SpeI restriction sites, respectively, were used as forward and reverse primers. The conditions to perform EP PCR were performed according to a previously published protocol with some modifications (36).

In this study, three reactions of $50\mu \mathrm{l}$ reaction mixture contained $5\mu \mathrm{l}$ of 10X PCR buffer -Mg, $0.8\mathrm{MnCl}_2$ , $5\mathrm{mM}\mathrm{MgCl}_2$ , $1\mathrm{mM}$ dATP, $1\mathrm{mM}$ dTTP, $0.2\mathrm{mM}$ dCTP, $0.2\mathrm{mM}$ dGTP, $10\mathrm{pmol}$ of each primer, $1\mathrm{ng}$ of template plasmid and $2.5\mathrm{U}$ recombinant Taq DNA Polymerase (Life Technologies) were performed to create bias upon sequence amplification.

The ePCR was conducted in a C-1000 Touch Thermal Cycler (Biorad) for 35 cycles consisting of denaturation at $94^{\circ}\mathrm{C}$ for $30\mathrm{s}$ , annealing at $58^{\circ}\mathrm{C}$ for $1\mathrm{min}$ , and extension at $72^{\circ}\mathrm{C}$ for $2\mathrm{min}$ . The ePCR products were digested with PvuI and SpeI and then ligated at $16^{\circ}\mathrm{C}$ overnight with pLSR previously digested with the same restriction enzymes and transformed into library efficiency NEB turbo electrocompetent E. coli (New England BioLabs).

# Library characterization

For selection, $lsr$ operon promoter library plasmid DNA was isolated from an aliquot of NEB turbo electrocompetent $E.$ coli, and $50~\mathrm{ng}$ of purified plasmid was used to transform strain LW6 (Supplementary Table S1). The transformed cells were plated on LB agar containing $34~\mu \mathrm{g / mL}$ chloramphenicol and then recovered en masse using a sweep buffer (LB containing $2\%$ glucose and $15\%$ glycerol) and stored in aliquots at $-80^{\circ}\mathrm{C}$ . These cells were first plated on a

10516

Nucleic Acids Research, 2016, Vol. 44, No. 21

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

$24.5\mathrm{cm}\times 24.5\mathrm{cm}$ LB agar plate containing $34~\mu \mathrm{g / ml}$ chloramphenicol. Then, colonies on this plate were recovered into sweep buffer and plated at a density of $10^{6}$ cells/ml on a $24.5\mathrm{cm}\times 24.5\mathrm{cm}$ LB media plate containing $34~\mu \mathrm{g / mL}$ chloramphenicol and $20~\mathrm{ug / ml}$ of tetracycline. A total of 14 library clones formed from this selection plate were individually recovered into LB liquid media containing $34~\mu \mathrm{g / ml}$ chloramphenicol and incubated in a shaker at $37^{\circ}\mathrm{C}$ . After $16\mathrm{h}$ of incubation, the plasmid DNA from each clone that survived the selection was isolated to be sequenced and analyzed through alignment with the wild type $lslr$ operon promoter as a reference.

# In silico analyses

All of the 14 clones that were able to grow in media containing $20~\mu \mathrm{g / ml}$ of tetracycline were sequenced and aligned using Clustal Omega (37) with the wild-type $lsr$ operon promoter sequence as a reference. To find potential CytR binding sites, we searched for the corresponding consensus sequence (consisting of a right motif and/or left motif and specified by Pedersen and Valentin-Hansen (38) in the Lsr intergenic region using the Biostrings package (39) of the open source software Bioconductor.

Here, the left and right portions of the consensus CytR sequences were compiled into position probability matrices. A position weight matrix (PWM) was generated by the PWM function, which outputs a $4\times 8$ matrix (4 possible nucleotides $\{\mathrm{A},\mathrm{C},\mathrm{G},\mathrm{T}\} \times$ motif base length of 8) with each element in the matrix representing a score relative to the possibility that nucleotide $i$ would be located at position $j$ . The matrix was scaled so that the total maximum score is 1.

The PWM was then applied to the matchPWM function, producing the top scoring sequences from among those in the intergenic region between $lSrR$ and $lSrA$ start codons. This procedure was applied to each, the right and left motifs.

# Restoring p-1srR-box sequence region and putative CytR-binding site

PCR-driven overlap extension protocol (40) was used to restore the original nucleotides into the p-1srR-box (4) that was corrupted through ePCR (on all the fourteen clones selected through $20~\mu \mathrm{g / ml}$ tetracycline). Mutagenic primers (Supplementary Table S3) and 1sr operon promoter flanking primers (lsrpF_PvuI and lsrpR_SpeI) (Supplementary Table S2) were used to generate intermediate overlapping PCR products that were combined to produce a full-length product using flanking primers. These fragments were cloned into pLSR between PvuI and SpeI sites.

We adopted the same protocol used above to restore a mutated nucleotide found in both clones 1 and 14 inside of the putative CytR-binding site (38). For this, we used a specific mutagenic primer pair, mutAG_F1 and mutAG_R1, and mutAG_F14 and mutAG_R14 to generate intermediate overlapping PCR for clones 1 and 14, respectively (Supplementary Table S3). To produce a full-length product we employed the same primers used to amplify 1sr promoter, lsrP_PvuI and lsrP_R_SpeI (Supplementary Table S2). All the full-length products were cloned into pLSR and transformed into E. coli DH5α.

# Promoter strength metric

All fourteen clones containing the LsrR binding-site corrupted by ePCR and their restored counterparts, and also the two putative CytR binding-site sequence restored clones (EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ ) had their in vivo promoter activities determined through experiments to measure tetracycline resistance and GFP fluorescence.

Tetracycline resistance. An $lsrR$ knockout strain (LW6) with the wild type $lsr$ operon promoter, the fourteen $lsr$ operon promoter mutants obtained through $20\mathrm{ug / ml}$ tetracycline from library selection, and those cells with the restored LsrR-binding site were grown in liquid LB media containing $34~\mu \mathrm{g / mL}$ chloramphenicol overnight in a shaker incubator at $37^{\circ}\mathrm{C}$ .

Each
sample was serially diluted in liquid LB media from $10^{-1}$ to $10^{-10}$ and then $3\mu \mathrm{l}$ of each serial dilution was spotted on LB media plates (OmniTray, $86\mathrm{mm} \times 128\mathrm{mm}$ , Nunc) containing $34~\mu \mathrm{g / ml}$ cloramphenicol and 0, 2.5, 5.0, 10 or $20~\mu \mathrm{g / ml}$ of tetracycline. These plates were incubated at $37^{\circ}\mathrm{C}$ for $16\mathrm{h}$ .

GFP fluorescence. Escherichia coli LW6 harboring pLSR wild type or mutants (EP01rec, EP14rec, EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ ) were grown overnight at $37^{\circ}\mathrm{C}$ in $2\mathrm{ml}$ of LB medium supplemented with $34~\mu \mathrm{g / ml}$ of chloramphenicol. Bacterial suspensions were then re-inoculated into $10\mathrm{ml}$ of fresh LB medium with chloramphenicol in order to have initial optical densities $(\mathrm{OD}_{600\mathrm{nm}})$ of 0.05. Cells were allowed to grow at $37^{\circ}\mathrm{C}$ with shaking at $250\mathrm{rpm}$ .

Bacterial cell samples $(200~\mu \mathrm{l}$ , technical triplicate) were collected at optical density $(\mathrm{OD}_{600\mathrm{nm}})\sim 0.5$ and 1.0. Samples were collected by centrifugation $(1000\mathrm{g},5\mathrm{min})$ , washed, resuspended in PBS and kept on ice until flow cytometry and microscopy analysis. Flow cytometry analyses were performed using a FACSCanto II flow cytometer equipped with $488~\mathrm{nm}$ , $633~\mathrm{nm}$ , and $405~\mathrm{nm}$ lasers (BD Biosciences, San Jose, CA, USA) and all flow cytometry data were analyzed with FACSDiva software (BD Biosciences).

Side and forward scatter of bacterial suspensions were determined using semi-log scale SSC/FSC plots with a threshold of 5000. Voltage settings for the SSC, FSC and FITC channels were kept constant for all flow cytometry experiments. Bacterial suspensions were analyzed at a medium flow rate with a maximum of 1000 events per second for $75\mathrm{s}$ and a minimum of 50 000 events. Image-based cytometry analysis were performed using fluorescent microscopy (Olympus U-HGLGPS) with $20\times$ objective and $1800~\mathrm{ms}$ exposure.

Positive cells for GFP fluorescence were compared with negative control (LW6 and wild type strain).

Transcriptional analysis. Cultures inoculated as previously were grown for $3\mathrm{h}$ $(\mathrm{OD}_{600\mathrm{nm}}\sim 0.5)$ , and the total RNA was isolated using Trizol Max Bacterial RNA isolation kit (Ambion, Life technologies) according to the manufacturer's instructions. In addition, RNA samples were treated with Amplification Grade DNase I (SigmaAldrich) to eliminate possible DNA contamination. PCR primer sequences were designed using PrimerQuest Design Tool (IDT) (Supplementary Table S2) and synthesized by IDT. The SensiFAST SYBR Hi-ROX One-Step kit (Bioline) was used for first-strand cDNA synthesis and

Nucleic Acids Research, 2016, Vol. 44, No. 21

10517

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

subsequent real-time PCR following to the manufacturer's instructions.

Real-time PCR conditions were carried out on an Applied Biosystems 7300 Real-Time PCR system using a two-step cycling protocol. Primers were used at a final concentration of $400\mathrm{nM}$ and $25\mathrm{ng}$ of RNA was used as template in each $20 - \mu \mathrm{l}$ reaction. Each reaction was performed in triplicate, with outlying data removed for select samples. 16s rRNA was used as the endogenous housekeeping gene.

To calculate the levels of $gfp$ expression $\Delta C_{\mathrm{T}}$ values were calculated by the following equation: $\Delta C_{\mathrm{T}} = C_{\mathrm{T}}$ Target $C_{\mathrm{T}}$ Reference.The $\Delta \Delta C_{\mathrm{T}}$ value was calculated as $\Delta \Delta C_{\mathrm{T}} =$ $\Delta C_{\mathrm{T,sample}} - \Delta C_{\mathrm{T,Wt}}$ where each $\mathrm{DC_T}$ are represented by the difference between the Target and Reference (16srRNA) values, as above. Also, the relative quantification (RQ) is calculated as $2^{-\Delta \Delta \mathrm{CT}}$ .

Data plotted in figures also include the standard deviation (s) of each RQ, so that $2^{-\Delta \Delta \mathrm{CT} + s}$ and $2^{-\Delta \Delta \mathrm{CT - s}}$ represent the limits as indicated. Details of the $2^{-\Delta \Delta \mathrm{CT}}$ method have been previously described (41,42). The relative quantification was based on the relative expression of $gfp\_ mut2$ versus 16S rRNA. Wild type $lsr$ operon promoter $C_{\mathrm{T}}$ 's values for $gfp$ were used as reference for all samples.

# Cloning of the strongest $lsr$ promoter variants, EP01rec and EP14rec into a low copy number plasmid

In order to clone the strongest $lsr$ promoter mutants (EP01rec and EP14rec) into the low copy plasmid pFZY1 harboring galK-lacZYA reporter segment (Supplementary Table S1), promoter sequences were amplified by PCR using the primers lsrpF_BamHI and lsrpR_HindIII (Supplementary Table S2) to create pPH01 and pPH14 (Supplementary Table S1). The plasmid pFZY1 and PCR fragments were digested with BamHI and HindIII, purified and then ligated using a T4 DNA ligase (New England Labs) at $16^{\circ}\mathrm{C}$ . The constructions pPH01 and pPH14, which are EP01rec and EP14rec cloned into pFZY1 were confirmed by sequencing.

# Measurement of $\beta$ -galactosidase activity to verify EP01rec and EP14rec promoters in presence of synthetic AI-2

Cultures of $E.$ coli strain LW7 (6) harboring the plasmid pLW11 (6), pPH01 or pPH14 (Supplementary Table S1) containing an ampicillin resistance marker were grown overnight in LB media, and then diluted 100-fold $(\mathrm{OD}_{600} = 0.05)$ into fresh LB media supplemented with $50~\mu \mathrm{g / ml}$ ampicillin. Cultures were incubated at $37^{\circ}\mathrm{C}$ with shaking at $250~\mathrm{rpm}$ in flasks.

When the $\mathrm{OD}_{600}$ reached approximately 0.2, the cultures were split into multiple $2\mathrm{ml}$ culture tubes and $40~\mu \mathrm{M}$ of synthetic AI-2 was added (graciously provided by the H.O. Sintim Research Group, Purdue University). Cultures grew in the absence or presence of AI-2, and were removed at intervals of 2 and $4\mathrm{h}$ for determination of $\mathrm{OD}_{600}$ and $\beta$ galactosidase activity. Specific activity of $\beta$ -galactosidase is expressed in Miller Units (43).

# RESULTS AND DISCUSSION

# Characterization of the promoter library variants

Figure 1 depicts the general scheme for design and construc

tion of promoter libraries and the methodology for selecting promoters with superior expression characteristics (as noted in Materials and Methods). The original library contained approximately $10^{6}$ members and was challenged to grow in the presence of $20~\mu \mathrm{g / ml}$ tetracycline (Figure 1A). Fourteen clones, denoted EP (error-prone), survived tetracycline selection and were sequenced. The mutations were analyzed by sequence alignment (Figure 1B).

In Figure 2, we found that most of the mutations were concentrated very close to or along the 6-bp (AACAAT) and 9-bp (AAGATTAA) sequences of the p-lsrR-box, which are important for LsrR binding of the $lsl$ promoter (4). In fact, most clones showed mutations in the sequence region (-214 to -208 relative to $lslA$ start codon) located between the two p-lsrR-box sequences, 6-bp (AACAAT) and 9-bp (AAGATTTAA).

Aiming to isolate those mutations that contributed to increased $lsl$ promoter strength but that also maintained LsrR repression, we performed site-directed mutagenesis in all the fourteen clones (Figure 1B). We replaced the mutations that were found along the region between the two 6-bp and 9-bp sequence in the p-lsrR-box (TAATGCA) (Figure 2) with the native sequence ( $lsl$ promoter wild type sequence). All fourteen clones with restored $p$ -lsrR-box (named EPrec), were selected again against tetracycline again to verify the influence of these mutations on the promoter strength (Figure 1B).

Interestingly, after this new selection only two clones were able to grow in high con
centration of tetracycline $(20~\mu \mathrm{g / ml})$ , EP01rec and EP14rec (Figures 1C and 4), indicating that except for these two promoter mutants, the mutations within the $p$ -lsrR-box were presumably responsible for the increase in promoter strength in the other 12 clones. This might, in part, be caused by the mutations located in the $p$ -lsrR-box affecting the ability of LsrR repressor to bind, promoting leaky expression in the mutant $lsl$ promoter.

Sequence alignment analysis of the fourteen clones revealed that EP01rec and EP14rec shared a mutation (TGTGCAAT $\rightarrow$ TGTACAAT) located very close to the CRPI sequence box (Figure 2), which led us to hypothesize that this mutation $\mathbf{G}\rightarrow \mathbf{A}$ might be important for enabling the observed increase in $lSr$ promoter strength. Due to this mutation being located near the CRPI sequence box, we searched for promoter elements related to CRP, such as CytR-binding site sequences, which had not been previously identified in the $lSr$ promoter sequence.

The CytR repressor and CRP bind cooperatively to several promoters in $E.$ coli to repress transcription initiation (38). Interestingly, Pederson and Valentin-Hansen (1997) (38) have already demonstrated sequences showing homology to the octameric motifs $5^{\prime}$ -AATG $^{\mathrm{T}}$ /C AAC-3 and $5^{\prime}$ -GTTGCATT-3', respectively termed left (L) and right (R) half-sites on the deoP2 E. coli promoter sequence (in absence of cAMP-CRP). The right (R) half-site $(5^{\prime}$ -GTTGCATT-3') described by Pederson and Valentin-Hansen (1997) (38) also has a consensus sequence (-TGCA), which is present in the same location as the mutual mutation $5^{\prime}$ -TG-(TACA)-AT-3 on both EP01rec and EP14rec found here.

Thus, we combined the in vitro assay findings of Pedersen and Valentin-Hansen (1997) (38) with the following in silico analysis in order to examine if a CytR-binding site exists in

10518

Nucleic Acids Research, 2016, Vol. 44, No. 21

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

![](dt=2026-04-09/ht=03/9694f0fce52d851eac6379c7cbdb26a645f0d6c815fc789f2e523c59fbaccae0.jpg)

![](dt=2026-04-09/ht=03/6cf5200d29e942ee0cd89a974edd4908ffe1c2f48d6c7dcf5fb11925d3922f7b.jpg)

![](dt=2026-04-09/ht=03/c37034281776bd1ad0847f70444ae98a1719693e95d2e653fa278be6a9d3fa30.jpg)

Nucleic Acids Research, 2016, Vol. 44, No. 21

10519

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

![](dt=2026-04-09/ht=03/2473477276fea817c73f7a2ef9437f579d9350eb913ba28e1fcd0e32d4f70fdc.jpg)

10520

Nucleic Acids Research, 2016, Vol. 44, No. 21

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

![](dt=2026-04-09/ht=03/c7b7120fb84f2da77743a3aa39a345173289a2b9ae5aa0becf583268fa213683.jpg)

the $E.$ coli lsrACDBFG operon promoter region [-307 to +92 relative to the start codon of lsrA].

We searched for putative CytR binding sites corresponding to a consensus sequence (consisting of a right motif and/or left motif and specified by Pedersen and Valentin-Hansen (38) in the Lsr intergenic region using the Biostrings package (39) of the open source software Bioconductor. Pedersen and Valentin-Hansen (1997) (38) isolated CytR binding sites, without the presence of cAMP-CRP, in 46 fragments. Using the matchPWM function in the Biostrings package of Bioconductor, we found 7 matches for the right motif (Supplementary Table S4) in wild type $lsr$ promoter sequence.

The highest scoring sequence was the TGTG-CAAT sequence (score = 0.741), which is the site of the (TGTGCAAT→TGTACAAT) mutation in EP01rec and EP14rec. We suggest that the higher expression levels afforded by our two mutated promoters is caused by the single ( $\mathbf{G} \rightarrow \mathbf{A}$ ) mutation, and that this is found in the putative CytR-binding site. Interestingly, this same sequence was also recognized as a match for the left motif (score = 0.765, the sixth highest score for the left motif) (Supplementary Table S4, Figure 3).

Our in silico results showing all high scores for the left motif sequence, as well the top scores for both motifs in the $lsrR$ (reverse-complement) direction, can be found in Figure 4.

Correspondingly, we next performed site directed mutagenesis in the putative CytR-binding site in order to revert the (TGTACAAT) mutation back to wild type (TGTGCAAT) (Figures 1C; 2 and 3, Supplementary Ta

ble S4). The resultant mutants, $\mathrm{EP01rec_{AG}}$ and $\mathrm{EP14rec_{AG}}$ , were tested to ascertain the ability of these mutants and EP01rec and EP14rec to grow at different tetracycline concentrations (0, 2.5, 5, 10 and 20 and $40~\mu \mathrm{g / ml}$ ). We found $\mathrm{EP01rec_{AG}}$ and $\mathrm{EP14rec_{AG}}$ both had decreased ability to grow at high concentrations of tetracycline when compared with EP01rec and EP14rec, demonstrating similarity with the wild type $lSrACDBFG$ operon promoter region (Figure 4).

These results, therefore, confirmed that the $\mathbf{G}\rightarrow \mathbf{A}$ mutation present in the TGTGCAAT sequence (Figures 2 and 3 and Supplementary Table S4), located very close to the CRPI binding site and that might be associated with the CytR binding, is pivotal to increasing the strength of the evolved promoters obtained from the library created in this study. Interestingly, the EP08rec clone was found to possess a mutation at the directly adjacent nucleotide $\mathbf{C}\rightarrow \mathbf{T}$ (TGTGCAA $\rightarrow$ TGTGTAA) found mutated in EP01rec and EP14rec (Figure 2).

Pederson and Valentin-Hansen (38) have demonstrated through qualitative interactions of CytR with DNA, using DNaseI and dimethyl sulfate (DMS) footprinting at repressor concentrations that saturate the binding site, that interaction of CytR with the octameric motifs AATGTAAC and GTTGCATT invariably protects the central guanine from DMS methylation, consistent with the well conserved G at this position in the deoP2 E. coli promoter sequence.

Experimental validation of this hypothesis involving CytR contact point in Lsr promoter sequence in a more focused study is needed, however.

Strength of EP01rec and EP14rec promoter variants in comparison with wild type $E.$ coli lsrACDBFG operon promoter region

In order to measure and compare the expression levels among the $lsr$ promoter mutants, we evaluated both $tet$ (C) and $gfp\_mut2$ expression to determine the promoters' strength. The pLSR plasmid was constructed using a pTS40 backbone, which contains two reporter genes, $gfp\_mut2$ and $tet$ (C), resulting in bicistronic expression under the $lsr$ promoter, either wild type or mutant.

Our earlier library screenings and selections were performed using the tet (C) gene because it facilitated rapid searching for the strongest promoter candidates from a library containing around $10^{6}$ members (based on the ability of these clones to survive in high concentrations of tetracycline). In order to confirm $l sr$ promoter strength, we performed assays to detect and measure GFP expression (Figure 5), which is encoded by the first gene after the promoter sequence.

Samples taken at early and late exponential phases $(\mathrm{OD}_{600\mathrm{nm}}0.5$ and 1.0) were evaluated for GFP expression (represented by mean fluorescence using FACS). In all cases, the p- $l srR$ -box sites restored promoters (EP01rec and EP14rec) resulted in higher GFP fluorescence than their respective controls. There was no difference between the $\mathrm{OD}_{600\mathrm{nm}}0.5$ and 1.0 data are not shown). Statistical sig
nificance was determined with ANOVA and the Tukey-Kramer method, adopting a significance level of $\alpha = 0.001$ .

EP01rec and EP14rec showed 8-fold and 14-fold more GFP than the wild type $l sr$ promoter, respectively. As expected,

Nucleic Acids Research, 2016, Vol. 44, No. 21

10521

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

![](dt=2026-04-09/ht=03/9b8950e9ccdf304e04a1d06f449974e358fbad86e18085628f13766ca7acc87d.jpg)

![](dt=2026-04-09/ht=03/e1651826a6d9e170d816eca13d915de2bfbb5bf0c2b201a04f778916f843e0b3.jpg)

![](dt=2026-04-09/ht=03/717e23b319cd41cced86e9871ee4e9fc47e7329fd759381a29efbe5b8bd01b75.jpg)

EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ achieved GFP expression levels comparable with the wild type $lsr$ promoter.

These same samples, as analyzed via FACS, were also visualized using a 20x microscope fluorescence field. EP01rec and EP14rec were brighter than other $lsr$ promoter mutants (Supplementary Figure S1), confirming the data obtained through FACS. Results obtained investigating GFP levels

![](dt=2026-04-09/ht=03/fd0a313e3eb892ee698b44a72be6b8e2529386e200cf3457fa550ec29271d273.jpg)

corroborated the data obtained through the tetracycline resistance assays (Figure 4). As noted earlier, and reconfirmed here with analysis of GFP expression, the single nucleotide found mutated in the putative CytR-binding site sequence (TGTGCAAT $\rightarrow$ TGTACAAT) in the strongest promoter mutants, EP01rec and EP14rec was important for increasing the strength of these promoters.

Finally, quantitative PCR (qPCR) was also performed to measure gfp_mut2 transcriptional levels obtained from each of the $lsr$ promoter mutants (EP01rec, EP01recAG, EP14rec, and EP14recAG) versus the wild-type $lsr$ promoter (Figure 6). The results corroborate the phenotypic patterns (Figure 6). We detected higher levels of gfp mRNA in EP01rec

10522

Nucleic Acids Research, 2016, Vol. 44, No. 21

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

![](dt=2026-04-09/ht=03/06a12e636d719413bdaf6cefa5730e9e484098c307fe9571ecdd2b54f3cea065.jpg)

![](dt=2026-04-09/ht=03/2f30a4735ab1b20b1b5a807c578f1a8e629d753ff55bc7a248a3d8ea860fcaa5.jpg)

and EP14rec when compared to the wild type $lsr$ promoter, and also to EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ . The gfp transcriptional levels found with the wild type $lsr$ promoter were lower than the levels found with EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ and are consistent with the GFP expression data. Recall, however, that other mutations are present in the EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ clones that may have influenced their levels relative to the wild type $lsr$ promoter.

Interestingly, the gfp transcriptional pattern for EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ (Figure 6) does not precisely replicate the GFP expression pattern (Figure 5) and that the difference in transcriptional levels is less than expected when one compares EP01rec and EP14rec to EP01rec $_{\mathrm{AG}}$ and EP14rec $_{\mathrm{AG}}$ .

# Inducible expression based in AI-2 quorum sensing components

Initial switching experiments using the $lsr$ operon promoter mutants, as well as the wild type promoter cloned into the original pLSR plasmid were carried out in $E.$ coli luxS and $lsrR$ mutants as hosts. In the case of the luxS mutant, we added exogenous AI-2 to stimulate expression and in the case of the $lsrR$ mutant, we looked for a step increase in expression per cell as cultures ensued. Interestingly, in these cases, there was minimal, if any, apparent repression, even after adding $0.2\%$ glucose in LB agar media (data not shown).

Perhaps this was an artifact of the pLSR plasmid carrying the CloDF13 replication origin (relatively high 20-40 copies/cell), which is originally from the pTS40 plasmid backbone (35). Thus, LsrR repressor levels, present in low or no copy number per cell might have been insufficient to repress the multi-copy $lsr$ promoter.

Accordingly, in order to more carefully evaluate whether both $lsr$ promoter mutants, EP01rec and EP14rec, had sim

ilar switching characteristics as the native promoter, we cloned the entire promoter regions into a lower copy number vector, the single copy plasmid pFZY1. pFZY1 is a galK-lacZYA transcriptional fusional vector that expresses LacZ under an inserted promoter (44). We transformed pPH01, pPH14 (Supplementary Table S1) or pLW11 (6), which contains the wild type $lsr$ promoter cloned into pFZY1, into the LW7 strain, which is a luxS knockout (Supplementary Table S1).

Miller assays were then performed to measure LacZ expression levels under the three promoters: wild type, EP01rec and EP14rec, in both the presence and absence of AI-2. That is, we measured LacZ expression levels after we allowed the cells to reach $\mathrm{OD}_{600\mathrm{nm}}\sim 0.2$ and added $40~\mu \mathrm{M}$ AI-2, continued growth and retested. This mid-exponential phase OD scenario is aligned with native QS signaling. Thus, most of $lslr$ promoter elements such as AI-2 uptake (lsrACBD), modification (phosphorylation by LsrK) and degradation (lsrFG) should have been at appropriate levels for subsequent analysis of engineered $lslr$ promoter activity.

As expected, after 2 and $4\mathrm{h}$ of AI-2 induction all three of the promoters showed higher LacZ expression levels when compared with the same experiments performed in absence of inducer (Figure 7). After $2\mathrm{h}$ of AI-2 induction, EP01rec showed a nearly 3-fold increase in LacZ expression (Miller units) when compared with wild type. EP14rec exhibited only a 1.5-fold increase (Figure 7). Then, at later times, the EP14rec promoter exhibited over a 2-fold increase. In both cases, the influence of LsrR repression (indicated by amplification upon addition of AI-2), was exhibited.

Experimental results obtained with $E.$ coli cells carrying both multi and single copy vectors for $lslr$ promoter driven gene expression were successful in that engineered promoter sequences exhibited greater expression while still retaining the LsrR

Nucleic Acids Research, 2016, Vol. 44, No. 21

10523

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

repression capability. Subsequent use of these evolved promoters EP01rec and EP14rec as simple expression vectors or as elements of additional genetic circuits will naturally require consideration of the availability of LsrR, LsrK and other AI-2 mediating components (13).

# CONCLUSIONS

By creating a library of $lsr$ operon promoter mutants through directed evolution we were able to obtain mutant promoters with activity stronger than the wild type promoter. Most of the mutants (12 out of the 14) showed increased strength that was related to mutations found near or within the LsrR-binding site; not just in the 6-bp (AACA AT) and 9-bp (AAGATTTAA) sequences of the p- $lsrR$ -box region. Even though the p- $lsrR$ -box sequence region has been shown to be essential to the LsrR-binding site (4), our results indicate that more nucleotides present in the vicinity of the proposed LsrR-binding site are also involved in controlling expression levels under this E. coli promoter.

In addition, we have isolated two $lsr$ promoter mutants, EP01rec and EP14rec that behaved differently than the wild type. We identified a defining mutation in EP01rec and EP14rec responsible for the increase in $lsr$ promoter strength. Th
is point mutation is located very close to the CRPI sequence and was examined using in silico methods, where we found its association with the CytR binding site. The combination of experimental data from a previous study performed with the $E.$ coli deoP2 promoter (38) and our in silico analysis revealed a possible association between increased promoter strength and a CytR-DNA-binding domain.

Finally, tests on inducible expression based on AI-2 QS were performed with EP01rec and EP14rec promoters. Data demonstrated that both promoters preserved their switching character based on the presence of AI-2. We note, however, that in order to demonstrate the full switching character of these promoters which are significantly stronger than the wild type, a single copy number plasmid was needed. This may be due to the need for enmeshing components of these vectors with other endogenous components (such as CytR or CRP) that are present at native levels.

Thus, should maximum overexpression be an objective for application of these promoters, additional investigations are envisioned that more optimally accommodate native regulatory components and those of the synthetic construct. In general, our work represents progress toward engineering the Lsr promoter system into a tool for applied biotechnology. We can envision future applications such as expression of therapeutic proteins (45) through engineered probiotic bacteria (46), controlled by gut flora or even by pathogenic bacteria.

# SUPPLEMENTARY DATA

Supplementary Data are available at NAR Online.

# FUNDING

Defense Threat Reduction Agency (DTRA) [HDTRA1-13-1-00037]; US National Science Foundation [CBET #1160005]. Funding for open access charge: DTRA.

Conflict of interest statement. None declared.

# REFERENCES

10524

Nucleic Acids Research, 2016, Vol. 44, No. 21

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016

Nucleic Acids Research, 2016, Vol. 44, No. 21

10525

Downloaded from http://nar.oxfordjournals.org/ by guest on December 5, 2016