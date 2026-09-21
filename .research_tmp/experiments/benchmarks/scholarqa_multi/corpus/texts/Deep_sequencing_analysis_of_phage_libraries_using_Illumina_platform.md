# Deep sequencing analysis of phage libraries using Illumina platform

Wadim L. Matochko a, Kiki Chu b, Bingjie Jin a, Sam W. Lee b, George M. Whitesides c, Ratmir Derda a,\*

$^{a}$ Department of Chemistry and Alberta Glycomics Centre, University of Alberta, Edmonton, Alberta, Canada T6G 2G2

b Cutaneous Biology Research Center, Massachusetts General Hospital and Harvard Medical School, Charlestown, MA 02129, USA

$^{c}$ Department of Chemistry, Harvard University, Cambridge, MA 02138, USA

# ARTICLE INFO

Article history:

Available online 20 July 2012

Keywords:

Deep sequencing

Phage display

Peptides

Illumina

Amplification

Diversity

# ABSTRACT

This paper presents an analysis of phage-displayed libraries of peptides using Illumina. We describe steps for the preparation of short DNA fragments for deep sequencing and MatLab software for the analysis of the results. Screening of peptide libraries displayed on the surface of bacteriophage (phage display) can be used to discover peptides that bind to any target. The key step in this discovery is the analysis of peptide sequences present in the library.

This analysis is usually performed by Sanger sequencing, which is labor intensive and limited to examination of a few hundred phage clones. On the other hand, Illumina deep-sequencing technology can characterize over $10^{7}$ reads in a single run. We applied Illumina sequencing to analyze phage libraries. Using PCR, we isolated the variable regions from M13KE phage vectors from a phage display library. The PCR primers contained (i) sequences flanking the variable region, (ii) barcodes, and (iii) variable $5^{\prime}$ -terminal region.

We used this approach to examine how diversity of peptides in phage display libraries changes as a result of amplification of libraries in bacteria. Using HiSeq single-end Illumina sequencing of these fragments, we acquired over $2 \times 10^{7}$ reads, 57 base pairs (bp) in length. Each read contained information about the barcode (6 bp), one complimentary region (12 bp) and a variable region (36 bp). We applied this sequencing to a model library of $10^{6}$ unique clones and observed that amplification enriches $\sim 150$ clones, which dominate $\sim 20\%$ of the library.

Deep sequencing, for the first time, characterized the collapse of diversity in phage libraries. The results suggest that screens based on repeated amplification and small-scale sequencing identify a few binding clones and miss thousands of useful clones. The deep sequencing approach described here could identify under-represented clones in phage screens. It could also be instrumental in developing new screening strategies, which can preserve diversity of phage clones and identify ligands previously lost in phage display screens.

© 2012 Elsevier Inc. All rights reserved.

# 1. Introduction

Phage display is a powerful method for the discovery of peptides that bind to any target [1,2]. The binding of phage library to a target, or "panning", narrows the naïve library of $10^{9}$ clones to $10^{5} - 10^{6}$ clones. This is a typical number of phage clones recovered after one round of panning, but only some of these clones have affinity for the target. To narrow the diversity of true binding clones the library is amplified in bacteria. Amplification multiplies the copy number of each clone and generates a focused library, which can be panned again [2].

Rounds of panning and amplification narrow the diversity of the library and enrich for phage clones that present target-binding peptides. The key to this process is the analysis of peptide sequences present in the library at various steps of the screening. Sequences enriched as a result of selection correspond to the specific binders against the target. Conventional San-

ger sequencing of clones require isolation of DNA from individual phage clones. It is a labor-intensive process and is rarely used to analyze more than a hundred library clones.

Analysis of a small number of sequences enriched in a screen can be used to predict one consensus motif [3,4]. Phage-display screens could also yield a large number of consensus motifs. For example, thousands of diverse sequence motifs should emerge from the panning against intact cells because an average cell contains thousands of structurally diverse receptors. If a screen selects a large number of independent binding clones, one has to sequence large numbers of clones to identify all the useful binding sequences.

Arap, Pasqualini and co-workers were the first to use 454 sequencing to analyze $\sim 50,000$ sequences from a library of 7-mer peptides; the author applied this technology to identify peptides emerging from panning against different organs in vivo [5]. Subsequently, the same sequencing was used by several groups to monitor selection of binding proteins from a library of open reading frames (ORF) displayed on phage [6,7]. Sidhu and co-workers used 454-sequencing to boost selection of peptides binding to

Methods 58 (2012) 47-55

SEVIER

Contents lists available at SciVerse ScienceDirect

Methods

journal homepage: www.elsevier.com/locate/ymeth

METHODS

* Corresponding author.

E-mail address: ratmir.derda@ualberta.ca (R. Derda).

1046-2023/$ - see front matter © 2012 Elsevier Inc. All rights reserved.

http://dx.doi.org/10.1016/j.ymeth.2012.07.006

different PDZ domains [8-10]. Notably, the authors used barcoded primers for the preparation of a library for sequencing and, thus, sequenced 22 independent panning experiments in one run.[8] Lerner and co-workers applied 454 sequencing to find antibodies that bind to various proteins displayed on the surface of bacteria [11]. Sequencing technologies of throughput higher than $10^{4}-10^{5}$ could provide more complete coverage of the libraries. Increased throughput could also allow analysis of multiple experiments in a single run.

Illumina/Solexa deep-sequencing technology analyzes a library of blunt-ended double stranded DNA (dsDNA) fragments and generates up to $10^{9}$ base pair (bp) reads in a single run. For example, Fisher and co-workers recently demonstrated the use of Illumina sequencing to characterize phage-displayed libraries of single chain antibodies (scFv) [12]. Fields and co-workers used Illumina sequencing to characterize selection from libraries of WW-protein displayed on T7 phage [13].

Johan den Dunnen and coworkers used Illumina to characterize peptide libraries after one round of panning against cell surface receptors [14]. In this paper, we present a one-step PCR that converts a library of M13KE plasmids isolated from the phage library to a collection of short dsDNA sequences suitable for Illumina sequencing. Using custom Matlab software, we perform large-scale analysis of sequence diversities.

Using deep sequencing, we explore the effects of amplification of phage libraries in bacteria on the diversity of peptides in these libraries. In previous publications, the result from sequencing of $\sim 100$ phage clones suggested that the amplification process enriches for specific peptide sequences [15,16]. Large scale sequencing, however, can provide observations that could not be interpreted from the sequences of 100 clones [17]. For example, deep sequencing of a library of DNA aptamers demonstrated that repeated amplification does not select for particular sequences.

Instead, it enriches DNA sequence motifs that have low stability [18,19]. In this report, we analyzed diversity of amplified libraries using Illumina and observed a collapse of diversity in phage-displayed libraries after a single round of growth in bacteria. The collapse of the $10^{6}$ -scale library to a few hundred abundant sequences would not be visible in small-scale Sanger sequencing [17,20,21]; it could also have been difficult to detect with smaller-throughput 454 Sequencing.

Characterization of sequence diversity is important for phage display technology, which has been used in over 5000 publicatio
ns and patents in the past 20 years. It has enabled the discovery of ligands for hundreds of targets, yet the literature still contains several poorly-explained observations: (1) identical sequences could emerge from unrelated screens for unrelated targets [22,23], and (2) screens that should yield a large number of diverse ligands often yield only one sequence motif (reviewed in [16]).

The nearly complete sequence coverage of libraries illuminates the origin of these observations. It highlights that the collapse of diversity in amplification might be one of the major limitations of phage-display technology. Deep-sequencing analysis will make it possible to bypass problems originating from the unwanted collapse of diversity [11]. Large-scale analysis could also help develop methods that preserve diversity of peptide libraries [24,25]. It can be used to enable discoveries of ligands that previously have been lost in phage display screens.

# 2. Experimental design

# 2.1. Choice of the library

In this report, we sequence a commercially-available library of random 12mers from New England Biolabs (Ph.D-12). This library has been used in $\sim 800$ publications (source of estimate: PLoMics database http://www.treeofmedicine.com/phagedisplay and Mi

moDB database [22,23]). According to the manufacturer (NEB), naive library contains up to $10^{9}$ different sequences. Since this number is beyond the sequencing capabilities of Illumina, we worked with 1/1000th portion of the library containing $10^{6}$ different sequences. If a sequencing run produces 20 million sequences, the observed frequency of sequences could be approximated by a Poisson distribution with an expectation value of 20. For the above uniform library of $10^{6}$ clones, the distribution predicts that every sequence will be observed at least 5 times. Over $99\%$ of the library should be observed within 3 standard deviation of the expectation value $(\mathrm{sqrt}(20)\times 3 = 13)$ . The majority of the clones, thus, should be present at 7-33 copies.

To explore the effect of amplification on library diversity, we amplified a pool of $10^{6}$ clones to $10^{13}$ pfu and isolated ssDNA from the combined pool of phage. Approximately $10^{8}$ copies of each clone should be present in this pool. If relative abundances of clones were not changed during amplification, abundances of clones observed after deep sequencing should follow the Poisson distribution described above. In reality, we observed that a distribution of clones was dramatically different from the Poisson distribution, suggesting that growth preference of individual clones led to enrichment of some clones and depletion of others.

# 3. Description of materials

# 3.1. Isolation of DNA from phage libraries

# 3.1.1. Reagents

Polyethylene glycol MW 8,000 (PEG) (Fisher BP233-1), sodium chloride (NaCl) (Fisher S271-500), chloroform (Sigma 319988), phenol (Fisher A931l-1), anhydrous ethanol, distilled water, sodium iodide (NaI) (Fisher BP323-100), sodium acetate (Fisher S78229-1), glycogen (Invitrogen 10814-010).

# 3.1.2. Materials

1.7 Microcentrifuge tubes (Fisher 14222168), PEG/NaCl solution $(20\% (\mathrm{w / v})$ PEG/2.5 M NaCl, sterilized by autoclaving), micropipettes (Mandel P2N, P10N, P200N, P1000N) and micropipette tips (Fisher 02-707-439 $(10~\mu \mathrm{L})$ , 02-707-430 $(200~\mu \mathrm{L})$ , 02-707-404 $(1000\mu)$ ), benchtop microcentrifuge.

# 3.2. Preparation of the DNA for sequencing

# 3.2.1. Reagents

Hot start high fidelity DNA polymerase (e.g.

Affymetrix HotStart-IT Taq DNA Polymerase (71195) and Phusion® Hot Start II High-Fidelity DNA Polymerase (Finnzymes F-549L)), Illumina paired-end DNA sample prep kit (Illumina), QIAquick PCR purification kit (Qiagen 28104), QIAquick MinElute PCR purification kit (Qiagen 28004), QIAquick gel extraction kit (Qiagen (28704), DNA loading buffer (50 mM Tris pH 8.0, 40 mM EDTA, 40% (w/v) sucrose), QIAEX II gel extraction kit (Qiagen 20021), certified low range ultra agarose (Bio-Rad 161-3106), 10× TBE buffer (Bio Basic A0026), 50× TAE buffer (Fisher FERB49), ethidium bromide (Fisher BP1302-10),

DNA ladder (New England Biolabs N3233S), chloroform (Sigma 319988), phenol (Fisher A9311-1), anhydrous ethanol, distilled water, sodium acetate (Fisher S78229-1), glycogen (Invitrogen 10814-010).

# 3.2.2.Materials

1.7 ml microcentrifuge tubes (Fisher 14222168), DNA gel electrophoresis apparatus, micropipettes (Gilson P2N, P10N, P200N, P1000N), micropipette tips (Fisher 02-707-439 $(10\mu \mathrm{L})$ 02-707-430 $(200~\mu \mathrm{L})$ 02-707-404 $(1000\mu)$ ), benchtop microcentrifuge, PCR thermal cycler.

48

W.L. Matochko et al./Methods 58 (2012) 47-55

# 3.3. Sequencing of the library

# 3.3.1.Materials

HiSeq Illumina sequencer.

# 3.4. Analysis of the library

# 3.4.1.Materials

Computer, MATLAB software, MATLAB scripts (Supporting information).

# 4. Description of methods

# 4.1. Isolation of DNA from phage libraries

DNA was isolated using standard NaI/EtOH precipitation method. The steps below are for $500~\mu \mathrm{L}$ of solution containing $10^{12}-$ $10^{13}$ pfu/mL of phage:

# 4.2. Preparation of the DNA for sequencing

DNA isolated from the Ph.D.™-12 Phage Display Peptide Library was subjected to PCR amplification with primers flanking the variable region. A list of optimized reaction conditions for PCR amplification is found in Supporting Table S1 along with cycling conditions specific for each primer listed in Supporting Table S2.

Extraction Kit. Purify and concentrate the extracted DNA fragment using phenol-chloroform and ethanol precipitation as described in the previous section.

![](dt=2026-05-09/ht=13/d41abd05efd0070bce07f08ad9f54dccaeaffa3f72dbf826b1f7fdc3fd91bd02.jpg)

![](dt=2026-05-09/ht=13/1788cc6cd9ec57063f148430cdd49eed03978b5a134f41aabb23737e11d1b4a7.jpg)

![](dt=2026-05-09/ht=13/bc65003f95f884a5fbe2c732cac48b719301c8f6e8952c35a9af59cefefc1e98.jpg)

W.L. Matochko et al./Methods 58 (2012) 47-55

49

# 4.3. Sequencing of the library

Concentration of dsDNA with ligated Illumina adapters was estimated using Qubit Fluorimeter (Invitrogen) or Agilent Bioanalyzer using manufacturer's protocol. The sample was diluted to the concentration of $10\mathrm{nM}$ and submitted for sequencing to Harvard FAS sequencing facility. The sequencing was performed using Illumina HiSeq and 50 bp single end reads.

# 4.4. Analysis of the library

# 5. Results

# 5.1. Isolation of variable dsDNA fragments from phage libraries

The majority of the phage display vectors share the same design: they contain a variable sequence flanked by constant regions containing restriction enzyme sequences (used for cloning of the library). We attempted to isolate the library sequences using KpnI and EagI restriction enzymes to isolate variable domains from M13KE vectors [26]. The collection of sticky-end fragments could be repaired to give blunt-ended fragments with identical termini.

These fragments, however, could not be reliably sequenced by Illumina because the sequencing algorithm uses differences in terminal nucleotides to distinguish sequence clusters [27]. We attempted to introduce variable termini by ligation of short random nucleotide sequences; this approach, however, gave poor yields and was eventually abandoned. Nevertheless, we expect that excision by restriction nucleases could be useful for other deep sequencing approaches, such as Ion Torrent, which could process fragments with identical termini.

Our successful method for the isolation of variable regions used PCR amplification with primers complementary to the 12-bp constant regions flanking the variable sequence in the M13KE vector. The forward PCR primer contained a NKKNKK se
quence at its $5^{\prime}$ position (Fig. 1A). Each primer, thus, was a mixture of $4 \times 2 \times 2 \times 4 \times 2 \times 2 = 256$ different primers. PCR with these primers generates dsDNA with 256 different bunt-end termini; this diversity should be sufficient for the algorithm that finds individual DNA clusters (polonies) during sequencing.

We selected the NKKNKK sequence to minimize the possibility for hybridization with $(\mathrm{NNK})_{12}$ motifs in the library. The forward primer also contained a barcode sequence ATCACT. We selected this particular sequence after aligning all 256 (NKKNKK)-(ACTATC)-TATTCTCACTCT sequences to $(+)$ and $(-)$ strand of M13KE vector. For all sequences, we observed hybridization of $<7$ bp, which should not interfere with PCR conditions optimized for 12 bp-long adapter sequences (Fig. 1C). We used a similar algorithm to find other barcode sequences (Supporting Information Tables S1 and S2).

The use of

multiple barcodes allows for processing of multiple phage libraries in a single run (Supporting Information Fig. S5).

Successful PCR amplification of variable fragments was confirmed as a single band on $2\%$ agarose gel. Amplification using primers with shorter variable regions or other barcode sequences yielded similar results (Supporting Table S1). Due to differences in melting temperatures of the primers, PCR conditions had to be re-optimized for each barcode sequence (Supporting Table S2). The fragments amplified from libraries of different size, such as 12-mer, 7-mer or 9-mer, gave dsDNA fragments of expected sizes. For example, the protocol described in Fig.

1A was validated using three different libraries: (1) Ph.D-12TM, a library of 12-mers, 36 bp variable region; (2) Ph.D-7TM, a library of 7-mers, 21-bp variable region, and (3) Ph.D-C7CTM, a library of 7-mers flanked by Cys, 27 bp variable region. We used two primers with a total length of 38 bps long and observed PCR products close to the expected (1) 74, (2) 59, and (3) 65 bps (Supporting Fig. S5A).

# 5.2. Preparation of Illumina-compatible dsDNA Fragments

Ligation of DNA adapters that enable Illumina sequencing was performed according to the protocols supplied with the Illumina paired-end adapter Kit. Successful ligation of Illumina adapter sequences to the blunt-ended PCR product occurred only after end-repair of the product (Fig. 1D). Ligation yielded two products, referred to as 2L and 2S, with length similar to that of the expected product (140 bp for the 12-mer library).

To enrich the DNA fragments, which were successfully ligated with the adapters, we run PCR amplification of purified 2L and 2S fragments with primers that complement the Illumina adapters. Both 2L and 2S yielded products of correct size after PCR (Fig. 1E) confirming that both 2L and 2S contained correctly ligated adapters. Both products were subjected to Illumina sequencing (single-read, 50 bp reads on HiSeq) yielding similar sequence abundances and diversities (see Fig. 3B below).

# 6. Overview of the analysis

# 6.1. Design of the analysis software

Sequencing by Illumina generates a $\sim 4 - 10$ Gigabyte text file. It is difficult to handle because, most desktop computers cannot open the file in a standard text editor. Additionally, Illumina is used primarily for genome sequencing, and most available software is written for assembly of genomes. Therefore, we wrote a software tailored for the analysis of phage libraries. The basic feature of the software is batch processing. The program first breaks the original $4 - 5\mathrm{Gb}$ FASTQ file into text files of $\sim 100\mathrm{Mb}$ each.

The subsequent processing, thus, requires less operational memory. Analysis proceeds in several steps: (i) conversion of one FASTQ file into smaller plain text files, (ii) identification of constant complementary regions and parsing of sequences,(iii) analysis of sequence quality, (iv) analysis of diversity of sequences, (iv) translation of sequences, (vi) plotting. After each step, the program saves intermediate files in plain text (.txt) format. Any intermediate text files can be opened and inspected in a standard text editor.

Software written in Matlab was effective in analyzing a $4 - 5\mathrm{Gb}$ FASTQ file in $6 - 8\mathrm{h}$ on an average desktop or laptop computer (Supporting Information Scheme S5). We anticipate that re-writing the same script in a lower-level language (e.g. C++) could further accelerate the processing.

# 6.2. Overview of the scripts

Although the length of the dsDNA construct depicted in Fig 1C is 72 bp, single-end sequencing yielded reads of only 57 bp and con

50

W.L. Matochko et al./Methods 58 (2012) 47-55

![](dt=2026-05-09/ht=13/9c1b319f742eb1b559cf699904aa49a35a3db22e913b17898474714a5f3341e0.jpg)

tained complete sequence for only one constant region: either from the forward or the reverse primer. We designed the algorithm which used one constant adapter region to map the functional portions of the sequence: (1) NKKNKK portion, (2) barcode portion, (3) left adapter, (4) R36, and (5) right adapter (see Fig. 1A, C, F).

The process starts from the rawseq.m scripts, which breaks the original FASTQ file into smaller text files, 250,000 lines each. The parseq.m script then searched for the forward or the reverse adapter sequences (highlighted grey or blue in Fig 1C). We used a multistep algorithm for the identification of the adapters. The majority of the sequences were mapped by perfect alignment to full-length adapter sequence ( $\langle \mathrm{PERF} \rangle$ in Fig. 2). $1\%$ of sequences contained adapters with one mutation ( $\langle 1\mathrm{MuT} \rangle$ in Fig. 2; mutation is highlighted in red).

Few adapters had one internal deletion ( $\langle 1\mathrm{Del} \rangle$ in Fig. 2; deletion is underscored, Fig. 2). A significant fraction of adapters had terminal truncations (lines tagged as $\langle 2\mathrm{TRN} \rangle$ to $\langle 7\mathrm{TRN} \rangle$ in Fig. 2). Truncated reads contained sequences of nucleotides from ith to $(56 + i)$ th position ( $i = 2 - 25$ ).

Finally, primers with excessive truncations in one complementary region could be identified by alignment with the complementary sequence at the opposite end of the variable region (lines labeled as $\langle \mathrm{EndA} \rangle$ in Fig. 2).

This algorithm mapped the majority of the forward and reverse reads (Fig. 2A forward and Fig. 2B for reverse search). Approximately $1.6\%$ of sequences (0.5 million) could not be mapped because they contained a large number of low-quality reads or reads with multiple mutations or deletions in the adapter regions.

The parsed files were then processed by the `quaseq.m` script that assessed the quality of the R36 region containing the $(\mathrm{NNK})_{12}$ sequences. We selected only high-quality output in which all nucleotides had Phred Quality Score above 5 (this value could be changed in `quaseq.m` script on demand). High-quality sequences were then analyzed by `uniseq.m` script to generate abundances of nucleotides and cognate peptide sequences. The results were saved to `uniqueN_QF.txt` and `uniqueN_QR.txt` file (where F and R designate analysis of forward and reverse reads). The files are available as part of Supporting information.

In summary, from 32 million raw reads, the software identified $\sim 11.1$ million forward and 20.2 million reverse reads from which R36 sequences could be extracted. From R36 motifs with NNK structure, the software extracted 8.5 and 17.8 million peptide sequences from forward and reverse reads respectively. In current analysis of 12-mer libraries, the majority of the forward reads were

W.L. Matochko et al./Methods 58 (2012) 47-55

51

![](dt=2026-05-09/ht=13/754913f1d81a5f258ff97a9e47ee4b1845319ca4df054b6e5bd9140548db175a.jpg)

![](image)
lakehouse2/hive-ha/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-05-09/ht=13//fa69aa9b1714db8b69c84a74d6a1455fd4341b2832d72ba6e332da27cdded215.jpg)

truncated at the 11th amino acid (see uniqueN_QF.txt). Reverse reads, however, contained sequences for full-length 12-mer peptides (see uniqueN_QR.txt). We focused the remaining analysis on the 17.8 million reverse reads.

The script had options to retain or discard the sequences that did not have NNK format (i.e., sequences with A or C in position 3, or 6, or 9, etc.). If non-NNK sequences were retained, the results contained a significant fraction of sequences with TGA stop codons. M13KE vectors with stop codon in the N-terminal region of the $pIII$ gene would lack N-terminal leader sequence and would not produce viable phage.[26] We concluded that TGA codons and other non-NNK codons are sequencing errors.

# 6.3. Preliminary analysis of sequence diversity in the library

Complete analysis of sequence diversities obtained using Illumina sequencing is beyond the scope of this manuscript. Here, we present the preliminary analysis of the sequences, and we confirm that sequencing runs are reproducible. Fig. 3 describes the distribution of sequence abundances in the library obtained by sequencing of two library preparations (bands 2S and 2L in Fig. 1D). The abundance of sequences in the two runs were similar

![](dt=2026-05-09/ht=13/3a1a06ffc289d8989bc2ea2a96ca91875335cf881e442190add85c16c322e84c.jpg)

(see Fig. 3): some unique peptides were found in copy numbers of $10^{4}$ and higher; nearly $10^{6}$ peptide sequences were found in low copy number. The abundances of specific peptide sequences were highly reproducible between two runs (Fig. 3B). Peptides, which were observed $10^{2} - 10^{5}$ times in sequencing run 1, were observed at similar copy number in the 2nd sequencing run. Deviation from 1:1 correlation were observed at copy number $< 100$ . Some peptides, observed at copy number of 10-100 in the 1st run, were present at much lower copy number in run 2 or completely absent from the other sequencing run.

Distribution of sequence abundance was dramatically different from the predicted Poisson distribution with an expectation value of 20. It could not be modeled as Poisson distribution with any expectation value. A mere 20 clones constitutes $8\%$ of the size of the library and were present at a copy numbers of $>30,000$ (Fig. 4). On the other hand, 500-800 thousand diverse sequences constituted another $8\%$ and were present at a copy number of $<10$ .

The distribution of sequence abundances followed the power-law distribution, producing a linear plot on a log-log scale (Fig. 3A, insert). We observed a deviation from this distribution for the low copy number peptides. Extrapolation of a log-log plot predicts that the number of single copy-number sequences should be $3 - 5 \times 10^{5}$ . The observed deviation suggested that a significant fraction of low-copy-number peptides could be the result of sequencing errors. Errors are abundant in Illumina sequencing [28], but we anticipate that many of these errors could be easily identified.

One possible algorithm could be based on the assumption that the library is sparse. In other words, a library of nucleotides with structure $(\mathrm{NNK})_{12}$ has $(4 \times 4 \times 2)^{12} = 10^{18}$ members, and in a pool of $10^{6}$ sequences, the probability to find a mutant is small. Despite this prediction, the search for point mutations of most abundant sequences yielded $\sim 100$ point-mutants for high-copy-number sequences (Supporting Information Fig. S6).

The majority of these mutated sequences were present at low abundance (Figure S6); average abundance was $\sim 1\%$ , which is sim

52

W.L. Matochko et al./Methods 58 (2012) 47-55

![](dt=2026-05-09/ht=13/dedd26d4d2a73658130b8772272ac87b7caac05fc49c68cbe88f00c8ab1d9ca2.jpg)

![](dt=2026-05-09/ht=13/be9a7e45af9a6d469a18be52961cc238b5bdf45e9d0c1e1014845ee8b1a3f429.jpg)

![](dt=2026-05-09/ht=13/76889d53baa4a0a341b2801f5315b89b8c268ef75ea2f1f541e7f3a1897657c2.jpg)

![](dt=2026-05-09/ht=13/fc16bfb45c7100a547600885b323eea0e81cd88a680cc538f8c1866d73e4a82a.jpg)

![](dt=2026-05-09/ht=13/e30ab7aa06841357c5f5f4bb1618a86379ac0d1656de1933af34002a19d67b9d.jpg)

![](dt=2026-05-09/ht=13/dff3594328a37ae74a74d272edcc936c489dafae33c69823831088fe5307b4a0.jpg)

![](dt=2026-05-09/ht=13/423e147e019c4a9be2c193dda17b042192bcc4cb1883adaa607ae0d8fb1141f1.jpg)

ilar to the frequency of point mutations in adapter sequences (compare $\langle \mathrm{PERF}\rangle$ and $\langle 1\mathrm{Mut}\rangle$ in Fig. 2). This preliminary analysis suggests that sequences with abundance of $>100$ copies contain no errors. Those with abundance of $<  100$ could be potentially repaired. Validation of the error analysis and repair algorithm, however, is beyond the scope of this manuscript.

Positional analysis of amino acid abundances (Fig. 5) demonstrated that the distribution of amino acids in the top 150 se

quences, present at copy number of $>10,000$ , was different from that of the remaining library. Distribution of amino acids in sequences present at copy number $< 10,000$ was similar to those in the overall library. Overall distribution of amino acids in peptides in the library was similar to those observed in earlier reports [17,21,29]. Library had abundant Ser/Thr in all positions. Abundance of Cys was low in all positions. N-terminus exhibited significant preference for some amino acids, presumably due to

W.L. Matochko et al./Methods 58 (2012) 47-55

53

proteolytic preference of the peptidase, which truncates leader peptide sequences following the displayed peptide [20,21].

Clustering analysis identified ten distinct sequence patterns in the top 150 fastest growing clones. Fig. 6 describes the clustering tree diagram and protein LOGO [30] display of the conserved sequence within each sub-sequence. Remarkably, a rare amino acid W appeared as a consensus amino acid in many sub-sequences, and it was present as the C-terminal amino acid in 50 out of 150 peptides. Our simple clustering analysis could be potentially replaced by more advanced software packages, such as MUltiple Specificity Identifier (MUSI) [31], which was designed to identify distinct families of consensus sequence motifs within deep sequencing data. The analysis could potentially identify conserved peptide motifs emerging as the results of growth-induced selection.

# 7. Conclusions and future directions

Illumina sequencing, for the first time has uncovered a strong amplification bias to a small number of sequences. The scale at which this bias is visible is difficult to attain by other next-generation sequencing techniques. The reason for this bias remains unknown, but we strongly believe that the bias results from growth preferences of individual phage. It is unlikely to be the result of simple bias in PCR preparation; the latter bias is unlikely to give abundances of 10,000-fold.

PCR also does not favor specific sequence but rather a class of sequences with specific melting point or specific GC-content [18,19]. The bias we observe is unlikely to be present in the naïve library, which should contain up to $10^{9}$ clones according to the manufacturer (New England Biolabs). Indeed, sequencing of naïve
(non-amplified) libraries demonstrated that there is little bias towards specific sequences in the library [14].

Deep sequencing of phage libraries also leaves a few open questions. One of them is general error analysis of random libraries. A growing body of literature confirms that a large number of errors are present in the Illumina results [28], but reliable identification of errors in random libraries is not trivial. The other unexplained observation is the dramatic abundance of reverse reads when compared to forward reads (Fig. 2). The preparation based on dsDNA should give equal number of forward and reverse strands; the reason for the observed bias towards reverse strands is unclear.

It is unlikely that the reads are lost in the analysis because our analysis maps account for mutations and frame shifts of constant primer regions and, thus, can map up to $>99\%$ of reads. We hypothesize that hybridization to Illumina chip and on-chip sequencing might be biased towards one read (or one type of DNA sequence). On-chip sequencing is known to discriminate against specific classes of sequences and introduce specific errors (frame shifts, etc.) [28]. The analysis of sequence bias in different reads and comprehensive error analysis will be described in our subsequent manuscript.

Overall, we foresee that Illumina sequencing and analysis similar to the one outlined in this manuscript will provide many advantages to the analysis of phage-display screens. Furthermore, analysis of the biological origin of sequences emerging from amplified libraries will enable identification of a mechanism that promotes or interferes with selection of useful binding sequences in phage display.

# Acknowledgements

The authors thank Christian Daly and Claire Reardon at Harvard FAS sequencing center for helpful discussions. This work was supported by Alberta Glycomics Centre, University of Alberta Startup Funds, NSERC Discovery Grant (402511-2011), Canadian Founda

tion for Innovation New Leaders Opportunity Grant, and DARPA InfoChemistry award (to R.D.).

# Appendix A. Supplementary data

Supplementary data associated with this article can be found, in the online version, at http://dx.doi.org/10.1016/j.ymeth.2012.07.006.

# References

54

W.L. Matochko et al./Methods 58 (2012) 47-55

Kerelska, A.D. Kersey, I. Khrebtukova, A.P. Kindwall, Z. Kingsbury, P.I. Kokko-Gonzales, A. Kumar, M.A. Laurent, C.T. Lawley, S.E. Lee, X. Lee, A.K. Liao, J.A. Loch, M. Lok, S.J. Luo, R.M. Mammen, J.W. Martin, P.G. McCauley, P. McNitt, P. Mehta, K.W. Moon, J.W. Mullens, T. Newington, Z.M. Ning, B.L. Ng, S.M. Novo, M.J. O'Neill, M.A. Osborne, A. Osnowski, O. Ostadan, L.L. Paraschos, L. Pickering, A.C. Pike, A.C. Pike, D.C. Pinkard, D.P. Pliskin, J. Podhasky, V.J. Quijano, C. Raczy, V.H. Rae, S.R. Rawlings, A.C. Rodriguez, P.M. Roe, J. Rogers, M.C.R. Bacigalupo, N. Romanov, A. Romieu, R.K.

Roth, N.J. Rourke, S.T. Ruediger, E. Rusman, R.M. Sanches-Kuiper, M.R. Schenker, J.M. Seone, R.J. Shaw, M.K. Shiver, S.W. Short, N.L. Sizto, J.P. Sluis, M.A. Smith, J.E.S. Sohna, E.J. Spence, K. Stevens, N. Sutton, L. Szajkowski, C.L. Tregidgo, G. Turcatti, S. vandeVondele, Y. Verhovsky, S.M. Virk

W.L. Matochko et al./Methods 58 (2012) 47-55

55