# A comparative analysis of recombinant Fab and full-length antibody production in Chinese hamster ovary cells

Hirra Hussain $^{1,2}$ | Tulshi Patel $^{3,4}$ | Angelica M. S. Ozanne $^{3}$ | Davide Vito $^{3,5}$ |

Mark Ellis $^{6}$ | Matthew Hinchliffe $^{6}$ | David P. Humphreys $^{6}$ | Paul E. Stephens $^{6}$ |

Bernie Sweeney $^{6,7}$ James White $^{6}$ Alan J. Dickson $^{1}$ Christopher M. Smales $^{3,8}$

# Correspondence

Alan J. Dickson, Faculty of Science and Engineering, Department of Chemical Engineering and Analytical Sciences, Manchester Institute of Biotechnology, University of Manchester, M1 7DN Manchester, UK.

Email: alan.dickson@manchester.ac.uk

Christopher M. Smales, Division of Natural Sciences, Industrial Biotechnology Centre and School of Biosciences, University of Kent, Canterbury, Kent CT2 7NJ, UK. Email: C.M.Smales@kent.ac.uk

# Funding information

UCB UK; Biotechnology and Biological Sciences Research Council, Grant/Award Numbers: BB/R001731/1 BB/R002096/1

# Abstract

Monoclonal antibodies are the leading class of biopharmaceuticals in terms of numbers approved for therapeutic purposes. Antigen-binding fragments (Fab) are also used as biotherapeutics and used widely in research applications. The dominant expression systems for full-length antibodies are mammalian cell-based, whereas for Fab molecules the preference has been an expression in bacterial systems. However, advances in CHO and downstream technologies make mammalian systems an equally viable option for small- and large-scale Fab production.

Using a panel of full-length IgG antibodies and their corresponding Fab pair with different antigen specificities, we investigated the impact of the IgG and Fab molecule format on production from Chinese hamster ovary (CHO) cells and assessed the cellular capability to process and produce these formats. The full-length antibody format resulted in the recovery of fewer mini-pools posttransfection when compared to the corresponding Fab fragment format that could be interpreted as indicative of a greater overall burden on cells.

Antibody-producing cell pools that did recover were subsequently able to achieve higher volumetric protein yields (mg/L) and specific productivity than the corresponding Fab pools. Importantly, when the actual molecules produced per cell of a given format was considered (as opposed to mass), CHO cells produced a greater number of Fab molecules per cell than obtained with the corresponding IgG, suggesting that cells were more efficient at making the smaller Fab molecule. Analysis of cell pools showed that gene copy number was not correlated to the subsequent protein production.

The amount of mRNA correlated with secreted Fab production but not IgG, whereby posttranscriptional processes act to limit antibody production. In summary, we provide the first comparative description of how full-length IgG and Fab antibody formats impact on the outcomes of a cell line construction process and identify potential limitations in their production that could be targeted for engineering increases in the efficiency in the manufacture of these recombinant antibody formats.

#

Check for updates

Received: 18 June 2021

Revised: 31 August 2021

Accepted: 12 September 2021

DOI: 10.1002/bit.27944

ARTICLE

BIOTECHNOLOGY BIOENGINEERING

Wiley

This is an open access article under the terms of the Creative Commons Attribution License, which permits use, distribution and reproduction in any medium, provided the original work is properly cited.

© 2021 The Authors. Biotechnology and Bioengineering published by Wiley Periodicals LLC.

Biotechnol Bioeng. 2021;118:4815-4828.

wileyonlinelibrary.com/journal/bit

4815

# KEYWORDS

antibody fragments, cell line construction, CHO cells, recombinant antibodies

# 1 | INTRODUCTION

Monoclonal antibodies and immunoglobulins are an important and prevalent class of biopharmaceuticals with many approved for therapeutic use or in development (Akram et al., 2021; O'flaherty et al., 2020; Walsh, 2018). Between 2014 and 2019, the total number of FDA-approved biopharmaceuticals was 129, out of which 66 (51%) were monoclonal antibodies or conjugates (O'flaherty et al. 2020). Engineering of recombinant proteins alongside bioprocess and host cell improvements has enabled antibody production processes to produce $>10\mathrm{g/L}$ yields in CHO cell cultures (Huang et al., 2010).

In addition, truncated forms of full-length antibodies including antigen-binding fragments (Fab) have also been used in therapeutic applications (reviewed in Sandomenico et al. [2020]). Examples of approved Fab fragments include Certolizumab pegol (CIMZIA®), developed and manufactured by UCB Pharma; a PEGylated Fab fragment from a humanized anti-TNF- $\alpha$ monoclonal antibody for the treatment of rheumatoid arthritis. Ranibizumab (Lucentis®), a humanized Fab fragment, has been approved for the treatment of angiogenesis and macular edema (Sandomenico et al., 2020).

The Fab antibody fragment Idarucizumab (Praxbind®) has also been approved for neutralization of the anticoagulant effect of dabigatran (Walsh, 2018). Applications of this alternative to full-length antibodies continue to grow alongside the development of other antibody fragment-inspired formats including bi-specific and tri-specific antibodies offering different modes of action (Spiess et al., 2015).

For the production of antibodies and antibody-based format molecules, a range of different expression systems has been used. Mammalian expression systems, in particular, Chinese hamster ovary (CHO) cells are the prevalent host for recombinant protein production due to their ability to perform the complex posttranslational modifications required (Walsh, 2018). In contrast, Escherichia coli (E.

coli) is a preferred expression host for Fab molecules due to the simplicity of the process compared to mammalian cells and reduced time-scales to manufacture Fab yields at gram/liter scale (Gupta & Shukla, 2017). However, challenges with bacterial cells include inclusion body formation, high endotoxin levels and, importantly for antibodies, the lack of correct posttranslational processing and modifications, in particular N-linked glycosylation. There is a large amount of literature describing the expression of Fab fragments and/or full-length antibodies in bacterial cells (e.g.

, Dariushnejad et al., 2019; Ellis et al., 2017; Farajnia et al., 2020; Gadkar et al., 2015; Gundinger & Spadiut, 2020; Humphreys et al., 1996; 2007; Kumar et al., 2019; Rodríguez-Carmona et al., 2012; Shatz et al., 2019; Yusakul et al., 2018) or alternative system such as insect and yeast expressions system (e.g., Joosten et al., 2003; Mizote et al., 2020; Nakamura et al., 2020). In comparison, there are limited examples describing Fab fragment expression in mammalian cells, where yields

of up to $4\mathrm{g / L}$ have been reported (Camper et al., 2011; Lebozec et al., 2018, Samuelsson et al., 1996; Schatz et al., 2003; Takagi et al., 2017; Tang et al., 2018; Vazquez-Lombardi et al., 2018). Surprisingly, few studies have compared the production of full-length antibodies and their corresponding Fab molecules in bacterial and/or mammalian cells.

Here, we describe the production of full-length humanized IgG antibodies and antibody-derived Fab molecules using a proprietary CHO cell expression system. A panel of full-length IgG1 and partner Fab constructs were designed with the same variable domain sequences or antigen specificity. We then compared how the IgG1 and Fab formats impacted the cell line construction process, recovery of mini-pools, and protein expression through a standard cell line construction process. The results of the subsequent analyses are described.

# 2 MATERIALS AND METHODS

# 2.1 Cell lines and DNA constructs

Suspension CHO-DG44 cells were grown in commercial DG44 medium (Life Technologies). Separate heavy chain (HC) and light chain (LC) genes were cloned into a propr
ietary vector using standard cloning methods. All HC and LC DNA sequences were commercially synthesized by ATUM and the codon usage was consistent across both IgG and Fab sequences. The plasmid vector topology was such that each individual recombinant gene (HC/LC) for the Fab or IgG1 was driven by a separate promoter of the same type.

For the full-length antibodies, HCs contained the same human IgG1 $C_H1$ , $C_H2$ , and $C_H3$ domain, and the same human kappa LC but with $V_H$ and $V_L$ domains with different antigen specificity termed A, B, and C. Fab molecules contained the same HC $C_H1$ and LC $C_L$ domain with $V_H$ and $V_L$ domains with the corresponding antigen specificity. The number of amino acids in the IgG1 and Fab HC and LC were within the normal range for these formats. Six molecules in total were generated for analysis, IgG1 A, IgG1 B, IgG1 C, Fab A, Fab B, and Fab C.

# 2.2 | Stable cell mini-pool generation

Stable CHO cell pools were generated by transfecting linear DNA, encoding the IgG1 and Fab molecules, into suspension CHO-DG44 cells using the Cell Line Nucleofactor™ V Kit (Lonza) according to manufacturer's instructions. $1 \times 10^{7}$ cells and $100\mu \mathrm{g}$ linearized DNA were used per T-75 flask. $1 \times 10^{7}$ viable CHO-DG44 cells were harvested and centrifuged $(190\times \mathrm{g}, 5\mathrm{min})$ . The supernatant was discarded and cells gently re-suspended in $1\mathrm{ml}$ nucleofactor solution and the appropriate volume of DNA was added and mixed. An equal

4816

WILEY-BIOTECHNOLOGY BIOENGINEERING

HUSSAIN ET AL.

volume of the DNA and cell suspension was transferred into 10 cuvettes. Each cuvette was electroporated (in a Nucleofactor® I device) and nonselective CD-DG44 medium (supplemented with glutamine and Pluronic acid) was added. After electroporation and media addition, the contents of all ten cuvettes were transferred into a T-flask containing pre-warmed nonselective media. The T-flask was incubated at $36.8^{\circ}\mathrm{C}$ , $7.5\%$ $\mathrm{CO}_{2}$ in a static incubator for $24\mathrm{h}$ . In total, two T-75 flasks were prepared per transfection.

At $24\mathrm{h}$ posttransfection, cells were counted and centrifuged at $900\times g$ , 5 min. After centrifugation, the supernatant was discarded and cells re-suspended into selective CD-CHO media (supplemented with glutamine and methotrexate [MTX]) and plated out into 96-well plates. For every transfection, twenty 96-well plates were plated out with 4000 cells seeded per well. The 96-well plates were incubated at $36.8^{\circ}\mathrm{C}$ , $7.5\%$ $\mathrm{CO}_{2}$ in a static incubator for approximately 14 days.

Single colonies (termed a mini-pool) in the 96-well plates were identified using the CloneSelect Imager (Molecular Devices) and transferred into wells of a 24-well plate (one colony per well). For each transfection, up to a maximum of 96 colonies were transferred. The resulting mini-pools were grown in selective CD-CHO media (supplemented with glutamine and MTX) for 10 days until cells were confluent.

After 10 days, $200\mu l$ of culture supernatant was sampled per well to determine the product titer and confirm expression of the molecule of interest using the Octet® QK with Protein G biosensors (FortéBio).

The top-20 ranking mini-pools by titer were transferred into T-25 flasks in selective CD-CHO media (supplemented with glutamine and MTX) for 7 days. Subsequently, all T-25 flasks were transferred into $125\mathrm{ml}$ Erlenmeyer shake flasks in HyClone ActiSM™

medium (supplemented with glutamine and MTX) and incubated in a humidified shaking incubator at $36.8^{\circ}\mathrm{C}$ , $7.5\%$ $\mathrm{CO}_{2}$ , and $198~\mathrm{rpm}$ . Mini-pools were cultured in shake flasks until the culture viability recovered ( $>90\%$ ). For the top-20 mini-pools, a 10-day batch overgrow culture was then performed. After 10 days, the product titer in the culture medium was determined using Protein G High-Performance Liquid Chromatography (HPLC). The top-12 ranking mini-pools by titer were selected for further study and cryopreserved.

# 2.3 | High-performance liquid chromatography analysis

After a 10-day batch overgrow culture, culture supernatants were harvested from each shake flask by centrifugation (5000×g, 20 min). Before analysis, the supernatants were filtered using Steriflip-GP sterile centrifuge tube filter units (Millipore). A 300 μl aliquot of the filtered supernatant was transferred into a Chromacol fixed insert glass vial (Thermo Fisher Scientific) and analyzed on an Agilent Technologies 1200 series HPLC system, using a Protein G column and the Agilent Chemstation (Rev.B.04.03(16)) software was used for analysis.

# 2.4 | Batch culture analysis of the highest expressing mini-pools

200 ml cultures in 1 L shake flasks were set up for the top-12 mini-pools for each molecule of interest for a 9-day batch overgrow culture.

TABLE 1 Summary of the different types of samples taken during the 9-day overgrowth batch culture for the top-12 mini-pools for each molecule of interest

![](dt=2026-03-25/ht=15/3b9d8d7fb71db6bc67280bd3cfbba8ef5e19afe51d0b2561737eb5854bc44090.jpg)

<table><tr><td></td><td>Cell counts</td><td>Product concentration (HPLC)</td><td>DNA (qPCR)</td><td>RNA (qPCR)</td><td>Protein (Western blot analysis)</td></tr><tr><td>Day 0:</td><td colspan="5">Cultures were seeded at 0.2 × 106cells/ml in 200 ml total volume</td></tr><tr><td>Day 1:</td><td>x</td><td></td><td></td><td></td><td></td></tr><tr><td>Day 2:</td><td>x</td><td></td><td></td><td></td><td></td></tr><tr><td>Day 3:</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Day 4:</td><td>x</td><td></td><td></td><td></td><td></td></tr><tr><td>Day 5:</td><td>x</td><td></td><td></td><td></td><td></td></tr><tr><td>Day 6:</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Day 7:</td><td>x</td><td></td><td></td><td></td><td></td></tr><tr><td>Day 8:</td><td>x</td><td></td><td></td><td></td><td></td></tr><tr><td>Day 9:</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Sample details:</td><td>0.6 ml/sample</td><td>2 ml culture medium/sample</td><td>5 × 106cells/sample</td><td>5 × 106cells/sample</td><td>1 × 107cells/sample and culture medium</td></tr></table>

Note: The "X" symbol marks each day a sample was taken for cell counting, titer measurements, DNA, RNA, and protein analysis. In addition, the culture volume or number of viable cells sampled are detailed.

HUSSAIN ET AL.

BIOTECHNOLOGY BIOENGINEERING WILEY

4817

Cell counts were taken every day using a Vi-CELL automated cell counter (Beckman Coulter). Cultures were sampled for culture medium and cell pellets on specific days (detailed in Table 1) for titer measurements (product concentration), DNA, RNA, and protein analyses. Triplicate samples were taken at the appropriate time points.

# 2.5 | Preparation of intracellular and secreted protein samples

From batch cell cultures, $1 \times 10^{7}$ viable cells were harvested by centrifugation $(5000 \times g, 10 \mathrm{~min})$ and the culture supernatant isolated. The cell pellet was lysed directly in radio-immunoprecipitation assay (RIPA) buffer $(125 \mathrm{mM}$ sodium chloride, $10 \mathrm{mM}$ sodium fluoride, $1 \%$ (v/v) Triton-X, $0.2 \%$ (w/v) SDS, $10 \mathrm{mM}$ sodium pyrophosphate, $0.5 \%$ (w/v) sodium deoxycholate, $25 \mathrm{mM}$ HEPES, and $10 \mathrm{mM}$ sodium orthovanadate) supplemented with protease inhibitor cocktail (cat no.

P8340, Sigma-Aldrich) and passed through a syringe and 21-gauge needle. The lysate was then incubated on ice for $15 \mathrm{~min}$ . An aliquot of the lysate was mixed in equal volumes with $2 \times$ sample buffer ( $20 \%$ (v/v) glycerol, $125 \mathrm{mM}$ Tris-HCl, $4 \%$ (w/v) SDS, and $0.01 \%$ (v/v) bromophenol blue) before Western blot analysis. For secreted protein analysis, an aliquo
t of the culture supernatant was mixed in equal volumes with a $2 \times$ sample buffer.

# 2.6 SDS-PAGE and Western blot analysis

Proteins were resolved via SDS-PAGE and transferred onto nitrocellulose membrane as detailed previously (Hussain et al., 2018). Membranes were blocked in $5\%$ (w/v) milk in phosphate-buffered saline (PBS; $137\mathrm{mM}$ NaCl, $2.7\mathrm{mM}$ KCl, $10\mathrm{mM}$ $\mathrm{Na_2HPO_4}$ , $2\mathrm{mM}$ $\mathrm{KH_2PO_4}$ , and pH 7.4) with $0.1\%$ (v/v) Tween-20 ( $5\%$ mPBS-T) for $1\mathrm{h}$ at room temperature before incubation with primary antibodies in $5\%$ mPBS-T solution for $1\mathrm{h}$ at room temperature.

Primary antibodies used include anti-human $\mathsf{C_H1}$ domain antibody (generated in-house by UCB), anti-human kappa antibody (1:1000, cat no. 9230-01, Southern Biotech), and anti-ERK2 (1:1000, cat no. sc-81459, Santa Cruz Biotechnology) as an intracellular sample loading control. Blots were incubated with a suitable IRDye® 800CW secondary antibody (1:15000, LI-COR Biosciences) in $5\%$ mPBS-T solution for $1\mathrm{h}$ at room temperature. The membrane was washed three times for $5\mathrm{min}$ each with $0.1\%$ (v/v) PBS-T after each antibody incubation.

Proteins were detected using Bio-Rad ChemiDoc MP imaging system according to the manufacturer's instructions. Quantification of bands was completed using the Bio-Rad Image Lab™ software (Version 6.0.1). All graphs were prepared and statistical analysis was performed in GraphPad Prism (Version 8.4.0).

# 2.7 RNA extraction

Total RNA was extracted from frozen pellets using the mirVana miRNA Isolation Kit with phenol and eluted in DNase-free water.

The eluted RNA was concentrated and on-column DNase treated using RNA Clean & Concentrator™-25 with Zymo-Spin™ IIC Columns (Zymo Research) and DNase I Set with DNA Digestion Buffer (Zymo Research). After assessing for the presence of contaminating genomic DNA (gDNA) by reverse transcription-quantitative polymerase chain reaction (RT-qPCR), a fraction of the total RNA volume was further treated with Thermo Scientific™ RapidOut DNA Removal Kit (Thermo Fisher Scientific). This RNA, free of gDNA contamination, was used for all subsequent RT-qPCR. RNA quality was checked at each step using a NanoDrop instrument.

# 2.8 | Reverse transcription-quantitative polymerase chain reaction

Power SYBR® Green RNA-to-CT™ 1-Step Kit (Thermo Fisher Scientific) was used following the manufacturer's instructions. $80\mathrm{ng}$ of RNA was used for each reaction. Transcript numbers were quantified using a standard curve.

# 2.9 | Genomic DNA extraction

Genomic DNA was extracted from frozen pellets using the PureLink Genomic DNA Kit (Invitrogen), following the manufacturer's instructions. $25\mathrm{ng}$ of gDNA was used for qPCR analysis using the QuantiFast SYBR Green PCR Kit (Qiagen). Copy number was quantified using a plasmid standard curve and subsequently calculated per cell.

# 3 | RESULTS AND DISCUSSION

# 3.1 | Generation of CHO cell pools expressing full-length IgGs show a lower recovery posttransfection compared to those expressing the Fab fragment format

A panel of three IgG1 and Fab formats (with the same variable domain/antigen specificity) were expressed side-by-side in the proprietary CHO cell expression system. The impact of the different molecule formats (with the same antigen-binding targeting) on cell pool recovery posttransfection including the number of colonies that emerge posttransfection, cell growth, and protein expression of colonies as they were expanded through a standard cell line construction process was then assessed. In addition, we performed a molecular characterization of the industrially standard cell lines generated to assess the capability of cells to process and produce IgG1 or Fab formats and identify potential limitations in the production of such molecules.

For the comparisons, we generated three pairs of molecules that shared the same antigen specificity or variable domain but were either in a Fab or IgG1 format giving a total of six molecules. The three

4818

WILEY-BIOTECHNOLOGY BIOENGINEERING

HUSSAIN ET AL.

antigen specificities were termed A, B, and C (Section 2.1) and did not target any CHO host cell proteins. In each Fab format, a heavy $\mathrm{V_H - C_H1}$ polypeptide chain was covalently paired with a separate Kappa $\mathrm{V_L - C_L}$ polypeptide chain ( $\sim 50\mathrm{kDa}$ ), whereas the IgG1 contains two full-length HCs and two Kappa LCs to form a full-length antibody structure ( $\sim 150\mathrm{kDa}$ ; Feige et al., 2010; Huber et al., 1976). The Fab domains did not contain a free C-terminal cysteine residue (no hinge Fab; Dave et al., 2016).

A further molecular distinction between the molecules was the presence of an $N$ -glycosylation site within the Fc region of the IgG1 format which was absent in the Fab format. No theoretical $N$ -glycan sites were present within the different variable regions.

Vectors were generated for each molecule and following linearisation were stably transfected into CHO cells, a summary of the process is depicted in Figure 1a. Following transfection, cells were plated into 96-well plates as pools and allowed to recover for 2-3 weeks (Section 2.2). At that stage, the 96-well plates were analyzed for the presence of presumed single colonies (here termed "mini-pools"). The percentage number of wells containing mini-pools out of the total number of wells seeded (at the 96-well plate stage) was greater for the Fab format $(5\% -14\%)$ than for the IgG format

$(2\% -4\%)$ , however, the difference was not statistically significant $(p = 0.0970$ ; Figure 1b). Within the general recovery banding observed for each format, differences were observed in the recovery depending on the variable domain sequence.

The greater percentage recovery of the Fab format offered the potential to expand a greater number of mini-pools through to the 24-well stage. Consequently, this allowed for the selection of 96 mini-pools to expand from the Fab formats whereas the expansion of mini-pools from the IgG1 formats was limited due to fewer wells recovering (Figure 1c). Indeed, all (<96) mini-pools that recovered were taken through from each of the IgG1 transfections.

Once mini-pools were at a sufficient cell concentration in 24-well plates they were subjected to a 9-day batch overgrow after which the product concentration in the culture medium was measured by Octet® (Protein G) to determine the proportion of mini-pools that gave a detectable protein concentration (Figure 1d). At this stage, a greater percentage of Fab mini-pools expressed product compared to the IgG1 transfectants (Figure 1d) and, on average, Fab mini-pools expressed significantly higher protein concentrations than their equivalent IgG1 partner (Figure 1e; $p < 0.0001$ ).

From the product concentration measured by Octet®, the 20 highest producing

![](dt=2026-03-25/ht=15/9441d1649f124e3a06561fb453d04b38c56de0ce3a8405415d38fc7c8c11f93a.jpg)

![](dt=2026-03-25/ht=15/0c53687fc607ae97981f13320578f4f6efd3ac5cdcec98d760450cfee2973f67.jpg)

![](dt=2026-03-25/ht=15/ad4b50755a6b8ae2e3d902e486aa365bb7da05f8fe301ecf484e12e3c482aee9.jpg)

![](dt=2026-03-25/ht=15/a8ec916a65ab0e1949717bdbea8839bc591b23f4cccbb82e30fbe27d7cfc1937.jpg)

![](dt=2026-03-25/ht=15/915a7955617288c373ee9e2bbce8276f291808aa0980ec5ee538b65de288c25b.jpg)

![](image)
)

![](dt=2026-03-25/ht=15/feb06cef8d31e858d58ac31076d7823ad81060428c7ca67524933082ee1a24cf.jpg)

![](dt=2026-03-25/ht=15/026c847e10a6f8621fef962e708ad5b936374b60e9a1e7edf744842799b658fa.jpg)

![](dt=2026-03-25/ht=15/a5e219a7857124422054288536796d690b27bd228bb181c432141f601cd6201e.jpg)

![](dt=2026-03-25/ht=15/2902e098d1c81f1e57da764c6d468cf7972e60396a90f81ebf2f2bb58e37330a.jpg)

HUSSAIN ET AL.

BIOTECHNOLOGY BIOENGINEERING WILEY

4819

mini-pools were expanded into $125\mathrm{ml}$ Erlenmeyer shake flasks (Figure 1a). These mini-pools were cultured until they reached predetermined culture viability ( $>90\%$ viability after 3-4 passages) and then subjected to a 9-day batch overgrow culture. Postharvest product concentration in the culture medium was measured by HPLC (Protein G; Figure 1f). The IgG1 mini-pools had significantly higher product concentration than the top-20 Fab mini-pools (Figure 1f; $p = 0.0277$ ).

When comparing between stages of the cell line development process, at the 24-well stage on average the Fab mini-pools expressed significantly higher protein concentration than the IgG. However, at the shake flask stage for the top-20 mini-pools, the pattern reversed and the IgG1s had significantly higher average protein concentration compared to the counterpart Fab mini-pools (Figure 1e,f).

Clear differences were observed in the number of mini-pools that recover posttransfection between the Fab and IgG1 formats (also for the average product concentrations for both formats at different stages of the cell line development process). Taken together, the recovery data and early stage cell line development process data suggest that the full-length IgG1 molecule imposes a different "burden" on transfected cells but this does not ultimately impact product concentration in terms of the mass of protein produced during later shake flask expression.

# 3.2 | Analysis of IgG and Fab formats reveals differences between product yield by mass and the number of molecules produced by cells

Given the molecular size differences, we calculated protein concentration on a molar basis as the number of molecules produced for all the Fab and IgG1 mini-pools at the 24-well (Figure 2a) and shake flask stages (Figure 2b) of cell line development. For the 24-well plate data, the number of molecules produced by Fab mini-pools was twofold greater than that for IgG1 mini-pools, a finding that correlates with the product concentration determined by Octet®

(Figure 1e). At the shake flask stage, Fab molecule production was two times that observed for the "partner" IgG1 mini-pools (Figure 2b). This contrasts with the interpretation based on product concentration where the concentration of IgG1 was greater than the Fab-expressing mini-pools (Figure 1f). The results suggest that although differences are observed in product concentration (mg/L) throughout the cell development process, more Fab molecules are consistently produced by cells at all process stages. Therefore, the data questions what measurements are appropriate to describe cell productivity and to compare the productivity of the different formats.

The product concentration is a measure of the total raw mass of material produced (grams per liter), whereas, the molar analysis or molecule productivity measures the number of molecules of product (this can be per cell or volume). Depending on the measurement used, the product concentration or molar analysis, a different perspective is obtained from the data in terms of productivity between different formats. The molar analysis allows discrimination between the differences in molecular weight between formats and provides a measure of the number of molecules a cell can produce.

This is perhaps a better descriptor of the efficiency of cells to fold, assemble and secrete molecules of different formats than raw mass per unit volume and so may have important utility when performing academic studies. We, therefore, suggest that traditional mass-based comparisons of the efficiency by which cells produce different format molecules can be misleading in terms of establishing whether a particular format is more or less easy to express than a comparator and a molecular basis is more revealing in this regard.

# 3.3 | High-throughput screening reveals no predictive indicators of recovery and high-producing mini-pools between molecule formats and cell culture stages

We also investigated potential predictive indicators of successful recovery from transfection of the different formats. The efficiency of transfection and the percentage of wells that gave successful

![](dt=2026-03-25/ht=15/86a1a6459bb752d9f8cd7e15085799c98be82d788af9a03b044874488f2aeac3.jpg)

![](dt=2026-03-25/ht=15/168958cf6d45fb227a67b212442d99299819bca8e9cec2cf02bb97a66996ba81.jpg)

4820

WILEY-BIOTECHNOLOGY BIOENGINEERING

HUSSAIN ET AL.

mini-pool growth were assessed by measuring colony size at the 96-well plate stage. No correlation was observed between the size of the mini-pool at the 96-well plate stage and the product concentration at the 24-well stage and the final performance at the shake flask stage.

The data for the highest expressing mini-pools at the shake flask stage was traced back to the 24-well plate stage (Figure 1a,e,f) to determine if there was any predictability between the different cell culture scales, and whether high performing mini-pools could be detected earlier in the process as others have published (Davies et al., 2013; Porter, Dickson, et al., 2010; Porter, Racher, et al., 2010; Poulain et al., 2019; Rouiller et al., 2016).

Examination of the data revealed that the rank was not preserved between the stages for any of the mini-pools (Figure S1), confirming that screening is necessary at each stage to identify high-producing mini-pools of either format.

# 3.4 | The molecular format and variable region sequence impact the observed cell pool characteristics

The top-12 mini-pools (based on product concentration) were taken forward for detailed analysis of cell pool characteristics such as growth, product concentration, and cell-specific productivity (qP) over a 9-day batch culture (Figure 1a) in shake flasks. The growth profiles for all mini-pools were alike, reaching similar maximum viable cell concentrations $(7 - 9 \times 10^{6}$ cells/ml; Figure S2). In most cases, the culture viability began to drop after Day 6 of culture, however with a number of mini-pools for Fab B and IgG1 B, the culture viability started to decrease earlier in culture. The fact that this was the case with both formats with the B variable region specificity, suggests potential cytotoxic effects arising from the molecular features of this specificity-region sequence.

To ascertain whether the extent of protein production had effects on cell growth, specific growth rate (in the exponential phase) and doubling time were calculated (Table S1). No significant difference was observed between the specific growth rate of the matching pair of molecules of the same variable region specificity $(p = 0.0753 - 0.6961)$ .

The Fab C mini-pools showed a negative correlation ( $R^2 = 0.511$ , $p = 0.009$ ; Figure S3) between specific growth rate and qP, but no other set of mini-pools showed a statistically significant correlation ( $p = 0.1886 - 0.5958$ ; Figure S3). The average cell diameter of all top-12 mini-pools for each format showed the same overall trend (decrease) across the 9-day batch culture with no clear differences between A,
B, and C mini-pools (Figure S4). Taken together, the cell growth data and cell characteristics examined revealed no defining features between mini-pools expressing either a Fab or IgG1 format, as well as, molecules with a specific variable region.

For all the Fab and IgG1 top-12 mini-pools, the cell growth data and product concentration were used to calculate the qP over the entire 9-day batch culture (Figures 3 and S5). Comparing the range in qP between all 12 mini-pools for each molecule, the Fab mini-pools displayed a narrow range of qP (1.9- to 3.8-fold range; Figures 3

and S5) whereas IgG1 pools had a wider range in qP compared to the Fab mini-pools (2.5- to 8.8-fold range; Figures 3 and S5). Values for qP for the IgG1 mini-pools were in line with published literature (Kunert & Reinhart, 2016; LE Fourn et al., 2014; Mason et al., 2012). The average specific productivity for the top-12 mini-pools for the Fab format was significantly lower than the corresponding IgG1 format ( $p \leq 0.0101$ ; Figure 3a). Though the qP of the Fab mini-pool were lower than the corresponding IgG1, the number of molecules produced per cell day (molecules/cell/day) were significantly greater in Fab A and Fab C mini-pools compared to IgG1 A and IgG1 C mini-pools, respectively ( $p \leq 0.0101$ ; Figure 3b).

Variable region sequence appeared to have a stronger influence on qP than the molecular format but the ranking of variable domain sequences was not preserved between Fab and IgG formats. For the Fab mini-pools, the average qP ranking was $C > A > B$ whereas for the IgG1 mini-pools the ranking was $C > B > A$ . The IgG1 C and Fab C mini-pools consistently had the highest qP and number of molecules produced (Figure 3). As a result, they were subjected to further study to investigate molecular processes that might determine the efficiency of recombinant protein production in these industry-grade high-producing mini-pools.

# 3.5 | Assessment of secreted and intracellular IgG and Fab protein

Fab C and IgG1 C mini-pools were sampled for secreted (Figures 4 and 5) and intracellular protein (Figure S6) on Day 6 of culture to assess the status of the protein and assembly of the different format molecules. A subset of mini-pools with a range of qP across those available (indicated as relative to each other low, mid and high) was selected for protein analysis (8 out of 12 mini-pools; Figure S5).

Secreted HC (detected with anti-human $\mathsf{C_H1}$ domain antibody; Figure 4a) and LC (detected with anti-human kappa antibody; Figure 4b) were examined under reducing Western blot conditions for all Fab C and IgG1 C mini-pools. Between the top-12 mini-pools, there were no apparent differences in the amounts of each polypeptide chain (Figure 4c,d) and there was no evidence of protein degradation. As expected, more HC and LC were detectable in culture medium samples than in cell lysates (Figure S6).

There was no evidence of material being retained within the cell and no obvious difference was observed in protein quality and integrity. Thus, there was no evidence that lower cell productivity was associated with the production of aberrant protein species.

To further assess the ability of cells to correctly fold and assemble the target molecules, and to examine if the efficiency of this process was affected by the format, the protein samples were analyzed under nonreducing Western blot conditions (Figure 5). The Fab C protein was detected predominantly at its expected molecular weight ( $\sim 50\mathrm{kDa}$ ) with both anti-HC (Figure 5a) and anti-LC (Figure 5b) antibodies with no difference in the protein species present. As expected for the IgG1 C mini-pools, multiple HC and LC intermediates were detected compared to the Fab mini-pools

HUSSAIN ET AL.

BIOTECHNOLOGY BIOENGINEERING WILEY

4821

![](dt=2026-03-25/ht=15/404eccfdc468b713c5df7555a8d3cc457cea34b180474729ce269eefa1b6cd96.jpg)

![](dt=2026-03-25/ht=15/25541715c6b501ffa0e503b1576f5f99cb0ebdd0f43572322d74c944fb07847b.jpg)

(Figure 5). The larger number of IgG1 intermediates is likely due to the increased posttranslational processing of this molecule with HCs of greater size and complexity. Complex banding patterns have generally been attributed substantially to incomplete intermolecular disulfide bond formation (Peters et al., 2012). Generally, the type and abundance of intermediates detected in the IgG1 samples were similar between mini-pools regardless of the qP ranking with the exception of the bottom two ranking IgG1 mini-pools.

The second-lowest ranked IgG1 C mini-pool (rank 11) contained no LC dimer (highlighted by an asterisk, Figure 5b). In addition, the IgG1 C mini-pool at rank 12 showed the presence of an unknown intermediate containing both HC and LC in culture medium samples (highlighted by an arrow, Figure 5). This IgG1 C intermediate was not

detected within cell lysates (data not shown). The data suggests the highest producing mini-pools for both formats generally process the proteins similarly.

# 3.6 | The amount of messenger RNA correlates to Fab production whereas posttranscriptional processes may limit IgG1 production

As the protein analyses revealed little difference between mini-pools for each format, both formats were analyzed for genomic DNA (gDNA) copy number and messenger RNA (mRNA) transcript number. The aim of this was to investigate whether gDNA and mRNA copies

4822

WILEY-BIOTECHNOLOGY BIOENGINEERING

HUSSAIN ET AL.

![](dt=2026-03-25/ht=15/b8a409bf262497ca651a10aa0894f99008c093a490ba69693eb3bfeacdcebce5.jpg)

![](dt=2026-03-25/ht=15/e3fa2b7e82f716314ae27895dc112c62d8dbbca6de1bd3c158e11c0e802a981d.jpg)

![](dt=2026-03-25/ht=15/b074d8df8c851ab450011768e5528870ddd3ebdfd804807ee184c681fc656f38.jpg)

![](dt=2026-03-25/ht=15/99a2b45a1cc9a6895c4893d3b0ee7739bc138aaf854d7c1851d34939d232bee0.jpg)

might define protein production for IgG1 and Fab formats and identify potential bottlenecks along the production pathway.

Genomic DNA was extracted from frozen pellets collected on day 6 of culture and the copy number of light and HC per cell was determined for Fab C and the IgG1 C samples (Figure S7) on a subset of pools (designated low, mid, and high qP, 6 out of 12 mini-pools; Figure S5). We note that when transfecting either the full-length IgG1 or Fab constructs into cells with the same amount of linearized plasmid DNA was used, and as a result, a different number of copies of IgG1 and Fab being transfected.

However, the nature of transfection means we cannot determine how many copies of a given plasmid are transfected into any of the recovering cells and thus it is not possible to comment on any relationship between transfected copy numbers and mini-pool recovery or cell productivity.

What we can deduce is the number of genomic integrated copies of each gene in the pools for the Fab C and IgG C pair which shows that between 2 and 10 HC and LC copies (y-axis; Figure S7) were integrated into five out of six of the Fab and IgG1 mini-pools evaluated and thus these were broadly equivalent (Figure S7). Further, for the Fab C and IgG C mini-pools, there was no correlation between the LC and HC gDNA

copies per cell and the number of molecules/cell/day (Figure S7). Other studies have also reported that cell productivity does not correlate to gene copy numbers (Balasubramanian et al., 2018; Barnes & Dickson, 2006; Lattenmayer et al., 2007; Reisinger et al., 2008; Sergeeva et al., 2020). The lack of correlation betwe
en gDNA copy number and the number of molecules/cell/day is most likely to be a result of the random integration of the DNA within the genome of the cell. As the DNA is not targeted to a specific site within the genome, integration at random locations can cause differences in gene expression due to site-specific differences in transcription rate (Barnes et al., 2003, 2004; Wilson et al., 1990).

Next, the mRNA copy number for the same mini-pools were examined. The mRNA copies of the LC and HC were also determined from pellets collected on Day 6, for Fab C and IgG1 C samples (Figures 6 and S8). Initially, we checked if the gDNA copies correlated with the mRNA copies. For the Fab C samples (Figure S8a,b), there was no correlation between the copies of gDNA and copies of LC or HC mRNA copies ( $p = 0.5457$ and 0.2277, respectively). The lack of correlation between gDNA and mRNA copy numbers again suggests that random integration is a major

HUSSAIN ET AL.

BIOTECHNOLOGY BIOENGINEERING WILEY

4823

![](dt=2026-03-25/ht=15/81d0d4242af9455f07bd0106fbb3a0f85d1c783f782fe8d8524ccf20437edab8.jpg)

![](dt=2026-03-25/ht=15/96c63a0af8fcfb71a188120ca4b18e9a547f3656105c44e76dd52874938928dd.jpg)

factor in determining the mRNA expression. Similarly, for the parallel IgG1 C samples, there was no clear correlation between the gDNA copies of LC or HC and their corresponding mRNA copy number ( $p = 0.0775$ and $0.0689$ , respectively; Figure S8d,e). The ratio of mRNA HC to LC was also examined in relationship to the molecules/cell/day observed and were found to lie within a very narrow range of 1.96-3.84 (Figure S8c,f), and there was no statistical correlation between ratio and number of molecules produced ( $p = 0.1320$ and $0.2133$ , respectively).

The mRNA data were further analyzed to determine if the LC and HC transcript numbers correlated (Figure 6a,b). For both the Fab C and IgG1 C, there was a strong correlation between LC and HC mRNA copies ( $R^2$ values of 0.780 and 0.761, respectively). This may have been expected since LC and HC are encoded on the same vector. For the Fab C mini-pools, the number of molecules/ cell/day positively correlated with LC and HC and dihydrofolate reductase (DHFR) mRNA (Figure 6c-e).

On the other hand, there was no correlation for IgG1 between the number of molecules/ cell/day and LC, HC, or DHFR mRNA (Figure 6f-h). This infers that, at least in the panel of pools investigated here, mRNA amounts are a limiting factor for Fab production in CHO cells but not for the IgG1 format. This suggests that for the Fab molecule LC and HC mRNA are not saturated and thus enhancing mRNA amounts for the Fab format could further enhance yields. This could be achieved by using stronger promoters or site-directed

integration to highly transcriptionally active sites in the genome. Bottlenecks at the mRNA level have previously been reported for other molecules (Godfrey et al., 2017; Jiang et al., 2006; Lee et al., 2009; Mason et al., 2012; Mead et al., 2009; Pekle et al., 2019). For IgG1, posttranscriptional processes must limit product yields as reported in the literature for other recombinant targets (Chusainow et al., 2009; Godfrey et al., 2017; Hussain et al., 2017; Ley et al., 2015; Mason et al., 2012; Mead et al., 2009, 2015; Mohan et al., 2008; O'callaghan et al., 2010; Reisinger et al., 2008).

Finally, we performed a comparison on variable regions A, B, and C in both Fab and IgG formats of mRNA versus molecular productivity (Figure S9). Whilst it is difficult to make broad conclusions when studying three variable regions only, some interesting observations are apparent. First, LC and HC mRNA to molecules per cell/day correlations (Figure S9a,b,d,e) are very different to those of DHFR mRNA (Figure S9c,f). This is perhaps not unsurprising, since mini-pools were selected for molecule production, not differential DHFR activity.

Second, variable region B (blue triangles) sit discretely away from variable regions A (red squares) and C (green circles) in both Fab and IgG formats when correlating with both LC and HC mRNA (Figure S9). Whilst we do not know the drivers behind these differences, it does underline the central influence of variable regions on molecular and cellular biology aspects of antibody cell line production processes.

4824

WILEY-BIOTECHNOLOGY BIOENGINEERING

HUSSAIN ET AL.

![](dt=2026-03-25/ht=15/5a00e8a7007600d10c16cf58222e29478807f40652a813fdcffbd50781e830f9.jpg)

![](dt=2026-03-25/ht=15/60a7c86635b165f15e7d00eae72206aa269983c3a2197d7772f4c2a771006533.jpg)

![](dt=2026-03-25/ht=15/e9983925dfc70469b73aed207e64db15cc9133b5b07e84861a72ff97a5fcc57d.jpg)

![](dt=2026-03-25/ht=15/ea6c1823d49a4b7e07beb86033f4a6c0e94bc7733baa6f3e967be3f12ebae19c.jpg)

![](dt=2026-03-25/ht=15/77e204f16ad3edd4f28147aa26d520c49e5e23a62447a9d70fd4941aed6b892f.jpg)

![](dt=2026-03-25/ht=15/73d5f342846dc97367d992ebfdc89c935e1251fac6637eac4b3bf02ad4a7f4f8.jpg)

![](dt=2026-03-25/ht=15/54013882f08719e5131e35101123e9d47766700ab434a115d8e03270a83a6bef.jpg)

<table><tr><td colspan="4">Pearson Correlation Summary (R2) - mRNA Analysis</td></tr><tr><td></td><td>GOI</td><td>R2Value</td><td>P-value</td></tr><tr><td rowspan="4">Fab C</td><td>Light Chain</td><td>0.5256</td><td>0.1030</td></tr><tr><td>Heavy Chain</td><td>0.7149</td><td>0.0340</td></tr><tr><td>DHF R</td><td>0.6217</td><td>0.0624</td></tr><tr><td>Heavy Chain vs Light Chain</td><td>0.7800</td><td>0.0197</td></tr><tr><td rowspan="4">IgG1 C</td><td>Light Chain</td><td>0.0373</td><td>0.7139</td></tr><tr><td>Heavy Chain</td><td>0.0077</td><td>0.8691</td></tr><tr><td>DHF R</td><td>0.0363</td><td>0.7175</td></tr><tr><td>Heavy Chain vs Light Chain</td><td>0.7608</td><td>0.0234</td></tr></table>

Values underline and bold have a P-value $< 0.05$

![](dt=2026-03-25/ht=15/54f945623002d570e4884af03a3276e7d8c501319859f8531827d15850068451.jpg)

![](dt=2026-03-25/ht=15/5f678f50a1cccd034b267c561534c45d692fc08271677a310da7d624103854e5.jpg)

# 4 SUMMARY

This study illustrates the different influences of different variable regions and antibody formats on a CHO cell line construction and molecule production process. During the cell line construction process, the full-length antibody format resulted in fewer mini-pools recovered posttransfection compared to the corresponding Fab molecule. However, later in the cell line construction process, the antibody-producing cell pools achieved higher protein yields (mg/L) and specific productivity compared to the corresponding Fab pools.

Conversely, the number of molecules produced per cell was greater for the Fab molecule compared to the IgG suggesting that the cells are more efficient at making Fab molecules than the larger full-length IgG. Further, we have established the potential for significant Fab expression (specific productivity of up to 11 pg/cell/day) in CHO cells with no evidence of aberrant protein species. Molecular analysis of the cell pools (genomic DNA, messenger RNA, as well as, intracellular and secreted protein analysis) showed that production of each format was not limited t
o a single locus.

The Fab format was limited in the amount of mRNA and/or posttranscriptional processes, whereas, posttranslational processes were limiting for the IgG format. Finally, antibody variable region sequences were shown to influence the cell line construction process, including the

recovery of cell pools after transfection and the cell-specific productivity for each format.

# ACKNOWLEDGMENTS

This study was funded by support from a BBSRC LINK grant awarded to AJD (BB/R002096/1, University of Manchester) and CMS (BB/R001731/1, University of Kent) and funding supplied by UCB Pharma.

# CONFLICT OF INTERESTS

The authors declare that there are no conflict of interests.

# AUTHOR CONTRIBUTIONS

Alan J. Dickson, Christopher M. Smales, Paul E. Stephens, and Bernie Sweeney conceived the study. Alan J. Dickson, Christopher M. Smales, Mark Ellis, Matthew Hinchliffe, David P. Humphreys, Paul E. Stephens, Bernie Sweeney, and James White supervised the study. Hirra Hussain, Tulshi Patel, Angelica M. S. Ozanne, and Davide Vito designed and completed the experiments. Hirra Hussain and Angelica M. S. Ozanne wrote the manuscript with the support of Alan J. Dickson, Christopher M. Smales, Mark Ellis, and David P. Humphreys. All authors reviewed and approved the manuscript before submission.

HUSSAIN ET AL.

BIOTECHNOLOGY BIOENGINEERING WILEY

4825

# DATA AVAILABILITY STATEMENT

The authors confirm that the data supporting the findings of this study are available within the article (and/or) its supplementary materials.

# ORCID

# REFERENCES

4826

WILEY-BIOTECHNOLOGY BIOENGINEERING

HUSSAIN ET AL.

HUSSAIN ET AL.

BIOTECHNOLOGY BIOENGINEERING WILEY

4827

Yusakul, G., Sakamoto, S., Tanaka, H., & Morimoto, S. (2018). Improvement of heavy and light chain assembly by modification of heavy chain constant region 1 (CH1): Application for the construction of an anti-paclitaxel fragment antigen-binding (Fab) antibody. Journal of Biotechnology, 288, 41-47.

# SUPPORTING INFORMATION

Additional supporting information may be found in the online version of the article at the publisher's website.

How to cite this article: Hussain, H., Patel, T., Ozanne, A. M. S., Vito, D., Ellis, M., Hinchliffe, M., Humphreys, D. P., Stephens, P. E., Sweeney, B., White, J., Dickson, A. J., & Smales, C. M. (2021). A comparative analysis of recombinant Fab and full-length antibody production in Chinese hamster ovary cells. Biotechnology and Bioengineering, 118, 4815-4828. https://doi.org/10.1002/bit.27944

4828

Wiley-BIOTECHNOLOGY BIOENGINEERING

HUSSAIN ET AL.