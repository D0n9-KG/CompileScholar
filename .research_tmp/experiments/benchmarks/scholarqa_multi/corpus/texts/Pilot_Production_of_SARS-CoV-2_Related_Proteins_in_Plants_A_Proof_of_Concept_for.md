# OPEN ACCESS

# Edited by:

Linda Avesani,

University of Verona, Italy

# Reviewed by:

Silvana Petruccelli,

National University of La Plata,

Argentina

Hugh S. Mason,

Arizona State University,

United States

Karen Ann McDonald,

University of California,

Davis, United States

Marcello Donini,

Italian National Agency for New

Technologies, Energy and

Sustainable Economic Development

(ENEA), Italy

# \*Correspondence:

Diego Orzáez

dorzaez@ibmcp.upv.es

$^{†}$ These authors have contributed

equally to this work

# Specialty section:

This article was submitted to

Plant Biotechnology,

a section of the journal

Frontiers in Plant Science

Received: 30 September 2020

Accepted: 02 December 2020

Published: 23 December 2020

# Citation:

Diego-Martin B, González B,

Vazquez-Vilar M, Selma S,

Mateos-Fernández R, Gianoglio S,

Fernández-del-Carmen A and

Orzáez D (2020) Pilot Production of

SARS-CoV-2 Related Proteins in

Plants: A Proof of Concept for Rapid

Repurposing of Indoor Farms Into

Biomanufacturing Facilities.

Front. Plant Sci. 11:612781.

doi: 10.3389/fpls.2020.612781

# Pilot Production of SARS-CoV-2 Related Proteins in Plants: A Proof of Concept for Rapid Repurposing of Indoor Farms Into Biomanufacturing Facilities

Borja Diego-Martin $^{†}$ , Beatriz González $^{†}$ , Marta Vazquez-Vilar, Sara Selma, Rubén Mateos-Fernández, Silvia Gianoglio, Asun Fernández-del-Carmen and Diego Orzáez $^{*}$

Instituto de Biología Molecular y Celular de Plantas (IBMCP), Consejo Superior de Investigaciones Científicas (CSIC) - Universidad Politécnica de Valencia (UPV), Valencia, Spain

The current CoVid-19 crisis is revealing the strengths and the weaknesses of the world's capacity to respond to a global health crisis. A critical weakness has resulted from the excessive centralization of the current biomanufacturing capacities, a matter of great concern, if not a source of nationalistic tensions. On the positive side, scientific data and information have been shared at an unprecedented speed fuelled by the preprint phenomena, and this has considerably strengthened our ability to develop new technology-based solutions. In this work, we explore how, in a context of rapid exchange of scientific information, plant biofactories can serve as a rapid and easily adaptable solution for local manufacturing of bioreagents, more specifically recombinant antibodies. For this purpose, we tested our ability to produce, in the framework of an academic lab and in a matter of weeks, milligram amounts of six different recombinant monoclonal antibodies against SARS-CoV-2 in Nicotiana benthamiana. For the design of the antibodies, we took advantage, among other data sources, of the DNA sequence information made rapidly available by other groups in preprint publications. mAbs were engineered as single-chain fragments fused to a human gamma Fc and transiently expressed using a viral vector. In parallel, we also produced the recombinant SARS-CoV-2 N protein and the receptor binding domain (RBD) of the Spike protein in planta and used them to test the binding specificity of the recombinant mAbs. Finally, for two of the antibodies, we assayed a simple scale-up production protocol based on the extraction of apoplastic fluid. Our results indicate that gram amounts of anti-SARS-CoV-2 antibodies could be easily produced in little more than 6 weeks in repurposed greenhouses with little infrastructure requirements using N. benthamiana as production platform. Similar procedures could be easily deployed to produce diagnostic reagents and, eventually, could be adapted for rapid therapeutic responses.

Keywords: COVID-19, SARS-CoV-2, Nicotiana benthamiana, plant-made antibody, plant-made antigen, biofactories, molecular farming

# INTRODUCTION

The current pandemic is evidencing several weaknesses in our ability to respond to a global crisis, one of which is the insufficient and heavily centralized distribution of the world manufacturing capacity of bioproducts such as antibodies, vaccines, and other biological reagents, especially proteins. Since it is economically impracticable to ensure readiness by maintaining inactive infrastructures during large periods of normality, the development of dual-use systems has been proposed, which would serve regular production needs in normal times but could be rapidly repurposed to strategic manufacturing requirements in times of crisis. Ideally, such adaptable infrastructures should be widespread to serve local demand in case of emergency.

Recombinant protein production in plants is a technologically mature bioengineering discipline, with most current plant-based bioproduction platforms making use of non-food crops, mainly the Nicotiana species tabacum and Nicotiana benthamiana as biomanufacturing chassis (Moon et al., 2019; Capell et al., 2020). N. benthamiana is most frequently used in association with Agrobacterium-mediated transient expression, also known as agroinfiltration, a technology that dramatically reduces the time required for product development. Briefly, agroinfiltration consists in the massive delivery of an Agrobacterium suspension culture to the intercellular space of plant leaves, either by pressure, using a syringe (small-scale), or applying vacuum to plants whose aerial parts have been submerged in a diluted Agrobacterium culture (large-scale). Agrobacterium transfers its T-DNA to the cell nucleus, therefore, massively reprogramming the plant cell machinery toward the synthesis of the T-DNA-encoded protein(s)-of-interest. Transient expression of the transgene is often assisted by self-replicating deconstructed virus vectors that amplify the transgene dose, thus boosting protein production by several orders of magnitude (Gleba et al., 2004). Other systems, such as the pEAQ system, rely on viral genetic elements for boosting expression without recurring to viral replication (Sainsbury et al., 2009). Transient expression in N. benthamiana has become the standard in plant-based recombinant protein production due to a unique combination of advantages, with speed and high yield as the most obvious ones. Maximum production levels in the g/Kg fresh weight (FW) range for certain highly stable proteins such as antibodies have been reported (Marillonnet et al., 2005). Regarding speed, the in-planta incubation times required to obtain maximum yield of recombinant protein are no more than 2 weeks.

An important, often insufficiently highlighted feature of N. benthamiana transient expression is its relatively small infrastructure requirements, partially overlapping with those employed in more conventional, medium/high-tech indoors agriculture, such as hydroponics, vertical farming, etc. (Buyel, 2019). In this context, when confronted with the CoVid-19 crisis, we decided to exercise a partial reorientation of the activities in our academic lab, which is equipped with a multipurpose glass greenhouse facility, toward the production of SARS-CoV-2 antigens and antibodies against the virus. Here, we describe the recombinant production, purification, and analysis of six anti-SARS-CoV-2 monoclonal antibodies at laboratory scale, plus a pilot upscaling of two of those six antibodies. Next to production scale, a critical parameter to assess was the response time. The process described here started in mid-April 2020 with the selection of literature-available antibody variable sequences and finalized 9 weeks later with approximately 0.2 g of anti-SARS-CoV-2 antibody (Ab) produced in modular batches of 56 N. benthamiana plants and formulated as 1 L of Ab-enriched plant apoplastic fluid. Based on this experience, we estimate that the same process can be reduced up to 6–7 weeks with small pre-adaptations, a remarkably short-reaction time for a de novo antibody production system. Absolutely key for this fast reaction is the immediate availability of scientific data including antibody sequences in pre-print repositories. This is, in our opinion, one of the most positive lessons that can be extracted from the CoVid-19 crisis. We discuss here the possible applications of the fast plant-produced antigens and antibodies in diagnostics and therapy and propose the repurpose of high-tech agricultural facilities as an alternative for decentralized biomanufacturing in times of crisis.

# MATERIALS AND METHODS

# Plant Material and Growth Conditions

Both N. benthamiana wild type plants and 1,2-xylosyltransferase/alpha1,3-fucosyltransferase ( $\Delta$ XT/FT) RNAi knock down lines (Strasser et al., 2008) were grown in the greenhouse. Growing conditions were 24°C (light)/20°C (darkness) with a 16-h-light/8-h-dark photoperiod.

# Cloning of Anti-SARS-CoV-2 Antibodies and SARS-CoV-2 Antigens

All sequences were cloned and assembled using the GoldenBraid (GB) assembly system $^{1}$ (Sarrion-Perdigones et al., 2013). Antibody sequences were obtained from literature (see Table 1). All antibodies were cloned as single chain antibodies fused to the human IgG1 Fc domain. Those antibodies derived from synthetic or camelid single domain VHH libraries (sybody 3, sybody 17, and nanobody 72) were designed as direct fusions. CR3009, CR3018, and CR3022 human monoclonal antibodies were redesigned as single chain variable fragment (ScFv) by connecting the variable light (VL) and heavy (VH) chains with a GGGGSGGGGSGGGGSSGGGS peptide linker. Antibody sequences were codon optimized for N. benthamiana with the IDT optimization tool at http://eu.idtdna.com/CodonOpt. Here, the optimization is based on the frequencies of each codon usage for each organism. In this direction, the tool algorithm eliminates codons with less than 10% frequency and re-normalizes the remaining frequencies to 100%. Moreover, it reduces complexities that can interfere with manufacturing and downstream expression, such as repeats, hairpins, and extreme GC content.

The SARS-CoV-2 antigen sequences used (N protein, YP\_009724397.2; and S-protein RBD domain, YP\_009724390.1,

TABLE 1 | Detailed information of six different antibodies selected for recombinant expression in plants. 

<table><tr><td>Name</td><td>Type</td><td>Size (with SP)</td><td>Target</td><td>Neutralizing activity</td><td>References</td></tr><tr><td>CR3022</td><td>ScFv-Fc IgG1 (VL-Linker-VH)</td><td>56.2 kDa</td><td>RBD (SARS-CoV and SARS-CoV-2)</td><td>No</td><td>Tian et al., 2020</td></tr><tr><td>sybody3</td><td>VHH-Fc IgG1</td><td>42.4 kDa</td><td>RBD (SARS-CoV-2)</td><td>Unknown</td><td>Walter et al., 2020</td></tr><tr><td>sybody17</td><td>VHH-Fc IgG1</td><td>42.2 kDa</td><td>RBD (SARS-CoV-2)</td><td>Unknown</td><td>Walter et al., 2020</td></tr><tr><td>nanobody72</td><td>VHH-Fc IgG1</td><td>43.3 kDa</td><td>RBD (SARS-CoV and SARS-CoV-2)</td><td>Yes</td><td>Wrapp et al., 2020</td></tr><tr><td>CR3009</td><td>ScFv-Fc IgG1 (VH-Linker-VL)</td><td>55.5 kDa</td><td>Protein N (SARS-CoV)</td><td>Unknown</td><td>van den Brink et al., 2005</td></tr><tr><td>CR3018</td><td>ScFv-Fc IgG1 (VH-Linker-VL)</td><td>54.9 kDa</td><td>Protein N (SARS-CoV)</td><td>Unknown</td><td>van den Brink et al., 2005</td></tr></table>

aa 319-541) derive from the Wuhan strain NC\_045512. Four different versions of RBD were designed corresponding to (i) the native sequence with a C-terminal 6xHis-Tag (nRBD:His) or (ii) an N-terminal 6xHis-Tag and a C-terminal KDEL sequence for ER retention (His:nRBD:KDEL), and (iii and iv) their corresponding N. benthamiana codon optimized counterparts (bRBD:His and His:bRBD:KDEL), using the same tool as above. For N protein, a 6xHis tag was fused to the N-terminus of an N. benthamiana codon optimized sequence (His:bN).

DNA sequences were domesticated as level 0 phytobricks for GB cloning and ordered for synthesis as double-stranded DNA fragments (gBlocks, Integrated DNA Technologies). gBlocks were first cloned into the domestication vector pUPD2 (Vazquez-Vilar et al., 2017) in a BsmBI Golden Gate restriction/ligation reaction [37°C - 10 min, 50x (37°C - 3 min/16°C - 4 min), 50°C - 10 min, 80°C - 10 min]. The ligation product was transformed into E. coli Top 10 electrocompetent cells and positive clones were verified by restriction digestion analysis and sequencing. pUPD2 level 0 phytobricks were then cloned into the expression vectors pGreen SP-hIgG1 (antibody sequences), pCambiaV1 (RBD sequences), or pCambiaV2 (N sequences). pGreen SP-hIgG1 is a pGreen vector-based adaptation of the MagnICON® 3' provector pICH7410 (ICON Genetics; Giritch et al., 2006) that is designed for BsaI cloning of GB (B4-B5) standard parts as in-frame fusions with the tobacco (1-3)-beta-glucanase signal peptide and the human IgG1 Fc domain. Similarly, pCambiaV1 and pCambiaV2 are pCambia based adaptations of the MagnICON® 3' provector pICH7410 that are designed for BsaI cloning of GB standard parts as in-frame fusions with tobacco (1-3)-beta-glucanase signal peptide (pCambiaV1, for expression of secreted proteins) or without any subcellular localization signal (pCambiaV2, for expression of cytoplasmic proteins). Assembly reactions were performed as above, and the ligation reactions were transformed into E. coli Top 10 electrocompetent cells. Positive clones were verified by restriction digestion analysis. All level 0 parts and expression vectors used in this work are listed in Supplementary Table 1.

# Transient Expression in Nicotiana benthamiana

For transient expression in N. benthamiana, the plasmids were transformed into Agrobacterium tumefaciens strain GV3101 C58C1 by electroporation. The same strain but carrying the pSoup helper plasmid was employed to allow the replication of the pGreen vectors, which encode the antibodies. Overnight grown exponential cultures were collected by centrifugation and the bacterial pellets were resuspended in agroinfiltration solution (10 mM MES, 20 mM MgCl2, 200 μM acetosyringone, pH 5.6) and incubated for 2 h at RT in a horizontal rolling mixer. For small-scale agroinfiltration, culture optical density at 600 nm was adjusted to 0.1 with agroinfiltration solution and the bacterial suspensions harboring the 3' antibody or antigen modules, the Integrase (pICH14011), and the 5' module (pICH17388) were mixed in equal volumes. Negative control samples were agroinfiltrated with pICH11599\_DsRed and Integrase module. Agroinfiltration of 5–6-week-old N. benthamiana plants was carried out through the abaxial leaf surface using a 1 ml needle-free syringe (Becton Dickinson S.A.). For pilot-scale production, the bacterial suspensions were prepared as above except that a lower OD $_{600}$ was used (0.005 for sybody17 agroinfiltration and 0.01 for nanobody72 agroinfiltration). Additionally, for sybody17 agroinfiltration, a bacterial suspension of pICH11599\_DsRed, a MagnICON® 3'module encoding the fluorescent protein DsRed, was added to the final Agrobacterium infiltration solution in a ratio 1:1:0.9:0.1 (pICH17388:pICH14011:pGreenSP-Sybody17-hIgG1:pICH11599\_DsRed). Delivery of Agrobacterium to the plant cells was carried out by vacuum infiltration in a vacuum degassing chamber (model DP118, Applied Vacuum Engineering) provided with a 30 L infiltration tank. The aerial part of whole plants (seven plants at a time) was immersed into the Agrobacterium infiltration solution; vacuum was applied for 1 min at a vacuum pressure of 0.8 bar and then slowly released.

# Apoplast Fluid Extraction

Fourteen days post-vacuum agroinfiltration leaves were excised and then infiltrated with 20 mM phosphate buffer (7.4 mM NaH $_2$ PO $_4$ , 12.6 mM Na $_2$ HPO $_4$ .7H $_2$ O, pH 7), without (sybody17) or with (nanobody72) 0.5 mM PMSF (Sigma-Aldrich, #78830), following the same procedure as the vacuum agroinfiltration. After eliminating the buffer excess with tissue paper, the leaves were introduced into mesh zipped bags and then centrifuged at 2800 rpm for 5 min (twice) using a portable cloth dryer (Orbegozo SC4500). Thus, the apoplastic fluid was obtained from the drain tube. The apoplastic fluid was centrifuged (10 min, 11,000 x g, at 4°C) to remove any cell debris and Agrobacterium, the supernatant was collected and then four fractions of 4 ml were concentrated eight times using 10 kDa Amicon Ultra-4 10K Centrifugal filters (Millipore) after centrifugation (20 min, 3,700 x g, at 4°C). The concentration of proteins in the apoplastic fluid was quantified using the

Bio-Rad Protein Assay following the manufacturer's instructions and using BSA for standard curve preparation. The concentration of intact antibodies was estimated by densitometry of Coomassie stained-gel using the software ImageJ. For this, the peak area corresponding to the full-size antibody band was integrated, and its abundance in relation with the remaining bands was estimated.

# Antibody Extraction and Purification

The N. benthamiana leaves infiltrated with the different anti-SARS-CoV-2 recombinant antibodies were collected 7 days post infiltration (dpi). Leaves were frozen in liquid nitrogen and stored at $-80^{\circ}C$ until used. Protein crude extracts were obtained by homogenizing ground frozen leaf tissue with cold PBS buffer (20 mM $NaH_{2}PO_{4}$ , 80 mM $Na_{2}HPO_{4}.7H_{2}O$ , 100 mM NaCl, pH 7.4) in a 1:3 (w/v) and were centrifuged at 13,000 rpm for 15 min at $4^{\circ}C$ . For antibody purification, 4 g of ground agroinfiltrated tissue were extracted in 12 ml of cold 20 mM phosphate buffer. Samples were centrifuged at 10,000 x g for 15 min and the supernatant was transferred to a clean tube and further clarified by filtration through a 0.22 $\mu$ m membrane filter. The recombinant antibodies were purified by affinity chromatography with Protein A agarose resin (ABT Technology) following a gravity-flow procedure according to the manufacturer's instructions. Hundred millimolar citrate buffer pH 3 was used for elution and 1 M Tris-HCl pH 9 was used for neutralization of the eluted sample (37.5 $\mu$ l for each 250 $\mu$ l elution fraction). Purified antibody preparations were quantified using the Bio-Rad Protein Assay following the manufacturer's instructions and using BSA for standard curve preparation. The concentration of intact antibody was estimated by densitometry as described in the previous section.

# Antigen Extraction and Purification

The N. benthamiana leaves infiltrated with the different SARS-CoV-2 proteins were collected 5 (RBD) or 7 (N protein) dpi. Leaves were frozen in liquid nitrogen and stored at $-80^{\circ}$ C until used. Protein extraction was performed using 3–6 g of ground frozen tissue in 1:3 (w/v) extraction buffer. Three different buffers were tested as a first approach in order to optimize the purification yields. Buffer A: PBS buffer with 10 mM imidazole, pH 8. Buffer B: Buffer A supplemented with 1% Triton X-100, and Buffer C: Buffer B supplemented with 20% glycerol, 10% sucrose, and 0.05% 2- $\beta$ -mercaptoethanol. Samples were vigorously vortexed and centrifuged at 10,000 x g for 15 min at 4°C. The supernatant was carefully transferred to a clean tube and filtered through a 0.22 $\mu$ m syringe filter. Protein purification was carried out by Ni-NTA affinity chromatography as described in (Fernandez-del-Carmen et al., 2013). Purified proteins were quantified using the Bio-Rad Protein Assay following the manufacturer's instructions and using BSA for standard curve preparation.

# SDS-PAGE and Western Blot Analysis

Proteins were separated by SDS-PAGE electrophoresis on NuPAGE 10% Bis-Tris Gels (Invitrogen) using MES-SDS running buffer (50 mM MES, 50 mM Tris-base, 3.5 mM SDS, 1 mM EDTA, pH 7.3) under reducing conditions. Gels were visualized by Coomassie blue staining. For Western blot analysis, proteins were transferred to PVDF membranes (Amersham Hybond™-P, GE Healthcare) by semi-wet blotting (XCell II™ Blot Module, Invitrogen, Life Technologies) following the manufacturer's instructions. Blots were blocked with 2% ECL Prime blocking agent (GE Healthcare) in PBS-T [PBS buffer supplemented with 0.1% (v/v) Tween-20]. For anti-SARS-CoV-2 antibody detection, the blots were incubated with 1:20,000 HRP-conjugated rabbit anti-human IgG (Sigma-Aldrich, #A8792). For SARS-CoV-2 antigen detection, the blots were incubated with 1:2,000 Anti-His mouse monoclonal primary antibody (Qiagen, #34660) and then incubated with 1:10,000 peroxidase labeled anti-mouse IgG secondary antibody (GE Healthcare). Blots were developed with ECL Prime Western blotting Detection Reagent (GE Healthcare) and visualized using a Fujifilm LAS-3000 imager.

# Antigen ELISA

The overnight coating of Costar 96 Well EIA/RIA plates (Corning) was carried out at 4°C with 100 μl of 4 μg/ml RBD (RayBiotech, #230-30162), His:bN or BSA (used as control) in Coating buffer (15 mM Na₂CO₃, 35 mM NaHCO₃, pH 9.6). After four washes with 300 μl of PBS, the plate was blocked with 200 μl of a 2% (w/v) ECL Advance Blocking Reagent (GE Healthcare) solution in PBS-T [PBS supplemented with 0.1% (v/v) Tween-20] for 2 h at RT. The plate was washed four times with PBS, and then starting at 2 μg of the purified antibody preparation per well, or 1 μg of the full-size antibody from apoplastic fluid per well (both 100 μl), 1:5 serial dilutions in blocking solution were incubated for 1 h 30 min at RT. After four washing steps with PBS-T (PBS buffer supplemented with 0.05% Tween-20), 1:2,000 HRP-labeled rabbit anti-human IgG (Sigma-Aldrich, #A8792) in blocking solution was added. After 1 h, the plate was washed with PBS and the substrate o-phenylenediamine dihydrochloride SIGMAFAST™ OPD tablet (Sigma-Aldrich, #P9187) was added (following manufacturer's instructions). Reactions were stopped with 50 μl 3 M HCl per well and absorbance was measured at 492 nm. The endpoint dilution titer was determined as the last concentration of each purified antibody showing an absorbance value higher than the value defined as cutoff (mean blank +3SD). Blank is defined as the values from each ELISA test against BSA (Zrein et al., 1986; Armbruster and Pry, 2008).

# Sandwich ELISA

The sandwich ELISAs were performed as described in the antigen ELISA section with a few changes. The plates were coated with 100 $\mu$ l of 4 $\mu$ g/ml murine anti-His mAb (Qiagen, #34660), and after blocking, the plates were incubated with 100 $\mu$ l of the crude extracts of the (nRBD:His/His:bN) antigen expressing leaves. WT crude extracts were used as negative control. The crude extracts were prepared by adding 1:3 (w/v) PBS buffer and then were subjected to sonication. The extract was centrifuged (13,000 rpm, 4°C, and 15 min) and the supernatant was used in the incubation step. The antigens were sandwiched with 1 $\mu$ g of the corresponding purified antibody preparation (100 $\mu$ l in blocking solution) per well (1 h 30 min

incubation, RT). The same procedure as in the antigen ELISA was followed for the incubation with the conjugated secondary antibody, colorimetric reaction, and measurement.

# RESULTS

# Cloning and Expression of Anti-SARS-CoV-2 Recombinant Antibodies

Six different antibody sequences were selected for recombinant production in N. benthamiana, following a plant deconstructed viral strategy based on Magnifection technology as described earlier (Marillonnet et al., 2005; see Table 1). Four of those were directed against the receptor binding domain (RBD) of the SARS-CoV-2 spike (S) protein, whereas the remaining two were directed against the N protein. All six antibodies were engineered as single polypeptide chains fused to the human Cy2-Cy3 constant immunoglobulin domains. Three of them, those derived from single chain camelid or synthetic VHH antibody libraries, were produced as direct fusions. The other three, derived from full-size human monoclonal antibodies, were redesigned as scFvs, using a linker peptide that connects VH and VL regions (see Figure 1A). The nucleotide sequences of the different variable regions were obtained from the literature, then chemically synthesized with appropriate extensions and cloned into a destination Magnifection-adapted vector using a type IIS restriction enzyme strategy. The cloning cassette was flanked by a $\beta$ -endoglucanase signal peptide for apoplastic localization in N-terminal and the human Cy2-Cy3 domains of the human IgG1 in the C-terminal side. The resulting vectors were transferred to Agrobacterium cultures and agroinfiltrated in $N.$ benthamiana leaves in combination with a $5^{\prime}$ MagnICON® module, containing the RNA polymerase and movement protein, and with an integrase module (Figure 1B). For antibody production, we used wild-type and RNAi $\Delta$ XT/FT glycoengineered $N.$ benthamiana plant lines, the latter with a downregulation of plant-specific xylose and fucose glycosylation (Strasser et al., 2008). Infiltrated leaves were examined daily, and only minimal damage was observed in the agroinfiltrated tissues during the incubation period. After 7 days, leaf samples were collected, ground, and crude extracts were analyzed in SDS-PAGE under reducing conditions. As can be observed in Figures 2A,B (upper panel), all samples produced Coomassie-detectable bands of the expected antibody

![](images/f8f09819eb63743236154193d7732d351ef393e3c7fb677fc72b0e194739f689.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_A["Anti-RBD"]
        A1["sybody3, sybody17 & nanobody72"] --> A2["VH"]
        A2 --> A3["Fc CH2"]
        A2 --> A4["Fc CH3"]
        A5["VHH-Fc"] --> A6["CR3022"]
        A7["ScFv-Fc (VL-Linker-VH)"] --> A8["VL"]
        A9["SCFv-Fc (VH-Linker-VL)"] --> A10["VL"]
    end

    subgraph_B["Anti-N"]
        B1["CR3009 & CR3018"] --> B2["VL"]
        B2 --> B3["Fc CH2"]
        B2 --> B4["Fc CH3"]
        B5["ScFv-Fc (VH-Linker-VL)"] --> B6["VL"]
    end

    subgraph_C["Activity Diagram"]
        C1["pUPD2"] --> C2["VHH/ScFv"]
        C2 --> C3["AGCT"]
        C3 --> C4["Bsal"]
        C4 --> C5["GCAG"]
        C5 --> C6["AGCT"]
        C6 --> C7["LacZ"]
        C7 --> C8["GCAG"]
        C8 --> C9["Fc hlgG1"]
        C9 --> C10["NTR"]
        C10 --> C11["tNOS"]
        C11 --> C12["RB"]
    end

    subgraph_D["Functional Environment"]
        D1["pICH17388"] --> D2["pAct2 Ω RdRP MP Intron 5' AttP tNOS RB"]
        D3["pICH14011"] --> D4["pHsp81 ΦC31 Integrase tNOS RB"]
    end

    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
```
</details>

FIGURE 1 | Design for the expression of anti-SARS-CoV-2 recombinant antibodies. (A) Schematic representations of VHH-Fc and ScFv-Fc antibody versions. VHH, heavy chain only single domain antibody; ScFv, single chain variable fragment; Fc, human IgG1 Cγ2-Cγ3 domains. (B) Schematic representation of the cloning procedure for transient expression via MagnICON®. VHHs and ScFvs cloning into the pGreen SP-hlgG1 vector and coinfiltration together with the plasmids pICH17388 and pICH14011 for the in planta viral reconstruction (through ΦC31 mediated PB recombination) and antibody expression. AttB & AttP, site-specific recombination site; SP, signal peptide; NTR, nontranslated region; tNOS, terminator of nopaline synthase; pAct2, actine2 promoter of Arabidopsis thaliana; RdRP, RNA-dependent RNA polymerase; MP, viral movement protein; pHsp81 promoter of the heat shock protein 81.

size. scFv-IgG1 55–56 kDa antibodies migrated slightly above the 50–55 kDa Rubisco large subunit, partially masking its detection. VHH-Fc antibodies migrated at the expected 42–43 kDa size. The identity of the Coomassie bands was confirmed by Western blot using an anti-human IgG1 antibody for detection (Figures 2A,B, lower panel). As shown in Figures 2A,B, under reducing conditions lower molecular weight (MW) bands were also detected, probably as a result of partial proteolytic degradation. Small-scale affinity purification was carried out for all six antibodies produced in $\Delta$ XT/FT plants using protein A affinity chromatography (Figures 2C,D). The concentration of affinity-purified antibody preparations was used to estimate the yield of the final product, which ranged between 73.06 $\mu$ g/g FW (CR3022 antibody) to 192.63 $\mu$ g/g FW (nanobody72 antibody) at most (see Table 2). The percentage of the full-size antibody with respect to the total antibody preparation including cleavage fragments is also shown in Table 2.

A   
![](images/a424307b405eff58de943cf46866eb0717ae432c0a7bc22db35dde50dbc80db8.jpg)

<details>
<summary>text_image</summary>

WT
CR3022 sybody3 sybody17 nanobody72 CR3009 CR3018 Control -
(kDa)
62
49
38
28
(kDa)
62
49
38
28
</details>

B   
![](images/2924e351198b862b2e13b474d85fc30ec0c4a5363bfb59d24252bf3f2487fa45.jpg)

<details>
<summary>text_image</summary>

ΔXT/FT
CR3022 sybody3 sybody17 nanobody72 CR3009 CR3018 Control -
(kDa)
62
49
38
28
(kDa)
62
49
38
28
</details>

C   
![](images/1fad251e9b0add356cd7bd11cd515feed5d6d634927325fa9c9dacf3f077f719.jpg)

<details>
<summary>other</summary>

| Condition       | C.E  | P    |
| --------------- | ---- | ---- |
| Control -       |      |      |
| CR3022          |      |      |
| sybody3         |      |      |
| sybody17        |      |      |
| nanobody72      |      |      |
</details>

D   
![](images/c9379f494930e72161ba5029de315f6054438bbc99e170dbd378867de92d4577.jpg)

<details>
<summary>other</summary>

| Condition | Band 1 (kDa) | Band 2 (kDa) |
|-----------|--------------|--------------|
| Control   | C.E          | P            |
| CR3009    | C.E          | P            |
| CR3018    | C.E          | P            |
</details>

FIGURE 2 | Analysis of the expression of all six anti-SARS-CoV-2 recombinant antibodies produced in N. benthamiana leaves. Coomassie stained SDS gels (upper panel) and Western blots (down panel) of protein extracts from wild-type (A) and $\Delta$ XT/FT (B) N. benthamiana leaves agrotransformed all of them with six antibodies. Seventeen microliter of crude extract was loaded per well in SDS gels. The blots were incubated with an antibody against constant region of human IgG. (C,D) Coomassie stained SDS gels with purified (P) recombinant antibodies from $\Delta$ XT/FT plants compared to their corresponding protein crude extract (C.E) for each analyzed antibody. Twenty microgram of total protein from crude extract per well and 3 $\mu$ g of purified protein per well were loaded in SDS gels. Control samples (−) were agroinfiltrated with Integrase module and pICH11599\_DsRed vectors only, as indicated in section Materials and Methods. Arrows indicate the presence and position of the corresponding band protein for each antibody.

# Cloning and Expression of SARS-CoV-2 Recombinant Antigens

The in-planta production of SARS-CoV-2 RBD and N protein antigens was also assayed in parallel using a similar strategy as described for antibody production. For this purpose, two versions of the expression vector were designed for RBD, one with the native viral sequence and the other with an N. benthamiana codon-optimized sequence. For the N protein, only the N. benthamiana codon-optimized sequence was employed. For RBD, native and codon optimized versions were targeted to the secretory pathway with the tobacco glucan

TABLE 2 | Yield estimation for all six purified recombinant antibodies, RBD and N proteins. 

<table><tr><td>Purified protein</td><td>Yield (μg/g FW)</td><td>% Full-size Ab</td><td>% TSP</td></tr><tr><td>CR3022</td><td>73.06</td><td>5.07</td><td>1.96</td></tr><tr><td>sybody3</td><td>122.53</td><td>56.57</td><td>2.56</td></tr><tr><td>sybody17</td><td>153.36</td><td>83.91</td><td>3.31</td></tr><tr><td>nanobody72</td><td>192.63</td><td>75.95</td><td>3.18</td></tr><tr><td>CR3009</td><td>73.38</td><td>83.97</td><td>1.88</td></tr><tr><td>CR3018</td><td>81.24</td><td>90.19</td><td>1.79</td></tr><tr><td>nRBD:His (Buffer A)</td><td>4.31</td><td>ND</td><td>0.18</td></tr><tr><td>nRBD:His (Buffer B)</td><td>4.02</td><td>ND</td><td>0.051</td></tr><tr><td>bRBD:His (Buffer A)</td><td>2.94</td><td>ND</td><td>0.097</td></tr><tr><td>bRBD:His (Buffer B)</td><td>5.21</td><td>ND</td><td>0.051</td></tr><tr><td>His:bN (Buffer A)</td><td>30.98</td><td>ND</td><td>0.45</td></tr></table>

ND: non determined.

endo-1,3-beta-glucosidase signal peptide (SP:nRBD:His and SP:bRBD:His) and versions containing a KDEL peptide for ER retention (SP:His:nRBD:KDEL and SP:His:bRBD:KDEL) were also generated. All nucleotide sequences were chemically synthesized with a small nucleotide extension coding for a histidine tag for detection (Figure 3A).

As described for antibody production, MagnICON $^{\circledR}$ -derived 3' vector modules encoding RBD and N proteins were agroinfiltrated in combination with an integrase module and a 5'-module lacking any additional subcellular localization signal. Shorter incubation times were decided in antigen production as compared to antibodies because antigen constructs produced different degrees of necrotic lesions in the leaves, ranging from mild symptoms in N protein to severe necrosis after 4 days in native RBD. For those constructs producing more severe lesions, incubation time was reduced to 5 days, and for the rest the incubation period was extended to 7 days. RBD:His from SP:nRBD:His and SP:bRBD:His expressing leaves was extracted and purified using small-scale affinity-chromatography with Ni-NTA columns and the resulting Coomassie and a Western blot analysis, under reducing conditions, are shown in Figure 3B. RBD:His can be detected as an estimated 30 kDa band with the presence of higher MW bands that suggest multimerization and an 18 kDa band probably corresponding to a degradation product. ER retention did not improve expression levels of RBD for the native version, nor for the N. benthamiana optimized one (data not shown). Addition of 1% Triton

![](images/5976bb632ca11664ebe468c6a8084fd48a24aae3f3d128c0565cba78f8ee5753.jpg)  
FIGURE 3 | SARS-CoV-2 RBD and N proteins expression in N. benthamiana agroinfiltrated leaves. (A) Schematic representation of the pUPD2 and 3' module vectors generated for antigen production. (B) Coomassie stained SDS-PAGE (left) and Western blot (right) of crude extracts from non-infiltrated N. benthamiana leaves (Control), leaves infiltrated with viral vectors for the expression of nRBD:His (n stands for native sequence), His:bN and bRBD:His (b stands for N. benthamiana codon optimized sequence) and the corresponding purified proteins.

X-100 to the standard extraction buffer (see Materials and Methods) did not improve the yield, which was estimated as 2–4 $\mu$ g/g FW (Table 2). N protein was extracted from agroinfiltrated leaves and affinity purified following the same procedure described for RBD. A major 49 kDa band was detected both on the crude extract and upon purification (Figure 3B). Small-scale affinity-chromatography with Ni-NTA columns gave an estimated yield of 30 $\mu$ g/g FW for N protein (Table 2).

![](images/07e62192166486ac97dc569348ab3f060feb07935edbba70341c7cbbfc76b4f9.jpg)

<details>
<summary>line</summary>

| Antibody (μg) | CR3022 (RBD) | CR3022 (BSA) | sybody3 (RBD) | sybody3 (BSA) | sybody17 (RBD) | sybody17 (BSA) | nanobody72 (RBD) | nanobody72 (BSA) |
| ------------- | ------------ | ------------ | ------------- | ------------- | -------------- | -------------- | ----------------- | ----------------- |
| 2.00          | 1.8          | 1.2          | 1.8           | 1.2           | 1.2            | 1.2            | 1.0               | 1.0               |
| 0.40          | 1.4          | 0.8          | 1.4           | 0.8           | 1.0            | 0.8            | 0.8               | 0.8               |
| 0.08          | 1.0          | 0.6          | 1.0           | 0.6           | 0.8            | 0.6            | 0.6               | 0.6               |
| 1.60E-02      | 0.6          | 0.4          | 0.6           | 0.4           | 0.6            | 0.4            | 0.4               | 0.4               |
| 3.20E-03      | 0.4          | 0.3          | 0.4           | 0.3           | 0.4            | 0.3            | 0.3               | 0.3               |
| 6.40E-04      | 0.2          | 0.2          | 0.2           | 0.2           | 0.2            | 0.2            | 0.2               | 0.2               |
| 1.28E-04      | 0.1          | 0.1          | 0.1           | 0.1           | 0.1            | 0.1            | 0.1               | 0.1               |
| 2.56E-05      | 0.1          | 0.1          | 0.1           | 0.1           | 0.1            | 0.1            | 0.1               | 0.1               |
</details>

![](images/fb761eec312382ffb0d6ab14abc911d53a5a290b0545f585878415b6951e59c9.jpg)

<details>
<summary>line</summary>

| Antibody (μg) | CR3009 (N) | CR3009 (BSA) | CR3018 (N) | CR3018 (BSA) |
| ------------- | ---------- | ------------ | ---------- | ------------ |
| 2.00          | 1.6        | 1.8          | 1.8        | 0.1          |
| 0.40          | 1.4        | 1.5          | 1.5        | 0.1          |
| 0.08          | 1.2        | 1.3          | 1.2        | 0.1          |
| 1.60E-02      | 1.0        | 1.1          | 0.8        | 0.1          |
| 3.20E-03      | 0.6        | 0.7          | 0.4        | 0.1          |
| 6.40E-04      | 0.3        | 0.4          | 0.2        | 0.1          |
| 1.28E-04      | 0.1        | 0.2          | 0.1        | 0.1          |
| 2.56E-05      | 0.1        | 0.1          | 0.1        | 0.1          |
</details>

![](images/8da0af3f70db26a1940d83231ef0b02056f656917990ac865e9d6fe364dfaa4e.jpg)

<details>
<summary>bar</summary>

| Sample     | RBD crude extract | N crude extract | WT crude extract |
| ---------- | ----------------- | --------------- | ----------------- |
| CR3022     | 0.5               | -               | 0.35              |
| sybody3    | 0.75              | -               | 0.4               |
| sybody17   | 0.6               | -               | 0.3               |
| nanobody72 | 0.5               | -               | 0.25              |
| CR3009     | -                 | 1.4             | 0.35              |
| CR3018     | -                 | 0.9             | 0.35              |
</details>

FIGURE 4 | Comparison of binding activities of anti-SARS-CoV-2 purified antibodies by ELISA tests. (A,B) Antigen ELISA titers graph of anti-RBD (A) or anti-N (B) purified antibody preparations. Binding activities were calculated by absorbance measurements at 492 nm. The X axis indicates the amount ( $\mu$ g) of antibody preparation per well. Solid lines indicate the ELISA test against RBD commercial or our purified His:bN protein, and dashed lines indicate the corresponding control ELISA tests against BSA. Bars represent mean $\pm$ SD, n = 3 independent replicates. (C) Sandwich ELISA plots obtained using anti-RBD and anti-N purified antibody preparations (1 $\mu$ g per well). Plates were coated with commercial anti\_His monoclonal antibody, incubated with crude extracts of plant-made antigens (nRBD:His or His:bN, respectively), and developed with the different plant-made antibody preparations as indicated. Wild type crude extracts samples were used as negative control. Bars represent mean $\pm$ SD, n = 3 independent replicates.

# Characterization of Antigen-Antibody Binding Activities

Binding activities of affinity purified anti-RBD and anti-N antibodies were analyzed by antigen ELISA as shown in Figures 4A,B. As expected, all assayed antibodies were active in binding their respective antigen. Endpoint dilution titers of antibody preparations were calculated for anti-RBD and anti-N antibodies using the commercial RBD antigen and our purified His:bN antigen. Sybodies 3 and 17 and nanobody72, and CR3009 showed high dilution titers, (0.75, 0.03, 0.03, and $0.58~\mu \mathrm{M}$ , respectively), but the performance of CR3022 and CR3018 was significantly lower (2.85 and $2.91~\mu \mathrm{M}$ , respectively). In a parallel experiment, we tested the ability of plant-made antibodies to selectively detect our own plant-made antigens, including here also the N protein, using a sandwich ELISA approach. For this analysis, ELISA plates were coated with a murine anti-His mAb, incubated with crude plant extracts from antigen-producing plants and sandwiched with purified plant-made antibodies. As shown in Figure 4C, all antigen-producing plant extracts gave sandwich-ELISA signals significantly above the background when assayed using their cognate antibodies, thus evidencing the capacity of both, antibodies and antigens, to function as potent diagnostic tools. Background signals in this experiment are likely due to cross-reaction of the anti-human IgG secondary antibody with the murine anti-His mAb, and could be easily reduced for more potent diagnostic applications by employing recombinant antibody formats other than IgG.

# Pilot Antibody Upscaling and Analysis of Apoplastic Fluid

In the design of a pilot upscaling experiment, we favored modularity and tried to maximize the affordability and adaptability of the process by reducing the requirements for highly specialized lab equipment. We carried out a final agroinfiltration for recombinant antibody production using a total of 112 plants, equivalent to approximately 2.5 kg of fresh plant material. The plants were divided in two batches of 56 plants each and used to produce sybody17 and nanobody72 respectively, as these antibodies showed the most promising binding activities and yields. To facilitate the upscaling of the agroinfiltration process, plant seedlings were transplanted in growth modules, each module comprising seven pots kept together in a double layer of disposable plastic-board hexagons as shown in Figure 5A. Each production batch consisted in eight hexagonal modules. When plants were 6 weeks old, they were agroinfiltrated by submerging each hexagon upside down into a 40 cm diameter cylindrical tank filled with 30 L of an Agrobacterium suspension, set inside a cylindrical vacuum degas chamber (Figure 5A). In this way, seven plants at a time were vacuum-agroinfiltrated by slowly releasing vacuum while leaves remained submerged in the solution. Next, plants were rinsed, brought back to the growth chamber and incubated for 14 days before harvest. Two different concentrations of the Agrobacterium suspension were used in this experiment. One of them (sybody17) consisted in an $OD_{600}$ 0.005 final mix containing plasmids pICH14011, pICH17388, pGreenSybody17-IgG1, and

A   
![](images/17dd5ec305f643b32286b39fa6cd26cba3eab046f64b65e7173bdadafadb9c3b.jpg)

<details>
<summary>text_image</summary>

Upper hexagon
Lower hexagon
Pots
Vacuum chamber
</details>

B   
![](images/8facbab469d4d573d881be8413481c9e57660531a1b2668de84ae835b7c4f155.jpg)

<details>
<summary>natural_image</summary>

Five green leaf specimens showing signs of disease or damage, with some showing discoloration (no text or symbols present)
</details>

C   
![](images/f826abdfe4c5f1305040baddccadf1bd5c274eb420abe21cdc6c14a3c4fcc4b2.jpg)

<details>
<summary>text_image</summary>

C
ΔXT/FT
Control -
sybody17
Control A
(kDa)
C.E P A C.A - +
62
49
38
28
(kDa)
D
WT
Control -
nanobody72
C.E P A C.A
(kDa)
</details>

E   
![](images/6c0e1705f8c2b58b8096d3b42b3050f62dbff71a6877654a3bf33cb23c889266.jpg)

<details>
<summary>line</summary>

| Antibody (μg) | A sybody17 (RBD) | A sybody17 (BSA) | A nanobody72 (RBD) | A nanobody72 (BSA) |
| ------------- | ----------------- | ----------------- | ------------------- | ------------------- |
| 1.00          | 0.65              | 0.75              | 0.70                | 0.15                |
| 0.20          | 0.85              | 0.75              | 0.70                | 0.10                |
| 0.04          | 0.70              | 0.65              | 0.60                | 0.08                |
| 8,00E-03      | 0.45              | 0.40              | 0.35                | 0.07                |
| 1,60E-03      | 0.20              | 0.25              | 0.25                | 0.07                |
| 3,20E-04      | 0.15              | 0.15              | 0.15                | 0.07                |
| 6,40E-05      | 0.10              | 0.10              | 0.10                | 0.07                |
| 1,28E-05      | 0.10              | 0.10              | 0.10                | 0.07                |
</details>

![](images/b76ea85bd464490781cb2b8c81f160f1bef9b24006e9a52cc373f2a2e70f0518.jpg)

<details>
<summary>text_image</summary>

62
49
38
28
</details>

FIGURE 5 | Analysis of upscaling production of anti-RBD sybody17 and nanobody72 antibodies as an example for a pilot assay. (A) Hexagon growth modules used in the pilot assay with N. benthamiana plants and vacuum degas chamber used for vacuum agroinfiltration. (B) Different agroinfiltrated $\Delta$ XT/FT N. benthamiana leaves with DsRed tiles as indicator of the extension of the recombinant protein production. (C) Coomassie stained SDS gel and Western blot of protein extracts from $\Delta$ XT/FT plants expressing sybody17. (D) Coomassie stained SDS gel of protein extracts from WT plants expressing nanobody72. Total crude extract (C.E), purified extract (P), total apoplastic fluid (A) and concentrated apoplastic fluid (C.A). Wild type crude extracts (control) and apoplastic fluid with (+) or without (−) DsRed expression were used as control samples. Seventeen microliter of total protein extract per well and 3 $\mu$ g of purified protein per well were loaded in SDS gels. Arrows show the position of each full-size antibody band. (E) Antigen ELISA titers graph of full-size antibody from apoplastic fluid of sybody17 and nanobody72-expressing plants. Binding activities were calculated by absorbance measurements at 492 nm. The X axis shows the amount ( $\mu$ g) of antibody per well. Solid lines indicate the ELISA test against commercial RBD, and dashed lines indicate the corresponding control ELISA tests against BSA. Bars represent mean ± SD, n = 3 independent replicates.

pICH11599\_DsRed at 1:1:0.9:0.1 ratio, where pICH11599\_DsRed is a MagnICON® 3'module encoding DsRed. The fluorescent marker was added to the infiltration mix to monitor the extension of the viral infection foci. As described elsewhere (Julve et al., 2013; Julve Parreño et al., 2018) superinfection exclusion among virial clones yields mosaic-like expression patterns of individual clones, therefore the tiles produced by red fluorescent proteins were used as an indication of the extension and distribution of the unlabeled foci producing the recombinant antibody. In parallel, nanobody72 up-scaled production was undertaken by agroinfiltration of an $\mathrm{OD}_{600}$ 0.01 Agrobacterium culture mix containing pICH14011, pICH17388, and pGreenNanobody72-IgG1 at 1:1:1 ratio. After 14 days, DsRed tiles in sybody17 experiment, clearly visible with the naked eye, finalized their expansion in most agroinfiltrated leaves, an indicator that the expression tiles had covered the whole leaf surface (Figure 5B). At this stage, leaves were harvested and submitted to an apoplastic fluid recovery assay, where >0.5 kg batches of detached leaves were vacuum-infiltrated in 20 mM phosphate buffer using the same vacuum device as described above. Once rinsed to remove the excess of buffer, leaves were packed in mesh zipped bags, spinned down in a spin portable cloth dryer, and the intercellular apoplastic fluid was recovered from the drain tube. With this simple procedure, between 940 and 1,200 ml of apoplastic fluid (sybody 17 and nanobody 72, respectively) was recovered from 1.2 kg of detached leaves. A fraction of the apoplastic fluid of both antibodies was concentrated eight times with

centrifugal filters (<10 kDa), and the rest was kept refrigerated for further analysis. Figures 5C,D show the Coomassie-staining and Western blot analysis, under reducing conditions, of crude extracts as well as apoplastic fluid preparations, and their corresponding purifications. Crude extracts in this pilot experiment showed a VHH-IgG1 band similar in intensity to that obtained in small-scale experiments (data not shown). Interestingly, apoplastic fluid consisted in a very simplified mix of proteins, with the recombinant antibody being among the most predominant ones. As shown, the different optical density of the Agrobacterium culture, together with the presence of a competing DsRed clone, clearly influenced the accumulation levels, with the yields of nanobody72 clearly outperforming those of its sybody counterpart. Unfortunately, the antibodies seemed partially degraded as indicated by the presence of two bands smaller than the expected VHH-Cγ2-Cγ3 size, which could be compatible with degradation fragments. Degradation was only partially solved with the addition of the protease inhibitor PMSF into the recovered phosphate buffer, as shown with nanobody72 production (Figure 5D). Despite degradation, in a densitometric analysis, we estimate that the recovered apoplastic fluid in this assay contained 159 mg per liter of intact mAb full-size (152 μg/g FW). Finally, we analyzed by antigen ELISA the binding activities of intact sybody17 and nanobody72 antibodies obtained from apoplastic fluid (Figure 5E). The high dilution titers observed (0.38 μM for sybody17 and 0.37 μM for nanobody72) showed that this simple antibody preparation can be directly employed in detection procedures without the need of additional purification steps.

# DISCUSSION

Several N. benthamiana-dedicated bioproduction facilities are functioning worldwide, as those from Leaf Expression Systems in United Kingdom (Dobon, 2019), Icon Genetics (Giritch et al., 2006) and Fraunhofer in Germany (Wirz et al., 2012), iBio facility (Holtz et al., 2015) or Kentucky Bioprocessing in United States (Olinger et al., 2012), among others. Notably, Medicago recently announced the building a new 44,000 sqm facility with capacity for around 40–50 million of planned doses of flu vaccine per year. Such facilities usually involve separated modules for upstream processing, namely a wet-lab module for preparation of the bacterial inoculum, a regular plant growth chamber, and agroinfiltration room, and a post-infiltration growth chamber. In addition, downstream processing facilities are often situated next or to the production ones to minimize the handling time of fresh tissues. Whereas installed capacity of plant-dedicated biofactories is in continue growth, they are clearly insufficient to respond to global or even regional demands in times of crisis. We reasoned that, at least for upstream processes, the infrastructures required for medium-scale N. benthamiana-based production are not radically different to those employed in high-tech agriculture practices as hydroponics, aeroponics, or vertical farming, and thus high-tech agriculture facilities could be easily repurposed as biomanufacturing facilities in a matter of days or weeks

(McDonald and Holtz, 2020). Also, the use of alternative plants and expression systems, in addition to N. benthamiana, could offer additional flexibility and speed. Interestingly in this context, plant-based platforms have important advantages in comparison to the use of more traditional production systems, especially in the case of diagnostic reagents in which the required production scale and the final purity are different from therapeutics. As an exercise to practically test the repurposing requirements, we describe here the partial adaptation of our research laboratory and greenhouse facilities to the production of SARS-CoV-2-related antigens and antibodies using N. benthamiana agroinfiltration as manufacturing platform.

In Figure 6, we show a chronogram of the activities undertaken by our team toward the production of SARS-CoV-2 antigens and antibodies, from the initial selection of the nucleotide sequences of the genes-of-interest to the production of 1 L of plant apoplastic fluid of recombinant sybody17 and nanobody72. In our hands, the whole process took a total of 9 weeks with non-exclusive personnel dedication and partially restricted access to our facilities. The process can be divided in three periods: the first step (DESIGN), taking approximately 10 days, was dedicated to construct design and gene synthesis. It was pivotal in this step to have open access to viral and antibody sequences deposited in pre-print repositories. Particularly remarkable was the openness of academic labs that immediately released primary sequence information of partially characterized anti-SARS-CoV-2 monoclonal antibodies, an exercise that should serve as an example in the future. Due to our limited testing capacity, the number of parallel designs per product was maintained relatively low, and several design decisions (e.g. codon optimization and purification tags) were taken based in a best-guess approach. Ideally, proper crisis preparedness should involve a centralized automated equipment such as a biofoundry (Hillson et al., 2019), with which the design space could be extended dramatically without causing delay. The second phase (BUILD) was dedicated to cloning and construct building and lasted less than 3 weeks. Adapted plasmids and cloning procedures from previous projects are available in our lab (Sarrion-Perdigones et al., 2011; Vazquez-Vilar et al., 2017), therefore no significant time lag occurred in this step. Importantly, this period also involved seeding a new plant batch at the scale required for pilot production in week seven (112 plants distributed in 16 hexagonal modules in this case). In a third phase (TEST), starting on week 5, all constructs were infiltrated at a small scale (three replicate leaves each), shortly incubated (5 or 7 days) and then tested functionally in parallel analyses. This small-scale assay took two additional weeks, summing a total of approximately 50 days for the complete process. Synthetic Biology-inspired Design-Build-Test (DBT) cycles are conceived as iterative processes. Here, we present it as a conceptual framework as no iterations are shown in this work. New DBT cycles can be run fuelled by the conclusion of previous cycles to generate new optimized versions of the product. Based on this experience, we estimate that the whole DBT cycle could be shortened to 30 days or less by optimizing the pipeline (e.g. introducing centralized, automatized design and build phases), and by improving

![](images/cc646378990b34c7ec37577d3f6abf11f732faa1ab0a9a0009d5554fc2ce9118.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["TIMING (In this work)"] --> B["Build"]
    B --> C["Design"]
    C --> D["Test"]
    D --> E["Produce"]
    E --> F["TIMING (optimized)"]
    
    subgraph Build
        G["In silico design"] --> H["Gene synthesis"]
        I["Agroinfiltration"] --> J["Coomassie/ELISA"]
        K["Upscale"] --> L["Extraction/testing"]
    end
    
    subgraph Design
        M["DNA assembly"] --> N["Sanger sequencing"]
    end
    
    subgraph Test
        O["DNA assembly"] --> P["Sanger sequencing"]
    end
    
    subgraph Produce
        Q["Produce"] --> R["Output"]
    end
```
</details>

FIGURE 6 | Schematic view of the Design-Build-Test and Production pipelines employed in this work. The upper timeline represents the actual timespan of the experiments. Note that the time points represent approximately the days required to produce and initially characterize the designated products; notwithstanding, some of the results shown in previous figures correspond to extended analysis obtained at a later stage, during the preparation of this manuscript. The lower timeline describes the estimated minimal timespan that would result by introducing some of the optimizations described in the text.

preparation and anticipation in the facilities (Figure 6). For instance, note that moving from step 2 to step 3 without delay requires a small batch of plants be always maintained in the facility, as it was in our case to supply our research requirements. This only involves transplanting 10–15 seedlings every 3 weeks, and then disposing of them every other 3 weeks once they start flowering. If a continuous plant supply is not maintained, a minimum of three extra weeks needs to be considered to have plants ready for the first TEST iteration.

Whereas the first version of products shown here lack iterative optimization, it would serve eventually to respond to the most urgent demands. In our case, as the results of the first DBT process arose, the best performing version (V1.0) of two of the products were taken to PRODUCTION phase. In the exercise shown here, the upscaling was relatively small (112 plants, approximately 2.5 kg FW). Post-agroinfiltration incubation time was extended to 14 days to maximize yields. In the meantime, optimization of the purification/extraction methods were undertaken at small scale, so that the new knowledge acquired could be applied in the batch purification of the pilot experiment. In a crisis-scenario, and given the modularity of the proposed scheme, several medium-size production modules can be replicated in a farming facility and reproduced in several farms, allowing easy scalability. Successive iterations with small-scale agroinfiltration could

be an effective way to maximize yields and reduce development times by comparing different small-scale strategies. It should be mentioned that the basic apoplast-based downstream processing proposed here could only be used, with the necessary adaptations, in a limited number of crisis-related applications, mainly related with detection and diagnosis. Other uses, certainly therapeutic ones, would involve additional regulatory considerations including GMP downstream facilities, which are beyond the scope of this exercise.

As a result of this experience, several improvements can be envisioned. We employed the MagnICON $^{®}$ vector system with few adaptations for all the attempted proteins. Although MagnICON $^{®}$ produces maximum yields for many products, some proteins, particularly viral antigens may express better with other (e.g., non-viral) systems. In our experiments, antigens showed rather low expression levels despite optimization attempts using codon optimization and different localization signals. In adapting to an emergency, it would be advisable to perform initial expression tests using different production platforms also involving non-replicative methods (Sainsbury and Lomonossoff, 2008) or DNA viruses (Diamos and Mason, 2018; Yamamoto et al., 2018; Diamos et al., 2020) and to incorporate them to the initial optimization test. As mentioned, this could be done in a centralized manner, later distributing expression clones to several repurposed production facilities. In contrast to

antigens, recombinant single-chain antibodies showed in general higher and more uniform expression levels, as could be expected from their more similar structure. We chose to adapt full human IgGs to a scFv-IgG1 format to facilitate cloning and expression procedures, since it has been earlier described in plants that single chain formats reproduce the binding activities of the original full-size antibodies from which they derive. Whereas full-size antibodies are likely more marketable for therapeutic uses, they require co-expression of heavy and light chains using non-interfering replicons, or the employ of non-replicative systems, which undergo chain shuffling and therefore are not suitable for the expression of antibody cocktails. We chose the single-chain antibody production first because this strategy facilitates the upstream process, making use of a single viral vector. Secondly, because our final goal is to produce antibody cocktails as described earlier (Julve Parreño et al., 2018), and this technology is currently optimized for single-chain antibody formats. Furthermore, the single chain format facilitated the comparisons with VHH antibodies, also produced as IgG1 fusions.

The plant-made SARS-CoV-2 products described here have several potential applications in the diagnosis area. Both RBD and N proteins can be used as reagents for serological assays (Amanat et al., 2020; Liu et al., 2020), although further yield optimizations should be required. For those assays, where antigen glycosylation is an important factor, glycoengineered plants (Strasser et al., 2008) can provide a competitive alternative to mammalian cells cultures (O'Flaherty et al., 2020). Regarding antibodies, they can serve as internal references for the quantification of serological responses. With small modifications, the same antibodies can be adapted for sandwich ELISA and employed in the detection and quantification of antigens or viral particles (e.g., using sandwich antibody pairs detecting different surface epitopes), a better proxy for infectiveness than RNA. We also show here that apoplastic fluid is an inexpensive antibody preparation suitable for certain applications that require low-cost preparations, e.g., the concentration of the virus from environmental samples. In terms of yields, the recovery of affinity-purified nanobody72 extracted from whole tissue (192 $\mu$ g/g FW at most) is in the same range that the estimated concentration of the same antibody in the apoplastic fluid (approximately 160 $\mu$ g/ml, equivalent to 152 $\mu$ g/g FW at most). Therefore, we can conclude that in our hands recovery rates from intercellular fluid are similar to those obtained in affinity purified preparations. As shown here, the protein complexity in the apoplast is greatly reduced compared to whole extracts, therefore the apoplast could be regarded as a plant-equivalent of hybridoma supernatant or ascited fluid, although at much lower cost. Unfortunately, apoplastic preparations as well as whole tissue extracts are prone to partial antibody degradation probably due to endogenous proteases. It has been suggested that the length of CDR H3 could influence antibody stability, however in the case of the scFv-Fc reported here, CDR H3 lengths do not seem to be a major factor affecting proteolytic degradation (12 amino acids in CR3022, 14 amino acids in CR3009; Puchol Tarazona et al., 2020). As earlier reported, proteolytic degradation can be minimized using extraction

buffers with appropriate protease inhibitors, as it was shown for nanobody72, or downregulating protease activity (Niemer et al., 2014). Other possibilities to solve degradation problems are to determine and redesign the cleavage sites, or ultimately to separate degradation bands during downstream processing via gel filtration or similar size exclusion technologies.

The current pandemic crisis has evidenced the power of new antibody selection procedures, either based on single-cell selection from human peripheral blood mononuclear cells, in the case of full-size antibodies, or based on ultra-high throughput selection of synthetic libraries (sybodies) in the case of camelid-derived nanobodies (Zimmermann et al., 2018; Walter et al., 2020). Large collections of anti-SARS-CoV-2 potentially neutralizing antibody sequences were made available to the scientific community in a question of weeks rather than months. It does not go unnoticed that the combination of rapid antibody selection procedures with fast, modular and scalable plant expression also has implications in the therapeutic arena as an ideal system for passive immunization. Intravenous polyclonal immunoglobulins (IVIG) from recovered patients have been shown a very effective CoVid-19 treatment in several studies (Montelongo-Jauregui et al., 2020, and references herein); however, the limited availability of patient sera hampers its application in practice. Interestingly, we showed in a recent work that large recombinant polyclonal antibody cocktails (pluribodies), mimicking a mammalian immune response can be produced in N. benthamiana with high batch-to-batch reproducibility (Julve Parreño et al., 2018). Passive immunization with recombinant antibody cocktails resembles a natural response more than a monoclonal therapy, requires shorter developmental times and is probably more robust against the development of resistances. Additionally, other antibody-based therapeutics, such as receptor decoys fused to Fc of IgG1 (such as ACE-Fc described recently; Glasgow et al., 2020), could be designed and produced rapidly in plants followed a similar scheme as described here.

In conclusion, based on the results of the exercise described here, we propose the repurposing of indoors farms into plant-based biomanufacturing facilities as a viable option to respond to local and global shortages of bioproducts such as diagnostics and therapeutic reagents in times of crisis.

# DATA AVAILABILITY STATEMENT

The raw data supporting the conclusions of this article will be made available by the authors, without undue reservation.

# AUTHOR CONTRIBUTIONS

BD-M and BG equally contributed to this work (authors' position only responds to their alphabetical order). All authors designed and performed the experiments and analyzed the data. DO wrote this manuscript. All authors revised and edited the written manuscript. All authors contributed to the article and approved the submitted version.

# FUNDING

This work was supported by H2020 EU projects 760331 Newcotiana and 774078 PharmaFactory. SS is recipient of a FPI fellowship BIO2016-78601-R from the Spanish Ministry of Science and Competitiveness and RM-F is recipient of a GVA fellowship (ACIF/2019/226).

# ACKNOWLEDGMENTS

We want to thank to the staff of the IBMCP and the Polytechnic University of Valencia who help us to access safely to our

# REFERENCES

Amanat, F., Stadlbauer, D., Strohmeier, S., Nguyen, T. H. O., Chromikova, V., McMahon, M., et al. (2020). A serological assay to detect SARS-CoV-2 seroconversion in humans. Nat. Med. 26, 1033–1036. doi: 10.1038/s41591-020-0913-5   
Armbruster, D. A., and Pry, T. (2008). Limit of blank, limit of detection and limit of quantitation. Clin. Biochem. Rev. 29, S49–S52.   
Buyel, J. F. (2019). Plant molecular farming—integration and exploitation of side streams to achieve sustainable biomanufacturing. Front. Plant Sci. 9:1893. doi: 10.3389/fpls.2018.01893   
Capell, T., Twyman, R. M., Armario-Najera, V., Ma, J. K. -C., Schillberg, S., and Christou, P. (2020). Potential applications of plant biotechnology against SARS-CoV-2. Trends Plant Sci. 25, 635–643. doi: 10.1016/j.tplants.2020.04.009   
Diamos, A. G., Hunter, J. G. L., Pardhe, M. D., Rosenthal, S. H., Sun, H., Foster, B. C., et al. (2020). High level production of monoclonal antibodies using an optimized plant expression system. Front. Bioeng. Biotechnol. 7:472. doi: 10.3389/fbioe.2019.00472   
Diamos, A. G., and Mason, H. S. (2018). High-level expression and enrichment of norovirus virus-like particles in plants using modified geminiviral vectors. Protein Expr. Purif. 151, 86–92. doi: 10.1016/j.pep.2018.06.011   
Dobon, A. (2019). Transient gene expression seeds plant-based bioproduction systems: Leaf Expression Systems' Hypertrans technology promises low costs and high yields. GEN. 39, 54–56. doi: 10.1089/gen.39.03.14   
Fernandez-del-Carmen, A., Juárez, P., Presa, S., Granell, A., and Orzáez, D. (2013). Recombinant jacalin-like plant lectins are produced at high levels in Nicotiana benthamiana and retain agglutination activity and sugar specificity. J. Biotechnol. 163, 391–400. doi: 10.1016/j.jbiotec.2012.11.017   
Giritch, A., Marillonnet, S., Engler, C., van Eldik, G., Botterman, J., Klimyuk, V., et al. (2006). Rapid high-yield expression of full-size IgG antibodies in plants coinfected with noncompeting viral vectors. Proc. Natl. Acad. Sci. 103, 14701–14706. doi: 10.1073/pnas.0606631103   
Glasgow, A., Glasgow, J., Limonta, D., Solomon, P., Lui, I., Zhang, Y., et al. (2020). Engineered ACE2 receptor traps potently neutralize SARS-CoV-2. Proc. Natl. Acad. Sci. U. S. A. 117, 28046–28055. doi: 10.1073/pnas.2016093117   
Gleba, Y., Marillonnet, S., and Klimyuk, V. (2004). Engineering viral expression vectors for plants: the ‘full virus’ and the ‘deconstructed virus’ strategies. Curr. Opin. Plant Biol. 7, 182–188. doi: 10.1016/j.pbi.2004.01.003   
Hillson, N., Caddick, M., Cai, Y., Carrasco, J. A., Chang, M. W., Curach, N. C., et al. (2019). Building a global alliance of biofoundries. Nat. Commun. 10:2040. doi: 10.1038/s41467-019-10079-2   
Holtz, B. R., Berquist, B. R., Bennett, L. D., Kommineni, V. J. M., Munigunti, R. K., White, E. L., et al. (2015). Commercial-scale biotherapeutics manufacturing facility for plant-made pharmaceuticals. Plant Biotechnol. J. 13, 1180–1190. doi: 10.1111/pbi.12469   
Julve, J. M., Gandía, A., Fernández-del-Carmen, A., Sarrion-Perdigones, A., Castelijns, B., Granell, A., et al. (2013). A coat-independent superinfection exclusion rapidly imposed in Nicotiana benthamiana cells by tobacco mosaic virus is not prevented by depletion of the movement protein. Plant Mol. Biol. 81, 553–564. doi: 10.1007/s11103-013-0028-1

laboratory and greenhouses during the CoVid-19 lockdown in Spain, and specially Eugenio Grau for his readiness to help us with Sanger sequencing during that period. We are also grateful to Prof. Steinkellner and Strasser for providing glycoengineered plant lines and to Prof. Gleba for sharing Magnifection plasmids.

# SUPPLEMENTARY MATERIAL

The Supplementary Material for this article can be found online at: https://www.frontiersin.org/articles/10.3389/fpls.2020.612781/full#supplementary-material

Julve Parreño, J. M., Huet, E., Fernández-del-Carmen, A., Segura, A., Venturi, M., Gandía, A., et al. (2018). A synthetic biology approach for consistent production of plant-made recombinant polyclonal antibodies against snake venom toxins. Plant Biotechnol. J. 16, 727–736. doi: 10.1111/pbi.12823   
Liu, W., Liu, L., Kou, G., Zheng, Y., Ding, Y., Ni, W., et al. (2020). Evaluation of nucleocapsid and spike protein-based enzyme-linked immunosorbent assays for detecting antibodies against SARS-CoV-2. J. Clin. Microbiol. 58, e00461–e004620. doi: 10.1128/JCM.00461-20   
Marillonnet, S., Thoeringer, C., Kandzia, R., Klimyuk, V., and Gleba, Y. (2005). Systemic Agrobacterium tumefaciens-mediated transfection of viral replicons for efficient transient expression in plants. Nat. Biotechnol. 23, 718–723. doi: 10.1038/nbt1094   
McDonald, K. A., and Holtz, R. B. (2020). From farm to finger prick—a perspective on how plants can help in the fight against COVID-19. Front. Bioeng. Biotechnol. 8:782. doi: 10.3389/fbioe.2020.00782   
Montelongo-Jauregui, D., Vila, T., Sultan, A. S., and Jabra-Rizk, M. A. (2020). Convalescent serum therapy for COVID-19: a 19th century remedy for a 21st century disease. PLoS Pathog. 16:e1008735. doi: 10.1371/journal.ppat.1008735   
Moon, K. -B., Park, J. -S., Park, Y. -I., Song, I. -J., Lee, H. -J., Cho, H. S., et al. (2019). Development of systems for the production of plant-derived biopharmaceuticals. Plants 9:30. doi: 10.3390/plants9010030   
Niemer, M., Mehofer, U., Torres Acosta, J. A., Verdianz, M., Henkel, T., Loos, A., et al. (2014). The human anti-HIV antibodies 2F5, 2G12, and PG9 differ in their susceptibility to proteolytic degradation: down-regulation of endogenous serine and cysteine proteinase activities could improve antibody production in plant-based expression platforms. Biotechnol. J. 9, 493–500. doi: 10.1002/biot.201300207   
O'Flaherty, R., Bergin, A., Flampouri, E., Mota, L. M., Obaidi, I., Quigley, A., et al. (2020). Mammalian cell culture for production of recombinant proteins: a review of the critical steps in their biomanufacturing. Biotechnol. Adv. 43:107552. doi: 10.1016/j.biotechadv.2020.107552   
Olinger, G. G., Pettitt, J., Kim, D., Working, C., Bohorov, O., Bratcher, B., et al. (2012). Delayed treatment of Ebola virus infection with plant-derived monoclonal antibodies provides protection in rhesus macaques. Proc. Natl. Acad. Sci. 109, 18030–18035. doi: 10.1073/pnas.1213709109   
Puchol Tarazona, A. A., Lobner, E., Taubenschmid, Y., Paireder, M., Torres Acosta, J. A., Göritzter, K., et al. (2020). Steric accessibility of the cleavage sites dictates the proteolytic vulnerability of the anti-HIV-1 antibodies 2F5, 2G12, and PG9 in plants. Biotechnol. J. 15:e1900308. doi: 10.1002/biot.201900308   
Sainsbury, F., and Lomonossoff, G. P. (2008). Extremely high-level and rapid transient protein production in plants without the use of viral replication. Plant Physiol. 148, 1212–1218. doi: 10.1104/pp.108.126284   
Sainsbury, F., Thuenemann, E. C., and Lomonossoff, G. P. (2009). pEAQ: versatile expression vectors for easy and quick transient expression of heterologous proteins in plants. Plant Biotechnol. J. 7, 682–693. doi: 10.1111/j.1467-7652.2009.00434.x   
Sarrion-Perdigones, A., Falconi, E. E., Zandalinas, S. I., Juárez, P., Fernández-del-Carmen, A., Granell, A., et al. (2011). GoldenBraid: an iterative cloning system for standardized assembly of reusable genetic modules. PLoS One 6:e21622. doi: 10.1371/journal.pone.0021622

Sarrion-Perdigones, A., Vazquez-Vilar, M., Palaci, J., Castelijns, B., Forment, J., Ziarsolo, P., et al. (2013). GoldenBraid 2.0: a comprehensive DNA assembly framework for plant synthetic biology. Plant Physiol. 162, 1618–1631. doi:10.1104/pp.113.217661   
Strasser, R., Stadlmann, J., Schähs, M., Stiegler, G., Quendler, H., Mach, L., et al. (2008). Generation of glyco-engineered Nicotiana benthamiana for the production of monoclonal antibodies with a homogeneous human-like N-glycan structure: XylT and FucT down-regulation in N. benthamiana. Plant Biotechnol. J. 6, 392–402. doi: 10.1111/j.1467-7652.2008.00330.x   
Tian, X., Li, C., Huang, A., Xia, S., Lu, S., Shi, Z., et al. (2020). Potent binding of 2019 novel coronavirus spike protein by a SARS coronavirus-specific human monoclonal antibody. Emerg. Microbes Infect. 9, 382–385. doi:10.1080/22221751.2020.1729069   
van den Brink, E. N., ter Meulen, J., Cox, F., Jongeneelen, M. A. C., Thijsse, A., Throsby, M., et al. (2005). Molecular and biological characterization of human monoclonal antibodies binding to the spike and nucleocapsid proteins of severe acute respiratory syndrome coronavirus. J. Virol. 79, 1635–1644. doi: 10.1128/JVI.79.3.1635-1644.2005   
Vazquez-Vilar, M., Quijano-Rubio, A., Fernandez-del-Carmen, A., Sarrion-Perdigones, A., Ochoa-Fernandez, R., Ziarsolo, P., et al. (2017). GB3.0: a platform for plant bio-design that connects functional DNA elements with associated biological data. Nucleic Acids Res. 45, 2196–2209. doi: 10.1093/nar/gkw1326   
Walter, J. D., Hutter, C. A. J., Zimmermann, I., Wyss, M., Egloff, P., Sorgenfrei, M., et al. (2020). Sybodies targeting the SARS-CoV-2 receptor-binding domain. bioRxiv. [Preprint]. doi: 10.1101/2020.04.16.045419   
Wirz, H., Sauer-Budge, A. F., Briggs, J., Sharpe, A., Shu, S., and Sharon, A. (2012). Automated production of plant-based vaccines and pharmaceuticals. J. Lab. Autom. 17, 449–457. doi: 10.1177/2211068212460037

Wrapp, D., De Vlieger, D., Corbett, K. S., Torres, G. M., Wang, N., Van Breedam, W., et al. (2020). Structural basis for potent neutralization of betacoronaviruses by single-domain camelid antibodies. Cell 181, 1004–1015. e15. doi: 10.1016/j.cell.2020.04.031   
Yamamoto, T., Hoshikawa, K., Ezura, K., Okazawa, R., Fujita, S., Takaoka, M., et al. (2018). Improvement of the transient expression system for production of recombinant proteins in plants. Sci. Rep. 8:4755. doi: 10.1038/s41598-018-23024-y   
Zimmermann, I., Egloff, P., Hutter, C. A., Arnold, F. M., Stohler, P., Bocquet, N., et al. (2018). Synthetic single domain antibodies for the conformational trapping of membrane proteins. eLife 7:e34317. doi: 10.7554/eLife.34317   
Zrein, M., De Marcillac, G., and Van Regenmortel, M. H. V. (1986). Quantitation of rheumatoid factors by enzyme immunoassay using biotinylated human IgG. J. Immunol. Methods 87, 229–237. doi: 10.1016/0022-1759(86)90536-3

Conflict of Interest: The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

Copyright © 2020 Diego-Martin, González, Vazquez-Vilar, Selma, Mateos-Fernández, Gianoglio, Fernández-del-Carmen and Orzáez. This is an open-access article distributed under the terms of the Creative Commons Attribution License (CC BY). The use, distribution or reproduction in other forums is permitted, provided the original author(s) and the copyright owner(s) are credited and that the original publication in this journal is cited, in accordance with accepted academic practice. No use, distribution or reproduction is permitted which does not comply with these terms.