# Article

# A platform for the generation of site-specific antibody-drug conjugates that allows for selective reduction of engineered cysteines

Ruud Coumans, Gerry Ariaans, Henri Spijker, Pascal Renart Verkerk, Patrick Beusker, Bas Kokke, Jan Schouten, Marion Blomenrohr, Miranda van der Lee, Patrick Groothuis, Ruud Ubink, Wim Dokter, and Marco Timmers

Bioconjugate Chem., Just Accepted Manuscript $\cdot$ DOI: 10.1021/acs.bioconjchem.0c00337 $\cdot$ Publication Date (Web): 22 Jul 2020

Downloaded from pubs.acs.org on July 22, 2020

# Just Accepted

"Just Accepted" manuscripts have been peer-reviewed and accepted for publication. They are posted online prior to technical editing, formatting for publication and author proofing. The American Chemical Society provides "Just Accepted" as a service to the research community to expedite the dissemination of scientific material as soon as possible after acceptance. "Just Accepted" manuscripts appear in full in PDF format accompanied by an HTML abstract. "Just Accepted" manuscripts have been fully peer reviewed, but should not be considered the official version of record.

They are citable by the Digital Object Identifier (DOI®). "Just Accepted" is an optional service offered to authors. Therefore, the "Just Accepted" Web site may not include all articles that will be published in the journal. After a manuscript is technically edited and formatted, it will be removed from the "Just Accepted" Web site and published as an ASAP article. Note that technical editing may introduce minor changes to the manuscript text and/or graphics which could affect content, and all legal disclaimers and ethical guidelines that apply to the journal pertain.

ACS cannot be held responsible for errors or consequences arising from the use of information contained in these "Just Accepted" manuscripts.

BC Bioconjugate Chemistry

UNIVERSITYOF BIRMINGHAM

Subscriber access provided by University of Birmingham

is published by the American Chemical Society. 1155 Sixteenth Street N.W., Washington, DC 20036

Published by American Chemical Society. Copyright © American Chemical Society. However, no copyright claim is made to original U.S. Government works, or works produced by employees of any Commonwealth realm Crown government in the course of their duties.

# A platform for the generation of site-specific antibody-drug conjugates that allows for selective reduction of engineered cysteines

Ruud G. E. Coumans*, Gerry J. A. Ariaans, Henri J. Spijker, Pascal Renart Verkerk, Patrick H. Beusker, Bas P. A. Kokke, Jan Schouten, Marion Blomenrohr, Miranda M. C. van der Lee, Patrick G. Groothuis, Ruud Ubink, Wim H. A. Dokter and C. Marco Timmers.

Byondis B.V., Microweg 22, 6545 CM Nijmegen, The Netherlands.

* To whom correspondence may be addressed. Email: ruud.coumans@byondis.com. Telephone: +31(0)24 679 5100

# TOC Graphic

![](dt=2026-03-30/ht=21/c6a69748453af41e58c46a310fc099ea2f45b3b4d973d06926302fb6070b0b15.jpg)

# Abstract

Engineering cysteines at specific sites in antibodies to create well-defined ADCs for the treatment of cancer is a promising approach to increase the therapeutic index and helps streamline the manufacturing process. Here, we report the development of an in silico screening procedure to select for optimal sites in an antibody to which a hydrophobic linker-drug can be conjugated. Sites were identified inside the cavity that is naturally present in the Fab part of the antibody.

Conjugating a linker-drug to these sites demonstrated the ability of the antibody to shield the hydrophobic character of the linker-drug while resulting ADCs maintained their cytotoxic potency in vitro. Comparison of site-specific ADCs versus randomly conjugated ADCs in an in vivo xenograft model revealed improved efficacy and exposure. We also report a selective reducing agent, that is able to reduce the engineered cysteines while leaving the interchain disulfides in the oxidized state.

This enables us to manufacture site-specific ADCs without introducing impurities associated with the conventional reduction/oxidation procedure for site-specific conjugation.

# Introduction

ADCs are targeted therapeutics for cancer therapy that are composed of a tumor-targeting antibody coupled to a highly potent chemotherapeutic agent. $^{1,2}$ Several ADCs have been approved for use in

Page 1 of 29

Bioconjugate Chemistry

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

hematological and solid cancers and the number of ADCs in clinical development is growing. $^{3}$ One of the challenges associated with ADCs is their manufacturing, as the often poorly soluble chemotherapeutic agents need to be coupled to the antibody in an aqueous environment without overly inducing aggregation in the resulting conjugate. Most drug-to-antibody couplings make use either of partially reduced interchain disulfide bonds enabling thiol-maleimide chemistry or of lysine functionalization using activated esters.

Both methods result in statistical mixtures of conjugated antibody species with different numbers of drugs attached. Adem et al. have shown that in an ADC with an MMAE payload, the ADC species with a drug-to-antibody ratio (DAR) of 4, 6 or 8, which were demonstrated to be increasingly hydrophobic, were much more prone to aggregation than the DAR2 species. $^{4}$ Not only does higher loading of the antibody with hydrophobic payloads lead to manufacturing and stability issues, the pharmacokinetics (PK) of ADCs are also influenced by the DAR. Hamblett et al.

have revealed that ADCs with a higher average drug loading demonstrate higher in vivo clearance rates than ADCs with a lower average drug load, leading to attenuated exposure and therefore lower efficacies. $^{5}$ As shown by Lyon et al., uptake by Kupffer cells in the liver contributed to the increased clearance rate of higher DAR species and a linear correlation was observed between the clearance rate and hydrophobicity of the ADC.

$^{6}$ The increased uptake of ADCs in ocular tissue of MMAF-based ADCs was shown to be mediated by aspecific macropinocytosis that could be counteracted by PEGylation of lysine residues of the ADC. $^{7}$ It is evident from the examples above that hydrophobicity is a key parameter in the design of improved ADCs. However, the intrinsic hydrophobicity that is observed for several suitable payloads for an ADC implies that some chemical modification of the payload, linker or antibody must be made in order for the ADCs to become less hydrophobic.

PEGylation of a doxorubicin-based linker-drug for example has been demonstrated to have a profound effect on the stability and manufacturability of ADCs as it results in much less aggregation as compared to the non-PEGylated ADCs. $^{8}$ Also, PEG-modified ADCs were shown to have better PK properties as a result of slower clearance by the liver.

$^{6,9}$ We recently introduced a novel ADC technology platform based on a highly potent duocarmycin linker-drug (vc-seco-DUBA) in which the hydrophobic duocarmycin payload is randomly conjugated to the free cysteines that result from reduced endogenous interchain disulfide bonds of a tumor-targeting antibody. Subsequent preparative Hydrophobic Interaction Chromatography (HIC) purification $^{10,11}$ enables the removal of unconjugated species and some of the more hydrophobic higher DAR species.

This method was used to make the HER2-targeting ADC SYD985, which has shown promising activity in preclinical models and has demonstrated to deliver clinical benefit in its ongoing clinical development pr
ogram. $^{12,13}$ In order to completely prevent the formation of random DAR4 or higher species that are prone to aggregation, Junutula et al. have

Bioconjugate Chemistry

Page 2 of 29

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

introduced cysteine handles at specific sites in the antibody that can be coupled to a payload via thiolmaleimide chemistry in order to generate ADCs with a drug load of two. $^{14}$ These cysteine-engineered antibodies can be optimized through site-selection leading to improved pharmacokinetics and broadening of the therapeutic index. $^{15}$ Of interest is also a paper by Ohri et al., which describes the systematic evaluation of all possible sites in an antibody for the site-specific conjugation of linker-drugs using stability and conjugatability as read-outs. $^{16}$ Tumey et al.

have shown that site-selection can influence the hydrophobicity of the resulting ADC, which was demonstrated to correlate with linker stability. $^{17}$ Finally, in an elegant example by Sussman et al., the coupling of a hydrophobic payload class called dimeric pyrrolobenzodiazepines (PBDs) to an engineered cysteine at the 239 position in the heavy chain of an antibody led to ADCs in which the hydrophobicity of the payload was shielded by the antibody.

$^{18}$ The resulting ADCs exhibited good physicochemical properties as evidenced by the presence of low amounts of high molecular weight (HMW) species. Also, improved in vivo activity in a tumor xenograft model was observed when compared to random conjugates or site-specific conjugates at other sites in the Fc region.

Considering the substantial numbers of possible sites for cysteine engineering on an antibody, the challenge remains where one needs to introduce the cysteines to get the ADC with the most optimal characteristics, especially when considering conjugation of hydrophobic payloads such as duocarmycins and dimeric PBDs.

Here, we use an in silico method to select for optimal conjugation sites (in an IgG1 scaffold) for the conjugation of hydrophobic payloads. By using a docking procedure, we were able to identify sites that were experimentally shown to be able to shield the hydrophobicity exerted by the conjugated duocarmycin payload. The resulting ADCs were less hydrophobic and had improved properties over randomly conjugated ones.

Moreover, during the course of these studies, we developed an improved method for the manufacturing of these ADCs by relying on a single-step, selective reduction of the engineered cysteines instead of a two-step reduction/oxidation protocol that is commonly used for these types of ADCs and that can lead to undesired side-products. This significantly improves the efficiency of the manufacturing process and the quality of the resulting ADCs.

The findings presented here lead to a comprehensive platform for the generation of ADCs with improved physicochemical characteristics and robust manufacturability.

Page 3 of 29

Bioconjugate Chemistry

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

# Results and Discussion

# In silico selection of suitable sites for conjugation of hydrophobic payloads

The seco-duocarmycin payload in the linker-drug vc-seco-DUBA (Figure 1) is a hydrophobic molecule due to its flat aromatic structure and coupling such a moiety to an antibody leads to the introduction of hydrophobic patches that render the resulting ADC more hydrophobic than the parent antibody. This leads to negative effects with respect to physicochemical properties (e.g. aggregation) and a less favorable pharmacological profile (e.g. increased clearance) as detailed in the Introduction.

![](dt=2026-03-30/ht=21/44b6b7abcc865f98763ee08797bebdee6e8efef25ed062a4eea18c91020a675c.jpg)

![](dt=2026-03-30/ht=21/f7162df701ba82e03fdd1a83136c3cbf11f755e10daeb7970f3b0fc3c726d07d.jpg)

We hypothesized that placing a linker-drug at a position of the antibody where it has strong interactions with the protein might reduce the newly introduced hydrophobic surface stemming from the payload. It was envisioned that a hydrophobic patch present on an antibody can have strong interactions with the linker-drug, thus replacing an already present hydrophobic surface with another (although larger) hydrophobic surface. We assumed that when such surfaces are identified, a suitable site for the introduction of a cysteine in the vicinity of this surface can then be selected. Moreover, we set the

Bioconjugate Chemistry

Page 4 of 29

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

condition that the linker-drug should be coupled in the Fab part of the antibody, as work by Fan et al. and Zhang et al. has shown that extracellular proteases present in tumors are able to cleave off a single chain of the Fc part of IgGs $^{19,20}$ and consequently, conjugation of a linker-drug to the Fc part of IgGs would lead to loss of potency. Thus, we created homology models using the YASARA protein modeling program of Fab parts of two different antibodies. $^{21}$ The antibodies used were: mAb1 based on the J591 antibody targeting PSMA and mAb2 based on the H8 antibody targeting 5T4.

$^{22,23}$ For both mAbs, the best results were generated by using the crystal structure of the Fab fragment of the anti-Shh 5E1 chimera as a template (PDB ID: 3MXV). $^{24}$ In order to facilitate the search for optimal sites, the Autodock VINA method was used to dock the linker-drug onto these protein models (See Materials and Methods). $^{25}$ Surprisingly, docking of the linker-drug on the model of Fab1 (of mAb1) using the VINA algorithm resulted in positioning of the linker-drug into the cavity present between the heavy- and light chains.

All the docking runs that were performed led to positioning of the linker-drug inside the Fab cavity. Docking of the linker-drug on Fab1 resulted in 15 unique docking conformations with calculated binding energies varying between -8.0 and -6.9 kcal/mol. To demonstrate the general applicability of this method, Fab2 (of mAb2) was also used to dock to the linker-drug. Again, this led to exclusive arrangement inside the cavity with 10 unique conformations that had calculated binding energies between -8.3 and -7.3 kcal/mol (Figure 2).

![](dt=2026-03-30/ht=21/ac3d42cf9901db9b5cbf77a83398b66f0d3a966db72291fe80714e8f1a48cd1b.jpg)

This prompted us to search for suitable sites for the introduction of a cysteine residue in or near the Fab cavity. We chose sites that complied with the following criteria: 1. Located in the vicinity of the calculated maleimide positions of the linker-drug; 2. Close to, or inside the Fab cavity and pointing into the cavity, which should prevent formation of disulfide bridges between antibodies; 3. Site residue side chains should

Page 5 of
29

Bioconjugate Chemistry

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

be in an accessible position; 4. Unlikely involved in strong structural interactions which could lead to reduced antigen affinity; 5. Not in the vicinity of other cysteines. Evaluating the model of Fab1 with docked linker-drug revealed a number of potential sites as shown in Figure 3.

![](dt=2026-03-30/ht=21/c56b86b3172647c224bbec952844068c2b7938f2066f0e3599283841957bc9df.jpg)

The following sites in Fab1 were found suitable for replacement by a cysteine residue: HC-41S, HC-89V, HC-110T, HC-152E, HC-153P, LC-40P, LC-41G, LC-165E and LC-168S. All positions in the variable domains are indicated by Kabat numbering, whereas the positions in the constant regions are indicated by Eu numbering.[26] The molecular modeling studies showed that also for mAb2 similar positions in the Fab were identified that would be suitable for conjugation (See Figure S1).

# Coupling of a duocarmycin payload in the Fab cavity reveals hydrophobicity shielding and favorable aggregation behavior

In order to experimentally validate the suitability of the in silico identified sites for attachment of linker-drugs to the antibody, several antibodies based on mAb1 and mAb2 were made that contain engineered cysteines at the proposed sites. All of the mutant proteins were well expressed and had low levels of aggregation. Furthermore, a HC-120C mutant was made with a cysteine located outside of the Fab cavity serving as a control antibody. In order to create the corresponding ADCs, a known procedure was used in which the antibody was first fully reduced with an excess of TCEP (which removes the cysteine or

Bioconjugate Chemistry

Page 6 of 29

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

glutathione caps from the engineered cysteines and reduces interchain disulfide bridges at the same time), after which the interchain disulfide bonds are reformed with the oxidant DHAA. The unmasked engineered cysteines were coupled to the linker-drug and the resulting ADCs were purified as described in the Materials and Methods section. The ADCs had average DARs in the range of 1.5-1.8 (as determined by analytical HIC).

In order to assess the overall hydrophobicity of the ADCs, HIC was also employed to analyze the retention times of the antibody with two drugs attached, as this is the main species present (see Supporting Information). $^{17,18}$ Some typical HIC chromatograms are shown Figure 4, in which randomly conjugated ADC, a control HC-120C ADC and a HC-41C ADC are compared (See Figure S2 for full traces).

![](dt=2026-03-30/ht=21/054e49fffdf0d9d228aedb6ef0d5e8c28c376716d8007327e1d7670500297535.jpg)

As is evident from the HIC profiles, there is a difference in retention times (RTs) between these ADCs as a result of conjugating the linker-drug at different attachment sites in the antibody. Because HIC separates proteins based on hydrophobicity, it was concluded that conjugation of the linker-drug inside the Fab

Page 7 of 29

Bioconjugate Chemistry

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

cavity at the HC-41C position leads to shielding of the hydrophobic payload by the surrounding protein mantle as compared to the random conjugate. A similar shielding effect was observed by Benjamin et al. for a payload that is conjugated in the cavity formed by the CH2-CH3 domains in the Fc part of an antibody.[27] Interestingly, the HC-120C variant, in which the payload was predicted not to benefit from shielding by the protein mantle as it points away from the antibody surface, even shows less shielding as compared to the random conjugate.

Table 1 summarizes the HIC retention times of the DAR2 peaks for the different mutants tested. To facilitate a comparison of hydrophobicity, the parameter relative hydrophobicity (RH) was defined as $\mathrm{RH} = (\mathrm{RT}_{\mathrm{dar2}} - \mathrm{RT}_{\mathrm{dar0}}) / (\mathrm{RT}_{\mathrm{dar2(random)}} - \mathrm{RT}_{\mathrm{dar0(random)}})$ . Using this parameter, it can be easily seen if an engineered site performs better (shields the payload) or worse (exposes the payload) in comparison to a random conjugate.

Table 1. HIC RTs (min), RHs, DARs and %HMWs for the cysteine-engineered ADCs tested.

![](dt=2026-03-30/ht=21/c595e58d9f38d571423394ddea4305fbb6577e9828afdbaf0b875588ea8bbc4c.jpg)

<table><tr><td>ADC</td><td>RT_dar0</td><td>RT_dar2</td><td>RH</td><td>DAR</td><td>%HMW</td></tr><tr><td>ADC1 random</td><td>6.9</td><td>9.7</td><td>1.0</td><td>1.8</td><td>7.7</td></tr><tr><td>ADC1 HC-120C</td><td>6.8</td><td>11.3</td><td>1.6</td><td>1.8</td><td>0.9</td></tr><tr><td>ADC1 HC-41C</td><td>6.8</td><td>8.5</td><td>0.6</td><td>1.7</td><td>1.4</td></tr><tr><td>ADC1 HC-152C</td><td>6.5</td><td>8.8</td><td>0.7</td><td>1.5</td><td>1.2</td></tr><tr><td>ADC1 HC-153C</td><td>6.5</td><td>8.7</td><td>0.8</td><td>1.5</td><td>2.4</td></tr><tr><td>ADC1 LC-40C</td><td>6.9</td><td>9.5</td><td>0.9</td><td>1.8</td><td>0.5</td></tr><tr><td>ADC1 LC-41C</td><td>6.9</td><td>8.7</td><td>0.6</td><td>1.8</td><td>0.6</td></tr><tr><td>ADC1 LC-165C</td><td>6.6</td><td>8.4</td><td>0.6</td><td>1.5</td><td>2.3</td></tr><tr><td>ADC2 random</td><td>6.4</td><td>9.9</td><td>1.0</td><td>2.0</td><td>4.4</td></tr><tr><td>ADC2 HC-40C</td><td>6.2</td><td>8.8</td><td>0.7</td><td>1.7</td><td>1.2</td></tr><tr><td>ADC2 HC-41C</td><td>6.2</td><td>7.4</td><td>0.3</td><td>1.7</td><td>0.4</td></tr></table>

Abbreviations: RT=retention time, RH=relative hydrophobicity, DAR=drug-to-antibody ratio, %HMW=%high molecular weight.

As can be seen from Table 1, for all of the sites that were proposed by the in silico docking method, there is a clear shielding effect as evidenced by relative hydrophobicity (RH) values lower than 1.0. Both the HC-41C and LC-41C have the lowest RH. Strikingly, the LC-40C mutant is less shielded than the neighboring LC-41C, although both sites are more shielded than the random conjugate. It is tentatively concluded that besides the site of attachment also the orientation of the engineered cysteine plays a role in the level of shielding that can be exerted by the protein mantle on the payload. For mAb2, both HC-40C and HC-41C

Bioconjugate Chemistry

Page 8 of 29

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

demonstrated shielding of the payload, with a larger shielding effect for the HC-41C (similar to the difference between the LC-40C and LC-41
C in mAb1). These results show that shielding of the payload at a position in the Fab cavity appears to be a general principle which is independent of the antibody. Because ADC2 HC-41C, with a RH of 0.3, displayed the most hydrophobicity shielding, it was decided to further evaluate this position in a set of 13 antibodies that have been used in clinical trials. RHs ranging from 0.2 to 0.9 were found for the corresponding ADCs, indicating that for all the antibodies tested shielding of the payload was observed (Table S1).

Another critical property of an ADC is its aggregation behavior. Coupling of hydrophobic payloads to an antibody can lead to the formation of dimers and trimers that can become a liability in process development. $^{18}$ Analysis of the $\%$ HMW species for the site-specific ADCs revealed clear advantages of coupling the payload inside the Fab cavity as compared to random conjugation. Low $\%$ HMW species for all the engineered sites were observed as shown in Table 1. However, as the $\%$ HMW species for HC-120C is also low, the decrease should be mainly attributed to the absence of higher DAR (DAR4 and DAR6) species in the random conjugates that are more prone to aggregation.

Shielding of the payload was also observed for vcMMAE and mcMMAF ADCs, demonstrating the generality of the concept (see Tables S2 and S3).[28,29]

# Conjugating in Framework Region 2 of an IgG1 does not affect its antigen binding affinity and in vitro cell killing activity on target-expressing cells

The HC-40C, HC-41C, LC-40C and LC-41C mutated antibodies all have in common that their site of mutation is in Framework Region 2 (FR2) of the antibody. Previous work by Albone et al. and Spidel et al. demonstrated that conjugating an MMAF-like payload in the Framework Region 3 of the light chain at LC-80C in antibodies was possible without compromising binding affinity resulting in active ADCs.

$^{30,31}$ In our case, although molecular modeling indicated that a cysteine at these positions did not significantly distort the CDR regions, it remained to be seen if our antibodies and their corresponding ADCs, bearing a hydrophobic payload at a proximal position to important binding site regions, would still bind to the antigen and potently kill antigen-expressing tumor cells.

Using a cellular binding assay, it was demonstrated that the randomly conjugated ADC1 and the site-specific ADC1 HC-41C had similar affinities for PSMA-positive LNCap-C4.2 cells ( $EC_{50}$ in the range $0.1 - 0.2 \mu \mathrm{g} / \mathrm{ml}$ , see Figure S3 for exemplary full binding curves). To further evaluate target binding and potency, the ADCs described in Table 1 were tested in PSMA-positive LNCap-C4.2 cells (ADC1 mutants) and 5T4-positive MDA-MD-468 cells (ADC2 mutants). Gratifyingly, as shown in Table 2, in vitro cytotoxicity tests on target expressing cells gave similar results

Page 9 of 29

Bioconjugate Chemistry

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

as compared to the random ADC (See Figure S3 for exemplary full cellular cytotoxicity curves). All ADCs were inactive on target-negative cells ( $\mathrm{IC}_{50} > 70\mathrm{nM}$ ) indicating selective killing of tumor cells mediated by the target (data not shown). These experiments demonstrate that conjugation with a hydrophobic payload at the indicated positions does not lead to alterations in binding affinity and in vitro cellular cytotoxicity when compared to a randomly conjugated ADC.

Table 2. In vitro cytotoxicity of cysteine-engineered ADCs

![](dt=2026-03-30/ht=21/b7c7b7531e20f7a661f92a6c2d4164093cc5986194cd19ac4dcd046342a40090.jpg)

<table><tr><td>ADC</td><td>IC50(nM)</td><td>95% CI (nM)</td><td>% efficacy</td></tr><tr><td>ADC1 random</td><td>0.23</td><td>0.20 - 0.27</td><td>82</td></tr><tr><td>ADC1 HC-120C</td><td>0.14</td><td>0.13 - 0.16</td><td>82</td></tr><tr><td>ADC1 HC-41C</td><td>0.25</td><td>0.21 - 0.28</td><td>78</td></tr><tr><td>ADC1 HC-152C</td><td>0.44</td><td>0.36 – 0.55</td><td>78</td></tr><tr><td>ADC1 HC-153C</td><td>0.34</td><td>0.28 – 0.41</td><td>79</td></tr><tr><td>ADC1 LC-40C</td><td>0.30</td><td>0.23 – 0.37</td><td>80</td></tr><tr><td>ADC1 LC-41C</td><td>0.31</td><td>0.25 – 0.38</td><td>80</td></tr><tr><td>ADC1 LC-165C</td><td>0.51</td><td>0.40 – 0.65</td><td>79</td></tr><tr><td>ADC2 random</td><td>0.09</td><td>0.08 – 0.10</td><td>91</td></tr><tr><td>ADC2 HC-40C</td><td>0.07</td><td>0.07 – 0.08</td><td>88</td></tr><tr><td>ADC2 HC-41C</td><td>0.07</td><td>0.06 – 0.08</td><td>88</td></tr></table>

Abbreviations: $\mathsf{IC}_{50} =$ half maximal inhibitory concentration, CI=confidence interval.

# Enhanced in vivo antitumor activity of a cysteine-engineered ADC

After having demonstrated that the site-specific ADCs have improved physicochemical properties over the random ADCs in terms of hydrophobicity and %HMW species and remain potent in in vitro cytotoxicity assays, a BT-474 xenograft model in mice was run in order to compare the in vivo efficacy of a set of site-specific ADCs based on the duocarmycin linker-drug and their respective randomly conjugated ADCs.

In contrast to humans, mice express carboxylesterase 1c (CES1c) which rapidly cleaves the linker-drug resulting in very high clearance of conjugated ADC in mice and a PK profile in mice that is irrelevant for humans. $^{32}$ Therefore, xenograft studies were done in CES1c knockout mice that were bred into a SCID background. $^{12}$ A BT-474 xenograft model was selected since this tumor is antigen-positive for 5T4 (see Figure 5B) and negative for PSMA enabling assessment of both antigen-mediated anti-tumor activity and potential non-antigen-mediated effects.

$^{33}$ It was decided to solely focus on HC-41C ADCs, as the hydrophobicity data showed that HC-41C had the best hydrophobicity shielding capacity with a RH of 0.3.

Bioconjugate Chemistry

Page 10 of 29

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

A single dose of $3\mathrm{mg / kg}$ 5T4-targeting ADC induced tumor regression for the site-specific ADC2 HC41C, which is in stark contrast to the tumor growth delay that was observed for the randomly conjugated ADC (Figure 5A). In contrast the PSMA-targeted ADCs show only marginal effects which is in line with activity of non-binding isotype control ADCs as reported previously and for other non-binding isotype control vcseco-DUBA-based ADCs in this experiment (data not shown). For the 5T4-targeting ADC (ADC2), the site-specific HC-41C ADC resulted in a substantially improved anti-tumor response when compared to the randomly conjugated ADC.

![](dt=2026-03-30/ht=21/80e08b5f93a644e575a3494079b8385f64ed95fb329556b4ce923ac8157eda04.jpg)

![](dt=2026-03-30/ht=21/7516d7ff24914f61e2e7e086f2f562824bdcb77b61da3616cfc1779de22d78d9.jpg)

![](dt=2026-03-30/ht=21/db81e4129fb16dc5d6eb73934c8912be7605e18200a2434434e3fdc27463ed66.jpg)

In order to obtain more insight in the mechanism behind the improvement in efficacy, pharmacokinetic (PK) studies were carried out to
obtain information about the exposure of the ADC in CES1C knockout mice. As can be seen in Figure 5C and Table 3, the site-specifically conjugated ADC2 HC-41C shows a longer half-life $(t_{\frac{1}{2}})$ and, due to a lower clearance (CL), an approximate two-fold higher exposure compared to the randomly conjugated ADC2. Cmax and volume of distribution (Vss) values were similar.

Table 3. PK parameters for ADC2 random and ADC2 HC-41C

![](dt=2026-03-30/ht=21/3a19bb1e3f9cc7c867f9b5bd8735da3ef0658ab69ad7c875c2a1f8c6aae3a4fb.jpg)

<table><tr><td>PK parameter</td><td>t½(h)</td><td>Cmax(μg/mL)</td><td>AUC(h□μg/mL)</td><td>CL(mL/h/kg)</td><td>Vss(mL/kg)</td></tr><tr><td>ADC2 random</td><td>113</td><td>54.5</td><td>4138</td><td>0.72</td><td>104</td></tr><tr><td>ADC2 HC-41C</td><td>273</td><td>53.0</td><td>9360</td><td>0.32</td><td>118</td></tr></table>

Page 11 of 29

Bioconjugate Chemistry

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

![](image)

Abbreviations: $t_{\frac{1}{2}} =$ terminal half-life, $C_{\max} =$ maximum concentration, AUC= area under the curve, CL= clearance, $V_{ss} =$ volume of distribution at steady state

# Improving the manufacturing process through the introduction of a selective reduction step

During the course of these studies we found that the generation of site-specific ADCs using the reduction/oxidation protocol described previously, $^{14}$ resulted in a significant amount of disulfide scrambling (incorrect reformation of interchain disulfide bonds) upon re-oxidation of the interchain disulfide bonds. The disulfide-scrambled product that we found to be formed was a half antibody, composed of a light chain and a heavy chain in which the hinge disulfide bonds were not correctly reformed between the heavy chains, but between the two cysteines within one heavy chain instead.

Since we believed that re-oxidation by DHAA caused the formation of half antibodies, we tried to limit the amount of reducing agent in order to selectively reduce the HC-41 cysteines and leave the interchain disulfide bonds intact. However, it was found that TCEP did not have a preference for the engineered cysteine at HC-41C, as treatment of the capped antibody with two equivalents of TCEP followed by conjugation with the linker-drug resulted in a mixture of random and HC-41C conjugated species.

In order to obtain a selective reduction, in which only the engineered cysteines are reduced and the interchain disulfide bonds are left intact, we reasoned that a more hydrophobic phosphine might have a preference for the reduction of the HC-41C, since it could bind inside the Fab cavity, analogous to the binding of the linker-drug in the ADC. We started out to test a small set of commercially available triphenylphosphine derivatives 1-4 as shown in Figure 6A.

![](dt=2026-03-30/ht=21/4d120ad2069a99fd1708e94f50e920a4ab035b77b9d20120a2929d409adbd2dc.jpg)

Bioconjugate Chemistry

Page 12 of 29

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

The presence of one or more sulfonyl groups ensured that the phosphines were soluble in water and buffers. After screening for selectivity and reactivity towards the interchain disulfide bonds, we found that reducing agent 2, known as 2-(diphenylphosphino)benzenesulfonic acid (diPPBS), fulfilled our criteria. Remarkably, incubating a capped antibody with a 32-fold excess of reducing agent 2 for 8 days at room temperature did not lead to any detectable amount of free light- or heavy chains as analyzed by nonreducing SDS-PAGE.

In sharp contrast, the other reducing agents that were tested (1, 3 and 4) showed full reduction of the disulfide bonds under these conditions as evidenced by the presence of only a heavy chain and a light chain band (Figure 6B). Apparently, having the sulfonyl-group ortho to the phosphine is critical for selective reduction, as the reducing agents 1, 3 and 4, having meta- and para substitutions, do not show selective behavior. We were able to monitor the progress of the selective reduction via RP-HPLC, because the capped and uncapped antibodies were shown to have different retention times.

After screening for the most optimal selective reduction conditions we found that pH and reducing agent concentration were the parameters that had most influence on the kinetics of the reduction reaction (See Figure S4). By varying the pH and the amount of reducing agent we concluded that pH 5 was the most optimal pH and 16-32 equivalents of reducing agent were sufficient to drive the reduction to completion in about 16 hrs which is suitable for application in a GMP manufacturing process.

Although attempts were made to further optimize reducing agent 2 via structural alterations, we were unable to improve it in terms of reaction rate and equivalents used. Using optimized conditions, as detailed in the Materials and Methods section, ADCs could be generated that were similar in terms of DAR (1.8) and %HMW (1-2%) species when compared to the reduction/oxidation protocol, but differed markedly in terms of the amount of half-antibody present.

As shown in Figure 7, the reduction/oxidation protocol led to an increase of a half-antibody product as indicated by non-reduced SDS page with a band at 75 kDa (Figure 7C, lane 2). This band was not increased in the ADC that resulted from the selective reduction protocol (Figure 7C, lane 3).

Page 13 of 29

Bioconjugate Chemistry

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

![](dt=2026-03-30/ht=21/9f5a63453b563118a623f88afdd84409cd42625e22832ddd56a78f48e49c1c2d.jpg)

![](dt=2026-03-30/ht=21/9e2b9e74eb726bc36b165a98514b3326ae761db8fb1bc9de7fbb2b2601ff495f.jpg)

![](dt=2026-03-30/ht=21/06935a2e9244eb539b1a5314775d3ef9beae442d73462d41b721224b9dcae1d4.jpg)

A side-by-side analytical comparison (HIC, SEC and RP-HPLC) between the ADCs resulting from the reduction/oxidation protocol and the selective reduction protocol can be found in Figure S5. Besides the improved product quality, the use of the newly found selective reducing agent decreased the number of process steps in the manufacturing process of site-specific ADCs. There is no need for the re-oxidation step using DHAA, which in terms of process characterization and logistics is very beneficial.

Moreover, whereas following the progress of the reduction of the engineered cysteine with TCEP would require an analytical MS method, the progress of the reduction of the engineered cysteine can now be followed using a convenient RP-HPLC method, which allows for simple in-process control measurements and expedites antibody-specific process optimization
.

# Conclusions

We have shown here that by using an in silico docking method, new IgG cysteine mutants were identified for conjugating payloads. Without exception, the docking runs placed the linker-duocarmycin inside the cavity that is naturally present in the Fab part of an IgG antibody. Several positions for introducing a

Bioconjugate Chemistry

Page 14 of 29

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

cysteine mutation were proposed and conjugation of these engineered sites with the linker-drug resulted in ADCs that were less hydrophobic than the corresponding random DAR2 conjugated species. Furthermore, aggregation tendency was very low as indicated by the small amounts of high molecular weight species present. It is hypothesized that (part of) the hydrophobic payload was shielded from its surroundings by the Fab cavity, resulting in less overall hydrophobicity of the ADC.

Within the set of mutants that were tested, a subset was identified (HC-40C, HC-41C, LC-40C and LC-41C) in Framework Region 2 of the variable part that had corresponding ADCs with an efficient linker-drug coupling, low amounts of high molecular weight species and pronounced shielding effects, with HC-41C being the preferable site as this site demonstrated a RH of as low as 0.3.

Despite conjugating a hydrophobic payload at these sites in the variable part, binding of the ADC to its target antigen was preserved and in vitro cell killing experiments showed that these mutants were equally effective in killing cancer cells as their randomly conjugated counterparts.

An exploratory in vivo xenograft model employing target-expressing BT-474 cells showed an improved anti-tumor response for the site-directed conjugated ADC as compared to the randomly conjugated ADC which is in line with a two-fold increase in exposure of the site-specific ADC as compared to the random conjugate.

Lastly, although the manufacturing process for cysteine-engineered ADCs using reduction/oxidation with TCEP/DHAA is well known and widely applied, we found that the re-oxidation step with DHAA of a fully reduced antibody leads to the formation of scrambled disulfide species as evidenced by the formation of half-antibody.

Inspired by the shielding effects observed when a hydrophobic payload was introduced in the Fab cavity of an antibody, we identified a reducing agent, 2-diphenylphosphino benzenesulfonic acid (diPPBS), that had no reductive capacity towards interchain disulfide bonds, but was able to reduce the (capped) engineered cysteines in the Fab cavity overnight in a robust manner. This led to the absence of the scrambled disulfide species in the end product and an overall simplification of the manufacturing process.

We believe that this newly developed platform enables us to generate high-quality ADCs bearing hydrophobic payloads that can be manufactured in a robust and simple manner, which will lead to more efficacious targeted therapeutics for cancer treatment. SYD1875, a 5T4-targeting ADC employing the site-specific conjugation of vc-seco-DUBA at HC-41C, has been manufactured under GMP conditions using the selective reduction process and is currently in clinical testing for advanced solid tumors (ClinicalTrials.gov Identifier: NCT04202705).

# Materials and Methods

# Modeling and docking

# Homology modeling

Page 15 of 29

Bioconjugate Chemistry

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

Homology modeling was performed using the YASARA software package (version 13.9.8). The sequence of the antibody Fab fragment to be modeled was entered as a suitable formatted FASTA file. The Fab sequences consisted of the heavy chain variable domain with the IgG1 constant domain 1 (VH+CH1[lgG1]), and the complete light chain Ig K sequence (VL+CL[lg-κ]).

Default settings for the YASARA homology experiment were used, with the exception that the number of potential templates to be evaluated was increased from 5 to 15. To evaluate the stability and suitability of the obtained models they were subjected to short Molecular Dynamics (MD) simulations using the YASARA2 force field (T=310K, up to 50ps) followed by simulated annealing. The MD simulations did not significantly alter the structures.

# Linker-drug docking on Fab models and X-ray Fab structures.

A 3D model of the linker-drug (see Figure 1) was created and optimized using the YASARA software package. No attempt was made to perform a full conformational analysis of this highly flexible molecule. Instead a single suitable low energy conformation was obtained by a MD simulation in YASARA, followed by simulated annealing. Docking of the linker-drug onto Fab models was performed using the Autodock VINA algorithm as incorporated in the YASARA software package.

The standard YASARA dock_run macro was used, which performs 25 docking runs with the ligand (the linker-drug) fully flexible and the protein ligand rigid. The simulation cell defined around the receptor molecule limits the allowable positions of the ligand in contact with the receptor. Its size was initially chosen to allow docking at all possible positions on and in the receptor molecule. Additional docking runs were performed with a simulation cell positioned to limit docking to the cavity of the Fab fragment only.

# Protein Production and Purification

See Supporting Information.

# ADC synthesis

General protocol conventional site-specific conjugation (reduction/oxidation)

A solution of cysteine-engineered antibody (250 μl, 48 mg/ml in 15 mM histidine, 50 mM sucrose, 0.01% polysorbate-20, pH 6) was diluted with 4.2 mM histidine, 50 mM trehalose, pH 6 (750 μl), and EDTA (25 mM in water, 4% v/v). The pH was adjusted to ~7.4 using TRIS.HCl (1 M in water, pH 8) after which tris(2-carboxyethyl)phosphine hydrochloride (TCEP.HCl, 10 mM in water, 20 equivalents) was added and the resulting mixture was incubated at room temperature (rt) for 1-3 hrs. The excess TCEP was removed by a Vivaspin centrifugal concentrator (30 kDa cut-off, polyethersulfone (PES)) using 4.2 mM histidine, 50 mM trehalose, pH 6. The pH of the resulting antibody solution was raised to ~7.4 using TRIS.HCl (1 M in water,

Bioconjugate Chemistry

Page 16 of 29

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

pH 8) after which dehydroascorbic acid (DHAA, $10~\mathrm{mM}$ in water, 20 equivalents) was added and the resulting mixture was incubated at rt for 2 hrs. Dimethylacetamide (DMA) was added followed by a solution of linker-drug (10 mM in DMA). The final concentration of DMA was 5-10%. The resulting mixture was incubated at rt in the absence of light for 1-2 hrs. In order to remove the excess of linker drug, activated charcoal was added and the mixture was incubated at rt for at least 0.5 hr. The charcoal was removed using a $0.2\mu \mathrm{m}$ PES filter and the resulting ADC was formulated in $4.2~\mathrm{mM}$ histidine, $50~\mathrm{mM}$ trehalose, pH 6 using a Vivaspin centrifugal concentrator (30 kDa cut-off, PES).

# General proto
col selective site-specific reduction and conjugation

A solution of cysteine-engineered antibody (10-15 mg/ml, pH 5, 100 mM histidine)) was treated with 2-(diphenylphosphino) benzenesulfonic acid (diPPBS, 16-32 equivalents, 10 mM in water) and the resulting mixture was incubated at rt for 16-24 hrs. The excess diPPBS was removed by a centrifugal concentrator (Vivaspin filter, 30 kDa cut-off, PES) using 4.2 mM histidine, 50 mM trehalose, pH 6 or by carbon filtration.

The pH of the resulting antibody solution was raised to $\sim 7.4$ using TRIS (1 M in water, pH 8) or kept at pH 6 after which DMA was added followed by a solution of linker-drug (10 mM in DMA). The final concentration of DMA was 5-10%. The resulting mixture was incubated at rt in the absence of light for 2-3 hrs or overnight in case of pH 6. In order to remove the excess of linker drug, activated charcoal was added and the mixture was incubated at rt for at least 0.5 hrs.

The charcoal was removed using a 0.2 μm PES filter and the resulting ADC was formulated in 4.2 mM histidine, 50 mM trehalose, pH 6 using a Vivaspin centrifugal concentrator (30 kDa cut-off, PES).

# In vitro cell viability

LNCap-C4.2 and MDA-MB-468 cells were obtained from ATCC. No further cell line authentication was conducted. LNCap-C4.2 was cultured in complete RPMI-1640 media (Lonza) supplemented with $10\%$ v/w qualified FBS (Gibco-Life Technologies) at $37^{\circ}\mathrm{C}$ , $5\%$ $\mathrm{CO}_{2}$ . MDA-MB-468 cells were cultured at $37^{\circ}\mathrm{C}$ , $5\%$ $\mathrm{CO}_{2}$ in complete RPMI-1640 media supplemented with $10\%$ v/w FBS, which was heat inactivated (Gibco).

For viability studies, cells were plated in growth medium/FBS in 96-well plates ( $90~\mu\mathrm{l}/\mathrm{well}$ ) and incubated at $37^{\circ}\mathrm{C}$ , $5\%$ $\mathrm{CO}_{2}$ at a density of 1000 LNCap-C4.2 or 5000 MDA-MB-468 cells per well. After an overnight incubation, $10~\mu\mathrm{l}$ of ADC was added. Serial dilutions were made in culture medium. Cell viability was assessed after 6 days using the CellTiter-Glo™ (CTG) luminescent assay kit from Promega Corporation (Madison, WI) according to the manufacturer's instructions and as detailed previously.[10]

Page 17 of 29

Bioconjugate Chemistry

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

# In vivo xenograft study

The in vivo anti-tumor activity of the ADCs was tested as a single dose therapy at $3\mathrm{mg / kg}$ in a BT-474 cell line xenograft model (Oncodesign, Dijon, France) in CES1c -/- SCID mice. Studies were conducted as previously described. $^{12}$ All in vivo studies and protocols were approved by the local animal care and use committee according to established guidelines.

# PK study

Site-directed and randomly conjugated PSMA-targeting ADCs (ADC1 random and ADC1 HC-41C) and 5T4-targeting ADCs (ADC2 random and ADC2 HC-41C) were dosed at $3\mathrm{mg / kg}$ via an intravenous bolus injection into the tail vein of CES1c $- / -$ mice. Blood samples were taken from the tail vein at multiple time points after dosing, cooled on ice water and processed to plasma as soon as possible. Plasma samples were snap frozen in liquid nitrogen and stored at $-80^{\circ}\mathrm{C}$ until bioanalysis. An ELISA-based method was used for determination of conjugated ADC levels as described previously. $^{10}$ PK parameters were calculated in WinNonlin version 6.3.

# Immunohistochemistry staining

Paraffin sections $(4\mu m)$ were prepared from BT474 tumors. Immunostainings were performed on the Discovery XT (Ventana Medical Systems, Inc., USA). Antigen retrieval was performed using the CC1 antigen retrieval buffer (1 hr at $95^{\circ}C$ ), and slides were incubated for 12 hrs with the primary anti-5T4 mouse monoclonal antibody (ab134162), diluted 1:200 in Discovery antibody diluent/Reaction buffer (1:1, v/v). Binding was visualized using the OmniMap anti-Rb HRP DAB detection system (30 min.) and a manual DAB chromogeny procedure. Sections were counterstained for 10 min. with Hematoxylin II and 5 min. with Bluing Reagent.

# Acknowledgements

This research was funded by Byondis B.V.

# Supporting Information Description

Protein production and purification protocols, analytical methods, potential attachment sites in Fab2, relative hydrophobicity of antibody-(HC 41C)-vc-seco-DUBA ADCs, full HIC chromatograms for ADC1 random, ADC1 HC-120C and ADC1 HC-41C, cellular binding and cellular cytotoxicity profiles, analytical data for several HC-41C ADCs, analytical data for cysteine-engineered vcMMAE and mcMMAF ADCs, kinetic data of selective reduction, HIC, SEC and RPHPLC analysis of HC-41C ADCs that result from the reduction/oxidation and selective reduction protocols.

Bioconjugate Chemistry

Page 18 of 29

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

# Abbreviations

ADC, antibody-drug conjugate; DAR, drug-to-antibody ratio; MMAE, monomethyl auristatin E; PK, pharmacokinetics; MMAF, monomethyl auristatin F; PEG, polyethylene glycol; HIC, hydrophobic interaction chromatography; PBD; pyrrolobenzodiazepine; HMW, high molecular weight; Fc, fragment crystallizable; Fab, fragment antigen-binding; LC, light chain; HC, heavy chain; VC, valine-citrulline; DUBA, duocarmycin hydroxybenzamide azaindole; TCEP, tris(2-carboxyethyl)phosphine; RT, retention time; RH, relative hydrophobicity; FR, framework region; CDR, complementarity-determining region; CES1C,

carboxylesterase 1C; $t_{\%}$ , terminal half-life; $C_{\max}$ , maximum concentration; AUC, area under the curve; CL, clearance; $V_{ss}$ , volume of distribution at steady state; DHAA, dehydroascorbic acid; diPPBS, 2-(diphenylphosphino) benzenesulfonic acid; SDS-PAGE, sodium dodecyl sulfate polyacrylamide gel electrophoresis; GMP, good manufacturing practice; RP-HPLC, reversed phase high performance liquid chromatography.

# References

Page 19 of 29

Bioconjugate Chemistry

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

Bioconjugate Chemistry

Page 20 of 29

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

Page 21 of 29

Bioconjugate Chemistry

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

![](dt=2026-03-30/ht=21/31fa2b1fc1f68f26576a6027a282a0d112b98a2e26584f8a73752847caea18bb.jpg)

![](image)
/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-03-30/ht=21//f631ce67162a36430433811577bbba36b3a3d977eda6718838aa6aedd41de364.jpg)

Bioconjugate Chemistry

Page 22 of 29

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

![](dt=2026-03-30/ht=21/e1e311d65e60a66e018ca511835d5afca1eccb9f06d193c26c2df86e81ac5c25.jpg)

$85 \times 58 \mathrm{~mm}$ (300 x 300 DPI)

Page 23 of 29

Bioconjugate Chemistry

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

![](dt=2026-03-30/ht=21/6277be925c2c12d8868474943e3114912d935d409d9b2f14bbed50c1f8c52c5d.jpg)

177x100mm (300 x 300 DPI)

Bioconjugate Chemistry

Page 24 of 29

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

![](dt=2026-03-30/ht=21/ab44d0091b7bb82a95fb27d09d8962788972b602303abb2ffaa5daff5e2b3bad.jpg)

177x123mm (300 x 300 DPI)

Page 25 of 29

Bioconjugate Chemistry

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

![](dt=2026-03-30/ht=21/40bc87eb14f8fc030ac6ee7338fccd795fedeaa81e09e09580ea977b35b54816.jpg)

![](dt=2026-03-30/ht=21/64b8c671ac81758963fdfdbfd37ace9d3eb64553883477cb1afc1f8437f54caa.jpg)

![](dt=2026-03-30/ht=21/7c21eae9fad09954c6fd4d50c69256d447532953a296c18b1f05fc25f558bae2.jpg)

177x98mm (300 x 300 DPI)

Bioconjugate Chemistry

Page 26 of 29

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

![](dt=2026-03-30/ht=21/8ea41a1fbd1e7f6a457a11e60d0c8d010dfe6dd81a79e6cbfb7fb2da9d02b8ec.jpg)

84x47mm (300 x 300 DPI)

Page 27 of 29

Bioconjugate Chemistry

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

![](dt=2026-03-30/ht=21/f58259c433911937d338b35469448ca318b0de04f91dbea79b384b72cf8660a4.jpg)

![](dt=2026-03-30/ht=21/300d01c99192842e87c185a39ce7c135dfdc8d48fbcc76d4c8abf48ca919f3f4.jpg)

177x110mm (300 x 300 DPI)

Bioconjugate Chemistry

Page 28 of 29

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

![](dt=2026-03-30/ht=21/491dd6e67783c7ee0ba1dab939164b57251105bf4105b895703dd7ab2e74daf3.jpg)

Page 29 of 29

Bioconjugate Chemistry

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