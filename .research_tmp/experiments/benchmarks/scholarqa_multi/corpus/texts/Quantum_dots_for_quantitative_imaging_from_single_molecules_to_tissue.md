# Quantum dots for quantitative imaging: from single molecules to tissue

Tania Q. Vu · Wai Yan Lam · Ellen W. Hatch · Diane S. Lidke

Received: 31 October 2014 / Accepted: 4 December 2014 / Published online: 27 January 2015
© Springer-Verlag Berlin Heidelberg 2015

Abstract Since their introduction to biological imaging, quantum dots (QDs) have progressed from a little known, but attractive, technology to one that has gained broad application in many areas of biology. The versatile properties of these fluorescent nanoparticles have allowed investigators to conduct biological studies with extended spatiotemporal capabilities that were previously not possible. In this review, we focus on QD applications that provide enhanced quantitative information concerning protein dynamics and localization, including single particle tracking and immunohistochemistry, and finish by examining the prospects of upcoming applications, such as correlative light and electron microscopy and super-resolution. Advances in single molecule imaging, including multi-color and three-dimensional QD tracking, have provided new insights into the mechanisms of cell signaling and protein trafficking. New forms of QD tracking in vivo have allowed the observation of biological processes at molecular level resolution in the physiological context of the whole animal. Further methodological development of multiplexed QD-based immunohistochemistry assays should enable more quantitative analysis of key proteins in tissue samples. These advances highlight the unique quantitative data sets that QDs can provide to further our understanding of biological and disease processes.

Keywords Quantum dots (QDs) · Single particle tracking · Immunohistochemistry · Fluorescence microscopy

# Introduction

In 1998, two papers appeared back-to-back in Science (Bruchez et al. 1998; Chan and Nie 1998) describing the first applications of fluorescent semiconducting nanocrystals or quantum dots (QDs) to biological imaging. The critical advance demonstrated in these papers was the development of water-soluble QDs that could be conjugated to biomolecules for molecular targeting. The studies included the targeting of QDs to living cells via ligand coupling (Chan and Nie 1998) and multi-color labeling of structures in fixed cells (Bruchez et al. 1998). Since these seminal papers, the application of QDs in bio-imaging has rapidly expanded to include many modalities that cover multiple time and length scales, from single molecule to in vivo imaging (Fig. 1). The reason for their widespread use comes from several key advantages that QDs provide over conventional fluorophores (see Table 1). In particular, QDs have high photostability such that long-term imaging can be achieved without artifacts from photobleaching. Additionally, the broad absorption spectra and narrow emission spectra allow the simultaneous excitation of spectrally distinct QDs and easy spectral separation of emission for multiplex imaging. A number of excellent reviews contain detailed information concerning QD chemistry and photophysical properties (Michalet et al. 2005; Giepmans

Wai Yan Lam and Ellen W. Hatch contributed equally to this work.

This work was supported by NIH 1RO1NS071116, NIH 1R21NS073113 to T.Q.V., and the OHSU Neuroscience Imaging Center (P30-NS061800); NIH 1R01GM100114 and NSF MCB-0845062 to D.S.L., and the NM Spatiotemporal Modeling Center (NIH P50GM085273). E.W.H. was supported by an NM Cancer Nanotechnology Training Grant.

T. Q. Vu (☒) · W. Y. Lam
Department of Biomedical Engineering, School of Medicine, Oregon Health and Science University, Portland, Ore., USA
e-mail: vuta@ohsu.edu

E. W. Hatch · D. S. Lidke (☒)
Department of Pathology and Cancer Research and Treatment Center, University of New Mexico School of Medicine,
Albuquerque, N.M., USA
e-mail: dlidke@salud.unm.edu

![](images/e13af10cc74e46d99975cee0825ce2147414831c333f0de89323617d39ddc431.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing a fluorescently labeled structure with a 0.5μm scale bar (no text or symbols beyond label)
</details>

![](images/2e5f3c5d2a21a11eb86fa28f6b803f4d9fcb5c721633d54697e8c2b635402a90.jpg)

<details>
<summary>natural_image</summary>

Fluorescent microscopy image of a single cell with green and orange staining, scale bar 10μm (no text or symbols)
</details>

![](images/fe131726d4c7ecdaa03d37f2c44609ed595aae5d2ce1542f589604355a385d01.jpg)

<details>
<summary>natural_image</summary>

Fluorescence microscopy image showing green-labeled cellular structures against a red background, with 5μm scale bar (no text or symbols beyond label)
</details>

![](images/e9890e89c4aa37080ac632fc94c7d328abb1c4e5722b571c96e50a212af732cf.jpg)

<table><tr><td></td><td>Single Molecule</td><td>Single Cell</td><td>Tissue</td><td>In vivo</td></tr><tr><td>Modality</td><td>widefieldconfocalTIRF</td><td>widefieldconfocalTIRF</td><td>widefieldconfocalspectral imaging</td><td>whole animalintravitaltwo-photon</td></tr><tr><td>Spatial resolution</td><td>nm</td><td>μm</td><td>μm</td><td>μm-mm</td></tr><tr><td>Temporal resolution</td><td>ms</td><td>ms-min</td><td>N/A</td><td>s-hr</td></tr></table>

Fig. 1 Quantum dots (QDs) are used in a range of biological imaging techniques. a Single molecule detection provides high spatiotemporal resolution (TIRF total internal reflection fluorescence microscopy). Example of two-color QD tracking of QD-labeled epidermal growth factor (EGF) bound to EGF receptor (EGFR; see also Low-Nam et al. 2011). b Live cell imaging captures dynamics of cellular processes. Image shows QD-EGF (red) binding to EGFR (green) on the surface of an A431 cell

et al. 2006; Pons and Mattoussi 2009; Pinaud et al. 2010; Petryayeva et al. 2013). Here, we highlight unique biological imaging applications that have been enabled by QDs, namely high-resolution imaging of protein behavior at the single molecule level and developments in multi-color quantitative immunohistochemistry (IHC). Advances in bioconjugation techniques and applications in correlative light and electron microscopy (CLEM) and super-resolution are also discussed.

# QD-enabled studies of single molecule behavior in living cells

The elucidation of complex biological phenomena requires approaches that reveal the dynamic behaviors and organization of molecules in living systems. QD single particle tracking (QD-SPT) represents a powerful method for probing the dynamics of these individual proteins of interest in living cells with high spatial and temporal resolution. This capability is afforded by their high photostability and brightness, which is superior to conventional fluorophores (fluorescent proteins and organic dyes). These advantageous properties overcome difficulties in photobleaching that limit fluorophore imaging time thereby allowing for the acquisition of biological events over long timescales, and contribute to the utility of QDs as an ultrasensitive detection probe for SPT. Moreover, QD-SPT generates quantifiable dynamic information regarding diffusional properties, co-localization, and spatial and temporal heterogeneity of molecules inside living cells, none of which conventional fluorescence and biochemical methods can capture (Courty et al. 2006a; Cognet et al. 2014; Breger et al. 2014).

The method of QD-SPT proceeds through multiple steps. Briefly, the first step involves generating a QD probe targeting (see also Lidke et al. 2004). c QDs used in immunohistochemistry (IHC) assays allow multiplex imaging (N/A not applicable). Example of two-color QD-IHC in human spleen tissue with QD-labeled antibodies against mast cell tryptase (green) and c-Kit (red; image by E.W. Hatch, Lidke Laboratory). d QDs can be visualized by in vivo imaging. Image shows simultaneous in vivo imaging of spectrally distinct QD-encoded microbeads. Image courtesy of X. Gao (see also Gao et al. 2004)

the molecule of interest. A number of strategies are available for targeting QDs to bio-molecules of interest in living cells (Medintz et al. 2005; Petryayeva et al. 2013). Second, once the QD probe is generated, a series of validations must be conducted to confirm that the QD probe binds with specificity to its cellular target, and that its function is not sterically hindered by QD size. Specificity of binding should be validated by comparing the cellular labeling of QDs with and without components necessary for binding, e.g., streptavidin-QDs alone versus streptavidin-QDs coupled to the biotinylated targeting molecule. Methods for validating retention of biological function include comparing QD-labeled versus fluorescent-dye-labeled (Cy3, Alexa dyes, fluorogen-activating proteins) targets and/or gold particle probes to ensure similar diffusion properties (Dahan et al. 2003; Bannai et al. 2006; Groc et al. 2007; Schwartz et al. 2014), measurement of protein signaling activity with QDs tagged to either ligand or protein (Lidke et al. 2004; Andrews et al. 2008), measurement of cellular activity such as cell outgrowth and survival with and without QDs (Cui et al. 2007; Vermehren-Schmaedick et al. 2014), and comparison of internalization kinetics in receptors pre-labeled with QDs to receptors post-labeled with QDs following fixation (Fichter et al. 2010). Following labeling and validation of the QD probe, a sequence of fluorescence images are acquired by time lapse photography to capture the biological event. The temporal resolution achieved by this acquisition will usually be limited by the camera readout time rather than by the brightness of the probe (Courty et al. 2006a). Finally, biological information is extracted from the recorded trajectories through single particle tracking to yield measurements such as diffusion coefficient and velocity that provide dynamic information about the molecule of interest (Bannai et al. 2006). A variety of commercial (IDL-Research Systems, Boulder, Colo., USA),

Table 1 Relative comparison of quantum dots (QDs) versus organic fluorophores. The relative performance level in each category is indicated: best, intermediate, and poor 

<table><tr><td>Parameter</td><td>QDs</td><td>Organic dyes</td><td>Fluorescent proteins</td></tr><tr><td>Size</td><td>1, poor</td><td>Best</td><td>Intermediate</td></tr><tr><td>Photostability</td><td>Best</td><td>2, intermediate</td><td>2, poor</td></tr><tr><td>Broad excitation</td><td>3, best</td><td>Intermediate</td><td>Intermediate</td></tr><tr><td>Multiplex imaging</td><td>4, best</td><td>Intermediate</td><td>Intermediate</td></tr><tr><td>Brightness</td><td>5, best</td><td>Intermediate</td><td>6</td></tr><tr><td>Flexible conjugation</td><td>7, best</td><td>Best</td><td>Intermediate</td></tr><tr><td>Genetically expressible</td><td>8, poor</td><td>8, poor</td><td>Best</td></tr><tr><td>Electron dense</td><td>Best</td><td>9, poor</td><td>9, poor</td></tr><tr><td>Sample fixation requirements</td><td>10, intermediate</td><td>Best</td><td>Best</td></tr><tr><td>Continuous emission</td><td>11, poor</td><td>Best</td><td>Best</td></tr><tr><td>Accessible time/length scales</td><td>12, intermediate</td><td>13, intermediate</td><td>13, intermediate</td></tr><tr><td>In vivo imaging</td><td>14, intermediate</td><td>Intermediate</td><td>15, best</td></tr></table>

$^{1}$ QDs are larger than organic fluorophores (10–20 nm in diameter). This can lead to steric interference with protein function and must be carefully tested

$^{2}$ Photobleaching can be advantageous for techniques such as fluorescence recovery after photobleaching (FRAP). Organic dye derivatives have improved photostability (Altman et al. 2011)

$^{3}$ QDs have a large Stoke's shift and broad excitation into the ultraviolet, allowing for simultaneous excitation of distinct QD species

$^{4}$ QDs have narrow emission spectra that fit to a Gaussian profile, making spectral unmixing more straightforward

$^{5}$ Although peak emission rate of QDs is less than that of organic fluorophores (QDs have a longer fluorescence lifetime), their high extinction coefficient ( $\varepsilon$ ) and quantum yield (QY) result in higher brightness ( $\varepsilon^{*}$ QY)

$^{6}$ New fluorescent proteins demonstrate improved brightness (Shaner et al. 2013)

$^{7}$ QDs have a range of conjugation schemes; however, truly monovalent coupling is difficult to achieve

$^{8}$ Although fluorescent proteins are the only directly genetically expressible fluorophores, a number of small genetically expressible peptides can be used to target QDs and organic dyes (Regoes and Hehl 2005; Jacquier et al. 2006; Szent-Gyorgyi et al. 2008)

$^{9}$ The QD core is inherently electron dense, but some organic fluorophores can provide contrast in electron microscopy (Shu et al. 2011)

$^{10}$ QDs have special requirements for fixation (paraformaldehyde must be used; methanol or cold fixation must be avoided) and mounting medium (some mounting media lead to QD signal degradation; nonpolar organic-solvent-based reagents are recommended, see Table 3)

$^{11}$ QDs demonstrate intermittent fluorescence. Groups are working on the generation of non-blinking QDs (Ghosh et al. 2012). Typically considered a disadvantage in single molecule imaging, although blinking properties have been used in super-resolution (Lidke et al. 2005a; Lagerholm et al. 2006; Dertinger et al. 2009)

$^{12}$ QD probes can be used in techniques that cover all spatiotemporal scales for biological imaging

$^{13}$ Photobleaching of organic fluorophores makes longer-term imaging difficult

$^{14}$ QDs are available in near infrared wavelengths with a high two-photon cross-section and the potential for use as a theranostic, but concerns exist regarding toxicity

$^{15}$ Bright and stable near-infrared fluorescent proteins are being developed (Filonov et al. 2011)

customized, and open source software (ImageJ plugins such as Particle Tracker, Manual Tracking) are available for conducting single QD tracking. Subsequent biophysical analyses of the data include computing molecular dynamic information of free and confined diffusive processes (Dahan et al. 2003; Lidke et al. 2005a; Bannai et al. 2006; Crane et al. 2008; Chang et al. 2012), molecular state changes such as dimerization dynamics including dimer status, rates of dimerization, and dimer diffusivity (Chung et al. 2010; Low-Nam et al. 2011) and receptor protein trafficking dynamics (Pierobon et al. 2009; Valentine et al. 2012). Through the capabilities afforded by the sensitivity of QD-SPT, many groups have exploited the potential of QDs to unravel complex biological processes previously thought impossible. We highlight examples demonstrating recent work involving QD-SPT for tracking extracellular and intracellular protein targets.

QDs have served as a powerful tool for single particle tracking of membrane and cytoplasmic targets in a number of live cell studies (Fig. 2, Table 2). The accessibility of membrane targets for QD labeling makes them ideal targets for QD tracking. Since the initial papers demonstrating QD tracking of membrane receptors (Dahan et al. 2003; Lidke et al. 2004), a wide variety of membrane proteins have been studied by using live-cell QD single particle tracking. These include dissecting the dynamic behavior of membrane-bound targets such as receptors, channels, transporters, and

a   
![](images/8e81f02606fa4d7432e9e5c4fc9f342932fe777f7913c7226f2c8b1a7ad8684b.jpg)

<details>
<summary>line</summary>

| Distance (μm) | Time (s) |
| ------------- | -------- |
| 0             | 0        |
| 1             | 40       |
| 2             | 35       |
| 3             | 30       |
| 4             | 25       |
| 5             | 20       |
| 6             | 15       |
| 7             | 10       |
| 8             | 5        |
| 9             | 0        |
| 10            | 0        |
| 11            | 0        |
| 12            | 0        |
| 13            | 0        |
| 14            | 0        |
| 15            | 0        |
| 16            | 0        |
| 17            | 0        |
| 18            | 0        |
| 19            | 0        |
| 20            | 0        |
| 21            | 0        |
| 22            | 0        |
| 23            | 0        |
| 24            | 0        |
| 25            | 0        |
| 26            | 0        |
| 27            | 0        |
| 28            | 0        |
| 29            | 0        |
| 30            | 0        |
| 31            | 0        |
| 32            | 0        |
| 33            | 0        |
| 34            | 0        |
| 35            | 0        |
| 36            | 0        |
| 37            | 0        |
| 38            | 0        |
| 39            | 0        |
| 40            | 0        |
| 41            | 0        |
| 42            | 0        |
| 43            | 0        |
| 44            | 0        |
| 45            | 0        |
| 46            | 0        |
| 47            | 0        |
| 48            | 0        |
| 49            | 0        |
| 50            | 0        |
| 51            | 0        |
| 52            | 0        |
| 53            | 0        |
| 54            | 0        |
| 55            | 0        |
| 56            | 0        |
| 57            | 0        |
| 58            | 0        |
| 59            | 0        |
| 60            | 0        |
| 61            | 0        |
| 62            | 0        |
| 63            | 0        |
| 64            | 0        |
| 65            | 0        |
| 66            | 0        |
| 67            | 0        |
| 68            | 0        |
| 69            | 0        |
| 70            | 0        |
| 71            | 0        |
| 72            | 0        |
| 73            | 0        |
| 74            | 0        |
| 75            | 0        |
| 76            | 0        |
| 77            | 0        |
| 78            | 0        |
| 79            | 0        |
| 80            | 0        |
| 81            | 0        |
| 82            | 0        |
| 83            | 0        |
| 84            | 0        |
| 85            | 0        |
| 86            | 0        |
| 87            | 0        |
| 88            | 0        |
| 89            | 0        |
| 90            | 0        |
| 91            | 0        |
| 92            | 0        |
| 93            | 0        |
| 94            | 0        |
| 95            | 0        |
| 96            | 0        |
| 97            | 0        |
| 98            | 0        |
| 99            | 0        |
| Note: The image contains a color scale (blue to red) and a zoomed-in view (red circle) at the center of the plot. The color scale is labeled 'Time (s)' and the x-axis is labeled 'Distance (μm)'. The y-axis is labeled 'Time (s)' and the color scale is labeled 'Time (s)' with a color bar ranging from blue to red. There is no label for the data series.
</details>

b   
![](images/e4978d636865ceefcdc761cf77efbbb9b70557b4a20809d8e09460d7ac5ccb29.jpg)

<details>
<summary>text_image</summary>

1µm
3
1
3
t (seconds)
y (µm)
x (µm)
λ value
700
650
600
550
2
1
0
-2
1
2
4
6
8
10
12
14
2
3
4
5
6
7
8
9
10
12
14
</details>

Fig. 2 Single and multi-color QD tracking of extracellular and intracellular molecular processes. a Single QD tracking reveals heterogeneous trafficking dynamics of brain-derived neurotrophic factor (BDNF)-TrkB receptor transport in neurons. Left Receptor trajectory overlaid on wide-field cell image (gray; WGA wheat germ agglutinin). Right Magnified view of trajectory showing region of confinement. Images from Vermehren-Schmaedick et al. 2014; reprinted, with permission of the   
investigators. b Multicolor QD tracking of EGFRs by using hyperspectral microscopy allows the tracking of up to eight spectrally distinct QDs. Left Trajectories of selected QDs overlaid on hyperspectral image. Right Particle trajectories plotted over time corresponding to the regions boxed (1–3; left). Color map (right) indicates QD emission peak. Images from Cutler et al. 2013; reproduced, with permission of the investigators

membrane components. The first single QD tracking experiments (Dahan et al. 2003) revealed the lateral diffusion of glycine receptors in living neurons for up to 20 min and showed that these diffusion behaviors varied in the different synaptic domains. Here, we describe recent examples of the use of QDs for understanding the dynamic properties of other membrane-bound components such as membrane receptors, lipid raft constituents, and water channels (Crane et al. 2008) in a variety of neuronal and epithelial cell systems (see Table 2). These include studies of the cystic fibrosis transmembrane conductance regulator (CFTR) revealing previously uncharacterized immobile behavior attributable to C-terminal PDZ interactions (Haggie et al. 2006) and the tracking of the aquaporin water channels that demonstrate non-anomalous diffusion (Crane and Verkman 2008). Chang and Rosenthal (2012) have studied the diffusion properties of lipid rafts by attaching QDs to the lipid raft constituent, GM1 ganglioside; they have found that lateral confinement persists on similar time scales to the signaling of raft-associated proteins, offering support for the role of lipid rafts as a possible signaling platform. In another study, the same group has employed the use of antagonist-conjugated QDs for the study of the serotonin transporter and found that populations of the transporter residing in cholesterol and ganglioside-GM1-enriched microdomains display restricted mobility in comparison with the freely diffusing population of transporters not localized to these regions; these findings suggest a role for membrane microdomains in aspects of transporter regulation (Chang et al. 2012). Together, these elegant studies serve to demonstrate the unique capability of QDs to explore the dynamic contributions of complex biological systems.

New studies have also focused on tracking membrane protein trafficking or the movement of specific proteins from the extracellular surface to the intracellular environment in cells. Two areas of research include the QD tracking of epidermal growth factor receptors (EGFR) and neuronal growth factor receptors. In neuronal systems, QDs have been most readily utilized in the study of the binding and transport of neuronal receptor complexes such as nerve growth factor (NGF)-TrkA and brain-derived neurotrophic factor (BDNF)-TrkB (Fig. 2a; Cui et al. 2007; Rajan et al. 2008; Vermehren-Schmaedick et al. 2014). In EGFR systems, Lidke et al. (2004, 2005a) demonstrated the first instance of the use of QDs for studying EGFR endosomal trafficking and found that EGFR is trafficked to the cell body along filopodia in a retrograde manner. Since their study, a number of other groups have

Table 2 Example of targets for QD single particle tracking (CFTR cystic fibrosis transmembrane conductance regulator, AQP aquaporin, OAP orthogonal array of particles, PDX pancreatic duodenal homeobox, EGFR epidermal growth factor receptor, WGA wheat germ agglutinin, NGF nerve growth factor, BDNF brain-derived neurotrophic factor, NA not applicable) 

<table><tr><td>Target</td><td>Number of colors</td><td>Novel findings</td><td>Cell</td><td>Bioconjugation</td><td>Delivery</td><td>Location</td></tr><tr><td colspan="7">Single-color</td></tr><tr><td>CFTR (Haggie et al. 2006)</td><td>NA</td><td>CFTR immobilization due to C-terminal PDZ interactions</td><td>Epithelial (MDCK, COS7, HT29)</td><td>Anti-HA IgG, biotin-Fab, streptavidin-QD</td><td>NA</td><td>Extracellular</td></tr><tr><td>Ganglioside GM1 (lipid raft constituent; Chang and Rosenthal 2012)</td><td>NA</td><td>GM1 complexes show lateral confinement persisting in orders of tens of seconds</td><td>Serotonergic neurons (RN46A)</td><td>Biotin-cholera toxin B subunit (binds to GM1), streptavidin-QD</td><td>NA</td><td>Extracellular</td></tr><tr><td>AQP4 (Crane et al. 2008)</td><td>NA</td><td>AQP4 is responsible for OAP formation and stability, independent of cytoskeletal or PDX interactions</td><td>Primary astrocytes from mice Epithelial (MDCK, CHO-K1, COS-7)</td><td>Anti-c-myc IgG, IgG-QD</td><td>NA</td><td>Extracellular</td></tr><tr><td>EGFR (Chung et al. 2010)</td><td>NA</td><td>Measured dimerization dynamics of individual EGFRs in various spatial contexts</td><td>Epithelial (CHO-K1)</td><td>EGFR-Fab-QD</td><td>NA</td><td>Extracellular</td></tr><tr><td>EGFR (Lidke et al. 2004, 2005b)</td><td>NA</td><td>EGF bound to EGFR on filopodia of cells and transported in a retrograde manner towards cell body</td><td>Epithelial (CHO, A431, HeLa, MCF7)</td><td>Biotin-EGF, streptavidin-QD</td><td>Endocytosis (facilitated)</td><td>Extracellular and Intracellular</td></tr><tr><td>EGFR (Li et al. 2012)</td><td>NA</td><td>Duration of EGFR endosomal trafficking is reduced upon paclitaxel treatment</td><td>Epithelial (A549)</td><td>Biotin-EGF, streptavidin-QD</td><td>Endocytosis (facilitated)</td><td>Extracellular and Intracellular</td></tr><tr><td>EGFR to observe endosomal trafficking (Zajac et al. 2013)</td><td>NA</td><td>Crowded intracellular environment impacts endosomal motility</td><td>Epithelial (Arpe-19)</td><td>Biotin-EGF, streptavidin-QD</td><td>Endocytosis (facilitated)</td><td>Extracellular and Intracellular</td></tr><tr><td>WGA (S.-L. Liu et al. 2011)</td><td>NA</td><td>Capability to visualize endocytosis and exocytosis, and transport of lectins</td><td>Epithelial (A549)</td><td>Biotin-WGA, streptavidin-QD</td><td>Endocytosis (facilitated)</td><td>Extracellular and Intracellular</td></tr><tr><td>Influenza virus (S.-L. Liu et al. 2012)</td><td>NA</td><td>Capability to visualize virus infection behavior</td><td>Epithelial (MDCK)</td><td>Biotin-virus, streptavidin-QD</td><td>Endocytosis (facilitated)</td><td>Extracellular and Intracellular</td></tr><tr><td>TrkA (Rajan et al. 2008)</td><td>NA</td><td>NGF bound to TrkA follows diffusive and active endosomal transport dynamics</td><td>Neuronal (PC12)</td><td>Biotin-NGF, streptavidin-QD</td><td>Endocytosis (facilitated)</td><td>Extracellular and Intracellular</td></tr><tr><td>TrkB (Vermehren-Schmaedick et al. 2014)</td><td>NA</td><td>BDNF bound to TrkB traffics in a heterogeneous fashion within the cytosol</td><td>Primary nodose ganglion sensory neurons</td><td>Biotin-BDNF, streptavidin-QD</td><td>Endocytosis (facilitated)</td><td>Extracellular and Intracellular</td></tr><tr><td>IBB domain of snurportin-1 (to study transport across nuclear pore complex; Lowe et al. 2010)</td><td>NA</td><td>Import proceeds by nuclear transport substeps consisting of cargo capture, filtering, translocation, and release into nucleus</td><td>Epithelial (HeLa)</td><td>QDs functionalized to IBB z-domain through maleimide and sulphydryl reaction</td><td>Digitonin permeabilization (active)</td><td>Intracellular</td></tr><tr><td>Myosin and kinesin (Nan et al. 2005)</td><td>NA</td><td>Able to observe 8-nm steps taken by motor proteins along microtubules</td><td>Epithelial (A549)</td><td>Biotin-peptide, streptavidin-QD</td><td>Endocytosis (passive)</td><td>Intracellular</td></tr><tr><td>Cytosol (no specific target; Keren et al. 2009)</td><td>NA</td><td>Measured fluid flow in lamellipodia of moving cells, found fluid flow is directed from cell body towards leading edge</td><td>Fibroblasts (keratocytes)</td><td>None</td><td>Electroporation (active)</td><td>Intracellular</td></tr><tr><td colspan="7">Multi-color</td></tr><tr><td>EGFR liganded/unliganded (Low-Nam et al. 2011)</td><td>2</td><td>Measurements of dimer off rates</td><td>Epithelial (A431)</td><td>Biotin-EGF, streptavidin-QD Biotin-VHH, streptavidin-QD</td><td>NA</td><td>Extracellular</td></tr></table>

Table 2 (continued) 

<table><tr><td>Target</td><td>Number of colors</td><td>Novel findings</td><td>Cell</td><td>Bioconjugation</td><td>Delivery</td><td>Location</td></tr><tr><td>GM1/CD59/EGFR (Clausen et al. 2014)</td><td>Up to 4</td><td>Simultaneous tracking of three independent membrane components by using filter-based approach</td><td>Fibroblast (MEF)</td><td>QD-cholera toxin to GM1 QD-CoA to ACP-tagged CD59 SAV-QD to biotinylated EGFR (BLAP-tag)</td><td>NA</td><td>Extracellular</td></tr><tr><td>FcεRI/EGFR (Cutler et al. 2013)</td><td>Up to 8</td><td>Simultaneous tracking of up to 8 spectrally distinct QDs by using spectral imaging</td><td>Mast cells (RBL)</td><td>Biotin-IgE, streptavidin-QD Biotin-EGF, streptavidin-QD</td><td>NA</td><td>Extracellular</td></tr></table>

exploited the use of QDs for exploring other aspects of EGFR dynamic behavior including drug effects on the duration of EGFR endosomal trafficking (Li et al. 2012), motility of early endosomes transporting EGFR as cargo (Zajac et al. 2013), and dimerization activity of liganded and unliganded EGFR in various spatial contexts (Chung et al. 2010). QD-enabled studies of NGF axonal transport by using biotin-NGF and streptavidin-QDs have been able to uncover previously unrecognized “stop-and-go” motions of NGF during retrograde transport (Cui et al. 2007) and to demonstrate the diffusive and active transport dynamics of NGF endosomal trafficking (Rajan et al. 2008). Similar use of BDNF-QDs has enabled the study of the cytoplasmic trafficking dynamics of BDNF-TrkB endosomes and revealed that BDNF undergoes a combination of rapid and directed motions interspersed with circuitous meanderings, within the cell body, that are characterized by mobile and immobile phases (Vermehren-Schmaedick et al. 2014). The use of QDs for tracking biomolecules that move from the cell surface into the cell cytoplasm also includes non-receptor targets such as viral particles (S.-L. Liu et al. 2012) and lectins (S.-L. Liu et al. 2011). The application of QDs will probably continue to expand further in the near future to other extracellular and extracellular/intracellular targets.

Whereas QDs have experienced a growth in their application to the study of extracellular targets, the use of QDs for the study of intracellular targets remains an area of slower growth because of the technical challenge of QD delivery into the cell interior. Despite these challenges, several groups have demonstrated successful targeting of QDs to intracellular targets. Studies have used non-specific membrane fusion to deliver endocytosed QDs into the cell interior. For example, in early work, Nan et al. (2005) employed tracked motor proteins transporting QD-endocytic cargoes to measure, at high spatial resolution, the 8-nm steps taken by microtubule motors in the plus- and minus-end directions. Using a different technique to deliver QDs directly into the cytosol and to bypass endocytic routes, Courty et al. (2006b) tracked QD-conjugated kinesin, enabling single molecule characterization of the dynamics of individual intracellular kinesin. QDs have also been used to probe exclusively intracellular events such as the transport of QD-cargoes through nuclear pore complexes (Lowe et al. 2010), fluid flow in lamellipodia through non-specifically delivered QDs (Keren et al. 2009), and diffusion of QD-mRNA in interchromatin regions (Ishihama and Funatsu 2009).

These and other studies have adopted a variety of strategies for introducing QDs into the cytosol, and these include: (1) passive delivery, (2) facilitated delivery, and (3) active delivery of QDs into the cytosolic environment. Passive delivery of QDs occurs through the induction of endocytic uptake through the inherent physical properties (surface coating and charge) of QDs (Nan et al. 2005). Facilitated delivery of QDs occurs through association of the QD with a peptide or protein

to promote intracellular uptake (Derfus et al. 2004; Hild et al. 2008; Delehanty et al. 2009) or through the use of methods such as pinocytosis (Courty et al. 2006b) and transfection reagents (Xu et al. 2013). Recent application of active delivery of QDs into the cellular environment include methods such as electroporation (Keren et al. 2009) and microinjection (Xu et al. 2013) and novel methods such as photothermal nanoblade delivery (Xu et al. 2012). Whereas successful cytosolic delivery of QDs has been demonstrated, ensuring QD nonspecific binding in the absence of functional modification in these systems remains a challenge. Attempts to demonstrate specific binding and functional integrity include a comparison of the kinetics and behaviors of delivered QD-conjugates that are functionally versus non-functionally competent; for example, intact kinesin-QD conjugates versus denatured kinesin-QD conjugates (Courty et al. 2006b) and QD conjugates with and without domains necessary for nuclear import (Lowe et al. 2010). Although strides have been made in strategies for the cytosolic delivery of QDs, the continued development of targeting strategies that improve delivery efficiency and specificity while retaining molecular function is necessary for the exploration, in greater detail, of the complex biological processes that occur inside cells.

# Multi-color single QD tracking

Often in biological studies, it is valuable to visualize protein behavior with respect to other proteins or their environment. The large Stokes shift of QDs makes them a good choice when imaging simultaneously blue-shifted organic dyes or green fluorescent protein (GFP). For example, by using two-color total internal reflection fluorescence (TIRF) microscopy, the motion of individual QD-labeled IgE receptors has been tracked with respect to the landscape of the membrane proximal actin bundles, labeled by GFP-actin (Andrews et al. 2008). Simultaneous imaging of the receptor and actin has provided direct proof of the ability of actin to restrict membrane protein motion. Consistent with the actin corral hypothesis, the long QD tracks enable the characterization of IgE receptor mobility when near actin and have shown that the receptor is deflected by the actin boundary.

Because of their broad absorption spectra and narrow emission spectra, QDs are also ideally suited for multi-color imaging at the single molecule level. Two-color single QD tracking is relatively easy to achieve by using a beam-splitter to separate the emission into separate spectral channels. Simultaneous imaging of distinctly tagged proteins allows for the visualization of protein-protein interactions at the single molecule level. For example, on homodimerization of the EGFR, erbB2 (HER2) and erbB3 (HER3) have been captured and quantified in live cells (Lidke et al. 2005a; Low-Nam et al. 2011; Steinkamp et al. 2014). Labeling the various protein species with spectrally distinct QDs also allows the direct comparison of protein diffusion or the capturing of heterodimer interactions (Low-Nam et al. 2011; Steinkamp et al. 2014). You et al. (2014) have used the pair correlation of dual-color imaging experiments to monitor QD-labeled interferon a2 (IFNa2) binding to its receptor, IFNAR2, and the recruitment of STAT2 to IFNAR2. Torreno-Pina et al. (2014) have used two-color single QD tracking as part of a study to examine the role of glycans in the micropatterning of the plasma membrane. By comparing mobility and protein interactions between DC-SIGN and a mutant that is not glycosylated, they found that glycan-mediated interactions did not lead to higher order clustering. However, glycosylated DC-SIGN exhibited a more restricted mobility, and these glycan-based interactions are important in regulating DC-SIGN interactions with clathrin (Torreno-Pina et al. 2014).

Typically, experiments focus on two-color tracking because of technical limitations of employing beam-splitters to separate the emission light into independent channels. Recently, several groups have developed methods for higher multiplexing capabilities. Lagerholm and colleagues have described a four-color beam-splitter approach that allows the simultaneous tracking of four distinct QD species (Arnspang et al. 2012). Recently, Clausen et al. (2014) used this technology to track three independent membrane components simultaneously: CD59 (a glycophosphatidylinositol-anchored protein), EGFR (a transmembrane protein), and GM1 (a lipid). Tracking of each species demonstrated that, whereas each has a distinct diffusive behavior, the mobility of each is reduced in response to cholesterol depletion (Clausen et al. 2014).

Keith Lidke and colleagues have developed a high-speed hyperspectral microscope that is not limited by filter-based detection but that acquires the full spectra of the sample in every pixel (Cutler et al. 2013). This line-scanning confocal instrument allows the simultaneous single QD tracking of up to eight spectrally distinct QDs at 30 frames/s (Fig. 2b). The ability to increase labeling up to eight colors allows higher density labeling, increasing the probability of capturing protein-protein interactions and enabling high spatiotemporal resolution in diffusion maps (Cutler et al. 2013). Since the confocal entrance slit to the spectrometer provides optical sectioning, the tracking of membrane protein motion at the apical cell surface is easily achieved.

# Tracking single molecule motion in three dimensions

The advances in multiplex imaging have opened new avenues for investigation of membrane component behavior. However, these experiments are still realistically limited to a two-dimensional focal plane. A number of groups have developed instrumentation, based on a range of approaches, that allows single particle tracking in three dimensions (Kao and Verkman

1994; Schütz et al. 2001; Prabhat et al. 2007; Watanabe et al. 2007; Lessard et al. 2007; Wells et al. 2009; Welsher and Yang 2014). In several cases, the brightness and photostability of QDs have been critical to the successful application of three-dimensional (3D) tracking. Ober and colleagues have designed an instrument that enables the simultaneous imaging of multiple focal planes in a sample (Prabhat et al. 2007). Using this microscope, they have monitored protein endocytosis, recycling, and exocytosis in real-time (Prabhat et al. 2007; Ram et al. 2008). By following the endocytic trafficking of QD-labeled transferrin, they have examined the intercellular transfer of cargo between adjacent cells (Ram et al. 2012). Izeddin et al. (2012) have used adaptive optics and point-spread function (PSF) engineering to track QD-labeled platelet-derived growth factor (PDGF) receptor motion in three dimensions. With this approach, they achieved $15\mathrm{-nm}$ z-precision and captured the 3D landscape of the plasma membrane (Izeddin et al. 2014). Werner and colleagues have developed a 3D tracking microscope that uses quad-APD detectors to monitor the emission of a single QD probe and track its motion in x, y, and z directions by moving the microscope stage to always keep the QD centered on the detector (Lessard et al. 2007; Wells et al. 2009). This instrument has been used to capture endocytosis dynamics of the activated IgE receptor (Wells et al. 2009), as the receptor undergoes large $(>1\mu \mathrm{m})$ changes in its z position when being trafficked from the plasma membrane to the cytosol. Recently, the group has collaborated with the Hollingsworth group to bioconjugate their giant non-blinking QDs (Ghosh et al. 2012). The stable emission from these QDs allows much longer-term tracking of the IgE receptor motion in three dimensions on the cell surface (Keller et al. 2014). Welsher and Yang (2014) have combined 3D tracking with two-photon microscopy, which permits the simultaneous imaging of cellular structure while single molecules are tracked. With this approach, they have been able to capture the real-time binding of individual Tat-coated QD nanoparticles to the cell surface (Welsher and Yang 2014).

# Single QD tracking in whole animals

QDs have been used in in vivo animal models as imaging agents, enabling the high contrast visualization of the vasculature and lymph nodes, and as nanoparticle platforms to track the pharmaco-biodistribution of drugs and other biomolecules (Larson et al. 2003; Kim et al. 2004; Michalet et al. 2005; Diagaradjane et al. 2008; Jung et al. 2011). An emerging area is the single particle tracking of QDs in in vivo animal models. In one of the first studies of single QD tracking in vivo, Tada et al. (2007) demonstrated the real-time tracking of single QDs in live in vivo animal preparations; they imaged the movement of QDs that were conjugated to the antibody drug Herceptin in mice containing tumors with overexpressed HER2 breast tumors. By using a confocal microscope with a dorsal skinfold chamber, they observed the movement of QDs from the blood into and within the tumor and were able to quantitate information such as velocity, direction, and modes of transport. Hamada et al. (2011) counted single QD-VEGF probes in movies made in the blood vessels of ischemic mouse models undergoing angiogenesis to examine the molecular distribution of vascular endothelial growth factor (VEGF) with high spatial resolution. Another investigation that exemplifies the use of QDs in high-resolution measurements of in vivo biological processes has involved the use of QDs to track the motion of the protease-activated receptor 1 (PAR1) on the surface of tumor cells in order to study the membrane dynamics at high spatial resolution (\~8 nm) during the process of extravasation (Gonda et al. 2010). Moreover, QDs have been employed to measure, in real time, the length changes in the sarcomeres of myocytes in vivo by infusing QD solution over intact myocytes and allowing their internalization via membrane fusion (Serizawa et al. 2011). Such studies, conducted in in vivo preparations, are valuable in that one can look at biological processes that occur on the resolution scale of tens of nanometers in the physiologically relevant context of the larger scale biological system.

# QDs for multiplex IHC

In addition to advancing single molecule and single cell imaging, QDs are being used to improve the multiplexing capabilities and sensitivity of IHC, a well-established method of diagnosis in surgical pathology enabling the in situ identification of characteristic antigens indicative of cell type and origin and of disease state. Typically, IHC of pathological samples involves the labeling of paraformaldehyde-fixed paraffin-embedded tissue with antibodies to specific targets. The extent of antibody labeling is traditionally detected by enzyme-substrate-based chromogenic reporters and a transmission light microscope. Fluorescence is becoming an attractive alternative to chromogenic detection in IHC. Specifically, fluorescence has an advantage when it comes to quantification, since the amount of labeling scales linearly with fluorescence intensity, whereas enzyme-based deposition is dependent on parameters such as time of incubation, temperature, and concentration of the substrate. Additionally, analysis of chromogenic substrates is only semi-quantitative, typically utilizing an H-score or other qualitative analysis (Barrow et al. 2011; Gonda et al. 2012).

In the early 2000s, QD-based IHC (QD-IHC) protocols were developed and have since been applied to a range of tissues (Sun et al. 2001; Zahavy et al. 2005; True and Gao 2007; Byers and Hitchman 2011). The unique properties of QDs (Table 1) provide enhance capabilities for sensitivity, multiplexing, and quantification. These permit a more detailed

analysis of tissue structure, cellular localization, relative amounts of antigens, and colocalization and require only the sparing use of limited tissue samples, such as biopsy tissue (Akhtar et al. 2007; Caldwell et al. 2008; Yu et al. 2013).

The QD-IHC protocol has come a long way from its debut in 2001, and many publications have examined the special considerations for fluorescence and QD label usage (Table 3). In addition to the considerations for traditional IHC, such as sample care and antigen retrieval, Xing et al. (2007) and Montón et al. (2012) provide a particularly rigorous analysis of reagents that may alter fluorescent signals and recommend conjugation and multiplexing methods. Additionally, key reviews have chronicled the progress of QD-IHC methodologies (Byers and Hitchman 2011; Chen et al. 2012; Fang et al. 2012; Kairdolf et al. 2013).

Sensitivity is an important characteristic of IHC for its traditional applications in diagnostic medicine, such as for the detection of receptor targets before the initiation of therapies including those targeting the estrogen receptor (ER; Hammond et al. 2010; Gonda et al. 2012). Many studies have validated the detection by QDs in IHC (Chen et al. 2009; Xu et al. 2012; Tabatabaei-Panah et al. 2013; Yu et al. 2013). The recent key studies that have demonstrated the statistical and functional advantages of QD-IHC over other forms of IHC include work in which QD-IHC has been found to have 5 % better sensitivity and 10 % better specificity over traditional IHC when examining the Tn antigen in breast cancer tissues (Au et al. 2014) and demonstrations that the QD-IHC detection of the proliferation marker Ki67 is able to predict more accurately 5-year disease-free survival in breast cancer patients (Sun et al. 2014). Additionally, QD-IHC is amenable to coupling with signal-amplifying techniques such as tyramide signal amplification (TSA) to enhance further the detection of subtle antigenic signal (Akhtar et al. 2007).

Multiplex QD-IHC has allowed new insights to be made into the spatial organization and biomarker relationships in disease processes. Since the advent of multi-color QD-IHC in 2005 (Zahavy et al. 2005), studies have examined the colocalization of biomarkers (Storch et al. 2007), tissue and microenvironment heterogeneity and co-evolution (Chen et al. 2010; J. Liu et al. 2010a, 2010b; Faratian et al. 2011), and structural changes, such as invasion (Xing et al. 2007; Peng et al. 2011; X.-L. Liu et al. 2011; Fang et al. 2012). An example of multiplex QD-IHC in tissue is shown in Fig. 3. J. Liu et al. (2010a) targeted multiple antigens implicated in prostate cancer to create a signal map of patient tissues that cumulatively indicated architectural abrogation and demonstrated the potential of QD-IHC to reconstruct malignant transformation of a heterogeneous tissue. Peng et al. (2011) have demonstrated patterns of spatial and temporal co-evolution of gastric and breast cancer cells and their surrounding stroma by following markers of basement membrane integrity or breakdown, angiogenesis, and macrophage invasion. Additionally, QD-IHC has been combined with other techniques such as QD in situ hybridization (QD-ISH) to examine RNA transcripts and to provide additional spatial information (Matsuno et al. 2005, 2006).

Despite the advantages of QD-IHC for achieving quantitative information, only relative quantification to date has been implemented, in practice through various means such as spectral image acquisition, region of interest assignment, and signal discretization, and the subsequent quantification of intensity values. Hardware and software are major components dictating the quality of data collected from QD-IHC. Whereas hardware is beyond the scope of this review, we should mention the state of software options. Many packages are available to separate fluorescence signatures in spectral images; the most published is the commercially available Nuance software (PerkinElmer, Mass., USA) and spectral capabilities of ZEN software (Carl Zeiss Microimaging). Many groups additionally are writing their own unmixing

Table 3 Key differences between traditional and QD-immunohistochemistry (IHC) protocols   
\*An important underlying theme in the difference between traditional and QD-IHC protocols is to utilize solvents such as toluene, chloroform, and hexane to prevent any interaction with QD surface chemistry possibly leading to quenching 

<table><tr><td>IHC workflow</td><td>QD-IHC specific</td></tr><tr><td>Sample preparation</td><td>Works well with fresh, frozen, or formalin-fixed paraffin-embedded samples</td></tr><tr><td>Deparaffinization/rehydration</td><td>Toluene recommended (replacing xylene*)</td></tr><tr><td>Decloaking of antigens</td><td>Same as traditional; temperature, pressure, pH, and time of cycles is crucial for retrieval of epitopes</td></tr><tr><td>Blocking and antibody incubation</td><td>Hydrophilic barrier pens such as ImmEdge by Vector* (Xing et al. 2007)</td></tr><tr><td>Dehydration and mounting</td><td>Toluene recommended (replacing xylene*), both for rinses and mounting media</td></tr><tr><td>Microscopy</td><td>Spectral camera and software recommended (e.g., Nuance)</td></tr><tr><td>Analysis</td><td>Spectral unmixing; generation of spectral fingerprints or comparisons of label intensity</td></tr></table>

![](images/5c7c7493ecacc4c72c5b4dd499c9ed7be8f7888ab5958ebb8eccb387bb3c43c0.jpg)

<details>
<summary>natural_image</summary>

Microscopic tissue image showing cellular structures with red and green fluorescent markers, labeled regions i, ii, iii, and a 15 μm scale bar (no textual content beyond labels)
</details>

Fig. 3 Multiplex QD-immunohistochemistry (QD-IHC). a QD-IHC performed on human spleen tissue of a patient with aggressive systemic mastocytosis (tissue courtesy of Tracy George). Staining includes anti-CD31 (QD565, blue), anti-tryptase (QD585, green), and anti-cKit (QD655, red), with autofluorescence represented as white. Note, in the image, a CD31+ splenic capillary (blue) filled with red blood cells;

protocols, which are similarly based on spectral separation, such as via a Gaussian mixture model, defining and thresholding a masked region of interest, and quantifying intensity based on maxima and total area. Examples include the WuDa Image Analysis System (Wuhan University; Lv et al. 2013), inForm (Caliper Life Sciences, Hopkinton, Mass., USA), and the open source wares FARSIGHT and Q-IHC (www.miblab.org; Xing et al. 2007; Yu et al. 2013). Analysis might expand on this relative quantification by providing a comparison between signals, in which one common housekeeping antigen might be used as an internal control to draw conclusions about the relative quantity of biomarkers. Common housekeeping targets include nuclear stains and proteins intimately involved in normal cellular processes, such as elongation factor 1 alpha (EF1α; Xing et al. 2007). Quantitative multiplex QD-IHC is an area of open opportunity by which to exploit the unique properties of QD photostability, multiplexed emission, and bright intensity to enable improved tissue diagnostics.

b   
Spectral Curves   
![](images/2be9748dabb28439df0a1885c692f66bd8bf26fec07fdff01d4c42c50e0c7d8b.jpg)  
surrounding it is an aggregate of mast cells that are positive for both tryptase and cKit biomarkers. b Spectral curves of the corresponding boxes (i–iii) in a. The splenic capillary is high in QD565 signal that labels CD31 (i), and mast cells show high labeling of QD585-tryptase and QD655-cKit (ii). Images acquired by E.W. Hatch by using a Nuance spectral camera; spectral unmixing performed with custom software

# Progress in QD bioconjugation paradigms

QDs must be bioconjugated with specific biomolecules (e.g., proteins, DNA, ligands, drugs) in order to conduct the intended biological applications such as imaging, delivery, and biosensing (Medintz et al. 2005; Petryayeva et al. 2013). Typically, QD bioconjugation entails the attachment of biomolecules to amphiphilic polymers that are assembled at the QD surface and serve to render the QD water-soluble (Medintz et al. 2005). QD bioconjugation methodologies that are currently used to generate functionalized targeted probes have been derived from standard protein-labeling chemistries and have been described in recent reviews (Petryayeva et al. 2013; Blanco-Canosa et al. 2014). Covalent chemistries are the preferred method of choice for use in cellular studies because of the more stable nature of these reactions. Covalent chemistries include carbodiimide chemistries, which offer cheap and fast covalent binding to the amine and carboxyl groups of the biomolecule of interest (Hermanson

2013). A disadvantage of these chemistries, however, is a lack of precise control over the orientation and stoichiometry of cross-linked biomolecules at the QD surface. Furthermore, aggregation is often a concern. Non-covalent biotin-streptavidin interactions are likely the most widely used for cellular-QD applications. This is because of the versatility of biotin-streptavidin bonds, which offer high affinity, wide pH, and salt stability (Hermanson 2013) and the practical ease of the use and some quantitative control of biomolecular valency at the QD surface. Biotinylation kits and biotinylated biomolecules are available from a wide range of suppliers, and streptavidin-QDs are commercially available and can be easily paired with a protein of interest. Other chemistries successfully used include self-assembly by histidine-metal affinity; this involves a His-appended biomolecule that interacts with the inorganic ZnS shell of the QD (Blanco-Canosa et al. 2014).

Despite the availability of methods in use, new bioconjugation chemistries will still be needed to continue to generate QD bioconjugates with the improved precision control and specific functionality necessary to achieve specific biological tasks (Zrazhevskiy et al. 2010). Alternative bioconjugation schemes that produce a repeatable product, control the valency of the biomolecule on the QD surface, produce a desired orientation of the biomolecule on the QD surface, and do not compromise the function of the biomolecule still await further developments, which could provide a greater impetus for the expansion and establishment of QD applications in biology and medicine. Along these lines, recent efforts include work carried out to improve QD probes for single protein tracking by producing monovalent QD bioconjugates by means of peptide surface coatings and by the demonstration of new bioconjugation schemes employing hydrazide, aldehyde, and thiol-based linkages (You et al. 2010; Clarke et al. 2010; Iyer et al. 2011). The development of new bioconjugation schemes will be powerful and necessary for achieving multifunctional tasks such as combined imaging, targeting, and delivery (Zrazhevskiy et al. 2010). Development of the use of specific types of QD bioconjugates together with their integration into novel methodologies will also be important for achieving multifunctional imaging, targeting, and delivery tasks (Zrazhevskiy et al. 2010). Recent work along these lines includes flexible adaptable procedures designed to tap the full potential of multicolor QDs for tagging multiple targets in the same biological sample (Zrazhevskiy and Gao 2013; Zrazhevskiy et al. 2013). Another emerging development is the use of “click” chemistries that are appealing because of their rapidity, high yield, and ease of use at room temperature. Click chemistries originally employed copper (I)-catalyzed azide-alkylene reactions, which are well-suited for high-selectivity, because these groups do not interact with native biological functional groups (Kolb et al. 2001). Click chemistries have been used to bioconjugate biomolecules to magnetic, gold, and other colloidal nanoparticles. Because copper might alter the luminescent properties of QDs, recently available copper-free bio-orthogonal approaches (Bernardin et al. 2010; Han et al. 2010; Schieber et al. 2012) have made possible the use of click chemistries for the bioconjugation of QDs; these, in the near future, are likely to grow in application.

# Potential areas for further development of QDs

A potential future opportunity for the application of QDs is in the exciting and growing area of correlated microscopy. CLEM offers the opportunity to visualize fluorescently tagged proteins in their surrounding context at high-resolution by electron microscopy. The dual QD fluorescence and electron-dense properties make QDs advantageous for CLEM (Giepmans et al. 2006; Sjollema et al. 2012). QDs can be distinctly discriminated between at the EM level, and the simultaneous labeling of multiple endogenous proteins has been demonstrated for CLEM in cells and tissue and at the postsection labeling of nuclear proteins (Nisman et al. 2004; Deerinck et al. 2007). The 3D EM reconstruction of QDs in corresponding fluorescence optical sections has been demonstrated to elucidate details of neurofibrillary tangles in Alzheimer's disease in the study by Uematsu et al. (2012), who have also employed energy-dispersive X-ray chemical analysis of Cd and Se to confirm the presence of QDs in appropriate locations in the EM image (Uematsu et al. 2012). A unique attribute of QDs for use in CLEM is the capability to distinguish, in a multiplexed fashion, the multiple QD probes in EM based on their different size and shape and fluorescent color (Sosinsky et al. 2007). As yet, the increased use of QDs in CLEM remains to be fully exploited. Another powerful use of QD correlative microscopy on the horizon is the combination of QD with other super-resolution fluorescence probes to bridge dynamic information of individual proteins in relation to their spatial organization in the cell. For example, information concerning the diffusive motion of glycine receptors by using QD-SPT has been studied in the context of the glycine receptor subsynaptic distribution by using PALM (Specht et al. 2013).

QDs also possess properties that make them uniquely suited for super-resolution imaging. To achieve resolution beyond the diffraction limit, a number of super-resolution techniques rely on the ability to switch the fluorescence emission from emitters on and off. Lidke et al. (2005b) were the first to propose and demonstrate that blinking fluorophores could be used for super-resolution; they use the natural blinking of QDs independently to localize individual emitters with separations of less than the diffraction limit. In a similar manner, Lagerholm et al. (2006) have used blinking to localize QDs attached to the ends of double-stranded DNA that were separated by $42\mathrm{nm}$ . Wang et al. have recently extended this idea to

3D super-resolution (Wang et al. 2013). In 2009, Dertinger et al. applied QDs for SOFI (Super-resolution Optical Fluctuating Imaging) in which the statistical analysis of the fluorescence fluctuations throughout a time series can achieve a super-resolution image (Dertinger et al. 2009, 2013). Whereas photostability is one of the greatest advantages of QDs, a blue-shift could be induced in their emission spectra (“blueing”) in the QD emission in response to high intensity illumination. Hoyer et al. used this property to achieve resolution below the diffraction limit using a simple webcam for detection (Hoyer et al. 2011).

# Concluding remarks

From single molecules to single cells to tissue, QDs have provided unique quantitative data sets for a better understanding of biological and disease processes. The ability of QDs to enhance bio-imaging has been enabled by the development of instrumentation that can take advantage of the photophysical properties of QDs and sophisticated analysis routines to extract biological parameters from single molecule and spectral imaging data. Improvements in QD properties, such as reduced size, constant emission, and monovalent conjugation will further increase their utility. Therefore, although QDs have had a strong impact on biological imaging, we probably have yet to realize the full potential of QDs for quantitative imaging.

Acknowledgments We thank Dr. Tracy George for collaboration in the generation of the QD-IHC samples in Figs. 1, 3.

# References

Akhtar RS, Latham CB, Siniscalco D, Fuccio C, Roth KA (2007) Immunohistochemical detection with quantum dots. Methods Mol Biol 374:11–28. doi:10.1385/1-59745-369-2:11   
Altman RB, Terry DS, Zhou Z, Zheng Q, Geggier P, Kolster RA, Zhao Y, Javitch JA, Warren JD, Blanchard SC (2011) Cyanine fluorophore derivatives with enhanced photostability. Nat Methods 9:68–71. doi:10.1038/nmeth.1774   
Andrews NL, Lidke KA, Pfeiffer JR, Burns AR, Wilson BS, Oliver JM, Lidke DS (2008) Actin restricts FcepsilonRI diffusion and facilitates antigen-induced receptor immobilization. Nat Cell Biol 10:955–963. doi:10.1038/ncb1755   
Arnspang EC, Brewer JR, Lagerholm BC (2012) Multi-color single particle tracking with quantum dots. PLoS One 7:e48521. doi:10.1371/journal.pone.0048521   
Au GH, Mejias L, Swami VK, Brooks AD, Shih WY, Shih WH (2014) Quantitative assessment of Tn antigen in breast tissue micro-arrays using CdSe aqueous quantum dots. Biomaterials 35:2971–2980. doi:10.1016/j.biomaterials.2013.12.034   
Bannai H, Lévi S, Schweizer C, Dahan M, Triller A (2006) Imaging the lateral diffusion of membrane molecules with quantum dots. Nat Protoc 1:2628–2634. doi:10.1038/nprot.2006.429

Barrow E, Evans DG, McMahon R, Hill J, Byers R (2011) A comparative study of quantitative immunohistochemistry and quantum dot immunohistochemistry for mutation carrier identification in Lynch syndrome. J Clin Pathol 64:208–214. doi:10.1136/jcp.2010.084418

Bernardin A, Cazet A, Guyon L, Delannoy P, Vinet F, Bonnaffé D, Texier I (2010) Copper-free click chemistry for highly luminescent quantum dot conjugates: application to in vivo metabolic imaging. Bioconjug Chem 21:583–588. doi:10.1021/bc900564w

Blanco-Canosa JB, Wu M, Susumu K, Petryayeva E, Jennings TL, Dawson PE, Algar WR, Medintz IL (2014) Recent progress in the bioconjugation of quantum dots. Coord Chem Rev 263–264:101–137. doi:10.1016/j.ccr.2013.08.030

Breger J, Delehanty JB, Medintz IL (2014) Continuing progress toward controlled intracellular delivery of semiconductor quantum dots. Wiley Interdiscip Rev Nanomed Nanobiotechnol. doi:10.1002/wnan.1281

Bruchez M, Moronne M, Gin P, Weiss S, Alivisatos AP (1998) Semiconductor nanocrystals as fluorescent biological labels. Science 281:2013–2016. doi:10.1126/science.281.5385.2013

Byers RJ, Hitchman ER (2011) Quantum dots brighten biological imaging. Prog Histochem Cytochem 45:201–237. doi:10.1016/j.proghi.2010.11.001

Caldwell ML, Moffitt RA, Liu J, Parry M, Sharma Y, Wang MD (2008) Simple quantification of multiplexed quantum dot staining in clinical tissue samples. Annu Int Conf IEEE Eng Med Biol Soc 2008:1907–1910. doi:10.1109/IEMBS.2008.4649559

Chan W, Nie S (1998) Quantum dot bioconjugates for ultrasensitive nonisotopic detection. Science 281:2016–2018. doi:10.1126/science.281.5385.2016

Chang JC, Rosenthal SJ (2012) Visualization of lipid raft membrane compartmentalization in living RN46A neuronal cells using single quantum dot tracking. ACS Chem Neurosci 3:737–743. doi:10.1021/cn3000845

Chang JC, Tomlinson ID, Warnement MR, Ustione A, Carneiro AM, Piston DW, Blakely RD, Rosenthal SJ (2012) Single molecule analysis of serotonin transporter regulation using antagonist-conjugated quantum dots reveals restricted, p38 MAPK-dependent mobilization underlying uptake activation. J Neurosci 32:8919–8929. doi:10.1523/JNEUROSCI.0048-12.2012

Chen C, Peng J, Xia HS, Yang GF, Wu QS, Chen LD, Zeng LB, Zhang ZL, Pang DW, Li Y (2009) Quantum dots-based immunofluorescence technology for the quantitative determination of HER2 expression in breast cancer. Biomaterials 30:2912–2918. doi:10.1016/j.biomaterials.2009.02.010

Chen C, Peng J, Xia H, Wu Q, Zeng L, Xu H, Tang H, Zhang Z, Zhu X, Pang D, Li Y (2010) Quantum-dot-based immunofluorescent imaging of HER2 and ER provides new insights into breast cancer heterogeneity. Nanotechnology 21:095101. doi:10.1088/0957-4484/21/9/095101

Chen C, Peng J, Sun SR, Peng CW, Li Y, Pang DW (2012) Tapping the potential of quantum dots for personalized oncology: current status and future perspectives. Nanomedicine (Lond) 7:411–428. doi:10.2217/nnm.12.9

Chung I, Akita R, Vandlen R, Toomre D, Schlessinger J, Mellman I (2010) Spatial control of EGF receptor activation by reversible dimerization on living cells. Nature 464:783–787. doi:10.1038/nature08827

Clarke S, Pinaud F, Beutel O, You C, Piehler J, Dahan M (2010) Covalent monofunctionalization of peptide-coated quantum dots for single-molecule assays. Nano Lett 10:2147–2154. doi:10.1021/nl100825n

Clausen MP, Arnspang EC, Ballou B, Bear JE, Lagerholm BC (2014) Simultaneous multi-species tracking in live cells with quantum dot conjugates. PLoS One 9:e97671. doi:10.1371/journal.pone.0097671

Cognet L, Leduc C, Lounis B (2014) Advances in live-cell single-particle tracking and dynamic super-resolution imaging. Curr Opin Chem Biol 20:78–85. doi:10.1016/j.cbpa.2014.04.015   
Courty S, Bouzigues C, Luccardini C, Ehrensperger MV, Bonneau S, Dahan M (2006a) Tracking individual proteins in living cells using single quantum dot imaging. Methods Enzymol 414:211–228. doi:10.1016/S0076-6879(06)14012-4   
Courty S, Luccardini C, Bellaiche Y, Cappello G, Dahan M (2006b) Tracking individual kinesin motors in living cells using single quantum-dot imaging. Nano Lett 6:1491–1495. doi:10.1021/nl060921t   
Crane JM, Verkman AS (2008) Long-range nonanomalous diffusion of quantum dot-labeled aquaporin-1 water channels in the cell plasma membrane. Biophys J 94:702–713. doi:10.1529/biophysj.107.115121   
Crane JM, Van Hoek AN, Skach WR, Verkman AS (2008) Aquaporin-4 dynamics in orthogonal arrays in live cells visualized by quantum dot single particle tracking. Mol Biol Cell 19:3369–3378. doi:10.1091/mbc.E08-03-0322   
Cui B, Wu C, Chen L, Ramirez A, Bearer EL, Li WP, Mobley WC, Chu S (2007) One at a time, live tracking of NGF axonal transport using quantum dots. Proc Natl Acad Sci U S A 104:13666–13671. doi:10.1073/pnas.0706192104   
Cutler PJ, Malik MD, Liu S, Byars JM, Lidke DS, Lidke KA (2013) Multi-color quantum dot tracking using a high-speed hyperspectral line-scanning microscope. PLoS One 8:e64320. doi:10.1371/journal.pone.0064320   
Dahan M, Lévi S, Luccardini C, Rostaing P, Riveau B, Triller A (2003) Diffusion dynamics of glycine receptors revealed by single-quantum dot tracking. Science 302:442–445. doi:10.1126/science.1088525   
Deerinck TJ, Giepmans BNG, Smarr BL, Martone ME, Ellisman MH (2007) Light and electron microscopic localization of multiple proteins using quantum dots. Methods Mol Biol 374:43–53. doi:10.1385/1-59745-369-2:43   
Delehanty JB, Mattoussi H, Medintz IL (2009) Delivering quantum dots into cells: strategies, progress and remaining issues. Anal Bioanal Chem 393:1091–1105. doi:10.1007/s00216-008-2410-4   
Derfus AM, Chan WCW, Bhatia SN (2004) Intracellular delivery of quantum dots for live cell labeling and organelle tracking. Adv Mater 16:961–966. doi:10.1002/adma.200306111   
Dertinger T, Colyer R, Iyer G, Weiss S, Enderlein J (2009) Fast, background-free, 3D super-resolution optical fluctuation imaging (SOFI). Proc Natl Acad Sci U S A 106:22287–22292. doi:10.1073/pnas.0907866106   
Dertinger T, Pallaoro A, Braun G, Ly S, Laurence TA, Weiss S (2013) Advances in superresolution optical fluctuation imaging (SOFI). Q Rev Biophys 46:210–221. doi:10.1017/S0033583513000036   
Diagaradjane P, Orenstein-Cardona JM, Colón-Casasnovas NE, Deorukhkar A, Shentu S, Kuno N, Schwartz DL, Gelovani JG, Krishnan S (2008) Imaging epidermal growth factor receptor expression in vivo: pharmacokinetic and biodistribution characterization of a bioconjugated quantum dot nanoprobe. Clin Cancer Res 14:731–741. doi:10.1158/1078-0432.CCR-07-1958   
Fang M, Peng C-W, Pang D-W, Li Y (2012) Quantum dots for cancer research: current status, remaining issues, and future perspectives. Cancer Biol Med 9:151–163. doi:10.7497/j.issn.2095-3941.2012.03.001   
Faratian D, Christiansen J, Gustavson M, Jones C, Scott C, Um I, Harrison DJ (2011) Heterogeneity mapping of protein expression in tumors using quantitative immunofluorescence. J Vis Exp 56: e3334. doi:10.3791/3334   
Fichter KM, Flajolet M, Greengard P, Vu TQ (2010) Kinetics of G-protein-coupled receptor endosomal trafficking pathways revealed by single quantum dots. Proc Natl Acad Sci U S A 107:18658–18663. doi:10.1073/pnas.1013763107

Filonov GS, Piatkevich KD, Ting LM, Zhang J, Kim K, Verkhusha VV (2011) Bright and stable near-infrared fluorescent protein for in vivo imaging. Nat Biotechnol 29:757–761. doi:10.1038/nbt.1918   
Gao X, Cui Y, Levenson RM, Chung LW, Nie S (2004) In vivo cancer targeting and imaging with semiconductor quantum dots. Nat Biotechnol 22:969–976. doi:10.1038/nbt994   
Ghosh Y, Mangum BD, Casson JL, Williams DJ, Htoon H, Hollingsworth JA (2012) New insights into the complexities of shell growth and the strong influence of particle volume in nonblinking “giant” core/shell nanocrystal quantum dots. J Am Chem Soc 134:9634–9643. doi:10.1021/ja212032q   
Giepmans BNG, Adams SR, Ellisman MH, Tsien RY (2006) The fluorescent toolbox for assessing protein location and function. Science 312:217–224. doi:10.1126/science.1124618   
Gonda K, Watanabe TM, Ohuchi N, Higuchi H (2010) In vivo nano-imaging of membrane dynamics in metastatic tumor cells using quantum dots. J Biol Chem 285:2750–2757. doi:10.1074/jbc.M109.075374   
Gonda K, Miyashita M, Watanabe M, Takahashi Y, Goda H, Okada H, Nakano Y, Tada H, Amari M, Ohuchi N (2012) Development of a quantitative diagnostic method of estrogen receptor expression levels by immunohistochemistry using organic fluorescent material-assembled nanoparticles. Biochem Biophys Res Commun 426:409–414. doi:10.1016/j.bbrc.2012.08.105   
Groc L, Lafourcade M, Heine M, Renner M, Racine V, Sibarita JB, Lounis B, Choquet D, Cognet L (2007) Surface trafficking of neurotransmitter receptor: comparison between single-molecule/quantum dot strategies. J Neurosci 27:12433–12437. doi:10.1523/JNEUROSCI.3349-07.2007   
Haggie PM, Kim JK, Lukacs GL, Verkman AS (2006) Tracking of quantum dot-labeled CFTR shows near immobilization by C-terminal PDZ interactions. Mol Biol Cell 17:4937–4945. doi:10.1091/mbc.E06-08-0670   
Hamada Y, Gonda K, Takeda M, Sato A, Watanabe M, Yambe T, Satomi S, Ohuchi N (2011) In vivo imaging of the molecular distribution of the VEGF receptor during angiogenesis in a mouse model of ischemia. Blood 118:e93–e100. doi:10.1182/blood-2010-12-322842   
Hammond MEH, Hayes DF, Dowsett M, Allred DC, Hagerty KL, Badve S, Fitzgibbons PL, Francis G, Goldstein NS, Hayes M, Hicks DG, Lester S, Love R, Mangu PB, McShane L, Miller K, Osborne CK, Paik S, Perlmutter J, Rhodes A, Sasano H, Schwartz JN, Sweep FC, Taube S, Torlakovic EE, Valenstein P, Viale G, Visscher D, Wheeler T, Williams RB, Wittliff JL, Wolff AC (2010) American Society of Clinical Oncology/College of American Pathologists guideline recommendations for immunohistochemical testing of estrogen and progesterone receptors in breast cancer (unabridged version). Arch Pathol Lab Med 134:e48–e72   
Han H-S, Devaraj NK, Lee J, Hilderbrand SA, Weissleder R, Bawendi MG (2010) Development of a bioorthogonal and highly efficient conjugation method for quantum dots using tetrazine-norbornene cycloaddition. J Am Chem Soc 132:7838–7839. doi:10.1021/ja101677r   
Hermanson GT (2013) Bioconjugate techniques. Academic Press, Amsterdam   
Hild WA, Breunig M, Goepferich A (2008) Quantum dots—nano-sized probes for the exploration of cellular and intracellular targeting. Eur J Pharm Biopharm 68:153–168. doi:10.1016/j.ejpb.2007.06.009   
Hoyer P, Staudt T, Engelhardt J, Hell SW (2011) Quantum dot blueing and blinking enables fluorescence nanoscopy. Nano Lett 11:245–250. doi:10.1021/nl103639f   
Ishihama Y, Funatsu T (2009) Single molecule tracking of quantum dot-labeled mRNAs in a cell nucleus. Biochem Biophys Res Commun 381:33–38. doi:10.1016/j.bbrc.2009.02.001   
Iyer G, Pinaud F, Xu J, Ebenstein Y, Li J, Chang J, Dahan M, Weiss S (2011) Aromatic aldehyde and hydrazine activated peptide coated

quantum dots for easy bioconjugation and live cell imaging. Bioconjug Chem 22:1006–1011. doi:10.1021/bc100593m   
Izeddin I, El Beheiry M, Andilla J, Ciepielewski D, Darzacq X, Dahan M (2012) PSF shaping using adaptive optics for three-dimensional single-molecule super-resolution imaging and tracking. Opt Express 20:4957–4967. doi:10.1364/OE.20.004957   
Jacquier V, Prummer M, Segura JM, Pick H, Vogel H (2006) Visualizing odorant receptor trafficking in living cells down to the single-molecule level. Proc Natl Acad Sci U S A 103:14325–14330. doi:10.1073/pnas.0603942103   
Jung K-H, Choe YS, Paik J-Y, Lee K-H (2011) 99mTc-Hydrazinonicotinamide epidermal growth factor-polyethylene glycol-quantum dot imaging allows quantification of breast cancer epidermal growth factor receptor expression and monitors receptor downregulation in response to cetuximab therapy. J Nucl Med 52:1457–1464. doi:10.2967/jnumed.111.087619   
Kairdolf BA, Smith AM, Stokes TH, Wang MD, Young AN, Nie S (2013) Semiconductor quantum dots for bioimaging and biodiagnostic applications. Annu Rev Anal Chem (Palo Alto, Calif) 6:143–162. doi:10.1146/annurev-anchem-060908-155136   
Kao HP, Verkman AS (1994) Tracking of single fluorescent particles in three dimensions: use of cylindrical optics to encode particle position. Biophys J 67:1291–1300. doi:10.1016/S0006-3495(94)80601-0   
Keller AM, Ghosh Y, DeVore MS, Phipps ME, Stewart MH, Lidke DS, Wilson BS, Hollingsworth JA, Werner JH (2014) Live cell imaging: 3-dimensional tracking of non-blinking “giant” quantum dots in live cells. Adv Funct Mater 24:4795–4795. doi:10.1002/adfm.201470200   
Keren K, Yam PT, Kinkhabwala A, Mogilner A, Theriot JA (2009) Intracellular fluid flow in rapidly moving cells. Nat Cell Biol 11:1219–1224. doi:10.1038/ncb1965   
Kim S, Lim YT, Soltesz EG, De Grand AM, Lee J, Nakayama A, Parker JA, Mihaljevic T, Laurence RG, Dor DM, Cohn LH, Bawendi MG, Frangioni JV (2004) Near-infrared fluorescent type II quantum dots for sentinel lymph node mapping. Nat Biotechnol 22:93–97. doi:10.1038/nbt920   
Kolb HC, Finn MG, Sharpless KB (2001) Click chemistry: diverse chemical function from a few good reactions. Angew Chem Int Ed Engl 40:2004–2021   
Lagerholm BC, Averett L, Weinreb GE, Jacobson K, Thompson NL (2006) Analysis method for measuring submicroscopic distances with blinking quantum dots. Biophys J 91:3050–3060. doi:10.1529/biophysj.105.079178   
Larson DR, Zipfel WR, Williams RM, Clark SW, Bruchez MP, Wise FW, Webb WW (2003) Water-soluble quantum dots for multiphoton fluorescence imaging in vivo. Science 300(80):1434–1436. doi:10.1126/science.1083780   
Lessard GA, Goodwin PM, Werner JH (2007) Three-dimensional tracking of individual quantum dots. Appl Phys Lett 91:224106. doi:10.1063/1.2819074   
Li H, Duan ZW, Xie P, Liu YR, Wang WC, Dou SX, Wang PY (2012) Effects of paclitaxel on EGFR endocytic trafficking revealed using quantum dot tracking in single cells. PLoS One 7:e45465. doi:10.1371/journal.pone.0045465   
Lidke DS, Nagy P, Heintzmann R, Arndt-Jovin DJ, Post JN, Grecco HE, Jares-Erijman EA, Jovin TM (2004) Quantum dot ligands provide new insights into erbB/HER receptor-mediated signal transduction. Nat Biotechnol 22:198–203. doi:10.1038/nbt929   
Lidke DS, Lidke KA, Rieger B, Jovin TM, Arndt-Jovin DJ (2005a) Reaching out for signals: filopodia sense EGF and respond by directed retrograde transport of activated receptors. J Cell Biol 170:619–626. doi:10.1083/jcb.200503140   
Lidke KA, Rieger B, Jovin TM, Heintzmann R (2005b) Superresolution by localization of quantum dots using blinking statistics. Opt Express 13:7052. doi:10.1364/OPEX.13.007052

Liu J, Lau SK, Varma VA, Moffitt RA, Caldwell M, Liu T, Young AN, Petros JA, Osunkoya AO, Krogstad T, Leyland-Jones B, Wang MD, Nie S (2010a) Molecular mapping of tumor heterogeneity on clinical tissue specimens with multiplexed quantum dots. ACS Nano 4:2755–2765. doi:10.1021/nn100213v   
Liu J, Lau SK, Varma VA, Kairdolf BA, Nie S (2010b) Multiplexed detection and characterization of rare tumor cells in Hodgkin's lymphoma with multicolor quantum dots. Anal Chem 82:6237–6243. doi:10.1021/ac101065b   
Liu SL, Zhang ZL, Sun EZ, Peng J, Xie M, Tian ZQ, Lin Y, Pang DW (2011) Visualizing the endocytic and exocytic processes of wheat germ agglutinin by quantum dot-based single-particle tracking. Biomaterials 32:7616–7624. doi:10.1016/j.biomaterials.2011.06.046   
Liu SL, Zhang ZL, Tian ZQ, Zhao HS, Liu H, Sun EZ, Xiao GF, Zhang W, Wang HZ, Pang DW (2012) Effectively and efficiently dissecting the infection of influenza virus by quantum-dot-based single-particle tracking. ACS Nano 6:141–150. doi:10.1021/nn2031353   
Liu XL, Peng CW, Chen C, Yang XQ, Hu MB, Xia HS, Liu SP, Pang DW, Li Y (2011) Quantum dots-based double-color imaging of HER2 positive breast cancer invasion. Biochem Biophys Res Commun 409:577–582. doi:10.1016/j.bbrc.2011.05.052   
Lowe AR, Siegel JJ, Kalab P, Siu M, Weis K, Liphardt JT (2010) Selectivity mechanism of the nuclear pore complex characterized by single cargo tracking. Nature 467:600–603. doi:10.1038/nature09285   
Low-Nam ST, Lidke KA, Cutler PJ, Roovers RC, Bergen en Henegouwen PM van, Wilson BS, Lidke DS (2011) ErbB1 dimerization is promoted by domain co-confinement and stabilized by ligand binding. Nat Struct Mol Biol 18:1244–1249. doi:10.1038/nsmb.2135   
Lv X, Lei X, Ji M, Guo XF, Wang J, Dong WG (2013) Clinical significance of EBP50 overexpression assessed by quantum dot analysis in gastric cancer. Oncol Lett 5:1844–1848   
Matsuno A, Itoh J, Takekoshi S, Nagashima T, Osamura RY (2005) Three-dimensional imaging of the intracellular localization of growth hormone and prolactin and their mRNA using nanocrystal (Quantum dot) and confocal laser scanning microscopy techniques. J Histochem Cytochem 53:833–838. doi:10.1369/jhc.4A6577.2005   
Matsuno A, Mizutani A, Takekoshi S, Itoh J, Okinaga H, Nishina Y, Takano K, Nagashima T, Osamura RY, Teramoto A (2006) Analyses of the mechanism of intracellular transport and secretion of pituitary hormone, with an insight of the subcellular localization of pituitary hormone and its mRNA. Brain Tumor Pathol 23:1–5. doi:10.1007/s10014-005-0189-y   
Medintz IL, Uyeda HT, Goldman ER, Mattoussi H (2005) Quantum dot bioconjugates for imaging, labelling and sensing. Nat Mater 4:435–446. doi:10.1038/nmat1390   
Michalet X, Pinaud FF, Bentolila LA, Tsay JM, Doose S, Li JJ, Sundaresan G, Wu AM, Gambhir SS, Weiss S (2005) Quantum dots for live cells, in vivo imaging, and diagnostics. Science 307:538–544. doi:10.1126/science.1104274   
Montón H, Roldán M, Merkoçi A, Rossinyol E, Castell O, Nogués C (2012) The use of quantum dots for immunochemistry applications. Methods Mol Biol 906:185–192. doi:10.1007/978-1-61779-953-2 13   
Nan X, Sims PA, Chen P, Xie XS (2005) Observation of individual microtubule motor steps in living cells with endocytosed quantum dots. J Phys Chem B 109:24220–24224. doi:10.1021/jp056360w   
Nisman R, Dellaire G, Ren Y, Li R, Bazett-Jones DP (2004) Application of quantum dots as probes for correlative fluorescence, conventional, and energy-filtered transmission electron microscopy. J Histochem Cytochem 52:13–18. doi:10.1177/002215540405200102   
Peng CW, Liu XL, Chen C, Liu X, Yang XQ, Pang DW, Zhu XB, Li Y (2011) Patterns of cancer invasion revealed by QDs-based quantitative multiplexed imaging of tumor microenvironment. Biomaterials 32:2907–2917. doi:10.1016/j.biomaterials.2010.12.053

Petryayeva E, Algar WR, Medintz IL (2013) Quantum dots in bioanalysis: a review of applications across various platforms for fluorescence spectroscopy and imaging. Appl Spectrosc 67:215–252   
Pierobon P, Achouri S, Courty S, Dunn AR, Spudich JA, Dahan M, Cappello G (2009) Velocity, processivity, and individual steps of single myosin V molecules in live cells. Biophys J 96:4268–4275. doi:10.1016/j.bpj.2009.02.045   
Pinaud F, Clarke S, Sittner A, Dahan M (2010) Probing cellular events, one quantum dot at a time. Nat Methods 7:275–285. doi:10.1038/nmeth.1444   
Pons T, Mattoussi H (2009) Investigating biological processes at the single molecule level using luminescent quantum dots. Ann Biomed Eng 37:1934–1959. doi:10.1007/s10439-009-9715-0   
Prabhat P, Gan Z, Chao J, Ram S, Vaccaro C, Gibbons S, Ober RJ, Ward ES (2007) Elucidation of intracellular recycling pathways leading to exocytosis of the Fc receptor, FcRn, by using multifocal plane microscopy. Proc Natl Acad Sci U S A 104:5889–5894. doi:10.1073/pnas.0700337104   
Rajan SS, Liu HY, Vu TQ (2008) Ligand-bound quantum dot probes for studying the molecular scale dynamics of receptor endocytic trafficking in live cells. ACS Nano 2:1153–1166. doi:10.1021/nn700399e   
Ram S, Prabhat P, Chao J, Ward ES, Ober RJ (2008) High accuracy 3D quantum dot tracking with multifocal plane microscopy for the study of fast intracellular dynamics in live cells. Biophys J 95:6025–6043. doi:10.1529/biophysj.108.140392   
Ram S, Kim D, Ober RJ, Ward ES (2012) 3D single molecule tracking with multifocal plane microscopy reveals rapid intercellular transferrin transport at epithelial cell barriers. Biophys J 103:1594–1603. doi:10.1016/j.bpj.2012.08.054   
Regoes A, Hehl AB (2005) SNAP-tag mediated live cell labeling as an alternative to GFP in anaerobic organisms. Biotechniques 39:809–812   
Schieber C, Bestetti A, Lim JP, Ryan AD, Nguyen TL, Eldridge R, White AR, Gleeson PA, Donnelly PS, Williams SJ, Mulvaney P (2012) Conjugation of transferrin to azide-modified CdSe/ZnS core-shell quantum dots using cyclooctyne click chemistry. Angew Chem Int Ed Engl 51:10523–10527. doi:10.1002/anie.201202876   
Schütz GJ, Axmann M, Schindler H (2001) Imaging single molecules in three dimensions. Single Mol 2:69–74. doi:10.1002/1438-5171(200107)2:2<69::AID-SIMO69>3.0.CO;2-N   
Schwartz SL, Yan Q, Telmer CA, Lidke KA, Bruchez MP, Lidke DS (2014) Fluorogen activating proteins provide tunable labeling densities for tracking FcεRI independent of IgE. ACS Chem Biol. doi:10.1021/cb5005146   
Serizawa T, Terui T, Kagemoto T, Mizuno A, Shimozawa T, Kobirumaki F, Ishiwata S, Kurihara S, Fukuda N (2011) Real-time measurement of the length of a single sarcomere in rat ventricular myocytes: a novel analysis with quantum dots. Am J Physiol Cell Physiol 301:C1116–C1127. doi:10.1152/ajpcell.00161.2011   
Shaner NC, Lambert GG, Chammas A, Ni Y, Cranfill PJ, Baird MA, Sell BR, Allen JR, Day RN, Israelsson M, Davidson MW, Wang J (2013) A bright monomeric green fluorescent protein derived from Branchiostoma lanceolatum. Nat Methods 10:407–409. doi:10.1038/nmeth.2413   
Shu X, Lev-Ram V, Deerinck TJ, Qi Y, Ramko EB, Davidson MW, Jin Y, Ellisman MH, Tsien RY (2011) A genetically encoded tag for correlated light and electron microscopy of intact cells, tissues, and organisms. PLoS Biol 9:e1001041. doi:10.1371/journal.pbio.1001041   
Sjollema KA, Schnell U, Kuipers J, Kalicharan R, Giepmans BN (2012) Correlated light microscopy and electron microscopy. Methods Cell Biol 111:157–173   
Sosinsky GE, Giepmans BNG, Deerinck TJ, Gaietta GM, Ellisman MH (2007) Markers for correlated light and electron microscopy.

Methods Cell Biol 79:575–591. doi:10.1016/S0091-679X(06)79023-9  
Specht CG, Izeddin I, Rodriguez PC, El Beheiry M, Rostaing P, Darzacq X, Dahan M, Triller A (2013) Quantitative nanoscopy of inhibitory synapses: counting gephyrin molecules and receptor binding sites. Neuron 79:308–321. doi:10.1016/j.neuron.2013.05.013   
Steinkamp MP, Low-Nam ST, Yang S, Lidke KA, Lidke DS, Wilson BS (2014) erbB3 is an active tyrosine kinase capable of homo- and heterointeractions. Mol Cell Biol 34:965–977. doi:10.1128/MCB.01605-13   
Storch KN, Taatjes DJ, Bouffard NA, Locknar S, Bishop NM, Langevin HM (2007) Alpha smooth muscle actin distribution in cytoplasm and nuclear invaginations of connective tissue fibroblasts. Histochem Cell Biol 127:523–530. doi:10.1007/s00418-007-0275-9   
Sun B, Xie W, Yi G, Chen D, Zhou Y, Cheng J (2001) Microminiaturized immunoassays using quantum dots as fluorescent label by laser confocal scanning fluorescence detection. J Immunol Methods 249:85–89. doi:10.1016/S0022-1759(00)00331-8   
Sun JZ, Chen C, Jiang G, Tian WQ, Li Y, Sun SR (2014) Quantum dot-based immunofluorescent imaging of Ki67 and identification of prognostic value in HER2-positive (non-luminal) breast cancer. Int J Nanomedicine 9:1339–1346. doi:10.2147/IJN.S58881   
Szent-Gyorgyi C, Schmidt BF, Creeger Y, Fisher GW, Zakel KL, Adler S, Fitzpatrick JA, Woolford CA, Yan Q, Vasilev KV, Berget PB, Bruchez MP, Jarvik JW, Waggoner A (2008) Fluorogen-activating single-chain antibodies for imaging cell surface proteins. Nat Biotechnol 26:235–240. doi:10.1038/nbt1368   
Tabatabaei-Panah AS, Jeddi-Tehrani M, Ghods R, Akhondi MM, Mojtabavi N, Mahmoudi AR, Mirzadegan E, Shojaeian S, Zarnani AH (2013) Accurate sensitivity of quantum dots for detection of HER2 expression in breast cancer cells and tissues. J Fluoresc 23:293–302. doi:10.1007/s10895-012-1147-9   
Tada H, Higuchi H, Wanatabe TM, Ohuchi N (2007) In vivo real-time tracking of single quantum dots conjugated with monoclonal anti-HER2 antibody in tumors of mice. Cancer Res 67:1138–1144. doi:10.1158/0008-5472.CAN-06-1185   
Torreno-Pina JA, Castro BM, Manzo C, Buschow SI, Cambi A, Garcia-Parajo MF (2014) Enhanced receptor-clathrin interactions induced by N-glycan-mediated membrane micropatterning. Proc Natl Acad Sci U S A 111:11037–11042. doi:10.1073/pnas.1402041111   
True LD, Gao X (2007) Quantum dots for molecular pathology: their time has arrived. J Mol Diagn 9:7–11. doi:10.2353/jmoldx.2007.060186   
Uematsu M, Adachi E, Nakamura A, Tsuchiya K, Uchihara T (2012) Atomic identification of fluorescent Q-dots on tau-positive fibrils in 3D-reconstructed pick bodies. Am J Pathol 180:1394–1397. doi:10.1016/j.ajpath.2011.12.029   
Valentine CD, Verkman AS, Haggie PM (2012) Protein trafficking rates assessed by quantum dot quenching with bromocresol green. Traffic 13:25–29. doi:10.1111/j.1600-0854.2011.01287.x   
Vermehren-Schmaedick A, Krueger W, Jacob T, Ramunno-Johnson D, Balkowiec A, Lidke KA, Vu TQ (2014) Heterogeneous intracellular trafficking dynamics of brain-derived neurotrophic factor complexes in the neuronal soma revealed by single quantum dot tracking. PLoS One 9:e95113. doi:10.1371/journal.pone.0095113   
Wang Y, Fruhwirth G, Cai E, Ng T, Selvin PR (2013) 3D super-resolution imaging with blinking quantum dots. Nano Lett 13:5233–5241. doi:10.1021/nl4026665   
Watanabe TM, Sato T, Gonda K, Higuchi H (2007) Three-dimensional nanometry of vesicle transport in living cells using dual-focus imaging optics. Biochem Biophys Res Commun 359:1–7. doi:10.1016/j.bbrc.2007.04.168   
Wells NP, Lessard GA, Phipps ME, Goodwin PM, Lidke DS, Wilson BS, Werner JH (2009) Going beyond 2D: following membrane diffusion and topography in the IgE-Fc[epsilon]RI system using 3-dimensional tracking microscopy. Proc SPIE 7185:71850Z1–71850Z13. doi:10.1117/12.809412

Welsher K, Yang H (2014) Multi-resolution 3D visualization of the early stages of cellular uptake of peptide-coated nanoparticles. Nat Nanotechnol 9:198–203. doi:10.1038/nnano.2014.12   
Xing Y, Chaudry Q, Shen C, Kong KY, Zhau HE, Chung LW, Petros JA, O'Regan RM, Yezhelyev MV, Simons JW, Wang MD, Nie S (2007) Bioconjugated quantum dots for multiplexed and quantitative immunohistochemistry. Nat Protoc 2:1152–1165. doi:10.1038/nprot.2007.107   
Xu J, Teslaa T, Wu TH, Chiou PY, Teitell MA, Weiss S (2012) Nanoblade delivery and incorporation of quantum dot conjugates into tubulin networks in live cells. Nano Lett 12:5669–5672. doi:10.1021/nl302821g   
Xu J, Chang J, Yan Q, Dertinger T, Bruchez M, Weiss S (2013) Labeling cytosolic targets in live cells with blinking probes. J Phys Chem Lett 4:2138–2146. doi:10.1021/jz400682m   
You C, Wilmes S, Beutel O, Löchte S, Podoplelowa Y, Roder F, Richter C, Seine T, Schaible D, Uzé G, Clarke S, Pinaud F, Dahan M, Piehler J (2010) Self-controlled monofunctionalization of quantum dots for multiplexed protein tracking in live cells. Angew Chem Int Ed Engl 49:4108–4112. doi:10.1002/anie.200907032   
You C, Richter CP, Löchte S, Wilmes S, Piehler J (2014) Dynamic submicroscopic signaling zones revealed by pair correlation

tracking and localization microscopy. Anal Chem 86:8593–8602. doi:10.1021/ac501127r   
Yu J, Monaco SE, Onisko A, Bhargava R, Dabbs DJ, Cieply KM, Fine JL (2013) A validation study of quantum dot multispectral imaging to evaluate hormone receptor status in ductal carcinoma in situ of the breast. Hum Pathol 44:394–401. doi:10.1016/j.humpath.2012.06.002   
Zahavy E, Freeman E, Lustig S, Keysary A, Yitzhaki S (2005) Double labeling and simultaneous detection of B- and T cells using fluorescent nano-crystal (q-dots) in paraffin-embedded tissues. J Fluoresc 15:661–665. doi:10.1007/s10895-005-2972-x   
Zajac AL, Goldman YE, Holzbaur ELF, Ostap EM (2013) Local cytoskeletal and organelle interactions impact molecular-motor-driven early endosomal trafficking. Curr Biol 23:1173–1180. doi:10.1016/j.cub.2013.05.015   
Zrazhevskiy P, Gao X (2013) Quantum dot imaging platform for single-cell molecular profiling. Nat Commun 4:1619. doi:10.1038/ncomms2635   
Zrazhevskiy P, Sena M, Gao X (2010) Designing multifunctional quantum dots for bioimaging, detection, and drug delivery. Chem Soc Rev 39:4326–4354. doi:10.1039/b915139g   
Zrazhevskiy P, True LD, Gao X (2013) Multicolor multicycle molecular profiling with quantum dots for single-cell analysis. Nat Protoc 8:1852–1869. doi:10.1038/nprot.2013.112