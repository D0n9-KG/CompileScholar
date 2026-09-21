# Multimodal 3D cancer-mimicking optical phantom

Gennifer T. Smith, $^{1}$ Kristen L. Lurie, $^{1}$ Dimitar V. Zlatev, $^{2,3}$ Joseph C. Liao, $^{2,3}$ and Audrey K. Ellerbee Bowden $^{1,*}$

$^{1}$ Department of Electrical Engineering, Stanford University, Stanford, CA 94305, USA

$^{2}$ Department of Urology, Stanford University School of Medicine, Stanford, CA 94305, USA

3 Veterans Affairs Palo Alto Health Care System, Palo Alto, CA 94304, USA

\*audrey@ee.stanford.edu

Abstract: Three-dimensional (3D) organ-mimicking phantoms provide realistic imaging environments for testing various aspects of optical systems, including for evaluating new probe designs, characterizing the diagnostic potential of new technologies, and assessing novel image processing algorithms prior to validation in real tissue. We introduce and characterize the use of a new material, Dragon Skin (Smooth-On Inc.), and fabrication technique, air-brushing, for fabrication of a 3D phantom that mimics the appearance of a real organ under multiple imaging modalities.

We demonstrate the utility of the material and technique by fabricating the first 3D, hollow bladder phantom with realistic normal and multi-stage pathology features suitable for endoscopic detection using the gold standard imaging technique, white light cystoscopy (WLC), as well as the complementary imaging modalities of optical coherence tomography and blue light cystoscopy, which are aimed at improving the sensitivity and specificity of WLC to bladder cancer detection.

The flexibility of the material and technique used for phantom construction allowed for the representation of a wide range of diseased tissue states, ranging from inflammation (benign) to high-grade cancerous lesions. Such phantoms can serve as important tools for trainee education and evaluation of new endoscopic instrumentation.

$\odot$ 2016 Optical Society of America

OCIS codes: (110.4500) Optical coherence tomography; (160.4760) Optical properties; (170.3880) Medical and biological imaging; (170.7230) Urology.

# References and links

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 648

Biomedical Optics EXPRESS

255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 649

# 1. Introduction

More than 74,000 new cases of bladder cancer are diagnosed each year in the United States, with more than 15,000 deaths annually [1]. Due to its high recurrence rate, bladder cancer has the highest lifetime treatment cost per patient of all cancers [2,3] and requires frequent surveillance. White light cystoscopy (WLC) is the standard imaging tool for detection, resection and surveillance of bladder cancer; however, WLC has well-recognized shortcomings that can adversely impact treatment outcomes.

Specifically, WLC has limited diagnostic accuracy for high grade, non-papillary cancerous lesions such as carcinoma in situ (CIS) [4], and it has limited capability to distinguish such lesions from benign conditions (e.g., inflammation). Further, precise delineation of tumor borders or identification of small satellite lesions may be difficult with WLC and can result in incomplete resection and understaging of bladder cancer [5].

Two imaging modalities with potential to improve bladder cancer detection include optical coherence tomography (OCT), a label-free, near-infrared imaging technique capable of visualizing disruptions in the subsurface layers of the bladder that are early indicators of cancer, and blue light cystoscopy (BLC), which utilizes a photoactive prophyrin (e.g., hexaminolevulinate) that accumulates in rapidly dividing cells (such as cancer cells) and fluoresces under blue light to render cancerous lesions visually distinct from surrounding healthy tissue.

Recently, OCT and BLC have shown great promise as adjunct technologies to traditional WLC for bladder cancer detection [6-10]. A hurdle in the development and testing of new imaging technologies, however, is the lack of a large animal model for bladder cancer to yield diseased samples of

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 650

reasonable size. This obstacle can potentially be overcome with tissue-mimicking optical phantoms. For endoscopic applications such as OCT and BLC, a phantom that mimics the distenable, ovoid, and three-dimensional (3D) shape of the bladder or other organs of heterogeneous morphology and shape could permit better simulations of realistic imaging environments and allow for more accurate evaluation of current and emerging imaging systems despite the lack of access to diseased animal or human tissue. Furthermore, disease-mimicking phantoms can incorporate ground truth features to facilitate better characterization of new software, hardware, and imaging procedures prior to clinical testing and deployment.

Existing phantom fabrication techniques and materials have limited capability to simultaneously 1) recreate the arbitrary, 3D shape of organs such as the bladder, 2) replicate the structure of multi-layered tissue in 3D, and 3) incorporate multi-stage disease-mimicking regions. Most tissue-mimicking phantoms simulate flat regions of healthy tissue [11, 12]. Notable exceptions to this include work by Yang et al. to fabricate a fluorescently-labeled esophageal phantom [13], work by Bisaillon et al.

to fabricate tubular artery phantoms with disease-mimicking inclusions for intravascular OCT [14], work by Lurie et al. to fabricate a multi-layered, distendable, 3D optical phantom of the bladder with stage-T2 tumor inclusions [15] and a modification of the latter technique to mimic a wider range of bladder cancer pathologies (i.e., dysplasia, CIS, nonmuscle invasive and muscle invasive tumors) [16].

Limitations of these earlier works include, respectively, limitations in the range of possible optical properties needed to mimic various layers of epithelial tissue; an inability to fabricate phantoms for non-tubular, irregularly-shaped organs such as bladder; an inability to mimic early-stage bladder cancers; restriction to a planar substrate.

The difficulty in creating realistic bladder phantoms stems largely from the limitations of currently used materials (e.g., PDMS [17-19], PVA-C [20-22], latex [13]) and fabrication techniques (e.g., lathe-spinning [14], painting [13], and spin-coating [17,23]) for phantom creation. Importantly, with the emergence and clear clinical utility of multi-modality imaging technologies (e.g., WLC+OCT and WLC+BLC) it is also crucial for such phantoms to exhibit properties visible with multiple modalities. To the extent of our knowledge, no phantom simultaneously exhibits features visible to WLC, OCT, and BLC.

In this paper, we introduce Dragon Skin (Smooth-On, Inc.) and air-brushing as a new material and new strategy, respectively, for the fabrication of complex disease-mimicking optical phantoms, using the cancerous bladder as our model organ system. In light of the unique optical and highly viscous properties of Dragon Skin, we first characterize its elastic tunability and attenuation coefficient (a source of contrast in OCT). Next we characterize the achievable layer thicknesses and uniformity possible with air-brushing.

We then demonstrate use of these strategies to fabricate a 3D bladder phantom that simultaneously mimics healthy bladder tissue, inflamed/hemorrhagic lesions and pathologies ranging from pre-cancerous dysplasia to stage T2 muscle-invasive cancer. The resulting phantom mimics subsurface features of the bladder that are visible to OCT, features visible with traditional WLC (e.g., tissue color, surf
ace morphology and vasculature), and fluorescently-labeled lesions visible with BLC.

To the best of our knowledge, neither the material nor the technique presented here have been used for fabricating tissue-mimicking phantoms before, though they are well known in the creative and visual arts industries. Ours is also the first phantom whose fabrication process permits arbitrary 3D form factors while including diseased states that span the range of benign and pre-cancerous to muscle-invasive cancer, an important clinical decision point for bladder cancer treatment.

Notably, the inclusion of inflammatory regions in the current work is an important benign state that can be useful to help validate the specificity of newly developed imaging systems (e.g., OCT) designed as adjuncts to WLC [24].

255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 651

# 2. Experimental details

# 2.1. Material Property and fabrication technique considerations for phantom fabrication

Consider the bladder - a hollow organ comprising several thin layers of tissue, each having distinct optical properties (e.g., reflectivity, scattering). To fabricate a bladder phantom using a 3D-printed mold of the desired shape (see section 3.3 for details), a highly elastic (i.e., low Young's Modulus, $< \sim 300\mathrm{kPa}$ for bladder shape) and fast-curing material with a simple curing process is essential to extricate the phantom from the mold.

To mimic the tissue appearance of distinct bladder layers as they appear with optical technologies, the material must also be optically tunable. We evaluated various materials on the basis of several attributes: Table 1 summarizes the properties of popular materials for phantom fabrication and Dragon Skin. Latex, commonly used for some tissue-mimicking phantoms [13,25,26], has sufficient elasticity [27], but it is highly absorbing and its optical properties cannot be tuned to mimic the optical properties of tissue. PDMS, though optically tunable, does not have sufficient elasticity.

The elasticity can be altered by decreasing the mixing ratio of base to curing agent or by adding silicone gel. However, to decrease the Young's modulus below $\sim 800\mathrm{kPa}$ requires a large amount of curing agent or silicone gel, which leads to prolonged curing times and leaching of gel, respectively. Thus, the mechanical properties of such highly elastic PDMS are not stable over time [28]. Fibrin phantoms, while compatible with biological materials, have limited shelf-life and limited ability to accommodate 3D-structure due to its high viscosity [12].

PVA-C is highly elastic and optically tunable, but its complicated curing process makes it difficult to create uniform layers (note: the presence or absence of such layers is a crucial morphological clue to indicate the presence or severity of disease in epithelial-based diseases such as bladder cancer). In contrast, Dragon Skin has sufficient elasticity, a simple curing process, and is optically tunable, making it the optimal candidate for our work.

Table 1. Comparison of candidate materials for phantom fabrication. Bold text highlights the property values that are favorable for creating a 3D disease-mimicking optical phantom.

![](dt=2026-04-11/ht=00/a0b0bf7ea13c4f7f1ae6a84111b551cc2d7800642040ee468f8988c87a6eb06e.jpg)

<table><tr><td rowspan="2"></td><td colspan="4">Previous materials</td><td>New material</td></tr><tr><td>Latex [13,27,29]</td><td>PDMS [28,30]</td><td>Fibrin [12,31,32]</td><td>PVA-C [20,33-35]</td><td>Dragon Skin [36]</td></tr><tr><td>Cure time</td><td>minutes multiple days</td><td>30 min -</td><td>~10 min</td><td>~ 8 hrs</td><td>30 minutes - 16 hours</td></tr><tr><td>Curing process</td><td>room temp</td><td>heat (60-90 °C)</td><td>enzyme-induced proteolysis</td><td>multiple freeze/ thaw cycles</td><td>room temp</td></tr><tr><td>Optically tunable</td><td>no</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td></tr><tr><td>Shelf-life</td><td>years</td><td>years</td><td>1 month</td><td>6 months</td><td>years</td></tr><tr><td>Storage conditions</td><td>air room temp</td><td>air room temp</td><td>saline refrigerate</td><td>water room temp</td><td>air room temp</td></tr><tr><td>Elasticity</td><td>160 kPa - 700 kPa</td><td>800 kPa - 1.72 MPa</td><td>5 kPa - 10 kPa</td><td>1.2 kPa - 120 MPa</td><td>150 kPa - 593 kPa</td></tr></table>

We also evaluated various fabrication techniques on the basis of their ability to produce uniform, thin layers and their ability to coat molds of arbitrary 3D shape. Table 2 compares the previously reported fabrication techniques with air-brushing, a technique where materials (e.g., ink, paint) are aerosolized and directed onto a surface using compressed air. Of these, only air-brushing is capable of creating uniform thin layers on non-planar surfaces. While air-brushing has previously been demonstrated with PDMS for coating microstructures [37] and for creating

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 652

porous tablets for controlled drug release [38], it has yet to be utilized for phantom fabrication.

Table 2. Comparison of previously used phantom fabrication techniques and air-brushing. Only air-brushing is capable of producing conformal, uniformly thin layers on an arbitrary 3D shape.

![](dt=2026-04-11/ht=00/19259ab9da44e4fbde15f2fe37a078fd0701a293c4a247006f28d4307e0f1877.jpg)

<table><tr><td rowspan="2"></td><td colspan="3">Previous techniques</td><td>New technique</td></tr><tr><td>Lathe-based [14]</td><td>Spin-coating [17, 23]</td><td>Painting [13]</td><td>Air-brushing</td></tr><tr><td>Uniform thin layers</td><td>yes</td><td>yes</td><td>no - uncontrolled process</td><td>yes</td></tr><tr><td>Arbitrary 3D shape</td><td>no - tubular shapes only</td><td>limited - planar or near-rotationally symmetric surfaces</td><td>yes</td><td>yes</td></tr></table>

# 2.2. Details of the characterization process for dragon skin and air-brushing

Dragon Skin is a silicone-based liquid polymer that is often used for mold-castings and fabrication of full-face masks [39, 40]. Sold commercially as a two-part polymer (part A and part B), it is usually prepared by mixing parts A and B in a 1:1 ratio by weight and is available in a variety of cure times ranging from 30 minutes to 16 hours. Unless otherwise noted, we used the fastest-curing Dragon Skin for fabrication of the phantom to avoid sagging while applying it to non-planar surfaces.

To alter the viscosity of Dragon Skin, parts A and B were mixed together and NOVOCs (Smooth-On, Inc.), a solvent capable of thinning various silicone-based polymers, was subsequently added. To evaluate the elasticity of various dilutions of Dragon Skin with NOVOCs, the stress-strain curve of the samples was measured using an Instron 3360 Dual Column Tabletop Testing System. This information, coupled with the size and shape of the samples (rectangle, $75\mathrm{mm}\times 25\mathrm{mm}\times 0.5\mathrm{mm}$ ), was used to calculate the Young's modulus of each sample. Mixtures that were too viscous for the air brush simply failed to aerosolize, leaving only air to be ejected from the machine.

To alter the optical properties of Dragon Skin, $\mathrm{TiO_2}$ was first mixed with NOVOCs and then added to pre-mixed Dragon Skin (parts A and B). To ensure homogeneous distribution of the $\mathrm{TiO_2}$ , the NOVOCs/TiO $_2$ mixture was sonicated for ten minutes prior to mixing with Dragon Skin. The attenuation coefficient of the various Dragon Skin samples were extracted using a curve-fitting method app
lied to the OCT data [41].

In the single-scattering assumption limit appropriate for OCT measurements, attenuation of the amplitude of the interferometric signal, $I(z)$ , follows Beer's law: $I(z) \propto e^{-2\mu z}$ , where $\mu$ is the attenuation coefficient and $2z$ is the roundtrip path length of light in the sample. The OCT signal is complicated by the confocal function of the system [41-43]; to obviate the need to account for this, all samples were placed below the focal plane making Beer's law an appropriate approximation for the OCT signal.

The attenuation coefficient of various PDMS samples was extracted in a similar manner. All PDMS samples were mixed in a 1:10 (curing agent:base) ratio.

To induce fluorescent contrast in the phantom, invisible UV pigments were added to Dragon Skin; invisible UV pigments appear colorless under white light and fluoresce under UV light (close to the excitation wavelength of typical blue light cystoscopes), similar to the behavior of contrast agents used in BLC. The choice and amount of pigment for both the bulk tissue (blue UV invisible pigment from GloMania USA) and the diseased regions (red UV-reactive blacklight paint from D-Tek) were chosen by qualitatively matching the colors seen in BLC images of actual bladder tissue. For the phantom prototype, matching the exact color of the bulk tissue and lesion under blue light is not essential; rather, it is most important to have

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 653

visible contrast between the two. For this reason, the exact emission spectra of the pigments were not carefully controlled. The pigments were chosen over clinical fluorescent agents for their increased shelf-life and stability at room temperature. The UV-invisible pigments (blue and/or red) were added to the Dragon Skin after mixing with NOVOCs. The red pigment is supplied as a liquid and the blue pigment is supplied as a fine powder and, therefore, did not require sonication.

For all experiments we used a Badger air brush (model 200-BWH) and Badger air compressor (model AS180-12) according to manufacturer directions. In between uses the air brush was cleaned by running pure NOVOCs through the system to dissolve any excess Dragon Skin. As a final cleaning step, DI water was run through the system for several minutes.

For assessing obtainable layer thickness using air-brushing, a custom Matlab script was written to determine the physical thicknesses of samples measured by OCT $(2.5\mathrm{mm}\times 2.5\mathrm{mm}$ B-scan images). The index of refraction was measured to be 1.4 and was used to convert the optical thicknesses obtained from OCT to physical thicknesses. Thicknesses were measured at 25 points along the sample $(20\mathrm{mm}\times 20\mathrm{mm})$ to assess the uniformity of layers across the substrate. The process was repeated five times in order to evaluate the repeatability of the technique.

Due to changing air pressure over time within the air compressor we noted that airbrushing for extended periods of time (i.e., more than $\sim 5$ seconds) led to non-linear results in thickness. For this reason, measurements of achievable thicknesses were carried out by repeatedly spraying coats of 0.5 seconds up to 2.5 seconds and coats of 1 second thereafter until the total desired time was reached, instead of spraying constantly for the entire time period. For all thickness experiments, the substrates to be coated were held six inches from the air brush to ensure consistent results.

# 2.3. Imaging set-up

OCT images of bladder tissue presented in section 3.4 are courtesy of Dr. Joerg Schmidbauer from the Department of Urology, Medical University of Vienna, Vienna, Austria. OCT images of the phantom, images for measuring layer thickness, and images for measuring attenuation coefficients, were all obtained with a commercial spectral-domain OCT system (Telesto, Thorlabs) having a center wavelength of $1325\mathrm{nm}$ . A lateral scanning lens (LSM03, Thorlabs) with a Rayleigh range of $105.91\mu \mathrm{m}$ was installed and served as the objective.

WLC and BLC images of bladder tissue were obtained by J.C.L. as part of standard clinical care at the Veterans Affairs Palo Alto Health Care System (VAPAHCS) using a clinical system (D-light, Karl Storz Endoscopy, El Segundo, California). The images are de-identified and no protected health information was provided to the engineering team. WLC and BLC images of the phantom were also obtained at the VAPAHCS using the same clinical system.

# 3. Results and discussion

# 3.1. Dragon skin: characterization of elasticity and optical tunability

Commercial Dragon Skin is too viscous for air brushing in its native state. It was thus necessary to thin the Dragon Skin by adding a silicone solvent, NOVOCs (Smooth-On, Inc.). Notably, adding solvent reduces both the viscosity of the Dragon Skin and its elasticity. To determine the appropriate balance between low viscosity to enable air brushing and high elasticity to facilitate extrication, we varied the amount of silicone solvent added from $0\%$ to $160\%$ by weight and evaluated the subsequent elasticity and ease of air-brushing.

Figure 1(a) demonstrates the relationship between the amount of NOVOCs added to Dragon Skin and the resulting elasticity. We evaluated the elasticity of pure Dragon Skin and also Dragon Skin with $0.2\%$ w/w of $\mathrm{TiO_2}$ particles added to facilitate tuning the scattering coefficient of Dragon Skin. The addition of $80\%$ or $100\%$ NOVOCs or greater by weight lowered the

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 654

viscosity sufficiently to enable air-brushing of Dragon Skin without or with $\mathrm{TiO_2}$ , respectively. Recipes amenable to air-brushing are denoted by a solid line. To maintain the highest elasticity possible for the phantom, we chose the highest viscosity that was amenable to air-brushing with $\mathrm{TiO_2}$ (i.e., $100\%$ by weight) for all parts of the phantom.

![](dt=2026-04-11/ht=00/f8f75c6117c0df27284483e3f8c403600692e8bcb5407f19ed144580005bc530.jpg)

![](dt=2026-04-11/ht=00/f7fdfcfb8258040595fc00c82902eeab58f82198dba7388705589c9e40a05b04.jpg)

The optical scattering of biological tissue is an important feature that serves as the basis for signal in OCT imaging. De Bruin et al. provided guidelines for altering the attenuation coefficient of PDMS using $\mathrm{TiO_2}$ for OCT imaging [30], but these guidelines are inappropriate for Dragon Skin because, although similar to PDMS in many respects, it is not optically clear.

To determine the volumetric dispersions appropriate to achieve attenuation coefficients matched to bladder tissue, we characterized the attenuation coefficient of Dragon Skin as a function of $\mathrm{TiO_2}$ (between $0.1\%$ and $1\%$ w/v). Figure 1(b) shows the relationship between the concentration of $\mathrm{TiO_2}$ and the attenuation coefficient for both PDMS and Dragon Skin.

The consistency of our results for PDMS with previously obtained results [30] validates our measurement and, as expected, we found that the attenuation coefficients of Dragon Skin for matched concentrations of $\mathrm{TiO_2}$ dispersions show a linear offset from those in PDMS due to the inherent scattering (and possibly absorption) of Dragon Skin and the additive nature of attenuation coefficients. The range of attenuation coefficient values presented in Fig.

1(b) covers a biologically relevant range $(0.45 - 8\mathrm{mm}^{-1})$ ; the typical attenuation coefficients pres
enting in the bladder (see section 3.4 for details) are indicated by orange arrows.

To mimic the appearance of tissue as seen with traditional WLC and BLC, we added red dye (Sharpie) and invisible UV pigments (D-tek and GloMania USA), respectively; the former was chosen for its low absorption in the NIR range, as the attenuation coefficient of tissue at typical OCT wavelengths is dominated by scattering. The pigments were chosen to approximately mimic the color of healthy and diseased tissue, but as the exact coloration is not critical for the current application, their concentration and emission spectra were not carefully controlled, as described in section 2.2.

A more detailed study of the absorption properties of Dragon Skin, bladder tissue, and various red dyes could lead to the ability to mimic both the scattering and absorption properties of tissues while preserving the appearance of the phantom under WLC. However, because tissue properties are dominated by scattering at typical OCT wavelengths,

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 655

we chose to limit our study to scattering and visual appearance.

# 3.2. Air-brushing: characterization of uniformity and repeatability

The rugged surface texture of the bladder (due to mucosal folds) is an important feature for white light imaging; for this reason, it is critical to assess the uniformity of air-brushing on textured surfaces. For comparison, we also considered the uniformity of layers achieved on planar substrates.

Figure 2(a) shows the relationship between the total amount of time air-brushed (i.e., sprayed) (x-axis) and the average thickness of air-brushed Dragon Skin (y-axis) deposited on both a smooth surface and a textured surface. As expected, both follow a linear trend. The plotted linear fit is nearly identical for both surfaces, though only one fit is shown (smooth surface). The error bars represent the standard deviation for five samples. This characterization curve allows us to consistently create layers of desired thickness by associating the desired thickness with a necessary spray time.

Intrasample variation (i.e., within a single sample) was also measured and is highlighted in Fig. 2(b), which shows the total time sprayed vs. the variation in layer thickness (i.e., standard deviation of the thickness across a single sample). The plotted value is the average intrasample variation over all five samples. The variation is higher for thinner samples, as evidenced by the representative OCT images. In general, the variation on a smooth surface and textured surface are almost indistinguishable (data not shown).

![](dt=2026-04-11/ht=00/14d484eba9f68014ec26279626716c3e37f3b63d4ea5ab6e5253646048b7e5c5.jpg)

![](dt=2026-04-11/ht=00/3f763109bc11b60177c3ad25b5f5d3ba6b2d4e69e522a5cd40916a59ad9bbca8.jpg)

# 3.3. Phantom fabrication

The core shape for the phantom was procured from a 3D model of the bladder, as described previously [15].

A general overview of the fabrication process is shown in Fig. 3. Step 1: We excised a hole from the top of the mold and inserted a metal post to permit access to all sides of the phantom during air-brushing. To induce a surface roughness similar to bladder rugae, the mold was coated with crumpled aluminum foil prior to air-brushing [15]. Step 2: We sprayed the mold for 1.5 seconds with Dragon Skin containing $0.05\%$ w/w of $\mathrm{TiO_2}$ and $5\%$ w/w of blue, invisible UV pigment to produce a urothelium layer with a thickness of $50~\mu \mathrm{m}$ , an attenuation coefficient of

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 656

$\sim 0.5 \mathrm{~mm}^{-1}$ , and a bluish/purple appearance under BLC. To better simulate the color of tissue under white light illumination, we mixed red Sharpie dye into the sonicated NOVOCs/TiO $_2$ . Step 3: Vessels were created by coloring and arbitrarily cutting elastic cord (Stretch Magic®) to various lengths between 1 and $10 \mathrm{~mm}$ . The elastic cord measures $0.5 \mathrm{~mm}$ in diameter. The elastic cord is a good approximation to real vascular networks and gives realistic features for use in image processing algorithms.

Vessels were added to the phantom simply by pressing pieces of cord against the phantom. To ensure good adhesion between the Dragon Skin and vessels, the vessels were applied to the phantom before the Dragon Skin was fully cured. When airbrushing in thin layers the Dragon Skin cures very quickly, necessitating the use of a slower curing Dragon Skin (30-minute cure time) for the layer beneath the vessels. Step 4: We sprayed the mold for seven seconds using Dragon Skin with $0.2\%$ w/w of TiO $_2$ and $5\%$ w/w of invisible UV pigment to create the lamina propria layer.

Step 5: Sandwich molding [15] was used to create the muscularis propria layer. An outer mold, made of two halves, was 3D-printed and designed to have a gap thickness of $4 \mathrm{~mm}$ . The two halves were placed around the inner mold and sealed together using a hot glue gun. Dragon Skin with $0.1\%$ w/w of TiO $_2$ and $5\%$ w/w of invisible pigment was poured into these outer molds via a hole cut in the top half. Step 6: Following curing of the muscularis propria layer, the outer molds were unsealed and removed.

Step 7: We removed the post from the mold, and the phantom was removed by stretching it over the hole where the post used to be. Because the Dragon Skin is highly elastic, it can be stretched over the inner mold without tearing. Step 8: A silicone tube was inserted and bonded to the hole left by the post to mimic the urethra and provide a realistic insertion channel for endoscopy. The final image in Fig. 3 shows an external view of the full 3D bladder phantom in which the realistic tissue coloration and arbitrary 3D shape are apparent.

Note that from the outside, the phantom appears to have a visible seam. This buildup of material is a result of a small gap between the outer 3D-printed molds and is not visible from the inside of the phantom.

The circles labelled "D" in Fig. 3 represent regions in which the layer structure was altered to mimic pathologies. The diseased regions were created using an excise-and-fill method, similar to a previously described process [16]; small sections of the phantom were excised during various steps of the fabrication process and then filled with Dragon Skin, the properties of which depended on the desired end diseased state. A brief overview of the relevant diseased states follows stemming from insights published in literature [7, 9, 10, 24].

Pre-cancerous dysplastic lesions manifest as a thickening of the urolthelium, appear flat under WLC, and do not exhibit fluorescent contrast under BLC. It should be noted that while most pre-cancerous, dysplastic regions do not exhibit fluorescent contrast, low fluorescent contrast has been observed in some cases of dysplasia resulting in false-positive outcomes [44]. Carcinoma in situ (CIS), an early-stage, high-grade cancer (stage TIS), manifests as a combined urothelium and lamina propria layer and is only distinguishable with subsurface imaging techniques like OCT.

CIS appears flat under WLC but is often accompanied by an increased redness (erythema) of the tissue. The increased redness is similar to what is seen with inflammation and hemorrhagic lesions, making it difficult to distinguish CIS from inflammation using WLC alone. However, unlike CIS, inflammation does not disrupt the layered structure of the bladder tissue, making the two distinguishable using OCT. Under BLC
there is clear fluorescent contrast between CIS and healthy tissue; inflammation similarly exhibits fluorescent contrast under BLC.

Similar to CIS, T1-staged tumors exhibit a combined urothelium and lamina propria layer; however, they protrude more deeply into the bladder wall and lack a clear demarcation with the subsequent muscularis propria layer. Note that the distinction between healthy and T1-staged tumors is the most clinically-relevant distinction. Finally, T2-staged (and higher) cancers are classified as muscle-invasive carcinomas and lack well-defined layers or horizontal structure.

Table 3 provides details regarding the appearance of several pathologies under WLC, BLC,

255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 657

![](dt=2026-04-11/ht=00/9d9149d9a852225493bf73b34d5b35a4bce85c5777fd9bcc5fae295074bed00b.jpg)

![](dt=2026-04-11/ht=00/bbf326087f10ffd65fa256f0990d087fbbb74d1ccff460b1c9586d11f91f9b95.jpg)

![](dt=2026-04-11/ht=00/1ce1a68aec7c7a56668d27d7b1766a2d511590abe1cea826904a106d4a06db2c.jpg)

![](dt=2026-04-11/ht=00/38ca35a40847e57ce88dc8f71c885cb6864cdaa8d4ff01d8944a11355692f35b.jpg)

![](dt=2026-04-11/ht=00/b55620da3fa906053e9cf956d9b86f5a9abcea19db302978a2208b62585cde0c.jpg)

![](dt=2026-04-11/ht=00/fcbf91176489dcf621425caffeb68b19d28c74524ad5d9661a5ec65f141ce5d6.jpg)

![](dt=2026-04-11/ht=00/bcff590419afeca243a9269e158b51520f141dbd053b64f9e7a5afcc2f8d8753.jpg)

![](dt=2026-04-11/ht=00/5f00080db4511a5e6eaaf8fc656dc077ef955a22e9e896e2da637023a55e1cd1.jpg)

![](dt=2026-04-11/ht=00/b7cda1aa939decc8788ec379d4dfbac27d4cd3ec56d338a23d6ca160d92c13b6.jpg)

and OCT, in addition to the final layered appearance of the phantom. Briefly, 1) inflamed regions were created by excising small sections of the urothelium and filling with Dragon Skin that mimics the urothelium but has an increased amount of red dye.

This preserved the layered structure of the bladder but manifested as increased redness when viewed with WLC; 2) dysplasia, characterized by a thickened urothelium, was created by air-brushing additional urothelium layers; 3) CIS, T1, and T2 were all created by excising small portions of the in-process phantom upon the completion of the lamina propria layer.

CIS, which is characterized as a combined urothelium and lamina propria layer with increased redness in tissue (under WLC) and adhesion to fluorescent labels, was fabricated by filling the excised sections with Dragon Skin that mimics the lamina propria but has an increased amount of red-dye and red transparent fluorescent paint. T1 tumors were fabricated by filling the excised portions with several different layers of Dragon Skin having attenuation coefficients between that of the lamina propria and the muscularis propria and containing fluorescent paint.

This created a gradual change in attenuation between these layers and the posterior muscle layer, in contrast to the clear demarcation between the combined urothelium - lamina propria layer and the muscle layer typical of CIS regions. T2 tumors were fabricated by filling the excised regions with Dragon Skin that mimics the muscularis propria.

# 3.4. Visual appearance of the phantom

The healthy bladder wall is a layered structure: the three layers from the lumen outward, along with their average thicknesses and attenuation coefficients, include the optically dark (attenuat-

255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 658

Table 3. Appearance descriptors for several diseased states in bladder tissue as seen under WLC, BLC, and OCT. Last column shows the fabricated layer structure to yield the corresponding pathology. In the healthy schematic, U is $50~\mu \mathrm{m}$ , LP is $250~\mu \mathrm{m}$ , and MP is 1.6 mm. Note that the images are not to scale. Solid red lines indicate the inclusion of red dye; dashed pink lines indicate the inclusion of transparent fluorescent dye. The slightly longer length of the urothelium layer of dysplastic tissue is intentional. U: urothelium, LP: lamina propria, MP: muscularis propria.

![](dt=2026-04-11/ht=00/d8ab1dbf37faa14053a611120341074bb1adb553244b424737167f9cf37ce346.jpg)

<table><tr><td rowspan="2"></td><td colspan="3">Tissue appearance</td><td colspan="3">Layered Appearance</td></tr><tr><td>WLC</td><td>BLC</td><td>OCT</td><td>U</td><td>LP</td><td>MP</td></tr><tr><td>Inflammation</td><td>increased redness (often)</td><td>identical to healthy</td><td>identical to healthy</td><td>U</td><td>LP</td><td>MP</td></tr><tr><td>Dysplasia</td><td>identical to healthy</td><td>identical to healthy</td><td>thickened U</td><td>U</td><td>LP</td><td>MP</td></tr><tr><td>CIS</td><td>increased redness (often)</td><td>fluorescent</td><td>fused U + LP</td><td colspan="2">LP</td><td>MP</td></tr><tr><td>T1</td><td>papillary structure (often)</td><td>fluorescent</td><td>fused U + LP without clear demarcation with MP</td><td colspan="2">LP</td><td>MP</td></tr><tr><td>T2</td><td>papillary structure</td><td>fluorescent</td><td>no layered structure</td><td colspan="3">MP</td></tr></table>

ing), thin urothelium $(50\mu \mathrm{m}, 0.49 \pm 0.25\mathrm{mm}^{-1})$ ; the brighter lamina propria $(250\mu \mathrm{m}, 2.0 \pm 0.7\mathrm{mm}^{-1})$ ; and the dark, thick muscularis propria $(1.6\mathrm{mm}, 1.38 \pm 0.7\mathrm{mm}^{-1})$ [45,46]. These layers are visible with subsurface imaging modalities like OCT. As described previously, cancerous lesions manifest in the bladder as varying degrees of disruption of the layered structure (i.e., thickened layers or loss of layers); the extent of disruption in the layered appearance closely parallels the stage of cancer.

Figure 4 shows comparison OCT images of pathological bladder tissue and our phantom "tissue." Qualitatively, the three distinct layers of healthy bladder tissue are accurately mimicked by the phantom, as are the disruptions of layers associated with various stages of cancer for the diseased regions.

Differences in the appearance of the attenuation coefficients may be easily explained: 1) the actual tissue images and phantom images were obtained with different systems; 2) the images likely differ in dynamic range; 3) the attenuation coefficients of the specific human bladder tissue imaged here likely differ from the average values reported in literature. In contrast, the attenuation coefficients of the phantom exactly match the average value previously reported for bladder tissue.

Collectively, these results showcase the first 3D phantom to include healthy tissue and such a wide range of pathologies.

While erythematous changes in tissue color (i.e., increased redness) associated with bladder cancer lesions are visible under WLC, benign conditions such as inflammation may appear similar to each other, which can result in false positive targets for biopsy. While it may be possible to better distinguish malignant lesions from benign using BLC, the fluorescence con
trast between the two often appears similar.

In such cases, subsurface imaging techniques such as OCT could provide better differentiation between the two conditions, as inflammation does not disturb the layered structure of the bladder. Figure 5 shows WLC and BLC images of healthy tissue, inflammation and cancer. The erythematous changes seen in both inflammation and cancer are difficult to distinguish under WLC alone, a challenge that is adequately mimicked in the 3D phantom (erythematous changes are indicated with circles). Moreover, the erythema

![](dt=2026-04-11/ht=00/c30ee6d646769aac90dbc181158b1eb72dad1fbe97cad22715273ba3d90bdde2.jpg)

tous changes themselves are difficult to distinguish from healthy tissue in the bladder to the untrained eye. While the erythematous changes in the phantom are more distinct due to the amount of dye added, we can also mimic cancerous regions that are difficult to observe with WLC.

![](dt=2026-04-11/ht=00/d88801a432d4dc25309212a17faab87cea280ea7b609edf27e03aaa4b33ef479.jpg)

The increased pink fluorescence emitted by cancerous lesions under BLC occurs in both the actual bladder tissue and the 3D phantom. Benign conditions (e.g., inflammation) can also present with fluorescence contrast, even though the level of contrast is typically lower than that of cancerous lesions, making it difficult to distinguish the two. For the phantom, we mimicked this low contrast by adding a low concentration of fluorescent pigment to inflamed regions (as shown in Fig. 5) and by keeping the layered structure visible with OCT intact (data not shown). Note that the exact appearance of tissue under both WLC and BLC (both the pink fluorescence

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 660

of cancer/inflammation and the blue autofluorescence of healthy tissue) varies widely between patients, as evidenced by the differences in background tissue color seen in the bladder tissue images, each of which is captured from a different patient. For this reason, it is impossible to create a phantom that perfectly matches the color of tissue for all patients.

The cancerous lesions shown in Fig. 5 are flat lesions, which project into the bladder wall but do not extend into the lumen, making them difficult to identify with WLC alone. However, higher stage tumors (i.e., T1 and T2) are often accompanied by a papillary structure that protrudes from the surface and is clearly visible with WLC. These tumors have a bubbly texture are an important WLC feature. Figure 6 shows an image of a papillary tumor in bladder tissue and a simulated papillary tumor within the phantom.

The texture of the papillary structure was replicated by placing fast-curing Dragon Skin inside a vacuum chamber, causing the bubbles to rise to the surface. Due to the fast cure time of the Dragon Skin, the bubbles were essentially solidified before popping. The papillary structure was then bonded to the inside of the phantom (after removal from the inside mold but prior to addition of a urethra) using additional Dragon Skin as glue. The resulting papillary tumors are composed of a single layer, which is not representative of the actual composition of papillary tumors.

However, since these tumors are clearly visible with white light alone we did not deem it necessary to mimic their subsurface properties. The dark features that accompany the image are air bubbles that appeared while visualizing the phantom during cystoscopy.

![](dt=2026-04-11/ht=00/b7d937c74518e6ff59f0d6e951d4c0ef99a272ebcbddf2c5d1c439c9720dc8a8.jpg)

![](dt=2026-04-11/ht=00/107c645973ada75530a83075f105fab3b914359089a656f9f9bf771c1a394197.jpg)

Lastly, As described in section 3.3, we also mimicked the appearance of vasculature, a feature commonly visible under WLC and a landmark often used for biopsy retargeting. The mimicked vasculature is apparent in all images in Fig. 5. Note that the vasculature in the phantom appear slightly fluorescent due to the red Sharpie dye; judicious choice of a different red pigment can avoid this problem in the future. Visualization 1 shows a WLC/BLC cystoscopy of the phantom performed under distention with saline.

# 4. Conclusion

In conclusion, we demonstrate the suitability of Dragon Skin and air-brushing as an enabling material and technique, respectively, to create 3D optical phantoms that mimic the realistic appearance of healthy and diseased tissue under various imaging modalities.

Dragon Skin offers many advantages over previously used phantom materials. Similar to PDMS, it is optically tunable but has increased elasticity. The high elasticity of Dragon Skin is a critical feature for 3D phantom fabrication, as it facilitates clean removal of the phantom from textured, 3D-printed molds having many points of stiction. Even though the elasticity of Dragon Skin is still lower than that of actual tissue (3.5kPa - 182kPa for the various layers of the bladder [47]), it represents a vast improvement over other optically tunable 3D phantoms

#255068

Received 3 Dec 2015; revised 12 Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 661

and provides sufficient elasticity to mimic the deformation typically seen in real tissue. Notably, mimicking this deformation can play an important role in evaluating novel image processing algorithms [48, 49]. Another advantage of Dragon Skin is its stability at room temperature and its long shelf-life. Finally, its fast cure time facilitates creation of layers on a non-planar surface that are uniform and minimally affected by gravity during curing.

The use of air-brushing was also a strategic decision to enable creation of the uniform, thin layers of the bladder important for the OCT imaging modality. While the uniformity of airbrushing lags that of other methods such as spin-coating, it has more flexibility to be used in a 3D context. That said, it is important to remember that the viscosity of the material being airbrushed must be sufficiently low to aerosolize, and particle suspensions should be adequately dispersed to avoid clogging the device.

Envisioned uses of the prototype multi-modal, disease-mimicking bladder phantom we developed include evaluation of new endoscopic technologies and the assessment of the diagnostic potential of new imaging systems and novel image processing algorithms. Implementation of bladder features visible under WLC and BLC also makes the phantom a useful tool for training new physicians: for example, to assess their technical ability to achieve complete surveying of the bladder wall or to identify various states of disease.

Further, the material and techniques introduced in this paper can be generalized for developing other large-scale organ phantoms such as for the stomach. In particular, the method for including diseased regions within the phantom is relevant to other epithelial cancer (e.g., colon and stomach) that also manifest as disrupted layered structures. The widespread availability of such 3D disease-mimicking organ phantoms can lead to improved system validation, cross-system comparisons and algorithm testing to enhance the diagnosis of cancer.

# Acknowledgments

GTS is supported by an NSFGRFP fellowship. KLL is supported by NDSEG and NSFGRFP fellowships. JCL is supported in part by NIH R01 CA160986. The authors would like to thank Joerg Schmidbauer for use of the in vivo OCT bladder images.

255068

Received 3 Dec 2015; revised 12
Jan 2016; accepted 13 Jan 2016; published 25 Jan 2016

(C) 2016 OSA

1 Feb 2016 | Vol. 7, No. 2 | DOI:10.1364/BOE.7.000648 | BIOMEDICAL OPTICS EXPRESS 662