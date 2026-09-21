# Rapid and Digital Detection of Inflammatory Biomarkers Enabled by a Novel Portable Nanoplasmonic Imager

Alexander Belushkin, Filiz Yesilkoy, Juan Jose González-López, Juan Carlos Ruiz-Rodríguez, Ricard Ferrer, Anna Fàbrega,* and Hatice Altug*

New point-of-care diagnostic devices are urgently needed for rapid and accurate diagnosis, particularly in the management of life-threatening infections and sepsis, where immediate treatment is key. Sepsis is a critical condition caused by systemic response to infection, with chances of survival drastically decreasing every hour. A novel portable biosensor based on nanoparticle-enhanced digital plasmonic imaging is reported for rapid and sensitive detection of two sepsis-related inflammatory biomarkers, procalcitonin (PCT) and C-reactive protein (CRP) directly from blood serum.

The device achieves outstanding limit of detection of $21.3\mathrm{pg}\mathrm{mL}^{-1}$ for PCT and $36\mathrm{pg}\mathrm{mL}^{-1}$ for CRP, and dynamic range of at least three orders of magnitude. The portable device is deployed at Vall d'Hebron University Hospital in Spain and tested with a wide range of patient samples with sepsis, noninfectious systemic inflammatory response syndrome (SIRS), and healthy subjects.

The results are validated against ultimate clinical diagnosis and currently used immunoassays, and show that the device provides accurate and robust performance equivalent to gold-standard laboratory tests. Importantly, the plasmonic imager can enable identification of PCT levels typical of sepsis and SIRS patients in less than 15 min. The compact and low-cost device is a promising solution for assisting rapid and accurate on-site sepsis diagnosis.

# 1. Introduction

Sepsis is a life-threatening condition where the body's own response to an infection progressively leads to organ system

A. Belushkin, Dr. F. Yesilkoy, Prof. H. Altug

Institute of BioEngineering

École Polytechnique Fédérale de Lausanne

CH-1015 Lausanne, Switzerland

E-mail: hatice.altug@epfl.ch

Dr. J. J. González-López, Dr. A. Fàbrega

Department of Clinical Microbiology

Vall d'Hebron University Hospital

Vall d'Hebron Institut de Recerca

Universitat Autonoma de Barcelona

08035 Barcelona, Spain

E-mail: anna.fabrega@vhir.org

Dr. J. C. Ruiz-Rodriguez, Dr. R. Ferrer

Intensive Care Department

Vall d'Hebron University Hospital, Shock

Organ Dysfunction and Resuscitation Research Group

Vall d'Hebron Institut de Recerca

08035 Barcelona, Spain

![](dt=2026-04-09/ht=18/cee4f9c7610c3e8a9b000f303847ab93ba16639106d58c8a1a32a693d840c7fb.jpg)

The ORCID identification number(s) for the author(s) of this article

can be found under https://doi.org/10.1002/smll.201906108.

DOI: 10.1002/smll.201906108

dysfunction, failure and finally death.[1] Although the ongoing advancements in modern medicine have increased the survival rates of organ recipients and patients with serious illnesses, such as cancer and HIV, the occurrence of sepsis has been increasing globally, imposing a significant challenge.[2] Moreover, the emergence of drug-resistant pathogens is contributing significantly to the global sepsis threat.[3] Current reports indicate that sepsis causes mortality reaching up to $50\%$ and affects more than 30 million people every year worldwide.

[2] In the United States alone, $\$24$ billion is spent every year for the management of sepsis, making it the most expensive condition to treat.[4] Sepsis progresses extremely rapidly and is considered a medical emergency. In fact, sepsis progresses so fast that every hour of delay in diagnosis and initiating appropriate treatment increases the mortality by $7.6\%$ .[5] Therefore, development of point-of-care devices enabling early and accurate diagnosis of sepsis at different clinical

settings is an urgent need as they can allow timely intervention and lower the mortality rate.[6-8]

However, sepsis diagnosis is not easy, especially in the early stages, or when sepsis presents with an atypical clinical picture. Early stages of sepsis usually present stealth symptoms with significant patient-to-patient heterogeneity, leading to potential delays from the onset of sepsis until the most appropriate treatment starts.[9,10] The dysregulated host response to infection, which is typically present in sepsis patients, is primarily supported by an inflammatory process, which is termed SIRS, for systemic inflammatory response syndrome.

However, it is challenging to discriminate whether the biological insult leading to SIRS is an infectious process or not, an essential criteria for the antimicrobial treatment to be worthy and effective.[1] Traditionally, for the microbiological diagnosis of sepsis, blood-culture based pathogen investigation is performed, which typically takes 4-5 days.

[9] The current recommendation is to administer broad-spectrum antibiotics as soon as possible, ideally in the first hour after sepsis recognition, before a definitive microbiological diagnosis is made based on the pathogen identification tests.[11] This approach leads to overuse of antibiotics, which worsens the global health challenge of antibiotic resistant bacteria development, because neither all SIRS patients have an underlying infectious process, nor all infections are caused by

FULL PAPER

NANO·MICRO SMA

Check for updates

www.small-journal.com

Small 2019, 1906108

1906108 (1 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

bacterial pathogens. In contrast, a new clinical trend in sepsis management relies on blood testing to monitor patient's systemic response for rapid patient stratification. These tests, which quantify circulating biomarkers, such as the two most universal sepsis-related biomarkers procalcitonin (PCT) and C-reactive protein (CRP), help diagnose sepsis timely and reduce overprescription of antibiotics.

Although these inflammatory biomarkers alone cannot be the sole basis for accurate sepsis diagnosis, their blood levels provide valuable quantitative information to triage patients based on their anomalous response allowing for personalized treatment.[12-18] The current gold standard clinical tools to test blood circulating proteins are largely based on complex and lengthy immunoassays, such as enzyme-linked immunosorbent, immunoturbidimetric (ITA), and chemiluminescent (CLIA) immunoassays, which require fully equipped clinical laboratories.

However, having rapid access to the patient's systemic response, using portable, low-cost, and easy-to-use devices that can rapidly detect biomarkers in a simple manner from small sample volumes can drastically improve sepsis care at different clinical settings such as emergency care units, ambulances, and outpatient health centers.

Currently, active ongoing efforts are being undertaken both in the biomedical industry and basic research to find new diagnostic solutions and develop point-of-care devices to rapidly diagnose sepsis on-site in a cost-effective way. For instance, a compact device proposed by Samsung (LABGEO IB10) operates by detecting PCT with a centrifugal microfluidic disk. Another recent technology that is being investigated by Abionic for sepsis detection combines nanofluidic channels with centrifugal disk. Although nanosized channels can improve the sensitivity they are prone to clogging.

Moreover, these technologies employ mechanically moving parts that can compromise device robustness over time. So far, no portable system has received US Food and Drug Administration approval for measuring biomarkers that effectively predict sepsis at the point of care. Research efforts have been made to develop devices for sepsis biomarkers detection based on optical schemes, including plasmonics and fluorescence. Surface plasmon resonance (SPR) and surface-enhanced Raman scattering sensors were reported, however they could not achieve clinically relevant sensitivity.

[19,20] Flu
orescent schemes have been implemented to detect sepsis markers, however they either rely on expensive and bulky optical setups, such as total internal reflection microscopy,[21] or do not meet clinical sensitivity.[22,23] Promising recent approaches use microfluidic extraction of cells from small amount of patient blood to quantify expression of CD64 biomarker, however they employ sophisticated design and control of microfluidic chip.[24,25] Alternatively, a sensitive system based on magnetic beads and fluorescent enzymatic amplification has been proposed.

The system provides a low limit of detection for measuring a sepsis-related biomarker IL-3 but at the expense of multiple assay steps and 1 h long detection time.[26]

Here, we report a portable digital nanoparticle-enhanced plasmonic imager for rapid detection of inflammatory biomarkers (DENIS). The unique nanoplasmonic imaging mechanism is based on gold nanoparticle (Au-NP) binding to plasmonic gold nanohole array (Au-NHA),[27] which enables quantification of individual molecule binding on the sensor surface in complex media. The bioassay is performed in a single

step without signal amplification or washing procedures, and the plasmonic detection is robust against variations in optical properties of samples. Importantly, the imager is made of low-cost off-the-shelf optical components, and its small size enables deployment and operation in on-site clinical settings. Moreover, the sensor chips are produced in a scalable cost-effective manufacturing and the bioassay reagents can be stored stably over weeks. We first show the application of DENIS for highly sensitive, quantitative and robust detection of PCT and CRP biomarkers.

The system enables ultrahigh detection sensitivities with limit of detection (LOD) down to 21 and $36~\mathrm{pg~mL^{-1}}$ for PCT and CRP, respectively, and a wide dynamic range of at least three orders of magnitude. We evaluate the performance of DENIS in clinical settings by testing the samples from sepsis, noninfectious SIRS (ni-SIRS), and healthy subjects provided by Vall d'Hebron Sepsis Bank.

We benchmarked our results to the clinically validated assays and the diagnoses of patients and in agreement, we observed that rapid and quantitative detection of PCT and CRP from blood serum of the selected 34 subject cohort assisted in discrimination of healthy and sick groups. Moreover, PCT detection with DENIS show performance equivalent to gold standard clinical assays in noninfectious SIRS and sepsis patients. Finally, DENIS can perform rapid detection by recording video of the plasmonic sensor and analyzing the signals in real time.

The time-resolved results revealed that we can identify samples with high (sepsis) and medium (noninfectious SIRS) PCT levels in 5 and $10\mathrm{min}$ , respectively, reducing the test time to under $15\mathrm{min}$ .

# 2. Results

# 2.1. Detection Mechanism

The DENIS detection mechanism is based on a large-area Au-NHA device supporting SPR and localized surface plasmon modes.[28-31] Au-NHAs exhibit extraordinary optical transmission (EOT) at its resonance wavelength, controlled by the period and geometry of the nanoholes, when illuminated in a collinear optical path.[32] This plasmonic resonance is locally distorted upon the binding of single Au-NPs (100 nm diameter). These distortions result in local suppression in the EOT, which enables detection of single NP-labeled molecules.

[27] Although the NPs are subwavelength in size, their plasmonic interaction with the Au-NHA allows for high-contrast imaging of surface-bound NPs in a simple bright field optical reader over large field of view directly inside complex samples. The imaged data is not affected by the bulk sample background because the resonance transmission through Au-NHAs is strictly dependent on the surface localized field with $\approx 100\mathrm{nm}$ decay length.

Moreover, the contrast generated by a single particle is significantly above the noise level induced by the camera and the sample, therefore individual molecules binding can be robustly quantified in a digital way directly in complex media (Figure S1, Supporting Information). In contrast, more conventional affinity sensors average the readout over sensor areas, and require a certain density of analyte binding to reach a sufficient signal to be detected over the system noise, such as background fluctuations or reader noise. The digitized detection of individual

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO·MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (2 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

![](dt=2026-04-09/ht=18/473fe67d2ed74ae2b7d0463d6cb9f8dea0a0c508892f2cc7ccbe09305aec797e.jpg)

![](dt=2026-04-09/ht=18/6099552ba73a74a0de07ad39e6afaed4724925475d1554dab1e1fa33382cfd70.jpg)

![](dt=2026-04-09/ht=18/8d0cc46bc8bd5a81357a348023cad59a794e2c648a3729b9cca46c371fbac1ac.jpg)

![](dt=2026-04-09/ht=18/e7f694ff0c5c2ff32790cdfe53a1d701168c352b64f39bb387006b532c8a49b1.jpg)

![](dt=2026-04-09/ht=18/f3d7180188a71d371caf392d4e9a0418e1cad30b24dcd02a3484d4bfe81fb32b.jpg)

![](dt=2026-04-09/ht=18/9b1e2d4a9e11dddcacf9b10c4ca0ba2b8eec1ceac545687558b1732c93625e7d.jpg)

![](dt=2026-04-09/ht=18/1ed017e22757d2154ca6424500dd5cbdab083cf20ae4652f85d9222e40d2ae61.jpg)

particles with DENIS enables highly sensitive quantification of low analyte concentrations at the diffusion limit, and statistical assessment of the signal.[27] Importantly, single NPs can be distinguished on plasmonic images from NPs aggregates and sediment by the size and the shape of the plasmonic image spikes.[27]

# 2.2. Plasmonic Microarray Sensors

We use the detection technique to quantify PCT and CRP biomarkers from human serum samples. In the presence of the biomarker in the sample, antibody-conjugated Au-NPs bind to the Au-NHA surface functionalized with complementary antibodies (Figure 1a) and can be detected using a portable reader (Figure 1b). Individual Au-NPs bound inside or close to the nanoholes (Figure 1c) create strong local intensity contrast (Figure 1d), which corresponds to digital detection of single analyte molecules.

To enable microarray based multiplexed biosensing, capture antibodies are bioprinted on the sensor surface using a high-throughput, noncontact, low-volume liquid dispenser as shown in Figure 1e. The Au-NHA sensors are manufactured using a low-cost wafer-scale deep UV lithography (DUVL) and ion

beam etching (Figure 1f). The nanofabrication process enables production of over 50 highly uniformly nanostructured chips $(1\mathrm{cm}\times 1\mathrm{cm})$ per 4 in. wafer and eight wafers per batch with negligible variation in optical properties.[33] This robustness in the manufacturing of nanostructured sensors is critical to enable the transfer of the technology to a clinical diagnostic tool. We postpatterned the fabricated chips with an array of quick response (QR) codes (Figure 1g; Figure S2, Supporting Information). In addition to tagging individual chips and microarray locations on the chips, QR codes encode the information about the fabrication batch parameters and the wafer enabling a registry record of the manufactured sensors.

# 2.3. Portable Optical Reader

The nanoplasmonic imager is made of off-the-shelf optical components and comprises a narrowband light emitting diode (LED) source, spectrally matching the EOT wa
velength of Au-NHAs, a custom-built aluminum holder for the nanoplasmonic chip, an objective, and a CMOS camera. The collimated LED light is transmitted through the sample chamber and modulated by the Au-NHA sensor. The sensor images are acquired by a $50\times$ objective and CMOS sensor with $4104\times 3006$ pixels and

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO·MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (3 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

![](dt=2026-04-09/ht=18/51eb97b682ab5a9a0bcffbf4dfe1950b08f60de69123739b80909fb9d5b1fa99.jpg)

![](dt=2026-04-09/ht=18/13b9b9b94b90b5a6f2b1f318bc37719092098c0c3938893eb0f57846522ebc21.jpg)

![](dt=2026-04-09/ht=18/ad497461d89e71a715225bd0b9ab0ffa544e1ca52028a27af86d964fe319a366.jpg)

$3.45\mu \mathrm{m}$ pixel size. Each image shows a $900~{\mu\mathrm{m}}\times 700~{\mu\mathrm{m}}$ field of view (FOV) with $\approx 1.2\mu \mathrm{m}$ resolution (Figure 1g). The reader weight is below $1\mathrm{kg}$ and the dimensions are $10\times 10\times 35\mathrm{cm}$ . Such compactness means it is easily transportable and can be deployed in most on-site settings.

The control of the camera, image acquisition, and data processing are performed on a laptop computer using custom Matlab functions. In brief, images are recorded with custom autoexposure to ensure optimal intensity profile, corrected for source illumination, and the NPs binding is quantified over the sensing spots by computing the percentage of dark pixels using a cutoff threshold. QR tags from the images can be automatically recognized by custom Matlab function to register individual FOVs.

For end-point measurements, signals from at least five microarray areas from each sensor chip are recorder for statistical significance. In the case of time-resolved measurements, images of a single FOV are recorded with $30~\mathrm{s}$ intervals, and the signals are extracted from at least three microarray spots to estimate mean and standard deviation. In order to control for nonspecific NPs binding and background fluctuations, in each measurement the signals from bovine serum albumin (BSA) blocked background are subtracted from the signals of Ab microarray spots.

# 2.4. Bioassay

Our simple and rapid bioassay is performed in a single step, where antibody functionalized Au-NP (Ab-NP) suspension is mixed directly with blood serum and then injected into the

measurement chamber (Figure 2). We prepare large batches of Ab-NPs with optical density $\mathrm{OD} = 20$ and store them in phosphate buffer saline (PBS $1\times$ ) buffer with BSA (1% w/v) and Tween20 (0.05 % w/v). The Ab-NP suspension is stable at $4^{\circ}\mathrm{C}$ for at least 5 weeks, over which the tests were performed. The sensor chips are first uniformly functionalized with copoly-DMA-MAPS-NAS-fluorinated polymer (MCP-2F), containing activated amine reactive NHS-ester groups, to ensure stable antibody immobilization.

Moreover, the fluorocarbon polymer helps prevent nonspecific NP binding, and fouling by serum proteins.[34] To form antibody microarrays, we spot antibodies specific to the biomarkers as well as mouse isotype immunoglobulin-G (IgG) for nonspecific negative control both at $200\mu \mathrm{g}\mathrm{mL}^{-1}$ . Each microarray spot of $\approx 150\mu \mathrm{m}$ diameter is formed by a single $400~\mathrm{pL}$ droplet dispensed with $400~\mu \mathrm{m}$ period.

Importantly, the use of microarray printer not only enables functionalization of large number of chips, but also minimizes the amount of antibodies used, which are one of the most expensive components of the assay. The uniform antibody immobilization on the sensor surface is ensured by adding trehalose (0.5% w/v) and Tween20 (0.01% w/v) in the PBS $1\times$ spotting buffer. The sensor chips with IgG microar

rays are blocked with BSA (1% w/v) to minimize the nonspecific binding. A capillarity-based disposable microfluidic platform is sealed on the sensor chips using a silicone spacer.[33] The antibody immobilized chips can be stored at $4^{\circ}\mathrm{C}$ stably for at least 5 weeks (Figure S6, Supporting Information). To increase the performance of our assay, we optimized a stabilizing buffer, which consists of PBS Tween20 (PBST, $1\times$ , $0.05\%$ w/v) and NaOH $(50\times 10^{-3}\mathrm{M})$ in 3:1 ratio.

To perform the bioassay, $20~\mu\mathrm{L}$ of serum is mixed with $4\mu\mathrm{L}$ of Ab-NP suspension and $16~\mu\mathrm{L}$ of stabilizing buffer, and the mixture is loaded into the measurement chamber with a pipet. The loaded cartridge is either inserted into the reader for time-resolved data acquisition or measured after an incubation period.

To characterize the performance of DENIS system, calibration titration curves were obtained by spiking known amounts of PCT and CRP biomarkers in reference samples (Figure 3). Each data point represents mean value and standard deviation from five measurements collected from different microarray areas. The LOD values were estimated by measuring the mean signal from blank samples and adding three times the standard deviation. Our detection method achieves LOD of 21 and $36~\mathrm{pg~mL^{-1}}$ for PCT and CRP, respectively.

Moreover, the PCT detection dynamic range of DENIS is three orders of magnitude (i.e., from $\approx 21$ to $>1\mathrm{E}4~\mathrm{pg~mL^{-1}}$ ). The clinical serum levels of PCT usually range from tens of $\mathrm{pg~mL^{-1}}$ for healthy individuals to few hundreds $\mathrm{pg~mL^{-1}}$ for SIRS patients free of bacterial infection, and above $\approx 1~\mathrm{ng~mL^{-1}}$ in sepsis patients.[35] Therefore, our method covers the full relevant diagnostic range. The clinical serum levels for CRP range from few $\mu \mathrm{g~mL^{-1}}$ for healthy

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO·MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (4 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

![](dt=2026-04-09/ht=18/8a6c8c08318adaa68b7561be8c2b3567e0f881d7f9b0f9cd4ac2026646e80c49.jpg)

![](dt=2026-04-09/ht=18/4a7e164d068e93e776d266e0ec1a344c919f60bb049d6e80166fc020c39d74e1.jpg)

individuals to hundreds of $\mu \mathrm{g} \, \mathrm{mL}^{-1}$ for septic patients,[35] which are significantly higher than the values for PCT. To measure them with the same assay, we diluted serum for CRP measurements in order to comply with the dynamic range of the device, which covers from 36 to $1\mathrm{E}5 \, \mathrm{pg} \, \mathrm{mL}^{-1}$ . Importantly, the sensitivity and dynamic range enabled by DENIS are similar to the performance of gold standard lab techniques, without the need for time-consuming amplification and washing steps or bulky readout equipment.

# 2.5. Clinical Tests of the DENIS System

We deployed the DENIS system at the Vall d'Hebron University Hospital and tested with clinical samples previously collected from patients with sepsis, noninfectious SIRS and healthy individuals (previous informed consent agreement). Even though the new sepsis definitions prefer to avoid the term SIRS, this clinical condition is still worthy when defining the appropriate study groups with research purposes.

[1] We obtained and measured on the DENIS platform samples, previously anonymized, from 34 individuals, including 18 patients with sepsis, 11 with noninfectious SIRS, and 5 healthy donors (Table S1, Supporting Information). The detection of PCT and CRP levels in the patient serum was performed as described above using only $20~\mu \mathrm{L}$ of sample for each measurement. The measur
ements were done in a blind manner, where neither the patient status nor the biomarker concentrations were disclosed during the testing in order to eliminate bias.

After the completion of the clinical tests, the results collected from DENIS assay were compared with the routine clinical determinations. Specifically, CRP and PCT detection were performed using ITA and CLIA, respectively, and bacterial infection was determined with blood culture. The example images of single PCT spots obtained by DENIS system directly in serum from the three patient groups are presented in Figure 4a.

Samples from septic patients contain high levels of PCT or CRP, and result in dense binding of particles on antibody microspots of the Au-NHA sensor, while noninfectious SIRS samples produce images with sparsely bound NPs, and healthy samples present negligible NP binding on the sensing areas. Figure 4b shows CRP and PCT values measured from nine representative individuals from the three groups. Each data point represents the mean and STD from at least five different spots. We observed consistent results correlating the PCT readouts from

DENIS to the CLIA measurements in the clinical lab (inset Figure 4b). In every sample measured, we controlled for nonspecific interactions between Ab-NPs and the Au-NHA surface by recording signals from mouse IgG spots and BSA blocked areas (Figure 5). Notably, we observed no significant signals from these control measurements on every chip tested, therefore no washing steps were needed.

To validate the performance of the DENIS platform, we mapped the PCT and CRP detection results of 34 individuals to the measurements with CLIA and ITA, based on the clinical diagnosis (see Figure 6a,b and Figure S3 in the Supporting Information).

In the sampled group, the measured levels of PCT and CRP using both DENIS and clinical assays were significantly elevated in noninfectious SIRS group compared to healthy subjects $(P_{\mathrm{PCTDENIS}} < 1.7\mathrm{E} - 5, P_{\mathrm{CRPDENIS}} < 1.7\mathrm{E} - 3,$ unpaired t-test), and in sepsis group compared to noninfectious SIRS $(P_{\mathrm{PCTDENIS}} < 2.3\mathrm{E} - 7, P_{\mathrm{CRPDENIS}} < 4.9\mathrm{E} - 3,$ unpaired t-test).

However, CRP was reported in multiple sources to be a general inflammatory marker that is not well discriminative of the stage or the source of inflammatory host response.[35,36] Therefore, in order to compare the performance of DENIS to clinically validated assays in assisting the discrimination between sepsis, noninfectious SIRS, and healthy groups, we constructed receiver operating characteristic (ROC) curves using PCT measurements with both techniques as a predictor (Figure 6c,d). We also calculated the corresponding area under the curve (AUC).

ROC curves are an established metric to analyze

![](dt=2026-04-09/ht=18/98e8d2d36e8cbc5f896b116eec5e370eed1c139fa27412bd0a7981b68f758c23.jpg)

![](dt=2026-04-09/ht=18/ae05a456dc94fca1f0e15bebb94b4fb6056251986c3c9d2212b05ed7e3258be2.jpg)

![](dt=2026-04-09/ht=18/41c3c491ca8195281d30b7519c13775b8f394496cc43f1567165dfd0631cd8e5.jpg)

![](dt=2026-04-09/ht=18/fdc1e4f9d73e1cf8494f3ce39e40a583d034bb9972b822f843b9ce9468ae5dd0.jpg)

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO - MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (5 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

![](dt=2026-04-09/ht=18/aa5e6dbcb980943c05034d346708d5fcc3953dfe4a050296dc2c96236225fd9d.jpg)

![](dt=2026-04-09/ht=18/1d036bf3627117ba86ccd9595306eafad3d4315fddb9c934727738b1b3193951.jpg)

diagnostic ability of a test to discriminate the true state of subjects, and to compare two alternative diagnostic tests when each is performed on the same subject.[37] DENIS measurements of PCT provided equivalent performance to CLIA in discriminating between the three study groups. With both methods, we observed $100\%$ sensitivity and specificity in classifying between healthy and sick subjects, moreover PCT was a good indicator of host response in sepsis compared to noninfectious SIRS, with AUC = 0.995, sensitivity = 0.944, and specificity = 1.

Although the patient cohort of 34 subjects is not large enough to reflect the general population statistics, these findings are in agreement with the literature, where PCT levels in blood were shown to be a useful indicator to assist in discrimination of noninfectious SIRS from sepsis patients.[35,36] Importantly, the results of the DENIS assay showed robust performance, even though the patients serum samples presented significant visual differences in the color and consistency (Figure S4, Supporting Information).

# 2.6. Rapid Biomarker Detection

Rapid turnaround time is an essential characteristic of a sepsis diagnostic device, as sepsis is a time-dependent emergency

with a severe disease progression within hours. We show the diagnostic use of DENIS with minimal time-to-result enabled by time-resolved measurements. The system can record real-time video of the sensor (Video S5, Supporting Information), from which NP binding signals are extracted and analyzed in real time. In Figure 7, we present time-resolved PCT detection signals from sera of representative sepsis patients, noninfectious SIRS patients, and healthy individuals. Each plot shows mean signal from three PCT sensing spots and its $95\%$ confidence interval in the shaded area.

To establish the background signal and its variation, we plot averaged signal from three BSA blocked areas and compute the running time averaged $95\%$ confidence interval. We consider a reliable detection when the PCT signal's $95\%$ lower confidence interval exceeds the background's upper confidence interval.

We characterized PCT cutoff detection times for high (sepsis), medium (noninfectious SIRS) and low (healthy) PCT concentration groups and observed that we detect sepsis levels in less than $5\mathrm{min}$ and noninfectious SIRS levels in around $10\mathrm{min}$ , while PCT signal from healthy individuals did not significantly exceed background over $2\mathrm{h}$ . These results show that PCT detection can be performed in under $15\mathrm{min}$ , including sample mixing, injection and readout, which is particularly important in intensive care units.

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO·MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (6 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

![](dt=2026-04-09/ht=18/4ee3236f891aeb0a83ca3181e533593bc127514e9d360e534b42b4619bba3ffb.jpg)

![](dt=2026-04-09/ht=18/bf753a3ef190dc24d66520edb647570cdf774a4445057c40ad6781a749d7a311.jpg)

![](dt=2026-04-09/ht=18/bd62a3d05767165c1300b6fe8442b3a50003d6d89a8b0b6852b3b97148a1a6ef.jpg)

![](dt=2026-04-09/ht=18/7c7a33808b90781457e4b48930bebeeaef68e0ad488e27c8d19d86a9f2d3c1ff.jpg)

# 3. Discussion

DENIS device proved to be particularly suitable for on-site operation due to its portability, inexpensive off-the-shelf optical components, and because it does not involve any moving parts or complex microfluidic elements prone to clogging. Importantly, the sensor chips and bioassay reagents can be stored in the fridge stably ove
r weeks and unlike previously reported optical detection schemes[21-23] do not require fluorescent tags that can be expensive, unstable, and difficult to produce.

The simple and rapid bioassay enables highly sensitive and robust detection of PCT and CRP, covering the full clinically relevant ranges. The high sensitivity of the assay is enabled by the unique plasmonic mechanism that works by detecting individual molecules on the Au-NHA surface through imaging of single Au-NPs, without the need for complex signal amplification procedures.

The system achieved $21.3\mathrm{pg}\mathrm{mL}^{-1}$ LOD sensitivity for PCT and $36\mathrm{pg}\mathrm{mL}^{-1}$ LOD for CRP, and a dynamic range of at least three orders of magnitude, which is well within the clinical biomarker levels.

DENIS was validated with measurements of PCT and CRP biomarkers from sera of 34 individuals, including 18 sepsis patients, 11 noninfectious SIRS patients, and 5 healthy controls, and was shown to provide consistent and robust performance

at the level of gold-standard immunoassays. Both PCT and CRP levels were significantly elevated in the sick groups compared to healthy subjects. In the sampled group DENIS measurements of PCT levels showed performance equivalent to clinically validated assay in assisting classification between sepsis and noninfectious SIRS patients, with $94.4\%$ sensitivity and $100\%$ specificity. These results show that DENIS platform can be effectively deployed to help rapidly triage patients between noninfectious SIRS and sepsis groups.

The detection is robust against background noise, nonspecific adsorption of proteins, and variations in optical properties of the serum, enabling reliable detection. In combination with small sample volume used $(20~\mu \mathrm{L})$ , this makes DENIS suitable for frequent biomarker testing, such as required during treatment response monitoring.

The sensor chips were fabricated in a robust wafer-scale cost-effective process using DUVL lithography. The functionalization of sensors was performed in a scalable way with microarray bioprinter that uses minimal amounts of antibodies. In combination with easy-to-use bioassay and a compact optical reader, this makes the DENIS technology suitable for the development of a clinical diagnostic device. Owing to the high sensitivity, rapid time to result, compactness and cost efficiency, DENIS is expected to be employed in hospitals as well as small clinic and

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO - MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (7 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

![](dt=2026-04-09/ht=18/9d6c5c49a72892eddbb20e19638becbdc63af894085aa8d9738f9d876bdb9917.jpg)

![](dt=2026-04-09/ht=18/926bf1428779d9fbf4e35b6247ea8689e812ea07c7ceb1b23527301630068ae2.jpg)

![](dt=2026-04-09/ht=18/087720719b3c65a08b099efa7bcb961423e9031a9066b8cad38dd8a16e02324a.jpg)

physician's office settings, where the access to large and costly laboratory testing equipment is limited, therefore assisting timely diagnosis and reducing antibiotic prescription.

We envision several ways in which the current DENIS system could be improved. First, the imaging readout enables scaling up the test for multiplexed measurements.[38] Sepsis manifests in a complex system of pathophysiological responses, spanning multiple organs. Creating a sepsis marker panel and expanding DENIS to multiplexed detection of relevant biomarkers with comparable dynamic ranges, such as presepsin, MR-pro-adrenomedullin, pancreatic stone protein, and interleukin 6 from

no more than few $\mu \mathrm{L}$ of sample could provide more robust characterization of the patient's host response and potentially predict survival.[13,25,39,40] Importantly, the detection principle is not limited to proteins and can be extended to small peptides, DNA, or RNA, as long as a pair of complementary recognition elements is available. The development of the bioassay to operate with whole blood would deliver even faster time to result by eliminating serum extraction step and better position the test for the use in resource-limited settings.

Further reducing the costs of the imager and scaling up the production of Au-NHA sensors would push the per-test costs even lower and open the path to potential market entry. Finally, the development of a fully integrated easy-to-use software interface will facilitate introduction of the device to clinical professionals and help validate the technology on large sets of clinical samples.

# 4. Conclusion

Our study reports a portable digital nanoparticle-enhanced plasmonic imager that enables rapid detection of two inflammatory sepsis-related biomarkers, PCT and CRP. The unique nanoplasmonic mechanism through imaging of single Au-NPs binding on the Au-NHAs enables highly sensitive and rapid biomarker detection directly in blood serum. The compact nanoplasmonic reader, built of inexpensive components, weighs less than $1\mathrm{kg}$ .

The portable device was deployed in hospital settings, where it was shown to provide equivalent performance to gold-standard laboratory assays using a wide range of samples from patients with sepsis, noninfectious SIRS, and healthy donors. Moreover, identification of PCT levels typical for sepsis, SIRS, and healthy subjects can be performed in less than 15 min, which is significantly faster than laboratory immunoassays.

Therefore, DENIS platform promises to become a powerful tool in managing acute inflammatory conditions at the point-of-care in an affordable way, improving the quality of health care for more people.

# 5. Experimental Section

Au-NHA Sensor Fabrication: Au-NHAs were produced using waferscale low-cost DUVL process and ion beam etching. Fused silica wafers with 4 in. diameter were used as substrate. The wafers were cleaned with RCA solution (1:1:5 $\mathsf{H}_2\mathsf{O}_2:\mathsf{NH}_4\mathsf{OH}:\mathsf{H}_2\mathsf{O}$ by volume) for $15\mathrm{min}$ , rinsed with deionized water for $5\mathrm{min}$ , and dried under nitrogen stream. Next, the cleaned wafers were coated with $10\mathrm{nm}$ of titanium (Ti) and $120\mathrm{nm}$ of gold (Au) using Alliance-Concept EVA 760 electron-gun evaporator.

The NHAs with $200\mathrm{nm}$ diameter and $600\mathrm{nm}$ period were patterned over whole wafers using a $248\mathrm{nm}$ deep-UV stepper (ASML PAS 5500/300 DUV). The exposed wafers were treated with ion-beam etching (Oxford Instruments PlasmaLab 300 IBE), to transfer the nanohole arrays into metal films. The photoresist was stripped using oxygen plasma.

To label the Au-NHA sensors, QR codes were formed with $50~\mathrm{nm}$ Ti film. The NHA wafers were coated with $1.3\mu \mathrm{m}$ thick lift-off positive photoresist (AZ1512) using Suss ACS200 Gen3 coater, and the QR microarray patterns were exposed using Heidelberg Instruments MLA150 Maskless Aligner. The resist was developed, and $50~\mathrm{nm}$ Ti film was evaporated using an electron gun evaporator. Wafers covered with photoresists and Ti were diced into $1\times 1\mathrm{cm}^2$ sensor chips, and the photoresist was removed from chips by immersing in resist remover with sonication at $70^{\circ}\mathrm{C}$ overnight. The final chip cleaning was

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO·MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (8 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

performed with oxygen plasma for 5 min at $500\mathrm{W}$ and 1 min immersion in RCA solution (1:1:5 $\mathsf{H}_2\mathsf{O}_2:\mathsf{NH}_4\mathsf{OH}:\mathsf{H}_2\mathsf{O}$ ) to ensure uniformly clean Au surface.

Chemicals and Biologicals: Ammonium hydroxide solution (ACS reagent, 28–30%), hyd
rogen peroxide $(\mathrm{H}_2\mathrm{O}_230\%)$ , BSA lyophilized, PBS, Tween20, and D-(+)-trehalose dihydrate were purchased from SigmaAldrich. MCP-2F polymer was bought from Lucidant. Ethanol (EtOH) absolute was from Thermo Chemicals. Pierce protease inhibitor (PI) tablets, recombinant human PCT and anti-PCT monoclonal antibodies QN05 and CALC, used for spotting on Au-NHA and NP conjugation, respectively, were provided by ThermoFisher (BRAHMS).

Antihuman C-reactive protein monoclonal IgG antibodies PCR-196 and PCR-183, used as capture and recognition antibodies, respectively, were provided by Diesse Diagnostica Senese. Au-NP conjugation kit with lyophilized spherical Au-NPs of $100~\mathrm{nm}$ diameter coated with polyethylene glycol (PEG, 10 kDa) and activated with 1-Ethyl-3-(3-dimethylaminopropyl) carbodiimide-N-hydroxysuccinimide chemistry (EDC-NHS), were purchased from Cytodiagnostics.

Au-NHA Functionalization: Au-NHA chips were coated with MCP-2F (copoly-DMA-MAPS-NAS-fluorinated) polymer (Lucidant) to yield stable and reproducible functionalization with Ab microarray spots and prevent the sensors from nonspecific fouling by serum and NPs. The polymer contains activated NHS-ester groups that enable covalent immobilization of molecules through amino groups. The MCP-2F polymer stock was diluted with coating solution from Lucidant, and the Au-NHA chips were immersed in the solution for 30 min at room temperature.

Next, the Au-NHA sensors were washed in large volumes of mQ water and dried under nitrogen stream. The sensors were placed in a vacuum oven at $80^{\circ}\mathrm{C}$ for 15 min $(<2\mathrm{mmHg})$ to dry the polymer and ensure stable adhesion to the Au surface. Polymer-functionalized chips were immediately spotted with Ab microarrays because NHS-ester groups of the polymer degrade in the presence of air humidity.

The Ab microarrays were formed on the sensors by spotting Ab solutions on polymer coated Au-NHA surface using noncontact piezoelectric ultralow volume dispensing system (sciFLEXARRAYER S3, Scienion). Spotting solutions containing Ab specific against PCT, or CRP, or mouse isotype control IgG, all at $200\mu \mathrm{g}\mathrm{mL}^{-1}$ were prepared in PBS $1\times$ with trehalose $(0.5\% \mathrm{w / v})$ and Tween20 $(0.01\% \mathrm{w / v})$ to ensure uniform Ab distribution over each spot and stabilize Abs during sensor storage.

Microarray spots of 400 pl were dispensed with a period of $400~{\mu\mathrm{m}}$ inside a humid chamber. The humidity control was set at dew point (65% relative humidity) to prevent evaporation. The sensor chips with IgG microarrays were blocked with BSA $(1\% \mathrm{w / v})$ to inactivate nonreacted EDC-NHS groups and minimize the nonspecific binding. The sensor chips were stored until use at $4^{\circ}\mathrm{C}$ for up to 4 weeks.

Au-NP Functionalization: Au-NPs of $100\mathrm{nm}$ diameter coated with $10\mathrm{kDa}$ PEG and activated with EDC-NHS were functionalized with aPCT or aCRP antibodies according to the following protocol. Antibodies were diluted in protein resuspension buffer (Cytodiagnostics) to $500~\mu \mathrm{g}~\mathrm{mL}^{-1}$ concentration. The antibody solution $(40~\mu \mathrm{L})$ was mixed with reaction buffer $(50~\mu \mathrm{L}$ , Cytodiagnostics), and added to the vial of lyophilized Au-NPs.

The Au-NP mixture with Abs was incubated at room temperature for $4\mathrm{h}$ with a mild mixing using Eppendorf ThermoMixer at $700~\mathrm{rpm}$ . After incubation, $10~\mu \mathrm{L}$ of quencher solution (Cytodiagnostics) was added to stop the reaction. The Au-NPs suspension was centrifuged at $150~\mathrm{rcf}$ for $30\mathrm{min}$ to wash the NPs from nonreacted Abs. The supernatant was removed and NPs were resuspended in $1\mathrm{mL}$ of PBS $1\times$ with $1\%$ w/v BSA and $0.05\%$ w/v Tween20. Centrifugation and resuspension steps were repeated 3 times.

At the final resuspension step, Au-NPs pellet was diluted in a $100~\mu \mathrm{L}$ of PBS $1\times$ with $1\%$ w/v BSA and $0.05\%$ w/v Tween20 to obtain Au-NP stock solution with optical density $\mathrm{OD} = 20$ (measured at $572~\mathrm{nm}$ ), or $7.7E + 10~\mathrm{NP~mL}^{-1}$ . The Ab-NPs solution was stored at $4^{\circ}\mathrm{C}$ for up to 4 weeks. Before performing bioassay, Ab-NPs were vortexed at $800~\mathrm{rpm}$ for $30~s$ to resuspend the NPs.

Biobank Samples: Samples and data from the patients used in this study were provided by the Sepsis Bank of Vall d'Hebron University

Hospital Biobank (PT17/0015/0047), integrated in the Spanish National Biobanks Network. Samples were processed following standard operating procedures with the appropriate approval of the Clinical Research Ethics Committee (approval reference number PR(AG)11/2016). Sample selection was done to ensure a wide range of concentrations to be measured with DENIS, from low PCT and CRP values associated with healthy individuals, to medium levels usually observed in noninfectious SIRS patients and increasing values that are characteristic of sepsis and septic shock patients.

It should be noted that sepsis is a complex syndrome and patient groups could have overlapping biomarker levels. For each patient the samples were collected at a single time point. For sepsis patients, samples were collected at the time sepsis was diagnosed. For noninfectious SIRS patients, samples were collected in the immediate postoperative period. The samples were stored in $500~\mu \mathrm{L}$ aliquots at $-80^{\circ}C$ in cryovials. Samples were handled on ice and protected from light.

Serum Bioassays: The bioassays were performed by mixing patient serum with the Ab-NP conjugates (aPCT or aCRP) and a stabilizing buffer and injecting the mixture into the sensor measurement chamber. A $20~\mu \mathrm{L}$ aliquot of serum sample was mixed with $16~\mu \mathrm{L}$ of buffer and $4~\mu \mathrm{L}$ of aPCT Ab-NP stock by gentle pipetting and inserted onto the measurement chamber. In the case of CRP measurements, the patient serum was first diluted in PBS $1\times$ with bovine serum albumin $7\%$ .

The serum dilution was done in two steps, first 100 times and then 500 times, to minimize the error, reaching the final dilution of 50 000 times. Then, a $20~\mu \mathrm{L}$ aliquot of the diluted serum was mixed with $16~\mu \mathrm{L}$ of measurement buffer and $4~\mu \mathrm{L}$ of anti-CRP Ab-NPs by gentle pipetting and inserted to the sensor chamber. Sensor chamber was formed by placing a capillarity-based disposable microfluidic cartridge (Microtec) on the Au-NHA sensor chip using a silicone spacer for sealing.

The loaded sensor was either immediately inserted into the portable reader for real-time measurements or incubated at room temperature, and end-point readouts were taken at the end of the incubation period (2 h).

The stabilizing buffer was optimized to ensure optimal bioassay performance and prevent NP aggregation during the measurements. The buffer consists of $0.05\%$ PBST $1\times$ and $50\times 10^{-3}$ NaOH mixed in 3:1 volume ratio.

Calibration Measurements: In order to perform calibration measurements of the PCT bioassay, known concentrations of recombinant human PCT $(50\mathrm{pg}\mathrm{mL}^{-1}, 100\mathrm{pg}\mathrm{mL}^{-1}, 300\mathrm{pg}\mathrm{mL}^{-1}, 500\mathrm{pg}\mathrm{mL}^{-1}, 1\mathrm{ng}\mathrm{mL}^{-1}, 3\mathrm{ng}\mathrm{mL}^{-1}, 10\mathrm{ng}\mathrm{mL}^{-1},$ and $50\mathrm{ng}\mathrm{mL}^{-1})$ were spiked in $7\%$ BSA PBS $1\times$ with protease inhibitors (1 tablet per $50~\mathrm{mL}$ of buffer).

Next, the calibration samples, as well as a blank negative control containing BSA $7\%$ in PBS $1\times$ and PI without PCT, were measured identically to the serum bioassays. A $20~\mu \mathrm{L}$ aliquot of calibration sample was mixed with $16~\mu \mathrm{L}$ of measurement buffer and $4~\mu \mathrm{L}$ of Ab-NP, incubated over $2\mathrm{h}$ and measured with the portable reader to obtain the corresponding signals. For each
calibration sample, at least five different antibody sensing spots were measured to compute the mean and the standard deviation of the signal.

To obtain calibration curve for CRP, serial dilutions of a real human sample were used with a known CRP concentration characterized by an immunoturbidimetric test $(266.9\mu \mathrm{g}\mathrm{mL}^{-1})$ . The sample was diluted in BSA $7 \%$ PBS $1\times$ at dilution ratios of $1:750\times 10^{3}$ $1:500\times 10^{3}$ $1:100\times 10^{3}$ $1:50\times 10^{3}$ $1:10\times 10^{3}$ , and $1:10^{3}$ times, and the measurements were performed identically to PCT. Calibration curves for PCT and CRP were fitted to the measured points, according to the following equation

$$
y = d + \frac {a - d}{1 + \left(\frac {x}{c}\right) ^ {b}} \tag {1}
$$

where $a$ is a theoretical response at zero concentration, $b$ is a slope factor, $c$ is a concentration inflection point, and $d$ is a theoretical response at saturation.

Optical Reader: The nanoplasmonic imager is made of off-the-shelf optical components and effectively comprises a portable bright-field imaging microscope. A narrowband light emitting diode (LED) source

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO - MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (9 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

(Thorlabs M660L4) in combination with a bandpass filter with $660\mathrm{nm}$ center wavelength and $10\mathrm{nm}$ full width at-half-maximum (Thorlabs) is used for narrow-band illumination at the wavelength of Au-NHA EOT peak. The light from the LED is collimated with an aspherical condenser lens (Thorlabs). A custom aluminium holder for the sensor chamber was fabricated to accommodate the sensor inside the reader.

The light is transmitted through the sample chamber, modulated by Au-NHA sensor, and the imaging is performed using a $50\times$ objective (Nikon) and a black-and-white CMOS camera (IDS B-UI-3200SE-M-GL). The CMOS camera has $4104\times 3006$ pixels with $3.45\mu \mathrm{m}$ pixel size. The system enables imaging of $\approx 900\mu \mathrm{m}\times 700\mu \mathrm{m}$ FOV area with an $\approx 1.5\mu \mathrm{m}$ resolution (diffraction-limited spot). The reader dimensions are $10\times 10\times 35\mathrm{cm}$ and the weight is less than $1\mathrm{kg}$ .

Image Processing: Image acquisition and processing was performed using custom Matlab functions and a graphical user interface from a laptop connected to the CMOS camera. Images from Au-NHA sensors were recorder using custom autoexposure function to ensure similar intensity profiles across all images, and so that image histogram is always covered by the dynamic range of the CMOS. The images were normalized by the background illumination to exclude the source variations. To extract the NP signal, image areas were analyzed using a fixed intensity threshold to quantify bound NPs.

The percentage number of pixels in an area darker than the threshold was used for quantification. The intensity threshold was optimized by matching plasmonic images to scanning electron micrographs of identical Au-NHA areas to maximize the number of detected NPs, while minimizing false-positive signals.[27] Importantly, Au-NP aggregates can be identified and filtered out from the signal, as described previously.[27] A custom Matlab function was used to recognize QR codes from the images and extract encoded information.

For the end-point measurements at least five different FOV were acquired and analyzed from each sensor chip for statistical significance. In order to control for nonspecific NPs binding and background fluctuations, signal from each measured spot was corrected by subtracting the signal from adjacent BSA blocked background area. During time-resolved measurements, the images of a single FOV were recorded with 30 s interval. Each field of view had at least three antibody spots, which enabled to assess the mean and standard deviation of the signal from each frame.

In order to evaluate detection cutoff times, signal from three PCT sensing spots and its $95\%$ confidence interval $(\pm 1.96^{*}\delta)$ were estimated at each time point. To establish the background signal and its variation, average from three BSA blocked areas is plotted at each time point and running time averaged at $95\%$ confidence interval is computed. The time-averaged standard deviation at every time point $T$ was computed to incorporate variances of background signals of all individual previous frames $t_i$

$$
\delta_ {T} = \sqrt {\frac {\sum_ {i = 0} ^ {n} \delta_ {t _ {i}} ^ {2}}{n}} \tag {2}
$$

A reliable detection was considered when the PCT signal's $95\%$ lower confidence interval exceeds the background's upper confidence interval.

Software Availability: All custom plug-ins associated with this manuscript are available upon request.

# Supporting Information

Supporting Information is available from the Wiley Online Library or from the author.

# Acknowledgements

The authors would like to thank Center of MicroNano Technology at École Polytechnique Fédérale de Lausanne and University of California Santa Barbara Nanofabrication Facility for providing support with chip manufacturing, Diesse Diagnostic Senese SPA for providing human

C-reactive protein (CRP) and anti-CRP monoclonal IgG antibodies, and BRAHMS ThermoFisher for providing recombinant human procalcitonin (PCT) and anti-PCT monoclonal antibodies. The authors are grateful for the support of the Ecole Polytechnique Fédérale de Lausanne, including the Technology Transfer Office's Enable program. The authors acknowledge the European Union's Horizon 2020 Research and Innovation Program under grant agreement no. 644956 (RAIS project).

The VHIR-HUVH is supported by Plan Nacional de I+D+i 2013-2016 and Instituto de Salud Carlos III, Subdirección General de Redes y Centros de Investigación Cooperativa, Ministerio de Economía, Industria y Competitividad, and Spanish Network for Research in Infectious Diseases (REIPI RD16/0016/0003)—cofinanced by European Development Regional Fund "A way to achieve Europe," Operative program Intelligent Growth 2014. The authors kindly appreciate the generous donation of samples and clinical data of the donors of the Sepsis Bank of HUVH Biobank.

# Conflict of Interest

The authors declare no conflict of interest.

# Author Contributions

A.B. and F.Y. contributed equally to this work. A.B., F.Y., and H.A. conceived the study. A.B. and F.Y. developed the DENIS device and designed the experiments. A.F., J.G., J.R., and R.F. coordinated collection of human samples and clinical data and assisted with interpretation of results. A.B. and F.Y. conducted experiments and analyzed the data. A.B., F.Y., and H.A. wrote the manuscript with review from all other authors.

# Keywords

gold nanoparticles, imaging biosensors, nanoplasmonics, point-of-care diagnostics, sepsis

Received: October 23, 2019

Published online:

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO·MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (10 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

NANO - MICRO small

www.small-journal.com

Small 2019, 1906108

1906108 (11 of 11)

© 2019 WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim