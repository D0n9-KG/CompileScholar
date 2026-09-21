# Signal-carrying speckle in Optical Coherence Tomography: a methodological review on biomedical applications

Vania Bastos Silva $^{a,b}$ , Danilo Andrade De Jesus $^{b,*}$ , Stefan Klein $^{b}$ , Theo van Walsum $^{b}$ , João Cardoso $^{a}$ , Luisa Sánchez Brea $^{b}$ , and Pedro G. Vaz $^{a}$

$^{a}$ Laboratory for Instrumentation, Biomedical Engineering and Radiation Physics (LIBPhys-UC), Department of Physics, University of Coimbra, Coimbra, Portugal $^{b}$ Biomedical Imaging Group Rotterdam, Department of Radiology & Nuclear Medicine, Erasmus MC, University Medical Center Rotterdam, the Netherlands $^{*}$ d.andradejeesus@erasmusmc.nl

# Abstract

Significance: Speckle has historically been considered a source of noise in coherent light imaging. However, a number of works in optical coherence tomography (OCT) imaging have shown that speckle patterns may contain relevant information regarding sub-resolution and structural properties of the tissues from which it is originated.

Aim: The objective of this work is to provide a comprehensive overview of the methods developed for retrieving speckle information in biomedical OCT applications.

Approach: PubMed and Scopus databases were used to perform a systematic review on studies published until April 2021. From 134-screened studies, 37 were eligible for this review.

Results: The studies have been clustered according to the nature of their analysis, namely static or dynamic, and all features were described and analysed. The results show that features retrieved from speckle can be used successfully in different applications, such as classification and segmentation. However, the results also show that speckle analysis is highly application-dependant, and the best approach varies between applications.

Conclusions: Several of the reviewed analysis were only performed in a theoretical context or using phantoms, showing that signal-carrying speckle analysis in OCT imaging is still in its early stage, and further work is needed to validate its applicability and reproducibility in a clinical context.

Key words: speckle, image processing, image analysis, imaging coherence, tomography

arXiv:2108.13109v1 [physics.med-ph] 30 Aug 2021

1

# 1 Introduction

Optical Coherence Tomography (OCT) is an optical imaging modality based on low-coherence interferometry. It is a non-invasive technique that provides in vivo cross sectional images of microscopic structures with high spatial and temporal resolutions, making it an appealing technique for multiple areas in pre-clinical and clinical research [1].

In OCT imaging, the tissue is scanned by an optical beam, and most of the light is either refracted or scattered. The incident light travels through different optical paths, with different lengths, until it reaches the image plane. The light intensity at each point of the plane results from destructive/constructive interference of all light waves at that single point. This phenomenon, illustrated in Figure 1, creates granular patterns, known as speckle patterns. Speckle appears everywhere when an optically rough surface is illuminated with coherent light, making it common to all coherent imaging modalities [2].

Schmitt et al. [3] were among the first to discuss the OCT speckle origin, distinguishing between two types of speckle that appear on a cross-sectional image: multiple backscattering of the light beam, and delays caused by multiple forward scattering. Since then, a number of works have interpreted speckle as a source of noise in OCT imaging, as it reduces the image quality and contrast, making boundaries between tissues less distinguishable. Because of this, methods to suppress and reduce speckle have been developed, including filtering [4], averaging [5] or wavelet processing techniques [6].

Speckle patterns change depending on different parameters, such as the properties of the light source, the propagating beam, the aperture of the detector, and the inner properties and structural organization of the tissues [7,8]. The latter indicates that speckle may contain relevant information regarding sub-resolution and structural properties of the tissues from which it originated [9]. In fact, in the work presented by Schmitt et al.

[3], speckle patterns in OCT are already mentioned as having a dual role, both as a source of noise, signal-degrading speckle, and as a carrier of information, signal-carrying speckle. This indicates that, besides the granular noise observed in OCT raw images, the imaged speckle also carries information. This information may be used to characterize the imaged tissue [10].

Since most of the works in the literature focus on signal-degrading speckle, the information regarding signal-carrying speckle analysis is diffuse and sometimes abstruse. Thus, this review intends to provide a comprehensive overview of the different methods used to retrieve information from OCT speckle in biomedical applications.

The remainder of this paper is structured as follows: Section 1 presents the introduction; Section 2 presents the literature search criteria; Section 3 presents the signal processing methods used for analysing the OCT signal-carrying speckle. Discussion of some of the most relevant approaches is presented in Section 4. Finally, the conclusions are presented in Section 5. Supplementary materials are provided in the Appendix, namely a table summarizing the reviewed works (Appendix A) and a short theoretical mathematical description of

2

![](dt=2026-05-29/ht=23/9613f78a712ce2bb4fe2a2d992a0756a1f04d287edad66001b24895101beb431.jpg)

light speckle (Appendix B).

# 2 Methods

The literature search was conducted in two databases on April $3^{\mathrm{rd}}$ , 2021. PubMed was chosen for being one of the largest databases in the medical field, and Scopus for combining articles from both medical and technical fields. The search query used was: "Optical Coherence Tomography" AND speckle AND (statistics OR statistical) NOT flowgraphy. After duplicate removal, the total number of articles obtained was 134. These articles were screened, and narrowed down to 37.

The applied exclusion criteria were: i) not written in English, ii) focusing on denoising/speckle reduction, iii) not considering speckle as a source of information, iv) not focusing on OCT, v) OCT used in plants, vi) not detailing the method used. The number of articles excluded by each criterion is detailed in Figure 2. The remaining 37 articles were then reviewed.

The data extracted from each article were: the implemented method, the OCT technique used, the light-source wavelength (for non-theoretical studies), the biomedical application, and performance metrics related to the application, when provided. This information is reported in Table 1, in Appendix A.

3

![](dt=2026-05-29/ht=23/156f49c151b6005e0e0a28ff86029b96d64181e24669c98cf015e9f29316745e.jpg)

# 3 Results

Figure 3 shows the distribution of the articles included in the review grouped by year. The results show a growing interest in the analysis of signal-carrying speckle in OCT imaging over the last two decades.

Speckle pattern analysis can either provide static or dynamic information about the imaged tissue, depending on whether the scatterers are stationary or in motion between consecutive image acquisitions. In this review, 12 articles were found performing a dynamic analysis, whereas the remaining 25 performed a static analysis. Figure 4 clusters the reviewed articles by technique, depicting the organization of Section 3.

# 3.1 Statistical properties

Local moment based statistical properties of OCT signal intensity have been used for inferring speckle characteristics. These properties have proven to be useful in classification tasks, allowing to discriminate between different types of tissue.

# 3.1.1 Static an
alysis

Roy et al. [11] used the mean $(\mu)$ , standard deviation $(\sigma)$ , kurtosis $(\kappa)$ , skewness $(\nu)$ , and an estimate of optical attenuation and signal confidence measures to detect tissue's susceptibility to rupture using intravascular OCT, with the objective of assessing atherosclerosis. The features proved to have high performance in

4

![](dt=2026-05-29/ht=23/27a0023678bec0a013844bc307ed9b180d2ede7e9c0b5875b343c793dd4e3ae2.jpg)

the identification of such tissues using a Random Forest predictive model (area under the receiver operating characteristics curve of 0.9676).

Wang et al. [12] implemented a model for the detection of soft tissue sarcomas in OCT images of ex vivo human tissues, also based on speckle statistical properties. Specifically, the standard deviation of the signal fluctuations (speckles) of a single axial line (A-scan) was used. The statistical analysis of the features showed they were effective for comparing normal fat tissue and soft tissue sarcoma ( $p$ -value $< 0.01$ , Student's t-test).

# 3.1.2 Dynamics Analysis of Speckle

Ossowski et al. [13] used statistical properties of the OCT speckle to infer the dynamic properties of blood samples. Specifically, they used the mean horizontal and vertical speckle sizes, calculated from intensity data, and the sum of standard deviations of selected windows, calculated from phase data. These three statistical parameters were computed in OCT images of blood samples (Figure 5a and 5b), enabling a visual distinction between the signal modulation from erythrocytes and leukocytes as shown in Figure 5.

# 3.2 Statistical distributions

In Appendix B, the theoretical distributions for speckle complex amplitude (Gaussian distribution) and intensity (exponential distribution) are presented. These distributions have a clear physical meaning, but they are only applicable to an ideal, fully developed, speckle pattern. This may not always be the case in real world applications, where speckle formation could be difficult to model. Therefore, different probability density functions (PDFs) have been proposed to describe the speckle statistics in real world applications of

5

![](dt=2026-05-29/ht=23/72ee3ceb1327ede52985605ef9a4229676ceef2327859030843819568f1baf8f.jpg)

OCT imaging. The parameters of these distributions are expected to change according to the light source properties and dimension/organization of the scatterers in the sample, thus providing information about the tissue properties. Given the different notations and formulations in the literature, a coherent mathematical notation of the proposed models is provided.

This section is organized as follows: in subsection 3.2.1, Rayleigh and K-distribution are presented. These are the two fundamental distributions used to represent fully and non-fully developed speckle patterns, respectively. In subsection 3.2.2, the Gamma and Generalized Gamma distributions are detailed. From these two, the remaining distributions, presented in subsection 3.2.3, can be derived, including Weibull, Nakagami, Rician, three-parameter Rayleigh, and Lognormal. Finally, subsection 3.2.4 details the distributions that have been applied in a dynamic analysis.

# 3.2.1 Fundamental distributions

The Rayleigh distribution is a 1-parameter distribution, used to model fully developed OCT speckle patterns. The application of this model is valid when the signal arises from multiple scatterers within the resolution of the system [14] and the light complex field amplitude is represented by circular Gaussian statistics, i.e., a fully developed speckle pattern [15]. Its PDF is given by:

$$
p _ {R L} (A; a) = \frac {A}{a ^ {2}} e ^ {\left(- \frac {A ^ {2}}{2 a ^ {2}}\right)}, \tag {1}
$$

6

![](dt=2026-05-29/ht=23/ad2ea78f987ad45ce628f4c82b7e7c4e2cd21a488aae807ae57372dd024b67a1.jpg)

![](dt=2026-05-29/ht=23/077bd574b720ada3de53661b4a74c821eeaff7ceb59ab3d6dcf010a6c5b1e597.jpg)

![](dt=2026-05-29/ht=23/2a1d8cc1abd754c80ba3247c5fa309929ac99bd42bbc88b91ed83ad043ccae04.jpg)

![](dt=2026-05-29/ht=23/729b625dacf8549e43ed90db8716e91eb8615af7da054025279575d97e2678f7.jpg)

![](dt=2026-05-29/ht=23/af3dfd79f23de0af2f6d25d5f5c2c6add8e68b682e63c00ad24e9faf88deba37.jpg)

where $a$ is the scale parameter.

Almasian et al. [16] experimentally verified the goodness of fit of the Rayleigh PDF for modelling speckle amplitude using controlled samples of silica microspheres suspended in water. They proved that OCT amplitude distribution for homogeneous samples can be described by a Rayleigh distribution for images with low optical depth (coefficient of determination, $R^2 \approx 0.98$ ). Also, assuming a Rayleigh distribution, expressions were analytically derived for speckle signal mean, amplitude, and variance in terms of sample optical properties.

The K-distribution is a 3-parameter distribution, used to model cases where a small number of scatterers are present in the sample [7], resulting in partially or non-fully developed speckle patterns. Its PDF can be written as:

$$
p _ {K} (A; v, \varphi , L) = \frac {2 \xi^ {(\beta + 1) / 2} A ^ {(\beta - 1) / 2}}{\Gamma (L) \Gamma (\varphi)} K _ {\varphi - L} \left(2 \sqrt {\xi A}\right), \tag {2}
$$

where $\beta = L + \varphi - 1$ , $\xi = L\varphi / \mu$ , $K_{\alpha}$ is a modified Bessel function of the second kind of order $\alpha$ . This distribution is the combination of two gamma distributions, one with mean 1 and shape parameter $\varphi$ and the other with mean $\upsilon$ and shape parameter $\mathrm{L}$ .

The K-distribution was tested by Jesus et al. [7] for corneal OCT speckle intensity characterization against other distributions. However, using a Kolmogorov-Smirnov (KS) goodness of fit with $95\%$ confidence level, K-distribution modelled data presented statistical differences from original raw data, showing that it is not

7

an adequate fit for the analysed problem.

# 3.2.2 Gamma distributions

The Gamma distribution is a 2-parameter distribution. Its PDF belongs to a family of PDFs with two degrees of freedom, and is defined as (3):

$$
p _ {G} (A; a, d) = \frac {A ^ {d - 1} e ^ {- A / a}}{a ^ {d} \Gamma (d)} \text {f o r} a, d > 0, \tag {3}
$$

where $d$ is the shape parameter, $a$ is the scale parameter, and $\Gamma$ represents the Gamma function [17].

Kirillin et al. [18] developed a Monte Carlo model for speckle statistic simulation of OCT data, and validated the model using a phantom. Also, they demonstrated by visual inspection that the Gamma distribution was a good fit for both phantom and the previously simulated data. The scale parameter, $a$ , showed an increase with the increase of scatterers concentration, whereas the shape parameter, $d$ , presented a concentration-independent behavior.

More recently, Niemczyk et al. [19] used the Gamma distribution to model speckle from corneal OCT data. Both Gamma parameters showed a statistically significant relation with intra-ocular pressure (IOP) $(p$ -value $< 0.001$ , ANOVA test).

The Generalized Gamma (GG) distribution is a 3-parameter generalization of the Gamma distribution, with a PDF given by:

$$
p _ {G G} (A; a, d, p) = \frac {p A ^ {d - 1}}{a ^ {d} \Gamma (d / p)} e ^ {- (A / a) ^ {p}} \text {f o r} p > 0, \tag {4}
$$

where $d$ and $p$ are shape parameters, and $a$ , the scale parameter. To obtain the Gamma PDF (Equation (3)), $p$ must be set to 1. Special cases of GG include t
he previously presented Rayleigh (Equation (1)), by setting the parameters to $p_{GG}(A; a\sqrt{2}, 2, 2), a > 0$ .

The GG distribution was used by Jesus et al. [7,20,21] and Iskander et al. [22] to model corneal OCT data. Jesus et al. [7,21] applied the GG distribution to healthy subjects divided in three age groups (24.4±0.5, 31.3±4.6; 61.2±8.4 years), as depicted in Figure 6. The goal was to study variations of the distribution parameters among groups. A significant statistical difference, ( $p$ -value<0.05, Kruskal-Wallis test) was observed for all three parameters.

In a later study [20], Jesus et al. analysed the parameters' relation with micro-structural corneal properties. Significant correlation $(p\text{-value} < 0.001)$ was found between both the scale parameter $(a)$ and the ratio of the shape parameters $(d / p)$ with intraocular pressure (IOP). The authors suggested that GG parameters can contribute to improve IOP measurements.

8

![](dt=2026-05-29/ht=23/e79cddeb1395eefbb7dff66c31a4a6f4dfcc67af7d1259338e277cd5b23797b1.jpg)

Iskander et al. [22], used the GG to model information from the micro-structure of the cornea to differentiate glaucoma suspects, glaucoma patients, and healthy controls. ANOVA tests showed that the scale parameter, $a$ , was correlated with the shape parameter, $p$ , and the relation between these two parameters was statistically significantly different between the three study groups ( $p$ -value $< 0.0001$ , Fisher's Test).

Finally, Seevaratnam et al. [1] used the GG distribution to investigate the effect of temperature variation in tissue phantoms. The scale parameter, $a$ , showed a linear increase with the increase of the tissue temperature. The correlation between $a$ and the temperature was statistically significant ( $p-value = 7.9 \times 10^{-6}$ , Student's t-test).

# 3.2.3 Gamma derived distributions

The 2-parameter Weibull distribution [7], can be obtained from GG distribution for $d = p$ :

$$
p _ {W} (A; a, d) = \frac {d A ^ {d - 1}}{a ^ {d}} e ^ {- (A / a) ^ {d}}. \tag {5}
$$

Jesus et al. tested this distribution to model speckle corneal intensities [7]. No statistically significant difference was observed between the fitted and the raw data ( $p$ -value $< 0.05$ , KS goodness of fit test), concluding that the Weibull distribution can be used to model these data.

The Nakagami distribution is a 2-parameter distribution that can be obtained from the Gamma distribu

9

tion by setting $a = \Omega /d$ and taking the square root of the original random variable, $A^{\prime} = \sqrt{A}$ [23].

$$
p _ {N K} \left(A ^ {\prime}; d, \Omega\right) = \frac {2 d ^ {d}}{\Gamma (d) \Omega^ {d}} A ^ {\prime 2 d - 1} e ^ {- \frac {d}{\Omega} A ^ {\prime 2}}, \tag {6}
$$

where $d$ is a shape parameter and $\Omega$ is a spread parameter.

This distribution has been proposed to represent the dispersion of several backscattered clusters of incoherently added waves [14], and has been tested for modeling skin speckle data against other distributions. It was considered the best fit, using the Kolmogorov-Smirnov test, and its parameters showed statistical differences between two different skin layers (Epidermis and Stratum Corneum).

Nakagami distribution was also tested in corneal data [7,21]. For this case, although it was not considered the best fit, fitted data did not present statistical significant differences from the raw data, also using the KS test.

The Rician, or Rice distribution (Equation (7)) is a 2-parameter generalization of the Rayleigh distribution (Equation (1)), obtained by the introduction of a noncentrality parameter:

$$
p _ {R I} (A; a, \nu) = \frac {A}{a ^ {2}} e ^ {- \frac {A ^ {2} + \nu^ {2}}{2 a ^ {2}}} I _ {0} \left(\frac {A \nu}{a ^ {2}}\right), \tag {7}
$$

where $\nu$ is the noncentrality parameter and $I_0$ is the zero order modified Bessel function of the first kind [24]. Thus, the Rayleigh can be obtained from the Rician distribution for $\nu = 0$ .

The validity of this distribution for modelling speckle amplitude distribution was tested for tissue phantom data by Seevaratnam et al. [1] and for corneal data, by Jesus et al. [7]. While it was not considered the best fit for the tissue phantoms, it was able to model the data successfully. However, for the corneal data, the data modelled with this distribution presented statistically significant difference from raw data (Kolmogorov-Smirnov test for a $95\%$ confidence level).

The 3-parameter Rayleigh distribution is obtained by modifying the Rayleigh (Equation (1)) including two new parameters, $b$ and $c$ :

$$
p _ {3 R L} (A; a, b, c) = \frac {b (A - c)}{a ^ {2}} e ^ {- \frac {- (A - c) ^ {2}}{2 a ^ {2}}} , \tag {8}
$$

with $a$ the scale parameter, $b$ the amplitude normalization parameter, and $c$ the shifting parameter.

Matveev et al. [25] and Demidov et al. [26] used spatial speckle statistics on OCT lymphangiography and neurography to map lymphatic vessels, based on the analysis of the parameters of $p_{3RL}$ . Their experiments, on normal skin and tumor tissues, showed that, by fitting Equation (8) to different regions of interest (ROIs) in an image, the obtained $R^2$ values statistically differed from each other, and could then be used as a feature for nerves and lymphatic vessels mapping. By using a threshold on the $R^2$ value ( $0.9 < R^2 < 0.99$ for the lymph vessels), the authors were able to obtain a discrimination of the tumor from the normal tissue. Following

10

their previous studies, Matveev et al. [27] presented an optimization model to automatically determine the threshold for the $R^2$ value and for the size of the ROI, both parameters previously empirically chosen.

The Lognormal distribution is a 2-parameter distribution of a variable whose logarithm follows a normal distribution, with mean $\nu$ and standard deviation $\sigma$ . Its PDF is given by Equation (9), and can be derived from the GG distribution (Equation (4)) by setting $d / p \to \infty$ :

$$
p _ {L} (A; \mu , \sigma) = \frac {1}{\sigma A \sqrt {2 \pi}} e ^ {- \frac {\left(\log A - v\right) ^ {2}}{2 \sigma^ {2}}} \tag {9}
$$

The Lognormal distribution has been applied by Jesus et al. [7] and Mcheik et al. [14], on corneal and skin speckle data, respectively. Both studies have been further described in Subsection 3.2.2, as the authors compare the Lognormal distribution with the GG distribution in both cases. They also obtain the same conclusion: the GG distribution is a better fit than the Lognormal for corneal and skin speckle data, and data modelled with Lognormal distribution show statistically significant difference from the original data, using the Kolmogorov-Smirnov test with a level of significance of 0.05.

# 3.2.4 Statistical Distribution in Dynamics Analysis of Speckle

The temporal speckle distribution of a single pixel is expected to follow different distributions according to the properties of the sample in that pixel. Therefore, a statistical distribution can be applied to the time-domain histogram of individual pixels to account for speckle dynamics.

Cheng et al. [28] analysed OCT voxels denoting fluid flow for large and small arterioles and venules in phantom and skin data. This analysis was performed as part of a visualization enhancement technique. The authors state that a pixel located within static tissue is expected to follow a Gaussian distribution over time, while pixels located in regions depicting flow will follow a different distribution. In their results, they concluded that Rayleigh distribution (Equation (1)) is suitable to describe the speckle temporal distribution of large arterioles and venules, while the Rician distribution (Equation (7)) can model the amplitude of OCT for tissues that are partially denoting static and flowing scatterers, such as capillaries.

# 3.3 Contrast ratio

The contrast ratio (C) of an OCT image can be de
fined as the ratio of the signal's standard deviation, $(\sigma)$ , and its mean, $(\mu)$ [29],

$$
C = \frac {\sigma}{\mu} \tag {10}
$$

The contrast ratio is expected to vary with the density of scatterers, and has been used for different applica

11

tions, such as segmentation and motion estimation.

# 3.3.1 Static analysis

Hillman et al. [29] and Duncan et al. [30] both proved that it is possible to obtain a correlation between the local contrast statistics and the scatterers density in an OCT sample. Hillman et al. [29] theoretically demonstrated that contrast ratio decreases with the increase of the effective number of scatterers that contribute to the signal. The authors also confirmed empirically their predictions by using an experimental set-up of controlled tissue phantoms of suspensions of microspheres in water with different concentrations. Duncan et al.

[30] computed the local contrast image of simulated synthetic speckle patterns. Then, they estimated the relation between the lognormal distribution (Equation (9)) parameters of this image, and the size of the image kernel. Experiments using chick embryo OCT images were also performed. Vessels and background were successfully segmented by choosing an adequate threshold of the lognormal PDF parameters computed in the contrast image.

# 3.3.2 Dynamics analysis of speckle

Kirkpatrick et al. [31] developed an approach to quantify the shift and temporal contrast in a translating speckle pattern. The end goal of the authors was to quantify local motion. Their proposed method, quantitative temporal speckle contrast imaging, is dependent on the speckle size in the image and the number of images in a sequence. The application of the method is also depends on the acquisition speed of the device, which should be fast enough that no motion occurs during the image acquisition.

# 3.4 Logarithmic pixel intensity contrasts

OCT images are often displayed in a logarithmic scale to enhance dynamic ranges. This transformation causes some properties to vary from the OCT data that has not been logarithmically transformed.

# 3.4.1 Static analysis

Considering a single image where amplitudes follow the Rayleigh distribution (Equation (1)), its logarithmic transformation will result in a contrast ratio defined in Equation (10), a function inversely dependent on the logarithmically transformed mean intensity signal of the signal intensity [15]. In these conditions, the speckle contrast is equivalent to:

$$
C = \frac {\pi}{\sqrt {6} (D _ {I} - \gamma)}, \tag {11}
$$

12

where $\gamma$ is the Euler-Mascheroni constant, and $D_I$ is the logarithmically transformed mean intensity signal. In an OCT acquisition, the light beam propagates through the tissue from top to bottom, which causes its intensity to decrease with the increase of the depth. Then, it is expected that speckle contrast of the logarithmically transformed OCT signal will also be dependent on depth. However, the level of signal attenuation will also depend on the scattering properties of the tissue.

Lee et al. [15] used simulated OCT images and in vitro rat liver images to explore these properties, and demonstrate the validity of the relation in Equation (11). Moreover, they showed that the contrast value changes according to the scattering coefficient of the sample, which can be used to characterize tissues.

# 3.4.2 Dynamics analysis of speckle

The logarithmic intensity variance (LOGIV) is defined as the variance of the intensity image after the logarithmic transformation. The differential logarithmic intensity variance (DLOGIV), is calculated by multiplying the LOGIV by a factor of two. LOGIV and DLOGIV values approach $\pi^2 /6$ and $\pi^2 /3$ , respectively, when the signal-to-noise ratio (SNR) approaches zero. These values can be used to compute LOGIV and DLOGIV tomograms by collecting multiple scans from the same region and measuring the quantitative variance of logarithmic intensities over scans.

Motaghiannezam et al. [32] proposed this technique for the analysis of in vivo human retinal vasculature visualization. Results of their experiments showed that static areas of the retina were invisible in the LOGIV and DLOGIV tomograms, while areas with detectable motion, such as blood vessels, were not. They also confirmed the low sensitivity of LOGIV and DLOGIV to the sample reflective strength, demonstrating the superiority of these methods in comparison to linear contrast ratios for detecting motion and visualizing microvasculature.

# 3.5 Spatial Gray-Level Dependence Matrices (SGLDM)

Spatial gray-level dependence matrices (SGLDM), also referred to as co-occurrence matrices [2, 10, 33, 34], are determined by the estimation of the second-order joint-probability distribution of each combination of grey-level values that occur next to each other, averaged over directions of $0^{\circ}$ , $45^{\circ}$ , $90^{\circ}$ , and $135^{\circ}$ . Assuming the images are normalized with $L$ grey scale levels, each $f(i,j \mid d,\theta)$ is the probability of a pixel with level $i$ being at a distance $d$ from a pixel with gray level of $j$ in the $\theta$ direction. Following this principle, an $L \times L$ matrix can be created for each direction and for each chosen distance, as is shown in Figure 7.

Several features can be extracted from these matrices, including energy, entropy, correlation, local homogeneity, and inertia, detailed in [2,33,34]. Similarly to the statistical properties described in subsection 3.1, these features are expected to be linked to changes in tissue. Thus, they can be used for classification tasks.

13

![](dt=2026-05-29/ht=23/78d32329ccfc787cb5c563483918bb38ba7253a8274f4174ea5e05cb66233848.jpg)

![](dt=2026-05-29/ht=23/b6b04a07b49a46106b67dbe407976785ad5951485fb7a6fa57feb8ddf6cfe8fa.jpg)

Gossage et al. [2] proposed and applied SGLDM features for differentiating mouse skin and fat, and normal versus abnormal mouse lung tissue, using a minimum-error-rate Bayesian model. Their results showed a high accuracy classification of mouse skin and fat, of 98.5 and $97.3\%$ , respectively. A satisfactory performance was also obtained for distinguishing normal and abnormal mouse lung, of 64.0 and $88.6\%$ , respectively. Finally, the features were used to classify five different bovine tissues, with a similar classification model, resulting in an average of $80\%$ correct classification rate.

Kasaragod et al. [10] used SGLDM to retrieve information from the speckle OCT images. A Bayesian model was applied to the classification of tissue phantoms with different amount of scatterers, and to identify the invasion of melanoma cell into tissue engineered skin. Their results were satisfactory in classifying the number of scatterers in the tissue phantoms, shown visually by a ROC curve plot. However, this approach provided limited results in the identification of the melanoma cells in the tissue.

14

# 3.6 Frequency Domain methods

After computing the 2D Discrete Fourier Transform (DFT) in an OCT image, the resulting image can be divided into regions, according to their frequency content. This will result in different texture parameters per region. The contribution of each region to the total frequency magnitude is calculated by summing all the values of the spatial frequencies in that region, and dividing by the total frequency magnitude of the image. This value represents the percentage of signal within a certain range of spatial frequencies, and it can be used as a feature in similar applications as described in 3.1 and 3.4.

Gossage et al. [2, 33] used the 2D-DFT for retrieving information from OCT images, together with the previously described SGLDM (subsection 3.5). The goal of the work was analysing and classifying texture of different tissues (mouse lung and bovine
tissue). The results detailed in section 3.5 were obtained with a combination of 2D-DFT and SGLDM features. A similar approach was used in [34], where the 2D-DFT and SGLDM features were used to differentiate living from non-living tissue phantoms with various sizes and distributions of scatterers.

# 3.7 Tissue dispersion

Tissue dispersion can be measured in a static image analysis from the degradation of the image Point Spread Function (PSF). The standard method to measure tissue dispersion is through the computation of the resolution degradation of single reflections. However, such a measurement requires distinct point reflectors, below and outside the sample, which rarely happens in vivo.

As alternative, Photiou et al. [35,36] proposed to estimate the tissue dispersion from the imaged speckle. Being a coherent phenomenon, speckle is affected by tissue dispersion. Changes in speckle size can be used to estimate the broadening of the image PSF, and later to calculate the group velocity dispersion (GVD). This method is based on the comparison of small regions of an OCT image at different depths without visible structures. These regions should only contain speckle information and it is expected that a region from the surface shows no dispersion, as opposed to deeper regions.

Photiou et al. [35,36] showed that their proposed method performs similarly to the standard procedure (from degradation of the image PSF), with a GVD difference less than $7\%$ . Also, this value proved to be a good feature for tissue classification, when comparing normal to malignant samples of human colon (accuracy of $96\%$ using linear discriminant analysis).

# 3.8 Speckle correlation

Considering a sequence of scans over time, the autocorrelation and/or decorrelation of intensities can carry information about motion of particles in a sample. Both features can be used as an approach for time-varying speckle analysis [9].

15

In terms of speckle analysis, the normalized autocorrelation, for a given point $p$ and time lag $\tau$ , is given by:

$$
g _ {p, \tau} = \frac {\sum_ {t = 0} ^ {N - 1} \left(I _ {p , t} - \langle I _ {p} \rangle\right) \left(I _ {p , t + \tau} - \langle I _ {p} \rangle\right)}{\sum_ {t = 0} ^ {N - 1} \left(I _ {p , t} - \langle I _ {p} \rangle\right) ^ {2}}, \tag {12}
$$

where $I_{p,t}$ is the intensity in the point $p$ at time $t$ , $\tau$ is the period of time between scans, and $N$ is the total number of scans considered. The temporal average intensity, $\langle I_p\rangle$ , is subtracted from each intensity value in order to consider only the intensity fluctuations. The autocorrelation of a speckle pattern with itself $(\tau = 0)$ is expected to be maximized. With the increase of the lag between scans, this value is expected to decrease until it reaches 0, when the scans are no longer correlated.

A vector of autocorrelation values can be obtained by changing $\tau$ . These autocorrelation values are related with the fluctuations in the intensity of the speckle patterns. At the same time, the fluctuations are related with the flow of particles in the tissue, which in turn are expected to be related with the flow of particles in the tissue [9].

The decorrelation time $\tau_{c}$ is used to analyse the shape of each autocorrelation function [9]. This metric corresponds to the time it takes for the autocorrelation value to fall to $1 / e$ .

De Pretto et al. [9] performed experiments with milk pumped through a microchannel at different velocities, and proved the inversely proportional relation between decorrelation time and flow velocity. As expected, this relation is highly dependent on the sampling frequency. A sampling rate of $8\mathrm{kHz}$ , makes the system appropriate for differentiating between low flow rates, up to $12\mu l / min$ . However, if the flow rates are higher, the temporal resolution of the system makes it unsuitable.

In a later study, De Pretto et al. [37] implemented the same approach to monitor blood sugar in OCT data. They used samples of heparinized mouse blood, phosphate buffer saline, and different concentration of glucose. They were able to differentiate between low level of glucose concentration, up to $355\mathrm{mg / dL}$ , indicating the suitability of OCT for non-invasive measurements of glucose levels.

Popov et al. [38] conducted an experiment using tissue phantoms. They obtained an expression for the spatio-temporal correlation function of scattered radiation, assuming a single scatterer regime, and were able to accurately measure viscosity.

Farhat et al. [39] assessed the changes that occur in intracellular motion as cells undergo apoptosis. To that end, they induced apoptosis in samples of acute myeloid leukemia cells, and they measured the decorrelation time of the speckle over a period of 48h. Their results showed an increase of motion in the cells (identified as a decrease of the decorrelation time of speckle) after 24h, which is in accordance with histology.

Ferris et al. [40] used phantom data to study the effects of multiple scattering on the speckle decorrelation. Their conclusions confirm that speckle decorrelation is dependent on parameters such as the concentration and size of particles, and velocity field inhomogeneities. They also concluded that an overestimation of blood flow velocities might occur because an increase in the rate of decorrelation is caused by the detection of

16

forward scattered light.

Uribe-Patarroyo et al. [41] proposed a new discrete normalized second-order autocorrelation.

The authors claim this approach proves to be more robust to the presence of noise, and can be used as a method for speed measurements in tissue phantoms. This same method was used in a later work [42] for the correction of the rotation distortion in catheter-based endoscopic OCT.

Finally, Liu et al. [43] used the cross-correlation coefficient between adjacent A-scans to analyse properties of simulated speckle images. Their results underline the importance of over-sampling when calculating motion properties from temporally dynamic speckle, such as the contrast or decorrelation time, due to its random nature.

# 4 Discussion

The reviewed approaches can be categorized as static or dynamic, depending on the type of data that is used, with some methods being applicable to both categories. In a static analysis, the inherent speckle pattern in a single image is studied, in order to obtain information about the micro-structures in the sample. In a dynamic analysis, the changes in speckle between images are analyzed to gain information on the dynamic nature of the moving scatterers [8,44,45].

In most of the reviewed works, the authors aimed to further understand the physical meaning of the light speckle, and how to model it mathematically [3, 15, 16, 18, 30, 31, 40, 43, 46]. Although its interpretability remains a challenging task, a number of works have shown the feasibility of speckle-derived quantifications for biomedical imaging related tasks, including classification (e.g. healthy/pathological) [1, 2, 7, 10-13, 19-22, 25-28, 33-36], segmentation (e.g. vessels) [14], or motion quantification (when dynamic data is provided) [9, 39-43]. Nevertheless, some aspects and limitations of those techniques should be discussed in order to improve future works.

Among all the reviewed methodologies, speckle modeling using statistical distributions has been the most studied, especially on static data. Several authors have proven the applicability of these approaches to different tasks, mainly to classify between different tissues. However, the conclusions on the optimal PDF often vary depending on the analysed tissue, the application, and the OCT device. For example, Mcheik et al. [14] and De Jesus et al.

[7] both presented a comparative study including several PDFs, but in different applications, one to differentiate speckle from different skin layers, and the second from different groups in corneal data. Des
pite both works including Nakagami, GG, and Lognormal distributions, their conclusions were different. For skin layers, the Nakagami distribution was found to be the best. However, for corneal data, the best fit was achieved by the GG distribution. Furthermore, other authors have successfully applied other distributions to the same application, such as Niemczyk et al.

[19], who used the Gamma distribution to study the effect of IOP on corneal OCT speckle from porcine eyes. This precludes drawing conclusions on

17

the PDF which provides the best fit for each application, since in most of them, exhaustive comparisons do not exist.

A drawback that the modeling of speckle through statistical distributions had to tackle is that the real world problems do not fulfill the theoretical assumptions of speckle formation. Theoretically speaking, when the number of elementary phasors is high, meaning a high number of scatterers per coherence volume, the central limit theorem is fulfilled [15], and the speckle pattern is fully developed, and its intensity distribution follows the Rayleigh distribution. However, this argument is only partially applicable to biological tissues, because different tissues may have different natures, i.

e. some can be more heterogeneous with a lower number of scatterers [26]. As a consequence, several authors propose more complex distributions (three parameters and higher), which presents new potential challenges. When a statistical model is used to represent the process that generated the data, the representation will not be fully accurate, as some information will be lost. In estimating the amount of information lost by a model, one needs to take into account the trade-off between the goodness of fit and the simplicity of the model, i.e. the risk of overfitting and underfitting.

In the corneal studies [7,21,22], the Akaike's Information Criterion is applied to minimize this risk.

Following the analysis of statistical distributions, the second most used technique is the analysis of the correlation, which is only applicable to dynamic data. The main application of this technique is in motion determination (e.g. blood flow). Although most of the reviewed works in OCT speckle are focused on theoretical modeling or validation with phantoms, this is in fact a type of signal analysis that has been widely explored in other imaging modalities.

Some examples are laser speckle imaging, which can be used for cutaneous blood flow determination [8, 44], blood pulse pressure waveform estimation [47, 48], cellular assessment in muscle tissue [49], and laser speckle flowgraphy which can be used for ocular blood flow determination [45]. Furthermore, the rationale behind these techniques has also been widely applied to compute OCT angiography (OCTA) from a set of temporal OCT data acquired at the exact same position [50].

In contrast with the correlation analysis, there have been little to none applications of the other methods for dynamic data. However, some authors have demonstrated the feasibility of statistical properties [13], statistical distributions [28], or logarithmic intensity contrast [32] to visualize and segment blood vessels within a tissue.

Finally, the methods that compute characteristics of static data (tissue dispersion, SGLDM, frequency domain methods, or the previously mentioned statistical properties/distributions and contrast (both with and without the application of the logarithmic transform)), are more applied to obtain features to use in a predictive model or classifier. While these approaches may under-perform classification models that take into account the complete image information or several features, many of these proposed features are very easy to interpret, and to link to the physical changes in the tissue, making them interesting for clinical practice.

Although a growing interest in the analysis of signal-carrying speckle has been observed over the last years, it is still a research line in an early stage. A considerable number of different methods have already been

18

proposed but only a few applications published in the literature. At the current stage, the analysis of OCT speckle is lacking on validation and information on its reproducibility in-vivo. None of the reviewed studies validated the proposed methods on large (the largest dataset in the studies included was 65 [7,21]) or multiple datasets. The speckle information is intrinsically related to the spatial arrangement and biomechanical properties of the scatterers in the sample.

Scatterers can either be collagen fibers and fibroblasts when imaging the cornea, blood cells flowing through vessels, or just silica particles in a phantom image. What is considered a scatterer in a sample will depend on the characteristics of the imaging system, namely the relationship between the particle size and the light source wavelength [51]. This is particularly important for OCT imaging, given the variability existing between devices (790-1330nm), as it can be observed in Table 1, Appendix A.

Consequently, for the same sample and method, different quantitative values for speckle may be obtained depending on the OCT system used.

Another important aspect that hampers the development of speckle-based techniques is the limited access to raw data. Images collected from commercial OCT devices are often filtered to reduce the speckle or transformed to increase visualization contrast. It is of utmost importance that raw OCT images are used in speckle studies, otherwise the obtained results are tainted by the used device and pre-processing algorithm [52]. If raw data is not available, the information of the applied image processing algorithms should be provided to understand how the speckle has been processed and hence, comprehend its physical meaning in a biomedical application.

Despite its early stage, research on methods to study the signal-carrying speckle has been a step forward on the comprehension of the physical meaning of the information retrieved from OCT imaging. Speckle analysis provides information on the size and distribution of the scatters that has not been considered in a clinical practice yet. Such advancements are also particularly interesting for other research lines such as OCT elastography, adaptive optics imaging, or machine learning applications.

For example, recent developments on machine learning, namely on convolutional neural networks have reported outperforming results in OCT image analysis in comparison to conventional image processing [53-55]. However, deep learning approaches still lack on interpretability and roughly remain a black box, despite the recent efforts to address this limitation [56]. Therefore, future research may focus on integrating physics and learning based approaches, to combine their strengths.

# 5 Conclusion

This paper presents an overview of the current state of the art in OCT signal-carrying speckle analysis in biomedical applications. The results of this literature review show that several methods have already been proposed for different applications, highlighting the potential of speckle analysis to infer the optical and spatial properties of the scatterers in a sample or tissue. However, signal-carrying speckle analysis in OCT is

19

still in its early stage and further work is needed to validate its applicability and reproducibility in a clinical context.

# Acknowledgments

The authors thank Prof. Dr. D. Robert Iskander and Dr. Monika E. Danielewska for their technical revision of the paper, and respective contributions. The authors thank also Prof. Dr. Ingeborg Stalmans, Dr. João Breda and Dr. Jan Van Eijgen for their clinical advisory and support.

# Disclosures

The authors have no relevant financial interests in this article and no potential conflicts of interest to disclose.

# Funding

This work was supported by the Horizon 2020 Research and Innovation Programme (grant agreement no. 780989: Multi-
modal, multi-scale retinal imaging project) and by Portuguese National Funds through the FCT, Fundação Para a Ciência e a Tecnologia, I.P., in the scope of the project UIDB/04559/2020.

# References

20

21

22

23

24

25

# Appendix A Articles Details

Table 1: Characteristics of the reviewed studies

![](dt=2026-05-29/ht=23/c9ab0c583d2d8c418d102b43dbf15e4ac8906c6d22c5ed4dfe11ab0c06830ee0.jpg)

<table><tr><td>Authors &amp; Publication year</td><td>Static/ Dynamic</td><td>Method</td><td>Aim</td><td>Application/ Data used</td><td>OCT technique (brand)</td><td>Light wavelength (nm)</td></tr><tr><td>Wang et al. [12] (2013)</td><td>Static</td><td>Statistical properties</td><td>Classification</td><td>Ex vivo human tissue</td><td>SS-OCT (custom made)</td><td>1310</td></tr><tr><td>Ossowski et al. [13] (2015)</td><td>Dynamic</td><td>Statistical properties</td><td>Classification</td><td>Blood</td><td>SD-OCT (custom made)</td><td>790</td></tr><tr><td>Roy et al. [11] (2015)</td><td>Static</td><td>Statistical properties</td><td>Classification</td><td>Coronary artery</td><td>SD-OCT (CV-M2, LightLab Imaging Inc)</td><td>1320</td></tr><tr><td>Schmitt et al. [3] (1999)</td><td>Static</td><td>Statistical distributions</td><td>Theoretical modeling</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Karamata et al. [46] (2005)</td><td>Static</td><td>Statistical distributions</td><td>Theoretical modeling</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Mcheik et al. [14] (2008)</td><td>Static</td><td>Statistical distributions</td><td>Segmentation</td><td>Skin</td><td>SD-OCT (SkinDex 300, ISIS)</td><td>1300</td></tr><tr><td>Kirillin et al. [18] (2014)</td><td>Static</td><td>Statistical distributions</td><td>Theoretical modeling</td><td>Tissue phantoms (polystyrene microspheres)</td><td>SS-OCT (custom made)</td><td>1310</td></tr><tr><td>Seevaratnam et al. [1] (2014)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Tissue phantoms (polystyrene microspheres)</td><td>SS-OCT (Biophotonics and Bioengineering Laboratory&#x27;s)</td><td>1310</td></tr><tr><td>Jesus et al. [21] (2015)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Cornea</td><td>SD-OCT (Copernicus HR)</td><td>850</td></tr><tr><td>Almasian et al. [16] (2017)</td><td>Static</td><td>Statistical distributions</td><td>Theoretical modeling</td><td>Tissue phantoms (silica microspheres)</td><td>SS-OCT (Santec IVS 2000)</td><td>1309</td></tr><tr><td>Jesus et al. [7] (2017)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Cornea</td><td>SD-OCT (IOLMaster 700)</td><td>850</td></tr><tr><td>Jesus et al. [20] (2017)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Cornea</td><td>SD-OCT (Copernicus HR)</td><td>851</td></tr><tr><td>Demidov et al. [26] (2019)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Mice (skin)</td><td>SS-OCT (custom made)</td><td>1320</td></tr><tr><td>Matveev et al. [25] (2019)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Mice (skin)</td><td>SS-OCT (custom made)</td><td>1320</td></tr></table>

26

Table 1: Characteristics of the reviewed studies

![](dt=2026-05-29/ht=23/6d9638b8b5e4d0544a98ec138a9deebf717fcd59f2e30f70fc51163661e39b14.jpg)

<table><tr><td>Authors &amp; Publication year</td><td>Static/ Dynamic</td><td>Method</td><td>Aim</td><td>Application/ Data used</td><td>OCT technique (brand)</td><td>Light wavelength (nm)</td></tr><tr><td>Matveev et al. [27] (2019)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Iskander et al. [22] (2020)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Cornea</td><td>SD-OCT (HRT 3, Heidelberg Engineering GmbH)</td><td>850</td></tr><tr><td>Cheng et al. [28] (2014)</td><td>Dynamic</td><td>Statistical distributions</td><td>Classification</td><td>Phantom: agrose and titanium dioxide / Skin</td><td>SS-OCT (Thorlabs Inc.)</td><td>1300</td></tr><tr><td>Niemczyk et al. [19] (2021)</td><td>Static</td><td>Statistical distributions</td><td>Classification</td><td>Cornea (porcine eyes)</td><td>SD-OCT (Copernicus REVO)</td><td>830</td></tr><tr><td>Photiou et al. [35] (2017)</td><td>Static</td><td>Tissue dispersion</td><td>Classification</td><td>Porcine muscle / Adipose tissues / Colon</td><td>SS-OCT (custom made)</td><td>-</td></tr><tr><td>Photiou et al. [36] (2017)</td><td>Static</td><td>Tissue dispersion</td><td>Classification</td><td>Porcine muscle / Adipose tissues / Colon</td><td>SS-OCT (custom made)</td><td>1300</td></tr><tr><td>Kasaragod et al. [10] (2010)</td><td>Static</td><td>SGLDM</td><td>Classification</td><td>Tissue phantoms (agar intralipid solution) / Tissue engineered (skin)</td><td>SS-OCT (custom made)</td><td>1315</td></tr><tr><td>Gossage et al. [2] (2003)</td><td>Static</td><td>SGLDM/Frequency Domain methods</td><td>Classification</td><td>Mouse lung</td><td>SS-OCT (custom made)</td><td>1300</td></tr><tr><td>Gossage et al. [33] (2003)</td><td>Static</td><td>SGLDM/Frequency Domain methods</td><td>Classification</td><td>Mouse lung / Bovine tissues</td><td>SS-OCT (custom made)</td><td>1300</td></tr><tr><td>Gossage et al. [34] (2006)</td><td>Static</td><td>SGLDM/Frequency Domain methods</td><td>Classification</td><td>Tissue phantoms (silica microspheres) / Bovine aorta endothelial cells</td><td>SS-OCT (custom made)</td><td>1300</td></tr><tr><td>Hillman et al. [29] (2006)</td><td>Static</td><td>Contrast ratio</td><td>Theoretical modeling</td><td>Tissue phantoms (polystyrene microspheres)</td><td>SD-OCT (custom made)</td><td>1330</td></tr><tr><td>Kirkpatrick et al. [31] (2007)</td><td>Dynamic</td><td>Contrast ratio</td><td>Theoretical modeling / Motion determination</td><td>Engineered tissue</td><td>SD-OCT (custom made)</td><td>843</td></tr><tr><td>Duncan et al. [30] (2008)</td><td>Static</td><td>Contrast ratio</td><td>Theoretical modeling / Segmentation</td><td>Embryonic chick heart</td><td>-</td><td>-</td></tr></table>

27

Table 1: Characteristics of the reviewed studies

![](dt=2026-05-29/ht=23/267213691a1051495d460710a27692302df49830f8cec402a52d92cb74c552e5.jpg)

<table><tr><td>Authors &amp; Publication year</td><td>Static/ Dynamic</td><td>Method</td><td>Aim</td><td>Application/ Data used</td><td>OCT technique (brand)</td><td>Light wavelength (nm)</td></tr><tr><td>Lee et al. [15] (2011)</td><td>Static</td><td>Logarithmic intensity contrasts</td><td>Theoretical modeling</td><td>Rat liver / Tissue phantoms</td><td>SD-OCT (custom made)</td><td>834</td></tr><tr><td>Motaghiannezam et al. [32] (2012)</td><td>Dynamic</td><td>Logarithmic intensity contrasts</td><td>Visualization</td><td>Retina</td><td>SS-OCT (custom made)</td><td>1060</td></tr><tr><td>Farhat et al. [39] (2011)</td><td>Dynamic</td><td>Speckle correlation</td><td>Motion determination</td><td>Acute myeloid leukemia cells</td><td>SS-OCT (Thorlabs Inc.)</td><td>1300</td></tr><tr><td>Liu et al. [43] (2013)</td><td>Dynamic</td><td>Speckle correlation</td><td>Motion determination</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Uribe-Patarroyo et al. [41] (2014)</td><td>Dynamic</td><td>Speckle correlation</td><td>Motion determination</td><td>Tissue phantoms (intralipid)</td><td>SS-OCT (custom made)</td><td>1285</td></tr><tr><td>De Pretto et al. [9] (2015)</td><td>Dynamic</td><td>Speckle correlation</td><td>Motion determination</td><td>Milk flow</td><td>SS-OCT (Thorlabs Inc.)</td><td>1325</td></tr><tr><td>Uribe-Patarroyo et al. [42] (2015)</td><td>Dynamic</td><td>Speckle correlation</td><td>Motion determination</td><td>Endoscopic (esophagus)</td><td>SD-OCT (NvisionVLE)</td><td>1310</td></tr>
<tr><td>De Pretto et al. [37] (2016)</td><td>Dynamic</td><td>Speckle correlation</td><td>Viscosity determination</td><td>Mice blood</td><td>SR-OCT (Thorlabs Inc./ SS-OCT (custom made)</td><td>930 /1325</td></tr><tr><td>Popov et al. [38] (2017)</td><td>Dynamic</td><td>Speckle correlation</td><td>Viscosity determination</td><td>Tissue phantoms</td><td>SD-OCT (custom made)</td><td>1313</td></tr><tr><td>Ferris et al. [40] (2020)</td><td>Dynamic</td><td>Speckle correlation</td><td>Motion determination</td><td>Tissue phantoms</td><td>SD-OCT (custom made)</td><td>1290 /1310</td></tr></table>

SGLDM = Spatial Gray Level Dependence Matrices; SD = Spectral Domain; SS = Swept Source; SR = Spectral Radar; NvisionVLE = NvisionVLE Imaging System (NinePoint Medical, Inc., Bedford, MA); Thorlabs Inc. = Thorlabs Inc. (Newton, NJ); IOLMaster 700 = IOLMaster 700 (Carl Zeiss Meditec AG, Germany); CV-M2, LightLab Imaging Inc. = CV-M2, LightLab Imaging Inc. (Westford, MA, USA); Copernicus HR = Copernicus HR (Optopol, Zawiercie, Poland); HRT 3, Heidelberg = HRT 3, Heidelberg Engineering GmbH (Heidelberg, Germany); Copernicus REVO = Copernicus REVO, (Optopol, Zawiercie, Poland).

28

# Appendix B Speckle Theory

As detailed in the manuscript, one of the main speckle analysis techniques is based on the determination of the speckle pattern intensity probability density function (PDF), since photodetectors measure light intensity and not complex amplitude. A short theoretical introduction on the speckle effect is mandatory to understand why the negative exponential function was historically the first one used to describe the speckle statistics, and to understand their limitations.

Light speckle is often modelled using the statistical perspective defined by J. W. Goodman [57]. To deduce the PDFs of the speckle signal amplitude (Gaussian) and its respective intensity (negative exponential) we shall follow Goodman's approach.

Assuming a monochromatic and perfectly polarized light source, for a given temporal instant, we can define the complex amplitude of the electrical field $a$ as [58]:

$$
a (x, y, z) = a (x, y, z) e ^ {i \theta (x, y, z)}, \tag {13}
$$

where $\mathrm{a(x,y,z)}$ is the amplitude and $\theta$ the phase. Regardless of the detector spatial position, the amplitude of the electrical field that reaches the detector $A$ corresponds to a sum of $N$ de-phased electrical fields coming from different regions of the tissue:

$$
A (x, y, z) = \frac {1}{\sqrt {N}} \sum_ {k = 1} ^ {N} a _ {k} e ^ {i \theta_ {k}}. \tag {14}
$$

where $a_{k}$ and $\theta_{k}, k = 1,2,\ldots N$ are the amplitudes and phases forming that field. In order to determine the PDF of the complex amplitude, two assumptions related to the physical mechanisms of speckle are made. First, the amplitude and phase of each phasor are statistically independent of each other. Second, the phases $\theta_{k}$ are uniformly distributed between $-\pi$ and $\pi$ . In terms of physical significance, these assumptions imply that each scattering volume is independent and that the reflection boundary irregularities are larger than the light wavelength.

By splitting the complex amplitude in its real and imaginary parts, it can be shown that both have zero mean and identical variances [57]. When the number of summed phasors is very large ( $N \to \infty$ ) the PDF of the real and imaginary part are asymptotically Gaussian as well as their joint PDF:

$$
p \left(A _ {R e}, A _ {I m}\right) = p \left(A _ {R e}\right) \cdot p \left(A _ {I m}\right) = \frac {1}{\sigma^ {2} 2 \pi} e ^ {- \frac {A _ {R e} ^ {2} + A _ {I m} ^ {2}}{2 \sigma^ {2}}}, \tag {15}
$$

where $A_{Re}$ corresponds to the real part and $A_{Im}$ to the imaginary part of $A(x,y,z)$ , and the variance $\sigma^2$ is

29

defined by:

$$
\sigma^ {2} = \lim  _ {N \rightarrow \infty} \frac {1}{N} \sum_ {k = 1} ^ {N} \frac {\left\langle a _ {k} ^ {2} \right\rangle}{2}. \tag {16}
$$

The PDF of the light intensity can also be deduced from equation (15). By definition, the light intensity $I$ and the phase $\theta$ are expressed by:

$$
I = | A (x, y, z) | ^ {2} = A _ {R e} ^ {2} + A _ {I m} ^ {2}, \tag {17}
$$

$$
\theta = \tan^ {- 1} \frac {A _ {I m}}{A _ {R e}}. \tag {18}
$$

The relation between the intensity PDF and the amplitude PDF can be found by applying random variables transformations:

$$
p (I, \theta) = p \left(A _ {R e}, A _ {I m}\right) | | J | | = \frac {1}{\sigma^ {2} 4 \pi} e ^ {- \frac {I}{2 \sigma^ {2}}}, \tag {19}
$$

where $||J||$ is the Jacobian matrix. Recalling the assumptions of independence between intensity and phase, the marginal PDF of the intensity is found using:

$$
p (I) = \int_ {- \pi} ^ {\pi} p (I, \theta) d \theta = \frac {1}{2 \sigma^ {2}} e ^ {- \frac {I}{2 \sigma^ {2}}}. \tag {20}
$$

By computing the first and second order moments of the amplitude, the field amplitude variance can be expressed by [58]:

$$
\sigma^ {2} = \langle I \rangle / 2, \tag {21}
$$

resulting in a negative exponential PDF which is characteristic of a fully developed speckle pattern in perfect conditions [3]:

$$
p (I) = \frac {1}{\langle I \rangle} e ^ {- \frac {I}{\langle I \rangle}}. \tag {22}
$$

30