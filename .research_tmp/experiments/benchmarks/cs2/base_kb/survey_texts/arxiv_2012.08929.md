Learning-Based Algorithms for Vessel Tracking: A Review 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2012.08929v1 [eess.IV] 16 Dec 2020 
 
 

# Learning-Based Algorithms for Vessel Tracking: A Review

 Journal:  Computerized Medical Imaging and Graphics 
 
 
 Dengqiang Jia
 
 Address:  School of Naval Architecture, Ocean and Civil Engineering, Shanghai Jiao Tong University, Shanghai, China
 
    
 Xiahai Zhuang
 
 Email:  zxh@fudan.edu.cn 
 
 Corresponding author:  Corresponding author
 
 Address:  School of Data Science, Fudan University, Shanghai, China
 

 Abstract 
 
 Developing efficient vessel-tracking algorithms is crucial for imaging-based diagnosis and treatment of vascular diseases.
Vessel tracking aims to solve recognition problems such as key (seed) point detection, centerline extraction, and vascular segmentation. Extensive image-processing techniques have been developed to overcome the problems of vessel tracking that are mainly attributed to the complex morphologies of vessels and image characteristics of angiography.
This paper presents a literature review on vessel-tracking methods, focusing on machine-learning-based methods.
First, the conventional machine-learning-based algorithms are reviewed, and then, a general survey of deep-learning-based frameworks is provided.
On the basis of the reviewed methods, the evaluation issues are introduced.
The paper is concluded with discussions about the remaining exigencies and future research.

 
 
 
 Keywords:  
Vessel tracking, Learning-based algorithms, Review

 
 

## 1 Introduction

 
 Blood vessels, spread throughout the human body, constitute a significant part of the circulatory system.
All body tissues rely on the normal functioning of different vessels such as cerebral arteries, retinal vessels, carotid arteries, pulmonary arteries, and coronaries.
Any abnormal change in or damage to the vessels will be manifested as diseases at different levels (e.g., stroke, arteriosclerosis, cardiovascular diseases, and hypertension).
Medical imaging and image analysis enable novel technologies and applications for better diagnosis and treatment of blood-vessel diseases.
Tracking the target vessels from the wide field of view of medical images is a prerequisite for the localization and identification of abnormal vessels or regions of interest.
However, manual annotation, which usually demands expertise, is particularly time-consuming and tedious.

 
 
 Vessel tracking aims to solve the problems encountered in vessel image analysis, including key-point (seed point) detection, centerline extraction, and vascular segmentation. These problems considerably differ because of the wide array of vessel anatomies and image characteristics. Accordingly, the problematic factors are categorized into two groups: those related to vessel morphologies (e.g., small size, branching pattern, tortuosity, and severe stenosis of vessels) and image characteristics (e.g., low contrast, noise, artifacts, dislocation, and adjacent hyper-intense structures).

 
 
 Localizing the key points and recognizing the key patterns of vascular structures are fundamental to perform vessel tracking; building models based on various assumptions on vascular appearances (i.e., the prior knowledge and intrinsic features) is also important.
Exploring such problems facilitates the development of new algorithms, particularly in learning-based tracking methods.

 
 
 In the literature, several articles focus on the survey of vessel-tracking methods.
To the best of our knowledge, Suri et al. (2002) published the first survey on this topic.
They concentrated on skeleton and indirect techniques for vascular segmentation.
In addition to vessel-tracking methods, Kirbas and Francis (2004) particularly reviewed the methods of detecting similar vessel characteristics, such as neurovascular and tubular structures.
The review of Lesage et al. (2009) further focused on lumen segmentations.
They categorized the methodologies according to three aspects (i.e., models, features, and extraction schemes) and provided general considerations for each aspects.

 
 
 Recently, more survey papers for tracking the vessels of certain organs in specific imaging modality, e.g., cerebral vessel segmentation from magnetic resonance (MR) images ( Klepaczko et al., 2016 ) , coronary reconstruction from X-ray angiography ( Çimen et al., 2016 ) , and lung vessel detection from computed tomography angiography (CT) ( Rudyanto et al., 2014 ) , have been published.
Reviews on retinal vessel segmentation were presented in the work ( Abramoff et al., 2010 ; Fraz et al., 2012b ; Singh and Kaur, 2015 ; Pohankar and Wankhade, 2016 ; Mansour, 2017 ; L Srinidhi et al., 2017 ; Vostatek et al., 2017 ; Soomro et al., 2019 ; Khan et al., 2019 ) .
Particularly, Moreno and Smedby (2015) were interested in formulating general methods for enhancement technologies, and Kerrien et al. (2017) focused on modeling approaches.
In view of the potential of learning-based methods for tracking retinal vessels, Moccia et al. (2018) and Zhao et al. (2019) reviewed the principles and applications.
 Soomro et al. (2019) particularly focused on deep-learning-based works of retinal blood vessel segmentation.

 
 
 Figure 1: Recapitulative diagram of vessel tracking using learning-based methods. 
 
 

### 1.1 Aims of this paper

 
 This work aims to provide an up-to-date review of vessel tracking based on machine-learning-methods, also referred to as learning-based methods.
We focus on the learning-based methods for tracking vessels of various organs using different imaging modalities.
The recapitulative diagram of the learning-based methods for vessel tracking is shown in Fig.  1 .
To cover the articles to a feasible extent, a search for the term vessel/vascular segmentation/extraction has been performed using engines such as PubMed 1 1 
 1 
 
 
 
 
 
 
 
 http://www.ncbi.nlm.nih.gov/pubmed , IEEE Xplore 2 2 
 2 
 
 
 
 
 
 
 
 http://ieeexplore.ieee.org , and Google Scholar 3 3 
 3 
 
 
 
 
 
 
 
 http://scholar.google.com .
Among over 300 collated articles, the focus of attention was on papers published during the last 10 years.
Note that this article does not cover all the details of databases and evaluation standards that can be found in the literature ( Schaap et al., 2009 ; Hameeteman et al., 2011 ; Kirisli et al., 2013 ; Rudyanto et al., 2014 ; Vostatek et al., 2017 ; Moccia et al., 2018 ; Yan et al., 2018 ) .

 
 
 The rest of the paper is organized as follows.
Section 2 reviews the vessel tracking methods using conventional machine learning.
Section 3 reviews the vessel tracking approaches using deep learning.
Based on the reviewed methods, Section 4 introduces the evaluation issues.
Finally, Section 5 concludes the review and explores potential directions for future work on learning-based methods for vessel tracking.

 
 
 
 

## 2 Vessel tracking using conventional machine learning

 
 This section reviews the vessel-tracking works that employ conventional learning-based algorithms including the methodologies of hand-crafted features, classifications and statistical models.
Tables  3 - 3 summarize the decomposition of a selection of representative works in this field according to the applications.
Tables  4 summarizes the existing conventional machine-learning-based works by grouping them into different subcategories.

 
 
 Figure 2: Diagram of the retinal vessel segmentation using the conventional machine-learning method with supervised training. 
 
 
 
 
 

 Table 1: Overview of conventional machine-learning-based methods for tracking retinal vessels : the evaluation metrics and datasets are presented in Section 4 ; see list of abbreviations at the bottom. 
 
 
 
 
 
 
 Authors 
 | 
 
 
 Methods 
 | 
 
 
 Data 
 | 
 
 
 Experiment 
 | 
 
 
 Results 
 | 

 
 
 
 Becker et al. (2013) 
 | 
 
 
 Boosting-based, learning kernels 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Precision-recall curves 
 | 

 
 
 
 Sironi et al. (2015) 
 | 
 
 
 Learning separable filters 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 AUC=0.962 
 | 

 
 
 
 Annunziata and Trucco (2016) 
 | 
 
 
 Convolutional sparse coding- 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing , one-off train + test 
 | 
 
 
 AUPRC=0.87 
 | 

 
 | 
 
 
 filter learning 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 AUPRC=0.86 
 | 

 
 
 
 Gu et al. (2017) 
 | 
 
 
 Boosting-based , structured features 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Pr=0.7931, Re=0.7595, Sp=0.9711 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 20 images, five-fold cross-validation 
 | 
 
 
 Pr=0.7761, Re=0.7791, Sp=0.9741 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 14 for training, 14 for testing, one-off train + test 
 | 
 
 
 Pr=0.6660, Re=0.6850, Sp=0.9664 
 | 

 
 | 
 | 
 
 
 Retinal colored image, HRF 
 | 
 
 
 half for training, half for testing, one-off train + test 
 | 
 
 
 Pr=0.7775, Re=0.7602, Sp=0.9795 
 | 

 
 
 
 Javidi et al. (2017) 
 | 
 
 
 Dictionary learning, 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing , one-off train + test 
 | 
 
 
 Acc=0.9446 
 | 

 
 | 
 
 
 vessel and non-vessel features 
 | 
 | 
 | 
 | 

 
 
 
 Kalaie and Gooya (2017) 
 | 
 
 
 Hierarchical probabilistic framework, 
 | 
 
 
 Retinal colored image, REVIEW 
 | 
 
 
 16 images, leave-one-out 
 | 
 
 
 Acc=0.9446 
 | 

 
 | 
 
 
 intensity features of the cross sections 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 40 images, leave-one-out 
 | 
 
 
 Acc=0.970 
 | 

 
 
 
 Orlando et al. (2017) 
 | 
 
 
 SVM, features in CRF 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Pr=0.7854, Se=0.7897, Sp=0.9684 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 Pr=0.7740, Se=0.7680, Sp=0.9738 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 8 for training, 20 for testing, one-off train + test 
 | 
 
 
 Pr=0.7438, Se=0.7277, Sp=0.9712 
 | 

 
 | 
 | 
 
 
 Retinal colored image, HRF 
 | 
 
 
 5 for training, 40 for testing, one-off train + test 
 | 
 
 
 Pr=0.6950, Se=0.7794, Sp=0.9650 
 | 

 
 
 
 Zhang et al. (2017a) 
 | 
 
 
 
 
 
 Random forest, gaussian-based 
 
 filters wavelet transform 
 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9466, AUC=0.9703, Se=0.7861, Sp=0.9712 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 Acc=0.9547, AUC=0.9740, Se=0.7882, Sp=0.9729 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 27 for training, 1 for testing, leave-one-out 
 | 
 
 
 Acc=0.9502, AUC=0.9706, Se=0.7644, Sp=0.9716 
 | 

 

 
 
 * 
 
 List of abbreviations: Acc=accuracy; AUC=area under the ROC curve; AUPRC=area under the precision-recall curve; CRF=conditional random field; Pr=precision; Re=recall; REVIEW ( Al-Diri et al., 2008 ) ; Se=sensitivity; Sp=specificity; SVM= support vector machine. 

 

 
 
 

 Table 2: Overview of conventional machine-learning-based methods for tracking coronary vessels : the evaluation metrics and datasets are presented in Section 4 ; see list of abbreviations at the bottom. 
 
 
 
 
 
 
 Authors 
 | 
 
 
 Methods 
 | 
 
 
 Data 
 | 
 
 
 Experiment 
 | 
 
 
 Results 
 | 

 
 
 
 
 
 Schaap et al. (2011) 
 | 
 
 
 Nonlinear regression, point- 
 | 
 
 
 Coronary CTA, local data 
 | 
 
 
 82 for training, 1 for testing, leave-one-out 
 | 
 
 
 Distance=0.15mm 
 | 

 
 | 
 
 
 distribution and intensity model 
 | 
 
 
 Coronary CTA, CAT08 
 | 
 
 
 8 for training, 24 for testing, one-off train + test 
 | 
 
 
 AI=0.23mm, OF=0.725, OT=0.971, OV=0.969 
 | 

 
 
 
 Lesage et al. (2016) 
 | 
 
 
 Bayesian vessel model and particle- 
 | 
 
 
 Coronary CTA, local data 
 | 
 
 
 10 for training, 51 for testing 
 | 
 
 
 AI=0.25mm, OT=0.925, OV=0.862 
 | 

 
 | 
 
 
 filtering, flux-based image feature 
 | 
 | 
 | 

 
 
 
 Mehmet et al. (2016) 
 | 
 
 
 Boosting-based, image features, 
 | 
 
 
 Coronary CTA, local data 
 | 
 
 
 90 for training, 20 for testing 
 | 
 
 
 Se 0.9, Sp 0.9 
 | 

 
 | 
 
 
 orientation and scale 
 | 
 | 
 | 

 

 
 
 * 
 
 List of abbreviations: AI=average inside; CTA = computed tomography angiography; OF=overlap until first error; OT=overlap with the clinically relevant part of the vessel; OV=overlap; Se=sensitivity; Sp=specificity. 

 

 
 
 

 Table 3: Overview of conventional machine-learning-based methods for tracking other vessels : the evaluation metrics and datasets are presented in Section 4 ; see list of abbreviations at the bottom. 
 
 
 
 
 
 
 Authors 
 | 
 
 
 Methods 
 | 
 
 
 Data 
 | 
 
 
 Experiment 
 | 
 
 
 Results 
 | 

 
 
 
 Bogunović et al. (2012) 
 | 
 
 
 SVM, bifurcation features 
 | 
 
 
 Internal carotid artery 3DRA, 
 | 
 
 
 96 images, cross-validation 
 | 
 
 
 Cross-validation success rate =0.99 
 | 

 
 | 
 | 
 
 
 local data 
 | 
 | 

 
 
 
 Cheng et al. (2012) 
 | 
 
 
 Seed searching 
 | 
 
 
 Mammography, local data 
 | 
 
 
 1800 samples, 1200 for training, 400 for testing, 
 | 
 
 
 Se=0.93, Sp=0.851 
 | 

 
 | 
 | 
 | 
 
 
 four-fold cross-validation 
 | 
 | 

 
 
 
 Zheng et al. (2012) 
 | 
 
 
 Non-rigid deformation, position, 
 | 
 
 
 Aorta C-arm CT, local data 
 | 
 
 
 319 volumes, four-fold cross-validation 
 | 
 
 
 Mean error=1.08 mm 
 | 

 
 | 
 
 
 orientation and scale 
 | 
 | 
 | 

 
 
 
 Cherry et al. (2015) 
 | 
 
 
 Random forest, intensity, vesselness, 
 | 
 
 
 Pelvis CT angiograms, local data 
 | 
 
 
 10 for training, 30 for testing, one-off train + test 
 | 
 
 
 Pr=0.752, Re=0.677 
 | 

 
 | 
 
 
 ray-casting, MIP, and spanning tree 
 | 
 | 
 | 

 
 
 
 Rempfler et al. (2015) 
 | 
 
 
 Probabilistic model, 
 | 
 
 
 Mouse brain MR images, local data 
 | 
 
 
 4 for training, 1 for testing, leave-one-out 
 | 
 
 
 DSC=0.516 
 | 

 
 | 
 
 
 physiological-geometric properties 
 | 
 | 
 | 

 
 
 
 Schneider et al. (2015) 
 | 
 
 
 Random forest, oriented features 
 | 
 
 
 Synthetic vascular data, local data 
 | 
 
 
 4 for testing, leave-one-out 
 | 
 
 
 DSC=0.95 
 | 

 
 
 
 Zhang et al. (2017b) 
 | 
 
 
 Random forest, steerable- 
 | 
 
 
 Perivascular 7T MR, local data 
 | 
 
 
 19 image sets, two-fold cross-validation 
 | 
 
 
 DSC=0.661, Se=0.651 
 | 

 
 | 
 
 
 frangi-filters and OOF 
 | 
 | 
 | 

 
 
 
 Lorza et al. (2018) 
 | 
 
 
 SVM, a radial basis function kernel 
 | 
 
 
 Carotid bifurcation MRI, local data 
 | 
 
 
 49 arteries for testing 
 | 
 
 
 DSC wall overlap= 0.741 
 | 

 

 
 
 * 
 
 List of abbreviations: 3DRA=3D rotational angiography; CT=computed tomography; DSC=dice similarity coefficient; MIP=maximum intensity projection; MR=magnetic resonance; MRI=magnetic resonance imaging; OOF=optimally oriented flux; Pr=precision; Re=recall; Se=sensitivity; Sp=specificity. 

 

 
 
 
 
 Table 4: Overview of the techniques in traditional machine-learning-based methods: see list of abbreviations at the bottom. 
 
 
 
 
 
 Conventinal learning-based methods 
 | 
 
 
 Techniques 
 | 
 
 
 
 | 

 
 
 
 Hand-crafted featues 
 | 
 
 
 Intensity features: 
 | 
 
 
 Vukadinovic et al. (2010) ; Cherry et al. (2015) ; Mehmet et al. (2016) 
 | 

 
 
 
 [0.8pt/2pt] 
 | 
 
 
 Intensity gradient features: 
 | 
 
 
 Mehmet et al. (2016) 
 | 

 
 
 
 [0.8pt/2pt] 
 | 
 
 
 Bifurcation feature vectors: 
 | 
 
 
 Bogunović et al. (2012) 
 | 

 
 
 
 [0.8pt/2pt] 
 | 
 
 
 Spatial location: 
 | 
 
 
 Vukadinovic et al. (2010) ; Mehmet et al. (2016) 
 | 

 
 
 
 [0.8pt/2pt] 
 | 
 
 
 Angles: 
 | 
 
 
 Bogunović et al. (2012) ; Mehmet et al. (2016) 
 | 

 
 | 
 
 
 Learning-based kernels: 
 | 
 
 
 Poletti and Grisan (2014) ; Liu et al. (2014) ; Lesage et al. (2016) ; Asl et al. (2017) 
 | 

 
 
 
 [0.8pt/2pt] 
 | 
 
 
 Learning-based filters: 
 | 
 
 
 Lin et al. (2012) ; Azzopardi and Petkov (2013) ; Annunziata et al. (2015a) ; Sironi et al. (2015) 
 | 

 
 | 
 | 
 
 
 Annunziata and Trucco (2016) ; Zhang et al. (2017b) ; Deng et al. (2018) ; Javidi et al. (2017) 
 | 

 
 
 
 Classifications 
 | 
 
 
 K-means: 
 | 
 
 
 Coates and Ng (2012) ; Saffarzadeh et al. (2014) ; Zhang et al. (2014) ; Samuel et al. (2017) 
 | 

 
 | 
 | 
 
 
 Goceri et al. (2017) ; Lu et al. (2017) ; Xia et al. (2018) 
 | 

 
 | 
 
 
 Fuzzy C-means clustering: 
 | 
 
 
 Mapayi et al. (2015) ; Mapayi and Tapamo (2016) ; Khan and NaliniPriya (2016) 
 | 

 
 | 
 | 
 
 
 Haddad et al. (2018) ; Zeng et al. (2018) 
 | 

 
 | 
 
 
 Support Vector Machine: 
 | 
 
 
 A. Osareh (2009) ; You et al. (2011) ; Hanaoka et al. (2015) ; Chen et al. (2015) 
 | 

 
 | 
 | 
 
 
 Kang et al. (2015) ; Jawaid et al. (2017) ; Orlando et al. (2017) ; Lorza et al. (2018) 
 | 

 
 | 
 
 
 Boosting-based methods: 
 | 
 
 
 Turetken et al. (2016) ; Lupascu et al. (2010) ; Gu et al. (2017) 
 | 

 
 | 
 | 
 
 
 Memari et al. (2017) ; Lupascu et al. (2013) 
 | 

 
 | 
 | 
 
 
 Fraz et al. (2012a) ; Lupascu et al. (2013) ; Hashemzadeh and Adlpour Azar (2019) 
 | 

 
 | 
 
 
 Random forest: 
 | 
 
 
 Annunziata et al. (2015b) ; Melki et al. (2014) ; Zhang et al. (2017a) 
 | 

 
 | 
 | 
 
 
 Schneider et al. (2015) ; Sankaran et al. (2016) ; Cherry et al. (2015) 
 | 

 
 | 
 
 
 Hybrid classifiers: 
 | 
 
 
 Rani et al. (2016) ; Lugauer et al. (2014) ; Chapman et al. (2015) ; Hu et al. (2018b) 
 | 

 
 
 
 Statistical models 
 | 
 
 
 Threshold: 
 | 
 
 
 Vukadinovic et al. (2010) ; Cheng et al. (2012) 
 | 

 
 | 
 
 
 Intensity/appearance model: 
 | 
 
 
 Schaap et al. (2011) ; Zheng et al. (2012) ; Rempfler et al. (2014) 
 | 

 
 | 
 
 
 Topological model: 
 | 
 
 
 Rempfler et al. (2015) ; Asl et al. (2017) ; Zhao et al. (2017) 
 | 

 
 | 
 | 
 
 
 Chai et al. (2013) ; Kalaie and Gooya (2017) 
 | 

 

 
 

### 2.1 Hand-crafted features

 
 A broad definition of hand-crafted features is provided in ( Lesage et al., 2009 ) .
Conventional machine-learning-based methods train models with numerous hand-crafted features, which should be well-designed according to the applications.
These features (i.e., global and local features) can be obtained by a series of filters such as those given in ( Frangi et al., 1998 ; Agam et al., 2005 ; Manniesing et al., 2006 ) . Vukadinovic et al. (2010) used a set of features for classifying the calcium candidate object of blood vessels.
These features include smoothed intensity, Gaussian derivative features, and a set of shape features including spatial locations (distance to the lumen).
 Bogunović et al. (2012) extracted a set of labeled bifurcation feature vectors of vessels to train the classifier.
 Mehmet et al. (2016) extracted local features based on image intensity, intensity gradient, sample positions, and angles.
They claimed that the Hessian-matrix-based features can aid in distinguishing between tubular and non-tubular structures.

 
 
 One application of hand-crafted features is in the development of learning-based kernels.
 Poletti and Grisan (2014) learned a set of optimal discriminative convolution kernels to be used in AdaBoost classifications.
The multi-kernel learning method proposed in ( Liu et al., 2014 ) utilizes the features from the Hessian-matrix-based vesselness measures, multi-scale Gabor filter responses, and multi-scale line strengths.
 Lesage et al. (2016) learned the non-parametric kernel with likelihood terms of direction and radius transition priors.
To estimate the vessel direction and diameter, Asl et al. (2017) formulated a kernelized covariance matrix from the training data.

 
 
 Another application is the learning-based filters of vessels.
Assuming that the vessel properties changed constantly, Lin et al. (2012) learned the continuity pattern of the current segment using the extended Kalman filter.
 Azzopardi and Petkov (2013) learned the appropriate prototype features in the filter-configuration process.
In addition to the appearance features of learning filters, Annunziata et al. (2015a) introduced the context information (i.e., relationships among objects) in the filter-learning process.
The authors assumed that the learned context filters had two clear advantages: incorporating high-level information and obtaining high efficiency and adaptability.
To accelerate the learning process, Sironi et al. (2015) computed the filters by linearly combining them with a smaller number of separable filters.
This operation can considerably address the problem of the computational complexity at no extra cost in terms of performance.
 Annunziata and Trucco (2016) proposed a warm-start strategy to solve this problem.
This strategy is based on carefully designing hand-crafted filters and modeling appearance properties of curvilinear structures, which are then refined by convolutional sparse coding.
Using vascular filters (e.g., Frangi filter and optimally oriented flux), Zhang et al. (2017b) extracted the corresponding types of vascular features and integrated these feature responses into a structured random forest to classify voxels into positive and negative classes. Deng et al. (2018) named all the features for the random forest (RF) as discriminative integrated features.
These features are classified as low-level features, vascular features, context features, and local self-similarity descriptor.
In addition to the kernels and filters, Javidi et al. (2017) constructed two separate dictionaries to learn the vessel and non-vessel representations. These learned dictionaries yield a strong representation containing the semantic concepts of the image.

 
 
 

### 2.2 Classifications

 
 The conventional machine-learning-based methods obtain vessels using classifiers.
For vessel-tracking tasks, the methodologies of classification can be broadly categorized into the unsupervised and supervised learning strategies.

 
 
 Unsupervised learning-based methods train the classifier without using labeled vessel data or explicitly using any supervised classification techniques.
To separate related regions, seeds and patches, the main schemes reported in the literature are clustering techniques (e.g., k-means and fuzzy C-means).
Instead of explicitly obtaining sparse representations, k-means clustering tends to discover sparse projections of the data ( Coates and Ng, 2012 ) .
As a pre-processing step, the k-means algorithm can be used to partition the pixels into several clusters; e.g., three clusters of related regions ( Saffarzadeh et al., 2014 ) or five groups of images ( Zhang et al., 2014 ) .
In the process, Samuel et al. (2017) employed the k-means algorithm with subtractive clustering to separate the vessel regions in the image according to the gray-scale intensity.
The k-means algorithm is also used for the final refinement of vessel segmentation ( Goceri et al., 2017 ) .
In addition to vessel regions, Lu et al. (2017) used the manually annotated seeds to represent the vascular features and utilized k-means clustering to exclude the wrong seeds.
To find the representative patches from numerous candidate patches, Xia et al. (2018) used k-means clustering to group the patches under the Euclidean distance metric.
Fuzzy C-means clustering is another unsupervised learning method for pattern recognition that employs various image properties for separation.
Image pixel intensities are not mutually independent; hense, Kande et al. (2010) used a thresholding technique based on the spatially weighted fuzzy C-means algorithm, which can well preserve the spatial structures in a binarized/thresholded image.
In ( Mapayi et al., 2015 ) , phase-congruency ( Kovesi, 1999 ) has been used to preserve the features with in-phase frequency components, and fuzzy C-means clustering is performed for accurate retinal vessel segmentation.
 Mapayi and Tapamo (2016) further investigated the difference image with fuzzy C-means for the detection of vessels in the retinal image.
An improved fuzzy C-means clustering in ( Khan and NaliniPriya, 2016 ) was also used for pixel classification based on the texture features.
Using the contrast-time curve of the pixel as inputs, Haddad et al. (2018) separated the major vessels from the capillary blush and background noise through fuzzy C-means clustering.
 Zeng et al. (2018) constructed the intensity model based on kernel fuzzy C-means to extract the intensity feature of thick vessels.

 
 
 The ground truth is absent; hense, the performance of unsupervised methods relies on particular features based on the statistical distribution of the overall input data.
In contrast, supervised learning methods require a manually annotated set of training images for classifications.
The extracted features and ground truth of every sample are collected to train the classifier.
Most of these methods in the supervised category use various classifiers (Fig.  2 )—support vector machine (SVM), boosting-based methods, and random forests—to distinguish the vascular patterns in the images.

 
 
 The SVM classifier performs vessel or non-vessel classification by constructing an N-dimensional hyperplane that optimally separates the vessel samples into different categories.
The classification ability of the SVM is based on the feature vectors obtained by different operators and vascular-dedicated filters.
The operators can be line operator ( Ricci and Perfetti, 2007 ) , Gabor filters ( A. Osareh, 2009 ) , and wavelets ( You et al., 2011 ) .
The vascular-dedicated filters can be Frangi filter ( Frangi et al., 1998 ) and optimally oriented flux ( Law and Chung, 2010 ) .
To distinguish between the vessels, the feature vectors can also be formulated via general measures; e.g., distance between adjacent nodes ( Hanaoka et al., 2015 ) , geometric shapes ( Kang et al., 2015 ) , and normal cross-sections ( Jawaid et al., 2017 ) .
To deal with more complex cases in the curve detection, Chen et al. (2015) utilized the joint feature representations—e.g., smoothness of points and position relationships—for classification.
Instead of classifying the pixels, in the framework of the fully connected conditional random field (CRF), the SVM methods are employed to adjust the weight parameters in the energy function ( Orlando et al., 2017 ) .
 Lorza et al. (2018) used the SVM with a radial basis function kernel to obtain the probability map of the vessel region.

 
 
 Boosting-based methods are dependent on building strong classification models from a linear combination of weak classifiers, making the training easier and faster.
Simple functions, such as regression stumps, can be employed in boosting-based methods to detect curvilinear objects ( Turetken et al., 2016 ) .
In vesse-segmentation tasks, a regression stump is a decision tree with two terminal nodes.
For a given vascular feature, the tree selects a branch according to the threshold based on a binary decision function.
As a special case of boosting-based methods, the AdaBoost learning model is trained to automatically detect the bifurcation points of vessels with elaborately selected features.
To improve the accuracy of classifications, numerous filters are necessary to obtain the vascular features ( Zhou et al., 2007 ) . Lupascu et al. (2010) used feature vectors, which are composed of eight elements including the output of filters, measures, and other transformation results, to encode vascular information on the local intensity structure, spatial properties, and geometry at multiple scales.
 Gu et al. (2017) constructed boosting-tree methods based on features such as variable sizes and locations.
These features represent the encoded global contextual information (e.g., context distance and local spatial label patterns).
 Memari et al. (2017) completed the segmentation based on feature extraction and selection steps, along with the AdaBoost classifier.
If a classification tree has an AdaBoost classifier at each node, then the tree will be the probabilistic boosting-tree (PBT) classifier. Zheng et al. (2011) exploited the PBT to identify the pixel inside the vessel using 24 feature vectors from the Hessian matrix.
To improve the performance of retinal segmentation methods in the presence of lesions, an ensemble classifier of boosted ( Freund and Schapire, 1995 ) and bagged decision trees ( Breiman, 1996 ) has been proposed to manage the healthy and pathological retinal images via several encoded features ( Fraz et al., 2012a ) .
The ensembles of bagged decision trees have also been employed to learn the mapping between vessel widths and corresponding points ( Lupascu et al., 2013 ) .
In ( Hashemzadeh and Adlpour Azar, 2019 ) , a root-guided decision tree was used to distinguish between the vessel and non-vessel regions.

 
 
 A collection of tree-structured classifiers can be assembled as an RF classifier.
Different from the SVM and decision trees, the RF ( Cutler et al., 2001 ; Zhang et al., 2016 ) tends to deliver high performance because of the embedded feature selection in the model-generation process.
In the vessel-tracking process, the selected features are invariably related to the intensity profile or vascular shapes (e.g., tubular structures ( Annunziata et al., 2015b ; Melki et al., 2014 ; Zhang et al., 2017a ) , vessel center ( Schneider et al., 2015 ) , and tree-like structures ( Sankaran et al., 2016 ) ).
To cover more features for the RF classifier, researchers used multiple techniques to generate representations in various spaces. Cherry et al. (2015) used two sets of features to distinguish the abnormal vessels: vessel cues and local information.

 
 
 To improve the performance and avoid the over-fitting problems, hybrid classifiers are used for vessel-classification problems.
 Rani et al. (2016) combined the SVM and tree-bagger techniques to distinguish between the vessel and non-vessel structures.
The works of Lugauer et al. (2014) and Chapman et al. (2015) used the RF, PBT, and logistic regression classifier to identify the lumen contours and edges of vessels.
 Hu et al. (2018b) intricately applied the cascade-AdaBoost-SVM classifiers to delineate the vessel boundaries.

 
 
 

### 2.3 Statistical models

 
 The profiles of intensity and geometries of vessels can be learned using statistical models.
 Vukadinovic et al. (2010) determined the vessel calcium object threshold by simply observing the calcium objects disappear in the image dataset; this threshold is used to segment the calcium regions of the vessels.
In ( Cheng et al., 2012 ) , the vessel center threshold, which is referred to as the strong peak near the center of the profile, was learned from a set of manually-labeled samples of parallel linear structures.
For more detailed information, Schaap et al. (2011) learned the local point distribution models and a nonlinear boundary intensity model by statistically analyzing the set of annotated training data.
 Zheng et al. (2012) divided the vessel model into four structures, each of which is recognized via learned detectors. Instead of designing individual classifiers for each geometrical constraint, Rempfler et al. (2014) simply learned the global statistic of the desired geometrical properties of the network from the dataset.

 
 
 Based on the probabilistic model, the topological structures of vessels, e.g., branches and connections, can also be learned for vessel tracking.
 Rempfler et al. (2015) learned the physiological geometric properties of vessels such as the relative frequencies of radii and deviation angles of vessel segments.
 Asl et al. (2017) learned these relationships using kernels.
 Zhao et al. (2017) learned the topological tree and geometrical statistics of parameters including tree hierarchy, branch angle, and length statistics.
Contrary to the pixel-based and object-based methods, Chai et al. (2013) modeled the vessel connections using graph theory.
To describe the shape of the graph, they constructed three sets of parameters: graph connectivity, edge orientation, and line width,
These parameters can be learned from annotated image samples through a maximum likelihood estimation.
Considering the intensity distributions, Kalaie and Gooya (2017) developed a directed probabilistic graphical model whose hyperparameters are estimated using a maximum likelihood solution based on Laplace approximation.

 
 
 
 

## 3 Vessel tracking based on deep learning

 

 Table 5: Overview of deep-learning-based methods for tracking retinal vessels : the evaluation metrics and datasets are presented in Section 4 ; see list of abbreviations at the bottom. 
 
 
 
 
 
 
 Authors 
 | 
 
 
 Methods 
 | 
 
 
 Data 
 | 
 
 
 Experiment 
 | 
 
 
 Results 
 | 

 
 
 
 Li et al. (2016) 
 | 
 
 
 Encoder-decoder, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing , one-off train + test 
 | 
 
 
 Acc=0.9527, AUC=0.9738, Se=0.7569, Sp=0.9816 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 Acc=0.9628, AUC=0.9879, Se=0.7726, Sp=0.9844 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB-1 
 | 
 
 
 20 for training, 8 for testing, one-off train + test 
 | 
 
 
 Acc=0.9581, AUC=0.9716, Se=0.7507, Sp=0.9793 
 | 

 
 
 
 Liskowski and Krawiec (2016) 
 | 
 
 
 CNN without pooling, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9495, AUC=0.9720 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 Acc=0.9566, AUC=0.9785 
 | 

 
 
 
 Lahiri et al. (2017) 
 | 
 
 
 GAN, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 AUC=0.962 
 | 

 
 
 
 Costa et al. (2018) 
 | 
 
 
 GANs, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 AUC=0.841 
 | 

 
 
 
 Guo et al. (2018) 
 | 
 
 
 Multiple CNNs, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9597, AUC=0.9726 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 20 for testing 
 | 
 
 
 Acc=0.9613, AUC=0.9737 
 | 

 
 
 
 Yan et al. (2018) 
 | 
 
 
 Encoder-decoder, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9542, AUC=0.9752, Se=0.7653, Sp=0.9818 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 Acc=0.9612, AUC=0.9801, Se=0.7581, Sp=0.9846 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 20 for training, 8 for testing, one-off train + test 
 | 
 
 
 Acc=0.9610, AUC=0.9781, Se=0.7633, Sp=0.9809 
 | 

 
 | 
 | 
 
 
 Retinal colored image, HRF 
 | 
 
 
 5 for training, 40 for testing, one-off train + test 
 | 
 
 
 Acc=0.9437, Pr=0.6647, Se=0.7881, Sp=0.9592 
 | 

 
 
 
 Wu et al. (2018) 
 | 
 
 
 Multiple CNNs, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9567, AUC=0.9807, Se=0.7844, Sp=0.9819 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 20 for training, 8 for testing, one-off train + test 
 | 
 
 
 Acc=0.9637, AUC=0.9825, Se=0.7538, Sp=0.9847 
 | 

 
 
 
 Zhao et al. (2018b) 
 | 
 
 
 GANs, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Se=0.8038, Sp=0.9815 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 
 
 
 Se=0.7896, Sp=0.9841 
 | 

 
 | 
 | 
 
 
 Retinal colored image, HRF 
 | 
 
 
 22 for training, 23 for testing, one-off train + test 
 | 
 
 
 Se=0.8001, Sp=0.9823 
 | 

 
 
 
 Zhang and Chung (2018) 
 | 
 
 
 U-net, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9504, AUC=0.9799, Se=0.8723, Sp=0.9618 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 15 for training, 5 for testing, four-fold cross-validation 
 | 
 
 
 Acc=0.9712, AUC=0.9882, Se=0.7673, Sp=0.9901 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 21 for training, 7 for testing, four-fold cross-validation 
 | 
 
 
 Acc=0.9770, AUC=0.9900, Se=0.7670, Sp=0.9909 
 | 

 
 
 
 Gu et al. (2019) 
 | 
 
 
 Context encoder network 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.955, AUC=0.978 
 | 

 
 
 
 Jin et al. (2019) 
 | 
 
 
 Deformable U-net, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9566, AUC=0.9802, TNR=0.9800, TPR=0.7963 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 Acc=0.9641, AUC=0.9832, TPR=0.7595, TNR=0.9878 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 14 for training, 14 for testing, one-off train + test 
 | 
 
 
 Acc=0.9610, AUC=0.9804, TNR=0.9752, TPR=0.8155 
 | 

 
 | 
 | 
 
 
 Retinal colored image, HRF 
 | 
 
 
 15 for training, 30 for testing, one-off train + test 
 | 
 
 
 Acc=0.9651, AUC=0.9831, TNR=0.9874, TPR=0.7464 
 | 

 
 
 
 Lian et al. (2019) 
 | 
 
 
 U-net, Res-net and 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9692, Pr=0.8637, Se=0.8278, Sp=0.9861 
 | 

 
 | 
 
 
 attention scheme, 2D 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 
 
 
 Acc=0.9740, Pr=0.8823, Se=0.8342, Sp=0.9916 
 | 

 
 
 
 Mou et al. (2019) 
 | 
 
 
 Dense dilate network, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 
 
 
 Acc=0.9594, AUC=0.9796, Se=0.8126, Sp=0.9788 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 15 for training, 5 for testing, four-fold cross-validation 
 | 
 
 
 Acc=0.9685, AUC=0.9858, Se=0.8391, Sp=0.9769 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 21 for training, 7 for testing, four-fold cross-validation 
 | 
 
 
 Acc=0.9637, AUC=0.9812, Se=0.8268, Sp=0.9773 
 | 

 
 
 
 Shin et al. (2019) 
 | 
 
 
 CNN + GNN, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing , one-off train + test 
 | 
 
 
 Acc=0.9271, AUC=0.9802, Se=0.9382, Sp=0.9255 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 
 
 
 Acc=0.9378, AUC=0.9877, Se=0.9598, Sp=0.9352 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 20 for training, 8 for testing, one-off train + test 
 | 
 
 
 Acc=0.9373, AUC=0.9830, Se=0.9463, Sp=0.9364 
 | 

 
 | 
 | 
 
 
 Retinal colored image, HRF 
 | 
 
 
 15 for training, 30 for testing, one-off train + test 
 | 
 
 
 Acc=0.9349, AUC=0.9838, Se=0.9546, Sp=0.9329 
 | 

 
 
 
 Cherukuri et al. (2020) 
 | 
 
 
 CNN + geometric prior, 2D 
 | 
 
 
 Retinal colored image, DRIVE 
 | 
 
 
 20 for training, 20 for testing, four-fold cross-validation 
 | 
 
 
 Acc=0.9563, AUC=0.9814 
 | 

 
 | 
 | 
 
 
 Retinal colored image, STARE 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 
 
 
 Acc=0.9687, AUC=0.9903 
 | 

 
 | 
 | 
 
 
 Retinal colored image, CHASE-DB1 
 | 
 
 
 14 for training, 14 for testing, one-off train + test 
 | 
 
 
 Acc=0.9672, AUC=0.9833 
 | 

 
 
 
 Ding et al. (2020b) 
 | 
 
 
 Pre-trained model, vessel 
 | 
 
 
 Retinal UWF FP, PRIME-FP20 
 | 
 
 
 15 images, four-fold cross-validation 
 | 
 
 
 AUCPR=0.842, Max DSC=0.772 
 | 

 
 | 
 
 
 maps, and noise label, 2D 
 | 
 | 
 | 
 | 

 

 
 
 * 
 
 List of abbreviations: Acc=accuracy; AUC=area under the ROC curve; CNN=convolutional neural networks; GAN=generative adversarial network; GNN=graph neural network; Pr=precision; Re=recall; Se=sensitivity; Sp=specificity; TNR=true negative rate; TPR=true positive rate; UWF FP=ultra-widefield fundus photography . 

 

 
 
 
 
 
 

 Table 6: Overview of deep-learning-based methods for tracking coronary vessels : the evaluation metrics and datasets are presented in Section 4 ; see list of abbreviations at the bottom. 
 
 
 
 
 
 
 Authors 
 | 
 
 
 Methods 
 | 
 
 
 Data 
 | 
 
 
 Experiment 
 | 
 
 
 Results 
 | 

 
 
 
 Lee et al. (2019) 
 | 
 
 
 CNN + Shape prior, 3D 
 | 
 
 
 Coronary CTA, local data 
 | 
 
 
 274 for training, 136 for testing, one-off train + test 
 | 
 
 
 DSC=0.768, HD=3.55mm 
 | 

 
 
 
 Shin et al. (2019) 
 | 
 
 
 CNN + GNN, 2D 
 | 
 
 
 Coronary x-ray, CA-XRA (local data) 
 | 
 
 
 2958 for training, 179 for testing, one-off train + test 
 | 
 
 
 Acc=0.9517, AUC=0.9914, Se=0.9700, Sp=0.9507 
 | 

 
 
 
 Wolterink et al. (2019) 
 | 
 
 
 
 
 
 CNN with dilated- 
 
 convolution, 3D 
 
 | 
 
 
 Coronary CTA, CAT08 
 | 
 
 
 7 for training, 1 for testing, leave-one-out 
 | 
 
 
 AI=0.21mm, OF=0.815, OT=0.970, OV=0.937 
 | 

 
 | 
 | 
 
 
 Coronary CTA, UMCU dataset (local data) 
 | 
 
 
 8 for training, 50 for testing, one-off train + test 
 | 
 
 
 Median of radius=0.81mm 
 | 

 
 | 
 | 
 
 
 Coronary CTA, orCaScore 
 | 
 
 
 8 for training, 36 for testing, one-off train + test 
 | 
 
 
 Visual check 
 | 

 

 
 
 * 
 
 List of abbreviations: 

 

 * 
 
 Acc=accuracy; AI=average inside; AUC=area under the ROC curve; CNN=convolutional neural networks; CTA = computed tomography angiography; GNN=graph neural network; DSC=dice similarity coefficient; HD=hausdorff distance; Se=sensitivity; Sp=specificity; OF=overlap until first error; OT=overlap with the clinically relevant part of the vessel; OV=overlap. 

 

 
 
 

 Table 7: Overview of deep-learning-based methods for tracking other vessels : the evaluation metrics and datasets are presented in Section 4 ; see list of abbreviations at the bottom. 
 
 
 
 
 
 
 Authors 
 | 
 
 
 Methods 
 | 
 
 
 Data 
 | 
 
 
 Experiment 
 | 
 
 
 Results 
 | 

 
 
 
 Marques et al. (2016) 
 | 
 
 
 U-net, 3D 
 | 
 
 
 Multiple organs, CT, MR, local data 
 | 
 
 
 67 for training, 19 for testing, one-off train + test 
 | 
 
 
 Pr=0.362 
 | 

 
 
 
 Huang et al. (2018) 
 | 
 
 
 Unet, 3D 
 | 
 
 
 Liver contrast-enhanced CT, 3D-IRCADb 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 
 
 
 DSC=0.675, Se=0.743 
 | 

 
 | 
 | 
 
 
 Liver CT, SLIVER07 
 | 
 
 
 20 for testing 
 | 
 
 
 Visual check 
 | 

 
 | 
 | 
 
 
 Liver CT, local data 
 | 
 
 
 10 for testing 
 | 
 
 
 Visual check 
 | 

 
 
 
 Lian et al. (2018) 
 | 
 
 
 Modified U-net, 2D 
 | 
 
 
 Perivascular spaces, 7T MR, local data 
 | 
 
 
 6 for training, 14 for testing, one-off train + test 
 | 
 
 
 DSC=0.77, PPV=0.83, Se=0.74 
 | 

 
 
 
 Nardelli et al. (2018a) 
 | 
 
 
 CNN + graph cut, 3D 
 | 
 
 
 Pulmonary CT, local data 
 | 
 
 
 4 for training, 16 for validation 
 | 
 
 
 Acc=0.87 
 | 

 
 
 
 Kitrungrotsakul et al. (2019) 
 | 
 
 
 DensNet, 2.5D 
 | 
 
 
 hepatic MR, IRCAD 
 | 
 
 
 19 for training, 1 for testing, leave-one-out 
 | 
 
 
 DSC=0.903, Se=0.929 
 | 

 
 | 
 | 
 
 
 hepatic MR, VASCUSYNTH 
 | 
 
 
 9 for training, 1 for testing, leave-one-out 
 | 
 
 
 DSC=0.901 
 | 

 
 
 
 He et al. (2020) 
 | 
 
 
 Auto-encoder + 
 | 
 
 
 Renal artery, CT, local data 
 | 
 
 
 52 for training, 104 for testing, one-off train + test 
 | 
 
 
 DSC=0.884, HD=25.439mm 
 | 

 
 | 
 
 
 dense bias connection, 3D 
 | 
 | 
 | 
 | 

 
 
 
 Nazir et al. (2020) 
 | 
 
 
 CNN + dilated- 
 | 
 
 
 Intracranial vessel, CTA, local data 
 | 
 
 
 50 for training, 20 for testing, one-off train + test 
 | 
 
 
 DSC=0.8946, HD=5.04mm 
 | 

 
 | 
 
 
 convolution, 3D 
 | 
 | 
 | 
 | 

 
 
 
 Ni et al. (2020) 
 | 
 
 
 CNN + channel attention 
 | 
 
 
 Intracranial vessel, CTA, local data 
 | 
 
 
 9488 slices for training, 480 images for testing 
 | 
 
 
 DSC=0.965 
 | 

 

 
 
 * 
 
 List of abbreviations: Acc=accuracy; CNN=convolutional neural networks; CT=computed tomography; CTA = computed tomography angiography; DSC= dice similarity coefficient; HD=hausdorff distance; IRCAD: http://www.ircad.fr ; MR=magnetic resonance; Pr=precision; PPV=positive predictive value; Se=sensitivity; VASCUSYNTH ( Jassi, 2010 ) . 

 

 
 
 
 

 Table 8: Overview of techniques in deep-learning-based methods: see list of abbreviations at the bottom. 
 
 
 
 
 
 
 Deep-learning-based methods 
 | 
 
 
 Techniques 
 | 
 
 
 
 | 

 
 
 
 Network architectures 
 | 
 
 
 CNN + Dilated convolution: 
 | 
 
 
 Wolterink et al. (2019) ; Mou et al. (2019) ; Nazir et al. (2020) 
 | 

 
 | 
 
 
 CNN + Deformation convolution: 
 | 
 
 
 Jin et al. (2019) 
 | 

 
 | 
 
 
 CNN + No Pooling: 
 | 
 
 
 Liskowski and Krawiec (2016) ; Tetteh et al. (2017) 
 | 

 
 | 
 
 
 CNN + Probability maps: 
 | 
 
 
 Ganin and Lempitsky (2015) ; Fu et al. (2016b) ; Mo and Zhang (2017) 
 | 

 
 | 
 | 
 
 
 Hu et al. (2018a) ; Uslu and Bharath (2019) ; Lin et al. (2019) ; Ding et al. (2020b) 
 | 

 
 | 
 
 
 CNN + Attention mechanism: 
 | 
 
 
 Shen et al. (2019) ; Li et al. (2019) ; Lian et al. (2019) ; Ni et al. (2020) 
 | 

 
 | 
 
 
 CNN + Skipping/short connection: 
 | 
 
 
 Feng et al. (2018) ; Guo et al. (2019) 
 | 

 
 | 
 
 
 CNN + CRF: 
 | 
 
 
 Fu et al. (2016a) ; Luo et al. (2017) 
 | 

 
 | 
 
 
 CNN + Prior: 
 | 
 
 
 Lee et al. (2019) ; Cherukuri et al. (2020) 
 | 

 
 | 
 
 
 Multi-task learning: 
 | 
 
 
 Maninis et al. (2016) ; Tan et al. (2017) 
 | 

 
 | 
 
 
 Encoder-decoder: 
 | 
 
 
 Li et al. (2016) ; Fan and Mo (2017) ; Dasgupta and Singh (2017) 
 | 

 
 | 
 | 
 
 
 Gu et al. (2019) ; He et al. (2020) 
 | 

 
 | 
 
 
 U-net: 
 | 
 
 
 Fan et al. (2018) ; Huang et al. (2018) 
 | 

 
 | 
 
 
 Modified U-net: 
 | 
 
 
 Chen et al. (2018) ; Kandil et al. (2018) ; Wang et al. (2019) ; Zhang et al. (2019) 
 | 

 
 | 
 | 
 
 
 Girard et al. (2019) ; Zhang and Chung (2018) ; Dharmawan et al. (2019) ; Zhang et al. (2019) 
 | 

 
 | 
 
 
 GNN: 
 | 
 
 
 Shin et al. (2019) 
 | 

 
 | 
 
 
 GANs: 
 | 
 
 
 Costa et al. (2018) ; Yu et al. (2019) 
 | 

 
 | 
 
 
 Multiple CNNs: 
 | 
 
 
 Wu et al. (2018) ; Guo et al. (2018) 
 | 

 
 
 
 Pre-processing: 
 | 
 
 
 Contrast/brightness normalization: 
 | 
 
 
 Vega et al. (2015) ; Liskowski and Krawiec (2016) 
 | 

 
 | 
 
 
 Whiting: 
 | 
 
 
 Liskowski and Krawiec (2016) ; Marques et al. (2016) 
 | 

 
 | 
 
 
 Augmentation: 
 | 
 
 
 Huang et al. (2018) ; Guo et al. (2019) ; Fan et al. (2018) 
 | 

 
 | 
 | 
 
 
 Livne et al. (2019) ; Zreik et al. (2018) ; Lin et al. (2019) 
 | 

 
 
 
 Sampling strategies 
 | 
 
 
 Patch as samples: 
 | 
 
 
 Nardelli et al. (2018b) ; Wolterink et al. (2019) 
 | 

 
 | 
 
 
 Image as samples; 
 | 
 
 
 Hu et al. (2018a) ; Mo and Zhang (2017) 
 | 

 
 | 
 
 
 Processed image as samples: 
 | 
 
 
 Nardelli et al. (2017) ; Nardelli et al. (2018a) ; Hajabdollahi et al. (2018) 
 | 

 
 | 
 | 
 
 
 Zhao et al. (2018a) ; Zreik et al. (2018) 
 | 

 
 
 
 Loss functions: 
 | 
 
 
 Loss based on cross-entropy: 
 | 
 
 
 Dasgupta and Singh (2017) ; Nardelli et al. (2018b) ; Wu et al. (2018) ; Jin et al. (2019) 
 | 

 
 | 
 | 
 
 
 Dharmawan et al. (2019) ; Mo and Zhang (2017) ; Guo et al. (2019) ; Lin et al. (2019) 
 | 

 
 | 
 
 
 Loss to tackle data imbalance: 
 | 
 
 
 Li et al. (2018) ; Zhang and Chung (2018) ; Hu et al. (2018a) ; Livne et al. (2019) 
 | 

 
 | 
 | 
 
 
 Soomro et al. (2019) ; Kitrungrotsakul et al. (2019) ; Huang et al. (2018) ; Lian et al. (2018) 
 | 

 
 | 
 
 
 Loss based on squared error: 
 | 
 
 
 Li et al. (2016) ; Fan and Mo (2017) 
 | 

 
 | 
 
 
 More complex: 
 | 
 
 
 Yan et al. (2018) ; Jiang et al. (2019) 
 | 

 

 
 
 * 
 
 List of abbreviations: CNN=convolutional neural networks; CRF=conditional random field; GAN=generative adversarial network; GNN=graph neural network. 

 

 
 
 
 Using deep-learning-based methods, deep neural networks can be developed to map the input data into vascular patterns such as center points and vascular regions.
These patterns can be used to obtain the vessels directly or indirectly.
To this end, various deep-learning techniques have been proposed.
This section reviews the deep-learning-based methods from three aspects: frameworks of vessel tracking (Section 3.1 ), architecture of deep neural networks (Section 3.2 ), and model training (Section 3.3 ).
Tables  5 - 7 summarize the decompositions of a selection of works which are representative of the main trends in the field according to the applications.
Tables  8 summarizes the existing deep-learning-based works by grouping them into different subcategories.

 
 

### 3.1 Frameworks of vessel tracking

 
 Vessel tracking can be achieved using hierarchical features via a unified framework or a two-step processing scheme.
The unified vessel-tracking methods are implemented by integrating feature extraction and pixel classification into one network.
In contrast, the two-step scheme generally employs a conventional method to track the vessel based on the preceding vessel features extracted using a deep convolutional neural network (CNN).

 
 
 The unified framework of vessel tracking can be transformed into resolving a classification or regression problem via the fully connected layers of CNN.
The output neurons of CNN that are generally connected to the fully connected layers of the network determine the labels of pixels.
To separate vascular regions from the background using CNN, two neurons are typically set as output layers, following the fully connected layers ( Liskowski and Krawiec, 2016 ; Marques et al., 2016 ; Dasgupta and Singh, 2017 ; Hu et al., 2017 ; Oliveira et al., 2017 ) .
More neurons are output simultaneously to segment vessels and other structures ( Maninis et al., 2016 ; Tan et al., 2017 ) ; this can be regarded as multi-task learning.
The idea of multiple tasks has been extended in ( Lahiri et al., 2017 ) , where a discriminator-classifier network differentiates between fake and real vessel samples and assigns correct class labels.
The neurons of fully connected layers in conventional CNNs have large receptive fields of input; hence, the results are extremely coarse in the pixel-level vessel segmentation.
To resolve the problem, Li et al. (2016) improved the coarse results of conventional CNNs by outputting label maps of the same size.
A pixel in the label maps can be affected by the multiple image patches in its neighborhood.
This idea is similar to the fully connected CRF, which considers the relationships among the neighbor pixels.
To achieve vessel segmentation, the CRF layers can also be used after using the convolutional layers ( Fu et al., 2016a ; Luo et al., 2017 ) .

 
 
 The vessel-tracking process can be divided into two steps: feature learning and vessel tracking.
In feature learning, the CNN maps the input image into intermediate representations located between the input and tracking results; e.g., probability maps ( Khowaja et al., 2016 ; Wu et al., 2016 ; Nasr-Esfahani et al., 2016 ; Wolterink et al., 2019 ; Mou et al., 2019 ) , geometric priors ( Cherukuri et al., 2020 ) , and other feature maps ( Wang et al., 2015 ) .
In vessel tracking, the conventional tracking method can be applied to these intermediate representations.
The simple approach to complete the tracking is thresholding the probability map ( Nasr-Esfahani et al., 2016 ; Mo and Zhang, 2017 ; Nasr-Esfahani et al., 2018 ) .
 Wang et al. (2015) employed ensemble RFs to classify the vascular pixels based on the output feature maps from the selected layers of CNN.
 Guo et al. (2018) used a voting scheme to determine the results obtained by the CNN.
 Mou et al. (2019) performed vessel tracking by integrating the predicted probability maps and local vessel directions into the regularized walk algorithm.
To refine the results of CNNs, Hu et al. (2018a) added CRF modules at the end of the network and Chu et al. (2013) used the rank-1 tensor-approximation approach to complete the tracking.
Inspired by the label-propagation steps of registration methods, Lee et al. (2019) employed a CNN to learn the deformations between the source and the target vessels.
The authors assumed that this template transformer network can provide guarantees on the resulting shapes of vessels.

 
 
 

### 3.2 Network architectures

 
 In vessel-tracking tasks, the CNNs are widely adopted for identifying hierarchical vascular features.
To design an effective CNN for the recognition of vascular patterns, two aspects require thorough investigation: network components and integration of multiple networks.

 
 

#### 3.2.1 Network components

 
 Figure 3: Illustration of three selected CNN frameworks for the coronary segmentation: pixel-wise CNN (top), encoder-decoder (middle), and U-net (bottom). 
 
 
 A CNN is composed of a series of layers (Fig.  3 ), typically including the convolutional layers, pooling layers, and fully connected layers.
The convolutional and pooling layers are used to build the CNNs in the early applications of vascular feature extraction ( Chu et al., 2013 ) , whereas fully connected layers are usually added at the end of a CNN as a part of pixel-classification tasks (Section 2.2 ).
The convolutional layers activate the localized vascular features of the image and feature map by using a set of convolutional units ( Nardelli et al., 2018b ; Zreik et al., 2018 ) .
A stack of dilated convolutions is used in convolutional layers ( Wolterink et al., 2019 ; Mou et al., 2019 ) to aggregate the features over multiple scales.
In addition to dilated convolution modules, Nazir et al. (2020) adopted the inception module fusion of residual connection, enabling the network to capture advanced visual information under a controlled computational complexity.
To capture the various shapes and scales of vessels, Jin et al. (2019) integrated the deformation convolution into the network.
After the convolution layers, the pooling layers in the CNNs nonlinearly down-sample the input representation and preserve the feature information in each sub-region.
The pooling layer aids in reducing the number of parameters irrelevant to the problem ( Nardelli et al., 2018b ; Lian et al., 2018 ; Zreik et al., 2018 ) .
However, the scaling operation of the pooling layer is considered to rapidly reduce the already extremely limited information contained in the potentially small patch, causing the classification to be more exigent.
Therefore, Liskowski and Krawiec (2016) constructed a NO-POOL architecture, which performs well on the datasets.
 Tetteh et al. (2017) also found that the pooling operations can lead to the loss of fine local details, which are extremely crucial in pixel-wise tasks.
To solve this problem, they removed all the pooling layers of the CNN, making the feature-extraction layers robust enough to objects of interest of any size.
By employing various size-pooling operations, Gu et al. (2019) used the residual multi-kernel pooling layer that encodes the multi-scale context features without extra learning weights.

 
 
 Feature maps are organized sets of units obtained through convolution operations.
In vessel segmentation, different spatial forms of feature maps can be used in the CNNs.
Using a three-dimensional (3D) CNN, Jin et al. (2017) generated 3D feature maps to learn the structure variations in the 3D space.
They assumed that 3D spatial information (especially the 3D branch-level continuity) and junction-bifurcating patterns are important for segmenting vascular structures.
Owing to high computational demands, they selected a relatively small region of interest (ROI) and trimmed the network.
 Yun et al. (2019) used a 2.5D CNN, which simultaneously takes three adjacent slices in each of the orthogonal directions, including axial, sagittal, and coronal, to improve the segmentation accuracy.
However, they assumed that they could use the 3D CNN to entirely capture the 3D vascular information in its 3D convolutional layers.

 
 
 The feature maps created in the network can be applied for final vessel-tracking tasks.
 Ganin and Lempitsky (2015) found that CNNs are insufficient for learning the mapping from the image patch for vessel annotation, leading to a severe under-fitting during the training and suboptimal performance during the test period.
To resolve this issue, the network maps the input image or patches into intermediate representations using the CNN.
In vessel-segmentation tasks, these mapping results may be the probability maps of vessels.
By applying the sigmoid activation function to the final convolution layer, the CNN output is converted to the probability values in the foreground and background regions.
The final predicted vessel segmentation can be obtained by fusing these probability maps, which describe the probability distributions of vessels and non-vessels.
To predict the vessel boundary, Fu et al. (2016b) utilized the full CNN architecture to output the vessel probability map, followed by the CRF, for a binary segmentation.
 Mo and Zhang (2017) generated a weighted fusion layer by fusing the multi-scale feature maps from each branch output.
In their framework, the probability map is computed using sigmoid functions on the fusion of feature maps.
 Hu et al. (2018a) obtained the probability map using a multi-scale CNN.
By fusing the middle layer feature maps, this CNN model fuses richer multi-scale information for comprehensive feature description to learn more detailed information on the retinal vessels.
 Uslu and Bharath (2019) produced the probability maps of vessel interior, centerline, and edge locations.
The authors assumed that the probability map can better explain the uncertainty and subjectivity of detecting vessels, especially those at the edge locations appearing in the ground truth data.
To formulate the vessel segmentation as a style-transfer problem, Ding et al. (2020b) used the binary probability maps as the tentative training data and the style targets.
Considering that shallower side-outputs capture rich detailed information, and deeper side-outputs have high-level but fuzzy knowledge, Lin et al. (2019) outputted the feature maps of each intermediate layer using VGGNet ( Simonyan and Zisserman, 2015 ) .

 
 
 

#### 3.2.2 Integration of multiple networks

 
 Because the designed forms of feature maps influence the performance of CNNs, feature maps of different layers are fused to further describe the vessels ( Wang et al., 2015 ; Fu et al., 2016b ) .
These feature-map forms can be derived by the various architectures of CNNs.
Based on the number of CNNs used in the vessel-segmentation task, the architecture can be designed as a single CNN or multiple CNNs.

 
 
 A single CNN for vessel tracking can extract meaningful vascular-structures representations; this is regarded as a problem of dimension reduction or sparse coding feature spaces.
For this problem, encoder-decoder architectures (Fig.  3 ) are introduced to encode the hierarchical features.
Instead of transforming the input into another space, Li et al. (2016) developed an auto-encoder network to learn the features, which could recover the input.
The auto-encoder network can be embedded into the CNNs to extract features, which could manage large inter-anatomy variations and thin structures such as renal arteries ( He et al., 2020 ) .
Inspired by the auto-encoder network, Fan and Mo (2017) developed an encoder-decoder style network to learn the mapping functions from the images to the vessels.
By formulating the vessel-tracking problem as a multi-label inference problem, Dasgupta and Singh (2017) used the encoder-decoder framework to learn the class-label dependencies of neighboring pixels.
By employing the skip connections between the encoder and decoder layers, the modified auto-encoder network facilitates the proper memorization of global and local features and alleviates the vanishing gradient problem of deep CNNs.
 Feng et al. (2018) concatenated different feature maps through a skipping connection.
To learn more inherent features from different scales, Guo et al. (2019) further exploited short connections to fuse multiple outputs of side output layers.
To fuse multi-scale features, He et al. (2020) employed a dense biased connection that compresses and transmits the feature maps in each layer to every forward layer.
The authors assumed that this connection can reduce feature redundancy and maintain the integrity of the information and gradient flows.
 Shin et al. (2019) used a graph neural network as a unified CNN to learn the vessel structures.

 
 
 Similar to an encoder-decoder architecture, the U-net ( Ronneberger et al., 2015 ) can extract vascular features using skipping connections (Fig.  3 ).
One feature map generated from a lower layer was concatenated to a corresponding higher layer.
The U-net has been used to segment the coronary arteries in X-ray angiograms ( Fan et al., 2018 ) and liver vessels in CT images ( Huang et al., 2018 ) .
The global contextual information from the low-level features and the spatial details from the previous convolution guide the precise segmentation.
Several methods attempt to efficiently extract or fuse vascular features by improving the structures of U-net ( Chen et al., 2018 ; Kandil et al., 2018 ; Wang et al., 2019 ) .
 Yan et al. (2018) added two separate branches at the end of U-net to simultaneously train the model with the segment-level and the pixel-wise losses.
To improve the robustness and facilitate convergence, Zhang and Chung (2018) applied a residual connection inside each resample block, which added feature maps before the convolution layers.
To reduce the over-fitting problems, the elements in the U-net framework were modified; e.g., adding a dropout layer ( Dharmawan et al., 2019 ) and reducing the number of channels ( Livne et al., 2019 ) .
 Zhang et al. (2019) modified the original U-net by applying an additional convolutional layer before implementing concatenation using the corresponding decoder layer.
This configuration also aids in transferring low-dimensional features to a higher-dimensional space.

 
 
 However, the general U-net may fail to extract some minuscule vessels because this feature accumulation is limited by the depth of U-net ( Jin et al., 2019 ) ; accordingly, modified U-Nets are developed to focus on vascular structures.
 Jin et al. (2019) developed a deformable CNN to capture various vessel shapes and scales via deformable receptive fields, which are adaptive to input features.
To highlight the vessel-like structures, the attention gate (AG) mechanism is introduced into the CNN ( Shen et al., 2019 ; Li et al., 2019 ) .
This AG mechanism can highlight salient features and gradually suppress the characteristic response in unrelated background regions without passing multi-level information ( Shen et al., 2019 ; Li et al., 2019 ) .
 Lian et al. (2019) incorporated a weighted attention mechanism into the U-net framework.
Using this mechanism, the network only focuses on the target ROI and eliminates the irrelevant background noise.
To better learn the global feature information, Ni et al. (2020) introduced the channel attention mechanism when aggregating high-level and shallow features.

 
 
 Different from the single CNN framework, multiple CNNs can be jointly adopted in a framework for vessel tracking. These CNNs can be designed according to different views of the image; e.g., three views (sagittal, coronal, and transverse) of patches ( Kitrungrotsakul et al., 2017 ) .
 Guo et al. (2018) employed multiple CNNs as a voted classifier to improve the performance.
 Zhao et al. (2018a) accomplished the voxel-level vessel segmentation via the hierarchical update of CNNs.
The authors assumed that this network absorbed the learning experience of the previous iteration, which gradually transformed a semi-supervised task into a supervised one.
 Zhang et al. (2019) proposed a more complicated cascade U-net network (i.e., three sub-networks are designed for different detection tasks).

 
 
 
 

### 3.3 Training strategies

 
 The successful training of a useful CNN model relies on a series of strategies.
It is essential to design a suitable training strategy to ensure that the network can focus on vascular regions.
Considering the vascular profiles and features, the training strategies should be carefully designed while considering the following aspects: pre-processing, sampling strategies, and formulation of loss functions.

 
 

#### 3.3.1 Pre-processing

 
 The trained networks tend to perform better on appropriately pre-processed images. The data pre-processing techniques include contrast/brightness normalization, whitening, and augmentation.

 
 
 The image brightness may vary across the fields of view, affectings the network performance.
To resolve this problem, the contrast or brightness normalization abstracts from these fluctuations and further focuses on vessel regions.
A Gaussian kernel is used to homogenize the background ( Vega et al., 2015 ) .
 Liskowski and Krawiec (2016) normalized the patches by subtracting the mean and dividing by the standard deviation of its elements.

 
 
 Similar to principal component analysis (PCA) processing, the whitening processing can remove the universal correlations among the neighbor pixels of the image.
These universal correlations are redundant for training the network. Liskowski and Krawiec (2016) used the zero-phase component analysis to process the image data using rotations, resulting in whitened data that are as close as possible to the original data.
Whitening pre-processing has also been used in ( Marques et al., 2016 ) .

 
 
 Training a CNN for a computer-vision task requires tens of thousands of natural images.
However, for vessel tracking, the number of training images is relatively small, which may cause an over-fitting problem.
To solve this problem, data augmentation can increase the number of training samples using image-transformation approaches (e.g., rotation, scaling, flipping, and mirroring).
These transformations yield the desired invariance and robustness properties of the resulting network.
The augmentation methods vary according to different tasks. Charbonnier et al. (2017) preserved the orientation of patches with four angles and obtained examples using horizontal flipping.
In addition to the rotation and flipping used in ( Zhao et al., 2018b ) , scaling and mirror operations were used in ( Huang et al., 2018 ) and ( Guo et al., 2019 ) , respectively.
Moreover, random elastic deformation was used to deform the training set ( Fan et al., 2018 ; Livne et al., 2019 ) .
 Zreik et al. (2018) observed the signs of over-fitting when training was implemented without data augmentation.
The results in ( Lin et al., 2019 ) show that data augmentation is essential to achieve excellent performance.

 
 
 The application of generative adversarial networks (GANs) is highly promising augmentation approach ( Costa et al., 2018 ; Yu et al., 2019 ) .
By simply sampling a multi-dimensional normal distribution, Costa et al. (2018) employed the encoder-decoder network to generate realistic vessel networks and extend the training samples. Yu et al. (2019) used the shape-consistent GAN to generate synthetic images that maintain the background of coronary angiography and preserve the vascular structures of retinal vessels.
This model can transfer the knowledge of the vessel segmentation from a public dataset to an unlabeled dataset.

 
 
 

#### 3.3.2 Sampling strategies

 
 There are two categories of CNNs according to the input: patch-based and image-based networks.
The former extracts numerous patches from the image data as training samples of the network, whereas the latter considers the entire image as a training sample.

 
 
 For patch-based networks, an efficient extraction strategy should be adopted.
 Nardelli et al. (2018b) extracted the patches from the CT image around the vessel of interest.
 Wolterink et al. (2019) directly selected the positive and negative samples focused on the vessel centerlines.
Instead of extracting the patches, images can be directly inputted into the network to optimize the model; examples can be found in ( Hu et al., 2018a ; Mo and Zhang, 2017 ) .
More flexible, Girard et al. (2019) used a scalable encoding-decoding CNN model that can input either the entire image directly or patches of any size to the network.

 
 
 In addition to the samples directly extracted from the image, several works selected the training samples from enhanced images to focus on problems originating from vessel tracking.
For example, Nardelli et al. (2017) ; Nardelli et al. (2018a) extracted patches from the bronchus image enhanced by the scale-space particle approach. Hajabdollahi et al. (2018) trained the CNN on the enhanced gray-scale level image.
 Zhao et al. (2018a) selected the patches from both the original and tube-level label images.
To directly reflect the stenosis of vessels, Zreik et al. (2018) collected the patches from multi-planar reformatted images.

 
 
 

#### 3.3.3 Formulation of loss functions

 
 The CNNs obtain the optimal network weights by optimizing the loss function.
The cross-entropy-based loss functions are generally used in vessel-tracking tasks ( Dasgupta and Singh, 2017 ; Nardelli et al., 2018b ; Wu et al., 2018 ; Jin et al., 2019 ; Dharmawan et al., 2019 ; Mo and Zhang, 2017 ; Guo et al., 2019 ; Lin et al., 2019 ) .
However, the vascular regions in the images are considerably smaller than the non-vascular regions, thereby inducing the imbalance problem for loss-function optimization.

 
 
 Data-imbalance problems occur in image segmentation, where the number of foreground pixels is usually less than background.
To resolve the imbalance problem, some researchers formulated weighted schemes for the categorical cross-entropy loss functions ( Li et al., 2018 ; Zhang and Chung, 2018 ; Hu et al., 2018a ) .
These loss functions incorporate the coefficients to reduce the importance of well-classified examples and focus on problematic samples.
Another solution that has been employed is the use of the dice-coefficient-based loss functions ( Livne et al., 2019 ; Soomro et al., 2019 ; Kitrungrotsakul et al., 2019 ) .
To balance the classes of voxels, weighted schemes of the dice coefficient are employed to formulate the loss functions.
 Huang et al. (2018) adjusted the penalty weights of misclassified voxels to obtain higher correct classification scores and lower number of misclassified voxels.
 Lian et al. (2018) employed a tuning parameter to determine whether precision (i.e., positive prediction value) contributes more than recall (i.e., true positive rate or sensitivity) or conversely during the training procedure.

 
 
 Alternative methods employ the squared error ( Li et al., 2016 ; Fan and Mo, 2017 ) or more complex formulations as loss functions to train CNNs.
Based on the L1 norm, Yan et al. (2018) generated a joint loss to simultaneously train the model with the vessel-segment-level loss and pixel-wise loss.
 Jiang et al. (2019) formulated the loss function by adjusting the weights of two parts: cross entropy and L2 norm.

 
 
 
 
 

## 4 Evaluation issues

 

### 4.1 Metrics for performance evaluation

 
 The results of vessel tracking can be presented as key points, vessel centerlines, and label images depending on requirements of the clinical applications.
The performances of the methods are evaluated by comparing the results with the ground truth.
The key points can be checked visually.
For the remaining two types of results, two groups of metrics are generally used: overlap metrics and classification metrics.

 
 
 Overlap metrics are used to evaluate the similarity between the extracted vessels and ground truth.
For label images, true positive (TP), true negative (TN), false negative (FN), and false positive (FP), compared with the ground truth, are typically used to evaluate the vessel and non-vessel patterns;
these four metrics can be further formulated as accuracy (Acc), sensitivity (Se), specificity (Sp), precision (Pr), recall (Re), positive predictive value (PPV), negative predictive value (NPV), and dice similarity coefficient (DSC ( Dice, 1945 ) ).
The DSC and the Hausdorff distance (HD) are widely adopted overlap metrics to assess the similarity between label images.
For centerlines, the four metrics can be computed according to the point-to-point correspondence between the ground truth and computed centerline.
The four metrics can be further formulated as overlap (OV), overlap until the first error (OF), and overlap with the clinically relevant part of the vessel (OT).
The average inside (AI) distance can also be used to describe the average distance of connections between two centerlines.
The details of the four metrics for assessing vessel centerlines are found in ( Schaap et al., 2009 ) .

 
 
 Classification metrics are derived from the curves (i.e., receiver operating characteristic (ROC) curve and precision-recall curve) to assess a binary classifier system.
The area under the receiver operating characteristic curve (AUC) metric can be used to indicate the probability that a classifier will rank a randomly chosen vessel instance higher than it will rank a randomly chosen non-vessel instance.
The AUPRC metric, i.e., the area under the precision-recall curve, can also be exploited to evaluate the results of vessel tracking.

 
 
 

### 4.2 Public datasets and validation strategies

 
 Standard datasets are required for an objective evaluation of vessel-tracking methods.
Here, we summarize the challenges and the public datasets related to vessel tracking.

 
 (1) 
 
 Retinal vessel segmentation: DRIVE 
 ( http://www.isi.uu.nl/Research/Databases/DRIVE/ ) 

 

 (2) 
 
 Retinal vessel segmentation: STARE 
 ( http://cecas.clemson.edu/~ahoover/stare/ ) 

 

 (3) 
 
 Retinal vessel segmentation: CHASE-DB1 
 ( http://blogs.kingston.ac.uk/retinal/chasedb1 ) 

 

 (4) 
 
 Retinal vessel segmentation: HRF ( Odstrcilik et al., 2013 ) 

 

 (5) 
 
 Retinal vessel segmentation: PRIME-FP20 ( Ding et al., 2020a ) 

 

 (6) 
 
 Coronary artery stenosis detection: CASDQEF 
 ( http://coronary.bigr.nl/stenoses ) 

 

 (7) 
 
 Coronary centerline extraction: CAT08 
 ( http://coronary.bigr.nl/centerlines ) 

 

 (8) 
 
 Identify coronary calcifications: orCaScore 
 ( https://orcascore.grand-challenge.org/ ) 

 

 (9) 
 
 Coronary segmentation: ASOCA 
 ( https://asoca.grand-challenge.org/ ) 

 

 (10) 
 
 Lung vessel segmentation: VESSEL12 ( Rudyanto et al., 2014 ) 

 

 (11) 
 
 Liver segmentation: SLIVER07 
 ( https://sliver07.grand-challenge.org/ ) 

 

 (12) 
 
 Liver vessel segmentation: 3D-IRCADb 
 ( https://www.ircad.fr/research/3d-ircadb-01/ ) 

 

 
 
 
 To validate the vessel-tracking methods, the dataset is divided into different training and test groups according, e.g., one-off train + test, leave-one-out and k-fold cross-validation.
The experiment columns in Tables  3 - 7 present the validation strategies of the selected methods.
Moreover, Table  9 presents the public datasets and selected state-of-the-art results.

 
 
 Table 9: The public datasets and selected results. Please refer to Section 4 for the abbreviations. 
 
 
 
 
 
 Public dataset 
 | 
 
 
 Selected Results 
 | 
 
 
 Validation strategy 
 | 

 
 
 
 
 
 DRIVE 
 | 
 
 
 Acc=0.9692 ( Lian et al., 2019 ) 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 

 
 | 
 
 
 AUC=0.9814 ( Cherukuri et al., 2020 ) 
 | 
 
 
 20 for training, 20 for testing, five-fold cross-validation 
 | 

 
 | 
 
 
 Se=0.9382 ( Shin et al., 2019 ) 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 

 
 | 
 
 
 Sp=0.9861 ( Lian et al., 2019 ) 
 | 
 
 
 20 for training, 20 for testing, one-off train + test 
 | 

 
 
 
 STARE 
 | 
 
 
 Acc=0.9740 ( Lian et al., 2019 ) 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 

 
 | 
 
 
 AUC=0.9882 ( Zhang and Chung, 2018 ) 
 | 
 
 
 15 for training, 5 for testing, four-fold cross-validation 
 | 

 
 | 
 
 
 Se=0.9598 ( Shin et al., 2019 ) 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 

 
 | 
 
 
 Sp=0.9916 ( Lian et al., 2019 ) 
 | 
 
 
 10 for training, 10 for testing, one-off train + test 
 | 

 
 
 
 CHASE-DB1 
 | 
 
 
 Acc=0.9770 ( Zhang and Chung, 2018 ) 
 | 
 
 
 21 for training, 7 for testing, four-fold cross-validation 
 | 

 
 | 
 
 
 AUC=0.9900 ( Zhang and Chung, 2018 ) 
 | 
 
 
 21 for training, 7 for testing, four-fold cross-validation 
 | 

 
 | 
 
 
 Se=0.9463 ( Shin et al., 2019 ) 
 | 
 
 
 20 for training, 8 for testing, one-off train + test 
 | 

 
 | 
 
 
 Sp=0.9909 ( Lian et al., 2019 ) 
 | 
 
 
 21 for training, 7 for testing, four-fold cross-validation 
 | 

 
 
 
 HRF 
 | 
 
 
 Acc=0.9651 ( Jin et al., 2019 ) 
 | 
 
 
 15 for training, 30 for testing, one-off train + test 
 | 

 
 | 
 
 
 AUC=0.9838 ( Shin et al., 2019 ) 
 | 
 
 
 15 for training, 30 for testing, one-off train + test 
 | 

 
 | 
 
 
 Se=0.9546 ( Shin et al., 2019 ) 
 | 
 
 
 15 for training, 30 for testing, one-off train + test 
 | 

 
 | 
 
 
 Sp=0.9823 ( Zhao et al., 2018b ) 
 | 
 
 
 22 for training, 23 for testing, one-off train + test 
 | 

 
 
 
 CAT08 
 | 
 
 
 AI=0.21mm ( Wolterink et al., 2019 ) 
 | 
 
 
 7 for training, 1 for testing, leave-one-out 
 | 

 
 | 
 
 
 OF=0.815 ( Wolterink et al., 2019 ) 
 | 
 
 
 7 for training, 1 for testing, leave-one-out 
 | 

 
 | 
 
 
 OT=0.971 ( Schaap et al., 2011 ) 
 | 
 
 
 8 for training, 24 for testing, one-off train + test 
 | 

 
 | 
 
 
 OV=0.969 ( Schaap et al., 2011 ) 
 | 
 
 
 8 for training, 24 for testing, one-off train + test 
 | 

 

 
 
 
 

## 5 Conclusion and discussion

 
 We have reviewed the recent literature on vessel tracking, particularly those on the methodologies that apply machine learning, including conventional machine-learning and deep-learning algorithms.
Instead of reviewing the methods for a single application (e.g., retinal vessel segmentation and coronary centerline extraction) or based on a specific imaging modality (e.g., colored image and CTA), this paper focuses on reviewing the learning-based methods of tracking vessels of various organs in different imaging modalities.
Learning-based methods offer the advantages of mapping the input data into representative and discriminative vascular features.
Particularly, conventional learning-based methods learn the vessel-dedicated information from numerous hand-crafted features.
They can employ different classifiers to distinguish the vessels from an analogous background according to the learnt features.
Moreover, these techniques can describe the vessels with learnt vessel-dedicated parameters using statistical models.
In contrast, based on various CNN architectures, deep-learning-based methods leverage hierarchical features that can encode global and local vascular structures.

 
 
 Owing to the complex morphologies of objects and image characteristics, vessel tracking is an exigent task.
Thin vessels are not observed in many vessel-segmentation tasks because of their complex structures and small sizes, and to distinguish the small-sized vessels from artifacts and noise, high-quality local textures are required.
Moreover, vessels with uncertain branches and tortuosity are difficult to track because of the complex branch connections.
More auxiliary information (e.g., key points and orientations) should be obtained to reconstruct these vessels.
Specifically, surrounding tissues and image noise may interfere with vessel tracking because of their positions and image intensities. To reduce the interference, a series of pre-processing techniques should be considered.

 
 
 The recent literature on vessel tracking mainly reports the advanced machine-learning methodologies in view of their considerable modeling capacities and potential in extracting effective features. Nevertheless, the following two problems should be considered when a new algorithm for this task is developed.

 
 
 First, learning-based methods may deliver limited performance in tracking a complete vessel because of the lack of high-level vascular features (e.g., branches and connections).
The models can be trained based on hand-crafted features (Section 2 ) or hierarchical features (Section 3 ).
However, in clinical practice, developing vessel-tracking methods may be exigent because of the problems involved in detecting abnormal vessels or vessels with pathologies.
To describe these vessels in detail, the extraction of high-level features for future learning-based tracking methods is required.

 
 
 The second problem is related to the strategies employed to deal with the limited training data because this insufficiency generally leads to poor generalization capacity of models.
Deep learning has achieved considerable success in many applications where public datasets with annotation are available.
However, in the field of medical image analysis, overcoming the limitation of training data is still a major challenge.
Currently, data augmentation is a common strategy employed to alleviate this issue.
Moreover, weakly supervised and self-supervised learning are potential approaches to resolve the problems of lacking annotated data.
Hence, in the future, more public databases are expected to be available, such as via open challenges, to promote the learning-based vessel-tracking algorithms.

 
 
 

## Acknowledgment

 
 This work was funded by the National Natural Science Foundation of China (grant no. 61971142 and 62011540404) and the development fund for Shanghai talents (no. 2020015)

 
 
 

## References

 
 
 A. Osareh (2009) 
 
A. Osareh, B.S., 2009.

 
 Automatic blood vessel segmentation in color images
of retina.

 
 Iranian Journal Of Science And Technology
Transaction B – Engineering 33,
191–206.

 

 
 Abramoff et al. (2010) 
 
Abramoff, M.D., Garvin, M.K.,
Sonka, M., 2010.

 
 Retinal imaging and image analysis.

 
 IEEE Reviews in Biomedical Engineering
3, 169–208.

 

 
 Agam et al. (2005) 
 
Agam, G., Armato, S.G.,
Wu, C., 2005.

 
 Vessel tree reconstruction in thoracic CT scans with
application to nodule detection.

 
 IEEE Transactions on Medical Imaging
24, 486–499.

 

 
 Al-Diri et al. (2008) 
 
Al-Diri, B., Hunter, A.,
Steel, D., Habib, M.,
Berry, S., 2008.

 
 REVIEW-a reference data set for retinal vessel
profiles, in: International Conference of the IEEE
Engineering in Medicine and Biology Society.

 

 
 Annunziata et al. (2015a) 
 
Annunziata, R., Kheirkhah, A.,
Hamrah, P., Trucco, E.,
2015a.

 
 Boosting Hand-Crafted Features for Curvilinear
Structure Segmentation by Learning Context Filters, in:
Medical Image Computing and Computer-Assisted
Intervention, Pt Iii. volume 9351, pp.
596–603.

 

 
 Annunziata et al. (2015b) 
 
Annunziata, R., Kheirkhah, A.,
Hamrah, P., Trucco, E.,
2015b.

 
 Scale and curvature invariant ridge detector for
tortuous and fragmented structures, in: Medical Image
Computing and Computer Assisted Intervention (MICCAI),
Springer, Cham. pp. 588–595.

 

 
 Annunziata and Trucco (2016) 
 
Annunziata, R., Trucco, E.,
2016.

 
 Accelerating Convolutional Sparse Coding for
Curvilinear Structures Segmentation by Refining SCIRD-TS Filter Banks.

 
 IEEE Transactions on Medical Imaging
35, 2381–2392.

 

 
 Asl et al. (2017) 
 
Asl, M.E., Koohbanani, N.A.,
Frangi, A.F., Gooya, A.,
2017.

 
 Tracking and diameter estimation of retinal vessels
using Gaussian process and Radon transform.

 
 Journal of Medical Imaging 4,
1.

 

 
 Azzopardi and Petkov (2013) 
 
Azzopardi, G., Petkov, N.,
2013.

 
 Trainable COSFIRE Filters for Keypoint Detection and
Pattern Recognition.

 
 IEEE Transactions on Pattern Analysis and Machine
Intelligence 35, 490–503.

 

 
 Becker et al. (2013) 
 
Becker, C., Rigamonti, R.,
Lepetit, V., Fua, P.,
2013.

 
 Supervised feature learning for curvilinear
structure segmentation.

 
 Medical Image Computing and Computer Assisted
Intervention (MICCAI) 8149 LNCS,
526–533.

 

 
 Bogunović et al. (2012) 
 
Bogunović, H., Pozo, J.M.,
Cárdenes, R., Villa-Uriol, M.C.,
Blanc, R., Piotin, M.,
Frangi, A.F., Bogunovic, H.,
Pozo, J.M., Cardenes, R.,
Villa-Uriol, M.C., Blanc, R.,
Piotin, M., Frangi, A.F.,
2012.

 
 Automated landmarking and geometric characterization
of the carotid siphon.

 
 Medical Image Analysis 16,
889–903.

 

 
 Breiman (1996) 
 
Breiman, L., 1996.

 
 Bagging predictors.

 
 Machine Learning 24,
123–140.

 

 
 Chai et al. (2013) 
 
Chai, D., Forstner, W.,
Lafarge, F., 2013.

 
 Recovering line-networks in images by junction-point
processes.

 
 Proceedings of the IEEE Computer Society Conference
on Computer Vision and Pattern Recognition , 1894–1901.

 

 
 Chapman et al. (2015) 
 
Chapman, B.E., Berty, H.P.,
Schulthies, S.L., 2015.

 
 Automated generation of directed graphs from
vascular segmentations.

 
 Journal of Biomedical Informatics
56, 395–405.

 

 
 Charbonnier et al. (2017) 
 
Charbonnier, J.P., Rikxoort, E.M.,
Setio, A.A., Schaefer-Prokop, C.M.,
van Ginneken, B., Ciompi, F.,
2017.

 
 Improving airway segmentation in computed tomography
using leak detection with convolutional networks.

 
 Medical Image Analysis 36,
52–60.

 

 
 Chen et al. (2015) 
 
Chen, J., Li, M., Jin,
Q., Bao, S., Su, Z.,
Yu, Y., 2015.

 
 Lead curve detection in drawings with complex
cross-points.

 
 Neurocomputing 168,
35–46.

 

 
 Chen et al. (2018) 
 
Chen, X.H., Lu, Y., Bai,
J.J., Yin, Y.B., Cao, K.L.,
Li, Y.W., Chen, H.B.,
Song, Q., Wu, J., 2018.

 
 Train a 3d u-net to segment cranial vasculature in
cta volume without manual annotation, in: 2018 IEEE
15th International Symposium on Biomedical Imaging, pp.
559–563.

 

 
 Cheng et al. (2012) 
 
Cheng, J.Z., Chen, C.M.,
Cole, E.B., Pisano, E.D.,
Shen, D.G., 2012.

 
 Automated Delineation of Calcified Vessels in
Mammography by Tracking With Uncertainty and Graphical Linking Techniques.

 
 IEEE Transactions on Medical Imaging
31, 2143–2155.

 

 
 Cherry et al. (2015) 
 
Cherry, K.M., Peplinski, B.,
Kim, L., Wang, S.J., Lu,
L., Zhang, W.D., Liu, J.F.,
Wei, Z.S., Summers, R.M.,
2015.

 
 Sequential Monte Carlo tracking of the marginal
artery by multiple cue fusion and random forest regression.

 
 Medical Image Analysis 19,
164–175.

 

 
 Cherukuri et al. (2020) 
 
Cherukuri, V., Bg, V.K.,
Bala, R., Monga, V.,
2020.

 
 Deep Retinal Image Segmentation with Regularization
under Geometric Priors.

 
 IEEE Transactions on Image Processing
29, 2552–2567.

 

 
 Chu et al. (2013) 
 
Chu, P., Pang, Y., Cheng,
E., Zhu, Y., Zheng, Y.,
2013.

 
 Structure-Aware Rank-1 Tensor Approximation for
Curvilinear Structure Tracking Using Learned Hierarchical Features.

 
 Medical Image Computing and Computer Assisted
Intervention (MICCAI) 8150, 413–421.

 

 
 Çimen et al. (2016) 
 
Çimen, S., Gooya, A.,
Grass, M., Frangi, A.F.,
2016.

 
 Reconstruction of coronary arteries from X-ray
angiography: A review.

 
 Medical Image Analysis 32,
46–68.

 

 
 Coates and Ng (2012) 
 
Coates, A., Ng, A.Y., 2012.

 
 Learning Feature Representations with K-Means ,
561–580.

 

 
 Costa et al. (2018) 
 
Costa, P., Galdran, A.,
Meyer, M.I., Niemeijer, M.,
Abràmoff, M., Mendonça, A.M.,
Campilho, A., 2018.

 
 End-to-End Adversarial Retinal Image Synthesis.

 
 IEEE Transactions on Medical Imaging
37, 781–791.

 

 
 Cutler et al. (2001) 
 
Cutler, A., Cutler, D.R.,
Stevens, J.R., 2001.

 
 Random forests.

 
 Machine Learning 45,
5–32.

 

 
 Dasgupta and Singh (2017) 
 
Dasgupta, A., Singh, S.,
2017.

 
 A fully convolutional neural network based
structured prediction approach towards the retinal vessel segmentation.

 
 Proceedings - International Symposium on Biomedical
Imaging , 248–251.

 

 
 Deng et al. (2018) 
 
Deng, X., Zheng, Y., Xu,
Y., Xi, X., Li, N.,
Yin, Y., 2018.

 
 Graph cut based automatic aorta segmentation with an
adaptive smoothness constraint in 3D abdominal CT images.

 
 Neurocomputing 310,
46–58.

 

 
 Dharmawan et al. (2019) 
 
Dharmawan, D.A., Li, D.,
Ng, B.P., Rahardja, S.,
2019.

 
 A New Hybrid Algorithm for Retinal Vessels
Segmentation on Fundus Images.

 
 IEEE Access 7,
41885–41896.

 

 
 Dice (1945) 
 
Dice, L.R., 1945.

 
 Measures of the amount of ecologic association
between species.

 
 Ecology 26.

 

 
 Ding et al. (2020a) 
 
Ding, L., Kuriyan, A.E.,
Ramchandran, R.S., Wykoff, C.C.,
Sharma, G., 2020a.

 
 PRIME-FP20: Ultra-widefield fundus photography
vessel segmentation dataset.

 

 
 Ding et al. (2020b) 
 
Ding, L., Kuriyan, A.E.,
Ramchandran, R.S., Wykoff, C.C.,
Sharma, G., 2020b.

 
 Weakly-supervised vessel detection in ultra-widefield
fundus photography via iterative multi-modal registration and learning.

 
 IEEE Transactions on Medical Imaging ,
1–1.

 

 
 Fan et al. (2018) 
 
Fan, J., Yang, J., Wang,
Y., Yang, S., Ai, D.,
Huang, Y., Song, H.,
Hao, A., Wang, Y., 2018.

 
 Multichannel Fully Convolutional Network for
Coronary Artery Segmentation in X-Ray Angiograms.

 
 IEEE Access 6,
44635–44643.

 

 
 Fan and Mo (2017) 
 
Fan, Z., Mo, J.J., 2017.

 
 Automated blood vessel segmentation based on
de-noising auto-encoder and neural network.

 
 Proceedings - International Conference on Machine
Learning and Cybernetics 2, 849–856.

 

 
 Feng et al. (2018) 
 
Feng, Z., Yang, J., Yao,
L., 2018.

 
 Patch-based fully convolutional neural network with
skip connections for retinal blood vessel segmentation.

 
 Proceedings - International Conference on Image
Processing, ICIP 2017-Septe, 1742–1746.

 

 
 Frangi et al. (1998) 
 
Frangi, A.F., Niessen, W.J.,
Vincken, K.L., Viergever, M.a.,
1998.

 
 Multiscale vessel enhancement filtering.

 
 International Conference on Medical Image Computing
and Computer-assisted Intervention , 130–137.

 

 
 Fraz et al. (2012a) 
 
Fraz, M.M., Barman, S.A.,
Remagnino, P., Hoppe, A.,
Basit, A., Uyyanonvara, B.,
Rudnicka, A.R., Owen, C.G.,
2012a.

 
 An approach to localize the retinal blood vessels
using bit planes and centerline detection.

 
 Computer methods and programs in biomedicine
108, 600–616.

 

 
 Fraz et al. (2012b) 
 
Fraz, M.M., Remagnino, P.,
Hoppe, A., Uyyanonvara, B.,
Rudnicka, A.R., Owen, C.G.,
Barman, S.A., 2012b.

 
 Blood vessel segmentation methodologies in retinal
images–A survey.

 
 Computer methods and programs in biomedicine
108, 407–433.

 

 
 Freund and Schapire (1995) 
 
Freund, Y., Schapire, R.E.,
1995.

 
 A desicion-theoretic generalization of on-line
learning and an application to boosting, pp. 23–37.

 

 
 Fu et al. (2016a) 
 
Fu, H., Xu, Y., Lin, S.,
Wing, D., Wong, K., Liu,
J., 2016a.

 
 DeepVessel: Retinal Vessel Segmentation via Deep
Learning and Conditional Random Field, in: Medical
Image Computing and Computer Assisted Intervention (MICCAI), pp.
132–139.

 

 
 Fu et al. (2016b) 
 
Fu, H.Z., Xu, Y.W., Wong,
D.W.K., Liu, J., 2016b.

 
 Retinal vessel segmentation via deep learning
network and fully-connected conditional random fields, in:
2016 IEEE 13th International Symposium on Biomedical
Imaging. volume 2016-June, pp. 698–701.

 

 
 Ganin and Lempitsky (2015) 
 
Ganin, Y., Lempitsky, V.,
2015.

 
 N4-Fields: Neural Network Nearest Neighbor Fields
for Image Transforms.

 
 Asian Conference on Computer Vision
9005, 536–551.

 

 
 Girard et al. (2019) 
 
Girard, F., Kavalec, C.,
Cheriet, F., 2019.

 
 Joint segmentation and classification of retinal
arteries/veins from fundus images.

 
 Artificial intelligence in medicine
94, 96–109.

 

 
 Goceri et al. (2017) 
 
Goceri, E., Shah, Z.K.,
Gurcan, M.N., 2017.

 
 Vessel segmentation from abdominal magnetic
resonance images: adaptive and reconstructive approach.

 
 International Journal for Numerical Methods in
Biomedical Engineering 33, 1–16.

 

 
 Gu et al. (2017) 
 
Gu, L., Zhang, X., Zhao,
H., Li, H., Cheng, L.,
2017.

 
 Segment 2D and 3D Filaments by Learning Structured
and Contextual Features.

 
 IEEE Transactions on Medical Imaging
36, 596–606.

 

 
 Gu et al. (2019) 
 
Gu, Z., Cheng, J., Fu,
H., Zhou, K., Hao, H.,
Zhao, Y., Zhang, T.,
Gao, S., Liu, J., 2019.

 
 CE-Net: Context Encoder Network for 2D Medical Image
Segmentation.

 
 IEEE Transactions on Medical Imaging
38, 2281–2292.

 

 
 Guo et al. (2019) 
 
Guo, S., Wang, K., Kang,
H., Zhang, Y., Gao, Y.,
Li, T., 2019.

 
 BTS-DSN: Deeply supervised neural network with short
connections for retinal vessel segmentation.

 
 International Journal of Medical Informatics
126, 105–113.

 

 
 Guo et al. (2018) 
 
Guo, Y., Budak, Ü.,
Şengür, A., 2018.

 
 A novel retinal vessel detection approach based on
multiple deep convolution neural networks.

 
 Computer Methods and Programs in Biomedicine
167, 43–48.

 

 
 Haddad et al. (2018) 
 
Haddad, C.W., Carroll, T.J.,
Christoforidis, G.A., Giger, M.L.,
Haddad, C.W., Drukker, K.,
Gullett, R., Carroll, T.J.,
Christoforidis, G.A., Haddad, C.W.,
Drukker, K., Gullett, R.,
Carroll, T.J., Christoforidis, G.A.,
Giger, M.L., 2018.

 
 Fuzzy c-means segmentation of major vessels in
angiographic images of stroke.

 
 Journal of Medical Imaging 5,
1.

 

 
 Hajabdollahi et al. (2018) 
 
Hajabdollahi, M., Esfandiarpoor, R.,
Najarian, K., Karimi, N.,
Samavi, S., Reza-Soroushmeh, S.M.,
2018.

 
 Low complexity convolutional neural network for
vessel segmentation in portable retinal diagnostic devices.

 
 Proceedings - International Conference on Image
Processing, ICIP , 2785–2789.

 

 
 Hameeteman et al. (2011) 
 
Hameeteman, K., Zuluaga, M.A.,
Freiman, M., Joskowicz, L.,
Cuisenaire, O., Valencia, L.F.,
Gülsün, M.A., Krissian, K.,
Mille, J., Wong, W.C.K.,
Orkisz, M., Tek, H.,
Hoyos, M.H., Benmansour, F.,
Chung, A.C.S., Rozie, S.,
Gils, M.V., Borne, L.V.D.,
Sosna, J., Berman, P.,
Cohen, N., Douek, P.C.,
Sánchez, I., Aissat, M.,
Schaap, M., Metz, C.T.,
Krestin, G.P., Lugt, A.V.D.,
Niessen, W.J., Walsum, T.V.,
2011.

 
 Evaluation framework for carotid bifurcation lumen
segmentation and stenosis grading.

 
 Medical Image Analysis 15,
477–488.

 

 
 Hanaoka et al. (2015) 
 
Hanaoka, S., Nomura, Y.,
Nemoto, M., Miki, S.,
Yoshikawa, T., Hayashi, N.,
Ohtomo, K., Masutani, Y.,
Shimizu, A., 2015.

 
 HoTPiG: A Novel Geometrical Feature for Vessel
Morphometry and Its Application to Cerebral Aneurysm Detection, pp.
103–110.

 

 
 Hashemzadeh and Adlpour Azar (2019) 
 
Hashemzadeh, M., Adlpour Azar, B.,
2019.

 
 Retinal blood vessel extraction employing effective
image features and combination of supervised and unsupervised machine
learning methods.

 
 Artificial Intelligence in Medicine
95, 1–15.

 

 
 He et al. (2020) 
 
He, Y., Yang, G., Yang,
J., Chen, Y., Kong, Y.,
Wu, J., Tang, L., Zhu,
X., Dillenseger, J.L., Shao, P.,
Zhang, S., Shu, H.,
Coatrieux, J.L., Li, S.,
2020.

 
 Dense biased networks with deep priori anatomy and
hard region adaptation: Semi-supervised learning for fine renal artery
segmentation.

 
 Medical image analysis 63,
101722.

 

 
 Hu et al. (2017) 
 
Hu, C., Hui, H., Wang,
S., Dong, D., Liu, X.,
Yang, X., Tian, J., 2017.

 
 Cerebral vessels segmentation for light-sheet
microscopy image using convolutional neural networks.

 
 Medical Imaging 2017: Biomedical Applications in
Molecular, Structural, and Functional Imaging 10137,
101370K.

 

 
 Hu et al. (2018a) 
 
Hu, K., Zhang, Z.Z., Niu,
X.R., Zhang, Y., Cao, C.H.,
Xiao, F., Gao, X.P.,
2018a.

 
 Retinal vessel segmentation of color fundus images
using multiscale convolutional neural network with an improved cross-entropy
loss function.

 
 Neurocomputing 309,
179–191.

 

 
 Hu et al. (2018b) 
 
Hu, X., Cheng, Y., Ding,
D., Chu, D., 2018b.

 
 Axis-Guided Vessel Segmentation Using a
Self-Constructing Cascade-AdaBoost-SVM Classifier.

 
 BioMed Research International
2018, 1–12.

 

 
 Huang et al. (2018) 
 
Huang, Q., Sun, J.F.,
Ding, H., Wang, X.D.,
Wang, G.Z., 2018.

 
 Robust liver vessel extraction using 3D U Net with
variant dice loss function.

 
 Computers in Biology and Medicine
101, 153–162.

 

 
 Jassi (2010) 
 
Jassi, H.P., 2010.

 
 Vascusynth: Simulating vascular trees for generating
volumetric image data with ground-truth segmentation and tree analysis.

 
 Computerized Medical Imaging and Graphics .

 

 
 Javidi et al. (2017) 
 
Javidi, M., Pourreza, H.R.,
Harati, A., 2017.

 
 Vessel segmentation and microaneurysm detection
using discriminative dictionary learning and sparse representation.

 
 Computer Methods and Programs in Biomedicine
139, 93–108.

 

 
 Jawaid et al. (2017) 
 
Jawaid, M.M., Riaz, A.,
Rajani, R., Reyes-Aldasoro, C.C.,
Slabaugh, G., 2017.

 
 Framework for detection and localization of coronary
non-calcified plaques in cardiac CTA using mean radial profiles.

 
 Computers in Biology and Medicine
89, 84–95.

 

 
 Jiang et al. (2019) 
 
Jiang, Y., Tan, N., Peng,
T.T., Zhang, H., 2019.

 
 Retinal Vessels Segmentation Based on Dilated
Multi-Scale Convolutional Neural Network.

 
 IEEE Access 7,
76342–76352.

 

 
 Jin et al. (2017) 
 
Jin, D.K., Xu, Z.Y.,
Harrison, A.P., George, K.,
Mollura, D.J., 2017.

 
 3D Convolutional Neural Networks with Graph
Refinement for Airway Segmentation Using Incomplete Data Labels, in:
Machine Learning in Medical Imaging. volume
10541, pp. 141–149.

 

 
 Jin et al. (2019) 
 
Jin, Q., Meng, Z., Pham,
T.D., Chen, Q., Wei, L.,
Su, R., 2019.

 
 DUNet: A deformable network for retinal vessel
segmentation.

 
 Knowledge-Based Systems .

 

 
 Kalaie and Gooya (2017) 
 
Kalaie, S., Gooya, A.,
2017.

 
 Vascular tree tracking and bifurcation points
detection in retinal images using a hierarchical probabilistic model.

 
 Computer Methods and Programs in Biomedicine
151, 139–149.

 

 
 Kande et al. (2010) 
 
Kande, G.B., Subbaiah, P.V.,
Savithri, T.S., 2010.

 
 Unsupervised Fuzzy Based Vessel Segmentation In
Pathological Digital Fundus Images.

 
 Journal of Medical Systems 34,
849–858.

 

 
 Kandil et al. (2018) 
 
Kandil, H., Soliman, A.,
Taher, F., Mahmoud, A.,
Elmaghraby, A., El-Baz, A.,
2018.

 
 Using 3-D CNNs and Local Blood Flow Information to
Segment Cerebral Vasculature, in: 2018 IEEE
International Symposium on Signal Processing and Information Technology, pp.
701–705.

 

 
 Kang et al. (2015) 
 
Kang, D., Ko, H., Kuo,
C.C.J., Dey, D., Li, D.,
Slomka, P.J., Arsanjani, R.,
Nakazato, R., Berman, D.S.,
2015.

 
 Structured learning algorithm for detection of
nonobstructive and obstructive coronary plaque lesions from computed
tomography angiography.

 
 Journal of Medical Imaging 2.

 

 
 Kerrien et al. (2017) 
 
Kerrien, E., Yureidini, A.,
Dequidt, J., Duriez, C.,
Anxionnat, R., Cotin, S.,
2017.

 
 Blood vessel modeling for interactive simulation of
interventional neuroradiology procedures.

 
 Medical Image Analysis 35,
685–698.

 

 
 Khan et al. (2019) 
 
Khan, K.B., Khaliq, A.A.,
Jalil, A., Iftikhar, M.A.,
Ullah, N., Aziz, M.W.,
Ullah, K., Shahid, M.,
2019.

 
 A review of retinal blood vessels extraction
techniques: challenges, taxonomy, and future trends.

 
 Pattern Analysis and Applications
22, 767–802.

 

 
 Khan and NaliniPriya (2016) 
 
Khan, Z.F., NaliniPriya, G.,
2016.

 
 Automatic Segmentation of Retinal Blood Vessels
Employing Textural Fuzzy C-Means Clustering.

 

 
 Khowaja et al. (2016) 
 
Khowaja, S.A., Unar, M.A.,
Ismaili, I.A., Khuwaja, P.,
2016.

 
 Supervised method for blood vessel segmentation from
coronary angiogram images using 7-D feature vector.

 
 Imaging Science Journal 64,
196–203.

 

 
 Kirbas and Francis (2004) 
 
Kirbas, C., Francis, Q.,
2004.

 
 A Review of Vessel Extraction Techniques and
Algorithms.

 
 Computing Surveys 36,
81–121.

 

 
 Kirisli et al. (2013) 
 
Kirisli, H.A., Schaap, M.,
Metz, C.T., Dharampal, A.S.,
Meijboom, W.B., Papadopoulou, S.L.,
Dedic, A., Nieman, K.,
de Graaf, M.A., Meijs, M.F.L.,
Cramer, M.J., Broersen, A.,
Cetin, S., Eslami, A.,
Florez-Valencia, L., Lor, K.L.,
Matuszewski, B., Melki, I.,
Mohr, B., Oksuz, I.,
Shahzad, R., Wang, C.,
Kitslaar, P.H., Unal, G.,
Katouzian, A., Orkisz, M.,
Chen, C.M., Precioso, F.,
Najman, L., Masood, S.,
Unay, D., Van Vliet, L.,
Moreno, R., Goldenberg, R.,
Vucini, E., Krestin, G.P.,
Niessen, W.J., van Walsum, T.,
2013.

 
 Standardized evaluation framework for evaluating
coronary artery stenosis detection, stenosis quantification and lumen
segmentation algorithms in computed tomography angiography.

 
 Medical Image Analysis 17,
859–876.

 

 
 Kitrungrotsakul et al. (2017) 
 
Kitrungrotsakul, T., Han, X.H.,
Iwamoto, Y., Foruzan, A.H.,
Lin, L., Chen, Y.W.,
2017.

 
 Robust hepatic vessel segmentation using multi deep
convolution network 1013711, 1013711.

 

 
 Kitrungrotsakul et al. (2019) 
 
Kitrungrotsakul, T., Han, X.H.,
Iwamoto, Y., Lin, L.,
Foruzan, A.H., Xiong, W.,
Chen, Y.W., 2019.

 
 VesselNet: A deep convolutional neural network with
multi pathways for robust hepatic vessel segmentation.

 
 Computerized Medical Imaging and Graphics
75, 74–83.

 

 
 Klepaczko et al. (2016) 
 
Klepaczko, A., Szczypiński, P.,
Deistung, A., Reichenbach, J.R.,
Materka, A., 2016.

 
 Simulation of MR angiography imaging for validation
of cerebral arteries segmentation algorithms.

 
 Computer Methods and Programs in Biomedicine
137, 293–309.

 

 
 Kovesi (1999) 
 
Kovesi, P., 1999.

 
 Image Features from Phase Congruency.

 
 Journal of computer vision research ,
309–18.

 

 
 L Srinidhi et al. (2017) 
 
L Srinidhi, C., Aparna, P.,
Rajan, J., 2017.

 
 Recent Advancements in Retinal Vessel
Segmentation.

 
 Journal of Medical Systems 41.

 

 
 Lahiri et al. (2017) 
 
Lahiri, A., Ayush, K.,
Biswas, P.K., Mitra, P.,
2017.

 
 Generative Adversarial Learning for Reducing Manual
Annotation in Semantic Segmentation on Large Scale Miscroscopy Images:
Automated Vessel Segmentation in Retinal Fundus Image as Test Case.

 
 IEEE Computer Society Conference on Computer Vision
and Pattern Recognition Workshops 2017-July,
794–800.

 

 
 Law and Chung (2010) 
 
Law, M.W.K., Chung, A.C.S.,
2010.

 
 An oriented flux symmetry based active contour model
for three dimensional vessel segme, in: Computer Vision
- ECCV, pp. 720–734.

 

 
 Lee et al. (2019) 
 
Lee, M.C.H., Petersen, K.,
Pawlowski, N., Glocker, B.,
Schaap, M., 2019.

 
 TETRIS: Template Transformer Networks for Image
Segmentation with Shape Priors.

 
 IEEE Transactions on Medical Imaging ,
1–1.

 

 
 Lesage et al. (2009) 
 
Lesage, D., Angelini, E.D.,
Bloch, I., Funka-Lea, G.,
2009.

 
 A review of 3D vessel lumen segmentation techniques:
Models, features and extraction schemes.

 
 Medical Image Analysis 13,
819–845.

 

 
 Lesage et al. (2016) 
 
Lesage, D., Angelini, E.D.,
Funka-Lea, G., Bloch, I.,
2016.

 
 Adaptive particle filtering for coronary artery
segmentation from 3D CT angiograms.

 
 Computer Vision and Image Understanding
151, 29–46.

 

 
 Li et al. (2018) 
 
Li, M., Yin, Q., Lu, M.,
2018.

 
 Retinal Blood Vessel Segmentation Based on
Multi-Scale Deep Learning 15, 117–123.

 

 
 Li et al. (2016) 
 
Li, Q., Feng, B., Xie,
L., Liang, P., Zhang, H.,
Wang, T., 2016.

 
 A cross-modality learning approach for vessel
segmentation in retinal images.

 
 IEEE Transactions on Medical Imaging
35, 109–118.

 

 
 Li et al. (2019) 
 
Li, R., Li, M., Li, J.,
2019.

 
 Connection Sensitive Attention U-NET for Accurate
Retinal Vessel Segmentation .

 

 
 Lian et al. (2018) 
 
Lian, C.F., Zhang, J.,
Liu, M.X., Zong, X.P.,
Hung, S.c.C., Lin, W.L.,
Shen, D.G., 2018.

 
 Multi-channel multi-scale fully convolutional
network for 3D perivascular spaces segmentation in 7T MR images.

 
 Medical Image Analysis 46,
106–117.

 

 
 Lian et al. (2019) 
 
Lian, S., Li, L., Lian,
G., Xiao, X., Luo, Z.,
Li, S., 2019.

 
 A Global and Local Enhanced Residual U-Net for
Accurate Retinal Vessel Segmentation.

 
 IEEE/ACM Transactions on Computational Biology and
Bioinformatics , 1–1.

 

 
 Lin et al. (2012) 
 
Lin, K.S., Tsai, C.L.,
Tsai, C.H., Sofka, M.,
Chen, S.J., Lin, W.Y.,
2012.

 
 Retinal Vascular Tree Reconstruction With Anatomical
Realism.

 
 IEEE Transactions on Biomedical Engineering
59, 3337–3347.

 

 
 Lin et al. (2019) 
 
Lin, Y., Zhang, H.G., Hu,
G., 2019.

 
 Automatic Retinal Vessel Segmentation via Deeply
Supervised and Smoothly Regularized Network.

 
 IEEE Access 7,
57717–57724.

 

 
 Liskowski and Krawiec (2016) 
 
Liskowski, P., Krawiec, K.,
2016.

 
 Segmenting Retinal Blood Vessels With Deep Neural
Networks.

 
 IEEE Transactions on Medical Imaging
35, 2369–2380.

 

 
 Liu et al. (2014) 
 
Liu, X.M., Zeng, Z.G.,
Wang, X.P., 2014.

 
 Vessel segmentation in retinal images with a
multiple kernel learning based method, in: Proceedings
of the 2014 International Joint Conference on Neural Networks, pp.
493–497.

 

 
 Livne et al. (2019) 
 
Livne, M., Rieger, J.,
Aydin, O.U., Taha, A.A.,
Akay, E.M., Kossen, T.,
Sobesky, J., Kelleher, J.D.,
Hildebrand, K., Frey, D.,
Madai, V.I., 2019.

 
 A U-Net Deep Learning Framework for High Performance
Vessel Segmentation in Patients With Cerebrovascular Disease.

 
 Frontiers in neuroscience 13,
97.

 

 
 Lorza et al. (2018) 
 
Lorza, A.M.A., van Engelen, A.,
Petersen, J., van der Lugt, A.,
de Bruijne, M., Arias Lorza, A.M.,
van Engelen, A., Petersen, J.,
van der Lugt, A., de Bruijne, M.,
Lorza, A.M.A., van Engelen, A.,
Petersen, J., van der Lugt, A.,
de Bruijne, M., Arias Lorza, A.M.,
van Engelen, A., Petersen, J.,
van der Lugt, A., de Bruijne, M.,
2018.

 
 Maximization of regional probabilities using Optimal
Surface Graphs: Application to carotid artery segmentation in MRI.

 
 Medical Physics 45,
1159–1169.

 

 
 Lu et al. (2017) 
 
Lu, S., Huang, H., Liang,
P., Chen, G., Xiao, L.,
2017.

 
 Hepatic vessel segmentation using variational level
set combined with non-local robust statistics.

 
 Magnetic Resonance Imaging 36,
180–186.

 

 
 Lugauer et al. (2014) 
 
Lugauer, F., Zhang, J.,
Zheng, Y., Hornegger, J.,
Kelm, B.M., 2014.

 
 Improving accuracy in coronary lumen segmentation
via explicit calcium exclusion, learning-based ray detection and surface
optimization.

 
 Medical Imaging 2014: Image Processing
9034, 90343U.

 

 
 Luo et al. (2017) 
 
Luo, Y., Yang, L., Wang,
L., B, H.C., 2017.

 
 Efficient CNN-CRF Network for Retinal Image
Segmentation 710, 157–165.

 

 
 Lupascu et al. (2010) 
 
Lupascu, C.A., Tegolo, D.,
Trucco, E., 2010.

 
 FABC: Retinal Vessel Segmentation Using AdaBoost.

 
 IEEE Transactions on Information Technology in
Biomedicine 14, 1267–1274.

 

 
 Lupascu et al. (2013) 
 
Lupascu, C.A., Tegolo, D.,
Trucco, E., Lupaşcu, C.A.,
Tegolo, D., Trucco, E.,
2013.

 
 Accurate estimation of retinal vessel width using
bagged decision trees and an extended multiresolution Hermite model.

 
 Medical Image Analysis 17,
1164–1180.

 

 
 Maninis et al. (2016) 
 
Maninis, K.K., Pont-Tuset, J.,
Arbeláez, P., Van Gool, L.,
2016.

 
 Deep retinal image understanding.

 
 Lecture Notes in Computer Science (including
subseries Lecture Notes in Artificial Intelligence and Lecture Notes in
Bioinformatics) 9901 LNCS, 140–148.

 

 
 Manniesing et al. (2006) 
 
Manniesing, R., Viergever, M.A.,
Niessen, W.J., 2006.

 
 Vessel enhancing diffusion A scale space
representation of vessel structures.

 
 Medical Image Analysis 10,
815–825.

 

 
 Mansour (2017) 
 
Mansour, R.F., 2017.

 
 Evolutionary Computing Enriched Computer-Aided
Diagnosis System for Diabetic Retinopathy: A Survey.

 
 IEEE Reviews in Biomedical Engineering
10, 334–349.

 

 
 Mapayi and Tapamo (2016) 
 
Mapayi, T., Tapamo, J.R.,
2016.

 
 Difference Image and Fuzzy C-Means for Detection of
Retinal Vessels, in: 2016 IEEE Southwest Symposium on
Image Analysis and Interpretation, pp. 169–172.

 

 
 Mapayi et al. (2015) 
 
Mapayi, T., Tapamo, J.R.,
Viriri, S., 2015.

 
 Retinal Vessel Segmentation: A Comparative Study of
Fuzzy C-Means and Sum Entropy Information on Phase Congruency.

 
 International Journal of Advanced Robotic Systems
12, 133.

 

 
 Marques et al. (2016) 
 
Marques, C., Ramalho, J.S.,
Pereira, P., Mota, M.C.,
2016.

 
 Dense Volume-to-Volume Vascular Boundary
Detection.

 
 International Conference on Medical Image Computing
and Computer-assisted Intervention 90,
371–279.

 

 
 Mehmet et al. (2016) 
 
Mehmet, A.G., Gülsün, M.A.,
Funka-Lea, G., Sharma, P.,
Rapaka, S., Zheng, Y.,
2016.

 
 Coronary Centerline Extraction via Optimal Flow
Paths and CNN Path Pruning.

 
 Lecture Notes in Computer Science (including
subseries Lecture Notes in Artificial Intelligence and Lecture Notes in
Bioinformatics) 9902 LNCS, 317–325.

 

 
 Melki et al. (2014) 
 
Melki, I., Cardon, C.,
Gogin, N., Talbot, H.,
Najman, L., 2014.

 
 Learning-based Automatic Detection of Severe
Coronary Stenoses In CT Angiographies, in: Medical
Imaging 2014: Computer-Aided Diagnosis. volume 9035, p.
903536.

 

 
 Memari et al. (2017) 
 
Memari, N., Ramli, A.R.,
Bin Saripan, M.I., Mashohor, S.,
Moghbel, M., 2017.

 
 Supervised retinal vessel segmentation from color
fundus images based on matched filtering and AdaBoost classifier.

 
 Plos One 12.

 

 
 Mo and Zhang (2017) 
 
Mo, J., Zhang, L., 2017.

 
 Multi-level deep supervised networks for retinal
vessel segmentation.

 
 International Journal of Computer Assisted
Radiology and Surgery 12, 2181–2193.

 

 
 Moccia et al. (2018) 
 
Moccia, S., Momi, E.D.,
El, S., Mattos, L.S.,
2018.

 
 Blood vessel segmentation algorithms – Review of
methods , datasets and evaluation metrics.

 
 Computer Methods and Programs in Biomedicine
158, 71–91.

 

 
 Moreno and Smedby (2015) 
 
Moreno, R., Smedby, Ö.,
2015.

 
 Gradient-based enhancement of tubular structures in
medical images.

 
 Medical Image Analysis 26,
19–29.

 

 
 Mou et al. (2019) 
 
Mou, L., Chen, L., Cheng,
w.J., Gu, Z., Zhao, Y.,
Liu, J., 2019.

 
 Dense Dilated Network with Probability Regularized
Walk for Vessel Detection.

 
 IEEE Transactions on Medical Imaging
PP, 1–1.

 

 
 Nardelli et al. (2017) 
 
Nardelli, P., Jimenez-Carretero, D.,
Bermejo-Pelaez, D., Ledesma-Carbayo,
M.J., Rahaghi, F.N., Estepar, R.S.,
2017.

 
 Deep-learning strategy for pulmonary artery-vein
classification of non-contrast ct images, in: 2017 IEEE
14th International Symposium on Biomedical Imaging, pp.
384–387.

 

 
 Nardelli et al. (2018a) 
 
Nardelli, P., Jimenez-Carretero, D.,
Bermejo-Pelaez, D., Washko, G.R.,
Rahaghi, F.N., Ledesma-Carbayo, M.J.,
San Jose Estepar, R., 2018a.

 
 Pulmonary Artery-Vein Classification in CT Images
Using Deep Learning.

 
 IEEE Transactions on Medical Imaging
37, 2428–2440.

 

 
 Nardelli et al. (2018b) 
 
Nardelli, P., Jimenez-Carretero, D.,
Bermejo-Pelaez, D., Washko, G.R.,
Rahaghi, F.N., Ledesma-Carbayo, M.J.,
San Jose Estepar, R., 2018b.

 
 Pulmonary Artery–Vein Classification in CT Images
Using Deep Learning.

 
 IEEE Transactions on Medical Imaging
37, 2428–2440.

 

 
 Nasr-Esfahani et al. (2018) 
 
Nasr-Esfahani, E., Karimi, N.,
Jafari, M.H., Soroushmehr, S.M.,
Samavi, S., Nallamothu, B.K.,
Najarian, K., 2018.

 
 Segmentation of vessels in angiograms using
convolutional neural networks, in: Biomedical Signal
Processing and Control, Elsevier Ltd. pp.
240–251.

 

 
 Nasr-Esfahani et al. (2016) 
 
Nasr-Esfahani, E., Samavi, S.,
Karimi, N., Soroushmehr, S.M.R.,
Ward, K., Jafari, M.H.,
Felfeliyan, B., Nallamothu, B.,
Najarian, K., 2016.

 
 Vessel Extraction in X-Ray Angiograms Using Deep
Learning, in: Annual International Conference of the
IEEE Engineering in Medicine and Biology Society, pp.
643–646.

 

 
 Nazir et al. (2020) 
 
Nazir, A., Cheema, M.N.,
Sheng, B., Li, H., Li,
P., Yang, P., Jung, Y.,
Qin, J., Kim, J., Feng,
D.D., 2020.

 
 OFF-eNET: An Optimally Fused Fully End-to-End
Network for Automatic Dense Volumetric 3D Intracranial Blood Vessels
Segmentation.

 
 IEEE Transactions on Image Processing
29, 7192–7202.

 

 
 Ni et al. (2020) 
 
Ni, J., Wu, J., Wang, H.,
Tong, J., Chen, Z.,
Wong, K.K., Abbott, D.,
2020.

 
 Global channel attention networks for intracranial
vessel segmentation.

 
 Computers in Biology and Medicine
118.

 

 
 Odstrcilik et al. (2013) 
 
Odstrcilik, J., Kolar, R.,
Budai, A., Hornegger, J.,
Jan, J., Gazarek, J.,
Kubena, T., Cernosek, P.,
Svoboda, O., Angelopoulou, E.,
2013.

 
 Retinal vessel segmentation by improved matched
filtering: evaluation on a new high-resolution fundus image database.

 
 IET Image Processing 7,
373–383.

 

 
 Oliveira et al. (2017) 
 
Oliveira, A., Pereira, S.,
Silva, C.A., 2017.

 
 Augmenting Data When Training a CNN for Retinal
Vessel Segmentation: How to Warp?

 

 
 Orlando et al. (2017) 
 
Orlando, J.I., Prokofyeva, E.,
Blaschko, M.B., 2017.

 
 A Discriminatively Trained Fully Connected
Conditional Random Field Model for Blood Vessel Segmentation in Fundus
Images.

 
 IEEE Transactions on Biomedical Engineering
64, 16–27.

 

 
 Pohankar and Wankhade (2016) 
 
Pohankar, N.P., Wankhade, N.R.,
2016.

 
 Different methods used for extraction of blood
vessels from retinal images.

 
 IEEE WCTFTR 2016 - Proceedings of 2016 World
Conference on Futuristic Trends in Research and Innovation for Social
Welfare , 1–4.

 

 
 Poletti and Grisan (2014) 
 
Poletti, E., Grisan, E.,
2014.

 
 A boosted optimal linear learner for retinal vessel
segmentation.

 
 Medical Imaging 2014: Computer-Aided Diagnosis
9035, 903530.

 

 
 Rani et al. (2016) 
 
Rani, P., Priyadarshini, N.,
Rajkumar, E.R., Rajamani, K.,
2016.

 
 Retinal vessel segmentation under pathological
conditions using supervised machine learning.

 

 
 Rempfler et al. (2014) 
 
Rempfler, M., Schneider, M.,
Ielacqua, G.D., Xiao, X.H.,
Stock, S.R., Klohs, J.,
Szekely, G., Andres, B.,
Menze, B.H., 2014.

 
 Extracting Vascular Networks under Physiological
Constraints via Integer Programming, in: Medical Image
Computing and Computer-Assisted Intervention - Miccai 2014, Pt Ii. volume
8674, pp. 505–512.

 

 
 Rempfler et al. (2015) 
 
Rempfler, M., Schneider, M.,
Ielacqua, G.D., Xiao, X.H.,
Stock, S.R., Klohs, J.,
Szekely, G., Andres, B.,
Menze, B.H., 2015.

 
 Reconstructing cerebrovascular networks under local
physiological constraints by integer programming.

 
 Medical Image Analysis 25,
86–94.

 

 
 Ricci and Perfetti (2007) 
 
Ricci, E., Perfetti, R.,
2007.

 
 Retinal Blood Vessel Segmentation Using Line
Operators and Support Vector Classification.

 
 IEEE Transactions on Medical Imaging
26, 1357–1365.

 

 
 Ronneberger et al. (2015) 
 
Ronneberger, O., Fischer, P.,
Brox, T., 2015.

 
 U-net: Convolutional networks for biomedical image
segmentation, in: International Conference on Medical
image computing and computer-assisted intervention,
Springer. pp. 234–241.

 

 
 Rudyanto et al. (2014) 
 
Rudyanto, R.D., Kerkstra, S.,
van Rilowort, E.M., Fetita, C.,
Brillet, P.Y., Lefevre, C.,
Xue, W.Z., Zhu, X.J.,
Liang, J.M., Oksuz, I.,
Unay, D., Kadipasoglu, K.,
Estepar, R.S., Ross, J.C.,
Washko, G.R., Prieto, J.C.,
Hoyos, M.H., Orkisz, M.,
Meine, H., Hullebrand, M.,
Stocker, C., Mir, F.L.,
Naranjo, V., Villanueva, E.,
Staring, M., Xiao, C.Y.,
Stoel, B.C., Fabijanska, A.,
Smistad, E., Elster, A.C.,
Lindseth, F., Foruzan, A.H.,
Kiros, R., Popuri, K.,
Cobzas, D., Jimenez-Carretero, D.,
Santos, A., Ledesma-Carbayo, M.J.,
Helmberger, M., Urschler, M.,
Pienn, M., Bosboom, D.G.H.,
Campo, A., Prokop, M.,
de Jong, P.A., Ortiz-de Solorzano, C.,
Munoz-Barrutia, A., van Ginneken, B.,
2014.

 
 Comparing algorithms for automated vessel
segmentation in computed tomography scans of the lung: the VESSEL12 study.

 
 Medical Image Analysis 18,
1217–1232.

 

 
 Saffarzadeh et al. (2014) 
 
Saffarzadeh, V., Osareh, A.,
Shadgar, B., 2014.

 
 Vessel Segmentation in Retinal Images Using
Multi-scale Line Operator and K-Means Clustering.

 
 Journal of Medical Signals and Sensors
4, 122–129.

 

 
 Samuel et al. (2017) 
 
Samuel, D., Silveira, A.,
Manuel, J., Tavares, R.S.,
Jodas, D.S., Pereira, A.S.,
Tavares, J.J.M.R.J., 2017.

 
 Automatic segmentation of the lumen region in
intravascular images of the coronary artery.

 
 Medical Image Analysis 40,
60–79.

 

 
 Sankaran et al. (2016) 
 
Sankaran, S., Schaap, M.,
Hunley, S.C., Min, J.K.,
Taylor, C.A., Grady, L.,
2016.

 
 HALE: Healthy Area of Lumen Estimation for Vessel
Stenosis Quantification, Springer, Cham, pp.
380–387.

 

 
 Schaap et al. (2009) 
 
Schaap, M., Metz, C.T.,
van Walsum, T., van der Giessen, A.G.,
Weustink, A.C., Mollet, N.R.,
Bauer, C., Bogunović, H.,
Castro, C., Deng, X.,
Dikici, E., O’Donnell, T.,
Frenay, M., Friman, O.,
Hoyos, M.H., Kitslaar, P.H.,
Krissian, K., Kühnel, C.,
Luengo-Oroz, M.a., Orkisz, M.,
Smedby, Ö., Styner, M.,
Szymczak, A., Tek, H.,
Wang, C., Warfield, S.K.,
Zambal, S., Zhang, Y.,
Krestin, G.P., Niessen, W.J.,
2009.

 
 Standardized evaluation methodology and reference
database for evaluating coronary artery centerline extraction algorithms.

 
 Medical Image Analysis 13,
701–714.

 

 
 Schaap et al. (2011) 
 
Schaap, M., Van Walsum, T.,
Neefjes, L., Metz, C.,
Capuano, E., De Bruijne, M.,
Niessen, W., 2011.

 
 Robust shape regression for supervised vessel
segmentation and its application to coronary segmentation in cta.

 
 IEEE Transactions on Medical Imaging
30, 1974–1986.

 

 
 Schneider et al. (2015) 
 
Schneider, M., Hirsch, S.,
Weber, B., Székely, G.,
Menze, B.H., Szekely, G.,
Menze, B.H., 2015.

 
 Joint 3-D vessel segmentation and centerline
extraction using oblique Hough forests with steerable filters.

 
 Medical Image Analysis 19,
220–249.

 

 
 Shen et al. (2019) 
 
Shen, Y., Fang, Z., Gao,
Y., Xiong, N., Zhong, C.,
Tang, X., 2019.

 
 Coronary Arteries Segmentation Based on 3D FCN With
Attention Gate and Level Set Function.

 
 IEEE Access 7,
42826–42835.

 

 
 Shin et al. (2019) 
 
Shin, S.Y., Lee, S., Yun,
I.D., Lee, K.M., 2019.

 
 Deep vessel segmentation by learning graphical
connectivity.

 
 Medical Image Analysis 58,
101556.

 

 
 Simonyan and Zisserman (2015) 
 
Simonyan, K., Zisserman, A.,
2015.

 
 Very deep convolutional networks for large-scale
image recognition, pp. 1–14.

 

 
 Singh and Kaur (2015) 
 
Singh, N., Kaur, L., 2015.

 
 A survey on blood vessel segmentation methods in
retinal images.

 
 2015 International Conference on Electronic Design,
Computer Networks and Automated Verification, EDCAV 2015 ,
23–28.

 

 
 Sironi et al. (2015) 
 
Sironi, A., Tekin, B.,
Rigamonti, R., Lepetit, V.,
Fua, P., 2015.

 
 Learning separable filters.

 
 IEEE Transactions on Pattern Analysis and Machine
Intelligence 37, 94–106.

 

 
 Soomro et al. (2019) 
 
Soomro, T.A., Afifi, A.J.,
Zheng, L.H., Soomro, S.,
Gao, J.B., Hellwich, O.,
Paul, M., 2019.

 
 Deep Learning Models for Retinal Blood Vessels
Segmentation: A Review.

 
 IEEE Access 7,
71696–71717.

 

 
 Suri et al. (2002) 
 
Suri, J.S., Liu, K.,
Reden, L., Laxminarayan, S.,
2002.

 
 A Review on MR Vascular Image Processing: Skeleton
Versus Nonskeleton Approaches: Part II.

 
 IEEE Transactions on Information Technology in
Biomedicine 6, 324–337.

 

 
 Tan et al. (2017) 
 
Tan, J.H., Acharya, U.R.,
Bhandary, S.V., Chua, K.C.,
Sivaprasad, S., 2017.

 
 Segmentation of optic disc, fovea and retinal
vasculature using a single convolutional neural network.

 
 Journal of Computational Science
20, 70–79.

 

 
 Tetteh et al. (2017) 
 
Tetteh, G., Rempfler, M.,
Zimmer, C., Menze, B.H.,
2017.

 
 Deep-FExt: Deep Feature Extraction for Vessel
Segmentation and Centerline Prediction, in: Machine
Learning in Medical Imaging. volume 10541, pp.
344–352.

 

 
 Turetken et al. (2016) 
 
Turetken, E., Benmansour, F.,
Andres, B., Glowacki, P.,
Pfister, H., Fua, P.,
2016.

 
 Reconstructing Curvilinear Networks Using Path
Classifiers and Integer Programming.

 
 IEEE Transactions on Pattern Analysis and Machine
Intelligence 38, 2515–2530.

 

 
 Uslu and Bharath (2019) 
 
Uslu, F., Bharath, A.A.,
2019.

 
 A recursive Bayesian approach to describe retinal
vasculature geometry.

 
 Pattern Recognition 87,
157–169.

 

 
 Vega et al. (2015) 
 
Vega, R., Sanchez-Ante, G.,
Falcon-Morales, L.E., Sossa, H.,
Guevara, E., 2015.

 
 Retinal vessel extraction using Lattice Neural
Networks with dendritic processing.

 
 Computers in Biology and Medicine
58, 20–30.

 

 
 Vostatek et al. (2017) 
 
Vostatek, P., Claridge, E.,
Uusitalo, H., Hauta-Kasari, M.,
Fält, P., Lensu, L.,
2017.

 
 Performance comparison of publicly available retinal
blood vessel segmentation methods.

 
 Computerized Medical Imaging and Graphics
55, 2–12.

 

 
 Vukadinovic et al. (2010) 
 
Vukadinovic, D., Walsum, T.V.,
Manniesing, R., Rozie, S.,
Hameeteman, R., Weert, T.T.D.,
Lugt, A.V.D., Niessen, W.J.,
2010.

 
 Segmentation of the Outer Vessel Wall of the Common
Carotid Artery in CTA.

 
 IEEE Transactions on Medical Imaging
29, 65–76.

 

 
 Wang et al. (2019) 
 
Wang, C., Zhao, Z.Y., Ren,
Q.Q., Xu, Y.T., Yu, Y.,
2019.

 
 Dense U-net Based on Patch-Based Learning for
Retinal Vessel Segmentation.

 
 Entropy 21.

 

 
 Wang et al. (2015) 
 
Wang, S., Yin, Y., Cao,
G., Wei, B., Zheng, Y.,
Yang, G., 2015.

 
 Hierarchical retinal blood vessel segmentation based
on feature and ensemble learning.

 
 Neurocomputing 149,
708–717.

 

 
 Wolterink et al. (2019) 
 
Wolterink, J.M., van Hamersvelt, R.W.,
Viergever, M.A., Leiner, T.,
Isgum, I., Hamersvelt, R.W.,
Viergever, M.A., Leiner, T.,
Išgum, I., 2019.

 
 Coronary artery centerline extraction in cardiac CT
angiography using a CNN-based orientation classifier.

 
 Medical Image Analysis 51,
46–60.

 

 
 Wu et al. (2016) 
 
Wu, A., Xu, Z., Gao, M.,
Buty, M., Mollura, D.J.,
2016.

 
 Deep vessel tracking: A generalized probabilistic
approach via deep learning.

 
 Proceedings - International Symposium on Biomedical
Imaging 2016-June, 1363–1367.

 

 
 Wu et al. (2018) 
 
Wu, Y.C., Xia, Y., Song,
Y., Zhang, Y.N., Cai, W.D.,
2018.

 
 Multiscale Network Followed Network Model for
Retinal Vessel Segmentation, in: Medical Image
Computing and Computer Assisted Intervention - Miccai 2018, Pt Ii, pp.
119–126.

 

 
 Xia et al. (2018) 
 
Xia, H.Y., Jiang, F.,
Deng, S.F., Xin, J.,
Doss, R., 2018.

 
 Mapping Functions Driven Robust Retinal Vessel
Segmentation via Training Patches.

 
 IEEE Access 6,
61973–61982.

 

 
 Yan et al. (2018) 
 
Yan, Z.Q., Yang, X.,
Cheng, K.t.T., 2018.

 
 Joint Segment-Level and Pixel-Wise Losses for Deep
Learning Based Retinal Vessel Segmentation.

 
 IEEE Transactions on Biomedical Engineering
65, 1912–1923.

 

 
 You et al. (2011) 
 
You, X., Peng, Q., Yuan,
Y., Cheung, Y.m., Lei, J.,
2011.

 
 Segmentation of retinal blood vessels using the
radial projection and semi-supervised approach.

 
 Pattern Recognition 44,
2314–2324.

 

 
 Yu et al. (2019) 
 
Yu, F., Zhao, J., Gong,
Y., Wang, Z., Li, Y.,
Yang, F., Dong, B., Li,
Q., Zhang, L., 2019.

 
 Annotation-free cardiac vessel segmentation via
knowledge transfer from retinal images, in:
International Conference on Medical Image Computing and
Computer-Assisted Intervention, Springer. pp.
714–722.

 

 
 Yun et al. (2019) 
 
Yun, J., Park, J., Yu,
D., Yi, J., Lee, M.,
Park, H.J., Lee, J.G.,
Seo, J.B., Kim, N., 2019.

 
 Improvement of fully automated airway segmentation
on volumetric computed tomographic images using a 2.5 dimensional
convolutional neural net.

 
 Medical Image Analysis 51,
13–20.

 

 
 Zeng et al. (2018) 
 
Zeng, Y.z., Zhao, Y.q.,
Liao, S.h., Liao, M.,
Chen, Y., Liu, X.y.,
2018.

 
 Liver vessel segmentation based on centerline
constraint and intensity model.

 
 Biomedical Signal Processing and Control
45, 192–201.

 

 
 Zhang et al. (2017a) 
 
Zhang, J., Chen, Y.,
Bekkers, E., Wang, M.L.,
Dashtbozorg, B., Romeny, B.M.T.,
2017a.

 
 Retinal vessel delineation using a brain-inspired
wavelet transform and random forest.

 
 Pattern Recognition 69,
107–123.

 

 
 Zhang et al. (2017b) 
 
Zhang, J., Gao, Y., Park,
S.H., Zong, X., Lin, W.,
Shen, D., 2017b.

 
 Structured Learning for 3-D Perivascular Space
Segmentation Using Vascular Features.

 
 IEEE Transactions on Biomedical Engineering
64, 2803–2812.

 

 
 Zhang et al. (2016) 
 
Zhang, J., Gao, Y.Z.,
Park, S.H., Zong, X.P.,
Lin, W.L., Shen, D.G.,
2016.

 
 Segmentation of Perivascular Spaces Using Vascular
Features and Structured Random Forest from 7T MR Image, in:
Machine Learning in Medical Imaging, Mlmi 2016. volume
10019, pp. 61–68.

 

 
 Zhang et al. (2014) 
 
Zhang, L., Fisher, M.,
Wang, W.J., 2014.

 
 Comparative performance of texton based vascular
tree segmentation in retinal images, in: 2014 IEEE
International Conference on Image Processing, pp. 952–956.

 

 
 Zhang et al. (2019) 
 
Zhang, S., Zheng, R., Luo,
Y., Wang, X., Mao, J.,
Roberts, C.J., Sun, M.,
2019.

 
 Simultaneous Arteriole and Venule Segmentation of
Dual-Modal Fundus Images Using a Multi-Task Cascade Network.

 
 IEEE Access 7,
57561–57573.

 

 
 Zhang and Chung (2018) 
 
Zhang, Y.S., Chung, A.C.S.,
2018.

 
 Deep Supervision with Additional Labels for Retinal
Vessel Segmentation Task, in: Medical Image Computing
and Computer Assisted Intervention, pp. 83–91.

 

 
 Zhao et al. (2018a) 
 
Zhao, F., Chen, Y., Chen,
F., He, X., Cao, X.,
Hou, Y., Yi, H., He,
X., Liang, J., 2018a.

 
 Semi-Supervised Cerebrovascular Segmentation by
Hierarchical Convolutional Neural Network.

 
 IEEE Access 6,
67841–67852.

 

 
 Zhao et al. (2019) 
 
Zhao, F., Chen, Y., Hou,
Y., He, X., 2019.

 
 Segmentation of blood vessels using rule-based and
machine-learning-based methods: a review.

 
 Multimedia Systems 25,
109–118.

 

 
 Zhao et al. (2018b) 
 
Zhao, H., Li, H.Q.,
Maurer-Stroh, S., Cheng, L.,
2018b.

 
 Synthesizing retinal and neuronal images with
generative adversarial nets.

 
 Medical Image Analysis 49,
14–26.

 

 
 Zhao et al. (2017) 
 
Zhao, M.L., Miles, B.,
Hamarneh, G., 2017.

 
 Leveraging Tree Statistics for Extracting Anatomical
Trees from 3D Medical Images.

 

 
 Zheng et al. (2012) 
 
Zheng, Y., John, M., Liao,
R., Nöttling, A., Boese, J.,
Kempfert, J., Walther, T.,
Brockmann, G., Comaniciu, D.,
2012.

 
 Automatic aorta segmentation and valve landmark
detection in C-Arm CT for transcatheter aortic valve implantation.

 
 IEEE Transactions on Medical Imaging
31, 2307–2321.

 

 
 Zheng et al. (2011) 
 
Zheng, Y., Loziczonek, M.,
Georgescu, B., Zhou, S.K.,
2011.

 
 Machine Learning Based Vesselness Measurement for
Coronary Artery Segmentation in Cardiac CT Volumes, in:
Spie.

 

 
 Zhou et al. (2007) 
 
Zhou, J.H., Chang, S.,
Metaxas, D., Axel, L.,
2007.

 
 Vascular structure segmentation and bifurcation
detection, in: 2007 4th IEEE International Symposium on
Biomedical Imaging : Macro to Nano, Vols 1-3, pp. 872–+.

 

 
 Zreik et al. (2018) 
 
Zreik, M., van Hamersvelt, R.W.,
Wolterink, J.M., Leiner, T.,
Viergever, M.A., Isgum, I.,
van Hamersvelt, R.W., Leiner, T.,
Viergever, M.A., Wolterink, J.M.,
Leiner, T., Viergever, M.A.,
Isgum, I., 2018.

 
 A Recurrent CNN for Automatic Detection and
Classification of Coronary Artery Plaque and Stenosis in Coronary CT
Angiography.

 
 IEEE Transactions on Medical Imaging
38, 1588–1598.