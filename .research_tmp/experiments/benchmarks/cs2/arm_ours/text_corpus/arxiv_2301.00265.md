Source-Free Unsupervised Domain Adaptation: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2301.00265v2 [cs.CV] 06 Jan 2023 
 
 

# Source-Free Unsupervised Domain Adaptation: A Survey

 
 
 Yuqi Fang
 
    
 Pew-Thian Yap
 
    
 Weili Lin
 
    
 Hongtu Zhu
 
    
 Mingxia Liu
 † † thanks: Y. Fang, P.-T. Yap, W. Lin and M. Liu are with the Department of Radiology and Biomedical Research Imaging Center, University of North Carolina at Chapel Hill, Chapel Hill, NC 27599, United States. H.˜Zhu is with the Department of Biostatistics and Biomedical Research Imaging Center, University of North Carolina at Chapel Hill, NC 27599, USA.
Corresponding author: M.˜Liu (mxliu@med.unc.edu).
 

 Abstract 
 
 Unsupervised domain adaptation (UDA) via deep learning has attracted appealing attention for tackling domain-shift problems caused by distribution discrepancy across different domains.
Existing UDA approaches highly depend on the accessibility of source domain data, which is usually limited in practical scenarios due to privacy protection, data storage and transmission cost, and computation burden.
To tackle this issue, many source-free unsupervised domain adaptation (SFUDA) methods have been proposed recently, which perform knowledge transfer from a pre-trained source model to unlabeled target domain with source data inaccessible.
A comprehensive review of these works on SFUDA is of great significance.
In this paper, we provide a timely and systematic literature review of existing SFUDA approaches from a technical perspective.
Specifically, we categorize current SFUDA studies into two groups, i.e. , white-box SFUDA and black-box SFUDA, and further divide them into finer subcategories based on different learning strategies they use.
We also investigate the challenges of methods in each subcategory, discuss the advantages/disadvantages of white-box and black-box SFUDA methods, conclude the commonly used benchmark datasets, and summarize the popular techniques for improved generalizability of models learned without using source data.
We finally discuss several promising future directions in this field.

 
 
 
 Index Terms:  Domain adaptation, source-free, unsupervised learning, survey.

 
 

## I Introduction 

 
 Deep learning, based on deep neural networks with representation learning, has emerged as a promising technique and made remarkable progress over the past decade, covering the field of computer vision  [ 1 , 2 ] , medical data analysis  [ 3 , 4 ] , natural language processing  [ 5 , 6 ] , etc.
For problems with multiple domains ( e.g. , different datasets or imaging sites), the typical learning process of a deep neural network is to transfer the model learned on a source domain to a target domain.
However, performance degradation is often observed when there exists a distribution gap between the source and target domains, which is termed “domain shift” problem  [ 7 , 8 , 9 ] .
To tackle this problem, various domain adaptation algorithms  [ 10 , 11 ] have been proposed to perform knowledge transfer by reducing inter-domain distribution discrepancy.
To avoid intensive burden of data annotation, unsupervised domain adaptation has achieved much progress  [ 12 , 13 , 14 , 15 ] .
As illustrated in Fig.  1  (a), unsupervised domain adaptation aims to transfer knowledge from a labeled source domain to a target domain without accessing any target label information.

 
 
 Existing deep learning studies on unsupervised domain adaptation highly depend on the accessibility of source data, which is usually limited in practical scenarios due to the following possible reasons.
(1) Data privacy protection . Many source datasets containing confidential information, such as medical and facial data, are not available to third parties due to privacy and security protection.
(2) Data storage and transmission cost . The storage and transmission of large-scale source datasets, such as ImageNet  [ 16 ] , could bring much economic burden.
(3) Computation burden . Training on extremely large source datasets requires high computational resources, which is not practical, especially in real-time deployment cases.
Thus, there is a high demand for source-free unsupervised domain adaptation (SFUDA) methods that transfer a pre-trained source model to the unlabeled target domain without accessing any source data   [ 17 , 18 , 19 , 20 ] .

 
 
 Fig. 1: Illustration of (a) conventional unsupervised domain adaptation (UDA), (b) white-box source-free UDA (SFUDA), and (c) black-box SFUDA.
Compared with (a) conventional UDA that relies on labeled source data { X S , Y S } \{X_{S},Y_{S}\} and unlabeled target data X T X_{T} , (b, c) SFUDA performs knowledge transfer by directly leveraging a pre-trained source model Φ S \Phi_{S} and unlabeled target data X T X_{T} . The difference between (b) white-box SFUDA and (c) black-box SFUDA lies in whether the learnable parameters of the source model Φ S \Phi_{S} are accessible or not.
API: application programming interface.
 
 
 
 Many promising SFUDA algorithms have been developed recently to address problems in the fields of object recognition  [ 21 ] , semantic segmentation  [ 22 ] , image classification  [ 23 ] , object detection  [ 24 ] , face anti-spoofing  [ 25 ] , etc.
A comprehensive review of current studies on SFUDA as well as an outlook on future research directions are urgently needed.
Liu  et al.   [ 26 ] present a review on data-free knowledge transfer, where SFUDA only accounts for part of the review and the taxonomy of SFUDA is generally rough.
And a large number of relevant studies have emerged in the past year, but the related papers are not included in that survey.
In addition, their work does not cover commonly used datasets in this research field.

 
 
 Fig. 2: Taxonomy of existing source-free unsupervised domain adaptation (SFUDA) methods, as well as future outlook. 
 
 
 To fill the gap, in this paper, we provide a timely and thorough literature review of existing deep learning studies on source-free unsupervised domain adaptation.
Our goal is to cover SFUDA studies of the past few years and provide a detailed and systematic SFUDA taxonomy.
Specifically, we classify existing SFUDA approaches into two broad categories: (1) white-box SFUDA as shown in Fig.  1  (b) and (2) black-box SFUDA as illustrated in Fig.  1  (c).
The difference between them lies in whether the model parameters of the pre-trained source model are available or not.
Based on different learning strategies they use, we further subdivide white-box and black-box SFUDA methods into finer categories, and the overall taxonomy is shown in Fig.  2 .
Moreover, we discuss the challenges and insight for methods in each category, provide a comprehensive comparison between white-box and black-box SFUDA approaches, summarize commonly used datasets in this field as well as popular techniques to improve model generalizability across different domains.
We have to point out that SFUDA is still under vigorous development, so we further discuss the main challenges and provide insights into potential future directions accordingly.

 
 
 The rest of this survey is organized as follows.
Section  II and Section  III review existing white-box and black-box SFUDA methods, respectively.
In Section  IV , we compare white-box and black-box SFUDA and present useful strategies to improve model generalization.
Section  V discusses challenges of existing studies and future research directions.
Finally, we conclude this paper in Section  VI .

 
 
 

## II White-Box Source-Free Unsupervised Domain Adaptation 

 
 Denote Φ S \Phi_{S} as the source model well-trained based on the labeled source domain { X S , Y S } \{X_{S},Y_{S}\} , where X S X_{S} and Y S Y_{S} represent source data and the corresponding label information, respectively.
Denote { X T \{X_{T} } as the unlabeled target domain with only target samples X T X_{T} .
The goal of SFUDA is to learn a target model Φ T \Phi_{T} for improved target inference based on the pre-trained source model Φ S \Phi_{S} and unlabeled target data X T X_{T} .
 In the setting of white-box source-free domain adaptation, the source data ( i.e. , X S X_{S} and Y S Y_{S} ) cannot be accessed but the training parameters of the source model Φ S \Phi_{S} are available .
As shown in the upper middle of Fig.  2 , existing white-box SFUDA studies can be divided into two categories: Data Generation Method and Model Fine-Tuning Method , with details elaborated as follows.

 
 

### II-A Data Generation Method 

 

#### II-A 1 Domain Image Generation

 
 Many studies aim to generate source-like image data and achieve cross-domain adaptation by readily applying standard unsupervised domain adaptation techniques.
Based on different image generation strategies, these studies can be divided into the following three subcategories: (1) batch normalization statistics transfer, (2) surrogate source data construction, and (3) generative adversarial network (GAN) based image generation.

 
 
 ( 1 ) Batch Normalization Statistics Transfer .
Considering that batch normalization (BN) stores the running mean and variance for a mini-batch of training data in each layer of a deep learning model, some studies  [ 27 , 28 , 29 ] explicitly leverage such BN statistics for image style transfer, as illustrated in Fig.  3 .
For instance, Yang  et al.   [ 27 ] generate source-like images via a two-stage coarse-to-fine learning strategy. In the coarse image generation step, BN statistics stored in the source model are leveraged to preserve the style characteristics of source images and also maintain the content information of target data.
In the fine image generation step, an image generator based on Fourier Transform is developed to remove ambiguous textural components of generated images and further improve image quality.
With generated source-like images and given target images, a contrast distillation module and a compact consistency measurement module are designed to perform feature-level and output-level adaptation, respectively.
Similarly, Hou  et al.   [ 28 ] perform style transfer by matching BN statistics of generated source-style image features with those saved in the pre-trained source model for image translation.
Hong  et al.   [ 29 ] generate source-like images by designing a style-compensation transformation architecture guided by BN statistics stored in the source model and the generated reliable target pseudo-labels.

 
 
 Fig. 3: 
Illustration of Batch Normalization Statistics Transfer methods for source image generation. By matching batch normalization (BN) statistics between the upper and lower branches, source-like data can be generated by preserving the target content but with source style.
Unsupervised domain adaptation (UDA) can then be performed between source-like data and target data.
 
 
 
 ( 2 ) Surrogate Source Data Construction .
To compensate for the inaccessible source domain, some studies  [ 30 , 31 , 32 , 33 , 34 ] construct surrogate/proxy source data by selecting appropriate samples from the target domain directly, as illustrated in Fig.  4 .
For example, Tian  et al.   [ 30 ] construct pseudo source samples directly from the provided target samples under the guidance of a designed sample transport rule .
The adaptation step and sample transport learning step are performed alternately to refine the approximated source domain and attain confident labels for target data, thus achieving effective cross-domain knowledge adaptation.
Ding  et al.   [ 31 ] build a category-balanced surrogate source domain using pseudo-labeled target samples based on a prototype similarity measurement .
During model adaptation, intra-domain and inter-domain mixup regularizations are introduced to transfer label information from the proxy source domain to the target domain, as well as simultaneously eliminate negative effects caused by noisy labels.
Ye  et al.   [ 32 ] select target samples with high prediction confidence to construct a virtual source set that mimics source distribution.
To align the target and virtual domains, they develop a weighted adversarial loss based on distribution and an uncertainty measurement to achieve cross-domain adaptation.
Moreover, an uncertainty-aware self-training mechanism is proposed to iteratively produce the pseudo-labeled target set to further enhance adaptation performance.
Du  et al.   [ 33 ] construct a surrogate source domain by first selecting target samples near the source prototypes based on an entropy criterion , and then enlarging them by a mixup augmentation strategy  [ 35 ] .
The adversarial training is then used to explicitly mitigate cross-domain distribution gap.
Yao  et al.   [ 34 ] simulate proxy source domain by freezing the source model and minimizing a supervised objective function for optimization.
For the simulated source set, global fitting is enforced by a model gradient based equality constraint, which is optimized by an alternating direction method of multipliers algorithm  [ 36 ] .

 
 
 Fig. 4: Illustration of Surrogate Source Data Construction methods for source data generation.
These methods first construct surrogate/proxy source data by selecting appropriate samples from the target domain and then perform standard unsupervised domain adaptation (UDA). 
 
 
 ( 3 ) GAN-based Image Generation . Instead of approximating the source domain directly using existing target data, Kurmi  et al.   [ 37 ] simulate the source data by training a GAN-based generator, as illustrated in Fig.  5 .
Specifically, they first use a parametric conditional GAN to generate labeled proxy source data by treating the source classifier as an energy based function.
Then, they learn feature patterns that are invariant across two domains via standard adversarial learning for further adaptation.
Hou  et al.   [ 38 ] also update an image generator framework but they aim to translate target images into the source-style ones instead of using the latent noise as in  [ 37 ] .
In their method, the knowledge adaptation is achieved by training 1) a knowledge distillation loss that mitigates the difference between features of newly generated source-style images and those of target images, and 2) a relation-preserving loss that maintains channel-level relationship across different domains.
Li  et al.   [ 39 ] propose a GAN-embedded generator conditioned on a pre-defined label to generate target-style data.
By incorporating real target samples, the learnable parameters of the generator and the adapted model can be updated in a collaborative manner.
Moreover, two constraints, i.e. , weight regularization and clustering-based regularization, are utilized during model adaptation to preserve source knowledge and ensure high-confident target prediction, respectively.

 
 
 

#### II-A 2 Domain Distribution Generation

 
 Instead of generating source-like images directly, some studies propose to align feature prototypes or feature distribution of source data  [ 40 , 41 , 42 , 43 , 44 ] with those in the target domain.
Specifically, Qiu  et al.   [ 40 ] generate feature prototypes for each source category based on a conditional generator and produce pseudo-labels for the target data. The cross-domain prototype adaptation is achieved by aligning the features derived from pseudo-labeled target samples to source prototype with the same category label via contrastive learning.
Tian  et al.   [ 41 ] construct a virtual domain by simply sampling from an approximated gaussian mixture model (GMM) to mimic unseen source domain distribution.
In terms of adaptation procedure, they reduce the distribution gap between the constructed virtual domain and the target domain via adversarial training, thus bypassing inaccessible source domain.
Their practice is based on the assumption that the feature prototype of each category can be mined from each row of the source classifier’ weights  [ 45 ] .
With the same assumption, Ding  et al.   [ 42 ] leverage such source classifier weights and reliable target pseudo-labels derived by spherical k-means clustering to estimate source feature distribution.
After that, proxy source data can be sampled from the estimated source distribution, and a conventional domain adaptation strategy  [ 46 ] is used to explicitly perform cross-domain feature distribution alignment.
Stan  et al.   [ 43 , 44 ] propose to first generate a prototypical distribution representing the source data in an embedding feature space via GMM, and then perform source-free adaptation by enforcing distribution alignment between source and target domains via sliced Wasserstein distance  [ 47 ] .

 
 
 Fig. 5: 
Illustration of Generative Adversarial
Network (GAN) based Image Generation methods for source data generation.
Typically, a pre-defined label and random noise act as the inputs of a GAN-based generator.
By utilizing the pre-trained source model, they synthesize source data for cross-domain adaptation.
 L C ​ E L_{CE} : Cross-entropy loss function. 
 
 
 

#### II-A 3 Challenges and Insight

 
 We classify existing domain image generation methods for SFUDA into three subcategories. We present the challenges of methods in each subcatetory and our insights below.

 
 • 
 
 Among the above-mentioned three subcategories, the first one ( i.e. , batch normalization statistics transfer ) explicitly performs BN statistics matching between source and target domains for style transfer.
Since the BN statistics of the source model are off-the-shelf, these methods are generally efficient and don’t require complex model training.
However, BN statistics mainly focus on keeping the style features while the content information cannot be well preserved.
Therefore, this strategy is more applicable to scenarios where the contextual structure of images between source and target domains does not differ too much. It may not show good adaptation performance, e.g. , from a natural image to a cartoon image, since the content information has significant changes.
Note that BN statistics transfer can also be used as a pre-processing step in source-free domain adaptation, and it can be combined with other strategies, e.g. , circular learning  [ 29 ] , for more effective knowledge transfer.

 

 • 
 
 Methods in the second subcategory ( i.e. , surrogate source data construction ) aim to approximate the proxy source domain using appropriate target samples directly, followed by conventional unsupervised domain adaptation.
Their application is quite broad, including semantic segmentation  [ 32 ] , object recognition  [ 31 , 33 , 34 ] , image classification  [ 30 ] , and digital recognition  [ 30 , 33 ] .
In general, methods in this group are straightforward and computation-efficient by avoiding introducing extra hyperparameters, which is different from generative models.
However, because the proxy source samples are directly selected from the target domain, these generated source data may not
effectively represent the original source domain.
Moreover, how to effectively select informative target data for source data approximation is an important topic to be investigated.
Some studies have proposed various strategies for target data selection based on entropy measurement  [ 32 ] , source prototype  [ 31 , 33 ] , aggregated source decision boundary  [ 30 ] , and equality constrained optimization  [ 34 ] .
This is still an open but very interesting future direction.
For multi-source settings, it is promising to study which source predictor(s) we should refer to for effective target data selection.

 

 • 
 
 Methods in the third category ( i.e. , GAN-based image generation ) typically synthesize images based on a generative model.
Since the generator can model underlying complex distribution of source data with given random noise, GAN-based models generally create more diverse images compared with methods in second category ( i.e. , surrogate source data construction).
However, these methods introduce additional frameworks and learnable parameters ( e.g. , generators and discriminators), which may cost more computation resources.
By comparing experimental results, we find the surrogate source data construction methods  [ 33 , 34 ] generally outperform the GAN-based generators  [ 37 , 39 ] .
The possible reason may be that the constructed source data in the former are closer to real data distributions, while those recovered in GAN-based methods usually suffer from a mode collapse problem  [ 31 ] that leads to low-quality images.
Note that the mode collapse problem can be partly mitigated by using a carefully tuned learning rate, manifold-guided training  [ 48 ] , and virtual mapping  [ 49 ] , which is worth exploring further.

 

 
 
 
 Different from image generation methods (Section  II-A1 ) that directly generate source/target-like images, the distribution generation methods (Section  II-A2 ) generate feature prototype/distribution to achieve cross-domain feature alignment.
By comparing the reported experimental results, we find that the distribution generation approaches  [ 40 , 41 , 42 ] usually outperform the GAN-based image generation method  [ 39 ] .
And surrogate source data construction methods  [ 31 , 33 ] usually show superior performance compared with the distribution generation methods  [ 40 , 41 ] .
The underlying reason could be that the source distributions directly derived from the existing target data  [ 31 , 33 ] are more accurate and stable than the approximated ones  [ 40 , 41 ] .
How to drive the approximated source distribution to the real one can be further explored in the future.

 
 
 
 

### II-B Model Fine-Tuning Method 

 
 Instead of generating source-like data for standard unsupervised domain adaptation, many studies attempt to fine-tune a pre-trained source model by exploiting unlabeled target data in a self-supervised training scheme.
Based on different strategies for fine-tuning the source model, we divide existing studies into five subcategories: (1) self-supervised knowledge distillation, (2) domain alignment via statistics, (3) contrastive learning, (4) uncertainty-guided adaptation, and (5) hidden structure mining methods, as shown in Fig.  2 .
More details are introduced in the following.

 
 

#### II-B 1 Self-Supervised Knowledge Distillation

 
 Fig. 6: Illustration of Self-Supervised Knowledge Distillation methods for source-free unsupervised domain adaptation.
With target data from different augmentations as inputs, a teacher-student framework is utilized to exploit target features, where parameters of teacher network are usually exponential moving average (EMA) of those of student network.
Aug- α \alpha and Aug- β \beta denote two data augmentation methods ( e.g. , flip, rotation, shift, noise addition, distortion, etc), respectively.
 L K ​ D L_{KD} : Knowledge distillation loss function.
 
 
 
 Many studies  [ 50 , 51 , 52 , 23 , 53 , 54 , 55 , 56 ] transfer knowledge learned from source data to the target model via knowledge distillation in a self-supervised manner, as illustrated in Fig.  6 .
In these works, most of them  [ 50 , 51 , 52 , 23 , 53 ] achieve source-free domain adaptation via a mean-teacher scheme for knowledge transfer  [ 57 ] , where the target model not only learns from unseen target domain but also well preserves source model information.
For instance, Liu  et al.   [ 50 ] propose a self-supervised distillation scheme for automatic polyp detection.
By means of keeping output consistency of weak and strong augmented polyp images, source knowledge is implicitly transferred to the target model with a mean teacher strategy  [ 57 ] .
Besides, a diversification flow paradigm is designed to gradually eliminate the style sensitivity among different domains, further enhancing model robustness towards style diversification.
Yang  et al.   [ 51 ] also propose a self-supervised mean-teacher approach for knowledge distillation, with a Transformer  [ 58 ] embedded.
This helps the target model focus on object regions rather than less informative background in an image, thus improving model generalizability.
Assuming that both source and target images are generated from a domain-invariant space by adding noise perturbations on each specific domain, Xiong  et al.   [ 52 ] establish a super target domain via augmenting perturbations based on original target domain.
The super and original target domains are fed into a mean-teacher framework, with three consistency regularization terms (w.r.t. image, instance, and class-wise) introduced for domain alignment.
Chen  et al.   [ 23 ] first divide the target data into clean and noisy subsets guided by a computation loss and regard them as labeled and unlabeled examples, and then utilize the mean teacher technique to self-generate pseudo-labels for the unlabeled target data for domain adaptation.

 
 
 Instead of utilizing the conventional one-teacher one-student paradigm, Liu  et al.   [ 53 ] construct a multi-teacher multi-student framework, where each teacher/student network is initialized using a public network pre-trained on a single dataset.
Here, a graph is constructed to model the similarity among samples, and such relationship predicted by the teacher networks is used to supervise the student networks via a mean-teacher technique.
Rather than leverage the mean-teacher paradigm that averages student’s weights, Yu  et al.   [ 54 ] propose to distill knowledge from teacher to student networks by style and structure regularizations, as well as physical prior constraints.
Instead of employing a teacher-student network as the studies mentioned above, Tang  et al.   [ 55 ] achieve data-free adaptation through gradual knowledge distillation .
Specifically, they first generate pseudo-labels via a constructed neighborhood geometry, and then use pseudo-labels obtained from the latest epoch to supervise the current training epoch for knowledge transfer.

 
 
 Fig. 7: Illustration of Domain Alignment via Statistics methods for source-free unsupervised domain adaptation.
The corresponding methods leverage batch statistics stored in the pre-trained source model to approximate the distribution of inaccessible source data, and then perform cross-domain adaptation by reducing distribution discrepancy between source and target domains. 
 
 
 

#### II-B 2 Domain Alignment via Statistics

 
 Many studies  [ 59 , 60 , 61 , 62 , 63 , 64 , 65 ] leverage batch statistics stored in the pre-trained source model to approximate the distribution of inaccessible source data, and then perform cross-domain adaptation by reducing distribution discrepancy between source and target domains, as demonstrated in Fig.  7 .
For example, Ishii  et al.   [ 59 ] approximate feature distribution of inaccessible source data by using batch normalization statistics (mean and variance) saved in the pre-trained source model.
Then, Kullback-Leibler (KL) divergence is utilized to minimize the distributional discrepancy between source and target domains, thus achieving domain-level alignment.
Inspired by  [ 66 , 67 ] , Paul  et al.   [ 61 ] update the mean and variance of BatchNorm  [ 68 ] or InstanceNorm  [ 69 ] of the pre-trained model based on unseen target data.
Not limited to matching low-order batch-wise statistics ( e.g. , mean and variance), Liu  et al.   [ 60 ] additionally incorporate high-order batch-wise statistics , such as scale and shift parameters, to explicitly keep cross-domain consistency.
Moreover, they quantify each channel’s transferability based on its inter-domain divergence and assume that the channels with lower divergence contribute more to domain adaptation.
Fan  et al.   [ 62 ] propose to align domain statistics adaptively by modulating a learnable blending factor.
By minimizing the total objective function, each BN layer can dynamically obtain its own optimal factor, which controls the contribution of each domain to BN statistics estimation.
The methods mentioned above are all based on Gaussian-based statistics domain alignment, while Eastwood  et al.   [ 63 ] attempt to align histogram-based statistics of the marginal feature distributions of the target domain with those stored in the pre-trained source model, thus well extending adaptation to non-Gaussian distribution scenarios.

 
 
 

#### II-B 3 Contrastive Learning

 
 Many contrastive learning studies  [ 70 , 71 , 19 , 72 , 73 , 25 ] perform data-free adaptation, which helps the target model capture discriminative representations among unlabeled target data.
The main idea is to pull instances of similar categories closer and push instances of different categories away in feature space, as illustrated in Fig.  8 .

 
 
 Fig. 8: Illustration of Contrastive Learning methods for source-free unsupervised domain adaptation.
These methods exploit discriminative representations among unlabeled target data by pulling instances of similar categories closer and
pushing instances of different categories away
in feature space. 
 
 
 For instance, Xia  et al.   [ 70 ] first adaptively divide target instances into source-similar and source-dissimilar sets, and then design a class-aware contrastive module for cross-set distribution alignment.
The idea is to enforce the compactness of target instances from the same category and reduce cross-domain discrepancy, thus prompting effective knowledge transfer from the source model to target data.
Wang  et al.   [ 71 ] present a cross-domain contrastive learning paradigm, which aims to minimize the distance between an anchor instance from one domain and instances from other domains that share the same category as the anchor.
Due to the unavailability of source data, they utilize source prototypical representations, i.e. , weight vectors in
the classifier layer of a pre-trained source model, for feature alignment across two domains.
Huang  et al.   [ 19 ] tackle the data-free domain adaptation by taking advantage of the historical source hypothesis .
Specifically, they propose a historical contrastive instance discrimination strategy to learn from target samples by contrasting their learned embeddings generated by the currently adapted and historical models.
And they also design a historical contrastive category discrimination strategy to weight pseudo-labels of target data to learn category-discriminative target representations, by calculating the consistency between the current and historical model predictions .
The two discrimination strategies help exploit historical source knowledge, bypassing the dependence on source data.
Inspired by  [ 74 ] , Agarwal  et al.   [ 72 ] introduce a pair-wise contrastive objective function to reduce intra-category distance and meanwhile increase inter-category distance based on generated target pseudo-labels.
They also introduce robust source and target models by taking advantage of the generated adversarial instances, which facilitates robust transfer of source knowledge to the target domain.

 
 
 

#### II-B 4 Uncertainty-Guided Adaptation

 
 Uncertainty can measure how well the target model fits the data distribution  [ 75 ] , and many studies  [ 76 , 77 , 78 , 79 , 80 , 81 , 82 , 83 ] utilize such valuable information to guide target predictions in source-free adaptation scenarios (see Fig.  9 ).

 
 
 For instance, Fleuret  et al.   [ 76 ] estimate uncertainty based on differences between predicted outputs with and without Dropout operation   [ 84 ] .
By minimizing such differences, the prediction uncertainty on target data is reduced, meanwhile the learnable feature abstractor can be more robust to noise perturbations.
Lee  et al.   [ 77 ] exploit aleatoric uncertainty by encouraging intra-domain consistency between target images and their augmented ones and enforcing inter-domain feature distribution consistency .
Chen  et al.   [ 78 ] introduce a prediction denoising approach for a cross-domain segmentation task.
In this study, a key component is introducing pixel-wise denoising via uncertainty evaluation using Monte Carlo Dropout   [ 85 , 86 ] , which calculates the standard deviation of several stochastic outputs and keeps it under a manually-designed threshold.
In this way, the noisy pseudo-labels can be filtered out, helping improve pseudo-label quality to achieve effective adaptation.
Xu  et al.   [ 79 ] also propose an uncertainty-guided pseudo-labeling denoising scheme, but they use soft label correction instead of manually discarding unreliable data points. Specifically, they first identify mislabeled data points by utilizing a joint distribution matrix  [ 87 , 88 ] , and then assign larger confident weights to those with higher certainty based on Monte Carlo Dropout.
Combining target data and the corresponding rectified pseudo-labeling, a commonly used cross-entropy objective function can be leveraged for training the target model.
Sharing the similar idea, Hegde  et al.   [ 80 ] allocate lower weights for uncertain pseudo-labels, where the uncertainty is measured by prediction variance based on Monte Carlo Dropout  [ 85 , 86 ] .
Considering that using Monte Carlo Dropout  [ 85 ] for uncertainty estimation requires manual hyperparameter adjustment  [ 89 ] , Roy  et al.   [ 81 ] quantify source model’s uncertainty using a Laplace approximation   [ 90 , 91 ] .
For model training, they assign smaller weights to those target samples that are farther away from source hypothesis (measured by uncertainty), avoiding misalignment of dissimilar samples.
Pei  et al.   [ 82 ] tackle the uncertainty issue from the perspective of improving source model transferability .
Specifically, they estimate channel-aware transferability of the source model to target data based on an uncertainty distance, which measures the closeness between target instances and source distribution.
With the aim of dynamically exploiting the source model and target data, the target model obtains the source knowledge from the transferable channels and neglects those less-transferable ones.
Unlike previous studies, Li  et al.   [ 83 ] quantify uncertainty using self-entropy and propose a self-entropy descent mechanism to seek the optimal confidence threshold for robust pseudo-labeling of target data.
They also leverage false negative mining and mosaic augmentation  [ 92 ] to further eliminate the negative influence of noisy labels to enhance adaptation performance.

 
 
 Fig. 9: Illustration of Uncertainty-Guided Adaptation methods for source-free unsupervised domain adaptation.
These studies utilize uncertainty to guide target predictions, and such valuable information can be measured by Monte Carlo Dropout, entropy, etc.
 
 
 
 Fig. 10: Illustration of Hidden Structure Mining methods for source-free unsupervised domain adaptation.
These methods take into consideration intrinsic feature structures of target domain and iterate between target model refinement and clustering centroid update. 
 
 
 

#### II-B 5 Hidden Structure Mining

 
 Many studies  [ 20 , 93 , 94 , 95 , 96 , 97 , 98 , 99 ] take into consideration intrinsic feature structures of target domain and update the target model via clustering-aware pseudo-labeling.
In Fig.  10 , we illustrate the main idea of hidden structure mining methods.

 
 
 For example, Yang  et al.   [ 20 ] observe that target data can intrinsically form a certain cluster structure that can be used for domain adaptation.
Specifically, they estimate affinity among target data by taking into account the neighborhood patterns captured from local-, reciprocal-, and expanded-neighbors.
Source-free adaptation is achieved by encouraging consistent predictions for those with high affinity.
Similarly, Yang  et al.   [ 93 ] also exploit neighborhood structure information of target data.
They propose a local structure clustering strategy to encourage prediction consistency among k k -nearest target features, thus pushing target data with semantically similar neighbors closer.
Tang  et al.   [ 94 ] leverage semantic constraints hidden in geometric structure among target data to encourage robust clustering based on a cognition mechanism  [ 100 ] .
Source hypothesis transfer (SHOT)  [ 95 ] and SHOT++  [ 96 ] attempt to mine the feature structure of the target domain, but they cannot fully exploit the meaningful context since the used self-supervised pseudo-labeling does not take into account each dimension’s covariance in the feature space.
To address this issue, Lee  et al.   [ 97 ] utilize GMM in the target domain to obtain data structure, and design a joint model-data structure score to concurrently exploit source and target knowledge.
Li  et al.   [ 98 ] utilize neighbor structure information from a new aspect by proposing a generic and model smoothness-assisted Jacobian norm regularization term, which is used to manipulate the consistency between each target instance and its neighbors.
This Jacobian norm regularizer can be easily plugged into existing source-free domain adaptation frameworks for boosting performance.

 
 
 Different from the above mentioned methods, some studies tackle source-free domain adaptation from other perspectives.
Li  et al.   [ 101 ] achieve data-free adaptation from an adversarial-attack aspect , which aims to generate adversarial target instances by adding diverse perturbations to attack the target model.
Then, mutual information maximization is performed between representations extracted by the source and target model for the same target instance.
The above two steps are performed alternatively, by which the domain-invariant source knowledge can be preserved and the rich target patterns can be well explored.
Instead of exploring domain-invariant features for cross-domain knowledge transfer, Wang  et al.   [ 102 ] mine domain-invariant parameters stored in the source model.
They assume that only partial domain-invariant parameters of the source model contribute to domain adaptation, and their goal is to capture such parameters while penalizing the domain-specific ones.
Liang  et al.   [ 103 ] explore source-free adaptation from the perspective of minimum centroid shift , with the aim of searching a subspace where target prototypes are mildly shifted from source prototypes.
An alternating optimization scheme is leveraged for model convergence and target pseudo-label update.
Inspired by maximum classifier discrepancy  [ 14 ] , Yang  et al.   [ 104 ] introduce an auxiliary bait classifier for cross-domain feature alignment combined with the source anchor classifier.
These two classifiers aim to collaboratively push uncertain target representations to the correct side of the source classifier boundary.

 
 
 

#### II-B 6 Challenges and Insight

 
 We classify existing model fine-tuning methods for SFUDA into five subcategories. The challenges of methods in each subcatetory and our insights are presented below.

 
 
 
 • 
 
 The methods in the first subcategory, i.e. , self-supervised knowledge distillation , interpret source-free domain adaptation as a knowledge extraction and transfer process, aiming to learn domain-invariant feature representations.
Most exiting studies transfer source knowledge to the target model via a mean teacher strategy  [ 57 ] , where teacher weights are an exponential moving average of student weights.
Hence, model parameters of both teacher and student networks are tightly coupled, which may lead to a performance bottleneck.
A possible solution is to introduce a dual-student framework and let one student learn features flexibly, which may disentangle teacher-student weights to some extent  [ 105 ] .

 

 • 
 
 The second subcategory, i.e. , domain alignment via statistics , leverages batch statistics stored in a pre-trained source model to approximate distribution of inaccessible source data.
Compared with other categories, these statistics-based methods are lightweight and prone to generalize to other tasks, since they require only a few update steps of batch-wise statistics parameters and are potentially applicable to real-time deployment  [ 65 ] .
However, they are not suitable for problems that use deep network architectures without batch normalization layers.

 

 • 
 
 The methods in the third subcategory, i.e. , contrastive learning , aim to bring similar-class samples closer and push dissimilar-class samples apart based on generated target pseudo-labels.
Therefore, if the pseudo-labels contain much noise, these methods may suffer from substantial performance degradation.
Moreover, a memory bank is usually required to store the similarity relationship between current and historical feature representations of target data, which could bring memory burden.
It is interesting to investigate the storage- and transmission-efficient contrastive learning strategies in source-free settings.
In addition, several recent studies  [ 106 , 107 ] have shown that data pair construction is crucial for effective contrastive learning.
One solution is utilizing contrastive information between target data and their augmented ones.
Previous studies  [ 108 ] often use either strong or weak transformations for data augmentation, where strong augmentations mostly distort the structures of original images ( e.g. , shape distortion) while weak augmentations usually limit transformations to preserve the images’ structures ( e.g. , flip).
Here we propose to dynamically mix strong and weak augmentation of target data, which may help learn more robust representations.

 

 • 
 
 The methods in fourth subcategory, i.e. , uncertainty-guided adaptation , focus on reducing prediction uncertainty of target data.
Many studies  [ 78 , 79 ] use Monte Carlo Dropout for uncertainty estimation, but this technique requires specialized network architecture design and model training, bringing troublesome hyperparameter tuning  [ 89 ] .
A recent study  [ 78 ] points out that their method can only handle problems with minor domain shift, and performs poorly on problems with severe domain shift.
It is interesting to explore this challenging problem in the future.

 

 • 
 
 The last subcategory, i.e. , hidden structure mining , considers intrinsic clustering structure of the target domain, assuming that geometric structure of target data may provide informative context  [ 94 ] .
The advantage of these methods is that no auxiliary frameworks are required, and thus, they can be easily incorporated into other adaptation frameworks.
However, these methods have at least three disadvantages.
(1) Most existing studies need to iterate between feature clustering and model update, which may hinder training efficiency and cause a memory burden.
(2) These methods may be infeasible to handle extremely large-scale datasets due to the difficulty of saving global latent feature embeddings of the whole dataset  [ 109 ] .
(3) Most studies construct target geometric structures in Euclidean space, which may not be suitable for problems with non-Euclidean data such as graphs. Thus, how to improve training efficiency and deal with the large-size dataset, as well as mining geometry information of non-Euclidean data deserve further research.

 

 
 
 
 From the application perspective, computation-efficient approaches are more applicable for pixel-wise semantic segmentation tasks, which require higher resources compared with classification tasks.
And those memory-intensive approaches such as contrastive learning may be not suitable for semantic segmentations.
Moreover, it is worth noting that data generation methods detailed in Section  II-A can be used in conjunction with the model fine-tuning methods described in this section.
For instance, one can first generate a virtual source domain by selecting appropriate target samples, and thus a standard unsupervised domain adaptation framework could be applied.
To further exploit target information, we then take account of geometric structure of target samples and generate corresponding target pseudo-labels to fine-tune the target model.
These two steps can be optimized iteratively, helping generate more representative source domain and refine the target model.

 
 
 
 
 

## III Black-Box Source-Free Unsupervised Domain Adaptation 

 
 Different from white-box methods, in the setting of black-box source-free domain adaptation, both the source data { X S , Y S } \{X_{S},Y_{S}\} and detailed parameters of the source model Φ S \Phi_{S} are not accessible .
Only the hard or soft model predictions of the target data X T X_{T} from the source model Φ S \Phi_{S} are leveraged for domain adaptation.
Depending on the utilization of the black-box predictor, the existing black-box SFUDA studies can be mainly divided into three categories: Self-Supervised Knowledge Distillation , Pseudo-Label Denoising , and Generative Distribution Alignment methods, with details introduced below.

 
 

### III-A Self-Supervised Knowledge Distillation 

 
 Some studies  [ 110 , 111 , 112 , 113 , 114 , 115 ] construct a teacher-student-style network architecture with knowledge distillation to translate the source knowledge to the target domain in a self-supervised manner.
For instance, Liang  et al.   [ 110 , 111 ] enforce output consistency between a source model ( i.e. , teacher) and a customized target model ( i.e. , student) via a self-distillation loss.
Specifically, a memory bank is first constructed to store the prediction of each target sample based on the black-box source model.
This source model then acts as a teacher to maintain an exponential moving averaging of source and target prediction following  [ 116 , 117 ] .
Additionally, structural regularization on the target domain is further incorporated during adaptation for more effective knowledge distillation.
Similarly, Liu  et al.   [ 112 , 113 ] 
employ an exponential mixup decay scheme to explicitly keep prediction consistency of source and target domains, thus gradually capturing target-specific feature representations and obtaining the target pseudo-labels.
Xu  et al.   [ 114 ] extend the teacher-student paradigm from image analysis to more challenging video analysis, where not only spatial features but also temporal information are taken into consideration during domain adaptation.
For knowledge distillation, the target model is regarded as a student, which aims to learn similar predictions generated by a teacher ( i.e. , source) model.
The teacher model is meanwhile updated to maintain an exponential moving averaging prediction.
Instead of distilling knowledge between source and target domains, Peng  et al.   [ 115 ] transfer knowledge between the target network and its subnetwork in a mutual way, where the subnetwork is a slimmer version generated from the original target network following Yang  et al.   [ 118 ] .
And target features are extracted by leveraging multi-resolution input images, helping improve the generalization ability of the target network.
Moreover, a novel data augmentation strategy, called frequency MixUp, is proposed to emphasize task-related regions-of-interests while simultaneously reducing background interference.

 
 
 

### III-B Pseudo-Label Denoising 

 
 Some studies  [ 119 , 120 ] tackle domain shift by carefully denoising unreliable target pseudo-labels.
For example, Zhang  et al.   [ 119 ] combat noisy pseudo-labels via noise rate estimation , which first preserves more training samples at the start of the training process following  [ 121 ] and then gradually filters out the noisy ones based on their loss values as training proceeds.
The pseudo-labels are iteratively refined according to a category-dependent sampling strategy, encouraging the model to capture more diverse representations to improve model generalization ability.
Different from Zhang  et al.   [ 119 ] that only select part of reliable target data during model training, Luo  et al.   [ 120 ] take into account all target data and rectify noisy pseudo-labels from a negative learning aspect.
Specifically, their approach assigns complementary ground-truth labels for each target sample, helping alleviate error accumulation for noisy prediction.
Moreover, a maximum squares objective function is utilized as confidence regularization to prevent the target model from being trapped in easy sample training.
Yang  et al.   [ 122 ] incorporate pseudo-label denoising and self-supervised knowledge distillation into a unified framework.
Specifically, domain knowledge is first distilled from the trained source predictor to warm up the target model by an exponential moving averaging scheme.
The unlabeled target domain is then split into two subsets ( i.e. , easy and hard groups) according to their adaptation difficulty  [ 123 ] , and the MixMatch strategy  [ 124 ] is leveraged to progressively exploit all target representations.
In this way, the noise accumulation is further suppressed, thereby improving the efficacy of pseudo-label denoising.

 
 
 

### III-C Generative Distribution Alignment 

 
 Different from the above methods, some studies perform distribution alignment across domains in a generative way.
For instance, Yeh  et al.   [ 125 ] perform domain adaptation by maximizing the lower bound in variational inference.
Specifically, they construct a generation path as well as an inference path , where the generation path produces a prior feature distribution derived from predicted category labels, and the inference path approximates a posterior feature distribution based on each target instance.
The latent distribution alignment can be achieved by maximizing the evidence lower vound in variational inference for cross-domain adaptation.
Similarly, Yang  et al.   [ 126 ] also construct the generation and inference paths, but they achieve adaptation via minimizing the upper bound of the prediction error of target data in variational inference.
Zhang  et al.   [ 127 ] achieve source-free adaptation by first building multiple source models and then generating a virtual intermediate surrogate domain to select target samples with minimum inconsistency predicted by the source models.
The knowledge transfer is achieved by feature distribution alignment between the virtual surrogate domain and the target domain based on a joint probability maximum mean discrepancy  [ 128 ] .

 
 
 

### III-D Challenges and Insight 

 
 In this section, we classify existing black-box SFUDA methods into three categories based on how they utilize the noisy target predictions.
The challenges of each category and our insights are presented below.

 
 • 
 
 The first category, i.e. , self-supervised knowledge distillation , aims to gradually transfer source knowledge to a customized target model by enforcing output consistency between a teacher (source) and a student (target) network.
This learning strategy has also been used in white-box SFUDA (see Section  II-B1 ).
The difference is that model weights of student networks are accessible in white-box SFUDA methods, but not in black-box SFUDA.
In black-box SFUDA, instead of leveraging any parameter details, the teacher network is only updated by source predictions and historical target predictions.
The two items are typically weighted by a momentum factor, which helps dynamically adjust their contributions.
Self-supervised knowledge distillation has shown promising performance in object recognition  [ 110 ] , semantic segmentation  [ 112 ] , and video action recognition tasks  [ 114 ] .

 

 • 
 
 The methods in the second category, i.e. , pseudo-Label denoising , tackle black-box SFUDA from the perspective of noisy label rectification.
It has shown that a pseudo-label denoising approach  [ 119 ] has inferior performance than the self-supervised knowledge distillation method  [ 110 ] .
The reason may be that the former  [ 119 ] only focuses on noisy prediction itself while neglecting target data structure that is well considered in the latter  [ 110 ] .
Considering that pseudo-label denoising methods can tackle unbalanced label noise via noise rate estimation, combining pseudo-label denoising with self-supervised knowledge distillation strategies will be a promising future direction, especially in class-imbalance scenarios. Moreover, if the black-box predictor only provides one-hot hard predictions instead of probability predictions, the utility of methods in this subcategory will be greatly reduced. The reason is that the noise rate cannot be well estimated in practice, e.g. , there is nearly no difference between the output of [ 0.45 , 0.55 ] [0.45,0.55] and that of [ 0.05 , 0.95 ] [0.05,0.95] because the source predictor produces the same output ( i.e. , [ 0 , 1 ] [0,1] ).

 

 • 
 
 The third category, i.e. , generative distribution alignment , attempts to perform domain adaptation by minimizing feature distribution discrepancy across the source and target domains.
Since source distribution is inaccessible in black-box models, some generative approaches are utilized to generate such reference distribution for target data to align with, including variational autoencoder  [ 125 , 126 ] and surrogate source domain construction  [ 127 ] .
These methods are more suitbale for recognition/classification tasks, but less suitable for semantic segmentation tasks.
For example, generating surrogate feature distribution of an object ( e.g. , car) is usually easier than that of a semantic scene ( e.g. , cityscape), since the latter contains different objects and thus the pixel-wise neighborhood relationship is difficult to model in practice.

 

 
 
 
 Besides the strategies proposed above, it is also crucial to build a general and robust black-box source model , with which the target predictions tend to be more accurate.
To achieve that, one possible solution is augmenting the diversity of source data ( e.g. , adding some perturbation) before constructing the source model, which may eliminate style discrepancy between two domains, thus improving the generalization ability of the source model.
Another solution is using soft probability labels instead of hard one-hot labels ( e.g. , [0, 1]) for model training, which prevents the source model from being over-confident and helps enhance its generalizability.
Compared to white-box methods, there are relatively few black-box SFUDA methods as well as benchmark datasets, which needs to be further explored.

 
 
 TABLE I: Commonly used datasets for evaluating the performance of source-free unsupervised domain adaptation (SFUDA) approaches. 
 
 
 
 Dataset | 
 Domain # | 
 Instance # | 
 Category # | 
 Description | 

 
 
 
 Digit Recognition | 
 | 
 | 
 | 
 | 

 
 Digits-Five  [ 129 ] | 
 5 | 
 215,695 | 
 10 | 
 MNIST  [ 130 ] , SVHN  [ 131 ] , USPS  [ 132 ] , MNIST-M  [ 13 ] , Synthetic Digits  [ 13 ] | 

 
 Semantic Segmentation | 
 | 
 | 
 | 
 | 

 
 Segmentation datasets | 
 4 | 
 45,766 | 
 - | 
 GTA5  [ 133 ] , Cityscapes  [ 134 ] , SYNTHIA  [ 135 ] , NTHU  [ 136 ] | 

 
 Object Recognition | 
 | 
 | 
 | 
 | 

 
 Office-31  [ 137 ] | 
 3 | 
 4,652 | 
 31 | 
 Amazon, Webcam, DSLR | 

 
 Office-Home  [ 138 ] | 
 4 | 
 15,500 | 
 65 | 
 Artistic, Clip Art, Product, Real-World | 

 
 VisDA  [ 139 ] | 
 2 | 
 280,000 | 
 12 | 
 Synthetic and real images | 

 
 Office-Caltech-10  [ 140 ] | 
 4 | 
 2,533 | 
 10 | 
 Amazon, DSLR, Webcam, Caltech10 | 

 
 ImageCLEF-DA  [ 141 ] | 
 4 | 
 2,400 | 
 12 | 
 Caltech-256  [ 142 ] , ImageNet ILSVRC2012  [ 16 ] , PASCAL VOC2012  [ 143 ] , Bing  [ 144 ] | 

 
 PACS  [ 7 ] | 
 4 | 
 9,991 | 
 7 | 
 Art painting, Cartoon, Photo, Sketch | 

 
 DomainNet  [ 129 ] | 
 6 | 
 600,000 | 
 345 | 
 Clipart, Infograph, Painting, Quickdraw, Real, Sketch | 

 
 MiniDomainNet  [ 145 ] | 
 4 | 
 140,000 | 
 126 | 
 Clipart, Painting, Real, Sketch | 

 
 PointDA-10  [ 146 ] | 
 3 | 
 33,067 | 
 10 | 
 ModelNet  [ 147 ] , ShapeNet  [ 148 ] , ScanNet  [ 149 ] | 

 
 Face Anti-Spoofing | 
 | 
 | 
 | 
 | 

 
 Face datasets | 
 4 | 
 7,130 | 
 - | 
 Replay-Attack  [ 150 ] , OULU-NPU  [ 151 ] , CASIA-MFSD  [ 152 ] , MSU-MFSD  [ 153 ] | 

 
 LiDAR Detection | 
 | 
 | 
 | 
 | 

 
 LiDAR datasets | 
 3 | 
 158,510 | 
 - | 
 Waymo  [ 154 ] , KITTI  [ 155 ] , nuScenes  [ 156 ] | 

 
 Video Action Recognition | 
 | 
 | 
 | 
 | 

 
 UCF-HMDB full {}_{\text{full}}   [ 157 ] | 
 2 | 
 3,209 | 
 12 | 
 UCF101  [ 158 ] , HMDB51  [ 159 ] | 

 
 Sports-DA  [ 160 ] | 
 3 | 
 40,718 | 
 23 | 
 UCF10  [ 158 ] , Sports-1M  [ 161 ] , Kinetics  [ 162 ] | 

 
 Daily-DA  [ 160 ] | 
 4 | 
 18,949 | 
 8 | 
 ARID  [ 163 ] , HMDB51  [ 159 ] , Moments-in-Time  [ 164 ] , Kinetics  [ 162 ] | 

 
 Traffic Sign Recognition | 
 | 
 | 
 | 
 | 

 
 Sign datasets | 
 2 | 
 151,839 | 
 43 | 
 Syn.Signs  [ 165 ] , GTSRB  [ 166 ] | 

 
 Image Classification | 
 | 
 | 
 | 
 | 

 
 VLCS  [ 167 ] | 
 4 | 
 10,729 | 
 5 | 
 Caltech101  [ 168 ] , LabelMe  [ 169 ] , SUN09  [ 170 ] , VOC2007  [ 143 ] | 

 
 Medical Data | 
 | 
 | 
 | 
 | 

 
 BraTS2018  [ 171 ] | 
 2 | 
 285 | 
 - | 
 Cross-disease (high and low grade glioma), cross-modality (T1, T1ce, T2, FLAIR) | 

 
 MMWHS  [ 172 ] | 
 2 | 
 40 | 
 - | 
 Cross-modality (magnetic resonance imaging, computed tomography) | 

 
 Brain skull stripping  [ 173 ] | 
 3 | 
 35 | 
 - | 
 NFBS  [ 174 ] , ADNI  [ 175 ] , dHCP  [ 176 ] | 

 
 Polyp segmentation | 
 4 | 
 2,718 | 
 - | 
 CVC-ClincDB  [ 177 ] , Abnormal Symptoms  [ 178 ] , ETIS-Larib  [ 179 ] , EndoScene  [ 180 ] | 

 
 EEG MI Classification  [ 127 ] | 
 4 | 
 528 | 
 2/4 | 
 MI2-2  [ 181 ] , MI2-4  [ 181 ] , MI2015  [ 182 ] , AlexMI  [ 183 ] | 

 
 Prostate segmentation | 
 2 | 
 682 | 
 - | 
 NCI-ISBI 2013 Challenge  [ 184 ] , PROMISE12 challenge  [ 185 ] | 

 
 Optic disc cup segmentation | 
 3 | 
 660 | 
 - | 
 REFUGE  [ 186 ] , RIMONE-r3  [ 187 ] , Drishti-GS   [ 188 ] | 

 
 Autism diagnosis | 
 4 | 
 411 | 
 2 | 
 NYU, USM, UCLA, and UM of ABIDE dataset  [ 189 ] | 

 

 
 
 
 

## IV Discussion 

 
 In this section, we first compare the white-box and black-box SFUDA methods and then summarize several useful strategies to improve model generalizability.
We also list datasets commonly used in the field in Table  I .

 
 

### IV-A Comparison of White-Box and Black-Box SFUDA 

 
 By comparing existing white-box and black-box SFUDA methods, we have the following interesting observations.

 
 • 
 
 Compared with black-box SFUDA that cannot access any source parameters, white-box SFUDA is capable of mining more source knowledge ( e.g. , batch statistics) that facilitates more effective domain adaptation.

 

 • 
 
 White-box SFUDA methods may suffer from data privacy leakage problems  [ 119 ] .
For instance, Yin  et al.   [ 190 ] reveal that raw data can be recovered based on source image distribution via a deep inversion technique.
Using a membership inference attack strategy  [ 191 , 192 ] , it is possible to infer whether a given sample exists or not in training dataset, thereby revealing private information.
The black-box SFUDA can help protect data privacy because only application programming interface (API) is accessible while detailed model weights are withheld, but it may suffer from performance degradation of cross-domain adaptation.

 

 • 
 
 Most white-box SFUDA methods assume that model architecture is shared between source and target domains, while the black-box SFUDA methods try to design task-specific target models for knowledge transfer.
Such flexible model design in black-box SFUDA methods is very useful for target users with low computation resources, since they can design more efficient and lightweight target models for domain adaptation.

 

 • 
 
 Black-box SFUDA methods neither require data synthesis nor model fine-tuning, which helps to accelerate the convergence process of model training.
In contrast, white-box methods are usually computationally intensive and time-consuming.
For instance, it is reported that the computational cost of a black-box SFUDA method  [ 127 ] is 0.83 s s while that of two competing white-box methods are 3.17 s s   [ 95 ] and 22.43 s s   [ 193 ] , respectively, reflecting the computation efficiency of black-box SFUDA.

 

 
 
 
 In summary, when using white-box and black-box SFUDA methods, we have to make a trade-off between obtaining better performance, protecting confidential information, and reducing computational and memory costs.

 
 
 

### IV-B Useful Strategies for Improved Generalizability 

 
 To facilitate research practice in this field, we summarize several useful techniques that could be used to improve the generalizability of learning models for source-free unsupervised domain adaptation.

 
 

#### IV-B 1 Entropy Minimization Loss

 
 Most SFUDA methods utilize an entropy minimization loss  [ 194 ] to reduce uncertainty of model predictions  [ 95 , 195 , 196 , 197 , 198 , 199 , 113 , 112 , 60 , 76 , 28 , 200 , 201 ] .
This simple yet effective strategy encourages the model to generate one-hot predictions for more confident learning.

 
 
 

#### IV-B 2 Diversity Enforcing Loss

 
 To prevent predicted labels from collapsing to categories with larger number of samples, many studies leverage a diversity enforcing loss to encourage diverse predictions over target domain  [ 95 , 202 , 203 , 81 , 196 , 204 , 205 , 206 ] .
The usual practice is to maximize the entropy of empirical label distribution over the batch-wise average of model predictions.

 
 
 

#### IV-B 3 Label Smoothing Technique

 
 In source-free adaptation studies, a pre-trained source model is generally obtained via training on labeled source data before adaptation stages.
Currently, many studies use a label smoothing technique  [ 207 , 208 ] to produce a robust source model  [ 95 , 102 , 20 , 31 , 209 , 210 ] .
This technique aims to transform original training labels from hard labels ( e.g. , 1) to soft labels ( e.g. , 0.95), which prevents the source model from being over-confident, helping enhance its generalization ability.
Also, the experiments have shown that label smoothing can encourage closer representations of training samples from the same category  [ 207 ] .
With a more general and robust source model, it is likely to boost adaptation performance on target domain.

 
 
 

#### IV-B 4 Model Regularization

 
 Many regularization terms are utilized in existing SFUDA methods by incorporating some prior knowledge.
For instance, an early learning regularization  [ 40 , 211 , 121 ] is used to prevent the model from over-fitting to label noise.
A stability regularization  [ 39 , 212 , 213 , 214 ] is leveraged to prevent parameters of the target model to deviate from those of the source model.
A local smoothness regularization  [ 39 , 215 ] is used to encourage output consistency between the target model and its noise-perturbed counterpart, helping improve robustness of the target model.
A mixup regularization  [ 110 , 31 , 216 , 115 , 217 ] is used to enforce prediction consistency between original and augmented data, which can mitigate the negative influence of noisy labels.

 
 
 

#### IV-B 5 Confidence Thresholding

 
 Many studies leverage pseudo-labeling to train the target model in a self-supervised way.
Instead of utilizing a manually-designed threshold to identify reliable/confident pseudo-labels, a commonly used strategy is automatically learning the confidence threshold for reliable pseudo-label selection  [ 218 ] .
To further tackle the class-imbalance problem, some studies  [ 213 , 219 , 76 , 79 , 220 ] propose to learn dynamic threshold for each category, which provides a fair chance for categories with limited samples to generate pseudo-labels for self-training.

 
 
 
 
 

## V Future Outlook 

 

### V-A Multi-Source/Target Domain Adaptation 

 
 To utilize diverse and rich information of multiple domains, a few studies  [ 30 , 193 , 205 , 221 , 222 ] propose multi-source data-free adaptation to transfer source knowledge to the target domain.
Tian  et al.   [ 30 ] introduce a sample transport learning method, but the proposed model is shallow, and thus cannot handle highly nonlinear feature extraction.
To tackle this problem, several deep learning based models  [ 193 , 205 ] are proposed.
But they ignore the fact that the generated target pseudo-labels may be noisy, which may cause training bias when matching target domains with large domain gaps.
The key to solving problems with multi-source domains is quantifying the transferability of different source models and utilizing their complementary information for promoting cross-domain adaptation.
Even several strategies are proposed ( e.g. , aggregation weight  [ 193 ] and source-specific transferable perception  [ 205 ] ), more explorations are encouraged to address the problem of negative transfer during cross-domain knowledge transfer.

 
 
 A few studies  [ 223 , 224 , 225 , 226 , 227 ] incorporate federated learning into domain adaptation scenarios.
Federated learning  [ 228 , 229 , 230 ] is a decentralized scheme to facilitate collaborative learning among multiple distributed clients without sharing training data or model parameters.
The constraint that prevents data and parameter transmission across different source domains is not required in multi-source-free domain adaptation.
For instance, federated adversarial domain adaptation (FADA) introduced by Peng  et al.   [ 223 ] is among the first attempts to propose the concept of federated domain adaptation, which employs a dynamic attention mechanism to transfer knowledge from multi-source domains to an unlabeled target domain.
In this method, each source model needs to synchronize with the target domain after each training batch, resulting in huge computation costs and potential risk of privacy leakage  [ 231 ] .
To tackle this problem, Feng  et al.   [ 224 ] introduces a consensus focus schema that greatly improves communication efficiency for decentralized domain adaptation.
Moreover, Song  et al.   [ 226 ] utilize a homomorphic encryption approach for privacy protection, and Qin  et al.   [ 227 ] introduce a flexible uncertainty-aware strategy for reliable source selection.
However, current federated learning studies usually produce a common model for all clients without considering heterogeneity of data distribution of different clients.
Therefore, the common model cannot adapt to each client adaptively, which may affect adaptation performance.
It would be very interesting to investigate personalized federated learning  [ 232 ] , with which current or new clients can easily adapt to their own local dataset by performing a few optimization steps.
Besides, all the methods mentioned above require labeled data from multiple sources to train a federated model, inevitably increasing annotation costs.
Therefore, approaches that effectively exploit unlabeled data from multiple source domains in a decentralized way are urgently needed.

 
 
 On the other hand, Yao  et al.   [ 233 ] and Shenaj  et al.   [ 234 ] have proposed several federated multi-target domain adaptation strategies for transferring knowledge of a labeled source server to multiple unlabeled target clients.
More advanced techniques for federated multi-target domain adaptation are highly desirable, by considering computation and communication cost, annotation burden, and privacy protection of different target domains.

 
 
 

### V-B Test-Time Domain Adaptation 

 
 Most SFUDA approaches require pre-collected unlabeled target data for model training, termed “training-time adaptation”.
Test-time adaptation  [ 198 , 235 , 236 , 237 , 238 ] has been investigated by adapting the source model to the target domain during inference procedure only.
The advantages of test-time adaptation are mainly twofold:
(1) The adaptation process does not need iterative training, which greatly improves computational efficiency, so the model can be easily deployed in an online manner.
(2) Without relying on target training data, test-time adaptation is expected to be well generalized to diverse target domains.
Even current studies have made promising achievements, there are still some problems worth exploring, listed as follows.

 
 
 Some studies  [ 198 , 239 , 240 ] need to access batch-sized ( 1) target samples during inference, which cannot handle scenarios where target samples arrive one-by-one sequentially.
Two studies  [ 241 , 242 ] perform image-wise adaptation rather than batch-wise adaptation, but they cannot deal with cases with large distribution shift.
It is interesting to explore how to handle the scenarios where test instances come from continuous changeable domains in the future.
Additionally, the solutions on how to adaptively exploit test data can be further explored  [ 243 ] , such as adjusting model weights dynamically based on sample discrepancy across domains.

 
 
 

### V-C Open/Partial/Universal-Set Domain Adaptation 

 
 This survey focuses on close-set source-free domain adaptation, where the label space of source and target domains is consistent.
But practical scenarios are much more complicated when the category shift issue occurs across different domains.
There are three non-close-set scenarios: (1) open-set ( C s ⊂ C t C_{s}\subset C_{t} ) problems, (2) partial-set ( C s ⊃ C t C_{s}\supset C_{t} ) problems, and (3) universal-set ( C s \ C t ≠ ∅ C_{s}\backslash C_{t}\neq\varnothing and C t \ C s ≠ ∅ C_{t}\backslash C_{s}\neq\varnothing , C s ⊂ C t C_{s}\subset C_{t} , C s ⊃ C t C_{s}\supset C_{t} ) problems, where C s C_{s} and C t C_{t} denote the category label set for source and target domains, respectively.

 
 
 Currently, only a few studies  [ 18 , 244 , 245 ] attempt to handle the category shift problem in source-free adaptation scenarios, including out-of-distribution data construction  [ 18 , 244 ] , neighborhood clustering learning  [ 246 ] , uncertainty-based progressive learning  [ 247 ] , and mutual information maximization  [ 204 ] .
The main idea behind these studies is to recognize out-of-source distribution samples and improve generalization ability of the source model.
However, the model performance of existing studies is not quite satisfactory due to the inaccessibility of valuable category-gap knowledge.
One possible solution is to adaptively learn a threshold instead of using a fixed one to determine the acceptance/rejection of each target sample as a “known” category via some similarity measurement.
Moreover, some strategies used in non-source-free domain adaptation can also be borrowed, such as distribution weighted combining rule  [ 248 ] , category-invariant representation learning  [ 249 ] , one-vs-all learning scheme  [ 250 ] , and global-local optimization  [ 251 ] .

 
 
 

### V-D Flexible Target Model Design 

 
 For black-box SFUDA methods, due to unavailability of structure and parameters of target model, one usually has to manually design a target model.
For instance, Liu  et al.   [ 112 ] choose a U-Net based framework as the target model for segmentation.
However, such manually designed architectures may be not suitable when adapting to the target domain.
It is expected that the automatic design of target models, e.g. , using neural architecture search (NAS)  [ 252 , 253 , 254 ] , helps improve the learning performance. Considering that NAS has recently become a popular strategy for searching proper network architectures in deep learning, we can integrate it into SFUDA scenarios to find more proper and efficient target models.
And how to balance the search space and search cost of network parameters can be further investigated.
Moreover, hyperparameters used in NAS ( e.g. , optimizer strategy, weight decay regularization) should be carefully considered since they also have a significant impact on network performance  [ 255 ] .

 
 
 

### V-E Cross-Modality Domain Adaptation 

 
 Existing studies mainly focus on one single modality for domain adaptation, while a few studies perform cross-modality adaptation in source-free settings  [ 29 , 256 ] .
For instance, for medical data analysis, the acquisition expense of computed tomography (CT) scans is generally less than that of magnetic resonance imaging (MRI) scans, hence it may greatly reduce annotation cost for a segmentation task when transferring a source model trained on CT images to MRI scans  [ 29 ] .
Moreover, in computer vision field, it would be promising to investigate cross-modality adaptation in the future, e.g. , image → \rightarrow video, which aims to achieve video recognition based on the source model trained on the image dataset.
Also, how to effectively integrate multi-modality ( e.g. , image, sound, text, and video) data for domain adaptation in a source-free way is an interesting but not yet widely studied problem.

 
 
 

### V-F Continual/Lifelong Domain Adaptation 

 
 Most current studies focus on improving adaptation performance on the target domain while neglecting the performance on source domain, running the risk of catastrophic forgetting problems  [ 257 ] .
To address this issue, several solutions have been developed from different aspects, such as domain expansion  [ 258 ] , historical contrastive learning  [ 19 ] , domain attention regularization  [ 93 ] , and model perturbation  [ 259 ] , while there is still massive room for performance improvement.
Inspired by continual/lifelong learning  [ 260 , 261 , 262 , 263 ] , continual domain adaptation has recently made great progress by investigating gradient regularization  [ 264 ] , iterative neuron restoration  [ 265 ] , buffer sample mixture  [ 266 ] , etc.
The continual domain adaptation in source-free settings for mitigation of catastrophic forgetting remains an underdeveloped topic that can be further explored in the future.

 
 
 

### V-G Semi-Supervised Domain Adaptation 

 
 Source-free domain adaptation in semi-supervised settings ( i.e. , with a few labeled target data involved for model training) has also been explored in recent years  [ 267 , 197 , 268 ] .
It usually performs semi-supervised adaptation with the help of active learning  [ 269 , 270 ] , model memorization revelation  [ 271 ] , and consistency and diversity learning  [ 272 ] .
There is still a lot of space for improvement with a limited number of labeled target samples, e.g. , by fine-tuning the current source-free adaptation frameworks, but this is not the focus of this survey.

 
 
 
 

## VI Conclusion 

 
 In this paper, we provide a comprehensive review of recent progress in source-free unsupervised domain adaptation (SFUDA).
We classify existing SFUDA studies into white-box and black-box groups, and each group is further categorized based on different learning strategies.
The challenges of methods in each category and our insights are provided.
We then compare white-box and black-box SFUDA methods, discuss effective techniques for improving adaptation performance, and summarize commonly used datasets. We finally discuss promising future research directions.
It is worth noting that the research topic of source-free unsupervised domain adaptation is still in its early stages, and we hope this survey can spark new ideas and attract more researchers to advance this high-impact research field.

 
 
 

## References

 
 
 [1] 
 
A. Voulodimos, N. Doulamis, A. Doulamis, and E. Protopapadakis, “Deep learning
for computer vision: A brief review,” Computational Intelligence and
Neuroscience , vol. 2018, 2018.

 

 
 [2] 
 
M. Hassaballah and A. I. Awad, Deep learning in computer vision:
Principles and applications . CRC
Press, 2020.

 

 
 [3] 
 
D. Shen, G. Wu, and H.-I. Suk, “Deep learning in medical image analysis,”
 Annual Review of Biomedical Engineering , vol. 19, p. 221, 2017.

 

 
 [4] 
 
G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi, M. Ghafoorian,
J. A. Van Der Laak, B. Van Ginneken, and C. I. Sánchez, “A survey on
deep learning in medical image analysis,” Medical Image Analysis ,
vol. 42, pp. 60–88, 2017.

 

 
 [5] 
 
D. W. Otter, J. R. Medina, and J. K. Kalita, “A survey of the usages of deep
learning for natural language processing,” IEEE Transactions on Neural
Networks and Learning Systems , vol. 32, no. 2, pp. 604–624, 2020.

 

 
 [6] 
 
T. Young, D. Hazarika, S. Poria, and E. Cambria, “Recent trends in deep
learning based natural language processing,” IEEE Computational
Intelligence Magazine , vol. 13, no. 3, pp. 55–75, 2018.

 

 
 [7] 
 
D. Li, Y. Yang, Y.-Z. Song, and T. M. Hospedales, “Deeper, broader and artier
domain generalization,” in Proceedings of the IEEE International
Conference on Computer Vision , 2017, pp. 5542–5550.

 

 
 [8] 
 
S. Sankaranarayanan, Y. Balaji, A. Jain, S. N. Lim, and R. Chellappa,
“Learning from synthetic data: Addressing domain shift for semantic
segmentation,” in Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition , 2018, pp. 3752–3761.

 

 
 [9] 
 
K. Zhou, Z. Liu, Y. Qiao, T. Xiang, and C. C. Loy, “Domain generalization: A
survey,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2022.

 

 
 [10] 
 
M. Wang and W. Deng, “Deep visual domain adaptation: A survey,”
 Neurocomputing , vol. 312, pp. 135–153, 2018.

 

 
 [11] 
 
H. Guan and M. Liu, “Domain adaptation for medical image analysis: A
survey,” IEEE Transactions on Biomedical Engineering , vol. 69, no. 3,
pp. 1173–1185, 2021.

 

 
 [12] 
 
J. Dong, Y. Cong, G. Sun, Z. Fang, and Z. Ding, “Where and how to transfer:
Knowledge aggregation-induced transferability perception for unsupervised
domain adaptation,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2021.

 

 
 [13] 
 
Y. Ganin and V. Lempitsky, “Unsupervised domain adaptation by
backpropagation,” in International Conference on Machine
Learning . PMLR, 2015, pp. 1180–1189.

 

 
 [14] 
 
K. Saito, K. Watanabe, Y. Ushiku, and T. Harada, “Maximum classifier
discrepancy for unsupervised domain adaptation,” in Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition , 2018, pp.
3723–3732.

 

 
 [15] 
 
Y. Fang, M. Wang, G. G. Potter, and M. Liu, “Unsupervised cross-domain
functional MRI adaptation for automated major depressive disorder
identification,” Medical Image Analysis , p. 102707, 2022.

 

 
 [16] 
 
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A
large-scale hierarchical image database,” in 2009 IEEE Conference on
Computer Vision and Pattern Recognition . IEEE, 2009, pp. 248–255.

 

 
 [17] 
 
G. K. Nayak, K. R. Mopuri, S. Jain, and A. Chakraborty, “Mining data
impressions from deep models as substitute for the unavailable training
data,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2021.

 

 
 [18] 
 
J. N. Kundu, N. Venkat, R. V. Babu et al. , “Universal source-free
domain adaptation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2020, pp. 4544–4553.

 

 
 [19] 
 
J. Huang, D. Guan, A. Xiao, and S. Lu, “Model adaptation: Historical
contrastive learning for unsupervised domain adaptation without source
data,” Advances in Neural Information Processing Systems , vol. 34,
pp. 3635–3649, 2021.

 

 
 [20] 
 
S. Yang, J. van de Weijer, L. Herranz, S. Jui et al. , “Exploiting the
intrinsic neighborhood structure for source-free domain adaptation,”
 Advances in Neural Information Processing Systems , vol. 34, pp.
29 393–29 405, 2021.

 

 
 [21] 
 
J. N. Kundu, S. Bhambri, A. Kulkarni, H. Sarkar, V. Jampani, and R. V. Babu,
“Concurrent subsidiary supervision for unsupervised source-free domain
adaptation,” in European Conference on Computer Vision . Springer, 2022, pp. 177–194.

 

 
 [22] 
 
Y. Liu, W. Zhang, and J. Wang, “Source-free domain adaptation for semantic
segmentation,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , 2021, pp. 1215–1224.

 

 
 [23] 
 
W. Chen, L. Lin, S. Yang, D. Xie, S. Pu, Y. Zhuang, and W. Ren,
“Self-supervised noisy label learning for source-free unsupervised domain
adaptation,” arXiv preprint arXiv:2102.11614 , 2021.

 

 
 [24] 
 
C. Saltori, S. Lathuiliére, N. Sebe, E. Ricci, and F. Galasso,
“SF-UDA 3D : Source-free unsupervised domain adaptation for
LiDAR-based 3D object detection,” in 2020 International Conference
on 3D Vision (3DV) . IEEE, 2020, pp.
771–780.

 

 
 [25] 
 
Y. Liu, Y. Chen, W. Dai, M. Gou, C.-T. Huang, and H. Xiong, “Source-free
domain adaptation with contrastive domain alignment and self-supervised
exploration for face anti-spoofing,” in European Conference on
Computer Vision . Springer, 2022, pp.
511–528.

 

 
 [26] 
 
Y. Liu, W. Zhang, J. Wang, and J. Wang, “Data-free knowledge transfer: A
survey,” arXiv preprint arXiv:2112.15278 , 2021.

 

 
 [27] 
 
C. Yang, X. Guo, Z. Chen, and Y. Yuan, “Source free domain adaptation for
medical image segmentation with fourier style mining,” Medical Image
Analysis , vol. 79, p. 102457, 2022.

 

 
 [28] 
 
Y. Hou and L. Zheng, “Source free domain adaptation with image translation,”
 arXiv preprint arXiv:2008.07514 , 2020.

 

 
 [29] 
 
J. Hong, Y.-D. Zhang, and W. Chen, “Source-free unsupervised domain adaptation
for cross-modality abdominal multi-organ segmentation,”
 Knowledge-Based Systems , p. 109155, 2022.

 

 
 [30] 
 
Q. Tian, C. Ma, F.-Y. Zhang, S. Peng, and H. Xue, “Source-free unsupervised
domain adaptation with sample transport learning,” Journal of Computer
Science and Technology , vol. 36, no. 3, pp. 606–616, 2021.

 

 
 [31] 
 
Y. Ding, L. Sheng, J. Liang, A. Zheng, and R. He, “ProxyMix: Proxy-based
mixup training with label refinery for source-free domain adaptation,”
 arXiv preprint arXiv:2205.14566 , 2022.

 

 
 [32] 
 
M. Ye, J. Zhang, J. Ouyang, and D. Yuan, “Source data-free unsupervised domain
adaptation for semantic segmentation,” in Proceedings of the 29th ACM
International Conference on Multimedia , 2021, pp. 2233–2242.

 

 
 [33] 
 
Y. Du, H. Yang, M. Chen, J. Jiang, H. Luo, and C. Wang, “Generation,
augmentation, and alignment: A pseudo-source domain based method for
source-free domain adaptation,” arXiv preprint arXiv:2109.04015 ,
2021.

 

 
 [34] 
 
H. Yao, Y. Guo, and C. Yang, “Source-free unsupervised domain adaptation with
surrogate data generation,” in Proceedings of NeurIPS 2021 Workshop on
Distribution Shifts: Connecting Methods and Applications , 2021.

 

 
 [35] 
 
H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “Mixup: Beyond
empirical risk minimization,” arXiv preprint arXiv:1710.09412 , 2017.

 

 
 [36] 
 
S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein et al. ,
“Distributed optimization and statistical learning via the alternating
direction method of multipliers,” Foundations and
Trends® in Machine learning , vol. 3, no. 1, pp. 1–122,
2011.

 

 
 [37] 
 
V. K. Kurmi, V. K. Subramanian, and V. P. Namboodiri, “Domain impression: A
source data free domain adaptation method,” in Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision , 2021, pp.
615–625.

 

 
 [38] 
 
Y. Hou and L. Zheng, “Visualizing adapted knowledge in domain transfer,” in
 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , 2021, pp. 13 824–13 833.

 

 
 [39] 
 
R. Li, Q. Jiao, W. Cao, H.-S. Wong, and S. Wu, “Model adaptation:
Unsupervised domain adaptation without source data,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2020,
pp. 9641–9650.

 

 
 [40] 
 
Z. Qiu, Y. Zhang, H. Lin, S. Niu, Y. Liu, Q. Du, and M. Tan, “Source-free
domain adaptation via avatar prototype generation and adaptation,”
 arXiv preprint arXiv:2106.15326 , 2021.

 

 
 [41] 
 
J. Tian, J. Zhang, W. Li, and D. Xu, “VDM-DA: Virtual domain modeling for
source data-free domain adaptation,” IEEE Transactions on Circuits and
Systems for Video Technology , 2021.

 

 
 [42] 
 
N. Ding, Y. Xu, Y. Tang, C. Xu, Y. Wang, and D. Tao, “Source-free domain
adaptation via distribution estimation,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp.
7212–7222.

 

 
 [43] 
 
S. Stan and M. Rostami, “Unsupervised model adaptation for continual semantic
segmentation,” in Proceedings of the AAAI Conference on Artificial
Intelligence , vol. 35, no. 3, 2021, pp. 2593–2601.

 

 
 [44] 
 
Stan, Serban and Rostami, Mohammad, “Privacy preserving domain adaptation
for semantic segmentation of medical images,” arXiv preprint
arXiv:2101.00522 , 2021.

 

 
 [45] 
 
W.-Y. Chen, Y.-C. Liu, Z. Kira, Y.-C. F. Wang, and J.-B. Huang, “A closer look
at few-shot classification,” arXiv preprint arXiv:1904.04232 , 2019.

 

 
 [46] 
 
G. Kang, L. Jiang, Y. Yang, and A. G. Hauptmann, “Contrastive adaptation
network for unsupervised domain adaptation,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2019, pp.
4893–4902.

 

 
 [47] 
 
C.-Y. Lee, T. Batra, M. H. Baig, and D. Ulbricht, “Sliced wasserstein
discrepancy for unsupervised domain adaptation,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2019, pp.
10 285–10 295.

 

 
 [48] 
 
D. Bang and H. Shim, “MGGAN: Solving mode collapse using manifold-guided
training,” in Proceedings of the IEEE/CVF International Conference on
Computer Vision , 2021, pp. 2347–2356.

 

 
 [49] 
 
A. Abusitta, O. A. Wahab, and B. C. Fung, “VirtualGAN: Reducing mode
collapse in generative adversarial networks using virtual mapping,” in
 2021 International Joint Conference on Neural Networks (IJCNN) . IEEE, 2021, pp. 1–6.

 

 
 [50] 
 
X. Liu and Y. Yuan, “A source-free domain adaptive polyp detection framework
with style diversification flow,” IEEE Transactions on Medical
Imaging , vol. 41, no. 7, pp. 1897–1908, 2022.

 

 
 [51] 
 
G. Yang, H. Tang, Z. Zhong, M. Ding, L. Shao, N. Sebe, and E. Ricci,
“Transformer-based source-free domain adaptation,” arXiv preprint
arXiv:2105.14138 , 2021.

 

 
 [52] 
 
L. Xiong, M. Ye, D. Zhang, Y. Gan, X. Li, and Y. Zhu, “Source data-free domain
adaptation of object detector through domain-specific perturbation,”
 International Journal of Intelligent Systems , vol. 36, no. 8, pp.
3746–3766, 2021.

 

 
 [53] 
 
X. Liu and S. Zhang, “Graph consistency based mean-teaching for unsupervised
domain adaptive person re-identification,” arXiv preprint
arXiv:2105.04776 , 2021.

 

 
 [54] 
 
H. Yu, J. Huang, Y. Liu, Q. Zhu, M. Zhou, and F. Zhao, “Source-free domain
adaptation for real-world image dehazing,” in Proceedings of the 30th
ACM International Conference on Multimedia , 2022, pp. 6645–6654.

 

 
 [55] 
 
S. Tang, Y. Shi, Z. Ma, J. Li, J. Lyu, Q. Li, and J. Zhang, “Model adaptation
through hypothesis transfer with gradual knowledge distillation,” in
 2021 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS) . IEEE, 2021, pp.
5679–5685.

 

 
 [56] 
 
V. VS, J. M. J. Valanarasu, and V. M. Patel, “Target and task specific
source-free domain adaptive image segmentation,” arXiv preprint
arXiv:2203.15792 , 2022.

 

 
 [57] 
 
A. Tarvainen and H. Valpola, “Mean teachers are better role models:
Weight-averaged consistency targets improve semi-supervised deep learning
results,” Advances in Neural Information Processing Systems , vol. 30,
2017.

 

 
 [58] 
 
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al. ,
“An image is worth 16 × \times 16 words: Transformers for image recognition
at scale,” arXiv preprint arXiv:2010.11929 , 2020.

 

 
 [59] 
 
M. Ishii and M. Sugiyama, “Source-free domain adaptation via distributional
alignment by matching batch normalization statistics,” arXiv preprint
arXiv:2101.10842 , 2021.

 

 
 [60] 
 
X. Liu, F. Xing, C. Yang, G. El Fakhri, and J. Woo, “Adapting off-the-shelf
source segmenter for target medical image segmentation,” in
 International Conference on Medical Image Computing and
Computer-Assisted Intervention . Springer, 2021, pp. 549–559.

 

 
 [61] 
 
S. Paul, A. Khurana, and G. Aggarwal, “Unsupervised adaptation of semantic
segmentation models without source data,” arXiv preprint
arXiv:2112.02359 , 2021.

 

 
 [62] 
 
J. Fan, H. Zhu, X. Jiang, L. Meng, C. Chen, C. Fu, H. Yu, C. Dai, and W. Chen,
“Unsupervised domain adaptation by statistics alignment for deep sleep
staging networks,” IEEE Transactions on Neural Systems and
Rehabilitation Engineering , vol. 30, pp. 205–216, 2022.

 

 
 [63] 
 
C. Eastwood, I. Mason, C. K. Williams, and B. Schölkopf, “Source-free
adaptation to measurement shift via bottom-up feature restoration,”
 arXiv preprint arXiv:2107.05446 , 2021.

 

 
 [64] 
 
D. Zhang, M. Ye, L. Xiong, S. Li, and X. Li, “Source-style transferred mean
teacher for source-data free object detection,” in ACM Multimedia
Asia , 2021, pp. 1–8.

 

 
 [65] 
 
M. Klingner, J.-A. Termöhlen, J. Ritterbach, and T. Fingscheidt,
“Unsupervised batchnorm adaptation (UBNA): A domain adaptation method
for semantic segmentation without using source domain representations,” in
 Proceedings of the IEEE/CVF Winter Conference on Applications of
Computer Vision , 2022, pp. 210–220.

 

 
 [66] 
 
W.-G. Chang, T. You, S. Seo, S. Kwak, and B. Han, “Domain-specific batch
normalization for unsupervised domain adaptation,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2019,
pp. 7354–7362.

 

 
 [67] 
 
Y. Li, N. Wang, J. Shi, J. Liu, and X. Hou, “Revisiting batch normalization
for practical domain adaptation,” arXiv preprint arXiv:1603.04779 ,
2016.

 

 
 [68] 
 
S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network
training by reducing internal covariate shift,” in International
Conference on Machine Learning . PMLR,
2015, pp. 448–456.

 

 
 [69] 
 
D. Ulyanov, A. Vedaldi, and V. Lempitsky, “Instance normalization: The
missing ingredient for fast stylization,” arXiv preprint
arXiv:1607.08022 , 2016.

 

 
 [70] 
 
H. Xia, H. Zhao, and Z. Ding, “Adaptive adversarial network for source-free
domain adaptation,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision , 2021, pp. 9010–9019.

 

 
 [71] 
 
R. Wang, Z. Wu, Z. Weng, J. Chen, G.-J. Qi, and Y.-G. Jiang, “Cross-domain
contrastive learning for unsupervised domain adaptation,” IEEE
Transactions on Multimedia , 2022.

 

 
 [72] 
 
P. Agarwal, D. P. Paudel, J.-N. Zaech, and L. Van Gool, “Unsupervised robust
domain adaptation without source data,” in Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision , 2022, pp. 2009–2018.

 

 
 [73] 
 
X. Zhao, R. Stanislawski, P. Gardoni, M. Sulowicz, A. Glowacz, G. Krolczyk, and
Z. Li, “Adaptive contrastive learning with label consistency for source data
free unsupervised domain adaptation,” Sensors , vol. 22, no. 11, p.
4238, 2022.

 

 
 [74] 
 
K. Sohn, “Improved deep metric learning with multi-class N-pair loss
objective,” Advances in Neural Information Processing Systems ,
vol. 29, 2016.

 

 
 [75] 
 
J. Gawlikowski, C. R. N. Tassi, M. Ali, J. Lee, M. Humt, J. Feng, A. Kruspe,
R. Triebel, P. Jung, R. Roscher et al. , “A survey of uncertainty in
deep neural networks,” arXiv preprint arXiv:2107.03342 , 2021.

 

 
 [76] 
 
F. Fleuret et al. , “Uncertainty reduction for model adaptation in
semantic segmentation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2021, pp. 9613–9623.

 

 
 [77] 
 
J. Lee and G. Lee, “Feature alignment by uncertainty and self-training for
source-free unsupervised domain adaptation,” arXiv preprint
arXiv:2208.14888 , 2022.

 

 
 [78] 
 
C. Chen, Q. Liu, Y. Jin, Q. Dou, and P.-A. Heng, “Source-free domain adaptive
fundus image segmentation with denoised pseudo-labeling,” in
 International Conference on Medical Image Computing and
Computer-Assisted Intervention . Springer, 2021, pp. 225–235.

 

 
 [79] 
 
Z. Xu, D. Lu, Y. Wang, J. Luo, D. Wei, Y. Zheng, and R. K.-y. Tong, “Denoising
for relaxing: Unsupervised domain adaptive fundus image segmentation
without source data,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention . Springer, 2022, pp. 214–224.

 

 
 [80] 
 
D. Hegde, V. Sindagi, V. Kilic, A. B. Cooper, M. Foster, and V. Patel,
“Uncertainty-aware mean teacher for source-free unsupervised domain adaptive
3D object detection,” arXiv preprint arXiv:2109.14651 , 2021.

 

 
 [81] 
 
S. Roy, M. Trapp, A. Pilzer, J. Kannala, N. Sebe, E. Ricci, and A. Solin,
“Uncertainty-guided source-free domain adaptation,” in European
Conference on Computer Vision . Springer, 2022, pp. 537–555.

 

 
 [82] 
 
J. Pei, Z. Jiang, A. Men, L. Chen, Y. Liu, and Q. Chen, “Uncertainty-induced
transferability representation for source-free unsupervised domain
adaptation,” arXiv preprint arXiv:2208.13986 , 2022.

 

 
 [83] 
 
X. Li, W. Chen, D. Xie, S. Yang, P. Yuan, S. Pu, and Y. Zhuang, “A free lunch
for unsupervised domain adaptive object detection without source data,” in
 Proceedings of the AAAI Conference on Artificial Intelligence ,
vol. 35, no. 10, 2021, pp. 8474–8481.

 

 
 [84] 
 
N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov,
“Dropout: A simple way to prevent neural networks from overfitting,”
 The Journal of Machine Learning Research , vol. 15, no. 1, pp.
1929–1958, 2014.

 

 
 [85] 
 
Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing
model uncertainty in deep learning,” in International Conference on
Machine Learning . PMLR, 2016, pp.
1050–1059.

 

 
 [86] 
 
A. Kendall and Y. Gal, “What uncertainties do we need in bayesian deep
learning for computer vision?” Advances in Neural Information
Processing Systems , vol. 30, 2017.

 

 
 [87] 
 
Z. Xu, D. Lu, Y. Wang, J. Luo, J. Jayender, K. Ma, Y. Zheng, and X. Li, “Noisy
labels are treasure: Mean-teacher-assisted confident learning for hepatic
vessel segmentation,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention . Springer, 2021, pp. 3–13.

 

 
 [88] 
 
C. Northcutt, L. Jiang, and I. Chuang, “Confident learning: Estimating
uncertainty in dataset labels,” Journal of Artificial Intelligence
Research , vol. 70, pp. 1373–1411, 2021.

 

 
 [89] 
 
Y. Gal, J. Hron, and A. Kendall, “Concrete dropout,” Advances in Neural
Information Processing Systems , vol. 30, 2017.

 

 
 [90] 
 
L. Tierney and J. B. Kadane, “Accurate approximations for posterior moments
and marginal densities,” Journal of the American Statistical
Association , vol. 81, no. 393, pp. 82–86, 1986.

 

 
 [91] 
 
D. J. MacKay, D. J. Mac Kay et al. , Information theory, inference
and learning algorithms . Cambridge
University Press, 2003.

 

 
 [92] 
 
A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “YOLOv4: Optimal speed and
accuracy of object detection,” arXiv preprint arXiv:2004.10934 , 2020.

 

 
 [93] 
 
S. Yang, Y. Wang, J. van de Weijer, L. Herranz, and S. Jui, “Generalized
source-free domain adaptation,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision , 2021, pp. 8978–8987.

 

 
 [94] 
 
S. Tang, Y. Yang, Z. Ma, N. Hendrich, F. Zeng, S. S. Ge, C. Zhang, and
J. Zhang, “Nearest neighborhood-based deep clustering for source data-absent
unsupervised domain adaptation,” arXiv preprint arXiv:2107.12585 ,
2021.

 

 
 [95] 
 
J. Liang, D. Hu, and J. Feng, “Do we really need to access the source data?
Source hypothesis transfer for unsupervised domain adaptation,” in
 International Conference on Machine Learning . PMLR, 2020, pp. 6028–6039.

 

 
 [96] 
 
J. Liang, D. Hu, Y. Wang, R. He, and J. Feng, “Source data-absent unsupervised
domain adaptation through hypothesis transfer and labeling transfer,”
 IEEE Transactions on Pattern Analysis and Machine Intelligence , 2021.

 

 
 [97] 
 
J. Lee, D. Jung, J. Yim, and S. Yoon, “Confidence score for source-free
unsupervised domain adaptation,” in International Conference on
Machine Learning . PMLR, 2022, pp.
12 365–12 377.

 

 
 [98] 
 
W. Li, M. Cao, and S. Chen, “Jacobian norm for unsupervised source-free domain
adaptation,” arXiv preprint arXiv:2204.03467 , 2022.

 

 
 [99] 
 
S. Yang, Y. Wang, K. Wang, S. Jui, and J. van de Weijer, “Attracting and
dispersing: A simple approach for source-free domain adaptation.”
 CoRR , 2022.

 

 
 [100] 
 
F. G. Ashby, W. T. Maddox et al. , “Human category learning,”
 Annual Review of Psychology , vol. 56, no. 1, pp. 149–178, 2005.

 

 
 [101] 
 
J. Li, Z. Du, L. Zhu, Z. Ding, K. Lu, and H. T. Shen, “Divergence-agnostic
unsupervised domain adaptation by adversarial attacks,” IEEE
Transactions on Pattern Analysis and Machine Intelligence , 2021.

 

 
 [102] 
 
F. Wang, Z. Han, Y. Gong, and Y. Yin, “Exploring domain-invariant parameters
for source free domain adaptation,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , 2022, pp. 7151–7160.

 

 
 [103] 
 
J. Liang, R. He, Z. Sun, and T. Tan, “Distant supervised centroid shift: A
simple and efficient approach to visual domain adaptation,” in
 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , 2019, pp. 2975–2984.

 

 
 [104] 
 
S. Yang, Y. Wang, J. van de Weijer, L. Herranz, and S. Jui, “Casting a BAIT
for offline and online source-free domain adaptation,” arXiv preprint
arXiv:2010.12427 , 2020.

 

 
 [105] 
 
Z. Ke, D. Wang, Q. Yan, J. Ren, and R. W. Lau, “Dual student: Breaking the
limits of the teacher in semi-supervised learning,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision , 2019, pp.
6728–6736.

 

 
 [106] 
 
T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for
contrastive learning of visual representations,” in International
Conference on Machine Learning . PMLR,
2020, pp. 1597–1607.

 

 
 [107] 
 
K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for
unsupervised visual representation learning,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2020, pp.
9729–9738.

 

 
 [108] 
 
T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for
contrastive learning of visual representations,” in International
Conference on Machine Learning . PMLR,
2020, pp. 1597–1607.

 

 
 [109] 
 
W. Chen, S. Pu, D. Xie, S. Yang, Y. Guo, and L. Lin, “Unsupervised image
classification for deep representation learning,” in European
Conference on Computer Vision . Springer, 2020, pp. 430–446.

 

 
 [110] 
 
J. Liang, D. Hu, J. Feng, and R. He, “DINE: Domain adaptation from single
and multiple black-box predictors,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , 2022, pp. 8003–8013.

 

 
 [111] 
 
J. Liang, D. Hu, R. He, and J. Feng, “Distill and fine-tune: Effective
adaptation from a black-box source model,” arXiv preprint
arXiv:2104.01539 , 2021.

 

 
 [112] 
 
X. Liu, C. Yoo, F. Xing, C.-C. J. Kuo, G. El Fakhri, J.-W. Kang, and J. Woo,
“Unsupervised black-box model domain adaptation for brain tumor
segmentation,” Frontiers in Neuroscience , p. 341, 2022.

 

 
 [113] 
 
Liu, Xiaofeng and Yoo, Chaehwa and Xing, Fangxu and Kuo, C-C Jay and El
Fakhri, Georges and Kang, Je-Won and Woo, Jonghye, “Unsupervised domain
adaptation for segmentation with black-box source model,” in Medical
Imaging 2022: Image Processing , vol. 12032. SPIE, 2022, pp. 255–260.

 

 
 [114] 
 
Y. Xu, J. Yang, M. Wu, X. Li, L. Xie, and Z. Chen, “EXTERN: Leveraging
endo-temporal regularization for black-box video domain adaptation,”
 arXiv preprint arXiv:2208.05187 , 2022.

 

 
 [115] 
 
Q. Peng, Z. Ding, L. Lyu, L. Sun, and C. Chen, “Toward better target
representation for source-free and black-box domain adaptation,” arXiv
preprint arXiv:2208.10531 , 2022.

 

 
 [116] 
 
S. Laine and T. Aila, “Temporal ensembling for semi-supervised learning,”
 arXiv preprint arXiv:1610.02242 , 2016.

 

 
 [117] 
 
K. Kim, B. Ji, D. Yoon, and S. Hwang, “Self-knowledge distillation with
progressive refinement of targets,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision , 2021, pp. 6567–6576.

 

 
 [118] 
 
T. Yang, S. Zhu, C. Chen, S. Yan, M. Zhang, and A. Willis, “MutualNet:
Adaptive convnet via mutual learning from network width and resolution,”
in European Conference on Computer Vision . Springer, 2020, pp. 299–315.

 

 
 [119] 
 
H. Zhang, Y. Zhang, K. Jia, and L. Zhang, “Unsupervised domain adaptation of
black-box source models,” arXiv preprint arXiv:2101.02839 , 2021.

 

 
 [120] 
 
X. Luo, W. Chen, Y. Tan, C. Li, Y. He, and X. Jia, “Exploiting negative
learning for implicit pseudo label rectification in source-free domain
adaptive semantic segmentation,” arXiv preprint arXiv:2106.12123 ,
2021.

 

 
 [121] 
 
D. Arpit, S. Jastrzebski, N. Ballas, D. Krueger, E. Bengio, M. S. Kanwal,
T. Maharaj, A. Fischer, A. Courville, Y. Bengio et al. , “A closer
look at memorization in deep networks,” in International Conference on
Machine Learning . PMLR, 2017, pp.
233–242.

 

 
 [122] 
 
J. Yang, X. Peng, K. Wang, Z. Zhu, J. Feng, L. Xie, and Y. You, “Divide to
adapt: Mitigating confirmation bias for domain adaptation of black-box
predictors,” arXiv preprint arXiv:2205.14467 , 2022.

 

 
 [123] 
 
E. Arazo, D. Ortego, P. Albert, N. O’Connor, and K. McGuinness,
“Unsupervised label noise modeling and loss correction,” in
 International Conference on Machine Learning . PMLR, 2019, pp. 312–321.

 

 
 [124] 
 
D. Berthelot, N. Carlini, I. Goodfellow, N. Papernot, A. Oliver, and C. A.
Raffel, “MixMatch: A holistic approach to semi-supervised learning,”
 Advances in Neural Information Processing Systems , vol. 32, 2019.

 

 
 [125] 
 
H.-W. Yeh, B. Yang, P. C. Yuen, and T. Harada, “SoFA: Source-data-free
feature alignment for unsupervised domain adaptation,” in Proceedings
of the IEEE/CVF Winter Conference on Applications of Computer Vision , 2021,
pp. 474–483.

 

 
 [126] 
 
B. Yang, H.-W. Yeh, T. Harada, and P. C. Yuen, “Model-induced generalization
error bound for information-theoretic representation learning in
source-data-free unsupervised domain adaptation,” IEEE Transactions on
Image Processing , vol. 31, pp. 419–432, 2021.

 

 
 [127] 
 
W. Zhang and D. Wu, “Lightweight source-free transfer for privacy-preserving
motor imagery classification,” IEEE Transactions on Cognitive and
Developmental Systems , 2022.

 

 
 [128] 
 
Zhang, Wen and Wu, Dongrui, “Discriminative joint probability maximum mean
discrepancy (DJP-MMD) for domain adaptation,” in 2020 International
Joint Conference on Neural Networks (IJCNN) . IEEE, 2020, pp. 1–8.

 

 
 [129] 
 
X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang, “Moment matching
for multi-source domain adaptation,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision , 2019, pp. 1406–1415.

 

 
 [130] 
 
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning
applied to document recognition,” Proceedings of the IEEE , vol. 86,
no. 11, pp. 2278–2324, 1998.

 

 
 [131] 
 
Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading
digits in natural images with unsupervised feature learning,” 2011.

 

 
 [132] 
 
J. J. Hull, “A database for handwritten text recognition research,”
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
vol. 16, no. 5, pp. 550–554, 1994.

 

 
 [133] 
 
S. R. Richter, V. Vineet, S. Roth, and V. Koltun, “Playing for data: Ground
truth from computer games,” in European Conference on Computer
Vision . Springer, 2016, pp. 102–118.

 

 
 [134] 
 
M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic
urban scene understanding,” in Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition , 2016, pp. 3213–3223.

 

 
 [135] 
 
G. Ros, L. Sellart, J. Materzynska, D. Vazquez, and A. M. Lopez, “The synthia
dataset: A large collection of synthetic images for semantic segmentation
of urban scenes,” in Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition , 2016, pp. 3234–3243.

 

 
 [136] 
 
Y.-H. Chen, W.-Y. Chen, Y.-T. Chen, B.-C. Tsai, Y.-C. Frank Wang, and M. Sun,
“No more discrimination: Cross city adaptation of road scene segmenters,”
in Proceedings of the IEEE International Conference on Computer
Vision , 2017, pp. 1992–2001.

 

 
 [137] 
 
K. Saenko, B. Kulis, M. Fritz, and T. Darrell, “Adapting visual category
models to new domains,” in European Conference on Computer
Vision . Springer, 2010, pp. 213–226.

 

 
 [138] 
 
H. Venkateswara, J. Eusebio, S. Chakraborty, and S. Panchanathan, “Deep
hashing network for unsupervised domain adaptation,” in Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition , 2017, pp.
5018–5027.

 

 
 [139] 
 
X. Peng, B. Usman, N. Kaushik, J. Hoffman, D. Wang, and K. Saenko, “VisDA:
The visual domain adaptation challenge,” arXiv preprint
arXiv:1710.06924 , 2017.

 

 
 [140] 
 
B. Gong, Y. Shi, F. Sha, and K. Grauman, “Geodesic flow kernel for
unsupervised domain adaptation,” in 2012 IEEE Conference on Computer
Vision and Pattern Recognition . IEEE,
2012, pp. 2066–2073.

 

 
 [141] 
 
B. Caputo, H. Müller, J. Martinez-Gomez, M. Villegas, B. Acar, N. Patricia,
N. Marvasti, S. Üsküdarlı, R. Paredes, M. Cazorla et al. ,
“ImageCLEF 2014: Overview and analysis of the results,” in
 International Conference of the Cross-Language Evaluation Forum for
European Languages . Springer, 2014,
pp. 192–211.

 

 
 [142] 
 
G. Griffin, A. Holub, and P. Perona, “Caltech-256 object category dataset,”
2007.

 

 
 [143] 
 
M. Everingham and J. Winn, “The PASCAL visual object classes challenge 2007
(VOC2007) development kit,” International Journal of Computer
Vision , vol. 88, no. 2, pp. 303–338, 2010.

 

 
 [144] 
 
A. Bergamo and L. Torresani, “Exploiting weakly-labeled web images to improve
object classification: A domain adaptation approach,” Advances in
Neural Information Processing Systems , vol. 23, 2010.

 

 
 [145] 
 
K. Zhou, Y. Yang, Y. Qiao, and T. Xiang, “Domain adaptive ensemble learning,”
 IEEE Transactions on Image Processing , vol. 30, pp. 8008–8018, 2021.

 

 
 [146] 
 
C. Qin, H. You, L. Wang, C.-C. J. Kuo, and Y. Fu, “PointDAN: A multi-scale
3D domain adaption network for point cloud representation,” Advances
in Neural Information Processing Systems , vol. 32, 2019.

 

 
 [147] 
 
Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao, “3D
ShapeNets: A deep representation for volumetric shapes,” in
 Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition , 2015, pp. 1912–1920.

 

 
 [148] 
 
A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li,
S. Savarese, M. Savva, S. Song, H. Su et al. , “ShapeNet: An
information-rich 3D model repository,” arXiv preprint
arXiv:1512.03012 , 2015.

 

 
 [149] 
 
A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner,
“ScanNet: Richly-annotated 3D reconstructions of indoor scenes,” in
 Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition , 2017, pp. 5828–5839.

 

 
 [150] 
 
I. Chingovska, A. Anjos, and S. Marcel, “On the effectiveness of local binary
patterns in face anti-spoofing,” in 2012 BIOSIG-Proceedings of the
International Conference of Biometrics Special Interest Group
(BIOSIG) . IEEE, 2012, pp. 1–7.

 

 
 [151] 
 
Z. Boulkenafet, J. Komulainen, L. Li, X. Feng, and A. Hadid, “OULU-NPU: A
mobile face presentation attack database with real-world variations,” in
 2017 12th IEEE International Conference on Automatic Face Gesture
Recognition (FG 2017) . IEEE, 2017,
pp. 612–618.

 

 
 [152] 
 
Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, and S. Z. Li, “A face antispoofing
database with diverse attacks,” in 2012 5th IAPR International
Conference on Biometrics (ICB) . IEEE,
2012, pp. 26–31.

 

 
 [153] 
 
D. Wen, H. Han, and A. K. Jain, “Face spoof detection with image distortion
analysis,” IEEE Transactions on Information Forensics and Security ,
vol. 10, no. 4, pp. 746–761, 2015.

 

 
 [154] 
 
P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo,
Y. Zhou, Y. Chai, B. Caine et al. , “Scalability in perception for
autonomous driving: Waymo open dataset,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2020, pp.
2446–2454.

 

 
 [155] 
 
A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving?
The KITTI vision benchmark suite,” in 2012 IEEE Conference on
Computer Vision and Pattern Recognition . IEEE, 2012, pp. 3354–3361.

 

 
 [156] 
 
H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan,
Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for
autonomous driving,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2020, pp. 11 621–11 631.

 

 
 [157] 
 
M.-H. Chen, Z. Kira, G. AlRegib, J. Yoo, R. Chen, and J. Zheng, “Temporal
attentive alignment for large-scale video domain adaptation,” in
 Proceedings of the IEEE/CVF International Conference on Computer
Vision , 2019, pp. 6321–6330.

 

 
 [158] 
 
K. Soomro, A. R. Zamir, and M. Shah, “UCF101: A dataset of 101 human
actions classes from videos in the wild,” arXiv preprint
arXiv:1212.0402 , 2012.

 

 
 [159] 
 
H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, “HMDB: A large
video database for human motion recognition,” in 2011 International
Conference on Computer Vision . IEEE,
2011, pp. 2556–2563.

 

 
 [160] 
 
Y. Xu, J. Yang, H. Cao, K. Wu, M. Wu, R. Zhao, and Z. Chen, “Multi-source
video domain adaptation with temporal attentive moment alignment,”
 arXiv preprint arXiv:2109.09964 , 2021.

 

 
 [161] 
 
A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei,
“Large-scale video classification with convolutional neural networks,” in
 Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition , 2014, pp. 1725–1732.

 

 
 [162] 
 
W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan,
F. Viola, T. Green, T. Back, P. Natsev et al. , “The kinetics human
action video dataset,” arXiv preprint arXiv:1705.06950 , 2017.

 

 
 [163] 
 
Y. Xu, J. Yang, H. Cao, K. Mao, J. Yin, and S. See, “ARID: A new dataset
for recognizing action in the dark,” in International Workshop on Deep
Learning for Human Activity Recognition . Springer, 2021, pp. 70–84.

 

 
 [164] 
 
M. Monfort, A. Andonian, B. Zhou, K. Ramakrishnan, S. A. Bargal, T. Yan,
L. Brown, Q. Fan, D. Gutfreund, C. Vondrick et al. , “Moments in time
dataset: One million videos for event understanding,” IEEE
Transactions on Pattern Analysis and Machine Intelligence , vol. 42, no. 2,
pp. 502–508, 2019.

 

 
 [165] 
 
B. Moiseev, A. Konev, A. Chigorin, and A. Konushin, “Evaluation of traffic
sign recognition methods trained on synthetically generated data,” in
 International Conference on Advanced Concepts for Intelligent Vision
Systems . Springer, 2013, pp.
576–583.

 

 
 [166] 
 
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, “The german traffic sign
recognition benchmark: A multi-class classification competition,” in
 The 2011 International Joint Conference on Neural Networks . IEEE, 2011, pp. 1453–1460.

 

 
 [167] 
 
C. Fang, Y. Xu, and D. N. Rockmore, “Unbiased metric learning: On the
utilization of multiple datasets and web images for softening bias,” in
 Proceedings of the IEEE International Conference on Computer Vision ,
2013, pp. 1657–1664.

 

 
 [168] 
 
L. Fei-Fei, R. Fergus, and P. Perona, “One-shot learning of object
categories,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , vol. 28, no. 4, pp. 594–611, 2006.

 

 
 [169] 
 
B. C. Russell, A. Torralba, K. P. Murphy, and W. T. Freeman, “LabelMe: A
database and web-based tool for image annotation,” International
Journal of Computer Vision , vol. 77, no. 1, pp. 157–173, 2008.

 

 
 [170] 
 
M. J. Choi, J. J. Lim, A. Torralba, and A. S. Willsky, “Exploiting
hierarchical context on a large database of object categories,” in
 2010 IEEE Computer Society Conference on Computer Vision and Pattern
Recognition . IEEE, 2010, pp.
129–136.

 

 
 [171] 
 
B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby,
Y. Burren, N. Porz, J. Slotboom, R. Wiest et al. , “The multimodal
brain tumor image segmentation benchmark (BRATS),” IEEE Transactions
on Medical Imaging , vol. 34, no. 10, pp. 1993–2024, 2014.

 

 
 [172] 
 
X. Zhuang and J. Shen, “Multi-scale patch and multi-modality atlases for whole
heart segmentation of MRI,” Medical Image Analysis , vol. 31, pp.
77–87, 2016.

 

 
 [173] 
 
Y. Li, R. Dan, S. Wang, Y. Cao, X. Luo, C. Tan, G. Jia, H. Zhou, Y. Wang, and
L. Wang, “Source-free domain adaptation for multi-site and lifespan brain
skull stripping,” arXiv preprint arXiv:2203.04299 , 2022.

 

 
 [174] 
 
S. F. Eskildsen, P. Coupé, V. Fonov, J. V. Manjón, K. K. Leung,
N. Guizard, S. N. Wassef, L. R. Østergaard, D. L. Collins, A. D. N.
Initiative et al. , “BEaST: Brain extraction based on nonlocal
segmentation technique,” NeuroImage , vol. 59, no. 3, pp. 2362–2373,
2012.

 

 
 [175] 
 
C. R. Jack Jr, M. A. Bernstein, N. C. Fox, P. Thompson, G. Alexander,
D. Harvey, B. Borowski, P. J. Britson, J. L. Whitwell, C. Ward et al. ,
“The Alzheimer’s disease neuroimaging initiative (ADNI): MRI
methods,” Journal of Magnetic Resonance Imaging: An Official Journal
of the International Society for Magnetic Resonance in Medicine , vol. 27,
no. 4, pp. 685–691, 2008.

 

 
 [176] 
 
A. Makropoulos, E. C. Robinson, A. Schuh, R. Wright, S. Fitzgibbon, J. Bozek,
S. J. Counsell, J. Steinweg, K. Vecchiato, J. Passerat-Palmbach
 et al. , “The developing human connectome project: A minimal
processing pipeline for neonatal cortical surface reconstruction,”
 NeuroImage , vol. 173, pp. 88–112, 2018.

 

 
 [177] 
 
J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil,
C. Rodríguez, and F. Vilariño, “WM-DOVA maps for accurate polyp
highlighting in colonoscopy: Validation vs. saliency maps from
physicians,” Computerized Medical Imaging and Graphics , vol. 43, pp.
99–111, 2015.

 

 
 [178] 
 
T.-H. Hoang, H.-D. Nguyen, V.-A. Nguyen, T.-A. Nguyen, V.-T. Nguyen, and M.-T.
Tran, “Enhancing endoscopic image classification with symptom localization
and data augmentation,” in Proceedings of the 27th ACM International
Conference on Multimedia , 2019, pp. 2578–2582.

 

 
 [179] 
 
J. Silva, A. Histace, O. Romain, X. Dray, and B. Granado, “Toward embedded
detection of polyps in WCE images for early diagnosis of colorectal
cancer,” International Journal of Computer Assisted Radiology and
Surgery , vol. 9, no. 2, pp. 283–293, 2014.

 

 
 [180] 
 
D. Vázquez, J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, A. M.
López, A. Romero, M. Drozdzal, and A. Courville, “A benchmark for
endoluminal scene segmentation of colonoscopy images,” Journal of
Healthcare Engineering , vol. 2017, 2017.

 

 
 [181] 
 
M. Tangermann, K.-R. Müller, A. Aertsen, N. Birbaumer, C. Braun,
C. Brunner, R. Leeb, C. Mehring, K. J. Miller, G. Mueller-Putz et al. ,
“Review of the BCI competition IV,” Frontiers in Neuroscience ,
p. 55, 2012.

 

 
 [182] 
 
J. Faller, C. Vidaurre, T. Solis-Escalante, C. Neuper, and R. Scherer,
“Autocalibration and recurrent adaptation: Towards a plug and play online
ERD-BCI,” IEEE Transactions on Neural Systems and Rehabilitation
Engineering , vol. 20, no. 3, pp. 313–319, 2012.

 

 
 [183] 
 
V. Jayaram and A. Barachant, “MOABB: Trustworthy algorithm benchmarking
for BCIs,” Journal of Neural Engineering , vol. 15, no. 6, p.
066011, 2018.

 

 
 [184] 
 
N. Bloch, A. Madabhushi, H. Huisman, J. Freymann, J. Kirby, M. Grauer,
A. Enquobahrie, C. Jaffe, L. Clarke, and K. Farahani, “NCI-ISBI 2013
challenge: Automated segmentation of prostate structures,” The
Cancer Imaging Archive , vol. 370, no. 6, p. 5, 2015.

 

 
 [185] 
 
G. Litjens, R. Toth, W. van de Ven, C. Hoeks, S. Kerkstra, B. van Ginneken,
G. Vincent, G. Guillard, N. Birbeck, J. Zhang et al. , “Evaluation of
prostate segmentation algorithms for MRI: The PROMISE12 challenge,”
 Medical Image Analysis , vol. 18, no. 2, pp. 359–373, 2014.

 

 
 [186] 
 
J. I. Orlando, H. Fu, J. B. Breda, K. van Keer, D. R. Bathula, A. Diaz-Pinto,
R. Fang, P.-A. Heng, J. Kim, J. Lee et al. , “Refuge challenge: A
unified framework for evaluating automated methods for glaucoma assessment
from fundus photographs,” Medical Image Analysis , vol. 59, p. 101570,
2020.

 

 
 [187] 
 
F. Fumero, S. Alayón, J. L. Sanchez, J. Sigut, and M. Gonzalez-Hernandez,
“RIM-ONE: An open retinal image database for optic nerve evaluation,”
in 2011 24th International Symposium on Computer-based Medical Systems
(CBMS) . IEEE, 2011, pp. 1–6.

 

 
 [188] 
 
J. Sivaswamy, S. Krishnadas, A. Chakravarty, G. Joshi, A. S. Tabish
 et al. , “A comprehensive retinal image dataset for the assessment of
glaucoma from the optic nerve head analysis,” JSM Biomedical Imaging
Data Papers , vol. 2, no. 1, p. 1004, 2015.

 

 
 [189] 
 
A. Di Martino, C.-G. Yan, Q. Li, E. Denio, F. X. Castellanos, K. Alaerts, J. S.
Anderson, M. Assaf, S. Y. Bookheimer, M. Dapretto et al. , “The autism
brain imaging data exchange: Towards a large-scale evaluation of the
intrinsic brain architecture in autism,” Molecular Psychiatry ,
vol. 19, no. 6, pp. 659–667, 2014.

 

 
 [190] 
 
H. Yin, P. Molchanov, J. M. Alvarez, Z. Li, A. Mallya, D. Hoiem, N. K. Jha, and
J. Kautz, “Dreaming to distill: Data-free knowledge transfer via
DeepInversion,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2020, pp. 8715–8724.

 

 
 [191] 
 
M. Nasr, R. Shokri, and A. Houmansadr, “Comprehensive privacy analysis of deep
learning: Passive and active white-box inference attacks against
centralized and federated learning,” in 2019 IEEE Symposium on
Security and Privacy (SP) . IEEE,
2019, pp. 739–753.

 

 
 [192] 
 
H. Hu, Z. Salcic, L. Sun, G. Dobbie, P. S. Yu, and X. Zhang, “Membership
inference attacks on machine learning: A survey,” ACM Computing
Surveys (CSUR) , vol. 54, no. 11s, pp. 1–37, 2022.

 

 
 [193] 
 
S. M. Ahmed, D. S. Raychaudhuri, S. Paul, S. Oymak, and A. K. Roy-Chowdhury,
“Unsupervised multi-source domain adaptation without access to source
data,” in Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2021, pp. 10 103–10 112.

 

 
 [194] 
 
Y. Grandvalet and Y. Bengio, “Semi-supervised learning by entropy
minimization,” Advances in Neural Information Processing Systems ,
vol. 17, 2004.

 

 
 [195] 
 
M. Bateson, H. Kervadec, J. Dolz, H. Lombaert, and I. B. Ayed, “Source-free
domain adaptation for image segmentation,” Medical Image Analysis ,
vol. 82, p. 102617, 2022.

 

 
 [196] 
 
J. Liu, X. Li, S. An, and Z. Chen, “Source-free unsupervised domain adaptation
for blind image quality assessment,” arXiv preprint arXiv:2207.08124 ,
2022.

 

 
 [197] 
 
D. Kothandaraman, R. Chandra, and D. Manocha, “SS-SFDA: Self-supervised
source-free domain adaptation for road segmentation in hazardous
environments,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision , 2021, pp. 3049–3059.

 

 
 [198] 
 
D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully
test-time adaptation by entropy minimization,” arXiv preprint
arXiv:2006.10726 , 2020.

 

 
 [199] 
 
M. Bateson, H. Kervadec, J. Dolz, H. Lombaert, and I. Ben Ayed,
“Source-relaxed domain adaptation for image segmentation,” in
 International Conference on Medical Image Computing and
Computer-Assisted Intervention . Springer, 2020, pp. 490–499.

 

 
 [200] 
 
Y. Xu, J. Yang, H. Cao, K. Wu, M. Wu, and Z. Chen, “Source-free video domain
adaptation by learning temporal consistency for action recognition,” in
 European Conference on Computer Vision . Springer, 2022, pp. 147–164.

 

 
 [201] 
 
J. N. Kundu, A. Kulkarni, A. Singh, V. Jampani, and R. V. Babu, “Generalize
then adapt: Source-free domain adaptive semantic segmentation,” in
 Proceedings of the IEEE/CVF International Conference on Computer
Vision , 2021, pp. 7046–7056.

 

 
 [202] 
 
Y. Huang, X. Yang, J. Zhang, and C. Xu, “Relative alignment network for
source-free multimodal video domain adaptation,” in Proceedings of the
30th ACM International Conference on Multimedia , 2022, pp. 1652–1660.

 

 
 [203] 
 
Y. Wang, J. Liang, and Z. Zhang, “Give me your trained model: Domain
adaptive semantic segmentation without source data,” arXiv preprint
arXiv:2106.11653 , 2021.

 

 
 [204] 
 
J. Liang, D. Hu, J. Feng, and R. He, “UMAD: Universal model adaptation
under domain and category shift,” arXiv preprint arXiv:2112.08553 ,
2021.

 

 
 [205] 
 
J. Dong, Z. Fang, A. Liu, G. Sun, and T. Liu, “Confident anchor-induced
multi-source free domain adaptation,” Advances in Neural Information
Processing Systems , vol. 34, pp. 2848–2860, 2021.

 

 
 [206] 
 
Q. Tian, S. Peng, and T. Ma, “Source-free unsupervised domain adaptation with
trusted pseudo samples,” ACM Transactions on Intelligent Systems and
Technology , 2022.

 

 
 [207] 
 
R. Müller, S. Kornblith, and G. E. Hinton, “When does label smoothing
help?” Advances in Neural Information Processing Systems , vol. 32,
2019.

 

 
 [208] 
 
C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the
Inception architecture for computer vision,” in Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition , 2016, pp.
2818–2826.

 

 
 [209] 
 
K. Li, J. Lu, H. Zuo, and G. Zhang, “Source-free multi-domain adaptation with
generally auxiliary model training,” in 2022 International Joint
Conference on Neural Networks (IJCNN) . IEEE, 2022, pp. 1–8.

 

 
 [210] 
 
P. Chen and A. J. Ma, “Source-free temporal attentive domain adaptation for
video action recognition,” in Proceedings of the 2022 International
Conference on Multimedia Retrieval , 2022, pp. 489–497.

 

 
 [211] 
 
Z. Xu, W. Wei, L. Zhang, and J. Nie, “Source-free domain adaptation for
cross-scene hyperspectral image classification,” in IGARSS 2022-2022
IEEE International Geoscience and Remote Sensing Symposium . IEEE, 2022, pp. 3576–3579.

 

 
 [212] 
 
H. Yan, Y. Guo, and C. Yang, “Augmented self-labeling for source-free
unsupervised domain adaptation,” in NeurIPS 2021 Workshop on
Distribution Shifts: Connecting Methods and Applications , 2021.

 

 
 [213] 
 
C.-Y. Yang, Y.-J. Kuo, and C.-T. Hsu, “Source free domain adaptation for
semantic segmentation via distribution transfer and adaptive class-balanced
self-training,” in 2022 IEEE International Conference on Multimedia
and Expo (ICME) . IEEE, 2022, pp.
1–6.

 

 
 [214] 
 
L. Xiong, M. Ye, D. Zhang, Y. Gan, and Y. Liu, “Source data-free domain
adaptation for a faster R-CNN,” Pattern Recognition , vol. 124, p.
108436, 2022.

 

 
 [215] 
 
N. Ma, J. Bu, L. Lu, J. Wen, Z. Zhang, S. Zhou, and X. Yan, “Semi-supervised
hypothesis transfer for source-free domain adaptation,” arXiv preprint
arXiv:2107.06735 , 2021.

 

 
 [216] 
 
X. Guan, H. Sun, N. Liu, and H. Zhou, “Polycentric clustering and structural
regularization for source-free unsupervised domain adaptation,” arXiv
preprint arXiv:2210.07463 , 2022.

 

 
 [217] 
 
J. N. Kundu, A. R. Kulkarni, S. Bhambri, D. Mehta, S. A. Kulkarni, V. Jampani,
and V. B. Radhakrishnan, “Balancing discriminability and transferability for
source-free domain adaptation,” in International Conference on Machine
Learning . PMLR, 2022, pp.
11 710–11 728.

 

 
 [218] 
 
C. Li, W. Chen, X. Luo, Y. He, and Y. Tan, “Adaptive pseudo labeling for
source-free domain adaptation in medical image segmentation,” in
 ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP) . IEEE,
2022, pp. 1091–1095.

 

 
 [219] 
 
Y. Kim, D. Cho, K. Han, P. Panda, and S. Hong, “Domain adaptation without
source data,” IEEE Transactions on Artificial Intelligence , vol. 2,
no. 6, pp. 508–518, 2021.

 

 
 [220] 
 
V. Prabhu, S. Khare, D. Kartik, and J. Hoffman, “S4T: Source-free domain
adaptation for semantic segmentation via self-supervised selective
self-training,” arXiv preprint arXiv:2107.10140 , 2021.

 

 
 [221] 
 
M. Shen, Y. Bu, and G. Wornell, “On the benefits of selectivity in
pseudo-labeling for unsupervised multi-source-free domain adaptation,”
 arXiv preprint arXiv:2202.00796 , 2022.

 

 
 [222] 
 
T. Han, X. Gong, F. Feng, J. Zhang, Z. Sun, and Y. Zhang, “Privacy preserving
mutli-source domain adaptaion for medical data,” IEEE Journal of
Biomedical and Health Informatics , 2022.

 

 
 [223] 
 
X. Peng, Z. Huang, Y. Zhu, and K. Saenko, “Federated adversarial domain
adaptation,” arXiv preprint arXiv:1911.02054 , 2019.

 

 
 [224] 
 
H. Feng, Z. You, M. Chen, T. Zhang, M. Zhu, F. Wu, C. Wu, and W. Chen,
“KD3A: Unsupervised multi-source decentralized domain adaptation via
knowledge distillation.” in ICML , 2021, pp. 3274–3283.

 

 
 [225] 
 
Y. Kang, Y. He, J. Luo, T. Fan, Y. Liu, and Q. Yang, “Privacy-preserving
federated adversarial domain adaptation over feature groups for
interpretability,” IEEE Transactions on Big Data , 2022.

 

 
 [226] 
 
L. Song, C. Ma, G. Zhang, and Y. Zhang, “Privacy-preserving unsupervised
domain adaptation in federated setting,” IEEE Access , vol. 8, pp.
143 233–143 240, 2020.

 

 
 [227] 
 
Z. Qin, L. Yang, F. Gao, Q. Hu, and C. Shen, “Uncertainty-aware aggregation
for federated open set domain adaptation,” IEEE Transactions on Neural
Networks and Learning Systems , 2022.

 

 
 [228] 
 
K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman, V. Ivanov,
C. Kiddon, J. Konečnỳ, S. Mazzocchi, B. McMahan et al. ,
“Towards federated learning at scale: System design,” Proceedings
of Machine Learning and Systems , vol. 1, pp. 374–388, 2019.

 

 
 [229] 
 
T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning:
Challenges, methods, and future directions,” IEEE Signal Processing
Magazine , vol. 37, no. 3, pp. 50–60, 2020.

 

 
 [230] 
 
K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel,
D. Ramage, A. Segal, and K. Seth, “Practical secure aggregation for
privacy-preserving machine learning,” in proceedings of the 2017 ACM
SIGSAC Conference on Computer and Communications Security , 2017, pp.
1175–1191.

 

 
 [231] 
 
L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” Advances in
Neural Information Processing Systems , vol. 32, 2019.

 

 
 [232] 
 
A. Z. Tan, H. Yu, L. Cui, and Q. Yang, “Towards personalized federated
learning,” IEEE Transactions on Neural Networks and Learning Systems ,
2022.

 

 
 [233] 
 
C.-H. Yao, B. Gong, H. Qi, Y. Cui, Y. Zhu, and M.-H. Yang, “Federated
multi-target domain adaptation,” in Proceedings of the IEEE/CVF Winter
Conference on Applications of Computer Vision , 2022, pp. 1424–1433.

 

 
 [234] 
 
D. Shenaj, E. Fanì, M. Toldo, D. Caldarola, A. Tavera, U. Michieli,
M. Ciccone, P. Zanuttigh, and B. Caputo, “Learning across domains and
devices: Style-driven source-free domain adaptation in clustered federated
learning,” arXiv preprint arXiv:2210.02326 , 2022.

 

 
 [235] 
 
M. Yazdanpanah and P. Moradi, “Visual domain bridge: A source-free domain
adaptation for cross-domain few-shot learning,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp.
2868–2877.

 

 
 [236] 
 
N. Karani, E. Erdil, K. Chaitanya, and E. Konukoglu, “Test-time adaptable
neural networks for robust medical image segmentation,” Medical Image
Analysis , vol. 68, p. 101907, 2021.

 

 
 [237] 
 
M. Boudiaf, R. Mueller, I. Ben Ayed, and L. Bertinetto, “Parameter-free online
test-time adaptation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2022, pp. 8344–8353.

 

 
 [238] 
 
Y. Iwasawa and Y. Matsuo, “Test-time classifier adjustment module for
model-agnostic domain generalization,” Advances in Neural Information
Processing Systems , vol. 34, pp. 2427–2440, 2021.

 

 
 [239] 
 
W. Ma, C. Chen, S. Zheng, J. Qin, H. Zhang, and Q. Dou, “Test-time adaptation
with calibration of medical image classification nets for label distribution
shift,” in International Conference on Medical Image Computing and
Computer-Assisted Intervention . Springer, 2022, pp. 313–323.

 

 
 [240] 
 
F. You, J. Li, and Z. Zhao, “Test-time batch statistics calibration for
covariate shift,” arXiv preprint arXiv:2110.04065 , 2021.

 

 
 [241] 
 
Y. He, A. Carass, L. Zuo, B. E. Dewey, and J. L. Prince, “Autoencoder based
self-supervised test-time adaptation for medical image analysis,”
 Medical Image Analysis , vol. 72, p. 102136, 2021.

 

 
 [242] 
 
He, Yufan and Carass, Aaron and Zuo, Lianrui and Dewey, Blake E and Prince,
Jerry L, “Self domain adapted network,” in International Conference
on Medical Image Computing and Computer-Assisted Intervention . Springer, 2020, pp. 437–446.

 

 
 [243] 
 
H. Yang, C. Chen, M. Jiang, Q. Liu, J. Cao, P. A. Heng, and Q. Dou, “DLTTA:
Dynamic learning rate for test-time adaptation on cross-domain medical
images,” arXiv preprint arXiv:2205.13723 , 2022.

 

 
 [244] 
 
J. N. Kundu, N. Venkat, A. Revanur, R. V. Babu et al. , “Towards
inheritable models for open-set domain adaptation,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2020,
pp. 12 376–12 385.

 

 
 [245] 
 
Y. Zhao, Z. Zhong, Z. Luo, G. H. Lee, and N. Sebe, “Source-free open compound
domain adaptation in semantic segmentation,” IEEE Transactions on
Circuits and Systems for Video Technology , vol. 32, no. 10, pp. 7019–7032,
2022.

 

 
 [246] 
 
K. Saito, D. Kim, S. Sclaroff, and K. Saenko, “Universal domain adaptation
through self supervision,” Advances in Neural Information Processing
Systems , vol. 33, pp. 16 282–16 292, 2020.

 

 
 [247] 
 
Y. Luo, Z. Wang, Z. Chen, Z. Huang, and M. Baktashmotlagh, “Source-free
progressive graph learning for open-set domain adaptation,” arXiv
preprint arXiv:2202.06174 , 2022.

 

 
 [248] 
 
R. Xu, Z. Chen, W. Zuo, J. Yan, and L. Lin, “Deep cocktail network:
Multi-source unsupervised domain adaptation with category shift,” in
 Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition , 2018, pp. 3964–3973.

 

 
 [249] 
 
E. Lekhtman, Y. Ziser, and R. Reichart, “DILBERT: Customized pre-training
for domain adaptation with category shift, with an application to aspect
extraction,” in Proceedings of the 2021 Conference on Empirical
Methods in Natural Language Processing , 2021, pp. 219–230.

 

 
 [250] 
 
K. Saito and K. Saenko, “OVANet: One-vs-all network for universal domain
adaptation,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision , 2021, pp. 9000–9009.

 

 
 [251] 
 
Y. Feng, J. Chen, S. He, T. Pan, and Z. Zhou, “Globally localized multisource
domain adaptation for cross-domain fault diagnosis with category shift,”
 IEEE Transactions on Neural Networks and Learning Systems , 2021.

 

 
 [252] 
 
T. Elsken, J. H. Metzen, and F. Hutter, “Neural architecture search: A
survey,” The Journal of Machine Learning Research , vol. 20, no. 1,
pp. 1997–2017, 2019.

 

 
 [253] 
 
M. Wistuba, A. Rawat, and T. Pedapati, “A survey on neural architecture
search,” arXiv preprint arXiv:1905.01392 , 2019.

 

 
 [254] 
 
Z. Lu, G. Sreekumar, E. Goodman, W. Banzhaf, K. Deb, and V. N. Boddeti,
“Neural architecture transfer,” IEEE Transactions on Pattern Analysis
and Machine Intelligence , vol. 43, no. 9, pp. 2971–2989, 2021.

 

 
 [255] 
 
P. Ren, Y. Xiao, X. Chang, P.-Y. Huang, Z. Li, X. Chen, and X. Wang, “A
comprehensive survey of neural architecture search: Challenges and
solutions,” ACM Computing Surveys (CSUR) , vol. 54, no. 4, pp. 1–34,
2021.

 

 
 [256] 
 
S. M. Ahmed, S. Lohit, K.-C. Peng, M. J. Jones, and A. K. Roy-Chowdhury,
“Cross-modal knowledge transfer without task-relevant source data,” in
 European Conference on Computer Vision . Springer, 2022, pp. 111–127.

 

 
 [257] 
 
R. Kemker, M. McClure, A. Abitino, T. Hayes, and C. Kanan, “Measuring
catastrophic forgetting in neural networks,” in Proceedings of the
AAAI Conference on Artificial Intelligence , vol. 32, no. 1, 2018.

 

 
 [258] 
 
J. Zhang, W. Li, C. Tang, P. Ogunbona et al. , “Unsupervised domain
expansion from multiple sources,” arXiv preprint arXiv:2005.12544 ,
2020.

 

 
 [259] 
 
M. Jing, X. Zhen, J. Li, and C. G. Snoek, “Variational model perturbation for
source-free domain adaptation,” arXiv preprint arXiv:2210.10378 ,
2022.

 

 
 [260] 
 
J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A.
Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al. ,
“Overcoming catastrophic forgetting in neural networks,” Proceedings
of the National Academy of Sciences , vol. 114, no. 13, pp. 3521–3526, 2017.

 

 
 [261] 
 
Z. Li and D. Hoiem, “Learning without forgetting,” IEEE Transactions on
Pattern Analysis and Machine Intelligence , vol. 40, no. 12, pp. 2935–2947,
2017.

 

 
 [262] 
 
D. Lopez-Paz and M. Ranzato, “Gradient episodic memory for continual
learning,” Advances in Neural Information Processing Systems ,
vol. 30, 2017.

 

 
 [263] 
 
M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis,
G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying
forgetting in classification tasks,” IEEE Transactions on Pattern
Analysis and Machine Intelligence , vol. 44, no. 7, pp. 3366–3385, 2021.

 

 
 [264] 
 
S. Tang, P. Su, D. Chen, and W. Ouyang, “Gradient regularized contrastive
learning for continual domain adaptation,” in Proceedings of the AAAI
Conference on Artificial Intelligence , vol. 35, no. 3, 2021, pp. 2665–2673.

 

 
 [265] 
 
Q. Wang, O. Fink, L. Van Gool, and D. Dai, “Continual test-time domain
adaptation,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , 2022, pp. 7201–7211.

 

 
 [266] 
 
A. M. N. Taufique, C. S. Jahan, and A. Savakis, “ConDA: Continual
unsupervised domain adaptation,” arXiv preprint arXiv:2103.11056 ,
2021.

 

 
 [267] 
 
B. Chidlovskii, S. Clinchant, and G. Csurka, “Domain adaptation in the absence
of source domain data,” in Proceedings of the 22nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining , 2016, pp.
451–460.

 

 
 [268] 
 
A. R. Nelakurthi, R. Maciejewski, and J. He, “Source free domain adaptation
using an off-the-shelf classifier,” in 2018 IEEE International
Conference on Big Data (Big Data) . IEEE, 2018, pp. 140–145.

 

 
 [269] 
 
F. Wang, Z. Han, Z. Zhang, and Y. Yin, “Active source free domain
adaptation,” arXiv preprint arXiv:2205.10711 , 2022.

 

 
 [270] 
 
D. Kothandaraman, S. Shekhar, A. Sancheti, M. Ghuhan, T. Shukla, and
D. Manocha, “DistillAdapt: Source-free active visual domain
adaptation,” arXiv preprint arXiv:2205.12840 , 2022.

 

 
 [271] 
 
B. Yang, A. J. Ma, and P. C. Yuen, “Revealing task-relevant model memorization
for source-protected unsupervised domain adaptation,” IEEE
Transactions on Information Forensics and Security , vol. 17, pp. 716–731,
2022.

 

 
 [272] 
 
X. Wang, J. Zhuo, S. Cui, and S. Wang, “Learning invariant representation with
consistency and diversity for semi-supervised source hypothesis transfer,”
 arXiv preprint arXiv:2107.03008 , 2021.