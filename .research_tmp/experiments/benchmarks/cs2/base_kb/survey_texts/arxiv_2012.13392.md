Deep Learning-Based Human Pose Estimation: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2012.13392v5 [cs.CV] 03 Jul 2023 
 
 

# Deep Learning-Based Human Pose Estimation: A Survey

 DOI:  10.1145/1122445.1122456 Journal:  JACM Volume:  37 4 111 8 CCS:  Computing methodologies Computer vision CCS:  General and reference Surveys and overviews 
 
 
 Ce Zheng
 
 Note:  Both authors contributed equally to this research.
 
 email: cezheng@knights.ucf.edu 
 
 Affiliation:  University of Central Florida , 4328 Scorpius St, Suite 245 , Orlando , Florida , USA , 32816 
 
 , 
 Wenhan Wu
 
 email: wwu25@uncc.edu 
 
 Affiliation:  University of North Carolina at Charlotte , 9201 University City Blvd , Charlotte , North Carolina , USA , 28223 
 
 , 
 Chen Chen
 
 email: chen.chen@crcv.ucf.edu 
 
 Affiliation:  University of Central Florida , 4328 Scorpius St, Suite 245 , Orlando , Florida , USA , 32816 
 
 , 
 Taojiannan Yang
 
 email: taoyang1122@knights.ucf.edu 
 
 Affiliation:  University of Central Florida , 4328 Scorpius St, Suite 245 , Orlando , Florida , USA , 32816 
 
 , 
 Sijie Zhu
 
 email: sizhu@knights.ucf.edu 
 
 Affiliation:  University of Central Florida , 4328 Scorpius St, Suite 245 , Orlando , Florida , USA , 32816 
 
 , 
 Ju Shen
 
 email: jshen1@udayton.edu 
 
 Affiliation:  University of Dayton , 300 College Park , Dayton , Ohio , USA , 45469 
 
 , 
 Nasser Kehtarnavaz
 
 email: kehtar@utdallas.edu 
 
 Affiliation:  University of Texas at Dallas , 800 W. Campbell Road , Richardson , Texas , USA , 75080 
 
 and 
 Mubarak Shah
 
 email: shah@crcv.ucf.edu 
 
 Affiliation:  University of Central Florida , 4328 Scorpius St, Suite 245 , Orlando , Florida , USA , 32816 
 
 2018 

 Abstract. 
 
 Human pose estimation aims to locate the human body parts and build human body representation (e.g., body skeleton) from input data such as images and videos. It has drawn increasing attention during the past decade and has been utilized in a wide range of applications including human-computer interaction, motion analysis, augmented reality, and virtual reality. Although the recently developed deep learning-based solutions have achieved high performance in human pose estimation, there still remain challenges due to insufficient training data, depth ambiguities, and occlusion.
The goal of this survey paper is to provide a comprehensive review of recent deep learning-based solutions for both 2D and 3D pose estimation via a systematic analysis and comparison of these solutions based on their input data and inference procedures. More than 260 research papers since 2014 are covered in this survey. Furthermore, 2D and 3D human pose estimation datasets and evaluation metrics are included. Quantitative performance comparisons of the reviewed methods on popular datasets are summarized and discussed. Finally, the challenges involved, applications, and future research directions are concluded. A regularly updated project page is provided:
 https://github.com/zczcwh/DL-HPE 

 
 
 
 Keywords:  Survey of human pose estimation, 2D and 3D pose estimation, deep learning-based pose estimation, pose estimation datasets, pose estimation metrics
 
 

## 1. Introduction

 
 Human pose estimation (HPE), which has been extensively studied in computer vision literature, involves estimating the configuration of human body parts from input data captured by sensors, in particular images and videos. HPE provides geometric and motion information about the human body which has been applied to a wide range of applications (e.g., human-computer interaction, motion analysis, augmented reality (AR), virtual reality (VR), healthcare, etc.). With the rapid development of deep learning solutions in recent years, such solutions have been shown to outperform classical computer vision methods in various tasks including image classification ( Krizhevsky
et al., 2012 ) , semantic segmentation ( Long
et al., 2015 ) , and object detection ( Ren
et al., 2015 ) . Significant progress and remarkable performance have already been made by employing deep learning techniques in HPE tasks. However, challenges such as occlusion, insufficient training data, and depth ambiguity still pose difficulties to be overcome. 2D HPE from images and videos with 2D pose annotations is easily achievable and high performance has been reached for the human pose estimation of a single person using deep learning techniques. More recently, attention has been paid to highly occluded multi-person HPE in complex scenes. In contrast, for 3D HPE, obtaining accurate 3D pose annotations is much more difficult than its 2D counterpart. Motion capture systems can collect 3D pose annotation in controlled lab environments; however, they have limitations for in-the-wild environments. For 3D HPE from monocular RGB images and videos, the main challenge is depth ambiguities. In a multi-view setting, viewpoints association is the key issue that needs to be addressed. Some works have utilized sensors such as depth sensors, inertial measurement units (IMUs), and radio frequency devices, but these approaches are usually not cost-effective and require special-purpose hardware.

 
 
 Given the rapid progress in HPE research, this article attempts to track recent advances and summarize their achievements in order to provide
a clear picture of current research on deep learning-based 2D and 3D HPE.

 
 
 Figure 1. Taxonomy of this survey. 
 
 

### 1.1. Previous surveys and our contributions

 
 There are several related surveys and reviews previously reported on HPE. Among them, ( Moeslund and
Granum, 2001 ; Moeslund
et al., 2006 ; Poppe, 2007 ; Ji and Liu, 2009 ) focus on the general field of visual-based human motion capture including pose estimation, tracking, and action recognition. Therefore, pose estimation is only one of the topics covered in these surveys. The research works on 3D HPE before 2012 are reviewed in ( Holte
et al., 2012 ) . The body parts parsing-based methods for single-view and multi-view HPE are reported in ( Liu
et al., 2015 ) . These surveys published during 2001-2015 mainly focused on conventional methods without deep learning.
A survey on both traditional and deep learning-based methods related to HPE is presented in ( Gong et al., 2016 ) . However, only a handful of deep learning-based approaches are included. The survey in ( Sarafianos et al., 2016 ) covers 3D HPE methods with RGB inputs, while the survey in ( Munea et al., 2020 ) only reviews 2D HPE methods. Monocular HPE from the classical to recent deep learning-based methods (till 2019, less than 100 papers) is summarized in ( Chen
et al., 2020f ) . However, it only covers 2D HPE and 3D single-view HPE from monocular cameras. 3D multi-view HPE from monocular cameras and 3D HPE from other sensors are ignored. Also, no extensive performance comparisons or in-depth analyses are given, and the discussion on existing challenges and future directions is relatively short.

 
 
 This survey aims to address the shortcomings of the previous surveys in terms of providing a systematic review of the recent deep learning-based solutions to 2D and 3D HPE but also covering other aspects of HPE including the performance evaluation of (2D and 3D) HPE methods on popular datasets, their applications, and comprehensive discussion. The key points that distinguish this survey from the previous ones are as follows: 

 
 • 
 
 A comprehensive review of recent deep learning-based 2D and 3D HPE methods (up to 2022 with more than 260 papers) is provided by categorizing them according to 2D or 3D scenarios, single-view or multi-view, from monocular images/videos or other sources, and learning paradigm.

 

 • 
 
 Extensive performance evaluation of 2D and 3D HPE methods. We summarize and compare reported performances of promising methods on common datasets based on their categories. The comparison of results provides cues for the strengths and weaknesses of different methods, revealing the research trends and future directions of HPE.

 

 • 
 
 An overview of a wide range of HPE applications, such as surveillance, AR/VR, and healthcare.

 

 • 
 
 An thorough discussion of 2D and 3D HPE is presented in terms of key challenges in HPE pointing to potential future research toward improving performance.

 

 
 
 
 

### 1.2. Paper organization

 
 HPE is divided into two main categories: 2D HPE (§  2 ) and 3D HPE (§  3 ). Fig. 1 shows the taxonomy of deep learning methods for HPE.
According to the number of people, 2D HPE methods are categorized into single-person and multi-person settings.
For single-person methods (§  2.1 ), there are two categories: regression methods and
heatmap-based methods.
For multi-person methods (§  2.2 ), there are also two types of methods: top-down methods and
bottom-up methods.

 
 
 3D HPE methods are classified according to the input source types: monocular RGB images and videos (§  3.1 ), or other sensors (e.g., inertial measurement unit sensors, §  3.2 ). The majority of these methods use monocular RGB images and videos, and they are further divided into single-view single-person (§  3.1.1 ); single-view multi-person (§  3.1.2 ); and multi-view methods (§  3.1.3 ). Multi-view settings are deployed mainly for multi-person pose estimation. Hence, single-person or multi-person is not specified in this category.

 
 
 Next, depending on the 2D and 3D HPE pipelines, the datasets and evaluation metrics commonly used are summarized followed by a comparison of results of the promising methods (§  4 ). In addition, various applications of HPE such as AR/VR are mentioned (§  5 ). Finally, the paper ends by an
thorough discussion of
some promising directions for future research
(§  6 ).

 
 
 
 

## 2. 2D human pose estimation

 
 2D HPE methods estimate the 2D position or spatial location of human body keypoints from images or videos. Traditional 2D HPE methods adopt different hand-crafted feature extraction techniques for body parts, and these early works describe the human body as a stick figure to obtain global pose structures. Recently, deep learning-based approaches have achieved a major breakthrough in HPE by improving the results significantly. In the following, we review deep learning-based 2D HPE methods with respect to single-person and multi-person scenarios.

 
 

### 2.1. 2D single-person pose estimation

 
 2D single-person pose estimation is used to localize human body joint positions when the input is a single-person image. If there are several people, the input image is cropped first so that there is only one person in each cropped patch (or sub-image). This process can be achieved automatically by an upper-body detector ( Micilotta
et al., 2006 ) or a full-body detector ( Ren
et al., 2015 ) .
In general, there are two categories for single-person pipelines that employ deep learning techniques: regression methods and heatmap-based methods. Regression methods apply an end-to-end framework to learn a mapping from the input image to the positions of body joints or parameters of human body models ( Toshev and
Szegedy, 2014 ) . The goal of heatmap-based methods is to predict approximate locations of body parts and joints ( Chen and Yuille, 2014 ) ( Newell
et al., 2016 ) , which are supervised by heatmaps representation ( Tompson et al., 2015 ; Wei
et al., 2016 ) . Heatmap-based frameworks are now widely used in 2D HPE tasks. The general frameworks of 2D single-person HPE methods are depicted in Fig. 2 .

 
 
 Figure 2. Single-person 2D HPE frameworks. (a) Regression methods directly learn a mapping (via a deep neural network) from the original image to the kinematic body model and produce joint coordinates. (b) Given the ground-truth 2D pose, the ground-truth heatmaps of each joint are generated by applying a Gaussian kernel to each joint’s location. Then, heatmap-based methods utilize a model to predict the heatmap of each joint. 
 
 

#### 2.1.1. Regression methods

 
 There are many works based on the regression framework (e.g., ( Toshev and
Szegedy, 2014 ; Pfister et al., 2014 ; Carreira et al., 2016 ; Sun
et al., 2017 ; Luvizon
et al., 2019 ; Nibali
et al., 2018 ; Li et al., 2014 ; Fan
et al., 2015 ; Luvizon
et al., 2018 ; Zhang
et al., 2019b ; Li
et al., 2021c ; Panteleris and
Argyros, 2021 ; Mao
et al., 2021 ; Mao et al., 2022 ) ) to predict joint coordinates from images as shown in Fig. 2 (a). Using AlexNet ( Krizhevsky
et al., 2012 ) as the backbone, Toshev and Szegedy ( Toshev and
Szegedy, 2014 ) proposed a cascaded deep neural network regressor named DeepPose to learn keypoints from images. Due to the impressive performance of DeepPose, the research paradigm of HPE began to shift from classic approaches to deep learning, in particular convolutional neural networks (CNNs).
Sun et al. ( Sun
et al., 2017 ) introduced a structure-aware regression method called "compositional pose regression" based on ResNet-50 ( He
et al., 2016 ) . This method adopts a re-parameterized and bone-based representation that contains human body information and pose structure, instead of the traditional joint-based representation. Luvizon et al. ( Luvizon
et al., 2019 ) proposed an end-to-end regression approach for HPE using soft-argmax function to convert feature maps into joint coordinates in a fully differentiable framework. Li et al. ( Li
et al., 2021c ) first designed a transformer-based cascade network for regressing human keypoints. The spatial correlation of joints and appearance is captured by self-attention mechanism. Different from the previous methods, Li et al. ( Li
et al., 2021a ) proposed a normalizing flow model named RLE (Log-likelihood Estimation) to capture the distribution of joint location, aiming for finding the optimized parameters by residual log-likelihood estimation.

 
 
 A good feature that encodes rich pose information is critical for regression-based methods. One popular strategy to learn better feature representation is multi-task learning ( Ruder, 2017 ) . By sharing representations between related tasks (e.g., pose estimation and pose-based action recognition), the model can generalize better on the original task (pose estimation).
Following this direction, Li et al. ( Li et al., 2014 ) proposed a heterogeneous multi-task framework that consists of two tasks: predicting joint coordinates from full images by a regressor and detecting body parts from image patches using a sliding window. Fan et al. ( Fan
et al., 2015 ) proposed a dual-source (i.e., image patches and full images) CNN for two tasks: joint detection which determines whether a patch contains a body joint, and joint localization which finds the exact location
of the joint in the patch. Each task corresponds to a loss function, and the combination of two tasks leads to improved results.

 
 
 

#### 2.1.2. Heatmap-based methods

 
 Instead of estimating the 2D coordinates of human joints directly, heatmap-based methods for HPE aim to estimate the 2D heatmaps which are generated by adding 2D Gaussian kernels on each joint’s location as shown in Fig. 2 .
Concretely, the goal is to estimate K K heatmaps { H 1 , H 2 , … , H K } \{H_{1},H_{2},...,H_{K}\} for a total of K K keypoints. The pixel value H i ​ ( x , y ) H_{i}(x,y) in each keypoint heatmap indicates the probability that the keypoint lies in the position ( x , y ) (x,y) (see Fig. 2 (b)). The target (or ground-truth) heatmap is generated by a 2D Gaussian centered at the ground-truth joint location ( Tompson et al., 2015 ) ( Tompson
et al., 2014 ) . Thus pose estimation networks are trained by minimizing the discrepancy (e.g., the Mean Squared-Error (MSE)) between the predicted heatmaps and target heatmaps. Compared with joint coordinates, heatmaps preserve the spatial location information while it can make the training process smoother.

 
 
 Therefore, there is a recent growing interest in leveraging heatmaps to represent the joint locations and developing effective CNN architectures for HPE, e.g., ( Tompson
et al., 2014 ; Ramakrishna et al., 2014 ; Tompson et al., 2015 ; Lifshitz
et al., 2016 ; Bulat and
Tzimiropoulos, 2016 ; Newell
et al., 2016 ; Wei
et al., 2016 ; Gkioxari
et al., 2016 ; Belagiannis and
Zisserman, 2017 ; Yang
et al., 2017 ; Luo et al., 2018 ; Debnath et al., 2018 ; Xiao et al., 2018 ; Zhang et al., 2019a ; Artacho and
Savakis, 2020 ; Li et al., 2022b ) .
As one of the fundamental works, Wei et al. ( Wei
et al., 2016 ) introduced a convolutional networks-based sequential framework named Convolutional Pose Machines (CPM) to predict the locations of keypoints with multi-stage processing (the convolutional networks in each stage utilize the 2D belief maps generated from previous stages and produce the increasingly refined predictions of body part locations). At the same time, Newell et al. ( Newell
et al., 2016 ) proposed an encoder-decoder network named "stacked hourglass" to repeat bottom-up and top-down processing with intermediate supervision. In this work, the encoder squeezes features through the bottleneck and then the decoder expands them for the substage.
The stacked hourglass (SHG) network consists of consecutive steps of pooling and upsampling layers to capture information at every scale. Since then, complex variations of the SHG architecture were developed for HPE. Following ( Newell
et al., 2016 ) , Chu et al. ( Chu
et al., 2017 ) designed novel Hourglass Residual Units (HRUs), which extend the residual units with a side branch of filters with larger receptive fields, to capture features from various scales. Yang et al. ( Yang
et al., 2017 ) designed a multi-branch Pyramid Residual Module (PRM) to replace the residual unit in SHG, leading to enhanced invariance in scales of deep CNNs. Sun et al. ( Sun
et al., 2019 ) presented a novel High-Resolution Net (HRNet) to learn reliable high-resolution representations by connecting multi-resolution subnetworks in parallel and conducting repeated multi-scale fusions, which results in more accurate keypoint heatmap prediction. Inspired by HRNet, Yu et al. ( Yu
et al., 2021 ) introduced a light-weighted HRNet named Lite-HRNet, which designed conditional channel weighting blocks to exchange information between channels and resolutions. Recently, due to the superior performance, the HRNet ( Sun
et al., 2019 ) and its variations ( Cheng et al., 2020 ; Yuan et al., 2021 ; Yu
et al., 2021 ) have been widely adopted in HPE and other pose-related tasks.

 
 
 With the emergence of Generative Adversarial Networks (GANs) ( Goodfellow et al., 2014 ) , they are explored in HPE to generate biologically plausible pose configurations and to discriminate the predictions with high confidence from those with low confidence, which could infer the poses of the occluded body parts. Inspired by hourglass architecture which efficiently refines joints, Chen et al. ( Chen
et al., 2017 ) constructed a structure-aware conditional adversarial network–Adversarial PoseNet–which contains an hourglass network-based pose generator and two discriminators to discriminate reasonable body poses from unreasonable ones. Chou et al. ( Chou
et al., 2018 ) built an adversarial learning-based network with two stacked hourglass networks sharing the same structure as the discriminator and generator, respectively. The generator estimates the location of each joint, and the discriminator distinguishes between the ground-truth heatmaps and predicted ones. Unlike GANs-based methods that take the HPE network as the generator and utilize the discriminator to provide supervision, Peng et al. ( Peng
et al., 2018 ) developed an adversarial data augmentation network to optimize data augmentation and network training by treating the HPE network as a discriminator and using augmentation network as a generator to perform adversarial augmentations.

 
 
 Besides these efforts in designing effective networks for HPE, body structure information is also investigated to provide more and better supervision information for building HPE networks. Yang et al. ( Yang
et al., 2016 ) designed an end-to-end CNN framework for HPE, which can find hard negatives by incorporating the spatial and appearance consistency among human body parts. A structured feature-level learning framework was proposed in ( Chu
et al., 2016 ) for reasoning the correlations among human body joints in HPE, which captures richer information of human body joints and improves the learning results.
Ke et al. ( Ke
et al., 2018 ) designed a multi-scale structure-aware neural network, which combines multi-scale supervision, multi-scale feature combination, structure-aware loss information scheme, and a keypoint masking training method to improve HPE models in complex scenarios. Tang et al. ( Tang et al., 2018 ) built an hourglass-based supervision network, termed as Deeply Learned Compositional Model, to describe the complex and realistic relationships among body parts and learn the compositional pattern information (the orientation, scale, and shape information of each body part) in human bodies. Different from the previous approaches which consider all body parts, Tang and Wu ( Tang and Wu, 2019 ) revealed that not all parts are related to each other, therefore introducing a Part-based Branches Network to learn representations specific to each part group rather than a shared representation for all parts.

 
 
 Human poses in video sequences are (3D) spatio-temporal signals. Therefore, modeling spatio-temporal information is important for HPE from videos. Jain et al. ( Jain
et al., 2014 ) designed a two-branch CNN framework to incorporate both color and motion features within frame pairs to build an expressive temporal-spatial model in HPE.
Pfister et al. ( Pfister
et al., 2015 ) proposed a CNN that can utilize temporal context information from multiple frames by using optical flow to align predicted heatmaps from neighboring frames. Different from the previous video-based methods which are computationally intensive, Luo et al. ( Luo et al., 2018 ) introduced a recurrent structure with Long Short-Term Memory to capture temporal geometric consistency and dependency from different frames. This method results in faster training time for the HPE network for videos. Zhang et al. ( Zhang
et al., 2020e ) introduced a keyframe proposal network for capturing spatial and temporal information from frames, and a human pose interpolation module for efficient video-based HPE.

 
 
 
 

### 2.2. 2D multi-person pose estimation 

 
 Compared to single-person HPE, multi-person HPE is more difficult and challenging because it needs to figure out the number of people and their positions, and how to group keypoints for different people. In order to solve these problems, multi-person HPE methods can be classified into top-down and bottom-up methods. Top-down methods employ off-the-shelf person detectors to obtain a set of boxes (each corresponding to one person) from the input images and then apply single-person pose estimators to each person box to generate multi-person poses. Different from top-down methods, bottom-up methods locate all the body joints in one image first and then group them into individual subjects. In the top-down pipeline, the number of people in the input image will directly affect the computing time. The computing speed for bottom-up methods is usually faster than top-down methods since they do not need to detect the pose for each person separately. Fig. 3 shows the general frameworks for 2D multi-person HPE methods.

 
 
 Figure 3. Illustration of the multi-person 2D HPE frameworks. (a) Top-down approaches have two sub-tasks: (1) human detection and (2) pose estimation in the region of a single human; (b) Bottom-up approaches also have two sub-tasks: (1) detect all keypoints candidates of body parts and (2) associate body parts in different human bodies and assemble them into individual pose representations. 
 
 

#### 2.2.1. Top-down pipeline

 
 In the top-down pipeline as shown in Fig. 3 (a), there are two important parts: a human body detector to obtain person bounding boxes and a single-person pose estimator to predict the locations of keypoints within these bounding boxes. A line of works focus on designing and improving the modules in HPE networks, e.g., ( Papandreou et al., 2017 ; Huang
et al., 2017 ; Xiao et al., 2018 ; Sun
et al., 2019 ; Li et al., 2019a ; Moon
et al., 2019b ; Wang
et al., 2020b ; Huang
et al., 2020d ; Cai et al., 2020 ; Zhang
et al., 2020h ; Liu
et al., 2021b ) . To answer the question "how good could a simple method be" in building an HPE network, Xiao et al. ( Xiao et al., 2018 ) added a few deconvolutional layers in the ResNet (backbone network) to build a simple yet effective structure to produce heatmaps for high-resolution representations.
To improve the accuracy of keypoint localization, Wang et al. ( Wang
et al., 2020b ) introduced a two-stage graph-based and model-agnostic framework, called Graph-PCNN. It consists of a localization subnet to obtain rough keypoint locations and a graph pose refinement module to get refined keypoints localization representations.
Cai et al. ( Cai et al., 2020 ) introduced a multi-stage network with a Residual Steps Network (RSN) module to learn delicate local representations by efficient intra-level feature fusion strategies, and a Pose Refine Machine (PRM) module to find a trade-off between local and global representations in the features.

 
 
 Estimating poses under occlusion and truncation scenes often occurs in multi-person settings since the overlapping of limbs is inevitable.
Human detectors may fail in the first step of top-down pipeline due to occlusion. Thus, robustness to occlusion or truncation is an important aspect of the multi-person HPE approaches. Towards this goal, Iqbal and Gall ( Iqbal and Gall, 2016 ) built a convolutional pose machine-based pose estimator to estimate the joint candidates. Then they used integer linear programming to solve the joint-to-person association problem and obtain human body poses even in presence of severe occlusions. Fang et al. ( Fang
et al., 2017 ) designed a regional multi-person pose estimation (RMPE) approach to improve the performance of HPE in complex scenes. The RMPE framework has three parts: Symmetric Spatial Transformer Network (to detect single person region within an inaccurate bounding box), Parametric Pose Non-Maximum-Suppression (to solve the redundant detection problem), and Pose-Guided Proposals
Generator (to augment training data). Papandreou et al. ( Papandreou et al., 2017 ) proposed a two-stage architecture with a Faster R-CNN person detector to create bounding boxes for candidate human bodies and a keypoint estimator to predict the locations of keypoints by using a form of heatmap-offset aggregation. The method works well in occluded and cluttered scenes. To alleviate the occlusion problem in HPE, Chen et al. ( Chen
et al., 2018b ) presented a Cascade Pyramid Network (CPN) which includes two parts: GlobalNet (a feature pyramid network to predict the invisible keypoints) and RefineNet (a network to integrate all levels of features from the GlobalNet with a keypoint mining loss). Their results reveal that CPN has a good performance in predicting occluded keypoints.
Su et al. ( Su
et al., 2019 ) designed two modules, the Channel Shuffle Module and the Spatial Channel-wise Attention Residual Bottleneck, to achieve channel-wise and spatial information enhancement for better multi-person HPE under occluded scenes.
Qiu et al. ( Qiu et al., 2020 ) developed an Occluded Pose Estimation and Correction module and an occluded pose dataset to solve the occlusion problem in crowd pose estimation.
Umer et al. ( Umer
et al., 2020 ) proposed a keypoint correspondence framework to recover missed poses using temporal information of the previous frame in occluded scenes. The network is trained using self-supervision to improve the pose estimation results on sparsely annotated video datasets.

 
 
 Recently, transformer-based methods have attracted more attention ( Li
et al., 2021b ; Li
et al., 2021e ; Yang
et al., 2021 ; Yuan et al., 2021 ; Shi
et al., 2022 ; Ma et al., 2022 ) since the attention modules in transformer can obtain long-range dependencies and global evidence of the predicted keypoints, which are more powerful than CNNs. The early exploration ( Yang
et al., 2021 ) proposed a transformer-based model for 2D HPE named TransPose, which utilizes the attention layers to predict the heatmaps of the keypoints and learn the fine-grained evidence for HPE in occlusion scenarios. Following ( Yang
et al., 2021 ) , Li et al. ( Li
et al., 2021e ) built a pure transformer-based model named TokenPose to capture the constraint cues and visual appearance relationships by using token representation. In contrast to the methods based on the vision transformer which learn the representations in low resolution, Yuan et al. ( Yuan et al., 2021 ) presented a high-resolution transformer named HRFormer by exchanging the blocks in HRNet ( Cheng et al., 2020 ) with transformer modules, which improves the memory and computing efficiency.
Ma et al. ( Ma et al., 2022 ) applied the token-Pruned Pose Transformer (PPT) for locating the areas of the human body, which enables the model to estimate the multi-view pose efficiently. Different from the traditional two-step structures in HPE, Shi et al. ( Shi
et al., 2022 ) proposed a fully end-to-end framework based on the attention mechanism, which directly estimates the instance-aware body poses.

 
 
 Besides the image-based works introduced above, multi-frame pose estimation in videos is also popular in multi-person 2D HPE ( Guo
et al., 2018 ; Bertasius et al., 2019 ; Liu
et al., 2021a ; Xu et al., 2021 ; Liu
et al., 2022 ) , which leverages the temporal information in video sequences to facilitate the pose estimation. In order to reduce the cost of labeling frames in the video, Bertasius et al. ( Bertasius et al., 2019 ) proposed a network named PoseWarper, which improves the label propagation between frames and benefits the training with the sparse annotations. To alleviate the motion blur and pose occlusions among video frames, Liu et al. ( Liu
et al., 2021a ) designed a network named DCpose for multi-frame HPE, which contains three modules (Pose Temporal Merger, Pose Residual Fusion, and Pose Correction Network) to exploit the temporal information between frames for keypoint detection. Nevertheless, these methods failed to fully utilize the information from neighboring frames. To solve this issue, Liu et al. ( Liu
et al., 2022 ) introduced a hierarchical alignment framework for alleviating the aggregation of unaligned contexts between two frames.

 
 
 

#### 2.2.2. Bottom-up pipeline

 
 The bottom-up pipeline (e.g., ( Insafutdinov et al., 2017 ; Cao
et al., 2017 ; Newell
et al., 2017 ; Fieraru et al., 2018 ; Tian
et al., 2019 ; Kreiss
et al., 2019 ; Nie
et al., 2019 ; Jin et al., 2020a ; Cheng et al., 2020 ; Wang
et al., 2022a ; Wang
et al., 2022c ) ) has two main steps including body joint detection (i.e., extracting local features and predicting body joint candidates) and joint candidates assembling for individual bodies (i.e., grouping joint candidates to build pose representations with part association strategies) as shown in Fig. 3 (b).

 
 
 Pishchulin et al. ( Pishchulin et al., 2016 ) proposed a Fast R-CNN-based body part detector named DeepCut, which is one of the earliest two-stage bottom-up approaches. It first detects all the body part candidates, then labels each part and assembles these parts using integer linear programming (ILP) to a final pose. However, DeepCut is computationally expensive. To this end, Insafutdinov et al. ( Insafutdinov et al., 2016 ) introduced DeeperCut to improve DeepCut by applying a stronger body part detector with a better incremental optimization strategy and image-conditioned pairwise terms to group body parts, leading to improved performance as well as a faster speed. Later, Cao et al. ( Cao
et al., 2017 ) built a detector named OpenPose, which uses Convolutional Pose Machines ( Wei
et al., 2016 ) to predict keypoint coordinates via heatmaps and Part Affinity Fields (PAFs, a set of 2D vector fields with vector maps that encode the position and orientation of limbs) to associate the keypoints to each person. OpenPose largely accelerates the speed of bottom-up multi-person HPE. Based on the OpenPose framework, Zhu et al. ( Zhu
et al., 2017 ) improved the OpenPose structure by adding redundant edges to increase the connections between joints in PAFs and obtained better performance than the baseline approach. Although OpenPose-based methods have achieved impressive results on high-resolution images, they have poor performance with low-resolution images and occlusions. To address this problem, Kreiss et al. ( Kreiss
et al., 2019 ) proposed a bottom-up method called PifPaf that uses a Part Intensity Field to predict the locations of body parts and a Part Association Field to represent the joints association. This method outperformed previous OpenPose-based approaches on low-resolution and occluded scenes.
Motivated by OpenPose ( Cao
et al., 2017 ) and stacked hourglass structure ( Newell
et al., 2016 ) , Newell et al. ( Newell
et al., 2017 ) introduced a single-stage deep network to simultaneously obtain pose detection and group assignments. Following ( Newell
et al., 2017 ) , Jin et al. ( Jin et al., 2020a ) proposed a new differentiable Hierarchical Graph Grouping method to learn the human part grouping. Based on ( Newell
et al., 2017 ) and ( Sun
et al., 2019 ) , Cheng et al. ( Cheng et al., 2020 ) proposed an extension of HRNet, named Higher Resolution Network, which deconvolves the high-resolution heatmaps generated by HRNet to solve the scale variation challenge in bottom-up multi-person HPE.

 
 
 Multi-task structures are also employed in bottom-up HPE methods. Papandreou et al. ( Papandreou et al., 2018 ) introduced PersonLab to combine the pose estimation module and the person segmentation module for keypoints detection and association. PersonLab consists of short-range offsets (for refining heatmaps), mid-range offsets (for predicting the keypoints), and long-range offsets (for grouping keypoints into instances). Kocabas et al. ( Kocabas
et al., 2018 ) presented a multi-task learning model with a pose residual net, named MultiPoseNet, which can perform keypoint prediction, human detection, and semantic segmentation tasks altogether. However, these methods are struggling in dealing with the variance of human scales, to address this problem, Luo et al. ( Luo
et al., 2021 ) introduced a method named SAHR (scale-adaptive heatmap regression) to optimize the joint standard deviation adaptively, which improved the tolerance of various human scales and labeling ambiguities in an efficient way.

 
 
 
 

### 2.3. 2D HPE Summary

 
 In summary, the performance of 2D HPE has been significantly improved with the blooming of deep learning techniques. In recent years, deeper and more powerful networks have enhanced the performance of 2D single-person HPE methods such as DeepPose ( Toshev and
Szegedy, 2014 ) and Stacked Hourglass Network ( Newell
et al., 2016 ) , as well as in 2D multi-person HPE like AlphaPose ( Fang
et al., 2017 ) and OpenPose ( Cao
et al., 2017 ) .

 
 
 Although promising performance has been achieved, there are several challenges in 2D HPE that need to be further addressed in future research. First is the reliable detection of individuals under significant occlusion ( Chen
et al., 2018b ) , e.g., in crowd scenarios. Person detectors in top-down 2D HPE methods may fail to identify the boundaries of largely overlapped human bodies. Similarly, the difficulty of keypoint association is more pronounced for bottom-up approaches in occluded scenes.

 
 
 The second challenge is computation efficiency. Although some methods like OpenPose ( Cao
et al., 2017 ) can achieve near real-time processing on special hardware with moderate computing power (e.g., 22 FPS with an Nvidia GTX 1080 Ti GPU), it is still difficult to implement the networks on resource-constrained devices. Real-world applications
(e.g., gaming, AR, and VR)
require more efficient HPE methods on commercial devices which can bring better interactive experiences for users.

 
 
 Another challenge lies in the limited data for rare poses. Although the size of current datasets for 2D HPE is large enough (e.g., COCO dataset ( Lin et al., 2014 ) ) for normal pose estimation (e.g., standing, walking, running), these datasets have limited training data for unusual poses, e.g., falling. The data imbalance may cause model bias, resulting in poor performance on those poses. It would be useful to develop effective data generation or augmentation techniques to generate extra pose data for training more robust models.

 
 
 
 

## 3. 3D human pose estimation

 
 3D HPE, which aims to predict the locations of body joints in 3D space, has attracted much interest in recent years since it can provide extensive 3D structure information related to the human body.
It can be applied to various applications (e.g., 3D movie and animation industries, virtual reality, and sports analysis). Although significant improvements have been achieved in 2D HPE, 3D HPE still remains a challenging task.
Most existing works tackle 3D HPE from monocular images or videos, which is an ill-posed and inverse problem due to projection of 3D to 2D where one dimension is lost.
When multiple views are available or other sensors such as IMU and LiDAR are deployed, 3D HPE can be a well-posed problem employing information fusion techniques. Another limitation is that deep learning models are data-hungry and sensitive to the data collection environment. Unlike 2D HPE datasets where accurate 2D pose annotation can be easily obtained, collecting accurate 3D pose annotation is time-consuming and manual labeling is not practical. Also, datasets are usually collected from indoor environments with selected daily actions. Recent works ( Zhou
et al., 2017 ; Yang
et al., 2018 ; Wandt and
Rosenhahn, 2019 ) revealed the poor generalization of models trained with biased datasets by cross-dataset inference. In this section, we first focus on 3D HPE from monocular RGB images and videos and then cover 3D HPE based on other sensors.

 
 

### 3.1. 3D HPE from monocular RGB images and videos

 
 The monocular camera is the most widely used sensor for HPE in both 2D and 3D scenarios. Recent progress in deep learning-based 2D HPE from monocular images and videos has enabled researchers to extend their works to 3D HPE.

 
 
 The reconstruction of 3D human poses from a single view of monocular images and videos is a nontrivial task that suffers from self-occlusions and other object occlusions, depth ambiguities, and insufficient training data. It is a severely ill-posed problem because different 3D human poses can be projected to a similar 2D pose projection.

 
 
 The problem of occlusion can be alleviated by estimating 3D human pose from multi-view cameras. In a multi-view setting, the viewpoints association needs to be addressed. Thus deep learning-based 3D HPE methods are divided into three categories: single-view single-person 3D HPE, single-view multi-person 3D HPE, and multi-view 3D HPE.

 
 

#### 3.1.1. Single-view single person 3D HPE

 
 Single-person 3D HPE approaches can be classified into skeleton-only and human mesh recovery (HMR) categories based on whether to reconstruct 3D human skeleton or to recover 3D human mesh by employing a human body model.

 
 
 A. Skeleton-only. The skeleton-only methods estimate 3D human joints as the final output. They do not employ human body models to reconstruct 3D human mesh representation. These methods can be further divided into direct estimation approaches and 2D to 3D lifting approaches.

 
 
 Figure 4. Single-person 3D HPE frameworks. (a) Direct estimation approaches directly estimate the 3D human pose from 2D images. (b) 2D to 3D lifting approaches leverage the predicted 2D human pose (intermediate representation) for 3D pose estimation. (c) Human mesh recovery methods incorporate parametric body models to recover a high-quality 3D human mesh. The 3D pose and shape parameters inferred by the 3D pose and shape network are fed into the model regressor to reconstruct 3D human mesh. Part of the figure is from ( Arnab
et al., 2019 ) . 
 
 
 Direct estimation : As shown in Fig. 4 (a), direct estimation methods infer the 3D human pose from 2D images without intermediately estimating 2D pose representation, e.g., ( Li et al., 2015 ; Tekin et al., 2016a ; Sun
et al., 2017 ; Pavlakos et al., 2017a ; Pavlakos
et al., 2018a ) . Li and Chan ( Li and Chan, 2014 ) employed a shallow network to train the body part detector with sliding windows and the pose coordinate regression synchronously.
Sun et al. ( Sun
et al., 2017 ) proposed a structure-aware regression approach. Instead of using a joint-based representation, they adopted a bone-based representation with more stability. A compositional loss was defined by exploiting the 3D bone structure with bone-based representation that encodes long-range interactions between the bones.
Pavlakos et al. ( Pavlakos et al., 2017a ; Pavlakos
et al., 2018a ) introduced a volumetric representation to convert the highly non-linear 3D coordinate regression problem to a manageable form in a discretized space. The voxel likelihoods for each joint in the volume were predicted by a convolutional network. Ordinal depth relations of human joints were used to alleviate the need for accurate 3D ground truth poses.

 
 
 2D to 3D lifting : Motivated by the recent success of 2D HPE, 2D to 3D lifting approaches that infer 3D human pose from the intermediately estimated 2D human pose have become a popular 3D HPE solution as illustrated in Fig. 4 (b). In the first stage, off-the-shelf 2D HPE models are employed to estimate 2D pose. Then 2D to 3D lifting is used to obtain 3D pose in the second stage, e.g., ( Chen and Ramanan, 2017 ; Martinez et al., 2017 ; Tekin et al., 2017 ; Zhou et al., 2019 ; Moreno-Noguer, 2017 ; Li and Lee, 2019 ) . Benefiting from the excellent performance of state-of-the-art 2D pose detectors, 2D to 3D lifting approaches generally outperform direct estimation approaches.
Martinez et al. ( Martinez et al., 2017 ) proposed a fully connected residual network to regress 3D joint locations based on the 2D joint locations. Despite achieving state-of-the-art results at that time, the method could fail due to reconstruction ambiguity of over-reliance on the 2D pose detector. Tekin et al. ( Tekin et al., 2017 ) and Zhou et al. ( Zhou et al., 2019 ) adopted 2D heatmaps instead of 2D pose as intermediate representations for estimating 3D pose.
Wang et al. ( Wang
et al., 2018 ) developed a pairwise ranking CNN to predict the depth ranking of pairwise human joints. Then, a coarse-to-fine pose estimator was used to regress the 3D pose from 2D joints and the depth ranking matrix. Jahangiri and Yuille ( Jahangiri and
Yuille, 2017 ) , Sharma et al. ( Sharma et al., 2019 ) , and Li and Lee ( Li and Lee, 2019 ) first generated multiple diverse 3D pose hypotheses then applied ranking networks to select the best 3D pose.

 
 
 Given that a human pose can be represented as a graph where the joints are the nodes and the bones are the edges, Graph Convolutional Networks (GCNs) have been applied to the 2D-to-3D pose lifting problem by showing promising performance ( Ci
et al., 2019 ; Zhao
et al., 2019b ; Choi
et al., 2020 ; Liu
et al., 2020a ; Zeng
et al., 2020 ) .
Ci et al. ( Ci
et al., 2019 ) proposed a Locally Connected Network (LCN), which leverages both a fully connected network and GCN to encode the relationship between local joint neighborhoods. LCN can overcome the limitations of GCN that the weight-sharing scheme harms the pose estimation model’s representation ability, and the structure matrix lacks the flexibility to support customized node dependence. Zhao et al. ( Zhao
et al., 2019b ) also tackled the limitation of the shared weight matrix of convolution filters for all the nodes in GCN. A Semantic-GCN was proposed to investigate the semantic information and relationship, which is not explicitly represented in the graph. The semantic graph convolution (SemGConv) operation is used to learn channel-wise weights for edges. Both local and global relationships among nodes are captured since SemGConv and non-local layers are interleaved. Zhou et al. ( Zou and Tang, 2021 ) further introduced a novel modulated GCN network which consists of weight modulation and affinity modulation. The weight modulation exploits different modulation vectors for different nodes that disentangles the feature transformations. The affinity modulation explores additional joint correlations beyond the defined human skeleton.

 
 
 The kinematic model is an articulated body representation by connected bones and joints with kinematic constraints, which has gained increasing attention in 3D HPE in recent years. Many methods leverage prior knowledge based on the kinematic model such as skeletal joint connectivity information, joint rotation properties, and fixed bone-length ratios for plausible pose estimation, e.g., ( Zhou
et al., 2016a ; Mehta et al., 2017 ; Nie
et al., 2017 ; Wang
et al., 2019a ; Kundu et al., 2020c ; Xu
et al., 2020c ; Nie et al., 2020 ; Georgakis et al., 2020 ) . Zhou et al. ( Zhou
et al., 2016a ) embedded a kinematic model into a network as kinematic layers to enforce the orientation and rotation constraints. Nie et al. ( Nie
et al., 2017 ) and Lee et al. ( Lee et al., 2018 ) employed a skeleton-LSTM network to leverage joint relations and connectivity. Observing that human body parts have a distinct degree of freedom (DOF) based on the kinematic structure, Wang et al. ( Wang
et al., 2019a ) and Nie et al. ( Nie et al., 2020 ) proposed bidirectional networks to model the kinematic and geometric dependencies of the human skeleton. Kundu et al. ( Kundu et al., 2020c ) ( Kundu et al., 2020b ) designed a kinematic structure preservation approach by inferring local-kinematic parameters with energy-based loss and explored 2D part segments based on the parent-relative local limb kinematic model. Xu et al. ( Xu
et al., 2020c ) demonstrated that noise in the 2D joint is one of the key obstacles for accurate 3D pose estimation. Hence a 2D pose correction module was employed to refine unreliable 2D joints based on the kinematic structure. Zanfir et al. ( Zanfir et al., 2020 ) introduced a kinematic latent normalizing flow representation (a sequence of invertible transformations applied to the original distribution) with differentiable semantic body part alignment loss functions.

 
 
 3D HPE datasets are usually collected from controlled environments with selected common motions. It is difficult to obtain accurate 3D pose annotations for in-the-wild data. Thus 3D HPE for in-the-wild data with unusual poses and occlusions is still a challenge.
To this end, a group of 2D to 3D lifting methods estimate the 3D human pose from in-the-wild images without 3D pose annotations such as ( Zhou
et al., 2017 ; Habibie et al., 2019 ; Chen et al., 2019 ; Yang
et al., 2018 ; Wandt and
Rosenhahn, 2019 ) . Zhou et al. ( Zhou
et al., 2017 ) proposed a weakly supervised transfer learning method that uses 2D annotations of in-the-wild images as weak labels. A 3D pose estimation module was connected with intermediate layers of the 2D pose estimation module. For in-the-wild images, 2D pose estimation module performed a supervised 2D heatmap regression and a 3D bone length constraint-induced loss was applied in the weakly supervised 3D pose estimation module. Habibie et al. ( Habibie et al., 2019 ) tailored a projection loss to refine the 3D human pose without 3D annotation. A 3D-2D projection module was designed to estimate the 2D body joint locations with the predicted 3D pose from the earlier network layer. The projection loss was used to update the 3D human pose without requiring 3D annotations. Inspired by ( Drover et al., 2018 ) , Chen et al. ( Chen et al., 2019 ) proposed an unsupervised lifting network based on the closure and invariance lifting properties with a geometric self-consistency loss for the lift-reproject-lift process. Closure means for a lifted 3D skeleton, after random rotation and re-projection, the resulting 2D skeleton will lie within the distribution of valid 2D poses. Invariance means when changing the viewpoint of 2D projection from a 3D skeleton, the re-lifted 3D skeleton should be the same.

 
 
 Instead of estimating 3D human pose from monocular images, videos can provide temporal information to improve accuracy and robustness of 3D HPE, e.g., ( Zhou
et al., 2016b ; Zhou et al., 2018 ; Dabral et al., 2018 ; Pavllo et al., 2019 ; Cheng et al., 2019 ; Cai et al., 2019 ; Wang
et al., 2020d ; Tekin
et al., 2016b ) .
Hossain and Little ( Rayat
Imtiaz Hossain and Little, 2018 ) proposed a recurrent neural network using a Long Short-Term Memory (LSTM) unit with shortcut connections to exploit temporal information from sequences of human pose. Their method exploits the past events in a sequence-to-sequence network to predict temporally consistent 3D pose. Noticing that the complementary property between spatial constraints and temporal correlations is usually ignored by prior work, Dabral et al. ( Dabral et al., 2018 ) , Cai et al. ( Cai et al., 2019 ) , and Li et al. ( Li
et al., 2019 ) exploited the spatial-temporal relationships and constraints (e.g., bone-length constraint and left-right symmetry constraint) to improve 3D HPE performance from sequential frames. Pavllo et al. ( Pavllo et al., 2019 ) proposed a temporal convolution network to estimate 3D pose over 2D keypoints from consecutive 2D sequences. However, their method is based on the assumption that prediction errors are temporally non-continuous and independent, which may not hold in the presence of occlusions ( Cheng et al., 2019 ) . Based on ( Pavllo et al., 2019 ) , Chen et al. ( Chen
et al., 2021 ) added a bone direction module and bone length module to ensure human anatomy temporal consistency across video frames, while Liu et al. ( Liu
et al., 2020c ) utilized the attention mechanism to recognize significant frames and model long-range dependencies in large temporal receptive fields. Zeng et al. ( Zeng
et al., 2020 ) employed the split-and-recombine strategy to address the rare and unseen pose problem. The human body is first split into local regions for processing through separate temporal convolutional network branches, then the low-dimensional global context obtained from each branch is combined for maintaining global coherence.

 
 
 Transformer architectures have become the model of choice in natural language processing due to the self-attention mechanism, and now are developing rapidly in the field of computer vision. Recent works have demonstrated the powerful global representation ability of transformer attention mechanism in various vision tasks ( Khan et al., 2021 ) . Zheng et al. ( Zheng et al., 2021 ) presented the first purely transformer-based approach, named PoseFormer, for 3D HPE without convolutional architectures involved. The spatial transformer module encodes local relationships between human body joints, and the temporal transformer module captures the global dependencies across frames in the entire sequence.
Li et al. ( Li
et al., 2022a ) further designed a multi-hypothesis transformer to exploit spatial-temporal representations of multiple pose hypotheses.
Zhao et al. ( Zhao
et al., 2023 ) further proposed PoseFormerV2, which exploits a compact representation of lengthy skeleton sequences in the
frequency domain to efficiently scale up the receptive field and boost robustness to noisy 2D joint detection.

 
 
 B. Human Mesh Recovery (HMR). HMR methods incorporate parametric body models such as SMPL ( Loper et al., 2015 ) to recovery human mesh as shown in Fig. 4 (c). The SMPL (Skinned Multi-Person Linear) model ( Loper et al., 2015 ) is a widely used model in 3D HPE, which can be modeled with natural pose-dependent deformations exhibiting soft-tissue dynamics. To learn how people deform with poses, there are 1786 high-resolution 3D scans of different subjects of poses with template mesh in SMPL to optimize the blend weights ( Kavan, 2014 ) , pose-dependent blend shapes, the mean template shape, and the regressor from vertices to joint locations. The 3D pose can be obtained by using the model-defined joint regression matrix ( Kolotouros et al., 2019 ) .
There are also other popular volumetric models such as DYNA ( Pons-Moll et al., 2015 ) , Stitched Puppet model ( Zuffi and Black, 2015 ) , Frankenstein Adam ( Joo
et al., 2018 ) , and GHUM GHUML(ite) ( Xu et al., 2020a ) .

 
 
 Volumetric models are used to recover high-quality human mesh , providing extra shape information of the human body. As one of the most popular volumetric models, the SMPL model ( Loper et al., 2015 ) has been widely used in 3D HPE, e.g.,
 ( Bogo et al., 2016 ; Kolotouros et al., 2019 ; Zhu
et al., 2019 ; Arnab
et al., 2019 ; Moon and Lee, 2020 ; Zhang
et al., 2020c ; Li
et al., 2021d ; Zhang et al., 2021 ) , because it is compatible with existing rendering engines. Tan et al. ( Vince Tan and
Cipolla, 2017 ) , Tung et al. ( Tung
et al., 2017 ) , Pavlakos et al. ( Pavlakos
et al., 2018b ) , and Omran et al. ( Omran et al., 2018 ) regressed SMPL parameters to reconstruct 3D human mesh. Instead of predicting SMPL parameters, Kolotouros et al. ( Kolotouros et al., 2019 ) regressed the locations of the SMPL mesh vertices using a Graph-CNN architecture.
Kocabas et al. ( Kocabas
et al., 2020 ) included the large-scale motion capture dataset AMASS ( Mahmood et al., 2019 ) for adversarial training of their SMPL-based method named VIBE (Video Inference for Body Pose and Shape Estimation). VIBE leveraged AMASS to discriminate between real human motions and those predicted by the pose regression module. Since low-resolution visual content is more common in real-world scenarios than high-resolution visual content, existing well-trained models may fail when the resolution is degraded. Xu et al. ( Xu et al., 2020b ) introduced the contrastive learning scheme into a self-supervised resolution-aware SMPL-based network. The self-supervised contrastive learning scheme uses a self-supervision loss and a contrastive feature loss to enforce feature and scale consistency. Choi et al. ( Choi
et al., 2021 ) presented a temporally consistent mesh recovery system (named TCMR) to smooth 3D human motion output using a bi-directional gated recurrent unit. Kolotouros et al. ( Kolotouros et al., 2021 ) proposed a probabilistic model using conditional normalizing flow for 3D human mesh recovery from 2D evidence. Zheng et al. ( Zheng
et al., 2022 ) designed a lightweight transformer-based method that can reconstruct human mesh from 2D human pose with a significant computation and memory cost reduction, while the performance is competitive with Pose2Mesh ( Choi
et al., 2020 ) .

 
 
 There are a few recent attempts to utilize transformer in HMR ( Lin
et al., 2021a ; Zheng
et al., 2022 ; Zheng
et al., 2023a ; Zheng
et al., 2023b ) . Lin et al. proposed METRO ( Lin
et al., 2021a ) and MeshGraphormer ( Lin
et al., 2021b ) that combine CNNs with transformer networks to regress SMPL mesh vertices from a single image. However, they pursued higher accuracy while sacrificing computation and memory.
FeatER ( Zheng
et al., 2023b ) and POTTER ( Zheng
et al., 2023a ) reduced the computational and memory cost by proposing lightweight transformer architectures, while both of them outperform METRO by only requiring less than 10%of total parameters and 15% MACs.

 
 
 There are several extended SMPL-based models to address the limitations of the SMPL model such as high computational complexity, and lack of hands and facial landmarks. SMPLify ( Lassner et al., 2017 ) is an optimization method that fits the SMPL model to the detected 2D joints and minimizes the re-projection error. Pavlakos et al. ( Pavlakos et al., 2019 ) introduced a new model, named SMPL-X, that can also predict fully articulated hands and facial landmarks. Following the SMPLify method, they also proposed SMPLify-X, which is an improved version learned from AMASS dataset ( Mahmood et al., 2019 ) . Hassan et al. ( Hassan
et al., 2019 ) further extended SMPLify-X to PROX – a method enforcing Proximal Relationships with Object eXclusion by adding 3D environmental constraints. Kolotouros et al. ( Kolotouros et al., 2019 ) integrated the regression-based and optimization-based SMPL parameter estimation methods to a new one named SPIN (SMPL oPtimization IN the loop) while employing SMPLify in the training loop. The estimated 2D pose served as the initialization in an iterative optimization routine for producing a more accurate 3D pose and shape.

 
 
 Instead of using the SMPL-based model, other models have also been used for recovering 3D human pose or mesh, e.g., ( Qammaz and
Argyros, 2019 ; Saito
et al., 2020 ; Wang et al., 2020a ) . Chen et al. ( Cheng
et al., 2019 ) introduced a Cylinder Man Model to generate occlusion labels for 3D data and performed data augmentation. A pose regularization term was introduced to penalize wrong estimated occlusion labels. Xiang et al. ( Xiang
et al., 2019 ) utilized the Adam model ( Joo
et al., 2018 ) to reconstruct the 3D motions. A 3D human representation, named 3D Part Orientation Fields (POFs), was introduced to encode the 3D orientation of human body parts in the 2D space.
Wang et al. ( Wang et al., 2020a ) presented a new Bone-level Skinned Model of human mesh, which decouples bone modeling and identity-specific variations by setting bone lengths and joint angles. Fisch and Clark ( Fisch and Clark, 2020 ) introduced an orientation keypoints model which can compute full 3-axis joint rotations including yaw, pitch, and roll for 6D HPE. 
 

 
 
 

#### 3.1.2. Single-view multi-person 3D HPE

 
 For 3D multi-person HPE from monocular RGB images or videos, similar categories as 2D multi-person HPE are noted here: top-down approaches and bottom-up approaches as shown in Fig. 5 (a) and Fig. 5 (b), respectively. The comparison between 2D top-down and bottom-up approaches in Section 2.2 is applicable to the 3D case.

 
 
 Figure 5. Illustration of the multi-person 3D HPE frameworks. (a) Top-Down methods first detect single-person regions by human detection network. For each single-person region, individual 3D poses can be estimated by 3D pose network. Then all 3D poses are aligned to the world coordinate. (b) Bottom-Up methods first estimate all body joints and depth maps, then associate body parts to each person according to the root depth and part relative depth. Part of the figure is from ( Zhen et al., 2020 ) . 
 
 
 Top-down approaches. 
Top-down approaches of 3D multi-person HPE first perform human detection to detect each individual person. Then for each detected person, the absolute root (center joint of the human) coordinate and 3D root-relative pose are estimated by 3D pose networks. Based on the absolute root coordinate of each person and their root-relative pose, all poses are aligned to the world coordinate. Rogez et al. ( Rogez
et al., 2017 ) localized candidate regions of each person to generate potential poses, and used a regressor to jointly refine the pose proposals. This localization-classification-regression method, named LCR-Net, performed well on the controlled environment datasets but could not generalize well to in-the-wild images. To address this issue, Rogez et al. ( Rogez
et al., 2019 ) proposed LCR-Net++ by using synthetic data augmentation for the training data to improve performance. Zanfir et al. ( Zanfir et al., 2018 ) added semantic segmentation to the 3D multi-person HPE module with scene constraints. Additionally, the 3D temporal assignment problem was tackled by the Hungarian matching method for video-based multi-person 3D HPE. Moon et al. ( Moon
et al., 2019a ) introduced a camera distance-aware approach under top-down pipeline. The RootNet estimated the camera-centered coordinates of human body’s roots.
Then the root-relative 3D pose of each cropped human was estimated by the proposed PoseNet. Benzine et al. ( Benzine et al., 2020 ) proposed a single-shot approach named PandaNet (Pose estimAtioN and Detection Anchor-based Network). A low-resolution anchor-based representation was introduced to avoid the occlusion problem. A pose-aware anchor selection module was developed to address the overlapping problem by removing ambiguous anchors. An automatic weighting of losses associated with different scales was used to handle the imbalance issue of different sizes of people. Li et al. ( Li
et al., 2020b ) tackled the lack of global information in the top-down approaches. They adopted a Hierarchical Multi-person Ordinal Relations method to leverage body-level semantic and global consistency for encoding the interaction information hierarchically.

 
 
 Bottom-up approaches. In contrast to top-down approaches, bottom-up approaches first produce all body joint locations and depth maps, then associate body parts to each person according to the root depth and part relative depth. A key challenge of bottom-up approaches is how to group human body joints belonging to each person. Zanfir et al. ( Zanfir et al., 2018 ) formulated the person grouping problem as a binary integer programming (BIP) problem. A limb scoring module was used to estimate candidate kinematic connections of detected joints and a skeleton grouping module assembled limbs into skeletons by solving the BIP problem. Nie et al. ( Nie
et al., 2019 ) proposed a Single-stage multi-person Pose Machine (SPM) to define the unique identity root joint for each person. The body joints were aligned to each root joint by using dense displacement maps. However, this method is limited in that only paired 2D images and 3D pose annotations can be used for supervised learning. Without paired 2D images and 3D pose annotations, Kundu et al. ( Kundu et al., 2020a ) proposed a frozen network to exploit the shared latent space between two diverse modalities under a practical deployment paradigm such that the learning could be cast as a cross-model alignment problem. Fabbri et al. ( Fabbri et al., 2020 ) developed a distance-based heuristic for linking joints in a multi-person setting. Specifically, starting from detected
heads (i.e., the joint with the highest confidence), the remaining joints are connected by selecting the closest ones in terms of 3D Euclidean distance. Chen et al. ( Cheng
et al., 2021b ) integrated top-down and bottom-up approaches in their method. A top-down network first estimates joint heatmaps inside each bounding box, then a bottom-up network incorporates estimated joint heatmaps to handle the scale variation.

 
 
 Another challenge of bottom-up approaches is occlusion. To cope with this challenge, Metha et al. ( Mehta et al., 2018 ) developed an Occlusion-Robust Pose-Maps (ORPM) approach to incorporate redundancy into the location-maps formulation, which facilitates person association in the heatmaps, especially for occluded scenes. Zhen et al. ( Zhen et al., 2020 ) leveraged a depth-aware part association algorithm to assign joints to individuals by reasoning about inter-person occlusion and bone-length constraints. Mehta et al. ( Mehta et al., 2020 ) quickly inferred intermediate 3D pose of visible body joints regardless of the accuracy. Then the completed 3D pose is reconstructed by inferring occluded joints using learned pose priors and global context. The final 3D pose was refined by applying temporal coherence and fitting the kinematic skeletal model.

 
 
 Comparison of top-down and bottom-up approaches. 
Top-down approaches usually achieve promising results by relying on state-of-the-art person detection methods and single-person HPE methods. But the computational complexity and the inference time may become excessive with the increase in the number of humans, especially in crowded scenes. Moreover, since top-down approaches first detect the bounding box for each person, global information in the scene may get neglected. The estimated depth of cropped region may be inconsistent with the actual depth ordering and the predicted human bodies may be placed in overlapping positions. On the contrary, the bottom-up approaches enjoy linear computation and time complexity. However, if the goal is to recover 3D body mesh, it is not straightforward for the bottom-up approaches to reconstruct human body meshes. For top-down approaches, after detecting each individual person, human body mesh of each person can be easily recovered by incorporating the 3D single-person human mesh recovery method. While for the bottom-up approaches, an additional model regressor module is needed to reconstruct human body meshes based on the final 3D poses.

 
 
 

#### 3.1.3. Multi-view 3D HPE

 
 Partial occlusion is a challenging problem for 3D HPE in the single-view setting. The natural solution to overcome this problem is to estimate 3D human pose from multiple views, since the occluded part in one view may become visible in other views. In order to reconstruct the 3D pose from multiple views, the association of corresponding locations between different cameras needs to be resolved. We do not specify single-person or multi-person in this category since the multi-view setting is deployed mainly for multi-person pose estimation.

 
 
 A group of methods ( Qiu
et al., 2019 ; Liang and Lin, 2019 ; Dong
et al., 2019 ; Tu et al., 2020 ; Dong
et al., 2021 ) used body models to tackle the association problem by optimizing model parameters to match the model projection with the 2D pose. The widely used 3D pictorial structure model ( Burenius et al., 2013 ) is such a model. However, these methods usually need large memory and expensive computational cost, especially for multi-person 3D HPE under multi-view settings. Rhodin et al. ( Rhodin et al., 2018b ) employed a multi-view consistency constraint in the network, however it requires a large amount of 3D ground-truth training data. To overcome this limitation, Rhodin et al. ( Rhodin
et al., 2018a ) further proposed an encoder-decoder framework to learn the geometry-aware 3D latent representation from multi-view images and background segmentation without 3D annotations.
Chen et al. ( Chen
et al., 2020b ) , Mitra et al. ( Mitra
et al., 2020 ) , Zhang et al. ( Zhang
et al., 2020a ) , and Huang et al. ( Huang et al., 2020b ) 
proposed multi-view matching frameworks to reconstruct 3D human pose across all viewpoints with consistency constraints. Pavlakos et al. ( Pavlakos et al., 2017b ) and Zhang et al. ( Zhang
et al., 2020g ) aggregated the 2D keypoint heatmaps of multi-view images into a 3D pictorial structure model based on all the calibrated camera parameters. However, when multi-view camera environments change, the model needs to be retrained.
Qiu et al. ( Qiu
et al., 2019 ) , and Kocabas et al. ( Kocabas
et al., 2019 ) employed epipolar geometry to match paired multi-view poses for 3D pose reconstruction and generalized their methods to new multi-view camera environments. It should be noted that matching each pair of views separately without the cycle consistency constraint may lead to incorrect 3D pose reconstructions ( Dong
et al., 2019 ) . Tu et al. ( Tu et al., 2020 ) aggregated all the features in each camera view in the 3D voxel space to avoid incorrect estimation in each camera view. A cuboid proposal network and a pose regression network were designed to localize all people and to estimate the 3D pose, respectively. When given sufficient viewpoints (more than ten), it is not practical to use all viewpoints for 3D pose estimation. Pirinen et al. ( Pirinen et al., 2019 ) proposed a self-supervised reinforcement learning approach to select a small set of viewpoints to reconstruct the 3D pose via triangulation. Wang et al. ( Wang
et al., 2021 ) introduced a transformer-based model that directly regresses 3D poses from multi-view images without relying on any intermediate task. The proposed Multi-view Pose transformer (MvP) was designed to represent query embedding of multi-person joints. The multi-view information was fused by a novel geometrically guided attention mechanism.

 
 
 Besides accuracy, lightweight architecture, fast inference time, and efficient adaptation to new camera settings also need to be taken into consideration in multi-view HPE. In contrast to ( Dong
et al., 2019 ) which matched all view inputs together, Chen et al. ( Chen
et al., 2020a ) applied an iterative processing strategy to match 2D poses of each view with the 3D pose while the 3D pose was updated iteratively. Compared to previous methods whose running time may explode with the increase in the number of cameras, their method has linear time complexity. Remelli et al. ( Remelli
et al., 2020 ) encoded images of each view into a unified latent representation so that the feature maps were disentangled from camera viewpoints. As a lightweight canonical fusion, these 2D representations are lifted to the 3D pose using a GPU-based Direct Linear Transform to accelerate processing. In order to improve the generalization ability of the multi-view fusion scheme, Xie et al. ( Xie
et al., 2020 ) proposed a pre-trained multi-view fusion model (MetaFuse), which can be efficiently adapted to new camera settings with few labeled data. They deployed the model-agnostic meta-learning framework to learn the optimal initialization of the generic fusion model for adaptation. To reduce the computational cost of the VoxelPose ( Tu et al., 2020 ) , Ye et al. presented Faster VoxelPose ( Ye
et al., 2022 ) which re-projects the feature volume to three two-dimensional coordinate planes for estimating X, Y, Z coordinates from them separately. The fps of Faster VoxelPose is 31.1, which is almost 10 × \times speed up compared to VoxelPose.

 
 
 
 

### 3.2. 3D HPE from other sources

 
 Although a monocular RGB camera is the most common device used for 3D HPE, other sensors (e.g., depth sensor, IMUs, and radio frequency device) are also used for this purpose.

 
 
 Depth and point cloud sensors : Depth sensors have gained more attention recently for 3D computer vision tasks due to their low-cost and increased utilization. As one of the key challenges in 3D HPE, the depth ambiguity problem can be alleviated by using depth sensors. Yu et al. ( Yu et al., 2019 ) , Xiong et al. ( Xiong et al., 2019 ) , Kadkhodamohammadi et al. ( Kadkhodamohammadi et al., 2017 ) , and Zhi et al. ( Zhi et al., 2020 ) utilized depth images to estimate 3D human pose.

 
 
 Compared with depth images, point clouds can provide more information. The state-of-the-art point cloud feature extraction techniques, PointNet ( Qi
et al., 2017a ) and PointNet++ ( Qi
et al., 2017b ) , have demonstrated excellent performance for classification and segmentation tasks. Jiang et al. ( Jiang
et al., 2019 ) and Wang et al. ( Wang
et al., 2020c ) combined PointNet++ with the 3D human body model to recover 3D human mesh.

 
 
 IMUs with monocular images : Wearable Inertial Measurement Units (IMUs) can track the orientation and acceleration of human body parts by recording motions without object occlusions and clothes obstructions. Marcard et al. ( Von Marcard et al., 2017 ; von Marcard et al., 2018 ) , Huang et al. ( Huang et al., 2018 ) , Zhang et al. ( Zhang
et al., 2020f ) , and Huang et al. ( Huang
et al., 2020c ) proposed IMU-based HPE methods to reconstruct 3D human pose. However, drifting may occur over time when using IMUs.

 
 
 Radio frequency device : Radio Frequency (RF) based sensing technology has also been used for 3D HPE, e.g., ( Zhao et al., 2018 ) and ( Zhao et al., 2019a ) . The ability to traverse walls and bounce off human bodies in the WiFi range without carrying wireless transmitters is the major advantage of deploying an RF-based sensing system. Also, privacy can be preserved due to non-visual data. However, RF signals have a relatively low spatial resolution compared to visual camera images, and RF systems have been shown to generate coarse 3D pose estimation.

 
 
 Other sensors/sources: Besides using the aforementioned sensors, Isogawa et al. ( Isogawa
et al., 2020 ) estimated 3D human pose from the 3D spatio-temporal histogram of photons captured by a non-line-of-sight (NLOS) imaging system. Some methods ( Tome
et al., 2019 ; Tome et al., 2020 ; Xu et al., 2019 ) tackled the egocentric 3D pose estimation via a fish-eye camera. Saini et al. ( Saini et al., 2019 ) estimated human motion using images captured by multiple Autonomous Micro Aerial Vehicles (MAVs). Clever et al. ( Clever et al., 2020 ) focused on the HPE of the rest position in bed from pressure images which were collected by a pressure sensing mat.

 
 
 

### 3.3. 3D HPE Summary

 
 3D HPE has made significant advancements in recent years.
Since a large number of 3D HPE methods apply the 2D to 3D lifting strategy,
the performance of 3D HPE has been improved considerably due to the progress made in 2D HPE. Some 2D HPE methods such as OpenPose ( Cao
et al., 2017 ) , AlphaPose ( Fang
et al., 2017 ) , and HRNet ( Sun
et al., 2019 ) have been extensively used as 2D pose detectors in 3D HPE methods. Besides the 3D pose, some methods also recover 3D human mesh from images or videos, e.g., ( Kolotouros et al., 2019 ; Kocabas
et al., 2020 ; Zeng et al., 2020 ; Zhou
et al., 2020 ) . However, despite the progress made so far, there are still several challenges.

 
 
 One challenge is model generalization. High-quality 3D ground truth pose annotations
depend on motion capture systems that cannot be easily deployed in a random environment.
Therefore, the existing datasets are mainly captured in constrained scenes.
The state-of-the-art methods can achieve promising results on these datasets, but their performance degrades when applied to in-the-wild data.
It is possible to leverage gaming engines to generate synthetic datasets with diverse poses and complex scenes, e.g., SURREAL dataset ( Varol et al., 2017 ) and GTA-IM dataset ( Cao
et al., 2020 ) . However, learning from synthetic data may not achieve the
desired performance due to a gap between synthetic and real data distributions.

 
 
 Like 2D HPE, robustness to occlusion and computational efficiency are two key challenges for 3D HPE as well.
The performance of current 3D HPE methods drops considerably in crowded scenarios due to severe mutual occlusion and possibly low-resolution content of each person. 3D HPE is more computation-demanding than 2D HPE. For example, 2D to 3D lifting approaches rely on 2D poses as intermediate representations for inferring 3D poses. It is critical to develop computationally efficient 2D HPE pipelines while maintaining high accuracy for pose estimation.

 
 
 
 

## 4. Datasets and Evaluation Metrics

 
 Datasets are very much needed in conducting HPE. They are also necessary to provide a fair comparison among different algorithms.
In this section, we present the most widely used datasets and evaluation metrics for evaluating 2D and 3D deep learning-based HPE methods.
The results achieved by existing methods on these popular datasets are summarized.

 
 

### 4.1. Datasets for 2D HPE

 
 Although there are several datasets used for 2D HPE tasks before 2014, only a few recent works use these datasets because they have several shortcomings such as a lack of diverse object movements and limited data. Since deep learning-based approaches are fueled by a large amount of training data, this section mainly discusses the recent and large-scale 2D human pose datasets.

 
 
 Max Planck Institute for Informatics (MPII) Human Pose Dataset ( Andriluka et al., 2014 ) is a popular dataset for evaluation of articulated HPE
which includes around 25,000 images containing over 40,000 individuals with annotated body joints.
Rich annotations including body part occlusions, 3D torso, and head orientations are labeled by workers on Amazon Mechanical Turk. Images in MPII are suitable for 2D single-person or multi-person HPE.

 
 
 Microsoft Common Objects in Context (COCO) Dataset ( Lin et al., 2014 ) is the most widely used large-scale dataset. It has more than 330,000 images and 200,000 labeled subjects with keypoints, and each individual person is labeled with 17 joints.
In addition, Jin et al. ( Jin et al., 2020b ) proposed COCO-WholeBody Dataset with whole-body annotations for HPE.

 
 
 PoseTrack Dataset ( Andriluka et al., 2018a ) is a large-scale dataset for HPE and articulated tracking in videos, including body part occlusion and truncation in crowded environments. There are two versions for PoseTrack dataset: PoseTrack2017 ( Andriluka et al., 2018a ) contains 514 video sequences with 16,219 pose annotations, which are split into 250 (training), 50 (validation), and 214 (testing) sequences. PoseTrack2018 ( Andriluka et al., 2018b ) has 1,138 video sequences with 153,615 pose annotations, which are divided into 593 for training, 170 for validation, and 375 for testing. Each person in PoseTrack is labeled with 15 joints and an additional label for keypoint visibility.

 
 
 We refer the readers to the original references for details about other datasets including LSP ( Johnson and
Everingham, 2010 ) , FLIC ( Sapp and Taskar, 2013 ) , AIC-HKD ( Wu et al., 2017 ) , CrowdPose ( Li
et al., 2019b ) , Penn Action ( Zhang
et al., 2013 ) , J-HMDB ( Jhuang
et al., 2013 ) , and HiEve ( Lin et al., 2020 ) . A summary of these datasets is presented in Table 1 .

 
 
 Table 1. Datasets for 2D HPE. 
 
 
 
 
     Image-based datasets | 

 
 
 
     Name | 
     Year | 
     
 
 
    Single-Person 
 
    /Multi-Person 
 | 
     Joints | 
    Number of images | 
     
 
 
    Evaluation 
 
    protocol 
 | 

 
    Train | 
    Val | 
    Test | 

 
    LSP ( Johnson and
Everingham, 2010 ) | 
    2010 | 
    Single | 
    14 | 
    1k | 
    - | 
    1k | 
    PCP | 

 
    LSP-extended ( Johnson and
Everingham, 2011 ) | 
    2011 | 
    Single | 
    14 | 
    10k | 
    - | 
    - | 
    PCP | 

 
    FLIC ( Sapp and Taskar, 2013 ) | 
    2013 | 
    Single | 
    10 | 
    5k | 
    - | 
    1k | 
    PCP PCK | 

 
    FLIC-full ( Sapp and Taskar, 2013 ) | 
    2013 | 
    Single | 
    10 | 
    20k | 
    - | 
    - | 
    PCP PCK | 

 
    FLIC-plus ( Tompson
et al., 2014 ) | 
    2014 | 
    Single | 
    10 | 
    17k | 
    - | 
    - | 
    PCP PCK | 

 
     MPII ( Andriluka et al., 2014 ) | 
     2014 | 
    Single | 
    16 | 
    29k | 
    - | 
    12k | 
    PCPm/PCKh | 

 
    Multiple | 
    16 | 
    3.8k | 
    - | 
    1.7k | 
    mAp | 

 
    COCO2016 ( Lin et al., 2014 ) | 
    2016 | 
    Multiple | 
    17 | 
    45k | 
    22k | 
    80k | 
    AP | 

 
    COCO2017 ( Lin et al., 2014 ) | 
    2017 | 
    Multiple | 
    17 | 
    64k | 
    2.7k | 
    40k | 
    AP | 

 
    AIC-HKD ( Wu et al., 2017 ) | 
    2017 | 
    Multiple | 
    14 | 
    210k | 
    30k | 
    60k | 
    AP | 

 
    CrowdPose ( Li
et al., 2019b ) | 
    2019 | 
    Multiple | 
    14 | 
    10k | 
    2k | 
    8k | 
    mAP | 

 
     Video-based datasets | 

 
     Name | 
     Year | 
     
 
 
    Single-Person 
 
    /Multi-Person 
 | 
     Joints | 
    Number of videos | 
     
 
 
    Evaluation 
 
    protocol 
 | 

 
    Train | 
    Val | 
    Test | 

 
    Penn Action ( Zhang
et al., 2013 ) | 
    2013 | 
    Single | 
    13 | 
    1k | 
    - | 
    1k | 
    - | 

 
    J-HMDB ( Jhuang
et al., 2013 ) | 
    2013 | 
    Single | 
    15 | 
    0.6k | 
    - | 
    0.3k | 
    - | 

 
    PoseTrack2017 ( Andriluka et al., 2018a ) | 
    2017 | 
    Multiple | 
    15 | 
    250 | 
    50 | 
    214 | 
    mAP | 

 
    PoseTrack2018 ( Andriluka et al., 2018b ) | 
    2018 | 
    Multiple | 
    15 | 
    593 | 
    170 | 
    375 | 
    mAP | 

 
    HiEve ( Lin et al., 2020 ) | 
    2020 | 
    Multiple | 
    14 | 
    19 | 
    - | 
    13 | 
    mAP | 

 

 
 
 
 

### 4.2. Evaluation Metrics for 2D HPE

 
 It is difficult to precisely evaluate the performance of HPE because there are many features and requirements that need to be considered (e.g., upper/full human body, single/multiple pose estimation, the size of human body). As a result, many evaluation metrics have been used for 2D HPE. Here we summarize the commonly used ones.

 
 
 Percentage of Correct Parts (PCP) ( Eichner et al., 2012 ) is a measure commonly used in early works on 2D HPE, which evaluates stick predictions to report the localization accuracy for limbs. The localization of limbs is determined when the distance between the predicted joint and ground truth joint is less than a fraction of the limb length (between 0.1 to 0.5). In some works, the PCP measure is also referred to as PCP@0.5, where the threshold is 0.5. This measure is used for single-person HPE evaluation. However, PCP has not been widely implemented in the latest works because it penalizes the limbs with short lengths which are hard to detect. The performance of a model is considered better when it has a higher PCP measure. In order to address the drawbacks of PCP, Percentage of Detected Joints (PDJ) is introduced, where a predicted joint is considered as detected if the distance between predicted joints and true joints is within a certain fraction of the torso diameter ( Toshev and
Szegedy, 2014 ) .

 
 
 Percentage of Correct Keypoints (PCK) ( Yang and Ramanan, 2012 ) is also used to measure the accuracy of localization of different keypoints within a given threshold. The threshold is set to 50 percent of the head segment length of each test image and it is denoted as PCKh@0.5. PCK is referred to as PCK@0.2 when the distance between detected joints and true joints is less than 0.2 times the torso diameter. The higher the PCK value, the better model performance is regarded.

 
 
 Average Precision (AP) and Average Recall (AR) . AP measure is an index to measure the accuracy of keypoints detection according to precision (the ratio of true positive results to the total positive results) and recall (the ratio of true positive results to the total number of ground truth positives). AP computes the average precision value for recall over 0 to 1. AP has several similar variants. For example, Average Precision of Keypoints (APK) is introduced in ( Yang and Ramanan, 2012 ) . Mean Average Precision (mAP), which is the mean of average precision over all classes, is a widely used metric on the MPII and PoseTrack datasets. Average Recall (AR) is another metric used in the COCO keypoint evaluation ( Lin et al., 2014 ) . Object Keypoint Similarity (OKS) plays a similar role as the Intersection over Union (IoU) in object detection and is used for AP or AR. This measure is computed from the scale of the subject and the distance between predicted points and ground truth points. The COCO evaluation usually uses mAP across 10 OKS thresholds as the evaluation metric.

 
 

 Table 2. Comparison of different methods on the MPII dataset for 2D single-person HPE using PCKh@0.5 measure (i.e., the threshold is equal to 50 percent of the head segment length of each test image). H : Heatmap; R : Regression. ( Red : best; Blue : second best)
 
 
 
 
 
 
    Max Planck Institute for Informatics (MPII) | 

 
 
 
 | 
     Year | 
     Method | 
     Head | 
     Shoulder | 
     Elbow | 
     Wrist | 
     Hip | 
     Knee | 
     Ankle | 
     Mean | 
     Params(M) | 
     FLOPs(G) | 

 
 | 
    2017 | 
     ( Chu
et al., 2017 ) | 
     98.5 | 
    96.3 | 
    91.9 | 
    88.1 | 
    90.6 | 
    88.0 | 
    85.0 | 
    91.5 | 
    - | 
    - | 

 
 | 
    2017 | 
     ( Yang
et al., 2017 ) | 
     98.5 | 
    96.7 | 
    92.5 | 
    88.7 | 
    91.1 | 
    88.6 | 
    86.0 | 
    92.0 | 
    7.3 | 
    14.7 | 

 
 | 
    2018 | 
     ( Ke
et al., 2018 ) | 
     98.5 | 
    96.8 | 
    92.7 | 
    88.4 | 
    90.6 | 
    89.3 | 
    86.3 | 
    92.1 | 
    - | 
    - | 

 
 | 
    2018 | 
     ( Tang et al., 2018 ) | 
    98.4 | 
    96.9 | 
    92.6 | 
    88.7 | 
     91.8 | 
    89.4 | 
    86.2 | 
    92.3 | 
    15.5 | 
    33.6 | 

 
 | 
    2018 | 
     ( Xiao et al., 2018 ) | 
     98.5 | 
    96.6 | 
    91.9 | 
    87.6 | 
    91.1 | 
    88.1 | 
    84.1 | 
    91.5 | 
    - | 
    - | 

 
 | 
    2019 | 
     ( Sun
et al., 2019 ) | 
     98.6 | 
    96.9 | 
    92.8 | 
    89.0 | 
    91.5 | 
    89.0 | 
    85.7 | 
    92.3 | 
    28.5 | 
    9.5 | 

 
 | 
    2019 | 
     ( Zhang et al., 2019a ) | 
     98.6 | 
    97.0 | 
    92.8 | 
    88.8 | 
    91.7 | 
    89.8 | 
     86.6 | 
    92.5 | 
    23.9 | 
    41.4 | 

 
 | 
    2019 | 
     ( Li et al., 2019a ) | 
    98.4 | 
     97.1 | 
     93.2 | 
     89.2 | 
     92.0 | 
     90.1 | 
    85.5 | 
    92.6 | 
    - | 
    - | 

 
 | 
    2020 | 
     ( Artacho and
Savakis, 2020 ) | 
    - | 
    - | 
    - | 
    - | 
    - | 
    - | 
    - | 
     92.7 | 
    - | 
    - | 

 
 | 
    2020 | 
     ( Cai et al., 2020 ) | 
     98.5 | 
     97.3 | 
     93.9 | 
     89.9 | 
     92.0 | 
     90.6 | 
     86.8 | 
     93.0 | 
    - | 
    - | 

 
 | 
    2021 | 
     ( Li
et al., 2021e ) | 
    97.1 | 
    95.9 | 
    90.4 | 
    86.0 | 
    89.3 | 
    87.1 | 
    82.5 | 
    90.2 | 
    28.1 | 
    - | 

 
     H | 
    2022 | 
     ( Li et al., 2022b ) | 
    97.2 | 
    96.0 | 
    90.4 | 
    85.6 | 
    89.5 | 
    85.8 | 
    81.8 | 
    90.0 | 
    - | 
    - | 

 
 | 
    2017 | 
     ( Sun
et al., 2017 ) | 
    97.5 | 
    94.3 | 
    87.0 | 
    81.2 | 
    86.5 | 
    78.5 | 
    75.4 | 
    86.4 | 
    - | 
    - | 

 
 | 
    2019 | 
     ( Zhang
et al., 2019b ) | 
    98.3 | 
    96.4 | 
    91.5 | 
    87.4 | 
    90.0 | 
    87.1 | 
    83.7 | 
    91.1 | 
    3.0 | 
    9.0 | 

 
 | 
    2019 | 
     ( Luvizon
et al., 2019 ) | 
    98.1 | 
    96.6 | 
    92.0 | 
    87.5 | 
    90.6 | 
    88 | 
    82.7 | 
    91.2 | 
    - | 
    - | 

 
 | 
    2021 | 
     ( Li
et al., 2021c ) | 
    97.3 | 
    96.0 | 
    90.6 | 
    84.5 | 
    89.7 | 
    85.5 | 
    79.0 | 
    89.5 | 
    - | 
    - | 

 
 | 
    2021 | 
     ( Mao
et al., 2021 ) | 
    98.0 | 
    95.9 | 
    91.0 | 
    86.0 | 
    89.8 | 
    86.6 | 
    82.6 | 
    90.4 | 
    - | 
    - | 

 
     R | 
    2022 | 
     ( Mao et al., 2022 ) | 
    - | 
    - | 
    - | 
    - | 
    - | 
    - | 
    - | 
    90.5 | 
    - | 
    - | 

 

 
 
 • 
 
 Note: ( Insafutdinov et al., 2016 ) , ( Sun
et al., 2019 ) , ( Li et al., 2019a ) , ( Cai et al., 2020 ) , ( Rogez
et al., 2017 ) , ( Li
et al., 2021e ) are 2D multi-person HPE methods, which are also applied to the single-person case here. 

 

 
 
 
 Table 3. Comparison of different 2D multi-person HPE methods on the test-dev set of the COCO dataset using AP measure (AP.5: AP at OKS = 0.50; AP.75: AP at OKS = 0.75; AP(M) is used for medium objects; AP(L) is used for large objects). Extra : extra data is used for training. T : Top-down; B : Bottom-up
 
 
 
 
 
    Microsoft Common Objects in Context (COCO) | 

 
 
 
 | 
     Year | 
     Method | 
     Extra | 
     Backbone | 
     Input size | 
     AP | 
     AP.5 | 
     AP.75 | 
     AP(M) | 
     AP(L) | 
     Params(M) | 
     FLOPs(G) | 

 
 | 
    2019 | 
     ( Sun
et al., 2019 ) | 
    no | 
    HRNet-W32 | 
    384 × \times 288 | 
    74.9 | 
    92.5 | 
    82.8 | 
    71.3 | 
    80.9 | 
    28.5 | 
    16.0 | 

 
 | 
    2019 | 
     ( Sun
et al., 2019 ) | 
    no | 
    HRNet-W48 | 
    384 × \times 288 | 
    75.5 | 
    92.5 | 
    83.3 | 
    71.9 | 
    81.5 | 
    63.6 | 
    32.9 | 

 
 | 
    2019 | 
     ( Sun
et al., 2019 ) | 
    yes | 
    HRNet-W48 | 
    384 × \times 288 | 
    77.0 | 
    92.7 | 
    84.5 | 
    73.4 | 
    83.1 | 
    63.6 | 
    32.9 | 

 
 | 
    2019 | 
     ( Li et al., 2019a ) | 
    yes | 
    4xResNet-50 | 
    384 × \times 288 | 
    77.1 | 
     93.8 | 
    84.6 | 
    73.4 | 
    82.3 | 
    - | 
    - | 

 
 | 
    2020 | 
     ( Zhang
et al., 2020h ) | 
    no | 
    HRNet-W48 | 
    384 × \times 288 | 
    76.2 | 
    92.5 | 
    83.6 | 
    72.5 | 
    82.4 | 
    63.6 | 
    32.9 | 

 
 | 
    2020 | 
     ( Zhang
et al., 2020h ) | 
    yes | 
    HRNet-W48 | 
    384 × \times 288 | 
    77.4 | 
    92.6 | 
    84.6 | 
    73.6 | 
     83.7 | 
    63.6 | 
    32.9 | 

 
 | 
    2020 | 
     ( Cai et al., 2020 ) | 
    no | 
    4xRSN-50 | 
    384 × \times 288 | 
     78.6 | 
     94.3 | 
     86.6 | 
     75.5 | 
    83.3 | 
    111.8 | 
    65.9 | 

 
 | 
    2021 | 
     ( Yang
et al., 2021 ) | 
    no | 
    HRNet-W48 | 
    256 × \times 192 | 
    75.0 | 
    92.2 | 
    82.3 | 
    71.3 | 
    81.1 | 
    17.5 | 
    21.8 | 

 
 | 
    2021 | 
     ( Li
et al., 2021e ) | 
    no | 
    HRNet-W48 | 
    384 × \times 288 | 
    75.9 | 
    92.3 | 
    83.4 | 
    72.2 | 
    82.1 | 
    29.8 | 
    22.1 | 

 
 | 
    2021 | 
     ( Liu
et al., 2021b ) | 
    no | 
    HRNet-W48 | 
    384 × \times 288 | 
     79.5 | 
    93.6 | 
     85.9 | 
     76.3 | 
     84.3 | 
    35.4 | 
    70.1 | 

 
     T | 
    2022 | 
     ( Ma et al., 2022 ) | 
    no | 
    Transformer | 
    256 × \times 192 | 
    75.2 | 
    89.8 | 
    81.7 | 
    71.7 | 
    82.1 | 
    20.8 | 
    8.7 | 

 
 | 
    2017 | 
     ( Newell
et al., 2017 ) | 
    no | 
    Hourglass | 
    512 × \times 512 | 
    65.5 | 
    86.8 | 
    72.3 | 
    60.6 | 
    72.6 | 
    - | 
    - | 

 
 | 
    2018 | 
     ( Papandreou et al., 2018 ) | 
    no | 
    ResNet-152 | 
    1401 × \times 1401 | 
    68.7 | 
    89.0 | 
    75.4 | 
    64.1 | 
    75.5 | 
    68.7 | 
    405.5 | 

 
 | 
    2019 | 
     ( Tian
et al., 2019 ) | 
    no | 
    ResNet-101 | 
    800 × \times 800 | 
    64.8 | 
    87.8 | 
    71.1 | 
    60.4 | 
    71.5 | 
    - | 
    - | 

 
 | 
    2020 | 
     ( Jin et al., 2020a ) | 
    no | 
    Hourglass | 
    512 × \times 512 | 
    67.6 | 
    85.1 | 
    73.7 | 
    62.7 | 
    74.6 | 
    - | 
    - | 

 
 | 
    2020 | 
     ( Cheng et al., 2020 ) | 
    no | 
    HRNet-W48 | 
    640 × \times 640 | 
    70.5 | 
    89.3 | 
    77.2 | 
    66.6 | 
    75.8 | 
    63.8 | 
    154.3 | 

 
 | 
    2021 | 
     ( Luo
et al., 2021 ) | 
    no | 
    HRNet-W48 | 
    640 × \times 640 | 
    72.0 | 
    90.7 | 
    78.8 | 
    67.8 | 
    77.7 | 
    63.8 | 
    154.6 | 

 
     B | 
    2022 | 
     ( Wang
et al., 2022c ) | 
    no | 
    HRNet-W32 | 
    640 × \times 640 | 
    72.8 | 
    91.2 | 
    79.9 | 
    68.3 | 
    79.3 | 
    - | 
    - | 

 

 
 
 
 Table 4. Comparison of different 2D video-based HPE methods on the PoseTrack2017 test set and PoseTrack2018 test set 
 
 
 
 
    PoseTrack2017 | 

 
    Year | 
    Method | 
    Backbone | 
    Head | 
    Shoulder | 
    Elbow | 
    Wrist | 
    Hip | 
    Knee | 
    Ankle | 
    Total | 

 
    2018 | 
     ( Doering
et al., 2018 ) | 
    VGG | 
    - | 
    - | 
    - | 
    53.1 | 
    - | 
    - | 
    50.4 | 
    63.4 | 

 
    2019 | 
     ( Bertasius et al., 2019 ) | 
    HRNet-W48 | 
    79.5 | 
    84.3 | 
    80.1 | 
    75.8 | 
    77.6 | 
    76.8 | 
    70.8 | 
    77.9 | 

 
    2020 | 
     ( Snower
et al., 2020 ) | 
    Transformer | 
    - | 
    - | 
    - | 
    71.9 | 
    - | 
    - | 
    65.0 | 
    74.0 | 

 
    2021 | 
     ( Liu
et al., 2021a ) | 
    HRNet-W48 | 
     84.3 | 
     84.9 | 
     80.5 | 
     76.1 | 
     77.9 | 
     77.1 | 
     71.2 | 
     79.2 | 

 
    2022 | 
     ( Liu
et al., 2022 ) | 
    HRNet-W48 | 
     86.1 | 
     86.1 | 
     81.8 | 
     77.4 | 
     79.5 | 
     79.1 | 
     73.6 | 
     80.9 | 

 
    PoseTrack2018 | 

 
    2018 | 
     ( Guo
et al., 2018 ) | 
    ResNet-152 | 
    - | 
    - | 
    - | 
    74.5 | 
    - | 
    - | 
    69.0 | 
    76.4 | 

 
    2019 | 
     ( Bertasius et al., 2019 ) | 
    HRNet-W48 | 
    78.9 | 
     84.4 | 
     80.9 | 
    76.8 | 
    75.6 | 
    77.5 | 
    71.8 | 
    78.0 | 

 
    2021 | 
     ( Liu
et al., 2021a ) | 
    HRNet-W48 | 
     82.8 | 
    84.0 | 
    80.8 | 
     77.2 | 
     76.1 | 
     77.6 | 
     72.3 | 
     79.0 | 

 
    2022 | 
     ( Liu
et al., 2022 ) | 
    HRNet-W48 | 
     83.6 | 
     84.5 | 
     81.4 | 
     77.9 | 
     76.8 | 
     78.3 | 
     72.9 | 
     79.6 | 

 

 
 
 
 Table 5. Datasets for 3D HPE. 
 
 
 
 
 Dataset | 
 Year | 
 Capture system | 
 Environment | 
 Size | 
 Single person | 
 Multi-person | 
 Single view | 
 Multi-view | 

 
 HumanEva ( Sigal
et al., 2010 ) | 
 2010 | 
 Marker-based MoCap | 
 Indoor | 
 6 subject, 7 actions, 40k frames | 
 Yes | 
 No | 
 Yes | 
 Yes | 

 
 Human3.6M ( Ionescu et al., 2014 ) | 
 2014 | 
 Marker-based MoCap | 
 Indoor | 
 11 subjects, 17 actions, 3.6M frames | 
 Yes | 
 No | 
 Yes | 
 Yes | 

 
 CMU Panoptic ( Joo
et al., 2017 ) | 
 2016 | 
 Marker-less MoCap | 
 Indoor | 
 8 subjects, 1.5M frames | 
 Yes | 
 Yes | 
 Yes | 
 Yes | 

 
 MPI-INF-3DHP ( Mehta et al., 2017 ) | 
 2017 | 
 Marker-less MoCap | 
 Indoor and outdoor | 
 8 subjects, 8 actions, 1.3M frames | 
 Yes | 
 No | 
 Yes | 
 Yes | 

 
 TotalCapture ( Trumble et al., 2017 ) | 
 2017 | 
 Marker-based MoCap with IMUs | 
 Indoor | 
 5 subjects, 5 actions, 1.9M frames | 
 Yes | 
 No | 
 Yes | 
 Yes | 

 
 3DPW ( von Marcard et al., 2018 ) | 
 2018 | 
 Hand-held cameras with IMUs | 
 Indoor and outdoor | 
 7 subjects, 51k frames | 
 Yes | 
 Yes | 
 Yes | 
 No | 

 
 MuPoTS-3D ( Mehta et al., 2018 ) | 
 2018 | 
 Marker-less MoCap | 
 Indoor and outdoor | 
 8 subjects, 8k frames | 
 Yes | 
 Yes | 
 Yes | 
 Yes | 

 
 AMASS ( Mahmood et al., 2019 ) | 
 2019 | 
 Marker-based MoCap | 
 Indoor and outdoor | 
 300 subjects, 9M frames | 
 Yes | 
 No | 
 Yes | 
 Yes | 

 
 NBA2K ( Zhu et al., 2020 ) | 
 2020 | 
 NBA2K19 game engine | 
 Indoor | 
 27 subjects, 27k poses | 
 Yes | 
 No | 
 Yes | 
 No | 

 
 GTA-IM ( Cao
et al., 2020 ) | 
 2020 | 
 GTA game engine | 
 Indoor | 
 1M frames | 
 Yes | 
 No | 
 Yes | 
 No | 

 
 Occlusion-Person ( Zhang
et al., 2020g ) | 
 2020 | 
 Unreal Engine 4 game engine | 
 Indoor | 
 73k frames | 
 Yes | 
 No | 
 Yes | 
 Yes | 

 

 
 
 
 

### 4.3. Performance Comparison of 2D HPE Methods

 
 Single-person 2D HPE: Table 2 shows the comparison results for different 2D single-person HPE methods on the MPII dataset using PCKh@0.5 measure. Although both heatmap-based and regression-based methods have impressive results, they have their own limitations in 2D single-person HPE. Regression methods can learn a nonlinear mapping from input images to keypoint coordinates with an end-to-end framework, which offers a fast learning paradigm and a sub-pixel level prediction accuracy. However, they usually give sub-optimal solutions ( Luvizon
et al., 2019 ) due to the highly nonlinear problem.
Heatmap-based methods outperform regression-based approaches and are more widely used in 2D HPE ( Li et al., 2019a ) ( Cai et al., 2020 ) ( Luvizon
et al., 2019 ) since (1) the probabilistic prediction of each pixel in a heatmap can improve the accuracy of locating the keypoints, and (2) heatmaps provide richer supervision information by preserving the spatial location information. However, the precision of the predicted keypoints is dependent on the resolution of heatmaps. The computational cost and memory footprint are significantly increased when using high-resolution heatmaps ( Sun
et al., 2019 ) .

 
 
 Multi-person 2D HPE: 
Table 3 presents the experimental results of different 2D HPE methods on the test-dev set of the COCO dataset, together with a summary of the experiment settings (extra data, backbones in models, input image size) and AP scores for each approach. The comparison experiments highlight the significant results of both top-down and bottom-up methods in multi-person HPE. Presumably, the top-down pipeline yields better results because it first detects each individual from the image using detection methods, then predicts the locations of keypoints using single-person HPE approaches. In this case, the keypoint estimation for each detected person is made easier, as the background is largely removed. But on the other hand, bottom-up methods are generally faster than top-down methods, because they directly detect all the keypoints and group them into individual poses using keypoint association strategies such as affinity linking ( Cao
et al., 2017 ) , associative embedding ( Newell
et al., 2017 ) , and pixel-wise keypoint regression ( Zhou
et al., 2019 ) . Besides the image-based methods listed above, Table 4 also illustrates the comparisons of the recent video-based works on PoseTrack2017 and PoseTrack2018 datasets. The detailed results of the test sets are summarized.

 
 
 

### 4.4. Datasets for 3D HPE

 
 In contrast to numerous 2D human pose datasets with high-quality annotation, acquiring accurate 3D annotation for 3D HPE datasets is a challenging task that requires motion capture systems such as MoCap and wearable IMUs.
Due to the page limit, we only review several widely used large-scale 3D pose datasets for deep learning-based 3D HPE in the following.

 
 
 Human3.6M ( Ionescu et al., 2014 ) is the most widely used indoor dataset for 3D HPE from monocular images and videos. There are 11 professional actors performing 17 activities
from 4 different views in an indoor laboratory environment. This dataset contains 3.6 million 3D human poses with 3D ground truth annotation captured by accurate marker-based MoCap systems. Protocol #1 uses images of subjects S1, S5, S6, and S7 for training, and images of subjects S9 and S11 for testing.

 
 
 MuPoTS-3D ( Mehta et al., 2018 ) is a multi-person 3D test set and its ground-truth 3D poses were captured by a multi-view marker-less MoCap system containing 20 real-world scenes (5 indoor and 15 outdoor). There are challenging samples with occlusions, drastic illumination changes, and lens flares in some of the outdoor footage. More than 8,000 frames were collected in the 20 sequences by 8 subjects.

 
 
 Readers are referred to the original references for details about other datasets including MPI-INF-3DHP ( Mehta et al., 2017 ) , HumanEva ( Sigal
et al., 2010 ) , CMU Panoptic Dataset ( Joo
et al., 2017 ) , TotalCapture ( Trumble et al., 2017 ) , MuCo-3DHP Dataset ( Mehta et al., 2018 ) , 3DPW ( von Marcard et al., 2018 ) , AMASS ( Mahmood et al., 2019 ) , NBA2K ( Zhu et al., 2020 ) , GTA-IM ( Cao
et al., 2020 ) , and Occlusion-Person ( Zhang
et al., 2020g ) . A summary of these datasets is shown in Table 5 .

 
 
 Table 6. Comparison of different 3D single-view single-person HPE approaches on the Human3.6M dataset (Protocol 1). In skeleton-only approaches, “Direct” indicates the methods directly estimating 3D pose without 2D pose representation. “Lifting” denotes the methods lifting the 2D pose representation to the 3D space. The reported total parameters (Params) and FLOPs for 2D-3D lifting methods are computed without including the params and FLOPs of the external 2D pose detector. 
 
 
 
 
 Skeleton-only methods | 
 MPJPE | 
 PA-MPJPE | 

 
 Approaches | 
 Year | 
 Methods | 
 2D pose detector | 
 Input | 
 Params(M) | 
 FLOPs(G) | 
 Dir. | 
 Disc. | 
 Eat. | 
 Greet | 
 Phone | 
 Photo | 
 Pose | 
 Purch. | 
 Sit | 
 SitD. | 
 Somke | 
 Wait | 
 WalkD. | 
 Walk | 
 WalkT. | 
 Average | 
 Average | 

 
 | 
 2017 | 
 ( Pavlakos et al., 2017a ) | 
 - | 
 image | 
 - | 
 - | 
 67.4 | 
 71.9 | 
 66.7 | 
 69.1 | 
 72.0 | 
 77.0 | 
 65.0 | 
 68.3 | 
 83.7 | 
 96.5 | 
 71.7 | 
 65.8 | 
 74.9 | 
 59.1 | 
 63.2 | 
 71.9 | 
 51.9 | 

 
 Direct | 
 2018 | 
 ( Pavlakos
et al., 2018a ) | 
 - | 
 image | 
 - | 
 - | 
 48.5 | 
 54.4 | 
 54.4 | 
 52.0 | 
 59.4 | 
 65.3 | 
 49.9 | 
 52.9 | 
 65.8 | 
 71.1 | 
 56.6 | 
 52.9 | 
 60.9 | 
 44.7 | 
 47.8 | 
 56.2 | 
 41.8 | 

 
 | 
 2019 | 
 ( Li and Lee, 2019 ) | 
 Hourglass | 
 image | 
 - | 
 - | 
 43.8 | 
 48.6 | 
 49.1 | 
 49.8 | 
 57.6 | 
 61.5 | 
 45.9 | 
 48.3 | 
 62.0 | 
 73.4 | 
 54.8 | 
 50.6 | 
 56.0 | 
 43.4 | 
 45.5 | 
 52.7 | 
 42.6 | 

 
 | 
 2019 | 
 ( Zhao
et al., 2019b ) | 
 own design | 
 image | 
 - | 
 - | 
 47.3 | 
 60.7 | 
 51.4 | 
 60.5 | 
 61.1 | 
 49.9 | 
 47.3 | 
 68.1 | 
 86.2 | 
 55.0 | 
 67.8 | 
 61.0 | 
 42.1 | 
 60.6 | 
 45.3 | 
 57.6 | 
 - | 

 
 | 
 2021 | 
 ( Zou and Tang, 2021 ) | 
 CPN | 
 image | 
 0.3 | 
 - | 
 45.4 | 
 49.2 | 
 45.7 | 
 49.4 | 
 50.4 | 
 58.2 | 
 47.9 | 
 46.0 | 
 57.5 | 
 63.0 | 
 49.7 | 
 46.6 | 
 52.2 | 
 38.9 | 
 40.8 | 
 49.4 | 
 39.1 | 

 
 | 
 2019 | 
 ( Pavllo et al., 2019 ) | 
 CPN | 
 Video | 
 17 | 
 0.03 | 
 45.2 | 
 46.7 | 
 43.3 | 
 45.6 | 
 48.1 | 
 55.1 | 
 44.6 | 
 44.3 | 
 57.3 | 
 65.8 | 
 47.1 | 
 44.0 | 
 49.0 | 
 32.8 | 
 33.9 | 
 46.8 | 
 36.5 | 

 
 | 
 2019 | 
 ( Cai et al., 2019 ) | 
 CPN | 
 Video | 
 - | 
 - | 
 44.6 | 
 47.4 | 
 45.6 | 
 48.8 | 
 50.8 | 
 59.0 | 
 47.2 | 
 43.9 | 
 57.9 | 
 61.9 | 
 49.7 | 
 46.6 | 
 51.3 | 
 37.1 | 
 39.4 | 
 48.8 | 
 39.0 | 

 
 | 
 2020 | 
 ( Liu
et al., 2020c ) | 
 CPN | 
 Video | 
 11 | 
 - | 
 41.8 | 
 44.8 | 
 41.1 | 
 44.9 | 
 47.4 | 
 54.1 | 
 43.4 | 
 42.2 | 
 56.2 | 
 63.6 | 
 45.3 | 
 43.5 | 
 45.3 | 
 31.3 | 
 32.2 | 
 45.1 | 
 35.6 | 

 
 | 
 2020 | 
 ( Zeng
et al., 2020 ) | 
 CPN | 
 Video | 
 - | 
 - | 
 46.6 | 
 47.1 | 
 43.9 | 
 41.6 | 
 45.8 | 
 49.6 | 
 46.5 | 
 40.0 | 
 53.4 | 
 61.1 | 
 46.1 | 
 42.6 | 
 43.1 | 
 31.5 | 
 32.6 | 
 44.8 | 
 - | 

 
 | 
 2020 | 
 ( Wang
et al., 2020d ) | 
 HRNet | 
 Video | 
 - | 
 - | 
 38.2 | 
 41.0 | 
 45.9 | 
 39.7 | 
 41.4 | 
 51.4 | 
 41.6 | 
 41.4 | 
 52.0 | 
 57.4 | 
 41.8 | 
 44.4 | 
 41.6 | 
 33.1 | 
 30.0 | 
 42.6 | 
 32.7 | 

 
 | 
 2020 | 
 ( Chen
et al., 2021 ) | 
 CPN | 
 Video | 
 59 | 
 0.1 | 
 41.4 | 
 43.5 | 
 40.1 | 
 42.9 | 
 46.6 | 
 51.9 | 
 41.7 | 
 42.3 | 
 53.9 | 
 60.2 | 
 45.4 | 
 41.7 | 
 46.0 | 
 31.5 | 
 32.7 | 
 44.1 | 
 35.0 | 

 
 | 
 2021 | 
 ( Zheng et al., 2021 ) | 
 CPN | 
 Video | 
 10 | 
 1.4 | 
 41.5 | 
 44.8 | 
 39.8 | 
 42.5 | 
 46.5 | 
 51.6 | 
 42.1 | 
 42.0 | 
 53.3 | 
 60.7 | 
 45.5 | 
 43.3 | 
 46.1 | 
 31.8 | 
 32.2 | 
 44.3 | 
 34.6 | 

 
 | 
 2022 | 
 ( Li
et al., 2022a ) | 
 CPN | 
 Video | 
 32 | 
 7.0 | 
 39.2 | 
 43.1 | 
 40.1 | 
 40.9 | 
 44.9 | 
 51.2 | 
 40.6 | 
 41.3 | 
 53.5 | 
 60.3 | 
 43.7 | 
 41.1 | 
 43.8 | 
 29.8 | 
 30.6 | 
 43.0 | 
 - | 

 
 Lifting | 
 2022 | 
 ( Zhang
et al., 2022 ) | 
 CPN | 
 Video | 
 41 | 
 0.6 | 
 37.6 | 
 40.9 | 
 37.3 | 
 39.7 | 
 42.3 | 
 49.9 | 
 40.1 | 
 39.8 | 
 51.7 | 
 55.0 | 
 42.1 | 
 39.8 | 
 41.0 | 
 27.9 | 
 27.9 | 
 40.9 | 
 32.6 | 

 
 Human mesh recovery methods | 
 MPJPE | 
 PA-MPJPE | 

 
 output | 
 Year | 
 Methods | 
 Model | 
 Input | 
 Params(M) | 
 FLOPs(G) | 
 Dir. | 
 Disc. | 
 Eat. | 
 Greet | 
 Phone | 
 Photo | 
 Pose | 
 Purch. | 
 Sit | 
 SitD. | 
 Somke | 
 Wait | 
 WalkD. | 
 Walk | 
 WalkT. | 
 Average | 
 Average | 

 
 | 
 2019 | 
 ( Kolotouros et al., 2019 ) | 
 SMPL | 
 Image | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 50.1 | 

 
 | 
 2019 | 
 ( Kolotouros et al., 2019 ) | 
 SMPL | 
 Image | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 41.1 | 

 
 | 
 2019 | 
 ( Xiang
et al., 2019 ) | 
 Adam | 
 Image | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 58.3 | 
 - | 

 
 | 
 2020 | 
 ( Choi
et al., 2020 ) | 
 - | 
 Image | 
 140 | 
 11 | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 64.9 | 
 47.0 | 

 
 | 
 2021 | 
 ( Zhang et al., 2021 ) | 
 SMPL | 
 Image | 
 45.2 | 
 11 | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 57.7 | 
 40.5 | 

 
 | 
 2021 | 
 ( Lin
et al., 2021a ) | 
 SMPL | 
 Image | 
 230 | 
 57 | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 54.0 | 
 36.7 | 

 
 | 
 2021 | 
 ( Lin
et al., 2021b ) | 
 SMPL | 
 Image | 
 230 | 
 57 | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 51.2 | 
 34.5 | 

 
 | 
 2022 | 
 ( Cho
et al., 2022 ) | 
 SMPL | 
 Image | 
 49 | 
 16 | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 52.2 | 
 33.7 | 

 
 | 
 2019 | 
 ( Arnab
et al., 2019 ) | 
 SMPL | 
 Video | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 77.8 | 
 54.3 | 

 
 | 
 2020 | 
 ( Kocabas
et al., 2020 ) | 
 SMPL | 
 Video | 
 59 | 
 10 | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 65.6 | 
 41.4 | 

 
 Mesh | 
 2021 | 
 ( Choi
et al., 2021 ) | 
 SMPL | 
 Video | 
 123 | 
 10 | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 62.3 | 
 41.1 | 

 

 
 
 
 

### 4.5. Evaluation Metrics for 3D HPE

 
 MPJPE (Mean Per Joint Position Error) is the most widely used metric to evaluate the performance of 3D HPE. MPJPE is computed by using the Euclidean distance between the estimated 3D joints and the ground truth positions
as follows:

 

 
 (1) | 
 | 
 M ​ P ​ J ​ P ​ E = 1 N ​ ∑ i = 1 N ‖ J i − J i ∗ ‖ 2 , \displaystyle\small{MPJPE}=\frac{1}{N}\sum_{i=1}^{N}\|J_{i}-J_{i}^{*}\|_{2}, | 
 | 
 

 where N N is the number of joints, J i J_{i} and J i ∗ J^{*}_{i} are the ground truth position and the estimated position of the i t ​ h i_{th} joint.

 
 
 PA-MPJPE , also called Reconstruction Error, is the MPJPE after rigid alignment by post-processing between the estimated pose and the ground truth pose.

 
 
 NMPJPE is defined as the MPJPE after normalizing the predicted positions in scale to the reference ( Rhodin et al., 2018b ) .

 
 
 MPVE (Mean Per Vertex Error) ( Pavlakos
et al., 2018b ) measures
the Euclidean distances between the ground truth vertices and the predicted vertices:

 

 
 (2) | 
 | 
 M ​ P ​ V ​ E = 1 N ​ ∑ i = 1 N ‖ V i − V i ∗ ‖ 2 , \displaystyle\small{MPVE}=\frac{1}{N}\sum_{i=1}^{N}\|V_{i}-V_{i}^{*}\|_{2}, | 
 | 
 

 where N N is the number of vertices, V V is the ground truth vertices, and V ∗ V^{*} is the estimated vertices.

 
 
 3DPCK is a 3D extended version of the Percentage of Correct Keypoints (PCK) metric used in 2D HPE evaluation. An estimated joint is considered as correct if the distance between the estimation and the ground truth is within a certain threshold. Generally, the threshold is set to 150 m ​ m mm .

 
 
 
 
 
 Table 7. Comparison of different 3D single-view multi-person HPE approaches on the MuPoTS-3D dataset. The reported fps is taken from ( Wang
et al., 2022b ) . 
 
 
 
 MuPoTS-3D | 

 
 
 
 | 
 | 
 | 
 3DPCK ↑ \uparrow | 
 fps | 

 
 | 
 Year | 
 Method | 
 All people | 
 Matched people | 

 
 | 
 2019 | 
 ( Moon
et al., 2019a ) | 
 81.8 | 
 82.5 | 
 9.3 | 

 
 | 
 2020 | 
 ( Jiang et al., 2020 ) | 
 69.1 | 
 72.2 | 
 - | 

 
 | 
 2020 | 
 ( Li
et al., 2020b ) | 
 82.0 | 
 - | 
 - | 

 
 Top down | 
 2021 | 
 ( Cheng
et al., 2021a ) | 
 87.5 | 
 - | 
 - | 

 
 | 
 2018 | 
 ( Mehta et al., 2018 ) | 
 65.0 | 
 69.8 | 
 - | 

 
 | 
 2019 | 
 ( Mehta et al., 2020 ) | 
 70.4 | 
 - | 
 - | 

 
 | 
 2020 | 
 ( Benzine et al., 2020 ) | 
 72.0 | 
 - | 
 - | 

 
 Bottom up | 
 2020 | 
 ( Zhen et al., 2020 ) | 
 73.5 | 
 80.5 | 
 9.3 | 

 
 | 
 2021 | 
 ( Cheng
et al., 2021b ) | 
 89.6 | 
 - | 
 - | 

 
 Integrated | 
 2022 | 
 ( Wang
et al., 2022b ) | 
 82.7 | 
 - | 
 13.3 | 

 

 
 
 Table 8. Comparison of different 3D multi-view HPE approaches on the Human3.6M dataset. 
 
 
 
 Human3.6M | 

 
 | 
 | 
 | 
 Protocol 1 | 
 Protocol 1 | 
 FLOPs(G) | 

 
 Year | 
 Method | 
 
 
 
 Use extra 
 
 3D data 
 | 
 MPJPE ↓ \downarrow | 
 
 
 
 Normalized 
 
 MPJPE ↓ \downarrow 
 | 
 PA-MPJPE ↓ \downarrow | 

 
 2019 | 
 ( Liang and Lin, 2019 ) | 
 Yes | 
 79.9 | 
 - | 
 45.1 | 
 - | 

 
 2019 | 
 ( Kocabas
et al., 2019 ) | 
 No | 
 60.6 | 
 60.0 | 
 47.5 | 
 - | 

 
 2019 | 
 ( Qiu
et al., 2019 ) | 
 No | 
 31.2 | 
 - | 
 - | 
 55 | 

 
 2019 | 
 ( Qiu
et al., 2019 ) | 
 Yes | 
 26.2 | 
 - | 
 - | 
 55 | 

 
 2020 | 
 ( Remelli
et al., 2020 ) | 
 No | 
 30.2 | 
 - | 
 - | 
 - | 

 
 2020 | 
 ( Xie
et al., 2020 ) | 
 No | 
 29.3 | 
 - | 
 - | 
 - | 

 
 2020 | 
 ( Zhang
et al., 2020g ) | 
 Yes | 
 19.5 | 
 - | 
 - | 
 - | 

 
 2021 | 
 ( Ma et al., 2021 ) | 
 No | 
 25.8 | 
 - | 
 - | 
 50 | 

 
 2021 | 
 ( Reddy et al., 2021 ) | 
 Yes | 
 18.7 | 
 - | 
 - | 
 - | 

 
 2022 | 
 ( Ma et al., 2022 ) | 
 Yes | 
 24.4 | 
 - | 
 - | 
 15 | 

 

 
 
 
 Summary. As pointed out by Ji et al. ( JI
et al., 2020 ) , low MPJPE does not always indicate an accurate pose estimation as it depends on the predicted scale of human shape and skeleton. Although 3DPCK is more robust to incorrect joints, it cannot evaluate the precision of correct joints.
Existing metrics are designed to evaluate the precision of an estimated pose in a single frame. However, the temporal consistency and smoothness of the reconstructed human pose cannot be examined over continuous frames by existing evaluation metrics. Designing frame-level evaluation metrics that can evaluate 3D HPE performance with temporal consistency and smoothness remains an open problem.

 
 
 

### 4.6. Performance Comparison of 3D HPE Methods

 
 Single-view single-person : In Table 6 , it is seen that most 3D single-view single-person HPE methods estimate 3D human pose with remarkable precision on the Human3.6M dataset. However, despite the fact that the Human3.6M dataset has a large size of training and testing data, it only contains 11 actors performing 17 activities in lab environments. When estimating 3D pose on in-the-wild data with more complex scenarios, the performance of these methods degrades quickly. Estimating 3D pose from videos can achieve better performance than from a single image because the temporal consistency is preserved.

 
 
 For skeleton-only methods, 2D-to-3D lifting approaches generally outperform direct estimation approaches due to the excellent performance of state-of-the-art 2D pose detectors.
Beyond estimated 3D coordinates of joints, a group of methods utilized volumetric models such as SMPL to recover human mesh. These methods still reported MPJPE of joints since the datasets do not provide the ground truth mesh vertices. However, the reported MPJPE is higher than those methods that only estimate 3D joints. One of the reasons is that these methods regressed both pose parameters and shape parameters, then fed in the volumetric model for mesh reconstruction, only evaluating MPJPE of joints can not show their strength.

 
 
 Single-view multi-person : Estimating 3D poses in the multi-person setting is a harder task than in single-person due to more severe occlusion. As shown in Tables 8 , good progress has been made in the single-view multi-person HPE methods in recent years. The Top-Down methods perform better than Bottom-Up methods due to the state-of-the-art person detection methods and single-person HPE methods. On the other hand, Bottom-Up methods are more computationally and time efficient.

 
 
 Multi-view : By comparing the results from Table 6 and Table 8 , it is evident that the performance (e.g., MPJPE under Protocol 1) of multi-view 3D HPE methods has improved compared to single-view 3D HPE methods using the same dataset and evaluation metric. Occlusion and depth ambiguity can be alleviated through the multi-view setting.

 
 
 
 

## 5. Applications

 
 In this section, we review related works of exploring HPE for a few popular applications.

 
 
 Action recognition, prediction, detection, and tracking : Pose information has been utilized as cues for various applications such as action recognition, prediction, detection, and tracking. Angelini et al. ( Angelini
et al., 2018 ) proposed a real-time action recognition method using a pose-based algorithm. Yan et al. ( Yan
et al., 2018 ) leveraged the dynamic skeleton modality of pose for action recognition. Markovitz et al. ( Markovitz et al., 2020 ) studied human pose graphs for anomaly detection of human actions in videos. Cao et al. ( Cao
et al., 2020 ) used the predicted 3D pose for long-term human motion prediction. Sun et al. ( Sun
et al., 2020 ) proposed a view-invariant probabilistic pose embedding for video alignment. Hua et al. ( Hua
et al., 2023 ) proposed an attention-based contrastive learning framework that integrates local similarity and global features for skeleton-based action
representations.

 
 
 Pose-based video surveillance enjoys the advantage of preserving privacy by monitoring through pose and human mesh representation instead of human sensitive identities. Das et al. ( Das et al., 2020 ) embedded video with poses to identify activities of daily living for monitoring human behavior.

 
 
 Action correction and online coaching :
Some activities such as dancing, sporting, and professional training require precise human body control guidance. Normally personal trainers are responsible for pose correction and action guidance in a face-to-face manner. With the help of 3D HPE and action detection, AI personal trainers can make coaching more convenient by simply setting up cameras without a personal trainer present. Wang et al. ( Wang
et al., 2019b ) designed an AI coaching system with a pose estimation module for personalized athletic training assistance.

 
 
 Clothes parsing :
The e-commerce trends have brought about a noticeable impact on various aspects including clothes purchases. Clothing products in pictures can no longer satisfy customers’ demands, and customers hope to see reliable appearances as they wear their selected clothes. Clothes parsing ( Yu et al., 2019 ) ( Saito et al., 2019 ) and pose transfer ( Li et al., 2019 ) make it possible by inferring the 3D appearance of a person wearing specific clothes. HPE can provide plausible human body regions for cloth parsing. Moreover, the recommendation system can be upgraded by evaluating appropriateness based on the inferred reliable 3D appearance of customers with selected items. Patel et al. ( Patel
et al., 2020 ) achieved clothing prediction from 3D pose, shape, and garment style.

 
 
 Animation, movie, and gaming: 
Motion capture is the key component to present characters with complex movements and realistic physical interactions in industries of animation, movies, and gaming. Equipments are usually expensive and complicated to set up. HPE can provide realistic pose information while alleviating the demand for professional high-cost equipment ( Willett et al., 2020 ) ( Liu et al., 2020b ) .

 
 
 AR and VR :
Augmented Reality (AR) technology aims to enhance the interactive experience of digital objects in the real-world environment. The objective of Virtual Reality (VR) technology is to provide an immersive experience for the users. AR and VR devices use human pose information as input to achieve their goals of different applications. A cartoon character can be generated in real-world scenes to replace a real person. Weng et al. ( Weng et al., 2019 ) created 3D character animation from a single photo with the help of 3D pose estimation and human mesh recovery. Zhang et al. ( Zhang et al., 2020d ) presented a pose-based system that converts broadcast tennis match videos into interactive and controllable video sprites.

 
 
 Healthcare :
HPE provides quantitative human motion information that physicians can use to diagnose some complex diseases, create rehabilitation training, and operate physical therapy. Lu et al. ( Lu et al., 2020 ) designed a pose-based estimation system for assessing Parkinson’s disease motor severity. Gu et al. ( Gu et al., 2019 ) developed a pose-based physical therapy system that patients can be evaluated and advised at their homes. Furthermore, such a system can be established to detect abnormal actions and predict subsequent actions ahead of time. Alerts are sent immediately if the system determines that danger may occur. Chen et al. ( Chen
et al., 2020d ) used HPE algorithms for fall detection monitoring in order to provide immediate assistance. Also, HPE methods can provide reliable posture labels of patients in hospital environments to augment research on neural correlates to natural behaviors ( Chen
et al., 2018a ) .

 
 
 

## 6. Conclusion and Future Directions

 
 In this survey, we have presented a systematic overview of recent deep learning-based 2D and 3D HPE methods. A comprehensive taxonomy and performance comparison of these methods have been covered.
Despite great success, there are still many
challenges as discussed in Sections 2.3 and 3.3 . We further point out a few promising future directions to promote advances in HPE research.

 
 
 
 • 
 
 Domain adaptation for HPE. For some applications such as estimating human pose from infant images ( Huang et al., 2020a ) or artwork collections ( Madhu et al., 2020 ) , there are not enough training data with ground truth annotations. Moreover, data for these applications exhibit different distributions from that of the standard pose datasets. HPE methods trained on existing standard datasets may not generalize well across different domains. The recent trend to alleviate the domain gap is utilizing GAN-based learning approaches. Nonetheless, how to effectively transfer the human pose knowledge to bridge domain gaps remains unaddressed.

 

 • 
 
 Human body models such as SMPL, SMPL-X, GHUM GHUML, and Adam are used to model human mesh representation.
However, these models have a huge number of parameters. How to reduce the number of parameters while preserving the reconstructed mesh quality is an intriguing problem. Also, different people have various deformations of body shape. A more effective human body model may utilize other information such as BMI ( Osman
et al., 2020 ) and silhouette ( Li
et al., 2020a ) for better generalization.

 

 • 
 
 Most existing methods ignore human interaction with 3D scenes. There are strong human-scene relationship constraints that can be explored such as a human subject cannot be simultaneously present in the locations of other objects in the scene. The physical constraints with semantic cues can provide reliable and realistic 3D HPE.

 

 • 
 
 3D HPE is employed in visual tracking and analysis. Existing 3D HPE from videos are not smooth and continuous. One reason is that the evaluation metrics such as MPJPE cannot evaluate the smoothness and the degree of realisticness. Appropriate frame-level evaluation metrics focusing on temporal consistency and motion smoothness should be developed.

 

 • 
 
 Existing well-trained networks pay less attention to resolution mismatch. The training data of HPE networks are usually high-resolution images or videos, which may lead to inaccurate estimation when predicting human pose from low-resolution input. The contrastive learning scheme ( Chen
et al., 2020e ) (e.g., the original image and its low-resolution version as a positive pair) might be helpful for building resolution-aware HPE networks.

 

 • 
 
 Deep neural networks in vision tasks are vulnerable to adversarial attacks. Imperceptible noise can significantly affect the performance of HPE. There are few works ( Liu
et al., 2019 ) ( Jain
et al., 2019 ) that consider adversarial attacks for HPE. The study of defense against adversarial attacks can improve the robustness of HPE networks and facilitate real-world pose-based applications.

 

 • 
 
 Human body parts may have different movement patterns and shapes due to the heterogeneity of the human body. A single shared network architecture may not be optimal for estimating all body parts with varying degrees of freedom. Neural Architecture Search (NAS) ( Elsken
et al., 2019 ) can search the optimal architecture for estimating each body part ( Chen
et al., 2020c ) . Also, NAS can be used for discovering efficient HPE network architectures to reduce the computational cost ( Zhang
et al., 2020b ) . It is also worth exploring multi-objective NAS in HPE when multiple objectives (e.g, latency, accuracy, and energy consumption) have to be met.

 

 
 
 
 HPE workshops and challenges: Finally, we list the HPE workshops and challenges (2017-2022) on our project page ( https://github.com/zczcwh/DL-HPE ) to facilitate research in this field.

 
 
 

## References

 
 
 Andriluka et al . (2018a) 
 
M. Andriluka, U. Iqbal,
E. Ensafutdinov, L. Pishchulin,
A. Milan, J. Gall, and
Schiele B. 2018a.

 
 PoseTrack: A Benchmark for Human Pose
Estimation and Tracking. In CVPR .

 
 
 

 
 Andriluka et al . (2018b) 
 
Mykhaylo Andriluka, Umar
Iqbal, Eldar Insafutdinov, Leonid
Pishchulin, Anton Milan, Juergen Gall,
and Bernt Schiele. 2018b.

 
 Posetrack: A benchmark for human pose estimation
and tracking. In CVPR .

 
 
 

 
 Andriluka et al . (2014) 
 
Mykhaylo Andriluka, Leonid
Pishchulin, Peter Gehler, and Bernt
Schiele. 2014.

 
 2d human pose estimation: New benchmark and state
of the art analysis. In CVPR .

 
 
 

 
 Angelini
et al . (2018) 
 
Federico Angelini, Zeyu
Fu, Yang Long, Ling Shao, and
Syed Mohsen Naqvi. 2018.

 
 Actionxpose: A novel 2d multi-view pose-based
algorithm for real-time human action recognition. In
 arXiv preprint arXiv:1810.12126 .

 
 
 

 
 Arnab
et al . (2019) 
 
Anurag Arnab, Carl
Doersch, and Andrew Zisserman.
2019.

 
 Exploiting Temporal Context for 3D Human Pose
Estimation in the Wild. In CVPR .

 
 
 

 
 Artacho and
Savakis (2020) 
 
Bruno Artacho and
Andreas Savakis. 2020.

 
 UniPose: Unified Human Pose Estimation in Single
Images and Videos. In CVPR .

 
 
 

 
 Belagiannis and
Zisserman (2017) 
 
Vasileios Belagiannis and
Andrew Zisserman. 2017.

 
 Recurrent human pose estimation. In
 FG .

 
 
 

 
 Benzine et al . (2020) 
 
Abdallah Benzine, Florian
Chabot, Bertrand Luvison, Quoc Cuong
Pham, and Catherine Achard.
2020.

 
 PandaNet: Anchor-Based Single-Shot Multi-Person 3D
Pose Estimation. In CVPR .

 
 
 

 
 Bertasius et al . (2019) 
 
Gedas Bertasius, Christoph
Feichtenhofer, Du Tran, Jianbo Shi,
and Lorenzo Torresani. 2019.

 
 Learning temporal pose estimation from
sparsely-labeled videos. In NeurIPS .

 
 
 

 
 Bogo et al . (2016) 
 
Federica Bogo, Angjoo
Kanazawa, Christoph Lassner, Peter
Gehler, Javier Romero, and Michael J.
Black. 2016.

 
 Keep it SMPL: Automatic Estimation of 3D Human
Pose and Shape from a Single Image. In ECCV .

 
 
 

 
 Bulat and
Tzimiropoulos (2016) 
 
Adrian Bulat and
Georgios Tzimiropoulos. 2016.

 
 Human pose estimation via convolutional part
heatmap regression. In ECCV .

 
 
 

 
 Burenius et al . (2013) 
 
M. Burenius, J.
Sullivan, and S. Carlsson.
2013.

 
 3D Pictorial Structures for Multiple View
Articulated Pose Estimation. In CVPR .

 
 
 

 
 Cai et al . (2019) 
 
Y. Cai, L. Ge,
J. Liu, J. Cai, T.
Cham, J. Yuan, and N. M.
Thalmann. 2019.

 
 Exploiting Spatial-Temporal Relationships for 3D
Pose Estimation via Graph Convolutional Networks. In
 ICCV .

 
 
 

 
 Cai et al . (2020) 
 
Yuanhao Cai, Zhicheng
Wang, Zhengxiong Luo, Binyi Yin,
Angang Du, Haoqian Wang,
Xinyu Zhou, Erjin Zhou,
Xiangyu Zhang, and Jian Sun.
2020.

 
 Learning Delicate Local Representations for
Multi-Person Pose Estimation. In arXiv preprint
arXiv:2003.04030 .

 
 
 

 
 Cao
et al . (2020) 
 
Zhe Cao, Hang Gao,
Karttikeya Mangalam, Qizhi Cai,
Minh Vo, and Jitendra Malik.
2020.

 
 Long-term human motion prediction with scene
context.

 
 In ECCV .

 
 
 

 
 Cao
et al . (2017) 
 
Zhe Cao, Tomas Simon,
Shih-En Wei, and Yaser Sheikh.
2017.

 
 Realtime multi-person 2d pose estimation using part
affinity fields. In CVPR .

 
 
 

 
 Carreira et al . (2016) 
 
Joao Carreira, Pulkit
Agrawal, Katerina Fragkiadaki, and
Jitendra Malik. 2016.

 
 Human pose estimation with iterative error
feedback. In CVPR .

 
 
 

 
 Chen and Ramanan (2017) 
 
Ching-Hang Chen and Deva
Ramanan. 2017.

 
 3D Human Pose Estimation = 2D Pose Estimation +
Matching. In CVPR .

 
 
 

 
 Chen et al . (2019) 
 
Ching-Hang Chen, Ambrish
Tyagi, Amit Agrawal, Dylan Drover,
Rohith MV, Stefan Stojanov, and
James M. Rehg. 2019.

 
 Unsupervised 3D Pose Estimation With Geometric
Self-Supervision. In CVPR .

 
 
 

 
 Chen
et al . (2020b) 
 
He Chen, Pengfei Guo,
Pengfei Li, Gim Hee Lee, and
Gregory Chirikjian. 2020b.

 
 Multi-person 3D Pose Estimation in Crowded Scenes
Based on Multi-View Geometry. In ECCV .

 
 
 

 
 Chen
et al . (2018a) 
 
Kenny Chen, Paolo
Gabriel, Abdulwahab Alasfour, Chenghao
Gong, Werner K Doyle, Orrin Devinsky,
Daniel Friedman, Patricia Dugan,
Lucia Melloni, Thomas Thesen,
et al . 2018a.

 
 Patient-specific pose estimation in clinical
environments. In JTEHM ,
Vol. 6. 1–11.

 
 
 

 
 Chen
et al . (2020a) 
 
Long Chen, Haizhou Ai,
Rui Chen, Zijie Zhuang, and
Shuang Liu. 2020a.

 
 Cross-View Tracking for Multi-Human 3D Pose
Estimation at Over 100 FPS. In CVPR .

 
 
 

 
 Chen
et al . (2021) 
 
Tianlang Chen, Chen Fang,
Xiaohui Shen, Yiheng Zhu,
Zhili Chen, and Jiebo Luo.
2021.

 
 Anatomy-aware 3D Human Pose Estimation with
Bone-based Pose Decomposition. In TCSVT ,
Vol. 32. 198–209.

 
 
 

 
 Chen
et al . (2020e) 
 
Ting Chen, Simon
Kornblith, Mohammad Norouzi, and
Geoffrey Hinton. 2020e.

 
 A simple framework for contrastive learning of
visual representations. In arXiv preprint
arXiv:2002.05709 .

 
 
 

 
 Chen
et al . (2020d) 
 
Weiming Chen, Zijie
Jiang, Hailin Guo, and Xiaoyang Ni.
2020d.

 
 Fall Detection Based on Key Points of
Human-Skeleton Using OpenPose. In Symmetry ,
Vol. 12. 744.

 
 
 

 
 Chen and Yuille (2014) 
 
Xianjie Chen and Alan L
Yuille. 2014.

 
 Articulated pose estimation by a graphical model
with image dependent pairwise relations. In
 NeurIPS .

 
 
 

 
 Chen
et al . (2017) 
 
Yu Chen, Chunhua Shen,
Xiu-Shen Wei, Lingqiao Liu, and
Jian Yang. 2017.

 
 Adversarial posenet: A structure-aware
convolutional network for human pose estimation. In
 ICCV .

 
 
 

 
 Chen
et al . (2020f) 
 
Yucheng Chen, Yingli
Tian, and Mingyi He. 2020f.

 
 Monocular human pose estimation: A survey of deep
learning-based methods. In CVIU ,
Vol. 192. 102897.

 
 
 

 
 Chen
et al . (2018b) 
 
Yilun Chen, Zhicheng
Wang, Yuxiang Peng, Zhiqiang Zhang,
Gang Yu, and Jian Sun.
2018b.

 
 Cascaded pyramid network for multi-person pose
estimation. In CVPR .

 
 
 

 
 Chen
et al . (2020c) 
 
Zerui Chen, Yan Huang,
Hongyuan Yu, Bin Xue, Ke
Han, Yiru Guo, and Liang Wang.
2020c.

 
 Towards Part-aware Monocular 3D Human Pose
Estimation: An Architecture Search Approach. In
 ECCV .

 
 
 

 
 Cheng et al . (2020) 
 
Bowen Cheng, Bin Xiao,
Jingdong Wang, Honghui Shi,
Thomas S Huang, and Lei Zhang.
2020.

 
 HigherHRNet: Scale-Aware Representation Learning
for Bottom-Up Human Pose Estimation. In CVPR .

 
 
 

 
 Cheng
et al . (2021a) 
 
Yu Cheng, Bo Wang,
Bo Yang, and Robby T. Tan.
2021a.

 
 Graph and Temporal Convolutional Networks for 3D
Multi-person Pose Estimation in Monocular Videos. In
 AAAI .

 
 
 

 
 Cheng
et al . (2021b) 
 
Yu Cheng, Bo Wang,
Bo Yang, and Robby T. Tan.
2021b.

 
 Monocular 3D Multi-Person Pose Estimation by
Integrating Top-Down and Bottom-Up Networks. In
 CVPR .

 
 
 

 
 Cheng et al . (2019) 
 
Y. Cheng, B. Yang,
B. Wang, Y. Wending, and
R. Tan. 2019.

 
 Occlusion-Aware Networks for 3D Human Pose
Estimation in Video. In ICCV .

 
 
 

 
 Cheng
et al . (2019) 
 
Yu Cheng, Bo Yang,
Bo Wang, Wending Yan, and
Robby T. Tan. 2019.

 
 Occlusion-Aware Networks for 3D Human Pose
Estimation in Video. In ICCV .

 
 
 

 
 Cho
et al . (2022) 
 
Junhyeong Cho, Kim
Youwang, and Tae-Hyun Oh.
2022.

 
 Cross-Attention of Disentangled Modalities for 3D
Human Mesh Recovery with Transformers. In ECCV .

 
 
 

 
 Choi
et al . (2021) 
 
Hongsuk Choi, Gyeongsik
Moon, Ju Yong Chang, and Kyoung Mu
Lee. 2021.

 
 Beyond Static Features for Temporally Consistent 3D
Human Pose and Shape From a Video. In CVPR .

 
 
 

 
 Choi
et al . (2020) 
 
Hongsuk Choi, Gyeongsik
Moon, and Kyoung Mu Lee.
2020.

 
 Pose2Mesh: Graph Convolutional Network for 3D Human
Pose and Mesh Recovery from a 2D Human Pose. In
 ECCV .

 
 
 

 
 Chou
et al . (2018) 
 
Chia-Jung Chou, Jui-Ting
Chien, and Hwann-Tzong Chen.
2018.

 
 Self adversarial training for human pose
estimation. In APSIPA ASC .

 
 
 

 
 Chu
et al . (2016) 
 
Xiao Chu, Wanli Ouyang,
Hongsheng Li, and Xiaogang Wang.
2016.

 
 Structured feature learning for pose estimation.
In CVPR .

 
 
 

 
 Chu
et al . (2017) 
 
Xiao Chu, Wei Yang,
Wanli Ouyang, Cheng Ma,
Alan L Yuille, and Xiaogang Wang.
2017.

 
 Multi-context attention for human pose estimation.
In CVPR .

 
 
 

 
 Ci
et al . (2019) 
 
H. Ci, C. Wang,
X. Ma, and Y. Wang.
2019.

 
 Optimizing Network Structure for 3D Human Pose
Estimation. In ICCV .

 
 
 

 
 Clever et al . (2020) 
 
Henry M. Clever, Zackory
Erickson, Ariel Kapusta, Greg Turk,
Karen Liu, and Charles C. Kemp.
2020.

 
 Bodies at Rest: 3D Human Pose and Shape Estimation
From a Pressure Image Using Synthetic Data. In
 CVPR .

 
 
 

 
 Dabral et al . (2018) 
 
Rishabh Dabral, Anurag
Mundhada, Uday Kusupati, Safeer Afaque,
Abhishek Sharma, and Arjun Jain.
2018.

 
 Learning 3D Human Pose from Structure and Motion.
In ECCV .

 
 
 

 
 Das et al . (2020) 
 
Srijan Das, Saurav
Sharma, Rui Dai, François
Brémond, and Monique Thonnat.
2020.

 
 VPN: Learning Video-Pose Embedding for Activities
of Daily Living. In ECCV .

 
 
 

 
 Debnath et al . (2018) 
 
Bappaditya Debnath, Mary
O’Brien, Motonori Yamaguchi, and
Ardhendu Behera. 2018.

 
 Adapting MobileNets for mobile based upper body
pose estimation. In AVSS .

 
 
 

 
 Doering
et al . (2018) 
 
Andreas Doering, Umar
Iqbal, and Juergen Gall.
2018.

 
 Joint flow: Temporal flow fields for multi person
tracking. In arXiv preprint arXiv:1805.04596 .

 
 
 

 
 Dong
et al . (2019) 
 
Junting Dong, Wen Jiang,
Qixing Huang, Hujun Bao, and
Xiaowei Zhou. 2019.

 
 Fast and Robust Multi-Person 3D Pose Estimation
From Multiple Views. In CVPR .

 
 
 

 
 Dong
et al . (2021) 
 
Zijian Dong, Jie Song,
Xu Chen, Chen Guo, and
Otmar Hilliges. 2021.

 
 Shape-aware Multi-Person Pose Estimation from
Multi-view Images. In ICCV .

 
 
 

 
 Drover et al . (2018) 
 
Dylan Drover, Ching-Hang
Chen, Amit Agrawal, Ambrish Tyagi, and
Cong Phuoc Huynh. 2018.

 
 Can 3d pose be learned from 2d projections alone?.
In ECCV .

 
 
 

 
 Eichner et al . (2012) 
 
Marcin Eichner, Manuel
Marin-Jimenez, Andrew Zisserman, and
Vittorio Ferrari. 2012.

 
 2d articulated human pose estimation and retrieval
in (almost) unconstrained still images. In IJCV ,
Vol. 99. 190–214.

 
 
 

 
 Elsken
et al . (2019) 
 
Thomas Elsken, Jan Hendrik
Metzen, and Frank Hutter.
2019.

 
 Neural Architecture Search: A Survey. In
 JMLR , Vol. 20.
1997–2017.

 
 
 

 
 Fabbri et al . (2020) 
 
Matteo Fabbri, Fabio
Lanzi, Simone Calderara, Stefano
Alletto, and Rita Cucchiara.
2020.

 
 Compressed Volumetric Heatmaps for Multi-Person 3D
Pose Estimation. In CVPR .

 
 
 

 
 Fan
et al . (2015) 
 
Xiaochuan Fan, Kang
Zheng, Yuewei Lin, and Song Wang.
2015.

 
 Combining local appearance and holistic view:
Dual-source deep neural networks for human pose estimation. In
 CVPR .

 
 
 

 
 Fang
et al . (2017) 
 
Hao-Shu Fang, Shuqin Xie,
Yu-Wing Tai, and Cewu Lu.
2017.

 
 Rmpe: Regional multi-person pose estimation. In
 ICCV .

 
 
 

 
 Fieraru et al . (2018) 
 
Mihai Fieraru, Anna
Khoreva, Leonid Pishchulin, and Bernt
Schiele. 2018.

 
 Learning to refine human pose estimation. In
 CVPR Workshops .

 
 
 

 
 Fisch and Clark (2020) 
 
Martin Fisch and Ronald
Clark. 2020.

 
 Orientation Keypoints for 6D Human Pose
Estimation. In arXiv preprint arXiv:2009.04930 .

 
 
 

 
 Georgakis et al . (2020) 
 
Georgios Georgakis, Ren
Li, Srikrishna Karanam, Terrence Chen,
Jana Kosecka, and Ziyan Wu.
2020.

 
 Hierarchical Kinematic Human Mesh Recovery. In
 ECCV .

 
 
 

 
 Gkioxari
et al . (2016) 
 
Georgia Gkioxari,
Alexander Toshev, and Navdeep Jaitly.
2016.

 
 Chained predictions using convolutional neural
networks. In ECCV .

 
 
 

 
 Gong et al . (2016) 
 
Wenjuan Gong, Xuena
Zhang, Jordi Gonzàlez, Andrews
Sobral, Thierry Bouwmans, Changhe Tu,
and El-hadi Zahzah. 2016.

 
 Human pose estimation from monocular images: A
comprehensive survey. In Sensors ,
Vol. 16. 1966.

 
 
 

 
 Goodfellow et al . (2014) 
 
Ian Goodfellow, Jean
Pouget-Abadie, Mehdi Mirza, Bing Xu,
David Warde-Farley, Sherjil Ozair,
Aaron Courville, and Yoshua Bengio.
2014.

 
 Generative adversarial nets. In
 NeurIPS .

 
 
 

 
 Gu et al . (2019) 
 
Yiwen Gu, Shreya Pandit,
Elham Saraee, Timothy Nordahl,
Terry Ellis, and Margrit Betke.
2019.

 
 Home-based physical therapy with an interactive
computer vision system. In ICCV Workshops .

 
 
 

 
 Guo
et al . (2018) 
 
Hengkai Guo, Tang Tang,
Guozhong Luo, Riwei Chen,
Yongchen Lu, and Linfu Wen.
2018.

 
 Multi-domain pose network for multi-person pose
estimation and tracking. In ECCV Workshops .

 
 
 

 
 Habibie et al . (2019) 
 
I. Habibie, W. Xu,
D. Mehta, G. Pons-Moll, and
C. Theobalt. 2019.

 
 In the Wild Human Pose Estimation Using Explicit 2D
Features and Intermediate 3D Representations. In
 CVPR .

 
 
 

 
 Hassan
et al . (2019) 
 
Mohamed Hassan, Vasileios
Choutas, Dimitrios Tzionas, and
Michael J. Black. 2019.

 
 Resolving 3D Human Pose Ambiguities with 3D
Scene Constraints. In ICCV .

 
 
 

 
 He
et al . (2016) 
 
Kaiming He, Xiangyu
Zhang, Shaoqing Ren, and Jian Sun.
2016.

 
 Deep residual learning for image recognition. In
 CVPR .

 
 
 

 
 Holte
et al . (2012) 
 
Michael B Holte, Cuong
Tran, Mohan M Trivedi, and Thomas B
Moeslund. 2012.

 
 Human pose estimation and activity recognition from
multi-view videos: Comparative explorations of recent developments. In
 IEEE J Sel Top Signal Process ,
Vol. 6. 538–552.

 
 
 

 
 Hua
et al . (2023) 
 
Yilei Hua, Wenhan Wu,
Ce Zheng, Aidong Lu,
Mengyuan Liu, Chen Chen, and
Shiqian Wu. 2023.

 
 Part Aware Contrastive Learning for Self-Supervised
Action Recognition. In IJCAI .

 
 
 

 
 Huang et al . (2020b) 
 
Congzhentao Huang, Shuai
Jiang, Yang Li, Ziyue Zhang,
Jason Traish, Chen Deng,
Sam Ferguson, and Richard Yi Da Xu.
2020b.

 
 End-to-end Dynamic Matching Network for Multi-view
Multi-person 3d Pose Estimation. In ECCV .

 
 
 

 
 Huang
et al . (2020c) 
 
Fuyang Huang, Ailing
Zeng, Minhao Liu, Qiuxia Lai, and
Qiang Xu. 2020c.

 
 DeepFuse: An IMU-Aware Network for Real-Time 3D
Human Pose Estimation from Multi-View Image. In
 WACV .

 
 
 

 
 Huang
et al . (2020d) 
 
Junjie Huang, Zheng Zhu,
Feng Guo, and Guan Huang.
2020d.

 
 The Devil is in the Details: Delving into Unbiased
Data Processing for Human Pose Estimation. In
 CVPR .

 
 
 

 
 Huang
et al . (2017) 
 
Shaoli Huang, Mingming
Gong, and Dacheng Tao. 2017.

 
 A coarse-fine network for keypoint localization.
In ICCV .

 
 
 

 
 Huang et al . (2020a) 
 
Xiaofei Huang, Nihang Fu,
Shuangjun Liu, Kathan Vyas,
Amirreza Farnoosh, and Sarah
Ostadabbas. 2020a.

 
 Invariant Representation Learning for Infant Pose
Estimation with Small Data. In arXiv preprint
arXiv:2010.06100 .

 
 
 

 
 Huang et al . (2018) 
 
Yinghao Huang, Manuel
Kaufmann, Emre Aksan, Michael J. Black,
Otmar Hilliges, and Gerard Pons-Moll.
2018.

 
 Deep Inertial Poser: Learning to Reconstruct Human
Pose from Sparse Inertial Measurements in Real Time. In
 ACM TOG .

 
 
 

 
 Insafutdinov et al . (2017) 
 
Eldar Insafutdinov,
Mykhaylo Andriluka, Leonid Pishchulin,
Siyu Tang, Evgeny Levinkov,
Bjoern Andres, and Bernt Schiele.
2017.

 
 Arttrack: Articulated multi-person tracking in the
wild. In CVPR .

 
 
 

 
 Insafutdinov et al . (2016) 
 
Eldar Insafutdinov, Leonid
Pishchulin, Bjoern Andres, Mykhaylo
Andriluka, and Bernt Schiele.
2016.

 
 Deepercut: A deeper, stronger, and faster
multi-person pose estimation model. In ECCV .

 
 
 

 
 Ionescu et al . (2014) 
 
C. Ionescu, D.
Papava, V. Olaru, and C.
Sminchisescu. 2014.

 
 Human3.6M: Large Scale Datasets and Predictive
Methods for 3D Human Sensing in Natural Environments. In
 IEEE TPAMI , Vol. 36.
1325–1339.

 
 
 

 
 Iqbal and Gall (2016) 
 
Umar Iqbal and Juergen
Gall. 2016.

 
 Multi-person pose estimation with local
joint-to-person associations. In ECCV .

 
 
 

 
 Isogawa
et al . (2020) 
 
Mariko Isogawa, Ye Yuan,
Matthew O’Toole, and Kris M. Kitani.
2020.

 
 Optical Non-Line-of-Sight Physics-Based 3D Human
Pose Estimation. In CVPR .

 
 
 

 
 Jahangiri and
Yuille (2017) 
 
Ehsan Jahangiri and
Alan L Yuille. 2017.

 
 Generating multiple diverse hypotheses for human 3d
pose consistent with 2d joint detections. In ICCV
Workshops .

 
 
 

 
 Jain
et al . (2014) 
 
Arjun Jain, Jonathan
Tompson, Yann LeCun, and Christoph
Bregler. 2014.

 
 Modeep: A deep learning framework using motion
features for human pose estimation. In ACCV .

 
 
 

 
 Jain
et al . (2019) 
 
Naman Jain, Sahil Shah,
Abhishek Kumar, and Arjun Jain.
2019.

 
 On the Robustness of Human Pose Estimation. In
 CVPR Workshops .

 
 
 

 
 Jhuang
et al . (2013) 
 
H. Jhuang, J. Gall,
S. Zuffi, C. Schmid, and
M. J. Black. 2013.

 
 Towards understanding action recognition. In
 ICCV .

 
 
 

 
 JI
et al . (2020) 
 
Xiaopeng JI, Qi FANG,
Junting DONG, Qing SHUAI,
Wen JIANG, and Xiaowei ZHOU.
2020.

 
 A survey on monocular 3D human pose estimation. In
 Virtual Reality \ Intelligent Hardware ,
Vol. 2. 471–500.

 
 
 

 
 Ji and Liu (2009) 
 
Xiaofei Ji and Honghai
Liu. 2009.

 
 Advances in view-invariant human motion analysis: a
review. In IEEE TSMC , Vol. 40.
13–24.

 
 
 

 
 Jiang
et al . (2019) 
 
Haiyong Jiang, Jianfei
Cai, and Jianmin Zheng.
2019.

 
 Skeleton-Aware 3D Human Shape Reconstruction From
Point Clouds. In ICCV .

 
 
 

 
 Jiang et al . (2020) 
 
Wen Jiang, Nikos
Kolotouros, Georgios Pavlakos, Xiaowei
Zhou, and Kostas Daniilidis.
2020.

 
 Coherent Reconstruction of Multiple Humans From a
Single Image. In CVPR .

 
 
 

 
 Jin et al . (2020a) 
 
Sheng Jin, Wentao Liu,
Enze Xie, Wenhai Wang,
Chen Qian, Wanli Ouyang, and
Ping Luo. 2020a.

 
 Differentiable Hierarchical Graph Grouping for
Multi-Person Pose Estimation. In arXiv preprint
arXiv:2007.11864 .

 
 
 

 
 Jin et al . (2020b) 
 
Sheng Jin, Lumin Xu,
Jin Xu, Can Wang, Wentao
Liu, Chen Qian, Wanli Ouyang, and
Ping Luo. 2020b.

 
 Whole-Body Human Pose Estimation in the Wild. In
 arXiv preprint arXiv:2007.11858 .

 
 
 

 
 Johnson and
Everingham (2010) 
 
Sam Johnson and Mark
Everingham. 2010.

 
 Clustered Pose and Nonlinear Appearance Models for
Human Pose Estimation. In BMVC .

 
 
 

 
 Johnson and
Everingham (2011) 
 
Sam Johnson and Mark
Everingham. 2011.

 
 Learning effective human pose estimation from
inaccurate annotation. In CVPR .

 
 
 

 
 Joo
et al . (2017) 
 
Hanbyul Joo, Tomas Simon,
Xulong Li, Hao Liu, Lei
Tan, Lin Gui, Sean Banerjee,
Timothy Scott Godisart, Bart Nabbe,
Iain Matthews, Takeo Kanade,
Shohei Nobuhara, and Yaser Sheikh.
2017.

 
 Panoptic Studio: A Massively Multiview System for
Social Interaction Capture. In IEEE TPAMI .
3334–3342.

 
 
 

 
 Joo
et al . (2018) 
 
H. Joo, T. Simon,
and Y. Sheikh. 2018.

 
 Total Capture: A 3D Deformation Model for Tracking
Faces, Hands, and Bodies. In CVPR .

 
 
 

 
 Kadkhodamohammadi et al . (2017) 
 
A. Kadkhodamohammadi, A.
Gangi, M. de Mathelin, and N.
Padoy. 2017.

 
 A Multi-view RGB-D Approach for Human Pose
Estimation in Operating Rooms. In WACV .

 
 
 

 
 Kavan (2014) 
 
Ladislav Kavan.
2014.

 
 Part I: direct skinning methods and deformation
primitives. In ACM SIGGRAPH .

 
 
 

 
 Ke
et al . (2018) 
 
Lipeng Ke, Ming-Ching
Chang, Honggang Qi, and Siwei Lyu.
2018.

 
 Multi-scale structure-aware network for human pose
estimation. In ECCV .

 
 
 

 
 Khan et al . (2021) 
 
Salman Khan, Muzammal
Naseer, Munawar Hayat, Syed Waqas Zamir,
Fahad Shahbaz Khan, and Mubarak Shah.
2021.

 
 Transformers in vision: A survey. In
 arXiv preprint arXiv:2101.01169 .

 
 
 

 
 Kocabas
et al . (2020) 
 
Muhammed Kocabas, Nikos
Athanasiou, and Michael J Black.
2020.

 
 VIBE: Video inference for human body pose and shape
estimation. In CVPR .

 
 
 

 
 Kocabas
et al . (2018) 
 
Muhammed Kocabas, Salih
Karagoz, and Emre Akbas.
2018.

 
 Multiposenet: Fast multi-person pose estimation
using pose residual network. In ECCV .

 
 
 

 
 Kocabas
et al . (2019) 
 
Muhammed Kocabas, Salih
Karagoz, and Emre Akbas.
2019.

 
 Self-Supervised Learning of 3D Human Pose Using
Multi-View Geometry. In CVPR .

 
 
 

 
 Kolotouros et al . (2019) 
 
N. Kolotouros, G.
Pavlakos, M. Black, and K.
Daniilidis. 2019.

 
 Learning to Reconstruct 3D Human Pose and Shape via
Model-Fitting in the Loop. In ICCV .

 
 
 

 
 Kolotouros et al . (2019) 
 
Nikos Kolotouros, Georgios
Pavlakos, and Kostas Daniilidis.
2019.

 
 Convolutional Mesh Regression for Single-Image
Human Shape Reconstruction. In CVPR .

 
 
 

 
 Kolotouros et al . (2021) 
 
Nikos Kolotouros, Georgios
Pavlakos, Dinesh Jayaraman, and Kostas
Daniilidis. 2021.

 
 Probabilistic Modeling for Human Mesh Recovery. In
 ICCV .

 
 
 

 
 Kreiss
et al . (2019) 
 
Sven Kreiss, Lorenzo
Bertoni, and Alexandre Alahi.
2019.

 
 Pifpaf: Composite fields for human pose
estimation. In CVPR .

 
 
 

 
 Krizhevsky
et al . (2012) 
 
Alex Krizhevsky, Ilya
Sutskever, and Geoffrey E Hinton.
2012.

 
 Imagenet classification with deep convolutional
neural networks. In NeurIPS .

 
 
 

 
 Kundu et al . (2020a) 
 
Jogendra Nath Kundu,
Ambareesh Revanur, Govind Vitthal
Waghmare, Rahul Mysore Venkatesh, and
R. Venkatesh Babu. 2020a.

 
 Unsupervised Cross-Modal Alignment for Multi-Person
3D Pose Estimation. In ECCV .

 
 
 

 
 Kundu et al . (2020b) 
 
Jogendra Nath Kundu,
Siddharth Seth, Varun Jampani,
Mugalodi Rakesh, R. Venkatesh Babu, and
Anirban Chakraborty. 2020b.

 
 Self-Supervised 3D Human Pose Estimation via Part
Guided Novel Image Synthesis. In CVPR .

 
 
 

 
 Kundu et al . (2020c) 
 
Jogendra Nath Kundu,
Siddharth Seth, MV Rahul,
Mugalodi Rakesh, Venkatesh Babu
Radhakrishnan, and Anirban Chakraborty.
2020c.

 
 Kinematic-Structure-Preserved Representation for
Unsupervised 3D Human Pose Estimation. In AAAI .

 
 
 

 
 Lassner et al . (2017) 
 
Christoph Lassner, Javier
Romero, Martin Kiefel, Federica Bogo,
Michael J. Black, and Peter V. Gehler.
2017.

 
 Unite the People: Closing the Loop Between 3D and
2D Human Representations. In CVPR .

 
 
 

 
 Lee et al . (2018) 
 
Kyoungoh Lee, Inwoong
Lee, and Sanghoon Lee. 2018.

 
 Propagating LSTM: 3D Pose Estimation based on Joint
Interdependency. In ECCV .

 
 
 

 
 Li and Lee (2019) 
 
Chen Li and Gim Hee
Lee. 2019.

 
 Generating Multiple Hypotheses for 3D Human Pose
Estimation With Mixture Density Network. In
 CVPR .

 
 
 

 
 Li
et al . (2021a) 
 
Jiefeng Li, Siyuan Bian,
Ailing Zeng, Can Wang,
Bo Pang, Wentao Liu, and
Cewu Lu. 2021a.

 
 Human pose regression with residual log-likelihood
estimation. In ICCV .

 
 
 

 
 Li
et al . (2020b) 
 
Jiefeng Li, Can Wang,
Wentao Liu, Chen Qian, and
Cewu Lu. 2020b.

 
 HMOR: Hierarchical Multi-Person Ordinal Relations
for Monocular Multi-Person 3D Pose Estimation. In
 ECCV .

 
 
 

 
 Li
et al . (2019b) 
 
Jiefeng Li, Can Wang,
Hao Zhu, Yihuan Mao,
Hao-Shu Fang, and Cewu Lu.
2019b.

 
 Crowdpose: Efficient crowded scenes pose estimation
and a new benchmark. In ICCV .

 
 
 

 
 Li
et al . (2021d) 
 
Jiefeng Li, Chao Xu,
Zhicun Chen, Siyuan Bian,
Lixin Yang, and Cewu Lu.
2021d.

 
 HybrIK: A Hybrid Analytical-Neural Inverse
Kinematics Solution for 3D Human Pose and Shape Estimation. In
 CVPR .

 
 
 

 
 Li
et al . (2021c) 
 
Ke Li, Shijie Wang,
Xiang Zhang, Yifan Xu,
Weijian Xu, and Zhuowen Tu.
2021c.

 
 Pose Recognition with Cascade Transformers. In
 CVPR .

 
 
 

 
 Li and Chan (2014) 
 
Sijin Li and Antoni B.
Chan. 2014.

 
 3D Human Pose Estimation from Monocular Images with
Deep Convolutional Neural Network. In ACCV .

 
 
 

 
 Li et al . (2014) 
 
Sijin Li, Zhi-Qiang Liu,
and Antoni B Chan. 2014.

 
 Heterogeneous multi-task learning for human pose
estimation with deep convolutional neural network. In
 CVPR Workshops .

 
 
 

 
 Li et al . (2015) 
 
Sijin Li, Weichen Zhang,
and Antoni B. Chan. 2015.

 
 Maximum-Margin Structured Learning With Deep
Networks for 3D Human Pose Estimation. In ICCV .

 
 
 

 
 Li
et al . (2022a) 
 
Wenhao Li, Hong Liu,
Hao Tang, Pichao Wang, and
Luc Van Gool. 2022a.

 
 MHFormer: Multi-Hypothesis Transformer for 3D Human
Pose Estimation. In CVPR .

 
 
 

 
 Li et al . (2019a) 
 
Wenbo Li, Zhicheng Wang,
Binyi Yin, Qixiang Peng,
Yuming Du, Tianzi Xiao,
Gang Yu, Hongtao Lu,
Yichen Wei, and Jian Sun.
2019a.

 
 Rethinking on multi-stage networks for human pose
estimation. In arXiv preprint arXiv:1901.00148 .

 
 
 

 
 Li
et al . (2021b) 
 
Yizhuo Li, Miao Hao,
Zonglin Di, Nitesh Bharadwaj Gundavarapu,
and Xiaolong Wang. 2021b.

 
 Test-time personalization with a transformer for
human pose estimation. In NeurIPS .

 
 
 

 
 Li et al . (2019) 
 
Yining Li, Chen Huang,
and Chen Change Loy. 2019.

 
 Dense Intrinsic Appearance Flow for Human Pose
Transfer. In CVPR .

 
 
 

 
 Li et al . (2022b) 
 
Yanjie Li, Sen Yang,
Peidong Liu, Shoukui Zhang,
Yunxiao Wang, Zhicheng Wang,
Wankou Yang, and Shu-Tao Xia.
2022b.

 
 SimCC: A Simple Coordinate Classification
Perspective for Human Pose Estimation. In ECCV .

 
 
 

 
 Li
et al . (2021e) 
 
Yanjie Li, Shoukui Zhang,
Zhicheng Wang, Sen Yang,
Wankou Yang, Shu-Tao Xia, and
Erjin Zhou. 2021e.

 
 TokenPose: Learning Keypoint Tokens for Human Pose
Estimation. In ICCV .

 
 
 

 
 Li
et al . (2020a) 
 
Zhongguo Li, Anders
Heyden, and Magnus Oskarsson.
2020a.

 
 A novel joint points and silhouette-based method to
estimate 3D human pose and shape. In arXiv
preprint arXiv:2012.06109 .

 
 
 

 
 Li
et al . (2019) 
 
Z. Li, X. Wang,
F. Wang, and P. Jiang.
2019.

 
 On Boosting Single-Frame 3D Human Pose Estimation
via Monocular Videos. In ICCV .

 
 
 

 
 Liang and Lin (2019) 
 
Junbang Liang and
Ming C. Lin. 2019.

 
 Shape-Aware Human Pose and Shape Reconstruction
Using Multi-View Images. In ICCV .

 
 
 

 
 Lifshitz
et al . (2016) 
 
Ita Lifshitz, Ethan
Fetaya, and Shimon Ullman.
2016.

 
 Human pose estimation using deep consensus voting.
In ECCV .

 
 
 

 
 Lin
et al . (2021a) 
 
Kevin Lin, Lijuan Wang,
and Zicheng Liu. 2021a.

 
 End-to-end human pose and mesh reconstruction with
transformers. In CVPR .

 
 
 

 
 Lin
et al . (2021b) 
 
Kevin Lin, Lijuan Wang,
and Zicheng Liu. 2021b.

 
 Mesh Graphormer. In
 ICCV .

 
 
 

 
 Lin et al . (2014) 
 
Tsung-Yi Lin, Michael
Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan,
Piotr Dollár, and C Lawrence
Zitnick. 2014.

 
 Microsoft coco: Common objects in context. In
 ECCV .

 
 
 

 
 Lin et al . (2020) 
 
Weiyao Lin, Huabin Liu,
Shizhan Liu, Yuxi Li,
Rui Qian, Tao Wang, Ning
Xu, Hongkai Xiong, Guo-Jun Qi, and
Nicu Sebe. 2020.

 
 Human in events: A large-scale benchmark for
human-centric video analysis in complex events. In
 arXiv preprint arXiv:2005.04490 .

 
 
 

 
 Liu
et al . (2021b) 
 
Huajun Liu, Fuqiang Liu,
Xinyi Fan, and Dong Huang.
2021b.

 
 Polarized self-attention: towards high-quality
pixel-wise regression. In arXiv preprint
arXiv:2107.00782 .

 
 
 

 
 Liu
et al . (2019) 
 
Jian Liu, Naveed Akhtar,
and Ajmal Mian. 2019.

 
 Adversarial Attack on Skeleton-based Human Action
Recognition. In arXiv preprint arXiv:1909.06500 .

 
 
 

 
 Liu et al . (2020b) 
 
Jingyuan Liu, Hongbo Fu,
and Chiew-Lan Tai. 2020b.

 
 PoseTween: Pose-driven Tween Animation. In
 ACM UIST .

 
 
 

 
 Liu
et al . (2020a) 
 
Kenkun Liu, Rongqi Ding,
Zhiming Zou, Le Wang, and
Wei Tang. 2020a.

 
 A Comprehensive Study of Weight Sharing in Graph
Networks for 3D Human Pose Estimation. In ECCV .

 
 
 

 
 Liu
et al . (2020c) 
 
Ruixu Liu, Ju Shen,
He Wang, Chen Chen,
Sen-ching Cheung, and Vijayan Asari.
2020c.

 
 Attention Mechanism Exploits Temporal Contexts:
Real-Time 3D Human Pose Reconstruction. In CVPR .

 
 
 

 
 Liu
et al . (2021a) 
 
Zhenguang Liu, Haoming
Chen, Runyang Feng, Shuang Wu,
Shouling Ji, Bailin Yang, and
Xun Wang. 2021a.

 
 Deep dual consecutive network for human pose
estimation. In CVPR .

 
 
 

 
 Liu
et al . (2022) 
 
Zhenguang Liu, Runyang
Feng, Haoming Chen, Shuang Wu,
Yixing Gao, Yunjun Gao, and
Xiang Wang. 2022.

 
 Temporal Feature Alignment and Mutual Information
Maximization for Video-Based Human Pose Estimation. In
 CVPR .

 
 
 

 
 Liu
et al . (2015) 
 
Zhao Liu, Jianke Zhu,
Jiajun Bu, and Chun Chen.
2015.

 
 A survey of human pose estimation: the body parts
parsing based methods. In JVCIR ,
Vol. 32. 10–19.

 
 
 

 
 Long
et al . (2015) 
 
Jonathan Long, Evan
Shelhamer, and Trevor Darrell.
2015.

 
 Fully Convolutional Networks for Semantic
Segmentation. In CVPR .

 
 
 

 
 Loper et al . (2015) 
 
Matthew Loper, Naureen
Mahmood, Javier Romero, Gerard
Pons-Moll, and Michael J. Black.
2015.

 
 SMPL: A Skinned Multi-Person Linear Model. In
 ACM TOG , Vol. 34.
1–16.

 
 
 

 
 Lu et al . (2020) 
 
Mandy Lu, Kathleen
Poston, Adolf Pfefferbaum, Edith V
Sullivan, Li Fei-Fei, Kilian M Pohl,
Juan Carlos Niebles, and Ehsan Adeli.
2020.

 
 Vision-based Estimation of MDS-UPDRS Gait Scores
for Assessing Parkinson’s Disease Motor Severity. In
 arXiv preprint arXiv:2007.08920 .

 
 
 

 
 Luo et al . (2018) 
 
Yue Luo, Jimmy Ren,
Zhouxia Wang, Wenxiu Sun,
Jinshan Pan, Jianbo Liu,
Jiahao Pang, and Liang Lin.
2018.

 
 Lstm pose machines. In
 CVPR .

 
 
 

 
 Luo
et al . (2021) 
 
Zhengxiong Luo, Zhicheng
Wang, Yan Huang, Liang Wang,
Tieniu Tan, and Erjin Zhou.
2021.

 
 Rethinking the heatmap regression for bottom-up
human pose estimation. In CVPR .

 
 
 

 
 Luvizon
et al . (2018) 
 
Diogo C Luvizon, David
Picard, and Hedi Tabia.
2018.

 
 2d/3d pose estimation and action recognition using
multitask deep learning. In CVPR .

 
 
 

 
 Luvizon
et al . (2019) 
 
Diogo C Luvizon, Hedi
Tabia, and David Picard.
2019.

 
 Human pose regression by combining indirect part
detection and contextual information. In Computers
 Graphics , Vol. 85. 15–22.

 
 
 

 
 Ma et al . (2021) 
 
Haoyu Ma, Liangjian Chen,
Deying Kong, Zhe Wang,
Xingwei Liu, Hao Tang,
Xiangyi Yan, Yusheng Xie,
Shih-Yao Lin, and Xiaohui Xie.
2021.

 
 Transfusion: Cross-view fusion with transformer for
3d human pose estimation. In BMVC .

 
 
 

 
 Ma et al . (2022) 
 
Haoyu Ma, Zhe Wang,
Yifei Chen, Deying Kong,
Liangjian Chen, Xingwei Liu,
Xiangyi Yan, Hao Tang, and
Xiaohui Xie. 2022.

 
 PPT: token-Pruned Pose Transformer for monocular
and multi-view human pose estimation. In ECCV .

 
 
 

 
 Madhu et al . (2020) 
 
Prathmesh Madhu, Angel
Villar-Corrales, Ronak Kosti, Torsten
Bendschus, Corinna Reinhardt, Peter
Bell, Andreas Maier, and Vincent
Christlein. 2020.

 
 Enhancing Human Pose Estimation in Ancient Vase
Paintings via Perceptually-grounded Style Transfer Learning. In
 arXiv preprint arXiv:2012.05616 .

 
 
 

 
 Mahmood et al . (2019) 
 
Naureen Mahmood, Nima
Ghorbani, Nikolaus F. Troje, Gerard
Pons-Moll, and Michael J. Black.
2019.

 
 AMASS: Archive of Motion Capture as Surface
Shapes. In ICCV .

 
 
 

 
 Mao
et al . (2021) 
 
Weian Mao, Yongtao Ge,
Chunhua Shen, Zhi Tian,
Xinlong Wang, and Zhibin Wang.
2021.

 
 Tfpose: Direct human pose estimation with
transformers. In arXiv preprint
arXiv:2103.15320 .

 
 
 

 
 Mao et al . (2022) 
 
Weian Mao, Yongtao Ge,
Chunhua Shen, Zhi Tian,
Xinlong Wang, Zhibin Wang, and
Anton van den Hengel. 2022.

 
 Poseur: Direct Human Pose Regression with
Transformers. In ECCV .

 
 
 

 
 Markovitz et al . (2020) 
 
Amir Markovitz, Gilad
Sharir, Itamar Friedman, Lihi
Zelnik-Manor, and Shai Avidan.
2020.

 
 Graph Embedded Pose Clustering for Anomaly
Detection. In CVPR .

 
 
 

 
 Martinez et al . (2017) 
 
Julieta Martinez, Rayat
Hossain, Javier Romero, and James J.
Little. 2017.

 
 A simple yet effective baseline for 3d human pose
estimation. In ICCV .

 
 
 

 
 Mehta et al . (2017) 
 
D. Mehta, H. Rhodin,
D. Casas, P. Fua, O.
Sotnychenko, W. Xu, and C.
Theobalt. 2017.

 
 Monocular 3D Human Pose Estimation in the Wild
Using Improved CNN Supervision. In 3DV .

 
 
 

 
 Mehta et al . (2020) 
 
Dushyant Mehta, Oleksandr
Sotnychenko, Franziska Mueller, Weipeng
Xu, Mohamed Elgharib, Pascal Fua,
Hans-Peter Seidel, Helge Rhodin,
Gerard Pons-Moll, and Christian
Theobalt. 2020.

 
 XNect: Real-time Multi-Person 3D Motion Capture
with a Single RGB Camera. In ACM TOG ,
Vol. 39. 82–1.

 
 
 

 
 Mehta et al . (2018) 
 
Dushyant Mehta, Oleksandr
Sotnychenko, Franziska Mueller, Weipeng
Xu, Srinath Sridhar, Gerard Pons-Moll,
and Christian Theobalt. 2018.

 
 Single-Shot Multi-Person 3D Pose Estimation From
Monocular RGB. In 3DV .

 
 
 

 
 Mehta et al . (2017) 
 
Dushyant Mehta, Srinath
Sridhar, Oleksandr Sotnychenko, Helge
Rhodin, Mohammad Shafiei, Hans-Peter
Seidel, et al . 2017.

 
 Vnect: Real-time 3d human pose estimation with a
single rgb camera. In ACM TOG ,
Vol. 36. 1–14.

 
 
 

 
 Micilotta
et al . (2006) 
 
Antonio S Micilotta,
Eng-Jon Ong, and Richard Bowden.
2006.

 
 Real-time upper body detection and 3D pose
estimation in monoscopic images. In ECCV .

 
 
 

 
 Mitra
et al . (2020) 
 
Rahul Mitra, Nitesh B.
Gundavarapu, Abhishek Sharma, and Arjun
Jain. 2020.

 
 Multiview-Consistent Semi-Supervised Learning for
3D Human Pose Estimation. In CVPR .

 
 
 

 
 Moeslund and
Granum (2001) 
 
Thomas B Moeslund and
Erik Granum. 2001.

 
 A survey of computer vision-based human motion
capture. In CVIU , Vol. 81.
231–268.

 
 
 

 
 Moeslund
et al . (2006) 
 
Thomas B Moeslund, Adrian
Hilton, and Volker Krüger.
2006.

 
 A survey of advances in vision-based human motion
capture and analysis. In CVIU ,
Vol. 104. 90–126.

 
 
 

 
 Moon
et al . (2019a) 
 
Gyeongsik Moon, Juyong
Chang, and Kyoung Mu Lee.
2019a.

 
 Camera Distance-aware Top-down Approach for 3D
Multi-person Pose Estimation from a Single RGB Image. In
 ICCV .

 
 
 

 
 Moon
et al . (2019b) 
 
Gyeongsik Moon, Ju Yong
Chang, and Kyoung Mu Lee.
2019b.

 
 Posefix: Model-agnostic general human pose
refinement network. In CVPR .

 
 
 

 
 Moon and Lee (2020) 
 
Gyeongsik Moon and
Kyoung Mu Lee. 2020.

 
 I2L-MeshNet: Image-to-Lixel Prediction Network for
Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image. In
 ECCV .

 
 
 

 
 Moreno-Noguer (2017) 
 
Francesc Moreno-Noguer.
2017.

 
 3d human pose estimation from a single image via
distance matrix regression. In CVPR .

 
 
 

 
 Munea et al . (2020) 
 
Tewodros Legesse Munea,
Yalew Zelalem Jembre, Halefom Tekle
Weldegebriel, Longbiao Chen, Chenxi
Huang, and Chenhui Yang.
2020.

 
 The Progress of Human Pose Estimation: A Survey and
Taxonomy of Models Applied in 2D Human Pose Estimation. In
 IEEE Access , Vol. 8.
133330–133348.

 
 
 

 
 Newell
et al . (2017) 
 
Alejandro Newell, Zhiao
Huang, and Jia Deng. 2017.

 
 Associative embedding: End-to-end learning for
joint detection and grouping. In NeurIPS .

 
 
 

 
 Newell
et al . (2016) 
 
Alejandro Newell, Kaiyu
Yang, and Jia Deng. 2016.

 
 Stacked hourglass networks for human pose
estimation. In ECCV .

 
 
 

 
 Nibali
et al . (2018) 
 
Aiden Nibali, Zhen He,
Stuart Morgan, and Luke Prendergast.
2018.

 
 Numerical coordinate regression with convolutional
neural networks. In arXiv preprint
arXiv:1801.07372 .

 
 
 

 
 Nie
et al . (2017) 
 
B. X. Nie, P. Wei,
and S. Zhu. 2017.

 
 Monocular 3D Human Pose Estimation by Predicting
Depth on Joints. In ICCV .

 
 
 

 
 Nie et al . (2020) 
 
Qiang Nie, Ziwei Liu,
and Yunhui Liu. 2020.

 
 Unsupervised Human 3D Pose Representation with
Viewpoint and Pose Disentanglement. In ECCV .

 
 
 

 
 Nie
et al . (2019) 
 
Xuecheng Nie, Jiashi
Feng, Jianfeng Zhang, and Shuicheng
Yan. 2019.

 
 Single-Stage Multi-Person Pose Machines. In
 ICCV .

 
 
 

 
 Omran et al . (2018) 
 
Mohamed Omran, Christoph
Lassner, Gerard Pons-Moll, Peter V.
Gehler, and Bernt Schiele.
2018.

 
 Neural Body Fitting: Unifying Deep Learning and
Model-Based Human Pose and Shape Estimation. In
 3DV .

 
 
 

 
 Osman
et al . (2020) 
 
Ahmed A A Osman, Timo
Bolkart, and Michael J. Black.
2020.

 
 STAR: A Spare Trained Articulated Human Body
Regressor. In ECCV .

 
 
 

 
 Panteleris and
Argyros (2021) 
 
Paschalis Panteleris and
Antonis Argyros. 2021.

 
 PE-former: Pose Estimation Transformer. In
 arXiv preprint arXiv:2112.04981 .

 
 
 

 
 Papandreou et al . (2018) 
 
George Papandreou, Tyler
Zhu, Liang-Chieh Chen, Spyros Gidaris,
Jonathan Tompson, and Kevin Murphy.
2018.

 
 Personlab: Person pose estimation and instance
segmentation with a bottom-up, part-based, geometric embedding model. In
 ECCV .

 
 
 

 
 Papandreou et al . (2017) 
 
George Papandreou, Tyler
Zhu, Nori Kanazawa, Alexander Toshev,
Jonathan Tompson, Chris Bregler, and
Kevin Murphy. 2017.

 
 Towards Accurate Multi-Person Pose Estimation in
the Wild. In CVPR .

 
 
 

 
 Patel
et al . (2020) 
 
Chaitanya Patel,
Zhouyingcheng Liao, and Gerard
Pons-Moll. 2020.

 
 TailorNet: Predicting Clothing in 3D as a Function
of Human Pose, Shape and Garment Style. In CVPR .

 
 
 

 
 Pavlakos et al . (2019) 
 
Georgios Pavlakos,
Vasileios Choutas, Nima Ghorbani,
Timo Bolkart, Ahmed A. A. Osman,
Dimitrios Tzionas, and Michael J.
Black. 2019.

 
 Expressive Body Capture: 3D Hands, Face, and Body
from a Single Image. In CVPR .

 
 
 

 
 Pavlakos
et al . (2018a) 
 
Georgios Pavlakos, Xiaowei
Zhou, and Kostas Daniilidis.
2018a.

 
 Ordinal Depth Supervision for 3D Human Pose
Estimation. In CVPR .

 
 
 

 
 Pavlakos et al . (2017a) 
 
Georgios Pavlakos, Xiaowei
Zhou, Konstantinos G Derpanis, and
Kostas Daniilidis. 2017a.

 
 Coarse-to-Fine Volumetric Prediction for
Single-Image 3D Human Pose. In CVPR .

 
 
 

 
 Pavlakos et al . (2017b) 
 
Georgios Pavlakos, Xiaowei
Zhou, Konstantinos G. Derpanis, and
Kostas Daniilidis. 2017b.

 
 Harvesting Multiple Views for Marker-Less 3D Human
Pose Annotations. In CVPR .

 
 
 

 
 Pavlakos
et al . (2018b) 
 
Georgios Pavlakos, Luyang
Zhu, Xiaowei Zhou, and Kostas
Daniilidis. 2018b.

 
 Learning to Estimate 3D Human Pose and Shape from
a Single Color Image. In CVPR .

 
 
 

 
 Pavllo et al . (2019) 
 
Dario Pavllo, Christoph
Feichtenhofer, David Grangier, and
Michael Auli. 2019.

 
 3D human pose estimation in video with temporal
convolutions and semi-supervised training. In
 CVPR .

 
 
 

 
 Peng
et al . (2018) 
 
Xi Peng, Zhiqiang Tang,
Fei Yang, Rogerio S Feris, and
Dimitris Metaxas. 2018.

 
 Jointly optimize data augmentation and network
training: Adversarial data augmentation in human pose estimation. In
 CVPR .

 
 
 

 
 Pfister
et al . (2015) 
 
Tomas Pfister, James
Charles, and Andrew Zisserman.
2015.

 
 Flowing convnets for human pose estimation in
videos. In ICCV .

 
 
 

 
 Pfister et al . (2014) 
 
Tomas Pfister, Karen
Simonyan, James Charles, and Andrew
Zisserman. 2014.

 
 Deep convolutional neural networks for efficient
pose estimation in gesture videos. In ACCV .

 
 
 

 
 Pirinen et al . (2019) 
 
Aleksis Pirinen, Erik
Gärtner, and Cristian Sminchisescu.
2019.

 
 Domes to Drones: Self-Supervised Active
Triangulation for 3D Human Pose Reconstruction.

 
 In NeurIPS .

 
 
 

 
 Pishchulin et al . (2016) 
 
Leonid Pishchulin, Eldar
Insafutdinov, Siyu Tang, Bjoern Andres,
Mykhaylo Andriluka, Peter V Gehler, and
Bernt Schiele. 2016.

 
 Deepcut: Joint subset partition and labeling for
multi person pose estimation. In CVPR .

 
 
 

 
 Pons-Moll et al . (2015) 
 
Gerard Pons-Moll, Javier
Romero, Naureen Mahmood, and Michael J.
Black. 2015.

 
 Dyna: A Model of Dynamic Human Shape in Motion. In
 ACM TOG , Vol. 34.
1–14.

 
 
 

 
 Poppe (2007) 
 
Ronald Poppe.
2007.

 
 Vision-based human motion analysis: An overview.
In CVIU , Vol. 108.
4–18.

 
 
 

 
 Qammaz and
Argyros (2019) 
 
Ammar Qammaz and
Antonis A Argyros. 2019.

 
 MocapNET: Ensemble of SNN Encoders for 3D Human
Pose Estimation in RGB Images. In BMVC .

 
 
 

 
 Qi
et al . (2017a) 
 
Charles R Qi, Hao Su,
Kaichun Mo, and Leonidas J Guibas.
2017a.

 
 Pointnet: Deep learning on point sets for 3d
classification and segmentation. In CVPR .

 
 
 

 
 Qi
et al . (2017b) 
 
Charles Ruizhongtai Qi, Li
Yi, Hao Su, and Leonidas J Guibas.
2017b.

 
 Pointnet++: Deep hierarchical feature learning on
point sets in a metric space. In NeurIPS .

 
 
 

 
 Qiu
et al . (2019) 
 
Haibo Qiu, Chunyu Wang,
Jingdong Wang, Naiyan Wang, and
Wenjun Zeng. 2019.

 
 Cross View Fusion for 3D Human Pose Estimation. In
 ICCV .

 
 
 

 
 Qiu et al . (2020) 
 
Lingteng Qiu, Xuanye
Zhang, Yanran Li, Guanbin Li,
Xiaojun Wu, Zixiang Xiong,
Xiaoguang Han, and Shuguang Cui.
2020.

 
 Peeking into occluded joints: A novel framework for
crowd pose estimation. In arXiv preprint
arXiv:2003.10506 .

 
 
 

 
 Ramakrishna et al . (2014) 
 
Varun Ramakrishna, Daniel
Munoz, Martial Hebert, James Andrew
Bagnell, and Yaser Sheikh.
2014.

 
 Pose machines: Articulated pose estimation via
inference machines. In ECCV .

 
 
 

 
 Rayat
Imtiaz Hossain and Little (2018) 
 
Mir Rayat Imtiaz Hossain and
James J. Little. 2018.

 
 Exploiting temporal information for 3D human pose
estimation. In ECCV .

 
 
 

 
 Reddy et al . (2021) 
 
N Dinesh Reddy, Laurent
Guigues, Leonid Pishchulin, Jayan
Eledath, and Srinivasa G Narasimhan.
2021.

 
 TesseTrack: End-to-End Learnable Multi-Person
Articulated 3D Pose Tracking. In CVPR .

 
 
 

 
 Remelli
et al . (2020) 
 
Edoardo Remelli, Shangchen
Han, Sina Honari, Pascal Fua, and
Robert Wang. 2020.

 
 Lightweight Multi-View 3D Pose Estimation Through
Camera-Disentangled Representation. In CVPR .

 
 
 

 
 Ren
et al . (2015) 
 
Shaoqing Ren, Kaiming He,
Ross Girshick, and Jian Sun.
2015.

 
 Faster r-cnn: Towards real-time object detection
with region proposal networks. In NeurIPS .

 
 
 

 
 Rhodin
et al . (2018a) 
 
Helge Rhodin, Mathieu
Salzmann, and Pascal Fua.
2018a.

 
 Unsupervised Geometry-Aware Representation for 3D
Human Pose Estimation. In ECCV .

 
 
 

 
 Rhodin et al . (2018b) 
 
Helge Rhodin, Jörg
Spörri, Isinsu Katircioglu, Victor
Constantin, Frédéric Meyer, Erich
Müller, Mathieu Salzmann, and Pascal
Fua. 2018b.

 
 Learning Monocular 3D Human Pose Estimation From
Multi-View Images. In CVPR .

 
 
 

 
 Rogez
et al . (2017) 
 
G. Rogez, P.
Weinzaepfel, and C. Schmid.
2017.

 
 LCR-Net: Localization-Classification-Regression for
Human Pose. In CVPR .

 
 
 

 
 Rogez
et al . (2019) 
 
Grégory Rogez, Philippe
Weinzaepfel, and Cordelia Schmid.
2019.

 
 LCR-Net++: Multi-person 2D and 3D Pose Detection in
Natural Images. In IEEE TPAMI ,
Vol. 42. 1146–1161.

 
 
 

 
 Ruder (2017) 
 
Sebastian Ruder.
2017.

 
 An overview of multi-task learning in deep neural
networks. In arXiv preprint arXiv:1706.05098 .

 
 
 

 
 Saini et al . (2019) 
 
Nitin Saini, Eric Price,
Rahul Tallamraju, Raffi Enficiaud,
Roman Ludwig, Igor Martinović,
Aamir Ahmad, and Michael Black.
2019.

 
 Markerless Outdoor Human Motion Capture Using
Multiple Autonomous Micro Aerial Vehicles. In
 ICCV .

 
 
 

 
 Saito et al . (2019) 
 
Shunsuke Saito, Zeng
Huang, Ryota Natsume, Shigeo Morishima,
Angjoo Kanazawa, and Hao Li.
2019.

 
 PIFu: Pixel-Aligned Implicit Function for
High-Resolution Clothed Human Digitization. In
 ICCV .

 
 
 

 
 Saito
et al . (2020) 
 
Shunsuke Saito, Tomas
Simon, Jason Saragih, and Hanbyul
Joo. 2020.

 
 PIFuHD: Multi-Level Pixel-Aligned Implicit Function
for High-Resolution 3D Human Digitization. In
 CVPR .

 
 
 

 
 Sapp and Taskar (2013) 
 
Ben Sapp and Ben
Taskar. 2013.

 
 Modec: Multimodal decomposable models for human
pose estimation. In CVPR .

 
 
 

 
 Sarafianos et al . (2016) 
 
Nikolaos Sarafianos,
Bogdan Boteanu, Bogdan Ionescu, and
Ioannis A Kakadiaris. 2016.

 
 3d human pose estimation: A review of the
literature and analysis of covariates. In CVIU ,
Vol. 152. 1–20.

 
 
 

 
 Sharma et al . (2019) 
 
Saurabh Sharma, Pavan Teja
Varigonda, Prashast Bindal, Abhishek
Sharma, and Arjun Jain.
2019.

 
 Monocular 3D Human Pose Estimation by Generation
and Ordinal Ranking. In ICCV .

 
 
 

 
 Shi
et al . (2022) 
 
Dahu Shi, Xing Wei,
Liangqi Li, Ye Ren, and
Wenming Tan. 2022.

 
 End-to-End Multi-Person Pose Estimation With
Transformers. In CVPR .

 
 
 

 
 Sigal
et al . (2010) 
 
L. Sigal, A. Balan, and
M. J. Black. 2010.

 
 HumanEva: Synchronized video and motion capture
dataset and baseline algorithm for evaluation of articulated human motion.
In IJCV , Vol. 87.
4.

 
 
 

 
 Snower
et al . (2020) 
 
Michael Snower, Asim
Kadav, Farley Lai, and Hans Peter
Graf. 2020.

 
 15 keypoints is all you need. In
 CVPR .

 
 
 

 
 Su
et al . (2019) 
 
Kai Su, Dongdong Yu,
Zhenqi Xu, Xin Geng, and
Changhu Wang. 2019.

 
 Multi-person pose estimation with enhanced
channel-wise and spatial information. In CVPR .

 
 
 

 
 Sun
et al . (2020) 
 
Jennifer J. Sun, Jiaping
Zhao, Liang-Chieh Chen, Florian Schroff,
Hartwig Adam, and Ting Liu.
2020.

 
 View-Invariant Probabilistic Embedding for Human
Pose. In ECCV .

 
 
 

 
 Sun
et al . (2019) 
 
Ke Sun, Bin Xiao,
Dong Liu, and Jingdong Wang.
2019.

 
 Deep high-resolution representation learning for
human pose estimation. In CVPR .

 
 
 

 
 Sun
et al . (2017) 
 
Xiao Sun, Jiaxiang Shang,
Shuang Liang, and Yichen Wei.
2017.

 
 Compositional human pose regression. In
 ICCV .

 
 
 

 
 Tang and Wu (2019) 
 
Wei Tang and Ying Wu.
2019.

 
 Does Learning Specific Features for Related Parts
Help Human Pose Estimation?. In CVPR .

 
 
 

 
 Tang et al . (2018) 
 
Wei Tang, Pei Yu, and
Ying Wu. 2018.

 
 Deeply learned compositional models for human pose
estimation. In ECCV .

 
 
 

 
 Tekin et al . (2016a) 
 
Bugra Tekin, Isinsu
Katircioglu, Mathieu Salzmann, Vincent
Lepetit, and Pascal Fua.
2016a.

 
 Structured Prediction of 3D Human Pose with Deep
Neural Networks. In BMVC .

 
 
 

 
 Tekin et al . (2017) 
 
Bugra Tekin, Pablo
Márquez-Neila, Mathieu Salzmann, and
Pascal Fua. 2017.

 
 Learning to fuse 2d and 3d image cues for monocular
body pose estimation. In ICCV .

 
 
 

 
 Tekin
et al . (2016b) 
 
Bugra Tekin, Artem
Rozantsev, Vincent Lepetit, and Pascal
Fua. 2016b.

 
 Direct Prediction of 3D Body Poses From Motion
Compensated Sequences. In CVPR .

 
 
 

 
 Tian
et al . (2019) 
 
Zhi Tian, Hao Chen, and
Chunhua Shen. 2019.

 
 DirectPose: Direct End-to-End Multi-Person Pose
Estimation. In arXiv preprint arXiv:1911.07451 .

 
 
 

 
 Tome et al . (2020) 
 
Denis Tome, Thiemo
Alldieck, Patrick Peluse, Gerard
Pons-Moll, Lourdes Agapito, Hernan
Badino, and Fernando De la Torre.
2020.

 
 SelfPose: 3D Egocentric Pose Estimation from a
Headset Mounted Camera. In arXiv preprint
arXiv:2011.01519 .

 
 
 

 
 Tome
et al . (2019) 
 
Denis Tome, Patrick
Peluse, Lourdes Agapito, and Hernan
Badino. 2019.

 
 xR-EgoPose: Egocentric 3D Human Pose from an HMD
Camera. In ICCV .

 
 
 

 
 Tompson et al . (2015) 
 
Jonathan Tompson, Ross
Goroshin, Arjun Jain, Yann LeCun, and
Christoph Bregler. 2015.

 
 Efficient object localization using convolutional
networks. In CVPR .

 
 
 

 
 Tompson
et al . (2014) 
 
Jonathan J Tompson, Arjun
Jain, Yann LeCun, and Christoph
Bregler. 2014.

 
 Joint training of a convolutional network and a
graphical model for human pose estimation. In
 NeurIPS .

 
 
 

 
 Toshev and
Szegedy (2014) 
 
Alexander Toshev and
Christian Szegedy. 2014.

 
 Deeppose: Human pose estimation via deep neural
networks. In CVPR .

 
 
 

 
 Trumble et al . (2017) 
 
Matt Trumble, Andrew
Gilbert, Charles Malleson, Adrian
Hilton, and John Collomosse.
2017.

 
 Total Capture: 3D Human Pose Estimation Fusing
Video and Inertial Sensors. In BMVC .

 
 
 

 
 Tu et al . (2020) 
 
Hanyue Tu, Chunyu Wang,
and Wenjun Zeng. 2020.

 
 VoxelPose: Towards Multi-Camera 3D Human Pose
Estimation in Wild Environment. In ECCV .

 
 
 

 
 Tung
et al . (2017) 
 
Hsiao-Yu Fish Tung,
Hsiao-Wei Tung, Ersin Yumer, and
Katerina Fragkiadaki. 2017.

 
 Self-Supervised Learning of Motion Capture. In
 NeurIPS .

 
 
 

 
 Umer
et al . (2020) 
 
Rafi Umer, Andreas
Doering, Bastian Leibe, and Juergen
Gall. 2020.

 
 Self-supervised Keypoint Correspondences for
Multi-Person Pose Estimation and Tracking in Videos. In
 arXiv preprint arXiv:2004.12652 .

 
 
 

 
 Varol et al . (2017) 
 
Gul Varol, Javier Romero,
Xavier Martin, Naureen Mahmood,
Michael J Black, Ivan Laptev, and
Cordelia Schmid. 2017.

 
 Learning from synthetic humans. In
 CVPR .

 
 
 

 
 Vince Tan and
Cipolla (2017) 
 
Ignas Budvytis Vince Tan and
Roberto Cipolla. 2017.

 
 Indirect deep structured learning for 3D human body
shape and pose prediction. In BMVC .

 
 
 

 
 von Marcard et al . (2018) 
 
Timo von Marcard, Roberto
Henschel, Michael J. Black, Bodo
Rosenhahn, and Gerard Pons-Moll.
2018.

 
 Recovering Accurate 3D Human Pose in The Wild Using
IMUs and a Moving Camera. In ECCV .

 
 
 

 
 Von Marcard et al . (2017) 
 
Timo Von Marcard, Bodo
Rosenhahn, Michael J Black, and Gerard
Pons-Moll. 2017.

 
 Sparse inertial poser: Automatic 3d human pose
estimation from sparse imus. In Computer Graphics
Forum .

 
 
 

 
 Wandt and
Rosenhahn (2019) 
 
Bastian Wandt and Bodo
Rosenhahn. 2019.

 
 RepNet: Weakly Supervised Training of an
Adversarial Reprojection Network for 3D Human Pose Estimation. In
 CVPR .

 
 
 

 
 Wang et al . (2020a) 
 
Haoyang Wang, Riza Alp
Guler, Iasonas Kokkinos, George
Papandreou, and Stefanos Zafeiriou.
2020a.

 
 BLSM: A Bone-Level Skinned Model of the Human
Mesh. In ECCV .

 
 
 

 
 Wang
et al . (2022c) 
 
Haixin Wang, Lu Zhou,
Yingying Chen, Ming Tang, and
Jinqiao Wang. 2022c.

 
 Regularizing Vector Embedding in Bottom-Up Human
Pose Estimation. In ECCV .

 
 
 

 
 Wang
et al . (2019a) 
 
Jue Wang, Shaoli Huang,
Xinchao Wang, and Dacheng Tao.
2019a.

 
 Not All Parts Are Created Equal: 3D Pose Estimation
by Modeling Bi-Directional Dependencies of Body Parts. In
 ICCV .

 
 
 

 
 Wang
et al . (2020b) 
 
Jian Wang, Xiang Long,
Yuan Gao, Errui Ding, and
Shilei Wen. 2020b.

 
 Graph-PCNN: Two Stage Human Pose Estimation with
Graph Pose Refinement. In arXiv preprint
arXiv:2007.10599 .

 
 
 

 
 Wang
et al . (2019b) 
 
Jianbo Wang, Kai Qiu,
Houwen Peng, Jianlong Fu, and
Jianke Zhu. 2019b.

 
 AI Coach: Deep Human Pose Estimation and Analysis
for Personalized Athletic Training Assistance. In
 ACM MM .

 
 
 

 
 Wang
et al . (2020d) 
 
Jingbo Wang, Sijie Yan,
Yuanjun Xiong, and Dahua Lin.
2020d.

 
 Motion Guided 3D Pose Estimation from Videos. In
 ECCV .

 
 
 

 
 Wang
et al . (2020c) 
 
Kangkan Wang, Jin Xie,
Guofeng Zhang, Lei Liu, and
Jian Yang. 2020c.

 
 Sequential 3D Human Pose and Shape Estimation From
Point Clouds. In CVPR .

 
 
 

 
 Wang
et al . (2018) 
 
Min Wang, Xipeng Chen,
Wentao Liu, Chen Qian,
Liang Lin, and Lizhuang Ma.
2018.

 
 DRPose3D: Depth Ranking in 3D Human Pose
Estimation. In IJCAI .

 
 
 

 
 Wang
et al . (2021) 
 
Tao Wang, Jianfeng Zhang,
Yujun Cai, Shuicheng Yan, and
Jiashi Feng. 2021.

 
 Direct Multi-view Multi-person 3D Human Pose
Estimation. In NeurIPS .

 
 
 

 
 Wang
et al . (2022a) 
 
Yihan Wang, Muyang Li,
Han Cai, Wei-Ming Chen, and
Song Han. 2022a.

 
 Lite pose: Efficient architecture design for 2d
human pose estimation. In CVPR .

 
 
 

 
 Wang
et al . (2022b) 
 
Zitian Wang, Xuecheng
Nie, Xiaochao Qu, Yunpeng Chen, and
Si Liu. 2022b.

 
 Distribution-Aware Single-Stage Models for
Multi-Person 3D Pose Estimation. In CVPR .

 
 
 

 
 Wei
et al . (2016) 
 
Shih-En Wei, Varun
Ramakrishna, Takeo Kanade, and Yaser
Sheikh. 2016.

 
 Convolutional pose machines. In
 CVPR .

 
 
 

 
 Weng et al . (2019) 
 
C. Weng, B. Curless,
and I. Kemelmacher-Shlizerman.
2019.

 
 Photo Wake-Up: 3D Character Animation From a Single
Photo. In CVPR .

 
 
 

 
 Willett et al . (2020) 
 
Nora S Willett,
Hijung Valentina Shin, Zeyu Jin,
Wilmot Li, and Adam Finkelstein.
2020.

 
 Pose2Pose: pose selection and transfer for 2D
character animation. In IUI .

 
 
 

 
 Wu et al . (2017) 
 
Jiahong Wu, He Zheng,
Bo Zhao, Yixin Li,
Baoming Yan, Rui Liang,
Wenjia Wang, Shipei Zhou,
et al . 2017.

 
 Ai challenger: A large-scale dataset for going
deeper in image understanding. In arXiv preprint
arXiv:1711.06475 .

 
 
 

 
 Xiang
et al . (2019) 
 
Donglai Xiang, Hanbyul
Joo, and Yaser Sheikh. 2019.

 
 Monocular total capture: Posing face, body, and
hands in the wild. In CVPR .

 
 
 

 
 Xiao et al . (2018) 
 
Bin Xiao, Haiping Wu,
and Yichen Wei. 2018.

 
 Simple Baselines for Human Pose Estimation and
Tracking. In ECCV .

 
 
 

 
 Xie
et al . (2020) 
 
Rongchang Xie, Chunyu
Wang, and Yizhou Wang. 2020.

 
 MetaFuse: A Pre-trained Fusion Model for Human Pose
Estimation. In CVPR .

 
 
 

 
 Xiong et al . (2019) 
 
Fu Xiong, Boshen Zhang,
Yang Xiao, Zhiguo Cao,
Taidong Yu, Joey Zhou Tianyi, and
Junsong Yuan. 2019.

 
 A2J: Anchor-to-Joint Regression Network for 3D
Articulated Pose Estimation from a Single Depth Image. In
 ICCV .

 
 
 

 
 Xu et al . (2020a) 
 
Hongyi Xu, Eduard Gabriel
Bazavan, Andrei Zanfir, William T.
Freeman, Rahul Sukthankar, and Cristian
Sminchisescu. 2020a.

 
 GHUM \ GHUML: Generative 3D Human Shape and
Articulated Pose Models. In CVPR .

 
 
 

 
 Xu
et al . (2020c) 
 
Jingwei Xu, Zhenbo Yu,
Bingbing Ni, Jiancheng Yang,
Xiaokang Yang, and Wenjun Zhang.
2020c.

 
 Deep Kinematics Analysis for Monocular 3D Human
Pose Estimation. In CVPR .

 
 
 

 
 Xu et al . (2021) 
 
Lumin Xu, Yingda Guan,
Sheng Jin, Wentao Liu,
Chen Qian, Ping Luo,
Wanli Ouyang, and Xiaogang Wang.
2021.

 
 Vipnas: Efficient video pose estimation via neural
architecture search. In CVPR .

 
 
 

 
 Xu et al . (2019) 
 
Weipeng Xu, Avishek
Chatterjee, Michael Zollhoefer, Helge
Rhodin, Pascal Fua, Hans-Peter Seidel,
and Christian Theobalt. 2019.

 
 Mo 2 cap 2: Real-time mobile 3d motion capture with
a cap-mounted fisheye camera. In IEEE TVCG Proc.
VR , Vol. 25. 2093–2101.

 
 
 

 
 Xu et al . (2020b) 
 
Xiangyu Xu, Hao Chen,
Francesc Moreno-Noguer, Laszlo A. Jeni,
and Fernando De la Torre.
2020b.

 
 3D Human Shape and Pose from a Single
Low-Resolution Image with Self-Supervised Learning. In
 ECCV .

 
 
 

 
 Yan
et al . (2018) 
 
Sijie Yan, Yuanjun Xiong,
and Dahua Lin. 2018.

 
 Spatial temporal graph convolutional networks for
skeleton-based action recognition. In AAAI .

 
 
 

 
 Yang
et al . (2021) 
 
Sen Yang, Zhibin Quan,
Mu Nie, and Wankou Yang.
2021.

 
 Transpose: Keypoint localization via transformer.
In ICCV .

 
 
 

 
 Yang
et al . (2017) 
 
Wei Yang, Shuang Li,
Wanli Ouyang, Hongsheng Li, and
Xiaogang Wang. 2017.

 
 Learning feature pyramids for human pose
estimation. In ICCV .

 
 
 

 
 Yang
et al . (2016) 
 
Wei Yang, Wanli Ouyang,
Hongsheng Li, and Xiaogang Wang.
2016.

 
 End-to-end learning of deformable mixture of parts
and deep convolutional neural networks for human pose estimation. In
 CVPR .

 
 
 

 
 Yang
et al . (2018) 
 
Wei Yang, Wanli Ouyang,
Xiaolong Wang, Jimmy Ren,
Hongsheng Li, and Xiaogang Wang.
2018.

 
 3D Human Pose Estimation in the Wild by Adversarial
Learning. In CVPR .

 
 
 

 
 Yang and Ramanan (2012) 
 
Yi Yang and Deva
Ramanan. 2012.

 
 Articulated human detection with flexible mixtures
of parts. In IEEE TPAMI ,
Vol. 35. 2878–2890.

 
 
 

 
 Ye
et al . (2022) 
 
Hang Ye, Wentao Zhu,
Chunyu Wang, Rujie Wu, and
Yizhou Wang. 2022.

 
 Faster VoxelPose: Real-time 3D Human Pose
Estimation by Orthographic Projection. In ECCV .

 
 
 

 
 Yu
et al . (2021) 
 
Changqian Yu, Bin Xiao,
Changxin Gao, Lu Yuan,
Lei Zhang, Nong Sang, and
Jingdong Wang. 2021.

 
 Lite-hrnet: A lightweight high-resolution network.
In CVPR .

 
 
 

 
 Yu et al . (2019) 
 
T. Yu, J. Zhao,
Z. Zheng, K. Guo, Q.
Dai, H. Li, G. Pons-Moll, and
Y. Liu. 2019.

 
 DoubleFusion: Real-time Capture of Human
Performances with Inner Body Shapes from a Single Depth Sensor. In
 IEEE TPAMI . 7287–7296.

 
 
 

 
 Yu et al . (2019) 
 
Tao Yu, Zerong Zheng,
Yuan Zhong, Jianhui Zhao,
Qionghai Dai, Gerard Pons-Moll, and
Yebin Liu. 2019.

 
 Simulcap: Single-view human performance capture
with cloth simulation. In CVPR .

 
 
 

 
 Yuan et al . (2021) 
 
Yuhui Yuan, Rao Fu,
Lang Huang, Weihong Lin,
Chao Zhang, Xilin Chen, and
Jingdong Wang. 2021.

 
 Hrformer: High-resolution vision transformer for
dense predict. In NeurIPS .

 
 
 

 
 Zanfir et al . (2020) 
 
Andrei Zanfir,
Eduard Gabriel Bazavan, Hongyi Xu,
Bill Freeman, Rahul Sukthankar, and
Cristian Sminchisescu. 2020.

 
 Weakly Supervised 3D Human Pose and Shape
Reconstruction with Normalizing Flows. In ECCV .

 
 
 

 
 Zanfir et al . (2018) 
 
A. Zanfir, E.
Marinoiu, and C. Sminchisescu.
2018.

 
 Monocular 3D Pose and Shape Estimation of Multiple
People in Natural Scenes: The Importance of Multiple Scene Constraints. In
 CVPR .

 
 
 

 
 Zanfir et al . (2018) 
 
Andrei Zanfir, Elisabeta
Marinoiu, Mihai Zanfir, Alin-Ionut Popa,
and Cristian Sminchisescu.
2018.

 
 Deep Network for the Integrated 3D Sensing of
Multiple People in Natural Images.

 
 In NeurIPS .

 
 
 

 
 Zeng
et al . (2020) 
 
Ailing Zeng, Xiao Sun,
Fuyang Huang, Minhao Liu,
Qiang Xu, and Stephen Lin.
2020.

 
 SRNet: Improving Generalization in 3D Human Pose
Estimation with a Split-and-Recombine Approach. In
 ECCV .

 
 
 

 
 Zeng et al . (2020) 
 
W. Zeng, W. Ouyang,
P. Luo, W. Liu, and
X. Wang. 2020.

 
 3D Human Mesh Regression With Dense
Correspondence. In CVPR .

 
 
 

 
 Zhang
et al . (2020h) 
 
Feng Zhang, Xiatian Zhu,
Hanbin Dai, Mao Ye, and
Ce Zhu. 2020h.

 
 Distribution-aware coordinate representation for
human pose estimation. In CVPR .

 
 
 

 
 Zhang
et al . (2019b) 
 
Feng Zhang, Xiatian Zhu,
and Mao Ye. 2019b.

 
 Fast human pose estimation. In
 CVPR .

 
 
 

 
 Zhang et al . (2019a) 
 
Hong Zhang, Hao Ouyang,
Shu Liu, Xiaojuan Qi,
Xiaoyong Shen, Ruigang Yang, and
Jiaya Jia. 2019a.

 
 Human pose estimation with spatial contextual
information. In arXiv preprint arXiv:1901.01760 .

 
 
 

 
 Zhang et al . (2020d) 
 
Haotian Zhang, Cristobal
Sciutto, Maneesh Agrawala, and Kayvon
Fatahalian. 2020d.

 
 Vid2Player: Controllable Video Sprites that Behave
and Appear like Professional Tennis Players. In
 arXiv preprint arXiv:2008.04524 .

 
 
 

 
 Zhang et al . (2021) 
 
Hongwen Zhang, Yating
Tian, Xinchi Zhou, Wanli Ouyang,
Yebin Liu, Limin Wang, and
Zhenan Sun. 2021.

 
 PyMAF: 3D Human Pose and Shape Regression with
Pyramidal Mesh Alignment Feedback Loop. In ICCV .

 
 
 

 
 Zhang
et al . (2022) 
 
Jinlu Zhang, Zhigang Tu,
Jianyu Yang, Yujin Chen, and
Junsong Yuan. 2022.

 
 MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for
3D Human Pose Estimation in Video. In CVPR .

 
 
 

 
 Zhang
et al . (2020c) 
 
Tianshu Zhang, Buzhen
Huang, and Yangang Wang.
2020c.

 
 Object-Occluded Human Shape and Pose Estimation
From a Single Color Image. In CVPR .

 
 
 

 
 Zhang
et al . (2020b) 
 
Wenqiang Zhang, Jiemin
Fang, Xinggang Wang, and Wenyu Liu.
2020b.

 
 EfficientPose: Efficient Human Pose Estimation with
Neural Architecture Search. In arXiv preprint
arXiv:2012.07086 .

 
 
 

 
 Zhang
et al . (2013) 
 
Weiyu Zhang, Menglong
Zhu, and Konstantinos G Derpanis.
2013.

 
 From actemes to action: A strongly-supervised
representation for detailed action understanding. In
 ICCV .

 
 
 

 
 Zhang
et al . (2020a) 
 
Yuxiang Zhang, Liang An,
Tao Yu, xiu Li, Kun Li,
and Yebin Liu. 2020a.

 
 4D Association Graph for Realtime Multi-person
Motion Capture Using Multiple Video Cameras. In
 CVPR .

 
 
 

 
 Zhang
et al . (2020e) 
 
Yuexi Zhang, Yin Wang,
Octavia Camps, and Mario Sznaier.
2020e.

 
 Key Frame Proposal Network for Efficient Pose
Estimation in Videos. In arXiv preprint
arXiv:2007.15217 .

 
 
 

 
 Zhang
et al . (2020f) 
 
Zhe Zhang, Chunyu Wang,
Wenhu Qin, and Wenjun Zeng.
2020f.

 
 Fusing Wearable IMUs With Multi-View Images for
Human Pose Estimation: A Geometric Approach. In
 CVPR .

 
 
 

 
 Zhang
et al . (2020g) 
 
Zhe Zhang, Chunyu Wang,
Weichao Qiu, Wenhu Qin, and
Wenjun Zeng. 2020g.

 
 AdaFuse: Adaptive Multiview Fusion for Accurate
Human Pose Estimation in the Wild. In IJCV ,
Vol. 129. 703–718.

 
 
 

 
 Zhao
et al . (2019b) 
 
Long Zhao, Xi Peng,
Yu Tian, Mubbasir Kapadia, and
Dimitris N. Metaxas. 2019b.

 
 Semantic Graph Convolutional Networks for 3D Human
Pose Regression. In CVPR .

 
 
 

 
 Zhao et al . (2019a) 
 
Mingmin Zhao, Yingcheng
Liu, Aniruddh Raghu, Tianhong Li,
Hang Zhao, Antonio Torralba, and
Dina Katabi. 2019a.

 
 Through-Wall Human Mesh Recovery Using Radio
Signals. In ICCV .

 
 
 

 
 Zhao et al . (2018) 
 
Mingmin Zhao, Yonglong
Tian, Hang Zhao, Mohammad Abu Alsheikh,
Tianhong Li, Rumen Hristov,
Zachary Kabelac, Dina Katabi, and
Antonio Torralba. 2018.

 
 RF-based 3D skeletons. In
 SIGCOMM .

 
 
 

 
 Zhao
et al . (2023) 
 
Qitao Zhao, Ce Zheng,
Mengyuan Liu, Pichao Wang, and
Chen Chen. 2023.

 
 PoseFormerV2: Exploring Frequency Domain for
Efficient and Robust 3D Human Pose Estimation. In
 CVPR .

 
 
 

 
 Zhen et al . (2020) 
 
Jianan Zhen, Qi Fang,
Jiaming Sun, Wentao Liu,
Wei Jiang, Hujun Bao, and
Xiaowei Zhou. 2020.

 
 SMAP: Single-Shot Multi-Person Absolute 3D Pose
Estimation. In ECCV .

 
 
 

 
 Zheng
et al . (2023a) 
 
Ce Zheng, Xianpeng Liu,
Guo-Jun Qi, and Chen Chen.
2023a.

 
 POTTER: Pooling Attention Transformer for Efficient
Human Mesh Recovery. In CVPR .

 
 
 

 
 Zheng
et al . (2022) 
 
Ce Zheng, Matias
Mendieta, Pu Wang, Aidong Lu, and
Chen Chen. 2022.

 
 A Lightweight Graph Transformer Network for Human
Mesh Reconstruction from 2D Human Pose. In ACM
Multimedia .

 
 
 

 
 Zheng
et al . (2023b) 
 
Ce Zheng, Matias
Mendieta, Taojiannan Yang, Guo-Jun Qi,
and Chen Chen. 2023b.

 
 FeatER: An Efficient Network for Human
Reconstruction via Feature Map-Based TransformER. In
 CVPR .

 
 
 

 
 Zheng et al . (2021) 
 
Ce Zheng, Sijie Zhu,
Matias Mendieta, Taojiannan Yang,
Chen Chen, and Zhengming Ding.
2021.

 
 3D Human Pose Estimation with Spatial and Temporal
Transformers. In ICCV .

 
 
 

 
 Zhi et al . (2020) 
 
Tiancheng Zhi, Christoph
Lassner, Tony Tung, Carsten Stoll,
Srinivasa G. Narasimhan, and Minh Vo.
2020.

 
 TexMesh: Reconstructing Detailed Human Texture and
Geometry from RGB-D Video. In ECCV .

 
 
 

 
 Zhou
et al . (2020) 
 
Keyang Zhou, Bharat Lal
Bhatnagar, and Gerard Pons-Moll.
2020.

 
 Unsupervised Shape and Pose Disentanglement for 3D
Meshes. In arXiv preprint arXiv:2007.11341 .

 
 
 

 
 Zhou et al . (2019) 
 
K. Zhou, X. Han,
N. Jiang, K. Jia, and
J. Lu. 2019.

 
 HEMlets Pose: Learning Part-Centric Heatmap
Triplets for Accurate 3D Human Pose Estimation. In
 ICCV .

 
 
 

 
 Zhou
et al . (2017) 
 
Xingyi Zhou, Qixing
Huang, Xiao Sun, Xiangyang Xue, and
Yichen Wei. 2017.

 
 Towards 3D Human Pose Estimation in the Wild: A
Weakly-Supervised Approach. In ICCV .

 
 
 

 
 Zhou
et al . (2016a) 
 
Xingyi Zhou, Xiao Sun,
Wei Zhang, Shuang Liang, and
Yichen Wei. 2016a.

 
 Deep kinematic pose regression. In
 ECCV .

 
 
 

 
 Zhou
et al . (2019) 
 
Xingyi Zhou, Dequan Wang,
and Philipp Krähenbühl.
2019.

 
 Objects as points. In
 arXiv preprint arXiv:1904.07850 .

 
 
 

 
 Zhou
et al . (2016b) 
 
Xiaowei Zhou, Menglong
Zhu, Kosta Derpanis, and Kostas
Daniilidis. 2016b.

 
 Sparseness Meets Deepness: 3D Human Pose Estimation
from Monocular Video. In CVPR .

 
 
 

 
 Zhou et al . (2018) 
 
Xiaowei Zhou, Menglong
Zhu, Georgios Pavlakos, Spyridon
Leonardos, Konstantinos G Derpanis, and
Kostas Daniilidis. 2018.

 
 Monocap: Monocular human motion capture using a cnn
coupled with a geometric prior. In IEEE TPAMI ,
Vol. 41. 901–914.

 
 
 

 
 Zhu
et al . (2019) 
 
Hao Zhu, Xinxin Zuo,
Sen Wang, Xun Cao, and
Ruigang Yang. 2019.

 
 Detailed human shape estimation from a single image
by hierarchical mesh deformation. In CVPR .

 
 
 

 
 Zhu et al . (2020) 
 
Luyang Zhu, Konstantinos
Rematas, Brian Curless, Steve Seitz,
and Ira Kemelmacher-Shlizerman.
2020.

 
 Reconstructing NBA players. In
 ECCV .

 
 
 

 
 Zhu
et al . (2017) 
 
Xiangyu Zhu, Yingying
Jiang, and Zhenbo Luo. 2017.

 
 Multi-person pose estimation for posetrack with
enhanced part affinity fields. In ICCV PoseTrack
Workshop .

 
 
 

 
 Zou and Tang (2021) 
 
Zhiming Zou and Wei
Tang. 2021.

 
 Modulated Graph Convolutional Network for 3D Human
Pose Estimation. In ICCV .

 
 
 

 
 Zuffi and Black (2015) 
 
Silvia Zuffi and
Michael J. Black. 2015.

 
 The Stitched Puppet: A Graphical Model of 3D
Human Shape and Pose. In CVPR .