Transformers in Action Recognition: A Review on Temporal Modeling 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2302.01921v1 [cs.CV] 29 Dec 2022 
 
 

# Transformers in Action Recognition: A Review on Temporal Modeling

 Journal:  arXiv 
 
 
 Elham Shabaninia
 
 Note:  Corresponding author. Email: e.shabaninia@kgut.ac.ir
 
 Affiliation:  Department of Applied Mathematics, Faculty of Sciences and Modern Technologies, Graduate University of Advanced Technology, Kerman, 7631818356, Iran,
 
 Affiliation:  Department of Electrical Engineering, Shahid Bahonar University of Kerman, Kerman, 76169133, Iran,
 
    
 Hossein Nezamabadi-pour
 
 Note:  Email: nezam@uk.ac.ir
 
 Affiliation:  Department of Electrical Engineering, Shahid Bahonar University of Kerman, Kerman, 76169133, Iran,
 
    
 Fatemeh Shafizadegan
 
 Note:  Email: fatemeh.shafizadegan.1990@eng.ui.ac.ir
 
 Affiliation:  Department of Computer Engineering, University of Isfahan, Isfahan, 8174673441, Iran,
 

 Abstract 
 
 In vision-based action recognition, spatio-temporal features from different modalities are used for recognizing activities. Temporal modeling is a long challenge of action recognition. However, there are limited methods such as pre-computed motion features, three-dimensional (3D) filters, and recurrent neural networks (RNN) for modeling motion information in deep-based approaches. Recently, transformers’ success in modeling long-range dependencies in natural language processing (NLP) tasks has gotten great attention from other domains; including speech, image, and video, to rely entirely on self-attention without using sequence-aligned RNNs or convolutions. Although the application of transformers to action recognition is relatively new, the amount of research proposed on this topic within the last few years is astounding. This paper especially reviews recent progress in deep learning methods for modeling temporal variations. It focuses on action recognition methods that use transformers for temporal modeling, discussing their main features, used modalities, and identifying opportunities and challenges for future research.

 
 
 
 Keywords:  
transformer , action recognition , deep learning , temporal modeling

 
 

## 1 Introduction

 
 Video-based action recognition is the task of recognizing human activities (including gestures, simple actions, human-object/human-human interactions, group activities, behaviors, and events) from video sequences or still images [ 1 , 2 ] . Compared with video-based methods, human activity recognition (HAR) from static images is still an open and challenging task [ 3 , 4 , 5 ] and includes a limited range of proposed methods. Due to many applications, vision-based human action recognition is known as an old field of computer vision and different data modalities are adopted for recognition in the literature, including RGB, depth, skeleton, infrared, point cloud, etc. while the three first modalities are used primarily for human action recognition. RGB data provides the details of a scene (including shape, color, and texture) and helps describe the semantics of actions, depth maps provide three-dimensional (3D) structural information about the scene. On the other hand, skeletal data is high-level information about the 3D location of joints. Multi-modal approaches use the knowledge of different modalities for the visual understanding of complex actions.

 
 
 Today, human action recognition methods are mainly established with the help of deep neural networks (DNNs) [ 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 ] . That is mainly due to the success of convolutional neural networks (CNNs) in encoding spatial information of images for object detection and recognition. Various studies discovered the abilities of CNNs in automatically extracting useful and discriminative features from images that generalize very well [ 17 , 18 , 19 , 20 ] . In addition, deep networks can scale up to tens of millions of parameters and huge labeled datasets [ 8 ] . Consequently, the computer vision community mainly focused on using the capacity of deep architectures in almost all fields of research, including human action recognition. However, besides encoding spatial information of frames, video analysis is involved with modeling temporal information.

 
 
 Encoding temporal information is of vital importance in recognizing different subtle sub-activities. Each activity is divided into different sub-activities. The sequence of these sub-activities differentiates among different activities. However, the temporal dimension typically causes action recognition to be challenging. On the other hand, existing deep architectures generally encode temporal information with limited solutions [ 21 , 22 , 23 , 24 ] such as 3D filters, pre-computed motion features, and recurrent neural networks (RNNs). These models are typically restricted in simultaneously acquiring local and global variations of temporal features.

 
 
 On the other hand, the transformer is a new encoder-decoder architecture that uses the attention mechanism to differentially weigh each part of the input data [ 25 ] . Although transformers are designed to handle sequential input data, they do not necessarily process it in order. Rather, the attention mechanism provides context for any position in the input sequence. This feature allows for more parallelization than RNNs and therefore reduces training times. Transformers achieved great success in natural language processing (NLP) tasks [ 25 ] and are now applied to images [ 26 , 27 , 28 ] .

 
 
 Along with NLP, video recognition is a perfect candidate for transformers, where videos are represented as a sequence of images (similar to language processing, in which the input words or characters are represented as a sequence of tokens [ 29 ] ). The application of transformers for action recognition is relatively new. However, the amount of research proposed on this topic within the last few years is increasing. This paper aims at capturing a snapshot of trends for temporal modeling in human action recognition. It focuses on supervised learning methods that often require a large amount of data with expensive labels for training models. Meanwhile, unsupervised and semi-supervised learning techniques [ 30 , 31 ] (which enable to leverage of the availability of unlabeled data to train the models) are beyond the scope of this paper. The methods are categorized into five groups: motion-based feature approaches, three-dimensional convolutional neural networks, recurrent neural networks, transformers, and hybrid methods (see Figure 1 ). This categorization is advised by some research approaches. Especially, three first categories of pre-computed motion features, 3D filters, and RNN are mentioned in [ 21 , 22 , 23 , 24 ] . However, transformers for video action recognition are a new approach. Finally, the hybrid category is designed to encompass combined methods.
So main contributions of this paper are as follows:

 
 1. 
 
 This paper reviews the main approaches proposed for modeling temporal information for human action recognition in deep-based methods.

 

 2. 
 
 The deep learning-based approaches are categorized into motion-based, RNNs, 3D filters, transformers, and hybrid methods.

 

 3. 
 
 In each category, methods are grouped based on used visual data modalities (RGB, depth, skeleton) to better compare similar approaches.

 

 4. 
 
 We provide a comprehensive survey of transformer-based human action recognition methods.

 

 5. 
 
 Some suggestions are proposed for future research on human action recognition using transformers.

 

 
 
 
 Figure 1: The taxonomy of this paper: different methods are categorized into five approaches. 
 
 
 The remainder of this paper is organized as follows. Section 2 reviews related survey papers on human action recognition. Section 3 provides a brief review of traditional approaches for temporal modeling and introduces five deep learning-based approaches for modeling the time dimension. These approaches are discussed in detail in different subsections. In each subsection, distinct modalities or combinations of multiple modalities used in methods are explored. Discussions and prospects are provided in section 4 . The paper concludes in section 5 .

 
 
 

## 2 Related Survey Papers

 
 Human action recognition is one of the old and interesting topics of computer vision. There are a lot of survey papers on this topic targeting different aspects of action recognition. Table 1 lists some recent survey papers on human action recognition. As this table shows, some existing papers provide a review of both traditional and deep-based approaches, while others only concentrate on deep-based methodologies. On the other hand, there are some surveys on applications of human action recognition [ 32 , 33 ] or benchmark datasets of action recognition [ 15 , 34 , 35 , 36 ] . In addition, a group of reviews focuses on specific data modalities such as visual or sensor-based methods [ 7 , 12 , 37 , 38 , 39 , 40 , 41 , 42 ] . While some others review approaches based on multiple data modalities [ 9 , 43 , 44 , 45 , 46 , 47 , 48 ] . Compared with the existing survey papers:

 
 
 
 1. 
 
 This paper provides a novel taxonomy to review the main approaches of temporal modeling in human action recognition, with the main emphasis on transformer-based methods.

 

 2. 
 
 The survey includes both single-modal and multi-modal approaches of human action recognition.

 

 3. 
 
 Vision-based methods with RGB, depth, skeleton and hybrid features are considered.

 

 4. 
 
 A short review of conventional methods besides a long review of deep-based approaches is included.

 

 
 
 
 Note that some sections of this paper that consist of input modality + modeling approach are mentioned in some other survey papers. For example in [ 49 ] , some approaches of skeleton + 3D filters, in [ 27 ] , some approaches of RGB/Skeleton + transformer, and in [ 15 ] , some approaches of RGB/depth + Motion features are reviewed. All these papers are comprehensive and informative. However, the interests of these papers are other topics (see Table 1 ) and there is no paper with the main emphasis on transformers and temporal modeling that reviews all these categories and modalities altogether.

 
 
 
 Table 1: Recent survey papers on human action recognition(C: Reviewing Conventional approaches, D: Reviewing deep-based approaches). 
 
 
 
 References | 
 Year | 
 
 
 Main Focus 
 | 
 C | 
 D | 

 
 
 
 Yuanyuan et al. [ 21 ] | 
 2021 | 
 
 
 Categorizing deep approaches into Two-stream, 3D CNN, and LSTM groups. 
 | 
 ✓ | 
 ✓ | 

 
 Pareek et al. [ 50 ] | 
 2021 | 
 
 
 Reviewing features in action presentation, applications, and challenges in HAR. 
 | 
 ✓ | 
 ✓ | 

 
 Khan et al. [ 22 ] | 
 2021 | 
 
 
 Categorizing deep approaches into CNN, RNN, and hybrid. 
 | 
 ✓ | 
 ✓ | 

 
 Rangasamy et al. [ 23 ] | 
 2020 | 
 
 
 Categorizing deep-based architectures into CNN, 3D CNN, RNN, and LSTM (with a focus on sports video analysis). 
 | 
 ✓ | 
 ✓ | 

 
 Jegham et al. [ 51 ] | 
 2020 | 
 
 
 approaches are grouped into template-based, generative, and discriminative models. 
 | 
 ✓ | 
 ✓ | 

 
 Zhang et al. [ 14 ] | 
 2019 | 
 
 
 Overviewing methods in action feature representation based on deep learning. 
 | 
 ✓ | 
 ✓ | 

 
 Dhiman et al. [ 52 ] | 
 2019 | 
 
 
 Investigating methods in abnormal human action recognition. 
 | 
 ✓ | 
 ✓ | 

 
 Estevam et al. [ 53 ] | 
 2021 | 
 
 
 Studying zero-shot video-based action recognition methods. 
 | 
 ✓ | 
 ✓ | 

 
 Zhu et al. [ 24 ] | 
 2020 | 
 
 
 Categorizing deep-based approaches into CNNs, two-stream, and 3D CNNs. 
 | 
 | 
 ✓ | 

 
 Yao et al. [ 54 ] | 
 2019 | 
 
 
 Studying CNN-based approaches. 
 | 
 | 
 ✓ | 

 
 Sreenu et al. [ 55 ] | 
 2019 | 
 
 
 Reviewing application of HAR in video surveillance for crowd analysis. 
 | 
 | 
 ✓ | 

 
 Chen et al. [ 37 ] | 
 2021 | 
 
 
 Surveying deep approaches in sensor-based methods. 
 | 
 | 
 ✓ | 

 
 Nguyen et al. [ 38 ] | 
 2021 | 
 
 
 Reviewing deep-based approaches and power requirements in mobile and wearable sensors. 
 | 
 | 
 ✓ | 

 
 Hussain et al. [ 39 ] | 
 2020 | 
 
 
 Categorizing methods into wearable, object-tagged, and device-free approaches. Further, device-free studies are grouped into action, motion, and interaction. 
 | 
 ✓ | 
 ✓ | 

 
 Dang et al. [ 7 ] | 
 2020 | 
 
 
 Analyzing vision-based and sensor-based methods and their corresponding procedure of data collection, preprocessing, feature engineering, and training. 
 | 
 ✓ | 
 ✓ | 

 
 Beddiar et al. [ 40 ] | 
 2020 | 
 
 
 Reviewing vision-based approaches according to feature extraction process, recognition stage, source of input data, and machine learning supervision level. 
 | 
 ✓ | 
 ✓ | 

 
 AI-Faris et al. [ 41 ] | 
 2020 | 
 
 
 Categorizing vision-based approaches using deep learning into generative and discriminative models. 
 | 
 ✓ | 
 ✓ | 

 
 Wang et al. [ 42 ] | 
 2019 | 
 
 
 Analyzing ten approaches using conventional and deep strategies on visual modality (Kinect-based) 
 | 
 ✓ | 
 ✓ | 

 
 Wang et al. [ 12 ] | 
 2018 | 
 
 
 Classifying deep-based segmented and continuous motion recognition approaches into RGB, depth, skeleton, and hybrid. 
 | 
 | 
 ✓ | 

 
 Yadav et al. [ 44 ] | 
 2021 | 
 
 
 Deep and conventional approaches are grouped into vision-based, wearables, and multimodal categories. 
 | 
 ✓ | 
 ✓ | 

 
 Majumder et al. [ 43 ] | 
 2020 | 
 
 
 Deep and conventional approaches are categorized into RGB, depth, and RGB depth. 
 | 
 ✓ | 
 ✓ | 

 
 Rahmani et al. [ 45 ] | 
 2021 | 
 
 
 Deep approaches are categorized into visual modality and non-visual modality. 
 | 
 | 
 ✓ | 

 
 Majumder et al. [ 46 ] | 
 2020 | 
 
 
 Traditional and deep approaches are grouped into fusions of RGB inertial, depth inertial, and RGB depth inertial. 
 | 
 ✓ | 
 ✓ | 

 
 Li et al. [ 47 ] | 
 2020 | 
 
 
 Methods in multi-user or group activity recognition are categorized into vision-based, sensor-based, radiofrequency, and hybrid groups. 
 | 
 ✓ | 
 ✓ | 

 
 Liu et al. [ 9 ] | 
 2019 | 
 
 
 Deep and conventional methods with depth, skeleton, and hybrid features are considered. 
 | 
 ✓ | 
 ✓ | 

 
 Ulhaq et al. [ 56 ] | 
 2022 | 
 
 
 The key contributions and trends to adapting Transformers
for visual recognition of human action are summarized. 
 | 
 ✓ | 
 ✓ | 

 

 
 
 
 
 
 

## 3 Temporal Modeling in Action Recognition Methods

 
 Video-based action recognition is a video content analysis (VCA) task responsible for automatically analyzing the captured video to detect or recognize specific actions performed. The critical issue in VCA is representing suitable spatio-temporal features and modeling dynamical patterns [ 57 ] . VCA approaches are categorized into frame-by-frame-based methods and volumetric approaches. The former typically extracts a set of features from each frame. The features are then usually considered as time-series data. On the other hand, volumetric approaches implicitly model temporal dynamics and consider the video as a 3D volume. They extend standard features used for images to the 3D case.

 
 
 In recent decades, the vision community has suggested numerous action recognition techniques using RGB or depth. Among them, there are some promising methods, including representations for local spatio-temporal features [ 58 ] such as SIFT3D [ 59 ] , ESURF [ 60 ] , HOG3D [ 61 ] , HOF [ 62 ] . These traditional action recognition methods use several detected salient points and local feature descriptors for each point. The local descriptors are then collected into a holistic descriptor for the entire video to be used for classification. The pro of using these local features is that they do not require detecting the human body, and the local features are almost robust to illumination changes, cluttered background, and noise. The con is the lack of semantics and limitation in discriminative capacity [ 63 ] . Approaches such as Motionlets [ 64 ] , Action Bank [ 65 ] , Motion Atoms [ 66 ] , Dynamic-Poselets [ 63 ] , and Actons [ 67 ] are proposed to account for these limitations.

 
 
 For modeling the temporal variation of skeletons, different approaches are proposed in the literature for traditional methods. In some approaches, the features computed from the action sequences are clustered into posture visual words (representing the prototypical poses of actions), and then the temporal evolutions of those visual words are modeled by explicit methods such as hidden Markov models (HMM) [ 68 , 69 ] or conditional random fields (CRF) [ 70 , 71 ] . Some other approaches consider the manifold of the trajectories [ 72 ] or use hierarchical extended histogram (HEH) for modeling temporal variation of features acquired from individual frames of input sequence [ 73 ] . These hand-crafted features and descriptors are recently substituted with deep representations to automatically extract high-level information from training data without using hand-crafted rules.

 
 
 As mentioned above, in recent years there has been rapid development in deep learning-based methods for human action recognition. Numerous studies are proposed in the literature for solving different challenges of human action recognition using deep architectures. Reviewing all these methods is a relatively comprehensive task. Many surveys discuss the pros and cons of different methods in detail [ 6 , 7 , 8 , 12 ] . Here the focus is on how different methods deal with the temporal dimension. The methods are categorized into five groups: motion-based feature approaches, three-dimensional convolutional neural networks, recurrent neural networks, transformers, and hybrid methods. These methods are discussed in detail in the following. In each group, methods are categorized based on used modalities (RGB, depth, skeleton, or combination of multiple modalities).

 
 

### 3.1 Motion-based Feature Approaches

 
 The first category is based on pre-computed motion features like 2D dense optical flow maps as input to the neural networks. These networks generally use multiple streams to encode both appearance and motion of human actions using different modalities. Finally, different types of information (learned from the input) are fused to get the final result. There are different fusion approaches in the literature. In [ 74 ] , fusion methods are categorized into early, late, and intermediate. Early fusion involves the integration of multiple raw or preprocessed data modalities into a vector ahead of feature extraction. In intermediate fusion, the features, respective to each stream, are concatenated before classification. Late fusion refers to collecting decisions from multiple classifiers and applying maximum or average scores to get the final decision. Similar taxonomies also exist in the literature; for example, in [ 75 ] fusion methods are grouped into feature-level, score-level, and decision-level.

 
 

#### 3.1.1 RGB

 
 In [ 76 ] , flow coding images computed from consecutive video frames are fed to a deep CNN network to extract deep temporal features from flow coding images. Then, the output features of several frames are concatenated together to learn the temporal convolution. Finally, a fully connected feedforward neural network is used for classification. In [ 19 ] , two separate streams (a spatial and a temporal convolution network) are simultaneously applied to learn both the appearance and motion of actions. In [ 77 ] , a trajectory-pooled deep-convolutional descriptor is proposed to learn discriminative convolutional feature maps, and conduct trajectory-constrained pooling to aggregate these convolutional features into effective descriptors. In [ 78 ] , temporal linear encoding (TLE), embedded inside of ConvNet architectures is presented to aggregate information from an entire video. The pooling layer called ActionVLAD [ 79 ] is also used to combine appearance and motion streams. It aggregates convolutional feature descriptors in different image portions and temporal spans. This layer is used to combine appearance and motion streams. As mentioned before, the problem with two-stream networks is the lack of transferring knowledge between the two streams [ 19 , 77 ] . Some approaches address this problem [ 80 ] to communicate between the two streams. However, this interaction between different streams is known difficult [ 8 ] .

 
 
 

#### 3.1.2 Depth

 
 In [ 81 ] , a CNN-based framework is proposed using dynamic images. Dynamic images (DIs) summarize the motion and temporal information of video sequences in a single image. Multi-view dynamic images created from multi-view depth video are employed to better model 3D specifications. Different views share the same convolutional layers but there are distinct fully connected layers (Figure 2 ). The main goal is to reduce the gradient vanishing problem, particularly on the shallow convolutional layers. Further, spatial-temporal action proposal is used to decrease the sensitivity of CNNs to scene variations.

 
 
 Figure 2: Multi-view dynamic image adaptive CNN learning model [ 81 ] . 
 
 
 In [ 82 ] , three different depth representations are proposed: dynamic depth image (DDI), dynamic depth normal image (DDNI), and dynamic depth motion normal image (DDMNI) for segmented and continuous action recognition. Dynamic images are constructed using hierarchical bidirectional rank pooling to extract spatial-temporal information. DDIs extract dynamics of postures, while DDNIs and DDMNIs exploit 3D structural information of depth maps. These three representations of depth are fed to a pre-trained CNN model for fine-tuning without any need for training the network from scratch. In [ 83 ] , weighted hierarchical depth motion maps (WHDMM) and three-channel deep convolutional neural networks (ConvNets) are suggested using depth maps of small human action recognition datasets. Depth maps from different viewpoints are used to make WHDMMs extract spatio-temporal features of actions into 2-D spatial structures. Then, these structures are converted to pseudo-color images. Finally, color-coded WHDMMs are trained via distinct pre-trained ConvNets. In [ 84 ] , a method for human action recognition from depth sequences is presented. Firstly, to form the depth motion maps (DMMs), the raw frames are projected onto three orthogonal Cartesian planes and the results are stacked into three still images (corresponding to the front, side, and top views). Then, the local ternary pattern (LTP) is introduced as an image filter for DMMs to improve the distinguishability of similar actions. Finally, corresponding LTP-encoded images are classified using CNN.

 
 
 

#### 3.1.3 Skeleton

 
 In [ 85 ] and [ 86 ] spatio-temporal information carried in 3D skeleton sequences is represented by three 2D images, referred to as joint trajectory maps (JTM), through encoding the joint trajectories and their dynamics into color distribution in the images. Then ConvNets are adopted to learn the discriminative features for human action recognition. Such an image-based representation enables to use of existing ConvNets models for the classification of skeleton sequences without training the networks afresh. In [ 87 ] and [ 88 ] , a skeleton image representation named SkeleMotion is introduced to be used as input of CNNs. In this method, the temporal dynamics are encoded by explicitly computing the magnitude and orientation values of the skeleton joints. Different temporal scales are employed to compute motion values to aggregate temporal dynamics to the representation. In [ 89 ] , the spatio-temporal information of a skeleton sequence is encoded into color texture images, called skeleton optical spectra. The encoding consists of four steps; mapping of joint distribution, spectrum coding of joint trajectories, spectrum coding of body parts, and joint velocity weighted saturation and brightness. Again, convolutional neural networks are used to learn the discriminative features for action recognition. In [ 90 ] , color texture images referred to as joint distance maps (JDMs) along with ConvNets are employed to exploit the discriminative features from the JDMs for human action and interaction recognition. The pair-wise distances between joints over a sequence of single or multiple-person skeletons are encoded into color variations to capture temporal information. In [ 91 ] , the 3D coordinates of the human body joints carried in skeleton sequences are transformed into image-based representations and stored as RGB images. Then a deep architecture based on ResNets is proposed to learn features from obtained color-based representations and classify them into action classes.

 
 
 

#### 3.1.4 RGB Depth

 
 In [ 92 ] , Asadi-Aghbolaghi et al. investigated the combination of hand-crafted features and deep techniques in human action recognition via RGB-D videos. Multimodal dense trajectories (MMDT) are created from RGB, depth, scene flow, and optical flow modalities which are the inputs to 2DCNNs. Dynamic images such as depth motion maps and motion history images (MHI) are the other pre-computed motion features initially presented in [ 93 ] . These features grounded on rank pooling summarize the motion and action information of a video in a single image to represent the whole sequence. A two-stream CNN network pre-trained on VGG16 [ 94 ] or ResNet-101 [ 95 ] is suggested in [ 96 ] . Dynamic images created independently from RGB videos and depth sequences are fed to the network. Finally, extracted features are concatenated and fed through a fully connected layer for action class prediction. Some studies suggest the difference of successive DMM frames projected on XY, YZ, and XZ planes corresponding to front, side, and top. Singh et al. [ 97 ] employed dynamic images created from RGB videos and three DMMs as inputs to the pre-trained VGG-F model [ 94 ] . A weighted product model is used to categorize the activity. Pre-trained networks with four streams are proposed in [ 98 , 99 ] that accept an MHI created from RGB and DMMs in three distinct views (top, front, and side). The score of each stream is late fused at the end of the network to categorize the activity. For surgical recognition tasks, Twin et al. [ 100 ] suggest a four-stream CNN network pre-trained on AlexNet [ 101 ] using RGB, depth, and their motions as DNN entries. Then, features are concatenated and the classifier predicts the class label. In [ 102 ] , four-channel data is used via combining RGB and depth, where extracting the scene flow from RGB-D videos is considered for action recognition to summarize the RGB-D videos. In this work, RGB and depth are considered as a single unit for extracting the features.
A branch of studies tries to extract common-specific features of different modalities to increase the accuracy of action prediction. Combining the common and specific components in input features may be quite complicated and highly nonlinear. Shahroudy et al. [ 103 ] proposed a deep shared-specific network using nonlinear autoencoder-based component factorization layers. Also, while RGB and depth images are inherently distinct in appearance, there is considerable consistency between them at a high-level [ 104 ] , which will affect the classification accuracy. Qin et al. [ 104 ] developed a unique two-stream framework to extract common-specific features through the constraint of similarity at a high level. In [ 105 ] , a multi-stream deep neural network is suggested for egocentric action recognition. The work [ 105 ] uses the complementary features of RGB and depth by learning the nonlinear structure of heterogeneous information. It strives to keep the unique features for each modality and concurrently explore their sharable information in a unified framework. In addition, it uses a Cauchy estimator to maximize the correlations of the sharable components and enforce the orthogonality constraints on the individual components to ensure their high independencies. The cross-modality complementary features are learned from RGB and depth modalities via a cross-modality compensation block (CMCB) [ 106 ] . The CMCB initially extracts features from the two separate information flows, then sends and intensifies them to the RGB-D paths using the convolution layers. To increase action recognition performance, CMCB includes two general DNN architectures: ResNet and VGG.
Wang et al. [ 107 ] employ two distinct cooperative convolutional networks (c-ConvNet) to extract information from dynamic images comprised of both visual RGB (VDIs) and depth (DDIs). The c-ConvNet comprises one feature extraction network and two branches, one for ranking loss and another for softmax loss. By utilizing bidirectional rank pooling, two dynamic images represent VDIs and DDIs: forward (f) and backward (b), VDIf VDIb and DDIf DDIb, respectively. In [ 108 ] , segmented bidirectional rank pooling is used to gather spatio-temporal information. Moreover, the multimodality hierarchical fusion method gets the complementary information of multimodal data for categorization. The multimodality hierarchical system contains visual RGB and depth dynamic images, i.e., VDIs-f, VDIs-b, DDIs-f, and DDIs-b (f for forward and b for backward) and optical flow fields (X-stream and Y-stream) formed by ConvNets. Dynamic images are created from the RGB-D series as ConvNets entries to extract spatio-temporal information [ 109 ] . Then a segmented cooperative ConvNet is applied to learn the complementary information of RGB-D modalities. In [ 110 ] , RGB and depth frames are used as training inputs, but only RGB is employed during test time. A hallucination network is utilized to simulate the depth stream for test time. A strategy based on inter-stream connection is used to improve the hallucination network’s learning process. A loss function that combines distillation and privileged information is also developed.

 
 
 

#### 3.1.5 RGB Skeleton

 
 Verma et al. [ 111 ] proposed a two-stream framework to exploit spatio-temporal features using both CNNs and RNNs. Motion history image and motion energy image (MEI) are the RGB descriptors. Further, the skeleton modality is used after developing intensity images in three views: top, side, and front. Features of each stream are fused and the final prediction is performed based on scores of each stream using the weighted product rule.
Tomas et al. [ 112 ] utilized appearance and motion information from RGB and skeleton joints to detect fine-grained motions. Motion representations are learned by CNN and motion history images that are generated from RGB images. In addition, stacked auto-encoders measure the distances of the joints from the mean joint in each frame to consider discriminative movements of human skeletal joints.

 
 
 

#### 3.1.6 Depth Skeleton

 
 Kamel et al. [ 113 ] employed depth motion image (DMI), moving joint descriptor (MJD), and fusion of DMI with MJD as inputs of the suggested CNN framework. DMI represents the body changes of depth maps in an image, while MJD indicates body joint position and orientation changes around a fixed point. Wang et al. [ 114 ] utilized the bidirectional rank pooling approach to three hierarchical spatial levels of depth maps driven by skeletons; body, part, and joint. Each level featured various components, which possessed joint positions. Spatio-temporal and structural information at all levels is learned via a spatially structured dynamic depth image (S2DDI) conserving the coordination and synchronization of body parts throughout the action. Besides, this framework contains three weights-shared ConvNets and scored fusion for classification. In [ 115 ] , a CNN-based human action recognition framework is proposed by fusing depth and skeleton modalities. The proposed adaptive multiscale depth motion maps (AM-DMMs) computed from depth maps capture shape and motion cues. Moreover, adaptive temporal windows help the robustness of AM-DMMs in front of motion speed variations. In addition, a method is also proposed for encoding the spatio-temporal information of each skeleton sequence into three maps, called stable joint distance maps (SJDMs) which describe spatial relationships between the joints. A multi-channel CNN is adopted to exploit the discriminative features from texture color images encoded from AM-DMMs and SJDMs for recognition.

 
 
 

#### 3.1.7 RGB Depth Skeleton

 
 Singh et al. [ 116 ] introduced a modality fusion technique called deep bottleneck multimodal feature fusion (D-BMFF) framework for three modalities of RGB, depth, and skeleton. 3D joints are transformed into a single RGB skeleton motion history image (RGB-SklMHI). Every ten RGB and depth frames with a single Skel-MHI image are fed to the framework to extract spatial and temporal features respectively. Extracted features of three-modality streams are combined by multiset discriminant correlation analysis. Then action classification is performed using a linear multiclass SVM. Khaire et al. [ 117 ] aim to enhance activity recognition by using skeleton images, a motion history image, and three depth motion maps from the side, top, and front as inputs of a five-stream CNN network. Elmadany et al. [ 118 ] introduced two fusion approaches to exploit common subspace from two sets and more than two sets, i.e., biset globality locality preserving canonical correlation analysis (BGLPCCA) and multiset globality locality preserving canonical correlation analysis (MGLPCCA), respectively. These strategies represent global and local data features using low-dimensional shared subspace. Besides, two descriptors are suggested for skeleton data and depth. Finally, a framework composed of proposed fusion methods and descriptors is used for action recognition. In [ 119 ] , various rank pooling and skeleton optical spectra approaches are examined to create dynamic images from RGB-D and skeleton. Dynamic images are divided into five categories: a dynamic color group (DC), a dynamic depth group (DD), and three dynamic skeleton groups (DXY, DYZ, DXZ). Several dynamic images featuring the major postures for each group are developed to represent different action postures. Then, a pre-trained flow-CNN extracting spatio-temporal features are used with a max-mean aggregation.
Wu et al. [ 120 ] described a deep hierarchical dynamic neural network for gesture recognition. The suggested framework is composed of a Gaussian-Bernouilli deep belief network (DBN) to extract dynamic skeletal features and a 3DCNN to represent features from RGB and depth images. Furthermore, intermediate and late fusion techniques are used to fuse RGB and depth with the skeleton. Finally, HMM predicts the gesture class label by learning emission probabilities. Romaissa et al. [ 121 ] proposed a four-step framework for action recognition. First dynamic images are created from RGB-D videos, and features of dynamic images are extracted via a pre-trained model utilizing the transfer learning approach. Then, the Canonical correlation analysis method fuses extracted features. Finally, a bidirectional LSTM is trained to recognize action labels.

 
 
 

#### 3.1.8 Discussions 

 
 In brief, utilizing motion-based features is a common approach for modeling temporal variations using distinct or multiple data modalities that allows using pre-trained 2D ConvNets for modeling motion information. These networks generally exploit multiple streams of convolutional networks to encode both appearance and motion of human actions using different modalities. Descriptors such as motion history/energy images for RGB, dynamic images for depth, and joint trajectory maps for skeleton are popular pre-computed motion features used for human action recognition. The most important problem in multi-stream networks is the necessity of communications between different streams to transfer information in learning multimodal spatiotemporal features. The lack of effective interactions between the streams is one of the major problems in multi-stream networks. Such interactions are important for learning spatiotemporal features. Finally, the multi-stream CNN architectures learn different types of information from the input (through separate networks) and then perform fusion to get the result. This enables the traditional 2D CNNs to effectively handle the video data and achieve high accuracy. However, this type of architecture is not powerful enough for modeling long-term dependencies, i.e., it has limitations in effectively modeling the video-level temporal information.

 
 
 
 

### 3.2 Three-Dimensional Filters

 
 In the second category, spatiotemporal filters (for example 3D convolution and 3D pooling) are used in the convolutional layer. Convolution as the essential operation in CNNs calculates pixel values according to a small neighborhood using a kernel (filter). Spatio-temporal filters extend 2D convolution networks by using 3D convolution. 3D convolution captures the temporal dynamics over some successive frames. However, some approaches convert the entire sequence into a 2D image and then use conventional 2D filters.

 
 

#### 3.2.1 RGB

 
 In [ 122 ] , the effects of different spatiotemporal convolutions are studied for action recognition. In this work, the improvement of the accuracy of 3D CNNs over 2D CNNs is empirically demonstrated within the framework of residual learning. In [ 123 ] , FAST 3D convolutions are introduced, a convolution block that combines a 2D spatial convolution with two orthogonal spatio-temporal convolutions. The block is motivated by the often characteristic horizontal and vertical motion of human actions. In [ 17 , 18 , 124 , 125 , 126 ] , the 3D convolution over consecutive frames is applied for action recognition. In [ 18 ] , the developed deep architecture model produces multiple channels of information from adjacent input frames and performs convolution and subsampling separately in each channel. Finally, feature representation is obtained by aggregating information from all channels. In [ 125 ] , pseudo-3D residual net architecture is proposed which aims to learn spatio-temporal video representation in deep networks by simplifying 3D convolutions with 2D filters on spatial dimension plus 1D temporal connections. In [ 126 ] , the learning of long-term video representations is considered by studying architectures with long-term temporal convolutions (LTC). To keep the complexity of networks tractable, the temporal extent of representations is increased at the cost of decreased spatial resolution.

 
 
 

#### 3.2.2 Depth

 
 In [ 127 ] , a 3D full CNN-based framework, called 3DFCNN, is developed for real-time human action recognition from depth videos captured from an RGB-D camera. The network exploits spatio-temporal information of depth sequences to use in the categorization of actions. The aim is the use of depth in privacy-aware systems because people’s identities are not recognized from depth images.

 
 
 

#### 3.2.3 Skeleton

 
 In [ 128 ] , a multi-scale temporal modeling module is designed following [ 129 ] . This module contains four branches, each containing a 1 × 1 convolution to reduce channel dimension. The first three branches contain two temporal convolutions with different dilations and one Max-Pool respectively following 1 × 1 convolution. The results of the four branches are concatenated to obtain the output. In [ 130 ] , a pre-trained 2D convolutional neural network is used as a pose module. A pre-trained 3DCNN is also used as an infrared module to respectively extract features from skeleton data and visual features from videos. Both feature vectors are then fused and jointly classified using a multilayer perceptron (MLP).

 
 
 

#### 3.2.4 RGB Depth

 
 Li et al. [ 131 ] employed RGB and depth as inputs to a pre-trained C3D network for gesture recognition. Extracted features are concatenated or averaged. Finally, the framework uses a linear SVM as a classifier. Zhu et al. [ 132 ] employed pyramid input and fusion with multiscale contextual information via 3D CNNs to learn gestures from the whole video. Zhang et al. [ 133 ] presented 3D lightweight structures for action recognition based on RGB-D data. The suggested lightweight 3D CNNs have considerably fewer parameters with reduced computing costs, and it results in desired recognition performance compared to common 3D CNNs.

 
 
 Qin et al. [ 134 ] employed 3D CNNs to extract common-specific features from RGB-D data. A novel end-to-end trainable framework called TSN-3DCSF is proposed for this purpose. In [ 135 ] , a fusion approach is proposed using the adaptive cross-modal weighting (ACmW) approach to extract complementarity features from RGB-D data. ACmW block explores the relationship between the complementary information from multiple streams and fuses them in the spatial and temporal dimensions. In [ 136 ] , a regional attention with architecture-rebuilt 3D network (RAAR3DNet) is suggested for gesture recognition. Fixed Inception modules are replaced with the automatically rebuilt structure through neural architecture search (NAS) to acquire the varied representations of features in the early, middle, and late levels of the network. In addition, a stacking regional attention module called dynamic-static attention (DSA) is used to highlight the hand/arm regions and the motion information.

 
 
 

#### 3.2.5 Depth Skeleton

 
 Liu et al. [ 137 ] suggest a 3D-based deep convolutional neural network (3D2CNN) to learn depth features along with the joint vector containing skeletal features. Finally, the decision fusion of SVM classifiers demonstrates the action class.

 
 
 

#### 3.2.6 Discussions 

 
 Shortly, 3D filters as the extension of 2D filters capture the temporal dynamics at the cost of requiring more parameters than 2D convolution networks. The advantage is capturing discriminative features along both spatial and temporal dimensions while the disadvantage is the limitation to a certain temporal structure (by considering very short temporal intervals) and complicated encoding of long-term temporal data. The 3D CNN-based methods generally perform spatio-temporal processing over limited intervals (using the window-based 3D convolutional operations), where each convolutional operation is only applied to a relatively short-term context in videos. For multi-modal approaches, the 3D filters can be applied to distinct modalities to simultaneously capture spatial and temporal intra-modal features. For this purpose, multi-stream networks and different strategies for fusion may be applied. However, the fusion of features is again a concern. Score fusion and feature fusion are two widely used multi-modality fusion schemes in human action recognition [ 45 ] . The score fusion integrates the separately made decisions based on different modalities to produce the final results. Meanwhile, feature fusion generally combines the features from different modalities to yield aggregated and powerful features for recognizing different actions. However, existing multi-modality methods are not as effective as expected owing to a series of challenges, such as over-fitting [ 45 , 138 ] .

 
 
 
 

### 3.3 Temporal Sequence Models

 
 The third group usually aggregates CNN features applied at individual frames with temporal sequence models such as recurrent neural networks [ 139 ] . RNNs that are designed to work with sequential data, use the previous information in the sequence to produce the current output. The main problem with RNNs is the short-term memory problem, caused by the vanishing gradient problem. As RNN processes more steps, it suffers from vanishing gradient more than other neural network architectures. To overcome this problem, two specialized versions of RNNs are created; GRU (gated recurrent unit) [ 140 ] and LSTM (long short-term memory) [ 141 ] . LSTM and GRU use memory cells to store the information of previous data in long sequences using gates. Gates that control the flow of information in the network, are capable of learning the importance of inputs in the sequence and storing or passing their information in long sequences. GRU structure is less complex compared with LSTM because it has less number of gates (two gates of reset and update for GRU compared with three gates of input, output, and forget for LSTM). However, other temporal sequence models like the hidden Markov model (HMM) are also applied [ 120 ] in the literature together with CNN features.

 
 

#### 3.3.1 RGB

 
 In [ 123 , 142 ] , both CNNs and LSTMs are utilized for capturing spatial motion patterns along with temporal dependencies. In [ 143 ] , a conflux long short-term memory network is proposed to recognize actions from multi-view cameras. The proposed framework first extracts deep features from a sequence of frames using a pre-trained VGG19 CNN model for each view. Second, the extracted features are forwarded to the conflux LSTM network to learn the view of self-reliant patterns. In the next step, the inter-view correlations using the pairwise dot product are computed from the output of the LSTM network corresponding to different views to learn the view inter-reliant patterns. Finally, flattened layers followed by a softmax classifier are used for action recognition.

 
 
 

#### 3.3.2 Depth

 
 In [ 144 ] , two networks based on ConvLSTM are suggested with different learning strategies and architectures. One network uses a video-length adaptive input data generator (stateless) while the latter discovers the stateful capability of general recurrent neural networks, but is applied in the specific case of human action recognition. This property allows the model to gather discriminative patterns from previous frames without compromising computer memory. In [ 145 ] , the ConvLSTM network is used with depth videos for home caring of elderly adults. In [ 146 ] , a bidirectional recurrent neural network (BRNN) is developed for depth-based human action recognition. First, the 3D depth image is projected on three 2D planes and is fed to three distinct BRNNs. In the following layers, extracted features of each BRNN are fused and fed to the next BRNNs. The network follows with fully connected and softmax layers.

 
 
 

#### 3.3.3 Skeleton

 
 In [ 147 ] , an end-to-end trainable hierarchical RNN model is developed using skeleton data for recognizing activities. The human skeleton is divided into five body parts instead of the whole skeleton data in the training phase. Then, each part is separately fed to a subnet. Next, extracted features by each subnet are hierarchically fused and fed to the higher layer. In the end, a high-level representation of the skeleton is used for the final classification. In [ 148 , 149 ] , a universal spatial RNN-based model uses geometric features. The multi-stream LSTM network is trained with different geometric features and a new smoothed score fusion method is used. The potential of learning complex time-series representations via high-order derivatives of states is investigated in [ 150 ] . In this work, a differential gating scheme is proposed for the LSTM neural network to highlight the change in information gain due to salient motions between consecutive frames. The proposed differential recurrent neural network (dRNN) quantifies the change in information gained by the derivative of states. In [ 151 ] , an end-to-end fully connected deep LSTM framework is proposed for action recognition. The co-occurrences of skeleton joints are learned via a regularization mechanism. Further, a dropout algorithm is suggested for gates, cells, and output responses of neurons. In [ 152 ] , RNNs are also used to model the temporal dependencies of the features of body parts in actions. In [ 153 ] , a three-structure-based traversal method is proposed. Besides, a new gating scheme in LSTM is proposed to handle noise and occlusion of skeleton data that learns the reliability of the sequential input data and adjusts its effect on updating the long-term context information stored in the memory cell. A two-stream RNN architecture is suggested to model spatial and temporal features of actions with skeleton data as input [ 154 ] . Two different structures are designed for the temporal stream, including stacked RNN and hierarchical RNN. Further, spatial structure is modeled by two methods. In addition, 3D-based data augmentation techniques such as rotation and scaling transformation are suggested. In [ 155 ] , an ensemble temporal sliding LSTM (TS-LSTM) network is introduced for skeleton-based action recognition, which consists of several parts including short-term, medium-term, and long-term TS-LSTM networks. Then, with an average ensemble among different parts various temporal dependencies are captured. In addition, features of multiple parts are visualized to demonstrate the relation between recognized action and its correspondent multi-term TS-LSTM features. In [ 156 ] , it is suggested to use an independently recurrent neural network (IndRNN). Besides, network weights are regularized to resolve the gradient vanishing problem. Moreover, IndRNN is over ten times faster than the commonly used LSTM. In [ 157 ] , an attentional recurrent relational network-LSTM (ARRN-LSTM) is proposed that models spatial and temporal dynamics in skeletons for action recognition. The recurrent relational part of the network learns the spatial features of a single skeleton, followed by a multi-layer LSTM that learns the temporal features in the skeleton sequences. An adaptive attentional module is used between the two modules to focus on the most discriminative parts in the single skeleton. In addition, a two-stream architecture is used to learn the structural features among joints and lines to use the complementarity from different geometries in the skeleton.

 
 
 

#### 3.3.4 RGB Depth

 
 Pigou et al. [ 158 ] developed an end-to-end trainable network employing temporal convolutions and bidirectional recurrence. RGB and depth are considered as four-channel data or a 4D entity. In this method, RNNs represent high-level spatial information. In addition, RNNs predict the beginning and ending frames of gestures. In [ 159 ] , two-stream RNNs are used for gesture recognition that utilizes RGB-D data to represent the contextual information of temporal sequences.

 
 
 

#### 3.3.5 Depth Skeleton

 
 Mahmud et al. [ 160 ] employed depth quantized images and skeleton joints for dynamic hand gesture recognition. Both CNN and LSTM structures are used in the network to extract depth features, while skeleton features are extracted via LSTM following distinct MLPs. Fused scores are used in prediction with MLP scores of fused extracted features from quantized images and skeleton joints in the previous process. Lai et al. [ 161 ] proposed a framework composed of CNNs and RNNs using depth and skeleton to hand gesture recognition. Further, several fusion strategies were investigated for enhancing performance, including feature-level fusion and score-level fusion. Shi et al. [ 162 ] proposed a privileged information-based recurrent neural network (PRNN). The privileged information (PI) is only provided during training but not through the testing procedure. This model considered skeletal joints as a PI in three-phase training processes, including; pre-training, learning, and refining. The recommended network was end-to-end trainable and the CNN and RNN parameters were jointly acquired. The final network enhances latent PI iteratively in an EM procedure.

 
 
 

#### 3.3.6 RGB Depth Skeleton

 
 Hu et al. [ 163 ] proposed a framework to learn modality-temporal mutual information from tensors called the deep bilinear framework. The bilinear block learns the time-varying dynamics and multimodal information consists of modality pooling and temporal layers. The deep bilinear model is created through accumulating bilinear blocks and other layers to extract video modality-temporal information. Further modality-temporal cube descriptor is presented as deep bilinear learning input.

 
 
 

#### 3.3.7 Discussions 

 
 Finally, another common approach in modeling temporal variations is using the features applied at individual frames with temporal sequence models such as RNNs. Especially for RGB and depth, it is straightforward to use CNNs to extract spatial information and then use RNNs to extract temporal information of a sequence. Although the third group can deal with longer-range temporal relations, temporal sequence models such as RNN or LSTM can only exploit partial temporal information because regular RNNs cannot access all input elements at each given time step. Having access to all elements of a sequence at each time step can be overwhelming. To help the RNNs focus on the most relevant elements, the attention mechanism can be used that assigns different attention weights to each input element. Since the skeleton encodes high-level information about important details of a scene, the skeleton may be used to guide RGB/depth features. So, the important information strongly related to the action is enhanced.

 
 
 
 

### 3.4 Transformers and Attention

 
 Before the arrival of transformers, most state-of-the-art methods were based on gated RNNs (such as LSTMs and GRUs) with considering attention mechanisms. Recently, transformer models such as BERT (bidirectional encoder representations from transformers) [ 164 ] , GPT (generative pre-trained transformer) [ 165 ] , RoBERTa (robustly optimized BERT pre-training) [ 166 ] , and T5 (text-to-text transfer transformer) [ 167 ] has shown promising results in the field of NLP for tasks such as text classification and translation [ 25 , 168 ] . Following these results, transformers are starting to be used in the field of computer vision (which was dependent on deep ConvNets and RNNs in the last decade) by introducing models such as ViT [ 26 ] and DeiT [ 169 ] for image classification, DETR for object detection [ 170 ] , and VisTR for video instance segmentation [ 171 ] .
For action recognition, considering the sequential nature of video makes it a perfect match for transformers to be used for modeling temporal variations. Although the application of transformers to action recognition is relatively new, the amount of research that has been proposed on this topic within the last few years is surprising. Now, transformers are built on attention technologies without using a recurrent neural network backbone, demonstrating the ability of the attention mechanisms alone compared with RNNs along with attention. Some approaches are strictly dependent on the transformer and self-attention mechanisms [ 172 ] to extract spatio-temporal features. Some others use CNN features besides transformers to make benefit from both architectures [ 173 , 174 , 175 ] .

 
 

#### 3.4.1 RGB

 
 In [ 176 ] , an attention-based model for action recognition is proposed. The model can selectively concentrate on important elements in video frames and dynamically pool convolutional feature maps to produce discriminative features using long short-term memory units. In [ 177 ] , hierarchical RNN and attention mechanisms are applied to capture both short-term and long-term motion information. In [ 178 ] , Girdhar and Ramanan proposed a new method for approximating bilinear pooling with low-rank decomposition. This yields an attentional pooling that substitutes calculating the second-order features with the product of two attention maps of top-down and bottom-up attention maps. Li et al. [ 179 ] introduced motion-based attention along with LSTM for end-to-end sequence learning of actions in the video. In [ 180 ] , Du et al. proposed a spatio-temporal attention mechanism to selectively focus on spatial visual elements as well as keyframes. In [ 181 ] , the spatial transformer network [ 182 ] is introduced with an attention mechanism to explicitly model the spatial structures of human poses. In [ 183 ] , a convolutional LSTM algorithm based on the attention mechanism is proposed to improve the accuracy of action recognition by mining the salient regions of actions in videos. First, GoogleNet [ 184 ] is used to extract the features of video frames. Then, those feature maps are processed by the spatial transformer network for attention. Finally, to classify the action, the sequential information of the features is handled by the convolutional LSTM network. In [ 185 ] , a video action recognition network called action transformer is proposed that uses a modified transformer architecture as a ’head’ to classify the action of a person of interest. It combines two other ideas of using a spatiotemporal I3D model [ 124 ] as the base backbone to extract features and a region proposal network (RPN) [ 186 ] to localize people performing actions. The I3D features and RPN produce the query that is the input for the transformer head and combines contextual information from other people and objects in the surrounding video. In this way, the network can implicitly learn both to track distinct persons and to consider the actions of other people in the video. In addition, the transformer attends to the hand and face as the most reassuring parts when discriminating an action. In [ 187 ] , 3D convolution is combined with late temporal modeling for action recognition. For this purpose, the temporal global average pooling (TGAP) layer at the end of 3D convolutional architecture is replaced with the BERT layer to model the temporal information with BERT’s attention mechanism (see Figure 3 ). It was shown that this replacement improves the performances of popular 3D convolution architectures such as ResNet, I3D, SlowFast, and R(2+1)D for action recognition.

 
 
 Figure 3: BERT-based Temporal Modeling with 3D CNNs for Action Recognition [ 187 ] . 
 
 
 In [ 174 ] , a sparse transformer-based Siamese network (called TBSN) is proposed for few-shot action recognition, which aims to recognize new categories with only a few labeled samples. TBSN applies the sparse transformer to learn the correlation and importance of video clips, and a new measurement to calculate the distance between samples. In this paper, an embedding module is designed based on sparse-transformer whose main ideas are attention mechanism and feedforward network. This method also substitutes the softmax function with the sparsemax function, where the sparsemax function can output zero probabilities. By introducing the sparsemax, zero attention values are assigned to clips containing noises. In [ 188 ] , video transformer network (VTN) architecture is proposed for real-time action recognition. The VTN is made up of an encoder that processes each frame of input sequence independently with 2D CNN (ResNet-34 [ 189 ] ), and the decoder that integrates intra-frame temporal information in a fully-attentional feed-forward approach. In [ 175 ] , two spatio-temporal feature extraction (GSF) and aggregation (XViT) modules are developed for action recognition: GSF is a spatio-temporal feature extracting module that can be plugged into 2D CNNs. XViT is a video feature extractor based on the transformer. The proposed method uses an ensemble of GSF and the XViT models to generate the final scores. In [ 190 ] , a simple fully self-attentional architecture called action transformer (AcT) is introduced that exploits 2D pose representations over small temporal windows. In [ 191 ] , a pure-transformer architecture adapted from the Swin transformer for image recognition [ 192 ] is proposed for video recognition that is based on spatiotemporal locality inductive bias. So this model is supposed to be able to leverage the power of the pre-trained image models. The Swin transformer [ 28 ] introduced the inductive biases of locality, hierarchy, and translation invariance and can be served as a general-purpose backbone for various image recognition tasks. In [ 193 ] , a two-pathway transformer network (TTN) is proposed that uses memory-based attention to explicitly model the relationship between appearance and motion. Specifically, each pathway is designed to produce spatial appearance information or temporal motion information. Then the generated features from two pathways are combined at the end of the framework. Here a transformer-based decoder is used to capture the underlying relationship between the appearance and motion information to improve action recognition. The decoder takes different features as its query, key, and value inputs so that the transformer heads can aggregate contextual information from one modality’s features in the value input to update the other modality’s features in the query input.

 
 
 In [ 194 ] , TimeSFormer is proposed that extends ViTs to videos. In this method, the video is considered as a sequence of patches extracted from individual frames. In addition, to capture spatio-temporal relationships, divided attention is proposed to separately apply spatial and temporal attention within each block. In [ 195 ] , multiscale vision transformers (MViT) are proposed for video and image recognition, to relate multiscale feature hierarchies with the transformer model. MViT hierarchically expands the feature complexity while reducing visual resolution. In [ 196 ] , ViViT is proposed as a video vision transformer to extract spatio-temporal tokens from the input video, which are then encoded by a series of transformer layers. To handle the long sequences of tokens encountered in the video, several variants of the model which factorize the spatial and temporal dimensions of the input are proposed. In [ 172 ] , a pure transformer-based approach called the multi-modal video transformer (MM-ViT) is proposed for video action recognition in the compressed video domain. MM-ViT exploits different modalities such as appearance (I-frames), motion (motion vectors and residuals), and audio waveform. To handle the large number of spatiotemporal tokens extracted from multiple modalities, four multi-modal video transformer architectures are introduced (see Figure 4 ). The simple architecture adopts the standard self-attention mechanism to measure all pairwise token relations. Three efficient model variants are also presented with different approaches which factorize the self-attention calculation over the space, time, and modality dimensions. In addition, to explore the inter-modal interactions, three distinct cross-modal attention mechanisms are developed that can be integrated into the transformer architecture. Experimental experiments on public datasets demonstrate that MM-ViT performs better or equally well to the state-of-the-art CNN counterparts with much less computational cost (compared with the cost of optical flow).

 
 
 Figure 4: Self-attention blocks in MM-ViT and cross-modal attention mechanisms [ 172 ] . 
 
 
 

#### 3.4.2 Skeleton

 
 For skeleton-based action recognition, some approaches use attention mechanisms for modeling dependencies. In [ 197 ] , a global context-aware attention LSTM (GCA-LSTM) framework is suggested for action recognition to selectively focus on the informative joints of each frame. Further, a recurrent attention mechanism is proposed to enhance attention efficiency. Thereby, the proposed two-stream framework consists of coarse-grained and fine-grained attention. In [ 198 ] , an LSTM-based approach called global context-aware attention LSTM (GCA-LSTM) is proposed to selectively focus on the informative joints in the action sequence using global contextual information. Besides, a recurrent attention mechanism for the GCA-LSTM network is introduced to achieve a reliable attention scheme for the action. An end-to-end spatial and temporal attention network is suggested for human action recognition from skeleton data in [ 199 ] . The network is based on LSTM to learn selectively focus on discriminative joints in each frame, giving each frame a different degree of attention. In [ 200 ] , an architecture named graph convolutional skeleton transformer (GCsT) is proposed to capture the long-term temporal context and enhance the flexibility of feature extraction for skeleton-based action recognition. The overall architecture is divided into three stages and each stage consists of two blocks. A spatial–temporal graph convolutional block (STGC) to extract the local-neighborhood relations and a spatial–temporal transformer block (STT) to capture global space–time dependencies. GCsT employs the benefits of both transformer and graph convolution network (GCN). In this way, hierarchy and local topology structure is conducted through GCN and the temporal attention and global context is provided with the transformer. In [ 201 ] , a hierarchical transformer-based framework is proposed for modeling the spatio-temporal structure of a sequence of 3D human skeletons of human action. Specifically in this method, the 3D human skeletons are split into five human body parts, then they are fused hierarchically with self-attention layers based on the articulation of the human body parts. Besides, to predict the motion of the 3D skeleton it tries to model the body parts’ interactions and the motion directions. In [ 202 ] , a transformer-based model called Motion-Transformer is proposed to capture the temporal dependencies via self-supervised pre-training on the sequence of human action. Besides, a flow prediction task is also introduced to pre-train the Motion-Transformer to capture the intrinsic temporal dependencies. The pre-trained model is then fine-tuned on the task of action recognition. In [ 203 ] , a multi-stream spatial-temporal relative transformer architecture is also used instead of graph convolution or recurrence and LSTM, to capture long-range dependencies. The proposed architecture called relative transformer is based on standard transformer. The relative transformer module respectively evolves into a spatial relative transformer and temporal relative transformer to extract spatio-temporal features (ST-RT module). In addition, the dynamic representation module combines multi-scale motion information to handle actions with different durations. Lastly, four streams of ST-RTs modules with four dynamic data streams are combined to improve the performance (see Figure 5 ) where each stream extracts features from a corresponding skeleton sequence to complement each other.

 
 
 Figure 5: The overall architecture of MSST-RT in [ 203 ] . 
 
 
 In [ 204 , 205 ] , a transformer self-attention approach is introduced in skeleton activity recognition as an alternative to graph convolution. This approach tries to model interactions between joints using a spatial self-attention module (SSA) to understand intra-frame interactions between different body parts and a temporal self-attention module (TSA) to model inter-frame correlations. The two modules are combined in a two-stream network to produce the final score for action recognition. In [ 206 ] , synchronous local nonlocal along with frequency attention (SLnL-rFA) model is proposed to extract synchronous detailed and semantic information from multi-domains. SLnL-rFA includes SLnL blocks for spatio-temporal learning and a residual rFA block for extracting frequency patterns. In [ 207 ] , spatial transformer block and directional temporal transformer block are designed for modeling skeleton sequences in spatial and temporal dimensions respectively. To adapt to the imperfect information condition (due to occlusion, noise, etc.), a multi-task self-supervised learning method is also introduced by providing confusing samples in different situations to improve the robustness of the model. In [ 208 ] , a transformer-based model is proposed with sparse attention and segmented linear attention mechanisms applied on spatial and temporal dimensions of action skeleton sequence to replace graph convolution operations with self-attention operations while requiring significantly less computational and memory resources.

 
 
 

#### 3.4.3 RGB Skeleton

 
 For multimodal action recognition, the cross-modality features are also a concern besides spatio-temporal features. In [ 209 ] , a spatio-temporal attention-based mechanism is proposed for human action recognition to automatically attend to the most important human hands and detect the most discriminative moments in an action. Attention is handled using a recurrent neural network and is fully differentiable. In contrast to standard soft-attention-based mechanisms, this approach does not use the hidden RNN state as input to the attention model. Instead, attention distributions are drawn using a human articulated pose as external information. In [ 210 ] , a closely related approach is proposed. However, the attention mechanism on the RGB space is conditioned on end-to-end learned deep features from the pose modality and not only hand-crafted pose features. In [ 211 ] , an end-to-end network for human activity recognition is proposed leveraging spatial attention on human body parts. This paper proposes an RNN attention mechanism to obtain an attention vector for soft assigning different importance to human body parts using spatio-temporal evolution of the human skeleton joints. It also designs the joint training strategy to efficiently combine the spatial attention model with the spatio-temporal video representation by formulating a regularized cross-entropy loss to achieve fast convergence.
In [ 212 ] , an attention-based body pose encoding is proposed for human activity recognition. To achieve this encoding, the approach exploits a spatial stream to encode the spatial relationship between various body joints at each time point to learn the spatial structure of different body joints. In addition, it also uses a temporal stream to learn the temporal variation of individual body joints over the entire sequence. Later, these two pose streams are fused with a multi-head attention mechanism. It also captures the contextual information from the RGB video stream using an Inception-ResNet-V2 model combined with multi-head attention and a bidirectional long short-term memory network. Finally, the RGB video stream is combined with the fused body pose stream to give an end-to-end deep model for human activity recognition.

 
 
 

#### 3.4.4 RGB Depth

 
 In [ 213 ] , a transformer-based framework is proposed for egocentric RGB-D action recognition. It consists of two inter-frame transformer encoders and the mutual-attentional cross-modality modules (see Figure 6 ). The temporal information of distinct modalities is encoded through the self-attention mechanism. Then features from different modalities are fused via the mutual-attention layer. The inputs of this network are aligned RGB frames and depth maps (two streams). The frames are passed through a CNN and after average pooling are converted into two sequences of feature embeddings. Then both sequence features are fed to the transformer encoders to model the temporal structure respectively. Features obtained from the encoders interact through the cross-modality block and are fused to produce the cross-modality representation. The features are processed through the linear layer to get per-frame classification. The final classification is performed by averaging the decisions over the frames of the video.

 
 
 Figure 6: proposed framework in [ 213 ] . Features from each modality are interacted with and incorporated through the mutual-attentional block. 
 
 
 

#### 3.4.5 Discussions 

 
 From studied papers, the interest in using attention-based and transformer networks for human action recognition is growing. These networks rely on self-attention mechanism to model dependencies across features over time, so the network can selectively extract the most relevant information and relationships. In addition, the great advantage of purely transformer-based networks is the fast-learning speed, and the lack of sequential operation, as with recurrent neural networks. Although video transformers have achieved promising results, they suffer from severe memory and computational overhead [ 45 ] . In addition, to obtain the input to the transformer, a video is mapped to a sequence of tokens and then the positional embedding is added. A straightforward method of tokenizing the input video [ 196 ] is to uniformly sample frames from the input video clip, embed each 2D frame independently using the same method as ViT [ 26 ] , and concatenate all these tokens together. However, this method results in a large number of tokens which increases the computation. Attention can also be guided through different modalities. In the case of multimodality, transformers are used for intra-modality spatial and temporal modeling and cross-modality feature fusion. Handling a large number of spatiotemporal tokens extracted from multiple modalities is a concern.

 
 
 
 

### 3.5 Hybrid Methods

 
 In a group of studies, combinations of two or more techniques are used to exploit spatiotemporal dynamics. Considering the four aforementioned approaches, 11 different arrangements are probable for the hybrid techniques (see Figure 7 ).

 
 
 Figure 7: Different Combinations of the aforementioned approaches. 
 
 

#### 3.5.1 RGB

 
 Conv3D + motion + RNN: In [ 214 ] , Ma et al. make use of both LSTMs and Temporal-ConvNets. In this method, spatial and temporal features are extracted from a two-stream ConvNet using ResNet-101 pre-trained on ImageNet and fine-tuned for single-frame activity prediction. The spatial-stream network takes RGB images as input, while the temporal-stream network takes stacked optical flow images as inputs. Spatial and temporal features are concatenated and temporally constructed into feature matrices. The constructed feature matrices are then used as input to both proposed methods: temporal segment LSTM (TS-LSTM) and Temporal-Inception (see Figure 8 ). In [ 215 ] , information from a video is integrated into a map called a motion map using a deep 3D convolutional network. A motion map and the next video frame can be integrated into a new motion map. This technique can be trained by increasing the training video length iteratively. The acquired network can be used for generating the motion map of the whole video. Next, a linear weighted fusion scheme is used to fuse the network feature maps into spatio-temporal features. Finally, a long short-term memory encoder-decoder is used for final predictions.

 
 
 Figure 8: proposed framework in [ 214 ] that makes use of motion-based features, LSTMs, and Temporal-ConvNets to exploit spatiotemporal dynamics. 
 
 
 

#### 3.5.2 Depth

 
 Conv3D + motion: In [ 216 ] , 3D dynamic voxel (3DV) is proposed to facilitate depth-based 3D action recognition. The key idea of 3DV is to encode 3D motion information within depth video into a regular voxel set (i.e., 3DV), via temporal rank pooling. Each available 3DV voxel intrinsically involves both 3D spatial and motion features. 3DV is then represented as a point set and input into PointNet++ [ 217 ] for 3D action recognition. In addition, a multi-stream 3D action recognition manner is also proposed to learn motion and appearance features jointly. To extract richer temporal order information of actions, the depth video is divided into temporal splits and encode this procedure in 3DV integrally.

 
 
 

#### 3.5.3 Skeleton

 
 Motion + RNN: In [ 218 ] , a feature selection network (FSN) is proposed with actor-critic reinforcement learning. Given the extracted feature sequence, FSN learns to adaptively select the most representative features and discard the ambiguous features for action recognition. In addition, a generalized graph generation module is proposed to capture latent dependencies and further propose a generalized graph convolution network (GGCN). The GGCN and FSN are combined in a three-stream recognition framework, in which different types of information from skeleton data are fused to improve recognition accuracy. The proposed method in [ 219 ] consists of three major steps; feature extraction from a skeleton sequence (as the input of ten neural networks; three LSTM models and seven CNN models), neural networks training, and the late score fusion. Applying various methods to a skeleton sequence can obtain various features prominently in the spatial or temporal domain, and that features are defined as SPF (spatial-domain-feature) and TPF (temporal-domain-feature). SPF is selected as the input of LSTM networks and TPF as the input of CNN networks. To obtain the SPF, three types of spatial domain features are extracted including R (relative position), J (distances between joints), and L (distances between joints and lines). To obtain the TPF, the methods used in [ 19 ] and [ 13 ] are followed to generate the joint distances map and joint trajectories map (JTM) respectively.
Conv3D + motion: In [ 220 ] , dynamic GCN is proposed in which a convolutional neural network named context-encoding network (CeN) is introduced to learn skeleton topology. In particular, when learning the dependency between two joints, contextual features from the rest joints are incorporated in a global manner. Multiple CeN-enabled graph convolutional layers are stacked to build dynamic GCN. In addition, static physical body connections and motion modalities are combined to improve results. In [ 221 , 222 ] , a two-stream model using 3D CNN is proposed for skeleton-based action recognition. In this method, skeleton joints are mapped into a 3D coordinate space and the spatial and temporal information are encoded. Second, 3D CNN models are separately applied to extract deep features from two streams. Third, to enhance the ability of deep features to capture global relationships, every stream is extended into a multitemporal version. In [ 223 ] , a data reorganizing strategy is proposed to represent the global and local structure information of human skeleton joints. It employs the data mirror to increase the relationship between skeleton joints. Based on this design, an end-to-end multi-dimensional CNN network is proposed to consider the spatial and temporal information to learn the feature extraction transform function. Particularly, in this CNN network, different convolution kernels are used on different dimensions to learn skeleton representation to generate robust features.

 
 
 

#### 3.5.4 RGB Depth

 
 Conv3D + motion + RNN: In [ 224 ] , weighted dynamic images, created from the depth and RGB videos are inputs of the framework. The framework is composed of bidirectional rank pooling, CNNs, and 3D ConvLSTM to extract complementary information from weighted dynamic images. Canonical correlation analysis is employed for feature-level fusion, and a linear SVM is used for predicting the class of action. In [ 225 ] , a three-stream framework via 3D CNN, ConvLSTM, 2D CNN, temporal pooling, and a fully connected layer with softmax is employed to extract spatio-temporal features of RGB, depth, and optical flow modalities. In [ 226 ] , Molchanov et al. employed 3D CNNs and RNNs in the proposed framework for hand gesture recognition using RGB, optical flow, depth, IR, and IR disparity modalities. Each modality’s class conditional probability vectors are averaged and fused to detect and classify hand gestures.

 
 
 motion + RNN: Dhiman et al. [ 227 ] proposed motion and shape temporal dynamics (STD) as action cues. They employed a framework with RGB dynamic images in motion stream and depth silhouette in STD stream for recognizing action from an unknown view.

 
 
 Conv3D + motion: In [ 228 ] and [ 229 ] , a spatio-temporal attention framework is presented to identify the most representative regions and frames in a video. Following this, RGB, flow, and depth features are retrieved by the ResC3D network and are fused using canonical correlation analysis. A linear SVM classifier estimates the class label. Duan et al. [ 230 ] propose a four-stream network for gesture recognition. This method uses a two-stream convolutional consensus voting network (2SCVN) for the RGB stream and optical flow fields to model short and long-term video sequences. Besides, a two-stream 3D depth-saliency ConvNet (3DDSN) is employed to learn fine-grain motion and eliminate background clutter. Some approaches try to predict pose from RGB data and use it in action prediction. In [ 231 ] , a framework is provided for hierarchical region-adaptive multi-time resolution depth motion map (RAMDMM) and multi-time resolution RGB action recognition system. The suggested approach presents a feature representation method for RGB-D data that allows multi-view and multi-temporal action recognition. Original and synthesized viewpoints are employed for multi-view human action recognition. In addition, to be invariant to changes in an action’s speed, it also employs temporal motion information by incorporating it into the depth sequences. Appearance information in terms of multi-temporal RGB data is utilized to emphasize the underlying appearance information that would otherwise be lost with depth data alone, which helps to enhance sensitivity to interactions with tiny objects. Wu et al. applied 3D CNNs with multimodal inputs to improve spatio-temporal features [ 232 ] . This method suggests two distinct video presentations; depth residual dynamic image sequence (DRDIS) and pose estimation map sequence (PEMS). DRDIS displays spatial motion changes of an action over time which is robust under lighting conditions, texture, and color variations. PEMS is created by pose (skeletal) estimation from an RGB video and removes the background clutter. In [ 233 ] , a gesture recognition framework called MultiD-CNN is suggested to learn spatio-temporal features from RGB-D videos. This method includes spatial and temporal information using two recognition models: 3D-CDCN and 2D-MRCN. 3D-CDCN adds the temporal dimension and makes use of 3D ResNets and ConvLSTM to concurrently learn spatio-temporal features. On the other hand, 2D-MRCN collects the motion throughout the video sequences into a motion representation and employs 2D ResNets to learn.
Conv3D + RNN: In [ 234 ] , a two-stream network is suggested to extract spatio-temporal features which are more robust to background clutter from RGB-D videos. The framework contains 3D CNN, convolutional LSTM, spatial pyramid pooling, and a fully connected layer to provide better long-term spatio-temporal learning.

 
 
 

#### 3.5.5 RGB Skeleton

 
 motion + RNN: Song et al. [ 235 ] offered an end-to-end trainable three-stream framework from RGB and optical flow videos. The framework employs skeleton data as a guide for the RGB stream besides skeleton data is trained in a separate stream. The framework is created from a ConvNet with LSTM. Visual features around critical joints are extracted automatically using a skeleton-indexed transform layer, and via a part-aggregated pooling. The visual features of different body parts and actors are uniformly regulated.
Conv3D + motion: In [ 236 ] , a fusion-based action recognition framework is proposed. The suggested framework is composed of three parts, including 3DCNN, human skeleton manifold representation, and classifier fusion. In [ 237 ] , a multi-stream attention-enhanced adaptive graph convolutional neural network is proposed for skeleton-based action recognition. The graph topology of the skeleton data is learned adaptively in this model. Besides, a spatial-temporal-channel (STC) attention module is embedded in every graph convolutional layer, which helps focus on more important joints, frames, and features. The joints, bones, and the corresponding motion information are modeled in a unified multi-stream framework. In addition, the skeleton data is fused with the skeleton-guided cropped RGB data, which brings additional improvement.

 
 
 Conv3D + attention: Das et al. [ 238 , 239 ] propose video-pose networks (VPN and VPN++) for the recognition of activities of daily living. VPN requires both RGB and 3D poses to classify actions. The RGB images are processed by a visual backbone that generates a spatio-temporal feature map. The VPN that takes as input the feature map and the 3D poses consists of two components: an attention network and a spatial embedding. The attention network further consists of a Pose Backbone and a spatio-temporal Coupler. VPN computes a modulated feature map that is used for classification. VPN++ is an extension of VPN to transfer the pose knowledge into RGB through a feature-level distillation and to mimic pose-driven attention through an attention-level distillation. Features of inputs are extracted via two distinct videos and pose backbones. The video backbone consists of 3D CNNs to extract spatio-temporal features, and the pose backbone contains a spatio-temporal GCN.

 
 
 Conv3D + RNN + attention: In [ 240 ] , an end-to-end separable spatio-temporal attention network is proposed. The input of the network is human body tracks of RGB videos and their 3D poses. Two separate branches are dedicated to spatial and temporal attention individually. Finally, both branches are combined to classify the activities. Figure 9 shows a detailed picture of pose driven RNN attention model which takes 3D pose input and computes m × n spatial and t temporal attention weights for the t × m × n × c spatio-temporal features from I3D. In this work, skeleton data is employed as a guide for the RGB stream. Besides, skeleton data participates in the learning itself in a stream.

 
 
 Figure 9: A detailed picture of pose-driven RNN attention model in [ 240 ] 
 
 
 

#### 3.5.6 Discussions 

 
 Different arrangements are probable for the hybrid techniques (11 combinations). However, some approaches are more established; such as the combination of motion-based features along with 3D filters or LSTMs and Conv3D. In brief, using Conv3D is common in hybrid methods while the combination with the transformers is a new approach. Applying various temporal modeling methods to a sequence can obtain the advantages of different approaches. For example, for motion-based + Conv3D, using motion-based features is a simple approach to obtain a global sense of motion and transactions among consecutive frames. These transactions may be learned through time using 3D filters. In this way, the motion is learned hierarchically through different approaches. However, there are limited methods for these hybrid approaches, and how these approaches should be combined is not well explored.

 
 
 
 
 

## 4 Discussions and Future Prospects 

 
 In brief, methods can learn temporal features by 3D filters in their 3D convolutional and pooling layers. 3D ConvNets are straightforward extensions of 2D ConvNets as they capture temporal information using 3D convolutions. One limitation of 3D ConvNets is that they typically consider very short temporal intervals, thereby failing to capture long-term temporal information. It has been also shown that using training networks on pre-computed motion features is a way to implicitly learn motion features. These networks generally utilize multiple convolutional networks to model both appearance and motion information in action videos. In this way, pre-trained 2D ConvNets can be utilized, allowing networks that are fine-tuned on stacked optical flow frames to achieve good performance despite limited training data. However, the major problem in the multi-stream networks is that they do not allow interactions among the streams while such an interaction is important for learning spatiotemporal features. Temporal models like RNN and LSTM cope with longer-range temporal relations but the problem with RNNs, and LSTMs, is that it’s hard to parallelize the work for processing sequences. The essential advantage of approaches in the transformer group is that they do not suffer from long dependency issues. The transformers process a sequence as a whole in parallel, instead of using past hidden states to capture dependencies in RNN and LSTM. This training parallelization allows training on larger datasets than was once possible. In addition, there is no risk to lose (or "forget") past information. Moreover, multi-head attention and positional embeddings both provide information about the relationship between different elements. Note that although transformers have the potential of learning longer-term dependency, but are limited by a fixed-length context. Some studies such as Transformer-XL architecture [ 241 ] try to learn dependency beyond a fixed length without disrupting temporal coherence. Finally, fusing various temporal modeling methods can obtain the advantages of different approaches. However, there are limited methods for these combinations, and how these approaches should combine is not well explored. Table 2 lists the pros and cons of deep-based temporal modeling approaches.

 
 
 Table 2: pros and cons of deep-based temporal modeling approaches. 
 
 
 
 | 
 
 
 Motion-based 
 | 
 
 
 3D-CNN 
 | 
 
 
 RNN/LSTM 
 | 
 
 
 Transformer 
 | 
 
 
 Hybrid 
 | 

 
 
 
 Pros | 
 
 
 -Pre-trained 2D-ConvNets can be utilized 
 | 
 
 
 -A natural extension of 2D ConvNets. 
 -Quite fast to train and effective with short sequences. 
 -Can capture inductive biases such as translation equivariance and locality 
 | 
 
 
 -Modeling longer-range temporal relations 
 -Past information is retained through past hidden states. 
 | 
 
 
 -Do not suffer from long dependency issues 
 -Requiring significantly less computational and memory resources 
 -Requiring significantly less computational and memory resources 
 -High speed in training and inference 
 | 
 
 
 -Benefiting from advantages of different approaches 
 -Hierarchical learning of features 
 -Parallel processing 
 | 

 
 Cons | 
 
 
 -The High computational cost of computing accurate optical flow 
 -Interaction among different streams is difficult. 
 | 
 
 
 - Short temporal interval 
 -demanding many computational resources in the training stage 
 -Rigid in capturing action sequences with fine-grained visual patterns 
 | 
 
 
 -Sequential processing 
 -Past information retained through past hidden states 
 -Vanishing gradients problems 
 -High number of learnable parameters 
 | 
 
 
 -Training transformer-based architectures can be expensive, especially for long sequences. 
 -Requiring large-scale training to surpass inductive bias 
 | 
 
 
 -Limited studies 
 | 

 

 
 

### 4.1 Which approaches and modalities are more common? 

 
 In total, more than 170 papers on human action recognition are reviewed in this study from 2015 to 2022 (see Table 3 ). In addition, Figure 10 shows the number of studied papers in each year.

 
 
 Table 3: All papers reviewed in this study. 
 
 
 
 Time Modeling Approach | 
 Modality | 
 Paper | 

 
 
 
 Motion-based | 
 RGB | 
 [ 222 , 76 , 77 , 78 , 79 , 80 , 242 ] [ 81 , 82 , 83 , 84 ] | 

 
 Depth | 
 [ 81 , 82 , 83 , 84 ] | 

 
 Skeleton | 
 [ 85 , 86 , 87 , 88 , 89 , 90 , 91 ] | 

 
 RGB+Depth | 
 [ 92 , 93 , 96 , 97 , 98 , 99 , 100 , 102 , 103 , 104 , 105 , 106 , 107 , 108 , 109 , 110 ] | 

 
 RGB+Skeleton | 
 [ 111 , 112 ] | 

 
 Depth+Skeleton | 
 [ 113 , 114 , 115 ] | 

 
 RGB+Depth+Skeleton | 
 [ 116 , 117 , 118 , 119 , 120 , 121 ] | 

 
 Three Dimensional Filters | 
 RGB | 
 [ 17 , 18 , 122 , 123 , 124 , 125 , 126 , 243 , 244 , 245 , 246 ] | 

 
 Depth | 
 [ 127 ] | 

 
 Skeleton | 
 [ 128 , 129 , 130 , 247 , 248 , 249 ] | 

 
 RGB+Depth | 
 [ 131 , 132 , 133 , 134 , 135 , 136 ] | 

 
 RGB+Skeleton | 
 [ 250 , 251 ] | 

 
 Depth+Skeleton | 
 [ 137 ] | 

 
 RGB+Depth+Skeleton | 
 | 

 
 Temporal Sequence Models | 
 RGB | 
 [ 20 , 142 , 143 ] | 

 
 Depth | 
 [ 144 , 145 , 146 ] | 

 
 Skeleton | 
 [ 147 , 148 , 149 , 150 , 151 , 152 , 153 , 154 , 155 , 156 , 157 , 252 ] | 

 
 RGB+Depth | 
 [ 158 , 159 ] | 

 
 RGB+Skeleton | 
 | 

 
 Depth+Skeleton | 
 [ 160 , 161 , 162 ] | 

 
 RGB+Depth+Skeleton | 
 [ 163 ] | 

 
 Transformers and Attention | 
 RGB | 
 [ 174 , 175 , 176 , 177 , 178 , 180 , 181 , 182 , 183 , 185 , 187 , 188 , 190 , 191 , 193 , 253 , 254 , 255 , 256 , 257 , 258 , 259 , 260 , 172 , 194 , 195 , 196 , 261 ] | 

 
 Depth | 
 | 

 
 Skeleton | 
 [ 197 , 198 , 199 , 200 , 201 , 202 , 203 , 204 , 205 , 206 , 207 , 208 , 262 ] | 

 
 RGB+Depth | 
 [ 213 ] | 

 
 RGB+Skeleton | 
 [ 209 , 210 , 211 , 212 ] | 

 
 Depth+Skeleton | 
 | 

 
 RGB+Depth+Skeleton | 
 | 

 
 Hybrid Methods | 
 RGB | 
 [ 214 , 215 , 263 , 264 ] | 

 
 Depth | 
 [ 216 ] | 

 
 Skeleton | 
 [ 218 , 219 , 220 , 221 , 222 , 223 , 265 ] | 

 
 RGB+Depth | 
 [ 224 , 225 , 226 , 227 , 228 , 229 , 230 , 231 , 232 , 233 , 234 ] | 

 
 RGB+Skeleton | 
 [ 235 , 236 , 237 , 238 , 239 , 240 ] | 

 
 Depth+Skeleton | 
 | 

 
 RGB+Depth+Skeleton | 
 | 

 

 
 
 Figure 10: the number of studied papers in each year from 2015 to 2022. 
 
 
 Figure 11 shows the number of studied papers in each category. As this figure shows, the greatest number of methods use motion-based features for temporal modeling. In this category, RGB depth is the popular used modality. For the categories of Conv3D and transformer, RGB is the common modality used among studied papers while temporal sequence models are used mostly along with the skeleton. This figure is represented from another view in Figure 12 where the numbers of studied papers in each modality and their combinations are shown. As this figure shows, RGB is the most popular modality among studied papers, since it contains rich information about the appearance and is easy to collect. However, methods based on the RGB modality are often sensitive to viewpoint variations and background clutters, etc. Hence, action recognition with other modalities, such as 3D skeleton data, has also received great attention. As Figure 12 shows, the skeleton is in the second rank. The skeleton provides the body structure information (as a simple, efficient, and informative representation of human behaviors). Nevertheless, action recognition using only skeleton data still faces challenges, due to its sparse representation, the noisy skeleton information, and the lack of shape information, especially for handling human-object interactions. So, the combination of multiple modalities is used frequently in the literature. In Figure 12 , The combination of RGB depth is in the third rank. It is the most popular combination among all multi-modal approaches. The RGB and depth videos respectively capture the rich appearance and 3D shape information, that are complementary and can be used for action recognition.

 
 
 Figure 11: The number of studied papers in each category. 
 
 
 Figure 12: The number of studied papers in each modality and their combinations. 
 
 
 For hybrid methods, the try is to learn temporal features by combining different approaches. While different combinations are possible, some are more promising. Figure 13 shows the number of studied papers in each combination. Note that only five combinations are shown in this figure because the remaining combinations did not include any paper among the studied papers. Note that, Conv3D is present in all five combinations since the 3D CNN-based methods are very powerful in modeling discriminative features from both the spatial and temporal dimensions. In addition, as this figure shows, Conv3D + motion is used a lot in the literature. Since utilizing motion-based features is also a common approach for modeling temporal variations, the combination of these two approaches (Conv3D and motion) may improve accuracy.

 
 
 Figure 13: the number of studied papers for hybrid methods. 
 
 
 

### 4.2 Which approaches and modalities are the winner? 

 
 To compare different approaches with each other, six benchmark datasets of human action recognition including NTU RGB+D [ 152 ] , NTU RGB+D 120 [ 266 ] , Toyota-Smarthome [ 240 ] , Kinetics 400 [ 267 ] , Kinetics 600 [ 268 ] , and Skeleton-Kinetics are selected. As one of the most fundamental tasks in computer vision, there are numerous benchmark datasets for unimodal or multimodal vision-based human action recognition [ 269 , 270 , 271 , 272 ] . These benchmark datasets are chosen due to their popularity and large number of actions. While the three first datasets are multimodal, the Kinetics 400 and 600 provide only RGB data and Skeleton-Kinetics include only skeletal data.
NTU RGB+D [ 152 ] : is a large-scale multimodal human action recognition dataset containing 56,880 action sequences of 60 action classes. The action samples are performed by 40 persons in the lab environment and are captured by three Microsoft Kinect v2 cameras from three different views. Each sample contains an action and contains at most 2 subjects. We report the two standard evaluation protocols recommended by the authors of this dataset namely cross-subject (CS) and cross-view (CV). In the CS setting, training data comes from 20 subjects and test data comes from the other 20 subjects. In the CV setting, training data comes from camera views 2 and 3, and test data comes from camera view 1. This dataset contains RGB videos, depth map sequences, 3D skeletal data, and infrared (IR) videos for each sample.
NTU RGB+D 120 [ 266 ] : extends NTU RGB+D with additional 57,600 sequences over 60 extra action classes. Totally 114,480 samples over 120 classes are performed by 106 individuals, captured with three camera views. There are two recommended evaluation protocols, namely cross-subject (CS) and cross setup (CSet). In the CS setting, 63,026 clips from 53 subjects are used for training, and the remaining subjects are reserved for testing. In the CSet setting, 54,471 clips with even setup IDs are used for training, and the rest clips with odd setup IDs are used for testing.
Toyota-Smarthome [ 240 ] : is a dataset of activities of daily living recorded in an apartment where 18 older subjects carry out tasks of daily living during a day. The dataset contains 16.1k video clips, 7 different camera views, and 31 complex activities performed in a natural way without strong prior instructions. This dataset provides RGB data and 3D skeletons which are extracted from LCRNet [ 273 ] . For the evaluation of this dataset, the cross-subject (CS) and two cross-view protocols (CV1 and CV2) [ 240 ] are reported.
Kinetics [ 267 , 268 ] : is a collection of large-scale, high-quality datasets of URL links consisting of 10-second videos sampled at 25fps from YouTube. Here both Kinetics 400 [ 267 ] and 600 [ 268 ] are considered, containing respectively 400 and 600 classes. Note that there is a relatively newer version of this dataset called Kinetics-700 [ 274 ] that is not considered here, because existing results are limited on this new version. As these are dynamic datasets and videos may be removed from YouTube, the size of these datasets are approximately 267k and 446k respectively. Kinetics-400 consists of ∼ \sim 240k training videos and 20k validation videos. Kinetics-600 has ∼ \sim 392k training videos and 30k validation videos. In addition, Skeleton-Kinetics is derived from the Kinetics-400 dataset. The skeletons are estimated by [ 275 ] from RGB videos using the OpenPose toolbox [ 276 ] . Each joint consists of 2D coordinates in the pixel coordinate system and its confidence score. There are 18 joints for each person. In each frame at most two subjects are considered. Skeleton-based approaches reported on Kinetics usually use Skeleton-Kinetics for evaluation. Here the same train-validation split as [ 275 ] is considered. That is, the training and validation sets contain 240k and 20k video clips respectively. Top-1 accuracies are reported.
Table 4 shows state-of-the-art approaches on these benchmark datasets. As this table shows, Conv3D or Conv3D and its combination with attention is the top time modeling approach for NTU RGB+D, NTU RGB+D 120, and Toyota-Smarthome, all using RGB skeleton as the input modalities. For Kinetics-400 and Kinetics-600 attention is the winner. However, for Kinetics-Skeleton, Conv3D is still at the first rank. Note that multimodal methods achieve superior results compared with unimodal methods on all three first multimodal datasets. Finally, Conv3D and attention are the most popular approaches among state-of-the-art methods for modeling temporal variations in human action recognition. Note that for RGB-based methods using pure-transformer architectures is becoming a common approach showing the capability and the increasing interest of the community in using transformers for temporal modeling. In addition, according to the table, the combination of RGB and pose is more frequent among top multi-modal human action recognition algorithms as they provide complementary information about the appearance and 3D coordinates of joints.

 
 
 Table 4: Top state-of-the-art methods on six benchmark datasets. 
 
 
 
 
 
 Dataset 
 | 
 
 
 Method 
 | 
 
 
 Accuracy 
 | 
 
 
 Modality 
 | 
 
 
 Time modeling 
 | 

 
 
 
 
 
 NTU RGB+D [ 152 ] 
 | 
 
 
 Duan et al. [ 250 ] 
 Das et al. [ 239 ] 
 Shi et al. [ 237 ] 
 Davoodikakhki et al. [ 251 ] 
 Das et al. [ 238 ] 
 Zhu et al. [ 243 ] 
 Das et al. [ 211 ] 
 Chen et al. [ 128 ] 
 Das et al. [ 240 ] 
 Piergiovanni et al. [ 244 ] 
 Liu et al. [ 129 ] 
 Ye et al.
 [ 220 ] 
 | 
 
 
 97.0 (cs) 99.6 (cv) 
 96.6 (cs) 99.1 (cv) 
 96.1 (cs) 99.0 (cv) 
 95.66(cs)98.79(cv) 
 95.5 (cs) 98.0 (cv) 
 94.3 (cs) 97.2 (cv) 
 93.0 (cs) 95.4 (cv) 
 92.4 (cs) 96.8 (cv) 
 92.2 (cs) 94.6 (cv) 
 93.7 (cv) 
 91.5 (cs) 96.2 (cv) 
 91.5 (cs) 96.0 (cv) 
 | 
 
 
 RGB skeleton 
 RGB skeleton 
 RGB skeleton 
 RGB skeleton 
 RGB skeleton 
 RGB 
 RGB skeleton 
 skeleton 
 RGB skeleton 
 RGB 
 skeleton 
 skeleton 
 | 
 
 
 Conv3D 
 Conv3D+attention 
 Conv3D+motion 
 Conv3D 
 Conv3D+attention 
 Conv3D 
 attention 
 Conv3D 
 Conv3D+RNN+attention 
 Conv3D 
 Conv3D 
 Conv3D+motion 
 | 

 
 
 
 NTU RGB+D 120 [ 266 ] 
 | 
 
 
 Duan et al. [ 250 ] 
 Das et al. [ 239 ] 
 Chen et al. [ 128 ] 
 Ye et
al. [ 220 ] 
 Das et al. [ 238 ] 
 Das et al. [ 240 ] 
 Papadopoulos et
al. [ 247 ] 
 Caetano et al. [ 88 ] 
 Caetano et al. [ 87 ] 
 Liu et al.
 [ 277 ] 
 | 
 
 
 95.3(cs) 96.4(cset) 
 90.7(cs) 92.5(cset) 
 88.9(cs) 90.6(cset) 
 87.3(cs) 88.6(cset) 
 86.3(cs) 87.8(cset) 
 83.8(cs) 82.5(cset) 
 78.3(cs) 79.2(cset) 
 67.9(cs) 62.8(cset) 
 66.9(cs) 67.7(cset) 
 64.6(cs) 66.9(cset) 
 | 
 
 
 RGB skeleton 
 RGB skeleton 
 skeleton 
 skeleton 
 RGB skeleton 
 RGB 
skeleton 
 skeleton 
 skeleton 
 skeleton 
 skeleton 
 | 
 
 
 Conv3D 
 Conv3D+attention 
 Conv3D 
 Conv3D+motion 
 Conv3D+attention 
 Conv3D+RNN+attention 
 Conv3D 
 motion 
 motion 
 Conv2D 
 | 

 
 
 
 Toyota-Smarthome [ 240 ] 
 | 
 
 
 Das et al. [ 239 ] 
 Yang et al. [ 262 ] 
 Ryoo et al. [ 263 ] 
 Kangaspunta et al. [ 264 ] 
 Das et al. [ 238 ] 
 Das et al. [ 240 ] 
 Wang et al. [ 261 ] 
 Carreira et al. [ 124 ] 
 Mahasseni et al.
 [ 252 ] 
 Ohnishi et al. [ 242 ] 
 | 
 
 
 71.0(cs) 58.1(cv2) 
 64.3(cs) 36.1(cv1) 65.0(cv2) 
 63.6(cs) 
 62.11(cs) 
 60.8(cs) 53.5(cv2) 
 54.2(cs) 35.2(cv1) 50.3(cv2) 
 53.6(cs) 34.3(cv1)
43.9(cv2) 
 53.4(cs) 34.9(cv1) 45.1(cv2) 
 42.5(cs) 13.4(cv1) 17.2(cv2) 
 41.9(cs) 20.9(cv1) 23.7(cv2) 
 | 
 
 
 RGB skeleton 
 skeleton 
 RGB 
 RGB 
 RGB skeleton 
 RGB skeleton 
 RGB 
 RGB 
 skeleton 
 RGB 
 | 
 
 
 Conv3D+attention 
 attention 
 motion+attention 
 Conv3D+motion 
 Conv3D+attention 
 Conv3D+RNN+attention 
 attention 
 Conv3D 
 RNN 
 motion 
 | 

 
 
 
 Kinetics-400 [ 267 ] 
 | 
 
 
 Yan et al. [ 253 ] 
 Zhang et al. [ 254 ] 
 Wei et al. [ 255 ] 
 Yuan et
al. [ 256 ] 
 Liu et al. [ 257 ] 
 Li et al. [ 258 ] 
 Tong et al.
 [ 259 ] 
 Ryoo et al. [ 260 ] 
 Arnab et al. [ 196 ] 
 Duan et al.
 [ 250 ] 
 Fan et al. [ 195 ] 
 Bertasius et al. [ 194 ] 
 Feichtenhofer
et al. [ 245 ] 
 Feichtenhofer et al. [ 246 ] 
 | 
 
 
 89.1 (top-1) 
 87.2 (top-1) 
 87.0 (top-1) 
 86.8 (top-1) 
 86.8 (top-1) 
 86.1
(top-1) 
 85.8 (top-1) 
 85.4 (top-1) 
 84.9 (top-1) 
 83.9 (top-1) 
 81.2 (top-1)
 
 80.7 (top-1) 
 80.4 (top-1) 
 79.8 (top-1) 
 | 
 
 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB skeleton 
 RGB 
 RGB 
 RGB 
 RGB 
 | 
 
 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 Conv3D 
 attention 
 attention 
 Conv3D 
 Conv3D 
 | 

 
 
 
 Kinetics-600 [ 268 ] 
 | 
 
 
 Yan et al. [ 253 ] 
 Wei et al. [ 255 ] 
 Yuan et al. [ 256 ] 
 Li et
al. [ 258 ] 
 Zhang et al. [ 254 ] 
 Ryoo et al. [ 260 ] 
 Liu et al.
 [ 191 ] 
 Arnab et al. [ 196 ] 
 Fan et al. [ 195 ] 
 Bertasius et al. [ 194 ] 
 
 Feichtenhofer et al. [ 245 ] 
 Feichtenhofer et al.
 [ 246 ] 
 | 
 
 
 89.6 (top-1) 
 88.3 (top-1) 
 88.0 (top-1) 
 87.9 (top-1) 
 87.9 (top-1) 
 86.3
(top-1) 
 86.1 (top-1) 
 85.8 (top-1) 
 84.1 (top-1) 
 82.2 (top-1) 
 81.9 (top-1)
 
 81.8 (top-1) 
 | 
 
 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 RGB 
 | 
 
 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 attention 
 Conv3D 
 Conv3D 
 | 

 
 
 
 Kinetics-Skeleton [ 275 ] 
 | 
 
 
 Duan et al. [ 250 ] 
 Obinata et al. [ 248 ] 
 Chen et al. [ 249 ] 
 Liu
et al. [ 129 ] 
 Ye et al. [ 220 ] 
 Shi et al. [ 237 ] 
 Yang et al.
 [ 265 ] 
 Plizzari et al. [ 204 ] 
 | 
 
 
 47.7 (top-1) 
 38.6 (top-1) 
 38.4 (top-1) 
 38.0 (top-1) 
 37.9 (top-1) 
 37.8
(top-1) 
 37.5 (top-1) 
 37.4 (top-1) 
 | 
 
 
 skeleton 
 skeleton 
 skeleton 
 skeleton 
 skeleton 
 RGB skeleton 
 skeleton
 
 skeleton 
 | 
 
 
 Conv3D 
 Conv3D 
 Conv3D 
 Conv3D 
 Conv3D+motion 
 Conv3D+motion 
 Conv3D+motion 
 attention 
 | 

 

 
 
 

### 4.3 Future directions and challenges

 
 We envision transformers will proceed in human action recognition. However, there are main challenges for using transformers in activity recognition that should be addressed by the community.

 
 
 Transformers with high performance and low resource cost for action recognition: Compared with CNN models, transformers are usually huge and computationally expensive, and efficient transformers are needed especially for devices with limited resources. So compressing and accelerating transformer models for efficient implementation; specifically, transformers with high performance and low resource cost is an open problem [ 28 ] . Some works attempt to compress pre-defined transformer models into smaller ones, some others attempt to design compact models directly. The research carried out for efficient implementation includes pruning networks and decomposition [ 278 , 279 ] , knowledge distillation [ 280 ] , network quantization [ 281 ] , and compact architecture design [ 282 ] . However, models originally designed for NLP, may not be suitable for action recognition.

 
 
 Transformer models capable of handling a large number of spatiotemporal tokens and extracting conjoint inter and intra-modal features: In the case of multimodality,
transformers are used for intramodality spatial and temporal modeling and cross-modality feature fusion. Handling a large number of spatiotemporal tokens extracted from multiple modalities is a concern. Some methods develop several scalable model variants which factorize self-attention across the space, time, and modality dimensions. For example in [ 205 ] , a spatial self-attention module is used to understand intra-frame interactions between different body parts, and a temporal self-attention module to model inter-frame correlations. The two are combined in a two-stream network. To further explore the rich inter-modal interactions and their effects, cross-modal attention mechanisms that can be seamlessly integrated into the transformer building block are also needed to effectively exploit the complementary nature of all modalities.

 
 
 Transformer models capable of processing multiple tasks of human action recognition in a single model: In addition, following the success of some new trends in NLP [ 283 ] and CV [ 284 , 285 , 286 ] to develop transformer models capable of processing multiple tasks in a single model, we believe that domains including images, audio, multimodal, etc. can be unified in only one model. Advances in hybrid models combining different approaches with transformers are also expected [ 238 , 239 ] .

 
 
 Practical action recognition: Existing approaches, such as 3D convolutional neural networks and transformer-based methods, usually process the videos in a clip-wise manner; requiring huge GPU memory and fixed-length video clips [ 287 ] . So, proposing models capable of working on variant-length video clips without requiring large GPU memory is needed.
Finally, we think the deep learning solutions for large-scale, real-time, multi-view, and realistic action recognition topics along with newer problems like early recognition, multi-task learning, few-shot learning, unsupervised and semi-supervised learning, and recognition from low-resolution videos will receive attention in the next years.

 
 
 
 

## 5 Conclusions

 
 In this paper, a comprehensive overview of a long concern in human action recognition i.e. temporal modeling is presented. It is especially important in recognizing similar actions with subtle time differences. The taxonomy is defined to cover most of the basic and crucial approaches for modeling temporal information and then some recent methods are reviewed. Key branches introduced include motion-based feature approaches, three-dimensional convolutional neural networks, recurrent neural networks, transformers, and hybrid methods. In brief, more than 170 recent papers on human action recognition are reviewed. In each category, methods are grouped based on used modalities; RGB, depth, skeleton, or combinations of multiple modalities. Finally, different approaches are compared with each other using benchmark datasets for human action recognition. This way, popular approaches for temporal modeling along with popular modalities are recognized. From studied papers, transformers are showing promising results due to properties such as not suffering from long dependency issues and parallel processing. However, transformers are still at the start point of the way, and the question that transformers will prevail in human action recognition or not remains (due to difficulties such as the expensive cost of training transformer-based architectures, the lack of inductive biases of CNNs in transformers, and the need for large-scale training to surpass these biases).

 
 
 

## Acknowledgment

 
 This research is partially supported by the Iran national science foundation (INSF) https://insf.org/en and the Shahid Bahonar University of Kerman https://uk.ac.ir/en/home under grant number 98006291.

 
 
 

## Appendix A From self-attention to transformers; formulation overview

 
 As mentioned before, the transformer that is primarily used in NLP [ 25 ] , is a new deep learning model based on the self-attention mechanism that weights the significance of different parts of the input data. The self-attention is a key idea behind transformers, which facilities capturing ‘long term’ dependencies between sequence elements and can be viewed as a kind of non-local filtering operation [ 288 , 261 ] . Note that encoding such dependencies is a challenge in CNNs and RNNs. The main difference between self-attention with convolution is that the filters are calculated dynamically for any input compared with static filters of the convolution operation.

 
 

### A.1 Building block: self-attention and multi-head attention layer

 
 A self-attention layer projects the input sequence X ∈ R n × d X\in R^{n\times d} 
onto three learnable weight matrices namely, Queries, Keys and Values
denoted as W Q ∈ R d × d q W^{Q}\in R^{d\times d_{q}} ,
 W K ∈ R d × d k W^{K}\in R^{d\times d_{k}} and W V ∈ R d × d v W^{V}\in R^{d\times d_{v}} . Where n is the number of entities in the input
sequence, d is the embedding dimension for representing each entity,
and d q = d k = d v = d model d_{q}=d_{k}=d_{v}=d_{\text{model}} . So the output of the self-attention layer Z ∈ R n × d v Z\in R^{n\times d_{v}} is obtained as:

 
 
 

 
 | 
 Q = X ​ W Q , K = X ​ W K , V = X ​ W V Q=XW^{Q},K=XW^{K},V=XW^{V} | 
 | 
 (1) | 
 

 
 
 

 
 | 
 Z = s ​ o ​ f ​ t ​ m ​ a ​ x ​ ( Q. ​ K T d k ) ​ .V Z=softmax\left(\frac{\text{Q.}K^{T}}{\sqrt{d_{k}}}\right)\text{.V}\ | 
 | 
 (2) | 
 

 
 
 The goal of self-attention is to capture the interaction among all n
entities by computing the scores between each pair of different vectors.
These scores determine the amount of attention given to other entities
when encoding the entity at the current position. Normalizing the scores
enhances gradient stability for improved training, and the softmax
function is used to convert the scores into probabilities. Vectors with
larger probabilities receive additional focus from the following layers.
In this way, each entity is encoded in terms of global contextual
information.

 
 
 In addition, to improve the performance of the simple self-attention
layer, multi-head attention is used to compute multiple dependencies
among different entities of the sequence. The multi-head attention
contains multiple self-attention blocks, with each block having its own
set of weight matrices [ W Q i , W K i , W V i ] [W^{Q_{i}},W^{K_{i}},W^{V_{i}}] , where i = 0 ​ … ​ ( h − 1 ) i=0\ldots(h-1) 
and h h is the number of self-attention blocks. Finally,
the outputs of all blocks are concatenated into a single matrix and
projected onto a weight matrix.

 
 
 It has been shown in the literature that self-attention (with positional
encodings) is theoretically a more flexible operation [ 289 ] . In
 [ 290 ] , the relationship between self-attention and convolution
operations is studied. Their empirical results showed that multi-head
self-attention (with sufficient parameters) is a more generic operation
that can model the expressiveness of convolution as a special case.
Self-attention can learn the global as well as local features, and
provide the capability of learning kernel weights adaptively as well as
the receptive field.

 
 
 

### A.2 Transformer model

 
 The architecture of the transformer model contains an encoder-decoder
structure, as shown in Figure 14 . The left side of this image shows a
simple schematic of the transformer. The encoder module contains
 N N stacked identical blocks, with each block having two
sub-layers of a multi-head self-attention network and a point-wise fully
connected feed-forward network. The decoder in the transformer model
also comprises N N identical blocks. Each decoder block
has three sub-layers of multi-head self-attention, encoder-decoder
attention, and feed-forward. The multi-head self-attention and
feed-forward are similar to the encoder, while the encoder-decoder
attention sublayer performs multi-head attention on the outputs of the
corresponding encoder block.

 
 
 The right side of Figure 14 shows more details of the transformer
proposed in [ 25 ] . Note that After each block in the encoder and
decoder, residual connections [ 189 ] and layer normalization
 [ 291 ] are also applied. Positional encodings are also added to the
input sequence to capture the relative position of each entity in the
sequence. Since there is no recurrence and convolution in the
transformer model, some information must be embedded about the position
of the entities in the sequence for the model to use the order of the
sequence. Positional encodings have the same dimensions as the input d
and can be learned or pre-defined, e.g., by sine or cosine functions. In
addition, the decoder of the transformer uses previous outputs to
predict the following entity in the sequence. So the decoder takes
inputs from the encoder and the preceding outputs to calculate the next
entity of the sequence.

 
 
 Figure 14: The transformer architecture: a) simple schematic and b) more
details of transformer model [ 25 ] . 
 
 
 
 

## References

 
 
 [1] 
 
M. Vrigkas, C. Nikou, I. A. Kakadiaris, A review of human activity recognition
methods, Frontiers in Robotics and AI 2 (2015) 28.

 

 
 [2] 
 
J. Aggarwal, M. S. Ryoo, Human activity analysis: A review, ACM Computing
Surveys (CSUR) 43 (3) (2011) 16.

 

 
 [3] 
 
G. Guo, A. Lai, A survey on still image based human action recognition, Pattern
Recognition 47 (10) (2014) 3343–3361.

 

 
 [4] 
 
Y. Hu, M. Lu, X. Lu, Driving behaviour recognition from still images by using
multi-stream fusion cnn, Machine Vision and Applications 30 (5) (2019)
851–865.

 

 
 [5] 
 
X. Yu, Z. Zhang, L. Wu, W. Pang, H. Chen, Z. Yu, B. Li, Deep ensemble learning
for human action recognition in still images, Complexity 2020 (2020).

 

 
 [6] 
 
M. Asadi-Aghbolaghi, A. Clapes, M. Bellantonio, H. J. Escalante,
V. Ponce-López, X. Baró, I. Guyon, S. Kasaei, S. Escalera, A survey
on deep learning based approaches for action and gesture recognition in image
sequences, in: 2017 12th IEEE international conference on automatic face 
gesture recognition (FG 2017), IEEE, 2017, pp. 476–483.

 

 
 [7] 
 
L. M. Dang, K. Min, H. Wang, M. J. Piran, C. H. Lee, H. Moon, Sensor-based and
vision-based human activity recognition: A comprehensive survey, Pattern
Recognition 108 (2020) 107561.

 

 
 [8] 
 
Y. Kong, Y. Fu, Human action recognition and prediction: A survey, arXiv
preprint arXiv:1806.11230 (2018).

 

 
 [9] 
 
B. Liu, H. Cai, Z. Ju, H. Liu, Rgb-d sensing based human action and interaction
analysis: A survey, Pattern Recognition 94 (2019) 1–12.

 

 
 [10] 
 
L. L. Presti, M. La Cascia, 3d skeleton-based human action classification: A
survey, Pattern Recognition 53 (2016) 130–147.

 

 
 [11] 
 
B. Ren, M. Liu, R. Ding, H. Liu, A survey on 3d
skeleton-based action recognition using learning method , arXiv preprint
arXiv:2002.05907 (2020).

 URL files/31/2002.html 

 

 
 [12] 
 
P. Wang, W. Li, P. Ogunbona, J. Wan, S. Escalera, Rgb-d-based human motion
recognition with deep learning: A survey, Computer Vision and Image
Understanding 171 (2018) 118–139.

 

 
 [13] 
 
O. Yurur, C. H. Liu, W. Moreno, A survey of context-aware middleware designs
for human activity recognition, Communications Magazine, IEEE 52 (6) (2014)
24–31.

 

 
 [14] 
 
H.-B. Zhang, Y.-X. Zhang, B. Zhong, Q. Lei, L. Yang, J.-X. Du, D.-S. Chen, A
comprehensive survey of vision-based human action recognition methods,
Sensors 19 (5) (2019) 1005.

 

 
 [15] 
 
J. Zhang, W. Li, P. O. Ogunbona, P. Wang, C. Tang, Rgb-d-based action
recognition datasets: A survey, Pattern Recognition 60 (2016) 86–105.

 

 
 [16] 
 
F. Zhu, L. Shao, J. Xie, Y. Fang, From handcrafted to learned representations
for human action recognition: A survey, Image and Vision Computing (2016).

 

 
 [17] 
 
D. Tran, L. Bourdev, R. Fergus, L. Torresani, M. Paluri, Learning
spatiotemporal features with 3d convolutional networks, in: Proceedings of
the IEEE international conference on computer vision, 2015, pp. 4489–4497.

 

 
 [18] 
 
S. Ji, W. Xu, M. Yang, K. Yu, 3d convolutional neural networks for human action
recognition, IEEE transactions on pattern analysis and machine intelligence
35 (1) (2013) 221–231.

 

 
 [19] 
 
K. Simonyan, A. Zisserman, Two-stream convolutional networks for action
recognition in videos, Advances in neural information processing systems 27
(2014).

 

 
 [20] 
 
J. Donahue, L. Anne Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan,
K. Saenko, T. Darrell, Long-term recurrent convolutional networks for visual
recognition and description, in: Proceedings of the IEEE conference on
computer vision and pattern recognition, 2015, pp. 2625–2634.

 

 
 [21] 
 
S. Yuanyuan, L. Yunan, F. Xiaolong, M. Kaibin, M. Qiguang, Review of dynamic
gesture recognition, Virtual Reality Intelligent Hardware 3 (3) (2021)
183–206.

 

 
 [22] 
 
N. S. Khan, M. S. Ghani, A survey of deep learning based models for human
activity recognition, Wireless Personal Communications 120 (2) (2021)
1593–1635.

 

 
 [23] 
 
K. Rangasamy, M. A. As’ari, N. A. Rahmad, N. F. Ghazali, S. Ismail, Deep
learning in sport video analysis: a review, TELKOMNIKA (Telecommunication
Computing Electronics and Control) 18 (4) (2020) 1926–1933.

 

 
 [24] 
 
Y. Zhu, X. Li, C. Liu, M. Zolfaghari, Y. Xiong, C. Wu, Z. Zhang, J. Tighe,
R. Manmatha, M. Li, A comprehensive study of deep video action recognition,
arXiv preprint arXiv:2012.06567 (2020).

 

 
 [25] 
 
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, I. Polosukhin, Attention is all you need, Vol. 30, 2017.

 

 
 [26] 
 
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, An image is
worth 16x16 words: Transformers for image recognition at scale, arXiv
preprint arXiv:2010.11929 (2020).

 

 
 [27] 
 
S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, M. Shah, Transformers in
vision: A survey, arXiv preprint arXiv:2101.01169 (2021).

 

 
 [28] 
 
K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu,
Y. Xu, A survey on vision transformer, IEEE Transactions on Pattern Analysis
and Machine Intelligence (2022).

 

 
 [29] 
 
D. Neimark, O. Bar, M. Zohar, D. Asselmann, Video transformer network, arXiv
preprint arXiv:2102.00719 (2021).

 

 
 [30] 
 
A. Singh, O. Chakraborty, A. Varshney, R. Panda, R. Feris, K. Saenko, A. Das,
Semi-supervised action recognition with temporal contrastive learning, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2021, pp. 10389–10399.

 

 
 [31] 
 
X. Song, S. Zhao, J. Yang, H. Yue, P. Xu, R. Hu, H. Chai, Spatio-temporal
contrastive domain adaptation for action recognition, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp.
9787–9795.

 

 
 [32] 
 
A. Prati, C. Shan, K. I.-K. Wang, Sensors, vision and networks: From video
surveillance to activity recognition and health monitoring, Journal of
Ambient Intelligence and Smart Environments 11 (1) (2019) 5–22.

 

 
 [33] 
 
M. A. R. Ahad, A. D. Antar, O. Shahid, Vision-based action understanding for
assistive healthcare: A short review., in: CVPR Workshops, 2019, pp. 1–11.

 

 
 [34] 
 
T. Singh, D. K. Vishwakarma, Video benchmarks of human action datasets: a
review, Artificial Intelligence Review 52 (2) (2019) 1107–1154.

 

 
 [35] 
 
T. Singh, D. K. Vishwakarma, Human activity recognition in video benchmarks: A
survey, Advances in Signal Processing and Communication (2019) 247–259.

 

 
 [36] 
 
Z. Cai, J. Han, L. Liu, L. Shao, Rgb-d datasets using microsoft kinect or
similar sensors: a survey, Multimedia Tools and Applications 76 (3) (2017)
4313–4355.

 

 
 [37] 
 
K. Chen, D. Zhang, L. Yao, B. Guo, Z. Yu, Y. Liu, Deep learning for
sensor-based human activity recognition: Overview, challenges, and
opportunities, ACM Computing Surveys (CSUR) 54 (4) (2021) 1–40.

 

 
 [38] 
 
B. Nguyen, Y. Coelho, T. Bastos, S. Krishnan, Trends in human activity
recognition with focus on machine learning and power requirements, Machine
Learning with Applications 5 (2021) 100072.

 

 
 [39] 
 
Z. Hussain, Q. Z. Sheng, W. E. Zhang, A review and categorization of techniques
on device-free human activity recognition, Journal of Network and Computer
Applications 167 (2020) 102738.

 

 
 [40] 
 
D. R. Beddiar, B. Nini, M. Sabokrou, A. Hadid, Vision-based human activity
recognition: a survey, Multimedia Tools and Applications 79 (41) (2020)
30509–30555.

 

 
 [41] 
 
M. Al-Faris, J. Chiverton, D. Ndzi, A. I. Ahmed, A review on computer
vision-based methods for human action recognition, Journal of imaging 6 (6)
(2020) 46.

 

 
 [42] 
 
L. Wang, D. Q. Huynh, P. Koniusz, A comparative review of recent kinect-based
action recognition algorithms, arXiv preprint arXiv:1906.09955 (2019).

 

 
 [43] 
 
S. Majumder, N. Kehtarnavaz, A review of real-time human action recognition
involving vision sensing, Real-Time Image Processing and Deep Learning 2021
11736 (2021) 53–64.

 

 
 [44] 
 
S. K. Yadav, K. Tiwari, H. M. Pandey, S. A. Akbar, A review of multimodal human
activity recognition with special emphasis on classification, applications,
challenges and future directions, Knowledge-Based Systems 223 (2021) 106970.

 

 
 [45] 
 
Z. Sun, Q. Ke, H. Rahmani, M. Bennamoun, G. Wang, J. Liu, Human action
recognition from various data modalities: A review, IEEE transactions on
pattern analysis and machine intelligence (2022).

 

 
 [46] 
 
S. Majumder, N. Kehtarnavaz, Vision and inertial sensing fusion for human
action recognition: A review, IEEE Sensors Journal 21 (3) (2020) 2454–2467.

 

 
 [47] 
 
Q. Li, R. Gravina, Y. Li, S. H. Alsamhi, F. Sun, G. Fortino, Multi-user
activity recognition: Challenges and opportunities, Information Fusion 63
(2020) 121–135.

 

 
 [48] 
 
A. Roitberg, T. Pollert, M. Haurilet, M. Martin, R. Stiefelhagen, Analysis of
deep fusion strategies for multi-modal gesture recognition, in: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
Workshops, 2019, pp. 0–0.

 

 
 [49] 
 
Y. Xing, J. Zhu, Deep learning-based action recognition with 3d skeleton: a
survey (2021).

 

 
 [50] 
 
P. Pareek, A. Thakkar, A survey on video-based human action recognition: recent
updates, datasets, challenges, and applications, Artificial Intelligence
Review 54 (3) (2021) 2259–2322.

 

 
 [51] 
 
I. Jegham, A. B. Khalifa, I. Alouani, M. A. Mahjoub, Vision-based human action
recognition: An overview and real world challenges, Forensic Science
International: Digital Investigation 32 (2020) 200901.

 

 
 [52] 
 
C. Dhiman, D. K. Vishwakarma, A review of state-of-the-art techniques for
abnormal human activity recognition, Engineering Applications of Artificial
Intelligence 77 (2019) 21–45.

 

 
 [53] 
 
V. Estevam, H. Pedrini, D. Menotti, Zero-shot action recognition in videos: A
survey, Neurocomputing 439 (2021) 159–175.

 

 
 [54] 
 
G. Yao, T. Lei, J. Zhong, A review of convolutional-neural-network-based action
recognition, Pattern Recognition Letters 118 (2019) 14–22.

 

 
 [55] 
 
G. Sreenu, S. Durai, Intelligent video surveillance: a review through deep
learning techniques for crowd analysis, Journal of Big Data 6 (1) (2019)
1–27.

 

 
 [56] 
 
A. Ulhaq, N. Akhtar, G. Pogrebna, A. Mian, Vision transformers for action
recognition: A survey, arXiv preprint arXiv:2209.05700 (2022).

 

 
 [57] 
 
E. Shabaninia, A. R. Naghsh-Nilchi, S. Kasaei, Extended histogram:
probabilistic modelling of video content temporal evolutions,
Multidimensional Systems and Signal Processing (2018) 1–19.

 

 
 [58] 
 
D. D. Dawn, S. H. Shaikh, A comprehensive survey of human action recognition
with spatio-temporal interest point (stip) detector, The Visual Computer
32 (3) (2016) 289–306.

 

 
 [59] 
 
P. Scovanner, S. Ali, M. Shah, A 3-dimensional sift descriptor and its
application to action recognition, in: Proceedings of the 15th ACM
international conference on Multimedia, 2007, pp. 357–360.

 

 
 [60] 
 
G. Willems, T. Tuytelaars, L. Van Gool, An efficient dense and scale-invariant
spatio-temporal interest point detector, Springer, 2008, pp. 650–663.

 

 
 [61] 
 
A. Klaser, M. Marszałek, C. Schmid, A spatio-temporal descriptor based on
3d-gradients, in: BMVC 2008-19th British Machine Vision Conference, British
Machine Vision Association, 2008, pp. 275–1.

 

 
 [62] 
 
I. Laptev, M. Marszalek, C. Schmid, B. Rozenfeld, Learning realistic human
actions from movies, in: 2008 IEEE Conference on Computer Vision and Pattern
Recognition, IEEE, 2008, pp. 1–8.

 

 
 [63] 
 
L. Wang, Y. Qiao, X. Tang, Video action detection with relational
dynamic-poselets, in: European conference on computer vision, Springer, 2014,
pp. 565–580.

 

 
 [64] 
 
L. Wang, Y. Qiao, X. Tang, Motionlets: Mid-level 3d parts for human motion
recognition, in: Proceedings of the ieee conference on computer vision and
pattern recognition, 2013, pp. 2674–2681.

 

 
 [65] 
 
S. Sadanand, J. J. Corso, Action bank: A high-level representation of activity
in video, in: 2012 IEEE Conference on computer vision and pattern
recognition, IEEE, 2012, pp. 1234–1241.

 

 
 [66] 
 
L. Wang, Y. Qiao, X. Tang, Mining motion atoms and phrases for complex action
recognition, in: Proceedings of the IEEE international conference on computer
vision, 2013, pp. 2680–2687.

 

 
 [67] 
 
J. Zhu, B. Wang, X. Yang, W. Zhang, Z. Tu, Action recognition with actons, in:
Proceedings of the IEEE International Conference on Computer Vision, 2013,
pp. 3559–3566.

 

 
 [68] 
 
L. Xia, C.-C. Chen, J. K. Aggarwal, View invariant human action recognition
using histograms of 3d joints, in: 2012 IEEE computer society conference on
computer vision and pattern recognition workshops, IEEE, 2012, pp. 20–27.

 

 
 [69] 
 
S. Gaglio, G. L. Re, M. Morana, Human activity recognition process using 3-d
posture data, IEEE Transactions on Human-Machine Systems 45 (5) (2015)
586–597.

 

 
 [70] 
 
H. Koppula, A. Saxena, Learning spatio-temporal structure from rgb-d videos for
human activity detection and anticipation, in: International conference on
machine learning, PMLR, 2013, pp. 792–800.

 

 
 [71] 
 
H. S. Koppula, R. Gupta, A. Saxena, Learning human activities and object
affordances from rgb-d videos, The International Journal of Robotics Research
32 (8) (2013) 951–970.

 

 
 [72] 
 
B. B. Amor, J. Su, A. Srivastava, Action recognition using rate-invariant
analysis of skeletal shape trajectories, Pattern Analysis and Machine
Intelligence, IEEE Transactions on 38 (1) (2016) 1–13.

 

 
 [73] 
 
E. Shabaninia, A. R. Naghsh-Nilchi, S. Kasaei, A weighting scheme for mining
key skeletal joints for human action recognition, Multimedia Tools and
Applications 78 (22) (2019) 31319–31345.

 

 
 [74] 
 
D. Ramachandram, G. W. Taylor, Deep multimodal learning: A survey on recent
advances and trends, IEEE signal processing magazine 34 (6) (2017) 96–108.

 

 
 [75] 
 
A. Jain, K. Nandakumar, A. Ross, Score normalization in multimodal biometric
systems, Pattern recognition 38 (12) (2005) 2270–2285.

 

 
 [76] 
 
Q. Ke, M. Bennamoun, S. An, F. Boussaid, F. Sohel, Human interaction prediction
using deep temporal features, in: European Conference on Computer Vision,
Springer, 2016, pp. 403–414.

 

 
 [77] 
 
L. Wang, Y. Qiao, X. Tang, Action recognition with trajectory-pooled
deep-convolutional descriptors, in: Proceedings of the IEEE conference on
computer vision and pattern recognition, 2015, pp. 4305–4314.

 

 
 [78] 
 
A. Diba, V. Sharma, L. Van Gool, Deep temporal linear encoding networks, in:
Proceedings of the IEEE conference on Computer Vision and Pattern
Recognition, 2017, pp. 2329–2338.

 

 
 [79] 
 
R. Girdhar, D. Ramanan, A. Gupta, J. Sivic, B. Russell, Actionvlad: Learning
spatio-temporal aggregation for action classification, in: Proceedings of the
IEEE conference on computer vision and pattern recognition, 2017, pp.
971–980.

 

 
 [80] 
 
C. Feichtenhofer, A. Pinz, R. P. Wildes, Spatiotemporal multiplier networks for
video action recognition, in: Proceedings of the IEEE conference on computer
vision and pattern recognition, 2017, pp. 4768–4777.

 

 
 [81] 
 
Y. Xiao, J. Chen, Y. Wang, Z. Cao, J. T. Zhou, X. Bai, Action recognition for
depth video using multi-view dynamic images, Information Sciences 480 (2019)
287–304.

 

 
 [82] 
 
P. Wang, W. Li, Z. Gao, C. Tang, P. O. Ogunbona, Depth pooling based
large-scale 3-d action recognition with convolutional neural networks, IEEE
Transactions on Multimedia 20 (5) (2018) 1051–1061.

 

 
 [83] 
 
P. Wang, W. Li, Z. Gao, J. Zhang, C. Tang, P. O. Ogunbona, Action recognition
from depth maps using deep convolutional neural networks, IEEE Transactions
on Human-Machine Systems 46 (4) (2015) 498–509.

 

 
 [84] 
 
Z. Li, Z. Zheng, F. Lin, H. Leung, Q. Li, Action recognition from depth
sequence using depth motion maps-based local ternary patterns and cnn,
Multimedia Tools and Applications 78 (14) (2019) 19587–19601.

 

 
 [85] 
 
P. Wang, W. Li, C. Li, Y. Hou, Action recognition based on joint trajectory
maps with convolutional neural networks, Knowledge-Based Systems 158 (2018)
43–53.

 

 
 [86] 
 
P. Wang, Z. Li, Y. Hou, W. Li, Action recognition based on joint trajectory
maps using convolutional neural networks, in: Proceedings of the 24th ACM
international conference on Multimedia, 2016, pp. 102–106.

 

 
 [87] 
 
C. Caetano, J. Sena, F. Brémond, J. A. Dos Santos, W. R. Schwartz,
Skelemotion: A new representation of skeleton joint sequences based on motion
information for 3d action recognition, in: 2019 16th IEEE international
conference on advanced video and signal based surveillance (AVSS), IEEE,
2019, pp. 1–8.

 

 
 [88] 
 
C. Caetano, F. Brémond, W. R. Schwartz, Skeleton image representation for
3d action recognition based on tree structure and reference joints, in: 2019
32nd SIBGRAPI conference on graphics, patterns and images (SIBGRAPI), IEEE,
2019, pp. 16–23.

 

 
 [89] 
 
Y. Hou, Z. Li, P. Wang, W. Li, Skeleton optical spectra-based action
recognition using convolutional neural networks, IEEE Transactions on
Circuits and Systems for Video Technology 28 (3) (2016) 807–811.

 

 
 [90] 
 
C. Li, Y. Hou, P. Wang, W. Li, Joint distance maps based action recognition
with convolutional neural networks, IEEE Signal Processing Letters 24 (5)
(2017) 624–628.

 

 
 [91] 
 
H.-H. Pham, L. Khoudour, A. Crouzil, P. Zegers, S. A. Velastin, Exploiting deep
residual networks for human action recognition from skeletal data, Computer
Vision and Image Understanding 170 (2018) 51–66.

 

 
 [92] 
 
M. Asadi-Aghbolaghi, H. Bertiche, V. Roig, S. Kasaei, S. Escalera, Action
recognition from rgb-d data: Comparison and fusion of spatio-temporal
handcrafted features and deep strategies, in: Proceedings of the IEEE
International Conference on Computer Vision Workshops, 2017, pp. 3179–3188.

 

 
 [93] 
 
H. Bilen, B. Fernando, E. Gavves, A. Vedaldi, S. Gould, Dynamic image networks
for action recognition, in: Proceedings of the IEEE conference on computer
vision and pattern recognition, 2016, pp. 3034–3042.

 

 
 [94] 
 
K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale
image recognition, arXiv preprint arXiv:1409.1556 (2014).

 

 
 [95] 
 
S. Xie, R. Girshick, P. Dollár, Z. Tu, K. He, Aggregated residual
transformations for deep neural networks, in: Proceedings of the IEEE
conference on computer vision and pattern recognition, 2017, pp. 1492–1500.

 

 
 [96] 
 
S. Mukherjee, L. Anvitha, T. M. Lahari, Human activity recognition in rgb-d
videos by dynamic images, Multimedia Tools and Applications 79 (27) (2020)
19787–19801.

 

 
 [97] 
 
R. Singh, R. Khurana, A. K. S. Kushwaha, R. Srivastava, Combining cnn streams
of dynamic image and depth data for action recognition, Multimedia Systems
(2020) 1–10.

 

 
 [98] 
 
A. S. Rajput, B. Raman, J. Imran, Privacy-preserving human action recognition
as a remote cloud service using rgb-d sensors and deep cnn, Expert Systems
with Applications 152 (2020) 113349.

 

 
 [99] 
 
J. Imran, P. Kumar, Human action recognition using rgb-d sensor and deep
convolutional neural networks, in: 2016 international conference on advances
in computing, communications and informatics (ICACCI), IEEE, 2016, pp.
144–148.

 

 
 [100] 
 
A. P. Twinanda, P. Winata, A. Gangi, M. Mathelin, N. Padoy, Multi-stream deep
architecture for surgical phase recognition on multi-view rgbd videos, in:
Proc. M2CAI Workshop MICCAI, 2016, pp. 1–8.

 

 
 [101] 
 
A. Krizhevsky, I. Sutskever, G. E. Hinton, Imagenet classification with deep
convolutional neural networks, Vol. 60, AcM New York, NY, USA, 2017, pp.
84–90.

 

 
 [102] 
 
P. Wang, W. Li, Z. Gao, Y. Zhang, C. Tang, P. Ogunbona, Scene flow to action
map: A new representation for rgb-d based action recognition with
convolutional neural networks, in: Proceedings of the IEEE conference on
computer vision and pattern recognition, 2017, pp. 595–604.

 

 
 [103] 
 
A. Shahroudy, T.-T. Ng, Y. Gong, G. Wang, Deep
multimodal feature analysis for action recognition in rgb+ d videos , IEEE
transactions on pattern analysis and machine intelligence 40 (5) (2017)
1045–1058.

 URL files/54/7892950.html 

 

 
 [104] 
 
X. Qin, Y. Ge, L. Zhan, G. Li, S. Huang, H. Wang, F. Chen, Joint deep learning
for rgb-d action recognition, in: 2018 IEEE Visual Communications and Image
Processing (VCIP), IEEE, 2018, pp. 1–6.

 

 
 [105] 
 
Y. Tang, Z. Wang, J. Lu, J. Feng, J. Zhou,
 Multi-stream deep neural networks for rgb-d
egocentric action recognition , IEEE Transactions on Circuits and Systems for
Video Technology 29 (10) (2018) 3001–3015.

 URL files/111/8489917.html 

 

 
 [106] 
 
J. Cheng, Z. Ren, Q. Zhang, X. Gao, F. Hao, Cross-modality compensation
convolutional neural networks for rgb-d action recognition, IEEE Transactions
on Circuits and Systems for Video Technology (2021).

 

 
 [107] 
 
P. Wang, W. Li, J. Wan, P. Ogunbona, X. Liu, Cooperative training of deep
aggregation networks for rgb-d action recognition, in: Proceedings of the
AAAI Conference on Artificial Intelligence, Vol. 32, 2018.

 

 
 [108] 
 
Z. Ren, Q. Zhang, X. Gao, P. Hao, J. Cheng,
 Multi-modality learning for
human action recognition , Multimedia Tools and Applications (2020).

 
 doi:10.1007/s11042-019-08576-z .

 URL https://doi.org/10.1007/s11042-019-08576-z 

 

 
 [109] 
 
Z. Ren, Q. Zhang, J. Cheng, F. Hao, X. Gao,
 Segment spatial-temporal representation and
cooperative learning of convolution neural networks for multimodal-based
action recognition , Neurocomputing 433 (2021) 142–153.

 
 doi:10.1016/j.neucom.2020.12.020 .

 URL https://www.sciencedirect.com/science/article/pii/S0925231220319019files/11/S0925231220319019.html 

 

 
 [110] 
 
N. C. Garcia, P. Morerio, V. Murino, Modality distillation with multiple stream
networks for action recognition, in: Proceedings of the European Conference
on Computer Vision (ECCV), 2018, pp. 103–118.

 

 
 [111] 
 
P. Verma, A. Sah, R. Srivastava, Deep learning-based multi-modal approach using
rgb and skeleton sequences for human activity recognition, Multimedia Systems
26 (6) (2020) 671–685.

 

 
 [112] 
 
A. Tomas, K. Biswas, Human activity recognition using combined deep
architectures, in: 2017 IEEE 2nd International Conference on Signal and Image
Processing (ICSIP), IEEE, 2017, pp. 41–45.

 

 
 [113] 
 
A. Kamel, B. Sheng, P. Yang, P. Li, R. Shen, D. D. Feng, Deep convolutional
neural networks for human action recognition using depth maps and postures,
IEEE Transactions on Systems, Man, and Cybernetics: Systems 49 (9) (2018)
1806–1819.

 

 
 [114] 
 
P. Wang, S. Wang, Z. Gao, Y. Hou, W. Li, Structured images for rgb-d action
recognition, in: Proceedings of the IEEE international conference on computer
vision workshops, 2017, pp. 1005–1014.

 

 
 [115] 
 
J. He, H. Xia, C. Feng, Y. Chu, Cnn-based action recognition using adaptive
multiscale depth motion maps and stable joint distance maps, in: 2018 IEEE
Global Conference on Signal and Information Processing (GlobalSIP), IEEE,
2018, pp. 439–443.

 

 
 [116] 
 
T. Singh, D. K. Vishwakarma, A deep multimodal network based on bottleneck
layer features fusion for action recognition, Multimedia Tools and
Applications (2021) 1–21.

 

 
 [117] 
 
P. Khaire, J. Imran, P. Kumar, Human activity recognition by fusion of rgb,
depth, and skeletal data, in: Proceedings of 2nd International Conference on
Computer Vision Image Processing, Springer, 2018, pp. 409–421.

 

 
 [118] 
 
N. E. D. Elmadany, Y. He, L. Guan, Information
fusion for human action recognition via biset/multiset globality locality
preserving canonical correlation analysis , IEEE Transactions on Image
Processing 27 (11) (2018) 5275–5287.

 URL files/71/8410604.html 

 

 
 [119] 
 
E. E. Cardenas, G. C. Chavez, Multimodal human action recognition based on a
fusion of dynamic images using cnn descriptors, in: 2018 31st SIBGRAPI
Conference on Graphics, Patterns and Images (SIBGRAPI), IEEE, 2018, pp.
95–102.

 

 
 [120] 
 
D. Wu, L. Pigou, P.-J. Kindermans, N. D.-H. Le, L. Shao, J. Dambre, J.-M.
Odobez, Deep dynamic neural networks for
multimodal gesture segmentation and recognition , IEEE transactions on
pattern analysis and machine intelligence 38 (8) (2016) 1583–1597.

 URL files/293/7423804.html 

 

 
 [121] 
 
B. D. Romaissa, O. Mourad, N. Brahim, Vision-based multi-modal framework for
action recognition, in: 2020 25th International Conference on Pattern
Recognition (ICPR), IEEE, 2021, pp. 5859–5866.

 

 
 [122] 
 
D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, M. Paluri, A closer look at
spatiotemporal convolutions for action recognition, in: Proceedings of the
IEEE conference on Computer Vision and Pattern Recognition, 2018, pp.
6450–6459.

 

 
 [123] 
 
A. Stergiou, R. Poppe, Spatio-temporal fast 3d convolutions for human action
recognition, arXiv preprint arXiv:1909.13474 (2019).

 

 
 [124] 
 
J. Carreira, A. Zisserman, Quo vadis, action recognition? a new model and the
kinetics dataset, in: proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 2017, pp. 6299–6308.

 

 
 [125] 
 
Z. Qiu, T. Yao, T. Mei, Learning spatio-temporal representation with pseudo-3d
residual networks, in: proceedings of the IEEE International Conference on
Computer Vision, 2017, pp. 5533–5541.

 

 
 [126] 
 
G. Varol, I. Laptev, C. Schmid, Long-term temporal convolutions for action
recognition, IEEE transactions on pattern analysis and machine intelligence
40 (6) (2017) 1510–1517.

 

 
 [127] 
 
A. Sanchez-Caballero, S. de López-Diz, D. Fuentes-Jimenez,
C. Losada-Gutiérrez, M. Marrón-Romera, D. Casillas-Perez, M. I. Sarker,
3dfcnn: Real-time action recognition using 3d deep neural networks with raw
depth information, arXiv preprint arXiv:2006.07743 (2020).

 

 
 [128] 
 
Y. Chen, Z. Zhang, C. Yuan, B. Li, Y. Deng, W. Hu, Channel-wise topology
refinement graph convolution for skeleton-based action recognition, in:
Proceedings of the IEEE/CVF International Conference on Computer Vision,
2021, pp. 13359–13368.

 

 
 [129] 
 
Z. Liu, H. Zhang, Z. Chen, Z. Wang, W. Ouyang, Disentangling and unifying graph
convolutions for skeleton-based action recognition, in: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2020, pp.
143–152.

 

 
 [130] 
 
A. M. De Boissiere, R. Noumeir, Infrared and 3d skeleton feature fusion for
rgb-d action recognition, IEEE Access 8 (2020) 168297–168308.

 

 
 [131] 
 
Y. Li, Q. Miao, K. Tian, Y. Fan, X. Xu, R. Li, J. Song, Large-scale gesture
recognition with a fusion of rgb-d data based on the c3d model, in: 2016 23rd
international conference on pattern recognition (ICPR), IEEE, 2016, pp.
25–30.

 

 
 [132] 
 
G. Zhu, L. Zhang, L. Mei, J. Shao, J. Song, P. Shen, Large-scale isolated
gesture recognition using pyramidal 3d convolutional networks, in: 2016 23rd
International Conference on Pattern Recognition (ICPR), IEEE, 2016, pp.
19–24.

 

 
 [133] 
 
H. Zhang, Y. Li, P. Wang, Y. Liu, C. Shen, Rgb-d based action recognition with
light-weight 3d convolutional networks, arXiv preprint arXiv:1811.09908
(2018).

 

 
 [134] 
 
X. Qin, Y. Ge, J. Feng, Y. Chen, L. Zhan, X. Wang, Y. Wang, Two-stream network
with 3d common-specific framework for rgb-d action recognition, in: 2019 IEEE
SmartWorld, Ubiquitous Intelligence Computing, Advanced Trusted
Computing, Scalable Computing Communications, Cloud Big Data Computing,
Internet of People and Smart City Innovation
(SmartWorld/SCALCOM/UIC/ATC/CBDCom/IOP/SCI), IEEE, 2019, pp. 731–738.

 

 
 [135] 
 
B. Zhou, J. Wan, Y. Liang, G. Guo, Adaptive cross-fusion learning for
multi-modal gesture recognition, Virtual Reality Intelligent Hardware
3 (3) (2021) 235–247.

 

 
 [136] 
 
B. Zhou, Y. Li, J. Wan, Regional attention with architecture-rebuilt 3d network
for rgb-d gesture recognition, arXiv preprint arXiv:2102.05348 (2021).

 

 
 [137] 
 
Z. Liu, C. Zhang, Y. Tian, 3d-based deep convolutional neural network for
action recognition with depth sequences, Image and Vision Computing 55 (2016)
93–100.

 

 
 [138] 
 
W. Wang, D. Tran, M. Feiszli, What makes training multi-modal classification
networks hard?, in: Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2020, pp. 12695–12705.

 

 
 [139] 
 
J. L. Elman, Finding structure in time, Cognitive science 14 (2) (1990)
179–211.

 

 
 [140] 
 
K. Cho, B. Van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk,
Y. Bengio, Learning phrase representations using rnn encoder-decoder for
statistical machine translation, arXiv preprint arXiv:1406.1078 (2014).

 

 
 [141] 
 
F. A. Gers, N. N. Schraudolph, J. Schmidhuber, Learning precise timing with
lstm recurrent networks, Journal of machine learning research 3 (Aug) (2002)
115–143.

 

 
 [142] 
 
J. Yue-Hei Ng, M. Hausknecht, S. Vijayanarasimhan, O. Vinyals, R. Monga,
G. Toderici, Beyond short snippets: Deep networks for video classification,
in: Proceedings of the IEEE conference on computer vision and pattern
recognition, 2015, pp. 4694–4702.

 

 
 [143] 
 
A. Ullah, K. Muhammad, T. Hussain, S. W. Baik, Conflux lstms network: A novel
approach for multi-view action recognition, Neurocomputing 435 (2021)
321–329.

 

 
 [144] 
 
A. Sanchez-Caballero, D. Fuentes-Jimenez, C. Losada-Gutiérrez, Exploiting the
convlstm: Human action recognition using raw depth video-based recurrent
neural networks, arXiv preprint arXiv:2006.07744 (2020).

 

 
 [145] 
 
F. D. Casagrande, O. O. Nedrejord, W. Lee, E. Zouganeli, Action recognition in
real homes using low resolution depth video data, in: 2019 IEEE 32nd
International Symposium on Computer-Based Medical Systems (CBMS), IEEE, 2019,
pp. 156–161.

 

 
 [146] 
 
X. Liu, Y. Li, Q. Wang, Multi-view hierarchical bidirectional recurrent neural
network for depth video sequence based action recognition, International
Journal of Pattern Recognition and Artificial Intelligence 32 (10) (2018)
1850033.

 

 
 [147] 
 
Y. Du, W. Wang, L. Wang, Hierarchical recurrent neural network for skeleton
based action recognition, in: Proceedings of the IEEE conference on computer
vision and pattern recognition, 2015, pp. 1110–1118.

 

 
 [148] 
 
S. Zhang, Y. Yang, J. Xiao, X. Liu, Y. Yang, D. Xie, Y. Zhuang, Fusing
geometric features for skeleton-based action recognition using multilayer
lstm networks, IEEE Transactions on Multimedia 20 (9) (2018) 2330–2343.

 

 
 [149] 
 
S. Zhang, X. Liu, J. Xiao, On geometric features for skeleton-based action
recognition using multilayer lstm networks, in: 2017 IEEE Winter Conference
on Applications of Computer Vision (WACV), IEEE, 2017, pp. 148–157.

 

 
 [150] 
 
V. Veeriah, N. Zhuang, G.-J. Qi, Differential recurrent neural networks for
action recognition, in: Proceedings of the IEEE international conference on
computer vision, 2015, pp. 4041–4049.

 

 
 [151] 
 
W. Zhu, C. Lan, J. Xing, W. Zeng, Y. Li, L. Shen, X. Xie, Co-occurrence feature
learning for skeleton based action recognition using regularized deep lstm
networks, in: Proceedings of the AAAI conference on artificial intelligence,
Vol. 30, 2016.

 

 
 [152] 
 
A. Shahroudy, J. Liu, T.-T. Ng, G. Wang, Ntu rgb+ d: A large scale dataset for
3d human activity analysis, in: Proceedings of the IEEE conference on
computer vision and pattern recognition, 2016, pp. 1010–1019.

 

 
 [153] 
 
J. Liu, A. Shahroudy, D. Xu, G. Wang, Spatio-temporal lstm with trust gates for
3d human action recognition, in: European conference on computer vision,
Springer, 2016, pp. 816–833.

 

 
 [154] 
 
H. Wang, L. Wang, Modeling temporal dynamics and spatial configurations of
actions using two-stream recurrent neural networks, in: Proceedings of the
IEEE conference on computer vision and pattern recognition, 2017, pp.
499–508.

 

 
 [155] 
 
I. Lee, D. Kim, S. Kang, S. Lee, Ensemble deep learning for skeleton-based
action recognition using temporal sliding lstm networks, in: Proceedings of
the IEEE international conference on computer vision, 2017, pp. 1012–1020.

 

 
 [156] 
 
S. Li, W. Li, C. Cook, Y. Gao, Deep independently recurrent neural network
(indrnn), arXiv preprint arXiv:1910.06251 (2019).

 

 
 [157] 
 
W. Zheng, L. Li, Z. Zhang, Y. Huang, L. Wang, Relational network for
skeleton-based action recognition, in: 2019 IEEE International Conference on
Multimedia and Expo (ICME), IEEE, 2019, pp. 826–831.

 

 
 [158] 
 
L. Pigou, A. Van Den Oord, S. Dieleman, M. Van Herreweghe, J. Dambre, Beyond
temporal pooling: Recurrence and temporal convolutions for gesture
recognition in video, International Journal of Computer Vision 126 (2) (2018)
430–439.

 

 
 [159] 
 
X. Chai, Z. Liu, F. Yin, Z. Liu, X. Chen, Two streams recurrent neural networks
for large-scale continuous gesture recognition, in: 2016 23rd international
conference on pattern recognition (ICPR), IEEE, 2016, pp. 31–36.

 

 
 [160] 
 
H. Mahmud, M. M. Morshed, M. Hasan, A deep-learning–based multimodal
depth-aware dynamic hand gesture recognition system, arXiv preprint
arXiv:2107.02543 (2021).

 

 
 [161] 
 
K. Lai, S. N. Yanushkevich, Cnn+ rnn depth and skeleton based dynamic hand
gesture recognition, in: 2018 24th international conference on pattern
recognition (ICPR), IEEE, 2018, pp. 3451–3456.

 

 
 [162] 
 
Z. Shi, T.-K. Kim, Learning and refining of privileged information-based rnns
for action recognition from depth sequences, in: Proceedings of the IEEE
conference on computer vision and pattern recognition, 2017, pp. 3461–3470.

 

 
 [163] 
 
J.-F. Hu, W.-S. Zheng, J. Pan, J. Lai, J. Zhang, Deep bilinear learning for
rgb-d action recognition, in: Proceedings of the European conference on
computer vision (ECCV), 2018, pp. 335–351.

 

 
 [164] 
 
J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep
bidirectional transformers for language understanding, arXiv preprint
arXiv:1810.04805 (2018).

 

 
 [165] 
 
A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al., Improving
language understanding by generative pre-training (2018).

 

 
 [166] 
 
Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, V. Stoyanov, Roberta: A robustly optimized bert pretraining
approach, arXiv preprint arXiv:1907.11692 (2019).

 

 
 [167] 
 
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou,
W. Li, P. J. Liu, Exploring the limits of transfer learning with a unified
text-to-text transformer, arXiv preprint arXiv:1910.10683 (2019).

 

 
 [168] 
 
M. Ott, S. Edunov, D. Grangier, M. Auli, Scaling neural machine translation,
arXiv preprint arXiv:1806.00187 (2018).

 

 
 [169] 
 
H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, H. Jégou,
Training data-efficient image transformers distillation through attention,
in: International Conference on Machine Learning, PMLR, 2021, pp.
10347–10357.

 

 
 [170] 
 
N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko,
End-to-end object detection with transformers, in: European conference on
computer vision, Springer, 2020, pp. 213–229.

 

 
 [171] 
 
Y. Wang, Z. Xu, X. Wang, C. Shen, B. Cheng, H. Shen, H. Xia, End-to-end video
instance segmentation with transformers, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2021, pp. 8741–8750.

 

 
 [172] 
 
J. Chen, C. M. Ho, Mm-vit: Multi-modal video transformer for compressed video
action recognition, arXiv preprint arXiv:2108.09322 (2021).

 

 
 [173] 
 
B. Zhao, Y. Wang, K. Su, H. Ren, H. Sun, “reading pictures instead of
looking”: Rgb-d image-based action recognition via capsule network and
kalman filter, Sensors 21 (6) (2021) 2217.

 

 
 [174] 
 
J. He, S. Gao, Tbsn: Sparse-transformer based siamese network for few-shot
action recognition, in: 2021 2nd Information Communication Technologies
Conference (ICTC), IEEE, 2021, pp. 47–53.

 

 
 [175] 
 
S. Sudhakaran, A. Bulat, J.-M. Perez-Rua, A. Falcon, S. Escalera, O. Lanz,
B. Martinez, G. Tzimiropoulos, Saic_cambridge-hupba-fbk submission to the
epic-kitchens-100 action recognition challenge 2021, arXiv preprint
arXiv:2110.02902 (2021).

 

 
 [176] 
 
S. Sharma, R. Kiros, R. Salakhutdinov, Action recognition using visual
attention, arXiv preprint arXiv:1511.04119 (2015).

 

 
 [177] 
 
Y. Wang, S. Wang, J. Tang, N. O’Hare, Y. Chang, B. Li, Hierarchical attention
network for action recognition in videos, arXiv preprint arXiv:1607.06416
(2016).

 

 
 [178] 
 
R. Girdhar, D. Ramanan, Attentional pooling for action recognition, arXiv
preprint arXiv:1711.01467 (2017).

 

 
 [179] 
 
Z. Li, K. Gavrilyuk, E. Gavves, M. Jain, C. G. Snoek, Videolstm convolves,
attends and flows for action recognition, Computer Vision and Image
Understanding 166 (2018) 41–50.

 

 
 [180] 
 
W. Du, Y. Wang, Y. Qiao, Recurrent spatial-temporal attention network for
action recognition in videos, IEEE Transactions on Image Processing 27 (3)
(2017) 1347–1360.

 

 
 [181] 
 
L. Huang, Y. Huang, W. Ouyang, L. Wang, Part-aligned pose-guided recurrent
network for action recognition, Pattern Recognition 92 (2019) 165–176.

 

 
 [182] 
 
M. Jaderberg, K. Simonyan, A. Zisserman, Spatial transformer networks, Advances
in neural information processing systems 28 (2015) 2017–2025.

 

 
 [183] 
 
H. Ge, Z. Yan, W. Yu, L. Sun, An attention mechanism based convolutional lstm
network for video action recognition, Multimedia Tools and Applications
78 (14) (2019) 20533–20556.

 

 
 [184] 
 
C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan,
V. Vanhoucke, A. Rabinovich, Going deeper with convolutions, in: Proceedings
of the IEEE conference on computer vision and pattern recognition, 2015, pp.
1–9.

 

 
 [185] 
 
R. Girdhar, J. Carreira, C. Doersch, A. Zisserman, Video action transformer
network, in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2019, pp. 244–253.

 

 
 [186] 
 
S. Ren, K. He, R. Girshick, J. Sun, Faster r-cnn: Towards real-time object
detection with region proposal networks, Advances in neural information
processing systems 28 (2015) 91–99.

 

 
 [187] 
 
M. Kalfaoglu, S. Kalkan, A. A. Alatan, Late temporal modeling in 3d cnn
architectures with bert for action recognition, in: European Conference on
Computer Vision, Springer, 2020, pp. 731–747.

 

 
 [188] 
 
A. Kozlov, V. Andronov, Y. Gritsenko, Lightweight network architecture for
real-time action recognition, in: Proceedings of the 35th Annual ACM
Symposium on Applied Computing, 2020, pp. 2074–2080.

 

 
 [189] 
 
K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition,
in: Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 770–778.

 

 
 [190] 
 
V. Mazzia, S. Angarano, F. Salvetti, F. Angelini, M. Chiaberge, Action
transformer: A self-attention model for short-time human action recognition,
arXiv preprint arXiv:2107.00606 (2021).

 

 
 [191] 
 
Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, H. Hu, Video swin
transformer, arXiv preprint arXiv:2106.13230 (2021).

 

 
 [192] 
 
Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin
transformer: Hierarchical vision transformer using shifted windows, arXiv
preprint arXiv:2103.14030 (2021).

 

 
 [193] 
 
B. Jiang, J. Yu, L. Zhou, K. Wu, Y. Yang, Two-pathway transformer network for
video action recognition, in: 2021 IEEE International Conference on Image
Processing (ICIP), IEEE, 2021, pp. 1089–1093.

 

 
 [194] 
 
G. Bertasius, H. Wang, L. Torresani, Is space-time attention all you need for
video understanding, arXiv preprint arXiv:2102.05095 2 (3) (2021) 4.

 

 
 [195] 
 
H. Fan, B. Xiong, K. Mangalam, Y. Li, Z. Yan, J. Malik, C. Feichtenhofer,
Multiscale vision transformers, in: Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 6824–6835.

 

 
 [196] 
 
A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lučić, C. Schmid,
Vivit: A video vision transformer, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 6836–6846.

 

 
 [197] 
 
J. Liu, G. Wang, L.-Y. Duan, K. Abdiyeva, A. C. Kot, Skeleton-based human
action recognition with global context-aware attention lstm networks, IEEE
Transactions on Image Processing 27 (4) (2017) 1586–1599.

 

 
 [198] 
 
J. Liu, G. Wang, P. Hu, L.-Y. Duan, A. C. Kot, Global context-aware attention
lstm networks for 3d action recognition, in: Proceedings of the IEEE
conference on computer vision and pattern recognition, 2017, pp. 1647–1656.

 

 
 [199] 
 
S. Song, C. Lan, J. Xing, W. Zeng, J. Liu, An end-to-end spatio-temporal
attention model for human action recognition from skeleton data, in:
Proceedings of the AAAI conference on artificial intelligence, Vol. 31, 2017.

 

 
 [200] 
 
R. Bai, M. Li, B. Meng, F. Li, J. Ren, M. Jiang, D. Sun, Gcst: Graph
convolutional skeleton transformer for action recognition, arXiv preprint
arXiv:2109.02860 (2021).

 

 
 [201] 
 
Y.-B. Cheng, X. Chen, J. Chen, P. Wei, D. Zhang, L. Lin, Hierarchical
transformer: Unsupervised representation learning for skeleton-based human
action recognition, in: 2021 IEEE International Conference on Multimedia and
Expo (ICME), IEEE, 2021, pp. 1–6.

 

 
 [202] 
 
Y.-B. Cheng, X. Chen, D. Zhang, L. Lin, Motion-transformer: self-supervised
pre-training for skeleton-based action recognition, in: Proceedings of the
2nd ACM International Conference on Multimedia in Asia, 2021, pp. 1–6.

 

 
 [203] 
 
Y. Sun, Y. Shen, L. Ma, Msst-rt: Multi-stream spatial-temporal relative
transformer for skeleton-based action recognition, Sensors 21 (16) (2021)
5339.

 

 
 [204] 
 
C. Plizzari, M. Cannici, M. Matteucci, Skeleton-based action recognition via
spatial and temporal transformer networks, Computer Vision and Image
Understanding 208 (2021) 103219.

 

 
 [205] 
 
C. Plizzari, M. Cannici, M. Matteucci, Spatial temporal transformer network for
skeleton-based action recognition, in: International Conference on Pattern
Recognition, Springer, 2021, pp. 694–701.

 

 
 [206] 
 
G. Hu, B. Cui, S. Yu, Skeleton-based action recognition with synchronous local
and non-local spatio-temporal learning and frequency attention, in: 2019 IEEE
International Conference on Multimedia and Expo (ICME), IEEE, 2019, pp.
1216–1221.

 

 
 [207] 
 
Y. Zhang, B. Wu, W. Li, L. Duan, C. Gan, Stst: Spatial-temporal specialized
transformer for skeleton-based action recognition, in: Proceedings of the
29th ACM International Conference on Multimedia, 2021, pp. 3229–3237.

 

 
 [208] 
 
F. Shi, C. Lee, L. Qiu, Y. Zhao, T. Shen, S. Muralidhar, T. Han, S.-C. Zhu,
V. Narayanan, Star: Sparse transformer-based action recognition, arXiv
preprint arXiv:2107.07089 (2021).

 

 
 [209] 
 
F. Baradel, C. Wolf, J. Mille, Human action recognition: Pose-based attention
draws focus to hands, in: Proceedings of the IEEE International Conference on
Computer Vision Workshops, 2017, pp. 604–613.

 

 
 [210] 
 
F. Baradel, C. Wolf, J. Mille, Human activity recognition with pose-driven
attention to rgb, in: BMVC 2018-29th British Machine Vision Conference, 2018,
pp. 1–14.

 

 
 [211] 
 
S. Das, A. Chaudhary, F. Bremond, M. Thonnat, Where to focus on for human
action recognition?, in: 2019 IEEE Winter Conference on Applications of
Computer Vision (WACV), IEEE, 2019, pp. 71–80.

 

 
 [212] 
 
B. Debnath, M. O’Brient, S. Kumar, A. Behera, Attention-driven body pose
encoding for human activity recognition, in: 2020 25th International
Conference on Pattern Recognition (ICPR), IEEE, 2021, pp. 5897–5904.

 

 
 [213] 
 
X. Li, Y. Hou, P. Wang, Z. Gao, M. Xu, W. Li, Trear: Transformer-based rgb-d
egocentric action recognition, IEEE Transactions on Cognitive and
Developmental Systems (2021).

 

 
 [214] 
 
C.-Y. Ma, M.-H. Chen, Z. Kira, G. AlRegib, Ts-lstm and temporal-inception:
Exploiting spatiotemporal dynamics for activity recognition, Signal
Processing: Image Communication 71 (2019) 76–87.

 

 
 [215] 
 
S. Arif, J. Wang, T. Ul Hassan, Z. Fei, 3d-cnn-based fused feature maps with
lstm applied to action recognition, Future Internet 11 (2) (2019) 42.

 

 
 [216] 
 
Y. Wang, Y. Xiao, F. Xiong, W. Jiang, Z. Cao, J. T. Zhou, J. Yuan, 3dv: 3d
dynamic voxel for action recognition in depth video, in: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2020, pp.
511–520.

 

 
 [217] 
 
C. R. Qi, L. Yi, H. Su, L. J. Guibas, Pointnet++: Deep hierarchical feature
learning on point sets in a metric space, Advances in neural information
processing systems 30 (2017).

 

 
 [218] 
 
Z. Xu, Y. Wang, J. Jiang, J. Yao, L. Li, Adaptive feature selection with
reinforcement learning for skeleton-based action recognition, IEEE Access 8
(2020) 213038–213051.

 

 
 [219] 
 
C. Li, P. Wang, S. Wang, Y. Hou, W. Li, Skeleton-based action recognition using
lstm and cnn, in: 2017 IEEE International Conference on Multimedia Expo
Workshops (ICMEW), IEEE, 2017, pp. 585–590.

 

 
 [220] 
 
F. Ye, S. Pu, Q. Zhong, C. Li, D. Xie, H. Tang, Dynamic gcn: Context-enriched
topology learning for skeleton-based action recognition, in: Proceedings of
the 28th ACM International Conference on Multimedia, 2020, pp. 55–63.

 

 
 [221] 
 
H. Liu, J. Tu, M. Liu, Two-stream 3d convolutional neural network for
skeleton-based action recognition, arXiv preprint arXiv:1705.08106 (2017).

 

 
 [222] 
 
J. Tu, M. Liu, H. Liu, Skeleton-based human action recognition using spatial
temporal 3d convolutional neural networks, in: 2018 IEEE International
Conference on Multimedia and Expo (ICME), IEEE, 2018, pp. 1–6.

 

 
 [223] 
 
W. Nie, W. Wang, X. Huang, Srnet: Structured relevance feature learning network
from skeleton data for human action recognition, IEEE Access 7 (2019)
132161–132172.

 

 
 [224] 
 
H. Wang, Z. Song, W. Li, P. Wang, A hybrid network for large-scale action
recognition from rgb and depth modalities, Sensors 20 (11) (2020) 3305.

 

 
 [225] 
 
L. Zhang, G. Zhu, P. Shen, J. Song, S. Afaq Shah, M. Bennamoun, Learning
spatiotemporal features using 3dcnn and convolutional lstm for gesture
recognition, in: Proceedings of the IEEE International Conference on Computer
Vision Workshops, 2017, pp. 3120–3128.

 

 
 [226] 
 
P. Molchanov, X. Yang, S. Gupta, K. Kim, S. Tyree, J. Kautz, Online detection
and classification of dynamic hand gestures with recurrent 3d convolutional
neural network, in: Proceedings of the IEEE conference on computer vision and
pattern recognition, 2016, pp. 4207–4215.

 

 
 [227] 
 
C. Dhiman, D. K. Vishwakarma, View-invariant deep architecture for human action
recognition using two-stream motion and shape temporal dynamics, IEEE
Transactions on Image Processing 29 (2020) 3835–3844.

 

 
 [228] 
 
Y. Li, Q. Miao, X. Qi, Z. Ma, W. Ouyang, A spatiotemporal attention-based
resc3d model for large-scale gesture recognition, Machine Vision and
Applications 30 (5) (2019) 875–888.

 

 
 [229] 
 
Q. Miao, Y. Li, W. Ouyang, Z. Ma, X. Xu, W. Shi, X. Cao, Multimodal gesture
recognition based on the resc3d network, in: Proceedings of the IEEE
International Conference on Computer Vision Workshops, 2017, pp. 3047–3055.

 

 
 [230] 
 
J. Duan, S. Zhou, J. Wan, X. Guo, S. Z. Li,
 Multi-modality fusion based on consensus-voting
and 3d convolution for isolated gesture recognition , arXiv preprint
arXiv:1611.06689 (2016).

 URL files/280/1611.html 

 

 
 [231] 
 
M. Al-Faris, J. P. Chiverton, Y. Yang, D. Ndzi, Multi-view region-adaptive
multi-temporal dmm and rgb action recognition, Pattern Analysis and
Applications 23 (4) (2020) 1587–1602.

 

 
 [232] 
 
H. Wu, X. Ma, Y. Li, Spatiotemporal multimodal learning with 3d cnns for video
action recognition, IEEE Transactions on Circuits and Systems for Video
Technology (2021).

 

 
 [233] 
 
A. Elboushaki, R. Hannane, K. Afdel, L. Koutti, Multid-cnn: A multi-dimensional
feature learning approach based on deep convolutional networks for gesture
recognition in rgb-d image sequences, Expert Systems with Applications 139
(2020) 112829.

 

 
 [234] 
 
G. Zhu, L. Zhang, P. Shen, J. Song, Multimodal
gesture recognition using 3-d convolution and convolutional lstm , Ieee
Access 5 (2017) 4517–4524.

 URL files/276/7880648.html 

 

 
 [235] 
 
S. Song, C. Lan, J. Xing, W. Zeng, J. Liu, Skeleton-indexed deep multi-modal
feature learning for high performance human action recognition, in: 2018 IEEE
International Conference on Multimedia and Expo (ICME), IEEE, 2018, pp. 1–6.

 

 
 [236] 
 
C. Zhao, M. Chen, J. Zhao, Q. Wang, Y. Shen, 3d behavior recognition based on
multi-modal deep space-time learning, Applied Sciences 9 (4) (2019) 716.

 

 
 [237] 
 
L. Shi, Y. Zhang, J. Cheng, H. Lu, Skeleton-based action recognition with
multi-stream adaptive graph convolutional networks, IEEE Transactions on
Image Processing 29 (2020) 9532–9545.

 

 
 [238] 
 
S. Das, S. Sharma, R. Dai, F. Bremond, M. Thonnat, Vpn: Learning video-pose
embedding for activities of daily living, in: European Conference on Computer
Vision, Springer, 2020, pp. 72–90.

 

 
 [239] 
 
S. Das, R. Dai, D. Yang, F. Bremond, Vpn++: Rethinking video-pose embeddings
for understanding activities of daily living, IEEE Transactions on Pattern
Analysis and Machine Intelligence (2021).

 

 
 [240] 
 
S. Das, R. Dai, M. Koperski, L. Minciullo, L. Garattoni, F. Bremond,
G. Francesca, Toyota smarthome: Real-world activities of daily living, in:
Proceedings of the IEEE/CVF International Conference on Computer Vision,
2019, pp. 833–842.

 

 
 [241] 
 
Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, R. Salakhutdinov,
Transformer-xl: Attentive language models beyond a fixed-length context,
arXiv preprint arXiv:1901.02860 (2019).

 

 
 [242] 
 
K. Ohnishi, M. Hidaka, T. Harada, Improved dense trajectory with cross streams,
in: Proceedings of the 24th ACM international conference on Multimedia, 2016,
pp. 257–261.

 

 
 [243] 
 
J. Zhu, W. Zou, L. Xu, Y. Hu, Z. Zhu, M. Chang, J. Huang, G. Huang, D. Du,
Action machine: Rethinking action recognition in trimmed videos, arXiv
preprint arXiv:1812.05770 (2018).

 

 
 [244] 
 
A. Piergiovanni, M. S. Ryoo, Recognizing actions in videos from unseen
viewpoints, in: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2021, pp. 4124–4132.

 

 
 [245] 
 
C. Feichtenhofer, X3d: Expanding architectures for efficient video recognition,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2020, pp. 203–213.

 

 
 [246] 
 
C. Feichtenhofer, H. Fan, J. Malik, K. He, Slowfast networks for video
recognition, in: Proceedings of the IEEE/CVF international conference on
computer vision, 2019, pp. 6202–6211.

 

 
 [247] 
 
K. Papadopoulos, E. Ghorbel, D. Aouada, B. Ottersten, Vertex feature encoding
and hierarchical temporal modeling in a spatial-temporal graph convolutional
network for action recognition, arXiv preprint arXiv:1912.09745 (2019).

 

 
 [248] 
 
Y. Obinata, T. Yamamoto, Temporal extension module for skeleton-based action
recognition, in: 2020 25th International Conference on Pattern Recognition
(ICPR), IEEE, 2021, pp. 534–540.

 

 
 [249] 
 
T. Chen, D. Zhou, J. Wang, S. Wang, Y. Guan, X. He, E. Ding, Learning
multi-granular spatio-temporal graph network for skeleton-based action
recognition, in: Proceedings of the 29th ACM International Conference on
Multimedia, 2021, pp. 4334–4342.

 

 
 [250] 
 
H. Duan, Y. Zhao, K. Chen, D. Shao, D. Lin, B. Dai, Revisiting skeleton-based
action recognition, arXiv preprint arXiv:2104.13586 (2021).

 

 
 [251] 
 
M. Davoodikakhki, K. Yin, Hierarchical action classification with network
pruning, in: International Symposium on Visual Computing, Springer, 2020, pp.
291–305.

 

 
 [252] 
 
B. Mahasseni, S. Todorovic, Regularizing long short term memory with 3d
human-skeleton sequences for action recognition, in: Proceedings of the IEEE
conference on computer vision and pattern recognition, 2016, pp. 3054–3062.

 

 
 [253] 
 
S. Yan, X. Xiong, A. Arnab, Z. Lu, M. Zhang, C. Sun, C. Schmid, Multiview
transformers for video recognition, arXiv preprint arXiv:2201.04288 (2022).

 

 
 [254] 
 
B. Zhang, J. Yu, C. Fifty, W. Han, A. M. Dai, R. Pang, F. Sha, Co-training
transformer with videos and images improves action recognition, arXiv
preprint arXiv:2112.07175 (2021).

 

 
 [255] 
 
C. Wei, H. Fan, S. Xie, C.-Y. Wu, A. Yuille, C. Feichtenhofer, Masked feature
prediction for self-supervised visual pre-training, arXiv preprint
arXiv:2112.09133 (2021).

 

 
 [256] 
 
L. Yuan, D. Chen, Y.-L. Chen, N. Codella, X. Dai, J. Gao, H. Hu, X. Huang,
B. Li, C. Li, Florence: A new foundation model for computer vision, arXiv
preprint arXiv:2111.11432 (2021).

 

 
 [257] 
 
Z. Liu, H. Hu, Y. Lin, Z. Yao, Z. Xie, Y. Wei, J. Ning, Y. Cao, Z. Zhang,
L. Dong, Swin transformer v2: Scaling up capacity and resolution, arXiv
preprint arXiv:2111.09883 (2021).

 

 
 [258] 
 
Y. Li, C.-Y. Wu, H. Fan, K. Mangalam, B. Xiong, J. Malik, C. Feichtenhofer,
Improved multiscale vision transformers for classification and detection,
arXiv preprint arXiv:2112.01526 (2021).

 

 
 [259] 
 
Z. Tong, Y. Song, J. Wang, L. Wang, Videomae: Masked autoencoders are
data-efficient learners for self-supervised video pre-training, arXiv
preprint arXiv:2203.12602 (2022).

 

 
 [260] 
 
M. S. Ryoo, A. Piergiovanni, A. Arnab, M. Dehghani, A. Angelova, Tokenlearner:
What can 8 learned tokens do for images and videos?, arXiv preprint
arXiv:2106.11297 (2021).

 

 
 [261] 
 
X. Wang, R. Girshick, A. Gupta, K. He, Non-local neural networks, in:
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 7794–7803.

 

 
 [262] 
 
D. Yang, Y. Wang, A. Dantcheva, L. Garattoni, G. Francesca, F. Bremond, Unik: A
unified framework for real-world skeleton-based action recognition, arXiv
preprint arXiv:2107.08580 (2021).

 

 
 [263] 
 
M. S. Ryoo, A. Piergiovanni, J. Kangaspunta, A. Angelova, Assemblenet++:
Assembling modality representations via attention connections, in: European
Conference on Computer Vision, Springer, 2020, pp. 654–671.

 

 
 [264] 
 
J. Kangaspunta, A. Piergiovanni, R. Jonschkowski, M. Ryoo, A. Angelova,
Adaptive intermediate representations for video understanding, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2021, pp. 1602–1612.

 

 
 [265] 
 
D. Yang, M. M. Li, H. Fu, J. Fan, H. Leung, Centrality graph convolutional
networks for skeleton-based action recognition, arXiv preprint
arXiv:2003.03007 (2020).

 

 
 [266] 
 
J. Liu, A. Shahroudy, M. Perez, G. Wang, L.-Y. Duan, A. C. Kot, Ntu rgb+ d 120:
A large-scale benchmark for 3d human activity understanding, IEEE
transactions on pattern analysis and machine intelligence 42 (10) (2019)
2684–2701.

 

 
 [267] 
 
W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan,
F. Viola, T. Green, T. Back, P. Natsev, The kinetics human action video
dataset, arXiv preprint arXiv:1705.06950 (2017).

 

 
 [268] 
 
J. Carreira, E. Noland, A. Banki-Horvath, C. Hillier, A. Zisserman, A short
note about kinetics-600, arXiv preprint arXiv:1808.01340 (2018).

 

 
 [269] 
 
R. Goyal, S. Ebrahimi Kahou, V. Michalski, J. Materzynska, S. Westphal, H. Kim,
V. Haenel, I. Fruend, P. Yianilos, M. Mueller-Freitag, et al., The" something
something" video database for learning and evaluating visual common sense,
in: Proceedings of the IEEE international conference on computer vision,
2017, pp. 5842–5850.

 

 
 [270] 
 
K. Soomro, A. R. Zamir, M. Shah, Ucf101: A dataset of 101 human actions classes
from videos in the wild, arXiv preprint arXiv:1212.0402 (2012).

 

 
 [271] 
 
Y. Li, Y. Li, N. Vasconcelos, Resound: Towards action recognition without
representation bias, in: Proceedings of the European Conference on Computer
Vision (ECCV), 2018, pp. 513–528.

 

 
 [272] 
 
J. Jang, D. Kim, C. Park, M. Jang, J. Lee, J. Kim, Etri-activity3d: A
large-scale rgb-d dataset for robots to recognize daily activities of the
elderly, in: 2020 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS), IEEE, 2020, pp. 10990–10997.

 

 
 [273] 
 
G. Rogez, P. Weinzaepfel, C. Schmid, Lcr-net++: Multi-person 2d and 3d pose
detection in natural images, IEEE transactions on pattern analysis and
machine intelligence 42 (5) (2019) 1146–1161.

 

 
 [274] 
 
L. Smaira, J. Carreira, E. Noland, E. Clancy, A. Wu, A. Zisserman, A short note
on the kinetics-700-2020 human action dataset, arXiv preprint
arXiv:2010.10864 (2020).

 

 
 [275] 
 
S. Yan, Y. Xiong, D. Lin, Spatial temporal graph convolutional networks for
skeleton-based action recognition, arXiv preprint arXiv:1801.07455 (2018).

 

 
 [276] 
 
Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, Realtime multi-person 2d pose
estimation using part affinity fields, in: Proceedings of the IEEE conference
on computer vision and pattern recognition, 2017, pp. 7291–7299.

 

 
 [277] 
 
M. Liu, J. Yuan, Recognizing human actions as the evolution of pose estimation
maps, in: Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, 2018, pp. 1159–1168.

 

 
 [278] 
 
Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut, Albert: A lite
bert for self-supervised learning of language representations, 2019.

 

 
 [279] 
 
P. Michel, O. Levy, G. Neubig, Are sixteen heads really better than one?,
Advances in neural information processing systems 32 (2019).

 

 
 [280] 
 
X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, L. Li, F. Wang, Q. Liu, Tinybert:
Distilling bert for natural language understanding, arXiv preprint
arXiv:1909.10351 (2019).

 

 
 [281] 
 
S. Shen, Z. Dong, J. Ye, L. Ma, Z. Yao, A. Gholami, M. W. Mahoney, K. Keutzer,
Q-bert: Hessian based ultra low precision quantization of bert, in:
Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34, 2020,
pp. 8815–8821.

 

 
 [282] 
 
C. Xu, W. Zhou, T. Ge, F. Wei, M. Zhou, Bert-of-theseus: Compressing bert by
progressive module replacing, arXiv preprint arXiv:2002.02925 (2020).

 

 
 [283] 
 
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, Language models are few-shot
learners, Advances in neural information processing systems 33 (2020)
1877–1901.

 

 
 [284] 
 
H. Chen, Y. Wang, T. Guo, C. Xu, Y. Deng, Z. Liu, S. Ma, C. Xu, C. Xu, W. Gao,
Pre-trained image processing transformer, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2021, pp.
12299–12310.

 

 
 [285] 
 
A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, J. Carreira,
Perceiver: General perception with iterative attention, in: International
conference on machine learning, PMLR, 2021, pp. 4651–4664.

 

 
 [286] 
 
A. Jaegle, S. Borgeaud, J.-B. Alayrac, C. Doersch, C. Ionescu, D. Ding,
S. Koppula, D. Zoran, A. Brock, E. Shelhamer, et al., Perceiver io: A general
architecture for structured inputs outputs, arXiv preprint
arXiv:2107.14795 (2021).

 

 
 [287] 
 
J. Yang, X. Dong, L. Liu, C. Zhang, J. Shen, D. Yu, Recurring the transformer
for video action recognition, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2022, pp. 14063–14073.

 

 
 [288] 
 
A. Buades, B. Coll, J.-M. Morel, A non-local algorithm for image denoising, in:
2005 IEEE computer society conference on computer vision and pattern
recognition (CVPR’05), Vol. 2, Ieee, 2005, pp. 60–65.

 

 
 [289] 
 
J. Pérez, J. Marinković, P. Barceló, On the turing completeness of modern
neural network architectures, arXiv preprint arXiv:1901.03429 (2019).

 

 
 [290] 
 
J.-B. Cordonnier, A. Loukas, M. Jaggi, On the relationship between
self-attention and convolutional layers, arXiv preprint arXiv:1911.03584
(2019).

 

 
 [291] 
 
J. L. Ba, J. R. Kiros, G. E. Hinton, Layer normalization, arXiv preprint
arXiv:1607.06450 (2016).