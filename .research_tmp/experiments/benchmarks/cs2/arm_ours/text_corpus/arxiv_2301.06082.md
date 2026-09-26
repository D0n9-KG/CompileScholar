A Survey on Human Action Recognition 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2301.06082v1 [cs.CV] 20 Dec 2022 
 
 

# A Survey on Human Action Recognition 

 
 
 Shuchang Zhou
 
 Affiliation: University of Electronic Science and Technology of China
 
 Email: 202222011839@std.uestc.edu.cn 
 

 Abstract 
 
 Human Action Recognition (HAR), one of the most important tasks in computer vision, has developed rapidly in the past decade and has a wide range of applications in health monitoring, intelligent surveillance, virtual reality, human-computer interaction and so on. Human actions can be represented by a wide variety of modalities, such as RGB-D cameras, audio, inertial sensors, etc. Consequently, in addition to the mainstream single modality-based HAR approaches, more and more research is devoted to the multimodal domain due to the complementary properties between multimodal data. In this paper, we present a survey of HAR methods in recent years according to the different input modalities. Meanwhile, considering that most of the recent surveys on HAR focus on the third perspective, while this survey aims to provide a more comprehensive introduction to HAR novices and researchers, we therefore also investigate the actions recognition methods from the first perspective in recent years. Finally, we give a brief introduction about the benchmark HAR datasets and show the performance comparison of different methods on these datasets.

 
 
 
   
   
 
 
 
 
 | 

 
 
 
 

## 1 Introduction

 
 As an important part of computer vision, HAR is a key technology for machines to understand the world as well as human behavior. In recent years, with the continuous development of deep learning and sensor technology, the performance of HAR has been significantly improved, and it has a wide range of practical applications, including health-care, human-computer interaction, virtual reality, etc.

 
 
 Reviewing the development of HAR, a large number of research was based on RGB at the beginning. At the early stage, extensive studies focusing on hand-crafted feature-based approaches have been carried out [ 8 , 72 , 138 , 139 ] . With the rise of deep learning, a growing number of neural network-based methods have been proposed with excellent performance, which are categorized into blocks, including two-stream Convolutional Neural Network (CNN) [ 119 , 142 , 143 , 44 , 141 , 104 , 147 ] , Recurrent Neural Network (RNN) [ 128 , 133 , 53 , 35 , 6 , 50 , 99 , 87 ] , 3D CNN [ 130 , 15 , 52 , 131 , 42 ] , and Transformer-based methods [ 134 , 36 , 7 , 5 , 153 , 155 ] . In recent years, due to the emergence of various sensors, human actions can be represented by other diverse modalities.3D skeleton modality has attracted the interest of many researchers, and skeleton-based HAR using deep learning framework for skeleton sequence perform well, which can be mainly divided into four categories: RNN [ 140 , 73 , 78 , 90 , 74 ] , CNN [ 75 , 144 , 81 , 11 ] ,) and graph convolution network (GCN) [ 154 , 76 , 160 , 159 , 22 , 123 , 77 , 23 , 79 ] .

 
 
 Since single mode has its own limitations, the concept of multi-modality has been proposed and attracted great attention. Multimodal-based methods can further improve the performance of action recognition by taking advantage of the complementary characteristics between multimodal data. Compared with the action recognition based on single modality, the multimodal-based HAR has higher accuracy. However, it may bring various challenges such as data acquisition, feature extraction, fusion, and temporal synchronization. From the traditional perspective, this paper focuses on the multimodal fusion between visual sensors and non-visual sensors, including fusion of RGB and audio modality [ 68 , 92 , 136 , 152 , 161 ] , fusion of RGB and inertial sensors modality [ 150 , 149 , 148 , 86 ] , and fusion of RGB-D and inertial sensors modality [ 97 , 111 , 60 , 38 , 1 , 30 , 28 , 2 , 3 ] .

 
 
 At the same time, in recent years, thanks to the popularity of wearable intelligent devices and the rapid development of video social platforms, there is a blowout type growth of video data under the first perspective, and egocentric action recognition(EAR) has become an active research area. The difference between first person point of view (FPV) and a third person view is that the former has a certain initiative, and it is closer to the human eye perception, namely, the camera’s angle is driven by people’s attention itself, while conventional perspective camera is fixed. In that case, the action recognition methods based on FPV and a third person view have certain differences, EAR is worth further exploration. Currently, EAR has been applied in various fields, including extreme sports, health monitoring, life recording, etc., and more and more researchers are exploring in the field of EAR [ 39 , 41 , 40 , 96 , 120 ] . In the past few years, several features based on egocentric cues [ 39 , 106 , 40 , 82 , 112 ] such as gaze, hand and head movements, and hand posture have been suggested for first-person action recognition. With the rise of deep learning, advanced deep learning-based methods for EAR are increasingly available. The networks used for extracting spatial-temporal information from self-centered videos can be divided into two categories. The first category [ 113 , 96 , 164 , 71 , 151 ] is based on CNN networks to generate spatial-temporal features, and the second category [ 126 , 134 , 125 , 46 ] mainly uses Long Short-Term Memory (LSTM) and its variants. However, most of these methods require large amounts of annotation data. EAR has been further investigated by reweighting spatial or temporal features through Attention Mechanism [ 94 , 95 , 127 , 125 , 105 ] . In addition, joint modeling studies on egocentric gaze prediction and action recognition [ 84 , 57 , 83 , 100 ] have addressed the uncertainty in gaze. What is more, there are some other approaches from recent years [ 116 , 146 , 108 ] .

 
 
 In spite of the fact that there are few methods for multimodal action recognition from the first perspective, they have already attracted the attention of researchers. This paper briefly introduces some viable multimodal fusion approaches for EAR, which utilizes data from RGB and depth [ 115 , 129 , 80 ] , RGB and audio [ 93 , 68 , 16 , 145 , 107 , 67 ] , RGB and inertial sensors [ 121 , 122 , 34 , 33 , 32 , 156 , 4 ] .

 
 
 The contributions of this survey can be summarized as follows:

 
 • 
 
 we classify and sort the third-person HAR methods in detail according to different learning frameworks. Meanwhile, multimodal action recognition methods from the third perspective are summarized as well.

 

 • 
 
 we investigate the egocentric action recognition methods based on single modality and multi-modality, which just came up in the last few years.

 

 • 
 
 we introduce a variety of widely used action recognition datasets, meanwhile, performance of representative action recognition methods are summarized and compared. This survey provides some guidance for researchers who are interested in this direction.

 

 
 The remainder of this survey is organized as follows. In Section 2 , for the third perspective, single-modal HAR based on RGB and skeleton data, multimodal fusion methods, and multimodal HAR are reviewed respectively. Section 3 introduces the egocentric action recognition based on single modality and multi-modality. Section 4 briefly introduces the benchmark datasets of action recognition and shows the performance of representative methods on these datasets. The structure of the paper is shown in Figure 1 .

 
 
 Figure 1: Structure of this paper. 
 
 
 

## 2 Third-Person Action Recognition

 

### 2.1 Single Modality-Based HAR 

 
 In the field of HAR, methods based on single modality are soon going to reach maturity. Different single modal data including RGB, skeleton, depth, infrared, audio, inertia sensors, etc. have their own characteristics and many researches have been conducted on action recognition based on different modalities [ 44 , 123 , 152 , 63 , 137 , 132 ] . This paper mainly focuses on the two mainstream single modality behavior recognition, which are based on RGB data and 3D skeleton data, respectively.

 
 

#### 2.1.1 RGB Modality

 
 Looking back at the development of HAR, a large number of studies are conducted based on RGB data, and among which the representative third-person HAR methods are shown in the figure 2 . In the first few years, extensive studies focusing on hand-crafted feature-based approaches have been proposed. With more advanced computers and networks, the proliferation of video data, as well as the rapid development of deep learning, a growing number of studies of HAR is conducted based on the deep learning framework. The results show that the HAR methods under the deep learning have superior performance, which gradually replace the traditional methods and have become the mainstream research. These representative deep learning frameworks are elaborated later, including two-stream CNN, RNN, 3D CNN, and Transformer.

 
 
 Figure 2: Representative third-person HAR methods based on RGB modality. 
 
 
 Hand-crafted feature methods .
Before the rise of deep learning, researchers used traditional recognition methods, i.e., manually extracting features through machine learning and classifying them by relevant algorithms. There are three main categories, spatio-temporal volume based methods [ 8 ] , spatio-temporal interest point (STIP) based methods [ 72 ] and trajectory based methods [ 138 , 139 ] . Bobicket et al. [ 8 ] proposed to use the motion-energy images (MEI) and motion-history images (MHI) by projecting the human body along the time axis in a 3D cube, then classify behavior through template matching method. However, this method is not applicable to complex scenes. The classical STIP method was mainly proposed by Laptev [ 72 ] , whose main idea is to extend the feature detection technology from 2D image to 3D spatio-temporal and calculate its feature descriptors, but this method ignores many details of video, and has weak generalization ability as well. Wang et al. [ 138 , 139 ] proposed the dense trajectory (DT) and improved trajectory (IDT). Spatial feature points are detected on each frame of image, and these feature points are tracked individually at each scale to form a trajectory of fixed length, which is finally described by descriptors. The advantage of IDT method lies in the estimation of camera motion by matching SURF descriptors and dense optical flow feature points between the front and back frames, thus eliminating the effect caused by camera motion. It is classified finally by a support vector machine after feature extraction. DT and IDT, although being traditional methods, are comparable to some deep learning methods, meanwhile the method can achieve fantastic results when combined with deep learning framework. The disadvantage is the speed of this algorithm is slow and it needs to accurately track feature points, which is challenging for computers.

 
 
 Two-stream CNN-based methods .
Simonyan et al. [ 119 ] proposed a two-stream CNN model composed of spatial flow and temporal flow, where spatial flow obtains appearance information and the temporal flow obtains motion information, and finally, the average method or SVM is used for fusion of classification scores. Subsequently, quite a few methods have been proposed based on the improvement of the two-stream model [ 142 , 143 , 44 , 141 , 104 , 147 ] . Wang et al. [ 142 ] found that many two-stream CNNS are relatively shallow, therefore, they designed very deep two-stream CNNS to obtain better recognition results. Feichtenhofer et al. [ 44 ] studied several fusion strategies and proposed that it is effective to fuse spatial flow and temporal flow in the last convolutional layer, thus reducing the number of parameters while maintaining accuracy. In order to solve the problem of not being able to model long time domain structures, Wang et al. [ 143 ] proposed temporal segment networks (TSN), which design a sparse sampling scheme to represent temporal features that could model the whole video. The authors proposed two additional input modalities: RGB difference and warped optical flow fields to improve the learning efficiency of the original two-stream network. Wang et al. [ 141 ] proposed trajectory-pooled deep-convolutional descriptors (TDD) by combining classical IDT manual features and two-stream depth features. A novel spatiotemporal pyramid network (SPN) [ 147 ] was proposed to integrate the spatiotemporal features in the pyramid structure so that they can enhance each other. Peng et al. [ 104 ] proposed the two-stream collaborative learning with spatial-temporal attention (TCLSTA) approach, which consists of the spatial-temporal attention model and the static-motion collaborative model, consequently improves the recognition performance by exploiting the strong complementarity of static and dynamic information.
Two-stream CNN networks are able to capture high semantic representations by using CNNS on spatiotemporal features. However, precomputed optical flow is computationally expensive and storage demanding, meanwhile, the networks have limitations in modeling the video-level temporal information effectively as it is insensitive to the time series information in the video.

 
 
 RNN-based methods .
The time series information in video data, which HAR needs to learn from videos, is one of the important factors. Therefore, RNN [ 128 , 133 , 53 , 35 , 6 ] is an ideal choice. Among them, the research areas in video-based HAR using LSTM [ 55 ] have attracted a lot of attention. In LRCN [ 128 ] , CNN features are extracted from a single frame and then input to LSTM for HAR task. Beyond Short-Snippets [ 157 ] extract features from pre-trained 2D CNNS and feed these features to the stacked LSTM framework. [ 133 , 53 ] adopted bidirectional LSTM, which is composed of two independent LSTMS for learning the forward and backward temporal information of HAR. Sun et al. proposed Latch-LSTM [ 35 ] to learn the independent hidden state transition of LSTM memory units, so as to further effectively model the dynamic information in video sequences for better classification. In addition to using LSTM, some studies have conducted HAR via GRU [ 6 , 37 , 69 , 117 , 165 ] , which has fewer gates [ 24 ] compared to LSTM, resulting in fewer model parameters, but it can generally provide similar performance to LSTM for HAR. ShuttleNet [ 117 ] is a deep network inspired by biology that is embedded in CNN-RNN framework containing a multilayer loop-connected GRU processor. In addition, there are some studies that incorporate attention mechanisms [ 50 , 99 , 87 ] to benefit LSTM-based framework to obtain better HAR performance.

 
 
 3D CNN-based methods .
Recently, 3D CNN-based methods [ 130 , 15 , 52 , 131 , 42 ] have achieved good performance in HAR. C3D [ 130 ] is one of the earliest video-based 3D CNN models for HAR, where video frames with spatial and temporal dimensions are directly fed into a 3D CNN without any preprocessing. I3D [ 15 ] combines two-stream network and 3D CNN to extend 2D Inception-v1 network to 3D structure. Given the ability of ResNet to alleviate the degradation problem of network deepening, Tran et al. [ 52 ] designed 3D Residual Networks (R3D), which extend 2D convolution of ResNet to 3D. Tran et al. [ 131 ] proposed to use (2+1)D convolution instead of 3D convolution, decomposing the 3D convolution operation into two-dimensional spatial convolution and one-dimensional temporal convolution. T3D [ 31 ] can intensively and effectively capture temporal information. SlowFast network [ 43 ] contains two 3D CNN networks to handle slow and fast paths running at different frame rates. Lin [ 88 ] added Temporal Shift Module (TSM) to the ResNet to achieve the performance of 3D CNN, while keeping the relatively small computation cost of 2D CNN. X3D [ 42 ] attempts to expand 2D convolution from an different dimension to make it suitable for 3D spatio-temporal data processing.

 
 
 Transformer-based methods .
Transformer [ 134 ] with attention mechanism as the core is a novel research hotspot recently, due to its powerful ability and broad prospects, and the application of transformer to RGB-based action recognition has achieved unprecedently superior performance [ 7 , 5 , 153 , 20 , 155 , 102 ] . Bertasius et al. [ 7 ] extended ViT (ViT [ 36 ] is another Transformer used for image classification) to videos by decomposing each video into a series of frame-level patches, then, a dividing attention mechanism is proposed to apply spatial and temporal attention separately within each block of the model. ViViT [ 5 ] extracts spatio-temporal feature information from the input video, which is then encoded by a series of transformer layers. MTV-H (WTS) [ 153 ] introduces multi-view converters consisting of multiple individual encoders, which effectively fuse information from different representations of the input video. However, they also suffer from severe memory and computation overhead. Therefore, many efforts have been made to reduce the computational complexity and memory cost such as RegionViT [ 20 ] , RViT [ 155 ] .

 
 
 

#### 2.1.2 3D Skeleton Modality

 
 3D skeleton data is another common modality as skeletal sequence encodes the trajectory of human joints, which represents informative human motion. Skeletal data therefore has been undergoing a lot of researches in HAR, and the use of skeletons is increasing. In addition, sensors such as Microsoft Kinect [ 162 ] and advanced human pose estimation algorithms [ 25 , 12 ] also make it easier to obtain accurate 3D skeleton data.
The earliest skeletal sequence-based action recognition use hand-crafted feature-based methods [ 135 , 58 , 163 ] . However, these traditional methods can only perform well in specific datasets, it is difficult for these methods to be applied in a wider application fields. With the rise of deep learning, skeleton sequence HAR based on deep learning framework has gradually become the mainstream. In the following, we review deep learning methods, which can be mainly divided into four categories: RNN, CNN, Graph Convolution Network (GCN), and Transformer-based methods. Representative third-person HAR methods based on 3D skeleton modality are shown in the figure 3 .

 
 
 Figure 3: Representative third-person HAR methods based on 3D skeleton modality. 
 
 
 RNN-based methods .
In the work of [ 140 ] , an end-to-end two-stream RNN architecture was proposed to simulate temporal dynamics and spatial configurations. To model the variable temporal dynamics of skeleton sequences, Lee et al. [ 74 ] proposed end-to-end Memory Attention Network (MAN), namely, it is a temporal-then-spatial recalibration scheme designed to alleviate complex spatio-temporal variations of skeleton joints. Lee et al. [ 73 ] proposed novel ensemble Temporal Sliding LSTM (TS-LSTM) networks containing short-term, medium-term and long-term TS-LSTM networks, respectively. IndRNN [ 78 ] constructs an independent recurrent neural network that not only solves the problems of gradient disappearance and explosion, but also faster than the original LSTM. Given that not all joints provide information for action analysis, global context-aware attention [ 90 ] is added to the LSTM network, which selectively focuses on the information joints in the skeleton sequence. Zhang et al. [ 158 ] proposed adaptive neural networks VA-RNN and VA-CNN, specifically, in each stream, the view adaptive module automatically determines and identifies the best observation point in the process.

 
 
 CNN-based methods .
CNN is obviously better than RNN in image extraction, however, its problem regarding temporal modeling is still the direction of focus. Wang [ 144 ] proposed Joint Trajectory Graph (JTM), which represents the spatio-temporal information in 3D skeleton sequences into three 2D images through color coding. Since only adjacent joints within the convolution kernel will be considered to learn co-occurrence features, that is, some potential correlations associated with all joints may be ignored. Li et al. [ 75 ] proposed an end-to-end framework for learning co-occurrence features using a hierarchical approach. A geometric algebraic representation [ 81 ] of shape motion was proposed, which makes full use of the information provided by skeletal sequences. Similarly, Caetano et al. proposed SkeleMotion [ 11 ] and the Tree Structure Reference Joints Image (TSRJI) [ 10 ] as representations of HAR skeleton sequences.

 
 
 GCN-based methods .
Given that skeletal sequences based on CNN or RNN cannot completely simulate the spatio-temporal information, while the skeleton data is in the form of graph structure, which also has strong expressive ability. Inspired by topological graph, HAR methods based on GNN or GCN have been proposed successively, and their experimental results show that the graph structure viewing skeleton data as edges and nodes can better carry out HAR. Since GCN methods have been extensively studied, this paper will focus on GCN-based HAR.

 
 
 Yan et al. [ 154 ] first proposed a novel skeleton-based action recognition model, namely, Spatial Temporal Graph Convolution Network (ST-GCN), which constructs a spatiotemporal graph with joints as graph vertices, meanwhile body structure and time as graph edges. In order to reflect implicit joint correlation, Li et al. [ 76 ] further proposed Actional-structural Graph Convolutional Networks (AS-GCN), which combine actional links and structural links into a generalized skeleton graph. Actional links are used to capture potential dependencies specific to actions, while structural links are used to represent higher-order dependencies. Furthermore, in the work of [ 160 ] , contextual information integration is used to effectively model long-term dependencies. A high-level semantics of joints is introduced for HAR in [ 159 ] . In addition, to reduce the computational cost of GCN, Cheng et al. designed Shift Graph Convolutional Network (Shift-GCN) [ 22 ] , which employs Shift-graph operations and lightweight point-wise convolutions instead of using heavy regular graph convolutions. Following this line of research, Song et al. [ 123 ] proposed a multi-stream GCN model, specifically, separable convolutional layers are embedded into an early fused Multiple Input Branches (MIB) network where the input branches including joint position, motion velocity, and skeletal features, thus, greatly reduce redundant trainable parameters while increasing the capacity of the model. Li et al. [ 77 ] proposed symbiotic GCN to simultaneously handle action recognition and motion prediction tasks, which allows the two tasks to enhance each other. In [ 23 ] , InfoGCN was proposed including an information bottleneck objective to learn the most informative action representations, and an attention-based graph convolution to capture context-dependent skeleton topologies.

 
 
 
 

### 2.2 Multimodal Fusion

 
 Information from different sensing modes can be fused in different ways. Generally speaking, three fusion methods are outlined: early fusion, late fusion, and hybrid fusion.

 
 

#### 2.2.1 Early Fusion

 
 Both feature-level and data-level fusion are referred to as early fusion. Data-level fusion is suitable for homogeneous multi-sensor data (e.g., two or more RGB cameras or depth cameras, etc.). When there are two or more heterogeneous sensors, feature-level fusion or decision-level fusion techniques are usually applied. The modalities are often highly correlated with each other, but it is difficult to extract this correlation in both feature layer and data layer. According to literature [ 54 ] , correlation between information contained in different data streams can only be found at a higher level. Researchers usually use dimensionality reduction techniques to eliminate redundancy problems in the input space. For example, the Principal Component Analysis (PCA) method in literature [ 119 ] is widely used in dimensionality reduction processing in multimodal deep learning. In addition, the multimodal early fusion method also needs to solve the problem of time synchronization among different data sources, several methods were proposed to solve this problem in [ 101 ] , such as Convolutional, Training and Pool Fusion, which can well integrate discrete event sequence and continuous signal to realize time synchronization between modalities.

 
 
 

#### 2.2.2 Late Fusion

 
 Late fusion is also called decision-level fusion, where deep learning models are trained on different modes first, and then the output results of multiple models are fused. This fusion approach is often favored as the fusion process is feature independent and errors from multiple models are generally uncorrelated. Currently, late fusion methods mainly use rules to determine the how to combination of output results on different models, that is, rule fusion, such as Max-Fusion, Averaged Fusion, Bayes Rule Fusion, and Ensemble Learning, etc. [ 64 ] .

 
 
 

#### 2.2.3 Hybrid Fusion

 
 In [ 101 ] , early and late fusion methods are compared and the performance of both methods had a lot to do with specific problems. Namely, early fusion is superior to the late fusion when the correlation between the modes is relatively large, while when each mode is not related to a great extent, such as dimension and the sampling rate are highly uncorrelated, adopting late fusion method is more suitable. Therefore, the two methods have their own advantages and disadvantages, which need to be selected according to the requirements in practical application. Hybrid fusion combines early and late fusion methods, which integrates the advantages of the both, while increases the structure complexity and training difficulty of the model. Due to the diversity and flexibility of deep learning model structures, it is more suitable to use hybrid fusion method, which is widely used in multimedia, gesture recognition and other fields.

 
 
 
 

### 2.3 Multimodal HAR

 
 Vision sensors include RGB-D cameras, infrared, time of flight, light field, thermal imagers, etc. Visual modalities are widely used because of their excellent representation capabilities. Non-visual sensors include accelerometer, gyroscope, magnetometer, audio signals, electrothermal activity response, etc. The sensors have been used individually or in combination for HAR. The following will mainly focus on multimodal fusion of visual sensors and non-visual sensors, including the multimodal fusion between RGB and audio modalities, RGB and inertial sensors modalities, RGB-D and inertial sensors modalities.

 
 

#### 2.3.1 RGB and Audio

 
 Audio data provides complementary information to appearance and motion in visual data. In recent years, several deep learning-based methods have been proposed to fuse RGB and audio modalities for HAR. The joint modeling of audio and visual signals is mainly carried out in the way of late fusion [ 68 , 92 ] . Wang et al. [ 136 ] introduced three-stream CNN to extract multimodal features from audio signals, RGB frames and optical flows. Both feature fusion and score fusion are evaluated, with the former achieving better performance. Inspired by the work of [ 43 ] , Xiao et al. [ 152 ] introduced a hierarchically integrated audio-visual representation framework with slow and fast visual paths that are deeply combined with multilayers audio paths. Chen et al. [ 21 ] proposed a multimodal video convert called Multi-Modal Video Transformer (MM-ViT), which operates in the compressed video domain and exploits all readily available modalities to avoid calculation of optical flow, i.e., I-frames, motion vectors, residuals and audio waveform. In [ 161 ] , an audio-infused recognizer was proposed, which effectively models the cross-modal interaction across domains.

 
 
 

#### 2.3.2 RGB and Inertial Sensors

 
 In addition to vision-based sensors, inertial sensors have been used for human action recognition, allowing recognition beyond the limited field of view of vision-based sensors. Inertial sensors contain accelerometers and gyroscopes that provide acceleration and angular velocity signals for HAR, and a survey [ 45 ] details the performance of current deep learning models and future challenges for sensor-based action recognition. The wearable inertial sensors provide 3D motion data, consisting of the three -axis acceleration of the accelerometer and the three-axis angular velocity of the gyroscope. Inertial and video data in [ 150 , 149 , 148 ] are captured simultaneously by inertial sensors and video cameras and converted into 2D and 3D images. These images are then fed into 2D and 3D CNNs to fuse their decisions in order to detect and identify a specific set of actions of interest from a continuous stream of actions. In [ 149 , 148 ] , a decision-level fusion method is considered. In [ 150 ] , both feature-level fusion and decision-level fusion are tested, and decision-level fusion achieves higher accuracy. In [ 86 ] , visual and inertial sensor integration algorithms were proposed for efficient and accurate generic abnormal behavior detection among the elderly, which closely cooperate to achieve high accuracy and real-time performance.

 
 
 

#### 2.3.3 RGB-D and Inertial Sensors

 
 In recent years, many studies have improved the accuracy of HAR by fusing features extracted from depth and inertia sensor data and using co-representation classifiers. Better accuracy results have been obtained due to the complementarity of the data from two modalities. In the vast majority of the work on action or gesture recognition, it is assumed that the action of interest has been separated from the action stream [ 91 , 97 , 17 , 19 , 111 ] . In [ 97 ] , decision-level fusion between depth camera data and wearable sensor data is performed to increase the ability of the robot to recognize human actions. In [ 91 ] , the depth and inertia data are effectively fused to train the Hidden Markov Model to improve the accuracy and robustness of gesture recognition. For continuous action flow, Dawar et al. [ 29 ] detected and recognized human actions from continuous action flow by fusing both depth and inertia sensing modalities.

 
 
 In addition, many deep learning methods have been proposed recently [ 38 , 1 , 30 , 28 , 2 , 3 , 45 ] . In [ 1 ] , a shared feature layer is used after multimodal feature-level fusion, and then support vector machines or softmax classifiers are used to recognize actions based on combined features. Considering that deep inertia training data is limited, Dawar et al. [ 30 ] proposed a data enhancement framework based on depth and inertia modalities, which are fed to CNN and LSTM, respectively, and then the scores of the two models are fused during testing for better classification. Given that deep learning model allows to extract features at all levels of the structure so that rich multi-layer features are obtained, while existing methods do not take advantage of these rich multi-level information. Specifically, the main drawback of existing deep learning-based HAR fusion methods based on depth and inertia sensors is that the fusion is performed at a single level or stage, either at the feature level or at the decision level, and therefore the true semantic information of the data cannot be mapped to the classifier. By designing different two-stream CNN architectures, several deep inertial fusion techniques are also studied in [ 2 , 3 ] , where inertial signals are converted into images using the techniques in [ 62 ] . Three new deep multilevel multimodal ( ​ M 2 \emph{M}^{2} ) fusion frameworks are proposed in [ 2 ] to take advantage of different fusion strategies at different stages. Ahmad et al. [ 3 ] proposed a new Multistage Gated Average Fusion (MGAF) network, which extracts and fuses features from all layers of CNN. Recently, Ijaz et al. [ 59 ] proposed a multimodal Transformer for nursing action recognition, in which the correlation between skeleton and acceleration data is modeled by Transformer.

 
 
 
 
 

## 3 Egocentric Action Recognition

 

### 3.1 Single Modality-Based EAR

 
 In the last decade, with the emergence of low-cost and lightweight multi-sensor wearable devices, such as GoPro, Google Glass, Microsoft Hololens, etc., video data from the first perspective has surged year by year, and HAR technology based on first view has been applied to various fields, including extreme sports, health monitoring, life recording and so on. The field of EAR has also published its representative dataset [ 26 , 118 , 27 ] and has attracted a lot of interest, with more and more researchers have explored the EAR field [ 39 , 41 , 40 , 96 , 120 ] in the past decade. Given that first-person action recognition is not like counterpart based third-view, where the camera is either static or moves smoothly, while in egocentric videos there is a large vibration due to the wearer’s head movement. Therefore, it is difficult to apply the third-person action recognition methods directly to the first view.

 
 
 In the past few years, several features based on egocentric cues [ 39 , 106 , 40 , 82 , 112 ] , such as gaze, movements of hand and head, and hand posture have been suggested for first-person HAR. First of all, [ 39 , 41 , 106 , 98 ] recognize the importance of hand in first-person action recognition. Pirsiavash and Ramanan [ 106 ] proposed an egocentric behavior representation based on hand-object interaction, developing a combination of HOG features for modeling object and hand appearance during activities. [ 40 , 82 ] find that gaze position is an important cue for EAR, but such fine-grained information is difficult to detect. However, the direct application of local features in egocentric videos is problematic in that it ignores the fact that camera motion is also a useful cue for understanding egocentric behavior. Motion features also play an important role in egocentric action analysis [ 112 , 109 ] , where Ryoo et al. integrated global and local motion information to model interaction-level human activities. In addition, Ying et al. [ 85 ] proposed egocentric cues that combine head movement, hand posture, and gaze to better characterize egocentric actions.

 
 
 With the rise of deep learning, advanced deep learning-based methods for EAR are increasingly available. The networks used to extract spatio-temporal information from egocentric videos can be divided into two main categories, and the first category is based on CNN to generate spatio-temporal features. An Ego-ConvNet combining egocentric cues is proposed in [ 120 ] . In [ 96 ] , the network is trained to segment the hand and locate the objects, and then cropped the objects to the input of the appearance flow. After that, appearance flow and motion flow are fused at late-level through the fully connected layer to jointly identify objects and action verbs. Zhou et al. [ 164 ] proposed a hybrid cascaded end-to-end deep CNN to jointly infer hand maps and manipulate foreground object maps. A two-stream CNN architecture with long-term fusion pools [ 71 ] was proposed to efficiently capture the temporal structure of actions with appearance and motion. Wu et al. [ 151 ] combined the long-term feature banks containing detection features with 3D CNN to improve the accuracy of target recognition. These approaches mentioned above utilize local information for EAR based on specialized hand segmentation and so on, for which requires a large amount of annotation data and additional information such as hand masks [ 120 , 85 ] or target information [ 96 ] in addition to input images. The second class uses LSTM and variants [ 126 , 134 , 125 , 46 ] to generate embedded representations based on temporal relationships between feature frames. In [ 134 ] , spatial attention is considered, where significant regions in each frame are taken as the input of LSTM for action recognition, while spatial attention is further correlated through successive frames in [ 125 ] , which a two-stage Long Short-Term Attention (LSTA) model is proposed for locating active objects. At the same time, as attention mechanism [ 94 , 95 , 127 , 125 , 105 ] can effectively localize the region of interest on the feature map, video-based EAR is further studied through the attention mechanism. Lu et al. [ 94 ] proposed a spatial attention network to predict human gaze in the form of attention maps, which help two streams focus on the most relevant spatial areas in a video frame and thus recognize actions. The method in [ 95 ] is a further extension of [ 94 ] , which combines gaze information and attention mechanism. Namely, it uses gaze information as a supervision, and then learns a spatial-temporal attention, lastly integrates the corresponding modules into the two-stream model.

 
 
 Many of the previous work clearly demonstrates the advantages of using egocentric gaze in HAR based on FPV. However, they all model gaze and action separately rather than jointly. Studies [ 84 , 57 , 83 , 100 ] show the joint modeling of egocentric gaze prediction and action recognition. Huang et al. [ 57 ] jointly modeled two coupled tasks of gaze prediction and action recognition while taking into account the context information, with two core modules, more specifically, action-based gaze prediction module and gaze-guided action recognition module are proposed. Li [ 83 ] proposed a deep model for jointly learning eye gaze prediction and action recognition, which is an extension of [ 40 ] , modeling eye gaze information as a probability variable to explain its uncertainty and then obtains the gaze map, which is combined with features extracted by the neural network later on , thereby getting final recognition results.

 
 
 Beyond that, recently, Shan et al. [ 116 ] developed a hand-object detector to locate moving objects. When the detector is well trained, it can be deployed directly on the target dataset without fine-tuning. An end-to-end interactive prototype learning (IPL) framework [ 146 ] was proposed to learn better active object representations by exploiting the motion cues of participants, without the additional cost of object detection and eye gaze annotation. Meanwhile, Plizzari [ 108 ] suggested that the intrinsic motion information encoded by the event data is a very valuable modality for EAR.

 
 
 

### 3.2 Multimodal EAR

 
 At present, there are few studies on multimodal fusion based on the first perspective. The following briefly surveys multimodal EAR, which are mainly divided into RGB and depth modalities, RGB and audio modalities, RGB and wearable inertial sensors modalities.

 
 

#### 3.2.1 RGB and Depth

 
 As described in [ 115 ] , the main limitations of RGB video are the lack of 3D information and sensitivity to illumination changes, which the depth modality is able to compensate for. A multi-stream network is proposed in [ 129 ] using Cauchy estimator and orthogonal constraints to combine features from RGB, depth and optical flow. The RGB-D egocentric dataset with hand posture annotations is published in [ 47 ] , but they do not propose any method based on RGB and depth. However, with the aforementioned methods, they either can only simulate short-term movements or can only consider the temporal structure as the activity proceeds sequentially. In view of this problem, the recently proposed Transformer [ 134 ] can be applied to the EAR of RGB-D, Li et al. [ 80 ] introduced the Transformer framework for the EAR, where RGB and its corresponding depth data are processed by the interframe attention encoder. whereafter fused by mutual attention blocks.

 
 
 

#### 3.2.2 RGB and Audio

 
 Although some work incorporating audiovisual resources have been reported in the first- person action recognition challenge [ 48 , 49 ] , they provide little model detail. Attention mechanisms for action recognition using audio as a modal branch are proposed in [ 92 , 93 ] , while the use of audio-visual cues for object interaction recognition is still very limited. Inspired by TSN [ 143 ] , a Temporal Binding Network (TBN) [ 68 ] was proposed, which takes audio, RGB and optical flow as input through three similar CNN network streams, and then uses a Temporal Binding Window (TBW) when fusing audiovisual features. In this method, modalities are fused before time aggregation, with shared modality and fusion weights over time. Simultaneously, the proposed architecture is trained end-to-end, it thus outperforms individual modalities as well as late-fusion of modalities. The model in [ 16 ] combines the sparse temporal sampling strategy with the late fusion of audio, spatial and temporal streams, specifically, its audio input is the spectral map extracted from the original audio of the video, while its visual input is RGB and optical stream frames. Wang et al. [ 145 ] solved both problems using a technique called gradient blending, which computes the best fusion of modalities according to their overfitting behavior, and demonstrates the importance of audio in EAR area. However, across all publications in the field, there remains an unresolved problem with EAR, that is, the network relies heavily on the environment in which the activity is recorded and does not perform well in unfamiliar environments, which means that the model trained on source labeled datasets does not generalize well to unseen datasets. The ability to generalize to unseen domains is demonstrated in [ 107 ] when audio modality combined with RGB, namely, the Relative Norm Alignment (RNA) loss is proposed, which progressively aligns the relative feature norms from the audio and RGB modalities to obtain the domain-invariant features consequently. Multimodal Temporal Context Network (MTCN) [ 67 ] was proposed, which learns to attend to surrounding actions and models multimodal temporal context, more specifically, it is constituted by a transformer encoder that utilizes visual and audio as input context and language as output context.

 
 
 

#### 3.2.3 RGB and Wearable Inertial Sensors

 
 Ozcan et al. [ 103 ] utilized histograms of edge orientations together with the gradient local binary patterns for fall detection, which then combined with three-axis signal of the accelerometer. Experimental results show that the proposed fusion method not only has higher sensitivity, but also significantly reduces the number of false alarms compared with the accelerometer and camera only methods. In [ 121 ] , a multi-stream convolutional network is extended to analysis activity in egocentric videos, meanwhile, a novel multi-stream LSTM is proposed for classifying wearable sensor data. Finally, two score fusion techniques, namely average pooling and maximum pooling, are evaluated to obtain recognition results. Song et al. [ 122 ] proposed a new technique to integrate temporal information into sensor data with similar trajectories. Moreover, the Fisher Kernel framework is applied to fuse sensor and video data for EAR with Multimodal Fisher Vector (MFV). In the work [ 34 ] , features are extracted by applying a sliding window to video and inertial data, Whereafter, using majority voting to fuse the results. For the classification task, methods of Random Forest and Support Vector Machine are taken into consideration. The methods in [ 33 , 32 ] are extensions of [ 34 ] , which use time and frequency domain features for acceleration data, and object information encoding hand interaction for visual data. Experiments are carried out on both feature-level fusion and decision-level fusion, and the latter achieves higher accuracy. In [ 156 ] , a hierarchical fusion framework based on deep learning is developed and implemented, where LSTM and CNN are used for EAR based on motion sensor data and photo streams at different levels, respectively. Experimental results show that the proposed model enables motion sensor data and photo streams to work in the most suitable classification mode, so as to effectively eliminate the negative impact of sensor differences on the fusion results. A novel framework is proposed in [ 4 ] , where multi-kernel learning (MKL) is used to fuse multimodal features in order to adaptively weighs the visual, audio, and sensor features, additionally, feature and kernel weighting and recognition tasks are performed simultaneously. Huang et al. [ 56 ] proposed a first-view multimodal framework based on knowledge driven, GCN and LSTM, which improve the performance of EAR under conditions of few samples and ultra-small datasets.

 
 
 
 
 

## 4 Datasets and Performance

 

### 4.1 Datasets

 
 At present, there are a large number of datasets for third-view action recognition, and they are quite perfect. While there is still room for improvement of egocentric datasets, though the video data from the first perspective is growing due to the popularity of wearable devices in the past few years. The following is a brief introduction to the action recognition datasets, which is shown in Table 1 at the same time.

 
 

#### 4.1.1 Third-Person Video Datasets

 
 HMDB51 .
Released by Brown university in 2011, HMDB51 [ 70 ] is based mostly on movies, but also on public databases and online video repositories such as YouTube. There are a total of 6849 samples, divided into 51 categories, each category contains at least 101 samples.

 
 
 UCF101 .
The UCF101 [ 124 ] dataset was released in 2012. Collected from YouTube, it is an extension of UCF50 [ 110 ] and includes 50 original action classes and 51 new classes. The 101 classes can be divided into 5 categories: human-object interaction, human action, human-person interaction, musical instrument playing, and sports, with a total of 13,320 videos.

 
 
 Sports-1M .
Sports-1M [ 65 ] consists of 1 million videos annotated with 487 classes. Each class contains about 1000-3000 video clips. Videos in Sports-1M are automatically collected by the YouTube Themes API.

 
 
 UTD-MHAD .
UTD-MHAD [ 18 ] consists of dataset collected synchronously in four modes, which include RGB, depth, skeletal position, inertial signals from the Kinect camera and a suite of wearable inertial sensors that can be used for 27 classes of human actions.

 
 
 Activitynet .
Activitynet [ 9 ] was launched in 2015. It is a human activity dataset with 200 activity categories and approximately 24K videos, where videos range in length from a few seconds to ten minutes.

 
 
 Something-Something .
Something-Something [ 51 ] contains 108,499 videos, spanning 174 tags, each lasting 2-6 seconds. Unlike other datasets that focus on human activities, which are human-object interactions, it is more accurate to say that "hands" interact with objects, such as actions like, drop, and poke.

 
 
 NTU RGB+D .
NTU RGB+D [ 114 ] contains 60 kinds of actions, a total of 56,880 samples, which are collected by Microsoft Kinect v2 sensor, and use three different camera angles, in the form of depth, 3D skeleton, RGB and infrared sequence. Two different partitioning criteria are used to divide the dataset into training set and test set, namely, Cross-Subject (C-Subject) and Cross-View (C-View). The Cross-Subject divides the training set and test set according to the person ID, while Cross-View divides the training set and test set by cameras, with samples collected by camera 1 as the test set, and cameras 2 and 3 as the training set.

 
 
 NTU-RGB+D 120 .
Recently, an extended version of the original NTU-RGB+D has also been proposed, called NTU-RGB+D 120 [ 89 ] , with 120 action classes and 114,480 samples and two protocols of Cross-Subject (C-Subject) and Cross-Setup (C-Setup) similarly.

 
 
 Kinetics-400/600/700 .
kinetics datasets, which have been updated annually over the past several years. In 2017, the Kinetics-400 dataset [ 66 ] was developed, which contains 400 activity categories and more than 300,000 10-second video clips extracted from YouTube. In 2018, Kinetics-600 [ 13 ] was introduced and 100 classes were added. Kinetics-700 [ 14 ] was released the following year, adding 100 classes.

 
 
 

#### 4.1.2 Egocentric Video Datasets

 
 ADL .
The ADL [ 106 ] dataset consists of 20 egocentric videos collected by 20 people. Action annotations and object annotations are provided, with a total of 18 action categories and 44 objects annotated.

 
 
 GTEA Gaze+ .
The GTEA [ 40 ] dataset was performed by 4 different subjects, consisting of 7 long-term activities, 28 videos with a total of 11 action categories, captured using head mounted cameras.

 
 
 Dogcentric .
The Dogcentric [ 61 ] dataset, one of the most popular FPV datasets, consists of 209 videos (102 training videos and 107 test videos), with a variety of first-person actions divided into 10 action categories.

 
 
 EGTEA Gaze+ .
EGTEA Gaze+ [ 84 ] is the largest and most comprehensive FPV action and gaze dataset. Specifically, it contains 86 unique phases of 28 hours cooking activities from 32 subjects, with 10,325 action annotations, 19 verbs, 51 nouns, and 106 unique actions. Moreover, the videos come with audio and gaze tracking, human annotations of actions and hand masks are provided simultaneously.

 
 
 EpicKitchens-55/100 .
EpicKitchens-55 [ 26 ] was recorded by 32 participants in four cities using head mounted cameras in their native kitchen environments, with 55 hours of video totaling 39,594 action clips. Meanwhile, the action is defined as a combination of verbs and nouns for a total of 125 verb classes and 331 noun classes. EpicKitchens-100 [ 27 ] is an extension of Epickitchens-55, which contains 100 hours of 90k action clips, including 97 verb classes and 300 noun classes, recorded by 45 participants in their kitchens in 4 cities.

 
 
 Table 1: Summary of representative video action recognition datasets. 
 R:RGB,D:Depth, S:Skeleton, Au:Audio, IR:Infrared, I:Inertial. 
 
 
 | 
 Dataset | 
 Samples | 
 Catagories | 
 Modality | 
 Year | 

 
 
 
 Third-Person 
 Video 
 Datasets 
 | 
 HMDB51 [ 70 ] | 
 7K | 
 51 | 
 R | 
 2011 | 

 
 UCF50 [ 110 ] | 
 7K | 
 50 | 
 R | 
 2011 | 

 
 UCF101 [ 124 ] | 
 13K | 
 101 | 
 R | 
 2012 | 

 
 Sports-1M [ 65 ] | 
 1.1M | 
 487 | 
 R | 
 2014 | 

 
 UTD-MHAD [ 18 ] | 
 861 | 
 27 | 
 R,D,S,I | 
 2015 | 

 
 ActivityNet [ 9 ] | 
 27K | 
 203 | 
 R | 
 2015 | 

 
 Something-Something [ 51 ] | 
 108K | 
 174 | 
 R | 
 2017 | 

 
 NTU RGB+D [ 114 ] | 
 57K | 
 60 | 
 R,D,S | 
 2016 | 

 
 NTU-RGB+D 120 [ 89 ] | 
 114K | 
 120 | 
 R,D,S | 
 2019 | 

 
 Kinetics-400 [ 66 ] | 
 306K | 
 400 | 
 R | 
 2017 | 

 
 Kinetics-600 [ 13 ] | 
 496K | 
 600 | 
 R | 
 2018 | 

 
 Kinetics-700 [ 14 ] | 
 650K | 
 700 | 
 R | 
 2019 | 

 
 
 
 Egocentric 
 Video 
 Datasets 
 | 
 ADL [ 106 ] | 
 436 | 
 32 | 
 R,I | 
 2012 | 

 
 GTEA Gaze+ [ 40 ] | 
 3K | 
 42 | 
 R,IR,Au | 
 2012 | 

 
 Dogcentric [ 61 ] | 
 209 | 
 10 | 
 RGB | 
 2014 | 

 
 EGTEA Gaze+ [ 84 ] | 
 10K | 
 106 | 
 R,IR,Au | 
 2018 | 

 
 EpicKitchens-55 [ 26 ] | 
 40K | 
 149 | 
 R,Au | 
 2018 | 

 
 EpicKitchens-100 [ 27 ] | 
 90K | 
 4053 | 
 R,Au,I | 
 2020 | 

 
 
 
 

### 4.2 Performance

 
 With regard of recognition methods from the third-person view, for RGB-based HAR, UCF101, HMDB51, and Kinectis-400 are widely used as benchmark datasets. Table 2 shows the performance of the representative methods on three datasets, which base on RGB modality and multimodal fusion between RGB and audio modality. Since many methods achieve accuracy of more than 97 % \% on UCF101, Kinetics dataset was introduced to evaluate the accuracy.

 
 
 Table 2: Performance of the methods based on RGB and multimodal fusion between RGB and audio which is occurred by UCF101,HMDB51,Kinetics400 datasets. 
 
 
 Modality | 
 Method | 
 UCF101 | 
 HMDB51 | 
 Kinetics400 | 
 Year | 

 
 RGB | 
 
 
 
 Hand-craft 
 feature 
 
 | 
 Temporal Template [ 8 ] | 
 - | 
 - | 
 - | 
 2001 | 

 
 STIP [ 72 ] | 
 - | 
 - | 
 - | 
 2005 | 

 
 DT [ 138 ] | 
 - | 
 46.60% | 
 - | 
 2011 | 

 
 IDT [ 139 ] | 
 85.90% | 
 57.20% | 
 - | 
 2013 | 

 
 
 
 Two-stream CNN 
 | 
 Two-stream [ 119 ] | 
 88.00% | 
 59.40% | 
 - | 
 2014 | 

 
 Deep Two-stream [ 142 ] | 
 91.40% | 
 57.20% | 
 - | 
 2015 | 

 
 TDD [ 141 ] | 
 91.50% | 
 65.90% | 
 - | 
 2015 | 

 
 TSN [ 143 ] | 
 94.20% | 
 69.40% | 
 - | 
 2016 | 

 
 Two-stream Fusion [ 44 ] | 
 92.50% | 
 65.40% | 
 - | 
 2016 | 

 
 SPN [ 147 ] | 
 94.60% | 
 68.90% | 
 - | 
 2017 | 

 
 TCLSTA [ 104 ] | 
 94.00% | 
 68.70% | 
 - | 
 2018 | 

 
 
 
 RNN 
 | 
 LRCN [ 35 ] | 
 82.70% | 
 - | 
 - | 
 2015 | 

 
 Beyond Short-Snippets [ 157 ] | 
 88.20% | 
 - | 
 - | 
 2015 | 

 
 Lattice-LSTM [ 128 ] | 
 93.60% | 
 66.20% | 
 - | 
 2017 | 

 
 Bi-LSTM [ 133 ] | 
 91.21% | 
 87.64% | 
 - | 
 2017 | 

 
 Db-LSTM [ 53 ] | 
 97.30% | 
 81.20% | 
 - | 
 2021 | 

 
 ShuttleNet [ 117 ] | 
 95.40% | 
 71.70% | 
 - | 
 2017 | 

 
 Attentional Pooling [ 50 ] | 
 - | 
 50.80% | 
 - | 
 2017 | 

 
 VideoLSTM [ 87 ] | 
 79.60% | 
 43.30% | 
 - | 
 2018 | 

 
 Spatio-temporal Attention [ 99 ] | 
 87.11% | 
 53.07% | 
 - | 
 2019 | 

 
 
 
 3D CNN 
 | 
 C3D [ 130 ] | 
 82.30% | 
 56.80% | 
 59.50% | 
 2015 | 

 
 I3D-Two Stream [ 15 ] | 
 97.90% | 
 80.20% | 
 75.70% | 
 2017 | 

 
 T3D [ 31 ] | 
 93.20% | 
 63.50% | 
 62.20% | 
 2017 | 

 
 R3D [ 52 ] | 
 94.50% | 
 70.20% | 
 65.10% | 
 2018 | 

 
 (2+1)D [ 131 ] | 
 97.30% | 
 78.70% | 
 75.40% | 
 2018 | 

 
 SlowFast 8×8, R101 [ 43 ] | 
 - | 
 - | 
 77.90% | 
 2019 | 

 
 TSM [ 88 ] | 
 95.90% | 
 73.50% | 
 74.70% | 
 2019 | 

 
 X3D-XL [ 42 ] | 
 - | 
 - | 
 79.10% | 
 2020 | 

 
 
 
 Transformer 
 | 
 VTN [ 102 ] | 
 - | 
 - | 
 79.80% | 
 2021 | 

 
 ViViT [ 5 ] | 
 - | 
 - | 
 84.80% | 
 2021 | 

 
 Timesformer [ 7 ] | 
 - | 
 - | 
 80.70% | 
 2021 | 

 
 MTV-H (WTS) [ 153 ] | 
 - | 
 - | 
 89.10% | 
 2022 | 

 
 RegionViT [ 20 ] | 
 - | 
 - | 
 77.60% | 
 2022 | 

 
 RViT [ 155 ] | 
 - | 
 - | 
 81.50% | 
 2022 | 

 
 
 
 RGB 
 and 
 Audio 
 | 
 | 
 Wang et al. [ 136 ] | 
 85.10% | 
 - | 
 - | 
 2016 | 

 
 Long et al. [ 92 ] | 
 94.60% | 
 69.20% | 
 79.40% | 
 2018 | 

 
 AVSlowFast 8×8, R101 [ 152 ] | 
 - | 
 - | 
 78.80% | 
 2020 | 

 
 MM-VIT(Kinetics pretain) [ 21 ] | 
 98.90% | 
 - | 
 - | 
 2022 | 

 
 
 As for 3D skeleton-based HAR, the performance based on 3D skeleton tada is shown in Table 3 below, with experiments conducted by NTU-RGB+D and NTU-RGB+D 120 datasets.

 
 
 Table 3: Performance of the skeleton-based HAR methods on NTU RGB+D and NTU RGB+D 120 datasets. 
 
 
 Modality | 
 Method | 
 NTU RGB+D | 
 NTU RGB+D 120 | 
 Year | 

 
 Cross-Subject | 
 Cross-View | 
 Cross-Subject | 
 Cross-Setup | 

 
 
 
 3D 
 Skeleton 
 | 
 
 
 RNN 
 | 
 Two-stream RNN [ 140 ] | 
 71.30% | 
 79.50% | 
 - | 
 - | 
 2017 | 

 
 TS-LSTM [ 73 ] | 
 74.60% | 
 81.30% | 
 - | 
 - | 
 2017 | 

 
 GCA-LSTM [ 90 ] | 
 74.40% | 
 82.80% | 
 58.30% | 
 59.20% | 
 2017 | 

 
 IndRNN [ 78 ] | 
 86.70% | 
 93.70% | 
 - | 
 - | 
 2018 | 

 
 MAN [ 74 ] | 
 82.70% | 
 93.20% | 
 - | 
 - | 
 2018 | 

 
 VA-RNN [ 158 ] | 
 79.80% | 
 88.90% | 
 - | 
 - | 
 2019 | 

 
 
 
 CNN 
 | 
 JIM [ 144 ] | 
 73.40% | 
 75.20% | 
 - | 
 - | 
 2016 | 

 
 HCN [ 75 ] | 
 86.50% | 
 91.10% | 
 - | 
 - | 
 2018 | 

 
 SkeleMotion [ 11 ] | 
 76.50% | 
 84.70% | 
 67.70% | 
 66.90% | 
 2019 | 

 
 TSRJI [ 10 ] | 
 73.30% | 
 80.30% | 
 67.90% | 
 62.80% | 
 2019 | 

 
 GA [ 81 ] | 
 82.90% | 
 90.00% | 
 - | 
 - | 
 2019 | 

 
 VA-CNN [ 158 ] | 
 88.70% | 
 94.30% | 
 - | 
 - | 
 2019 | 

 
 
 
 GCN 
 | 
 ST-GCN [ 154 ] | 
 81.50% | 
 88.30% | 
 - | 
 - | 
 2018 | 

 
 AS-GCN [ 76 ] | 
 86.80% | 
 94.20% | 
 - | 
 - | 
 2019 | 

 
 CA-GCN [ 160 ] | 
 83.50% | 
 91.40% | 
 - | 
 - | 
 2020 | 

 
 Shift-GCN [ 22 ] | 
 90.70% | 
 96.50% | 
 85.90% | 
 87.60% | 
 2020 | 

 
 SGN [ 159 ] | 
 89.00% | 
 94.50% | 
 79.20% | 
 81.50% | 
 2020 | 

 
 Sym-GNN [ 77 ] | 
 90.10% | 
 96.40% | 
 - | 
 - | 
 2021 | 

 
 Info-GCN [ 23 ] | 
 93.00% | 
 97.10% | 
 89.80% | 
 91.20% | 
 2022 | 

 
 
 Additionally, with respect to multimodal action recognition using inertial sensor modality, Table 4 also shows the performance on UTD-MHAD dataset.

 
 
 Table 4: Performance of the multimodal HAR methods on UTD-MHAD dataset.
 SS:Subject-Specific, SG:Subject-Generic 
 
 
 Modality | 
 Method | 
 UTD-MHAD | 
 Year | 

 
 
 
 RGB 
 and 
 Inertial Sensors 
 | 
 Wei et al. [ 149 ] | 
 95.6%(SG) | 
 2019 | 

 
 Wei et al. [ 150 ] | 
 
 
 91.3%-smart TV gestures, 
 
85.2%-sports action(SG) 
 | 
 2020 | 

 
 
 
 RGB-D 
 and 
 Inertial Sensors 
 | 
 Dawar et al. [ 29 ] | 
 86.3%(SS) | 
 2018 | 

 
 Dawar et al. [ 30 ] | 
 89.20%(SG) | 
 2018 | 

 
 Dawar et al. [ 28 ] | 
 92.80% | 
 2018 | 

 
 Fuad et al. [ 45 ] | 
 95.00%(SG) | 
 2018 | 

 
 Ahmad et al. [ 1 ] | 
 98.70%(SS) | 
 2018 | 

 
 Javed et al. [ 38 ] | 
 98.30%(SG) | 
 2019 | 

 
 Imran et al. [ 60 ] | 
 97.90% | 
 2020 | 

 
 MGAF [ 3 ] | 
 96.80%(SS) | 
 2020 | 

 
 ( ​ M 2 \emph{M}^{2} ) fusion [ 2 ] | 
 99.20%(SS) | 
 2020 | 

 
 
 As for EAR field, some of the more representative methods, which are employed to recognize action from the first view, are compared on the egocentric datasets in Table 5 . The Acc in Table 5 are referred to as mean class accuracy, where the accuracy of each category is calculated separately, and the results of all classes were averaged.
 

 
 
 Table 5: Performance of the first-person action recognition methods on egocentric datasets.
 Acc:mean class accuracy 
 
 
 Dataset | 
 Reference | 
 Acc | 
 Year | 

 
 Dogcentric | 
 Pooled Motion [ 113 ] | 
 73.00% | 
 2015 | 

 
 Kwon et al. [ 71 ] | 
 83.00% | 
 2018 | 

 
 ADL | 
 Fathi et al. [ 41 ] | 
 49.80% | 
 2011 | 

 
 Pirsiavash et al. [ 106 ] | 
 36.70% | 
 2012 | 

 
 McCandless et al. [ 98 ] | 
 38.70% | 
 2013 | 

 
 Ego-ConvNet [ 120 ] | 
 37.58% | 
 2016 | 

 
 DCNN [ 164 ] | 
 55.20% | 
 2016 | 

 
 GTEA gaze+ | 
 Fathi et al. [ 39 ] | 
 47.70% | 
 2011 | 

 
 Li et al. [ 85 ] | 
 60.50% | 
 2015 | 

 
 Ma et al. [ 96 ] | 
 66.40% | 
 2016 | 

 
 Ego-ConvNet [ 120 ] | 
 68.50% | 
 2016 | 

 
 Ego-RNN [ 127 ] | 
 60.13% | 
 2018 | 

 
 EGTEA gaze+ | 
 Li et al. [ 85 ] | 
 46.50% | 
 2015 | 

 
 Ego-ConvNet [ 120 ] | 
 54.19% | 
 2016 | 

 
 Ego-RNN [ 127 ] | 
 52.40% | 
 2018 | 

 
 Li et al. [ 84 ] | 
 53.30% | 
 2018 | 

 
 Spatio-temporal Attention [ 95 ] | 
 60.54% | 
 2019 | 

 
 LSTA [ 125 ] | 
 53.00% | 
 2019 | 

 
 Lu et al. [ 94 ] | 
 46.84% | 
 2019 | 

 
 MCN [ 57 ] | 
 55.63% | 
 2020 | 

 
 Li et al. [ 83 ] | 
 57.20% | 
 2021 | 

 
 APL [ 146 ] | 
 60.15% | 
 2021 | 

 
 
 
 

## 5 Conclusion

 
 This paper firstly sorts out behavior recognition methods from the third-person view. For single modality, we introduce methods based on RGB video and 3D skeleton sequence respectively, which are the two most mainstream methods for single modal behavior recognition. In regard to multimodal input, the paper focuses on the multimodal fusion of visual sensors and non-visual sensors, including the multimodal fusion between RGB and audio modality, RGB and inertial sensors modality, RGB-D and inertial sensors modality. Given that the paper aims to provide a more comprehensive introduction to HAR for novices and researchers, thus we also investigate the action recognition methods from the first perspective in recent years. With respect to single modality, it mainly includes the traditional methods based on egocentric cue and the recognition methods based on deep learning. As for multimodal input, there are still relatively few studies, and paper briefly surveys the multimodal fusion between RGB and depth modality, RGB and audio modality, RGB and wearable inertial sensors modality. In addition, we also introduce the third-person view and egocentric datasets, and summarize the performance of the related methods on the corresponding datasets.

 
 
 

## References

 
 
 [1] 
 
Zeeshan Ahmad and Naimul Khan.

 
 Towards improved human action recognition using convolutional neural
networks and multimodal fusion of depth and inertial sensor data.

 
 In 2018 IEEE International Symposium on Multimedia (ISM) , pages
223–230. IEEE, 2018.

 

 
 [2] 
 
Zeeshan Ahmad and Naimul Khan.

 
 Human action recognition using deep multilevel multimodal
( ​ M 2 \emph{M}^{2} ) fusion of depth and inertial sensors.

 
 IEEE Sensors Journal , 20(3):1445–1455, 2019.

 

 
 [3] 
 
Zeeshan Ahmad and Naimul Khan.

 
 Cnn-based multistage gated average fusion (mgaf) for human action
recognition using depth and inertial sensors.

 
 IEEE Sensors Journal , 21(3):3623–3634, 2020.

 

 
 [4] 
 
Mehmet Ali Arabacı, Fatih Özkan, Elif Surer, Peter Jančovič,
and Alptekin Temizel.

 
 Multi-modal egocentric activity recognition using multi-kernel
learning.

 
 Multimedia Tools and Applications , 80(11):16299–16328, 2021.

 

 
 [5] 
 
Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić,
and Cordelia Schmid.

 
 Vivit: A video vision transformer.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 6836–6846, 2021.

 

 
 [6] 
 
Nicolas Ballas, Li Yao, Chris Pal, and Aaron Courville.

 
 Delving deeper into convolutional networks for learning video
representations.

 
 arXiv preprint arXiv:1511.06432 , 2015.

 

 
 [7] 
 
Gedas Bertasius, Heng Wang, and Lorenzo Torresani.

 
 Is space-time attention all you need for video understanding?

 
 In ICML , volume 2, page 4, 2021.

 

 
 [8] 
 
Aaron F. Bobick and James W. Davis.

 
 The recognition of human movement using temporal templates.

 
 IEEE Transactions on pattern analysis and machine intelligence ,
23(3):257–267, 2001.

 

 
 [9] 
 
Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles.

 
 Activitynet: A large-scale video benchmark for human activity
understanding.

 
 In Proceedings of the ieee conference on computer vision and
pattern recognition , pages 961–970, 2015.

 

 
 [10] 
 
Carlos Caetano, François Brémond, and William Robson Schwartz.

 
 Skeleton image representation for 3d action recognition based on tree
structure and reference joints.

 
 In 2019 32nd SIBGRAPI conference on graphics, patterns and
images (SIBGRAPI) , pages 16–23. IEEE, 2019.

 

 
 [11] 
 
Carlos Caetano, Jessica Sena, François Brémond, Jefersson A
Dos Santos, and William Robson Schwartz.

 
 Skelemotion: A new representation of skeleton joint sequences based
on motion information for 3d action recognition.

 
 In 2019 16th IEEE international conference on advanced video and
signal based surveillance (AVSS) , pages 1–8. IEEE, 2019.

 

 
 [12] 
 
Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh.

 
 Realtime multi-person 2d pose estimation using part affinity fields.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 7291–7299, 2017.

 

 
 [13] 
 
Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew
Zisserman.

 
 A short note about kinetics-600.

 
 arXiv preprint arXiv:1808.01340 , 2018.

 

 
 [14] 
 
Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman.

 
 A short note on the kinetics-700 human action dataset.

 
 arXiv preprint arXiv:1907.06987 , 2019.

 

 
 [15] 
 
Joao Carreira and Andrew Zisserman.

 
 Quo vadis, action recognition? a new model and the kinetics dataset.

 
 In proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 6299–6308, 2017.

 

 
 [16] 
 
Alejandro Cartas, Jordi Luque, Petia Radeva, Carlos Segura, and Mariella
Dimiccoli.

 
 Seeing and hearing egocentric actions: How much can we learn?

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision Workshops , pages 0–0, 2019.

 

 
 [17] 
 
Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz.

 
 Improving human action recognition using fusion of depth camera and
inertial sensors.

 
 IEEE Transactions on Human-Machine Systems , 45(1):51–61, 2014.

 

 
 [18] 
 
Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz.

 
 Utd-mhad: A multimodal dataset for human action recognition utilizing
a depth camera and a wearable inertial sensor.

 
 In 2015 IEEE International conference on image processing
(ICIP) , pages 168–172. IEEE, 2015.

 

 
 [19] 
 
Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz.

 
 A survey of depth and inertial sensor fusion for human action
recognition.

 
 Multimedia Tools and Applications , 76(3):4405–4425, 2017.

 

 
 [20] 
 
Chun-Fu Chen, Rameswar Panda, and Quanfu Fan.

 
 Regionvit: Regional-to-local attention for vision transformers.

 
 arXiv preprint arXiv:2106.02689 , 2021.

 

 
 [21] 
 
Jiawei Chen and Chiu Man Ho.

 
 Mm-vit: Multi-modal video transformer for compressed video action
recognition.

 
 In Proceedings of the IEEE/CVF Winter Conference on Applications
of Computer Vision , pages 1910–1921, 2022.

 

 
 [22] 
 
Ke Cheng, Yifan Zhang, Xiangyu He, Weihan Chen, Jian Cheng, and Hanqing Lu.

 
 Skeleton-based action recognition with shift graph convolutional
network.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 183–192, 2020.

 

 
 [23] 
 
Hyung-gun Chi, Myoung Hoon Ha, Seunggeun Chi, Sang Wan Lee, Qixing Huang, and
Karthik Ramani.

 
 Infogcn: Representation learning for human skeleton-based action
recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 20186–20196, 2022.

 

 
 [24] 
 
Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau,
Fethi Bougares, Holger Schwenk, and Yoshua Bengio.

 
 Learning phrase representations using rnn encoder-decoder for
statistical machine translation.

 
 arXiv preprint arXiv:1406.1078 , 2014.

 

 
 [25] 
 
Xiao Chu, Wei Yang, Wanli Ouyang, Cheng Ma, Alan L Yuille, and Xiaogang Wang.

 
 Multi-context attention for human pose estimation.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 1831–1840, 2017.

 

 
 [26] 
 
Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino
Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett,
Will Price, et al.

 
 Scaling egocentric vision: The epic-kitchens dataset.

 
 In Proceedings of the European Conference on Computer Vision
(ECCV) , pages 720–736, 2018.

 

 
 [27] 
 
Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari,
Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett,
Will Price, et al.

 
 Rescaling egocentric vision: collection, pipeline and challenges for
epic-kitchens-100.

 
 International Journal of Computer Vision , 130(1):33–55, 2022.

 

 
 [28] 
 
Neha Dawar and Nasser Kehtarnavaz.

 
 Action detection and recognition in continuous action streams by deep
learning-based sensing fusion.

 
 IEEE Sensors Journal , 18(23):9660–9668, 2018.

 

 
 [29] 
 
Neha Dawar and Nasser Kehtarnavaz.

 
 Real-time continuous detection and recognition of subject-specific
smart tv gestures via fusion of depth and inertial sensing.

 
 IEEE Access , 6:7019–7028, 2018.

 

 
 [30] 
 
Neha Dawar, Sarah Ostadabbas, and Nasser Kehtarnavaz.

 
 Data augmentation in deep learning-based fusion of depth and inertial
sensing for action recognition.

 
 IEEE Sensors Letters , 3(1):1–4, 2018.

 

 
 [31] 
 
Ali Diba, Mohsen Fayyaz, Vivek Sharma, Amir Hossein Karami, Mohammad Mahdi
Arzani, Rahman Yousefzadeh, and Luc Van Gool.

 
 Temporal 3d convnets: New architecture and transfer learning for
video classification.

 
 arXiv preprint arXiv:1711.08200 , 2017.

 

 
 [32] 
 
Alexander Diete and Heiner Stuckenschmidt.

 
 Fusing object information and inertial data for activity recognition.

 
 Sensors , 19(19):4119, 2019.

 

 
 [33] 
 
Alexander Diete, Timo Sztyler, and Heiner Stuckenschmidt.

 
 Vision and acceleration modalities: Partners for recognizing complex
activities.

 
 In 2019 IEEE International Conference on Pervasive Computing and
Communications Workshops (PerCom Workshops) , pages 101–106. IEEE, 2019.

 

 
 [34] 
 
Alexander Diete, Timo Sztyler, Lydia Weiland, and Heiner Stuckenschmidt.

 
 Improving motion-based activity recognition with ego-centric vision.

 
 In 2018 IEEE International Conference on Pervasive Computing and
Communications Workshops (PerCom Workshops) , pages 488–491. IEEE, 2018.

 

 
 [35] 
 
Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach,
Subhashini Venugopalan, Kate Saenko, and Trevor Darrell.

 
 Long-term recurrent convolutional networks for visual recognition and
description.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 2625–2634, 2015.

 

 
 [36] 
 
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, et al.

 
 An image is worth 16x16 words: Transformers for image recognition at
scale.

 
 arXiv preprint arXiv:2010.11929 , 2020.

 

 
 [37] 
 
Debidatta Dwibedi, Pierre Sermanet, and Jonathan Tompson.

 
 Temporal reasoning in videos using convolutional gated recurrent
units.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition Workshops , pages 1111–1116, 2018.

 

 
 [38] 
 
Muhammad Ehatisham-Ul-Haq, Ali Javed, Muhammad Awais Azam, Hafiz MA Malik, Aun
Irtaza, Ik Hyun Lee, and Muhammad Tariq Mahmood.

 
 Robust human activity recognition using multimodal feature-level
fusion.

 
 IEEE Access , 7:60736–60751, 2019.

 

 
 [39] 
 
Alireza Fathi, Ali Farhadi, and James M Rehg.

 
 Understanding egocentric activities.

 
 In 2011 international conference on computer vision , pages
407–414. IEEE, 2011.

 

 
 [40] 
 
Alireza Fathi, Yin Li, and James M Rehg.

 
 Learning to recognize daily actions using gaze.

 
 In European Conference on Computer Vision , pages 314–327.
Springer, 2012.

 

 
 [41] 
 
Alireza Fathi, Xiaofeng Ren, and James M Rehg.

 
 Learning to recognize objects in egocentric activities.

 
 In CVPR 2011 , pages 3281–3288. IEEE, 2011.

 

 
 [42] 
 
Christoph Feichtenhofer.

 
 X3d: Expanding architectures for efficient video recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 203–213, 2020.

 

 
 [43] 
 
Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.

 
 Slowfast networks for video recognition.

 
 In Proceedings of the IEEE/CVF international conference on
computer vision , pages 6202–6211, 2019.

 

 
 [44] 
 
Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman.

 
 Convolutional two-stream network fusion for video action recognition.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 1933–1941, 2016.

 

 
 [45] 
 
Zain Fuad and Mustafa Unel.

 
 Human action recognition using fusion of depth and inertial sensors.

 
 In International Conference Image Analysis and Recognition ,
pages 373–380. Springer, 2018.

 

 
 [46] 
 
Antonino Furnari and Giovanni Maria Farinella.

 
 Rolling-unrolling lstms for action anticipation from first-person
video.

 
 IEEE transactions on pattern analysis and machine intelligence ,
43(11):4021–4036, 2020.

 

 
 [47] 
 
Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim.

 
 First-person hand action benchmark with rgb-d videos and 3d hand pose
annotations.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 409–419, 2018.

 

 
 [48] 
 
Bernard Ghanem, Juan Carlos Niebles, Cees Snoek, Fabian Caba Heilbron, Humam
Alwassel, Victor Escorcia, Ranjay Krishna, Shyamal Buch, and Cuong Duc Dao.

 
 The activitynet large-scale activity recognition challenge 2018
summary.

 
 arXiv preprint arXiv:1808.03766 , 2018.

 

 
 [49] 
 
Bernard Ghanem, Juan Carlos Niebles, Cees Snoek, Fabian Caba Heilbron, Humam
Alwassel, Ranjay Khrisna, Victor Escorcia, Kenji Hata, and Shyamal Buch.

 
 Activitynet challenge 2017 summary.

 
 arXiv preprint arXiv:1710.08011 , 2017.

 

 
 [50] 
 
Rohit Girdhar and Deva Ramanan.

 
 Attentional pooling for action recognition.

 
 Advances in neural information processing systems , 30, 2017.

 

 
 [51] 
 
Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska,
Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos,
Moritz Mueller-Freitag, et al.

 
 The" something something" video database for learning and evaluating
visual common sense.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 5842–5850, 2017.

 

 
 [52] 
 
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh.

 
 Can spatiotemporal 3d cnns retrace the history of 2d cnns and
imagenet?

 
 In Proceedings of the IEEE conference on Computer Vision and
Pattern Recognition , pages 6546–6555, 2018.

 

 
 [53] 
 
Jun-Yan He, Xiao Wu, Zhi-Qi Cheng, Zhaoquan Yuan, and Yu-Gang Jiang.

 
 Db-lstm: Densely-connected bi-directional lstm for human action
recognition.

 
 Neurocomputing , 444:319–331, 2021.

 

 
 [54] 
 
Geoffrey E Hinton and Ruslan R Salakhutdinov.

 
 Reducing the dimensionality of data with neural networks.

 
 science , 313(5786):504–507, 2006.

 

 
 [55] 
 
Sepp Hochreiter and Jürgen Schmidhuber.

 
 Long short-term memory.

 
 Neural computation , 9(8):1735–1780, 1997.

 

 
 [56] 
 
Yi Huang, Xiaoshan Yang, Junyu Gao, Jitao Sang, and Changsheng Xu.

 
 Knowledge-driven egocentric multimodal activity recognition.

 
 ACM Transactions on Multimedia Computing, Communications, and
Applications (TOMM) , 16(4):1–133, 2020.

 

 
 [57] 
 
Yifei Huang, Minjie Cai, Zhenqiang Li, Feng Lu, and Yoichi Sato.

 
 Mutual context network for jointly estimating egocentric gaze and
action.

 
 IEEE Transactions on Image Processing , 29:7795–7806, 2020.

 

 
 [58] 
 
Mohamed E Hussein, Marwan Torki, Mohammad A Gowayyed, and Motaz El-Saban.

 
 Human action recognition using a temporal hierarchy of covariance
descriptors on 3d joint locations.

 
 In Twenty-third international joint conference on artificial
intelligence , 2013.

 

 
 [59] 
 
Momal Ijaz, Renato Diaz, and Chen Chen.

 
 Multimodal transformer for nursing activity recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 2065–2074, 2022.

 

 
 [60] 
 
Javed Imran and Balasubramanian Raman.

 
 Evaluating fusion of rgb-d and inertial sensors for multimodal human
action recognition.

 
 Journal of Ambient Intelligence and Humanized Computing ,
11(1):189–208, 2020.

 

 
 [61] 
 
Yumi Iwashita, Asamichi Takamine, Ryo Kurazume, and Michael S Ryoo.

 
 First-person animal activity recognition from egocentric videos.

 
 In 2014 22nd International Conference on Pattern Recognition ,
pages 4310–4315. IEEE, 2014.

 

 
 [62] 
 
Wenchao Jiang and Zhaozheng Yin.

 
 Human activity recognition using wearable sensors by deep
convolutional neural networks.

 
 In Proceedings of the 23rd ACM international conference on
Multimedia , pages 1307–1310, 2015.

 

 
 [63] 
 
Zhuolin Jiang, Viktor Rozgic, and Sancar Adali.

 
 Learning spatiotemporal features for infrared action recognition with
3d convolutional neural networks.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition workshops , pages 115–123, 2017.

 

 
 [64] 
 
Samira Ebrahimi Kahou, Christopher Pal, Xavier Bouthillier, Pierre Froumenty,
Çaglar Gülçehre, Roland Memisevic, Pascal Vincent, Aaron
Courville, Yoshua Bengio, Raul Chandias Ferrari, et al.

 
 Combining modality specific deep neural networks for emotion
recognition in video.

 
 In Proceedings of the 15th ACM on International conference on
multimodal interaction , pages 543–550, 2013.

 

 
 [65] 
 
Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul
Sukthankar, and Li Fei-Fei.

 
 Large-scale video classification with convolutional neural networks.

 
 In Proceedings of the IEEE conference on Computer Vision and
Pattern Recognition , pages 1725–1732, 2014.

 

 
 [66] 
 
Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra
Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al.

 
 The kinetics human action video dataset.

 
 arXiv preprint arXiv:1705.06950 , 2017.

 

 
 [67] 
 
Evangelos Kazakos, Jaesung Huh, Arsha Nagrani, Andrew Zisserman, and Dima
Damen.

 
 With a little help from my temporal context: Multimodal egocentric
action recognition.

 
 arXiv preprint arXiv:2111.01024 , 2021.

 

 
 [68] 
 
Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, and Dima Damen.

 
 Epic-fusion: Audio-visual temporal binding for egocentric action
recognition.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 5492–5501, 2019.

 

 
 [69] 
 
Pil-Soo Kim, Dong-Gyu Lee, and Seong-Whan Lee.

 
 Discriminative context learning with gated recurrent unit for group
activity recognition.

 
 Pattern Recognition , 76:149–161, 2018.

 

 
 [70] 
 
Hildegard Kuehne, Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio, and
Thomas Serre.

 
 Hmdb: a large video database for human motion recognition.

 
 In 2011 International conference on computer vision , pages
2556–2563. IEEE, 2011.

 

 
 [71] 
 
Heeseung Kwon, Yeonho Kim, Jin S Lee, and Minsu Cho.

 
 First person action recognition via two-stream convnet with long-term
fusion pooling.

 
 Pattern Recognition Letters , 112:161–167, 2018.

 

 
 [72] 
 
Ivan Laptev.

 
 On space-time interest points.

 
 International journal of computer vision , 64(2):107–123, 2005.

 

 
 [73] 
 
Inwoong Lee, Doyoung Kim, Seoungyoon Kang, and Sanghoon Lee.

 
 Ensemble deep learning for skeleton-based action recognition using
temporal sliding lstm networks.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 1012–1020, 2017.

 

 
 [74] 
 
Ce Li, Chunyu Xie, Baochang Zhang, Jungong Han, Xiantong Zhen, and Jie Chen.

 
 Memory attention networks for skeleton-based action recognition.

 
 IEEE Transactions on Neural Networks and Learning Systems ,
2021.

 

 
 [75] 
 
Chao Li, Qiaoyong Zhong, Di Xie, and Shiliang Pu.

 
 Co-occurrence feature learning from skeleton data for action
recognition and detection with hierarchical aggregation.

 
 arXiv preprint arXiv:1804.06055 , 2018.

 

 
 [76] 
 
Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yanfeng Wang, and Qi Tian.

 
 Actional-structural graph convolutional networks for skeleton-based
action recognition.

 
 In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , pages 3595–3603, 2019.

 

 
 [77] 
 
Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yanfeng Wang, and Qi Tian.

 
 Symbiotic graph neural networks for 3d skeleton-based human action
recognition and motion prediction.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
44(6):3316–3333, 2021.

 

 
 [78] 
 
Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao.

 
 Independently recurrent neural network (indrnn): Building a longer
and deeper rnn.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 5457–5466, 2018.

 

 
 [79] 
 
Tianjiao Li, Qiuhong Ke, Hossein Rahmani, Rui En Ho, Henghui Ding, and Jun Liu.

 
 Else-net: Elastic semantic network for continual action recognition
from skeleton data.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 13434–13443, 2021.

 

 
 [80] 
 
Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, and Wanqing
Li.

 
 Trear: Transformer-based rgb-d egocentric action recognition.

 
 IEEE Transactions on Cognitive and Developmental Systems ,
14(1):246–252, 2021.

 

 
 [81] 
 
Yanshan Li, Rongjie Xia, Xing Liu, and Qinghua Huang.

 
 Learning shape-motion representations from geometric algebra
spatio-temporal model for skeleton-based action recognition.

 
 In 2019 IEEE International Conference on Multimedia and Expo
(ICME) , pages 1066–1071. IEEE, 2019.

 

 
 [82] 
 
Yin Li, Alireza Fathi, and James M Rehg.

 
 Learning to predict gaze in egocentric video.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 3216–3223, 2013.

 

 
 [83] 
 
Yin Li, Miao Liu, and Jame Rehg.

 
 In the eye of the beholder: Gaze and actions in first person video.

 
 IEEE transactions on pattern analysis and machine intelligence ,
2021.

 

 
 [84] 
 
Yin Li, Miao Liu, and James M Rehg.

 
 In the eye of beholder: Joint learning of gaze and actions in first
person video.

 
 In Proceedings of the European conference on computer vision
(ECCV) , pages 619–635, 2018.

 

 
 [85] 
 
Yin Li, Zhefan Ye, and James M Rehg.

 
 Delving into egocentric actions.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 287–295, 2015.

 

 
 [86] 
 
Ying Li, Qiang Zhai, Sihao Ding, Fan Yang, Gang Li, and Yuan F Zheng.

 
 Efficient health-related abnormal behavior detection with visual and
inertial sensor integration.

 
 Pattern Analysis and Applications , 22(2):601–614, 2019.

 

 
 [87] 
 
Zhenyang Li, Kirill Gavrilyuk, Efstratios Gavves, Mihir Jain, and Cees GM
Snoek.

 
 Videolstm convolves, attends and flows for action recognition.

 
 Computer Vision and Image Understanding , 166:41–50, 2018.

 

 
 [88] 
 
Ji Lin, Chuang Gan, and Song Han.

 
 Tsm: Temporal shift module for efficient video understanding.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 7083–7093, 2019.

 

 
 [89] 
 
Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C
Kot.

 
 Ntu rgb+ d 120: A large-scale benchmark for 3d human activity
understanding.

 
 IEEE transactions on pattern analysis and machine intelligence ,
42(10):2684–2701, 2019.

 

 
 [90] 
 
Jun Liu, Gang Wang, Ping Hu, Ling-Yu Duan, and Alex C Kot.

 
 Global context-aware attention lstm networks for 3d action
recognition.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 1647–1656, 2017.

 

 
 [91] 
 
Kui Liu, Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz.

 
 Fusion of inertial and depth sensor data for robust hand gesture
recognition.

 
 IEEE Sensors Journal , 14(6):1898–1903, 2014.

 

 
 [92] 
 
Xiang Long, Chuang Gan, Gerard De Melo, Jiajun Wu, Xiao Liu, and Shilei Wen.

 
 Attention clusters: Purely attention based local feature integration
for video classification.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 7834–7843, 2018.

 

 
 [93] 
 
Xiang Long, Chuang Gan, Gerard Melo, Xiao Liu, Yandong Li, Fu Li, and Shilei
Wen.

 
 Multimodal keyless attention fusion for video classification.

 
 In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 32, 2018.

 

 
 [94] 
 
Minlong Lu, Ze-Nian Li, Yueming Wang, and Gang Pan.

 
 Deep attention network for egocentric action recognition.

 
 IEEE Transactions on Image Processing , 28(8):3703–3713, 2019.

 

 
 [95] 
 
Minlong Lu, Danping Liao, and Ze-Nian Li.

 
 Learning spatiotemporal attention for egocentric action recognition.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision Workshops , pages 0–0, 2019.

 

 
 [96] 
 
Minghuang Ma, Haoqi Fan, and Kris M Kitani.

 
 Going deeper into first-person activity recognition.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 1894–1903, 2016.

 

 
 [97] 
 
Alessandro Manzi, Alessandra Moschetti, Raffaele Limosani, Laura Fiorini, and
Filippo Cavallo.

 
 Enhancing activity recognition of self-localized robot through depth
camera and wearable sensors.

 
 IEEE Sensors Journal , 18(22):9324–9331, 2018.

 

 
 [98] 
 
Tomas McCandless and Kristen Grauman.

 
 Object-centric spatio-temporal pyramids for egocentric activity
recognition.

 
 In BMVC , volume 2, page 3, 2013.

 

 
 [99] 
 
Lili Meng, Bo Zhao, Bo Chang, Gao Huang, Wei Sun, Frederick Tung, and Leonid
Sigal.

 
 Interpretable spatio-temporal attention for video action recognition.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision Workshops , pages 0–0, 2019.

 

 
 [100] 
 
Kyle Min and Jason J Corso.

 
 Integrating human gaze into attention for egocentric activity
recognition.

 
 In Proceedings of the IEEE/CVF Winter Conference on Applications
of Computer Vision , pages 1069–1078, 2021.

 

 
 [101] 
 
Robin R Murphy.

 
 Computer vision and machine learning in science fiction.

 
 Science Robotics , 4(30):eaax7421, 2019.

 

 
 [102] 
 
Daniel Neimark, Omri Bar, Maya Zohar, and Dotan Asselmann.

 
 Video transformer network.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 3163–3172, 2021.

 

 
 [103] 
 
Koray Ozcan and Senem Velipasalar.

 
 Wearable camera-and accelerometer-based fall detection on portable
devices.

 
 IEEE Embedded Systems Letters , 8(1):6–9, 2015.

 

 
 [104] 
 
Yuxin Peng, Yunzhen Zhao, and Junchao Zhang.

 
 Two-stream collaborative learning with spatial-temporal attention for
video classification.

 
 IEEE Transactions on Circuits and Systems for Video Technology ,
29(3):773–786, 2018.

 

 
 [105] 
 
Juan-Manuel Perez-Rua, Brais Martinez, Xiatian Zhu, Antoine Toisoul, Victor
Escorcia, and Tao Xiang.

 
 Knowing what, where and when to look: Efficient video action modeling
with attention.

 
 arXiv preprint arXiv:2004.01278 , 2020.

 

 
 [106] 
 
Hamed Pirsiavash and Deva Ramanan.

 
 Detecting activities of daily living in first-person camera views.

 
 In 2012 IEEE conference on computer vision and pattern
recognition , pages 2847–2854. IEEE, 2012.

 

 
 [107] 
 
Mirco Planamente, Chiara Plizzari, Emanuele Alberti, and Barbara Caputo.

 
 Domain generalization through audio-visual relative norm alignment in
first person action recognition.

 
 In Proceedings of the IEEE/CVF Winter Conference on Applications
of Computer Vision , pages 1807–1818, 2022.

 

 
 [108] 
 
Chiara Plizzari, Mirco Planamente, Gabriele Goletto, Marco Cannici, Emanuele
Gusso, Matteo Matteucci, and Barbara Caputo.

 
 E2 (go) motion: Motion augmented event stream for egocentric action
recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 19935–19947, 2022.

 

 
 [109] 
 
Yair Poleg, Chetan Arora, and Shmuel Peleg.

 
 Temporal segmentation of egocentric videos.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 2537–2544, 2014.

 

 
 [110] 
 
Kishore K Reddy and Mubarak Shah.

 
 Recognizing 50 human action categories of web videos.

 
 Machine vision and applications , 24(5):971–981, 2013.

 

 
 [111] 
 
James Rwigema, Hyo-Rim Choi, and TaeYong Kim.

 
 A differential evolution approach to optimize weights of dynamic time
warping for multi-sensor based gesture recognition.

 
 Sensors , 19(5):1007, 2019.

 

 
 [112] 
 
Michael S Ryoo and Larry Matthies.

 
 First-person activity recognition: What are they doing to me?

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 2730–2737, 2013.

 

 
 [113] 
 
Michael S Ryoo, Brandon Rothrock, and Larry Matthies.

 
 Pooled motion features for first-person videos.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 896–904, 2015.

 

 
 [114] 
 
Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang.

 
 Ntu rgb+ d: A large scale dataset for 3d human activity analysis.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 1010–1019, 2016.

 

 
 [115] 
 
Amir Shahroudy, Tian-Tsong Ng, Yihong Gong, and Gang Wang.

 
 Deep multimodal feature analysis for action recognition in rgb+ d
videos.

 
 IEEE transactions on pattern analysis and machine intelligence ,
40(5):1045–1058, 2017.

 

 
 [116] 
 
Dandan Shan, Jiaqi Geng, Michelle Shu, and David F Fouhey.

 
 Understanding human hands in contact at internet scale.

 
 In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , pages 9869–9878, 2020.

 

 
 [117] 
 
Yemin Shi, Yonghong Tian, Yaowei Wang, Wei Zeng, and Tiejun Huang.

 
 Learning long-term dependencies for action recognition with a
biologically-inspired deep network.

 
 In Proceedings of the IEEE International Conference on Computer
Vision , pages 716–725, 2017.

 

 
 [118] 
 
Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek
Alahari.

 
 Actor and observer: Joint modeling of first and third-person videos.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 7396–7404, 2018.

 

 
 [119] 
 
Karen Simonyan and Andrew Zisserman.

 
 Two-stream convolutional networks for action recognition in videos.

 
 Advances in neural information processing systems , 27, 2014.

 

 
 [120] 
 
Suriya Singh, Chetan Arora, and CV Jawahar.

 
 First person action recognition using deep learned descriptors.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 2620–2628, 2016.

 

 
 [121] 
 
Sibo Song, Vijay Chandrasekhar, Bappaditya Mandal, Liyuan Li, Joo-Hwee Lim,
Giduthuri Sateesh Babu, Phyo Phyo San, and Ngai-Man Cheung.

 
 Multimodal multi-stream deep learning for egocentric activity
recognition.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition workshops , pages 24–31, 2016.

 

 
 [122] 
 
Sibo Song, Ngai-Man Cheung, Vijay Chandrasekhar, Bappaditya Mandal, and Jie
Liri.

 
 Egocentric activity recognition with multimodal fisher vector.

 
 In 2016 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP) , pages 2717–2721. IEEE, 2016.

 

 
 [123] 
 
Yi-Fan Song, Zhang Zhang, Caifeng Shan, and Liang Wang.

 
 Constructing stronger and faster baselines for skeleton-based action
recognition.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
2022.

 

 
 [124] 
 
Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah.

 
 Ucf101: A dataset of 101 human actions classes from videos in the
wild.

 
 arXiv preprint arXiv:1212.0402 , 2012.

 

 
 [125] 
 
Swathikiran Sudhakaran, Sergio Escalera, and Oswald Lanz.

 
 Lsta: Long short-term attention for egocentric action recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 9954–9963, 2019.

 

 
 [126] 
 
Swathikiran Sudhakaran and Oswald Lanz.

 
 Convolutional long short-term memory networks for recognizing first
person interactions.

 
 In Proceedings of the IEEE International Conference on Computer
Vision Workshops , pages 2339–2346, 2017.

 

 
 [127] 
 
Swathikiran Sudhakaran and Oswald Lanz.

 
 Attention is all we need: Nailing down object-centric attention for
egocentric activity recognition.

 
 arXiv preprint arXiv:1807.11794 , 2018.

 

 
 [128] 
 
Lin Sun, Kui Jia, Kevin Chen, Dit-Yan Yeung, Bertram E Shi, and Silvio
Savarese.

 
 Lattice long short-term memory for human action recognition.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 2147–2156, 2017.

 

 
 [129] 
 
Yansong Tang, Zian Wang, Jiwen Lu, Jianjiang Feng, and Jie Zhou.

 
 Multi-stream deep neural networks for rgb-d egocentric action
recognition.

 
 IEEE Transactions on Circuits and Systems for Video Technology ,
29(10):3001–3015, 2018.

 

 
 [130] 
 
Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri.

 
 Learning spatiotemporal features with 3d convolutional networks.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 4489–4497, 2015.

 

 
 [131] 
 
Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar
Paluri.

 
 A closer look at spatiotemporal convolutions for action recognition.

 
 In Proceedings of the IEEE conference on Computer Vision and
Pattern Recognition , pages 6450–6459, 2018.

 

 
 [132] 
 
Nilay Tufek, Murat Yalcin, Mucahit Altintas, Fatma Kalaoglu, Yi Li, and
Senem Kursun Bahadir.

 
 Human action recognition using deep learning methods on limited
sensory data.

 
 IEEE Sensors Journal , 20(6):3101–3112, 2019.

 

 
 [133] 
 
Amin Ullah, Jamil Ahmad, Khan Muhammad, Muhammad Sajjad, and Sung Wook Baik.

 
 Action recognition in video sequences using deep bi-directional lstm
with cnn features.

 
 IEEE access , 6:1155–1166, 2017.

 

 
 [134] 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

 
 Attention is all you need.

 
 Advances in neural information processing systems , 30, 2017.

 

 
 [135] 
 
Raviteja Vemulapalli, Felipe Arrate, and Rama Chellappa.

 
 Human action recognition by representing 3d skeletons as points in a
lie group.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 588–595, 2014.

 

 
 [136] 
 
Cheng Wang, Haojin Yang, and Christoph Meinel.

 
 Exploring multimodal video representation for action recognition.

 
 In 2016 International Joint Conference on Neural Networks
(IJCNN) , pages 1924–1931. IEEE, 2016.

 

 
 [137] 
 
Fei Wang, Yunpeng Song, Jimuyang Zhang, Jinsong Han, and Dong Huang.

 
 Temporal unet: Sample level human action recognition using wifi.

 
 arXiv preprint arXiv:1904.11953 , 2019.

 

 
 [138] 
 
Heng Wang, A Klaser, C Schmid, and L Cheng-Lin.

 
 Action recognition by dense trajectories. computer vision and pattern
recognition (cvpr).

 
 In 2011 IEEE Conference on , pages 3169–3176, 2011.

 

 
 [139] 
 
Heng Wang and Cordelia Schmid.

 
 Action recognition with improved trajectories.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 3551–3558, 2013.

 

 
 [140] 
 
Hongsong Wang and Liang Wang.

 
 Modeling temporal dynamics and spatial configurations of actions
using two-stream recurrent neural networks.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 499–508, 2017.

 

 
 [141] 
 
Limin Wang, Yu Qiao, and Xiaoou Tang.

 
 Action recognition with trajectory-pooled deep-convolutional
descriptors.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 4305–4314, 2015.

 

 
 [142] 
 
Limin Wang, Yuanjun Xiong, Zhe Wang, and Yu Qiao.

 
 Towards good practices for very deep two-stream convnets.

 
 arXiv preprint arXiv:1507.02159 , 2015.

 

 
 [143] 
 
Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc
Van Gool.

 
 Temporal segment networks: Towards good practices for deep action
recognition.

 
 In European conference on computer vision , pages 20–36.
Springer, 2016.

 

 
 [144] 
 
Pichao Wang, Wanqing Li, Chuankun Li, and Yonghong Hou.

 
 Action recognition based on joint trajectory maps with convolutional
neural networks.

 
 Knowledge-Based Systems , 158:43–53, 2018.

 

 
 [145] 
 
Weiyao Wang, Du Tran, and Matt Feiszli.

 
 What makes training multi-modal classification networks hard?

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 12695–12705, 2020.

 

 
 [146] 
 
Xiaohan Wang, Linchao Zhu, Heng Wang, and Yi Yang.

 
 Interactive prototype learning for egocentric action recognition.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 8168–8177, 2021.

 

 
 [147] 
 
Yunbo Wang, Mingsheng Long, Jianmin Wang, and Philip S Yu.

 
 Spatiotemporal pyramid network for video action recognition.

 
 In Proceedings of the IEEE conference on Computer Vision and
Pattern Recognition , pages 1529–1538, 2017.

 

 
 [148] 
 
Haoran Wei, Pranav Chopada, and Nasser Kehtarnavaz.

 
 C-mhad: Continuous multimodal human action dataset of simultaneous
video and inertial sensing.

 
 Sensors , 20(10):2905, 2020.

 

 
 [149] 
 
Haoran Wei, Roozbeh Jafari, and Nasser Kehtarnavaz.

 
 Fusion of video and inertial sensing for deep learning–based human
action recognition.

 
 Sensors , 19(17):3680, 2019.

 

 
 [150] 
 
Haoran Wei and Nasser Kehtarnavaz.

 
 Simultaneous utilization of inertial and video sensing for action
detection and recognition in continuous action streams.

 
 IEEE Sensors Journal , 20(11):6055–6063, 2020.

 

 
 [151] 
 
Chao-Yuan Wu, Christoph Feichtenhofer, Haoqi Fan, Kaiming He, Philipp
Krahenbuhl, and Ross Girshick.

 
 Long-term feature banks for detailed video understanding.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 284–293, 2019.

 

 
 [152] 
 
Fanyi Xiao, Yong Jae Lee, Kristen Grauman, Jitendra Malik, and Christoph
Feichtenhofer.

 
 Audiovisual slowfast networks for video recognition.

 
 arXiv preprint arXiv:2001.08740 , 2020.

 

 
 [153] 
 
Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, and
Cordelia Schmid.

 
 Multiview transformers for video recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 3333–3343, 2022.

 

 
 [154] 
 
Sijie Yan, Yuanjun Xiong, and Dahua Lin.

 
 Spatial temporal graph convolutional networks for skeleton-based
action recognition.

 
 In Thirty-second AAAI conference on artificial intelligence ,
2018.

 

 
 [155] 
 
Jiewen Yang, Xingbo Dong, Liujun Liu, Chao Zhang, Jiajun Shen, and Dahai Yu.

 
 Recurring the transformer for video action recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 14063–14073, 2022.

 

 
 [156] 
 
Haibin Yu, Guoxiong Pan, Mian Pan, Chong Li, Wenyan Jia, Li Zhang, and Mingui
Sun.

 
 A hierarchical deep fusion framework for egocentric activity
recognition using a wearable hybrid sensor system.

 
 Sensors , 19(3):546, 2019.

 

 
 [157] 
 
Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vijayanarasimhan, Oriol Vinyals,
Rajat Monga, and George Toderici.

 
 Beyond short snippets: Deep networks for video classification.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 4694–4702, 2015.

 

 
 [158] 
 
Pengfei Zhang, Cuiling Lan, Junliang Xing, Wenjun Zeng, Jianru Xue, and Nanning
Zheng.

 
 View adaptive neural networks for high performance skeleton-based
human action recognition.

 
 IEEE transactions on pattern analysis and machine intelligence ,
41(8):1963–1978, 2019.

 

 
 [159] 
 
Pengfei Zhang, Cuiling Lan, Wenjun Zeng, Junliang Xing, Jianru Xue, and Nanning
Zheng.

 
 Semantics-guided neural networks for efficient skeleton-based human
action recognition.

 
 In proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , pages 1112–1121, 2020.

 

 
 [160] 
 
Xikun Zhang, Chang Xu, and Dacheng Tao.

 
 Context aware graph convolution for skeleton-based action
recognition.

 
 In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , pages 14333–14342, 2020.

 

 
 [161] 
 
Yunhua Zhang, Hazel Doughty, Ling Shao, and Cees GM Snoek.

 
 Audio-adaptive activity recognition across video domains.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 13791–13800, 2022.

 

 
 [162] 
 
Zhengyou Zhang.

 
 Microsoft kinect sensor and its effect.

 
 IEEE multimedia , 19(2):4–10, 2012.

 

 
 [163] 
 
Qiaoyun Zhou, Shiqi Yu, Xinyu Wu, Qiao Gao, Chongguo Li, and Yangsheng Xu.

 
 Hmms-based human action recognition for an intelligent household
surveillance robot.

 
 In 2009 IEEE International Conference on Robotics and
Biomimetics (ROBIO) , pages 2295–2300. IEEE, 2009.

 

 
 [164] 
 
Yang Zhou, Bingbing Ni, Richang Hong, Xiaokang Yang, and Qi Tian.

 
 Cascaded interactional targeting network for egocentric video
analysis.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 1904–1913, 2016.

 

 
 [165] 
 
Linchao Zhu, Du Tran, Laura Sevilla-Lara, Yi Yang, Matt Feiszli, and Heng Wang.

 
 Faster recurrent networks for efficient video classification.

 
 In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 34, pages 13098–13105, 2020.