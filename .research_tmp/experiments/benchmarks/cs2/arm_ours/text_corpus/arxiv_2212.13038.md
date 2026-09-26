A Survey of Face Recognition 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.13038v1 [cs.CV] 26 Dec 2022 
 
 

# A Survey of Face Recognition

 
 
 Xinyi Wang
 
 Affiliation: OPPO Research Institute, Beijing, China 
 
    
 Jianteng Peng
 
 Affiliation: OPPO Research Institute, Beijing, China 
 
    
 Sufang Zhang
 
 Affiliation: OPPO Research Institute, Beijing, China 
 
    
 Bihui Chen
 
 Affiliation: OPPO Research Institute, Beijing, China 
 
    
 Yi Wang
 
    
 Yandong Guo
 
 Affiliation: OPPO Research Institute, Beijing, China 
 

 Abstract 
 
 Recent years witnessed the breakthrough of face recognition with deep convolutional neural networks. Dozens of papers in the field of FR are published every year. Some of them were applied in the industrial community and played an important role in human life such as device unlock, mobile payment, and so on. This paper provides an introduction to face recognition, including its history, pipeline, algorithms based on conventional manually designed features or deep learning, mainstream training, evaluation datasets, and related applications. We have analyzed and compared state-of-the-art works as many as possible, and also carefully designed a set of experiments to find the effect of backbone size and data distribution. This survey is a material of the tutorial named The Practical Face Recognition Technology in the Industrial World in the FG2023. 

 
 
 
 Keywords: Face Recognition, Deep Learning, Industrial Application

 

 
 

## 1 Introduction

 
 As one of the most important applications in the field of artificial intelligence and computer vision, face recognition (FR) has attracted the wide attention of researchers. Almost every year, major CV conferences or journals, including CVPR, ICCV, PAMI, etc. publish dozens of papers in the field of FR. 

 
 
 In this work, we give an introduction to face recognition. Chapter 2 is about the history of face recognition. Chapter 3 introduces the pipeline in the deep learning framework. Chapter 4 provides the details of face recognition algorithms including loss functions, embedding techniques, face recognition with massive IDs, cross-domain, pipeline acceleration, closed-set training, mask face recognition, and privacy-preserving. In Chapter 5, we carefully designed a set of experiments to find the effect of backbone size and data distribution. Chapter 6 includes the frequently-used training and test datasets and comparison results. Chapter 7 shows the applications. Chapter 8 introduces the competitions and open-source programs. 

 
 
 

## 2 History

 
 Recent years witnessed the breakthrough of face recognition (FR) with deep Convolutional Neural Networks. But before 2014, FR was processed in non-deep learning ways.
In this section, we introduce conventional FR algorithms with manually designed features. 

 
 
 Eigenface [ 1 ] is the first method to solve the problem of FR in computer vision.
Eigenface reshaped a grayscale face image into a 1-D vector as its feature.
Then, a low-dimensional subspace can be found by Principal Component Analysis (PCA), where the distributions of features from similar faces were closer. This subspace can be used to represent all face samples.
For a query face image with an unknown ID, Eigenface adopted the Euclidean distance to measure the differences between its feature and all gallery face features with known categories in the new subspace.
The category of query image can be predicted by finding the smallest distance to all features in the gallery.
Fisherface [ 2 ] pointed out that PCA used in Eigenface for dimensionality reduction maximizes the variance of all samples in the new subspace.
Eigenface did not take sample category into account.
Therefore, Fisherface considered face labels in the training set and used Linear Discriminant Analysis (LDA) for dimensionality reduction, which maximized the ratios of inter-class and intra-class variance. 

 
 
 As a commonly used mathematical classification model, SVM was brought into the field of FR.
Since SVM and face feature extractors can be de-coupled, many SVM-based FR algorithms with different extractors have been proposed.
Deniz et al. [ 3 ] used ICA [ 4 ] for feature extraction, and then used SVM to predict face ID.
In order to speed up the training of ICA+SVM,
Kong et al. [ 5 ] designed fast Least Squares SVM to accelerate the training process by modifying the Least Squares SVM algorithm.
Jianhong et al. [ 6 ] combined kernel PCA and Least Squares SVM to get a better result. 

 
 
 As one of the most successful image local texture extractors, Local Binary Pattern (LBP) [ 7 ] obtained characteristics of images through the statistical histogram.
Many scholars have proposed FR algorithms based on the LBP feature extractor and its variants.
Ahonen et al. [ 8 ] adopted LBP to extract features of all regions from one face image, and then concatenated those feature vectors for histogram statistics to obtain an embedding of the whole image.
Finally, they used Chi square as the distance metric to measure the similarity of faces.
Wolf et al. [ 9 ] proposed an improved region descriptor based on the LBP, and got a better result with higher accuracy.
Tan et al. [ 10 ] refined the LBP and proposed the Local Ternary Patterns (LTP), which had a higher tolerance of image noise. 

 
 
 

## 3 Pipelines in deep learning framework

 
 As the computing resources were increasing, mainstream face recognition methods applied deep learning to model this problem, which replaced the aforementioned manually designed feature extraction methods. In this section, we give the pipelines of ’training’ a FR deep model and ’inference’ to get the ID of a face image (as shown in Fig. 1 ). And we will elaborate some representative methods related to them. 

 
 
 
 
 
 (a) Pipeline of training 
 
 
 (b) Pipeline of inference 
 
 Figure 1: Pipelines of training and inference in face recognition 
 
 
 In general, the pipeline of training FR model contains two steps: face images preprocessing (preprocessing) and model training (training). Getting the FR result by the trained model contains three parts: face images preprocessing (preprocessing), model inference to get face embedding (inference), and recognition by matching features between test image and images with known labels (recognition). In industry, face anti-spoofing will be added before inference in the testing pipeline since FR systems are vulnerable to presenatation attacks ranging from print, video replay, 3D mask, etc.
It needs to be mentioned that, both preprocessing steps in training and inference should be the same. 

 
 

### 3.1 Preprocessing

 
 In practice, faces for FR are located in a complicated image or a video. Therefore we need to apply face detection (with face landmark detection) to cut out the specific face patch first, and use face alignment to transform this face patch in a good angle or position. After these two preprocessing, a face image patch is ready for further FR procedures.
It is noticed that, face detection is necessary for FR preprocessing, while face alignment is not. 

 
 

#### 3.1.1 Face detection

 
 Face detection (or automatic face localisation) is a long-standing problem in CV.
Considering human face as a object, many object detection algorithms can be adopted to get a great face detection result, such as Faster-RCNN [ 11 ] , SSD [ 12 ] , YOLO (with different versions [ 13 , 14 , 15 ] ). 

 
 
 However, some researchers treated human face as a special object and designed variable deep architectures for finding faces in the wild with higher accuracy.
Mtcnn [ 16 ] is one of the most famous face detection methods in academy. Mtcnn adopted a cascaded structure with three stages of deep convolutional networks (P-net, R-net, O-net) that predict both face and landmark location in a coarse-to-fine manner.
Non-maximum suppression (NMS) is employed to merge highly overlapped candidates, which are produced by P-net, R-net and O-net.
Faceness-Net [ 17 ] observed that facial attributes based supervision can effectively enhance the capability of a face detection network in handling severe occlusions.
As a result, Faceness-Net proposed a two-stage network, where the first stage applies several branches to generate response maps of different facial parts and the second stage refines candidate window using a multi-task CNN. At last, face attribute prediction and bounding box regression are jointly optimized. 

 
 
 As FPN (Feature Pyramid Network) is widely used in object detection, many researchers tried to bring in FPN in face detection and obtained success in detecting tiny faces.
SRN (Selective Refinement Network) [ 18 ] adopted FPN to extract face features, and introduced two-step classification and regression operations selectively into an anchor-based face detector to reduce false positives and improve location accuracy simultaneously.
In particular, the SRN consists of two modules: the Selective Two-step Classification (STC) module and the Selective Two-step Regression (STR) module.
The STC aims to filter out most simple negative anchors from low level detection layers to reduce the search space for the subsequent classifier,
while the STR is designed to coarsely adjust the locations and sizes of anchors from high level detection layers to provide better initialization for the subsequent regressor.
The pipeline of SRN can be seen in Fig. 2 

 
 
 Figure 2: The Pipeline of SRN 
 
 
 RetinaFace [ 19 ] is another face detector using FPN to extract multi-level image features. However, different from two-step detecting methods, such as SRN, RetinaFace presented a single-stage face detector.
Further more, RetinaFace performed face detection in a multi-task way.
In specific, RetinaFace predicted following 4 aspects at the same time: (1) a face score, (2) a face box, (3) five facial landmarks, and (4) dense 3D face vertices projected on the image plane.
The pipeline of RetinaFace can be seen in Fig. 3 

 
 
 Figure 3: The Pipeline of RetinaFace 
 
 
 CRFace [ 20 ] proposed a confidence ranker to refine face detection result in high resolution images or videos.
HLA-Face [ 21 ] solved face detection problem in low light scenarios.
As the development of general 2D object detection, face detection has reached a high performance, even in real-world scenarios.
As a result, innovative papers in face detection seldom show up in recent years. 

 
 
 

#### 3.1.2 Face anti-spoofing

 
 In industry, faces fed into recognition system need to be check whether they are real or not, in case someone uses printed images, videos or 3D mask to pass through the system, especially in secure scenarios.
As a result, while inference, face anti-spoofing (or presentation attack detection) is an important prepossessing step, which is located after face detection. 

 
 
 The major methods for face anti-spoofing usually input signals from RGB cameras.
Both hand-crafted feature based (e.g., LBP [ 22 ] , SIFT [ 23 ] , SURF [ 24 ] and HoG [ 25 ] ) and deep learning based methods have been proposed. CDCN [ 26 ] designed a novel convolutional operator called Central Difference Convolution to get fine-grained invariant information in diverse diverse enviroments. The output feature map y y can be formulated as: 

 

 
 | 
 y ⁡ ( p 0 ) = ∑ p n ∈ R w ⁡ ( p n ) ⋅ x ⁡ ( p 0 + p n ) y(p_{0})=\sum_{p_{n}\in R}w(p_{n})\cdot x(p_{0}+p_{n}) | 
 | 
 (1) | 
 

 where p 0 p_{0} is the current location on both input and output feature map, and p n p_{n} is the location in R R . In addition, CDCN used NAS to search backbone for face anti-spoofing task, which is supervised by depth loss instead of binary loss. DCDN [ 27 ] decoupled the central gradient features into two cross directions (horizontal/vertical or diagonal) and achieved better performance with less computation cost when compared with CDCN. STDN [ 28 ] observed that there is little explanation of the classifier’s decision. As a result, STDN proposed an adversarial learning network to extract the patterns differentiating a spoof and live face. The pipeline of STDN can be seen in Fig 4 . PatchNet [ 29 ] rephrased face anti-spoofing as a fine-grained material recognition problem. Specifically, PatchNet splitted the categories finely based on the capturing devices and presenting materials, and used patch-level inputs to learn discriminative features. In addition, the asymmetric margin-based classification loss and self-supervised similarity loss were proposed to further improve the generalization ability of the spoof feature. SSAN [ 30 ] used a two-stream structure to extract content and style features, and reassembled various content and style features to get a stylized feature space, which was used to distinguish a spoof and live face. In specific, adversarial learning was used to get a shared feature distribution for content information and a contrastive learning strategy was proposed to enhance liveness-related style information while suppress domain-specific one. 

 
 
 Figure 4: The Pipeline of STDN 
 
 
 Merely using RGB signals which belong to visible spectrum has a lot of limitations in face anti-spoofing, therefore an increasing numbers of methods tried to adopt multi-channel signals to achieve higher accuracy. FaceBagNet [ 31 ] used patch-level image to extract the spoof-specific discriminative information and proposed a multi-modal features fusion strategy. CMFL [ 32 ] designed a cross-modal focal loss to modulate the individual channels’ loss contributions, thus captured complementary information among modalities. 

 
 
 

#### 3.1.3 Face alignment

 
 Face image patches from face detection are often different in shapes due to factors such as pose,
perspective transformation and so on, which will lead to a decline on recognition performance.
Under this condition, face alignment is an effective approach to alleviate this issue by transforming face patches into a similar angle and position [ 33 , 34 ] .
Face image set with an identical ID will get a smaller intra-class difference after face alignment, further making the training classifier more discriminative. 

 
 
 Common way to align face images is using a 2D transformation to calibrate facial landmarks to predefined frontal templates or mean face mode. This 2D transformation is normally processed by affine transformation [ 33 ] .
Deepface [ 35 ] proposed a FR pipeline which employed a 3D alignment method in order to align faces undergoing out-of-plane rotations.
Deepface used a generic 3D shape model and registered a 3D affine camera, which are used to warp the 2D aligned crop to the image plane of the 3D shape.
STN (Spatial Transform Networks) [ 36 ] introduced a learnable module which explicitly allowed the spatial manipulation of image within the network, and it can also be utilised in face alignment.
Wu et al. [ 37 ] merged STN based face alignment and face feature extractor together, and put forward Recursive Spatial Transformer (ReST) for end-to-end FR training with alignment.
2D face alignment is faster than both 3D alignment and STN, and thus it is widely used in preprocessing step of FR.
APA [ 38 ] proposed a more general 2D face alignment method.
Instead of aligning all faces to near-frontal shape,
APA adaptively learned multiple pose-specific templates, which preserved the face appearance with less artifact and information loss. 

 
 
 As the mainstream FR training sets are becoming larger gradually, some FR methods choose to omit face alignment step and train (or test) with face patches directly from face detection. Technically, face alignment is a way to increase intra-class compactness. And a well-trained FR model with more IDs can also obtain a compact feature distribution for each class. Therefore, face alignment may not be necessary nowadays. 

 
 
 
 

### 3.2 Training and testing a FR deep model

 
 After preprocessing, face images with their ground truth IDs can be used to train a FR deep model.
As the progressing of computational hardware, any mainstream backbone is able to build a FR network.
Different backbones with similar parameter amount have similar accuracy on FR.
As a result, you can use Resnet, ResNext, SEResnet, Inception net, Densenet, etc. to form your FR system.
If you need to design a FR system with limited calculation resource, Mobilenet with its variation will be good options.
In addition, NAS can be used to search a better network hyper-parameter.
Network distilling technology is also welcomed while building industrial FR system. 

 
 
 The training algorithms will be elaborated in the following sections. For testing, a face image after preprocessing can be inputted into a trained FR model to get its face embedding. Here, we list some practical tricks which will be utilized in the model training and testing steps.
Firstly, training image augmentation is useful, especially for IDs with inadequate samples. Common augmentation ways, such as adding noise, blurring, modifying colors, are usually employed in FR training. However, randomly cropping should be avoided, since it will ruin face alignment results.
Secondly, flipping face images horizontally is a basic operation in both of training and testing FR model. In training, flipping faces can be consider as a data augmentation process. In testing, we can put image I I and its flipped mirror image I ′ I^{\prime} into the model, and get their embeddings f f and f ′ f^{\prime} . Then their mean feature f + f ′ 2 \frac{f+f^{\prime}}{2} can be used as the embedding of image I I . This trick will further improve the accuracy of FR results. 

 
 
 

### 3.3 Face recognition by comparing face embeddings

 
 The last part of face inference pipeline is face recognition by comparing face embeddings. According to the application, it can be divided into face verification and face identification.
To apply FR, a face gallery needs to be built. First we have a face ID set S S , where each ID contains one (or several) face image(s). All face embeddings from the gallery images will be extracted by a trained model and saved in a database. 

 
 
 The protocol of face verification (FV) is: giving a face image I I and a specific ID in the gallery, output a judgement whether the face I I belongs to the ID. So FV is a 1:1 problem. We extract the feature of image I I and calculate its similarity s ⁡ ( I , I ​ D ) s(I,ID) with the ID’s embedding in the database. s ⁡ ( I , I ​ D ) s(I,ID) is usually measured by cosine similarity. If s ⁡ ( I , I ​ D ) s(I,ID) is larger than a threshold μ \mu , we will output ‘True’ which represents image I I belongs to the ID, and vice verse. 

 
 
 However, face identification (FI) is a 1:N problem. Giving a face image I I and a face ID set S S , it outputs the ID related to the face image, or ‘not recognized’.
Similarly, we extract the feature of I I and calculate its similarity s ⁡ ( I , I ​ D ) , I ​ D ∈ S s(I,ID),ID\in S with all IDs’ embeddings in the database.
Then, we find the maximum of all similarity values. If max ⁡ ( s ⁡ ( I , I ​ D ) ) \max(s(I,ID)) is larger than a threshold μ \mu , we output its related ID arg I ​ D ⁡ max ⁡ ( s ⁡ ( I , I ​ D ) ) \arg_{ID}\max(s(I,ID)) as identification target; otherwise, we output ‘not recognized’, which shows that the person of image I I is not in the database.
Phan et al. [ 39 ] enriched the FI process by adopting an extra Earth Mover’s Distance.
They first used cosine similarity to obtain a part of the most similar faces of the query image. Then a patch-wise Earth Mover’s Distance was employed to re-rank these similarities to get final identification results.
In the subsection 4.5 , we will introduce some methods to accelerate the process of face identification. 

 
 
 
 

## 4 Algorithms

 
 In this section, we will introduce FR algorithms in recent years. Based on different aspects in deep FR modeling, we divided all FR methods into several categories: designing loss function, refining embedding, FR with massive IDs, FR on uncommon images, FR pipeline acceleration, and close-set training. 

 
 

### 4.1 Loss Function

 

#### 4.1.1 Loss based on metric learning

 
 Except from softmax based classification, FR can be also regarded as extracting face features and performing feature matching. Therefore, training FR model is a process to learn a compact euclidean feature space, where distance directly correspond to a measure of face similarity. This is the basic modeling of metric learning.
In this section, we introduce some representative FR methods based on metric learning. 

 
 
 Inspired by the pipeline of face verification, one intuitive loss designing is judging whether two faces in a pair have identical ID or not.
Han et al. [ 40 ] used this designing in a cross-entropy way, and the loss is: 

 

 
 | 
 L i , j = − [ y i ​ j ​ log ⁡ p i ​ j + ( 1 − y i ​ j ) ​ log ⁡ ( 1 − p i ​ j ) ] L_{i,j}=-[y_{ij}\log p_{ij}+(1-y_{ij})\log(1-p_{ij})] | 
 | 
 (2) | 
 

 where y i ​ j y_{ij} is the binary GT of whether the two compared face images i i and j j belong to the same identity. p i ​ j p_{ij} is the logits value. 

 
 
 Different from [ 40 ] , contrastive loss [ 41 ] was proposed to directly compare the features of two face images in a metric learning way.
If two images belong to a same ID, their features in the trained space should be closer to each other, and vice verses.
As a result, the contrastive loss is: 

 

 
 | 
 L i , j = { 1 2 ​ ‖ f i − f j ‖ 2 2 , i ​ f ​ y i ​ j = 1 1 2 ​ m ​ a ​ x ​ ( 0 , m − ‖ f i − f j ‖ 2 2 ) , i ​ f ​ y i ​ j = − 1 L_{i,j}=\left\{\begin{array}[]{ll}\frac{1}{2}\|f_{i}-f_{j}\|^{2}_{2}, if\ y_{ij}=1\\
\frac{1}{2}max(0,m-\|f_{i}-f_{j}\|^{2}_{2}), if\ y_{ij}=-1\end{array}\right. | 
 | 
 (3) | 
 

 where i i , j j are two samples of a training pair, and f i f_{i} and f j f_{j} are their features. ‖ f i − f j ‖ 2 2 \|f_{i}-f_{j}\|^{2}_{2} is their Euclidean distance. m m is a margin for enlarging the distance of sample pairs with different IDs (negative pairs).
Further more, Euclidean distance can be replaced by cosine distance, and contrastive loss become the following form: 

 

 
 | 
 L i , j = 1 2 ​ ( y i ​ j − σ ⁡ ( w ​ d + b ) ) 2 , d = f i ​ f j ‖ f i ‖ 2 ​ ‖ f j ‖ 2 L_{i,j}=\frac{1}{2}(y_{ij}-\sigma(wd+b))^{2}\ \ ,\ \ d=\frac{f_{i}f_{j}}{\|f_{i}\|_{2}\|f_{j}\|_{2}} | 
 | 
 (4) | 
 

 where d d is the cosine similarity between feature f i f_{i} and f j f_{j} , w w and b b are learnable scaling and shifting parameters, σ \sigma is the sigmoid function. 

 
 
 Different from contrastive loss, BioMetricNet [ 42 ] did not impose any specific metric on facial features. Instead, it shaped the decision space by learning a latent representation in which matching (positive) and non-matching (negative) pairs are mapped onto clearly separated and well-behaved target distributions.
BioMetricNet first extracted face features of matching and non-matching pairs, and mapped them into a new space in which a decision is made. Its loss used to measure the statistics distribution of both matching and non-matching pairs in a complicated form. 

 
 
 FaceNet [ 34 ] proposed the triplet loss to minimize the distance between an anchor and a positive, both of which have the same identity, and maximize the distance between the anchor and a negative of a different identity.
Triplet loss was motivated in [ 43 ] in the context of nearest-neighbor classification.
The insight of triplet loss, is to ensure that an image x a x^{a} (anchor) of a specific person is closer to all other images x p x^{p} (positive) of the same person than it is to any image x n x^{n} (negative) of any other person.
Thus the loss is designed as: 

 

 
 | 
 L = ∑ ( x a , x p , x n ) ∈ T max ⁡ ( ‖ f ⁡ ( x a ) − f ⁡ ( x p ) ‖ 2 2 − ‖ f ⁡ ( x a ) − f ⁡ ( x n ) ‖ 2 2 + α , 0 ) L=\sum_{(x^{a},x^{p},x^{n})\in T}\max(\|f(x^{a})-f(x^{p})\|_{2}^{2}-\|f(x^{a})-f(x^{n})\|_{2}^{2}+\alpha,0) | 
 | 
 (5) | 
 

 where T T is is the set of all possible triplets in the training set. f ⁡ ( x ) f(x) is the embedding of face x x . α \alpha is a margin that is enforced between positive and negative pairs.
In order to ensure fast convergence, it is crucial to select triplets that violate the triplet constraint of: 

 

 
 | 
 ‖ f ⁡ ( x a ) − f ⁡ ( x p ) ‖ 2 2 + α ‖ f ⁡ ( x a ) − f ⁡ ( x n ) ‖ 2 2 \|f(x^{a})-f(x^{p})\|_{2}^{2}+\alpha \|f(x^{a})-f(x^{n})\|_{2}^{2} | 
 | 
 (6) | 
 

 This means that, given x a x^{a} , we want to select an x p x^{p} (hard positive) such that arg ⁡ max x p ⁡ ‖ f ⁡ ( x a ) − f ⁡ ( x p ) ‖ 2 2 \arg\max_{x^{p}}\|f(x^{a})-f(x^{p})\|_{2}^{2} , and similarly x n x^{n} (hard negative) such that arg ⁡ min x n ⁡ ‖ f ⁡ ( x a ) − f ⁡ ( x n ) ‖ 2 2 \arg\min_{x^{n}}\|f(x^{a})-f(x^{n})\|_{2}^{2} .
As a result, the accuracy of the triplet loss model is highly sensitive to the training triplet sample selection. 

 
 
 Kang et al. [ 44 ] simplified both of contrastive and triplet loss and designed a new loss as: 

 
 
 
 | 
 L \displaystyle L | 
 = λ 1 ​ L t + λ 2 ​ L p + λ 3 ​ L s ​ o ​ f ​ t ​ m ​ a ​ x \displaystyle=\lambda_{1}L_{t}+\lambda_{2}L_{p}+\lambda_{3}L_{softmax} | 
 | 
 (7) | 

 
 | 
 L t \displaystyle L_{t} | 
 = ∑ ( x a , x p , x n ) ∈ T max ⁡ ( 0 , 1 − ‖ f ⁡ ( x a ) − f ⁡ ( x n ) ‖ 2 ‖ f ⁡ ( x a ) − f ⁡ ( x p ) ‖ 2 + m ) \displaystyle=\sum_{(x^{a},x^{p},x^{n})\in T}\max(0,1-\frac{\|f(x^{a})-f(x_{n})\|_{2}}{\|f(x^{a})-f(x_{p})\|_{2}+m}) | 
 | 

 
 | 
 L p \displaystyle L_{p} | 
 = ∑ ( x a , x p , . ) ∈ T ∥ f ( x a ) − f ( x p ) ∥ 2 2 \displaystyle=\sum_{(x^{a},x^{p},.)\in T}\|f(x^{a})-f(x^{p})\|_{2}^{2} | 
 | 
 

 where L t L_{t} , L p L_{p} , L s ​ o ​ f ​ t ​ m ​ a ​ x L_{softmax} are modified triplet ratio loss, pairwise loss and regular softmax loss, respectively. Factor m m in L t L_{t} is a margin to enlarge the difference between positive pair and negative pair, which has similar use with α \alpha in ( 5 ). 

 
 
 Contrastive loss measures the difference of a image pair; triplet loss represented the relations in a triplet: anchor, positive and negative samples.
Some researchers have developed more metric learning based FR losses by describing more samples. 

 
 
 In order to enhance the discrimination of the deeply learned features, the center loss [ 45 ] simultaneously learned a center for deep features of each class and
penalized the distances between the deep features and their corresponding class centers.
Center loss can be added on softmax or other FR loss to refine feature distribution while training.
The form of center loss is: 

 

 
 | 
 L = L s ​ o ​ f ​ t ​ m ​ a ​ x + λ ​ L c , L c ​ ( i ) = 1 2 ​ ∑ i = 1 m ‖ x i − c y i ‖ 2 2 L=L_{softmax}+\lambda L_{c}\ \ ,\ \ L_{c}(i)=\frac{1}{2}\sum_{i=1}^{m}\|x_{i}-c_{y_{i}}\|^{2}_{2} | 
 | 
 (8) | 
 

 where L s ​ o ​ f ​ t ​ m ​ a ​ x L_{softmax} is softmax loss on labelled training data. i i is a image sample with its face label y i y_{i} , and x i x_{i} is its deep feature. m m is the number of training classes.
 c y i c_{y_{i}} denotes the y i y_{i} -th class center of deep features. The formulation
of center loss effectively characterizes the intra-class variations. 

 
 
 

#### 4.1.2 Larger margin loss

 
 Face recognition is naturally treated as a classification problem, and thus using softmax as the loss to train a FR model is an intuitive consideration.
Larger margin loss (also is called as angular margin based loss) is derived from softmax.
Larger margin loss is a major direction in FR, since it has largely improved the performance of FR in academy and industry.
The intuition is that facial images with same identities are expected to be closer in the representation space, while different identities expected to be far apart.
As a result, larger margin losses encourage the intra-class compactness and penalize the similarity of different identities. 

 
 
 First, we give the formulation of softmax as follows: 

 

 
 | 
 L i = − log ⁡ e W y i ⋅ x i + b y i ∑ j e W j ⋅ x j + b j L_{i}=-\log\frac{e^{{W_{y_{i}}\cdot x_{i}+b_{y_{i}}}}}{\sum_{j}e^{{W_{j}\cdot x_{j}+b_{j}}}} | 
 | 
 (9) | 
 

 where W {W} is the weighting vector of the last fully connected layer, and W j {W_{j}} represents the weight of class j j ; x i x_{i} and y i y_{i} are the face embedding of sample i i and its ground truth ID.
For class y i y_{i} , sample i i is a positive sample; and for class j ⁡ ( j ≠ y i ) j(j\neq y_{i}) , sample i i is a negative sample.
These two definitions will be used in this whole paper.
Considering positive and negative samples, softmax can be rewritten as: 

 

 
 | 
 L i = − log ⁡ e W y i ⋅ x i + b y i e W y i ⋅ x i + b y i + ∑ j ≠ y i e W j ⋅ x j + b j L_{i}=-\log\frac{e^{{W_{y_{i}}\cdot x_{i}+b_{y_{i}}}}}{e^{{W_{y_{i}}\cdot x_{i}+b_{y_{i}}}}+\sum_{j\neq y_{i}}e^{{W_{j}\cdot x_{j}+b_{j}}}} | 
 | 
 (10) | 
 

 
 
 L-Softmax [ 46 ] first designed margin based loss by measuring features angles. First, it omits the bias of each class b j b_{j} , and changes the inner product of features and weights W j ⋅ x i W_{j}\cdot x_{i} to a new form ‖ W j ‖ ⋅ ‖ x i ‖ ⋅ cos ⁡ ( θ j ) \|W_{j}\|\cdot\|x_{i}\|\cdot\cos(\theta_{j}) , where θ j \theta_{j} is the angle between x i x_{i} and the weight W j W_{j} of class j.
In order to enlarge the margin of angels between each class, L-Softmax modifies cos ⁡ ( θ y i ) \cos(\theta_{y_{i}}) to ψ ⁡ ( θ y i ) \psi(\theta_{y_{i}}) by narrowing down the space between decision boundary and the class centers. In specific, it has: 

 

 
 | 
 L i = − log ⁡ e ‖ W y i ‖ ​ ‖ x i ‖ ​ ψ ​ ( θ y i ) e ‖ W y i ‖ ​ ‖ x i ‖ ​ ψ ​ ( θ y i ) + ∑ j ≠ y i e ‖ W j ‖ ​ ‖ x i ‖ ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{\left\|{W}_{y_{i}}\right\|\left\|{x}_{i}\right\|\psi\left(\theta_{y_{i}}\right)}}{e^{\left\|{W}_{y_{i}}\right\|\left\|{x}_{i}\right\|\psi\left(\theta_{y_{i}}\right)}+\sum_{j\neq y_{i}}e^{\left\|{W}_{j}\right\|\left\|{x}_{i}\right\|\cos\left(\theta_{j}\right)}} | 
 | 
 (11) | 
 

 

 
 | 
 ψ ⁡ ( θ ) = ( − 1 ) k ​ cos ⁡ ( m ​ θ ) − 2 ​ k \psi(\theta)=(-1)^{k}\cos(m\theta)-2k | 
 | 
 (12) | 
 

 where m m is a fix parameter which is an integer; the angle θ ∈ [ 0 , π ] \theta\in[0,\pi] has been divided in m m intervals: [ k ​ π m , ( k + 1 ) ​ π m ] [\frac{k\pi}{m},\frac{(k+1)\pi}{m}] ; k k is an integer and k ∈ [ 0 , m − 1 ] k\in[0,m-1] . Therefore, it has ψ ⁡ ( θ ) ≤ cos ⁡ ( θ ) \psi(\theta)\leq\cos(\theta) . 

 
 
 Similar with L-Softmax, Sphereface [ 47 ] proposed A-Softmax, which normalized each class weight W j {W}_{j} before calculating the loss. Therefore, the loss became: 

 

 
 | 
 L i = − log ⁡ e ‖ x i ‖ ​ ψ ​ ( θ y i ) e ‖ x i ‖ ​ ψ ​ ( θ y i ) + ∑ j ≠ y i e ‖ x i ‖ ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{\left\|{x}_{i}\right\|\psi\left(\theta_{y_{i}}\right)}}{e^{\left\|{x}_{i}\right\|\psi\left(\theta_{y_{i}}\right)}+\sum_{j\neq y_{i}}e^{\left\|{x}_{i}\right\|\cos\left(\theta_{j}\right)}} | 
 | 
 (13) | 
 

 In the training step of Sphereface, its actual loss is (1- α \alpha )*softmax + α \alpha *A-softmax, where α \alpha is a parameter in [0,1]. During training, α \alpha is changing gradually from 0 to 1. This design is based on two motivations. Firstly, directly training A-Softmax loss will lead to hard convergence, since it brutally pushes features from different ID apart. Secondly, training a softmax loss first will decrease the angle θ y i \theta_{y_{i}} between feature i i and its related weight W y i W_{y_{i}} , which causes the part cos ⁡ ( m ​ θ ) \cos({m\theta}) in A-softmax loss in a monotonous area. And it is easier to get a lower loss while gradient descending. 

 
 
 Besides the weight of each class, NormFace [ 48 ] proposed that face embeddings need to be normalized as well.
Also, NormFace scales up the normalized face embeddings, which will alleviate the data unbalancing problem of positive and negative samples for better convergence. However, NormFace doesn’t use extra margin for positive samples as Sphereface.
Therefore, the NormFace loss is shown as: 

 

 
 | 
 L i = − log ⁡ e s ​ W ~ y i ​ x ~ i ∑ j e s ​ W ~ j ​ x ~ j = − log ⁡ e s ​ cos ⁡ ( θ y i ) ∑ j e s ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{s{\tilde{W}_{y_{i}}\tilde{x}_{i}}}}{\sum_{j}e^{{s\tilde{W}_{j}\tilde{x}_{j}}}}=-\log\frac{e^{s\cos(\theta_{y_{i}})}}{\sum_{j}e^{s\cos(\theta_{j})}} | 
 | 
 (14) | 
 

 where W ~ {\tilde{W}} and x ~ {\tilde{x}} are the normalized class weights and face embeddings; s s is a scaling up parameter and s 1 s 1 .
To this end, both face embeddings and class weights are distributed on the hypersphere manifold, due to the normalization. 

 
 
 AM-Softmax [ 49 ] and CosFace [ 50 ] enlarged softmax based loss by minus a explicit margin m m outside cos ⁡ ( θ ) \cos(\theta) as follows: 

 

 
 | 
 L i = − log ⁡ e s ⁡ ( cos ⁡ ( θ y i ) − m ) e s ⁡ ( cos ⁡ ( θ y i ) − m ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{s(\cos(\theta_{y_{i}})-m)}}{e^{s(\cos(\theta_{y_{i}})-m)}+\sum_{j\neq y_{i}}e^{s\cos(\theta_{j})}} | 
 | 
 (15) | 
 

 
 
 ArcFace [ 51 ] putted the angular margin m m inside cos ⁡ ( θ ) \cos(\theta) , and made the margin more explainable. The ArcFace loss is shown as: 

 

 
 | 
 L i = − log ⁡ e s ⁡ ( c ​ o ​ s ​ ( θ y i + m ) ) e s ⁡ ( cos ⁡ ( θ y i + m ) ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{s(cos(\theta_{y_{i}}+m))}}{e^{s(\cos(\theta_{y_{i}}+m))}+\sum_{j\neq{y_{i}}}e^{s\cos(\theta_{j})}} | 
 | 
 (16) | 
 

 
 
 The aforementioned angular margin based losses (such as CosFace and ArcFace) includes sensitive hyper-parameters which can make training process unstable.
P2SGrad [ 52 ] was proposed to address this challenge by directly designing the gradients for training in an adaptive manner. 

 
 
 AM-Softmax [ 49 ] , CosFace [ 50 ] and ArcFace [ 51 ] only added angular margins of positive samples. Therefore, their decision boundaries only enforced positive samples getting closer to its centers.
On the contrary, SV-Softmax [ 53 ] got the intra-class compactness by pushing the hard negative samples away from positive class centers, which was chosen by hard example mining.
In SV-Softmax, a binary label I j I_{j} adaptively indicates whether a sample j j is hard negative or not, which was defined as: 

 

 
 | 
 I j = { 0 , cos ⁡ ( θ y i ) − cos ⁡ ( θ j ) ≥ 0 1 , cos ⁡ ( θ y i ) − cos ⁡ ( θ j ) 0 I_{j}=\left\{\begin{array}[]{ll}0, \cos\left(\theta_{y_{i}}\right)-\cos\left(\theta_{j}\right)\geq 0\\
1, \cos\left(\theta_{y_{i}}\right)-\cos\left(\theta_{j}\right) 0\end{array}\right. | 
 | 
 (17) | 
 

 Therefore, the loss of SV-Softmax is: 

 

 
 | 
 L i = − log ⁡ e s ⁡ ( cos ⁡ ( θ y i ) ) e s ⁡ ( cos ⁡ ( θ y i ) ) + h ⁡ ( t , θ j , I j ) ​ ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{s(\cos(\theta_{y_{i}}))}}{e^{s(\cos(\theta_{y_{i}}))}+h(t,\theta_{j},I_{j})\sum_{j\neq y_{i}}e^{s\cos(\theta_{j})}} | 
 | 
 (18) | 
 

 

 
 | 
 h ⁡ ( t , θ j , I j ) = e s ⁡ ( t − 1 ) ​ ( cos ⁡ ( θ j ) + 1 ) ​ I k h(t,\theta_{j},I_{j})=e^{s(t-1)(\cos(\theta_{j})+1)I_{k}} | 
 | 
 (19) | 
 

 
 
 One step forward, SV-softmax evolve to SV-X-Softmax by adding large margin loss on positive samples: 

 

 
 | 
 L i = − log ⁡ e s ​ f ​ ( m , θ y i ) e s ​ f ​ ( m , θ y i ) + ∑ j ≠ y i h ⁡ ( t , θ j , I j ) ​ e s ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{sf(m,\theta_{y_{i}})}}{e^{sf(m,\theta_{y_{i}})}+\sum_{j\neq y_{i}}h(t,\theta_{j},I_{j})e^{s\cos(\theta_{j})}} | 
 | 
 (20) | 
 

 where f ⁡ ( m , θ y i ) f(m,\theta_{y_{i}}) can be any type of large margin loss above, such as CosFace or ArcFace.
At the same time, I j I_{j} is changed to: 

 

 
 | 
 I j = { 0 , f ⁡ ( m , θ y i ) − cos ⁡ ( θ j ) ≥ 0 1 , f ⁡ ( m , θ y i ) − cos ⁡ ( θ j ) 0 I_{j}=\left\{\begin{array}[]{ll}0, f(m,\theta_{y_{i}})-\cos\left(\theta_{j}\right)\geq 0\\
1, f(m,\theta_{y_{i}})-\cos\left(\theta_{j}\right) 0\end{array}\right. | 
 | 
 (21) | 
 

 SV-Softmax has also evolved to MV-Softmax [ 54 ] by re-defining h ⁡ ( t , θ j , I j ) h(t,\theta_{j},I_{j}) . 

 
 
 In aforementioned methods such as CosFace [ 50 ] and ArcFace [ 51 ] , features have been normalized in the loss.
Ring loss [ 55 ] explicitly measured the length of features and forced them to a same length R R , which is shown as: 

 

 
 | 
 L R = λ 2 ​ m ​ ∑ i = 1 m ( ‖ f ⁡ ( x i ) ‖ 2 − R ) 2 L_{R}=\frac{\lambda}{2m}\sum_{i=1}^{m}(\|f(x_{i})\|_{2}-R)^{2} | 
 | 
 (22) | 
 

 where f ⁡ ( x i ) f(x_{i}) is the feature of the sample x i x_{i} . R R is the target norm value which is also learned and m m is the batch-size.
 λ \lambda is a weight for trade-off between the primary loss function.
In [ 55 ] , the primary loss function was set to softmax and SphereFace [ 47 ] . 

 
 
 Considering a training set with noise, Hu et al. [ 56 ] concluded that the smaller angle θ \theta between a sample and its related class center, the greater probability that this sample is clean.
Based on this fact, a paradigm of noisy learning is proposed, which calculates the weights of samples according to θ \theta , where samples with less noise will be distributed higher weights during training.
In a training set with noisy samples, the distribution of θ \theta normally consists of one or two Gaussian distributions.
These two Gaussian distributions correspond to noisy and clean samples.
For the entire distribution, left and right ends of θ \theta distribution are defined as δ l \delta_{l} and δ r \delta_{r} . And the peaks of the two Gaussian distributions are μ l \mu_{l} and μ r \mu_{r} ( μ l = μ r \mu_{l}=\mu_{r} if the distribution consists of only one Gaussian).
During training, θ \theta tends to decreasing, so δ r \delta_{r} can indicate the progress of the training.
In the initial training stage, the θ \theta distribution of clean and noisy samples will be mixed together. At this stage, the weights of all samples are set to the same, that is, w 1 , i = 1 w_{1,i}=1 .
As the training progressing, θ \theta of the clean sample is gradually smaller than the noisy sample.
At this time, clean sample should be given a higher weight, that is, w 2 i = s ​ o ​ f ​ t ​ m ​ a ​ x ​ ( λ ​ z ) s ​ o ​ f ​ t ​ m ​ a ​ x ​ ( λ ) w_{2_{i}}=\frac{softmax(\lambda z)}{softmax(\lambda)} , where z = cos ⁡ θ − μ l δ r − μ l z=\frac{\cos\theta-\mu_{l}}{\delta_{r}-\mu_{l}} , s ​ o ​ f ​ t ​ m ​ a ​ x ​ ( x ) = log ⁡ ( 1 + e x ) softmax(x)=\log(1+e^{x}) , λ \lambda is the regularization coefficient.
At later training stage, the weight of semi-hard samples should be higher than that of easy samples and hard samples, that is, w 3 , i = e − ( cos θ − μ r ) 2 / 2 σ 2 w_{3,i}=e^{-(\cos\theta-\mu_{r})^{2}/2\sigma^{2}} . 

 
 
 In conclusion, the final loss based on AM-Softmax is 

 

 
 | 
 L i = − w i ​ log ⁡ e s ⁡ ( cos ⁡ ( θ y i ) − m ) e s ⁡ ( cos ⁡ ( θ y i ) − m ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) L_{i}=-w_{i}\log\frac{e^{s(\cos(\theta_{y_{i}})-m)}}{e^{s(\cos(\theta_{y_{i}})-m)}+\sum_{j\neq y_{i}}e^{s\cos(\theta_{j})}} | 
 | 
 (23) | 
 

 

 
 | 
 w i = α ⁡ ( δ r ) ​ w 1 , i + β ⁡ ( δ r ) ​ w 2 , i + γ ⁡ ( δ r ) ​ w 3 , i w_{i}=\alpha(\delta_{r})w_{1,i}+\beta(\delta_{r})w_{2,i}+\gamma(\delta_{r})w_{3,i} | 
 | 
 (24) | 
 

 where α ⁡ ( δ r ) \alpha(\delta_{r}) , β ⁡ ( δ r ) \beta(\delta_{r}) and γ ⁡ ( δ r ) \gamma(\delta_{r}) are parameters to reflect the training stage, which are designed empirically. 

 
 
 Similar with [ 56 ] , the sub-center ArcFace [ 57 ] also solved the problem of noisy sample learning on the basis of ArcFace.
The proposed sub-centers encourage one dominant sub-class that contains the majority of clean faces and non-dominant sub-classes that include hard or noisy faces aggregate, and thus they relax the intra-class constraint of ArcFace to improve the robustness to label noise. 

 
 
 In this method, Deng et al. modified the fully connected layer classifier with dimension d × n d\times n to a sub-classes classifier with dimension d × n × K d\times n\times K ,
where d d is the dimension of embedding, n n is the number of identities in the training set, and K K is the number of sub-centers for each identity.
The sub-center ArcFace loss is shown as 

 

 
 | 
 L i = − log ⁡ e s ⁡ ( cos ⁡ ( θ i , y i + m ) ) e s ⁡ ( cos ⁡ ( θ i , y i + m ) ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ i , j ) L_{i}=-\log\frac{e^{s(\cos(\theta_{i,y_{i}}+m))}}{e^{s(\cos(\theta_{i,y_{i}}+m))}+\sum_{j\neq{y_{i}}}e^{s\cos(\theta_{i,j})}} | 
 | 
 (25) | 
 

 

 
 | 
 θ i , j = arccos max k ( W j k T x i ) , k = 1 , ⋯ , K \theta_{i,j}=\arccos\max_{k}(W_{j_{k}}^{T}x_{i}),k={1,\cdots,K} | 
 | 
 (26) | 
 

 where θ i , j \theta_{i,j} is the smallest angular between embedding x i x_{i} and all sub-class weights in class j j . 

 
 
 CurricularFace [ 58 ] adopted the idea of curriculum learning and weighted the class score of negative samples. So in the earlier stage of training CurricularFace, the loss fits easy samples. And at the later training stage, the loss fits hard samples. In specific, CurricularFace loss is : 

 

 
 | 
 L i = − log ⁡ e s ​ f ​ ( m , θ y i ) e s ​ f ​ ( m , θ y i ) + ∑ j ≠ y i e s ​ N ​ ( t , m , θ j , θ y i ) L_{i}=-\log\frac{e^{sf(m,\theta_{y_{i}})}}{e^{sf(m,\theta_{y_{i}})}+\sum_{j\neq y_{i}}e^{sN(t,m,\theta_{j},\theta_{y_{i}})}} | 
 | 
 (27) | 
 

 and 

 

 
 | 
 N ⁡ ( t , m , θ j , θ y i ) = { cos ⁡ θ j , cos ⁡ ( θ y i + m ) − cos ⁡ ( θ j ) ≥ 0 cos ⁡ θ j ​ ( t + cos ⁡ θ j ) , cos ⁡ ( θ y i + m ) − cos ⁡ ( θ j ) 0 N(t,m,\theta_{j},\theta_{y_{i}})=\left\{\begin{array}[]{ll}\cos\theta_{j}, \cos(\theta_{y_{i}}+m)-\cos\left(\theta_{j}\right)\geq 0\\
\cos\theta_{j}(t+\cos\theta_{j}), \cos(\theta_{y_{i}}+m)-\cos\left(\theta_{j}\right) 0\end{array}\right. | 
 | 
 (28) | 
 

 
 
 Different from SV-Softmax, t t in N ⁡ ( t , θ j ) N(t,\theta_{j}) from CurricularFace is dynamically updated by Exponential Moving Average (EMA), which is shown as follows: 

 

 
 | 
 t ( k ) = α ​ r ( k ) + ( 1 − α ) ​ r ( k − 1 ) t^{(k)}=\alpha r^{(k)}+(1-\alpha)r^{(k-1)} | 
 | 
 (29) | 
 

 where t 0 = 0 t_{0}=0 , α \alpha is the momentum parameter and set to 0.99. 

 
 
 NPCface [ 59 ] observed that high possibility of co-occurrence of hard positive and hard negative appeared in large-scale dataset. It means that if one sample with ground truth class i i is hard positive, it has a larger chance to be a hard negative sample of class j j . Therefore, NPCface emphasizes the training on both negative and positive hard cases via the collaborative margin mechanism in the softmax logits, which is shown as: 

 
 
 

 
 | 
 L i = − log ⁡ e s ​ cos ⁡ ( θ y i + m i ~ ) e s ​ cos ⁡ ( θ y i + m i ~ ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j + m j ~ ) L_{i}=-\log\frac{e^{s\cos(\theta_{y_{i}}+\tilde{m_{i}})}}{e^{s\cos(\theta_{y_{i}}+\tilde{m_{i}})}+\sum_{j\neq y_{i}}e^{s\cos(\theta_{j}+\tilde{m_{j}})}} | 
 | 
 (30) | 
 

 

 
 | 
 m ~ i = { m 0 + ∑ j ≠ y i ( M i , j ​ cos ⁡ ( θ j ) ) ∑ j ≠ y i M i , j ​ m 1 , if ​ ∑ j ≠ y i M i , j ≠ 0 m 0 , if ​ ∑ j ≠ y i M i , j = 0 \tilde{m}_{i}=\left\{\begin{array}[]{ll}m_{0}+\frac{\sum_{j\neq y_{i}}(M_{i,j}\cos(\theta_{j}))}{\sum_{j\neq y_{i}}M_{i,j}}m_{1}, \text{if}\ \sum_{j\neq y_{i}}M_{i,j}\neq 0\\
m_{0}, \text{if}\ \sum_{j\neq y_{i}}M_{i,j}=0\end{array}\right. | 
 | 
 (31) | 
 

 where M i , j M_{i,j} is a binary indicator function, which represents whether sample i i is a hard negative sample of class j j . 

 
 
 UniformFace [ 60 ] proposed that above large margin losses did not consider the distribution of all training classes.
With the prior that faces lie on a hypersphere manifold,
UniformFace imposed an equidistributed constraint by uniformly
spreading the class centers on the manifold, so that the
minimum distance between class centers can be maximized
through completely exploitation of the feature space.
Therefore, the loss is : 

 

 
 | 
 L = L l ​ m ​ l + L u , L u = λ M ⁡ ( M − 1 ) ​ ∑ j 1 = 1 M ∑ j 2 ≠ j 1 1 d ⁡ ( 𝒄 j 1 , 𝒄 j 2 ) L=L_{lml}+L_{u}\ \ ,\ \ L_{u}=\frac{\lambda}{M(M-1)}\sum^{M}_{j_{1}=1}\sum_{j_{2}\neq j_{1}}\frac{1}{d(\boldsymbol{c}_{j_{1}},\boldsymbol{c}_{j_{2}})} | 
 | 
 (32) | 
 

 where L l ​ m ​ l L_{lml} can be any large margin loss above, such as CosFace or ArcFace; and L u L_{u} is a uniform loss, which enforces the distribution of all training classes more uniform.
 M M is the number of classes in L u L_{u} , and 𝒄 i \boldsymbol{c}_{i} is the center of class i i , λ \lambda is a weight.
As the class centers c j c_{j} are continuously changing during
the training, the entire training set is require to update c j c_{j} in each iteration, which is not applicable in practice. Therefore, UniformFace updated the centers on each mini-batch by a newly designed step as follows: 

 

 
 | 
 Δ ​ 𝒄 j = ∑ i = 1 n δ ⁡ ( y i = j ) ⋅ ( 𝒄 j − 𝒙 j ) 1 + ∑ i = 1 n δ ⁡ ( y i = j ) \Delta\boldsymbol{c}_{j}=\frac{\sum_{i=1}^{n}\delta(y_{i}=j)\cdot(\boldsymbol{c}_{j}-\boldsymbol{x}_{j})}{1+\sum_{i=1}^{n}\delta(y_{i}=j)} | 
 | 
 (33) | 
 

 where n n is the number of samples in a mini-batch, x j x_{j} is a embedding in this batch, δ ( . ) = 1 \delta(.)=1 
if the condition is true and δ ( . ) = 0 \delta(.)=0 otherwise. 

 
 
 Similar with [ 60 ] , Zhao et al. [ 61 ] also considered the distribution of each face class in the feature space. They brought in the concept of ‘inter-class separability’ in large-margin based loss and put forward RegularFace. 

 
 
 The inter-class separability of the i i -th identity class: S ​ e ​ p i Sep_{i} is defined as the largest cosine similarity between class i i and other centers with different IDs, where: 

 

 
 | 
 S ​ e ​ p i = max j ≠ i ⁡ cos ⁡ ( φ i , j ) = max j ≠ i ⁡ W i ⋅ W j ‖ W i ‖ ⋅ ‖ W j ‖ Sep_{i}=\max_{j\neq i}\cos(\varphi_{i,j})=\max_{j\neq i}\frac{W_{i}\cdot W_{j}}{\|W_{i}\|\cdot\|W_{j}\|} | 
 | 
 (34) | 
 

 where W i W_{i} is the i i -th column of W W which represents the weight vector for ID i i .
RegularFace pointed out that different ids should be uniformly distributed in the feature space, as a result, the mean and variance of S ​ e ​ p i Sep_{i} will be small. Therefore RegularFace jointly supervised the FR model with angular softmax loss
and newly designed exclusive regularization. The overall loss function is: 

 

 
 | 
 L = L s + λ ​ ℒ r ​ ( W ) , ℒ r ​ ( W ) = 1 C ​ ∑ i max j ≠ i ⁡ W i ⋅ W j ‖ W i ‖ ⋅ ‖ W j ‖ L=L_{s}+\lambda\mathcal{L}_{r}(W)\ \ ,\ \ \mathcal{L}_{r}(W)=\frac{1}{C}\sum_{i}\max_{j\neq i}\frac{W_{i}\cdot W_{j}}{\|W_{i}\|\cdot\|W_{j}\|} | 
 | 
 (35) | 
 

 where C C is the number of classes. λ \lambda is the weight coefficient. L s L_{s} is the softmax-based classification loss function or other large margin loss [ 51 , 50 ] , and ℒ r ​ ( θ , W ) \mathcal{L}_{r}(\theta,W) is an exclusive regularization, which provides extra inter-class separability and encourages uniform distribution of each class. 

 
 
 Above large margin loss based classification framework only consider the distance representation between samples and class centers (prototypes).
VPL [ 62 ] proposed a sample-to-sample distance representation and embedded it into the original sample-to-prototype classification framework.
In VPL, both sample-to-sample and sample-to-prototype distance is replaced by the distance between sample to its variational prototype.
In specific, the loss of VPL is: 

 

 
 | 
 L i = − log ⁡ e s ​ cos ⁡ ( θ ~ y i + m ) e s ​ cos ⁡ ( θ ~ y i + m ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ ~ y i ) L_{i}=-\log\frac{e^{s\cos(\tilde{\theta}_{y_{i}}+m)}}{e^{s\cos(\tilde{\theta}_{y_{i}}+m)}+\sum_{j\neq y_{i}}e^{s\cos(\tilde{\theta}_{y_{i}})}} | 
 | 
 (36) | 
 

 where θ ~ j \tilde{\theta}_{j} is the angle between the feature x i x_{i} and the variational prototype W ~ j \tilde{W}_{j} . The definition of m m and s s are similar with ArcFace [ 51 ] : m m is the additive angular margin set as 0.5, and s s is the feature scale parameter set as 64.
Variational prototype W ~ j \tilde{W}_{j} is iteratively updated in training step as: 

 

 
 | 
 W ~ j = λ 1 ​ W j + λ 2 ​ M j \tilde{W}_{j}=\lambda_{1}W_{j}+\lambda_{2}M_{j} | 
 | 
 (37) | 
 

 where W j W_{j} is the prototype of class j j , namely the weight of class j j . M j M_{j} is the feature centers of all embeddings in the mini batch which are belongs to class j j . λ 1 \lambda_{1} and λ 2 \lambda_{2} are their weights. 

 
 
 UIR [ 63 ] trained the face feature extractor in semi-supervised way and modified original softmax loss by adding more unlabelled data.
UIR design a loss λ ​ L u ​ i ​ r \lambda L_{uir} to measure the penalty on unlabelled training data, which dose not belong to any ID of labelled data. UIR assumed that, inference the unlabelled data on a well-trained face classifier, their logits scores of each class should be as equal as possible. Therefore, the assumption can be abstracted as the following optimization problem: 

 

 
 | 
 max ⁡ p 1 ⋅ p 2 ⋅ … ⋅ p n , s . t . ∑ 1 n p i = 1 \max p_{1}\cdot p_{2}\cdot...\cdot p_{n},\ s.t.\sum_{1}^{n}p_{i}=1 | 
 | 
 (38) | 
 

 where n n is the number of IDs in the training set. As a result, changing this optimization in a loss way, we have the UIR loss follows: 

 

 
 | 
 L i = L s ​ o ​ f ​ t ​ m ​ a ​ x + λ L u ​ i ​ r , L u ​ i ​ r = − ∑ i = 1 n log ( p i ) L_{i}=L_{softmax}+\lambda L_{uir}\ \ ,\ \ L_{uir}=-\sum_{i=1}^{n}\log(p_{i}) | 
 | 
 (39) | 
 

 where L s ​ o ​ f ​ t ​ m ​ a ​ x L_{softmax} is softmax loss on labelled training data, and it can be replaced by other large margin loss. 

 
 
 AdaCos [ 64 ] took CosFace [ 50 ] as a benchmark and proposed new method to automatically modify the value of margin and scale hyper-parameter: m m and s s as the training iteration goes. 

 
 
 AdaCos observed that when s s is too small, the positive probability of one sample i i after margin based loss is also relatively small, even if the angular distance θ ⁡ ( i , y i ) \theta(i,y_{i}) between i i and its class center is small enough. On the contrary, when s s is too large, the positive probability of sample i i is reaching 1, even if the angular distance θ ⁡ ( i , y i ) \theta(i,y_{i}) is still large. 

 
 
 As a result, changing the value of s s during for a proper value will be good at FR model convergence.
The loss of AdaCos is similar with CosFace. But its scale parameter s s is changing as follows: 

 

 
 | 
 s ( t ) = { 2 ​ log ⁡ ( N − 1 ) , t = 0 log ⁡ B a ​ v ​ g ( t ) cos ⁡ ( min ⁡ ( π / 4 , θ m ​ e ​ d ( t ) ) ) , t 0 s^{(t)}=\left\{\begin{array}[]{ll}\sqrt{2}\log(N-1), t=0\\
\frac{\log B_{avg}^{(t)}}{\cos(\min(\pi/4,\theta_{med}^{(t)}))}, t 0\end{array}\right. | 
 | 
 (40) | 
 

 where N N is the number of training IDs; t t is the iteration times.
 θ m ​ e ​ d ( t ) \theta_{med}^{(t)} is the median of all corresponding classes’ angles,
 θ i , y i ( t ) \theta^{(t)}_{i,y_{i}} from the mini-batch at the t t -th iteration.
 B a ​ v ​ g ( t ) B_{avg}^{(t)} is the average of B i ( t ) B_{i}^{(t)} , which is: 

 

 
 | 
 B a ​ v ​ g ( t ) = 1 ‖ N ( t ) ‖ ​ ∑ i ∈ N ( t ) B i ( t ) , B i ( t ) = ∑ k ≠ y i e s ( t − 1 ) ​ cos ⁡ ( θ i , k ) B_{avg}^{(t)}=\frac{1}{\|N^{(t)}\|}\sum_{i\in N^{(t)}}B_{i}^{(t)}\ \ ,\ \ B_{i}^{(t)}=\sum_{k\neq y_{i}}e^{s^{(t-1)}\cos(\theta_{i,k})} | 
 | 
 (41) | 
 

 where B i ( t ) B_{i}^{(t)} is the total loss of all negative samples related to ID i i . N ( t ) N^{(t)} is the set of IDs in the mini-batch at t t -th iteration, and ‖ N ( t ) ‖ \|N^{(t)}\| is its number. The analysis of s ( t ) s^{(t)} in detail can be check in the article of AdaCos [ 64 ] , which will not be elaborated here. 

 
 
 Similar with AdaCos [ 64 ] , Fair loss [ 65 ] also chose to adaptively modify the value of m m in each iteration, however it is controlled by reinforcement learning.
In detail, Fair loss is shown as follows: 

 

 
 | 
 L i = − log ⁡ P y i ∗ ​ ( m i ​ ( t ) , x i ) P y i ∗ ​ ( m i ​ ( t ) , x i ) + ∑ j ≠ y i P j ​ ( x i ) L_{i}=-\log\frac{P^{*}_{y_{i}}(m_{i}(t),x_{i})}{P^{*}_{y_{i}}(m_{i}(t),x_{i})+\sum_{j\neq y_{i}}P_{j}(x_{i})} | 
 | 
 (42) | 
 

 where P j ​ ( x i ) = e s ​ cos ⁡ ( ( θ j ) CLOSE P_{j}(x_{i})=e^{s\cos((\theta_{j})} is the same with ( 16 ) and ( 15 ).
The specific formulation of P y i ∗ ​ ( m i ​ ( t ) , x i ) P^{*}_{y_{i}}(m_{i}(t),x_{i}) can be different according to different large margin loss.
Based on CosFace [ 50 ] or ArcFace [ 51 ] , P ∗ P^{*} can be formulated as follows: 

 

 
 | 
 P y i ∗ ​ ( m i ​ ( t ) , x i ) = e s ⁡ ( cos ⁡ ( θ y i ) − m i ​ ( t ) ) , o ​ r ​ P y i ∗ ​ ( m i ​ ( t ) , x i ) = e s ⁡ ( cos ⁡ ( θ y i + m i ​ ( t ) ) ) P^{*}_{y_{i}}(m_{i}(t),x_{i})=e^{s(\cos(\theta_{y_{i}})-m_{i}(t))}\ \ ,or\ \ P^{*}_{y_{i}}(m_{i}(t),x_{i})=e^{s(\cos(\theta_{y_{i}}+m_{i}(t)))} | 
 | 
 (43) | 
 

 Then the problem of finding an appropriate margin adaptive strategy is formulated as a Markov Decision Process (MDP),
described by ( S , A , T , R ) (S,A,T,R) as the states, actions, transitions and rewards.
An agent is trained to adjust the margin in every state based on enforcement learning. 

 
 
 Representative large margin methods such as CosFace [ 50 ] and ArcFace
have an implicit assumption that all the classes possess sufficient samples to describe its distribution, so that a manually set margin is enough to equally squeeze each intra-class variations.
Therefore, they set same values of margin m m and scale s s for all classes.
In practice, data imbalance of different face IDs widely exists in 

 
 
 mainstream training datasets.
For those IDs with rich samples and large intra-class variations, the space spanned by existing training samples can represent the real distribution. 

 
 
 But for the IDs with less samples, its features will be pushed to a smaller hyper space if we give it a same margin as ID with more samples.
AdaptiveFace [ 66 ] proposed a modified AdaM-Softmax loss. 

 
 
 by adjusting the margins for different classes adaptively. AdaM-Softmax modified from CosFace is shown as follows: 

 

 
 | 
 L a ​ d ​ a = − log ⁡ e s ⁡ ( cos ⁡ ( θ y i + m y i ) ) e s ⁡ ( cos ⁡ ( θ y i + m y i ) ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) L_{ada}=-\log\frac{e^{s(\cos(\theta_{y_{i}}+m_{y_{i}}))}}{e^{s(\cos(\theta_{y_{i}}+m_{y_{i}}))}+\sum_{j\neq{y_{i}}}e^{s\cos(\theta_{j})}} | 
 | 
 (44) | 
 

 where the m y i m_{y_{i}} is the margin corresponding to class y i y_{i} and it is learnable.
Intuitively, a larger m m is preferred to reduce the intra-class variations. Therefore, the final loss function of AdaptiveFace is: 

 

 
 | 
 L = L a ​ d ​ a + λ ( − 1 N ∑ i m i ) L=L_{ada}+\lambda(-\frac{1}{N}\sum_{i}m_{i}) | 
 | 
 (45) | 
 

 where N N is the number of IDs in the training set; and λ \lambda is positive and controls the strength of the margin constraint. As the training goes by, AdaM-Softmax can adaptively allocate large margins to
poor classes and allocate small margins to rich classes. 

 
 
 Huang et al. [ 67 ] thought the aforementioned large margin loss is usually fail on hard samples.
As a result, they adopted ArcFace to construct a teacher distribution from easy samples and a student distribution from hard samples.
Then a Distribution Distillation Loss (DDL) is proposed to constrain the student distribution to approximate the teacher distribution. DDL can lead to smaller overlap between the positive and negative pairs in the student distribution, which is similar with histogram loss [ 68 ] .
Ustinova et al. [ 68 ] first obtained the histogram of similarity set of positive pairs 𝒮 + \mathcal{S}^{+} and that of negative pairs 𝒮 − \mathcal{S}^{-} in a batch as their distribution. Then histogram loss is formulated by calculating the probability that the similarity of 𝒮 − \mathcal{S}^{-} is greater than the one of 𝒮 + \mathcal{S}^{+} through a discretized integral.
In DDL, Huang et al. divided the training dataset into hard samples ℋ \mathcal{H} and easy samples ℰ \mathcal{E} 
 , and constructed positive and negative pairs in sets ℋ \mathcal{H} and ℰ \mathcal{E} , respectively. Then calculated the distribution of positive pairs similarity H r + H_{r}^{+} and the distribution of negative pairs similarity H r − H_{r}^{-} , and calculated the KL divergence of ℰ \mathcal{E} (as teacher) and ℋ \mathcal{H} (as student) on the positive/negative pairs similarity distribution as loss, namely 

 
 
 
 | 
 ℒ K ​ L \displaystyle\mathcal{L}_{KL} | 
 = λ 1 𝔻 K ​ L ( P + | | Q + ) + λ 2 𝔻 K ​ L ( P − | | Q − ) \displaystyle=\lambda_{1}\mathbb{D}_{KL}(P^{+}||Q^{+})+\lambda_{2}\mathbb{D}_{KL}(P^{-}||Q^{-}) | 
 | 
 (46) | 

 
 | 
 | 
 = λ 1 ​ ∑ s P + ​ ( s ) ​ log ⁡ P + ​ ( s ) Q + ​ ( s ) + λ 2 ​ ∑ s P − ​ ( s ) ​ log ⁡ P − ​ ( s ) Q − ​ ( s ) \displaystyle=\lambda_{1}\sum_{s}P^{+}(s)\log\frac{P^{+}(s)}{Q^{+}(s)}+\lambda_{2}\sum_{s}P^{-}(s)\log\frac{P^{-}(s)}{Q^{-}(s)} | 
 | 
 

 where λ 1 \lambda_{1} and λ 2 \lambda_{2} are the weight coefficients.
 P + P^{+} and P − P^{-} are the similarity distributions of positive and negative pairs in ℰ \mathcal{E} .
 Q + Q^{+} and Q − Q^{-} are the similarity distributions of positive and negative pairs in ℋ \mathcal{H} .
 ℒ K ​ L \mathcal{L}_{KL} may make the distribution of teachers close to the distribution of students, so the author proposes order loss to add a constraint, namely 

 

 
 | 
 ℒ o ​ r ​ d ​ e ​ r = − λ 3 ∑ ( i , j ) ∈ ( p , q ) ( 𝔼 [ 𝒮 i + ] − 𝔼 [ 𝒮 j − ] ) \mathcal{L}_{order}=-\lambda_{3}\sum_{(i,j)\in(p,q)}(\mathbb{E}[\mathcal{S}_{i}^{+}]-\mathbb{E}[\mathcal{S}_{j}^{-}]) | 
 | 
 (47) | 
 

 where 𝒮 p + \mathcal{S}^{+}_{p} and 𝒮 p − \mathcal{S}^{-}_{p} represent the positive pairs and negative pairs of the teacher, respectively, and 𝒮 q + \mathcal{S}^{+}_{q} and 𝒮 q − \mathcal{S}^{-}_{q} represent the student’s positive pairs and negative pairs.
The final form of DDL is the sum of ℒ K ​ L \mathcal{L}_{KL} , ℒ o ​ r ​ d ​ e ​ r \mathcal{L}_{order} and L A ​ r ​ c ​ F ​ a ​ c ​ e L_{ArcFace} . 

 
 
 In equation ( 14 ), ( 15 ), ( 16 ), ( 18 ), ( 20 ), ( 27 ) and ( 30 ), all face embeddings x x and class weights W {W} are normalized and thus distributed on the hypersphere manifold.
And Ring loss [ 55 ] explicitly set
a constrain where the length of embeddings should be as same as possible.
However, many papers have demonstrated that the magnitude of face embedding can measure the quality of the given face.
It can be proven that the magnitude of the feature embedding monotonically increases if the subject is more likely to be recognized. As a result, MagFace [ 69 ] introduced an adaptive mechanism to learn a well-structured within-class feature distributions by pulling easy
samples to class centers with larger magnitudes while pushing hard samples away and shirking their magnitudes. In specific, the loss of MagFace is: 

 

 
 | 
 L i = − log ⁡ e s ​ cos ⁡ ( θ y i + m ⁡ ( a i ) ) e s ​ cos ⁡ ( θ y i + m ⁡ ( a i ) ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) + λ g ​ g ​ ( a i ) L_{i}=-\log\frac{e^{s\cos(\theta_{y_{i}}+m(a_{i}))}}{e^{s\cos(\theta_{y_{i}}+m(a_{i}))}+\sum_{j\neq{y_{i}}}e^{s\cos(\theta_{j})}}+\lambda_{g}g(a_{i}) | 
 | 
 (48) | 
 

 where a i a_{i} is the magnitude of face feature of sample i i without normalization. m ⁡ ( a i ) m(a_{i}) represents a magnitude-aware angular
margin of positive sample i i , which is monotonically increasing.
 g ⁡ ( a i ) g(a_{i}) is a regularizer and designed as a monotonically decreasing convex function.
 m ⁡ ( a i ) m(a_{i}) and g ⁡ ( a i ) g(a_{i}) simultaneously enforce direction and magnitude of face embedding, and λ g \lambda_{g} is a parameter balancing these two factors.
In detail, magnitude a i a_{i} of a high quality face image is large, and m ⁡ ( a i ) m(a_{i}) enforces the embedding x i x_{i} closer to class center W i W_{i} by giving a larger margin; and g ⁡ ( a i ) g(a_{i}) gives a smaller penalty if a i a_{i} is larger. 

 
 
 Circle loss [ 70 ] analyzed that, the traditional loss functions (such as triplet and softmax loss) are all optimizing ( s n − s p ) (s_{n}-s_{p}) distance, where s n s_{n} is inter-class similarity and s p s_{p} is intra-class similarity.
This symmetrical optimization has two problems: inflexible optimization and fuzzy convergence state.
Based on this two facts, Sun et al. proposed Circle loss, where greater penalties are given to similarity scores that are far from the optimal results. 

 
 
 Similar with MagFace, Kim et al. [ 71 ] also emphasize misclassified samples should be adjusted according to their image quality.
They proposed AdaFace to adaptively control gradient changing during back propagation. Kim et al. assume that hard samples should be emphasized when the image quality is high, and vice versa.
As a result, AdaFace is designed as: 

 

 
 | 
 L i = − log ⁡ e s ​ cos ⁡ ( θ y i + g angle ) − g add e s ​ cos ⁡ ( θ y i + g angle ) − g add + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) L_{i}=-\log\frac{e^{s\cos(\theta_{y_{i}}+g_{\textmd{angle}})-g_{\textmd{add}}}}{e^{s\cos(\theta_{y_{i}}+g_{\textmd{angle}})-g_{\textmd{add}}}+\sum_{j\neq y_{i}}e^{s\cos(\theta_{j})}} | 
 | 
 (49) | 
 

 

 
 | 
 g angle = − m ⋅ ‖ z i ‖ ^ , g add = m ⋅ ‖ z i ‖ ^ + m , ‖ z i ‖ ^ = ⌊ ‖ z i ‖ − μ z σ z / h ⌉ − 1 1 g_{\textmd{angle}}=-m\cdot\widehat{\|z_{i}\|},\ \ g_{\textmd{add}}=m\cdot\widehat{\|z_{i}\|}+m,\ \ \widehat{\|z_{i}\|}=\lfloor\frac{\|z_{i}\|-\mu_{z}}{\sigma_{z}/h}\rceil^{1}_{-1} | 
 | 
 (50) | 
 

 where ‖ z i ‖ \|z_{i}\| measures the quality of face i i , and ‖ z i ‖ ^ \widehat{\|z_{i}\|} is normalized quality by using batch statistics μ z \mu_{z} and σ z \sigma_{z} with a factor h h . Similar with Arcface and CosFace, m m represents the angular margin.
AdaFace can be treated as the generalization of Arcface and CosFace: when ‖ z i ‖ ^ = − 1 \widehat{\|z_{i}\|}=-1 , function ( 49 ) becomes ArcFace; when ‖ z i ‖ ^ = 0 \widehat{\|z_{i}\|}=0 , it becomes CosFace. 

 
 
 At the end of this subsection, we introduce a FR model quantization method, which was specially designed for large margin based loss.
Traditionally, the quantization error for feature 𝒇 i \boldsymbol{f}_{i} is defined as follows: 

 

 
 | 
 QE ⁡ ( 𝒇 i ) = 1 d ​ ∑ l = 1 d ( 𝒇 i l − Q ⁡ ( 𝒇 i l ) ) 2 {\rm QE}(\boldsymbol{f}_{i})=\frac{1}{d}\sum_{l=1}^{d}(\boldsymbol{f}_{i}^{l}-Q(\boldsymbol{f}_{i}^{l}))^{2} | 
 | 
 (51) | 
 

 where 𝒇 i \boldsymbol{f}_{i} and Q ⁡ ( 𝒇 i ) Q(\boldsymbol{f}_{i}) denote a full precision (FP) feature and its quantization. d d and the superscript l l represent the length of features and the l l -th dimension.
Wu et al. [ 72 ] redefined the quantization error (QE) of face feature as the angle between its FP feature and its quantized feature : 

 

 
 | 
 AQE ⁡ ( 𝒇 i ) = arccos ⁡ ( ⟨ 𝒇 i ‖ 𝒇 i ‖ 2 , Q ~ ​ ( 𝒇 i ) ‖ Q ~ ​ ( 𝒇 i ) ‖ 2 ⟩ ) {\rm AQE}(\boldsymbol{f}_{i})=\arccos\left(\left \frac{\boldsymbol{f}_{i}}{\left\|\boldsymbol{f}_{i}\right\|_{2}},\frac{{\tilde{Q}}(\boldsymbol{f}_{i})}{\left\|{\tilde{Q}}(\boldsymbol{f}_{i})\right\|_{2}}\right \right) | 
 | 
 (52) | 
 

 
 
 Wu et al. [ 72 ] believed that for each sample, quantization error AQE \rm AQE can be divided into two parts: error caused by the category center of the sample after quantization (class error), and error caused by the sample deviating from the category center due to quantization (individual error), namely 

 

 
 | 
 AQE ⁡ ( 𝒇 i ) = AQE ⁡ ( 𝒄 y i ) + ℐ ⁡ ( 𝒇 i ) {\rm AQE}(\boldsymbol{f}_{i})={\rm AQE}(\boldsymbol{c}_{y_{i}})+\mathcal{I}(\boldsymbol{f}_{i}) | 
 | 
 (53) | 
 

 The former term (class error of class y i y_{i} ) will not affect the degree of compactness within the class, while the latter term (individual error) will. Therefore, the individual error should be mainly optimized. They introduced the individual error into CosFace as an additive angular margin, named rotation consistent margin (RCM): 

 

 
 | 
 ℒ i = − log ⁡ exp ⁡ ( s ⋅ cos ⁡ ( θ i , j + δ ⁡ ( j = y i ) ⋅ m + δ ⁡ ( j = y i ) ⋅ λ ​ θ Q ) CLOSE ∑ j exp ⁡ ( s ⋅ cos ⁡ ( θ i , j + δ ⁡ ( j = y i ) ⋅ m + δ ⁡ ( j = y i ) ⋅ λ ​ θ Q ) CLOSE \mathcal{L}_{i}=-\log\frac{\exp(s\cdot\cos(\theta_{i,j}+\delta(j=y_{i})\cdot m+\delta(j=y_{i})\cdot\lambda\theta_{Q})}{\sum_{j}\exp(s\cdot\cos(\theta_{i,j}+\delta(j=y_{i})\cdot m+\delta(j=y_{i})\cdot\lambda\theta_{Q})} | 
 | 
 (54) | 
 

 

 
 | 
 θ Q = ‖ ℐ ⁡ ( 𝒇 i ) ‖ = ‖ AQE ⁡ ( 𝒇 i ) − AQE ⁡ ( 𝒄 y i ) ‖ \theta_{Q}=\|\mathcal{I}(\boldsymbol{f}_{i})\|=\|{\rm AQE}(\boldsymbol{f}_{i})-{\rm AQE}(\boldsymbol{c}_{y_{i}})\| | 
 | 
 (55) | 
 

 where δ ⁡ ( j = y i ) \delta(j=y_{i}) is the indicative function, where j = y i j=y_{i} gives its value 1, otherwise 0. 

 
 
 

#### 4.1.3 FR in unbalanced training data

 
 Large-scale face datasets usually exhibit a massive number of classes with unbalanced distribution.
Features with non-dominate IDs are compressed into a small area in the hypersphere, leading to training problems.
Therefore, for different data unbalance phenomena, different methods were proposed. 

 
 
 The first data unbalance phenomenon is long tail distributed, which widely exists in the mainstream training set, such as MS-Celeb-1M.
In MS-Celeb-1M dataset, the number of face images per person falls drastically, and only a small part of persons have large number of images.
Zhang et al. [ 73 ] set an experiment to show that including all tail data in training can not help to obtained a better FR model by contrastive loss [ 41 ] , triplet loss [ 34 ] , and center loss [ 45 ] . Therefore, the loss needs to be delicately designed. 

 
 
 Inspired by contrastive loss, range loss [ 73 ] was designed to penalize intra-personal variations especially for
infrequent extreme deviated value, while enlarge the inter-personal differences simultaneously.
The range loss is shown as: 

 

 
 | 
 L = L s ​ o ​ f ​ t ​ m ​ a ​ x + α ​ L R i ​ n ​ t ​ r ​ a + β ​ L R i ​ n ​ t ​ e ​ r L=L_{softmax}+\alpha L_{R_{intra}}+\beta L_{R_{inter}} | 
 | 
 (56) | 
 

 where α \alpha and β \beta are two weights, L R i ​ n ​ t ​ r ​ a L_{R_{intra}} denotes the intra-class loss and L R i ​ n ​ t ​ e ​ r L_{R_{inter}} represents the inter-class loss.
 L R i ​ n ​ t ​ r ​ a L_{R_{intra}} penalizes the maximum harmonic range within each class: 

 

 
 | 
 L R i ​ n ​ t ​ r ​ a = ∑ i ∈ I L R i ​ n ​ t ​ r ​ a i = ∑ i ∈ I k ∑ j = 1 k 1 D j L_{R_{intra}}=\sum_{i\in I}L^{i}_{R_{intra}}=\sum_{i\in I}\frac{k}{\sum_{j=1}^{k}\frac{1}{D_{j}}} | 
 | 
 (57) | 
 

 where I I denotes the complete set of identities in current mini-batch, and D j D_{j} is the j j -th largest Euclidean distance between all samples with ID i i in this mini-batch.
Equivalently, the overall cost is the harmonic mean of the first k-largest ranges within each class, and k k is set to 2 in the experiment.
 L R i ​ n ​ t ​ e ​ r L_{R_{inter}} represents the inter-class loss that 

 

 
 | 
 L R i ​ n ​ t ​ e ​ r = max ⁡ ( M − D c ​ e ​ n ​ t ​ e ​ r , 0 ) = max ⁡ ( M − ‖ x ¯ Q − x ¯ R ‖ 2 2 , 0 ) L_{R_{inter}}=\max(M-D_{center},0)=\max(M-\|\overline{x}_{Q}-\overline{x}_{R}\|^{2}_{2},0) | 
 | 
 (58) | 
 

 where D C ​ e ​ n ​ t ​ e ​ r D_{Center} is the shortest distance between the centers of two classes,
and M M is the max optimization margin of D C ​ e ​ n ​ t ​ e ​ r D_{Center} .
 Q Q and R R are the two nearest classes within the current mini-batch, while x ¯ Q \overline{x}_{Q} and x ¯ R \overline{x}_{R} represents their centers. 

 
 
 Zhong et al. [ 74 ] first adopted a noise resistance (NR) loss based on large margin loss to train on head data, which is shown as follows: 

 

 
 | 
 L N ​ A ​ S ​ B ​ ( i ) = − [ α ⁡ ( P y i ​ p ) ​ log ⁡ ( P y i ) + β ⁡ ( P y i ) ​ log ⁡ ( P y i ​ p ) ] L_{NASB}(i)=-[\alpha(P_{y_{ip}})\log(P_{y_{i}})+\beta(P_{y_{i}})\log(P_{y_{ip}})] | 
 | 
 (59) | 
 

 where y i y_{i} is the training label and y i ​ p y_{ip} is the current predict
label.
 P y i ​ p P_{y_{ip}} is the predict probability of training label class and
 P y i P_{y_{i}} is that of the current predict class. Namely: 

 

 
 | 
 y i ​ p = arg ⁡ max y i ⁡ e W j T ​ x i + b j ∑ k e W k T ​ x i + b k y_{ip}=\arg\max_{y_{i}}\frac{e^{W_{j}^{T}x_{i}+b_{j}}}{\sum_{k}e^{W_{k}^{T}x_{i}+b_{k}}} | 
 | 
 (60) | 
 

 

 
 | 
 P y i = e W y i T ​ x i + b y i ∑ k e W k T ​ x i + b k , P y i ​ p = e W y i ​ p T ​ x i + b y i ​ p ∑ k e W k T ​ x i + b k P_{y_{i}}=\frac{e^{W_{y_{i}}^{T}x_{i}+b_{y_{i}}}}{\sum_{k}e^{W_{k}^{T}x_{i}+b_{k}}}\ ,\ P_{y_{ip}}=\frac{e^{W_{y_{ip}}^{T}x_{i}+b_{y_{ip}}}}{\sum_{k}e^{W_{k}^{T}x_{i}+b_{k}}} | 
 | 
 (61) | 
 

 α ⁡ ( P ) \alpha(P) and β ⁡ ( P ) \beta(P) control the degree of combination: 

 

 
 | 
 α ⁡ ( P ) = { ρ , P t 0 , P ≤ 0 , β ⁡ ( P ) = { 1 − ρ , P t 0 , P ≤ 0 \alpha(P)=\left\{\begin{array}[]{ll}\rho, P t\\
0, P\leq 0\end{array}\right.\ ,\ \beta(P)=\left\{\begin{array}[]{ll}1-\rho, P t\\
0, P\leq 0\end{array}\right. | 
 | 
 (62) | 
 

 The NR loss ( 59 ) can be further modified to CosFace [ 50 ] or ArcFace [ 51 ] forms.
After a relatively discriminative model have been learned on the head data by L N ​ A ​ S ​ B L_{NASB} , center-dispersed loss is employed to deal with the tail data. It extracts features of tail identities using the base model supervised by head data;
then add the tail data gradually in an iterative way and disperse these identities in the feature space so that we can take full advantage of their modest but indispensable information.
To be more specifically, Center-dispersed Loss can be formulated as: 

 

 
 | 
 L C ​ D = min ⁡ 1 m ⁡ ( m − 1 ) ​ ∑ 1 ≤ i j ≤ m S i , j 2 , S i , j = ( C i ‖ C i ‖ ) T ​ ( C i ‖ C i ‖ ) L_{CD}=\min\frac{1}{m(m-1)}\sum_{1\leq i j\leq m}S_{i,j}^{2}\ \ ,\ \ S_{i,j}=(\frac{C_{i}}{\|C_{i}\|})^{T}(\frac{C_{i}}{\|C_{i}\|}) | 
 | 
 (63) | 
 

 where S i , j S_{i,j} is the similarity between identity i i and j j in mini-batch,
and the most hard m m identities are mined from a candidate bag to construct a tail data mini-batch for efficiency.
 C i C_{i} and C j C_{j} represent normalized features centers of identity i i and j j , which can be relatively robust even to moderate noise. 

 
 
 The second data unbalance phenomenon is shallow data.
In many real-world scenarios of FR, the training dataset is limited in depth, i.e. only small number of samples are available for most IDs.
By applying softmax loss or its angular modification loss (such as CosFace [ 50 ] ) on shallow training data, results are damaged by the model degeneration and over-fitting issues.
Its essential reason consists in feature space collapse [ 75 ] . 

 
 
 Li et al. [ 76 ] proposed a concept of virtual class to give the unlabeled data a virtual identity in mini-batch, and treated these virtual classes as negative classes.
Since the unlabeled data is shallow such that it is hard to find samples from the same identity in a mini-batch, each unlabeled feature can be a substitute to represent the centroid of its virtual class.
As a result, by adding a virtual class term into the large margin based loss (such as ArcFace [ 51 ] ), the loss has: 

 

 
 | 
 L i = − log ⁡ e s ⁡ ( c ​ o ​ s ​ ( θ y i + m ) ) e s ⁡ ( cos ⁡ ( θ y i + m ) ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) + ∑ u = 1 U e s ​ cos ⁡ ( θ u ) L_{i}=-\log\frac{e^{s(cos(\theta_{y_{i}}+m))}}{e^{s(\cos(\theta_{y_{i}}+m))}+\sum_{j\neq{y_{i}}}e^{s\cos(\theta_{j})}+\sum_{u=1}^{U}e^{s\cos(\theta_{u})}} | 
 | 
 (64) | 
 

 where U U is the number of unlabeled shallow data in mini-batch, and θ u \theta_{u} is the angular between embedding x i x_{i} and x u x_{u} , which is also the centroid of virtual class u u .
For the purpose of exploiting more potential of the unlabeled shallow data, a feature generator was designed to output more enhanced feature from unlabeled data.
More details about the generator can be check in [ 76 ] . 

 
 
 The above methods solve data unbalance problem in class-level therefore treated the images from the same person with equal importance. However, Liu et al. [ 77 ] thought that they have different importance and utilized meta-learning to re-weighted each sample based on multiple variation factors. Specifically, it updated four learnable margins, each corresponding to a variation factor during training. 

 
 
 The factors included ethnicity, pose, blur, and occlusion. 

 
 
 
 

### 4.2 Embedding

 
 Different from designing delicate losses in the last subsection, embedding refinement is another way to enhance FR results.
The first idea of embedding refinement set a explicit constraint on face embeddings with a face generator.
The second idea changed the face embedding with auxiliary information from training images, such as occlusion and resolution.
The third idea models FR in a multi-task way. Extra tasks such as age and pose prediction were added in the network. 

 
 

#### 4.2.1 Embedding refinement by face generator

 
 FR methods based on face generator usually focus on age or pose invariant FR problems. DR-GAN [ 78 ] solved pose-invariant FR by synthesizing faces with different poses. DR-GAN learned an identity representation for a face image by an encoder-decoder structured generator.
And the decoder output synthesized various faces of the same ID with different poses.
Given identity label y d y^{d} and pose label y p y^{p} of an face x x , the encoder G e ​ n ​ c G_{enc} first extract its pose-invariant identity representation f ​ ( x ) = G e ​ n ​ c ​ ( x ) f(x)=G_{enc}(x) . Then f ⁡ ( x ) f(x) is concatenated with a pose code c c and a random noise z z . The decoder G d ​ e ​ c G_{dec} generates the synthesized face image x ^ = G d ​ e ​ c ​ ( f ⁡ ( x ) , c , z ) \hat{x}=G_{dec}(f(x),c,z) with the same identity y d y^{d} but a different pose specified by a pose code c c .
Given a synthetic face image from the generator, the discriminator D D attempts to estimate its identity and pose, which classifies x ^ \hat{x} as fake. 

 
 
 Liu et al. [ 79 ] proposed an identity Distilling and Dispelling Auto-encoder (D2AE) framework that adversarially learnt the identity-distilled features for identity verification. The structure of D2AE is shown in Fig. 5 .
The encoder E θ e ​ n ​ c E_{\theta_{enc}} in D2AE extracted a feature of an input image x x , which was followed by
the parallel identity distilling branch B θ T B_{\theta_{T}} and identity dispelling branch B θ P B_{\theta_{P}} .
The output f T = B θ T ​ ( E θ e ​ n ​ c ​ ( x ) ) f_{T}=B_{\theta_{T}}(E_{\theta_{enc}}(x)) and f P = B θ P ​ ( E θ e ​ n ​ c ​ ( x ) ) f_{P}=B_{\theta_{P}}(E_{\theta_{enc}}(x)) are identity-distilled feature and identity-dispelled feature.
 f T f_{T} predicted the face ID of x x by optimizing a softmax loss.
On the contrary, B θ P B_{\theta_{P}} needs to fool the identity classifier, where the so-called
“ground truth” identity distribution is required to be constant over all identities and equal to 1 N I ​ D \frac{1}{N_{ID}} 
( N I ​ D N_{ID} is the number of IDs in training set).
Thus, f P f_{P} has a loss: 

 

 
 | 
 L H = 1 N I ​ D ​ ∑ j = 1 N I ​ D log ⁡ y P j L_{H}=\frac{1}{N_{ID}}\sum_{j=1}^{N_{ID}}\log y_{P}^{j} | 
 | 
 (65) | 
 

 where y P j y_{P}^{j} is the logits of this classifier with index j j . The gradients for L H L_{H} are back-propagated to B θ P B_{\theta_{P}} and E θ e ​ n ​ c E_{\theta_{enc}} with this identity classifier fixed.
At last, an decoder D θ d ​ e ​ c D_{\theta_{dec}} is used to further enhance f T f_{T} and f P f_{P} by imposing a bijective mapping between an input image x x and its semantic features, with a reconstruction loss: 

 

 
 | 
 L X = 1 2 ​ ‖ x − D θ d ​ e ​ c ​ ( f T , f P ) ‖ 2 2 L_{X}=\frac{1}{2}\|x-D_{\theta_{dec}}(f_{T},f_{P})\|^{2}_{2} | 
 | 
 (66) | 
 

 As shown in Fig. 5 , f ~ T \tilde{f}_{T} and f ~ P \tilde{f}_{P} are the augmented feature of f T f_{T} and f P f_{P} by adding Gaussian noise on them, which can also be employed to train the decoder. The generated image with f ~ T \tilde{f}_{T} and f ~ P \tilde{f}_{P} needed to preserve the ID of input image x x . 

 
 
 Figure 5: The architectures of R3AN. 
 
 
 Chen et al. [ 80 ] brought forth cross model FR (CMFR) problem, and proposed R3AN to solve it. CMFR is defined as recognizing feature extracted from one model with another model’s gallery. Chen et al. built a encoder to generate a face image of the source feature (from FR model 1), and trained a encoder to match with target feature (in FR model 2).
The architecture in R3AN is shown in Fig. 6 .
The training of R3AN has 3 parts: reconstruction, representation and regression.
In reconstruction, a generator G G is trained by source feature X X and its related face image I I .
The reconstruction loss is formulated as follows: 

 

 
 | 
 L R ​ e ​ c ​ ( G ) = 𝔼 X , I ​ [ ‖ I − G ⁡ ( X ) ‖ 2 ] L_{Rec}(G)=\mathbb{E}_{X,I}[\|I-G(X)\|_{2}] | 
 | 
 (67) | 
 

 Similar with GAN, the generated face image is needed adversarial learning L A ​ d ​ v L_{Adv} to become as real as possible.
In representation, a encoder E E is trained, which takes the original face image I I as input and learns representation Y Y of the target model.
An L2 loss is adopted to supervise the training of representation module as follows: 

 

 
 | 
 L R ​ e ​ p ​ ( E ) = 𝔼 I , Y ​ [ ‖ Y − E ⁡ ( I ) ‖ 2 ] L_{Rep}(E)=\mathbb{E}_{I,Y}[\|Y-E(I)\|_{2}] | 
 | 
 (68) | 
 

 Finally, regression module synchronizes the G G and E E in our feature-to-feature learning framework, and it maps source X X to target Y Y . As a result, the regression loss exists is expressed as: 

 

 
 | 
 L R ​ e ​ g ​ ( G , E ) = 𝔼 X , Y ​ [ ‖ Y − E ⁡ ( G ⁡ ( X ) ) ‖ 2 ] L_{Reg}(G,E)=\mathbb{E}_{X,Y}[\|Y-E(G(X))\|_{2}] | 
 | 
 (69) | 
 

 
 
 Figure 6: The architectures of R3AN. 
 
 
 Huang et al. [ 81 ] proposed MTLFace to jointly learn age-invariant FR and face age synthesis.
Fig. 7 depicts the pipeline of this method.
First, an encoder E E extracts face feature X X of a image I I , which is further fed into attention-based feature decomposition (AFD) module. AFD decomposes the feature X X into age related feature X a ​ g ​ e X_{age} and identity related feature X i ​ d X_{id} by attention mask, namely: 

 

 
 | 
 X = X a ​ g ​ e + X i ​ d = X ∘ σ ⁡ ( X ) + X ∘ ( 1 − σ ⁡ ( X ) ) X=X_{age}+X_{id}=X\circ\sigma(X)+X\circ(1-\sigma(X)) | 
 | 
 (70) | 
 

 where ∘ \circ denotes element-wise multiplication and σ \sigma represents an attention module which is composed of channel and spacial attention modules.
Then X a ​ g ​ e X_{age} and X i ​ d X_{id} are used for age estimation and age-invariant FR (AIFR).
In AIFR, CosFace [ 50 ] supervises the learning of X i ​ d X_{id} .
In addition, a cross-age domain adversarial learning is proposed to encourage X i ​ d X_{id} to be age-invariant with a gradient reversal layer (GRL) [ 82 ] . The loss for AIFR is formulated as: 

 

 
 | 
 L A ​ I ​ F ​ R = L C ​ o ​ s ​ F ​ a ​ c ​ e ​ ( X i ​ d ) + λ 1 ​ L A ​ E ​ ( X a ​ g ​ e ) + λ 2 ​ L A ​ E ​ ( G ​ R ​ L ​ ( X i ​ d ) ) L^{AIFR}=L_{CosFace}(X_{id})+\lambda_{1}L_{AE}(X_{age})+\lambda_{2}L_{AE}(GRL(X_{id})) | 
 | 
 (71) | 
 

 where L A ​ E L_{AE} is the age estimation loss, which contains age value regression and age group classification.
In order to achieve a better performance of AIFR, MTLFace designed a decoder to generate synthesized faces to modify X i ​ d X_{id} explicitly.
In face age synthesis (FAS), X i ​ d X_{id} is first fed into identity condition module to get a new feature with age information (age group t t ). Then a decoder is utilized to generate synthesized face I t I_{t} .
In order to make sure I t I_{t} get correct ID and age information, I t I_{t} is also fed into the encoder and AFD. The final loss of the generator becomes: 

 

 
 | 
 X a ​ g ​ e t , X i ​ d t = A ​ F ​ D ​ ( E ⁡ ( I t ) ) , L a ​ g ​ e F ​ A ​ S = L C ​ E ​ ( X a ​ g ​ e t , t ) , L i ​ d F ​ A ​ S = 𝔼 ​ ‖ X i ​ d t − X i ​ d ‖ F 2 X_{age}^{t},X_{id}^{t}=AFD(E(I_{t}))\ ,\ \ L^{FAS}_{age}=L_{CE}(X_{age}^{t},t)\ ,\ \ L^{FAS}_{id}=\mathbb{E}\|X_{id}^{t}-X_{id}\|^{2}_{F} | 
 | 
 (72) | 
 

 where L a ​ g ​ e F ​ A ​ S L^{FAS}_{age} constrains I t I_{t} has age t t , where L C ​ E L_{CE} is cross-entropy loss. L i ​ d F ​ A ​ S L^{FAS}_{id} encourages the identity related features of I I and I t I_{t} to get closer, where ∥ ∥ ˙ F \|\dot{\|}_{F} denotes the Frobenius norm.
Finally, MTLFace builds a discriminator to optimize I t I_{t} to get a real looking appearance. 

 
 
 Figure 7: An overview of MTLFace. 
 
 
 Uppal et al. [ 83 ] proposed the Teacher-Student Generative Adversarial Network (TS-GAN) to generate depth images from single RGB images in order to boost the performance of FR systems.
TS-GAN includes a teacher and a student component.
The teacher, which consists of a generator and a discriminator, aims to learn a latent mapping between RGB channel and
depth from RGB-D images.
The student refines the learned mapping for RGB images by further training the generator.
While training FR, the model inputs a RGB image and its generates depth image, and extracts their features independently.
Then the final face embedding by fusing RGB and depth features is used to predict face ID. 

 
 
 [ 84 ] learned facial representations from unlabeled facial images by generating face with de-expression. The architecture of the proposed model is shown in Fig. 8 . 

 
 
 Figure 8: The architecture of [ 84 ] . 
 
 
 In this method, a face image F F can be decomposed as F = F ~ + id + exp = F ^ + exp F=\tilde{F}+\textmd{id}+\textmd{exp}=\hat{F}+\textmd{exp} ,
where F ~ \tilde{F} is the global mean face shared among all the faces, and F ^ \hat{F} is the neutral face of a particular identity specified by id.
id and exp are the identity and expression factors respectively.
In order to get F ~ \tilde{F} and F ^ \hat{F} , a image F F is first used to extract expression and identity representations by using a unsupervised disentangling method.
By exploring the disentangled representations, networks D f ​ l ​ o ​ w D_{flow} , D e ​ x ​ p D_{exp} , MLPs and D i ​ d D_{id} 
are trained to generate the representation-removed images F ~ \tilde{F} and F ^ \hat{F} , and to reconstruct the representation-added images, the input face F ′ F^{{}^{\prime}} and the neutral face F ^ ′ \hat{F}^{{}^{\prime}} . 

 
 
 

#### 4.2.2 Embedding refinement by extra representations

 
 Both [ 85 ] and [ 86 ] considered face embedding as a low-rank representation problem.
Their frameworks aim at adding noise into images, which can be divided into face features linear reconstruction (from a dictionary) and sparsity constraints. 

 
 
 Neural Aggregation Network (NAN) [ 87 ] is a typical video FR method by manipulating face embeddings. Yang et al. proposed that, face images (in a video) with a same ID should be merged to build one robust embedding. Given a set of features with a same ID from one video { f i | i = 1 , 2 , … , K } \{f_{i}\lvert i=1,2,\dots,K\} , it will be merged into an embedding r r by weighted summation as: 

 

 
 | 
 r = ∑ i = 1 K a i ​ f i , ∑ i a i = 1 r=\sum_{i=1}^{K}a_{i}f_{i}\ ,\ \sum_{i}a_{i}=1 | 
 | 
 (73) | 
 

 where a i a_{i} is weight for feature f i f_{i} . And a i a_{i} is learned by a neural aggregation network, which is based on stacking attention blocks. 

 
 
 He et al. [ 88 ] proposed the Dynamic Feature Matching method to address partial face images due to occlusion or large pose.
First, a fully convolutional network is adopted to get features from probe and gallery face image with arbitrary size, which are denoted as p p and g c g_{c} ( c c is the label of the gallery image).
Normally, it fails to compute the similarity of p p and g c g_{c} on account of feature dimension-inconsistent.
As a result, a sliding window of the same size as p p is used to decompose g c g_{c} into k k sub-feature maps G c = [ g c 1 , g c 2 , … , g c k ] G_{c}=[g_{c_{1}},g_{c_{2}},\dots,g_{c_{k}}] .
Then the coefficients w c w_{c} of p p with respect to G c G_{c} is computed by following loss: 

 

 
 | 
 L ⁡ ( w c ) = y c ​ ( ‖ p − G c ​ w c ‖ 2 2 − α ​ p T ​ G c ​ w c ) + β ​ ‖ w c ‖ 1 L(w_{c})=y_{c}(\|p-G_{c}w_{c}\|^{2}_{2}-\alpha p^{T}G_{c}w_{c})+\beta\|w_{c}\|_{1} | 
 | 
 (74) | 
 

 where term p T ​ G c ​ w c p^{T}G_{c}w_{c} is the similarity-guided constraint, and term β ​ ‖ w c ‖ 1 \beta\|w_{c}\|_{1} is a l1 regularizer. α \alpha and β \beta are constants that control the strength of these two constraints. y ​ c = { 1 , − 1 } yc=\{1,-1\} means that p p and G c G_{c} are from the same identity or not. 

 
 
 Zhao et al. [ 89 ] proposed the Pose Invariant Model (PIM) for FR in the wild, by the Face Frontalization Sub-Net (FFSN).
First, a face image is input into a face landmark detector to get its landmark patches. The input of PIM are profile face images with four landmark patches, which are collectively denoted as I t ​ r I_{tr} . Then the recovered frontal face is I ′ = G ⁡ ( I t ​ r ) I^{\prime}=G(I_{tr}) , where G G is a encoder-decoder structure.
Similar with traditional GAN, a discriminative learning sub-net is further connected to the FFSN, to make sure I ′ I^{\prime} visually resemble a real face with identity information.
As a result, features from a profile face with its generated frontal face will be used to get a better face representation.
An overview of the PIM framework is shown in Fig. 9 

 
 
 Figure 9: An overview of the PIM framework. 
 
 
 Yin et al. [ 90 ] proposed a Feature Activation Diversity (FAD) loss to enforce face representations to be insensitive to local changes by occlusions.
A siamese network is first constructed to learn face representations from two faces: one with synthetic occlusion I I and one without I I . 

 
 
 Wang et al. [ 91 ] solved age-invariant FR problem by factorizing a mixed face feature into
two uncorrelated components: identity-dependent component and age-dependent component. The identity dependent component includes information that is useful for FR, and age-dependent component is treated as a distractor in the problem of FR.
The network proposed in [ 91 ] is shown in Fig. 10 . 

 
 
 Figure 10: An overview of the proposed method of [ 91 ] . 
 
 
 The initial feature x x of a face image is extracted by a backbone net F F , followed by the residual factorization module.
The two factorized components x i ​ d x_{id} (identity-dependent component) and x a ​ g ​ e x_{age} (age-dependent component), where x = x i ​ d + x a ​ g ​ e x=x_{id}+x_{age} .
 x a ​ g ​ e x_{age} is obtained through a mapping function R R ( x a ​ g ​ e = R ⁡ ( x ) x_{age}=R(x) ),
and the residual part is regarded as x i ​ d x_{id} ( x i ​ d = x − R ⁡ ( x ) x_{id}=x-R(x) )
Then, x i ​ d x_{id} and x a ​ g ​ e x_{age} are used for face ID classification and age prediction.
In addition, a Decorrelated Adversarial Learning (DAL) regulizer is designed to reduce the correlation between the decomposed features, namely: 

 

 
 | 
 L D ​ A ​ L = | ρ | = | C ​ o ​ v ​ ( v i ​ d , v a ​ g ​ e ) V ​ a ​ r ​ ( v i ​ d ) ​ V ​ a ​ r ​ ( v a ​ g ​ e ) | L_{DAL}=\lvert\rho\rvert=\lvert\frac{Cov(v_{id},v_{age})}{\sqrt{Var(v_{id})Var(v_{age})}}\rvert | 
 | 
 (75) | 
 

 

 
 | 
 v i ​ d = c ⁡ ( x i ​ d ) = w i ​ d T ​ x i ​ d , v a ​ g ​ e = c ⁡ ( x a ​ g ​ e ) = w a ​ g ​ e T ​ x a ​ g ​ e v_{id}=c(x_{id})=w_{id}^{T}x_{id}\ ,\\
v_{age}=c(x_{age})=w_{age}^{T}x_{age} | 
 | 
 (76) | 
 

 where variables v i ​ d v_{id} and v a ​ g ​ e v_{age} are mappings from x i ​ d x_{id} and x a ​ g ​ e x_{age} by a linear Canonical Mapping Module, and w i ​ d w_{id} and x a ​ g ​ e x_{age} are the learning parameters for canonical mapping.
 ρ \rho is the correlation coefficient. A smaller value of | ρ | \lvert\rho\rvert represents the irrelevance between v i ​ d v_{id} and v a ​ g ​ e v_{age} , which means x a ​ g ​ e x_{age} is decoupled with x i ​ d x_{id} successfully. 

 
 
 Yin et al. [ 92 ] proposed a feature transfer framework to augment the feature space of under-represented subjects with less samples from the regular subjects with sufficiently diverse samples.
The network (which can be viewed in Fig. 11 ) is trained with an alternating bi-stage strategy.
At first stage, an encoder E ​ n ​ c Enc is fixed and generates feature g g of a image x x . Then the feature transfer G G of under-represented subjects is applied to generate new feature samples g ~ \tilde{g} that are more diverse.
These original and new features g g and g ~ \tilde{g} of under-represented subjects will be used to reshape the decision boundary.
Then a filtering network R R is applied to generate discriminative identity features f = R(g) that are fed to a linear
classifier F ​ C FC with softmax as its loss.
In stage two, we fix the FC, and update all the other models.
As a result, the samples that are originally on or across the boundary are pushed towards their center.
Also, while training the encoder, a decoder D ​ e ​ c Dec is added after feature g g to recover the image x x with L2 loss. 

 
 
 Figure 11: An overview of the proposed method of [ 91 ] . 
 
 
 PFE [ 93 ] proposed that, an image x i x_{i} should have an ideal embedding f ⁡ ( x i ) f(x_{i}) representing its identity and less unaffected by any identity irrelevant information. The n ⁡ ( x i ) n(x_{i}) is the uncertainty information of x i x_{i} in the embedding space.
So, the embedding predicted by DNNs can reformulated as z i = f ⁡ ( x i ) + n ⁡ ( x i ) z_{i}=f(x_{i})+n(x_{i}) , where z i z_{i} can be defined as a Gaussian distribution: p ( z i | x i ) = N ( z i ; μ i , σ i 2 I ) p(z_{i}\lvert x_{i})=N(z_{i};\mu_{i},\sigma^{2}_{i}I) .
Inspired by [ 93 ] , [ 94 ] proposed Data Uncertainty Learning (DUL) to extract face feature (mean) and its uncertainty (variance) simultaneously.
Specifically, DUL first sample a random noise ϵ \epsilon from a normal distribution,
and then generate s i s_{i} as the equivalent sampling representation:
 s i = μ i + ϵ ​ σ i , ϵ ∼ N ⁡ ( 0 , I ) s_{i}=\mu_{i}+\epsilon\sigma_{i},\epsilon\sim N(0,I) .
As a result, a classification loss L s ​ o ​ f ​ t ​ m ​ a ​ x L_{softmax} can be used to optimize representation s i s_{i} .
In addition, an regularization term L k ​ l L_{kl} explicitly constrains N ⁡ ( μ i , ϵ i ) N(\mu_{i},\epsilon_{i}) to be close to a normal distribution, N ⁡ ( 0 , I ) N(0,I) , measured by Kullback-Leibler divergence (KLD).
Therefore, the final loss became L = L s ​ o ​ f ​ t ​ m ​ a ​ x + λ ​ L k ​ l L=L_{softmax}+\lambda L_{kl} , where: 

 

 
 | 
 L k ​ l = K L [ N ( z i | μ i , σ i 2 ) | | N ( ϵ i | 0 , I ) ] = − 1 2 ( 1 + log σ 2 − μ 2 − σ 2 ) L_{kl}=KL[N(z_{i}\lvert\mu_{i},\sigma_{i}^{2})\lvert\rvert N(\epsilon_{i}\lvert 0,I)]=-\frac{1}{2}(1+\log\sigma^{2}-\mu^{2}-\sigma^{2}) | 
 | 
 (77) | 
 

 
 
 [ 95 ] pointed out the failure of PFE theoretically, and addressed its issue by extending the von Mises Fisher density to its r-radius counterpart and deriving a new optimization objective in closed form. 

 
 
 Shi et al. [ 96 ] proposed a universal representation learning method for FR.
In detail, a high-quality data augumentation method was used according to pre-defined variations such as blur, occlusion and pose.
The feature representation extracted by a backbone is then split into sub-embeddings associated with sample-specific confidences.
Confidence-aware identification loss and variation decorrelation loss are developed to learn the sub-embeddings.
In specific, confidence-aware identification loss is similar with the CosFace, where the confidences of each image are used as scale parameter in the Equation ( 15 ). The framework is shown in Fig. 12 . 

 
 
 Figure 12: The framework of the proposed method. 
 
 
 [ 97 ] proposed a hierarchical pyramid diverse attention (HPDA) network to learn multi-scale diverse local representations adaptively.
Wang et al. observed that face local patches played important roles in FR when the global face appearance changed dramatically.
A face feature is first extracted by a stem CNN, then fed into a global CNN with global average pooling and fully connected layer to get global feature.
At the same time, Local CNNs are developed to extract multi-scale diverse local features hierarchically.
Local CNNs mainly consist of a pyramid diverse attention (PDA) and a hierarchical bilinear pooling (HBP).
The PDA aims at learning local features at different scales by the Local Attention Network [ 98 ] .
The HBP aggregates local information from hierarchical layers to obtain a more comprehensive local representation.
At last, local and global features are concatenated together to get the final rich feature for classification.
The framework of HPDA is shown in Fig. 13 . 

 
 
 Figure 13: The framework of the proposed hierarchical pyramid diverse attention (HPDA) model. 
 
 
 In order to handle long-tail problem [ 73 , 74 ] in FR, [ 99 ] proposed the Domain Balancing (DB) mechanism to obtain more discriminative features of long-tail samples, which contains three main modules: the Domain Frequency Indicator (DFI), the Residual Balancing Mapping (RBM) and the Domain Balancing Margin (DBM).
The DFI is designed to judge whether a sample is from head domains or tail domains, based on the inter-class compactness.
The classes with smaller compactness (larger DFI value) are more likely to come from a tail domain and should be relatively upweighted. DFI value is calculated by the weights in the final classifier.
The light-weighted RBM block is applied to balance the domain distribution.
RBM block contains a soft gate f ⁡ ( x ) f(x) and a feature enhancement module R ⁡ ( x ) R(x) . f ⁡ ( x ) f(x) is used to measure the feature x x depending on DFI value and R ⁡ ( x ) R(x) is a boost for feature x x when it comes from tail samples.
In RBM block, if feature x x probably belongs to a tail class, a large enhancement is assigned to the output rebalancing feature x r ​ e x_{re} , namely x r ​ e = x + f ⁡ ( x ) ​ R ​ ( x ) x_{re}=x+f(x)R(x) .
Finally, the DBM in the loss function to further optimize the feature space of the tail domains to improve generalization, by embedding the DFI value into CosFace: 

 

 
 | 
 L d ​ b ​ m = − log ⁡ e s ⁡ ( cos ⁡ ( θ y i ) − β y i ​ m ) e s ⁡ ( cos ⁡ ( θ y i ) − β y i ​ m ) + ∑ j ≠ y i e s ​ cos ⁡ ( θ j ) L_{dbm}=-\log\frac{e^{s(\cos(\theta_{y_{i}})-\beta_{y_{i}}m)}}{e^{s(\cos(\theta_{y_{i}})-\beta_{y_{i}}m)}+\sum_{j\neq y_{i}}e^{s\cos(\theta_{j})}} | 
 | 
 (78) | 
 

 where β y i \beta_{y_{i}} is the DFI value of class y i y_{i} . A large DFI value of y i y_{i} shows that class y i y_{i} is in tail domain and it gets a larger margin while training.
An overview of this network is shown in Fig. 14 . 

 
 
 Figure 14: An overview of DB mechanism with three main modules: DFI, RBM and DBM. 
 
 
 GroupFace [ 100 ] learned the group-aware representations by providing self-distributed labels that balance the number of samples belonging to each group without additional annotations, which can narrow down the search space of the target identity.
In specific, given a face sample x x , GroupFace first extracts a shared feature and deploys a FC layer to get an instance-based representation v x v_{x} and
K FC layers for group-aware representations v x G v_{x}^{G} (K is set to 32 in their experiments with best performance).
Here, a group is a set of samples that share any common visual-or-non-visual features that are used for FR.
Then, a group decision network, which is supervised by the self-distributed labeling, outputs a set of group probabilities
 { p ( G 0 | x ) , p ( G 1 | x ) , … , p ( G K − 1 | x ) } \{p(G_{0}\lvert x),p(G1\lvert x),\dots,p(G_{K-1}\lvert x)\} 
from the instance-based representation.
The final representation v ¯ x \bar{v}_{x} is an aggregation of the instance-based representation and
the weighted sum v x G v_{x}^{G} of the group-aware representations with the group probabilities.
At last, a FC with weights W W is adopted to predict face ID, where ArcFace is used to train the network.
In training, in order to construct the optimal group-space, a self-grouping loss, which reduces
the difference between the prediction and the self-generated label, is defined as: 

 

 
 | 
 L s ​ g = − 1 N ∑ i = 1 N CrossEntropy ( softmax ( f ( x i ) ) , G ∗ ( x i ) ) L_{sg}=-\frac{1}{N}\sum_{i=1}^{N}\textmd{CrossEntropy}(\textmd{softmax}(f(x_{i})),G^{*}(x_{i})) | 
 | 
 (79) | 
 

 

 
 | 
 G ∗ ( x ) = arg max k p ~ ( G k | x ) , p ~ ( G k | x ) = 1 K [ p ( G k | x ) − 1 K ] + 1 K G^{*}(x)=\arg\max_{k}\tilde{p}(G_{k}\lvert x)\ \ ,\ \ \tilde{p}(G_{k}\lvert x)=\frac{1}{K}[p(G_{k}\lvert x)-\frac{1}{K}]+\frac{1}{K} | 
 | 
 (80) | 
 

 where N N is the number of samples in a minibatch. f ⁡ ( x i ) f(x_{i}) is the output logits of the final FC (prediction). G ∗ ​ ( x i ) G^{*}(x_{i}) represents self-generated label, which is the optimal self-distributed label with largest group probability.
Therefore, the final loss is L A ​ r ​ c ​ F ​ a ​ c ​ e + L s ​ g L_{ArcFace}+L_{sg} .
The structure of GroupFace is shown in Fig. 15 . 

 
 
 Figure 15: An overview of GroupFace. 
 
 
 Gong et al. [ 101 ] thought the faces of every demographic group should be more equally represented. Thus they proposed an unbiased FR system which can obtain equally salient features for faces across demographic groups.
The propose FR network is based on a group adaptive classifier (GCA) which utilizes dynamic kernels and attention maps to boost FR performance in all demographic groups.
GAC consists of two main modules, an adaptive layer and an automation module.
In an adaptive layer, face features are convolved with a unique kernel for each demographic group, and multiplied with adaptive attention maps to obtain demographic-differential features.
The automation module determines in which layers of the network adaptive kernels and attention maps should be applied.
The framework of GAC is shown in Fig. 16 . 

 
 
 Figure 16: An overview of the GAC for mitigating FR bias. 
 
 
 Similar with [ 81 ] , Hou et al. [ 102 ] solved the AIFR problem by factorizing identity-related and age-related representations x i ​ d x_{id} and x a ​ g ​ e x_{age} .
Then x i ​ d x_{id} and x a ​ g ​ e x_{age} were optimized by age and identity discriminators.
In addition, a MI (mutual information) Estimator is designed as a disentanglement constraint to reduce the mutual information
between x i ​ d x_{id} and x a ​ g ​ e x_{age} . 

 
 
 

#### 4.2.3 Multi-task modeling with FR

 
 Besides face ID, many methods chose to bring in more supervised information while training a FR model.
In this subsection, we introduce multi-task modeling. 

 
 
 Peng et al. [ 103 ] presented a method for learning pose-invariant feature representations.
First, a 3D facial model is applied to synthesize new viewpoints from near-frontal faces.
Besides ID labels e i e^{i} , face pose e p e^{p} and landmarks e l e^{l} are also used as supervisions, and rich embedding is then achieved by jointly learning the identity and non-identity features with extractor θ r \theta^{r} .
In training, the rich embedding is split into identity, pose and landmark features, which will be fed into different losses, softmax loss for ID estimation, and L2 regression loss for pose and landmark prediction.
Finally, a genuine pair, a near-frontal face x 1 x_{1} and a non-frontal face x 2 x_{2} , is fed into the recognition network θ r \theta^{r} to obtain the embedding e 1 r e^{r}_{1} and e 2 r e^{r}_{2} . The x 1 x_{1} and the x 2 x_{2} share the same identity.
Disentangling based on reconstruction is applied to distill the identity feature from the non-identity one for robust and pose-invariant representation.
The framework of [ 103 ] is presented in Fig. 17 

 
 
 Figure 17: The architectures of R3AN. 
 
 
 Wang et al. [ 104 ] solved age-invariant FR by adding age prediction task.
To reduce the intra-class discrepancy caused by the aging, Wang et al. proposed an approach named Orthogonal Embedding CNNs (OE-Cnns) to learn the age-invariant deep face features.
OE-Cnns first trains a face feature extractor to get the feature x i x_{i} of sample i i . Then x i x_{i} is decomposed into two components.
One is identity-related component x i ​ d x_{id} , which will be optimized by identity
classification task by SphereFace [ 47 ] ;
and the other is age-related component x a ​ g ​ e x_{age} , which is used to estimate age and will be optimized by regression loss formulated as follows: 

 

 
 | 
 L a ​ g ​ e = 1 2 ​ M ​ ∑ i = 1 M ‖ f ⁡ ( x i ‖ x i ‖ 2 ) − z i ‖ 2 2 L_{age}=\frac{1}{2M}\sum_{i=1}^{M}\|f(\frac{x_{i}}{\|x_{i}\|_{2}})-z_{i}\|^{2}_{2} | 
 | 
 (81) | 
 

 where ‖ x i ‖ 2 \|x_{i}\|_{2} is the length of embedding x i x_{i} , z i z_{i} is the corresponding
age label. M M is the batchsize. f ( . ) f(.) is a mapping function aimed to associate x i ‖ x i ‖ 2 \frac{x_{i}}{\|x_{i}\|_{2}} and z i z_{i} .
While inference, after removing x a ​ g ​ e x_{age} from x x , x i ​ d x_{id} will be obtained that is supposed to be age-invariant. 

 
 
 Liu et al. [ 105 ] merged 3D face reconstruction and recognition.
In [ 105 ] each 3D face shape 𝒔 \boldsymbol{s} is represented by the concatenation of its vertex coordinates
 𝒔 = [ x 1 , y 1 , z 1 , x 2 , y 2 , z 2 , … , x n , y n , z n ] T \boldsymbol{s}=[x_{1},y_{1},z_{1},x_{2},y_{2},z_{2},\dots,x_{n},y_{n},z_{n}]^{T} ,
where n n is the number of vertices in the point cloud of the 3D face.
Based on the assumption that 3D face shapes are composed by identity-sensitive and identity-irrelevant parts,
 𝒔 \boldsymbol{s} of a subject is rewritten as: 

 

 
 | 
 𝒔 = 𝒔 ¯ + Δ ​ 𝒔 i ​ d + Δ ​ 𝒔 r ​ e ​ s \boldsymbol{s}=\overline{\boldsymbol{s}}+\Delta\boldsymbol{s}_{id}+\Delta\boldsymbol{s}_{res} | 
 | 
 (82) | 
 

 where 𝒔 ¯ \overline{\boldsymbol{s}} is the mean 3D face shape
, Δ ​ 𝒔 i ​ d \Delta\boldsymbol{s}_{id} is the
identity-sensitive difference between 𝒔 \boldsymbol{s} and 𝒔 ¯ \overline{\boldsymbol{s}} , and Δ ​ 𝒔 r ​ e ​ s \Delta\boldsymbol{s}_{res} 
denotes the residual difference.
A encoder is built to extract the face feature of a 2D image, which will be divided into two parts: 𝒄 i ​ d \boldsymbol{c}_{id} and 𝒄 r ​ e ​ s \boldsymbol{c}_{res} .
 𝒄 i ​ d \boldsymbol{c}_{id} is the latent representation employed as features for face recognition
and 𝒄 r ​ e ​ s \boldsymbol{c}_{res} is the representation face shape.
 𝒄 i ​ d \boldsymbol{c}_{id} and 𝒄 r ​ e ​ s \boldsymbol{c}_{res} are further input into two decoders to generate Δ ​ 𝒔 i ​ d \Delta\boldsymbol{s}_{id} and Δ ​ 𝒔 r ​ e ​ s \Delta\boldsymbol{s}_{res} .
Finally, identification loss L C L_{C} and reconstruction loss L R L_{R} will be designed to predict face ID and reconstruct 3D face shape. L C L_{C} is softmax loss, which directly optimizes 𝒄 i ​ d \boldsymbol{c}_{id} .
The overall network of this method can be seen in 18 

 
 
 Figure 18: Overview of method [ 105 ] with encoder-decoder based joint learning pipeline for face recognition and 3D shape reconstruction 
 
 
 Wang et al. [ 106 ] proposed a unified
Face Morphological Multi-branch Network (FM2u-Net) to generate face with makeup for makeup-invariant face verification.
FM2u-Net has two parts: FM-Net and AttM-Net.
FM-Net can synthesize realistic makeup face images by transferring specific regions of cosmetics via cycle consistent loss.
Because of the lack of sufficient and diverse makeup/non-makeup training pairs,
FM-Net uses the sets of original images and facial patches as supervision information, and employ cycle consistent loss [ 107 ] to guide realistic makeup face generation.
A softmax for ID prediction and a ID-preserving loss are added on FM-Net to constrain the generated face consistent with original face ID.
AttM-Net, consisting of one global and three local (task-driven on two eyes and mouth) branches, can capture the complementary holistic and detailed information of each face part, and focuses on generating makeup-invariant facial representations by fusing features of those parts. In specific, the total loss for AttM-Net is performed on the four part features and the fused one f c ​ l ​ s f_{cls} : 

 

 
 | 
 L A ​ t ​ t ​ M = γ 1 ​ L l − e ​ y ​ e + γ 2 ​ L r − e ​ y ​ e + γ 3 ​ L m ​ o ​ u ​ t ​ h + γ 4 ​ L g ​ l ​ o ​ b ​ a ​ l + γ 5 ​ L c ​ l ​ s L_{AttM}=\gamma_{1}L_{l-eye}+\gamma_{2}L_{r-eye}+\gamma_{3}L_{mouth}+\gamma_{4}L_{global}+\gamma_{5}L_{cls} | 
 | 
 (83) | 
 

 where γ ∗ \gamma_{*} are weights. L l − e ​ y ​ e L_{l-eye} , L r − e ​ y ​ e L_{r-eye} and L m ​ o ​ u ​ t ​ h L_{mouth} are softmax loss of left, right eye and mouth patch. L g ​ l ​ o ​ b ​ a ​ l L_{global} and L c ​ l ​ s L_{cls} are losses based on the whole face feature and fused feature f c ​ l ​ s f_{cls} , which are also softmax pattern. 

 
 
 Gong et al. [ 108 ] proposed a de-biasing adversarial network (DebFace) to jointly learn FR and demographic attribute estimation (gender, age and race).
DebFace network consists of four components:
the shared image-to-feature encoder E I ​ m ​ g E_{Img} ,
the four attribute classifiers (including gender C G C_{G} , age C A C_{A} , race C R C_{R} , and identity C I ​ D C_{ID} ),
the distribution classifier C D ​ i ​ s ​ t ​ r C_{Distr} ,
and the feature aggregation network E F ​ e ​ a ​ t E_{Feat} .
DebFace first projects an image x i x_{i} to its feature representation E I ​ m ​ g ​ ( x i ) E_{Img}(x_{i}) by the encoder E I ​ m ​ g E_{Img} .
Then the representation is decoupled into gender, age, race and identity vectors.
Next, each attribute classifier operates the corresponding vector to classify the target attribute
by optimizing parameters of E I ​ m ​ g E_{Img} and the respective classifier C ∗ C_{*} .
The learning objective L C D ​ e ​ m ​ o L_{C_{Demo}} ( C D ​ e ​ m ​ o = { C G , C A , C R } C_{Demo}=\{C_{G},C_{A},C_{R}\} ) is cross entropy loss.
For the identity classification, AM-Softmax [ 49 ] is adopted in L C I ​ D L_{C_{ID}} .
To de-bias all of the representations, adversarial loss L A ​ d ​ v L_{Adv} is applied to
the above four classifiers such that each of them will NOT be able to predict
correct labels when operating irrelevant feature vectors.
To further improve the disentanglement, the mutual
information among the attribute features is reduced by a distribution classifier C D ​ i ​ s ​ t ​ r C_{Distr} .
At last, a factorization objective function L F ​ a ​ c ​ t L_{Fact} 
is utilized to minimize the mutual information of the four attribute representations.
Altogether, DebFace endeavors to minimize the joint loss: 

 

 
 | 
 L = L C D ​ e ​ m ​ o + L C I ​ D + L C D ​ i ​ s ​ t ​ r + λ 1 ​ L A ​ d ​ v + λ 2 ​ L F ​ a ​ c ​ t L=L_{C_{Demo}}+L_{C_{ID}}+L_{C_{Distr}}+\lambda_{1}L_{Adv}+\lambda_{2}L_{Fact} | 
 | 
 (84) | 
 

 where λ ∗ \lambda_{*} are hyper-parameters determining how much the representation is decomposed and decorrelated in each training iteration.
The pipeline of DebFace is shown in fig 19 

 
 
 Figure 19: Overview of the DebFace network. 
 
 
 Similar with DebFace, PASS (Protected Attribute Suppression System) [ 109 ] proposed by Dhar et al. also learned face features which are insensitive to face attribute.
In PASS, a feature f i ​ n f_{in} is firstly extracted by a pretrained model.
Then a generator M inputs f i ​ n f_{in} and generates a new feature f o ​ u ​ t f_{out} that are agnostic to face attributes (such as gender and skintone).
A classifier is applied to optimized f o ​ u ​ t f_{out} to identify face ID.
In addition, f o ​ u ​ t f_{out} is fed to ensemble E, and each of the attribute discriminators in E is used to compute attribute classification.
In order to force the feature f o ​ u ​ t f_{out} insensitive to face attributes, an adversarial loss for model M with respect to all the models in E is calculated by constraining a posterior probability of 1 N a ​ t ​ t \frac{1}{N_{att}} for all categories in the attribute, where N a ​ t ​ t N_{att} denotes the number of classes in the considered attribute. 

 
 
 Wang et al. [ 110 ] relieved the problem of FR with extreme poses by
lightweight pseudo facial generation.
This method can depict the facial contour and make appropriate modifications to preserve the critical identity information without generating any frontal facial image.
The proposed method includes a generator and a discriminator.
Different from traditional GAN, the generator does not use encoder-decoder style. Instead, the lightweight pseudo profile facial generator is designed as a residual network, whose computational costs are much lower.
To preserve the identity consistent information,
the embedding distances between the generated pseudo ones and their corresponding frontal facial images are minimized, which is similar with the ID-preserving loss in [ 106 ] .
Suppose D ( . ) D(.) denotes a face embedding from a facial
discriminator, the identity preserving loss is formulated as
 L i ​ d = ‖ D ⁡ ( I g ) − D ⁡ ( I f ) ‖ 2 2 L_{id}=\|D(I^{g})-D(I^{f})\|_{2}^{2} ,
where I f I^{f} and I g I^{g} are real frontal and generated pseudo face, which share a same ID.
Fig 20 depicts the pipeline of this method. 

 
 
 Figure 20: The pipeline of pseudo facial generation framework. 
 
 
 Besides big poses, the performance of FR algorithm also degenerates under the uncontrolled illumination. To solve it, He et al. [ 111 ] introduced 3D face reconstruction into the FR training process as an auxiliary task. The reconstruction was performed based on imaging principles. Four important imaging parameters were learned by two auxiliary networks in training. In the parameters, view matrix and illumination were identity irrelevant and extracted from shallow layers, while depth and albedo were identity relevant and extracted from the intermediate layer. Based on the parameters extraction and reconstruction loss, the FR network can focus on identity relevant features. 

 
 
 
 

### 4.3 FR with massive IDs

 
 Larger number of IDs (width) in training set can usually achieve a greater FR result. Thus in the real-world FR applications, it is crucial to adopt large scale face datasets in the wild.
However, the computing and memory cost linearly scales up to the number of classes.
Therefore, some methods aim at enlarging the throughput of IDs while training. 

 
 
 Larger number of IDs leads to a larger classifier, which may exceed the memory of GPU(s). An intuitive solution is to split the classifier along the class dimension and evenly distribute it to each card.
On this basis, Partial FC [ 112 ] [ 113 ] further proposed that, for each sample, it is not necessary to use all the negative class centers when calculating logits. A relatively high accuracy can also be achieved by sampling only a part of the negative class centers. 

 
 
 BroadFace [ 114 ] thought that only a small number of samples are included in the minibatch, and each parameter update of the minibatch may cause bias, which will fail to converge to the optimal solution.
As a result, BroadFace saved the embedding of the previous iteration e i − e_{i}^{-} into a queue, and optimized the classifier and model parameters together with the embedding of the current iteration e i e_{i} . Compared with the traditional training method, BroadFace involved more samples in each iteration.
However, as the training progressing, the feature space of the model will gradually drift. There will be a gap between the embedding in the queue and the current feature space. Directly using the embedding of the queue for optimization may result in a decrease in model performance. In this case, the classifier parameters of the previous iteration are used to compensate the embedding in the queue, that is 

 

 
 | 
 e i ∗ = e i − + ρ ⁡ ( y ) ≈ e i − + ‖ e i − ‖ ‖ W y i − ‖ ​ ( W y − W y − ) e^{*}_{i}=e_{i}^{-}+\rho(y)\approx e_{i}^{-}+\frac{\left\|e_{i}^{-}\right\|}{\left\|W_{y_{i}}^{-}\right\|}(W_{y}-W_{y}^{-}) | 
 | 
 (85) | 
 

 where y y is the label of e i − e_{i}^{-} , W y W_{y} and W y − W_{y}^{-} are the classifier weights of current and previous iteration.
Compared with traditional methods whose batchsize is usually 256 or 512, BroadFace can adopt thousands of embeddings to optimize the model in each iteration. 

 
 
 Partial FC proposed that only 10% of the class number can be used to train a high-precision recognition model.
Inspired by Partial FC, Li et al. [ 115 ] proposed a method to train FR model with massive IDs, and also improve the performance on long-tail distributed data.
Li et al. tried to discard the FC layer in the training process, and used a queue of size K K to store the category center ( K ≪ C K\ll C , C C is the number of IDs in the training set), which is treated as the negative class center.
And the dynamic class generation method is used to generate the positive class center.
Then the positive class center is pushed into the queue, and the oldest class centers of the queue are popped out.
This training process has two problems:
1, there is no guarantee that the positive class center of the current batch is not included in the queue; 2, as training progressing, the feature space will drift, and the previous negative class center cannot match the current feature space.
For the first problem, the author simply setted the duplicate logits of all positive class centers in the queue to negative infinity, so that the response to softmax is 0.
For the second problem, the author referred to the idea of MOCO and used the model composed of EMA with feature extractor parameters to generate the positive center.
This article used a queue of size B + K B+K to store the category center of the previous iteration, where B B is the batchsize and K K is the capacity of the queue. In each iteration, only B B elements in the queue are updated. 

 
 
 In [ 116 ] , a new layer called Virtual FC layer was proposed to reduce the computational consumption of the classification paradigm in training.
The algorithm splits N N training IDs into M M groups randomly.
The identities from group l l share the l l -th column in the projection matrix W ∈ ℝ D × M W\in\mathbb{R}^{D\times M} , where D D is the dimension of face embedding.
The l l -th column of W W is called a ​ n ​ c ​ h ​ o ​ r l anchor_{l} .
If the mini-batch contains identities from group l l , a ​ n ​ c ​ h ​ o ​ r l anchor_{l} is of type a ​ n ​ c ​ h ​ o ​ r c ​ o ​ r ​ r anchor_{corr} .
Otherwise, it is of type a ​ n ​ c ​ h ​ o ​ r f ​ r ​ e ​ e anchor_{free} . The anchor type is adaptive in every training iteration.
Each column in the projection matrix W W of the final FC layer indicates the
centroid of a category representation, thus the corresponding
anchors is a ​ n ​ c ​ h ​ o ​ r c ​ o ​ r ​ r , l = 1 K ​ ∑ i = 1 K f i , l anchor_{corr,l}=\frac{1}{K}\sum_{i=1}^{K}f_{i,l} ,
where f i , l f_{i,l} the feature of the i i -th image that belongs to group l, and K K is the number of features in group l l .
In training, a ​ n ​ c ​ h ​ o ​ r l anchor_{l} will be updated by the above equation if it belongs to a ​ n ​ c ​ h ​ o ​ r c ​ o ​ r ​ r anchor_{corr} , and it stays the same if it is of type a ​ n ​ c ​ h ​ o ​ r f ​ r ​ e ​ e anchor_{free} .
As mentioned above, several IDs are related in group l l . Therefore, [ 116 ] further came up with a regrouping strategy to avoid sampling identities that are from the same group into a mini-batch. 

 
 
 Faster Face Classification ( F 2 ​ C F^{2}C ) [ 117 ] , adopted Dynamic Class Pool (DCP) for storing and updating the identities’ features dynamically, which could be regarded as a substitute for the FC layer. 

 
 
 

### 4.4 Cross domain in FR

 
 Generally, when we utilized the FR algorithms, the training and testing data should have similar distributions.
However, face images from different races, or in different scenes (mobile photo albums, online videos, ID cards) have obvious domain bias.
Unsatisfactory performance will occur when training and testing domain gap exits, due to the poor generalization ability of neural networks to handle unseen data in practice.
Therefore, domain adaptation is proposed to solve this problem.
In this subsection, we first proposed general domain adaptation methods in FR.
And then we list some FR methods with uncommon training images. 

 
 
 Model-agnostic meta-learning (MAML) [ 118 ] is one of the most representative methods in domain adaptation. It aims to learn a good weights initialization of a model, so that it can get good results on new tasks with a small amount of data and training iterations.
The input of the algorithm is a series of tasks with its corresponding support set and query set.
The output is a pre-trained model.
A special optimization process was proposed in this work, that is, for each iteration, the initial parameter of the model is θ \theta , for each task 𝒯 i \mathcal{T}_{i} , a gradient descent is performed on the support set with a larger learning rate to get special parameters θ i ′ \theta^{\prime}_{i} for each task’s model, and then use the model under parameter θ i ′ \theta_{i}^{\prime} on the query set to find the gradient of each task to θ \theta , and then perform gradient descent for θ \theta with a smaller learning rate.
The whole process is divided into two gradient calculations.
The first time is to calculate the gradient that can improve performance for each task.
The second time is to update the model parameters under the guidance of the first gradient descent result. 

 
 
 Many researchers brought MAML in FR algorithms to alleviate cross domain problem.
Guo et al. [ 119 ] proposed meta face recognition (MFR) to solve the domain adaptation problem in FR through meta-learning.
In each iteration of training, only one of N domains in the training set will be chosen as meta-test data, corresponding to the query set of MAML; and the rest of (N-1) domains in the training data are used as meta-train data, corresponding to the support set of MAML. All these data constitutes a meta-batch.
Meta-test data is utilized to simulate the domain shift phenomenon in the application scenario.
Then, the hard-pair attention loss ℒ h ​ p \mathcal{L}_{hp} , the soft-classification loss ℒ c ​ l ​ s \mathcal{L}_{cls} , and the domain alignment loss ℒ d ​ a \mathcal{L}_{da} are proposed. The ℒ h ​ p \mathcal{L}_{hp} optimizes hard positive and negative pairs by shrinking the Euclidean distance in hard positive pairs and pushing hard negative pairs away.
The ℒ c ​ l ​ s \mathcal{L}_{cls} is for face ID classification, which is modified from cross-entropy loss.
To perform domain alignment, the ℒ d ​ a \mathcal{L}_{da} is designed to make the mean embeddings of multiple meta-train domains close to each other.
In the optimization process, this article also followed a similar method to MAML: for model parameters θ \theta , MFR first uses meta-train data to update the model parameters to obtain the θ ′ \theta^{\prime} by ℒ h ​ p \mathcal{L}_{hp} , ℒ c ​ l ​ s \mathcal{L}_{cls} , and ℒ d ​ a \mathcal{L}_{da} .
Then MFR uses the updated θ ′ \theta^{\prime} to calculate meta-test loss on ℒ h ​ p \mathcal{L}_{hp} , ℒ c ​ l ​ s \mathcal{L}_{cls} . After that, the gradient is utilized to further update the model parameters.
The difference is that when updating the θ \theta , not only the meta-test loss is used to calculate the gradient, but also the meta-train loss is used, which is for the balance of meta-train and meta-test. 

 
 
 Faraki et al. [ 120 ] pointed out that the domain alignment loss ℒ d ​ a \mathcal{L}_{da} in MFR may lead to a decrease in model performance. The reason is that while the mean value of the domain is pulled closer, samples belonging to different IDs may be pulled closer, resulting in a decrease in accuracy.
Therefore, they proposed cross domain triplet loss based on triplet loss, which is shown as follows: 

 
 
 
 | 
 | 
 l c ​ d ​ t ( i 𝕋 , j 𝕋 ; θ r ) = \displaystyle l_{cdt}(^{i}\mathbb{T},^{j}\mathbb{T};\theta_{r})= | 
 | 
 (86) | 

 
 | 
 | 
 1 B ∑ b = 1 B [ 1 H ​ W ∑ h = 1 H ∑ w = 1 W d j ​ Σ + 2 ( [ f r ( i a b ) ] h , w , [ f r ( i p b ) ] h , w ) − \displaystyle\frac{1}{B}\sum_{b=1}^{B}\left[\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}d_{j\Sigma^{+}}^{2}([f_{r}(^{i}a_{b})]_{h,w},[f_{r}(^{i}p_{b})]_{h,w})-\right. | 
 | 

 
 | 
 | 
 1 H ​ W ∑ h = 1 H ∑ w = 1 W d j ​ Σ − 2 ( [ f r ( i a b ) ] h , w , [ f r ( i n b ) ] h , w ) + τ ] + \displaystyle\left.\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}d_{j\Sigma^{-}}^{2}([f_{r}(^{i}a_{b})]_{h,w},[f_{r}(^{i}n_{b})]_{h,w})+\tau\right]_{+} | 
 | 
 

 where 𝕋 j {}^{j}\mathbb{T} represents the triplets of domain j j , θ r \theta_{r} represents the parameters of the representation model, f r f_{r} represents the representation model, the output of f r f_{r} is a tensor with dimensions ( H , W , D ) (H,W,D) . The ( a , p , n ) (a,p,n) denotes the anchor, positive and negative sample in triplets. d j ​ Σ + 2 d^{2}_{j\Sigma^{+}} represents the Mahalanobis distance of the covariance matrix based on the distance of the positive pairs in domain j j .
The cross domain triplet loss can align the distribution between different domains.
The proposed optimization process is also similar to MAML and MFR. The training data is divided into meta-train and meta-test.
The initial parameter is θ \theta . Then in each iteration, meta-train data is firstly optimized by CosFace [ 50 ] and triplet loss [ 34 ] , to obtain the updated parameters θ ′ \theta^{\prime} . The θ ′ \theta^{\prime} is used to calculate the meta-test loss on the meta-test. Finally the meta-test loss is calculated to update the model parameter. The optimization on meta-test data uses cross entropy loss, triplet loss, and cross domain triplet loss. 

 
 
 Sohn et al. [ 121 ] proposed an unsupervised domain adaptation method for video FR using large-scale unlabeled videos and labeled still images.
It designed a reference net named RFNet with supervised images as a reference, and a video net named VDNet based on the output of RFNet with labeled and synthesized still images and unlabeled videos.
The network architecture can be seen in Fig. 21 . 

 
 
 Figure 21: The network architecture for RFNet and VDNet. 
 
 
 Based on RFNet and VDNet, four losses are proposed, namely the feature match loss ℒ F ​ M \mathcal{L}_{FM} , the feature restoration loss ℒ F ​ R \mathcal{L}_{FR} , the image classification loss ℒ I ​ C \mathcal{L}_{IC} and the adversarial loss ℒ A ​ d ​ v \mathcal{L}_{Adv} .
The ℒ F ​ M \mathcal{L}_{FM} is for shortening the gap between the embeddings extracted from RFNet and VDNet on labeled images ℐ \mathcal{I} , whose formulation is as follows: 

 

 
 | 
 ℒ F ​ M = 1 | ℐ | ​ ∑ x ∈ ℐ ‖ ϕ ⁡ ( x ) − ψ ⁡ ( x ) ‖ 2 2 \mathcal{L}_{FM}=\frac{1}{\lvert\mathcal{I}\rvert}\sum_{x\in\mathcal{I}}\left\|\phi(x)-\psi(x)\right\|_{2}^{2} | 
 | 
 (87) | 
 

 where ϕ ⁡ ( x ) \phi(x) and ψ ⁡ ( x ) \psi(x) are the output features of VDNet and RFNet for a same input image x x .
Then the authors set a feature restoration constrain on VDNet. They add a series of data argumentation operations B ⁡ ( ⋅ ) B(\cdot) to image set ℐ \mathcal{I} , including linear motion blur, scale variation, JPEG compression, etc.,
and use the ℒ F ​ R \mathcal{L}_{FR} to optimize VDNet to “restore” the original RFNet representation of an image without data
augmentation. The ℒ F ​ R \mathcal{L}_{FR} is designed as follows 

 

 
 | 
 ℒ F ​ R = 1 | ℐ | ​ ∑ x ∈ ℐ 𝔼 B ⁡ ( ⋅ ) ​ [ ‖ ϕ ⁡ ( B ⁡ ( x ) ) − ψ ⁡ ( x ) ‖ 2 2 ] {\mathcal{L}_{FR}}=\frac{1}{\lvert\mathcal{I}\rvert}\sum_{x\in\mathcal{I}}\mathbb{E}_{B(\cdot)}\left[\left\|\phi(B(x))-\psi(x)\right\|_{2}^{2}\right] | 
 | 
 (88) | 
 

 where 𝔼 B ⁡ ( ⋅ ) \mathbb{E}_{B(\cdot)} is the expectation over the distribution of the image transformation kernel B ⁡ ( ⋅ ) B(\cdot) .
The ℒ I ​ C \mathcal{L}_{IC} is designed in metric learning formulation to reduce the gap between features of images from the RFNet and their related synthesis from the VDNet. Given N N pairs of examples from N N different classes
 { ( x i , x i + ) } i = 1 N \{(x_{i},x_{i}^{+})\}^{N}_{i=1} , the ℒ I ​ C \mathcal{L}_{IC} is shown as follows: 

 

 
 | 
 ℒ I ​ C = − 1 N ∑ i = 1 N log exp ⁡ ( ϕ ​ ( B i ​ ( x i + ) ) ⊤ ​ ψ ​ ( x i ) ) ∑ n = 1 N exp ⁡ ( ϕ ​ ( B i ​ ( x i + ) ) ⊤ ​ ψ ​ ( x n ) ) \mathcal{L}_{IC}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp\left(\phi\left(B_{i}\left(x_{i}^{+}\right)\right)^{\top}\psi\left(x_{i}\right)\right)}{\sum_{n=1}^{N}\exp\left(\phi\left(B_{i}\left(x_{i}^{+}\right)\right)^{\top}\psi\left(x_{n}\right)\right)} | 
 | 
 (89) | 
 

 The ℒ A ​ d ​ v \mathcal{L}_{Adv} is adopted to fool the discriminator and refine the gaps between the domain of image (y=1), synthesized images (y=2), and videos (y=3), which is shown as follows: 

 

 
 | 
 ℒ A ​ d ​ v = − 𝔼 x ∈ ℬ ⁡ ( ℐ ) ∪ 𝒱 [ log 𝒟 ( y = 1 | ϕ ( x ) ) ] \mathcal{L}_{Adv}=-\mathbb{E}_{x\in\mathcal{B}(\mathcal{I})\cup\mathcal{V}}[\log\mathcal{D}(y=1\lvert\phi(x))] | 
 | 
 (90) | 
 

 where 𝒱 \mathcal{V} represents video images domain. 

 
 
 Some methods try to solve FR problems of uncommon images (such as NIR images or radial lens distortion images), which has huge domain gap with conventional RGB images in mainstream FR datasets. 

 
 
 To solve pose-invariant FR problem, Sengupta et al. [ 122 ] adopted facial UV map in their algorithm.
An adversarial generative network named UV-GAN was proposed to generate the completed UV from the incompleted UV that came from the 3D Morphable Model [ 123 ] .
Then face images with different poses can be synthesized from the UV, which will be utilized to get pose-invariant face embeddings. The UV-GAN is composed of the generator, the global and local discriminator, and the pre-trained identity classification network (Fig. 22 ).
Two discriminators ensure that the generated images are consistent with their surrounding contexts with vivid details.
The pre-trained identity classification network is an identity preserving module and fixed during the training process. 

 
 
 Figure 22: The pipeline of the UV-GAN. [ 122 ] 
 
 
 In addition to pose variation, two-dimensional (2D) FR is also faced with other challenges such as illumination, scale and makeup.
One solution to these problems is the three-dimensional (3D) FR.
Gilani et al. [ 124 ] proposed the Deep 3D FR Network (FR3DNet) to solve the shortage of 3D face data. Inspired by [ 125 ] , they presented a synthesized method.
The synthesized method firstly conducted dense correspondence.
After obtaining synthesized images, commercial software was utilized to synthesis varying facial shapes, ethnicities, and expressions.
The architecture of the FR3DNet was similar to [ 33 ] and it adopted a large kernel size to process the point cloud information.
To feed the 3D point cloud data into the network, the authors changed the data to three channel images.
The first channel was the depth information obtained from the g ​ r ​ i ​ d ​ f ​ i ​ t gridfit algorithm [ 126 ] .
The second and third channels were generated based on the azimuth angles and the elevation angle in spherical coordinates. 

 
 
 FR in surveillance scenario is another challenge domain, because most of faces in this situation are poor quality images with low-resolution (LR). While most images in academic training sets are high-resolution (HR).
Fang et al. [ 127 ] built an resolution adaption network (RAN) (Fig. 23 ) to alleviate low-resolution problem.
It contained three steps.
First, the multi-resolution generative adversarial network was proposed to generate LR images.
It input three images ( x r ​ 1 x_{r1} , x r ​ 2 x_{r2} , and x r ​ 3 x_{r3} ) with different resolutions, which were processed by the parallel sub-networks [ 128 ] .
In the second step, HR and LR FR model were trained separately to obtain face representations.
At last, a feature adaption network was designed to allow the model have high recognition ability in both HR and LR domains.
The loss L H ​ R L_{HR} using ArcFace [ 51 ] was applied to the HR face representations( f H ​ R f_{HR} ) to directly make the model applicable to HR faces.
At the same time, the f H ​ R f_{HR} was input into a novel translation gate to minimize the gap between the HR and LR domains.
The output of the translation gate( T L ​ R ​ ( f H ​ R ) T_{LR}(f_{HR}) ) preserved LR information contained in the f H ​ R f_{HR} and the preserving result was monitored by a discriminator, which was used to distinguish T L ​ R ​ ( f H ​ R ) T_{LR}(f_{HR}) from the synthesized LR embedding.
The final LR representations f L ​ R T ​ r ​ a ​ n ​ s ​ l ​ a ​ t ​ e f_{LR}^{Translate} combined T L ​ R ​ ( f H ​ R ) T_{LR}(f_{HR}) and f H ​ R f_{HR} .
 L 1 L_{1} loss and KL loss were applied on f L ​ R T ​ r ​ a ​ n ​ s ​ l ​ a ​ t ​ e f_{LR}^{Translate} and the real LR image representations to further ensure the quality of f L ​ R T ​ r ​ a ​ n ​ s ​ l ​ a ​ t ​ e f_{LR}^{Translate} . 

 
 
 Figure 23: The pipeline of the RAN [ 127 ] . 
 
 
 Surveillance cameras often capture near infrared (NIR) images in low-light environments.
FR systems trained by the visible light spectrum (VIS) face images can not work effectively in this situation, due to the domain gap.
Lezama et al. [ 129 ] proposed a NIR-VIS FR system, which can perform a cross-spectral FR and match NIR to VIS face images.
NIR-VIS contains two main components: cross-spectral hallucination and low-rank embedding.
The cross-spectral hallucination learns a NIR-VIS mapping on a patch-to-patch basis.
After hallucination, the CNN output of luminance channel is blended with the original NIR image to avoid losing the information contained in the NIR image.
The blending formula is shown as follows: 

 

 
 | 
 Y = Y ^ − α ⋅ G σ 2 ∗ ( N i ​ r − Y ^ ) Y=\hat{Y}-\alpha\cdot G_{\sigma}^{2}*(N_{ir}-\hat{Y}) | 
 | 
 (91) | 
 

 where Y Y is the output after combining the hallucinated result with the NIR image, Y ^ \hat{Y} is the hallucinated result, N i ​ r N_{ir} is the NIR image, G σ G_{\sigma} is a Gaussian filter with σ = 1 \sigma=1 , and ∗ * denotes convolution.
The second component is low-rank embedding, which performs a low-rank transform [ 130 ] to embed the output of the VIS model. 

 
 
 To alleviate the effects of radial lens distortion on face image,
a distortion-invariant FR method called RDCFace [ 131 ] was proposed for wide-angle cameras of surveillance and safeguard systems.
Inspired by STN [ 36 ] , RDCFace can learn rectification, alignment parameters and face embedding in a end-to-end way, therefore it did not require supervision of facial landmarks and distortion parameters.
The pipeline of RDCFace is shown in Fig. 24 . 

 
 
 Figure 24: The pipeline of the RDCFace. 
 
 
 In training, the data preparation module generates radial lens distortion on common face images with random parameters.
Then the cascaded network of RDCFace sequentially rectifies distortion by correction network, aligns faces by alignment network, and extracts features by recognition network for FR.
The correction network predicts the distortion coefficient k k to rectify the radial
distortion based on the inverse transformation of the division model: 

 

 
 | 
 r d = 1 − 1 − 4 ​ k ​ r u 2 2 ​ k ​ r u r_{d}=\frac{1-\sqrt{1-4kr_{u}^{2}}}{2kr_{u}} | 
 | 
 (92) | 
 

 where r u r_{u} and r d r_{d} represent the Euclidean distance from an arbitrary pixel to the image center of original and distorted
images.
Taking a distorted face image I d I^{d} as input,
a well-trained correction network f c ​ o ​ r ​ r ​ e ​ c ​ t f_{correct} should predict the distortion
parameter k k accurately.
Then the rectification layer L R L_{R} then use parameter k k to eliminate the distortion and generates corrected
image I c I^{c} .
After that, I c I^{c} is sent into f c ​ o ​ r ​ r ​ e ​ c ​ t f_{correct} again. This time, the output parameter is
expected to be zero since the input I c I^{c} is expected to be corrected perfectly without distortion.
Therefore, the re-correction loss is designed to
1, encourage the correction network to better eliminate the distortion and
2, suppress the excessive deformation on the distortion free image. 

 

 
 | 
 L c ​ o ​ r ​ r ​ e ​ c ​ t = E ​ ‖ f c ​ o ​ r ​ r ​ e ​ c ​ t ​ ( I c ) ‖ 2 2 = E ​ ‖ f c ​ o ​ r ​ r ​ e ​ c ​ t ​ ( L R ​ ( f c ​ o ​ r ​ r ​ e ​ c ​ t ​ ( I d ) , I d ) ) ‖ 2 2 L_{correct}=E\|f_{correct}(I^{c})\|^{2}_{2}=E\|f_{correct}(L_{R}(f_{correct}(I^{d}),I^{d}))\|^{2}_{2} | 
 | 
 (93) | 
 

 The alignment networks predicts projective transformation parameters, which is similar with STN.
The final training loss in RDCFace is sum of ArcFace and L c ​ o ​ r ​ r ​ e ​ c ​ t L_{correct} . 

 
 
 Some previous works utilized synthetic faces to train face recognition model to solve the problems caused by real data, such as label noise, unbalanced data distribution, and privacy. While, the model trained on synthetic images can not perform as well as the model trained on real images. SynFace [ 132 ] found one of the reason is the limitation of the intra-class variations within the synthetic data, therefore it proposed identity mixup (IM) for the input parameters of the generator DiscoFace-GAN [ 133 ] . The other technique named domain mixup, by adding a small quantity of real data into the training, the performance of model was improved greatly. 

 
 
 

### 4.5 FR pipeline acceleration

 
 Liu et al. [ 134 ] proposed a novel technique named network slimming. The core idea is that each scaling factor in the batch normalization (BN) layer can indicate the importance of the corresponding channel. The smaller the scaling factor is, the more insignificant this channel will be. Based on this idea, the author performed L1 regularization on the scaling factor in the BN layer during training to make it sparse, and then cut out unimportant channels according to the size of the scaling factor. 

 
 
 Then, by experiments, Liu et al. [ 135 ] further proved that the structure of the network is more important than network weights and pruning can be regarded as the process of network architecture search.
After the pruning, retraining the network with randomly initialized parameters allows the network achieve a better result. 

 
 
 In face identification, we need to match the query face feature with all features in gallery. In order to accelerate this matching process, different searching technologies were adopted.
The K-D tree [ 136 ] is a common tree-based search algorithm.
The K-D tree continuously divides the whole feature space into two parts by a hyperplane to form a binary tree structure.
Each point in the feature space corresponds to a node in the tree, and then searches the tree to find the nearest point.
However, the K-D tree degenerates into linear search in higher dimensions due to very sparse data distribution in high dimensional space.
The efficiency of feature space segmentation is very low. Based on the segmentation, the search precision declines. 

 
 
 Vector quantization is a process of encoding a feature space with a limited set of points.
After the encoding, the resulting set of limited points is called a codebook, and each point in the codebook can represent a region in the feature space.
Vector quantization can speed up the calculation of the distance between vectors.
When m vectors in a feature space are encoded by a codebook of size N (m ⁣ N),
the number of calculations for comparison is reduced from m times to N times.
Product quantization [ 137 ] is an ANN algorithm based on vector quantization. Vectors in the feature space are first divided into m sub-vectors, and m groups of codebooks are used for quantization. Each codebook has a size of k. Then, the author proposed a novel method combining an inverted file system with the asymmetric distance computation (IVFADC). 

 
 
 The core idea of graph-based search algorithm is to build points in the feature space into a graph, and start from a point n ​ o ​ d ​ e c ​ u ​ r ​ r node_{curr} randomly during each retrieval.
Then it calculate the distance between the neighbor of the point and the vector to be queried,
and select the neighbor with the smallest distance as the next retrieval point n ​ o ​ d ​ e c ​ u ​ r ​ r node_{curr} .
Until the distance between n ​ o ​ d ​ e c ​ u ​ r ​ r node_{curr} and query is less than the distance between any of n ​ o ​ d ​ e c ​ u ​ r ​ r node_{curr} ’s neighbors and query.
The n ​ o ​ d ​ e c ​ u ​ r ​ r node_{curr} is considered as the closest point to query in the graph. 

 
 
 Navigable small world (NSW) algorithm [ 138 ] is a graph-based search algorithm. Delauenian triangulation has a high time complexity in constructing graph, so NSW algorithm does not adopt this method. During the graph construction process of NSW algorithm, each point is inserted in sequence, and after insertion, neighbors are found and joined in the current graph. In a graph constructed this way, there are two kinds of edges: short-range links to approximate the delaunay graph, and long-range links to reduce the number of searches on a logarithmic scale. 

 
 
 During the insertion point process, the early short-range links may become long-range links at a later stage, and these long-range links make the whole graph navigable Small World. 

 
 
 The core idea of the locality sensitive hashing search algorithm is to construct a hash function h h so that h ⁡ ( p ) = h ⁡ ( q ) h(p)=h(q) has a high probability when vectors p p and q q are close enough.
During search, locality sensitive hashing is performed on all features in the data set to obtain hash values. Then, only vectors with hash values similar to query vectors need to be compared to reduce computation. 

 
 
 Knowledge distillation is an effective tool to compress large pre-trained CNNs into models applicable to mobile and embedded devices, which is also a major method for model acceleration.
Wang et al. [ 139 ] put forward a knowledge distillation method specially for FR, which aims to improve the capability of the target FR student network.
They first reshape all filters from a convolutional layer W W from ℝ N × M × K 1 × K 2 \mathbb{R}^{N\times M\times K_{1}\times K_{2}} to ℝ N × D \mathbb{R}^{N\times D} , where N N and M M are numbers of filters and input channels; K 1 K_{1} and K 2 K_{2} are the spatial height and width of filters; D = M × K 1 × K 2 D=M\times K_{1}\times K_{2} .
Then they define weight exclusivity for weights W W as : 

 

 
 | 
 L W ​ E ​ ( W ) = ∑ 1 ≤ j ≠ i ≤ N ∑ k = 1 D | w i ​ ( k ) | ⋅ | w j ​ ( k ) | L_{WE}(W)=\sum_{1\leq j\neq i\leq N}\sum_{k=1}^{D}\lvert w_{i}(k)\rvert\cdot\lvert w_{j}(k)\rvert | 
 | 
 (94) | 
 

 where w i ∈ ℝ 1 × D w_{i}\in\mathbb{R}^{1\times D} is a filter in W W with index i i .
It can be seen that, L W ​ E ​ ( W ) L_{WE}(W) encourages each of two filter vectors in W W to be as diverse as possible. Thus applying weight exclusivity on student network will force to enlarge its capability.
Finally, the proposed exclusivity-consistency regularized knowledge distillation becomes: 

 

 
 | 
 L = L H ​ F ​ C + λ 1 ​ L W ​ D + λ 2 ​ L W ​ E L=L_{HFC}+\lambda_{1}L_{WD}+\lambda_{2}L_{WE} | 
 | 
 (95) | 
 

 where L H ​ F ​ C L_{HFC} is L2 based hardness-aware feature consistency loss, which encourages the face features from teacher and student as similar as possible. L W ​ D L_{WD} is L2 weight decay. 

 
 
 The knowledge distillation methods mostly regarded the relationship between samples as knowledge to force the student model learn the correlations rather than embedding features from the teacher model. Huang et al. [ 140 ] found that allowing the student study all relationships was inflexible and proposed an evaluation-oriented technique. The relationships that obtained different evaluation result from teacher and student models were defined as crucial relationships. Through a novel rank-based loss function, the student model can focus on these crucial relationships in training. 

 
 
 

### 4.6 Closed-set Training

 
 For some face verification and identification projects in industry, FR problem can be treated as a closed-set classification problem.
In the case of “face identification of politicians in news” or “face identification of Chinese entertainment stars”, the business side which applies the FR requirements usually has a list of target people IDs.
In this situation, the gallery of this identification work is given, and we can train with those IDs to achieve a higher accuracy.
As a result, FR model training becomes a closed-set problem.
In summary, based on whether all testing identities are predefined in the training set, FR systems can
be further categorized into closed-set systems and open-set systems, as illustrated in Fig. 25 . 

 
 
 Figure 25: Closed-set and open-set FR systems. 
 
 
 A close-set FR task is equivalent to a multi-class classification problem by using the standard softmax loss function in the training phase [ 41 , 141 , 35 ] . 

 
 
 Tong et al. [ 142 ] proposed a framework for fine-grained robustness evaluation of both closed-set and open-set FR systems. Experimental results shows that, open-set FR systems are more vulnerable than closed-set systems under different types of attacks (digital attack, pixel-level physically realizable attack, and grid-level physically realizable attack).
As a result, we can conclude that, closed-set FR problem is easier than open-set FR.
Thus we can adopt more classification with delicate design to acquire a better performance in FR business.
Techniques such as fine-grained recognition, attention based classification, etc. can be employed in closed-set FR. 

 
 
 

### 4.7 Mask face recognition

 
 Face recognition has achieved remarkable progress in the past few years. However, when applying those face recognition models to unconstrained scenarios, face recognition performance drops sharply, particularly when faces are occluded. The COVID-19 pandemic makes people have to wear masks on daily trips, which makes the face recognition performance of occlusion need improvement. Current methods for occluded face recognition are usually the variants of two sets of approaches, one is recovering occluded facial parts [ 143 , 144 , 145 , 146 ] , the other is removing features that
are corrupted by occlusions [ 147 , 148 , 149 ] . The pioneering works usually remove features that are corrupted. 

 
 
 FROM [ 150 ] proposed Mask Decoder and Occlusion Pattern Predictor networks to predict the occlusion patterns. The structure of FROM is shown in Fig.
 26 . 

 
 
 Figure 26: The architectures of FROM. 
 
 
 The structure first took a mini-batch images as input, through a Feature Pyramid Extractor got three different scale feature maps(including X 1 X_{1} , X 2 X_{2} , X 3 X_{3} ). Then X 3 X_{3} was used to decode the Mask, which contains the occlusion’s location information. Mask applied to X 1 X_{1} to mask out the corrupted feature elements and get the pure feature X 1 ′ X_{1}^{\prime} for the final recognition. Finally, Occlusion Pattern Predictor predicted occlusion patterns as the extra supervision. The whole network was trained end-to-end. 

 
 
 In order to encourage the network to recognize diverse occluded face, random occlusion was dynamically generated by sunglasses, scarf, face mask, hand, eye mask, eyeglasses etc. The overall loss was a combination of the face recognition loss and the occlusion pattern prediction loss. Mathematically, they defined total loss as follows: 

 

 
 | 
 L t ​ o ​ t ​ a ​ l = L m ​ a ​ r ​ g ​ i ​ n + λ ​ L p ​ r ​ e ​ d L_{total}=L_{margin}+\lambda L_{pred} | 
 | 
 

 
 
 L m ​ a ​ r ​ g ​ i ​ n L_{margin} is the cosFace loss, L p ​ r ​ e ​ d L_{pred} is MSE loss or Cross entropy loss. 

 
 
 Different from FROM, [ 147 ] adopted a multi-scale segmentation based mask learning (MSML) face recognition network, which alleviate the different scales of occlusion information and purify different scales of face information. The proposed MSML consisted of a face recognition branch (FRB), an occlusion segmentation branch (OSB), and hierarchical feature masking (FM) operators, as shown in Fig.
 27 . 

 
 
 Figure 27: The architectures of MSML. 
 
 
 Using scarves and glasses to randomly generate various occlusions at any position of the original images, meanwhile obtained binary segmentation labels. Different scales characteristics of occlusion was get through the OSB branch and the binary segmentation map was generated by the decoder. In the training stage, segmentation map was constrained by the segmentation loss. Occlusion features extracted by OSB and original face features extracted by FB were fused at FM module to get pure face feature. The whole network was trained through a joint optimization. Total loss can be formulated as: 

 

 
 | 
 L t ​ o ​ t ​ a ​ l = L c ​ l ​ s + λ ​ L o ​ c ​ c L_{total}=L_{cls}+\lambda L_{occ} | 
 | 
 

 L c ​ l ​ s L_{cls} is Cross-entropy loss or other SOTA face recognition losses, L o ​ c ​ c L_{occ} is a consensus segmentation loss. 

 
 
 Similar to MSML, [ 151 ] also integrated segmentation tasks to assist mask face recognition as shown in Fig.
 28 . 

 
 
 Figure 28: Outline of they proposed Model. 
 
 
 X denoted a masked face which through the whole backbone get a feature map F = D ​ c ​ o ​ n ​ v ​ ( X ) F=Dconv(X) , F can entirely or split into two subfeature maps in the channel dimension, one for occlusion prediction(OP) module and the other for identity embedding. The predict mask segmentation result was obtained through OP module. The segmentation task was constrained by segmentation loss. They proposed that the actual 2-D mask should be transformed into a 3-D mask which is more suitable for feature maps. Therefore, they proposed a channel refinement(CR) network for the transformation, the CR network can be defined as follows: 

 

 
 | 
 F r k = H ⁡ ( C ​ o ​ n ​ c ​ a ​ t ​ ( F r k − 1 , F p ( 4 − k ) , 2 ) ) F_{r}^{k}=H(Concat(F_{r}^{k-1},F_{p}^{(4-k),2})) | 
 | 
 (96) | 
 

 where C o n c a t ( . , . ) Concat(.,.) refers to the concatenation operation in channel dimension and k ​ ϵ ​ { 1 , 2 , 3 } k\epsilon\{1,2,3\} . F r k F_{r}^{k} represents the CR feature of the k t ​ h k^{th} layer. H(·) denotes the downsampling process.
The final goal was to produce discriminative facial features free from occlusion. They multiplied the occlusion mask map with original face features to filter corrupted feature elements for recognition, formulated as: 

 

 
 | 
 F n = F i ⊗ σ ⁡ ( F r 3 ) F_{n}=F_{i}\otimes\sigma(F_{r}^{3}) | 
 | 
 (97) | 
 

 where F i F_{i} denotes the identity feature extracted by the backbone, F r 3 F_{r}^{3} represents the output of the CR network, and σ \sigma is the sigmoid activation function. They used the loss function from CosFace to optimize the recognition network. 

 
 
 For face OP model, they introduced an occlusion-aware loss to guide the training of mask prediction, which can be formulated as: 

 

 
 | 
 L m ​ a ​ s ​ k = 1 n b ​ a ​ t ​ c ​ h ​ ∑ i = 1 h ∑ j = 1 w ‖ M i , j g ​ t − M i , j n ​ o ​ r ​ m ‖ 2 2 h ​ w L_{mask}=\frac{1}{n_{batch}}\sqrt{\frac{\sum_{i=1}^{h}\sum_{j=1}^{w}\|M_{i,j}^{gt}-M_{i,j}^{norm}\|^{2}_{2}}{hw}} | 
 | 
 (98) | 
 

 where M g ​ t M_{gt} refers to the occlusion mask labels in the training dataset, and n b ​ a ​ t ​ c ​ h n_{batch} is the batch size during the training stage. M n ​ o ​ r ​ m M^{norm} is the normalized mask map. The final loss function for the end-to-end training was defined as: 

 

 
 | 
 L = L i ​ d + α × L m ​ a ​ s ​ k L=L_{id}+\alpha\times L_{mask} | 
 | 
 (99) | 
 

 where α \alpha is a hyperparameter to trade off the classification and segmentation losses. 

 
 
 Consistent Sub-decision Network [ 152 ] proposed to obtain sub-decisions that correspond to different facial regions and constrain sub-decisions by weighted bidirectional KL divergence to make the network concentrate on the upper faces without occlusion. The whole network is shown as Fig.
 29 . 

 
 
 Figure 29: Outline of the Consistent Sub-decision Network. 
 
 
 The core of occluded face recognition is to lean a masked face embeddings which approximated normal face embeddings. So they proposed different dropout modules to obtain multiple sub-decisions. Every sub-decision uses a concept branch to get a face embeddings information degree ω \omega . [ 152 ] adopted simulation-based methods to generate masked faces from unmasked faces. However, among simulated faces, there were low-quality samples, which leads to ambiguous or absent facial features. The low sub-decision consistency values ω \omega correspond to low-quality samples in simulated face images. They applied the ω \omega as weights in the bidirectional KL divergence constraints. 

 

 
 | 
 L k ​ l = ∑ i j ( w i × K ​ L ​ ( s i , s j ) + w j × K ​ L ​ ( s j , s i ) ) L_{kl}=\sum_{i j}(w_{i}\times KL(s_{i},s_{j})+w_{j}\times KL(s_{j},s_{i})) | 
 | 
 (100) | 
 

 where s i s_{i} , s j s_{j} indicates different sub-decisions that correspond to diverse face regions of each image.
The normal face contains more discriminative identity information than the masked face, so they adopt knowledge distillation to drive the masked face embeddings towards an approximation of the normal face embeddings to mitigate the information loss, which can be formulated as: 

 

 
 | 
 f N = ℳ t ​ e ​ a ​ c ​ h ​ e ​ r ​ ( X N ) f^{N}=\mathcal{M}_{teacher}(X^{N}) | 
 | 
 (101) | 
 

 

 
 | 
 f M = ∑ i = 1 i = 3 ω i × 𝒟 i ​ ( ℊ ⁡ ( X M ) ) f^{M}=\sum_{i=1}^{i=3}\omega_{i}\times\mathcal{D}_{i}(\mathcal{g}(X^{M})) | 
 | 
 (102) | 
 

 where X N X^{N} is normal face, X M X^{M} is masked face, ℳ t ​ e ​ a ​ c ​ h ​ e ​ r \mathcal{M}_{teacher} is the embedding encoder of pretrained model, ℊ \mathcal{g} is the feature map generator, 𝒟 i \mathcal{D}_{i} is the i-th dropout block,
and w i w_{i} indicates the output of concept branch. The overall loss function is formulated as follows: 

 

 
 | 
 L = L c ​ l ​ s + λ 1 ​ L k ​ l + λ 2 ​ L k ​ d L=L_{cls}+\lambda_{1}L_{kl}+\lambda_{2}L_{kd} | 
 | 
 (103) | 
 

 L c ​ l ​ s L_{cls} is the CosFace loss, L k ​ l L_{kl} is the KL divergence constraints, L k ​ d L_{kd} is a cosine distance to perform knowledge distillation. 

 
 
 The champion of ICCV 2021-Masked Face Recognition (MFR) Challenge [ 148 ] proposed some contributions in industrial. They adopt the mask-to-face texture mapping approach and then generated the masked face images with rendering. They constructed a self-learning based cleaning framework which utilized the DBSCAN to realize Inter-ID Cleaning and Intra-ID Cleaning. In the cleaning procedure, they performed self-learning by initializing the i-th model as the (i+1)-th model. Besides they proposed a Balanced Curricular Loss. The loss adaptively adjusted the relative importance of easy and hard samples during different training stages. The Balanced Curricular Loss can be formulated as: 

 

 
 | 
 ℒ = − l ​ o ​ g ​ n y i ​ e s ​ c ​ o ​ s ​ ( θ y i + m ) n y i ​ e s ​ c ​ o ​ s ​ ( θ y i + m ) + ∑ j = 1 , j ≠ y i n n j ​ e s ​ N ​ ( t ( k ) , c ​ o ​ s ​ θ j ) \mathcal{L}=-log\frac{n_{y_{i}}e^{scos(\theta_{y_{i}}+m)}}{n_{y_{i}}e^{scos(\theta_{y_{i}}+m)}+\sum_{j=1,j\neq y_{i}}^{n}n_{j}e^{sN(t^{(k),cos\theta_{j}})}} | 
 | 
 (104) | 
 

 where c ​ o ​ s ​ ( θ y i + m ) cos(\theta_{y_{i}}+m) and N ⁡ ( t ( k ) , c ​ o ​ s ​ θ j ) N(t^{(k),cos\theta_{j}}) are the cosine similarity function of positive and negative [ 58 ] . 

 
 
 

### 4.8 Privacy-Preserving FR

 
 Face recognition technology has brought many conveniences to people’s daily life, such as public security, access control, railway station gate, etc.
However, misuse of this technique brings people hidden worries about personal data security.
Interview and investigation of some TV programs have called out several well-known brands for illegal face collection without explicit user consent.
Conversely, On the premise of ensuring user data privacy, personal client’s private data or public client’s data are beneficial to the training of existing models.
Federated Learning (FL) is a technique to address the privacy issue, which can collaboratively optimize the model without sharing the data between clients. 

 
 
 [ 153 ] is the pioneering work of Federated Learning. They proposed the problem of training on decentralized data from mobile devices as an important research direction.
Experiments demonstrated that the selection of a straightforward and practical algorithm can be applied to this setting.
They also proposed Federated Averaging algorithm, which is the basic framework of Federal Learning. 

 
 
 [ 153 ] defined the ideal problems for federated learning have three properties.
They also fingered out the federated optimization has several key properties which are differentiate from a typical distributed optimization.
The unique attributes of federated optimization are the Non-IID, unbalanced similarly, massively distributed, and limited communication. 

 
 
 They proposed a synchronous update scheme that proceeds in rounds of communication.
There is a fixed set of K clients, every client has a fixed local dataset. And there is T round communication.
The Federated Averaging algorithm is shown as below. 

 
 
 1)At the beginning of i-th round, a random fraction C of clients is selected, and the server
sends the current global mdoel parameters to each of these clients. 

 
 
 2) In every client compute a update through every local data and sent the local model parameters to the server. 

 
 
 3) The server collect and aggregate the local parameters and get a new global parameters. 

 
 
 4) Repeat the above three steps until the end of round T communication. 

 
 
 Their experiments showed diminishing returns for adding more clients beyond a certain point, so they only selected a fraction of clients for efficiency.
Their experiments on MNIST training set demonstrated that common initialization can produces a significant reduction than independent initialization, so they conducted independent initialization on the client. 

 
 
 Federated learning can alleviate the public’s concern about privacy data leakage to a certain extent. Because the base network parameters are updated by the client network parameters jointly, the performance of federated learning is not as good as the ordinary face recognition.
Therefore, the PrivacyFace [ 154 ] proposed a method to improve the performance of federated face recognition by using differential privacy clustering, desensitizing the local facial feature space information and sharing it among clients, thus greatly improving the performance of federated face recognition model.
PrivacyFace was mainly composed of two components: the first is the Differently Private Local Clustering (DPLC) algorithm based on differential privacy, which desensitizes privacy independent group features from many average face features of the client; the second is the consensus-aware face recognition loss, which makes use of the desensitization group features of each client to promote the global feature space distribution. 

 
 
 Figure 30: Framework of classical FL and the FL framework with DPLC. 
 
 
 As show in Fig. 30 , in the general FL framework, average face space W c W^{c} cannot be updated during communication, the global optimization of average face space cannot be carried out, resulting in the problem of face space overlap between clients. 

 
 
 The core of their DPLC approximation algorithm was to find the most dense location centered on each sample, and then add noise processing to meet differential privacy. 

 
 
 After the local client has calculated the desensitized cluster center, the server will share the information of each client. Then the local client uses the local data and the group information of other clients to perform the global optimization of the space according to the newly designed consciousness loss. 

 
 
 

 
 | 
 L c ( ϕ c , W c ) = − ∑ i = 1 N c e u ⁡ ( w y i c , f i c ) e u ⁡ ( w y i c , f i c ) + ∑ j = 1 , j ≠ y i c n c e v ⁡ ( w j , f i c ) + ∑ k = 1 , k ≠ c C ∑ l = 1 Q k e μ ⁡ ( p ^ l k , f i c , ρ ) L^{c}(\phi^{c},W^{c})=-\sum_{i=1}^{N_{c}}\frac{e^{u(w_{y_{i}^{c}},f_{i}^{c})}}{e^{u(w_{y_{i}^{c}},f_{i}^{c})}+\sum_{j=1,j\neq y_{i}^{c}}^{n_{c}}e^{v(w_{j},f_{i}^{c})}+\sum_{k=1,k\neq c}^{C}\sum_{l=1}^{Q_{k}}e^{\mu(\hat{p}_{l}^{k},f_{i}^{c},\rho)}} | 
 | 
 (105) | 
 

 There are C clients,where f i c f_{i}^{c} is the feature extracted by i-th model instance in client c.
C owns a training dataset D c D_{c} with N c N_{c} images from n c n_{c} identities.
For client c, its backbone is parameterized by ϕ c \phi^{c} and the last classifier layer encodes class centers in W c W^{c} = [ w 1 c , w 2 c ​ … ​ w n c c ] [w_{1}^{c},w_{2}^{c}...w_{n_{c}}^{c}] .
 μ ⁡ ( p ^ , f , ρ ) \mu(\hat{p},f,\rho) = s × c ​ o ​ s ​ ( m ​ a ​ x ​ ( θ − ρ , 0 ) ) s\times cos(max(\theta-\rho;0)) computes the similarity between f and the cluster centered at p ^ \hat{p} with margin ρ \rho .
 Q c Q_{c} clusters defined by the centers P c P^{c} = [ p ^ 1 ​ … ​ p ^ Q c ] [\hat{p}_{1}...\hat{p}_{Q_{c}}] with margin ρ \rho , which is generated in each client. 

 
 
 FedGC [ 155 ] explored a new idea to modify the gradient from the perspective of back propagation, and proposed a regularizer based on softmax, which corrects the gradient of class embeddings by accurately injecting cross client gradient terms. In theory, they proved that FedGC constitutes an effective loss function similar to the standard softmax.
Class embeddings W was upgraded as: 

 
 
 

 
 | 
 W t + 1 = W ~ t + 1 − λ ​ ϵ ​ ∇ W ~ t + 1 R ​ e ​ g ​ ( W ~ t + 1 ) W^{t+1}=\tilde{W}^{t+1}-\lambda\epsilon\nabla_{\tilde{W}^{t+1}}Reg(\tilde{W}^{t+1}) | 
 | 
 (106) | 
 

 
 
 As we can see in the Fig. 31 , gradient correction term ensures the update moves towards the true optimum. 

 
 
 Figure 31: (a) is the divergence between FedPE and SGD, (b) is gradient correction term. 
 
 
 Figure 32: Framework of classical FL and the FL framework with DPLC. 
 
 
 The whole training schedule is shown in Fig. 32 .
In the t-th round communication, the server broadcast model parameters ( θ t , W k t ) (\theta^{t},W_{k}^{t}) to the selected clients.
The clients compute a update model parameters with the local data asynchornously, and then sent the new model parameters to the server.
Finally, The server conduct cross-client optimization by collecting and aggregating the client updates.
The process of server optimization can make correct gradients and make cross-client embeddings spreadout. 

 
 
 [ 156 ] proposed a frequency domain method to achieve Pricacy-Preserving.
As show in Fig. 33 , the low frequency of the picture directly affects the human visual perception of the picture, and accounts for most of the semantic. [ 157 ] proposed existing deep neural network based FR systems rely on both low- and high-frequency components.
Yellow rectangles represent frequency components that really contribute to face recognition.
We need a trade-off analysis network to get the corresponding frequency components. 

 
 
 Figure 33: Image perception in the frequency domain. 
 
 
 Different from some cryptography based method [ 158 , 159 , 160 , 161 ] , [ 156 ] proposed a frequency-domain
privacy-preserving FR scheme, which integrated an analysis network to collect the components with
the same frequency from different blocks and a fast masking method to further protect the remaining frequency components.
As show in Fig. 34 , the proposed privacy-preserving method includes client data processing part and cloud server training part. 

 
 
 Figure 34: Framework of privacy-preserving FR. 
 
 
 The analysis network is show in Fig. 35 (a), the first step is block discrete cosine transform (BDCT), which is carried out on the face image obtained after converting it from a color image to a gray one. Then send the BDCT frequency component channels into the channel selection module.
In this work, for simplicity, they chose to remove a pre-fixed subset of frequency component channels that span low to relatively high frequencies.
The whole network was constrained by the following Loss function. 

 

 
 | 
 L ​ o ​ s ​ s a ​ n ​ a ​ l ​ y ​ s ​ i ​ s − n ​ e ​ t ​ w ​ o ​ r ​ k = L ​ o ​ s ​ s F ​ R + λ ​ L ​ o ​ s ​ s p ​ r ​ i Loss_{analysis-network}=Loss_{FR}+\lambda Loss_{pri} | 
 | 
 (107) | 
 

 L ​ o ​ s ​ s F ​ R Loss_{FR} is the loss function of FR, such as arcface, CosFace, and λ \lambda is a hyper
parameter. L ​ o ​ s ​ s p ​ r ​ i Loss_{pri} is a loss function to quantify face images’ privacy protection level. 

 
 
 

 
 | 
 L ​ o ​ s ​ s p ​ r ​ i = ∑ n = 1 M R ​ e ​ L ​ u ​ ( a i − γ ) ​ p Loss_{pri}=\sum_{n=1}^{M}ReLu(a_{i}-\gamma)p | 
 | 
 (108) | 
 

 
 
 where a i a_{i} is the trainable weight coefficient for the i t ​ h i_{th} channel,
p is the energy of channel i, denotes a threshold, and
M is the number of considered frequency channels. Clearly, if a i a_{i} is less than γ \gamma 
, the corresponding channel is considered unimportant in terms of its contribution to the loss function L ​ o ​ s ​ s p ​ r ​ i Loss_{pri} . This is realized by the use of ReLu(y) function, which becomes zero when y is negative. 

 
 
 The whole diagram of the face image masking method is shown in Fig. 35 (b). It performs the BDCT and selects channels according to the analysis network.
Next, the remaining channels are shuffled two times with a channel mixing in between.
After each shuffling operation, channel self-normalization is performed.
The result of the second channel self-normalization is the masked face image that will be transmitted to third-party servers for face recognition.
The proposed face masking method aims at further increasing the difficulty in recovering the original face image from its masked version. Face fast masking feature then feed into the cloud server to finetune the model, as show in Fig. 34 . 

 
 
 Figure 35: Schematic diagrams of (a) the proposed analysis network and (b) the proposed fast face image masking method. 
 
 
 As we know that a model is predominantly trained on RGB images, it generalizes poorly for images captured by infrared cameras.
Likewise, for a model pre-trained on Caucasian only, it performs substantially worse for African and Indian.
 [ 162 ] proposed a Local-Adaptive face recognition method to properly adapt the pre-trained model to a ’specialized’ one that tailors for the specific environment in an automatic and unsupervised manner.
As show in Fig. 36 ,
it starts from an imperfect pre-trained global model deployed to a specific environment.
The first step of [ 162 ] was to train a graph-based meta-clustering model, which refer to [ 163 ] .
The conventional GCN training is via the following equation: 

 

 
 | 
 ϕ ′ = ϕ − α ∇ ϕ L m ​ t ​ r g ( ϕ ) \phi^{{}^{\prime}}=\phi-\alpha\nabla_{\phi}L_{mtr}^{g}(\phi) | 
 | 
 (109) | 
 

 They changed the usually GCN training strategy to a meta-learning, with the objective loss function: 

 

 
 | 
 ϕ = ϕ − β ( ∇ ϕ L m ​ t ​ r g ( ϕ ) + ξ ∇ ϕ L m ​ t ​ e g ( ϕ ′ ) ) \phi=\phi-\beta(\nabla_{\phi}L_{mtr}^{g}(\phi)+\xi\nabla_{\phi}L_{mte}^{g}(\phi^{{}^{\prime}})) | 
 | 
 (110) | 
 

 The client private data label are obtained by running the meta-clustering model.
The pseudo ’identity’ labels as well as their corresponding images are used to train the adapted model θ A \theta_{A} from an imperfect model θ 0 \theta_{0} .
They employed AM-softmax as the training objective to optimization initial imperfect model, the loss function is as below. 

 

 
 | 
 L ( θ ) = − 1 N ∑ i = 1 N l o g e γ ⁡ ( c ​ o ​ s ​ ( θ y i − m ) ) e γ ⁡ ( c ​ o ​ s ​ ( θ y i − m ) ) + ∑ j ≠ − y i C e γ ​ c ​ o ​ s ​ ( θ y i ) L(\theta)=-\frac{1}{N}\sum_{i=1}^{N}log\frac{e^{\gamma(cos(\theta_{y_{i}}-m))}}{e^{\gamma(cos(\theta_{y_{i}}-m))}+\sum_{j\neq-y_{i}}^{C}e^{\gamma cos(\theta_{y_{i}})}} | 
 | 
 (111) | 
 

 where C is the total number of classes, and m is the margin that needs empirically determined. 

 
 
 Figure 36: Local-Adaptive Face Recognition (LaFR). 
 
 
 They thinked that the initial mdoel θ 0 \theta_{0} is already has strong discriminative power,
so they transfered this pre-trained class center as prior knowledge to the adapted model θ A \theta_{A} .
They wanted to have C y i θ A C_{y_{i}^{\theta_{A}}} to be as close to C y i θ 0 C_{y_{i}^{\theta_{0}}} as possible during the adaptation.
The C y i θ 0 C_{y_{i}^{\theta_{0}}} denotes as the center of face face embedding for y i y_{i} on the pre-trained model. 

 

 
 | 
 C y i θ 0 = 1 M i ​ 1 ​ ( y k = y i ) ​ f k θ 0 C_{y_{i}^{\theta_{0}}}=\frac{1}{M_{i}1(y_{k}=y_{i})f_{k}^{\theta_{0}}} | 
 | 
 (112) | 
 

 To further reduce the overfitting risk when adapting to a small dataset, they added another model regularization term to let θ A \theta_{A} not deviate too much from θ 0 \theta_{0} .
They final loss function was defined as follow: 

 

 
 | 
 L ( θ ) = − 1 N ∑ i = 1 N l o g e γ ⁡ ( c ​ o ​ s ​ ( θ y i − m ) ) e γ ⁡ ( c ​ o ​ s ​ ( θ y i − m ) ) + ∑ j ≠ − y i C e γ ​ c ​ o ​ s ​ ( θ y i ) + λ | θ A − θ 0 | 2 2 L(\theta)=-\frac{1}{N}\sum_{i=1}^{N}log\frac{e^{\gamma(cos(\theta_{y_{i}}-m))}}{e^{\gamma(cos(\theta_{y_{i}}-m))}+\sum_{j\neq-y_{i}}^{C}e^{\gamma cos(\theta_{y_{i}})}}+\lambda\lvert\theta_{A}-\theta_{0}\rvert_{2}^{2} | 
 | 
 (113) | 
 

 subject to 

 

 
 | 
 W = W ∗ / ‖ W ∗ ‖ , W=W^{*}/||W^{*}||, | 
 | 
 (114) | 
 

 

 
 | 
 f = f ∗ / ‖ f ∗ ‖ , f=f^{*}/||f^{*}||, | 
 | 
 (115) | 
 

 

 
 | 
 c ​ o ​ s ​ θ j = W j T ​ f i , cos\theta_{j}=W_{j}^{T}f_{i}, | 
 | 
 (116) | 
 

 

 
 | 
 W y i = C y i θ 0 W_{y_{i}}=C_{y_{i}}^{\theta_{0}} | 
 | 
 (117) | 
 

 
 
 where W is the normalized clssifier matrix, λ \lambda is a hyperparameter to trade-off the face loss and model regularization term.
They initialize each W y i W_{y_{i}} with C y i θ 0 C_{y_{i}^{\theta_{0}}} . 

 
 
 As show in Fig. 37 ,
it shows the training process of the agent’s adaptive model.
First, GCN model is obtained through a meta-cluster model training, which used to get the client privacy-data’s pseudo label.
Second, the regularized center transfer (RCT) initialization method is used to optimize the pretrained model to obtain a local adaptive model.
Third, the locally client are aggregated via federated learning [ 164 ] in a secure way. 

 
 
 Figure 37: Overview of our Local-Adaptive face
recognition (LaFR) framework. 
 
 
 FedFR [ 165 ] proposed a new FL joint optimization framework for generic and personalized face recognition.
As we all konw, the universal FedAvg framework usually adopt public face recognition datasets used for the training of server model.
The server model performance may degenerated during the communication with local agent. 

 
 
 FedFR supposed to simultaneously improve the general face representation at the center server, and generate an optimal personalized model for each client without transmitting private identities’ images or features out of the local devices.
As show in Fig. 38 , they add some novel ideas on the classical FL framework. 

 
 
 First, they proposed a hard negative sampling which can auxiliary local agent training.
The D H ​ N t D_{HN}^{t} represent the hard negative samples selected from the global shared public dataset.
They calculated the pair-wise cosine
similarity between the proxy and the global data and local data features. Then selected the similarity score large than a threshold.
 D H ​ N t D_{HN}^{t} and the local data were used for training the local client, which can prevent θ l t \theta_{l}^{t} overfitting on local data. 

 
 
 Second, they proposed a contrastive regularization on local clients, which can be used for optimizing the server’s general recognition ability. 

 
 
 Figure 38: Overview of the FedFR framework. 
 
 
 The contrastive regularization formula is shown as below: 

 
 
 

 
 | 
 L c ​ o ​ n = − l ​ o ​ g ​ e ​ x ​ p ​ ( s ​ i ​ m ​ ( f , f g ​ l ​ o ​ b ) / ϕ ) e ​ x ​ p ​ ( s ​ i ​ m ​ ( f , f g ​ l ​ o ​ b ) / ϕ + e ​ x ​ p ​ ( s ​ i ​ m ​ ( f , f p ​ r ​ e ​ v ) / ϕ CLOSE CLOSE L_{con}=-log\frac{exp(sim(f,f_{glob})/\phi)}{exp(sim(f,f_{glob})/\phi+exp(sim(f,f_{prev})/\phi} | 
 | 
 (118) | 
 

 
 
 f g ​ l ​ o ​ b f_{glob} means the global
model ( f g ​ l ​ o ​ b = θ g t ​ ( x ) f_{glob}=\theta_{g}^{t}(x) ), and the f p ​ r ​ e ​ v f_{prev} represent the face representation learned by the local model at time t-1.
Namely, the expected to decrease the distance
between the face representation learned by the local model
at time t t and the one learned by the global model, and increase the distance between the face representation learned by the local model at time t t and time t − 1 t-1 . 

 
 
 Third, they added a new decoupled feature customization module to achieve the personalization, which can improve the local user experience.
They adopt a transformation with a fully-connected
layer to map the global feature to a client-specific feature space, which can recognize the k l k_{l} identities well.
Given the transformed feature f, they feed it into k l k_{l} binary classification branches. Every branch module contains learnable parameters which only target on classifying the positive samples from the k-th class and the negative samples from “any other” classes. 

 
 
 
 

## 5 Backbone size and data distribution

 
 Previous parts provide a comprehensive survey of face recognition algorithms, but rarely mentioned the effect caused by the backbone size and the distribution of training data, which is as important as the former. Different from algorithms which are mostly designed for a specific hypothesis, backbone size and data distribution can affect all scenarios. In this section, we will discuss them from
three aspects: backbone size, data depth and breadth, and long tail distribution. 

 
 

### 5.1 Backbone size

 
 It is widely known that training on a large dataset can improve algorithm performance.
However, for a specific backbone, when the training data size achieves a certain amount, its performance is no longer able to be significantly enhanced by adding data. Meanwhile, the training cost is dramatically increased. Therefore, we aimed at figuring out the effect on the performance of different algorithms caused by increasing training data amount. 

 
 
 We chose Iresnet50, Iresnet100 and Mobilefacenet as the backbone
and selected 10%, 40%, 70% and 100% ids respectively from Webface42m as training data.
We adopted Arcface loss and PartialFC operation to achieve convergence.
The network is trained on 8 Tesla V100 GPUs with a
batch size of 512 and a momentum of 0.9. We employed
SGD as the optimizer. The weight decay is set to 5 ​ e − 4 5e-4 .
We evaluated the model on four test datasets the LFW, the AgeDB, the CFP-FP, and the IJB-C. 

 
 
 The results are shown in Fig. 39 . For the Mobilefacenet, as the sample rate increases from 10% to 40%, model performance obviously enhances on four test datasets, from 99.75% to 99.80% on LFW, from 97.13% to 97.92% on AgeDB, from 98.73% to 98.99% on CFP-FP, and from 95.29% to 96.46% on IJB-C. When the sample rate is over 40%, the performance of Mobilefacenet remains stable. For the Iresnet50, the turning point is 70% sample rate. While, the performance of the Iresnet100 improves slightly and continually with the increase in training data. For three different backbones, it is clear that the model performance improves as the amount of training data increases. 

 
 
 
 
 
 (a) Mobilefacenet 
 
 
 (b) Iresnet50 
 
 
 (c) Iresnet100 
 
 Figure 39: Results of models with different backbone sizes on LFW, AgeDB, CFP-FP, and IJB-C test datasets. 
 
 
 

### 5.2 Data depth and breadth

 
 For data collection, we can collect images from a limited number of subjects, but each subject contains lots of images, which improves the depth of a dataset and ensures the intra-class variations of a dataset. We also can gather images from lots of subjects, but only collect a limited number of images for each subject, which improves the breadth of a dataset and allows the algorithm to be trained by sufficient different identities. Previous works [ 166 ] mentioned the two kinds of data collection methods, however, did not discuss their influence. In industry, improving the breadth is easier than improving the depth. Therefore, in this part, we aimed at comparing the importance of data breadth and data depth when the number of training data is fixed, which can guide the formulation of a data acquisition strategy. 

 
 
 We chose Iresnet100 as the backbone, which had enough ability to express the distribution of training data.
We had four settings, the product of person numbers and images of each person is fixed to 80k in every setting.
So the four settings can be expressed as 1w_80, 2w_40, 4w_20 and 8w_10.
For example, 1w_80 meaned that 10k person and each id contained 80 images.
The training details were the same as section 5.1. 

 
 
 Fig. 40 compares the model performance with different training data distributions on four test sets. From “1w_80” to “4w_20”, we can see that the model performance rises dramatically, from 99.37% to 99.70% on LFW, from 95.73% to 97.43% on AgeDB, from 93.67% to 95.91% on CFP-FP, and from 88.14% to 94.14% on IJB-C. But when the data width is 8w and the depth is 10, the performance declines sharply on CFP-FP (92.41%) and IJB-C (92.47%). Based on the results, there is a balance point between the depth of training data and the width of training data. 

 
 
 Figure 40: Results on LFW, AgeDB, CFP-FP, and IJB-C. N_S implies that the corresponding dataset has N identities with S samples per identity. 
 
 
 

### 5.3 Long tail distribution

 
 The purity and long-tail distribution of training data were essential factors affecting the performance of state-of-the-art face recognition models. In order to explore the impact of these two factors on the performance of face recognition models, we conducted self-learning based cleaning similar with [ 148 ] to get a pure dataset WebFace35M.
The details are introduced as follows:
(1) An initial model is first trained with the WebFace42M
to clean the original dataset, which mainly consist of DBSCAN-cluster,
and Intra-ID cleaning;
(2) Then the the i-th model is trained on the cleaned datasets from (1);
(3) We iterate this process by initializing the i-th model as the
(i+1)-th model.
Different from [ 148 ] , we only conducted Intra-ID cleaning.
Because WebFace42M is derived from WebFace260M, and through our manual observation, we found that WebFace42M contained intra-id noisy.
We conducted DBSCAN-cluster on each folder and eliminated data that didn’t belong to this category to achieve intra-id cleaning.
Through two rounds of data cleaning, we got the WebFace35M, which further filtered out the ID with images numberes less than 10 as the long tail data.
The following long tail distribution and noisy experiment were based on the WebFace35M. 

 
 
 In the previous chapters, we have discussed long-tail distribution and how to alleviate its effect. It is a typical unbalanced distribution and is defined as a dataset containing a large number of identities that have a small number of face images. Increasing the number of identities can increase training costs due to the last fully connected Layer. In this part, we carefully designed a set of experiments to visually display the effect of the distribution. 

 
 
 We chose Iresnet100 as the backbone and then we added 0%, 25%, 50%, 75% and 100% long tail data from id dimension.
The training details were the same as section 5.1. 

 
 
 The results are shown in Fig. 41 . With the addition of long tail data, model performance keeps steady. The evaluation results on LFW, AgeDB, CFP-FP, and the IJB-C are roughly 99.8%, 98.5%, 99.3%, and 97.5%, respectively. 

 
 
 Figure 41: Results on LFW, AgeDB, CFP-FP, and IJB-C with different percentage of long-tail data.
 
 
 
 
 

## 6 Datasets and Comparison Results

 

### 6.1 Training datasets

 
 In this part, we list all the major training datasets for FR with their details.
All information can be checked in Table 1 , including the number of images and identities in these training sets. 

 
 
 
 
 
 Training Dataset | 
 IDs | 
 Images | 
 Details | 

 
 
 
 CASIA-Webface [ 167 ] | 
 10,575 | 
 494,414 | 
 
 
 A semi-automatical way is used to collect face images from Internet 
 | 

 
 [0.5pt/1pt]
CelebA [ 168 ] | 
 10k celebs | 
 0.2M | 
 | 

 
 [0.5pt/1pt]
UMDFaces [ 169 ] | 
 8,277 | 
 367,888 | 
 
 
 A semi-automatical annotation procedure is used to the images crawled from Yahoo, Yandex, Google and Bing. The annotation of this set includes face ID, face bounding box, 21 face landmarks, face pose and gender. 
 | 

 
 [0.5pt/1pt]
vggface [ 33 ] | 
 2,622 | 
 2.6M | 
 | 

 
 [0.5pt/1pt]
vggface2(VGG2) [ 170 ] | 
 9,131 | 
 3.31M | 
 
 
 VGG2 consists of a training set with 8,631 identities (3,141,890 images) and a test set with 500 identities (169,396 images). The annotation of this set includes face ID, face bounding box, 5 face landmarks, predicted face pose and age. 
 | 

 
 [0.5pt/1pt]
MS-Celeb-1m(MS1M) [ 171 ] | 
 100k celebs | 
 10M | 
 | 

 
 [0.5pt/1pt]
MS1M-ibug [ 172 ] | 
 85k celebs | 
 3.8M | 
 
 
 This dataset is also named as MS1MV1, which is obtained by cleansing on MS1M. 
 | 

 
 [0.5pt/1pt]
MS1M-ArcFace [ 51 ] | 
 85k celebs | 
 5.8M | 
 
 
 This dataset is also named as MS1MV2, which is obtained by cleansing on MS1M. 
 | 

 
 [0.5pt/1pt]
Asian-Celeb | 
 94k celebs | 
 2.8M | 
 
 
 This dataset has been excluded from both LFW and MS-Celeb-1M-v1c. 
 | 

 
 [0.5pt/1pt]
Glint360K [ 112 ] | 
 360k | 
 17M | 
 | 

 
 [0.5pt/1pt]
DeepGlint | 
 181k | 
 6.75M | 
 
 
 This dataset is obtained by merging MS1M and Asian-Celeb with data cleansing. 
 | 

 
 [0.5pt/1pt]
IMDB-Face [ 173 ] | 
 59k | 
 1.7M | 
 
 
 The images in this set is collected from IMDb website. 
 | 

 
 [0.5pt/1pt]
Celeb500k [ 174 ] | 
 500k | 
 50M | 
 | 

 
 [0.5pt/1pt]
WebFace260M [ 175 ] | 
 4M | 
 260M | 
 | 

 
 [0.5pt/1pt]
MegaFace(train) [ 176 ] | 
 672k | 
 4.7M | 
 
 
 This No longer being distributed from official website 
 | 

 

 Table 1: Information about training datasets in FR 
 
 
 

### 6.2 Testing datasets and Metrics

 
 In this part, we first list all commonly used test sets in table 2 .
Then we will introduce the evaluation metrics/protocols on these set.
And we will provide some evaluation results of each state-of-the-art on these metrics. 

 
 
 
 
 
 Test Dataset | 
 IDs | 
 Images | 
 Details | 

 
 
 
 LFW [ 177 ] | 
 5749 | 
 13,233 | 
 | 

 
 [0.5pt/1pt]
CPLFW [ 178 ] | 
 - | 
 - | 
 
 
 For cross-pose challenge in FR. 
 | 

 
 [0.5pt/1pt]
CALFW [ 179 ] | 
 - | 
 - | 
 
 
 Both CPLFW and CALFW are derivative datasets of LFW, addressing cross-pose and cross-age challenge in FR. They contains different 6k positive and negative pairs. 
 | 

 
 [0.5pt/1pt]
YTF [ 180 ] | 
 1,595 | 
 3,424 videos | 
 | 

 
 [0.5pt/1pt]
MegaFace gallery | 
 0.69M | 
 more than 1M | 
 | 

 
 [0.5pt/1pt]
Facescrub [ 181 ] | 
 530 celebs | 
 106,863 | 
 
 
 The images were retrieved from the Internet and are taken under real-world situations (uncontrolled conditions). Name and gender annotations of the faces are included. 
 | 

 
 [0.5pt/1pt]
FGNet [ 182 ] | 
 82 | 
 1002 | 
 
 
 The dataset includes lots of face images at the age phase of the child and the elderly,with age from 0 to 69. 
 | 

 
 [0.5pt/1pt]
CACD [ 183 ] | 
 2,000 celebs | 
 163,446 | 
 
 
 CACD-VS is a subset of CACD, which consists of 4000 face image pairs with different ages for age-invariant face verification, and the face pairs are divided into 2,000 positive pairs and 2,000 negative pairs. 
 | 

 
 [0.5pt/1pt]
MORPH Album 2 [ 184 ] | 
 20,000 | 
 78,000 | 
 
 
 Containing individuals across different ages. 
 | 

 
 [0.5pt/1pt]
CFP [ 185 ] | 
 500 | 
 7,000 | 
 
 
 Each ID contains 10 frontal and 4 profile images. 
 | 

 
 [0.5pt/1pt]
IJB-A [ 186 ] | 
 500 | 
 
 
 
 5,396 images | 

 
 20,412 frames | 

 | 
 
 
 Proposed by NIST(National Institute of Standards and Technology) 
 | 

 
 [0.5pt/1pt]
IJB-B | 
 1,845 | 
 
 
 
 21,798 images | 

 
 55K frames | 

 | 
 
 
 IJB-B is an extension of the IJB-A. 
 | 

 
 [0.5pt/1pt]
IJB-C | 
 3,531 | 
 
 
 
 31,334 images | 

 
 117,542 frames | 

 
 11779 videos | 

 | 
 
 
 IJB-C is derived from IJB-A, and it includes 10040 non human face images. 
 | 

 
 [0.5pt/1pt]
Multi-PIE [ 187 ] | 
 337 | 
 754,204 | 
 
 
 Identities in this set is from 15 view points and 20 illumination conditions for evaluating pose invariant FR. 
 | 

 
 [0.5pt/1pt]
AGE-DB [ 188 ] | 
 568 celebs | 
 16,488 | 
 
 
 The annotation contains age information. 
 | 

 

 Table 2: Information about test datasets in FR 
 
 
 Here, we give some metrics used by latest FR papers. 

 
 
 1, verification accuracy from unrestricted with labeled outside data protocol [ 189 ] .
In this metrics, a list of face pairs is be provided, with binary labels which shows each pair sharing same ID or not. And the accuracy is the proportion of right prediction by evaluating algorithm.
LFW is a typical set for testing verification accuracy. 6000 face pairs from LFW/CALFW/CPLFW are provided for evaluation.
The metrics of verification accuracy has also been applied in other test datasets, such as CFP-FP and AgeDB.
CFP-FP is a verification experiment from dataset CFP, which contains face frontal-profile pairs.
YTF provides 5000 video pairs for evaluation on this metrics.
Here we list some comparison results of state-of-the-arts in verification accuracy in Table 3 .
We will not further list evaluation results of all methods in FR on verification accuracy, because different methods trained on different training sets with different backbones. 

 
 
 
 
 
 Method | 
 LFW | 
 AgeDB | 
 CFP-FP | 
 CALFW | 
 CPLFW | 

 
 
 
 Softmax | 
 99.45 | 
 96.58 | 
 92.67 | 
 93.52 | 
 86.27 | 

 
 Center loss [ 45 ] | 
 99.65 | 
 96.83 | 
 93.37 | 
 94.23 | 
 86.58 | 

 
 Triplet loss [ 34 ] | 
 99.58 | 
 96.27 | 
 92.30 | 
 93.27 | 
 85.07 | 

 
 UniformFace [ 60 ] | 
 99.70 | 
 96.90 | 
 94.34 | 
 94.40 | 
 87.45 | 

 
 SphereFace [ 47 ] | 
 99.70 | 
 96.43 | 
 93.86 | 
 94.17 | 
 87.81 | 

 
 CosFace [ 50 ] | 
 99.73 | 
 97.53 | 
 94.83 | 
 95.07 | 
 88.63 | 

 
 ArcFace [ 51 ] | 
 99.75 | 
 97.68 | 
 94.27 | 
 95.12 | 
 88.53 | 

 
 AdaCos [ 64 ] | 
 99.68 | 
 97.15 | 
 94.03 | 
 94.38 | 
 87.03 | 

 
 AdaM-Softmax [ 66 ] | 
 99.74 | 
 97.68 | 
 94.96 | 
 95.05 | 
 88.80 | 

 
 MV-softmax [ 54 ] | 
 99.72 | 
 97.73 | 
 93.77 | 
 95.23 | 
 88.65 | 

 
 ArcNegFace [ 190 ] | 
 99.73 | 
 97.37 | 
 93.64 | 
 95.15 | 
 87.87 | 

 
 CurricularFace [ 58 ] | 
 99.72 | 
 97.43 | 
 93.73 | 
 94.98 | 
 87.62 | 

 
 CircleLoss [ 70 ] | 
 99.73 | 
 - | 
 96.02 | 
 - | 
 - | 

 
 NPCFace [ 59 ] | 
 99.77 | 
 97.77 | 
 95.09 | 
 95.60 | 
 89.42 | 

 
 MagFace [ 69 ] | 
 99.83 | 
 98.17 | 
 98.46 | 
 96.15 | 
 92.87 | 

 
 AdaFace [ 71 ] | 
 99.80 | 
 97.90 | 
 99.17 | 
 96.05 | 
 94.63 | 

 

 Table 3: Verification accuracy ( % \% ) on easy benchmarks 
 
 
 2, testing benchmark of MegaFace dataset.
MegaFace test dataset contains a gallery set and a probe set. The
gallery set contains more than 1 million images from 690K
different individuals.
The probe set consists of two existing datasets: Facescrub [ 181 ] and FGNet.
MegaFace has several testing scenarios including identification, verification and
pose invariance under two protocols (large or small training set).
(The training set is viewed as small if it is less than 0.5M.)
For face identification protocol, CMC (cumulative match characteristic) and ROC (receiver operating characteristics) curves will be evaluated.
For face verification protocol, “Rank-1 Acc.” and “Ver.” will be evaluated. “Rank-1 Acc.” indicates rank-1 identification accuracy with 1M distractors.
“Ver.” indicates verification TAR for 10-6 FAR. TAR and FAR denote True Accept Rate and False Accept Rate respectively. 

 
 
 3, IJB-A testing protocol.
This protocol includes ‘compare’ protocol for 1:1 face verification and the ‘search’ protocol for 1:N face identification.
For verification, the true accept rates (TAR) vs. false positive rates (FAR) are reported.
For identification, the true positive identification rate (TPIR) vs. false positive identification rate (TPIR) and the Rank-N accuracy are reported.
Recently, less papers published their results on IJB-A, since they have reached a high performance on it.
The protocol on IJB-A can also be applied on IJB-B and IJB-C.
The comparison results on IJB-B and IJB-C (TAR@FAR=1e-4, FAR=1e-5) are shown in Table . 

 
 
 
 
 
 Method | 
 IJB-B 1e-4 | 
 IJB-B 1e-5 | 
 IJB-C 1e-4 | 
 IJB-C 1e-5 | 

 
 
 
 Softmax | 
 85.66 | 
 73.63 | 
 86.62 | 
 76.48 | 

 
 Center loss [ 45 ] | 
 86.43 | 
 74.16 | 
 86.87 | 
 76.64 | 

 
 Triplet loss [ 34 ] | 
 73.21 | 
 40.37 | 
 78.12 | 
 48.07 | 

 
 UniformFace [ 60 ] | 
 87.22 | 
 75.01 | 
 88.87 | 
 79.64 | 

 
 SphereFace [ 47 ] | 
 86.67 | 
 74.75 | 
 87.92 | 
 78.77 | 

 
 CosFace [ 50 ] | 
 90.60 | 
 82.28 | 
 91.72 | 
 86.68 | 

 
 ArcFace [ 51 ] | 
 90.83 | 
 82.68 | 
 91.82 | 
 85.75 | 

 
 AdaCos [ 64 ] | 
 86.04 | 
 73.34 | 
 87.53 | 
 78.91 | 

 
 AdaM-Softmax [ 66 ] | 
 90.54 | 
 82.70 | 
 91.64 | 
 86.84 | 

 
 MV-softmax [ 54 ] | 
 90.67 | 
 83.17 | 
 92.03 | 
 87.52 | 

 
 ArcNegFace [ 190 ] | 
 90.62 | 
 81.59 | 
 90.91 | 
 85.64 | 

 
 CurricularFace [ 58 ] | 
 90.04 | 
 81.15 | 
 90.95 | 
 84.63 | 

 
 NPCFace [ 59 ] | 
 92.02 | 
 85.59 | 
 92.90 | 
 88.08 | 

 
 MagFace [ 69 ] | 
 94.51 | 
 90.36 | 
 95.97 | 
 94.08 | 

 
 AdaFace [ 71 ] | 
 96.03 | 
 - | 
 97.39 | 
 - | 

 

 Table 4: Verification accuracy ( % \% ) on IJB-B and IJB-C. 
 
 
 The aforementioned protocols are used to evaluating FR methods without any restriction.
On the contrary, the Face Recognition Under Inference Time conStraint (FRUITS) protocol [ 175 ] is designed to comprehensively evaluate FR systems (face matchers in verification way) with time limitation.
In particular, the protocol FRUITS-x (x can be 100, 500 and 100) evaluates a whole FR system which must distinguish
image pairs within x milliseconds, including preprocessing (detection and alignment), feature embedding, and matching.
FRUITS-100 targets on evaluating lightweight FR system which can be deployed on mobile devices.
FRUITS-500 aims to evaluate modern and popular networks deployed in the local surveillance system.
FRUITS-1000 aims to compare capable recognition models performed on clouds. 

 
 
 More specific metrics in the other competitions in the field of FR will be concluded in section 8 . 

 
 
 
 

## 7 Applications

 
 In this section, we will introduce applications by using face embeddings.
Most important applications will be face verification and identification.
We will not give more information about them in this section, because section 3.3 has presented the details.
Using the face features extracted from FR system, we can implement a lot of other applications, such as face clustering, attribute recognition and face generation, which will be elaborated in the following subsections. 

 
 

### 7.1 Face clustering

 
 Given a collection of unseen face images, face clustering groups images from the same person together.
This application can be adopted in many areas of industry, such as face clustering in photo album, characters summarizing of videos.
Face clustering usually uses face features from a well trained FR system as input.
Therefore high quality face embeddings have positive effect on face clustering.
In this section, we do not consider the generation of face embedding. Instead, we give the clustering procedure afterwards. 

 
 
 There are two types of face clustering methods.
The first type treats each face embedding as a point in a feature space, and utilizes unsupervised clustering algorithms.
Unsupervised methods usually achieve great clustering results in the condition of data distribution conforming to certain assumptions.
For instance, K-Means [ 191 ] requires the clusters to be convex-shaped,
Spectral Clustering [ 192 ] needs different clusters to be balanced in the number of instances,
and DBSCAN [ 193 ] assumes different clusters to be in the same density.
The second one adopts GCN (Graph Convolutional Network) to group features. GCN based cluster methods are supervised, thus they can generally outperform unsupervised clustering algorithms. 

 
 
 In this subsection, we introduce some recently published GCN based cluster methods. First, we set some annotations here.
Given a face dataset, features of all face images will be extracted by a trained CNN, forming a set of features
 X = [ x 1 , x 2 , … , x N ] T ∈ ℝ n × d X=[x_{1},x_{2},\dots,x_{N}]^{T}\in\mathbb{R}^{n\times d} , where n n is the number of images, and d d is the dimension of features.
Then each feature is regarded as a vertex and cosine similarity is used to find K K nearest neighbors for
each sample.
By connecting between neighbors, an affinity graph G = ( V , E ) G=(V,E) is obtained.
A symmetric adjacent matrix A ∈ ℝ n × n A\in\mathbb{R}^{n\times n} will be calculated, where the element a i , j a_{i,j} is the cosine similarity between x i x_{i} and x j x_{j} if two vertices are connected, or zero otherwise. 

 
 
 Yang et al. [ 194 ] proposed a GCN based clustering framework which consists of three modules,
namely proposal generator, GCN-D, and GCN-S.
The first module generates cluster proposals (i.e. sub-graphs likely to be clusters), from the affinity graph A A .
To do so, they remove edges with affinity values below a threshold, and constrain the size of sub-graphs below a maximum number.
GCN-D performs cluster detection. Taking a cluster proposal P P as input, it evaluates how likely the proposal
constitutes a desired cluster by two metrics, namely IoU and IoP scores,
which are defined as: 

 

 
 | 
 I ​ o ​ U ​ ( P ) = | P ∩ P ^ | | P ∪ P ^ | , I ​ o ​ P ​ ( P ) = | P ∩ P ^ | | P | IoU(P)=\frac{\lvert P\cap\hat{P}\lvert}{\lvert P\cup\hat{P}\lvert}\ \ ,\ \ IoP(P)=\frac{\lvert P\cap\hat{P}\lvert}{\lvert P\lvert} | 
 | 
 (119) | 
 

 where P ^ \hat{P} is the ground-truth set comprised all the vertices with label l ⁡ ( P ) l(P) , and l ⁡ ( P ) l(P) is the majority label of the cluster
P.
IoU reflects how close P is to the desired ground-truth P ^ \hat{P} , while IoP reflects the purity.
GCN-D is used to predict both the IoU and IoP scores, which consists of L L layers. The computation of each layer in GCN-D can be formulated as: 

 

 
 | 
 F l + 1 = σ ⁡ ( D − 1 ​ ( A + I ) ​ F l ​ W l ) F_{l+1}=\sigma(D^{-1}(A+I)F_{l}W_{l}) | 
 | 
 (120) | 
 

 where D = ∑ j A i , j D=\sum_{j}A_{i,j} is a diagonal degree matrix.
 F l F_{l} contains the embeddings of the l l -th layer.
 W l W_{l} is a learnable parameter matrix which transforms the embeddings.
 σ \sigma is the ReLU activation function.
While training, the objective is to minimize the mean square error(MSE) between ground-truth and predicted IoU and IoP scores.
Then GCN-S performs the segmentation to refine the selected proposals, which has similar structure with GCN-D. 

 
 
 Different from [ 194 ] , Wang et al. [ 195 ] adopted GCN to predict the similarity between two features.
In detail, Wang et al. first proposed the Instance Pivot Subgraphs (IPS). For each instance p p in the graph G G , an IPS is a subgraph centered at a pivot instance p p , which is comprised of nodes including the KNNs of p p and the high-order neighbors up to 2-hop of p p .
Each layer of the proposed GCN is formulated as: 

 

 
 | 
 F l + 1 = σ ( [ F l ∥ ( D − 1 2 A D − 1 2 ) F l ] W l ) F_{l+1}=\sigma([F_{l}\|(D^{-\frac{1}{2}}AD^{-\frac{1}{2}})F_{l}]W_{l}) | 
 | 
 (121) | 
 

 where operator ∥ \| represents matrix concatenation along the feature dimension.
The annotations in this equation is as the same as eq.( 120 ).
The input F 0 F_{0} of this GCN is not feature matrix of an IPS, instead,
 F 0 = [ … , x q − x p , … ] T , q ∈ V p F_{0}=[\dots,x_{q}-x_{p},\dots]^{T}\ ,\ q\in V_{p} ,
where V p V_{p} is the node set of IPS with pivot p p .
While training, the supervised ground-truth is a binary vector whose value in index q q is 1 if node q q shares a same label with pivot p p , and 0 if not.
In inference, this GCN predicts the likelihood of linkage between each node with the pivot.
To get clustering result, IPS with each instance as the pivot will be built, linkages in the whole face graph will be obtained.
At last, a pseudo label propagation strategy [ 196 ] will be adopted to cut the graph to get final cluster result. 

 
 
 Yang et al. [ 163 ] proposed a concept of confidence for each face vertex in a graph.
The confidence is the probability of a vertex belonging to a specific cluster.
For a face with high confidence, its neighboring faces tend to belong to the same class
while a face with low confidence is usually adjacent to the faces from the other classes.
As a result, the confidence c i c_{i} of a vertex i i can be calculated as: 

 

 
 | 
 c i = 1 | N i | ​ ∑ v j ∈ N i ( 1 y j = y i − 1 y j ≠ y i ) ​ a i , j c_{i}=\frac{1}{\lvert N_{i}\rvert}\sum_{v_{j}\in N_{i}}(1_{y_{j}=y_{i}}-1_{y_{j}\neq y_{i}})a_{i,j} | 
 | 
 (122) | 
 

 where N i N_{i} is the neighbour set of node i i according to KNN results.
So, they proposed GCN-V with L L layers to predict the confidence of each node. And computation of each layer can be formulated as: 

 

 
 | 
 F l + 1 = σ ( [ F l ∥ D − 1 ( A + I ) F l ] W l ) F_{l+1}=\sigma([F_{l}\|D^{-1}(A+I)F_{l}]W_{l}) | 
 | 
 (123) | 
 

 While training GCN-V, the objective is to minimize the mean square error (MSE) between ground truth and predicted confidence scores.
Then they built GCN-E to calculate connectivity of edges, which has similar structure with GCN-V.
The edge with high connectivity indicates the two connected samples tend to belong to the same class.
The input of GCN-E is a candidate set S S for each vertex:
 S i = { v j | c j c i , v j ∈ N i } S_{i}=\{v_{j}\lvert c_{j} c_{i},v_{j}\in N_{i}\} .
Set S i S_{i} only contains the vertices with higher confidence than the confidence c i c_{i} .
In training, ground-truth for input S i S_{i} is a binary vector, where for a vertex v i v_{i} , the GT connectivity (index j j ) is set to 1 if a neighbor v j v_{j} shares the same label with the v i v_{i} , otherwise it is 0.
MSE loss is also used to trained GCN-E. 

 
 
 The aforementioned GCN methods can be roughly divided into global-based (such as [ 163 ] ) and local-based ones (such as [ 195 , 194 ] ) according to whether their GCN inputs are the whole graph or not.
Global-based methods suffer from the limitation of training data scale, while local-based ones are difficult to grasp the whole graph structure information and usually take a long time for inference.
To address the dilemma of large-scale training and efficient inference, a STructure-AwaRe Face Clustering (STAR-FC) method [ 197 ] was proposed.
The proposed GCN consists of 2-layer of MLP, and each layer has similar structure with GCN-V in [ 163 ] (eq. ( 123 )).
However, this GCN takes pair features as input and predicts the two dimension edge confidence corresponding to these two nodes, which are connected in the affinity graph G G .
For inference, a single threshold τ 1 \tau_{1} is used to eliminate most of the wrong edges with smaller confidence.
After that, all subgraphs can be treated as clusters, which form a cluster set C C .
Then the concept of node intimacy ( N ​ I NI ) between two nodes v 1 v_{1} and v 2 v_{2} is defined as
 N ​ I = max ⁡ ( k n 1 , k n 2 ) NI=\max(\frac{k}{n_{1}},\frac{k}{n_{2}}) ,
where n 1 n_{1} and n 2 n_{2} are the numbers of edges connected to node v 1 v_{1} and v 2 v_{2} ; k k is the number of their common neighbor nodes.
Node intimacy will further purify the clusters.
In detail, a smaller set of clusters will be sampled from C C firstly, and all nodes in this set build a new subgraph S S .
Then, the NI values in S S will be calculated, and the edges with lower N ​ I NI value than τ 2 \tau_{2} will be cut.
Finally, the newly obtained clusters with less false positive edges become final clustering results. 

 
 
 Despite of GCN, transformer can also be used in face clustering.
 [ 198 ] abstracted face clustering problem as forming a face chain.
First, data density ρ \rho of a face node v ​ i vi is proposed as: 

 

 
 | 
 ρ ⁡ ( v i ) = ∑ v j ∈ N ⁡ ( v i ) x i ⋅ x j \rho(v_{i})=\sum_{v_{j}\in N(v_{i})}x_{i}\cdot x_{j} | 
 | 
 (124) | 
 

 where ⋅ \cdot is inner product, which calculates the cosine similarity of two normalized features. N ⁡ ( v i ) N(v_{i}) is as set of neighbours of v i v_{i} on affinity graph G G .
The node with high data density tend to have a high probability to be a certain person.
Given a node v k v_{k} , a node chain C ⁡ ( v k ) = { v k = c k 1 , c k 2 , … , c k N } C(v_{k})=\{v_{k}=c_{k}^{1},c_{k}^{2},\dots,c_{k}^{N}\} can be generated by gradually finding its nearest neighbors with higher density: 

 

 
 | 
 c k i + 1 = arg max v { x c k i ⋅ x v } , v ∈ { u | ρ ( u ) ρ ( c k i ) , u ∈ N ( c k i ) } c_{k}^{i+1}=\arg\max_{v}\{x_{c_{k}^{i}}\cdot x_{v}\}\ ,\ v\in\{u\lvert\rho(u) \rho(c_{k}^{i}),u\in N(c_{k}^{i})\} | 
 | 
 (125) | 
 

 Then, a transformer architecture is designed to further update the node feature.
For a node chain C ⁡ ( v k ) C(v_{k}) with N N nodes, this transformer predicts a weight set { w ( c k i ) | i = 1 , … , N } \{w(c_{k}^{i})\lvert i=1,\dots,N\} for each of its node.
The final density-aware embedding ψ ⁡ ( v k ) \psi(v_{k}) for node v k v_{k} is: 

 

 
 | 
 ψ ⁡ ( v k ) = ∑ i = 1 N w ⁡ ( c k i ) ⋅ x c k i \psi(v_{k})=\sum_{i=1}^{N}w(c_{k}^{i})\cdot x_{c_{k}^{i}} | 
 | 
 (126) | 
 

 At last, these feature is compatible with all kinds of clustering methods, e.g., merging nodes with high similarity, K-means, DBSCAN. 

 
 
 

### 7.2 Face attribute recognition

 
 Predicting face attributes is another widely used application for face embedding. By extracting features from face images, the network could estimate the age, gender, expression, hairstyle, and other attributes of this face. Mostly, the attributes prediction is performed based on the localization results, which have been summarized in Section 3.1 . 

 
 
 For prediction, multi-task learning was widely utilized to recognize a cluster of attributes at the same time. Liu et al. [ 168 ] proposed the ANet model to extract face features and used multiple support vector machine (SVM) classifiers to predict 40 face attributes. The ANet was pre-trained by the identity recognition task, and then fine-tuned by attributes tags. In the fine-tuned stage, multiple patches of the face region were generated from each face image, and a fast feature extraction method named interweaved operation was proposed to analyze these patches. The outputs of ANet were feature vectors and were utilized to train the SVM classifiers. 

 
 
 ANet utilized the same features to predict all the attributes, however, attributes heterogeneity had not been considered. To address this limitation, some works adjusted the network structure and allowed its last few layers to be shared among a specific category of attributes through a multi-branch structure [ 199 , 200 , 201 , 202 ] . The attributes were grouped following different grouping strategies. Han et al. [ 199 ] proposed a joint estimation model. Data type, data scale, and semantic meaning were utilized to build the grouping strategy; based on that, four types of attributes were defined, i.e., holistic-nominal, holistic-ordinal, local-nominal, and local-ordinal. The loss function is formulated as: 

 

 
 | 
 arg ⁡ min W c , { W j } j = 1 M ​ ∑ g = 1 G ∑ j = 1 M g ∑ i = 1 N λ g ​ ℒ g ​ ( y i j , ℱ ⁡ ( X i , W g ∘ W c ) ) + γ 1 ​ ϕ ​ ( W c ) + γ 2 ​ ϕ ​ ( W g ) \arg\min_{W_{c},\{W^{j}\}^{M}_{j=1}}\sum_{g=1}^{G}\sum_{j=1}^{M^{g}}\sum_{i=1}^{N}\lambda^{g}\mathcal{L}^{g}(y_{i}^{j},\mathcal{F}(X_{i},W^{g}\circ W_{c}))+\gamma_{1}\phi(W_{c})+\gamma_{2}\phi(W_{g}) | 
 | 
 (127) | 
 

 where G and M g M^{g} represent the number of heterogeneous attribute categories and attributes within each attribute category, separately; λ g \lambda^{g} is an adjustment factor to adjust the weight of each category; ℱ ( . , . ) \mathcal{F}(.,.) denotes the predicted result based on weight vectors W c W_{c} and W g W^{g} ; W c W_{c} and W g W^{g} control shared features among all the face attributes and shared features among the attributes within each category, separately; y i j y_{i}^{j} is the ground truth; X i X_{i} is the input; ℒ \mathcal{L} is the loss function; ϕ \phi is regularization function, and γ 1 \gamma_{1} and γ 2 \gamma_{2} are regularization parameters. 

 
 
 For the multi-branch structure, each branch is trained separately and can not be affected by other branches. However, some researchers thought that the task relation was conducive to the attributes prediction, thus, they added connections between the branches. Cao et al. [ 203 ] proposed a Partially Shared Multi-task Convolutional Neural Network (PS-MCNN) which consisted of the Shared Network (SNet) and the Task Specific Network (TSNet) (Fig. 42 ). SNet extracted the task relation and shared informative representations. TSNet learned the specific information corresponding to each task. The number of TSNet was consistent with the number of face attribute groups. Face attributes were split into four categories based on their locations, i.e. upper, middle, lower, and whole group. 

 
 
 Figure 42: The pipeline of PS-MCNN. [ 203 ] . 
 
 
 Based on the PS-MCNN, Cao et al. further developed a model named Partially Shared Network with Local Constraint (PS-MCNN-LC) to utilize identity information [ 203 ] because a high degree of similarity existed among the face attribute tags from the same identity. A novel loss function named LCLoss was proposed to add this constraint into the network training process and is formulated as: 

 

 
 | 
 L ​ C ​ L ​ o ​ s ​ s = 1 N ⁡ ( N − 1 ) ​ ∑ i = 1 N ∑ j = i + 1 N w i , j ​ ‖ f ​ e ​ a ​ t s ​ i t − f ​ e ​ a ​ t s ​ j t ‖ 2 2 LCLoss=\frac{1}{N(N-1)}\sum_{i=1}^{N}\sum_{j=i+1}^{N}w_{i,j}\|feat^{t}_{si}-feat^{t}_{sj}\|^{2}_{2} | 
 | 
 (128) | 
 

 where w i , j = 1 w_{i,j}=1 if sample i i and j j have the same identity, otherwise w i , j = 0 w_{i,j}=0 ; f ​ e ​ a ​ t s t feat^{t}_{s} means features that are extracted from the t t ​ h t^{th} layer of the SNet. 

 
 
 Besides multi-task learning, previous works also focused on improving model performance on some hard attribute prediction tasks using single-task learning. Age estimation [ 204 , 205 , 206 , 207 ] and expression recognition [ 208 , 209 , 210 , 211 ] are representative. For age estimation, traditionally, it was treated as an over-simplified linear regression task or a multi-class classification task, which ignored the ordinal information, the semantic information, and the nonlinear aging pattern. To overcome it, Chen et al. [ 204 ] proposed a ranking-CNN and treated it as multiple binary classification problems. Each binary classifier predicted whether the age of the input face was greater than a certain value k k , where k ∈ { 1 , … , K } k\in\{1,...,K\} and the value of K K depends on the ordinal labels. Then a set of binary classification results is aggregated by the following formula: 

 

 
 | 
 r ( x i ) = 1 + ∑ k = 1 K − 1 [ f k ( x i ) 0 ] r(x_{i})=1+\sum_{k=1}^{K-1}[f_{k}(x_{i}) 0] | 
 | 
 (129) | 
 

 where f k ​ ( x i ) f_{k}(x_{i}) is the binary classification result, if f k f k , f = 1 f=1 , otherwise, f = − 1 f=-1 ; [ . ] [.] is an operator, if the inner condition is true, it is 1, otherwise, it is 0. 

 
 
 Zhang et al. [ 206 ] proposed an efficient and effective age estimation model named C3AE (Fig. 43 ). Instead of using a large and deep network, C3AE utilized only five convolution layers and two dense layers. The inputs of the model were three scales of cropped face images. Their results were concatenated at the last convolution layer and analyzed by the first dense layer, which output an age distribution utilizing a novel age encoding method named the two-points representation. The second dense layer output the final prediction of age. Cascade training was utilized. 

 
 
 Figure 43: The pipeline of the C3AE model which was developed for age estimation [ 206 ] . 
 
 
 For expression recognition, due to the huge variance within the emotional classes caused by different demographic characteristics, previous work focused on utilizing embedding methods to extract the expression features from images and made predictions based on them. Zhang et al. [ 209 ] proposed the Deviation Learning Network (DLN) to explicitly remove identity attributes from input face. An identity model and a face model were contained in the DLN and both were pre-trained Inception-Reasnet FaceNet [ 34 ] models. The difference was that the parameters of the identity model were fixed, but those of the face model were trainable during the training process. The outputs of the face model ( V f ​ a ​ c ​ e V_{face} ) and the identity model ( V i ​ d V_{id} ) were 512-dimensional vectors. The expression vector was given by ( V f ​ a ​ c ​ e − V i ​ d V_{face}-V_{id} ), and then converted to a 16-dimensional feature space through a proposed high-order module. The final prediction was made based on the 16-dimensional features by a crowd layer [ 212 ] , which was used to eliminate the annotation bias. 

 
 
 

### 7.3 Face generation

 
 The last application of FR we introduce here is face generation, especially for the ones with ID preserving.
We divide face generation methods into three types: GAN, 3D, and residual learning based face generation.
As we introduced before, algorithms [ 78 , 79 , 80 , 81 ] in subsection 4.2.1 have merged face generation and recognition together.
A lot of applications, such as face editing (age/expression changing, glasses/bread removing), face swapping use GAN to generate synthetic face images.
3D based methods usually generate faces with different angle, namely face frontalization and rotation.
Residual learning based methods usually focus on generating faces without much content changing, such as face denoising, deblurring, super resolution.
However, many face denoising and super resolution problems adopt GAN to model the problem. 

 
 
 
 

## 8 Competitions and Open Source Programs

 
 The first FR competition introduced in this section is Face Recognition Vendor Test (FRVT) [ 213 ] .
FRVT is regularly held by the National Institute of Standards and Technology (NIST) to evaluate FR algorithms of state-of-the-art.
It is the most authoritative and largest FR testing competition recently.
Nearly 100 companies and research institutions have participated in this test to date.
The FRVT does not restrict the face training set.
After participants provide the algorithm SDK, FRVT tests the performance of these algorithms directly.
FRVT has strict restrictions on the submitted algorithms.
In specific, all submissions can only use no more than 1 second of computational resources in a single CPU thread to handle a whole FR pipeline of a single image, from face detection and alignment to feature extraction and recognition.
FRVT is divided into four tracks, which are FRVT 1:1, FRVT 1:N, FRVT MORPH and FRVT Quality. 

 
 
 FRVT 1:1 evaluates algorithms with the metric of FNMR at FMR.
FNMR is the proportion of mated comparisons below a threshold set to achieve the false match rate (FMR) specified.
FMR is the proportion of impostor comparisons at or above that threshold.
FRVT 1:1 tests the algorithm on multiple datasets (scenes) with and without constraint environment respectively.
The former contains visa photos, mugshot photos, mugshot photos 12+years, visaborder photos and border photos; the latter contains child photos and child exp photos. 

 
 
 FRVT 1:N mainly tests the identification performance and investigation performance of FR algorithms. The evaluation metrics are FNIR at FPIR, and matching accuracy.
FNIR is the proportion of mated searches failing to return the mate above threshold.
FPIR is the proportion of non-mated searches producing one or more candidates above threshold.
Matching accuracy evaluates whether the probe image matches rank1’s with a threshold of 0. 

 
 
 FRVT MORPH measures the performance of face forgery, whose evaluation metric is the APCER corresponding to BPCER at 0.1 and 0.01.
APCER, or morph miss rate, is the proportion of morphs that are incorrectly classified as bona fides (nonmorphs).
BPCER, or false detection rate, is the proportion of bona fides falsely classified as morphs. FRVT MORPH is divided into three tier, which are low quality morphs, automated morphs and high quality morphs. 

 
 
 FRVT Quality evaluates face quality assessment algorithms (QAAs).
In face identification, the quality of face images in gallery is crucial to identification performance.
As a result, the measurement of face identification is used in FRVT Quality metrics.
In detail, given a gallery set with high and low quality face images, FNMR of a FR system will be calculated first (FNMR-1).
Then the part of faces in gallery with lowest quality are discarded, and FNMR is calculated again (FNMR-2).
A smaller value of FNMR-2 indicates a better quality model performance.
Theoretically, when FNMR-1 = 0.01, after discarding the lowest quality 1% of the images, the FNMR-2 will become 0%.
To find the images with lowest quality in gallery, this track contains two metrics: a quality scalar and a quality vector.
The quality scalar directly evaluates the quality of an input image by a scalar score.
The quality vector scores multiple attributes of the input face image, such as focus, lighting, pose, sharpness, etc.
This quality vector result can provide more precise feedback to contestants for a specific attribute which may affect image quality. 

 
 
 Besides FRVT from NIST, there are some famous FR competitions, such as MegaFace challenge [ 176 ] , and MS-Celeb-1M challenge [ 171 ] .
These two competitions are no longer updating nowadays, since their goals have been met with high evaluation performance.
MegaFace competition has two challenges.
In challenge 1, contestants can use any face images to train the model. In evaluation, face verification and verification task are performed under up to 1 million distractors.
Performance is measured using probe and gallery images from FaceScrub and FGNet.
In challenge 2, contestants need to train on a provided set with 672K identities, and then test recognition and verification performance under 1 million distractors.
Probe and gallery images are used from FaceScrub and FGNet.
As we mention before, FaceScrub dataset is used to test FR on celebrity photos, and FGNet is to test age invariance FR.
MS1M challenge was proposed in 2016, based on real world large scale dataset on celebrities, and open evaluation system.
This challenge has provided the training datasets to recognize 1M celebrities from their face images.
The 1M celebrities are obtained from Freebase based on their occurrence frequencies (popularities) on the web, thus this dataset contains heavy noise and needs cleaning.
In evaluation, the measurement set consists of 1000 celebrities sampled from the 1M celebrities (which is not disclosed). For each celebrity, up to 20 images are manually labeled for evaluation.
To obtain high recognition recall and precision rates, the contestants should develop a recognizer to cover as many as possible celebrities. 

 
 
 

## 9 Conclusion

 
 In this paper, we introduce about 100 algorithms in face recognition (FR), including every sides of FR, such as its history, pipeline, algorithms, training and evaluation datasets and related applications. 

 
 
 

## References

 
 
 [1] 
 
M. Turk and A. Pentland, “Eigenfaces for recognition,” Journal of
cognitive neuroscience , vol. 3, no. 1, pp. 71–86, 1991.
 
 

 
 [2] 
 
P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman, “Eigenfaces vs.
fisherfaces: Recognition using class specific linear projection,” IEEE
Transactions on pattern analysis and machine intelligence , vol. 19, no. 7,
pp. 711–720, 1997.
 
 

 
 [3] 
 
O. Déniz, M. Castrillon, and M. Hernández, “Face recognition using
independent component analysis and support vector machines,” Pattern
recognition letters , vol. 24, no. 13, pp. 2153–2157, 2003.
 
 

 
 [4] 
 
E. Oja and A. Hyvarinen, “Independent component analysis: algorithms and
applications,” Neural networks , vol. 13, no. 4-5, pp. 411–430, 2000.
 
 

 
 [5] 
 
R. Kong and B. Zhang, “A new face recognition method based on fast least
squares support vector machine,” Physics Procedia , vol. 22,
pp. 616–621, 2011.
 
 

 
 [6] 
 
X. Jianhong, “Kpca based on ls-svm for face recognition,” in 2008 Second
International Symposium on Intelligent Information Technology Application ,
vol. 2, pp. 638–641, IEEE, 2008.
 
 

 
 [7] 
 
T. Ojala, M. Pietikäinen, and D. Harwood, “A comparative study of texture
measures with classification based on featured distributions,” Pattern
recognition , vol. 29, no. 1, pp. 51–59, 1996.
 
 

 
 [8] 
 
T. Ahonen, A. Hadid, and M. Pietikäinen, “Face recognition with local
binary patterns,” in European conference on computer vision ,
pp. 469–481, Springer, 2004.
 
 

 
 [9] 
 
L. Wolf, T. Hassner, and Y. Taigman, “Descriptor based methods in the wild,”
in Workshop on faces in’real-life’images: Detection, alignment, and
recognition , 2008.
 
 

 
 [10] 
 
X. Tan and B. Triggs, “Enhanced local texture feature sets for face
recognition under difficult lighting conditions,” IEEE transactions on
image processing , vol. 19, no. 6, pp. 1635–1650, 2010.
 
 

 
 [11] 
 
S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time
object detection with region proposal networks,” Advances in neural
information processing systems , vol. 28, pp. 91–99, 2015.
 
 

 
 [12] 
 
W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg,
“Ssd: Single shot multibox detector,” in European conference on
computer vision , pp. 21–37, Springer, 2016.
 
 

 
 [13] 
 
J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once:
Unified, real-time object detection,” in Proceedings of the IEEE
conference on computer vision and pattern recognition , pp. 779–788, 2016.
 
 

 
 [14] 
 
J. Redmon and A. Farhadi, “Yolo9000: better, faster, stronger,” in Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 7263–7271, 2017.
 
 

 
 [15] 
 
J. Redmon and A. Farhadi, “Yolov3: An incremental improvement,” arXiv
preprint arXiv:1804.02767 , 2018.
 
 

 
 [16] 
 
K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment
using multitask cascaded convolutional networks,” IEEE Signal
Processing Letters , vol. 23, no. 10, pp. 1499–1503, 2016.
 
 

 
 [17] 
 
S. Yang, P. Luo, C. C. Loy, and X. Tang, “Faceness-net: Face detection through
deep facial part responses,” IEEE transactions on pattern analysis and
machine intelligence , vol. 40, no. 8, pp. 1845–1859, 2017.
 
 

 
 [18] 
 
C. Chi, S. Zhang, J. Xing, Z. Lei, S. Z. Li, and X. Zou, “Selective refinement
network for high performance face detection,” in Proceedings of the
AAAI conference on artificial intelligence , vol. 33, pp. 8231–8238, 2019.
 
 

 
 [19] 
 
J. Deng, J. Guo, Y. Zhou, J. Yu, I. Kotsia, and S. Zafeiriou, “Retinaface:
Single-stage dense face localisation in the wild,” arXiv preprint
arXiv:1905.00641 , 2019.
 
 

 
 [20] 
 
N. Vesdapunt and B. Wang, “Crface: Confidence ranker for model-agnostic face
detection refinement,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 1674–1684, 2021.
 
 

 
 [21] 
 
W. Wang, W. Yang, and J. Liu, “Hla-face: Joint high-low adaptation for low
light face detection,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 16195–16204, 2021.
 
 

 
 [22] 
 
Z. Boulkenafet, J. Komulainen, and A. Hadid, “Face anti-spoofing based on
color texture analysis,” in 2015 IEEE international conference on image
processing (ICIP) , pp. 2636–2640, IEEE, 2015.
 
 

 
 [23] 
 
K. Patel, H. Han, and A. K. Jain, “Secure face unlock: Spoof detection on
smartphones,” IEEE transactions on information forensics and security ,
vol. 11, no. 10, pp. 2268–2283, 2016.
 
 

 
 [24] 
 
Z. Boulkenafet, J. Komulainen, and A. Hadid, “Face antispoofing using
speeded-up robust features and fisher vector encoding,” IEEE Signal
Processing Letters , vol. 24, no. 2, pp. 141–145, 2016.
 
 

 
 [25] 
 
J. Komulainen, A. Hadid, and M. Pietikäinen, “Context based face
anti-spoofing,” in 2013 IEEE Sixth International Conference on
Biometrics: Theory, Applications and Systems (BTAS) , pp. 1–8, IEEE, 2013.
 
 

 
 [26] 
 
Z. Yu, C. Zhao, Z. Wang, Y. Qin, Z. Su, X. Li, F. Zhou, and G. Zhao,
“Searching central difference convolutional networks for face
anti-spoofing,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 5295–5305, 2020.
 
 

 
 [27] 
 
Z. Yu, Y. Qin, H. Zhao, X. Li, and G. Zhao, “Dual-cross central difference
network for face anti-spoofing,” arXiv preprint arXiv:2105.01290 ,
2021.
 
 

 
 [28] 
 
Y. Liu, J. Stehouwer, and X. Liu, “On disentangling spoof trace for generic
face anti-spoofing,” in European Conference on Computer Vision ,
pp. 406–422, Springer, 2020.
 
 

 
 [29] 
 
C.-Y. Wang, Y.-D. Lu, S.-T. Yang, and S.-H. Lai, “Patchnet: A simple face
anti-spoofing framework via fine-grained patch recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 20281–20290, 2022.
 
 

 
 [30] 
 
Z. Wang, Z. Wang, Z. Yu, W. Deng, J. Li, T. Gao, and Z. Wang, “Domain
generalization via shuffled style assembly for face anti-spoofing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 4123–4133, 2022.
 
 

 
 [31] 
 
T. Shen, Y. Huang, and Z. Tong, “Facebagnet: Bag-of-local-features model for
multi-modal face anti-spoofing,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition Workshops , pp. 0–0,
2019.
 
 

 
 [32] 
 
A. George and S. Marcel, “Cross modal focal loss for rgbd face
anti-spoofing,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 7882–7891, 2021.
 
 

 
 [33] 
 
O. M. Parkhi, A. Vedaldi, and A. Zisserman, “Deep face recognition,” 2015.
 
 

 
 [34] 
 
F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for
face recognition and clustering,” in Proceedings of the IEEE conference
on computer vision and pattern recognition , pp. 815–823, 2015.
 
 

 
 [35] 
 
Y. Taigman, M. Yang, M. Ranzato, and L. Wolf, “Deepface: Closing the gap to
human-level performance in face verification,” in Proceedings of the
IEEE conference on computer vision and pattern recognition , pp. 1701–1708,
2014.
 
 

 
 [36] 
 
M. Jaderberg, K. Simonyan, A. Zisserman, et al. , “Spatial transformer
networks,” Advances in neural information processing systems , vol. 28,
pp. 2017–2025, 2015.
 
 

 
 [37] 
 
W. Wu, M. Kan, X. Liu, Y. Yang, S. Shan, and X. Chen, “Recursive spatial
transformer (rest) for alignment-free face recognition,” in Proceedings
of the IEEE International Conference on Computer Vision , pp. 3772–3780,
2017.
 
 

 
 [38] 
 
Z. An, W. Deng, Y. Zhong, Y. Huang, and X. Tao, “Apa: Adaptive pose alignment
for robust face recognition,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition Workshops , pp. 0–0, 2019.
 
 

 
 [39] 
 
H. Phan and A. Nguyen, “Deepface-emd: Re-ranking using patch-wise earth
mover’s distance improves out-of-distribution face identification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 20259–20269, 2022.
 
 

 
 [40] 
 
C. Han, S. Shan, M. Kan, S. Wu, and X. Chen, “Face recognition with
contrastive convolution,” in Proceedings of the European Conference on
Computer Vision (ECCV) , pp. 118–134, 2018.
 
 

 
 [41] 
 
Y. Sun, Deep learning face representation by joint
identification-verification .
 
 
 The Chinese University of Hong Kong (Hong Kong), 2015.
 
 

 
 [42] 
 
A. Ali, M. Testa, T. Bianchi, and E. Magli, “Biometricnet: deep unconstrained
face verification through learning of metrics regularized onto gaussian
distributions,” in European Conference on Computer Vision ,
pp. 133–149, Springer, 2020.
 
 

 
 [43] 
 
K. Q. Weinberger and L. K. Saul, “Distance metric learning for large margin
nearest neighbor classification.,” Journal of machine learning
research , vol. 10, no. 2, 2009.
 
 

 
 [44] 
 
B.-N. Kang, Y. Kim, and D. Kim, “Pairwise relational networks for face
recognition,” in Proceedings of the European Conference on Computer
Vision (ECCV) , pp. 628–645, 2018.
 
 

 
 [45] 
 
Y. Wen, K. Zhang, Z. Li, and Y. Qiao, “A discriminative feature learning
approach for deep face recognition,” in European conference on computer
vision , pp. 499–515, Springer, 2016.
 
 

 
 [46] 
 
W. Liu, Y. Wen, Z. Yu, and M. Yang, “Large-margin softmax loss for
convolutional neural networks.,” in ICML , vol. 2, p. 7, 2016.
 
 

 
 [47] 
 
W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, and L. Song, “Sphereface: Deep
hypersphere embedding for face recognition,” in Proceedings of the IEEE
conference on computer vision and pattern recognition , pp. 212–220, 2017.
 
 

 
 [48] 
 
F. Wang, X. Xiang, J. Cheng, and A. L. Yuille, “Normface: L2 hypersphere
embedding for face verification,” in Proceedings of the 25th ACM
international conference on Multimedia , pp. 1041–1049, 2017.
 
 

 
 [49] 
 
F. Wang, J. Cheng, W. Liu, and H. Liu, “Additive margin softmax for face
verification,” IEEE Signal Processing Letters , vol. 25, no. 7,
pp. 926–930, 2018.
 
 

 
 [50] 
 
H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu,
“Cosface: Large margin cosine loss for deep face recognition,” in Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 5265–5274, 2018.
 
 

 
 [51] 
 
J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin
loss for deep face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 4690–4699, 2019.
 
 

 
 [52] 
 
X. Zhang, R. Zhao, J. Yan, M. Gao, Y. Qiao, X. Wang, and H. Li, “P2sgrad:
Refined gradients for optimizing deep face models,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 9906–9914, 2019.
 
 

 
 [53] 
 
X. Wang, S. Wang, S. Zhang, T. Fu, H. Shi, and T. Mei, “Support vector guided
softmax loss for face recognition,” arXiv preprint arXiv:1812.11317 ,
2018.
 
 

 
 [54] 
 
X. Wang, S. Zhang, S. Wang, T. Fu, H. Shi, and T. Mei, “Mis-classified vector
guided softmax loss for face recognition,” in Proceedings of the AAAI
Conference on Artificial Intelligence , vol. 34, pp. 12241–12248, 2020.
 
 

 
 [55] 
 
Y. Zheng, D. K. Pal, and M. Savvides, “Ring loss: Convex feature normalization
for face recognition,” in Proceedings of the IEEE conference on
computer vision and pattern recognition , pp. 5089–5097, 2018.
 
 

 
 [56] 
 
W. Hu, Y. Huang, F. Zhang, and R. Li, “Noise-tolerant paradigm for training
face recognition cnns,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 11887–11896, 2019.
 
 

 
 [57] 
 
J. Deng, J. Guo, T. Liu, M. Gong, and S. Zafeiriou, “Sub-center arcface:
Boosting face recognition by large-scale noisy web faces,” in European
Conference on Computer Vision , pp. 741–757, Springer, 2020.
 
 

 
 [58] 
 
Y. Huang, Y. Wang, Y. Tai, X. Liu, P. Shen, S. Li, J. Li, and F. Huang,
“Curricularface: adaptive curriculum learning loss for deep face
recognition,” in proceedings of the IEEE/CVF conference on computer
vision and pattern recognition , pp. 5901–5910, 2020.
 
 

 
 [59] 
 
D. Zeng, H. Shi, H. Du, J. Wang, Z. Lei, and T. Mei, “Npcface: A
negative-positive cooperation supervision for training large-scale face
recognition,” arXiv preprint arXiv:2007.10172 , 2020.
 
 

 
 [60] 
 
Y. Duan, J. Lu, and J. Zhou, “Uniformface: Learning deep equidistributed
representation for face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 3415–3424, 2019.
 
 

 
 [61] 
 
K. Zhao, J. Xu, and M.-M. Cheng, “Regularface: Deep face recognition via
exclusive regularization,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) , June 2019.
 
 

 
 [62] 
 
J. Deng, J. Guo, J. Yang, A. Lattas, and S. Zafeiriou, “Variational prototype
learning for deep face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 11906–11915,
2021.
 
 

 
 [63] 
 
H. Yu, Y. Fan, K. Chen, H. Yan, X. Lu, J. Liu, and D. Xie, “Unknown identity
rejection loss: Utilizing unlabeled data for face recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision
Workshops , pp. 0–0, 2019.
 
 

 
 [64] 
 
X. Zhang, R. Zhao, Y. Qiao, X. Wang, and H. Li, “Adacos: Adaptively scaling
cosine logits for effectively learning deep face representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 10823–10832, 2019.
 
 

 
 [65] 
 
B. Liu, W. Deng, Y. Zhong, M. Wang, J. Hu, X. Tao, and Y. Huang, “Fair loss:
Margin-aware reinforcement learning for deep face recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision ,
pp. 10052–10061, 2019.
 
 

 
 [66] 
 
H. Liu, X. Zhu, Z. Lei, and S. Z. Li, “Adaptiveface: Adaptive margin and
sampling for face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 11947–11956,
2019.
 
 

 
 [67] 
 
Y. Huang, P. Shen, Y. Tai, S. Li, X. Liu, J. Li, F. Huang, and R. Ji,
“Improving face recognition from hard samples via distribution distillation
loss,” in European Conference on Computer Vision , pp. 138–154,
Springer, 2020.
 
 

 
 [68] 
 
E. Ustinova and V. Lempitsky, “Learning deep embeddings with histogram loss,”
 arXiv preprint arXiv:1611.00822 , 2016.
 
 

 
 [69] 
 
Q. Meng, S. Zhao, Z. Huang, and F. Zhou, “Magface: A universal representation
for face recognition and quality assessment,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 14225–14234, 2021.
 
 

 
 [70] 
 
Y. Sun, C. Cheng, Y. Zhang, C. Zhang, L. Zheng, Z. Wang, and Y. Wei, “Circle
loss: A unified perspective of pair similarity optimization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 6398–6407, 2020.
 
 

 
 [71] 
 
M. Kim, A. K. Jain, and X. Liu, “Adaface: Quality adaptive margin for face
recognition,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 18750–18759, 2022.
 
 

 
 [72] 
 
Y. Wu, Y. Wu, R. Gong, Y. Lv, K. Chen, D. Liang, X. Hu, X. Liu, and J. Yan,
“Rotation consistent margin loss for efficient low-bit face recognition,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 6866–6876, 2020.
 
 

 
 [73] 
 
X. Zhang, Z. Fang, Y. Wen, Z. Li, and Y. Qiao, “Range loss for deep face
recognition with long-tailed training data,” in Proceedings of the IEEE
International Conference on Computer Vision , pp. 5409–5418, 2017.
 
 

 
 [74] 
 
Y. Zhong, W. Deng, M. Wang, J. Hu, J. Peng, X. Tao, and Y. Huang,
“Unequal-training for deep face recognition with long-tailed noisy data,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 7812–7821, 2019.
 
 

 
 [75] 
 
H. Du, H. Shi, Y. Liu, J. Wang, Z. Lei, D. Zeng, and T. Mei, “Semi-siamese
training for shallow face learning,” in European Conference on Computer
Vision , pp. 36–53, Springer, 2020.
 
 

 
 [76] 
 
W. Li, T. Guo, P. Li, B. Chen, B. Wang, W. Zuo, and L. Zhang, “Virface:
Enhancing face recognition via unlabeled shallow data,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 14729–14738, 2021.
 
 

 
 [77] 
 
C. Liu, X. Yu, Y.-H. Tsai, M. Faraki, R. Moslemi, M. Chandraker, and Y. Fu,
“Learning to learn across diverse data biases in deep face recognition,” in
 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 4072–4082, 2022.
 
 

 
 [78] 
 
L. Tran, X. Yin, and X. Liu, “Disentangled representation learning gan for
pose-invariant face recognition,” in Proceedings of the IEEE conference
on computer vision and pattern recognition , pp. 1415–1424, 2017.
 
 

 
 [79] 
 
Y. Liu, F. Wei, J. Shao, L. Sheng, J. Yan, and X. Wang, “Exploring
disentangled feature representation beyond face identification,” in Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition , pp. 2080–2089, 2018.
 
 

 
 [80] 
 
K. Chen, Y. Wu, H. Qin, D. Liang, X. Liu, and J. Yan, “R3 adversarial network
for cross model face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 9868–9876, 2019.
 
 

 
 [81] 
 
Z. Huang, J. Zhang, and H. Shan, “When age-invariant face recognition meets
face age synthesis: A multi-task learning framework,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 7282–7291, 2021.
 
 

 
 [82] 
 
Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette,
M. Marchand, and V. Lempitsky, “Domain-adversarial training of neural
networks,” The journal of machine learning research , vol. 17, no. 1,
pp. 2096–2030, 2016.
 
 

 
 [83] 
 
H. Uppal, A. Sepas-Moghaddam, M. Greenspan, and A. Etemad, “Teacher-student
adversarial depth hallucination to improve face recognition,” arXiv
preprint arXiv:2104.02424 , 2021.
 
 

 
 [84] 
 
J.-R. Chang, Y.-S. Chen, and W.-C. Chiu, “Learning facial representations from
the cycle-consistency of face,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision , pp. 9680–9689, 2021.
 
 

 
 [85] 
 
M. Iliadis, H. Wang, R. Molina, and A. K. Katsaggelos, “Robust and low-rank
representation for fast face identification with occlusions,” IEEE
Transactions on Image Processing , vol. 26, no. 5, pp. 2203–2218, 2017.
 
 

 
 [86] 
 
J. Dong, H. Zheng, and L. Lian, “Low-rank laplacian-uniform mixed model for
robust face recognition,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 11897–11906, 2019.
 
 

 
 [87] 
 
J. Yang, P. Ren, D. Zhang, D. Chen, F. Wen, H. Li, and G. Hua, “Neural
aggregation network for video face recognition,” in Proceedings of the
IEEE conference on computer vision and pattern recognition , pp. 4362–4371,
2017.
 
 

 
 [88] 
 
L. He, H. Li, Q. Zhang, and Z. Sun, “Dynamic feature learning for partial face
recognition,” in Proceedings of the IEEE conference on computer vision
and pattern recognition , pp. 7054–7063, 2018.
 
 

 
 [89] 
 
J. Zhao, Y. Cheng, Y. Xu, L. Xiong, J. Li, F. Zhao, K. Jayashree, S. Pranata,
S. Shen, J. Xing, et al. , “Towards pose invariant face recognition in
the wild,” in Proceedings of the IEEE conference on computer vision and
pattern recognition , pp. 2207–2216, 2018.
 
 

 
 [90] 
 
B. Yin, L. Tran, H. Li, X. Shen, and X. Liu, “Towards interpretable face
recognition,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision , pp. 9348–9357, 2019.
 
 

 
 [91] 
 
H. Wang, D. Gong, Z. Li, and W. Liu, “Decorrelated adversarial learning for
age-invariant face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 3527–3536, 2019.
 
 

 
 [92] 
 
X. Yin, X. Yu, K. Sohn, X. Liu, and M. Chandraker, “Feature transfer learning
for face recognition with under-represented data,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 5704–5713, 2019.
 
 

 
 [93] 
 
Y. Shi and A. K. Jain, “Probabilistic face embeddings,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision , pp. 6902–6911,
2019.
 
 

 
 [94] 
 
J. Chang, Z. Lan, C. Cheng, and Y. Wei, “Data uncertainty learning in face
recognition,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 5710–5719, 2020.
 
 

 
 [95] 
 
S. Li, J. Xu, X. Xu, P. Shen, S. Li, and B. Hooi, “Spherical confidence
learning for face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 15629–15637,
2021.
 
 

 
 [96] 
 
Y. Shi, X. Yu, K. Sohn, M. Chandraker, and A. K. Jain, “Towards universal
representation learning for deep face recognition,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 6817–6826, 2020.
 
 

 
 [97] 
 
Q. Wang, T. Wu, H. Zheng, and G. Guo, “Hierarchical pyramid diverse attention
networks for face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 8326–8335, 2020.
 
 

 
 [98] 
 
Q. Wang and G. Guo, “Ls-cnn: Characterizing local patches at multiple scales
for face recognition,” IEEE Transactions on Information Forensics and
Security , vol. 15, pp. 1640–1653, 2019.
 
 

 
 [99] 
 
D. Cao, X. Zhu, X. Huang, J. Guo, and Z. Lei, “Domain balancing: Face
recognition on long-tailed domains,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 5671–5679, 2020.
 
 

 
 [100] 
 
Y. Kim, W. Park, M.-C. Roh, and J. Shin, “Groupface: Learning latent groups
and constructing group-based representations for face recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 5621–5630, 2020.
 
 

 
 [101] 
 
S. Gong, X. Liu, and A. K. Jain, “Mitigating face recognition bias via group
adaptive classifier,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 3414–3424, 2021.
 
 

 
 [102] 
 
X. Hou, Y. Li, and S. Wang, “Disentangled representation for age-invariant
face recognition: A mutual information minimization perspective,” in Proceedings of the IEEE/CVF International Conference on Computer Vision ,
pp. 3692–3701, 2021.
 
 

 
 [103] 
 
X. Peng, X. Yu, K. Sohn, D. N. Metaxas, and M. Chandraker,
“Reconstruction-based disentanglement for pose-invariant face recognition,”
in Proceedings of the IEEE international conference on computer vision ,
pp. 1623–1632, 2017.
 
 

 
 [104] 
 
Y. Wang, D. Gong, Z. Zhou, X. Ji, H. Wang, Z. Li, W. Liu, and T. Zhang,
“Orthogonal deep features decomposition for age-invariant face
recognition,” in Proceedings of the European conference on computer
vision (ECCV) , pp. 738–753, 2018.
 
 

 
 [105] 
 
F. Liu, R. Zhu, D. Zeng, Q. Zhao, and X. Liu, “Disentangling features in 3d
face shapes for joint face reconstruction and recognition,” in Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 5216–5225, 2018.
 
 

 
 [106] 
 
W. Wang, Y. Fu, X. Qian, Y.-G. Jiang, Q. Tian, and X. Xue, “Fm2u-net: Face
morphological multi-branch network for makeup-invariant face verification,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 5730–5740, 2020.
 
 

 
 [107] 
 
J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image
translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision ,
pp. 2223–2232, 2017.
 
 

 
 [108] 
 
S. Gong, X. Liu, and A. K. Jain, “Jointly de-biasing face recognition and
demographic attribute estimation,” in European Conference on Computer
Vision , pp. 330–347, Springer, 2020.
 
 

 
 [109] 
 
P. Dhar, J. Gleason, A. Roy, C. D. Castillo, and R. Chellappa, “Pass:
Protected attribute suppression system for mitigating bias in face
recognition,” in Proceedings of the IEEE/CVF International Conference
on Computer Vision , pp. 15087–15096, 2021.
 
 

 
 [110] 
 
G. Wang, J. Ma, Q. Zhang, J. Lu, and J. Zhou, “Pseudo facial generation with
extreme poses for face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 1994–2003, 2021.
 
 

 
 [111] 
 
M. He, J. Zhang, S. Shan, and X. Chen, “Enhancing face recognition with
self-supervised 3d reconstruction,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 4062–4071, 2022.
 
 

 
 [112] 
 
X. An, X. Zhu, Y. Xiao, L. Wu, M. Zhang, Y. Gao, B. Qin, D. Zhang, and Y. Fu,
“Partial fc: Training 10 million identities on a single machine,” arXiv preprint arXiv:2010.05222 , 2020.
 
 

 
 [113] 
 
X. An, J. Deng, J. Guo, Z. Feng, X. Zhu, J. Yang, and T. Liu, “Killing two
birds with one stone: Efficient and robust training of face recognition cnns
by partial fc,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 4042–4051, 2022.
 
 

 
 [114] 
 
Y. Kim, W. Park, and J. Shin, “Broadface: Looking at tens of thousands of
people at once for face recognition,” in European Conference on
Computer Vision , pp. 536–552, Springer, 2020.
 
 

 
 [115] 
 
B. Li, T. Xi, G. Zhang, H. Feng, J. Han, J. Liu, E. Ding, and W. Liu, “Dynamic
class queue for large scale face recognition in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 3763–3772, 2021.
 
 

 
 [116] 
 
P. Li, B. Wang, and L. Zhang, “Virtual fully-connected layer: Training a
large-scale face recognition dataset with limited computational resources,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 13315–13324, 2021.
 
 

 
 [117] 
 
K. Wang, S. Wang, P. Zhang, Z. Zhou, Z. Zhu, X. Wang, X. Peng, B. Sun, H. Li,
and Y. You, “An efficient training approach for very large scale face
recognition,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 4083–4092, 2022.
 
 

 
 [118] 
 
C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast
adaptation of deep networks,” in International Conference on Machine
Learning , pp. 1126–1135, PMLR, 2017.
 
 

 
 [119] 
 
J. Guo, X. Zhu, C. Zhao, D. Cao, Z. Lei, and S. Z. Li, “Learning meta face
recognition in unseen domains,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 6163–6172, 2020.
 
 

 
 [120] 
 
M. Faraki, X. Yu, Y.-H. Tsai, Y. Suh, and M. Chandraker, “Cross-domain
similarity learning for face recognition in unseen domains,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 15292–15301, 2021.
 
 

 
 [121] 
 
K. Sohn, S. Liu, G. Zhong, X. Yu, M.-H. Yang, and M. Chandraker, “Unsupervised
domain adaptation for face recognition in unlabeled videos,” in Proceedings of the IEEE International Conference on Computer Vision ,
pp. 3210–3218, 2017.
 
 

 
 [122] 
 
J. Deng, S. Cheng, N. Xue, Y. Zhou, and S. Zafeiriou, “Uv-gan: Adversarial
facial uv map completion for pose-invariant face recognition,” in Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 7093–7102, 2018.
 
 

 
 [123] 
 
J. Booth, E. Antonakos, S. Ploumpis, G. Trigeorgis, Y. Panagakis, and
S. Zafeiriou, “3d face morphable models” in-the-wild”,” in Proceedings
of the IEEE conference on computer vision and pattern recognition ,
pp. 48–57, 2017.
 
 

 
 [124] 
 
S. Z. Gilani and A. Mian, “Learning from millions of 3d scans for large-scale
3d face recognition,” in Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition , pp. 1896–1905, 2018.
 
 

 
 [125] 
 
S. Z. Gilani, A. Mian, F. Shafait, and I. Reid, “Dense 3d face
correspondence,” IEEE transactions on pattern analysis and machine
intelligence , vol. 40, no. 7, pp. 1584–1598, 2017.
 
 

 
 [126] 
 
J. D’Errico, “Surface fitting using gridfit,” MATLAB central file
exchange , vol. 643, 2005.
 
 

 
 [127] 
 
H. Fang, W. Deng, Y. Zhong, and J. Hu, “Generate to adapt: Resolution adaption
network for surveillance face recognition,” in European Conference on
Computer Vision , pp. 741–758, Springer, 2020.
 
 

 
 [128] 
 
K. Sun, B. Xiao, D. Liu, and J. Wang, “Deep high-resolution representation
learning for human pose estimation,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 5693–5703, 2019.
 
 

 
 [129] 
 
J. Lezama, Q. Qiu, and G. Sapiro, “Not afraid of the dark: Nir-vis face
recognition via cross-spectral hallucination and low-rank embedding,” in
 Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 6628–6637, 2017.
 
 

 
 [130] 
 
Q. Qiu and G. Sapiro, “Learning transformations for clustering and
classification.,” J. Mach. Learn. Res. , vol. 16, no. 1, pp. 187–225,
2015.
 
 

 
 [131] 
 
H. Zhao, X. Ying, Y. Shi, X. Tong, J. Wen, and H. Zha, “Rdcface: Radial
distortion correction for face recognition,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 7721–7730, 2020.
 
 

 
 [132] 
 
H. Qiu, B. Yu, D. Gong, Z. Li, W. Liu, and D. Tao, “Synface: Face recognition
with synthetic data,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision , pp. 10880–10890, 2021.
 
 

 
 [133] 
 
Y. Deng, J. Yang, D. Chen, F. Wen, and X. Tong, “Disentangled and controllable
face image generation via 3d imitative-contrastive learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 5154–5163, 2020.
 
 

 
 [134] 
 
Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan, and C. Zhang, “Learning efficient
convolutional networks through network slimming,” in Proceedings of the
IEEE international conference on computer vision , pp. 2736–2744, 2017.
 
 

 
 [135] 
 
Z. Liu, M. Sun, T. Zhou, G. Huang, and T. Darrell, “Rethinking the value of
network pruning,” arXiv preprint arXiv:1810.05270 , 2018.
 
 

 
 [136] 
 
J. L. Bentley, “Multidimensional binary search trees used for associative
searching,” Communications of the ACM , vol. 18, no. 9, pp. 509–517,
1975.
 
 

 
 [137] 
 
H. Jegou, M. Douze, and C. Schmid, “Product quantization for nearest neighbor
search,” IEEE transactions on pattern analysis and machine
intelligence , vol. 33, no. 1, pp. 117–128, 2010.
 
 

 
 [138] 
 
Y. Malkov, A. Ponomarenko, A. Logvinov, and V. Krylov, “Approximate nearest
neighbor algorithm based on navigable small world graphs,” Information
Systems , vol. 45, pp. 61–68, 2014.
 
 

 
 [139] 
 
X. Wang, T. Fu, S. Liao, S. Wang, Z. Lei, and T. Mei, “Exclusivity-consistency
regularized knowledge distillation for face recognition,” in Computer
Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28,
2020, Proceedings, Part XXIV 16 , pp. 325–342, Springer, 2020.
 
 

 
 [140] 
 
Y. Huang, J. Wu, X. Xu, and S. Ding, “Evaluation-oriented knowledge
distillation for deep face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 18740–18749,
2022.
 
 

 
 [141] 
 
Y. Sun, X. Wang, and X. Tang, “Deep learning face representation from
predicting 10,000 classes,” in Proceedings of the IEEE conference on
computer vision and pattern recognition , pp. 1891–1898, 2014.
 
 

 
 [142] 
 
L. Tong, Z. Chen, J. Ni, W. Cheng, D. Song, H. Chen, and Y. Vorobeychik,
“Facesec: A fine-grained robustness evaluation framework for face
recognition systems,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 13254–13263, 2021.
 
 

 
 [143] 
 
F. Zhao, J. Feng, J. Zhao, W. Yang, and S. Yan, “Robust lstm-autoencoders for
face de-occlusion in the wild,” IEEE Transactions on Image Processing ,
vol. 27, no. 2, pp. 778–790, 2017.
 
 

 
 [144] 
 
J. Wright, A. Y. Yang, A. Ganesh, S. S. Sastry, and Y. Ma, “Robust face
recognition via sparse representation,” IEEE transactions on pattern
analysis and machine intelligence , vol. 31, no. 2, pp. 210–227, 2008.
 
 

 
 [145] 
 
Y. Deng, Q. Dai, and Z. Zhang, “Graph laplace for occluded face completion and
recognition,” IEEE Transactions on Image Processing , vol. 20, no. 8,
pp. 2329–2338, 2011.
 
 

 
 [146] 
 
S. Zhang, R. He, Z. Sun, and T. Tan, “Demeshnet: Blind face inpainting for
deep meshface verification,” IEEE Transactions on Information Forensics
and Security , vol. 13, no. 3, pp. 637–647, 2017.
 
 

 
 [147] 
 
G. Yuan, H. Zheng, and J. Dong, “Msml: Enhancing occlusion-robustness by
multi-scale segmentation-based mask learning for face recognition,” 2022.
 
 

 
 [148] 
 
T. Feng, L. Xu, H. Yuan, Y. Zhao, M. Tang, and M. Wang, “Towards mask-robust
face recognition,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision , pp. 1492–1496, 2021.
 
 

 
 [149] 
 
C. Shao, J. Huo, L. Qi, Z.-H. Feng, W. Li, C. Dong, and Y. Gao, “Biased
feature learning for occlusion invariant face recognition,” in Proceedings of the Twenty-Ninth International Conference on International
Joint Conferences on Artificial Intelligence , pp. 666–672, 2021.
 
 

 
 [150] 
 
H. Qiu, D. Gong, Z. Li, W. Liu, and D. Tao, “End2end occluded face recognition
by masking corrupted features,” IEEE Transactions on Pattern Analysis
and Machine Intelligence , 2021.
 
 

 
 [151] 
 
B. Huang, Z. Wang, K. Jiang, Q. Zou, X. Tian, T. Lu, and Z. Han, “Joint
segmentation and identification feature learning for occlusion face
recognition,” IEEE Transactions on Neural Networks and Learning
Systems , 2022.
 
 

 
 [152] 
 
W. Zhao, X. Zhu, H. Shi, X.-Y. Zhang, and Z. Lei, “Consistent sub-decision
network for low-quality masked face recognition,” IEEE Signal
Processing Letters , vol. 29, pp. 1147–1151, 2022.
 
 

 
 [153] 
 
B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
“Communication-efficient learning of deep networks from decentralized
data,” in Artificial intelligence and statistics , pp. 1273–1282,
PMLR, 2017.
 
 

 
 [154] 
 
Q. Meng, F. Zhou, H. Ren, T. Feng, G. Liu, and Y. Lin, “Improving federated
learning face recognition via privacy-agnostic clusters,” arXiv
preprint arXiv:2201.12467 , 2022.
 
 

 
 [155] 
 
Y. Niu and W. Deng, “Federated learning for face recognition with gradient
correction,” in Proceedings of the AAAI Conference on Artificial
Intelligence , vol. 36, pp. 1999–2007, 2022.
 
 

 
 [156] 
 
Y. Wang, J. Liu, M. Luo, L. Yang, and L. Wang, “Privacy-preserving face
recognition in the frequency domain,” 2022.
 
 

 
 [157] 
 
H. Wang, X. Wu, Z. Huang, and E. P. Xing, “High-frequency component helps
explain the generalization of convolutional neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 8684–8694, 2020.
 
 

 
 [158] 
 
P. Mohassel and Y. Zhang, “Secureml: A system for scalable privacy-preserving
machine learning,” in 2017 IEEE symposium on security and privacy
(SP) , pp. 19–38, IEEE, 2017.
 
 

 
 [159] 
 
E. Makri, D. Rotaru, N. Smart, and F. Vercauteren, “Epic: Efficient private
image classification,” in Proc. CT-RSA , pp. 473–492.
 
 

 
 [160] 
 
Z. Ma, Y. Liu, X. Liu, J. Ma, and K. Ren, “Lightweight privacy-preserving
ensemble classification for face recognition,” IEEE Internet of Things
Journal , vol. 6, no. 3, pp. 5778–5790, 2019.
 
 

 
 [161] 
 
S. Wagh, S. Tople, F. Benhamouda, E. Kushilevitz, P. Mittal, and T. Rabin,
“Falcon: Honest-majority maliciously secure framework for private deep
learning,” arXiv preprint arXiv:2004.02229 , 2020.
 
 

 
 [162] 
 
W. Zhu, C.-Y. Wang, K.-L. Tseng, S.-H. Lai, and B. Wang, “Local-adaptive face
recognition via graph-based meta-clustering and regularized adaptation,” in
 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 20301–20310, 2022.
 
 

 
 [163] 
 
L. Yang, D. Chen, X. Zhan, R. Zhao, C. C. Loy, and D. Lin, “Learning to
cluster faces via confidence and connectivity estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pp. 13369–13378, 2020.
 
 

 
 [164] 
 
H. B. Mcmahan, E. Moore, D. Ramage, and B. Arcas, “Federated learning of deep
networks using model averaging,” 2016.
 
 

 
 [165] 
 
C.-T. Liu, C.-Y. Wang, S.-Y. Chien, and S.-H. Lai, “Fedfr: Joint optimization
federated framework for generic and personalized face recognition,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 36,
pp. 1656–1664, 2022.
 
 

 
 [166] 
 
M. Wang and W. Deng, “Deep face recognition: A survey,” Neurocomputing ,
vol. 429, pp. 215–244, 2021.
 
 

 
 [167] 
 
D. Yi, Z. Lei, S. Liao, and S. Z. Li, “Learning face representation from
scratch,” arXiv preprint arXiv:1411.7923 , 2014.
 
 

 
 [168] 
 
Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the
wild,” in Proceedings of the IEEE international conference on computer
vision , pp. 3730–3738, 2015.
 
 

 
 [169] 
 
A. Bansal, A. Nanduri, C. D. Castillo, R. Ranjan, and R. Chellappa, “Umdfaces:
An annotated face dataset for training deep networks,” in 2017 IEEE
international joint conference on biometrics (IJCB) , pp. 464–473, IEEE,
2017.
 
 

 
 [170] 
 
Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman, “Vggface2: A dataset
for recognising faces across pose and age,” in 2018 13th IEEE
international conference on automatic face gesture recognition (FG 2018) ,
pp. 67–74, IEEE, 2018.
 
 

 
 [171] 
 
Y. Guo, L. Zhang, Y. Hu, X. He, and J. Gao, “Ms-celeb-1m: A dataset and
benchmark for large-scale face recognition,” in European conference on
computer vision , pp. 87–102, Springer, 2016.
 
 

 
 [172] 
 
J. Deng, Y. Zhou, and S. Zafeiriou, “Marginal loss for deep face
recognition,” in Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition Workshops , pp. 60–68, 2017.
 
 

 
 [173] 
 
F. Wang, L. Chen, C. Li, S. Huang, Y. Chen, C. Qian, and C. C. Loy, “The devil
of face recognition is in the noise,” in Proceedings of the European
Conference on Computer Vision (ECCV) , pp. 765–780, 2018.
 
 

 
 [174] 
 
J. Cao, Y. Li, and Z. Zhang, “Celeb-500k: A large training dataset for face
recognition,” in 2018 25th IEEE International Conference on Image
Processing (ICIP) , pp. 2406–2410, IEEE, 2018.
 
 

 
 [175] 
 
Z. Zhu, G. Huang, J. Deng, Y. Ye, J. Huang, X. Chen, J. Zhu, T. Yang, J. Lu,
D. Du, et al. , “Webface260m: A benchmark unveiling the power of
million-scale deep face recognition,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 10492–10502,
2021.
 
 

 
 [176] 
 
I. Kemelmacher-Shlizerman, S. M. Seitz, D. Miller, and E. Brossard, “The
megaface benchmark: 1 million faces for recognition at scale,” in Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 4873–4882, 2016.
 
 

 
 [177] 
 
G. B. Huang, M. Mattar, T. Berg, and E. Learned-Miller, “Labeled faces in the
wild: A database forstudying face recognition in unconstrained
environments,” in Workshop on faces in’Real-Life’Images: detection,
alignment, and recognition , 2008.
 
 

 
 [178] 
 
T. Zheng and W. Deng, “Cross-pose lfw: A database for studying cross-pose face
recognition in unconstrained environments,” Beijing University of Posts
and Telecommunications, Tech. Rep , vol. 5, p. 7, 2018.
 
 

 
 [179] 
 
T. Zheng, W. Deng, and J. Hu, “Cross-age lfw: A database for studying
cross-age face recognition in unconstrained environments,” arXiv
preprint arXiv:1708.08197 , 2017.
 
 

 
 [180] 
 
L. Wolf, T. Hassner, and I. Maoz, “Face recognition in unconstrained videos
with matched background similarity,” in CVPR 2011 , pp. 529–534, IEEE,
2011.
 
 

 
 [181] 
 
H.-W. Ng and S. Winkler, “A data-driven approach to cleaning large face
datasets,” in 2014 IEEE international conference on image processing
(ICIP) , pp. 343–347, IEEE, 2014.
 
 

 
 [182] 
 
A. Lanitis, C. J. Taylor, and T. F. Cootes, “Toward automatic simulation of
aging effects on face images,” IEEE Transactions on pattern Analysis
and machine Intelligence , vol. 24, no. 4, pp. 442–455, 2002.
 
 

 
 [183] 
 
B.-C. Chen, C.-S. Chen, and W. H. Hsu, “Cross-age reference coding for
age-invariant face recognition and retrieval,” in European conference
on computer vision , pp. 768–783, Springer, 2014.
 
 

 
 [184] 
 
K. Ricanek and T. Tesafaye, “Morph: A longitudinal image database of normal
adult age-progression,” in 7th International Conference on Automatic
Face and Gesture Recognition (FGR06) , pp. 341–345, IEEE, 2006.
 
 

 
 [185] 
 
S. Sengupta, J.-C. Chen, C. Castillo, V. M. Patel, R. Chellappa, and D. W.
Jacobs, “Frontal to profile face verification in the wild,” in 2016
IEEE Winter Conference on Applications of Computer Vision (WACV) , pp. 1–9,
IEEE, 2016.
 
 

 
 [186] 
 
B. F. Klare, B. Klein, E. Taborsky, A. Blanton, J. Cheney, K. Allen,
P. Grother, A. Mah, and A. K. Jain, “Pushing the frontiers of unconstrained
face detection and recognition: Iarpa janus benchmark a,” in Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 1931–1939, 2015.
 
 

 
 [187] 
 
R. Gross, I. Matthews, J. Cohn, T. Kanade, and S. Baker, “Multi-pie,” Image and vision computing , vol. 28, no. 5, pp. 807–813, 2010.
 
 

 
 [188] 
 
S. Moschoglou, A. Papaioannou, C. Sagonas, J. Deng, I. Kotsia, and
S. Zafeiriou, “Agedb: the first manually collected, in-the-wild age
database,” in Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition Workshops , pp. 51–59, 2017.
 
 

 
 [189] 
 
G. B. Huang and E. Learned-Miller, “Labeled faces in the wild: Updates and new
reporting procedures,” Dept. Comput. Sci., Univ. Massachusetts Amherst,
Amherst, MA, USA, Tech. Rep , vol. 14, no. 003, 2014.
 
 

 
 [190] 
 
Y. Liu et al. , “Towards flops-constrained face recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision
Workshops , pp. 0–0, 2019.
 
 

 
 [191] 
 
S. Lloyd, “Least squares quantization in pcm,” IEEE transactions on
information theory , vol. 28, no. 2, pp. 129–137, 1982.
 
 

 
 [192] 
 
J. Shi and J. Malik, “Normalized cuts and image segmentation,” IEEE
Transactions on pattern analysis and machine intelligence , vol. 22, no. 8,
pp. 888–905, 2000.
 
 

 
 [193] 
 
M. Ester, H.-P. Kriegel, J. Sander, X. Xu, et al. , “A density-based
algorithm for discovering clusters in large spatial databases with noise.,”
in kdd , vol. 96, pp. 226–231, 1996.
 
 

 
 [194] 
 
L. Yang, X. Zhan, D. Chen, J. Yan, C. C. Loy, and D. Lin, “Learning to cluster
faces on an affinity graph,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pp. 2298–2306, 2019.
 
 

 
 [195] 
 
Z. Wang, L. Zheng, Y. Li, and S. Wang, “Linkage based face clustering via
graph convolution network,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pp. 1117–1125, 2019.
 
 

 
 [196] 
 
X. Zhan, Z. Liu, J. Yan, D. Lin, and C. C. Loy, “Consensus-driven propagation
in massive unlabeled data for face recognition,” in Proceedings of the
European Conference on Computer Vision (ECCV) , pp. 568–583, 2018.
 
 

 
 [197] 
 
S. Shen, W. Li, Z. Zhu, G. Huang, D. Du, J. Lu, and J. Zhou, “Structure-aware
face clustering on a large-scale graph with 107 nodes,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
pp. 9085–9094, 2021.
 
 

 
 [198] 
 
S. Guo, J. Xu, D. Chen, C. Zhang, X. Wang, and R. Zhao, “Density-aware feature
embedding for face clustering,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 6698–6706, 2020.
 
 

 
 [199] 
 
H. Han, A. K. Jain, F. Wang, S. Shan, and X. Chen, “Heterogeneous face
attribute estimation: A deep multi-task learning approach,” IEEE
transactions on pattern analysis and machine intelligence , vol. 40, no. 11,
pp. 2597–2609, 2017.
 
 

 
 [200] 
 
F. Wang, H. Han, S. Shan, and X. Chen, “Deep multi-task learning for joint
prediction of heterogeneous face attributes,” in 2017 12th IEEE
International Conference on Automatic Face Gesture Recognition (FG 2017) ,
pp. 173–179, IEEE, 2017.
 
 

 
 [201] 
 
E. M. Hand and R. Chellappa, “Attributes for improved attributes: A multi-task
network utilizing implicit and explicit relationships for facial attribute
classification,” in Thirty-First AAAI Conference on Artificial
Intelligence , 2017.
 
 

 
 [202] 
 
A. V. Savchenko, “Facial expression and attributes recognition based on
multi-task learning of lightweight neural networks,” arXiv preprint
arXiv:2103.17107 , 2021.
 
 

 
 [203] 
 
J. Cao, Y. Li, and Z. Zhang, “Partially shared multi-task convolutional neural
network with local constraint for face attribute learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition , pp. 4290–4299, 2018.
 
 

 
 [204] 
 
S. Chen, C. Zhang, M. Dong, J. Le, and M. Rao, “Using ranking-cnn for age
estimation,” in Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition , pp. 5183–5192, 2017.
 
 

 
 [205] 
 
H. Pan, H. Han, S. Shan, and X. Chen, “Mean-variance loss for deep age
estimation from a face,” in Proceedings of the IEEE conference on
computer vision and pattern recognition , pp. 5285–5294, 2018.
 
 

 
 [206] 
 
C. Zhang, S. Liu, X. Xu, and C. Zhu, “C3ae: Exploring the limits of compact
model for age estimation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pp. 12587–12596, 2019.
 
 

 
 [207] 
 
X. Zeng, J. Huang, and C. Ding, “Soft-ranking label encoding for robust facial
age estimation,” IEEE Access , vol. 8, pp. 134209–134218, 2020.
 
 

 
 [208] 
 
H. Yang, U. Ciftci, and L. Yin, “Facial expression recognition by
de-expression residue learning,” in Proceedings of the IEEE conference
on computer vision and pattern recognition , pp. 2168–2177, 2018.
 
 

 
 [209] 
 
W. Zhang, X. Ji, K. Chen, Y. Ding, and C. Fan, “Learning a facial expression
embedding disentangled from identity,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pp. 6759–6768, 2021.
 
 

 
 [210] 
 
P. Antoniadis, P. P. Filntisis, and P. Maragos, “Exploiting emotional
dependencies with graph convolutional networks for facial expression
recognition,” arXiv preprint arXiv:2106.03487 , 2021.
 
 

 
 [211] 
 
F. Xue, Q. Wang, and G. Guo, “Transfer: Learning relation-aware facial
expression representations with transformers,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision , pp. 3601–3610, 2021.
 
 

 
 [212] 
 
F. Rodrigues and F. Pereira, “Deep learning from crowds,” in Proceedings
of the AAAI Conference on Artificial Intelligence , vol. 32, 2018.
 
 

 
 [213] 
 
P. J. Phillips, P. Grother, R. Micheals, D. M. Blackburn, E. Tabassi, and
M. Bone, “Face recognition vendor test 2002,” in 2003 IEEE
International SOI Conference. Proceedings (Cat. No. 03CH37443) , p. 44, IEEE,
2003.