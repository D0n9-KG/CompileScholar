Fighting Malicious Media Data: A Survey on Tampering Detection and Deepfake Detection 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.05667v1 [cs.CV] 12 Dec 2022 
 
 

# Fighting Malicious Media Data: A Survey on Tampering Detection and Deepfake Detection Thanks:  J. Wang, Z. Li, C. Zhang, J. Chen, Z. Wu and Y.G. Jiang are with School of Computer Science, Fudan University.
L. Davis is with University of Maryland. 

 
 
 Junke Wang
 
    
 Zhenxin Li
 
    
 Chao Zhang
 
    
 Jingjing Chen
 
    
 Zuxuan Wu
 
 Affiliation:  Larry S. Davis , Yu-Gang Jiang 

 

 Abstract 
 
 Online media data, in the forms of images and videos, are becoming mainstream communication channels. However, recent advances in deep learning, particularly deep generative models, open the doors for producing perceptually convincing images and videos at a low cost, which not only poses a serious threat to the trustworthiness of digital information but also has severe societal implications. This motivates a growing interest of research in media tampering detection, i.e. , using deep learning techniques to examine whether media data have been maliciously manipulated. Depending on the content of the targeted images, media forgery could be divided into image tampering and Deepfake techniques. The former typically moves or erases the visual elements in ordinary images, while the latter manipulates the expressions and even the identity of human faces. Accordingly, the means of defense include image tampering detection and Deepfake detection, which share a wide variety of properties. In this paper, we provide a comprehensive review of the current media tampering detection approaches, and discuss the challenges and trends in this field for future research.

 
 
 
 Index Terms:  Media forensics, Tampering detection, Deepfake detection.

 
 

## I Introduction 

 
 Recent years have witnessed the rapid development of deep generative models  [ 1 , 2 , 3 ] , which are able to generate perceptually convincing images and videos. While these techniques can benefit applications like AR/VR, creative designs, image editing, etc , they are essentially a double-edged sword considering the potential impact on human society. For instance, the ubiquity of open-sourced tools based on deep generative models makes it extremely easy to create manipulated media data, e.g. , inpainted images  [ 4 ] or Deepfakes  [ 5 , 6 ] that could be disseminated on the Internet for malicious purposes. This poses serious threats to the integrity of social applications like influencing president elections.

 
 
 The above challenges have stimulated a growing interest to automatically discriminate whether media data have been tampered or not with handcrafted features  [ 7 , 8 ] and even deep learning techniques  [ 9 , 10 , 11 , 12 , 13 ] , with an aim to fight misinformation. In this paper, we classify the existing media forensics method into two categories: tampering detection and Deepfake detection. More specifically, tampering detection aims to detect whether generic objects have been added or removed in images and videos  [ 14 , 9 , 15 , 11 ] , while Deepfake detection is more face-oriented that aims to detect whether the expression or identity of human faces in images/videos has been manipulated  [ 16 , 17 , 13 ] . Current literature typically treats tampering detection and Deepfake detection as separate problems due to their differences in the overall process: (1) as Deepfake detection focuses on human faces, the input is typically a cropped face image, while tampering detection often takes in the entire image as inputs and aim to identify forged regions; (2) tampering detection requires to locate the manipulated areas, while most Deepfake detection methods only produce a binary Real/Fake prediction.

 
 
 Fig. 1: Visual artifacts of the manipulated images (top) and Deepfakes (bottom) in existing datasets. 
 
 
 Fig. 2: An overview of media forensics, which can be divided into tampering detection (TD) and Deepfake detection (DD). Typically, TD methods take a complete image as input and simultaneously identify the authenticity and locate the manipulated regions, while DD methods take a cropped face as input and produce a binary Real/Fake prediction. 
 
 
 But essentially, both tampering detection and Deepfake detection are discriminative tasks that attempt to discover forgery traces through careful examination of visual contents. As a result, they both highly rely on (1) inconsistency modeling: the imaging principle of the camera determines that the pixels of pristine images follow a certain statistical distribution, which generative models typically struggle to reconstruct  [ 18 ] . Therefore, media tampering including image manipulation and Deepfake, followed by blending techniques that mix altered objects/faces with background images, will lead to intrinsic inconsistency, between the manipulated and authentic regions. This inspires forensics approaches to carefully examine the visual artifacts (see Figure  1 ) in suspicious images to discover both global  [ 19 ] and local  [ 12 , 20 , 21 ] inconsistent information and (2) robust features: media data inevitably go through various post-processing during its creation and dissemination. In addition, malicious users will deliberately impose perturbations on the tampered images to fool the detection tools. This could erase the manipulation traces and significantly increase the difficulty for forensics, requiring the defending approaches to capture more robust and meaningful forgery evidence.

 
 
 The common nature shared by tampering detection and Deepfake detection motivates us to present a unique and comprehensive survey of both fields in this literature to facilitate future research. While there are a few concurrent surveys, they typically put a primary focus on a single aspect of tampering detection  [ 22 , 23 ] or Deepfake detection  [ 24 , 25 , 26 ] . Our work differs in that we believe tampering detection and Deepfake detection are similar in spirit and thus we summarize the newest advances of both fields systematically, and hope that future research can learn from the best of both worlds.

 
 
 The remainder of this survey is organized as follows: in Sec.  II and Sec.  III , we separately introduce the progress in the field of tampering detection and Deepfake detection, including manipulation techniques, public datasets for evaluation, and various detection techniques. Sec.  IV presents the remaining challenges in the field of tampering detection, and outlines the future research directions. Finally, we briefly conclude this paper in the following Sec.  V . The outline of this survey is illustrated in Figure  2 .

 
 
 

## II Tampering Detection 

 
 With the rise of data-driven techniques and the vast amount of media content within easy reach, media tampering methods  [ 27 , 28 , 29 ] allow users to edit or create realistic images automatically, and as a result, there grows an urgent need for effective and trustworthy detection methods  [ 19 , 30 , 31 ] that can distinguish fake images from real ones to preserve the credibility of media content. In this section, we introduce the advancements made in the generation (Sec.  II-A ), the datasets (Sec.  II-B ) and the detection (Sec.  II-C ) of tampered images. For tampering detection, our main focus is on the images whose subjects are generic and not constrained to a particular type, while for the following section Deepfake detection (Sec.  III ), we dive into the detection of manipulated human faces.

 
 

### II-A Tampering Generation 

 
 From the earliest automatic inpainting methods that remove unwanted elements from images  [ 32 ] , much research in generating images with authenticity based on provided source objects ( e.g. , splicing, editing, removal of subjects) has been carried out. Early approaches  [ 33 , 34 , 35 ] focus on the texture and structure information presented in images to reconstruct target image patches or regions. Recently, deep learning-based generative models, e.g. , GANs  [ 1 ] and diffusion models  [ 36 , 37 ] , are also proposed to achieve realistic synthesis without devoting too much effort to the analysis of the intrinsic distributional information. This also opens up new opportunities for more user-interactive manipulations  [ 29 , 27 ] since researchers can focus more on how to embed user interactions into the model rather than merely improve the synthesis quality. Based on the algorithms adopted, tampering generation can be categorized into conventional methods (Sec.  II-A1 ) and deep learning-based methods (Sec.  II-A2 ). We illustrate the pipeline of image tampering approaches in Figure  3 .

 
 

#### II-A 1 Conventional Methods

 
 Conventional tampering methods conduct forgeries based on the information within the same image, which can be cheaply calculated and directly applied to the target region. Even though these methods do not require training on large amounts of data, they are still sometimes time-consuming due to the computational complexity of the algorithm, e.g. , the cost of matching similar patches  [ 35 ] , which may significantly hinder user interactions.

 
 
 Specifically, [ 34 , 33 ] can handle the removal of large objects in images through exemplar-based synthesis, which iteratively select a template region along the target contour to inpaint given the best matching patch with similar textures and structures. After each iteration, the target region is shrunk as the template region is filled with new content and its contour is propelled inward. Furthermore, [ 35 ] proposes a randomized algorithm to optimize the process of patch-matching, which first assigns some initial guesses of similar patches and then refines the best-fitting patch through random searches in its concentric neighborhoods.

 
 
 Fig. 3: Illustration of the process to manipulate an image through (a) copy-move, (b) splicing, and (c) removal. 
 
 
 

#### II-A 2 Deep Learning-Based Methods

 
 Compared with conventional methods, deep learning-based tampering methods, typically deep generative models like GANs  [ 1 ] , are able to synthesize realistic images by training a generator and a discriminator in an adversarial manner. Conditional GANs  [ 38 ] , on the other hand, take additional driving signals as inputs to guide the creation of images. With these approaches, manipulating images in a data-driven manner becomes feasible.
Here, we divide these deep learning-based tampering methods into three categories in terms of their manipulation objective, i.e. , splicing, object editing, and object removal.

 
 
 Splicing copies regions from different source images and pastes them to the target image. Built upon the GAN-based architecture, several approaches  [ 39 , 40 , 41 ] adopt the Spatial Transformer Network in their architecture to warp the source object into the background geometrically, and  [ 40 , 41 ] further model the appearance patterns of sources to obtain appearance-preserving results.
To better maintain the characteristics of sources, [ 42 ] first combines two images and then separates them using a decomposition network, so the reconstructed objects serve as an additional supervisory signal.
 [ 43 , 44 ] both utilize semantic maps and user input to combine source images. Specifically, [ 44 ] estimates depth information and allows users to decide the occlusion relationships between different objects.

 
 
 Object Editing aims to manipulate the attributes of objects according to the instructions of users. We particularly focus on multi-modal object editing ( i.e. text-based  [ 45 , 27 , 29 ] , sound-based   [ 28 ] ) in this part.
A text-driven generative model  [ 45 , 27 , 29 ] typically contains an image encoder and a text encoder, from which the latent representations of the source image and the driving information are respectively derived. To fuse the representations, [ 45 ] proposes a text-image affine combination module, which proves to be more effective than simply concatenating features. To further generate entity-level believable results, [ 27 ] adopts the transformer architecture  [ 46 ] with tokenized text and image as input to model the relationships between
different modalities, while [ 29 ] combines a diffusion model  [ 36 , 37 ] with CLIP and allows zero-shot image manipulation between unseen domains.
Furthermore,  [ 28 ] extends the CLIP-based manipulation methods  [ 47 , 29 ] with an additional audio branch and a shared multi-modal latent space to embed triplet pairs ( i.e. , image, text, and sound). The derived latent code is then given as input to a StyleGAN2  [ 48 ] generator alongside encoded audio features to generate the final image.
Besides the aforementioned cross-modal editing methods, [ 49 ] generates a semantic scene graph, whose nodes and edges respectively represent the objects present in the image and their relations, to model the target image in a semantically manipulable manner.

 
 
 Object Removal , introduced in Sec.  II-A1 , erases regions from
images and inpaints missing regions with visually plausible
contents. Observed by [ 50 ] , relying on the generator itself to perform object removal tends to lead to a completely re-synthesized image, while we only desire to manipulate one region. To address this issue, [ 50 ] proposes a two-stage architecture composed of a mask generator and an inpainting network, which are jointly trained to achieve satisfactory removal results. It is worth noting that removing a node in the scene graphs produced by  [ 49 ] can also perform object removal in an image.

 
 
 TABLE I: Basic information of existing image tampering detection datasets. 
 
 
 Dataset | 
 
 
 
 Images 
 | 
 
 
 
 Manipulations 
 | 

 
 
 
 
 Real 
 | 
 
 
 
 Forged 
 | 
 
 
 
 splicing 
 | 
 
 
 
 copy-move 
 | 
 
 
 
 removal 
 | 

 
 Columbia Gray  [ 51 ] | 
 933 | 
 912 | 
 ✓ | 
 | 
 | 

 
 Columbia Color  [ 52 ] | 
 183 | 
 180 | 
 ✓ | 
 | 
 | 

 
 MICC-F8multi  [ 53 ] | 
 - | 
 8 | 
 | 
 ✓ | 
 | 

 
 MICC-F220  [ 53 ] | 
 110 | 
 110 | 
 | 
 ✓ | 
 | 

 
 MICC-F600  [ 53 ] | 
 440 | 
 160 | 
 | 
 ✓ | 
 | 

 
 MICC-F2000  [ 53 ] | 
 1300 | 
 700 | 
 | 
 ✓ | 
 | 

 
 VIPP Synth.  [ 54 ] | 
 4800 | 
 4800 | 
 ✓ | 
 | 
 | 

 
 VIPP Real.  [ 54 ] | 
 69 | 
 69 | 
 ✓ | 
 | 
 | 

 
 CoMoFod  [ 55 ] | 
 260 | 
 260 | 
 | 
 ✓ | 
 | 

 
 CASIA V1.0  [ 56 ] | 
 800 | 
 921 | 
 ✓ | 
 ✓ | 
 | 

 
 CASIA V2.0  [ 56 ] | 
 7200 | 
 5123 | 
 ✓ | 
 ✓ | 
 | 

 
 Wild Web  [ 57 ] | 
 90 | 
 9657 | 
 ✓ | 
 | 
 | 

 
 NC2016  [ 58 ] | 
 560 | 
 564 | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 NC2017  [ 58 ] | 
 2667 | 
 1410 | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 MFC2018  [ 58 ] | 
 14156 | 
 3265 | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 MFC2019  [ 58 ] | 
 10279 | 
 5750 | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 PS-Battles  [ 59 ] | 
 11142 | 
 102028 | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 DEFACTO  [ 60 ] | 
 - | 
 229000 | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 IMD2020  [ 61 ] | 
 35000 | 
 35000 | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 
 
 

### II-B Tampering Datasets 

 
 In order to facilitate the development of tampering detection, various datasets are proposed to benchmark detection approaches. Table  I presents the commonly used tampering datasets as well as their corresponding manipulation approaches.

 
 
 Since the release of the earliest Columbia   [ 51 , 52 ] comprising splicing forgeries, numerous datasets focusing on other manipulations, such as copy-move  [ 53 , 62 , 54 , 55 ] have been developed. CASIA v1.0 and CASIA v2.0   [ 56 ] are among the first to include two kinds of manipulations in one dataset. Even though CASIA v2.0  [ 56 ] is already much larger ( i.e. , 7200 real images and 5123 forged images) than previously released datasets, it was not scalable enough because the forged images were manually crafted by a group of image makers using Adobe Photoshop. To produce a large number of tampered images and add more comprehensiveness to the data, the Wild Web   [ 57 ] collects forged images from the Internet and its scale greatly surpasses the aforementioned datasets, while the PS-Battles   [ 59 ] similarly gathers an extensive number of images edited by Adobe Photoshop from Reddit communities. In addition, the MFC Datasets   [ 58 ] released by NIST involve a series of comprehensive datasets ( i.e. , NC2016, NC2017, MFC2018, MFC2019), which sets a notable benchmark for the evaluation of media tampering detection. DEFACTO   [ 60 ] and IMD2020   [ 61 ] are the other two recently published large-scale datasets, and DEFACTO obtains its sources from the Microsoft COCO dataset  [ 63 ] , while IMD2020 derives its samples from manually selected online images that are free of distinct manipulation traces. The latest ImageForensicsOSN dataset  [ 31 ] leverages tampered images from multiple sources  [ 58 , 64 , 52 , 56 ] and produces OSN-transmitted ( i.e. , transmitted via an online social network, which induces noises in images) versions of them to facilitate detection methods that are robust against OSN-transmissions. We demonstrate several samples from the above tampering datasets  [ 52 , 55 , 56 , 61 ] in Figure  4 .

 
 
 Fig. 4: Examples from the existing tampering datasets. We display both tampered images and masks. 
 
 
 

### II-C Tampering Detection 

 
 In this section, we elaborate on media tampering detection, which is a research field of growing importance since deep generative models described in Sec.  II-A2 exhibit effectiveness and handiness. Nevertheless, the emergence of deep learning benefits not only manipulation tools but also forensic techniques by sparing forgery analysts from handcrafting features or rules for forensic purposes. Though the combination of deep learning (DL) and tampering detection marks a notable milestone in the field, DL-based detectors lack interpretabilities, yielding questionable results that
can be difficult to accept on serious occasions such as a court.

 
 
 In an effort to frame the problem of tampering detection systematically, we take both current research trends and classic forensic techniques in early works into account and thus formulate three major categories for tampering detection. The first and second class, i.e. Rule-based (Sec.  II-C1 ) and Learning-based (Sec.  II-C2 ) methods, are categorized based on whether machine learning (ML) or deep learning (DL) algorithms are utilized to detect tampering clues.
On the other hand, we find that some methods gracefully combine the rule-based methods devised by forgery analysts with neural networks, which make them different from methods falling into the previous two categories. As a result, we come up with the third category, i.e. Hybrid Detection Methods (Sec.  II-C3 ), alongside the previous two to present readers the whole picture of tampering detection.

 
 

#### II-C 1 Rule-based Detection Methods

 
 For rule-based detection methods, we classify them into intrinsic patterns analysis and methods with statistical references. Considering the differences in the inference process of these methods, the first category requires no prior knowledge of image samples and carries out tampering detection on them straightly, while methods from the second category estimate whether an image is tampered with based on statistics of pristine samples.

 
 
 Intrinsic Patterns Analysis is a kind of detection techniques that explores the underlying patterns within image samples. By identifying the abnormalities induced by image tampering, it carries out tampering detection exclusively on the image samples without the help of additional prior knowledge. The intrinsic patterns of a naturally produced image result from the acquisition chain and the encoding process, and the consistency in these patterns can be disrupted after the image is manipulated.

 
 
 Different color components tend to focus on different lateral offsets on the sensor when an image is captured by the camera’s optical lens due to their varied wavelengths. This phenomenon, named lateral chromatic aberration (LCA), can be exploited for tampering detection  [ 65 , 66 ] by identifying the inconsistent LCA displacement in manipulated regions. Nevertheless, LCA can be eliminated by image post-processing, which may invalidate these methods.

 
 
 After the light goes through the lens, it becomes post-processed by a color filter array, which filters light of different wavelengths and separates color information for the image sensor. This process is generally followed by a procedure known as demosaicing, where color interpolation is performed to fill the missing pixel values for the final image, leaving detectable demosaicing traces on the image as a fingerprint. The lack of such fingerprints in an image region usually suggests the occurrence of image tampering. Following  [ 67 ] , studies
made by  [ 68 , 69 ] seek to model the periodic pattern produced by demosaicing. Apart from leaving demosaicing traces on the image, the image sensor, as well as the rest of the acquisition chain, tends to impose unwanted noise on the image. Despite its randomness, the noise can similarly be modeled as a periodic pattern. [ 70 , 71 ] reveal splicing and copy-move forgeries by detecting irregular noise levels, while  [ 72 ] applies noise residuals to the estimation of a camera-imposed fingerprint.

 
 
 Another crucial step in acquiring the image content suitable for transmission is compression, especially JPEG compression, which happens both at the device level and the software level. When a compressed image is manipulated and then saved by a photo-editing software, e.g. , Adobe Photoshop, it is compressed again, leaving detectable double compression artifacts on the image  [ 73 ] . However, double compression artifacts alone do not suffice to indicate the presence of image tampering since the user may only save the image without editing it. To address this issue,   [ 7 ] points out that the tampered regions are very likely to be compressed again under misaligned grids, leaving detectable traces for tampering detection, while  [ 74 ] performs periodicity analysis on double-compressed images in both spatial and transform domains, which proves robust enough against different alignment situations. Besides,  [ 75 , 76 ] look into the histograms of DCT (Discrete Cosine Transform) coefficients in JPEG compression to discover statistical inconsistencies, while  [ 77 ] inventively compresses the image further with different JPEG qualities to expose the “JPEG ghost”. Aside from the previously mentioned methods,  [ 78 ] analyzes the patterns of blocking artifacts produced by JPEG compression, while  [ 79 ] investigates the artifacts originating from chroma sub-sampling in JPEG compression.

 
 
 Eventually, to create a realistic tampered image, editors need to make the tampered region blend in with its background. Common photo editing approaches such as blurring, shape and color transformations are often adopted during handcrafted image tampering, which may lead to detectable manipulation traces. For the splicing detection techniques  [ 80 , 81 ] , whose test images are the combination of multiple sources rather than single-sourced, the inconsistent pixel correlations  [ 80 ] and blur types  [ 81 ] are explored. As for copy-move detection, many methods seek to find the repetitive occurrence of the same image component. The earliest method proposed in  [ 82 ] detects copy-move manipulations by matching and comparing small image segments. Later, [ 83 , 84 , 53 , 62 , 85 , 86 , 87 , 88 , 89 ] all perform copy-move forensics by either utilizing image keypoints  [ 53 , 85 , 87 , 89 , 88 ] , or image block features  [ 83 , 84 , 86 ] . By using hierarchical feature point matching, [ 90 ] improves keypoint-based methods in terms of their insufficient keypoint sampling and their inabilities to cope with regions with varied sizes and textures.

 
 
 Methods with Statistical References refer to the forensic methods that require prior knowledge related to the image acquisition chain as detection guidance. Here, we specifically discuss the methods using the PRNU ( i.e. , photo-response non-uniformity) noise, which can be regarded as a sensor-specific fingerprint. The major difference between this genre and the aforementioned intrinsic patterns analysis consists in the fact that this genre first approximates a pattern of an imaging sensor given a statistical sample and then compares the pattern of test image with it to derive a detection result  [ 91 , 92 ] .
It is worth noting that the test image is preprocessed with a denoising filter to remove low-frequency components of PRNU and high-level scene content to speed up PRNU estimation  [ 91 ] . However, PRNU signals are generally weak and thus hinder the detection of small-sized forgeries, so  [ 93 ] designs an adaptive filtering technique to enhance the resolution of PRNU-based methods. Other methods proposed in  [ 94 , 95 ] are further integrated with machine learning algorithms.

 
 
 

#### II-C 2 Learning-based Detection Methods

 
 In this section, we further expand on tampering detection by discussing learning-based forensic tools, among which machine learning-based methods will first be introduced, followed by today’s trending topic, i.e. , deep learning-based tampering detectors.

 
 
 Machine Learning Methods with Handcrafted Features require the researchers to investigate the manipulation traces left on a forged image, which are further analyzed by an ML algorithm to perform a binary Pristine/Fake classification. Frequency domain statistics are widely explored to capture the subtle forgery artifacts by DCT transformation  [ 96 ] , DWT transformation [ 97 , 98 ] , and even their combinations  [ 99 ] . In addition,   [ 100 , 52 , 101 ] model the camera characteristics consistency, and  [ 102 , 103 ] investigate features from the color space of an image. With these handcrafted features available,   [ 104 , 99 , 105 , 106 , 107 ] further adopt different fusion strategies to produce the final predictions, which achieves more competitive results than operating on a single type of features. Specifically, [ 104 , 99 ] extract multiple types of features and select them accordingly to construct the best feature set, while  [ 105 , 106 ] leverage decisions made by various approaches ( e.g. , the ML algorithm, the patch-matching algorithm, and the PRNU-based method), and  [ 107 ] combines classifiers with the BKS (Behavior-Knowledge Space) method  [ 108 ] .

 
 
 Deep Learning Methods have dominated the field of tampering detection demonstrating superior performance since their emergence. An extensive number of studies are carried out to explore different network structures or training methods to achieve better results. A line of work  [ 109 , 110 , 111 ] focuses on a specific type of image tampering ( e.g. splicing and copy-move). For the splicing detection,   [ 109 ] performs deep matching between the features from a query image and a potential donor image ( i.e. , a part of the donor’s region may be spliced with the query image) extracted by two weight-sharing CNNs. Besides,   [ 110 ] proposes a multi-task CNN to simultaneously predict an edge mask and a commonly used splicing mask to develop more sharp boundaries, and [ 112 ] leverages global and local features derived from the entire RGB image and image patches. In addition,  [ 111 ] renovates the structure of the U-Net  [ 113 ] by adding a residual feedback connection to a ResNet layer  [ 114 ] for feature enhancement.
For the copy-move detection,  [ 115 ] proposes a Self-Correlation module to calculate the similarity between different pixels in a feature map, which is further utilized to localize the repeated regions. Furthermore,  [ 116 ] proposes a two-branch network where one branch localizes only the manipulated area while the other one follows  [ 115 ] to predict cloned regions.   [ 117 ] designs a network with dense feature connections and a key-point-based Feature Correlation, which outperforms  [ 116 ] on unseen manipulated objects during the training stage. Building upon the attention mechanism,  [ 118 ] designs a Dual-Order Attention module that first computes two attention maps based on image self-correlation, with which attentive features are calculated by element-wise product and matrix multiplication. Given these robust features, the localization network is then trained in an adversarial manner with a PatchGAN discriminator  [ 119 ] , which improves the localization ability of the generator and the image encoder. Also involving generative models, [ 120 ] uses an Autoencoder to learn the data distribution of pristine images by performing image reconstruction and discriminative labeling on encoded features to locate the anomalies without training on forged images.

 
 
 Instead of focusing on one particular tampering method, universal forensic tools  [ 121 ] are agnostic to the tampering method, as well as what pre- and post-processing operations are used and which camera the image is captured by. Without the constraints of certain assumptions, such methods are able to detect various forgeries in a single model. An early method proposed in  [ 122 ] relies only on MLPs and Stacked Autoencoders  [ 123 ] to detect anomalous patches, while methods in  [ 121 , 124 , 125 ] base their analysis on CNNs and make adaptions to them for better detection results. In particular, [ 121 , 124 ] redesign the convolutional layer to suppress image content and highlight manipulation features, while  [ 125 ] designs a CNN consisting of only 2 convolutional layers and no pooling layers to better distinguish the boundaries of forged regions.
Moreover, to overcome the limited receptive field of a CNN, an LSTM is appended to the convolutional layers in  [ 15 , 126 ] to further model the interdependencies of different blocks in 2D feature maps, and thus fosters the learning of boundary transformation between neighboring blocks. Concerning how image quality affects tampering detection,  [ 127 ] points out that resizing the image to fit it into the GPU memory ruins high-frequency details that are precious for media forensics, while studies made by  [ 128 , 31 ] bring forward methods making detectors robust against JPEG-compressed images  [ 128 ] and images transmitted via OSN (Online Social Network)  [ 31 ] .  [ 10 ] considers numerous manipulation types (up to 385) and trains the network to learn manipulation traces via self-supervised manipulation classification. It should be clarified that compared with the several tampering methods included in our survey (Sec.  3 ), the 385 manipulation types in  [ 10 ] are categorized according to what editing operation is used and its corresponding parameters.

 
 
 Fig. 5: Visualization of several handcrafted features for image tampering detection and localization. From left to right, we show the manipulated images, masks, noise, and high-frequency components, respectively. 
 
 
 By integrating features of different scales and modalities, universal forensic methods achieve more appreciable performances. In  [ 129 , 11 , 130 , 131 , 132 ] , the detectors are constructed to learn the relationships between image patches at different scales, which are produced by different layers of the CNN backbone  [ 130 , 131 ] , extracted from patches with different sizes  [ 129 ] , or derived from a pyramid structure  [ 11 , 132 ] . Rather than predict a single manipulation mask with fused multi-scale features,   [ 132 ] first derives a mask from the smallest scale and then progressively refines it to larger scales.
Meanwhile, multi-modal detectors  [ 9 , 133 , 126 , 19 ] tackle forgeries by combining features from multiple domains, e.g. , RGB domain, noise domain  [ 9 ] , and frequency domain  [ 133 , 126 , 19 ] , as is visualized in Figure  5 . In  [ 133 , 126 ] frequency features extracted by LSTM are fused with spatial features produced by convolutional layers, while  [ 19 ] combines RGB features with high-frequency features as multimodal patch embeddings, which are further refined by learnable object prototypes, and as a result, patch-level and object-level consistencies are captured in such cross-modal interactions.

 
 
 

#### II-C 3 Hybrid Detection Methods

 
 Even though DL-based forensic techniques can work well without engineering suitable rules for tampering detection, several studies look in a different direction, integrating deep learning with a priori knowledge and handcrafted rules of image tampering that is considered in Rule-based Detection Methods (Sec.  II-C1 ). Guided by extra knowledge and rules, hybrid detectors can not only pay more attention to tampering-related details but also are equipped with better interpretabilities since they usually give more specific predictions, which can be used for further analysis.

 
 
 DL-based Methods examining Intrinsic Patterns add conventional elements to the data-driven scenario in deep learning. They generally operate on handcrafted features as inputs rather than raw pixels and rely on the strong capability of deep neural networks to learn the decision boundary from these features. A good case in point is a median filtering forensic method proposed in  [ 134 ] , as it employs a filter layer to extract median filtering residuals from RGB images before the CNN model.

 
 
 As is mentioned in Sec.  II-C1 , double JPEG compression leaves distinct artifacts on the manipulated image and can be exploited for tampering detection. [ 135 , 136 , 137 , 138 ] uses deep learning methods to expose double compression artifacts by feeding the CNN with DCT coefficient histograms. [ 136 ] designs two encoders respectively for RGB patches and DCT coefficient histograms and concatenates these cross-modal features before proceeding to the fully connected network, which demonstrates superior performance to single-modal network under various JPEG quality factors. Furthermore, to prepare the detection model for real-world applications, [ 138 ] generates a dataset of mixed JPEG qualities and handles these varied qualities in one model, in which the vectorized quantization table is concatenated with DCT histogram features for enrichment.

 
 
 The second type operates on image pixels but is supervised with unique signals to expose certain artifacts. To detect demosaicing artifacts, an innovative training method named Positional Learning is proposed in  [ 139 , 30 ] , which consists in training a CNN to predict the modulo-2 position of image pixels. This self-supervised training method enables the CNN to be aware of the underlying periodic mosaic pattern, so a well-trained CNN’s misprediction indicates inconsistencies in the pattern and therefore reveals forgeries. In specific,   [ 139 ] trains the network on a database of pristine images, while  [ 30 ] further extends  [ 139 ] ’s work and shows it is feasible to fine-tune the network on a single test image, whether pristine or forged, to adapt the network’s weights to its domain. Even though the manipulated regions of a forged test image can mislead the network, these regions are generally small compared with the pristine regions, and as a result, the network can still benefit from the retraining process, which is also verified in  [ 30 ] ’s ablation studies.

 
 
 DL-based Methods based on Camera Consistency refer to the neural networks looking at disrupted camera consistency incurred by image tampering. An analogy can be drawn between these methods and those with statistical references in the rule-based section as both types first require training (or statistical estimation) to construct a camera-aware model and then make decisions based on learned camera information.

 
 
 Among these methods, [ 140 ] is the most similar one to PRNU-based methods as it estimates the “noiseprint” of a test image and the average one of some batched reference images using CNN, and computes the distance between two noiseprints to derive a heatmap for forgery localization. Besides, [ 141 ] performs CRF (Camera Response Function) analysis through a CNN on edge patches and intensity gradient histogram (IGH) that carries information of CRF, while  [ 142 , 143 , 144 , 145 , 146 ] extract patch-level features and look for anomalies by training the network with patches labeled with the camera ID, given that tampered patches are usually captured by a different camera model and spliced with the original photo. Specifically, [ 142 ] learns the similarity between pairs of patches to indicate whether these patches are from the same camera model, while  [ 143 ] further extends  [ 142 ] ’s work and scores the similarity of pairs of patches based on not only the camera source but also the editing operation and manipulation parameters. Furthermore, [ 144 ] proposes an improved method based on  [ 142 , 143 ] named forensic similarity graph to model the similarity between patches, in which patches are represented as vertices, whose edges are assigned based on their forensic similarity. Aside from the methods above, [ 147 ] trains a CNN in a self-supervised manner using real images and photo EXIF metadata to learn the correlation between an image and its camera model. To be specific, the CNN is fed two image sources and performs a binary classification on each of the EXIF entries to determine whether the two sources share the same EXIF information. Given a test image, the network indicates a spliced region if it discovers an image patch leading to inconsistent EXIF predictions.

 
 
 
 
 

## III Deepfake Detection 

 
 Face manipulation techniques  [ 6 , 148 , 149 , 150 ] , also known as Deepfakes, give rise to widespread concerns on the malicious tampering of human faces. This demands effective approaches that can detect photo-realistic forged faces generated by advanced Deepfake methods automatically. In this section, we first introduce the recent progress in Deepfake creation (Sec.  III-A ), and then summarize the existing public face forgery datasets (Sec.  III-B ), finally, we will focus on the detection methods to fight against face manipulation (Sec.  III-C ).

 
 

### III-A Deepfake Generation 

 
 Human face analysis is a well-studied field in computer vision, which relates to various applications. As a result, existing analysis methods lay the foundation for the development of face manipulation techniques  [ 151 , 150 , 152 ] . Specifically, based on the target and level of manipulation, Deepfake methods could be classified into four categories: face synthesis (Sec.  III-A1 ), face attribute editing (Sec.  III-A2 ), face swapping (Sec.  III-A3 ), and facial reenactment (Sec.  III-A4 ). We illustrate the process of them in Figure  6 briefly.

 
 
 Fig. 6: Illustration of four Deepfake synthesis techniques. 
 
 

#### III-A 1 Face Synthesis

 
 Built upon the powerful generative models like GANs  [ 1 ] , face synthesis methods  [ 153 , 154 , 155 , 156 ] create non-existent faces with natural appearances and even realistic textures. ProGAN  [ 157 ] proposes to train both the generator and discriminator progressively by gradually deepening the networks, which however still lacks the control over the style of the generated images. To address this issue, the StyleGAN family  [ 2 , 48 , 158 , 159 , 160 , 161 ] re-designs the generator architecture to enable the intuitive and scale-specific control of the synthesized results, inspired by the style transfer literature  [ 162 ] . In addition, a group of work has also explored the use of multimodal information or structural priors to guide face generation, including landmarks  [ 163 ] , sketch  [ 164 , 165 , 166 , 167 , 168 , 169 , 170 ] , and voice  [ 171 , 172 , 173 ] .

 
 
 Compared with static image synthesis, video generation  [ 174 ] is much more challenging due to the fact that both appearance learning and dynamic modeling are required to create visually natural videos of human faces. To address this issue, video generation methods  [ 174 , 175 ] typically learn the content and motion within videos in a decoupled manner, either with separate image and video generators  [ 176 , 177 , 178 , 179 ] , or by sampling from separate latent spaces  [ 180 , 181 ] .

 
 
 

#### III-A 2 Face Attribute Editing

 
 Instead of synthesizing the complete faces, face attribute editing  [ 182 , 151 , 183 , 184 ] aims to manipulate specific attributes, e.g. , gender and age, of human faces, which therefore can be seen as fine-grained face manipulation. Denoting face attributes as different “domains”, a series of methods  [ 3 , 185 , 186 ] define face attribute editing as a multi-domain image-to-image translation task. These approaches  [ 3 , 187 ] take the domain labels, e.g. , binary vectors, as inputs to translate the images into the target domain. However, [ 188 ] argues that utilizing the target attribution vector is redundant and may even lead to poor results; instead, they directly adopt the difference attribute vectors to provide more valuable guidance. Unlike the above methods which reply on explicit domain labels, other work manipulates the latent representations in 2D  [ 189 , 190 , 191 , 192 , 193 , 194 ] or 3D space  [ 195 , 196 ] to implicitly control the editing process, leveraging the entangled nature of the GAN latent space.

 
 
 In addition, the interactive face attribute manipulation has also attracted emerging attention due to its broad application prospects. With more user-friendly inputs, like texts  [ 197 , 47 , 198 , 199 ] , masks  [ 200 , 201 , 202 , 203 ] , sketches  [ 204 ] , and color-histograms  [ 205 ] , the semantic manipulation of facial images could be achieved more flexibly.

 
 
 

#### III-A 3 Face Swapping

 
 Face swapping  [ 206 , 207 , 208 , 209 , 210 ] , also known as face replacement or identity swapping, is another type of fine-grained face manipulation, which replaces the face of one person in an image with the face of another person. Typically, face swapping methods could be divided into two categories: classical computer graphics techniques  [ 206 , 207 , 211 , 212 , 213 , 214 , 215 ] and deep generative models  [ 216 , 217 , 151 , 218 ] , the latter of which has gained considerate attention recently for its simpler pipeline and better results. Early methods  [ 6 , 208 , 210 ] require training the model separately for each pair of subjects, which will result in expensive training cost and thus limiting their potential applications. To tackle this problem, several works have developed subject-agnostic methods  [ 150 , 219 ] that can perform facial identity swapping on arbitrary faces with one unified model by decomposing the identity information from the remaining attributes. In addition, developing efficient face swapping frameworks  [ 220 , 221 ] to enable the deployment on mobile devices, one-shot face swapping  [ 222 ] , and 3D-based facial replacement  [ 223 , 224 , 225 ] have also drawn emerging concern in the community.

 
 
 TABLE II: Statistics of existing datasets for Deepfake detection. 
 
 
 Generation | 
 Dataset | 
 Real | 
 Forged | 
 
 
 
 Generation 
 
 Approaches 
 | 

 
 Video | 
 Frame | 
 Video | 
 Frame | 

 
 First | 
 UADFV  [ 226 ] | 
 49 | 
 17.3k | 
 49 | 
 17.3k | 
 1 | 

 
 DF-TIMIT-LQ  [ 227 ] | 
 320 | 
 34.0k | 
 320 | 
 34.0k | 
 2 | 

 
 DF-TIMIT-HQ  [ 227 ] | 
 320 | 
 34.0k | 
 320 | 
 34.0k | 
 2 | 

 
 Second | 
 DFD  [ 228 ] | 
 363 | 
 315.4k | 
 3,068 | 
 2,242.7k | 
 5 | 

 
 Celeb-DF  [ 229 ] | 
 590 | 
 225.4k | 
 5,639 | 
 2,116.8k | 
 1 | 

 
 FF++  [ 230 ] | 
 1,000 | 
 509.9k | 
 4000 | 
 1,830.1k | 
 4 | 

 
 Third | 
 DFFD  [ 231 ] | 
 1,000 | 
 58,703 | 
 3,000 | 
 240,336 | 
 7 | 

 
 DeeperForensics-1.0  [ 232 ] | 
 1,000 | 
 - | 
 5,000 | 
 - | 
 5 | 

 
 DFDC  [ 233 ] | 
 23,564 | 
 - | 
 104,500 | 
 - | 
 8 | 

 
 ForgeryNet  [ 234 ] | 
 99,630 | 
 1,438.2k | 
 121,617 | 
 1,457.9k | 
 15 | 

 
 
 

#### III-A 4 Facial Reenactment

 
 Facial reenactment  [ 235 , 236 ] transfers the expression from the source person to the target person while preserving the identity information. Face2Face  [ 237 ] , one of the most prominent graphics-based expression manipulation methods  [ 235 , 236 , 238 , 239 , 240 ] , combines 3D model reconstruction and image-based rendering techniques for real-time face reenactment. Recently, the development of generative networks  [ 1 ] has stimulated the progress of deep learning-based techniques  [ 150 , 241 , 202 , 242 ] . Early methods  [ 243 , 3 , 149 , 152 , 244 , 245 ] adopt simple encoder-decoder architectures for facial reenactment, which however suffers from poor transfer performance and prominent visual artifacts since facial expressions are subtle. It is widely-known that landmarks are neat, sufficient, and robust to reflect the structure of human faces  [ 246 ] , inspired by this,   [ 247 , 248 , 249 , 250 , 251 , 252 ] apply driving landmarks to explicitly guide the reenactment process. Although decent synthesis results could be achieved with the above methods, training on a specific domain limits their capability to generate human faces with arbitrary expressions  [ 253 ] . In order to deal with this problem, a collection of work has explored the manipulation of expressions in a continuous space to achieve more natural reenactment results, with action units  [ 253 , 254 , 255 ] , expression codes  [ 256 ] , or motion field  [ 257 ] .

 
 
 Apart from manipulating the facial expression based on driving faces, talking head generation  [ 258 , 259 , 260 , 261 ] could be regarded as a special form of facial reenactment methods, which synthesizes a talking head video synchronized with a given audio clip. Resolving such a problem is highly challenging because it is non-trivial to faithfully relate the audio signals and face deformations, including expressions and lip motions  [ 262 ] . To mitigate this problem, most existing approaches achieve audio-head alignment by estimating the intermediate facial representations, e.g. , 3D face shapes  [ 263 , 264 , 265 , 266 ] , dynamic kernels  [ 267 ] , action units  [ 268 ] , dense motion fields  [ 269 ] , and landmarks  [ 258 , 270 , 271 , 272 ] . However, such multi-step transformation will inevitably lead to a loss of information and further affect the synthesis results. In contrast,   [ 262 ] directly map the extracted audio features to dynamic neural radiance fields  [ 273 , 274 ] for motion estimation, and finally synthesize the high-fidelity talking head using volume rendering.

 
 
 Fig. 7: Comparisons between the Deepfake datasets of different generations. From top to down, we separately display the examples from UADFV, CelebDF, and DFFD. 
 
 
 
 

### III-B Deepfake Datasets 

 
 The development and evaluation of face manipulation detection methods require large-scale datasets. In this section, we describe existing DeepFake datasets which are widely used to benchmark the effectiveness of different Deepfake detection methods. Based on the scale of forgery data and the generation techniques, these datasets could be divided into three generations.

 
 

#### III-B 1 First-generation Datasets

 
 The first-generation datasets are relatively small-scale datasets generated with open-source Deepfake creation tools, including UADFV  [ 226 ] and DeepFake-TIMIT  [ 227 ] . UADFV dataset  [ 226 ] contains 49 real videos collected from YouTube and 49 Deepfake videos that are generated using FakeAPP  [ 5 ] . DeepFake-TIMIT   [ 227 ] includes 320 real videos and 640 Deepfake videos (320 high-quality and 320 low-quality) generated with faceswap-GAN  [ 275 ] .

 
 
 

#### III-B 2 Second-generation Datasets

 
 The second-generation Deepfake datasets consist of DFD  [ 228 ] , Celeb-DF  [ 229 ] , and FF++  [ 230 ] , which are the most widely adopted datasets by far. The Google/Jigsaw DeepFake Detection dataset   [ 228 ] (DFD) includes 3,068 Deepfake videos that are generated based on 363 original videos. The Celeb-DF dataset  [ 229 ] contains 590 real videos and 5,639 Deepfake videos created using the same synthesis algorithm. The FaceForensics++ (FF++) dataset  [ 230 ] has 1,000 real videos from YouTube and 4,000 corresponding Deefake videos that are generated with 4 face manipulation methods: Deepfakes  [ 6 ] , FaceSwap  [ 215 ] , Face2Face  [ 148 ] , and NeuralTextures  [ 240 ] .

 
 
 

#### III-B 3 Third-generation Datasets

 
 The most recent datasets, including DFFD  [ 231 ] , DeeperForensics-1.0  [ 232 ] , DFDC  [ 233 ] , and ForgeryNet  [ 234 ] , are typically regarded as the third-generation datasets. Diverse Fake Face Dataset   [ 231 ] , also termed as DFFD, adopts the images from FFHQ  [ 2 ] and
CelebA  [ 276 ] datasets source subset, and synthesizes forged images with various Deepfake generation methods. DeeperForensics-1.0  [ 232 ] consists of 60,000 carefully-collected videos with a total of 17.6 million frames. It is worth mentioning that extensive real-world perturbations are applied on the generated images to further expand the scale and diversity. The Facebook DeepFake Detection Challenge dataset   [ 233 ] (DFDC) is part of the DeepFake detection challenge, which has 1,131 original videos and 4,113 Deepfake videos. The ForgeryNet   [ 234 ] is currently the largest publicly available Deepfake dataset, which contains 2.9 million images and 221,247 videos. The forged images in ForgeryNet  [ 234 ] are generated with 7 image-level approaches and 8 video-level approaches, from which four forgery identification tasks are derived: 1) image forgery classification; 2) spatial forgery localization; 3) video forgery classification; 4) temporal forgery localization.

 
 
 We summarize the statistics of these existing Deepfake datasets in Table  II . Examples from the Deepfake datasets of different generations are displayed in Figure  7 .

 
 
 
 

### III-C Deepfake Detection 

 
 With the prominent advances in Deepfake generation techniques, the synthesized images are becoming more photo-realistic, making it extremely difficult to distinguish fake faces even for the human eyes. At the same time, these forged images might be distributed on the Internet for malicious purposes,
which could bring societal implications.

 
 
 As a response to the increasing concern of the above challenges, various forensic methods  [ 277 , 278 , 279 ] are proposed, which typically take as inputs a face region cropped out of an entire image and produce a binary real/fake prediction. In the section below, we present a comprehensive discussion of existing Deepfake detection methods, where we group them into two major categories: image detection models (Sec.  III-C1 ) and video detection models (Sec.  III-C2 ).

 
 

#### III-C 1 Deepfake Image Detection

 
 Manipulating face images will inevitably leave some tampering clues, e.g. , handcrafted features, GAN fingerprints, and deep visual features, which, can be adopted by defense models for Deepfake image detection  [ 280 ] .

 
 
 Handcrafted features-based methods typically take a closer look at the inconsistencies produced by the synthesis process of Deepfakes for face forensics. Color space is widely adopted by forgery discrimination methods due to its robustness to various post-precessing, e.g. , resizing and compression. Specifically, based on the observation that Deepfake images are significantly different from real ones in the chrominance components of HSV and YCbCr color spaces, [ 281 ] introduces a color statistics-based feature set to identify the forged faces. [ 282 ] further extends this to Lab color space and concatenate the deep representations from different color spaces to obtain final detection results. Moreover, taking into consideration the consistent relationships among different color channels, [ 283 ] identifies the Deepfake images through the frequency of over-exposed pixels, while   [ 284 , 285 ] distinguishes GAN-generated images from camera-generated images by calculating cross-band co-occurrence matrices and spatial co-occurrence matrices.

 
 
 In addition, considering it is difficult for deep networks to replicate the high frequency details in real images, [ 286 ] combines the statistical, oriented gradient and blob features in the frequency domain for fake painting detection. Besides, the lack of global constraints may also lead to forged faces with unreasonable head poses, which inspires a group of work to distinguish GAN synthesized fake faces with detected landmarks  [ 287 ] or estimated 3D head models  [ 226 ] . Other than these, [ 288 ] detects the warping artifacts resulted from transforming the synthetic face regions to the pristine images, which achieves a competitive performance with a limited amount of fake data. [ 289 ] explores the use of photo response non uniformity (PRNU) analysis to identify Deepfakes. [ 290 ] derives the rich face representation using PixelHop++ units  [ 291 ] , and finally adopts the ensemble of different regions and frames for classification.   [ 292 ] detects the Deepfake videos using the state-of-the-art attribution based confidence (ABC) metric, which does not require the access to the training data.

 
 
 GAN fingerprints-based methods explore the marks that are commonly shared by GAN-generated images  [ 18 ] for forgery detection. Early methods adopt unsupervised machine learning techniques to discover the GAN-specific features. [ 293 ] first extracts a set of local features with an Expectation Maximization (EM) algorithm to model the convolutional traces, which are input to naive classifiers to discriminate between authentic and synthesized images. In order to improve the robustness towards various attacks on images such as JPEG Compression, mirroring, rotation, [ 294 ] transforms the suspicious images to frequency domain through Discrete Cosine Transform (DCT) and examines the statistics of the DCT coefficients for forgery detection. The above methods capture the GAN fingerprints at a single scale, which, therefore, cannot achieve competitive results on high-quality fake images. To mitigate this issue,   [ 295 ] employs a hierarchical Bayesian approach for multi-scale latent GAN fingerprint modeling.

 
 
 Recent work has attempted to employ deep neural networks (DNN)  [ 296 ] to identify the subtle artifacts left by GAN. [ 297 , 298 , 299 ] study the unique artifacts induced by the up-sampling of GAN pipelines to develop robust GAN image classifiers. [ 18 ] discovers that different training strategies will leave distinctive fingerprints over all generated images, which can be utilized to enable fine-grained image attribution and even model attribution. [ 296 ] explores the existence and properties of globally consistent GAN fingerprints through empirical study. Based on this, they develop a simple yet effective approach to capture GAN artifacts by pre-training on image transformation classification and patchwise contrastive learning.

 
 
 Deep features-based methods rely on deep models to mitigate the security threat brought by Deepfakes. Pioneering work  [ 280 , 300 ] typically captures artifacts from the face regions with stacked convolutional operations. More specifically, [ 277 ] detects tampered face images with a two-stream network, where a face classification stream captures the forgery artifacts and a patch triplet stream recognizes noise residual evidence. [ 301 , 302 ] improve the robustness of Deepfake detectors with Lipschitz regularization and model fusion, separately. In addition, considering binary classification tends to result in easily overfitted models, [ 279 , 303 , 304 ] propose to locate the manipulated regions to mitigate overfitting. Beyond these, the growing concerns towards Deepfakes have promoted the emergence of more advanced forgery detection approaches, as will be discussed in the following. We visualize the feature map of several Deepfake detection models with Grad-CAM  [ 305 ] in Figure  8 .

 
 
 
 
 
 (a) Deepfake 
 
 
 (b) Mask 
 
 
 (c) Xcep  [ 306 ] 
 
 
 (d) EN  [ 307 ] 
 
 
 (e) Gram  [ 308 ] 
 
 Fig. 8: Visualization of several deep features-based Deepfake detection models using Grad-CAM. Xcep denotes Xception  [ 306 ] , EN denotes EfficientNet  [ 307 ] , Gram denotes GramNet  [ 308 ] . 
 
 
 a) Contrastive-based approaches. Deepfake forensics is essentially a discriminative problem, i.e. , fitting the decision boundary on large-scale datasets. Therefore, focusing solely on the differences between binary categories will lead to limited generalization in unseen domains  [ 309 ] . With this in mind, a variety of methods  [ 310 , 311 , 312 , 313 ] seek to expand the discrepancies between authentic and fake images in feature space by means of contrastive learning. [ 310 ] learns joint discriminative features from heterogeneous training samples, i.e. , fake images generated with different GAN models, with a vanilla contrastive loss. In order to improve the robustness towards image compression algorithms, [ 314 ] encodes paired original and compressed forgeries to a compression-insensitive embedding feature space, in which the distance between genuine and forgery data are maximized while the distance between paired images are reduced through the supervision of a metric loss. The instance-level contrastive learning models the association between different images or the same image under different compression levels, which is limited to learn coarse-grained representations. Comparatively, [ 309 ] further introduces a intra-instance contrastive learning mechanism to mine deeper on the local inconsistencies within the forged faces.

 
 
 b) Attention-based approaches. Considering most Deepfake generation methods, e.g. , expression reenactment, only manipulate parts of the face images, identifying the suspect regions and “paying more attention to” their visual contents could effectively improve the detection results. Inspired by this, a series of attention-based methods have been proposed. [ 231 ] applies Principal Component Analysis (PCA) to several ground-truth manipulation masks to obtain the base attention maps, which are dynamically weighted and summed to the mean face with a manipulation appearance modeling module. [ 315 ] first segments the key fragments, e.g. , eyes and mouths, from a face image, and then looks into these regions with local attention modeling. The methods mentioned above rely on manually-designed areas of interest, which limits the representative capability of Deepfake detectors. Comparatively, [ 13 ] combines multiple spatial attention heads to automatically attend to different local parts with trainable convolution layers. Based on the observation that the reconstruction difference of real and fake faces follows significantly different distributions, [ 316 ] uses a reconstruction-guided attention module to enhance the features derived from the multi-scale graph reasoning. In addition, [ 317 ] presents an attention-based data augmentation framework to encourage the detector to attend to subtle forgery traces. Specifically, they first generate a Forgery Attention Map (FAM) by calculating the gradients with respect to the classification loss, and then occlude the Top-N
sensitive facial regions intentionally to facilitate the detector to mine deeper into the regions ignored before.

 
 
 c) Local relation-based approaches. Most existing Deepfake detection methods formulate face forgery detection as a classification problem with global supervisions, e.g. , binary labels or manipulated masks, which are insufficient to learn discriminative features and prone to overfitting  [ 21 ] without explicit local relation modeling. To tackle this issue, [ 308 ] enhances texture features with stacked Gram blocks. [ 20 ] assumes a forged image would contain different source features at different positions, therefore, the forged images could be identified by measuring their self-consistency. Similarly, [ 21 ] models the relations of local patches by calculating the similarity between features of different regions.

 
 
 d) Frequency-based approaches. The prominent advances in Deepfake techniques have made the visual artifacts in fake images more and more subtle and imperceptible. In addition, the common image processing algorithms, e.g. , compression, will further contaminate the forgery clues in RGB domain. Fortunately, several prior studies suggest that these artifacts can be captured in frequency domain  [ 318 , 319 , 320 , 18 ] , in the form of unusual frequency distributions when compared with real faces  [ 321 ] . This motivates a sequence of work  [ 322 , 323 , 324 , 325 , 326 ] to use frequency information as a complementary modality to develop stronger and more robust Deepfake detection methods. [ 321 ] uses a two-stream collaborative framework by combing a Frequency-aware Decomposition stream to discover salient frequency bands and a Local Frequency Statistics stream to extract local frequency statistics. [ 326 ] notices that the artifacts have large magnitudes in the high-frequency components and are oftentimes located in the surrounding background of the image rather than the central region, therefore, they combine the frequency-level High-Pass Filters (HPF) for amplifying the magnitudes of the artifacts in the high-frequency components, and the pixel-level HPF for emphasizing the pixel values in the surrounding background in the pixel domain. [ 325 ] distills a teacher model pretrained on high-quality images with a novel frequency attention distillation to detect low-quality deepfakes. The aforementioned methods generally make use of the frequency information from a global perspective, [ 324 ] instead employs the patch-wise Discrete Fourier Transform  [ 327 ] with the sliding window to decompose the RGB input into fine-grained frequency components, considering the forgery clues are oftentimes hidden in the local patches.

 
 
 The aforementioned approaches detect forgery artifacts with convolutional layers, which fail to model the consistency of pixels globally due to the limited size of receptive fields. To tackle this issue, [ 328 ] adopts the Transformer architecture  [ 46 ] to capture long-term dependencies between different face regions, and extract multi-scale tampering features by splitting the feature maps into patches of different sizes. Following  [ 328 ] , [ 329 ] further selects multi-scale salient patches based on attention weights to capture sensitive information.   [ 330 ] distills the knowledge from EfficientNet  [ 307 ] , the state-of-the-art model on the DFDC dataset, to fully boost the performance of vision transformer for Deepfake detection.

 
 
 Fig. 9: Illustration of Spatial-Temporal forgery clues in Deepfake videos. Both visual artifacts (spatially) and biological behaviors (temporally, e.g. , abnormal blinking rate) can be exploited to detect Deepfake videos. 
 
 
 

#### III-C 2 Deepfake Video Detection

 
 Nowadays, forged faces that are circulating on the Internet take the form of videos, since videos can be more persuasive. Although image-level tampering detection methods can be directly applied to videos, they lack the capability to model temporal dynamics in videos. We visualize the spatial-temporal artifacts in Deepfake videos in Figure  9 . This has motivated a line of work on video-level tampering detection  [ 331 , 332 ] . According to the type of features of interest, we further classify the Deepfake video detection methods into biological behaviors-based approaches and spatial-temporal visual clues based approaches.

 
 
 Biological behaviors. 

 
 Although deep generative models can synthesize realistic faces, they oftentimes fail to reproduce some biological behaviors intrinsic to human beings. This inspires lots of work  [ 333 ] to detect forged videos through the analysis of biological behaviors. Eye blinking is a widely used clue by forensics methods  [ 334 , 335 ] since it is a voluntary and spontaneous action that does not require conscious effort for humans  [ 335 ] . Specifically, [ 334 ] discovers the absence of faces with eye closed in the training datasets results in the lack of eye blinking in forged videos. To this end, they combine a convolutional neural network with a recursive neural network (RNN) to capture the phenomenological and temporal regularities of eye blinking for forgery detection. [ 335 ] analyzes the pattern of eye blinking in more detail, i.e. , the period, repeated number, and elapsed eye blink time when eye blinks were continuously repeated within a very short period of time.

 
 
 Some other appearance cues like lip movements  [ 336 , 337 , 338 , 339 ] and head pose  [ 226 , 340 ] have also gained widespread concern for Deepfake video detection. [ 336 ] focuses on the visemes associated with words like M (mama), B (baba), or P (papa) in which the mouth must completely close to pronounce these phonemes but usually not the case in fake videos. [ 339 ] first pretrains a Visual Speech Recognition (VSR) network to learn the internal consistency between the shape of lips and spoken words, and then fine-tunes a temporal network on fixed mouth embeddings for video forgery detection. [ 340 ] proposes a customized forensic method for specific individuals by modeling their distinct patterns of facial and head movements. [ 226 ] observes that most Deepfake synthesis methods are incapable of matching the location of facial landmarks reasonably, which are invisible to human eyes but can be revealed from 3D head poses estimation. Inspired by this, they train a simple SVM classifier on the estimated head poses to differentiate original and fake videos.

 
 
 Another interesting point is that several recent studies  [ 333 , 341 , 342 ] present the heart rate estimated from visual contents can be utilized for face video forensics. [ 341 ] , for instance, predicts the heart rate with a state-of-the-art Neural Ordinary Differential Equations (Neural-ODE) model, which could further facilitate the Deepfake detection by identifying abnormal heartbeat rhythms. Additionally, it is recognized that heartbeat will change the skin reflectance over time due to the hemoglobin content in the blood, which however are oftentimes disrupted or even entirely broken in forged videos. This inspires a line work  [ 343 , 342 , 344 ] to make use of photoplethysmography (PPG) to identify Deepfake videos. Specifically, [ 344 ] separately input video frames and differences between adjacent frames to an appearance model and a motion model for Deepfake detection, which initializes the weights from an existing heart rate estimation framework to transfer its knowledge. [ 342 ] , from another point of view, incorporates both a heart rhythm motion amplification module and a spatial-temporal
attention module to expose subtle manipulation artifacts.

 
 
 
 Deep features. 

 
 The prominent success of deep learning makes it attractive to detect forged videos using end-to-end trainable deep networks  [ 278 , 280 , 279 , 345 ] . Although current Deepfake techniques achieve impressive performance regarding quality and controllability, they typically manipulate images at frame-level, thus struggling to preserve temporal coherence  [ 346 ] . To this end, a series of work  [ 346 , 347 ] focus on uncovering the temporal discordance to identify forged videos.   [ 347 ] learns a set of prototypes to represent specific activation patterns in a patch of the convolutional feature maps, which further interact with a test video to makes predictions based on their similarities. [ 348 ] presents a motion feature-based video forensic method, which calculates the correlation matrices from local motion features to represent the temporal smoothness of the whole video. [ 346 ] notices that some discontinuity in fake videos may happen in frames that are not in the neighborhood, therefore, they adopt a light-weight temporal Transformer to explore the long-term temporal coherence. [ 349 ] solves the Deepfake detection problem from a different perspective, which predicts the facial representations of future frames using an autoregressive model and distinguishes the fake videos according to the estimation error. The sparse sampling strategy is shared by most existing methods, which however disables them from modeling the local motions among adjacent frames. To counter this issue,   [ 350 ] proposes a novel sampling unit named snippet which contains a few successive videos frames for local temporal inconsistency learning.

 
 
 Despite the favorable results achieved, ignoring spatial cues still limits the performance and scalability of the above temporal-focused approaches. Comparatively, a variety of work utilizes both spatial and temporal information via 2D/3D CNN  [ 351 ] or Conv-RNN architectures  [ 352 , 16 ] . Among them, a considerable amount of work  [ 353 , 354 , 355 ] focuses on capturing the spatial-temporal inconsistency of suspicious videos. Specifically,   [ 355 , 356 ] adopt a two-branch network to separately attend to the intra-frame and inter-frame inconsistencies.   [ 357 ] presents an identity-driven face forgery detection approaches, which requires the access to both a suspect image and a reference image indicating the target identity information, and outputs a decision on whether the two faces share the same identities.

 
 
 
 Multimodal information. 

 
 Videos are embedded with multimodal information, i.e. , motion (optical flow), and audio by nature, which inspires several works to explore multimodal inputs  [ 358 , 359 , 359 ] or cross-modal alignment  [ 360 , 361 , 362 ] for Deepfake forensics. [ 358 ] simply calculates the optical flow fields in an off-line manner, which are further input a CNN classifier to discover the inter-frame dissimilarities. In addition, the audio-visual synchronization is an intrinsic property of pristine videos, therefore, manipulation of either modality will result in non-negligible dissonance, i.e. , loss of lip-sync, unnatural facial and lip movements, et al.   [ 361 ] . Based on this,   [ 360 ] learns discriminative representations for the audio and visual inputs in a chunk-wise manner, employing the cross-entropy loss for individual modalities, and a contrastive loss that models inter-modality similarity. [ 363 ] adopts a Siamese network-based architecture to separately extract the features of video and audio modalities, and constrain the consistency learning by a triplet loss function.   [ 364 ] first learns the “temporally dense” video features, e.g. , facial movements, expression, and identity, in a self-supervised cross-modal manner, and then fine-tunes a binary forgery classifier on the learned representations.

 
 
 
 

#### III-C 3 Generalizable Deepfake Detection

 
 The Deepfake generation techniques are evolving rapidly and continuously, therefore, the development of face forensic methods with distinguished generalization capabilities has gained sustained attention  [ 365 , 366 , 367 , 368 , 369 , 370 ] . However, the data-driven property makes most face forgery detection methods struggle to achieve competitive results in cross-database evaluations  [ 368 ] . In order to counter this challenge,   [ 12 ] captures the visual discrepancies across the blending boundaries for face tampering detection, based on the observation that most Deepfake methods blend the manipulated face regions with background images.   [ 371 ] proposes an incremental learning method, aiming to classify Deepfakes generated by novel models without worsening the performance on the previous ones.   [ 372 ] , further formulates the Deepfake detection as a one-class anomaly detection problem, which trains a one-class Variational Autoencoder (VAE) only on real face images and detects fake images by treating them as anomalies.

 
 
 Defining the forged images generated by diversified Deepfake techniques as different domains, a series of face forgery forensic methods  [ 373 , 374 , 367 , 374 , 375 ] build upon domain adaptation for Deepfake detection by transferring the knowledge from the source (teacher) to target (student) domains.   [ 367 ] divides the source domain into training domain and meta-domain to simulate the domain shift, and updates the model weights using a meta-optimization strategy.   [ 373 ] proposes a weakly-supervised setting, where only a handful
of training samples from target domain are available. They disentangle the decision space combining an activation loss and a reconstruction loss, which facilitate the detector to learn discriminative and complete representations respectively.

 
 
 In order to encourage the Deep forensic models to learn the intrinsic features to classify the generated and real face images, several studies propose to apply preprocessing on the training data, e.g. , Gaussian blur and Gaussian noise  [ 320 ] and JPEG compression  [ 320 ] , to eliminate noisy low level cues. Along the same lines,   [ 369 ] further makes use of the adversarial training strategy to dynamically synthesize both challenging and diversified forgeries.

 
 
 Fig. 10: Illustration of the diversified forgery patterns in different tampering (top) and Deepfake (bottom) datasets. 
 
 
 
 
 

## IV Challenges and Future Directions 

 
 Although recent years have witnessed significant progress in media forensics, there still exists a number of open problems that are worth exploring. It is worth mentioning that the common tug-of-war nature between generation methods and forensic methods makes both tampering detection and Deepfake detection face many shared problems. In this section, we summarize the current challenges together with possible future directions in this field.

 
 
 Generalization : the blooming media manipulation techniques could generate more and more realistic and diversified forged data (see Figure  10 ), which requires the media forensic algorithms to develop favorable generalization ability, i.e. , a model trained on a specific forgery methods should work well against another unknown one because potential manipulation types are typically unknown in the real-world scenarios.

 
 
 To mitigate this issue,   [ 57 , 59 ] collect a large quantity of forged images in the wild to construct a large-scale tampering datasets, which generally go through numerous manual manipulations to foster the generalization of data-driven detection methods. On the other side, several Deepfake detection methods  [ 373 , 375 ] have also explored to alleviate this issue from the perspective of model design and training strategy (Sec.  III-C3 ), but more work needs to be done in both tampering detection and Deepfake detection.

 
 
 Robustness : multimedia data, e.g. , images and videos circulating on the Internet are oftentimes processed, e.g. , compression, resizing, and Gaussian blurring, during upload, transmission and download, which however would inevitably destroy the details in pristine data and thus bringing more severe challenges to forgery detection. In addition, malicious users even deliberately impose invisible perturbations on the forged data to fool the detection tools.

 
 
 However, forensic models trained on high-quality images oftentimes suffer from severe performance degradation on post-processed images. For instance, we visualize the Grad-CAM on the raw fake images and the corresponding compressed images using the model pretrained on raw images in Figure  11 . The results demonstrate that although compression algorithms will not lead to a significant visual difference, the detector still fails to work. From a defensive standpoint, nevertheless, a “good” manipulation detector should be robust towards the common distortion algorithms. A good case in point is  [ 31 ] , as it thoroughly analyzes the impact of Online Social Networks on tampered images and offers a solution. Since there is generally no standard of handling images in web applications for detectors, detectors themselves must be robust enough against these various scenarios.

 
 
 Fig. 11: Visualization of the Grad-CAM on different subsets of FF++. From left to right, we display the results on raw and c40 (compressed with a rate factor of 40) images using the model pretrained on the raw subset of FF++. 
 
 
 Model Attribution : most existing forensics methods typically focus on binary classification i.e. , real/fake. Nevertheless, attributing the manipulated images/videos to the model generating them is also of prominent significance for legal accountability and intellectual
property protection purposes  [ 376 ] . Although several approaches  [ 377 , 296 ] have explored to perform attribution on Deepfake images for multiple GAN architectures, the problem of model attribution for both tampering and Deepfake detection is far from studied and solved sufficiently.

 
 
 Multimodal information : with the advancement of tampering technology, it is difficult to effectively identify the manipulated data with the single RGB modality. Comparatively, different modalities could be leveraged to collaboratively detect the subtle forgery artifacts. Indeed, there has been a range of work  [ 19 , 321 , 21 ] exploring the integration of multimodal information for media forensics. For tampering detection, several methods reveal that the plain RGB features can be well enhanced by exploiting high-frequency features containing rich manipulation details, as for Deepfake detection, a celebrity video that spread fake news could be easily detected through a comprehensive analysis of visual artifacts and lip-text/lip-audio mis-synchrony. The discovery of more representative modalities and the more effective integration of different modality information are both promising directions for further improvements in forgery detection performance.

 
 
 Interpretability : the lack of interpretability of deep neural networks has always been one of their deficiencies despite their powerful discriminative abilities, making it hard for decisions from neural networks to be accepted on serious occasions such as a court. As the current trend of Deepfake and tampering detection methods is still based on deep learning, more efforts need to be made to add more interpretabilities and trustworthiness to these methods. Comparatively, rule-based tampering detection methods, especially the intrinsic pattern analysis, introduced in Sec.  II-C1 have better interpretabilities since they rely on concrete rules rather than data or complex models full of parameters.
As hybrid methods prove feasible (See Sec.  II-C3 ), the future of a truly reliable forensic tool for tampering detection may constantly revolve around rule-based methods and deep learning.

 
 
 

## V Conclusion 

 
 This paper presents a comprehensive survey of deep learning-based media forensic methods. Depending on the content of suspicious media data, we divide media manipulation detection into tampering detection and Deepfake detection, which share a wide variety of commonalities in the capture of forgery traces. Public datasets used to benchmark different detection methods and a multitude of forgery detection techniques are presented in detail. Finally, we also discuss current challenges in tampering detection, and provide some insights into future research.

 
 
 

## References

 
 
 [1] 
 
I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair,
A. Courville, and Y. Bengio, “Generative adversarial nets,” NeurIPS ,
2014.

 

 
 [2] 
 
T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for
generative adversarial networks,” in CVPR , 2019.

 

 
 [3] 
 
Y. Choi, M. Choi, M. Kim, J.-W. Ha, S. Kim, and J. Choo, “Stargan: Unified
generative adversarial networks for multi-domain image-to-image
translation,” in CVPR , 2018.

 

 
 [4] 
 
L. Cleaner, https://github.com/Sanster/lama-cleaner , 2021.

 

 
 [5] 
 
Fakeapp, https://www.fakeapp.com/ , 2018.

 

 
 [6] 
 
Deepfakes, https://github.com/deepfakes/faceswap , 2018.

 

 
 [7] 
 
M. Barni, A. Costanzo, and L. Sabatini, “Identification of cut and paste
tampering by means of double-jpeg detection and image segmentation,” in
 ISCAS , 2010.

 

 
 [8] 
 
A. C. Popescu and H. Farid, “Exposing digital forgeries by detecting traces of
resampling,” ITSP , 2005.

 

 
 [9] 
 
P. Zhou, X. Han, V. I. Morariu, and L. S. Davis, “Learning rich features for
image manipulation detection,” in CVPR , 2018.

 

 
 [10] 
 
Y. Wu, W. AbdAlmageed, and P. Natarajan, “Mantra-net: Manipulation tracing
network for detection and localization of image forgeries with anomalous
features,” in CVPR , 2019.

 

 
 [11] 
 
X. Hu, Z. Zhang, Z. Jiang, S. Chaudhuri, Z. Yang, and R. Nevatia, “Span:
Spatial pyramid attention network for image manipulation localization,” in
 ECCV , 2020.

 

 
 [12] 
 
L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen, and B. Guo, “Face x-ray for
more general face forgery detection,” in CVPR , 2020.

 

 
 [13] 
 
H. Zhao, W. Zhou, D. Chen, T. Wei, W. Zhang, and N. Yu, “Multi-attentional
deepfake detection,” in CVPR , 2021.

 

 
 [14] 
 
P. Ferrara, T. Bianchi, A. De Rosa, and A. Piva, “Image forgery localization
via fine-grained analysis of cfa artifacts,” TIFS , 2012.

 

 
 [15] 
 
J. H. Bappy, A. K. Roy-Chowdhury, J. Bunk, L. Nataraj, and B. Manjunath,
“Exploiting spatial structure for localizing manipulated image regions,” in
 ICCV , 2017.

 

 
 [16] 
 
I. Masi, A. Killekar, R. M. Mascarenhas, S. P. Gurudatt, and W. AbdAlmageed,
“Two-branch recurrent network for isolating deepfakes in videos,” in
 ECCV , 2020.

 

 
 [17] 
 
X. Wu, Z. Xie, Y. Gao, and Y. Xiao, “Sstnet: Detecting manipulated faces
through spatial, steganalysis and temporal features,” in ICASSP ,
2020.

 

 
 [18] 
 
N. Yu, L. S. Davis, and M. Fritz, “Attributing fake images to gans: Learning
and analyzing gan fingerprints,” in ICCV , 2019.

 

 
 [19] 
 
J. Wang, Z. Wu, J. Chen, X. Han, A. Shrivastava, S.-N. Lim, and Y.-G. Jiang,
“Objectformer for image manipulation detection and localization,” in
 CVPR , 2022.

 

 
 [20] 
 
T. Zhao, X. Xu, M. Xu, H. Ding, Y. Xiong, and W. Xia, “Learning
self-consistency for deepfake detection,” in ICCV , 2021.

 

 
 [21] 
 
S. Chen, T. Yao, Y. Chen, S. Ding, J. Li, and R. Ji, “Local relation learning
for face forgery detection,” in AAAI , 2021.

 

 
 [22] 
 
L. Verdoliva, “Media forensics and deepfakes: an overview,” JSTSP ,
2020.

 

 
 [23] 
 
G. K. Birajdar and V. H. Mankar, “Digital image forgery detection using
passive techniques: A survey,” Digital investigation , 2013.

 

 
 [24] 
 
Y. Mirsky and W. Lee, “The creation and detection of deepfakes: A survey,”
 CSUR , 2021.

 

 
 [25] 
 
T. T. Nguyen, Q. V. H. Nguyen, D. T. Nguyen, D. T. Nguyen, T. Huynh-The,
S. Nahavandi, T. T. Nguyen, Q.-V. Pham, and C. M. Nguyen, “Deep learning for
deepfakes creation and detection: A survey,” CVIU , 2022.

 

 
 [26] 
 
A. Malik, M. Kuribayashi, S. M. Abdullahi, and A. N. Khan, “Deepfake detection
for human face images and videos: a survey,” IEEE Access , 2022.

 

 
 [27] 
 
J. Wang, G. Lu, H. Xu, Z. Li, C. Xu, and Y. Fu, “Manitrans: Entity-level
text-guided image manipulation via token-wise semantic alignment and
generation,” in CVPR , 2022.

 

 
 [28] 
 
S. H. Lee, W. Roh, W. Byeon, S. H. Yoon, C. Kim, J. Kim, and S. Kim,
“Sound-guided semantic image manipulation,” in CVPR , 2022.

 

 
 [29] 
 
G. Kim, T. Kwon, and J. C. Ye, “Diffusionclip: Text-guided diffusion models
for robust image manipulation,” in CVPR , 2022.

 

 
 [30] 
 
Q. Bammey, R. G. von Gioi, and J.-M. Morel, “Forgery detection by internal
positional learning of demosaicing traces,” in WACV , 2022.

 

 
 [31] 
 
H. Wu, J. Zhou, J. Tian, and J. Liu, “Robust image forgery detection over
online social network shared images,” in CVPR , 2022.

 

 
 [32] 
 
M. Bertalmio, G. Sapiro, V. Caselles, and C. Ballester, “Image inpainting,”
in PACMCGIT , 2000.

 

 
 [33] 
 
A. Criminisi, P. Pérez, and K. Toyama, “Region filling and object removal
by exemplar-based image inpainting,” TIP , 2004.

 

 
 [34] 
 
A. Criminisi, P. Perez, and K. Toyama, “Object removal by exemplar-based
inpainting,” in CVPR , 2003.

 

 
 [35] 
 
C. Barnes, E. Shechtman, A. Finkelstein, and D. B. Goldman, “Patchmatch: A
randomized correspondence algorithm for structural image editing,”
 TOG , 2009.

 

 
 [36] 
 
J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,”
 NeurIPS , 2020.

 

 
 [37] 
 
P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,”
 NeurIPS , 2021.

 

 
 [38] 
 
M. Mirza and S. Osindero, “Conditional generative adversarial nets,”
 arXiv preprint arXiv:1411.1784 , 2014.

 

 
 [39] 
 
C.-H. Lin, E. Yumer, O. Wang, E. Shechtman, and S. Lucey, “St-gan: Spatial
transformer generative adversarial networks for image compositing,” in
 CVPR , 2018.

 

 
 [40] 
 
F. Zhan, H. Zhu, and S. Lu, “Spatial fusion gan for image synthesis,” in
 CVPR , 2019.

 

 
 [41] 
 
X. Li, G. Teng, P. An, and H.-Y. Yao, “Image synthesis via adversarial
geometric consistency pursuit,” SP , 2021.

 

 
 [42] 
 
S. Azadi, D. Pathak, S. Ebrahimi, and T. Darrell, “Compositional gan: Learning
image-conditional binary composition,” IJCV , 2020.

 

 
 [43] 
 
S. Hong, X. Yan, T. S. Huang, and H. Lee, “Learning hierarchical semantic
image manipulation through structured representations,” NeurIPS ,
2018.

 

 
 [44] 
 
X. Tan, P. Xu, S. Guo, and W. Wang, “Image composition of partially occluded
objects,” in CGF , 2019.

 

 
 [45] 
 
B. Li, X. Qi, T. Lukasiewicz, and P. H. Torr, “Manigan: Text-guided image
manipulation,” in CVPR , 2020.

 

 
 [46] 
 
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in
 NeurIPS , 2017.

 

 
 [47] 
 
O. Patashnik, Z. Wu, E. Shechtman, D. Cohen-Or, and D. Lischinski, “Styleclip:
Text-driven manipulation of stylegan imagery,” in ICCV , 2021.

 

 
 [48] 
 
T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila,
“Analyzing and improving the image quality of stylegan,” in CVPR ,
2020.

 

 
 [49] 
 
H. Dhamo, A. Farshad, I. Laina, N. Navab, G. D. Hager, F. Tombari, and
C. Rupprecht, “Semantic image manipulation using scene graphs,” in
 CVPR , 2020.

 

 
 [50] 
 
R. R. Shetty, M. Fritz, and B. Schiele, “Adversarial scene editing: Automatic
object removal from weak supervision,” NeurIPS , 2018.

 

 
 [51] 
 
T.-T. Ng, S.-F. Chang, and Q. Sun, “A data set of authentic and spliced image
blocks,” Columbia University, ADVENT Technical Report , 2004.

 

 
 [52] 
 
Y.-F. Hsu and S.-F. Chang, “Detecting image splicing using geometry invariants
and camera characteristics consistency,” in ICME , 2006.

 

 
 [53] 
 
I. Amerini, L. Ballan, R. Caldelli, A. Del Bimbo, and G. Serra, “A sift-based
forensic method for copy–move attack detection and transformation
recovery,” TIFS , 2011.

 

 
 [54] 
 
T. Bianchi and A. Piva, “Image forgery localization via block-grained analysis
of jpeg artifacts,” TIFS , 2012.

 

 
 [55] 
 
D. Tralic, I. Zupancic, S. Grgic, and M. Grgic, “Comofod—new database for
copy-move forgery detection,” in ELMAR , 2013.

 

 
 [56] 
 
J. Dong, W. Wang, and T. Tan, “Casia image tampering detection evaluation
database,” in ICSIP , 2013.

 

 
 [57] 
 
M. Zampoglou, S. Papadopoulos, and Y. Kompatsiaris, “Detecting image splicing
in the wild (web),” in ICMEW , 2015.

 

 
 [58] 
 
H. Guan, M. Kozak, E. Robertson, Y. Lee, A. N. Yates, A. Delgado, D. Zhou,
T. Kheyrkhah, J. Smith, and J. Fiscus, “Mfc datasets: Large-scale benchmark
datasets for media forensic challenge evaluation,” in WACVW , 2019.

 

 
 [59] 
 
S. Heller, L. Rossetto, and H. Schuldt, “The ps-battles dataset-an image
collection for image manipulation detection,” arXiv preprint
arXiv:1804.04866 , 2018.

 

 
 [60] 
 
G. Mahfoudi, B. Tajini, F. Retraint, F. Morain-Nicolier, J. L. Dugelay, and
P. Marc, “Defacto: Image and face manipulation dataset,” in EUSIPCO ,
2019.

 

 
 [61] 
 
A. Novozamsky, B. Mahdian, and S. Saic, “Imd2020: A large-scale annotated
dataset tailored for detecting manipulated images,” in WACV , 2020.

 

 
 [62] 
 
V. Christlein, C. Riess, J. Jordan, C. Riess, and E. Angelopoulou, “An
evaluation of popular copy-move forgery detection approaches,” TIFS ,
2012.

 

 
 [63] 
 
T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,
P. Dollár, and C. L. Zitnick, “Microsoft coco: Common objects in
context,” in ECCV , 2014.

 

 
 [64] 
 
T. J. De Carvalho, C. Riess, E. Angelopoulou, H. Pedrini, and
A. de Rezende Rocha, “Exposing digital image forgeries by illumination color
classification,” TIFS , 2013.

 

 
 [65] 
 
T. Gloe, K. Borowka, and A. Winkler, “Efficient estimation and large-scale
evaluation of lateral chromatic aberration for digital image forensics,” in
 MFS , 2010.

 

 
 [66] 
 
O. Mayer and M. C. Stamm, “Accurate and efficient image forgery detection
using lateral chromatic aberration,” TIFS , 2018.

 

 
 [67] 
 
A. C. Popescu and H. Farid, “Exposing digital forgeries in color filter array
interpolated images,” ITSP , 2005.

 

 
 [68] 
 
H. Cao and A. C. Kot, “Accurate detection of demosaicing regularity for
digital image forensics,” TIFS , 2009.

 

 
 [69] 
 
J. S. Ho, O. C. Au, J. Zhou, and Y. Guo, “Inter-channel demosaicking traces
for digital image forensics,” in ICME , 2010.

 

 
 [70] 
 
B. Liu, C.-M. Pun, and X.-C. Yuan, “Digital image forgery detection using jpeg
features and local noise discrepancies,” Sci. World J. , 2014.

 

 
 [71] 
 
D. Cozzolino, G. Poggi, and L. Verdoliva, “Splicebuster: A new blind image
splicing detector,” in WIFS , 2015.

 

 
 [72] 
 
A. Swaminathan, M. Wu, and K. R. Liu, “Digital image forensics via intrinsic
fingerprints,” TIFS , 2008.

 

 
 [73] 
 
A. C. Popescu and H. Farid, “Statistical tools for digital forensics,” in
 International Workshop on Information Hiding , 2004.

 

 
 [74] 
 
Y.-L. Chen and C.-T. Hsu, “Detecting recompression of jpeg images via
periodicity analysis of compression artifacts for tampering detection,”
 TIFS , 2011.

 

 
 [75] 
 
S. Agarwal and H. Farid, “Photo forensics from jpeg dimples,” in WIFS ,
2017.

 

 
 [76] 
 
C. Pasquini, G. Boato, and F. Pérez-González, “Statistical detection
of jpeg traces in digital images in uncompressed formats,” TIFS ,
2017.

 

 
 [77] 
 
H. Farid, “Exposing digital forgeries from jpeg ghosts,” TIFS , 2009.

 

 
 [78] 
 
C. Iakovidou, M. Zampoglou, S. Papadopoulos, and Y. Kompatsiaris,
“Content-aware detection of jpeg grid inconsistencies for intuitive image
forensics,” VC , 2018.

 

 
 [79] 
 
B. Lorch and C. Riess, “Image forensics from chroma subsampling of
high-quality jpeg images,” in MMSec , 2019.

 

 
 [80] 
 
J. Dong, W. Wang, T. Tan, and Y. Q. Shi, “Run-length and edge statistics based
approach for image splicing detection,” in IWDW , 2008.

 

 
 [81] 
 
K. Bahrami, A. C. Kot, L. Li, and H. Li, “Blurred image splicing localization
by exposing blur type inconsistency,” TIFS , 2015.

 

 
 [82] 
 
A. J. Fridrich, B. D. Soukal, and A. J. Luk’aš, “Detection of
copy-move forgery in digital images,” in DFRWS , 2003.

 

 
 [83] 
 
S. Bayram, H. T. Sencar, and N. Memon, “An efficient and robust method for
detecting copy-move forgery,” in ICASSP , 2009.

 

 
 [84] 
 
E. Ardizzone, A. Bruno, and G. Mazzola, “Copy-move forgery detection via
texture description,” in ACMW , 2010.

 

 
 [85] 
 
I. Amerini, L. Ballan, R. Caldelli, A. Del Bimbo, L. Del Tongo, and G. Serra,
“Copy-move forgery detection and localization by means of robust clustering
with j-linkage,” SP , 2013.

 

 
 [86] 
 
S.-J. Ryu, M. Kirchner, M.-J. Lee, and H.-K. Lee, “Rotation invariant
localization of duplicated image regions based on zernike moments,”
 TIFS , 2013.

 

 
 [87] 
 
J. Li, X. Li, B. Yang, and X. Sun, “Segmentation-based image copy-move forgery
detection scheme,” TIFS , 2014.

 

 
 [88] 
 
E. Silva, T. Carvalho, A. Ferreira, and A. Rocha, “Going deeper into copy-move
forgery detection: Exploring image telltales via multi-scale analysis and
voting processes,” VC , 2015.

 

 
 [89] 
 
E. Ardizzone, A. Bruno, and G. Mazzola, “Copy–move forgery detection by
matching triangles of keypoints,” TIFS , 2015.

 

 
 [90] 
 
Y. Li and J. Zhou, “Fast and effective image copy-move forgery detection via
hierarchical feature point matching,” TIFS , 2018.

 

 
 [91] 
 
J. Lukáš, J. Fridrich, and M. Goljan, “Detecting digital image
forgeries using sensor pattern noise,” in SPIE , 2006.

 

 
 [92] 
 
M. Chen, J. Fridrich, J. Lukáš, and M. Goljan, “Imaging sensor noise
as digital x-ray for revealing forgeries,” in International Workshop
on Information Hiding , 2007.

 

 
 [93] 
 
G. Chierchia, D. Cozzolino, G. Poggi, C. Sansone, and L. Verdoliva, “Guided
filtering for prnu-based localization of small-size image forgeries,” in
 ICASSP , 2014.

 

 
 [94] 
 
G. Chierchia, G. Poggi, C. Sansone, and L. Verdoliva, “A bayesian-mrf approach
for prnu-based image forgery detection,” TIFS , 2014.

 

 
 [95] 
 
S. Chakraborty and M. Kirchner, “Prnu-based image manipulation localization
with discriminative random fields,” Electronic Imaging , 2017.

 

 
 [96] 
 
J. He, Z. Lin, L. Wang, and X. Tang, “Detecting doctored jpeg images via dct
coefficient analysis,” in ECCV , 2006.

 

 
 [97] 
 
S. Lyu and H. Farid, “How realistic is photorealistic?” IEEE Trans.
Signal Process , 2005.

 

 
 [98] 
 
W. Chen, Y. Q. Shi, and W. Su, “Image splicing detection using 2d phase
congruency and statistical moments of characteristic function,” in
 SPIE , 2007.

 

 
 [99] 
 
Z. He, W. Lu, W. Sun, and J. Huang, “Digital image splicing detection based on
markov features in dct and dwt domain,” Pattern recognition , 2012.

 

 
 [100] 
 
Z. Lin, R. Wang, X. Tang, and H.-Y. Shum, “Detecting doctored images using
camera response normality and consistency,” in CVPR , 2005.

 

 
 [101] 
 
Y.-F. Hsu and S.-F. Chang, “Camera response functions for image forensics: an
automatic algorithm for splicing detection,” TIFS , 2010.

 

 
 [102] 
 
W. Wang, J. Dong, and T. Tan, “Effective image splicing detection based on
image chroma,” in ICIP , 2009.

 

 
 [103] 
 
Y. Ke, Q. Zhang, W. Min, and S. Zhang, “Detecting image forgery based on noise
estimation,” IJMUE , 2014.

 

 
 [104] 
 
S. Bayram, I. Avcibas, B. Sankur, and N. D. Memon, “Image manipulation
detection,” Electronic Imaging , 2006.

 

 
 [105] 
 
D. Cozzolino, D. Gragnaniello, and L. Verdoliva, “Image forgery detection
through residual-based local descriptors and block-matching,” in
 ICIP , 2014.

 

 
 [106] 
 
——, “Image forgery localization through the fusion of camera-based,
feature-based and pixel-based techniques,” in ICIP , 2014.

 

 
 [107] 
 
A. Ferreira, S. C. Felipussi, C. Alfaro, P. Fonseca, J. E. Vargas-Munoz, J. A.
Dos Santos, and A. Rocha, “Behavior knowledge space-based fusion for
copy–move forgery detection,” TIP , 2016.

 

 
 [108] 
 
Y. S. Huang and C. Y. Suen, “A method of combining multiple experts for the
recognition of unconstrained handwritten numerals,” TPAMI , 1995.

 

 
 [109] 
 
Y. Wu, W. Abd-Almageed, and P. Natarajan, “Deep matching and validation
network: An end-to-end solution to constrained image splicing localization
and detection,” in ACM MM , 2017.

 

 
 [110] 
 
R. Salloum, Y. Ren, and C.-C. J. Kuo, “Image splicing localization using a
multi-task fully convolutional network (mfcn),” J. Vis. Commun. Image
Represent. , 2018.

 

 
 [111] 
 
X. Bi, Y. Wei, B. Xiao, and W. Li, “Rru-net: The ringed residual u-net for
image splicing forgery detection,” in CVPR , 2019.

 

 
 [112] 
 
X. Cun and C.-M. Pun, “Image splicing localization via semi-global network and
fully connected conditional random fields,” in ECCV , 2018.

 

 
 [113] 
 
O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for
biomedical image segmentation,” in MICCAI , 2015.

 

 
 [114] 
 
K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in CVPR , 2016.

 

 
 [115] 
 
Y. Wu, W. Abd-Almageed, and P. Natarajan, “Image copy-move forgery detection
via an end-to-end deep neural network,” in WACV , 2018.

 

 
 [116] 
 
——, “Busternet: Detecting copy-move image forgery with source/target
localization,” in ECCV , 2018.

 

 
 [117] 
 
J.-L. Zhong and C.-M. Pun, “An end-to-end dense-inceptionnet for image
copy-move forgery detection,” TIFS , 2019.

 

 
 [118] 
 
A. Islam, C. Long, A. Basharat, and A. Hoogs, “Doa-gan: Dual-order attentive
generative adversarial network for image copy-move forgery detection and
localization,” in CVPR , 2020.

 

 
 [119] 
 
P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation
with conditional adversarial networks,” in CVPR , 2017.

 

 
 [120] 
 
D. Cozzolino and L. Verdoliva, “Single-image splicing localization through
autoencoder-based anomaly detection,” in WIFS , 2016.

 

 
 [121] 
 
B. Bayar and M. C. Stamm, “A deep learning approach to universal image
manipulation detection using a new convolutional layer,” in MMSec ,
2016.

 

 
 [122] 
 
Y. Zhang, J. Goh, L. L. Win, and V. L. Thing, “Image region forgery detection:
A deep learning approach.” SG-CRC , 2016.

 

 
 [123] 
 
P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P.-A. Manzagol, and L. Bottou,
“Stacked denoising autoencoders: Learning useful representations in a deep
network with a local denoising criterion.” JMLR , 2010.

 

 
 [124] 
 
B. Bayar and M. C. Stamm, “Constrained convolutional neural networks: A new
approach towards general purpose image manipulation detection,” TIFS ,
2018.

 

 
 [125] 
 
Z. Zhang, Y. Zhang, Z. Zhou, and J. Luo, “Boundary-based image forgery
detection by fast shallow cnn,” in ICPR , 2018.

 

 
 [126] 
 
J. H. Bappy, C. Simons, L. Nataraj, B. Manjunath, and A. K. Roy-Chowdhury,
“Hybrid lstm and encoder–decoder architecture for detection of image
forgeries,” TIP , 2019.

 

 
 [127] 
 
F. Marra, D. Gragnaniello, L. Verdoliva, and G. Poggi, “A full-image
full-resolution end-to-end-trainable cnn framework for image forgery
detection,” IEEE Access , 2020.

 

 
 [128] 
 
Y. Rao and J. Ni, “Self-supervised domain adaptation for forgery localization
of jpeg compressed images,” in ICCV , 2021.

 

 
 [129] 
 
Y. Liu, Q. Guan, X. Zhao, and Y. Cao, “Image forgery localization based on
multi-scale convolutional neural networks,” in MMSec , 2018.

 

 
 [130] 
 
J. Hao, Z. Zhang, S. Yang, D. Xie, and S. Pu, “Transforensics: image forgery
localization with dense self-attention,” in ICCV , 2021.

 

 
 [131] 
 
X. Chen, C. Dong, J. Ji, J. Cao, and X. Li, “Image manipulation detection by
multi-view multi-scale supervision,” in ICCV , 2021.

 

 
 [132] 
 
X. Liu, Y. Liu, J. Chen, and X. Liu, “Pscc-net: Progressive spatio-channel
correlation network for image manipulation detection and localization,”
 IEEE Trans. Circuits Syst. , 2022.

 

 
 [133] 
 
G. Mazaheri, N. C. Mithun, J. H. Bappy, and A. K. Roy-Chowdhury, “A skip
connection architecture for localization of image manipulations.” in
 CVPRW , 2019.

 

 
 [134] 
 
J. Chen, X. Kang, Y. Liu, and Z. J. Wang, “Median filtering forensics based on
convolutional neural networks,” SPL , 2015.

 

 
 [135] 
 
Q. Wang and R. Zhang, “Double jpeg compression forensics based on a
convolutional neural network,” Eurasip J. Inf. Secur. , 2016.

 

 
 [136] 
 
I. Amerini, T. Uricchio, L. Ballan, and R. Caldelli, “Localization of jpeg
double compression through multi-domain convolutional neural networks,” in
 CVPRW , 2017.

 

 
 [137] 
 
M. Barni, L. Bondi, N. Bonettini, P. Bestagini, A. Costanzo, M. Maggini,
B. Tondi, and S. Tubaro, “Aligned and non-aligned double jpeg detection
using convolutional neural networks,” J. Vis. Commun. Image
Represent. , 2017.

 

 
 [138] 
 
J. Park, D. Cho, W. Ahn, and H.-K. Lee, “Double jpeg detection in mixed jpeg
quality factors using deep convolutional neural network,” in ECCV ,
2018.

 

 
 [139] 
 
Q. Bammey, R. G. v. Gioi, and J.-M. Morel, “An adaptive neural network for
unsupervised mosaic consistency analysis in image forensics,” in
 CVPR , 2020.

 

 
 [140] 
 
D. Cozzolino and L. Verdoliva, “Camera-based image forgery localization using
convolutional neural networks,” in EUSIPCO , 2018.

 

 
 [141] 
 
C. Chen, S. McCloskey, and J. Yu, “Image splicing detection via camera
response function analysis,” in CVPR , 2017.

 

 
 [142] 
 
O. Mayer and M. C. Stamm, “Learned forensic source similarity for unknown
camera models,” in ICASSP , 2018.

 

 
 [143] 
 
——, “Forensic similarity for digital images,” TIFS , 2019.

 

 
 [144] 
 
——, “Exposing fake images with forensic similarity graphs,” IEEE J.
Sel. Top. Signal Process. , 2020.

 

 
 [145] 
 
A. Ghosh, Z. Zhong, T. E. Boult, and M. Singh, “Spliceradar: A learned method
for blind image forensics.” in CVPR , 2019.

 

 
 [146] 
 
L. Bondi, S. Lameri, D. Guera, P. Bestagini, E. J. Delp, S. Tubaro
 et al. , “Tampering detection and localization through clustering of
camera-based cnn features.” in CVPR , 2017.

 

 
 [147] 
 
M. Huh, A. Liu, A. Owens, and A. A. Efros, “Fighting fake news: Image splice
detection via learned self-consistency,” in ECCV , 2018.

 

 
 [148] 
 
J. Thies, M. Zollhofer, M. Stamminger, C. Theobalt, and M. Niessner,
“Face2face: Real-time face capture and reenactment of rgb videos,” in
 CVPR , 2016.

 

 
 [149] 
 
W. Wu, Y. Zhang, C. Li, C. Qian, and C. C. Loy, “Reenactgan: Learning to
reenact faces via boundary transfer,” in ECCV , 2018.

 

 
 [150] 
 
Y. Nirkin, Y. Keller, and T. Hassner, “Fsgan: Subject agnostic face swapping
and reenactment,” in ICCV , 2019.

 

 
 [151] 
 
R. Natsume, T. Yatagawa, and S. Morishima, “Rsgan: face swapping and editing
using face and hair representation in latent spaces,” in SIGGRAPH ,
2018.

 

 
 [152] 
 
P.-H. Huang, F.-E. Yang, and Y.-C. F. Wang, “Learning identity-invariant
motion representations for cross-id face reenactment,” in CVPR , 2020.

 

 
 [153] 
 
J. Bao, D. Chen, F. Wen, H. Li, and G. Hua, “Towards open-set identity
preserving face synthesis,” in CVPR , 2018.

 

 
 [154] 
 
Y. Shen, P. Luo, J. Yan, X. Wang, and X. Tang, “Faceid-gan: Learning a
symmetry three-player gan for identity-preserving face synthesis,” in
 CVPR , 2018.

 

 
 [155] 
 
C. Yang and S.-N. Lim, “One-shot domain adaptation for face generation,” in
 CVPR , 2020.

 

 
 [156] 
 
Y. Deng, J. Yang, D. Chen, F. Wen, and X. Tong, “Disentangled and controllable
face image generation via 3d imitative-contrastive learning,” in
 CVPR , 2020.

 

 
 [157] 
 
T. Karras, T. Aila, S. Laine, and J. Lehtinen, “Progressive growing of GANs
for improved quality, stability, and variation,” in ICLR , 2018.

 

 
 [158] 
 
T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila,
“Training generative adversarial networks with limited data,” in
 NeurIPS , 2020.

 

 
 [159] 
 
P. Zhu, R. Abdal, Y. Qin, J. Femiani, and P. Wonka, “Improved stylegan
embedding: Where are the good latents?” arXiv preprint
arXiv:2012.09036 , 2020.

 

 
 [160] 
 
T. Karras, M. Aittala, S. Laine, E. Härkönen, J. Hellsten, J. Lehtinen,
and T. Aila, “Alias-free generative adversarial networks,” NeurIPS ,
2021.

 

 
 [161] 
 
Z. Wu, D. Lischinski, and E. Shechtman, “Stylespace analysis: Disentangled
controls for stylegan image generation,” in CVPR , 2021.

 

 
 [162] 
 
X. Huang and S. Belongie, “Arbitrary style transfer in real-time with adaptive
instance normalization,” in ICCV , 2017.

 

 
 [163] 
 
P. Sun, Y. Li, H. Qi, and S. Lyu, “Landmarkgan: Synthesizing faces from
landmarks,” PR , 2022.

 

 
 [164] 
 
J. Zhao, X. Xie, L. Wang, M. Cao, and M. Zhang, “Generating photographic faces
from the sketch guided by attribute using gan,” IEEE Access , 2019.

 

 
 [165] 
 
W. Chao, L. Chang, X. Wang, J. Cheng, X. Deng, and F. Duan, “High-fidelity
face sketch-to-photo synthesis using generative adversarial network,” in
 ICIP , 2019.

 

 
 [166] 
 
J. Yu, X. Xu, F. Gao, S. Shi, M. Wang, D. Tao, and Q. Huang, “Toward realistic
face photo–sketch synthesis via composition-aided gans,” TC , 2020.

 

 
 [167] 
 
S.-Y. Chen, W. Su, L. Gao, S. Xia, and H. Fu, “Deepfacedrawing: Deep
generation of face images from sketches,” TOG , 2020.

 

 
 [168] 
 
Y. Lin, S. Ling, K. Fu, and P. Cheng, “An identity-preserved model for face
sketch-photo synthesis,” SPL , 2020.

 

 
 [169] 
 
Y. Li, X. Chen, B. Yang, Z. Chen, Z. Cheng, and Z.-J. Zha, “Deepfacepencil:
Creating face images from freehand sketches,” in ACM MM , 2020.

 

 
 [170] 
 
L. Li, J. Tang, Z. Shao, X. Tan, and L. Ma, “Sketch-to-photo face generation
based on semantic consistency preserving and similar connected component
refinement,” VC , 2021.

 

 
 [171] 
 
T.-H. Oh, T. Dekel, C. Kim, I. Mosseri, W. T. Freeman, M. Rubinstein, and
W. Matusik, “Speech2face: Learning the face behind a voice,” in
 CVPR , 2019.

 

 
 [172] 
 
Y. Wen, B. Raj, and R. Singh, “Face reconstruction from voice using generative
adversarial networks,” NeurIPS , 2019.

 

 
 [173] 
 
A. C. Duarte, F. Roldan, M. Tubau, J. Escur, S. Pascual, A. Salvador,
E. Mohedano, K. McGuinness, J. Torres, and X. Giro-i Nieto, “Wav2pix:
Speech-conditioned face generation using generative adversarial networks.”
in ICASSP , 2019.

 

 
 [174] 
 
C. Vondrick, H. Pirsiavash, and A. Torralba, “Generating videos with scene
dynamics,” in NeurIPS , 2016.

 

 
 [175] 
 
A. Siarohin, S. Lathuilière, S. Tulyakov, E. Ricci, and N. Sebe,
“Animating arbitrary objects via deep motion transfer,” in CVPR ,
2019.

 

 
 [176] 
 
M. Saito, E. Matsumoto, and S. Saito, “Temporal generative adversarial nets
with singular value clipping,” in ICCV , 2017.

 

 
 [177] 
 
Y. Wang, P. Bilinski, F. Bremond, and A. Dantcheva, “G3an: Disentangling
appearance and motion for video generation,” in CVPR , 2020.

 

 
 [178] 
 
——, “Imaginator: Conditional spatio-temporal gan for video generation,”
in WACV , 2020.

 

 
 [179] 
 
Y. Tian, J. Ren, M. Chai, K. Olszewski, X. Peng, D. N. Metaxas, and
S. Tulyakov, “A good image generator is what you need for high-resolution
video synthesis,” in ICLR , 2021.

 

 
 [180] 
 
S. Tulyakov, M.-Y. Liu, X. Yang, and J. Kautz, “Mocogan: Decomposing motion
and content for video generation,” in CVPR , 2018.

 

 
 [181] 
 
S. Yu, J. Tack, S. Mo, H. Kim, J. Kim, J.-W. Ha, and J. Shin, “Generating
videos with dynamics-aware implicit generative adversarial networks,” in
 ICLR , 2022.

 

 
 [182] 
 
G. Perarnau, J. van de Weijer, B. Raducanu, and J. M. Álvarez, “Invertible
Conditional GANs for image editing,” in NeurIPSW , 2016.

 

 
 [183] 
 
J. Zhang, Y. Shu, S. Xu, G. Cao, F. Zhong, M. Liu, and X. Qin, “Sparsely
grouped multi-task generative adversarial networks for facial attribute
manipulation,” in ACM MM , 2018.

 

 
 [184] 
 
Y.-C. Chen, H. Lin, M. Shu, R. Li, X. Tao, X. Shen, Y. Ye, and J. Jia,
“Facelet-bank for fast portrait manipulation,” in CVPR , 2018.

 

 
 [185] 
 
P.-W. Wu, Y.-J. Lin, C.-H. Chang, E. Y. Chang, and S.-W. Liao, “Relgan:
Multi-domain image-to-image translation via relative attributes,” in
 ICCV , 2019.

 

 
 [186] 
 
Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha, “Stargan v2: Diverse image synthesis for
multiple domains,” in CVPR , 2020.

 

 
 [187] 
 
Z. He, W. Zuo, M. Kan, S. Shan, and X. Chen, “Attgan: Facial attribute editing
by only changing what you want,” TIP , 2019.

 

 
 [188] 
 
M. Liu, Y. Ding, M. Xia, X. Liu, E. Ding, W. Zuo, and S. Wen, “Stgan: A
unified selective transfer network for arbitrary image attribute editing,”
in CVPR , 2019.

 

 
 [189] 
 
Y.-C. Chen, X. Shen, Z. Lin, X. Lu, I. Pao, J. Jia et al. , “Semantic
component decomposition for face attribute manipulation,” in CVPR ,
2019.

 

 
 [190] 
 
A. Shoshan, N. Bhonker, I. Kviatkovsky, and G. Medioni, “Gan-control:
Explicitly controllable gans,” in ICCV , 2021.

 

 
 [191] 
 
R. Abdal, P. Zhu, N. J. Mitra, and P. Wonka, “Styleflow: Attribute-conditioned
exploration of stylegan-generated images using conditional continuous
normalizing flows,” TOG , 2021.

 

 
 [192] 
 
Y. Shi, X. Yang, Y. Wan, and X. Shen, “Semanticstylegan: Learning
compositional generative priors for controllable image synthesis and
editing,” in CVPR , 2022.

 

 
 [193] 
 
S. He, W. Liao, M. Y. Yang, Y.-Z. Song, B. Rosenhahn, and T. Xiang,
“Disentangled lifespan face synthesis,” in ICCV , 2021.

 

 
 [194] 
 
Y. Xu, Y. Yin, L. Jiang, Q. Wu, C. Zheng, C. C. Loy, B. Dai, and W. Wu,
“Transeditor: Transformer-based dual-space gan for highly controllable
facial editing,” in CVPR , 2022.

 

 
 [195] 
 
J. Sun, X. Wang, Y. Zhang, X. Li, Q. Zhang, Y. Liu, and J. Wang, “Fenerf: Face
editing in neural radiance fields,” in CVPR , 2022.

 

 
 [196] 
 
S. C. Medin, B. Egger, A. Cherian, Y. Wang, J. B. Tenenbaum, X. Liu, and T. K.
Marks, “Most-gan: 3d morphable stylegan for disentangled face image
manipulation,” in AAAI , 2022.

 

 
 [197] 
 
Y. Jiang, Z. Huang, X. Pan, C. C. Loy, and Z. Liu, “Talk-to-edit: Fine-grained
facial editing via dialog,” in ICCV , 2021.

 

 
 [198] 
 
W. Xia, Y. Yang, J.-H. Xue, and B. Wu, “Tedigan: Text-guided diverse face
image generation and manipulation,” in CVPR , 2021.

 

 
 [199] 
 
T. Wei, D. Chen, W. Zhou, J. Liao, Z. Tan, L. Yuan, W. Zhang, and N. Yu,
“Hairclip: Design your hair by text and reference image,” in CVPR ,
2022.

 

 
 [200] 
 
T. Dekel, C. Gan, D. Krishnan, C. Liu, and W. T. Freeman, “Sparse, smart
contours to represent and edit images,” in CVPR , 2018.

 

 
 [201] 
 
Y. Jo and J. Park, “Sc-fegan: Face editing generative adversarial network with
user’s sketch and color,” in ICCV , 2019.

 

 
 [202] 
 
C.-H. Lee, Z. Liu, L. Wu, and P. Luo, “Maskgan: Towards diverse and
interactive facial image manipulation,” in CVPR , 2020.

 

 
 [203] 
 
R. Durall Lopez, J. Jam, D. Strassel, M. H. Yap, and J. Keuper, “Facialgan:
Style transfer and attribute manipulation on synthetic faces,” in
 BMVC , 2021.

 

 
 [204] 
 
Y. Zeng, Z. Lin, and V. M. Patel, “Sketchedit: Mask-free local image
manipulation with partial sketches,” in CVPR , 2022.

 

 
 [205] 
 
M. Afifi, M. A. Brubaker, and M. S. Brown, “Histogan: Controlling colors of
gan-generated and real images via color histograms,” in CVPR , 2021.

 

 
 [206] 
 
V. Blanz, K. Scherbaum, T. Vetter, and H.-P. Seidel, “Exchanging faces in
images,” in CGF , 2004.

 

 
 [207] 
 
J.-S. Pierrard, “Skin segmentation for robust face image analysis,” Ph.D.
dissertation, University_of_Basel, 2008.

 

 
 [208] 
 
I. Korshunova, W. Shi, J. Dambre, and L. Theis, “Fast face-swap using
convolutional neural networks,” in ICCV , 2017.

 

 
 [209] 
 
W. Shen and R. Liu, “Learning residual images for face attribute
manipulation,” in CVPR , 2017.

 

 
 [210] 
 
Y. Nirkin, I. Masi, A. T. Tuan, T. Hassner, and G. Medioni, “On face
segmentation, face swapping, and face perception,” in FG , 2018.

 

 
 [211] 
 
D. Bitouk, N. Kumar, S. Dhillon, P. Belhumeur, and S. K. Nayar, “Face
swapping: automatically replacing faces in photographs,” in SIGGRAPH ,
2008.

 

 
 [212] 
 
K. Dale, K. Sunkavalli, M. K. Johnson, D. Vlasic, W. Matusik, and H. Pfister,
“Video face replacement,” in SIGGRAPH Asia , 2011.

 

 
 [213] 
 
P. Garrido, L. Valgaerts, O. Rehmsen, T. Thormahlen, P. Perez, and C. Theobalt,
“Automatic face reenactment,” in CVPR , 2014.

 

 
 [214] 
 
I. Kemelmacher-Shlizerman, “Transfiguring portraits,” TOG , 2016.

 

 
 [215] 
 
Faceswap, “github,” https://github.com/MarekKowalski/FaceSwap/ , 2018.

 

 
 [216] 
 
S. Yan, S. He, X. Lei, G. Ye, and Z. Xie, “Video face swap based on
autoencoder generation network,” in ICALIP , 2018.

 

 
 [217] 
 
R. Natsume, T. Yatagawa, and S. Morishima, “Fsnet: An identity-aware
generative model for image-based face swapping,” in ACCV , 2018.

 

 
 [218] 
 
J. Kim, J. Lee, and B.-T. Zhang, “Smooth-swap: A simple enhancement for
face-swapping with smoothness,” in CVPR , 2022.

 

 
 [219] 
 
L. Li, J. Bao, H. Yang, D. Chen, and F. Wen, “Advancing high fidelity identity
swapping for forgery detection,” in CVPR , 2020.

 

 
 [220] 
 
R. Chen, X. Chen, B. Ni, and Y. Ge, “Simswap: An efficient framework for high
fidelity face swapping,” in ACM MM , 2020.

 

 
 [221] 
 
Z. Xu, Z. Hong, C. Ding, Z. Zhu, J. Han, J. Liu, and E. Ding, “Mobilefaceswap:
A lightweight framework for video face swapping,” in AAAI , 2022.

 

 
 [222] 
 
Y. Zhu, Q. Li, J. Wang, C.-Z. Xu, and Z. Sun, “One shot face swapping on
megapixels,” in CVPR , 2021.

 

 
 [223] 
 
J. R. A. Moniz, C. Beckham, S. Rajotte, S. Honari, and C. Pal, “Unsupervised
depth estimation, 3d face rotation and replacement,” in NeurIPS ,
2018.

 

 
 [224] 
 
Q. Sun, A. Tewari, W. Xu, M. Fritz, C. Theobalt, and B. Schiele, “A hybrid
model for identity obfuscation by face replacement,” in ECCV , 2018.

 

 
 [225] 
 
Y. Wang, X. Chen, J. Zhu, W. Chu, Y. Tai, C. Wang, J. Li, Y. Wu, F. Huang, and
R. Ji, “Hififace: 3d shape and semantic prior guided high fidelity face
swapping,” in IJCAI , 2021.

 

 
 [226] 
 
X. Yang, Y. Li, and S. Lyu, “Exposing deep fakes using inconsistent head
poses,” in ICASSP , 2019.

 

 
 [227] 
 
P. Korshunov and S. Marcel, “Deepfakes: a new threat to face recognition?
assessment and detection,” arXiv preprint arXiv:1812.08685 , 2018.

 

 
 [228] 
 
G. ai blog. contributing data to deepfake detection research., “Deepfake
detection dataset,”
 https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html ,
2019.

 

 
 [229] 
 
Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-df: A large-scale
challenging dataset for deepfake forensics,” in CVPR , 2020.

 

 
 [230] 
 
A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner,
“Faceforensics++: Learning to detect manipulated facial images,” in
 ICCV , 2019.

 

 
 [231] 
 
H. Dang, F. Liu, J. Stehouwer, X. Liu, and A. K. Jain, “On the detection of
digital face manipulation,” in CVPR , 2020.

 

 
 [232] 
 
L. Jiang, R. Li, W. Wu, C. Qian, and C. C. Loy, “Deeperforensics-1.0: A
large-scale dataset for real-world face forgery detection,” in CVPR ,
2020.

 

 
 [233] 
 
B. Dolhansky, R. Howes, B. Pflaum, N. Baram, and C. C. Ferrer, “The deepfake
detection challenge (dfdc) dataset,” arXiv preprint arXiv:2006.07397 ,
2020.

 

 
 [234] 
 
Y. He, B. Gan, S. Chen, Y. Zhou, G. Yin, L. Song, L. Sheng, J. Shao, and
Z. Liu, “Forgerynet: A versatile benchmark for comprehensive forgery
analysis,” in CVPR , 2021.

 

 
 [235] 
 
J. Thies, M. Zollhöfer, M. Nießner, L. Valgaerts, M. Stamminger, and
C. Theobalt, “Real-time expression transfer for facial reenactment.”
 TOG , 2015.

 

 
 [236] 
 
H. Averbuch-Elor, D. Cohen-Or, J. Kopf, and M. F. Cohen, “Bringing portraits
to life,” TOG , 2017.

 

 
 [237] 
 
J. Thies, M. Zollhofer, M. Stamminger, C. Theobalt, and M. Nießner,
“Face2face: Real-time face capture and reenactment of rgb videos,” in
 CVPR , 2016.

 

 
 [238] 
 
H. Kim, P. Garrido, A. Tewari, W. Xu, J. Thies, M. Niessner, P. Pérez,
C. Richardt, M. Zollhöfer, and C. Theobalt, “Deep video portraits,”
 TOG , 2018.

 

 
 [239] 
 
J. Thies, M. Zollhöfer, C. Theobalt, M. Stamminger, and M. Nießner,
“Headon: Real-time reenactment of human portrait videos,” TOG , 2018.

 

 
 [240] 
 
J. Thies, M. Zollhöfer, and M. Nießner, “Deferred neural rendering:
Image synthesis using neural textures,” TOG , 2019.

 

 
 [241] 
 
S. Qian, K.-Y. Lin, W. Wu, Y. Liu, Q. Wang, F. Shen, C. Qian, and R. He, “Make
a face: Towards arbitrary high fidelity face manipulation,” in ICCV ,
2019.

 

 
 [242] 
 
Z. Chen, C. Wang, B. Yuan, and D. Tao, “Puppeteergan: Arbitrary portrait
animation with semantic-aware appearance transformation,” in CVPR ,
2020.

 

 
 [243] 
 
J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image
translation using cycle-consistent adversarial networks,” in ICCV ,
2017.

 

 
 [244] 
 
E. Burkov, I. Pasechnik, A. Grigorev, and V. Lempitsky, “Neural head
reenactment with latent pose descriptors,” in CVPR , 2020.

 

 
 [245] 
 
G. Yao, Y. Yuan, T. Shao, S. Li, S. Liu, Y. Liu, M. Wang, and K. Zhou,
“One-shot face reenactment using appearance adaptive normalization,” in
 AAAI , 2021.

 

 
 [246] 
 
J. Wang, S. Chen, Z. Wu, and Y.-G. Jiang, “Ft-tdr: Frequency-guided
transformer and top-down refinement network for blind face inpainting,”
 TMM , 2022.

 

 
 [247] 
 
L. Song, Z. Lu, R. He, Z. Sun, and T. Tan, “Geometry guided adversarial facial
expression synthesis,” in ACM MM , 2018.

 

 
 [248] 
 
H. Hao, S. Baireddy, A. R. Reibman, and E. J. Delp, “Far-gan for one-shot face
reenactment,” in CVPRW , 2020.

 

 
 [249] 
 
N. Otberdout, M. Daoudi, A. Kacem, L. Ballihi, and S. Berretti, “Dynamic
facial expression generation on hilbert hypersphere with conditional
wasserstein generative adversarial nets,” TPAMI , 2020.

 

 
 [250] 
 
S. Ha, M. Kersner, B. Kim, S. Seo, and D. Kim, “Marionette: Few-shot face
reenactment preserving identity of unseen targets,” in AAAI , 2020.

 

 
 [251] 
 
E. Zakharov, A. Ivakhnenko, A. Shysheya, and V. Lempitsky, “Fast bi-layer
neural synthesis of one-shot realistic head avatars,” in ECCV , 2020.

 

 
 [252] 
 
J. Liu, P. Chen, T. Liang, Z. Li, C. Yu, S. Zou, J. Dai, and J. Han, “Li-net:
Large-pose identity-preserving face reenactment network,” in ICME ,
2021.

 

 
 [253] 
 
A. Pumarola, A. Agudo, A. M. Martinez, A. Sanfeliu, and F. Moreno-Noguer,
“Ganimation: Anatomically-aware facial animation from a single image,” in
 ECCV , 2018.

 

 
 [254] 
 
S. Tripathy, J. Kannala, and E. Rahtu, “Icface: Interpretable and controllable
face reenactment using gans,” in WACV , 2020.

 

 
 [255] 
 
——, “Facegan: Facial attribute controllable reenactment gan,” in
 WACV , 2021.

 

 
 [256] 
 
H. Ding, K. Sricharan, and R. Chellappa, “Exprgan: Facial expression editing
with controllable expression intensity,” in AAAI , 2018.

 

 
 [257] 
 
A. Siarohin, S. Lathuilière, S. Tulyakov, E. Ricci, and N. Sebe, “First
order motion model for image animation,” in NeurIPS , 2019.

 

 
 [258] 
 
R. Kumar, J. Sotelo, K. Kumar, A. de Brébisson, and Y. Bengio, “Obamanet:
Photo-realistic lip-sync from text,” in NeurIPS , 2017.

 

 
 [259] 
 
S. Suwajanakorn, S. M. Seitz, and I. Kemelmacher-Shlizerman, “Synthesizing
obama: learning lip sync from audio,” TOG , 2017.

 

 
 [260] 
 
O. Wiles, A. Koepke, and A. Zisserman, “X2face: A network for controlling face
generation using images, audio, and pose codes,” in ECCV , 2018.

 

 
 [261] 
 
H. Zhou, Y. Sun, W. Wu, C. C. Loy, X. Wang, and Z. Liu, “Pose-controllable
talking face generation by implicitly modularized audio-visual
representation,” in CVPR , 2021.

 

 
 [262] 
 
Y. Guo, K. Chen, S. Liang, Y.-J. Liu, H. Bao, and J. Zhang, “Ad-nerf: Audio
driven neural radiance fields for talking head synthesis,” in ICCV ,
2021.

 

 
 [263] 
 
T. Karras, T. Aila, S. Laine, A. Herva, and J. Lehtinen, “Audio-driven facial
animation by joint end-to-end learning of pose and emotion,” TOG ,
2017.

 

 
 [264] 
 
L. Song, W. Wu, C. Qian, R. He, and C. C. Loy, “Everybody’s talkin’: Let
me talk as you want,” TIFS , 2022.

 

 
 [265] 
 
J. Thies, M. Elgharib, A. Tewari, C. Theobalt, and M. Nießner, “Neural
voice puppetry: Audio-driven facial reenactment,” in ECCV , 2020.

 

 
 [266] 
 
Z. Zhang, L. Li, Y. Ding, and C. Fan, “Flow-guided one-shot talking face
generation with a high-resolution audio-visual dataset,” in CVPR ,
2021.

 

 
 [267] 
 
Z. Ye, M. Xia, R. Yi, J. Zhang, Y.-K. Lai, X. Huang, G. Zhang, and Y.-j. Liu,
“Audio-driven talking face video generation with dynamic convolution
kernels,” TMM , 2022.

 

 
 [268] 
 
S. Chen, Z. Liu, J. Liu, Z. Yan, and L. Wang, “Talking head generation with
audio and speech related facial action units,” in BMVC , 2021.

 

 
 [269] 
 
S. Wang, L. Li, Y. Ding, and X. Yu, “One-shot talking face generation from
single-speaker audio-visual correlation learning,” in AAAI , 2022.

 

 
 [270] 
 
E. Zakharov, A. Shysheya, E. Burkov, and V. Lempitsky, “Few-shot adversarial
learning of realistic neural talking head models,” in ICCV , 2019.

 

 
 [271] 
 
L. Chen, R. K. Maddox, Z. Duan, and C. Xu, “Hierarchical cross-modal talking
face generation with dynamic pixel-wise loss,” in CVPR , 2019.

 

 
 [272] 
 
K. Wang, Q. Wu, L. Song, Z. Yang, W. Wu, C. Qian, R. He, Y. Qiao, and C. C.
Loy, “Mead: A large-scale audio-visual dataset for emotional talking-face
generation,” in ECCV , 2020.

 

 
 [273] 
 
B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and
R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” in ECCV , 2020.

 

 
 [274] 
 
G. Gafni, J. Thies, M. Zollhofer, and M. Nießner, “Dynamic neural radiance
fields for monocular 4d facial avatar reconstruction,” in CVPR , 2021.

 

 
 [275] 
 
Faceswap-GAN, https://github.com/shaoanlu/faceswap-GAN , 2019.

 

 
 [276] 
 
Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the
wild,” in ICCV , 2015.

 

 
 [277] 
 
P. Zhou, X. Han, V. I. Morariu, and L. S. Davis, “Two-stream neural networks
for tampered face detection,” in CVPRW , 2017.

 

 
 [278] 
 
D. Afchar, V. Nozick, J. Yamagishi, and I. Echizen, “Mesonet: a compact facial
video forgery detection network,” in WIFS , 2018.

 

 
 [279] 
 
H. H. Nguyen, F. Fang, J. Yamagishi, and I. Echizen, “Multi-task learning for
detecting and segmenting manipulated facial images and videos,” arXiv
preprint arXiv:1906.06876 , 2019.

 

 
 [280] 
 
H. H. Nguyen, J. Yamagishi, and I. Echizen, “Capsule-forensics: Using capsule
networks to detect forged images and videos,” in ICASSP , 2019.

 

 
 [281] 
 
H. Li, B. Li, S. Tan, and J. Huang, “Identification of deep network generated
images using disparities in color components,” Signal Processing ,
2020.

 

 
 [282] 
 
P. He, H. Li, and H. Wang, “Detection of fake images via the ensemble of deep
representations from multi color spaces,” in ICIP , 2019.

 

 
 [283] 
 
S. McCloskey and M. Albright, “Detecting gan-generated imagery using
saturation cues,” in ICIP , 2019.

 

 
 [284] 
 
M. Barni, K. Kallas, E. Nowroozi, and B. Tondi, “Cnn detection of
gan-generated face images based on cross-band co-occurrences analysis,” in
 WIFS , 2020.

 

 
 [285] 
 
M. Goebel, L. Nataraj, T. Nanjundaswamy, T. M. Mohammed, S. Chandrasekaran, and
B. Manjunath, “Detection, attribution and localization of gan generated
images,” Electronic Imaging , 2021.

 

 
 [286] 
 
Y. Bai, Y. Guo, J. Wei, L. Lu, R. Wang, and Y. Wang, “Fake generated painting
detection via frequency analysis,” in ICIP , 2020.

 

 
 [287] 
 
X. Yang, Y. Li, H. Qi, and S. Lyu, “Exposing gan-synthesized faces using
landmark locations,” in WIHMS , 2019.

 

 
 [288] 
 
Y. Li and S. Lyu, “Exposing deepfake videos by detecting face warping
artifacts,” in CVPRW , 2019.

 

 
 [289] 
 
M. Koopman, A. M. Rodriguez, and Z. Geradts, “Detection of deepfake video
manipulation,” in IMVIP , 2018.

 

 
 [290] 
 
H.-S. Chen, M. Rouhsedaghat, H. Ghani, S. Hu, S. You, and C.-C. J. Kuo,
“Defakehop: A light-weight high-performance deepfake detector,” in
 ICME , 2021.

 

 
 [291] 
 
Y. Chen, M. Rouhsedaghat, S. You, R. Rao, and C.-C. J. Kuo, “Pixelhop++: A
small successive-subspace-learning-based (ssl-based) model for image
classification,” in ICIP , 2020.

 

 
 [292] 
 
S. Fernandes, S. Raj, R. Ewetz, J. S. Pannu, S. K. Jha, E. Ortiz, I. Vintila,
and M. Salter, “Detecting deepfake videos using attribution-based confidence
metric,” in CVPRW , 2020.

 

 
 [293] 
 
L. Guarnera, O. Giudice, and S. Battiato, “Deepfake detection by analyzing
convolutional traces,” in CVPRW , 2020.

 

 
 [294] 
 
O. Giudice, L. Guarnera, and S. Battiato, “Fighting deepfakes by detecting gan
dct anomalies,” Journal of Imaging , 2021.

 

 
 [295] 
 
Y. Ding, N. Thakur, and B. Li, “Does a gan leave distinct model-specific
fingerprints?” in BMVC , 2021.

 

 
 [296] 
 
T. Yang, Z. Huang, J. Cao, L. Li, and X. Li, “Deepfake network architecture
attribution,” in AAAI , 2022.

 

 
 [297] 
 
X. Zhang, S. Karaman, and S.-F. Chang, “Detecting and simulating artifacts in
gan fake images,” in WIFS , 2019.

 

 
 [298] 
 
J. Frank, T. Eisenhofer, L. Schönherr, A. Fischer, D. Kolossa, and T. Holz,
“Leveraging frequency analysis for deep fake image recognition,” in
 ICML , 2020.

 

 
 [299] 
 
Y. Huang, F. Juefei-Xu, Q. Guo, Y. Liu, and G. Pu, “Fakelocator: Robust
localization of gan-based face manipulations,” TIFS , 2022.

 

 
 [300] 
 
Z. Guo, G. Yang, J. Chen, and X. Sun, “Fake face detection via adaptive
manipulation traces extraction network,” CVIU , 2021.

 

 
 [301] 
 
A. Gandhi and S. Jain, “Adversarial perturbations fool deepfake detectors,”
in IJCNN , 2020.

 

 
 [302] 
 
S. A. Khan, A. Artusi, and H. Dai, “Adversarially robust deepfake media
detection using fused convolutional neural network predictions,” arXiv
preprint arXiv:2102.05950 , 2021.

 

 
 [303] 
 
J. Li, T. Shen, W. Zhang, H. Ren, D. Zeng, and T. Mei, “Zooming into face
forensics: A pixel-level analysis,” arXiv preprint arXiv:1912.05790 ,
2019.

 

 
 [304] 
 
C. Kong, B. Chen, H. Li, S. Wang, A. Rocha, and S. Kwong, “Detect and locate:
Exposing face manipulation by semantic-and noise-level telltales,”
 TIFS , 2022.

 

 
 [305] 
 
R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra,
“Grad-cam: Visual explanations from deep networks via gradient-based
localization,” in ICCV , 2017.

 

 
 [306] 
 
F. Chollet, “Xception: Deep learning with depthwise separable convolutions,”
in CVPR , 2017.

 

 
 [307] 
 
M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional
neural networks,” in ICML , 2019.

 

 
 [308] 
 
Z. Liu, X. Qi, and P. H. Torr, “Global texture enhancement for fake face
detection in the wild,” in CVPR , 2020.

 

 
 [309] 
 
K. Sun, T. Yao, S. Chen, S. Ding, J. Li, and R. Ji, “Dual contrastive learning
for general face forgery detection,” in AAAI , 2022.

 

 
 [310] 
 
C.-C. Hsu, C.-Y. Lee, and Y.-X. Zhuang, “Learning to detect fake face images
in the wild,” in IS3C , 2018.

 

 
 [311] 
 
D. Feng, X. Lu, and X. Lin, “Deep detection for face manipulation,” in
 ICNIP , 2020.

 

 
 [312] 
 
A. Kumar, A. Bhavsar, and R. Verma, “Detecting deepfakes with metric
learning,” in IWBF , 2020.

 

 
 [313] 
 
S. Fung, X. Lu, C. Zhang, and C.-T. Li, “Deepfakeucl: Deepfake detection via
unsupervised contrastive learning,” in IJCNN , 2021.

 

 
 [314] 
 
S. Cao, Q. Zou, X. Mao, D. Ye, and Z. Wang, “Metric learning for
anti-compression facial forgery detection,” in ACM MM , 2021.

 

 
 [315] 
 
Z. Chen and H. Yang, “Attentive semantic exploring for manipulated face
detection,” in ICASSP , 2021.

 

 
 [316] 
 
J. Cao, C. Ma, T. Yao, S. Chen, S. Ding, and X. Yang, “End-to-end
reconstruction-classification learning for face forgery detection,” in
 CVPR , 2022.

 

 
 [317] 
 
C. Wang and W. Deng, “Representative forgery mining for fake face detection,”
in CVPR , 2021.

 

 
 [318] 
 
R. Durall, M. Keuper, F.-J. Pfreundt, and J. Keuper, “Unmasking deepfakes with
simple features,” arXiv preprint arXiv:1911.00686 , 2019.

 

 
 [319] 
 
J. Li, Y. Wang, T. Tan, and A. K. Jain, “Live face detection based on the
analysis of fourier spectra,” in BTHI , 2004.

 

 
 [320] 
 
S.-Y. Wang, O. Wang, R. Zhang, A. Owens, and A. A. Efros, “Cnn-generated
images are surprisingly easy to spot… for now,” in CVPR , 2020.

 

 
 [321] 
 
Y. Qian, G. Yin, L. Sheng, Z. Chen, and J. Shao, “Thinking in frequency: Face
forgery detection by mining frequency-aware clues,” in ECCV , 2020.

 

 
 [322] 
 
H. Liu, X. Li, W. Zhou, Y. Chen, Y. He, H. Xue, W. Zhang, and N. Yu,
“Spatial-phase shallow learning: rethinking face forgery detection in
frequency domain,” in CVPR , 2021.

 

 
 [323] 
 
J. Li, H. Xie, J. Li, Z. Wang, and Y. Zhang, “Frequency-aware discriminative
feature learning supervised by single-center loss for face forgery
detection,” in CVPR , 2021.

 

 
 [324] 
 
Q. Gu, S. Chen, T. Yao, Y. Chen, S. Ding, and R. Yi, “Exploiting fine-grained
face forgery clues via progressive enhancement learning,” in AAAI ,
2022.

 

 
 [325] 
 
S. Woo et al. , “Add: Frequency attention and multi-view based knowledge
distillation to detect low-quality compressed deepfake images,” in
 AAAI , 2022.

 

 
 [326] 
 
Y. Jeong, D. Kim, S. Min, S. Joe, Y. Gwon, and J. Choi, “Bihpf: Bilateral
high-pass filters for robust deepfake detection,” in WACV , 2022.

 

 
 [327] 
 
K. Xu, M. Qin, F. Sun, Y. Wang, Y.-K. Chen, and F. Ren, “Learning in the
frequency domain,” in CVPR , 2020.

 

 
 [328] 
 
J. Wang, Z. Wu, J. Chen, and Y.-G. Jiang, “M2tr: Multi-modal multi-scale
transformers for deepfake detection,” in ICMR , 2022.

 

 
 [329] 
 
A. Khormali and J.-S. Yuan, “Dfdt: An end-to-end deepfake detection framework
using vision transformer,” Applied Sciences , 2022.

 

 
 [330] 
 
Y.-J. Heo, Y.-J. Choi, Y.-W. Lee, and B.-G. Kim, “Deepfake detection scheme
based on vision transformer and distillation,” arXiv preprint
arXiv:2104.01353 , 2021.

 

 
 [331] 
 
L. Bondi, E. D. Cannas, P. Bestagini, and S. Tubaro, “Training strategies and
data augmentations in cnn-based deepfake video detection,” in WIFS ,
2020.

 

 
 [332] 
 
P. Chen, J. Liu, T. Liang, G. Zhou, H. Gao, J. Dai, and J. Han, “Fsspotter:
spotting face-swapped video by spatial and temporal clues,” in ICME ,
2020.

 

 
 [333] 
 
U. A. Ciftci, I. Demir, and L. Yin, “Fakecatcher: Detection of synthetic
portrait videos using biological signals,” TPAMI , 2020.

 

 
 [334] 
 
Y. Li, M.-C. Chang, and S. Lyu, “In ictu oculi: Exposing ai created fake
videos by detecting eye blinking,” in WIFS , 2018.

 

 
 [335] 
 
T. Jung, S. Kim, and K. Kim, “Deepvision: Deepfakes detection using human eye
blinking pattern,” IEEE Access , 2020.

 

 
 [336] 
 
S. Agarwal, H. Farid, O. Fried, and M. Agrawala, “Detecting deep-fake videos
from phoneme-viseme mismatches,” in CVPRW , 2020.

 

 
 [337] 
 
C.-Z. Yang, J. Ma, S. Wang, and A. W.-C. Liew, “Preventing deepfake attacks on
speaker authentication by dynamic lip movement analysis,” TIFS , 2020.

 

 
 [338] 
 
J. Lin, W. Zhou, H. Liu, H. Zhou, W. Zhang, and N. Yu, “Lip forgery video
detection via multi-phoneme selection,” in WSSDL , 2021.

 

 
 [339] 
 
A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, “Lips don’t lie: A
generalisable and robust approach to face forgery detection,” in
 CVPR , 2021.

 

 
 [340] 
 
S. Agarwal, H. Farid, Y. Gu, M. He, K. Nagano, and H. Li, “Protecting world
leaders against deep fakes.” in CVPRW , 2019.

 

 
 [341] 
 
S. Fernandes, S. Raj, E. Ortiz, I. Vintila, M. Salter, G. Urosevic, and S. Jha,
“Predicting heart rate variations of deepfake videos using neural ode,” in
 ICCVW , 2019.

 

 
 [342] 
 
H. Qi, Q. Guo, F. Juefei-Xu, X. Xie, L. Ma, W. Feng, Y. Liu, and J. Zhao,
“Deeprhythm: Exposing deepfakes with attentional visual heartbeat rhythms,”
in ACM MM , 2020.

 

 
 [343] 
 
U. A. Ciftci, I. Demir, and L. Yin, “How do the hearts of deep fakes beat?
deep fake source detection via interpreting residuals with biological
signals,” in IJCB , 2020.

 

 
 [344] 
 
J. Hernandez-Ortega, R. Tolosana, J. Fierrez, and A. Morales,
“Deepfakeson-phys: Deepfakes detection based on heart rate estimation,”
 arXiv preprint arXiv:2010.00400 , 2020.

 

 
 [345] 
 
X. Li, Y. Lang, Y. Chen, X. Mao, Y. He, S. Wang, H. Xue, and Q. Lu, “Sharp
multiple instance learning for deepfake video detection,” in ACM MM ,
2020.

 

 
 [346] 
 
Y. Zheng, J. Bao, D. Chen, M. Zeng, and F. Wen, “Exploring temporal coherence
for more general video face forgery detection,” in ICCV , 2021.

 

 
 [347] 
 
L. Trinh, M. Tsang, S. Rambhatla, and Y. Liu, “Interpretable and trustworthy
deepfake detection via dynamic prototypes,” in WACV , 2021.

 

 
 [348] 
 
G. Wang, J. Zhou, and Y. Wu, “Exposing deep-faked videos by anomalous
co-motion pattern detection,” arXiv preprint arXiv:2008.04848 , 2020.

 

 
 [349] 
 
J. Hu, X. Liao, J. Liang, W. Zhou, and Z. Qin, “Finfer: Frame inference-based
deepfake detection for high-visual-quality videos,” TPAMI , 2022.

 

 
 [350] 
 
Z. Gu, Y. Chen, T. Yao, S. Ding, J. Li, and L. Ma, “Delving into the local:
Dynamic inconsistency learning for deepfake video detection,” in
 AAAI , 2022.

 

 
 [351] 
 
D. Zhang, C. Li, F. Lin, D. Zeng, and S. Ge, “Detecting deepfake videos with
temporal dropout 3dcnn.” in IJCAI , 2021.

 

 
 [352] 
 
D. Güera and E. J. Delp, “Deepfake video detection using recurrent neural
networks,” in AVSS , 2018.

 

 
 [353] 
 
F. Matern, C. Riess, and M. Stamminger, “Exploiting visual artifacts to expose
deepfakes and face manipulations,” in WACVW , 2019.

 

 
 [354] 
 
E. Tursman, M. George, S. Kamara, and J. Tompkin, “Towards untrusted social
video verification to combat deepfakes via face geometry consistency,” in
 CVPRW , 2020.

 

 
 [355] 
 
Z. Gu, Y. Chen, T. Yao, S. Ding, J. Li, F. Huang, and L. Ma, “Spatiotemporal
inconsistency learning for deepfake video detection,” in ACM MM ,
2021.

 

 
 [356] 
 
Z. Hu, H. Xie, Y. Wang, J. Li, Z. Wang, and Y. Zhang, “Dynamic
inconsistency-aware deepfake video detection,” in IJCAI , 2021.

 

 
 [357] 
 
X. Dong, J. Bao, D. Chen, W. Zhang, N. Yu, D. Chen, F. Wen, and B. Guo,
“Identity-driven deepfake detection,” arXiv preprint
arXiv:2012.03930 , 2020.

 

 
 [358] 
 
I. Amerini, L. Galteri, R. Caldelli, and A. Del Bimbo, “Deepfake video
detection through optical flow based cnn,” in ICCVW , 2019.

 

 
 [359] 
 
A. Chintha, A. Rao, S. Sohrawardi, K. Bhatt, M. Wright, and R. Ptucha,
“Leveraging edges and optical flow on faces for deepfake detection,” in
 IJCB , 2020.

 

 
 [360] 
 
K. Chugh, P. Gupta, A. Dhall, and R. Subramanian, “Not made for each
other-audio-visual dissonance-based deepfake detection and localization,” in
 ACM MM , 2020.

 

 
 [361] 
 
Y. Zhou and S.-N. Lim, “Joint audio-visual deepfake detection,” in
 ICCV , 2021.

 

 
 [362] 
 
T. Fernando, C. Fookes, S. Denman, and S. Sridharan, “Exploiting human social
cognition for the detection of fake and fraudulent faces via memory
networks,” arXiv preprint arXiv:1911.07844 , 2019.

 

 
 [363] 
 
T. Mittal, U. Bhattacharya, R. Chandra, A. Bera, and D. Manocha, “Emotions
don’t lie: An audio-visual deepfake detection method using affective cues,”
in ACM MM , 2020.

 

 
 [364] 
 
A. Haliassos, R. Mira, S. Petridis, and M. Pantic, “Leveraging real talking
faces via self-supervision for robust forgery detection,” in CVPR ,
2022.

 

 
 [365] 
 
N. Hulzebosch, S. Ibrahimi, and M. Worring, “Detecting cnn-generated facial
images in real-world scenarios,” in CVPRW , 2020.

 

 
 [366] 
 
L. Chai, D. Bau, S.-N. Lim, and P. Isola, “What makes fake images detectable?
understanding properties that generalize,” in ECCV , 2020.

 

 
 [367] 
 
K. Sun, H. Liu, Q. Ye, Y. Gao, J. Liu, L. Shao, and R. Ji, “Domain general
face forgery detection by learning to weight,” in AAAI , 2021.

 

 
 [368] 
 
Y. Luo, Y. Zhang, J. Yan, and W. Liu, “Generalizing face forgery detection
with high-frequency features,” in CVPR , 2021.

 

 
 [369] 
 
L. Chen, Y. Zhang, Y. Song, L. Liu, and J. Wang, “Self-supervised learning of
adversarial example: Towards good generalizations for deepfake detection,”
in CVPR , 2022.

 

 
 [370] 
 
P. Yu, J. Fei, Z. Xia, Z. Zhou, and J. Weng, “Improving generalization by
commonality learning in face forgery detection,” TIFS , 2022.

 

 
 [371] 
 
F. Marra, C. Saltori, G. Boato, and L. Verdoliva, “Incremental learning for
the detection and classification of gan-generated images,” in WIFS ,
2019.

 

 
 [372] 
 
H. Khalid and S. S. Woo, “Oc-fakedect: Classifying deepfakes using one-class
variational autoencoder,” in CVPRW , 2020.

 

 
 [373] 
 
D. Cozzolino, J. Thies, A. Rössler, C. Riess, M. Nießner, and
L. Verdoliva, “Forensictransfer: Weakly-supervised domain adaptation for
forgery detection,” arXiv preprint arXiv:1812.02510 , 2018.

 

 
 [374] 
 
H. Jeon, Y. O. Bang, J. Kim, and S. Woo, “T-GD: Transferable GAN-generated
images detection framework,” in ICML , 2020.

 

 
 [375] 
 
M. Kim, S. Tariq, and S. S. Woo, “Fretal: Generalizing deepfake detection
using knowledge distillation and representation learning,” in CVPR ,
2021.

 

 
 [376] 
 
S. Girish, S. Suri, S. S. Rambhatla, and A. Shrivastava, “Towards discovery
and attribution of open-world gan generated images,” in ICCV , 2021.

 

 
 [377] 
 
N. Yu, V. Skripniuk, S. Abdelnabi, and M. Fritz, “Artificial fingerprinting
for generative models: Rooting deepfake attribution in training data,” in
 ICCV , 2021.