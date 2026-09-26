Review on 6D Object Pose Estimation with the focus on Indoor Scene Understanding 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2212.01920v1 [cs.CV] 04 Dec 2022 
 
 

# Review on 6D Object Pose Estimation with the focus on Indoor Scene Understanding

 
 
 Negar Nejatishahidin
 
 Affiliation:  George Mason University
 
 Affiliation:  Fairfax, VA, USA
 
 Email:  nnejatis@gmu.edu 
 
    
 Pooya Fayyazsanavi
 
 Affiliation:  George Mason University
 
 Affiliation:  Fairfax, VA, USA
 
 Email:  pfayyazs@gmu.edu 
 

 Abstract 
 
 6D object pose estimation problem has been extensively studied in the field of Computer Vision and Robotics. It has wide range of applications such as robot manipulation, augmented reality, and 3D scene understanding. With the advent of Deep Learning, many breakthroughs have been made; however, approaches continue to struggle when they encounter unseen instances, new categories, or real-world challenges such as cluttered backgrounds and occlusions. In this study, we will explore the available methods based on input modality, problem formulation, and whether it is a category-level or instance-level approach. As a part of our discussion, we will focus on how 6D object pose estimation can be used for understanding 3D scenes.

 
 
 
 
 

 
 

## 1 Introduction

 
 An object’s 6D pose is defined as its 3D rotation and 3D translation in the camera coordinate frame. An illustration of the 6D pose concept is shown in Figure 1 .
To reason about the importance of pose estimation, we can start with its application in Augmented Reality (AR) [ 46 ] . In order to virtually place an object in an environment, information such as the 6D pose and scale of the objects in the scene can be helpful. Pose estimation models are commonly used in robot grasping as well [ 15 ] . The Amazon Picking Challenge [ 15 ] is one of the most well-known challenges that has heavily benefited from 6D pose estimation models. Furthermore, improving the performance of the object-oriented Simultaneous Localization and Mapping (SLAM) is directly influenced by camera-object constraints made with accurate pose estimation [ 72 , 50 ] . In addition, 3D detection of cars and motor vehicles [ 49 , 18 ] is essential to autonomous driving. Lastly, a holistic 3D understanding of all objects’ poses, and geometry is required to transform an RGB image into a full 3D CAD representation [ 31 , 55 , 11 ] . This representation can help to reduce the gap between the real world and the synthetic data.
 

 
 
 

 Figure 1 : This figure illustrates the pose definition. { C } \{C\} is camera frame, { O } \{O\} is object frame, R R is the rotation matrix, T T is the translation. 
 
 
 To gain a better understanding of pose estimation and the pros and cons of the various methods, we have categorized the area according to its main components. 6D Pose estimation can be defined as either classification [ 53 , 77 , 47 , 61 ] , regression [ 84 ] , or classification followed by the regression task [ 48 ] . It can also be divided based on the input modality. The RGB images [ 28 , 60 , 23 ] and RGB-D images [ 14 , 75 , 39 ] are the most common input modalities. A method either aims to solve the problem of pose estimation for instances [ 14 , 27 , 26 , 37 ] , or for categories [ 77 , 19 , 39 , 84 , 71 ] . In most instance-level approaches, the effectiveness of the method is evaluated for tabletop objects (e. g. a can, cup, bottle, or camera). In contrast, most of the Category level approaches focus on the main categories of objects (e.g. sofas, chairs, tables, desks, and cars). These approaches address the pose of the unseen instances within the seen category.

 
 
 The current approaches [ 24 , 48 ] to pose estimation typically fail when faced with real-world challenges like cluttered backgrounds, occlusions, truncation, different lighting, dark objects, glossiness, and shiny objects. Figure 2 shows some of the real-world challenges.

 
 
 Moreover, these methods cannot reliably handle unseen objects, significant differences in appearance between instances within the same category. The absence of 3D CAD models for all instances, and difficulty in understanding geometry structures are just some of the main reasons for these shortcomings.

 
 
 Most of the proposed approaches target rigid objects, however pose estimation of deformable or articulated objects has yet to be addressed [ 41 ] . Glassy [ 59 ] , shiny, or reflective objects are also challenging for pose estimation. The majority of existing approaches for top-of-the-table objects are instance-level. More recent works [ 84 ] are addressing it at the level of categories, but still, they only consider categories with slight variation in appearance. Additionally, models that propose techniques for estimating the pose of indoor objects on real-world data using category-agnostic approaches [ 53 , 67 ] and usually do not perform well enough to be applied in practice.

 
 
 

 Figure 2 : Real-world challenges from Active vision dataset  [ 2 ] ; occlusion, truncation, cluttered background, large lighting variations, glassiness, and shiny objects. Such challenges impact the accuracy of pose estimation models.. 
 
 
 

## 2 What is 6D Pose?

 
 An object 6D pose is defined as a 3D rotation R ∈ S ​ O ​ ( 3 ) R\in SO(3) and 3D translation T = [ t x , t y , t z ] T T=[t_{x},t_{y},t_{z}]^{T} matrix between each point X o = [ x o , y o , z o ] T X_{o}=[x_{o},y_{o},z_{o}]^{T} of an object defined in the object frame { O } \{O\} and the same point X c = [ x c , y c , z c ] T X_{c}=[x_{c},y_{c},z_{c}]^{T} defined in the camera frame { C } \{C\} . For rigid objects all the points have the same rotation and translation matrix, transformation matrix. In general, pose parameterizes the orientation and translation of an object in camera coordinate frame. As illustrated in Figure 1 , the 3D translation T = [ x t , y t , z t ] T T=[x_{t},y_{t},z_{t}]^{T} can be interpreted as origin of the object coordinate frame in the camera frame, and the rotation
 R = R z ​ ( α ) ​ R y ​ ( β ) ​ R x ​ ( γ ) R=R_{z}(\alpha)R_{y}(\beta)R_{x}(\gamma) are the angles between each object frame axis and correspondent camera frame axis, when the object frame origin transformed to the camera frame origin. The relationship between any point X o X_{o} in { O } \{O\} and the correspondence point X c X_{c} in { C } \{C\} can be expressed as follow:

 
 
 

 
 | 
 𝐗 𝐜 = [ R ∣ T ] ​ 𝐗 𝐨 z c ​ [ u v 1 ] = K ​ [ R T ] ​ [ x o y o z o 1 ] \begin{array}[]{c}\mathbf{X}_{\mathbf{c}}=[R\mid T]\mathbf{X}_{\mathbf{o}}\\
z_{c}\left[\begin{array}[]{l}u\\
v\\
1\end{array}\right]=K\left[\begin{array}[]{ll}R T\end{array}\right]\left[\begin{array}[]{c}x_{o}\\
y_{o}\\
z_{o}\\
1\end{array}\right]\end{array} | 
 | 
 (1) | 
 

 
 
 Where R R is:

 

 
 | 
 R = R z ​ ( α ) ​ R y ​ ( β ) ​ R x ​ ( γ ) R=R_{z}(\alpha)R_{y}(\beta)R_{x}(\gamma) | 
 | 
 

 
 
 
 = [ cos ⁡ α − sin ⁡ α 0 sin ⁡ α cos ⁡ α 0 0 0 1 ] ​ [ cos ⁡ β 0 sin ⁡ β 0 1 0 − sin ⁡ β 0 cos ⁡ β ] ​ [ 1 0 0 0 cos ⁡ γ − sin ⁡ γ 0 sin ⁡ γ cos ⁡ γ ] =\left[\begin{array}[]{ccc}\cos\alpha -\sin\alpha 0\\
\sin\alpha \cos\alpha 0\\
0 0 1\end{array}\right]\left[\begin{array}[]{ccc}\cos\beta 0 \sin\beta\\
0 1 0\\
-\sin\beta 0 \cos\beta\end{array}\right]\left[\begin{array}[]{ccc}1 0 0\\
0 \cos\gamma -\sin\gamma\\
0 \sin\gamma \cos\gamma\end{array}\right] 

 
 
 
 
 = [ cos ⁡ α ​ cos ⁡ β cos ⁡ α ​ sin ⁡ β ​ sin ⁡ γ − sin ⁡ α ​ cos ⁡ γ cos ⁡ α ​ sin ⁡ β ​ cos ⁡ γ + sin ⁡ α ​ sin ⁡ γ sin ⁡ α ​ cos ⁡ β sin ⁡ α ​ sin ⁡ β ​ sin ⁡ γ + cos ⁡ α ​ cos ⁡ γ sin ⁡ α ​ sin ⁡ β ​ cos ⁡ γ − cos ⁡ α ​ sin ⁡ γ − sin ⁡ β cos ⁡ β ​ sin ⁡ γ cos ⁡ β ​ cos ⁡ γ ] ={\left[\begin{array}[]{cccc}\cos\alpha\cos\beta \cos\alpha\sin\beta\sin\gamma-\sin\alpha\cos\gamma \cos\alpha\sin\beta\cos\gamma+\sin\alpha\sin\gamma\\
\sin\alpha\cos\beta \sin\alpha\sin\beta\sin\gamma+\cos\alpha\cos\gamma \sin\alpha\sin\beta\cos\gamma-\cos\alpha\sin\gamma\\
-\sin\beta \cos\beta\sin\gamma \cos\beta\cos\gamma\end{array}\right]} 

 
 
 
 In 3D bounding box definition, in addition to R R and T T the size of the object is also considered. The size is defined as D = [ d x , d y , d z ] D=[d_{x},d_{y},d_{z}] which is the dimension on each side of the bounding box. Other formulation of 3D bounding box can be defined with the exact eight corners of the 3D bounding box. Using these coordinate and the correspondence in the object coordinate frame, the pose can be recovered.

 
 
 Earlier, we identified three ways to divide the methods based on input, problem formulation, and whether the method is instance or category level. In the following sections, we consider a fraction of available models to more clearly clarify the area and discuss pros and cons for each method.

 
 
 

## 3 Problem Formulation

 
 Pose estimation problem can be formulated as direct classification [ 53 ] , regression [ 1 ] , 2D-3D correspondences [ 58 ] , or 3D-3D correspondences [ 93 ] . The following sections discuss various methods, along with some of the common loss functions associated with them.

 
 

### 3.1 Classification

 
 Classification approaches discretize the rotation space [ 77 , 53 ] and cast the 3D rotation estimation into a classification task. Such discretization produces a coarse result, and a post-refinement is essential to get an accurate 6D pose, using either ICP algorithm or regressing the offset [ 49 ] . Figure 3 shows a common pipeline. These models consist of Convolution layers for feature extraction followed with MLP at the end to classify the output into a number of bins. The classification loss is the cross-entropy loss as follow:

 

 
 | 
 ℒ c ​ l ​ s = − ∑ i = 1 M y o , c i log ( p o , c i ) \mathcal{L}_{cls}=-\sum_{i=1}^{M}y_{o,c_{i}}\log\left(p_{o,c_{i}}\right) | 
 | 
 

 Where M is number of classes, y y is a binary indicator ( 0 or 1 ) if class label c c is the correct classification for observation o o , and p \mathrm{p} is predicted probability of each class c c .

 
 
 

 Figure 3 : Overall pose classification model. 
 
 
 To get the full 6D pose, the classification is combined with offset regression [ 49 ] . The offset can be interpreted as the residual rotation correction that needs to be applied to the center of the bin. This offset can be represented by two numbers, the sine and the cosine of the angle, which are usually predicted with a separate branch of the network. Mostly these models perform better than approaches that directly regress the 6D pose. This results in 3 outputs for each bin i : ( c ​ l ​ a ​ s ​ s ​ _ ​ o ​ f ​ _ ​ t ​ h ​ e ​ _ ​ p ​ o ​ s ​ e , c ​ o ​ s ​ ( o ​ f ​ f ​ s ​ e ​ t ) , s ​ i ​ n ​ ( o ​ f ​ f ​ s ​ e ​ t ) ) i:({class\_of\_the\_pose},cos({offset}),sin({offset})) . The loss function for these methods combines classification and regression losses. The regression part is as follow:

 

 
 | 
 ℒ o ​ f ​ f ​ s ​ e ​ t = − 1 n θ ∗ ∑ cos ( θ ∗ − c i − Δ θ i ) \mathcal{L}_{offset}=-\frac{1}{n_{\theta^{*}}}\sum\cos\left(\theta^{*}-c_{i}-\Delta\theta_{i}\right) | 
 | 
 

 where offset angle is θ ∗ \theta^{*} , c i c_{i} is the angle of the center of bin i i and Δ ​ θ i \Delta\theta_{i} is the change that needs to be applied to the center of bin i i (offset).

 
 
 

### 3.2 Regression

 
 The pose estimation task can be addressed as a regression problem. The most simple approaches use a Convolutional Neural Network to directly regress the 3D rotation and 3D translation. Figure 4 shows the vanilla pipeline. These classification methods usually outperform these methods in pose estimation. Amini et al. [ 1 ] regressed the pose with the regression formulation. In this paper the loss is formulated as a distance of the projected points in 2D as follow:

 

 
 | 
 ℒ pose  ​ ( R i , t i , R ^ σ ⁡ ( i ) , t ^ σ ⁡ ( i ) ) = ℒ R ​ ( R i , R ^ σ ⁡ ( i ) ) + ‖ t i − t ^ σ ⁡ ( i ) ‖ \mathcal{L}_{\text{pose }}\left(R_{i},t_{i},\hat{R}_{\sigma(i)},\hat{t}_{\sigma(i)}\right)=\mathcal{L}_{R}\left(R_{i},\hat{R}_{\sigma(i)}\right)+\left\|t_{i}-\hat{t}_{\sigma(i)}\right\| | 
 | 
 

 

 
 | 
 ℒ R = 1 | ℳ | ​ ∑ x ∈ ℳ ‖ ( R i ​ x − R ^ σ ⁡ ( i ) ​ x ) ‖ \mathcal{L}_{R}=\frac{1}{|\mathcal{M}|}\sum_{\mathrm{x}\in\mathcal{M}}\left\|\left(R_{i}\mathrm{x}-\hat{R}_{\sigma(i)}\mathrm{x}\right)\right\| | 
 | 
 

 
 
 where ℳ \mathcal{M} indicates the set of 3D points from provided meshe representation of the object. R i R_{i} is the ground truth rotation and t i t_{i} is the ground truth translation. R ^ σ ⁡ ( i ) \hat{R}_{\sigma(i)} and t ^ σ ⁡ ( i ) \hat{t}_{\sigma(i)} are the predicted rotation and translation, respectively.

 
 
 

 Figure 4 : Overall 6D pose regression model. 
 
 
 

### 3.3 2D to 3D

 
 A large body of works in this area address the pose in two stages. They first estimate some correspondences between the image in-camera coordinate and the CAD model in object coordinate space. With image-CAD model pairs of rich textures, traditional feature descriptors such as Scale-Invariant Feature Transform (SIFT) can be used to compute the correspondences. However, these methods cannot address texture-less objects and low-resolution images. In section 4.1.3 we will go into more details about how to find these correspondences. Second, the correspondences are used along with Perspective-n-Point (PnP) algorithm and Random sample consensus (RANSAC) to calculate the 6D pose. PnP solves for the pose given a set of 3D points in the world coordinate frame, their corresponding 2D projections in the image, and a calibrated camera. The PnP can be solved with three points correspondences in the minimal form, called P3P. This results in four real, geometrically feasible solutions. Therefore, a fourth correspondence can be used to remove ambiguity. A more recent method, Efficient PnP (EPnP), is a method that solves the general problem of PnP for more than four correspondences.

 
 
 PnP suffers from outliers in the set of point correspondences. RANSAC can be used in conjunction with PnP to make the final solution for the camera pose more robust to outliers. RANSAC is an iterative algorithm that chooses a random subset of the original correspondences (or data points). The PnP is computed for that set (or other mathematical models for different tasks). All other data are then tested against the computed pose (or model). According to some model-specific loss functions, points that fit the estimated model well are considered part of the consensus set. The estimated model is reasonably good if most points are in the consensus set. Finally, the model will improve by re-estimating it using all members of the consensus set. Compared to classification and regression, this method is more informative and robust to occlusions and truncation.

 
 
 

### 3.4 3D to 3D

 
 In the presence of the 3D data, 6D pose estimation can be addressed using correspondences between the 3D representation of the object and the CAD model.
Later these correspondences are used to solve the translation and rotation. In order to calculate translation, we only need to subtract the average of the point in one coordinate system with the average of their correspondences in the other coordinate system. If the set of points in coordinate frame one is A A and the correspondences in coordinate frame two is B B , and the single value decomposition of B ​ A T = U ​ S ​ V T BA^{T}=USV^{T} , then the R = V ​ U T R=VU^{T} .

 
 
 To address the 6D pose and dimensions of unseen object instances in an RGB-D image, group of work try to combine the regression and correspondences formulation. These methods regress the 3D correspondence of every pixel of points inside the object’s mask to its normalized CAD model. NOCS [ 84 ] presented the Normalized coordinate space, a shared canonical representation for all possible object instances within a category, to handle different and unseen object instances within the category. The region-based neural network, based on mask_RCNN [ 21 ] , is then trained to directly regress the correspondence from observed pixels to this shared object representation (NOCS) along with other object information such as class label and instance mask. These predictions can be combined with the depth map to jointly estimate the 6D pose using Umeyama [ 81 ] and RANSAC [ 17 ] . One of the challenges in NOCS is requiring CAD models for all categories in the training stage. DRACO [ 71 ] proposed a self-supervised version of the NOCS, which doesn’t require NOCS map annotation anymore. These methods work for the category of objects with slight appearance differences such as bowl, cup, laptop, and can.

 
 
 
 

## 4 Input 

 
 Pose estimation methods can be classified according to their input modalities. Representations used in different techniques can have a significant impact on their effectiveness and the properties of the model. Various representations are investigated for current deep learning models in [ 5 ] . For pose estimation tasks, inputs to models can be RGB images or 3D information.
The 3D input uses 3D information solely or along with the RGB image. 3D data can be represented as RGB-D images, CAD models, PointClouds, Voxels, Meshes, or Truncated Sign Distance Fields (TSDF). These representations will be discussed in more detail in Section 4.2.1 .

 
 

### 4.1 RGB Images

 
 Using only RGB images to estimate the pose is a challenging task. Inherently, the shape and geometry of the object give away its pose regardless of its appearance. In order to pose estimation models capture this information from pixel-level RGB images, they require a huge number of training data. However, the labeling process is usually challenging, costly, and time-consuming. For example, in the Pix3D dataset [ 77 ] , the CAD models for all the objects are collected first. Then, the corresponding Key points between the CAD models and RGB images are hand-labeled. The pose calculated based on these correspondences using the Efficient Perspective-n-Point (EPnP) algorithm [ 40 ] . Despite all the challenges, the availability of these datasets, [ 77 , 87 , 22 , 8 , 25 ] , and easy access to RGB images in the real-world make this area an active research problem. In the following, we describe RGB-based approaches and divide them based on the commonly used methods.

 
 

#### 4.1.1 Direct Classification or Regression

 
 As discussed earlier, one way to estimate the pose is classification and regression models applied on top of RGB images. It either the directly fed to the model or a detected bounding box of the object is fed to the model, and then the pose will be estimated. We call these single shot and two-stage models.

 
 
 Two-Stage. The early approaches used two stages network. The models first detect the 2D object bounding box [ 65 ] and then estimate a single object pose from it. In Viewpoint and Keypoint [ 80 ] , pose estimation addressed as a classification problem from the 2D bounding box. The pre-trained VGG [ 73 ] on ImageNet was used to predict the Viewpoint. In Render for CNN [ 76 ] the same approach applied, but with more than 2 million rendered synthetic images with ground truth pose, resulting in a better accuracy on the pose. Mousavian et al. [ 49 ] also used VGG backbone with extra CNN and Fully connected (FC) layers on top to classify the pose from the 2D bounding box and regress the offset. The two-stage model challenge mentioned earlier [ 24 ] is the inability to estimate the pose for multiple objects in the image at once. Furthermore, using only the bounding box of an object instead of the entire image may result in the loss of useful information. For example, the room layout or object-to-object relations, which are outside of the object’s bounding box, can help the model apply some constraints and improve the performance.

 
 
 Single-Shot. Later approaches estimate the 2D bounding box and pose of multiple objects simultaneously in a fast single-shot [ 61 ] to address mentioned challenges. Poirson et al. [ 61 ] network is based on SSD [ 43 ] . These approaches are all addressed pose estimation as a classification problem using only RGB images. The SSD-6D [ 33 ] , which is also an extension to SSD, uses multi-scale features to regress the objects bounding boxes and classify the pose to the discrete number of viewpoints. It later applies a refinement stage using an edge-based algorithm to compute the exact pose. These methods are faster than the two-stage models, but they need many training samples. A single network both needs to address the category agnostic pose and 2D bounding box from RGB pixel-level information.

 
 
 

#### 4.1.2 With CAD Models

 
 Having the CAD model of objects is a big advantage for pose estimation as the object’s geometry is responsible for the pose rather than the appearance. In order to get the benefit of the CAD models, RGB-based methods use them in the training stage to render training samples. They can also be used in training and testing as an input to model along with the image. It’s hard to apply the second group to the real-world data, as the CAD models of all instances are not available.

 
 
 In Training Stage. A group of RGB-based approaches is trying to take advantage of the CAD model during the training stage. With textured CAD models, one way to use them is to augment training samples. In [ 86 ] the author used these rendered synthetic data and object masks as an intermediate representation. The model consists of 2 parts, one segmentation and the other one using the mask to classify the pose. These models mostly fail when the mask’s quality is poor or when the object is occluded or truncated.

 
 
 In Training and Testing. Rendered views of textured CAD models can be used during testing as well. SilhoNet [ 7 ] used the renderings to improve the quality of the mask, especially for occluded objects. It predicts the unoccluded mask of the occluded objects and then classifies the pose. A better mask will be estimated using rendered RGB images of textured CAD models from different viewpoints and feeding them along with the original object bounding box to the model. This mask will be fed to a model to classify the pose. Although this method addresses the occlusion problem, having the textured CAD models for each instance is challenging. In these models, objects’ masks are used as an intermediate representation to bridge synthetic and real data, simplifying the transfer of models trained on synthetic data to the real-world.

 
 
 

#### 4.1.3 2D-3D Correspondences

 
 Estimating correspondences between the image in camera coordinate and the object coordinate space is a commonly used approach. Later, these correspondences are used along with the PNP algorithm to calculate the pose. These correspondences can be the 2D projection of Corners of the 3D Bounding Box or keypoints.

 
 
 

 Figure 5 : Schematic of keypoint-based 6D pose estimation, adopted from [ 77 ] . 
 
 
 Keypoint-Based. These methods adopt a two-stage pipeline, they first predict 2D keypoints of the object and then compute the pose. For objects of rich textures, traditional methods [ 45 ] detect local keypoints robustly, so the object pose is estimated both efficiently and accurately, even under cluttered scenes and severe occlusions. However, traditional methods have difficulty handling texture-less objects and processing low-resolution images. Recent works define a set of semantic keypoints and use CNNs as keypoint detectors to solve this problem. In Semantic keypoint [ 57 ] the keypoints are predicted using hourglass [ 54 ] network. Given the predicted 2D keypoints and their correspondences on the 3D model, one naive approach simply applies an existing PnP algorithm to solve the 6-DoF pose. Due to occlusions, false detection in the background, and unavailability of exact 3D CAD models, the 6D pose is solved as an optimization problem. A simple pipline for estimating correspondences is shown in 5 . The correspondences would be the estimated keypoints and annotated keypoints on CAD models. The L 2 L_{2} Loss is minimized during training to estimate the right keypoints as follow:

 

 
 | 
 ℒ 2 = ∑ i = 1 n ( k true  i − k predicted  i ) 2 \mathcal{L}_{2}=\sum_{i=1}^{n}\left(k_{\text{true }_{i}}-k_{\text{predicted }_{i}}\right)^{2} | 
 | 
 

 
 
 Corners of Bounding Box. In BB8 [ 64 ] the correspondences are in the form of 2D projections of the eight corners of the object 3D bounding boxes. 3D pose can then be estimated using PNP algorithm and the corners correspondences.
 Voting-Based Models. To address the occlusion problem for keypoint detection, PVNet [ 58 ] predicts unit vectors pointing to keypoints for each pixel in the mask of the object and localize 2D keypoints in a RANSAC voting scheme. Later, the pose will be calculated using the PnP algorithm and predicted 2D keypoints and 3D keypoints.

 
 
 

#### 4.1.4 Transformer-Based Models.

 
 The recently proposed language model, Transformer [ 82 ] , attracts lots of attention in all areas. T6D-Direct [ 1 ] , inspiring from Transformer-based object detection model DETR [ 9 ] , proposed an approach which formulate 6D object pose direct regression as a set prediction problem. It is a single-stage direct method that directly estimates multi-object poses and bounding boxes. It predicts the center, height, width of each bounding box, and an extra branch to regress the pose.

 
 
 

#### 4.1.5 2.5D Sketch.

 
 As mentioned earlier, training a model for learning geometry information directly from the pixel level RGB images requires a large amount of training data and costly pose annotations. The resulting models do not generalize well even for the same instance with different textures or backgrounds, as they are more prone to RGB image pixels. This makes transferring from one domain (synthetic) to another domain (real-world) hard.
The goal of the 2.5D sketch estimation step is to distill intrinsic object properties from input images, such as geometry structure, while discarding properties that are non-essential for pose estimation, such as object texture. These models are more likely to transfer from synthetic to real data, as they are less sensitive to pixel-level information and more dependent on the intrinsic structure. In addition, they can learn more effectively from synthetic data. This is due to rendering realistic 2.5D sketches without modeling object appearance variations in real images, including lighting, texture, etc., are applicable. However, applying occlusion and some real-world challenges are still problematic.

 
 
 CAD Model Supervision in Training. Pix3D [ 77 ] trained a model to estimate mid-level representations (surface normal, silhouette, depth) on Pix3D dataset using a CAD model supervision during training. Later, these mid-level representations are fed to a CNN model to classify the pose and reconstruct the 3D Voxel representation.

 
 
 No CAD Model Supervision. A recently published paper [ 53 ] generated these mid-level features without the CAD model supervision during training. It proposed training a lightweight CNN network on top of Taskonomy [ 92 ] generic mid-level features. The results show that these representations help the model outperform available models by a considerable margin with a small amount of data.

 
 
 

 Figure 6 : The visualization of different 3D representations for Stanford bunny [ 51 ] . (a) 3D CAD model. (b) Pointcloud. (c) Mesh. (d) Voxel. (e) Octree. (f) TSDF. (g) Depth map. 
 
 
 
 

### 4.2 3D Input

 
 For pose estimation models, 3D data provide considerable geometric and structural information. A 3D translation, for example, is the depth of the object center. However, determining the center of an object in a partial pointcloud is a challenging task. The 3D data can be represented in different forms.

 
 

#### 4.2.1 3D Data Representations

 
 The 3D input data can be represented as CAD model, 3D Point-cloud, Mesh, Voxel, Octree, Truncated Signed Distance Field (TSDF), or depth-map. Figure 6 shows these representations. A pointcloud is an unordered set of points in three dimensions. Points are spatially defined by the X , Y , Z X,Y,Z coordinates. This representation is memory efficient and can be easily converted to any other representation. A voxel can be seen as a 3D base cubical unit that can be used to represent 3D models, which require large amounts of memory. Meshes represent the surface of 3D data with polygons. They are particularly used in computer graphics to represent surfaces or in modeling to discretize a continuous surface. An octree is a voxelized representation of a 3D shape that provides high compactness. Its underlying data structure is a tree. A TSDF is a 3D voxel array representing objects within a volume of space in which each voxel is labeled with the distance to the nearest surface. A depth map is an image or an “image channel” containing information related to the distance of the points constituting the scene from the camera coordinate frame. The RGB-D representation attaches the depth map and color information (RGB).

 
 
 

#### 4.2.2 Pointcloud Registration

 
 Pointcloud registration algorithms similar to Iterative closest point (ICP) has been used for a long time to estimate the pose of 3D shapes [ 6 ] . The 6D pose can be calculated using ICP from CAD models and the object depth. However, these algorithms usually fail when only a part of pointcloud is visible, or the rotation difference is significant. The Deep Closest Point [ 85 ] instead of using the points value, learned representations for pointcloud registration. This method can address significant rotation in comparison to other pointcloud registration models [ 90 , 94 , 3 ] . Some other groups of papers also use ICP after pose classification step as a refinement stage [ 33 ] .

 
 
 

#### 4.2.3 3D Bounding Box Detection

 
 To detect the 3D bounding box, Deep Sliding Shapes [ 75 ] Use a 3D Voxel representation of the depth as input. A fully convolutional 3D network extracts 3D proposals with different receptive fields at different scales. Later they use the 2D RGB features and these 3D proposals to regress the 3D bonding box and classify its category. In [ 39 ] they used RGB and pointcloud representation. The 2D image is used to reduce the search space in 3D. They use the 2D objects detection from RGB to crop the 3D scene into a frustum. Surface normal is used to estimate the object orientation within the frustum. A multi-layer perceptron is used to regress the object boundaries. This method fails when the bounding box doesn’t perform well, or only part of the object is visible.

 
 
 

#### 4.2.4 Keypoint-Based Techniques

 
 Like keypoint estimation in RGB images, the keypoints can be estimated from RGB and depth. As mentioned earlier, the keypoint-based models are sensitive to occlusion. These methods are primarily instance-level [ 44 ] and require the exact keypoint annotations from the exact CAD models. In addition, the ground truth label for keypoints on RGB image and CAD models are required. Georgakis et al. [ 19 ] proposed an approach, which is Category-level, and nonsensitive. In addition, the keypoint annotation are not required. The model learns consistent keypoint selection across different modalities with the pose supervision, the local features are enforced to be viewpoint invariant and modality invariant. This paper rendered different views of CAD models and forced the similar points from different view result in the same feature descriptor even for the RGB image.

 
 
 

#### 4.2.5 Voting-Based Techniques

 
 Another group of approaches addressed the occlusion with voting. VoteNet [ 12 ] proposed an approach for the whole scene, represented as a pointcloud. The points were down-sampled; each point then voted for its object’s centroid in the scene. The 3D bounding box corners are regressed on the top of clustered votes. VoteNet stated that by concatenating RGB and depth features, the performance does not improve . ImVoteNet [ 62 ] proposed an alternative to VoteNet so the RGB image can be used effectively along with the pointcloud. The author used the image to reduce the search space for the object’s center. Also, the color texture in the image used to provide a strong semantic prior. 3DPVNet [ 44 ] , inspired from VoteNet [ 12 ] and pvnet [ 58 ] , employed deep learning and Hough voting simultaneously to achieve a patch-level 3D Hough voting method for object 6D pose estimation.

 
 
 

#### 4.2.6 CAD Model and Pose

 
 CAD models bring significant geometry and structure priors to the models. [ 35 , 36 ] learn a global descriptor either from the RGB image to retrieve the most similar CAD model from a database of available CAD models. Mask2CAD [ 35 ] approach learns to map the RGB image and the renderings of the CAD model to a shared embedding space. The model was trained via contrastive loss between positive and negative pairs of the image-CAD. The model estimates the pose as a classification problem with regressing the offset. Pathch2CAD [ 36 ] is based on Mask2CAD with a focus on addressing occlusion and new viewpoints challenges. They learned a shared image-CAD embedding space by embedding patches of detected objects from the RGB images and patches of CAD models. By establishing patch-wise correspondence between image and CAD, they can establish object correspondence based on part similarities, enabling more effective shape retrieval for new views and occluded objects.

 
 
 

#### 4.2.7 3D-3D Correspondences

 
 In order to address the pose we can learn 3D to 3D correspondences and then solve the transformation. The 3D input data can be represented with pointcloud and its corresponding positive and negative pairs (pointclouds of CAD models). Using contrastive loss, an Encoder-Decoder model can be trained to generate local features per point for all the inputs. The features of the corresponding points should be similar and far from the others. During testing, Local features are generated for the CAD model and the object pointcloud. Then matched pairs are used to recover the pose of the query using RANSAC. Figure 7 show this pipeline.
The contrastive loss for feature learning can be represented as follow:

 

 
 | 
 ℒ con ​ ( 𝐅 𝐗 , 𝐅 𝐘 ) \displaystyle\mathcal{L}_{\mathrm{con}}\left(\mathbf{F}^{\mathbf{X}},\mathbf{F}^{\mathbf{Y}}\right) | 
 = ∑ ( i , j ) ∈ 𝒫 ℒ max ⁡ ( 0 , ‖ 𝐟 i 𝐗 − 𝐟 j 𝐲 ‖ 2 − p + ) 2 \displaystyle=\sum_{(i,j)\in\mathcal{P}_{\mathcal{L}}}\max\left(0,\left\|\mathbf{f}_{i}^{\mathbf{X}}-\mathbf{f}_{j}^{\mathbf{y}}\right\|_{2}-p_{+}\right)^{2} | 
 | 
 
 
 | 
 | 
 + ∑ ( i , j ) ∈ 𝒩 ℒ max ( 0 , p − − ‖ 𝐟 i 𝐱 − 𝐟 j 𝐲 ‖ 2 ) 2 \displaystyle+\sum_{(i,j)\in\mathcal{N}_{\mathcal{L}}}\max\left(0,p_{-}-\left\|\mathbf{f}_{i}^{\mathbf{x}}-\mathbf{f}_{j}^{\mathbf{y}}\right\|_{2}\right)^{2} | 
 | 
 

 Where 𝒩 ℒ \mathcal{N}_{\mathcal{L}} is the positive pairs of points and 𝒩 ℒ \mathcal{N}_{\mathcal{L}} is the negative pairs of points. 𝐅 𝐗 \mathbf{F}^{\mathbf{X}} and 𝐅 𝐘 \mathbf{F}^{\mathbf{Y}} are associated features to two pointclouds from the training set. p + p_{+} and p − p_{-} are the positive and negative thresholds.

 
 
 

 Figure 7 : Schematic of 3D-3D correspondence-based 6D pose estimation, similar to [ 93 ] . (Y) is object partial pointcloud representation in camera coordinate frame. (P) is a full pointcloud representation of ground-truth CAD model in object coordinate frame. (N) is a full pointcloud representation of other CAD model in object coordinate frame. The Encoder-Decoder architecture can be either based on sparse convolutions [ 10 ] or Pointnet++ [ 63 ] . 
 
 
 In CORSAIR [ 93 ] along with learning the point-wise local features to address the correspondences, a retrieval block is trained to generate a global shape embedding. During testing, given a pointcloud, using the pre-computed CAD models global embedding, the nearest neighbors model is retrieved. Local features are then generated for both pointclouds, and matching pairs are used to recover the pose of the query using RANSAC.

 
 
 
 
 

## 5 Instance vs Category Level Approaches 

 
 There is a large body of works focusing on instance-level 6D pose estimation. These models mainly address the pose for the top of the table objects (e.g. cup, laptop, camera, and can). The dominant techniques in these models are trying to do matching between either the RGB or RGB-D input, and the CAD model [ 20 , 38 , 56 ] but having the CAD model for all instances of all categories is not feasible in real-world. The challenges mainly encountered at the level of instances are viewpoint variability, texture-less objects, occlusion, and clutter. Instance level approaches fail when it comes to the unseen instances within the seen categories.
In order to models robustly work in a generalized fashion NOCS [ 84 ] proposed an approach for category level pose estimation for the top of the table objects with a dataset.
Most of the category-level pose estimation models [ 53 , 77 , 49 , 19 ] works for main categories of objects (e.g., chair, sofa, desk, table, bench, and car). These models mostly make simplifying assumptions about the pose using available constraints in the real-world. For example, the categories mentioned above are parallel to the ground, and only azimuth and elevation are estimated. The main challenges of the category-level 6D object pose estimation are intra-class variation and unseen instances in the test stage.

 
 
 

## 6 Datasets

 
 Datasets can be divided into two main groups: Top of the table objects, which are addressed mainly by the instance-level approaches, and Main categories of objects, which mostly address the category-level approaches.

 
 

### 6.1 Instance-Level Datasets

 
 Linemod [ 22 ] , Linemod-Occluded [ 8 ] , T-LESS [ 25 ] , IC-MI [ 78 ] , IC-BIN [ 13 ] are the datasets most frequently used to test the performances of instance-level full 6D pose estimators. In a recently proposed benchmark for 6D object pose estimation [ 27 ] , these datasets are refined and are presented in a unified format along with three new datasets (Rutgers Amazon Picking Challenge [ 66 ] , TUD Light, and Toyota Light).

 
 
 Texture-Less Datasets. Linemod has 15 texture-less household objects with discriminative color, shape, and size. The test sets show an annotated object instance with significant clutter but only mild occlusion.
Linemod-Occluded introduced challenging test cases with various levels of occlusion over Linemod.
T-LESS contains 30 industry-relevant objects with no significant texture or discriminative color. The objects exhibit symmetries and mutual similarities in shape or size, and a few objects are a composition of other objects. Test images originate from 20 scenes with varying complexity.

 
 
 Textured Datasets. IC-MI has four textured households and two texture-less objects. The test images show multiple object instances with clutter and slight occlusion. IC-BIN added test images of two objects from IC-MI, which appear in multiple locations with heavy occlusion in a bin-picking scenario. Rutgers Amazon Picking Challenge contains 14 textured products from the Amazon Picking Challenge, each associated with test images of a cluttered warehouse shelf. TUD Light contains three moving objects under eight lighting conditions. Toyota Light contains 21 objects, each captured in multiple poses on a table-top setup, with four different backgrounds and five different lighting conditions.

 
 
 

### 6.2 Category-Level Datasets

 
 KITTI [ 18 ] , SUN RGB-D [ 74 ] , NYU-Depth v2 [ 52 ] , PASCAL3D [ 88 ] , and Pix3D [ 77 ] are the datasets for main categories of objects including, car, sofa, chair, desk ,table, cabinet, bed, and lamp.

 
 
 CAD Model is Available. 
In particular, KITTI has three categories, car, pedestrian, and cyclist, with a total number of 80256 labeled objects. It has 14999 images, 7481 for training the detectors, and the remaining is for testing.
PASCAL3D+ [ 88 ] is a 3D version of PASCAL VOC 2012 [ 16 ] (airplane, bicycle, boat, bottle, bus, car, chair, dining table, motorbike, sofa, train, and monitor) are augmented with 3D annotations. For each category, more images are added from ImageNet [ 68 ] , resulting in a total number of 30899 images with annotated objects [ 69 ] .
The pix3D [ 77 ] dataset has nine categories (bed, chair, table, desk, sofa, wardrobes, bookcase, misc, and tool). For each instance, around 50 images are available; for example, the bed category contains 1000 images with 20 instances (around 50 images per instance). Pix3D contains 10,069 images and 395 CAD models.

 
 
 Depth is Available. 
SUN RGB-D [ 74 ] has 10335 RGB-D images in total, 3784 of which are captured by Kinect v2 and 1159 by IntelRealSense. The remaining are collected from the datasets of NYU-Depth v2 [ 52 ] , B3DO [ 32 ] , and SUN3D [ 74 ] . This dataset has bounding box annotation for 10,335 RGB-D images. The dominant categories are chair, table, sofa, desk, bed, cabinet, and lamp. There are around 5K training in the dataset for 37 object categories, and the rest is for testing. The dataset also has room layout annotation, making it a good candidate for a 3D holistic understanding. The NYU-Depth v2 dataset [ 52 ] involves 1449 images, 795 of which are of training, and the rest is for the test. Even though it has 894 objects, there are mainly 19 object categories on which the methods are evaluated (bathtub, bed, bookshelf, box, chair, counter, desk, door, dresser, garbage bin, lamp, monitor, nightstand, pillow, sink, sofa, table, tv, toilet).

 
 
 
 

## 7 3D Scene Modeling

 
 Understanding 3D scene from a single image is fundamental to various tasks, such as robotics, motion planning, or augmented reality. To fully understand a 3D scene from a single image, we need to know the pose, dimension, and 3D reconstruction of all objects, along with the room layout and the camera pose. With this information, given a single photo of a room, the scene can be synthesized, resulting in a scene that is as close to the photographed scene as possible, see Figure 8 . The 3D reconstruction can be represented in any form of CAD model, Voxel, Mesh, or NeRF. 3D Layout estimation aims to reconstruct the intrinsic 3D structure of a room (the geometry of the floor, ceiling, and walls). This task is challenging as the scenes are usually full of various objects and furniture. In an indoor setting, the room layout, object pose, and camera pose are all related and can apply certain constraints to reduce the complexity of the search space. For example, the main indoor objects are always parallel to the ground, which means with knowing the layout, we only have one degree of freedom for the rotation of the object pose (Azimuth). We describe some of the pipelines that address the 3D Visual Understanding problem in the following.

 
 
 

 Figure 8 : This picture is adopted from [ 31 ] . It shows the idea of fully understanding a 3D scene is composed of object poses, object 3D Reconstruction, camera pose, and layout of the room. 
 
 
 IM2CAD [ 31 ] is a completely automatic system addressing the 3D scene modeling problem. First, the layout of the room is estimated by classifying pixels as being on walls, floor, or ceiling and fitting a box shape to the result. In parallel, they detect all of the main categories of the object such as chairs, tables, sofas, bookshelves, beds, night tables, and windows in the scene using [ 65 ] . The CAD model and pose of the objects are estimated by comparing their appearance with renderings of hundreds of beds from many different angles, using a deep convolutional distance metric trained for this purpose. Finally, using the difference between the rendered room and the photograph, they optimize the placement of objects. Although these models perform well, pose estimation and CAD model recovery may not work when the objects are more cluttered and occluded. Additionally, the layout of the room has a box shape, but in reality, it can have another shape. Total3DUnderestanding [ 55 ] proposed a network, learning end-to-end for comprehensive 3D scene understanding with mesh reconstruction at the instance level. It is composed of three modules, the Layout Estimation Network, 3D Object Detection Network, and Mesh Generation Network. SceneCAD [ 4 ] detects the object, estimates the layout and retrieves a similar CAD model. Later on, it builds a graph out of these estimations to understand the holistic of the scene. This enables a robust retrieval and alignment of CAD models to the scan as well as layout generation, resulting in a lightweight CAD-based representation of the scene. In [ 11 ] the author defines 3D scene modeling as a panoptic 3D scene reconstruction. It is composed of predicting the geometry of the scene, semantic labels of all the points, and finding object instances within the image view.

 
 
 Some of the approaches in this area more focus on using effective representations [ 29 , 42 ] to address the challenges and boost performance. recently proposed approach, InstPIFu [ 42 ] , has used instance-aligned implicit function for detailed object reconstruction. This makes InstPIFu more robust to occlusion and has a better generalization ability on real-world datasets. In addition, it proposes using an implicit representation for the layout of the room instead of the box representation which results in generalization toward all room shapes.

 
 
 

## 8 Open Challenges

 
 Detecting objects and their 6D poses are an integral part of spatial 3D perception relevant to semantic simultaneous localization and mapping approaches  [ 72 ] , target-driven navigation, autonomous driving, object manipulation, and augmented reality. Although the state-of-the-art deep learning approaches have marked notable advancements by training pose estimation models, the resulting models do not generalize well even to the same instance in different environments, especially when it comes to real-world data.

 
 
 The main challenges in pose estimation can be divided into three major categories. 1) Real-world challenges, 2) addressing unseen instances, and 3) addressing unseen object categories.

 
 
 It has been noted that model performances are not up to par due to real-world challenges. Figure 2 shows some of these challenges. A key factor behind this performance drop is the difficulty in understanding the geometry of an object in real-world settings. For example, understanding the structure of an object which is partially occluded, transparent, or dark is challenging. This type of difficulty is not restricted to only pose estimation; semantic segmentation, object classification, and detection can also fail in these scenarios. Many approaches [ 53 , 49 , 76 ] are dependent on these state-of-the-art object detection [ 65 ] or mask segmentation [ 21 ] as a part of their pipeline.

 
 
 [ 31 , 55 , 30 , 89 ] tries to address some of the real-world challenges for main categories of indoor objects. However, DeMF [ 89 ] , which is the best-performing model on SunRGBD, is 46 % 46\% . For some categories like bookshelves and desks, the performance is even less than 17 % 17\% . Performances also drop when they are applied to other datasets. According to the results, the models have difficulty understanding the geometry when there is noise in the depth data or when occlusion or truncation are present. The promise of voting-based techniques [ 58 , 44 ] are being robust to these challenges. But these models address the occlusion and truncation for a small number of object instances. The other voting-based techniques [ 12 , 62 ] vote for the center which improves the translation accuracy. However, these models require the entire scene pointcloud, which is not applicable to many robotics applications depending on pose estimation approaches.

 
 
 The second main problem is the difficulty in addressing unseen instances. Most of the RGB-based models, which train an end-to-end CNN model to estimate the pose, do not perform well on other datasets, even on the same categories. As they learn a direct mapping from the pixel level information to the object pose, and they are fitting to the distribution of the dataset, especially when there is no supervision or constraint to learn the geometry information. The models which get the advantage of the 3D information are either using RGB-D or CAD models. Those which are highly dependent on CAD models are mainly limited to the instances in which the CAD models are available. These models are more stable and have a better idea of the object structure. Although some techniques are trying to define a deformable shape priors [ 57 ] to address the unavailability of CAD models for all the instances, but still not performing well for all the categories, especially the ones with more appearance change within the category.

 
 
 Lastly, it is extremely challenging to address novel categories and objects, a newly proposed approach called FS6D [ 91 ] inspired by SuperGLUE [ 83 ] , a feature matching technique, addressed the object pose estimation as a feature matching problem between few 2D views of exactly the same object with the ground truth pose annotation and the query image. Although the idea is novel and extends the pose estimation to novel objects in a few-shot manner, still zero-shot techniques for novel object pose estimation are missing. Techniques that understand the geometry and do not highly depend on the texture.

 
 
 A holistic understanding of the scene can be used to improve pose estimation performance and build more generalized models for addressing real-world challenges in this area. This holistic understanding can be interpreted as object-to-object and object-to-layout relations. These can be applied in the prepossessing step as a constraint [ 34 ] , or in the post-processing as an optimization part.

 
 
 In addition, multi-view information [ 70 ] can help to address truncated or occlusion to some extent. Also, it can assist in applying constraints to the pose of a static object; the pose of a static object depends on the transformation of the camera.

 
 
 The unavailability of the CAD models for all instances of the objects can be tackled by defining representative shape priors [ 79 ] for each category. For ones with more appearance changes, sub-category shape priors can be defined. These shape priors represent CAD models’ essential and representative components throughout the entire category.

 
 
 

## 9 Conclusion

 
 We have presented a short review on state-of-the-art approaches to Object Pose estimation. The approaches are classified according to the essential properties that affect both performance and the targeted objects. However, models continue to fail when they encounter novel objects or challenging scenes. The use of approaches that consider a holistic 3D understanding can assist in addressing challenging real-world scenarios. Another option is to develop models which have a deeper understanding of the object structure.

 
 
 

## References

 
 
 [1] 
 
A. A. Amini, A. S. Periyasamy, and S. Behnke.
 
 
 T6d-direct: Transformers for multi-object 6d pose direct regression.
 
 
 In GCPR , 2021.
 
 

 
 [2] 
 
P. Ammirato, P. Poirson, E. Park, J. Kosecka, and A. C. Berg.
 
 
 A dataset for developing and benchmarking active vision.
 
 
 CoRR , abs/1702.08272, 2017.
 
 

 
 [3] 
 
Y. Aoki, H. Goforth, R. A. Srivatsan, and S. Lucey.
 
 
 Pointnetlk: Robust efficient point cloud registration using
pointnet.
 
 
 2019 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , pages 7156–7165, 2019.
 
 

 
 [4] 
 
A. Avetisyan, T. Khanova, C. B. Choy, D. Dash, A. Dai, and M. Nießner.
 
 
 Scenecad: Predicting object alignments and layouts in rgb-d scans.
 
 
 In ECCV , 2020.
 
 

 
 [5] 
 
K. T. Baghaei, A. Payandeh, P. Fayyazsanavi, S. Rahimi, Z. Chen, and S. B.
Ramezani.
 
 
 Deep representation learning: Fundamentals, perspectives,
applications, and open challenges, 2022.
 
 

 
 [6] 
 
P. J. Besl and N. D. McKay.
 
 
 A method for registration of 3-d shapes.
 
 
 IEEE Trans. Pattern Anal. Mach. Intell. , 14:239–256, 1992.
 
 

 
 [7] 
 
G. Billings and M. Johnson-Roberson.
 
 
 Silhonet: An rgb method for 6d object pose estimation.
 
 
 IEEE Robotics and Automation Letters , 4:3727–3734, 2019.
 
 

 
 [8] 
 
E. Brachmann, A. Krull, F. Michel, S. Gumhold, J. Shotton, and C. Rother.
 
 
 Learning 6d object pose estimation using 3d object coordinates.
 
 
 In ECCV , 2014.
 
 

 
 [9] 
 
N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko.
 
 
 End-to-end object detection with transformers.
 
 
 ArXiv , abs/2005.12872, 2020.
 
 

 
 [10] 
 
C. Choy, J. Park, and V. Koltun.
 
 
 Fully convolutional geometric features.
 
 
 In ICCV , 2019.
 
 

 
 [11] 
 
M. Dahnert, J. Hou, M. Nießner, and A. Dai.
 
 
 Panoptic 3d scene reconstruction from a single rgb image.
 
 
 ArXiv , abs/2111.02444, 2021.
 
 

 
 [12] 
 
Z. Ding, X. Han, and M. Niethammer.
 
 
 Votenet: A deep learning label fusion method for multi-atlas
segmentation.
 
 
 ArXiv , abs/1904.08963, 2019.
 
 

 
 [13] 
 
A. Doumanoglou, R. Kouskouridas, S. Malassiotis, and T.-K. Kim.
 
 
 Recovering 6d object pose and predicting next-best-view in the crowd.
 
 
 2016 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , pages 3583–3592, 2016.
 
 

 
 [14] 
 
T. El-Gaaly and M. Torki.
 
 
 Rgbd object pose recognition using local-global multi-kernel
regression.
 
 
 Proceedings of the 21st International Conference on Pattern
Recognition (ICPR2012) , pages 2468–2471, 2012.
 
 

 
 [15] 
 
C. Eppner, S. Höfer, R. Jonschkowski, R. Martín-Martín,
A. Sieverling, V. Wall, and O. Brock.
 
 
 Lessons from the amazon picking challenge: Four aspects of building
robotic systems.
 
 
 In IJCAI , 2017.
 
 

 
 [16] 
 
M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman.
 
 
 The pascal visual object classes (voc) challenge.
 
 
 International Journal of Computer Vision , 88(2):303–338, June
2010.
 
 

 
 [17] 
 
M. A. Fischler and R. C. Bolles.
 
 
 Random sample consensus: a paradigm for model fitting with
applications to image analysis and automated cartography.
 
 
 Commun. ACM , 24:381–395, 1981.
 
 

 
 [18] 
 
A. Geiger, P. Lenz, C. Stiller, and R. Urtasun.
 
 
 Vision meets robotics: The kitti dataset.
 
 
 The International Journal of Robotics Research , 32:1231 –
1237, 2013.
 
 

 
 [19] 
 
G. Georgakis, S. Karanam, Z. Wu, J. Ernst, and J. Košecká.
 
 
 End-to-end learning of keypoint detector and descriptor for pose
invariant 3d matching.
 
 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , June 2018.
 
 

 
 [20] 
 
R. L. Haugaard and A. G. Buch.
 
 
 Surfemb: Dense and continuous correspondence distributions for object
pose estimation with learnt surface embeddings.
 
 
 CoRR , abs/2111.13489, 2021.
 
 

 
 [21] 
 
K. He, G. Gkioxari, P. Dollár, and R. B. Girshick.
 
 
 Mask r-cnn.
 
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
42:386–397, 2020.
 
 

 
 [22] 
 
S. Hinterstoisser, V. Lepetit, S. Ilic, S. Holzer, G. Bradski, K. Konolige, ,
and N. Navab.
 
 
 Model based training, detection and pose estimation of texture-less
3d objects in heavily cluttered scenes.
 
 
 2012.
 
 

 
 [23] 
 
T. Hodan, D. Baráth, and J. Matas.
 
 
 Epos: Estimating 6d pose of objects with symmetries.
 
 
 2020 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , pages 11700–11709, 2020.
 
 

 
 [24] 
 
T. Hodan, R. Kouskouridas, T.-K. Kim, F. Tombari, K. E. Bekris, B. Drost,
T. Groueix, K. Walas, V. Lepetit, A. Leonardis, C. Steger, F. Michel,
C. Sahin, C. Rother, and J. Matas.
 
 
 A summary of the 4th international workshop on recovering 6d object
pose.
 
 
 In ECCV Workshops , 2018.
 
 

 
 [25] 
 
T. Hodan, F. Michel, E. Brachmann, W. Kehl, A. G. Buch, D. Kraft, B. Drost,
J. Vidal, S. Ihrke, X. Zabulis, C. Sahin, F. Manhardt, F. Tombari, T.-K. Kim,
J. Matas, and C. Rother.
 
 
 Bop: Benchmark for 6d object pose estimation.
 
 
 ArXiv , abs/1808.08319, 2018.
 
 

 
 [26] 
 
T. Hodaň, F. Michel, E. Brachmann, W. Kehl, A. Glent Buch, D. Kraft,
B. Drost, J. Vidal, S. Ihrke, X. Zabulis, C. Sahin, F. Manhardt, F. Tombari,
T.-K. Kim, J. Matas, and C. Rother.
 
 
 BOP: Benchmark for 6D object pose estimation.
 
 
 European Conference on Computer Vision (ECCV) , 2018.
 
 

 
 [27] 
 
T. Hodaň, M. Sundermeyer, B. Drost, Y. Labbé, E. Brachmann,
F. Michel, C. Rother, and J. Matas.
 
 
 BOP challenge 2020 on 6D object localization.
 
 
 European Conference on Computer Vision Workshops (ECCVW) , 2020.
 
 

 
 [28] 
 
Y. Hu, P. Fua, W. Wang, and M. Salzmann.
 
 
 Single-stage 6d object pose estimation.
 
 
 2020 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , pages 2927–2936, 2020.
 
 

 
 [29] 
 
S. Huang, Y. Chen, T. Yuan, S. Qi, Y. Zhu, and S.-C. Zhu.
 
 
 Perspectivenet: 3d object detection from a single rgb image via
perspective points.
 
 
 In Neural Information Processing Systems , 2019.
 
 

 
 [30] 
 
S. Huang, S. Qi, Y. Xiao, Y. Zhu, Y. N. Wu, and S. Zhu.
 
 
 Cooperative holistic scene understanding: Unifying 3d object, layout,
and camera pose estimation.
 
 
 CoRR , abs/1810.13049, 2018.
 
 

 
 [31] 
 
H. Izadinia, Q. Shan, and S. M. Seitz.
 
 
 IM2CAD.
 
 
 CoRR , abs/1608.05137, 2016.
 
 

 
 [32] 
 
A. Janoch, S. Karayev, Y. Jia, J. T. Barron, M. Fritz, K. Saenko, and
T. Darrell.
 
 
 A category-level 3-d object dataset: Putting the kinect to work.
 
 
 In ICCV Workshops , 2011.
 
 

 
 [33] 
 
W. Kehl, F. Manhardt, F. Tombari, S. Ilic, and N. Navab.
 
 
 Ssd-6d: Making rgb-based 3d detection and 6d pose estimation great
again.
 
 
 2017 IEEE International Conference on Computer Vision (ICCV) ,
pages 1530–1538, 2017.
 
 

 
 [34] 
 
N. Kulkarni, I. Misra, S. Tulsiani, and A. Gupta.
 
 
 3d-relnet: Joint object and relational network for 3d prediction.
 
 
 CoRR , abs/1906.02729, 2019.
 
 

 
 [35] 
 
W. Kuo, A. Angelova, T.-Y. Lin, and A. Dai.
 
 
 Mask2cad: 3d shape prediction by learning to segment and retrieve.
 
 
 In ECCV , 2020.
 
 

 
 [36] 
 
W. Kuo, A. Angelova, T.-Y. Lin, and A. Dai.
 
 
 Patch2cad: Patchwise embedding learning for in-the-wild shape
retrieval from a single image.
 
 
 ArXiv , abs/2108.09368, 2021.
 
 

 
 [37] 
 
Y. Labbé, J. Carpentier, M. Aubry, and J. Sivic.
 
 
 Cosypose: Consistent multi-view multi-object 6d pose estimation.
 
 
 CoRR , abs/2008.08465, 2020.
 
 

 
 [38] 
 
Y. Labbé, J. Carpentier, M. Aubry, and J. Sivic.
 
 
 Cosypose: Consistent multi-view multi-object 6d pose estimation.
 
 
 In European Conference on Computer Vision , pages 574–591.
Springer, 2020.
 
 

 
 [39] 
 
J. Lahoud and B. Ghanem.
 
 
 2d-driven 3d object detection in rgb-d images.
 
 
 In 2017 IEEE International Conference on Computer Vision
(ICCV) , pages 4632–4640, 2017.
 
 

 
 [40] 
 
V. Lepetit, F. Moreno-Noguer, and P. Fua.
 
 
 Epnp: An accurate o(n) solution to the pnp problem.
 
 
 International Journal of Computer Vision , 81:155–166, 2008.
 
 

 
 [41] 
 
Y. Li, Y. Wang, M. Case, S.-F. Chang, and P. K. Allen.
 
 
 Real-time pose estimation of deformable objects using a volumetric
approach.
 
 
 In 2014 IEEE/RSJ International Conference on Intelligent Robots
and Systems , pages 1046–1052, 2014.
 
 

 
 [42] 
 
H. Liu, Y. Zheng, G. Chen, S. Cui, and X. Han.
 
 
 Towards high-fidelity single-view holistic reconstruction of indoor
scenes.
 
 
 In European Conference on Computer Vision , 2022.
 
 

 
 [43] 
 
W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. E. Reed, C.-Y. Fu, and A. C.
Berg.
 
 
 Ssd: Single shot multibox detector.
 
 
 In ECCV , 2016.
 
 

 
 [44] 
 
Y. Liu, J. Zhou, Y. Zhang, C. Ding, and J. Wang.
 
 
 3dpvnet: Patch-level 3d hough voting network for 6d pose estimation.
 
 
 ArXiv , abs/2009.06887, 2020.
 
 

 
 [45] 
 
D. G. Lowe.
 
 
 Object recognition from local scale-invariant features.
 
 
 Proceedings of the Seventh IEEE International Conference on
Computer Vision , 2:1150–1157 vol.2, 1999.
 
 

 
 [46] 
 
É. Marchand, H. Uchiyama, and F. Spindler.
 
 
 Pose estimation for augmented reality: A hands-on survey.
 
 
 IEEE Transactions on Visualization and Computer Graphics ,
22:2633–2651, 2016.
 
 

 
 [47] 
 
I. Misra, R. Girdhar, and A. Joulin.
 
 
 An end-to-end transformer model for 3d object detection.
 
 
 CoRR , abs/2109.08141, 2021.
 
 

 
 [48] 
 
A. Mousavian, D. Anguelov, J. Flynn, and J. Kosecka.
 
 
 3d bounding box estimation using deep learning and geometry.
 
 
 CoRR , abs/1612.00496, 2016.
 
 

 
 [49] 
 
A. Mousavian, D. Anguelov, J. Flynn, and J. Kosecka.
 
 
 3d bounding box estimation using deep learning and geometry.
 
 
 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , pages 5632–5640, 2017.
 
 

 
 [50] 
 
R. Mur-Artal, J. M. M. Montiel, and J. D. Tardós.
 
 
 ORB-SLAM: a versatile and accurate monocular SLAM system.
 
 
 CoRR , abs/1502.00956, 2015.
 
 

 
 [51] 
 
M. Naseer, S. H. Khan, and F. M. Porikli.
 
 
 Indoor scene understanding in 2.5/3d for autonomous agents: A survey.
 
 
 IEEE Access , 7:1859–1887, 2019.
 
 

 
 [52] 
 
P. K. Nathan Silberman, Derek Hoiem and R. Fergus.
 
 
 Indoor segmentation and support inference from rgbd images.
 
 
 In ECCV , 2012.
 
 

 
 [53] 
 
N. Nejatishahidin, P. Fayyazsanavi, and J. Kosecka.
 
 
 Object pose estimation using mid-level visual representations.
 
 
 arXiv preprint arXiv:2203.01449 , 2022.
 
 

 
 [54] 
 
A. Newell, K. Yang, and J. Deng.
 
 
 Stacked hourglass networks for human pose estimation.
 
 
 In ECCV , 2016.
 
 

 
 [55] 
 
Y. Nie, X. Han, S. Guo, Y. Zheng, J. Chang, and J. Zhang.
 
 
 Total3dunderstanding: Joint layout, object pose and mesh
reconstruction for indoor scenes from a single image.
 
 
 CoRR , abs/2002.12212, 2020.
 
 

 
 [56] 
 
K. Park, T. Patten, and M. Vincze.
 
 
 Pix2pose: Pixel-wise coordinate regression of objects for 6d pose
estimation.
 
 
 CoRR , abs/1908.07433, 2019.
 
 

 
 [57] 
 
G. Pavlakos, X. Zhou, A. Chan, K. G. Derpanis, and K. Daniilidis.
 
 
 6-dof object pose from semantic keypoints.
 
 
 CoRR , abs/1703.04670, 2017.
 
 

 
 [58] 
 
S. Peng, Y. Liu, Q. Huang, H. Bao, and X. Zhou.
 
 
 Pvnet: Pixel-wise voting network for 6dof pose estimation.
 
 
 2019 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , pages 4556–4565, 2019.
 
 

 
 [59] 
 
C. J. Phillips, M. Lecce, and K. Daniilidis.
 
 
 Seeing glassware: from edge detection to pose estimation and shape
recovery.
 
 
 In Robotics: Science and Systems , 2016.
 
 

 
 [60] 
 
P. Poirson, P. Ammirato, C. Fu, W. Liu, J. Kosecka, and A. C. Berg.
 
 
 Fast single shot detection and pose estimation.
 
 
 CoRR , abs/1609.05590, 2016.
 
 

 
 [61] 
 
P. Poirson, P. Ammirato, C.-Y. Fu, W. Liu, J. Kosecka, and A. C. Berg.
 
 
 Fast single shot detection and pose estimation.
 
 
 2016 Fourth International Conference on 3D Vision (3DV) , pages
676–684, 2016.
 
 

 
 [62] 
 
C. R. Qi, X. Chen, O. Litany, and L. J. Guibas.
 
 
 Imvotenet: Boosting 3d object detection in point clouds with image
votes.
 
 
 CoRR , abs/2001.10692, 2020.
 
 

 
 [63] 
 
C. R. Qi, L. Yi, H. Su, and L. J. Guibas.
 
 
 Pointnet++: Deep hierarchical feature learning on point sets in a
metric space.
 
 
 CoRR , abs/1706.02413, 2017.
 
 

 
 [64] 
 
M. Rad and V. Lepetit.
 
 
 Bb8: A scalable, accurate, robust to partial occlusion method for
predicting the 3d poses of challenging objects without using depth.
 
 
 2017 IEEE International Conference on Computer Vision (ICCV) ,
pages 3848–3856, 2017.
 
 

 
 [65] 
 
S. Ren, K. He, R. B. Girshick, and J. Sun.
 
 
 Faster r-cnn: Towards real-time object detection with region proposal
networks.
 
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
39:1137–1149, 2015.
 
 

 
 [66] 
 
C. Rennie, R. Shome, K. E. Bekris, and A. F. de Souza.
 
 
 A dataset for improved rgbd-based object detection and pose
estimation for warehouse pick-and-place.
 
 
 IEEE Robotics and Automation Letters , 1:1179–1185, 2016.
 
 

 
 [67] 
 
D. Rukhovich, A. Vorontsova, and A. Konushin.
 
 
 Imvoxelnet: Image to voxels projection for monocular and multi-view
general-purpose 3d object detection.
 
 
 CoRR , abs/2106.01178, 2021.
 
 

 
 [68] 
 
O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang,
A. Karpathy, A. Khosla, M. S. Bernstein, A. C. Berg, and L. Fei-Fei.
 
 
 Imagenet large scale visual recognition challenge.
 
 
 International Journal of Computer Vision , 115:211–252, 2015.
 
 

 
 [69] 
 
C. Sahin, G. Garcia-Hernando, J. Sock, and T.-K. Kim.
 
 
 A review on object pose recovery: from 3d bounding box detectors to
full 6d pose estimators.
 
 
 ArXiv , abs/2001.10609, 2020.
 
 

 
 [70] 
 
R. Sajnani, A. Sanchawala, K. M. Jatavallabhula, S. Sridhar, and K. M. Krishna.
 
 
 Draco: Weakly supervised dense reconstruction and canonicalization of
objects.
 
 
 2021 IEEE International Conference on Robotics and Automation
(ICRA) , pages 10302–10309, 2021.
 
 

 
 [71] 
 
R. Sajnani, A. J. Sanchawala, K. M. Jatavallabhula, S. Sridhar, and K. M.
Krishna.
 
 
 DRACO: weakly supervised dense reconstruction and canonicalization
of objects.
 
 
 CoRR , abs/2011.12912, 2020.
 
 

 
 [72] 
 
R. F. Salas-Moreno, R. A. Newcombe, H. Strasdat, P. H. Kelly, and A. J.
Davison.
 
 
 Slam++: Simultaneous localisation and mapping at the level of
objects.
 
 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 1352–1359, 2013.
 
 

 
 [73] 
 
K. Simonyan and A. Zisserman.
 
 
 Very deep convolutional networks for large-scale image recognition.
 
 
 CoRR , abs/1409.1556, 2015.
 
 

 
 [74] 
 
S. Song, S. P. Lichtenberg, and J. Xiao.
 
 
 Sun rgb-d: A rgb-d scene understanding benchmark suite.
 
 
 2015 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , pages 567–576, 2015.
 
 

 
 [75] 
 
S. Song and J. Xiao.
 
 
 Deep Sliding Shapes for amodal 3D object detection in RGB-D
images.
 
 
 2016.
 
 

 
 [76] 
 
H. Su, C. Qi, Y. Li, and L. J. Guibas.
 
 
 Render for cnn: Viewpoint estimation in images using cnns trained
with rendered 3d model views.
 
 
 2015 IEEE International Conference on Computer Vision (ICCV) ,
pages 2686–2694, 2015.
 
 

 
 [77] 
 
X. Sun, J. Wu, X. Zhang, Z. Zhang, C. Zhang, T. Xue, J. B. Tenenbaum, and W. T.
Freeman.
 
 
 Pix3d: Dataset and methods for single-image 3d shape modeling.
 
 
 In IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , 2018.
 
 

 
 [78] 
 
A. Tejani, D. Tang, R. Kouskouridas, and T.-K. Kim.
 
 
 Latent-class hough forests for 3d object detection and pose
estimation.
 
 
 In ECCV , 2014.
 
 

 
 [79] 
 
M. Tian, M. H. Ang, and G. H. Lee.
 
 
 Shape prior deformation for categorical 6d object pose and size
estimation.
 
 
 ArXiv , abs/2007.08454, 2020.
 
 

 
 [80] 
 
S. Tulsiani and J. Malik.
 
 
 Viewpoints and keypoints.
 
 
 2015 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , pages 1510–1519, 2015.
 
 

 
 [81] 
 
S. Umeyama.
 
 
 Least-squares estimation of transformation parameters between two
point patterns.
 
 
 IEEE Trans. Pattern Anal. Mach. Intell. , 13:376–380, 1991.
 
 

 
 [82] 
 
A. Vaswani, N. M. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
L. Kaiser, and I. Polosukhin.
 
 
 Attention is all you need.
 
 
 ArXiv , abs/1706.03762, 2017.
 
 

 
 [83] 
 
A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy,
and S. R. Bowman.
 
 
 Superglue: A stickier benchmark for general-purpose language
understanding systems.
 
 
 CoRR , abs/1905.00537, 2019.
 
 

 
 [84] 
 
H. Wang, S. Sridhar, J. Huang, J. P. C. Valentin, S. Song, and L. J. Guibas.
 
 
 Normalized object coordinate space for category-level 6d object pose
and size estimation.
 
 
 2019 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , pages 2637–2646, 2019.
 
 

 
 [85] 
 
Y. Wang and J. M. Solomon.
 
 
 Deep closest point: Learning representations for point cloud
registration.
 
 
 2019 IEEE/CVF International Conference on Computer Vision
(ICCV) , pages 3522–3531, 2019.
 
 

 
 [86] 
 
J. Wu, B. Zhou, R. Russell, V. Kee, S. Wagner, M. Hebert, A. Torralba, and
D. M. S. Johnson.
 
 
 Real-time object pose estimation with pose interpreter networks.
 
 
 2018 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS) , pages 6798–6805, 2018.
 
 

 
 [87] 
 
Y. Xiang, R. Mottaghi, and S. Savarese.
 
 
 Beyond pascal: A benchmark for 3d object detection in the wild.
 
 
 In IEEE Winter Conference on Applications of Computer Vision
(WACV) , 2014.
 
 

 
 [88] 
 
Y. Xiang, R. Mottaghi, and S. Savarese.
 
 
 Beyond pascal: A benchmark for 3d object detection in the wild.
 
 
 IEEE Winter Conference on Applications of Computer Vision ,
pages 75–82, 2014.
 
 

 
 [89] 
 
H. Yang, C. Shi, Y. Chen, and L. Wang.
 
 
 Boosting 3d object detection via object-focused image fusion.
 
 
 ArXiv , abs/2207.10589, 2022.
 
 

 
 [90] 
 
J. Yang, H. Li, D. Campbell, and Y. Jia.
 
 
 Go-icp: A globally optimal solution to 3d icp point-set registration.
 
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
38:2241–2254, 2016.
 
 

 
 [91] 
 
H. Yisheng, W. Yao, F. Haoqiang, C. Qifeng, and S. Jian.
 
 
 Fs6d: Few-shot 6d pose estimation of novel objects.
 
 
 CVPR , 2022.
 
 

 
 [92] 
 
A. R. Zamir, A. Sax, B. W. Shen, L. J. Guibas, J. Malik, and S. Savarese.
 
 
 Taskonomy: Disentangling task transfer learning.
 
 
 2018 IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 3712–3722, 2018.
 
 

 
 [93] 
 
T. Zhao, Q. Feng, S. Jadhav, and N. A. Atanasov.
 
 
 Corsair: Convolutional object retrieval and symmetry-aided
registration.
 
 
 ArXiv , abs/2103.06911, 2021.
 
 

 
 [94] 
 
Q.-Y. Zhou, J. Park, and V. Koltun.
 
 
 Fast global registration.
 
 
 In ECCV , 2016.