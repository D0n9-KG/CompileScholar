Multiple Sclerosis Lesion Segmentation - A Survey of Supervised CNN-Based Methods 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2012.08317v2 [eess.IV] 26 Dec 2020 
 
 

# Multiple Sclerosis Lesion Segmentation - 
 A Survey of Supervised CNN-Based Methods

 
 
 Huahong Zhang
 
 Affiliation:  Vanderbilt University, Nashville, TN 37235, USA 
 
 E-mail  {huahong.zhang,ipek.oguz}@vanderbilt.edu 
 
    
 Ipek Oguz✉
 

 Abstract 
 
 Lesion segmentation is a core task for quantitative analysis of MRI scans of Multiple Sclerosis patients. The recent success of deep learning techniques in a variety of medical image analysis applications has renewed community interest in this challenging problem and led to a burst of activity for new algorithm development.
In this survey, we investigate the supervised CNN-based methods for MS lesion segmentation. We decouple these reviewed works into their algorithmic components and discuss each separately. For methods that provide evaluations on public benchmark datasets, we report comparisons between their results.

 
 
 
 Keywords:  Multiple Sclerosis Deep Learning Segmentation MRI
 
 

## 1 Introduction

 
 Multiple Sclerosis (MS) is a demyelinating disease of the central nervous system. For monitoring the disease course, focal lesion quantification using magnetic resonance imaging (MRI) is commonly used. Recently, deep learning has achieved great success in the computer vision community [ 44 ] as well as in medical image analysis tasks, and has been applied to MS lesion segmentation [ 14 , 79 , 91 ] . Accurate lesion segmentation is valuable for clinical application and subsequent analysis (e.g., [ 92 ] ).

 
 
 In this survey, we focus on the methods which are CNN-based, and we review about 100 papers that were published until late October 2020. However, we do not intend to review these papers exhaustively.
Instead, we break down the reviewed segmentation pipelines into algorithmic components such as data augmentation and network architecture and compare the representative advances for each component in Sec.  2 .
This is different from the previous surveys of MS segmentation methods (e.g., [ 41 , 19 ] ).

 
 
 Unsupervised learning methods are not included in this survey since an excellent comprehensive survey of these is already provided by Baur et al. [ 8 ] . As a general note, supervised methods tend to perform better than unsupervised methods for MS lesion segmentation, given the difficulty of the task ( [ 14 , 17 ] ).

 
 
 To identify the articles to include in this review, we conducted a Google Scholar search using the keywords “Multiple Sclerosis + lesion + segmentation + neural network”. To ensure we include important advances, we also went through the references cited in each reviewed paper. In addition, papers who cite publicly available MS datasets (e.g., [ 14 , 17 ] ) were considered.

 
 
 

## 2 Review of Methods

 

### 2.1 Data Pre-processing

 
 Pre-processing is a common practice in the reviewed papers. It usually includes skull stripping (e.g., BET [ 73 ] ), bias field correction (e.g., N4ITK [ 75 ] ), rigid registration and intensity normalization.
Rigid registration is used to register between different MRI modalities acquired during a scanning session, between scans of a given patient acquired at different time points, as well as between different subjects of a study or to standard template spaces (e.g., MNI152 [ 27 ] ).

 
 
 For intensity normalization, there are two popular approaches: 1) histogram matching (e.g., [ 78 , 10 ] ) and 2) normalizing data to a specific range. For this latter approach, it is common to enforce zero mean and unit variance (whitening-like, e.g., [ 33 , 79 , 80 ] ) or fit the intensities into the range [0, 1] (e.g., [ 25 , 76 , 31 ] ). Some pipelines (e.g., [ 78 , 10 ] ) also chose to only preserve the values between a range (e.g., 1 st and 99 th percentile) before normalization to minimize the effect of intensity outliers. Further combinations of these two approaches are also possible. Ravnik et al. [ 63 ] argued that in a same-scanner, homogeneous situation, using (whitening-like) intensity normalization, histogram standardization, or both all achieved similar results, and indeed these had no statistically significant improvement over even no normalization at all. But in a multi-scanner, heterogeneous dataset, only using normalization is slightly better than using both, and all performed statistically better than no pre-processing. Other advanced pre-processing techniques (e.g., white stripe [ 72 ] ) can also be considered.

 
 
 

### 2.2 Data Representation

 
 For feeding the deep neural network, the input data are usually represented as patches of raw MRI images since whole images are too large.
These patches can be 2D, 3D, or any format in between.
Choosing the format of patches is an important design decision and will affect the performance of networks. Whether the data is of isotropic resolution needs to be considered for making this decision [ 36 ] .

 
 
 With 2D (slice-based) patches (e.g., [ 66 , 4 , 88 ] ), the advantage is that there are far fewer network parameters to train, and therefore, these tend to be not as prone to over-fitting compared to 3D networks. However, contextual information along the third axis is missing.
In contrast, 3D approaches (e.g., [ 11 , 79 , 32 , 25 ] ) take advantage of the local contextual information, but they do not have any global context. Compared to 2D methods, due to the small size used in 3D patches, the long-distance spatial information cannot be preserved. Further, for processing 3D data, the networks are computationally expensive, need more parameters, and are prone to over-fitting.

 
 

#### Multi-view/2.5D.

 
 To obtain a balance between 2D and 3D representations, Birenbaum et al. [ 10 ] proposed to use multi-view data, in which three orthogonal 2D views passing through the same voxel are input to the network jointly.
On the other hand, Aslani et al. [ 4 ] and Zhang et al.   [ 91 ] used 2.5D by extracting slices from different planes, which are used to train the networks independently from each other. Zhang et al. [ 91 ] also use “stacked slices” by stacking 2D patches along the third axis, which generates thinner but larger 3D input than normal 3D patches. Using 2.5D and prediction fusion, the networks are able to learn and utilize global information along all three axes.

 
 
 
 

### 2.3 Data preparation

 

#### Candidate Extraction.

 
 To extract input patches from the raw data, an intuitive way is to move a sliding window voxel-by-voxel throughout the raw MRI volume. However, this approach will generate a lot of similar patches and leads to class imbalance.
The strategy used to alleviate these problems is slightly different between fully convolutional network (FCN) and non-FCN methods (as discussed in Sec 2.4 ). This distinction is mainly due to the labels of patches for non-FCN methods being single scalars, which are more vulnerable to class imbalance. Furthermore, FCN methods usually solve class imbalance with loss functions.

 
 
 For non-FCN methods,
Birenbaum et al. [ 10 ] proposed to apply a probabilistic WM template and choose the patches whose center is a voxel with high intensity in FLAIR and high probability in WM template. The method does not need lesion maps, so it can be used in both training and test phases to reduce the computation burden.
Valverde et al. [ 79 , 80 ] randomly under-sample the negative class to make a balanced dataset. Kazancli et al. [ 42 ] considered strategies of random sampling and sampling around the lesions, with the latter providing better model performance.
Ulloa et al. [ 76 ] augmented data from the lesion class to balance the dataset instead of down-sampling the non-lesion class. Also, they use circular non-uniform sampling, which allows greater contextual information with a radial extension. They further presented a stratified sampling method [ 77 ] . Among voxels not labeled as lesions, a portion p p of the candidates is extracted from the neighborhood of lesions and the remaining 1 − p 1-p from the remaining voxels.

 
 
 For FCN methods, Kamnitsas et al. [ 37 ] extract the patches with a 50% probability of being centered on a foreground or background voxel. Feng et al. [ 25 ] centered the patches at a lesion voxel with a probability p = 0.99 p=0.99 .
Many methods (e.g., [ 66 , 4 , 91 , 3 ] ) only extract patches with at least one lesion voxel.

 
 
 

#### Augmentation.

 
 After extracting the patches, data augmentation techniques can be applied. Commonly used augmentations include random flip, random rotation, and random scale. Usually, the rotation is 2D and the angle is n × 90 ​ ° ​ ( n ∈ ℤ ) n\times 90\degree(n\in\mathbb{Z}) . However, a random angle (e.g., [ 10 , 3 ] ) or a 3D rotation have also been used. Sometimes random noise and random “bias field” can be added [ 45 ] .
Additionally, Salem et al. [ 68 ] suggested synthesizing lesions may be helpful as an augmentation.

 
 
 

#### Label Processing/Denoising.

 
 For training and evaluating models, the label can be a scalar representing whether the central voxel of the input patch is a lesion or not, or it can be the lesion map the same size as the input, depending on the network type (discussed in Sec.  2.4 ). Either the scalar or the lesion map is extracted from the expert delineations. These unavoidably contain noise, which usually comes from the lesion borders [ 39 ] . Also, some datasets (e.g., [ 14 , 17 ] ) are delineated by more than one expert, and inter-rater variability needs to be addressed. While a simple majority vote can be used, STAPLE [ 84 ] and its variation [ 2 ] are very common.
On the other hand, a few methods (e.g., [ 91 ] ) treat delineations from different experts as different samples. In other words, they train networks with the same input patch but different labels.

 
 
 Even consensus delineations still contain noise that can be further mitigated. Roy et al. [ 66 ] generated training memberships (labels) by convolving the binary segmentations with a 3 × 3 3\times 3 Gaussian kernel to get a softer version of boundaries.
Kats et al. [ 39 ] proposed to assign fixed soft labels to the voxels near the lesions by 3D morphological dilation. Even though they define the soft-Dice loss for this purpose, an easier solution may be to generate the soft version of labels at the data preparation stage. Cohen et al. [ 16 ] further proposed to learn soft labels instead of fixed values, and then apply the soft-STAPLE algorithm [ 40 ] to fuse the masks.

 
 
 

#### Extra Information.

 
 Extra information, for example, spatial locations or intermediate results provided by other models, can also be used as input to CNNs.
Ghafoorian et al. [ 30 ] incorporated eight spatial features, with dense features extracted by convolutional layers and fed into fully connected layers. La Rosa et al. [ 46 ] provided the CNNs with the probability maps generated by a Bayesian partial volume estimation algorithm [ 24 ] .

 
 
 
 

### 2.4 Network Architecture

 
 The network architecture plays a very important role in deep learning. Many works focus on crafting the structure to improve the segmentation performance.

 
 

#### Network Backbones.

 
 After Krizhevsky et al. [ 44 ] won the ImageNet 2012 challenge, Convolutional Neural Networks ( CNN s) became very popular and have been successfully applied to medical imaging problems [ 14 , 17 ] .
For MS lesion segmentation with CNN, the early methods use voxel-wise classification to form the lesion segmentation map [ 10 , 42 , 79 ] . A typical network of this type consists of a few convolutional layers followed by 2-3 fully connected layers (also called Multilayer Perceptron, MLP) for the final prediction.
The input is an image patch, and the networks are trained to classify whether the central voxel of this patch corresponds to a lesion. While these methods have outperformed conventional methods, they have disadvantages: 1) lack of spatial information as only small patches are used; 2) computational inefficiency due to the repetition of similar patches.

 
 
 Kang et al. [ 38 ] introduced the fully convolutional neural network ( FCN /FCNN) for the segmentation task. FCNs do not need to make the lesion prediction by classifying the central voxel of each patch. Instead, they directly generate lesion maps of the same (or similar) size as the input images. However, due to the successive use of convolutional and pooling layers, this approach produces segmentations at a lower resolution. Long et al. [ 51 ] preserve the localization information from low-level features and contextual information from high-level features by adding skip connections. Applications of FCNs for MS lesion segmentation include [ 53 , 11 , 66 ] and some of these methods take advantage of shortcut connections [ 53 , 11 ] .

 
 
 Ronneberger et al. [ 65 ] then used a symmetrical u-shape network called U-Net to combine features. The network has an encoder-decoder structure and adds shortcut connections between corresponding layers of the two parts.
The pooling operations in the encoder are replaced by upsampling operations in the decoder path. Many recent MS segmentation methods [ 67 , 4 , 25 ] are based on the original U-Net or slight modifications thereof.

 
 
 

#### Variations of U-Net.

 
 For CNNs with voxel-wise prediction or FCN methods that are not U-Net-based, the network structures are quite flexible. As for pipelines that use the U-Net, the most common modification is to introduce some crafted modules (residual block, dense block, attention module, etc.). These new modules can be added to replace the convolutions or between the shortcut connections.
Aslani et al. [ 4 , 3 ] presented a U-Net-like structure with convolutions replaced by residual blocks in the encoder path.
Hashemi et al. [ 32 ] and Zhang et al. [ 91 ] adopted the Tiramisu network, which replaces the convolution layers with densely connected blocks (skip connection between layers).
Hu et al. [ 35 ] presented a context-guided module to expand the perception field and utilize contextual information.
Vang et al. [ 81 ] augmented the U-Net with the Mask R-CNN framework.

 
 
 Attention Module. Attention mechanism has been researched in many works for MS lesion segmentation. In general, it can be divided into spatial attention, channel attention, and longitudinal attention.
Zhang et al. [ 89 ] presented a recurrent slice-wise attention network for 3D CNNs. By performing slice-wise attention from three orientations, they reduced the memory demands for 3D spatial attention.
Hu et al. [ 35 ] included 3D spatial attention blocks in the decoding stage.
Durso-Finley et al. [ 23 ] used a saliency-based attention module before the U-Net structure to make the network realize the difference between pre- and post-contrast T1-w images and thus focus on the contrast-enhancing lesions.
Hou et al. [ 34 ] proposed a cross-attention block that combines spatial attention and channel attention.
Zhang et al. [ 90 ] extend their folded (slice-wise) attention to the spatial-channel attention module. They use four permutations (corresponding to four dimensions) to build four small sub-afﬁnity matrices to approximate the original affinity matrix. In such a case, the original affinity matrix is regularized as a rank-one matrix and the computational burden is alleviated.
Gessert et al. [ 29 ] introduced attention-guided interactions to enable effective information exchange between the processing paths of the two time points.

 
 
 

#### Multi-task Networks.

 
 Narayana et al. [ 58 ] performed segmentation of brain tissue and T2 lesions at the same time. McKinley et al. [ 54 ] illustrated that the inclusion of additional tissue classes during the segmentation of lesions is helpful for MS segmentation. Duong et al. [ 22 ] trained the networks with data from many different tasks, making the trained CNN usable for multiple tasks without tuning and thus more applicable in a clinical context.

 
 
 
 

### 2.5 Multiple Modalities, Timepoints, Views and Scales

 
 In the context of MS lesion segmentation, the incorporation of multi-modality, multi-timepoints, multi-view and multi-scale data are similar. The data from different sources have to be fused at some point of the pipeline: input, feature map, and/or output. Fusing at the input can be simple concatenation along channels while fusing the output is roughly equivalent to making an ensemble to reach consensus. Fusing the features usually needs parallel paths and interaction between paths, which typically happens in the encoder path in a U-Net-like structure.

 
 

#### Multi-modalities.

 
 The commonly used MRI sequences for MS white matter lesion segmentation include T1-weighted (T1-w), T2-weighted (T2w), proton density-weighted (PD-w) and fluid attenuated inversion recovery T2 (FLAIR).

 
 
 Narayana et al. [ 59 ] evaluated the performance of U-Net when it is trained with different combinations of modalities on a large multi-center dataset. They concluded that using all the modalities, especially with FLAIR, achieved the best performance. A similar conclusion can be found in [ 11 ] and other works that use multiple MRI sequences as input. For fusing the different modalities,
Roy et al. [ 66 ] use parallel pathways for processing different modalities and then concatenate the features along the channels (only once). Aslani et al. [ 4 ] use parallel encoder paths to process different modalities, and they fuse the different modalities after each convolutional block.
Zhang et al. [ 88 ] also use a similar strategy.
Zhang et al. [ 91 ] fuse the patches from different sequences before feeding into the network.

 
 
 Multi-modality methods are usually trained on a specific set of modalities and thus require these sequences to be available at the test phase, which can be limiting.
To deal with missing modalities, Havaei et al. [ 33 ] propose to use parallel CNN paths to learn the embeddings of different input sequences into a single latent space for which arithmetic operations are well deﬁned.
They randomly drop modalities during training.
As such, any subset of available modalities can be used as input at test time.
Feng et al. [ 25 ] also use random “dropout” of modalities but substitute the missing modalities with background values.

 
 
 

#### Multi-timepoints.

 
 Longitudinal studies are common in MS, but the ongoing inflammatory disease activity complicates the analysis of longitudinal MRIs as new lesions can appear and existing lesions can heal between scans. To improve individual segmentation performance , Birenbaum et al. [ 10 ] propose to process the two time-points individually with a Siamese architecture, where the parallel paths share weights, and then concatenate the features for classification.
Denner et al. [ 20 ] argue that this late-fusion strategy [ 10 ] does not properly take advantage of learning from structural changes and they propose two complementary networks for multi-timepoints. The longitudinal network fuses the two time-points early to implicitly use the structural differences. The multi-task network is trained to learn the segmentation with an additional task of deformable registration between the two time-points, which explicitly guides the network to use spatio-temporal information.

 
 
 To identify lesion activity , Placidi et al. [ 62 ] simply segment the lesions at the two time-points independently and register the previous examination to the current examination to compare the segmentations. However, comparing differences between time-points relies on high similarity between scans and requires highly accurate registration.
McKinley et al. [ 55 ] introduce a segmentation confidence for comparing the lesion segmentation between timepoints. Comparisons are based on the “confident” lesions of each timepoint.
Kruger et al. [ 45 ] fed two timepoints into the same encoder (share weights) and the feature maps are concatenated after each residual block before going to the corresponding decoder block.
Salem et al. [ 69 ] use cascaded networks for detecting new lesions. The first network learns the deformation field between the baseline and follow-up images. The second network takes the two images and the deformation field, and outputs the segmentation.
To assist the network in learning to detect new and enlarging T2w lesions, Sepahvand et al. [ 70 ] illustrate an attention-like mechanism. They multiply the multi-modal MRI at the reference (in contrast to follow-up) with subtraction images, which acts as the attention gate. Then the product is concatenated with the lesion map at the reference to feed the network.
Gessert et al. [ 28 ] propose convolutional gated recurrent units for temporal aggregation. The units are inserted into the bottleneck and skip connections of U-Net. In another work [ 29 ] , the same team process two timepoints with parallel encoder paths that interact with each other using attention modules. In such a scenario, the attention mechanism functions similarly to masking early time-points.

 
 
 

#### Multi-views.

 
 As discussed in Sec. 2.2 , multi-view data can be utilized as data representation between 2D and 3D. To handle multi-view data, Birenbaum et al. [ 10 ] processed different views by parallel sub-networks and then concatenated the features to feed the fully connected layers.
McKinley et al. [ 53 ] integrate three networks for the three views and the outputs are averaged.
Zhang et al. [ 91 ] use one network for different views, i.e., the network parameters are shared between views.
Shachor et al. [ 71 ] propose a gated model Mixture of Views (MoV) to fuse different views.

 
 
 

#### Multi-scales.

 
 The networks based on FCN and U-Net inherently incorporate multiple scales. However, explicit use of multi-scales may also be useful. For non-FCN networks, Kamnitsas et al. [ 37 ] propose to use two parallel paths, one for full resolution and another for a lower resolution; these two paths are fused before fully connected layers.
As for U-Net-based methods, Wang et al. [ 83 ] argue that different types of segmentation biases may be generated by networks of different input sizes. To address this issue, they train 3 networks with different input sizes and use another stage for fusing the results. Hou et al. [ 34 ] feed multi-scale input (original and downsampled patches) to the first three layers and they aggregate the multi-scale outputs from the last three layers to make the final prediction.
Hu et al. [ 35 ] use one input and average the multi-scale outputs.

 
 
 

#### Others.

 
 For training data with annotations from multiple experts, Vaidya et al. [ 78 ] train two separate networks using the same images but different delineations from two experts. The outputs of the two networks are averaged to get the final prediction. Zhang et al. [ 93 ] present a segmentation network to estimate the ground truth segmentation and an annotator network to estimate the characteristics of each expert, which can be viewed as a translation of STAPLE to CNN.

 
 
 No-new-UNet (nnU-Net) [ 36 ] is a multi-architecture framework that adaptively chooses an ensemble from 2D, 3D, 3D cascaded networks.

 
 
 
 

### 2.6 Loss Functions and Regularization

 
 For MS lesions segmentation, which is usually binary, the most commonly used loss function is the Binary Cross-Entropy ( BCE , e.g., [ 10 , 3 ] ).
Other losses such as L2 loss (e.g., [ 11 , 91 ] ) are also explored. As class imbalance exists within the dataset, the original losses can be weighted based on the probability (prevalence) of each class. The class with the lower probability (i.e., lesion) is compensated with a higher weight. Brosch et al. [ 11 ] implicitly weight lesion voxels and non-lesion voxels by calculating the weighted L2 loss of the lesion voxels and non-lesion voxels. Feng et al. [ 25 ] used weighted BCE with a lesion/non-lesion ratio of 3 to 1. Focal loss [ 50 ] , as the generalization of BCE, was proposed not only to weight the lesion class but also to give more importance to hard examples (e.g., [ 91 , 32 ] ).

 
 
 For FCN-based methods, since the labels/outputs are patches, region-based losses are used to address the intra-patch class imbalance.
Milletari et al. [ 56 ] proposed the Dice ( F 1 F_{1} ) loss to balance between precision and recall equally.
 Tversky loss [ 67 , 32 ] is the generalization of the Dice loss and F β F_{\beta} loss (such that β = 1 \beta=1 is the Dice loss).
The networks trained with higher β \beta have higher recall and lower precision. Based on this property, Ma et al. [ 52 ] trained individual models with high β \beta to make up an ensemble. Their assumption is that diverse low-precision-high-recall models tend to make different false-positive errors but similar consistent true-positives. Thus the false-positive errors can be canceled out by aggregating the predictions of the ensemble.
Further, the Focal Tversky loss (e.g., [ 35 ] ) is the generalization of the Tversky loss. It is similar to Focal loss, which is capable of focusing on mislabeled samples and minority samples by controlling the parameters.

 
 
 Combining loss functions and introducing domain-specific regularization are helpful in some cases.
McKinley et al. [ 53 ] calculated the 25 th percentile of the intensity within the lesion mask and weighted the loss function from these voxels higher than others.
Zhang et al. [ 88 ] proposed to use Generative Adversarial Network (GAN) architecture to provide an additional discriminator-based constraint. They use a combination of BCE loss, Dice loss, L1 loss and GAN-loss.
Additionally, loss functions have been proposed to help the networks with uncertainty analysis [ 54 ] , domain adaptation [ 6 , 1 , 3 ] and other goals.

 
 
 

### 2.7 Implementation

 
 To train the networks, a simple strategy is using fixed epochs. However, early stopping is a common practice to avoid over-fitting. Thus, the training dataset is divided into a fixed training subset and validation subset, or k-fold cross-validation can be utilized [ 10 , 32 , 91 ] . The “best” models are chosen based on model performance (e.g., Dice score) on the validation set. To provide a fair evaluation, the test set is usually held-out from the training/validation data and different scans of the same patient should not be placed into different datasets.

 
 
 For optimizing the parameters, stochastic gradient descent (SGD) and its variations are used. To avoid oscillation in local optima, the momentum variable was introduced (e.g., [ 35 ] .). Further, Nesterov accelerated gradient was proposed to have some prescience about the next update direction (e.g., [ 33 ] ). To adapt the learning rate to the parameters, Adagrad [ 21 ] was proposed. However, Adagrad accumulates the squared gradients in the denominator and leads to monotonically decreasing learning rate. RMSprop and Adadelta [ 87 ] are proposed to resolve Adagrad’s radically diminishing learning rates. Adaptive Moment Estimation (Adam) [ 43 ] is an optimizer with momentum and adaptive learning rates. AMSGrad [ 64 ] is a variation of Adam. For MS lesion segmentation, Adadelta (e.g., [ 10 , 11 , 79 , 80 , 3 , 53 ] ) and Adam (e.g., [ 42 , 66 , 4 , 25 ] ) are most widely used.

 
 
 

### 2.8 Prediction and Post-processing

 

#### Prediction.

 
 For non-FCN methods, segmentation is made by classifying all the candidates extracted from the test image voxel-by-voxel. For FCN methods, 2D networks predict the segmentation slices-by-slice, and 3D methods are able to predict the whole image at once.
However, Hashemi et al. [ 32 ] argue border predictions made by FCN are not as accurate as center voxel predictions, so they propose to predict the results patch-by-patch and then fuse predictions using B-spline weighted soft voting, such that border predictions are given lower weights.
For some methods where data from multiple sources (e.g., views) is used or multiple models are trained, a label aggregation step is necessary. Aslani et al. [ 3 , 4 ] and Zhang et al. [ 91 ] use majority vote to aggregate labels, but other methods (e.g., STAPLE [ 84 ] ) can also be considered.

 
 
 Since the outputs of networks are usually soft predictions (i.e., in the range [0, 1]) indicating the probability of being lesions, the simplest way to get hard predictions is to use a threshold of 0.5 [ 67 ] . However, McKinley et al. [ 55 ] argue that the scores output by deep networks do not correspond to observed probabilities and are typically overconfident. Brosch et al. [ 11 ] and Roy et al. [ 66 ] attempt to choose the optimal threshold by maximizing the Dice on the training set.

 
 
 

#### Post-processing.

 
 Many methods attempt to remove false positives from the hard segmentation using post-processing strategies.
A common post-processing approach consists of discarding lesions smaller than a volume threshold (e.g., [ 12 ] ). Vaidya et al. [ 78 ] use pre-built brain templates to remove predicted lesions outside of white matter, but this strategy is problematic for detecting cortical lesions. Kamnitsas et al. [ 37 ] use an additional stage of machine learning for post-processing, specifically, a fully-connected Conditional Random Field (CRF), a common strategy.
Valverde et al. [ 79 , 80 ] use cascaded networks, in which the first network is trained to recall many possible lesions and the second network refines the output of the first network. Specifically, the training data for the second model is balanced between all the lesion voxels and the random selection of misclassified lesion voxels on the first model. Others [ 42 , 86 ] use similar strategies.

 
 
 Nair et al. [ 57 ] present multiple uncertainty estimates based on Monte Carlo (MC) dropout. Then lesions with high uncertainty can be removed. Their results suggest that uncertainty measures allow choosing superior operating points, compared to only using the network’s sigmoid output as a probability.

 
 
 
 

### 2.9 Transfer Learning and Domain Adaptation

 
 Transfer learning is an active topic in deep learning, and in the context of MS lesion segmentation, it includes two aspects: 1) using pre-trained models from other domains; 2) applying trained model on different MS datasets (e.g., for clinical use). The latter scenario is also considered as domain adaptation, in which the target task remains the same as the source (i.e., MS lesion segmentation) but the domains (i.e., MRI protocols and therefore image appearance) are different. Multi-task training (Sec. 2.4 ) is also a form of transfer learning.

 
 

#### Pre-training with Other Domains.

 
 The pre-trained blocks (layers) from other domains are typically used to replace the encoder.
Brosch et al. [ 11 ] propose to pre-train the model layer-by-layer with convolutional restricted Boltzmann machines and then apply parameters on both encoder and decoder. Aslani et al. [ 4 , 3 ] use the ResNet50 pre-trained on ImageNet as the encoder. Fenneteau et al. [ 26 ] present a self-supervision method to pre-train the encoder to predict the location of an input patch. However, their results illustrate that pre-training is not helpful. Kruger et al. [ 45 ] pre-train the encoder path with single time point data and then train the entire network with longitudinal data.

 
 
 

#### Generalization of Trained Models.

 
 For domain adaptation when a few labeled images are available in the target (new) domain, Ghafoorian et al. [ 31 ] propose to freeze the first few layers of the model trained on the source domain and fine-tune the last few layers. Their results show that using even just 2 images can achieve a good Dice score. Valverde et al. [ 80 ] propose to freeze all convolutional layers and fine-tune the fully connected layers. A single image for re-training could generate segmentation with human-level performance. Weeda et al. [ 85 ] further test the one-shot learning proposed by [ 80 ] with an independent dataset, and the performance is better than unsupervised methods and is comparable to fully trained supervised methods.

 
 
 For domain adaptation without any labeled data in the target domain, Baur et al. [ 6 ] propose to add auxiliary manifold embedding loss for utilizing unlabeled data from target domains. The idea is that latent feature vectors that share the same label (for labeled data) or same noisy prior (unlabeled data) should be similar, and otherwise differ from each other. Baur et al. [ 7 ] propose to train an auto-encoder for unsupervised anomaly detection in the target domain, and use this unsupervised model to generate artificial labels for jointly training a supervised model with labeled data from the source domain.
Ackaouy et al. [ 1 ] propose a method to perform unsupervised domain adaptation with optimal transport. In the deep learning context, their strategy is implemented as two losses, which ensures the heavily connected source samples and target samples to have similar representations in the latent space and the output, while maintaining good segmentation performance.

 
 
 Billast et al. [ 9 ] present domain adaptation with adversarial training. The discriminator is trained to discriminate whether the two input segmentations are from the same scanner, so that the generator learns to map scans from different scanners to the same latent space and thus produce a consistent lesion segmentation.
Aslani et al. [ 5 ] propose a similar idea with a regularization network predicting the feature domain. They use a combination loss of Pearson correlation, randomized cross-entropy and discrete uniform to encourage the latent features to be domain agnostic.
Varsavsky et al. [ 82 ] combine domain adversarial learning and consistency regularization, which enforces invariance to data augmentation.

 
 
 
 

### 2.10 Methods for Subtypes of MS Lesions

 
 Most of the methods we have discussed are proposed to segment white matter lesions. Among these, some pipelines focus on detecting contrast-enhancing (CE) lesions since these are indicative of active disease. Gadolinium (Gad) is commonly used in the context of MS.
Durso-finley et al. [ 23 ] and Coronado et al. [ 18 ] present to detect Gad lesions using pre- and post-contrast T1-weighted images.
Brugnara et al. [ 12 ] propose a network to detect both CE lesions and T2/FLAIR-hyperintense lesions and report the performance separately.
On the other hand, radiological monitoring of disease progression also requires detecting new and enlarging T2w lesions, which can be explored by longitudinal approaches (Sec.  2.5 , multi-timepoints).
La Rosa et al. [ 46 ] propose to detect the early stage lesions by combining deep neural networks with a shallow model (supervised k-NN with partial volume modeling).

 
 
 Cortical lesions are also important in MS [ 13 ] . La Rosa et al. [ 48 ] use simplified U-Net to detect both cortical and white matter lesions at 3T MRI.
To achieve this, they utilize 3T 3D-FLAIR and magnetization-prepared 2 rapid acquisition with gradient echo (MP2RAGE).
They further use 7T MRI (7T MP2RAGE, T2*w echo planar imaging, T2*w gradient recalled echo) for cortical lesion segmentation, which has higher resolution and SNR than 3T [ 47 ] .

 
 
 
 

## 3 Comparison of Experiments and Results

 

### 3.1 Datasets

 

#### Public Datasets.

 
 Currently, the challenge datasets, including the MICCAI 2008 1 1 
 1 
 
 
 
 http://www.ia.unc.edu/MSseg [ 74 ] , ISBI 2015 2 2 
 2 
 
 
 
 https://smart-stats-tools.org/lesion-challenge [ 14 ] , MICCAI (MSSEG) 2016 3 3 
 3 
 
 
 
 https://portal.fli-iam.irisa.fr/msseg-challenge/overview [ 17 ] challenges, are widely used. The dataset descriptions can be found on the respective websites. The first two challenges are still (as of November 2020) accepting segmentation submissions on their test dataset, which provides objective comparisons between state-of-the-art MS segmentation methods.
Lesjak et al. [ 49 ] provided a novel public dataset 4 4 
 4 
 
 
 
 http://lit.fe.uni-lj.si/tools , for which three expert raters performed segmentation of WM lesions and reached consensus by several joint sessions.
They illustrated that the consensus-based segmentation have better consistency than a single rater’s segmentation.
It is worth noting that all these public datasets only delineate white matter lesions.

 
 
 

#### Private Datasets.

 
 Using private datasets makes it difficult to compare algorithms but has the advantage of including more subjects than currently available in public datasets.
Some proprietary datasets can be of a quite large scale (e.g., 6830 multi-channel MRI [ 23 ] ). For such large datasets, the “ground truth” labels are usually created by automated or semi-automated algorithms and corrected by experts [ 46 , 23 ] .

 
 
 Narayana et al. [ 60 ] considered the effect of training data size for training the neural networks. They argue that at least 50 image volumes are necessary to get a meaningful segmentation. But this work does not mention the data augmentations and other advanced techniques for training a network. Based on the results of the ISBI 2015 challenge [ 14 ] , human-level performance can be achieved by state-of-the-art algorithms with only about 20 images for training.

 
 
 
 

### 3.2 Evaluation Metrics

 
 In the task of MS lesion segmentation, the commonly reported metrics include:
Dice similarity coefficient (DSC), Jaccard coefficient, absolute volume difference (AVD), average symmetric surface distance (ASSD/SD), true positives rate (TPR, sensitivity, recall), false positives rate (FPR), positive predictive value (PPV, Precision), lesion-wise true positives rate (LTPR) and false positives rate (LFPR).

 
 
 The above metrics are individually calculated based on each image. Then, the results are aggregated and reported. In addition to reporting mean and standard deviation values, the Wilcoxon signed-rank test is used to statistically test performance differences between methods. Precision-Recall (PR) curve is suitable for evaluating the performance of the highly unbalanced dataset. Receiver Operating Characteristic (ROC) curve is also used (e.g., [ 57 ] ). The area under curve (AUC) for these curves is a common aggregation metric.
Further, considering the relationship between the model performance and lesion volume, some works divide lesions into groups of different sizes and calculated the metrics (e.g., [ 83 ] ). Volumes of lesions estimated and the ground truth segmentation can be shown in the correlation (e.g., [ 79 , 42 , 66 , 4 , 22 ] ) and Bland-Altman (e.g., [ 46 , 54 ] ) plots.
A more systematic analysis of algorithm performance can differentiate between correctly detected lesions, nearby lesions merged into one or a single lesion split into many, as well as characterize the performance as a function of lesion size [ 61 , 15 ] .

 
 
 

### 3.3 Results

 
 As previously mentioned, MICCAI 2008 [ 74 ] and ISBI 2015 [ 14 ] are still accepting submissions and providing the evaluation results on the test dataset, thus serving as objective benchmarks. In this survey, we compare the state-of-the-art methods that have evaluated their performance on these datasets in Table 2 and Table 1 .

 
 
 Table 1: Results on the ISBI 2015 challenge test set. All metrics in percent. DSC: Dice; PPV: Precision; TPR: true positives rate; LTPR: lesion-wise TPR; LFPR: lesion-wise false positives rate; VD: volume difference; SC: total weighted score of other metrics. Code: links to code repositories,if available. 
 
 
 
 | 
 SC | 
 DSC | 
 PPV | 
 TPR | 
 LFPR | 
 LTPR | 
 VD | 
 Code | 

 
 Zhang et al. [ 91 ] | 
 93.21 | 
 64.3 | 
 90.8 | 
 53.3 | 
 12.4 | 
 52.0 | 
 42.8 | 
 Yes | 

 
 Isensee et al. [ 36 ] | 
 92.87 | 
 67.9 | 
 84.7 | 
 60.5 | 
 15.9 | 
 52.2 | 
 36.8 | 
 Yes | 

 
 Hu et al. [ 35 ] | 
 92.61 | 
 63.4 | 
 86.9 | 
 52.6 | 
 13.4 | 
 48.2 | 
 39.7 | 
 No | 

 
 Hashemi et al. [ 32 ] | 
 92.49 | 
 58.4 | 
 92.1 | 
 45.6 | 
 8.7 | 
 41.3 | 
 49.7 | 
 No | 

 
 Feng et al. [ 25 ] | 
 92.41 | 
 68.2 | 
 78.2 | 
 64.5 | 
 27.0 | 
 60.0 | 
 32.6 | 
 No | 

 
 Denner et al. [ 20 ] | 
 92.12 | 
 64.3 | 
 85.9 | 
 54.5 | 
 19.5 | 
 47.1 | 
 38.6 | 
 Yes | 

 
 Aslani et al. [ 4 ] | 
 92.12 | 
 61.1 | 
 89.9 | 
 49.0 | 
 13.9 | 
 41.0 | 
 45.4 | 
 No | 

 
 Valverde et al. [ 79 ] | 
 91.33 | 
 63.0 | 
 78.7 | 
 55.5 | 
 15.3 | 
 36.7 | 
 33.8 | 
 Yes | 

 
 Roy et al. [ 66 ] | 
 90.48 | 
 52.4 | 
 86.6 | 
 N/A | 
 11.0 | 
 N/A | 
 52.1 | 
 Yes | 

 
 Valverde et al. [ 80 ] 5 5 
 5 
 
 
 
 trained on other datasets and fine-tune with one sample from this dataset. | 
 90.32 | 
 57.7 | 
 83.1 | 
 47.5 | 
 18.9 | 
 29.7 | 
 44.6 | 
 Yes | 

 
 Birenbaum et al. [ 10 ] | 
 90.07 | 
 62.7 | 
 78.9 | 
 55.5 | 
 49.8 | 
 56.8 | 
 35.2 | 
 No | 

 

 
 
 Table 2: Results on the MICCAI 2008 challenge test set. Subscript 1: UNC Rater; 2: CHB Rater. All metrics in percent, except SD in millimeters. VD: volume difference; SD: surface distance; TPR: true positives rate; FPR: false positives rate; SC: total weighted score of other metrics. 
 
 
 
 | 
 SC | 
 VD 1 | 
 SD 1 | 
 TPR 1 | 
 FPR 1 | 
 VD 2 | 
 SD 2 | 
 TPR 2 | 
 FPR 2 | 

 
 
 
 Valverde et al. [ 79 ] | 
 87.1 | 
 62.5 | 
 5.8 | 
 55.5 | 
 46.8 | 
 40.8 | 
 5.2 | 
 68.7 | 
 46.0 | 

 
 Brosch et al. [ 11 ] | 
 84.0 | 
 63.5 | 
 7.4 | 
 47.1 | 
 52.7 | 
 52.0 | 
 6.4 | 
 56.0 | 
 49.8 | 

 
 Havaei et al. [ 33 ] | 
 83.2 | 
 127 | 
 7.5 | 
 66.1 | 
 55.3 | 
 68.2 | 
 6.6 | 
 52.3 | 
 61.3 | 

 

 
 
 From the results, we observe that 3D and 2.5D methods seem to outperform 2D approaches with the development of GPUs. As in Table 1 , U-Net-based methods ( [ 91 , 36 , 35 , 32 , 20 , 4 ] ) tend to perform better than non-FCN CNN-based ( [ 10 , 79 ] ) and non-U-Net FCN-based ( [ 66 ] ) methods.

 
 
 
 

## 4 Conclusion

 
 In this survey, we explored the advances in different components of supervised CNN MS lesion segmentation methods. Among these, topics including attention mechanism, network designs to combine information from multiple sources, loss functions to handle class imbalance, and domain adaptation are of interest for many researchers.

 
 

#### Acknowledgements.

 
 This work was supported, in part, by the NIH grant R01-NS094456 and National Multiple Sclerosis Society award PP-1905-34001.

 
 
 
 

## References

 
 
 [1] 
 
Ackaouy, A., et al.: Unsupervised domain adaptation with optimal transport in
multi-site segmentation of multiple sclerosis lesions from mri data. Front
Comput Neurosci (2020)

 

 
 [2] 
 
Akhondi-Asl, A., et al.: A logarithmic opinion pool based staple algorithm for
the fusion of segmentations with associated reliability weights. IEEE Trans
Med Imaging (2014)

 

 
 [3] 
 
Aslani, S., et al.: Deep 2d encoder-decoder convolutional neural network for
multiple sclerosis lesion segmentation in brain mri. In: BrainLes (2018)

 

 
 [4] 
 
Aslani, S., et al.: Multi-branch convolutional neural network for multiple
sclerosis lesion segmentation. NeuroImage (2019)

 

 
 [5] 
 
Aslani, S., et al.: Scanner invariant multiple sclerosis lesion segmentation
from mri. In: ISBI (2020)

 

 
 [6] 
 
Baur, C., et al.: Semi-supervised deep learning for fully convolutional
networks. In: MICCAI (2017)

 

 
 [7] 
 
Baur, C., et al.: Fusing unsupervised and supervised deep learning for white
matter lesion segmentation. In: MIDL (2019)

 

 
 [8] 
 
Baur, C., et al.: Autoencoders for unsupervised anomaly segmentation in brain
mr images: A comparative study. arXiv:2004.03271 (2020)

 

 
 [9] 
 
Billast, M., et al.: Improved inter-scanner ms lesion segmentation by
adversarial training on longitudinal data. In: BrainLes (2019)

 

 
 [10] 
 
Birenbaum, A., Greenspan, H.: Longitudinal multiple sclerosis lesion
segmentation using multi-view convolutional neural networks. In: LABELS
(2016)

 

 
 [11] 
 
Brosch, T., et al.: Deep 3d convolutional encoder networks with shortcuts for
multiscale feature integration applied to multiple sclerosis lesion
segmentation. IEEE Trans Med Imaging (2016)

 

 
 [12] 
 
Brugnara, G., et al.: Automated volumetric assessment with artificial neural
networks might enable a more accurate assessment of disease burden in
patients with multiple sclerosis. Eur Radiol (2020)

 

 
 [13] 
 
Calabrese, M., et al.: Cortical lesions and atrophy associated with cognitive
impairment in relapsing-remitting multiple sclerosis. Archives of neurology
(2009)

 

 
 [14] 
 
Carass, A., et al.: Longitudinal multiple sclerosis lesion segmentation:
resource and challenge. NeuroImage (2017)

 

 
 [15] 
 
Carass, A., et al.: evaluating white matter lesion segmentations with refined
sørensen-dice analysis. Scientific Reports (2020)

 

 
 [16] 
 
Cohen, G., et al.: Learning probabilistic fusion of multilabel lesion contours.
In: ISBI (2020)

 

 
 [17] 
 
Commowick, O., et al.: Objective evaluation of multiple sclerosis lesion
segmentation using a data management and processing infrastructure. Sci Rep
(2018)

 

 
 [18] 
 
Coronado, I., et al.: Deep learning segmentation of gadolinium-enhancing
lesions in multiple sclerosis. Multiple Sclerosis Journal (2020)

 

 
 [19] 
 
Danelakis, A., et al.: Survey of automated multiple sclerosis lesion
segmentation techniques on magnetic resonance imaging. Comput Med Imaging
Graph (2018)

 

 
 [20] 
 
Denner, S., et al.: Spatio-temporal learning from longitudinal data for
multiple sclerosis lesion segmentation. arXiv:2004.03675 (2020)

 

 
 [21] 
 
Duchi, J., et al.: Adaptive subgradient methods for online learning and
stochastic optimization. J Mach Learn Res (2011)

 

 
 [22] 
 
Duong, M.T., et al.: Convolutional neural network for automated flair lesion
segmentation on clinical brain mr imaging. AJNR Am J Neuroradiol (2019)

 

 
 [23] 
 
Durso-Finley, J., et al.: Saliency based deep neural network for automatic
detection of gadolinium-enhancing multiple sclerosis lesions in brain mri.
In: BrainLes (2019)

 

 
 [24] 
 
Fartaria, M.J., et al.: Segmentation of cortical and subcortical multiple
sclerosis lesions based on constrained partial volume modeling. In: MICCAI
(2017)

 

 
 [25] 
 
Feng, Y., et al.: A self-adaptive network for multiple sclerosis lesion
segmentation from multi-contrast mri with various imaging sequences. In: ISBI
(2019)

 

 
 [26] 
 
Fenneteau, A., et al.: Learning a cnn on multiple sclerosis lesion segmentation
with self-supervision. In: IS T Electronic Imaging 2020 Symposium (2020)

 

 
 [27] 
 
Fonov, V.S., et al.: Unbiased nonlinear average age-appropriate brain templates
from birth to adulthood. NeuroImage (2009)

 

 
 [28] 
 
Gessert, N., et al.: 4d deep learning for multiple sclerosis lesion activity
segmentation. arXiv:2004.09216 (2020)

 

 
 [29] 
 
Gessert, N., et al.: Multiple sclerosis lesion activity segmentation with
attention-guided two-path cnns. Comput Med Imaging Graph (2020)

 

 
 [30] 
 
Ghafoorian, M., et al.: Location sensitive deep convolutional neural networks
for segmentation of white matter hyperintensities. Sci Rep (2017)

 

 
 [31] 
 
Ghafoorian, M., et al.: Transfer learning for domain adaptation in mri:
Application in brain lesion segmentation. In: MICCAI (2017)

 

 
 [32] 
 
Hashemi, S.R., et al.: Asymmetric loss functions and deep densely-connected
networks for highly-imbalanced medical image segmentation: Application to
multiple sclerosis lesion detection. IEEE Access (2018)

 

 
 [33] 
 
Havaei, M., et al.: Hemis: Hetero-modal image segmentation. In: MICCAI (2016)

 

 
 [34] 
 
Hou, B., et al.: Cross attention densely connected networks for multiple
sclerosis lesion segmentation. In: BIBM (2019)

 

 
 [35] 
 
Hu, C., et al.: Acu-net: A 3d attention context u-net for multiple sclerosis
lesion segmentation. In: ICASSP (2020)

 

 
 [36] 
 
Isensee, F., et al.: nnu-net: Breaking the spell on successful medical image
segmentation. arXiv:1904.08128 (2019)

 

 
 [37] 
 
Kamnitsas, K., et al.: Efficient multi-scale 3d cnn with fully connected crf
for accurate brain lesion segmentation. Med Image Anal (2017)

 

 
 [38] 
 
Kang, K., Wang, X.: Fully convolutional neural networks for crowd segmentation.
arXiv:1411.4464 (2014)

 

 
 [39] 
 
Kats, E., et al.: Soft labeling by distilling anatomical knowledge for improved
ms lesion segmentation. In: ISBI (2019)

 

 
 [40] 
 
Kats, E., et al.: A soft staple algorithm combined with anatomical knowledge.
In: MICCAI (2019)

 

 
 [41] 
 
Kaur, A., et al.: State-of-the-art segmentation techniques and future
directions for multiple sclerosis brain lesions. Arch Comput Methods Eng
(2020)

 

 
 [42] 
 
Kazancli, E., et al.: Multiple sclerosis lesion segmentation using improved
convolutional neural networks. In: VISIGRAPP (2018)

 

 
 [43] 
 
Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization.
arXiv:1412.6980 (2014)

 

 
 [44] 
 
Krizhevsky, A., et al.: Imagenet classification with deep convolutional neural
networks. In: NIPS (2012)

 

 
 [45] 
 
Krüger, J., et al.: Fully automated longitudinal segmentation of new or
enlarging multiple scleroses (ms) lesions using 3d convolution neural
networks. In: RöFo-Fortschritte auf dem Gebiet der Röntgenstrahlen
und der bildgebenden Verfahren (2020)

 

 
 [46] 
 
La Rosa, F., et al.: Shallow vs deep learning architectures for white matter
lesion segmentation in the early stages of multiple sclerosis. In: BrainLes
(2018)

 

 
 [47] 
 
La Rosa, F., et al.: Automated detection of cortical lesions in multiple
sclerosis patients with 7t mri. arXiv:2008.06780 (2020)

 

 
 [48] 
 
La Rosa, F., et al.: Multiple sclerosis cortical and wm lesion segmentation at
3t mri: a deep learning method based on flair and mp2rage. Neuroimage Clin
(2020)

 

 
 [49] 
 
Lesjak, Ž., et al.: A novel public mr image dataset of multiple sclerosis
patients with lesion segmentations based on multi-rater consensus.
Neuroinformatics (2018)

 

 
 [50] 
 
Lin, T.Y., et al.: Focal loss for dense object detection. In: ICCV (2017)

 

 
 [51] 
 
Long, J., et al.: Fully convolutional networks for semantic segmentation. In:
CVPR (2015)

 

 
 [52] 
 
Ma, T., et al.: Ensembling low precision models for binary biomedical image
segmentation. arXiv:2010.08648 (2020)

 

 
 [53] 
 
McKinley, R., et al.: Nabla-net: A deep dag-like convolutional architecture for
biomedical image segmentation. In: BrainLes (2016)

 

 
 [54] 
 
McKinley, R., et al.: Simultaneous lesion and neuroanatomy segmentation in
multiple sclerosis using deep neural networks. arXiv:1901.07419 (2019)

 

 
 [55] 
 
McKinley, R., et al.: Automatic detection of lesion load change in multiple
sclerosis using convolutional neural networks with segmentation confidence.
Neuroimage Clin (2020)

 

 
 [56] 
 
Milletari, F., et al.: V-net: Fully convolutional neural networks for
volumetric medical image segmentation. In: 3DV (2016)

 

 
 [57] 
 
Nair, T., et al.: Exploring uncertainty measures in deep networks for multiple
sclerosis lesion detection and segmentation. Med Image Anal (2020)

 

 
 [58] 
 
Narayana, P.A., et al.: Multimodal mri segmentation of brain tissue and
t2-hyperintense white matter lesions in multiple sclerosis using deep
convolutional neural networks and a large multi-center image database. In:
CIBEC (2018)

 

 
 [59] 
 
Narayana, P.A., et al.: Are multi-contrast magnetic resonance images necessary
for segmenting multiple sclerosis brains? a large cohort study based on deep
learning. Magn Reson Imaging (2020)

 

 
 [60] 
 
Narayana, P.A., et al.: Deep-learning-based neural tissue segmentation of mri
in multiple sclerosis: Effect of training set size. J Magn Reson Imaging
(2020)

 

 
 [61] 
 
Oguz, I., et al.: Dice overlap measures for objects of unknown number:
application to lesion segmentation. In: BrainLes (2017)

 

 
 [62] 
 
Placidi, G., et al.: Automatic framework for multiple sclerosis follow-up by
magnetic resonance imaging for reducing contrast agents. In: ICIAP (2019)

 

 
 [63] 
 
Ravnik, D., et al.: Dataset variability leverages white-matter lesion
segmentation performance with convolutional neural network. In: Medical
Imaging 2018: Image Processing (2018)

 

 
 [64] 
 
Reddi, S.J., et al.: On the convergence of adam and beyond. arXiv:1904.09237
(2019)

 

 
 [65] 
 
Ronneberger, O., et al.: U-net: Convolutional networks for biomedical image
segmentation. In: MICCAI (2015)

 

 
 [66] 
 
Roy, S., et al.: Multiple sclerosis lesion segmentation from brain mri via
fully convolutional neural networks. arXiv:1803.09172 (2018)

 

 
 [67] 
 
Salehi, S.S.M., et al.: Tversky loss function for image segmentation using 3d
fully convolutional deep networks. In: MLMI (2017)

 

 
 [68] 
 
Salem, M., et al.: Multiple sclerosis lesion synthesis in mri using an
encoder-decoder u-net. IEEE Access (2019)

 

 
 [69] 
 
Salem, M., et al.: A fully convolutional neural network for new t2-w lesion
detection in multiple sclerosis. Neuroimage Clin (2020)

 

 
 [70] 
 
Sepahvand, N.M., et al.: Cnn detection of new and enlarging multiple sclerosis
lesions from longitudinal mri using subtraction images. In: ISBI (2020)

 

 
 [71] 
 
Shachor, Y., et al.: A mixture of views network with applications to multi-view
medical imaging. Neurocomputing (2020)

 

 
 [72] 
 
Shinohara, R.T., et al.: Statistical normalization techniques for magnetic
resonance imaging. NeuroImage: Clinical (2014)

 

 
 [73] 
 
Smith, S.M.: Fast robust automated brain extraction. Hum Brain Mapp (2002)

 

 
 [74] 
 
Styner, M., et al.: 3d segmentation in the clinic: A grand challenge ii: Ms
lesion segmentation. Midas J (2008)

 

 
 [75] 
 
Tustison, N.J., et al.: N4itk: improved n3 bias correction. IEEE Trans Med
Imaging (2010)

 

 
 [76] 
 
Ulloa, G., et al.: Circular non-uniform sampling patch inputs for cnn applied
to multiple sclerosis lesion segmentation. In: CIARP (2018)

 

 
 [77] 
 
Ulloa, G., et al.: Improving multiple sclerosis lesion boundaries segmentation
by convolutional neural networks with focal learning. In: ICIAR (2020)

 

 
 [78] 
 
Vaidya, S., et al.: Longitudinal multiple sclerosis lesion segmentation using
3d convolutional neural networks. Proceedings of the 2015 Longitudinal
Multiple Sclerosis Lesion Segmentation Challenge (2015)

 

 
 [79] 
 
Valverde, S., et al.: Improving automated multiple sclerosis lesion
segmentation with a cascaded 3d convolutional neural network approach.
NeuroImage (2017)

 

 
 [80] 
 
Valverde, S., et al.: One-shot domain adaptation in multiple sclerosis lesion
segmentation using convolutional neural networks. Neuroimage Clin (2019)

 

 
 [81] 
 
Vang, Y.S., et al.: Synergynet: A fusion framework for multiple sclerosis brain
mri segmentation with local refinement. In: ISBI (2020)

 

 
 [82] 
 
Varsavsky, T., et al.: Test-time unsupervised domain adaptation. In: MICCAI
(2020)

 

 
 [83] 
 
Wang, Z., et al.: Ensemble of multi-sized fcns to improve white matter lesion
segmentation. In: MLMI (2018)

 

 
 [84] 
 
Warfield, S.K., et al.: Simultaneous truth and performance level estimation
(staple): an algorithm for the validation of image segmentation. IEEE Trans
Med Imaging (2004)

 

 
 [85] 
 
Weeda, M., et al.: Comparing lesion segmentation methods in multiple sclerosis:
Input from one manually delineated subject is sufficient for accurate lesion
segmentation. Neuroimage Clin (2019)

 

 
 [86] 
 
Xiang, Y., et al.: Segmentation method of multiple sclerosis lesions based on
3d-cnn networks. IET Image Processing (2020)

 

 
 [87] 
 
Zeiler, M.D.: Adadelta: an adaptive learning rate method. arXiv:1212.5701
(2012)

 

 
 [88] 
 
Zhang, C., et al.: Ms-gan: Gan-based semantic segmentation of multiple
sclerosis lesions in brain magnetic resonance imaging. In: DICTA (2018)

 

 
 [89] 
 
Zhang, H., et al.: Rsanet: Recurrent slice-wise attention network for multiple
sclerosis lesion segmentation. In: MICCAI (2019)

 

 
 [90] 
 
Zhang, H., et al.: Efficient folded attention for 3d medical image
reconstruction and segmentation. arXiv:2009.05576 (2020)

 

 
 [91] 
 
Zhang, H., et al.: Multiple sclerosis lesion segmentation with tiramisu and 2.5
d stacked slices. In: MICCAI (2019)

 

 
 [92] 
 
Zhang, H., et al.: Robust multiple sclerosis lesion inpainting with edge prior.
In: MLMI (2020)

 

 
 [93] 
 
Zhang, L., Tanno, R., Bronik, K., Jin, C., Nachev, P., Barkhof, F., Ciccarelli,
O., Alexander, D.C.: Learning to segment when experts disagree. In: MICCAI
(2020)