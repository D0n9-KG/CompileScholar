# ImOV3D: Learning Open-Vocabulary Point Clouds 3D Object Detection from Only 2D Images

Timing Yang $^{1,2*}$ Yuanliang Ju $^{1,2*}$ Li Yi $^{2,3,1\dagger}$

$^{1}$ Shanghai Qi Zhi Institute, $^{2}$ IIIS, Tsinghua University, $^{3}$ Shanghai AI Lab

# Abstract

Open-vocabulary 3D object detection (OV-3Det) aims to generalize beyond the limited number of base categories labeled during the training phase. The biggest bottleneck is the scarcity of annotated 3D data, whereas 2D image datasets are abundant and richly annotated. Consequently, it is intuitive to leverage the wealth of annotations in 2D images to alleviate the inherent data scarcity in OV-3Det. In this paper, we push the task setup to its limits by exploring the potential of using solely 2D images to learn OV-3Det. The major challenges for this setup is the modality gap between training images and testing point clouds, which prevents effective integration of 2D knowledge into OV-3Det. To address this challenge, we propose a novel framework ImOV3D to leverage pseudo multimodal representation containing both images and point clouds (PC) to close the modality gap. The key of ImOV3D lies in flexible modality conversion where 2D images can be lifted into 3D using monocular depth estimation and can also be derived from 3D scenes through rendering. This allows unifying both training images and testing point clouds into a common image-PC representation, encompassing a wealth of 2D semantic information and also incorporating the depth and structural characteristics of 3D spatial data. We carefully conduct such conversion to minimize the domain gap between training and test cases. Extensive experiments on two benchmark datasets, SUNRGBD and ScanNet, show that ImOV3D significantly outperforms existing methods, even in the absence of ground truth 3D training data. With the inclusion of a minimal amount of real 3D data for fine-tuning, the performance also significantly surpasses previous state-of-the-art. Codes and pre-trained models are released on the https://github.com/yangtiming/ImOV3D.

# 1 Introduction

In the 3D vision community, there is a notable surge in interest surrounding open-vocabulary 3D object detection (OV-3Det). This task focuses on the detection of objects from unbounded categories that were not present during the training phase, using 3D point clouds as input. Such capability holds immense significance in dynamic 3D environments where a wide range of object categories constantly emerge and evolve, which is critical in downstream applications including robotics[8, 18, 28, 23], autonomous driving [31, 46], and augmented reality [35, 44].

With the advancements in OV-3Det, which is not only scarce in terms of labels but also in the data itself. However, the collection and annotation of 3D point clouds scenes pose significant challenges. The availability of accessible and scannable scenes (e.g. indoor scenes) may be limited. Additionally, obtaining 3D annotations often requires substantial human effort and time-consuming. These limitations impact the model's performance in handling novel objects. Existing methods [30, 29, 28, 10, 49] seek help from powerful open-vocabulary 2D detectors. A common method

![](images/a89038fdaa37129f9350c779a5bb8f679ac8baf61065657144f4d0418b399f25.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Training"] --> B["Point Cloud Detector"]
    B --> C["Shared Weights"]
    C --> D["Point Cloud Detector"]
    D --> E["Imov3D☆"]
    
    subgraph Inference
        F["GT Point Clouds"] --> G["Point Cloud Detector"]
    end
    
    subgraph Previous Methods
        H["RGB Images"] --> I["Point Clouds"]
        I --> J["Point Cloud Detector"]
    end
    
    subgraph Training Label
        K["Please help me find bed, lamp and dresser."] --> L["Point Cloud Detector"]
        L --> M["Point Cloud Detector"]
    end
    
    subgraph Inference
        N["Pseudo Image"] --> O["Multimodal Detector"]
        O --> P["Imov3D☆"]
        Q["Pseudo Point Clouds"] --> R["Multimodal Detector"]
        R --> S["Imov3D☆"]
        T["Pseudo Image"] --> U["Multimodal Detector"]
        U --> V["Imov3D☆"]
        W["Pseudo Image"] --> X["Multimodal Detector"]
        X --> Y["Imov3D☆"]
        Z["Pseudo Image"] --> AA["Multimodal Detector"]
        AA --> AB["Imov3D☆"]
        AC["Pseudo Image"] --> AD["Multimodal Detector"]
        AD --> AE["Imov3D☆"]
        AF["Pseudo Image"] --> AG["Multimodal Detector"]
        AG --> AH["Imov3D☆"]
        AI["Pseudo Image"] --> AJ["Multimodal Detector"]
        AJ --> AK["Imov3D☆"]
        AL["Pseudo Image"] --> AM["Multimodal Detector"]
        AM --> AN["Imov3D☆"]
        AO["Pseudo Image"] --> AP["Multimodal Detector"]
        AP --> AQ["Imov3D☆"]
        AR["Pseudo Image"] --> AS["Multimodal Detector"]
        AS --> AT["Imov3D☆"]
        AU["Pseudo Image"] --> AV["Multimodal Detector"]
        AV --> AW["Imov3D☆"]
        AX["Pseudo Image"] --> AY["Multimodal Detector"]
        AY --> AZ["Imov3D☆"]
        BA["Pseudo Image"] --> BB["Multimodal Detector"]
        BB --> BC["Imov3D☆"]
        BD["Pseudo Image"] --> BE["Multimodal Detector"]
        BE --> BF["Imov3D☆"]
        BG["Pseudo Image"] --> BH["Multimodal Detector"]
        BH --> BI["Imov3D☆"]
        BJ["Pseudo Image"] --> BK["Multimodal Detector"]
        BK --> BL["Imov3D☆"]
        BM["Pseudo Image"] --> BN["Multimodal Detector"]
        BN --> BO["Imov3D☆"]
        BP["Pseudo Image"] --> BQ["Multimodal Detector"]
        BQ --> BR["Imov3D☆"]
        BS["Pseudo Image"] --> BT["Multimodal Detector"]
        BT --> BU["Imov3D☆"]
        BV["Pseudo Image"] --> BW["Multimodal Detector"]
        BW --> BX["Imov3D☆"]
        BY["Pseudo Image"] --> BZ["Multimodal Detector"]
        BZ --> CA["Imov3D☆"]
        CB["Pseudo Image"] --> CC["Multimodal Detector"]
        CC --> CD["Imov3D☆"]
        CE["Pseudo Image"] --> CF["Multimodal Detector"]
        CF --> CG["Imov3D☆"]
        CH["Pseudo Image"] --> CI["Multimodal Detector"]
        CI --> CJ["Imov3D☆"]
        CK["Pseudo Image"] --> CL["Multimodal Detector"]
        CL --> CD
    end
```
</details>

Figure 1: Left: Traditional methods require paired RGB-D data for training and use single-modality point clouds as input during inference. Right: ImOV3D involves using a vast amount of 2D images to generate pseudo point clouds during the training phase, which are then rendered back into images. In the inference phase, with only point clouds as input, we still construct a pseudo-multimodal representation to enhance detection performance.

leverages paired RGB-D data together with 2D detectors to generate 3D pseudo labels to address the label scarcity issue, as shown in Figure 1 left. But they are still restricted by the small scale of existing paired RGB-D data. Moreover, the from scratch trained 3D detector can hardly inherit from powerful open-vocabulary 2D detector models directly due to the modality difference. We then ask the question, what is the best way to transfer 2D knowledge to 3D for OV-3Det?

Observing that the modality gap prevents a direct knowledge transfer, we propose to leverage a pseudo multi-modal representation to close the gap. On one hand, we can lift a 2D image into a pseudo-3D representation through estimating the depth and camera matrix. On the other hand, we can convert a 3D point cloud into a pseudo-2D representation through rendering. The pseudo RGB image-PC multimodal representation could serve as a common ground for better transferring knowledge from 2D to 3D.

In this paper, we present ImOV3D, which addresses these challenges by employing pseudo-multimodal representation as a unified framework. As shown in Figure 1 right side, In both the training and the inference phase, we construct pseudo-multimodal representation to achieve our goal of training solely with 2D images and better integrating multimodal features to enhance the performance of OV-3Det. Our key idea lies in proper modality conversion. Specifically, the entire pipeline consists of two flows: (1) Image → Pseudo PC, by leveraging a large-scale 2D images training set, our method begins by converting images to pseudo point clouds through monocular depth estimation and approximate camera parameter. We automatically generate pseudo 3D labels based on 2D annotations, providing the necessary training data. We also designed a set of revision modules, which significantly improve the quality of the pseudo 3D data through the use of GPT-4 [1]'s size prior and the orientation of the estimated normal map. (2) Pseudo PC → Pseudo Image, we learn a point clouds renderer capable of producing natural-looking textured 2D images from pseudo 3D point clouds. This enables ImOV3D to leverage pseudo-multimodal 3D detection even for point cloud-only inputs during inference, transferring 2D rich semantic information and proposals into the 3D space, further enhancing the detector's performance.

Despite being trained solely with the 2D image set, ImOV3D exhibits impressive detection results when directly processing real 3D scans. This is attributed to the high fidelity of the lifted point clouds and the point clouds rendering. Additionally, when a small amount of real 3D data becomes available, even without any 3D annotations, ImOV3D can further narrow the gap between pseudo and real data by fine-tuning on such 3D data, leading to improved detection performance. To validate the effectiveness of ImOV3D, we perform extensive experiments on two benchmark datasets: SUNRGBD [43] and ScanNet [12]. Notably, in scenarios where real 3D training data is unavailable, ImOV3D surpasses previous state-of-the-art open-vocabulary 3D detectors by an mAP@0.25 improvement of at least $7.14\%$ on SUNRGBD and $6.78\%$ on ScanNet. Furthermore, when real 3D training data is accessible, ImOV3D continues to outperform various challenging baselines by a large margin. Thorough ablations are also conducted to validate the efficacy of our designs. In summary, our contributions are three-fold:

\- We propose ImOV3D, the first OV-3Det method that can be trained solely with 2D images without requiring any 3D point clouds or 3D annotations.

- We introduce a novel pseudo-multimodal representation pipeline which converts 2D internet images and corresponding detections into pseudo point clouds, pseudo 3D annotations, and point clouds renderings to support point clouds-based multimodal OV-3Det.   
- ImOV3D achieves state-of-the-art performance on two general OV-3Det benchmark datasets across various settings, showcasing its ability to enhance open world 3D understanding despite the lack of 3D data and annotations.

# 2 Related Work

Open-Vocabulary 2D Object Detection encompasses two primary series of works: The first [50, 19, 32, 42, 3, 15, 37, 45, 11, 9, 50], which draws upon knowledge from pre-trained Vision-Language models (e.g., CLIP [40]), comprehends the relationships between images and their corresponding textual descriptions, thereby enhancing object recognition and classification. The second series [26, 47, 48, 51, 17, 55, 33, 25, 56] involves the use of extensive training data, specifically text/image pairs, enabling the model to learn a more diverse set of object representations. Detic [56] leverages vocabularies from image classification datasets to train the classification head of object detectors, addressing the issue of insufficient training data and enabling inference on a larger vocabulary set. In the 2D component of our pseudo-multimodal detector, we utilize Detic [56] to predict 2D labels and bounding boxes. These 2D visual information features are then converted and augmented for the 3D point cloud detector, significantly enhancing our model's ability to recognize a broader range of objects.

Open-Vocabulary Scene Understanding has recently gained increased attention $[36, 41, 53, 21, 22, 24, 14, 10]$ and plays a critical role in robotics, autonomous driving, etc. OpenScene $[36]$ achieves open-world scene understanding without the need for labeled data by densely embedding 3D scene points together with text and image pixels into the CLIP $[40]$ feature space. PLA $[14]$ develops a hierarchical approach to pairing 3D data with text for open-world 3D learning. We focus on OV-3Det, where merely extracting CLIP $[40]$ features is insufficient. We also require the intricate spatial structure of point clouds to enhance detection accuracy and robustness. By integrating both CLIP $[40]$ 's visual knowledge and the detailed geometric information from point clouds, our approach aims to enable the recognition of a broader range of objects beyond the predefined categories.

Open-Vocabulary 3D Object Detection in 3D vision is still in its early stages, especially when compared to traditional 3D object detection [38, 39, 34, 27, 6]. OV-3DETIC [29] leverages ImageNet1K [13] to expand the detector's vocabulary set and conducts contrastive learning between images and point clouds modalities for more effective knowledge transfer. OV-3DET [30] generates pseudo 3D annotations for localization using a pre-trained 2D open-vocabulary detector [56]. CoDA [5] leverages 2D and 3D prior information and a cross-modal alignment module to simultaneously learn the localization and classification capabilities. CoDAv2 [7] improves CoDA [5] further by proposing the novel object enrichment strategy and 2D box guidance. FM-OV3D [52] combines multiple foundation models without the need for 3D annotations. However, they are still subject to the influence of the volume of 3D data and still require strict correspondence between RGB-D data. Our method can generate training data for OV-3Det task using only 2D images, without any 3D ground truth data. It can directly achieve state-of-the-art performance when tested on the evaluation set. The designed pseudo-multimodal representation pipeline provides a novel solution for the utilization of both 2D and 3D information.

# 3 Method

# 3.1 Overview

An overview of the proposed open world 3D Object Detection model, ImOV3D, is shown in Figure 2. ImOV3D is a point cloud-only model that addresses the challenges of the scarcity of annotated 3D datasets in open-vocabulary 3D Object Detection. To overcome this challenge, ImOV3D uses large-scale 2D datasets to generate pseudo 3D point clouds and annotations. We use a monocular depth estimation model to create metric depth images, which are then converted into pseudo 3D point clouds for both indoor and outdoor scenes. To generate pseudo 3D annotations, we lift 2D bounding boxes into 3D space. To leverage multimodal data, we transform the point clouds into pseudo images using a point cloud renderer. Our training strategy involves a two-stage approach. Firstly, we conduct

![](images/96c983ff5a2f7c43fb763634d3a0fd3febf45eb8590d06b18bdbe80822471151.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input: Pseudo Point Clouds"] --> B["3D Backbone"]
    B --> C["2D OV Detector"]
    C --> D["Fusion"]
    D --> E["CLIP"]
    E --> F["Inference Phase: Text Prompts: A Photo of {Pillow, Lamp, Bed ...}"]
    
    subgraph_Image_1["Pseudo Point Clouds"]
        G["RGB Image"] --> H["Point Cloud Lifting Module"]
        H --> I["Pseudo 3D Annotation Generator"]
        I --> J["Pseudo Point Clouds and Annotations"]
        J --> K["Point Cloud Renderer"]
        K --> L["Pseudo Image"]
    end
    
    subgraph_Image_2["Pseudo Point Clouds"]
        M["Input"] --> N["Pseudo Image"]
        N --> O["3D Backbone"]
        O --> P["2D OV Detector"]
        P --> Q["Fusion"]
        Q --> R["CLIP"]
        R --> S["Inference Phase"]
    end
```
</details>

Figure 2: Overview of ImOV3D: Our model takes 2D images as input and puts them into the Pseudo 3D Annotation Generator to produce pseudo annotations. These 2D images are also fed into the Point Cloud Lifting Module to generate pseudo point clouds. Subsequently, using the Point Cloud Renderer, these pseudo point clouds are rendered into pseudo images, which then get processed by a 2D open vocabulary detector to detect 2D proposals and transfer the 2D semantic information to 3D space. Armed with pseudo point clouds, annotations, and pseudo images data, we proceed to train a multimodal 3D detector.

pre-training using pseudo 3D point clouds and corresponding annotations. Subsequently, we initiate an adaptation stage aimed at minimizing the domain discrepancy between 2D and 3D datasets.

# 3.2 Point Cloud Lifting Module

The success of open-vocabulary object detection relies heavily on the availability of large-scale labeled datasets. However, the scarcity of comparable 3D datasets poses a challenge for open world 3D Object Detection. To address this, we bridge 2D images $\mathcal{I}_{\mathrm{2D}} \in \mathbb{R}^{M \times H \times W \times 3}$ (where $M$ is the number of images, and $H$ and $W$ are the height and width, respectively) to 3D detection by generating pseudo 3D point clouds $\mathcal{P}_{\mathrm{pseudo}} \in \mathbb{R}^{M \times N \times 3}$ (where $N$ is the number of points, each with coordinates (x, y, z)).

Utilizing 2D datasets for 3D detection presents difficulties due to the absence of metric depth images and camera parameters. To overcome these obstacles, we use a metric depth estimation model to obtain single-view depth images $\mathcal{D}_{\mathrm{metric}} \in \mathbb{R}^{M \times H \times W}$ . Additionally, we employ fixed camera intrinsics $K \in \mathbb{R}^{3 \times 3}$ , with the focal length $f$ calculated based on a 55-degree field of view (FOV) and the image dimensions.

However, the absence of camera extrinsics $\mathbf{E} = \{R\mid t\}$ (where $R$ is the rotation matrix and $t$ is the translation vector set to $[0,0,0]^{\top}$ ) results in the arbitrary orientation of point clouds. To correct this, we use a rotation correction module to ensure the ground plane is horizontal, as shown in Figure 3 (a). First, we estimate the surface normal vector at each pixel using a normal estimation model [2], creating a normal map. From this, we selectively extract the horizontal normal vector $N_{i}$ at each pixel, defined as $(N_x,N_y,N_z)$ . We then compute the normal vector of the horizon surface as $N_{\mathrm{pred}} = Cluster(N_i)$ . To align $N_{\mathrm{pred}}$ with the $Z_{axis}$ , we calculate the rotation matrix $R$ using the following equation:

$$
R = I + K + K ^ {2} \frac {1 - N _ {\text { p   r   e   d }} \cdot Z _ {\text { a   x   i   s }}}{\| v \| ^ {2}} \tag {1}
$$

where $I$ is the identity matrix, $v$ is the cross product of $N_{pred}$ and $Z_{axis}$ , expressed as $N_{pred} \times Z_{axis}$ , $K$ is the skew symmetric matrix constructed from the vector $v$ , represented as:

$$
K = \left[ \begin{array}{c c c} 0 & - v _ {z} & v _ {y} \\ v _ {z} & 0 & - v _ {x} \\ - v _ {y} & v _ {x} & 0 \end{array} \right] \tag {2}
$$

![](images/2821ee3a8d1611797327a0593fc9a5b940f480b91b325cd06caacc221cd03b1b.jpg)  
Figure 3: Illustration of 3D Data Revision Module: (a) The rotation correction module involves processing an RGB image through a Normal Estimator to generate a normal map. This map then helps extract a horizontal surface mask for identifying horizontal point clouds, from which normal vectors $N_{pred}$ are obtained. These vectors are aligned with the Z-axis to compute the rotation matrix R. (b) In the 3D box filtering module, prompts related to object dimensions are first provided to GPT-4 to determine the mean size for each category. This mean size is then used to filter out boxes that do not meet the threshold criteria.

After obtaining the camera intrinsics matrix K and the camera extrinsics matrix E through the previous steps, depth images $D_{metric}$ are converted into point clouds $P_{pseudo}$ .

# 3.3 Pseudo 3D Annotation Generator

Building upon the vast collection of pseudo 3D point clouds $P_{pseudo}$ acquired from 2D datasets, our next step is to generate pseudo 3D bounding boxes $B_{3Dpseudo} \in R^{M \times K \times 7}$ (where K is the number of bounding boxes and each box has 7 parameters: center coordinates, dimensions, and orientation).

2D datasets contain rich segmentation information that can be used to generate 3D boxes by lifting. Using the camera intrinsics matrix $K$ and camera extrinsics matrix $\mathbf{E}$ obtained through the Point Cloud Lifting Module, we lift the 2D bounding boxes $\mathcal{B}_{2Dgt} \in \mathbb{R}^{M \times K \times 4}$ from the 2D datasets into 3D space by extracting 3D points that fall within the predicted 2D boxes, generating frustum 3D boxes $\mathcal{B}_{3Dpseudo}$ . The extracted point clouds may contain background points and outliers. To remove these, we employ a clustering [16] algorithm to analyze point clouds. Through the clustering results, we can identify and remove background points and outliers that do not belong to the target objects.

However, these lifted 3D boxes may still contain noise from the depth images $\mathcal{D}_{\mathrm{metric}}$ obtained by monocular depth estimation. To address this issue, we use a 3D box filtering module to filter out inaccurate 3D boxes, as shown in Figure 3 (b). First, we construct a database of median object sizes using GPT-4 [1]. By prompting GPT-4 with "Please tell me the average length, width, and height of this [category], using meters as the unit", we obtain the median dimensions $L_{GPT}, W_{GPT}, H_{GPT}$ . Each object in a scene, defined by dimensions $L, W, H$ , is compared to these median dimensions using a threshold $T$ . An object is preserved if each element of:

$$
T <   \frac {R}{R _ {\mathrm{GPT}}} <   \frac {1}{T}, \quad \forall R \in \{L, W, H \} \tag {3}
$$

The 3D box filtering module consists of two components: Train Phase Prior Size Filtering and Inference Phase Semantic Size Filtering. The first component filters out boxes that do not match the size criteria before training. The second component removes semantically similar but size-different categories during inference, preventing errors such as misidentifying a book as a bookcase.

# 3.4 Point Cloud Renderer

Point cloud data has inherent limitations, such as the inability of sparse point clouds to capture detailed textures. 2D images can enrich 3D data by providing additional texture information that point clouds lack. To utilize 2D images, we transform point clouds $\mathcal{P}$ into rendered images $\mathcal{I}_{\text{rendered}} \in \mathbb{R}^{M \times H \times W}$ .

Integrating rendered images into a 3D detection pipeline is challenging. A naive approach, as mentioned in PointClip [57], is to append raw depth values across the RGB channels, but this fails to apply a mature open-world 2D detector effectively. To leverage multimodal information without

additional inputs beyond 3D point clouds, we develop a point cloud renderer to convert point clouds into detailed pseudo images. This process can also be learned solely from 2D image datasets.

The point cloud renderer has two key modules: The point cloud rendering module converts point clouds $\mathcal{P}$ into rendered images $\mathcal{I}_{\text{rendered}}$ , and the color rendering module then processes these images to produce colorized outputs using ControlNet [54]. ControlNet [54] is a method designed to control diffusion models, transforming rendered images $\mathcal{I}_{\text{rendered}}$ into pseudo images $\mathcal{I}_{\text{pseudo}} \in \mathbb{R}^{M \times H \times W \times 3}$ .

In the pretraining stage, we use the camera intrinsics K and extrinsics E from the Point Cloud Lifting Module to render $P_{pseudo}$ into rendered images $I_{rendered}$ . During adaptation and inference, we render ground truth point clouds $P_{gt}$ into images using the intrinsics K obtained in the same way. Due to the lack of extrinsics E, the final rendered images $I_{rendered}$ are obtained by finding the optimal angle from different horizontal and vertical perspectives to make the images most compact.

In reality, we cannot project a point cloud while guaranteeing that every pixel corresponds to some points. There will be holes and missing areas due to point cloud imperfections or incompatible viewpoint selection. We adjust the camera's position horizontally and vertically to observe point clouds from various angles, removing obscured portions. Finally, we render the point clouds back into images from their original perspective, resulting in partial view rendered images $\mathcal{I}_{\mathrm{partial}} \in \mathbb{R}^{M \times N \times 3}$ . The angle range for adjustments is set from -75 to 75 degrees, with a 15-degree interval:

$$
\theta_ {h}, \theta_ {v} \in \{- 7 5 + 1 5 k | k = 0, 1, 2, \dots , 1 0 \} ^ {\circ} \tag {4}
$$

where $k$ is an integer indicating the stepwise adjustment of the camera's angle.

After generating partial view rendered images $I_{partial}$ , the next step is to fine-tune ControlNet [54] using these images to obtain pseudo images $I_{pseudo}$ . Three types of data are prepared for fine-tuning: prompts, targets, and sources. RGB images $I_{2D}$ from a 2D dataset serve as the targets, while the partial view rendered images $I_{partial}$ are the training sources. Prompts are not used during training.

Finally, we use the pseudo images $I_{pseudo}$ and annotations $B_{2Dgt}$ from 2D datasets to fine-tune an open-vocabulary 2D detector. Thus, we can use $I_{pseudo}$ to obtain corresponding pseudo 2D bounding boxes $B_{2DTpseudo} \in R^{M \times K \times 4}$ .

# 3.5 Pseudo Multimodal 3D Object Detector

With an extensive dataset comprising abundant 3D data ( $P + B_{3Dpseudo}$ ) and pseudo images data $I_{pseudo}$ , our next step is to train a pseudo multimodal 3D detector using a two-stage approach.

Training Strategy Our training process includes pretraining and adaptation stages. In the pretraining stage, we train on pseudo 3D point clouds $P_{pseudo}$ and annotations $B_{3Dpseudo}$ , combined with pseudo images $I_{pseudo}$ . While the pre-trained model performs well for zero-shot detection, a significant domain gap exists between 2D and 3D datasets.

In the adaptation stage, to minimize the domain gap, we follow the same approach as OV-3DET. First, a pre-trained open-vocabulary 2D detector is used to detect objects in the image. Then, these 2D boxes $\mathcal{B}_{\mathrm{2Dpseudo}} \in \mathbb{R}^{M \times K \times 4}$ , along with RGBD data, are lifted into 3D space. Through clustering to remove background and outlier points, we obtain precise and compact 3D boxes $\mathcal{B}_{\mathrm{3Dpseudo}}$ . Finally, this processed data is used for adaptation. To further explore the benefits of pretrain, we use 3D datasets of varying sizes to test the model's performance under different data availability conditions.

Loss Function In this section, we describe loss function used in the pretrain stage. By leveraging $P_{pseudo}$ and $B_{3Dpseudo}$ , a 3D backbone is trained to obtain seed points $K \in R^{K \times 3}$ , where K represents the number of seeds, along with 3D feature representations $F_{pc} \in \mathbb{R}^{K \times (3 + F)}$ , with F denoting the feature dimension. Then, seed points are projected back into 2D space via the camera matrix. These seeds that fall within the 2D bounding boxes $B_{2DTpseudo}$ retrieve the corresponding 2D cues associated with these boxes and bring them back into 3D space. These lifted 2D cues features are represented as $F_{img} \in \mathbb{R}^{K \times (3 + F')}$ , where $F'$ represents the feature dimension. Finally, the point cloud features $F_{pc}$ and image features $F_{img}$ are concatenated, forming the joint representation $F_{joint} \in \mathbb{R}^{K \times (3 + F + F')}$ . In the adaptation stage, $P_{pseudo}$ is replaced with $P_{gt}$ , keeping the workflow consistent with the pretrain stage.

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { loc }} + \sum W _ {i} \times \text { CrossEntropy } (\text { Cls - header } (\mathcal {F} _ {i}) \cdot \mathcal {F} _ {\text { text }}) \tag {5}
$$

where i represents different features, such as pc, img, joint. $W_{i}$ is the weight corresponding to feature i. $L_{loc}$ represents the original localization loss function used in ImVoteNet[39]. $F_{text}$ denotes the feature extracted by the text encoder in CLIP.

Implementation Details Our model is a point cloud-only ImVoteNet [39] +Clip architecture. The monocular depth estimation model used is ZoeDepth[4], jointly trained on both indoor and outdoor scenes. In the pre-training phase, similar to ImVoteNet, we train for 180 epochs with an initial learning rate of 0.001. In the adaptation phase, we train for 100 epochs, reducing the learning rate to 0.0005.

For 2D voting, there are three types of cues: Geometric cues, Texture cues, and Semantic Cues. Unlike ImVoteNet, we retain geometric cues but remove texture cues. For Semantic cues, instead of using a one-hot class vector, we use pre-trained CLIP text encoder features, which is more suitable for an open-vocabulary setting.

# 4 Experiments

In this section, we compare our proposed ImOV3D with other baseline models. Our experimental setup is divided into two main stages: Pretraining and Adaptation. During the pretraining stage, the training data is pseudo 3D data, referring to pseudo 3D point clouds and their corresponding annotations (3D boxes). During the adaptation stage, we use ground truth point clouds and pseudo labels to minimize the domain gap. All experiments are conducted on two commonly used Object Detection datasets: SUNRGBD [43] and ScanNet [12]. Additionally, we carry out comprehensive ablation studies to validate the effectiveness of our model's components and the data generation pipeline.

# 4.1 Experimental Setup

2D Images Dataset: We select the LVIS [20] dataset as our 2D image source for generating pseudo 3D data, utilizing 42,000 images provided in its training set, which spans 1,203 categories with rich and detailed annotations.

3D Point Clouds Dataset: We select SUNRGBD [43] and ScanNet [12] as our 3D point clouds datasets for adaptation and testing, SUNRGBD [43] and ScanNet [12] encompass a diverse range of indoor environments and offer comprehensive annotations, including 2D and 3D bounding boxes for objects. We test on 20 common categories in both datasets.

Evaluation Metrics: We employ mean Average Precision (mAP) at an IoU threshold of 0.25 as our primary evaluation metric. This metric effectively balances precision and recall in assessing how well our models perform on selected datasets.

Table 1: Results from the Pretraining stage comparison experiments on SUNRGBD and ScanNet, ImOV3D only require point clouds input. 

<table><tr><td>Stage</td><td>Data Type</td><td>Method</td><td>Input</td><td>Training Strategy</td><td>SUNRGBD mAP@0.25</td><td>ScanNet mAP@0.25</td></tr><tr><td rowspan="4">Pre-training</td><td rowspan="4">Pseudo Data</td><td>OV-VoteNet [38]</td><td>Point Cloud</td><td>One-Stage</td><td>5.18</td><td>5.86</td></tr><tr><td>OV-3DETR [34]</td><td>Point Cloud</td><td>One-Stage</td><td>5.24</td><td>5.30</td></tr><tr><td>OV-3DET [30]</td><td>Point Cloud + Image</td><td>Two-Stage</td><td>5.47</td><td>5.69</td></tr><tr><td>Ours</td><td>Point Cloud</td><td>One-Stage</td><td> $12.61 \uparrow 7.14$ </td><td> $12.64 \uparrow 6.78$ </td></tr></table>

Table 2: Results from the Adaptation stage comparison experiments on SUNRGBD and ScanNet 

<table><tr><td>Stage</td><td>Method</td><td>Input</td><td>Training Strategy</td><td>SUNRGBD mAP@0.25</td><td>ScanNet mAP@0.25</td></tr><tr><td rowspan="3">Adaptation</td><td>OV-3DET [30]</td><td>Point Cloud + Image</td><td>Two-Stage</td><td>20.46</td><td>18.02</td></tr><tr><td>CoDA [5]</td><td>Point Cloud</td><td>One-Stage</td><td>—</td><td>19.32</td></tr><tr><td>Ours</td><td>Point Cloud</td><td>One-Stage</td><td>22.53↑ 2.07</td><td>21.45↑ 2.13</td></tr></table>

# 4.2 Main Results

Pretraining: Due to the absence of existing baseline methods except OV-3DET [30], we utilize CLIP [40] to make previous high-performance 3D detectors such as 3DETR [34] and VoteNet [38] compatible with OV3Det. Specifically, to adapt traditional point cloud detectors for Open Vocabulary detection, we first extract geometric features from point clouds. Then, we integrate

CLIP [40] for classification by converting these features for compatibility with CLIP [40] visual encoder and creating textual prompts for zero-shot classification. Finally, we compare the encoded prompts with the visual features to classify objects beyond the predefined categories. Therefore, these baselines are denoted as OV-VoteNet [38], OV-3DETR [34].

Adaptation: To ensure a fair comparison with the current SOTA OV3Det methods, during the adaptation stage, all baselines use OV-3DET [30]'s approach to generating pseudo labels for ground truth point cloud data, which serve as training data for adaptation. In this stage, comparisons are made with CoDA [5] and OV-3DET [30].

# 4.2.1 Pretraining → 3D Training Data Free OV-3Det

As shown in Table 1, training solely with pseudo 3D data generated by our method, ImOV3D improves mAP@0.25 by $7.14\%$ on SUNRGBD and $6.78\%$ on ScanNet over the best baseline. This achievement, made without using any 3D ground truth annotated data, demonstrates the high quality of our generated data and the effectiveness of using extensive 2D datasets to enhance Open World perception. Unlike OV-VoteNet, which lacks 2D image integration, our method's mAP@0.25 outperforms OV-VoteNet by $7.43\%$ and $6.78\%$ on the two datasets, proving the effectiveness of our multimodal approach even with only point cloud inputs. OV-3DET and ImOV3D visualization results are shown in Figure 6(b).

# 4.2.2 Adaptation → 3D Training Data Guided OV-3Det

Table 2 shows original OV-3DET results in the first row. CoDA only compares with OV-3DET on ScanNet. Our experiments indicate that after pretraining with pseudo 3D data, ImOV3D outperforms the best baseline by 2.07% on SUNRGBD and 2.13% on ScanNet in mAP@0.25. This highlights the crucial role of pseudo 3D data in training and its effectiveness as data augmentation.

# 5 Ablation Study

# 5.1 Ablation Study of 3D Data Revision

To validate the effectiveness of enhancing pseudo 3D data quality, we conducted ablation experiments with the Rotation Correction Module and 3D Box Filtering Module. The 3D Box Filtering Module includes Train Phase Prior Size Filtering and Inference Phase Semantic Size Filtering. Table 3 shows the results: the baseline without any modules, adding Train Phase Prior Size Filtering improves mAP@0.25 by 1.65% on SUNRGBD and 1.27% on ScanNet. Adding the Rotation Correction Module improves by 1.3% on SUNRGBD and 1.96% on ScanNet. Combining both modules results in a 2.98% improvement on SUNRGBD and 3.31% on ScanNet. Adding Semantic Size Filtering during inference further increases mAP@0.25 by 4.26% on SUNRGBD and 4.31% on ScanNet. These results highlight the effectiveness of each module in improving data quality and OV3Det accuracy.

Table 3: Results from the ablation study on the Rotation Correction Module and the 3D Box Filtering Module, conducted on SUNRGBD and ScanNet, are presented. The 3D Box Filtering Module is divided into two components: Train Phase Prior Size Filtering and Inference Phase Semantic Size Filtering. 

<table><tr><td>Stage</td><td>Train Phase Prior Size</td><td>Rotation Correction</td><td>Inference Phase Semantic Size</td><td>SUNRGBD mAP@0.25</td><td>ScanNet mAP@0.25</td></tr><tr><td rowspan="5">Pre-training</td><td>✕</td><td>✕</td><td>✕</td><td>8.35</td><td>8.33</td></tr><tr><td>✕</td><td>✕</td><td>✕</td><td>10.00</td><td>9.60</td></tr><tr><td>✕</td><td>✕</td><td>✕</td><td>9.65</td><td>10.29</td></tr><tr><td>✕</td><td>✕</td><td>✕</td><td>11.33</td><td>11.64</td></tr><tr><td>✕</td><td>✕</td><td>✕</td><td>12.61</td><td>12.64</td></tr></table>

We also discuss the efficiency of GPT-4 [1] in the 3D box filtering module using the SUNRGBD dataset [43]. For comparison, we select the top 10 classes with the most instances in the validation set. The volume ratio for these 10 classes is defined as $\mathrm{Ratio}_V = \frac{L\times W\times H}{L_{\mathrm{GT / GPT}}\times W_{\mathrm{GT / GPT}}\times H_{\mathrm{GT / GPT}}}$ . This ratio is an insightful metric for comparing the performance of the GPT-4 powered 3D box filter module

to the ground truth (GT). A ratio close to 1 indicates high precision. We calculate $\mathrm{Ratio}_V$ for each instance and use Kernel Density Estimation (KDE) to analyze and plot the distributions of the volume ratios. Results are presented in Figure 6(a).

# 5.2 Ablation Study of Depth and Pseudo Images

To validate the effectiveness of pseudo images generated by ControlNet [54], we compare 2D depth maps from pseudo point clouds with pseudo images, shown in Figure 4. On the SUNRGBD dataset, mAP@0.25 increased from $4.38\%$ to $12.61\%$ , and on the ScanNet dataset, it rose from $4.47\%$ to $12.64\%$ (see Table 4). This shows that rich texture information in 2D images significantly enhances 3D detection performance.

Table 4: The results from different types of 2D rendering images include depth maps and pseudo images. 

<table><tr><td>Stage</td><td>Rendered Images Data Types</td><td>SUNRGBD mAP@0.25</td><td>ScanNet mAP@0.25</td></tr><tr><td rowspan="2">Pre-training</td><td>Depth Map</td><td>4.38</td><td>4.47</td></tr><tr><td>Pseudo Images</td><td>12.61</td><td>12.64</td></tr></table>

![](images/0b40801939a7c3b4f3c3f52341a20e50c83bb1cd1af2e1f2a48ae5cc37882b57.jpg)

![](images/4867d7a13aad1e96ccdb5e5be6bb94421d20efffd9456605de36954bc3ceecd4.jpg)  
(a)

![](images/37279095cd1098cd3805834d1f6e1dc82e82ca68a2a8dd0d726dcba3d494552f.jpg)

![](images/dfe6f5fa6619de89818978d48880b560676fac0902e7bfa0563fd92704f8fe5c.jpg)

![](images/218ff3a1a8dd321dac67f6bf2d318fd9d7ca3b1ddf175db9db4c7bf72fbded9a.jpg)

![](images/798eba4b21160089642fd48b70cba8958718df2dec43f02f31ee2f1cd6447224.jpg)  
(b)

![](images/c2524cd2f605b97b673dca20a7c559d93fca1f5cfe5b6995e086c90e2c43ad5e.jpg)

![](images/f5c7edc4239fcfcb1b31de9dc70fb988e92812b8da7c9a97ab63c3de5baba6ac.jpg)

![](images/f7ae09cf018ac391a55c0cb977b533ffa924b368734ec359c64c892b337a8b5a.jpg)

![](images/8207c3958d983b29870f2949ee1de704c360a5e18e86b658cc657f9b12027a70.jpg)  
(c)   
Figure 4: Qualitative results include (a) 2D RGB images, (b) 2D depth maps with 2D OVDetector annotations, and (c) pseudo images with annotations from a fine-tuned 2D detector.

# 5.3 Ablation Study of Data Volume

Our method fine-tunes with limited real ground truth 3D point cloud data and pseudo 3D annotations. Using OV-3DET's code, we train with varying data volumes. With $10\%$ adaptation data, OV-3DET's mAP@0.25 on SUNRGBD drops from $20.46\%$ to $15.24\%$ , while ours drops from $22.53\%$ to $19.24\%$ . On ScanNet, OV-3DET's mAP@0.25 falls from $18.02\%$ to $14.35\%$ , and ours falls from $21.45\%$ to $18.45\%$ (Figure 5(a)(b)). We observed a decrease in performance compared to using the full data set; however, our method was still able to maintain relatively high detection accuracy. This confirmed the robustness of our method and its adaptability to small datasets, enabling effective 3D Object Detection even under constrained data conditions. It also underscores the importance of developing models for OV3Det that are capable of learning from limited data and generalizing to a broader range of scenarios.

![](images/37f25704e6498126af6302ec5c11d35b4a0752b0b303d193fb428ec5eac95a36.jpg)

<details>
<summary>bar</summary>

| Data Volume | ImOV3D(Durs) | OV-DET |
| ----------- | ------------ | ------ |
| 100%        | 22.53        | 20.46  |
| 10%         | 19.24        | 15.24  |
</details>

(a)

![](images/bbdd7f1fde5c1b2de59300844cedf117f8fd2f44f722ddded9cc0169bc093296.jpg)

<details>
<summary>bar</summary>

| Data Volume | IMOD3d(Ours) | OX-3DET |
| ----------- | ------------ | ------- |
| 100%        | 21.45        | 18.02   |
| 10%         | 18.45        | 14.35   |
</details>

(b)

![](images/330a6d9b39ac4a1030d7cdaf0e4ecc9318ba10bbb69f403064f97cacbd8363ce.jpg)

<details>
<summary>bar</summary>

| Transfer Direction | In-DVD3D(Durs) | O-V3DET |
|---|---|---|
| SUNRGBD→ScanNet | 19.40 | 12.30 |
| ScanNet→SUNRGBD | 20.39 | 12.57 |
</details>

(c)   
Figure 5: (a) and (b) show data volume ablation results. (c) illustrates transferability ablation results.

# 5.4 Analysis of Transferability

Traditional 3D detectors struggle with transferability due to training and testing class differences. We test ImOV3D on ScanNet and SUN RGB-D on the opposite datasets. The results, shown in Figure

5(c), demonstrate that our model outperforms OV-3DET by 7.1% on SUN RGB-D and 7.82% on ScanNet. ImOV3D demonstrates superior transferability across domains despite the domain gap.

# 5.5 Analysis of Fine-tuned 2D Detector

Table 5: Comparison of fine-tuned 2D detector: Off-the-Shelf vs. Fine-Tuned Detic. 

<table><tr><td>Pretraining</td><td>Adaptation</td><td>SUNRGBD mAP@0.25</td><td>ScanNet mAP@0.25</td></tr><tr><td>-</td><td>2D Off-the-shelf + 3D Adaptation</td><td>18.8</td><td>18.96</td></tr><tr><td>Off-the-shelf + 3D Pretraining</td><td>2D Off-the-shelf + 3D Adaptation</td><td>19.67</td><td>19.25</td></tr><tr><td>2D Pretraining + 3D Pretraining</td><td>2D Adaptation + 3D Adaptation</td><td>22.53</td><td>21.45</td></tr></table>

To validate the benefits of fine-tuning Detic with pseudo images, we compare the off-the-shelf Detic to the fine-tuned version. The fine-tuned Detic shows clear advantages in handling pseudo images. On the SUNRGBD dataset, the mAP@0.25 increases from $19.67\%$ to $22.53\%$ , and on the ScanNet dataset, it rises from $19.25\%$ to $21.45\%$ (see Table 5). These experiments were conducted under the adaptation setting, illustrating the model's ability to learn from and improve detection capabilities with not entirely real data. This not only confirms the efficacy of textured image but also highlights the importance of fine-tuning models to enhance their adaptability and accuracy.

![](images/b38a2130cf3c1c7c8f26d7060de16c3575e939a320c7507aa68f89b88ce4983b.jpg)

<details>
<summary>line</summary>

| Category | Density |
| -------- | ------- |
| bed - L: 2.1, W: 1.6, H: 0.9 | 0.8 |
| cabinet - L: 0.6, W: 1.2, H: 1.0 | 0.7 |
| chair - L: 0.6, W: 0.6, H: 0.8 | 0.8 |
| table - L: 0.8, W: 1.3, H: 0.7 | 0.7 |
| desk - L: 0.7, W: 1.3, H: 0.7 | 0.8 |
| lamp - L: 0.4, W: 0.4, H: 0.7 | 0.8 |
| sofa - L: 0.9, W: 1.9, H: 0.8 | 0.7 |
| pillow - L: 0.4, W: 0.6, H: 0.3 | 0.8 |
| sink - L: 0.5, W: 0.6, H: 0.5 | 0.8 |
| bookself - L: 0.4, W: 1.1, H: 1.7 | 0.8 |
</details>

Ground Truth Point Clouds   
![](images/fb80328f8abfcf7b36d59bf1867fe05c162d3ba932bdd69c9af96859f8bad47d.jpg)

<details>
<summary>natural_image</summary>

Interior sketch of a bedroom with bed, window, and plant (no text or symbols)
</details>

OV-3DET   
![](images/0af887a6deae178ead24fbbca99f8fc4e3039c5fd95ad6f415f5f5d287edfb49.jpg)

<details>
<summary>natural_image</summary>

Interior layout diagram of a bedroom with furniture and window (no visible text or symbols)
</details>

![](images/4a903756bdaa68976a1404a0654abb70170229e66f30f065c14959ce133a8347.jpg)  
ImOV3D

![](images/8e3f2895701df1125c95d31123bde167aa68578ccdcb31ef4fe5c6c4a5ee143c.jpg)

<details>
<summary>natural_image</summary>

Interior sketch of a room with furniture and window (no visible text or symbols)
</details>

![](images/78868e43f862588245d6f3320f6f2bee0c394fdb41b9e4e4bd4e3570b8c1fef7.jpg)

<details>
<summary>line</summary>

| Rate | bed - L: 2.0, W: 1.6, H: 0.5 | cabinet - L: 0.5, W: 1.0, H: 1.0 | chair - L: 0.6, W: 0.6, H: 0.9 | table - L: 1.0, W: 1.5, H: 0.8 | desk - L: 0.7, W: 0.6, H: 0.8 | lamp - L: 0.3, W: 0.3, H: 1.5 | sofa - L: 2.0, W: 0.9, H: 0.9 | pillow - L: 0.4, W: 0.6, H: 0.2 | sink - L: 0.5, W: 0.6, H: 0.3 | bookshelf - L: 0.3, W: 1.0, H: 2.0 |
|------|-----------------------------|----------------------------------|-------------------------------|----------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|----------------------------------|
| 0    | ~0.2                        | ~0.2                             | ~0.2                          | ~0.2                             | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                             |
| 1    | ~1.4                        | ~1.4                             | ~1.4                          | ~1.4                             | ~1.4                          | ~1.4                          | ~1.4                          | ~1.4                          | ~1.4                          | ~1.4                             |
| 2    | ~0.6                        | ~0.6                             | ~0.6                          | ~0.6                             | ~0.6                          | ~0.6                          | ~0.6                          | ~0.6                          | ~0.6                          | ~0.6                             |
| 3    | ~0.2                        | ~0.2                             | ~0.2                          | ~0.2                             | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                             |
| 4    | ~0.1                        | ~0.1                             | ~0.1                          | ~0.1                             | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                             |
| 5    | ~0.1                        | ~0.1                             | ~0.1                          | ~0.1                             | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                             |
</details>

(a)

![](images/1501a97e54c105b29e4f1cf2dcb6242926b529616919c026ab343221f8cf1dde.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern office or study room with a desk, chair, and coffee table (no visible text or symbols)
</details>

![](images/e534ab5873d2a63c6bd2bd3fed52c5600b3167084bb8dd91f016983dd91d397d.jpg)

<details>
<summary>natural_image</summary>

3D architectural rendering of a modern office building (no visible text or symbols)
</details>

(b)

![](images/424ae0a2affc8f5054e43909fa566b791fb0b93400fa27d4e61ce1ebad097d77.jpg)

<details>
<summary>natural_image</summary>

3D architectural rendering of a building complex with concrete foundations and a central tower, enclosed in a blue bounding box (no visible text or symbols)
</details>

Figure 6: (a) KDE plots of volume ratios (Ratio $_{V}$ ) for top 10 classes in SUNRGBD validation set. (b) Visualization comparison of OV-3DET with ours in SUNRGBD.

# 6 Conclusion and Limitation

In conclusion, this paper introduces ImOV3D, a novel framework that tackles the scarcity of annotated 3D data in OV-3Det by harnessing the extensive availability of 2D images. The framework's key innovation lies in its flexible modality conversion, which integrates 2D annotations into 3D space, thereby minimizing the domain gap between training and testing data. Empirical results on two common datasets confirm ImOV3D's superiority over existing methods, even without ground truth 3D training data, and its significant performance boost with the addition of minimal real 3D data for fine-tuning. Our method's success showcases the potential of leveraging 2D images for enhancing 3D object detection, opening new avenues for future research in pseudo-multimodal data generation and its application in 3D detection methodologies.

Limitation: Although our method has demonstrated the potential of 2D images in OV-3Det tasks, especially with the proposed pseudo multimodal representation, we need dense point clouds here to ensure that the rendered images can help improve performance. In the future, we will explore more generalized strategies.

# 7 Acknowledge

We would like to express our gratitude to Yuanchen Ju, Wenhao Chai, Macheng Shen, and Yang Cao for their insightful discussions and contributions.

# References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[2] Gwangbin Bae, Ignas Budvytis, and Roberto Cipolla. Estimating and exploiting the aleatoric uncertainty in surface normal estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13137–13146, 2021.   
[3] Hanoona Bangalath, Muhammad Maaz, Muhammad Uzair Khattak, Salman H Khan, and Fahad Shahbaz Khan. Bridging the gap between object and image-level representations for open-vocabulary detection. Advances in Neural Information Processing Systems, 35:33781–33794, 2022.   
[4] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.   
[5] Yang Cao, Yihan Zeng, Hang Xu, and Dan Xu. Coda: Collaborative novel box discovery and cross-modal alignment for open-vocabulary 3d object detection. arXiv preprint arXiv:2310.02960, 2023.   
[6] Yang Cao, Yuanliang Jv, and Dan Xu. 3dgs-det: Empower 3d gaussian splatting with boundary guidance and box-focused sampling for 3d object detection. arXiv preprint arXiv:2410.01647, 2024.   
[7] Yang Cao, Yihan Zeng, Hang Xu, and Dan Xu. Collaborative novel object discovery and box-guided cross-modal alignment for open-vocabulary 3d object detection. arXiv preprint arXiv:2406.00830, 2024.   
[8] Boyuan Chen, Fei Xia, Brian Ichter, Kanishka Rao, Keerthana Gopalakrishnan, Michael S Ryoo, Austin Stone, and Daniel Kappler. Open-vocabulary queryable scene representations for real world planning. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 11509–11522. IEEE, 2023.   
[9] Keyan Chen, Xiaolong Jiang, Yao Hu, Xu Tang, Yan Gao, Jianqi Chen, and Weidi Xie. Ovarnet: Towards open-vocabulary object attribute recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23518–23527, 2023.   
[10] Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge Zhu, Yuexin Ma, Tongliang Liu, and Wenping Wang. Towards label-free scene understanding by vision foundation models. Advances in Neural Information Processing Systems, 36, 2024.   
[11] Han-Cheol Cho, Won Young Jhoo, Wooyoung Kang, and Byungseok Roh. Open-vocabulary object detection using pseudo caption labels. arXiv preprint arXiv:2303.13040, 2023.   
[12] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828–5839, 2017.   
[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.   
[14] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Pla: Language-driven open-vocabulary 3d scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7010–7019, 2023.   
[15] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14084–14093, 2022.   
[16] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu, et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, number 34, pp. 226–231, 1996.   
[17] Mingfei Gao, Chen Xing, Juan Carlos Niebles, Junnan Li, Ran Xu, Wenhao Liu, and Caiming Xiong. Open vocabulary object detection with pseudo bounding-box labels. In European Conference on Computer Vision, pp. 266–282. Springer, 2022.   
[18] Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. arXiv preprint arXiv:2309.16650, 2023.   
[19] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921, 2021.

[20] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5356–5364, 2019.   
[21] Deepti Hegde, Jeya Maria Jose Valanarasu, and Vishal Patel. Clip goes 3d: Leveraging prompt tuning for language grounded 3d recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2028–2038, 2023.   
[22] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al. Conceptfusion: Open-set multimodal 3d mapping. arXiv preprint arXiv:2302.07241, 2023.   
[23] Yuanchen Ju, Kaizhe Hu, Guowei Zhang, Gu Zhang, Mingrun Jiang, and Huazhe Xu. Robo-abc: Affordance generalization beyond categories via semantic correspondence for robot manipulation. arXiv preprint arXiv:2401.07487, 2024.   
[24] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19729–19739, 2023.   
[25] Dahun Kim, Anelia Angelova, and Weicheng Kuo. Detection-oriented image-text pretraining for open-vocabulary detection. arXiv preprint arXiv:2310.00161, 2023.   
[26] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10965–10975, 2022.   
[27] Ze Liu, Zheng Zhang, Yue Cao, Han Hu, and Xin Tong. Group-free 3d object detection via transformers. 2021 ieee. In CVF International Conference on Computer Vision (ICCV), pp. 2929–2938, 2021.   
[28] Shiyang Lu, Haonan Chang, Eric Pu Jing, Abdeslam Boularias, and Kostas Bekris. Ovir-3d: Open-vocabulary 3d instance retrieval without training on 3d data. In Conference on Robot Learning, pp. 1610–1620. PMLR, 2023.   
[29] Yuheng Lu, Chenfeng Xu, Xiaobao Wei, Xiaodong Xie, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. Open-vocabulary 3d detection via image-level class and debiased cross-modal contrastive learning. arXiv preprint arXiv:2207.01987, 2022.   
[30] Yuheng Lu, Chenfeng Xu, Xiaobao Wei, Xiaodong Xie, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. Open-vocabulary point-cloud object detection without 3d annotation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1190–1199, 2023.   
[31] Zeyu Ma, Yang Yang, Guoqing Wang, Xing Xu, Heng Tao Shen, and Mingxing Zhang. Rethinking open-world object detection in autonomous driving scenarios. In Proceedings of the 30th ACM International Conference on Multimedia, pp. 1279–1288, 2022.   
[32] Zongyang Ma, Guan Luo, Jin Gao, Liang Li, Yuxin Chen, Shaoru Wang, Congxuan Zhang, and Weiming Hu. Open-vocabulary one-stage detection with hierarchical visual-language knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14074–14083, 2022.   
[33] M Minderer, A Gritsenko, A Stone, M Neumann, D Weissenborn, A Dosovitskiy, A Mahendran, A Arnab, M Dehghani, Z Shen, et al. Simple open-vocabulary object detection with vision transformers. arxiv 2022. arXiv preprint arXiv:2205.06230.   
[34] Ishan Misra, Rohit Girdhar, and Armand Joulin. An end-to-end transformer model for 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2906–2917, 2021.   
[35] Benjamin Nuernberger, Eyal Ofek, Hrvoje Benko, and Andrew D Wilson. Snaptoreality: Aligning augmented reality to the real world. In Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems, pp. 1233–1244, 2016.   
[36] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 815–824, 2023.

[37] Chau Pham, Truong Vu, and Khoi Nguyen. Lp-ovod: Open-vocabulary object detection by linear probing. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 779–788, 2024.   
[38] Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9277–9286, 2019.   
[39] Charles R Qi, Xinlei Chen, Or Litany, and Leonidas J Guibas. Invotenet: Boosting 3d object detection in point clouds with image votes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4404–4413, 2020.   
[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
[41] Nur Muhammad Mahi Shafiullah, Chris Paxton, Lerrel Pinto, Soumith Chintala, and Arthur Szlam. Clip-fields: Weakly supervised semantic fields for robotic memory. arXiv preprint arXiv:2210.05663, 2022.   
[42] Cheng Shi and Sibei Yang. Edadet: Open-vocabulary object detection using early dense alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15724–15734, 2023.   
[43] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 567–576, 2015.   
[44] Daniel Wagner, Gerhard Reitmayr, Alessandro Mulloni, Tom Drummond, and Dieter Schmalstieg. Real-time detection and tracking for augmented reality on mobile phones. IEEE transactions on visualization and computer graphics, 16(3):355–368, 2009.   
[45] Luting Wang, Yi Liu, Penghui Du, Zihan Ding, Yue Liao, Qiaosong Qi, Biaolong Chen, and Si Liu. Object-aware distillation pyramid for open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11186–11196, 2023.   
[46] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. arXiv preprint arXiv:2405.01533, 2024.   
[47] Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, and Hang Xu. Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection. Advances in Neural Information Processing Systems, 35:9125–9138, 2022.   
[48] Lewei Yao, Jianhua Han, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, and Hang Xu. Detclipv2: Scalable open-vocabulary object detection pre-training via word-region alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23497–23506, 2023.   
[49] Ping-Chung Yu, Cheng Sun, and Min Sun. Data efficient 3d learner via knowledge transferred from 2d model. In European Conference on Computer Vision, pp. 182–198. Springer, 2022.   
[50] Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, and Chen Change Loy. Open-vocabulary detr with conditional matching. In European Conference on Computer Vision, pp. 106–122. Springer, 2022.   
[51] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-vocabulary object detection using captions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14393–14402, 2021.   
[52] Dongmei Zhang, Chang Li, Ray Zhang, Shenghao Xie, Wei Xue, Xiaodong Xie, and Shanghang Zhang. Fm-ov3d: Foundation model-based cross-modal knowledge blending for open-vocabulary 3d detection. arXiv preprint arXiv:2312.14465, 2023.   
[53] Junbo Zhang, Runpei Dong, and Kaisheng Ma. Clip-fo3d: Learning free open-world 3d scene representations from 2d dense clip. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2048–2059, 2023.   
[54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023.

[55] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Region-based language-image pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16793–16803, 2022.   
[56] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision, pp. 350–368. Springer, 2022.   
[57] Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyao Zeng, Shanghang Zhang, and Peng Gao. Pointclip v2: Adapting clip for powerful 3d open-world learning. arXiv preprint arXiv:2211.11682, 2022.