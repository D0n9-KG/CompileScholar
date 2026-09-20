# SyntheOcc: Synthesize Geometric-Controlled Street View Images through 3D Semantic MPIs

Leheng Li $^{1}$ Weichao Qiu $^{3}$ Yingjie Cai $^{3}$ Xu Yan $^{3}$ Qing Lian $^{2}$ Bingbing Liu $^{3}$ Ying-Cong Chen $^{1,2*}$

HKUST(GZ) $^{1}$ HKUST $^{2}$ HUAWEI Noah's Ark Lab $^{3}$ Project page: len-li.github.io/syntheocc-web

# Abstract

The advancement of autonomous driving is increasingly reliant on high-quality annotated datasets, especially in the task of 3D occupancy prediction, where the occupancy labels require dense 3D annotation with significant human effort. In this paper, we propose SytheOcc, which denotes a diffusion model that Synthesize photorealistic and geometric-controlled images by conditioning Occupancy labels in driving scenarios. This yields an unlimited amount of diverse, annotated, and controllable datasets for applications like training perception models and simulation. SyntheOcc addresses the critical challenge of how to efficiently encode 3D geometric information as conditional input to a 2D diffusion model. Our approach innovatively incorporates 3D semantic multi-plane images (MPIs) to provide comprehensive and spatially aligned 3D scene descriptions for conditioning. As a result, SyntheOcc can generate photorealistic multi-view images and videos that faithfully align with the given geometric labels (semantics in 3D voxel space). Extensive qualitative and quantitative evaluations of SyntheOcc on the nuScenes dataset prove its effectiveness in generating controllable occupancy datasets that serve as an effective data augmentation to perception models.

# 1 Introduction

With the rapid development of generative models, they have shown realistic image synthesis and diverse controllability. This progress has opened up new avenues for dataset generation in autonomous driving $[5, 12, 24, 31]$ . The task of dataset generation is usually modeled as controllable image generation, where the ground truth (e.g. 3D Box) is employed to control the generation of new datasets in downstream tasks (e.g. 3D detection). This approach helps to mitigate the data collection and annotation effort as it can generate labeled data for free. However, a novel task of vital importance, occupancy prediction $[25, 28]$ , poses new challenges for dataset generation compared with 3D detection. It requires finer and more nuanced geometry controllability, which refers to use the occupancy state and semantics of voxels in the whole 3D space to control the image generation. We argue that solving this problem not only allows us to synthesize occupancy datasets, but also empowers valuable applications such as editing geometry to generate rare data for corner case evaluation, as shown in Fig. 1. In the following, we first illustrate why prior work struggles to achieve the above objective, and then demonstrate how we address these challenges.

In the area of diffusion models, several representative works have displayed high-quality image synthesis; however, they are constrained by limited 3D controllability: they are incapable of editing 3D voxels for precise control. For example, BEVGen [24] generates street view images by conditioning BEV layouts using diffusion models. MagicDrive [5] extend BEVGen and additionally converts the

![](images/abe112dc999f47247a14338d9898b8fc955bd7b75998ac495294312f2d286bbf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Add traffic cone"] --> B["Original occupancy"]
    B --> C["SyntheOcc"]
    C --> D["Generated image"]
    D --> E["Testing End-to-end Planner"]
    E --> F["Predicted waypoints"]
    G["Edited occupancy"] --> H["Generated image"]
    H --> I["Predicted waypoints"]
```
</details>

Figure 1: A showcase of application of SytheOcc. We enable geometric-controlled generation that conveys the user editing in 3D voxel space to generate realistic street view images. In this case, we create a rare scene that traffic cones block the way. This advancement facilitates the evaluation of autonomous systems, such as the end-to-end planner VAD [9], in simulated corner case scenes.

3D box parameters into text embedding through Fourier mapping that is similar to NeRF $[20]$ , and uses cross-attention to learn conditional generation. Although these methods achieve satisfactory results in image generation, their 3D controllability is inherently limited. These approaches are restricted to manipulating the scene in types of 3D boxes and BEV layouts, and hardly adapt to finer geometry control such as editing the shape of objects and scenes. Meanwhile, they usually convert conditional input into 1D embedding that aligns with prompt embedding, which is less effective in 3D-aware generation due to lack of spatial alignment with the generated images. This limitation hinders their utility in downstream applications, such as occupancy prediction and editing scene geometry to create long-tailed scenes, where granular volumetric control is paramount in both tasks.

ControlNet [42] and GLIGEN [14] is another type of prominent method in the field of controllable image generation. These approaches exhibit several desirable attributes in terms of controllability. They leverage conditional images such as semantic masks for control, thereby offering a unified framework to manipulate both foreground and background. However, despite its precise spatial control, ControlNet does not align with our specific requirements. Their conditions of pixel-level images differ fundamentally from what we require in 3D contexts. Our experimental results also find that ControlNet struggles to handle overlapping objects with varying depths (see Fig. 6 (a)), as it only utilizes an ambiguous 2D semantic map as conditional input. As a result, it is non-trivial to extend the ControlNet framework and convey their desirable attributes for 3D conditioning.

To address the above challenges, we propose an innovative representation, 3D semantic multi-plane images (MPIs), which contribute to image generation with finer geometric control. In detail, we employ multi-plane images [44] to represent the occupancy, where each plane represents a slice of semantic label at a specific depth. Our 3D semantic MPIs not only preserve accurate and authentic 3D information, but also keep pixel-wise alignment with the generated images. We additionally introduce the MPI encoder to encode features, and the reweighing methods to ease the training with long-tailed cases. As a collection, our framework enables 3D geometry and semantic control for image generation and further facilitates corner case evaluation as depicted in Fig. 1. Finally, experimental results demonstrate that our synthetic data achieve better recognizability, and are effective in improving the perception model on occupancy prediction. In summary, our contributions include:

- We present SytheOcc, a novel image generation framework to attain finer and precise 3D geometric control, thereby unlocking a spectrum of applications such as 3D editing, dataset generation, and long-tailed scene generation.   
- Incorporating the proposed 3D semantic MPI, MPI encoder, and reweighing strategy, we deliver a substantial advancement in image quality and recognizability over prior works.   
- Our extensive experimental results demonstrate that our synthetic data yields an effective data augmentation in the realm of 3D occupancy prediction.

# 2 Related Work

# 2.1 3D Occupancy Prediction

The task of 3D occupancy prediction aims to predict the occupancy status of each voxel in 3D space, as well as its semantic label if occupied. Compared with previous perception methods like 3D object detection, occupancy prediction offers a more detailed and nuanced understanding of the environment, as it provides finer geometric details, is capable of handling general, out-of-vocabulary objects, and finally, enriches the planning stack with comprehensive 3D information. Early methods exploited LiDAR as inputs to complete the 3D occupancy of the entire 3D scene $[19, 34]$ . Recent methods began to explore the more challenging vision-based 3D occupancy prediction $[25, 26, 28, 30]$ . By predicting the geometric and semantic properties of both dynamic and static elements, 3D occupancy prediction offers a more comprehensive understanding of the surrounding environment.

# 2.2 Diffusion-based Image Generation

Recent advancements in diffusion models (DMs) have achieved remarkable progress in image generation. In particular, Stable Diffusion (SD) $[22]$ employs DMs within the latent space of autoencoders, striking a balance between computational efficiency and high image quality. Beyond text control, there is also the introduction of additional control signals. A noteworthy work is ControlNet $[42]$ , which incorporates a trainable copy of the SD encoder to extract the feature of conditional images and adds it to the UNet feature. It significantly enhances the controllability and unlocking pathways for advanced applications. We refer readers to recent survey $[36]$ for more details.

# 2.3 Image Generation in Autonomous Driving

As training neural networks relies heavily on labeled data, numerous studies are delving into dataset generation to boost training. Lift3D $[12]$ designs generative NeRF to synthesize labeled datasets for 3D detection for the first time. Several other works employ BEV layouts to synthesize image data, proving beneficial for perception models. For example, BEVGen $[24]$ conditions BEV layouts to generate multi-view street images, while BEVControl $[35]$ separately generates foregrounds and backgrounds from BEV layouts. MagicDrive $[5]$ generates images with 3D geometry controls by independently encoding objects and maps through a text encoder or map encoder. Compared with MagicDrive, our geometry control is characterized by a more detailed and lossless representation of 3D scenes for control, which poses significant challenges than projected layout or box embedding.

Recently, DriveDreamer $[27]$ , DrivingDiffusion $[13]$ , Drive-WM $[29]$ and Panacea $[31]$ use a ControlNet framework, which involves projecting bounding boxes and road maps onto 2D FoV images as a conditioning input. This approach has proven to be effective for geometric control. However, it is limited in that it only achieves alignment at the 2D-pixel level. Consequently, this method falls short in capturing the depth hierarchy and fails to account for the occlusion relationships present in the 3D real world. Besides, adding a depth channel like Panacea $[31]$ may address the limitations of depth order, but it discards the occluded part and only contains partial observation. UrbanGiraffe $[38]$ train a generative NeRF to perform image generation. WoVoGen $[18]$ creates a 4D world volume feature using occupancy to guide the generation, but seems to rely on object mask guidance.

As described above, most of the prior work is restricted by only modeling a projected primitive of 3D boxes and road maps as conditions. They suffer from ill-posed un-projection ambiguity. In contrast, we model 3D occupancy labels as conditions, as they provide finer geometric details and semantic information. However, designing an input representation of 3D occupancy labels into a 2D diffusion model is challenging. In this paper, we propose a novel representation: 3D semantic Multi-Plane Images (MPIs) as conditional inputs, which not only provide spatial alignment that improves visual consistency, but also encode comprehensive 3D geometric information including occluded parts.

# 3 Method

Overview The overview of our method is depicted in Fig. 2. Built upon the SD pipeline, we aim to perform geometry-controlled image generation by conditioning on 3D geometry labels with semantics (occupancy labels). One requirement is that the images should faithfully align with the given label. This task is more challenging than conditioned on 3D box due to the sparse and irregular nature of occupancy. We first discuss how to efficiently represent occupancy in Sec. 3.2, followed by our designed MPI encoder to enhance generation quality in Sec. 3.3, and reweighing strategy to handle the long-tailed depth and category in Sec. 3.5.

![](images/d04768dc31961c957e642fa3aad5d24c2a7c2b9c1fa5e3b7c76d8d9e0014ad6e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Occupancy Label"] --> B["Optional User Edit: Add Delete ..."]
    B --> C["3D Semantic Multiplane Images"]
    C --> D["N×D×H×W"]
    D --> E["MPI Encoder"]
    E --> F["Conv1×1"]
    E --> G["ReLU"]
    E --> H["Conv1×1"]
    E --> I["UNet Encoder"]
    F --> J["Long-tailed Scene Generation"]
    G --> K["Training Perception Model"]
    H --> L["Corner Case Evaluation"]
    I --> M["Simulation"]
    N["Latent Noise N×4×H×W"] --> O["Diffusion UNet"]
    P["Prompt: show a realistic street view image"] --> O
    O --> Q["Intra View"]
    O --> R["Cross View/Frame"]
    Q --> S["VAE Decoder"]
    R --> S
    S --> T["Multi-view images"]
```
</details>

Figure 2: The overall architecture of SytheOcc. We achieve 3D geometric control in image generation by utilizing our proposed 3D semantic multiplane images to encode scene occupancy. In our framework, we can edit the occupied state and semantics of every voxel in 3D space to control the image generation, thereby opening up a wide spectrum of applications as shown in the top right.

# 3.1 Representation of Condition: Local Control Aligns Better than Global Control

One of the key challenges is how to represent our conditional occupancy input. A straightforward method $[3, 5]$ is to convert the 3D occupancy voxel to 1D global embedding that is similar to text embedding, and then use cross-attention to learn controllable generation. However, these global methods can be less effective when dealing with dense or irregular data due to the following reasons: (i) They perform controllable generation through hard encoding the spatial relationship between 1D global embedding and 2D UNet features. (ii) Ignore the underlying geometry alignment between the conditional input and the generated image. In contrast, local methods like ControlNet, directly add spatial features to the UNet features, providing 2D local control with pixel-level spatial alignment. They are better than the global method (see Tab. 1), but suffer from 3D ambiguity (see Fig. 6(a)). Consequently, this comparison motivates us to seek a more compact and efficient manner to encode and condition our 3D occupancy labels.

# 3.2 Represent Occupancy as 3D Semantic Multiplane Images

It is non-trivial to design a 3D representation for conditioning. To efficiently store both the semantic and geometric information of the irregular occupancy input, we propose to use multiplane images (MPIs) [44] as representation. An MPI is composed of a series of fronto-parallel RGBA layers within the frustum of the source camera with a specific viewpoint. These planes are arranged at varying depths, from $d_{min}$ to $d_{max}$ , starting from the nearest to the farthest. Each layer of these images contains both an RGB image and an alpha map, which collectively capture the visual and geometric details of the scene at the respective depth. In our work, instead of storing RGB value and alpha map in the original MPI, we store our 3D semantic labels. Each layer of MPI represents the semantic index at the corresponding depth. We display the colored MPI in the top row of Fig. 2 for visual clarity, but we actually use the integer index for learning. We obtain our 3D semantic MPI by:

$$
P _ {l} = (u \times d _ {l}, v \times d _ {l}, d _ {l}) ^ {T}, d _ {l} = d _ {\min} + (d _ {\max} - d _ {\min}) \times l / D, \tag {1}
$$

$$
\mathbf {M P I} _ {n, l} = \text { Interpolate } (0 \text { occupancy }, \mathbf {T} _ {\mathbf {n}} \cdot \mathbf {K} _ {\mathbf {n}} ^ {- 1} \cdot P _ {l}), \tag {2}
$$

$$
\mathbf {M P I} = \text { Concatenate } (\mathbf {M P I} _ {i, j}), i \in (0, N), j \in (0, D), \tag {3}
$$

where $(u,v)$ is a pixel coordinate in image space, $d_{l}$ is depth value of the $l^{th}$ layer, n denotes the $n^{th}$ camera view. This equation implies we first back project points P in camera frustum space $(u,v,d)$ to Euclid space $(x,y,z)$ by multiplying inverse intrinsic $K^{-1}$ . Then we use transformation matrix T to map points from camera coordinates to occupancy coordinates. We then use the point coordinates to interpolate the nearest semantic index from the dense occupancy voxel to form a slice of MPI. Finally, we concatenate all slices to form $MPI \in R^{N \times D \times H \times W}$ , where D is the number of layers that is set at 256, N is the number of camera views in the case of batch size = 1.

By representing occupancy as 3D semantic MPI, every pixel in MPI contains geometry and semantic information with implicit depth, seamlessly integrating occluded elements, and ensuring a precise spatial alignment with the generated images.

(a) Raw generation   
![](images/3d78c73bea5b7e51a2b66b56ec95fb640b647ca05d265bf7527082d0a87f0975.jpg)

<details>
<summary>natural_image</summary>

Composite image showing a street scene with cars and buildings, overlaid with a colorful abstract graphic (no text or symbols)
</details>

(b) Editing: Object manipulation (copy object)   
![](images/a3ba6fa770ee146e51f26ac218513f17605afdead7e3ce70a439f883b017d64d.jpg)

<details>
<summary>natural_image</summary>

Composite image showing a colorful abstract overlay over a street scene with cars and trees (no visible text or symbols)
</details>

(c) Editing: Foreground removal   
![](images/318322f59b5f23c37cd5c9521dd044064df14624be4b91a8bf6b3b9dc8ae9fac.jpg)

<details>
<summary>natural_image</summary>

Street scene with trees and a road, overlaid with a color-coded map or diagram (no readable text or symbols)
</details>

Figure 3: Visualizations of geometric controlled generation. Top row: Fusion of 3D semantic MPI. Bottom row: our generation concatenated from neighboring views.

# 3.3 3D Semantic MPI Encoder

To enable local control with spatially aligned conditions, we develop a simple but effective MPI encoder that aligns the 3D multi-plane feature to the latent space of the diffusion model. The purpose of the MPI encoder is to obtain features from multi-plane images to perform 3D-aware image synthesis. Unlike the original ControlNet which downsampling conditional input through $3 \times 3$ convolutions with padding, we design a $1 \times 1$ convolutional encoder without downsampling to encode features. In detail, the 3D multiplane features which have the sample resolution with latent features, are transformed by a $1 \times 1$ convolution layer and ReLU activation [1] in the MPI encoder.

After obtaining the multi-scale feature after the MPI encoder, we add the feature to the decoder of diffusion UNet to provide spatial features. Experimental results in Tab. 3 will show that our $1 \times 1$ conv in MPI encoder is more effective than $3 \times 3$ conv, as the $1 \times 1$ conv with receptive field = 1 provides a spatial align feature to the latent feature in the diffusion UNet. In contrast, $3 \times 3$ conv is conducted in a camera frustum space rather than Euclid space, making an imprecise correspondence between 3D multiplane features and 2D image features. Moreover, using $3 \times 3$ conv to process 3D semantic MPI will introduce a large computational burden as the channel number increases from 3 channels of RGB to 256 planes. We display our 3D geometry and semantic control property in Fig. 3.

In summary, we chose MPIs as the representation because they (i) Incorporate lossless 3D information, including scene geometry rather than 2.5D depth. (ii) Provide spatially aligned conditional features that naturally extend the ControlNet framework from image level to 3D level. (iii) Capable of representing geometry and semantics including occluded elements.

# 3.4 Cross-View and Cross-Frame Attention

The sensor arrangement in a self-driving car usually requires a full surround view of cameras to capture the entire 360-degree environment. To effectively simulate the multi-view and subsequent multi-frame generation, zero-initialized $[42]$ cross-view and cross-frame attention are integrated into the diffusion model to maintain consistency between views and frames. Following prior work $[5, 29, 31, 32]$ , each cross-view attention allows the target view to access information from its neighboring left and right views, thus training cross-view attention using multi-view consistent images will enforce it to generate the same instance in the overlapping region of multi-view cameras.

$$
\text { Attention } (Q, K, V) = \operatorname{softmax} (\frac {Q K ^ {T}}{\sqrt {d}}) \cdot V, \tag {4}
$$

$$
h _ {o u t} = h _ {i n} + \sum_ {i \in \{l, r \}} \text { Attention } (Q _ {i n}, K _ {i}, V _ {i}), \tag {5}
$$

where l, and r is the camera view of left and right. $Q_{in}$ and $h_{in}$ denotes the query and the hidden state of input view. Similarly, we add cross-frame attention that attend previous frame and future frame to enable video generation. In this case, we use the same formulation while $i \in \{f, h\}$ , where f and h is the camera view of future and history frames.

# 3.5 Importance Reweighing

To deal with the extreme imbalance problem between foreground, background, and object categories, and also to ease the training, we propose three types of reweighting methods to improve the generation quality of foreground objects.

Progressive Foreground Enhancement To mitigate the complexity of the learning task, we propose a progressive reweighting method that incrementally enhances the loss associated with the foreground regions

![](images/8a6a74cbc6e12c1a3e221a1e1a6dd7d68af6e87c74062693fc3312c81b0a85df.jpg)

<details>
<summary>line</summary>

| x    | m = 3, n = 700 | m = 3, n = 1000 | m = 2, n = 1000 |
| ---- | -------------- | --------------- | --------------- |
| 0    | 1.00           | 1.00            | 1.00            |
| 200  | ~1.50          | ~1.25           | ~1.15           |
| 400  | ~2.50          | ~1.85           | ~1.45           |
| 600  | ~3.00          | ~2.50           | ~1.75           |
| 800  | ~3.00          | ~2.75           | ~1.95           |
| 1000 | ~3.00          | ~3.00           | ~2.00           |
</details>

Figure 4: Visualizations of the reweighing function in Eq. 6.

![](images/d8a23fc7cbcd4149c555687b01198efc011daa302ce308f6bd262f184e9f6ca5.jpg)  
Figure 5: Visualizations of generated multi-view images. The generation conditions (occupancy labels) are from nuScenes validation set. We highlight that (i) Geometry alignment of trees in red rectangle in (b). (ii) Use text prompt to control high-level appearance in (c,d).

(based on semantic class) as the training progresses.

The detailed formulation is:

$$
w (x, m, n) = \frac {(m - 1)}{2} \cdot (1 + \cos (\frac {x}{n} \cdot \pi + \pi)) + 1, \tag {6}
$$

where x is the current training step, m is the maximum value of weights that set at 2, and n is the total training steps. This approach is engineered to facilitate a learning trajectory that progresses from simplicity to complexity, thereby aiding in the convergence of the model. This curve can be interpreted as a cosine annealing but inverted to amplify the importance of the foreground region.

Depth-aware Foreground Reweighing In the meantime, we acknowledge the learning difficulty in different depth places in 3D scenes. Following GeoDiffusion [3], we perform depth reweighing to foreground objects by adaptively assigning higher weights to farther foreground areas. This enables the model to focus more thoroughly on hard examples with depth-aware importance reweighting. Instead of using their exponential function to increase weights, we use our designed cosine function Eq. 6 for stability. Here x is the input depth value, and n is the maximum depth that set at 50.

CBGS Sampling To deal with the class imbalance problem in driving scenarios, where certain object categories appear infrequently, we employ the Class-Balanced Grouping and Sampling (CBGS) [45] to better handle the long-tailed classes. CBGS addresses the challenge of class imbalance by grouping and re-sampling training data to ensure each group has a balanced distribution of sample frequency across different object categories. This method reduces the bias towards more frequent classes and enables better generalization to rare scenarios.

<table><tr><td>Method</td><td>Train</td><td>Val</td><td>mIoU</td><td>barrier</td><td>bicycle</td><td>bus</td><td>car</td><td>cons. veh.</td><td>moto.</td><td>pedes.</td><td>traf. cone</td><td>trailer</td><td>truck</td><td>drive. suf.</td><td>other flat</td><td>sidewalk</td><td>terrain</td><td>manmade</td><td>vegetation</td></tr><tr><td>Oracle (FB-Occ [15])</td><td>Real</td><td>Real</td><td>39.3</td><td>45.4</td><td>28.2</td><td>44.1</td><td>49.4</td><td>25.9</td><td>28.8</td><td>28.0</td><td>27.7</td><td>32.4</td><td>37.3</td><td>80.4</td><td>42.2</td><td>49.9</td><td>55.2</td><td>42.0</td><td>37.7</td></tr><tr><td>SytheOcc-Aug</td><td>Real+Gen</td><td>Real</td><td>40.3</td><td>45.4</td><td>27.2</td><td>46.6</td><td>49.5</td><td>26.4</td><td>27.8</td><td>28.4</td><td>29.4</td><td>34.0</td><td>37.2</td><td>81.3</td><td>46.0</td><td>52.4</td><td>56.5</td><td>43.3</td><td>38.9</td></tr><tr><td>MagicDrive</td><td>Real</td><td>Gen</td><td>13.4</td><td>0.7</td><td>0.0</td><td>11.8</td><td>32.4</td><td>0.0</td><td>6.6</td><td>2.8</td><td>0.3</td><td>2.6</td><td>19.6</td><td>60.1</td><td>12.1</td><td>26.2</td><td>23.4</td><td>15.5</td><td>12.8</td></tr><tr><td>ControlNet</td><td>Real</td><td>Gen</td><td>17.3</td><td>17.7</td><td>0.2</td><td>13.6</td><td>21.0</td><td>0.6</td><td>0.8</td><td>8.6</td><td>10.4</td><td>6.9</td><td>11.9</td><td>67.4</td><td>18.8</td><td>36.4</td><td>36.9</td><td>20.8</td><td>22.4</td></tr><tr><td>ControlNet+depth</td><td>Real</td><td>Gen</td><td>17.5</td><td>19.3</td><td>0.3</td><td>14.0</td><td>23.7</td><td>1.0</td><td>0.6</td><td>9.2</td><td>9.2</td><td>5.7</td><td>12.1</td><td>68.8</td><td>19.2</td><td>36.0</td><td>35.3</td><td>19.8</td><td>22.8</td></tr><tr><td>SytheOcc-Gen</td><td>Real</td><td>Gen</td><td>25.5</td><td>32.6</td><td>13.8</td><td>27.7</td><td>33.4</td><td>7.5</td><td>6.5</td><td>15.7</td><td>16.5</td><td>16.5</td><td>25.6</td><td>74.3</td><td>24.5</td><td>39.4</td><td>40.5</td><td>28.6</td><td>28.8</td></tr></table>

Table 1: Downstream evaluation on the nuScenes-Occupancy validation set. Based on the used train and val data, two types of settings are reported. The first is to use generated training set to augment the real training set, and evaluate on the real validation set, denoted as Aug. The second is to use pretrained models trained on the real training datasets to test on the generated validation set, denoted as Gen.

# 3.6 Model Training

To ease the training of the MPI encoder and added attention module, we use a two stage training pipeline. We first train MPI encoder and cross-view attention in a multi-view image generation setting. Then we train cross-frame attention and freeze other components in a video generation setting.

Objective Function Our final objective function can be formulated as a standard denoising objective with reweighing:

$$
\mathcal {L} = \mathbb {E} _ {\mathcal {E} (x), \epsilon , t} \| \epsilon - \epsilon_ {\theta} (z _ {t}, t, \tau_ {\theta} (y)) \| ^ {2} \odot w, \tag {7}
$$

where w is the multiplication of progressive reweighing and depth-aware reweighing.

# 4 Experiments

# 4.1 Dataset and Setups

We conduct our experiments on the nuScenes dataset $[2]$ , which is collected using 6 surrounded-view cameras that cover the full $360^{\circ}$ field of view around the ego-vehicle. It contains 700 scenes for training and 150 scenes for validation. We resize the original image from $1600 \times 900$ to $800 \times 448$ for training. In our work, we use the occupancy label with a resolution of 0.2m from OpenOccupancy $[28]$ as condition input, while the benchmark of occupancy prediction uses a resolution of 0.4m from Occ3D $[25]$ dataset for its popularity.

Networks We use Stable Diffusion $[22]$ v2.1 checkpoint as initialization and only train occupancy encoder, cross-view attention. We additionally add cross-frame attention if in video experiments. We adopt FB-Occ $[15]$ as the target model for occupancy prediction for its SOTA performance in this task. The pretrained checkpoint of the network is obtained from their official repository. Since FB-Occ predicts occupancy using only single frame images, we thus train SyntheOcc without cross-frame attention in related experiments. For video generation, we provide experimental results in appendix.

Metrics We use Frechet Inception Distance (FID) [6] to measure the perceptual quality of generated images, and use mIoU to measure the precision of occupancy prediction.

Hyperparameters We set D = 256, $d_{min} = 0$ and $d_{max} = 50$ . The depth resolution of MPI is thus higher than occupancy voxel. We train our model in 6 epochs with batch size = 8. The learning rate is set at $2e^{-5}$ . The training phase takes around 1 day using 8 NVIDIA A100 80G GPUs. We use UniPC scheduler [43] with the classifier-free guidance (CFG) [7] that is set as 7.0. During inference, we use 20 denoising steps for dataset generation.

Baselines We compare our method with prior methods in Tab. 1. ControlNet denotes we train a ControlNet using an RGB semantic mask as the condition. ControlNet+depth denotes we add a depth channel after the semantic mask to provide 2.5D depth information. The depth map rendered by occupancy is normalized to [0-255] to accommodate the RGB value. The ControlNet+depth can be regarded as a degradation of SytheOcc which is reduced to a single plane. Then we evaluate MagicDrive since it is the only open-sourced method in this area. MagicDrive separately encodes foreground and background using prompt and BEV layout. Furthermore, we evaluate the image

![](images/6ca1634be2e17e4902709e27f4826d30afc040ac650bb451c0c23a73f8bdb2b7.jpg)  
(a0) Fusion of MPI

![](images/af356b316564491462f492e5df7222c6e9e05b8a347a655ad299527575b2910e.jpg)  
(a1) Occupancy in 3D space

![](images/393eee07b16591b7aa8a66aab2e0ee4d9ca27f8ed65985c122e9d0d0c5fe9b14.jpg)  
(a2) ControlNet generation

![](images/48c43a1b6eb4cb8e7e581a9e6b7bc825c5769bfe5904282fb6289829d93f8e9b.jpg)  
(a3) Ours generation

![](images/fd19af50f9d55e7d1058718afbed1362620c8b6027a729a0724c4d812751da46.jpg)  
(a4) Ground truth

![](images/32043943363b8def2c53b12ead8acfe1dc0b51a87a9cb92f5d8d1a95aba3e22d.jpg)  
(b) A scene with human

![](images/98a4fe1985cd2a698f6a14fe12d520dab0905e0051b8f1dac174dce3ba13f9e5.jpg)

![](images/74755d45c557d34312ed4dc2f152ca53ba6b7190cff3bdf891318934a4f930f1.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a commercial street with parked trucks and buildings in the background (no visible text or signage)
</details>

(d) A scene with hinged-articulated trucks

![](images/8c1cbfa4c02795c77923b6daf9fa81a3e44594e2551819761dc0b1d6cd73d904.jpg)

![](images/e4970dca7dd2caa9670e18149af262555c6b905ae5eb017136dc695a50035934.jpg)

(c) A scene of seasonal changes use prompt control   
![](images/6829fc6e823d9dc3d3b5d2b4795e316681b1f4747050aa31f7654339c5a9d3af.jpg)

![](images/e7a6a1f7cbf4b05237f989f24ac968a43f41b16ddf5fd79709fe6995640c8484.jpg)

![](images/09f13c8631d80b9a1df922a52985496fbad629e77e72e28e842c93b6c13a8b07.jpg)  
(e) A scene with excavator use geometry control   
Figure 6: Top row: Comparison with ControlNet. We achieve a precise alignment between conditional labels and synthesized images, while ControlNet generates objects with incorrect pose due to ambiguous 2D condition. Mid and Bottom row: Visualizations of geometry-controlled image generation. We can faithfully generate objects with the desired topology in a specific 3D position.

quality (FID [6]) of our method in Tab. 2. Compared with prior methods, we use a unified 3D representation that seamlessly handles foreground and background, surpassing them by a large margin.

# 4.2 Qualitative Results

High-level Control using Prompt In Fig. 5 (c,d) and Fig. 6 (c), we demonstrate the capability to employ user-defined prompts to generate images with specific weather conditions and high-level style. Although the nuScenes dataset doesn't contain rare weather images like snow and sandstorms, our method successfully conveys prior knowledge pretrained from stable diffusion to our scenes. Compared with visualization results in prior work like Fig. 8 of MagicDrive, our method shows better alignment with the text prompt, demonstrating the cross-domain generalization ability of our method.

3D Geometric Control Our flexible framework enables us to create novel scenes by manipulating voxels as displayed in Fig. 1 and Fig. 3. Basically, we can edit the occupied state and semantics of every voxel in our scenes for generation. We highlight that we can create a hinged-articulated truck and an excavator as shown in Fig. 6 (d,e). The generated excavator image exhibits a remarkable alignment with the input occupancy that is delineated by a black outline.

Long-tailed Scene Generation The flexibility of 3D semantic MPI has conferred significant advantages upon our approach. In the following, we create long-tail scenes that rarely occur in our real world for evaluation. In Fig. 1, we show that we manually add parallel traffic cones in front of the ego vehicle. This scene has never happened in the training dataset, but our geometric controllability provides us the capability to create such data. We then use the created scene to test autonomous driving systems such as end-to-end planner VAD [9] to validate its effectiveness. In this case, VAD successfully predicts correct waypoints with the high-level command ‘turn left’. Moreover, in appendix Sec. B, we generate long-tailed scenes with extreme weather such as snow and sandstorms, and evaluate perception model on it to examine its generalizability of rare weather.

<table><tr><td>Method</td><td>Condition Type</td><td>FID</td></tr><tr><td>BEVGen [24]</td><td>BEV map</td><td>25.54</td></tr><tr><td>BEVControl [35]</td><td>BEV map</td><td>24.85</td></tr><tr><td>DriveDreamer [27]</td><td>Box + FoV map</td><td>52.60</td></tr><tr><td>MagicDrive [5]</td><td>Box + BEV map</td><td>16.20</td></tr><tr><td>Panacea [31]</td><td>Box + FoV map</td><td>16.96</td></tr><tr><td>Ours</td><td>3D Semantic MPI</td><td>14.75</td></tr></table>

Table 2: Comparison of FID with previous methods on the nuScenes dataset.

<table><tr><td>MPI Encoder</td><td colspan="3">Reweighing Method</td><td>Metric</td></tr><tr><td>Design</td><td>Progressive</td><td>Depth</td><td>CBGS</td><td>mIoU</td></tr><tr><td>3×3</td><td>-</td><td>-</td><td>-</td><td>21.96</td></tr><tr><td>1×1</td><td>-</td><td>-</td><td>-</td><td>23.05</td></tr><tr><td>1×1</td><td>√</td><td>-</td><td>-</td><td>23.63</td></tr><tr><td>1×1</td><td>√</td><td>√</td><td>-</td><td>24.40</td></tr><tr><td>1×1</td><td>√</td><td>√</td><td>√</td><td>25.50</td></tr></table>

Table 3: Ablation of different designs of the MPI encoder and reweighing methods.

Comparison with Baselines In Fig. 6 (a), we visualize a comparison with ControlNet. We find that ControlNet struggles to distinguish the overlapping instances in 2D-pixel space. This leads to the two parked cars being merged into a single car with incorrect pose. In contrast, our 3D semantic MPIs contain more than 2D semantic mask, but also account for complete scene geometry with occluded parts. Together with our proposed MPI encoder and reweighing strategy, our framework yields a realistic image generation with high-quality label alignment. More comparison is provided in Sec. D.

# 4.3 Quantitative Results

Recognizability, Realism and Controllability Evaluation To evaluate whether our generated images aligned with given annotations, we provide Gen experiment in Tab. 1. Using the annotation of val set, we synthesize a copy of val set's images, then use perception model trained on real training set to perform evaluation. The performance will be more effective as it is close to the oracle performance. We find that local method (ControlNet) perform better than global method (MagicDrive). Furthermore, SytheOcc generalizes the locality for 3D conditioning to yield better performance.

Data Augmentation for 3D Occupancy Prediction Notably, we conduct experiments using our synthesized dataset to enhance the real training set in Tab. 1. We first use the occupancy labels from training set to create a synthetic training set. Then we modify the loading pipeline in perception model to randomly sample images from real dataset or synthetic dataset and train network from scratch. Therefore, our approach preserves the inherent training dynamics of the neural network by solely modifying the training images, without any alteration to the number of training iterations or epochs. As MagicDrive-Aug exhibits numerical overflow when training FB-Occ, which may attributed to unsatisfactory recognizability, we have to omit it and only provide MagicDrive-Gen experiments.

As shown in Tab. 1, where SytheOcc-Aug denotes the augmentation experiments using our generated dataset, shows a satisfactory improvement over the prior state of the art. We emphasize that surpassing the performance of the original dataset is not the primary objective of our work; rather, it is an ancillary benefit that emerges from our framework for geometry-controlled generation.

Ablations In Tab. 3, we present ablation studies across several design spaces of our model, analogous to the Gen experiment in Tab. 1. We find that our designed MPI encoder of $1 \times 1$ conv have significant improvement when compared to the conventional $3 \times 3$ conv approach. Besides, our proposed three types of reweighing methods demonstrate a consistent improvement over the baseline. As a result, the improved image quality and label alignment enable higher precision in downstream tasks.

# 5 Limitation and Broader Impacts

Layout Generation Our method is restricted in a conditional generation framework that should have a conditional input at first. Our condition signal is from the original dataset annotation. Thus most of the augmented data is generated using the same occupancy layout, or with minimal human editing. Future research can incorporate the recent research $[10,16,18,33,41]$ that generates occupancy and traffic descriptions of the scenes to synthesize images with novel occupancy or traffic layouts.

Closed-loop Simulation Given the underlying diverse and controllable image generation of our method, it would be advantageous and valuable to extend our work to a broader domain such as closed-loop simulation $[17,39]$ , to enable high-fidelity autonomous systems testing. This line of work can be conducted by utilizing motion conditions to generate future frames as in world model $[18,29,37]$ , or by explicitly modeling scene graph as in the case of UniSim $[21,39]$ and NeuroNCAP $[17]$ .

Long-tailed Scene Generation In this paper, we only investigate a limited number of long-tailed scene generation and corner case evaluations such as rare layout in Fig. 1 and extreme weather in Sec. B. Future work can extend our framework to (i) Synthesize more samples for tail classes to boost performance. (ii) Generate or replicate large-scale databases of corner cases [11] for robust perception.

# 6 Conclusion

In this paper, we propose SytheOcc, an innovative image generation framework that is empowered with geometry-controlled capabilities using occupancy. We introduce a novel 3D representation, 3D semantic MPIs, to address the critical challenge of how to efficiently encode occupancy. This representation not only preserves the authentic and complete 3D geometry details with semantics, but also provides a spatial-align feature representation for 2D diffusion models. With this property, our

method enjoys photorealistic appearances and fine-grained 3D controllability, serves as a generative data engine to enable a broad range of applications. Extensive experiments demonstrate that our synthetic data facilitate the training for perception models on occupancy prediction, and provide valuable corner case evaluation in a simulated world.

# References

[1] Abien Fred Agarap. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375, 2018. 5   
[2] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020. 7   
[3] Kai Chen, Enze Xie, Zhe Chen, Lanqing Hong, Zhenguo Li, and Dit-Yan Yeung. Integrating geometric control into text-to-image diffusion models for high-quality detection data generation via text prompt. arXiv preprint arXiv:2306.04607, 2023. 4, 6   
[4] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 13   
[5] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. In ICLR, 2024. 1, 3, 4, 5, 8, 15   
[6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 7, 8   
[7] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint:2207.12598, 2022. 7   
[8] Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In ICCV, 2023. 13   
[9] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In ICCV, 2023. 2, 8, 13, 14   
[10] Jumin Lee, Sebin Lee, Changho Jo, Woobin Im, Juhyeong Seon, and Sung-Eui Yoon. Semcity: Semantic scene generation with triplane diffusion. arXiv preprint arXiv:2403.07773, 2024. 9   
[11] Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, et al. Coda: A real-world road corner case dataset for object detection in autonomous driving. In ECCV, 2022. 9   
[12] Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, and Ying-Cong Chen. Lift3d: Synthesize 3d training data by lifting 2d gan to 3d generative radiance field. In CVPR, 2023. 1, 3   
[13] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Driving diffusion: Layout-guided multi-view driving scene video generation with latent diffusion model. arXiv preprint arXiv:2310.07771, 2023. 3   
[14] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023. 2   
[15] Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M Alvarez. Fb-occ: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492, 2023. 7   
[16] Zhiyuan Liu, Leheng Li, Yuning Wang, Haotian Lin, Zhizhe Liu, Lei He, and Jianqiang Wang. Controllable traffic simulation through llm-guided hierarchical chain-of-thought reasoning. arXiv preprint arXiv:2409.15135, 2024. 9   
[17] William Ljungbergh, Adam Tonderski, Joakim Johnander, Holger Caesar, Kalle Åström, Michael Felsberg, and Christoffer Petersson. Neuroncap: Photorealistic closed-loop safety testing for autonomous driving. arXiv preprint arXiv:2404.07762, 2024. 9   
[18] Jiachen Lu, Ze Huang, Jiahui Zhang, Zeyu Yang, and Li Zhang. Wovogen: World volume-aware diffusion for controllable multi-camera driving scene generation. arXiv preprint arXiv:2312.02934, 2023. 3, 9   
[19] Jianbiao Mei, Yu Yang, Mengmeng Wang, Tianxin Huang, Xuemeng Yang, and Yong Liu. Ssc-rs: Elevate lidar semantic scene completion with representation separation and bev fusion. In IROS, 2023. 3   
[20] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2   
[21] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs for dynamic scenes. In CVPR, 2021. 9

[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3, 7   
[23] Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, and Yang Zhao. Roomdreamer: Text-driven 3d indoor scene synthesis with coherent geometry and texture. arXiv preprint arXiv:2305.11337, 2023. 13   
[24] Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. Street-view image generation from a bird's-eye view layout. IEEE RAL, 2024. 1, 3, 8   
[25] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. NeurIPS, 2024. 1, 3, 7   
[26] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, et al. Scene as occupancy. In ICCV, 2023. 3   
[27] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023. 3, 8   
[28] Xiaofeng Wang, Zheng Zhu, Wenbo Xu, Yunpeng Zhang, Yi Wei, Xu Chi, Yun Ye, Dalong Du, Jiwen Lu, and Xingang Wang. Openoccupancy: A large scale benchmark for surrounding semantic occupancy perception. In ICCV, 2023. 1, 3, 7   
[29] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. arXiv preprint arXiv:2311.17918, 2023. 3, 5, 9   
[30] Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. In ICCV, 2023. 3   
[31] Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. arXiv preprint arXiv:2311.16813, 2023. 1, 3, 5, 8   
[32] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 5, 15   
[33] Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, et al. Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. arXiv preprint arXiv:2401.17053, 2024. 9   
[34] Xu Yan, Jiantao Gao, Jie Li, Ruimao Zhang, Zhen Li, Rui Huang, and Shuguang Cui. Sparse single sweep lidar point cloud segmentation via learning contextual shape priors from scene completion. In AAAI, 2021. 3   
[35] Kairui Yang, Enhui Ma, Jibin Peng, Qing Guo, Di Lin, and Kaicheng Yu. Bevcontrol: Accurately controlling street-view elements with multi-perspective consistency via bev sketch layout. arXiv preprint arXiv:2308.01661, 2023. 3, 8   
[36] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 2023. 3   
[37] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023. 9   
[38] Yuanbo Yang, Yifei Yang, Hanlei Guo, Rong Xiong, Yue Wang, and Yiyi Liao. Urbangiraffe: Representing urban scenes as compositional generative neural feature fields. In ICCV, 2023. 3   
[39] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In CVPR, 2023. 9   
[40] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney: Going from anywhere to everywhere. arXiv preprint arXiv:2312.03884, 2023. 13   
[41] Junge Zhang, Qihang Zhang, Li Zhang, Ramana Rao Kompella, Gaowen Liu, and Bolei Zhou. Urban scene diffusion through semantic occupancy map. arXiv preprint arXiv:2403.11697, 2024. 9   
[42] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 3, 5   
[43] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. NeurIPS, 2023. 7   
[44] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 2, 4

[45] Benjin Zhu, Zhengkai Jiang, Xiangxin Zhou, Zeming Li, and Gang Yu. Class-balanced grouping and sampling for point cloud 3d object detection. arXiv preprint arXiv:1908.09492, 2019. 6

# Appendix

In the appendix, we provide the following content:

<table><tr><td>Sec. A: Statement of Geometric Control.</td><td>Sec. E: Results of Video Generation.</td></tr><tr><td>Sec. B: Long-Tailed Scene Evaluation.</td><td>Sec. F: Generalize to New Cameras.</td></tr><tr><td>Sec. C: Ablation of plane number in MPIs.</td><td>Sec. G: Impact of Amount of Augment Data.</td></tr><tr><td>Sec. D: Additional Qualitative Comparison.</td><td>Sec. H: Visualization of Failure Cases.</td></tr></table>

# A Statement of Geometric Control

In our paper, we refer the geometric controllable generation as using a voxel grid in 3D space to control the image generation. Although the voxel is a quantized representation of the 3D world, when the resolution goes larger, it can already faithfully represent the geometry detail of scenes. Currently, we are limited by the precision of ground truth labels. The $0.2m$ occupancy grid is a tensor of $500 \times 500 \times 40$ that covers a space in x-axis spanning $[-50m, 50m]$ , y-axis spanning $[-50m, 50m]$ , z-axis spanning $[-5m, 3m]$ . In the future, we plan to explore a higher resolution of geometric control to refine our generation.

Except for occupancy, several other 3D representations can be expressed by 3D semantic MPI, such as mesh, dense point clouds, and even 3D boxes or HD maps. The underlying mechanism is to cast several slices of multi-plane images at different depths to retrieve geometric information. Thus, our 3D semantic MPI can be regarded as a general 3D conditioning representation to benefit a wide spectrum of practical systems. These encompass but are not limited to 3D generation such as text2room [8], RoomDreamer [23], WonderJourney [40], and LucidDreamer [4], each of which stands to benefit from the rich geometric context provided by our approach.

# B Long-Tailed Scene Evaluation

In this section, we explore to use SytheOcc to create long-tailed scenes for downstream evaluation. This also stands for evaluating our model using several corner cases. Similar to the SytheOcc-Gen experiment in Tab. 1, we generate a synthetic validation set but use prompts control to manipulate weather patterns or the intensity of illumination.

As depicted in Fig. 7. We create a variety of weather conditions including sandstorms, snow, foggy, rainy, day night, and day time. The motivation behind the creation of these scenes lies in their extreme rarity compared to the ordinary scenes we have captured. The generation of such data is of significant value, as it aids in addressing the long-tailed distribution of scenes, thereby enriching the diversity of our dataset. More visualization is provided in Fig. 13 to Fig. 14.

In Tab. 4, we observe that all kinds of extreme weather lead to a degradation in performance. This observation underscores the limitations of the perception model in terms of its generalizability to infrequent weather scenarios. Among them, we find that foggy, rainy, and day night exert the most severe impact, as they contribute to a large reduction in visibility as shown in Fig. 7. To improve the generalizability to handle various weather conditions, future work can leverage our generated data to cover the long-tailed scenes, or use adversarial search to find severe scenes based on our framework.

<table><tr><td>Scenes</td><td>Sandstorm</td><td>Snow</td><td>Foggy</td><td>Rainy</td><td>Day night</td><td>Day time (raw data)</td></tr><tr><td>FB-Occ mIOU</td><td>22.88</td><td>18.25</td><td>10.29</td><td>9.71</td><td>9.95</td><td>25.50</td></tr></table>

Table 4: Experiments of downstream evaluation on long-tailed scenes with extreme weather.

Furthermore, we perform long-tailed scene evaluation in Fig. 8. We display the failure of the downstream model VAD [9] in our synthetic long-tailed scene. In this case, we simulate a foggy environment that the dense fog obscures the majority of the ego view. Our experiment reveals that due to the lack of training images of foggy scenes, VAD erroneously predicts waypoints that would result in a collision with the bus. This experiment elucidates the boundaries and failure cases of the VAD model [9]. It exposes the limitations of the system under certain conditions, thereby providing insights into scenarios where the model's performance may be compromised.

![](images/65264ac66d86cf60fe11419db60a5cf1c7028bf4c1550c87611dbcee572d592c.jpg)

<details>
<summary>natural_image</summary>

Collage of urban street scenes at dusk, including roads, buildings, and pedestrian crossings (no visible text or symbols)
</details>

Figure 7: From top to bottom, we display images of fusion of 3D semantic MPI, synthesized images of sandstorm, snow, foggy, rainy, day night, day time, and ground truth.

![](images/d44789eb14cfff61ff9b5b3d2c1a0d1abc7b0c35d9ce8a45c516951561d11aac.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Scene geometry"] --> B["SyntheOcc"]
    B --> C["Raw image"]
    C --> D["Generated image"]
    D --> E["VAD"]
    E --> F["Predicted waypoints (Static)"]
    F --> G["Predicted waypoints"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fff,stroke:#333
```
</details>

Figure 8: Use SytheOcc to create long-tailed scenes for testing. Top: In the ordinary scene of a bus placed in front of the ego vehicle, the end-to-end planner VAD [9] predicts future waypoints without movement, thus not plotted in the image. Bottom: By harnessing the prompt-level control in our framework, we simulate a scene with the same layout but filled with fog. VAD predicts wrong waypoints that will collide with the bus.

# C Ablation of plane number of MPIs

In our proposed 3D semantic MPIs, the number of planes is a hyperparameter that affects the precision of 3D representation. The plane number can be regarded as the 3D resolution in depth axis. The larger the plane number, the MPI will contain more details. We find that an increase in the number of planes is associated with improved accuracy in downstream tasks. This finding denotes that more condition information leads to better downstream task performance.

![](images/e363a0066d1138500873a2825c31b063edaf377eb947df2d494eab1975d4334f.jpg)

<details>
<summary>text_image</summary>

Fusion of MPIs
MagicDrive
ControlNet
ControlNet+depth
SyntheOcc
GT
</details>

Figure 9: Comparison with baselines.

<table><tr><td>Number of Planes</td><td>96</td><td>128</td><td>256</td></tr><tr><td>FB-Occ mIOU</td><td>23.36</td><td>24.28</td><td>25.50</td></tr></table>

Table 5: Ablation of the number of multi-plane images.

# D Qualitative Comparison with Baselines and SOTA

In Fig. 9, we conduct a qualitative comparison of our method against MagicDrive, ControlNet, and ControlNet+depth. We find that all the methods display a satisfactory image quality, as they build upon the foundation of the stable diffusion model. The generation of MagicDrive fails to synthesize barriers as shown in the bottom row. ControlNet struggles to generate objects with the correct pose solely from only 2D conditions as shown in the second row. ControlNet+depth, a degradation of our method, an enhancement over ControlNet in terms of alignment, nevertheless suffers from a loss of finer detail in scenes with heavy occlusion, as shown in the human of the third row. Our method, in contrast, aims to address these challenges and provide a more nuanced and accurate generation of complex scenes.

# E Extend to Video Generation

As described in the main paper Sec. 3.4, we further extend the cross-view attention to cross-frame attention to perform video generation. Our generation results are Fig. 11, Fig. 12 and Fig. 16. Our implementation is adopted from MagicDrive [5] which is similar to Tune-a-video [32]. The formulation of cross-frame attention is:

$$
\text { Attention } (Q, K, V) = \operatorname{softmax} (\frac {Q K ^ {T}}{\sqrt {d}}) \cdot V, \tag {8}
$$

$$
h _ {o u t} = h _ {i n} + \sum_ {i \in \{f, h \}} \text { Attention } (Q _ {i n}, K _ {i}, V _ {i}), \tag {9}
$$

where f, and h are the camera view of future and history frames. $Q_{in}$ and $h_{in}$ denotes the query and the hidden state of input view. We train our model in a two-stage pipeline. We first train the MPI encoder and cross-view attention in a multi-view image generation setting. Then we train cross-frame attention and freeze other components in a video generation setting.

In practice, we use the keyframe annotation of the nuScenes dataset to train our video model. We start with our pretrained MPI encoder and cross-view attention and only train our cross-frame attention while keeping others frozen. We employ a sequence of 7 frames as a batch, resulting in a batch size of 42 images for the training process.

Given that our primary contribution does not lie in video generation, this experiment serves as a proof of concept, demonstrating the potential of our framework. Future research may extend our methodology to facilitate the generation of longer video sequences, thereby expanding the scope and applicability of our framework.

![](images/392b9895219a93ae30d5d5a85d0aef8855f60624be68830de6bb34a06224c213.jpg)

<details>
<summary>natural_image</summary>

Composite image showing a city street with vehicles and a colorful pixelated overlay of urban buildings (no text or symbols)
</details>

(a) Generation with original intrinsic

![](images/36471656e248e1a1ed2cc99d65ccd3d44eebd8272bc0539c80ef12865e93c8b1.jpg)

<details>
<summary>natural_image</summary>

Collage of urban street scenes with vehicles and buildings, no visible text or symbols
</details>

(b) Generation with intrinsic modification (focal length \* 0.8)

![](images/6cd5a88eb8e49f81e568b6f9909a619404cbda38d75e4c6d19c2f3e8311ad47d.jpg)

<details>
<summary>natural_image</summary>

Collage of urban street scenes including vehicles, buildings, and a large map with colored overlays (no visible text or symbols)
</details>

(c) Generation with intrinsic modification (focal length \* 1.2)   
Figure 10: We demonstrate the generalizability of SytheOcc to new camera intrinsic. We multiply factors to the focal length while keeping the resolution the same. In (b,c), focal length ×0.8 denotes a camera with a larger field of view similar to zoom out, focal length ×1.2 denotes a camera with a smaller field of view similar to zoom in.

# F Generalize to New Cameras

In this section, we investigate the adaptability of our method to a new set of cameras with different intrinsic. Given that our training set has a fixed camera intrinsic and extrinsic, generalizing to novel cameras indicates that our approach possesses robust generalization capabilities. As shown in Fig. 10, benefiting from our local type of condition, SytheOcc generates images that faithfully align with the new intrinsic, proving that SytheOcc do not over-fit certain parameters. Regarding extrinsic parameters, we can cast our MPI at the desirable locations to retrieve geometric information, thus inherently ensuring generalizability without doubt.

# G The Influence of the Amount of Augmented Data

As SytheOcc is capable of generating an infinite number of synthetic data, we investigate the influence of the amount of augmented data on downstream tasks in Tab. 6. We find that when our augmented data is expanded from one-fold to two-fold of the training dataset, the performance of perception model slightly decreases. This may indicate the generated data has an optimal ratio for downstream tasks. Due to limited computational resources, we only experiment with a limited amount of ratio. Future work can conduct more thorough experiments to find a universal theorem.

<table><tr><td>Amount of Augmented Data</td><td>0 (no augmentation)</td><td>1</td><td>2</td></tr><tr><td>FB-Occ mIOU</td><td>39.3</td><td>40.3</td><td>40.1</td></tr></table>

Table 6: Ablation of the amount of augmented data.

![](images/f26b63c641ac5931db7857a7b9b3efcb0d5825edd555211c0b2eb7ea5190de88.jpg)

<details>
<summary>natural_image</summary>

Grid of urban street scenes with cars and buildings, no visible text or signage
</details>

Figure 11: Video generation results. In the temporal progression, the distant buildings maintain a high degree of consistency, and objects retain their identical shapes and textures across different views and frames.

![](images/974ebfc4d3e8b65ce869399c6acdc5368ce83aefef5bfa8a71b058009d21a583.jpg)

<details>
<summary>natural_image</summary>

Grid of urban street scenes with cars and vehicles, no visible text or signage
</details>

Figure 12: Video generation results of large dynamics scenes. The white car comes across different views and frames depicting consistent shapes with only a slight appearance change.

![](images/bf22e1efdff5264b12110c8e5ec5e70e7e09f7133954c9d8776e0b83ecf7ac4d.jpg)

<details>
<summary>natural_image</summary>

Grid of urban street scenes at night, including roads, buildings, and traffic lights (no visible text or symbols)
</details>

Figure 13: From top to bottom, we display images of fusion of 3D semantic MPI, synthesized images of sandstorm, snow, foggy, rainy, day night, day time, and ground truth.

![](images/2b069f570fd0327236d5815938045a2061da2f14ad3a3324a9874f378ebd9b45.jpg)

<details>
<summary>natural_image</summary>

Collage of urban street scenes at night, including modern buildings, highways, and pedestrian traffic (no visible text or signage)
</details>

Figure 14: Weather variation. Same structure with Fig. 13.

![](images/24274fd5e14a96588b265fac38caa41ba251a6fd402333c7128c2c38268a89da.jpg)

<details>
<summary>natural_image</summary>

Grid of urban street scenes with roads, buildings, and greenery (no visible text or symbols)
</details>

Figure 15: Out of distribution generation. We use prompts to control the high-level appearance of images with specific styles. From top to bottom, we display (1) fusion of 3D semantic MPI. (2) Sunny day. (3) Science fiction style. (4) 8-bit pixel art style. (5) Snowfall. (6) Minecraft style. (7) Pokémon style. (8) Diablo style. (9) Ghibli style. (10) Metropolis style. (11) Gotham style. (12) Ground truth.

# H Failure Cases

We display several failure cases of our method. In Fig. 16, we show a crowd scenes. In this scenario, the excessive number of pedestrians presents a challenge to the cross-view attention and cross-frame attention modules. We find our method incapable of discerning individual entities with clarity. Future research can improve the model capacity or enrich high-quality data to mitigate this problem.

![](images/c44cb0eb6656f482be70e2c5ffa146876d6273a34bfb1a3960cfd6bf9fcdf2d3.jpg)

<details>
<summary>natural_image</summary>

Collage of urban street scenes including roads, landmarks, and construction sites (no visible text or symbols)
</details>

Figure 16: Failure case of video generation results. Our cross-frame attention module is challenging to distinguish a crowd of people across different views and frames.