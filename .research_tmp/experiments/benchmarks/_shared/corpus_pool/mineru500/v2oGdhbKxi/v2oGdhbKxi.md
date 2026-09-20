# MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues

Jinrang Jia\*

Baidu Inc.

Beijing, China

jiajinrang@baidu.com

Zhenjia Li\*

Baidu Inc.

Beijing, China

lizhenjia@baidu.com

Yifeng Shi $^{†}$

Baidu Inc.

Beijing, China

shiyifeng@baidu.com

# Abstract

Monocular 3D detection of vehicle and infrastructure sides are two important topics in autonomous driving. Due to diverse sensor installations and focal lengths, researchers are faced with the challenge of constructing algorithms for the two topics based on different prior knowledge. In this paper, by taking into account the diversity of pitch angles and focal lengths, we propose a unified optimization target named normalized depth, which realizes the unification of 3D detection problems for the two sides. Furthermore, to enhance the accuracy of monocular 3D detection, 3D normalized cube depth of obstacle is developed to promote the learning of depth information. We posit that the richness of depth clues is a pivotal factor impacting the detection performance on both the vehicle and infrastructure sides. A richer set of depth clues facilitates the model to learn better spatial knowledge, and the 3D normalized cube depth offers sufficient depth clues. Extensive experiments demonstrate the effectiveness of our approach. Without introducing any extra information, our method, named MonoUNI, achieves state-of-the-art performance on five widely used monocular 3D detection benchmarks, including Rope3D and DAIR-V2X-I for the infrastructure side, KITTI and Waymo for the vehicle side, and nuScenes for the cross-dataset evaluation.

# 1 Introduction

Accurate 3D detection [14, 21, 25] is crucial for autonomous driving. Although LIDAR sensors [8, 18, 47, 51] provide high precision, camera sensors [7, 39, 52] are cost-effective and have a wider range of perception. Typically, autonomous driving systems employ frontal-view cameras mounted on the vehicle for 3D detection. As intelligent transportation continues to advance, there is increasing interest in using infrastructure-side cameras for 3D detection [13, 49, 53].

Due to the different installations and focal lengths between the vehicle and infrastructure-side cameras, researchers usually design algorithms to solve these two problems separately, which adds an additional application limitation. Fig. 1 illustrates the imaging process and the corresponding visual feature under different installations and focal lengths. With respect to the different installations, most vehicle-side cameras are installed on the top of the vehicle with a near-zero pitch angle, leading to a prior assumption that the optical axis is parallel to the ground. Most vehicle-side methods $[4, 19, 27]$ are designed based on this assumption. In contrast, the infrastructure-side cameras, which are mounted on poles, typically have large pitch angles, rendering most of the existing vehicle-side

![](images/aa59d157f2989fdc5fd4e2cfd65cbf07793a32858d0ae4882bec2a2eda9a5282.jpg)

<details>
<summary>text_image</summary>

z₁
</details>

Image captured by camera 1

![](images/a0b740bd77790045237c269b580fbf7f4a15ba43303fe193e4ca5830678721b4.jpg)

<details>
<summary>text_image</summary>

Street photo with visible store signboards and a car driving on a multi-lane road intersection
</details>

Image captured by camera 2

![](images/16dd2b0640a2c48babe7a08f86bd4cfd3eb6b62d66aa16bc4215f6dcf81548df.jpg)

<details>
<summary>text_image</summary>

z₃
</details>

Image captured by camera 3

![](images/9e048596f184534e3258bef95fafe113799afc2ac50b4f6b9d3def9f4cf451b0.jpg)

<details>
<summary>text_image</summary>

Z₄
</details>

Image captured by camera 4

![](images/d1f58cf1b7adbfb57ba6aa607e63ba29ccf74897e54b0b121073c02109fa0b73.jpg)

<details>
<summary>text_image</summary>

Camera 1
θ₁
f₁
Camera 3
θ₃
f₃
Plane₃
θ₃ > θ₁ ≈ θ₂ > θ₄ ≈ 0
f₁ > f₂ ≈ f₃
f₄ ≈ f_veh_uniform
z₁ > z₂ ≈ z₃
Plane₁
Camera 2
θ₂
f₂
Plane₂
Camera 4
θ₄ ≈ 0
f₄
Plane₄
</details>

Figure 1: Schematic diagram of the camera imaging process and visual feature under different focal lengths and pitch angles. We assume that there is a scene where four cameras in different positions capture the same car (the white car highlighted in the four corresponding images). Camera 1,2,3 are infrastructure-side cameras, and camera 4 is the vehicle-side camera. The pitch angles of camera 1 and camera 2 are similar. Camera 2 is closer to the vehicle, and the focal length is relatively small. Due to the substantial variation in focal length between the two cameras, although the distance between the two cameras and the white car is significantly different, the visual features of the vehicle in the two images are nearly identical, which creates ambiguity in the depth estimation. The focal lengths of camera 2 and camera 3 are comparable. However, due to the different pitch angles of the two cameras, the same car at a similar depth appears with different visual features, resulting in increased difficulty for accurate depth regression. Camera 4 is on the vehicle side with a near-zero pitch angle, leading to a prior assumption that the optical axis is parallel to the ground which is not satisfied on the infrastructure side.

methods unsuitable for direct application, and the diversity of pitch angles also further increases the difficulty of object detection. According to the various focal lengths, the type of vehicle-side cameras are relatively uniform, and the focal lengths are thus similar, while the infrastructure-side cameras have a wide range of focal lengths. This results in a new challenge of 1-to-N ambiguity for depth estimation in monocular 3D detection, which has already been an ill-posed problem.

In this work, to address the above problems, we developed the obstacle projection models of the vehicle and infrastructure sides, and found that the former can actually be regarded as a special case of the latter, where the pitch angle is approximately 0 and the focal length is fixed. Consequently, we propose a unified optimization target: normalized depth, which is independent of focal length and pitch angle, so that the prediction of the obstacle depth on the two sides is no longer disturbed by the diversity of focal length and pitch angle. Furthermore, to enhance the performance of 3D detection, we draw inspiration from methods like AutoShape [26] and DID-M3D [37], which introduce new depth clues. For instance, AutoShape uses dense key points on the vehicle surface to establish geometric constraints, while DID-M3D employs depth maps to enable the network to predict object surface depth. They both add spatially correlated depth clues to the model in different ways with auxiliary data. We suggest that incorporating rich depth clues has a greater impact on depth estimation accuracy. Thus, we use the geometric relationship to create a 3D normalized cube depth for the obstacle, allowing the model to predict the normalized depth of the obstacle's 3D bounding box without introducing any extra data. By predicting the bias depth from each point on the 3D bounding box to the obstacle center, we can obtain the final depth conveniently. The approach of predicting obstacle 3D normalized cube depth maximizes depth clues and improves detection performance without additional data.

In summary, incorporating the above techniques, our method MonoUNI first achieves state-of-the-art (SOTA) performance on both vehicle and infrastructure-side monocular 3D detection. The main contributions of this work are as follows: 1) We proposed a unified optimization target, namely normalized depth, to address the differences between vehicle and infrastructure-side 3D detection caused by the diversity of pitch angle and focal length. This optimization target can be considered as

the standard solution for future monocular 3D detection of both sides. 2) We posit that rich depth clues are crucial in 3D detection, and thus propose the use of a 3D normalized cube depth to facilitate the learning of depth information. 3) Without using any additional information, MonoUNI ranks 1st in both the Rope3D [53] and DAIR-V2X-I [55] infrastructure-side benchmarks. Moreover, we apply our method to the vehicle-side benchmarks, such as KITTI [10] and Waymo [42], and also achieve competitive results. Cross-dataset evaluation of the KITTI val model on the nuScenes [3] val set demonstrates the generalizability of our method. The code is available at https://github.com/Traffic-X/MonoUNI.

# 2 Related Work

Vehicle-side Monocular 3D Object Detection. Vehicle-side monocular 3D detection refers to the process of analyzing a single image captured by a camera mounted on a vehicle to predict the 3D locations, dimensions, and orientations of the interest obstacles. These methods can be mainly categorized into two groups based on whether additional data is utilized. The first type of method uses additional data, such as depth maps [9, 28, 38], CAD models [5, 26, 35, 34] or LIDAR [6, 39] to enhance the detection accuracy. Pseudo-LIDAR methods [29, 46] utilize the depth map predicted by the additional network to assist the monocular image to generate a pseudo point cloud and then adopt existing LIDAR-based 3D object detection pipeline. DID-M3D [37] uses a dense depth map to generate visual depth for powerful data augmentation. CaDDN [39] uses LIDAR points to supervise additional monocular network estimates dense depth map and converts the feature to BEV perspective for prediction. AutoShape [26] utilizes CAD models to generate dense key points to alleviate the sparse constraints. NeurOCS [32] attempts to introduce the Neural Radiance Field (NeRF) [31] to solve the problem of lack of supervision information. Mix-Teaching [50] utilizes unlabeled data to achieve semi-supervised learning. Although utilizing additional information allows these methods to achieve improved performance, it inevitably leads to increased labeling and computational costs. Moreover, obtaining such data is challenging in various scenarios, especially when it comes to the infrastructure side.

The second kind of method only uses a single image as input without any extra information. Some methods $[23, 27, 45, 56]$ use geometric projection assumptions to improve the accuracy of 3D detection. GUPNet $[27]$ uses the 2D and 3D heights of the object to construct similar triangles to assist in regression depth. MonoFlex $[56]$ and MonoDDE $[23]$ further extend this similar triangle relationship using the position of key corner points to assist regression depth. PGD $[45]$ constructs geometric relation graphs across predicted objects and uses the graph to facilitate depth estimation. Due to the installed pitch angle of infrastructure-side cameras, these geometric projections cannot be applied to infrastructure-side monocular 3D detection. Other methods $[25, 30, 44]$ directly predict the dimensions, orientations, and locations of obstacles without the aid of geometric projections. However, these methods are designed for the vehicle side, ignoring the pitch angle diversity and the ambiguity introduced by the focal length gap between different cameras on the infrastructure side.

Infrastructure-side Monocular 3D Object Detection. Compared to vehicle-side monocular 3D object detection, which is typically limited to short-range perception, infrastructure-side 3D detection can overcome the problem of frequent occlusion in vehicle-based detection by increasing the sensor installation height, thereby providing long-range perception capabilities. Recently, DAIR-V2X-I [55] and Rope3D [53] are proposed to promote the development of 3D perception in infrastructure-side scenarios. Nonetheless, the challenges mentioned above, including the focal length difference and the variable pitch angle of the cameras, make it arduous to transfer the vehicle-side 3D detection method to the infrastructure-side setting, which results in relatively sluggish progress in infrastructure-side monocular 3D detection. BEVHeight [49] mitigates the issues arising from variations in camera pose parameters by directly predicting object height instead of object depth and achieves competitive performance in Rope3D and DAIR-V2X-I.

# 3 Method

# 3.1 Overview

Monocular 3D object detection extracts features from a single RGB image, predicting the category and 3D bounding box for each object in the image. The 3D bounding box can be further divided

![](images/50a634654bdaff490270fdcb89f1521a4d3f485b5011061b59efd35b3b98dbc2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Vehicle-side"] --> B["2D Detector"]
    C["Infrastructure-side"] --> B
    B --> D["Object feature"]
    D --> E["Bias Depth uncertainty"]
    D --> F["Bias Depth"]
    D --> G["Cube Depth uncertainty"]
    D --> H["Cube Depth"]
    D --> I["Other properties"]
    E --> J["L_bias"]
    F --> K["L_cube"]
    G --> L["Normalized Cube Depth Generation"]
    H --> L
    I --> L
    L --> M["Bias depth"]
    L --> N["Norm cube depth"]
    M --> O["Calculate the bias"]
    N --> P["Normalize"]
    O --> Q["Object label"]
    P --> Q
    Q --> R["Location"]
    Q --> S["Dimension"]
    Q --> T["Orientation"]
    Q --> U["2D box"]
    U --> V["Pitch angle"]
    U --> W["Focal length"]
    V --> X["Camera parameters"]
```
</details>

Figure 2: Overview. The left side of the picture describes the network architecture of MonoUNI, which uses a 2D detector to obtain object features, and then adopts different heads to estimate the cube depth, bias depth, and corresponding uncertainty, as well as other 3D properties. The right side depicts how we use the 3D label to generate the normalized cube depth and bias depth.

into 3D center location $(x, y, z)$ , dimension $(h, w, l)$ and orientation (yaw angle) $\theta$ . Among these properties, dimension, and orientation are easily learned by the network due to their strong correlation with visual features [15], but the 3D location is challenging because depth estimation is ill-posed.

The main idea of MonoUNI is to unify 3D detection targets for both vehicle and infrastructure sides, and further achieve accurate 3D location. The overall framework is depicted in Fig. 2. We use CenterNet [57] as the base model to generate discriminative representation, with DLA34 [54] serving as the backbone for feature extraction from images. Several network heads are established in MonoUNI to predict object properties, such as categorical heatmap, 2D bounding box, 3D offset, dimension, orientation, 3D normalized cube depth, bias depth, and depth uncertainty items [16, 27]. Depth uncertainty is widely used in 3D detection, which can enhance the loss's robustness against noisy inputs. The loss function is roughly similar to DID-M3D [37]. In the inference process, by leveraging the predicted 3D normalized cube depth and bias depth, and utilizing the known pitch angle and focal length information, our method can obtain the final obstacle depth.

# 3.2 Unified Optimization Target

Problem Analysis. In order to construct a unified optimization target for the vehicle and infrastructure sides, we analyze the significant difference in monocular 3D detection between them. First, the camera on the infrastructure side is typically installed at an elevated position with a specific pitch angle (the angle between the optical axis of the camera and the ground) to capture a wider view and detect more potential obstacles, which invalidates the commonly used assumption on the vehicle side that the optical axis of the camera is parallel to the ground. Not only that, due to different scenarios and installation methods, the pitch angle of the installed camera is varied. For instance, in the Rope3D dataset, the pitch angle of the camera ranges from 5 to 20 degrees. Second, to cope with different situations, the internal parameters of different infrastructure-side cameras usually have a relatively large difference. For example, the camera focal length of Rope3D ranges from 2100 to 2800 pixels, while the camera focal lengths of the KITTI dataset are all between 715 pixels and 721 pixels. The huge gap in focal length can lead to an ambiguity that obstacles of similar visual features in two images captured by cameras with different focal lengths can have different depths. To explore the influence of focal length diversity, we conducted experiments on Rope3D using both the popular single-stage and two-stage monocular object detection methods. Since the focal length of most images is centered around 2100 and 2700 pixels, we further split the dataset into train\_2100, train\_2700, val\_2100, and val\_2700. The experimental results are shown in Table 1.

As a consequence of the aforementioned ambiguity, utilizing two train sets for mixed training will decrease the accuracy of both validation sets when compared to training the network solely with images in a single focal length range. In particular, the average precision (AP) of GUPNet experiences a significant reduction of 70% on the val\_2700. This decrease in accuracy can be attributed to the dominance of images with a focal length of around 2100 pixels during the training process.

Normalized Depth. In this subsection, we first build the simple projection models of the vehicle (Fig. 3 (a)) and infrastructure sides (Fig. 3 (b)). Taking the infrastructure-side projection model as an example, O in the figure is the optical center of the camera, the ray OZ denotes the optical axis (z-axis), point C represents the center point of the obstacle, P represents the intersection point

Table 1: Analysis for different focal lengths on Rope3D dataset with new train/val division. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Train_set</td><td colspan="3"> $AP_{3D}(IOU=0.5|R_{40})$ </td></tr><tr><td>val_2100</td><td>val_2700</td><td>val_all</td></tr><tr><td>GUPNet [27]</td><td>train_2100</td><td>13.20</td><td>0.03</td><td>7.42</td></tr><tr><td>GUPNet [27]</td><td>train_2700</td><td>0.17</td><td>21.65</td><td>3.07</td></tr><tr><td>GUPNet [27]</td><td>train_all</td><td>10.82</td><td>5.85</td><td>9.38</td></tr><tr><td>SMOKE [25]</td><td>train_2100</td><td>9.77</td><td>0.13</td><td>6.19</td></tr><tr><td>SMOKE [25]</td><td>train_2700</td><td>0.04</td><td>23.20</td><td>3.64</td></tr><tr><td>SMOKE [25]</td><td>train_all</td><td>6.04</td><td>18.01</td><td>8.48</td></tr></table>

![](images/48176f71a7cb8188ae91d2fea9cf805bbded773bbe8f12725a25c2e98ba2402a.jpg)

<details>
<summary>text_image</summary>

Imaging Plane
θ ≈ 0
f
O
Z
z
δ
h
Y
H = H'
P
y + d = 0
</details>

(a) Vehicle-side projection model

![](images/2c5aa9cad53c41500ab514460573771e2c4dca0b5752859e1ddd708093f01bbb.jpg)

<details>
<summary>text_image</summary>

Imaging Plane
f
O
z
θ
δ
h
Y
Z
C
P P'
H'
αx + βy + γz + d = 0
H
</details>

(b) Infrastructure-side projection model   
Figure 3: Simple projection models of the vehicle and infrastructure-side.

between the vertical line from the center point C to the ground and the ground plane, H denotes the 3D distance of PC, and h is the pixel distance from the center point C to the ground in the imaging plane, $\theta$ is the pitch angle of the camera, z denotes the depth of center point C, f represents the focal length and $\delta$ is the included angle between the line connecting the point P to the optical center O and OZ. Extend a line from the obstacle's center point C along the camera's imaging plane direction, intersecting line OP at point $P'$ . The distance $CP'$ is denoted as $H'$ . According to the parallel relationship, the following can be easily deduced:

$$
\frac {H ^ {\prime}}{h} = \frac {z}{f} \tag {1}
$$

After a simple geometric calculation, the following equation can be derived $^{3}$ :

$$
H ^ {\prime} = H * (\cos \theta - \sin \theta * \tan \delta) \tag {2}
$$

Substitute it into equation (1) and we get:

$$
z = \frac {H}{h} * (\cos \theta - \sin \theta * \tan \delta) * f \tag {3}
$$

Among them, H and h are easy to learn based on visual features, while $\delta$ , $\theta$ , and f are challenging to directly learn from the image. Therefore, learning z directly requires the model to have a strong ability to infer focal length and angle information. However, from the above analysis, it is difficult or even ambiguous. Typically, the focal length f can be obtained as prior knowledge, the pitch angle $\theta$ can be calculated as $\theta = \arctan(\gamma/\beta)$ from the ground equation $\alpha x + \beta y + \gamma z + d = 0$ , and $\delta$ can be calculated through simple calculation $\delta = \arctan((v_{p} - c_{y})/f)$ , where $v_{p}$ is the y-axis pixel coordinates of Point P and $c_{y}$ is the principal point in y-axis. In particular, $v_{p}$ can be simply and roughly replaced by $v_{c}$ which may introduce an average relative error of 2.9% (statistics based on the

![](images/5466b40973ee995a4ce761b77a2d990d852798171b6d5751f49ff210df067b36.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Original Label"] --> B["Corner Points Generate"]
    B --> C["Planes Fitting"]
    C --> D["Depth Filling"]
    D --> E["3D Cube Depth"]
    
    subgraph Original Label
        A1["Location"]
        A2["Orientation"]
        A3["Dimension"]
        A4["2D Box"]
    end
    
    subgraph Corner Points Generate
        B1["6\n2\n3\n7\n1\n4\nX_obj\nY_obj\nZ_obj\nZ_c\nO_c\nX_c\nY_c"]
    end
    
    subgraph Planes Fitting
        C1["6\n2\n3\n7\n1\n8\nPlane_i\n\alpha_i x + \beta_i y + \gamma_i z + d_i = 0\nZ_c\nO_c\nX_c\nY_c"]
    end
    
    subgraph Depth Filling
        D1["6\n2\n3\n7\nPlane_i\nP_j\nZ_c\nO_c\nX_c\nY_c"]
    end
    
    B --> C
    C --> D
    D --> E
```
</details>

Figure 4: The generation process of the 3D cube depth.

Rope3D dataset), or be predicted by the model. Therefore, a simple transformation of the formula (3) yields:

$$
\text { Normalized\_depth } = \frac {z}{(\cos \theta - \sin \theta * \tan \delta) * f} \tag {4}
$$

Formula (4) represents the new unified optimization target: normalized depth, which makes depth prediction independent of pitch angle and focal length, simplifying the task difficulty. When $\theta$ is equal to 0, the normalized depth degenerates into a situation where only the focal length is used for normalization $(\frac{z}{f})$ , that is, the vehicle-side projection model. In fact, simplifying the normalized depth with either the focal length $(\frac{z}{f})$ or pitch angle $(\frac{z}{\cos\theta - \sin\theta*\tan\delta})$ can also partially reduce the learning difficulty and improve detection accuracy, which we will analyze in detail in the ablation experiments in section 4.4. During inference, the model outputs normalized depth, and we multiply it by the denominator in the formula (4) to get the final instance depth.

# 3.3 3D Normalized Cube Depth

To enrich depth clues and further improve detection performance, we utilize original labels to generate 3D cube depth for each obstacle. Based on the analysis in section 3.2, our 3D cube depth need to be normalized and become 3D normalized cube depth finally. The generation process of the 3D cube depth is shown in Fig. 4. Firstly, we use the location, orientation, and dimension information from the label to obtain the positions of the 8 corner points in the 3D bounding box in the camera coordinate system. These points are then marked in a specific sequence (1-8). Secondly, we calculate the plane equation $\alpha_{i}x + \beta_{i}y + \gamma_{i}z + d_{i} = 0$ for each surface of the obstacle. To determine the equation, we select any 3 of the 4 vertices of the surface since a plane can be uniquely defined by 3 non-collinear points. Thirdly, we generate the 3D cube depth for the obstacle and project it onto the imaging plane. Specifically, each pixel $(u,v)$ in the 2D bounding box is traversed to determine whether it belongs to any surface in the 2D image. Then, utilizing the internal parameters of the camera, the intersection point $(x,y,z)$ of the pixel's ray with the corresponding plane can be computed and the depth $z$ is thus achieved. The mathematical expression for this computation is as follows:

$$
\left\{ \begin{array}{l} \alpha_ {i} x + \beta_ {i} y + \gamma_ {i} z + d _ {i} = 0, f o r i = 1, \dots , 6 \\ f _ {x} \frac {x}{z} + c _ {x} = u \\ f _ {y} \frac {\tilde {g}}{z} + c _ {y} = v \end{array} \right. \tag {5}
$$

where $f_{x}$ and $f_{y}$ are the focal length of the camera, and $c_{x}$ and $c_{y}$ are the principal points. i denotes the index of the surface on the 3D bounding box. The depth z can be calculated as:

$$
z = \frac {- d _ {i}}{\frac {\alpha_ {i} (u - c _ {x})}{f _ {x}} + \frac {\beta_ {i} (v - c _ {y})}{f _ {y}} + \gamma_ {i}} \tag {6}
$$

In cases where a pixel belongs to multiple faces in the 2D image, we select the smallest depth value as the final depth. This is because, during the camera imaging process, points with larger depth values are obstructed by ones closer to the camera. The same strategy is also applied when a pixel simultaneously belongs to multiple obstacles. Once the 3D cube depth is obtained, we apply Equation 3 to each depth value in the cube, resulting in a 3D normalized cube depth that can be used for

Table 2: Monocular 3D detection performance of Car category on Rope3D val, DAIR-V2X-I val and KITTI test sets. We highlight the best results in bold and the second ones in underlined. For the extra data, the first column means whether using depth maps as additional input on the infrastructure side and the second column means on the vehicle side. '-' means that no official results or reasonable reproduction results in the new dataset. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Extra Data</td><td colspan="2">Rope3D</td><td colspan="3">DAIR</td><td colspan="3">KITTI</td></tr><tr><td> $AP_{3D}$ </td><td> $R_{score}$ </td><td>Easy</td><td>Mod.</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>M3D-RPN [1]</td><td>Depth | None</td><td>67.17</td><td>73.14</td><td>-</td><td>-</td><td>-</td><td>14.76</td><td>9.71</td><td>7.42</td></tr><tr><td>MonoDLE [30]</td><td>Depth | None</td><td>77.50</td><td>80.84</td><td>-</td><td>-</td><td>-</td><td>7.23</td><td>12.26</td><td>10.29</td></tr><tr><td>MonoFlex [56]</td><td>Depth | None</td><td>59.78</td><td>66.66</td><td>-</td><td>-</td><td>-</td><td>19.94</td><td>13.89</td><td>12.07</td></tr><tr><td>DID-M3D [37]</td><td>None | Depth</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>24.40</td><td>16.29</td><td>13.75</td></tr><tr><td>CMKD [11]</td><td>None | Depth</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>25.09</td><td>16.99</td><td>15.30</td></tr><tr><td>LPCG+MonoFlex [36]</td><td>None | Depth</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>25.56</td><td>17.80</td><td>15.38</td></tr><tr><td>MonoEF [58]</td><td>None | None</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>21.29</td><td>13.87</td><td>11.71</td></tr><tr><td>DEVIANT [20]</td><td>None | None</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>21.88</td><td>14.46</td><td>11.89</td></tr><tr><td>MonoCon [48]</td><td>None | None</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>22.50</td><td>16.46</td><td>13.95</td></tr><tr><td>MonoATT [60]</td><td>None | None</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>24.72</td><td>17.37</td><td>15.00</td></tr><tr><td>MoGDE [59]</td><td>None | None</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>27.07</td><td>17.88</td><td>15.66</td></tr><tr><td>Kinematic3D [2]</td><td>None | None</td><td>50.57</td><td>58.86</td><td>-</td><td>-</td><td>-</td><td>19.07</td><td>12.72</td><td>9.17</td></tr><tr><td>SMOKE [25]</td><td>None | None</td><td>72.13</td><td>76.26</td><td>66.03</td><td>62.24</td><td>60.71</td><td>14.03</td><td>9.76</td><td>7.84</td></tr><tr><td>GUPNet [27]</td><td>None | None</td><td>66.52</td><td>70.14</td><td>62.22</td><td>55.94</td><td>55.90</td><td>22.26</td><td>15.02</td><td>13.12</td></tr><tr><td>Imvoxelnet [40]</td><td>None | None</td><td>-</td><td>-</td><td>44.78</td><td>37.58</td><td>37.55</td><td>17.15</td><td>10.97</td><td>9.15</td></tr><tr><td>BEVFormer [24]</td><td>None | None</td><td>50.62</td><td>58.78</td><td>61.37</td><td>50.73</td><td>50.73</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BEVDepth [22]</td><td>None | None</td><td>69.63</td><td>74.70</td><td>75.50</td><td>63.58</td><td>63.67</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BEVHeight [49]</td><td>None | None</td><td>74.60</td><td>78.72</td><td>77.78</td><td>65.77</td><td>65.85</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MonoUNI(Ours)</td><td>None | None</td><td>92.45</td><td>92.63</td><td>90.92</td><td>87.24</td><td>87.20</td><td>24.75</td><td>16.73</td><td>13.49</td></tr></table>

supervision. In order to get the center point depth, we also supervise the bias depth from each point on the cube depth to the center point depth. The utilization of cube depth can maximize the enrichment of depth clues based on existing labels without additional information such as depth maps, CAD, or LIDAR points. Compared with only supervising the depth of the center point and corner points, the 3D cube depth is a sufficient way to utilize the depth information.

# 4 Experiments

# 4.1 Dataset and Evaluation Metrics

Rope3D. Rope3D [53] is a comprehensive real-world benchmark for infrastructure-side 3D detection, featuring over 50,000 images and more than 1.5 million 3D objects in diverse scenes. The benchmark comprises various settings, such as different cameras with ambiguous mounting positions, diverse camera specifications, as well as different environmental conditions. For the evaluation metrics, we follow the official settings using $AP_{3D|R40}$ and $Rope_{score}$ [53], which is a combined metric of the 3D AP and other similarities, such as Average Ground Center Similarity (ACS). We follow the proposed homologous setting to utilize $70\%$ of the images as training, and the remaining as validating. All images are randomly sampled.

DAIR-V2X-I. DAIR-V2X-I [55] is a subset of DAIR-V2X, which is a large-scale multimodal vehicle-infrastructure collaborative perception dataset. It contains 10,000 infrastructure-side images and 493,000 3D bounding boxes with 10 classes. We follow the official train/val division for evaluation, and use $AP_{3D|R40}$ as the main metric, with an IOU threshold of 0.5.

KITTI. KITTI [10] is a widely used benchmark consisting of 7481 training images and 7518 testing images for vehicle-side monocular 3D detection. The dataset contains three classes: Car, Pedestrian, and Cyclist, each with three difficulty levels: Easy, Moderate, and Hard. Moderate difficulty is considered the official rank level of the KITTI leaderboard. To ensure a fair comparison with previous methods, results are submitted to an official server for evaluation on the test set. $AP_{3D|R40}$ is used as the main metric with IOU thresholds of 0.7 for cars, and 0.5 for Pedestrians and Cyclists respectively.

Table 3: Monocular 3D detection performance of Pedestrian and Cyclist category on KITTI test set. 

<table><tr><td rowspan="3">Method</td><td colspan="6"> $AP_{3D}$ </td></tr><tr><td colspan="3">Pedestrian</td><td colspan="3">Cyclist</td></tr><tr><td>Easy</td><td>Mod.</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>DFR-Net [61]</td><td>6.09</td><td>3.62</td><td>3.39</td><td>5.69</td><td>3.58</td><td>3.10</td></tr><tr><td>CaDDN [39]</td><td>12.87</td><td>8.14</td><td>6.76</td><td>7.00</td><td>3.41</td><td>3.30</td></tr><tr><td>MonoCon [48]</td><td>13.10</td><td>8.41</td><td>6.94</td><td>2.80</td><td>1.92</td><td>1.55</td></tr><tr><td>GUPNet [27]</td><td>14.95</td><td>9.76</td><td>8.41</td><td>5.58</td><td>3.21</td><td>2.66</td></tr><tr><td>MonoDTR [12]</td><td>15.33</td><td>10.18</td><td>8.61</td><td>5.05</td><td>3.27</td><td>3.19</td></tr><tr><td>MonoUNI</td><td>15.78</td><td>10.34</td><td>8.74</td><td>7.34</td><td>4.28</td><td>3.78</td></tr></table>

Table 4: Monocular 3D detection performance of Big Vehicle category on Rope3D val set. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Big Vehicle</td></tr><tr><td> $AP_{3D}$ </td><td> $R_{score}$ </td></tr><tr><td>M3D-RPN [1]</td><td>32.19</td><td>40.52</td></tr><tr><td>MonoDLE [30]</td><td>44.71</td><td>52.83</td></tr><tr><td>SMOKE [25]</td><td>52.04</td><td>59.23</td></tr><tr><td>GUPNet [27]</td><td>45.27</td><td>52.19</td></tr><tr><td>BEVFormer [24]</td><td>34.58</td><td>45.16</td></tr><tr><td>BEVDepth [22]</td><td>45.02</td><td>54.64</td></tr><tr><td>BEVHeight [49]</td><td>48.93</td><td>57.70</td></tr><tr><td>MonoUNI</td><td>76.30</td><td>79.20</td></tr></table>

Table 5: Ablation Study on different components of our overall framework on Rope3D and KITTI val set for Car category. 

<table><tr><td rowspan="2">Experiments</td><td colspan="2">Normlized Depth</td><td rowspan="2">Cube</td><td colspan="2">Rope3D</td><td colspan="3">KITTI</td></tr><tr><td>Focal</td><td>Pitch</td><td> $AP_{3D}$ </td><td> $R_{score}$ </td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>(a)</td><td></td><td></td><td></td><td>79.37</td><td>81.38</td><td>21.71</td><td>14.93</td><td>12.10</td></tr><tr><td>(b)</td><td>√</td><td></td><td></td><td>83.42</td><td>84.91</td><td>22.35</td><td>15.29</td><td>12.34</td></tr><tr><td>(c)</td><td></td><td>√</td><td></td><td>81.55</td><td>83.53</td><td>21.68</td><td>14.85</td><td>12.12</td></tr><tr><td>(d)</td><td>√</td><td>√</td><td></td><td>86.69</td><td>87.99</td><td>22.19</td><td>14.94</td><td>12.45</td></tr><tr><td>(e)</td><td></td><td></td><td>√</td><td>87.07</td><td>88.16</td><td>24.66</td><td>17.07</td><td>14.06</td></tr><tr><td>(f)</td><td>√</td><td>√</td><td>√</td><td>92.45</td><td>92.63</td><td>24.51</td><td>17.18</td><td>14.01</td></tr></table>

Waymo. Waymo [42] assesses objects using a dual-tiered categorization: Level 1 and Level 2. The assessment is performed across three distance intervals: [0, 30), [30, 50), and $[50, \infty)$ meters. Waymo employs the $APH_{3D}$ percentage metric, which integrates heading data into the $AP_{3D}$ , as a reference benchmark for evaluation.

nuScenes. nuScenes [3] comprises 28,130 training and 6,019 validation images captured from the front camera. We use validation split for cross-dataset evaluation.

# 4.2 Implementation Details

Our proposed MonoUNI is trained on 4 Tesla V100 GPUs with a batch size of 16 for 150 epochs. We use Adam [17] as our optimizer with an initial learning rate $1.25 \times e - 3$ . Images are all resized to the same size of $960 \times 512$ for infrastructure side and $1280 \times 384$ for vehicle side. Following [37], the ROI-Align size $d \times d$ is set to $7 \times 7$ . Inspired by [33], we adopt the multi-bin strategy for heading angle and depth prediction in our baseline. Random crop and expand along principal points are used to achieve more physical data augmentation.

# 4.3 Main Results

Results of Car Category. Since there was no previous method designed for supporting both the vehicle and infrastructure sides at the same time, the results of some methods were reproduced by us or other published papers (BEVHeight [49], Rope3D [53] and DAIR-V2X [55]). As shown in Table 2, our proposed MonoUNI achieves superior performance than previous methods on infrastructure-side benchmarks, even those with extra data. Specifically, compared with BEVHeight which is the recent top1-ranked image-only method for infrastructure side, MonoUNI gains significant improvement of $17.85\% / 13.91\%$ in $AP_{3D}$ and $R_{score}$ on Rope3D benchmark. According to DAIR-V2X, our method achieves over $21\%$ improvement compared with BEVHeight on Moderate difficulty. For vehicle-side benchmark, although our method did not rank first, it reached a competitive result on Easy difficulty. On the Moderate and Hard difficulties, MonoUNI is slightly inferior to methods such as MonoATT [60] and MoGDE [59], probably because the gain of the 3D cube depth is weakened in the case of far-distance or severe truncation.

Results of Other Categories. In Table 3, we present the results of pedestrians and cyclists on the test set of KITTI. MonoUNI outperforms all methods by a large margin. This may be because the

Table 6: Analysis of MonoUNI for different focal lengths on Rope3D with new train/val division. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Train</td><td colspan="3"> $AP_{3D}(IOU = 0.5|R_{40})$ </td></tr><tr><td>val_2100</td><td>val_2700</td><td>val_all</td></tr><tr><td>MonoUNI</td><td>2100</td><td>25.78</td><td>21.30</td><td>24.47</td></tr><tr><td>MonoUNI</td><td>2700</td><td>5.77</td><td>23.42</td><td>9.25</td></tr><tr><td>MonoUNI</td><td>all</td><td>26.63</td><td>38.10</td><td>28.91</td></tr></table>

Table 7: Cross-dataset evaluation of the KITTI val model on KITTI val and nuScenes val cars with depth MAE. 

<table><tr><td rowspan="2">Method</td><td colspan="4">KITTI Val</td><td colspan="4">nuScenes frontal Val</td></tr><tr><td>0-20</td><td>20-40</td><td>40- $\infty$ </td><td>All</td><td>0-20</td><td>20-40</td><td>40- $\infty$ </td><td>All</td></tr><tr><td>M3D-RPN [1]</td><td>0.56</td><td>1.33</td><td>2.73</td><td>1.26</td><td>0.94</td><td>3.06</td><td>10.36</td><td>2.67</td></tr><tr><td>MonoRCNN [41]</td><td>0.46</td><td>1.27</td><td>2.59</td><td>1.14</td><td>0.94</td><td>2.84</td><td>8.65</td><td>2.39</td></tr><tr><td>GUPNet [27]</td><td>0.45</td><td>1.10</td><td>1.85</td><td>0.89</td><td>0.82</td><td>1.70</td><td>6.20</td><td>1.45</td></tr><tr><td>DEVIANT [20]</td><td>0.40</td><td>1.09</td><td>1.80</td><td>0.87</td><td>0.76</td><td>1.60</td><td>4.50</td><td>1.26</td></tr><tr><td>MonoUNI</td><td>0.38</td><td>0.92</td><td>1.79</td><td>0.865</td><td>0.72</td><td>1.79</td><td>4.98</td><td>1.43</td></tr></table>

Table 8: Monocular 3D detection performance of Vehicle category on Waymo val set. 

<table><tr><td rowspan="2"> $IOU_{3D}$ </td><td rowspan="2">Difficulty</td><td rowspan="2">Method</td><td rowspan="2">Extra</td><td colspan="4"> $AP_{3D}$ </td><td colspan="4"> $APH_{3D}$ </td></tr><tr><td>All</td><td>0-30</td><td>30-50</td><td>50- $\infty$ </td><td>All</td><td>0-30</td><td>30-50</td><td>50- $\infty$ </td></tr><tr><td rowspan="7">0.7</td><td rowspan="7">Level_1</td><td>CaDDN [39]</td><td>LIDAR</td><td>5.03</td><td>15.54</td><td>1.47</td><td>0.10</td><td>4.99</td><td>14.43</td><td>1.45</td><td>0.10</td></tr><tr><td>PatchNet [28] in [43]</td><td>Depth</td><td>0.39</td><td>1.67</td><td>0.13</td><td>0.03</td><td>0.39</td><td>1.63</td><td>0.12</td><td>0.03</td></tr><tr><td>PCT [43]</td><td>Depth</td><td>0.89</td><td>3.18</td><td>0.27</td><td>0.07</td><td>0.88</td><td>3.15</td><td>0.27</td><td>0.07</td></tr><tr><td>M3D-RPN [1] in [39]</td><td>None</td><td>0.35</td><td>1.12</td><td>0.18</td><td>0.02</td><td>0.34</td><td>1.10</td><td>0.18</td><td>0.02</td></tr><tr><td>GUPNet [27] in [20]</td><td>None</td><td>2.28</td><td>6.15</td><td>0.81</td><td>0.03</td><td>2.27</td><td>6.11</td><td>0.80</td><td>0.03</td></tr><tr><td>DEVIANT [20]</td><td>None</td><td>2.69</td><td>6.95</td><td>0.99</td><td>0.02</td><td>2.67</td><td>6.90</td><td>0.98</td><td>0.02</td></tr><tr><td>MonoUNI (Ours)</td><td>None</td><td>3.20</td><td>8.61</td><td>0.87</td><td>0.13</td><td>3.16</td><td>8.50</td><td>0.86</td><td>0.12</td></tr><tr><td rowspan="7">0.7</td><td rowspan="7">Level_2</td><td>CaDDN [39]</td><td>LIDAR</td><td>4.49</td><td>14.50</td><td>1.42</td><td>0.09</td><td>4.45</td><td>14.38</td><td>1.41</td><td>0.09</td></tr><tr><td>PatchNet [28] in [43]</td><td>Depth</td><td>0.38</td><td>1.67</td><td>0.13</td><td>0.03</td><td>0.36</td><td>1.63</td><td>0.11</td><td>0.03</td></tr><tr><td>PCT [43]</td><td>Depth</td><td>0.66</td><td>3.18</td><td>0.27</td><td>0.07</td><td>0.66</td><td>3.15</td><td>0.26</td><td>0.07</td></tr><tr><td>M3D-RPN [1] in [39]</td><td>None</td><td>0.35</td><td>1.12</td><td>0.18</td><td>0.02</td><td>0.33</td><td>1.10</td><td>0.17</td><td>0.02</td></tr><tr><td>GUPNet [27] in [20]</td><td>None</td><td>2.14</td><td>6.13</td><td>0.78</td><td>0.02</td><td>2.12</td><td>6.08</td><td>0.77</td><td>0.02</td></tr><tr><td>DEVIANT [20]</td><td>None</td><td>2.52</td><td>6.93</td><td>0.95</td><td>0.02</td><td>2.50</td><td>6.87</td><td>0.94</td><td>0.02</td></tr><tr><td>MonoUNI (Ours)</td><td>None</td><td>3.04</td><td>8.59</td><td>0.85</td><td>0.12</td><td>3.00</td><td>8.48</td><td>0.84</td><td>0.12</td></tr><tr><td rowspan="7">0.5</td><td rowspan="7">Level_1</td><td>CaDDN [39]</td><td>LIDAR</td><td>17.54</td><td>45.00</td><td>9.24</td><td>0.64</td><td>17.31</td><td>44.46</td><td>9.11</td><td>0.62</td></tr><tr><td>PatchNet [28] in [43]</td><td>Depth</td><td>2.92</td><td>10.03</td><td>1.09</td><td>0.23</td><td>2.74</td><td>9.75</td><td>0.96</td><td>0.18</td></tr><tr><td>PCT [43]</td><td>Depth</td><td>4.20</td><td>14.70</td><td>1.78</td><td>0.39</td><td>4.15</td><td>14.54</td><td>1.75</td><td>0.39</td></tr><tr><td>M3D-RPN [1] in [39]</td><td>None</td><td>3.79</td><td>11.14</td><td>2.16</td><td>0.26</td><td>3.63</td><td>10.70</td><td>2.09</td><td>0.21</td></tr><tr><td>GUPNet [27] in [20]</td><td>None</td><td>10.02</td><td>24.78</td><td>4.84</td><td>0.22</td><td>9.94</td><td>24.59</td><td>4.78</td><td>0.22</td></tr><tr><td>DEVIANT [20]</td><td>None</td><td>10.98</td><td>26.85</td><td>5.13</td><td>0.18</td><td>10.89</td><td>26.64</td><td>5.08</td><td>0.18</td></tr><tr><td>MonoUNI (Ours)</td><td>None</td><td>10.98</td><td>26.63</td><td>4.04</td><td>0.57</td><td>10.73</td><td>26.30</td><td>3.98</td><td>0.55</td></tr><tr><td rowspan="7">0.5</td><td rowspan="7">Level_2</td><td>CaDDN [39]</td><td>LIDAR</td><td>16.51</td><td>44.87</td><td>8.99</td><td>0.58</td><td>16.28</td><td>44.33</td><td>8.86</td><td>0.55</td></tr><tr><td>PatchNet [28] in [43]</td><td>Depth</td><td>2.42</td><td>10.01</td><td>1.07</td><td>0.22</td><td>2.28</td><td>9.73</td><td>0.97</td><td>0.16</td></tr><tr><td>PCT [43]</td><td>Depth</td><td>4.03</td><td>14.67</td><td>1.74</td><td>0.36</td><td>4.15</td><td>14.51</td><td>1.71</td><td>0.35</td></tr><tr><td>M3D-RPN [1] in [39]</td><td>None</td><td>3.61</td><td>11.12</td><td>2.12</td><td>0.24</td><td>3.46</td><td>10.67</td><td>2.04</td><td>0.20</td></tr><tr><td>GUPNet [27] in [20]</td><td>None</td><td>9.39</td><td>24.69</td><td>4.67</td><td>0.19</td><td>9.31</td><td>24.50</td><td>4.62</td><td>0.19</td></tr><tr><td>DEVIANT [20]</td><td>None</td><td>10.29</td><td>26.75</td><td>4.95</td><td>0.16</td><td>10.20</td><td>26.54</td><td>4.90</td><td>0.16</td></tr><tr><td>MonoUNI (Ours)</td><td>None</td><td>10.38</td><td>26.57</td><td>3.95</td><td>0.53</td><td>10.24</td><td>26.24</td><td>3.89</td><td>0.51</td></tr></table>

introduction of depth clues promotes pedestrians and cyclists better learn spatial features. Table 4 shows the results of Big Vehicle on the Rope3D benchmark.

# 4.4 Ablation Study

Effectiveness of Different Components in MonoUNI on Rope3D and KITTI val set for Car category. We evaluate the effectiveness of the normalized depth and the 3D cube depth through ablations on two datasets. As shown in Table 5, the normalized depth, particularly the focal length, has a significant improvement on the infrastructure side, but only a minor effect on the vehicle side. This is due to the fact that the focal length of the camera is similar across the KITTI data, and the pitch angle is close to 0. On the other hand, the cube depth improves results for both the vehicle and infrastructure sides, providing evidence of its efficacy.

Analysis for different focal lengths. Table 6 presents the performance of MonoUNI at different focal lengths. Compared with the results in Table 1, our method has obvious benefits in solving the problem of focal length diversity. In the case of using all training data, our method can improve 19.53% and 20.43% AP respectively compared with GUPNet and SMOKE.

![](images/1e9ae2de23693d810e36e7bc2d27809235f78b18daa79844d2248d7fb674ec88.jpg)

<details>
<summary>text_image</summary>

Rope3D
KITTI
</details>

Figure 5: Qualitative visualization on the Rope3D and KITTI val set. The 3D green boxes are produced by MonoUNI and the red boxes are the ground truths.

Cross-dataset evaluation. Tabel 7 shows the result of our KITTI val model on the KITTI val and nuScenes [3] frontal val images, using mean absolute error (MAE) of the depth [41]. MonoUNI is better than GUPNet [27] and achieves similar competitive performance to DEVIANT [20]. This is because DEVIANT is equivariant to the depth translations and is more robust to data distribution changes.

Performance of cube depth on large dataset Waymo. In order to fully verify the effectiveness of cube depth, we verified MonoUNI on Waymo, which is a large-scale vehicle-side 3D detection dataset, as shown in Table 8.

# 4.5 Qualitative Results

We visualize the detection results of the MonoUNI on the both vehicle and infrastructure sides in Fig. 5. As can be seen, the MonoUNI can accurately estimate the 3D position of objects, even those not labeled by the annotators due to severe occlusion in the first row of KITTI. However, for far-distance and severely truncated obstacles, our model suffers from missed detections.

# 5 Conclusion

In this paper, we propose a new optimization target named normalized depth to unify monocular 3D detection for both vehicle and infrastructure sides, addressing the problem of focal length and pitch angle diversity. Furthermore, we introduce 3D Cube Depth as an additional supervision clue to improve the 3D detection performance. Experiments on five datasets (Rope3D, DAIR-V2X-I, KITTI, Waymo and nuScenes) fully demonstrate the effectiveness of our method.

# 6 Limitations and Future Work

Although the normalized depth unifies the optimization target of the vehicle and infrastructure sides, separate training for the two sides is still necessary instead of direct hybrid training. In the future, we aim to enable one model file to support 3D detection for both sides which would hold significant value for industrial applications, such as the domain gap problem in the vehicle-infrastructure collaborative perception and fusion. In addition, global depth clues such as depth similarity (i.e., contrastive learning) or the global topological relationship can be further developed.

# References

[1] Garrick Brazil and Xiaoming Liu. M3d-rpn: Monocular 3d region proposal network for object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.   
[2] Garrick Brazil, Gerard Pons-Moll, Xiaoming Liu, and Bernt Schiele. Kinematic 3d object detection in monocular video. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIII 16, pages 135–152, 2020.   
[3] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.   
[4] Yingjie Cai, Buyu Li, Zeyu Jiao, Hongsheng Li, Xingyu Zeng, and Xiaogang Wang. Monocular 3d object detection with decoupled structured polygon estimation and height-guided depth estimation, 2021.   
[5] Florian Chabot, Mohamed Chaouch, Jaonary Rabarisoa, Celine Teuliere, and Thierry Chateau. Deep manta: A coarse-to-fine many-task network for joint 2d and 3d vehicle analysis from monocular image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.   
[6] Hansheng Chen, Yuyao Huang, Wei Tian, Zhong Gao, and Lu Xiong. Monorun: Monocular 3d object detection by reconstruction and uncertainty propagation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10379–10388, June 2021.   
[7] Yongjian Chen, Lei Tai, Kai Sun, and Mingyang Li. Monopair: Monocular 3d object detection using pairwise spatial relationships. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.   
[8] Ziming Chen, Yifeng Shi, and Jinrang Jia. Transiff: An instance-level feature fusion framework for vehicle-infrastructure cooperative 3d detection with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 18205–18214, October 2023.   
[9] Mingyu Ding, Yuqi Huo, Hongwei Yi, Zhe Wang, Jianping Shi, Zhiwu Lu, and Ping Luo. Learning depth-guided convolutions for monocular 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2020.   
[10] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354–3361. IEEE, 2012.   
[11] Yu Hong, Hang Dai, and Yong Ding. Cross-modality knowledge distillation network for monocular 3d object detection. In ECCV, Lecture Notes in Computer Science. Springer, 2022.   
[12] Kuan-Chih Huang, Tsung-Han Wu, Hung-Ting Su, and Winston H. Hsu. Monodtr: Monocular 3d object detection with depth-aware transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4012–4021, June 2022.   
[13] Jinrang Jia, Yifeng Shi, Yuli Qu, Rui Wang, Xing Xu, and Hai Zhang. Competition for roadside camera monocular 3D object detection. National Science Review, 05 2023. nwad121.   
[14] Bo Ju, Wei Yang, Jinrang Jia, Xiaoqing Ye, Qu Chen, Xiao Tan, Hao Sun, Yifeng Shi, and Errui Ding. Danet: Dimension apart network for radar object detection. In Proceedings of the 2021 International Conference on Multimedia Retrieval, ICMR '21, page 533–539, New York, NY, USA, 2021. Association for Computing Machinery.   
[15] Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and Li Fei-Fei. Large-scale video classification with convolutional neural networks. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 1725–1732, 2014.   
[16] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems, 30, 2017.   
[17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
[18] Xianghao Kong, Wentao Jiang, Jinrang Jia, Yifeng Shi, Runsheng Xu, and Si Liu. Dusa: Decoupled unsupervised sim2real adaptation for vehicle-to-everything collaborative perception. In Proceedings of the 31st ACM International Conference on Multimedia, October 2023.   
[19] Jason Ku, Alex D. Pon, and Steven L. Waslander. Monocular 3d object detection leveraging accurate proposals and shape reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[20] Abhinav Kumar, Garrick Brazil, Enrique Corona, Armin Parchami, and Xiaoming Liu. Deviant: Depth equivariant network for monocular 3d object detection. In European Conference on Computer Vision, pages 664–683. Springer, 2022.   
[21] Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[22] Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, and Zeming Li. Bevdepth: Acquisition of reliable depth for multi-view 3d object detection. arXiv preprint arXiv:2206.10092, 2022.   
[23] Zhuoling Li, Zhan Qu, Yang Zhou, Jianzhuang Liu, Haoqian Wang, and Lihui Jiang. Diversity matters: Fully exploiting depth clues for reliable monocular 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2791–2800, June 2022.

[24] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. Bevformer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IX, pages 1–18, 2022.   
[25] Zechen Liu, Zizhang Wu, and Roland Toth. Smoke: Single-stage monocular 3d object detection via keypoint estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2020.   
[26] Zongdai Liu, Dingfu Zhou, Feixiang Lu, Jin Fang, and Liangjun Zhang. Autoshape: Real-time shape-aware monocular 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15641–15650, October 2021.   
[27] Yan Lu, Xinzhu Ma, Lei Yang, Tianzhu Zhang, Yating Liu, Qi Chu, Junjie Yan, and Wanli Ouyang. Geometry uncertainty projection network for monocular 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3111–3121, October 2021.   
[28] Xinzhu Ma, Shinan Liu, Zhiyi Xia, Hongwen Zhang, Xingyu Zeng, and Wanli Ouyang. Rethinking pseudo-lidar representation. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.   
[29] Xinzhu Ma, Zhihui Wang, Haojie Li, Pengbo Zhang, Wanli Ouyang, and Xin Fan. Accurate monocular 3d object detection via color-embedded 3d reconstruction for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.   
[30] Xinzhu Ma, Yinmin Zhang, Dan Xu, Dongzhan Zhou, Shuai Yi, Haojie Li, and Wanli Ouyang. Delving into localization errors for monocular 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4721–4730, June 2021.   
[31] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.   
[32] Zhixiang Min, Bingbing Zhuang, Samuel Schulter, Buyu Liu, Enrique Dunn, and Manmohan Chandraker. Neurocs: Neural nocs supervision for monocular 3d object localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21404–21414, June 2023.   
[33] Arsalan Mousavian, Dragomir Anguelov, John Flynn, and Jana Kosecka. 3d bounding box estimation using deep learning and geometry. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.   
[34] J. Krishna Murthy, G.V. Sai Krishna, Falak Chhaya, and K. Madhava Krishna. Reconstructing vehicles from a single image: Shape priors for road scene understanding. In 2017 IEEE International Conference on Robotics and Automation (ICRA), pages 724–731, 2017.   
[35] Dennis Park, Rares Ambrus, Vitor Guizilini, Jie Li, and Adrien Gaidon. Is pseudo-lidar needed for monocular 3d object detection? In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3142–3152, October 2021.   
[36] Liang Peng, Fei Liu, Zhengxu Yu, Senbo Yan, Dan Deng, Zheng Yang, Haifeng Liu, and Deng Cai. Lidar point cloud guided monocular 3d object detection. In European Conference on Computer Vision, 2022.   
[37] Liang Peng, Xiaopei Wu, Zheng Yang, Haifeng Liu, and Deng Cai. Did-m3d: Decoupling instance depth for monocular 3d object detection. In European Conference on Computer Vision, 2022.   
[38] Zengyi Qin, Jinglu Wang, and Yan Lu. Monogrnet: A geometric reasoning network for monocular 3d object localization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 8851–8858, 2019.   
[39] Cody Reading, Ali Harakeh, Julia Chae, and Steven L. Waslander. Categorical depth distribution network for monocular 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8555–8564, June 2021.   
[40] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Imvoxelnet: Image to voxels projection for monocular and multi-view general-purpose 3d object detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2397–2406, January 2022.   
[41] Xuepeng Shi, Qi Ye, Xiaozhi Chen, Chuangrong Chen, Zhixiang Chen, and Tae-Kyun Kim. Geometry-based distance decomposition for monocular 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15172–15181, 2021.   
[42] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020.   
[43] Li Wang, Li Zhang, Yi Zhu, Zhi Zhang, Tong He, Mu Li, and Xiangyang Xue. Progressive coordinate transforms for monocular 3d object detection. Advances in Neural Information Processing Systems, 34:13364–13377, 2021.   
[44] Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin. Fcos3d: Fully convolutional one-stage monocular 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 913–922, October 2021.   
[45] Tai Wang, Xinge ZHU, Jiangmiao Pang, and Dahua Lin. Probabilistic and geometric depth: Detecting objects in perspective. In Aleksandra Faust, David Hsu, and Gerhard Neumann, editors, Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pages 1475–1485. PMLR, 08–11 Nov 2022.

[46] Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell, and Kilian Q. Weinberger. Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[47] Xiaopei Wu, Liang Peng, Honghui Yang, Liang Xie, Chenxi Huang, Chengqi Deng, Haifeng Liu, and Deng Cai. Sparse fuse dense: Towards high quality 3d detection with depth completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5418–5427, June 2022.   
[48] Tianfu Wu Xianpeng Liu, Nan Xue. Learning auxiliary monocular contexts helps monocular 3d object detection. In 36th AAAI Conference on Artificial Intelligence (AAAI), February 2022.   
[49] Lei Yang, Kaicheng Yu, Tao Tang, Jun Li, Kun Yuan, Li Wang, Xinyu Zhang, and Peng Chen. Bevheight: A robust framework for vision-based roadside 3d object detection. In IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), Mar. 2023.   
[50] Lei Yang, Xinyu Zhang, Li Wang, Minghan Zhu, Chuan-Fang Zhang, and Jun Li. Mix-teaching: A simple, unified and effective semi-supervised learning framework for monocular 3d object detection. ArXiv, abs/2207.04448, 2022.   
[51] Zetong Yang, Yanan Sun, Shu Liu, Xiaoyong Shen, and Jiaya Jia. Std: Sparse-to-dense 3d object detector for point cloud. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.   
[52] Xiaoqing Ye, Liang Du, Yifeng Shi, Yingying Li, Xiao Tan, Jianfeng Feng, Errui Ding, and Shilei Wen. Monocular 3d object detection via feature domain adaptation. In European Conference on Computer Vision, pages 17–34. Springer, 2020.   
[53] Xiaoqing Ye, Mao Shu, Hanyu Li, Yifeng Shi, Yingying Li, Guangjie Wang, Xiao Tan, and Errui Ding. Rope3d: The roadside perception dataset for autonomous driving and monocular 3d object detection task. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21341–21350, June 2022.   
[54] Fisher Yu, Dequan Wang, Evan Shelhamer, and Trevor Darrell. Deep layer aggregation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.   
[55] Haibao Yu, Yizhen Luo, Mao Shu, Yiyi Huo, Zebang Yang, Yifeng Shi, Zhenglong Guo, Hanyu Li, Xing Hu, Jirui Yuan, and Zaiqing Nie. Dair-v2x: A large-scale dataset for vehicle-infrastructure cooperative 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21361–21370, June 2022.   
[56] Yunpeng Zhang, Jiwen Lu, and Jie Zhou. Objects are different: Flexible monocular 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3289–3298, June 2021.   
[57] Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as points. In arXiv preprint arXiv:1904.07850, 2019.   
[58] Yunsong Zhou, Yuan He, Hongzi Zhu, Cheng Wang, Hongyang Li, and Qinhong Jiang. Monocular 3d object detection: An extrinsic parameter free approach. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7556–7566, 2021.   
[59] Yunsong Zhou, Quan Liu, Hongzi Zhu, Yunzhe Li, Shan Chang, and Minyi Guo. Mogde: Boosting mobile monocular 3d object detection with ground depth estimation, 2023.   
[60] Yunsong Zhou, Hongzi Zhu, Quan Liu, Shan Chang, and Minyi Guo. Monoatt: Online monocular 3d object detection with adaptive token transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17493–17503, June 2023.   
[61] Zhikang Zou, Xiaoqing Ye, Liang Du, Xianhui Cheng, Xiao Tan, Li Zhang, Jianfeng Feng, Xiangyang Xue, and Errui Ding. The devil is in the task: Exploiting reciprocal appearance-localization features for monocular 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2713–2722, October 2021.