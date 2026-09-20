# Skinned Motion Retargeting with Dense Geometric Interaction Perception

Zijie Ye $^{1,2,*}$ , Jia-Wei Liu $^{3}$ , Jia Jia $^{1,2,\dagger}$ , Shikun Sun $^{1,2}$ , Mike Zheng Shou $^{3}$

$^{1}$ Department of Computer Science and Technology, BNRist, Tsinghua University

$^{2}$ Key Laboratory of Pervasive Computing, Ministry of Education

$^{3}$ Show Lab, National University of Singapore

# Abstract

Capturing and maintaining geometric interactions among different body parts is crucial for successful motion retargeting in skinned characters. Existing approaches often overlook body geometries or add a geometry correction stage after skeletal motion retargeting. This results in conflicts between skeleton interaction and geometry correction, leading to issues such as jittery, interpenetration, and contact mismatches. To address these challenges, we introduce a new retargeting framework, MeshRet, which directly models the dense geometric interactions in motion retargeting. Initially, we establish dense mesh correspondences between characters using semantically consistent sensors (SCS), effective across diverse mesh topologies. Subsequently, we develop a novel spatio-temporal representation called the dense mesh interaction (DMI) field. This field, a collection of interacting SCS feature vectors, skillfully captures both contact and non-contact interactions between body geometries. By aligning the DMI field during retargeting, MeshRet not only preserves motion semantics but also prevents self-interpenetration and ensures contact preservation. Extensive experiments on the public Mixamo dataset and our newly-collected ScanRet dataset demonstrate that MeshRet achieves state-of-the-art performance. Code available at https://github.com/abcyzj/MeshRet.

# 1 Introduction

Skinned character animation is prevalent in virtual reality [16], game development [21], and various other fields. However, animating these characters often presents significant challenges due to differences in body proportions between the motion source and the target character, leading to issues such as loss of motion semantics, mesh interpenetration, and contact mismatches. Consequently, motion retargeting is essential to adjust for these discrepancies in body proportions. This process is crucial for maintaining the integrity of the source motion's characteristics in the animation of the target character.

Motion retargeting presents challenges due to the complex interactions among character limbs and the wide range of body geometries. Accurately preserving these interactions is crucial, as incorrect interactions can result in mesh interpenetration and contact mismatches. Prior research has typically addressed these interactions from two perspectives: skeleton interactions and geometry corrections. Early methods $[1, 29, 15]$ employ cycle-consistency to implicitly align skeleton interaction semantics, yet they do not address the complexities of geometric interactions between different body parts. Villegas et al. $[28]$ introduced mesh self-contact modeling; however, their approach does not extend

to non-contact interactions. More recently, Zhang et al. [32] implemented a two-stage pipeline that first aligns skeleton interaction semantics and then corrects geometric artifacts. Nonetheless, the inherent conflict between preserving skeleton interaction semantics and correcting geometry leads to jittery movements, severe interpenetration and imprecise contacts. Zhang et al. [30] subsequently proposed adding a stage that aligns visual semantics with a visual language model, but this requires detailed pair-by-pair finetuning due to the loss of spatial information when projecting 3D motion into 2D images.

To resolve the conflict between skeleton interaction and geometry correction, we propose a new approach: focusing solely on dense geometric interaction for motion retargeting. Character animation videos, rendered from the skinned mesh, rely on geometric interactions to shape user perception. Skeleton interaction, in contrast, merely represents a simplified, sparse form of geometric interaction. Therefore, maintaining correct interactions between different body part geometries not only preserves motion semantics but also prevents mesh interpenetration and ensures contact preservation, as illustrated in Figure 1.

Given the significance of geometric interactions, we propose a new framework, named MeshRet, for skinned motion retargeting. In contrast to earlier methods that adjust skeletal motion retargeting outcomes, our approach models the intricate interactions among character meshes without depending on predefined vertex correspondences.

The design of MeshRet necessitates several technical innovations. Initially, there is a requirement for dense mesh correspondence across different characters. Drawing inspiration from the medial axis inverse transform (MAIT) [22], we have devised a technique, termed semantically consistent sensors (SCS), to automatically derive dense mesh correspondence from sparse skeleton correspondence. This technique enables us to sample a point cloud of sensors on the mesh to represent each character. Following this, to illustrate dense mesh interaction between body parts, we employ interacting mesh sensor pairs, maintaining generality. These pair-wise interactions are encoded within a novel spatial-temporal representation termed the Dense Mesh Interaction (DMI) field. The DMI field adeptly encapsulates both contact and non-contact interaction semantics. Finally, we proceed to learn a motion manifold that aligns with the target character geometry and the source motion DMI field.

To align our evaluation process more closely with real animation production, we gathered an in-the-wild motion dataset, termed ScanRet, characterized by abundant contact semantics and minimal mesh interpenetration. ScanRet consists of 100 human actors ranging from bulky to skinny, each performing 83 motion clips scrutinized by human animators. The MeshRet model is trained on both the ScanRet dataset and the widely used Mixamo [2] dataset. We assessed our method across a large variety of motions and a diverse array of target characters. Both qualitative and quantitative analyses show that our MeshRet model significantly outperforms existing methods.

To summarize, we present the following contributions:

- We introduce MeshRet, a pioneering solution that facilitates geometric interaction-aware motion retargeting across varied mesh topologies in a single pass.   
- We present the SCS and the novel DMI field to guide the training of MeshRet, effectively encapsulating both contact and non-contact interaction semantics.   
- We develop ScanRet, a novel dataset specifically tailored for assessing motion retargeting technologies, which includes detailed contact semantics and ensures smooth mesh interaction.   
- Our experiments demonstrate that MeshRet delivers exceptional performance, marked by accurate contact preservation and high-quality motion.

# 2 Related Work

Skeletal motion retargeting Motion retargeting seeks to preserve the characteristics of source motions when transferring them to a different target character. Skeletal motion retargeting primarily addresses the challenge of differing bone ratios. Gleicher [8] initially formulated motion retargeting as a spatio-temporal optimization problem, using source motion features as kinematic constraints. Subsequent researches [5, 7, 14] have focused on optimization-based approaches with various constraints. However, these methods, while requiring extensive optimization, often yield suboptimal

Existing Method   
![](images/f481a9cce6bcbfc2c5daffaa8eb357de27854851aab35a4beda1ae4d9db185e0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Skeleton-aware Retargeting"] --> B["Geometry Correction"]
    B --> C["Contradiction"]
```
</details>

The Proposed MeshRet   
![](images/70ba06ed70de9918330a610fba15762f5fe7295b0c419b57a99ca0e116e681a1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image"] --> B["DMI Field Extraction"]
    B --> C["Geometric Interaction-aware Retargeting"]
    C --> D["Final Object Tracking"]
```
</details>

Figure 1: Comparison with the existing method. Contrary to the earlier retargeting-correction approach [32], which suffer from internal contradictions leading to interpenetration, jitter, and contact mismatches, our pipeline leverages the DMI field to accurately model complex geometric interactions.

results. Consequently, recent studies have explored learning-based motion retargeting algorithms. Jang et al. [11] trained a motion retargeting network using a U-Net [26] architecture on paired motion data. Villegas et al. [29] introduced a recurrent neural network combined with cycle-consistency [35] for unsupervised motion retargeting. Lim, Chang, and Choi [15] propose to learn frame-by-frame poses and overall movements separately. Aberman et al. [1] develop differentiable operators for cross-structural motion retargeting among homeomorphic skeletons. However, these methods generally neglect the geometry of characters, leading to frequent contact mismatches and severe mesh interpenetrations.

Geometry-aware motion retargeting Previous studies have generally processed character geometries through two approaches: contact preservation and interpenetration avoidance. Lyard and Magnenat-Thalmann [19] developed a heuristic optimization algorithm to maintain character self-contact, while Ho, Komura, and Tai [9] proposed to maintain character interactions by minimizing the deformation of interaction meshes. Ho and Shum [10] introduced a spatio-temporal optimization framework to prevent self-collisions in robot motion retargeting. Jin, Kim, and Lee [12] employed a proxy volumetric mesh to preserve spatial relationships during retargeting. Subsequently, Basset et al. [4] combined both attraction and repulsion terms in an optimization-based method to avoid interpenetration and preserve contact. However, these methods necessitate per-vertex correspondence and involve costly optimization processes. More recently, Villegas et al. [28] attempted to retarget skinned motion through optimization in a latent space of a pretrained network, although their method does not accommodate non-contact interactions. Zhang et al. [32] implemented a two-stage pipeline that initially aligns skeleton interaction semantics and subsequently corrects geometric artifacts. Nevertheless, the inherent conflict between maintaining skeleton interaction semantics and correcting geometry often results in jittery movements and imprecise contacts. In a later study, Zhang et al. [30] added a stage that aligns visual semantics using a visual language model, but this approach requires extensive pair-by-pair fine-tuning due to the loss of spatial information when projecting 3D motion into 2D images.

Existing geometry-aware motion retargeting methods either require expensive optimization or employ multi-stage strategies for skeleton and geometry semantics, resulting in a contradiction between stages that often leads to unsatisfactory results. In contrast, our method processes both contact and non-contact semantics using a dense mesh interaction field in a single stage.

# 3 Method

# 3.1 Overview

We introduce a novel geometric interaction-aware motion retargeting framework MeshRet, as illustrated in Figure 2. Unlike previous methods that either overlook character geometries [1, 29, 15] or apply geometry correction after skeleton retargeting [32, 30], our framework directly addresses dense geometric interactions with the Dense Mesh Interaction (DMI) field. This provides a detailed representation of the interactions within skinned character motions, preserving motion semantics by preventing mesh interpenetration and ensuring precise contact preservation.

Motion & geometry representations Assume the motion sequence has $T$ frames and the character has $N$ skeletal joints. The motion sequence $\mathbf{m}$ is represented by the global root translation $\mathbf{X} \in \mathbb{R}^{T \times 3}$

![](images/8559d5e69f1eed3827efc310d968d89934cdf9684ad84f02650581bb02a56bfd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Source Motion"] --> B["Q_A"]
    B --> C["F_k"]
    C --> D["\overline{D}_A"]
    D --> E["F_c"]
    E --> F["Source DMI"]
    F --> G["DMI Encoder"]
    G --> H["Transformer Encoder"]
    H --> I["Target M_DMI"]
    I --> J["\hat{D}_B"]
    J --> K["F_k"]
    K --> L["\overline{D}_B"]
    L --> M["F_c"]
    M --> N["Target M_tet"]
    N --> O["S_B"]
    O --> P["F_g"]
    P --> Q["Transformer Decoder"]
    Q --> R["Target M_B"]
    R --> S["F_g"]
    S --> T["S_A"]
    T --> U["M_src"]
    U --> V["Source DMI"]
    V --> W["DMI Consistency"]
    style A fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style P fill:#f9f,stroke:#333
    style Q fill:#f9f,stroke:#333
    style R fill:#f9f,stroke:#333
    style S fill:#f9f,stroke:#333
    style T fill:#f9f,stroke:#333
```
</details>

Figure 2: Overview of the proposed MeshRet. The pipeline begins with the extraction of the DMI field using sensor forward kinematics, denoted as $\mathcal{F}_k$ , and pairwise interaction feature selection, represented by $\mathcal{F}_c$ . This DMI field, in conjunction with geometric features derived from $\mathcal{F}_g$ , is fed into an encoder-decoder network. The network predicts the target motion sequence, which is aligned with the target character's geometry and the original DMI field.

and the local joint rotation $\mathbf{Q} \in \mathbb{R}^{T \times N \times 6}$ , where we adopt the 6D representation [34] for the joint rotations. The rest-pose geometry $\mathbf{G}$ of the character is represented by the rest-pose mesh $\mathbf{O}$ and the rest-pose joint locations $\mathbf{J} \in \mathbb{R}^{N \times 3}$ .

Task definition Given the source motion sequence $m_{A}$ , and the geometries $G_{A}$ and $G_{B}$ of the source and target characters in their T-poses, our objective is to generate the motion $m_{B}$ for the target character. This process aims to retain essential aspects of the source motion, including its semantics, contact preservation, and the avoidance of interpenetration.

Following the definition of the task, our MeshRet model initially derives Semantically Consistent Sensors (SCS) $\mathbf{S} \in \mathbb{R}^{S \times 4 \times 3}$ , which provide dense geometric correspondences essential for the retargeting process, where $\mathbf{S} = \mathcal{F}_{\mathrm{s}}(\mathbf{G})$ . $\mathbf{S}$ captures the sensor location and the sensor tangent space matrix, facilitating an enhanced perception of the geometry surface. Subsequently, we conduct sensor forward kinematics (FK) and pairwise interaction extraction to generate the source DMI field $\mathbf{D}_{\mathrm{A}} = \mathcal{F}_{\mathrm{d}}(\mathbf{m}_{\mathrm{A}}, \mathbf{S}_{\mathrm{A}})$ , where $\mathbf{D}_{\mathrm{A}} \in \mathbb{R}^{T \times K \times L \times P}$ . Here, $K$ is the number of SCS in the DMI field, $L$ represents a hyper-parameter of feature selection, and $P$ indicates the feature dimension of the DMI. Lastly, a transformer-based network [27] ingests $\mathbf{m}_{\mathrm{A}}$ , $\mathbf{D}_{\mathrm{A}}$ , $\mathbf{S}_{\mathrm{A}}$ , and $\mathbf{S}_{\mathrm{B}}$ , and predicts a target motion sequence $\mathbf{m}_{\mathrm{B}}$ that aligns with the target character's geometry and the source DMI field. The entire pipeline is denoted as follows:

$$
\mathbf {m} _ {\mathrm{B}} = \mathcal {F} _ {\mathrm{r}} (\mathbf {m} _ {\mathrm{A}}, \mathbf {D} _ {\mathrm{A}}, \mathbf {S} _ {\mathrm{A}}, \mathbf {S} _ {\mathrm{B}}) \tag {1}
$$

# 3.2 Semantically consistent sensors

To facilitate dense geometric interactions, our MeshRet framework necessitates establishing dense mesh correspondence between source and target characters. Previous studies have typically derived correspondence from vertex coordinates $[33]$ , virtual sensor $[31]$ or through a bounding mesh $[12]$ ; however, these methods are confined to template meshes sharing identical topology, such as MANO $[25]$ or SMPL $[18]$ . Villegas et al. $[28]$ suggested determining vertex correspondence using nearest neighbor searches on predefined feature vectors. Nevertheless, this approach often lacks precision and brevity, resulting in inaccurate contact representations and substantial optimization burdens.

In this study, we introduce Semantically Consistent Sensors (SCS) that are effective across various mesh topologies while ensuring precise semantic correspondence. Our approach draws inspiration

![](images/7faf9bf87c3a098d3d0ba82cdacefbb55260c89dbe27caccd07d840c6c0d2911.jpg)

<details>
<summary>text_image</summary>

Joints (by b)
Sensor Tangent
Space
Bone Matrix
</details>

Figure 3: Left: Illustration of the method to derive a sensor feature s from the semantic coordinate $(b, l, \phi)$ across different characters. The red line represents the projected ray. The feature s encompasses the sensor's location and its tangent space matrix. Right: The DMI field effectively captures both contact and non-contact interactions. Red lines represent $d^{t,i,j}$ in the DMI field. In the second example, the body sensors (yellow points) are located in the tangent plane of the hand sensors (blue points), signifying a contact interaction.

from the Medial Axis Inverse Transform (MAIT) [22]. We conceptualize the skeleton bones of each character as approximate medial axes of their limbs and torso. For each bone, a MAIT-like transform is applied to generate the corresponding SCS. This involves casting rays from the bone axis across a plane perpendicular to it. The origin parameter l and direction parameter $\phi$ of the rays, combined with the bone index b, establish the semantic coordinates of the SCS. The semantic coordinates describe connection between the sensor and the skeleton bones. A sensor is deemed valid if its ray intersects the mesh linked to the bone; otherwise, it is considered invalid. Through this method, we establish a dense geometric correspondence based on sparse skeletal correspondence. The procedure for deriving SCS is illustrated in Figure 3. Given a unified set of SCS semantic coordinates $\{(b_{1}, l_{1}, \phi_{1}), (b_{2}, l_{2}, \phi_{2}), \cdots, (b_{S}, l_{S}, \phi_{S})\}$ , we can derive SCS feature $S = \{s_{1}, s_{2}, \cdots, s_{S}\}$ for each character. Further details can be found in Algorithm 1.

# 3.3 Dense mesh interaction field

To effectively represent the interactions between character limbs and the torso, we have developed the DMI field. Based on SCS detailed in Section 3.2, the DMI field comprehensively captures both contact and non-contact interactions across different body part geometries. Utilizing the DMI field allows for dense geometry interaction-aware motion retargeting, thereby eliminating the need for a geometry correction stage.

Sensor forward kinematics For a given motion sequence, denoted as $\mathbf{m}$ , we initially conduct forward kinematics (FK) on $\mathbf{S}$ to derive sensor features $\mathbf{S}^{1:T} \in \mathbb{R}^{T \times S \times 4 \times 3}$ . Each $\mathbf{S}^t$ encompasses the locations and tangent matrices for $S$ sensors at frame $t$ . The FK transformation for an individual sensor is expressed as:

$$
\mathbf {s} _ {i} ^ {t} = \sum_ {n = 1} ^ {N} \omega (\mathbf {p} _ {i}) _ {n} G _ {n} (\mathbf {Q} ^ {t}) \cdot \mathbf {s} _ {i}, \tag {2}
$$

where $G_{n}(\mathbf{Q}^{t}) \in SE(3)$ is the global transformation matrix for bone n, derived from its local rotation matrix, and $\omega(\mathbf{p}_{i})_{n}$ represents the linear blend skinning (LBS) weight for sensor $s_{i}$ , determined through barycentric interpolation of its adjacent mesh vertices.

Pairwise interaction feature Next, we model the geometric interactions as pairwise interaction features between sensors. Ideally, for each frame, we obtain a comprehensive DMI field, $\overline{D}^{t}$ , representing pairwise vectors across $K^{2}$ sensor pairs:

$$
\mathbf {d} ^ {t, i, j} = \mathbf {t} _ {i} ^ {- 1} (\mathbf {p} _ {j} ^ {t} - \mathbf {p} _ {i} ^ {t}), \tag {3}
$$

$$
\overline {{{\mathbf {D}}}} ^ {t} = \left\{\left(\mathbf {d} ^ {t, i, j}, b _ {i}, b _ {j}, l _ {i}, l _ {j}, \phi_ {i}, \phi_ {j}\right) \right\} _ {i = 1: S} ^ {j = 1: S}, \tag {4}
$$

where $t_{i} \in R^{3 \times 3}$ is the tangent matrix if sensor i, and $d^{t,i,j}$ represents the relative position of target sensor j in the tangent space of observation sensor i. $\overline{D}^{t}$ is composed of two components: the relative position of the sensor pair and the semantic coordinates of both the observation and target sensors. The use of semantic rather than spatial coordinates is essential, as it obviates the need for actual sensor positions, thereby making DMI suitable for motion retargeting applications.

However, $\overline{D}^{t} \in R^{S \times S \times P}$ exhibits quadratic growth with respect to S because it includes $S^{2}$ sensor pairs, rendering it impractical when managing thousands of sensors. To address this, we implement two sparsification strategies for $\overline{D}^{t}$ . Initially, we restrict interactions to critical body parts only, such as arm-torso, arm-head, arm-arm, and leg-leg, rather than between all sensor pairs, thereby restricting our focus to K observation sensors. Subsequently, for each observation sensor, we select L target sensors from each relevant body part, where L is a predetermined hyper-parameter. Specifically, we empirically choose L/2 nearest and L/2 furthest target sensors. We find that proximate sensor pairs are crucial for minimizing interpenetration and maintaining contact, while distant pairs delineate the overall spatial relationships between body parts, as shown in Figure 3. These strategies lead to the formulation of the final DMI field $D \in R^{K \times L \times P}$ , with selected sensor pairs indicated by the sparse DMI mask $M_{src} \in R^{S \times S}$ shown in Figure 2.

# 3.4 Geometry interaction-aware motion retargeting

To avoid the conflict between skeleton interaction and geometric correction, the proposed MeshRet employs the DMI field to model geometric interactions directly. As shown in Figure 2, MeshRet initially extracts the DMI field $\mathbf{D}_{\mathrm{A}}$ from the source motion sequence $\mathbf{m}_{\mathrm{A}}$ , as described in Section 3.3. The field $\mathbf{D}_{\mathrm{A}}$ encapsulates interactions among various body parts within the source motion, encompassing both contact and non-contact interactions, further depicted in Figure 3. The DMI field, composed of sensor pair feature vectors, possesses the unordered characteristics of a point cloud. Consequently, we implement a PointNet-like architecture [24] for our DMI encoder, which is divided into two components: the per-sensor encoder and the per-frame encoder. Given $\mathbf{D}_{\mathrm{A}} \in \mathbb{R}^{T \times K \times L \times P}$ , the per-sensor encoder initially processes it as $T * K$ separate point clouds, producing representations $\mathbf{H}_{\mathrm{A}}^{\mathrm{s}} \in \mathbb{R}^{T \times K \times D_{\mathrm{model}}}$ for each observation sensor, where $D_{\mathrm{model}}$ denotes the feature dimension. Subsequently, the per-frame encoder generates per-frame representations $\mathbf{H}_{\mathrm{A}}^{\mathrm{f}} \in \mathbb{R}^{T \times D_{\mathrm{model}}}$ by encoding these $T$ point clouds.

Since DMI field $D_{A}$ lacks geometric information about characters, we introduced a geometry encoder $F_{g}$ to extract geometric features from their SCS. For each sensor, we form a feature vector by concatenating its rest-pose feature $s_{i}$ with its semantic coordinates $(b_{i}, l_{i}, \phi_{i})$ . The resultant geometric features are represented as $C_{A} \in R^{S_{A} \times C}$ for character A and $C_{B} \in R^{S_{B} \times C}$ for character B. The semantic coordinates of sensors act as intermediaries linking the DMI field to character geometry. The geometry encoder employs a PointNet-like architecture [24] to transform the geometric features C into a geometric latent code $H^{g} \in R^{D_{model}}$ .

The transformer-based retargeting network processes input features including the source DMI feature $\mathbf{H}_{\mathrm{A}}^{\mathrm{f}}$ , source joint rotation $\mathbf{Q}_{\mathrm{A}}$ , source geometry latent $\mathbf{H}_{\mathrm{A}}^{\mathrm{g}}$ , and target geometry latent $\mathbf{H}_{\mathrm{B}}^{\mathrm{g}}$ . Specifically, the encoder processes $\mathbf{H}_{\mathrm{A}}^{\mathrm{f}}$ and $\mathbf{H}_{\mathrm{B}}^{\mathrm{g}}$ , while the decoder processes $\mathbf{Q}_{\mathrm{A}}$ and $\mathbf{H}_{\mathrm{A}}^{\mathrm{g}}$ . The latents $\mathbf{H}_{\mathrm{A}}^{\mathrm{g}}$ and $\mathbf{H}_{\mathrm{B}}^{\mathrm{g}}$ serve as the initial tokens in the sequence, enabling both the encoder and decoder to operate over a sequence of length $T + 1$ . The output sequence's final $T$ frames are represented as $\hat{\mathbf{Q}}_{\mathrm{B}}$ .

Due to the lack of paired ground-truth data, we employ the unsupervised method described by Lim, Chang, and Choi [15]. Our network utilizes four loss functions for training: reconstruction loss, DMI consistency loss, adversarial loss, and end-effector loss. Supervision signals are derived from the source motion. We maintain geometric interactions by aligning the source DMI field $D_{A}$ with the target DMI field $\hat{D}_{B}$ . The target DMI field $\hat{D}_{B}$ is generated by first applying sensor forward kinematics to $\hat{Q}_{B}$ , followed by selecting sensor pairs using the target sparse DMI mask $M_{tgt} \in R^{S \times S}$ . This mask, $M_{tgt}$ , is derived by excluding invalid sensors of the target character from $M_{src}$ . The DMI consistency loss is quantified as the cosine similarity loss between pair-wise relative positions in $\hat{D}_{B}$ and $D_{A}$ :

$$
\mathcal {L} _ {\mathrm{dmi}} = - \frac {1}{T} \sum_ {t = 1} ^ {T} \sum_ {k = 1} ^ {K} \sum_ {l = 1} ^ {L} c (k, l) \frac {\mathbf {d} _ {\mathrm{A}} ^ {t , k , l} \cdot \hat {\mathbf {d}} _ {\mathrm{B}} ^ {t , k , l}}{\left| \left| \mathbf {d} _ {\mathrm{A}} ^ {t , k , l} \right| \right| _ {2} \cdot \left| \left| \hat {\mathbf {d}} _ {\mathrm{B}} ^ {t , k , l} \right| \right| _ {2}}, \tag {5}
$$

where $c(k,l)$ takes the value 1 if sensor pair $(k,l)$ is valid in both $M_{src}$ and $M_{tgt}$ , and 0 otherwise. The reconstruction loss serves as a regularization mechanism to minimize motion alterations during retargeting, defined as follows:

$$
\mathcal {L} _ {\text { rec }} = \left| \left| \hat {\mathbf {Q}} _ {\mathrm{B}} - \mathbf {Q} _ {\mathrm{A}} \right| \right| _ {2} ^ {2}. \tag {6}
$$

To facilitate realistic motion retargeting, a discriminator, denoted as $\delta(\cdot)$ , is employed. The adversarial loss is subsequently defined as:

$$
\mathcal {L} _ {\mathrm{adv}} = \mathbb {E} _ {\mathbf {Q} \sim p _ {\text { real }}} [ \log \delta (\mathbf {Q}) ] + \mathbb {E} _ {\mathbf {Q} \sim p (\hat {\mathbf {Q}} _ {\mathrm{B}})} [ \log (1 - \delta (\mathbf {Q})) ]. \tag {7}
$$

We observed that the global orientation of end-effectors significantly influences user experience. Consequently, we introduced an end-effector loss to promote consistent orientations of end-effectors in the retargeted motion.

$$
\mathcal {L} _ {\mathrm{ef}} = \frac {1}{T | \mathcal {X} |} \sum_ {t = 1} ^ {T} \sum_ {i \in \mathcal {X}} | | R (\mathbf {Q} _ {\mathrm{A}} ^ {t}, i) - R (\hat {\mathbf {Q}} _ {\mathrm{B}} ^ {t}, i) | |, \tag {8}
$$

where $R(\cdot)$ transforms local joint rotations into global rotations for joint i along the kinematic chain and X represents the set of end-effectors. Our MeshRet is trained by:

$$
\mathcal {L} _ {\text { total }} = \lambda_ {\text { rec }} \mathcal {L} _ {\text { rec }} + \lambda_ {\text { dmi }} \mathcal {L} _ {\text { dmi }} + \lambda_ {\text { adv }} \mathcal {L} _ {\text { adv }} + \lambda_ {\text { ef }} \mathcal {L} _ {\text { ef }}. \tag {9}
$$

# 4 Experiments

# 4.1 Settings

Datasets We trained and evaluated our method using the Mixamo dataset $[2]$ and the newly curated ScanRet dataset. We downloaded 3,675 motion clips performed by 13 cartoon characters from the Mixamo dataset contains, while the ScanRet dataset consists of 8,298 clips executed by 100 human actors. Notably, the Mixamo dataset frequently features corrupted data due to interpenetration and contact mismatches. To overcome these issues, we created the ScanRet dataset, which provides detailed contact semantics and improved mesh interactions, with each clip being scrutinized by human animators. The training set comprises 90% of the motion clips from both datasets, involving nine characters from Mixamo and 90 from ScanRet. Our experiments tested the motion retargeting capabilities between cartoon characters and real humans, aligning closely with typical retargeting workflows. During inference, we adopted four data splits based on character and motion visibility: unseen character with unseen motion (UC+UM), unseen character with seen motion (UC+SM), seen character with unseen motion (SC+UM), and seen character with seen motion (SC+SM), as delineated by Zhang et al. $[32]$ . We present the average results across these splits. Additional details available in Appendix A.

Implementation details The hyper-parameters $\lambda_{rec}$ , $\lambda_{dmi}$ , $\lambda_{adv}$ , $\lambda_{ef}$ , and L were empirically set to 1.0, 5.0, 1.0, 1.0, and 20, respectively. We use $\{0,1,\cdots,N_{body}-1\}\times\{0,0.25,0.5,0.75\}\times\{0,0.5\pi,\pi,1.5\pi\}$ as the SCS semantic coordinates set, where $N_{body}=18$ is the number of body bones and $\times$ represents the Cartesian product. We employed the Adam optimizer [13] with a learning rate of $10^{-4}$ to optimize our network. The training process required 36 epochs. For further details, please refer to Appendix C.

Evaluation metrics We assess the effectiveness of our method through three metrics: joint accuracy, contact preservation, and geometric interpenetration. Joint accuracy is quantified by calculating the Mean Squared Error (MSE) between the retargeted joint positions and the ground-truth data provided by animators in ScanRet. This analysis considers both global and local joint positions, normalized by the character heights. Contact preservation is evaluated by measuring the Contact Error, defined as the mean squared distance between sensors that were originally in contact in the source motion clip. Geometric interpenetration is determined by the ratio of penetrated limb vertices to the total limb vertices per frame. Further details are available in Appendix B.

# 4.2 Comparison with state-of-the-arts

Qualitative results Figure 4 demonstrates the performance of skinned motion retargeting across characters with diverse body shapes, where the motion sequences are novel to the target characters

![](images/fb41dfd867986865771659493f56fd9d90e6a1cfd749ec32ab142c8eb10d6178.jpg)

<details>
<summary>text_image</summary>

Source
Copy
PMnet
SAN
R² ET
Ours
</details>

Figure 4: Qualitative comparison with baseline methods. Our method ensures precise contact preservation and minimal geometric interpenetration.

during training. Most baseline methods, except $R^{2}ET$ [32], fail to consider the geometry of characters, leading to significant geometric interpenetration and contact mismatches. Unlike these methods, $R^{2}ET$ [32] includes a geometry correction phase after skeleton-aware retargeting. However, this creates a conflict between the two stages, resulting in oscillations in $R^{2}ET$ 's outcomes, which manifest as alternating contact misses and severe interpenetrations, as shown in the first two rows. Additionally, these oscillations appear variably across different frames within the same motion clip, producing jittery motion, as illustrated in Figure 1 and Figure 8. A further limitation of $R^{2}ET$ is its neglect of hand contacts. In contrast, our method employs the innovative DMI field to preserve such detailed interactions, such as those observed in the “Praying” pose in the third row.

Table 1: Quantitative comparison between our method and state-of-the-arts. Mixamo+ represents the mixed dataset of Mixamo and ScanRet. MSE $^{lc}$ denotes the local MSE. 

<table><tr><td>Metric</td><td>MSE↓</td><td>MSE $^{lc}$ ↓</td><td colspan="2">Contact Error↓</td><td colspan="2">Penetration(%)↓</td></tr><tr><td>Dataset</td><td>ScanRet</td><td>ScanRet</td><td>Mixamo+</td><td>ScanRet</td><td>Mixamo+</td><td>ScanRet</td></tr><tr><td>Source</td><td>-</td><td>-</td><td>-</td><td>0.234</td><td>3.04</td><td>1.37</td></tr><tr><td>Copy</td><td>0.026</td><td>0.006</td><td>1.702</td><td>0.387</td><td>5.26</td><td>2.16</td></tr><tr><td>PMnet [15]</td><td>0.130</td><td>0.029</td><td>2.716</td><td>0.890</td><td>5.23</td><td>2.23</td></tr><tr><td>SAN [1]</td><td>0.049</td><td>0.011</td><td>2.432</td><td>0.627</td><td>4.95</td><td>1.72</td></tr><tr><td>R $^2$ ET [32]</td><td>0.063</td><td>0.017</td><td>2.209</td><td>0.589</td><td>4.21</td><td>2.01</td></tr><tr><td>Ours $_{cls}$ </td><td>0.048</td><td>0.013</td><td>0.800</td><td>0.426</td><td>3.35</td><td>1.73</td></tr><tr><td>Ours $_{far}$ </td><td>0.045</td><td>0.010</td><td>1.642</td><td>0.610</td><td>5.37</td><td>1.77</td></tr><tr><td>Ours $_{dm}$ </td><td>0.048</td><td>0.010</td><td>2.568</td><td>0.797</td><td>4.69</td><td>1.78</td></tr><tr><td>Ours</td><td>0.047</td><td>0.009</td><td>0.772</td><td>0.284</td><td>3.45</td><td>1.59</td></tr></table>

Quantitative results Table 1 presents a comparison between our methods and state-of-the-arts. We initially measure the joint location error using MSE and $MSE^{lc}$ on ScanRet. The ground truth in ScanNet is established by human animators. Our observations indicate that human animators typically retarget motions by initially replicating joint rotations and subsequently modifying frames

![](images/2fe806416998df2ea9adfbd28ba16809471bc506a216d0f86e9484ab477ffc19.jpg)

<details>
<summary>text_image</summary>

Source
Ours
Ours_cls
Ours_far
Ours_dm
</details>

![](images/927b3a7e6926b9b273497f1366124d429ce9f46ad2d7a910196aa7f6ea5527e8.jpg)

<details>
<summary>text_image</summary>

Source Ours Ours_cls Ours_far Ours_dm
</details>

Figure 5: Qualitative comparison of ablation studies. A red circle highlights areas of interpenetration, while a red rectangle identifies errors in non-contact semantics.

that display incorrect interactions. Conversely, our method modifies the entire motion sequence, resulting in a higher MSE compared to the Copy strategy. Nevertheless, MSE remains a valuable auxiliary reference. In comparison to PMnet [15], R²ET [32], and SAN [1], our method achieves MSE reductions of 65%, 29%, and 8%, respectively. These results demonstrate that our approach more closely aligns with the outputs produced by human animators.

As shown in Table 1, PMnet [15] and SAN [1], exhibit high interpenetration ratios and contact errors due to their neglect of character geometries. R²ET [32] effectively reduces interpenetration through a geometry correction stage; nonetheless, it still encounters high contact errors stemming from conflicts between the retargeting and correction stages. Our approach explicitly models geometry interactions and thereby achieves low contact error and penetration ratio, illustrating the effectiveness of our proposed MeshRet in generating high-quality retargeted motions with detailed contact semantics and smooth mesh interactions. Additionally, we observe that retargeting using the mixed Mixamo+ dataset is more challenging than with the ScanRet dataset, attributable to significant body shape variations between cartoon characters and real person characters.

# 4.3 Ablation Studies

We conducted ablation studies to demonstrate the significance of pairwise interaction feature selection and the implementation of DMI similarity loss. Initially, we evaluated the performance of a model trained exclusively with the nearest L sensor pairs, denoted as $Ours_{cls}$ , and another model trained solely with the farthest L sensor pairs, referred to as $Ours_{far}$ . As indicated in Table 1 and Figure 5, $Ours_{far}$ compromises contact semantics and leads to significant interpenetration, while $Ours_{cls}$ also exhibits inferior performance. This outcome suggests that proximal sensor pairs are essential for minimizing interpenetration and preserving contact, whereas distal pairs provide insights into the non-contact spatial relationships among body parts. Further, we investigated the effect of incorporating a distance matrix loss, as proposed by Zhang et al. [32], on our sensor pairs, designated as $Ours_{dm}$ . The results imply that the distance matrix loss fails to yield meaningful supervisory signals, likely because distance is non-directional and insufficient to discern the relative spatial positions among numerous sensors.

Table 2: Human preferences between our method and baselines. 

<table><tr><td>Methods</td><td>Semantics Preservation</td><td>Contact Accuracy</td><td>Overall Quality</td></tr><tr><td>Copy</td><td>20.7% v.s. 79.3%(Ours)</td><td>22.7% v.s. 77.3%(Ours)</td><td>18.7% v.s. 81.3%(Ours)</td></tr><tr><td>PMnet [15]</td><td>2.7% v.s. 97.3%(Ours)</td><td>5.3% v.s. 94.7%(Ours)</td><td>1.3% v.s. 98.7%(Ours)</td></tr><tr><td>SAN [1]</td><td>9.3% v.s. 90.7%(Ours)</td><td>15.3% v.s. 84.7%(Ours)</td><td>7.3% v.s. 92.7%(Ours)</td></tr><tr><td>R2ET [32]</td><td>14.6% v.s. 85.4%(Ours)</td><td>16.0% v.s. 84.0%(Ours)</td><td>13.3% v.s. 86.7%(Ours)</td></tr></table>

# 4.4 User study

We conducted a user study to assess the performance of our MeshRet model in comparison with the Copy strategy, PMnet [15], SAN [1], and R²ET [32]. Fifteen sets of motion videos were presented to participants, each consisting of one source skinned motion and five anonymized skinned results. Participants were requested to rate their preferences based on three criteria: semantic preservation, contact accuracy, and overall quality. Users were recruited from Amazon Mechanical Turk [3],

resulting in a total of 600 comparative evaluations. As indicated in Table 2, approximately 81% of the comparisons favored our results. Details can be found in Appendix D

# 5 Conclusion

We introduce a novel framework for geometric interaction-aware motion retargeting, named MeshRet. This framework explicitly models the dense geometric interactions among various body parts by first establishing a dense mesh correspondence between characters using semantically consistent sensors. We then develop a unique spatio-temporal representation, termed the DMI field, which adeptly captures both contact and non-contact interactions between body geometries. By aligning this DMI field, MeshRet achieves detailed contact preservation and seamless geometric interaction. Performance evaluations using the Mixamo dataset and our newly compiled ScanRet dataset confirm that MeshRet offers state-of-the-art results.

Limitations The primary limitation of MeshRet is its dependence on inputs with clean contact; motion clips exhibiting severe interpenetration yield poor outcomes. Consequently, it is unable to process noisy inputs effectively. Refer to Figure 12 and Figure 13 for failure cases under noisy inputs. Future efforts will focus on enhancing its robustness to noisy data. Additionally, SCS extraction can be compromised by noisy meshes, particularly those with complex clothing. A potential solution is to employ a Laplacian-smoothed proxy mesh for SCS extraction. Lastly, the method cannot handle characters with missing limbs.

# Acknowledgments and Disclosure of Funding

This work is supported by the National Key R&D Program of China under Grant No. 2024QY1400, the National Natural Science Foundation of China No. 62425604, and the Tsinghua University Initiative Scientific Research Program. Mike Shou does not receive any funding for this work.

# References

[1] Kfir Aberman et al. “Skeleton-aware networks for deep motion retargeting”. In: ACM Trans. Graph. 39.4 (2020), p. 62.   
[2] Adobe. Mixamo. https://www.mixamo.com/. 2018.   
[3] Amazon. Amazon Mechanical Turk. https://www.mturk.com/.   
[4] Jean Basset et al. “Contact preserving shape transfer: Retargeting motion from one shape to another”. In: Computers & Graphics 89 (2020), pp. 11–23.   
[5] Antonin Bernardin et al. “Normalized Euclidean distance matrices for human motion retargeting”. In: Proceedings of the 10th International Conference on Motion in Games. 2017, pp. 1–6.   
[6] Yingruo Fan et al. “Faceformer: Speech-driven 3d facial animation with transformers”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022, pp. 18770–18780.   
[7] Andrew Feng et al. “Automating the transfer of a generic set of behaviors onto a virtual character”. In: Motion in Games: 5th International Conference, MIG 2012, Rennes, France, November 15-17, 2012. Proceedings 5. Springer. 2012, pp. 134–145.   
[8] Michael Gleicher. “Retargetting motion to new characters”. In: Proceedings of the 25th annual conference on Computer graphics and interactive techniques. 1998, pp. 33–42.   
[9] Edmond S. L. Ho, Taku Komura, and Chiew-Lan Tai. “Spatial relationship preserving character motion adaptation”. In: ACM Trans. Graph. 29.4 (2010), 33:1–33:8.   
[10] Edmond SL Ho and Hubert PH Shum. “Motion adaptation for humanoid robots in constrained environments”. In: 2013 IEEE International Conference on Robotics and Automation. IEEE. 2013, pp. 3813–3818.   
[11] Hanyoung Jang et al. “A variational u-net for motion retargeting”. In: SIGGRAPH Asia 2018 Posters. 2018, pp. 1–2.   
[12] Taeil Jin, Meekyoung Kim, and Sung-Hee Lee. “Aura mesh: Motion retargeting to preserve the spatial relationships between skinned characters”. In: Computer Graphics Forum. Vol. 37. 2. Wiley Online Library. 2018, pp. 311–320.   
[13] Diederik P. Kingma and Jimmy Ba. “Adam: A Method for Stochastic Optimization”. In: 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings. Ed. by Yoshua Bengio and Yann LeCun. 2015.   
[14] Jehee Lee and Sung Yong Shin. “A hierarchical approach to interactive motion editing for human-like figures”. In: Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999, pp. 39–48.   
[15] Jongin Lim, Hyung Jin Chang, and Jin Young Choi. “PMnet: Learning of Disentangled Pose and Movement for Unsupervised Motion Retargeting.” In: BMVC. Vol. 2. 6. 2019, p. 7.   
[16] Jinghuai Lin and Marc Erich Latoschik. “Digital body, identity and privacy in social virtual reality: A systematic review”. In: Frontiers in Virtual Reality 3 (2022), p. 974652.   
[17] Matthew Loper, Naureen Mahmood, and Michael J Black. “MoSh: motion and shape capture from sparse markers.” In: ACM Trans. Graph. 33.6 (2014), pp. 220–1.   
[18] Matthew Loper et al. “SMPL: a skinned multi-person linear model”. In: ACM Trans. Graph. 34.6 (2015), 248:1–248:16.   
[19] Etienne Lyard and Nadia Magnenat-Thalmann. “Motion adaptation based on character shape”. In: Computer Animation and Virtual Worlds 19.3-4 (2008), pp. 189–198.   
[20] Naureen Mahmood et al. “AMASS: Archive of motion capture as surface shapes”. In: Proceedings of the IEEE/CVF international conference on computer vision. 2019, pp. 5442–5451.   
[21] Lucas Mourot et al. “A survey on deep learning for skeleton-based human animation”. In: Computer Graphics Forum. Vol. 41. 1. Wiley Online Library. 2022, pp. 122–157.   
[22] Henning Naß et al. “Medial axis (inverse) transform in complete 3-dimensional Riemannian manifolds”. In: 2007 International Conference on Cyberworlds (CW’07). IEEE. 2007, pp. 386–395.   
[23] Adam Paszke et al. “Pytorch: An imperative style, high-performance deep learning library”. In: Advances in neural information processing systems 32 (2019).

[24] Charles R Qi et al. “Pointnet: Deep learning on point sets for 3d classification and segmentation”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, pp. 652–660.   
[25] Javier Romero, Dimitrios Tzionas, and Michael J. Black. “Embodied hands: modeling and capturing hands and bodies together”. In: ACM Trans. Graph. 36.6 (2017), 245:1–245:17.   
[26] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. “U-net: Convolutional networks for biomedical image segmentation”. In: Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer. 2015, pp. 234–241.   
[27] Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural information processing systems 30 (2017).   
[28] Ruben Villegas et al. “Contact-aware retargeting of skinned motion”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021, pp. 9720–9729.   
[29] Ruben Villegas et al. “Neural kinematic networks for unsupervised motion retargetting”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018, pp. 8639–8648.   
[30] Haodong Zhang et al. “Semantics-aware Motion Retargeting with Vision-Language Models”. In: arXiv preprint arXiv:2312.01964 (2023).   
[31] He Zhang et al. “ManipNet: neural manipulation synthesis with a hand-object spatial representation”. In: ACM Trans. Graph. 40.4 (2021), 121:1–121:14.   
[32] Jiaxu Zhang et al. “Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, pp. 13864–13872.   
[33] Keyang Zhou et al. “Toch: Spatio-temporal object-to-hand correspondence for motion refinement”. In: European Conference on Computer Vision. Springer. 2022, pp. 1–19.   
[34] Yi Zhou et al. “On the continuity of rotation representations in neural networks”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019, pp. 5745–5753.   
[35] Jun-Yan Zhu et al. “Unpaired image-to-image translation using cycle-consistent adversarial networks”. In: Proceedings of the IEEE international conference on computer vision. 2017, pp. 2223–2232.

# A Dataset Details

ScanRet details The primary motivation for collecting the ScanRet dataset stemmed from two main concerns. First, the data quality in the Mixamo [2] dataset was relatively low, suffering from significant issues such as interpenetration and contact mismatch. Second, the Mixamo dataset exclusively contained cartoon characters, whose body type distributions differed markedly from those of real human motion capture actors. In response, we developed the ScanRet dataset. We recruited 100 participants, evenly split between males and females, representing common ranges of height and BMI. Each participant underwent a 3D scan to create a T-pose mesh. We intentionally did not collect texture information for the body or face to protect privacy. Subsequently, we used motion capture equipment to build a library of 83 actions characterized by extensive physical contact. We enlisted human animators to map each action onto the 100 T-pose meshes, ensuring both semantic integrity and correct physical contact were maintained. All participants and animators received fair compensation. After discarding some invalid data, we compiled a total of 8,298 motion data entries. The ScanRet dataset is designed to simulate data obtained from real human motion capture, such as the MoSh [20, 17] algorithm, thus enhancing the realism of our evaluation process in the context of actual animation production workflows.

![](images/c93c99c70690dded93031e1ef61c27f24f23a1e3ecba1cc05cdf7a7c2e483e32.jpg)

<details>
<summary>natural_image</summary>

Two 3D-rendered male figures in dynamic poses, one standing and one wearing a purple helmet with 'MILIPACARELLA' text (no other text or symbols)
</details>

![](images/f849f6b7cc20915f7c253711aa6f78da33f011c78bdb08514b72eff8741b7386.jpg)

<details>
<summary>natural_image</summary>

Two 3D human figures in standing poses, one with hands clasped, the other with hands clasped (no text or symbols)
</details>

Figure 6: Left: Characters of varying body types in the Mixamo dataset do not always maintain reasonable hand contact during clapping actions. Right: In our ScanRet dataset, characters of diverse body types consistently maintain appropriate hand contact while performing the same clapping actions.

Data splits We collected motion data for 13 characters from the Mixamo website, totaling 3,675 motion sequences, with each character having approximately the same number of sequences. The characters are: Aj, Amy, Kaya, Mousey, Ortiz, Remy, Sporty Granny, Swat, The Boss, Timmy, X Bot, and Y Bot. Among them, Ortiz, Kaya, X Bot, and Amy were not encountered by the network during training. Overall, our training set included motion data for 9 Mixamo characters and 90 randomly selected characters from the ScanNet dataset, where 90% of the motion sequences was randomly chosen from both datasets. Details regarding the train/test split for specific motion sequences and characters are provided in the code.

# B Evaluation metric details

We evaluate the performance of our method from three perspectives: joint accuracy, contact preservation, and geometric interpenetration. In terms of joint accuracy, we calculate the Mean Squared Error (MSE) between the ground-truth joint positions $X_{gt}$ and the retargeted joint positions $\hat{X}$ , normalized by the character's height $h$ :

$$
M S E = \frac {1}{h} | | X _ {g t} - \hat {X} | | _ {2} ^ {2} \tag {10}
$$

Previous work [32] assessing the accuracy of self-contact measurements merely utilized the distance between hand vertices and the body surface to determine contact presence. Such experimental metrics fail to accurately reflect the precision of the contact location. Therefore, we adopted a metric similar to the vertex contact mean squared error (MSE) proposed by Villegas et al. [28], termed “Contact Error”. Specifically, we first identified sensor pairs where the distance between hand and body sensors

in the source action was less than the arm's diameter $d_{src}$ . We then located the same sensor pairs in the retargeted motion. If the distance between these sensor pairs in the retargeted motion exceeded that in the source action, we calculated the MSE of the distance differences; otherwise, the contact error was zero. The formula is as follows:

$$
\text { Contact   Error } = \left\{ \begin{array}{l} \left(\left| \left| \frac {\mathbf {d} _ {\mathrm{A}} ^ {t , k , l}}{R _ {\mathrm{A}}} \right| \right| _ {2} - \left| \left| \frac {\hat {\mathbf {d}} _ {\mathrm{B}} ^ {t , k , l}}{R _ {\mathrm{B}}} \right| \right| _ {2}\right) ^ {2}, \quad \text { if } \left| \left| \frac {\mathbf {d} _ {\mathrm{A}} ^ {t , k , l}}{R _ {\mathrm{A}}} \right| \right| _ {2} > \left| \left| \frac {\hat {\mathbf {d}} _ {\mathrm{B}} ^ {t , k , l}}{R _ {\mathrm{B}}} \right| \right| _ {2} \\ 0, \quad \text { otherwise }, \end{array} \right. \tag {11}
$$

where $\mathbf{d}_{\mathrm{A}}^{t,k,l}$ indicates the contact sensor pairs with $||\mathbf{d}_{\mathrm{A}}^{t,k,l}||_2 < d_{src}$ , while $R_{\mathrm{A}}$ and $R_{\mathrm{B}}$ represent the radius of each character's arms.

For geometric interpenetration, we assess the percentage of interpenetration, calculated as the ratio of penetrated vertices to the total vertices per frame. A lower ratio signifies reduced interpenetration. In our evaluation, we calculate the interpenetration ratio between arms (including hands) and the body.

$$
\text { Penetration } = \frac {\text { Number   of   penetrated   arm   vertices }}{\text { Total   number   of   arm   vertices }}. \tag {12}
$$

# C Implementation Details

SCS details As introduced in Section 3.2, we establish semantic correspondences between character meshes with different topologies using semantically consistent sensors. Specifically, given the semantic coordinates $(b, l, \phi)$ of a sensor, we can identify semantically consistent sensor positions on the meshes of different roles and obtain the feature vectors of the sensors. This process is detailed in Algorithm 1.

Algorithm 1: Derive Semantically Consistent Sensors from Semantic Coordinate   
Input: Mesh O, joint locations $J \in R^{N \times 3}$ , bone index $b \in \{0, 1, \cdots, N\}$ , origin parameter $l \in [0, 1)$ , direction parameter $\phi \in [0, 2\pi)$ Output: Sensor feature $s \in R^{4 \times 3}$ $i_{parent} \leftarrow \text{bone\_parent\_joint}(b)$ , $i_{child} \leftarrow \text{bone\_child\_joint}(b)$ ; $x_{parent} \leftarrow J[i_{parent}], x_{child} \leftarrow J[i_{child}]$ ; $o \leftarrow (1 - l)x_{parent} + lx_{child}$ ; /* Ray origin */ $d_{forward} \leftarrow forward\_direction(O)$ ; /* Face forward direction */ $d_{bone} \leftarrow normalize(x_{child} - x_{parent})$ ; /* Bone unit direction vector */ $d_{other} \leftarrow d_{forward} \times d_{bone}$ ; $n \leftarrow \cos(\phi)d_{forward} + \sin(\phi)d_{other}$ ; /* Ray direction */ $B \leftarrow bone\_mesh(O, b)$ ; /* Bone associated mesh */ $r \leftarrow ray(o, n)$ ; $p \leftarrow ray\_mesh\_intersection(B, r)$ ;

if $p \neq \emptyset$ then $t \leftarrow tangent\_matrix(x_p, B)$ ; $s \leftarrow concat(p, t)$ ;

else $s \leftarrow 0$ ;

end

Network architecture The network architectures of both our DMI Encoder and Geometry Encoder resemble the structure of PointNet. However, since all our data is inherently situated within the canonical space, we have eliminated the T-Net from PointNet to reduce network complexity. Before being input into the encoder, sensor features pass through a sensor group embedding layer, which converts the bone index b into an 8-dimensional embedding vector. This embedding vector is updated during training. The Geometry Encoder consists of six PointNet layers with $D_{model}$ set at 256, and there is a distinct Geometry Encoder for the body, head, arms, and legs. The DMI Encoder comprises a per-sensor encoder and a per-frame encoder, each built with six PointNet layers, with each interaction pair having its own encoder. Specific interaction pairs include: [(Left Arm), (Right Arm, Head, Torso)], [(Right Arm), (Left Arm, Head, Torso)], [(Left Leg), (Right Leg, Torso)], and [(Right Leg), (Left Leg, Torso)]. The Motion Encoder is a multilayer perceptron (MLP). Both the

![](images/5f607a88c26105c739c6d4b45576e6d9e17d5786f4894561e94d830646281f10.jpg)

<details>
<summary>text_image</summary>

Instructions
1. Watch the Three Video:
Source Video: The video shows the original motion sequence.
Source Video: The video features a different character mirroring the motion from the Source Video.
2. Compare Video A and Video B
3. Sound to your content, please answer the following questions:
See Section Preferences (Which video better matches the source motion in terms of the overall meaning and intent of the motion?
Notice Details: Which video has more accurate and detailed motion? Look to less self-interpretation and better self-contact precision.
Overall Quality: Considering all factors, which video do you think is better overall?
Question
1. Which video better matches the semantics of the source motion?
Video A
Video B
2. Which video has less self-interpretation and better self-contact precision?
Video A
Video B
3. Which video do you think is better overall?
Video A
Video B
Video A
Video B
Video B
Video B
Subtest
</details>

Figure 7: User interface presented to participants during the user study.

Transformer Encoder and Transformer Decoder have eight layers, with the number of heads set to four and the feed-forward size to 256. Between the Transformer Encoder and Transformer Decoder, we employ an alignment mask proposed by Fan et al. [6], which ensures that each frame feature in the decoder attends only to the corresponding DMI frame and initial token, thereby aligning the network's output motion sequence with the input features.

Training details We implemented our network using PyTorch [23], running on a machine equipped with an NVIDIA RTX A6000 GPU and an AMD EPYC 9654 CPU. The dataset was uniformly processed at a frame rate of 30 fps. During training, we randomly clipped a sequence of 30 frames from the dataset. The target character was set to be the same as the source character with a $50\%$ probability, and different with a $50\%$ probability, selected randomly from the dataset. On our system, training for 36 epochs required approximately 40 hours. During inference, our MeshRet model can achieve performance exceeding 30 fps.

# D User study details

We recruited participants via the Amazon Mechanical Turk [3] platform to partake in a user study. As shown in Figure 7, during each session, subjects were presented with one source video and two retargeted motion videos: Video A and Video B. Participants were asked to watch all three videos and then compare Video A and Video B. At the conclusion of the viewing, they were requested to answer the following three questions:

1. Which video better matches the source motion in terms of the overall meaning and intent of the motion?

2. Which video has more accurate and detailed motion? Look for less self-interpenetration and better self-contact precision.   
3. Considering all factors, which video do you think is better overall?

For each question answered, participants received a compensation of \$0.04. We collected 600 comparison results in the end.

# E Additional results

![](images/c1faccd6f9c1c3a7e2968bf58b82a5f3e62afb2eab017f22775d9b138fe45e2b.jpg)  
Figure 8: Left: We visualized three consecutive frames within an motion sequence. It is evident that while there was no jitter in the motion source, significant jitter occurred in the t-th frame of the $R^{2}ET$ [32] results, which was not the case with our method. Right: We visualize the corresponding right-hand height for this segment of the sequence. The results indicate that the jitter in the $R^{2}ET$ output was pronounced.

Motion jitter comparison To better illustrate the jitter issue present in the results from the $R^{2}ET$ [32] method, we visualized consecutive frames generated by $R^{2}ET$ and our method in Figure 8, and provided a line graph depicting the variations in height of the right-hand joint over time. These results demonstrate that $R^{2}ET$ is adversely affected by contradictions between skeletal retargeting and geometry correction phases, leading to significant motion jitter. In contrast, our method successfully avoids this problem.

![](images/f799ab3c777cd833c64906c25d51191c7afab2bfebe69248eed4a64a6fdf9eb4.jpg)  
Figure 9: Qualitative comparison with Zhang et al. [30].

Qualitative comparison with Zhang et al. [30] Since Zhang et al. [30] did not open-source their code, we were unable to conduct a complete and fair comparison of their method with ours in our experiments. However, we endeavored to locate several examples presented in their paper and applied our MeshRet to the same motion sequences. The comparative results are displayed in Figure 9. As observed in these examples, our method maintains the semantic integrity of the source motions, and it performs better in the Fireball case (the second motion sequence shown). This indicates that our method can achieve, and even surpass, the performance of their approach.

Metrics across different data splits Tables 3 and 4 present the contact error and penetration ratio of our method compared to the baseline method across four different data splits. A consistent pattern observed is that performance improves for seen characters or motions. It is evident that our method outperforms the baseline across all data splits.

Ablation studies on ratios of proximal sensor pairs The full approach can be considered a mixed version of Ours $_{far}$ and Ours $_{cls}$ , utilizing an equal distribution of proximal and distal sensor pairs. To

Table 3: Contact errors of MeshRet and baselines across all data splits on Mixamo+. 

<table><tr><td colspan="2">Metric</td><td colspan="3">Contact Error↓</td></tr><tr><td>Data Split</td><td>UC+UM</td><td>SC+UM</td><td>UC+SM</td><td>SC+SM</td></tr><tr><td>Copy</td><td>1.462</td><td>1.188</td><td>2.477</td><td>1.682</td></tr><tr><td>PMnet [15]</td><td>1.826</td><td>1.774</td><td>4.134</td><td>3.132</td></tr><tr><td>SAN [1]</td><td>1.416</td><td>1.181</td><td>4.229</td><td>2.902</td></tr><tr><td>R2ET [32]</td><td>1.653</td><td>1.498</td><td>3.372</td><td>2.314</td></tr><tr><td>Ours</td><td>0.573</td><td>0.837</td><td>1.248</td><td>0.432</td></tr></table>

Table 4: Penetration ratios of MeshRet and baselines across all data splits on Mixamo+. 

<table><tr><td>Metric</td><td colspan="4">Penetration(%)↓</td></tr><tr><td>Data Split</td><td>UC+UM</td><td>SC+UM</td><td>UC+SM</td><td>SC+SM</td></tr><tr><td>Copy</td><td>1.57</td><td>4.16</td><td>5.78</td><td>9.56</td></tr><tr><td>PMnet [15]</td><td>1.43</td><td>4.20</td><td>5.71</td><td>9.56</td></tr><tr><td>SAN [1]</td><td>1.81</td><td>5.52</td><td>4.66</td><td>7.81</td></tr><tr><td>R2ET [32]</td><td>1.54</td><td>4.66</td><td>4.92</td><td>5.71</td></tr><tr><td>Ours</td><td>1.55</td><td>2.63</td><td>4.60</td><td>5.04</td></tr></table>

better illustrate this balance, we provide additional experimental results by testing different ratios of proximal to distal sensor pairs. Table 5 compares our method's performance with varying percentages of proximal sensor pairs under the Mixamo+ setting. As the percentage of proximal sensor pairs decreases, the interpenetration ratio fluctuates mildly, while the contact error initially decreases and then increases. Finally, with no proximal pairs (equivalent to the "far" version), the performance drops significantly. In Figure 10, we present a qualitative comparison of our methods using different proximal sensor pair ratios. Except for the $100\%$ Proximal version (equivalent to $\mathrm{Ours}_{cls}$ ) and the $0\%$ Proximal version (equivalent to $\mathrm{Ours}_{far}$ ), our method demonstrates fair robustness to the proximal sensor ratio in the $25\% - 75\%$ interval. Based on these results, we conclude that choosing $50\%$ proximal sensor pairs strikes a reasonable balance for achieving good performance.

Table 5: Quantitative comparison between our methods with varying percentages of proximal sensor pairs under the Mixamo+ setting. 

<table><tr><td>Method</td><td>Contact Error↓</td><td>Penetration(%)↓</td></tr><tr><td>100% Proximal Pairs</td><td>0.800</td><td>3.35</td></tr><tr><td>75% Proximal Pairs</td><td>0.909</td><td>3.61</td></tr><tr><td>50% Proximal Pairs</td><td>0.772</td><td>3.45</td></tr><tr><td>25% Proximal Pairs</td><td>0.781</td><td>3.29</td></tr><tr><td>0% Proximal Pairs</td><td>1.642</td><td>5.37</td></tr></table>

Ablation studies on different sensor arrangements We conducted further ablation studies on different sensor arrangements. Specifically, we evaluated the performance of a model trained with half the sample points in the $\phi$ space in SCS, denoted as Ours $_{\phi}$ , and another model trained with half the sample points in the l space in SCS, referred to as Ours $_{l}$ . As shown in Table 6, Ours $_{\phi}$ compromises the interpenetration ratio, indicating that sufficient sample points in the space are crucial for avoiding interpenetration. We also found that both models introduce artifacts; please refer to Figure 11.

Failure cases with noisy inputs We provide results with clean and noisy inputs in Figure12 and Figure13. The results of MeshRet exhibit interpenetration with noisy inputs.

![](images/31c6e56aaecfaf79232c2ed3fc5503bac426f3e4319ade2afbcedb2c3b2426a6.jpg)

<details>
<summary>text_image</summary>

Source 100% Proximal 75% Proximal 50% Proximal 25% Proximal 0% Proximal
</details>

![](images/90af6d138a5d87a8e5accb292cb3de346074d68fbfe22c21c60150eb48c02dbb.jpg)

<details>
<summary>other</summary>

| Source | 100% Proximal | 75% Proximal | 50% Proximal | 25% Proximal | 0% Proximal |
|--------|---------------|--------------|--------------|--------------|-------------|
| Proximal | 100% | 75% | 50% | 25% | 0% |
</details>

Figure 10: Qualitative results with different proximal sensor pair ratios.

Table 6: Quantitative comparison between methods with different sensor arrangements. 

<table><tr><td>Metric</td><td>MSE↓</td><td>MSE $^{lc}$ ↓</td><td colspan="2">Contact Error↓</td><td colspan="2">Penetration(%)↓</td></tr><tr><td>Dataset</td><td>ScanRet</td><td>ScanRet</td><td>Mixamo+</td><td>ScanRet</td><td>Mixamo+</td><td>ScanRet</td></tr><tr><td>Ours $_{\phi}$ </td><td>0.052</td><td>0.011</td><td>0.793</td><td>0.410</td><td>3.90</td><td>1.60</td></tr><tr><td>Ours $_{l}$ </td><td>0.049</td><td>0.010</td><td>0.805</td><td>0.293</td><td>3.46</td><td>1.56</td></tr><tr><td>Ours</td><td>0.047</td><td>0.009</td><td>0.772</td><td>0.284</td><td>3.45</td><td>1.59</td></tr></table>

![](images/744548787ac8c8b107d903ecb2dae9a86958f2b51387f1dc81504b31a3e6c246.jpg)

<details>
<summary>text_image</summary>

Source
Ours
Oursφ
Oursl
</details>

![](images/16c660ac68aea013615f74bb53f06612c1ffdb6c9c8454b2f0618463bb14d345.jpg)

<details>
<summary>text_image</summary>

Source
Ours
Oursφ
Oursl
</details>

Figure 11: Qualitative comparison of additional ablation studies on sensor arrangements. The red rectangles identify artifacts introduced by different sensor arrangements.

![](images/56b64ef4164dbf75e78cc2d658c504519adf3ac2b7faabe34e6b62f827532d47.jpg)

<details>
<summary>text_image</summary>

Clean
Source
Clean
Result
Noisy
Source
Noisy
Result
</details>

![](images/25120a3540e251d325391d7bd8bec5be27642c4115b468ab1abede215c207446.jpg)

<details>
<summary>text_image</summary>

Clean Source
Clean Result
Noisy Source
Noisy Result
</details>

Figure 12: Qualitative results on the Mixamo dataset with clean and noisy inputs. A red rectangle indicates interpenetration.

![](images/20f5305834adcbf2202c18aedee8678abce66b670fc9afb7d61231d4f8c65e1c.jpg)

<details>
<summary>natural_image</summary>

Four 3D human figures in dynamic poses, one highlighted with a red dashed box (no text or symbols)
</details>

Clean   
Source

![](images/5db81cb907810b6c86911889cdabec6247888fac062808c8d33410b20358f400.jpg)  
Clean   
Result

![](images/84628ba305ad17e0e4615896e59bceeb8fe3b738ef60e0caba249efa78e12a0e.jpg)  
Noisy   
Source

![](images/b700d4940bb9ca38d69b6de89ef3ee16ce394a7a1c3259faf26134d94460563d.jpg)  
Noisy   
Result

![](images/91a62ffe470550ee37da0b2c84c3a6623e215cdff8babb45b0da2b741e821c3d.jpg)  
Clean   
Source

![](images/f9ea3d0a47ce4a6c8f4844aa8eecf7a51bd00d3803c2f051d18f0a769313b006.jpg)  
Clean   
Result

![](images/19330951bbc2cecb8ea8d9f0895b1c65f75402a18d93c8897c25bcbd446a10f3.jpg)  
Noisy   
Source

![](images/1581e556e7a6ad144adbf60c904b36eee246fd15ce63ef4b7d1c93ce4227c627.jpg)  
Noisy   
Result   
Figure 13: Qualitative results on the Mixamo dataset with ScanRet characters as targets. A red rectangle indicates interpenetration.

More cases We present additional cases to validate the effectiveness of our MeshRet. Figures 14, 15, 16, and 17 depict four motion sequences retargeted from the source character to distinct target characters. These examples illustrate that our MeshRet is capable of generating high-quality motion sequences on target characters with diverse body shapes.

![](images/2f51ff7045616ca5d2eea9d78961ef824bd65cbe0b5dcc47b0f547eb3d54ca98.jpg)

<details>
<summary>natural_image</summary>

3D-rendered human figure poses in various dynamic poses, labeled Source, Target 1, Target 2, and Target 3 (no text or symbols on the figures themselves)
</details>

Figure 14: Snapshots of motion sequence 4 in ScanRet, retargeted from the source character to three distinct characters.

![](images/8b0941c91628b5b30a56c0664aa1df9ef1890fe564b82d4abe76c1c4cae19af5.jpg)

<details>
<summary>text_image</summary>

Source
Target 1
Target 2
Target 3
</details>

Figure 15: Snapshots of motion sequence 43 in ScanRet, retargeted from the source character to three distinct characters.

![](images/5d746e610fa2a7ea3d6262656d5729f5bcc000fb2eac4ab32695b0c0d5d46f1f.jpg)

<details>
<summary>natural_image</summary>

Sequence of 3D-rendered human figures in various poses and poses, labeled Source, Target 1, Target 2, and Target 3 (no text or symbols on the figures themselves)
</details>

Figure 16: Snapshots of motion sequence 9 in ScanRet, retargeted from the source character to three distinct characters.

![](images/3a754a66ea5bb44e971cb11ff2224761a5361ed89dd2ad31592ebbce98b8b38b.jpg)

<details>
<summary>natural_image</summary>

3D-rendered character poses in a grid, showing multiple poses from Source to Target 3 (no text or symbols on the figures themselves)
</details>

Figure 17: Snapshots of motion sequence 45 in ScanRet, retargeted from the source character to three distinct characters.

# F Broader impacts

Our work can provide animation professionals with enhanced results in motion retargeting, thereby alleviating their workload and increasing productivity in fields such as virtual reality, game development, and animation production. Regarding potential negative social impacts, we believe the likelihood of misuse of our work is minimal. This is because our work is situated in the midstream phase of the animation production pipeline, whereas privacy-invading forgeries, such as DeepFake, primarily occur during the downstream rendering phase.