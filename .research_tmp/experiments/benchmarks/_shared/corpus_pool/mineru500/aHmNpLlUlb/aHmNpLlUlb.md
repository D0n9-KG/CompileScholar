# INSERTNERF: INSTILLING GENERALIZABILITY INTO NERF WITH HYPERNET MODULES

Yanqi Bao $^{1}$ , Tianyu Ding $^{2}$ , Jing Huo $^{1*}$ , Wenbin Li $^{1}$ , Yuxin Li $^{1}$ , Yang Gao $^{1}$

$^{1}$ State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China
yq\_bao@smail.nju.edu.cn, {huojing, liwenbin, gaoy}@nju.edu.cn,
liyuxin16@smail.nju.edu.cn $^{2}$ Applied Sciences Group, Microsoft Corporation, Redmond, USA
tianyuding@microsoft.com

# ABSTRACT

Generalizing Neural Radiance Fields (NeRF) to new scenes is a significant challenge that existing approaches struggle to address without extensive modifications to vanilla NeRF framework. We introduce InsertNeRF, a method for INStilling gEneRalizabiliTy into NeRF. By utilizing multiple plug-and-play HyperNet modules, InsertNeRF dynamically tailors NeRF's weights to specific reference scenes, transforming multi-scale sampling-aware features into scene-specific representations. This novel design allows for more accurate and efficient representations of complex appearances and geometries. Experiments show that this method not only achieves superior generalization performance but also provides a flexible pathway for integration with other NeRF-like systems, even in sparse input settings. Code will be available at: https://github.com/bbbbby-99/InsertNeRF.

# 1 INTRODUCTION

Novel view synthesis, a fundamental discipline in computer vision and graphics, aspires to create photorealistic images from reference inputs. Early works (Debevec et al., 1996; Lin & Shum, 2004) primarily focused on developing explicit representations, facing challenges due to the absence of 3D supervision. This issue has been alleviated by recent advancements in implicit neural representation research, which have led to improved performance. In particular, Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) has attracted significant interest. NeRF, and its derivative works, extract scene-specific implicit representations through overfitting training on posed scene images. Although NeRF uses neural scene representations effectively to yield realistic images, the scene-specific nature of these representations requires retraining when faced with novel scenarios.

An emerging topic known as Generalizable NeRF (GNeRF) has recently garnered considerable attention for this challenge. GNeRF aims to learn a scene-independent inference approach that facilitates the transition from references to target view. Current methods enhance the NeRF architecture by adding structures that aggregate reference-image features, or reference features. Examples include pixel-wise feature cost volumes (Johari et al., 2022), transformers (Wang et al., 2022; Suhail et al., 2022), and 3D visibility predictors (Liu et al., 2022). However, fitting these additions into conventional NeRF-like frameworks such as mip-NeRF (Barron et al., 2021), NeRF++ Zhang et al. (2020), and others, often proves challenging and may fail to effectively harness the guiding potential of reference features. Furthermore, the extensive use of transformers or cost volumes can be time-consuming. Thus, an intriguing question arises: Is it possible to directly INStill gEneRalizabiliTy into NeRF (InsertNeRF) while staying faithful to the original framework?

A straightforward way to accomplish this goal is to adaptively modify the NeRF network's weights, or implicit representations, for different reference scenes while preserving the original framework. The concept of hypernetwork (Ha et al., 2016), which conditionally parameterizes a target network, is an effective strategy in this scenario. The features extracted from the reference scene can be used as inputs to generate scene-specific network weights. However, initial experiments in Tab. 2 indicate that constructing a hypernetwork directly based on the NeRF framework can be inadequate, and often fails to predict different attributes like emitted color and volume density. To address this, we

![](images/399fbe2e24fa50ec72cc4a585e3dcdad29b7c74791e9df939785f43bb610c47e.jpg)

<details>
<summary>text_image</summary>

NeRF
PSNR: 7.29
mip-NeRF
Full Res.
PSNR: 13.33
1/8 Res.
1/4 Res.
1/2 Res.
NeRF++
PSNR: 10.18
Background
Fore+Background
Exiting GNeRF works (Color-Depth)
InsertNeRF
PSNR: 26.44
Insert-mip-NeRF
Full Res.
PSNR: 30.62
1/8 Res.
1/4 Res.
1/2 Res.
Insert-NeRF++
PSNR: 24.06
Background
Fore+Background
Our InsertNeRF (Color-Depth)
(a)
(b)
</details>

Figure 1: Overview of motivation. (a) We instill generalizability into NeRF-like systems, including vanilla NeRF, mip-NeRF, and NeRF++ frameworks, to achieve consistent performance across scenes without modifying the base framework or requiring scene-specific retraining. (b) InsertNeRF significantly improves depth estimation compared to its original counterpart.

propose to use HyperNet modules, which are designed to serve as easily integrable additions to existing NeRF-like frameworks. Owing to their flexibility, the resulting InsertNeRF excels at predicting the NeRF attributes by capitalizing on sampling-aware features and various module structures.

In InsertNeRF, we insert multiple HyperNet modules to instill generalizability throughout the framework's progression. This approach allows us to fully leverage the guiding role of scene features in determining the entire network's weights. Unlike existing works that solely utilize reference features as inputs, InsertNeRF exhibits a thorough grasp of reference scene knowledge. To further unlock the full potential of the HyperNet modules, it is crucial to aggregate scene features from a set of nearby reference images. To achieve this, we introduce a multi-layer dynamic-static aggregation strategy. Compared to existing works, it not only harnesses the inherent completion capabilities of global features, but it also implicitly models occlusion through dynamic-static weights, as demonstrated on the depth renderings shown in Fig. 1b. By feeding the aggregated scene features into the HyperNet modules, we can generate scene-related weights based on the well-understood reference scene.

In summary, we make the following specific contributions:

- We introduce InsertNeRF, a novel paradigm that inserts multiple plug-and-play HyperNet modules into the NeRF-like framework, endowing NeRF-like systems with instilled generalizability.   
- We design two types of HyperNet module structures tailored to different NeRF attributes, aiming for predicting scene-specific weights derived from sampling-aware scene features. For these features, we further propose a multi-layer dynamic-static aggregation strategy, which models the views-occlusion and globally completes information based on the multi-view relationships.   
- We demonstrate that InsertNeRF achieves state-of-the-art performance with extensive generalization experiments by integrating the modules into the vanilla NeRF. Furthermore, we show the significant potential of our modules in various NeRF-like systems, such as mip-NeRF (Barron et al., 2021), NeRF++ (Zhang et al., 2020), as shown in Fig. 1a, and in task with sparse inputs.

# 2 RELATED WORKS

# 2.1 GENERALIZABLE NEURAL RADIANCE FIELDS

Neural Radiance Fields (NeRF) by (Mildenhall et al., 2021) and its subsequent derivatives (Barron et al., 2022; Isaac-Medina et al., 2023; Bao et al., 2023) have gained momentum and are capable of producing realistic images. However, a significant drawback is the need to retrain them for every new scene, which is not efficient in real-world applications. Recent works by (Wang et al., 2021; 2022) introduce Generalizable Neural Radiance Fields that can represent multiple scenes, regardless of whether they are in the training set. To achieve this, many studies have focused on understanding the relationships between reference views and refining NeRF's sampling-rendering mechanism. For instance, NeuRay (Liu et al., 2022) and GeoNeRF (Johari et al., 2022) use pre-generated depth maps or cost volumes as prior to alleviate occlusion issues. On the other hand, IBRNet (Wang et al., 2021) and GNT (Wang et al., 2022) implicitly capture these relationships through MLPs or transformers. Regarding the sampling-rendering process, most works (Xu et al., 2023; Suhail et al., 2022; Wang

et al., 2021; Zhu et al., 2023) utilize the transformer-based architectures to aggregate the sampling point features and replace traditional volume rendering with a learnable technique. However, a common limitation is that most of these methods replace NeRF's network with transformers, making it challenging to apply to NeRF derivatives and leading to increased computational complexity. Our research aims to address this by instilling generalizability into NeRF-like systems with scene-related weights while preserving its original framework and efficiency.

# 2.2 HYPERNETWORKS

The hypernetwork (Ha et al., 2016; Chauhan et al., 2023), often abbreviated as hypernet, is invented to generate weights for a target neural network. Unlike traditional networks that require training from scratch, hypernets offer enhanced generalization and flexibility by adaptively parameterizing the target network (Alaluf et al., 2022; Yang et al., 2022; Li et al., 2020). Leveraging these benefits, hypernets have found applications in various domains including few-shot learning (Li et al., 2020), continual learning (Von Oswald et al., 2019), computer vision (Alaluf et al., 2022), etc. In the realm of NeRF, there have been efforts to incorporate hypernets to inform the training of the rendering process. For instance, (Chiang et al., 2022) propose to train a hypernet using style image features for style transfer, while (Zimny et al., 2022) employ encoded point-cloud features for volume rendering. On a related note, (Peng et al., 2023) utilize a dynamic MLP mapping technique to create volumetric videos and (Kania et al., 2023) use a hypernet for 3D-aware NeRF GAN. In our work, instead of using the hypernet in NeRF framework directly, we introduce a plug-and-play HyperNet module, with a focus on providing reference scene knowledge to enable generalization to new scenarios.

# 3 METHOD

# 3.1 BACKGROUND

Neural Radiance Fields. Neural radiance fields (NeRF) (Mildenhall et al., 2021) is a neural representation of scenes. It employs MLPs to map a 3D location $x \in R^{3}$ and viewing direction $d \in S^{2}$ to an emitted color $c \in [0, 1]^{3}$ and a volume density $\sigma \in [0, \infty)$ , which can be formalized as:

$$
\mathcal {F} (\boldsymbol {x}, \boldsymbol {d}; \boldsymbol {\Theta}) \mapsto (\boldsymbol {c}, \sigma), \tag {1}
$$

where $\mathcal{F}$ is the MLPs, and $\Theta$ is the set of learnable parameters of NeRF. Note that $\mathcal{F}$ can be further split into an appearance part $\mathcal{F}_{app}$ and a geometry part $\mathcal{F}_{geo}$ for the view-dependent attribute $c$ and view-invariant attribute $\sigma$ , respectively (Zhang et al., 2023).

Volume Rendering. Given a ray in a NeRF, $\boldsymbol{r}(t) = \boldsymbol{o} + t\boldsymbol{d}$ , where o is the camera center and d is the ray's unit direction vector, we sample K points, $\{\boldsymbol{r}(t_{i}) | i = 1, ..., K\}$ , along the ray and predict their color values $c_{i}$ and volume densities $\sigma_{i}$ . The ray's color is then calculated by:

$$
\hat {C} (\boldsymbol {r}) = \sum_ {i = 1} ^ {K} w _ {i} \boldsymbol {c} _ {i}, \quad \text { where } \quad w _ {i} = \exp \left(- \sum_ {j = 1} ^ {i - 1} \sigma_ {j} \delta_ {j}\right) (1 - \exp (- \sigma_ {i} \delta_ {i}))  , \tag {2}
$$

where $\delta_{i}$ is the distance between adjacent samples, and $w_{i}$ is considered to be the hitting probability or the weight of the i-th sampling point (Liu et al., 2022).

Generalizable NeRF. Given N reference scene views with known camera poses $\{I_{n}, P_{n}\}_{n=1}^{N}$ , the goal of GNeRF is to synthesize a target novel view $I_{T}$ based on these reference views, even for scenes not observed in the training set, thereby achieving generalizability. Current works (Wang et al., 2021; Liu et al., 2022; Wang et al., 2022) primarily focus on aggregating features along with the ray $r(t)$ from multiple reference views. The overall process can be outlined as:

$$
\mathcal {F} _ {\text { sample }} \left(\left\{\mathcal {F} _ {\text { view }} \left(\left\{\boldsymbol {F} _ {n} \left(\Pi_ {n} (\boldsymbol {r} (t _ {i}))\right) \right\} _ {n = 1} ^ {N}\right) \right\} _ {i = 1} ^ {K}\right) \mapsto (\boldsymbol {c}, \sigma). \tag {3}
$$

Here, $\Pi_{n}(\boldsymbol{x})$ projects x onto $I_{n}$ , and $F_{n}(\boldsymbol{z})$ queries the corresponding feature vectors according to the projected points in n-th reference. $F_{view}$ and $F_{sample}$ specifically denote the aggregation of multi-view features and the accumulation of multiple sampling point features along the ray. These aggregations are often carried out using common techniques such as MLPs and transformers.

![](images/5ace3a83209e7542b8e248c0b1ec374a40cfa43deafe62835d5cbdebc0fd28ad.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Novel Target View"] -->|Sampling| B["(x, d)"]
    B --> C["F_geo"]
    C --> D["HyperNet Modules"]
    D --> E["σ"]
    E --> F["F_app"]
    F --> G["HyperNet Modules"]
    G --> H["→"]
    H --> I["L_MSE Photometric Loss"]
    I --> J["Volume Rendering"]
    J --> K["C"]
    K --> L["w"]
    L --> M["(a) InsertNeRF"]
    M --> N["Vanilla NeRF"]
```
</details>

![](images/f83fe1f315a2d60b0c23bc6c4c3cc2c0e1a9bc4ad5ca4082087fa5ff86325c79.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Reference Images × N"] --> B["Additional Decoder"]
    B --> C["L_backbone Loss"]
    C --> D["U-Net"]
    D --> E["L_DY Loss"]
    E --> F["Static-Weight"]
    F --> G["Multi-Layer Dynamic-Weights"]
    G --> H["(x, d)"]
    H --> I["Sampling Pose Project II"]
    I --> J["(d) HyperNet Modules"]
    J --> K["HyperfNet Modules"]
    K --> L["HyperfNet Modules"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#cff,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#cfc,stroke:#333
```
</details>

Figure 2: Overview of InsertNeRF. (a) Within the NeRF framework, two types of HyperNet modules are inserted into $F_{geo}$ and $F_{app}$ . The HyperNet modules begin by (b) extracting features among multiple (N) reference images, and (c) using a multi-layer dynamic-static aggregation strategy to aggregate the scene representations. Based on these scene representations and specially designed sampling-aware filters, (d) we develop dynamic MLPs and activation functions to guide the weights and instill generalizability into vanilla NeRF. Finally, (a) standard volume rendering is performed.

# 3.2 INSERTNERF

We introduce InsertNeRF, a novel paradigm that instills generalizability into the NeRF framework, as illustrated in Fig. 2. While this method can be adapted to a variety of NeRF-based systems (Sec. 4.4), we focus on its application on the vanilla NeRF in this section.

Overview. InsertNeRF achieves generalizability by inserting multiple HyperNet modules into NeRF. These modules dynamically generate scene-specific weights for NeRF that are tailored to specific reference scene, denoted by $\Omega_T$ . Based on it, by incorporating $\Omega_T$ into Eq. (1) and combining $\Theta$ as well as $\Omega_T$ , NeRF's implicit representation (or weights) gains the generalizability across multi-scenes, which is explained in Appendix D.3. Specifically, it can be described as follows:

$$
\mathcal {F} (\pmb {x}, \pmb {d}; \pmb {\Theta}, \pmb {\Omega} _ {T}) \mapsto (\pmb {c}, w),
$$

$$
\text { where } \boldsymbol {\Omega} _ {T} = \text { HyperNet } \left(\left\{\mathcal {F} _ {\text { view }} \left(\left\{\boldsymbol {F} _ {n} \left(\Pi_ {n} (\boldsymbol {r} (t _ {i}))\right) \right\} _ {n = 1} ^ {N}\right) \right\} _ {i = 1} ^ {K}\right). \tag {4}
$$

Comparing Eq. (4) to Eq. (3), the key to InsertNeRF is the newly introduced architectures with dynamic weights $\Omega_{T}$ , guided by the HyperNet modules based on specific reference inputs. The process begins with reference features extraction (Sec. 3.2.1), then a multi-layer dynamic-static aggregation strategy is employed to fuse reference features from multi-views into scene features (Sec. 3.2.2). Subsequently, these aggregated scene features are used to adaptively generate NeRF's sampling-aware weights via the HyperNet modules, which consist of sampling-aware filters, dynamic MLPs and dynamic activation functions (Sec. 3.2.3). These novel HyperNet modules are inserted before each MLP layer in the original NeRF, serving as an enhancement to the original MLP layers.

A notable aspect of InsertNeRF is its ability to directly calculate the hitting probability $w_{i}$ in Eq. (2) for volume rendering, rather than simply outputting the volume density from $F_{geo}$ . This capability stems from the implicit modeling of the relationships between spatial points and the advantage of using multi-scale features. By combining $F_{geo}$ with $F_{app}$ , the entire pipeline is trained end-to-end. Our unique design not only leads to superior rendering performance in GNeRF but also offers improved computational efficiency compared to transformer-based structures (Wang et al., 2022).

# 3.2.1 REFERENCE FEATURES EXTRACTION

In the exploration of reference images, generalizable methods often combine U-Net (Ronneberger et al., 2015) and ResNet (He et al., 2016) to extract local dense feature maps. These have proven effective in dealing with occlusion problems (Liu et al., 2022). Yet, there is a risk that an overemphasis on local dense features might neglect global features, which are key to occlusion completion and global inference (Iizuka et al., 2017). In our work, we take advantage of the spatial representation capabilities of multi-scale features to model complex geometry and detailed appearance. Specifically, we bring in global-local features to successively update the Hypernet module's weights for $\mathcal{F}_{geo}$ and dense feature for $\mathcal{F}_{app}$ . Here, geometry requires multi-scale information to deduce occluded portions, while appearance concentrates on dense fine-grained details. This process begins with multi-scale features $F_{l,n}$ from U-Net for each reference input $I_n$ , and can be expressed as:

$$
\boldsymbol {F} _ {l, n} \in \mathbb {R} ^ {\frac {W}{2 ^ {l + 1}} \times \frac {H}{2 ^ {l + 1}} \times C _ {l}}, \quad l = 2, 1, 0; n = 1, \dots , N. \tag {5}
$$

Here, $W \times H$ defines the image resolution, and $C_l$ is the number of channels. During feature upsampling (as $l$ decreases), we output each layer's features, transitioning from global to local.

# 3.2.2 MULTI-LAYER DYNAMIC-STATIC AGGREGATION STRATEGY

Following the feature extraction, the next essential step is the aggregation of scene features. This is not only foundational for scene generalizability but also significantly impacts the effectiveness of the HyperNet modules. Most existing techniques focus primarily on preserving local geometry and appearance consistency, often employing visibility to model occlusions. A straightforward approach is to deduce the view-weight based on differences between reference and target views (Wang et al., 2021). We refer to it as static weight, denoted by $M^{ST} \in R^{B \times K \times N}$ , where B represents the batch size, and it assigns higher weights to closer views in a fixed manner. However, it may be unreliable as it overlooks the correlation among the features. To remedy this, we introduce a dynamic prediction of multi-layer weights based on multi-scale features, involving a blend of Maxpool-MLPs and Softmax layers, termed dynamic weights and denoted by $M_{l}^{DY} \in R^{B \times K \times N}$ . Our approach hence adopts a dynamic-static aggregation strategy for more nuanced multi-view scene feature aggregation.

Formally, given the corresponding features $F_{l} \in R^{B \times K \times N \times d_{l}}$ of $B \times K$ points in the space, where $d_{l}$ is the latent feature dimension, we calculate the weighted means and variances as $\mu_{l} = E_{n} \left[ F_{l} \odot M_{l}^{DY} \right] \in R^{B \times K \times d_{l}}$ and $v_{l} = V_{n} \left[ F_{l} \odot M_{l}^{DY} \right] \in R^{B \times K \times d_{l}}$ , respectively. After concatenating $F_{l}$ for each reference view with $\mu_{l}$ and $v_{l}$ and halvely projecting its dimension, denoted as $\widetilde{F}_{l} \in R^{B \times K \times N \times d_{l}/2}$ , it is applied to the static weight to obtain $\widetilde{\mu}_{l} = E_{n} \left[ \widetilde{F}_{l} \odot M^{ST} \right] \in R^{B \times K \times d_{l}/2}$ and $\widetilde{v}_{l} = V_{n} \left[ \widetilde{F}_{l} \odot M^{ST} \right] \in R^{B \times K \times d_{l}/2}$ . With $F_{l}^{\max} \in R^{B \times K \times d_{l}}$ representing the maximum features among all the reference views, and by concatenating $\widetilde{\mu}_{l}$ and $\widetilde{v}_{l}$ , and adding it to $F_{l}^{\max}$ , we accomplish the feature aggregation phase $F_{view}$ in Eq. (4). $^{1}$

The use of global-local dynamic weights leads to a significant enhancement in edge sharpness and the thorough completion of detail in the depth rendering images, as evidenced in Fig. 1b. Note that unlike static weights, dynamic weights are guided by the relationships between multi-scale reference features and are learned with auxiliary supervision (Sec. 3.3).

# 3.2.3 HYPERNET MODULES

We now turn our attention to the HyperNet modules, the core element of InsertNeRF, integrated within both $F_{geo}$ and $F_{app}$ . These modules are composed of three basic components: sampling-aware filters, dynamic MLPs (D-MLP), and dynamic activation functions.

Sampling-aware Filter. Unlike traditional hypernetworks, where reference features are generally stable, those based on pose-related epipolar geometric constraints in GNeRF are noisy. This noise complicates their direct use for weights generation. To address this challenge, we introduce a sampling-aware filter that seeks to implicitly find correlations between inter-samples and reduce noise within the reference features through graph reasoning. Specifically, following the aggregation phase $F_{view}$ , each aggregated point-feature is regarded as a node within a graph structure. The

relationships between these K points are then modeled using graph convolutions, formulated as:

$$
\boldsymbol {H} _ {l} = \left(\boldsymbol {I} - \boldsymbol {A} _ {l}\right) \boldsymbol {F} _ {\text { view }} \boldsymbol {W} _ {l} ^ {a}, \tag {6}
$$

where $F_{view} \in R^{B \times K \times d_l}$ denotes the aggregated K point-features after $F_{view}$ , and $A_l$ and $W_l^a$ represent the $K \times K$ node adjacency matrix and the learnable state update function, respectively. I here denotes the identity matrix. This specific graph structure helps filter out noise by state-updating, enabling the network to concentrate on key features more effectively. Additionally, for intricate tiny structures, we adopt an approach inspired by Chen et al. (2019), where linear layers across different dimensions are utilized instead of standard matrix multiplications within the graph convolutions.

Dynamic MLP. Using the filtered features $H_{l}$ , the HyperNet module is designed to generate corresponding Weight $_{H_{l}}$ and Bias $_{H_{l}}$ within specific MLPs. This instills scene-awareness into vanilla NeRF, ensuring compatibility with $F_{input}$ , the output of the previous layer in the original NeRF framework. To enhance efficiency, these MLPs are integrated within the sampling-aware filter.

Dynamic Activation Function. Activation functions plays an essential role in the NeRF framework (Sitzmann et al., 2020). Traditional options, such as the ReLU function, may struggle with detail rendering and hinder the performance of D-MLPs due to their static nature. To address this, we introduce a dynamic activation function. This function adaptively activates features in accordance with the unique characteristics of a given scene. Inspired by Perez et al. (2018), we propose the Dynamic Feature-wise Linear Modulation (DFiLM), in which the frequencies (Freq $_{H_{l}}$ ) and phase-shifts (Shift $_{H_{l}}$ ) are dynamically determined from $H_{l}$ , allowing for more responsive activation.

The entire MLP-Block, including both the D-MLP and the activation function, can be expressed as:

$$
\boldsymbol {F} _ {\text { output }} = \operatorname{Shift} _ {\boldsymbol {H} _ {l}} (\operatorname{Freq} _ {\boldsymbol {H} _ {l}} (\operatorname{Weight} _ {\boldsymbol {H} _ {l}} \times \boldsymbol {F} _ {\text { input }} + \operatorname{Bias} _ {\boldsymbol {H} _ {l}})), \tag {7}
$$

To insert the HyperNet modules into the NeRF framework, $F_{output}$ is subsequently fed into an original NeRF's MLP layer for the final result. This yields superior performance, as validated through experimental results. We remark that the parameters are not shared among the HyperNet modules. Moreover, their compact structures ensure that the impact on rendering efficiency is negligible.

HyperNet Modules in $F_{geo}$ and $F_{app}$ . In vanilla NeRF, $F_{geo}$ and $F_{app}$ serve distinct purposes but employ similar MLP structures, albeit with varying complexities. $F_{geo}$ focuses on geometric properties, whereas $F_{app}$ encodes view-dependent features using a smooth BRDF prior for surface reflectance. This smoothness can be facilitated by progressively exploiting guided scene features, along with a reduction in both MLP parameters and activation functions for variable d (Zhang et al., 2020). Recognizing this need, we propose a modified HyperNet module architecture specifically for $F_{app}$ . Our design employs a progressive guidance mechanism within $F_{app}$ , incorporating multiple parallel dynamic branches into the NeRF framework. The weights of the D-MLP in each branch are progressively generated from the preceding branch, enabling the capture of reference features at different levels for complex appearance modeling. Finally, the results of all branches are summed and used as input to the original MLP for predicting the RGB value. In accordance with our analysis, the DFiLM is not used in $F_{app}$ , setting it apart from other elements in the architecture. [Appendix C.1]

# 3.3 LOSS FUNCTIONS

The InsertNeRF pipeline is trained end-to-end utilizing three carefully designed loss functions.

Photometric loss. First, we employ the photometric loss in NeRF (Mildenhall et al., 2021), i.e., the Mean Square Error (MSE) between the rendered and true pixel colors:

$$
\mathcal {L} _ {\mathrm{MSE}} = \sum_ {\boldsymbol {r} \in \mathcal {R}} \left\| \hat {C} (\boldsymbol {r}) - C (\boldsymbol {r}) \right\| _ {2} ^ {2}, \tag {8}
$$

where R is the set of rays in a batch, and $C(\boldsymbol{r})$ is the ground-truth RGB color for ray $r \in R$ .

Backbone loss. During end-to-end training, optimizing the feature extraction without additional guidance poses considerable challenges. To address this, we draw inspiration from autoencoding (Kingma & Welling, 2013). By adding an additional upsampling layer and a small decoder (used exclusively for loss computation), we seek to reconstruct reference images from encoded features. The original images serve as supervision, and we refer to this particular loss term as $L_{backbone}$ .

Table 1: Comparisons of InsertNeRF against SOTA methods with Setting I. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">NeRF Synthetic</td><td colspan="3">LLFF</td><td colspan="3">DTU</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>PixelNeRF (CVPR2021)</td><td>22.65</td><td>0.808</td><td>0.202</td><td>18.66</td><td>0.588</td><td>0.463</td><td>19.40</td><td>0.463</td><td>0.447</td></tr><tr><td>MVSNeRF (ICCV2021)</td><td>25.15</td><td>0.853</td><td>0.159</td><td>21.18</td><td>0.691</td><td>0.301</td><td>23.83</td><td>0.723</td><td>0.286</td></tr><tr><td>IBRNet (CVPR2021)</td><td>26.73</td><td>0.908</td><td>0.101</td><td>25.17</td><td>0.813</td><td>0.200</td><td>25.76</td><td>0.861</td><td>0.173</td></tr><tr><td>ContraNeRF (CVPR2023)</td><td>-</td><td>-</td><td>-</td><td>25.44</td><td>0.842</td><td>0.178</td><td>27.69</td><td>0.904</td><td>0.129</td></tr><tr><td>GeoNeRF $^{\dagger}$  (CVPR2022)</td><td>28.33</td><td>0.938</td><td>0.087</td><td>25.44</td><td>0.839</td><td>0.180</td><td>-</td><td>-</td><td>-</td></tr><tr><td>WaveNeRF $^{\dagger}$  (ICCV2023)</td><td>26.12</td><td>0.918</td><td>0.113</td><td>24.28</td><td>0.794</td><td>0.212</td><td>-</td><td>-</td><td>-</td></tr><tr><td>NeuRay(CVPR2022)</td><td>28.92</td><td>0.920</td><td>0.096</td><td>25.85</td><td>0.832</td><td>0.190</td><td>28.30</td><td>0.907</td><td>0.130</td></tr><tr><td>InsertNeRF (Ours)</td><td>30.35</td><td>0.938</td><td>0.065</td><td>26.44</td><td>0.844</td><td>0.169</td><td>29.75</td><td>0.925</td><td>0.077</td></tr></table>

Table 2: Comparisons and ablations with Setting II. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">NeRF Synthetic</td><td colspan="3">LLFF</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>GNT (ICLR2023)</td><td>27.29</td><td>0.937</td><td>0.056</td><td>25.59</td><td>0.858</td><td>0.128</td></tr><tr><td>Baseline (NeRF)</td><td>7.29</td><td>0.512</td><td>0.690</td><td>11.46</td><td>0.328</td><td>0.582</td></tr><tr><td>NeRF with HyperNetwork</td><td>25.86</td><td>0.902</td><td>0.081</td><td>24.25</td><td>0.793</td><td>0.177</td></tr><tr><td>InsertNeRF w/o MLDS</td><td>25.12</td><td>0.896</td><td>0.098</td><td>24.41</td><td>0.814</td><td>0.156</td></tr><tr><td>InsertNeRF (Ours)</td><td>27.57</td><td>0.936</td><td>0.056</td><td>25.68</td><td>0.861</td><td>0.126</td></tr></table>

Table 3: Results with sparse inputs. 

<table><tr><td>Methods</td><td>PSNR↑</td><td>3-view SSIM↑</td><td>LPIPS↓</td></tr><tr><td>DietNeRF (ICCV 2021)</td><td>14.94</td><td>0.370</td><td>0.496</td></tr><tr><td>RegNeRF (CVPR 2022)</td><td>19.08</td><td>0.587</td><td>0.336</td></tr><tr><td>GeCoNeRF (ICML 2023)</td><td>18.77</td><td>0.596</td><td>0.338</td></tr><tr><td>FreeNeRF (CVPR 2023)</td><td>19.63</td><td>0.612</td><td>0.308</td></tr><tr><td>InsertNeRF (w/o retrain)</td><td>19.41</td><td>0.618</td><td>0.330</td></tr></table>

Dynamic weights loss. Initiating the learning of dynamic weights from scratch introduces difficulties in understanding the connections among multi-scale features. To tackle this issue, we introduce an auxiliary supervision to encompass global-local information. Specifically, we let $C_{n}^{\mathrm{ref}}(\boldsymbol{r}) \in \mathbb{R}^{B \times K \times N \times 3}$ represent the ground-truth RGB values in corresponding reference images for K points in ray r within a batch R. We compute $\boldsymbol{c}_{i}^{\prime} = \sum_{n,l,\boldsymbol{r} \in \mathcal{R}} C_{n}^{\mathrm{ref}}(\boldsymbol{r}) \odot M_{l}^{\mathrm{DY}}$ , the weighted sum of these RGB values by dynamic weights. Utilizing $\boldsymbol{c}_{i}^{\prime}, \hat{C}^{\prime}(\boldsymbol{r})$ is subsequently calculated according to Eq. (2), and supervised by the true color $C(\boldsymbol{r})$ . We designate this loss term as $L_{DY}$ .

We formulate our final loss function as

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{MSE}} + \lambda_ {1} \mathcal {L} _ {\text { backbone }} + \lambda_ {2} \mathcal {L} _ {\mathrm{DY}} \tag {9}
$$

where $\lambda_{1}$ and $\lambda_{2}$ are hyperparameters controlling the relative importance of these terms.

# 4 EXPERIMENTS

We conduct comparative experiments with state-of-the-art (SOTA) methods across different settings on mainstream datasets. Additionally, we validate the effectiveness of the proposed paradigm in the context of derivative NeRF-like systems generalizations and tasks involving sparse inputs.

# 4.1 EXPERIMENTAL PROTOCOL AND SETTINGS

Following IBRNet (Wang et al., 2021), GNeRF exploits the target-reference pairs sampling strategy during both the training and inference phases. Here, reference views are selected from a set of nearby views surrounding the target view. Specifically, N reference views are chosen from a pool of $P \times N$ ( $P \geq 1$ ) neighboring views of target, ensuring that the target view is excluded from the reference views. During the evaluation phase, we conduct evaluations using three metrics: PSNR, SSIM, and LPIPS, on well-established datasets such as NeRF Synthetic, LLFF, and DTU. More training and inference details are provided in the Appendix A and Algorithm in the Appendix C.2.

In our experiments, we follow two GNeRF settings of existing methods:

Setting I. Following NeuRay (Liu et al., 2022), we use three types of training datasets for training GNeRF, including three forward-facing datasets, the synthetic Google Scanned Object dataset and the DTU dataset. Note that we only select training scenes in the DTU dataset, excluding the four evaluation scenes. Following their setting in the experiments, we set N = 8.

Setting II. Following GNT (Wang et al., 2022), we train GNeRF using three forward-facing datasets and the Google Scanned Object dataset. Unlike Setting I, the DTU dataset is not used for either training or evaluation. In addition, we set N = 10 in this setting.

![](images/51a2de385ed52102313afd85faec8d588f9552f6a4892ad6a9976a4256a4d3a2.jpg)

Figure 3: (a) Qualitative comparisons of InsertNeRF against SOTA methods. (b) A t-SNE plot of the scene-specific representations from our HyperNet modules. More analysis in the Appendix D.2   
Table 4: HyperNet modules ablations. 

<table><tr><td>Methods</td><td>PSNR↑</td><td>LLFF SSIM↑</td><td>LPIPS↓</td></tr><tr><td>w/o D-MLP</td><td>23.33</td><td>0.774</td><td>0.198</td></tr><tr><td>w/o Sampling Filter</td><td>24.67</td><td>0.815</td><td>0.158</td></tr><tr><td>w/o DFiLM</td><td>25.04</td><td>0.832</td><td>0.152</td></tr><tr><td>w/o original MLP</td><td>25.44</td><td>0.848</td><td>0.131</td></tr><tr><td>InsertNeRF (Ours)</td><td>25.68</td><td>0.861</td><td>0.126</td></tr></table>

Table 5: MLDS aggregation strategy ablations. 

<table><tr><td>Static-Weight</td><td>Dynamic-Weight</td><td>Auxiliary-Supervision</td><td>Multi-Layers</td><td>Single-Layer</td><td>PSNR↑</td><td>LLFFSSIM↑</td><td>LPIPS↓</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td></td><td>24.88</td><td>0.827</td><td>0.154</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td></td><td>25.55</td><td>0.851</td><td>0.128</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td></td><td>25.53</td><td>0.850</td><td>0.131</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>25.15</td><td>0.838</td><td>0.139</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>25.68</td><td>0.861</td><td>0.126</td></tr></table>

# 4.2 COMPARATIVE EXPERIMENTS

We evaluate InsertNeRF for its generalization based on the vanilla NeRF framework, comparing its performance with SOTA methods under two GNeRF settings. Through extensive quantitative and qualitative experiments, we explore the advantages of our approach, even with fewer references.

Quantitative comparisons. We present quantitative comparisons with SOTA methods under Setting I and Setting II, as reported in Tab. 1 and Tab. 2. For Setting I, the quantitative comparisons in Tab. 1 display our model's competitive results in evaluation datasets, with significant improvements in PSNR, SSIM and LPIPS in comparison to existing SOTA methods. Specifically, PSNR and LPIPS exhibit substantial enhancements by $\sim 1.16\mathrm{dB}\uparrow$ and $\sim 23.6\% \downarrow$ respectively. For Setting II, InsertNeRF consistently outperforms the SOTA method (Wang et al., 2022), as substantiated by the results in Tab. 2. We observe that these improvements become even more pronounced with fewer reference images, alongside higher efficiency, as demonstrated in subsequent sections.

Qualitative comparisons. Fig. 1 and Fig. 3 (a) show the qualitative performances of our method against baseline and SOTA methods. InsertNeRF achieves improved geometric fidelity and clear edges, attributable to the completion capability of global features and the modeling of sample spatial relationships from graph structures. For more analysis and results, please refer to the Appendix E.5.

# 4.3 ABLATION STUDIES

In Tab. 2, we analyze the core components of our method. The findings underscore the vital role of HyperNet modules in enhancing rendering performance, as further evidenced in Fig. 3 (b) that they instill scene-specific capabilities into NeRF's representation. Additionally, the multi-layer dynamic-static aggregation strategy proves to be essential. By integrating both modules, our novel paradigm instills generalizability into the NeRF framework, leading to a performance boost of approximately two to three times compared to the baseline model, i.e., vanilla NeRF. Additionally, we explore the underlying mechanisms driving the effectiveness of these components. More experiments, including single-scene setting, fine-tuning and ablation studies about $F_{app}$ in the Appendix E

![](images/0193895e99462d95add36688b8c647182e0199b1f78a2b152dd1c3cbdfa07793.jpg)

<details>
<summary>bar_line</summary>

| X-Axis | GNT PSNR | InsertNeRF PSNR | GNT Evaluation Time (s) | InsertNeRF Evaluation Time (s) |
|---|---|---|---|---|
| 1 | 16.5 | 13.8 | 25.8 | 25.8 |
| 3 | 19.7 | 11.8 | 30.7 | 42.4 |
| 5 | 26.25 | 19.32 | 36.25 | 60.0 |
| 8 | 45.5 | 27.15 | 45.5 | 70.0 |
| 10 | 51.88 | 33.1 | 51.88 | 78.0 |
</details>

Figure 4: Performance and efficiency under different input-number N on NeRF Synthetic.

Table 6: Quantitative results of InsertNeRF and Insert-mip-NeRF on multi-scale NeRF Synthetic. 

<table><tr><td rowspan="2">Methods</td><td colspan="4">PSNR↑</td><td colspan="4">SSIM↑</td><td colspan="4">LPIPS↓</td></tr><tr><td>Full Res.</td><td>1/2 Res.</td><td>1/4 Res.</td><td>1/8 Res.</td><td>Full Res.</td><td>1/2 Res.</td><td>1/4 Res.</td><td>1/8 Res.</td><td>Full Res.</td><td>1/2 Res.</td><td>1/4 Res.</td><td>1/8 Res.</td></tr><tr><td>mip-NeRF</td><td>12.94</td><td>13.03</td><td>13.18</td><td>13.33</td><td>0.700</td><td>0.636</td><td>0.563</td><td>0.469</td><td>0.424</td><td>0.460</td><td>0.470</td><td>0.530</td></tr><tr><td>InsertNeRF</td><td>27.60</td><td>28.58</td><td>29.45</td><td>29.85</td><td>0.926</td><td>0.943</td><td>0.960</td><td>0.972</td><td>0.066</td><td>0.054</td><td>0.045</td><td>0.036</td></tr><tr><td>Insert-mip-NeRF</td><td>28.15</td><td>29.17</td><td>30.22</td><td>30.62</td><td>0.935</td><td>0.951</td><td>0.966</td><td>0.977</td><td>0.056</td><td>0.045</td><td>0.037</td><td>0.029</td></tr></table>

HyperNet modules. Tab. 4 demonstrates that both the sampling-aware filters and dynamic activation functions are vital in the HyperNet modules, with the sampling-aware filters having a more substantial impact. This could be due to the need to consider relationships between sampled points in the rendering process, which implicitly models occlusions, as noted in Liu et al. (2022). Solely using dynamic activation functions without D-MLP leads to a marked decline in performance, highlighting the essential role of MLPs in neural representation. Furthermore, using only the HyperNet modules and omitting the original NeRF's MLP layers results in inferior performance, reducing training stability.

Multi-layer dynamic-static aggregation strategy. In Tab. 5, ablation studies reveal the significance of dynamic-static weights and multi-layer features. Using only dynamic weights appears more effective than static weight, likely because they are adaptively generated to suit different scene features. The auxiliary supervision for dynamic weights and multi-layer global-local features also play essential roles in aggregating multi-view features, underlining their importance in this strategy.

Input number $(N)$ and efficiency. Since feature extraction is time-consuming, reducing the number of reference images substantially improves the training and inference efficiency of the network. Fig. 4 illustrates the performance of InsertNeRF as the number of reference images $(N)$ varies for training on NeRF Synthetic. In comparison to GNT (Wang et al., 2022), InsertNeRF consistently demonstrates superior rendering performance and inference efficiency. This success can be attributed to our novel generalization paradigm and the compact structures of the HyperNet modules.

# 4.4 INSERT-NERF-LIKE FRAMEWORKS

Thanks to the plug-and-play advantage of the HyperNet modules, we extend the study of generalization to derived domains of NeRF, such as mip-NeRF (Barron et al., 2021) and NeRF++ (Zhang et al., 2020), areas that have rarely been discussed before. More details are provided in the Appendix A.

Insert-mip-NeRF. Mip-NeRF is a multi-scale NeRF-like model used to address the inherent aliasing of NeRF, a significant challenge for GNeRF. Unlike Huang et al. (2023), we explore how to instill generalizability into mip-NeRF, following its original setup. We report the qualitative and quantitative performance of mip-NeRF, InsertNeRF, and Insert-mip-NeRF on multi-scale NeRF Synthetic in a cross-scene generalization setting (see Tab. 6, Fig. 1 and Fig. 5). One can observe that incorporating the HyperNet modules not only enhances generalization for mip-NeRF but also addresses the inherent aliasing of InsertNeRF and improves the performance in the task of multi-scale rendering.

Insert-NeRF++. NeRF++, an unbounded NeRF. Fig. 1 and Fig. 13 depicts qualitative and quantitative rendering results of Insert-NeRF++. It is evident that our approach has successfully instilled generalizability into the NeRF++ framework, doubling its PSNR compared to the original.

Sparse Inputs. Training NeRF with sparse inputs has become a notable focus recently (Niemeyer et al., 2022; Yang et al., 2023). Unlike our nearby reference views setting (Sec. 4.1), this task often involves training from a limited number of fixed viewpoints to represent the entire scene. Under this setting, we relax constraints on selecting nearby viewpoints and uniformly select fixed sparse seen viewpoints to infer on arbitrary unseen viewpoints. Unlike existing works, our method trains on extensive auxiliary datasets, allowing us to represent the entire evaluation scene from sparse inputs without retraining (see Tab. 3). To ensure fairness, all scenes in evaluation are excluded in the training phase. In conclusion, InsertNeRF offers a novel insight that employs pre-training on auxiliary datasets to enhance representation capabilities with sparse inputs. We believe that, through

![](images/2ef2817c4428e0cf66e5cdb00bd6619208433de3bfffd5e53f3d32ad3ccf400f.jpg)

<details>
<summary>text_image</summary>

GroundTruth
InsertNeRF
Insert-mip-NeRF
</details>

Figure 5: Qualitative results of Insert-mip-NeRF. Please refer to the Appendix E.5 for more results.

fine-tuning on the evaluation scene and incorporating existing technologies like geometry and color regularization, our paradigm will achieve even better performance under sparse inputs.

# 5 CONCLUSION

We present InsertNeRF, a novel paradigm that instills generalizability into NeRF systems. Unlike popular transformer-based structures, our HyperNet modules are efficiently incorporated into the original NeRF-like framework, leveraging reference scene features to generate scene-specific network weights. To achieve this, we design a multi-layer dynamic-static feature aggregation strategy for extracting scene features from reference images and employ sampling-aware filters to explore relationships between sample points. Experiments on well-established datasets show that InsertNeRF and other Insert-NeRF-like frameworks can render high-quality images across different scenes without retraining. This offers insights for future works on: (i) generalization tasks for additional NeRF-like systems such as mip-NeRF 360; and (ii) sparse inputs tasks based on auxiliary datasets.

# ACKNOWLEDGEMENTS

This work was supported in part by the National Natural Science Foundation of China under Grant 62276128, Grant 62192783, and Grant 62106100; in part by the Jiangsu Natural Science Foundation under Grant BK20221441; in part by the Young Elite Scientists Sponsorship Program by CAST under Grant 2023QNRC001; in part by the Collaborative Innovation Center of Novel Software Technology and Industrialization.

# REFERENCES

Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit Bermano. Hyperstyle: Stylegan inversion with hypernetworks for real image editing. In Proceedings of the IEEE/CVF conference on computer Vision and pattern recognition, pp. 18511–18521, 2022.   
Yanqi Bao, Yuxin Li, Jing Huo, Tianyu Ding, Xinyue Liang, Wenbin Li, and Yang Gao. Where and how: Mitigating confusion in neural radiance fields from sparse inputs. arXiv preprint arXiv:2308.02908, 2023.   
Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5855–5864, 2021.   
Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5470–5479, 2022.   
SC Chan, Heung-Yeung Shum, and King-To Ng. Image-based rendering and synthesis. IEEE Signal Processing Magazine, 24(6):22–33, 2007.   
Vinod Kumar Chauhan, Jiandong Zhou, Ping Lu, Soheila Molaei, and David A Clifton. A brief review of hypernetworks in deep learning. arXiv preprint arXiv:2306.06955, 2023.   
Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14124–14133, 2021.   
Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision, pp. 333–350. Springer, 2022.   
Yunpeng Chen, Marcus Rohrbach, Zhicheng Yan, Yan Shuicheng, Jiashi Feng, and Yannis Kalantidis. Graph-based global reasoning networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 433–442, 2019.

Pei-Ze Chiang, Meng-Shiun Tsai, Hung-Yu Tseng, Wei-Sheng Lai, and Wei-Chen Chiu. Stylizing 3d scene via implicit representation and hypernetwork. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1475–1484, 2022.   
Paul E Debevec, Camillo J Taylor, and Jitendra Malik. Modeling and rendering architecture from photographs: A hybrid geometry-and image-based approach. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pp. 11–20, 1996.   
Emilien Dupont, Miguel Bautista Martin, Alex Colburn, Aditya Sankar, Josh Susskind, and Qi Shan. Equivariant neural rendering. In International Conference on Machine Learning, pp. 2761–2770. PMLR, 2020.   
Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna, and Thomas Funkhouser. Local deep implicit functions for 3d shape. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4857–4866, 2020.   
David Ha, Andrew Dai, and Quoc V Le. Hypernetworks. arXiv preprint arXiv:1609.09106, 2016.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.   
Xin Huang, Qi Zhang, Ying Feng, Xiaoyu Li, Xuan Wang, and Qing Wang. Local implicit ray function for generalizable radiance field representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 97–107, 2023.   
Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Globally and locally consistent image completion. ACM Transactions on Graphics (ToG), 36(4):1–14, 2017.   
Brian KS Isaac-Medina, Chris G Willcocks, and Toby P Breckon. Exact-nerf: An exploration of a precise volumetric parameterization for neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 66–75, 2023.   
Nishant Jain, Suryansh Kumar, and Luc Van Gool. Enhanced stable view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13208–13217, 2023.   
Chiyu Jiang, Avneesh Sud, Ameesh Makadia, Jingwei Huang, Matthias Nießner, Thomas Funkhouser, et al. Local implicit grid representations for 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6001–6010, 2020.   
Mohammad Mahdi Johari, Yann Lepoittevin, and François Fleuret. Geonerf: Generalizing nerf with geometry priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18365–18375, 2022.   
Adam Kania, Artur Kasymov, Maciej Zięba, and Przemysław Spurek. Hypernerfgan: Hypernetwork approach to 3d nerf gan. arXiv preprint arXiv:2301.11631, 2023.   
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4):1–13, 2017.   
Johannes Kopf, Fabian Langguth, Daniel Scharstein, Richard Szeliski, and Michael Goesele. Image-based rendering in the gradient domain. ACM Transactions on Graphics (TOG), 32(6):1–9, 2013.   
Jonáš Kulhánek, Erik Derner, Torsten Sattler, and Robert Babuška. Viewformer: Nerf-free neural rendering from few images using transformers. In European Conference on Computer Vision, pp. 198–216. Springer, 2022.   
Yawei Li, Shuhang Gu, Kai Zhang, Luc Van Gool, and Radu Timofte. Dhp: Differentiable meta pruning via hypernetworks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16, pp. 608–624. Springer, 2020.

Zhouchen Lin and Heung-Yeung Shum. A geometric analysis of light field rendering. International Journal of Computer Vision, 58:121–138, 2004.   
Shichen Liu, Shunsuke Saito, Weikai Chen, and Hao Li. Learning to infer implicit surfaces without 3d supervision. Advances in Neural Information Processing Systems, 32, 2019.   
Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng Wang, Christian Theobalt, Xiaowei Zhou, and Wenping Wang. Neural rays for occlusion-aware image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7824–7833, 2022.   
Zhen Liu, Yao Feng, Michael J Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. Meshdiffusion: Score-based generative 3d mesh modeling. arXiv preprint arXiv:2303.08133, 2023.   
Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4460–4470, 2019.   
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.   
Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5480–5490, 2022.   
Sida Peng, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Representing volumetric videos as dynamic mlp maps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4252–4262, 2023.   
Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pp. 523–540. Springer, 2020.   
Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.   
Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.   
Gernot Riegler and Vladlen Koltun. Free view synthesis. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIX 16, pp. 623–640. Springer, 2020.   
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241. Springer, 2015.   
Sudipta Sinha, Drew Steedly, and Rick Szeliski. Piecewise planar stereo for image-based rendering. In 2009 International Conference on Computer Vision, pp. 1881–1888, 2009.   
Vincent Sitzmann, Michael Zollhöfer, and Gordon Wetzstein. Scene representation networks: Continuous 3d-structure-aware neural scene representations. Advances in Neural Information Processing Systems, 32, 2019.   
Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33:7462–7473, 2020.

Przemysław Spurek, Sebastian Winczowski, Jacek Tabor, Maciej Zamorski, Maciej Zięba, and Tomasz Trzciński. Hypernetwork approach to generating point clouds. arXiv preprint arXiv:2003.00802, 2020a.   
Przemysław Spurek, Maciej Zięba, Jacek Tabor, and Tomasz Trzciński. Hyperflow: Representing 3d objects as surfaces. arXiv preprint arXiv:2006.08710, 2020b.   
Mohammed Suhail, Carlos Esteves, Leonid Sigal, and Ameesh Makadia. Generalizable patch-based neural rendering. In European Conference on Computer Vision, pp. 156–174. Springer, 2022.   
Johannes Von Oswald, Christian Henning, Benjamin F Grewe, and João Sacramento. Continual learning with hypernetworks. arXiv preprint arXiv:1906.00695, 2019.   
Chu Wang, Babak Samari, and Kaleem Siddiqi. Local spectral graph convolution for point set feature learning. In Proceedings of the European conference on computer vision (ECCV), pp. 52–66, 2018.   
Peihao Wang, Xuxi Chen, Tianlong Chen, Subhashini Venugopalan, Zhangyang Wang, et al. Is attention all nerf needs? arXiv preprint arXiv:2207.13298, 2022.   
Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multiview image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4690–4699, 2021.   
Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.   
Muyu Xu, Fangneng Zhan, Jiahui Zhang, Yingchen Yu, Xiaoqin Zhang, Christian Theobalt, Ling Shao, and Shijian Lu. Wavenerf: Wavelet-based generalizable neural radiance fields. arXiv preprint arXiv:2308.04826, 2023.   
Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8254–8263, 2023.   
Lingfeng Yang, Xiang Li, Renjie Song, Borui Zhao, Juntian Tao, Shihao Zhou, Jiajun Liang, and Jian Yang. Dynamic mlp for fine-grained image classification by leveraging geographical and temporal information. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10945–10954, 2022.   
Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pp. 767–783, 2018.   
Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4578–4587, 2021.   
Maciej Zamorski, Maciej Zięba, Piotr Klukowski, Rafał Nowak, Karol Kurach, Wojciech Stokowiec, and Tomasz Trzciński. Adversarial autoencoders for compact representations of 3d point clouds. Computer Vision and Image Understanding, 193:102921, 2020.   
Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492, 2020.   
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.   
Zicheng Zhang, Yinglu Liu, Congying Han, Yingwei Pan, Tiande Guo, and Ting Yao. Transforming radiance field with lipschitz network for photorealistic 3d scene stylization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20712–20721, 2023.

Linqi Zhou, Yilun Du, and Jiajun Wu. 3d shape generation and completion through point-voxel diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5826–5835, 2021.   
Haidong Zhu, Tianyu Ding, Tianyi Chen, Ilya Zharkov, Ram Nevatia, and Luming Liang. Caesarnerf: Calibrated semantic representation for few-shot generalizable neural rendering. arXiv preprint arXiv:2311.15510, 2023.   
Dominik Zimny, T Trzciński, and Przemysław Spurek. Points2nerf: Generating neural radiance fields from 3d point cloud. arXiv preprint arXiv:2206.01290, 2022.   
Dominik Zimny, Jacek Tabor, Maciej Zięba, and Przemysław Spurek. Multiplanenerf: Neural radiance field with non-trainable representation. arXiv preprint arXiv:2305.10579, 2023.

# A EXPERIMENTAL DETAILS

During the training process, the feature extraction and InsertNeRF-systems are trained with different learning rates in an end-to-end manner, and we apply the Adam to optimize the entire network with an exponentially decaying learning rate. In our experiments, $\lambda_{1}$ is set as 0.1 and $\lambda_{2}$ is set as 1. During the cross-scene evaluation phase, we individually test the average PSNR, SSIM, and LPIPS metrics of all testing-view renderings for each scene and report the average values across all scenes, as shown in Tab. 1. For the sake of fair experimental comparison and efficiency with (Wang et al., 2022), we set the rendering stride size to 2 across all experiments. Empirical evidence demonstrates that this choice influences the evaluation of metrics, as shown in Tab. 8. Our model is implemented using PyTorch 1.11.0 and all experiments are conducted on Nvidia RTX 3090 GPUs with CUDA 11.4.

Metrics. We calculate the Peak Signal to-Noise Ratio (PSNR) and Structural SIMilarity (SSIM) (Wang et al., 2004) to evaluate rendering quality for target novel viewpoints. Additionally, the Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018) is adopted as a perceptual metric. For all our experiments, we report the average of the PSNR, SSIM and LPIPS under different testing views in multiple scenes to verify the generalization. It is noteworthy that, similar to the majority of GNeRF research (Liu et al., 2022; Wang et al., 2022), our focus lies on foreground-centric metrics for both the NeRF Synthetic and DTU datasets during the evaluation process.

# A.1 EVALUATION DATASETS.

NeRF Synthetic. The dataset consists of 8 synthetic objects with viewpoints uniformly sampled on the upper hemisphere. Each scene comprises 200 test images, wherein a sampling strategy with an interval of 8 images is employed during the evaluation phase.

Local Light Field Fusion (LLFF). The dataset consists of 8 complex real-world scenes. Each scene includes real images, 1/8 of which are used as the evaluation dataset.

DTU Dataset. The dataset consists of 128 object-scenes, which is a classic dataset of MVS. In the experiments, we select 4 scenes (birds, tools, bricks and snowman) for evaluation (Liu et al., 2022).

# A.2 EXPERIMENTAL DETAILS FOR INSERTNERF

For all our experiments, we maintain the experimental protocols of NeuRay (Liu et al., 2022) and GNT (Wang et al., 2022). In Setting I, we also employ depth maps (Liu et al., 2022) as priors to assist the Hypernet modules to generate adaptive weights. Following (Liu et al., 2022), we randomly sample 2048 rays from Target-Reference pairs, and it trains for a total of 600,000 steps. In Setting II (Wang et al., 2022), we randomly sample 512 rays from Target-Reference pairs, and it trains for a total of 400,000 steps without any priors. In order to enhance training and inference efficiency, we sample K = 64 points along each ray and simplify the volume rendering process in our paradigm.

# A.3 EXPERIMENTAL DETAILS FOR INSERT-MIP-NERF

Insert-mip-NeRF substitutes point-samplings with a sequence of conical-frustums and introduces the integrated positional encoding into the InsertNeRF framework. In contrast to InsertNeRF, we

sample K = 65 positions for integrated positional encoding on conical-frustums. Throughout all experiments, we employ the multi-scale variants of NeRF Synthetic, such as Full, 1/2, 1/4 and 1/8 Resolutions, to simulate multi-resolution scenes for training and evaluation as (Barron et al., 2021). The remaining experimental configurations follow those of mipNeRF and our Setting II.

# A.4 EXPERIMENTAL DETAILS FOR INSERT-NERF++

Insert-NeRF++ divides the InsertNeRF's scene representations into foreground and background components and combines both for finally rendering (Zhang et al., 2020). In Insert-NeRF++, we avoid using inverted sphere parametrization and chose standard parameterization across foreground-background spatial ranges, which possesses the capability to offer precise spatial positions for projection process $\Pi_n(x)$ . Following (Zhang et al., 2020), we employ the Tanks and Temples dataset (Knapitsch et al., 2017), a real-world unbound dataset captured with hand-held cameras, for training and evaluation under cross-scenes setting. During the evaluation process, we separately report the renderings for the foreground and background as (Zhang et al., 2020). The remaining experimental configurations also follow those of vanilla NeRF++ and our Setting II.

# B RELATED WORKS

# B.1 IMAGE-BASED RENDERING

Image-based rendering (IBR) (Chan et al., 2007) is a classic technique that aims to generate novel views within a specific scene by warping and integrating pixel information from reference-view images. To ensure spatial consistency, most existing works simplify this challenge by resorting to estimated explicit geometry or depth maps (Riegler & Koltun, 2020; Jain et al., 2023). In order to obtain proxy geometry, Structure-from-Motion (SfM) (Sinha et al., 2009) and MultiView Stereo (MVS) (Yao et al., 2018) have recently attracted the attention of researchers in the field of IBR. However, hints from explicit geometry without 3D supervision are unstable in most scenarios. Recently light field rendering (Lin & Shum, 2004) has become one of the alternatives to explicit representations, which considers the lighting and reflection properties of the scene to ensure visually plausible rendering. Moreover, some works (Kopf et al., 2013; Chen et al., 2021) focus on aggregating information from multiple reference views, which exploits the relationships between references and also implicitly solves the occlusion. As opposed to explicit geometry, these methods rely on constructing implicit representations to enable reasoning about novel views (Liu et al., 2019). Different from the above explicit or implicit scene-customized representations for IBR, our method can be used in a large number of scenes simultaneously without retraining.

# B.2 NEURAL SCENE REPRESENTATION

Representing the geometry and appearance of a scenes with neural networks has been considered an alternative to 3D scene representations in recent years (Mescheder et al., 2019; Peng et al., 2020). Existing works demonstrate the potential of Multi-Layer Perceptrons (MLPs) in implicit representations, which activate spatial features by continuous functions (Genova et al., 2020; Jiang et al., 2020). Neural radiance fields (NeRF) (Mildenhall et al., 2021) apply such functions for coordinate-based representations, which use high-dimensional interpolation to produce photorealistic renderings of target views. On its basis, mip-NeRF (Barron et al., 2021) replaces rays with casting cones during volume rendering, changing the input of NeRF from points to cone frustums, and introduces an integrated positional encoding for multi-resolutions images. Subsequently, NeRF++ (Zhang et al., 2020) and mip-NeRF 360 (Barron et al., 2022) further improve NeRF and mip-NeRF to adapt to distant targets under unbounded scenes. To further enhance representation efficiency, (Chen et al., 2022) and (Zimny et al., 2023) proposed efficient representation methods to replace a large number of neural network parameters. Although derivative works have been a surge, similar to most IBR works, NeRF must also be trained for each novel scene, which is time-consuming in practice.

# B.3 GENERATIVE MODELS

With the development of generative models, 3D generative models have been widely discussed, enabling the direct construction of 3D representations such as point clouds (Zamorski et al., 2020),

surfaces (Spurek et al., 2020b), voxels (Zhou et al., 2021) and NeRF (Poole et al., 2022). A significant amount of works have leveraged techniques from image generative models and applied them to 3D generation, including GAN (Kania et al., 2023) and diffusion model (Liu et al., 2023). In this work, we focus on some 3D generative models with HyperNetwork. (Spurek et al., 2020a) is an early work that builds variable size representations of point clouds with HyperNetwork. Then, HyperFlow (Spurek et al., 2020b) uses a hypernetwork to model 3D objects as families of surfaces and Points2NeRF (Zimny et al., 2022) utilizes a HyperNetwork to generate NeRF from a 3D point cloud. Additionally, in recent years, there have been some NeRF works that focus on this technique, they directly incorporate HyperNetwork into NeRF, as described in Section 2.2. However, in the generalizable NeRF task, such idea is suboptimal, overlooking the characteristic of different attributes, such as volume density and color. Furthermore, they struggle to capture the relationship between the inputs (reference images) in the target's sampling process. Therefore, this paper proposes two types of HyperNet module structures for $\mathcal{F}_{geo}$ and $\mathcal{F}_{app}$ and Sampling-aware Filter separately to mitigate the aforementioned two issues.

# C IMPLEMENTATION DETAILS

# C.1 HYPERNET MODULE ARCHITECTURE

![](images/d541a18026a2b92ee7f6e5fee72b75d114b5b7ca0baad2a8a172e141b20867cc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Dynamic MLP"]
    B --> C["Weights"]
    B --> D["Bias"]
    C --> E["Output"]
    D --> F["Dynamic AF"]
    F --> G["Freq"]
    F --> H["Shifts"]
    G --> I["BN+ReLU"]
    H --> I
    I --> J["Output"]
```
</details>

(a) HyperNet module in $F_{geo}$

![](images/cc69f0e90001f35bcef45756479a5a80e08ad366d223526e027217d2a6db40eb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Dynamic MLP 1"]
    B --> C["Weights"]
    C --> D["T₁"]
    D --> E["Dynamic MLP 2"]
    E --> F["Weights"]
    F --> G["T₂"]
    G --> H["Output"]
    H --> I["Input"]
    I --> J["Dynamic MLP N"]
    J --> K["Weights"]
    K --> L["Output"]
    L --> M["H₀"]
    M --> N["Dynamic MLP 1"]
```
</details>

(b) HyperNet module in $\mathcal{F}_{app}$   
Figure 6: Detailed HyperNet module of $F_{geo}$ and $F_{app}$

Diverging from the conventional HyperNetworks (Ha et al., 2016), the direct prediction of distinct attributes, such as emitted color and volume density, within the NeRF framework often proves suboptimal. Drawing an analogy to the vanilla NeRF (Mildenhall et al., 2021), where different network depths were assigned for $F_{geo}$ and $F_{app}$ , it is essential for InsertNeRF to discuss distinct HyperNet module structures for them. $F_{geo}$ plays a pivotal role in the NeRF's geometric representations, which necessitates accurate inference and completion of the relationship between references based on global-local features. Based on this analysis, multi-scale features $H_{l}$ are systematically introduced into the $F_{geo}$ 's HyperNet modules, as shown in Fig. 6(a). After Eq. (7), $F_{output}$ is subsequently fed into an original NeRF's MLP layer. Note that we also incorporate an additional ReLU and BatchNorm after MLP to ensure training stability for HyperNet modules.

$$
\boldsymbol {F} _ {\text { final }} = \operatorname{ReLU} (\text { BatchNorm } (\mathrm{MLP} _ {l} (\boldsymbol {F} _ {\text { output }}))), \tag {10}
$$

Note that in $\mathcal{F}_{geo}$ , as the network's depth increases, we leverage denser features for guiding weights prediction, i.e., global to dense. Additionally, when the number of MLP layers surpasses that of the feature layers, $H_0$ will be recurrently utilized in the remaining MLP layers.

Unlike $F_{geo}$ , $F_{app}$ exhibits a heightened focus on dense features (Wang et al., 2022) and the smooth BRDF prior for surface reflectance (Zhang et al., 2020). As shown in Fig. 6(b), $F_{app}$ 's HyperNet modules employ a parallel progressive generation paradigm and residual connection that respond to the desired smoothness. Specifically, given dense feature $H_{0}$ ,

$$
\tilde {\boldsymbol {F}} _ {\text { final }} = \mathrm{MLP} \left(\sum_ {z = 0} ^ {Z} \left(\operatorname{Weight} _ {\boldsymbol {T} _ {z}} \times \boldsymbol {F} _ {\text { input }}\right) + \boldsymbol {F} _ {\text { input }}\right), \quad \left\{ \begin{array}{c c} \boldsymbol {T} _ {z} = \operatorname{Weight} _ {\boldsymbol {T} _ {z - 1}} & z \geq 1 \\ \boldsymbol {T} _ {z} = \boldsymbol {H} _ {0} & z = 0 \end{array} , \right. \tag {11}
$$

where $Z$ represents the number of parallel branches. Note that DFiLM and dynamic bias are not utilized in $\mathcal{F}_{app}$ for the smooth BRDF prior.

# C.2 PSEUDOCODE

In contrast to the scene-customized vanilla NeRF and its derivative works, GNeRF primarily concentrates on cross-scene rendering tasks without any retraining. As shown in Fig. 1, our HyperNet modules possess the capacity to instill generalizability into NeRF-like systems. Here, we delve further into the training processes of InsertNeRF-like systems. In Algorithm 1, due to $\Omega_T^{\text{TrScene}}$ 's adaptive response to the stochastic sampling of scenes by $\mathcal{D}_{\text{TrScene}}$ (that represents data of training scenes), InsertNeRF-like systems acquires inherent generalizability, where NeRF-like systems encompass diverse frameworks, including but not limited to mipNeRF, NeRF++, NeRF- -, and others. In the evaluation phase, for any given $P_T$ , we sample neighboring views $\{I_n, P_n\}_{n=1}^N$ from $\mathcal{D}_{\text{TeScene}}$ , rendering for $I_T$ with the pretrained $\Theta_{\text{NeRF-like Systems}}$ and $\Theta_{\text{HyperNet}}$ , as shown in Algorithm 2.

Algorithm 1: Training for InsertNeRF-like Systems   
Data: Training Datasets $D_{Train}$ Result: $\Theta_{NeRF-like Systems}, \Theta_{HyperNet}$ 1 while $t \leq it$ do

2    Sample: $r(t_i) \leftarrow \left\{ \{I_T, P_T\}, \{I_n, P_n\}_{n=1}^N \right\} \leftarrow D_{TrScene} \leftarrow D_{Train}$ 3 $\Omega_T^{TrScene} \leftarrow HyperNet \left( \left\{ F_{view} \left( \{F_n (\Pi_n(r(t_i)))\}_{n=1}^N \right) \right\}_{i=1}^K \right)$ 4    InsertNeRF-like Systems $\leftarrow NeRF-like Systems (r(t_i), \Omega_T^{TrScene})$ 5 $\Theta_{NeRF-like Systems}, \Theta_{HyperNet} \leftarrow L$ 6 $t \leftarrow t + 1;$ 7 end

Algorithm 2: Testing for InsertNeRF-like Systems   
Input : $\Theta_{NeRF-like Systems}$ , $\Theta_{HyperNet}$ , $D_{Test}$ , and Random $P_{T}$ in Testing Scenes.
Output: $I_{T}$ 1 Sample: $r(t_{i}) \leftarrow \left\{ \left\{ \{I_{n}, P_{n}\}_{n=1}^{N} \leftarrow \mathcal{D}_{TeScene} \leftarrow \mathcal{D}_{Test} \right\}, \{P_{T}\} \right\}$ 2 $\Omega_{T}^{TeScene} \leftarrow HyperNet \left( \left\{ \mathcal{F}_{view} \left( \{ F_{n} (\Pi_{n}(r(t_{i})) \}_{n=1}^{N} \right) \right\}_{i=1}^{K} \right)$ 3 $I_{T} \leftarrow InsertNeRF-like Systems (r(t_{i}), \Omega_{T}^{TeScene})$

# C.3 GRAPH REASONING

Graph-based methods have been the focus of extensive research recently and shown to be an efficient way of relation reasoning (Wang et al., 2018). Following the spatial properties, we conceptualize all sampled points along a ray in a fully connected graph to find correlations between inter-samples and further update node features. Specifically, as shown in Eq. (6), InsertNeRF initially predicts a learnable adjacency matrix $A_{l}$ to parameterize the edge weights between nodes, which models the relationships between sampled points. Subsequently, $W_{l}^{a}$ is employed to update node states, mitigating the noise from epipolar geometric constrains. Furthermore, the identity matrix I is introduced to guide the learning process to pay more attention to the intrinsic characteristics of node features. An naive approach is to calculate the adjacency matrix based on the similarity between node features or Euclidean distance, and update node states, similar to existing graph convolution works. However, this is computationally expensive, especially for a large number of MLP blocks in InsertNeRF. Inspired by Chen et al. (2019), $A_{l}$ and $W_{l}^{a}$ are replaced by two separate linear layers operating in different dimensions, while the identity matrix is represented as a residual connection,

$$
\boldsymbol {H} _ {l} = \text { Linear } \left(\text { Linear } \left(F _ {\text { view }}\right) ^ {T}\right) ^ {T} + F _ {\text { view }}. \tag {12}
$$

Table 7: Ablation studys for Multi-layer Dynamic-Static Aggregation Strategy with IBRNet (Wang et al., 2021) 

<table><tr><td rowspan="2">Methods</td><td colspan="3">NeRF Synthetic</td><td colspan="3">LLFF</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>InsertNeRF w/o Max</td><td>27.44</td><td>0.930</td><td>0.061</td><td>25.52</td><td>0.854</td><td>0.131</td></tr><tr><td>IBR-InsertNeRF</td><td>25.71</td><td>0.909</td><td>0.085</td><td>25.00</td><td>0.836</td><td>0.140</td></tr><tr><td>Multi-layer IBR-InsertNeRF</td><td>26.89</td><td>0.915</td><td>0.074</td><td>25.31</td><td>0.845</td><td>0.131</td></tr><tr><td>InsertNeRF (OUR)</td><td>27.57</td><td>0.936</td><td>0.056</td><td>25.68</td><td>0.861</td><td>0.126</td></tr></table>

![](images/d960e67df0d9440fc2ebe2698c79c8193c86dc9c73d56ae0500ab0d80f094e73.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Reference features"] --> B["Input"]
    B --> C["Pose × N"]
    C --> D["Static Weight"]
    D --> E["Visual Map"]
    E --> F["Volume Rendering"]
    F --> G["Adaptive Generation"]
    G --> H["Max"]
    H --> I["Dynamic Weights"]
    I --> J["Merge Multi-layers"]
    J --> K["Sum"]
    K --> L["Auxiliary Supervision"]
    M["HyperNet Modules"] --> N["Input"]
    N --> O["Pose × N"]
    O --> P["Static Weight"]
    P --> Q["Output"]
    style A fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style P fill:#f9f,stroke:#333
```
</details>

Figure 7: The structures of multi-views feature aggregation parts in IBRNet and InsertNeRF

# D DISCUSSION

# D.1 DIFFERENT FROM IBRNET ABOUT DYNAMIC-STATIC

For inputs, we employ the global-dense features as our multi-layer inputs, compared to IBRNet's single-layer dense feature, it not only retains the detailed information from dense features but also utilizes global features to predict occluded regions, as demonstrated on the depth renderings shown in Fig. 1b, where reports the rendering results of IBRNet (top) and InsertNeRF (bottom) in terms of color-depth. It is evident that InsertNeRF produces sharper edges and achieves more accurate depth predictions for background regions, even when occluded in the reference views.

For architecture, IBRNet generates the visual maps using features based on the $M^{ST}$ , while our InsertNeRF directly predicts $M_{l}^{DY}$ using multi-layer features before $M^{ST}$ . Intuitively, our strategy prevents excessive reliance on the $M^{ST}$ and contributes to the adaptive inference of relationships between reference-target images. To provide additional validation, we conduct ablation experiments. Concretely, we replace the Multi-layer DynamicStatic with the aggregation strategy from IBRNet and introduce identical multi-layer inputs into IBR-InsertNeRF for fairness. As shown in Tab. 7, despite its simplicity, our approach yields significant improvements under GNeRF settings. In addition, we also observe that the Multi-layer inputs still contribute to a notable enhancement in IBR-InsertNeRF, which is consistent with the findings in Tab. 5.

For supervision, in contrast to the direct generation of visual maps without any supervision, we introduce auxiliary supervision to guide $M_{l}^{DY}$ in fully encoding global-dense features. As shown in Tab. 5, the significance of

our auxiliary supervision cannot be disregarded. In summary, as shown in Fig. 7, compared to IBR-Net, the Multi-layer Dynamic-Static Aggregation Strategy focuses on predicting dynamic weights and combines them with static weight based on the multi-layer inputs and the auxiliary supervision to aggregate multiple reference features.

![](images/c0e1531bcbccb06dbcd1319da1144c484bd57681eca3fb20294698afe8900dc1.jpg)

<details>
<summary>scatter</summary>

| Category   | X Coordinate | Y Coordinate |
| ---------- | ------------ | ------------ |
| Ficus      | 0.1          | 0.9          |
| Ficus      | 0.2          | 0.85         |
| Ficus      | 0.3          | 0.8          |
| Ficus      | 0.4          | 0.75         |
| Ficus      | 0.5          | 0.7          |
| Ficus      | 0.6          | 0.65         |
| Ficus      | 0.7          | 0.6          |
| Ficus      | 0.8          | 0.55         |
| Ficus      | 0.9          | 0.5          |
| Ficus      | 1.0          | 0.45         |
| Materials  | 0.1          | 0.8          |
| Materials  | 0.2          | 0.75         |
| Materials  | 0.3          | 0.7          |
| Materials  | 0.4          | 0.65         |
| Materials  | 0.5          | 0.6          |
| Materials  | 0.6          | 0.55         |
| Materials  | 0.7          | 0.5          |
| Materials  | 0.8          | 0.45         |
| Materials  | 0.9          | 0.4          |
| Materials  | 1.0          | 0.35         |
| Mic        | 0.1          | 0.7          |
| Mic        | 0.2          | 0.65         |
| Mic        | 0.3          | 0.6          |
| Mic        | 0.4          | 0.55         |
| Mic        | 0.5          | 0.5          |
| Mic        | 0.6          | 0.45         |
| Mic        | 0.7          | 0.4          |
| Mic        | 0.8          | 0.35         |
| Mic        | 0.9          | 0.3          |
| Mic        | 1.0          | 0.25         |
| Ship       | 0.1          | 0.6          |
| Ship       | 0.2          | 0.55         |
| Ship       | 0.3          | 0.5          |
| Ship       | 0.4          | 0.45         |
| Ship       | 0.5          | 0.4          |
| Ship       | 0.6          | 0.35         |
| Ship       | 0.7          | 0.3          |
| Ship       | 0.8          | 0.25         |
| Ship       | 0.9          | 0.2          |
| Ship       | 1.0          | 0.15         |
</details>

Figure 8: A t-SNE plot of the scene-specific representations in NeRF Synthetic's scenes.

Table 8: Ablation studies for rendering resolutions. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">LLFF</td><td colspan="3">DTU</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>InsertNeRF(Original Res)</td><td>26.22</td><td>0.838</td><td>0.184</td><td>29.88</td><td>0.929</td><td>0.096</td></tr><tr><td>InsertNeRF(1/2 Res)</td><td>26.44</td><td>0.844</td><td>0.169</td><td>29.75</td><td>0.925</td><td>0.077</td></tr></table>

# D.2 SCENE REPRESENTATION ANALYSIS

The HyperNet modules instill generalizability into NeRF by generating scene-specific weights in the original framework. To verify this, we visualize the intermediate representations of InsertNeRF through a t-SNE plot. As shown in Fig. 3(b), it is noteworthy that the reduced-dimensional features exhibit scene clustering characteristics in LLFF evaluation data, which may be attributed to the dynamic MLPs and activation functions in our HyperNet modules. NeRF Synthetic also exhibits a similar trend. In Fig. 8, although the scenes still possess clustering characteristics, they appear relatively more dispersed, which may be attributed to significant disparities between evaluation viewpoints in the NeRF Synthetic.

# D.3 WHY INSERTNERF DESIGNS COULD HELP NERF GENERALIZATION

Vanilla NeRF can be considered as an implicit representation used to depict a scene through the parameters of a neural network, i.e. $\Theta$ , as described in Eq. (1)

A natural idea is how to alter $\Theta$ for different scenes s, so that $\Theta_{s}$ possesses the ability to represent this new scene. By sampling different $s \in S$ and generating different $\Theta_{s}$ , this can be considered as endowing vanilla NeRF representation with generalizability in multi-scenes. However, unlike explicit 3D representations such as voxels, meshes, and point clouds, constructing an implicit representation $\Theta_{s}$ directly for a given s is challenging.

Therefore, in this paper, we introduce the HyperNet modules, which is invented to generate weights for a target neural network, to address this issue. Through two types of the HyperNet modules we propose, scene-customization weights (parameters) $\Omega_{T}^{s}$ in the NeRF framework are generated in a given s. Here, we predict $\Omega_{T}^{s}$ by combining the feature extraction from reference images and the multi-Layer dynamic-static aggregation strategy. Finally, by combining $\Theta$ and new weights $\Omega_{T}^{s}$ within the NeRF framework, we obtain $\Theta_{s}$ that can adapt to different scenes s, as described in Sec. 3.2

# E ADDITIONAL RESULTS AND ANALYSIS

# E.1 ADDITIONAL ABLATION STUDIES

Rendering Resolutions: During the evaluation phase, prior works employed different rendering resolutions, which has some impact on the metric. To investigate this issue, we evaluate the rendering performance at different resolutions without altering the training settings. In Tab. 8, reducing the rendering resolution not only improved rendering efficiency but also demonstrated performance enhancements in LLFF. However, in the DTU dataset, a contrasting trend is evident, which may be attributed to its emphasis on foreground rendering.

Parallel Branches Z: We further analyze the influence of the number of parallel branches Z on network rendering performance, as shown in Tab. 9. When Z = 1, i.e., $F_{app}$ is replaced by $F_{geo}$ with the same input, a significant performance drop occurs. This might be attributed to the implicit modeling of BRDF prior by $F_{app}$ . Furthermore, with an increase in the number of branches, $F_{app}$ endows the NeRF framework with enhanced fine-detail generalizability.

Table 9: Ablation studies for $F_{app}$ . 

<table><tr><td>Z</td><td>PSNR↑</td><td>DTU SSIM↑</td><td>LPIPS↓</td></tr><tr><td> $Z = 1$  ( $\mathcal{F}_{geo}$ )</td><td>29.03</td><td>0.918</td><td>0.086</td></tr><tr><td> $Z = 2$ </td><td>29.75</td><td>0.925</td><td>0.077</td></tr><tr><td> $Z = 3$ </td><td>29.83</td><td>0.925</td><td>0.075</td></tr></table>

Table 10: Comparisons of HyperNet modules against SOTA methods on ShapeNet. 

<table><tr><td rowspan="2">Methods</td><td colspan="2">Chairs 1-view</td><td colspan="2">Chairs 2-views</td><td colspan="2">Cars 1-view</td><td colspan="2">Cars 2-views</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>ENR (Dupont et al., 2020)</td><td>22.83</td><td>-</td><td>-</td><td>-</td><td>22.26</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SRN (Sitzmann et al., 2019)</td><td>22.89</td><td>0.89</td><td>24.48</td><td>0.92</td><td>22.25</td><td>0.89</td><td>24.84</td><td>0.92</td></tr><tr><td>ViewFormer (ECCV 2022)</td><td>14.74</td><td>0.79</td><td>17.20</td><td>0.84</td><td>19.03</td><td>0.83</td><td>20.09</td><td>0.85</td></tr><tr><td>pixelNeRF (CVPR 2021)</td><td>23.72</td><td>0.91</td><td>26.20</td><td>0.94</td><td>23.17</td><td>0.90</td><td>25.66</td><td>0.94</td></tr><tr><td>pixelNeRF+HyperNet modules</td><td>24.51</td><td>0.92</td><td>26.71</td><td>0.94</td><td>24.18</td><td>0.91</td><td>26.05</td><td>0.95</td></tr></table>

# E.2 RESULTS FOR SHAPENET

In this section, we explore the performance of the our InsertNeRF in ShapeNet under Chairs and Cars scenes. Due to the primary emphasis of InsertNeRF on multi-view settings I and II, the validation for multi-layer dynamic-static aggregation strategy under few-views settings is unnecessary. Therefore, we integrate the HyperNet modules into the original pixelNeRF Yu et al. (2021), altering its training inputs accordingly. As shown in the Tab. 10, our modules exhibit significant improvements compared to pixelNeRF Yu et al. (2021), especially in the 1-view setting. It's also evident that compared to NeRF Synthetic, LLFF, and DTU, InsertNeRF shows less improvement on ShapeNet. This might be due to the relatively simplistic appearance and geometry of ShapeNet-Scenes, and our work primarily focuses on multi-view settings as mentioned in (Kulhánek et al., 2022).

# E.3 RESULTS FROM FINE-TUNING

We also explore the rendering performance of InsertNeRF after fine-tuning in various scenes. In contrast to the fine-tuning methodology adopted by (Liu et al., 2022), we fine-tune directly on the pre-trained model. Tab. 11 presents the performance of InsertNeRF across different scenes and the results after fine-tuning in NeRF Synthetic.

# E.4 SINGLE SCENE RESULTS

Existing works (Wang et al., 2022) also tend to focus on single-scene rendering within the framework of GNeRF. We conduct a quantitative comparison with existing works in the single-scene setting and achieve satisfactory performance, as shown in Tab. 12.

# E.5 MORE QUALITATIVE RESULTS

We present additional qualitative results to further analyze the superiority of InsertNeRF. i). Fig. 9 and Fig. 10 report qualitative results in LLFF and NeRF Synthetic. ii). Fig. 11 showcase more Color-Depth results in LLFF and DTU datasets. iii). We also qualitatively analyze the generalizability of the InsertNeRF-systems including Insert-mip-NeRF Fig. 12, and Insert-NeRF++ Fig. 13. Note that the presence of color distortion in the Insert-NeRF++'s foreground rendering is observed, yet it does not impact the combined results, possibly attributable to the replaced sampling process.

# F LIMITATIONS

In the majority of scenarios, a higher number of sample points along the rays often leads to improved rendering performance. In essence, thanks to the transformer architecture, existing works (Wang et al., 2022; 2021) can be trained on a limited number of sample points (64 training samples) and evaluated on the more sample points (192 evaluation samples), resulting in elevated training efficiency and improved rendering performance. However, in the InsertNeRF-system, it is essential to maintain consistency in the number of sample points between the training and evaluation processes. In order to ensure fairness in comparative experiments and strike the trade-off between training efficiency and rendering performance, we set K = 64 both during training and evaluation, which imposes certain limitations on the rendering performance. Naturally, as we increase the number of sample points for training, the rendering performance will further improve.

Table 11: The performance in different scenes and the results after Fine-Tuning in NeRF Synthetic.   
(a) PSNR 

<table><tr><td>Method</td><td>FT</td><td>Lego</td><td>Chair</td><td>Drums</td><td>Ficus</td><td>Hotdog</td><td>Materials</td><td>Mic</td><td>Ship</td><td>Avg.</td></tr><tr><td>InsertNeRF</td><td></td><td>30.00</td><td>31.97</td><td>25.59</td><td>26.08</td><td>35.04</td><td>28.91</td><td>35.13</td><td>30.08</td><td>30.35</td></tr><tr><td>MVSNeRF</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>27.21</td></tr><tr><td>IBRNet</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>30.05</td></tr><tr><td>GNT</td><td>√</td><td>31.38</td><td>33.70</td><td>26.98</td><td>29.55</td><td>36.95</td><td>29.11</td><td>33.35</td><td>30.54</td><td>31.45</td></tr><tr><td>NeuRay</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>32.35</td></tr><tr><td>InsertNeRF</td><td>√</td><td>31.93</td><td>34.41</td><td>27.76</td><td>29.08</td><td>37.25</td><td>31.46</td><td>36.43</td><td>31.98</td><td>32.54</td></tr></table>

(b) SSIM 

<table><tr><td>Method</td><td>FT</td><td>Lego</td><td>Chair</td><td>Drums</td><td>Ficus</td><td>Hotdog</td><td>Materials</td><td>Mic</td><td>Ship</td><td>Avg.</td></tr><tr><td>InsertNeRF</td><td></td><td>0.939</td><td>0.969</td><td>0.910</td><td>0.915</td><td>0.969</td><td>0.922</td><td>0.972</td><td>0.888</td><td>0.936</td></tr><tr><td>MVSNeRF</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.888</td></tr><tr><td>IBRNet</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.935</td></tr><tr><td>NeuRay</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.960</td></tr><tr><td>InsertNeRF</td><td>√</td><td>0.962</td><td>0.981</td><td>0.937</td><td>0.959</td><td>0.988</td><td>0.957</td><td>0.978</td><td>0.931</td><td>0.962</td></tr></table>

(c) LPIPS 

<table><tr><td>Method</td><td>FT</td><td>Lego</td><td>Chair</td><td>Drums</td><td>Ficus</td><td>Hotdog</td><td>Materials</td><td>Mic</td><td>Ship</td><td>Avg.</td></tr><tr><td>InsertNeRF</td><td></td><td>0.056</td><td>0.033</td><td>0.079</td><td>0.085</td><td>0.037</td><td>0.076</td><td>0.029</td><td>0.129</td><td>0.066</td></tr><tr><td>MVSNeRF</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.162</td></tr><tr><td>IBRNet</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.066</td></tr><tr><td>NeuRay</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.048</td></tr><tr><td>InsertNeRF</td><td>√</td><td>0.044</td><td>0.024</td><td>0.065</td><td>0.054</td><td>0.027</td><td>0.048</td><td>0.021</td><td>0.102</td><td>0.048</td></tr></table>

Table 12: Comparison of InsertNeRF for single scene rendering on the NeRF Synthetic.   
(a) PSNR 

<table><tr><td>Method</td><td>Room</td><td>Fern</td><td>Leaves</td><td>Fortress</td><td>Orchids</td><td>Flower</td><td>T-Rex</td><td>Horns</td></tr><tr><td>LLFF</td><td>24.54</td><td>28.72</td><td>21.13</td><td>21.79</td><td>18.52</td><td>20.72</td><td>27.48</td><td>23.22</td></tr><tr><td>NeRF</td><td>32.70</td><td>25.17</td><td>20.92</td><td>31.16</td><td>20.36</td><td>27.40</td><td>26.80</td><td>27.45</td></tr><tr><td>GNT</td><td>32.96</td><td>24.31</td><td>22.57</td><td>32.28</td><td>20.67</td><td>27.32</td><td>28.15</td><td>29.62</td></tr><tr><td>InsertNeRF</td><td>32.55</td><td>24.88</td><td>22.59</td><td>31.82</td><td>21.18</td><td>28.39</td><td>27.49</td><td>29.37</td></tr></table>

(b) SSIM 

<table><tr><td>Method</td><td>Room</td><td>Fern</td><td>Leaves</td><td>Fortress</td><td>Orchids</td><td>Flower</td><td>T-Rex</td><td>Horns</td></tr><tr><td>LLFF</td><td>0.932</td><td>0.753</td><td>0.697</td><td>0.872</td><td>0.588</td><td>0.844</td><td>0.857</td><td>0.840</td></tr><tr><td>NeRF</td><td>0.948</td><td>0.792</td><td>0.690</td><td>0.881</td><td>0.641</td><td>0.827</td><td>0.880</td><td>0.828</td></tr><tr><td>GNT</td><td>0.963</td><td>0.846</td><td>0.852</td><td>0.934</td><td>0.752</td><td>0.893</td><td>0.936</td><td>0.935</td></tr><tr><td>InsertNeRF</td><td>0.961</td><td>0.846</td><td>0.853</td><td>0.925</td><td>0.756</td><td>0.904</td><td>0.928</td><td>0.932</td></tr></table>

(c) SSIM 

<table><tr><td>Method</td><td>Room</td><td>Fern</td><td>Leaves</td><td>Fortress</td><td>Orchids</td><td>Flower</td><td>T-Rex</td><td>Horns</td></tr><tr><td>LLFF</td><td>0.155</td><td>0.247</td><td>0.216</td><td>0.173</td><td>0.313</td><td>0.174</td><td>0.222</td><td>0.193</td></tr><tr><td>NeRF</td><td>0.178</td><td>0.280</td><td>0.316</td><td>0.171</td><td>0.321</td><td>0.219</td><td>0.249</td><td>0.268</td></tr><tr><td>GNT</td><td>0.060</td><td>0.116</td><td>0.109</td><td>0.061</td><td>0.153</td><td>0.092</td><td>0.080</td><td>0.076</td></tr><tr><td>InsertNeRF</td><td>0.063</td><td>0.121</td><td>0.109</td><td>0.062</td><td>0.152</td><td>0.070</td><td>0.085</td><td>0.074</td></tr></table>

![](images/899de7c3608aa7cdacada3cc03a15cc07654eb14c07f492025ffdad4d17e3e0c.jpg)  
Figure 9: Qualitative comparisons of InsertNeRF against SOTA methods under LLFF scenes.

# G FUTURE WORK

We aspire to construct an all-encompassing InsertNeRF framework, endowing generalizability into various NeRF-derived works, such as TensoRF, NeRF-, NeuS, and so forth. This can facilitate existing or future NeRF research to transcend the constraints of scene-customization.

In addition, we have provided a pre-trained model to address NeRF under sparse inputs in Sec. 4.4. While such models have demonstrated satisfactory performance without any retraining Tab. 3, we still plan to design a fine-tuning approach for sparse inputs to further enhance rendering quality.

![](images/8bd6a385c3117b254c33de5852aef572342194980a0fb1aaa1d36559f7b58235.jpg)

![](images/848b8abe76954495370efa9ba730944b1deb534c4d16de71649defd77e355e77.jpg)

![](images/0267fb08f7bd35156f85399f55984f623d13193be293272845aad864cb7d3a96.jpg)

![](images/3edb61276a6604a6529da0ccd8ddbd88eb975afa9e4e89956ab2380b3bccc93d.jpg)

![](images/c80774683f59310b25b9da3410e7369bc8b2f8fc258d45b4a330f499874ace13.jpg)

![](images/2c9b769d70e42a752fe01380a458610c3fee0e7595f39a5bfd4b34870bf68f26.jpg)

![](images/fbd914805a5b94dd5be63d82d1fda174db2ef8bdc815d4d56167bd788e83394b.jpg)

![](images/2ca96f7a9c5a0f96a9ed981a51a2ea3115af3ace258e8e1e554655f8a79d309f.jpg)

![](images/737daaa8bd4d75b03bfd58ea7a734f55483f6cb0494cb15f0bd72dc75af40cd3.jpg)

![](images/d461e4595eb0de5581966d1d8cd8ebff7d27443b44dc6eb127fdbb81f4a4e1cc.jpg)

![](images/fa1c9975b33d776d3adec0480cecfa65fe91831b017ac1157668c62b40b93d25.jpg)

<details>
<summary>natural_image</summary>

Close-up of a metallic spherical object with a stylized '6' and circular indentations, possibly a button or knob (no text or symbols visible)
</details>

![](images/d4c651d7b3ceb92e0ef283b5952be438a5b0bf7457832cc77f1a41c76dd65e05.jpg)

<details>
<summary>natural_image</summary>

Close-up of a metallic, glossy, spherical object with a central hole and textured surface (no visible text or symbols)
</details>

![](images/7ac86b752a6f2013494e88f8b926a08eb09009ed835361f212868ac6573291a0.jpg)

<details>
<summary>natural_image</summary>

Close-up of a metallic, spherical object with a circular opening and textured surface (no visible text or symbols)
</details>

![](images/8b2a205e42b8676eb9a674b533d00e5651da8e305727b33a8d83bb189a7cfd32.jpg)

<details>
<summary>natural_image</summary>

Close-up of a metallic helmet with visible internal components and surface debris (no text or symbols)
</details>

![](images/a5adefab52d99caf7778957f511dbfcf2581c84d544a786cc3b7bee2b9cc845e.jpg)

<details>
<summary>natural_image</summary>

Close-up of a metallic, glossy, spherical object with a circular opening and textured surface (no visible text or symbols)
</details>

NeuRay   
GNT   
InsertNeRF   
Figure 10: Qualitative comparisons of InsertNeRF against SOTA methods under NeRF Synthetic scenes.

GoundTruth   
IBRNet   
![](images/c42c2f7bac0ebf5c7b8b62a8dcbbb0b70c9af57a48c1c532816b7a1a1bd759f7.jpg)

![](images/1abe4afbcc0f4e8b15ca236d8c4ff3114df9953d935cf8c3c5483b8c775b6556.jpg)

![](images/dcd014b1f42f82292f424f62fb0524cc46f6067f7375ecd842bf79039a11a931.jpg)

![](images/20a521e588917a9154cdd9c2baf65c021567f4c2b3b505a9b7591d2c15d699ef.jpg)

![](images/71b3ead96dd3443b68b8ad91a0bc781659ce15f2b0432d59700a3c620e75b3bd.jpg)

![](images/b994531520bd673396c315794aad4b79e234a6045524f684317def0c1549fe9f.jpg)

![](images/b3aae3e4b0e75693213f5d365c915b8a3da23bf66816adc3a05be2106d373702.jpg)

![](images/853dca07b3530a4147bc5fa1566957513e97be45967193158e65f690dc7fbf53.jpg)  
IBRNet   
InsertNeRF

![](images/0279710b1e95b4b63e5505777237fd1b0a235ad37ebdc3c0d5a3efb485e4633b.jpg)

![](images/0a8dc193f688217cc86caff4d046608de11e29fad4e705b066a2db9f480c4b9e.jpg)

![](images/edc18f3f3a197cfba28b86d5971585a65e1fa94dce766195efe1820279022e1d.jpg)

![](images/e32093461880318e9eada915e1dc16630b5fcee54dfe6dd240345a208609c229.jpg)

![](images/1a6d3fd1191545c7eb480b4fac1c4d3c5a26e302aca36dd4cd23f605c478d85a.jpg)

![](images/1d9e947279b800010ac8c70a4550cd84ca9ffd8e7395b99eaf8d0afe8afe8845.jpg)

![](images/327ae753dbca692001f93337f73afbcbb42382e25d2350c68d513955ebc04351.jpg)

![](images/34a37dfe29603a94b21db08dee81e151f7c37cb2c20320106258f353a6b223c3.jpg)  
NeuRay   
InsertNeRF   
Figure 11: Color-Depth results of InsertNeRF with existing works under LLFF and DTU's scenes.

![](images/057cdbb8b59ec7b23fb5b0f0c667e4d135982007b332fe00ed1acb8660a6290a.jpg)

Figure 12: Qualitative results of Insert-mip-NeRF in multi-scale NeRF Synthetic.   
![](images/53db9f7cf11c5977ab58aab16a23304cf6164c47d702f0b9d4e26d9d24315c14.jpg)

![](images/2c3f12f5fedf3d36b62473a411b632f37420a1ee813a3b5598f071f511738e53.jpg)

<details>
<summary>natural_image</summary>

Three side-by-side photos of a green CRICFC train with visible front and rear panels, no text or symbols present.
</details>

![](images/d8aed5f915d514aef2ece04d4e66b8fed3cc285463370a8499bac5ad78507be2.jpg)

![](images/54a5f535d4aed65db9e80161cd79d70c79439754c7afcb0884966769f2758012.jpg)

![](images/038b538702cd6c47633fd28bb3116b4f821856936644ab40c4b2f4862a11c35e.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo collage showing a person in a wheelchair near a building, with no visible text or symbols.
</details>

![](images/d0c9ed180d433fde11e62f5d57b3ea626d94bf3f754628db17a7dcce60f96c57.jpg)

![](images/0479d1f391fe7015a31792d844173dd5c65aab5828017c58b0962a52792449b9.jpg)

![](images/5ba8ada13cad0893fc6f6005cf85a6cc95e70dbe9875b15a291dad14902ba0bf.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo collage showing outdoor fitness equipment and outdoor park area with trees (no visible text or symbols)
</details>

![](images/3dec80a5719c7adb058e98f54962b69c935dc2efdf1c93cb7813b9af219a9fcd.jpg)

![](images/d34d1073aeabce7aa53377ad35971229ff45368c4aacae95fbf6c7348cafdafc.jpg)

![](images/2461a61e73138c84aaeac8855a8c3647b4f2b1940bda58ff43689059d10b4435.jpg)

<details>
<summary>natural_image</summary>

Three side-by-side photos of a light blue truck with cargo boxes, parked outdoors under trees (no visible text or symbols)
</details>

![](images/f9d137aa056d282edcd806a000935a91b3529f62115a5a51cb98258c0793aeb0.jpg)  
Baseline NeRF++   
Foreground   
Background
Insert-NeRF++   
Combined   
GoundTruth   
Figure 13: Qualitative results of Insert-NeRF++ in Tanks and Temples.