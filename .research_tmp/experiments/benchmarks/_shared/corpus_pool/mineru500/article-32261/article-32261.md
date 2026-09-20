# What to Preserve and What to Transfer: Faithful, Identity-Preserving Diffusion-based Hairstyle Transfer

Chaeyeon Chung, Sunghyun Park, Jeongho Kim, and Jaegul Choo

KAIST AI, South Korea

cy\_chung@kaist.ac.kr, psh01087@gmail.com, {rlawjdghek, jchoo}@kaist.ac.kr

![](images/e8899a21380a1fd1018c34a0dd171fac4efe0f50cfe8e43df35829d561b62458.jpg)  
Figure 1: Generated results by HairFusion using in-the-wild images. Given a pair of a face image and a reference hairstyle image, our method can generate high-fidelity images. These results show our model's generalizability to diverse face images.

# Abstract

Hairstyle transfer is a challenging task in the image editing field that modifies the hairstyle of a given face image while preserving its other appearance and background features. The existing hairstyle transfer approaches heavily rely on StyleGAN, which is pre-trained on cropped and aligned face images. Hence, they struggle to generalize under challenging conditions such as extreme variations of head poses or focal lengths. To address this issue, we propose a one-stage hairstyle transfer diffusion model, HairFusion, that applies to real-world scenarios. Specifically, we carefully design a hair-agnostic representation as the input of the model, where the original hair information is thoroughly eliminated. Next, we introduce a hair align cross-attention (Align-CA) to accurately align the reference hairstyle with the face image while considering the difference in their head poses. To enhance the preservation of the face image's original features, we leverage adaptive hair blending during the inference, where the output's hair regions are estimated by the cross-attention map in Align-CA and blended with non-hair areas of the face image. Our experimental results show that our method achieves state-of-the-art performance compared to the existing methods in preserving the integrity of both the transferred hairstyle and the surrounding features. The codes are available at https://github.com/cychungg/HairFusion.

# Introduction

With the recent advancement of generative models (Rombach et al. 2022; Saharia et al. 2022), image editing technologies (Zhang, Rao, and Agrawala 2023; Yang et al. 2023a; Hertz et al. 2022) have shown impressive results. The task of hairstyle transfer is one of the most challenging image editing tasks, focusing on modifying an input face image's hairstyle to the reference hairstyle. This technology can offer users a preview of how different hairstyles would look on them, enhancing customer satisfaction. The key challenge of hairstyle transfer is to transfer the reference hairstyle to the face while preserving other features (e.g., identity, clothing, and background) in the face image.

Recent hairstyle transfer (Kim et al. 2022; Khwanmuang et al. 2023; Wei et al. 2023; Nikolaev et al. 2024) allows users to manipulate hairstyles using StyleGAN2 (Karras et al. 2020), which can generate high-fidelity face images. However, such GAN-based approaches still suffer from several challenges. First, since most of them rely on StyleGAN2 pre-trained on FFHQ dataset (Karras, Laine, and Aila 2019), they often fail to generalize to face images with various head poses or focal lengths that lie outside the latent space of StyleGAN2. This hinders its application to real-world sce-

narios (Yang et al. 2022, 2023b) (see Fig. 5). Moreover, previous latent optimization methods have limitations in preserving fine-grained details of reference hairstyles (e.g., curl), especially under extreme pose variations.

On the other hand, recent diffusion-based image editing demonstrates superior performance in manipulating human images compared to GANs. Thanks to the powerful generative prior of text-to-image diffusion models, there has been success in image editing based on human poses (Hu 2024; Kim et al. 2024b), depth (Zhang, Rao, and Agrawala 2023), reference images (Yang et al. 2023a), and clothing (Kim et al. 2024a; Choi et al. 2024). However, it is still underexplored to leverage diffusion models for hairstyle transfer.

To address the above-mentioned challenges, we propose a novel hairstyle transfer diffusion model called HairFusion, which generates a high-fidelity image with the reference hairstyle while faithfully preserving the surrounding features such as head shape, clothing, and backgrounds. We design a one-stage diffusion hairstyle transfer model building upon a pre-trained diffusion model (Rombach et al. 2022) to enhance generalizability on various face images. Inspired by recent literature on virtual try-on (Choi et al. 2021; Lee et al. 2022; Kim et al. 2024a), we posit that hairstyle transfer can be conceptualized as a process of filling in the hair region of a face image, conditioned on the reference hairstyle (i.e., exemplar-based image inpainting). Adapting the diffusion-based image inpainting for the hairstyle transfer, we introduce a hair-agnostic representation and a hair align cross-attention (Align-CA). We first obtain a hair-agnostic representation by thoroughly eliminating the original hair information in the face image. Next, the Align-CA aligns the reference hairstyle with the hair region of the face image by learning the correspondence between them. Dense pose representations (Guler, Neverova, and Kokkinos 2018) are provided to the Align-CA to encourage the model to indicate the relative difference in pose and face shape between reference hair and face images.

Nevertheless, when editing the hairstyle via diffusion-based inpainting, challenges remain in effectively preserving essential original features such as identity, clothing, and background of the source face image. To solve this issue, we introduce adaptive hair blending. We exploit a cross-attention map to identify hair regions in the output and seamlessly blend them with non-hair areas of the face image during the inference. Our extensive experiments demonstrate that our method significantly outperforms the existing approaches in terms of synthesized image quality both qualitatively and quantitatively. Also, our method achieves superior performance than the existing diffusion models for exemplar-based image inpainting, even when applied to diverse, real-world images, as presented in Fig. 1. We summarize our contributions as follows:

- We present HairFusion, the first one-stage diffusion-based hairstyle transfer framework, capable of generalizing across arbitrary face images in real-world scenarios.   
- We propose a hair align cross-attention module (Align-CA) that aligns the reference hairstyle with the face image by accounting for differences in head poses.

\- Our novel adaptive hair blending effectively preserves the original features in non-hair regions by blending the estimated hair regions of the generated image with the non-hair regions of the face image during inference.

# Related Work

Diffusion-based Image Editing. Text-to-image diffusion models (Saharia et al. 2022; Rombach et al. 2022) have demonstrated remarkable success in generating high-fidelity images based on textual descriptions. Several approaches have been proposed for image manipulation under diverse conditions, incorporating conditions such as pose (Hu 2024; Kim et al. 2024b), reference images (Yang et al. 2023a; Chen et al. 2024), and multiple conditions (Zhang, Rao, and Agrawala 2023; Kim et al. 2023) like depth maps and edge maps. Furthermore, several studies have extended to image inpainting based on the reference images, including objects (Yang et al. 2023a; Chen et al. 2024) or clothing (Kim et al. 2024a). However, hairstyles, compared to other objects like clothing, involve more diverse inpainting regions due to the high variability in hair width and length across different styles. This study seeks to broaden the applicability of diffusion models by using them to transfer the hairstyle from the reference image to a source face image.

Hairstyle Transfer. The existing hairstyle transfer approaches exploit GANs to apply the reference hairstyle to the face image. MichiGAN (Tan et al. 2020) employed conditional generators incorporating hair attributes such as hair shape and appearance. LOHO (Saha et al. 2021) and Barbershop (Zhu et al. 2021) introduced latent optimization techniques based on pre-trained StyleGAN (Karras et al. 2020) to preserve the overall structure of the face image while reflecting the reference hairstyle. To broaden the scope of hairstyle transfer applications, HairFIT (Chung et al. 2021), StyleYourHair (Kim et al. 2022), StyleGANSalon (Khwanmuang et al. 2023), and HairFastGAN (Nikolaev et al. 2024) have developed the pose-invariant hairstyle transfer, accommodating significant differences in head pose between the face and reference images. Moreover, HairCLIP (Wei et al. 2022) and HairCLIPv2 (Wei et al. 2023) interactively edit hairstyles based on multiple conditions, such as text descriptions, reference images, masks, and sketches. On the other hand, HairNeFR (Chang, Kim, and Kim 2023) leverages neural rendering to geometrically align target hair in the volumetric space. However, due to the capability of pretrained StyleGAN, most previous methods still have limitations in generalizing to the face images with various head poses or focal length and preserving fine-grained details of reference hairstyles (i.e., texture and curl). In this paper, we introduce a novel diffusion-based hairstyle transfer model performing on multi-view images, enhancing the applicability of hairstyle transfer in real-world scenarios.

# Method

# Preliminary

Large-scale diffusion probabilistic models have shown promising performance in image generation. Diffusion models (Ho, Jain, and Abbeel 2020) learn to generate images

![](images/13911c50b3e00b745847224ffa8753a1b54003dcc23d5fc8ec2728a02b91a3ce.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Ref. Hair"] --> B["Extract Hair"]
    B --> C["x_hair"]
    C --> D["CLIP Image Encoder"]
    D --> E["Hair-Agnostic Representation Preprocess"]
    E --> F["m_hair ∪ f_agn = m_agn ⊙ x = x_agn"]
    F --> G["z_t"]
    G --> H["C"]
    H --> I["ε(x_agn)R(m_agn)ε(f_agn)"]
    I --> J["ε(x_hair)"]
    J --> K["+"]
    K --> L["Align-CA"]
    L --> M["Pose Encoder"]
    M --> N["p_hair"]
    N --> O["Cross Attention"]
    O --> P["V"]
    P --> Q["Self Attention"]
    Q --> R["Q"]
    R --> S["Cross Attention"]
    S --> T["Zero Convolution"]
    T --> U["(b) Align-CA"]
    
    subgraph "(a) Overview of HairFusion"
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
    end
    
    subgraph "(b) Align-CA"
        L
        M
        N
        O
        P
        Q
        R
        S
        T
    end
    
    subgraph "Hair-Agnostic Representation Preprocess"
        E
        F
        G
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
    end
    
    H -->|C| I
    I -->|+| J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    
    style H fill:#f9f,stroke:#333
    style I fill:#ccf,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#cff,stroke:#333
    style M fill:#ffc,stroke:#333
    style N fill:#cfc,stroke:#333
    style O fill:#fcc,stroke:#333
    style P fill:#ffc,stroke:#333
    
    subgraph "(a) Overview of HairFusion"
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
        M --> N
        N --> O
        O --> P
    
    subgraph "(b) Align-CA"
        L --> Q
        Q --> R
        R --> S
        S --> T
    
    subgraph "(c) Overview of HairFusion"
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
    
    end
    
    subgraph "(d) Align-CA"
        Q --> R
        R --> S
        S --> T
    
    style H fill:#fff,stroke:#333,stroke-width:2px
    style I fill:#fff,stroke:#333,stroke-width:2px
    style J fill:#fff,stroke:#333,stroke-width:2px
    style K fill:#fff,stroke:#333,stroke-width:2px
    style L fill:#fff,stroke:#333,stroke-width:2px
    style M fill:#fff,stroke:#333,stroke-width:2px
    style N fill:#fff,stroke:#333,stroke-width:2px
    style O fill:#fff,stroke:#333,stroke-width:2px
    style P fill:#fff,stroke:#333,stroke-width:2px
    
    subgraph "(e) Overview of HairFusion"
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
    
    subgraph "(f) Align-CA"
        Q --> R
        R --> S
        S --> T
    
    subgraph "(g) Align-CA"
        S --> Q
        Q --> R
    
    end
    
    subgraph "(h) Align-CA"
        R --> S
    
    subgraph "(i) Align-CA"
        S --> T
    
    subgraph "(j) Align-CA"
        T --> R
    
    subgraph "(k) Align-CA"
        R --> S
    
    subgraph "(l) Align-CA"
        S --> T
    
    subgraph "(m) Align-CA"
        T --> R
    
    end
    
    subgraph "(n) Align-CA"
        S --> T
    
    subgraph "(o) Align-CA"
        T --> R
    
    subgraph "(p) Align-CA"
        S --> T
    
    subgraph "(q) Align-CA"
        T --> R
    
    subgraph "(r) Align-CA"
        S --> T
    
    subgraph "(s) Align-CA"
        T --> R
    
    end
    
    subgraph "(t) Align-CA"
        S --> T
    
    subgraph "(u) Align-CA"
        T --> R
    
    end
    
    subgraph "(v) Align-CA"
        S --> T
    
    subgraph "(w) Align-CA"
        T --> R
    
    end
    
    subgraph "(x) Align-CA"
        S --> T
    
    subgraph "(y) Align-CA"
        T --> R
    
    end
    
    subgraph "(z) Align-CA"
        S --> T
    
    subgraph "(w) Align-CA"
        T --> R
    
    end
    
    subgraph "(x_w) Align-CA"
        S --> T
    
    subgraph "(y_w) Align-CA"
        T --> R
    
    end
    
    subgraph "(z_w) Align-CA"
        S --> T
    
    subgraph "(x_w) Align-CA"
        T --> R
    
    end
```
</details>

Figure 2: Overall pipeline of HairFusion. (a) HairFusion consists of a pre-trained U-Net, a hair encoder, a hair align cross-attention (Align-CA), and a pose encoder. We first preprocess hair-agnostic image $x_{agn}$ using a hair mask $m_{hair}$ and a face outline image $f_{agn}$ . Then, we provide $x_{agn}$ , a hair-agnostic mask $m_{agn}$ , a hair image $x_{hair}$ and dense pose images $p_{agn}$ , $p_{hair}$ as inputs to the model. (b) To inject $x_{hair}$ into $x_{agn}$ , we leverage the Align-CA, which aligns the hair features with the face features via cross-attention. Here, the pose features are added to the query (Q) and the key (K) as additional guidance.

from a target data distribution by progressively denoising a normally distributed variable. Stable Diffusion (SD) (Rombach et al. 2022) is a latent diffusion model (LDM) that performs the denoising process in the latent space of an autoencoder. To be specific, a pre-trained encoder ( $E$ ) first transforms an input image x to latent feature $\mathbf{z}_{0} = \mathcal{E}(\mathbf{x})$ . Then, a forward diffusion process is performed with a pre-defined variance schedule $\beta_{t}$ , following denoising diffusion probabilistic models (Ho, Jain, and Abbeel 2020):

$$
q (\mathbf {z} _ {t} | \mathbf {z} _ {0}) = \mathcal {N} (\mathbf {z} _ {t}; \sqrt {\bar {\alpha} _ {t}} \mathbf {z} _ {0}, (1 - \bar {\alpha} _ {t}) \mathbf {I}), \tag {1}
$$

where $t \in \{1, \ldots, T\}$ and T denotes the number of steps in the forward diffusion process. $\alpha_{t}$ is defined to be $1 - \beta_{t}$ and $\bar{\alpha}_{t}$ to be $\Pi_{s=1}^{t}\alpha_{s}$ . SD is trained with the following objective function:

$$
\mathcal {L} _ {L D M} = \mathbb {E} _ {\mathcal {E} (\mathbf {x}), \mathbf {y}, \epsilon \sim \mathcal {N} (0, 1), t} \left[ \| \epsilon - \epsilon_ {\theta} \left(\mathbf {z} _ {t}, t, \tau_ {\theta} (\mathbf {y})\right) \| _ {2} ^ {2} \right], \tag {2}
$$

where $\epsilon_{\theta}(\cdot)$ is a denoising U-Net (Ronneberger, Fischer, and Brox 2015) and $\tau_{\theta}(\cdot)$ is a CLIP (Radford et al. 2021) text encoder that embeds the text prompt y.

# Overview

We address hairstyle transfer from the perspective of exemplar-based image inpainting. HairFusion aims to newly generate the masked hair region of a source face image $x \in R^{3 \times H \times W}$ according to the reference hairstyle $x_{hair} \in R^{3 \times H \times W}$ . As presented in Fig. 2(a), we first preprocess a hair-agnostic image $x_{agn} \in R^{3 \times H \times W}$ that faithfully removes the original hair and potential hair regions for various reference hairstyles while preserving the regions that need to be reconstructed. Then, HairFusion takes $x_{agn}$ , $x_{hair}$ , and face outline image $f_{agn} \in R^{3 \times H \times W}$ as the input during denoising. Following Paint-by-Example (Yang et al. 2023a), we inject the class token of $x_{hair}$ extracted from a pretrained CLIP image encoder into the denoising U-Net as an exemplar condition. Additionally, HairFusion leverages a hair encoder to extract detailed spatial features of $x_{hair}$ . We align the features of $x_{hair}$ with $x_{agn}$ via a hair align cross-attention (Align-CA). The Align-CA is designed to consider the relative difference of head poses, focal lengths, and face shape between $x_{agn}$ and $x_{hair}$ based on their dense pose features. We obtain dense pose features by injecting $p_{agn}$ and $p_{hair}$ into a pose encoder $E_{p}$ . During the inference, our newly proposed adaptive hair blending enhances the preservation of the original features in the source face image, which are largely occluded by $m_{agn}$ .

# Hair-Agnostic Representation

To apply various reference hairstyles, all the potential hair regions in the source face x need to be masked as an inpainting region. Also, the remaining hairstyle information in x can harm the model's generalization ability at inference. To address this issue, we design a hair-agnostic representation $x_{agn}$ that eliminates potential hair regions for $x_{hair}$ and the original hair of x while preserving the regions that need to be maintained. As illustrated in the top-right of Fig. 2, we first zero out the hair region of x according to its hair segmentation mask $m_{hair}$ . Next, we remove the potential hair region based on the face outline image $f_{agn}$ . To be specific, we remove the area above the eyebrows and the region extending from the leftmost to the rightmost coordinate along

the jawline with additional margins. We preserve the face region that needs to be accurately reconstructed. Note that we also preserve the neck and body beneath the chin, where hair typically does not exist. More details are illustrated in the supplementary material. Our hair-agnostic representation enables the model to process a wide variety of reference hairstyles while maintaining the core identity features.

# Hair Align Cross-Attention

We leverage the cross-attention (CA) to align the reference hairstyle $x_{hair}$ with $x_{agn}$ . As shown in Fig. 2(b), our Align-CA exploits reference hairstyle features as the key (K) and value (V), where the output of the followed self-attention layer is given as the query (Q). We extract intermediate features of the reference hairstyle using a trainable hair encoder whose initial weights are copied from the denoising U-Net.

Since $x_{hair}$ can vary in face shapes and head poses, it is challenging to accurately determine the hair shape or length based solely on $x_{hair}$ . For example, even if the hair in $x_{hair}$ appears to be long, it would be short hair if the reference's face shape is elongated. Therefore, we add the features of dense pose images, $p_{agn}$ and $p_{hair}$ , to Q and K in Align-CA as follows:

$$
\text { Align - CA } = \operatorname{softmax} \left(\frac {\left(\mathbf {Q} + \mathcal {E} _ {p} (\mathbf {p} _ {a g n})\right) \cdot \left(\mathbf {K} + \mathcal {E} _ {p} (\mathbf {p} _ {h a i r})\right) ^ {T}}{\sqrt {d}}\right) \cdot \mathbf {V}, \tag {3}
$$

where $E_{p}$ is a pose encoder consisting of a stack of convolutional layers and d indicates the dimension of the feature. In this way, we can assist our model to accurately approximate the hair shape and length in the generated image in relation to the face based on the correspondence between $x_{hair}$ and $x_{agn}$ . We replace a few CA layers in the denoising U-Net decoder with our Align-CA.

# Training Strategy

For training, we use a multi-view dataset to encourage the model to learn hair alignment across various head poses. Ideally, training samples consist of pairs with the same identity but different hairstyles and head poses. Since no existing datasets include such pairs, we utilize a multi-view dataset as an alternative to obtain pairs with the same identity and hairstyle but different head poses. During the training, one pair is given as x, and the other's hairstyle is given as $x_{hair}$ .

We train the hair encoder, Align-CA, and pose encoder $(\mathcal{E}_{p})$ of HairFusion with the following objective:

$$
\mathcal {L} _ {L D M} = \mathbb {E} _ {\zeta , \epsilon \sim \mathcal {N} (0, 1), t} \left[ \| \epsilon - \epsilon_ {\theta} (\zeta , t, \tau_ {\phi} (\mathbf {x} _ {h a i r}), \mathcal {E} (\mathbf {x} _ {h a i r}) \| _ {2} ^ {2} \right], \tag {4}
$$

where

$$
\zeta = \left(\left[ \mathbf {z} _ {t}; \mathcal {E} \left(\mathbf {x} _ {a g n}\right); \mathcal {R} \left(\mathbf {m} _ {a g n}\right); \mathcal {E} \left(\mathbf {f} _ {a g n}\right) \right], \mathcal {E} _ {p} \left(\mathbf {p} _ {a g n}\right), \mathcal {E} _ {p} \left(\mathbf {p} _ {\text {hair}}\right)\right), \tag {5}
$$

$\tau_{\phi}$ is a CLIP image encoder, and $\mathcal{R}$ indicates resize function.

# Adaptive Hair Blending

One major challenge in hairstyle transfer is to preserve the original features in the source face image except for hair. However, the occluded region in $x_{agn}$ inevitably changes due to the autoencoder's reconstruction error, degrading performance. As a naive solution, one can apply latent blending as in Blended Diffusion (Avrahami, Lischinski, and Fried 2022) using a hair-agnostic mask $m_{agn}$ as a blending mask. However, since $m_{agn}$ is designed to eliminate all the potential hair regions of the generated image, using $m_{agn}$ as a blending mask unnecessarily loses useful information of x.

![](images/7d91ff5e16eebc5cb338cfbd0643e940404bb017eaa91e2082ff7cb102dbf12c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Extract CA Mask"] --> B["Input Image z_t+1"]
    A --> C["Input Image z_t"]
    A --> D["Output Image 1 - m_ca"]
    D --> E["Representation of the first image"]
    E --> F["Representation of the second image"]
    F --> G["Output Image 1 - m_hair"]
    G --> H["Representation of the third image"]
    H --> I["Representation of the fourth image"]
    I --> J["Output Image m_blend"]
```
</details>

Adaptive Hair Blending

![](images/30c0cd79f83576179f8a505238ad293ee05e194f0aff4b4029abcc314fcce22c.jpg)  
$\mathbf{z}_t^{\mathrm{x}}\odot \mathbf{m}_{blend}$

![](images/bdc048f7e220d49f7d314272464943703e32e26ad96d333dac37385fd84ee5e3.jpg)  
$\mathbf{z}_t\odot (1 - \mathbf{m}_{blend})$

![](images/5c41e8fc6f7662833c6d5ed0dc6a647392fa2f3ab7ef0469ed3a0bf9b7c89e09.jpg)  
$\mathbf{z}_t$   
Figure 3: Overview of adaptive hair blending. We obtain $m_{blend}$ using $m_{ca}$ extracted from CA maps in Align-CA and the source hair mask $m_{hair}$ . $m_{blend}$ blends the generated hair features with the other features in the source.

To address this issue, we propose an adaptive hair blending that maximally preserves the original features in x. Since our Align-CA learns to align $x_{hair}$ with $x_{agn}$ , the aligned hair regions of the generated image are expected to be activated in the CA maps of Align-CA. With this motivation, we derive an adaptive blending mask ( $m_{blend}$ ) by leveraging the CA mask ( $m_{ca}$ ) which indicates the hair region of the output and the source hair mask ( $m_{hair}$ ) as illustrated in Fig. 3. Specifically, $m_{ca}$ is computed by normalizing and thresholding the CA maps extracted from the Align-CA layers. We obtain $m_{blend}$ with the union of $1 - m_{ca}$ and $1 - m_{hair}$ , indicating the regions to preserve in x as 1. Finally, we perform the adaptive hair blending with $m_{blend}$ , and update $z_{t}$ at timestep t with the blended feature before proceeding to the next timestep (t - 1) as follows:

$$
\mathbf {z} _ {t} \leftarrow \mathbf {z} _ {t} ^ {\mathbf {x}} \odot \mathbf {m} _ {\text { b   l   e   n   d }} + \mathbf {z} _ {t} \odot (1 - \mathbf {m} _ {\text { b   l   e   n   d }}), \tag {6}
$$

where $z_{t}^{x}$ indicates the noisy latent of x at timestep t, and $m_{blend}$ is a binary mask, where the regions to be preserved are 1. We apply adaptive hair blending during the final N timesteps of the denoising process, and N is set as 10 in the entire experiment. In this way, we can faithfully preserve the features of the source face image x, minimizing the reconstruction error of the autoencoder.

# Experiment

# Experimental Setup

Dataset. We utilize two multi-view datasets, the K-hairstyle (Kim et al. 2021) and the CelebV-Text (Yu et al. 2023) datasets for the experiments. The K-hairstyle dataset consists of 500,000 high-resolution face images with various focal lengths and head poses with more than 6,400 identities. Following the Style Your Hair (Kim et al. 2022), we excluded the images whose hairstyle or face is significantly occluded. Due to the privacy issue, we blur the face of the

![](images/76e5891a87dc7efa074fff2bf69d6f78426494ae96a1bbf4938f0058b6fd3f1b.jpg)

Figure 4: Qualitative comparison with the diffusion-based baselines.   
![](images/b115a9a6dc76e31188e9c8a5aafc5ed0ce79868ab36d4198532af86f1d39b305.jpg)

<details>
<summary>text_image</summary>

Face
Hair
SYH
PbE
Anydoor
Ours
</details>

Figure 5: Qualitative comparison with baselines using web-crawled images.

images from the K-hairstyle dataset. Also, the CelebV-Text dataset is a large-scale facial text-video dataset that contains 70,000 in-the-wild face video clips with text pairs. For the experiments, we randomly sampled 6,000 video clips with different identities and sampled 20 frames from each video. We remove images whose facial landmarks or hair masks are not detected. All the images are resized to $512 \times 512$ in the experiments. More details for dataset preprocessing are described in the supplementary material.

Baselines. We first compare our HairFusion with state-of-the-art diffusion models for exemplar-based image inpainting: Paint-by-Example (PbE) (Yang et al. 2023a) and Anydoor (Chen et al. 2024). For a fair comparison, we re-

place their input with our newly designed hair-agnostic representation during the training and inference. Since Anydoor uses various multi-view datasets to train the model, we fine-tune the Anydoor with our dataset instead of training the model from scratch. Also, we compare our method with StyleGAN-based hairstyle transfer approaches, Style Your Hair (SYH) (Kim et al. 2022) and HairCLIPv2 (Wei et al. 2023). We apply StyleGAN (Karras et al. 2020) trained on our dataset to StyleGAN-based approaches. Since the methods based on StyleGAN pre-trained with the FFHQ dataset (Karras, Laine, and Aila 2019) mainly tackle cropped aligned faces, we also compare our method with StyleGAN-based methods using cropped aligned images.

![](images/cab2202e7beddd30ebf664b6a69c0888784fe57230fdc8ff7f340fabe93784ca.jpg)

Figure 6: Qualitative comparison with StyleGAN-based methods using cropped and aligned images. 

<table><tr><td>Dataset</td><td>Method</td><td>FID↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td rowspan="4">K-Hairstyle</td><td>SYH</td><td>22.84</td><td>0.62</td><td>18.41</td><td>0.32</td></tr><tr><td>PbE</td><td>13.26</td><td>0.59</td><td>19.26</td><td>0.29</td></tr><tr><td>Anydoor</td><td>19.00</td><td>0.63</td><td>18.78</td><td>0.27</td></tr><tr><td>Ours</td><td>10.82</td><td>0.70</td><td>21.15</td><td>0.18</td></tr><tr><td rowspan="4">CelebV-Text</td><td>SYH</td><td>35.77</td><td>0.62</td><td>21.14</td><td>0.30</td></tr><tr><td>PbE</td><td>32.72</td><td>0.41</td><td>14.55</td><td>0.42</td></tr><tr><td>Anydoor</td><td>35.70</td><td>0.50</td><td>15.45</td><td>0.37</td></tr><tr><td>Ours</td><td>23.50</td><td>0.74</td><td>22.55</td><td>0.18</td></tr></table>

Table 1: Quantitative comparison to baselines, including SYH, PbE, and Anydoor.

Specifically, we crop and align the test images in the way the FFHQ dataset is preprocessed for StyleGAN-based approaches. Also, we crop and align the results of HairFusion in the same way for comparison.

Evaluation Metric. We evaluate our method on two tasks, hairstyle transfer and reconstruction. Hairstyle transfer modifies hairstyle between two images with different identities and hairstyles. We measure the fréchet inception distance (FID) score (Heusel et al. 2017) to evaluate how similar the distributions of the synthesized and the real images are. In the reconstruction task, we utilize two images with the same identity and hairstyle but different head poses. One pair is considered the source face (i.e., the ground truth image for the model to reconstruct) while the other serves as the reference hairstyle image. We measure SSIM, PSNR, and LPIPS for the reconstruction task to evaluate if the model accurately reflects the detailed features (e.g., shape, length, color, etc.) of the reference hair in the result. For the evaluation, we randomly sample 2,000 pairs from the test sets.

# Comparison to Baselines

Qualitative Comparison. Fig. 4 presents a comparison between our model and diffusion-based image inpainting models, PbE (Yang et al. 2023a) and Anydoor (Chen et al. 2024), using the K-Hairstyle and CelebV-Text datasets. PbE struggles to reflect the shape or color of the reference hair in the output. This is because PbE injects the reference hair features using only the class token extracted by the CLIP image encoder, which results in a significant loss of detailed

<table><tr><td>Datset</td><td>Method</td><td>FID↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">K-Hairstyle</td><td>SYH</td><td>25.79</td><td>0.50</td><td>15.56</td><td>0.36</td></tr><tr><td>HairCLIPv2</td><td>24.36</td><td>0.48</td><td>14.45</td><td>0.38</td></tr><tr><td>Ours</td><td>15.41</td><td>0.55</td><td>19.26</td><td>0.30</td></tr><tr><td rowspan="3">CelebV-Text</td><td>SYH</td><td>35.69</td><td>0.65</td><td>22.09</td><td>0.28</td></tr><tr><td>HairCLIPv2</td><td>35.40</td><td>0.66</td><td>21.21</td><td>0.29</td></tr><tr><td>Ours</td><td>22.00</td><td>0.70</td><td>21.77</td><td>0.21</td></tr></table>

Table 2: Quantitative comparison to StyleGAN-based baselines using cropped and aligned dataset.

hair shape and color information. While Anydoor preserves the features of the reference hair, it fails to maintain the head shape or clothing of the source face image.

Additionally, we conduct a comparison with web-crawled images in Fig. 5 using ours and the baselines trained on the K-Hairstyle dataset. Overall, SYH produces blurry outputs compared to the diffusion-based models. Although SYH employs StyleGAN trained on multi-view images, it fails to preserve the source face's identity and loses the detail of the reference hair when applied to in-the-wild images containing diverse poses. Moreover, PbE and Anydoor struggle to maintain both the hairstyle features of the reference hair and the non-hair features in the face image.

Lastly, we compare our method to StyleGAN-based methods using cropped and aligned test sets in Fig. 6. The outputs generated by SYH and HairCLIPv2 have blurrier hair compared to ours. Also, SYH and HairCLIPv2 fail to align the reference hair to a face when they have a large pose difference, even when the inputs are cropped and aligned.

Unlike the existing methods, HairFusion generates realistic outputs not only in both the K-Hairstyle and CelebV-Text datasets but also in the web-crawled in-the-wild images while preserving both the fine details of the reference hair and the surrounding features in the source face image.

Quantitative Comparison. Table 1 shows that our model achieves superior performance compared to the existing methods, including SYH (Kim et al. 2022), PbE (Yang et al. 2023a), and Anydoor (Chen et al. 2024). For a fair comparison, we apply our newly designed hair-agnostic representation to the diffusion-based baselines. PbE achieves inferior performance in the reconstruction task (i.e., SSIM, PSNR,

![](images/62684e706516d15fee5c75b4ad602218f6d0411ad0f3cb8d3026395475b53867.jpg)

<details>
<summary>text_image</summary>

Face
Hair
CA Mask in Align-CA
Output
t = 40
t = 30
t = 20
t = 10
t = 0
</details>

Figure 7: Visualization of estimated hair mask in Align-CA. The CA masks are obtained by normalizing and thresholding the CA maps in Align-CA. The CA masks successfully indicate the hair regions of the output. t indicates the timestep of the reverse denoising process, where the total timestep is 50.

![](images/6a8b17a830f28753c3dc6383297ec02fb574f3db73dc11989ecdfeb5494aa126.jpg)

<details>
<summary>text_image</summary>

Face
Hair
Baseline
+ Align-CA
+ Blending (Ours)
</details>

Figure 8: Qualitative evaluation for the ablation study using K-Hairstyle dataset. The red box indicates the reference hair.

and LPIPS), showing that it fails to reflect the detailed features of the reference hair. Although SYH and Anydoor achieve better reconstruction performance than PbE, they produce unrealistic outputs, achieving higher FID scores.

Also, we compare our method to the existing StyleGAN-based methods in Table 2 using cropped and aligned images. Our method achieves superior performance in both hairstyle transfer and the reconstruction task, except for CelebV-Text PSNR. Since most of the non-hair regions in face images are eliminated in cropped and aligned images, our adaptive hair blending may show only a slight improvement or comparable performance in the reconstruction task.

# Ablation Study

We evaluate our method by gradually adding each component of HairFusion to the baseline. As in Fig. 8 and Table 3, we start from the baseline which only contains the denoising U-Net, CLIP image encoder, and the hair encoder. Then, we gradually add Align-CA and the adaptive hair blending. The results show that the baseline struggles to estimate the exact hair shape and length of the aligned reference hair, achieving lower reconstruction performance. Although adding Align-CA largely improves hair alignment performance, it still fails to preserve detailed features in the non-hair region of the face image such as clothing. Our method with adaptive hair blending achieves the best performance in both hairstyle transfer and reconstruction. Fig. 8 shows that only ours successfully maintains the reference hair features as well as the clothing' details of the face image.

<table><tr><td></td><td>FID↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Baseline</td><td>15.43</td><td>0.58</td><td>18.09</td><td>0.30</td></tr><tr><td>+ Align-CA</td><td>15.59</td><td>0.62</td><td>19.33</td><td>0.27</td></tr><tr><td>+ Adaptive Hair Blending</td><td>10.82</td><td>0.70</td><td>21.15</td><td>0.18</td></tr></table>

Table 3: Quantitative ablation study using K-Hairstyle.

# Analysis of Adaptive Hair Blending

Adaptive hair blending leverages the CA mask $m_{ca}$ in the Align-CA to estimate hair regions of the output. Fig. 7 visualizes $m_{ca}$ extracted from Align-CA every 10 timesteps as the 50 timesteps of the reverse denoising process proceed. The first and the second row present a test pair from K-Hairstyle and CelebV-Text, respectively. We visualize $m_{ca}$ from the 7-th CA map in Align-CA while we use the 6-th and 7-th CA map in adaptive hair blending. The figure shows that $m_{ca}$ successfully indicates the hair regions of the generated image. HairFusion effectively reconstructs the original features of x by blending the generated hair features and the non-hair region features in x.

# Conclusion

This paper proposes the first one-stage diffusion-based hairstyle transfer model, HairFusion, conceptualizing hairstyle transfer as an exemplar-based image inpainting. HairFusion introduces Align-CA that aligns the target hairstyle with a face image based on dense pose features, accounting for their pose difference. Our novel adaptive hair blending technique allows HairFusion to blend the transferred reference hair features with the source face's other appearance and background features based on the hair region estimated by CA maps of Align-CA. HairFusion achieves state-of-the-art performance compared to the existing approaches, including StyleGAN-based methods and diffusion models for exemplar-based inpainting. We also demonstrate that HairFusion can generalize to in-the-wild samples with diverse head poses and focal lengths.

# Supplementary Material

This supplementary material includes a comparison to recent approaches, additional qualitative results using in-the-wild images, details of hair-agnostic representation and dataset, implementation details, limitations, and future work.

Comparison to Recent Approaches 

<table><tr><td></td><td>FID↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>HairFastGAN</td><td>19.37</td><td>0.52</td><td>17.19</td><td>0.36</td></tr><tr><td>Stable-Hair</td><td>30.02</td><td>0.50</td><td>16.91</td><td>0.32</td></tr><tr><td>Ours</td><td>15.41</td><td>0.55</td><td>19.26</td><td>0.30</td></tr></table>

Table 4: Quantitative comparison to recent approaches using cropped and aligned K-Hairstyle dataset.

We conduct a comparison to the most recent StyleGAN-based method, HairFastGAN (Nikolaev et al. 2024), and the concurrent diffusion-based method, Stable-Hair (Zhang et al. 2024). We conduct experiments with K-Hairstyle dataset (Kim et al. 2021) using pre-trained models from the official codes and cropped and aligned images as in Table 2. According to Table 4, ours achieves superior performance over the others. In our experiments, Stable-Hair may achieve inferior FID due to artifacts generated in bald proxy images.

# Additional Qualitative Results

Fig. 9 shows additional qualitative results of HairFusion using web-crawled in-the-wild images. The result shows that our method achieves robust performance in real-world scenarios where the images have diverse hair lengths and poses.

Moreover, Fig. 10 presents additional results of the qualitative comparison to Style Your Hair (SYH) (Kim et al. 2022), Paint-by-Example (PbE) (Yang et al. 2023a), and Anydoor (Chen et al. 2024). The models are trained with the K-Hairstyle dataset (Kim et al. 2021) that contains multiview images of various head poses and focal lengths.

According to the figure, SYH generates unrealistic images due to poor hair alignment performance when applied to in-the-wild images of various head poses and focal lengths. Also, PbE fails to reflect the reference hairstyle including color, shape, and length in the output. This is because PbE depends solely on the class token of the pre-trained CLIP image encoder to inject hairstyle features, which limits its ability to capture detailed spatial information of the reference hair. While Anydoor is better at preserving the reference hairstyle, it generates unrealistic head shapes by simply copying the hairstyle from the reference image (see the fourth and the last rows of Fig. 10). Moreover, Anydoor struggles to maintain the non-hair region of the source face image such as clothing and background.

In contrast, HairFusion effectively generalizes to web-crawled images by preserving the detailed features of the reference hair through Align-CA, while also maintaining the surrounding features of the source face via adaptive hair blending.

# Details of Hair-Agnostic Representation

Our hair-agnostic representation $\mathbf{x}_{agn}$ is designed to faithfully eliminate potential hair regions for $\mathbf{x}_{hair}$ and the original hair information in $\mathbf{x}$ while preserving the other regions that should be maintained. We obtain $\mathbf{x}_{agn}$ by masking out $\mathbf{x}$ using the agnostic mask $\mathbf{m}_{agn}$ , a binary mask where the regions to be preserved are 1.

To obtain $m_{agn}$ , we first remove the hair region of x based on its hair mask $m_{hair}$ . In case x has no hair, we use DensePose (Guler, Neverova, and Kokkinos 2018) to roughly estimate the head shape and remove the head above the eyebrows as well. To fully remove the original hair, we eliminate a larger area to account for where hair might potentially be, using the DensePose and facial landmarks. Specifically, we remove the region from the top of the DensePose to the bottom of the image, and from the leftmost to rightmost jaw points in the facial landmarks with additional margin. The width of the eliminated region is roughly twice the face width. We preserve the face region and the neck and body beneath the chin, where hair typically does not exist.

In this way, $\mathbf{x}_{agn}$ minimizes dependency on the original hairstyle during the training, completely obscuring the original hair shape and length. Furthermore, the removed potential hair region allows various reference hairstyles to be transferred at the inference.

# Dataset Details

We utilize two multi-view datasets, K-Hairstyle (Kim et al. 2021) and CelebV-Text (Yu et al. 2023) for the experiments. In the K-Hairstyle dataset, we use images where the head pose angle (yaw) ranges from -30 to 30 degrees, where 0 degrees indicate a forward-facing position. We use the ground truth hair mask for the K-Hairstyle and the estimated hair segmentation mask for the CelebV-Text. For the face parsing masks including the hair mask, we utilize a pre-trained face parsing model (Yu et al. 2018). Since the CelebV-Text is a video dataset, we exclude images with extreme motion blur as well. We generate the face outline image $f_{agn}$ by drawing lines of different colors based on the facial landmarks extracted by a pre-trained detector (Bulat and Tzimiropoulos 2017). We obtain dense pose images, $p_{agn}$ and $p_{hair}$ , using a pre-trained dense human pose estimation model (Guler, Neverova, and Kokkinos 2018).

# Implementation Details

# Architecture

We employ the architecture of the autoencoder and the denoising U-Net of Stable Diffusion v1.4 (Rombach et al. 2022). We replace the last nine cross-attention layers in the decoder of the denoising U-Net with our Align-CA. The architecture of the hair encoder follows the encoder of the U-Net. Also, the pose encoder follows the architecture of the condition embedding network in ControlNet (Zhang, Rao, and Agrawala 2023), which is designed to generate a conditioning vector from a given condition.

![](images/7a14d98f22831e850ad3d13deffc237e37dcd7f47e521d7dfec4f7456bcde639.jpg)

<details>
<summary>text_image</summary>

Hair
Face
</details>

Figure 9: Additional qualitative results of HairFusion using in-the-wild images.

# Training and Inference

The hair encoder, Align-CA, and pose encoder $(\mathcal{E}_{p})$ are trainable. We train our model with the following objective:

$$
\mathcal {L} _ {L D M} = \mathbb {E} _ {\zeta , \epsilon \sim \mathcal {N} (0, 1), t} \left[ \| \epsilon - \epsilon_ {\theta} (\zeta , t, \tau_ {\phi} (\mathbf {x} _ {h a i r}), \mathcal {E} (\mathbf {x} _ {h a i r}) \| _ {2} ^ {2} \right], \tag {7}
$$

$$
\zeta = \left(\left[ \mathbf {z} _ {t}; \mathcal {E} \left(\mathbf {x} _ {\text {agn}}\right); \mathcal {R} \left(\mathbf {m} _ {\text {agn}}\right); \mathcal {E} \left(\mathbf {f} _ {\text {agn}}\right) \right], \mathcal {E} _ {p} \left(\mathbf {p} _ {\text {agn}}\right), \mathcal {E} _ {p} \left(\mathbf {p} _ {\text {hair}}\right)\right), \tag {8}
$$

where R indicates resize function. To further encourage the model to generate a realistic face image, we spatially upweight $L_{LDM}$ with $(1 + \mathbf{m}_{\lambda})$ , where $m_{\lambda} \in R^{1 \times h \times w}$ is $\mathcal{R}(\lambda_{hair}\mathbf{m}_{hair} + \lambda_{face}\mathbf{m}_{face} + \lambda_{fg}\mathbf{m}_{fg})$ . Here, $m_{hair}$ , $m_{face}$ , and $m_{fg}$ denote a hair, face, and foreground mask, respectively. In the experiment, we set $\lambda_{hair}$ , $\lambda_{face}$ , and $\lambda_{fg}$ as 5. We employ the weights of the autoencoder of Realistic Vision V5.1 (Civitai 2024) to enhance the quality of face reconstruction. The weights of our denoising U-Net are initialized with the pre-trained U-Net of the PbE (Yang et al. 2023a). To stabilize the training, we zero-initialize a linear layer after the feed-forward operation in Align-CA and the projection layers followed by the pose encoder. We train HairFusion using an AdamW optimizer with a fixed learning rate of 1e-4 and a batch size of 48 for 600 and 300 epochs for K-Hairstyle and CelebV-Text, respectively. We use NVIDIA A100 GPUs for training and NVIDIA RTX A6000 for inference. For inference, we employ a DDIM sampler (Song, Meng, and Ermon 2021) with 50 sampling steps.

# Limitations and Future Work

Since HairFusion employs an external face parsing model and facial landmark detector, the output quality highly depends on the performance of the external models. For instance, in the sixth row of Fig. 10, a single strand of bangs from the reference hair is not reflected in the output because the estimated hair mask failed to capture this detail. Additionally, HairFusion still struggles to generate or reconstruct detailed features (e.g., text) located near the hair region in the output, as in the third row of Fig. 10. Lastly, while HairFusion does not currently support additional control over the length or shape of the reference hairstyle, providing users with a preview of the hairstyle in various lengths could significantly improve user satisfaction. Exploring how to control the reference hairstyle based on user input could be a promising direction for future research.

# Acknowledgements

This work was supported by the Institute of Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.RS-2019-II190075 Artificial Intelligence Graduate School Program(KAIST) and RS-2021-II212068, Artificial Intelligence Innovation Hub) and the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT)(No. 2022R1A5A7083908).

![](images/ae3f64c162aa8b8dca7f078a1e4443f91d0e66aa6112eee219a606d96340bc71.jpg)  
Figure 10: Additional qualitative comparison using in-the-wild images.

# References

Avrahami, O.; Lischinski, D.; and Fried, O. 2022. Blended Diffusion for Text-Driven Editing of Natural Images. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 18208–18218.   
Bulat, A.; and Tzimiropoulos, G. 2017. How far are we from solving the 2D & 3D face alignment problem? (and a dataset of 230,000 3D facial landmarks). In Proc. of the IEEE international conference on computer vision (ICCV).   
Chang, S.; Kim, G.; and Kim, H. 2023. HairNeRF: Geometry-Aware Image Synthesis for Hairstyle Transfer. In Proc. of the IEEE international conference on computer vision (ICCV), 2448–2458.   
Chen, X.; Huang, L.; Liu, Y.; Shen, Y.; Zhao, D.; and Zhao, H. 2024. Anydoor: Zero-shot object-level image customization. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 6593–6602.   
Choi, S.; Park, S.; Lee, M.; and Choo, J. 2021. VITON-HD: High-resolution virtual try-on via misalignment-aware normalization. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR).   
Choi, Y.; Kwak, S.; Lee, K.; Choi, H.; and Shin, J. 2024. Improving diffusion models for virtual try-on. In Proc. of the European Conference on Computer Vision (ECCV).   
Chung, C.; Kim, T.; Nam, H.; Choi, S.; Gu, G.; Park, S.; and Choo, J. 2021. HairFIT: Pose-Invariant Hairstyle Transfer via Flow-based Hair Alignment and Semantic-Region-Aware Inpainting. In Proc. of the British Machine Vision Conference (BMVC). British Machine Vision Association.   
Civitai. 2024. Realistic Vision V5.1.   
Guler, R. A.; Neverova, N.; and Kokkinos, I. 2018. Dense-Pose: Dense Human Pose Estimation In The Wild. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR).   
Hertz, A.; Mokady, R.; Tenenbaum, J.; Aberman, K.; Pritch, Y.; and Cohen-Or, D. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626.   
Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Proc. the Advances in Neural Information Processing Systems (NeurIPS).   
Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Proc. the Advances in Neural Information Processing Systems (NeurIPS), 33: 6840–6851.   
Hu, L. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 8153–8163.   
Karras, T.; Laine, S.; and Aila, T. 2019. A Style-Based Generator Architecture for Generative Adversarial Networks. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR).

Karras, T.; Laine, S.; Aittala, M.; Hellsten, J.; Lehtinen, J.; and Aila, T. 2020. Analyzing and Improving the Image Quality of StyleGAN. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR).

Khwanmuang, S.; Phongthawee, P.; Sangkloy, P.; and Suwajanakorn, S. 2023. StyleGAN Salon: Multi-View Latent Optimization for Pose-Invariant Hairstyle Transfer. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 8609–8618.

Kim, J.; Gu, G.; Park, M.; Park, S.; and Choo, J. 2024a. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 8176–8185.

Kim, J.; Kim, M.-J.; Lee, J.; and Choo, J. 2024b. TCAN: Animating Human Images with Temporally Consistent Pose Guidance using Diffusion Models. Proc. of the European Conference on Computer Vision (ECCV).

Kim, K.; Park, S.; Lee, J.; and Choo, J. 2023. Reference-based image composition with sketch via structure-aware diffusion model. In Proc. of the IEEE conference on computer vision and pattern recognition workshop (CVPRW).

Kim, T.; Chung, C.; Kim, Y.; Park, S.; Kim, K.; and Choo, J. 2022. Style your hair: Latent optimization for pose-invariant hairstyle transfer via local-style-aware hair alignment. In Proc. of the European Conference on Computer Vision (ECCV), 188–203. Springer.

Kim, T.; Chung, C.; Park, S.; Gu, G.; Nam, K.; Choe, W.; Lee, J.; and Choo, J. 2021. K-Hairstyle: A Large-Scale Korean Hairstyle Dataset For Virtual Hair Editing And Hairstyle Classification. In Proc. of the IEEE International Conference on Image Processing (ICIP), 1299–1303. IEEE.

Lee, S.; Gu, G.; Park, S.; Choi, S.; and Choo, J. 2022. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In Proc. of the European Conference on Computer Vision (ECCV), 204–219. Springer.

Nikolaev, M.; Kuznetsov, M.; Vetrov, D.; and Alanov, A. 2024. HairFastGAN: Realistic and Robust Hair Transfer with a Fast Encoder-Based Approach. Proc. the Advances in Neural Information Processing Systems (NeurIPS).

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In Proc. the International Conference on Machine Learning (ICML), 8748–8763. PMLR.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 10684–10695.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer Assisted Intervention, 234–241. Springer.

Saha, R.; Duke, B.; Shkurti, F.; Taylor, G.; and Aarabi, P. 2021. LOHO: Latent optimization of hairstyles via orthogonalization. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR).

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, K.; Gontijo Lopes, R.; Karagol Ayan, B.; Salimans, T.; et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Proc. the Advances in Neural Information Processing Systems (NeurIPS), 35: 36479–36494.   
Song, J.; Meng, C.; and Ermon, S. 2021. Denoising diffusion implicit models. Proc. the International Conference on Learning Representations (ICLR).   
Tan, Z.; Chai, M.; Chen, D.; Liao, J.; Chu, Q.; Yuan, L.; Tulyakov, S.; and Yu, N. 2020. MichiGAN: Multi-input-conditioned hair image generation for portrait editing. Proc. the ACM Transactions on Graphics (ToG), 39(4): 1–13.   
Wei, T.; Chen, D.; Zhou, W.; Liao, J.; Tan, Z.; Yuan, L.; Zhang, W.; and Yu, N. 2022. Hairclip: Design your hair by text and reference image. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 18072–18081.   
Wei, T.; Chen, D.; Zhou, W.; Liao, J.; Zhang, W.; Hua, G.; and Yu, N. 2023. HairCLIPv2: Unifying Hair Editing via Proxy Feature Blending. In Proc. of the IEEE international conference on computer vision (ICCV), 23589–23599.   
Yang, B.; Gu, S.; Zhang, B.; Zhang, T.; Chen, X.; Sun, X.; Chen, D.; and Wen, F. 2023a. Paint by example: Exemplar-based image editing with diffusion models. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 18381–18391.   
Yang, S.; Jiang, L.; Liu, Z.; ; and Loy, C. C. 2023b. StyleGANEX: StyleGAN-Based Manipulation Beyond Cropped Aligned Faces. In Proc. of the IEEE international conference on computer vision (ICCV).   
Yang, S.; Jiang, L.; Liu, Z.; and Loy, C. C. 2022. VToonify: Controllable High-Resolution Portrait Video Style Transfer. Proc. of the ACM SIGGRAPH Asia.   
Yu, C.; Wang, J.; Peng, C.; Gao, C.; Yu, G.; and Sang, N. 2018. Bisenet: Bilateral segmentation network for real-time semantic segmentation. In Proc. of the European Conference on Computer Vision (ECCV).   
Yu, J.; Zhu, H.; Jiang, L.; Loy, C. C.; Cai, W.; and Wu, W. 2023. Celebv-text: A large-scale facial text-video dataset. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 14805–14814.   
Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. In Proc. of the IEEE international conference on computer vision (ICCV), 3836–3847.   
Zhang, Y.; Zhang, Q.; Song, Y.; and Liu, J. 2024. Stable-Hair: Real-World Hair Transfer via Diffusion Model. arXiv:2407.14078.   
Zhu, P.; Abdal, R.; Femiani, J.; and Wonka, P. 2021. Barbershop: GAN-based Image Compositing using Segmentation Masks. Proc. of the ACM SIGGRAPH Asia, 40(6).