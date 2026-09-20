# Dynamic Contrastive Knowledge Distillation for Efficient Image Restoration

# Yunshuai Zhou $^{1*}$ , Junbo Qiao $^{1*}$ , Jincheng Liao $^{1}$ , Wei Li $^{2}$ , Simiao Li $^{2}$ , Jiao Xie $^{1}$ , Yunhang Shen $^{3}$ , Jie Hu $^{2}$ , Shaohui Lin $^{1,4\dagger}$

$^{1}$ East China Normal University, Shanghai, China

$^{2}$ Huawei Noah's Ark Lab, China

$^{3}$ Xiamen University, China

$^{4}$ Key Laboratory of Advanced Theory and Application in Statistics and Data Science - MOE, China

# Abstract

Knowledge distillation (KD) is a valuable yet challenging approach that enhances a compact student network by learning from a high-performance but cumbersome teacher model. However, previous KD methods for image restoration overlook the state of the student during the distillation, adopting a fixed solution space that limits the capability of KD. Additionally, relying solely on L1-type loss struggles to leverage the distribution information of images. In this work, we propose a novel dynamic contrastive knowledge distillation (DCKD) framework for image restoration. Specifically, we introduce dynamic contrastive regularization to perceive the student's learning state and dynamically adjust the distilled solution space using contrastive learning. Additionally, we also propose a distribution mapping module to extract and align the pixel-level category distribution of the teacher and student models. Note that the proposed DCKD is a structure-agnostic distillation framework, which can adapt to different backbones and can be combined with methods that optimize upper-bound constraints to further enhance model performance. Extensive experiments demonstrate that DCKD significantly outperforms the state-of-the-art KD methods across various image restoration tasks and backbones. Our codes are available at https://github.com/super-SSS/DCKD.

# Introduction

Image restoration aims to recover high-quality images from low-quality ones degraded by processes such as subsampling, blurring, and rain streaks. It is a highly challenging ill-posed inverse problem since crucial content information is missing during degradation. Recently, convolutional neural networks (CNNs) (Dong et al. 2015; Lim et al. 2017; Zhang et al. 2018) and Transformers (Liang et al. 2021; Chen et al. 2021; Wang et al. 2022; Zamir et al. 2022) have been extensively investigated for designing various models, achieving remarkable success in image restoration. However, these models demand high resources and exhibit inefficiencies, making deployment on resource-constrained devices challenging. To facilitate their real-world applications, there is a growing research focus on compressing image restoration models.

![](images/7a8f618513e735806a2b61fbc0cc71812a767615256cd9496d27d56232ebda50.jpg)  
(a) Vanilla Knowledge Distillation

![](images/c4eaa180916a864fdec0979948f14c07907b60d8a388abfa8a1ad85303a04ff6.jpg)  
(b) Contrastive Knowledge Distillation

![](images/27a40602bdfe2d4c1190b69346e13429816625f7a59c8831f942fbbc7836696c.jpg)  
(c) Dynamic Contrastive Knowledge Distillation

![](images/d8955b5092e6d11592bd319299a2f888dd4397a8678b4ee3feea723582775c73.jpg)

![](images/a5b976c024043bbf30e324838e2f7c5f35d7742ddccf42854379ffcebc8fb95a.jpg)  
★ Positive   
History Anchor   
Anchor   
▲ Negative   
→←— Pull   
History Negative   
← → Push

Figure 1: Difference between our DCKD and existing KD methods. (a) The Vanilla KD method overlooks the information from negative images as a lower bound. (b) Existing contrastive KD methods adopt a fixed lower bound that limits the capability of KD. (c) Our DCKD introduces dynamic contrastive regularization to perceive the student's learning state and dynamically adjust the distilled solution space.

Knowledge distillation (KD) is an effective model compression method that transfers knowledge from a cumbersome teacher model to a lightweight student model. This process allows the student model to inherit the capabilities of the teacher model, resulting in significant performance improvements while reducing computational and storage requirements. KD has gained broad recognition for its excellent performance and broad applicability. It also can be combined with other model compression techniques, such as quantization (Du et al. 2021; Ayazoglu 2021; Hong et al. 2022), pruning (Fan et al. 2020; Wang et al. 2021a; Oh et al. 2022), compact architecture design (Ahn, Kang, and Sohn 2018; Zhang et al. 2022; Chen et al. 2022a), and neural architecture search (NAS) (Gou et al. 2020; Kim et al. 2021; He et al. 2022), to enhance the compactness of student models further.

Since KD has been well-established in natural language processing (Hahn and Choi 2019; Sanh et al. 2019; Fu et al. 2021) and high-level vision tasks (Touvron et al. 2021; Lin

et al. 2022; Chen et al. 2022b), researchers have been investigating KD for image restoration methods (He et al. 2020; Lee et al. 2020; Wang et al. 2021d; Zhang et al. 2023; Li et al. 2024; Jiang et al. 2024; Zhang et al. 2024). However, these methods adopt a fixed solution space, which limits their adaptability to the evolving state of the student model during the distillation process. As illustrated in Fig. 1 (a), the vanilla distillation method (Hinton, Vinyals, and Dean 2015) only constrains the upper bound of the solution space. Although existing works (Zhang et al. 2023; Jiang et al. 2024) explore more effective and diverse upper bounds, the lack of constraints on the lower bound of the output image increases the difficulty of optimizing the solution space. This often generates low-quality images with artifacts, color distortion, and blurring. As illustrated in Fig. 1 (b), CSD (Wang et al. 2021d) introduces contrastive learning to design lower-bound constraints, significantly enhancing the transfer of knowledge from the teacher. However, in the later stages of training, the student anchor moves far from the lower bound, leading to a diminished constraint effect.

To address this problem, we propose a novel dynamic contrastive knowledge distillation framework named DCKD. Specifically, we first propose the dynamic contrastive regularization, which generates dynamic lower bound constraints. In addition, we also propose a distribution mapping module (DMM) to extract and align the pixel-level category distribution between the output of the teacher and student. Compared with previous image restoration distillation methods that primarily relied on L1 loss, DMM successfully introduces category distribution information distillation into low-level vision tasks. DCKD not only adapts to various backbones but also can be combined with methods (Zhang et al. 2023; Li et al. 2024; Jiang et al. 2024) that improve the upper bound of the solution space to further enhance distillation performance. We validate the effectiveness of DCKD across multiple image restoration tasks, including image super-resolution, deblurring, and deraining.

Overall, our main contributions are summarized as:

- We propose a dynamic contrastive distillation framework (DCKD), which can perceive the student's learning state and dynamically optimize the lower bound of the solution space.   
- We introduce a distribution mapping module to leverage category distribution information distillation, which has been significantly ignored in previous KD works for image restoration.   
- Extensive experiments across various image restoration tasks demonstrate that the proposed DCKD framework significantly outperforms previous methods.

# Related Work

# Image Restoration

Since the pioneering works SRCNN (Dong et al. 2015) and DnCNN (Zhang et al. 2017) are firstly to employ CNNs for image restoration, various works (Lim et al. 2017; Nah, Hyun Kim, and Mu Lee 2017a; Lefkimmiatis 2017; Li et al. 2018; Ren et al. 2019; Chen et al. 2022a) have been proposed to improve the performance by increasing the parameters. Recently, Transformer-based methods (Chen et al. 2021; Liang et al. 2021; Chen et al. 2022c, 2023) have leveraged self-attention mechanisms to capture long-range dependencies, leading to significant performance improvements in image restoration. To reduce computational overhead, SAFMN (Sun et al. 2023) enhances model efficiency by utilizing spatially adaptive feature pyramid attention maps. Restormer (Zamir et al. 2022) designs channel self-attention, which is more efficient than spatial self-attention. Although lightweight designs of CNN and Transformer architectures significantly reduce computational overhead, they still face challenges regarding direct deployment on resource-constrained platforms.

# Knowledge Distillation for Image Restoration

Knowledge distillation aims to significantly reduce deployment costs while improving the performance of student models by emulating the behavior of teacher models (Hinton, Vinyals, and Dean 2015; Lee et al. 2020; Gou et al. 2021). In recent years, numerous works have focused on knowledge distillation for image restoration. He et al. proposed FAKD to align the spatial affinity matrix of the feature maps between the teacher and student models (He et al. 2020). To alleviate the semantic differences between features of the teacher and student, Li et al. proposed MiPKD, which achieves feature and stochastic network block mixture in latent space (Li et al. 2024). Jiang et al. proposed MTKD that designs a composite output from multiple teachers to provide the student with a more robust teacher model (Jiang et al. 2024). However, these KD methods primarily focus on improving the upper bound of the model's solution space, without leveraging the distribution information of the images.

# Contrastive Learning for Knowledge Distillation

Recently, many researchers have explored the combination of contrastive learning and knowledge distillation to construct a comprehensive solution space (Tian, Krishnan, and Isola 2019; Xu et al. 2020; Yang et al. 2023). Wang et al. introduced contrastive learning in knowledge distillation, utilizing other images within the same dataset to construct a lower bound for the solution space (Wang et al. 2021b). Similarly, CSD (Wang et al. 2021d) proposed a contrastive self-distillation method, utilizing different images within the batch to provide lower bound constraints. Luo et al. enriched the lower bound constraints in the image deraining task by altering the direction of falling raindrops for the student model (Luo et al. 2023). However, these methods employ a fixed solution space, which leads to a weakening of the lower-bound constraint when the student anchor moves away from the lower bound in the later stages of training. Different from these methods, DCKD proposes a dynamic lower-bound constraint that progressively narrows the solution space, and can also be combined with enhanced upper-bound approaches. Additionally, DMM is proposed to extract and align the pixel-level category distribution information between the teacher and student networks.

![](images/3230c21547756b0342253f70e35377e72dd93b618f567d1eda5c447502892483.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher Model"] --> B["DMM"]
    C["Student Model"] --> B
    D["Degradation Module"] --> E["History Model"]
    E --> F["Encoder"]
    F --> G["Contrastive Loss"]
    
    subgraph (a) Dynamic Contrastive Regularization
        H["I_LQ"] --> I["Visual Image"]
        J["EMA"] --> K["Visual Image"]
        L["Dynamic Negative Sample Generator"] --> M["Visual Image"]
        N["Visual Image"] --> O["I^T_HQ"]
        P["Visual Image"] --> Q["I^S_HQ"]
        R["Visual Image"] --> S["I^N_HQ"]
        T["Visual Image"] --> U["I^1_Neg"]
        V["Visual Image"] --> W["I^N_Neg"]
    end
    
    subgraph (b) Distribution Mapping Module
        X["I^T_HQ"] --> Y["Encoder"]
        Z["C^T"] --> AA["CodeBook"]
        AB["C^S"] --> AC["CodeBook"]
        AD["Positive"] --> AE["Anchor"]
        AF["Negative"] --> AG["Negative"]
    end
    
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> AA
    AA --> AB
    AB --> AC
    AC --> AD
```
</details>

Figure 2: Illustration of the proposed Dynamic Contrastive Knowledge Distillation framework. Our DCKD consists of two parts: (a) Dynamic Contrastive Regularization (DCR), and (b) Distribution Mapping Module (DMM).

# Methodology

# Preliminaries

Given a low-quality image $I_{LQ}$ as input, the image restoration (IR) model $\mathcal{F}(\cdot)$ generates the corresponding high-quality image $I_{HQ}$ , which can be formulated as:

$$
I _ {H Q} = \mathcal {F} (I _ {L Q}; \theta), \tag {1}
$$

where $\theta$ represents the model parameters. The IR model $\mathcal{F}(\cdot ;\theta)$ is typically optimized using the L1 norm reconstruction loss, which is defined as:

$$
\mathcal {L} _ {r e c} = \left\| I _ {H Q} - I _ {G T} \right\| _ {1}, \tag {2}
$$

where the $I_{GT}$ is the ground-truth image. The vanilla knowledge distillation method adds a KD loss to minimize the difference between the student model and the teacher model:

$$
\mathcal {L} _ {k d} = \left\| \mathcal {F} _ {S} (I _ {L Q}; \theta_ {s}) - \mathcal {F} _ {T} (I _ {L Q}; \theta_ {t}) \right\| _ {1}, \tag {3}
$$

where $\mathcal{F}_{S}(I_{LQ};\theta_{s})$ and $\mathcal{F}_{T}(I_{LQ};\theta_{t})$ represent the outputs of student and teacher models, respectively.

# Dynamic Contrastive Regularization

Previous KD methods (Hinton, Vinyals, and Dean 2015; Wang et al. 2021d) employed a fixed solution space, which causes the lower-bound constraints on the student model to weaken in the later stages of training. To address this problem, we propose dynamic contrastive regularization (DCR), which dynamically adjusts the solution space based on the student model's state.

As shown in Fig. 2 (a), we first feed the input image $I_{LQ}$ into the dynamic negative sample generator, which consists of the degradation module and the history model. The degradation module applies random degradation operations on $I_{LQ}$ , generating N different degraded images $I_{dirty}$ . The history model $F_{S}^{his}$ then reconstructs these degraded images based on the historical state of the student model, producing N different negative images $I_{Neg}$ as the lower-bound of the solution space:

$$
I _ {N e g} ^ {1}, \dots , I _ {N e g} ^ {N} = \mathcal {F} _ {S} ^ {h i s} (\mathbb {D} _ {1} (I _ {L Q}), \dots , \mathbb {D} _ {N} (I _ {L Q})), \tag {4}
$$

where $\mathbb{D}_{N}(\cdot)$ and $I_{Neg}^{N}$ represent the N-th type of random degradation and the corresponding negative image, respectively.

For the upper-bound of the solution space, we not only rely on ground-truth $I_{GT}$ to optimize the student output, as described in Eq. 2 but also consider the output of teacher model $I_{HQ}^{T}$ as the positive image $I_{Pos}$ . At this point, we have defined the upper-bound and the dynamic lower-bound of the model, allowing us to construct a novel dynamic contrastive loss. We employ the pre-trained VQGAN (Esser, Rombach, and Ommer 2021) as the feature encoder. The dynamic contrastive loss is formulated as follows:

$$
\mathcal {L} _ {d c l} = \sum_ {i = 1} ^ {L} \lambda_ {i} \frac {\left| \left| f _ {i} ^ {A n c} - f _ {i} ^ {P o s} \right| \right| _ {1}}{\sum_ {j = 1} ^ {N} \left| \left| f _ {i} ^ {A n c} - f _ {i , j} ^ {N e g} \right| \right| _ {1}}, \tag {5}
$$

where $f^{Anc}$ , $f^{Pos}$ , and $f^{Neg}$ represent the features extracted from the output of the student model, positive image and negative images, respectively. i denotes the i-th layer of Encoder. $\lambda_{i}$ is the balancing weight for the i-th layer.

To better capture the state of the student model, we introduce exponential moving averages (EMA) to update the history model:

$$
\theta_ {h i s} = \alpha \theta_ {h i s} + (1 - \alpha) \theta_ {s t u}, \text {   s.t.   } t \% s = 0, \tag {6}
$$

where $\theta_{his}$ and $\theta_{stu}$ represent the parameters of the history model and the current student model, respectively. $\alpha$ is the attenuation rate, t is the current iteration, and s denotes the update step. During training, the update step s gradually increases.

# Distribution Mapping Module

Existing image restoration KD methods (Li et al. 2024; Jiang et al. 2024) only rely on L1-type loss to align the teacher

Feature map   
![](images/cb3c38e14b40e9056ea907451d690f4b035f7d4c379d0771acb2f86da190f127.jpg)  
(a) Image-level Distribution

Feature map   
![](images/7106042536c1ee6cd6eb50de97a947f48616a5bc5e51347a8821f0a00eb18b05.jpg)  
(b) Pixel-level Distribution   
Figure 3: Illustration of the Image-level Distribution and our Pixel-level Distribution (DMM).

and student models, thereby overlooking the distribution information of image content. However, high-level task KD methods (Hinton, Vinyals, and Dean 2015; Park et al. 2019; Huang et al. 2022) that align the entire output distribution of the teacher and student networks fail in low-level vision tasks. To address this, we design a distribution mapping module (DMM) to extract and align pixel-wise image distribution information, which is well-suited for pixel-level image restoration tasks.

As shown in Fig. 2 (b), we employ a pre-trained image encoder to extract deep features $F^{T}$ and $F^{S}$ from the output images of the teacher model $I_{HQ}^{T}$ and the student model $I_{HQ}^{S}$ , respectively:

$$
F ^ {T} = \text { Encoder } \left(I _ {H Q} ^ {T}\right), F ^ {S} = \text { Encoder } \left(I _ {H Q} ^ {S}\right). \tag {7}
$$

We assume that high-level KD methods struggle to provide fine-grained distribution constraints, which are crucial for restoring image details. Inspired by VQGAN (Esser, Rombach, and Ommer 2021), we use the codebook e pretrained on ImageNet (Krizhevsky, Sutskever, and Hinton 2012) to obtain the pixel-wised category distribution as illustrated in Fig. 3. We also employ the corresponding VQGAN as the image encoder. The codebook further transforms the extracted deep features $F^{T}$ and $F^{S}$ into category distributions can be formulated as:

$$
C ^ {T} = \psi (\| F ^ {T} - e \| _ {2} ^ {2}), C ^ {S} = \psi (\| F ^ {S} - e \| _ {2} ^ {2}), \tag {8}
$$

where $C^{T}$ and $C^{S}$ represent the pixel-wised category distributions of the teacher and student models, respectively. $\psi$ denotes the softmax operation.

Finally, we use cross-entropy loss to align $C^{T}$ and $C^{S}$ :

$$
\mathcal {L} _ {c e} = - \sum_ {i = 1} ^ {M} C _ {i} ^ {T} \log C _ {i} ^ {S}, \tag {9}
$$

where $C_{i}, i = 1, 2, \cdots M$ is category i. M is the total number of categories.

# Overall Loss

Following (Hinton, Vinyals, and Dean 2015; Zhang et al. 2023; Li et al. 2024), our DCKD also compute the reconstruction loss $L_{res}$ in Eq. 2 and the vanilla distillation loss $L_{kd}$ in Eq. 3. In addition, the dynamic contrastive loss $L_{dcl}$ in Eq. 5 and the cross-entropy loss $L_{ce}$ in Eq. 9 are also accumulated. The overall loss function can be expressed as:

$$
\mathcal {L} = \mathcal {L} _ {r e c} + \mathcal {L} _ {k d} + \lambda_ {d c l} \mathcal {L} _ {d c l} + \lambda_ {c e} \mathcal {L} _ {c e}, \tag {10}
$$

where the $\lambda_{dcl}$ and $\lambda_{ce}$ are the balancing parameters. The teacher model and the encoder are frozen during the training stage.

# Experiments

# Experimental Settings

Teacher Backbones The proposed DCKD is evaluated on three image restoration tasks: image super-resolution, image deblurring, and image deraining. Following (Li et al. 2024), we verify the effectiveness of DCKD on Transformer-based SwinIR (Liang et al. 2021) and CNN-based RCAN (Zhang et al. 2018) in image super-resolution. For image deblurring, we use NAFNet (Chen et al. 2022a) and Restormer (Zamir et al. 2022) as the teacher backbones. For image deraining, we employ Restormer as the backbone. The configuration details of teacher and student models are presented in Tab. 1.

<table><tr><td>Model</td><td>Role</td><td>Channel</td><td>Block/Group</td><td>#Params</td></tr><tr><td rowspan="2">SwinIR</td><td>Teacher</td><td>180</td><td>6/-</td><td>11.9M</td></tr><tr><td>Student</td><td>60</td><td>4/-</td><td>1.2M</td></tr><tr><td rowspan="2">RCAN</td><td>Teacher</td><td>64</td><td>20/10</td><td>15.6M</td></tr><tr><td>Student</td><td>64</td><td>6/10</td><td>5.2M</td></tr><tr><td rowspan="2">NAFNet</td><td>Teacher</td><td>32</td><td>36/-</td><td>17.1M</td></tr><tr><td>Student</td><td>16</td><td>22/-</td><td>2.7M</td></tr><tr><td rowspan="2">Restormer</td><td>Teacher</td><td>48</td><td>44/-</td><td>26.1M</td></tr><tr><td>Student</td><td>24</td><td>22/-</td><td>3.8M</td></tr></table>

Table 1: The specifications of teacher and student models.

Datasets and Evaluation For image super-resolution, DCKD is trained using 800 images from DIV2K (Timofte et al. 2017) and evaluated on four benchmark datasets. For image deblurring, the models are trained and tested both on GoPro dataset (Nah, Hyun Kim, and Mu Lee 2017b). For image deraining, we train DCKD on 13,712 clean-rainy image pairs collected from multiple datasets (Fu et al. 2017; Yang et al. 2017; Zhang, Sindagi, and Patel 2019; Li et al. 2016) and evaluate it on Test100 (Zhang, Sindagi, and Patel 2019), Rain100H (Yang et al. 2017), Rain100L (Yang et al. 2017), Test2800 (Fu et al. 2017), and Test1200 (Zhang and Patel 2018). We employ the PSNR and SSIM (Wang et al. 2004) metrics to evaluate the restoration performance. For image super-resolution and image deraining tasks, the metrics are computed on the Y channel in the YCbCr color space. For image deblurring, PSNR and SSIM are evaluated in the RGB color space.

Implementation Details For image super-resolution, the input is randomly cropped into $48 \times 48$ patches and augmented by random horizontal and vertical flips and rotations. All the models are trained using ADAM optimizer (Kingma and Ba 2014) with $\beta_{1} = 0.9$ , $\beta_{2} = 0.99$ , and $\epsilon = 10^{-8}$ . The training batch size is set to 16 with a total of $2.5 \times 10^{5}$ iterations. The initial learning rate is set to $10^{-4}$ and is decayed by a factor of 10 at every $10^{5}$ update. DCKD is implemented by PyTorch using 4 NVIDIA V100 GPUs. For other image restoration tasks, we strictly adhere to the original training configurations of each teacher model. More training configurations for other tasks are presented in the Appendix $^{1}$ .

<table><tr><td>Scale</td><td>Method</td><td>Model</td><td>Set5PSNR/SSIM</td><td>Set14PSNR/SSIM</td><td>BSD100PSNR/SSIM</td><td>Urban100PSNR/SSIM</td><td>Model</td><td>Set5PSNR/SSIM</td><td>Set14PSNR/SSIM</td><td>BSD100PSNR/SSIM</td><td>Urban100PSNR/SSIM</td></tr><tr><td rowspan="6">×2</td><td>Teacher</td><td></td><td>38.36/0.9620</td><td>34.14/0.9227</td><td>32.45/0.9030</td><td>33.40/0.9394</td><td></td><td>38.27/0.9614</td><td>34.13/0.9216</td><td>32.41/0.9027</td><td>33.34/0.9384</td></tr><tr><td>Scratch</td><td></td><td>38.00/0.9607</td><td>33.56/0.9178</td><td>32.19/0.9000</td><td>32.05/0.9279</td><td></td><td>38.13/0.9610</td><td>33.78/0.9194</td><td>32.26/0.9007</td><td>32.63/0.9327</td></tr><tr><td>Logits</td><td></td><td>38.04/0.9608</td><td>33.61/0.9184</td><td>32.22/0.9003</td><td>32.09/0.9282</td><td></td><td>38.17/0.9611</td><td>33.83/0.9197</td><td>32.29/0.9010</td><td>32.67/0.9329</td></tr><tr><td>FAKD</td><td></td><td>38.03/0.9608</td><td>33.63/0.9182</td><td>32.21/0.9001</td><td>32.06/0.9279</td><td></td><td>38.17/0.9612</td><td>33.83/0.9199</td><td>32.29/0.9011</td><td>32.65/0.9330</td></tr><tr><td>MiPKD</td><td></td><td>38.14/0.9611</td><td>33.76/0.9194</td><td>32.29/0.9011</td><td>32.46/0.9313</td><td></td><td>38.21/0.9613</td><td>33.92/0.9203</td><td>32.32/0.9015</td><td>32.83/0.9344</td></tr><tr><td>DCKD</td><td></td><td>38.17/0.9613</td><td>33.85/0.9205</td><td>32.31/0.9017</td><td>32.59/0.9330</td><td></td><td>38.25/0.9615</td><td>34.01/0.9213</td><td>32.35/0.9019</td><td>32.92/0.9351</td></tr><tr><td rowspan="6">×3</td><td>Teacher</td><td rowspan="6">SwinIR</td><td>34.89/0.9312</td><td>30.77/0.8503</td><td>29.37/0.8124</td><td>29.29/0.8744</td><td rowspan="6">RCAN</td><td>34.74/0.9299</td><td>30.65/0.8482</td><td>29.32/0.8111</td><td>29.09/0.8702</td></tr><tr><td>Scratch</td><td>34.41/0.9273</td><td>30.43/0.8437</td><td>29.12/0.8062</td><td>28.20/0.8537</td><td>34.61/0.9288</td><td>30.45/0.8444</td><td>29.18/0.8074</td><td>28.59/0.8610</td></tr><tr><td>Logits</td><td>34.44/0.9275</td><td>30.45/0.8443</td><td>29.14/0.8066</td><td>28.23/0.8545</td><td>34.61/0.9291</td><td>30.47/0.8447</td><td>29.21/0.8080</td><td>28.62/0.8612</td></tr><tr><td>FAKD</td><td>34.42/0.9273</td><td>30.42/0.8437</td><td>29.12/0.8062</td><td>28.18/0.8533</td><td>34.63/0.9290</td><td>30.51/0.8453</td><td>29.21/0.8079</td><td>28.62/0.8612</td></tr><tr><td>MiPKD</td><td>34.53/0.9283</td><td>30.52/0.8456</td><td>29.19/0.8079</td><td>28.47/0.8591</td><td>34.72/0.9296</td><td>30.55/0.8458</td><td>29.25/0.8087</td><td>28.76/0.8640</td></tr><tr><td>DCKD</td><td>34.59/0.9291</td><td>30.57/0.8475</td><td>29.22/0.8093</td><td>28.63/0.8633</td><td>34.74/0.9299</td><td>30.60/0.8472</td><td>29.29/0.8099</td><td>28.87/0.8662</td></tr><tr><td rowspan="7">×4</td><td>Teacher</td><td rowspan="7"></td><td>32.72/0.9021</td><td>28.94/0.7914</td><td>27.83/0.7459</td><td>27.07/0.8164</td><td rowspan="7"></td><td>32.63/0.9002</td><td>28.87/0.7889</td><td>27.77/0.7436</td><td>26.82/0.8087</td></tr><tr><td>Scratch</td><td>32.31/0.8955</td><td>28.67/0.7833</td><td>27.61/0.7379</td><td>26.15/0.7884</td><td>32.38/0.8971</td><td>28.69/0.7842</td><td>27.63/0.7379</td><td>26.36/0.7947</td></tr><tr><td>Logits</td><td>32.27/0.8954</td><td>28.67/0.7833</td><td>27.62/0.7380</td><td>26.15/0.7887</td><td>32.45/0.8980</td><td>28.76/0.7860</td><td>27.67/0.7400</td><td>26.49/0.7982</td></tr><tr><td>FAKD</td><td>32.22/0.8950</td><td>28.65/0.7831</td><td>27.61/0.7380</td><td>26.09/0.7870</td><td>32.46/0.8980</td><td>28.77/0.7860</td><td>27.68/0.7400</td><td>26.50/0.7980</td></tr><tr><td>MiPKD</td><td>32.39/0.8971</td><td>28.76/0.7854</td><td>27.68/0.7403</td><td>26.37/0.7956</td><td>32.46/0.8982</td><td>28.77/0.7860</td><td>27.69/0.7402</td><td>26.55/0.7998</td></tr><tr><td>DCKD</td><td>32.49/0.8991</td><td>28.82/0.7877</td><td>27.72/0.7422</td><td>26.53/0.8007</td><td>32.56/0.8995</td><td>28.82/0.7877</td><td>27.73/0.7423</td><td>26.69/0.8041</td></tr><tr><td>DCKD*</td><td>32.51/0.8992</td><td>28.88/0.7890</td><td>27.74/0.7430</td><td>26.62/0.8032</td><td>32.58/0.8996</td><td>28.86/0.7885</td><td>27.74/0.7425</td><td>26.74/0.8054</td></tr></table>

Table 2: Quantitative comparison on the benchmark datasets for image super-resolution. The best and second-best performances are highlighted in bold and underlined, respectively. The FAKD results on SwinIR are from our reproduction experiments.

<table><tr><td>Model</td><td>Method</td><td>GoPro PSNR/SSIM</td><td>#Params</td></tr><tr><td>MT-RNN</td><td>-</td><td>31.15/0.9450</td><td>2.6M</td></tr><tr><td>DMPHN</td><td>-</td><td>31.20/0.9400</td><td>21.7M</td></tr><tr><td rowspan="4">NAFNet</td><td>Teacher</td><td>32.87/0.9606</td><td>17.1M</td></tr><tr><td>Scratch</td><td>31.17/0.9457</td><td>2.7M</td></tr><tr><td>Logits</td><td>31.26/0.9464</td><td>2.7M</td></tr><tr><td>DCKD</td><td>31.43/0.9487</td><td>2.7M</td></tr><tr><td rowspan="4">Restormer</td><td>Teacher</td><td>32.92/0.9610</td><td>26.1M</td></tr><tr><td>Scratch</td><td>31.57/0.9497</td><td>3.8M</td></tr><tr><td>Logits</td><td>31.61/0.9501</td><td>3.8M</td></tr><tr><td>DCKD</td><td>31.78/0.9521</td><td>3.8M</td></tr></table>

Table 3: Quantitative comparison for image deblurring.

# Results and Comparison

Image Super-Resolution We compare our framework with the representative KD methods: train from scratch, Logits (Hinton, Vinyals, and Dean 2015), FAKD (He et al. 2020), and MiPKD (Li et al. 2024), on ×2, ×3, and ×4 super-resolving scales.

The quantitative results for SwinIR and RCAN are presented in Tab. 2. Existing KD methods provide limited improvement for the student models, and in some cases, they even lead to worse performance compared to models trained without KD. For example, using FAKD for distillation on Urban100 results in worse performance than training the model from scratch on SwinIR. DCKD is effective for both Transformer-based and CNN-based architectures, significantly outperforming existing KD methods by more than 0.1dB across all three scales on Urban100.

![](images/fcb3de50bcfd98e89f2cf30812769b37ee0ebf6714b38d17660f5066833306eb.jpg)

<details>
<summary>text_image</summary>

Input
</details>

![](images/f4775bdf80cad1ebcbc46e01bdc42bbb529cf7b9e7c12956f98e6dea8e198704.jpg)

<details>
<summary>text_image</summary>

Scratch (21.32)
DCKD (28.07)
</details>

![](images/f30f2a3eb2f2782241649101e679717f5418ad6d00fc99a2010dc1204700e720.jpg)

<details>
<summary>text_image</summary>

Logits (23.55)
GT (PSNR)
</details>

Figure 4: Visual comparison for image deblurring.

We deliberately use the most straightforward approach to demonstrate DCKD, showing that dynamic lower-bound constraints can yield strong results even without improving upper-bound constraints. To demonstrate that DCKD can be combined with methods that optimize upper-bound constraints, we incorporate DUKD (Zhang et al. 2023) into the DCKD framework to enhance upper-bound constraints, resulting in DCKD\*. As we can see, the proposed DCKD can be combined with existing KD methods that optimize the upper bound, further significantly enhancing performance. DCKD\* significantly outperforms the SOTA method MiPKD by 0.25dB on SwinIR and 0.19dB on RCAN at ×4.

Fig. 5 and Fig. 6 present challenging visual examples for Transformer and CNN backbones, respectively. Compared to existing KD methods, our approach enables the student models to better capture structural textures, such as more accurately reconstructing sidewalk lines and building structures. More visual comparisons for various examples and models are presented in the Appendix.

![](images/9aa3a4fc7e2e8a5e91ba84cd7232fd7ae1be724bb3d576921a25aba515897116.jpg)

Figure 5: The visual comparison of distilling SwinIR on Urban100 for ×4 SR.   
![](images/fcafb673d90315e13338ad8e81b5440aa4c90d3ccc18f840c7f7b1a0f0864494.jpg)

<details>
<summary>natural_image</summary>

Interior view of a server room with rows of black server racks and a person walking nearby (no visible text or symbols)
</details>

img004

![](images/d72707886b388ec662a4c4527af11558a4285d6a85d61172c5c9cbbc3632a7b4.jpg)  
GT (PSNR)

![](images/7900e5f6ceda111d46e7fd14f54c9e9d2bfb04a4205999e69617d5d254c658fa.jpg)  
Scratch (22.02)

![](images/9b96759c82c4e6e2ed50b51737cc06420a9146a5da27ae1a19ffa3bfc32b42ae.jpg)  
Logits (22.04)

![](images/c73e008818ced0a0e74e9cd91477a6c04a90b9aa700cb822bd5ba0d120877437.jpg)  
FAKD (22.38)

![](images/f6f0f31ba8bf1dc0ac2170412ada12e84ad80e624c7bf21bf697aa4f64114a54.jpg)  
MiPKD (22.54)

![](images/c6e655f5dd1f7b16c66c37d35e9fdea6382895d2e57a17b9fb8300594de46eab.jpg)  
DCKD (23.91)

![](images/6dea79e0961eebb821cdc80b8c2b4dae81034ae6e9cc55bc8949fe02c05c0e94.jpg)

<details>
<summary>natural_image</summary>

Interior view of a large server room with rows of server racks and a red-labeled storage cabinet (no visible text or symbols)
</details>

img078

![](images/45ce048b8b07371273587b0fd4fd3b075bc57b1017604398c14515d89a1ab2ed.jpg)  
GT (PSNR)

![](images/ad652eadd0aed9ca5366afc1e762979e886b36d89aca4fbe5beac5c435fa6e97.jpg)  
Scratch (18.82)

![](images/ec29d799130bc26a91b99f222468a559c045e039b792e529810ad8c1b70ff86f.jpg)  
Logits (19.97)

![](images/e96a898ac49b1ae7800fdee6a7af5c4ac15838ef98770543173bb2688cbf1547.jpg)  
FAKD (20.38)

![](images/04a6aba377f4a6f582432764cd90eb849441f0ac1c195a6e8bae230ec5825c60.jpg)  
MiPKD (19.13)

![](images/fdfc120ed2a7825b754efcce0336b61327dae8400ae24d6da55bb765795d5aa8.jpg)  
DCKD (20.96)

Figure 6: The visual comparison of distilling RCAN on Urban100 for ×4 SR. 

<table><tr><td>Model</td><td>Method</td><td>Test100 PSNR/SSIM</td><td>Rain100H PSNR/SSIM</td><td>Rain100L PSNR/SSIM</td><td>Test2800 PSNR/SSIM</td><td>Test1200 PSNR/SSIM</td><td>#Params</td></tr><tr><td>MPRNet</td><td>-</td><td>30.27/0.8970</td><td>30.41/0.8900</td><td>36.40/0.9650</td><td>33.64/0.9380</td><td>32.91/0.9160</td><td>20.1M</td></tr><tr><td>SPAIR</td><td>-</td><td>30.35/0.9090</td><td>30.95/0.8920</td><td>36.93/0.9690</td><td>33.34/0.9360</td><td>33.04/0.9220</td><td>-</td></tr><tr><td rowspan="4">Restormer</td><td>Teacher</td><td>32.02/0.9237</td><td>31.48/0.9054</td><td>39.08/0.9785</td><td>34.21/0.9449</td><td>33.22/0.9270</td><td>26.1M</td></tr><tr><td>Scratch</td><td>31.01/0.9122</td><td>30.51/0.8932</td><td>37.47/0.9714</td><td>33.78/0.9396</td><td>33.67/0.9295</td><td>3.8M</td></tr><tr><td>Logits</td><td>31.04/0.9143</td><td>30.48/0.8915</td><td>37.17/0.9712</td><td>33.81/0.9399</td><td>33.78/0.9310</td><td>3.8M</td></tr><tr><td>DCKD</td><td>31.08/0.9167</td><td>30.54/0.8969</td><td>38.02/0.9762</td><td>33.91/0.9411</td><td>33.95/0.9326</td><td>3.8M</td></tr></table>

Table 4: Quantitative comparison on the benchmark datasets for image deraining.

Image Deblurring Tab. 3 provides a quantitative comparison on GoPro dataset. Our method demonstrates consistent effectiveness across both CNN-based NAFNet and Transformer-based Restormer. Compared to the Logits KD, our DCKD achieves 0.17dB improvement on different backbones. Moreover, with comparable parameters, the student model of NAFNet significantly outperforms MT-RNN (Park et al. 2020) by 0.28dB. Fig. 4 illustrates the deblurring visualization results. DCKD restores the clearest window outlines, significantly enhancing the deblurring capability of the student model.

Image Deraining Tab. 4 shows the performance of various methods on several benchmark datasets for image deraining. With only $18.9\%$ of MPRNet's (Zamir et al. 2021) parameters, DCKD significantly surpasses it by 1.8dB on Rain100L dataset. Additionally, compared to other distillation methods, DCKD outperforms the Logits KD by 0.55dB on Rain100L. Fig. 7 provides a visual comparison of de

![](images/f352e72c2801e79f37930c0a0f5b34ed649d2eafb312333e11c9b95baa368823.jpg)

<details>
<summary>natural_image</summary>

Coastal landscape with a small boat on calm water, a red box highlighting a coastal cliff face (no text or symbols)
</details>

Input

![](images/d198f2ec8d8a200d5d1e02a26d3877a0c230a84d2bc829cf98efdf4d0825bb15.jpg)  
Scratch (38.70)

![](images/abdecbbcd6168a5dd39f18ba050a38604ab54d10c8ebdd25f116d869a61c2cd1.jpg)  
Logits (41.37)

![](images/7cb8f3b776377ff32c472af70f5b8c384350e3033e5b67e5609e4065b91d122f.jpg)

DCKD (43.12)   
![](images/fb94454f5448a55c4670c080b0f70bca0cc4c1a3237291877f0fb7c2517b09bb.jpg)

![](images/f3587a1e9b7b88a3795fb1cbe1f1ca6b9591098e8697e7c09ea6c394ad976f97.jpg)  
GT (PSNR)   
Figure 7: Visual comparison for image deraining.

rained images. DCKD further enhances the student's ability to remove rain streaks compared to the logits distillation method.

<table><tr><td>DCR</td><td>DMM</td><td>Set14PSNR/SSIM</td><td>Urban100PSNR/SSIM</td></tr><tr><td>✘</td><td>✘</td><td>33.83/0.9197</td><td>32.67/0.9329</td></tr><tr><td>✓</td><td>✘</td><td>33.98/0.9208</td><td>32.83/0.9346</td></tr><tr><td>✘</td><td>✓</td><td>33.92/0.9205</td><td>32.81/0.9343</td></tr><tr><td>✓</td><td>✓</td><td>34.01/0.9213</td><td>32.92/0.9351</td></tr></table>

Table 5: Ablation study on components of our framework.

<table><tr><td>Degradation Type</td><td>Set14PSNR/SSIM</td><td>Urban100PSNR/SSIM</td></tr><tr><td>Random Blur</td><td>34.01/0.9212</td><td>32.89/0.9352</td></tr><tr><td>Random Noise</td><td>34.01/0.9213</td><td>32.92/0.9351</td></tr><tr><td>Random Resize</td><td>33.99/0.9212</td><td>32.90/0.9354</td></tr><tr><td>Random Mix</td><td>34.00/0.9212</td><td>32.87/0.9353</td></tr></table>

Table 6: Ablation study on the degradation module.

# Ablation Study

For ablation experiments, we train DCKD on RCAN for the SR task with a scaling factor of $\times2$ . We then validate the results on Set14 and Urban100 datasets.

Components of the Proposed Framework As shown in Tab. 5, we first conduct an ablation study on the two main modules of DCKD. The results indicate that our DCR and DMM outperform the baseline by 0.16dB and 0.14dB in PSNR on Urban100, respectively. This demonstrates the effectiveness of the proposed modules. Furthermore, the combination of DCR and DMM further enhances model performance, achieving PSNR improvements of 0.18dB on Set14 and 0.25dB on Urban100 compared to the baseline.

Impact of the Degradation Module We conduct an ablation study on the impact of the degradation module in the Dynamic Negative Sample Generator (DNSG). Following Real-ESRGAN (Wang et al. 2021c), the degradation module is simple to implement by adding the Gaussian blur, Gaussian noise, resize operation (i.e., downsampling and then upsampling), or mixed degradation. The degradation results are summarized in Tab. 5 and Tab. 6. We observe that the addition degradation module (in Tab. 6) achieves at least 0.06dB PSNR improvement on Urban100, compared to that without degradation (only DMM in Tab. 5). Furthermore, the degradation with random noise achieves the highest PSNR of 32.92dB, which outperforms randomly mixed degradation by 0.05dB PSNR.

Impact of the Balancing Weights We investigate the impact of the balancing coefficients $\lambda_{dcl}$ and $\lambda_{ce}$ in Equ. 10, as shown in Tab. 7. We find that excessively large or small values for these coefficients negatively affect the outcomes. The experiments indicate that the model achieves optimal results when $\lambda_{dcl}$ is set to 0.1 and $\lambda_{ce}$ to 0.001. Given the broad applicability of our method to various image restoration tasks, we adopt $\lambda_{dcl} = 0.1$ and $\lambda_{ce} = 0.001$ as the default settings across different tasks.

<table><tr><td> $\lambda_{dcl}$ </td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>PSNR/SSIM</td><td>32.87/0.9346</td><td>32.92/0.9351</td><td>32.81/0.9350</td></tr><tr><td> $\lambda_{ce}$ </td><td>0.0001</td><td>0.001</td><td>0.01</td></tr><tr><td>PSNR/SSIM</td><td>32.84/0.9345</td><td>32.92/0.9351</td><td>32.82/0.9341</td></tr></table>

Table 7: Ablation study on the balancing weights.

![](images/1fdb125dc4ce57376ed0bd75f431556f6db900eecf662f4946e013ee99a0b4ef.jpg)

<details>
<summary>bar</summary>

| Number of Negative Samples | PSNR(dB) |
| -------------------------- | -------- |
| 3                          | 32.82    |
| 4                          | 32.85    |
| 5                          | 32.92    |
| 6                          | 32.93    |
| 7                          | 32.93    |
</details>

![](images/6283942613fac4c760a2580575ad5c96cf06a3a9da55800305f615eb96ca94ba.jpg)

<details>
<summary>bar</summary>

| the initial update step | PSNR(dB) |
| ----------------------- | -------- |
| 100                     | 32.79    |
| 500                     | 32.82    |
| 1000                    | 32.92    |
| 2000                    | 32.84    |
| 5000                    | 32.83    |
</details>

Figure 8: Ablation studies on the number of negative samples and the initial update step.

Impact of the Number of Negative Samples The impact of the number of negative samples is reported in Fig. 8 (a). The results indicate that as the number of negative samples increases, the performance improves consistently. However, when the number of negative samples exceeds 5, the performance gains diminish while significantly increasing training time and memory costs. Therefore, the number of negative samples is set to 5, which achieves the best trade-off between PSNR and training time.

Impact of the Initial Update Step In Fig. 8 (b), we investigate the impact of the initial update step for the history model within the dynamic negative sample generator. The experimental results show that when using a smaller step to update the historical model, the quality of the negative samples becomes very close to, or even surpasses, that of the anchor points, leading to instability in the solution space and a decline in performance. Conversely, when using a larger step, the negative sample quality deteriorates, weakening the lower bound constraint. When the initial update step is set to 1000 achieve the best performance.

# Conclusion

In this work, we propose a dynamic contrastive knowledge distillation framework for image restoration, named DCKD, which consists of the Dynamic Contrastive Regularization (DCR) and the Distribution Mapping Module (DMM). Most previous knowledge distillation methods utilize a fixed solution space, causing the lower bound constraints to weaken gradually during training. DCR constructs a dynamic solution space based on the student's learning state to enhance the lower-bound constraints. DMM introduces pixel-level category information to knowledge distillation for low-level vision tasks for the first time. Experiments on image super-resolution, image deblurring, and image deraining tasks validate that the proposed DCKD achieves state-of-the-art results on various benchmark datasets, both quantitatively and visually.

<table><tr><td>Model</td><td>Scale</td><td>Method</td><td>Set14PSNR/SSIM</td><td>BSD100PSNR/SSIM</td><td>Urban100PSNR/SSIM</td><td>Manga109PSNR/SSIM</td></tr><tr><td rowspan="3">SwinIR-light</td><td rowspan="3"> $\times 4$ </td><td>-</td><td>28.77/0.7858</td><td>27.69/0.7406</td><td>26.47/0.7980</td><td>30.92/0.9151</td></tr><tr><td>MCLIR</td><td>28.85/0.7874</td><td>27.72/0.7414</td><td>26.57/0.8010</td><td>31.04/0.9158</td></tr><tr><td>DCKD</td><td>28.88/0.7891</td><td>27.75/0.7432</td><td>26.64/0.8039</td><td>31.20/0.9180</td></tr></table>

Table 8: Quantitative comparison DCKD with MCLIR (Wu et al. 2024) on the benchmark datasets for SR task.

![](images/02585411fb0cca565348021d10e0a5f688bfdc917f83431238dd41e9df3e62d4.jpg)  
Figure 9: More visual comparison of image super-resolution at ×4.

# Acknowledgments

This work is supported by the National Natural Science Foundation of China (NO. 62102151), the Open Research Fund of Key Laboratory of Advanced Theory and Application in Statistics and Data Science, Ministry of Education (KLATASDS2305), the Fundamental Research Funds for the Central Universities.

# Appendix

# Implementation Details

The details of image encoder. For the latent features, we extract the features from five different layers of the pre-trained VQGAN (Esser, Rombach, and Ommer 2021), while with the corresponding coefficients $\lambda_{i}, i = 1, \cdots 5$ to $\frac{1}{32}, \frac{1}{16}, \frac{1}{8}, \frac{1}{4}$ and 1, respectively. We set the attenuation rate $\alpha$ to 0.1.

The details of different backbones. For NAFNet (Chen et al. 2022a), we train the model with AdamW optimizer

![](images/6e90df434b9b497f77c87b54d582ac132037dcb2cdc4b8ab6d5b1720df581ecb.jpg)

<details>
<summary>text_image</summary>

Street photo with visible store signboards and a red overlay on a vehicle
</details>

Input

![](images/55a619c4074a13acfb5cac5edf78efef3a9a7d4dd740a536609a3b8d96a755b8.jpg)  
Scratch (27.90)

![](images/ef0f0922aa238b53b6dfb8b5e76faff826078645acbeebe48dbf6b6633748b18.jpg)  
Logits (28.60)

![](images/d313c0b8cac0dbafa97ae0153e85cb959174eb59a0c7afe9c81712497b21ba1d.jpg)

<details>
<summary>text_image</summary>

NABEER
ROAD
</details>

Input

![](images/bf885ecfb99ad9746b3665d861989eccc1780ff1b0d930fca1b88085e172a933.jpg)  
DCKD (30.47)

![](images/e9fd8648feb71abd3c3111abb310a264bacfae4138979d7aa986f71cf254c4fc.jpg)  
GT (PSNR)

![](images/1d20088c778935e6b4d3c8dfccbe87fe626a2db81e58eeba2e767125748507aa.jpg)  
Scratch (28.57)

![](images/0dfdf5062ad9eece93f6b91546a246f6b18c8458fff6a5f9ba7f76851457a1b2.jpg)  
Logits (27.74)

![](images/c9fd057bbf863e393991cff41bdfbeec64e088ee5a3f3663a1fc8fcd15069c6b.jpg)  
DCKD (34.14)

![](images/725152b2ae540cd5a811e9bf2326266a6cf8126a42b286d78aca20c56225d3ee.jpg)  
GT (PSNR)

![](images/8ca737bfc3080a3c8f0f35738c1575b9a4055ea5f87cfc09469dc12f6b15d298.jpg)

<details>
<summary>text_image</summary>

63th 4823
</details>

Input

![](images/e7a70020102e58897e26991e2a1b5a58e7aea4487fe272cea362c11cec082aca.jpg)  
Scratch (29.17)

![](images/2f238d76d1b6df2c06685e4b34672ae10c8883fefd35b0750c1a2adaae0d203a.jpg)  
Logits (29.48)

![](images/bc74b97494ee8ebe0fdb1d9abb5f3c8f04d646f6d1716a9329aac7cc9e928c18.jpg)  
DCKD (29.66)

![](images/e42de12c7520623eeb06bbfedf325907aef944c8f84d8689d2b0f0edbd6024c1.jpg)  
GT (PSNR)

![](images/ebb19e5d87ab70a7363ab019f5a8be7b76677296522a9470f9e98302bdcbfa47.jpg)

![](images/0ce845e52104ba6345af61c40d4100ddef98963c7c8b0b0925e666bc80c1f38c.jpg)  
Input

![](images/721471758955867449809e91d25e56d15bf13d2d873f9b9e9e8c9702f0b931f4.jpg)  
Scratch (23.22)

![](images/1c6a6fa216d9c1d9bf1c0c9333c69bd94ec820c8b7f81e6cf9d89cf240b44e1a.jpg)  
Logits (22.42)

![](images/2983230ccbc5e3c04884662e8ddbb8841ea421a9a5612d2cfe493d9ac638fcb7.jpg)  
DCKD (26.52)

![](images/e2878069d6c422fc431227c4ff18ecd6a649401525a70d5b4fc15c4f6b2aab60.jpg)  
GT (PSNR)

Figure 10: More visual comparison of image deblurring.   
![](images/2c90131747538077f330d734dfee5d943f9a63c82d5b557cada28c53052947bd.jpg)

<details>
<summary>natural_image</summary>

Two polar bears interacting in snowy terrain, one touching the mouth (no text or symbols visible)
</details>

Input

![](images/6d2f291dbfc353b1fc6042ce13ce2dbb83699b127be78651bbce126a16d23863.jpg)  
Scratch (37.75)

![](images/201fdc41cf3d7ed6118f1c0e2e90f162c100f4395ecacfa534b4520203077310.jpg)  
Logits (39.98)

![](images/2ad3481b74d78b983033316cfadbc84d14ee72e108aafb3484b7b89c36e509f0.jpg)  
DCKD (40.99)

![](images/02a82c091c07930b1538002c0fad1380658c8e94769020e27ab525b1eeca3e5c.jpg)  
GT (PSNR)

![](images/6c114f590e0e752eb151042c935c71040f31aecee9ecebb2dfd633e765ac5038.jpg)

<details>
<summary>natural_image</summary>

A farmer in a straw hat and straw hat cattle in a field, with a red box highlighting a specific area (no text or symbols visible)
</details>

Input

![](images/6656ab13a9e63f909483113415b3d220524a8a86d4c8b387223cc551fe01ab20.jpg)  
Scratch (25.49)

![](images/cc89dad7fec5a1e7c23928d76bf488937f6de69fc230e9c989be836ec99c70cd.jpg)  
Logits (26.94)

![](images/780f46137a8f00b838b38e98a7be5f9976eca7af6dc112996f2c4ea275177fa7.jpg)  
DCKD (28.20)

![](images/537eabbf480d90f65a67cd37ce59e861cdecbc24f093707a6b8118b1d0648d31.jpg)  
GT (PSNR)   
Figure 11: More visual comparison of image deraining.

$(\beta_{1} = 0.9, \beta_{2} = 0.9, \text{weight decay} = 10^{-3})$ for total $2 \times 10^{5}$ iterations with the initial learning rate $10^{-3}$ gradually reduced to $10^{-7}$ with the cosine annealing schedule (Loshchilov and Hutter 2016). The training patch size is $256 \times 256$ and the batch size is 32.

For Restormer (Zamir et al. 2022), we train the model with AdamW optimizer ( $\beta_{1} = 0.9$ , $\beta_{2} = 0.999$ , weight decay $= 10^{-4}$ ) for $3 \times 10^{5}$ iterations with the initial learning rate $3 \times 10^{-4}$ gradually reduced to $1 \times 10^{-6}$ with the cosine annealing (Loshchilov and Hutter 2016). We start training with patch size $128 \times 128$ and batch size 64 for progressive learning. The patch size and batch size pairs are updated to $[(160^{2}, 40), (192^{2}, 32), (256^{2}, 16), (320^{2}, 8), (384^{2}, 8)]$ at iterations [92K, 156K, 204K, 240K, 276K]. For data augmentation, we use horizontal and vertical flips.

# Comparison of DCKD with Contrastive Learning

The quantitative comparison of DCKD with contrastive learning on the benchmark datasets for the SR task is reported in Tab. 8. DCKD significantly surpasses state-of-the-art method MCLIR (Wu et al. 2024) 0.16dB on Manga109. This shows that DCKD provides more comprehensive lower-bound constraints, ensuring that each negative sample exerts a consistent force on the anchor sample. Additionally, our method requires only a single negative sample model to generate an arbitrary number of negative samples, significantly reducing computational overhead.

# More Visual Comparison

We provide more visual comparisons. Fig. 9 presents some challenging images from the x4 super-resolution task. As observed, previous distillation methods often result in blurred artifacts and a struggle to effectively restore high-frequency details such as lines and architectural structures. For example, in img002 and img026, previous methods produce significant artifacts between lines or blur the gaps between them. In contrast, our DCKD not only removes these artifacts but also clearly separates the lines. Overall, our

method enhances the student model's ability to handle high-frequency details and accurately restore them.

Further visual comparisons for deblurring and deraining tasks are shown in Fig. 10 and Fig. 11, where our DCKD effectively removes motion blur and rain, with the restored image quality closely matching that of the ground truth (GT).

# References

Ahn, N.; Kang, B.; and Sohn, K.-A. 2018. Fast, accurate, and lightweight super-resolution with cascading residual network. In Proceedings of the European conference on computer vision (ECCV), 252–268.   
Ayazoglu, M. 2021. Extremely lightweight quantization robust real-time single-image super resolution for mobile devices. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2472–2479.   
Chen, H.; Wang, Y.; Guo, T.; Xu, C.; Deng, Y.; Liu, Z.; Ma, S.; Xu, C.; Xu, C.; and Gao, W. 2021. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12299–12310.   
Chen, L.; Chu, X.; Zhang, X.; and Sun, J. 2022a. Simple baselines for image restoration. In European conference on computer vision, 17–33. Springer.   
Chen, X.; Cao, Q.; Zhong, Y.; Zhang, J.; Gao, S.; and Tao, D. 2022b. Dearkd: data-efficient early knowledge distillation for vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12052–12062.   
Chen, X.; Wang, X.; Zhou, J.; Qiao, Y.; and Dong, C. 2023. Activating more pixels in image super-resolution transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 22367–22377.   
Chen, Z.; Zhang, Y.; Gu, J.; Kong, L.; Yuan, X.; et al. 2022c. Cross aggregation transformer for image restoration. Advances in Neural Information Processing Systems, 35: 25478–25490.   
Dong, C.; Loy, C. C.; He, K.; and Tang, X. 2015. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2): 295–307.   
Du, Z.; Liu, J.; Tang, J.; and Wu, G. 2021. Anchor-based plain net for mobile image super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2494–2502.   
Esser, P.; Rombach, R.; and Ommer, B. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.   
Fan, Y.; Yu, J.; Mei, Y.; Zhang, Y.; Fu, Y.; Liu, D.; and Huang, T. S. 2020. Neural sparse representation for image restoration. Advances in Neural Information Processing Systems, 33: 15394–15404.   
Fu, H.; Zhou, S.; Yang, Q.; Tang, J.; Liu, G.; Liu, K.; and Li, X. 2021. LRC-BERT: latent-representation contrastive knowledge distillation for natural language understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 12830–12838.   
Fu, X.; Huang, J.; Zeng, D.; Huang, Y.; Ding, X.; and Paisley, J. 2017. Removing rain from single images via a deep detail network. In Proceedings of the IEEE conference on computer vision and pattern recognition, 3855–3863.   
Gou, J.; Yu, B.; Maybank, S. J.; and Tao, D. 2021. Knowledge distillation: A survey. International Journal of Computer Vision, 129(6): 1789–1819.

Gou, Y.; Li, B.; Liu, Z.; Yang, S.; and Peng, X. 2020. Clearer: Multi-scale neural architecture search for image restoration. Advances in neural information processing systems, 33: 17129–17140.   
Hahn, S.; and Choi, H. 2019. Self-knowledge distillation in natural language processing. arXiv preprint arXiv:1908.01851.   
He, W.; Yao, Q.; Yokoya, N.; Uezato, T.; Zhang, H.; and Zhang, L. 2022. Spectrum-aware and transferable architecture search for hyperspectral image restoration. In European Conference on Computer Vision, 19–37. Springer.   
He, Z.; Dai, T.; Lu, J.; Jiang, Y.; and Xia, S.-T. 2020. Fakd: Feature-affinity based knowledge distillation for efficient image super-resolution. In 2020 IEEE International Conference on Image Processing (ICIP), 518–522. IEEE.   
Hinton, G.; Vinyals, O.; and Dean, J. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.   
Hong, C.; Baik, S.; Kim, H.; Nah, S.; and Lee, K. M. 2022. Cadyq: Content-aware dynamic quantization for image super-resolution. In European Conference on Computer Vision, 367–383. Springer.   
Huang, T.; You, S.; Wang, F.; Qian, C.; and Xu, C. 2022. Knowledge distillation from a stronger teacher. Advances in Neural Information Processing Systems, 35: 33716–33727.   
Jiang, Y.; Feng, C.; Zhang, F.; and Bull, D. 2024. MTKD: Multi-Teacher Knowledge Distillation for Image Super-Resolution. arXiv preprint arXiv:2404.09571.   
Kim, H.; Baik, S.; Choi, M.; Choi, J.; and Lee, K. M. 2021. Searching for controllable image restoration networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14234–14243.   
Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.   
Krizhevsky, A.; Sutskever, I.; and Hinton, G. E. 2012. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.   
Lee, W.; Lee, J.; Kim, D.; and Ham, B. 2020. Learning with privileged information for efficient image super-resolution. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIV 16, 465–482. Springer.   
Lefkimmiatis, S. 2017. Non-local color image denoising with convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 3587–3596.   
Li, C.; Guo, J.; Porikli, F.; Fu, H.; and Pang, Y. 2018. A cascaded convolutional neural network for single image dehazing. IEEE Access, 6: 24877–24887.   
Li, S.; Zhang, Y.; Li, W.; Chen, H.; Wang, W.; Jing, B.; Lin, S.; and Hu, J. 2024. Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution. arXiv preprint arXiv:2404.02573.   
Li, Y.; Tan, R. T.; Guo, X.; Lu, J.; and Brown, M. S. 2016. Rain streak removal using layer priors. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2736–2744.   
Liang, J.; Cao, J.; Sun, G.; Zhang, K.; Van Gool, L.; and Timofte, R. 2021. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, 1833–1844.   
Lim, B.; Son, S.; Kim, H.; Nah, S.; and Mu Lee, K. 2017. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 136–144.

Lin, S.; Xie, H.; Wang, B.; Yu, K.; Chang, X.; Liang, X.; and Wang, G. 2022. Knowledge distillation via the target-aware transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10915–10924.   
Loshchilov, I.; and Hutter, F. 2016. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983.   
Luo, Y.; Huang, Q.; Ling, J.; Lin, K.; and Zhou, T. 2023. Local and global knowledge distillation with direction-enhanced contrastive learning for single-image deraining. Knowledge-Based Systems, 268: 110480.   
Nah, S.; Hyun Kim, T.; and Mu Lee, K. 2017a. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE conference on computer vision and pattern recognition, 3883–3891.   
Nah, S.; Hyun Kim, T.; and Mu Lee, K. 2017b. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE conference on computer vision and pattern recognition, 3883–3891.   
Oh, J.; Kim, H.; Nah, S.; Hong, C.; Choi, J.; and Lee, K. M. 2022. Attentive fine-grained structured sparsity for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 17673–17682.   
Park, D.; Kang, D. U.; Kim, J.; and Chun, S. Y. 2020. Multitemporal recurrent neural networks for progressive non-uniform single image deblurring with incremental temporal training. In European Conference on Computer Vision, 327–343. Springer.   
Park, W.; Kim, D.; Lu, Y.; and Cho, M. 2019. Relational knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 3967–3976.   
Ren, D.; Zuo, W.; Hu, Q.; Zhu, P.; and Meng, D. 2019. Progressive image deraining networks: A better and simpler baseline. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 3937–3946.   
Sanh, V.; Debut, L.; Chaumond, J.; and Wolf, T. 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.   
Sun, L.; Dong, J.; Tang, J.; and Pan, J. 2023. Spatially-adaptive feature modulation for efficient image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13190–13199.   
Tian, Y.; Krishnan, D.; and Isola, P. 2019. Contrastive representation distillation. arXiv preprint arXiv:1910.10699.   
Timofte, R.; Agustsson, E.; Van Gool, L.; Yang, M.-H.; and Zhang, L. 2017. Ntire 2017 challenge on single image super-resolution: Methods and results. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 114–125.   
Touvron, H.; Cord, M.; Douze, M.; Massa, F.; Sablayrolles, A.; and Jégou, H. 2021. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, 10347–10357. PMLR.   
Wang, L.; Dong, X.; Wang, Y.; Ying, X.; Lin, Z.; An, W.; and Guo, Y. 2021a. Exploring sparsity in image super-resolution for efficient inference. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4917–4926.   
Wang, L.; Huang, J.; Li, Y.; Xu, K.; Yang, Z.; and Yu, D. 2021b. Improving weakly supervised visual grounding by contrastive knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 14090–14100.   
Wang, X.; Xie, L.; Dong, C.; and Shan, Y. 2021c. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, 1905–1914.

Wang, Y.; Lin, S.; Qu, Y.; Wu, H.; Zhang, Z.; Xie, Y.; and Yao, A. 2021d. Towards compact single image super-resolution via contrastive self-distillation. arXiv preprint arXiv:2105.11683.   
Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.   
Wang, Z.; Cun, X.; Bao, J.; Zhou, W.; Liu, J.; and Li, H. 2022. Uformer: A general u-shaped transformer for image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 17683–17693.   
Wu, G.; Jiang, J.; Jiang, K.; and Liu, X. 2024. Learning from history: Task-agnostic model contrastive learning for image restoration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 5976–5984.   
Xu, G.; Liu, Z.; Li, X.; and Loy, C. C. 2020. Knowledge distillation meets self-supervision. In European conference on computer vision, 588–604. Springer.   
Yang, C.; An, Z.; Zhou, H.; Zhuang, F.; Xu, Y.; and Zhang, Q. 2023. Online knowledge distillation via mutual contrastive learning for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8): 10212–10227.   
Yang, W.; Tan, R. T.; Feng, J.; Liu, J.; Guo, Z.; and Yan, S. 2017. Deep joint rain detection and removal from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1357–1366.   
Zamir, S. W.; Arora, A.; Khan, S.; Hayat, M.; Khan, F. S.; and Yang, M.-H. 2022. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5728–5739.   
Zamir, S. W.; Arora, A.; Khan, S.; Hayat, M.; Khan, F. S.; Yang, M.-H.; and Shao, L. 2021. Multi-stage progressive image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 14821–14831.   
Zhang, H.; and Patel, V. M. 2018. Density-aware single image de-raining using a multi-stream dense network. In Proceedings of the IEEE conference on computer vision and pattern recognition, 695–704.   
Zhang, H.; Sindagi, V.; and Patel, V. M. 2019. Image de-raining using a conditional generative adversarial network. IEEE transactions on circuits and systems for video technology, 30(11): 3943–3956.   
Zhang, K.; Zuo, W.; Chen, Y.; Meng, D.; and Zhang, L. 2017. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7): 3142–3155.   
Zhang, Q.; Liu, X.; Li, W.; Chen, H.; Liu, J.; Hu, J.; Xiong, Z.; Yuan, C.; and Wang, Y. 2024. Distilling Semantic Priors from SAM to Efficient Image Restoration Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 25409–25419.   
Zhang, X.; Zeng, H.; Guo, S.; and Zhang, L. 2022. Efficient long-range attention network for image super-resolution. In European conference on computer vision, 649–667. Springer.   
Zhang, Y.; Li, K.; Li, K.; Wang, L.; Zhong, B.; and Fu, Y. 2018. Image super-resolution using very deep residual channel attention networks. In Proceedings of the European conference on computer vision (ECCV), 286–301.   
Zhang, Y.; Li, W.; Li, S.; Hu, J.; Chen, H.; Wang, H.; Tu, Z.; Wang, W.; Jing, B.; and Wang, Y. 2023. Data upcycling knowledge distillation for image super-resolution. arXiv preprint arXiv:2309.14162.