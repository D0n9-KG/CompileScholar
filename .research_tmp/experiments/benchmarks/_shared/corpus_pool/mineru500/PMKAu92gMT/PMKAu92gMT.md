# SSHR: More Secure Generative Steganography with High-Quality Revealed Secret Images

Jiannian Wang $^{1}$ Yao Lu $^{1}$ Guangming Lu $^{1}$

# Abstract

Image steganography ensures secure information transmission and storage by concealing secret messages within images. Recently, the diffusion model has been incorporated into the generative image steganography task, with text prompts being employed to guide the entire process. However, existing methods are plagued by three problems: (1) the restricted control exerted by text prompts causes generated stego images resemble the secret images and seem unnatural, raising the severe detection risk; (2) inconsistent intermediate states between Denoising Diffusion Implicit Models and its inversion, coupled with limited control of text prompts degrade the revealed secret images; (3) the descriptive text of images(i.e. text prompts) are also deployed as the keys, but this incurs significant security risks for both the keys and the secret images. To tackle these drawbacks, we systematically propose the SSHR, which joints the Reference Images with the adaptive keys to govern the entire process, enhancing the naturalness and imperceptibility of stego images. Additionally, we methodically construct an Exact Reveal Process to improve the quality of the revealed secret images. Furthermore, adaptive Reference-Secret Image Related Symmetric Keys are generated to enhance the security of both the keys and the concealed secret images. Various experiments indicate that our model outperforms existing methods in terms of recovery quality and secret image security.

$^{1}$ Department of Computer Science and Technology, University of Harbin Institute of Technology (Shenzhen), Shenzhen, Guangdong, China. Correspondence to: Yao Lu <luyao2021@hit.edu.cn>, Guangming Lu <luguangm@hit.edu.cn>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

# 1. Introduction

Due to the demand of privacy, security, and data protection in the era of digital communication and AI development, steganography has emerged as an essential technology. Steganography embeds diverse types of secret messages into a container medium in an undetectable manner, which has broad applications across fields (Vyas et al., 2023; Wouters, 2024; Feng et al., 2024; Liu & Bu, 2024).

Currently, image steganography techniques are primarily divided into cover-based and generative steganography methods. Cover-based methods, such as LSBM (Mielikainen, 2006), HUGO (Pevnÿ et al., 2010), UNIWARD (Sameer & Naskar, 2018), and deep learning based methods (Baluja, 2017; 2019; Lu et al., 2021; Jing et al., 2021; Guan et al., 2022), embed secret messages within the cover images. Compared to cover-based methods, generative steganography methods generating the suitable stego images directly from the secret images via neural network without using the cover images, such as CycleGAN (Zhu et al., 2017) and encoder-decoder models (Zhou et al., 2015). Recently, the diffusion model(Ho et al., 2020) has been incorporated into the generative image steganography task, such as CRoSS (Yu et al., 2024) and DiffStega (Yang et al., 2024), with text prompts, which are the descriptive text of images, being employed to guide the entire conceal and reveal processes.

However, existing diffusion model-based generative methods still face several challenges, as illustrated in Figure 1. Firstly, constrained by the restricted control over specific semantic regions via text prompts, the generated stego images closely resemble the secret images, particularly in the background. Occasionally, the generated stego images seem weird and unnatural, raising the severe detection risk from third parties. Additionally, owing to the inconsistency of intermediate states between Denoising Diffusion Implicit Models (DDIM) (Song et al., 2021) and its inversion, coupled with the restricted control exerted by text prompts (Hertz et al., 2023; Zhang et al., 2024; Wang et al., 2024), the quality of the revealed secret images has been significantly declined, posing substantial challenges for practical applications. Finally, text prompts, also utilized as cryptographic keys to enhance steganography security, are vulnerable to being guessed due to their essence of descriptive text

![](images/1d17882ee7cb7cac1506ab1ce6cffd2ad679a38c59a3e2877369a2578d4299d9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a)CROSS
        A["Sender"] --> B["Conceal Process"]
        B --> C["Stego Image"]
        C --> D["Reveal Process"]
        D --> E["Stego Image"]
        E --> F["Reveal Process"]
        F --> G["Stego Image"]
        G --> H["Reveal Process"]
        H --> I["Stego Image"]
        I --> J["Reveal Process"]
        J --> K["Stego Image"]
        K --> L["Reveal Process"]
        L --> M["Stego Image"]
        M --> N["Reveal Process"]
        N --> O["Stego Image"]
        O --> P["Reveal Process"]
        P --> Q["Stego Image"]
        Q --> R["Reveal Process"]
        R --> S["Stego Image"]
        S --> T["Reveal Process"]
        T --> U["Stego Image"]
        U --> V["Reveal Process"]
        V --> W["Stego Image"]
        W --> X["Reveal Process"]
        X --> Y["Stego Image"]
        Y --> Z["Reveal Process"]
        Z --> AA["Stego Image"]
        AA --> AB["Reveal Process"]
        AB --> AC["Stego Image"]
        AC --> AD["Reveal Process"]
        AD --> AE["Stego Image"]
        AE --> AF["Reveal Process"]
        AF --> AG["Stego Image"]
        AG --> AH["Reveal Process"]
        AH --> AI["Stego Image"]
        AI --> AJ["Reveal Process"]
        AJ --> AK["Stego Image"]
        AK --> AL["Reveal Process"]
        AL --> AM["Stego Image"]
        AM --> AN["Reveal Process"]
        AN --> AO["Stego Image"]
        AO --> AP["Reveal Process"]
        AP --> AQ["Stego Image"]
        AQ --> AR["Reveal Process"]
        AR --> AS["Stego Image"]
        AS --> AT["Reveal Process"]
        AT --> AU["Stego Image"]
        AU --> AV["Reveal Process"]
        AV --> AW["Stego Image"]
        AW --> AX["Reveal Process"]
        AX --> AY["Stego Image"]
        AY --> AZ["Reveal Process"]
        AZ --> BA["Stego Image"]
        BA --> BB["Reveal Process"]
        BB --> BC["Stego Image"]
        BC --> BD["Reveal Process"]
        BD --> BE["Stego Image"]
        BE --> BF["Reveal Process"]
        BF --> BG["Stego Image"]
        BG --> BH["Reveal Process"]
        BH --> BI["Stego Image"]
        BI --> BJ["Reveal Process"]
        BJ --> BK["Stego Image"]
        BK --> BL["Reveal Process"]
        BL --> BM["Stego Image"]
        BM --> BN["Reveal Process"]
        BN --> BO["Stego Image"]
        BO --> BP["Reveal Process"]
        BP --> BQ["Stego Image"]
        BQ --> BR["Reveal Process"]
        BR --> BS["Stego Image"]
        BS --> BT["Reveal Process"]
        BT --> BU["Stego Image"]
        BU --> BV["Reveal Process"]
        BV --> BW["Stego Image"]
        BW --> BX["Reveal Process"]
        BX --> BY["Stego Image"]
        BY --> BZ["Reveal Process"]
        BZ --> CA["Stego Image"]
        CA --> CB["Reveal Process"]
        CB --> CC["Stego Image"]
        CC --> CD["Reveal Process"]
        CD --> CE["Stego Image"]
        CE --> CF["Reveal Process"]
        CF --> CG["Stego Image"]
        CG --> CH["Reveal Process"]
        CH --> CI["Stego Image"]
        CI --> CJ["Reveal Process"]
        CJ --> CK["Stego Image"]
        CK --> CL["Reveal Process"]
        CL --> CM["Stego Image"]
        CM --> CN["Reveal Process"]
        CN --> CO["Stego Image"]
        CO --> CP["Reveal Process"]
        CP --> CQ["Stego Image"]
        CQ --> CR["Reveal Process"]
        CR --> CS["Stego Image"]
        CS --> CT["Reveal Process"]
        CT --> CU["Stego Image"]
        CU --> CV["Reveal Process"]
        CV --> CW["Stego Image"]
        CW --> CX["Reveal Process"]
        CX --> CY["Stego Image"]
        CY --> CZ["Reveal Process"]
        CZ --> DA["Stego Image"]
        DA --> DB["Reveal Process"]
        DB --> DC["Stego Image"]
        DC --> DD["Reveal Process"]
        DD --> DE["Csym: Symmetric Key\nk_wwg: Gaussian noise/Constant Public key\nk_wwg: Normal image! Unable to decrypt"]

    subgraph (b)DiffStega
        E1((Sender))
        E2((Secret Image))
    end

    subgraph (c)Ours
        E3((Sender))
    end

    style (a)CROSS fill:#f9f,stroke:#333
    style (b)DiffStega fill:#ccf,stroke:#333
    style (c)Ours fill:#cfc,stroke:#333
```
</details>

Figure 1. The pipeline comparison of (a) CRoSS, (b) DiffStega, and (c) our method, with main differences highlighted in red dotted boxes. Our approach enables significant modifications to the secret image while enhancing the quality of revealed images. Additionally, We use a Reference-Secret Image Related Symmetric Key ( $k_{sym}$ ) as the cryptographic key, rather than text prompt used in previous methods.

for images, creating serious security risks for both the keys and concealed secret images.

To address these problems, this paper systematically proposes a secure generative steganography method, SSHR, based on the diffusion model. Given the inadequate control exerted by text prompts in diffusion-based generative steganography models, this paper substitutes text prompts with reference images as guiding conditions. The abundant semantic information inherent in reference images facilitates superior control of stego images generation. Distinct from cover-based methods, our model treats these reference images as guides for stego images generation, rather than as simple containers for concealed secret images. An exact reveal process is performed to minimize damage to the revealed secret images. Furthermore, the adaptive keys, which will be joined with the reference images to guide the entire process, are generated to compensate for the absence of keys after removing text prompts. Experimental results indicate that our method achieves notable improvements in effectiveness and security over existing models. Our main contributions are:

- We systematically propose a novel generative steganography method joints the Reference Images with the adaptive keys to govern the entire process, enhancing the naturalness and imperceptibility of stego images.   
- We methodically construct an Exact Reveal Process to precisely reverse the conceal process, diminishing the errors in the reveal process and enhancing the quality of the revealed secret images.   
- We propose a Reference-Secret Image Related Symmetric Key (RSRK) generation module to generate the keys, enhancing the security of both the keys and the concealed secret images.   
- We conduct extensive experiments to demonstrate that our model surpasses existing methods in terms of stego images quality and security, as well as the quality of the revealed secret images.

# 2. Related work

# 2.1. Cover-based Image Steganography

Traditional cover-based image steganography conceals secret data within a cover image, either in the spatial domain using techniques like Least Significant Bits (Mielikainen, 2006) or in the transform domain, such as Discrete Fourier Transform. Baluja (Baluja, 2017; 2019) initially proposed a deep learning method to conceal a full-size image within another. Universal Deep Hiding (UDH) (Zhang et al., 2020) introduced a universal pipeline for image steganography that contrasts with the pipeline in (Baluja, 2017; 2019). ISN (Lu et al., 2021) pioneered the use of Invertible Neural Networks (INN), setting a benchmark by utilizing INN's reversible properties for conceal and reveal, showcasing superior capabilities. Following this, models such as HiNet (Jing et al., 2021), DeepMIH (Guan et al., 2022), and other INN-based approaches have been proposed to further explore INN's potential in image steganography.

The reliance on cover images limits the applicability of these methods, and their stability and detection resistance need enhancement. In contrast, our model utilizes a reference image to guide stego image generation rather than serving as a container for the secret image.

# 2.2. Generative Steganography

Generative steganography eliminates the reliance on cover images by directly generating stego images from the secret images using neural networks, such as CycleGAN (Zhu et al., 2017) and encoder-decoder models (Zhou et al., 2015). Wei et al. (Wei et al., 2022) established a bi-directional mapping between stego images and secret data, integrating the generator and extractor within a single Flow-based network. Distinguished from these methods, Generative Steganography Diffusion (GSD) (Wei et al., 2023) leverages the advanced capabilities of diffusion models. Additionally, CRoSS (Yu et al., 2024) and DiffStega (Yang et al., 2024), both diffusion-based models with text prompts being employed to guide the entire process, demonstrate excellent controllability and resilience against attacks.

![](images/236200c69ad571c3faa100b24a8e3c36032f06fcffb774b1289f45d81510cf7f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Secret Image x_sec"] --> B["DWT"]
    B --> C["z_sec"]
    C --> D["Conceal Process"]
    D --> E["Z_stego"]
    E --> F["IWT"]
    F --> G["Stego Image x_stego"]
    G --> H["Sender"]
    H --> I["Attacker"]
    I --> J["Reveal With Wrong Key k_wrg"]
    J --> K["Reveal With Symmetric Key k_sym"]
    K --> L["Hiding With Symmetric Key k_sym"]
    L --> M["Revealed Image x_rev"]
    M --> N["IWT"]
    N --> O["z_rev"]
    O --> P["Reveal Process"]
    P --> Q["z'_stego"]
    Q --> R["DWT"]
    R --> S["Stego Image x_stego"]
    S --> T["Receiver"]
    T --> U["Social Media Internet Transmission"]
    U --> V["Condition Information Guidance Module"]
    V --> W["Output"]
    W --> X["1x1 Conv"]
    W --> Y["5x5 Rep-Conv"]
    W --> Z["1x1 Conv"]
    W --> AA["Conv(SiLU(•))"]
    W --> AB["SiLU"]
    W --> AC["SiLU"]
    W --> AD["θ₁"]
    W --> AE["θ₂"]
    W --> AF["θ₃"]
    W --> AG["θ₁'"]
    H --> AH["Reveal With Wrong Key k_wrg"]
    AH --> AI["k_wrg"]
    AI --> AJ["Reveal Process"]
    AJ --> AK["Z_stego"]
    AK --> AL["IWT"]
    AL --> AM["z_rev"]
    AM --> AN["Reveal Process"]
    AN --> AO["z'_stego"]
    AO --> AP["DWT"]
    AP --> AQ["Stego Image x_stego"]
    AQ --> AR["Receiver"]
    AR --> AS["Hiding With Symmetric Key k_sym"]
    AS --> AT["Hiding With Symmetric Key k_sym"]
```
</details>

Figure 2. The overall structure of our SSHR model. In the conceal stage, the secret image $x_{sec}$ and reference image $x_{ref}$ jointly generate the symmetric key $k_{sym}$ first. Guided by $k_{sym}$ and $x_{ref}$ , the secret image will be gradually encrypted and ultimately generate the stego image $x_{stego}$ . In the reveal stage, both $k_{sym}$ and $x_{ref}$ are fed into the model simultaneously to accurately reveal the secret image $x_{sec}$ .

Nevertheless, the limited control over text prompts, coupled with inconsistency between DDIM and its inversion, significantly damage the quality of stego and revealed secret images. This drives us to replace the text prompts with images and construct the exact reveal process. Furthermore, adaptive keys compensate for the absence of keys after removing text prompts.

# 2.3. Perona-Malik model

The Perona-Malik (PM) model (Perona & Malik, 1990) which has demonstrated strong performance in various image processing tasks, is a nonlinear diffusion model and formulated by the following partial differential equation:

$$
\left\{ \begin{array}{l} \frac {\partial z}{\partial t} = d i v (g (| \nabla z |) \nabla z), \\ z | _ {t = 0} = f, \end{array} \right. \tag {1}
$$

where $\nabla$ represents the gradient operator, t denotes the time, f represents the initial image and g is the diffusion function (Weickert et al., 1998). The discrete version of the PM model is derived through an explicit finite difference scheme and represented as:

$$
\begin{array}{l} \frac {z _ {t + 1} - z _ {t}}{\Delta t} = - \sum_ {i \in x, y} \nabla_ {i} ^ {T} \Lambda (z _ {t}) \nabla_ {i} z _ {t} \\ = - \sum_ {i \in x, y} \nabla_ {i} ^ {T} \phi (\nabla_ {i} z _ {t}), \\ \end{array}
$$

where $\phi$ is the influence function (Black et al., 1997) or flux function (Weickert et al., 1998). As demonstrated in previous works (Scherzer & Weickert, 2000; Zhu & Mumford, 1997), the diffusion step Equation (2) aligns with a gradient descent step to minimize the following energy function:

$$
\mathcal {P} (z) = \sum_ {i \in x, y} \sum_ {p = 1} ^ {N} \rho ((k _ {i} * z) _ {p}), \tag {3}
$$

where the functions $\rho$ is the penalty function, $k_{i}$ denotes a two-dimensional convolution filter kernel and $*$ signifies the convolution operation. It is noteworthy that the matrix-vector product $\nabla_x(z)$ can be interpreted as a 2D convolution of $z$ with the linear filter $k_{x} = [-1,1]^{T}$ . Similarly, $\nabla_y \in \mathcal{R}^{N \times N}$ corresponds to the linear filter $k_{y} = [-1,1]^{T}$ .

This paper systematically presents a novel and secure generative steganography model. Motivated by the limited control of text prompts, this paper introduces a secondary image with a new identity, replacing the text prompt for a more intuitive and effective approach. The exact reveal process is conducted to mitigate the inconsistency between the conceal and reveal process in existing methods. Moreover, the absence of keys after removing text prompts will be compensated with the adaptively generated symmetric keys. These keys will be integrated with reference images, serving as guiding elements throughout the process and enhancing security beyond conventional text prompts.

# 3. Proposed Methods

# 3.1. Framework

This paper proposes SSHR, a novel and secure generative steganography model, with the overall framework presented

in Figure 2. Building on the exceptional performance of prior work (Jing et al., 2021), our steganography process is conducted in the frequency domain. Specifically, the secret image $x_{sec}$ is first accepted as input and transformed into the latent space as $z_{sec}$ using discrete wavelet transform (DWT). Subsequently, the $z_{sec}$ is encrypted with the symmetric key $k_{sym}$ and the reference image $x_{ref}$ , which is preprocessed with the condition information guidance module (CIGM). This step generates the latent representation $z_{stego}$ of the stego image $x_{stego}$ . The final stego image $x_{stego}$ is obtained through inverse wavelet transform (IWT). The reveal process follows the exact inverse of the conceal process, ultimately yielding the revealed secret image $x_{rev}$ .

Conceal process. To meet the specific requirements of our steganography task, we introduce a condition term $\mathcal{R}(z, k_{sym}, c)$ into the original PM diffusion model. Referred to as the condition reaction term following the terminology in (Chen & Pock, 2016), this term serves to guide the generation process. Consequently, Equation (3) is modified as follows:

$$
\mathcal {F} = \sum_ {i = 1} ^ {N _ {k}} \rho_ {i} (k _ {i} * z, k _ {s y m}) + \lambda \mathcal {R} (z, k _ {s y m}, c), \tag {4}
$$

where $k_{sym}$ denotes the Reference-Secret Image Related Symmetric Key, which is utilized to strengthen the security of both the key and the concealed secret images, with additional details provided later. $N_{k}$ represents the amount of the filters. c is the condition infused into the condition reaction term to regulate the generated stego images and enhance their naturalness and imperceptibility. In our model, c is set to the reference images by default. Equation (4) results in a Nonlinear Diffusion Model incorporating the condition reaction term, depicted as a forward-backward step at $z_{t-1}$ of the energy functional, expressed as:

$$
E ^ {t} (x) = \sum_ {i = 1} ^ {N _ {k}} \mathcal {P} _ {i} ^ {t} (z, k _ {s y m}) + \mathcal {R} ^ {t} (z, k _ {s y m}, c), \tag {5}
$$

where $\mathcal{P}_{i}^{t}(z,k_{sym})=\sum_{j=1}^{N_{k}}\rho_{i}^{t}((K_{i}^{t}z)_{j},k_{sym})$ . By applying an explicit finite difference scheme, we derive the following discretized iteration law, with the initial input set as $z_{0}=z_{sec}$ :

$$
\begin{array}{l} z _ {t + 1} = z _ {t} - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * z _ {t}, k _ {s y m}) \tag {6} \\ - \lambda \varphi_ {t} (z _ {t}, k _ {s y m}, c). \\ \end{array}
$$

To perform the convolution $k_{i}^{t} * z_{t}$ , we substitute traditional convolution with Re-parameterization Depthwise Convolution (Rep-DWConv) (Tu et al., 2024), which offers enhanced performance and parameter efficiency. The weights of Rep-DWConv, given by $W_{re-c} = \text{Linear}(SiLU(\text{Linear}(k_{sym})))$ , are dynamically generated using the symmetric key $k_{sym}$ to ensure security. Consequently, this module is referred to as Conditional Reparameterization Convolution (CRC). In line with (Chen & Pock, 2016), we use 63 Gaussian radial basis functions(GRBFs) to parameterize $\phi_{i}^{t}$ for $i \in \{1, 2, ..., N_{k}\}$ . The overall pipeline for the Diffusion Term (DT) and the Condition Reaction Term (CRT) is illustrated in Figure 3.

![](images/cb5008c0d3d04cd92b891ebbc86240a6c12c98cf51a3e71f2ed43dc8ea8d7f1d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input"] --> B["CRC"]
    B --> C["GRBF"]
    C --> D["CRC"]
    D --> E["Output"]
    F["Linear"] --> G["w_t"]
    G --> C
    H["GRBF"] --> C
    I["Diffusion Term"] --> C
    style C fill:#f9f,stroke:#333
```
</details>

![](images/c891e9ed532fc212e0b2c61c1554f83efac904951baa1db1490fa69ab9b26d20.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Scale SH0"]
    B --> C["Attention"]
    C --> D["FFN"]
    D --> E["Scale"]
    E --> F["Output"]
    G["Cyanom"] --> H["Conv"]
    H --> I["α'p'"]
    I --> J["Condition Reaction Term"]
    J --> K["λ"]
    K --> L["Output"]
```
</details>

Figure 3. Illustration of (a) the Diffusion Term (DT) and (b) the Condition Reaction Term (CRT).

Inspired by the coupling function and affine coupling layer introduced in (Dinh et al., 2014; 2016), we define an auxiliary variable $y_{t}$ that satisfies $y_{t} = z_{t}$ . Consequently, the iteration step in Equation (6) is transformed into:

$$
\left\{ \begin{array}{l l} y _ {t} = z _ {t}, \\ z _ {t + 1} = z _ {t} & - \sum_ {i = 1} ^ {N _ {k}} \bar {k _ {i} ^ {t}} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * y _ {t}, k _ {s y m}) \\ & - \lambda \varphi_ {t} (y _ {t}, k _ {s y m}, c). \end{array} \right. \tag {7}
$$

Therefore, $y_{t+1}$ is calculated as:

$$
\begin{array}{l} y _ {t + 1} = z _ {t + 1} \\ = z _ {t} - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * y _ {t}, k _ {s y m}) \\ - \lambda \varphi_ {t} (y _ {t}, k _ {s y m}, c) \tag {8} \\ = y _ {t} - \sum_ {i = 1} ^ {N _ {k}} \bar {k _ {i} ^ {t}} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * z _ {t}, k _ {s y m}) \\ - \lambda \varphi_ {t} (z _ {t}, k _ {s y m}, c). \\ \end{array}
$$

By combining the equation above with the second equation in Equation (7), we obtain:

$$
\left\{ \begin{array}{l l} z _ {t + 1} = z _ {t} & - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * y _ {t}, k _ {s y m}) \\ & - \lambda \varphi_ {t} (y _ {t}, k _ {s y m}, c), \\ y _ {t + 1} = y _ {t} & - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * z _ {t}, k _ {s y m}) \\ & - \lambda \varphi_ {t} (z _ {t}, k _ {s y m}, c). \end{array} \right. \tag {9}
$$

To accelerate model convergence and drawing inspiration from the Gauss-Seidel method, we use $z_{t+1}$ instead of $z_{t}$ as

the input of the second equation, leading to:

$$
\left\{ \begin{array}{l l} z _ {t + 1} = z _ {t} & - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * y _ {t}, k _ {s y m}) \\ & - \lambda \varphi_ {t} (y _ {t}, k _ {s y m}, c), \\ y _ {t + 1} = y _ {t} & - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * z _ {t + 1}, k _ {s y m}) \\ & - \lambda \varphi_ {t} (z _ {t + 1}, k _ {s y m}, c). \end{array} \right. \tag {10}
$$

Following the recommendation in (Wallace et al., 2023), we introduce a learnable parameter $p \in [0.90, 1.0]$ to enhance robustness. Additionally, intermediate mixing layers are applied after each gradient descent step to compute weighted averages of z and y, defined as:

$$
z ^ {m i x} = p \cdot z + (1 - p) \cdot y. \tag {11}
$$

In summary, the conceal process of the proposed SSHR, with the input $z_{0} = z_{sec}$ and final output $z_{T} = z_{stego}$ , can be formulated as:

$$
\left\{ \begin{array}{l} z _ {t + 1} ^ {m i x} = z _ {t} - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * y _ {t}, k _ {s y m}) \\ \quad - \lambda \varphi_ {t} (y _ {t}, k _ {s y m}, c), \\ y _ {t + 1} ^ {m i x} = y _ {t} - \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * z _ {t + 1} ^ {m i x}, k _ {s y m}) \\ \quad - \lambda \varphi_ {t} (z _ {t + 1} ^ {m i x}, k _ {s y m}, c), \\ z _ {t + 1} = p \cdot z _ {t + 1} ^ {m i x} + (1 - p) \cdot y _ {t + 1} ^ {m i x}, \\ y _ {t + 1} = p \cdot y _ {t + 1} ^ {m i x} + (1 - p) \cdot z _ {t + 1}. \end{array} \right. \tag {12}
$$

Exact reveal process. To minimize errors in the reveal process and enhance the quality of the revealed secret images, we deterministically reverse the conceal process, ensuring a reliable reveal process in the proposed model. Equation (11) represents an invertible affine transformation, with the inversion process defined as follows:

$$
z = (z ^ {m i x} - (1 - p) \cdot y) / p. \tag {13}
$$

Thus, reveal Equation (12) and the exact reveal process of our model is calculated, with input $z_{T} = z_{stego}$ and output $z_{0} = z_{rev}$ , as follows:

$$
\left\{ \begin{array}{l} y _ {t + 1} ^ {m i x} = (y _ {t + 1} - (1 - p) \cdot z _ {t + 1}) / p, \\ z _ {t + 1} ^ {m i x} = (z _ {t + 1} - (1 - p) \cdot y _ {t + 1} ^ {m i x}) / p, \\ y _ {t} = y _ {t + 1} ^ {m i x} + \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * z _ {t + 1} ^ {m i x}, k _ {s y m}) \\ \quad + \lambda \varphi_ {t} (z _ {t + 1} ^ {m i x}, k _ {s y m}, c), \\ z _ {t} = z _ {t + 1} ^ {m i x} + \sum_ {i = 1} ^ {N _ {k}} \bar {k} _ {i} ^ {t} * \phi_ {i} ^ {t} (k _ {i} ^ {t} * y _ {t}, k _ {s y m}) \\ \quad + \lambda \varphi_ {t} (y _ {t}, k _ {s y m}, c). \end{array} \right. \tag {14}
$$

In practice, the calculation steps of z and y in both the conceal and reveal process are executed alternately, ensuring a symmetrical transformation between the two sequences.

# 3.2. Reference-Secret Image Related Key

Ensuring the security of the secret image hidden within the stego images is of paramount in image steganography. The concept of the key, originally from cryptography and first introduced to steganography in (Mou et al., 2023), is well-suited to address this requirement. However, existing methods do not fully guarantee key security and vulnerable to key guessing. In (Mou et al., 2023), the key is merely a single numeric value; in CRoSS (Yu et al., 2024), the private and public keys correspond to the descriptive text of the secret and stego images, respectively. Similarly, DiffStega (Yang et al., 2024) employs a pre-determined password-related reference image, alongside a text prompt as the public key. In contrast, and drawing inspiration from the Elliptic Curve Diffie-Hellman Ephemeral (ECDHE) key exchange algorithm, our model introduces a dynamically generated symmetric key that is adaptively derived from the reference-secret image pair, significantly boosting security.

Inspired by the principle of ECDHE, we treat each steganography task as an independent encryption session, where the symmetric key is adaptively generated based on the secret and reference images. The process begins by extracting features $F_{s}$ and $F_{r}$ from the secret and reference image, respectively, using a pre-trained neural network, with AlexNet employed as the default model. Next, a MLP is applied to generate the private key, defined as $k_{pri-i} = MLP(F_{i}), i \in r, s$ , where $k_{pri-r}$ and $k_{pri-s}$ represent the private keys of the reference and secret images, respectively. The public key is then derived using a specific parameter matrix W, formulated as $k_{pub-i} = W \cdot k_{pri-i}$ , where $\cdot$ represents the Hadamard product.

Following the principles of the ECDHE algorithm, the symmetric key $k_{sym}$ utilized in our model is derived as:

$$
\begin{array}{l} k _ {s y m} = \left(W _ {L} + W _ {S}\right) \cdot k _ {p r i - s} \cdot k _ {p r i - r} \tag {15} \\ = (W _ {L} + W _ {S}) \cdot k _ {p r i - r} \cdot k _ {p r i - s}, \\ \end{array}
$$

where $W_{S}$ denotes the weight matrix generated based on the secret image, and the overall framework is illustrated in Figure 2. Based on the above equation, the parameter matrix W used for public key generation is defined as $W = W_{L} + W_{S}$ . Thus, for each unique pair of images, a distinct symmetric key is generated and used in both the conceal and reveal processes. During transmission, only the public key is shared, ensuring strict protection of both the private keys and the symmetric key. This approach significantly enhances the security of the secret images concealed within the stego images.

When transmitting the stego image $x_{stego}$ , the public key

$k_{pub-s}$ associated with the secret image is sent alongside it. Upon receiving the public key of the secret image, the receiver can use the same pipeline outlined in Equation (15) to generate the symmetric key $k_{sym}$ , as the sender did during the conceal process. With this symmetric key, the receiver can accurately reveal the secret image and complete the reveal process. Further details are provided in the Supplementary. To the best of our knowledge, this is the first exploration to integrate key exchange protocols into image steganography.

Subsequently, the symmetric key $k_{sym}$ is utilized to generate the weights for CRC, denoted as $W_{re-c}$ , as well as the modulation parameters $\alpha$ , $\beta$ , $\gamma$ in CRT. This design ensures that the symmetric key plays a crucial role in the conceal and reveal process.

# 3.3. Condition Information Guidance Module

We aim to progressively encrypt the secret images, allowing the stego images to gradually improve in quality and increasingly resemble the reference images over time. To achieve this, we first pre-process the reference images using the Condition Information Guidance Module (CIGM) before generating the modulation parameters. This ensures that the input to AdaLN-Zero (Peebles & Xie, 2023) varies across different timesteps, thereby enabling the stego images to incrementally converge toward the reference images. The CIGM is analogous to a transformer block and comprises a separate linear transformation(SLT) (Lu et al., 2024), followed by a Re-parameterization Feed-Forward Network(Re-FFN)(Tu et al., 2024). The SLT serves a function similar to that of the attention module in a transformer block and can be formulated as follows:

$$
\mathcal {O} _ {s l t} (c) = W (\theta_ {1}) \cdot c + \theta_ {2} \cdot c + \theta_ {3}, \tag {16}
$$

where $\theta_{1} \in R^{B \times C \times H \times W}$ , $\theta_{2} \in R^{B \times C \times 1 \times 1}$ , $\theta_{3} \in R^{B \times C \times 1 \times 1}$ and c is the input of it. The $W(\theta_{1})$ is a convolution layer, expressed as $W(\theta_{1}) = Conv(SiLU(\theta_{1}))$ . The Re-FFN is structured with two pointwise convolution layers and a Rep-Conv layer, as illustrated in Figure 2.

Subsequently, the modulation parameters $\alpha$ , $\beta$ , $\gamma$ are generated based on the output of CIGM, denoted as $\mathcal{O}_{CIPM}(c)$ , along with the parameter matrix $W_{A}$ . The parameters $W_{A}$ derived from the symmetric key $k_{sym}$ via a MLP layer. The overall pipeline is formulated as:

$$
\begin{array}{l} \alpha , \beta , \gamma = s p l i t ((\mathcal {O} _ {C I G M} (c)) \cdot W _ {A}) \\ = \operatorname{split} \left(\left(\mathcal {O} _ {C I G M} (c)\right) \cdot M L P \left(k _ {\text {sym}}\right). \right. ^ {(1 7)} \\ \end{array}
$$

Subsequently, they are incorporated into the CRT to guide the generation process and encrypt the secret image.

In summary, the output of CRT $\mathcal{O}_{CRT}(z,k_{sym},c)$ is:

$$
\mathcal {O} _ {C R T} (z, k _ {s y m}, c) = \gamma \cdot N (z \cdot (1 + \alpha) + \beta), \tag {18}
$$

where N is the Transformer block illustrated in Figure 3.

# 3.4. Loss function

Our loss function comprises frequency loss and perceptual loss, which we will describe in detail below.

Frequency loss. The steganography process intends to generate the stego image $x_{stego}$ based on the secret image $x_{sec}$ . For security purposes, the frequency content of the stego image $z_{stego}$ should closely match that of the reference image $z_{ref}$ , making them indistinguishable. In the reveal phase, it is essential that the frequency profile of the revealed image $z_{rev}$ closely aligns with that of the original secret image $z_{sec}$ . To enforce these constraints, we define the frequency loss as follows:

$$
\mathcal {L} _ {F} = l _ {s} (z _ {r e f}, z _ {s t e g o}) + l _ {s} (z _ {s e c}, z _ {r e v}), \tag {19}
$$

where $l_{s}$ represents the $l_{1}$ or $l_{2}$ norm, serving as a measure of the difference between two images. In our experiments, we use the $l_{1}$ norm as the default.

Perceptual loss. To ensure that the stego image $x_{stego}$ , generated during the conceal process, closely resembles the reference image $x_{ref}$ , and that the revealed image $x_{rev}$ visually aligns with the secret image $x_{sec}$ , we introduce a perceptual loss. This loss function quantifies the high-level perceptual differences between these image pairs. The perceptual loss is defined as follows:

$$
\mathcal {L} _ {P} = l _ {p} \left(\mathscr {A} \left(x _ {\text {ref}}\right), \mathscr {A} \left(x _ {\text {stego}}\right)\right) \tag {20}
$$

$$
+ l _ {p} (\mathscr {A} (x _ {s e c}), \mathscr {A} (x _ {r e v})),
$$

where A is the pre-trained AlexNet (Krizhevsky et al., 2012), used to extract high-level features from the images, and $l_{p}$ measures the difference between these features.

Total loss. The total loss function $L_{Total}$ is defined as the weighted sum of the frequency loss $L_{F}$ and perceptual loss $L_{P}$ , formulated as:

$$
\mathcal {L} _ {\text { Total }} = \lambda_ {1} \mathcal {L} _ {F} + \lambda_ {2} \mathcal {L} _ {P}, \tag {21}
$$

where $\lambda_{1}$ and $\lambda_{2}$ are trade-off parameters set to 2.0 and 1.0, respectively, to balance the different losses.

# 4. Experiments

# 4.1. Experimental Setting

Datasets and settings. Our model is implemented with PyTorch and trained on the DIV2K (Agustsson & Timofte, 2017) training dataset. The evaluation is performed on the DIV2K (Agustsson & Timofte, 2017) test dataset (100 images), COCO (Lin et al., 2014) (5000 images), ImageNet (Russakovsky et al., 2015) (10,000 images), and UniStega (Yang et al., 2024) (100 images) at a resolution of $256 \times 256$ . Additional details are presented in the Supplementary.

![](images/4e81490dfa81f11793739688f944049c1e305c5593a29f4151f9aded58230de7.jpg)  
Prompt 1: (CRoSS) a black and white cat with yellow eyes /(DiffStega)None Prompt 2: a painting in the style of Pixar   
Figure 4. Visual comparisons of our model with generative steganography models on the UniStega dataset across three different prompts. These prompts are utilized in CRoSS and DiffStega, whereas our model functions without text prompts. The reference image is used as the image condition in both DiffStega and our model. More comparison results are presented in the Supplementary.

Table 1. Numerical comparisons of secret/revealed image pairs with cover-based steganography methods across various datasets, highlighting the best results in red and the second-best in bold. 

<table><tr><td rowspan="2">METHOD</td><td colspan="4">DIV2K</td><td colspan="4">COCO</td><td colspan="4">IMAGENET</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>RMSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>RMSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>RMSE↓</td></tr><tr><td>HIDDEN</td><td>36.43</td><td>0.9496</td><td>6.02</td><td>5.50</td><td>37.68</td><td>0.9145</td><td>4.72</td><td>6.33</td><td>35.70</td><td>0.9301</td><td>4.57</td><td>6.92</td></tr><tr><td>BALUJA</td><td>35.88</td><td>0.9377</td><td>4.68</td><td>6.11</td><td>35.01</td><td>0.9341</td><td>6.52</td><td>8.00</td><td>34.13</td><td>0.9247</td><td>5.31</td><td>8.37</td></tr><tr><td>WENG ET.AL</td><td>33.24</td><td>0.8582</td><td>5.04</td><td>6.32</td><td>33.05</td><td>0.8921</td><td>4.80</td><td>6.06</td><td>33.34</td><td>0.8921</td><td>4.80</td><td>6.06</td></tr><tr><td>UDH</td><td>35.22</td><td>0.8036</td><td>4.03</td><td>4.78</td><td>35.07</td><td>0.8220</td><td>3.77</td><td>4.67</td><td>35.39</td><td>0.8252</td><td>3.73</td><td>4.58</td></tr><tr><td>ISN</td><td>37.06</td><td>0.9672</td><td>2.80</td><td>4.30</td><td>36.58</td><td>0.9016</td><td>3.04</td><td>3.78</td><td>37.73</td><td>0.9548</td><td>2.97</td><td>3.31</td></tr><tr><td>HiNET</td><td>46.64</td><td>0.9962</td><td>0.93</td><td>1.31</td><td>44.05</td><td>0.9952</td><td>1.17</td><td>1.70</td><td>46.78</td><td>0.9947</td><td>1.12</td><td>1.23</td></tr><tr><td>OURS</td><td> $48.56(1.92\uparrow)$ </td><td> $0.9988(0.0022\uparrow)$ </td><td> $0.74(0.19\downarrow)$ </td><td> $0.97(0.34\downarrow)$ </td><td> $47.67(3.62\uparrow)$ </td><td> $0.9985(0.0033\uparrow)$ </td><td> $0.80(0.37\downarrow)$ </td><td> $1.08(0.62\downarrow)$ </td><td> $49.52(2.74\uparrow)$ </td><td> $0.9986(0.0039\uparrow)$ </td><td> $0.68(0.44\downarrow)$ </td><td> $0.88(0.35\downarrow)$ </td></tr></table>

Table 2. Numerical comparisons with generative steganography methods on the UniStega dataset. 

<table><tr><td rowspan="2">METHOD</td><td colspan="2">STEGO IMAGES</td><td colspan="2">CORRECT KEY</td><td colspan="2">WRONG KEY(CONstant)</td></tr><tr><td>PSNR↓</td><td>SSIM↓</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↓</td><td>SSIM↓</td></tr><tr><td>CROSS</td><td>19.03</td><td>0.66</td><td>21.25</td><td>0.71</td><td>-</td><td>-</td></tr><tr><td>DIFFSTEGA</td><td>18.61</td><td>0.59</td><td>23.29</td><td>0.77</td><td>17.53</td><td>0.54</td></tr><tr><td>DIFFSTEGA $^{\ddagger}$ </td><td>19.73</td><td>0.65</td><td>23.92</td><td>0.79</td><td>20.68</td><td>0.70</td></tr><tr><td>OURS</td><td>8.96(9.65↓)</td><td>0.11(0.48↓)</td><td>47.04(23.12↑)</td><td>0.99(0.20↑)</td><td>5.70(11.83↓)</td><td>0.24(0.30↓)</td></tr></table>

Table 3. NIQE scores for various models. 

<table><tr><td></td><td>ORIGINAL IMAGE</td><td>ISN</td><td>HiNet</td><td>CROSS</td><td>DIFFSTEGA</td><td>OURS</td></tr><tr><td>NIQE↓</td><td>5.14</td><td>11.28</td><td>5.35</td><td>5.60</td><td>5.58</td><td>5.46</td></tr></table>

Benchmarks. To comprehensively evaluate our method's effectiveness, we compare it against state-of-the-art (SOTA) steganography methods, including cover-based methods: Baluja et al. (Baluja, 2017), HiDDeN (Zhu, 2018), UDH (Zhang et al., 2020), Weng et al. (Weng et al., 2019), ISN (Lu et al., 2021), HiNet (Jing et al., 2021); and generative methods: CRoSS (Yu et al., 2024), DiffStega (Yang et al., 2024). To maintain objectivity, we re-trained the cover-based models with the same training dataset as ours.

Evaluation Metrics. To assess the quality of secret/revealed pairs, we utilize Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) (Wang et al., 2004), Root Mean Square Error (RMSE), and Mean Absolute Error (MAE) as performance metrics. In addition, we use the Naturalness Image Quality Evaluator (NIQE) (Mittal et al., 2012) to evaluate the naturalness of the stego images.

# 4.2. Quality Analysis

Quantitative results. We compare the proposed method against existing models, with numerical results presented in Table 1 and Table 2. Table 1 presents a quantitative comparison between the proposed SSHR model and other cover-based models on the DIV2K, COCO, and ImageNet datasets. Notably, on the ImageNet dataset, our model demonstrates a PSNR gain of 2.74 dB and an SSIM improvement of 0.39%. Meanwhile, MAE and RMSE are reduced by 0.44 and 0.35, respectively. Similarly, on the DIV2K and COCO datasets, the proposed SSHR model outperforms other cover-based models, achieving PSNR/SSIM gains of 1.92 dB/0.22% and 3.62 dB/0.33%, respectively, along with lower MAE and RMSE. These results illustrate that SSHR significantly enhances the quality of revealed secret images compared to existing cover-based steganography methods.

Table 2 presents the quantitative results in comparison with generative methods on the UniStega dataset. Specifically,

![](images/60780b77e324c1b5a1a872c8ed7c0c1cf6a57b68fd20d33c6f95d98ac5c85b5d.jpg)  
Figure 5. Visual comparisons of stego images(top) and revealed secret images(bottom) for our model and various cover-based steganography models on the DIV2K dataset.

our model achieves an impressive 24 dB gain in PSNR and 20% improvement in SSIM for correct recovery/secret pairs, indicating a substantial enhancement. For secret/stego image pairs, the proposed model reduces PSNR by 9.65 dB and SSIM by 48% compared to other generative methods, effectively diminishing the resemblance between stego and their corresponding secret images. Additionally, lower similarity scores for incorrect recovery/secret image pairs, with a notable 11.83 dB decrease in PSNR and 30% reduction in SSIM, demonstrate the proposed model effectively bolsters security for recovery attempts with incorrect key. Evidently, the SSHR excels in encryption and decryption compared to other generative steganography models, offering high-quality revealed secret images and enhanced security.

Qualitative Results. The qualitative comparison outcomes for the stego and recovery images of our model and other models are presented in Figure 4 and Figure 5. Figure 4 illustrates comparisons between our SSHR model and other generative steganography models on the UniStega dataset. The results demonstrate that the proposed model generates stego images that are distinctly different from the secret images, while simultaneously producing higher-quality revealed secret images. Additionally, the SSHR model shows enhanced security when the image is revealed with three types of incorrect keys. Figure 5 showcases the visual performance of our model in comparison to various cover-based steganography methods. The figure highlights the superior performance of the proposed method in terms of secret/reveal image pairs, as well as the comparative results for cover/stego image pairs. These results clearly demonstrate that the proposed model achieves significant improvements in effectiveness and security over existing SOTA models.

# 4.3. Security Analysis

Naturalness and Imperceptibility. We apply the NIQE to evaluate the naturalness and visual security of the images against human suspicion without the assistance of reference images or human feedback. The results in Table 3 show a 0.12 reduction in NIQE value for SSHR model compared to other generative steganography models. This indicates that the proposed model outperforms other generative steganography models in terms of naturalness and imperceptibility.

Table 4. The detection accuracy (%) detected by SRNet and XuNet. 

<table><tr><td rowspan="2"></td><td colspan="4">COVER-BASED METHOD</td><td colspan="3">GENERATIVE METHOD</td></tr><tr><td>WENG</td><td>UDH</td><td>ISN</td><td>HiNet</td><td>CROSS</td><td>DIFFSTEGA</td><td>OURS</td></tr><tr><td>SRM</td><td>81.47</td><td>76.34</td><td>54.86</td><td>53.52</td><td>51.73</td><td>51.68</td><td> $\mathbf{50.64(1.04\downarrow)}$ </td></tr><tr><td>SRNET</td><td>85.31</td><td>79.82</td><td>55.68</td><td>55.54</td><td>52.10</td><td>52.03</td><td> $\mathbf{51.05(0.98\downarrow)}$ </td></tr><tr><td>XuNET</td><td>86.24</td><td>80.26</td><td>55.42</td><td>55.37</td><td>53.12</td><td>52.86</td><td> $\mathbf{50.93(1.93\downarrow)}$ </td></tr></table>

![](images/73dfcae1c72ecf79de2c191ae88d9604874aa1b9bccf7af449bf2ebd102825af.jpg)

<details>
<summary>line</summary>

| Model       | AUC    |
|-------------|--------|
| HiNet       | 0.80   |
| ISN         | 0.70   |
| UDH         | 0.65   |
| Weng        | 0.85   |
| CRoSS       | 0.54   |
| DiffStega   | 0.52   |
| Ours        | 0.51   |
</details>

Figure 6. Security performance detected by StegExpose.

Steganographic analysis. The anti-steganalysis ability is crucial for evaluating the security of image steganography, as it measures the likelihood that stego images can be distinguished from reference images using steganalysis tools. Following the common practices in the cover-based and generative steganography task (e.g., ISN (Lu et al., 2021), HiNet (Jing et al., 2021), CRoSS (Yu et al., 2024), DiffStega (Yang et al., 2024)), we employ the open-source steganalysis tool StegExpose (Boehm, 2014), hand-crafted feature-based steganalyzer SRM (Fridrich & Kodovsky, 2012) and two steganalysis networks: SRNet (Boroumand et al., 2018) and XuNet (Xu et al., 2016), to systematically evaluate the anti-steganalysis capabilities of the proposed model alongside other methods. Lower detection accuracy and a smaller area under curve (AUC) indicates better security performance. The evaluate results are presented in Figure 6 and Table 4 respectively. These steganalysis results indicate that the proposed SSHR model outperforms other SOTA methods in terms of anti-steganalysis performance.

Effect of the Symmetric Key. The RSRK is designed to secure both the key and the concealed secret image. We tested the model with the correct symmetric key, alongside three types of incorrect keys: random Gaussian noise, a constant value, and the public key of the secret image (which could be intercepted by attackers during transmission). As shown in Figure 1 and Figure 4, only the correct key enables accurate recovery of the secret image. Figure 1 uses a constant key as the incorrect key, and Table 2 provides numerical results for revealing the secret image with a constant key. Both numerical and visual results consistently show that an

Table 5. Effectiveness of PM, Wavelet transform, Rep-Conv and CIGM. 

<table><tr><td rowspan="2">PM</td><td rowspan="2">WAVELET TRANSFORM</td><td rowspan="2">REP-CONV</td><td rowspan="2">CIGM</td><td colspan="4">REFERENCE/STEGO</td><td colspan="4">SRCRET/RECOVERY</td></tr><tr><td>PSNR</td><td>SSIM</td><td>MAE</td><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>RMSE</td></tr><tr><td>X</td><td>X</td><td>X</td><td>X</td><td>30.25</td><td>0.7862</td><td>8.57</td><td>9.23</td><td>37.77</td><td>0.9489</td><td>3.58</td><td>4.23</td></tr><tr><td>√</td><td>X</td><td>X</td><td>X</td><td>35.37</td><td>0.8050</td><td>7.77</td><td>7.90</td><td>41.36</td><td>0.9624</td><td>2.75</td><td>2.88</td></tr><tr><td>√</td><td>√</td><td>X</td><td>X</td><td>37.96</td><td>0.9318</td><td>2.99</td><td>3.57</td><td>42.24</td><td>0.9969</td><td>1.69</td><td>2.18</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>39.92</td><td>0.9670</td><td>2.16</td><td>2.91</td><td>45.19</td><td>0.9758</td><td>0.98</td><td>1.45</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>41.32</td><td>0.9897</td><td>1.13</td><td>1.25</td><td>48.56</td><td>0.9988</td><td>0.74</td><td>0.97</td></tr></table>

Table 6. Quantitative results of revealed images on the UniStega dataset when stego images undergo various distortion. 

<table><tr><td>METHOD</td><td>CLEAN</td><td>GAUSSIAN BLUR</td><td>GAUSSIAN NOISE</td><td>JPEG COMPRESSION</td><td>POISSON</td></tr><tr><td>ISN</td><td>21.73</td><td>5.16</td><td>3.97</td><td>4.33</td><td>4.05</td></tr><tr><td>HiNET</td><td>46.55</td><td>9.32</td><td>9.97</td><td>10.13</td><td>10.29</td></tr><tr><td>CROSS</td><td>21.25</td><td>20.08</td><td>18.97</td><td>20.20</td><td>19.27</td></tr><tr><td>DIFFSTEGA</td><td>23.29</td><td>20.85</td><td>18.62</td><td>21.16</td><td>20.15</td></tr><tr><td>OURS</td><td>47.04</td><td>24.01(3.16↑)</td><td>23.95(4.98↑)</td><td>24.74(3.58↑)</td><td>23.96(3.81↑)</td></tr></table>

attacker cannot retrieve the secret image with an incorrect key. These findings underscore the essential role of the proposed RSRK and key generation module in enhancing the security of both the key and the concealed secret image.

Effectiveness of various setting. As indicated in Table 5, the various settings we employed significantly improve the quantitative metrics for both stego and revealed images, enhancing performance in both the conceal and reveal processes. For example, the introduced PM model leads to a PSNR gain of 5.12 dB for reference/stego pairs and 3.59 dB for secret/recovery pairs. Figure 7 illustrates the encryption and decryption process, showcasing how the SSHR model achieves secure, stepwise encryption and decryption of secret images with reference image guidance. These results demonstrate that the proposed model effectively reduces the similarity between the stego and secret images, while producing high-quality revealed secret images, validating the module's effectiveness and overall model performance.

Performances on Various Distortions. To assess the robustness of our SSHR model, we test it under various degradations, including Gaussian blur, Gaussian noise, JPEG compression and Poisson noise. The results, presented in Table 6 in terms of PSNR, indicate that the proposed SSHR model outperforms other SOTA models. These results demonstrate the favorable robustness of the proposed SSHR against attacks. Additional details on the distortions are provided in the Supplementary.

# 5. Conclusion

This paper presents SSHR, an innovative steganography method based on the PM diffusion model, designed to overcome the limitations of existing generative steganography approaches. The proposed model uses reference images to guide stego image generation, ensuring outputs are visually natural yet highly dissimilar to the secret images, thereby enhancing concealment effectiveness. We also establish an exact reveal process to improve the quality of the revealed secret images. Additionally, we propose an RSRK generation module that strengthens key security by dynamically linking them to both the reference and secret images, significantly enhancing the overall security of the concealed secret images. Extensive experiments demonstrate that SSHR outperforms SOTA generative steganography methods in both effectiveness and security.

![](images/6a729601535af7b5f7f450ff38339ef1b7839e03fe5ad135230a8a542f726b1a.jpg)

<details>
<summary>natural_image</summary>

Sequence of images showing a dark, textured surface with white and yellow spheres, labeled 'Secret' and 'Stego' (no text or symbols in the visual content)
</details>

Figure 7. The CIGM facilitates the achievement of progressive encryption.

# Acknowledgements

This work was supported in part by the NSFC fund (NO. 62206073, 62176077), in part by the Shenzhen Key Technical Project (NO. JCYJ20241202123728037, NO. JSGG20220831092805009, JSGG20220831105603006, JSGG20201103153802006, KJZD20230923115117033, KJZD20240903100712017), in part by the Guangdong International Science and Technology Cooperation Project (NO. 2023A0505050108), in part by the Shenzhen Fundamental Research Fund (NO. JCYJ20210324132210025), and in part by the Guangdong Provincial Key Laboratory of Novel Security Intelligence Technologies (NO. 2022B1212010005), in part by the Guangdong Shenzhen joint Youth Fund under Grant 2021A151511074, and in part by the Natural Science Foundation of Shenzhen General Project under Grant JCYJ20240813110007010, in part by the Natural Science Foundation of Guangdong Province under Grant 2023A1515010893, in part by the Shenzhen Doctoral Initiation Technology Plan under Grant RCBS20221008093222010, in part by the Shenzhen Pengcheng Peacock Startup Fund.

# Impact Statement

This paper delves into an exploration of generative image steganography technology, with the objective of generating the stego images from secret images. Through a succession of innovative algorithmic designs and meticulous experimental analyses, it has significantly boosted the security and effectiveness of generative image steganography methods. The integration of the key-exchange algorithm has effectively forged an initial connection between image steganography and cryptology. This linkage not only enriches the theoretical framework of image steganography

but also endows cryptology with new application scenarios.

# References

Agustsson, E. and Timofte, R. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 126–135, 2017.   
Baluja, S. Hiding images in plain sight: Deep steganography. Advances in Neural Information Processing Systems, 30, 2017.   
Baluja, S. Hiding images within images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(7):1685–1697, 2019.   
Black, M., Sapiro, G., Marimont, D., and Heeger, D. Robust anisotropic diffusion and sharpening of scalar and vector images. In Proceedings of International Conference on Image Processing, volume 1, pp. 263–266. IEEE, 1997.   
Boehm, B. Stegexpose-a tool for detecting lsb steganography. arXiv preprint arXiv:1410.6656, 2014.   
Boroumand, M., Chen, M., and Fridrich, J. Deep residual network for steganalysis of digital images. IEEE Transactions on Information Forensics and Security, 14(5):1181–1193, 2018.   
Chen, Y. and Pock, T. Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(6):1256–1272, 2016.   
Dinh, L., Krueger, D., and Bengio, Y. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.   
Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.   
Feng, W., Zhou, W., He, J., Zhang, J., Wei, T., Li, G., Zhang, T., Zhang, W., and Yu, N. AqualoRA: Toward white-box protection for customized stable diffusion models via watermark IoRA. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=8xKGZsnV2a.   
Fridrich, J. and Kodovsky, J. Rich models for steganalysis of digital images. IEEE Transactions on information Forensics and Security, 7(3):868–882, 2012.   
Guan, Z., Jing, J., Deng, X., Xu, M., Jiang, L., Zhang, Z., and Li, Y. Deepmih: Deep invertible network for multiple image hiding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(1):372–390, 2022.

Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., and Cohen-or, D. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=\_CDixzkzeyb.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jing, J., Deng, X., Xu, M., Wang, J., and Guan, Z. Hinet: Deep image hiding by invertible network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4733–4742, 2021.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 2012.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.

Liu, Y. and Bu, Y. Adaptive text watermark for large language models. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=7emOSb5UfX.

Lu, S.-P., Wang, R., Zhong, T., and Rosin, P. L. Large-capacity image steganography based on invertible neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10816–10825, 2021.

Lu, Y., Jiang, B., Lu, G., and Zhang, B. Implicit prompt learning for image denoising. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pp. 4678–4686, 2024.

Mielikainen, J. Lsb matching revisited. IEEE signal processing letters, 13(5):285–287, 2006.

Mittal, A., Soundararajan, R., and Bovik, A. C. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.

Mou, C., Xu, Y., Song, J., Zhao, C., Ghanem, B., and Zhang, J. Large-capacity and flexible video steganography via invertible neural network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22606–22615, 2023.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.   
Perona, P. and Malik, J. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(7):629–639, 1990.   
Pevnÿ, T., Filler, T., and Bas, P. Using high-dimensional image models to perform highly undetectable steganography. In Information Hiding: 12th International Conference, IH 2010, Calgary, AB, Canada, June 28-30, 2010, Revised Selected Papers 12, pp. 161–177. Springer, 2010.   
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211–252, 2015.   
Sameer, V. U. and Naskar, R. Universal wavelet relative distortion: A new counter–forensic attack on photo response non-uniformity based source camera identification. In Information Security Practice and Experience: 14th International Conference, ISPEC 2018, Tokyo, Japan, September 25-27, 2018, Proceedings 14, pp. 37–49. Springer, 2018.   
Scherzer, O. and Weickert, J. Relations between regularization and diffusion filtering. Journal of Mathematical Imaging and Vision, 12:43–63, 2000.   
Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=StlgiarCHLP.   
Tu, Z., Du, K., Chen, H., Wang, H., Li, W., Hu, J., and Wang, Y. Ipt-v2: Efficient image processing transformer using hierarchical attentions. arXiv preprint arXiv:2404.00633, 2024.   
Vyas, N., Kakade, S. M., and Barak, B. On provable copyright protection for generative models. In International Conference on Machine Learning, pp. 35277–35299. PMLR, 2023.   
Wallace, B., Gokul, A., and Naik, N. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22532–22541, 2023.   
Wang, F., Yin, H., Dong, Y.-J., Zhu, H., Zhang, C., Zhao, H., Qian, H., and Li, C. BELM: Bidirectional explicit linear multi-step sampler for exact inversion in diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=ccQ4fmwLDb.

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.   
Wei, P., Luo, G., Song, Q., Zhang, X., Qian, Z., and Li, S. Generative steganographic flow. In 2022 IEEE International Conference on Multimedia and Expo, pp. 1–6. IEEE, 2022.   
Wei, P., Zhou, Q., Wang, Z., Qian, Z., Zhang, X., and Li, S. Generative steganography diffusion. arXiv preprint arXiv:2305.03472, 2023.   
Weickert, J. et al. Anisotropic diffusion in image processing, volume 1. Teubner Stuttgart, 1998.   
Weng, X., Li, Y., Chi, L., and Mu, Y. High-capacity convolutional video steganography with temporal residual modeling. In Proceedings of the 2019 on International Conference on Multimedia Retrieval, pp. 87–95, 2019.   
Wouters, B. Optimizing watermarks for large language models. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=QGAeWRRe6e.   
Xu, G., Wu, H.-Z., and Shi, Y.-Q. Structural design of convolutional neural networks for steganalysis. IEEE Signal Processing Letters, 23(5):708–712, 2016.   
Yang, Y., Liu, Z., Jia, J., Gao, Z., Li, Y., Sun, W., Liu, X., and Zhai, G. Diffstega: Towards universal training-free coverless image steganography with diffusion models. In 33rd International Joint Conference on Artificial Intelligence (IJCAI), 2024.   
Yu, J., Zhang, X., Xu, Y., and Zhang, J. Cross: Diffusion model makes controllable, robust and secure image steganography. Advances in Neural Information Processing Systems, 36, 2024.   
Zhang, C., Benz, P., Karjauv, A., Sun, G., and Kweon, I. S. Udh: Universal deep hiding for steganography, watermarking, and light field messaging. Advances in Neural Information Processing Systems, 33:10223–10234, 2020.   
Zhang, G., Lewis, J. P., and Kleijn, W. B. Exact diffusion inversion via bi-directional integration approximation. ECCV, 2024.   
Zhou, Z., Sun, H., Harit, R., Chen, X., and Sun, X. Coverless image steganography without embedding. In Cloud Computing and Security: First International Conference, ICCCS 2015, Nanjing, China, August 13-15, 2015. Revised Selected Papers 1, pp. 123–132. Springer, 2015.   
Zhu, J. Hidden: hiding data with deep networks. arXiv preprint arXiv:1807.09937, 2018.

Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international Conference on Computer Vision, pp. 2223–2232, 2017.
Zhu, S. C. and Mumford, D. Prior learning and gibbs reaction-diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(11):1236–1250, 1997.

# A. Security of the symmetric key

The ECDHE algorithm, widely used in the Transport Layer Security (TLS) handshake, is an effective key exchange method based on elliptic curves. Inspired by this, we treat each steganography task as an independent encryption session, adaptively generating symmetric keys linked to both the secret and reference images using the Reference-Secret Image Related Symmetric Key Generation (RSRK) module. This approach enhances the security of both the keys and the concealed secret images during the conceal and reveal processes.

We model each steganography task as an independent encryption session, where the secret image and reference image function as the client and server, respectively. Initially, features $F_{s}$ and $F_{r}$ are extracted from the secret and reference images using AlexNet. An Multilayer Perceptron (MLP) is then used to generate the private keys, formulated as $k_{pri-i} = MLP(F_{i}), i \in r, s$ , where $k_{pri-r}$ and $k_{pri-s}$ are the private keys of the reference and secret images, respectively. These private keys are securely stored to maintain their integrity. Using specific parameters W, analogous to the base point G in the ECDHE algorithm, we derive the public keys for the reference image $k_{pub-r}$ and secret image $k_{pub-s}$ as follows:

$$
\begin{array}{l} k _ {p u b - r} = W \cdot k _ {p r i - r}, \\ l _ {1} = W - 1 \end{array} \tag {22}
$$

$$
k _ {p u b - s} = W \cdot k _ {p r i - s},
$$

where $\cdot$ denotes the Hadamard product, which is commutative and satisfies the necessary conditions for the subsequent operations in the ECDHE algorithm.

After obtaining the private and public key for the secret and reference image, we follow the principle of the ECDHE algorithm to derive the symmetric key $k_{sym}$ utilized in our model, expressed as follows:

$$
\begin{array}{l} k _ {s y m} = k _ {p u b - s} \cdot k _ {p r i - r} \\ = \left(W _ {L} + W _ {S}\right) \cdot k _ {p r i - s} \cdot k _ {p r i - r} (23) \\ = \left(W _ {L} + W _ {S}\right) \cdot k _ {p r i - r} \cdot k _ {p r i - s} (23) \\ = k _ {p u b - r} \cdot k _ {p r i - s}, \\ \end{array}
$$

where $W_{S}$ represents the weight generated based on the secret image, ensuring that the weight used to generate the symmetric key is dynamically tied to the secret image. This operation guarantees that different parameters are used to derive the public key $k_{pub-s}$ and the symmetric key $k_{sym}$ when concealing different secret images, analogous to selecting specific base points G and elliptic curves E in the ECDHE algorithm. This not only aligns the symmetric key generation process with the principles and procedures of the ECDHE algorithm but also enhances the security of the symmetric key. Following the above equation, the parameters W are used for generating the public key and are defined as $W = W_{L} + W_{S}$ . As a result, for each unique combination of images, a distinct symmetric key $k_{sym}$ is generated and used in both the conceal and reveal process.

In practical application, we assume that the sender and receiver share the same reference image and employ the same image preprocessing methods. Additionally, the model parameters used for generating the private key are common to both parties. Therefore, when using the same reference image, both the sender and receiver can derive the same private key $k_{pri-r}$ corresponding to the reference image.

When transmitting the stego image $x_{stego}$ , the public key $k_{pub-s}$ associated with the secret image is sent alongside it. Upon receiving the public key of the secret image, the receiver can use the same pipeline outlined in Equation (23) to generate the symmetric key $k_{sym}$ , as the sender did during the conceal process. With this symmetric key, the receiver can accurately reveal the secret image and complete the decrypt process.

The stego image $x_{stego}$ and public key $k_{pub-s}$ can be intercepted by an attacker during transmission. In existing generative steganography methods, the security of the keys is not guaranteed, exposing the secret image $x_{sec}$ to potential risk. In contrast, in our method, even if an attacker intercepts the stego image and the public key, they cannot obtain the correct symmetric key $k_{sym}$ as the private key $k_{pri-r}$ of the reference image $x_{ref}$ is never transmitted. Without the correct symmetric key, the attacker is incapable of revealing the secret image $x_{sec}$ from the stego image. This ensures the security of the key and significantly enhance the protection of the concealed secret image.

The proposed key generation process, grounded in robust, well-established algorithms, mitigates potential theoretical vulnerabilities. Specifically, the symmetric key generation method employs the Elliptic Curve Diffie-Hellman Ephemeral (ECDHE) algorithm. The security of key exchange protocols (e.g. RSA, ECDH) is mathematically grounded in computationally hard

problems like integer factorization for RSA, and elliptic curve discrete logarithm problem for ECDH and so on. These foundational problems guarantee the computational infeasibility of deriving private keys or shared secrets from publicly available parameters. This ensures strong theoretical security guarantees, safeguarding the symmetric key's generation and protection.

![](images/0e02f18384770a712d24f1ed8631f4a60f500a8317067aa1833341bd63309dc3.jpg)

<details>
<summary>text_image</summary>

Cover
Stego
Secret
Correct key
10 ×|Secret – Revealed|
Constant
Random noise
Public Key
</details>

Figure 8. Visual results on DIV2K dataset. The secret images are revealed with four types of keys: correct keys, constant, random noise and the public keys tied to secret images.

# B. Implementation Details

Experimental Setting. The model is implemented in PyTorch and trained on the DIV2K (Agustsson & Timofte, 2017) training dataset. Training images are randomly cropped to $256 \times 256$ and augmented with random horizontal and vertical flips. Evaluation is performed on DIV2K (Agustsson & Timofte, 2017) (100 images), COCO (Lin et al., 2014) (5000 images), ImageNet (Russakovsky et al., 2015) (10,000 images), and UniStega (Yang et al., 2024) (100 images). Comparatively, images in the DIV2K dataset are center-cropped, while in the other datasets, the images are resized to $256 \times 256$ . The AdamW optimizer with an initial learning rate of $1 \times 10^{-4}$ is used for training. All experiments are conducted on a Nvidia 4090 GPU.

Degradation setting. To evaluate the robustness of our SSHR model, we test it under a variety of degradations, including Gaussian blur, Gaussian noise, JPEG compression, and Poisson noise. For Gaussian blur, we apply a Gaussian kernel with $\sigma = 0.1 \times std$ to blur the stego images via convolution with a kernel size of 3, where std denotes the standard deviation of the stego images. In the case of Gaussian noise, we generate the noise with $\sigma = 0.1 \times std$ . In addition, we set the quality factor Q = 75 for JPEG compression in our experiments.

# C. Additional Results

Table 7 provides a numerical comparison of the reference/stego image pairs with cover-based steganography methods across the DIV2K, COCO, and ImageNet datasets. For instance, on the ImageNet dataset, our model achieves a PSNR improvement of 1.81 dB and an SSIM enhancement of 0.64%. Additionally, both MAE and RMSE are reduced by 0.37 and 0.35, respectively. Similarly, on the DIV2K and COCO datasets, the SSHR model outperforms other cover-based methods,

Table 7. Numerical comparisons of reference/stego image pairs with cover-based steganography methods across various datasets, highlighting the best results in red and the second-best in bold. 

<table><tr><td rowspan="3">Method</td><td colspan="12">Reference/Stego</td></tr><tr><td colspan="4">DIV2K</td><td colspan="4">COCO</td><td colspan="4">Imagenet</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>RMSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>RMSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MAE↓</td><td>RMSE↓</td></tr><tr><td>HiDDeN</td><td>35.21</td><td>0.9691</td><td>6.98</td><td>6.82</td><td>36.71</td><td>0.9676</td><td>6.58</td><td>8.73</td><td>34.79</td><td>0.9380</td><td>6.12</td><td>7.33</td></tr><tr><td>Baluja</td><td>36.77</td><td>0.9645</td><td>3.79</td><td>5.02</td><td>36.38</td><td>0.9563</td><td>5.98</td><td>7.43</td><td>36.59</td><td>0.9520</td><td>5.61</td><td>5.41</td></tr><tr><td>Weng et.al</td><td>37.34</td><td>0.9341</td><td>3.03</td><td>3.57</td><td>37.68</td><td>0.9323</td><td>2.82</td><td>3.42</td><td>38.20</td><td>0.9368</td><td>2.68</td><td>3.24</td></tr><tr><td>UDH</td><td>38.78</td><td>0.9658</td><td>2.82</td><td>2.94</td><td>38.90</td><td>0.9650</td><td>2.77</td><td>2.90</td><td>38.96</td><td>0.9624</td><td>2.75</td><td>2.88</td></tr><tr><td>ISN</td><td>39.28</td><td>0.9853</td><td>2.34</td><td>2.91</td><td>37.95</td><td>0.9751</td><td>2.76</td><td>3.23</td><td>40.13</td><td>0.9748</td><td>1.95</td><td>2.51</td></tr><tr><td>HiNet</td><td>39.53</td><td>0.9868</td><td>2.08</td><td>2.87</td><td>39.01</td><td>0.9844</td><td>2.09</td><td>2.96</td><td>44.61</td><td>0.9927</td><td>1.52</td><td>1.63</td></tr><tr><td>Ours</td><td>41.03(1.50↑)</td><td>0.9894(0.0026↑)</td><td>1.65(0.43↓)</td><td>1.93(0.94↓)</td><td>40.21(1.20↑)</td><td>0.9885(0.0041↑)</td><td>1.63(0.46↓)</td><td>1.90(1.06↓)</td><td>46.42(1.81↑)</td><td>0.9991(0.0064↑)</td><td>1.15(0.37↓)</td><td>1.28(0.35↓)</td></tr></table>

with PSNR/SSIM improvements of 1.50 dB/0.26% and 1.20 dB/0.41%, respectively, alongside lower MAE and RMSE values. These results demonstrate that the SSHR model substantially enhances the quality and imperceptibility of stego images.

Figure 8 shows the visual results on the DIV2K dataset. The results clearly indicate that when the correct keys are used, the SSHR model successfully reveals high-quality secret images. The residual map, which is nearly entirely black, suggests minimal divergence from the ground truth. However, when an attacker uses incorrect keys, the system's security is evident, as no meaningful information can be extracted from the stego images. Even if the attacker intercepts the public key during transmission and attempts to decrypt the secret image, the model's security remains intact. Moreover, the model preserves the naturalness and imperceptibility of the stego images. These results collectively show that the suggested approach provides high security and efficacy.

Figure 9 illustrates the performance of our SSHR model compared to other generative steganography models on the UniStega dataset, using three different prompts. When compared to CRoSS (Yu et al., 2024) and DiffStega (Yang et al., 2024), our SSHR model significantly improves the naturalness and imperceptibility of the stego images. It also facilitates substantial modification of the secret image content, guided by the reference image, to minimize the similarity between the stego and secret image pairs. When incorrect keys are used during the reveal process, the security of our model is evident, as the exposed image differs drastically from the secret image and contains minimal secret information. Moreover, an attacker cannot recover any useful data if they attempt to expose the secret image using the intercepted public key. Additionally, our model almost perfectly recovers the secret image, as demonstrated by the near-black residual map between the recovered image and the ground-truth secret image, in stark contrast to the larger residuals found in CRoSS (Yu et al., 2024) and DiffStega (Yang et al., 2024). Visual results confirm that our SSHR model surpasses previous state-of-the-art (SOTA) models in terms of both effectiveness and security.

Effectiveness of Wavelet Transform. Following the success of prior work (Jing et al., 2021), we adopt Wavelet Transform to perform steganography in the frequency domain, converting the image from the spatial domain to the frequency domain. As shown in Table 5, Wavelet Transform significantly enhances our model, yielding a 2.59 dB improvement for reference/stego image pairs and a 0.88 dB improvement for revealed/secret image pairs. These results suggest that Wavelet Transform boosts performance at both the concealment and reveal stages, enhancing the quality of stego images while preserving the integrity of revealed secret images.

Effectiveness of Rep-Conv. As shown in Table 5, the Rep-Conv module significantly improves the quantitative metrics for both the stego and revealed images, with a 1.96 dB improvement for reference/stego image pairs and a 2.95 dB improvement for revealed/secret image pairs. These enhancements positively affect both the concealment and reveal stages, demonstrating that the Rep-Conv module effectively boosts the performance of the proposed model.

Effectiveness of CIGM. The CIGM preprocesses the reference images to optimize the generation process and gradually encrypts the secret images over time. As indicated in Table 5, the CIGM leads to substantial improvements in the quantitative metrics, providing a 1.4 dB improvement for reference/stego image pairs and a 3.37 dB improvement for revealed/secret image pairs. This enhancement enhances both the concealment and reveal stages. The results highlight that the CIGM reduces the similarity between the stego and secret images, while the high-quality recovered secret images further confirm the module's effectiveness and the overall performance of the model.

![](images/b77dc7abf7ec525348de2a56659dce277eace3923900e3aa2194db2d42709307.jpg)  
Figure 9. Visual contrast of our model and other generative steganography models on the UniStega dataset across three different prompts. These prompts are utilized in CRoSS and DiffStega, whereas our model functions without text prompts. The reference image is used as the image condition in both DiffStega and our model. The secret images are revealed with four types of keys: correct keys, constant, random noise and the public keys tied to secret images.