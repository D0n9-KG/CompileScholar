# DiffusionFake: Enhancing Generalization in Deepfake Detection via Guided Stable Diffusion

Ke Sun $^{1}$ , Shen Chen $^{2}$ , Taiping Yao $^{2}$ , Hong Liu $^{3,*}$

Xiaoshuai Sun $^{1}$ , Shouhong Ding $^{2}$ , Rongrong Ji $^{1}$

$^{1}$ Key Laboratory of Multimedia Trusted Perception and Efficient Computing,

Ministry of Education of China, Xiamen University, 361005, P.R. China.

$^{2}$ Youtu Lab, Tencent, P.R. China.

$^{3}$ Osaka University, Japan.

# Abstract

The rapid progress of Deepfake technology has made face swapping highly realistic, raising concerns about the malicious use of fabricated facial content. Existing methods often struggle to generalize to unseen domains due to the diverse nature of facial manipulations. In this paper, we revisit the generation process and identify a universal principle: Deepfake images inherently contain information from both source and target identities, while genuine faces maintain a consistent identity. Building upon this insight, we introduce DiffusionFake, a novel plug-and-play framework that reverses the generative process of face forgeries to enhance the generalization of detection models. DiffusionFake achieves this by injecting the features extracted by the detection model into a frozen pre-trained Stable Diffusion model, compelling it to reconstruct the corresponding target and source images. This guided reconstruction process constrains the detection network to capture the source and target related features to facilitate the reconstruction, thereby learning rich and disentangled representations that are more resilient to unseen forgeries. Extensive experiments demonstrate that DiffusionFake significantly improves cross-domain generalization of various detector architectures without introducing additional parameters during inference. Our Codes are available in https://github.com/skJack/DiffusionFake.git.

# 1 Introduction

The rapid progress in AI-generated content (AIGC) has led to the emergence of highly sophisticated forged face content, making it increasingly challenging for humans to distinguish between genuine and forged faces $[31, 54, 8, 6]$ . Face swapping, also known as Deepfakes, is one of the most well-known techniques for generating forged facial images. It replaces the face of a target individual with that of a source person to create a seamless and realistic composite image $[43]$ . The widespread proliferation of Deepfakes content on social media platforms has raised significant security concerns, including the spread of disinformation, fraud, and impersonation. As a result, developing effective and generalizable face forgery detection methods to counter these malicious attacks has become a critical challenge in the field of computer vision.

The growing diversity of facial forgery techniques has spurred interest in the general face forgery detection task $[40, 37, 26]$ , which aims to develop models that detect forgeries from unseen domains. Previous approaches primarily utilize forgery simulation $[22, 35, 3, 38]$ to augment data by simulating various forgery traces, or framework engineering to enhance generalization through specialized designs like contrastive learning, attention mechanisms, and reconstruction learning $[41, 50, 39, 2, 12]$ .

![](images/14b00ac4a18629ecfa44560803d58f4d4451e75f84803e1a6a5dd5846b8e559a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Source"] --> B["Feature Extractor"]
    C["Target"] --> B
    D["Source Feature"] --> E["Feature Fusion"]
    B --> F["Forgery Face"]
    E --> F
    F --> G["Encoder"]
    G --> H["Classification"]
    H --> I["Guide Module"]
    I --> J["Stable Diffusion"]
    J --> K["Diffusion Process"]
    L["Source Transformation"] --> M["Source-Related Feature"]
    M --> I
    N["Target Transformation"] --> O["Target-Related Feature"]
    O --> I
    P["Real"] --> H
    Q["Fake"] --> H
    R["Vanilla Detection for Inference"] -.-> G
    S["Target"] --> T["Target"]
    U["Source"] --> V["Source"]
```
</details>

Figure 1: Pipeline of the generation process of Deepfake (a) and our proposed DiffusionFake (b).

However, their generalization capabilities remain limited due to the reliance on simulating specific forgery artifacts or designing specialized architectures tailored to certain manipulation techniques.

In this paper, we aim to identify the universal features common to all Deepfake faces by revisiting the generative process underlying forged face images. As depicted in Figure 1 (a), this process can be distilled into two key steps: (1) a feature extractor module captures salient features from both the source and target images; (2) these features are seamlessly fused through a generalized feature blending module to synthesize a novel Deepfake image. While the specific implementation of feature extraction and fusion may vary across different forgery methods, ranging from learning-based to graphics-based approaches, they all adhere to this fundamental generative paradigm.

Through this analysis, we uncover a crucial insight: Deepfake images inherently amalgamate information from both source and target faces, whereas genuine images maintain a consistent identity throughout. This amalgamated information can manifest as low-level artifacts, such as injection noise patterns and spectral discrepancies, or as high-level attributes, including facial expressions and mouth movements, depending on the specific forgery method employed.

Building upon this insight, we raise a question: Can we invert the generative process to extract and leverage the amalgamated source and target features, thereby enhancing the generalization capability of existing forgery detectors?

To answer this question, we introduce DiffusionFake, a novel plug-and-play framework that harnesses the power of Stable Diffusion to guide the forgery detector in learning disentangled source and target features inherent in Deepfakes. The core idea behind DiffusionFake is to inject the features extracted by the detector into a frozen pre-trained Stable Diffusion model, compelling the detector to capture the amalgamated source and target information by optimizing the features to reconstruct the corresponding source and target images.

As illustrated in Figure 1 (b), DiffusionFake is a plug-and-play framework that can be seamlessly integrated into existing forgery detectors. The features extracted by the encoder are first passed through the Target and Source Transformation modules, which filter and weight the features to obtain target and source-related representations. These features are then injected into the Stable Diffusion model using a Guide Module, leveraging its pre-trained knowledge to reconstruct the corresponding source and target images and optimize the feature representation.

During inference, only the encoder and classification modules are used, ensuring no additional parameters or computational overhead. By compelling the detector to learn more discriminative and generalized features, DiffusionFake enhances its ability to handle unseen forgeries without compromising efficiency. For example, when integrated with EfficientNet-B4, DiffusionFake improves AUC scores on unseen Celeb-DF dataset by around 10%, demonstrating its effectiveness in enhancing the generalization capability of existing detectors.

The main contributions of our work can be summarized as follows:

\- We analyze Deepfake images from a generative perspective and propose a framework that leverages the reverse generation process to enhance the generalization capabilities of face forgery detectors.

- We introduce the DiffusionFake framework, a plug-and-play model that integrates a frozen pre-trained Stable Diffusion network to guide the forgery detector in learning disentangled source and target features inherent in Deepfakes, further enhancing the generalization.   
- Extensive experimental validations demonstrate that the DiffusionFake framework significantly improves generalization capabilities across various architectures without introducing additional inference parameters.

# 2 Related Work

# 2.1 General Face Forgery Detection

General face forgery detection aims to improve the performance of forgery detectors on unseen domains and become one of the most critical issues in this field. Previous work to enhance generalization can be broadly divided into two categories: forgery simulation and framework engineering. The former utilizes data augmentation methods to simulate certain forgery traces, such as blending artifacts $[13, 22, 35]$ , Inconsistency between internal and external faces $[51]$ , subtle jitter and blur traces $[19]$ , and fine-grained facial disharmony $[3, 38]$ . The latter improves network architectures or training procedures to help capture more generalized traces. Such methods approach the problem from different angles. Some employ attention mechanisms to enhance the capture of forgery traces $[50, 39, 47, 33]$ , while others improve generalization by jointly modeling frequency and spatial domains $[29, 21, 28, 25]$ . Reconstruction-based methods enhance discriminability against unseen domain forgeries via modeling genuine faces $[5, 2, 34]$ . Additionally, some approaches use implicit identity as a clue to improve the generalization of Deepfake faces $[17, 9]$ and some explore the local and global relationships of unseen forgeries $[4, 1, 45, 13, 10, 27]$ . Furthermore, decoupling methods $[24, 32, 14, 20]$ , such as ICT $[11]$ and UCF $[48]$ , aim to enhance generalization by disentangling different facial information. Our DiffusionFake method addresses this by reversing the forgery process and leveraging pre-trained generative models to complete missing information, enhancing the capture of source-related and target-related features.

# 2.2 Diffusion Model

Diffusion models have emerged as a powerful framework for image generation and manipulation. The seminal work on Denoising Diffusion Probabilistic Models (DDPM) $[15]$ introduced a novel approach to learn the data distribution by iteratively denoising a Gaussian noise signal. This process allows for high-quality image generation but requires a large number of sampling steps. To address this issue, the Denoising Diffusion Implicit Models (DDIM) $[36]$ proposed a deterministic sampling process that significantly accelerates the generation process while maintaining image quality. Building upon these advancements, the Latent Diffusion Model (LDM) $[30]$ combines the strengths of Variational Autoencoders (VAEs) $[18]$ and diffusion models. By applying the diffusion process in the latent space learned by a VAE, LDM substantially reduces the computational cost and memory requirements during training. This innovative architecture has given rise to powerful AIGC pre-trained generative models, such as Stable Diffusion ${}^{2}$ , which enable high-quality image generation and manipulation with unprecedented efficiency. Recent developments in controllable diffusion models have further expanded their applicability. ControlNet $[49]$ introduces a mechanism to guide the image generation process by conditioning the diffusion model on additional control signals, such as segmentation masks or edge maps. Inspired by ControlNet, our DiffusionFake leverages a guide module to inject the source and target-related features into Stable Diffusion to reconstruct the corresponding images.

# 3 Methodology

Figure 2 illustrates the detailed framework of our proposed DiffusionFake method, which aims to enhance the generalization capability of forgery detectors by guiding the learning of amalgamated source and target features through a frozen pre-trained Stable Diffusion model. Specifically, the features extracted by the encoder are first filtered and weighted by the Feature Filter and Weight Modules to obtain source and target-related representations. These features are then injected into a

![](images/0e0c68c229978b724538192006e67a9df85ab6dd280a60e9d35ac80eb59f7513.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x"] --> B["Encoder"]
    B --> C["f"]
    C --> D["Target Filter"]
    D --> E["ft"]
    E --> F["SD Enc 1"]
    F --> G["SD Enc 2"]
    G --> H["SD Enc 3"]
    H --> I["SD Enc 4"]
    I --> J["SD Mid"]
    J --> K["Multi-scale Features"]
    K --> L["Zero Conv"]
    L --> M["Feature Filter"]
    M --> N["Conv1"]
    N --> O["Channel Attention"]
    O --> P["Space Attention"]
    P --> Q["SelfAttention"]
    Q --> R["Conv2"]
    R --> S["Conv3"]
    S --> T["Alignment"]
    T --> U["Final Feature"]
    
    subgraph Feature Filters
        V["Conv1"] --> W["Channel Attention"]
        W --> X["Space Attention"]
        X --> Y["SelfAttention"]
        Y --> Z["Conv2"]
        Z --> AA["Conv3"]
        AA --> AB["Alignment"]
        AB --> AC["Final Feature"]
    
    subgraph Stable Diffusion
        AD["x_t"] --> AE["VAE-E"]
        AF["x_s"] --> AE
        AE --> AG["Z_t"]
        AG --> AH["Diffusion Forward"]
        AH --> AI["Z_s"]
        AI --> AJ["Z_{t}"]
        AJ --> AK["Z_{t+1}"]
        AK --> AL["Z_{t-1}"]
        AL --> AM["U_E"]
        AM --> AN["U_D"]
        AN --> AO["Z_{t+1}"]
        AO --> AP["Z_{t-1}"]
        AP --> AQ["Z_{t-1}"]
        AQ --> AR["Z_{t+1}"]
        AR --> AS["Z_{t-1}"]
        AS --> AT["Z_{t+1}"]
        AT --> AU["Z_{t-1}"]
        AU --> AV["Z_{t-1}"]
        AV --> AW["Z_{t+1}"]
        AW --> AX["Z_{t-1}"]
        AX --> AY["Z_{t-1}"]
        AY --> AZ["Z_{t+1}"]
        AZ --> BA["Z_{t-1}"]
        BA --> BB["Z_{t-1}"]
        BB --> BC["Z_{t+1}"]
        BC --> BD["Z_{t-1}"]
        BD --> BE["Z_{t-1}"]
        BE --> BF["Z_{t+1}"]
        BF --> BG["Z_{t-1}"]
        BG --> BH["Z_{t-1}"]
        BH --> BI["Z_{t+1}"]
        BI --> BJ["Z_{t-1}"]
        BJ --> BK["Z_{t-1}"]
        BK --> BL["Z_{t+1}"]
        BL --> BM["Z_{t-1}"]
        BM --> BN["Z_{t-1}"]
        BN --> BO["Z_{t+1}"]
        BO --> BP["Z_{t-1}"]
        BP --> BQ["Z_{t-1}"]
        BQ --> BR["Z_{t+1}"]
        BR --> BS["Z_{t-1}"]
        BS --> BT["Z_{t-1}"]
        BT --> BU["Z_{t+1}"]
        BU --> BV["Z_{t-1}"]
        BV --> BW["Z_{t-1}"]
        BW --> BX["Z_{t+1}"]
        BX --> BY["Z_{t-1}"]
        BY --> BZ["Z_{t-1}"]
        BZ --> CA["Z_{t+1}"]
        CA --> CB["Z_{t-1}"]
        CB --> CC["Z_{t-1}"]
        CC --> CD["Z_{t+1}"]
        CD --> CE["Z_{t-1}"]
        CE --> CF["Z_{t-1}"]
        CF --> CG["Z_{t+1}"]
        CG --> CH["Z_{t-1}"]
        CH --> CI["Z_{t-1}"]
        CI --> CJ["Z_{t+1}"]
        CJ --> CK["Z_{t-1}"]
        CK --> CL["Z_{t-1}"]
        CL --> CM["Z_{t+1}"]
        CM --> CN["Z_{t-1}"]
        CN --> CO["Z_{t-1}"]
        CO --> CP["Z_{t+1}"]
        CP --> CQ["Z_{t-1}"]
        CQ --> CR["Z_{t-1}"]
        CR --> CS["Z_{t+1}"]
        CS --> CT["Z_{t-1}"]
        CT --> CU["Z_{t-1}"]
        CU --> CV["Z_{t+1}"]
        CV --> CW["Z_{t-1}"]
        CW --> CX["Z_{t-1}"]
    end
```
</details>

Figure 2: The details of the DiffusionFake method. The blue arrow represents the target branch, the red arrow represents the source branch, the ✿ represents the parameter frozen and does not participate in training, and the 🔊 represents the trainable module.

frozen pre-trained Stable Diffusion model via the Guide Module, which reconstructs the corresponding source and target images, compelling the encoder to learn rich and discriminative features.

# 3.1 Preliminaries

Diffusion Process. Denoising Diffusion Probabilistic Models (DDPMs) [15] are latent variable models that learn to generate data by reversing a gradual noising process. The forward diffusion process gradually adds Gaussian noise to the data $x_{0}$ according to a variance schedule $\beta_{1},\ldots,\beta_{T}$ , producing a sequence of noisy samples $x_{1},\ldots,x_{T}$ . The forward process can be described as:

$$
q (x _ {t} | x _ {t - 1}) = \mathcal {N} (x _ {t}; \sqrt {1 - \beta_ {t}} x _ {t - 1}, \beta_ {t} \mathbf {I}) \tag {1}
$$

The reverse denoising process learns to generate samples from the data distribution by starting with a Gaussian noise sample $x_{T}$ and iteratively denoising it using a learned denoising function $\epsilon_{\theta}$ . The reverse process is defined as:

$$
p _ {\theta} (x _ {t - 1} | x _ {t}) = \mathcal {N} (x _ {t - 1}; \mu_ {\theta} (x _ {t}, t), \sigma_ {t} ^ {2} \mathbf {I}), \tag {2}
$$

$$
\mu_ {\theta} (x _ {t}, t) = \frac {1}{\sqrt {\alpha_ {t}}} \left(x _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \bar {\alpha} t}} \epsilon_ {\theta} (x _ {t}, t)\right), \tag {3}
$$

$\alpha_{t} = 1 - \beta_{t}, \bar{\alpha}_{t} = \prod_{s=1}^{t} \alpha_{s}$ , and $\sigma_{t}^{2} = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \beta_{t}$ . The training objective is to minimize the weighted sum of the denoising error at each step:

$$
L = \mathbb {E} _ {\epsilon \sim \mathcal {N} (0, 1), t \sim [ 1, T ]} \left[ | | \epsilon - \epsilon_ {\theta} (x _ {t}, t) | | _ {2} ^ {2} \right] \tag {4}
$$

Stable Diffusion. The Stable Diffusion Model is a powerful pre-trained model with impressive generative capabilities, able to synthesize various types of images, including different types of human faces. Built upon the DDPM framework, the Stable Diffusion models employs a Latent Diffusion Model (LDM) [30] to reduce resource consumption. LDM applies the diffusion process in a learned latent space instead of pixel space, which is obtained by training an autoencoder. The training objective is:

$$
L = \mathbb {E} _ {\epsilon \sim \mathcal {N} (0, 1), t \sim [ 1, T ]} \left[ | | \epsilon - \epsilon_ {\theta} (z _ {t}, t) | | _ {2} ^ {2} \right], \tag {5}
$$

where $z_{t}$ is the latent representation encoded by the VAE encoder. This strategic application of latent space modeling not only enhances efficiency but also preserves the high quality of generated images.

# 3.2 Feature Transformation

Given an input image x and its corresponding label y, where y = 0 represents a real face and y = 1 represents a forged face, let $x_{s}$ and $x_{t}$ denote the corresponding source and target images from the training dataset, respectively. For real faces, $x_{s}$ and $x_{t}$ are identical to x. Let E be the encoder, and the extracted features be $f = E(x)$ . To transform the extracted feature f into components that can guide the Stable Diffusion process, we first introduce two key modules: the Feature Filter Module and the Weight Module.

Feature Filter Module. The Feature Filter Module F is to extract source-related and target-related features from the encoded features f. To achieve this, we employ two filter networks, $F_{s}$ and $F_{t}$ , to obtain the source-related feature $f_{s} = F_{s}(f)$ and target-related feature $f_{t} = F_{t}(f)$ , respectively.

As shown in Figure 2, the Feature Filter Module combines convolutional layers and attention mechanisms. The features first pass through a convolutional layer to transform the channels. Then channel-wise [16] and spatial-wise attention [46] are applied to adaptively weight and filter the features. These attention mechanisms help to emphasize the most relevant features while suppressing less informative ones, leading to more discriminative representations.

Subsequently, a Multi-Head Attention mechanism $[44]$ is then applied to perform cross-attention between the original features (query) and the attention-filtered features (key and value). This operation captures long-range dependencies and enhances the receptive field, enabling more effective feature refinement. Finally, to ensure compatibility with the encoder of the Stable Diffusion model, we apply upsampling and pooling operations to align the feature dimensions.

Weight Module. The Weight Module W addresses the varying levels of source and target information embedded in different types of forged images. For example, Deepfakes may evenly blend source and target features, while expression-driven methods like NeuralTextures may predominantly feature target image information with minimal source information confined to specific regions like mouth movements. Uniformly feeding these into the guide module would be suboptimal.

To mitigate this, we use two separate weight modules, $W_{s}$ and $W_{t}$ , to estimate the information content for source and target features. Each weight module starts with a pooling layer, following five MLP layers, and a sigmoid function finally outputs the weight. We train these modules using the similarity scores between the input image and its respective source and target images as ground truth.

Specifically, we encode $x, x_{t}$ , and $x_{s}$ using the pre-trained VAE-Encoder from Stable Diffusion to obtain latent representations $z, z_{t}$ , and $z_{s}$ . Such a well-pretrained model can effectively capture and quantify the differences between images, providing a reliable basis for measuring the similarity between the input image and its corresponding source and target images. The similarity scores between z and $z_{t}$ , and between z and $z_{s}$ , are computed and used to train the weight modules with mean squared error (MSE) loss as follows:

$$
\mathcal {L} _ {w s} = \left| \left| W _ {s} (f) - \operatorname{sim} \left(z, z _ {s}\right) \right| \right| _ {2} ^ {2} \tag {6}
$$

$$
\mathcal {L} _ {w t} = \left| \left| W _ {t} (f) - \operatorname{sim} \left(z, z _ {t}\right) \right| \right| _ {2} ^ {2} \tag {7}
$$

where $\mathrm{sim}(a,b) = \frac{a\cdot b}{|a||b|}$ denotes the cosine similarity between vectors $a$ and $b$ .

By dynamically adjusting the influence of source and target features during the diffusion process, the Weight Module ensures optimal guidance for the Stable Diffusion model, thereby enhancing the encoder's ability to extract generalized features suitable for detecting a wide range of forgeries.

# 3.3 Guide Module

The Guide Module is designed to inject the source-related and target-related features into the frozen pre-trained Stable Diffusion model to guide the reconstruction of the source and target images. As illustrated in Figure 2, the Guide Module employs trainable copy and zero convolution layers for feature injection, inspired by ControlNet [49].

Let $U_{E}(\cdot)$ and $U_{D}(\cdot)$ denote the neural block of the encoder and decoder in the U-Net $\epsilon_{\theta}$ network of the Stable Diffusion model, respectively. The Guide Module first creates a trainable copy of $U_{E}(\cdot)$ , denoted as $U_{E}^{\prime}(\cdot)$ . The source-related feature $f_{s}$ and target-related feature $f_{t}$ are then independently fed into $U_{E}^{\prime}(\cdot)$ . The resulting features are combined with the corresponding features from the locked

model's decoder $U_{D}(;)$ using zero convolution layers $Z(\cdot)$ , which are $1 \times 1$ convolutional layers with weights and biases initialized to zeros. This initialization minimizes the impact on the pre-trained model at the beginning of training, stabilizing the training process [49]. The final output of the Guide Module can be summarized as:

$$
f s ^ {\prime} = U _ {D} \left(U _ {E} \left(z _ {s}\right)\right) + Z \left(U _ {E} ^ {\prime} \left(f _ {s}\right)\right) \times W _ {s} (f) \tag {8}
$$

$$
f t ^ {\prime} = U _ {D} \left(U _ {E} \left(z _ {t}\right)\right) + Z \left(U _ {E} ^ {\prime} \left(f _ {t}\right)\right) \times W _ {t} (f) \tag {9}
$$

where $z_{s}$ and $z_{t}$ are the latent representations of the source and target images, respectively, obtained from the pre-trained VAE-Encoder of Stable Diffusion, and $W_{s}(f)$ and $W_{t}(f)$ are the weights computed by the Weight Module.

Unlike ControlNet, which aims to control the generated results of the diffusion model, our objective is to optimize the features f by fixing the output and encouraging the capture of more generalizable and disentangled features. By guiding the reconstruction of the source and target images using the respective features, the Guide Module facilitates the learning of rich and discriminative representations that enhance the performance of the forgery detector across various domains and attack types.

During training, we follow a process similar to the LDM [30], gradually executing the diffusion process, including the time step t, to guide the reconstruction of the source and target images. At each time step t, the model learns to predict the noise $\epsilon$ that was added to the latent representation of the source or target image. The overall learning objective for the source and target diffusion models can be formulated as:

$$
L _ {s} = \mathbb {E} _ {\epsilon \sim \mathcal {N} (0, 1), t \sim [ 1, T ]} \left[ | | \epsilon - \epsilon_ {\theta} (f s _ {t} ^ {\prime}, t) | | _ {2} ^ {2} \right] \tag {10}
$$

$$
L _ {t} = \mathbb {E} _ {\epsilon \sim \mathcal {N} (0, 1), t \sim [ 1, T ]} \left[ | | \epsilon - \epsilon_ {\theta} (f t _ {t} ^ {\prime}, t) | | _ {2} ^ {2} \right] \tag {11}
$$

where $fs_t'$ and $ft_t'$ represent the features within the embedding of time step $t$ .

# 3.4 Loss Function and Inference

We apply a simple binary classification head to the extracted feature f to obtain the predicted label $y'$ , which is calculated via typical cross-entropy loss as follows:

$$
L _ {c e} = - \left[ y \log y ^ {\prime} + (1 - y) \log (1 - y ^ {\prime}) \right] \tag {12}
$$

Thus, the final loss function combines Eq.12 with the losses from our Weight module and Gudie module, which is defined as follows:

$$
L = L _ {c e} + \lambda_ {s} L _ {s} + \lambda_ {t} L _ {t} + L _ {w s} + L _ {w t} \tag {13}
$$

where $\lambda_{s}$ and $\lambda_{t}$ are hyperparameters that balance the contributions of the source and target diffusion losses, $L_{s}$ and $L_{t}$ , respectively.

Inference. During inference, only the Encoder and the classification head are retained, as shown in the purple dotted box in Figure 2. It is worth noting that DiffusionFake ensures the encoder network extracts generalized features only during training. Consequently, our DiffusionFake framework does not introduce any additional parameters during inference, thereby enhancing generalizability and reducing computational overhead.

# 4 Experiment

# 4.1 Experimental Setting

Dataset. To evaluate the generalization ability of DiffusionFake, we conduct experiments on several challenging datasets: (1) FaceForensics++ (FF++) [31]: This widely-used dataset contains 1,000 videos with four manipulation methods: DeepFakes, NeuralTextures, Face2Face, and FaceSwap. The pairwise real and forged data enable the generation of mixed forgery images. (2) Celeb-DF [23]: A high-quality DeepFake dataset containing various scenarios. (3) DeepFake Detection (DFD): This

Table 1: Frame-level cross-database evaluation from FF++(HQ) to Celeb-DF, Wild Deepfake, DFDC-P, DFD, and DiffSwap in terms of AUC and EER. \* represents the results reproduced using open-source code or model. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Celeb-DF</td><td colspan="2">Wild Deepfake</td><td colspan="2">DFDC-P</td><td colspan="2">DFD</td><td colspan="2">DiffSwap</td><td colspan="2">Average</td></tr><tr><td>AUC</td><td>EER</td><td>AUC</td><td>EER</td><td>AUC</td><td>EER</td><td>AUC</td><td>EER</td><td>AUC</td><td>EER</td><td>AUC</td><td>EER</td></tr><tr><td>Xception [7]</td><td>65.27</td><td>38.77</td><td>66.17</td><td>40.14</td><td>69.80</td><td>35.41</td><td>87.86</td><td>21.04</td><td>74.25</td><td>32.04</td><td>72.67</td><td>33.48</td></tr><tr><td>Face X-ray [42]</td><td>74.20</td><td>-</td><td>-</td><td>-</td><td>70.00</td><td>-</td><td>85.60</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>F3-Net* [29]</td><td>71.21</td><td>34.03</td><td>67.71</td><td>40.17</td><td>72.88</td><td>33.38</td><td>86.10</td><td>26.17</td><td>76.89</td><td>30.83</td><td>74.96</td><td>32.92</td></tr><tr><td>MAT* [50]</td><td>70.65</td><td>35.83</td><td>70.15</td><td>36.53</td><td>67.34</td><td>38.31</td><td>87.58</td><td>21.73</td><td>79.93</td><td>27.77</td><td>75.13</td><td>32.03</td></tr><tr><td>GFF* [28]</td><td>75.31</td><td>32.48</td><td>66.51</td><td>41.52</td><td>71.58</td><td>34.77</td><td>85.51</td><td>25.64</td><td>78.38</td><td>28.15</td><td>75.46</td><td>32.51</td></tr><tr><td>LTW [40]</td><td>77.14</td><td>29.34</td><td>67.12</td><td>39.22</td><td>74.58</td><td>33.81</td><td>88.56</td><td>20.57</td><td>77.95</td><td>29.01</td><td>77.07</td><td>30.39</td></tr><tr><td>LRL [4]</td><td>78.26</td><td>29.67</td><td>68.76</td><td>37.50</td><td>76.53</td><td>32.41</td><td>89.24</td><td>20.32</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DCL [41]</td><td>82.30</td><td>26.53</td><td>71.14</td><td>36.17</td><td>76.71</td><td>31.97</td><td>91.66</td><td>16.63</td><td>80.21</td><td>27.37</td><td>80.40</td><td>27.73</td></tr><tr><td>PCL+I2G [51]</td><td>81.80</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SBI* [35]</td><td>80.76</td><td>26.97</td><td>68.22</td><td>38.11</td><td>76.53</td><td>30.22</td><td>88.13</td><td>17.25</td><td>75.20</td><td>31.49</td><td>77.77</td><td>28.81</td></tr><tr><td>UIA-ViT [53]</td><td>82.41</td><td>-</td><td>-</td><td>-</td><td>75.80</td><td>-</td><td>94.68</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RECCE* [2]</td><td>70.50</td><td>35.34</td><td>67.93</td><td>39.82</td><td>75.88</td><td>32.41</td><td>89.91</td><td>19.95</td><td>77.59</td><td>29.38</td><td>76.36</td><td>31.38</td></tr><tr><td>UCF [48]</td><td>75.27</td><td>-</td><td>-</td><td>-</td><td>75.94</td><td>-</td><td>80.74</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CADDM* [9]</td><td>77.56</td><td>30.63</td><td>72.56</td><td>33.63</td><td>72.45</td><td>33.56</td><td>82.90</td><td>25.20</td><td>75.58</td><td>31.01</td><td>76.21</td><td>30.81</td></tr><tr><td>EN-b4* [42]</td><td>73.51</td><td>34.17</td><td>70.04</td><td>37.03</td><td>70.51</td><td>33.98</td><td>87.57</td><td>21.31</td><td>77.38</td><td>29.44</td><td>75.80</td><td>31.19</td></tr><tr><td>VIT-B* [42]</td><td>74.64</td><td>33.07</td><td>75.46</td><td>31.53</td><td>74.24</td><td>34.29</td><td>84.38</td><td>24.15</td><td>78.50</td><td>28.14</td><td>77.44</td><td>30.24</td></tr><tr><td>En-b4+Ours</td><td>83.17</td><td>24.59</td><td>75.17</td><td>33.25</td><td>77.35</td><td>30.17</td><td>91.71</td><td>16.27</td><td>82.02</td><td>25.55</td><td>81.88</td><td>25.97</td></tr><tr><td>VIT-B+Ours</td><td>80.46</td><td>27.51</td><td>80.14</td><td>29.62</td><td>80.95</td><td>27.66</td><td>90.36</td><td>19.73</td><td>86.98</td><td>21.32</td><td>83.78</td><td>25.17</td></tr></table>

dataset comprises 363 real videos and 3,068 fake videos, primarily generated using the DeepFake method. (4) DFDC Preview (DFDC-P) [8]: A challenging dataset with 1,133 real videos and 4,080 fake videos, featuring various manipulation methods and backgrounds. (5) WildDeepfake [54]: A diverse dataset obtained from the internet, capturing a wide range of real-world scenarios. (6) DiffSwap [6]: A recently released dataset containing 30,000 high-quality face swaps generated using the diffusion-based DiffSwap method [52] on the MM-Celeb-A dataset. This dataset allows for evaluating cross-method generalization.

Training details. DiffusionFake is a plug-and-play architecture that can be integrated with different backbone networks by simply adjusting the dimensions of the alignment layer in the Feature Filter module. During training, we utilize a pre-trained Stable Diffusion 1.5 model with frozen parameters. Input images are resized to $224 \times 224$ pixels. We employ the Adam optimizer with a learning rate of 1e-5 and a batch size of 32. The model is trained for 20 epochs. The hyperparameters $\lambda_{s}$ and $\lambda_{t}$ are set to 0.7 and 1, respectively. We employ widely used data augmentations, such as HorizontalFlip, and CutOut. To ensure a fair comparison, we follow the data split strategy used in FaceForensics++ [31]. The overall framework is implemented in Pytorch on one NVIDIA A-100 GPU.

# 4.2 Experimental Results

We use AUC and EER to evaluate all the methods, including ours, both of them are widely used in deepfake detection. $^{3}$ We compare DiffusionFake with several state-of-the-art methods.

Cross-dataset evaluation. To validate the generalization capability of DiffusionFake, we first evaluate its performance on unseen datasets against recent state-of-the-art methods. Following previous settings, we train the models on the FF++ dataset and test them on several unseen domain datasets. The frame-level results are shown in Table 1, where \* denotes results obtained using official code with consistent training settings and data.

We evaluate its performance using two representative backbones: EfficientNet-B4 (En-B4) and ViT-B. We observe that incorporating DiffusionFake significantly improves the generalization ability of both architectures compared to their original classification backends. For En-B4, our method boosts performance on Celeb-DF by 11% and achieves an average improvement of 6%. Similarly, ViT-B sees a 6% increase on DFDC and an average gain of 6% when trained with DiffusionFake. Remarkably, these enhancements are achieved without increasing the parameter count or computational

Table 2: Abalation study of different components of DiffusionFake. 

<table><tr><td rowspan="2">SD</td><td rowspan="2">Filter</td><td rowspan="2">Weight</td><td colspan="2">Celeb-DF</td><td colspan="2">DFDC-P</td></tr><tr><td>AUC</td><td>EER</td><td>AUC</td><td>EER</td></tr><tr><td>×</td><td>×</td><td>×</td><td>71.87</td><td>34.28</td><td>71.78</td><td>35.01</td></tr><tr><td>×</td><td>√</td><td>√</td><td>73.87</td><td>32.06</td><td>72.41</td><td>34.25</td></tr><tr><td>√</td><td>×</td><td>×</td><td>77.35</td><td>29.05</td><td>75.69</td><td>32.12</td></tr><tr><td>√</td><td>√</td><td>×</td><td>80.79</td><td>26.37</td><td>76.17</td><td>31.57</td></tr><tr><td>√</td><td>×</td><td>√</td><td>78.67</td><td>28.33</td><td>76.59</td><td>31.22</td></tr><tr><td>√</td><td>√</td><td>√</td><td>83.17</td><td>24.59</td><td>77.35</td><td>30.17</td></tr></table>

Table 3: Abalation study of backbones. 

<table><tr><td rowspan="2">Backbone</td><td colspan="2">Celeb-DF</td><td colspan="2">WDF</td></tr><tr><td>AUC</td><td>EER</td><td>AUC</td><td>EER</td></tr><tr><td>ResNet</td><td>68.89</td><td>36.78</td><td>69.91</td><td>38.07</td></tr><tr><td>ResNet+Ours</td><td>75.27</td><td>32.44</td><td>73.25</td><td>34.27</td></tr><tr><td>En-b0</td><td>71.74</td><td>34.56</td><td>69.24</td><td>38.32</td></tr><tr><td>En-b0+Ours</td><td>76.31</td><td>31.56</td><td>74.40</td><td>33.99</td></tr><tr><td>Vit-S</td><td>70.59</td><td>35.87</td><td>70.60</td><td>37.59</td></tr><tr><td>Vit-S+Ours</td><td>74.58</td><td>32.95</td><td>75.10</td><td>33.87</td></tr></table>

overhead during inference. Compared to state-of-the-art methods, DiffusionFake outperforms disentanglement-based approaches like UCF and CAADM on Celeb-DF. Moreover, our method demonstrates substantial improvements on the latest diffusion-based face swapping dataset, DiffSwap, highlighting its effectiveness against the most recent forgery techniques. These results validate the ability of the guide module and Stable Diffusion network to encourage the encoder to learn more generalizable features by reconstructing source and target images. Due to space limitations, we provide the results of single-source and multi-source cross-manipulation evaluations in the appendix.

# 4.3 Ablation Study

Ablation of components. We conducted an ablation study to investigate the impact of the key modules in DiffusionFake: 1) the pre-trained Stable Diffusion (SD) model, 2) the Feature Filter module, and 3) the Weight Module. The results are shown in Table 2, where without SD refers to not loading the pre-trained weights of the SD model, and without Filter means directly feeding the encoder's output features f into the guide module.

We can observe that the pre-trained Stable Diffusion model is crucial for the DiffusionFake framework. Without the pre-trained weights, the network struggles to reconstruct the source and target images due to information loss, hindering the training process. Furthermore Both the Feature Filter and Weight modules play significant roles, and removing either of them leads to a performance decline. Specifically, eliminating the Filter module results in a 5% AUC drop, as the filtering component allows the reconstruction to focus on relevant information without interference from redundant features. On the other hand, the absence of the Weight module causes a 3% performance decrease, as this module assesses the amount of source and target information contained in the image, providing a prior for the generative network to determine the importance of guided information during the reconstruction process.

Ablation of backbones. As our method can be flexibly embedded into different backbones by adjusting the alignment of the Feature Filter, we conduct an ablation study on various backbone architectures to demonstrate the versatility of DiffusionFake. We experiment with traditional ResNet-34, lightweight EfficientNet-B0, and the ViT-based ViT-Small. The results in Table 3 show that integrating our method into these backbones significantly improves generalization performance. For instance, applying DiffusionFake to the lightweight EfficientNet-B0 increases the generalization accuracy on Celeb-DF from 71.75% to 76.31%, surpassing the original EfficientNet-B4 (73.51%). This evidence suggests that our method can effectively drive different encoders to extract more generalizable features.

# 4.4 Analysis and Visualizations

Visualizations of reverse results. Figure 3 showcases the reconstruction results for both training and unseen samples using DiffusionFake. For training samples (Figure 3 A), DiffusionFake effectively reconstructs the target image, despite the source image being slightly blurry due to information loss, capturing the basic characteristics of the ground truth. In order to compare the reconstruction effect more intuitively, we use the RECCE method to directly reconstruct the source and target images of the fake image. It can be seen that the reconstruction effect is very poor. In contrast, the reconstruction effect of our method is better due to the help of the pre-trained SD model. For unseen samples (Figure 3 B), fake images with mixed features result in significant differences between reconstructed target

![](images/8dec4a0e09f1b25b7630a293f709d148ea1a7df952a300a0ecac77639f5e5fbc.jpg)

Figure 3: Reconstruction results of DiffusionFake for training (A) and unseen (B) samples. For unseen samples, the model is provided with three sets of initial Gaussian noise, differing only in the injected guide information. The numbers below represent the Euclidean distance between the corresponding source and target features.   
![](images/fd02539f69d2151434df4c702625dc175198ab693c0f749113ab52e5e1069373.jpg)

<details>
<summary>histogram</summary>

| Distance Range | Real Density | Fake Density |
| -------------- | ------------ | ------------ |
| 0.00 - 0.01    | 55           | 0            |
| 0.01 - 0.02    | 35           | 0            |
| 0.02 - 0.03    | 15           | 0            |
| 0.03 - 0.04    | 5            | 0            |
| 0.04 - 0.05    | 2            | 0            |
| 0.05 - 0.06    | 1            | 0            |
| 0.06 - 0.07    | 0            | 0            |
| 0.07 - 0.08    | 0            | 0            |
| 0.08 - 0.09    | 0            | 0            |
| 0.09 - 0.10    | 0            | 0            |
| 0.10 - 0.11    | 0            | 0            |
| 0.11 - 0.12    | 0            | 0            |
| 0.12 - 0.13    | 0            | 5            |
| 0.13 - 0.14    | 0            | 10           |
| 0.14 - 0.15    | 0            | 15           |
| 0.15 - 0.16    | 0            | 20           |
| 0.16 - 0.17    | 0            | 25           |
| 0.17 - 0.18    | 0            | 28           |
| 0.18 - 0.19    | 0            | 25           |
| 0.19 - 0.20    | 0            | 20           |
| 0.20 - 0.21    | 0            | 15           |
| 0.21 - 0.22    | 0            | 10           |
| 0.22 - 0.23    | 0            | 5            |
| 0.23 - 0.24    | 0            | 2            |
| 0.24 - 0.25    | 0            | 1            |
</details>

![](images/c825166ccf23c7f09bfb510ab1e864acd2abfecf81abe34b24b7d05f2b2fef42.jpg)

<details>
<summary>histogram</summary>

| Distance Range | Real Density | Fake Density |
| -------------- | ------------ | ------------ |
| 0.00 - 0.01    | 35           | 2            |
| 0.01 - 0.02    | 28           | 3            |
| 0.02 - 0.03    | 15           | 4            |
| 0.03 - 0.04    | 8            | 5            |
| 0.04 - 0.05    | 6            | 6            |
| 0.05 - 0.06    | 5            | 7            |
| 0.06 - 0.07    | 4            | 8            |
| 0.07 - 0.08    | 3            | 9            |
| 0.08 - 0.09    | 2            | 10           |
| 0.09 - 0.10    | 1            | 11           |
| 0.10 - 0.11    | 1            | 12           |
| 0.11 - 0.12    | 1            | 13           |
| 0.12 - 0.13    | 1            | 14           |
| 0.13 - 0.14    | 1            | 15           |
| 0.14 - 0.15    | 1            | 16           |
| 0.15 - 0.16    | 1            | 17           |
| 0.16 - 0.17    | 1            | 18           |
| 0.17 - 0.18    | 1            | 19           |
| 0.18 - 0.19    | 1            | 20           |
| 0.19 - 0.20    | 1            | 21           |
| 0.20 - 0.21    | 1            | 22           |
| 0.21 - 0.22    | 1            | 23           |
| 0.22 - 0.23    | 1            | 24           |
| 0.23 - 0.24    | 1            | 25           |
| 0.24 - 0.25    | 1            | 26           |
| 0.25 - 0.26    | 1            | 27           |
| 0.26 - 0.27    | 1            | 28           |
| 0.27 - 0.28    | 1            | 29           |
| 0.28 - 0.29    | 1            | 30           |
| 0.29 - 0.30    | 1            | 31           |
| 0.30 - 0.31    | 1            | 32           |
| 0.31 - 0.32    | 1            | 33           |
| 0.32 - 0.33    | 1            | 34           |
| 0.33 - 0.34    | 1            | 35           |
| 0.34 - 0.35    | 1            | 36           |
| 0.35 - 0.36    | 1            | 37           |
| 0.36 - 0.37    | 1            | 38           |
| 0.37 - 0.38    | 1            | 39           |
| 0.38 - 0.39    | 1            | 40           |
| 0.39 - 0.40    | 1            | 41           |
| 0.40 - 0.41    | 1            | 42           |
| 0.41 - 0.42    | 1            | 43           |
| 0.42 - 0.43    | 1            | 44           |
| 0.43 - 0.44    | 1            | 45           |
| 0.44 - 0.45    | 1            | 46           |
| 0.45 - 0.46    | 1            | 47           |
| 0.46 - 0.47    | 1            | 48           |
| 0.47 - 0.48    | 1            | 49           |
| 0.48 - 0.49    | 1            | 50           |
| 0.49 - 0.50    | 1            | 51           |
| 0.50 - 0.51    | 1            | 52           |
| 0.51 - 0.52    | 1            | 53           |
| 0.52 - 0.53    | 1            | 54           |
| 0.53 - 0.54    | 1            | 55           |
| 0.54 - 0.55    | 1            | 56           |
| 0.55 - 0.56    | 1            | 57           |
| 0.56 - 0.57    | 1            | 58           |
| 0.57 - 0.58    | 1            | 59           |
| 0.58 - 0.59    | 1            | 60           |
| 0.59 - 0.60    | 1            | 61           |
| 0.60 - 0.61    | 1            | 62           |
| Note: The actual values may vary due to the random nature of the data generation process (e.g., Gaussian or Gaussian noise). The provided values are just examples from the code execution.
</details>

![](images/97bd79b749ff660252d6c5998bf41073fe4deda93018a4bd5989dc93b71a6699.jpg)

<details>
<summary>histogram</summary>

| Distance Range | Real Density | Fake Density |
| -------------- | ------------ | ------------ |
| 0.00 - 0.01    | 50           | 5            |
| 0.01 - 0.02    | 40           | 10           |
| 0.02 - 0.03    | 30           | 15           |
| 0.03 - 0.04    | 20           | 20           |
| 0.04 - 0.05    | 10           | 25           |
| 0.05 - 0.06    | 5            | 30           |
| 0.06 - 0.07    | 3            | 25           |
| 0.07 - 0.08    | 2            | 20           |
| 0.08 - 0.09    | 1            | 15           |
| 0.09 - 0.10    | 1            | 10           |
| 0.10 - 0.11    | 1            | 5            |
| 0.11 - 0.12    | 1            | 3            |
| 0.12 - 0.13    | 1            | 2            |
| 0.13 - 0.14    | 1            | 1            |
| 0.14 - 0.15    | 1            | 1            |
| 0.15 - 0.16    | 1            | 1            |
| 0.16 - 0.17    | 1            | 1            |
| 0.17 - 0.18    | 1            | 1            |
| 0.18 - 0.19    | 1            | 1            |
| 0.19 - 0.20    | 1            | 1            |
| 0.20 - 0.21    | 1            | 1            |
| 0.21 - 0.22    | 1            | 1            |
| 0.22 - 0.23    | 1            | 1            |
| 0.23 - 0.24    | 1            | 1            |
| 0.24 - 0.25    | 1            | 1            |
</details>

![](images/0d8f208f4f126cebd05b7c312f095385acd800418f8e719d3fb428332226a96d.jpg)

<details>
<summary>other</summary>

| Distance Range | Real Density | Fake Density |
| -------------- | ------------ | ------------ |
| 0.00 - 0.01    | 55           | 5            |
| 0.01 - 0.02    | 30           | 5            |
| 0.02 - 0.03    | 15           | 5            |
| 0.03 - 0.04    | 5            | 5            |
| 0.04 - 0.05    | 5            | 5            |
| 0.05 - 0.06    | 5            | 10           |
| 0.06 - 0.07    | 5            | 15           |
| 0.07 - 0.08    | 5            | 10           |
| 0.08 - 0.09    | 5            | 5            |
| 0.09 - 0.10    | 5            | 5            |
| 0.10 - 0.11    | 5            | 5            |
| 0.11 - 0.12    | 5            | 5            |
| 0.12 - 0.13    | 5            | 5            |
| 0.13 - 0.14    | 5            | 5            |
| 0.14 - 0.15    | 5            | 5            |
| 0.15 - 0.16    | 5            | 5            |
| 0.16 - 0.17    | 5            | 5            |
| 0.17 - 0.18    | 5            | 5            |
| 0.18 - 0.19    | 5            | 5            |
| 0.19 - 0.20    | 5            | 5            |
| 0.20 - 0.21    | 5            | 5            |
| 0.21 - 0.22    | 5            | 5            |
| 0.22 - 0.23    | 5            | 5            |
| 0.23 - 0.24    | 5            | 5            |
| 0.24 - 0.25    | 5            | 5            |
| 0.25 - 0.26    | 5            | 5            |
| 0.26 - 0.27    | 5            | 5            |
| 0.27 - 0.28    | 5            | 5            |
| 0.28 - 0.29    | 5            | 5            |
| 0.29 - 0.30    | 5            | 5            |
| 0.30 - 0.31    | 5            | 5            |
| 0.31 - 0.32    | 5            | 5            |
| 0.32 - 0.33    | 5            | 5            |
| 0.33 - 0.34    | 5            | 5            |
| 0.34 - 0.35    | 5            | 5            |
| 0.35 - 0.36    | 5            | 5            |
| 0.36 - 0.37    | 5            | 5            |
| 0.37 - 0.38    | 5            | 5            |
| 0.38 - 0.39    | 5            | 5            |
| 0.39 - 0.40    | 5            | 5            |
| 0.40 - 0.41    | 5            | 5            |
| 0.41 - 0.42    | 5            | 5            |
| 0.42 - 0.43    | 5            | 5            |
| 0.43 - 0.44    | 5            | 5            |
| 0.44 - 0.45    | 5            | 5            |
| 0.45 - 0.46    | 5            | 5            |
| 0.46 - 0.47    | 5            | 5            |
| 0.47 - 0.48    | 5            | 5            |
| 0.48 - 0.49    | 5            | 5            |
| 0.49 - 0.50    | 5            | 5            |
| Note: The density values are not explicitly provided in the code, so they are estimated based on the code's actual output from the 'Real' and 'Fake' arrays in this visualization.
</details>

Figure 4: Histogram of feature divergence on FFpp, Celeb-DF, Wild-Deepfake, and DiffSwap.

![](images/231e482f6505fa7c87a719c47374b1fb9e73d55e0b5d318c2f3254f953b47c3b.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (various) | (various) | Red |
| (various) | (various) | Blue |
</details>

Celeb-DF

![](images/84afc55a4b2cf89a5d0688d4e0587486ea929afe6c0493d979440f9a9987fecf.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (approximate) | (approximate) | Red |
| (approximate) | (approximate) | Blue |
</details>

Celeb-DF

![](images/1d282d59edbc33836bcd798a2ce26d57307feeb78dce11c9580e3d553c50d48c.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (various) | (various) | Red |
| (various) | (various) | Blue |
</details>

Wild-Deepfake

![](images/13ea16055553f8b238bd5cabb8e214ae38f27b300fe910b31a511cbb099272bb.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (approximate) | (approximate) | Red |
| (approximate) | (approximate) | Blue |
</details>

Wild-Deepfake   
Figure 5: Feature distribution of En-b4 model and the En-b4 model trained with our DiffusionFace on two unseen datasets Celeb-DF and Wild-Deepfake via t-SNE. The red represents the real samples while the blue represents the fake ones.

and source images, while real images exhibit smaller differences. The Euclidean distances at the bottom quantify these differences, indicating larger differences in fake images compared to real ones.

Analysis of feature divergence. DiffusionFake utilizes two Feature Filter modules to separate source-related and target-related features, expecting significant divergence between $f_{s}$ and $f_{t}$ for forged images and minimal differences for genuine faces. To validate this, we visualize the Euclidean distance distribution between these features across various datasets, including FFpp, Celeb-DF, Wild-Deepfake, and DiffSwap, as shown in Figure 4. The plots clearly distinguish real from fake samples: real samples have small feature distances, mostly within 0.05, whereas fake samples show larger distances due to mixed source and target information. These observations strongly support that DiffusionFake effectively disentangles source and target information in the extracted features.

Analysis of feature distribution. To demonstrate that DiffusionFake enhances the discriminative power and generalization ability of the extracted features, we visualize the t-SNE plots of the last layer features from two encoders: the original EfficientNet-B4 (En-B4) and En-B4 trained with DiffusionFake. The feature distributions are examined on two unseen datasets, Celeb-DF and Wild-Deepfake. As illustrated in Figure 5, the original En-B4 exhibits poor generalization on both datasets, with the real and fake features being nearly inseparable. In contrast, when trained with DiffusionFake, the encoder learns to capture the generalizable hybrid features present in forged images via the reverse process. Consequently, the real and fake features become more distinctly separated, forming clear decision boundaries on both unseen datasets.

# 5 Conclusion

In this paper, we introduce DiffusionFake, a novel framework that leverages the generative process of face forgery to enhance the generalization capabilities of detection models. DiffusionFake inverts this generative process to extract and utilize hybrid features from source and target identities for effective forgery detection. Extensive experiments demonstrate that DiffusionFake significantly improves the generalization performance of various detector architectures without increasing inference parameters. The proposed framework enables the learning of discriminative and generalizable features, enhancing the robustness of detectors against a wide range of unseen forgeries.

# References

[1] Weiming Bai, Yufan Liu, Zhipeng Zhang, Bing Li, and Weiming Hu. Aunet: Learning relations between action units for face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24709–24719, 2023.   
[2] Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang Yang. End-to-end reconstruction-classification learning for face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4113–4122, 2022.   
[3] Liang Chen, Yong Zhang, Yibing Song, Lingqiao Liu, and Jue Wang. Self-supervised learning of adversarial example: Towards good generalizations for deepfake detection. In CVPR, pages 18710–18719, 2022.   
[4] Shen Chen, Taiping Yao, Yang Chen, Shouhong Ding, Jilin Li, and Rongrong Ji. Local relation learning for face forgery detection. AAAI, 2021.   
[5] Zhikai Chen, Lingxi Xie, Shanmin Pang, Yong He, and Bo Zhang. Magdr: Mask-guided detection and reconstruction for defending deepfakes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9014–9023, 2021.   
[6] Zhongxi Chen, Ke Sun, Ziyin Zhou, Xianming Lin, Xiaoshuai Sun, Liujuan Cao, and Rongrong Ji. Diffusionface: Towards a comprehensive dataset for diffusion-based face forgery analysis. arXiv preprint arXiv:2403.18471, 2024.   
[7] François Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, pages 1251-1258, 2017.   
[8] Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jikuo Lu, Russ Howes, Menglin Wang, and Cristian Canton Ferrer. The deepfake detection challenge dataset. arXiv preprint arXiv:2006.07397, 2020.   
[9] Shichao Dong, Jin Wang, Renhe Ji, Jiajun Liang, Haoqiang Fan, and Zheng Ge. Implicit identity leakage: The stumbling block to improving deepfake detection generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3994–4004, 2023.   
[10] Shichao Dong, Jin Wang, Jiajun Liang, Haoqiang Fan, and Renhe Ji. Explaining deepfake detection by analysing image matching. In European Conference on Computer Vision, pages 18–35. Springer, 2022.   
[11] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Ting Zhang, Weiming Zhang, Nenghai Yu, Dong Chen, Fang Wen, and Baining Guo. Protecting celebrities from deepfake with identity consistency transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9468–9478, 2022.   
[12] Zhihao Gu, Taiping Yao, Yang Chen, Shouhong Ding, and Lizhuang Ma. Hierarchical contrastive inconsistency learning for deepfake video detection. In European Conference on Computer Vision, pages 596–613. Springer, 2022.   
[13] Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, and Luisa Verdoliva. Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20606–20615, 2023.   
[14] Ying Guo, Cheng Zhen, and Pengfei Yan. Controllable guide-space for generalizable face forgery detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20818–20827, 2023.   
[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
[16] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In CVPR, pages 7132-7141, 2018.

[17] Baojin Huang, Zhongyuan Wang, Jifan Yang, Jiaxin Ai, Qin Zou, Qian Wang, and Dengpan Ye. Implicit identity driven deepfake face swapping detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4490–4499, 2023.   
[18] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. International Conference on Learning Representations, 2014.   
[19] Nicolas Larue, Ngoc-Son Vu, Vitomir Struc, Peter Peer, and Vassilis Christophides. Seeable: Soft discrepancies and bounded contrastive learning for exposing deepfakes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21011–21021, 2023.   
[20] Binh M Le and Simon S Woo. Quality-agnostic deepfake detection with intra-model collaborative learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22378–22389, 2023.   
[21] Jiaming Li, Hongtao Xie, Jiahong Li, Zhongyuan Wang, and Yongdong Zhang. Frequency-aware discriminative feature learning supervised by single-center loss for face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6458–6467, 2021.   
[22] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, and Baining Guo. Face x-ray for more general face forgery detection. In CVPR, pages 5001–5010, 2020.   
[23] Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and Siwei Lyu. Celeb-df: A new dataset for deepfake forensics. arXiv preprint arXiv:1909.12962, 2019.   
[24] Jiahao Liang, Huafeng Shi, and Weihong Deng. Exploring disentangled content information for face forgery detection. In European Conference on Computer Vision, pages 128–145. Springer, 2022.   
[25] Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen, Yuan He, Hui Xue, Weiming Zhang, and Nenghai Yu. Spatial-phase shallow learning: rethinking face forgery detection in frequency domain. In CVPR, pages 772–781, 2021.   
[26] Anwei Luo, Chenqi Kong, Jiwu Huang, Yongjian Hu, Xiangui Kang, and Alex C Kot. Beyond the prior forgery knowledge: Mining critical clues for general face forgery detection. IEEE Transactions on Information Forensics and Security, 19:1168–1182, 2023.   
[27] Anwei Luo, Enlei Li, Yongliang Liu, Xiangui Kang, and Z Jane Wang. A capsule network based approach for detection of audio spoofing attacks. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6359–6363. IEEE, 2021.   
[28] Yuchen Luo, Yong Zhang, Junchi Yan, and Wei Liu. Generalizing face forgery detection with high-frequency features. In CVPR, pages 16317-16326, 2021.   
[29] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In ECCV, pages 86–103. Springer, 2020.   
[30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
[31] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. Faceforensics++: Learning to detect manipulated facial images. In ICCV, pages 1–11, 2019.   
[32] Rui Shao, Tianxing Wu, and Ziwei Liu. Detecting and recovering sequential deepfake manipulation. In European Conference on Computer Vision, pages 712–728. Springer, 2022.   
[33] Rui Shao, Tianxing Wu, and Ziwei Liu. Detecting and grounding multi-modal media manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6904–6913, 2023.   
[34] Liang Shi, Jie Zhang, and Shiguang Shan. Real face foundation representation learning for generalized deepfake detection. arXiv preprint arXiv:2303.08439, 2023.   
[35] Kaede Shiohara and Toshihiko Yamasaki. Detecting deepfakes with self-blended images. In CVPR, pages 18720-18729, 2022.   
[36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020.

[37] Luchuan Song, Zheng Fang, Xiaodan Li, Xiaoyi Dong, Zhenchao Jin, Yuefeng Chen, and Siwei Lyu. Adaptive face forgery detection in cross domain. In European Conference on Computer Vision, pages 467–484. Springer, 2022.   
[38] Ke Sun, Shen Chen, Taiping Yao, Xiaoshuai Sun, Shouhong Ding, and Rongrong Ji. Towards general visual-linguistic face forgery detection. arXiv preprint arXiv:2307.16545, 2023.   
[39] Ke Sun, Hong Liu, Taiping Yao, Xiaoshuai Sun, Shen Chen, Shouhong Ding, and Rongrong Ji. An information theoretic approach for attention-driven face forgery detection. In European Conference on Computer Vision, pages 111–127. Springer, 2022.   
[40] Ke Sun, Hong Liu, Qixiang Ye, Jianzhuang Liu, Yue Gao, Ling Shao, and Rongrong Ji. Domain general face forgery detection by learning to weight. In AAAI, volume 35, pages 2638–2646, 2021.   
[41] Ke Sun, Taiping Yao, Shen Chen, Shouhong Ding, Rongrong Ji, et al. Dual contrastive learning for general face forgery detection. In AAAI, 2022.   
[42] Mingxing Tan and Quoc V Le. Efficientnet: Rethinking model scaling for convolutional neural networks. ICML, 2019.   
[43] Ruben Tolosana, Ruben Vera-Rodriguez, Julian Fierrez, Aythami Morales, and Javier Ortega-Garcia. Deepfakes and beyond: A survey of face manipulation and fake detection. arXiv preprint arXiv:2001.00179, 2020.   
[44] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[45] Yuan Wang, Kun Yu, Chen Chen, Xiyuan Hu, and Silong Peng. Dynamic graph learning with content-guided spatial-frequency relation reasoning for deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7278–7287, 2023.   
[46] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. Cbam: Convolutional block attention module. In ECCV, pages 3–19, 2018.   
[47] Yuting Xu, Jian Liang, Gengyun Jia, Ziming Yang, Yanhao Zhang, and Ran He. Tall: Thumbnail layout for deepfake video detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22658–22668, 2023.   
[48] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. arXiv preprint arXiv:2304.13949, 2023.   
[49] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.   
[50] Hanqing Zhao, Wenbo Zhou, Dongdong Chen, Tianyi Wei, Weiming Zhang, and Nenghai Yu. Multi-attentional deepfake detection. CVPR, 2021.   
[51] Tianchen Zhao, Xiang Xu, Mingze Xu, Hui Ding, Yuanjun Xiong, and Wei Xia. Learning self-consistency for deepfake detection. In CVPR, pages 15023-15033, 2021.   
[52] Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie Zhou, and Jiwen Lu. Diffswap: High-fidelity and controllable face swapping via 3d-aware masked diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8568–8577, 2023.   
[53] Wanyi Zhuang, Qi Chu, Zhentao Tan, Qiankun Liu, Haojie Yuan, Changtao Miao, Zixiang Luo, and Nenghai Yu. Uia-vit: Unsupervised inconsistency-aware method based on vision transformer for face forgery detection. ECCV, 2022.   
[54] Bojia Zi, Minghao Chang, Jingjing Chen, Xingjun Ma, and Yu-Gang Jiang. Wilddeepfake: A challenging real-world dataset for deepfake detection. In ACM MM, pages 2382-2390, 2020.

# A Appendix

# A.1 Cross-manipulation evaluation.

Cross-manipulation evaluation. To further validate the generalization ability across different manipulation methods, we conduct a cross-manipulation evaluation. We train models on a single manipulation method within the high-quality FF++ dataset and test them on all four methods. Using EfficientNet-B4 (En-B4) as the backbone, we compare our approach with the MAT method, which employs attention mechanisms to enhance the generalization ability of En-B4. Table 4 shows that DiffusionFake improves generalization performance across all manipulation methods. Notably, when trained on the FaceSwap method and tested on the Deepfake method, our approach outperforms the original En-B4 by 6%. Moreover, compared to the MAT method, DiffusionFake achieves a 4% improvement in generalization when trained on NeuralTextures and tested on FaceSwap. These results demonstrate the effectiveness of DiffusionFake in learning generalizable features that can be applied to unseen manipulation methods.

We also evaluate the multi-source generalization performance by training on three forgery methods and testing on the unknown method. Additionally, we assess the performance under low-quality (LQ) training conditions. As reported in Table 5, DiffusionFake achieves state-of-the-art results across all protocols and quality levels. Specifically, integrating our method with En-B4 improves generalization by approximately 8% compared to the backbone alone. Even under low-quality training conditions, DiffusionFake maintains a 7% performance gain, demonstrating the robustness and generalization capability of our proposed framework.

Table 4: Cross-manipulation evaluation in terms Table 5: Multi-source manipulation evaluation of AUC. Diagonal results indicate the intra-in terms of ACC, which follows [40]. H means domain performance. high-quality image (c23) in FFpp, while L represents low-quality (c40). 

<table><tr><td>Train</td><td>Method</td><td>DF</td><td>F2F</td><td>FS</td><td>NT</td></tr><tr><td rowspan="3">DF</td><td>EN-b4</td><td>99.97</td><td>76.32</td><td>46.24</td><td>72.72</td></tr><tr><td>MAT</td><td>99.91</td><td>78.23</td><td>40.61</td><td>71.08</td></tr><tr><td>Ours</td><td>99.82</td><td>78.46</td><td>52.29</td><td>74.43</td></tr><tr><td rowspan="3">F2F</td><td>EN-b4</td><td>84.52</td><td>99.20</td><td>58.14</td><td>63.71</td></tr><tr><td>MAT</td><td>86.15</td><td>99.13</td><td>60.14</td><td>64.59</td></tr><tr><td>Ours</td><td>88.92</td><td>99.36</td><td>63.19</td><td>68.55</td></tr><tr><td rowspan="3">FS</td><td>EN-b4</td><td>69.25</td><td>67.69</td><td>99.89</td><td>48.61</td></tr><tr><td>MAT</td><td>64.13</td><td>66.39</td><td>99.67</td><td>50.10</td></tr><tr><td>Ours</td><td>75.28</td><td>70.91</td><td>99.12</td><td>52.17</td></tr><tr><td rowspan="3">NT</td><td>EN-b4</td><td>85.99</td><td>48.86</td><td>73.05</td><td>98.25</td></tr><tr><td>MAT</td><td>87.23</td><td>48.22</td><td>75.33</td><td>98.66</td></tr><tr><td>Ours</td><td>89.54</td><td>51.71</td><td>79.15</td><td>98.71</td></tr></table>

<table><tr><td>Method</td><td>DF (H)</td><td>DF (L)</td><td>F2F(H)</td><td>F2F(L)</td></tr><tr><td>Xception</td><td>78.25</td><td>68.12</td><td>61.53</td><td>59.58</td></tr><tr><td>EN-B4</td><td>82.40</td><td>67.60</td><td>63.32</td><td>61.41</td></tr><tr><td>VIT-B</td><td>81.15</td><td>73.38</td><td>62.19</td><td>61.93</td></tr><tr><td>Multi-task</td><td>70.30</td><td>66.76</td><td>58.74</td><td>56.50</td></tr><tr><td>MLDG</td><td>84.21</td><td>67.15</td><td>63.46</td><td>58.12</td></tr><tr><td>LTW</td><td>85.60</td><td>69.15</td><td>65.60</td><td>65.70</td></tr><tr><td>DCL</td><td>87.70</td><td>75.90</td><td>68.40</td><td>67.85</td></tr><tr><td>RECCE</td><td>86.69</td><td>75.89</td><td>62.71</td><td>68.02</td></tr><tr><td>MAT</td><td>84.40</td><td>73.71</td><td>66.28</td><td>66.39</td></tr><tr><td>UCF</td><td>86.70</td><td>74.59</td><td>67.87</td><td>67.33</td></tr><tr><td>En-b4+Ours</td><td>88.17</td><td>75.13</td><td>70.17</td><td>71.25</td></tr><tr><td>VIT-b+Ours</td><td>87.23</td><td>77.33</td><td>68.93</td><td>68.75</td></tr></table>

# A.2 Visualizations of CAM result.

To further illustrate the ability of our method to accurately focus on relevant locations in generalized images, we visualize the Class Activation Mapping (CAM) results of both our approach and the vanilla encoder across different datasets. As shown in Figure 6, the conventional EfficientNet-B4 (En-B4) encoder often fails to highlight key areas, such as the blurred mouth region in Celeb-DF images. This limitation can reduce the effectiveness of forgery detection. In contrast, our method demonstrates a broader focus during training, targeting significantly larger regions that may include latent forgery areas. This comprehensive attention to detail contributes to enhancing the generalization performance of the detection model. By effectively identifying and concentrating on these critical regions, our method provides a more robust defense against sophisticated forgery techniques, ultimately leading to more accurate and reliable detection outcomes across diverse datasets.

# A.3 Visualizations of weight module.

Figure 7 presents the source and target scores computed by our Weight Module for four different attack types. It is evident that the target scores are generally higher than the source scores, indicating

![](images/ce2b9bba7fbc2221075f30ce483d08b387c944305325ff5f914285748dc74285.jpg)

<details>
<summary>text_image</summary>

Celeb-DF
En-B4
En-B4+Ours
WDF
En-B4
En-B4+Ours
DiffSwap
En-B4
En-B4+Ours
</details>

Figure 6: CAM maps of the baseline model (EN-b4) and En-b4 trained with DiffusionFake method on three unseen datasets: Celeb-DF, WDF (Wild-Deepfake), and DiffSwap.   
![](images/65372824615d0987ab2f94618fd635876617e8ba15d6adb9c643884d981a12e4.jpg)

<details>
<summary>line</summary>

| Model          | Accuracy |
| -------------- | -------- |
| DeepFakes      | 0.90     |
| Face2Face      | 0.97     |
| FaceSwap       | 0.92     |
| NeuralTextures | 0.97     |
| Source         | 0.62     |
</details>

Figure 7: Visualization of weights for different attack types. The blue lines connect the target weights, while the red lines connect the source weights.

that reconstructing the target information contributes more significantly to the overall reconstruction process, while the reconstruction of the source image relies more heavily on the pre-trained knowledge. Moreover, the scores vary across different attack types. For samples that are more similar to the target, such as NeuralTextures and Face2Face, the corresponding target scores are higher (greater than 0.95) due to the high proportion of target features they contain, while the source scores are lower due to the limited presence of source features. On the other hand, for Deepfakes and FaceSwap, which involve replacing the source's facial region onto the target, the proportion of source information is relatively higher, resulting in slightly elevated source scores compared to other attack types.

# A.4 Evaluation Metric

We use two common metrics to evaluate the performance of our forgery detection method: the Area Under the Receiver Operating Characteristic Curve (AUC) and the Equal Error Rate (EER).

The AUC is a widely adopted metric that measures the overall performance of a binary classifier across all possible decision thresholds. It represents the probability that a randomly chosen positive instance (i.e., a forged image) will be ranked higher than a randomly chosen negative instance (i.e., a real image). The EER is another commonly used metric that represents the point on the ROC curve where the False Positive Rate (FPR) and the False Negative Rate (FNR) are equal.

In summary, we use AUC and EER as our primary evaluation metrics, where a higher AUC and a lower EER indicate better forgery detection performance. These metrics provide a comprehensive assessment of the classifier's ability to distinguish between forged and genuine images across various decision thresholds.

# A.5 Limitations and Broader Impacts.

Limitation: The framework relies on paired source and target images for training, which may not always be feasible in real-world scenarios. We aim to integrate self-supervised methods to generate these images in the future. Additionally, the effectiveness of DiffusionFake against more sophisticated forgery techniques, such as those involving multiple source identities or partial manipulations, requires further investigation.

Broader Impacts: Our method could potentially be used as an adversarial discriminator to create more difficult-to-detect images. Future research needs to address how to prevent this misuse.