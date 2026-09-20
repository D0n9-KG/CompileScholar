# CognitionCapturer: Decoding Visual Stimuli From Human EEG Signal With Multimodal Information

Kaifan Zhang $^{1}$ , Lihuo He $^{1*}$ Xin Jiang $^{1}$ , Wen Lu $^{1}$ Di Wang $^{1}$ Xinbo Gao $^{1,2}$

$^{1}$ School of Electronic Engineering, Xidian University, Xi'an, China $^{2}$ Chongqing University of Posts and Telecommunications, Chongqing, China

# Abstract

Electroencephalogram (EEG) signals have attracted significant attention from researchers due to their non-invasive nature and high temporal sensitivity in decoding visual stimuli. However, most recent studies have focused solely on the relationship between EEG and image data pairs, neglecting the valuable “beyond-image-modality” information embedded in EEG signals. This results in the loss of critical multimodal information in EEG. To address this limitation, we propose CognitionCapturer, a unified framework that fully leverages multimodal data to represent EEG signals. Specifically, CognitionCapturer trains Modality Expert Encoders for each modality to extract cross-modal information from the EEG modality. Then, it introduces a diffusion prior to map the EEG embedding space to the CLIP embedding space, followed by using a pretrained generative model, the proposed framework can reconstruct visual stimuli with high semantic and structural fidelity. Notably, the framework does not require any fine-tuning of the generative models and can be extended to incorporate more modalities. Through extensive experiments, we demonstrate that CognitionCapturer outperforms state-of-the-art methods both qualitatively and quantitatively. Code: https://github.com/XiaoZhangYES/CognitionCapturer.

# Introduction

Since its inception, a fundamental challenge in brain decoding is optimally expressing the meaningful information within brain signals. Reconstructing visual stimuli from brain signals is one of the interesting tasks with exciting application prospects. Initially, pioneering work using fMRI data (Kay et al. 2008; Miyawaki et al. 2008; Naselaris et al. 2009) validated the possibility of reconstructing visual stimuli from fMRI data and successfully decoded simple textures and shapes. More recently, with the rapid development of deep learning methods, the use of deep learning models to decode fMRI brain signals has produced significant advancements (Ren et al. 2021; Takagi and Nishimoto 2023; Scotti et al. 2024).

However, brain signals exhibit diverse forms, among which EEG and MEG data offer high temporal resolution and portability, making them particularly suitable for real-time decoding compared to fMRI. This versatility has led to a broader range of downstream applications. Recent works (Benchetrit, Banville, and King 2024; Song et al. 2024; Li et al. 2024) have attempted to align the brain-image modalities using EEG and MEG signals through contrastive learning. These approaches have achieved notable accuracy in decoding related visual stimuli.

![](images/8dcc06599b82248e871ccd37811cec5ac0830732fbba66ec8ffed737c0d93c91.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Visual Stimulus"] --> B["Reconstructed Stimuli"]
    B --> C["Brain space"]
    B --> D["Image space"]
    C --> E["Visual Stimulus"]
    D --> F["Reconstructed Stimuli"]
    E --> G["Animal"]
    E --> H["Bone"]
    E --> I["Mouse"]
    E --> J["Bone"]
    E --> K["Mouse"]
    E --> L["Bone"]
    E --> M["Mouse"]
    E --> N["Bone"]
    E --> O["Mouse"]
    E --> P["Bone"]
    E --> Q["Mouse"]
    E --> R["Bone"]
    E --> S["Mouse"]
    E --> T["Bone"]
    E --> U["Bone"]
    E --> V["Bone"]
    E --> W["Bone"]
    E --> X["Bone"]
    E --> Y["Bone"]
    E --> Z["Bone"]
    E --> AA["Bone"]
    E --> AB["Bone"]
    E --> AC["Bone"]
    E --> AD["Bone"]
    E --> AE["Bone"]
    E --> AF["Bone"]
    E --> AG["Bone"]
    E --> AH["Bone"]
    E --> AI["Bone"]
    E --> AJ["Bone"]
    E --> AK["Bone"]
    E --> AL["Bone"]
    E --> AM["Bone"]
    E --> AN["Bone"]
    E --> AO["Bone"]
    E --> AP["Bone"]
    E --> AQ["Bone"]
    E --> AR["Bone"]
    E --> AS["Bone"]
    E --> AT["Bone"]
    E --> AU["Bone"]
    E --> AV["Bone"]
    E --> AW["Bone"]
    E --> AX["Bone"]
    E --> AY["Bone"]
    E --> AZ["Bone"]
    E --> BA["Bone"]
    E --> BB["Bone"]
    E --> BC["Bone"]
    E --> BD["Bone"]
    E --> BE["Bone"]
    E --> BF["Bone"]
    E --> BG["Bone"]
    E --> BH["Bone"]
    E --> BI["Bone"]
    E --> BJ["Bone"]
    E --> BK["Bone"]
    E --> BL["Bone"]
    E --> BM["Bone"]
    E --> BN["Bone"]
    E --> BO["Bone"]
    E --> BP["Bone"]
    E --> BQ["Bone"]
    E --> BR["Bone"]
    E --> BS["Bone"]
    E --> BT["Bone"]
    E --> BU["Bone"]
    E --> BV["Bone"]
    E --> BW["Bone"]
    E --> BX["Bone"]
    E --> BY["Bone"]
    E --> BZ["Bone"]
    E --> CAB["Bone"]
    E --> CBB["Bone"]
    E --> CCB["Bone"]
    E --> CDB["Bone"]
    E --> CEB["Bone"]
    E --> CFB["Bone"]
    E --> CGB["Bone"]
    E --> CHB["Bone"]
    E --> CIB["Bone"]
    E --> CJB["Bone"]
    E --> CKB["Bone"]
    E --> CLB["Bone"]
    E --> CMB["Bone"]
    E --> CNB["Bone"]
    E --> COB["Bone"]
    E --> CPB["Bone"]
    E --> CQB["Bone"]
    E --> CRB["Bone"]
    E --> CSB["Bone"]
    E --> CTB["Bone"]
    E --> CUB["Bone"]
    E --> CVB["Bone"]
    E --> CWB["Bone"]
    E --> CXB["Bone"]
    E --> CYB["Bone"]
    E --> CZB["Bone"]
```
</details>

Figure 1: We believe that for image-EEG pairs, relying solely on the mutual information between images and EEG signals can lead to underutilization of EEG information. To address this issue, we utilize multimodal information to capture meaningful information in the EEG signals. The dashed lines in the figure below illustrate some of our successful reconstruction results.

However, the internal mechanisms of brain function are diverse and complex. Human perception of visual stimuli is influenced by both the characteristics of the visual stimuli and individual past experiences (Lupyan et al. 2020; Du et al. 2023). Recent works (Benchetrit, Banville, and King 2024; Song et al. 2024; Li et al. 2024) have primarily relied on the image modality as a reference for alignment, enabling the decoding of meaningful visual stimuli. Nonetheless, the objective of contrastive learning may lead to models that predominantly focus on the shared information between modalities, potentially overlooking the more diverse and complex “beyond-image-modality” information present in the brain signals.

To address this issue, we introduce a novel brain decod-

ing model named CognitionCapturer, as illustrated in Fig. 1. CognitionCapturer can be trained jointly with brain signals and multiple modalities, effectively capturing the shared information between brain signals and a broader spectrum of modalities.

Specifically, based on the understanding that brain data contains information “beyond-image-modality”, we first extend image data using depth estimation models and image captioning models to construct a Image-Text-Depth multimodal aligned dataset. Then introduce Modality Expert Encoders, which focus on different EEG - single modality data. The embeddings obtained in this stage can be directly used for downstream tasks such as classification and retrieval. Subsequently, in the generation phase, we map the EEG embeddings to the CLIP image space via a diffusion prior and feed EEG embeddings associated with different modalities into a pre-trained image generation model, thus decoding fine-grained visual stimuli.

In contrast to previous methods, CognitionCapturer's training strategy enables models for different modalities to focus on capturing the relationships between information in EEG signals and modality-specific characteristics. This allows the model to capture fine-grained low-level visual information and abstract high-level semantic information. Furthermore, our proposed approach inherently possesses scalability, enabling the Modality Expert Encoder to be extended infinitely to any modality.

Another advantage of the proposed method is that the constructed dataset effectively decouples certain image features, allowing different Modality Expert Encoders to focus on structural and semantic features during training, thereby preventing fine-grained information from being overshadowed by coarse-grained information. The main contributions are as follows:

# Main Contribution

- We propose CognitionCapturer, a contrastive learning-based model that effectively decodes brain signals from multiple modalities.   
- Using an alignment module and a pre-trained image generation model without any fine-tuning, we achieve fine-grained reconstruction of images with performance surpassing that of any single modality.   
- Through experiments, we validate the effectiveness and rationality of incorporating more modal information for brain signal decoding, providing new insights for subsequent research in neuroscience.

# Related Work

# Decode Visual Stimuli from Brain Signal

Decoding visual stimuli from fMRI brain signals has been widely studied and yielded successful results (Gu et al. 2024; Takagi and Nishimoto 2023; Scotti et al. 2024; Miyawaki et al. 2008; Kay et al. 2008). However, the difficulty of acquiring fMRI data and its low temporal resolution pose challenges for practical applications. In contrast, EEG signals offer higher temporal resolution and lower acquisition costs, leading researchers to attempt decoding visual stimuli from EEG. Early EEG decoding work typically relied on supervised learning methods and was limited to a finite set of image categories, overlooking the intrinsic relationship between visual stimuli and brain responses (Li et al. 2020; Liu et al. 2023a). Recently, (Song et al. 2024; Scotti et al. 2024) successfully constructed an image decoding framework using a contrastive learning approach, achieving zero-shot recognition. (Li et al. 2024) built upon song's work (Song et al. 2024) by further reconstructing decoded visual information into high-quality images using a diffusion model. However, these works only considered EEG-image modality pairs, neglecting the diversity of brain data. Compared with their approaches, our method successfully leverages multiple modalities of data to decode visual stimuli, resulting in superior performance.

# Contrastive Learning for Brain Decoding

Contrastive learning, as an effective cross-modal learning approach, has achieved significant success in works such as CLIP, Moco, etc. (Radford et al. 2021; He et al. 2020), However, its effectiveness is closely related to the quality and scale of the data, and the selection of high-quality samples is crucial for improving model performance (Cherti et al. 2023). Works that use contrastive learning to decode brain signals have also shown promising results. For instance, as a representative work, (Défossez et al. 2023) utilizes a pre-trained speech encoder to decode speech from MEG signals through contrastive learning, and subsequently, (Benchetrit, Banville, and King 2024) adopts a similar idea to decode images from MEG. A series of similar methods emerged subsequently (Song et al. 2024; Liu et al. 2023b; Li et al. 2024). However, during the process of using brain data for contrastive learning, the limited amount of brain signal data may lead the model to focus only on the most discriminative features. After transforming image modality into other modalities, since these modalities are less information-rich compared to image modality, this forces our model to attend to finer-grained features, thereby better representing EEG signals.

# Method

CognitionCapturer aims to address the loss of “beyond-image-modality” information in brain decoding. The method overview is depicted in Fig 2, where EEG-Modality pairs $^{1}$ are processed by dedicated Modality Expert Encoders to decouple the effective information from different modalities in the EEG signal. In our experiments, we observed that binding the brain modality with different modalities improves classification and reconstruction performance. Subsequently, through a diffusion prior, the EEG embedding space is mapped to the CLIP space and fed into assembled SDXL-turbo and IP-Adapters to reconstruct visual stimuli.

![](images/0f07a7401ff0d3b0235e7906e19db92736172df75ae34bafdd8be0bb1ba0d597.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1: CONTRASTIVE LEARNING"] --> B["A basketball ball on the court"]
    B --> C["CLIP Image Encoder"]
    C --> D["Project Layer"]
    B --> E["CLIP Text Encoder"]
    E --> F["Project Layer"]
    B --> G["CLIP Depth Encoder"]
    G --> H["Project Layer"]
    B --> I["InfoNCE loss"]
    I --> J["EEG\Image Embedding"]
    I --> K["EEG\Text Embedding"]
    I --> L["EEG\Depth Embedding"]
    M["2: MAP TO CLIP SPACE"] --> N["Diffusion Prior"]
    N --> O["Noise"]
    N --> P["Noise"]
    N --> Q["EEG\Image Embedding"]
    N --> R["EEG\Text Embedding"]
    N --> S["EEG\Depth Embedding"]
    T["3: GENERATION"] --> U["IP-Adapter (Layout+Style)"]
    U --> V["Image Embedding"]
    U --> W["Text Embedding"]
    U --> X["Depth Embedding"]
    Y["Stable Diffusion XL Turbo"] --> Z["IP-Adapter (Style)"]
    Y --> AA["IP-Adapter (Layout)"]
    Y --> AB["Image Embedding"]
    Y --> AC["Text Embedding"]
    Y --> AD["Depth Embedding"]
    AE["Trainable Frozen"] --> AF["EEG/Image Encoder"]
    AE --> AG["EEG/Text Encoder"]
    AE --> AH["EEG/Depth Encoder"]
```
</details>

Figure 2: Overall framework of CognitionCapturer. 1: In the contrastive learning stage, different EEG-Modality data pairs are fed into different Modality Expert Encoders for processing. The embeddings obtained from the contrastive learning stage can be used for various downstream tasks. 2: To use pre-trained image generation models, we apply a Diffusion Prior model to map the EEG embeddings into CLIP space while retaining their original information. 3: Using pre-trained SDXL and IP-Adapters with different structures, we integrate the EEG embeddings from different modalities to reconstruct visual stimuli.

# Modality Expert Encoder

CognitionCapturer uses modality pairs $(E, M)$ , where E represents the EEG signal and M represents other modalities. For each modality pair $(E, M_{i})$ , where i represents the index of different modalities.

we construct a dedicated network $f_{i}$ and $g_{i}$ , which we refer to as Modality Expert Encoders. This way, each modality pair $(E, M_{i})$ is mapped to the same dimension by its corresponding Modality Expert Encoder for subsequent constraints.

In the encoding of the EEG data, raw EEG signals are typically represented as matrices $C \times T$ , where C denotes the number of electrode channels and T denotes the number of time samples. Analysis of EEG signals primarily occurs along these two dimensions.

Our EEG encoder, based on a lightweight Transformer and STConv architecture (Li et al. 2024; Vaswani et al.

2017), effectively extracts topological and spatiotemporal information from EEG channels. The network structure is shown in Table 1. Specifically, we first process the raw EEG data $e \in R^{C \times T}$ through a layer of Transformer encoder and a linear transformation to organize the topological information, then feed it into a feature extraction module based on STConv to extract spatiotemporal features. Finally, a residual linear layer maps the features output by STConv to the same dimension as the target modality features. Detailed network descriptions are provided in the appendix.

When extracting features for the target modality $M_{i}$ paired with EEG data E, there are many successful pretrained encoders that can effectively extract img, text, and depth features. Recent work (Zhang et al. 2022) and our experiments indicate that CLIP image embeddings contain depth information. To be compatible with generative models and maintain distribution consistency initially, we used

the Open CLIP ViT-H/14 (Radford et al. 2021) as both the visual and text encoder, and added a residual linear layer with the same dimension as the original features to ensure stability during training.

<table><tr><td>Layer</td><td>Input Shape</td><td>Output Shape</td></tr><tr><td>Transformer Block</td><td> $(N,C,T)$ </td><td> $(N,C,T)$ </td></tr><tr><td>Linear</td><td> $(N,C,T)$ </td><td> $(N,C,T)$ </td></tr><tr><td>STConv</td><td> $(N,C,T)$ </td><td> $(N,C_1,T_1)$ </td></tr><tr><td>Project Layer</td><td> $(N,C_1,T_1)$ </td><td> $(N,D)$ </td></tr></table>

Table 1: Architecture of Modality Expert Encoder

# Align EEG-Modality Pairs by Contrastive Learning

After the modality pairs $(E, M_i)$ are processed by their respective Modality Expert Encoder $f_i$ and $g_i$ , they are encoded into the same dimension, resulting in embedding pairs $(e_i, m_i)$ . Here, $(e_i, m_i)$ represents a set consisting of $n$ samples, i.e., $e_i = \{q_1^i, q_2^i, ..., q_n^i\}$ , $m_i = \{k_1^i, k_2^i, ..., k_n^i\}$ .

Subsequently, for different $(e_{i}, m_{i})$ embedding pairs, we adopted an improved version of the infoNCE loss (van den Oord, Li, and Vinyals 2019) as the loss function:

$$
L _ {E, M _ {i}} = - \log \frac {L _ {+}}{L _ {+} + L _ {-}} \tag {1}
$$

$$
L _ {+} = \sum_ {P (i d x) = 1} \exp (q _ {i d x} ^ {T} k _ {i} / \tau) \tag {2}
$$

$$
L _ {-} = \sum_ {P (i d x) = 0} \exp (q _ {i d x} ^ {T} k _ {i} / \tau) \tag {3}
$$

$$
P (i d x) = \left\{ \begin{array}{l l} 1 & \text { when   } i d x \text {   is   the   same   as   image   label } \\ 0 & \text { otherwise } \end{array} \right. \tag {4}
$$

In equation (2) and (3), $\tau$ is a scalar temperature parameter that controls the smoothness of the softmax distribution. Given that the same image is repeatedly viewed in EEG experiments (Gifford et al. 2022), multiple EEG data may correspond to the same image. This can create a contradictory phenomenon where the same data pairs are both pulled closer and pushed apart by the loss function. To address this, we utilize image index as supervisory information. Specifically, when idx is the same in multiple EEG data, we choose to pull together all the EEG data and the corresponding modality data, thereby avoiding the contradictory phenomenon. In practice, we employ a symmetric loss $L_{E,M_{i}} + L_{M_{i},E}$ .

# Map EEG Embedding into CLIP Image Space

After obtaining the aligned embeddings $e_{i}$ for EEG and $m_{i}$ for other modalities, due to the existence of the modality gap and differences in distribution spaces (Scotti et al. 2024), directly using the EEG embedding $e_{i}$ would make it difficult for pre-trained generative models to identify effective information. Following the works of (Scotti et al. 2024; Li et al. 2024; Ramesh et al. 2022), we use a diffusion prior model to map the EEG embeddings $e_{i}$ to the CLIP space, thereby making the EEG embeddings recognizable by pretrained generative models. In practice, we used the MSE loss to train our diffusion prior from scratch.

$$
L _ {\text { prior }} = E _ {t \sim [ 1, T ], m _ {i} ^ {(t)} \sim q _ {t}} \left[ | | f _ {\theta} (m _ {i} ^ {(t)}, t, e _ {i}) - m _ {i} | | ^ {2} \right] \tag {5}
$$

In equation (5), $m_{i}^{(t)}$ represents the CLIP embedding disturbed after a given diffusion timestep t, and $f_{\theta}$ denotes the diffusion prior network. The specific training details are provided in the Implementation Details section and supplementary material.

# Generate visual stimulus with Multi-modal associated EEG embeddings

After the EEG embeddings pass through the diffusion prior, they can be used like the original CLIP embeddings. Specifically, to reconstruct high-fidelity visual stimuli and effectively utilize information from three modalities, we employ Multi IP-Adapters (Ye et al. 2023) and SDXL-turbo (Sauer et al. 2023) to simultaneously leverage embeddings from different modalities. As shown in Fig 2's generation phase, for the image modality, which contains the richest information, we use a full IP-Adapter to process the image embedding. For text and depth modalities, which focus on semantic and structural information respectively, we use modified versions of IP-Adapter, namely IP-Adapter-Style and IP-Adapter-Layout, to process the text and depth embeddings. This approach enables CognitionCapturer to reconstruct semantic information while preserving underlying visual details.

# Experimental Setup

# Datasets and Preprocessing

We utilized Thing-EEG Dataset for our experiments. The Thing-EEG dataset (Gifford et al. 2022) contains EEG data collected from 10 subjects under an RSVP paradigm. The training set comprises 1654 concepts, each associated with 10 images presented four times, resulting in a total of 66,160 EEG recordings. The test set includes 200 unique concepts, each represented by a single image repeated 80 times, totaling 16,000 EEG recordings. Both the training and test images are presented in a pseudorandom order to minimize habituation effects. Each image is displayed for 100 milliseconds followed by a blank screen for another 100 milliseconds to reduce blink-related and other artifacts. The raw EEG data were filtered between 0.1 Hz and 100 Hz, sampled at 1,000 Hz, and recorded using 63 channels.

For EEG preprocessing, we follow the methodology outlined in (Song et al. 2024; Li et al. 2024). We segment the EEG data into trials ranging from 0 to 1000 ms post-stimulus onset and perform baseline correction using the average value over the 200 ms period preceding the stimulus. All electrodes are retained, and the data are downsampled to 250 Hz. Multivariate noise normalization is applied to the training data, and the EEG repetitions for each image

<table><tr><td>Method</td><td>sub-01</td><td>sub-02</td><td>sub-03</td><td>sub-04</td><td>sub-05</td><td>sub-06</td><td>sub-07</td><td>sub-08</td><td>sub-09</td><td>sub-10</td><td>Ave</td></tr><tr><td rowspan="2">CognitionCapturer (all)</td><td>31.41</td><td>31.44</td><td>38.19</td><td>40.37</td><td>24.44</td><td>34.84</td><td>34.65</td><td>48.10</td><td>37.42</td><td>35.57</td><td>35.64</td></tr><tr><td>79.65</td><td>77.80</td><td>85.65</td><td>85.80</td><td>66.34</td><td>78.75</td><td>80.95</td><td>88.60</td><td>79.36</td><td>79.29</td><td>80.22</td></tr><tr><td rowspan="2">CognitionCapturer (image)</td><td>27.22</td><td>28.72</td><td>37.19</td><td>37.69</td><td>21.84</td><td>31.55</td><td>32.80</td><td>47.60</td><td>33.36</td><td>35.07</td><td>33.30</td></tr><tr><td>59.50</td><td>56.95</td><td>66.10</td><td>63.20</td><td>47.75</td><td>58.05</td><td>59.55</td><td>73.50</td><td>57.64</td><td>63.57</td><td>60.58</td></tr><tr><td rowspan="2">CognitionCapturer (text)</td><td>17.97</td><td>16.16</td><td>20.19</td><td>26.75</td><td>13.12</td><td>19.90</td><td>22.10</td><td>29.40</td><td>21.93</td><td>21.29</td><td>20.88</td></tr><tr><td>35.45</td><td>33.85</td><td>38.10</td><td>46.30</td><td>29.90</td><td>36.45</td><td>37.90</td><td>48.60</td><td>37.86</td><td>40.64</td><td>38.51</td></tr><tr><td rowspan="2">CognitionCapturer (depth)</td><td>23.10</td><td>21.85</td><td>29.65</td><td>34.40</td><td>15.75</td><td>27.50</td><td>30.90</td><td>36.90</td><td>27.14</td><td>26.86</td><td>27.41</td></tr><tr><td>57.40</td><td>53.25</td><td>61.65</td><td>65.50</td><td>40.25</td><td>50.20</td><td>54.55</td><td>60.20</td><td>49.00</td><td>49.14</td><td>54.11</td></tr><tr><td rowspan="2">BraVL (Du et al. 2023)</td><td>6.1</td><td>4.9</td><td>5.6</td><td>5.0</td><td>4.0</td><td>6.0</td><td>6.5</td><td>8.8</td><td>4.3</td><td>7.0</td><td>5.8</td></tr><tr><td>17.9</td><td>14.9</td><td>17.4</td><td>15.1</td><td>13.4</td><td>18.2</td><td>20.4</td><td>23.7</td><td>14.0</td><td>19.7</td><td>17.5</td></tr><tr><td rowspan="2">NICE (Song et al. 2024)</td><td>12.3</td><td>10.4</td><td>13.1</td><td>16.4</td><td>8.0</td><td>15.1</td><td>15.2</td><td>20.0</td><td>13.1</td><td>14.9</td><td>13.8</td></tr><tr><td>36.6</td><td>33.9</td><td>39.0</td><td>47.0</td><td>26.9</td><td>40.6</td><td>42.1</td><td>49.9</td><td>37.1</td><td>41.9</td><td>39.5</td></tr><tr><td rowspan="2">ATM (Li et al. 2024)</td><td>25.6</td><td>22.0</td><td>25.0</td><td>31.4</td><td>12.9</td><td>21.3</td><td>30.5</td><td>38.8</td><td>24.4</td><td>29.1</td><td>26.1</td></tr><tr><td>60.4</td><td>54.5</td><td>62.4</td><td>60.9</td><td>43.0</td><td>51.1</td><td>61.5</td><td>72.0</td><td>51.5</td><td>63.5</td><td>58.1</td></tr></table>

Table 2: Overall accuracy (acc±std) of 200-way zero-shot classification: Top-1 and Top-5. The first line in each cell represents the Top-1 accuracy, and the second line represents the Top-5 accuracy. (In the calculation of CognitionCapturer (all)'s classification accuracy, if any Modality Expert Encoder correctly classifies a sample, the sample is considered correctly classified.) 

<table><tr><td rowspan="2">Method (Averaged across subject)</td><td colspan="3">Low-level</td><td colspan="4">High-level</td></tr><tr><td>PixCorr↑</td><td>SSIM ↑</td><td>AlexNet(2) ↑</td><td>AlexNet(5) ↑</td><td>Inception ↑</td><td>CLIP ↑</td><td>SwAV ↓</td></tr><tr><td>CognitionCapturer (all)</td><td>0.150</td><td>0.347</td><td>0.754</td><td>0.623</td><td>0.669</td><td>0.715</td><td>0.590</td></tr><tr><td>CognitionCapturer (image)</td><td>0.132</td><td>0.321</td><td>0.813</td><td>0.671</td><td>0.664</td><td>0.705</td><td>0.599</td></tr><tr><td>CognitionCapturer (text)</td><td>0.102</td><td>0.288</td><td>0.727</td><td>0.582</td><td>0.586</td><td>0.598</td><td>0.673</td></tr><tr><td>CognitionCapturer (depth)</td><td>0.104</td><td>0.370</td><td>0.796</td><td>0.638</td><td>0.565</td><td>0.579</td><td>0.686</td></tr><tr><td>META-MEG Benchetrit, Banville, and King</td><td>0.090</td><td>0.341</td><td>0.774</td><td>0.876</td><td>0.703</td><td>0.811</td><td>0.567</td></tr><tr><td>MindEye-fMRI Scotti et al.</td><td>0.309</td><td>0.323</td><td>0.947</td><td>0.978</td><td>0.938</td><td>0.941</td><td>0.367</td></tr></table>

Table 3: Quantitative comparison results on Things-EEG (Gifford et al. 2022) (compared to MEG data on Things-MEG (Hebart et al. 2023) and fMRI data on NSD (Allen et al. 2022)). We report 7 different metrics to quantify the model's performance in reconstructing images at both low-level and high-level aspects.

in the test set are averaged to improve the signal-to-noise ratio. Subsequently, to obtain a multimodally aligned dataset, we use BLIP2 (Li et al. 2023) for textual descriptions of the images and DepthAnything (Yang et al. 2024) for depth estimation, resulting in an aligned text and depth dataset.

# Implementation Details

We implemented our method on a single GeForce RTX 2080 Ti GPU. following the training strategy described in (Song et al. 2024). The model was evaluated on the test set at the end of each epoch, with both training and testing conducted on separate subjects. For the training of the Modality Expert Encoder phase, we used the AdamW optimizer with a learning rate of 0.0003, a batch size of 1024, and trained for 20 epochs. Training for one subject took approximately 30 minutes.

Images were resized to $224 \times 224$ pixels and normalized before being processed by the Modality Expert Encoder. During the training of the diffusion prior, we used a batch size of 512, trained for 100 epochs, and set the number of inference steps to 50. The guidance scale was set to 7.5. In each batch, 10% of the image embeddings were randomly replaced with noise. The embedding dimension was 1024.

In the generation process, we utilized SDXL-Turbo and IP-Adapter from Hugging Face. We set the inference steps for SDXL-Turbo to 5. When configuring the IP-Adapter, for the image modality, we used the full IP-Adapter with the scale set to 1. For the text and Depth modalities, we set the scale of their respective IP-Adapter's Layout block and Style block to 0, ensuring a focus on structural and semantic control in the reconstruction results.

# Results and Discussion

# Classification Performance

The classification results of CognitionCapturer are shown in Table 2. We evaluated CognitionCapturer's ability to decode EEG embeddings based on different baseline modalities. To verify whether CognitionCapturer extracts complementary information across multiple modalities, we combined the top-5 results from three modalities, as shown in the upper bound row of Table 2. The results indicate that compared to previous work (Li et al. 2024; Du et al. 2023), CognitionCapturer achieves state-of-the-art performance on the image modality. With the introduction of the text and depth modalities, the model gains access to more complementary information $^{2}$ , leading to a significant increase in the poten-

![](images/55f0786e4dbc62d027eba868ab4036fdd070010daa35ee30c42f374f7a96e345.jpg)

<details>
<summary>text_image</summary>

Visual
Stimulus
Reconstructed
Stimuli
(Ours)
Reconstructed
Stimuli
(ATM)
</details>

Figure 3: Visual Comparison. Selected reconstruction results from subject-08 show that our reconstructed visual stimuli exhibit finer-grained features.

tial amount of effective information. This suggests that complementary information across different modalities is indeed effective.

# Visual Stimuli Reconstruction Performance $^{3}$

Since subject-08 showed the highest classification results in both our model and ATM, we chose subject-08 for the comparison. Some of the visual stimuli reconstructed by CognitionCapturer are shown in Fig 3.

The results show that CognitionCapturer outperforms previous work (Li et al. 2024) in the fine-grained alignment of reconstructed visual stimuli. To further qualitatively analyze the effectiveness of CognitionCapturer's reconstruction, we recovered visual stimuli for each individual modality and compared them with the complete CognitionCapturer. As shown in Fig. 4, there are differences in reconstruction performance when using single modalities: stimuli recovered only using the Text modality tend to be more abstract, while the Depth modality can better reconstruct structural information but performs poorly on semantic information. Notably, the image modality, which contains the richest information, sometimes loses certain details in its reconstructions. However, with the assistance of the Text and Depth modalities, CognitionCapturer recovers more reasonable visual stimuli. For instance, in Fig. 4, when the visual stimulus is a basketball, the image modality misses the “circular” feature, whereas the depth modality retains this information well.

To quantitatively compare our approach with the current state-of-the-art methods, we follow the evaluation metrics outlined in (Benchetrit, Banville, and King 2024) and conduct further quantitative comparisons on the reconstructed images. The results in Table 3 show that CognitionCapturer, when using all modality information, outperforms the use of a single modality in both low-level and high-level metrics. In low-level metrics, CognitionCapturer even matches or surpasses work using higher spatial resolution MEG signals. However, in high-level metrics, there remains a significant gap relative to MEG and fMRI signals, indicating that MEG and fMRI signals are easier to decode for meaningful information than EEG signals.

![](images/8834e6f6a22b065baf5878195ea25c605364bf3149a6ae791f809cc34c4bd345.jpg)

<details>
<summary>text_image</summary>

Visual Stimuli
Reconstructed Stimuli (ATM)
Reconstructed Stimuli (Ours) (All modality)
Reconstructed Stimuli (Ours) (Image modality)
Reconstructed Stimuli (Ours) (Text modality)
Reconstructed Stimuli (Ours) (Depth modality)
</details>

Figure 4: Reconstruction results of CognitionCapturer on different modality and comparison with prior work.

# How Different Modality Expert Encoders Focus on Brain Regions

In the previous section, we analyzed the reconstruction results of CognitionCapturer. To provide evidence for the feasibility and interpretability of CognitionCapturer, we use Grad-CAM (Selvaraju et al. 2017) to visualize the regions of interest for different modality encoders. To mitigate the influence of individual subjects, we conducted an average analysis of the Grad-CAM results across all subjects' models. As shown in Fig. 10(A), the raw EEG signal is heavily influenced by frontal lobe responses, whereas our Modality Expert Encoder primarily focuses on the occipital and tem-

poral lobes, areas responsible for processing visual information (DiCarlo and Cox 2007). Notably, compared to the Image Expert Encoder, which mainly attends to the occipital region, the Text Expert Encoder and Depth Expert Encoder attend to broader regions including both the occipital and temporal lobes.

Surprisingly, the Depth Expert Encoder exhibits more significant attention to the right inferior temporal lobe, an area primarily involved in object recognition but less sensitive to object shape, size, and orientation (Epstein and Kanwisher 1998). We believe this is because depth information lacks many lower-level visual features such as color and texture, leaving only shape and depth information. Similar to the phenomenon of sensory compensation (Rauschecker 1995), this forces the model to seek higher-level brain information to ensure effective recognition of similar objects. This demonstrates that our modality-specific expert models reasonably focus on different brain regions, aligning with existing neuroscience theories.

# How Different Brain Area Interact with Visual Stimuli

The analysis in the previous section demonstrated exciting results. To provide additional evidence for the effective interaction between EEG and image information, we further used Grad-CAM to visualize the image regions attended to by the embeddings produced by our Modality Expert Encoders and compared them with the original CLIP embeddings.

As shown in Fig. 10(B), first, in the original CLIP model, the text embedding focuses more on the object itself, while the image and depth embeddings have broader attention areas. Our Modality Expert Encoders yield EEG embeddings for different modalities that show similar results to those of CLIP. Specifically, the EEG embedding from the Text Expert Encoder focuses more on high-level information in the image, such as the baseball bats. In contrast, the Image and Depth Expert Encoders have broader attention over the image. Correspondingly, the brain regions attended to by the Image and Depth models are also more extensive compared to Text. This provides strong evidence for the interpretability of CognitionCapturer.

# Conclusion

In this work, we propose CognitionCapturer to extract multimodal representations from EEG signals and decode visual stimuli from them. Specifically, we introduce multiple Modality Expert Encoders to specialize in aligning EEG embeddings with those of different modalities, enabling the model to capture both semantic and structural information simultaneously. The analysis of brain activity and the interpretability of our model demonstrate that it successfully obtains meaningful representations of brain signals. This provides new insights for subsequent work in brain decoding.

# Acknowledgments

Thank you, Dr. Jili Xia, for your kindness and your advice on this work!

![](images/f202e9f8d2191518141ec2287c5c53499e041a571aa8f465423fd9a3566e0b8a.jpg)

<details>
<summary>text_image</summary>

Input
Image/EEG
Text/EEG
Depth/EEG
A
B
</details>

Figure 5: (A) The Grad-CAM results from different Modality Expert Encoders show the activation in the occipital and temporal lobes related to the input EEG signals. (B) The Grad-CAM results from different modality Expert Encoders on the brain signals corresponding to the example image, visualizing the regions of attention in the images and comparing them with the original CLIP embeddings.

# References

Allen, E. J.; St-Yves, G.; Wu, Y.; Breedlove, J. L.; Prince, J. S.; Dowdle, L. T.; Nau, M.; Caron, B.; Pestilli, F.; Charest, I.; et al. 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature neuroscience, 25(1): 116–126.   
Benchetrit, Y.; Banville, H.; and King, J.-R. 2024. Brain decoding: toward real-time reconstruction of visual perception. arXiv:2310.19812.   
Cherti, M.; Beaumont, R.; Wightman, R.; Wortsman, M.; Ilharco, G.; Gordon, C.; Schuhmann, C.; Schmidt, L.; and Jitsev, J. 2023. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2818–2829.   
Défossez, A.; Caucheteux, C.; Rapin, J.; Kabeli, O.; and King, J.-R. 2023. Decoding speech perception from non-invasive brain recordings. Nature Machine Intelligence, 5(10): 1097–1107.   
DiCarlo, J. J.; and Cox, D. D. 2007. Untangling invariant object recognition. Trends in cognitive sciences, 11(8): 333–341.   
Du, C.; Fu, K.; Li, J.; and He, H. 2023. Decoding Visual Neural Representations by Multimodal Learning of Brain-Visual-Linguistic Features. IEEE Transactions on Pattern Analysis and Machine Intelligence.   
Epstein, R.; and Kanwisher, N. 1998. A cortical representation of the local visual environment. Nature, 392(6676): 598–601.   
Gifford, A. T.; Dwivedi, K.; Roig, G.; and Cichy, R. M. 2022. A large and rich EEG dataset for modeling human visual object recognition. NeuroImage, 264: 119754.   
Gu, Z.; Jamison, K.; Kuceyeski, A.; and Sabuncu, M. R. 2024. Decoding natural image stimuli from fMRI data with a surface-based convolutional network. In Medical Imaging with Deep Learning, 107–118. PMLR.   
He, K.; Fan, H.; Wu, Y.; Xie, S.; and Girshick, R. 2020. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9729–9738.

Hebart, M. N.; Contier, O.; Teichmann, L.; Rockter, A. H.; Zheng, C. Y.; Kidder, A.; Corriveau, A.; Vaziri-Pashkam, M.; and Baker, C. I. 2023. THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. \*Elife\*, 12: e82580.   
Kay, K. N.; Naselaris, T.; Prenger, R. J.; and Gallant, J. L. 2008. Identifying natural images from human brain activity. Nature, 452(7185): 352–355.   
Li, D.; Wei, C.; Li, S.; Zou, J.; and Liu, Q. 2024. Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion. arXiv:2403.07721.   
Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, 19730–19742. PMLR.   
Li, R.; Johansen, J. S.; Ahmed, H.; Ilyevsky, T. V.; Wilbur, R. B.; Bharadwaj, H. M.; and Siskind, J. M. 2020. The perils and pitfalls of block design for EEG classification experiments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1):316–333.   
Liu, D.; Dai, W.; Zhang, H.; Jin, X.; Cao, J.; and Kong, W. 2023a. Brain-machine coupled learning method for facial emotion recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9): 10703–10717.   
Liu, Y.; Ma, Y.; Zhou, W.; Zhu, G.; and Zheng, N. 2023b. BrainCLIP: Bridging Brain and Visual-Linguistic Representation Via CLIP for Generic Natural Visual Stimulus Decoding. arXiv:2302.12971.   
Lupyan, G.; Rahman, R. A.; Boroditsky, L.; and Clark, A. 2020. Effects of language on visual perception. Trends in cognitive sciences, 24(11): 930–944.   
Miyawaki, Y.; Uchida, H.; Yamashita, O.; Sato, M.-a.; Morito, Y.; Tanabe, H. C.; Sadato, N.; and Kamitani, Y. 2008. Visual image reconstruction from human brain activity using a combination of multiscale local image decoders. Neuron, 60(5): 915–929.   
Naselaris, T.; Prenger, R. J.; Kay, K. N.; Oliver, M.; and Gallant, J. L. 2009. Bayesian reconstruction of natural images from human brain activity. Neuron, 63(6): 902–915.   
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PMLR.   
Ramesh, A.; Dhariwal, P.; Nichol, A.; Chu, C.; and Chen, M. 2022. Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv:2204.06125.   
Rauschecker, J. P. 1995. Compensatory plasticity and sensory substitution in the cerebral cortex. Trends in neurosciences, 18(1):36–43.   
Ren, Z.; Li, J.; Xue, X.; Li, X.; Yang, F.; Jiao, Z.; and Gao, X. 2021. Reconstructing seen image from brain activity by visually-guided cognitive representation and adversarial learning. NeuroImage, 228: 117602.   
Sauer, A.; Lorenz, D.; Blattmann, A.; and Rombach, R. 2023. Adversarial Diffusion Distillation. arXiv:2311.17042.   
Scotti, P.; Banerjee, A.; Goode, J.; Shabalin, S.; Nguyen, A.; Dempster, A.; Verlinde, N.; Yundler, E.; Weisberg, D.; Norman, K.; et al. 2024. Reconstructing the mind's eye: fMRI-to-image with contrastive learning and diffusion priors. Advances in Neural Information Processing Systems, 36.

Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; and Batra, D. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, 618–626.

Song, Y.; Liu, B.; Li, X.; Shi, N.; Wang, Y.; and Gao, X. 2024. Decoding Natural Images from EEG for Object Recognition. In International Conference on Learning Representations.

Takagi, Y.; and Nishimoto, S. 2023. High-Resolution Image Reconstruction With Latent Diffusion Models From Human Brain Activity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14453–14463.

van den Oord, A.; Li, Y.; and Vinyals, O. 2019. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Yang, L.; Kang, B.; Huang, Z.; Xu, X.; Feng, J.; and Zhao, H. 2024. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10371–10381.

Ye, H.; Zhang, J.; Liu, S.; Han, X.; and Yang, W. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv:2308.06721.

Zhang, R.; Zeng, Z.; Guo, Z.; and Li, Y. 2022. Can language understand depth? In Proceedings of the 30th ACM International Conference on Multimedia, 6868–6874.

# Supplementary Material of CognitionCapturer: Decoding Visual Stimuli From Human EEG Signal With Multimodal Information

Details of CognitionCapturer

# Modality Expert Encoder (EEG signal)

The Modality Expert Encoder for EEG-Modality pairs uses a consistent structure adapted from (Li et al. 2024). Specifically, the input EEG data is processed through a single-head attention model. Positional encoding is applied before passing the data through a transformer encoder, resulting in a vector of the same shape as the EEG input. The output is then passed through a linear layer. The resulting features are fed into a Temporal-Spatial Convolution (TSConv) module to generate the EEG embedding. Within the TSConv module, two consecutive convolution layers, a pooling layer, BatchNorm2d, and ELU are utilized for feature extraction.

- The first convolution layer extracts local temporal features, with an output channel count of 40, a kernel size of (1, 25), and a stride of (1, 1).   
- The second convolution layer extracts global channel-wise features, maintaining an output channel count of 40, a kernel size of (63, 1), and a stride of (1, 1).

Finally, the features are projected via a projection to generate the output. In the projection layer, the dimensions are first mapped to 1024 using a linear transformation. This is followed by a residual connection with an internal structure consisting of GELU and another linear layer. The output is then normalized using LayerNorm before being returned as the final output. For detailed dimension changes and parameters, refer to Table 4.

# Modality Expert Encoder (Image Text Depth)

The encoders for images, text, and depth all use the Image Text Encoder from OpenCLIP-ViT-H/14. We consider depth as a form of image but add a projection layer to ensure stability. The structure of the projection layer is the same as the Project Linear structure in the Modality Expert Encoder.

# Diffusion Prior

The model architecture of the Diffusion Prior is adapted from (Li et al. 2024), and classifier-free guidance [reference here] is utilized during inference. Detailed structures can be found in the code file Scripts/train\_align/diffusion\_prior.py.

# Impact of Batch Size and Learning Rate

To identify the optimal hyperparameters, we experimented with various batch sizes and learning rates. Given that Subject-08 demonstrated the most representative performance, we focused solely on this subject for hyperparameter tuning. For brevity, we report only the classification performance using all modalities, with the results presented in Table 5.

# Reconstruction Performance and Results

To supplement the reconstruction performance reported in the main paper, we present the reconstruction performance for each subject and modalities in Table 6 - 9. As the reconstruction performance of ATM(Li et al. 2024) was not specified for a particular subject or averaged across subjects, a direct comparison cannot be made under the same conditions. Therefore, we provide the test results obtained for each subject in the supplementary material.

In the sample images, to avoid cherry-picked results and overestimating the capabilities of CognitionCapture, we follow the (Benchetrit, Banville, and King 2024)'s approach by ranking the reconstruction results according to the SwAV and PixCorr metrics from highest to lowest. We present the representative visual stimuli generated under the best, average, and worst metric conditions. We only display the results of the best-performing subject-08 and the worst-performing subject-05. The results are shown in Fig. 1 - 4.

# More Model Visualization Results

In this section, we present the visualization results obtained using Grad-CAM(Selvaraju et al. 2017) on the test set for each subject. These visualizations show the attention regions of the Modality Expert Encoders compared to the input attention regions of the respective subjects. The results are depicted in Fig. 10.

Regarding the Grad-CAM visualization of attention regions on images, we present the visualization results of the attention areas focused on by different Modality Expert Encoders for subject-08. The example images selected for visualization are the same as those used in the reconstruction results; see Fig. 11 for these results.

<table><tr><td>Layer</td><td>Type</td><td>Input Shape</td><td>Output Shape</td><td>Parameters</td></tr><tr><td>Spatial attention block</td><td>Positional Encoding + Attention</td><td>(batch, 63, 250)</td><td>(batch, 63, 250)</td><td>553K</td></tr><tr><td>Linear</td><td>Linear</td><td>(batch, 63, 250)</td><td>(batch, 63, 250)</td><td>63K</td></tr><tr><td>TSConv</td><td>Convolution + MaxPooling + BatchNorm</td><td>(batch, 63, 250)</td><td>(batch, 36, 40)</td><td>104K</td></tr><tr><td>Temporal Aggregation</td><td>Dimention Transform</td><td>(batch, 36, 40)</td><td>(batch, 1440)</td><td>0</td></tr><tr><td>Project Linear</td><td>Residual Linear</td><td>(batch, 1440)</td><td>(batch, 1440)</td><td>2527K</td></tr><tr><td>All</td><td></td><td>(batch, 63, 250)</td><td>(batch, 1024)</td><td>3247K</td></tr></table>

Table 4: Dimension changes and parameter counts in the modules of the Modality Expert Encoder.

<table><tr><td>Batchsize / Learning rate</td><td>1.00E-04</td><td>3.00E-04</td><td>6.00E-04</td><td>1.00E-03</td></tr><tr><td>32</td><td>0.505</td><td>0.445</td><td>0.435</td><td>0.410</td></tr><tr><td>64</td><td>0.485</td><td>0.490</td><td>0.430</td><td>0.435</td></tr><tr><td>128</td><td>0.470</td><td>0.455</td><td>0.480</td><td>0.410</td></tr><tr><td>256</td><td>0.470</td><td>0.495</td><td>0.505</td><td>0.460</td></tr><tr><td>512</td><td>0.450</td><td>0.450</td><td>0.480</td><td>0.420</td></tr><tr><td>1024</td><td>0.480</td><td>0.520</td><td>0.500</td><td>0.470</td></tr></table>

Table 5: The classification performance of CognitionCapturer under different batch sizes and learning rates for sub-08.

![](images/818e943bebe48de6e53d19df6b8f37d0728098aaa8eb0b32a68c81251e439c65.jpg)

<details>
<summary>text_image</summary>

Best
Visual Stimuli
Reconstructed Stimuli
Medium
Visual Stimuli
Reconstructed Stimuli
Worst
Visual Stimuli
Reconstructed Stimuli
</details>

Figure 6: Subject-08's Best, Medium, and Worst images selected based on the Pixcorr metric.

Best

Reconstructed Visual Stimuli Stimuli

![](images/db6a996493844855f766464b5cc67d291077ce3823fce7de6e342a3f269dfe67.jpg)  
Medium   
Reconstructed Visual Stimuli Stimuli   
Reconstructed Visual Stimuli Stimuli   
Reconstructed Visual Stimuli Stimuli

Worst

Figure 7: Subject-08's Best, Medium, and Worst images selected based on the SwAV metric.

Best

![](images/585a3af0adb17b3716c04de2bd14435bc0991ae4cba1dd43362966f8442180d9.jpg)  
Medium   
Reconstructed Visual Stimuli Stimuli   
Reconstructed Visual Stimuli Stimuli

Worst

Figure 8: Subject-05's Best, Medium, and Worst images selected based on the Pixcorr metric.

![](images/3ca059847bb2cdcf7a09c30961645cd4d984524ae8b640d881f9693f4b5b63bb.jpg)  
Reconstructed Visual Stimuli Stimuli   
Reconstructed Visual Stimuli Stimuli   
Reconstructed Visual Stimuli

Figure 9: Subject-05's Best, Medium, and Worst images selected based on the SwAV metric. 

<table><tr><td rowspan="2">Subject</td><td colspan="3">Low-level</td><td colspan="4">High-level</td></tr><tr><td>Pixcorr↑</td><td>SSIM↑</td><td>AlexNet(2) ↑</td><td>AlexNet(5) ↑</td><td>Inception↑</td><td>CLIP↑</td><td>SwAV↓</td></tr><tr><td>1</td><td>0.148</td><td>0.334</td><td>0.741</td><td>0.626</td><td>0.666</td><td>0.711</td><td>0.592</td></tr><tr><td>2</td><td>0.147</td><td>0.344</td><td>0.764</td><td>0.618</td><td>0.661</td><td>0.725</td><td>0.590</td></tr><tr><td>3</td><td>0.140</td><td>0.307</td><td>0.715</td><td>0.549</td><td>0.690</td><td>0.710</td><td>0.603</td></tr><tr><td>4</td><td>0.166</td><td>0.355</td><td>0.801</td><td>0.660</td><td>0.701</td><td>0.765</td><td>0.543</td></tr><tr><td>5</td><td>0.130</td><td>0.343</td><td>0.731</td><td>0.639</td><td>0.594</td><td>0.655</td><td>0.611</td></tr><tr><td>6</td><td>0.152</td><td>0.337</td><td>0.748</td><td>0.620</td><td>0.646</td><td>0.688</td><td>0.630</td></tr><tr><td>7</td><td>0.145</td><td>0.355</td><td>0.777</td><td>0.623</td><td>0.731</td><td>0.721</td><td>0.576</td></tr><tr><td>8</td><td>0.175</td><td>0.366</td><td>0.760</td><td>0.610</td><td>0.721</td><td>0.744</td><td>0.577</td></tr><tr><td>9</td><td>0.148</td><td>0.337</td><td>0.731</td><td>0.623</td><td>0.625</td><td>0.692</td><td>0.605</td></tr><tr><td>10</td><td>0.152</td><td>0.389</td><td>0.773</td><td>0.664</td><td>0.657</td><td>0.736</td><td>0.569</td></tr><tr><td>Ave</td><td>0.150</td><td>0.347</td><td>0.754</td><td>0.623</td><td>0.669</td><td>0.715</td><td>0.590</td></tr><tr><td>ATM</td><td>/</td><td>0.345</td><td>0.776</td><td>0.866</td><td>0.734</td><td>0.786</td><td>0.582</td></tr><tr><td>1</td><td>0.126</td><td>0.317</td><td>0.812</td><td>0.653</td><td>0.655</td><td>0.700</td><td>0.595</td></tr><tr><td>2</td><td>0.109</td><td>0.309</td><td>0.809</td><td>0.638</td><td>0.634</td><td>0.702</td><td>0.602</td></tr><tr><td>3</td><td>0.137</td><td>0.300</td><td>0.803</td><td>0.604</td><td>0.651</td><td>0.689</td><td>0.609</td></tr><tr><td>4</td><td>0.125</td><td>0.328</td><td>0.853</td><td>0.732</td><td>0.738</td><td>0.781</td><td>0.551</td></tr><tr><td>5</td><td>0.116</td><td>0.327</td><td>0.769</td><td>0.668</td><td>0.618</td><td>0.667</td><td>0.612</td></tr><tr><td>6</td><td>0.138</td><td>0.280</td><td>0.787</td><td>0.665</td><td>0.592</td><td>0.643</td><td>0.654</td></tr><tr><td>7</td><td>0.135</td><td>0.330</td><td>0.842</td><td>0.700</td><td>0.695</td><td>0.711</td><td>0.598</td></tr><tr><td>8</td><td>0.154</td><td>0.327</td><td>0.830</td><td>0.655</td><td>0.711</td><td>0.748</td><td>0.583</td></tr><tr><td>9</td><td>0.138</td><td>0.310</td><td>0.799</td><td>0.670</td><td>0.664</td><td>0.673</td><td>0.603</td></tr><tr><td>10</td><td>0.138</td><td>0.378</td><td>0.825</td><td>0.725</td><td>0.676</td><td>0.736</td><td>0.580</td></tr><tr><td>Ave</td><td>0.132</td><td>0.321</td><td>0.813</td><td>0.671</td><td>0.664</td><td>0.705</td><td>0.599</td></tr></table>

Table 6: The reconstruction performance of CognitionCapture when using ALL modalities.

Table 7: The reconstruction performance of CognitionCapture when using IMAGE modality.

<table><tr><td rowspan="2">Subject</td><td colspan="3">Low-level</td><td colspan="4">High-level</td></tr><tr><td>Pixcorr↑</td><td>SSIM↑</td><td>AlexNet(2) ↑</td><td>AlexNet(5) ↑</td><td>Inception↑</td><td>CLIP↑</td><td>SwAV↓</td></tr><tr><td>1</td><td>0.114</td><td>0.309</td><td>0.722</td><td>0.551</td><td>0.568</td><td>0.591</td><td>0.678</td></tr><tr><td>2</td><td>0.105</td><td>0.280</td><td>0.716</td><td>0.589</td><td>0.575</td><td>0.604</td><td>0.679</td></tr><tr><td>3</td><td>0.104</td><td>0.214</td><td>0.700</td><td>0.551</td><td>0.557</td><td>0.553</td><td>0.730</td></tr><tr><td>4</td><td>0.117</td><td>0.341</td><td>0.761</td><td>0.628</td><td>0.662</td><td>0.636</td><td>0.624</td></tr><tr><td>5</td><td>0.105</td><td>0.303</td><td>0.680</td><td>0.546</td><td>0.540</td><td>0.607</td><td>0.681</td></tr><tr><td>6</td><td>0.111</td><td>0.273</td><td>0.716</td><td>0.566</td><td>0.591</td><td>0.600</td><td>0.675</td></tr><tr><td>7</td><td>0.098</td><td>0.278</td><td>0.731</td><td>0.594</td><td>0.608</td><td>0.611</td><td>0.661</td></tr><tr><td>8</td><td>0.078</td><td>0.267</td><td>0.776</td><td>0.615</td><td>0.589</td><td>0.572</td><td>0.695</td></tr><tr><td>9</td><td>0.080</td><td>0.306</td><td>0.701</td><td>0.578</td><td>0.550</td><td>0.585</td><td>0.659</td></tr><tr><td>10</td><td>0.109</td><td>0.308</td><td>0.769</td><td>0.599</td><td>0.621</td><td>0.626</td><td>0.649</td></tr><tr><td>Ave</td><td>0.102</td><td>0.288</td><td>0.727</td><td>0.582</td><td>0.586</td><td>0.598</td><td>0.673</td></tr></table>

Table 8: The reconstruction performance of CognitionCapture when using TEXT modality.

<table><tr><td rowspan="2">Subject</td><td colspan="3">Low-level</td><td colspan="4">High-level</td></tr><tr><td>Pixcorr↑</td><td>SSIM↑</td><td>AlexNet(2) ↑</td><td>AlexNet(5) ↑</td><td>Inception↑</td><td>CLIP↑</td><td>SwAV↓</td></tr><tr><td>1</td><td>0.116</td><td>0.340</td><td>0.798</td><td>0.618</td><td>0.556</td><td>0.561</td><td>0.701</td></tr><tr><td>2</td><td>0.106</td><td>0.368</td><td>0.789</td><td>0.633</td><td>0.559</td><td>0.601</td><td>0.668</td></tr><tr><td>3</td><td>0.097</td><td>0.365</td><td>0.775</td><td>0.621</td><td>0.565</td><td>0.578</td><td>0.710</td></tr><tr><td>4</td><td>0.093</td><td>0.359</td><td>0.843</td><td>0.694</td><td>0.587</td><td>0.625</td><td>0.670</td></tr><tr><td>5</td><td>0.082</td><td>0.417</td><td>0.744</td><td>0.582</td><td>0.533</td><td>0.528</td><td>0.699</td></tr><tr><td>6</td><td>0.107</td><td>0.385</td><td>0.767</td><td>0.568</td><td>0.521</td><td>0.539</td><td>0.692</td></tr><tr><td>7</td><td>0.115</td><td>0.375</td><td>0.812</td><td>0.641</td><td>0.572</td><td>0.578</td><td>0.676</td></tr><tr><td>8</td><td>0.081</td><td>0.370</td><td>0.852</td><td>0.652</td><td>0.586</td><td>0.586</td><td>0.671</td></tr><tr><td>9</td><td>0.119</td><td>0.361</td><td>0.764</td><td>0.642</td><td>0.543</td><td>0.554</td><td>0.697</td></tr><tr><td>10</td><td>0.128</td><td>0.363</td><td>0.818</td><td>0.732</td><td>0.627</td><td>0.641</td><td>0.670</td></tr><tr><td>Ave</td><td>0.104</td><td>0.370</td><td>0.796</td><td>0.638</td><td>0.565</td><td>0.579</td><td>0.686</td></tr></table>

Table 9: The reconstruction performance of CognitionCapture when using DEPTH modality.

![](images/bce21af434e43f097ef042f5128cd81e5649ff883321cb64c0fdd856be15d792.jpg)  
Figure 10: The input topographies of the EEG signals for all subjects, along with the brain regions attended to by the different Modality Expert Encoders.

Worst    Medium    Best   
![](images/9a18899f4bb83eed2ffbc78573f6f2550646cd94fcf9a22d01ff492f2c567a33.jpg)

<details>
<summary>heatmap</summary>

| Modality | Encoder Type | Text | Image |
| --- | --- | --- | --- |
| 1 | Visual Stimuli | 1 | 1 |
| 2 | Visual Stimuli | 2 | 2 |
| 3 | Visual Stimuli | 3 | 3 |
| 4 | Visual Stimuli | 4 | 4 |
| 5 | Visual Stimuli | 5 | 5 |
| 6 | Visual Stimuli | 6 | 6 |
| 7 | Visual Stimuli | 7 | 7 |
| 8 | Visual Stimuli | 8 | 8 |
| 9 | Visual Stimuli | 9 | 9 |
| 10 | Visual Stimuli | 10 | 10 |
| 11 | Visual Stimuli | 11 | 11 |
| 12 | Visual Stimuli | 12 | 12 |
| 13 | Visual Stimuli | 13 | 13 |
| 14 | Visual Stimuli | 14 | 14 |
| 15 | Visual Stimuli | 15 | 15 |
| 16 | Visual Stimuli | 16 | 16 |
| 17 | Visual Stimuli | 17 | 17 |
| 18 | Visual Stimuli | 18 | 18 |
| 19 | Visual Stimuli | 19 | 19 |
| 20 | Visual Stimuli | 20 | 20 |
| 21 | Visual Stimuli | 21 | 21 |
| 22 | Visual Stimuli | 22 | 22 |
| 23 | Visual Stimuli | 23 | 23 |
| 24 | Visual Stimuli | 24 | 24 |
| 25 | Visual Stimuli | 25 | 25 |
| 26 | Visual Stimuli | 26 | 26 |
| 27 | Visual Stimuli | 27 | 27 |
| 28 | Visual Stimuli | 28 | 28 |
| 29 | Visual Stimuli | 29 | 29 |
| 30 | Visual Stimuli | 30 | 30 |
| 31 | Visual Stimuli | 31 | 31 |
| 32 | Visual Stimuli | 32 | 32 |
| 33 | Visual Stimuli | 33 | 33 |
| 34 | Visual Stimuli | 34 | 34 |
| 35 | Visual Stimuli | 35 | 35 |
| 36 | Visual Stimuli | 36 | 36 |
| 37 | Visual Stimuli | 37 | 37 |
| 38 | Visual Stimuli | 38 | 38 |
| 39 | Visual Stimuli | 39 | 39 |
| 40 | Visual Stimuli | 40 | 40 |
| 41 | Visual Stimuli | 41 | 41 |
| 42 | Visual Stimuli | 42 | 42 |
| 43 | Visual Stimuli | 43 | 43 |
| 44 | Visual Stimuli | 44 | 44 |
| 45 | Visual Stimuli | 45 | 45 |
| 46 | Visual Stimuli | 46 | 46 |
| 47 | Visual Stimuli | 47 | 47 |
| 48 | Visual Stimuli | 48 | 48 |
| 49 | Visual Stimuli | 49 | 49 |
| 50 | Visual Stimuli | 50 | 50 |
| -10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000<nl>
<fcel>-750000000000000000000000000000000000000000000000000000000000<fcel>-7555555555555555555555555555555555555555555555555555555555555555555555555555<fcel>-76666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666<fcel>-7777888888888888888888888888888888888888888888888888888888888888888888888888<fcel>-77779999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999<fcel>-7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/7777/<nl>
<fcel>-1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111<fcel>-<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>-<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>-<fcel>Visual Stimuli<fcel>-<fcel>+<nl>
<fcel>(2)<ecel><ecel><ecel><nl>
</details>

Figure 11: Grad-CAM visualizations of the image regions attended to by different Modality Expert Encoders on example images