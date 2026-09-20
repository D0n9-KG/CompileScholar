# RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement

Bochao Zou $^{1}$ , Zizheng Guo $^{1}$ , Xiaocheng Hu $^{2}$ , Huimin Ma $^{1*}$

$^{1}$ University of Science and Technology Beijing, Beijing, China

$^{2}$ China Academy of Electronics and Information Technology, Beijing, China

zoubochao@ustb.edu.cn, guozizheng@xs.ustb.edu.cn, 675342900@qq.com, mhmpub@ustb.edu.cn

# Abstract

Remote photoplethysmography (rPPG) is a method for non-contact measurement of physiological signals from facial videos, holding great potential in various applications such as healthcare, affective computing, and anti-spoofing. Existing deep learning methods struggle to address two core issues of rPPG simultaneously: understanding the periodic pattern of rPPG among long contexts and addressing large spatiotemporal redundancy in video segments. These represent a trade-off between computational complexity and the ability to capture long-range dependencies. In this paper, we introduce RhythmMamba, a state space model-based method that captures long-range dependencies while maintaining linear complexity. By viewing rPPG as a time series task through the proposed frame stem, the periodic variations in pulse waves are modeled as state transitions. Additionally, we design multi-temporal constraint and frequency domain feed-forward, both aligned with the characteristics of rPPG time series, to improve the learning capacity of Mamba for rPPG signals. Extensive experiments show that RhythmMamba achieves state-of-the-art performance with 319% throughput and 23% peak GPU memory.

Code — https://github.com/zizheng-guo/RhythmMamba

# 1 Introduction

Blood Volume Pulse (BVP) is a vital physiological signal, further enabling the extraction of key signs such as heart rate (HR) and heart rate variability (HRV). Photoplethysmography (PPG) is a non-invasive monitoring method that utilizes optical means to measure changes in blood volume within living tissues. The physiological mechanism of PPG stems from variations in blood volume during cardiac contraction and relaxation in subcutaneous blood vessels, leading to changes in light absorption and scattering. These changes result in periodic color signal variations on imaging sensors, which are imperceptible to the human eye (Verkruysse, Svaasand, and Nelson 2008; Chen and McDuff 2018). Traditionally, BVP extraction requires the use of contact sensors, which brings inconvenience and limitations. In recent years, non-contact methods for obtaining BVP, particularly rPPG, have garnered increasing attention (McDuff 2023; Li, Yu, and Shi 2023; Choi, Kang, and Kim 2024).

Early rPPG research primarily relied on traditional signal processing methods to recover weak rPPG signals from facial videos, which are susceptible to interference from environmental light, motion, and other noises. In complex environments, relying solely on signal processing methods often struggles to achieve satisfactory accuracy. In recent years, data-driven methods have become mainstream, represented by convolutional neural networks (CNNs) and transformers. However, CNNs have limited receptive fields and transformer-based architectures exhibit mediocre performance in capturing long-term dependencies from the computational complexity perspective, especially when dealing with long video sequences.

Recently, Mamba (Dao and Gu 2024) has emerged with its selective state space model, striking a balance between maintaining linear complexity and facilitating long-term dependency modeling. It has been successfully applied to various artificial intelligence tasks such as video understanding (Li et al. 2025). For rPPG tasks that typically require long-term monitoring and are suitable for deployment on mobile devices, the linear complexity and ability to capture long-term dependencies give Mamba an advantage.

However, the direct application of Mamba to rPPG tasks performs poorly. Our experiments reveal that embedding spatiotemporal information into token sequences through patch embedding leads to spatial information significantly disrupting Mamba's comprehension of temporal information (see Section 4.5). This phenomenon may stem from Mamba's linear modeling characteristics, where the states are associated with the temporal phases of the rPPG signals. The incorporation of spatial information increases the dimensionality of the state transition process, thereby adding complexity and impeding the model's learning efficacy.

Motivated by the aforementioned discussion, we propose RhythmMamba, a state space model-based architecture for remote physiological measurement. The proposed frame stem embeds spatial information from a single frame into the channels, thereby viewing the rPPG task as a time series task, allowing the periodic variations in pulse waves to be modeled as state transitions. As shown in Fig. 1, considering the linear modeling characteristics of Mamba, the phase shifts of rPPG signals can be viewed as state tran-

![](images/ef79559fa60c7614dbaaab872b8c2650eec04becd6cd216a590bc865e45b8673.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x(t)"] --> B["h(t)"]
    C["x(t+1)"] --> D["h(t+1)"]
    E["x(t+2)"] --> F["h(t+2)"]
    G["y(t)"] --> H["y(t+1)"]
    I["y(t+2)"] --> J["y(t+2)"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style G fill:#ccf,stroke:#333
    style I fill:#ccf,stroke:#333
    style J fill:#ccf,stroke:#333
```
</details>

Figure 1: A schematic diagram of state transitions. Considering the periodic nature of rPPG, the rPPG signal can be represented using a finite number of states. Where $h(t)$ represents the state vector, $x(t)$ represents the input vector, and $y(t)$ represents the output vector.

sitions within the state space. The periodic nature of rPPG allows the signal to be represented using a finite set of states.

Additionally, we design multi-temporal constraint and frequency domain feed-forward, both aligned with the characteristics of rPPG time series, to improve the learning capacity of Mamba for rPPG signals. By learning from the same sequences with varying temporal lengths, a single Mamba block can simultaneously be constrained by the periodicity of long sequences and the trends of short sequences. Subsequently, through frequency domain feed-forward, the learned temporal features by Mamba undergo inter-channel spatial interaction in the frequency domain, enabling a better discernment of the periodic nature of rPPG signals.

The main contributions are as follows:

- We propose RhythmMamba, which leverages state space models to model periodic variations as state transitions, combining multi-temporal constraints Mamba and frequency domain feed-forward to learn the quasi-periodic patterns of rPPG. To the best of our knowledge, this is the first work to investigate state space models in the rPPG domain.   
- In response to the observed phenomenon where spatial information interferes with Mamba's understanding of temporal sequences, we design the frame stem to embed spatial information into channels, reducing the dimensionality of state transitions to boost Mamba's learning.   
- We conduct extensive experiments on intra-dataset and cross-dataset scenarios. The results demonstrate that Rhythm-mMamba achieves state-of-the-art performance with $319\%$ throughput and $23\%$ GPU memory, as illustrated in Fig. 2.

# 2 Related Work

# 2.1 Remote Physiological Measurement

Early research on rPPG primarily relied on traditional signal processing methods to recover weak rPPG signals from facial videos (Verkruysse, Svaasand, and Nelson 2008; Poh, McDuff, and Picard 2010; De Haan and Jeanne 2013; Wang et al. 2016). In recent years, data-driven approaches have dominated due to their remarkable performance, showcasing a trend in the transition of backbone from 2D CNNs (Špetlík, Franc, and Matas 2018; Niu et al. 2018; Chen and McDuff 2018; Niu et al. 2020; Liu et al. 2020) to 3D CNNs (Yu, Li, and Zhao 2019; Yu et al. 2019; Zhao et al. 2021; Li, Yu, and Shi 2023) and further to transformers (Yu et al. 2022, 2023; Liu et al. 2023a; Shao et al. 2023; Liu et al. 2024; Zou et al. 2024). However, none of them have been able to effectively address the two core issues of rPPG: understanding the periodic pattern of rPPG among long contexts and addressing large spatiotemporal redundancy in video segments. This dilemma underscores a trade-off between computational complexity and the ability to capture long-range dependencies, thereby presenting a barrier to deploying rPPG solutions on mobile devices. Although previously dominant 3D CNNs and video transformers have effectively tackled one of the above issues by utilizing local convolutions or long-range attention, they fail to address both problems simultaneously. Unlike them, the proposed Rhythm-Mamba can capture long-range dependencies while maintaining linear complexity, making it fast, lightweight, and accurate for remote physiological measurement.

![](images/011380aaf0eb4742cd2e6b1844a08c41f243c9d31ac7aaf76e0bdc3d14240f15.jpg)

<details>
<summary>bubble</summary>

| Model | Throughput (Kfps) | RMSE (bpm) | Diameter (M) |
| :--- | :--- | :--- | :--- |
| RhythmMamba | 20 | 3 | 77% lighter |
| RhythmFormer | 8 | 4 | 219% faster |
| Comparable Accuracy | 6 | 5 | 0 |
| PhysNet | 12 | 12 | 0 |
| PhysFormer | 10 | 18 | 0 |
| TS-CAN | 6 | 16 | 0 |
| EfficientPhys | 8 | 21 | 0 |
| DeepPhys | 8 | 28 | 39 |
Diameter: 0M 13M 26M 39M
</details>

Figure 2: Performance and efficiency evaluation for intra-dataset testing on MMPD. The diameter of the circle indicates the peak GPU memory. The proposed RhythmMamba is faster, lighter, and achieves comparable accuracy, with these advantages becoming more pronounced as the scale increases due to its linear complexity.

# 2.2 Vision Mamba

Recently, Mamba has distinguished itself with a data-dependent state space model (SSM) and a selection mechanism utilizing parallel scanning, striking a balance between maintaining linear complexity and facilitating long-term dependency modeling. Compared to transformers with quadratic complexity attention (Vaswani et al. 2017; Arnab et al. 2021), Mamba excels at handling long sequences with linear complexity. Subsequently, the immense potential of Mamba has sparked a series of works (Zhu et al. 2024; Patro and Agneeswaran 2024; Li et al. 2025), demonstrating superior performance and higher GPU efficiency of Mamba over Transformers on downstream vision tasks. However, unlike other video tasks, rPPG signals are particularly weak and highly susceptible to noise from factors such as lighting and motion, making the direct application of the traditional Mamba architecture to rPPG tasks perform poorly. In contrast to previous works, we view the rPPG task as the time series task, fully integrating spatial information into the channels and designing multiple modules tailored for time series to boost Mamba's learning.

![](images/597cfe5ed794bbe995eee05196eaf609440e1110f1b18f6d80cae3d8528edc68.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image"] --> B["Frame Stem"]
    B --> C["Multi-temporal Constraint Mamba"]
    C --> D["Add & Norm"]
    D --> E["Frequency Domain Feed-forward"]
    E --> F["Add & Norm"]
    F --> G["Predictor"]
    G --> H["rPPG Signal"]
    
    subgraph Top-Left
        I["Image"] --> J["Diff-fusion"]
        J --> K["⊕"]
        K --> L["σ"]
        L --> M["×"]
        M --> N["Frame Avgpool"]
        N --> O["1 2 3 ... T"]
        O --> P["Frame Avgpool"]
    end
    
    sub-Right
        Q["Slice"] --> R["1 2 3 ... T ×1"]
        R --> S["Conv"]
        S --> T["σ"]
        T --> U["SSM"]
        U --> V["+"]
        V --> W["×"]
        W --> X["Output"]
    end
    
    sub-Left
        Y["Input"] --> Z["Diff-fusion"]
        Z --> AA["⊕"]
        AA --> AB["σ"]
        AB --> AC["×"]
        AC --> AD["Frame Avgpool"]
        AD --> AE["1 2 3 ... T"]
        AE --> AF["Frame Avgpool"]
    end
    
    sub-Right
        AG["Input"] --> AH["Diff-fusion"]
        AH --> AI["⊕"]
        AI --> AJ["σ"]
        AJ --> AK["×"]
        AK --> AL["Frame Avgpool"]
        AL --> AM["1 2 3 ... T"]
        AM --> AN["Frame Avgpool"]
    end
    
    sub-Left
        AO["Input"] --> AP["Diff-fusion"]
        AP --> AQ["⊕"]
        AQ --> AR["σ"]
        AR --> AS["×"]
        AS --> AT["Frame Avgpool"]
        AT --> AU["1 2 3 ... T"]
        AU --> AV["Frame Avgpool"]
    end
    
    sub-Right
        AW["Input"] --> AX["Diff-fusion"]
        AX --> AY["⊕"]
        AY --> AZ["σ"]
        AZ --> BA["×"]
        BA --> BB["Frame Avgpool"]
        BB --> BC["1 2 3 ... T"]
        BC --> BD["Frame Avgpool"]
    end
    
    sub-Left
        BE["Input"] --> BF["Diff-fusion"]
        BF --> BG["⊕"]
        BG --> BH["σ"]
        BH --> BI["×"]
        BI --> BJ["Frame Avgpool"]
        BJ --> BK["1 2 3 ... T"]
        BK --> BL["Frame Avgpool"]
    end
    
    sub-Right
        BM["Input"] --> BN["Diff-fusion"]
        BN --> BO["⊕"]
        BO --> BP["σ"]
        BP --> BQ["×"]
        BQ --> BR["Frame Avgpool"]
        BR --> BS["1 2 3 ... T"]
        BS --> BT["Frame Avgpool"]
    end
    
    sub-Left
        BU["Input"] --> BV["Diff-fusion"]
        BV --> BW["⊕"]
        BW --> BX["σ"]
        BX --> BY["×"]
        BY --> BZ["Frame Avgpool"]
        BZ --> CA["1 2 3 ... T"]
        CA --> CB["Frame Avgpool"]
    end
    
    sub-Right
        CC["Input"] --> DD["Diff-fusion"]
        DD --> DE["T ×1"]
        DE --> DF["Conv"]
        DF --> DG["σ"]
        DG --> DH["SSM"]
        DH --> DI["+"]
        DI --> DJ["×"]
        DJ --> DK["Output"]
    end
    
    sub-Left
        EE["Input"] --> EF["Slice"]
        EF --> GF["T ×2"]
        GF --> GH["Conv"]
        GH --> ID["σ"]
        ID --> DJ
        DJ --> DK
        DK --> DL["Share Weights"]
        DL --> DV["×4"]
        DV --> DW["Conv"]
        DW --> DX["σ"]
        DX --> DY["SMM"]
        DY --> DB
        DB --> DC
        DC --> DV
    end
    
    sub-Right
        DD
        DE
        GF
        DH
        ID
        DJ
    end
    
    sub-Right
    AE["Input"] --> AF
    AF --> AF
    AF --> AG["T ×N"]
    AG --> AH["RPPG Signal"]

    sub-Left
    AE --> AI["Slice"]
    AI --> AJ["T ×N"]
    AJ --> AK["SMM"]
    AK --> AL["Diverging to Output"]

    sub-Right
    AE --> AM["Slice"]
    AM --> AN["T ×N"]
    AN --> AO["SMM"]
    AO --> AP["Diverging to Output"]

    sub-Left
    AE --> AQ["Slice"]
    AQ --> AR["T ×N"]
    AR --> AS["SMM"]
    AS --> AT["Diverging to Output"]

    sub-Right
    AE --> AU["Slice"]
    AU --> AV["T ×N"]
    AV --> AW["SMM"]
    AW --> AX["Diverging to Output"]

    sub-Left
    AE --> AX
    AX --> AY["SMM"]
    AY --> AZ["Diverging to Output"]

    sub-Right
    AE --> BA["Slice"]
    BA --> BB["T ×N"]
    BB --> BC["SMM"]
    BC --> BD["Diverging to Output"]

    sub-Left
    AE --> BA
    BA --> AC["SMM"]
    AC --> AD["Diverging to Output"]

    sub-Right
    AE --> AD
    AD --> AE
```
</details>

Figure 3: The framework of RhythmMamba. It consists of frame stem, multi-temporal constraint Mamba, frequency domain feed-forward, and rPPG predictor head. Where “+” represents addition, “×” represents multiplication, “σ” represents the activation layer, and trapezoid represents the linear layer.

# 3 Methodology

Section 3.1 introduces the general framework of Rhythm-Mamba, followed by the presentation of its main components: the frame stem in Section 3.2, the multi-temporal constraint Mamba in Section 3.3, and lastly, the frequency domain feed-forward in Section 3.4.

# 3.1 The General Framework of RhythmMamba

As shown in Figure 3, RhythmMamba consists of frame stem, multi-temporal constraint Mamba, frequency domain feed-forward, and rPPG predictor head. The frame stem utilizes diff-fusion, self-attention, and frame average pooling to extract rPPG features and embed all spatial information into channels. Specifically, given an RGB video input $X \in \mathbb{R}^{3 \times T \times H \times W}$ , $X_{stem} = \text{frame\_stem}(X)$ , where $X_{stem} \in \mathbb{R}^{T \times C}$ , and C, T, W, H indicate channel, sequence length, width, and height, respectively.

Subsequently, the output of the frame stem will be fed into the multi-temporal constraint Mamba. The tokens will be sliced into sequences of varying temporal lengths, followed by the processing of hidden information between tokens with the SSM. Then, the output of Mamba will be passed into the frequency domain feed-forward, facilitating the interaction of information across multiple channels. The outputs of Mamba and feed-forward network (FFN) will undergo normalization and residual connections. The two outputs have dimensions identical to the output of the frame

stem $X_{stem} \in R^{T \times C}$ . Finally, the rPPG features will be projected into PPG waves through the predictor head.

# 3.2 Frame Stem

In the field of video understanding, existing transformer-based and Mamba-based methods typically embed spatiotemporal information into token sequences through patch embedding (Arnab et al. 2021; Zhu et al. 2024; Dosovitskiy et al. 2020). Previous works on rPPG have also been based on such foundational models for improvements. For transformer-based methods, spatiotemporal token sequences can inspire long-range spatiotemporal attention both within frames and across frames. However, we found that for linear Mamba, spatial information may interfere with Mamba's understanding of temporal sequences [see section 4.5]. The frame stem is utilized to initially extract rPPG features and embed spatial information fully into the channels, thereby boosting the learning of state transitions in the multi-temporal constraint Mamba and the channel interactions in the frequency domain feed-forward.

Firstly, the diff-fusion module (Zou et al. 2024) integrates frame differences into the raw frames, enabling frame-level representation awareness of BVP wave variations. This effectively enhances the features of rPPG with a small additional computational cost. Additionally, for rPPG, high-frequency information across frames and low-frequency information within frames are required. Therefore, relatively

large convolutional kernels are used to obtain low-frequency information within frames, ensuring that spatial information is fully incorporated into the channels. Here, 'relatively large' refers to the size relative to the image resolution, enabling a large receptive field.

Specifically, for an RGB video input $X \in R^{3 \times T \times H \times W}$ , temporal shift is initially applied to obtain $X_{t-2}$ , $X_{t-1}$ , $X_{t}$ , $X_{t+1}$ and $X_{t+2}$ . Subsequently, frame differences between consecutive frames are computed in reverse chronological order, yielding $D_{t-2}$ , $D_{t-1}$ , $D_{t+1}$ , and $D_{t+2}$ . The frame differences and the raw frames are then passed through $Stem_{1}$ for feature extraction. $Stem_{1}$ consists of a 2D convolution layer with $(7 \times 7)$ kernel, followed by batch normalization (BN), ReLU, and MaxPool. The input dimension is 3 when taking raw frames as input, and 12 when taking the concatenation of frame differences as input.

$$
\begin{array}{l} \begin{array}{l} X _ {d i f f} = \operatorname{Stem} _ {1} \left(\text {Concat} \left(D _ {t - 2}, D _ {t - 1}, D _ {t + 1}, D _ {t + 2}\right)\right), \\ \mathbf {Y} = \operatorname{Stem} _ {1} (\mathbf {Y}) \end{array} \tag {1} \\ X _ {r a w} = S t e m _ {1} (X _ {t}). \\ \end{array}
$$

Then frame differences $X_{diff}$ and raw frames $X_{raw}$ are merged and the feature representation is further enhanced through $Stem_{2}$ , which consists of a 2D convolution layer with $(7 \times 7)$ kernel, followed by BN, ReLU and MaxPool.

$$
X _ {f u s i o n} = \operatorname{Stem} _ {2} \left(X _ {\text { raw }} + X _ {\text { diff }}\right) + \operatorname{Stem} _ {2} \left(X _ {\text { diff }}\right). \tag {2}
$$

Subsequently, at the resolution of $(16 \times 16)$ , $Stem_{3}$ utilizes a convolution layer with $(5 \times 5)$ kernel to fully integrate spatial information into the channels, followed by BN. Before frame-level global average pooling, a self-attention module is employed to enhance skin regions with rPPG signals in the spatial domain. This self-attention module utilizes sigmoid activation followed by L1 normalization, which is softer than softmax and generates fewer masks (Liu et al. 2023a). The attention mask can be computed as:

$$
\text { Mask } = \frac {(H / 8) (W / 8) \cdot \sigma (\text { Stem } _ {3} (X _ {\text { fusion }}))}{2 | | \sigma (\text { Stem } _ {3} (X _ {\text { fusion }})) | | _ {1}}. \tag {3}
$$

Finally, the attention output $X_{attn} \in R^{C \times T \times H/8 \times W/8}$ , undergoes global average pooling within each frame, resulting in the stem output $X_{stem} \in R^{T \times C}$ .

# 3.3 Multi-temporal Constraint Mamba

Previous studies (Hu et al. 2022; Kong, Bian, and Jiang 2022; Dai et al. 2022) have shown the effectiveness of modeling periodic tasks with multi-temporal scales, primarily achieved by the extraction and fusion of features from different temporal scales. Unlike these studies, we replace multi-temporal fusion with multi-temporal constraint, which better aligns with the characteristics of the Mamba. Specifically, we slice a video segment into numerous sub-segments of varying lengths to constrain a single Mamba block, rather than downsampling the video segment to different resolutions and using multiple Mamba blocks to extract and fuse features. The aim is to subject a Mamba block to both the periodic constraints of long sequences and the trend constraints of short sequences simultaneously, instead of extracting different features from multi-temporal scales.

After the frame stem, the token sequence can be regarded as a time series, and the state transition of Mamba can be interpreted as the temporal phase shift. Due to the quasi-periodicity of rPPG signals, the signal can be represented using a finite set of states. Specifically, as illustrated by the multi-temporal constraint Mamba in Figure 3, the input $X_{stem}$ is first linearly projected and then processed through three weight-shared paths. Along these paths, the input is sliced into temporal sequences of varying lengths. For the $i_{th}$ path, the sequence is divided into $2^{i-1}$ sub-sequences, each of which undergoes sequential processing through a convolution layer, activation layer, and selective state space model (see supplementary material A for details). Subsequently, they are recombined into a sequence of the original length, forming the output of the $i_{th}$ path, denoted as $X_{path_{i}}$ . The output before projection can be represented as follows:

$$
X _ {m a m b a} = \sum_ {i = 1} ^ {3} X _ {p a t h _ {i}} \times \sigma (P r o j (X _ {s t e m})). \tag {4}
$$

# 3.4 Frequency Domain Feed-forward

The FFN employs linear transformations to project data into a higher-dimensional space before mapping it back into a lower-dimensional space. Through this channel interaction, deeper features are extracted. In previous rPPG studies, spatio-temporal FFN was frequently applied, which introduced a depthwise 3D convolution layer between the two linear layers of the vanilla FFN, to refine the local inconsistency and provide relative positional cues. (Yu et al. 2022).

In our study, due to the input sequences being solely time-dependent, channel interaction in the frequency domain enables a better discernment of the periodic nature of rPPG signals. So we introduce frequency domain feed-forward, which adds a frequency domain linear layer between the two linear layers of the vanilla FFN. The frequency domain linear layer consists of three stages: domain conversion, frequency domain channel interaction, and domain inversion.

Domain Conversion/Inversion. Domain conversion and inversion utilize fast Fourier transform and inverse Fourier transform, respectively. The utilization of the Fourier transform enables the decomposition of rPPG signals into their constituent frequencies, facilitating the recognition of periodic patterns in the rPPG signals. We transform the input $H(t)$ to the frequency domain $H(f)$ as follows:

$$
\begin{array}{l} H (f) = \int_ {- \infty} ^ {\infty} H (t) e ^ {- j 2 \pi f t} d t \\ = \int_ {- \infty} ^ {\infty} H (t) \cos (2 \pi f t) d t + j \int_ {- \infty} ^ {\infty} H (t) \sin (2 \pi f t) d t \tag {5} \\ = H (f) _ {r e} + j H (f) _ {i m}. \\ \end{array}
$$

Where f represents frequency, t represents time, and the subscripts re and im denote the real and imaginary components of the corresponding complex data, respectively. After channel Interaction in the frequency domain, inverse Fourier transform is employed to revert to the temporal domain:

$$
\begin{array}{l} H (t) = \int_ {- \infty} ^ {\infty} H (f) e ^ {j 2 \pi f t} d f \tag {6} \\ = \int_ {- \infty} ^ {\infty} (H (f) _ {r e} + j H (f) _ {i m}) e ^ {j 2 \pi f t} d f. \\ \end{array}
$$

Frequency Domain Channel Interaction. Through the frame stem, spatial information is embedded into the channels, each of which is treated as a time series. Consequently,

<table><tr><td rowspan="2">Method</td><td colspan="3">PURE</td><td colspan="3">UBFC</td><td colspan="3">VIPL-HR</td><td colspan="3">MMPD</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td> $\rho \uparrow$ </td><td>MAE↓</td><td>RMSE↓</td><td> $\rho \uparrow$ </td><td>MAE↓</td><td>RMSE↓</td><td> $\rho \uparrow$ </td><td>MAE↓</td><td>RMSE↓</td><td> $\rho \uparrow$ </td></tr><tr><td>HR-CNN</td><td>1.84</td><td>2.37</td><td>0.98</td><td>4.90</td><td>5.89</td><td>0.64</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DeepPhys</td><td>0.83</td><td>1.54</td><td>0.99</td><td>6.27</td><td>10.82</td><td>0.65</td><td>11.00</td><td>13.80</td><td>0.11</td><td>22.27</td><td>28.92</td><td>-0.03</td></tr><tr><td>PhysNet</td><td>2.10</td><td>2.60</td><td>0.99</td><td>2.95</td><td>3.67</td><td>0.97</td><td>10.80</td><td>14.80</td><td>0.20</td><td>4.80</td><td>11.80</td><td>0.60</td></tr><tr><td>TS-CAN</td><td>2.48</td><td>9.01</td><td>0.92</td><td>1.70</td><td>2.72</td><td>0.99</td><td>-</td><td>-</td><td>-</td><td>9.71</td><td>17.22</td><td>0.44</td></tr><tr><td>Gideon et al.</td><td>2.30</td><td>2.90</td><td>0.99</td><td>1.85</td><td>4.28</td><td>0.93</td><td>9.01</td><td>14.02</td><td>0.58</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Dual-GAN</td><td>0.82</td><td>1.31</td><td>0.99</td><td>0.44</td><td>0.67</td><td>0.99</td><td>4.93</td><td>7.68</td><td>0.81</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PhysFormer</td><td>1.10</td><td>1.75</td><td>0.99</td><td>0.50</td><td>0.71</td><td>0.99</td><td>4.97</td><td>7.79</td><td>0.78</td><td>11.99</td><td>18.41</td><td>0.18</td></tr><tr><td>EfficientPhys</td><td>-</td><td>-</td><td>-</td><td>1.14</td><td>1.81</td><td>0.99</td><td>-</td><td>-</td><td>-</td><td>13.47</td><td>21.32</td><td>0.21</td></tr><tr><td>TFA-PFE</td><td>1.44</td><td>2.50</td><td>-</td><td>0.76</td><td>1.62</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>NEST</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.76</td><td>7.51</td><td>0.84</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Li et al.</td><td>0.64</td><td>1.16</td><td>0.99</td><td>0.48</td><td>0.64</td><td>0.99</td><td>5.19</td><td>8.26</td><td>0.78</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Yue et al.</td><td>1.23</td><td>2.01</td><td>0.99</td><td>0.58</td><td>0.94</td><td>0.99</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PhysFormer++</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.88</td><td>7.62</td><td>0.80</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Contrast-Phys+</td><td>0.48</td><td>0.98</td><td>0.99</td><td>0.21</td><td>0.80</td><td>0.99</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RhythmFormer</td><td>0.27</td><td>0.47</td><td>0.99</td><td>0.50</td><td>0.78</td><td>0.99</td><td>-</td><td>-</td><td>-</td><td>3.07</td><td>6.81</td><td>0.86</td></tr><tr><td>Ours</td><td>0.23</td><td>0.34</td><td>0.99</td><td>0.50</td><td>0.75</td><td>0.99</td><td>4.30</td><td>7.49</td><td>0.81</td><td>3.16</td><td>7.27</td><td>0.84</td></tr></table>

Table 1: Intra-dataset evaluation. Best results are marked in bold and second best in underline.

the frequency domain features obtained after domain conversion can clearly represent the signal's frequency composition. This allows channel interactions in the frequency domain to refine noise interference and more easily focus on critical channels. Specifically, channel interaction is implemented through a linear layer, the theoretical feasibility of which has been demonstrated by (Yi et al. 2024). For complex input $H \in \mathbb{R}^{T \times C}$ , given complex weight matrix $W \in \mathbb{R}^{C \times C}$ and complex bias $B \in \mathbb{R}^C$ , according to the rules of complex multiplication, it can be expressed as:

$$
H _ {r e} ^ {\prime} = H _ {r e} W _ {r e} - H _ {i m} W _ {i m} + B _ {r e},
$$

$$
H _ {i m} ^ {\prime} = H _ {r e} W _ {i m} + H _ {i m} W _ {r e} + B _ {i m}, \tag {7}
$$

$$
H ^ {\prime} = H _ {r e} ^ {\prime} + j \cdot H _ {i m} ^ {\prime}.
$$

After inverse FFT transform, the frequency domain feedforward outputs through linear projection, resulting in $X_{FFN} \in R^{T \times C}$ .

# 4 Experiment

# 4.1 Dataset and Performance Metric

The experiments of remote physiological measurement were conducted on four publicly available datasets: PURE (Stricker, Müller, and Gross 2014), UBFC-rPPG (Bobbia et al. 2019), VIPL-HR (Niu et al. 2019), and MMPD (Tang et al. 2023). PURE comprises 59 1-minute videos, documenting records of 10 subjects, each engaging in six different activities. UBFC-rPPG consists of 42 videos, recording 42 subjects. These videos were derived from a setting where subjects participated in a time-limited digital game. VIPL-HR includes 2,378 RGB videos from 107 participants, captured using three RGB cameras, with an unstable fps. MMPD includes 660 1-minute videos, documenting records of 33 subjects. Participants engaged in four different activities under four distinct lighting conditions. Metircs. The evaluation was conducted using five metrics for video-level heart rate estimations: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), Pearson Correlation Coefficient ( $\rho$ ), and Signal-to-Noise Ratio (SNR).

# 4.2 Implementation Details

The proposed RhythmMamba was implemented based on PyTorch, and we utilized an open-source rPPG toolbox (Liu et al. 2023b) to conduct a fair comparison against several state-of-the-art methods. In the pre-processing, video inputs were divided into segments of 160 frames. Facial recognition was applied on the first frame of each segment, followed by cropping and resizing of the facial region. These adjustments were then maintained throughout the subsequent frames. In the post-processing, a second-order Butterworth filter (cutoff frequencies: 0.75 and 2.5 Hz) was applied to filter the rPPG waveform, and power spectral density was computed by the Welch algorithm for further heart rate estimation. Following the protocol outlined in (Yu et al. 2020), random upsampling, downsampling, and horizontal flipping were applied for data augmentation. The experiment was conducted on NVIDIA RTX 3090.

Loss. We employed a loss function that integrates constraints from both the temporal and frequency domains (Yu et al. 2020). The negative Pearson correlation coefficient is utilized as temporal constraint $L_{Time}$ , while cross-entropy between the power spectral density of prediction and the HR derived from the power spectral density of ground truth, is employed as frequency constraint $L_{Freq}$ . $\mathcal{L}_{Freq} = CE(maxIndex(PSD(PPG_{gt})), PSD(PPG_{pred}))$ , where PSD represents Power Spectral Density and maxIndex represents the index of the maximum value. The overall loss is expressed by: $L_{overall} = a \cdot L_{Time} + b \cdot L_{Freq}$ .

Comparison. We compare our method with state-of-the-art approaches in intra-dataset testing (Špetlík, Franc, and Matas 2018; Chen and McDuff 2018; Yu, Li, and Zhao 2019; Liu et al. 2020; Gideon and Stent 2021; Lu, Han, and Zhou 2021; Yu et al. 2022; Liu et al. 2023a; Li, Yu, and Shi 2023; Lu et al. 2023; Li and Yin 2023; Yue, Shi, and Ding 2023; Yu et al. 2023; Sun and Li 2024; Zou et al. 2024). Building on this, additional comparisons with (Verkruysse, Svaasand, and Nelson 2008; Poh, McDuff, and Picard 2010; De Haan and Jeanne 2013; Pilz et al. 2018; De Haan and Van Leest

<table><tr><td rowspan="3">Method</td><td rowspan="3">TrainSet</td><td colspan="15">Test Set</td></tr><tr><td colspan="5">PURE</td><td colspan="5">UBFC</td><td colspan="5">MMPD</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE</td><td> $\rho$ </td><td>SNR</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td> $\rho$ </td><td>SNR</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td> $\rho$ </td><td>SNR</td></tr><tr><td>GREEN</td><td>-</td><td>10.09</td><td>23.85</td><td>10.28</td><td>0.34</td><td>-2.66</td><td>19.73</td><td>31.00</td><td>18.72</td><td>0.37</td><td>-11.18</td><td>21.68</td><td>27.69</td><td>24.39</td><td>-0.01</td><td>-14.34</td></tr><tr><td>ICA</td><td>-</td><td>4.77</td><td>16.07</td><td>4.47</td><td>0.72</td><td>5.24</td><td>16.00</td><td>25.65</td><td>15.35</td><td>0.44</td><td>-9.91</td><td>18.60</td><td>24.30</td><td>20.88</td><td>0.01</td><td>-13.84</td></tr><tr><td>CHROM</td><td>-</td><td>5.77</td><td>14.93</td><td>11.52</td><td>0.81</td><td>4.58</td><td>4.06</td><td>8.83</td><td>3.84</td><td>0.89</td><td>-2.96</td><td>13.66</td><td>18.76</td><td>16.00</td><td>0.08</td><td>-11.74</td></tr><tr><td>LGI</td><td>-</td><td>4.61</td><td>15.38</td><td>4.96</td><td>0.77</td><td>4.50</td><td>15.80</td><td>28.55</td><td>14.70</td><td>0.36</td><td>-8.15</td><td>17.08</td><td>23.32</td><td>18.98</td><td>0.04</td><td>-13.15</td></tr><tr><td>PBV</td><td>-</td><td>3.92</td><td>12.99</td><td>4.84</td><td>0.84</td><td>2.30</td><td>15.90</td><td>26.40</td><td>15.17</td><td>0.48</td><td>-9.16</td><td>17.95</td><td>23.58</td><td>20.18</td><td>0.09</td><td>-13.88</td></tr><tr><td>POS</td><td>-</td><td>3.67</td><td>11.82</td><td>7.25</td><td>0.88</td><td>6.87</td><td>4.08</td><td>7.72</td><td>3.93</td><td>0.92</td><td>-2.39</td><td>12.36</td><td>17.71</td><td>14.43</td><td>0.18</td><td>-11.53</td></tr><tr><td>OMIT</td><td>-</td><td>4.66</td><td>15.82</td><td>4.97</td><td>8.76</td><td>4.37</td><td>16.99</td><td>29.54</td><td>15.91</td><td>0.34</td><td>-7.29</td><td>17.02</td><td>23.23</td><td>18.89</td><td>0.04</td><td>-12.77</td></tr><tr><td rowspan="2">DeepPhys</td><td>UBFC</td><td>5.54</td><td>18.51</td><td>5.32</td><td>0.66</td><td>4.40</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>17.50</td><td>25.00</td><td>19.27</td><td>0.06</td><td>-11.72</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.21</td><td>2.90</td><td>1.42</td><td>0.99</td><td>1.74</td><td>16.92</td><td>24.61</td><td>18.54</td><td>0.05</td><td>-11.53</td></tr><tr><td rowspan="2">PhysNet</td><td>UBFC</td><td>8.06</td><td>19.71</td><td>13.67</td><td>0.61</td><td>6.68</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>9.47</td><td>16.01</td><td>11.11</td><td>0.31</td><td>-8.15</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.98</td><td>2.48</td><td>1.12</td><td>0.99</td><td>1.49</td><td>13.94</td><td>21.61</td><td>15.15</td><td>0.20</td><td>-9.94</td></tr><tr><td rowspan="2">TS-CAN</td><td>UBFC</td><td>3.69</td><td>13.80</td><td>3.39</td><td>0.82</td><td>5.26</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>14.01</td><td>21.04</td><td>15.48</td><td>0.24</td><td>-10.18</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.30</td><td>2.87</td><td>1.50</td><td>0.99</td><td>1.49</td><td>13.94</td><td>21.61</td><td>15.15</td><td>0.20</td><td>-9.94</td></tr><tr><td rowspan="2">PhysFormer</td><td>UBFC</td><td>12.92</td><td>24.36</td><td>23.92</td><td>0.47</td><td>2.16</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>12.10</td><td>17.79</td><td>15.41</td><td>0.17</td><td>-10.53</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.44</td><td>3.77</td><td>1.66</td><td>0.98</td><td>0.18</td><td>14.57</td><td>20.71</td><td>16.73</td><td>0.15</td><td>-12.15</td></tr><tr><td rowspan="2">EfficientPhys</td><td>UBFC</td><td>5.47</td><td>17.04</td><td>5.40</td><td>0.71</td><td>4.09</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>13.78</td><td>22.25</td><td>15.15</td><td>0.09</td><td>-9.13</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.07</td><td>6.32</td><td>2.10</td><td>0.94</td><td>-0.12</td><td>14.03</td><td>21.62</td><td>15.32</td><td>0.17</td><td>-9.95</td></tr><tr><td rowspan="2">Spiking-Phys.</td><td>UBFC</td><td>3.83</td><td>-</td><td>5.70</td><td>0.83</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>14.15</td><td>-</td><td>16.22</td><td>0.15</td><td>-</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.80</td><td>-</td><td>2.81</td><td>0.95</td><td>-</td><td>14.57</td><td>-</td><td>16.55</td><td>0.14</td><td>-</td></tr><tr><td rowspan="2">RhythmFormer</td><td>UBFC</td><td>0.97</td><td>3.36</td><td>1.60</td><td>0.99</td><td>12.01</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>9.08</td><td>15.07</td><td>11.17</td><td>0.53</td><td>-7.73</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.89</td><td>1.83</td><td>0.97</td><td>0.99</td><td>6.05</td><td>8.98</td><td>14.85</td><td>11.11</td><td>0.51</td><td>-8.39</td></tr><tr><td rowspan="2">Ours</td><td>UBFC</td><td>1.98</td><td>6.51</td><td>3.59</td><td>0.96</td><td>8.94</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>10.63</td><td>17.14</td><td>12.14</td><td>0.34</td><td>-8.28</td></tr><tr><td>PURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.95</td><td>1.83</td><td>1.04</td><td>0.99</td><td>6.35</td><td>10.44</td><td>16.70</td><td>12.25</td><td>0.36</td><td>-8.18</td></tr></table>

Table 2: Cross-dataset evaluation. Best results are marked in bold and second best in underline.

2014; Wang et al. 2016; Casado and López 2023; Liu et al. 2024) are presented in cross-dataset testing.

# 4.3 Intra-Dataset Evaluation

We conducted intra-dataset evaluation on the PURE and UBFC datasets to validate the feasibility of the Mamba architecture. For the evaluation of the PURE dataset, we followed the protocols outlined in (Lu, Han, and Zhou 2021), splitting the dataset sequentially into training and testing sets with a ratio of 6:4. Similarly, for the evaluation of the UBFC dataset, we followed the protocols in (Lu, Han, and Zhou 2021), selecting the first 30 samples as the training set and the remaining 12 samples as the testing set. Due to the absence of the validation set, we selected the checkpoint from the last epoch for testing and compared them with the reported results from previous methods. As shown in Table 1, on the PURE dataset, our method outperformed all state-of-the-art methods across all metrics, achieving the minimum MAE (0.23) and RMSE (0.34). On the UBFC dataset, our method also achieved comparable performance to others.

Due to the relative simplicity of PURE and UBFC, the performance of state-of-the-art methods on these datasets is nearing saturation. To further evaluate the performance, we employed the more challenging dataset. For the VIPL-HR dataset, we followed the subject-exclusive 5-fold cross-validation protocol, as outlined in (Niu et al. 2019; Yu et al. 2022). For the MMPD dataset, following the protocols outlined in (Zou et al. 2024), the dataset was sequentially split into training, validation, and testing sets with a ratio of 7:1:2. As shown in Table 1, our method achieved comparable performance to the previous methods. This indicates that RhythmMamba can accurately extract weak rPPG signals and understand their periodic nature, which provides ample empirical evidence for the feasibility of the Mamba architecture in the rPPG task.

# 4.4 Cross-Dataset Evaluation

To objectively evaluate the generalization capability to out-of-distribution data, we followed the protocols outlined in (Liu et al. 2023b) for cross-dataset evaluation. The models were trained on either the PURE or UBFC datasets and tested on the PURE, UBFC, and MMPD datasets. The training dataset was sequentially split into training and validation sets with a ratio of 8:2. All comparative methods were implemented based on the rPPG toolbox (Liu et al. 2023b). As shown in Table 2, the proposed RhythmMamba also achieved SOTA performance, demonstrating its capability in modeling domain-invariant features and generalizing to unseen domains. Based on the comparisons in Tables 1 and 2, the improvement in cross-dataset results is less significant compared to intra-dataset results, possibly due to the fine-grained token-wise self-attention, which may have an advantage in capturing domain-invariant features. Nevertheless, founded on fewer parameters and lower computational complexity, RhythmMamba showcases its potential in real-world applications through its robustness and generalization in complex environments. Additional visualization results can be found in Supplementary Material B.

# 4.5 Ablation Study

Ablation studies were conducted on the MMPD dataset to assess the impact of different modules.

<table><tr><td>Diff-fusion</td><td>Self-attention</td><td>Large Kernel</td><td>Multi-temporal</td><td>FFN</td><td>MAE↓</td><td>RMSE↓</td><td>MAPE↓</td><td> $\rho \uparrow$ </td><td>SNR↑</td></tr><tr><td>×</td><td>√</td><td>√</td><td>√</td><td>Frequency</td><td>5.71</td><td>10.00</td><td>5.97</td><td>0.68</td><td>-0.29</td></tr><tr><td>√</td><td>×</td><td>√</td><td>√</td><td>Frequency</td><td>4.02</td><td>8.43</td><td>4.15</td><td>0.79</td><td>2.98</td></tr><tr><td>√</td><td>√</td><td>×</td><td>√</td><td>Frequency</td><td>3.51</td><td>7.53</td><td>3.81</td><td>0.83</td><td>4.22</td></tr><tr><td>√</td><td>√</td><td>√</td><td>×</td><td>Frequency</td><td>3.60</td><td>7.78</td><td>3.83</td><td>0.81</td><td>4.38</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>3.53</td><td>7.65</td><td>3.67</td><td>0.83</td><td>2.89</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>Vanilla</td><td>3.54</td><td>7.68</td><td>3.72</td><td>0.82</td><td>2.88</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>Spatio-Temporal</td><td>3.82</td><td>8.06</td><td>3.95</td><td>0.81</td><td>4.06</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>Frequency</td><td>3.16</td><td>7.27</td><td>3.37</td><td>0.84</td><td>4.74</td></tr></table>

Table 3: Impact of key modules.

<table><tr><td>Spatial Token numbers</td><td>MAE↓</td><td>RMSE↓</td><td>MAPE↓</td><td> $\rho \uparrow$ </td><td>SNR↑</td></tr><tr><td>8×8</td><td>4.90</td><td>10.14</td><td>5.15</td><td>0.71</td><td>-2.10</td></tr><tr><td>4×4</td><td>4.62</td><td>8.87</td><td>4.83</td><td>0.77</td><td>-0.88</td></tr><tr><td>4×4 (Temporal Embed)</td><td>4.92</td><td>10.04</td><td>5.07</td><td>0.69</td><td>-1.43</td></tr><tr><td>4×4 (Position Embed)</td><td>5.47</td><td>10.34</td><td>5.77</td><td>0.71</td><td>-3.03</td></tr><tr><td>2×2</td><td>4.69</td><td>9.97</td><td>4.80</td><td>0.70</td><td>-0.98</td></tr><tr><td>1×1 (Avgpool)</td><td>3.54</td><td>7.68</td><td>3.72</td><td>0.82</td><td>2.88</td></tr></table>

Table 4: Impact of spatial information (with vanilla FFN).

Impact of Spatial Information. As shown in Table 4, the ablation study of spatial information is presented, where both position embedding and temporal embedding were implemented using learnable parameters. For a fair comparison, the vanilla FFN was used, as the frequency domain FFN might have an advantage with purely temporal token sequences. It is evident that tokenized spatiotemporal information performs poorly, even with temporal embedding or position embedding. The integration of spatial information increases the dimensionality of the state transition process, thereby elevating complexity and making the model more difficult to train. We view the rPPG task as a time series task, embedding spatial information into the channels, with each channel being treated as a purely temporal sequence. This ensures that the state transition process occurs purely along the temporal dimension, while spatial information interactions are facilitated through subsequent channel interactions, effectively resolving this issue.

Impact of Key Modules. As illustrated in Table 3, the comparison between the first four rows and the last row indicates the significant roles played by these modules. The diffusion module enables frame-level representation awareness of BVP wave variations, effectively enhancing rPPG features with a small additional computational cost. The use of relatively large convolution kernels and self-attention allows for the integration of spatial information into channels effectively, thereby providing sufficient information for subsequent processing. The multi-temporal constraint Mamba constrains a single Mamba block simultaneously to short-term trends and periodic patterns, facilitating the accurate comprehension of rPPG features.

Impact of Frequency Domain Feed-forward. As shown in Table 3, the last four rows show that the frequency domain FFN plays an important role. Among them, vanilla FFN refers to using two linear layers to compose the FFN, and Spatio-Temporal FFN refers to the addition of a depth-

<table><tr><td>Method</td><td>Para.</td><td>MACs</td><td>Throughput</td><td>Memory</td></tr><tr><td>DeepPhys</td><td>1.98</td><td>744.45</td><td>5.65</td><td>37.28</td></tr><tr><td>PhysNet</td><td>0.77</td><td>438.24</td><td>11.80</td><td>11.43</td></tr><tr><td>TS-CAN</td><td>1.98</td><td>744.45</td><td>5.21</td><td>38.91</td></tr><tr><td>PhysFormer</td><td>7.38</td><td>316.29</td><td>8.50</td><td>28.63</td></tr><tr><td>EfficientPhys</td><td>1.91</td><td>373.72</td><td>8.42</td><td>26.68</td></tr><tr><td>RhythmFormer</td><td>3.25</td><td>240.55</td><td>6.30</td><td>29.06</td></tr><tr><td>Ours</td><td>1.07</td><td>80.90</td><td>20.09</td><td>6.66</td></tr></table>

Table 5: Computational cost. The horizontal axis represents Parameters (M), MACs (M), Throughput (Kfps), and Peak GPU Memory Usage (M).

wise convolution layer between the linear layers (Yu et al. 2022). Frequency domain FFN adds a frequency domain linear layer between the linear layers, effectively extracting the most critical frequency domain features in rPPG signals.

# 4.6 Computational Cost

We conducted a 30-second inference test at a resolution of $128 \times 128$ , reporting the parameters, average MACs per frame, average throughput per frame, and average peak GPU memory usage per frame. As shown in Table 5 and Figure 2, RhythmMamba achieved 319% throughput and 23% peak GPU memory, demonstrating the potential for effective mobile-level rPPG applications. Building on this advantage, we also found that the proposed method can accept inputs of arbitrary length during inference without any performance degradation (see Supplementary Material C for details).

# 5 Conclusion

We approach the rPPG task as a time series task, designing multiple modules that align with the temporal characteristics of rPPG signals to boost state space model learning. This approach boasts strong long-range dependency modeling capabilities while maintaining linear complexity. It achieves state-of-the-art performance both within and across datasets with a faster and more lightweight design. However, since Mamba's state transitions align closely with the periodic variations of rPPG signals, we believe that Mamba's potential in rPPG extends beyond the current results. Our utilization of periodic priors is currently limited and we would like to delve into this more deeply in the future.

# Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (62206015, 62227801, U21B2048), the National Science and Technology Major Project (2022ZD0117901), and the Fundamental Research Funds for the Central Universities (FRF-TP-22-043A1).

# References

Arnab, A.; Dehghani, M.; Heigold, G.; Sun, C.; Lučić, M.; and Schmid, C. 2021. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 6836–6846.   
Bobbia, S.; Macwan, R.; Benezeth, Y.; Mansouri, A.; and Dubois, J. 2019. Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern Recognition Letters, 124: 82–90.   
Casado, C. A.; and López, M. B. 2023. Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. IEEE Journal of Biomedical and Health Informatics.   
Chen, W.; and McDuff, D. 2018. Deepphys: Video-based physiological measurement using convolutional attention networks. In Proceedings of the European Conference on Computer Vision (ECCV), 349–365.   
Choi, J.-H.; Kang, K.-B.; and Kim, K.-T. 2024. Fusion-Vital: Video-RF Fusion Transformer for Advanced Remote Physiological Measurement. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 1344–1352.   
Dai, R.; Das, S.; Kahatapitiya, K.; Ryoo, M. S.; and Brémond, F. 2022. MS-TCT: multi-scale temporal conv-transformer for action detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20041–20051.   
Dao, T.; and Gu, A. 2024. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. In International Conference on Machine Learning (ICML).   
De Haan, G.; and Jeanne, V. 2013. Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10): 2878–2886.   
De Haan, G.; and Van Leest, A. 2014. Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9): 1913.   
Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.   
Gideon, J.; and Stent, S. 2021. The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video. In Proceedings of the IEEE/CVF international conference on computer vision, 3995–4004.   
Hu, H.; Dong, S.; Zhao, Y.; Lian, D.; Li, Z.; and Gao, S. 2022. Transrac: Encoding multi-scale temporal correlation with transformers for repetitive action counting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19013–19022.

Kong, J.; Bian, Y.; and Jiang, M. 2022. MTT: Multi-scale temporal transformer for skeleton-based action recognition. IEEE Signal Processing Letters, 29: 528–532.   
Li, J.; Yu, Z.; and Shi, J. 2023. Learning motion-robust remote photoplethysmography through arbitrary resolution videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 1334–1342.   
Li, K.; Li, X.; Wang, Y.; He, Y.; Wang, Y.; Wang, L.; and Qiao, Y. 2025. Videomamba: State space model for efficient video understanding. In European Conference on Computer Vision, 237–255. Springer.   
Li, Z.; and Yin, L. 2023. Contactless Pulse Estimation Leveraging Pseudo Labels and Self-Supervision. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 20531–20540.   
Liu, M.; Tang, J.; Li, H.; Qi, J.; Li, S.; Wang, K.; Wang, Y.; and Chen, H. 2024. Spiking-PhysFormer: Camera-Based Remote Photoplethysmography with Parallel Spike-driven Transformer. arXiv:2402.04798.   
Liu, X.; Fromm, J.; Patel, S.; and McDuff, D. 2020. Multi-task temporal shift attention networks for on-device contactless vitals measurement. Advances in Neural Information Processing Systems, 33: 19400–19411.   
Liu, X.; Hill, B.; Jiang, Z.; Patel, S.; and McDuff, D. 2023a. Efficientphys: Enabling simple, fast and accurate camera-based cardiac measurement. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 5008–5017.   
Liu, X.; Narayanswamy, G.; Paruchuri, A.; Zhang, X.; Tang, J.; Zhang, Y.; Sengupta, S.; Patel, S.; Wang, Y.; and McDuff, D. 2023b. rPPG-Toolbox: Deep Remote PPG Toolbox. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.   
Lu, H.; Han, H.; and Zhou, S. K. 2021. Dual-gan: Joint bvp and noise modeling for remote physiological measurement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12404–12413.   
Lu, H.; Yu, Z.; Niu, X.; and Chen, Y.-C. 2023. Neuron Structure Modeling for Generalizable Remote Physiological Measurement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18589–18599.   
McDuff, D. 2023. Camera measurement of physiological vital signs. ACM Computing Surveys, 55(9): 1–40.   
Niu, X.; Han, H.; Shan, S.; and Chen, X. 2018. Synrhythm: Learning a deep heart rate estimator from general to specific. In 2018 24th International Conference on Pattern Recognition (ICPR), 3580–3585. IEEE.   
Niu, X.; Shan, S.; Han, H.; and Chen, X. 2019. Rhythmnet: End-to-end heart rate estimation from face via spatial-temporal representation. IEEE Transactions on Image Processing, 29: 2409–2423.   
Niu, X.; Yu, Z.; Han, H.; Li, X.; Shan, S.; and Zhao, G. 2020. Video-based remote physiological measurement via cross-verified feature disentangling. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, 295–310. Springer.

Patro, B. N.; and Agneeswaran, V. S. 2024. SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series. arXiv preprint arXiv:2403.15360.   
Pilz, C. S.; Zaunseder, S.; Krajewski, J.; and Blazek, V. 2018. Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 1254–1262.   
Poh, M.-Z.; McDuff, D. J.; and Picard, R. W. 2010. Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10):10762–10774.   
Shao, H.; Luo, L.; Qian, J.; Chen, S.; Hu, C.; and Yang, J. 2023. TranPhys: Spatiotemporal Masked Transformer Steered Remote Photoplethysmography Estimation. IEEE Transactions on Circuits and Systems for Video Technology.   
Špetlík, R.; Franc, V.; and Matas, J. 2018. Visual heart rate estimation with convolutional neural network. In Proceedings of the British Machine Vision Conference, Newcastle, UK, 3–6.   
Stricker, R.; Müller, S.; and Gross, H.-M. 2014. Non-contact video-based pulse rate measurement on a mobile service robot. In The 23rd IEEE International Symposium on Robot and Human Interactive Communication, 1056–1062. IEEE.   
Sun, Z.; and Li, X. 2024. Contrast-Phys+: Unsupervised and Weakly-Supervised Video-Based Remote Physiological Measurement via Spatiotemporal Contrast. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8):5835–5851.   
Tang, J.; Chen, K.; Wang, Y.; Shi, Y.; Patel, S.; McDuff, D.; and Liu, X. 2023. MMPD: Multi-Domain Mobile Video Physiology Dataset. In 45th Annual International Conference of the IEEE Engineering in Medicine and Biology Society.   
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30.   
Verkruysse, W.; Svaasand, L. O.; and Nelson, J. S. 2008. Remote plethysmographic imaging using ambient light. Optics express, 16(26): 21434–21445.   
Wang, W.; Den Brinker, A. C.; Stuijk, S.; and De Haan, G. 2016. Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7): 1479–1491.   
Yi, K.; Zhang, Q.; Fan, W.; Wang, S.; Wang, P.; He, H.; An, N.; Lian, D.; Cao, L.; and Niu, Z. 2024. Frequency-domain MLPs are more effective learners in time series forecasting. Advances in Neural Information Processing Systems, 36.   
Yu, Z.; Li, X.; Niu, X.; Shi, J.; and Zhao, G. 2020. Autohr: A strong end-to-end baseline for remote heart rate measurement with neural searching. IEEE Signal Processing Letters, 27: 1245–1249.   
Yu, Z.; Li, X.; and Zhao, G. 2019. Remote photoplethysmograph signal measurement from facial videos using spatiotemporal networks. In 30th British Machine Visison Conference: BMVC 2019. 9th-12th September 2019, Cardiff, UK. The British Machine Vision Conference (BMVC).

Yu, Z.; Peng, W.; Li, X.; Hong, X.; and Zhao, G. 2019. Remote heart rate measurement from highly compressed facial videos: an end-to-end deep learning solution with video enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 151–160.

Yu, Z.; Shen, Y.; Shi, J.; Zhao, H.; Cui, Y.; Zhang, J.; Torr, P.; and Zhao, G. 2023. Physformer++: Facial video-based physiological measurement with slowfast temporal difference transformer. International Journal of Computer Vision, 131(6): 1307–1330.

Yu, Z.; Shen, Y.; Shi, J.; Zhao, H.; Torr, P. H.; and Zhao, G. 2022. Physformer: Facial video-based physiological measurement with temporal difference transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4186–4196.

Yue, Z.; Shi, M.; and Ding, S. 2023. Facial video-based remote physiological measurement via self-supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Zhao, Y.; Zou, B.; Yang, F.; Lu, L.; Belkacem, A. N.; and Chen, C. 2021. Video-based physiological measurement using 3d central difference convolution attention network. In 2021 IEEE International Joint Conference on Biometrics (IJCB), 1–6. IEEE.

Zhu, L.; Liao, B.; Zhang, Q.; Wang, X.; Liu, W.; and Wang, X. 2024. Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model. In Forty-first International Conference on Machine Learning.

Zou, B.; Guo, Z.; Chen, J.; and Ma, H. 2024. RhythmFormer: Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer. arXiv preprint arXiv:2402.12788.

# A State Space Model

The state space model (SSM) is a type of linear time-invariant system that maps inputs to outputs through hidden layers. It is modeled by the following ordinary differential equations:

$$
\begin{array}{l} h ^ {\prime} (t) = A h (t) + B x (t), \\ \text {(8)} \end{array}
$$

$$
y (t) = C h (t).
$$

Where $A \in \mathbb{R}^{N \times N}$ is the evolution matrix, $B \in \mathbb{R}^{N \times 1}$ and $C \in \mathbb{R}^{1 \times N}$ are the projection matrices. $N$ represents the sequence length. This continuous system is challenging to apply, while the SSM in Mamba serves as its discrete counterpart. It discretizes continuous parameters A and B into their discrete counterparts $\overline{A}$ and $\overline{B}$ using a time-scale parameter $\Delta$ . This transformation typically employs the zero-order hold method:

$$
\overline {{A}} = e x p (\Delta A),
$$

$$
\overline {{{B}}} = (\Delta A) ^ {- 1} (e x p (\Delta A) - I) \Delta B, \tag {9}
$$

$$
h _ {t} = \overline {{A}} h _ {t - 1} + \overline {{B}} x _ {t},
$$

$$
y _ {t} = C h _ {t}.
$$

The practical application of this discretization form is hindered by its inherent sequential nature. Nevertheless, it can be effectively represented through a convolution operation:

$$
\overline {{{K}}} = (C \overline {{{B}}}, C \overline {{{A B}}},..., C \overline {{{A}}} ^ {N - 1} \overline {{{B}}}), \tag {10}
$$

$$
y = x * \overline {{K}}.
$$

Where $\overline{K} \in R^{N}$ denotes a structured convolution kernel, and \* indicates a convolution operation.

# B Visualization

As shown in the left half of Figure a, we visualize the spectrum of an example from MMPD. From top to bottom, these represent the averaging of spectra across all channels in the frequency domain feed-forward of the last block, the power spectral density of the PPG signal before bandpass filtering, and the power spectral density of the PPG after bandpass filtering. The range from 0.75 Hz to 2.5 Hz corresponds to the heart rate frequency band. Since the spectrum of the Fre FFN is obtained through FFT, the number of frequency points in the spectrum depends on the input length, resulting in a less smooth output. Nevertheless, it still accurately captures the frequency domain characteristics of the BVP ground truth.

As shown in the right half of Figure a, an example of the PPG waveform is provided to demonstrate the efficacy and precision of our approach. Notably, our model robustly captures the peaks and variations of PPG signals, serving as the key basis for predicting heart rate from video data. Additionally, the scatter plots and Bland-Altman plots in Figure b and Figure c further demonstrate the strong correlation between the predictions and the ground truth.

# C Arbitrary Length Videos Input

We attempt to test the trained RhythmMamba model on videos of arbitrary lengths, with the longest test video being limited to 60 seconds due to dataset constraints. As observed in Table a, the trained model enables seamless adaptation to video segments of any length without performance degradation. Only when the test length is reduced to 1 second (second row), which is close to or shorter than a single heartbeat, does some performance degradation occur. The results at various test lengths clearly demonstrate that RhythmMamba has effectively learned the periodic pattern of rPPG.

Table a: Arbitrary length videos input. 

<table><tr><td>Test Length</td><td>MAE↓</td><td>RMSE↓</td><td>MAPE↓</td><td> $\rho \uparrow$ </td><td>SNR↑</td></tr><tr><td>160</td><td>3.16</td><td>7.27</td><td>3.37</td><td>0.84</td><td>4.74</td></tr><tr><td>30 (1s)</td><td>5.50</td><td>11.65</td><td>6.02</td><td>0.66</td><td>-1.70</td></tr><tr><td>80 (2.6s)</td><td>3.11</td><td>6.61</td><td>3.30</td><td>0.87</td><td>3.09</td></tr><tr><td>300 (10s)</td><td>3.21</td><td>7.43</td><td>3.42</td><td>0.83</td><td>6.11</td></tr><tr><td>600 (20s)</td><td>3.10</td><td>7.61</td><td>3.32</td><td>0.82</td><td>8.15</td></tr><tr><td>900 (30s)</td><td>3.51</td><td>7.30</td><td>3.70</td><td>0.84</td><td>8.56</td></tr><tr><td>1800 (60s)</td><td>3.14</td><td>7.04</td><td>3.30</td><td>0.85</td><td>10.59</td></tr></table>

![](images/c8a21795226a89334ad4f64987f8abe735ad9d904306d92108f876f62991c7a6.jpg)

<details>
<summary>line</summary>

| Frequency (Hz) | Pre FFN |
| -------------- | ------- |
| 0.75           | 0.75    |
| 2.5            | 2.5     |
</details>

![](images/9fa1343c43bd166ab775f8c297edad9f3a7d96d84b9f6b8faa00d891772043be.jpg)

<details>
<summary>line</summary>

| Time (s) | Ground Truth | Prediction |
| -------- | ------------ | ---------- |
| 0.0      | -1.5         | -0.5       |
| 0.5      | 1.8          | 2.0        |
| 1.0      | -0.5         | 2.0        |
| 1.5      | 2.5          | 1.5        |
| 2.0      | -1.5         | -0.5       |
| 2.5      | 1.8          | 1.0        |
| 3.0      | -0.5         | -0.5       |
| 3.5      | 1.8          | 1.5        |
| 4.0      | -1.5         | -1.0       |
| 4.5      | 2.0          | 2.0        |
| 5.0      | -0.5         | -1.0       |
| 5.5      | -1.5         | -1.5       |
</details>

![](images/a207114cbb8629889b509b32162e77c6461e46f74a8f792029cb20527b40ab13.jpg)

<details>
<summary>line</summary>

| Frequency (Hz) | Before Filter |
| -------------- | ------------- |
| 0.15           | 0.0           |
| 1.5            | 2.2           |
| 2.25           | 0.6           |
| 3.0            | 0.3           |
| 3.75           | 0.4           |
| 4.5            | 0.0           |
| 5.25           | 0.0           |
| 6.0            | 0.0           |
| 6.75           | 0.0           |
| 7.5            | 0.0           |
</details>

![](images/3936481171ab23be3b07d7c43570786b35d7d52d00f50f953a599a54cd3eded6.jpg)

<details>
<summary>line</summary>

| Time (s) | After Filter |
| -------- | ------------ |
| 0.0      | 0.0          |
| 0.5      | 1.3          |
| 1.0      | -1.2         |
| 1.5      | 1.5          |
| 2.0      | -1.1         |
| 2.5      | 1.4          |
| 3.0      | -0.9         |
| 3.5      | 1.2          |
| 4.0      | -1.3         |
| 4.5      | 1.6          |
| 5.0      | -1.4         |
</details>

![](images/fb329af827ffb83ddda88e06bd4722538812ba0e82949871bcfc66e47eb77495.jpg)

<details>
<summary>line</summary>

| Frequency (Hz) | After Filter |
| -------------- | ------------ |
| 0.75           | 0.0          |
| 1.5            | 2.0          |
| 2.25           | 0.0          |
| 3.0            | 0.0          |
| 4.5            | 0.0          |
| 6.0            | 0.0          |
| 7.5            | 0.0          |
</details>

Figure a: An example of results on MMPD.

![](images/5a16cd2667613678a85c5919b16560459db32fd5d5e9c2203e1008ec5bd15c16.jpg)

<details>
<summary>scatter</summary>

| Average of rPPG HR and GT PPG HR [bpm] | Difference between rPPG HR and GT PPG HR [bpm] |
| -------------------------------------- | ----------------------------------------------- |
| 60                                     | 0                                               |
| 70                                     | 0                                               |
| 80                                     | -30                                             |
| 90                                     | 0                                               |
| 100                                    | 20                                              |
| 110                                    | 0                                               |
| 120                                    | 15                                              |
</details>

(a) Intra-dataset on MMPD.

![](images/ab7419a965a467c32b6bcdedf48ae378379a4e9fcf1d1b6f80b3af29ba81f929.jpg)

<details>
<summary>scatter</summary>

| Average of rPPG HR and GT PPG HR [bpm] | Difference between rPPG HR and GT PPG HR [bpm] |
| -------------------------------------- | ----------------------------------------------- |
| 60                                     | 5.5                                             |
| 70                                     | 2.2                                             |
| 80                                     | -1.5                                            |
| 90                                     | 1.8                                             |
| 100                                    | -2.0                                            |
| 110                                    | 1.5                                             |
| 120                                    | -7.5                                            |
| 130                                    | 1.0                                             |
</details>

(b) Cross-dataset on PURE-UBFC.

![](images/1c395754972f7f105bf79bcda61f19405b59d59010c2ca613e9dd4716c07ef91.jpg)

<details>
<summary>scatter</summary>

| Average of rPPG HR and GT PPG HR [bpm] | Difference between rPPG HR and GT PPG HR [bpm] |
| -------------------------------------- | ----------------------------------------------- |
| 60                                     | -10                                             |
| 70                                     | -30                                             |
| 80                                     | -40                                             |
| 90                                     | -50                                             |
| 100                                    | 0                                               |
| 110                                    | 20                                              |
| 120                                    | 0                                               |
| 130                                    | -20                                             |
| 140                                    | -40                                             |
| 150                                    | -60                                             |
| 160                                    | -80                                             |
| 170                                    | -100                                            |
| 180                                    | -120                                            |
| 190                                    | -140                                            |
| 200                                    | -160                                            |
| 210                                    | -180                                            |
| 220                                    | -200                                            |
| 230                                    | -220                                            |
| 240                                    | -240                                            |
| 250                                    | -260                                            |
| 260                                    | -280                                            |
| 270                                    | -300                                            |
| 280                                    | -320                                            |
| 290                                    | -340                                            |
| 300                                    | -360                                            |
| 310                                    | -380                                            |
| 320                                    | -400                                            |
| 330                                    | -420                                            |
| 340                                    | -440                                            |
| 350                                    | -460                                            |
| 360                                    | -480                                            |
| 370                                    | -500                                            |
| 380                                    | -520                                            |
| 390                                    | -540                                            |
| 400                                    | -560                                            |
| 410                                    | -580                                            |
| 420                                    | -600                                            |
| 430                                    | -620                                            |
| 440                                    | -640                                            |
| 450                                    | -660                                            |
| 460                                    | -680                                            |
| 470                                    | -700                                            |
| 480                                    | -720                                            |
| 490                                    | -740                                            |
| 500                                    | -760                                            |
| 510                                    | -780                                            |
| 520                                    | -800                                            |
| 530                                    | -820                                            |
| 540                                    | -840                                            |
| 550                                    | -860                                            |
| 560                                    | -880                                            |
| 570                                    | -900                                            |
| 580                                    | -920                                            |
| 590                                    | -940                                            |
| 600                                    | -960                                            |
| 610                                    | -980                                            |
| 620                                    | -1000                                           |
| 630                                    | -1020                                           |
| 640                                    | -1040                                           |
| 650                                    | -1060                                           |
| 660                                    | -1080                                           |
| 670                                    | -1100                                           |
| 680                                    | -1120                                           |
| 690                                    | -1140                                           |
| 700                                    | -1160                                           |
| 710                                    | -1180                                           |
| 720                                    | -1200                                           |
| 730                                    | -1220                                           |
| 740                                    | -1240                                           |
| 750                                    | -1260                                           |
| 760                                    | -1280                                           |
| 770                                    | -1300                                           |
| 780                                    | -1320                                           |
| 790                                    | -1340                                           |
| 800                                    | -1360                                           |
| 810                                    | -1380                                           |
| 820                                    | -1400                                           |
| 830                                    | -1420                                           |
| 840                                    | -1440                                           |
| 850                                    | -1460                                           |
| 860                                    | -1480                                           |
| 870                                    | -1500                                           |
| 880                                    | -1520                                           |
| 890                                    | -1540                                           |
| 900                                    | -1560                                           |
| 910                                    | -1580                                           |
| 920                                    | -1600                                           |
| 930                                    | -1620                                           |
| 940                                    | -1640                                           |
| 950                                    | -1660                                           |
| 960                                    | -1680                                           |
| 970                                    | -1700                                           |
| 980                                    | -1720                                           |
| 990                                    | -1740                                           |
| 1000                                   | -1760                                           |
| 115                                    | -35                                             |
| 125                                    | -37                                             |
| 135                                    | -39                                             |
| 145                                    | -41                                             |
| 155                                    | -43                                             |
| 165                                    | -45                                             |
| 175                                    | -47                                             |
| 185                                    | -49                                             |
| 195                                    | -51                                             |
| 205                                    | -53                                             |
| 215                                    | -55                                             |
| 225                                    | -57                                             |
| 235                                    | -59                                             |
| 245                                    | -61                                             |
| 255                                    | -63                                             |
| 265                                    | -65                                             |
| 275                                    | -67                                             |
| 285                                    | -69                                             |
| 295                                    | -71                                             |
| 305                                    | -73                                             |
| 315                                    | -75                                             |
| 325                                    | -77                                             |
| 335                                    | -79                                             |
| 345                                    | -81                                             |
| 355                                    | -83                                             |
| 365                                    | -85                                             |
| 375                                    | -87                                             |
| 385                                    | -89                                             |
| 395                                    | -91                                             |
| 405                                    | -93                                             |
| 415                                    | -95                                             |
| 425                                    | -97                                             |
| 435                                    | -99                                             |
| 445                                    | -101                                            |
| 455                                    | -103                                            |
| 465                                    | -105                                            |
| 475                                    | -107                                            |
| 485                                    | -109                                            |
| 495                                    | -111                                            |
| 505                                    | -113                                            |
| Note: The y-axis label is 'Difference between rPPG HR and GT PPG HR' and the x-axis label is 'Average of rPPG HR and GT PPG HR'. There are no labels for the data series. The y-axis label is 'Difference between rPPG HR and GT PPG HR'. The legend indicates 'Observations' as the data series. The y-axis label is 'Mean Error' as a solid line. The confidence intervals are shown as dashed lines above and below the x-axis. The y-axis label is 'Observed' as a dot plot above the x-axis. The y-axis label is 'Observations' as a circle plot above it.
</details>

(c) Cross-dataset on UBFC-PURE.   
Figure b: Bland-Altman plots of results.

![](images/4e6b39f37a9a50c1f950e05984b0c3e7a6f62b00d89fe6045d29b273b1cc709d.jpg)

<details>
<summary>scatter</summary>

| GT PPG HR [bpm] | rPPG HR [bpm] |
| --------------- | ------------- |
| 60              | 60            |
| 70              | 70            |
| 80              | 80            |
| 90              | 90            |
| 100             | 100           |
| 110             | 110           |
| 120             | 120           |
| 130             | 130           |
</details>

(a) Intra-dataset on MMPD.

![](images/0b93d80385f4c86bcad0b4e543fbb08bde52f0f4f55a57be1429d21813515fe9.jpg)

<details>
<summary>scatter</summary>

| GT PPG HR [bpm] | rPPG HR [bpm] |
| --------------- | ------------- |
| 60              | 58            |
| 70              | 68            |
| 80              | 80            |
| 90              | 90            |
| 100             | 100           |
| 110             | 110           |
| 120             | 120           |
| 130             | 130           |
</details>

(b) Cross-dataset on PURE-UBFC.

![](images/4dba21dff9346bab4de2b1346c93fc8b33c30ec73c61a08458a68bc0f8215c70.jpg)

<details>
<summary>scatter</summary>

| GT PPG HR [bpm] | rPPG HR [bpm] |
| --------------- | ------------- |
| 40              | 40            |
| 50              | 60            |
| 60              | 70            |
| 70              | 80            |
| 80              | 90            |
| 90              | 100           |
| 100             | 100           |
| 110             | 120           |
| 120             | 130           |
| 130             | 135           |
| 140             | 140           |
</details>

(c) Cross-dataset on UBFC-PURE.   
Figure c: Scatter plots of results.