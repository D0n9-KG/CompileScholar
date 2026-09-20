# TimeStacker: A Novel Framework with Multilevel Observation for Capturing Nonstationary Patterns in Time Series Forecasting

Qinglong Liu $^{1}$ Cong Xu $^{1}$ Wenhao Jiang $^{1}$ Kaixuan Wang $^{1}$ Lin Ma $^{1}$ Haifeng Li $^{1}$

# Abstract

Real-world time series inherently exhibit significant non-stationarity, posing substantial challenges for forecasting. To address this issue, this paper proposes a novel prediction framework, TimeStacker, designed to overcome the limitations of existing models in capturing the characteristics of non-stationary signals. By employing a unique stacking mechanism, TimeStacker effectively captures global signal features while thoroughly exploring local details. Furthermore, the framework integrates a frequency-based self-attention module, significantly enhancing its feature modeling capabilities. Experimental results demonstrate that TimeStacker achieves outstanding performance across multiple real-world datasets, including those from the energy, finance, and weather domains. It not only delivers superior predictive accuracy but also exhibits remarkable advantages with fewer parameters and higher computational efficiency.

# 1. Introduction

Time series forecasting, which involves inferring future trends and patterns from historical observations, is widely applied in diverse domains, including weather forecasting(Wu et al., 2023), energy scheduling(Chou & Tran, 2018), traffic management(Zhou et al., 2021), medical analysis(Čepulionis & Lukoševičiūtė, 2016), and financial economics(Cheng et al., 2022). However, the complexity of real-world systems often results in non-stationary time series(Wang et al., 2024b), which complicates accurate prediction using traditional methods and presents substantial challenges for time series forecasting.

With the rapid advancement of deep learning, numerous neu-

$^{1}$ Faculty of Computing, Harbin Institute of Technology, Harbin, China, Harbin, China. Correspondence to: Haifeng Li <li-haifeng@hit.edu.cn>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

ral network models have been developed, exhibiting remarkable performance in time series forecasting. For example, MLP-based approaches such as DLinear(Zeng et al., 2023), SOFTS(Han et al., 2024a), SparseTSF(Lin et al., 2024) and TimeMixer(Wang et al., 2024a), and Transformer-based architectures include Crossformer(Zhang & Yan, 2023), PatchTST(Nie et al., 2022), SAMformer(Ilbert et al.), and iTransformer(Liu et al., 2023). These models have achieved state-of-the-art performance in time series forecasting due to their advanced architectural designs and innovative methodologies.

![](images/5cad08ec1eb48e4bc22950c67955cf35a3dd71775903e9e232789ba9561cf0ed.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Signal"] --> B["Time Domain"]
    B --> C1["Window size=L"]
    B --> C2["Window size=L/2"]
    B --> C3["Window size=L/4"]
    C1 --> D1["Amplitude vs Frequency"]
    C2 --> D2["Amplitude vs Frequency"]
    C3 --> D3["Amplitude vs Frequency"]
    D1 --> E1["If we stack all the results, we can fully represent the sequence."]
    D2 --> E2["If we stack all the results, we can fully represent the sequence."]
    D3 --> E3["If we stack all the results, we can fully represent the sequence."]
```
</details>

Figure 1. Under observations with different window sizes, the frequency of the same non-stationary signal exhibits varying patterns. The blue signal represents the original non-stationary signal, while the red components illustrate the frequency patterns obtained through Short-Time Fourier Transform (STFT) with window sizes of L, 2/L, and 4/L, respectively.

Despite the substantial advancements made by these methods in time series forecasting, the majority of studies primarily explore temporal correlations, often overlooking the frequency-domain characteristics of non-stationary signals. Based on stochastic process theory(Cox, 2017), the frequency of non-stationary signals fluctuates over time, and frequency-domain representations are more effective than time-domain signals in capturing signal characteristics within specific time intervals. Thus, analyzing the frequency variation patterns of non-stationary signals is essential for time series forecasting.

However, the uncertainty principle of time-frequency analysis(Cohen, 1995) precludes the precise observation of a signal's frequency at a specific moment. To overcome this limitation, the short-time Fourier transform (STFT) is commonly employed to segment the original signal (i.e., divide it into patches) for frequency analysis within specific time intervals. The selection of patch size significantly influences the ability to capture frequency variation patterns: larger patches are more effective in capturing global signal features, whereas smaller patches better reveal local details. As depicted in Figure 1, signal frequencies corresponding to different patch sizes are visualized. Hence, identifying an optimal patch size for effectively extracting and representing signal patterns remains a key challenge in revealing the intrinsic regularities of sequences.

To overcome these challenges, this paper introduces a novel framework, TimeStacker. Rather than selecting a single optimal patch size, patches of varying sizes are sequentially stacked and aggregated based on frequency. Through iterative stacking, the most expressive patterns within the signal are progressively emphasized, enhancing the representation of the overall time series. Specifically, patterns within the signal are aggregated layer by layer in descending order of patch size, allowing the model to capture global features while retaining local details.

To further optimize the stacking process, a frequency-based enhanced self-attention module is designed to aggregate patches of the same size. Within this module, signal similarity is computed in the frequency domain, whereas aggregation operations are conducted in the time domain. This approach effectively mitigates the detrimental effects of inherent Fourier transform errors and spectral leakage in signal modeling. Experimental results indicate that TimeStacker attains state-of-the-art performance across most forecasting tasks while maintaining fewer parameters and significantly greater efficiency than other models.

The main contributions of this paper are summarized as follows:

i) A novel framework, TimeStacker, is proposed to comprehensively capture the variation patterns of frequency scales. By stacking patches sequentially from large to small, it facilitates a simple yet effective approach to time series forecasting.

ii) A novel frequency-based self-attention module is designed to more effectively compute the similarity between patches based on their frequencies. Additionally, this module mitigates the detrimental effects of inherent Fourier transform errors and spectral leakage in signal modeling.

iii) Experimentally, TimeStacker demonstrates state-of-the-art predictive accuracy across most real-world datasets while utilizing fewer parameters and exhibiting higher computa-

tional efficiency than other benchmark models. This framework offers a viable solution for time series forecasting of non-stationary signals.

# 2. Related Work

Time series forecasting models have undergone substantial evolution over time. Early approaches to time series forecasting were primarily based on statistical theories, which typically assumed that time series exhibit stationarity or linear relationships(Box et al., 2015). These methods predicted trends, seasonality, and stochastic variations by modeling these components. Representative models include AutoRegressive Integrated Moving Average (ARIMA)(Lee & Tong, 2011), Exponential Smoothing (ETS)(De Livera et al., 2011), and Seasonal AutoRegressive Integrated Moving Average (SARIMA)(Dubey et al., 2021).

The emergence of deep learning facilitated significant advancements in time series forecasting, particularly with the introduction of Transformer models(Li et al., 2019). In contrast to traditional approaches, deep learning models are capable of automatically extracting features and effectively capturing complex nonlinear relationships. For example, Recurrent Neural Networks (RNNs)(Sagheer & Kotb, 2019) capture temporal dependencies in time series through their recursive structure, making them particularly effective for modeling short-term dependencies. Convolutional Neural Networks (CNNs)(Sezer et al., 2020) utilize one-dimensional convolutional operations to extract local features, effectively capturing short-term patterns and regularities in sequences.

The introduction of the Transformer architecture represented a fundamental paradigm shift in time series forecasting. For instance, PatchTST partitioned time series into independent patches embedded in high-dimensional spaces while preserving channel independence, enabling all series to share weights and establishing a foundation for subsequent research. Crossformer improved multivariate forecasting capabilities by capturing cross-dimensional patch dependencies in multivariate time series. iTransformer revisited the hierarchical design of traditional Transformer architectures, utilizing self-attention mechanisms to model inter-variable relationships and employing feedforward networks to capture nonlinear variable transformations.

Beyond Transformer-based architectures, recent years have seen substantial advancements in MLP-based models. DLinear highlighted the effectiveness of linear layers in time series forecasting, particularly excelling in long-sequence modeling. SparseTSF simplifies the forecasting task by disentangling the periodic and trend components of time series data through cross-period sparse prediction techniques. TimeMixer introduced a fully MLP-based model designed

![](images/80ea9f6b09824f6e805903a33a5a608a803cb4a2f37f5bbcdacb58ad5bbe11f7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Data"] --> B["Normalization"]
    B --> C["Stacker Block"]
    C --> D["Patch size₁"]
    C --> E["Inter-Patch Frequency based Attention Module"]
    C --> F["Channel Independent"]
    D --> G["+"]
    E --> H["+"]
    F --> I["+"]
    G --> J["Predictor"]
    H --> J
    I --> J
    J --> K["Denormalization"]
    
    style A fill:#f9f,stroke:#333
    style K fill:#bbf,stroke:#333
    
    subgraph Input Data
        L["Time Points"] --> M["Line Chart"]
        M --> N["Waveform Graph"]
        N --> O["Waveform Graph"]
        O --> P["Waveform Graph"]
        P --> Q["Waveform Graph"]
        Q --> R["Waveform Graph"]
        R --> S["Waveform Graph"]
        S --> T["Waveform Graph"]
        T --> U["Waveform Graph"]
        U --> V["Waveform Graph"]
        V --> W["Waveform Graph"]
        W --> X["Waveform Graph"]
        X --> Y["Waveform Graph"]
        Y --> Z["Waveform Graph"]
        Z --> AA["Waveform Graph"]
        AA --> AB["Waveform Graph"]
        AB --> AC["Waveform Graph"]
        AC --> AD["Waveform Graph"]
        AD --> AE["Waveform Graph"]
        AE --> AF["Waveform Graph"]
        AF --> AG["Waveform Graph"]
        AG --> AH["Waveform Graph"]
        AH --> AI["Waveform Graph"]
        AI --> AJ["Waveform Graph"]
        AJ --> AK["Waveform Graph"]
        AK --> AL["Waveform Graph"]
        AL --> AM["Waveform Graph"]
        AM --> AN["Waveform Graph"]
        AN --> AO["Waveform Graph"]
        AO --> AP["Waveform Graph"]
        AP --> AQ["Waveform Graph"]
        AQ --> AR["Waveform Graph"]
        AR --> AS["Waveform Graph"]
        AS --> AT["Waveform Graph"]
        AT --> AU["Waveform Graph"]
        AU --> AV["Waveform Graph"]
        AV --> AW["Waveform Graph"]
        AW --> AX["Waveform Graph"]
        AX --> AY["Waveform Graph"]
        AY --> AZ["Waveform Graph"]
        AZ --> BA["Waveform Graph"]
        BA --> BB["Waveform Graph"]
        BB --> BC["Waveform Graph"]
        BC --> BD["Waveform Graph"]
        BD --> BE["Waveform Graph"]
        BE --> BF["Waveform Graph"]
        BF --> BG["Waveform Graph"]
        BG --> BH["Waveform Graph"]
        BH --> BI["Waveform Graph"]
        BI --> BJ["Waveform Graph"]
        BJ --> BK["Waveform Graph"]
        BK --> BL["Waveform Graph"]
        BL --> BM["Waveform Graph"]
        BM --> BN["Waveform Graph"]
        BN --> BO["Waveform Graph"]
        BO --> BP["Waveform Graph"]
        BP --> BQ["Waveform Graph"]
        BQ --> BR["Waveform Graph"]
        BR --> BS["Waveform Graph"]
        BS --> BT["Waveform Graph"]
        BT --> BU["Waveform Graph"]
        BU --> BV["Waveform Graph"]
        BV --> BW["Waveform Graph"]
        BW --> BX["Waveform Graph"]
        BX --> BY["Waveform Graph"]
        BY --> BZ["Waveform Graph"]
        BZ --> CA["Waveform Graph"]
        CA --> CB["Waveform Graph"]
        CB --> CC["Waveform Graph"]
        CC --> CD["Waveform Graph"]
        CD --> CE["Waveform Graph"]
        CE --> CF["Waveform Graph"]
        CF --> CG["Waveform Graph"]
        CG --> CH["Waveform Graph"]
        CH --> CI["Waveform Graph"]
        CI --> CJ["Waveform Graph"]
        CJ --> CK["Waveform Graph"]
        CK --> CL["Waveform Graph"]
        CL --> CM["Waveform Graph"]
        CM --> CN["Waveform Graph"]
        CN --> CO["Waveform Graph"]
        CO --> CP["Waveform Graph"]
        CP --> CQ["Waveform Graph"]
        CQ --> CR["Waveform Graph"]
        CR --> CS["Waveform Graph"]
        CS --> CT["Waveform Graph"]
        CT --> CU["Waveform Graph"]
        CU --> CV["Waveform Graph"]
        CV --> CW["Waveform Graph"]
        CW --> CX["Waveform Graph"]
        CX --> CY["Waveform Graph"]
        CY --> CZ["Waveform Graph"]
        CZ --> DA["Waveform Graph"]
        DA --> DB["Waveform Graph"]
        DB --> DC["Waveform Graph"]
        DC --> DD["Waveform Graph"]
        DD --> DE["Forecast: Patch size gradually decreases ×L"]
    end
```
</details>

Figure 2. Overall Architecture of TimeStacker. The overall architecture comprises multiple Stacker Blocks. Each Stacker Block consists of a Smooth Layer and an Inter-Patch Frequency-based Attention Module, responsible for smoothing the time series and aggregating patches, respectively. Within each block, patches of the same size are aggregated based on their frequency characteristics. Subsequent blocks sequentially process patches of decreasing sizes.

to explore multiscale temporal information in time series across various temporal domains.

# 3. Method

This section begins with the task definition, followed by an overview of the preliminaries, the overall structure of TimeStacker, implementation details, theoretical analysis, and complexity analysis. The overall architecture of the proposed model is depicted in Figure 2.

# 3.1. Problem Definition

The time series forecasting problem is formulated as follows: Given a time series $X_{t-T+1:t} = \{x_{t-T+1}, \ldots, x_t\} \in R^{D \times T}$ , where t represents a specific timestamp, Dis the number of variables, and $x_t \in R^D$ denotes the observed value at time t, the objective is to predict the future values $\hat{X}_{t+1:t+\tau} = \{\hat{x}_{t+1}, \ldots, \hat{x}_{t+\tau}\} \in R^{D \times \tau}$ , where $\tau$ represents the prediction horizon.

# 3.2. Preliminaries

Normalization. Time series datasets often exhibit varying numerical ranges, which can result in unequal model attention to different data and lead to biased parameter updates. Levin(Kim et al., 2021) demonstrated that appropriate normalization significantly enhances time series forecasting performance and plays a crucial role in model training. Therefore, this model employs the same normalization and denormalization approach as Levin, utilizing mean and variance for standardization. The formulas are as follows:

$$
X ^ {\prime} = \text { normal } (X) = \frac {X - \mu}{\sigma} \tag {1}
$$

$$
\hat {X} = \text { denormal } \left(\hat {X} ^ {\prime}\right) = \hat {X} ^ {\prime} \sigma + \mu \tag {2}
$$

Here, $X'$ denotes the normalized time series, $\mu$ represents the mean of the input time series, and $\sigma$ represents the standard deviation. This normalization process ensures that the data have a mean of 0 and a variance of 1, thereby mitigating the impact of scale differences on model training.

Channel Independence. Channel independence is a fundamental strategy in time series forecasting. The core principle involves treating each variable in a multivariate time series as an independent channel and modeling each channel separately rather than as a whole. This approach was first introduced by PatchTST(Han et al., 2024b), which demonstrated its effectiveness in time series forecasting and has since been widely adopted in subsequent neural network models for time series prediction. This method mitigates noise interference between channels and reduces modeling complexity, thereby facilitating a more effective representation of individual variable characteristics.

# 3.3. Overall Architecture

Patches of different sizes capture the frequency variation patterns of time series to different extents. A decrease in patch size enhances the temporal resolution of the sequence while reducing its frequency resolution. Stacking patches enables a more comprehensive analysis of complex variation patterns in time series, thereby enhancing forecasting accuracy.

As illustrated in Figure 2, the overall framework comprises L StackerBlocks, a normalization-denormalization module, and a predictor module. The StackerBlock is designed to capture variation patterns within patches of the same size and is elaborated in Section 3.4. The normalization-denormalization module, as discussed in Section 3.1, performs data preprocessing, while the predictor module is implemented as a linear layer.

Consider a univariate time series of length $T$ , represented as $X = \{x_{1}, x_{2}, \ldots, x_{T}\} \in \mathbb{R}^{1 \times T}$ . Given $L$ non-overlapping patch sizes defined by the vector $P = \{p_{1}, p_{2}, \ldots, p_{L}\}$ , where $p_{1} > p_{2} > \ldots > p_{L}$ and each element in $P$ is required to be a divisor of $T$ , the goal is to extract features and progressively stack patches to capture variation pat-

![](images/10ca75b95b125b5df49a7ae2f59d7aafd7e8c97ee4a89b881980690dc9cea3c6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Patch Size"] --> B["DFT"]
    B --> C["Hadamard Product"]
    C --> D["Query"]
    D --> E["MatMul"]
    E --> F["Inter-Patch Correlation Matrix"]
    F --> G["MatMul"]
    G --> H["Linear Projection"]
    H --> B
    C --> I["Wq"]
    C --> JWk
    D --> K["Key"]
    E --> L["Softmax"]
    F --> M["Grid"]
```
</details>

Figure 3. Internal Structure of FreqAttention. This module initially computes the similarity between patches in the frequency domain. It then aggregates these patches in the time domain based on the computed similarity.

terns effectively. The time series $X$ is first standardized as follows:

$$
\overline {{{X}}} ^ {(1)} = \text { normal } (X) \tag {3}
$$

Let $p_1$ be the first element of $P$ , representing the initial patch size. The standardized time series $\overline{X}^{(1)}$ is partitioned into subsequences of size $p_1$ , yielding:

$$
\mathcal {X} ^ {(1)} = \left\{\S_ {1} ^ {(1)}, \S_ {2} ^ {(1)}, \dots , \S_ {k _ {1}} ^ {(1)} \right\} \in \mathbb {R} ^ {k _ {1} \times p _ {1}} \tag {4}
$$

Here $k_{1} = \frac{T}{p_{1}}$ . Next, §1 $^{(1)}$ is fed into the StackerBlock associated with $p_{1}$ , which extracts variation patterns within patches of the same size, yielding $\overline{\mathcal{X}}^{(1)}$ . Similarly, let $p_{2}$ be the next patch size, and re-segment $\overline{\mathcal{X}}^{(1)}$ into patches, yielding:

$$
\mathcal {X} ^ {(2)} \in \mathbb {R} ^ {k _ {2} \times p _ {2}}, k _ {2} = \frac {T}{p _ {2}} \tag {5}
$$

After applying the same operations, $\overline{\mathcal{X}}^{(2)}$ is obtained. This process is iterated L times, with each step further partitioning patches based on the preceding iteration. Through iterative patch stacking, the variation patterns of the time series are progressively captured by integrating patches of different sizes. The formal representation is as follows:

$$
\begin{array}{c} \mathcal {X} ^ {(l)} = \text { Concat } \left[ \begin{array}{c} \text { StackerBlock } _ {l - 1} \left(\mathcal {X} ^ {(l - 1)}\right) _ {1} \\ \text { StackerBlock } _ {l - 1} \left(\mathcal {X} ^ {(l - 1)}\right) _ {2} \\ \text { StackerBlock } _ {l - 1} \left(\mathcal {X} ^ {(l - 1)}\right) _ {3} \\ \dots \\ \text { StackerBlock } _ {l - 1} \left(\mathcal {X} ^ {(l - 1)}\right) _ {p _ {l}} \end{array} \right], \\ l = 1, 2, \dots , L \end{array} \tag {6}
$$

# 3.4. Stacker Block

To effectively capture variation patterns across patches (inter-patch), the StackerBlock is introduced. This module comprises two core components: the Smooth Layer and the Inter-Patch Frequency-Based Attention Module.

In real-world time series, outliers can significantly impact the model's ability to capture sequence variation patterns. To mitigate this issue, the Smooth Layer is introduced, utilizing time points within a patch to reduce the influence of outliers. Specifically, the Smooth Layer is implemented using a convolution operation with a kernel size of $p_l$ . The formal representation is given by:

$$
\text { SmoothLayer } _ {l} \left(\mathcal {X} ^ {(l)}\right) = W * \mathcal {X} ^ {(l)} + b \tag {7}
$$

To exploit frequency variation information within the sequence, the Inter-Patch Frequency-Based Attention Module (FreqAttention) is introduced. The structure of FreqAttention is depicted in Figure 3. Unlike traditional self-attention mechanisms, similarity in this approach is computed in the frequency domain. The formal representation is given by:

$$
\widetilde {\mathcal {X}} ^ {(l)} = \mathcal {F} \left(\mathcal {X} ^ {(l)}\right) \tag {8}
$$

$$
Q = W _ {q} \odot \widetilde {\mathcal {X}} ^ {(l)}, K = W _ {k} \odot \widetilde {\mathcal {X}} ^ {(l)} \tag {9}
$$

$$
\operatorname{CorMat} = \text { Softmax } \left(\frac {W _ {q} \odot \widetilde {\mathcal {X}} ^ {(l)} \cdot \left(W _ {k} \odot \widetilde {\mathcal {X}} ^ {(l)}\right) ^ {T}}{\sqrt {d _ {k}}}\right) \tag {10}
$$

Here, $\mathcal{F}(\cdot)$ denotes the Fourier transform, $\odot$ represents the Hadamard product, and $W_{q}, W_{k} \in R^{\lfloor \frac{pl}{2} \rfloor + 1}$ are learnable parameters used to compute the query and key vectors, and $d_{k}$ is a scalar matching the dimension of $widetilde\mathcal{X}^{(l)}$ . From a signal processing perspective, the Hadamard product enables $W_{q}$ and $W_{k}$ to function as learnable filters, extracting frequency components relevant to subsequent sequences. Consequently, $CorMat \in R^{kl \times kl}$ represents the frequency-based similarity between patches and reflects the sequence's variation patterns in the frequency domain. Aggregating the time series in the time domain using CorMat yields the following formal expression:

$$
V = \mathcal {X} ^ {(l)} W _ {v} \tag {11}
$$

$$
\begin{array}{c} \text { FreqAttn } (\mathcal {X} ^ {(l)}) = \\ \text { Softmax } \left(\frac {W _ {q} \odot \tilde {\mathcal {X}} ^ {(l)} \cdot (W _ {k} \odot \tilde {\mathcal {X}} ^ {(l)}) ^ {T}}{\sqrt {d _ {k}}}\right) \mathcal {X} ^ {(l)} W _ {v} \end{array} \tag {12}
$$

Here, $W_{v} \in R^{pl}$ is a learnable parameter for computing the value vector. By combining Equations (7) to (12), the formal expression of the StackerBlock is given by:

$$
\begin{array}{c} \text {StackerBlock} _ {l} (\mathcal {X} ^ {(l)}) = \\ \text {FreqAttn(SmoothLayer} _ {l} (\mathcal {X} ^ {(l)}) + \mathcal {X} ^ {(l)}) + \mathcal {X} ^ {(l)} \end{array} \tag {13}
$$

Table 1. Main Results. All results are based on input sequences of length 96 and are calculated as the average across four different prediction lengths {96, 192, 336, 720}. The prediction performance is evaluated using MSE or MAE as metrics, where lower values indicate closer alignment between the predicted and actual sequences. Complete experimental results are provided in Appendix D.1. 

<table><tr><td>Models</td><td colspan="2">TimeStacker(ours)</td><td colspan="2">SOFTS(2024)</td><td colspan="2">SparseTSF(2024)</td><td colspan="2">iTransformer(2024)</td><td colspan="2">TimeMixer(2024)</td><td colspan="2">SAMformer(2024)</td><td colspan="2">PatchTST(2023)</td><td colspan="2">Crossformer(2023)</td><td colspan="2">DLinear(2023)</td><td colspan="2">RLinear(2023)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.433</td><td>0.423</td><td>0.449</td><td>0.442</td><td>0.441</td><td>0.425</td><td>0.454</td><td>0.447</td><td>0.447</td><td>0.440</td><td>0.444</td><td>0.432</td><td>0.469</td><td>0.454</td><td>0.529</td><td>0.522</td><td>0.456</td><td>0.452</td><td>0.446</td><td>0.434</td></tr><tr><td>ETTh2</td><td>0.368</td><td>0.390</td><td>0.373</td><td>0.400</td><td>0.421</td><td>0.438</td><td>0.383</td><td>0.407</td><td>0.364</td><td>0.395</td><td>0.383</td><td>0.401</td><td>0.387</td><td>0.407</td><td>0.942</td><td>0.684</td><td>0.559</td><td>0.515</td><td>0.374</td><td>0.398</td></tr><tr><td>ETTm1</td><td>0.381</td><td>0.381</td><td>0.393</td><td>0.403</td><td>0.425</td><td>0.401</td><td>0.407</td><td>0.410</td><td>0.381</td><td>0.395</td><td>0.415</td><td>0.407</td><td>0.387</td><td>0.400</td><td>0.513</td><td>0.496</td><td>0.403</td><td>0.407</td><td>0.414</td><td>0.407</td></tr><tr><td>ETTm2</td><td>0.274</td><td>0.316</td><td>0.287</td><td>0.330</td><td>0.297</td><td>0.331</td><td>0.288</td><td>0.332</td><td>0.275</td><td>0.323</td><td>0.285</td><td>0.327</td><td>0.281</td><td>0.326</td><td>0.757</td><td>0.610</td><td>0.350</td><td>0.401</td><td>0.286</td><td>0.327</td></tr><tr><td>Traffic</td><td>0.508</td><td>0.335</td><td>0.409</td><td>0.267</td><td>0.578</td><td>0.350</td><td>0.428</td><td>0.282</td><td>0.484</td><td>0.297</td><td>0.595</td><td>0.382</td><td>0.481</td><td>0.304</td><td>0.550</td><td>0.304</td><td>0.625</td><td>0.383</td><td>0.626</td><td>0.378</td></tr><tr><td>Electricity</td><td>0.194</td><td>0.275</td><td>0.174</td><td>0.264</td><td>0.222</td><td>0.289</td><td>0.178</td><td>0.270</td><td>0.182</td><td>0.272</td><td>0.217</td><td>0.295</td><td>0.205</td><td>0.290</td><td>0.244</td><td>0.334</td><td>0.212</td><td>0.300</td><td>0.219</td><td>0.298</td></tr><tr><td>Weather</td><td>0.243</td><td>0.264</td><td>0.255</td><td>0.278</td><td>0.290</td><td>0.302</td><td>0.258</td><td>0.278</td><td>0.240</td><td>0.271</td><td>0.264</td><td>0.285</td><td>0.259</td><td>0.348</td><td>0.259</td><td>0.315</td><td>0.265</td><td>0.317</td><td>0.272</td><td>0.291</td></tr><tr><td>Exchange</td><td>0.336</td><td>0.389</td><td>0.348</td><td>0.395</td><td>0.365</td><td>0.401</td><td>0.360</td><td>0.403</td><td>0.355</td><td>0.399</td><td>0.346</td><td>0.399</td><td>0.367</td><td>0.404</td><td>0.940</td><td>0.707</td><td>0.354</td><td>0.414</td><td>0.378</td><td>0.417</td></tr></table>

# 3.5. Theoretical Analysis

Definition 3.1. The statistical properties of non-stationary signals (e.g., mean, variance, autocorrelation function) change over time, and their frequency characteristics also dynamically evolve with time. These can be expressed using the Fourier series as follows:

$$
x (t) = a _ {0} (t) + \sum_ {n = 1} ^ {\infty} (a _ {n} (t) \cos {(2 \pi n f _ {0} t)} + b _ {n} (t) \sin {(2 \pi n f _ {0} t)}) \tag {14}
$$

Let $x(t)$ denote a non-stationary signal, where $a_{n}(t)$ and $b_{n}(t)$ are time-varying nonlinear functions that capture the dynamic variations in the signal components. This implies that the frequency content of a non-stationary signal varies over time.

From a frequency-domain perspective, time series forecasting fundamentally involves analyzing the latent frequency characteristics within historical sequences to identify the temporal variation patterns of their Fourier coefficients $a_{n}(t)$ and $b_{n}(t)$ . These coefficients encode the amplitude and phase information of the signal at specific frequencies, serving as key indicators of its time-varying properties in the frequency domain. Therefore, capturing and modeling the variation patterns of these coefficients can effectively reveal the periodicity and trends in time series, establishing a robust foundation for accurate future forecasting.

Theorem 3.2. The time-frequency uncertainty principle states that a signal cannot achieve arbitrarily high resolution simultaneously in both the time and frequency domains, reflecting the resolution limit of a signal in the time-frequency domain. The mathematical expression is as follows:

$$
\Delta t \cdot \Delta f \geq \frac {1}{4 \pi} \tag {15}
$$

Here $\Delta t$ denotes the standard deviation in the time domain, while, and $\Delta f$ denotes the standard deviation in the frequency domain. This reveals an inherent limitation in precisely measuring both the temporal location and frequency components of a signal: achieving arbitrarily high time and frequency resolution simultaneously is fundamentally impossible.

To overcome this limitation, TimeStacker sequentially stacks patches of varying sizes, from large to small, effectively reducing $\Delta f$ (frequency resolution) while progressively enhancing time resolution. This approach enables the model to capture the dynamic evolution of the spectrum over time at multiple temporal resolutions, allowing it to focus on both fine-grained local details and overarching global trends.

Specifically, larger patches yield lower time resolution and higher frequency resolution, enabling the model to capture global periodicity and long-term trends within the signal. Conversely, smaller patches yield higher time resolution, allowing the model to accurately capture dynamic variations in local signal components.

# 3.6. Complexity Analysis

The Inter-Patch Frequency-based Attention Module distinguishes itself from traditional self-attention mechanisms primarily in the computation of Q and K. To facilitate explanation, we use the calculation of Q as an example for analyzing computational complexity.

Let the input for similarity computation be $\mathcal{X} \in \mathbb{R}^{k \times p}$ .

Table 2. Ablation Experiment Results. Ablation experiments were conducted on the FreqAttention module of TimeStacker, involving replacement(REPLACE) and removal(W/O) operations based on traditional self-attention. MAE and MSE were used as evaluation metrics, with all input sequence lengths set to 96. 

<table><tr><td rowspan="2" colspan="2">Method</td><td>W/O FreqAttention</td><td>REPLACE FreqAttention</td><td>W/O Hadamard</td><td>REPLACE Hadamard</td><td>TimeStacker</td></tr><tr><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.388 0.395</td><td>0.386 0.395</td><td>0.384 0.393</td><td>0.381 0.393</td><td>0.379 0.385</td></tr><tr><td>192</td><td>0.436 0.420</td><td>0.431 0.424</td><td>0.435 0.425</td><td>0.431 0.424</td><td>0.429 0.416</td></tr><tr><td>336</td><td>0.476 0.444</td><td>0.472 0.445</td><td>0.472 0.444</td><td>0.474 0.447</td><td>0.459 0.436</td></tr><tr><td>720</td><td>0.480 0.471</td><td>0.474 0.469</td><td>0.466 0.464</td><td>0.476 0.468</td><td>0.464 0.455</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.288 0.338</td><td>0.296 0.343</td><td>0.292 0.338</td><td>0.293 0.342</td><td>0.280 0.327</td></tr><tr><td>192</td><td>0.398 0.397</td><td>0.389 0.396</td><td>0.390 0.399</td><td>0.393 0.394</td><td>0.373 0.385</td></tr><tr><td>336</td><td>0.418 0.426</td><td>0.416 0.424</td><td>0.419 0.426</td><td>0.414 0.423</td><td>0.407 0.416</td></tr><tr><td>720</td><td>0.418 0.437</td><td>0.419 0.437</td><td>0.420 0.439</td><td>0.418 0.436</td><td>0.412 0.431</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.087 0.206</td><td>0.085 0.203</td><td>0.086 0.204</td><td>0.085 0.203</td><td>0.084 0.200</td></tr><tr><td>192</td><td>0.179 0.303</td><td>0.173 0.293</td><td>0.171 0.294</td><td>0.171 0.293</td><td>0.171 0.293</td></tr><tr><td>336</td><td>0.323 0.415</td><td>0.317 0.411</td><td>0.320 0.412</td><td>0.316 0.410</td><td>0.314 0.408</td></tr><tr><td>720</td><td>0.807 0.675</td><td>0.791 0.669</td><td>0.780 0.666</td><td>0.796 0.672</td><td>0.776 0.656</td></tr></table>

The computation of Q is detailed in Equations (8) and (9). The computational complexity of the Fast Fourier Transform (FFT) is $O(kplog_{2}p)$ , while the complexity of the Hadamard product is $O\left(k\left(\left\lfloor\frac{p}{2}\right\rfloor+1\right)\right)$ approximately $O\left(k\frac{p}{2}\right)$ . Substituting L = kp into the formulas, the complexity of computing Q is derived as $O\left(L\left(\log_{2}p+\frac{1}{2}\right)\right)$ , which approximates to $O(Llog_{2}p)$ . In comparison, the computational complexity of traditional self-attention is $O(kp^{2})$ , which simplifies to $O(Lp)$ . This demonstrates that the proposed method significantly reduces complexity, particularly for high-dimensional inputs, where its advantages become especially evident.

This method leverages the frequency characteristics of signals to alleviate the performance bottlenecks in traditional self-attention models, which are caused by high computational complexity in the time domain. This optimization provides a more efficient computational pathway for time series forecasting, maintaining the model's predictive accuracy and robustness.

# 4. Experiments

In this section, extensive experiments and analyses are conducted on real-world datasets from various domains, encompassing both long-term and short-term forecasting tasks, to comprehensively evaluate the performance and computational efficiency of TimeStacker.

# 4.1. Experimental Setup

Dataset. To evaluate the performance of TimeStacker, experiments were conducted on multiple widely used real-world datasets, following a processing approach similar to iTransformer(Liu et al., 2023). These datasets span various domains, including energy, transportation, and weather. Specifically, the datasets used in the experiments include ETT (comprising four subsets: ETTh1, ETTh2, ETTm1, and ETTm2), Weather, Traffic, Electricity, and Exchange. Each dataset was divided into training, validation, and test sets in a 6:2:2 ratio. For more details on the datasets, refer to Appendix A.

Implementation Details. TimeStacker employs Mean Absolute Error (MAE) as the loss function and utilizes Adam as the optimizer. The learning rate is set to $1 \times 10^{-3}$ , the weight decay to $1 \times 10^{-3}$ , and $\epsilon$ to $1 \times 10^{-8}$ . The first and second moment decay rates are set to 0.9 and 0.999, respectively. All experiments were implemented using PyTorch 2.0(Paszke et al., 2019) and conducted on an NVIDIA RTX 4080 GPU with 16GB of memory. Detailed parameter settings can be found in Appendix B.2.

# 4.2. Experimental Results

Baseline. We compared TimeStacker with state-of-the-art and representative models proposed in the past two years to evaluate its effectiveness. The main baselines include: (1) Transformer-based models such as Pathformer(Chen et al., 2024), SAMformer(Ilbert et al.), PatchTST(Nie et al., 2022), and Crossformer(Zhang & Yan, 2023); (2) Linear-layer-based models such as SparseTSF(Lin et al., 2024), SOFTS(Han et al., 2024a), TimeMixer(Wang et al., 2024a), DLinear(Zeng et al., 2023), and RLinear(Li et al., 2023).

Main Results. Table 1 presents the time series forecasting results. Red highlights the best performance, while blue with underlining indicates the second-best performance. Lower Mean Squared Error (MSE) and MAE values cor-

![](images/d43809bbe43708e724107d28c993a9d449b9c49a0d471e3803ca4b49c5101c14.jpg)

<details>
<summary>line</summary>

| Look-Back Length | TimeStacker(ours) | SAMformer | iTransformer | SOFTS | timeMixer |
| ---------------- | ----------------- | --------- | ------------ | ----- | --------- |
| 96               | 0.43              | 0.44      | 0.45         | 0.45  | 0.44      |
| 192              | 0.42              | 0.44      | 0.44         | 0.43  | 0.43      |
| 336              | 0.41              | 0.43      | 0.44         | 0.43  | 0.42      |
| 720              | 0.40              | 0.43      | 0.43         | 0.43  | 0.42      |
</details>

![](images/17929c1b3a97f944a8f860561411f9dc508f7b20dd3b6b5a27b04dbea1a7e9c1.jpg)

<details>
<summary>line</summary>

| Look-Back Length | TimeStacker(ours) | SAMformer | iTTransformer | SOFTS | timeMixer |
| ---------------- | ----------------- | --------- | ------------ | ----- | --------- |
| 96               | 0.37              | 0.38      | 0.38         | 0.37  | 0.36      |
| 192              | 0.36              | 0.38      | 0.37         | 0.38  | 0.36      |
| 336              | 0.35              | 0.35      | 0.38         | 0.37  | 0.35      |
| 720              | 0.34              | 0.35      | 0.39         | 0.38  | 0.38      |
</details>

![](images/a1deabb6e355a01e5d2b535140b2d65d76bbbef65073e32f5d58b497e1b0ade6.jpg)

<details>
<summary>line</summary>

| Look-Back Length | TimeStacker(ours) | SAMformer | iTransformer | SOFTS | timeMixer |
| ---------------- | ----------------- | --------- | ----------- | ----- | --------- |
| 96               | 0.34              | 0.35      | 0.36        | 0.35  | 0.35      |
| 192              | 0.33              | 0.35      | 0.35        | 0.33  | 0.35      |
| 336              | 0.34              | 0.38      | 0.34        | 0.36  | 0.37      |
| 720              | 0.34              | 0.42      | 0.37        | 0.38  | 0.38      |
</details>

Figure 4. Impact of Look-Back Lengths. All models use look-back window lengths of 96, 192, 336, and 720, with MSE serving as the evaluation metric. The results are calculated as the average across four different prediction lengths $\{96, 192, 336, 720\}$ , to assess the performance of different models under varying look-back lengths.

Table 3. The impact of different look-back window lengths on model prediction performance. The look-back window lengths are {96, 192, 336, 720}, and the prediction lengths are {96, 192, 336, 720}. The evaluation metric is MSE. 

<table><tr><td rowspan="2" colspan="2">Method</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>MSE</td><td>MSE</td><td>MSE</td><td>MSE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.379</td><td>0.370</td><td>0.362</td><td>0.364</td></tr><tr><td>192</td><td>0.429</td><td>0.424</td><td>0.401</td><td>0.399</td></tr><tr><td>336</td><td>0.459</td><td>0.448</td><td>0.428</td><td>0.42</td></tr><tr><td>720</td><td>0.464</td><td>0.447</td><td>0.432</td><td>0.427</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.280</td><td>0.283</td><td>0.270</td><td>0.267</td></tr><tr><td>192</td><td>0.373</td><td>0.366</td><td>0.348</td><td>0.336</td></tr><tr><td>336</td><td>0.407</td><td>0.397</td><td>0.362</td><td>0.354</td></tr><tr><td>720</td><td>0.412</td><td>0.409</td><td>0.399</td><td>0.389</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.084</td><td>0.084</td><td>0.085</td><td>0.085</td></tr><tr><td>192</td><td>0.171</td><td>0.173</td><td>0.174</td><td>0.171</td></tr><tr><td>336</td><td>0.314</td><td>0.317</td><td>0.320</td><td>0.321</td></tr><tr><td>720</td><td>0.776</td><td>0.756</td><td>0.792</td><td>0.772</td></tr></table>

respond to higher predictive accuracy. Compared to other models, TimeStacker demonstrates superior performance across multiple datasets (ETTh1\~ETTm2, Weather, Exchange). Notably, in comparison with iTransformer, TimeStacker achieves significant improvements. For example, on the ETTm1 dataset, MAE is reduced by 7.07%; on the Weather dataset, MAE is reduced by 5.04%; and on the Exchange dataset, MSE is reduced by 6.67%. These results clearly demonstrate that TimeStacker not only excels in time series forecasting accuracy but also exhibits strong generalization capabilities. In particular, in critical application domains such as energy, weather, and finance, TimeStacker showcases broad applicability and practical utility.

# 4.3. Model Analysis

Ablation Study. A series of ablation experiments were conducted to evaluate the core components of TimeStacker, namely the FreqAttention module. These experiments involved replacement (REPLACE) and removal (W/O) operations, as outlined below:

- W/O FreqAttention: Removed the FreqAttention module from TimeStacker.   
- REPLACE FreqAttention: Replaced the FreqAttention module in TimeStacker with traditional Self-Attention.   
- W/O Hadamard: Removed the Hadamard product operation from the FreqAttention module.   
- REPLACE Hadamard: Replaced the Hadamard product operation in the FreqAttention module with a linear transformation.

As shown in Table 2, incorporating frequency information into similarity computation significantly enhances the accuracy of time series forecasting compared to traditional self-attention methods. Introducing the Hadamard product in the computation of Q and K further reveals hidden pattern features within the sequence. The proposed FreqAttention exhibits a distinct advantage in capturing time series dynamics, confirming its effectiveness in modeling the complex characteristics of dynamic signals.

Impact of Look-Back Lengths. In general, longer look-back windows provide more information about the sequence, allowing the model to capture sequence features more comprehensively. However, longer sequences may also introduce additional noise, potentially affecting prediction accuracy. To examine the impact of look-back window length on model performance, experiments were conducted using varying window lengths of $\{96, 192, 336, 720\}$ .

The detailed results, presented in Table 3, indicate that prediction accuracy on the Exchange dataset remained largely unchanged regardless of window length. In contrast, for the ETTh1 and ETTh2 datasets, accuracy significantly improved as the look-back window length increased. These

findings suggest that TimeStacker effectively utilizes the additional sequence information from longer windows to enhance forecasting performance.

Additionally, our model was compared with SOFT, TimeMixer, iTransformer, and SAMformer. As illustrated in Figure 4, the results demonstrate the superior ability of TimeStacker in capturing sequence features. Further details can be found in Appendix D.2.

![](images/2f4f0b1116d91668cd1be281811094bb37c6dee036d4a67f5b830a10af42d8c0.jpg)

<details>
<summary>bubble</summary>

| Model | Training Time (ms/iter) | MSE |
|---|---|---|
| Crossformer | 162 | 0.53 |
| iTransformer | 153 | 0.49 |
| TimeMixer | 143 | 0.47 |
| SoftS | 130 | 0.46 |
| TimeStacker(ours) | 115 | 0.43 |
| DLinear | 101 | 0.46 |
| PatchTST | 76 | 0.45 |
| RLinear | 59 | 0.44 |
| SAMformer | 68 | 0.44 |
| SparesTSF | 98 | 0.44 |
| 40MB | 40 | 0.52 |
| 200MB | 20 | 0.52 |
| 1GB | 1 | 0.52 |
TimeStacker(ours) - TimeStacker(ours) - TimeMixer - SoftS - iTransformer - Crossformer - Memory Footprint
TimeStacker(ours) - TimeMixer - Crossformer - Crossformer - Memory Footprint
TimeStacker(ours) - TimeMixer - Crossformer - Crossformer - Memory Footprint
TimeStacker(ours) - TimeMixer - Crossformer - Crossformer - Memory Footprint
TimeStacker(ours) - TimeMixer - Crossformer - Crossformer - Memory Footprint
TimeStacker(ours) - TimeMixer - Crossformer - Crossformer - Memory Footprint
TimeStacker(ours) - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center - Data Center
</details>

Figure 5. Performance Comparison of Models. All models were evaluated on the ETTh1 dataset using an experimental setup with a look-back length of 96, a prediction length of 720, and a batch size of 128. The final evaluation results were obtained by averaging the outcomes of five independent trials. The evaluation metrics included memory usage during runtime, execution time, and MSE.

Model Effectiveness. In Section 3.6, the computational complexity of the core module in TimeStacker was derived as $O(L\log_2p)$ . To provide a more comprehensive evaluation of the model's efficiency, additional analyses were conducted on memory usage and training time. The ETTh1 dataset was used for this evaluation, with all models configured to a look-back length of 96, a prediction length of 720, and a batch size of 128. The results were averaged over five independent trials.

As illustrated in Figure 5, the models with the lowest memory usage were SparseTSF and DLinear, each consuming only 13.7 MB, while the fastest model was RLinear, with an iteration time of just 59 ms. In comparison, TimeStacker required 24.9 MB of memory and had an iteration time of 115 ms, while achieving the lowest MSE among all models. Clearly, TimeStacker demonstrates strong predictive performance and computational efficiency, achieving the lowest prediction error while maintaining relatively low memory consumption and runtime.

# 5. Conclusion, Limitation, and Future Works

Conclusion. This paper introduces the TimeStacker framework, which leverages the frequency variations of non-stationary signals over time. By employing a unique stacking mechanism and balancing modeling between the time and frequency domains, TimeStacker effectively captures the complex characteristics of non-stationary time series. To further enhance the stacking mechanism and fully utilize frequency information, a frequency-based self-attention module was designed. This module not only mitigates the impact of spectral leakage but also enhances feature representation by comprehensively modeling both frequency and time domain information.

Additionally, extensive experiments were conducted on various real-world datasets. The results demonstrate that TimeStacker effectively extracts the dynamic characteristics of non-stationary time series and achieves a comprehensive feature representation. Performance analysis further reveals that this approach delivers high prediction accuracy while significantly reducing computational complexity, highlighting its potential for applications in resource-constrained scenarios.

Limitation and Future Work. While TimeStacker effectively captures time series features, comparative experiments indicate a slight decline in predictive performance as the number of variables in the time series increases. To investigate this phenomenon, additional experiments were conducted (in Appendix D.5) by varying the number of variables and testing the model on the Traffic and Electricity datasets. The results suggest that the model may encounter performance bottlenecks when handling multivariate time series.

Therefore, future work will focus on optimizing multi-channel prediction strategies by designing additional modules to enhance TimeStacker's performance on multivariate time series tasks.

# Acknowledgements

This study is supported by the Young Scientists Fund of the National Natural Science Foundation of China(Grant No.62306094), Independent Research Exploration Projects of Songjiang Laboratory(Grant No.SL20230203), Project supported by the Special Funds of the National Natural Science Foundation of China(Grant No. 32441112).

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Box, G. E., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. Time series analysis: forecasting and control. John Wiley & Sons, 2015.   
Čepulionis, P. and Lukoševičiūtė, K. Electrocardiogram time series forecasting and optimization using ant colony optimization algorithm. Mathematical Models in Engineering, 2(1):69–77, 2016.   
Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., and Dubrawski, A. Nhits: Neural hierarchical interpolation for time series forecasting. 37(6):6989–6997, 2023.   
Chen, P., Zhang, Y., Cheng, Y., Shu, Y., Wang, Y., Wen, Q., Yang, B., and Guo, C. Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting. arXiv preprint arXiv:2402.05956, 2024.   
Cheng, D., Yang, F., Xiang, S., and Liu, J. Financial time series forecasting with multi-modality graph neural network. Pattern Recognition, 121:108218, 2022.   
Chou, J.-S. and Tran, D.-S. Forecasting energy consumption time series using machine learning techniques based on usage patterns of residential householders. Energy, 165:709–726, 2018.   
Cohen, L. Time-frequency analysis, volume 778. Prentice Hall PTR New Jersey, 1995.   
Cox, D. R. The theory of stochastic processes. Routledge, 2017.   
De Livera, A. M., Hyndman, R. J., and Snyder, R. D. Forecasting time series with complex seasonal patterns using exponential smoothing. Journal of the American statistical association, 106(496):1513–1527, 2011.   
Dubey, A. K., Kumar, A., García-Díaz, V., Sharma, A. K., and Kanhaiya, K. Study and analysis of sarima and lstm in forecasting time series data. Sustainable Energy Technologies and Assessments, 47:101474, 2021.   
Han, L., Chen, X.-Y., Ye, H.-J., and Zhan, D.-C. Softs: Efficient multivariate time series forecasting with series-core fusion. arXiv preprint arXiv:2404.14197, 2024a.   
Han, L., Ye, H.-J., and Zhan, D.-C. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. IEEE Transactions on Knowledge and Data Engineering, 2024b.   
Ilbert, R., Odonnat, A., Feofanov, V., Virmaux, A., Paolo, G., Palpanas, T., and Redko, I. Samformer: Unlocking the potential of transformers in time series forecasting with

sharpness-aware minimization and channel-wise attention. In Forty-first International Conference on Machine Learning.

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2021.

Lee, Y.-S. and Tong, L.-I. Forecasting time series using a methodology based on autoregressive integrated moving average and genetic programming. Knowledge-Based Systems, 24(1):66–72, 2011.

Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.-X., and Yan, X. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. Advances in neural information processing systems, 32, 2019.

Li, Z., Qi, S., Li, Y., and Xu, Z. Revisiting long-term time series forecasting: An investigation on linear mapping. arXiv preprint arXiv:2305.10721, 2023.

Lin, S., Lin, W., Wu, W., Chen, H., and Yang, J. Sparsetsf: Modeling long-term time series forecasting with 1k parameters. arXiv preprint arXiv:2405.00946, 2024.

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625, 2023.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730, 2022.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

Sagheer, A. and Kotb, M. Time series forecasting of petroleum production using deep lstm recurrent networks. Neurocomputing, 323:203–213, 2019.

Sezer, O. B., Gudelek, M. U., and Ozbayoglu, A. M. Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied soft computing, 90:106181, 2020.

Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., and Zhou, J. Timemixer: Decomposable multiscale mixing for time series forecasting. arXiv preprint arXiv:2405.14616, 2024a.

Wang, Y., Wu, H., Dong, J., Liu, Y., Long, M., and Wang, J. Deep time series models: A comprehensive survey and benchmark. arXiv preprint arXiv:2407.13278, 2024b.   
Wu, H., Zhou, H., Long, M., and Wang, J. Interpretable weather forecasting for worldwide stations with a unified deep model. Nature Machine Intelligence, 5(6):602–611, 2023.   
Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023.   
Zhang, Y. and Yan, J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The eleventh international conference on learning representations, 2023.   
Zhou, K., Wang, W., Huang, L., and Liu, B. Comparative study on the time series forecasting of web traffic based on statistical model and generative adversarial model. Knowledge-Based Systems, 213:106467, 2021.   
Zhou, T., Ma, Z., Wen, Q., Sun, L., Yao, T., Yin, W., Jin, R., et al. Film: Frequency improved legendre memory model for long-term time series forecasting. Advances in neural information processing systems, 35:12677–12690, 2022a.   
Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., and Jin, R. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. pp. 27268–27286, 2022b.

# A. DATASET DESCRIPTION

Time series forecasting experiments were conducted on widely used real-world datasets (details provided in Table 4). The datasets are described as follows:

- ETT Dataset: Comprising two hourly datasets (ETTh) and two 15-minute datasets (ETTm), this dataset includes load features of seven oil and electricity transformers recorded from July 2016 to July 2018.   
- Traffic Dataset: This dataset contains hourly road occupancy rates recorded by sensors on San Francisco freeways from 2015 to 2016.   
- Electricity Dataset: Recording hourly electricity consumption for 321 customers, this dataset spans the period from 2012 to 2014.   
- Weather Dataset: Comprising 21 weather indicators, including air temperature and humidity, this dataset was recorded every 10 minutes throughout 2020.   
- Exchange-rate Dataset: This dataset contains daily exchange rates for eight countries, collected from 1990 to 2016.

Table 4. Variates denotes the number of variables in each dataset, Dataset Size denotes the number of time points in the dataset, Frequency denotes the sampling frequency of the dataset, and Information denotes the category information of the dataset. 

<table><tr><td>Dataset</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Traffic</td><td>Electricity</td><td>Weather</td><td>Exchange-rate</td></tr><tr><td>Variates</td><td>7</td><td>7</td><td>7</td><td>7</td><td>862</td><td>321</td><td>21</td><td>8</td></tr><tr><td>Dataset Size</td><td>14,307</td><td>14,307</td><td>57,507</td><td>57,507</td><td>17,451</td><td>26,211</td><td>52,603</td><td>7,207</td></tr><tr><td>Frequency</td><td>Hourly</td><td>Hourly</td><td>15min</td><td>15min</td><td>Hourly</td><td>Hourly</td><td>10min</td><td>Daily</td></tr><tr><td>Information</td><td>Electricity</td><td>Electricity</td><td>Electricity</td><td>Electricity</td><td>Transportation</td><td>Electricity</td><td>Weather</td><td>Economy</td></tr></table>

# B. IMPLEMENT DETAILS

# B.1. TimeStacker

The complete algorithmic process of TimeStacker is outlined as follows. It takes the time series $x$ as input and generates the corresponding prediction $\widetilde{x}$ .

Algorithm 1 TimeStacker   
Input: Historical look-back window $x_{t-T+1:t} \in R^{T}$ , Patch size list $P = \{p_{1}, p_{2}, \ldots, p_{L}\}$ Output: Forecasting horizon $\widetilde{x}_{t+1:t+H} \in R^{H}$ , $\bar{x} = \text{Normalize}(x) / * \text{Normalizer the input sequence with mean and variance */}$ for $p_{i}$ in P do $\bar{x} = \text{SmoothLayer}(\bar{x}) / * \text{Apply conv1d with kernel size of } p_{i} */$ $\bar{x} = \text{Reshape}(\bar{x}, (p_{i}, T/p_{i}))$ $\bar{x} = \text{FreqAttention}(\bar{x}) / * \text{Aggregate different patches */}$ end for $\bar{x} = \text{Reshape}(\bar{x}, T)$ $\widetilde{x} = \text{Denormalize}(\text{predictor}(\bar{x})) / * \text{Apply linear layer for prediction */}$

# B.2. Model Configuration

TimeStacker primarily consists of multiple Stacker Blocks, with the number of blocks determined by the Patchlist parameter. The number of elements in Patchlist defines the depth of TimeStacker, while the size of each element determines the observation window for the corresponding block. Given the varying sampling frequencies and physical characteristics of different datasets, customized configurations are adopted for each dataset. Detailed information is provided in Table 5.

Table 5. Model Configuration. 

<table><tr><td>DataSet</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Traffic</td><td>Electricity</td><td>Weather</td><td>Exchange</td></tr><tr><td>Patch Size List</td><td>(96,48,32,24,16,12)</td><td>(96,48,32,24,16,12)</td><td>(48,32,24,16)</td><td>(48,32,24,16)</td><td>(96,48,32,16,12)</td><td>(96,48,24,12)</td><td>(96,48,32,24,12)</td><td>(96,48,24)</td></tr></table>

# C. EXPERIMENT DETAILS

# C.1. Metric Details

To comprehensively evaluate model performance across different datasets, Mean Absolute Error (MAE) and Mean Squared Error (MSE) were selected as evaluation metrics. These metrics assess prediction accuracy from different perspectives, offering a comprehensive and intuitive basis for model comparison.

MAE:

$$
M A E = \frac {1}{L} \sum_ {l = 1} ^ {L} \left| x _ {i} - \widetilde {x} _ {i} \right| \tag {16}
$$

MSE:

$$
M S E = \frac {1}{L} \sum_ {l = 1} ^ {L} \left(x _ {i} - \widetilde {x} _ {i}\right) ^ {2} \tag {17}
$$

Here, $x_{i}, \widetilde{x}_{i} \in \mathbb{R}^{C \times L}$ denote the ground truth and predicted values, respectively, where $L$ represents the number of time points, and $C$ is the number of channels. The MSE primarily emphasizes the squared differences between predicted and actual values, thereby amplifying the impact of larger errors. This makes MSE particularly sensitive to outliers and suitable for evaluating the model's ability to capture global trends. On the other hand, the MAE directly computes the average absolute differences between predicted and actual values. It focuses on assessing the model's overall control of error magnitude in practical applications and is less affected by outliers.

# C.2. Baseline

Representative methods from the past two years in time series forecasting were selected as baseline approaches to comprehensively evaluate the performance of the proposed model. A detailed introduction to these methods is provided below:

DLinear: Utilizes a simple yet effective single-layer linear model to capture temporal relationships between input and output sequences.

Crossformer: Segments multivariate time series data along each dimension, embeds them into feature vectors, and employs a two-stage attention mechanism to efficiently capture both intra- and inter-series dependencies.

PatchTST: Splits time series data into subsequence-level patches to extract local semantics, adopting a channel-independent strategy where each channel shares the same embedding and Transformer weights across all sequences.

RLinear: Uses linear mapping to model periodic features in multivariate time series, demonstrating robustness across different periods as input length increases.

SAMformer: Enhances the model's generalization ability by leveraging sharpness-aware optimization techniques.

TimeMixer: Addresses complex temporal variations in time series forecasting through a multi-scale mixing perspective, improving complementary predictions from multi-scale sequences by decoupling variations.

iTransformer: Reverses the Transformer structure by encoding each individual series as variable tokens without modifying any existing modules.

SparseTSF: Simplifies the forecasting task by decoupling the periodicity and trend of time series data using cross-period sparse prediction techniques.

SOFTS: Introduces a novel centralized structure to transfer information across channels, addressing the limitations of

channel-independent approaches in leveraging inter-channel correlations and mitigating robustness challenges in channel-dependent methods.

# C.3. Experiment details

TimeStacker adopts MAE as the loss function and uses Adam as the optimizer, with a learning rate set to $1 \times 10^{-3}$ , weight decay set to $1 \times 10^{-3}$ , and epsilon set to $1 \times 10^{-8}$ . The first-order and second-order moments are configured as 0.9 and 0.999, respectively. All experiments are implemented in PyTorch 2.0 and executed on an NVIDIA RTX 4080 GPU with 16GB of memory.

In the experiments, the same dataset processing approach as TimeMixer was followed, ensuring that datasets were divided into training, validation, and test sets in a strict temporal sequence with a 6:2:2 ratio to prevent data leakage. The look-back window length was fixed at 96, and the forecasting horizons were set to $\{96,192,336,720\}$ .

# D. FULL RESULTS

# D.1. Complete Experimental Result

The complete experimental results are presented in Table 12. Experiments were conducted on six widely used real-world datasets spanning domains such as energy, traffic, and weather to comprehensively validate the effectiveness of the proposed method. To further assess model performance, comparisons were made against several representative models in the field.

The results demonstrate that the proposed method achieves outstanding performance across multiple datasets, consistently surpassing most baseline models in both predictive accuracy and computational efficiency. Notably, the method exhibits significant advantages in handling complex non-stationary time series, effectively capturing both global trends and local details within the signals. These findings further underscore the generalizability and robustness of the approach, offering an efficient and accurate solution for time series forecasting tasks.

The results of the mean and standard deviation of MSE and MAE for multiple runs of all datasets are shown in Table 6.

Table 6. Mean/Standard deviation for MSE and MAE across multiple runs. 

<table><tr><td>DataSet</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Traffic</td><td>Electricity</td><td>Weather</td><td>Exchange</td></tr><tr><td>MSE</td><td>0.433/0.00145</td><td>0.368/0.00091</td><td>0.381/0.00119</td><td>0.274/0.00061</td><td>0.508/0.00052</td><td>0.194/0.00056</td><td>0.243/0.00092</td><td>0.336/0.00101</td></tr><tr><td>MAE</td><td>0.423/0.00167</td><td>0.390/0.00057</td><td>0.381/0.00052</td><td>0.316/0.00042</td><td>0.335/0.00087</td><td>0.275/0.00077</td><td>0.264/0.00042</td><td>0.389/0.00137</td></tr></table>

# D.2. Impact of Look-Back Lengths

The impact of look-back length on model performance was examined, with detailed experimental results presented in Table 11. Prediction experiments were conducted using look-back lengths of $\{96, 192, 336, 720\}$ , and the results were compared against SOFT, TimeMixer, iTransformer, and SAMformer. The results demonstrate that as the look-back length increases, the predictive accuracy of TimeStacker improves significantly. Compared to other models, TimeStacker effectively utilizes the extended sequence information, resulting in a substantial enhancement in prediction performance. This outcome validates the advantages and robustness of the proposed method in time series modeling.

# D.3. Impact of Patch Size List

To demonstrate how TimeStacker adapts to various non-stationary signals, we configured the parameter Patch Size List and conducted experiments on the ETTm1 dataset. The results are shown in Table 7. These results indicate that employing various window combinations can more effectively capture the underlying dynamic patterns of the sequence, thereby improving prediction performance.

# D.4. Complexity Analysis

To enable a more in-depth analysis of the model's computational complexity, the input length was increased to evaluate its temporal and spatial complexity (GPU Memory (MB) / Training Time (ms/iter)). The experimental results are shown in

Table 7. Impact of different patch size list. 

<table><tr><td>Patch Size List</td><td>[16, 16, 16, 16]</td><td>[16, 16, 16, 24]</td><td>[16, 16, 16, 32]</td><td>[16, 16, 16, 48]</td><td>[16, 16, 24, 32]</td><td>[16, 16, 24, 48]</td><td>[16, 24, 32, 48]</td></tr><tr><td>MSE</td><td>0.465</td><td>0.468</td><td>0.468</td><td>0.465</td><td>0.463</td><td>0.463</td><td>0.460</td></tr><tr><td>MAE</td><td>0.433</td><td>0.439</td><td>0.436</td><td>0.431</td><td>0.431</td><td>0.430</td><td>0.428</td></tr></table>

Table 8. Complexity Analysis(GPU Memory(MB)/Training Time(ms/iter)). 

<table><tr><td>Input Length</td><td>Timestacker</td><td>SparseTSF</td><td>TimeMixer</td><td>DLinear</td><td>PatchTST</td><td>Crossformer</td></tr><tr><td>192</td><td>28.8/134</td><td>15.6/108</td><td>518.3/180</td><td>13.6/45</td><td>145.8/87.8</td><td>5214/238</td></tr><tr><td>384</td><td>29.3/133</td><td>16.1/112</td><td>875.4/193</td><td>16.7/49</td><td>334.7/90.1</td><td>5734/273</td></tr><tr><td>768</td><td>34.3/137</td><td>20.6/115</td><td>1763/202</td><td>23.2/99</td><td>830.0/93.3</td><td>6814/342</td></tr><tr><td>1536</td><td>59.2/137</td><td>28.7/127</td><td>3744/282</td><td>34.8/108</td><td>2404/137</td><td>9016/1007</td></tr><tr><td>3072</td><td>110.7/134</td><td>44.1/131</td><td>7376/462</td><td>59.1/108</td><td>7832/1315</td><td>12470/3121</td></tr></table>

Table 8.

# D.5. Additional Experiments

To investigate the impact of the number of variables on model performance, we selected the Traffic and Electricity datasets, where our model performed less effectively, and conducted comparative experiments with existing models including SOFTS, iTransformer, and TimeMixer. These experiments focused specifically on single-variable time series forecasting tasks.

The results not only revealed the differences in how various models handle single-channel time series but also provided a clearer understanding of how the number of variables affects model performance. Under single-channel conditions, our model demonstrated a stronger ability to capture the dynamic changes of individual sequences, whereas certain other models appeared to rely on correlations between multiple channels to enhance predictive accuracy. Furthermore, these findings offer valuable insights for optimizing model design, such as exploring strategies to balance feature modeling capabilities between single-channel and multi-channel sequences. This series of experiments helps to clarify the applicability of different models across various datasets and variable scales, offering more targeted solutions for time series forecasting tasks.

To comprehensively evaluate the performance of the proposed model, it was also compared with the representative multiresolution method, N-HiTS, and the representative frequency-based methods, FEDformer and FiLM, on the ETTm2, Electricity, Traffic, and Weather datasets. The experimental results are shown in Table 9.

Table 9. Additional Comparative Experiments(MSE/MAE). 

<table><tr><td>Input Length</td><td>Timestacker</td><td>N-HiTS</td><td>FEDformer</td><td>FiLM</td></tr><tr><td>ETTm2</td><td>0.274/0.316</td><td>0.279/0.330</td><td>0.305/0.349</td><td>0.287/0.329</td></tr><tr><td>Electricity</td><td>0.194/0.275</td><td>0.186/0.287</td><td>0.214/0.327</td><td>0.223/0.302</td></tr><tr><td>Traffic</td><td>0.508/0.335</td><td>0.452/0.311</td><td>0.610/0.376</td><td>0.637/0.384</td></tr><tr><td>Weather</td><td>0.243/0.264</td><td>0.249/0.274</td><td>0.309/0.360</td><td>0.271/0.291</td></tr></table>

# E. SHOW CASE

A visualization of TimeStacker's prediction results was conducted across all datasets. As illustrated in Figure 6, in the 96-to-96 forecasting task, TimeStacker exhibited consistent performance across different datasets, clearly demonstrating its superior predictive capability.

To further highlight the capability of TimeStacker in capturing non-stationary signals, synthetic signals exhibiting nonlinear frequency variations over time were generated and evaluated alongside an MLP as a baseline. As illustrated in Figure 7, it can be observed that TimeStacker successfully captures high-frequency components, though slight phase misalignment is observed, whereas MLP-based models struggle to adapt to rapidly changing frequencies.

Table 10. Additional Experiments. All results are based on input sequences of length 96 and are calculated as the average across four different prediction lengths {96, 192, 336, 720}. 

<table><tr><td colspan="2">Models</td><td colspan="2">TimeStacker</td><td colspan="2">SOFTS</td><td colspan="2">SparseTSF</td><td colspan="2">iTransformer</td><td colspan="2">TimeMixer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.167</td><td>0.235</td><td>0.165</td><td>0.227</td><td>0.281</td><td>0.326</td><td>0.170</td><td>0.232</td><td>0.166</td><td>0.223</td></tr><tr><td>192</td><td>0.157</td><td>0.223</td><td>0.158</td><td>0.223</td><td>0.236</td><td>0.282</td><td>0.158</td><td>0.22</td><td>0.157</td><td>0.230</td></tr><tr><td>336</td><td>0.151</td><td>0.223</td><td>0.158</td><td>0.226</td><td>0.223</td><td>0.272</td><td>0.155</td><td>0.216</td><td>0.154</td><td>0.230</td></tr><tr><td>720</td><td>0.169</td><td>0.242</td><td>0.174</td><td>0.244</td><td>0.242</td><td>0.292</td><td>0.182</td><td>0.254</td><td>0.171</td><td>0.242</td></tr><tr><td colspan="2">AVG</td><td>0.161</td><td>0.231</td><td>0.164</td><td>0.230</td><td>0.246</td><td>0.293</td><td>0.166</td><td>0.231</td><td>0.162</td><td>0.231</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.296</td><td>0.384</td><td>0.292</td><td>0.381</td><td>0.477</td><td>0.504</td><td>0.299</td><td>0.4</td><td>0.297</td><td>0.391</td></tr><tr><td>192</td><td>0.304</td><td>0.392</td><td>0.301</td><td>0.389</td><td>0.457</td><td>0.493</td><td>0.299</td><td>0.391</td><td>0.314</td><td>0.401</td></tr><tr><td>336</td><td>0.364</td><td>0.42</td><td>0.365</td><td>0.418</td><td>0.489</td><td>0.512</td><td>0.362</td><td>0.426</td><td>0.367</td><td>0.431</td></tr><tr><td>720</td><td>0.425</td><td>0.473</td><td>0.427</td><td>0.473</td><td>0.501</td><td>0.517</td><td>0.426</td><td>0.477</td><td>0.424</td><td>0.46</td></tr><tr><td colspan="2">AVG</td><td>0.347</td><td>0.417</td><td>0.346</td><td>0.415</td><td>0.481</td><td>0.507</td><td>0.347</td><td>0.424</td><td>0.351</td><td>0.421</td></tr></table>

Table 11. Impact of Look-Back Lengths. The backtracking window lengths were set to $\{96, 192, 336, 720\}$ , and the prediction lengths were set to $\{96, 192, 336, 720\}$ . The evaluation metrics used were MSE and MAE. The best performance of the model is highlighted in red. 

<table><tr><td colspan="2">DataSet</td><td colspan="10">Exchange</td><td colspan="8">ETTh1</td><td colspan="7">ETTh2</td></tr><tr><td colspan="2">Look-Back Lengths</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td><td></td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td></tr><tr><td rowspan="4">TimeMixer</td><td>96</td><td>0.082</td><td>0.199</td><td>0.090</td><td>0.210</td><td>0.089</td><td>0.211</td><td>0.120</td><td>0.251</td><td>0.375</td><td>0.400</td><td>0.378</td><td>0.389</td><td>0.370</td><td>0.393</td><td>0.366</td><td>0.391</td><td>0.289</td><td>0.341</td><td>0.286</td><td>0.338</td><td>0.276</td><td>0.334</td><td>0.339</td><td>0.387</td><td></td></tr><tr><td>192</td><td>0.177</td><td>0.297</td><td>0.181</td><td>0.303</td><td>0.178</td><td>0.301</td><td>0.173</td><td>0.299</td><td>0.429</td><td>0.421</td><td>0.428</td><td>0.420</td><td>0.414</td><td>0.415</td><td>0.411</td><td>0.424</td><td>0.372</td><td>0.392</td><td>0.367</td><td>0.388</td><td>0.347</td><td>0.382</td><td>0.387</td><td>0.410</td><td></td></tr><tr><td>336</td><td>0.324</td><td>0.408</td><td>0.329</td><td>0.418</td><td>0.323</td><td>0.416</td><td>0.335</td><td>0.420</td><td>0.484</td><td>0.458</td><td>0.463</td><td>0.464</td><td>0.432</td><td>0.427</td><td>0.434</td><td>0.440</td><td>0.386</td><td>0.414</td><td>0.394</td><td>0.413</td><td>0.365</td><td>0.401</td><td>0.374</td><td>0.408</td><td></td></tr><tr><td>720</td><td>0.837</td><td>0.691</td><td>0.817</td><td>0.679</td><td>0.887</td><td>0.699</td><td>0.904</td><td>0.741</td><td>0.498</td><td>0.482</td><td>0.465</td><td>0.462</td><td>0.447</td><td>0.455</td><td>0.450</td><td>0.472</td><td>0.412</td><td>0.434</td><td>0.413</td><td>0.436</td><td>0.405</td><td>0.434</td><td>0.410</td><td>0.443</td><td></td></tr><tr><td colspan="2">AVG</td><td>0.355</td><td>0.399</td><td>0.354</td><td>0.403</td><td>0.369</td><td>0.407</td><td>0.383</td><td>0.428</td><td>0.447</td><td>0.440</td><td>0.434</td><td>0.434</td><td>0.416</td><td>0.423</td><td>0.415</td><td>0.432</td><td>0.365</td><td>0.395</td><td>0.365</td><td>0.394</td><td>0.348</td><td>0.388</td><td>0.378</td><td>0.412</td><td></td></tr><tr><td rowspan="4">SOFTS</td><td>96</td><td>0.084</td><td>0.201</td><td>0.088</td><td>0.209</td><td>0.088</td><td>0.211</td><td>0.102</td><td>0.233</td><td>0.381</td><td>0.399</td><td>0.385</td><td>0.405</td><td>0.390</td><td>0.406</td><td>0.393</td><td>0.417</td><td>0.297</td><td>0.347</td><td>0.302</td><td>0.349</td><td>0.293</td><td>0.354</td><td>0.312</td><td>0.313</td><td></td></tr><tr><td>192</td><td>0.172</td><td>0.294</td><td>0.181</td><td>0.303</td><td>0.190</td><td>0.315</td><td>0.183</td><td>0.309</td><td>0.435</td><td>0.431</td><td>0.431</td><td>0.432</td><td>0.428</td><td>0.432</td><td>0.429</td><td>0.438</td><td>0.373</td><td>0.394</td><td>0.387</td><td>0.407</td><td>0.383</td><td>0.407</td><td>0.386</td><td>0.415</td><td></td></tr><tr><td>336</td><td>0.324</td><td>0.412</td><td>0.314</td><td>0.413</td><td>0.339</td><td>0.430</td><td>0.377</td><td>0.454</td><td>0.480</td><td>0.452</td><td>0.451</td><td>0.442</td><td>0.439</td><td>0.443</td><td>0.457</td><td>0.458</td><td>0.410</td><td>0.426</td><td>0.399</td><td>0.419</td><td>0.385</td><td>0.416</td><td>0.404</td><td>0.424</td><td></td></tr><tr><td>720</td><td>0.811</td><td>0.672</td><td>0.732</td><td>0.645</td><td>0.815</td><td>0.690</td><td>0.845</td><td>0.701</td><td>0.499</td><td>0.488</td><td>0.444</td><td>0.460</td><td>0.450</td><td>0.468</td><td>0.463</td><td>0.477</td><td>0.411</td><td>0.433</td><td>0.427</td><td>0.444</td><td>0.435</td><td>0.454</td><td>0.429</td><td>0.455</td><td></td></tr><tr><td colspan="2">AVG</td><td>0.348</td><td>0.395</td><td>0.329</td><td>0.393</td><td>0.358</td><td>0.413</td><td>0.377</td><td>0.424</td><td>0.449</td><td>0.443</td><td>0.428</td><td>0.435</td><td>0.427</td><td>0.437</td><td>0.436</td><td>0.448</td><td>0.373</td><td>0.400</td><td>0.379</td><td>0.405</td><td>0.374</td><td>0.408</td><td>0.383</td><td>0.402</td><td></td></tr><tr><td rowspan="4">iTransformer</td><td>96</td><td>0.086</td><td>0.206</td><td>0.090</td><td>0.211</td><td>0.096</td><td>0.220</td><td>0.109</td><td>0.237</td><td>0.386</td><td>0.405</td><td>0.387</td><td>0.402</td><td>0.394</td><td>0.407</td><td>0.390</td><td>0.413</td><td>0.297</td><td>0.349</td><td>0.298</td><td>0.347</td><td>0.296</td><td>0.352</td><td>0.309</td><td>0.370</td><td></td></tr><tr><td>192</td><td>0.177</td><td>0.299</td><td>0.193</td><td>0.316</td><td>0.192</td><td>0.319</td><td>0.193</td><td>0.319</td><td>0.441</td><td>0.436</td><td>0.446</td><td>0.429</td><td>0.444</td><td>0.433</td><td>0.418</td><td>0.431</td><td>0.380</td><td>0.400</td><td>0.383</td><td>0.399</td><td>0.389</td><td>0.405</td><td>0.390</td><td>0.415</td><td></td></tr><tr><td>336</td><td>0.331</td><td>0.417</td><td>0.345</td><td>0.433</td><td>0.373</td><td>0.453</td><td>0.377</td><td>0.456</td><td>0.487</td><td>0.458</td><td>0.459</td><td>0.439</td><td>0.446</td><td>0.445</td><td>0.441</td><td>0.449</td><td>0.428</td><td>0.432</td><td>0.396</td><td>0.418</td><td>0.396</td><td>0.416</td><td>0.403</td><td>0.431</td><td></td></tr><tr><td>720</td><td>0.847</td><td>0.691</td><td>0.757</td><td>0.657</td><td>0.694</td><td>0.645</td><td>0.796</td><td>0.683</td><td>0.503</td><td>0.491</td><td>0.464</td><td>0.468</td><td>0.467</td><td>0.463</td><td>0.489</td><td>0.490</td><td>0.427</td><td>0.445</td><td>0.422</td><td>0.441</td><td>0.426</td><td>0.447</td><td>0.450</td><td>0.465</td><td></td></tr><tr><td colspan="2">AVG</td><td>0.360</td><td>0.403</td><td>0.346</td><td>0.404</td><td>0.339</td><td>0.409</td><td>0.369</td><td>0.424</td><td>0.454</td><td>0.448</td><td>0.439</td><td>0.435</td><td>0.438</td><td>0.437</td><td>0.435</td><td>0.446</td><td>0.383</td><td>0.407</td><td>0.375</td><td>0.401</td><td>0.377</td><td>0.405</td><td>0.388</td><td>0.420</td><td></td></tr><tr><td rowspan="4">SAMformer</td><td>96</td><td>0.088</td><td>0.209</td><td>0.093</td><td>0.212</td><td>0.098</td><td>0.221</td><td>0.122</td><td>0.254</td><td>0.383</td><td>0.392</td><td>0.388</td><td>0.402</td><td>0.385</td><td>0.403</td><td>0.389</td><td>0.412</td><td>0.289</td><td>0.338</td><td>0.347</td><td>0.293</td><td>0.283</td><td>0.345</td><td>0.279</td><td>0.346</td><td></td></tr><tr><td>192</td><td>0.176</td><td>0.299</td><td>0.186</td><td>0.307</td><td>0.196</td><td>0.318</td><td>0.229</td><td>0.339</td><td>0.438</td><td>0.423</td><td>0.438</td><td>0.429</td><td>0.419</td><td>0.424</td><td>0.425</td><td>0.436</td><td>0.401</td><td>0.398</td><td>0.382</td><td>0.401</td><td>0.36</td><td>0.397</td><td>0.36</td><td>0.396</td><td></td></tr><tr><td>336</td><td>0.322</td><td>0.414</td><td>0.333</td><td>0.42</td><td>0.347</td><td>0.431</td><td>0.385</td><td>0.452</td><td>0.475</td><td>0.445</td><td>0.461</td><td>0.442</td><td>0.451</td><td>0.446</td><td>0.456</td><td>0.455</td><td>0.419</td><td>0.428</td><td>0.395</td><td>0.421</td><td>0.364</td><td>0.406</td><td>0.365</td><td>0.409</td><td></td></tr><tr><td>720</td><td>0.799</td><td>0.673</td><td>0.804</td><td>0.678</td><td>0.856</td><td>0.701</td><td>0.957</td><td>0.741</td><td>0.478</td><td>0.468</td><td>0.464</td><td>0.469</td><td>0.454</td><td>0.468</td><td>0.467</td><td>0.482</td><td>0.421</td><td>0.44</td><td>0.415</td><td>0.441</td><td>0.405</td><td>0.438</td><td>0.4</td><td>0.439</td><td></td></tr><tr><td colspan="2">AVG</td><td>0.346</td><td>0.399</td><td>0.354</td><td>0.404</td><td>0.374</td><td>0.418</td><td>0.423</td><td>0.447</td><td>0.444</td><td>0.432</td><td>0.438</td><td>0.436</td><td>0.427</td><td>0.435</td><td>0.434</td><td>0.446</td><td>0.383</td><td>0.401</td><td>0.385</td><td>0.389</td><td>0.353</td><td>0.397</td><td>0.351</td><td>0.398</td><td></td></tr><tr><td rowspan="4">TimeStacker</td><td>96</td><td>0.084</td><td>0.200</td><td>0.084</td><td>0.201</td><td>0.085</td><td>0.204</td><td>0.085</td><td>0.205</td><td>0.379</td><td>0.385</td><td>0.37</td><td>0.384</td><td>0.362</td><td>0.383</td><td>0.364</td><td>0.383</td><td>0.28</td><td>0.327</td><td>0.283</td><td>0.332</td><td>0.27</td><td>0.333</td><td>0.267</td><td>0.331</td><td></td></tr><tr><td>192</td><td>0.171</td><td>0.293</td><td>0.173</td><td>0.293</td><td>0.174</td><td>0.294</td><td>0.171</td><td>0.293</td><td>0.429</td><td>0.416</td><td>0.424</td><td>0.414</td><td>0.401</td><td>0.409</td><td>0.399</td><td>0.418</td><td>0.373</td><td>0.385</td><td>0.366</td><td>0.386</td><td>0.348</td><td>0.383</td><td>0.336</td><td>0.374</td><td></td></tr><tr><td>336</td><td>0.314</td><td>0.408</td><td>0.317</td><td>0.411</td><td>0.32</td><td>0.413</td><td>0.321</td><td>0.408</td><td>0.459</td><td>0.436</td><td>0.448</td><td>0.429</td><td>0.428</td><td>0.427</td><td>0.42</td><td>0.428</td><td>0.407</td><td>0.416</td><td>0.397</td><td>0.413</td><td>0.362</td><td>0.397</td><td>0.354</td><td>0.396</td><td></td></tr><tr><td>720</td><td>0.776</td><td>0.656</td><td>0.756</td><td>0.644</td><td>0.792</td><td>0.658</td><td>0.772</td><td>0.656</td><td>0.464</td><td>0.455</td><td>0.447</td><td>0.453</td><td>0.432</td><td>0.45</td><td>0.427</td><td>0.447</td><td>0.412</td><td>0.431</td><td>0.409</td><td>0.428</td><td>0.399</td><td>0.428</td><td>0.389</td><td>0.426</td><td></td></tr><tr><td colspan="2">AVG</td><td>0.336</td><td>0.389</td><td>0.333</td><td>0.387</td><td>0.343</td><td>0.392</td><td>0.337</td><td>0.391</td><td>0.433</td><td>0.423</td><td>0.422</td><td>0.420</td><td>0.406</td><td>0.417</td><td>0.403</td><td>0.419</td><td>0.368</td><td>0.390</td><td>0.364</td><td>0.390</td><td>0.345</td><td>0.385</td><td>0.337</td><td>0.382</td><td></td></tr></table>

Table 12. Complete Experimental Result. All results are based on input sequences of length 96 and are calculated as the average across four different prediction lengths $\{96, 192, 336, 720\}$ . 

<table><tr><td colspan="2">Models</td><td colspan="2">TimeStacker(ours)</td><td colspan="2">SOFTS(2024)</td><td colspan="2">SparseTSF(2024)</td><td colspan="2">iTransformer(2024)</td><td colspan="2">TimeMixer(2024)</td><td colspan="2">SAMformer(2024)</td><td colspan="2">PatchTST(2023)</td><td colspan="2">Crossformer(2023)</td><td colspan="2">DLinear(2023)</td><td colspan="2">RLinear(2023)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.379</td><td>0.385</td><td>0.381</td><td>0.399</td><td>0.388</td><td>0.387</td><td>0.386</td><td>0.405</td><td>0.375</td><td>0.400</td><td>0.383</td><td>0.392</td><td>0.414</td><td>0.419</td><td>0.423</td><td>0.448</td><td>0.386</td><td>0.400</td><td>0.386</td><td>0.395</td></tr><tr><td>192</td><td>0.429</td><td>0.416</td><td>0.435</td><td>0.431</td><td>0.438</td><td>0.417</td><td>0.441</td><td>0.436</td><td>0.429</td><td>0.421</td><td>0.438</td><td>0.423</td><td>0.460</td><td>0.445</td><td>0.471</td><td>0.474</td><td>0.437</td><td>0.432</td><td>0.437</td><td>0.424</td></tr><tr><td>336</td><td>0.459</td><td>0.436</td><td>0.480</td><td>0.452</td><td>0.469</td><td>0.438</td><td>0.487</td><td>0.458</td><td>0.484</td><td>0.458</td><td>0.475</td><td>0.445</td><td>0.501</td><td>0.466</td><td>0.570</td><td>0.546</td><td>0.481</td><td>0.459</td><td>0.479</td><td>0.446</td></tr><tr><td>720</td><td>0.464</td><td>0.455</td><td>0.499</td><td>0.488</td><td>0.468</td><td>0.457</td><td>0.503</td><td>0.491</td><td>0.498</td><td>0.482</td><td>0.478</td><td>0.468</td><td>0.500</td><td>0.488</td><td>0.653</td><td>0.621</td><td>0.519</td><td>0.516</td><td>0.481</td><td>0.470</td></tr><tr><td colspan="2">AVG</td><td>0.433</td><td>0.423</td><td>0.449</td><td>0.442</td><td>0.441</td><td>0.425</td><td>0.454</td><td>0.447</td><td>0.447</td><td>0.440</td><td>0.444</td><td>0.432</td><td>0.469</td><td>0.454</td><td>0.529</td><td>0.522</td><td>0.456</td><td>0.452</td><td>0.446</td><td>0.434</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.280</td><td>0.327</td><td>0.297</td><td>0.347</td><td>0.304</td><td>0.346</td><td>0.297</td><td>0.349</td><td>0.289</td><td>0.341</td><td>0.289</td><td>0.338</td><td>0.302</td><td>0.348</td><td>0.745</td><td>0.584</td><td>0.333</td><td>0.387</td><td>0.288</td><td>0.338</td></tr><tr><td>192</td><td>0.373</td><td>0.385</td><td>0.373</td><td>0.394</td><td>0.409</td><td>0.403</td><td>0.380</td><td>0.400</td><td>0.372</td><td>0.392</td><td>0.401</td><td>0.398</td><td>0.388</td><td>0.400</td><td>0.877</td><td>0.656</td><td>0.477</td><td>0.476</td><td>0.374</td><td>0.390</td></tr><tr><td>336</td><td>0.407</td><td>0.416</td><td>0.410</td><td>0.426</td><td>0.426</td><td>0.430</td><td>0.428</td><td>0.432</td><td>0.386</td><td>0.414</td><td>0.419</td><td>0.428</td><td>0.426</td><td>0.433</td><td>1.043</td><td>0.731</td><td>0.594</td><td>0.541</td><td>0.415</td><td>0.426</td></tr><tr><td>720</td><td>0.412</td><td>0.431</td><td>0.411</td><td>0.433</td><td>0.421</td><td>0.438</td><td>0.427</td><td>0.445</td><td>0.412</td><td>0.434</td><td>0.421</td><td>0.440</td><td>0.431</td><td>0.446</td><td>1.104</td><td>0.763</td><td>0.831</td><td>0.657</td><td>0.420</td><td>0.440</td></tr><tr><td colspan="2">AVG</td><td>0.368</td><td>0.390</td><td>0.373</td><td>0.400</td><td>0.421</td><td>0.438</td><td>0.383</td><td>0.407</td><td>0.364</td><td>0.395</td><td>0.383</td><td>0.401</td><td>0.387</td><td>0.407</td><td>0.942</td><td>0.684</td><td>0.559</td><td>0.515</td><td>0.374</td><td>0.398</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.311</td><td>0.337</td><td>0.325</td><td>0.361</td><td>0.366</td><td>0.369</td><td>0.334</td><td>0.368</td><td>0.320</td><td>0.357</td><td>0.352</td><td>0.374</td><td>0.329</td><td>0.367</td><td>0.404</td><td>0.426</td><td>0.345</td><td>0.372</td><td>0.355</td><td>0.376</td></tr><tr><td>192</td><td>0.364</td><td>0.367</td><td>0.375</td><td>0.389</td><td>0.404</td><td>0.387</td><td>0.377</td><td>0.391</td><td>0.361</td><td>0.381</td><td>0.392</td><td>0.392</td><td>0.367</td><td>0.385</td><td>0.450</td><td>0.451</td><td>0.380</td><td>0.389</td><td>0.391</td><td>0.392</td></tr><tr><td>336</td><td>0.389</td><td>0.391</td><td>0.405</td><td>0.412</td><td>0.432</td><td>0.406</td><td>0.426</td><td>0.420</td><td>0.390</td><td>0.404</td><td>0.425</td><td>0.413</td><td>0.399</td><td>0.410</td><td>0.532</td><td>0.515</td><td>0.413</td><td>0.413</td><td>0.424</td><td>0.415</td></tr><tr><td>720</td><td>0.460</td><td>0.428</td><td>0.466</td><td>0.447</td><td>0.496</td><td>0.442</td><td>0.491</td><td>0.459</td><td>0.454</td><td>0.441</td><td>0.49</td><td>0.449</td><td>0.454</td><td>0.439</td><td>0.666</td><td>0.589</td><td>0.474</td><td>0.453</td><td>0.487</td><td>0.450</td></tr><tr><td colspan="2">AVG</td><td>0.381</td><td>0.381</td><td>0.393</td><td>0.403</td><td>0.425</td><td>0.401</td><td>0.407</td><td>0.410</td><td>0.381</td><td>0.395</td><td>0.415</td><td>0.407</td><td>0.387</td><td>0.400</td><td>0.513</td><td>0.496</td><td>0.403</td><td>0.407</td><td>0.414</td><td>0.407</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.171</td><td>0.250</td><td>0.180</td><td>0.261</td><td>0.198</td><td>0.272</td><td>0.180</td><td>0.264</td><td>0.175</td><td>0.258</td><td>0.181</td><td>0.264</td><td>0.175</td><td>0.259</td><td>0.287</td><td>0.366</td><td>0.193</td><td>0.292</td><td>0.182</td><td>0.265</td></tr><tr><td>192</td><td>0.235</td><td>0.292</td><td>0.246</td><td>0.306</td><td>0.259</td><td>0.308</td><td>0.250</td><td>0.309</td><td>0.237</td><td>0.299</td><td>0.245</td><td>0.305</td><td>0.241</td><td>0.302</td><td>0.414</td><td>0.492</td><td>0.284</td><td>0.362</td><td>0.246</td><td>0.304</td></tr><tr><td>336</td><td>0.293</td><td>0.329</td><td>0.319</td><td>0.352</td><td>0.315</td><td>0.343</td><td>0.311</td><td>0.348</td><td>0.298</td><td>0.340</td><td>0.305</td><td>0.341</td><td>0.305</td><td>0.343</td><td>0.597</td><td>0.542</td><td>0.369</td><td>0.427</td><td>0.307</td><td>0.342</td></tr><tr><td>720</td><td>0.395</td><td>0.391</td><td>0.405</td><td>0.401</td><td>0.416</td><td>0.399</td><td>0.412</td><td>0.407</td><td>0.391</td><td>0.396</td><td>0.409</td><td>0.398</td><td>0.402</td><td>0400</td><td>1.730</td><td>1.042</td><td>0.554</td><td>0.522</td><td>0.407</td><td>0.398</td></tr><tr><td colspan="2">AVG</td><td>0.274</td><td>0.316</td><td>0.287</td><td>0.330</td><td>0.297</td><td>0.331</td><td>0.288</td><td>0.332</td><td>0.275</td><td>0.323</td><td>0.285</td><td>0.327</td><td>0.281</td><td>0.326</td><td>0.757</td><td>0.610</td><td>0.350</td><td>0.401</td><td>0.286</td><td>0.327</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.496</td><td>0.331</td><td>0.376</td><td>0.251</td><td>0.559</td><td>0.335</td><td>0.395</td><td>0.268</td><td>0.462</td><td>0.285</td><td>0.552</td><td>0.367</td><td>0.462</td><td>0.295</td><td>0.522</td><td>0.290</td><td>0.650</td><td>0.396</td><td>0.649</td><td>0.389</td></tr><tr><td>192</td><td>0.491</td><td>0.331</td><td>0.398</td><td>0.261</td><td>0.567</td><td>0.346</td><td>0.417</td><td>0.276</td><td>0.473</td><td>0.296</td><td>0.569</td><td>0.368</td><td>0.466</td><td>0.296</td><td>0.530</td><td>0.293</td><td>0.598</td><td>0.370</td><td>0.601</td><td>0.366</td></tr><tr><td>336</td><td>0.505</td><td>0.334</td><td>0.415</td><td>0.269</td><td>0.575</td><td>0.349</td><td>0.433</td><td>0.283</td><td>0.498</td><td>0.296</td><td>0.586</td><td>0.376</td><td>0.482</td><td>0.304</td><td>0.558</td><td>0.305</td><td>0.605</td><td>0.373</td><td>0.609</td><td>0.369</td></tr><tr><td>720</td><td>0.541</td><td>0.343</td><td>0.447</td><td>0.287</td><td>0.609</td><td>0.368</td><td>0.467</td><td>0.302</td><td>0.506</td><td>0.313</td><td>0.63</td><td>0.401</td><td>0.514</td><td>0.322</td><td>0.589</td><td>0.328</td><td>0.645</td><td>0.394</td><td>0.647</td><td>0.387</td></tr><tr><td colspan="2">AVG</td><td>0.508</td><td>0.335</td><td>0.409</td><td>0.267</td><td>0.578</td><td>0.350</td><td>0.428</td><td>0.282</td><td>0.484</td><td>0.297</td><td>0.595</td><td>0.382</td><td>0.481</td><td>0.304</td><td>0.550</td><td>0.304</td><td>0.625</td><td>0.383</td><td>0.626</td><td>0.378</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.168</td><td>0.251</td><td>0.143</td><td>0.233</td><td>0.202</td><td>0.261</td><td>0.148</td><td>0.240</td><td>0.153</td><td>0.247</td><td>0.199</td><td>0.277</td><td>0.181</td><td>0.270</td><td>0.219</td><td>0.314</td><td>0.197</td><td>0.282</td><td>0.201</td><td>0.281</td></tr><tr><td>192</td><td>0.176</td><td>0.262</td><td>0.158</td><td>0.248</td><td>0.207</td><td>0.277</td><td>0.162</td><td>0.253</td><td>0.166</td><td>0.256</td><td>0.199</td><td>0.279</td><td>0.188</td><td>0.284</td><td>0.231</td><td>0.322</td><td>0.196</td><td>0.285</td><td>0.201</td><td>0.283</td></tr><tr><td>336</td><td>0.195</td><td>0.278</td><td>0.178</td><td>0.269</td><td>0.219</td><td>0.292</td><td>0.178</td><td>0.269</td><td>0.185</td><td>0.277</td><td>0.214</td><td>0.294</td><td>0.204</td><td>0.293</td><td>0.246</td><td>0.337</td><td>0.209</td><td>0.301</td><td>0.215</td><td>0.298</td></tr><tr><td>720</td><td>0.235</td><td>0.310</td><td>0.218</td><td>0.305</td><td>0.261</td><td>0.324</td><td>0.225</td><td>0.317</td><td>0.225</td><td>0.310</td><td>0.257</td><td>0.328</td><td>0.246</td><td>0.324</td><td>0.280</td><td>0.363</td><td>0.245</td><td>0.333</td><td>0.257</td><td>0.331</td></tr><tr><td colspan="2">AVG</td><td>0.194</td><td>0.275</td><td>0.174</td><td>0.264</td><td>0.222</td><td>0.289</td><td>0.178</td><td>0.270</td><td>0.182</td><td>0.272</td><td>0.217</td><td>0.295</td><td>0.205</td><td>0.290</td><td>0.244</td><td>0.334</td><td>0.212</td><td>0.300</td><td>0.219</td><td>0.298</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.161</td><td>0.198</td><td>0.166</td><td>0.208</td><td>0.213</td><td>0.250</td><td>0.174</td><td>0.214</td><td>0.163</td><td>0.209</td><td>0.193</td><td>0.205</td><td>0.177</td><td>0.218</td><td>0.158</td><td>0.230</td><td>0.196</td><td>0.255</td><td>0.192</td><td>0.232</td></tr><tr><td>192</td><td>0.207</td><td>0.241</td><td>0.217</td><td>0.253</td><td>0.259</td><td>0.282</td><td>0.221</td><td>0.254</td><td>0.208</td><td>0.250</td><td>0.242</td><td>0.274</td><td>0.225</td><td>0.259</td><td>0.206</td><td>0.277</td><td>0.237</td><td>0.296</td><td>0.240</td><td>0.271</td></tr><tr><td>336</td><td>0.261</td><td>0.281</td><td>0.282</td><td>0.300</td><td>0.308</td><td>0.315</td><td>0.278</td><td>0.296</td><td>0.251</td><td>0.287</td><td>0.284</td><td>0.309</td><td>0.278</td><td>0.297</td><td>0.272</td><td>0.335</td><td>0.283</td><td>0.335</td><td>0.292</td><td>0.307</td></tr><tr><td>720</td><td>0.343</td><td>0.334</td><td>0.356</td><td>0.351</td><td>0.380</td><td>0.361</td><td>0.358</td><td>0.347</td><td>0.339</td><td>0.341</td><td>0.358</td><td>0.351</td><td>0.354</td><td>0.348</td><td>0.398</td><td>0.418</td><td>0.345</td><td>0.381</td><td>0.364</td><td>0.353</td></tr><tr><td colspan="2">AVG</td><td>0.243</td><td>0.264</td><td>0.255</td><td>0.278</td><td>0.290</td><td>0.302</td><td>0.258</td><td>0.278</td><td>0.240</td><td>0.271</td><td>0.264</td><td>0.285</td><td>0.259</td><td>0.348</td><td>0.259</td><td>0.315</td><td>0.265</td><td>0.317</td><td>0.272</td><td>0.291</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.084</td><td>0.200</td><td>0.084</td><td>0.201</td><td>0.093</td><td>0.217</td><td>0.086</td><td>0.206</td><td>0.082</td><td>0.199</td><td>0.088</td><td>0.209</td><td>0.088</td><td>0.205</td><td>0.256</td><td>0.367</td><td>0.088</td><td>0.218</td><td>0.093</td><td>0.217</td></tr><tr><td>192</td><td>0.171</td><td>0.293</td><td>0.172</td><td>0.294</td><td>0.179</td><td>0.304</td><td>0.177</td><td>0.299</td><td>0.177</td><td>0.297</td><td>0.176</td><td>0.299</td><td>0.176</td><td>0.299</td><td>0.470</td><td>0.509</td><td>0.176</td><td>0.315</td><td>0.184</td><td>0.307</td></tr><tr><td>336</td><td>0.314</td><td>0.406</td><td>0.324</td><td>0.412</td><td>0.319</td><td>0.410</td><td>0.331</td><td>0.417</td><td>0.324</td><td>0.408</td><td>0.322</td><td>0.414</td><td>0.301</td><td>0.397</td><td>1.268</td><td>0.883</td><td>0.313</td><td>0.427</td><td>0.351</td><td>0.432</td></tr><tr><td>720</td><td>0.776</td><td>0.656</td><td>0.811</td><td>0.672</td><td>0.823</td><td>0.683</td><td>0.847</td><td>0.691</td><td>0.837</td><td>0.691</td><td>0.799</td><td>0.673</td><td>0.901</td><td>0.714</td><td>1.767</td><td>1.068</td><td>0.839</td><td>0.695</td><td>0.886</td><td>0.714</td></tr><tr><td colspan="2">AVG</td><td>0.336</td><td>0.389</td><td>0.348</td><td>0.395</td><td>0.365</td><td>0.401</td><td>0.360</td><td>0.403</td><td>0.355</td><td>0.399</td><td>0.346</td><td>0.399</td><td>0.367</td><td>0.404</td><td>0.940</td><td>0.707</td><td>0.354</td><td>0.414</td><td>0.378</td><td>0.417</td></tr><tr><td colspan="2">1st Count</td><td>15</td><td>26</td><td>11</td><td>10</td><td>0</td><td>0</td><td>0</td><td>0</td><td>13</td><td>4</td><td>0</td><td>0</td><td>2</td><td>1</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td></tr></table>

![](images/03aa7f45ce322b6e7513b65dbbcd2040b23022d637fd0bff8ae12eb59378973d.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | 1.5         | 1.5      |
| 25   | 0.2         | 0.3      |
| 50   | 1.6         | 1.4      |
| 75   | 1.7         | 1.3      |
| 100  | 1.8         | 1.2      |
| 125  | 1.6         | 1.1      |
| 150  | 2.2         | 1.6      |
| 175  | 1.9         | 1.5      |
| 200  | 1.7         | 1.3      |
</details>

(a) ETTh1

![](images/03ceb91ea201748290617d1735a612ce49d66bd54d0cc41416845b280ce9a405.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | 0.50        | 0.50     |
| 25   | -0.75       | -0.75    |
| 50   | 0.50        | 0.50     |
| 75   | -0.75       | -0.75    |
| 100  | 0.25        | 0.25     |
| 125  | -0.75       | -0.75    |
| 150  | 0.25        | 0.25     |
| 175  | -0.75       | -0.75    |
| 200  | 0.00        | 0.00     |
</details>

(b) ETTh2

![](images/69b246fb58f9f20a201217e8883b95245f2d1da4a6088681d57a300bf92448a5.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | -0.9        | -0.9     |
| 25   | -0.7        | -0.7     |
| 50   | -0.4        | -0.4     |
| 75   | -0.8        | -0.8     |
| 100  | -0.6        | -0.6     |
| 125  | -0.5        | -0.5     |
| 150  | -0.3        | -0.3     |
| 175  | -0.7        | -0.7     |
| 200  | -0.8        | -0.8     |
</details>

(c) ETTm1

![](images/a6356dacda5d5f3f06c33cbeb5ba61ac5bd05024e209e3a77923a7fa32870910.jpg)

<details>
<summary>line</summary>

| x    | GroundTruth | Prediction |
| ---- | ----------- | ---------- |
| 0    | -0.7        | -          |
| 25   | -0.8        | -          |
| 50   | 0.7         | -          |
| 75   | -0.4        | -          |
| 100  | -0.7        | -          |
| 125  | -0.7        | -0.8       |
| 150  | 0.6         | 0.6        |
| 175  | -0.4        | -0.4       |
| 200  | -0.3        | -0.5       |
</details>

(d) ETTm2

![](images/c2e2df3ac791d6b56c8647f2324e46876ccc2aab8680f90cb496769d4c00fb30.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
| ---- | ----------- | ---------- |
| 0    | 1.9         | 1.9        |
| 25   | 2.1         | 2.1        |
| 50   | 2.3         | 2.3        |
| 75   | 2.4         | 2.3        |
| 100  | 2.3         | 2.3        |
| 125  | 2.4         | 2.3        |
| 150  | 2.5         | 2.3        |
| 175  | 2.4         | 2.3        |
| 200  | 2.4         | 2.3        |
</details>

(e) Exchange

![](images/4c7b7a50fa0fcbe978e2e4b8d32b8ba863bee64c0b3c7626a815e0f5a38f6097.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
|------|-------------|----------|
| 0    | -0.5        | -0.5     |
| 25   | 1.5         | 1.5      |
| 50   | -0.5        | -0.5     |
| 75   | 1.0         | 1.0      |
| 100  | -0.5        | -0.5     |
| 125  | 2.0         | 2.0      |
| 150  | -0.5        | -0.5     |
| 175  | 2.5         | 2.5      |
| 200  | -0.5        | -0.5     |
</details>

(f) Electricity

![](images/2315bcb8a03a054a08e486bfa4abd5e17f79d97eb77706b15f63792972e058ec.jpg)

<details>
<summary>line</summary>

| x    | GroundTruth | Prediction |
| ---- | ----------- | ---------- |
| 0    | -1.0        | -1.0       |
| 25   | 2.0         | 2.0        |
| 50   | -1.0        | -1.0       |
| 75   | 2.0         | 2.0        |
| 100  | -1.0        | -1.0       |
| 125  | 2.0         | 2.0        |
| 150  | -1.0        | -1.0       |
| 175  | 2.0         | 2.0        |
| 200  | -1.0        | -1.0       |
</details>

(g) Traffic

![](images/bcee74e6a58e49efd5ba5113f664a7c4a7f9f30ecb40a09cdf1551928c40a4a8.jpg)

<details>
<summary>line</summary>

| Step | GroundTruth | Prediction |
| ---- | ----------- | ---------- |
| 0    | 0.03        | 0.03       |
| 25   | 0.08        | 0.08       |
| 50   | 0.06        | 0.06       |
| 75   | 0.10        | 0.10       |
| 100  | 0.01        | 0.01       |
| 125  | 0.04        | 0.06       |
| 150  | 0.06        | 0.07       |
| 175  | 0.07        | 0.08       |
| 200  | 0.07        | 0.07       |
</details>

(h) Weather   
Figure 6. Visualization of forecasting results on the real-world dataset with look-back window length L = 96 and prediction length H = 96

![](images/185cfd3cfcfe9cf747e32a4117d3252c805a54936ba5102156802234d5be94a0.jpg)

<details>
<summary>line</summary>

| Time Step | Input Sequence | True Future | Predicted Future |
| --------- | -------------- | ----------- | ---------------- |
| 0         | 0.0            | 0.0         | 0.0              |
| 25        | 0.0            | 0.0         | 0.0              |
| 50        | 0.0            | 0.0         | 0.0              |
| 75        | 0.0            | 0.0         | 0.0              |
| 100       | 0.0            | 0.0         | 0.0              |
| 125       | 0.0            | 0.0         | 0.0              |
| 150       | 0.0            | 0.0         | 0.0              |
| 175       | 0.0            | 0.0         | 0.0              |
| 200       | 0.0            | 0.0         | 0.0              |
</details>

![](images/31c8716a6a70970db655ccba3f70e9be4d577f1edd9db8b2e3c380f3a28c1729.jpg)

<details>
<summary>line</summary>

| Time Step | Input Sequence | True Future | Predicted Future |
| --------- | -------------- | ----------- | ---------------- |
| 0         | 1.0            | 1.0         | 1.0              |
| 25        | 1.0            | 1.0         | 1.0              |
| 50        | 1.0            | 1.0         | 1.0              |
| 75        | 1.0            | 1.0         | 1.0              |
| 100       | -1.0           | -1.0        | -2.0             |
| 125       | 1.0            | 1.0         | 1.5              |
| 150       | 1.0            | 1.0         | 1.0              |
| 175       | 1.0            | 1.0         | 0.5              |
| 200       | 1.0            | 1.0         | -0.5             |
</details>

(a) TimeStcker

![](images/6119456c1e13d4cfd9ea4196ce4dcc7ade98c0bafc70a441953ebec485ffbe7d.jpg)

<details>
<summary>line</summary>

| Time Step | Input Sequence | True Future | Predicted Future |
| --------- | -------------- | ----------- | ---------------- |
| 0         | 1.0            | 1.0         | 1.0              |
| 25        | 1.0            | 1.0         | 1.0              |
| 50        | 1.0            | 1.0         | 1.0              |
| 75        | 1.0            | 1.0         | 1.0              |
| 100       | 1.0            | 1.0         | 1.0              |
| 125       | 1.0            | 1.0         | 1.0              |
| 150       | 1.0            | 1.0         | 1.0              |
| 175       | 1.0            | 1.0         | 1.0              |
| 200       | 1.0            | 1.0         | 1.0              |
</details>

![](images/39fa5b0472bfaa1a3d8552e33441b39ea7dc7ff47b7ffb28fed95327ec224a22.jpg)

<details>
<summary>line</summary>

| Time Step | Input Sequence | True Future | Predicted Future |
| --------- | -------------- | ----------- | ---------------- |
| 0         | 1.0            | 1.0         | 1.0              |
| 25        | -1.0           | 1.0         | 1.0              |
| 50        | 1.0            | 1.0         | 1.0              |
| 75        | -1.0           | 1.0         | 1.0              |
| 100       | 1.0            | 1.0         | 2.0              |
| 125       | -1.0           | 1.0         | 1.0              |
| 150       | 1.0            | 1.0         | 1.5              |
| 175       | -1.0           | 1.0         | 1.0              |
| 200       | 1.0            | 1.0         | 0.5              |
</details>

(b) MLP   
Figure 7. Visualization of results on synthetic data. Non-stationary signals with time-varying frequencies were used for training and testing. (a) presents the visualization produced by TimeStacker, while (b) presents the visualization produced by the MLP.