# VISIONTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters

Mouxiang Chen $^{1}$ Lefei Shen $^{1}$ Zhuo Li $^{2}$ Xiaoyun Joy Wang $^{2}$ Jianling Sun $^{1}$ Chenghao Liu $^{3}$

![](images/28356b7a976052fb242da6f6b4d303af9d5a7b90fbf393f5e5265393bb2274b4.jpg)

<details>
<summary>bar</summary>

| Model | Mean Square Error (MSE) |
|---|---|
| ETSformer (2022) | 0.75 |
| Stationary (2022) | 0.68 |
| Autoformer (2021) | 0.55 |
| TimesNet (2023) | 0.49 |
| DLinear (2023) | 0.41 |
| PatchTST (2023) | 0.38 |
| GPT4TS (2023) | 0.36 |
| TimeLLM (2024) | 0.34 |
| MoiraiSmall (2024) | 0.33 |
| MoiraiLarge (2024) | 0.33 |
| VisionTS (ours) | 0.31 |
The game began development in 2010, carrying over a large portion of the work.
Natural Texts
Time-Series
Natural Images (no time-series training)
GIFT-Eval Leaderboard
Cut-off date: Nov 2024
Lag-Llama
Timer
TTMs
TimesFM
Moirai (S)
Moirai (B)
Chronos (S)
Chronos (B)
Moirai (L)
Chronos (L)
VisionTS
</details>

Figure 1. Long-term forecasting (left) and GIFT-Eval (right) performance comparison. Our VISIONTS, without any training on time series data, outperforms the pure time series foundation models in the zero-shot setting.

# Abstract

Foundation models have emerged as a promising approach in time series forecasting (TSF). Existing approaches either repurpose large language models (LLMs) or build large-scale time series datasets to develop TSF foundation models for universal forecasting. However, these methods face challenges due to the severe cross-domain gap or in-domain heterogeneity. This paper explores a new road to building a TSF foundation model from rich, high-quality natural images. Our key insight is that a visual masked autoencoder, pre-trained on the ImageNet dataset, can naturally be a numeric series forecaster. By refor-

mulating TSF as an image reconstruction task, we bridge the gap between image pre-training and TSF downstream tasks. Surprisingly, without further adaptation in the time series domain, the proposed VISIONTS could achieve better zero-shot forecast performance than existing TSF foundation models. With fine-tuning for one epoch, VISIONTS could further improve the forecasting and achieve state-of-the-art performance in most cases. Extensive experiments reveal intrinsic similarities between images and real-world time series, suggesting that visual models may offer a “free lunch” for TSF and highlight the potential for future cross-modality research. Our code is publicly available at https://github.com/Keytoyze/VisionTS.

# 1. Introduction

Foundation models (Bommasani et al., 2021) have revolutionized natural language processing (NLP) and computer

vision (CV) in recent years (Brown et al., 2020; He et al., 2022). By pretraining on large-scale data, they have shown remarkable few-shot and even zero-shot performance across various downstream tasks. This has motivated an emergent paradigm shift in time series forecasting (TSF), moving from a traditional one-model-per-dataset framework to universal forecasting with a single pre-trained model (Woo et al., 2024; Goswami et al., 2024). A TSF foundation model can greatly reduce the need for downstream data and demonstrate strong forecasting performance on diverse domains, such as energy consumption planning, weather forecasting, and traffic flow.

We have recently witnessed two roads to building a TSF foundation model for universal forecasting. The first tries to repurpose large language models (LLMs) that have been pre-trained on text data for TSF tasks (i.e., text-based) (Zhou et al., 2023; Jin et al., 2024), based on the observation that LLMs and TSF models share a similar left-to-right forecasting paradigm. However, due to the significant gap between these two modalities, the effectiveness of such transferability between language and time series has recently been questioned by Tan et al. (2024).

The second road focuses on constructing large-scale time-series datasets collected from diverse domains to train a TSF foundation model from scratch (i.e., time series-based or TS-based) (Woo et al., 2024; Das et al., 2024). Nevertheless, unlike images or language with unified formats, time series data is highly heterogeneous in length, frequency, number of variates, domains, and semantics, limiting the transferability between pre-training and downstream domains. Until recently, constructing a high-quality dataset remains challenging and is still in the early exploration stage.

In this paper, we investigate a third road that is less explored yet promising: building TSF foundation models with pre-trained visual models. Our key idea is that pixel variations in a natural image can be interpreted as temporal sequences, which share many intrinsic similarities with time series: ① Similar modalities: Unlike discrete texts, both images and time series are continuous; ② Similar origin: Both time series and images are observations of real-world physical systems, whereas languages are products of human cognitive processes; ③ Similar information density: Languages are human-generated signals with high semantic density, while images and time series are natural signals with heavy redundancy (He et al., 2022); and ④ Similar features: As shown in Section 1, images often display many features of real-world time series, which are rarely found in language data. Based on these findings, images could be a promising modality for transferring to TSF. We are motivated to answer the question: Can a visual model pre-trained on images be a free-lunch foundation model for time series forecasting?

![](images/53ecaa80780187c5ce47e51109beb8f9f2f43c558fbdcbe0b1c011fd321185fe.jpg)

<details>
<summary>text_image</summary>

Trend
Seasonality
Greyscale
Position
Greyscale
Position
Greyscale
Position
Stationarity
Sudden Change
Greyscale
Position
Greyscale
Position
</details>

Figure 2. An image of the ImageNet dataset (Deng et al., 2009), in which the pixel arrays can display many well-known features of real-world time series, such as trend, seasonality, and stationarity (Qiu et al., 2024). By self-supervised pre-training on ImageNet, it is reasonable that a visual model could understand these features and exhibit a level of time series forecasting ability.

We focus on visual masked autoencoder (MAE) $^{1}$ , a popular CV foundation model (He et al., 2022) by self-supervised pre-training on ImageNet (Deng et al., 2009). As an image reconstruction and completion model, MAE can naturally be a numeric series forecaster. Inspired by the well-known prompt technique in NLP (Schick & Schütze, 2021), we propose a simple method to reformulate TSF as a patch-level image reconstruction task to bridge the gap between pre-training and downstream tasks. Specifically, we transform 1D time-series data into 2D matrices via segmentation. Then, we render the matrices into images and align the forecasting window with masked image patches. This allows us to make a zero-shot forecast without further adaptation.

We evaluate our proposed VISIONTS on large-scale benchmarks, including 8 long-term TSF (Zhou et al., 2021), 29 Monash (Godahewa et al., 2021), and 23 GIFT-Eval (Aksu et al., 2024) datasets, spanning diverse domains, frequencies, and multivariates. To the best of our knowledge, the scale of our evaluation benchmark is the largest among existing TSF foundation models. As demonstrated in Fig. 1, without further adaptation on time series, a vanilla MAE can

surprisingly achieve a comparable performance or even outperform the strong zero-shot TSF foundation models. By fine-tuning MAE in each downstream dataset for a single epoch, VISIONTS can lead to SOTA performance in most long-term TSF benchmarks.

To further understand and explain the transferability, we use an MAE encoder to visualize both modalities, showing a level of similarity between time series and natural image representations. Additionally, we observe considerable heterogeneity within time-series data across domains, and images can serve as a bridge to connect these isolated time-series representations. This could further explain why VISIONTS performs better than some cross-domain TSF models. Our findings suggest that time series and natural images may be two sides of a coin, and visual models can be a free lunch for time series forecasting. We hope our findings inspire future cross-modal research on CV and TSF.

Our contributions are summarized as follows:

- We explore a road to building a TSF foundation model from natural images, conceptually different from the existing text-based and TS-based pre-training methods.   
- We introduce VISIONTS, a novel TSF foundation model based on a visual MAE. To bridge the gap between the two modalities, we reformulate the TSF task into an image reconstruction task.   
- Comprehensive evaluations of VISIONTS on large-scale benchmarks across multiple domains demonstrate its significant forecasting performance, surpassing few-shot text-based TSF foundation models and achieving comparable or superior results to zero-shot TS-based models.

# 2. Preliminaries

Time Series Forecasting (TSF) For a multivariate time series with M variables, let $x_{t} \in R^{M}$ represent the value at t-th time step. Given a historical sequence (i.e., look-back window) $X_{t-L:t} = [x_{t-L}, \cdots, x_{t-1}] \in R^{L \times M}$ with context length L, the TSF task is to predict future values (i.e., forecast horizon) with prediction length H: $\hat{X}_{t:t+H} = [x_{t}, \cdots, x_{t+H-1}] \in R^{H \times M}$ .

Patch-Level Image Reconstruction To obtain high-quality visual representation for downstream CV tasks, He et al. (2022) proposed masked autoencoder (MAE) to pretrain a Vision Transformer (ViT) (Dosovitskiy et al., 2021) using a patch-level image reconstruction task on ImageNet. Specifically, for an image of size $W \times W$ (where W represents both the width and height, as ImageNet images are square), the image is evenly divided into $N \times N$ patches, each with a width and height of $S = W/N$ . During pretraining, some random patches are masked, while the remaining visible patches are fed into the ViT with their position encodings. MAE are trained to reconstruct the masked pixel values from these visible patches.

# 3. Methodology

As noted in the Introduction, time series and images share intrinsic similarities, suggesting the transfer potential of pre-trained visual models (particularly MAE in this paper) for TSF. To reformulate TSF tasks into MAE's pre-training task, our high-level idea is straightforward: map the look-back/forecasting windows to visible/masked patches, respectively. This idea is supported by the prompt tuning (Schick & Schütze, 2021) in NLP, where the predictions for [mask] token in pre-trained language models, e.g., BERT (Devlin et al., 2019), are directly used for downstream tasks. By unifying the forms of the two tasks, we bridge the gap between the two modalities without further training.

However, implementing this idea poses a challenge: the dimension of time-series data (1D) is different from images (2D). Moreover, the size of images in the pre-training dataset is fixed at $224 \times 224$ , while the lengths of time series data can vary dynamically. In the following, we describe the details of VISIONTS to address this challenge. Our architecture is depicted in Fig. 3.

Segmentation Given a univariate input $X \in R^{L}$ , the first goal is to transform it into a 2D matrix. We propose to segment it into $\lfloor L/P \rfloor$ subsequences of length P, where P is the periodicity. Notably, when the time series lacks clear periodicity, we can set P = 1 directly, which is also effective in our experiments (Appendix B.6). In practice, P can be determined using statistical methods like Fast Fourier Transform (Wu et al., 2023; Chen et al., 2024) or domain knowledge like sampling frequency (Godahewa et al., 2021; Alexandrov et al., 2020). In this paper, we select P based on the sampling frequency, elaborated in Appendix A.2.

After that, these subsequences are then stacked into a 2D matrix, denoted by $I_{\mathrm{raw}} \in \mathbb{R}^{P \times \lfloor L / P \rfloor}$ . This encoding strategy is proven to be efficient by recent work like TimesNet (Wu et al., 2023) and SparseTSF (Lin et al., 2024), as it allows for the simultaneous capture of both variations within the same period (i.e., intra-period) and across periods with the same phase (i.e., inter-period). Moreover, it ensures that each element in $I_{\mathrm{raw}}$ and its neighbors align with the spatial locality property of images (Krizhevsky et al., 2012), where nearby pixels tend to be similar due to the inherent cohesiveness of objects in the real world. Therefore, this further narrows the gap between time series and images.

![](images/9ef6587400269430f3ac7bd6af8950f7697830f7fdbd4870ad7723048c0a6f89.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["look-back window"] --> B["① segmentation"]
    B --> C["② render"]
    C --> D["Resize the image through interpolation"]
    D --> E["③ alignment"]
    
    subgraph_A["look-back window"]
        F["time waveforms"] --> G["time waveforms"]
        H["seconds"] --> I["seconds"]
        J["seconds"] --> K["seconds"]
        L["seconds"] --> M["seconds"]
        N["seconds"] --> O["seconds"]
        P["seconds"] --> Q["seconds"]
        R["seconds"] --> S["seconds"]
        T["seconds"] --> U["seconds"]
        V["seconds"] --> W["seconds"]
        X["seconds"] --> Y["seconds"]
        Z["seconds"] --> AA["seconds"]
        AB["seconds"] --> AC["seconds"]
        AD["seconds"] --> AE["seconds"]
        AF["seconds"] --> AG["seconds"]
        AH["seconds"] --> AI["seconds"]
        AJ["seconds"] --> AK["seconds"]
        AL["seconds"] --> AM["seconds"]
        AN["seconds"] --> AO["seconds"]
        AP["seconds"] --> AQ["seconds"]
        AR["seconds"] --> AS["seconds"]
        AT["seconds"] --> AU["seconds"]
        AV["seconds"] --> AW["seconds"]
        AX["seconds"] --> AY["seconds"]
        AZ["seconds"] --> BA["seconds"]
        BB["seconds"] --> BC["seconds"]
        BD["seconds"] --> BE["seconds"]
        BF["seconds"] --> BG["seconds"]
        BH["seconds"] --> BI["seconds"]
        BJ["seconds"] --> BK["seconds"]
        BL["seconds"] --> BM["seconds"]
        BN["seconds"] --> BO["seconds"]
        BP["seconds"] --> BQ["seconds"]
        BR["seconds"] --> BS["seconds"]
        BT["seconds"] --> BU["seconds"]
        BV["seconds"] --> BW["seconds"]
        BX["seconds"] --> BY["seconds"]
        BZ["⑤ forecasting"]
    end
    
    subgraph B
        B1[① segmentation: time waveforms, segments, periods, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrowheads
    end
    
    subgraph C
        C1[② render: time waveforms, segments, periods, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrows, arrowheads
    end
    
    subgraph D
        D1[④ reconstruction: patch-level image reconstruction: visual MAE: visual MAE with ?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?/ ?
    end
```
</details>

Figure 3. VISIONTS architecture. The input is first segmented by period, rendered into a grayscale image, and then aligned with the visible patches on the left through resampling. MAE is used to predict the masked patches on the right, and the reconstructed image is then reversed to forecasting.

Normalization MAE standardizes each image based on the mean and standard deviation computed on ImageNet. Therefore, we apply instance normalization to $I_{raw}$ , which is also a standard practice in current TSF (Kim et al., 2022). Notably, we observed that normalizing $I_{raw}$ to a standard deviation of r, where r is a hyperparameter less than 1, yields superior performance. One explanation is that the magnitude of inputs/outputs during MAE pretraining is constrained by the limited range of color values. Therefore, reducing the magnitude of $I_{raw}$ prevents exceeding these limits. However, an excessively low r can result in values that are difficult to distinguish. We found that a moderate value (0.4) of r performs well across most scenarios (See Appendix B.9 for more details). Let $I_{norm}$ denote the normalized matrix, which is computed as follows:

$$
\boldsymbol {I} _ {\text { norm }} = r \cdot \frac {\boldsymbol {I} _ {\text { raw }} - \operatorname{Mean} (\boldsymbol {I} _ {\text { raw }})}{\text { Standard   -   Deviation } (\boldsymbol {I} _ {\text { raw }})}.
$$

Rendering Since each image has three channels, we simply render $I_{norm}$ as a grayscale image $I_{grey} \in R^{P \times \lfloor L/P \rfloor \times 3}$ , where all three channels are identical to $I_{norm}$ . This choice is purely result-driven: In our early experiments, we added a convolutional layer with three output channels to convert the grayscale image into a color image and then fine-tuned it to find the optimal color transformation, which, however, did not significantly improve the performance.

Alignment Our goal is to predict the columns on the right of $I_{grey}$ to forecast the future sequence. A straightforward approach is to treat $I_{grey}$ as the visible left portion and the predicted columns as the masked right portion. However, since the image size during pre-training may not match the size of $I_{grey}$ , we propose to resize $I_{grey}$ to align with the pre-training data. Formally, let the total number of 2D patches used in pre-training be $N \times N$ and the size of each patch be $S \times S$ . We set the number of visible patches to $N \times n$ and the masked patches to $N \times (N - n)$ , where $n = \lfloor N \cdot L/(L + H) \rfloor$ is determined by the ratio of context length L to prediction length H. We resample the image $I_{grey}$ to adjust the size from the original dimensions $(P, \lfloor L/P \rfloor)$ to $(N \cdot S, n \cdot S)$ , making it more compatible with MAE. We select bilinear interpolation for the resampling process.

Moreover, we found that reducing the width of the visible portion can further improve performance. One possible explanation is that MAE uses a large masked ratio during pre-training, with only 25% of patches visible. Reducing the image width may align the masked ratio more closely with pre-training. Therefore, we propose multiplying n by a hyperparameter $c \in [0,1]$ . Similar to r, we found that setting c = 0.4 performs well in our experiments (See Appendix B.9). This can be formulated as $n = \lfloor c \cdot N \cdot L/(L+H) \rfloor$ .

Reconstruction and Forecasting After obtaining the MAE-reconstructed image, we simply reverse the previous steps for forecasting. Specifically, we resize the entire image back to the original time series segmentations through the same bilinear interpolation, and average the three channels to obtain a single-channel image. After de-normalizing and flattening, the forecasting window can be extracted.

Discussion on Multivariate Forecasting In addition to the temporal interactions, multivariate time series data sometimes show interactions between variables. While pre-

Table 1. Zero-shot or few-shot results on the long-term TSF benchmark. Results are averaged across prediction lengths {96, 192, 336, 720}, with full results in Table 9 (Appendix B.2). Bold: the best result. 

<table><tr><td rowspan="3" colspan="2">Pretrain Method</td><td colspan="4">Zero-Shot</td><td colspan="7">Few-Shot (10% In-distribution Downstream Dataset)</td></tr><tr><td>Images</td><td colspan="3">Time series</td><td colspan="2">Text</td><td colspan="5">No Pretrain</td></tr><tr><td>VISIONTS</td><td>MOIRAI $_S$ </td><td>MOIRAI $_B$ </td><td>MOIRAI $_L$ </td><td>TimeLLM</td><td>GPT4TS</td><td>DLinear</td><td>PatchTST</td><td>TimesNet</td><td>Autoformer</td><td>Informer</td></tr><tr><td rowspan="2">ETTh1</td><td>MSE</td><td>0.390</td><td>0.400</td><td>0.434</td><td>0.510</td><td>0.556</td><td>0.590</td><td>0.691</td><td>0.633</td><td>0.869</td><td>0.702</td><td>1.199</td></tr><tr><td>MAE</td><td>0.414</td><td>0.424</td><td>0.439</td><td>0.469</td><td>0.522</td><td>0.525</td><td>0.600</td><td>0.542</td><td>0.628</td><td>0.596</td><td>0.809</td></tr><tr><td rowspan="2">ETTh2</td><td>MSE</td><td>0.333</td><td>0.341</td><td>0.346</td><td>0.354</td><td>0.370</td><td>0.397</td><td>0.605</td><td>0.415</td><td>0.479</td><td>0.488</td><td>3.872</td></tr><tr><td>MAE</td><td>0.375</td><td>0.379</td><td>0.382</td><td>0.377</td><td>0.394</td><td>0.421</td><td>0.538</td><td>0.431</td><td>0.465</td><td>0.499</td><td>1.513</td></tr><tr><td rowspan="2">ETTm1</td><td>MSE</td><td>0.374</td><td>0.448</td><td>0.382</td><td>0.390</td><td>0.404</td><td>0.464</td><td>0.411</td><td>0.501</td><td>0.677</td><td>0.802</td><td>1.192</td></tr><tr><td>MAE</td><td>0.372</td><td>0.410</td><td>0.388</td><td>0.389</td><td>0.427</td><td>0.441</td><td>0.429</td><td>0.466</td><td>0.537</td><td>0.628</td><td>0.821</td></tr><tr><td rowspan="2">ETTm2</td><td>MSE</td><td>0.282</td><td>0.300</td><td>0.272</td><td>0.276</td><td>0.277</td><td>0.293</td><td>0.316</td><td>0.296</td><td>0.320</td><td>1.342</td><td>3.370</td></tr><tr><td>MAE</td><td>0.321</td><td>0.341</td><td>0.321</td><td>0.320</td><td>0.323</td><td>0.335</td><td>0.368</td><td>0.343</td><td>0.353</td><td>0.930</td><td>1.440</td></tr><tr><td rowspan="2">Electricity</td><td>MSE</td><td>0.207</td><td>0.233</td><td>0.188</td><td>0.188</td><td>0.175</td><td>0.176</td><td>0.180</td><td>0.180</td><td>0.323</td><td>0.431</td><td>1.195</td></tr><tr><td>MAE</td><td>0.294</td><td>0.320</td><td>0.274</td><td>0.273</td><td>0.270</td><td>0.269</td><td>0.280</td><td>0.273</td><td>0.392</td><td>0.478</td><td>0.891</td></tr><tr><td rowspan="2">Weather</td><td>MSE</td><td>0.269</td><td>0.242</td><td>0.238</td><td>0.260</td><td>0.234</td><td>0.238</td><td>0.241</td><td>0.242</td><td>0.279</td><td>0.300</td><td>0.597</td></tr><tr><td>MAE</td><td>0.292</td><td>0.267</td><td>0.261</td><td>0.275</td><td>0.273</td><td>0.275</td><td>0.283</td><td>0.279</td><td>0.301</td><td>0.342</td><td>0.495</td></tr><tr><td rowspan="2">Average</td><td>MSE</td><td>0.309</td><td>0.327</td><td>0.310</td><td>0.329</td><td>0.336</td><td>0.360</td><td>0.407</td><td>0.378</td><td>0.491</td><td>0.678</td><td>1.904</td></tr><tr><td>MAE</td><td>0.345</td><td>0.357</td><td>0.344</td><td>0.350</td><td>0.368</td><td>0.378</td><td>0.416</td><td>0.389</td><td>0.446</td><td>0.579</td><td>0.995</td></tr><tr><td colspan="2"> $1^{st}$  count</td><td>7</td><td>0</td><td>3</td><td>1</td><td>2</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

![](images/c2ac161fb7929252a46ae15f8bee47b26034295d462181dc55b2373d3c08a52c.jpg)

<details>
<summary>bar</summary>

| Method          | Norm. MASE |
| --------------- | ---------- |
| Lag-Llama       | 1.2        |
| Timer           | 1.0        |
| TTMs            | 0.95       |
| TimesFM         | 0.95       |
| Moirai (S)      | 0.85       |
| Moirai (B)      | 0.8        |
| Chronos (S)     | 0.8        |
| Chronos (B)     | 0.8        |
| Moirai (L)      | 0.8        |
| Chronos (L)     | 0.8        |
| VisionTS        | 0.75       |
</details>

Figure 4. Performance on the GIFT-Eval Leaderboard (cut-off at VISIONTS's release).

![](images/56d994d199d1004cd046d0f01d9f3425e9b78e2d145c3fa11ea72460d1ca1c51.jpg)

<details>
<summary>bar</summary>

| Model | Norm. MAE |
|---|---|
| LLMTime | 1.05 |
| SES | 1.03 |
| Theta | 0.92 |
| ARIMA | 0.90 |
| ETS | 0.87 |
| PR | 0.78 |
| N-BEATS | 0.78 |
| Trsf | 0.77 |
| CatBoost | 0.76 |
| DeepAR | 0.76 |
| TBATS | 0.76 |
| WaveNet | 0.75 |
| FFNN | 0.74 |
| VisionTS | 0.73 |
| Moirai(S) | 0.66 |
</details>

Figure 5. Aggregated results on the Monash TSF Benchmark, with full results in Table 15 (Appendix B.6).

trained vision models effectively capture temporal interactions based on the intrinsic similarities between images and time series, they struggle to capture inter-variable interactions due to a limited number of image channels, especially without further training. Fortunately, recent work shows that channel independence — forecasting each variable separately — can be effective and is widely used in recent deep forecasting models (Nie et al., 2022; Han et al., 2024; Jin et al., 2024; Zhou et al., 2023; Lin et al., 2024). Following these works, we adopt channel independence in our paper while leaving the exploration of capturing inter-variable interactions to future work.

# 4. Experiments

We follow the standard evaluation protocol proposed by Woo et al. (2024) to test our VISIONTS on 35 widely-used TSF benchmarks, and additionally evaluate it on the GIFT-Eval (Aksu et al., 2024) which is the largest TSF benchmark for zero-shot foundation models. We use MAE (Base) as our backbone by default. Baseline and benchmark details are elaborated in Appendix A.1.

# 4.1. Zero-Shot Time Series Forecasting

Setups We first evaluate VISIONTS's zero-shot TSF performance without fine-tuning on time-series modalities. To prevent data leakage, we selected six widely-used datasets from the long-term TSF benchmark that are not included in MOIRAI's pre-training set for evaluation. Since most baselines cannot perform zero-shot forecasting, we report their few-shot results by fine-tuning on the 10% of the individual target datasets. We also evaluate the Monash benchmark and GIFT-Eval benchmark. Notably, the Monash benchmark is more challenging for VISIONTS since they were used in MOIRAI's pre-training but not for VISIONTS. We set the hyperparameters to r = c = 0.4. Following common practice (Zhou et al., 2023; Woo et al., 2024), we conduct hyperparameter tuning on validation sets to determine the optimal context length L, detailed in Appendix B.1.

Table 2. Average MSE of different MAE variants, with full results in Table 17 (Appendix B.7). 

<table><tr><td></td><td>Base112M</td><td>Large330M</td><td>Huge657M</td></tr><tr><td>ETTh1</td><td>0.390</td><td>0.378</td><td>0.391</td></tr><tr><td>ETTh2</td><td>0.333</td><td>0.340</td><td>0.339</td></tr><tr><td>ETTm1</td><td>0.374</td><td>0.379</td><td>0.383</td></tr><tr><td>ETTm2</td><td>0.282</td><td>0.286</td><td>0.284</td></tr><tr><td>Electricity</td><td>0.207</td><td>0.209</td><td>0.202</td></tr><tr><td>Weather</td><td>0.269</td><td>0.272</td><td>0.292</td></tr><tr><td>Avg.</td><td>0.309</td><td>0.311</td><td>0.315</td></tr></table>

Table 3. Computational cost in terms of seconds for forecasting a batch of 32 time series data. 

<table><tr><td>Context Length</td><td colspan="4">1k</td><td>1k</td><td>2k</td><td>3k</td><td>4k</td></tr><tr><td>Prediction Length</td><td>1k</td><td>2k</td><td>3k</td><td>4k</td><td></td><td colspan="3">1k</td></tr><tr><td>PatchTST</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.03</td><td>0.04</td></tr><tr><td>DeepAR</td><td>0.26</td><td>0.32</td><td>0.37</td><td>0.43</td><td>0.26</td><td>4.06</td><td>6.10</td><td>8.17</td></tr><tr><td>GPT4TS</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.01</td><td>0.03</td><td>0.04</td><td>0.06</td></tr><tr><td>MOIRAI $_{Base}$ </td><td>0.03</td><td>0.04</td><td>0.04</td><td>0.05</td><td>0.03</td><td>0.04</td><td>0.05</td><td>0.06</td></tr><tr><td>TimesFM</td><td>0.08</td><td>0.14</td><td>0.20</td><td>0.27</td><td>0.07</td><td>0.13</td><td>0.20</td><td>0.25</td></tr><tr><td>LLMTime (8B)</td><td colspan="4">&gt;200</td><td colspan="4">&gt;200</td></tr><tr><td>VISIONTS (c=0.4)</td><td>0.04</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.04</td><td>0.05</td><td>0.05</td></tr></table>

Results on Long-Term TSF Benchmark Table 1 shows that VISIONTS surprisingly achieves the best forecasting performance in most cases (7 out of 14). Specifically, VISIONTS demonstrates a relative average MSE reduction of approximately 6% compared to MOIRAI $_{Small}$ and MOIRAI $_{Large}$ , and performs comparably to MOIRAI $_{Base}$ . When compared to the various few-shot baselines, VISIONTS shows a relative average MSE reduction ranging from 8% to 84%. Given that all baselines except for VISIONTS are trained on the time-series domain, this result is particularly encouraging. It suggests that the transferability from images to time-series is stronger than from text to time-series, and even comparable to the in-domain transferability between time-series. We also include a comparison with two TSF foundation models, TimesFM (Das et al., 2024) and LLMTime (Gruver et al., 2023), in Appendix B.3, as well as traditional algorithms (ETS, ARIMA, and Seasonal Naïve) in Appendix B.4. Results show that VISIONTS still outperforms all of these baselines.

Results on GIFT-Eval Benchmarks Fig. 4 shows the comparison of VISIONTS with six previously published TSF foundation models on the GIFT-Eval TSF Leaderboard $^{2}$ , where VISIONTS surprisingly ranked first in terms of normalized MASE. It should be noted that although some concurrent works (after the release of VISIONTS) in the current leaderboard outperform VISIONTS, there may be data leakage issues for these works. In contrast, visual MAE was trained on ImageNet, long before the release of GIFT-Eval leaderboard, which can ensure no leakage.

Results on Monash Benchmark Fig. 5 shows the results aggregated from 29 Monash datasets, showing that VISIONTS in the zero-shot setting surpasses all models individually trained on each dataset and significantly outperforms the other cross-domain baseline (i.e., LLMTime). It achieves second place among all baselines, just behind MOIRAI that pre-trained on all the training datasets. This promising result highlights VisionTS's strong zero-shot forecasting ability and effective cross-modality transferability.

# 4.2. Further Analysis of VISIONTS

Backbone Analysis In Table 2 (full results in Appendix B.7), we observe that the overall performance of three MAE variants (112M, 330M, and 657M) outperforms MOIRAI $_{Small}$ and MOIRAI $_{Large}$ . Particularly, larger models show a slight decrease in performance. This may be due to larger visual models overfitting image-specific features, reducing their transferability. A similar phenomenon was reported in MOIRAI, where larger models were found to degrade performance. We leave the exploration of scaling laws in image-based TSF foundation models for the future. Additionally, to explore the potential with other vision models, we also test LaMa (Suvorov et al., 2022), a visual inpainting model. Results in Appendix B.7 demonstrate that VISIONTS with LaMa performs similarly to MOIRAI in the zero-shot setting. This suggests that the performance is driven by the inherent similarity between images and time series, not solely by the MAE model.

Computational Cost We evaluate the computation cost of different baselines on an NVIDIA A800 GPU. Results are averaged on 90 runs. Table 3 shows the results between various TSF foundation models, showing that VISIONTS are comparable to MOIRAI $_{Base}$ and GPT4TS and faster than TimesFM, which is an auto-regressive model. While computation time increases with context length for all the other Transformer-based baselines, VISIONTS remains nearly constant. This is because VISIONTS encodes input sequences into an image with constant size, ensuring $O(1)$ efficiency. In contrast, Transformer-based methods operate at $O(L^{2})$ relative to context length L.

Hyperparameter Analysis Appendix B.9 illustrates the impact of three hyperparameters. For context length L, as shown in Fig. 6, performance typically improves with increasing L, particularly on high-frequency datasets like

![](images/08b05c1e579cdec381464f96e1908f99403e1932c51219a41e9d826a916578b9.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 500  | 0.325 |
| 1000 | 0.320 |
| 1500 | 0.315 |
| 2000 | 0.310 |
| 2500 | 0.295 |
| 3000 | 0.285 |
| 3500 | 0.275 |
| 4000 | 0.270 |
</details>

(a) Weather

![](images/322846ca6bed50cda7bfa3bbed485f352e92da3c45e7eb016d13d881e41cc952.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.42  |
| 1000 | 0.41  |
| 2000 | 0.39  |
| 3000 | 0.38  |
| 4000 | 0.38  |
</details>

(b) ETTm1

![](images/abafd4ecf8c8c546d6aa652b9a96ef879050acec6f88a5c6a197a27430f59b63.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.315 |
| 1000 | 0.305 |
| 2000 | 0.285 |
| 3000 | 0.287 |
| 4000 | 0.282 |
</details>

(c) ETTm2

![](images/57cde399c907aa337eb8e5c722747a43f9a4f6804ceced18481023aa2d190485.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 500  | 0.25  |
| 1000 | 0.22  |
| 1500 | 0.21  |
| 2000 | 0.21  |
| 2500 | 0.21  |
| 3000 | 0.21  |
| 3500 | 0.21  |
| 4000 | 0.215 |
</details>

(d) Electricity

Figure 6. MSE (Y-axis) performance of different context lengths L (X-axis), averaged on four prediction lengths.   
![](images/f38f9309d6ff1d35d6ac2431c9a98b7c8924f76322a315581cc78f1fa82f8931.jpg)

<details>
<summary>scatter</summary>

| Category    | Count |
| ----------- | ----- |
| Monash      | 120   |
| Weather     | 85    |
| ETTm1       | 60    |
| Electricity | 45    |
| ImageNet    | 30    |
</details>

Figure 7. Modality visualization of the images (ImageNet) and time series (Monash, Weather, Electricity, and ETTm1) based on the MAE encoder.

Table 4. Aggregated full-shot forecasting performance on eight long-term TSF benchmarks (ETTh1, ETTh2, ETTm1, ETTm2, Illness, Weather, Traffic, and Electricity). VISIONTS is fine-tuned only a single epoch on each dataset except for Illness. Due to the space limit, we report the $1^{\text{st}}$ count for each baseline, with full results in Table 21 (Appendix C.2). 

<table><tr><td rowspan="2">Pretrain Method</td><td>Images</td><td colspan="2">Text</td><td colspan="8">No Pretrain</td></tr><tr><td>VISIONTS</td><td>Time-LLM</td><td>GPT4TS</td><td>Dlinear</td><td>PatchTST</td><td>TimesNet</td><td>FEDformer</td><td>Autoformer</td><td>Stationary</td><td>ETSformer</td><td>Informer</td></tr><tr><td> $1^{st}$  count</td><td>46</td><td>4</td><td>12</td><td>0</td><td>19</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Weather (10-minute frequency) and ETTm1/ETTm2 (15-minute frequency). This aligns with other TSF foundation models like MOIRAI. As for the normalization constant r and alignment constant c, when both of them are around 0.4, performance is generally well across most benchmarks.

Modality Analysis: Where does the zero-shot forecastability come from? We further examine the gap between time series and images to explain the transferability of zero-shot forecasting. We sampled 1,000 images from ImageNet-1k and 300 samples from each time series dataset. We fed them into the MAE, maintaining a consistent image mask across all data. Fig. 7 visualizes the MAE encoder outputs of these data, which are flattened and reduced to 2-dimension by t-SNE. Notably, some time series, such as ETTm1 and Electricity, fall within the ImageNet distribution. It suggests a relatively small gap between images and some time series (e.g., Electricity and ETTm1), which could explain the good transferability. Additionally, while ImageNet displays a concentrated distribution, time series are generally more scattered. For instance, ETTm1 clusters in the upper right, whereas Monash is found in the lower left, with a significant gap. This indicates strong heterogeneity within time series data and suggests that images may serve as a bridge to connect isolated time series modality.

Ablation Study We conduct experiments to validate our choices in the Alignment step, detailed in Appendix B.8. First, we test three different interpolation strategies, which shows that Bilinear interpolation performs best. Second, we apply horizontal and vertical flips on the image to examine whether the assumed left-to-right, top-to-bottom order is efficient. Results show that these changes do not significantly affect performance, suggesting that image reconstruction is isotropic and not influenced by certain orientation.

Qualitative Analysis: When does VISIONTS perform well, and when does it not? In Appendix D, we visualize the zero-shot forecasting of VISIONTS alongside the input and reconstruction images, highlighting both successful cases (where VISIONTS outperforms MOIRAI) and failures (where MOIRAI prevails). When the input exhibits strong regularity (Fig. 11), VISIONTS effectively forecasts both the periodicity (via segmentation) and trends (via MAE's capabilities). In contrast, MOIRAI, akin to seasonal naïve methods, struggles to capture inter-period trends. For less-structured input (Figs. 12 to 14), MOIRAI adopts a conservative approach with lower volatility to minimize errors, while VISIONTS takes a more aggressive stance. This strategy occasionally yields more accurate trend predictions (Figs. 12

and 13) but may also result in greater MAE (Fig. 14).

# 4.3. Full-Shot Long-Term Time Series Forecasting

Setups We evaluate the full-shot capability of each baseline trained on individual long-term TSF benchmarks. In addition to the six datasets used for zero-shot forecasting, we also include the popular Traffic and Illness datasets. As self-attention and feed-forward layers contain rich knowledge that can be transferred to TSF, we choose to fine-tune only the layer normalization (LN) layers while freezing the other parameters, which is also adopted by Zhou et al. (2023). Training details are elaborated in Appendix C.1.

Main Results Table 4 summarizes the full-shot results, with the full results and standard deviations detailed in Appendix C.2. It shows that VISIONTS outperforms other baselines in most cases (46 out of 80), surpassing the non-pretrained PatchTST and the language-pretrained GPT4TS. Remarkably, except for Illness with the least data, VISIONTS demands only a single epoch of fine-tuning. This suggests that even minimal fine-tuning enables VisionTS to adapt to time series effectively. Compared with Table 1, fine-tuning provides limited benefits for ETTh1 and ETTh2 but significantly improves other datasets. We attribute this to the smaller data scale of ETTh1 and ETTh2.

Ablation Study Tan et al. (2024) proposed several ablation variants for text-based foundation models, including w/o LLM (removing the LLM), LLM2Attn/LLM2Trsf (replacing the LLM with a single self-attention/Transformer layer), and RandLLM (randomly initializing the LLM). They found no significant performance differences and concluded that textual knowledge is unnecessary for TSF. We conducted similar ablations to assess the role of the vision model (VM), including w/o VM, VM2Attn, VM2Trsf, and RandVM. Appendix C.3 shows that these variants lead to worse performance, indicating that visual knowledge is beneficial for TSF.

Analysis: Fine-tuning strategies As stated before, we fine-tune only the layer normalization (LN). We also tested fine-tuning the bias, MLP, or attention layers, in addition to full fine-tuning and freezing. All hyperparameters were kept constant. Note that freezing differs from the previous zero-shot experiment, where a longer context length was used. Appendix C.3 show that fine-tuning LN is the best. Modifying MLP or attention layers results in significant performance drops, suggesting that valuable knowledge resides in these components.

# 5. Related Work

Depending on the pre-training data, TSF foundation models can be categorized into Text-based and TS-based models. We first review related works and then introduce recent research on image-based time series analysis.

Text-based TSF Foundation Models Large Language Models (LLMs) pre-trained on large amounts of text data are being applied to TSF tasks. For example, Zhou et al. (2023) fine-tuned a pre-trained GPT (Radford et al., 2019) on each time-series downstream task, such as forecasting, classification, imputation, and anomaly detection. Based on Llama (Touvron et al., 2023), Jin et al. (2024) froze the pre-trained LLM and reprogrammed the time series to align with the language modality. Bian et al. (2024) adopted a two-stage approach by continually pre-training GPT (Radford et al., 2019) on the time-series domain. Nevertheless, the TSF performance of LLMs has recently been questioned by Tan et al. (2024), which designed several ablation studies to show that textual knowledge is unnecessary for forecasting. In this paper, we attribute it to the large modality gap. Some recent approaches focus on directly transforming the time series into natural texts for LLMs, allowing for zero-shot forecasting. For example, PromptCast (Xue & Salim, 2023) used pre-defined templates to describe numerical time series data, while LLMTime (Gruver et al., 2023) directly separated time steps using commas and separates digits using spaces to construct the text input. However, due to the efficiency issue of the autoregressive decoding strategy and the expensive inference cost of large language models, their practical use is limited.

Time Series-Based TSF Foundation Models Self-supervised pre-training a TSF model on the same dataset used for downstream TSF tasks is a well-explored topic (Ma et al., 2023; Zhang et al., 2024), such as denoising autoencoders (Zerveas et al., 2021) or contrastive learning (Woo et al., 2022a; Yue et al., 2022). They follow a similar paradigm to the masked autoencoder (MAE) in computer vision, which is a well-studied topic in other machine learning fields, such as BERT (Devlin et al., 2019), CBraMod (Wang et al., 2025), and HuBERT (Hsu et al., 2021). However, these methods rarely examine the cross-dataset generalization capabilities. Recently, research has shifted towards training universal foundation models, by collecting large-scale time series datasets from diverse domains (Ansari et al., 2024; Goswami et al., 2024; Liu et al., 2024; Das et al., 2024; Dong et al., 2024; Feng et al., 2024) or generating numerous synthetic time series data (Fu et al., 2024; Yang et al., 2024). As a representative method, Woo et al. (2024) collected 27 billion observations across nine domains and trained TSF foundation models of various scales, achieving strong zero-shot performance. However, given the severe

heterogeneity, constructing high-quality large datasets poses significant challenges for building these foundation models.

Image-Based Time-Series Analysis Previous research has investigated encoding time series data into images and used convolutional neural networks (CNNs) trained from scratch for classification (Wang & Oates, 2015a;b; Hatami et al., 2018) or forecasting (Li et al., 2020; Sood et al., 2021; Semenoglou et al., 2023). Recent researchers explored using pre-trained models for these imaging time series. Li et al. (2024) used a pre-trained vision transformer (ViT) for classification. Wimmer & Rekabsaz (2023) and Zhang et al. (2023) employed vision-language multimodal pre-trained models to extract predictive features and generate text descriptions. Yang et al. (2024) generated synthetic time series data to pre-train a vision model for the TSF task. However, these studies did not deeply examine the transferability from natural images to TSF. Despite early efforts by Zhou et al. (2023) to fine-tune a BEiT (Bao et al., 2022) trained on images for time series forecasting, it still falls short of the leading text-based and TS-based TSF foundation models. To the best of our knowledge, we are the first to show that an image-based foundation model, without further time-series adaptation, can match or even surpass other types of TSF foundation models.

# 6. Conclusion

In this paper, we explore a novel approach to building a time series forecasting (TSF) foundation model using natural images, offering a new perspective distinct from the traditional text-based and TS-based methods. By leveraging the intrinsic similarities between images and time series, we introduced VISIONTS, an MAE-based TSF foundation model that reformulates the TSF task as an image reconstruction problem. Our extensive evaluations demonstrate that VISIONTS achieves outstanding forecasting performance in zero-shot and full-shot settings, being a free lunch for a TSF foundation model. We hope our findings could open new avenues for further cross-modality research.

Limitations and Future Directions. (1) As a preliminary study, we employed MAE and LaMa. Utilizing more advanced models like diffusion models (Rombach et al., 2022; Peebles & Xie, 2023) presents a promising research direction. (2) Due to limitations in the visual model, VISIONTS cannot capture multivariate interactions and perform distribution forecasting. Future modifications to the model structure may empower it with more time series capabilities.

# Acknowledgments

Research work mentioned in this paper is supported by State Street Zhejiang University Technology Center. We would also like to thank reviewers for their valuable comments.

# Impact Statement

This paper presents work whose goal is to advance the field of time series forecasting. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Aksu, T., Woo, G., Liu, J., Liu, X., Liu, C., Savarese, S., Xiong, C., and Sahoo, D. Gift-eval: A benchmark for general time series forecasting model evaluation, 2024. URL https://arxiv.org/abs/2410.10393.   
Alexandrov, A., Benidis, K., Bohlke-Schneider, M., Flunkert, V., Gasthaus, J., Januschowski, T., Maddix, D. C., Rangapuram, S., Salinas, D., Schulz, J., Stella, L., Türkmen, A. C., and Wang, Y. GluonTS: Probabilistic and Neural Time Series Modeling in Python. Journal of Machine Learning Research, 21(116):1–6, 2020. URL http://jmlr.org/papers/v21/19-820.html.   
Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024.   
Bao, H., Dong, L., Piao, S., and Wei, F. BEit: BERT pretraining of image transformers. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=p-BhZSz59o4.   
Bian, Y., Ju, X., Li, J., Xu, Z., Cheng, D., and Xu, Q. Multi-patch prediction: Adapting llms for time series representation learning. arXiv preprint arXiv:2402.04852, 2024.   
Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.   
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165.

Chen, M., Shen, L., Fu, H., Li, Z., Sun, J., and Liu, C. Calibration of time-series forecasting: Detecting and adapting context-driven distribution shift. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD '24, pp. 341–352, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704901. doi: 10.1145/3637528.3671926. URL https://doi.org/10.1145/3637528.3671926.   
Das, A., Kong, W., Sen, R., and Zhou, Y. A decoder-only foundation model for time-series forecasting. In Forty-first International Conference on Machine Learning, 2024.   
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi:10.1109/CVPR.2009.5206848.   
Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423.   
Dong, J., Wu, H., Wang, Y., Qiu, Y.-Z., Zhang, L., Wang, J., and Long, M. Timesiam: A pre-training framework for siamese time-series modeling. In Forty-first International Conference on Machine Learning, 2024.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.   
Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S., Nguyen, N., Gifford, W. M., Reddy, C., and Kalagnanam, J. Tiny time mixers (ttms): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series. Advances in Neural Information Processing Systems, 37:74147–74181, 2024.   
Feng, C., Huang, L., and Krompass, D. Only the curve shape matters: Training foundation models for zero-shot multivariate time series forecasting through next curve shape prediction. arXiv preprint arXiv:2402.07570, 2024.

Fu, F., Chen, J., Zhang, J., Yang, C., Ma, L., and Yang, Y. Are synthetic time-series data really not as good as real data?, 2024. URL https://arxiv.org/abs/2402.00607.   
Godahewa, R. W., Bergmeir, C., Webb, G. I., Hyndman, R., and Montero-Manso, P. Monash time series forecasting archive. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=wEc1mgAjU-.   
Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., and Dubrawski, A. Moment: A family of open time-series foundation models. In Forty-first International Conference on Machine Learning, 2024.   
Gruver, N., Finzi, M., Qiu, S., and Wilson, A. G. Large language models are zero-shot time series forecasters. Advances in Neural Information Processing Systems, 36, 2023.   
Han, L., Ye, H.-J., and Zhan, D.-C. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. IEEE Transactions on Knowledge & Data Engineering, (01): 1–14, 2024.   
Hatami, N., Gavet, Y., and Debayle, J. Classification of time-series images using deep convolutional neural networks. In Tenth international conference on machine vision (ICMV 2017), volume 10696, pp. 242–249. SPIE, 2018.   
He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.   
Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., and Mohamed, A. HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021.   
Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., et al. Time-llm: Time series forecasting by reprogramming large language models. In The Twelfth International Conference on Learning Representations, 2024.   
Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=cGDAkQo1C0p.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.   
Li, X., Kang, Y., and Li, F. Forecasting with time series imaging. Expert Systems with Applications, 160:113680, 2020.   
Li, Z., Li, S., and Yan, X. Time series as images: Vision transformer for irregularly sampled time series. Advances in Neural Information Processing Systems, 36, 2024.   
Lin, S., Lin, W., Wu, W., Chen, H., and Yang, J. Sparsetsf: Modeling long-term time series forecasting with 1k parameters. In Forty-first International Conference on Machine Learning, 2024.   
Liu, P., Guo, H., Dai, T., Li, N., Bao, J., Ren, X., Jiang, Y., and Xia, S.-T. Calf: Aligning llms for time series forecasting via cross-modal fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 18915–18923, 2025.   
Liu, Y., Wu, H., Wang, J., and Long, M. Non-stationary transformers: Exploring the stationarity in time series forecasting, 2022.   
Liu, Y., Zhang, H., Li, C., Huang, X., Wang, J., and Long, M. Timer: Generative pre-trained transformers are large time series models. In Forty-first International Conference on Machine Learning, 2024.   
Ma, Q., Liu, Z., Zheng, Z., Huang, Z., Zhu, S., Yu, Z., and Kwok, J. T. A survey on time-series pre-trained models. arXiv preprint arXiv:2305.10716, 2023.   
Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2022.   
Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.   
Qiu, X., Hu, J., Zhou, L., Wu, X., Du, J., Zhang, B., Guo, C., Zhou, A., Jensen, C. S., Sheng, Z., and Yang, B. TFB: towards comprehensive and fair benchmarking of time series forecasting methods. Proc. VLDB Endow., 17(9):2363–2377, 2024.   
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.   
Schick, T. and Schütze, H. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 255–269, 2021.   
Semenoglou, A.-A., Spiliotis, E., and Assimakopoulos, V. Image-based time series forecasting: A deep convolutional neural network approach. Neural Networks, 157:39–53, 2023.   
Shi, X., Wang, S., Nie, Y., Li, D., Ye, Z., Wen, Q., and Jin, M. Time-moe: Billion-scale time series foundation models with mixture of experts. arXiv preprint arXiv:2409.16040, 2024.   
Sood, S., Zeng, Z., Cohen, N., Balch, T., and Veloso, M. Visual time series forecasting: an image-driven approach. In Proceedings of the Second ACM International Conference on AI in Finance, pp. 1–9, 2021.   
Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., and Lempitsky, V. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2149–2159, 2022.   
Tan, M., Merrill, M. A., Gupta, V., Althoff, T., and Hartvigsen, T. Are language models actually useful for time series forecasting? arXiv preprint arXiv:2406.16964, 2024.   
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., Li, T., and Pan, G. CBramod: A criss-cross brain foundation model for EEG decoding. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=NPNUHgHF2w.   
Wang, Z. and Oates, T. Imaging time-series to improve classification and imputation. In Proceedings of the 24th International Conference on Artificial Intelligence, pp. 3939–3945, 2015a.   
Wang, Z. and Oates, T. Spatially encoding temporal correlations to classify temporal data using convolutional neural networks. arXiv preprint arXiv:1509.07481, 2015b.

Wimmer, C. and Rekabsaz, N. Leveraging vision-language models for granular market change prediction. arXiv preprint arXiv:2301.10166, 2023.   
Woo, G., Liu, C., Sahoo, D., Kumar, A., and Hoi, S. CoST: Contrastive learning of disentangled seasonal-trend representations for time series forecasting. In International Conference on Learning Representations, 2022a. URL https://openreview.net/forum?id=PilZY3omXV2.   
Woo, G., Liu, C., Sahoo, D., Kumar, A., and Hoi, S. C. H. Etsformer: Exponential smoothing transformers for time-series forecasting. CoRR, abs/2202.01381, 2022b. URL https://arxiv.org/abs/2202.01381.   
Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. Unified training of universal time series forecasting transformers. In Forty-first International Conference on Machine Learning, 2024.   
Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34:22419–22430, 2021.   
Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. Timesnet: Temporal 2d-variation modeling for general time series analysis. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=ju\_Uqw3840q.   
Xue, H. and Salim, F. D. Promptcast: A new prompt-based learning paradigm for time series forecasting. IEEE Transactions on Knowledge and Data Engineering, 2023.   
Yang, L., Wang, Y., Fan, X., Cohen, I., Zhao, Y., and Zhang, Z. Vitime: A visual intelligence-based foundation model for time series forecasting. arXiv preprint arXiv:2407.07311, 2024.   
Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., and Xu, B. Ts2vec: Towards universal representation of time series. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8980–8987, 2022.   
Zaken, E. B., Goldberg, Y., and Ravfogel, S. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1–9, 2022.   
Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023.

Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., and Eickhoff, C. A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pp. 2114–2124, 2021.   
Zhang, K., Wen, Q., Zhang, C., Cai, R., Jin, M., Liu, Y., Zhang, J. Y., Liang, Y., Pang, G., Song, D., et al. Self-supervised learning for time series analysis: Taxonomy, progress, and prospects. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.   
Zhang, Y., Zhang, Y., Zheng, M., Chen, K., Gao, C., Ge, R., Teng, S., Jelloul, A., Rao, J., Guo, X., et al. Insight miner: A time series analysis dataset for cross-domain alignment with natural language. In NeurIPS 2023 AI for Science Workshop, 2023.   
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 11106–11115, 2021.   
Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., and Jin, R. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International Conference on Machine Learning, pp. 27268–27286. PMLR, 2022.   
Zhou, T., Niu, P., Sun, L., Jin, R., et al. One fits all: Power general time series analysis by pretrained lm. In Advances in neural information processing systems, volume 36, pp. 43322–43355, 2023.

# Appendix

# A. Details of Experiments

# A.1. Benchmark and baselines

Long-Term TSF Benchmark We evaluate our model on 8 widely used long-term TSF datasets (Zhou et al., 2021; Wu et al., 2021), including ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Illness, and Weather. Performance is assessed using Mean Squared Error (MSE) and Mean Absolute Error (MAE), with lower values indicating better forecasting accuracy.

Monash Benchmark Following Woo et al. (2024), we tested 29 Monash datasets (Godahewa et al., 2021) using GluonTS (Alexandrov et al., 2020), including M1 Monthly, M3 Monthly, M3 Other, M4 Monthly, M4 Weekly, M4 Daily, M4 Hourly, Tourism Quarterly, Tourism Monthly, CIF 2016, Australian Electricity Demand, Bitcoin, Pedestrian Counts, Vehicle Trips, KDD Cup, Weather, NN5 Daily, NN5 Weekly, Carparts, FRED-MD, Traffic Hourly, Traffic Weekly, Rideshare, Hospital, COVID Deaths, Temperature Rain, Sunspot, Saugeen River Flow, and US Births. Performance is assessed using MAE.

GIFT-Eval Benchmark Aksu et al. (2024) introduces the General Time Series Forecasting Model Evaluation, GIFT-Eval, encompasses 23 datasets over 144,000 time series and 177 million data points, spanning seven domains, 10 frequencies, multivariate inputs, and prediction lengths ranging from short to long-term forecasts. We use a constant context length 2,000 for VISIONTS and we report the point forecast performance using MAPE.

Baselines We select representative baselines for comparison, including TS-based and Text-based foundation models, and other popular TSF baselines covering both Transformer-based, MLP-based and CNN-based architectures. The baseline models selected for comparison are briefly described below:

1. MOIRAI (Woo et al., 2024) is a TSF foundation model trained on the Large-scale Open Time Series Archive (LOTSA), with over 27B observations across nine domains. It has three variants: small, base, and large.   
2. TimesFM (Das et al., 2024) is a decoder-style TSF foundation model, using a large time-series corpus comprising both real-world and synthetic datasets.   
3. Time-LLM (Jin et al., 2024) is a text-based TSF foundation model built on Llama, which reprograms time series data to align with the language modality, keeping the LLM frozen.   
4. GPT4TS (Zhou et al., 2023) (OneFitsAll) is another text-based model based on GPT, fine-tuned for forecasting tasks.   
5. LLMTime (Gruver et al., 2023) encodes time series data to a text sequence, supporting zero-shot forecasting.   
6. DLinear (Zeng et al., 2023) proposes a linear forecasting model, enhanced by seasonal-trend decomposition or normalization.   
7. PatchTST (Nie et al., 2022) uses Transformer encoders with patching and channel independence techniques for improved predictions.   
8. TimesNet (Wu et al., 2023) applies convolution kernels along the time dimension, using temporal decomposition and periodical segmentation to capture temporal patterns.   
9. FEDformer (Zhou et al., 2022) employs a sparse frequency domain representation, using frequency-enhanced blocks for cross-time dependency.   
10. Autoformer (Wu et al., 2021) uses series decomposition blocks and Auto-Correlation to capture cross-time dependency.   
11. Stationary (Liu et al., 2022) introduces stationarization and de-stationary attention mechanisms.   
12. ETSFormer (Woo et al., 2022b) leverages exponential smoothing principles, including exponential smoothing and frequency attention mechanisms.   
13. Informer (Zhou et al., 2021) proposes ProbSparse self-attention and distillation operations.

Table 5. Periodicity $(P)$ search range for the sampling frequency. $x$ denotes the number of sampling frequencies. For example, for data with a sampling frequency of 2 minutes (2T), we have $x = 2$ , and the possible search range of $P$ is $\{^{1440} / x, ^{10080} / x, 1\} = \{720, 5040, 1\}$ . 

<table><tr><td>Sampling Frequency</td><td>Possible Seasonalities</td><td>Possible P</td></tr><tr><td>Second (S)</td><td>1 hour</td><td> $\{3600/x, 1\}$ </td></tr><tr><td>Minute (T)</td><td>1 day or 1 week</td><td> $\{1440/x, 10080/x, 1\}$ </td></tr><tr><td>Hour (H)</td><td>1 day or 1 week</td><td> $\{24/x, 168/x, 1\}$ </td></tr><tr><td>Day (D)</td><td>1 week, 1 month, or 1 year</td><td> $\{7/x, 30/x, 365/x, 1\}$ </td></tr><tr><td>Week (W)</td><td>1 year or 1 month</td><td> $\{52/x, 4/x, 1\}$ </td></tr><tr><td>Month (M)</td><td>1 year, 6 months, or 3 months</td><td> $\{12/x, 6/x, 3/x, 1\}$ </td></tr><tr><td>Business Day (B)</td><td>1 week</td><td> $\{5/x, 1\}$ </td></tr><tr><td>Quarter (Q)</td><td>1 year or 6 months</td><td> $\{4/x, 2/x, 1\}$ </td></tr><tr><td>Others</td><td>-</td><td> $\{1\}$ </td></tr></table>

Table 6. Final P used for each dataset in our experiment. 

<table><tr><td></td><td>Frequency</td><td>P</td><td>Datasets</td><td></td><td></td><td></td></tr><tr><td rowspan="4">Long-Term TSF</td><td>H</td><td>24</td><td>ETTh1</td><td>ETTh2</td><td>Electricity</td><td>Traffic</td></tr><tr><td>W</td><td>52</td><td>Illness</td><td></td><td></td><td></td></tr><tr><td>15T</td><td>96</td><td>ETTm1</td><td>ETTm2</td><td></td><td></td></tr><tr><td>10T</td><td>144</td><td>Weather</td><td></td><td></td><td></td></tr><tr><td rowspan="15">Monash</td><td>D</td><td>1</td><td>M4 Daily</td><td>COVID Deaths</td><td></td><td></td></tr><tr><td>W</td><td>1</td><td>NN5 Weekly</td><td></td><td></td><td></td></tr><tr><td>M</td><td>1</td><td>FRED-MD</td><td></td><td></td><td></td></tr><tr><td>Q</td><td>1</td><td>M3 Other</td><td></td><td></td><td></td></tr><tr><td>M</td><td>3</td><td>M3 Monthly</td><td>M4 Monthly</td><td>CIF 2016 (6)</td><td></td></tr><tr><td>W</td><td>4</td><td>M4 Weekly</td><td>Traffic Weekly</td><td></td><td></td></tr><tr><td>Q</td><td>4</td><td>Tourism Quarterly</td><td></td><td></td><td></td></tr><tr><td>M</td><td>6</td><td>CIF 2016 (12)</td><td>Car Parts</td><td></td><td></td></tr><tr><td>D</td><td>7</td><td>Bitcoin</td><td>Vehicle Trips</td><td>Weather</td><td>NN5 Daily</td></tr><tr><td>D</td><td>7</td><td>US Births</td><td>Saugeen Day</td><td>Temperature Rain</td><td></td></tr><tr><td>M</td><td>12</td><td>Tourism Monthly</td><td>Hospital</td><td>M1 Monthly</td><td></td></tr><tr><td>H</td><td>24</td><td>M4 Hourly</td><td>KDD cup</td><td>Pedestrian Counts</td><td></td></tr><tr><td>H</td><td>24</td><td>Traffic Hourly</td><td>Rideshare</td><td></td><td></td></tr><tr><td>D</td><td>30</td><td>Sunspot</td><td></td><td></td><td></td></tr><tr><td>0.5H</td><td>336</td><td>Aus. Elec. Demand</td><td></td><td></td><td></td></tr></table>

Table 7. Comparison of setting P = 1 for VISIONTS. 

<table><tr><td rowspan="2"></td><td colspan="2">VISIONTS</td><td colspan="2">P=1</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.390</td><td>0.414</td><td>0.840</td><td>0.628</td></tr><tr><td>ETTh2</td><td>0.333</td><td>0.375</td><td>0.424</td><td>0.445</td></tr><tr><td>ETTm1</td><td>0.374</td><td>0.372</td><td>0.660</td><td>0.533</td></tr><tr><td>ETTm2</td><td>0.282</td><td>0.321</td><td>0.312</td><td>0.363</td></tr><tr><td>Average</td><td>0.344</td><td>0.370</td><td>0.559</td><td>0.492</td></tr></table>

For the long-term TSF benchmark, we include TS-based foundation model results from their original papers, Text-based model results from Tan et al. (2024), and other baseline results from Zhou et al. (2023). For the Monash and PF benchmark, we include results from Woo et al. (2024).

Environment All experiments are conducted using Time-Series-Library (https://github.com/thuml/Time-Series-Library) and GluonTS library (Alexandrov et al., 2020) on an NVIDIA A800 GPU.

# A.2. Periodicity selection

We first determine a range of period lengths based on the sampling frequency of the data, shown in Table 5. This frequency-based strategy is also employed by Alexandrov et al. (2020) while we extend the search range for tuning. We select the optimal P from this range on the validation set. The final P used in our experiments are summarized in Table 6.

To demonstrate the influence of P and the effectiveness of our periodicity selection strategy, we set P = 1 and compare the results with the above strategy. Table 7 shows that such strategy (denoted as VISIONTS) significantly outperforms the naive strategy that sets P = 1.

# B. Zero-Shot Forecasting

# B.1. Hyperparameters

Table 8. Hyperparameters for VISIONTS used in our zero-shot forecasting (Long-term TSF). 

<table><tr><td></td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Weather</td><td>Electricity</td></tr><tr><td>Normalization constant r</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td>Alignment constant c</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td>Context length L</td><td>2880</td><td>1728</td><td>2304</td><td>4032</td><td>4032</td><td>2880</td></tr></table>

We conduct hyperparameter tuning on validation sets to determine the optimal context length L. Final used hyperparameters are summarized in Table 8.

# B.2. Full forecasting results of the long-term TSF benchmark

Table 9 shows the full results of zero-shot/few-shot long-term forecasting performance. VISIONTS achieves the best results in most cases (32 out of 62), outperforming MOIRAI $_{Base}$ (10 out of 62) and MOIRAI $_{Large}$ (8 out of 62).

# B.3. Comparison of TimesFM and LLMTime

Due to the step-by-step output of the decoder architecture, the efficiency of TimesFM (Das et al., 2024) and LLMTime (Gruver et al., 2023) are relatively slower. Thus, Das et al. (2024) only reported results for the last test window of the original split. We compared VISIONTS with their results under the same setting, as shown in Table 10. VISIONTS outperforms TimesFM and LLMTime in terms of MAE, indicating that image-based TSF models are on par with or even better than TS-based and text-based models.

# B.4. Comparison of traditional methods

In addition to deep learning models, we also compare traditional methods, including ARIMA, ETS, and two methods that require periodicity as our VISIONTS: Seasonal Naïve (repeating the last period) and Seasonal Avg (similar to Seasonal Naïve but repeating the average of all periods in the look-back window). Due to the high computational cost of ARIMA and ETS, we only compare them on the small-scale benchmarks, i.e., four ETT datasets. Table 12 shows that VISIONTS also achieves the best performance.

# B.5. Comparison of concurrent works

We compare our work with other concurrent TSF methods. Table 13 presents the comparison from Time-MoE (Shi et al., 2024) and TTM (Ekambaram et al., 2024), and Table 14 shows the comparison with CALF (Liu et al., 2025), which is the existing SOTA LLMs-based time series forecasting work. These findings highlight the promising potential of vision models in TSF scenarios.

# B.6. Full forecasting results of the Monash TSF benchmark

Setup Table 6 lists the sampling frequency and the selected period P for each dataset. Datasets with P = 1 indicate no significant periodicity, where we use a context length of L = 300. For other datasets with P > 1, we select a longer context

Table 9. Full results of Table 1: Zero-shot or few-shot results on the long-term TSF benchmark. Bold: the best result. 

<table><tr><td rowspan="4" colspan="2">Pretrain Method Metric</td><td colspan="8">Zero-Shot</td><td colspan="14">Few-Shot (10% Downstream Dataset)</td></tr><tr><td colspan="2">Images</td><td colspan="6">Time-series</td><td colspan="4">Text</td><td colspan="10">No Pretrain</td></tr><tr><td colspan="2">VISIONTIME</td><td colspan="2">MOIRAI $_S$ </td><td colspan="2">MOIRAI $_B$ </td><td colspan="2">MOIRAI $_L$ </td><td colspan="2">TimeLLM</td><td colspan="2">GPT4TS</td><td colspan="2">DLinear</td><td colspan="2">PatchTST</td><td colspan="2">TimesNet</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.353</td><td>0.383</td><td>0.375</td><td>0.402</td><td>0.384</td><td>0.402</td><td>0.380</td><td>0.398</td><td>0.448</td><td>0.460</td><td>0.458</td><td>0.456</td><td>0.492</td><td>0.495</td><td>0.516</td><td>0.485</td><td>0.861</td><td>0.628</td><td>0.613</td><td>0.552</td><td>1.179</td><td>0.792</td></tr><tr><td>192</td><td>0.392</td><td>0.410</td><td>0.399</td><td>0.419</td><td>0.425</td><td>0.429</td><td>0.440</td><td>0.434</td><td>0.484</td><td>0.483</td><td>0.570</td><td>0.516</td><td>0.565</td><td>0.538</td><td>0.598</td><td>0.524</td><td>0.797</td><td>0.593</td><td>0.722</td><td>0.598</td><td>1.199</td><td>0.806</td></tr><tr><td>336</td><td>0.407</td><td>0.423</td><td>0.412</td><td>0.429</td><td>0.456</td><td>0.450</td><td>0.514</td><td>0.474</td><td>0.589</td><td>0.540</td><td>0.608</td><td>0.535</td><td>0.721</td><td>0.622</td><td>0.657</td><td>0.550</td><td>0.941</td><td>0.648</td><td>0.750</td><td>0.619</td><td>1.202</td><td>0.811</td></tr><tr><td>720</td><td>0.406</td><td>0.441</td><td>0.413</td><td>0.444</td><td>0.470</td><td>0.473</td><td>0.705</td><td>0.568</td><td>0.700</td><td>0.604</td><td>0.725</td><td>0.591</td><td>0.986</td><td>0.743</td><td>0.762</td><td>0.610</td><td>0.877</td><td>0.641</td><td>0.721</td><td>0.616</td><td>1.217</td><td>0.825</td></tr><tr><td>avg</td><td>0.390</td><td>0.414</td><td>0.400</td><td>0.424</td><td>0.434</td><td>0.439</td><td>0.510</td><td>0.469</td><td>0.556</td><td>0.522</td><td>0.590</td><td>0.525</td><td>0.691</td><td>0.600</td><td>0.633</td><td>0.542</td><td>0.869</td><td>0.628</td><td>0.702</td><td>0.596</td><td>1.199</td><td>0.809</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.271</td><td>0.328</td><td>0.281</td><td>0.334</td><td>0.277</td><td>0.327</td><td>0.287</td><td>0.325</td><td>0.275</td><td>0.326</td><td>0.331</td><td>0.374</td><td>0.357</td><td>0.411</td><td>0.353</td><td>0.389</td><td>0.378</td><td>0.409</td><td>0.413</td><td>0.451</td><td>3.837</td><td>1.508</td></tr><tr><td>192</td><td>0.328</td><td>0.367</td><td>0.340</td><td>0.373</td><td>0.340</td><td>0.374</td><td>0.347</td><td>0.367</td><td>0.374</td><td>0.373</td><td>0.402</td><td>0.411</td><td>0.569</td><td>0.519</td><td>0.403</td><td>0.414</td><td>0.490</td><td>0.467</td><td>0.474</td><td>0.477</td><td>3.856</td><td>1.513</td></tr><tr><td>336</td><td>0.345</td><td>0.381</td><td>0.362</td><td>0.393</td><td>0.371</td><td>0.401</td><td>0.377</td><td>0.393</td><td>0.406</td><td>0.429</td><td>0.406</td><td>0.433</td><td>0.671</td><td>0.572</td><td>0.426</td><td>0.441</td><td>0.537</td><td>0.494</td><td>0.547</td><td>0.543</td><td>3.952</td><td>1.526</td></tr><tr><td>720</td><td>0.388</td><td>0.422</td><td>0.380</td><td>0.416</td><td>0.394</td><td>0.426</td><td>0.404</td><td>0.421</td><td>0.427</td><td>0.449</td><td>0.449</td><td>0.464</td><td>0.824</td><td>0.648</td><td>0.477</td><td>0.480</td><td>0.510</td><td>0.491</td><td>0.516</td><td>0.523</td><td>3.842</td><td>1.503</td></tr><tr><td>avg</td><td>0.333</td><td>0.375</td><td>0.341</td><td>0.379</td><td>0.346</td><td>0.382</td><td>0.354</td><td>0.377</td><td>0.370</td><td>0.394</td><td>0.397</td><td>0.421</td><td>0.605</td><td>0.538</td><td>0.415</td><td>0.431</td><td>0.479</td><td>0.465</td><td>0.488</td><td>0.499</td><td>3.872</td><td>1.513</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.341</td><td>0.347</td><td>0.404</td><td>0.383</td><td>0.335</td><td>0.360</td><td>0.353</td><td>0.363</td><td>0.346</td><td>0.388</td><td>0.390</td><td>0.404</td><td>0.352</td><td>0.392</td><td>0.410</td><td>0.419</td><td>0.583</td><td>0.501</td><td>0.774</td><td>0.614</td><td>1.162</td><td>0.785</td></tr><tr><td>192</td><td>0.360</td><td>0.360</td><td>0.435</td><td>0.402</td><td>0.366</td><td>0.379</td><td>0.376</td><td>0.380</td><td>0.373</td><td>0.416</td><td>0.429</td><td>0.423</td><td>0.382</td><td>0.412</td><td>0.437</td><td>0.434</td><td>0.630</td><td>0.528</td><td>0.754</td><td>0.592</td><td>1.172</td><td>0.793</td></tr><tr><td>336</td><td>0.377</td><td>0.374</td><td>0.462</td><td>0.416</td><td>0.391</td><td>0.394</td><td>0.399</td><td>0.395</td><td>0.413</td><td>0.426</td><td>0.469</td><td>0.439</td><td>0.419</td><td>0.434</td><td>0.476</td><td>0.454</td><td>0.725</td><td>0.568</td><td>0.869</td><td>0.677</td><td>1.227</td><td>0.908</td></tr><tr><td>720</td><td>0.416</td><td>0.405</td><td>0.490</td><td>0.437</td><td>0.434</td><td>0.419</td><td>0.432</td><td>0.417</td><td>0.485</td><td>0.476</td><td>0.569</td><td>0.498</td><td>0.490</td><td>0.477</td><td>0.681</td><td>0.556</td><td>0.769</td><td>0.549</td><td>0.810</td><td>0.630</td><td>1.207</td><td>0.797</td></tr><tr><td>avg</td><td>0.374</td><td>0.372</td><td>0.448</td><td>0.410</td><td>0.382</td><td>0.388</td><td>0.390</td><td>0.389</td><td>0.404</td><td>0.427</td><td>0.464</td><td>0.441</td><td>0.411</td><td>0.429</td><td>0.501</td><td>0.466</td><td>0.677</td><td>0.537</td><td>0.802</td><td>0.628</td><td>1.192</td><td>0.821</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.228</td><td>0.282</td><td>0.205</td><td>0.282</td><td>0.195</td><td>0.269</td><td>0.189</td><td>0.260</td><td>0.177</td><td>0.261</td><td>0.188</td><td>0.269</td><td>0.213</td><td>0.303</td><td>0.191</td><td>0.274</td><td>0.212</td><td>0.285</td><td>0.352</td><td>0.454</td><td>3.203</td><td>1.407</td></tr><tr><td>192</td><td>0.262</td><td>0.305</td><td>0.261</td><td>0.318</td><td>0.247</td><td>0.303</td><td>0.247</td><td>0.300</td><td>0.241</td><td>0.314</td><td>0.251</td><td>0.309</td><td>0.278</td><td>0.345</td><td>0.252</td><td>0.317</td><td>0.270</td><td>0.323</td><td>0.694</td><td>0.691</td><td>3.112</td><td>1.387</td></tr><tr><td>336</td><td>0.293</td><td>0.328</td><td>0.319</td><td>0.355</td><td>0.291</td><td>0.333</td><td>0.295</td><td>0.334</td><td>0.274</td><td>0.327</td><td>0.307</td><td>0.346</td><td>0.338</td><td>0.385</td><td>0.306</td><td>0.353</td><td>0.323</td><td>0.353</td><td>2.408</td><td>1.407</td><td>3.255</td><td>1.421</td></tr><tr><td>720</td><td>0.343</td><td>0.370</td><td>0.415</td><td>0.410</td><td>0.355</td><td>0.377</td><td>0.372</td><td>0.386</td><td>0.417</td><td>0.390</td><td>0.426</td><td>0.417</td><td>0.436</td><td>0.440</td><td>0.433</td><td>0.427</td><td>0.474</td><td>0.449</td><td>1.913</td><td>1.166</td><td>3.909</td><td>1.543</td></tr><tr><td>avg</td><td>0.282</td><td>0.321</td><td>0.300</td><td>0.341</td><td>0.272</td><td>0.321</td><td>0.276</td><td>0.320</td><td>0.277</td><td>0.323</td><td>0.293</td><td>0.335</td><td>0.316</td><td>0.368</td><td>0.296</td><td>0.343</td><td>0.320</td><td>0.353</td><td>1.342</td><td>0.930</td><td>3.370</td><td>1.440</td></tr><tr><td rowspan="5">Electricity</td><td>96</td><td>0.177</td><td>0.266</td><td>0.205</td><td>0.299</td><td>0.158</td><td>0.248</td><td>0.152</td><td>0.242</td><td>0.139</td><td>0.241</td><td>0.139</td><td>0.237</td><td>0.150</td><td>0.253</td><td>0.140</td><td>0.238</td><td>0.299</td><td>0.373</td><td>0.261</td><td>0.348</td><td>1.259</td><td>0.919</td></tr><tr><td>192</td><td>0.188</td><td>0.277</td><td>0.220</td><td>0.310</td><td>0.174</td><td>0.263</td><td>0.171</td><td>0.259</td><td>0.151</td><td>0.248</td><td>0.156</td><td>0.252</td><td>0.164</td><td>0.264</td><td>0.160</td><td>0.255</td><td>0.305</td><td>0.379</td><td>0.338</td><td>0.406</td><td>1.160</td><td>0.873</td></tr><tr><td>336</td><td>0.207</td><td>0.296</td><td>0.236</td><td>0.323</td><td>0.191</td><td>0.278</td><td>0.192</td><td>0.278</td><td>0.169</td><td>0.270</td><td>0.175</td><td>0.270</td><td>0.181</td><td>0.282</td><td>0.180</td><td>0.276</td><td>0.319</td><td>0.391</td><td>0.410</td><td>0.474</td><td>1.157</td><td>0.872</td></tr><tr><td>720</td><td>0.256</td><td>0.337</td><td>0.270</td><td>0.347</td><td>0.229</td><td>0.307</td><td>0.236</td><td>0.313</td><td>0.240</td><td>0.322</td><td>0.233</td><td>0.317</td><td>0.223</td><td>0.321</td><td>0.241</td><td>0.323</td><td>0.369</td><td>0.426</td><td>0.715</td><td>0.685</td><td>1.203</td><td>0.898</td></tr><tr><td>avg</td><td>0.207</td><td>0.294</td><td>0.233</td><td>0.320</td><td>0.188</td><td>0.274</td><td>0.188</td><td>0.273</td><td>0.175</td><td>0.270</td><td>0.176</td><td>0.269</td><td>0.180</td><td>0.280</td><td>0.180</td><td>0.273</td><td>0.323</td><td>0.392</td><td>0.431</td><td>0.478</td><td>1.195</td><td>0.891</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.220</td><td>0.257</td><td>0.173</td><td>0.212</td><td>0.167</td><td>0.203</td><td>0.177</td><td>0.208</td><td>0.161</td><td>0.210</td><td>0.163</td><td>0.215</td><td>0.171</td><td>0.224</td><td>0.165</td><td>0.215</td><td>0.184</td><td>0.230</td><td>0.221</td><td>0.297</td><td>0.374</td><td>0.401</td></tr><tr><td>192</td><td>0.244</td><td>0.275</td><td>0.216</td><td>0.250</td><td>0.209</td><td>0.241</td><td>0.219</td><td>0.249</td><td>0.204</td><td>0.248</td><td>0.210</td><td>0.254</td><td>0.215</td><td>0.263</td><td>0.210</td><td>0.257</td><td>0.245</td><td>0.283</td><td>0.270</td><td>0.322</td><td>0.552</td><td>0.478</td></tr><tr><td>336</td><td>0.280</td><td>0.299</td><td>0.260</td><td>0.282</td><td>0.256</td><td>0.276</td><td>0.277</td><td>0.292</td><td>0.261</td><td>0.302</td><td>0.256</td><td>0.292</td><td>0.258</td><td>0.299</td><td>0.259</td><td>0.297</td><td>0.305</td><td>0.321</td><td>0.320</td><td>0.351</td><td>0.724</td><td>0.541</td></tr><tr><td>720</td><td>0.330</td><td>0.337</td><td>0.320</td><td>0.322</td><td>0.321</td><td>0.323</td><td>0.365</td><td>0.350</td><td>0.309</td><td>0.332</td><td>0.321</td><td>0.339</td><td>0.320</td><td>0.346</td><td>0.332</td><td>0.346</td><td>0.381</td><td>0.371</td><td>0.390</td><td>0.396</td><td>0.739</td><td>0.558</td></tr><tr><td>avg</td><td>0.269</td><td>0.292</td><td>0.242</td><td>0.267</td><td>0.238</td><td>0.261</td><td>0.260</td><td>0.275</td><td>0.234</td><td>0.273</td><td>0.238</td><td>0.275</td><td>0.241</td><td>0.283</td><td>0.242</td><td>0.279</td><td>0.279</td><td>0.301</td><td>0.300</td><td>0.342</td><td>0.597</td><td>0.495</td></tr><tr><td colspan="2">Average 1st count</td><td colspan="2">0.309 32</td><td colspan="2">0.327 0</td><td colspan="2">0.310 10</td><td colspan="2">0.310 8</td><td colspan="2">0.336 10</td><td colspan="2">0.368 6</td><td colspan="2">0.360 6</td><td colspan="2">0.378 0</td><td colspan="2">0.407 0</td><td colspan="2">0.416 0</td><td colspan="2">0.478 0</td></tr></table>

length of L = 1000. All datasets were tested with the hyperparameters r = c = 0.4 as we had done for the long-term TSF benchmark.

Results Table 15 presents VISIONTS's MAE test results, with the normalized MAE calculated by dividing each dataset's MAE by the naive forecast's MAE and aggregated using the geometric mean across datasets. We include the result of each baseline from Woo et al. (2024). Particularly, we find that VISIONTS outperforms MOIRAI on some datasets with $P = 1$ (e.g., FRED-MD and NN5 Weekly), showing that VISIONTS can still work effectively without significant periodicity.

# B.7. Impact of backbones

Table 17 compares zero-shot forecasting performance of three MAE variants (112M, 330M, and 657M), showing that the three variants are similar, but larger models show a slight decrease. Particularly, the smallest model excels in ETTh2, ETTm1, ETTm2, and Weather, while the largest model excels in Electricity. Additionally, Table 16 compares VISIONTS with another visual backbone, LaMa.

# B.8. Impact of the different image encoding strategies

Table 18 summarizes the impact of interpolation strategies and image orientations in the Alignment step. It shows that the smoother Bilinear and Bicubic interpolation perform similarly, both significantly better than the rougher Nearest Neighbor. This suggests that smooth resizing effectively handles time series interpolation. Moreover, image orientation has little impact on performance.

Table 10. MAE results of TimesFM and LLM-Time for zero-shot forecasting, on the last test window of the original test split. 

<table><tr><td colspan="2">Method</td><td>VISIONTS</td><td>TimesFM</td><td>LLMTime</td></tr><tr><td rowspan="2">ETTh1</td><td>96</td><td>0.35</td><td>0.45</td><td>0.42</td></tr><tr><td>192</td><td>0.45</td><td>0.53</td><td>0.50</td></tr><tr><td rowspan="2">ETTh2</td><td>96</td><td>0.24</td><td>0.35</td><td>0.33</td></tr><tr><td>192</td><td>0.60</td><td>0.62</td><td>0.70</td></tr><tr><td rowspan="2">ETTm1</td><td>96</td><td>0.12</td><td>0.19</td><td>0.37</td></tr><tr><td>192</td><td>0.23</td><td>0.26</td><td>0.71</td></tr><tr><td rowspan="2">ETTm2</td><td>96</td><td>0.19</td><td>0.24</td><td>0.29</td></tr><tr><td>192</td><td>0.24</td><td>0.27</td><td>0.31</td></tr><tr><td colspan="2">Average</td><td>0.30</td><td>0.36</td><td>0.45</td></tr></table>

Table 11. Comparison of traditional forecasting baselines in the zero-shot setting. 

<table><tr><td rowspan="2" colspan="2">MethodMetric</td><td colspan="2">VISIONTS</td><td colspan="2">ETS</td><td colspan="2">ARIMA</td><td colspan="2">Seasonal Naïve</td><td colspan="2">Seasonal Avg</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.353</td><td>0.383</td><td>1.289</td><td>0.710</td><td>0.900</td><td>0.719</td><td>0.512</td><td>0.433</td><td>0.589</td><td>0.585</td></tr><tr><td>192</td><td>0.392</td><td>0.410</td><td>1.319</td><td>0.730</td><td>0.906</td><td>0.724</td><td>0.581</td><td>0.469</td><td>0.598</td><td>0.590</td></tr><tr><td>336</td><td>0.407</td><td>0.423</td><td>1.324</td><td>0.742</td><td>0.908</td><td>0.731</td><td>0.650</td><td>0.501</td><td>0.610</td><td>0.597</td></tr><tr><td>720</td><td>0.406</td><td>0.441</td><td>1.329</td><td>0.751</td><td>0.932</td><td>0.753</td><td>0.655</td><td>0.514</td><td>0.656</td><td>0.624</td></tr><tr><td>avg</td><td>0.390</td><td>0.414</td><td>1.315</td><td>0.733</td><td>0.912</td><td>0.732</td><td>0.600</td><td>0.479</td><td>0.613</td><td>0.599</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.271</td><td>0.328</td><td>0.399</td><td>0.408</td><td>0.488</td><td>0.508</td><td>0.391</td><td>0.380</td><td>0.457</td><td>0.494</td></tr><tr><td>192</td><td>0.328</td><td>0.367</td><td>0.500</td><td>0.459</td><td>0.497</td><td>0.514</td><td>0.482</td><td>0.429</td><td>0.466</td><td>0.500</td></tr><tr><td>336</td><td>0.345</td><td>0.381</td><td>0.562</td><td>0.498</td><td>0.507</td><td>0.522</td><td>0.532</td><td>0.466</td><td>0.476</td><td>0.509</td></tr><tr><td>720</td><td>0.388</td><td>0.422</td><td>0.558</td><td>0.506</td><td>0.572</td><td>0.557</td><td>0.525</td><td>0.474</td><td>0.542</td><td>0.548</td></tr><tr><td>avg</td><td>0.333</td><td>0.375</td><td>0.505</td><td>0.468</td><td>0.516</td><td>0.525</td><td>0.483</td><td>0.437</td><td>0.485</td><td>0.513</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.341</td><td>0.347</td><td>1.204</td><td>0.659</td><td>0.702</td><td>0.568</td><td>0.423</td><td>0.387</td><td>0.369</td><td>0.399</td></tr><tr><td>192</td><td>0.360</td><td>0.360</td><td>1.251</td><td>0.685</td><td>0.704</td><td>0.570</td><td>0.463</td><td>0.406</td><td>0.374</td><td>0.402</td></tr><tr><td>336</td><td>0.377</td><td>0.374</td><td>1.276</td><td>0.702</td><td>0.709</td><td>0.574</td><td>0.496</td><td>0.426</td><td>0.382</td><td>0.407</td></tr><tr><td>720</td><td>0.416</td><td>0.405</td><td>1.311</td><td>0.724</td><td>0.713</td><td>0.580</td><td>0.574</td><td>0.464</td><td>0.394</td><td>0.416</td></tr><tr><td>avg</td><td>0.374</td><td>0.372</td><td>1.261</td><td>0.693</td><td>0.707</td><td>0.573</td><td>0.489</td><td>0.421</td><td>0.380</td><td>0.406</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.228</td><td>0.282</td><td>0.257</td><td>0.324</td><td>0.397</td><td>0.434</td><td>0.263</td><td>0.301</td><td>0.365</td><td>0.411</td></tr><tr><td>192</td><td>0.262</td><td>0.305</td><td>0.331</td><td>0.366</td><td>0.402</td><td>0.436</td><td>0.321</td><td>0.337</td><td>0.369</td><td>0.414</td></tr><tr><td>336</td><td>0.293</td><td>0.328</td><td>0.402</td><td>0.406</td><td>0.407</td><td>0.439</td><td>0.376</td><td>0.370</td><td>0.375</td><td>0.418</td></tr><tr><td>720</td><td>0.343</td><td>0.370</td><td>0.512</td><td>0.462</td><td>0.413</td><td>0.443</td><td>0.471</td><td>0.422</td><td>0.380</td><td>0.423</td></tr><tr><td>avg</td><td>0.282</td><td>0.321</td><td>0.376</td><td>0.390</td><td>0.405</td><td>0.438</td><td>0.358</td><td>0.357</td><td>0.372</td><td>0.417</td></tr><tr><td rowspan="2" colspan="2">Average $1^{st}$  count</td><td>0.344</td><td>0.370</td><td>0.864</td><td>0.571</td><td>0.635</td><td>0.567</td><td>0.482</td><td>0.424</td><td>0.463</td><td>0.484</td></tr><tr><td colspan="2">41</td><td colspan="2">0</td><td colspan="2">0</td><td colspan="2">0</td><td colspan="2">1</td></tr></table>

# B.9. Hyperparameter analysis

Figs. 8 to 10 show the influence of three hyperparameters, r, c, and L. We report the MSE averaged on four prediction lengths $\{96, 192, 336, 720\}$ .

Table 12. Comparison of traditional zero-shot forecasting baselines. 

<table><tr><td colspan="2">Method</td><td colspan="2">VISIONTS</td><td colspan="2">ETS</td><td colspan="2">ARIMA</td><td colspan="2">Seasonal Naïve</td><td colspan="2">Seasonal Avg</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.353</td><td>0.383</td><td>1.289</td><td>0.710</td><td>0.900</td><td>0.719</td><td>0.512</td><td>0.433</td><td>0.589</td><td>0.585</td></tr><tr><td>192</td><td>0.392</td><td>0.410</td><td>1.319</td><td>0.730</td><td>0.906</td><td>0.724</td><td>0.581</td><td>0.469</td><td>0.598</td><td>0.590</td></tr><tr><td>336</td><td>0.407</td><td>0.423</td><td>1.324</td><td>0.742</td><td>0.908</td><td>0.731</td><td>0.650</td><td>0.501</td><td>0.610</td><td>0.597</td></tr><tr><td>720</td><td>0.406</td><td>0.441</td><td>1.329</td><td>0.751</td><td>0.932</td><td>0.753</td><td>0.655</td><td>0.514</td><td>0.656</td><td>0.624</td></tr><tr><td>avg</td><td>0.390</td><td>0.414</td><td>1.315</td><td>0.733</td><td>0.912</td><td>0.732</td><td>0.600</td><td>0.479</td><td>0.613</td><td>0.599</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.271</td><td>0.328</td><td>0.399</td><td>0.408</td><td>0.488</td><td>0.508</td><td>0.391</td><td>0.380</td><td>0.457</td><td>0.494</td></tr><tr><td>192</td><td>0.328</td><td>0.367</td><td>0.500</td><td>0.459</td><td>0.497</td><td>0.514</td><td>0.482</td><td>0.429</td><td>0.466</td><td>0.500</td></tr><tr><td>336</td><td>0.345</td><td>0.381</td><td>0.562</td><td>0.498</td><td>0.507</td><td>0.522</td><td>0.532</td><td>0.466</td><td>0.476</td><td>0.509</td></tr><tr><td>720</td><td>0.388</td><td>0.422</td><td>0.558</td><td>0.506</td><td>0.572</td><td>0.557</td><td>0.525</td><td>0.474</td><td>0.542</td><td>0.548</td></tr><tr><td>avg</td><td>0.333</td><td>0.375</td><td>0.505</td><td>0.468</td><td>0.516</td><td>0.525</td><td>0.483</td><td>0.437</td><td>0.485</td><td>0.513</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.341</td><td>0.347</td><td>1.204</td><td>0.659</td><td>0.702</td><td>0.568</td><td>0.423</td><td>0.387</td><td>0.369</td><td>0.399</td></tr><tr><td>192</td><td>0.360</td><td>0.360</td><td>1.251</td><td>0.685</td><td>0.704</td><td>0.570</td><td>0.463</td><td>0.406</td><td>0.374</td><td>0.402</td></tr><tr><td>336</td><td>0.377</td><td>0.374</td><td>1.276</td><td>0.702</td><td>0.709</td><td>0.574</td><td>0.496</td><td>0.426</td><td>0.382</td><td>0.407</td></tr><tr><td>720</td><td>0.416</td><td>0.405</td><td>1.311</td><td>0.724</td><td>0.713</td><td>0.580</td><td>0.574</td><td>0.464</td><td>0.394</td><td>0.416</td></tr><tr><td>avg</td><td>0.374</td><td>0.372</td><td>1.261</td><td>0.693</td><td>0.707</td><td>0.573</td><td>0.489</td><td>0.421</td><td>0.380</td><td>0.406</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.228</td><td>0.282</td><td>0.257</td><td>0.324</td><td>0.397</td><td>0.434</td><td>0.263</td><td>0.301</td><td>0.365</td><td>0.411</td></tr><tr><td>192</td><td>0.262</td><td>0.305</td><td>0.331</td><td>0.366</td><td>0.402</td><td>0.436</td><td>0.321</td><td>0.337</td><td>0.369</td><td>0.414</td></tr><tr><td>336</td><td>0.293</td><td>0.328</td><td>0.402</td><td>0.406</td><td>0.407</td><td>0.439</td><td>0.376</td><td>0.370</td><td>0.375</td><td>0.418</td></tr><tr><td>720</td><td>0.343</td><td>0.370</td><td>0.512</td><td>0.462</td><td>0.413</td><td>0.443</td><td>0.471</td><td>0.422</td><td>0.380</td><td>0.423</td></tr><tr><td>avg</td><td>0.282</td><td>0.321</td><td>0.376</td><td>0.390</td><td>0.405</td><td>0.438</td><td>0.358</td><td>0.357</td><td>0.372</td><td>0.417</td></tr><tr><td rowspan="2" colspan="2">Average $1^{st}$  count</td><td>0.344</td><td>0.370</td><td>0.864</td><td>0.571</td><td>0.635</td><td>0.567</td><td>0.482</td><td>0.424</td><td>0.463</td><td>0.484</td></tr><tr><td colspan="2">41</td><td colspan="2">0</td><td colspan="2">0</td><td colspan="2">0</td><td colspan="2">1</td></tr></table>

Table 13. Comparison of Time-MoE and TTM in the zero-shot setting. We report the base and large model results for Time-MoE, as the ultra model weights are not yet released. For TTM, we used the official HuggingFace model for replication. The following table summarizes the performance of various zero-shot foundation models. 

<table><tr><td colspan="2"></td><td>VISIONTS</td><td>Time-MoE (base)</td><td>Time-MoE (large)</td><td>TTM (v1)</td></tr><tr><td rowspan="2">ETTh1</td><td>MSE</td><td>0.390</td><td>0.400</td><td>0.394</td><td>0.398</td></tr><tr><td>MAE</td><td>0.414</td><td>0.424</td><td>0.419</td><td>0.421</td></tr><tr><td rowspan="2">ETTh2</td><td>MSE</td><td>0.333</td><td>0.366</td><td>0.405</td><td>0.348</td></tr><tr><td>MAE</td><td>0.375</td><td>0.404</td><td>0.415</td><td>0.393</td></tr><tr><td rowspan="2">ETTm1</td><td>MSE</td><td>0.374</td><td>0.394</td><td>0.376</td><td>0.520</td></tr><tr><td>MAE</td><td>0.372</td><td>0.415</td><td>0.405</td><td>0.479</td></tr><tr><td rowspan="2">ETTm2</td><td>MSE</td><td>0.282</td><td>0.317</td><td>0.316</td><td>0.312</td></tr><tr><td>MAE</td><td>0.321</td><td>0.365</td><td>0.361</td><td>0.348</td></tr><tr><td rowspan="2">Electricity</td><td>MSE</td><td>0.207</td><td>(data leakage)</td><td>(data leakage)</td><td>0.201</td></tr><tr><td>MAE</td><td>0.294</td><td>(data leakage)</td><td>(data leakage)</td><td>0.293</td></tr><tr><td rowspan="2">Weather</td><td>MSE</td><td>0.269</td><td>0.265</td><td>0.270</td><td>0.234</td></tr><tr><td>MAE</td><td>0.292</td><td>0.297</td><td>0.300</td><td>0.266</td></tr><tr><td rowspan="2">Average</td><td>MSE</td><td>0.309</td><td>-</td><td>-</td><td>0.335</td></tr><tr><td>MAE</td><td>0.345</td><td>-</td><td>-</td><td>0.367</td></tr><tr><td colspan="2"> $1^{st}$  count</td><td>10</td><td>0</td><td>0</td><td>4</td></tr></table>

Table 14. Comparison of CALF in both zero-shot and full-shot settings. 

<table><tr><td colspan="2"></td><td>VISIONTS (zero-shot)</td><td>VISIONTS (full-shot)</td><td>CALF</td></tr><tr><td rowspan="2">ETTh1</td><td>MSE</td><td>0.390</td><td>0.395</td><td>0.432</td></tr><tr><td>MAE</td><td>0.414</td><td>0.409</td><td>0.428</td></tr><tr><td rowspan="2">ETTh2</td><td>MSE</td><td>0.333</td><td>0.336</td><td>0.349</td></tr><tr><td>MAE</td><td>0.375</td><td>0.382</td><td>0.382</td></tr><tr><td rowspan="2">ETTm1</td><td>MSE</td><td>0.374</td><td>0.338</td><td>0.395</td></tr><tr><td>MAE</td><td>0.372</td><td>0.367</td><td>0.390</td></tr><tr><td rowspan="2">ETTm2</td><td>MSE</td><td>0.282</td><td>0.261</td><td>0.281</td></tr><tr><td>MAE</td><td>0.321</td><td>0.319</td><td>0.321</td></tr><tr><td rowspan="2">Electricity</td><td>MSE</td><td>0.207</td><td>0.156</td><td>0.175</td></tr><tr><td>MAE</td><td>0.294</td><td>0.249</td><td>0.265</td></tr><tr><td rowspan="2">Weather</td><td>MSE</td><td>0.269</td><td>0.227</td><td>0.250</td></tr><tr><td>MAE</td><td>0.292</td><td>0.262</td><td>0.274</td></tr><tr><td rowspan="2">Average</td><td>MSE</td><td>0.309</td><td>0.286</td><td>0.314</td></tr><tr><td>MAE</td><td>0.345</td><td>0.331</td><td>0.343</td></tr></table>

Table 15. Full results of Fig. 5: Forecasting results (MAE) on the Monash TSF benchmark. We reported the reproduction results of LLMTime based on the GPT3.5 API from Woo et al. (2024). 

<table><tr><td></td><td>VISIONTS</td><td>LLMTime</td><td>MOIRAI $_{Small}$ </td><td>Naive</td><td>SES</td><td>Theta</td><td>TBATS</td><td>ETS</td><td>(DHR-)ARIMA</td><td>PR</td><td>CatBoost</td><td>FFNN</td><td>DeepAR</td><td>N-BEATS</td><td>WaveNet</td><td>Transformer</td></tr><tr><td>M1 Monthly</td><td>1987.69</td><td>2562.84</td><td>2082.26</td><td>2707.75</td><td>2259.04</td><td>2166.18</td><td>2237.5</td><td>1905.28</td><td>2080.13</td><td>2088.25</td><td>2052.32</td><td>2162.58</td><td>1860.81</td><td>1820.37</td><td>2184.42</td><td>2723.88</td></tr><tr><td>M3 Monthly</td><td>737.93</td><td>877.97</td><td>713.41</td><td>837.14</td><td>743.41</td><td>623.71</td><td>630.59</td><td>626.46</td><td>654.8</td><td>692.97</td><td>732</td><td>692.48</td><td>728.81</td><td>648.6</td><td>699.3</td><td>798.38</td></tr><tr><td>M3 Other</td><td>315.85</td><td>300.3</td><td>263.54</td><td>278.43</td><td>277.83</td><td>215.35</td><td>189.42</td><td>194.98</td><td>193.02</td><td>234.43</td><td>318.13</td><td>240.17</td><td>247.56</td><td>221.85</td><td>245.29</td><td>239.24</td></tr><tr><td>M4 Monthly</td><td>666.54</td><td>728.27</td><td>597.6</td><td>671.27</td><td>625.24</td><td>563.58</td><td>589.52</td><td>582.6</td><td>575.36</td><td>596.19</td><td>611.69</td><td>612.52</td><td>615.22</td><td>578.48</td><td>655.51</td><td>780.47</td></tr><tr><td>M4 Weekly</td><td>404.23</td><td>518.44</td><td>339.76</td><td>347.99</td><td>336.82</td><td>333.32</td><td>296.15</td><td>335.66</td><td>321.61</td><td>293.21</td><td>364.65</td><td>338.37</td><td>351.78</td><td>277.73</td><td>359.46</td><td>378.89</td></tr><tr><td>M4 Daily</td><td>215.63</td><td>266.52</td><td>189.1</td><td>180.83</td><td>178.27</td><td>178.86</td><td>176.6</td><td>193.26</td><td>179.67</td><td>181.92</td><td>231.36</td><td>177.91</td><td>299.79</td><td>190.44</td><td>189.47</td><td>201.08</td></tr><tr><td>M4 Hourly</td><td>288.37</td><td>576.06</td><td>268.04</td><td>1218.06</td><td>1218.06</td><td>1220.97</td><td>386.27</td><td>3358.1</td><td>1310.85</td><td>257.39</td><td>285.35</td><td>385.49</td><td>886.02</td><td>425.75</td><td>393.63</td><td>320.54</td></tr><tr><td>Tourism Quarterly</td><td>12931.88</td><td>16918.86</td><td>18352.44</td><td>15845.1</td><td>15014.19</td><td>7656.49</td><td>9972.42</td><td>8925.52</td><td>10475.47</td><td>9092.58</td><td>10267.97</td><td>8981.04</td><td>9511.37</td><td>8640.56</td><td>9137.12</td><td>9521.67</td></tr><tr><td>Tourism Monthly</td><td>2560.19</td><td>5608.61</td><td>3569.85</td><td>5636.83</td><td>5302.1</td><td>2069.96</td><td>2940.08</td><td>2004.51</td><td>2536.77</td><td>2187.28</td><td>2537.04</td><td>2022.21</td><td>1871.69</td><td>2003.02</td><td>2095.13</td><td>2146.98</td></tr><tr><td>CIF 2016</td><td>570907.24</td><td>599313.8</td><td>655888.58</td><td>578596.5</td><td>581875.97</td><td>714818.6</td><td>855578.4</td><td>642421.4</td><td>469059</td><td>563205.57</td><td>603551.3</td><td>1495923</td><td>3200418</td><td>679034.8</td><td>5998225</td><td>4057973</td></tr><tr><td>Aus. Elec. Demand</td><td>237.44</td><td>760.81</td><td>266.57</td><td>659.6</td><td>659.6</td><td>665.04</td><td>370.74</td><td>1282.99</td><td>1045.92</td><td>247.18</td><td>241.77</td><td>258.76</td><td>302.41</td><td>213.83</td><td>227.5</td><td>231.45</td></tr><tr><td>Bitcoin</td><td>2.33E+18</td><td>1.74E+18</td><td>1.76E+18</td><td>7.78E+17</td><td>5.33E+18</td><td>5.33E+18</td><td>9.9E+17</td><td>1.1E+18</td><td>3.62E+18</td><td>6.66E+17</td><td>1.93E+18</td><td>1.45E+18</td><td>1.95E+18</td><td>1.06E+18</td><td>2.46E+18</td><td>2.61E+18</td></tr><tr><td>Pedestrian Counts</td><td>52.01</td><td>97.77</td><td>54.88</td><td>170.88</td><td>170.87</td><td>170.94</td><td>222.38</td><td>216.5</td><td>635.16</td><td>44.18</td><td>43.41</td><td>46.41</td><td>44.78</td><td>66.84</td><td>46.46</td><td>47.29</td></tr><tr><td>Vehicle Trips</td><td>22.08</td><td>31.48</td><td>24.46</td><td>31.42</td><td>29.98</td><td>30.76</td><td>21.21</td><td>30.95</td><td>30.07</td><td>27.24</td><td>22.61</td><td>22.93</td><td>22</td><td>28.16</td><td>24.15</td><td>28.01</td></tr><tr><td>KDD cup</td><td>38.16</td><td>42.72</td><td>39.81</td><td>42.13</td><td>42.04</td><td>42.06</td><td>39.2</td><td>44.88</td><td>52.2</td><td>36.85</td><td>34.82</td><td>37.16</td><td>48.98</td><td>49.1</td><td>37.08</td><td>44.46</td></tr><tr><td>Weather</td><td>2.06</td><td>2.17</td><td>1.96</td><td>2.36</td><td>2.24</td><td>2.51</td><td>2.3</td><td>2.35</td><td>2.45</td><td>8.17</td><td>2.51</td><td>2.09</td><td>2.02</td><td>2.34</td><td>2.29</td><td>2.03</td></tr><tr><td>NN5 Daily</td><td>3.51</td><td>7.1</td><td>5.37</td><td>8.26</td><td>6.63</td><td>3.8</td><td>3.7</td><td>3.72</td><td>4.41</td><td>5.47</td><td>4.22</td><td>4.06</td><td>3.94</td><td>4.92</td><td>3.97</td><td>4.16</td></tr><tr><td>NN5 Weekly</td><td>14.67</td><td>15.76</td><td>15.07</td><td>16.71</td><td>15.66</td><td>15.3</td><td>14.98</td><td>15.7</td><td>15.38</td><td>14.94</td><td>15.29</td><td>15.02</td><td>14.69</td><td>14.19</td><td>19.34</td><td>20.34</td></tr><tr><td>Carparts</td><td>0.58</td><td>0.44</td><td>0.53</td><td>0.65</td><td>0.55</td><td>0.53</td><td>0.58</td><td>0.56</td><td>0.56</td><td>0.41</td><td>0.53</td><td>0.39</td><td>0.39</td><td>0.98</td><td>0.4</td><td>0.39</td></tr><tr><td>FRED-MD</td><td>1893.67</td><td>2804.64</td><td>2568.48</td><td>2825.67</td><td>2798.22</td><td>3492.84</td><td>1989.97</td><td>2041.42</td><td>2957.11</td><td>8921.94</td><td>2475.68</td><td>2339.57</td><td>4264.36</td><td>2557.8</td><td>2508.4</td><td>4666.04</td></tr><tr><td>Traffic Hourly</td><td>0.01</td><td>0.03</td><td>0.02</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.03</td><td>0.04</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.02</td><td>0.01</td></tr><tr><td>Traffic Weekly</td><td>1.14</td><td>1.15</td><td>1.17</td><td>1.19</td><td>1.12</td><td>1.13</td><td>1.17</td><td>1.14</td><td>1.22</td><td>1.13</td><td>1.17</td><td>1.15</td><td>1.18</td><td>1.11</td><td>1.2</td><td>1.42</td></tr><tr><td>Rideshare</td><td>5.92</td><td>6.28</td><td>1.35</td><td>6.29</td><td>6.29</td><td>7.62</td><td>6.45</td><td>6.29</td><td>3.37</td><td>6.3</td><td>6.07</td><td>6.59</td><td>6.28</td><td>5.55</td><td>2.75</td><td>6.29</td></tr><tr><td>Hospital</td><td>19.36</td><td>25.68</td><td>23</td><td>24.07</td><td>21.76</td><td>18.54</td><td>17.43</td><td>17.97</td><td>19.6</td><td>19.24</td><td>19.17</td><td>22.86</td><td>18.25</td><td>20.18</td><td>19.35</td><td>36.19</td></tr><tr><td>COVID Deaths</td><td>137.51</td><td>653.31</td><td>124.32</td><td>353.71</td><td>353.71</td><td>321.32</td><td>96.29</td><td>85.59</td><td>85.77</td><td>347.98</td><td>475.15</td><td>144.14</td><td>201.98</td><td>158.81</td><td>1049.48</td><td>408.66</td></tr><tr><td>Temperature Rain</td><td>6.37</td><td>6.37</td><td>5.3</td><td>9.39</td><td>8.18</td><td>8.22</td><td>7.14</td><td>8.21</td><td>7.19</td><td>6.13</td><td>6.76</td><td>5.56</td><td>5.37</td><td>7.28</td><td>5.81</td><td>5.24</td></tr><tr><td>Sunspot</td><td>2.81</td><td>5.07</td><td>0.11</td><td>3.93</td><td>4.93</td><td>4.93</td><td>2.57</td><td>4.93</td><td>2.57</td><td>3.83</td><td>2.27</td><td>7.97</td><td>0.77</td><td>14.47</td><td>0.17</td><td>0.13</td></tr><tr><td>Saugeen River Flow</td><td>30.22</td><td>34.84</td><td>24.07</td><td>21.5</td><td>21.5</td><td>21.49</td><td>22.26</td><td>30.69</td><td>22.38</td><td>25.24</td><td>21.28</td><td>22.98</td><td>23.51</td><td>27.92</td><td>22.17</td><td>28.06</td></tr><tr><td>US Births</td><td>519.94</td><td>1374.99</td><td>872.51</td><td>1152.67</td><td>1192.2</td><td>586.93</td><td>399</td><td>419.73</td><td>526.33</td><td>574.93</td><td>441.7</td><td>557.87</td><td>424.93</td><td>422</td><td>504.4</td><td>452.87</td></tr><tr><td>Normalized MAE</td><td>0.729</td><td>1.041</td><td>0.657</td><td>1.000</td><td>1.028</td><td>0.927</td><td>0.758</td><td>0.872</td><td>0.898</td><td>0.785</td><td>0.760</td><td>0.741</td><td>0.759</td><td>0.783</td><td>0.749</td><td>0.770</td></tr><tr><td>Rank</td><td>2</td><td>16</td><td>1</td><td>14</td><td>15</td><td>13</td><td>5</td><td>11</td><td>12</td><td>10</td><td>7</td><td>3</td><td>6</td><td>9</td><td>4</td><td>8</td></tr></table>

Table 16. Comparison of LaMa as the backbone. Results are averaged on four prediction lengths. 

<table><tr><td rowspan="2"></td><td colspan="2">MAE</td><td colspan="2">LaMa</td><td colspan="2">MOIRAI $_{Small}$ </td><td colspan="2">MOIRAI $_{Large}$ </td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.390</td><td>0.414</td><td>0.425</td><td>0.433</td><td>0.400</td><td>0.424</td><td>0.510</td><td>0.469</td></tr><tr><td>ETTh2</td><td>0.333</td><td>0.375</td><td>0.376</td><td>0.408</td><td>0.341</td><td>0.379</td><td>0.354</td><td>0.377</td></tr><tr><td>ETTm1</td><td>0.374</td><td>0.372</td><td>0.400</td><td>0.391</td><td>0.448</td><td>0.410</td><td>0.390</td><td>0.389</td></tr><tr><td>ETTm2</td><td>0.282</td><td>0.321</td><td>0.294</td><td>0.337</td><td>0.300</td><td>0.341</td><td>0.276</td><td>0.320</td></tr><tr><td>Average</td><td>0.344</td><td>0.370</td><td>0.374</td><td>0.392</td><td>0.372</td><td>0.388</td><td>0.382</td><td>0.388</td></tr></table>

Table 17. Full results of Table 2: zero-shot forecasting results of different MAE variants. Bold: best results among three variants. We also include the results from MOIRAI for reference. 

<table><tr><td colspan="2">Method</td><td colspan="2">MAE (Base)112M</td><td colspan="2">MAE (Large)330M</td><td colspan="2">MAE (Huge)657M</td><td colspan="2">MOIRAI (Small)14M</td><td colspan="2">MOIRAI (Base)91M</td><td colspan="2">MOIRAI (Huge)311M</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.353</td><td>0.383</td><td>0.346</td><td>0.382</td><td>0.362</td><td>0.384</td><td>0.375</td><td>0.402</td><td>0.384</td><td>0.402</td><td>0.380</td><td>0.398</td></tr><tr><td>192</td><td>0.392</td><td>0.410</td><td>0.379</td><td>0.406</td><td>0.407</td><td>0.414</td><td>0.399</td><td>0.419</td><td>0.425</td><td>0.429</td><td>0.440</td><td>0.434</td></tr><tr><td>336</td><td>0.407</td><td>0.423</td><td>0.391</td><td>0.416</td><td>0.399</td><td>0.419</td><td>0.412</td><td>0.429</td><td>0.456</td><td>0.450</td><td>0.514</td><td>0.474</td></tr><tr><td>720</td><td>0.406</td><td>0.441</td><td>0.397</td><td>0.433</td><td>0.395</td><td>0.433</td><td>0.413</td><td>0.444</td><td>0.470</td><td>0.473</td><td>0.705</td><td>0.568</td></tr><tr><td>avg</td><td>0.390</td><td>0.414</td><td>0.378</td><td>0.409</td><td>0.391</td><td>0.412</td><td>0.400</td><td>0.424</td><td>0.434</td><td>0.439</td><td>0.510</td><td>0.469</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.271</td><td>0.328</td><td>0.286</td><td>0.334</td><td>0.285</td><td>0.333</td><td>0.281</td><td>0.334</td><td>0.277</td><td>0.327</td><td>0.287</td><td>0.325</td></tr><tr><td>192</td><td>0.328</td><td>0.367</td><td>0.346</td><td>0.375</td><td>0.337</td><td>0.369</td><td>0.340</td><td>0.373</td><td>0.340</td><td>0.374</td><td>0.347</td><td>0.367</td></tr><tr><td>336</td><td>0.345</td><td>0.381</td><td>0.356</td><td>0.387</td><td>0.357</td><td>0.388</td><td>0.362</td><td>0.393</td><td>0.371</td><td>0.401</td><td>0.377</td><td>0.393</td></tr><tr><td>720</td><td>0.388</td><td>0.422</td><td>0.371</td><td>0.409</td><td>0.379</td><td>0.412</td><td>0.380</td><td>0.416</td><td>0.394</td><td>0.426</td><td>0.404</td><td>0.421</td></tr><tr><td>avg</td><td>0.333</td><td>0.375</td><td>0.340</td><td>0.377</td><td>0.339</td><td>0.375</td><td>0.341</td><td>0.379</td><td>0.346</td><td>0.382</td><td>0.354</td><td>0.377</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.341</td><td>0.347</td><td>0.344</td><td>0.349</td><td>0.352</td><td>0.351</td><td>0.404</td><td>0.383</td><td>0.335</td><td>0.360</td><td>0.353</td><td>0.363</td></tr><tr><td>192</td><td>0.360</td><td>0.360</td><td>0.365</td><td>0.363</td><td>0.360</td><td>0.367</td><td>0.435</td><td>0.402</td><td>0.366</td><td>0.379</td><td>0.376</td><td>0.380</td></tr><tr><td>336</td><td>0.377</td><td>0.374</td><td>0.381</td><td>0.376</td><td>0.381</td><td>0.383</td><td>0.462</td><td>0.416</td><td>0.391</td><td>0.394</td><td>0.399</td><td>0.395</td></tr><tr><td>720</td><td>0.416</td><td>0.405</td><td>0.429</td><td>0.411</td><td>0.440</td><td>0.412</td><td>0.490</td><td>0.437</td><td>0.434</td><td>0.419</td><td>0.432</td><td>0.417</td></tr><tr><td>avg</td><td>0.374</td><td>0.372</td><td>0.379</td><td>0.375</td><td>0.383</td><td>0.378</td><td>0.448</td><td>0.410</td><td>0.382</td><td>0.388</td><td>0.390</td><td>0.389</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.228</td><td>0.282</td><td>0.225</td><td>0.282</td><td>0.229</td><td>0.282</td><td>0.205</td><td>0.282</td><td>0.195</td><td>0.269</td><td>0.189</td><td>0.260</td></tr><tr><td>192</td><td>0.262</td><td>0.305</td><td>0.262</td><td>0.305</td><td>0.265</td><td>0.306</td><td>0.261</td><td>0.318</td><td>0.247</td><td>0.303</td><td>0.247</td><td>0.300</td></tr><tr><td>336</td><td>0.293</td><td>0.328</td><td>0.299</td><td>0.331</td><td>0.286</td><td>0.324</td><td>0.319</td><td>0.355</td><td>0.291</td><td>0.333</td><td>0.295</td><td>0.334</td></tr><tr><td>720</td><td>0.343</td><td>0.370</td><td>0.358</td><td>0.377</td><td>0.355</td><td>0.374</td><td>0.415</td><td>0.410</td><td>0.355</td><td>0.377</td><td>0.372</td><td>0.386</td></tr><tr><td>avg</td><td>0.282</td><td>0.321</td><td>0.286</td><td>0.324</td><td>0.284</td><td>0.322</td><td>0.300</td><td>0.341</td><td>0.272</td><td>0.321</td><td>0.276</td><td>0.320</td></tr><tr><td rowspan="5">Electricity</td><td>96</td><td>0.177</td><td>0.266</td><td>0.177</td><td>0.268</td><td>0.170</td><td>0.259</td><td>0.205</td><td>0.299</td><td>0.158</td><td>0.248</td><td>0.152</td><td>0.242</td></tr><tr><td>192</td><td>0.188</td><td>0.277</td><td>0.192</td><td>0.283</td><td>0.182</td><td>0.273</td><td>0.220</td><td>0.310</td><td>0.174</td><td>0.263</td><td>0.171</td><td>0.259</td></tr><tr><td>336</td><td>0.207</td><td>0.296</td><td>0.213</td><td>0.303</td><td>0.207</td><td>0.295</td><td>0.236</td><td>0.323</td><td>0.191</td><td>0.278</td><td>0.192</td><td>0.278</td></tr><tr><td>720</td><td>0.256</td><td>0.337</td><td>0.256</td><td>0.337</td><td>0.250</td><td>0.333</td><td>0.270</td><td>0.347</td><td>0.229</td><td>0.307</td><td>0.236</td><td>0.313</td></tr><tr><td>avg</td><td>0.207</td><td>0.294</td><td>0.209</td><td>0.298</td><td>0.202</td><td>0.290</td><td>0.233</td><td>0.320</td><td>0.188</td><td>0.274</td><td>0.188</td><td>0.273</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.220</td><td>0.257</td><td>0.222</td><td>0.257</td><td>0.235</td><td>0.265</td><td>0.173</td><td>0.212</td><td>0.167</td><td>0.203</td><td>0.177</td><td>0.208</td></tr><tr><td>192</td><td>0.244</td><td>0.275</td><td>0.246</td><td>0.275</td><td>0.276</td><td>0.288</td><td>0.216</td><td>0.250</td><td>0.209</td><td>0.241</td><td>0.219</td><td>0.249</td></tr><tr><td>336</td><td>0.280</td><td>0.299</td><td>0.283</td><td>0.301</td><td>0.304</td><td>0.309</td><td>0.260</td><td>0.282</td><td>0.256</td><td>0.276</td><td>0.277</td><td>0.292</td></tr><tr><td>720</td><td>0.330</td><td>0.337</td><td>0.338</td><td>0.343</td><td>0.351</td><td>0.350</td><td>0.320</td><td>0.322</td><td>0.321</td><td>0.323</td><td>0.365</td><td>0.350</td></tr><tr><td>avg</td><td>0.269</td><td>0.292</td><td>0.272</td><td>0.294</td><td>0.292</td><td>0.303</td><td>0.242</td><td>0.267</td><td>0.238</td><td>0.261</td><td>0.260</td><td>0.275</td></tr><tr><td colspan="2">Average $1^{st}$  count</td><td colspan="2">0.30938</td><td colspan="2">0.31117</td><td colspan="2">0.34617</td><td colspan="2">0.31517</td><td colspan="2">0.327-</td><td colspan="2">0.310-</td></tr></table>

Table 18. Impact of resampling filters and image orientations. 

<table><tr><td rowspan="3" colspan="2">MethodMetric</td><td colspan="6">Interpolation strategies in resampling</td><td rowspan="3" colspan="2">MethodMetric</td><td rowspan="2" colspan="2">-</td><td rowspan="2" colspan="2">Image orientationHorizontal flip</td><td rowspan="2" colspan="2">Vertical flip</td></tr><tr><td colspan="2">Bilinear</td><td colspan="2">Bicubic</td><td colspan="2">Nearest Neighbor</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.353</td><td>0.383</td><td>0.351</td><td>0.383</td><td>0.426</td><td>0.424</td><td rowspan="5">ETTh1</td><td>96</td><td>0.353</td><td>0.383</td><td>0.348</td><td>0.379</td><td>0.355</td><td>0.385</td></tr><tr><td>192</td><td>0.392</td><td>0.410</td><td>0.392</td><td>0.409</td><td>0.450</td><td>0.443</td><td>192</td><td>0.392</td><td>0.410</td><td>0.386</td><td>0.404</td><td>0.394</td><td>0.411</td></tr><tr><td>336</td><td>0.407</td><td>0.423</td><td>0.407</td><td>0.422</td><td>0.451</td><td>0.450</td><td>336</td><td>0.407</td><td>0.423</td><td>0.401</td><td>0.416</td><td>0.408</td><td>0.423</td></tr><tr><td>720</td><td>0.406</td><td>0.441</td><td>0.405</td><td>0.440</td><td>0.454</td><td>0.470</td><td>720</td><td>0.406</td><td>0.441</td><td>0.399</td><td>0.430</td><td>0.406</td><td>0.442</td></tr><tr><td>avg</td><td>0.390</td><td>0.414</td><td>0.389</td><td>0.414</td><td>0.445</td><td>0.446</td><td>avg</td><td>0.390</td><td>0.414</td><td>0.384</td><td>0.407</td><td>0.391</td><td>0.415</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.271</td><td>0.328</td><td>0.274</td><td>0.329</td><td>0.298</td><td>0.349</td><td rowspan="5">ETTh2</td><td>96</td><td>0.271</td><td>0.328</td><td>0.274</td><td>0.329</td><td>0.274</td><td>0.330</td></tr><tr><td>192</td><td>0.328</td><td>0.367</td><td>0.330</td><td>0.367</td><td>0.343</td><td>0.380</td><td>192</td><td>0.328</td><td>0.367</td><td>0.331</td><td>0.370</td><td>0.330</td><td>0.367</td></tr><tr><td>336</td><td>0.345</td><td>0.381</td><td>0.345</td><td>0.380</td><td>0.373</td><td>0.401</td><td>336</td><td>0.345</td><td>0.381</td><td>0.347</td><td>0.386</td><td>0.345</td><td>0.381</td></tr><tr><td>720</td><td>0.388</td><td>0.422</td><td>0.386</td><td>0.419</td><td>0.404</td><td>0.431</td><td>720</td><td>0.388</td><td>0.422</td><td>0.376</td><td>0.416</td><td>0.388</td><td>0.422</td></tr><tr><td>avg</td><td>0.333</td><td>0.375</td><td>0.334</td><td>0.374</td><td>0.354</td><td>0.390</td><td>avg</td><td>0.333</td><td>0.375</td><td>0.332</td><td>0.375</td><td>0.334</td><td>0.375</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.341</td><td>0.347</td><td>0.366</td><td>0.354</td><td>0.399</td><td>0.374</td><td rowspan="5">ETTm1</td><td>96</td><td>0.341</td><td>0.347</td><td>0.345</td><td>0.348</td><td>0.342</td><td>0.347</td></tr><tr><td>192</td><td>0.360</td><td>0.360</td><td>0.383</td><td>0.367</td><td>0.397</td><td>0.376</td><td>192</td><td>0.360</td><td>0.360</td><td>0.364</td><td>0.362</td><td>0.360</td><td>0.360</td></tr><tr><td>336</td><td>0.377</td><td>0.374</td><td>0.396</td><td>0.381</td><td>0.386</td><td>0.380</td><td>336</td><td>0.377</td><td>0.374</td><td>0.378</td><td>0.375</td><td>0.377</td><td>0.374</td></tr><tr><td>720</td><td>0.416</td><td>0.405</td><td>0.429</td><td>0.409</td><td>0.417</td><td>0.409</td><td>720</td><td>0.416</td><td>0.405</td><td>0.419</td><td>0.408</td><td>0.417</td><td>0.405</td></tr><tr><td>avg</td><td>0.374</td><td>0.372</td><td>0.393</td><td>0.378</td><td>0.400</td><td>0.384</td><td>avg</td><td>0.374</td><td>0.372</td><td>0.376</td><td>0.373</td><td>0.374</td><td>0.372</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.228</td><td>0.282</td><td>0.246</td><td>0.296</td><td>0.264</td><td>0.326</td><td rowspan="5">ETTm2</td><td>96</td><td>0.228</td><td>0.282</td><td>0.230</td><td>0.286</td><td>0.228</td><td>0.283</td></tr><tr><td>192</td><td>0.262</td><td>0.305</td><td>0.273</td><td>0.313</td><td>0.273</td><td>0.328</td><td>192</td><td>0.262</td><td>0.305</td><td>0.264</td><td>0.308</td><td>0.262</td><td>0.305</td></tr><tr><td>336</td><td>0.293</td><td>0.328</td><td>0.303</td><td>0.334</td><td>0.297</td><td>0.343</td><td>336</td><td>0.293</td><td>0.328</td><td>0.298</td><td>0.332</td><td>0.293</td><td>0.328</td></tr><tr><td>720</td><td>0.343</td><td>0.370</td><td>0.343</td><td>0.370</td><td>0.334</td><td>0.369</td><td>720</td><td>0.343</td><td>0.370</td><td>0.350</td><td>0.373</td><td>0.343</td><td>0.369</td></tr><tr><td>avg</td><td>0.282</td><td>0.321</td><td>0.291</td><td>0.328</td><td>0.292</td><td>0.341</td><td>avg</td><td>0.282</td><td>0.321</td><td>0.285</td><td>0.325</td><td>0.282</td><td>0.321</td></tr><tr><td colspan="2">Average $1^{st}$  count</td><td>0.34430</td><td>0.37018</td><td></td><td>0.35218</td><td></td><td>0.3732</td><td></td><td>Average $1^{st}$  count</td><td></td><td>0.34428</td><td></td><td>0.37016</td><td></td><td>0.34521</td></tr></table>

![](images/d0b16d343e6e66fe06bc7657eda4fe9b7821a0bb317ea4cc9ec6e2a28f4db7c2.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.41  |
| 0.4  | 0.395 |
| 0.6  | 0.385 |
| 0.8  | 0.39  |
| 1.0  | 0.425 |
</details>

(a) ETTh1

![](images/f1019d66cfe0d1d282f0883e7b8f7b180713805ac75abc09d1d29330941fda36.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.36  |
| 0.4  | 0.34  |
| 0.6  | 0.34  |
| 0.8  | 0.37  |
| 1.0  | 0.42  |
</details>

(b) ETTh2

![](images/caeb8100af6a432de2ed7a939f44b9e9cfad5968833d36f16ac9ba081160878e.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.38  |
| 0.4  | 0.375 |
| 0.6  | 0.38  |
| 0.8  | 0.42  |
| 1.0  | 0.46  |
</details>

(c) ETTm1

![](images/9ec6b7bcd3ee1d9a921cbbea45fb3d19368960ba864ac33a859b6562e7c5e8c1.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.285 |
| 0.4  | 0.282 |
| 0.6  | 0.281 |
| 0.8  | 0.305 |
| 1.0  | 0.335 |
</details>

(d) ETTm2

![](images/99e0772149b5011ddd80bdcad789571bc03596f2fb459486a71461cbb57e17cd.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.235 |
| 0.4  | 0.215 |
| 0.6  | 0.210 |
| 0.8  | 0.225 |
| 1.0  | 0.250 |
</details>

(e) Electricity

![](images/1141c72c2f0a08f25d191b7b2cabaae63fab5065e7e010e4c837103d0b8c2a8f.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.275 |
| 0.4  | 0.270 |
| 0.6  | 0.275 |
| 0.8  | 0.300 |
| 1.0  | 0.315 |
</details>

(f) Weather   
Figure 8. MSE (Y-axis) performance of different normalization constants r (X-axis).

![](images/891f2a309c93513091dd5a9da5331aa8f175a4ca05754dc8f76bf1d7f74cd795.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.44  |
| 0.4  | 0.39  |
| 0.6  | 0.41  |
| 0.8  | 0.43  |
| 1.0  | 0.43  |
</details>

(a) ETTh1

![](images/4135622a2eec28dd756ce63ee8c8a654048660ca3e27749fdd181c4f516296a6.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.38  |
| 0.4  | 0.33  |
| 0.6  | 0.34  |
| 0.8  | 0.36  |
| 1.0  | 0.37  |
</details>

(b) ETTh2

![](images/5d78531fe70d7bf76c3be5c528062ca24e004c0ff42b034fcbdd69342aaeb054.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.38  |
| 0.4  | 0.37  |
| 0.6  | 0.45  |
| 0.8  | 0.40  |
| 1.0  | 0.42  |
</details>

(c) ETTm1

![](images/f3e5804977e5348d27a2ca556f33f4a7efca06e12481bc51668b72f719976a00.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.26  |
| 0.4  | 0.28  |
| 0.6  | 0.285 |
| 0.8  | 0.29  |
| 1.0  | 0.30  |
</details>

(d) ETTm2

![](images/dba3ac92dbd962613c04f79cd38c5b674e3b29a930e76bb3a83eb661c295aef1.jpg)

<details>
<summary>line</summary>

| x    | Electricity |
| ---- | ----------- |
| 0.2  | 0.24        |
| 0.4  | 0.21        |
| 0.6  | 0.21        |
| 0.8  | 0.22        |
| 1.0  | 0.23        |
</details>

(e) Electricity

![](images/4162d51e778b856a43c8a21cc033ba69bdfe56816d553e2a63f7b90f0f99e85e.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.2  | 0.25  |
| 0.4  | 0.27  |
| 0.6  | 0.29  |
| 0.8  | 0.32  |
| 1.0  | 0.31  |
</details>

(f) Weather   
Figure 9. MSE (Y-axis) performance of different alignment constants c (X-axis).

![](images/70ee6cb070efa0cee51ccce221af47a39f74f65cdaecdb94a2affea562d9db87.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.45  |
| 1000 | 0.42  |
| 2000 | 0.40  |
| 3000 | 0.39  |
| 4000 | 0.41  |
</details>

(a) ETTh1

![](images/514fb50c155b8154ec3bac013e213a2d28afc31b37da9364c864cfe9494ad5a0.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 1000 | 0.38  |
| 2000 | 0.34  |
| 3000 | 0.36  |
| 4000 | 0.39  |
</details>

(b) ETTh2

![](images/c1ebe5d5ad9e8911d92be3b2b7c18fa2c86d41a4ad9aa87450fba0f743cec36f.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.42  |
| 1000 | 0.41  |
| 2000 | 0.39  |
| 3000 | 0.38  |
| 4000 | 0.38  |
</details>

(c) ETTm1

![](images/79b6a4962102157a24064509ea96b8a99ebe0c6e4c07fd9a2fc0ba64fffc0450.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 500  | 0.315 |
| 1000 | 0.305 |
| 1500 | 0.295 |
| 2000 | 0.285 |
| 2500 | 0.287 |
| 3000 | 0.288 |
| 3500 | 0.287 |
| 4000 | 0.282 |
</details>

(d) ETTm2

![](images/7b556963ef5b7d11fdf17861bbed4d58118ec44c8a9dcbcb2e89368ffe5a51ad.jpg)

<details>
<summary>line</summary>

| x    | Electricity |
| ---- | ----------- |
| 0    | 0.25        |
| 1000 | 0.22        |
| 2000 | 0.21        |
| 3000 | 0.21        |
| 4000 | 0.21        |
</details>

(e) Electricity

![](images/3168cce9e98fca15679f6a8e519286fb86a24eaec9d392e7666249be72fd06f9.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.325 |
| 1000 | 0.315 |
| 2000 | 0.305 |
| 3000 | 0.295 |
| 4000 | 0.275 |
</details>

(f) Weather   
Figure 10. MSE (Y-axis) performance of different context lengths L (X-axis).

# C. Full-Shot Forecasting

# C.1. Training details

Table 19. Final hyperparameters for VISIONTS used in our full-shot forecasting. 

<table><tr><td></td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Illness</td><td>Weather</td><td>Traffic</td><td>Electricity</td></tr><tr><td>Normalization constant r</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>1.0</td><td>1.0</td><td>0.4</td><td>0.4</td></tr><tr><td>Alignment constant c</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.7</td><td>0.4</td><td>0.4</td></tr><tr><td>Context length L</td><td>1152</td><td>1152</td><td>2304</td><td>1152</td><td>104</td><td>576</td><td>1152</td><td>1152</td></tr></table>

Based on the principle of channel independence (Nie et al., 2022; Han et al., 2024), we treat the variables of each time series as individual data samples. We use an Adam optimizer with a learning rate 0.0001 and a batch size 256 to fine-tune MAE. All experiments are repeated three times. The training epoch is one for all the datasets except Illness, for which we train MAE for 100 epochs with an early stop due to the limited training dataset scale. We conduct tuning on validation sets for the three hyperparameters, r, c, and L. The final hyperparameters used are summarized in Table 19.

# C.2. Full results and standard deviations

Table 20. Standard deviations of full-shot experiments. 

<table><tr><td colspan="2">Method</td><td colspan="2">VISIONTS</td><td colspan="2">Time-LLM</td><td colspan="2">GPT4TS</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td> $0.347 \pm 0.002$ </td><td> $0.376 \pm 0.000$ </td><td> $0.376 \pm 0.003$ </td><td> $0.402 \pm 0.002$ </td><td> $0.370 \pm 0.003$ </td><td> $0.389 \pm 0.001$ </td></tr><tr><td>192</td><td> $0.385 \pm 0.001$ </td><td> $0.400 \pm 0.000$ </td><td> $0.407 \pm 0.003$ </td><td> $0.421 \pm 0.002$ </td><td> $0.412 \pm 0.003$ </td><td> $0.413 \pm 0.001$ </td></tr><tr><td>336</td><td> $0.407 \pm 0.001$ </td><td> $0.415 \pm 0.001$ </td><td> $0.430 \pm 0.004$ </td><td> $0.438 \pm 0.001$ </td><td> $0.448 \pm 0.003$ </td><td> $0.431 \pm 0.001$ </td></tr><tr><td>720</td><td> $0.439 \pm 0.001$ </td><td> $0.443 \pm 0.000$ </td><td> $0.457 \pm 0.003$ </td><td> $0.468 \pm 0.001$ </td><td> $0.441 \pm 0.003$ </td><td> $0.449 \pm 0.001$ </td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td> $0.269 \pm 0.003$ </td><td> $0.328 \pm 0.002$ </td><td> $0.286 \pm 0.003$ </td><td> $0.346 \pm 0.002$ </td><td> $0.280 \pm 0.001$ </td><td> $0.335 \pm 0.001$ </td></tr><tr><td>192</td><td> $0.332 \pm 0.001$ </td><td> $0.374 \pm 0.001$ </td><td> $0.361 \pm 0.003$ </td><td> $0.391 \pm 0.002$ </td><td> $0.348 \pm 0.002$ </td><td> $0.380 \pm 0.001$ </td></tr><tr><td>336</td><td> $0.351 \pm 0.002$ </td><td> $0.395 \pm 0.002$ </td><td> $0.390 \pm 0.003$ </td><td> $0.414 \pm 0.002$ </td><td> $0.380 \pm 0.002$ </td><td> $0.405 \pm 0.001$ </td></tr><tr><td>720</td><td> $0.390 \pm 0.003$ </td><td> $0.430 \pm 0.002$ </td><td> $0.405 \pm 0.003$ </td><td> $0.434 \pm 0.002$ </td><td> $0.406 \pm 0.002$ </td><td> $0.436 \pm 0.001$ </td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td> $0.281 \pm 0.001$ </td><td> $0.322 \pm 0.001$ </td><td> $0.291 \pm 0.001$ </td><td> $0.341 \pm 0.001$ </td><td> $0.300 \pm 0.001$ </td><td> $0.340 \pm 0.000$ </td></tr><tr><td>192</td><td> $0.322 \pm 0.006$ </td><td> $0.353 \pm 0.002$ </td><td> $0.341 \pm 0.001$ </td><td> $0.369 \pm 0.001$ </td><td> $0.343 \pm 0.001$ </td><td> $0.368 \pm 0.000$ </td></tr><tr><td>336</td><td> $0.356 \pm 0.003$ </td><td> $0.379 \pm 0.002$ </td><td> $0.359 \pm 0.002$ </td><td> $0.379 \pm 0.001$ </td><td> $0.376 \pm 0.001$ </td><td> $0.386 \pm 0.000$ </td></tr><tr><td>720</td><td> $0.391 \pm 0.001$ </td><td> $0.413 \pm 0.001$ </td><td> $0.433 \pm 0.001$ </td><td> $0.419 \pm 0.001$ </td><td> $0.431 \pm 0.001$ </td><td> $0.416 \pm 0.000$ </td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td> $0.169 \pm 0.003$ </td><td> $0.256 \pm 0.002$ </td><td> $0.162 \pm 0.001$ </td><td> $0.248 \pm 0.001$ </td><td> $0.163 \pm 0.001$ </td><td> $0.249 \pm 0.001$ </td></tr><tr><td>192</td><td> $0.225 \pm 0.003$ </td><td> $0.294 \pm 0.003$ </td><td> $0.235 \pm 0.002$ </td><td> $0.304 \pm 0.001$ </td><td> $0.222 \pm 0.001$ </td><td> $0.291 \pm 0.000$ </td></tr><tr><td>336</td><td> $0.278 \pm 0.002$ </td><td> $0.334 \pm 0.001$ </td><td> $0.280 \pm 0.002$ </td><td> $0.329 \pm 0.001$ </td><td> $0.273 \pm 0.001$ </td><td> $0.327 \pm 0.001$ </td></tr><tr><td>720</td><td> $0.372 \pm 0.002$ </td><td> $0.392 \pm 0.002$ </td><td> $0.366 \pm 0.002$ </td><td> $0.382 \pm 0.001$ </td><td> $0.357 \pm 0.001$ </td><td> $0.376 \pm 0.001$ </td></tr><tr><td rowspan="4">Weather</td><td>96</td><td> $0.142 \pm 0.000$ </td><td> $0.192 \pm 0.001$ </td><td> $0.155 \pm 0.001$ </td><td> $0.199 \pm 0.001$ </td><td> $0.148 \pm 0.001$ </td><td> $0.188 \pm 0.000$ </td></tr><tr><td>192</td><td> $0.191 \pm 0.000$ </td><td> $0.238 \pm 0.000$ </td><td> $0.223 \pm 0.001$ </td><td> $0.261 \pm 0.001$ </td><td> $0.192 \pm 0.001$ </td><td> $0.230 \pm 0.000$ </td></tr><tr><td>336</td><td> $0.246 \pm 0.003$ </td><td> $0.282 \pm 0.001$ </td><td> $0.251 \pm 0.001$ </td><td> $0.279 \pm 0.001$ </td><td> $0.246 \pm 0.001$ </td><td> $0.273 \pm 0.000$ </td></tr><tr><td>720</td><td> $0.328 \pm 0.004$ </td><td> $0.337 \pm 0.001$ </td><td> $0.345 \pm 0.001$ </td><td> $0.342 \pm 0.001$ </td><td> $0.320 \pm 0.001$ </td><td> $0.328 \pm 0.000$ </td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td> $0.344 \pm 0.001$ </td><td> $0.236 \pm 0.000$ </td><td> $0.392 \pm 0.001$ </td><td> $0.267 \pm 0.000$ </td><td> $0.396 \pm 0.001$ </td><td> $0.264 \pm 0.000$ </td></tr><tr><td>192</td><td> $0.372 \pm 0.001$ </td><td> $0.249 \pm 0.001$ </td><td> $0.409 \pm 0.001$ </td><td> $0.271 \pm 0.000$ </td><td> $0.412 \pm 0.001$ </td><td> $0.268 \pm 0.000$ </td></tr><tr><td>336</td><td> $0.383 \pm 0.001$ </td><td> $0.257 \pm 0.001$ </td><td> $0.434 \pm 0.001$ </td><td> $0.296 \pm 0.000$ </td><td> $0.421 \pm 0.001$ </td><td> $0.273 \pm 0.000$ </td></tr><tr><td>720</td><td> $0.422 \pm 0.001$ </td><td> $0.280 \pm 0.000$ </td><td> $0.451 \pm 0.001$ </td><td> $0.291 \pm 0.000$ </td><td> $0.455 \pm 0.001$ </td><td> $0.291 \pm 0.000$ </td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td> $0.126 \pm 0.000$ </td><td> $0.218 \pm 0.000$ </td><td> $0.137 \pm 0.000$ </td><td> $0.233 \pm 0.000$ </td><td> $0.141 \pm 0.000$ </td><td> $0.239 \pm 0.000$ </td></tr><tr><td>192</td><td> $0.146 \pm 0.001$ </td><td> $0.239 \pm 0.001$ </td><td> $0.152 \pm 0.000$ </td><td> $0.247 \pm 0.000$ </td><td> $0.158 \pm 0.000$ </td><td> $0.253 \pm 0.000$ </td></tr><tr><td>336</td><td> $0.161 \pm 0.001$ </td><td> $0.255 \pm 0.001$ </td><td> $0.169 \pm 0.000$ </td><td> $0.267 \pm 0.000$ </td><td> $0.172 \pm 0.000$ </td><td> $0.266 \pm 0.000$ </td></tr><tr><td>720</td><td> $0.193 \pm 0.000$ </td><td> $0.286 \pm 0.000$ </td><td> $0.200 \pm 0.000$ </td><td> $0.290 \pm 0.000$ </td><td> $0.207 \pm 0.000$ </td><td> $0.293 \pm 0.000$ </td></tr><tr><td colspan="2"> $1^{st} count$ </td><td colspan="2">42</td><td colspan="2">2</td><td colspan="2">12</td></tr></table>

Table 21 shows the full results of the full-shot experiments. We also report the standard deviations of our full-shot experiments computed on three runs in Table 20, including the results of Time-LLM and GPT4TS from Tan et al. (2024) for reference.

# C.3. Ablation study and fine-tuning strategy comparison

We compare the following ablation variants to verify the role of the visual model (VM), similar to Tan et al. (2024).

\- w/o VM removes all the transformer blocks in encoders and decoders.

Table 21. Full results of Table 4: Full-shot forecasting performance on the long-term TSF benchmark. VISIONTS is fine-tuned only a single epoch on each dataset except for Illness. 

<table><tr><td rowspan="3" colspan="2">PretrainMethodMetric</td><td colspan="2">Images</td><td colspan="4">Text</td><td colspan="16">No Pretrain</td></tr><tr><td colspan="2">VISIONTS</td><td colspan="2">Time-LLM</td><td colspan="2">GPT4TS</td><td colspan="2">DLinear</td><td colspan="2">PatchTST</td><td colspan="2">TimesNet</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Stationary</td><td colspan="2">ETSformer</td><td colspan="2">Informer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.347</td><td>0.376</td><td>0.376</td><td>0.402</td><td>0.370</td><td>0.389</td><td>0.375</td><td>0.399</td><td>0.370</td><td>0.399</td><td>0.384</td><td>0.402</td><td>0.376</td><td>0.419</td><td>0.449</td><td>0.459</td><td>0.513</td><td>0.491</td><td>0.494</td><td>0.479</td><td>0.865</td><td>0.713</td></tr><tr><td>192</td><td>0.385</td><td>0.400</td><td>0.407</td><td>0.421</td><td>0.412</td><td>0.413</td><td>0.405</td><td>0.416</td><td>0.413</td><td>0.421</td><td>0.436</td><td>0.429</td><td>0.420</td><td>0.448</td><td>0.500</td><td>0.482</td><td>0.534</td><td>0.504</td><td>0.538</td><td>0.504</td><td>1.008</td><td>0.792</td></tr><tr><td>336</td><td>0.407</td><td>0.415</td><td>0.430</td><td>0.438</td><td>0.448</td><td>0.431</td><td>0.439</td><td>0.443</td><td>0.422</td><td>0.436</td><td>0.491</td><td>0.469</td><td>0.459</td><td>0.465</td><td>0.521</td><td>0.496</td><td>0.588</td><td>0.535</td><td>0.574</td><td>0.521</td><td>1.107</td><td>0.809</td></tr><tr><td>720</td><td>0.439</td><td>0.443</td><td>0.457</td><td>0.468</td><td>0.441</td><td>0.449</td><td>0.472</td><td>0.490</td><td>0.447</td><td>0.466</td><td>0.521</td><td>0.500</td><td>0.506</td><td>0.507</td><td>0.514</td><td>0.512</td><td>0.643</td><td>0.616</td><td>0.562</td><td>0.535</td><td>1.181</td><td>0.865</td></tr><tr><td>avg</td><td>0.395</td><td>0.409</td><td>0.418</td><td>0.432</td><td>0.418</td><td>0.421</td><td>0.423</td><td>0.437</td><td>0.413</td><td>0.431</td><td>0.458</td><td>0.450</td><td>0.440</td><td>0.460</td><td>0.496</td><td>0.487</td><td>0.570</td><td>0.537</td><td>0.542</td><td>0.510</td><td>1.040</td><td>0.795</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.269</td><td>0.328</td><td>0.286</td><td>0.346</td><td>0.280</td><td>0.335</td><td>0.289</td><td>0.353</td><td>0.274</td><td>0.336</td><td>0.340</td><td>0.374</td><td>0.358</td><td>0.397</td><td>0.346</td><td>0.388</td><td>0.476</td><td>0.458</td><td>0.340</td><td>0.391</td><td>3.755</td><td>1.525</td></tr><tr><td>192</td><td>0.332</td><td>0.374</td><td>0.361</td><td>0.391</td><td>0.348</td><td>0.380</td><td>0.383</td><td>0.418</td><td>0.339</td><td>0.379</td><td>0.402</td><td>0.414</td><td>0.429</td><td>0.439</td><td>0.456</td><td>0.452</td><td>0.512</td><td>0.493</td><td>0.430</td><td>0.439</td><td>5.602</td><td>1.931</td></tr><tr><td>336</td><td>0.351</td><td>0.395</td><td>0.390</td><td>0.414</td><td>0.380</td><td>0.405</td><td>0.448</td><td>0.465</td><td>0.329</td><td>0.380</td><td>0.452</td><td>0.452</td><td>0.496</td><td>0.487</td><td>0.482</td><td>0.486</td><td>0.552</td><td>0.551</td><td>0.485</td><td>0.479</td><td>4.721</td><td>1.835</td></tr><tr><td>720</td><td>0.390</td><td>0.430</td><td>0.405</td><td>0.434</td><td>0.406</td><td>0.436</td><td>0.605</td><td>0.551</td><td>0.379</td><td>0.422</td><td>0.462</td><td>0.468</td><td>0.463</td><td>0.474</td><td>0.515</td><td>0.511</td><td>0.562</td><td>0.560</td><td>0.500</td><td>0.497</td><td>3.647</td><td>1.625</td></tr><tr><td>avg</td><td>0.336</td><td>0.382</td><td>0.361</td><td>0.396</td><td>0.354</td><td>0.389</td><td>0.431</td><td>0.447</td><td>0.330</td><td>0.379</td><td>0.414</td><td>0.427</td><td>0.437</td><td>0.449</td><td>0.450</td><td>0.459</td><td>0.526</td><td>0.516</td><td>0.439</td><td>0.452</td><td>4.431</td><td>1.729</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.281</td><td>0.322</td><td>0.291</td><td>0.341</td><td>0.300</td><td>0.340</td><td>0.299</td><td>0.343</td><td>0.290</td><td>0.342</td><td>0.338</td><td>0.375</td><td>0.379</td><td>0.419</td><td>0.505</td><td>0.475</td><td>0.386</td><td>0.398</td><td>0.375</td><td>0.398</td><td>0.672</td><td>0.571</td></tr><tr><td>192</td><td>0.322</td><td>0.353</td><td>0.341</td><td>0.369</td><td>0.343</td><td>0.368</td><td>0.335</td><td>0.365</td><td>0.332</td><td>0.369</td><td>0.374</td><td>0.387</td><td>0.426</td><td>0.441</td><td>0.553</td><td>0.496</td><td>0.459</td><td>0.444</td><td>0.408</td><td>0.410</td><td>0.795</td><td>0.669</td></tr><tr><td>336</td><td>0.356</td><td>0.379</td><td>0.359</td><td>0.379</td><td>0.376</td><td>0.386</td><td>0.369</td><td>0.386</td><td>0.366</td><td>0.392</td><td>0.410</td><td>0.411</td><td>0.445</td><td>0.459</td><td>0.621</td><td>0.537</td><td>0.495</td><td>0.464</td><td>0.435</td><td>0.428</td><td>1.212</td><td>0.871</td></tr><tr><td>720</td><td>0.391</td><td>0.413</td><td>0.433</td><td>0.419</td><td>0.431</td><td>0.416</td><td>0.425</td><td>0.421</td><td>0.416</td><td>0.420</td><td>0.478</td><td>0.450</td><td>0.543</td><td>0.490</td><td>0.671</td><td>0.561</td><td>0.585</td><td>0.516</td><td>0.499</td><td>0.462</td><td>1.166</td><td>0.823</td></tr><tr><td>avg</td><td>0.338</td><td>0.367</td><td>0.356</td><td>0.377</td><td>0.363</td><td>0.378</td><td>0.357</td><td>0.379</td><td>0.351</td><td>0.381</td><td>0.400</td><td>0.406</td><td>0.448</td><td>0.452</td><td>0.588</td><td>0.517</td><td>0.481</td><td>0.456</td><td>0.429</td><td>0.425</td><td>0.961</td><td>0.734</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.169</td><td>0.256</td><td>0.162</td><td>0.248</td><td>0.163</td><td>0.249</td><td>0.167</td><td>0.269</td><td>0.165</td><td>0.255</td><td>0.187</td><td>0.267</td><td>0.203</td><td>0.287</td><td>0.255</td><td>0.339</td><td>0.192</td><td>0.274</td><td>0.189</td><td>0.280</td><td>0.365</td><td>0.453</td></tr><tr><td>192</td><td>0.225</td><td>0.294</td><td>0.235</td><td>0.304</td><td>0.222</td><td>0.291</td><td>0.224</td><td>0.303</td><td>0.220</td><td>0.292</td><td>0.249</td><td>0.309</td><td>0.269</td><td>0.328</td><td>0.281</td><td>0.340</td><td>0.280</td><td>0.339</td><td>0.253</td><td>0.319</td><td>0.533</td><td>0.563</td></tr><tr><td>336</td><td>0.278</td><td>0.334</td><td>0.280</td><td>0.329</td><td>0.273</td><td>0.327</td><td>0.281</td><td>0.342</td><td>0.274</td><td>0.329</td><td>0.321</td><td>0.351</td><td>0.325</td><td>0.366</td><td>0.339</td><td>0.372</td><td>0.334</td><td>0.361</td><td>0.314</td><td>0.357</td><td>1.363</td><td>0.887</td></tr><tr><td>720</td><td>0.372</td><td>0.392</td><td>0.366</td><td>0.382</td><td>0.357</td><td>0.376</td><td>0.397</td><td>0.421</td><td>0.362</td><td>0.385</td><td>0.408</td><td>0.403</td><td>0.421</td><td>0.415</td><td>0.433</td><td>0.432</td><td>0.417</td><td>0.413</td><td>0.414</td><td>0.413</td><td>3.379</td><td>1.338</td></tr><tr><td>avg</td><td>0.261</td><td>0.319</td><td>0.261</td><td>0.316</td><td>0.254</td><td>0.311</td><td>0.267</td><td>0.334</td><td>0.255</td><td>0.315</td><td>0.291</td><td>0.333</td><td>0.305</td><td>0.349</td><td>0.327</td><td>0.371</td><td>0.306</td><td>0.347</td><td>0.293</td><td>0.342</td><td>1.410</td><td>0.810</td></tr><tr><td rowspan="5">Illness</td><td>24</td><td>2.034</td><td>0.937</td><td>1.792</td><td>0.807</td><td>1.869</td><td>0.823</td><td>2.215</td><td>1.081</td><td>1.319</td><td>0.754</td><td>2.317</td><td>0.934</td><td>3.228</td><td>1.260</td><td>3.483</td><td>1.287</td><td>2.294</td><td>0.945</td><td>2.527</td><td>1.020</td><td>5.764</td><td>1.677</td></tr><tr><td>36</td><td>1.866</td><td>0.888</td><td>1.833</td><td>0.833</td><td>1.853</td><td>0.854</td><td>1.963</td><td>0.963</td><td>1.430</td><td>0.834</td><td>1.972</td><td>0.920</td><td>2.679</td><td>1.080</td><td>3.103</td><td>1.148</td><td>1.825</td><td>0.848</td><td>2.615</td><td>1.007</td><td>4.755</td><td>1.467</td></tr><tr><td>48</td><td>1.784</td><td>0.870</td><td>2.269</td><td>1.012</td><td>1.886</td><td>0.855</td><td>2.130</td><td>1.024</td><td>1.553</td><td>0.815</td><td>2.238</td><td>0.940</td><td>2.622</td><td>1.078</td><td>2.669</td><td>1.085</td><td>2.010</td><td>0.900</td><td>2.359</td><td>0.972</td><td>4.763</td><td>1.469</td></tr><tr><td>60</td><td>1.910</td><td>0.912</td><td>2.177</td><td>0.925</td><td>1.877</td><td>0.877</td><td>2.368</td><td>1.096</td><td>1.470</td><td>0.788</td><td>2.027</td><td>0.928</td><td>2.857</td><td>1.157</td><td>2.770</td><td>1.125</td><td>2.178</td><td>0.963</td><td>2.487</td><td>1.016</td><td>5.264</td><td>1.564</td></tr><tr><td>avg</td><td>1.899</td><td>0.902</td><td>2.018</td><td>0.894</td><td>1.871</td><td>0.852</td><td>2.169</td><td>1.041</td><td>1.443</td><td>0.798</td><td>2.139</td><td>0.931</td><td>2.847</td><td>1.144</td><td>3.006</td><td>1.161</td><td>2.077</td><td>0.914</td><td>2.497</td><td>1.004</td><td>5.137</td><td>1.544</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.142</td><td>0.192</td><td>0.155</td><td>0.199</td><td>0.148</td><td>0.188</td><td>0.176</td><td>0.237</td><td>0.149</td><td>0.198</td><td>0.172</td><td>0.220</td><td>0.217</td><td>0.296</td><td>0.266</td><td>0.336</td><td>0.173</td><td>0.223</td><td>0.197</td><td>0.281</td><td>0.300</td><td>0.384</td></tr><tr><td>192</td><td>0.191</td><td>0.238</td><td>0.223</td><td>0.261</td><td>0.192</td><td>0.230</td><td>0.220</td><td>0.282</td><td>0.194</td><td>0.241</td><td>0.219</td><td>0.261</td><td>0.276</td><td>0.336</td><td>0.307</td><td>0.367</td><td>0.245</td><td>0.285</td><td>0.237</td><td>0.312</td><td>0.598</td><td>0.544</td></tr><tr><td>336</td><td>0.246</td><td>0.282</td><td>0.251</td><td>0.279</td><td>0.246</td><td>0.273</td><td>0.265</td><td>0.319</td><td>0.245</td><td>0.282</td><td>0.280</td><td>0.306</td><td>0.339</td><td>0.380</td><td>0.359</td><td>0.395</td><td>0.321</td><td>0.338</td><td>0.298</td><td>0.353</td><td>0.578</td><td>0.523</td></tr><tr><td>720</td><td>0.328</td><td>0.337</td><td>0.345</td><td>0.342</td><td>0.320</td><td>0.328</td><td>0.333</td><td>0.362</td><td>0.314</td><td>0.334</td><td>0.365</td><td>0.359</td><td>0.403</td><td>0.428</td><td>0.419</td><td>0.428</td><td>0.414</td><td>0.410</td><td>0.352</td><td>0.388</td><td>1.059</td><td>0.741</td></tr><tr><td>avg</td><td>0.227</td><td>0.262</td><td>0.244</td><td>0.270</td><td>0.227</td><td>0.255</td><td>0.249</td><td>0.300</td><td>0.226</td><td>0.264</td><td>0.259</td><td>0.287</td><td>0.309</td><td>0.360</td><td>0.338</td><td>0.382</td><td>0.288</td><td>0.314</td><td>0.271</td><td>0.334</td><td>0.634</td><td>0.548</td></tr><tr><td rowspan="5">Traffic</td><td>96</td><td>0.344</td><td>0.236</td><td>0.392</td><td>0.267</td><td>0.396</td><td>0.264</td><td>0.410</td><td>0.282</td><td>0.360</td><td>0.249</td><td>0.593</td><td>0.321</td><td>0.587</td><td>0.366</td><td>0.613</td><td>0.388</td><td>0.612</td><td>0.338</td><td>0.607</td><td>0.392</td><td>0.719</td><td>0.391</td></tr><tr><td>192</td><td>0.372</td><td>0.249</td><td>0.409</td><td>0.271</td><td>0.412</td><td>0.268</td><td>0.423</td><td>0.287</td><td>0.379</td><td>0.256</td><td>0.617</td><td>0.336</td><td>0.604</td><td>0.373</td><td>0.616</td><td>0.382</td><td>0.613</td><td>0.340</td><td>0.621</td><td>0.399</td><td>0.696</td><td>0.379</td></tr><tr><td>336</td><td>0.383</td><td>0.257</td><td>0.434</td><td>0.296</td><td>0.421</td><td>0.273</td><td>0.436</td><td>0.296</td><td>0.392</td><td>0.264</td><td>0.629</td><td>0.336</td><td>0.621</td><td>0.383</td><td>0.622</td><td>0.337</td><td>0.618</td><td>0.328</td><td>0.622</td><td>0.396</td><td>0.777</td><td>0.420</td></tr><tr><td>720</td><td>0.422</td><td>0.280</td><td>0.451</td><td>0.291</td><td>0.455</td><td>0.291</td><td>0.466</td><td>0.315</td><td>0.432</td><td>0.286</td><td>0.640</td><td>0.350</td><td>0.626</td><td>0.382</td><td>0.660</td><td>0.408</td><td>0.653</td><td>0.355</td><td>0.632</td><td>0.396</td><td>0.864</td><td>0.472</td></tr><tr><td>avg</td><td>0.380</td><td>0.256</td><td>0.422</td><td>0.281</td><td>0.421</td><td>0.274</td><td>0.434</td><td>0.295</td><td>0.391</td><td>0.264</td><td>0.620</td><td>0.336</td><td>0.610</td><td>0.376</td><td>0.628</td><td>0.379</td><td>0.624</td><td>0.340</td><td>0.621</td><td>0.396</td><td>0.764</td><td>0.416</td></tr><tr><td rowspan="5">Electricity</td><td>96</td><td>0.126</td><td>0.218</td><td>0.137</td><td>0.233</td><td>0.141</td><td>0.239</td><td>0.140</td><td>0.237</td><td>0.129</td><td>0.222</td><td>0.168</td><td>0.272</td><td>0.193</td><td>0.308</td><td>0.201</td><td>0.317</td><td>0.169</td><td>0.273</td><td>0.187</td><td>0.304</td><td>0.274</td><td>0.368</td></tr><tr><td>192</td><td>0.144</td><td>0.237</td><td>0.152</td><td>0.247</td><td>0.158</td><td>0.253</td><td>0.153</td><td>0.249</td><td>0.157</td><td>0.240</td><td>0.184</td><td>0.289</td><td>0.201</td><td>0.315</td><td>0.222</td><td>0.334</td><td>0.182</td><td>0.286</td><td>0.199</td><td>0.315</td><td>0.296</td><td>0.386</td></tr><tr><td>336</td><td>0.162</td><td>0.256</td><td>0.169</td><td>0.267</td><td>0.172</td><td>0.266</td><td>0.169</td><td>0.267</td><td>0.163</td><td>0.259</td><td>0.198</td><td>0.300</td><td>0.214</td><td>0.329</td><td>0.231</td><td>0.338</td><td>0.200</td><td>0.304</td><td>0.212</td><td>0.329</td><td>0.300</td><td>0.394</td></tr><tr><td>720</td><td>0.192</td><td>0.286</td><td>0.200</td><td>0.290</td><td>0.207</td><td>0.293</td><td>0.203</td><td>0.301</td><td>0.197</td><td>0.290</td><td>0.220</td><td>0.320</td><td>0.246</td><td>0.355</td><td>0.254</td><td>0.361</td><td>0.222</td><td>0.321</td><td>0.233</td><td>0.345</td><td>0.373</td><td>0.439</td></tr><tr><td>avg</td><td>0.156</td><td>0.249</td><td>0.165</td><td>0.259</td><td>0.170</td><td>0.263</td><td>0.166</td><td>0.264</td><td>0.162</td><td>0.253</td><td>0.193</td><td>0.295</td><td>0.214</td><td>0.327</td><td>0.227</td><td>0.338</td><td>0.193</td><td>0.296</td><td>0.208</td><td>0.323</td><td>0.311</td><td>0.397</td></tr></table>

- VM2Attn replaces both the encoder and decoder with a self-attention layer, matching MAE structure but with random initialization.   
- VM2Trsf is similar to VM2Attn but replaces them with a Transformer block (i.e., a self-attention layer plus an MLP layer).   
- Rand-VM keeps the same architecture as the vanilla MAE, but all the weights are randomly initialized.

We also compare fine-tuning different components in MAE as follows:

- All fine-tunes all the trainable weights in MAE.   
- LN fine-tunes only the layer normalization, which is the default setting used in our experiments.   
- Bias fine-tunes only the bias term of all the linear layers, proposed by Zaken et al. (2022).   
- MLP and Attn fine-tune only the feed-forward layer and the self-attention layer, respectively.

Table 22. Ablation studies (left) and fine-tuning strategies (right). Results are averaged on four prediction lengths: {96, 192, 336, 720}. 

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">-</td><td colspan="4">Ablation on Visual MAE (VM)</td><td rowspan="2" colspan="7">Ablation on trained parameters</td><td></td></tr><tr><td>w/o VM</td><td>VM2Attn</td><td>VM2Trsf</td><td>Rand-VM</td><td></td></tr><tr><td rowspan="2">ETTh1</td><td>MSE</td><td>0.395</td><td>0.785</td><td>0.448</td><td>0.459</td><td>0.534</td><td rowspan="2">ETTh1</td><td>MSE</td><td>0.534</td><td>0.395</td><td>0.401</td><td>0.534</td><td>0.554</td><td>0.419</td></tr><tr><td>MAE</td><td>0.409</td><td>0.649</td><td>0.458</td><td>0.462</td><td>0.470</td><td>MAE</td><td>0.470</td><td>0.409</td><td>0.414</td><td>0.471</td><td>0.479</td><td>0.418</td></tr><tr><td rowspan="2">ETTh2</td><td>MSE</td><td>0.336</td><td>0.420</td><td>0.418</td><td>0.448</td><td>0.411</td><td rowspan="2">ETTh2</td><td>MSE</td><td>0.411</td><td>0.336</td><td>0.347</td><td>0.401</td><td>0.392</td><td>0.340</td></tr><tr><td>MAE</td><td>0.382</td><td>0.453</td><td>0.445</td><td>0.457</td><td>0.432</td><td>MAE</td><td>0.432</td><td>0.382</td><td>0.392</td><td>0.419</td><td>0.414</td><td>0.376</td></tr><tr><td rowspan="2">ETTm1</td><td>MSE</td><td>0.338</td><td>0.676</td><td>0.397</td><td>0.398</td><td>0.433</td><td rowspan="2">ETTm1</td><td>MSE</td><td>0.433</td><td>0.338</td><td>0.343</td><td>0.441</td><td>0.444</td><td>0.374</td></tr><tr><td>MAE</td><td>0.367</td><td>0.562</td><td>0.415</td><td>0.410</td><td>0.413</td><td>MAE</td><td>0.413</td><td>0.367</td><td>0.368</td><td>0.415</td><td>0.415</td><td>0.372</td></tr><tr><td rowspan="2">ETTm2</td><td>MSE</td><td>0.261</td><td>0.379</td><td>0.274</td><td>0.292</td><td>0.288</td><td rowspan="2">ETTm2</td><td>MSE</td><td>0.288</td><td>0.261</td><td>0.256</td><td>0.292</td><td>0.289</td><td>0.305</td></tr><tr><td>MAE</td><td>0.319</td><td>0.415</td><td>0.334</td><td>0.344</td><td>0.341</td><td>MAE</td><td>0.341</td><td>0.319</td><td>0.318</td><td>0.342</td><td>0.339</td><td>0.334</td></tr><tr><td rowspan="2">Average</td><td>MSE</td><td>0.333</td><td>0.565</td><td>0.384</td><td>0.399</td><td>0.417</td><td rowspan="2">Average</td><td>MSE</td><td>0.417</td><td>0.333</td><td>0.337</td><td>0.417</td><td>0.420</td><td>0.360</td></tr><tr><td>MAE</td><td>0.369</td><td>0.520</td><td>0.413</td><td>0.418</td><td>0.414</td><td>MAE</td><td>0.414</td><td>0.369</td><td>0.373</td><td>0.412</td><td>0.412</td><td>0.375</td></tr><tr><td colspan="2"> $1^{st}$  count</td><td>10</td><td>0</td><td>0</td><td>0</td><td>0</td><td colspan="2"> $1^{st}$  count</td><td>0</td><td>7</td><td>2</td><td>0</td><td>0</td><td>1</td></tr></table>

\- Freeze does not fine-tune any weight. Note that it differs from the previous zero-shot experiment, where a longer context length was used (see Table 8 and Table 19).

The results are shown in Table 22, suggesting that visual knowledge is crucial for VISIONTS and fine-tuning the layer normalization is the best.

# D. Visualization

We visualized the predictions of VISIONTS in the zero-shot setting, including its input and reconstructed images. We also visualized the predictions of MOIRAI $_{Large}$ and Seasonal Naïve, with their MAE metrics for comparison. Figs. 11 to 13 show examples where VISIONTS performed well, with Fig. 11 depicting a more regular pattern, while Figs. 12 and 13 display less obvious patterns. Fig. 14 illustrates a case where VISIONTS underperformed, as it aggressively predicted the trend despite the lack of clear patterns in the input sequence, whereas MOIRAI $_{Large}$ made more conservative predictions.

![](images/361ab5b9d70fa1f339f25bd07e498887124e3799292432d77882ec49e10cdb9d.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects
</details>

(a) Input Image

![](images/82093b703f3d3cc29aa98faaf139e6f72e295e4499252adcd59efa34964afa19.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects
</details>

(b) Reconstructed Image

![](images/a58d3e8b0cc6c9fa20d89cbab5e37e145ddf035da6000196a299dc53ff2b44fb.jpg)

<details>
<summary>line</summary>

| x    | target | prediction |
| ---- | ------ | ---------- |
| 0    | 0      | 0          |
| 1    | 1      | 0.5        |
| 2    | 0      | 0.3        |
| 3    | 1      | 0.7        |
| 4    | 0      | 0.4        |
| 5    | 1      | 0.6        |
| 6    | 0      | 0.8        |
| 7    | 1      | 0.9        |
| 8    | 0      | 1.0        |
| 9    | 1      | 1.1        |
| 10   | 0      | 1.2        |
| 11   | 1      | 1.3        |
| 12   | 0      | 1.4        |
| 13   | 1      | 1.5        |
| 14   | 0      | 1.6        |
| 15   | 1      | 1.7        |
| 16   | 0      | 1.8        |
| 17   | 1      | 1.9        |
| 18   | 0      | 2.0        |
| 19   | 1      | 2.1        |
| 20   | 0      | 2.2        |
| 21   | 1      | 2.3        |
| 22   | 0      | 2.4        |
| 23   | 1      | 2.5        |
| 24   | 0      | 2.6        |
| 25   | 1      | 2.7        |
| 26   | 0      | 2.8        |
| 27   | 1      | 2.9        |
| 28   | 0      | 3.0        |
| 29   | 1      | 3.1        |
| 30   | 0      | 3.2        |
| 31   | 1      | 3.3        |
| 32   | 0      | 3.4        |
| 33   | 1      | 3.5        |
| 34   | 0      | 3.6        |
| 35   | 1      | 3.7        |
| 36   | 0      | 3.8        |
| 37   | 1      | 3.9        |
| 38   | 0      | 4.0        |
| 39   | 1      | 4.1        |
| 40   | 0      | 4.2        |
| 41   | 1      | 4.3        |
| 42   | 0      | 4.4        |
| 43   | 1      | 4.5        |
| 44   | 0      | 4.6        |
| 45   | 1      | 4.7        |
| 46   | 0      | 4.8        |
| 47   | 1      | 4.9        |
| 48   | 0      | 5.0        |
| 49   | 1      | 5.1        |
| 50   | 0      | 5.2        |
| ...  | ...    | ...        |
| ...+ | ...    | ...        |
</details>

(c) VISIONTS (MAE = 0.312)

![](images/24fdaab9ccee406ba96a4e2fe048724a05d6e622be60edac84d3576d51785c00.jpg)

<details>
<summary>line</summary>

| x    | target | prediction |
| ---- | ------ | ---------- |
| 0    | 0      | 0          |
| 1    | 1      | 0          |
| 2    | 0      | 0          |
| 3    | 1      | 0          |
| 4    | 0      | 0          |
| 5    | 1      | 0          |
| 6    | 0      | 0          |
| 7    | 1      | 0          |
| 8    | 0      | 0          |
| 9    | 1      | 0          |
| 10   | 0      | 0          |
| 11   | 1      | 0          |
| 12   | 0      | 0          |
| 13   | 1      | 0          |
| 14   | 0      | 0          |
| 15   | 1      | 0          |
| 16   | 0      | 0          |
| 17   | 1      | 0          |
| 18   | 0      | 0          |
| 19   | 1      | 0          |
| 20   | 0      | 0          |
| 21   | 1      | 0          |
| 22   | 0      | 0          |
| 23   | 1      | 0          |
| 24   | 0      | 0          |
| 25   | 1      | 0          |
| 26   | 0      | 0          |
| 27   | 1      | 0          |
| 28   | 0      | 0          |
| 29   | 1      | 0          |
| 30   | 0      | 0          |
| 31   | 1      | 0          |
| 32   | 0      | 0          |
| 33   | 1      | 0          |
| 34   | 0      | 0          |
| 35   | 1      | 0          |
| 36   | 0      | 0          |
| 37   | 1      | 0          |
| 38   | 0      | 0          |
| 39   | 1      | 0          |
| 40   | 0      | 0          |
| 41   | 1      | 0          |
| 42   | 0      | 0          |
| 43   | 1      | 0          |
| 44   | 0      | 0          |
| 45   | 1      | 0          |
| 46   | 0      | 0          |
| 47   | 1      | 0          |
| 48   | 0      | 0          |
| 49   | 1      | 0          |
| 50   | 0      | 0          |
| ...  | ...    | ...        |
| ...+ | ...    | ...        |
</details>

(d) MOIRAI $_{LARGE}$ (MAE = 0.503)

![](images/433954eb757cebfc54bb1b346a43629840abea247ce10cd209ed18dd24fa42ea.jpg)

<details>
<summary>line</summary>

| x    | target | prediction |
| ---- | ------ | ---------- |
| 0    | 0      | 0          |
| 1    | 1      | 0          |
| 2    | 0      | 0          |
| 3    | 1      | 0          |
| 4    | 0      | 0          |
| 5    | 1      | 0          |
| 6    | 0      | 0          |
| 7    | 1      | 0          |
| 8    | 0      | 0          |
| 9    | 1      | 0          |
| 10   | 0      | 0          |
| 11   | 1      | 0          |
| 12   | 0      | 0          |
| 13   | 1      | 0          |
| 14   | 0      | 0          |
| 15   | 1      | 0          |
| 16   | 0      | 0          |
| 17   | 1      | 0          |
| 18   | 0      | 0          |
| 19   | 1      | 0          |
| 20   | 0      | 0          |
| 21   | 1      | 0          |
| 22   | 0      | 0          |
| 23   | 1      | 0          |
| 24   | 0      | 0          |
| 25   | 1      | 0          |
| 26   | 0      | 0          |
| 27   | 1      | 0          |
| 28   | 0      | 0          |
| 29   | 1      | 0          |
| 30   | 0      | 0          |
| 31   | 1      | 0          |
| 32   | 0      | 0          |
| 33   | 1      | 0          |
| 34   | 0      | 0          |
| 35   | 1      | 0          |
| 36   | 0      | 0          |
| 37   | 1      | 0          |
| 38   | 0      | 0          |
| 39   | 1      | 0          |
| 40   | 0      | 0          |
| 41   | 1      | 0          |
| 42   | 0      | 0          |
| 43   | 1      | 0          |
| 44   | 0      | 0          |
| 45   | 1      | 0          |
| 46   | 0      | 0          |
| 47   | 1      | 0          |
| 48   | 0      | 0          |
| 49   | 1      | 0          |
| 50   | 0      | 0          |
| ...  | ...    | ...        |
| ...+ | ...    | ...        |
</details>

(e) Seasonal Naïve (MAE = 0.774)   
Figure 11. Forecasting visualization on a sample from ETTh1. (a-b) Input/output images of VISIONTS. (c-e) Forecasting visualization.

![](images/cc247f30b15733fc44a88e79654ac823c750d787799805124eab87e47648e995.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects
</details>

(a) Input Image

![](images/2077a5c3bbb761cbf7dfa7b952a0c02675681c92c9b1bc55fb9199c664194c0e.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects
</details>

(b) Reconstructed Image

![](images/bf4970f0f1e51806a81cb6066feecd3f792f479f72d14ce22b3b21a90055e5de.jpg)

<details>
<summary>line</summary>

| x    | target | prediction |
| ---- | ------ | ---------- |
| 0    | 100    | 100        |
| 1    | 80     | 95         |
| 2    | 70     | 90         |
| 3    | 60     | 85         |
| 4    | 50     | 80         |
| 5    | 40     | 75         |
| 6    | 30     | 70         |
| 7    | 20     | 65         |
| 8    | 10     | 60         |
| 9    | 5      | 55         |
| 10   | 0      | 50         |
| 11   | -5     | 45         |
| 12   | -10    | 40         |
| 13   | -15    | 35         |
| 14   | -20    | 30         |
| 15   | -25    | 25         |
| 16   | -30    | 20         |
| 17   | -35    | 15         |
| 18   | -40    | 10         |
| 19   | -45    | 5          |
| 20   | -50    | 0          |
</details>

(c) VISIONTS (MAE = 0.157)

![](images/281020f0fc29ec5a441e4e541f50094d2f6a72ec721072f085ae2da3c484772c.jpg)

<details>
<summary>line</summary>

| Step | target | prediction |
| ---- | ------ | ---------- |
| 1    | 100    | -          |
| 2    | 80     | -          |
| 3    | 70     | -          |
| 4    | 60     | -          |
| 5    | 50     | -          |
| 6    | 40     | -          |
| 7    | 30     | -          |
| 8    | 20     | -          |
| 9    | 10     | -          |
| 10   | 5      | -          |
| 11   | 0      | -          |
| 12   | -5     | -          |
| 13   | -10    | -          |
| 14   | -15    | -          |
| 15   | -20    | -          |
| 16   | -25    | -          |
| 17   | -30    | -          |
| 18   | -35    | -          |
| 19   | -40    | -          |
| 20   | -45    | -          |
| 21   | -50    | -          |
| 22   | -55    | -          |
| 23   | -60    | -          |
| 24   | -65    | -          |
| 25   | -70    | -          |
| 26   | -75    | -          |
| 27   | -80    | -          |
| 28   | -85    | -          |
| 29   | -90    | -          |
| 30   | -95    | -          |
| 31   | -100   | -          |
| 32   | -105   | -          |
| 33   | -110   | -          |
| 34   | -115   | -          |
| 35   | -120   | -          |
| 36   | -125   | -          |
| 37   | -130   | -          |
| 38   | -135   | -          |
| 39   | -140   | -          |
| 40   | -145   | -          |
| 41   | -150   | -          |
| 42   | -155   | -          |
| 43   | -160   | -          |
| 44   | -165   | -          |
| 45   | -170   | -          |
| 46   | -175   | -          |
| 47   | -180   | -          |
| 48   | -185   | -          |
| 49   | -190   | -          |
| 50   | -195   | -          |
| 51   | -200   | -          |
| 52   | -205   | -          |
| 53   | -210   | -          |
| 54   | -215   | -          |
| 55   | -220   | -          |
| 56   | -225   | -          |
| 57   | -230   | -          |
| 58   | -235   | -          |
| 59   | -240   | -          |
| 60   | -245   | -          |
| 61   | -250   | -          |
| 62   | -255   | -          |
| 63   | -260   | -          |
| 64   | -265   | -          |
| 65   | -270   | -          |
| 66   | -275   | -          |
| 67   | -280   | -          |
| 68   | -285   | -          |
| 69   | -290   | -          |
| 70   | -295   | -          |
| 71   | -300   | -          |
| 72   | -305   | -          |
| 73   | -310   | -          |
| 74   | -315   | -          |
| 75   | -320   | -          |
| 76   | -325   | -          |
| 77   | -330   | -          |
| 78   | -335   | -          |
| 79   | -340   | -          |
| 80   | -345   | -          |
| 81   | -350   | -          |
| 82   | -355   | -          |
| 83   | -360   | -          |
| 84   | -365   | -          |
| 85   | -370   | -          |
| 86   | -375   | -          |
| 87   | -380   | -          |
| 88   | -385   | -          |
| 89   | -390   | -          |
| 90   | -395   | -          |
| 91   | -400   | -          |
| 92   | -405   | -          |
| 93   | -410   | -          |
| 94   | -415   | -          |
| 95   | -420   | -          |
| 96   | -425   | -          |
| 97   | -430   | -          |
| 98   | -435   | -          |
| 99   | -440   | -          |
| 100  | -445   | -          |
| 101  | -450   | ~-6        |
| 102  | ~-6    | ~-6        |
| 103  | ~-8    | ~-6        |
| 104  | ~-10   | ~-6        |
| 105  | ~-12   | ~-6        |
| 106  | ~-14   | ~-6        |
| 107  | ~-16   | ~-6        |
| 108  | ~-18   | ~-6        |
| 109  | ~-20   | ~-6        |
| 110  | ~-22   | ~-6        |
| 111  | ~-24   | ~-6        |
| 112  | ~-26   | ~-6        |
| 113  | ~-28   | ~-6        |
| 114  | ~-30   | ~-6        |
| 115  | ~-32   | ~-6        |
| 116  | ~-34   | ~-6        |
| 117  | ~-36   | ~-6        |
| 118  | ~-38   | ~-6        |
| 119  | ~-40   | ~-6        |
| 120  | ~-42   | ~-6        |
| 121  | ~-44   | ~-6        |
| 122  | ~-46   | ~-6        |
| 123  | ~-48   | ~-6        |
| 124  | ~-50   | ~-6        |
| 125  | ~-52   | ~-6        |
| 126  | ~-54   | ~-6        |
| 127  | ~-56   | ~-6        |
| 128  | ~-58   | ~-6        |
| 129  | ~-60   | ~-6        |
| 130  | ~-62   | ~-6        |
| 131  | ~-64   | ~-6        |
| 132  | ~-66   | ~-6        |
| 133  | ~-68   | ~-6        |
| 134  | ~-70   | ~-6        |
| 135  | ~-72   | ~-6        |
| 136  | ~-74   | ~-6        |
| 137  | ~-76   | ~-6        |
| 138  | ~-78   | ~-6        |
| 139  | ~-80   | ~-6        |
| 140  | ~-82   | ~-6        |
| 141  | ~-84   | ~-6        |
| 142  | ~-86   | ~-6        |
| 143  | ~-88   | ~-6        |
| 144  | ~-90   | ~-6        |
| 145  | ~-92   | ~-6        |
| 146  | ~-94   | ~-6        |
| 147  | ~-96   | ~-6        |
| 148  | ~-98   | ~-6        |
| 149  | ~-100                  | ~-6        |
| 150+ (post-) start: $ \texttt{target} $ to $ \texttt{post}(\texttt{post})$; $ \texttt{post}(\texttt{post})$ to $ \texttt{post}(\texttt{post})$; $ \texttt{post}(\texttt{post})$ to $ \texttt{post}(\texttt{post})$; $ \texttt{post}(\texttt{post})$ to $ \texttt{post}(\texttt{post})$; $ \texttt{post}(\texttt{post})$ to $ \texttt{post}(\mathbb{R})$; $ \texttt{post}(\texttt{post})$ to $ \texttt{post}(\texttt{post})$; $ \texttt{post}(\texttt{post})$ to $ \texttt{post}(\texttt{post})$; $ \texttt{post}(\texttt{post})$ to $ \texttt{post}(\texttt{post})$; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \texttt{post}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y)$; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \mathbb{R}(x,y) $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $; $ \texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\texttt{x,y} $$;\textit{x,y}$;
\end{array}
</details>

(d) MOIRAI $_{\text{LARGE}}$ (MAE = 0.251)

![](images/99dfb71bfa1b5a64add08f875a725982b39e81ca6280fc4d0c8014bccd818e5c.jpg)  
(e) Seasonal Naïve (MAE = 0.235)   
Figure 12. Forecasting visualization on a sample from ETTh2. (a-b) Input/output images of VISIONTS. (c-e) Forecasting visualization.

![](images/5abcc98db1bd1d340f97dfbc8886d911b7b89466ac02c6345eb79159c171513c.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects
</details>

(a) Input Image

![](images/1f2e5b002785b75eff9d1b627d2872104286446503bbbf4461ed6a3683de171f.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects
</details>

(b) Reconstructed Image

![](images/df01d9e7f633a9a163967ea96d304793ef6d8aa9e82719cbf523602428b56509.jpg)  
(c) VISIONTS (MAE = 0.821)

![](images/5cb014c5e036a9362ddc3663895203a6651b8add9517831caa4ea7926ff515c1.jpg)

<details>
<summary>line</summary>

| Step | target | prediction |
| ---- | ------ | ---------- |
| 1    | 0.8    | 0.7        |
| 2    | 0.9    | 0.6        |
| 3    | 0.7    | 0.5        |
| 4    | 0.8    | 0.6        |
| 5    | 0.9    | 0.7        |
| 6    | 0.8    | 0.6        |
| 7    | 0.7    | 0.5        |
| 8    | 0.8    | 0.6        |
| 9    | 0.9    | 0.7        |
| 10   | 0.8    | 0.6        |
| 11   | 0.7    | 0.5        |
| 12   | 0.8    | 0.6        |
| 13   | 0.9    | 0.7        |
| 14   | 0.8    | 0.6        |
| 15   | 0.7    | 0.5        |
| 16   | 0.8    | 0.6        |
| 17   | 0.9    | 0.7        |
| 18   | 0.8    | 0.6        |
| 19   | 0.7    | 0.5        |
| 20   | 0.8    | 0.6        |
| 21   | 0.9    | 0.7        |
| 22   | 0.8    | 0.6        |
| 23   | 0.7    | 0.5        |
| 24   | 0.8    | 0.6        |
| 25   | 0.9    | 0.7        |
| 26   | 0.8    | 0.6        |
| 27   | 0.7    | 0.5        |
| 28   | 0.8    | 0.6        |
| 29   | 0.9    | 0.7        |
| 30   | 0.8    | 0.6        |
| 31   | 0.7    | 0.5        |
| 32   | 0.8    | 0.6        |
| 33   | 0.9    | 0.7        |
| 34   | 0.8    | 0.6        |
| 35   | 0.7    | 0.5        |
| 36   | 0.8    | 0.6        |
| 37   | 0.9    | 0.7        |
| 38   | 0.8    | 0.6        |
| 39   | 0.7    | 0.5        |
| 40   | 0.8    | 0.6        |
| 41   | 0.9    | 0.7        |
| 42   | 0.8    | 0.6        |
| 43   | 0.7    | 0.5        |
| 44   | 0.8    | 0.6        |
| 45   | 0.9    | 0.7        |
| 46   | 0.8    | 0.6        |
| 47   | 0.7    | 0.5        |
| 48   | 0.8    | 0.6        |
| 49   | 0.9    | 0.7        |
| 50   | 0.8    | 0.6        |
| ... (dashed line) | ... (dashed line) | ... (dashed line) |
| Target (approx)      | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | -          |
| Prediction (approx)     | -      | ... (dashed line)|
| Prediction (approx)     | -      | ... (dashed line)|
</details>

(d) $\mathrm{MOIRAI}_{\mathrm{LARGE}}$ (MAE = 1.285)

![](images/4c617a3a7ddf6ba9d336289652e95916e025a2223ded953360b45c27af7af293.jpg)

<details>
<summary>line</summary>

| Step | target | prediction |
|------|--------|----------|
| 1    | 0.8    | 0.0      |
| 2    | 0.7    | 0.0      |
| 3    | 0.6    | 0.0      |
| 4    | 0.5    | 0.0      |
| 5    | 0.4    | 0.0      |
| 6    | 0.3    | 0.0      |
| 7    | 0.2    | 0.0      |
| 8    | 0.1    | 0.0      |
| 9    | 0.0    | 0.0      |
| 10   | -0.1   | 0.0      |
| 11   | -0.2   | 0.0      |
| 12   | -0.3   | 0.0      |
| 13   | -0.4   | 0.0      |
| 14   | -0.5   | 0.0      |
| 15   | -0.6   | 0.0      |
| 16   | -0.7   | 0.0      |
| 17   | -0.8   | 0.0      |
| 18   | -0.9   | 0.0      |
| 19   | -1.0   | 0.0      |
| 20   | -1.1   | 0.0      |
| 21   | -1.2   | 0.0      |
| 22   | -1.3   | 0.0      |
| 23   | -1.4   | 0.0      |
| 24   | -1.5   | 0.0      |
| 25   | -1.6   | 0.0      |
| 26   | -1.7   | 0.0      |
| 27   | -1.8   | 0.0      |
| 28   | -1.9   | 0.0      |
| 29   | -2.0   | 0.0      |
| 30   | -2.1   | 0.0      |
| 31   | -2.2   | 0.0      |
| 32   | -2.3   | 0.0      |
| 33   | -2.4   | 0.0      |
| 34   | -2.5   | 0.0      |
| 35   | -2.6   | 0.0      |
| 36   | -2.7   | 0.0      |
| 37   | -2.8   | 0.0      |
| 38   | -2.9   | 0.0      |
| 39   | -3.0   | 0.0      |
| 40   | -3.1   | 0.0      |
| 41   | -3.2   | 0.0      |
| 42   | -3.3   | 0.0      |
| 43   | -3.4   | 0.0      |
| 44   | -3.5   | 0.0      |
| 45   | -3.6   | 0.0      |
| 46   | -3.7   | 0.0      |
| 47   | -3.8   | 0.0      |
| 48   | -3.9   | 0.0      |
| 49   | -4.0   | 0.0      |
| 50   | -4.1   | 0.0      |
| 51   | -4.2   | 0.0      |
| 52   | -4.3   | 0.0      |
| 53   | -4.4   | 0.0      |
| 54   | -4.5   | 0.0      |
| 55   | -4.6   | 0.0      |
| 56   | -4.7   | 0.0      |
| 57   | -4.8   | 0.0      |
| 58   | -4.9   | 0.0      |
| 59   | -5.0   | 0.0      |
| 60   | -5.1   | 0.0      |
| 61   | -5.2   | 0.0      |
| 62   | -5.3   | 0.0      |
| 63   | -5.4   | 0.0      |
| 64   | -5.5   | 0.0      |
| 65   | -5.6   | 0.0      |
| 66   | -5.7   | 0.0      |
| 67   | -5.8   | 0.0      |
| 68   | -5.9   | 0.0      |
| 69   | -6.0   | 0.0      |
| 70   | -6.1   | 0.0      |
| 71   | -6.2   | 0.0      |
| 72   | -6.3   | 0.0      |
| 73   | -6.4   | 0.0      |
| 74   | -6.5   | 0.0      |
| 75   | -6.6   | 0.0      |
| 76   | -6.7   | 0.0      |
| 77   | -6.8   | 0.0      |
| 78   | -6.9   | 0.0      |
| 79   | -7.0   | 0.0      |
| 80   | -7.1   | 0.0      |
| 81   | -7.2   | 0.0      |
| 82   | -7.3   | 0.0      |
| 83   | -7.4   | 0.0      |
| 84   | -7.5   | 0.0      |
| 85   | -7.6   | 0.0      |
| 86   | -7.7   | 0.0      |
| 87   | -7.8   | 0.0      |
| 88   | -7.9   | 0.0      |
| 89   | -8.0   | 0.0      |
| 90   | -8.1   | 0.0      |
| 91   | -8.2   | 0.0      |
| 92   | -8.3   | 0.0      |
| 93   | -8.4   | 0.0      |
| 94   | -8.5   | 0.0      |
| 95   | -8.6   | 0.0      |
| 96   | -8.7   | 0.0      |
| 97   | -8.8   | 0.0      |
| 98   | -8.9   | 0.0      |
| 99   | -9.0   | 0.0      |
| Note: The actual values for 'target' and 'prediction' are not provided in the code snippet, so they are represented as placeholders (e.g., 'value' is not explicitly labeled). The actual values may be the result of the error bars or data points used to generate the actual values from the 'target' and 'prediction' arrays in the code.
</details>

(e) Seasonal Naïve (MAE = 1.523)   
Figure 13. Forecasting visualization on a sample from ETTh2. (a-b) Input/output images of VISIONTS. (c-e) Forecasting visualization.

![](images/dcd5268e5ac1124e05506c86f599dc25479e5adf700abe3dc278d653b55ad299.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects
</details>

(a) Input Image

![](images/889e1fcf6a57f432d464cfbfd4c92d0b67ec36bdfee761f844642fdf4ca93b2b.jpg)

<details>
<summary>natural_image</summary>

Blurred grayscale image with no discernible text, symbols, or identifiable objects.
</details>

(b) Reconstructed Image

![](images/d1240752aac63be2355e2d7fadeae053c05786e4b940063440846bc160d945d0.jpg)

<details>
<summary>line</summary>

| Step | target | prediction |
| ---- | ------ | ---------- |
| 1    | 0.5    | 0.4        |
| 2    | 0.7    | 0.6        |
| 3    | 0.9    | 0.8        |
| 4    | 1.1    | 1.0        |
| 5    | 1.3    | 1.2        |
| 6    | 1.5    | 1.4        |
| 7    | 1.7    | 1.6        |
| 8    | 1.9    | 1.8        |
| 9    | 2.1    | 2.0        |
| 10   | 2.3    | 2.2        |
| 11   | 2.5    | 2.4        |
| 12   | 2.7    | 2.6        |
| 13   | 2.9    | 2.8        |
| 14   | 3.1    | 3.0        |
| 15   | 3.3    | 3.2        |
| 16   | 3.5    | 3.4        |
| 17   | 3.7    | 3.6        |
| 18   | 3.9    | 3.8        |
| 19   | 4.1    | 4.0        |
| 20   | 4.3    | 4.2        |
| 21   | 4.5    | 4.4        |
| 22   | 4.7    | 4.6        |
| 23   | 4.9    | 4.8        |
| 24   | 5.1    | 5.0        |
| 25   | 5.3    | 5.2        |
| 26   | 5.5    | 5.4        |
| 27   | 5.7    | 5.6        |
| 28   | 5.9    | 5.8        |
| 29   | 6.1    | 6.0        |
| 30   | 6.3    | 6.2        |
| 31   | 6.5    | 6.4        |
| 32   | 6.7    | 6.6        |
| 33   | 6.9    | 6.8        |
| 34   | 7.1    | 7.0        |
| 35   | 7.3    | 7.2        |
| 36   | 7.5    | 7.4        |
| 37   | 7.7    | 7.6        |
| 38   | 7.9    | 7.8        |
| 39   | 8.1    | 8.0        |
| 40   | 8.3    | 8.2        |
| 41   | 8.5    | 8.4        |
| 42   | 8.7    | 8.6        |
| 43   | 8.9    | 8.8        |
| 44   | 9.1    | 9.0        |
| 45   | 9.3    | 9.2        |
| 46   | 9.5    | 9.4        |
| 47   | 9.7    | 9.6        |
| 48   | 9.9    | 9.8        |
| 49   | 10.1   | 10.0       |
| 50   | 10.3   | 10.2       |
| 51   | 10.5   | 10.4       |
| 52   | 10.7   | 10.6       |
| 53   | 10.9   | 10.8       |
| 54   | 11.1   | 11.0       |
| 55   | 11.3   | 11.2       |
| 56   | 11.5   | 11.4       |
| 57   | 11.7   | 11.6       |
| 58   | 11.9   | 11.8       |
| 59   | 12.1   | 12.0       |
| 60   | 12.3   | 12.2       |
| 61   | 12.5   | 12.4       |
| 62   | 12.7   | 12.6       |
| 63   | 12.9   | 12.8       |
| 64   | 13.1   | 13.0       |
| 65   | 13.3   | 13.2       |
| 66   | 13.5   | 13.4       |
| 67   | 13.7   | 13.6       |
| 68   | 13.9   | 13.8       |
| 69   | 14.1   | 14.0       |
| 70   | 14.3   | 14.2       |
| 71   | 14.5   | 14.4       |
| 72   | 14.7   | 14.6       |
| 73   | 14.9   | 14.8       |
| 74   | 15.1   | 15.0       |
| 75   | 15.3   | 15.2       |
| 76   | 15.5   | 15.4       |
| 77   | 15.7   | 15.6       |
| 78   | 15.9   | 15.8       |
| 79   | 16.1   | 16.0       |
| 80   | 16.3   | 16.2       |
| 81   | 16.5   | 16.4       |
| 82   | 16.7   | 16.6       |
| 83   | 16.9   | 16.8       |
| 84   | 17.1   | 17.0       |
| 85   | 17.3   | 17.2       |
| 86   | 17.5   | 17.4       |
| 87   | 17.7   | 17.6       |
| 88   | 17.9   | 17.8       |
| 89   | -      | -          |
| ... (multiple lines from left to right) are not explicitly labeled in the image.
</details>

(c) VISIONTS (MAE = 0.327)

![](images/01070f21646b6739fa4dec51ceef3364a1bfbcdf30819367ee5e3bc40d3d07d0.jpg)

<details>
<summary>line</summary>

| Step | target | prediction |
| ---- | ------ | ---------- |
| 1    | 0.5    | 0.4        |
| 2    | 0.7    | 0.6        |
| 3    | 0.9    | 0.8        |
| 4    | 1.1    | 1.0        |
| 5    | 1.3    | 1.2        |
| 6    | 1.5    | 1.4        |
| 7    | 1.7    | 1.6        |
| 8    | 1.9    | 1.8        |
| 9    | 2.1    | 2.0        |
| 10   | 2.3    | 2.2        |
| 11   | 2.5    | 2.4        |
| 12   | 2.7    | 2.6        |
| 13   | 2.9    | 2.8        |
| 14   | 3.1    | 3.0        |
| 15   | 3.3    | 3.2        |
| 16   | 3.5    | 3.4        |
| 17   | 3.7    | 3.6        |
| 18   | 3.9    | 3.8        |
| 19   | 4.1    | 4.0        |
| 20   | 4.3    | 4.2        |
| 21   | 4.5    | 4.4        |
| 22   | 4.7    | 4.6        |
| 23   | 4.9    | 4.8        |
| 24   | 5.1    | 5.0        |
| 25   | 5.3    | 5.2        |
| 26   | 5.5    | 5.4        |
| 27   | 5.7    | 5.6        |
| 28   | 5.9    | 5.8        |
| 29   | 6.1    | 6.0        |
| 30   | 6.3    | 6.2        |
| 31   | 6.5    | 6.4        |
| 32   | 6.7    | 6.6        |
| 33   | 6.9    | 6.8        |
| 34   | 7.1    | 7.0        |
| 35   | 7.3    | 7.2        |
| 36   | 7.5    | 7.4        |
| 37   | 7.7    | 7.6        |
| 38   | 7.9    | 7.8        |
| 39   | 8.1    | 8.0        |
| 40   | 8.3    | 8.2        |
| 41   | 8.5    | 8.4        |
| 42   | 8.7    | 8.6        |
| 43   | 8.9    | 8.8        |
| 44   | 9.1    | 9.0        |
| 45   | 9.3    | 9.2        |
| 46   | 9.5    | 9.4        |
| 47   | 9.7    | 9.6        |
| 48   | 9.9    | 9.8        |
| 49   | 10.1   | 10.0       |
| 50   | 10.3   | 10.2       |
| 51   | 10.5   | 10.4       |
| 52   | 10.7   | 10.6       |
| 53   | 10.9   | 10.8       |
| 54   | 11.1   | 11.0       |
| 55   | 11.3   | 11.2       |
| 56   | 11.5   | 11.4       |
| 57   | 11.7   | 11.6       |
| 58   | 11.9   | 11.8       |
| 59   | 12.1   | 12.0       |
| 60   | 12.3   | 12.2       |
| 61   | 12.5   | 12.4       |
| 62   | 12.7   | 12.6       |
| 63   | 12.9   | 12.8       |
| 64   | 13.1   | 13.0       |
| 65   | 13.3   | 13.2       |
| 66   | 13.5   | 13.4       |
| 67   | 13.7   | 13.6       |
| 68   | 13.9   | 13.8       |
| 69   | 14.1   | 14.0       |
| 70   | 14.3   | 14.2       |
| 71   | 14.5   | 14.4       |
| 72   | 14.7   | 14.6       |
| 73   | 14.9   | 14.8       |
| 74   | 15.1   | 15.0       |
| 75   | 15.3   | 15.2       |
| 76   | 15.5   | 15.4       |
| 77   | 15.7   | 15.6       |
| 78   | 15.9   | 15.8       |
| 79   | 16.1   | 16.0       |
| 80   | 16.3   | 16.2       |
| 81   | 16.5   | 16.4       |
| 82   | 16.7   | 16.6       |
| 83   | 16.9   | 16.8       |
| 84   | 17.1   | 17.0       |
| 85   | 17.3   | 17.2       |
| 86   | 17.5   | 17.4       |
| 87   | 17.7   | 17.6       |
| 88   | 17.9   | 17.8       |
| 89   | -      | -          |
| ... (multiple lines from left to right) represent the actual values of the target and prediction lines.
</details>

(d) MOIRAI $_{LARGE}$ (MAE = 0.172)

![](images/c5b263081c6b7a4f4a5906908a83edd3488dc15aeb6f694766aa0552dc986c7c.jpg)

<details>
<summary>line</summary>

| Step | target | prediction |
| ---- | ------ | ---------- |
| 1    | 0.5    | 0.4        |
| 2    | 0.7    | 0.6        |
| 3    | 0.9    | 0.8        |
| 4    | 1.1    | 1.0        |
| 5    | 1.3    | 1.2        |
| 6    | 1.5    | 1.4        |
| 7    | 1.7    | 1.6        |
| 8    | 1.9    | 1.8        |
| 9    | 2.1    | 2.0        |
| 10   | 2.3    | 2.2        |
| 11   | 2.5    | 2.4        |
| 12   | 2.7    | 2.6        |
| 13   | 2.9    | 2.8        |
| 14   | 3.1    | 3.0        |
| 15   | 3.3    | 3.2        |
| 16   | 3.5    | 3.4        |
| 17   | 3.7    | 3.6        |
| 18   | 3.9    | 3.8        |
| 19   | 4.1    | 4.0        |
| 20   | 4.3    | 4.2        |
| 21   | 4.5    | 4.4        |
| 22   | 4.7    | 4.6        |
| 23   | 4.9    | 4.8        |
| 24   | 5.1    | 5.0        |
| 25   | 5.3    | 5.2        |
| 26   | 5.5    | 5.4        |
| 27   | 5.7    | 5.6        |
| 28   | 5.9    | 5.8        |
| 29   | 6.1    | 6.0        |
| 30   | 6.3    | 6.2        |
| 31   | 6.5    | 6.4        |
| 32   | 6.7    | 6.6        |
| 33   | 6.9    | 6.8        |
| 34   | 7.1    | 7.0        |
| 35   | 7.3    | 7.2        |
| 36   | 7.5    | 7.4        |
| 37   | 7.7    | 7.6        |
| 38   | 7.9    | 7.8        |
| 39   | 8.1    | 8.0        |
| 40   | 8.3    | 8.2        |
| 41   | 8.5    | 8.4        |
| 42   | 8.7    | 8.6        |
| 43   | 8.9    | 8.8        |
| 44   | 9.1    | 9.0        |
| 45   | 9.3    | 9.2        |
| 46   | 9.5    | 9.4        |
| 47   | 9.7    | 9.6        |
| 48   | 9.9    | 9.8        |
| 49   | 10.1   | 10.0       |
| 50   | 10.3   | 10.2       |
| 51   | 10.5   | 10.4       |
| 52   | 10.7   | 10.6       |
| 53   | 10.9   | 10.8       |
| 54   | 11.1   | 11.0       |
| 55   | 11.3   | 11.2       |
| 56   | 11.5   | 11.4       |
| 57   | 11.7   | 11.6       |
| 58   | 11.9   | 11.8       |
| 59   | 12.1   | 12.0       |
| 60   | 12.3   | 12.2       |
| 61   | 12.5   | 12.4       |
| 62   | 12.7   | 12.6       |
| 63   | 12.9   | 12.8       |
| 64   | 13.1   | 13.0       |
| 65   | 13.3   | 13.2       |
| 66   | 13.5   | 13.4       |
| 67   | 13.7   | 13.6       |
| 68   | 13.9   | 13.8       |
| 69   | 14.1   | 14.0       |
| 70   | 14.3   | 14.2       |
| 71   | 14.5   | 14.4       |
| 72   | 14.7   | 14.6       |
| 73   | 14.9   | 14.8       |
| 74   | 15.1   | 15.0       |
| 75   | 15.3   | 15.2       |
| 76   | 15.5   | 15.4       |
| 77   | 15.7   | 15.6       |
| 78   | 15.9   | 15.8       |
| 79   | 16.1   | 16.0       |
| 80   | 16.3   | 16.2       |
| 81   | 16.5   | 16.4       |
| 82   | 16.7   | 16.6       |
| 83   | 16.9   | 16.8       |
| 84   | 17.1   | 17.0       |
| 85   | 17.3   | 17.2       |
| 86   | 17.5   | 17.4       |
| 87   | 17.7   | 17.6       |
| 88   | 17.9   | 17.8       |
| 89   | -      | -          |
| ... (multiple lines from left to right) represent cumulative values of target and prediction data points.
</details>

(e) Seasonal Naïve (MAE = 0.364)   
Figure 14. Forecasting visualization on a sample from ETTh1, where MOIRAI outperforms VISIONTS in terms of MAE. (a-b) Input/output images of VISIONTS. (c-e) Forecasting visualization.