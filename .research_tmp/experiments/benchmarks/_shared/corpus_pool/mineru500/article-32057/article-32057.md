# HDT: Hierarchical Discrete Transformer for Multivariate Time Series Forecasting

Shibo Feng $^{1,2}$ , Peilin Zhao $^{3*}$ , Liu Liu $^{3}$ , Pengcheng Wu $^{2}$ , Zhiqi Shen $^{1*}$

$^{1}$ College of Computing and Data Science, Nanyang Technological University (NTU), Singapore

$^{2}$ Webank-NTU Joint Research Institute on Fintech, NTU, Singapore

$^{3}$ Tencent AI Lab, Shenzhen, China

{shibo001, pengcheng.wu, zqshen}@ntu.edu.sg, {leonliuliu, masonzhao}@tencent.com

# Abstract

Generative models have gained significant attention in multivariate time series forecasting (MTS), particularly due to their ability to generate high-fidelity samples. Forecasting the probability distribution of multivariate time series is a challenging yet practical task. Although some recent attempts have been made to handle this task, two major challenges persist: 1) some existing generative methods underperform in high-dimensional multivariate time series forecasting, which is hard to scale to higher dimensions; 2) the inherent high-dimensional multivariate attributes constrain the forecasting lengths of existing generative models. In this paper, we point out that discrete token representations can model high-dimensional MTS with faster inference time, and forecasting the target with long-term trends of itself can extend the forecasting length with high accuracy. Motivated by this, we propose a vector quantized framework called Hierarchical Discrete Transformer (HDT) that models time series into discrete token representations with $\ell_2$ normalization enhanced vector quantized strategy, in which we transform the MTS forecasting into discrete tokens generation. To address the limitations of generative models in long-term forecasting, we propose a hierarchical discrete Transformer. This model captures the discrete long-term trend of the target at the low level and leverages this trend as a condition to generate the discrete representation of the target at the high level that introduces the features of the target itself to extend the forecasting length in high-dimensional MTS. Extensive experiments on five popular MTS datasets verify the effectiveness of our proposed method.

Code — https://github.com/hdtkk/HDT

# Introduction

Multivariate time series forecasting task has been applied to many real-world applications, such as economics (Sezer, Gudelek, and Ozbayoglu 2020; Feng et al. 2022), traffic (Wu et al. 2020; Liu et al. 2016), energy (Zhicheng et al. 2024) and weather (Qiu et al. 2017; Jin et al. 2023). As a generative task, MTS forecasting presents challenges in two key aspects: the inherent high-dimensionality of the data distribution, and the long-term forecasting. To model the complex distributions of high-dimensional data, previous studies have established deep generative models in both autoregressive and non-autoregressive ways. To our knowledge, most of the work in the context of high-dimensional MTS has focused on short-term forecasting (predicted length: 24, 48) (Rasul et al. 2020, 2024; Fan et al. 2024). To improve long-term forecasting, various Transformer architectures (Nie et al. 2022; Liu et al. 2023) have been proposed, but most are focused on low-dimensional scenarios. Effectively modeling high-dimensional distributions with longer forecasting lengths remains a challenge. A key issue is integrating deep generative models with sequence modeling frameworks to handle both high-dimensional data and long-term forecasting tasks.

Existing works (Salinas et al. 2020; Rasul et al. 2021; Li et al. 2022; Feng et al. 2023) have several attempts to utilize various forms of deep generative models, such as Normalizing flows (Dinh, Sohl-Dickstein, and Bengio 2016), Variational Auto-Encoder (VAEs) (Kingma and Welling 2013), Diffusion models (Li et al. 2024; Fan et al. 2024) to model high-dimensional MTS. They apply deep generative models to the high-dimensional distributions over time, learning the patterns of distribution changes along the temporal dimension for precise prediction. Due to complex patterns and long temporal dependencies of MTS, directly modeling high-dimensional MTS distributions in the time domain can lead to issues of distribution drift (Kim et al. 2021) and overlook the correlations between variables, limited to short-term forecasting settings.

Recently, several attention-variant Transformer frameworks (Liu et al. 2023; Rao, Li, and Miao 2022) and LLM-based structures (Zhou et al. 2023; Bian et al. 2024) have been applied to long-term forecasting of MTS, showing excellent performance on MTS datasets. Building on the success of these methods, we identified two key modules: the series decomposition block (Wu et al. 2021; Liu et al. 2022), which uses moving averages to smooth periodic fluctuations and highlight long-term trends, and the discrete Transformer for MTS modeling. Inspired by these approaches, we first learn the discrete representations of the MTS and then incorporate the long-term trends of the forecasting target into our model. This allows us to enhance forecasting length capability with high accuracy.

As a discrete framework, Vector Quantized (Gray 1984)

techniques have shown strong competitiveness in high-dimensional image fields (Rao et al. 2021; Zheng et al. 2022; Chang et al. 2023), These approaches utilize the pre-quantizing images into discrete latent variables and modeling them autoregressively. For the time series domain, VQ-based methods such as TimeVQVAE (Lee, Malacarne, and Aune 2023), TimeVAE (Desai et al. 2021) and TimeGAN (Yoon, Jarrett, and Van der Schaar 2019) all focus on time series generation task, the lateset VQ-TR (Rasul et al. 2024) introduce the VQ strategy within the transformer architecture as part of the encoder attention blocks, which attends over larger context windows with linear complexity in sequence length for efficient probabilistic forecasting. Inspired by their success of discrete strategy, we aim to explore the application of these techniques in the domain of high-dimensional MTS. Our model differs VQ-TR in two key aspects: i) HDT is two-stage, whereas it is end-to-end. ii) We focus on enhancing the long-term forecasting performance by introducing discrete representation of target itself, while they take efforts to reduce time and space complexity by discretizing the context inputs for efficient forecasting.

To extend the forecasting length within the high-dimensional MTS, we propose an effective generative framework, which is called Hierarchical Discrete Transformer HDT. It is a two-stage learning framework, consisting of a pre-quantizing module to obtain the discrete latent tokens of the forecasting targets, called tokenization, and a hierarchical modeling strategy for generating the discrete tokens. In the stage 1, we design two discrete token learning modules: one for obtaining latent tokens of our forecasting targets, and the other for obtaining latent tokens of downsampled targets using the downsampled input. This approach yields two key benefits: i) compressed latent discrete tokens effectively extend the prediction length for high-dimensional MTS, and ii) by incorporating the discrete latent space features of the targets, we reduce time complexity through shorter discrete token generation in stage 2.

In the stage 2, we devise a hierarchical discrete Transformer. At the low-level, we perform cross-attention between the contextual information and the discrete downsampled targets to generation task of downsampling target. At the high-level, we use the discrete downsampled results generated at the low-level as conditions to perform self-conditioned cross-attention with the discrete target, thereby achieving the generation of the discrete target. We summarize our main contributions as follows.

- We propose an effective hierarchical vector quantized method to introduce the long-term trend of targets for future target forecasting with higher accuracy and faster inference time.   
- We build a vector quantized MTS framework with $\ell_2$ normalization and self-conditioned cross attention for MTS forecasting, which can scale to high-dimensional and extend the prediction length with high accuracy.   
- Extensive experiments conducted on real-world datasets demonstrate the superiority of our HDT, achieving an average $16.7\%$ improvement on $\mathrm{CRPS}_{\mathrm{sum}}$ and $15.4\%$ on $\mathrm{NRMSE}_{\mathrm{sum}}$ , compared to the state-of-the-art methods.

# Methods

Our model comprises several key components. In this section, we present an overview of these components, which are divided into two stages. The training and inference details are shown in Algorithm 1, 2 and 3. Figure 1 provides an overview of the model architecture. In the stage 1, we have two types of VQGAN (Esser, Rombach, and Ommer 2021) structures (Encoder, Quantization, Decoder): one is based on the discrete representation learning of the downsampled time series, and the other is based on the discrete representation learning structure corresponding to the prediction targets. Since the VQ strategy is operated on the channel dimension, the inter-variate correlations are captured in stage 1. In stage 2, a context encoder and a base Transformer decoder perform temporal cross-attention to generate discrete downsampled targets. The output from these low-level modules is then fed into a self-conditioned Transformer decoder to autoregressively predict discrete target tokens. This two-stage approach captures inter- and intra-correlations with discrete tokens, enhancing the accuracy of time series forecasting.

# Stage 1: Modulating Quantized Vector

Series Downsample Module. According to the Autoformer (Wu et al. 2021), the moving average operation of non-stationary time series can smooth out periodic fluctuations and highlight long-term trends. As the objective of our work is to address the challenge of long-term forecasting in high-dimensional MTS, it is crucial for us to retain long-term patterns with the downsampled time series. For length- $\tau$ input series $X_{pred} \in R^{\tau \times D}$ , the process is:

$$
\mathcal {X} _ {\text { down }} = \text { AvgPool } (\text { Padding } (\mathcal {X} _ {\text { pred }})), \tag {1}
$$

where $X_{down} \in R^{\tau \times D}$ denotes the long-term pattern representations. Here, we introduce the AvgPool(.) for moving average with the Padding(.) to keep the series length unchanged. $X_{down}$ is the self-condition of targets, which consists of long-term patterns for the following future targets forecasting.

Discrete Tokenization using VQGAN. In the discrete representation learning of stage 1, the discrete learning modules of targets and downsampled targets show the same structure, which consists of an encoder and a decoder, with a quantization layer that maps a time series input into a sequence of tokens from a learned codebook. The details of these modules are provided in the Appendix C. Specifically, given any time series $\mathcal{X}_{pred} \in \mathbb{R}^{\tau \times D}$ can be represented by a spatial collection of codebook entries $z_{\mathbf{q}_t} \in \mathbb{R}^{s \times n_z}$ , where $n_z$ is the dimensionality of quantized vectors in the codebook and $s$ is the length of the discrete token sequence. In this way, each time series can be equivalently represented as a compact sequence with $s$ indices of the code vectors. The quantization operates on the channel dimension, capturing inter-variate correlations. Formally, the observed target $\mathcal{X}_{pred}$ and down-

![](images/e20554824ccc45626f1c19198655895c18a7023d64cff0606026ddc7b1888526.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Stage 1
        A["Input Image"] --> B["Encoder εd"]
        B --> C["Discrete Tokenizer"]
        C --> D["Decoder g̅d"]
        D --> E["Discriminator Dd"]
        E --> F["Discriminate Loss"]
        G["Future Target Image"] --> H["Encoder εt"]
        H --> I["Discrete Tokenizer"]
        I --> J["Decoder g̅t"]
        J --> K["Discriminator Dt"]
        K --> L["Discriminate Loss"]
    end

    subgraph Stage 2
        M["History Input Image"] --> N["Contextual Encoder"]
        N --> O["Base Transformer Decoder"]
        O --> P["Low-level Generation Ex~p(x)[-logp(sdown)"]]
        O --> Q["High-level Generation Ex~p(x)[-logp(spred)"]]
        R["Contextual Encoder"] --> S["Self-cond Transformer Decoder"]
    end

    style Stage 1 fill:#f9f,stroke:#333
    style Stage 2 fill:#bbf,stroke:#333
    note right of M: History Input
    note left of N: Contextual Encoder
    note right of O: Base Transformer Decoder
    note right of P: Self-cond Transformer Decoder
    note right of Q: Frozen weights
    note left of N: AvgPool: Moving average and pooling
    note right of Q: Updating weights
```
</details>

Figure 1: An illustration of our proposed HDT is provided. In stage 1, the model generates discrete downsampled targets and discrete targets, which are passed to Stage 2 for further processing. In stage 2, the contextual encoder and base Transformer decoder are trained with historical inputs and discrete downsampled tokens at the low level. Once trained, these low-level modules are fixed, and their outputs are fed into the high-level framework to generate the final discrete target sequence.

sampled target $\mathcal{X}_{down}$ are reconstructed by:

$$
\hat {\mathcal {X} _ {p r e d}} = \mathcal {G} _ {\theta_ {t}} \left(z _ {\mathbf {q} _ {t}}\right) = \mathcal {G} _ {\theta_ {t}} (\mathbf {q} _ {t} (\hat {z} ^ {t})) = \mathcal {G} _ {\theta_ {t}} \left(\mathbf {q} _ {t} \left(\mathcal {E} _ {\psi_ {t}} (\mathcal {X} _ {p r e d})\right)\right), \tag {2}
$$

$$
\hat {\mathcal {X} _ {d o w n}} = \mathcal {G} _ {\theta_ {d}} \left(z _ {\mathbf {q} _ {d}}\right) = \mathcal {G} _ {\theta_ {d}} (\mathbf {q} _ {d} (\hat {z} ^ {d})) = \mathcal {G} _ {\theta_ {d}} \left(\mathbf {q} _ {d} \left(\mathcal {E} _ {\psi_ {d}} (\mathcal {X} _ {d o w n})\right)\right). \tag {3}
$$

In particular, the $E_{\psi_{[t,d]}}, q_{[t,d]}, G_{\theta_{[t,d]}}$ are the encoders, quantization layers and decoders corresponding to $X_{pred}$ and $X_{down}$ , respectively. To avoid confusion and redundant expressions, we have removed the subscript symbols corresponding to the discrete learning and training process in the stage 1 formulas. The quantization operator q is conducted to transfer the continuous feature into the discrete space by looking up the closest codebook entry $z_{k}$ for each timestamp feature $\hat{z}_{i}$ within $\hat{z}$ , and note that $\hat{z}$ represents the execution process corresponding to both $\hat{z}^{t}$ and $\hat{z}^{d}$ .

$$
z _ {q} = \mathbf {q} (\hat {z}) = \underset {z _ {k} \in \mathcal {Z}} {\arg \min} \| \hat {z} _ {i} - z _ {k} \|, \tag {4}
$$

where $Z \in R^{K \times n_{z}}$ is the codebook that consists of K entries with $n_{z}$ dimensions and $\hat{z}_{i}$ is the continuous feature of the timestamp. Note that $z_{q_{t}}$ and $z_{q_{d}}$ each correspond to their respective codebooks $Z^{t}$ and $Z^{d}$ . The subscript for Z is omitted to maintain the brevity of the paper. The above models and the codebook can be learned by optimizing the following objectives:

$$
\begin{array}{l} \mathcal {L} _ {V Q} \left(\mathcal {E} _ {\psi}, \mathcal {G} _ {\theta}, \mathcal {Z}\right) = \| \mathcal {X} - \hat {\mathcal {X}} \| _ {2} ^ {2} + \| \mathrm{sg} \left[ \mathcal {E} _ {\psi} (\mathcal {X}) \right] - z _ {q} \| _ {2} ^ {2} \\ + \beta \| \operatorname{sg} [ z _ {q} ] - \mathcal {E} _ {\psi} (\mathcal {X}) \| _ {2} ^ {2}. \tag {5} \\ \end{array}
$$

In detail, sg denotes the stop-gradient operator, $\beta$ is a hyperparameter for the last term commitment loss. The first term is reconstruction loss and the second is codebook loss to optimize the entries in the codebook.

To learn a perceptually rich codebook in VQGAN, it introduces an adversarial training procedure with a patch-based discriminator $D=\{D_{t},D_{d}\}$ (Isola et al. 2017) that aims to differentiate between real and reconstructed images. In our setting, we introduce a shallow Conv1d network to enhance the reconstruction results:

$$
\mathcal {L} _ {\mathrm{GAN}} \left(\left\{\mathcal {E} _ {\psi}, \mathcal {G} _ {\theta}, \mathcal {Z} \right\}, D\right) = \left[ \log D (\mathcal {X}) + \log \left(1 - D (\hat {\mathcal {X}})\right) \right]. \tag {6}
$$

The final objective for finding the optimal Model $Q^{*} = E_{\psi}, G_{\theta}, Z$ is:

$$
\begin{array}{l} \mathcal {Q} ^ {*} = \arg \min _ {\mathcal {E} _ {\psi}, \mathcal {G} _ {\theta}, \mathcal {Z}} \max _ {D} \mathbb {E} _ {\mathcal {X} \sim p (\mathcal {X})} \left[ \mathcal {L} _ {\mathrm{VQ}} (\mathcal {E} _ {\psi}, \mathcal {G} _ {\theta}, \mathcal {Z}) \right. \\ + \lambda \mathcal {L} _ {\mathrm{GAN}} (\{\mathcal {E} _ {\psi}, \mathcal {G} _ {\theta}, \mathcal {Z} \}, D) ], \\ \end{array}
$$

where the $\lambda$ is an adaptive weight parameter, which is computed by the gradient of $\mathcal{G}_{\theta}$ and $D$ .

$\ell_{2}$ Regularization. However, in our experiments, we observed that applying $l_{2}$ normalization ( $\frac{x}{||x||_{2}}$ ) to the entries in the codebook can enhance the reconstruction performance.

$$
\mathcal {L} _ {\text { norm }} = \left\| \ell_ {2} \left(\mathcal {E} _ {\psi} (\mathcal {X})\right) - \ell_ {2} \left(z _ {k}\right) \right\| _ {2} ^ {2}. \tag {7}
$$

Finally, the training loss function is described as:

$$
\mathcal {L} = \mathcal {L} _ {V Q} \left(\mathcal {E} _ {\psi}, \mathcal {G} _ {\theta}, \mathcal {Z}\right) + \mathcal {L} _ {\text {GAN}} \left(\left\{\mathcal {E} _ {\psi}, \mathcal {G} _ {\theta}, \mathcal {Z} \right\}, D\right) + \mathcal {L} _ {\text {norm}}. \tag {8}
$$

Overall, in the stage 1, $\mathcal{X}_{pred}$ and $\mathcal{X}_{down}$ each obtain their respective codebooks $\mathcal{Z}^t$ and $\mathcal{Z}^d$ .

# Stage 2: Modelling Prior Distribution with HDT

In this section, we introduce the details of the hierarchical discrete transformer. In stage 2, we establish a framework to

estimate the underlying prior distribution over the discrete space for generating discrete time series tokens. This allows the post-quantization layers and the decoder from stage 1 to reconstruct the continuous targets. First, we present the overall generation process for the discrete tokens, as illustrated in Figure 1. Then, we detail the specific implementation procedures for both the low-level and high-level generation separately.

Low-level Token Generation. This process can be considered a preliminary process of target token generation of high-level. Specifically, we now have the context data $X_{p} \in R^{h \times D}$ and the discrete representation of the downsampled target $s_{down} = \{z_{q_{d}}^{s_{1}}, z_{q_{d}}^{s_{2}}, ..., z_{q_{d}}^{s_{d}}\} \in R^{s_{d} \times n_{z}}$ , where h is the look-back window length and D is the number of variates, $s_{d}$ is the length of discrete downsampled target sequence and $n_{z}$ is the feature dimension of the discrete representation. We formulate the training process by:

$$
\mathcal {H} _ {p} = \mathcal {E} _ {T} (\mathcal {X} _ {p}), \tag {9}
$$

$$
p (s _ {d o w n} | c) = \prod_ {i} p \left(z _ {\mathbf {q} _ {d}} ^ {s _ {i}} \mid z _ {\mathbf {q} _ {d}} ^ {s _ {<   i}}, c = \mathcal {H} _ {p}\right), \tag {10}
$$

$$
\mathcal {L} _ {\text { base }} = \mathbb {E} _ {x \sim p (x)} [ - \log p (s _ {d o w n}) ], \tag {11}
$$

where $E_{T}$ is the contextual encoder that is the Transformer encoder in our experiment. $H_{p} \in R^{h \times n_{z}}$ is the output of the context encoder and $L_{base}$ is the loss function of base Transformer decoder at the low-level framework. $p\left(z_{\mathbf{q}_{d}}^{s_{i}} \mid z_{\mathbf{q}_{d}}^{s_{<i}}, c = \mathcal{H}_{p}\right)$ is to compute the likelihood of the full representation $p(s_{down}|c) = \prod_{i} p\left(z_{\mathbf{q}_{d}}^{s_{i}} \mid z_{\mathbf{q}_{d}}^{s_{<i}}, c = \mathcal{H}_{p}\right)$ . We then obtain the trained context embedding $H_{p}$ and the downsampled tokens $s_{down}$ . Moreover, the discrete downsampled results directly impact the generation of high-level discrete targets, we explored three different methods for obtaining $H_{p}$ . These methods are explained in detail in the subsequent experimental section.

High-level Token Generation. After training the context encoder and base Transformer decoder in the low-level framework, we not only capture the content features of the context but also ensure that the discrete downsampled sequences retain long-term patterns. This provides additional conditions related to the target's own features in the high-level framework, thereby enhancing the accuracy of long-term forecasting. We have the discrete target $s_{pred} = \{z_{q_t}^{s_1}, z_{q_t}^{s_2}, ..., z_{q_t}^{s_p}\} \in R^{s_p \times n_z}$ , $s_{down}$ and $H_p$ , where the $s_p$ is the length of discrete target sequence. The process of autoregressively generating $s_{pred}$ can be described as follows:

$$
p (s _ {p r e d} | c) = \prod_ {i} p \left(z _ {\mathbf {q} _ {t}} ^ {s _ {i}} \mid z _ {\mathbf {q} _ {t}} ^ {s _ {<   i}}, c = \{s _ {d o w n}, \mathcal {H} _ {p} \}\right), \tag {12}
$$

$$
\mathcal {L} _ {\text { self - cond }} = \mathbb {E} _ {x \sim p (x)} [ - \log p (s _ {p r e d}) ], \tag {13}
$$

where the $s_{down}$ and $H^{p}$ are fixed, the cross-attention of self-conditioned Transformer decoder is operating between the $s_{down}$ and $s_{pred}$ , the temporal cross-attention is introduced to the $H^{p}$ and $s_{pred}$ , as shown in Figure 1. After completing the high-level training, we can input the discrete form of the target into the stage 1 decoder $G_{\theta_{t}}$ to reconstruct the predicted target. Notably, unlike the popular diffusion models, the VQ discretization strategy effectively avoids the efficiency issues associated with iterative diffusion structures and autoregressive prediction methods.

# Algorithm 1: Training of Stage 1

Input: Set of time series targets $X_{pred}$

Output: Encoder $E_{\psi_{t}}$ and $E_{\psi_{d}}$ , Decoder $G_{\theta_{t}}$ and $G_{\theta_{d}}$ , Discriminator $D_{t}$ and $D_{d}$ , quantization codebook $q_{t}$ and $q_{d}$ .

1: for $k \leftarrow 1$ to $K$ do

2: Get the $\mathrm{X}_{\mathrm{pred}} \sim \mathcal{X}_{pred}$ ;

3: Obtain the $X_{down}$ by Eqn. 1;

4: Feed $X_{pred}$ and $X_{down}$ to encoder $\{E_{\psi_{t}}, E_{\psi_{d}}\}$ , and quantization $\{q_{t}, q_{d}\}$ , by Eqn.(2, 3, 4), respectively;

5: Compute the $\ell_2$ Regularization and loss by Eqn.(5, 7);

6: if k ≥ k̂ is 0.75K then

7: Introduce the Discriminator $D_{t}$ , $D_{d}$ respectively and compute the loss by Eqn. 8;

8: end if

9: end for

10: Return trained $E_{\psi_{t}}$ , $E_{\psi_{d}}$ , $G_{\theta_{t}}$ , $G_{\theta_{d}}$ , $q_{t}$ , $q_{d}$ , $D_{t}$ and $D_{d}$ .

# Experiments

We conducted experiments to evaluate the performance and efficiency of HDT, covering short-term and long-term forecasting as well as robustness to missing values. The evaluation includes 5 real-world benchmarks and 12 baselines. Detailed model and experiment configurations are summarized in Appendix C.

Datasets. We extensively evaluate the proposed HDT on five real-world benchmarks, covering the mainstream high-dimensional MTS probabilistic forecasting applications, Solar (Lai et al. 2018), Electricity (Lai et al. 2018), Traffic (Salinas et al. 2019), Taxi (Salinas et al. 2019) and Wikipedia (Gasthaus et al. 2019). These data are recorded at intervals of 30 minutes, 1 hour, and 1 day frequencies, more details refer to Appendix B.

Baselines. We include several competitive multivariate time series baselines to verify the effectiveness of HDT. Previous work DeepAR (Salinas et al. 2020), GP-Copula (Salinas et al. 2019) and Transformer-MANF (Rasul et al. 2020). Then, we compare HDT against the diffusion-based methods, TimeGrad (Rasul et al. 2021), MG\_TSD (Fan et al. 2024), D³VAE (Li et al. 2022), CSDI (Tashiro et al. 2021), SSSD (Alcaraz and Strodthoff 2022), TSDiff (Kollovieh et al. 2023) with additional Transformer layers followed by S4 layer and TimeDiff (Shen and Kwok 2023). Among the MTS forecasting with VQ-Transformer, we introduce and VQ-TR (Rasul et al. 2023) for comparisons. The details of baselines are shown in Appendix F.

Evaluation Metrics. For probabilistic estimates, we report the continuously ranked probability score across summed time series (CRPS $_{sum}$ ) (Matheson and Winkler 1976), a widely used metric for probabilistic time series forecasting, as well as a deterministic estimation metric NRMSE $_{sum}$ (Normalized Root Mean Squared Error). For detailed de-

# Algorithm 2: Training of Stage II

Input: Set of history time series $X_{p}$ , targets $X_{pred}$ and trainable BOS token [BOS]. The optimized encoders $E_{\psi_{d}}$ and $G_{\theta_{t}}$ , trained quantization codebooks $q_{t}$ and $q_{d}$ .

Output: The base Transformer decoder B, contextual encoder $E_{T}$ , and self-cond Transformer decoder S.

1: for k ← 1 to K do   
2: Obtain the $X_{down}$ from $X_{pred}$ by Eqn. 1;   
3: Get the token sequences $s_{down}$ and $s_{pred}$ from trained $\mathbf{q}_t$ and $\mathbf{q}_d$ of stage 1 by Eqn. 4 with $X_{down}$ and $X_{pred}$ , respectively;   
4: Minimize the negative log-likelihood with training $E_{T}$ and B by Eqn.(9, 10, 11) with concatenating the [BOS] token at the beginning of token sequence $s_{down}$ .

# 5: end for

6: for $k \leftarrow 1$ to $K$ do

7: Introduce the output $s_{down}$ from the combination of trained $\mathcal{E}_T$ and $\mathcal{B}$ ;

8: Minimize the negative log-likelihood with frozen $E_{T}$ , B and trainable S by Eqn. 13 with concatenating the [BOS] token at the beginning of token sequence $s_{pred}$ .

# 9: end for

10: Return trained contextual encoder $\mathcal{E}_T$ , base Transformer decoder $\mathcal{B}$ , self-cond Transformer decoder $S$ and [BOS] token.

scriptions, refer to Appendix B.

Implementation Details. Our method relies on the ADAM optimizer with initial learning rates of 0.0005 and 0.001, and a batch size of 64 across all datasets. The history length is fixed at 96, with prediction lengths of $\{48, 96, 144\}$ . We sample 100 times to report metrics on the test set. All experiments are conducted on a single Nvidia A-100 GPU, and results are based on 3 runs.

# Main results

Probabilistic Forecasting Performance. As shown in Table 1, HDT achieves consistent state-of-the-art performance in most of benchmarks, covering three prediction settings, large span of dimensions and more showcases are shown in Supplementary due to the page limitation. Especially, HDT achieves a large performance gain over recent popular discrete method VQ-TR, such as average $25.9\%$ $\mathrm{CRPM}_{sum}$ improvement on Traffic, $23.1\%$ $\mathrm{CRPM}_{sum}$ improvement on Taxi. Also, we observe that HDT outperforms some diffusion-based methods TimeDiff, TSDiff and marginal improvement against recent strong baseline MG\_TSD that is more obvious in the case of high-dimensional and nonstationary datasets, such as average $13.3\%$ improvement on Traffic and $10.2\%$ on Taxi. This implies that the trends of target may introduce more future information gains into our forecasting model.

Deterministic Forecasting Performance. In our experiments, we observe that some models exhibit higher values for CRPS $_{sum}$ , yet lack true predictive accuracy. Therefore, we report the NRMSE $_{sum}$ for deterministic estimation,

# Algorithm 3: Inference

Input: Set of history time series $X_{p}$ , trained BOS token [BOS], trained contextual encoder $E_{T}$ , base Transformer decoder B and self-cond Transformer S and Decoder $G_{\theta_{t}}$ .

Output: Reconstructed future targets $X_{pred}$ .
1: for $i \leftarrow 1$ to I in test samples do

2: Sample the downsampled tokens $s_{down}$ with trained [BOS] token and $\mathcal{X}_p$ from the combination of $\mathcal{E}_T$ and $\mathcal{B}$ by Eqn.(9, 10, 11);   
3: Sample the target tokens $s$ with [BOS] token from trained $\mathcal{S}$ , $\mathcal{E}_T$ and $\mathcal{B}$ by Eqn.(12, 13);   
4: Return the target $X_{pred}$ by $G_{\theta_{t}}$ of Eqn.2.   
6: Return the prediction target $\mathcal{X}_{pred}$ .

# 5: end for

which is shown in Table 1. We found that HDT achieves the best results cross all datasets, especially in the Traffic and Taxi datasets, we achieve average improvements of 15.3% and 15.6% NRMSE $_{sum}$ comparing to the strong baseline MG\_TSD. It is worth noting that D $^{3}$ VAE and TimeDiff show significant deviations in point evaluations, but TSD-iff with self-guidance demonstrate competitive performance, implying the effectiveness of self-guided strategy. The time and space efficiencies of HDT are shown in Appendix D, due to limited space.

# Ablation studies

Effect of Discrete Representation $z_{q}$ in Eqn. (4). To verify the effectiveness of discrete representations in MTS, we conducted an experiment by bypassing the discretization of the intermediate variable $\hat{z}$ in stage 1, directly inputting it into stage 2 for autoregressive generation via cross-attention with the context encoder. We tested this on three datasets (Electricity, Traffic, Taxi) with two prediction lengths (48 and 96), covering dimensions from 370 to 1214. As shown in Table 2, the continuous structure (C-Transformer) performed poorly in both probabilistic and deterministic scenarios. We believe that without discretization, $\hat{z}$ acts as an infinitely large codebook, making it difficult for the stage 2 Transformer to fit properly. This highlights the effectiveness of our discrete Transformer structure.

Effect of Historical Condition $H^{p}$ in Eqn. (13). To verify the applicability of discrete representations in multivariate time series, we set four different forms of historical sequences during the second stage of training: (i) $HDT-h_{c}$ : the continuous features from the stage1, not transformed into discrete form; (ii) $HDT-h_{d}$ : transformed into the corresponding discrete form in the stage 1; (iii) $HDT-h_{d*}$ : the discrete features, without entering the Encoder of stage 2 (iv) $HDT-h_{dc}$ : concatenation of discrete and continuous representations from the stage 1. We test on two high-dimensional and distinct types of multivariate time series and the results are shown in Table 3, the relatively stable and periodic Traffic, and the Taxi series, which is of higher frequency of fluctuations and more outliers. We observe that in the Traffic and Taxi of all prediction settings, $HDT-h_{c}$ performs obviously lower than $HDT-h_{d}$ and $HDT-h_{d*}$ , while $HDT-h_{dc}$ is com-

<table><tr><td colspan="2">Models</td><td colspan="2">HDT(Ours)</td><td colspan="2">VQ-TR(2024)</td><td colspan="2">MG_TSD(2024)</td><td colspan="2">TSDiff(2024)</td><td colspan="2">TimeDiff(2023)</td><td colspan="2">SSSD(2023)</td><td colspan="2"> $D^3VAE$ (2022)</td><td colspan="2">CSDI(2021)</td><td colspan="2">TimeGrad(2021)</td><td colspan="2">Trans-MAF(2020)</td><td colspan="2">DeepAR(2020)</td><td colspan="2">GP-Copula(2019)</td></tr><tr><td colspan="2">Metric</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td></tr><tr><td rowspan="3">Solar</td><td>48</td><td>0.329</td><td>0.653</td><td>0.334</td><td>0.657</td><td>0.328</td><td>0.645</td><td>0.324</td><td>0.651</td><td>0.376</td><td>0.814</td><td>0.340</td><td>0.654</td><td>0.382</td><td>0.692</td><td>0.336</td><td>0.651</td><td>0.357</td><td>0.667</td><td>0.341</td><td>0.672</td><td>0.362</td><td>0.691</td><td>0.426</td><td>0.891</td></tr><tr><td>96</td><td>0.330</td><td>0.694</td><td>0.357</td><td>0.734</td><td>0.339</td><td>0.707</td><td>0.336</td><td>0.715</td><td>0.415</td><td>0.935</td><td>0.365</td><td>0.704</td><td>0.413</td><td>0.757</td><td>0.359</td><td>0.712</td><td>0.384</td><td>0.731</td><td>0.376</td><td>0.743</td><td>0.402</td><td>0.775</td><td>0.475</td><td>0.921</td></tr><tr><td>144</td><td>0.357</td><td>0.776</td><td>0.377</td><td>0.885</td><td>0.373</td><td>0.825</td><td>0.379</td><td>0.847</td><td>0.438</td><td>1.312</td><td>0.392</td><td>0.830</td><td>0.448</td><td>0.914</td><td>0.387</td><td>0.865</td><td>0.429</td><td>0.916</td><td>0.394</td><td>0.824</td><td>0.448</td><td>0.936</td><td>0.559</td><td>1.207</td></tr><tr><td></td><td>Avg</td><td>0.338</td><td>0.707</td><td>0.356</td><td>0.758</td><td>0.347</td><td>0.726</td><td>0.346</td><td>0.738</td><td>0.410</td><td>1.020</td><td>0.366</td><td>0.729</td><td>0.414</td><td>0.788</td><td>0.361</td><td>0.743</td><td>0.390</td><td>0.771</td><td>0.370</td><td>0.746</td><td>0.404</td><td>0.801</td><td>0.487</td><td>1.006</td></tr><tr><td rowspan="3">Electricity</td><td>48</td><td>0.025</td><td>0.030</td><td>0.034</td><td>0.033</td><td>0.023</td><td>0.030</td><td>0.024</td><td>0.029</td><td>0.036</td><td>0.092</td><td>0.037</td><td>0.032</td><td>0.046</td><td>0.096</td><td>0.032</td><td>0.034</td><td>0.043</td><td>0.031</td><td>0.039</td><td>0.034</td><td>0.043</td><td>0.035</td><td>0.047</td><td>0.055</td></tr><tr><td>96</td><td>0.028</td><td>0.032</td><td>0.045</td><td>0.040</td><td>0.034</td><td>0.035</td><td>0.039</td><td>0.036</td><td>0.049</td><td>0.109</td><td>0.045</td><td>0.041</td><td>0.062</td><td>0.114</td><td>0.049</td><td>0.039</td><td>0.067</td><td>0.035</td><td>0.060</td><td>0.038</td><td>0.058</td><td>0.044</td><td>0.069</td><td>0.058</td></tr><tr><td>144</td><td>0.036</td><td>0.057</td><td>0.049</td><td>0.070</td><td>0.042</td><td>0.064</td><td>0.047</td><td>0.072</td><td>0.063</td><td>0.147</td><td>0.056</td><td>0.084</td><td>0.086</td><td>0.142</td><td>0.067</td><td>0.088</td><td>0.082</td><td>0.085</td><td>0.101</td><td>0.093</td><td>0.104</td><td>0.097</td><td>0.125</td><td>0.109</td></tr><tr><td></td><td>Avg</td><td>0.028</td><td>0.038</td><td>0.043</td><td>0.048</td><td>0.033</td><td>0.043</td><td>0.036</td><td>0.046</td><td>0.049</td><td>0.116</td><td>0.046</td><td>0.052</td><td>0.065</td><td>0.117</td><td>0.049</td><td>0.054</td><td>0.064</td><td>0.050</td><td>0.066</td><td>0.055</td><td>0.068</td><td>0.059</td><td>0.080</td><td>0.074</td></tr><tr><td rowspan="3">Traffic</td><td>48</td><td>0.034</td><td>0.060</td><td>0.039</td><td>0.074</td><td>0.036</td><td>0.067</td><td>0.057</td><td>0.070</td><td>0.064</td><td>0.175</td><td>0.053</td><td>0.074</td><td>0.082</td><td>0.312</td><td>-</td><td>-</td><td>0.067</td><td>0.072</td><td>0.070</td><td>0.074</td><td>0.069</td><td>0.081</td><td>0.082</td><td>0.136</td></tr><tr><td>96</td><td>0.037</td><td>0.063</td><td>0.052</td><td>0.082</td><td>0.042</td><td>0.072</td><td>0.068</td><td>0.076</td><td>0.081</td><td>0.246</td><td>0.069</td><td>0.080</td><td>0.091</td><td>0.465</td><td>-</td><td>-</td><td>0.095</td><td>0.087</td><td>0.086</td><td>0.081</td><td>0.099</td><td>0.128</td><td>0.093</td><td>0.148</td></tr><tr><td>144</td><td>0.047</td><td>0.076</td><td>0.068</td><td>0.093</td><td>0.056</td><td>0.096</td><td>0.095</td><td>0.114</td><td>0.109</td><td>0.304</td><td>0.084</td><td>0.106</td><td>0.129</td><td>0.472</td><td>-</td><td>-</td><td>0.124</td><td>0.105</td><td>0.107</td><td>0.096</td><td>0.113</td><td>0.142</td><td>0.125</td><td>0.185</td></tr><tr><td></td><td>Avg</td><td>0.039</td><td>0.066</td><td>0.053</td><td>0.083</td><td>0.045</td><td>0.078</td><td>0.073</td><td>0.087</td><td>0.085</td><td>0.241</td><td>0.069</td><td>0.087</td><td>0.101</td><td>0.416</td><td>-</td><td>-</td><td>0.095</td><td>0.088</td><td>0.088</td><td>0.084</td><td>0.093</td><td>0.117</td><td>0.100</td><td>0.156</td></tr><tr><td rowspan="3">Taxi</td><td>48</td><td>0.166</td><td>0.264</td><td>0.274</td><td>0.363</td><td>0.217</td><td>0.327</td><td>0.243</td><td>0.330</td><td>0.272</td><td>0.391</td><td>0.234</td><td>0.338</td><td>0.246</td><td>0.617</td><td>-</td><td>-</td><td>0.264</td><td>0.348</td><td>0.236</td><td>0.345</td><td>0.259</td><td>0.368</td><td>0.276</td><td>0.388</td></tr><tr><td>96</td><td>0.356</td><td>0.513</td><td>0.473</td><td>0.577</td><td>0.379</td><td>0.528</td><td>0.469</td><td>0.534</td><td>0.491</td><td>0.590</td><td>0.371</td><td>0.542</td><td>0.481</td><td>0.849</td><td>-</td><td>-</td><td>0.488</td><td>0.571</td><td>0.464</td><td>0.563</td><td>0.476</td><td>0.607</td><td>0.617</td><td>0.625</td></tr><tr><td>144</td><td>0.465</td><td>0.538</td><td>0.536</td><td>0.724</td><td>0.485</td><td>0.703</td><td>0.517</td><td>0.706</td><td>0.532</td><td>0.915</td><td>0.483</td><td>0.712</td><td>0.527</td><td>1.124</td><td>-</td><td>-</td><td>0.515</td><td>0.717</td><td>0.522</td><td>0.726</td><td>0.559</td><td>0.774</td><td>0.664</td><td>0.815</td></tr><tr><td></td><td>Avg</td><td>0.329</td><td>0.438</td><td>0.428</td><td>0.555</td><td>0.360</td><td>0.519</td><td>0.410</td><td>0.523</td><td>0.432</td><td>0.632</td><td>0.363</td><td>0.531</td><td>0.418</td><td>0.863</td><td>-</td><td>-</td><td>0.422</td><td>0.545</td><td>0.407</td><td>0.545</td><td>0.431</td><td>0.583</td><td>0.519</td><td>0.609</td></tr><tr><td rowspan="2">Wikipedia</td><td>48</td><td>0.073</td><td>0.095</td><td>0.063</td><td>0.086</td><td>0.066</td><td>0.093</td><td>0.074</td><td>0.090</td><td>0.091</td><td>0.142</td><td>0.077</td><td>0.103</td><td>0.112</td><td>1.625</td><td>-</td><td>-</td><td>0.081</td><td>0.102</td><td>0.084</td><td>0.111</td><td>0.083</td><td>0.109</td><td>0.092</td><td>0.107</td></tr><tr><td>96</td><td>0.074</td><td>0.126</td><td>0.086</td><td>0.153</td><td>0.080</td><td>0.137</td><td>0.086</td><td>0.143</td><td>0.116</td><td>0.191</td><td>0.093</td><td>0.146</td><td>0.187</td><td>2.234</td><td>-</td><td>-</td><td>0.119</td><td>0.194</td><td>0.107</td><td>0.148</td><td>0.105</td><td>0.163</td><td>0.131</td><td>0.160</td></tr><tr><td></td><td>Avg</td><td>0.073</td><td>0.110</td><td>0.074</td><td>0.120</td><td>0.073</td><td>0.115</td><td>0.080</td><td>0.116</td><td>0.104</td><td>0.167</td><td>0.085</td><td>0.125</td><td>0.150</td><td>1.929</td><td>-</td><td>-</td><td>0.100</td><td>0.148</td><td>0.095</td><td>0.130</td><td>0.094</td><td>0.136</td><td>0.111</td><td>0.135</td></tr><tr><td colspan="2"> $1^{st}Count$ </td><td>16</td><td>16</td><td>1</td><td>1</td><td>2</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 1: Model performance comparisons on the test set CRPS:CRPS $_{sum}$ , NRMSE:NRMSE $_{sum}$ (lower is better) show baselines and our HDT model. – marks out-of-memory failures. Trans-MAF stands for Transformer-MAF. The underlined ones as the second best.

<table><tr><td>Datasets</td><td colspan="4">Electricity</td><td colspan="4">Traffic</td><td colspan="4">Taxi</td></tr><tr><td>Lengths</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">48</td><td colspan="2">96</td></tr><tr><td>Metrics</td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td></tr><tr><td>C-Transformer</td><td>0.327(.004)</td><td>0.532(.018)</td><td>0.212(.007)</td><td>0.228(.014)</td><td>0.467(.011)</td><td>0.845(.009)</td><td>1.004(.005)</td><td>1.150(.012)</td><td>0.861(.006)</td><td>1.121(.013)</td><td>0.977(.008)</td><td>1.118(.011)</td></tr><tr><td>HDT</td><td>0.025(.002)</td><td>0.030(.002)</td><td>0.028(.001)</td><td>0.032(.003)</td><td>0.034(.001)</td><td>0.060(.004)</td><td>0.037(.003)</td><td>0.063(.005)</td><td>0.166(.005)</td><td>0.264(.003)</td><td>0.356(.002)</td><td>0.513(.007)</td></tr></table>

Table 2: Performance of HDT with Continuous Transformer structure C-Trasformer, which does not include the quantization layer in the stage 1 and replace the discrete token sequences of HDT with continuous representation from stage 1.

![](images/15b1d9e0c15bdd5f5f4c63c494575c95940ea962ef1ad87b069576a87001aed9.jpg)

Figure 2: Performance of HDT with different temperature levels of different prediction lengths in Traffic and Taxi datasets. The comparison results against MG\_TSD and VQ-TR with HDT on different levels of missing rate. 

<table><tr><td>Datasets</td><td colspan="6">Traffic</td><td colspan="6">Taxi</td></tr><tr><td>Lengths</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">144</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">144</td></tr><tr><td>Metrics</td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td></tr><tr><td>HDT-h $_c$ </td><td>0.042(.003)</td><td>0.069(.005)</td><td>0.046(.002)</td><td>0.077(.006)</td><td>0.056(.002)</td><td>0.084(.003)</td><td>0.206(.004)</td><td>0.316(.005)</td><td>0.373(.006)</td><td>0.548(.010)</td><td>0.481(.004)</td><td>0.573(.006)</td></tr><tr><td>HDT-h $_d$ *</td><td>0.038(.002)</td><td>0.065(.008)</td><td>0.039(.004)</td><td>0.067(.004)</td><td>0.050(.004)</td><td>0.078(.003)</td><td>0.189(.004)</td><td>0.278(.004)</td><td>0.371(.004)</td><td>0.537(.005)</td><td>0.480(.005)</td><td>0.568(.011)</td></tr><tr><td>HDT-h $_d$ </td><td>0.034(.001)</td><td>0.060(.004)</td><td>0.037(.003)</td><td>0.063(.005)</td><td>0.048(.005)</td><td>0.079(.006)</td><td>0.166(.003)</td><td>0.264(.005)</td><td>0.356(.002)</td><td>0.513(.004)</td><td>0.467(.006)</td><td>0.540(.007)</td></tr><tr><td>HDT-h $_dc$ </td><td>0.036(.003)</td><td>0.062(.006)</td><td>0.037(.001)</td><td>0.064(.002)</td><td>0.047(.004)</td><td>0.076(.008)</td><td>0.171(.005)</td><td>0.266(.003)</td><td>0.356(.004)</td><td>0.517(.009)</td><td>0.465(.003)</td><td>0.537(.008)</td></tr></table>

Table 3: Performance of HDT with Different Types of contextual conditions. Bold numbers represent the best outcomes and the underlined ones as the second best.

petitive. This suggests that within a probabilistic framework,

discrete representations, serving as an approximate expres-

![](images/c1eaadcd72db3750eea3887cc13f52cb0c072806a6733739f7c66d181a406f81.jpg)

<details>
<summary>bar</summary>

|        | HDT-var.T | HDT-var.L | HDT   |
| ------ | --------- | --------- | ----- |
| Pred48 | 0.235     | 0.231     | 0.167 |
| Pred96 | 0.467     | 0.446     | 0.356 |
| Pred144| 0.586     | 0.544     | 0.465 |
</details>

![](images/d4fb8ae3d94ca03db6f9fb56c423644f5e515d696dd5c5f1d3bdb542e11509c2.jpg)

<details>
<summary>bar</summary>

| Model   | NRMSE_sum |
| ------- | --------- |
| Pred48  | 0.352     |
| Pred48  | 0.354     |
| Pred48  | 0.317     |
| Pred96  | 0.592     |
| Pred96  | 0.607     |
| Pred96  | 0.513     |
| Pred144 | 0.692     |
| Pred144 | 0.664     |
| Pred144 | 0.576     |
</details>

![](images/8c1987c190e317616ec33f38bd185d83d66ac33250b6bd3481107ecafa240fc5.jpg)

<details>
<summary>bar</summary>

|          | HDT-var.T | HDT-var.L | HDT   |
| -------- | --------- | --------- | ----- |
| Pred48   | 0.05      | 0.054     | 0.034 |
| Pred96   | 0.064     | 0.067     | 0.037 |
| Pred144  | 0.103     | 0.091     | 0.047 |
</details>

![](images/ce6400b925e4f0be6c98b0399b81c830502314008d0c9be8a76afc4dc7fa5c78.jpg)

<details>
<summary>bar</summary>

| Model   | NRMSE_sum |
|---------|-----------|
| Pred48  | 0.079     |
| Pred48  | 0.073     |
| Pred48  | 0.06      |
| Pred96  | 0.089     |
| Pred96  | 0.092     |
| Pred96  | 0.063     |
| Pred144 | 0.146     |
| Pred144 | 0.124     |
| Pred144 | 0.074     |
</details>

Figure 3: Probabilistic and deterministic performance of HDT and HDT-variants on different prediction length and datasets. HDT-var.T is the same structure with HDT without the self-conditions in stage 2. HDT-var.L replaces the Transformer with LSTM in stage 2 and without self-conditions. 

<table><tr><td>Datasets</td><td colspan="6">Traffic</td><td colspan="6">Taxi</td></tr><tr><td>Lengths</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">144</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">144</td></tr><tr><td>Metrics</td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td></tr><tr><td>2</td><td>0.036(.003)</td><td>0.070(.005)</td><td>0.044(.002)</td><td>0.079(.007)</td><td>0.061(.004)</td><td>0.094(.008)</td><td>0.226(.005)</td><td>0.338(.006)</td><td>0.373(.007)</td><td>0.530(.012)</td><td>0.542(.005)</td><td>0.714(.012)</td></tr><tr><td>3</td><td>0.036(.005)</td><td>0.073(.003)</td><td>0.037(.003)</td><td>0.063(.005)</td><td>0.049(.003)</td><td>0.080(.006)</td><td>0.166(.003)</td><td>0.264(.005)</td><td>0.378(.005)</td><td>0.569(.009)</td><td>0.530(.008)</td><td>0.635(.014)</td></tr><tr><td>4</td><td>0.034(.001)</td><td>0.060(.004)</td><td>0.039(.004)</td><td>0.067(.006)</td><td>0.047(.004)</td><td>0.076(.008)</td><td>0.172(.002)</td><td>0.277(.007)</td><td>0.356(.002)</td><td>0.513(.004)</td><td>0.465(.003)</td><td>0.537(.008)</td></tr><tr><td>5</td><td>0.037(.002)</td><td>0.075(.003)</td><td>0.363(.003)</td><td>0.524(.005)</td><td>0.052(.002)</td><td>0.082(.005)</td><td>0.173(.003)</td><td>0.268(.004)</td><td>0.363(.004)</td><td>0.526(.007)</td><td>0.476(.007)</td><td>0.578(.011)</td></tr></table>

Table 4: Performance of HDT with Transformer layers under different prediction lengths on Traffic (Stationary) and Taxi (Non-stationary). We report mean&stdev.results of 3 runs.

sion, can be seen as a “Clustering” result that is more resilient to stochastic changes. By incorporating target trends, HDT can achieve a higher level of deterministic forecasting performance.

Effect of Discrete Self-Condition $s_{down}$ in Eqn. (10). From Figure 3, we have: i) For the short-term prediction length (e.g.48) of two datasets, both HDT-var.T and HDT-var.L show marginal differences between HDT, implying the effectiveness of discrete representations. In contrast, these variants show obvious differences between HDT of 96 and 144 settings, which further verifies the merits of our self-conditioned strategy.ii) Discrete features demonstrate stable performance in relatively steady dataset(e.g.Traffic), without significant declines as the forecasting horizon extends. However, in non-stationary dataset(e.g.Taxi), it still exhibits notable performance fluctuations of discrete representations, which implies the effectiveness of our self-condition strategy.

Effect of Missing Ratios in Eqn. (13). To evaluate HDT's robustness, we implemented a timestamp masking strategy, allowing the network to infer representations under incomplete contexts. We randomly masked observations (historical sequences) in the test sets of the Traffic (pred 96) and Taxi (pred 48) datasets at designated missing rates. Figure 2 illustrates that excluding the target condition from the forecasting model leads to a rapid decline in probabilistic performance as the missing rate increases in two diffusion models. From the Taxi dataset, with the missing rate of historical conditions nearing $100\%$ , HDT's performance remains largely unaffected, in contrast to the obvious performance degradation observed in the other two history-conditioned diffusion models.

Effect of Temperature Levels in Inference. During our experiments, we observed that sampling temperature is a crucial hyperparameter in a probabilistic setting. As shown in Figure 2, tests on the Traffic and Taxi datasets revealed significant differences in results with varying temperatures. As for the Traffic dataset, a slightly higher temperature improved probabilistic forecasting performance, while a substantial increase led to model bias. For the Taxi dataset, we found that a moderate temperature is optimal, with no significant change in short-term accuracy at higher temperatures compared to long-term settings. This suggests that HDT can achieve better results by adjusting temperature variations to suit different datasets and forecasting lengths.

Effect of Number of Layers in Eqn. (13). To investigate the effect of the self-cond Transformer layers in Eqn. (13), we report the CRPS $_{sum}$ and NRMSE $_{sum}$ results of our SDT with different number of layers (e.g.2, 3, 4, 5) in Table 4. We observe that in short-term forecasting, a smaller number of layers (e.g., 2, 3) shows competitive results in both datasets. As the forecast length increases, Traffic exhibits superior performance with a moderately increased number of layers, while high-stochastic Taxi excels in deeper Transformer structures. These experimental results were all conducted under the condition that the base Transformer decoder layers in Eqn. 11 are fixed at 3.

# Conclusion

In this paper, we propose a hierarchical self-conditioned discrete method HDT to enhance high-dimensional multivariate time series (MTS) forecasting. Our novel two-stage vector quantized generative framework maps targets into discrete token representations, capturing target trends for long-term forecasting. To the best of our knowledge, this is the first discrete Transformer architecture applied to high-dimensional, long-term forecasting tasks. Extensive experiments on benchmark datasets demonstrate the effectiveness of our approach. Future research will explore integrating multimodal data into MTS forecasting.

# Acknowledgments

This research is supported by the Joint NTU-WeBank Research Centre on Fintech, Nanyang Technological University, Singapore.

# References

Alcaraz, J. M. L.; and Strodthoff, N. 2022. Diffusion-based time series imputation and forecasting with structured state space models. arXiv preprint arXiv:2208.09399.   
Bian, Y.; Ju, X.; Li, J.; Xu, Z.; Cheng, D.; and Xu, Q. 2024. Multi-patch prediction: Adapting llms for time series representation learning. arXiv preprint arXiv:2402.04852.   
Brophy, E.; Wang, Z.; She, Q.; and Ward, T. 2023. Generative adversarial networks in time series: A systematic literature review. ACM Computing Surveys, 55(10): 1–31.   
Chang, H.; Zhang, H.; Barber, J.; Maschinot, A.; Lezama, J.; Jiang, L.; Yang, M.-H.; Murphy, K.; Freeman, W. T.; Rubinstein, M.; et al. 2023. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704.   
Chen, X.; Mishra, N.; Rohaninejad, M.; and Abbeel, P. 2018. Pixelsnail: An improved autoregressive generative model. In International Conference on Machine Learning, 864–872. PMLR.   
Desai, A.; Freeman, C.; Wang, Z.; and Beaver, I. 2021. Timevae: A variational auto-encoder for multivariate time series generation. arXiv preprint arXiv:2111.08095.   
Dinh, L.; Sohl-Dickstein, J.; and Bengio, S. 2016. Density estimation using real nvp. arXiv preprint arXiv:1605.08803.   
Dong, E.; Du, H.; and Gardner, L. 2020. An interactive web-based dashboard to track COVID-19 in real time. The Lancet infectious diseases, 20(5): 533–534.   
Esser, P.; Rombach, R.; and Ommer, B. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.   
Fan, X.; Wu, Y.; Xu, C.; Huang, Y.; Liu, W.; and Bian, J. 2024. MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process. arXiv preprint arXiv:2403.05751.   
Feng, S.; Miao, C.; Xu, K.; Wu, J.; Wu, P.; Zhang, Y.; and Zhao, P. 2023. Multi-scale attention flow for probabilistic time series forecasting. IEEE Transactions on Knowledge and Data Engineering.   
Feng, S.; Miao, C.; Zhang, Z.; and Zhao, P. 2024. Latent diffusion transformer for probabilistic time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, 11979–11987.   
Feng, S.; Xu, C.; Zuo, Y.; Chen, G.; Lin, F.; and XiaHou, J. 2022. Relation-aware dynamic attributed graph attention network for stocks recommendation. Pattern Recognition, 121: 108119.   
Gasthaus, J.; Benidis, K.; Wang, Y.; Rangapuram, S. S.; Salinas, D.; Flunkert, V.; and Januschowski, T. 2019. Probabilistic forecasting with spline quantile function RNNs. In The

22nd international conference on artificial intelligence and statistics, 1901–1910. PMLR.   
Gray, R. 1984. Vector quantization. IEEE Assp Magazine, 1(2): 4–29.   
Han, X.; Zheng, H.; and Zhou, M. 2022. Card: Classification and regression diffusion models. Advances in Neural Information Processing Systems, 35: 18100–18115.   
Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.   
Hyndman, R.; Koehler, A. B.; Ord, J. K.; and Snyder, R. D. 2008. Forecasting with exponential smoothing: the state space approach. Springer Science & Business Media.   
Isola, P.; Zhu, J.-Y.; Zhou, T.; and Efros, A. A. 2017. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1125–1134.   
Jin, M.; Koh, H. Y.; Wen, Q.; Zambon, D.; Alippi, C.; Webb, G. I.; King, I.; and Pan, S. 2023. A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection. arXiv preprint arXiv:2307.03759.   
Kim, T.; Kim, J.; Tae, Y.; Park, C.; Choi, J.-H.; and Choo, J. 2021. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations.   
Kingma, D. P.; and Welling, M. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.   
Kollovieh, M.; Ansari, A. F.; Bohlke-Schneider, M.; Zschiegner, J.; Wang, H.; and Wang, Y. 2023. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting. arXiv preprint arXiv:2307.11494.   
Kollovieh, M.; Ansari, A. F.; Bohlke-Schneider, M.; Zschiegner, J.; Wang, H.; and Wang, Y. B. 2024. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting. Advances in Neural Information Processing Systems, 36.   
Lai, G.; Chang, W.-C.; Yang, Y.; and Liu, H. 2018. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, 95–104.   
Lee, D.; Malacarne, S.; and Aune, E. 2023. Vector Quantized Time Series Generation with a Bidirectional Prior Model. arXiv preprint arXiv:2303.04743.   
Li, Y.; Chen, W.; Hu, X.; Chen, B.; Zhou, M.; et al. 2024. Transformer-Modulated Diffusion Models for Probabilistic Multivariate Time Series Forecasting. In The Twelfth International Conference on Learning Representations.   
Li, Y.; Lu, X.; Wang, Y.; and Dou, D. 2022. Generative time series forecasting with diffusion, denoise, and disentanglement. Advances in Neural Information Processing Systems, 35: 23009–23022.   
Liu, C.; Hoi, S. C.; Zhao, P.; and Sun, J. 2016. Online arima algorithms for time series prediction. In Proceedings of the AAAI conference on artificial intelligence.

Liu, Y.; Hu, T.; Zhang, H.; Wu, H.; Wang, S.; Ma, L.; and Long, M. 2023. itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625.   
Liu, Y.; Wu, H.; Wang, J.; and Long, M. 2022. Nonstationary transformers: Exploring the stationarity in time series forecasting. Advances in Neural Information Processing Systems, 35: 9881–9893.   
Matheson, J. E.; and Winkler, R. L. 1976. Scoring rules for continuous probability distributions. Management science, 22(10): 1087–1096.   
Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730.   
Papamakarios, G.; Pavlakou, T.; and Murray, I. 2017. Masked autoregressive flow for density estimation. Advances in neural information processing systems, 30.   
Qiu, M.; Zhao, P.; Zhang, K.; Huang, J.; Shi, X.; Wang, X.; and Chu, W. 2017. A short-term rainfall prediction model using multi-task convolutional neural networks. In 2017 IEEE international conference on data mining (ICDM), 395–404. IEEE.   
Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021. Zero-shot text-to-image generation. In International Conference on Machine Learning, 8821–8831. PMLR.   
Rao, H.; Li, Y.; and Miao, C. 2022. Revisiting k-reciprocal distance re-ranking for skeleton-based person re-identification. IEEE Signal Processing Letters, 29: 2103–2107.   
Rao, H.; Xu, S.; Hu, X.; Cheng, J.; and Hu, B. 2021. Multi-Level Graph Encoding with Structural-Collaborative Relation Learning for Skeleton-Based Person Re-Identification. In Zhou, Z.-H., ed., Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, 973–980. International Joint Conferences on Artificial Intelligence Organization. Main Track.   
Rasul, K.; Bennett, A.; Vicente, P.; Gupta, U.; Ghonia, H.; Schneider, A.; and Nevmyvaka, Y. 2023. VQ-TR: Vector Quantized Attention for Time Series Forecasting. In The Twelfth International Conference on Learning Representations.   
Rasul, K.; Bennett, A.; Vicente, P.; Gupta, U.; Ghonia, H.; Schneider, A.; and Nevmyvaka, Y. 2024. VQ-TR: Vector Quantized Attention for Time Series Forecasting. In The Twelfth International Conference on Learning Representations.   
Rasul, K.; Seward, C.; Schuster, I.; and Vollgraf, R. 2021. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International Conference on Machine Learning, 8857–8868. PMLR.   
Rasul, K.; Sheikh, A.-S.; Schuster, I.; Bergmann, U.; and Vollgraf, R. 2020. Multivariate probabilistic time series forecasting via conditioned normalizing flows. arXiv preprint arXiv:2002.06103.

Razavi, A.; Van den Oord, A.; and Vinyals, O. 2019. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32.   
Salinas, D.; Bohlke-Schneider, M.; Callot, L.; Medico, R.; and Gasthaus, J. 2019. High-dimensional multivariate forecasting with low-rank gaussian copula processes. Advances in neural information processing systems, 32.   
Salinas, D.; Flunkert, V.; Gasthaus, J.; and Januschowski, T. 2020. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3): 1181–1191.   
Sezer, O. B.; Gudelek, M. U.; and Ozbayoglu, A. M. 2020. Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied soft computing, 90: 106181.   
Shen, L.; and Kwok, J. 2023. Non-autoregressive Conditional Diffusion Models for Time Series Prediction. arXiv preprint arXiv:2306.05043.   
Tashiro, Y.; Song, J.; Song, Y.; and Ermon, S. 2021. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in Neural Information Processing Systems, 34: 24804–24816.   
Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. Advances in neural information processing systems, 30.   
Woo, G.; Liu, C.; Kumar, A.; Xiong, C.; Savarese, S.; and Sahoo, D. 2024. Unified training of universal time series forecasting transformers. arXiv preprint arXiv:2402.02592.   
Wu, H.; Xu, J.; Wang, J.; and Long, M. 2021. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34: 22419–22430.   
Wu, S.; Xiao, X.; Ding, Q.; Zhao, P.; Wei, Y.; and Huang, J. 2020. Adversarial sparse transformer for time series forecasting. Advances in neural information processing systems, 33: 17105–17115.   
Yoon, J.; Jarrett, D.; and Van der Schaar, M. 2019. Time-series generative adversarial networks. Advances in neural information processing systems, 32.   
Yu, J.; Li, X.; Koh, J. Y.; Zhang, H.; Pang, R.; Qin, J.; Ku, A.; Xu, Y.; Baldridge, J.; and Wu, Y. 2021. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627.   
Zheng, C.; Vuong, T.-L.; Cai, J.; and Phung, D. 2022. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems, 35: 23412–23425.   
Zhicheng, C.; SHIBO, F.; Zhang, Z.; Xiao, X.; Gao, X.; and Zhao, P. 2024. SDformer: Similarity-driven Discrete Transformer For Time Series Generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.   
Zhou, T.; Niu, P.; Sun, L.; Jin, R.; et al. 2023. One fits all: Power general time series analysis by pretrained lm. Advances in neural information processing systems, 36:43322–43355.

# Appendix

In the supplementary, we provide more implementation details, more experimental results, and visualization of test samples of our HDT. We organize our supplementary as follows

- In Section A, we give the Related Work of HDT, including vector quantization-based frameworks and deep generative model-based MTS two parts.   
- In Section B, we provide more details of used datasets and metrics in our experiment.   
- In Section C, we provide the experiment setup, including the hyperparameters and detailed structures of stage 1, 2 frameworks.   
- In Section D, we draw the comparison of the memory usage and inference time between HDT and other strong baselines, highlighting the source-efficient and efficiency.   
- In Section E, we show more experimental results with obvious non-stationary datasets Hospital (Hyndman et al. 2008) and COVID Deaths (Dong, Du, and Gardner 2020) with different prediction lengths {24, 48, 96} to compare the prediction performance with state-of-the-art generative methods.   
- In Section F, we provide the details of baselines in our main experiment.   
- In Section G, we summarize the limitations and showcase more test samples on seven MTS datasets.

# A. Related work

# Vector quantization-based frameworks

Unlike many deep learning methods directly focusing on the continuous data domains, Vector Quantization-based frameworks map complex continuous domains into finite discrete domains. VQVAE (Van Den Oord, Vinyals et al. 2017; Razavi, Van den Oord, and Vinyals 2019) decomposes the image generation process into two parts: initially, it trains a vector quantized autoencoder aimed at image reconstruction, transforming images into a compressed sequence of discrete tokens. Then the second stage learns an autoregressive model, e.g., PixelSNAIL (Chen et al. 2018), to model the underlying distribution of token sequences. Driven by the effectiveness of VQVAE and progress in sequence modeling, many approaches follow the two-stage paradigm. DALL-E (Ramesh et al. 2021) improves token prediction in the second stage by using Transformers, resulting in a strong text-to-image synthesis model. VQGAN (Esser, Rombach, and Ommer 2021; Zheng et al. 2022; Yu et al. 2021) employs adversarial loss during its first stage, training a more efficient autoencoder, which allows for the synthesis of images with greater details.

# Deep generative model-based MTS

To improve the reliability and performance of high-dimensional MTS, instead of modeling the raw data, there exist works inferring the underlying distribution of the time series data with deep generative models (Yoon, Jarrett, and Van der Schaar 2019; Brophy et al. 2023). Normalizing flow (Papamakarios, Pavlakou, and Murray 2017; Dinh, Sohl-Dickstein, and Bengio 2016) based MTS framework, e.g., MAF (Rasul et al. 2020) explicitly models multivariate time series and their temporal dynamics by employing a normalizing flow for probabilistic forecasting. Variational Autoencoder-based models, e.g., Timevae (Desai et al. 2021) a novel architecture with interpretability, can encode domain knowledge, and reduce training times.

Existing diffusion-based (Ho, Jain, and Abbeel 2020) MTS forecasting models can be roughly divided into two categories. The first one is autoregressive, e.g., TimeGrad (Rasul et al. 2021) operates by sequentially generating future predictions over time. Nonetheless, its ability to forecast over long ranges is constrained by the accumulation of errors and a sluggish inference speed. The other is the non-autoregressive diffusion model, such as CSDI (Tashiro et al. 2021), LDT (Feng et al. 2024), SSSD (Alcaraz and Strodthoff 2022), D $^{3}$ VAE (Li et al. 2022), TSDiff (Kollovieh et al. 2023), TimeDiff (Shen and Kwok 2023), MG\_TSD (Fan et al. 2024) and TMDM (Li et al. 2024). These models perform conditioning and unconditioning strategies to train the denoising networks and introduce some guidance strategies to predict the denoising objective more accurately.

# B. Dataset and Metric Details

Dataset. We summarize the dataset details of our MTS long-term forecasting. As shown in Table 5, Solar is the hourly photo-voltaic production of 137 stations in Alabama State; Electricity is the hourly time series of the electricity consumption of 370 customers; Traffic is the hourly occupancy rate, between 0 and 1, of 963 San Francisco car lanes; Taxi is the spatio-temporal half hourly traffic time series of New York taxi rides taken at 1214 locations; Wikipedia (Gasthaus et al. 2019) is the daily page views of 2000 Wikipedia pages. Among them, the Solar shows certain periodic patterns, whereas the others predominantly display non-stationary characteristics.

<table><tr><td>DATASET</td><td>Dimension</td><td>Domain</td><td>Freq</td><td>Total Time Steps</td><td>Context Length</td><td>Pred Length</td></tr><tr><td>Solar</td><td>137</td><td> $\mathbb{R}^{+}$ </td><td>Hourly</td><td>7,009</td><td>96</td><td>{48, 96, 144}</td></tr><tr><td>COVID Deaths</td><td>266</td><td> $\mathbb{R}^{+}$ </td><td>Daily</td><td>212</td><td>96</td><td>{48, 96}</td></tr><tr><td>Electricity</td><td>370</td><td> $\mathbb{R}^{+}$ </td><td>Hourly</td><td>5,790</td><td>96</td><td>{48, 96, 144}</td></tr><tr><td>Hospital</td><td>767</td><td> $\mathbb{R}^{+}$ </td><td>Monthly</td><td>84</td><td>24</td><td>{24, 48}</td></tr><tr><td>Traffic</td><td>963</td><td>(0,1)</td><td>Hourly</td><td>10,413</td><td>96</td><td>{48, 96, 144}</td></tr><tr><td>Taxi</td><td>1214</td><td> $\mathbb{N}$ </td><td>30-Min</td><td>1,488</td><td>96</td><td>{48, 96, 144}</td></tr><tr><td>Wikipedia</td><td>2000</td><td> $\mathbb{N}$ </td><td>Daily</td><td>792</td><td>96</td><td>{48, 96}</td></tr></table>

Table 5: Properties of the datasets in experiments

Metric. CRPS measures the compatibility of a cumulative distribution function $P$ with an observation $x$ as: $\mathrm{CRPS}(\mathcal{F}, x) = \int_{\mathbb{R}} (P(y) - \mathbb{I}\{x \leq y\})^2 dy$ , where $\mathbb{I}\{x \leq y\}$ is the indicator function which is one if $x \leq y$ and zero otherwise. The empirical CDF of $P$ , i.e., $\hat{P}(y) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\{X_i \leq y\}$ with $n$ samples $X_i \sim P$ as the approximation of the predictive CDF. It utilizes $N$ samples to estimate the empirical CDF and take the CRPS-sum in the

<table><tr><td>Datasets</td><td colspan="4">Electricity</td><td colspan="4">Traffic</td><td colspan="4">Taxi</td></tr><tr><td>Lengths</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">48</td><td colspan="2">96</td><td colspan="2">48</td><td colspan="2">96</td></tr><tr><td>Metrics</td><td>QICE ↓</td><td>PICP ↑</td><td>QICE ↓</td><td>PICP ↑</td><td>QICE ↓</td><td>PICP ↑</td><td>QICE ↓</td><td>PICP ↑</td><td>QICE ↓</td><td>PICP ↑</td><td>QICE ↓</td><td>PICP ↑</td></tr><tr><td>TimeGrad</td><td>10.17</td><td>80.16</td><td>12.29</td><td>77.23</td><td>12.36</td><td>88.24</td><td>14.73</td><td>86.39</td><td>6.47</td><td>41.76</td><td>8.76</td><td>38.23</td></tr><tr><td>HDT</td><td>7.74</td><td>83.72</td><td>7.96</td><td>82.94</td><td>4.72</td><td>95.27</td><td>6.39</td><td>96.23</td><td>4.36</td><td>54.03</td><td>6.63</td><td>49.24</td></tr></table>

Table 6: Probabilistic forecasting performance of HDT with TimeGrad on the QICE and PICP, which are popular metrics for probabilistic multivariate time series forecasting models

multivariate case.

$$
\mathrm{CRPS} _ {\text { sum }} = \mathbb {E} _ {t} \left[ \mathrm{CRPS} \left(\widehat {P} _ {\text { sum }} (t), \sum_ {i} x _ {i} ^ {t}\right) \right]. \tag {14}
$$

The Normalized Root Mean Squared Error (NRMSE) is a standardized version of the Root Mean Squared Error (RMSE) that accounts for the scale of the target values. The formula for NRMSE is given below:

$$
\mathrm{NRMSE} = \sqrt {\frac {1}{T} \sum_ {t = 1} ^ {T} \left(\frac {y _ {t} - \hat {y} _ {t}}{y _ {\text { max }} - y _ {\text { min }}}\right) ^ {2}}, \tag {15}
$$

where $\hat{y}_{t}$ represents the predicted target, and $y_{t}$ represents the true target. $y_{max}$ and $y_{min}$ are the minimum and maximum of the measured target values, respectively. The NRMSE quantifies the average squared discrepancy between the predictions and actual observations, normalized by the range of the target values. A lower NRMSE indicates higher predictive accuracy.

# C. Detailed Experiment Setup and Architectures

In this section, we summarize the detailed experiment setup of our HDT. Table 7 shows the hyperparameters of our overall structure in stages 1 and 2. $\{^{*}\}$ represents the hyperparameters used in our experiments. Table 8, 9 and 10 show the detailed modules of our HDT, among these components, Conv1d refers to the 1-d convolution operation, while the Self-Attn Block and Cross-Attn Block represent the standard multi-head self-attention and cross-attention of TransformerDecoderLayer, respectively.

# D. Memory Usage and Model Efficiency

We comprehensively compare the performance of inference time and memory usage of the following models: TimeGrad, SSSD, TimeDiff, TSDiff and MG\_TSD with our efficient discrete framework. The results are recorded with the official model configuration and the same samples numbers=100. In Figure 4, we compare the efficiency under two representative datasets (963 variates in Traffic and 1214 in Taxi) with different forecasting length (Traffic:96, Taxi:48) and same 96 time steps for lookback.

From Figure 4, we observe that HDT demonstrates a significant advantage in memory usage and inference time compared to diffusion models on high-dimensional multivariate time series. It's evident that diffusion-based models tend to increase diffusion steps significantly to enhance prediction performance, especially in high-dimensional data. For example, MG\_TSD introduces the concept of multiple granularities, which not only suffers from the inherent limitations of autoregressive structures but also adds additional inference results from different granularities, thereby further slowing down the inference speed and increasing model size. Non-autoregressive forms like TimeDiff sacrifice some inference accuracy to expedite inference speed. HDT, on the other hand, leverages a compressed discrete structure without relying on a large diffusion framework, which enhances prediction accuracy by utilizing the target itself while ensuring the model is both source-efficient and time-efficient.

# E. More Experiment Results

To validate the advantages of HDT in high-dimensional, non-stationary datasets, we added COVID Deaths and Hospital, as shown in Table 11. From Table 11, we demonstrate significant performance improvements in the complex Hospital dataset. Notably, HDT-var.T, a discrete Transformer without a self-condition strategy, outperforms diffusion-based methods in short-term forecasting settings. However, it struggles to adapt to increased forecast lengths. HDT addresses this issue by leveraging its own trend, confirming its effectiveness in long-term predictions for high-dimensional settings.

Furthermore, to further demonstrate the effectiveness of HDT in probabilistic forecasting of high-dimensional MTS, we introduce two metrics Prediction Interval Coverage Probability (PICP) and Quantile Interval Coverage Error (QICE) from TMDM(Li et al. 2024) and CARD (Han, Zheng, and Zhou 2022), which are defined as follows:

$$
\mathrm{PICP} := \frac {1}{N} \sum_ {n = 1} ^ {N} \mathbb {I} _ {y _ {n} \geq \hat {y} _ {n} ^ {\text { low }}} \cdot \mathbb {I} _ {y _ {n} \leq \hat {y} _ {n} ^ {\text { high }}}, \tag {16}
$$

$$
\mathrm{QICE} := \frac {1}{M} \sum_ {m = 1} ^ {M} \left| r _ {m} - \frac {1}{M} \right|, \tag {17}
$$

where $r_{m} = \frac{1}{N} \sum_{n=1}^{N} I_{y_{n} \geq \hat{y}_{n}^{\text{low} m}} \cdot I_{y_{n} \leq \hat{y}_{n}^{\text{high} m}}$ , $\hat{y}_{n}^{low}$ and $\hat{y}_{n}^{high}$ represent the low and high percentiles, respectively. In our setting, we choose the $2.5^{th}$ and $97.5^{th}$ percentile, thus an ideal PICP value for the learned model should be 95% and we set M = 10, and obtain the following 10 quantile intervals (QIs) of the generated samples (generated sample $\hat{y} \in R^{S \times \tau \times D}$ , target $y \in R^{\tau \times D}$ , S is the sample size=100 of our setting): below the $10^{th}$ percentile, between the $10^{th}$

Stage 1   
Stage 2 

<table><tr><td>Dataset</td><td>Codebook Size</td><td>Codebook dim</td><td>Hidden dim</td><td>Enc/Dec Layers</td><td>Trasformer Layers</td><td>Hidden dim</td><td>History Encoder</td><td>Base Layers</td><td>Self-cond Layers</td><td>Temperature</td></tr><tr><td>Solar</td><td>128</td><td>{64, 128, 128}</td><td>{64, 128, 128}</td><td>3</td><td>2</td><td>{64, 128}</td><td>2</td><td>3</td><td>{3, 4, 5}</td><td>[1.0, 1.5, 2.0, 3.0, 6.0]</td></tr><tr><td>Electricity</td><td>128</td><td>128</td><td>128</td><td>3</td><td>2</td><td>128</td><td>2</td><td>3</td><td>{3, 4, 5}</td><td>[1.0, 2.0, 3.0, 5.0, 8.0]</td></tr><tr><td>Traffic</td><td>128</td><td>256</td><td>256</td><td>3</td><td>2</td><td>256</td><td>2</td><td>3</td><td>{3, 4, 5}</td><td>[1.0, 2.0, 3.0, 5.0, 8.0]</td></tr><tr><td>Taxi</td><td>256</td><td>256</td><td>256</td><td>3</td><td>3</td><td>256</td><td>2</td><td>3</td><td>{3, 4, 5}</td><td>[1.0, 2.0, 3.0, 5.0, 8.0]</td></tr><tr><td>Wikipedia</td><td>256</td><td>512</td><td>512</td><td>3</td><td>2</td><td>512</td><td>2</td><td>3</td><td>{3, 4, 5}</td><td>[1.0, 2.0, 3.0, 5.0, 8.0]</td></tr></table>

Table 7: Detailed hyperparameters of stages 1 and 2.

![](images/a8ecc4aed9885724440a747c7e5a451fd4aeb77242a31b92f991ad00842c7622.jpg)  
Figure 4: Model memory usage and time efficiency comparison under input-96-predict-48, 96 of Traffic and Taxi, respectively.

and $20^{th}$ percentiles, ..., between the $80^{th}$ and $90^{th}$ percentiles, and above the $90^{th}$ percentile. From the Table 6, we observe that HDT show competitive performance comparing to the TimeGrad, which demonstrate the effectiveness of HDT on the probabilistic forecasting setting.

# F. Details of baselines

In our experiments, we compared HDT against 5 types of models, which are shown as follows.

1. Gaussian process based model

\- GP-Copula (Salinas et al. 2019): It employs a separate LSTM unrolling for each time series, and models the joint emission distribution using a Gaussian copula with a low-rank plus diagonal covariance structure.

2. Probabilistic Deep Learning model

\- DeepAR (Salinas et al. 2020): A probabilistic model based on RNNs that learns the distribution parameters for predicting the next time point.

3. Normalizing flow based models

\- Transformer-MAF (Rasul et al. 2020): Replace the LSTM of the LSTM-MAF with the Transformer.

4. Diffusion based models

- TimeGrad (Rasul et al. 2021): An auto-regressive model based on the diffusion model, which is used for generating each timestamp value autoregressively.   
- CSDI (Rasul et al. 2020): A two types of Transformer based non-autoregressive diffusion model for generating multivariate time series.   
- D $^{3}$ VAE (Li et al. 2022): A coupled diffusion probabilistic model with bidirectional variational auto-encoder (BVAE) for time series generation.

- SSSD (Alcaraz and Strodthoff 2022): Replaces the transformers in CSDI by a structured state space model to avoid the quadratic complexity issue with non-autoregressive way.   
- TimeDiff (Shen and Kwok 2023): A non-autoregressive diffusion model with future mixup and autoregressive initialization strategies for multivariate time series forecasting.   
- TSDiff (Kollovieh et al. 2023): An unconditional diffusion model with self-guidance strategy for probabilistic time series forecasting

5. Discrete vector quantization models

\- VQ-TR (Rasul et al. 2023): Map large sequences to a discrete set of latent representations as part of the Attention module for time series forecasting.

# G. Limitations and Visualizations

In the section, we summarize the limitation of this work and showcase ground-truths and generations on the five datasets of our main experiment, as shown in Fig. 6 to Fig. 11.

Limitations: In HDT, we need to train a separate codebook for each dataset, as we have not yet achieved the discretization of all datasets under a unified codebook setup. It is important to note that due to significant distribution differences between various time series, discretizing all datasets with a single codebook is highly challenging. In the future, we plan to draw on the approach of MOIRAI (Woo et al. 2024) to construct a unified discretized representation based on the unified time series modeling approach.

Visualization: We showcase the visualized results cross five datasets with the corresponding prediction length used in our main experiment, which are shown in Fig. 6 to Fig. 11.

![](images/3f0873ff661aad432c353459fb852315bb63134ece41e9827cb149749634e3f2.jpg)

Figure 5: Comparison of prediction intervals with TiemGrad and MG\_TSG for the Taxi dataset, which comprise 1214 dimensions. The predicted median is displayed, along with visualization of the 50% and 90% distribution intervals. The blue line in the graph represents the ground truth of the test sample. 

<table><tr><td>Layer</td><td>Function</td><td>Descriptions</td></tr><tr><td>1</td><td>Convolution</td><td>input channel=H, output channel=D, kernel size=4, stride=2, padding=1</td></tr><tr><td>2</td><td>ReLU</td><td>nn.ReLU()</td></tr><tr><td>3</td><td>Dropout</td><td>nn.Dropout(p=0.1)</td></tr><tr><td>4</td><td>LayerNorm</td><td>nn.LayerNorm()</td></tr><tr><td>5</td><td>Convolution</td><td>input channel=D, output channel=D, kernel size=3, stride=1, padding=1</td></tr><tr><td>6</td><td>ReLU</td><td>nn.ReLU()</td></tr><tr><td>7</td><td>Dropout</td><td>nn.Dropout(p=0.1)</td></tr><tr><td>8</td><td>LayerNorm</td><td>nn.LayerNorm()</td></tr><tr><td>9</td><td>Convolution</td><td>input channel=D, output channel=D, kernel size=3, stride=1, padding=1</td></tr><tr><td>10</td><td>Tanh</td><td>nn.Tanh()</td></tr></table>

Table 8: The detailed architecture of the Conv-Enc.

<table><tr><td>Layer</td><td>Function</td><td>Descriptions</td></tr><tr><td>1</td><td>DeConvolution</td><td>input channel=D, output channel=D, kernel size=3, stride=1, padding=1</td></tr><tr><td>2</td><td>ReLU</td><td>nn.ReLU()</td></tr><tr><td>3</td><td>Dropout</td><td>nn.Dropout(p=0.1)</td></tr><tr><td>4</td><td>LayerNorm</td><td>nn.LayerNorm()</td></tr><tr><td>5</td><td>DeConvolution</td><td>input channel=D, output channel=D, kernel size=3, stride=1, padding=1</td></tr><tr><td>6</td><td>ReLU</td><td>nn.ReLU()</td></tr><tr><td>7</td><td>Dropout</td><td>nn.Dropout(p=0.1)</td></tr><tr><td>8</td><td>LayerNorm</td><td>nn.LayerNorm()</td></tr><tr><td>9</td><td>DeConvolution</td><td>input channel=D, output channel=H, kernel size=4, stride=2, padding=1</td></tr><tr><td>1</td><td>Layernorm</td><td>nn.LayerNorm()</td></tr><tr><td>2</td><td>Self-attention</td><td>Attention(q=x, k=x, v=x)</td></tr><tr><td>3</td><td>Cross-attention</td><td>Attention(q=x, k=history, v=history)</td></tr><tr><td>4</td><td>Self-condition attention</td><td>Attention(q=x, k=downsampled x, v=downsampled x)</td></tr><tr><td>5</td><td>Layernorm</td><td>nn.LayerNorm()</td></tr><tr><td>6</td><td>MLP</td><td>nn.Linear()</td></tr><tr><td>7</td><td>ReLU</td><td>nn.ReLU()</td></tr><tr><td>8</td><td>MLP</td><td>nn.Linear()</td></tr></table>

Table 9: The detailed architecture of the DeConv-Dec.

Table 10: The detailed architecture of the Self-Transformer block.

<table><tr><td>Datasets</td><td colspan="4">Hospital</td><td colspan="4">Covid Deaths</td></tr><tr><td>Lengths</td><td colspan="2">inputs:24-forecast:24</td><td colspan="2">inputs:24-forecast:48</td><td colspan="2">inputs:48-forecast:48</td><td colspan="2">inputs:96-forecast:96</td></tr><tr><td>Metrics</td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td><td> $CRPS_{sum}$ </td><td> $NRMSE_{sum}$ </td></tr><tr><td>TSDiff</td><td>0.059(.002)</td><td>0.085(.001)</td><td>0.089(.001)</td><td>0.142(.000)</td><td>0.167(.012)</td><td>0.196(.024)</td><td>0.224(.016)</td><td>0.248(.014)</td></tr><tr><td>MG_TSD</td><td>0.051(.001)</td><td>0.074(.000)</td><td>0.094(.001)</td><td>0.158(.001)</td><td>0.154(.008)</td><td>0.172(.014)</td><td>0.207(.013)</td><td>0.231(.009)</td></tr><tr><td>HDT-var.T</td><td>0.037(.003)</td><td>0.047(.002)</td><td>0.072(.003)</td><td>0.091(.001)</td><td>0.134(.008)</td><td>0.169(.011)</td><td>0.211(.008)</td><td>0.236(.014)</td></tr><tr><td>HDT (ours)</td><td>0.035(.001)</td><td>0.043(.001)</td><td>0.057(.002)</td><td>0.066(.001)</td><td>0.127(.006)</td><td>0.165(.008)</td><td>0.154(.011)</td><td>0.178(.007)</td></tr><tr><td>improvement ↑</td><td>31% ↑</td><td>41% ↑</td><td>35.2% ↑</td><td>53.7% ↑</td><td>17.2%↑</td><td>4.3% ↑</td><td>25.6% ↑</td><td>22.9% ↑</td></tr></table>

Table 11: Performance of HDT with TSDiff and MG\_TSD on two non-stationary datasets, Hospital and Covid Deaths. Bold numbers represent the best outcomes and the underlined ones as the second best.

<table><tr><td colspan="2">Models</td><td colspan="2">HDT(Ours)</td><td colspan="2">VQ-TR(2024)</td><td colspan="2">MG_TSD(2024)</td><td colspan="2">TSDiff(2024)</td><td colspan="2">TimeDiff(2023)</td><td colspan="2">SSSD(2023)</td><td colspan="2"> $D^3VAE$ (2022)</td><td colspan="2">CSDI(2021)</td><td colspan="2">TimeGrad(2021)</td><td colspan="2">Trans-MAF(2020)</td><td colspan="2">DeepAR(2020)</td><td colspan="2">GP-Copula(2019)</td></tr><tr><td colspan="2">Metric</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td><td>CRPS</td><td>NRMSE</td></tr><tr><td rowspan="3">Solar</td><td>48</td><td>.004</td><td>.007</td><td>.005</td><td>.014</td><td>.006</td><td>.008</td><td>.005</td><td>.007</td><td>.007</td><td>.012</td><td>.003</td><td>.009</td><td>.004</td><td>.008</td><td>.006</td><td>.007</td><td>.008</td><td>.011</td><td>.002</td><td>.009</td><td>.006</td><td>.010</td><td>.014</td><td>.007</td></tr><tr><td>96</td><td>.002</td><td>.006</td><td>.005</td><td>.009</td><td>.004</td><td>.011</td><td>.002</td><td>.011</td><td>.009</td><td>.015</td><td>.001</td><td>.005</td><td>.004</td><td>.014</td><td>.005</td><td>.009</td><td>.004</td><td>.014</td><td>.004</td><td>.012</td><td>.004</td><td>.007</td><td>.020</td><td>.012</td></tr><tr><td>144</td><td>.002</td><td>.005</td><td>.006</td><td>.021</td><td>.004</td><td>.016</td><td>.007</td><td>.014</td><td>.011</td><td>.020</td><td>.002</td><td>.014</td><td>.003</td><td>.011</td><td>.007</td><td>.011</td><td>.011</td><td>.017</td><td>.004</td><td>.015</td><td>.008</td><td>.016</td><td>.023</td><td>.014</td></tr><tr><td></td><td>Avg</td><td>.003</td><td>.006</td><td>.005</td><td>.011</td><td>.005</td><td>.012</td><td>.005</td><td>.011</td><td>.009</td><td>.016</td><td>.002</td><td>.009</td><td>.004</td><td>.011</td><td>.006</td><td>.009</td><td>.011</td><td>.014</td><td>.003</td><td>.012</td><td>.006</td><td>.011</td><td>.019</td><td>.011</td></tr><tr><td rowspan="3">Electricity</td><td>48</td><td>.002</td><td>.002</td><td>.004</td><td>.007</td><td>.002</td><td>.005</td><td>.001</td><td>.004</td><td>.003</td><td>.008</td><td>.003</td><td>.002</td><td>.007</td><td>.014</td><td>.000</td><td>.005</td><td>.002</td><td>.009</td><td>.001</td><td>.006</td><td>.002</td><td>.004</td><td>.004</td><td>.007</td></tr><tr><td>96</td><td>.001</td><td>.003</td><td>.006</td><td>.011</td><td>.003</td><td>.006</td><td>.006</td><td>.012</td><td>.008</td><td>.012</td><td>.005</td><td>.007</td><td>.012</td><td>.017</td><td>.006</td><td>.005</td><td>.004</td><td>.013</td><td>.002</td><td>.006</td><td>.005</td><td>.009</td><td>.006</td><td>.014</td></tr><tr><td>144</td><td>.002</td><td>.007</td><td>.009</td><td>.006</td><td>.004</td><td>.009</td><td>.003</td><td>.007</td><td>.006</td><td>.019</td><td>.007</td><td>.011</td><td>.004</td><td>.021</td><td>.006</td><td>.013</td><td>.002</td><td>.014</td><td>.003</td><td>.011</td><td>.007</td><td>.008</td><td>.004</td><td>.011</td></tr><tr><td></td><td>Avg</td><td>.002</td><td>.004</td><td>.006</td><td>.008</td><td>.003</td><td>.007</td><td>.003</td><td>.008</td><td>.006</td><td>.013</td><td>.005</td><td>.007</td><td>.008</td><td>.017</td><td>.004</td><td>.008</td><td>.003</td><td>.012</td><td>.002</td><td>.011</td><td>.005</td><td>.007</td><td>.005</td><td>.011</td></tr><tr><td rowspan="3">Traffic</td><td>48</td><td>.001</td><td>.004</td><td>.003</td><td>.009</td><td>.003</td><td>.006</td><td>.006</td><td>.012</td><td>.004</td><td>.009</td><td>.003</td><td>.005</td><td>.006</td><td>.034</td><td>-</td><td>-</td><td>.005</td><td>.004</td><td>.001</td><td>.007</td><td>.008</td><td>.005</td><td>.003</td><td>.009</td></tr><tr><td>96</td><td>.003</td><td>.005</td><td>.006</td><td>.007</td><td>.007</td><td>.009</td><td>.003</td><td>.010</td><td>.004</td><td>.007</td><td>.002</td><td>.004</td><td>.005</td><td>.028</td><td>-</td><td>-</td><td>.006</td><td>.012</td><td>.004</td><td>.006</td><td>.005</td><td>.007</td><td>.007</td><td>.010</td></tr><tr><td>144</td><td>.004</td><td>.007</td><td>.005</td><td>.012</td><td>.004</td><td>.011</td><td>.004</td><td>.017</td><td>.005</td><td>.021</td><td>.007</td><td>.012</td><td>.011</td><td>.026</td><td>-</td><td>-</td><td>.006</td><td>.018</td><td>.002</td><td>.011</td><td>.007</td><td>.016</td><td>.007</td><td>.016</td></tr><tr><td></td><td>Avg</td><td>.003</td><td>.005</td><td>.005</td><td>.009</td><td>.005</td><td>.009</td><td>.005</td><td>.013</td><td>.004</td><td>.012</td><td>.004</td><td>.007</td><td>.007</td><td>.029</td><td>-</td><td>-</td><td>.006</td><td>.011</td><td>.002</td><td>.008</td><td>.006</td><td>.009</td><td>.006</td><td>.011</td></tr><tr><td rowspan="3">Taxi</td><td>48</td><td>.003</td><td>.004</td><td>.005</td><td>.011</td><td>.006</td><td>.012</td><td>.005</td><td>.009</td><td>.011</td><td>.014</td><td>.003</td><td>.006</td><td>.003</td><td>.024</td><td>-</td><td>-</td><td>.003</td><td>.009</td><td>.002</td><td>.007</td><td>.004</td><td>.011</td><td>.008</td><td>.007</td></tr><tr><td>96</td><td>.002</td><td>.007</td><td>.004</td><td>.009</td><td>.004</td><td>.013</td><td>.007</td><td>.005</td><td>.012</td><td>.018</td><td>.006</td><td>.009</td><td>.004</td><td>.032</td><td>-</td><td>-</td><td>.002</td><td>.007</td><td>.004</td><td>.008</td><td>.005</td><td>.009</td><td>.009</td><td>.012</td></tr><tr><td>144</td><td>.001</td><td>.010</td><td>.008</td><td>.007</td><td>.005</td><td>.016</td><td>.004</td><td>.007</td><td>.005</td><td>.022</td><td>.006</td><td>.016</td><td>.004</td><td>.019</td><td>-</td><td>-</td><td>.007</td><td>.013</td><td>.005</td><td>.009</td><td>.004</td><td>.007</td><td>.015</td><td>.017</td></tr><tr><td></td><td>Avg</td><td>.002</td><td>.007</td><td>.006</td><td>.009</td><td>.005</td><td>.013</td><td>.006</td><td>.007</td><td>.009</td><td>.018</td><td>.015</td><td>.010</td><td>.004</td><td>.025</td><td>-</td><td>-</td><td>.004</td><td>.010</td><td>.004</td><td>.008</td><td>.005</td><td>.009</td><td>.011</td><td>.012</td></tr><tr><td rowspan="2">Wikipedia</td><td>48</td><td>.004</td><td>.005</td><td>.004</td><td>.006</td><td>.003</td><td>.008</td><td>.003</td><td>.007</td><td>.005</td><td>.011</td><td>.005</td><td>.007</td><td>.003</td><td>.043</td><td>-</td><td>-</td><td>.006</td><td>.015</td><td>.004</td><td>.007</td><td>.004</td><td>.006</td><td>.004</td><td>.009</td></tr><tr><td>96</td><td>.006</td><td>.011</td><td>.007</td><td>.013</td><td>.004</td><td>.012</td><td>.004</td><td>.009</td><td>.008</td><td>.007</td><td>.007</td><td>.011</td><td>.006</td><td>.051</td><td>-</td><td>-</td><td>.007</td><td>.008</td><td>.005</td><td>.006</td><td>.008</td><td>.012</td><td>.009</td><td>.011</td></tr><tr><td></td><td>Avg</td><td>.005</td><td>.008</td><td>.006</td><td>.010</td><td>.004</td><td>.010</td><td>.004</td><td>.008</td><td>.006</td><td>.009</td><td>.006</td><td>.009</td><td>.005</td><td>.047</td><td>-</td><td>-</td><td>.006</td><td>.011</td><td>.005</td><td>.006</td><td>.006</td><td>.009</td><td>.006</td><td>.010</td></tr></table>

Table 12: Model performance variances on the test set CRPS:CRPS $_{sum}$ , NRMSE:NRMSE $_{sum}$ show baselines and our HDT model. – marks out-of-memory failures. Trans-MAF stands for Transformer-MAF.

![](images/dc6c2b753dac1175da0ba6028c2de7e8f1f6bf7437210370dcf41626dbcc51d4.jpg)

<details>
<summary>line</summary>

| Date       | Observations | Median Prediction | 90.0% Prediction Interval | 50.0% Prediction Interval |
| ---------- | ------------ | ----------------- | ------------------------- | ------------------------- |
| Oct 2006   | 180          | 180               | 180                       | 180                       |
| 22         | 180          | 180               | 180                       | 180                       |
| 23         | 180          | 180               | 180                       | 180                       |
| 24         | 400          | 400               | 400                       | 400                       |
| 25         | 400          | 400               | 400                       | 400                       |
</details>

![](images/1e83e420bee063b342310ebf33d495c7048c45c46246ca6a59c26798404c8883.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 120   |
| Oct 22     | 0     |
| Oct 23     | 140   |
| Oct 24     | 260   |
| Oct 25     | 270   |
</details>

![](images/eccf68c0355c7e84f4caaca2993f1164c89c9f0f13e3a9b42f3afc277db5101e.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 22     | 150   |
| Oct 23     | 80    |
| Oct 24     | 190   |
</details>

![](images/1e6f49f0bd1004d4ba56b3163e963a57592e1c6055d6dd0357b7135c2a0a820c.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 22     | 130   |
| Oct 23     | 110   |
| Oct 24     | 160   |
| Oct 25     | 170   |
</details>

![](images/c639ac75fbc83dca0ee5728ed9b607ae6736c96c49b4e9319271eee14889439e.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 0     |
| Oct 22     | 200   |
| Oct 23     | 0     |
| Oct 24     | 330   |
| Oct 25     | 0     |
</details>

![](images/601f381e2618746dccbe51e71802907e08124094b0a739e1cc62f30d2b48296c.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 400   |
| Oct 22     | 300   |
| Oct 23     | 600   |
| Oct 24     | 600   |
| Oct 25     | 600   |
</details>

![](images/809475c41f31d3673f73c308d16acd4742e616d442f08a01497b74bd2ebb2a3c.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 2006   | 0     |
| 22         | 130   |
| 23         | 0     |
| 24         | 175   |
| 25         | 0     |
</details>

![](images/424eeb0bc9978cd3b61ded3536f53f1acc8aec4ad0df9be063607112a223fde9.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 140   |
| Oct 22     | 105   |
| Oct 23     | 160   |
| Oct 24     | 175   |
</details>

![](images/57a8d8ca6303e77907433032a5ee8fbf9beb74fc81d56b7a9c0ae04bcb0a70a5.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 150   |
| Oct 22     | 0     |
| Oct 23     | 125   |
| Oct 24     | 175   |
| Oct 25     | 0     |
</details>

![](images/8b17dfe8d91111ac7bcab979aedc94d4684fc7d64d9c7875cc29bef60704dbc7.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 0     |
| Oct 22     | 150   |
| Oct 23     | 0     |
| Oct 24     | 160   |
| Oct 25     | 0     |
</details>

![](images/7c841ffd13b4f1d48db17d6be0b4f2a2027abff41ed1e55ed0ef0ef2e69c45b3.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 0     |
| Oct 22     | 140   |
| Oct 23     | 0     |
| Oct 24     | 170   |
| Oct 25     | 0     |
</details>

![](images/91dc0a35861dc4293568e565cd5f3c9ddb7c7c4f725436b203f00ff397f641eb.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 80    |
| Oct 22     | 0     |
| Oct 23     | 60    |
| Oct 24     | 140   |
| Oct 25     | 0     |
</details>

![](images/3afde6caf929caa946495e8c7c523a3cb9b2919f2aa0b5eac262d13ac9c4f290.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 0     |
| Oct 22     | 150   |
| Oct 23     | 0     |
| Oct 24     | 160   |
| Oct 25     | 170   |
</details>

![](images/b5bb9e40d97ff505add1104a027d6425682cec4acf462741d9d5b4c8d3f8c672.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 0     |
| Oct 22     | 145   |
| Oct 23     | 0     |
| Oct 24     | 160   |
| Oct 25     | 170   |
</details>

![](images/aa94f20af310ae38cc4eadb52d8da8f2cc9a27ee5c384339fc3a39563da9858d.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 0     |
| Oct 22     | 160   |
| Oct 23     | 130   |
| Oct 24     | 180   |
| Oct 25     | 170   |
</details>

![](images/c987f677e977a2ae4dd4f743b075ab05f33744c5c5b178d47b9fe8644aceba66.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Oct 21     | 0     |
| Oct 22     | 70    |
| Oct 23     | 0     |
| Oct 24     | 130   |
| Oct 25     | 0     |
</details>

Figure 6: The forecasting results of 16 samples from the Solar dataset with input-96-predict-48.

![](images/f5023c300044bf341c7bd0874f2de6b96d7f14c8a433cf5c517220e8b71513c6.jpg)  
Figure 7: The forecasting results of 16 samples from the Electricity dataset with input-96-predict-48.

![](images/cc70aad3caeb3d3ad7e499d289b11b6e253658ca9b805220f0f5a5f63637a22e.jpg)

<details>
<summary>line</summary>

| Date       | observations | median prediction | 90% prediction interval | 50% prediction interval |
| ---------- | ------------ | ----------------- | ----------------------- | ----------------------- |
| Jun 2008   | 0.00         | 0.00              | 0.00                    | 0.00                    |
| 09         | 0.04         | 0.03              | 0.02                    | 0.01                    |
| 10         | 0.01         | 0.02              | 0.01                    | 0.01                    |
| 11         | 0.07         | 0.06              | 0.05                    | 0.04                    |
| 12         | 0.08         | 0.07              | 0.06                    | 0.05                    |
| 13         | 0.06         | 0.05              | 0.04                    | 0.03                    |
| 14         | 0.10         | 0.09              | 0.08                    | 0.07                    |
| 15         | 0.09         | 0.08              | 0.07                    | 0.06                    |
| 16         | 0.11         | 0.10              | 0.09                    | 0.08                    |
</details>

![](images/49ae2a5ccfdc4d84fd2293b08c23f3f6cd84364365903014a9c44330715725e9.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| Jun 09     | 0.3      | 0.3      |
| Jun 10     | 0.3      | 0.3      |
| Jun 11     | 0.3      | 0.3      |
| Jun 12     | 0.6      | 0.6      |
| Jun 13     | 0.8      | 0.4      |
| Jun 14     | 0.3      | 0.4      |
| Jun 15     | 0.5      | 0.5      |
| Jun 16     | 0.3      | 0.4      |
</details>

![](images/46b47921005fd60ae903fc16448e8bbf4f5d02542a93da8c5ce8186e2faae2b7.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| Jun 09     | 0.15     | 0.0      |
| Jun 10     | 0.15     | 0.0      |
| Jun 11     | 0.15     | 0.0      |
| Jun 12     | 0.2      | 0.0      |
| Jun 13     | 0.4      | 0.2      |
| Jun 14     | 0.2      | 0.2      |
| Jun 15     | 0.2      | 0.2      |
| Jun 16     | 0.15     | 0.15     |
</details>

![](images/159012e50507c0ab4757c9289f28d2538d495d25ada3d9ce2d18bd5a1a15ff6e.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| 09 Jun 2008 | 0.15     | 0.00     |
| 10         | 0.14     | 0.00     |
| 11         | 0.15     | 0.00     |
| 12         | 0.20     | 0.00     |
| 13         | 0.15     | 0.00     |
| 14         | 0.17     | 0.22     |
| 15         | 0.16     | 0.21     |
| 16         | 0.15     | 0.15     |
</details>

![](images/a9b891b6a7a79972aa520c8832f76339848e8cff927d7baf7a93fd69609c53e7.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| 09 Jun 2008 | 0.25     | 0.03     |
| 10         | 0.17     | 0.18     |
| 11         | 0.12     | 0.17     |
| 12         | 0.27     | 0.13     |
| 13         | 0.28     | 0.28     |
| 14         | 0.29     | 0.29     |
| 15         | 0.30     | 0.30     |
| 16         | 0.29     | 0.29     |
</details>

![](images/911eccb6052b22608a7a8f5b917b86892c24fb5f65f056053f597010ff4b010a.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| Jun 9      | 0.3      | 0.0      |
| Jun 10     | 0.4      | 0.0      |
| Jun 11     | 0.3      | 0.0      |
| Jun 12     | 0.5      | 0.0      |
| Jun 13     | 0.4      | 0.0      |
| Jun 14     | 0.6      | 0.0      |
| Jun 15     | 0.9      | 0.5      |
| Jun 16     | 0.4      | 0.4      |
</details>

![](images/0d80a94e2165d1cabab47547705ca5722fc77c266c51cca9e3a7a9892db9384c.jpg)

<details>
<summary>line</summary>

| Date       | Value  |
| ---------- | ------ |
| Jun 9      | 0.05   |
| Jun 10     | 0.10   |
| Jun 11     | 0.25   |
| Jun 12     | 0.15   |
| Jun 13     | 0.18   |
| Jun 14     | 0.19   |
| Jun 15     | 0.17   |
| Jun 16     | 0.12   |
</details>

![](images/31898331a00effe9b91912cda904af8fe87fcf2b16ee23005196e3bdef3a063c.jpg)

<details>
<summary>line</summary>

| Date       | Value  |
| ---------- | ------ |
| Jun 9      | 0.15   |
| Jun 10     | 0.02   |
| Jun 11     | 0.12   |
| Jun 12     | 0.15   |
| Jun 13     | 0.18   |
| Jun 14     | 0.17   |
| Jun 15     | 0.28   |
| Jun 16     | 0.13   |
</details>

![](images/022c33644595fab5373f3cdba12e6108da37709beeea9f1ca05e93efaf18cae2.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| Jun 9      | 0.2      | 0.0      |
| Jun 10     | 0.1      | 0.0      |
| Jun 11     | 0.1      | 0.0      |
| Jun 12     | 0.3      | 0.0      |
| Jun 13     | 0.5      | 0.0      |
| Jun 14     | 0.4      | 0.0      |
| Jun 15     | 0.3      | 0.4      |
| Jun 16     | 0.2      | 0.1      |
</details>

![](images/18777984ce5f8de02a25ec01512542aa68f0e7167db1063938005b26ea12fef2.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Jun 09     | 0.0   |
| Jun 10     | 0.3   |
| Jun 11     | 0.3   |
| Jun 12     | 0.6   |
| Jun 13     | 0.8   |
| Jun 14     | 0.8   |
| Jun 15     | 0.8   |
| Jun 16     | 0.3   |
</details>

![](images/6aa373fcd54d936480f7644c6dd17385fc32236c2ff7fec0ec52dff0ab12ae11.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| 09 Jun 2008 | 0.2      | 0.2      |
</details>

![](images/4a9c8016be7bb41e6300d4c0e19b115e118fa44007243df66da2b66121c7371d.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| 09 Jun 2008 | 0.4      | 0.1      |
| 10         | 0.1      | 0.15     |
| 11         | 0.1      | 0.1      |
| 12         | 0.55     | 0.6      |
| 13         | 0.7      | 0.75     |
| 14         | 0.6      | 0.7      |
| 15         | 0.7      | 0.75     |
| 16         | 0.15     | 0.15     |
</details>

![](images/a29710ff3c6b970ce14020b036409d688adf3e61ae9113a17beebdf017a6eb14.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| Jun 09     | 0.1   |
| Jun 10     | 0.15  |
| Jun 11     | 0.1   |
| Jun 12     | 0.15  |
| Jun 13     | 0.2   |
| Jun 14     | 0.35  |
| Jun 15     | 0.7   |
| Jun 16     | 0.45  |
</details>

![](images/751677e8dcb7a6232677bb517882cd76b876838d061cd7075d0bf1bffea85d54.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| 09 Jun 2008 | 0.35     | 0.35     |
| 10         | 0.15     | 0.35     |
| 11         | 0.35     | 0.35     |
| 12         | 0.45     | 0.60     |
| 13         | 0.45     | 0.55     |
| 14         | 0.75     | 0.75     |
| 15         | 0.75     | 0.75     |
| 16         | 0.40     | 0.40     |
</details>

![](images/c0190565ac457a71da49bd6c3c7beb691bd0098bb61fcaa6d9ac7f4efc326be2.jpg)

<details>
<summary>line</summary>

| Date       | Value  |
| ---------- | ------ |
| Jun 2008   | 0.30   |
</details>

![](images/44e7c78029fd36bcd801e48e3895f961db05a3bd17486a6642808b02b59e3058.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| 09 Jun 2008 | 0.2      | 0.0      |
</details>

Figure 8: The forecasting results of 16 samples from the Traffic dataset with input-96-predict-96.

![](images/532ba59cee8007589e8798e88d7b4da6a79529afb375840a10dc45e1b1253349.jpg)  
Figure 9: The forecasting results of 16 samples from the Taxi dataset with input-96-predict-96.

![](images/c3f48cbda5b784d84f32a04ba906a7fca7871b53787bc0b87a2f813386d986d9.jpg)

<details>
<summary>line</summary>

| Date       | observations | median prediction | 90% prediction interval | 50% prediction interval |
| ---------- | ------------ | ----------------- | ----------------------- | ----------------------- |
| 05-Jan-2016| 0.7          | 0.4               | 0.3                     | 0.2                     |
| 12:00      | 0.6          | 0.3               | 0.2                     | 0.1                     |
| 06-Jan-2016| 0.5          | 0.2               | 0.1                     | 0.0                     |
| 12:00      | 0.6          | 0.3               | 0.2                     | 0.1                     |
| 07-Jan-2016| 0.5          | 0.2               | 0.1                     | 0.0                     |
| 12:00      | 0.6          | 0.3               | 0.2                     | 0.1                     |
| 08-Jan-2016| 0.5          | 0.2               | 0.1                     | 0.0                     |
</details>

![](images/e456d1d6c3d6d28249feb08997a694580f4b4543129c33b4890a098db5b3b681.jpg)

<details>
<summary>line</summary>

| Date       | Value  |
| ---------- | ------ |
| 05-jan 2016 | 0.7    |
| 06-jan     | 0.3    |
| 07-jan     | -0.1   |
| 08-jan     | 0.4    |
</details>

![](images/76112dd085807e6c667d6e60140a325f251a1d2e29d6d5c50755106a72f13bcb.jpg)

<details>
<summary>line</summary>

| Time       | Value |
| ---------- | ----- |
| 05-Jan-01  | 0.3   |
| 06-Jan-01  | 0.8   |
| 07-Jan-01  | 0.9   |
| 08-Jan-01  | 0.5   |
</details>

![](images/596e95ccb1b1013a08373e86d3760af4d2ec7bd4a79de8e948526867c7b2834f.jpg)

<details>
<summary>line</summary>

| Time       | Value  |
| ---------- | ------ |
| 05-jan 2016 | 0.7    |
| 12:00      | 0.6    |
| 06-jan     | 0.8    |
| 12:00      | 0.4    |
| 07-jan     | 0.3    |
| 12:00      | 0.6    |
| 08-jan     | 0.8    |
</details>

![](images/3da02d5068b26175c52ef40ea9281172ee3e7d46daa3c0b2c0c60c0f8f0853eb.jpg)

<details>
<summary>line</summary>

| Date       | Value  |
| ---------- | ------ |
| 05-jan 2016 | 0.3    |
| 12:00      | 0.4    |
| 06-jan     | 0.3    |
| 12:00      | 0.4    |
| 07-jan     | 0.3    |
| 08-jan     | 0.3    |
</details>

![](images/3f017be90584e3beae1a30c5378f6b04694f4098f968b3681409d719cc28d9ab.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| 05-Jan-01  | 0.5   |
| 12:00      | 0.4   |
| 06-Jan-01  | 0.3   |
| 12:00      | 0.2   |
| 07-Jan-01  | 0.1   |
| 12:00      | 0.3   |
| 08-Jan-01  | 0.4   |
| 12:00      | 0.9   |
</details>

![](images/33939bd92f375773b1215f8fcca2c4ce71ab2f1d624d34834fd6152c0b6703c3.jpg)

<details>
<summary>line</summary>

| Time       | Value  |
| ---------- | ------ |
| 05-jan 2016 | 0.1    |
| 12:00      | 0.2    |
| 06-jan     | 0.4    |
| 12:00      | 0.3    |
| 07-jan     | 0.5    |
| 12:00      | 0.7    |
| 08-jan     | 0.3    |
</details>

![](images/588277eb98339db0c596238f6d3e956010f7ec6969fb76166d2ab499183834ed.jpg)

<details>
<summary>line</summary>

| Time       | Value |
| ---------- | ----- |
| 05-jan 2016 | 0.4   |
| 12:00      | 0.6   |
| 06-jan     | 0.8   |
| 12:00      | 0.4   |
| 07-jan     | 0.3   |
| 08-jan     | 0.5   |
</details>

![](images/308f776c86c92cea2e13498a1db5cff1f4fe148465e9ea0169277098acf7ed09.jpg)

<details>
<summary>line</summary>

| Date       | Series 1 | Series 2 |
| ---------- | -------- | -------- |
| 05-Jan 2016 | 0.8      | 0.3      |
| 12:00      | 0.7      | 0.4      |
| 06-Jan     | 0.6      | 0.3      |
| 12:00      | 0.5      | 0.2      |
| 07-Jan     | 0.9      | 0.3      |
| 12:00      | 0.7      | 0.2      |
| 08-Jan     | 0.8      | 0.3      |
</details>

![](images/145159d26822fb047437f83d38a89a623b5fad7364811ddb5a596088d21ccba6.jpg)

<details>
<summary>line</summary>

| Time       | Value  |
| ---------- | ------ |
| 05-Jan 2016 | 0.3    |
| 12:00      | 0.4    |
| 06-Jan     | 0.5    |
| 12:00      | 0.3    |
| 07-Jan     | 0.4    |
| 12:00      | 0.5    |
| 08-Jan     | 0.6    |
| 12:00      | 0.5    |
</details>

![](images/4dd7e0bbca655ceadceefb2b58a7d6516ae5fa9b41b07c3e88950135b9dd75b7.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| 05-Jan 2016 | 0.3   |
| 06-Jan     | 1.0   |
| 07-Jan     | 0.9   |
| 08-Jan     | 0.6   |
</details>

![](images/fe5bdb3c900bb74b9a9714e9caa7191d2f4f54739d57c2c66c6e7a997959fa76.jpg)

<details>
<summary>line</summary>

| Time       | Value  |
| ---------- | ------ |
| 05-Jan 2016 | 0.1    |
| 06-Jan     | 0.4    |
| 07-Jan     | 0.3    |
| 08-Jan     | 0.6    |
</details>

![](images/05c88246c18b764143cefcad3854621072a33d0c5771a8babb55c59e93151d3c.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| 05-Jan 2016 | 0.4   |
| 12:00      | 0.9   |
| 06-Jan     | 0.4   |
| 12:00      | 0.8   |
| 07-Jan     | 0.7   |
| 12:00      | 0.8   |
| 08-Jan     | 1.0   |
</details>

![](images/73a2c6193848c6596fcb9930f3fab312ae226a6067bcbe7d12cb9f618226cda6.jpg)

<details>
<summary>line</summary>

| Time       | Value  |
| ---------- | ------ |
| 05-Jan     | 0.5    |
| 06-Jan     | 0.7    |
| 07-Jan     | 0.9    |
| 08-Jan     | 1.0    |
</details>

![](images/22d5eff3d101177df0a225b142742d6734f0672c0f554a042396ea82c211e4f7.jpg)

<details>
<summary>line</summary>

| Date       | Value |
| ---------- | ----- |
| 05-Jan 2016 | 0.4   |
| 12:00      | 0.5   |
| 06-Jan     | 0.3   |
| 12:00      | 0.6   |
| 07-Jan     | 0.2   |
| 12:00      | 0.7   |
| 08-Jan     | 0.9   |
</details>

![](images/558c581ca5b342a379fc0ef7d425e6e559a1a78bea58dd1290bcf6e9fafab31f.jpg)

<details>
<summary>line</summary>

| Time       | Value  |
| ---------- | ------ |
| 05-Jan 2016 | 0.35   |
| 06-Jan     | 0.45   |
| 07-Jan     | 0.25   |
| 08-Jan     | 0.30   |
</details>

Figure 10: The forecasting results of 16 samples from the Wikipedia dataset with input-96-predict-96.

![](images/2608dc28487479c5de407268b594a2f9c967b08a6eedbf1b2dea50fdc97639d5.jpg)

<details>
<summary>line</summary>

| Year | observations | median prediction | 90.0% prediction interval | 50.0% prediction interval |
|------|--------------|-------------------|---------------------------|---------------------------|
| 2000 | 0.8          | 0.8               | 0.8                       | 0.8                       |
| 2001 | 0.0          | 0.4               | 0.4                       | 0.4                       |
| 2002 | 0.4          | 0.4               | 0.4                       | 0.4                       |
| 2003 | 0.8          | 0.8               | 0.8                       | 0.8                       |
| 2004 | 0.6          | 0.6               | 0.6                       | 0.6                       |
| 2005 | 0.8          | 0.8               | 0.8                       | 0.8                       |
| 2006 | 0.6          | 0.6               | 0.6                       | 0.6                       |
</details>

![](images/8f76c5d28a58b20cf6ae5530e77cd8e19ffd277aef9e892f68f9d6cf7d9b15f3.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.8      | 0.6      |
| 2001 | 1.0      | 0.4      |
| 2002 | 0.2      | 0.3      |
| 2003 | 0.6      | 0.2      |
| 2004 | 0.7      | 0.5      |
| 2005 | 0.9      | 0.7      |
| 2006 | 1.0      | 0.9      |
</details>

![](images/b95a66eab93de807c202fff04102da70055bcbdd9a5ce17bf7bb8b28f778b039.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
| ---- | -------- | -------- |
| 2000 | 0.7      | 0.8      |
| 2001 | 0.9      | 0.0      |
| 2002 | 0.1      | 0.2      |
| 2003 | 0.6      | 0.8      |
| 2004 | 0.7      | 0.8      |
| 2005 | 0.9      | 0.8      |
| 2006 | 0.7      | 1.0      |
</details>

![](images/7a2861cb0358517d97df402cf15140effa1fec51e8df6679eee542733dccdde2.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.4      | 0.4      |
| 2001 | 0.0      | 0.4      |
| 2002 | 0.2      | 0.4      |
| 2003 | 0.8      | 0.8      |
| 2004 | 0.6      | 0.6      |
| 2005 | 0.7      | 0.7      |
| 2006 | 0.5      | 0.5      |
</details>

![](images/7a82df88995264fb43773e78ca961dc47567e5552c09038154acbb5965b62817.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.5      | 0.1      |
| 2001 | 0.7      | 0.8      |
| 2002 | 0.9      | 0.6      |
| 2003 | 0.8      | 0.7      |
| 2004 | 0.6      | 0.9      |
| 2005 | 0.5      | 0.4      |
| 2006 | 0.8      | 0.6      |
</details>

![](images/24c2dba127a948a57b37a3cb8f6db4e7f1303dc832df6917f74d31f7ebe21840.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.5      | 0.3      |
| 2001 | 1.0      | 0.2      |
| 2002 | 0.5      | 0.1      |
| 2003 | 0.7      | 0.4      |
| 2004 | 0.8      | 0.9      |
| 2005 | 0.6      | 0.5      |
| 2006 | 0.9      | 1.0      |
</details>

![](images/1efcb9b5598f9c34d48c852ed3ed00471da4bc0a44f18eb49ebc0d7532298736.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.5      | 0.3      |
| 2001 | 0.8      | 0.6      |
| 2002 | 1.0      | 0.7      |
| 2003 | 0.9      | 0.8      |
| 2004 | 1.0      | 0.9      |
| 2005 | 1.1      | 1.0      |
| 2006 | 0.7      | 0.8      |
</details>

![](images/9d04435bdd56c4aba608885735223c1e07a3435c8359af752c6acafaa84c880b.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.5      | 0.1      |
| 2001 | 0.3      | 0.4      |
| 2002 | 0.6      | 0.2      |
| 2003 | 0.8      | 0.5      |
| 2004 | 0.7      | 0.6      |
| 2005 | 0.9      | 0.7      |
| 2006 | 1.0      | 0.8      |
</details>

![](images/a835c242e135e55bf95d3c8e9a1ee774fe4ff8a8c86613f8bea6c72a4b00282b.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.5      | 0.1      |
| 2001 | 0.7      | 0.8      |
| 2002 | 0.6      | 0.9      |
| 2003 | 0.8      | 0.7      |
| 2004 | 1.0      | 0.9      |
| 2005 | 0.9      | 1.0      |
| 2006 | 0.5      | 0.3      |
</details>

![](images/2bbbdf4de8c6e64c70245ad0f0a46599f2f2ce27cba847bb32d435cf70dce847.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.0      | 0.4      |
| 2001 | 0.8      | 1.0      |
| 2002 | 0.7      | 0.9      |
| 2003 | 0.5      | 1.0      |
| 2004 | 0.6      | 0.8      |
| 2005 | 0.7      | 0.9      |
| 2006 | 0.4      | 0.8      |
</details>

![](images/f8cf56615ae412e3fadae847c048b1011124fac9380e516ce8549499ea84dd2f.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.4      | 0.1      |
| 2001 | 0.6      | 0.3      |
| 2002 | 0.5      | 0.4      |
| 2003 | 0.8      | 0.6      |
| 2004 | 0.7      | 0.5      |
| 2005 | 0.9      | 0.7      |
| 2006 | 1.0      | 0.8      |
</details>

![](images/d3398e3f156b337bb080bf544c1f345cb6e45ec021270aaa4ff765e0f098605d.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.3      | 0.4      |
| 2001 | 0.7      | 0.8      |
| 2002 | 1.0      | 0.9      |
| 2003 | 0.5      | 0.8      |
| 2004 | 0.6      | 0.7      |
| 2005 | 0.5      | 0.6      |
| 2006 | 1.0      | 0.9      |
</details>

![](images/23d11866e16e866daa7d373941aa7d7854ad1a27c4871bbb4e0cd03c75afb6e8.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.3      | 0.4      |
| 2001 | 0.8      | 0.5      |
| 2002 | 0.7      | 0.6      |
| 2003 | 0.2      | 0.7      |
| 2004 | 0.6      | 0.8      |
| 2005 | 0.9      | 0.7      |
| 2006 | 1.0      | 0.9      |
</details>

![](images/694152e7ba08c545add6966b04319f563294d38b2e7812fa6ae9bee891baf6ae.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.6      | 0.4      |
| 2001 | 0.7      | 0.5      |
| 2002 | 0.8      | 0.6      |
| 2003 | 0.9      | 0.7      |
| 2004 | 0.8      | 0.6      |
| 2005 | 0.7      | 0.5      |
| 2006 | 0.9      | 0.8      |
</details>

![](images/4c9e80240c0ef617e39201e858b93e9065748f4a95dc329d99d8cc63aa38591e.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.1      | 0.65     |
| 2001 | 0.4      | 0.3      |
| 2002 | 0.6      | 0.6      |
| 2003 | 1.0      | 0.4      |
| 2004 | 0.5      | 1.0      |
| 2005 | 0.6      | 0.5      |
| 2006 | 0.8      | 0.8      |
</details>

![](images/b32e3252fa54b852eefc077a3934696b450f454bce2f7ec3343859b311c02b7b.jpg)

<details>
<summary>line</summary>

| Year | Series 1 | Series 2 |
|------|----------|----------|
| 2000 | 0.8      | 0.1      |
| 2001 | 0.9      | 0.7      |
| 2002 | 0.7      | 0.5      |
| 2003 | 1.0      | 0.4      |
| 2004 | 0.5      | 0.3      |
| 2005 | 0.8      | 0.5      |
| 2006 | 0.9      | 0.6      |
</details>

Figure 11: The forecasting results of 16 samples from the hospital dataset with input-24-predict-48.