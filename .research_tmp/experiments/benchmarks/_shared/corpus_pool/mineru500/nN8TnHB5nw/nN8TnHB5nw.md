# Memory Efficient Optimizers with 4-bit States

Bingrui Li $^{1}$ , Jianfei Chen $^{1\dagger}$ , Jun Zhu $^{1}$

$^{1}$ Dept. of Comp. Sci. and Tech., Institute for AI, BNRist Center, THBI Lab, Tsinghua-Bosch Joint ML Center, Tsinghua University lbr22@mails.tsinghua.edu.cn; {jianfeic, dcszj}@tsinghua.edu.cn

# Abstract

Optimizer states are a major source of memory consumption for training neural networks, limiting the maximum trainable model within given memory budget. Compressing the optimizer states from 32-bit floating points to lower bitwidth is promising to reduce the training memory footprint, while the current lowest achievable bitwidth is 8-bit. In this work, we push optimizer states bitwidth down to 4-bit through a detailed empirical analysis of first and second moments. Specifically, we find that moments have complicated outlier patterns, that current block-wise quantization cannot accurately approximate. We use a smaller block size and propose to utilize both row-wise and column-wise information for better quantization. We further identify a zero point problem of quantizing the second moment, and solve this problem with a linear quantizer that excludes the zero point. Our 4-bit optimizers are evaluated on a wide variety of benchmarks including natural language understanding, machine translation, image classification, and instruction tuning. On all the tasks our optimizers can achieve comparable accuracy with their full-precision counterparts, while enjoying better memory efficiency.\*

# 1 Introduction

Large-scale models with a massive amount of parameters $[5, 9, 20, 22, 49, 58]$ have shown impressive few-shot learning abilities on general tasks $[52]$ . Despite being powerful, training these models is challenging. Memory capacity is one of the main bottlenecks of training large-scale models. Modern neural networks are typically trained with stateful optimizers such as Adam $[26]$ , which need to maintain one or two optimizer states (i.e., first and second moments) per each parameter. As the model size grows, the memory consumed by optimizer states can be a dominating factor of memory consumption $[40, 41]$ .

There are several attempts to reduce optimizers' memory consumption. Factorization [2, 8, 46] applies low-rank approximation to optimizer states, delta tuning [18, 23, 24, 27, 28] avoids maintaining optimizer states for most parameters by only tuning a small subset, and low-precision optimizers [15, 44] represent their states with low-precision numerical formats, which consume less memory.

Among these methods, low-precision optimizers are attractive due to their simplicity and wide applicability. Dettmers et al. [15] propose an 8-bit optimizer with reparameterized embedding layers (“stable embedding layers”) and a block-wise 8-bit dynamic exponential numerical format for optimizer states. Their 8-bit optimizers achieve similar convergence to full-precision optimizers on language modeling, image classification, machine translation, and language understanding tasks.

In this work, we push the required numerical precision of low-precision optimizers from 8 to 4-bit through analyzing the patterns in the first and second moments and designing dedicated quantizers

Algorithm 1 Compression-based Memory Efficient Optimization Framework   
Require: black-box stochastic optimization algorithm A, initial parameter $w_{0} \in R^{p}$ , initial optimizer state $\bar{s}_{0} = 0$ , total number of iterations T

1: for $t = 1, 2, \ldots, T$ do

2: Sample a minibatch $\zeta_{t}$ and get stochastic gradient $g_{t} = \nabla_{w} f(w_{t-1}, \zeta_{t})$ 3: $s_{t-1} \leftarrow decompress(\bar{s}_{t-1})$ 4: $w_{t}, s_{t} \leftarrow A(w_{t-1}, s_{t-1}, g_{t})$ 5: $\bar{s}_{t} \leftarrow compress(s_{t})$ 6: end for

7: return $w_{T}$

for optimizer states. Specifically, we find that moments exhibit complicated outlier patterns, that vary across different parameter tensors. The large block size proposed by Dettmers et al. [15] cannot properly handle all different outlier patterns. Based on this observation, we propose to use a smaller block size, which improves the approximation of first moment.

For the second moment, we find its quantization suffers from a zero-point problem. Since the update direction is usually inversely proportional to the square root of the second moment, quantizing non-zero quantities to zero will cause significant deviation. To address this problem, we propose a simple linear quantizer to exclude the zero-point for second moment. We further propose a stronger quantization technique, rank-1 normalization, to improve the approximation of second moment by better handling the outlier patterns. Our proposed quantization techniques are robust enough to achieve lossless convergence under 4-bit, even without the stable embedding layers proposed by Dettmers et al. [15].

Finally, we investigate the combination of factorization methods $[2, 46]$ with low-precision optimizers, and propose a memory efficient optimizer which utilizes quantized 4-bit first moment and factorized second moment. For applicable tasks, the hybrid optimizer enjoys best of both worlds: good convergence and memory efficiency.

We evaluate our 4-bit optimizers on a wide range of tasks, including natural language understanding, machine translation, image classification, and instruction tuning of large language models. On all the benchmarks, our 4-bit optimizer can converge similarly fast with full-precision optimizers, and the converged models do not have noticeable accuracy loss. Our optimizers consume less memory than existing 8-bit optimizers [15], while improves the throughput of language model finetuning with optimizer offloading [41, 45] due to reduced communication cost.

# 2 Preliminaries

In this section, we present some preliminaries of compression-based memory efficient optimizers and discuss quantization methods for compression of optimizer states in a general formulation.

# 2.1 A Framework for Compression-based Memory Efficient Optimizers

Gradient-based stateful optimizers like SGDM [38, 47], Adam [26], AdamW [32] are the principal choices in deep neural network training. However, the memory footprint of stateful optimizers is several times of model itself, which results in a bottleneck for large model pretraining/finetuning. Consider the update rule of the Adam optimizer:

$$
\mathbf {A d a m} \left(\mathbf {w} _ {t - 1}, \mathbf {m} _ {t - 1}, \mathbf {v} _ {t - 1}, \mathbf {g} _ {t}\right) = \left\{ \begin{array}{l l} \mathbf {m} _ {t} & = \beta_ {1} \mathbf {m} _ {t - 1} + (1 - \beta_ {1}) \mathbf {g} _ {t} \\ \mathbf {v} _ {t} & = \beta_ {2} \mathbf {v} _ {t - 1} + (1 - \beta_ {2}) \mathbf {g} _ {t} ^ {2} \\ \hat {\mathbf {m}} _ {t} & = \mathbf {m} _ {t} / \left(1 - \beta_ {1} ^ {t}\right) \\ \hat {\mathbf {v}} _ {t} & = \mathbf {v} _ {t} / \left(1 - \beta_ {2} ^ {t}\right) \\ \mathbf {w} _ {t} & = \mathbf {w} _ {t - 1} - \eta \cdot \hat {\mathbf {m}} _ {t} / (\sqrt {\hat {\mathbf {v}} _ {t}} + \epsilon) \end{array} \right. \tag {1}
$$

During the training process, the model parameters $\mathbf{w}_t$ and optimizer states (i.e., first and second moments) $\mathbf{m}_t$ , $\mathbf{v}_t$ need to be stored persistently in the GPU memory. As the model grows large, optimizer states are a main source of training memory consumption. Each parameter dimension takes

two 32-bit optimizer states, making the memory consumption of Adam-style optimizers three times larger than stateless optimizers like SGD.

Compressing the optimizer states is a promising method to achieve memory efficient optimization. Formally, given a gradient-based optimizer A, a memory efficient version of the optimization algorithm with compressed optimizer states is given by Alg. 1. The algorithm A can be any gradient-based optimizers like SGDM, Adam, AdamW, etc. See App. F for examples. In Alg. 1, the precise states $s_{t}$ are only temporal, and only compressed states $\bar{s}_{t}$ are stored persistently in the GPU memory. Memory footprint reduction can be achieved since in neural networks, the state vectors $m_{t}$ and $v_{t}$ are usually concatenation of state vectors of each parameterized layer. Therefore, we can perform the optimizer steps (Line 3-5) separately for each layer, so only one single layer of the precise states are presented in the memory at a time. The states for all other layers are kept compressed. In the rest of this paper, we focus on the compression and decompression method, to achieve high compression rate of optimizer states while maintaining good convergence of the optimizer.

# 2.2 Main Compression Method: Quantization

Quantizing the optimizer states to lower precision (e.g. 8-bit integers) is an effective way to compression optimizer states [15]. In this case, the optimizer states are compressed with a quantizer and decompressed with a dequantizer. The low-precision numerical format significantly impacts the accuracy of quantization methods. Here, we present a general framework for numerical formats we considered for compressing optimizer states.

A quantizer converts full-precision tensors to low-precision formats. Based on the formulation proposed by Dettmers and Zettlemoyer [16], we disentangle the quantizer $\mathbf{Q}(\cdot)$ into two parts: normalization $\mathbf{N}(\cdot)$ and mapping $\mathbf{M}(\cdot)$ , which applies sequentially and element-wisely to a tensor to be quantized. For concise presentation, we only discuss quantizers for unsigned inputs, and defer the discussion of signed inputs in App. E.1. Formally, the quantizer for a tensor $x \in R^{p}$ is given by

$$
q _ {j} := \mathbf {Q} (x _ {j}) = \mathbf {M} \circ \mathbf {N} (x _ {j}).
$$

Normalization The normalization operator N scales each elements of x into the unit interval, i.e. [0, 1]. Normalization can have different granularity, such as per-tensor, per-token (row) [37, 55], per-channel (column) [4], group-wise [37, 55] and block-wise [15]. The per-tensor and block-wise normalization operators are given by

$$
n _ {j} := \mathrm{N} _ {\text { per - tensor }} (x _ {j}) = x _ {j} / \max _ {1 \leq i \leq p} | x _ {i} |,
$$

$$
n _ {j} := \mathbf {N} _ {\text { block - wise }} (x _ {j}) = x _ {j} / \max \left\{\left| x _ {i} \right|: 1 + B \left\lfloor j / B \right\rfloor \leq i \leq B (\left\lfloor j / B \right\rfloor + 1) \right\},
$$

respectively, where the involved scaling factors are called quantization scale, which are persistently stored together with quantized tensor until dequantization. The granularity of normalization presents a trade-off of quantization error and memory overhead. Normalization method with low quantization error and acceptable memory overhead is preferred. In this case, the coarsest per-tensor normalization operator has negligible memory overhead, i.e. only 1 scaling factor regardless of tensor size, but suffers from largest error due to outliers. Block-wise normalization views the tensor as an 1-dimensional array, divides the array into blocks of size B called block and assigns a quantization scale within each block, which leads to $\lceil p/B \rceil$ quantization scales totally. The block size could be adapted to trade-off quantization error and memory overhead.

Mapping A mapping [15] converts normalized quantities to low-bitwidth integers. Formally, the mapping operator $\mathbf{M} = \mathbf{M}_{\mathbf{T},b}$ is equipped with a bitwidth $b$ and a predefined increasing mapping, named quantization mapping $\mathbf{T}:[0,2^b -1]\cap \mathbb{Z}\to [0,1]$ . Then $\mathbf{M}$ is defined as

$$
q _ {j} := \mathbf {M} (n _ {j}) = \arg \min _ {0 \leq i <   2 ^ {b}} | n _ {j} - \mathbf {T} (i) |.
$$

The design of T is critical as it could effectively mitigate quantization error by capturing the distribution information of n. There are two kinds mappings that are of specific interest to optimizer states quantization, linear mapping and dynamic exponent (DE) mapping [13]. A linear mapping $\mathbf{T}(i) = (i + 1)/2^{b}$ defines a linear quantizer, where the quantization intervals distribute evenly in each block. The DE mapping can approximate small values well, similar to floating point numbers. DE splits the binary representation of a low-precision integer i into three parts: a leading

(a)   
![](images/49cc9a50d49cacddba9b2bca29b2ccc7d75dc465d5e4259c8972f569301656ed.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 1000 | 0    | 0.4   |
| 2000 | 0    | 0.6   |
| 3000 | 0    | 0.8   |
| 400  | 0    | 1.0   |
| 500  | 0    | 0.8   |
| 600  | 0    | 0.6   |
| 700  | 0    | 0.4   |
| 800  | 0    | 0.2   |
</details>

(b)   
![](images/e8a4a52f8338d5c793c8c097cb92cb4bbeaf8607b07cb7aa3cd6196e316fba35.jpg)

<details>
<summary>area</summary>

| x        | y     |
| -------- | ----- |
| -10^-3   | 0     |
| -10^-4   | 5000  |
| -10^-3   | 20000 |
| 0        | 20000 |
| 10^-4    | 5000  |
| 10^-3    | 0     |
</details>

(c)   
![](images/258f66acbb4e0752860ff3f5d25920f027f5b9bb81aa5d4f4655effcae979a10.jpg)

<details>
<summary>histogram</summary>

| Bin Range       | Frequency |
| --------------- | --------- |
| -10⁻³ to -10⁻⁴  | ~0        |
| -10⁻⁴ to -10⁻³  | ~5000     |
| -10⁻⁵ to -10⁻⁴  | ~20000    |
| -10⁻⁶ to -10⁻⁵  | ~35000    |
| -10⁻⁷ to -10⁻⁶  | ~50000    |
| -10⁻⁸ to -10⁻⁷  | ~20000    |
| -10⁻⁹ to -10⁻⁸  | ~5000     |
| -10⁻¹⁰ to -10⁻⁹ | ~1000     |
| -10⁻¹¹ to -10⁻⁸ | ~200      |
| -10⁻¹² to -10⁻⁷ | ~50       |
| -10⁻¹³ to -10⁻⁶ | ~10       |
| -10⁻¹⁴ to -10⁻⁵ | ~2        |
| -10⁻¹⁵ to -10⁻⁴ | ~5        |
| -10⁻¹⁶ to -10⁻³ | ~1        |
| -10⁻¹⁷ to -10⁻² | ~2        |
| -10⁻¹⁸ to -10⁻¹ | ~5        |
| -10⁻¹⁹ to -10⁻⁰ | ~10       |
| -10⁻²⁰ to -10⁻⁰ | ~20       |
| -10⁻²¹ to -10⁻⁰ | ~50       |
| -10⁻²² to -10⁻⁰ | ~100      |
| -10⁻²³ to -10⁻⁰ | ~200      |
| -10⁻²⁴ to -10⁻⁰ | ~500      |
| -10⁻²⁵ to -10⁻⁰ | ~100      |
| -10⁻²⁶ to -10⁻⁰ | ~2        |
| -10⁻²⁷ to -10⁻⁰ | ~5        |
| -10⁻²⁸ to -10⁻⁰ | ~1        |
| -10⁻²⁹ to -10⁻⁰ | ~2        |
| -10⁻³⁰ to -10⁻⁰ | ~5        |
</details>

(d)   
![](images/ccc438aac34f60607da3a419fb9a76843e29409ccfe36850a15963bb31e8287f.jpg)

<details>
<summary>histogram</summary>

| Bin Range       | Frequency |
| --------------- | --------- |
| -10⁻³ to -10⁻⁴  | ~0        |
| -10⁻⁴ to -10⁻³  | ~5000     |
| -10⁻⁵ to -10⁻³  | ~10000    |
| -10⁻⁶ to -10⁻³  | ~2000     |
| -10⁻⁷ to -10⁻³  | ~500      |
| -10⁻⁸ to -10⁻³  | ~100      |
| -10⁻⁹ to -10⁻³  | ~50       |
| -10⁻¹⁰ to -10⁻²⁹ | ~20       |
| -10⁻¹¹ to -10⁻²⁸ | ~10       |
| -10⁻¹² to -10⁻²⁷ | ~5        |
| -10⁻¹³ to -10⁻²⁶ | ~2        |
| -10⁻¹⁴ to -10⁻²⁵ | ~1        |
| -10⁻¹⁵ to -10⁻²⁴ | ~0        |
| -10⁻¹⁶ to -10⁻²³ | ~0        |
| -10⁻¹⁷ to -10⁻²² | ~0        |
| -10⁻¹⁸ to -10⁻²¹ | ~0        |
| -10⁻¹⁹ to -10⁻²⁰ | ~0        |
| -10⁻²⁰ to -10⁻¹9 | ~0        |
| -10⁻²¹ to -10⁻¹⁸ | ~0        |
| -10⁻²² to -10⁻¹⁷ | ~0        |
| -10⁻²³ to -10⁻¹⁶ | ~0        |
| -10⁻²⁴ to -10⁻¹⁵ | ~0        |
| -10⁻²⁵ to -10⁻¹⁴ | ~0        |
| -10⁻²⁶ to -10⁻¹³ | ~0        |
| -10⁻²⁷ to -10⁻¹² | ~0        |
| -10⁻²⁸ to -10⁻¹¹ | ~0        |
| -10⁻²⁹ to -10⁻¹⁰ | ~0        |
| -10⁻³⁰ to -10⁻⁹  | ~0        |
</details>

Figure 1: Visualization of the first moment in the layers.3.blocks.1.mlp.fc1 layer in a Swin-T model. (a): Magnitude of the first moment. (b): Histogram of the first moment. (c): Moment approximated by B128/DE. (d): Moment approximated by B2048/DE.

sequence of E zeros, followed by an indicator bit one, and remaining F fraction bits. DE defines $\mathbf{T}(i) = 10^{-E(i)}\text{fraction}(i)$ , where $\text{fraction}(i) \in [0.1, 1]$ . See App. E.2 for full specifications and visualizations of different quantization mappings.

The normalization N and mapping M roughly play the same role in finding good quantization point candidates and they affect each other. If an oracle normalization scaling the original tensor x to a uniform distribution is accessible, linear mapping could be used generally. On the contrary, if the optimal mapping could be readily identified with respect to certain metrics for a per-tensor normalized tensor, there is no necessity to develop additional normalization methods that incur extra overhead for mitigating the negative effects of outliers. In essence, if one of the two operators approaches optimality, the quantizer can still perform well even when the other operator is set to its most basic configuration. Additionally, as one operator becomes increasingly powerful, the potential for improvement in the other operator gradually diminishes.

Dequantization The dequantizer is just the inverse of the quantizer, which is simply

$$
\tilde {x} _ {j} := \mathbf {N} ^ {- 1} \circ \mathbf {T} (q _ {j}).
$$

Based on our formulation, we name quantizers by their normalization and mapping methods as Norm./Map.. For example, 8-bit optimizers [15] use block-wise normalization with a block size of 2048 and dynamic exponent mapping, which we call B2048/DE in the rest of the paper.

# 3 Compressing First Moment

In the next two sections, we describe our design of compression and decompression methods in Alg. 1 to realize our memory efficient 4-bit optimizers. We first discuss the compression of the first moment. Our compression method for first moment is based on Dettmers et al. [15], which uses block-wise normalization with a block size of 2048 and dynamic exponent mapping [13]. We preliminary reduce the bitwidth from 8-bit to 4-bit and discover that the first moment is rather robust to quantization. This simple optimizer can already converge with 4-bit first moment, though in some cases there is accuracy degradation.

To further improve the performance, we investigate the patterns of the first moment. There are some outliers in the moments and outliers significantly affect quantization scale due to their large magnitude. Outlier patterns have been studied for weights and activations. It is shown that the weights are rather smooth $[14, 54, 57]$ , while the activation have column-wise outliers $[4, 14, 53, 54]$ , i.e., the outliers in activation always lie in some fixed channels. However, we find that the outlier patterns in optimizer states are quite complicated. Specifically, Fig. 2 shows patterns of first moment in a Swin-T model during training. Outliers of the layers.3.blocks.0.mlp.fc1 layer lie in fixed rows while outliers of the layers.3.blocks.0.mlp.fc2 layer lie in fixed columns. Actually, the outlier patterns vary across different architectures, layers and parameters. See more patterns in App. B.

The complicated outlier pattern makes optimizer states harder to quantize. There exist some moment tensors, that the outliers persist roughly in certain columns (dimension 1), as shown in Fig. 1. Block-wise normalization treats this tensor as an flattened one-dimensional sequence, in the row-first order. Therefore, any block size of 2048 would include an entry in the outlier column, resulting

(a)   
![](images/bf7a55da24b9ccae8e8d0f3f4dda8472eb378bb9d29668e2ed323f837bafcf39.jpg)

(b)   
![](images/fc572bdda6c447c284632b6395530c752674cf03a176d13f328cbda3f5abe5d6.jpg)  
Figure 2: Outlier patterns vary across two first moment tensors. (a): outliers lie in fixed rows (dimension 0). (b): outliers lie in fixed columns (dimension 1).

![](images/03d73f707842681915748e05dbfdc307c7d5c51d98637cbaea676ffb1d4af6ec.jpg)

<details>
<summary>area</summary>

| x    | y     |
| ---- | ----- |
| 2.0  | 0.00  |
| 2.5  | 0.00  |
| 3.0  | 0.00  |
| 3.5  | 0.00  |
| 4.0  | 0.00  |
| 4.5  | 0.50  |
| 5.0  | 1.50  |
| 5.5  | 0.25  |
| 6.0  | 0.00  |
</details>

![](images/f6fd4b0f30d2fa73bc73695dadff3a0e87825bfcca1181ee84d45a27e828be56.jpg)

![](images/c254fd9cbdd3208cb14ba127f11d78da7ba86d92629e8855993388b3744a8138.jpg)

<details>
<summary>area</summary>

| x    | y     |
| ---- | ----- |
| 2.0  | 0.00  |
| 2.5  | 0.00  |
| 3.0  | 0.00  |
| 3.5  | 0.00  |
| 4.0  | 0.00  |
| 4.5  | 0.50  |
| 5.0  | 1.50  |
| 5.5  | 0.25  |
| 6.0  | 0.00  |
</details>

Figure 3: Histogram of the inverse square root of second moment. (a) full-precision; (b) quantized with B128/DE; (c) quantized with B128/DE-0. All figures are at log10 scale and y-axis represents density.

in a large quantization scale. In this case, block-wise normalization is not better than per-tensor normalization. Therefore, we adopt a smaller block size of 128, as it provides enhanced performance while incurring only a little memory overhead. In Fig. 1, we show that quantizing with a smaller block size approximates the moments better than a large block size.

# 4 Compressing Second Moment

Compared to the first moment, quantizing the second moment is more difficult and incurs training instability. Besides its sharper outlier pattern and ill-conditioned distribution compared with first moment, we further identify the zero-point problem as the main bottleneck of quantizing the second moment. We propose an improved normalization method with a quantization mapping excluding zero. We also propose a factorization method for compressing the second moment, which leads to even better memory efficiency.

# 4.1 Zero-point Problem

The problem of quantizing second moment different from quantizing other tensors in neural networks. For weights [25], activations [25], and gradients [1], it is desirable to contain zero in the quantization mapping for lower approximation error and training stability. Empirically, zero is often the most frequent element [57]. But for second moment in Adam, small values around zero significantly impact the update direction, which is proportional to the inverse square root of the second moment, as shown in Eq. 1. In this case, a small quantization error of values around zero will cause catastrophically large deviation of the update direction. Fig. 3 shows the histogram of the inverse square root (i.e., transformed with $h(v) = 1 / (\sqrt{v} + 10^{-6})$ ) of the second moment quantized with B128/DE. The quantizer pushes most of entries of the tensor to zero, so the inverse square root of most points fall into $10^{6}$ due to zero-point problem, resulting in a complete degradation in approximation. One remedy is to simply remove zero from the DE quantization map, which we call DE-0. The smallest number representable by DE-0 is 0.0033. With the B2048/DE-0 quantizer applied to second moment, the approximation of second moment is more precise (Fig. 3), and the training with 4-bit optimizer stabilizes (Tab. 1). However, by removing zero, DE-0 wastes one of the $2^{4} = 16$ quantization points. We propose to use a Linear mapping $\mathbf{T}(i) = (i + 1)/2^b$ , whose smallest representable number is 0.0625. The linear mapping performs better than DE-0 for quantizing second moment.

In Tab. 1, we ablate different quantization schemes and show that excluding zero from the mapping is indeed the key factor for second moment quantization, which cannot be replaced by a smaller block size or/and stochastic rounding [11]. We also test Stable Embedding layers proposed by Dettmers et al. [15], which are reparameterized embedding layers which can be more stably optimized. While Stable Embedding could improve training stability, it cannot fully retain accuracy since non-embedding layers still suffer from zero points. On the other hand, when the zero-point problem is properly addressed, Stable Embedding is no longer required to retain accuracy.

# 4.2 Rank-1 Normalization

We propose an empirically stronger normalization method named rank-1 normalization based on the observation of the heterogeneous outlier pattern in Sec. 3, inspired by the SM3 optimizer [2]. Formally, for a matrix-shaped non-negative (second moment) tensor $\mathbf{x} \in \mathbb{R}^{n \times m}$ , its 1-dimensional

Table 1: Ablation analysis of 4-bit optimizers on the second moment on the GPT-2 Medium E2E-NLG finetuning task. The first line barely turns 8-bit Adam [15] into 4-bit, i.e. B2048/DE for both first and second moments. We only vary the quantization scheme for second moment. SR=stochastic rounding (see App. E.3 for details). Stable Embedding layers are not quantized. 32-bit AdamW achieves a BLEU of 67.7. 

<table><tr><td>Normalization</td><td>Mapping</td><td>Stable Embed.*</td><td>Factorized</td><td>Unstable(%)</td><td>BLEU</td></tr><tr><td>B2048</td><td>DE</td><td> $\times$ </td><td> $\times$ </td><td>33</td><td>66.6 ± 0.61</td></tr><tr><td>B2048</td><td>DE</td><td> $\checkmark$ </td><td> $\times$ </td><td>0</td><td>66.9 ± 0.52</td></tr><tr><td>B128</td><td>DE</td><td> $\times$ </td><td> $\times$ </td><td>66</td><td>65.7 ± N/A</td></tr><tr><td>B128</td><td>DE+SR*</td><td> $\times$ </td><td> $\times$ </td><td>33</td><td>65.4 ± 0.02</td></tr><tr><td>B128</td><td>DE</td><td> $\checkmark$ </td><td> $\times$ </td><td>0</td><td>67.2 ± 1.13</td></tr><tr><td>B2048</td><td>DE-0</td><td> $\times$ </td><td> $\times$ </td><td>0</td><td>67.5 ± 0.97</td></tr><tr><td>B2048</td><td>DE-0</td><td> $\checkmark$ </td><td> $\times$ </td><td>0</td><td>67.1 ± 1.02</td></tr><tr><td>B128</td><td>DE-0</td><td> $\times$ </td><td> $\times$ </td><td>0</td><td>67.4 ± 0.59</td></tr><tr><td>Rank-1</td><td>DE-0</td><td> $\times$ </td><td> $\times$ </td><td>0</td><td>67.5 ± 0.58</td></tr><tr><td>Rank-1</td><td>Linear</td><td> $\times$ </td><td> $\times$ </td><td>0</td><td>67.8 ± 0.51</td></tr><tr><td>Rank-1</td><td>Linear</td><td> $\times$ </td><td> $\checkmark$ </td><td>0</td><td>67.6 ± 0.33</td></tr></table>

statistics $\mathbf{r} \in \mathbb{R}^n$ and $\mathbf{c} \in \mathbb{R}^m$ are defined as $r_i = \max_{1 \leq j \leq m} x_{i,j}$ and $c_j = \max_{1 \leq i \leq n} x_{i,j}$ , which are exactly the quantization scales of per-row and per-column normalization. Rank-1 normalization utilizes two scales jointly and produce a tighter bound for entry, which is defined as

$$
\mathbf {N} _ {\mathrm{rank-1}} (x _ {i, j}) = \frac {1}{\min \{r _ {i} , c _ {j} \}} x _ {i, j}.
$$

Rank-1 normalization could be easily generalized to high-dimensional tensors and signed tensors. See App. G for details and the pseudocode.

Compared with per-tensor, per-token (row), and per-channel (column) normalization, rank-1 normalization utilizes the 1-dimensional information in a more fine-grained manner and gives element-wisely unique quantizaion scales. It deals with the outliers more smartly and effectively when outlier persists in fixed rows or columns but the pattern across tensors are unknown and/or varied (Fig. 2). On the other hand, block-wise normalization is also capable to capture the local information and avoid outliers effectively regardless of the patterns when a small block size is taken, but rank-1 normalization provides a better trade-off between memory overhead and approximation. Rank-1 normalization falls back to per-tensor normalization for 1-dimensional tensors, so we use B128 normalization in those cases. Empirically, rank-1 normalization match or exceed the performance of B128 normalization at a moderate model scale (hidden\_size=1024). It is also possible to combine block-wise and rank-1 normalization together, which we leave for future work.

# 4.3 Factorization

Many memory efficient optimization methods $[2, 8, 46]$ represent the entire second moment with a few number of statistics, which differ from quantization and they take only sublinear memory cost. These works bring more memory saving but they are only applicable to the second moment. In this work, we find the factorization method proposed in Adafactor $[46]$ could also avoid zero-point problem effectively, as shown in Tab. 1. Further, we explore the effects of factorization on second moment based on quantized optimizers to attain maximal memory saving while maintain lossless accuracy. Specifically, when factorization is enabled, we factorize all second moment with dimension greater than 1, and quantize 1d second moment tensors.

# 5 Experiments

We compare our 4-bit optimizers with their full-precision counterparts, as well as other memory efficient optimizers including 8-bit AdamW [15] $^{†}$ , Adafactor [46] and SM3 [2]. 8-bit AdamW's optimizer states are not quantized for embedding layers. For Adafactor, we compare both the

Table 2: Performance on language and vision tasks. Metric: NLU=Mean Accuracy/Correlation. CLS=Accuracy. NLG=BLEU. QA=F1. MT=SacreBleu. $^{\dagger}$ : do not quantize optimizer states for embedding layers; $^{\ddagger}$ : $\beta_{1}=0$ . See more results in App. A. 

<table><tr><td>Optimizer</td><td>NLU RoBERTa-L</td><td>CLS Swin-T</td><td>NLG GPT-2 M</td><td>QA RoBERTa-L</td><td>MT Transformer</td></tr><tr><td>32-bit AdamW</td><td> $88.9 \pm 0.01$ </td><td> $81.2 \pm 0.05$ </td><td> $67.7 \pm 0.67$ </td><td> $94.6 \pm 0.13$ </td><td> $26.61 \pm 0.08$ </td></tr><tr><td>32-bit Adafactor</td><td> $89.1 \pm 0.00$ </td><td> $80.0 \pm 0.03$ </td><td> $67.2 \pm 0.81$ </td><td> $94.6 \pm 0.14$ </td><td> $26.52 \pm 0.02$ </td></tr><tr><td> $32-bit Adafactor^{\ddagger}$ </td><td> $89.3 \pm 0.00$ </td><td> $79.5 \pm 0.05$ </td><td> $67.2 \pm 0.63$ </td><td> $94.7 \pm 0.10$ </td><td> $26.45 \pm 0.16$ </td></tr><tr><td>32-bit SM3</td><td> $87.5 \pm 0.00$ </td><td> $79.0 \pm 0.03$ </td><td> $66.9 \pm 0.58$ </td><td> $91.7 \pm 0.29$ </td><td> $22.72 \pm 0.09$ </td></tr><tr><td> $8-bit AdamW^{\dagger}$ </td><td> $89.1 \pm 0.00$ </td><td> $81.0 \pm 0.01$ </td><td> $67.5 \pm 0.87$ </td><td> $94.5 \pm 0.04$ </td><td> $26.66 \pm 0.10$ </td></tr><tr><td>4-bit AdamW (ours)</td><td> $89.1 \pm 0.01$ </td><td> $80.8 \pm 0.02$ </td><td> $67.8 \pm 0.51$ </td><td> $94.5 \pm 0.10$ </td><td> $26.28 \pm 0.05$ </td></tr><tr><td>4-bit Factor (ours)</td><td> $88.9 \pm 0.00$ </td><td> $80.9 \pm 0.06$ </td><td> $67.6 \pm 0.33$ </td><td> $94.6 \pm 0.20$ </td><td> $26.45 \pm 0.05$ </td></tr></table>

$\beta_{1} > 0$ and the $\beta_{1} = 0$ (no first moment) configuration. For our 4-bit optimizers, we report two versions both based on 32-bit AdamW: (1) “4-bit AdamW” quantizes first moment with B128/DE and second moment with Rank-1/Linear. (2) the more memory efficient “4-bit Factor” quantizes first moment with B128/DE, factorizes second moment when the dimension of tensor is greater than 1, and quantizes lefted 1-dimensional second moment with Rank-1/Linear. See App. D for details about the experiments.

Models, datasets and hyperparameters We report performance metrics on standard benchmarks, including image classification (CLS) with Swin-T $[31]^{\ddagger}$ on ImageNet-1k $[12]$ , natural language understanding (NLU) by fine-tuning RoBERTa-L $[30]^{\S}$ fine-tuning on GLUE $[51]$ , question answering (QA) by fine-tuning RoBERTa-L on SQuAD $[42, 43]$ , natural language generation (NLG) by fine-tuning GPT-2 Medium $[39]^{\parallel}$ on E2E-NLG $[35]$ , machine translation (MT) by training TransformerBase $[50]^{\parallel}$ on WMT14 en-de $[3]$ and LLaMA $[49]$ fine-tuning. We fine-tune LLaMA-7B, LLaMA-13B and LLaMA-33B $[49]$ on the Alpaca dataset $[48]**$ and evaluate them on MMLU $[21]$ and standard common sense reasoning benchmarks: HellaSwag $[56]$ , ARC easy and challenge $[10]$ and OpenBookQA $[33]$ .

We mainly follow the hyperparameters in the original paper or/and codebase. In each benchmark, we keep same hyperparameters in one optimizer on different quantization schemes, which gives an out-of-box transfer from full-precision optimizer to low-bit optimizer without extra hyperparameter tuning. See App. D for hyperparameters and training details.

Accuracy of 4-bit Optimizers We first check whether our memory efficient 4-bit optimizers could retain accuracy. According to Tab. 2, our 4-bit optimizers can match or exceed 32-bit AdamW performance on all fine-tuning tasks (NLU, QA, and NLG) and are comparable on all pretraining tasks (CLS and MT). Sublinear memory optimizers Adafactor ( $\beta_{1}=0$ ) and SM3 could have better memory efficiency, but they suffer from performance degradation, particularly on the CLS task. According to Tab. 3, our 4-bit AdamW will not destroy the capability of pretrained models while enabling them to obtain instruction-following ability. 4-bit AdamW is comparable with 32-bit AdamW on all tasks and does not get worse when the model size grows. Moreover, their convergence curves closely align (Fig. 4).

Memory and Computing Efficiency We evaluate the memory and computation efficiency of proposed 4-bit optimizers on instruction tuning, NLU, and NLG tasks, in Tab. 4. Our 4-bit optimizer offers more memory saving compared to 8-bit optimizers, reducing the training memory consumption by up to 57.7%. It may look like the memory saving saturates when the bitwidth goes down. This is because we report the total memory consumption (including data, activations, and memory fragments)

Table 3: Performance on LLaMA fine-tuning on MMLU and commonsense reasoning tasks across different sizes. 

<table><tr><td>Model</td><td>Optimizer</td><td>MMLU (5-shot)</td><td>HellaSwag</td><td>ARC-e</td><td>ARC-c</td><td>OBQA</td></tr><tr><td rowspan="3">LLaMA-7B</td><td>Original</td><td>33.1</td><td>73.0</td><td>52.4</td><td>40.9</td><td>42.4</td></tr><tr><td>32-bit AdamW</td><td>38.7</td><td>74.6</td><td>61.5</td><td>45.1</td><td>43.4</td></tr><tr><td>4-bit AdamW</td><td>38.9</td><td>74.7</td><td>61.2</td><td>44.4</td><td>43.0</td></tr><tr><td rowspan="3">LLaMA-13B</td><td>Original</td><td>47.4</td><td>76.2</td><td>59.8</td><td>44.5</td><td>42.0</td></tr><tr><td>32-bit AdamW</td><td>46.5</td><td>78.8</td><td>63.6</td><td>48.3</td><td>45.2</td></tr><tr><td>4-bit AdamW</td><td>47.4</td><td>79.0</td><td>64.1</td><td>48.0</td><td>45.2</td></tr><tr><td rowspan="3">LLaMA-33B</td><td>Original</td><td>54.9</td><td>79.3</td><td>58.9</td><td>45.1</td><td>42.2</td></tr><tr><td>32-bit AdamW</td><td>56.4</td><td>79.2</td><td>62.6</td><td>47.1</td><td>43.8</td></tr><tr><td>4-bit AdamW</td><td>54.9</td><td>79.2</td><td>61.6</td><td>46.6</td><td>45.4</td></tr></table>

rather than the optimizer memory consumption alone. In principle, the optimizer states is 2x smaller for 4-bit AdamW than 8-bit AdamW, and about 4x smaller for 4-bit Factor.

The instruction tuning task uses two Nvidia A100 80GB GPUs, while the model is sharded across GPUs with PyTorch's FSDP. In this case, our 4-bit optimizer speeds up training due to reduced communication cost. For the smaller RoBERTa-L and GPT-2 Medium, our 4-bit optimizers appear to be slower than 8-bit AdamW. This is because we have not yet optimize our implementation with fused operators. The speed of our 4-bit optimizers should match or surpass 8-bit optimizers after operator fusion.

We report the largest OPT and LLaMA models trainable under a given memory budget with full-precision and our 4-bit optimizers in Tab. 5. Our optimizer allows for the training of 4x large OPT models, and enables the training of LLaMA-7B model with a single 80GB GPU.

![](images/c0072e4352445e88f6e9528b817f2191405ee150cb04a7df9044ad959c59ed3f.jpg)

<details>
<summary>line</summary>

| Training Step | 32-bit AdamW | 4-bit AdamW |
| ------------- | ------------ | ----------- |
| 0             | 1.1          | 1.4         |
| 400           | 0.7          | 0.7         |
| 800           | 0.4          | 0.4         |
| 1200          | 0.4          | 0.4         |
</details>

Figure 4: Training loss curve of LLaMA-7B fine-tuning on Alpaca dataset (averaged over 3 runs). Only result of 4-bit AdamW is reported since all parameters are 1-dimension via FSDP packing. See more details in App. D.

Ablation Study Finally, we investigate the effectiveness of our proposed quantization schemes and the sensitivity of each moment to quantization in Tab. 6. We see that quantizing both the first and second moment brings a marginal drop in accuracy. Smaller block size on first moment improves accuracy and takes a step towards lossless performance. Factorizing second moment improves accuracy while leads to better memory efficiency.

# 6 Related Work

Compression-based memory efficient optimizers There have been some works trying to approximate the gradient statistics with sublinear memory cost relative to the number of parameters. Adafactor [46] achieves memory reduction by approximating the second-moment of matrix-shaped

Table 4: Memory and Time of 4-bit optimizers compared with 32-bit AdamW and 8-bit Adam [15]. 

<table><tr><td>Task</td><td>Optimizer</td><td>Time</td><td>Total Mem.</td><td>Saved Mem.</td></tr><tr><td rowspan="3">LLaMA-7B</td><td>32-bit AdamW</td><td>3.35 h</td><td>75.40 GB</td><td>0.00 GB (0%)</td></tr><tr><td>4-bit AdamW</td><td>3.07 h</td><td>31.87 GB</td><td>43.54 GB (57.7%)</td></tr><tr><td>4-bit AdamW (fused)</td><td>3.11 h</td><td>31.88 GB</td><td>43.53 GB (57.7%)</td></tr><tr><td rowspan="5">RoBERTa-L</td><td>32-bit AdamW</td><td>3.93 min</td><td>5.31 GB</td><td>0.00 GB (0%)</td></tr><tr><td>8-bit AdamW</td><td>3.38 min</td><td>3.34 GB</td><td>1.97 GB (37.1%)</td></tr><tr><td>4-bit AdamW</td><td>5.59 min</td><td>3.02 GB</td><td>2.29 GB (43.1%)</td></tr><tr><td>4-bit AdamW (fused)</td><td>3.17 min</td><td>3.00 GB</td><td>2.31 GB (43.5%)</td></tr><tr><td>4-bit Factor</td><td>4.97 min</td><td>2.83 GB</td><td>2.48 GB (46.7%)</td></tr><tr><td rowspan="5">GPT-2 Medium</td><td>32-bit AdamW</td><td>2.13 h</td><td>6.89 GB</td><td>0.00 GB (0%)</td></tr><tr><td>8-bit AdamW</td><td>2.04 h</td><td>4.92 GB</td><td>1.97 GB (28.6%)</td></tr><tr><td>4-bit AdamW</td><td>2.43 h</td><td>4.62 GB</td><td>2.37 GB (34.4%)</td></tr><tr><td>4-bit AdamW (fused)</td><td>2.11 h</td><td>4.62 GB</td><td>2.37 GB (34.4%)</td></tr><tr><td>4-bit Factor</td><td>2.30 h</td><td>4.44 GB</td><td>2.45 GB (35.6%)</td></tr></table>

Table 5: Largest trainable model under given memory budget. We use a batch size of 1 and max length of 512 for this comparison. FSDP is enabled at GPUs of 80 GB. 

<table><tr><td rowspan="2">GPU Mem.</td><td colspan="2">Largest fine-tunable Model</td></tr><tr><td>32-bit AdamW</td><td>4-bit AdamW</td></tr><tr><td>24 GB</td><td>OPT-350M</td><td>OPT-1.3B</td></tr><tr><td>80 GB</td><td>OPT-1.3B</td><td>OPT-6.7B</td></tr><tr><td>80 GB</td><td>-</td><td>LLaMA-7B</td></tr></table>

Table 6: Ablation study on the impact of compressing different moments to Swin-T pretraining on ImageNet1k. 

<table><tr><td>Quant. 1st</td><td>Quant. 2nd</td><td>Factor. 2nd</td><td>Acc.</td></tr><tr><td>-</td><td>-</td><td>✗</td><td>81.2 ± 0.05</td></tr><tr><td>B2048/DE</td><td>-</td><td>✗</td><td>80.9 ± 0.04</td></tr><tr><td>B128/DE</td><td>-</td><td>✗</td><td>81.0 ± 0.06</td></tr><tr><td>B128/DE</td><td>Rank-1/Linear</td><td>✗</td><td>80.8 ± 0.02</td></tr><tr><td>B128/DE</td><td>Rank-1/Linear</td><td>√</td><td>80.9 ± 0.06</td></tr></table>

parameters with the outer product of “row” and “column”. SM3 [2] considers the cover of parameters and maintain one statistics for each element in the cover. Experimentally, cover composed of slices of co-dimension 1 for each tensor has been adopted. Extreme tensoring [8], compared with SM3, has a different motivation but similar formulation for the experimental choice. These memory efficient methods only focus on memory reduction on second moment and are applicable to most of the second moment based optimizers, like Adagrad [19], Adam [26], AdamW [32], etc.

Another line of work achieves memory reduction by using compression-based method and maintaining coordinate-wise low-bit precision optimizer states. The stability of 16-bit Adam is firstly studied in DALL-E [44]. Dettmers et al. [15] uses block-wise and dynamic exponent quantization to reduce the coordinate-wise optimizer states from 16-bit to 8-bit and is applicable to SGDM and Adam/AdamW. Compression-based methods only reduce memory by a fixed percentage, irrelevant to the number and shape of parameters. Compression-based methods are less memory efficient compared with sublinear memory methods, but exhibit superior performance across a wider range of benchmarks.

Other memory efficient techniques Some works focus on the memory efficiency of activations. Activation compressed training $[6, 29]$ keeps low-precision activations in GPU memory via quantization in forward phase and dequantize the low-precision activations to full-precision layer-wisely in backward phase. Gradient checkpointing $[7]$ only keeps activation in a small number of layers in forward phase and recompute the activations in all layers when gradient computation is needed, which leads a trade-off between storage overhead of activations and additional computation cost. These methods can be combined with our optimizer for better memory efficiency. LoRA $[24]$ freezes the pretrained weights and only tunes new initialized low-rank parameters, which greatly reduce the size of computation graph but is only applicable to language model finetuning.

Sharding $[40]$ divides model parameters and optimizer states among multiple devices, allowing for more efficient model training through the use of additional GPUs. Offloading $[41, 45]$ reduces GPU memory consumption by transferring data to CPU memory. Our optimizer can be utilized by these methods to reduce the communication overhead.

# 7 Conclusions, Limitations, and Broader Impact

We propose compression-based memory efficient 4-bit optimizers using quantization and factorization. We observe the heterogeneous outlier patterns in optimizer states and identify the zero-point problem in quantizing second moment as the main bottleneck. 4-bit optimizers achieve lossless performance in finetuning and comparable accuracy in pretraining on a wide range of tasks.

Limitations The optimal quantization setting probably depends on task, datasets, and training details. While rank-1 normalization and linear mapping for second moment quantization performs consistently well in our experiments, task-specific quantization settings not in the scope of the study might perform better and be helpful to achieve lossless performance. Our evaluation is currently limited to language and vision tasks, while the applicability of our method to reinforcement learning, audios, and graph learning tasks still needs further study.

Broader Impact Our work can facilitate the access to large models for pretraining and finetuning, which were previously constrained by GPU memory limitations. This could help democratizing large models and opens up new avenues for research that were previously unattainable due to restricted GPU memory, especially benefiting researchers with limited resources. However, our work can also exaggerate the abuse large models.

# Acknowledgements

This work was supported by the National Key Research and Development Program of China (No. 2021ZD0110502), NSFC Projects (Nos. 62061136001, 62106123, 62076147, U19A2081, 61972224, 62106120), Tsinghua Institute for Guo Qiang, and the High Performance Computing Center, Tsinghua University. J.Z is also supported by the XPlorer Prize.

# References

[1] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communication-efficient sgd via gradient quantization and encoding. Advances in Neural Information Processing Systems, 30, 2017.   
[2] Rohan Anil, Vineet Gupta, Tomer Koren, and Yoram Singer. Memory efficient adaptive optimization. Advances in Neural Information Processing Systems, 32, 2019.   
[3] Ondřej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, et al. Findings of the 2014 workshop on statistical machine translation. In Proceedings of the ninth workshop on statistical machine translation, pages 12–58, 2014.   
[4] Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Understanding and overcoming the challenges of efficient transformer quantization. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7947–7969, 2021.   
[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.   
[6] Jianfei Chen, Lianmin Zheng, Zhewei Yao, Dequan Wang, Ion Stoica, Michael Mahoney, and Joseph Gonzalez. Actnn: Reducing training memory footprint via 2-bit activation compressed training. In International Conference on Machine Learning, pages 1803–1813. PMLR, 2021.   
[7] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

[8] Xinyi Chen, Naman Agarwal, Elad Hazan, Cyril Zhang, and Yi Zhang. Extreme tensoring for low-memory preconditioning. In International Conference on Learning Representations, 2020.   
[9] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.   
[10] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.   
[11] Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Binaryconnect: Training deep neural networks with binary weights during propagations. In Advances in Neural Information Processing Systems, pages 3123–3131, 2015.   
[12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, 2009.   
[13] Tim Dettmers. 8-bit approximations for parallelism in deep learning. arXiv preprint arXiv:1511.04561, 2015.   
[14] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm. int8(): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022a.   
[15] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization. In International Conference on Learning Representations, 2022b.   
[16] Tim Dettmers and Luke Zettlemoyer. The case for 4-bit precision: k-bit inference scaling laws. arXiv preprint arXiv:2212.09720, 2022c.   
[17] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019.   
[18] Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models. arXiv preprint arXiv:2203.06904, 2022.   
[19] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011.   
[20] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.   
[21] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.   
[22] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack William Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.   
[23] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019.

[24] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.   
[25] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2704–2713, 2018.   
[26] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.   
[27] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045-3059, 2021.   
[28] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.   
[29] Xiaoxuan Liu, Lianmin Zheng, Dequan Wang, Yukuo Cen, Weize Chen, Xu Han, Jianfei Chen, Zhiyuan Liu, Jie Tang, Joey Gonzalez, et al. Gact: Activation compressed training for general architectures. arXiv preprint arXiv:2206.11357, 2022.   
[30] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.   
[31] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.   
[32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[33] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.   
[34] Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2003.   
[35] Jekaterina Novikova, Ondřej Dušek, and Verena Rieser. The e2e dataset: New challenges for end-to-end generation. arXiv preprint arXiv:1706.09254, 2017.   
[36] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. arXiv preprint arXiv:1904.01038, 2019.   
[37] Gunho Park, Baeseong Park, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee. nuqmm: Quantized matmul for efficient inference of large-scale generative language models. arXiv preprint arXiv:2206.09557, 2022.   
[38] Ning Qian. On the momentum term in gradient descent learning algorithms. Neural networks, 12(1):145–151, 1999.   
[39] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
[40] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

[41] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. Zero-infinity: Breaking the gpu memory wall for extreme scale deep learning. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–14, 2021.   
[42] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for squad. arXiv preprint arXiv:1806.03822, 2018.   
[43] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.   
[44] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021.   
[45] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. {ZeRO-Offload}: Democratizing {Billion-Scale} model training. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pages 551–564, 2021.   
[46] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR, 2018.   
[47] Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In International Conference on Machine Learning, pages 1139–1147. PMLR, 2013.   
[48] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.   
[49] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
[50] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.   
[51] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.   
[52] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.   
[53] Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. arXiv preprint arXiv:2209.13325, 2022.   
[54] Guangxuan Xiao, Ji Lin, Mickael Seznec, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. arXiv preprint arXiv:2211.10438, 2022.   
[55] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. Advances in Neural Information Processing Systems, 35:27168–27183, 2022.   
[56] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

[57] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130b: An open bilingual pre-trained model. In International Conference on Learning Representations, 2023.   
[58] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

# A Additional Experiment Results

In this section, we show additional experiment results beyond Tab. 2. Tab. 7 shows the results of RoBERTa-L finetuning on each task in GLUE datasets. Tab. 8 shows the results of GPT-2 Medium finetuning on E2E-NLG via different metrics. Tab. 9 shows the EM and F1 of RoBERTa-L finetuning on SQuAD and SQuAD 2.0 datasets.

Table 7: Performance of RoBERTa-Large finetuning on GLUE with diverse optimizers. Medians and std over 5 runs are reported on all tasks. $\dagger$ : do not quantize optimizer states for embedding layers; $\ddagger$ : $\beta_{1}=0$ . 

<table><tr><td>Optimizer</td><td>MNLI</td><td>QNLI</td><td>QQP</td><td>RTE</td><td>SST-2</td><td>MRPC</td><td>CoLA</td><td>STS-B</td></tr><tr><td>32-bit AdamW</td><td> $90.2 \pm 0.00$ </td><td> $94.9 \pm 0.00$ </td><td> $92.2 \pm 0.00$ </td><td> $85.2 \pm 0.14$ </td><td> $96.3 \pm 0.00$ </td><td> $93.2 \pm 0.01$ </td><td> $66.9 \pm 0.01$ </td><td> $92.3 \pm 0.00$ </td></tr><tr><td>32-bit Adafactor</td><td> $90.4 \pm 0.00$ </td><td> $94.7 \pm 0.00$ </td><td> $92.2 \pm 0.00$ </td><td> $85.9 \pm 0.02$ </td><td> $96.3 \pm 0.00$ </td><td> $92.8 \pm 0.00$ </td><td> $67.3 \pm 0.01$ </td><td> $92.3 \pm 0.00$ </td></tr><tr><td> $32-bit Adafactor^{\dagger}$ </td><td> $90.5 \pm 0.00$ </td><td> $94.8 \pm 0.00$ </td><td> $92.2 \pm 0.00$ </td><td> $87.0 \pm 0.03$ </td><td> $96.3 \pm 0.00$ </td><td> $92.9 \pm 0.00$ </td><td> $68.2 \pm 0.01$ </td><td> $92.2 \pm 0.00$ </td></tr><tr><td>32-bit SM3</td><td> $90.6 \pm 0.00$ </td><td> $94.2 \pm 0.00$ </td><td> $89.5 \pm 0.00$ </td><td> $85.2 \pm 0.02$ </td><td> $96.0 \pm 0.00$ </td><td> $90.5 \pm 0.01$ </td><td> $62.3 \pm 0.04$ </td><td> $91.4 \pm 0.01$ </td></tr><tr><td> $8-bit AdamW^{\dagger}$ </td><td> $90.4 \pm 0.00$ </td><td> $94.8 \pm 0.00$ </td><td> $92.2 \pm 0.00$ </td><td> $84.8 \pm 0.02$ </td><td> $96.2 \pm 0.00$ </td><td> $93.2 \pm 0.00$ </td><td> $68.0 \pm 0.00$ </td><td> $92.2 \pm 0.00$ </td></tr><tr><td>4-bit AdamW</td><td> $90.2 \pm 0.00$ </td><td> $94.5 \pm 0.00$ </td><td> $92.0 \pm 0.00$ </td><td> $85.2 \pm 0.12$ </td><td> $96.3 \pm 0.00$ </td><td> $92.8 \pm 0.00$ </td><td> $67.3 \pm 0.01$ </td><td> $92.5 \pm 0.00$ </td></tr><tr><td>4-bit Factor</td><td> $90.1 \pm 0.00$ </td><td> $94.7 \pm 0.00$ </td><td> $92.2 \pm 0.00$ </td><td> $85.9 \pm 0.00$ </td><td> $96.4 \pm 0.00$ </td><td> $92.7 \pm 0.00$ </td><td> $68.1 \pm 0.00$ </td><td> $92.3 \pm 0.00$ </td></tr></table>

Table 8: Performance of GPT-2 Medium finetuning on E2E-NLG Challenge with diverse optimizers. Means and std over 3 runs are reported. 

<table><tr><td>Optimizer</td><td>BLEU</td><td>NIST</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr</td></tr><tr><td>32-bit AdamW</td><td>67.7 ± 0.67</td><td>8.60 ± 0.08</td><td>45.7 ± 0.28</td><td>68.7 ± 0.61</td><td>2.35 ± 0.04</td></tr><tr><td>32-bit Adafactor</td><td>67.2 ± 0.81</td><td>8.61 ± 0.60</td><td>45.3 ± 0.08</td><td>68.3 ± 0.22</td><td>2.35 ± 0.01</td></tr><tr><td> $32-bit \text{Adafactor}^{\ddagger}$ </td><td>67.2 ± 0.63</td><td>8.54 ± 0.09</td><td>45.6 ± 0.32</td><td>68.5 ± 0.30</td><td>2.32 ± 0.02</td></tr><tr><td>32-bit SM3</td><td>66.9 ± 0.58</td><td>8.59 ± 0.04</td><td>45.4 ± 0.32</td><td>68.2 ± 0.49</td><td>2.33 ± 0.03</td></tr><tr><td>8-bit  $AdamW^†$ </td><td>67.5 ± 0.87</td><td>8.59 ± 0.08</td><td>45.7 ± 0.52</td><td>68.7 ± 0.97</td><td>2.34 ± 0.06</td></tr><tr><td>4-bit AdamW</td><td>67.8 ± 0.51</td><td>8.61 ± 0.08</td><td>45.8 ± 0.23</td><td>68.9 ± 0.33</td><td>2.35 ± 0.07</td></tr><tr><td>4-bit Factor</td><td>67.6 ± 0.33</td><td>8.59 ± 0.03</td><td>45.6 ± 0.43</td><td>68.6 ± 0.60</td><td>2.34 ± 0.06</td></tr></table>

Table 9: Performance of RoBERTa-Large on SQuAD and SQuAD 2.0 with diverse optimizers. Medians and std over 5 runs are reported. 

<table><tr><td rowspan="2">Optimizer</td><td colspan="2">SQuAD</td><td colspan="2">SQuAD 2.0</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>32-bit AdamW</td><td> $89.0 \pm 0.10$ </td><td> $94.6 \pm 0.13$ </td><td> $85.8 \pm 0.18$ </td><td> $88.8 \pm 0.15$ </td></tr><tr><td>32-bit Adafactor</td><td> $88.8 \pm 0.12$ </td><td> $94.6 \pm 0.14$ </td><td> $85.8 \pm 0.44$ </td><td> $88.7 \pm 0.21$ </td></tr><tr><td> $32-bit Adafactor^{\dagger}$ </td><td> $89.0 \pm 0.18$ </td><td> $94.7 \pm 0.10$ </td><td> $85.9 \pm 0.15$ </td><td> $88.8 \pm 0.15$ </td></tr><tr><td>32-bit SM3</td><td> $84.2 \pm 0.49$ </td><td> $91.7 \pm 0.29$ </td><td> $77.2 \pm 0.71$ </td><td> $81.1 \pm 0.66$ </td></tr><tr><td> $8-bit AdamW^{\dagger}$ </td><td> $88.8 \pm 0.15$ </td><td> $94.5 \pm 0.04$ </td><td> $86.1 \pm 0.26$ </td><td> $89.0 \pm 0.26$ </td></tr><tr><td>4-bit AdamW</td><td> $88.8 \pm 0.08$ </td><td> $94.5 \pm 0.10$ </td><td> $85.4 \pm 0.28$ </td><td> $88.4 \pm 0.26$ </td></tr><tr><td>4-bit Factor</td><td> $88.8 \pm 0.38$ </td><td> $94.6 \pm 0.20$ </td><td> $85.9 \pm 0.36$ </td><td> $88.9 \pm 0.18$ </td></tr></table>

# B Outlier Patterns of Moments

In this section, we give a comprehensive visualization about the outlier pattern of optimizer states. [2] did similar analysis for Adagrad's second moment but here we give a better demonstration about various patterns in optimizer states. The same technique has been applied to parameters and activations in [54].

The outlier pattern of first moment depends on many factors such as data and training hyperparameters. Here we mainly focus on different transformer models and different layers inside. In one transformer block, there is one Attention module and one MLP module, including 6 main parameter matrices. We do not focus on additional parameters including bias and parameters in layer normalization (LN) since they only account a small portion. We denote the 6 matrices by $\mathbf{W}^Q$ , $\mathbf{W}^K$ , $\mathbf{W}^V$ , $\mathbf{W}^O$ , $\mathbf{W}^1$ , $\mathbf{W}^2$ , respectively. When the matrices across different layers are involved at the same time, we add a subscript indicating layer index. Note that $\mathbf{W}$ has shape $\mathbb{R}^{C_{\mathrm{o}} \times C_{\mathrm{i}}}$ in a linear layer, we call the output and input dimension by dim0/row and dim1/column, respectively. The per-channel quantization used in other works actually correspond to per-row(dim0) normalization here.

Swin Transformer ImageNet pretraining In Fig. 5,6,7, the magnitude of first moment in transformer blocks at different depths are shown. It can be seen that the 1-dimensional structure in all parameter matrices are vague at the initial layer. At layer 2, the pattern in $W^{O}$ and $W^{1}$ , that outliers occur at fixed columns, becomes obvious while other parameter matrices remain noisy. At layer 3, the patterns in $W^{O}$ , $W^{1}$ , $W^{2}$ are quite obvious. In $W^{O}$ and $W^{2}$ , the outliers occur at fixed rows while the pattern in $W^{1}$ remain unchanged. The 1-dimensional structure in $W^{Q}$ , $W^{K}$ , $W^{V}$ also seem to appear even though not remarkable. It is notable that the pattern of same parameter matrix at different depths are not necessarily same. See $W^{O}$ in Fig. 6,7.

![](images/d7a14c027203b657dfdda909163298ceda43ff368f31d9e3b8b46b0581909fe5.jpg)

![](images/140a1f4e6abd657e447d10e324f9da25be01b508e4510e1a1a159e3001e12636.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 20   | 0    | 0    |
| 40   | 0    | 0    |
| 60   | 0    | 0    |
| 80   | 0    | 0    |
| 100  | 0    | 0    |
| 120  | 0    | 0    |
| 140  | 0    | 0    |
| 160  | 0    | 0    |
| 180  | 0    | 0    |
| 200  | 0    | 0    |
| 220  | 0    | 0    |
| 240  | 0    | 0    |
| 260  | 0    | 0    |
| 280  | 0    | 0    |
| 300  | 0    | 0    |
| 320  | 0    | 0    |
| 340  | 0    | 0    |
| 360  | 0    | 0    |
| 380  | 0    | 0    |
| 400  | 0    | 0    |
| 420  | 0    | 0    |
| 440  | 0    | 0    |
| 460  | 0    | 0    |
| 480  | 0    | 0    |
| 500  | 0    | 0    |
| 520  | 0    | 0    |
| 540  | 0    | 0    |
| 560  | 0    | 0    |
| 580  | 0    | 0    |
| 600  | 0    | 0    |
| 620  | 0    | 0    |
| 640  | 0    | 0    |
| 660  | 0    | 0    |
| 680  | 0    | 0    |
| 700  | 0    | 0    |
| 720  | 0    | 0    |
| 740  | 0    | 0    |
| 760  | 0    | 0    |
| 780  | 0    | 0    |
| 800  | 0    | 0    |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ... nan|
| ... , dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim11, dim13, dim15, dim17, dim19, dim21, dim23, dim25, dim27, dim29, dim31, dim33, dim35, dim37, dim39, dim41, dim43, dim45, dim47, dim49, dim51, dim53, dim55, dim57, dim59, dim61, dim63, dim65, dim67, dim69, dim71, dim73, dim75, dim77, dim79, dim81, dim83, dim85, dim87, dim89, dim91, dim93, dim95, dim97, dim99, dim111, dim113, dim115, dim117, dim119, dim121, dim123, dim125, dim127, dim129, dim131, dim133, dim135, dim137, dim139, dim141, dim143, dim145, dim147, dim149, dim151, dim153, dim155, dim157, dim159, dim161, dim163, dim165, dim167, dim169, dim171, dim173, dim175, dim177, dim179, dim181, dim183, dim185, dim187, dim189, dim191, dim193, dim195, dim197, dim199, dim211, dim213, dim215, dim217, dim219, dim221, dim223, dim225, dim227, dim229, dim231, dim233, dim235, dim237, dim239, dim241, dim243, dim245, dim247, dim249, dim251, dim253, dim255, dim257, dim259, dim261, dim263, dim265, dim267, dim269, dim271, dim273, dim275, dim277, dim279, dim281, dim283, dim285, dim287, dim289, dim291, dim293, dim295, dim297, dim299, dim311 , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim] , [dim]
</details>

![](images/aaa1036a5bca6f8384a2ffe00386a97486711f92317809902409463d25988278.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 20   | 0    | 0    |
| 40   | 0    | 0    |
| 60   | 0    | 0    |
| 80   | 0    | 0    |
| 100  | 0    | 0    |
| 120  | 0    | 0    |
| 140  | 0    | 0    |
| 160  | 0    | 0    |
| 180  | 0    | 0    |
| 200  | 0    | 0    |
| 220  | 0    | 0    |
| 240  | 0    | 0    |
| 260  | 0    | 0    |
| 280  | 0    | 0    |
| 300  | 0    | 0    |
| 320  | 0    | 0    |
| 340  | 0    | 0    |
| 360  | 0    | 0    |
| 380  | 0    | 0    |
| 400  | 0    | 0    |
| 420  | 0    | 0    |
| 440  | 0    | 0    |
| 460  | 0    | 0    |
| 480  | 0    | 0    |
| 500  | 0    | 0    |
| 520  | 0    | 0    |
| 540  | 0    | 0    |
| 560  | 0    | 0    |
| 580  | 0    | 0    |
| 600  | 0    | 0    |
| 620  | 0    | 0    |
| 640  | 0    | 0    |
| 660  | 0    | 0    |
| 680  | 0    | 0    |
| 700  | 0    | 0    |
| 720  | 0    | 0    |
| 740  | 0    | 0    |
| 760  | 0    | 0    |
| 780  | 0    | 0    |
| 800  | 0    | 0    |
| ...  | ...   | ...  |
| ...+   | ...   | ...+
</details>

![](images/0e180c74758a655c697d94cc00faa22bdbb9d009c9c5b35ada5b3e434dd8c0b7.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 20   | 0    | 0.4   |
| 40   | 0    | 0.6   |
| 60   | 0    | 0.8   |
| 80   | 0    | 1.0   |
</details>

![](images/6e0a088fb03941ff8c891537f551f1ebec9ed7684fb6e7dbaa31a78f450d9ceb.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 100  | 0    | 0    |
| 200  | 0    | 0    |
| 300  | 0    | 0    |
| 400  | 0    | 0    |
| 500  | 0    | 0    |
| 600  | 0    | 0    |
| 700  | 0    | 0    |
| 800  | 0    | 0    |
</details>

![](images/76b43a2d5918b0a92feab38c2d2ed3a54fed60d7f880ffa10a9f3949a058190f.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 20   | 0    | 0    |
| 40   | 0    | 0    |
| 60   | 0    | 0    |
| 80   | 0    | 0    |
| 100  | 0    | 0    |
| 0    | 200  | 0    |
| 20   | 200  | 0    |
| 40   | 200  | 0    |
| 60   | 200  | 0    |
| 80   | 200  | 0    |
| 100  | 200  | 0    |
| 0    | 400  | 0    |
| 20   | 400  | 0    |
| 40   | 400  | 0    |
| 60   | 400  | 0    |
| 80   | 400  | 0    |
| 100  | 400  | 0    |
| 0    | 600  | 0    |
| 20   | 600  | 0    |
| 40   | 600  | 0    |
| 60   | 600  | 0    |
| 80   | 600  | 0    |
| 100  | 600  | 0    |
| 0    | 800  | 0    |
| 20   | 800  | 0    |
| 40   | 800  | 0    |
| 60   | 800  | 0    |
| 80   | 800  | 0    |
| 100  | 800  | 0    |
| 0    | 1.0  | 1.0   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   |...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | ...   |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .      |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .     |
| ...  | ...   | .<nl>
</details>

Figure 5: Outlier patterns of first moment in transformer block layers.0.blocks.0 of Swin-T at epoch 210.

![](images/dd79fb26f874545d0f6cf2846b872b5e003e662b672641dfce0fa8c6f11a9894.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 100  | 0    | 0    |
| 200  | 0    | 0    |
| 300  | 0    | 0    |
| 400  | 0    | 0    |
</details>

![](images/8c6c43ae869d35a85d55726c7215f7c6e58134fc47bf5b8e3dcaa8b2a710793d.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 100  | 0    | 0    |
| 200  | 0    | 0    |
| 300  | 0    | 0    |
| 400  | 0    | 0    |
</details>

![](images/762a3dfa2672de0a63d5f86327bf5878d8ac81e72d25b6967f398558cda7050a.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 100  | 0    | 0    |
| 200  | 0    | 0    |
| 300  | 0    | 0    |
| 400  | 0    | 0    |
</details>

![](images/ab366d26e087d23e28cbd437ad565f00139e636ef37eed0fca84bf457720efc9.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 100  | 0    | 0.4   |
| 200  | 0    | 0.6   |
| 300  | 0    | 0.8   |
| 400  | 0    | 1.0   |
</details>

![](images/8c461219336f995ac67ceeeb416d4021174621a3114b5cfac29b0f9ba4396e9b.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 250  | 0    | 0    |
| 500  | 0    | 0    |
| 750  | 0    | 0    |
| 1000 | 0    | 0    |
| 1250 | 0    | 0    |
| 1500 | 0    | 0    |
</details>

![](images/44d0314eca65df9a0fbf1e2c392081901b92cb58741498734fcbab0ddcf263c4.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 100  | 0    | 0    |
| 200  | 0    | 0    |
| 300  | 0    | 0    |
| 400  | 0    | 0    |
</details>

Figure 6: Outlier patterns of first moment in transformer block layers.2.blocks.0 of Swin-T at epoch 210.

![](images/343f9246c2ad369071799738e57df750871678e5e65c30ecd5e3674a0bf35bf4.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
</details>

![](images/4fcf6ac396c64a011dda3c8c5d648323681480e7f5bd0d652d92c47f222c9706.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
</details>

![](images/0a056ea9257db81cd64b1d8c6b3d1e4408d9becafce5500b4c8eeb427f148fde.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
</details>

![](images/c43268c744ee712eeca13b3ff5307da25296b09adcaa75818cce0c45ac7cfa75.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
</details>

![](images/ee4a188c04b2ec65140fac80d1391a306bf7946693a601d33a46928372aa5009.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 500  | 0    | 0    |
| 1000 | 0    | 0    |
| 1500 | 0    | 0    |
| 2000 | 0    | 0    |
| 2500 | 0    | 0    |
| 3000 | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
</details>

![](images/ea23f00ddc945dcbb6688b395b3a9b1a9974de2e9b15420d7913961b092df358.jpg)

<details>
<summary>bar</summary>

| dim0 | 1000dim1 | 2500 | 3000 |
|------|----------|------|------|
| 0    | 0.2      | 0.4  | 0.6  |
| 200  | 0.4      | 0.6  | 0.8  |
| 400  | 0.6      | 0.8  | 1.0  |
| 600  | 0.8      | 1.0  | 1.2  |
| 800  | 1.0      | 1.2  | 1.4  |
</details>

Figure 7: Outlier patterns of first moment in transformer block layers.3.blocks.0 of Swin-T at epoch 210.

RoBERTa-Large GLUE finetuning In Fig. 8,9,10,11,12,13, the magnitude of first moment in transformer blocks of RoBERTa-Large at different depths are shown. At layer 0 and layer 1 (initial layers), patterns in $\mathbf{W}^O$ , $\mathbf{W}^2$ are obvious. At layer 11 and layer 12 (intermediate layers), patterns are all noisy. At layer 22 and layer 23 (last layers), patterns in $\mathbf{W}^Q$ , $\mathbf{W}^K$ are obvious. Patterns in other matrices are weak.

![](images/1a38799495f19df0532b081e585fdd089cfe38f68b7bd6cad3a907b627bfdee8.jpg)

Figure 8: Outlier patterns of first moment in transformer block layer-0 of RoBERTa-Large at epoch 8.   
WQ   
![](images/203168587f6afa38986e42804c9d0cb3185fd8806dfe9af3ccb4bf55f0696f07.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 200  | 0    | 0.4   |
| 400  | 0    | 0.6   |
| 600  | 0    | 0.8   |
| 800  | 0    | 1.0   |
| 1000 | 0    | 0.8   |
</details>

WK   
![](images/06f7735f74e8a2b8e082233ebddcafe21d128a437e55ef80fbed456b99635777.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 1.0  | 0.2  |
| 400  | 1.0  | 0.4  |
| 600  | 1.0  | 0.6  |
| 800  | 1.0  | 0.8  |
| 1000 | 1.0  | 1.0  |
</details>

WV   
![](images/0e8d9f6d0bd365af43e320b15eefa2cba1a6aae52513d22a2e29099951b2643e.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 0    | 1.0  | 0.2  |
| 200  | 1.0  | 0.4  |
| 400  | 1.0  | 0.6  |
| 600  | 1.0  | 0.8  |
| 800  | 1.0  | 1.0  |
| 1000 | 1.0  | 1.0  |
</details>

WO   
![](images/02bc78b5c1385ddb74d4b4b45d0320042ed04716dbfbfa92a25750ff08f3af7f.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 200  | 0    | 0.3   |
| 400  | 0    | 0.4   |
| 600  | 0    | 0.5   |
| 800  | 0    | 0.6   |
| 1000 | 0    | 0.7   |
| 0    | 1000 | 0.8   |
| 200  | 1000 | 0.9   |
| 400  | 1000 | 1.0   |
| 600  | 1000 | 0.9   |
| 800  | 1000 | 0.8   |
| 1000 | 1000 | 0.7   |
</details>

W1   
![](images/427a4a827b83d4f04a483b65f279cbff8071f961eb4bb77ea28fca9cdce2eda7.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 3000 | 600  | 800  |
</details>

W2   
![](images/e35027e7ecb5570aebcab488cef12f49e5f74470cdd273352b718c7ba0a10a4a.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 4000 | 0    | 0    |
</details>

Figure 9: Outlier patterns of first moment in transformer block layer-1 of RoBERTa-Large at epoch 8.

![](images/c7fd6f5b75a9154a37f2103ae13cb2f433c151143adb3c59b1ec5ba74e2a3dbe.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 400  | 600  |
| 400  | 600  | 800  |
| 600  | 800  | 1000 |
</details>

![](images/fa98d340510873c8d5f140f49a8ab4de4516faf0c0e53ac1dffddbe7922da1f3.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | value |
|------|------|-------|
| 200  | 0    | 0.2   |
| 400  | 200  | 0.4   |
| 600  | 400  | 0.6   |
| 800  | 600  | 0.8   |
| 1000 | 800  | 1.0   |
</details>

![](images/26760814565bc3becb7136dfa06671286536bb243a36705cbb557ee75bd81263.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 400  | 800  |
| 400  | 600  | 1000 |
| 600  | 800  | 1200 |
| 800  | 1000 | 1400 |
| 1000 | 1200 | 1600 |
</details>

![](images/50f20f7eacee46b67aafa54fe9e8758e77d550a8f43e718b6b655fda3a7aa64c.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 200  | 0    | 0.4   |
| 400  | 0    | 0.6   |
| 600  | 0    | 0.8   |
| 800  | 0    | 1.0   |
| 1000 | 0    | 0.8   |
| 200  | 200  | 0.6   |
| 400  | 400  | 0.4   |
| 600  | 600  | 0.2   |
| 800  | 800  | 0.4   |
| 1000 | 1000 | 0.6   |
</details>

![](images/9bf87bcbc89ff324c2900a3b627e28155230c431e222cfb348fc6edbbeaf111a.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 1000 | 0    | 0    |
| 2000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
| 5000 | 0    | 0    |
| 6000 | 0    | 0    |
| 7000 | 0    | 0    |
| 8000 | 0    | 0    |
| 9000 | 0    | 0    |
| 10000| 0    | 0    |
</details>

![](images/d14751c762356da4177f438ef1d7b5a79b8105a5d6bc2d497289225ab7640a7d.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
</details>

Figure 10: Outlier patterns of first moment in transformer block layer-11 of RoBERTa-Large at epoch 8.

![](images/52bbf939ab69853992f58ed73766ac6e5efdaeeb3e967fde2326b55b15ca61bb.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 600  | 800  | 1000 |
</details>

![](images/8f4d10eb063f6e69907deffa2b3ad130d27dd8fb1babea4b52606d409915238c.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
</details>

![](images/036a177f143672e464423786eff965b5d7fe0c9b09aef4bd19ee94c07612dbc9.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 1.0  |
| 200  | 0    | 1.0  |
| 400  | 0    | 1.0  |
| 600  | 0    | 1.0  |
| 800  | 0    | 1.0  |
| 1000 | 0    | 1.0  |
</details>

![](images/ee65907dcd7ec8c30d9c35f1b8fc22eba4b802015cbecae54bb73e529c2d36e3.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 200  | 0    | 0.4   |
| 400  | 0    | 0.6   |
| 600  | 0    | 0.8   |
| 800  | 0    | 1.0   |
| 1000 | 0    | 0.8   |
| 200  | 200  | 0.6   |
| 400  | 400  | 0.4   |
| 600  | 600  | 0.2   |
| 800  | 800  | 0.4   |
| 1000 | 1000 | 0.6   |
</details>

![](images/2b17e7795b036c350f484709c97f63fd94bc27d25db05f3732cc3ed00c98b4f0.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 1000 | 0    | 0    |
| 2000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
| 5000 | 0    | 0    |
| 6000 | 0    | 0    |
| 7000 | 0    | 0    |
| 8000 | 0    | 0    |
| 9000 | 0    | 0    |
| 10000| 0    | 0    |
</details>

![](images/269cfe70bd1b24e5e5c252ea838e55236f750b50b4d05edd12373c87e1247ee0.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
</details>

Figure 11: Outlier patterns of first moment in transformer block layer-12 of RoBERTa-Large at epoch 8.

![](images/bd3c7da027302de6c232fa7b8e2dba9d7d55adadbf2aacf4c61687dec3164009.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 1.0  | 0.8  |
| 400  | 1.0  | 0.6  |
| 600  | 1.0  | 0.4  |
| 800  | 1.0  | 0.2  |
| 1000 | 1.0  | 0.1  |
</details>

![](images/0e09baff63ba93bbc0457ac74a420cc59965c88ae276ea53ac1e4be407285617.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 400  | 800  |
| 400  | 600  | 1000 |
| 600  | 800  | 1000 |
| 800  | 1000 | 1000 |
</details>

![](images/a2696e867657413ca92f422ea54025439ccea658583a1686e489d6c19f5f64d8.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 1.0  | 0.8  |
| 400  | 1.0  | 0.6  |
| 600  | 1.0  | 0.4  |
| 800  | 1.0  | 0.2  |
| 1000 | 1.0  | 0.8  |
</details>

![](images/48d9b02248c3610b35156bbdcb87c5ce850de9eceb4f5bc283e5387848987018.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 200  | 0    | 0.4   |
| 400  | 0    | 0.6   |
| 600  | 0    | 0.8   |
| 800  | 0    | 1.0   |
| 1000 | 0    | 0.8   |
| 200  | 400  | 0.6   |
| 400  | 400  | 0.4   |
| 600  | 400  | 0.2   |
| 800  | 400  | 0.4   |
| 1000 | 400  | 0.6   |
| 200  | 800  | 0.8   |
| 400  | 800  | 1.0   |
| 600  | 800  | 0.8   |
| 800  | 800  | 0.6   |
| 1000 | 800  | 0.4   |
| 200  | 1200 | 0.2   |
| 400  | 1200 | 0.4   |
| 600  | 1200 | 0.6   |
| 800  | 1200 | 0.8   |
| 1000 | 1200 | 1.0   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  |...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | ...   |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .        |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .       |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .      |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .      |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...  | .     |
| ...  | ...    | .     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    |..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | ..     |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .     |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | .      |
| ..    | ..    | ~      |

Note: The values in the table represent the magnitude of the variable 'WO' for each dimension 'dim'. The values are estimated based on the provided code.
</details>

![](images/54bdad1d0049964fe7bceef1c8456d4f7d0b5bea4fb63fcb481c0fb32b3e8eea.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 1000 | 0    | 0    |
| 2000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
| 5000 | 0    | 0    |
| 6000 | 0    | 0    |
| 7000 | 0    | 0    |
| 8000 | 0    | 0    |
| 9000 | 0    | 0    |
| 10000| 0    | 0    |
</details>

![](images/af52c757a4f6722cec3d2818151b7eddbd334a6e02916f5890dd1c8dff2b0c8b.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 1200 | 0    | 0    |
| 1400 | 0    | 0    |
| 1600 | 0    | 0    |
| 1800 | 0    | 0    |
| 2000 | 0    | 0    |
| 2200 | 0    | 0    |
| 2400 | 0    | 0    |
| 2600 | 0    | 0    |
| 2800 | 0    | 0    |
| 3000 | 0    | 0    |
| 3200 | 0    | 0    |
| 3400 | 0    | 0    |
| 3600 | 0    | 0    |
| 3800 | 0    | 0    |
| 4000 | 0    | 0    |
</details>

Figure 12: Outlier patterns of first moment in transformer block layer-22 of RoBERTa-Large at epoch 8.

![](images/4d8e95f02eaf6904eebe4785e942973b92a49856ad010f5bdc426d0fb4ef2393.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 0    | 0    |
|
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0.5  | 1.0  |
| 800  | 1.0  | 1.0  |
| 1000 | 1.0  | 1.0  |
</details>

![](images/f80f2898e1e573c40cdbcea830c37d6369877ed0a6b3f0b25475dba2de9edbd3.jpg)

<details>
<summary>area_stacked</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
</details>

![](images/005cdd339e280fcddff7b6361ab0e6e695c09372f8b3acbbcc469ae2cb6274e7.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 1.0  | 1.0  |
| 400  | 1.0  | 1.0  |
| 600  | 1.0  | 1.0  |
| 800  | 1.0  | 1.0  |
| 1000 | 1.0  | 1.0  |
</details>

![](images/36dbbfe0062f232f02ffb41a8e91231ca51f84cf3f249dc0036df3894b5922a9.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 600  | 800  |
| 400  | 800  | 1000 |
| 600  | 1000 | 1200 |
</details>

![](images/46e9ec2679f04104e7168536b4fa7836e4b7b27068bf4e04cf0c9f8902c05068.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 1000 | 0    | 0.4   |
| 2000 | 0    | 0.6   |
| 3000 | 0    | 0.8   |
| 4000 | 0    | 1.0   |
| 5000 | 0    | 0.8   |
| 6000 | 0    | 0.6   |
| 7000 | 0    | 0.4   |
| 8000 | 0    | 0.2   |
</details>

![](images/a0211f8481b6bf18fbf88e42dc6565d1062a66679648bd92a8bc74d35fc643c0.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
</details>

Figure 13: Outlier patterns of first moment in transformer block layer-23 of RoBERTa-Large at epoch 8.

GPT-2 Medium E2E-NLG finetuning In Fig. 14,15,16,17,18,19, the magnitude of first moment in transformer blocks of GPT-2 Medium at different depths are shown. At layer 1 and layer 2 (initial layers), patterns in $\mathbf{W}^O$ are obvious. At layer 13 and layer 14 (intermediate layers), patterns in $\mathbf{W}^K$ , $\mathbf{W}^O$ are obvious. At layer 21 and layer 22 (last layers), patterns in $\mathbf{W}^Q$ , $\mathbf{W}^K$ , $\mathbf{W}^V$ , $\mathbf{W}^O$ are obvious. First moment of $\mathbf{W}^1$ , $\mathbf{W}^2$ are consistently noisy throughout layers. It is notable that the rows(or columns) that gather outliers are different across different layers.

![](images/d4232102c7a93f3ab42292603e787e980bf06538b410cadc74258bf94cedc2d8.jpg)

Figure 14: Outlier patterns of first moment in transformer block layer-1 of GPT-2 Medium at epoch 2.   
WQ   
![](images/98d00870d25d6d4b341e4e57c2e8f2c59236b23f095f3b294a7c2c91a36257e7.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 1200 | 0    | 0    |
| 1400 | 0    | 0    |
| 1600 | 0    | 0    |
| 1800 | 0    | 0    |
| 2000 | 0    | 0    |
| 2200 | 0    | 0    |
| 2400 | 0    | 0    |
| 2600 | 0    | 0    |
| 2800 | 0    | 0    |
| 3000 | 0    | 0    |
| 3200 | 0    | 0    |
| 3400 | 0    | 0    |
| 3600 | 0    | 0    |
| 3800 | 0    | 0    |
| 4000 | 0    | 0    |
| 4200 | 0    | 0    |
| 4400 | 0    | 0    |
| 4600 | 0    | 0    |
| 4800 | 0    | 0    |
| 5000 | 0    | 0    |
| 5200 | 0    | 0    |
| 5400 | 0    | 0    |
| 5600 | 0    | 0    |
| 5800 | 0    | 0    |
| 6000 | 0    | 0    |
| 6200 | 0    | 0    |
| 6400 | 0    | 0    |
| 6600 | 0    | 0    |
| 6800 | 0    | 0    |
| 7000 | 0    | 0    |
| 7200 | 0    | 0    |
| 7400 | 0    | 0    |
| 7600 | 0    | 0    |
| 7800 | 0    | 0    |
| 8000 | 0    | 0    |
| 8200 | 0    | 0    |
| 8400 | 0    | 0    |
| 8600 | 0    | 0    |
| 8800 | 0    | 0    |
| 9000 | 0    | 0    |
| 9200 | 0    | 0    |
| 9400 | 0    | 0    |
| 9600 | 0    | 0    |
| 9800 | 0    | 0    |
| 12855| -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    | -   | -   |
| -    = -      | -   | -   |
| -    = -      | -   | -   |
| -    = -      | -   | -   |
| -    = -      | -   | -   |
| -    = -      | -   | -   |
| -    = -      | -   | -   |
| -    = -      | -   | -   |
| -    = -      | -   | -   |
| ... (additional values are not labeled in the image). The actual values may vary due to the random nature of the data generation. There is no label for the data series.
</details>

WK   
![](images/288a325cee2f0f5b307b7237b27cbe8fb289dc76dc7acb38bca61008a0f66334.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 400  | 600  | 1.0  |
| 600  | 800  | 0.8  |
| 800  | 1000 | 0.6  |
| 1000 | 1200 | 0.4  |
| 1200 | 1400 | 0.2  |
</details>

WV   
![](images/4fda5177f3e76464722d1ea7ceec2c14a45f636f0916c7bcf00b342728446804.jpg)

WO   
![](images/e7a7115c8a1b8833e38c1605589ee93d8ea774767478f37003f4ab658197ae99.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
</details>

W1   
![](images/143b9518c9a7d7c2adf966d12259cf1463bbc46b327cf43b68ef261d72fd2020.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 1000 | 0    | 0    |
| 2000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
| 5000 | 0    | 0    |
| 6000 | 0    | 0    |
| 7000 | 0    | 0    |
| 8000 | 0    | 0    |
| 9000 | 0    | 0    |
| 10000| 0    | 0    |
</details>

W2   
![](images/4ff820ff46aed99cbcd8d4aa75888d7e7ea660143bcfeb3b63c5f4d050be8d70.jpg)

<details>
<summary>scatter</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 4000 | 0    | 0    |
| 3000 | 0    | 0    |
| 2000 | 0    | 0    |
| 1000 | 0    | 0    |
| 0    | 0    | 0    |
| 200  | 0    | 0.2  |
| 400  | 0    | 0.4  |
| 600  | 0    | 0.6  |
| 800  | 0    | 0.8  |
| 1000 | 0    | 1.0  |
| 400   | 0.2  | 1.2  |
| 300   | 0.4  | 1.4  |
| 200   | 0.6  | 1.6  |
| 100   | 0.8  | 1.8  |
| 50   | 1.0  | 2.0  |
| 250   | 1.2  | 2.2  |
| 150   | 1.4  | 2.4  |
| 75   | 1.6  | 2.6  |
| 50   | 1.8  | 2.8  |
| 25   | 2.0  | 3.0  |
| 15   | 2.2  | 3.2  |
| 7.5  | 2.4  | 3.4  |
| 5    | 2.6  | 3.6  |
| 2.5  | 2.8  | 3.8  |
| 1    | 3.0  | 4.0  |
| -2.5 | 3.2  | 4.2  |
| -5   | 3.4  | 4.4  |
| -7.5 | 3.6  | 4.6  |
| -1     | 3.8  | 4.8  |
| -2.5 | 4.0  | 5.0  |
| -5   | 4.2  | 5.2  |
| -7.5 | 4.4  | 5.4  |
| -1     | 4.6  | 5.6  |
| -2.5 | 4.8  | 5.8  |
| -5   | 5.0  | 6.0  |
| -7.5 | 5.2  | 6.2  |
| -1     | 5.4  | 6.4  |
| -2.5 | 5.6  | 6.6  |
| -5   | 5.8  | 6.8  |
| -7.5 | 6.0  | 7.0  |
| -1     | 6.2  |      |
| -2.5 |      |      |
| -5   |      |      |
| -7.5 |      |      |
| -1     |      |      |
| -2.5 |      |      |
| -5   |      |      |
| -7.5 |      |      |
| -1     |      |      |
| -2.5 |      |      |
| -5   |      |      |
| -7.5 |      |      |
| -1     |      |      |
| -2.5 |      |      |
| -5   |      |      |
| ... (additional values estimated) are not provided in the image.
</details>

Figure 15: Outlier patterns of first moment in transformer block layer-2 of GPT-2 Medium at epoch 2.

![](images/ca7eeec2bf3ef1b292344e27cb94cc4b1b6b557b7dabd67166ac02ea3c89599e.jpg)

<details>
<summary>heatmap</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 400  | 600  |
| 400  | 600  | 800  |
| 600  | 800  | 1000 |
</details>

![](images/1f1b3a517ec5e5acd19a36485d2d7ed631fbdb9ce912f8efc18a5d3185769cf0.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
</details>

![](images/9402e0baddd9ecb0b3bb14261a0a6a9e31ae646dfe9dd5091d875115be560cf7.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
</details>

![](images/884f44e57ce7d0a892d6c6269e6ddd4c8c1d444cde9aa8e484aee079f3278853.jpg)

<details>
<summary>bar</summary>

| dim0 | 400 | 600 | 800 | 1000 |
|------|-----|-----|-----|------|
| 200  | 0.2 | 0.4 | 0.6 | 0.8  |
| 400  | 0.4 | 0.6 | 0.8 | 1.0  |
| 600  | 0.6 | 0.8 | 1.0 | 1.2  |
| 800  | 0.8 | 1.0 | 1.2 | 1.4  |
| 1000 | 1.0 | 1.2 | 1.4 | 1.6  |
</details>

![](images/f3a9ea1be1cb103272de2ff8854fc9efc1ee1d1f91070510454c096d6afbe577.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 2000 | 600  | 0    |
| 3000 | 800  | 0    |
| 4000 | 1000 | 0    |
</details>

![](images/12676a77ece3d63030c084006aa81c6d481dfd0c0513af3b1fa776f8ca9d252b.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
</details>

Figure 16: Outlier patterns of first moment in transformer block layer-13 of GPT-2 Medium at epoch 2.

![](images/5041a04dd5c9005e13b7cb9212f9550739195ef3cf7006f128b40fee80223d84.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 200  | 400  |
| 400  | 400  | 600  |
| 600  | 600  | 800  |
| 800  | 800  | 1000 |
| 1000 | 1000 | 1200 |
</details>

![](images/8e61bbc5a27cc00e00483e8b9567ecd9424b4b416059ee9a95d60426a8e4a0e8.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 1.0  | 1.0  |
| 400  | 1.0  | 1.0  |
| 600  | 1.0  | 1.0  |
| 800  | 1.0  | 1.0  |
| 1000 | 1.0  | 1.0  |
</details>

![](images/c92cb2ae27478b68324f68949db61792e7301c50055b967127cfea8531d0c717.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 600  | 800  | 1000 |
</details>

![](images/9c343492f299385d92fb73e8e890e954fb199431b6c7b550550e1058bd8afa6b.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 200  | 0    | 0.4   |
| 400  | 0    | 0.6   |
| 600  | 0    | 0.8   |
| 800  | 0    | 1.0   |
| 1000 | 0    | 0.8   |
| 200  | 400  | 0.6   |
| 400  | 400  | 0.4   |
| 600  | 400  | 0.2   |
| 800  | 400  | 0.4   |
| 1000 | 400  | 0.6   |
| 200  | 800  | 0.8   |
| 400  | 800  | 1.0   |
| 600  | 800  | 0.8   |
| 800  | 800  | 0.6   |
| 1000 | 800  | 0.4   |
| 200  | 1200 | 0.2   |
| 400  | 1200 | 0.4   |
| 600  | 1200 | 0.6   |
| 800  | 1200 | 0.8   |
| 1000 | 1200 | 1.0   |
| dim1 | dim2 | 60    |
| dim2 | dim3 | 8     |
| dim3 | dim4 | 1     |
| dim4 | dim5 | 2     |
| dim5 | dim6 | 4     |
| dim6 | dim7 | 6     |
| dim7 | dim8 | 8     |
| dim8 | dim9 | 1     |
| dim9 | dim10| 2     |
| dim10| dim11| 4     |
| dim11| dim12| 6     |
| dim12| dim13| 8     |
| dim13| dim14| 1     |
| dim14| dim15| 2     |
| dim15| dim16| 4     |
| dim16| dim17| 6     |
| dim17| dim18| 8     |
| dim18| dim19| 1     |
| dim19| dim20| 2     |
| dim20| dim21| 4     |
| dim21| dim22| 6     |
| dim22| dim23| 8     |
| dim23| dim24| 1     |
| dim24| dim25| 2     |
| dim25| dim26| 4     |
| dim26| dim27| 6     |
| dim27| dim28| 8     |
| dim28| dim29| 1     |
| dim29| dim30| 2     |
| dim30| dim31| 4     |
| dim31| dim32| 6     |
| dim32| dim33| 8     |
| dim33| dim34| 1     |
| dim34| dim35| 2     |
| dim35| dim36| 4     |
| dim36| dim37| 6     |
| dim37| dim38| 8     |
| dim38| dim39| 1     |
| dim39| dim40| 2     |
| dim40| dim41| 4     |
| dim41| dim42| 6     |
| dim42| dim43| 8     |
| dim43| dim44| 1     |
| dim44| dim45| 2     |
| dim45| dim46| 4     |
| dim46| dim47| 6     |
| dim47| dim48| 8     |
| dim48| dim49| 1     |
| dim49| dim50| 2     |
| dim50| dim51| 4     |
| dim51| dim52| 6     |
| dim52| dim53| 8     |
| dim53| dim54| 1     |
| dim54| dim55| 2     |
| dim55| dim56| 4     |
| dim56| dim57| 6     |
| dim57| dim58| 8     |
| dim58| dim59| 1     |
| dim59| dim60| 2     |
| dim60| dim61| 4     |
| dim61| dim62| 6     |
| dim62| dim63| 8     |
| dim63| dim64| 1     |
| dim64| dim65| 2     |
| dim65| dim66| 4     |
| dim66| dim67| 6     |
| dim67| dim68| 8     |
| dim68+ | |      |
</details>

![](images/dd3a215a19b5757ce1e57e0f718d718aa607359432a9bc98d4082d2fc1b49a16.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 1000 | 0    | 0    |
| 2000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
| 5000 | 0    | 0    |
| 6000 | 0    | 0    |
| 7000 | 0    | 0    |
| 8000 | 0    | 0    |
| 9000 | 0    | 0    |
| 10000| 0    | 0    |
</details>

![](images/bba26ae63ac71b6bfa8cdabdd36006a13bbf2cc677ab1a98eecb3a3e8a2546aa.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 4000 | 0    | 0    |
| 2000 | 1.0  | 0.8  |
| 1000 | 1.0  | 1.0  |
| 0    | 1.0  | 1.0  |
</details>

Figure 17: Outlier patterns of first moment in transformer block layer-14 of GPT-2 Medium at epoch 2.

![](images/a60cae81cc143edc17770e6c4fce244c02fa07f2f9a9abfeb89ca64d43b4ac55.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 1000 | 1000 | 1.0  |
</details>

![](images/0f9e55a989b43b8859620ac44c0aa74f7a1fedc9a95536d48e5452aa53400dec.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | value |
|------|------|-------|
| 200  | 600  | 0.2   |
| 400  | 800  | 0.4   |
| 600  | 1000 | 0.6   |
</details>

![](images/968477e861ffac199e3cf14c46a42052fab57db467a065af4b2d4964eeaf3c4e.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 400  | 800  |
| 400  | 600  | 1000 |
| 600  | 800  | 1000 |
</details>

![](images/92d129e1b43d9c1c50572e59cc3595d06362f4c1964df82483f6fc0ea76d3aa6.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | Value |
|------|------|-------|
| 0    | 0    | 0.2   |
| 200  | 0    | 0.4   |
| 400  | 0    | 0.6   |
| 600  | 0    | 0.8   |
| 800  | 0    | 1.0   |
| 1000 | 0    | 0.8   |
| 200  | 200  | 0.6   |
| 400  | 400  | 0.4   |
| 600  | 600  | 0.2   |
| 800  | 800  | 0.4   |
| 1000 | 1000 | 0.6   |
</details>

![](images/40410ecea95d63363de2d0f661e41ee8a8b82e21d8bede5fe6a3221688eb07d4.jpg)

<details>
<summary>line</summary>

| dim0 | 600 | 400 | 200 |
|------|-----|-----|-----|
| 0    | 0   | 0   | 0   |
| 1000 | 0   | 0   | 0   |
| 2000 | 0   | 0   | 0   |
| 3000 | 0   | 0   | 0   |
| 4000 | 0   | 0   | 0   |
| 5000 | 0   | 0   | 0   |
| 6000 | 0   | 0   | 0   |
| 7000 | 0   | 0   | 0   |
| 8000 | 0   | 0   | 0   |
| 9000 | 0   | 0   | 0   |
| 10000| 0   | 0   | 0   |
</details>

![](images/d457f65996b2d555b96aa0061bc352fbc4cdd9405d69a95cb1050f8f74ae16d7.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
</details>

Figure 18: Outlier patterns of first moment in transformer block layer-21 of GPT-2 Medium at epoch 2.

![](images/4c723b61056b48bb5e84387888a9b0cf6881f80cfc78f003515217aeffc70942.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 0    | 0    |
|
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0.5  | 1.0  |
| 800  | 1.0  | 1.0  |
| 1000 | 1.5  | 1.0  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   |...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ...  |
| ...  | ...   | ..
</details>

![](images/cdf470ce142f211998b81e7951c43fe4fd0128e7bc287ee5ea76d52def759f1b.jpg)

<details>
<summary>area_stacked</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
</details>

![](images/7d6378cabe0476b2df61472d9b3c681916aa7d248ab4b5ed320d0d8f55b4b874.jpg)

<details>
<summary>surface_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 200  | 400  | 800  |
| 400  | 600  | 1000 |
| 600  | 800  | 1000 |
| 800  | 1000 | 1000 |
</details>

![](images/bb63fd09ee70c9669a12090abecbc7d50dcce90b7246f80f365a1e57e4451421.jpg)

<details>
<summary>line</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 200  | 1.0  | 0.2  |
| 400  | 1.0  | 0.4  |
| 600  | 1.0  | 0.6  |
| 800  | 1.0  | 0.8  |
| 1000 | 1.0  | 1.0  |
</details>

![](images/aa0780e906eba6ae6bdfb664bfc401ec2ed46bff89d4134f3f7eb1783aacd5f2.jpg)

<details>
<summary>bar</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 1000 | 0    | 0    |
| 2000 | 0    | 0    |
| 3000 | 0    | 0    |
| 4000 | 0    | 0    |
| 5000 | 0    | 0    |
| 6000 | 0    | 0    |
| 7000 | 0    | 0    |
| 8000 | 0    | 0    |
| 9000 | 0    | 0    |
| 10000| 0    | 0    |
</details>

![](images/f98c1b7cbbe6d5c632bad486663536e31ce7e2a14789cbdc8bb0eed87d899b26.jpg)

<details>
<summary>scatter_3d</summary>

| dim0 | dim1 | dim2 |
|------|------|------|
| 0    | 0    | 0    |
| 200  | 0    | 0    |
| 400  | 0    | 0    |
| 600  | 0    | 0    |
| 800  | 0    | 0    |
| 1000 | 0    | 0    |
| 1200 | 0    | 0    |
| 1400 | 0    | 0    |
| 1600 | 0    | 0    |
| 1800 | 0    | 0    |
| 2000 | 0    | 0    |
| 2200 | 0    | 0    |
| 2400 | 0    | 0    |
| 2600 | 0    | 0    |
| 2800 | 0    | 0    |
| 3000 | 0    | 0    |
| 3200 | 0    | 0    |
| 3400 | 0    | 0    |
| 3600 | 0    | 0    |
| 3800 | 0    | 0    |
| 4000 | 0    | 0    |
| 4200 | 0    | 0    |
| 4400 | 0    | 0    |
| 4600 | 0    | 0    |
| 4800 | 0    | 0    |
| 5000 | 0    | 0    |
| 5200 | 0    | 0    |
| 5400 | 0    | 0    |
| 5600 | 0    | 0    |
| 5800 | 0    | 0    |
| 6000 | 0    | 0    |
| 6200 | 0    | 0    |
| 6400 | 0    | 0    |
| 6600 | 0    | 0    |
| 6800 | 0    | 0    |
| 7000 | 0    | 0    |
| 7200 | 0    | 0    |
| 7400 | 0    | 0    |
| 7600 | 0    | 0    |
| 7800 | 0    | 0    |
| 8000 | 0    | 0    |
| 8200 | 0    | 0    |
| 8400 | 0    | 0    |
| 8600 | 0    | 0    |
| 8800 | 0    | 0    |
| 9000 | 0    | 0    |
| 9200 | 0    | 0    |
| 9400 | 0    | 0    |
| 9600 | 0    | 0    |
| 9800 | 0    | 0    |
| 10000| -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
|
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   | -1   |
| -1   | -1   }<fcel>-1   |
| -1   }<fcel>-1   }<fcel>-1   |
| -1   }<fcel>-1   }<fcel>-1   |
| -1   }<fcel>-1   }<fcel>-1   |
| -1   }<fcel>-1   }<fcel>-1   |
| -1   }<fcel>-1   }<fcel>-1   |
| -1   }<fcel>-1   }<fcel>-1   |
| -1   }<fcel>-1   }<fcel>-1   |
| ... (additional values) are not provided in the image. The actual values may vary due to the random nature of the data generation.
</details>

Figure 19: Outlier patterns of first moment in transformer block layer-22 of GPT-2 Medium at epoch 2.

# C Quantization Quality via Histogram

# C.1 Zero-point Problem

In Fig. 20,21,22, we show the effect of zero-point on quantization error for second moment via histogram. All those figures show the negative impact of zero-point on quantizing second moment. After removing zero-point, the quantization quality improves at a great scale.

![](images/29b3a8a75d4f8e7a9588e0870e8a9b5d9f852a905be521c91597a9ffdf06253e.jpg)

Figure 20: Histogram of second moment of the attention layer ( $W^{Q}$ , $W^{K}$ , $W^{V}$ ) in transformer block-wise layer-2 of GPT-2 Medium at epoch 2. In one horizontal line, the first figure is the original second moment. The second figure is the tensor after normalization. The third figure is the quantized tensor. The last figure is the dequantized object. Both the first and last figure is at log10 scale. Both the second and third take values in [0, 1]. All y-axis represents density. Good quantization methods try to make the third figure identical to the second figure and make the last figure identical to the first figure. Top: B128/DE quantization. Bottom: B128/DE-0 quantization.   
![](images/249c4eefad0072cf948b4ae2af52cb99a0078ae5220d96c70d4d5a30ab1701a2.jpg)  
Figure 21: Histogram of second moment of the $W^{V}$ in transformer block layer-10 of RoBERTa-Large at epoch 8. Top: B128/DE quantization. Bottom: B128/DE-0 quantization.

![](images/cfb7bc7826db083a8366609442845fa7a5db98b1760228a7f67e14f02a382fb0.jpg)  
Figure 22: Histogram of second moment of the attention layer ( $W^{Q}$ , $W^{K}$ , $W^{V}$ ) in transformer block layers.0.blocks.0 of Swin-T at epoch 210. Top: B128/DE quantization. Bottom: B128/DE-0 quantization.

# C.2 Comparison between Block-wise and Rank-1 Normalization

To show the differences in quantization error for second moment between block-wise normalization and rank-1 normalization, some cases where rank-1 normalization approximates better than block-wise normalization are shown in Fig. 23, 25, 27. Also, some cases where rank-1 normalization approximates worse than block-wise normalization is shown in Fig. 24, 26, 28. Empirically, it has been observed that rank-1 normalization yields superior results when the distribution exhibits long-distance multimodal characteristics. On the other hand, block-wise normalization tends to outperform when the distribution displays short-distance multimodal patterns and/or intricate local structures.

![](images/f858849cc7fadc0a06f2cb250a6dda1f37e14a58c7d0adddc875cd3fea425d3c.jpg)  
Figure 23: Histogram of second moment of $W^{1}$ in transformer block layer-23 of GPT-2 Medium at epoch 2. A case where rank-1 normalization is better than block-wise normalization with block size 128. In this case, the tail in the right side of distribution is captured by rank-1 normalization but lost in block-wise normalization. Top: B128/DE-0 quantization. Bottom: Rank-1/DE-0 quantization.

# C.3 Effectiveness of Block Size in Block-wise Normalization

In Fig. 29,30,31, we show the effect of block size on quantization error for both first and second moments. Fig. 29,30 shows that B2048 normalization quantizes a significant portion of the points to zero, resulting in poor approximation based on the histogram. However, when we utilize a smaller block size of 128, the quantization performance improves. Fig. 31 shows smaller block size improves quantization quality on second moment.

![](images/f374af13edbe9cbf0d05e7c46771d5258bd0ffa6c833ab26eb6d1bbeed298f35.jpg)

Figure 24: Histogram of second moment of the attention layer $(\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V)$ in transformer block layer-2 of GPT-2 Medium at epoch 2. A case where rank-1 normalization is worse than block-wise normalization with block size 128. Top: B128/DE-0 quantization. Bottom: Rank-1/DE-0 quantization.   
![](images/5257c82d697ba5e389cafde05a10d685acbee99fa523f4d8141e0d4519040983.jpg)

Figure 25: Histogram of second moment of $W^{2}$ in transformer block layer-4 of RoBERTa-Large at epoch 8. A case where rank-1 normalization is better than block-wise normalization with block size 128. Top: B128/DE-0 quantization. Bottom: Rank-1/DE-0 quantization.   
![](images/8b37df3adeae30e9b6e4f86fb640d46546cf87b0705db8f0fabde500f11b76c5.jpg)  
Figure 26: Histogram of second moment of $W^{V}$ in transformer block layer-2 of RoBERTa-Large at epoch 8. A case where rank-1 normalization is worse than block-wise normalization with block size 128. Top: B128/DE-0 quantization. Bottom: Rank-1/DE-0 quantization.

![](images/3ea8d7b4d7ac66de2dd24218bac9dc65343f6f0d7331b79ea83cae8e10e47e4f.jpg)

Figure 27: Histogram of second moment of $W^{2}$ in transformer block layers.0.blocks.0 of Swin-T at epoch 210. A case where rank-1 normalization is better than block-wise normalization with block size 128. Top: B128/DE-0 quantization. Bottom: Rank-1/DE-0 quantization.   
![](images/7d59c9e840a0ac21c347c2d12c8c68a88fba4e2c19666ded4ae86b47a68b6ced.jpg)

Figure 28: Histogram of second moment of $W^{O}$ in transformer block layers.1.blocks.0 of Swin-T at epoch 210. A case where rank-1 normalization is worse than block-wise normalization with block size 128. Top: B128/DE-0 quantization. Bottom: Rank-1/DE-0 quantization.   
![](images/8170418ec9b77704608aa768f0b09ac09af355e4a706381d9b03d268a638de24.jpg)  
Figure 29: Histogram of first moment of $W^{1}$ in transformer block layer-20 of GPT-2 Medium at epoch 2. Top: B128/DE quantization. Bottom: B2048/DE quantization.

![](images/76ba466220cb6340ba3cffbbd85ee55f124d6f1c1d15e7a11a5a6445890a987a.jpg)

Figure 30: Histogram of first moment of $\mathbf{W}^O$ in transformer block layer-22 of RoBERTa-L at epoch 8. Top: B128/DE quantization. Bottom: B2048/DE quantization.   
![](images/a3c3af78fb130f963dada619522b4129376898e6740f8e880df606f6fcfd6276.jpg)  
Figure 31: Histogram of second moment of $W^{1}$ in transformer block layers.0.blocks.0 of Swin-T at epoch 210. Top: B128/DE-0 quantization. Bottom: B2048/DE-0 quantization.

# D Experimental Details

# D.1 Quantization

There are several parameters in deep neural networks that play a delicate role without occupying too much memory, such as normalization layers and bias. In this context, we establish a rule to determine which parameters should not be quantized. For all the experiments we conducted, the rule is straightforward: tensors with a size smaller than or equal to 4096 will not be quantized. However, for larger models with a hidden size exceeding 4096, it is advisable to exclude the bias and normalization layers from quantization. Regarding the quantization settings, as stated in Sec. 5, we employ block-wise normalization with a block size of 128, dynamic exponent mapping for first moment and rank-1 normalization, linear mapping for second moment. When we apply factorization on second moment, only tensors with a dimension greater than or equal to 2 will be factorized while 1-dimensional tensors that meet the specified rule will still be quantized.

8-bit Adam [15] also uses the threshold of 4096 about size to determine whether or not to quantize parameters. Additionally, the implementation in huggingface does not quantize the parameters in Embedding layers regardless of the model used. Consequently, we compare our method with the 8-bit Adam that does not quantize Embedding.

# D.2 Hyperparameters and Training Details

In each benchmark, unless otherwise specified, we maintain the same hyperparameters for a given optimize across different quantization schemes. Additionally, we use same optimizer hyperparameters across various optimizers, including our 4-bit optimizers, 8-bit Adam [15], SM3 [2], Adafactor [46] and the full precision counterpart AdamW [32]. For Adafactor, we use $\beta_{1} > 0$ as default setting which is same as the $\beta_{1}$ value used in AdamW. Also, the case where $\beta_{1} = 0$ is compared. The other newly introduced hyperparameters in Adafactor are set to their default values and remain fixed throughout the experiments. For SM3, we compare with the $\beta_{1} > 0$ configuration, same as the $\beta_{1}$ value used in AdamW.

Table 10: The hyperparameters for RoBERTa-L fine-tuning on GLUE. 

<table><tr><td>Dataset</td><td>MNLI</td><td>QNLI</td><td>QQP</td><td>RTE</td><td>MRPC</td><td>SST-2</td><td>CoLA</td><td>STS-B</td></tr><tr><td>Batch Size</td><td>32</td><td>32</td><td>32</td><td>16</td><td>16</td><td>32</td><td>16</td><td>16</td></tr><tr><td>LR</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>2e-5</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>2e-5</td></tr><tr><td>Warmup</td><td>7432</td><td>1986</td><td>28318</td><td>122</td><td>137</td><td>1256</td><td>320</td><td>214</td></tr><tr><td>Max Train Steps</td><td>123873</td><td>33112</td><td>113272</td><td>2036</td><td>2296</td><td>20935</td><td>5336</td><td>3598</td></tr><tr><td>Max Seq. Len.</td><td>128</td><td>128</td><td>128</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td></tr></table>

Table 11: The hyperparameters for RoBERTa-L fine-tuning on SQuAD and SQuAD 2.0. 

<table><tr><td>Dataset</td><td>SQuAD &amp; SQuAD 2.0</td></tr><tr><td>Batch Size</td><td>48</td></tr><tr><td>LR</td><td>1.5e-5</td></tr><tr><td># Epochs</td><td>2</td></tr><tr><td>Warmup Ratio</td><td>0.06</td></tr><tr><td>Max Seq. Len.</td><td>384</td></tr></table>

Table 12: The hyperparameters for GPT-2 on E2E. 

<table><tr><td>Dataset</td><td>E2E</td></tr><tr><td></td><td>Training</td></tr><tr><td>Batch Size</td><td>8</td></tr><tr><td>LR</td><td>4e-5</td></tr><tr><td># Epochs</td><td>5</td></tr><tr><td>Warmup</td><td>500</td></tr><tr><td>Max Seq. Len.</td><td>512</td></tr><tr><td>Label Smooth</td><td>0.1</td></tr><tr><td></td><td>Inference</td></tr><tr><td>Beam Size</td><td>10</td></tr><tr><td>Length Penalty</td><td>0.8</td></tr><tr><td>no repeat ngram size</td><td>4</td></tr></table>

RoBERTa We train all of our RoBERTa-L models with PyTorch Huggingface $^{††}$ . On GLUE benchmark, we mainly follow the hyperparameters in fairseq [36]. We use $\beta_{1}=0.9$ , $\beta_{2}=0.98$ , $\epsilon=1e-6$ , a weight decay factor of 0.1 and linear learning rate schedule. Other hyperparameters are listed in Tab. 10. On SQuAD benchmark, we mainly follow the reported hyperparameters in RoBERTa paper [30]. We use $\beta_{1}=0.9$ , $\beta_{2}=0.98$ , $\epsilon=1e-6$ , a weight decay factor of 0.01 and linear learning rate schedule. The other hyperparameters are listed in Tab. 11. On both datasets, we report the median and standard deviation results over 5 runs, the result in each run is taken from the best epoch. We utilize single RTX 3090 or 4090 GPU for runs of each task in GLUE datasets and four RTX 3090 or 4090 GPUs for SQuAD and SQuAD 2.0.

On SQuAD 2.0, there may be a performance gap observed between the reproduced results using 32-bit AdamW and the original results reported in the original paper. This is because there are some questions without answers in SQuAD 2.0. It is worth noting that the approach employed by Liu et al. [30] to handle unanswered questions may differ from the solutions utilized in the BERT paper [17], which is the reference implementation we are using

GPT-2 We train all of our GPT-2 Medium models with the LoRA codebase $^{‡‡}$ . We mainly follow the hyperparameters in [28] and [24]. We use $\beta_{1}=0.9$ , $\beta_{2}=0.999$ , $\epsilon=1e-6$ , a weight decay factor of 0.01 and linear learning rate schedule. The other hyperparameters used in GPT-2 are listed in Tab. 12. We report the mean and standard deviation results over 3 runs, the result in each run is taken from the best epoch. We utilize fours RTX 3090 or 4090 GPUs for runs of this task.

Transformer We train all of our Transformer-Base models for machine translation with codebase $^{\S\S}$ . We completely follow the hyperparameters in the codebase. We report the mean and standard deviation results over 3 runs, the result in each run is taken from the best epoch. We utilize eight RTX 3090 or 4090 GPUs for runs of this task.

Swin We train all of our Swin-T models with its official codebase $^{™}$ . We completely follow the hyperparameters in the codebase. We report the mean and standard deviation results over 3 runs, the result in each run is taken from the best epoch. We utilize eight RTX 3090 or 4090 GPUs for runs of this task.

LLaMA We fine-tune LLaMA-7B, LLaMA-13B and LLaMA-33B with Alpaca codebase\*\*\*. We follow the hyperparameters in the codebase for LLaMA-7B and LLaMA-13B, and the hyperparameters of LLaMA-33B are consistent with LLaMA-13B. We fine-tune LLaMA-7B with two A100 80GB GPUs. The training loss curve is the mean results over 3 runs. For LLaMA-7B, we enable Fully Sharded Data Parallelism (FSDP), which packs parameters into 1-dimensional array. This packing process makes it difficult to apply factorization directly without additional engineering efforts. Consequently, we only compare the performance of 4-bit AdamW with its full precision counterpart.

# D.3 Memory and Computing Efficiency

In Tab. 4, we present measurements of memory usage in practical settings, i.e. training configuration described in Sec. D.2. Specifically, we measure the memory usage for LLaMA-7B using 2 A100 80G GPUs, RoBERTa-L using 1 RTX 4090 GPU, and GPT-2 Medium using 4 RTX 4090 GPUs. Additionally, the time measurement for RoBERTa-L is conducted on the RTE task.

# E Quantization Formulation Details

# E.1 Signed Case

In this section, we discuss quantization function for signed tensors and the differences compared to unsigned case. Regarding the normalization operator, the only difference lies in the fact that the sign of the tensor remains unchanged before and after normalization. Formally, let N be the normalization operator for the unsigned cases. For the signed case, the normalization can be defined as

$$
n _ {j} := \operatorname{sign} (x _ {j}) \mathbf {N} (| x _ {j} |).
$$

Therefore, the unit interval for signed case is [-1, 1]. Regarding the mapping operator, the difference lies in the values of quantization mappings. See App. E.2 for more details.

# E.2 Quantization Mappings

In this work, we mainly consider linear mapping and dynamic exponent mapping [13]. See Fig. 32 for illustration of quantization mappings.

```txt
https://github.com/microsoft/LoRA
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer
https://github.com/microsoft/Swin-Transformer
https://github.com/tatsu-lab/stanford_alpaca 
```

![](images/7373e89baefc6dc958e45aa2a19d7ab11e5d1fb693eafd1ebf20c1db2b01ebd8.jpg)

<details>
<summary>line</summary>

| x    | linear | dynamic exponent |
| ---- | ------ | ---------------- |
| 0.0  | -1.00  | -0.90            |
| 0.2  | -0.60  | -0.40            |
| 0.4  | -0.20  | 0.00             |
| 0.6  | 0.20   | 0.10             |
| 0.8  | 0.60   | 0.45             |
| 1.0  | 1.00   | 1.00             |
</details>

![](images/bd07bfc452f86403383638bcb9e9cf444d2ab6425353aa003323dc7f4b7592df.jpg)

<details>
<summary>line</summary>

| x    | linear | dynamic exponent |
| ---- | ------ | ---------------- |
| 0.0  | 0.05   | 0.00             |
| 0.1  | 0.12   | 0.01             |
| 0.2  | 0.20   | 0.02             |
| 0.3  | 0.28   | 0.04             |
| 0.4  | 0.36   | 0.08             |
| 0.5  | 0.44   | 0.16             |
| 0.6  | 0.52   | 0.28             |
| 0.7  | 0.60   | 0.40             |
| 0.8  | 0.68   | 0.52             |
| 0.9  | 0.76   | 0.64             |
| 1.0  | 0.84   | 0.76             |
</details>

Figure 32: Visualization of the quantization mappings for the linear and dynamic exponent at 4-bit precision. Left: Signed case. Right: Unsigned case.

Linear mapping It is notable that the linear mapping considered in our work does not include zero in both signed case and unsigned case. Actually, we only use linear mapping in unsigned case, which is defined as torch.linspace(0, 1, (2 \*\* b) + 1)[1:].

Dynamic exponent mapping Let b be the total bits. In the main text, we mentioned that dynamic exponent takes the form $\mathbf{T}(i) = 10^{-E(i)}\text{fraction}(i)$ . In following paragraphs, we will define the dynamic exponent mapping formally based on the binary representation.

In unsigned case, dynamic exponent mapping [13] is composed of exponent bits E, one indicator bit and fraction bits F, where $b = 1 + E + F$ . It uses the number of leading zero bits E represents the exponent with base 10. The first bit, which is one, serves as an indicator bit that separates the exponent and the unsigned linear fraction. The remaining bits F represent an unsigned linear fraction distributed evenly in (0.1, 1), which is formally defined as

$$
p _ {j} = \frac {1 - 0 . 1}{2 ^ {F}} j + 0. 1, \quad 0 \leq j \leq 2 ^ {F},
$$

$$
\text { fraction } [ k ] = \frac {p _ {k} + p _ {k + 1}}{2}, \quad 0 \leq k \leq 2 ^ {F} - 1.
$$

Therefore, a number with $E$ exponent bits and $F$ fraction bits valued $k$ has a value of

$$
1 0 ^ {- E} \times \text { fraction } [ k ].
$$

For signed case, the only difference is that dynamic exponent mapping additionally uses the first bit as the sign thus we have $b = 1 + E + 1 + F$ . Specially, at 8-bit case, we learn from the codebase $^{\dagger\dagger\dagger}$ that dynamic exponent mapping assign $00000000_{2} = 0_{10}$ , $00000001_{2} = 1_{10}$ in unsigned case and assign $10000000_{2}$ and $00000000_{2}$ with $1_{10}$ and $0_{10}$ , respectively. This means $-1_{10}$ is not defined and the mapping is not symmetric in signed case. Finally, after collecting all the represented numbers and arranging them in a sorted, increasing list, which has a length of $2^{b}$ , the quantization mapping $\mathbf{T}(i)$ returns the i-th element of this list.

The construction of dynamic exponent mapping is unrelated to the number of bits. Therefore, when we say we barely turn the 8-bit optimizer into 4-bit optimizer, it just use 4 total bits. The corner cases mentioned in last paragraph remain unchanged.

# E.3 Stochastic Rounding

Stochastic rounding is only used in Tab. 1. In this section, we talk about how to integrate stochastic rounding into our formulation of quantization. When stochastic rounding is used, the definition of mapping M has some minor changes. Specifically, M is still an element-wise function and defined as

$$
\mathbf {M} (n _ {j}) = \arg \min _ {0 \leq i <   2 ^ {b}} \left\{n _ {j} - \mathbf {T} (i): n _ {j} - \mathbf {T} (i) \geq 0 \right\} \cup \arg \max _ {0 \leq i <   2 ^ {b}} \left\{n _ {j} - \mathbf {T} (i): n _ {j} - \mathbf {T} (i) \leq 0 \right\}.
$$

In other words, $\mathbf{M}$ maps each entry $n_j$ to the maximal index set $\mathbf{M}(n_j)$ such that for any $i \in \mathbf{M}(n_j)$ there is no other $0 \leq k \leq 2^b - 1$ with $\mathbf{T}(k)$ lying between $\mathbf{T}(i)$ and $n_j$ . Actually, $\mathbf{M}$ acts as a filter of $\mathbf{T}$ and give a more fine-grained range of quantized output candidates. In this definition, $\mathbf{M}(n_j)$ has only one or two points since only stochastic rounding is considered in the final step.

Finally, we define stochastic rounding $\mathbf{R}_s$ . When $\mathbf{M}(n_j)$ only has one point, $\mathbf{R}_s$ just output this point. When $\mathbf{M}(n_j)$ has two points $q_1$ and $q_2$ with $\mathbf{T}(q_1) < n_j < \mathbf{T}(q_2)$ , stochastic rounding $(\mathbf{R}_s)$ is defined as

$$
\mathbf {R} _ {s} \left(n _ {j}, q _ {1}, q _ {2}\right) = \left\{ \begin{array}{l} q _ {2}, \text {with proba.} \frac {n _ {j} - \mathbf {T} (q _ {1})}{\mathbf {T} (q _ {2}) - \mathbf {T} (q _ {1})} \\ q _ {1}, \text {with proba.} \frac {\mathbf {T} (q _ {2}) - n _ {j}}{\mathbf {T} (q _ {2}) - \mathbf {T} (q _ {1})} \end{array} \right.
$$

# F Compression-based Memory Efficient Optimizer Instances

In this section, we present some examples about compression-based memory efficient optimizers. See Compression-based Memory Efficient SGDM in Alg. 2 and Adam in Alg. 3.

Algorithm 2 Compression-based Memory Efficient SGDM   
Require: initial parameter $\theta_0\in \mathbb{R}^p$ , learning rate $\alpha$ , initial first moment $\bar{m}_0 = 0$ , total number of iterations $T$ and momentum parameter $\beta$ 1: for $t = 1,2,\ldots ,T$ do   
2: Sample a minibatch $\zeta_t$ and get stochastic gradient $g_{t} = \nabla_{\theta}f(\theta_{t - 1},\zeta_t)$ 3: $m_{t - 1}\gets \mathrm{decompress}(\bar{m}_{t - 1})$ 4: $m_{t}\gets \beta \cdot m_{t - 1} + g_{t}$ 5: $\theta_t\gets \theta_{t - 1} - \alpha \cdot m_t$ 6: $\bar{m}_t\gets \mathrm{compress}(m_t)$ 7: end for   
8: return $\theta_T$

Algorithm 3 Compression-based Memory Efficient Adam   
Require: initial parameter $\theta_0 \in \mathbb{R}^p$ , learning rate $\alpha$ , initial moments $\bar{m}_0 = 0, \bar{v}_0 = 0$ , total number of iterations $T$ and hyperparameters $\beta_1, \beta_2, \epsilon$ .

1: for $t = 1, 2, \ldots, T$ do

2: Sample a minibatch $\zeta_t$ and get stochastic gradient $g_t = \nabla_\theta f(\theta_{t-1}, \zeta_t)$ 3: $m_{t-1}, v_{t-1} \leftarrow \text{decompress}(\bar{m}_{t-1}), \text{decompress}(\bar{v}_{t-1})$ 4: $m_t \leftarrow \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$ 5: $v_t \leftarrow \beta_2 \cdot v_{t-1} + (1 - \beta_1) \cdot g_t^2$ 6: $\hat{m}_t \leftarrow m_t / (1 - \beta_1^t)$ 7: $\hat{v}_t \leftarrow v_t / (1 - \beta_2^t)$ 8: $\theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ 9: $\bar{m}_t, \bar{v}_t \leftarrow \text{compress}(m_t), \text{compress}(v_t)$ 10: end for

11: return $\theta_T$

# G Rank-1 Normalization

In this section, we present the detailed formulation of rank-1 normalization in Alg. 4.

# H Theoretical Analysis

The convergence of low-bit optimizers can be guaranteed if their fp32 counterparts converge. Here we provide a theorem about the convergence of quantized SGDM (Alg. 2) under some assumptions. We believe the convergence of low-bit AdamW could be inferred from the convergence of AdamW.

# Algorithm 4 Rank-1 Normalization

Require: tensor $x \in R^{d_{1} \times \cdots \times d_{p}}$ ; statistics $\mu_{r} \in R^{d_{r}}$ for $1 \leq r \leq p$ ; permutation function $\Phi$ mapping $\{1, \ldots, d\}$ to indices of tensor x, where $d = d_{1} \times \cdots \times d_{p}$ .

1: for $r = 1, 2, \ldots, p$ do   
2: for $j = 1,2,\ldots ,d_r$ do   
3: $\mu_{r,j} = \max_{i_1,\dots,i_{r-1},i_{r+1},\dots,i_p}\left|x_{[i_1,\dots,i_{r-1},j,i_{r+1},\dots,i_p]}\right|$   
4: end for   
5: end for   
6: for i = 1, 2, ..., d do   
7: $M_{i} = \min_{1\leq r\leq p}\mu_{r,\Phi (i)_{r}}$   
8: end for   
9: reshape 1-dimensional array $M$ to the same shape as $x$   
10: return x/M

First, we make some assumptions. The first three are rather standard in stochastic optimization literature, while last two depict properties of stochastic quantizers.

1. (Convexity) The objective function is convex and has an unique global minimum $f(\theta^{*})$ .   
2. (Smoothness) The objective $f(\theta)$ is continuous differentiable and $L$ -smooth;   
3. (Moments of stochastic gradient) The stochastic gradient $g$ is unbiased, i.e., $\mathbb{E}[g(\theta)] = \nabla f(\theta)$ , and has bounded variance, i.e., $\mathbb{E}\left[\| g(\theta) - \nabla f(\theta)\|^2\right] < \sigma^2$ , $\forall \theta \in \mathbb{R}^d$ .   
4. (Unbiased quantizer) $\forall x\in \mathbb{R}^d,\mathbb{E}[Q(x)] = x.$   
5. (Bounded quantization variance) $\forall x\in \mathbb{R}^d,\mathbb{E}\left[\| Q_m(x) - x\|^2\right]\leq \sigma_m^2.$

Then, we have following theorem:

Theorem 1. Consider the Algorithm 2 with Assumptions 1-5. Let $\alpha \in (0, \frac{1 - \beta}{L}]$ , then for all $T > 0$ we have

$$
\begin{array}{l} \mathbb {E} \left[ f \left(\bar {\theta} _ {T}\right) - f _ {*} \right] \leq \frac {1}{2 T} \left(\frac {L \beta}{1 - \beta} + \frac {1 - \beta}{\alpha}\right) \| \theta_ {0} - \theta_ {*} \| ^ {2} \\ + \frac {\alpha \sigma^ {2}}{(1 - \beta)} + \frac {\alpha \sigma_ {m} ^ {2}}{(1 - \beta)}. \tag {2} \\ \end{array}
$$

where $\bar{\theta}_T = \frac{1}{T}\sum_{i = 0}^{T - 1}\theta_i.$

# H.1 Proof of Theorem 1

To prove Theorem 1, we need some useful lemmas.

Lemma 1. In Algorithm 2, The conditional first and second moments of $g_{t}$ satisfies

$$
\mathbb {E} [ g _ {t} | \theta_ {t - 1} ] = \nabla f (\theta_ {t - 1}) \tag {3}
$$

$$
\mathbb {E} \left[ \| g _ {t} \| ^ {2} \mid \theta_ {t - 1} \right] \leq \| \nabla f (\theta_ {t - 1}) \| ^ {2} + \sigma^ {2} \tag {4}
$$

Proof. By assumption, we easily have

$$
\mathbb {E} \left[ g _ {t} | \theta_ {t - 1} \right] = \nabla f (\theta_ {t - 1}).
$$

With Assumption 3, it holds true that

$$
\begin{array}{l} \mathbb {E} \left[ \| g (\theta) \| ^ {2} \right] = \mathbb {E} \left[ \| g (\theta) - \nabla f (\theta) + \nabla f (\theta) \| ^ {2} \right] \\ = \mathbb {E} \left[ \| g (\theta) - \nabla f (\theta) \| ^ {2} \right] + \mathbb {E} \left[ \| \nabla f (\theta) \| ^ {2} \right] + 2 \mathbb {E} [ \langle g (\theta) - \nabla f (\theta), \nabla f (\theta) \rangle ] \\ = \mathbb {E} \left[ \| g (\theta) - \nabla f (\theta) \| ^ {2} \right] + \mathbb {E} \left[ \| \nabla f (\theta) \| ^ {2} \right] \\ \leq \sigma^ {2} + \| \nabla f (\theta) \| ^ {2}, \\ \end{array}
$$

which implies the second part.

![](images/989a2f00f86c48418e82ed73a9d72d50dd0f4b645c7577208cc68400a7d6096d.jpg)

Lemma 2. If Assumptions 3-5 hold, then sequence $\{z_t\}$ satisfies

$$
z _ {t + 1} - z _ {t} = \frac {1}{1 - \beta} (\theta_ {t + 1} - \theta_ {t}) - \frac {\beta}{1 - \beta} (\theta_ {t} - \theta_ {t - 1}) \tag {5}
$$

$$
\mathbb {E} [ z _ {t + 1} - z _ {t} ] = \frac {- \alpha}{1 - \beta} \nabla f (\theta_ {t}) \tag {6}
$$

$$
\mathbb {E} [ \| z _ {t + 1} - z _ {t} \| ^ {2} ] \leq 2 \left(\frac {\alpha}{1 - \beta}\right) ^ {2} \left(\mathbb {E} [ \| g _ {t + 1} \| ^ {2} ] + \sigma_ {m} ^ {2}\right). \tag {7}
$$

Proof. By definition of $z_{t}$ , we have the first equation immediately. Take expectation on the first equation and we get

$$
\mathbb {E} \left[ z _ {t + 1} - z _ {t} \right] = \frac {1}{1 - \beta} \mathbb {E} \left[ \theta_ {t + 1} - \theta_ {t} \right] - \frac {\beta}{1 - \beta} \mathbb {E} \left[ \theta_ {t} - \theta_ {t - 1} \right].
$$

Note that

$$
\begin{array}{l} \mathbb {E} \left[ \theta_ {t + 1} - \theta_ {t} \right] = \mathbb {E} \left[ \theta_ {t + 1} - \left(\theta_ {t} - \alpha m _ {t + 1}\right) \right] - \mathbb {E} \left[ \alpha m _ {t + 1} \right] \\ = - \alpha \mathbb {E} \left[ m _ {t + 1} \right] \\ = - \alpha \mathbb {E} \left[ \beta m _ {t} + g _ {t + 1} \right] \\ = - \alpha \beta \mathbb {E} [ m _ {t} ] - \alpha \nabla f (\theta_ {t}), \\ \end{array}
$$

and

$$
\begin{array}{l} \mathbb {E} \left[ \theta_ {t} - \theta_ {t - 1} \right] = \mathbb {E} \left[ \theta_ {t} - \left(\theta_ {t - 1} - \alpha m _ {t}\right) \right] - \mathbb {E} \left[ \alpha m _ {t} \right] \\ = - \alpha \mathbb {E} \left[ m _ {t} \right], \\ \end{array}
$$

which gives the second equation.

$$
\mathbb {E} [ z _ {t + 1} - z _ {t} ] = \frac {- \alpha}{1 - \beta} \nabla f (\theta_ {t})
$$

For the last equation, since

$$
\begin{array}{l} z _ {t + 1} - z _ {t} = \frac {1}{1 - \beta} (\theta_ {t + 1} - \theta_ {t}) - \frac {\beta}{1 - \beta} (\theta_ {t} - \theta_ {t - 1}) \\ = - \frac {\alpha}{1 - \beta} (m _ {t + 1} - \beta m _ {t}) \\ \end{array}
$$

Take expectation and we have

$$
\begin{array}{l} \mathbb {E} \left[ \| z _ {t + 1} - z _ {t} \| ^ {2} \right] = \left(\frac {\alpha}{1 - \beta}\right) ^ {2} \mathbb {E} \left[ \| m _ {t + 1} - \beta m _ {t} \| ^ {2} \right] \\ \leq 2 \left(\frac {\alpha}{1 - \beta}\right) ^ {2} \left(\mathbb {E} \left[ \| m _ {t + 1} - (\beta m _ {t} + g _ {t + 1}) \| ^ {2} \right] + \mathbb {E} \left[ \| g _ {t + 1} \| ^ {2} \right]\right) \\ \leq 2 \left(\frac {\alpha}{1 - \beta}\right) ^ {2} \left(\mathbb {E} \left[ \| g _ {t + 1} \| ^ {2} \right] + \sigma_ {m} ^ {2}\right). \\ \end{array}
$$

![](images/c4f679f42b8fa01b4e941e5346a32ad6f6a5c349ab5feea2f7f77fdcb0c6a211.jpg)

Proof of Theorem 1. From Lemma 2, we have

$$
\mathbb {E} \left[ \left\| z _ {t + 1} - z _ {t} \right\| ^ {2} \right] \leq 2 \left(\frac {\alpha}{1 - \beta}\right) ^ {2} \left(\mathbb {E} \left[ \left\| g _ {t + 1} \right\| ^ {2} \right] + \sigma_ {m} ^ {2}\right).
$$

Substituting Lemma 1 gives

$$
\mathbb {E} \left[ \| z _ {t + 1} - z _ {t} \| ^ {2} \right] \leq 2 \left(\frac {\alpha}{1 - \beta}\right) ^ {2} \left(\| \nabla f (\theta_ {t}) \| ^ {2} + \sigma^ {2} + \sigma_ {m} ^ {2}\right). \tag {8}
$$

Suppose $\theta_{*}$ is the optimal parameter and $f_{*} = f(\theta_{*})$ is the minimal objective value. First, we have

$$
\left\| z _ {t + 1} - \theta_ {*} \right\| ^ {2} = \left\| z _ {t} - \theta_ {*} \right\| ^ {2} + 2 \left\langle z _ {t} - \theta_ {*}, z _ {t + 1} - z _ {t} \right\rangle + \left\| z _ {t + 1} - z _ {t} \right\| ^ {2}
$$

Take expectation over the randomness in the $(t + 1)$ -th step, we have

$$
\begin{array}{l} \mathbb {E} [ \| z _ {t + 1} - \theta_ {*} \| ^ {2} ] = \| z _ {t} - \theta_ {*} \| ^ {2} - \frac {2 \alpha}{1 - \beta} \langle z _ {t} - \theta_ {*}, \nabla f (\theta_ {t}) \rangle + \mathbb {E} [ \| z _ {t + 1} - z _ {t} \| ^ {2} ] \\ = \left\| z _ {t} - \theta_ {*} \right\| ^ {2} - \frac {2 \alpha}{1 - \beta} \left\langle \theta_ {t} - \theta_ {*}, \nabla f (\theta_ {t}) \right\rangle \\ - \frac {2 \alpha \beta}{(1 - \beta) ^ {2}} \left\langle \theta_ {t} - \theta_ {t - 1}, \nabla f (\theta_ {t}) \right\rangle + \mathbb {E} [ \| z _ {t + 1} - z _ {t} \| ^ {2} ] \\ \end{array}
$$

Since $f$ is continuously differentiable and L-smooth, we have the following inequalities. [34]

$$
\left\langle \theta_ {t} - \theta_ {*}, \nabla f (\theta_ {t}) \right\rangle \geq \frac {1}{L} \| \nabla f (\theta_ {t}) \| ^ {2} \tag {9}
$$

$$
\left\langle \theta_ {t} - \theta_ {*}, \nabla f (\theta_ {t}) \right\rangle \geq f (\theta_ {t}) - f _ {*} + \frac {1}{2 L} \| \nabla f (\theta_ {t}) \| ^ {2} \tag {10}
$$

$$
\left\langle \theta_ {t} - \theta_ {t - 1}, \nabla f (\theta_ {t}) \right\rangle \geq f (\theta_ {t}) - f (\theta_ {t - 1}) \tag {11}
$$

Substitute them and get

$$
\mathbb {E} [ \| z _ {t + 1} - \theta_ {*} \| ^ {2} ] \leq \| z _ {t} - \theta_ {*} \| ^ {2} - \frac {2 \alpha (1 - \rho)}{L (1 - \beta)} \| \nabla f (\theta_ {t}) \| ^ {2} - \frac {2 \alpha \rho}{1 - \beta} (f (\theta_ {t}) - f _ {*})
$$

$$
- \frac {\alpha \rho}{L (1 - \beta)} \| \nabla f (\theta_ {t}) \| ^ {2} - \frac {2 \alpha \beta}{(1 - \beta) ^ {2}} (f (\theta_ {t}) - f (\theta_ {t - 1})) + \mathbb {E} [ \| z _ {t + 1} - z _ {t} \| ^ {2} ]
$$

where $\rho \in (0,1]$ is a parameter used to balance the first two inequalities. Denote $M = 2\left(\frac{\alpha}{1 - \beta}\right)^2 (\sigma^2 +\sigma_m^2)$ . Substitute Eq. 8 into this inequality and collect the terms, we get

$$
\left(\frac {2 \alpha \rho}{1 - \beta} + \frac {2 \alpha \beta}{(1 - \beta) ^ {2}}\right) (f (\theta_ {t}) - f _ {*}) + \mathbb {E} [ \| z _ {t + 1} - \theta_ {*} \| ^ {2} ]
$$

$$
\leq \frac {2 \alpha \beta}{(1 - \beta) ^ {2}} \left(f (\theta_ {t - 1}) - f _ {*}\right) + \| z _ {t} - \theta_ {*} \| ^ {2} + \left(\frac {2 \alpha^ {2}}{(1 - \beta) ^ {2}} - \frac {\alpha (2 - \rho)}{L (1 - \beta)}\right) \| \nabla f (\theta_ {t}) \| ^ {2} + M
$$

When $\alpha$ satisfies the condition $\frac{2\alpha^2}{(1 - \beta)^2} -\frac{\alpha(2 - \rho)}{L(1 - \beta)}\leq 0$ , i.e. $0\leq \alpha \leq \frac{(1 - \beta)(2 - \rho)}{2L}$ , the term about $\| \nabla f(\theta_t)\|^2$ is non-positive, thus we have

$$
\begin{array}{l} \left(\frac {2 \alpha \rho}{1 - \beta} + \frac {2 \alpha \beta}{(1 - \beta) ^ {2}}\right) (f (\theta_ {t}) - f _ {*}) + \mathbb {E} [ \| z _ {t + 1} - \theta_ {*} \| ^ {2} ] \\ \leq \frac {2 \alpha \beta}{(1 - \beta) ^ {2}} \left(f (\theta_ {t - 1}) - f _ {*}\right) + \| z _ {t} - \theta_ {*} \| ^ {2} + M \\ \end{array}
$$

Summing this inequality from 0 to T - 1 and taking full expectation gives

$$
\frac {2 \alpha \rho}{1 - \beta} \sum_ {i = 0} ^ {T - 1} \mathbb {E} [ f (\theta_ {i}) - f _ {*} ] + \sum_ {i = 0} ^ {T - 1} \left(\frac {2 \alpha \beta}{(1 - \beta) ^ {2}} \mathbb {E} [ f (\theta_ {i}) - f _ {*} ] + \mathbb {E} [ \| z _ {i + 1} - \theta_ {*} \| ^ {2} ]\right)
$$

$$
\leq \sum_ {i = 0} ^ {T - 1} \left(\frac {2 \alpha \beta}{(1 - \beta) ^ {2}} \mathbb {E} [ f (\theta_ {i - 1}) - f _ {*} ] + \mathbb {E} [ \| z _ {i} - \theta_ {*} \| ^ {2} ]\right) + T \cdot M
$$

which implies that

$$
\frac {2 \alpha \rho}{1 - \beta} \sum_ {i = 0} ^ {T - 1} \mathbb {E} [ f (\theta_ {i}) - f _ {*} ] \leq \frac {2 \alpha \beta}{(1 - \beta) ^ {2}} (f (\theta_ {0}) - f _ {*}) + \| \theta_ {0} - \theta_ {*} \| ^ {2} + T \cdot M
$$

Since $f$ is convex, we have $Tf(\bar{\theta}_T) \leq \frac{1}{T} \sum_{i=0}^{T-1} f(\theta_i)$ . Subsequently we have

$$
\begin{array}{l} \mathbb {E} \left[ f \left(\bar {\theta} _ {T}\right) - f _ {*} \right] \leq \frac {1}{T} \left(\frac {\beta}{\rho (1 - \beta)} \left(f \left(\theta_ {0}\right) - f _ {*}\right) + \frac {1 - \beta}{2 \alpha \rho} \| \theta_ {0} - \theta_ {*} \| ^ {2}\right) \\ + \frac {1 - \beta}{2 \alpha \rho} M \\ \end{array}
$$

Finally, when $\alpha \in (0, \frac{1 - \beta}{L}]$ , we can take $\rho = 1$ , use L-smooth condition again and substitute $M$ , which gives

$$
\begin{array}{l} \mathbb {E} \left[ f \left(\bar {\theta} _ {T}\right) - f _ {*} \right] \leq \frac {1}{2 T} \left(\frac {L \beta}{1 - \beta} + \frac {1 - \beta}{\alpha}\right) \| \theta_ {0} - \theta_ {*} \| ^ {2} \\ + \frac {\alpha \sigma^ {2}}{(1 - \beta)} + \frac {\alpha \sigma_ {m} ^ {2}}{(1 - \beta)} \\ \end{array}
$$

![](images/7eef4887d28e5799965cb3b49ee5ee971eec03dd97c76aeecabbfeb2476b6527.jpg)