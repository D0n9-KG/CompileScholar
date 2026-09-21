# LVPruning: An Effective yet Simple Language-Guided Vision Token Pruning Approach for Multi-modal Large Language Models

Yizheng Sun $^{1}$ , Yanze Xin $^{2}$ , Hao Li $^{1}$ , Jingyuan Sun $^{1,*}$ , Chenghua Lin $^{1}$ , Riza Batista-Navarro $^{1}$ ,

<sup>1</sup>University of Manchester, <sup>2</sup>Imperial College London,

*Correspondence: jingyuan.sun@manchester.ac.uk

# Abstract

Multi-modal Large Language Models (MLLMs) have achieved remarkable success by integrating visual and textual modalities. However, they incur significant computational overhead due to the large number of vision tokens processed, limiting their practicality in resource-constrained environments. We introduce Language-Guided Vision Token Pruning (LVPruning) for MLLMs, an effective yet simple method that significantly reduces the computational burden while preserving model performance.

LVPruning employs cross-attention modules to compute the importance of vision tokens based on their interaction with language tokens, determining which to prune. Importantly, LVPruning can be integrated without modifying the original MLLM parameters, which makes LVPruning simple to apply or remove.

Our experiments show that LVPruning can effectively reduce up to $90\%$ of vision tokens by the middle layer of LLaVA-1.5, resulting in a $62.1\%$ decrease in inference Tera Floating-Point Operations Per Second (TFLOPs), with an average performance loss of just $0.45\%$ across nine multi-modal benchmarks.

# 1 Introduction

Multi-modal Large Language Models (MLLMs) have achieved impressive results by combining visual and textual information to perform complex tasks that require understanding both modalities (Dai et al., 2023; Li et al., 2023a; Liu et al., 2023a; Sun et al., 2024). However, these models can be highly computationally intensive, limiting their practicality in resource-constrained environments (Liu et al., 2023a,b). One important fact that leads to such substantial computational overhead is that these models often process a large number of vision tokens representing input image patches, but not all visual information is equally important for understanding. The human brain, for instance, can

![](dt=2026-06-04/ht=03/bb58954b4908598f621820742c4fccc4ecb5dc80d396f687da74116897625ef5.jpg)

![](dt=2026-06-04/ht=03/9bb4aea0e243eb038e740fa0206f3d118b346f66c71e0968990f61ed38e784e3.jpg)

focus on salient features while ignoring irrelevant details, allowing for highly efficient visual perception (Treisman, 1988). Inspired by this, there is a growing need to develop MLLMs that can prioritize crucial vision tokens, reducing computational costs without largely sacrificing performance.

Previous approaches to enhance the computational efficiency of MLLMs have explored various strategies. Models utilizing Q-former as the vision encoder condense visual information into a smaller set of tokens. Though effectively reducing computational load, such condensation potentially leads to a loss of essential visual information, compromising performance (Li et al., 2023a; Dai et al., 2023; Zhu et al., 2024).

On the other hand, models such as LLaVA pass all vision tokens through a simple Multi-Layer Perceptron (MLP) connector to the language model, achieving high performance but at the cost of increased computational demands (Liu et al., 2023a,b). Additionally, token compression techniques (Rao et al., 2021; Bolya et al., 2023; Chen et al., 2023) that detect important vision tokens solely based on visual features have shown

arXiv:2501.13652v2 [cs.CL] 9 Mar 2025

promise in single-modal tasks but can not make full of the interaction between visual and linguistic information in MLLMs. These highlight a tradeoff between computational efficiency and model performance, indicating a need for solutions that can balance both aspects effectively and efficiently.

To address these challenges, we propose Language-Guided Vision Token Pruning (LVPruning), a simple yet effective method that dynamically reduces the number of vision tokens in MLLMs based on their relevance to the language context. We introduce lightweight cross-attention decision modules where vision tokens attend to language tokens to compute importance scores. This relevance scoring allows the model to decide whether to keep or prune each vision token, effectively filtering out less informative visual data.

By integrating these decision modules into various layers of the MLLM, LVPruning enables progressive token pruning as the model processes deeper layers. During training, we freeze all original model parameters and only train the inserted decision modules, ensuring that the base model remains unchanged and the pruning mechanism can be easily applied or removed.

Our contributions are threefold. First, as shown in Figure 1, we demonstrate that LVPruning can significantly reduce computational costs—up to a $62.1\%$ decrease in inference TFLOPs—by pruning as much as $90\%$ of vision tokens without substantially affecting model performance. Second, we introduce a novel, language-guided token pruning mechanism that is both effective and easy to integrate into existing MLLMs, requiring minimal changes to the original architecture.

Third, our method allows for adjustable token pruning ratios during inference without retraining, offering flexibility in balancing efficiency and performance according to specific needs. Through extensive experiments on various multi-modal benchmarks, we show that LVPruning provides a practical solution to enhance the efficiency of MLLMs while maintaining their ability to understand and generate accurate multi-modal content.

# 2 Related Work

Multi-modal Large Language Models: Recent advancements in MLLMs have significantly enhanced the integration of visual and textual modalities. BLIP-2 (Li et al., 2023a) introduced a two-stage learning framework that connects pre-trained vision models with language models using a Q

former as a vision encoder, effectively generating a condensed set of vision tokens for efficient processing. Building on it, InstructBLIP (Dai et al., 2023) and MiniGPT-4 (Zhu et al., 2024) incorporated instruction tuning to improve the model's ability to follow complex prompts and perform diverse tasks. Alternatively, models such as LLaVA-1.5 (Liu et al., 2023a) directly input all vision tokens from pre-trained vision encoders into the language model. While this approach achieves higher performance owing to richer visual information, it results in substantial computational overhead. These models exemplify the trade-off between computational efficiency and performance, underscoring the need for approaches that can balance both aspects without compromising accuracy.

Efficient Transformers: Many techniques have been proposed to improve computation efficiency for Transformer models, such as knowledge distillation (Hinton et al., 2015), token merging/pruning (Bolya et al., 2023; Rao et al., 2021; Chen et al., 2023), and quantization (Gong et al., 2014; Wang et al., 2019). For NLP tasks, methods like DistillBERT (Sanh et al., 2019), MiniLM (Wang et al., 2020) use knowledge distillation to create smaller models for more efficient inference. For computer vision tasks, Liang et al. (2022); Bolya et al. (2023) and Chen et al.

(2023) focus on pruning or merging tokens based on their importance in image classification. These approaches reduce the number of vision tokens by identifying less informative patches or merging similar tokens during inference. The closest work to ours is DynamicVit (Rao et al., 2021), who use MLP layers to predict token pruning decisions. They hierarchically insert multiple pruning layers into Vision-Transformer-based models for the image classification task. However, these methods are specifically designed for single-modal targets a
nd do not address challenges in multimodal settings.

Our research distinguishes itself by focusing on token pruning for MLLMs in the context of image comprehension tasks.

# 3 Methodology

In this section, we present the LVPPruning framework in detail. As shown in Figure 2, LVPPruning is designed for MLLMs that pass all vision tokens through an MLP connector into the language model. The architecture consists of a Transformer-based pre-trained CLIP vision encoder, an MLP vision-language connector, and an LLM backbone. First,

![](dt=2026-06-04/ht=03/46d6415094fc71df101589598f6341e84cd188e6188ecea027c592b22a37ba4c.jpg)

an image input is divided into patches and processed by the CLIP model (Radford et al., 2021), such that each image patch becomes a representative vision token. Next, with the vision language connector projecting vision tokens into the dimension of the LLM's text space, the concatenated vision and text tokens are fed into the LLM for causal text generation. In specific layers of the LLM, cross-attention decision modules dynamically select the most salient token (the one with the highest attention score) to guide inference, removing redundant tokens.

During training, we apply attention masks (Rao et al., 2021) to mask out pruned vision tokens. Importantly, instead of updating positional embeddings after token pruning, we retain the original positional embeddings for the remaining vision tokens, as used in standard LLMs (Touvron et al., 2023).

# 3.1 Cross-Attention Decision Module

We now describe the detailed architecture of the cross-attention decision module designed for token pruning. A decision module, which is responsible for selecting and discarding vision tokens, comprises cross-attention layers and an MLP layer. Multiple instances of these modules are inserted into different layers of the LLM backbone for progressive pruning. Let $\mathbf{H} \in \mathbb{R}^{N \times d}$ represent the output from an LLM hidden layer, where $N$ is the sequence length and $d$ is the dimension of the hidden representations.

$\mathbf{H}$ contains the subset of vision tokens and text tokens. We define the set of the vision tokens indices as $\mathbf{I}_{\mathbf{V}} = \{n_{v_i} \mid v_i \in \mathbb{N}, 0 \leq v_i < N\}$ , and the set of text tokens indices as $\mathbf{I}_{\mathbf{T}} = \{n_{t_i} \mid t_i \in \mathbb{N}, (0 < t_i \leq N) \land (t_i \notin I_V)\}$ . We use the vision tokens as the query tokens

$$
\mathbf {Q} = W _ {q} H _ {I _ {V}} \in \mathbb {R} ^ {\left| I _ {V} \right| \times d}, \tag {1}
$$

and text tokens as the Key and Value tokens

$$
\mathbf {K}, \mathbf {V} = \left(W _ {k} / W _ {v}\right) H _ {I _ {T}} \in \mathbb {R} ^ {\left| I _ {T} \right| \times d}, \tag {2}
$$

where $W_{q}, W_{k}, W_{v}$ are linear projection layers. We then compute the attention matrix and feed the output to an FFN, as described by Vaswani et al. (2017).

$$
\mathbf {O} = \operatorname {S o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d}}\right) V + Q. \tag {3}
$$

Inspired by Rao et al. (2021), we feed the output from the FFN to a linear layer $W_{O}$ to predict the scores of keeping and removing a vision token:

$$
\gamma = W _ {O} O \in \mathbb {R} ^ {\left| I _ {V} \right| \times 2}, \tag {4}
$$

where $\gamma_{i,0}$ represents the score for keeping the vision token $\mathbf{H}_{\mathbf{I}_{\mathbf{V},\mathbf{i}}}$ and $\gamma_{i,1}$ represents the score for removing the vision token $\mathbf{H}_{\mathbf{I}_{\mathbf{V},\mathbf{i}}}$ . The decisions of vision token pruning are then generated based on $\gamma$ . The mechanism for generating and applying decisions differs between training and inference. Multiple decision modules are inserted into different layers of the LLM, such that the vision tokens are pruned progressively throughout the LLM.

# 3.2 End-to-End Training

To ensure that the process of generating and applying token pruning decisions based on $\gamma$ is differentiable, where $\gamma$ denoted as the output from each cross-attention decision module, we draw inspiration from Rao et al. (2021). Specifically, we

apply the Gumbel-Softmax distribution to $\gamma$ , redistributing it into one-hot vectors $D^{GS}$ . The first dimension of each vector is then used as the decision $D$ , determining whether to retain a given token:

$$
\begin{array}{l} D ^ {G S} = \text {G u m b e l - S o f t m a x} (\gamma) \in \{0, 1 \} ^ {\left| I _ {V} \right| \times 2}, (5) \\ \boldsymbol {D} = D _ {:, 0} ^ {G S} \in \{0, 1 \} ^ {| I _ {V} | \times 1}, (6) \\ \end{array}
$$

where $D_{i} = 1$ means we keep the vision token $\mathbf{H}_{\mathbf{I}_{\mathbf{V},\mathbf{i}}}$ and vice versa. The number of kept tokens given by the decision is not fixed during training, and directly removing unwanted vision tokens will impede batch processing. To solve this concern, we make attention masks $M$ based on $D$ for both vision and language tokens. Specifically,

$$
\begin{array}{l} M _ {i, j} = \left\{ \begin{array}{l l} 1 & \text {i f} i = j \text {o r} j \in I _ {T} \\ D _ {j} & \text {i f} i \neq j \text {a n d} j \in I _ {V} \end{array} \right., \\ 1 \leq i, j \leq N. \tag {7} \\ \end{array}
$$

Therefore, $M$ is constructed such that all language tokens are assigned a mask value of 1, while the vision tokens are masked according to the token pruning decisions $D$ . Additionally, all diagonal elements of $M$ are set to 1 to improve numerical stability.

However, $M$ disregards the original causal and padding attention masks. To address this, we first apply the original attention mask $\bar{M}$ to the raw attention scores to obtain causal attention matrix $\bar{A}$ . Then, we apply $M$ with the Softmax operation to $\bar{A}$ to get the final attention matrix $\hat{A}$ . Specifically, we define $M_l$ as the attention mask generated from the decision module at layer $l$ . The attention scores at layer $l + x$ is calculated by

$$
\hat {\boldsymbol {A}} _ {l + \boldsymbol {x}} = \operatorname {S o f t m a x} (\bar {A} _ {l + x}, M _ {l}), \tag {8}
$$

$$
\begin{array}{l} \mathit {S o f t m a x} (A, M) = \frac {\exp (A _ {i , j}) M _ {i , j}}{\sum_ {k = 1} ^ {N} \exp (A _ {i , k}) M _ {i , k}}, \\ 1 \leq i, j \leq N, \tag {9} \\ \end{array}
$$

where $(l + x)\in \mathbb{N} < l'$ . $l^{\prime}$ is the position layer of the next decision module. If $M_{i,j}^{L} = 0$ , the attention score for token $H_{j}$ will be 0 in the final attention matrix, resulting in $H_{j}$ won't contribute to any other tokens. In addition, we define $D_{l}, D_{l'}$ as the token pruning decisions get from layer $l, l'$ , respectively and $l' > l$ . We update $D_{l'}$ by

$$
\boldsymbol {D} _ {l ^ {\prime}} \leftarrow D _ {l} \odot D _ {l ^ {\prime}}, \tag {10}
$$

where $\odot$ is element-wise production, which means a previously removed vision token will never be used again.

In conclusion, Equations 7 - 9 remove the effects of unwanted vision tokens on other tokens while keeping the number of total tokens unchanged. By Equations 5,6,8,9, the process of generating and applying token pruning decisions is fully differentiable. These two factors achieve the end-to-end training capability of LVPruning.

Training Objectives: The training objectives of LVPruning are designed to teach the decision modules to remove vision tokens to predetermined ratios at different layers while fine-tuning the MLLMs to maintain their vision instruction-following capability despite the token pruning. The primary training objective is causal language modeling for instruction tuning, as described by Vaswani et al. (2017); Liu et al. (2023b); Touvron et al. (2023). Since causal language modeling is a widely used loss function, we do not detail its formal definition in this paper, referring to it as $\mathcal{L}_{\text {causal }}$ .

Additionally, to ensure that the ratio of retained vision tokens aligns with predefined values at each de
cision module, we insert $S$ decision modules into the LLM at specific layer indices $L_{idx} = [l_1, \dots, l_S]$ , with target token retention ratios $\mathbf{P} = [\rho_1, \dots, \rho_S]$ . To enforce this, we apply Mean Squared Error (MSE) loss $\mathcal{L}_{\mathrm{ratio}}$ to constrain the token pruning decisions:

$$
\mathcal {L} _ {\text {r a t i o}} = \frac {1}{S} \sum_ {s = 1} ^ {S} \left(\rho_ {s} - \frac {1}{| I _ {V} |} \sum_ {i = 1} ^ {| I _ {V} |} D _ {l _ {s}, i}\right) ^ {2}, \tag {11}
$$

where $\delta(D_{l_s}, \rho_s)$ is the Huber loss, $\beta$ is a threshold that determines the loss function used. We set $\beta = 0.5$ in all our experiments. The final training objective is the weighted sum of $\mathcal{L}_{\text{causal}}$ and $\mathcal{L}_{\text{ratio}}$ :

$$
\mathcal {L} = \lambda_ {\text {c a u s a l}} \mathcal {L} _ {\text {c a u s a l}} + \lambda_ {\text {r a t i o}} \mathcal {L} _ {\text {r a t i o}}. \tag {12}
$$

# 3.3 Inference

During the training phase, attention masks are employed to exclude the impact of irrelevant vision tokens. However, during inference, it is necessary to remove these tokens to reduce computational expenses, which introduces significant practical difficulties. First, the quantity of retained vision tokens, as determined by $D$ , is variable, thereby complicating the process of batch inference. Second,

contemporary LLMs generally utilize positional embeddings for tokens at each layer. It is essential to maintain the original positional embeddings for the retained tokens to ensure alignment with the distribution seen during training.

To overcome the first issue, we define a set of token kept ratios $\hat{\mathbf{P}} = [\hat{\rho_1},\dots,\hat{\rho_S} ]$ during inference. Note that the inference ratios do not have to be the same as the training ratios. At the $s$ -th token pruning layer, we first sort the decision scores

$$
\mathbf {Q} ^ {s} = \operatorname {a r g s o r t} \left(D _ {l s}\right). \tag {13}
$$

Then keep the top $\pmb{k}_s = \rho_s \times |I_V|$ vision tokens with the highest scores. The kept vision token indices among all vision tokens are $\hat{I}^s = \{Q_{1:k_s}^s\}$ , and the kept vision token indices among all tokens are $\hat{I}_v^s = I_{v,\hat{I}^s}$ . We define $\mathbf{PE}^1$ as the positional embeddings at the first layer. To ensure that the positional embeddings for both vision and text tokens remain unchanged after each pruning, the positional embeddings at the $s$ -th token pruning layer are

$$
\boldsymbol {P} \boldsymbol {E} ^ {\mathbf {s}} = \left[ P E _ {\hat {I} _ {v} ^ {s}}, P E _ {I _ {T}} ^ {1} \right]. \tag {14}
$$

# 4 Experimental Setup

The objective of our experiments is to investigate the feasibility of employing token pruning techniques to enhance the efficiency of MLLMs. Specifically, we ask: (i) Does LVPruning effectively reduce the computational costs while keeping the performance unchanged and to what extent can vision tokens be pruned? (ii) Compared with state-of-the-art MLLMs, does LVPruning achieve a balance between computational costs and model performance? To answer question (i) we compare LVPruning with the base MLLM on various benchmarks using different vision token kept ratios. To answer ii, we compare the relationship between inference TFLOPs and model performance for LVPruning and various state-of-the-art MLLMs.

# 4.1 Implementation Details

In all our experiments, we apply LVPPruning to LLaVA-1.5-7B (Liu et al., 2023a) (hereafter referred as LVPPruning) by inserting $S = 3$ decision modules with token kept ratio $\mathbf{P} = [\rho, \rho - 0.2, \rho - 0.4]$ , where $\rho = 0.5$ for training. These modules are inserted after the 1st, 8th, and 16th layers of LLaMA LLM. In each token pruning layer, we utilize 2 sequential cross-attention blocks, each with 8 attention heads. The Feed-Forward Networks

(FFNs) in these cross-attention blocks follow the architecture: [LayerNorm, Linear(C, 2C), SiLu activation, Linear(2C, C), LayerNorm]. We follow most of LLaVA-1.5's training settings, freezing all parameters from the LLaVA and only training inserted modules. The learning rate is set to 2e-6, with a 0.03 warm-up ratio and a cosine learning rate scheduler. The batch size is 64, and no weight decay is applied. Additionally, a maximum gradient norm of 1.0 is used to stabilize convergence. Training runs on 8 A100 (80G) GPUs. During inference, we evaluate using three different token kept ratios $(\rho = 0.6, \rho = 0.5,$ and $\rho = 0.45)$ without tuning any model parameters. All inference TFLOPs reported in this paper are computed using a dummy input consisting of 1 image and 30 text tokens.

# 4.2 Dataset and Benchmarks

To prove LVPruning's data efficiency, we use a subset of the training data for LLaVA-1.5. The LLaVA-1.5 Vision Instruction Tuning dataset (Liu et al., 2023a) consists of 665k data samples. We remove all entries without image inputs, which results in approximately 620k training samples, and the model is trained for 1 epoch. To evaluate the performance and computational efficiency of LVPruning, we calculate its inference FLOPs and assess it on nine multi-modal benchmark datasets. These include VQAv2 (Goyal et al., 2017), GQA (Hudson and Manning, 2019), VizWiz (Gurari et al.

, 2018), SciQA-IMG (Lu et al., 2022), and TextQA (Singh et al., 2019), with top1 accuracy (acc@1) used as the evaluation metric for all these benchmarks. Additionally, POPE (Li et al., 2023b) is assessed using the F1 score across three splits. MMBench (Liu et al., 2023c) is evaluated through multiple-choice questions on both English and Chinese-translated versions. LLaVA-Wild (Liu et al., 2023b) and MMVet (Yu et al., 2024) assess model responses with the assistance of GPT-4. The details about each benchmark can be found in Appendix A.

# 5 Experimental Results

In this section, we analyze the experimental results of LVPruning across nine multi-modal benchmarks. Section 5.1 examines the performance and inference TFLOPs of LVPruning compared to the base MLLM, LLaVA-1.5 (Liu et al., 2023a), highlighting its effectiveness in reducing computational cost. Section 5.2 compares LVPruning with state

![](dt=2026-06-04/ht=03/6a69d8d5123b41dac28e76fce429ba13bafb1d0c7e3e91301b0eb66d2b9bdbf4.jpg)

<table><tr><td>Method</td><td>TFLOPs</td><td>VQAv2</td><td>GQA</td><td>Vizwiz</td><td>SQA-IMG</td><td>TextVQA</td></tr><tr><td>LLaVA-1.5-7B</td><td>8.38</td><td>78.5</td><td>62.0</td><td>50.0</td><td>69.4*</td><td>58.2</td></tr><tr><td>LVPruning (ρ = 0.6)</td><td>3.97-52.6%</td><td>78.1-0.4</td><td>61.5-0.5</td><td>51.2+1.2</td><td>69.0-0.4</td><td>58.2+0</td></tr><tr><td>LVPruning (ρ = 0.5)</td><td>3.18-62.1%</td><td>77.3-1.2</td><td>60.7-1.3</td><td>51.3+1.3</td><td>68.7-0.7</td><td>58.0-0.2</td></tr><tr><td>LVPruning (ρ = 0.45)</td><td>2.79-66.7%</td><td>75.7-2.8</td><td>59.3-2.7</td><td>50.8+0.8</td><td>68.6-0.8</td><td>57.5-0.7</td></tr></table>

Table 1: Performance and inference TFLOPs comparison between LLaVA-1.5-7B (Liu et al., 2023a) and LVPruning on Visual Question Answering (VQA) benchmarks. LVPruning can significantly reduce the inference cost while maintaining marginal performance loss.

Table 2: Performance comparison between LLaVA-1.5-7B (Liu et al., 2023a) and LVPruning on visual instruction following benchmarks. LVPruning achieves competitive results, with improvements observed on three benchmarks.

![](dt=2026-06-04/ht=03/b9b1d8dc0f17f5b6139f68082e32197e5948829441a4aaee5741abc1c3ef9101.jpg)

<table><tr><td rowspan="2">Method</td><td rowspan="2">rad</td><td rowspan="2">POPE pop</td><td rowspan="2">adv</td><td colspan="2">MMBench</td><td rowspan="2">LLaVA -Wild</td><td rowspan="2">MM -Vet</td></tr><tr><td>en</td><td>cn</td></tr><tr><td>LLaVA-1.5-7B</td><td>87.3</td><td>86.1</td><td>84.2</td><td>64.3</td>
<td>58.3</td><td>65.4</td><td>31.1</td></tr><tr><td>LVPruning (ρ = 0.6)</td><td>88.1+0.8</td><td>86.5+0.4</td><td>84.4+0.2</td><td>64.0-0.3</td><td>57.4-0.9</td><td>67.9+2.5</td><td>33.3+2.2</td></tr><tr><td>LVPruning (ρ = 0.5)</td><td>87.6+0.3</td><td>86.2+0.1</td><td>84.1-0.1</td><td>63.9-0.4</td><td>56.3-2.0</td><td>65.5+0.1</td><td>31.6+0.5</td></tr><tr><td>LVPruning (ρ = 0.45)</td><td>87.4+0.1</td><td>86.1+0</td><td>84.0-0.2</td><td>63.4-0.9</td><td>56.3-2.0</td><td>60.7-4.7</td><td>30.8-0.3</td></tr></table>

![](dt=2026-06-04/ht=03/225e0c3d086464b03e0a83f7ae823b41c775b0ee3455a86599b5e29ea43cf7a1.jpg)

of-the-art Q-former-based MLLMs, demonstrating its ability to balance performance and efficiency.

# 5.1 Performance Preservation and Computation Cost Reduction

To address research question $(i)$ , we compare the model performance and inference TFLOPs of LVPruning at various vision token kept ratios,

![](dt=2026-06-04/ht=03/261d2ac03cfe6bf2ab5b79f2796cdd4f2c068ebf8447b4702cb3b965aa38e482.jpg)

evaluating both computational savings and performance trade-off. Generally speaking, Table 1 and Table 2 show that LVPruning significantly reduces the inference cost while maintaining competitive performance. As shown in Table 1, on Visual Question Answering (VQA) benchmarks, with a pruning ratio of $\rho = 0.6$ , LVPruning achieves a $52.6\%$ reduction in TFLOPs, dropping from 8.38 to 3.97 TFLOPs, with only a minor performance degrad

![](dt=2026-06-04/ht=03/cf7d9876c3ca5f692e7c1af491e9b19dafd48cf615c944702d157a6d826f1583.jpg)

<table><tr><td>Method</td><td>TFLOPs</td><td>VQAv2</td><td>GQA</td><td>Vizwiz</td><td>SQA-IMG</td><td>TextVQA</td></tr><tr><td>BLIP2-14B</td><td>2.14</td><td>65.0</td><td>41.0</td><td>19.6</td><td>61</td><td>42.5</td></tr><tr><td>InstructBLIP-8B</td><td>1.36</td><td>-</td><td>49.2</td><td>34.5</td><td>60.5</td><td>50.1</td></tr><tr><td>InstructBLIP-14B</td><td>2.14</td><td>-</td><td>49.5</td><td>33.4</td><td>63.1</td><td>50.7</td></tr><tr><td>IDEFICS-9B</td><td>0.87</td><td>50.9</td><td>38.4</td><td>35.5</td><td>-</td><td>25.9</td></tr><tr><td>Qwen-VL</td><td>1.75</td><td>78.8</td><td>59.3</td><td>35.2</td><td>67.1</td><td>63.8</td></tr><tr><td>Qwen-VL-Chat</td><td>1.75</td><td>78.2</td><td>57.5</td><td>38.9</td><td>68.2</td><td>61.5</td></tr><tr><td>LVPruning (ρ = 0.5)</td><td>3.18</td><td>77.3</td><td>60.7</td><td>51.34</td><td>68.7</td><td>58.0</td></tr></table>

Table 3: Performance comparison between LVPPruning with a token retention ratio of $\rho = 0.5$ and state-of-the-art Q-former-based models on Visual Question Answering (VQA) benchmarks. It shows that LVPPruning ( $\rho = 0.5$ ) outperforms the state-of-the-art MLLMs on three benchmarks, while delivering competitive results on the others.

Table 4: Performance comparison between LVPPruning with a token retention ratio of $\rho = 0.5$ and state-of-the-art Q-former-based models on visual instruction following benchmarks. It shows that LVPPruning outperforms nearly all benchmarks, with only a slight performance drop compared to BLIP2-14B (Li et al., 2023a) in POPE(rad)(Li et al., 2023b).

![](dt=2026-06-04/ht=03/857610ba6114d45dbacdb46d0e6574c947d1cd5420950abfcecb426a25b02710.jpg)

<table><tr><td>Method</td><td>rad</td><td>POPE pop</td><td>adv</td><td>MMBench en</td><td>cn</td><td>LLaVA -Wild</td><td>MM -Vet</td></tr><tr><td>BLIP2-14B</td><td>89.6</td><td>85.5</td><td>80.9</td><td>-</td><td>-</td><td>38.1</td><td>22.4</td></tr><tr><td>InstructBLIP-8B</td><td>-</td><td>-</td><td>-</td><td>36.0</td><td>23.7</td><td>60.9</td><td>26.2</td></tr><tr><td>InstructBLIP-14B</td><td>87.7</td><td>77</td><td>72</td><td>-</td><td>-</td><td>58.2</td><td>25.6</td></tr><tr><td>IDEFICS-9B</td><td>-</td><td>-</td><td>-</td><td>48.2</td><td>25.2</td><td>-</td><td>-</td></tr><tr><td>Qwen-VL</td><td>-</td><td>-</td><td>-</td><td>38.2</td><td>7.4</td><td>-</td><td>-</td></tr><tr><td>Qwen-VL-Chat</td><td>-</td><td>-</td><td>-</td><td>60.6</td><td>56.7</td><td>-</td><td>-</td></tr><tr><td>LVPruning (ρ = 0.5)</td><td>87.6</td><td>86.2</td><td>84.1</td><td>63.9</td><td>56.3</td><td>65.5</td><td>31.6</td></tr></table>

tion. For instance, on VQAv2, GQA, and VizWiz benchmarks, LVPruning maintains similar accuracy, with minimal decreases of 0.4, 0.5, and even an increase of 1.2 points, respectively. Even with higher pruning ratios, such as $\rho = 0.45$ (66.7% TFLOPs reduction), the model's performances on certain benchmarks like VizWiz (+0.8) and SQA-IMG (-0.8) are still comparable to LLaVA-1.5-7B. Table 2 shows the results for visual instruction following benchmarks. On the POPE and MMBench (en and cn) benchmarks, LVPruning yields similar or improved performance with performance gains of up to 0.8 points on POPE. Notably, the LLaVA-Wild and MM-Vet benchmarks show obvious performance gains with $\rho = 0.6$ which increases by 2.5 and 2.2 points, respectively.

Overall, Figure 3 visualizes the performance variance between LLaVA-1.5 (Liu et al., 2023a) and LVPruning with different vision token kept ratios $\rho$ on nine multi-modal benchmarks, and Figure 4 demonstrates their inference FLOPs. LVPruning demonstrates that significant computational

savings can be achieved with minimal impact on performance. Even at a high pruning ratio, such as $\rho = 0.45$ , where TFLOPs are reduced by $66.7\%$ , the performance degradation remains relatively small, indicating that LVPruning effectively balances model efficiency with task performance. More details about the relationship between token kept ratios and inference FLOPs are shown in Appendix B.

# 5.2 Comparisons with state-of-the-art MLLMs

To address research question (ii), we use LVPruning with $\rho = 0.5$ as a representative configuration to compare against state-of-the-art Q-former-based MLLMs. Generally speaking, Tables 3 and 4 demonstrate that LVPruning achieves superior performance with competitive inference FLOPs.

As shown in Table 3, on VQA benchmarks, LVPruning with $\rho = 0.5$ outperforms several state-of-the-art models, such as BLIP2-14B (Li et al., 2023a), InstructBLIP-14B (Dai et al., 2023),

![](dt=2026-06-04/ht=03/53081a19d136116ca6d186ca4e2393a71cd7fef0e8175e9440f1ff7d2dc6fcd3.jpg)

and IDEFICS-9B (Laurençon et al., 2023). For example, LVPruning achieves a higher VQAv2 accuracy (77.3) compared to BLIP2-14B (65.0) and IDEFICS-9B (50.9), while utilizing only 3.18 TFLOPs, which is relatively higher than IDEFICS but remains efficient compared to other models like BLIP2 and Qwen-VL (Bai et al., 2023). In tasks such as VizWiz, LVPruning also achieves a notable boost in performance (51.34), surpassing BLIP2 and InstructBLIP models by a large margin. Table 4 shows that on visual instruction following benchmarks, LVPruning consistently delivers competitive results.

For instance, LVPruning achieves 87.6 accuracy on the POPE (rad) benchmark, closely trailing BLIP2-14B's 89.6. However, it outperforms BLIP2 on multiple datasets, such as MMBench (en) and LLaVA-Wild, with scores of 63.9 and 65.5, respectively. These results illustrate that LVPruning maintains competitive performance across various instruction following benchmarks, balancing between efficiency and effectiveness. Figure 5 further illustrates the relationship between inference TFLOPs and the performance of LVPruning and state-of-the-art MLLMs on the GQA benchmark. LVPruning, represented by red

points, showcases a balanced trade-off between computational efficiency and performance.

# 6 Conclusion

In this work, we introduce LVPruning, a novel language-guided vision token pruning meth
od that can be integrated into existing MLLMs with minimal architectural changes. LVPruning computes relevance scores for each vision token based on language tokens, progressively removing redundant tokens throughout the LLM. By the middle layer, it eliminates up to $90\%$ of vision tokens, achieving a $62.1\%$ reduction in FLOPs with only a $\sim 0.45\%$ average performance loss across nine multi-modal benchmarks. This makes LVPruning a practical solution for enhancing MLLM efficiency while preserving performance in multi-modal tasks.

# 7 Limitations

While LVPruning shows significant promise in reducing computational load, several limitations to our study should be acknowledged. Our evaluation has been conducted on a specific set of benchmarks. The performance of LVPruning on other datasets or in real-world applications remains unexplored. Therefore, although LVPruning is effective in reducing computational overhead, it is crucial to consider the specific requirements of different tasks when applying this method to ensure that essential visual information is not compromised. Future research could involve evaluating the performance of LVPruning with human feedback to better understand its practical implications.

# References

Scialom. 2023. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288.

Anne Treisman. 1988. Features and objects: The fourteenth bartlett memorial lecture. The Quarterly Journal of Experimental Psychology Section A, 40(2):201-237. PMID: 3406448.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998-6008.

Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, and Song Han. 2019. HAQ: hardware-aware automated quantization with mixed precision. In CVPR, pages 8612-8620. Computer Vision Foundation / IEEE.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In NeurIPS.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2024. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML. Open-Review.net.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2024. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In ICLR. OpenReview.net.

# A Benchmark Datasets Details

VQAv2 (Goyal et al., 2017) includes approximately 11k test samples and focuses on visual question answering, where models must answer questions based on images depicting various real-world scenes. GQA (Hudson and Manning, 2019), with around 12k test samples, emphasizes compositional reasoning through graph-structured annotations, assessing a model's ability to understand object relationships. VisWiz (Gurari et al.

, 2018), containing 8k test samples, presents accessibility challenges with real-world images from visually impaired users, which are often of low quality and ambiguous, demanding robust model interpretation. SciQA-IMG (Lu et al., 2022) consists of around 4k test samples, targeting science-related visual question answering in specific domains. TextVQA (Singh et al., 2019), with 5k test samples, focuses on understanding and answering questions from textual images. POPE (Li et al., 2023b) contains approximately 9k test samples on three subsets: random, common, and adversarial. It evaluates a

![](dt=2026-06-04/ht=03/c139792918daed0ab2d593ad82ac524e701518550ca8b2609d64e9c0309483b2.jpg)

model's ability to predict human preference judgments on hallucination of multimodal tasks. MM-Bench (en) (Liu et al., 2023c), with around 4k test samples, serves as a comprehensive benchmark for evaluating general-purpose multimodal models across various tasks, while MM-Bench (cn) (Liu et al., 2023c) is its Chinese translation. LLaVA-Wild (Liu et al., 2023b) has 60 test samples and emphasizes answering questions about complex, in-the-wild images. Finally, MM-Vet (Yu et al., 2024) includes 218 samples and is designed to test multimodal capabilities across multiple visual and language tasks, providing a robust evaluation framework for emerging multimodal systems.

# B Detailed TFLOPs of LVPruning with Different Vision Token Kept Ratio

Figure 6 shows the detailed TFLOPs of LVPruning with 0.05 change step of $\rho$ . Additionally, compared with LLaVA-1.5-7B (Liu et al., 2023a), the extra computation cost introduced by the inserted decision modules is 0.71 TFLOPs.