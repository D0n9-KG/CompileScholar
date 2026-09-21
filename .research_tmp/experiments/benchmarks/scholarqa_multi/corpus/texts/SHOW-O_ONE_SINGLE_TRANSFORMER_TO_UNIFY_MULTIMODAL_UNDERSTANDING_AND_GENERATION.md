# SHOW-O: ONE SINGLE TRANSFORMER TO UNIFY MULTIMODAL UNDERSTANDING AND GENERATION

Jinheng Xie $^{1\dagger}$ Weijia Mao $^{1\dagger}$ Zechen Bai $^{1\dagger}$ David Junhao Zhang $^{1\dagger}$ Weihao Wang $^{2}$ Kevin Qinghong Lin $^{1}$ Yuchao Gu $^{1}$ Zhijie Chen $^{2}$ Zhenheng Yang $^{2}$ Mike Zheng Shou $^{1*}$

$^{1}$ Show Lab, National University of Singapore $^{2}$ ByteDance

# ABSTRACT

We present a unified transformer, i.e., Show-o, that unifies multimodal understanding and generation. Unlike fully autoregressive models, Show-o unifies autoregressive and (discrete) diffusion modeling to adaptively handle inputs and outputs of various and mixed modalities. The unified model flexibly supports a wide range of vision-language tasks including visual question-answering, text-to-image generation, text-guided inpainting/extrapolation, and mixed-modality generation. Across various benchmarks, it demonstrates comparable or superior performance to existing individual models with an equivalent or larger number of parameters tailored for understanding or generation. This significantly highlights its potential as a next-generation foundation model. Code and models are released at https://github.com/showlab/Show-o.

# 1 INTRODUCTION

“Alone we can do so little; together we can do so much.” – Helen Keller

Over the past few years, significant advancements have blossomed in the two key pillars of multimodal intelligence: understanding and generation (Fig. 1(a) and (b)). For multimodal understanding, Multimodal Large Language Models (MLLMs) like LLaVA (Liu et al., 2024c) have demonstrated exceptional capabilities in vision-language tasks such as visual question-answering (VQA). For the other pillar of visual generation, denoising diffusion probabilistic models (DDPMs) (Sohl-Dickstein et al., 2015; Ho et al., 2020b) have revolutionized the traditional generative paradigms (Kingma & Welling, 2013; Goodfellow et al., 2014), achieving unprecedented performance in text-to-image/video generation (Podell et al., 2023; Esser et al., 2024; Ho et al., 2022; Wu et al., 2023a).

Given these achievements in individual fields, it is natural to explore the potential of connecting them. Recent works (Wu et al., 2023b; Ge et al., 2024; Ye et al., 2024a; Dong et al., 2024) have tried to assemble expert models from different domains to form a unified system that can handle both multimodal understanding and generation. However, existing attempts mainly treat each domain independently and often involve individual models responsible for understanding and generation separately (as shown on the left of Fig. 1(c)). For instance, NExT-GPT (Wu et al., 2023b) employs a base language model for multimodal understanding but requires an additional pre-trained diffusion model for image generation. Nonetheless, the mainstream understanding models like LLaVA are of transformer architecture (Vaswani et al., 2017b) while each leading generation models like Stable Diffusion 3 (SD3) (Esser et al., 2024) are just another transformer. This motivates a research question: can one single transformer handle both multimodal understanding and generation?

Very recently, Chameleon (Team, 2024) has demonstrated this is possible. Specifically, Chameleon enables an early fusion of different modalities to generate both text and image tokens through the same manner of autoregressive modeling. While it is reasonable to model text tokens autoregressively (Touvron et al., 2023; Liu et al., 2024c), it is less clear whether it is better to model image/video patches (or pixels) autoregressively as well. An apparent and significant bottleneck of autoregressively predicting an image is the large number of sampling steps required due to its causal attention, particularly when dealing with images/videos in higher resolution. Further, (continuous)

![](images/f8ccde6371c629d043af314cdf15c4ad104f3aacd687b28b432cd092c5069784.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Vision"] --> C["LLM"]
    B["Language"] --> C["LLM"]
    C["LLM"] --> D["Language"]
```
</details>

![](images/1aaa0d48fc46c1d66796bd48b67370d220250d9ba6b56830e8b6b6d581980cb0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Language"] --> B["Diffusion"]
    B --> C["Vision"]
```
</details>

![](images/432cae0ca5c5c4f3054bc45ad4c9a3792fcf276f506fa362784b1fc7588b7e40.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Language"] --> B["LLM (AR)"]
    B --> C["Vision"]
```
</details>

![](images/49f16af738d1ece366a3402cd06bc2dfcc935bed7955ee183c0daf0c21862e4f.jpg)

![](images/6b8290996fb7f035c754232fe64355e2c7b0e48b0176646741b05490748abb3b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Language"] --> C["LLM"]
    B["Vision"] --> C["LLM"]
    C["LLM"] --> D["Language"]
    E["Diffusion"] --> F["Vision"]
    C["LLM"] --> G["Language"]
    E["Diffusion"] --> H["Language"]
```
</details>

![](images/a28154910c32e91b81a7a52ae9e4a93ae0a6328b79e4d45f6f269bc1dd5e79f1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Language"] --> C["LLM (AR)"]
    B["Vision"] --> C["LLM (AR)"]
    C --> D["Language"]
    C --> E["Vision"]
```
</details>

![](images/6394540ef814557f6ee1bb090dd2f7a85ae44ac8813804882f13140d24317fb5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Language"] --> C["LLM (AR + Diffusion)"]
    B["Vision"] --> C
    C --> D["Language"]
    C --> E["Vision"]
```
</details>

Figure 1: Characteristics comparison among understanding only, generation only, and unified (understanding & generation) models. “Vision” and “Language” indicate the representations from specific input modalities. In this context, “Diffusion” represents both continuous and discrete diffusion.

diffusion models (Podell et al., 2023; Esser et al., 2024) have exhibited superior capabilities in visual generation than autoregressive ones and are in full attention.

This motivates us to ponder: can such one single transformer involve both autoregressive and diffusion modeling? Here we envision a new paradigm that text is represented as discrete tokens and modeled autoregressively, same with large language models (LLMs), and continuous image pixels are modeled using denoising diffusion. However, it is non-trivial to integrate these two distinct techniques into one single network due to the significant differences between discrete text tokens and continuous image/video representations. Another challenge lies in the fact that existing state-of-the-art diffusion models typically rely on two distinct models, i.e., a text encoder to encode text conditional information and a denoising network to predict noise.

To this end, we present a novel unified model, i.e., Show-o, capable of addressing both multimodal understanding and generation tasks simultaneously with mixed autoregressive and diffusion modeling (as shown in Fig. 2). Specifically, Show-o is built upon a pre-trained LLM and inherits the autoregressive modeling capability for text-based reasoning. Inspired by Gu et al. (2022); Chang et al. (2022), we employ a simplified discrete denoising diffusion, similar to MaskGIT (Chang et al., 2022), to model discrete image tokens instead of continuous diffusion used in existing works (Ge et al., 2024; Dong et al., 2024). Besides, Show-o inherently encodes text conditional information, eliminating additional text encoders. To accommodate diverse input data and variations of tasks, a text tokenizer and image tokenizer are employed to encode them into discrete tokens, and a unified prompting strategy is proposed further to process these tokens into structure sequences as input. Consequently, given an image accompanying questions, Show-o gives the answers autoregressively. When provided only text tokens, Show-o generates images in a style of discrete denoising diffusion.

Quantitatively, Show-o demonstrates comparable even better performance to individual models with an equivalent or larger number of parameters across benchmarks. In contrast to autoregressively generating an image, Show-o requires approximately 20 times fewer sampling steps, exhibiting inherent potential in acceleration. Besides, as shown in Fig. 2, Show-o naturally supports various downstream applications like text-guided inpainting and extrapolation, without any fine-tuning. Moreover, we have demonstrated that Show-o has the potential for mixed-modality generation like interleaved video keyframe generation with text descriptions, video understanding, and video generation. This demonstrates the potential of the unified model as a feasible paradigm for long-form video understanding and generation. Beyond, we investigate the impact of dataset scale, image resolution, and different types of image representations (discrete or continuous) on the multimodal understanding performance, presenting systematic insights for the design of a unified model in the future.

In Fig. 1, we present a comparison of model characteristics between Show-o and existing representative methods across various domains. One can observe that Show-o is a unified model that flexibly involves existing advanced techniques to comprehensively address multimodal understanding and generation. Collectively, the main contributions of this paper can be summarized as:

- We present a unified model, i.e., Show-o, which unifies multimodal understanding and generation using one single transformer.   
- Show-o innovatively unifies autoregressive and (discrete) diffusion modeling within one single transformer, demonstrating versatility in handling both text and images distinctly.   
- As a unified model, Show-o demonstrates comparable even better performance to individual baseline models with an equivalent or larger number of parameters in multimodal understanding and generation benchmarks.   
- Show-o inherently supports various downstream applications like text-based inpainting and extrapolation, without necessitating any fine-tuning. Besides, it also demonstrates the potential for mixed-modality generation, video understanding, and video generation.   
- We explore the impact of dataset scale, image resolution, and different types of representations (discrete or continuous) on multimodal understanding, providing valuable insights for improving multimodal understanding capabilities of a unified model.

# 2 RELATED WORK

# 2.1 MULTIMODAL UNDERSTANDING

Significant advancements in large language models (LLMs) (Touvron et al., 2023; Brown et al., 2020; Chowdhery et al., 2023) have inspired the development of multimodal large language models (MLLMs) (Li et al., 2024; Yin et al., 2023; Bai et al., 2024). Early MLLM efforts, such as LLaVA (Liu et al., 2024c), MiniGPT-4 (Zhu et al., 2023a), and InstructBLIP (Dai et al., 2023), demonstrate notable multimodal understanding capabilities. To integrate LLMs into multimodal domains, these studies explored projecting features from a pre-trained modal-specific encoder, such as CLIP (Radford et al., 2021), into the input space of LLMs, enabling multimodal understanding and reasoning within the transformer backbone. There are various design choices of MLLM (McKinzie et al., 2024; Tong et al., 2024) in vision encoders, feature alignment adapters, and datasets.

# 2.2 VISUAL GENERATION

Autoregressive models. Transformer models (Vaswani et al., 2017a; Raffel et al., 2020; Brown et al., 2020; Touvron et al., 2023) have demonstrated great success of autoregressive modeling in natural language processing. Inspired by such progress, previous studies (Parmar et al., 2018; Esser et al., 2021; Ravuri & Vinyals, 2019; Chen et al., 2020; Kondratyuk et al., 2023) directly apply the same autoregressive modeling to learn the dependency of image pixels for image/video generation. For instance, VideoPoet (Kondratyuk et al., 2023) also employs the decoder-only transformer architecture for synthesizing high-quality videos from multimodal inputs. More recently, LlamaGen (Sun et al., 2024) has demonstrated LLM-architecture based image token autoregression.

Diffusion models. In recent years, diffusion-based methods (Rombach et al., 2022; Ramesh et al., 2022b;a; Peebles & Xie, 2023; Bao et al., 2023; Podell et al., 2023; Chen et al., 2024; Nichol et al., 2021; Xue et al., 2024; Xie et al., 2023; Wu et al., 2023a) have demonstrated exceptional capabilities in text-to-image/video generation. Typically, the denoising diffusion process is operated on the continuous latent space, in which the model is tasked with predicting the added Gaussian noise. In contrast, D3PM (Austin et al., 2021), Mask-predict (Ghazvininejad et al., 2019), ARDM (Hoogeboom et al., 2022), MaskGIT (Chang et al., 2022), UniD3 (Hu et al., 2023), and Copilot4D (Zhang et al., 2024) formulate a discrete corruption process as an alternative to Gaussian diffusion.

# 2.3 UNIFIED VISION-LANGUAGE FOUNDATION MODEL

In recent years, an increasing number of studies (Wu et al., 2023b; Tang et al., 2024; Ye et al., 2024a; Aiello et al., 2024; Lu et al., 2024) have focused on unified multimodal language models capable of both comprehension and generation. Some efforts (Zhu et al., 2023b; Sun et al., 2023b;a) use continuous representations interleaved with text tokens for autoregressive modeling to generate images. SEED-X (Ge et al., 2024) proposes a unified and versatile foundation system capable of handling both multimodal understanding and generation tasks. DreamLLM (Dong et al., 2024) also explores

![](images/3b5a366d520ab9b6225364fb74f9691b75a3db1303701db8cb5eedee014bf02a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text Tokenizer & Image Tokenizer"] --> B["Visual Generation"]
    B --> C["Mixed-modality generation"]
    C --> D["Text De-Tokenizer & Image De-Tokenizer"]

    subgraph_A1["The image features a young girl sitting on the grass, surrounded by a colourful backdrop. She is holding a ..."]
    end

    subgraph_A2["Yes, there is a rainbow in the image, as the girl is painting a rainbow on the canvas."]
    end

    subgraph_B["Text to-Image Generation / Text-guided Inpainting and Extrapolation"]
    end

    subgraph_C["Text keyframe generation with text descriptions"]
    end

    subgraph_D["Text De-Tokenizer & Image De-Tokenizer"]
    end

    A -->|Q1: Please describe this image in detail.
Q2: Is there a rainbow in this image?| A
    B -->|a dog sitting on the bench.
a vibrant hot air balloon floats over a clear lake.| B
    C -->|Slicing avocado.| C
    D -->|A woman is cutting an avocado with a knife...| D
    end

    subgraph_A3["Special task tokens for distinguishing various tasks"]
    end

    subgraph B
    end

    subgraph C
    end

    A -->|□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    end

    subgraph B
    end

    A -->|a punk rock frog in a studded leather jacket shouting into a microphone while standing on a boulder.|
    B -->|a dog sitting on the bench.
a vibrant hot air balloon floats over a clear lake.|
    C -->|a vibrant hot air balloon floats over a clear lake.|
    D -->|a woman is cutting an avocado with a knife...|
    end

    subgraph C
    end

    A -->|a woman is cutting an avocado with a knife...|
    B -->|a woman is cutting an avocado with a knife...|
    C -->|a woman is cutting an avocado with a knife...|
    D -->|a woman is cutting an avocado with a knife...|
    end

    subgraph_A4["Special task tokens for distinguishing various tasks"]
    end

    subgraph B
    end

    A -->|a punk rock frog in a studded leather jacket shouting into a microphone while standing on a boulder.|
    B -->|a dog sitting on the bench.
a vibrant hot air balloon floats over a clear lake.|
    C -->|a woman is cutting an avocado with a knife...|
    end

    subgraph C
    end

    A -->|a woman is cutting an avocado with a knife...|
    B -->|a woman is cutting an avocado with a knife...|
    C -->|a woman is cutting an avocado with a knife...|
    end

    subgraph D
    end

    A -->|a woman is cutting an avocado with a knife...|
    B -->|a woman is cutting an avocado with a knife...|
    C -->|a woman is cutting an avocado with a knife...|
    end

    subgraph_A5["Special task tokens for distinguishing various tasks"]
    end

    subgraph B
    end

    A -->|a punk rock frog in a studded leather jacket shouting into a microphone while standing on a boulder.|
    B -->|a dog sitting on the bench.
a vibrant hot air balloon floats over a clear lake.|
    C -->|a woman is cutting an avocado with a knife...|
    end

    subgraph C
    end

    A -->|A woman is cutting an avocado with a knife...|
    B -->|a woman is cutting an avocado with a knife...|
    C -->|a woman is cutting an avocado with a knife...|
    end

    subgraph D
    end

    A -->|A woman is cutting an avocado with a knife...|
    B -->|a woman is cutting an avocado with a knife...|
    C -->|a woman is cutting an avocado with a knife...|
    end

    subgraph_A6["Special task tokens for distinguishing various tasks"]
    end

    subgraph B
    end

    A -->|a punk rock frog in a studded leather jacket shouting into a microphone while standing on a boulder.|
    B -->|a dog sitting on the bench.
a vibrant hot air balloon floats over a clear lake.|
    C -->|a woman is cutting an avocado with a knife...|
    end

    subgraph C
    end

    A -->|B: cut off from image to display image of text tokenization and text descriptions|
    B -->|B: cut off from image to display image of text descriptions and text descriptions|
    C -->|B: cut off from image to display image of text descriptions and text descriptions|
```
</details>

Figure 2: An overview of Show-o. The input data, regardless of its modalities, is tokenized and then prompted into a formatted input sequence. Show-o processes text tokens autoregressively with causal attention and image tokens in (discrete) diffusion modeling via full attention, and then generates the desired output. Specifically, Show-o can handle image captioning, visual question answering, text-to-image generation, text-guided inpainting/extrapolation, and mixed modality generation. the potential of enabling multimodal comprehension and creation. Chameleon (Team, 2024) introduces token-based mixed-modal models capable of comprehending and generating images.

# 3 METHODOLOGY

Preliminaries. Instead of continuous diffusion, this work employs mask token prediction used in MaskGIT as a simplified discrete diffusion modeling to enable a more unified learning objective, i.e., predicting discrete tokens within one single transformer. We draw the connection between mask token prediction used in this work and discrete diffusion modeling in Appendix A.

# 3.1 TOKENIZATION

Show-o is built upon pre-trained LLMs (Li et al., 2023), it is natural to perform the unified learning on the discrete space. We maintain a unified vocabulary to include discrete text and image tokens.

Text Tokenization. Show-o is based on a pre-trained LLM such that we utilize the same tokenizer for text data tokenization without any modifications.

Image Tokenization. Following MAGVIT-v2 (Yu et al., 2023), we train a lookup-free quantizer using a large-scale image data. The quantizer maintains a codebook of size K = 8, 192 and encodes images of $256 \times 256$ resolution into $16 \times 16$ discrete tokens (option (a) in Fig. 3).

An alternative approach is to use different tokenizers for understanding and generation, respectively. Inspired by existing studies (Liu et al., 2024c;b), we also extract the continuous image representations from the pretrained MAGVIT-v2 and CLIP-ViT (Radford et al., 2021) encoder as input for exploring the improvement of multimodal understanding capabilities (options (b) and (c) in Fig. 3). We will present more details and discuss this exploration in Section 4.6. In the following sections, the default Show-o employs discrete image tokens as input for both multimodal understanding and generation (option (a) in

![](images/36708cf0d62481cd995891889d8fd2372e1e00111bd7559098ada4dde60b4489.jpg)

<details>
<summary>flowchart</summary>

Three-layer architecture diagram for a 3D convolutional model, showing input layers (Image, ConvNet, Embedding Layer) and output layer (Continuous) with CLIP components.
</details>

Figure 3: Optional inputs for multimodal understanding.

Fig. 3). For simplicity, we only elaborate on the default Show-o in the methodology sections.

![](images/23c8fa5bda6522f906eb7eaf2e2dec5541ae1608381bdf71a0e26d03fcb9f462.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Multi-modal Understanding"] --> B["MMU"]
    A --> C["SOI"]
    A --> D["EOI"]
    A --> E["SOT"]
    A --> F["EOT"]
    G["Visual Generation"] --> H["T2I"]
    G --> I["SOT"]
    G --> J["EOT"]
    G --> K["SOI"]
    G --> L["EOI"]
    M["Mixed-Modality Generation"] --> N["T2I"]
    M --> O["SOT"]
    M --> P["EOT"]
    M --> Q["SOI"]
    M --> R["EOI"]
    M --> S["Text tokens v"]
    M --> T["Image tokens u"]
    U["Special task tokens"] --> V["MMU"]
    U --> W["T2I"]
    U --> X["Start & end of text tokens"]
    Y["SOT"] --> Z["EOT"]
    AA["EOI"] --> AB["Text tokens v"]
    AC["SOI"] --> AD["Image tokens u"]
    AE["EOI"] --> AF["Text tokens v"]
    AG["SOI"] --> AH["Image tokens u"]
```
</details>

Figure 4: Illustration of the proposed unified prompting format.

# 3.2 ARCHITECTURE

Show-o inherits the architecture of existing LLM (Li et al., 2023) without any architecture modifications except for prepending a QK-Norm operation (Dehghani et al., 2023; Wortsman et al., 2023; Team, 2024) to each attention layer. We initialize Show-o with the weights of a pre-trained LLM and expand the size of the embedding layer by incorporating 8,192 new learnable embeddings for discrete image tokens. Unlike state-of-the-art diffusion models that require an additional text encoder, Show-o inherently encodes text conditional information by itself for text-to-image generation.

Unified Prompting. To perform unified learning on multimodal understanding and generation, we design a unified prompting strategy to format various kinds of input data. Given an image-text pair $(\mathbf{x}, \mathbf{y})$ , it is first tokenized into M image tokens $u = \{u_i\}_{i=1}^M$ and N text tokens $v = \{v_i\}_{i=1}^N$ by the image and text tokenizer, respectively. We form them into an input sequence according to the type of task in the format illustrated in Fig. 4. Specifically, [MMU] and [T2I] are pre-defined task tokens that indicate the learning task for the input sequence. [SOT] and [EOT] serve as special tokens denoting the start and end of text tokens, respectively. Similarly, [SOI] and [EOI] are pre-defined special tokens marking the start and end of image tokens.

By employing this prompt design, we can effectively encode various input data for multi-modal understanding, text-to-image generation, and mixed-modality generation as sequential data. This setup enables unified learning to operate seamlessly within sequences across these various tasks. Once trained, we can accordingly prompt Show-o to handle various vision-language tasks including visual question answering and text-to-image generation (as shown in Fig. 2).

![](images/419809472a0a95a2deb0db0f05f78142fca5cfafd12978af68f9263f6be1f842.jpg)

<details>
<summary>text_image</summary>

Image tokens u
Text tokens v
Image tokens u Text tokens v
</details>

(a) Multimodal Understanding

![](images/b7b2732669142bf0f29c2540e15e54205c43451cfbaef513da7869171f750ba6.jpg)

<details>
<summary>text_image</summary>

Text tokens v
Image tokens u
Text tokens v
Image tokens u
</details>

(b) Text-to-Image Generation

![](images/d7e31999bcd5ed264c978c8197bc353f9e579fbbe5b0e839654ea98989ce6c9d.jpg)

<details>
<summary>text_image</summary>

Text tokens v
Text tokens v
</details>

(c) Language Modeling

![](images/2a1292ad400b60d417e55ab5f6d40caf10cecec9a3bac61b0768ffb5b9111f44.jpg)

<details>
<summary>text_image</summary>

Text
Image
Text
Image
Text
Image
</details>

(d) Mixed-Modality Generation   
Figure 5: Omni-Attention Mechanism (The dark squares represent ‘allow to attend’, while the white squares indicate ‘prevent from attending’). It is a versatile attention mechanism with causal and full attention that adaptively mixes and changes according to the format of the input sequence.

Omni-Attention Mechanism. Different from existing works (Touvron et al., 2023; Team, 2024) that model sequence auto-regressively only, we propose an omni-attention mechanism to enable Show-o to model various types of signals in distinct ways. It is a comprehensive attention mechanism with causal and full attention that adaptively mixes and changes according to the format of the input sequence. We illustrate omni-attention examples for different input sequences in Fig. 5. Specifically, Show-o model text tokens v within the sequence via causal attention. For image tokens u, Show-o processes them via full attention, allowing each token to comprehensively interact with all others. Given a formatted input sequence, it is apparent that in multimodal understanding (Fig. 5(a)), text tokens in a sequence can attend to all previous image tokens, and in text-to-image generation (Fig. 5(b)), image tokens are able to interact with all preceding text tokens. When given only text tokens, it degrades to causal attention (Fig. 5(c)).

Training Objectives. To perform both auto-regressive and (discrete) diffusion modeling, we employ two learning objectives: i) Next Token Prediction (NTP) and ii) Mask Token Prediction (MTP). Given a sequence with M image tokens $u = \{u_{1}, u_{2}, \cdots, u_{M}\}$ and N text tokens

Table 1: Evaluation on multimodal understanding benchmarks. Show-o is currently built upon Phi-1.5 and thus we implement LLaVA-v1.5-Phi-1.5 as our apple-to-apple baseline. Und. and Gen. denote “understanding” and “generation”, respectively. $^{\ddagger}$ denotes the improved Show-o that employs CLIP-ViT continuous representations. We highlight the model size of Show-o and LLaVA baseline in green, and we use blue to highlight the larger model size than ours. 

<table><tr><td>Type</td><td>Model</td><td># Params</td><td>POPE↑</td><td>MME↑</td><td>Flickr30k↑</td><td>VQAv2(test)↑</td><td>GQA↑</td><td>MMMU↑</td></tr><tr><td rowspan="5">Und. Only</td><td>LLaVA-v1.5 (Liu et al., 2024b)</td><td>7B</td><td>85.9</td><td>1510.7</td><td>-</td><td>78.5</td><td>62.0</td><td>35.4</td></tr><tr><td>InstructBLIP (Dai et al., 2023)</td><td>13B</td><td>78.9</td><td>1212.8</td><td>-</td><td>-</td><td>49.5</td><td>-</td></tr><tr><td>Qwen-VL-Chat Bai et al. (2023)</td><td>7B</td><td>-</td><td>1487.5</td><td>-</td><td>78.2</td><td>57.5</td><td>-</td></tr><tr><td>mPLUG-Owl2 (Ye et al., 2024b)</td><td>7B</td><td>85.8</td><td>1450.2</td><td>-</td><td>79.4</td><td>56.1</td><td>-</td></tr><tr><td>LLaVA-v1.5-Phi-1.5</td><td>1.3B</td><td>84.1</td><td>1128.0</td><td>69.6</td><td>75.3</td><td>56.5</td><td>30.7</td></tr><tr><td rowspan="11">Und. and Gen.</td><td>Gemini-Nano-1 (Anil et al., 2023)</td><td>1.8B</td><td>-</td><td>-</td><td>-</td><td>62.7</td><td>-</td><td>26.3</td></tr><tr><td>CoDI (Tang et al., 2024)</td><td>-</td><td>-</td><td>-</td><td>12.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Emu (Sun et al., 2023c)</td><td>13B</td><td>-</td><td>-</td><td>77.4</td><td>57.2</td><td>-</td><td>-</td></tr><tr><td>NExT-GPT (Wu et al., 2023b)</td><td>13B</td><td>-</td><td>-</td><td>84.5</td><td>66.7</td><td>-</td><td>-</td></tr><tr><td>SEED-X (Ge et al., 2024)</td><td>17B</td><td>84.2</td><td>1435.7</td><td>52.3</td><td>-</td><td>47.9</td><td>35.6</td></tr><tr><td>DreamLLM (Dong et al., 2024)</td><td>7B</td><td>-</td><td>-</td><td>-</td><td>72.9</td><td>-</td><td>-</td></tr><tr><td>VILA-U (Wu et al., 2024)</td><td>7B</td><td>85.8</td><td>1401.8</td><td>-</td><td>79.4</td><td>60.8</td><td>31.6</td></tr><tr><td>Emu3 (Wang et al., 2024)</td><td>8B</td><td>85.2</td><td>-</td><td>-</td><td>75.1</td><td>60.3</td><td>-</td></tr><tr><td>Chameleon (Team, 2024)</td><td>34B</td><td>-</td><td>-</td><td>74.7</td><td>66.0</td><td>-</td><td>-</td></tr><tr><td>Show-o (Ours)</td><td>1.3B</td><td>80.0</td><td>1097.2</td><td>62.5</td><td>69.4</td><td>58.0</td><td>26.7</td></tr><tr><td>Show-o $^{\ddagger}$  (Ours)</td><td>1.3B</td><td>84.5</td><td>1232.9</td><td>67.6</td><td>74.7</td><td>61.0</td><td>27.4</td></tr></table>

$v = \{v_{1}, v_{2}, \cdots, v_{N}\}$ for multimodal understanding, we maximize the likelihood of text tokens by employing the standard language modeling objective:

$$
\mathcal {L} _ {\mathrm{NTP}} = \sum_ {i} \log p _ {\theta} (v _ {i} | v _ {1}, \dots , v _ {i - 1}, u _ {1}, \dots , u _ {M}), \tag {1}
$$

where $p(\cdot|\cdot)$ indicates the conditional probability which is modeled by the weights $\theta$ of Show-o and stochastic gradient descent is used to train the model. Note that, if the input sequence involves only text tokens, there are no conditional terms on image tokens $u = \{u_{1}, u_{2}, \cdots, u_{M}\}$ .

With the proof in Appendix A, we seamlessly integrate the simplified discrete diffusion modeling within Show-o by employing the mask token prediction as a learning objective. Hence, for modeling image tokens $u = \{u_{1}, u_{2}, \cdots, u_{M}\}$ within the input sequence, we first randomly replace the image tokens with the [MASK] token, notated as $u_{*}$ , at a random ratio (controlling by a time step) to create a masked sequence $u_{*} = \{u_{*}, u_{2}, \cdots, u_{*}, u_{M}\}$ . An illustration can be found in Fig. 9. Next, we aim to reconstruct the original image token from the masked tokens conditioning on unmasked regions and preceding text tokens by maximizing the following likelihood:

$$
\mathcal {L} _ {\mathrm{MTP}} = \sum_ {j} \log p _ {\theta} (u _ {j} | u _ {*}, u _ {2}, \dots , u _ {*}, u _ {M}, v _ {1}, \dots , v _ {N}). \tag {2}
$$

Note that the loss is only applied to the masked tokens. Specifically, we follow the sampling strategy used by MaskGIT Chang et al. (2022; 2023) to mask image tokens and reconstruct them via the information from all text and unmasked image tokens within the input sequence. Following the classifier-free guidance introduced by Ho & Salimans (2022), we randomly replace the conditioned text tokens using a null text “” with some probability.

Given a batch size of input sequences, the overall training loss is the combination of $L_{MTP}$ and $L_{NTP}$ :

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{MTP}} + \alpha \mathcal {L} _ {\mathrm{NTP}}, \tag {3}
$$

where $\alpha$ is the hyper-parameter weighting the loss term $L_{NTP}$ . The training schedule mainly involves three stages, and we provide more details in Appendix C.

Inference Stage. In multimodal understanding, given an image accompanying visual questions, Show-o autoregressively predicts textual answers. In visual generation, we use all $[MASK]$ tokens as initial input for Show-o, in which $[MASK]$ tokens will be iteratively replaced by the predicted image tokens within T steps. More inference details are provided in Appendix D.

# 4 EXPERIMENTS

# 4.1 EXPERIMENTAL SETUP

Datasets. We assemble two scales of datasets, i.e., around 35M and 2.0B image-text pairs, and collect around 2M high-quality data for multimodal understanding and generation fine-tuning. Besides,

![](images/d2185c5d3535dc9ca90b59a71390220f6158806952e901e2a05fe3fa6169db40.jpg)

<details>
<summary>natural_image</summary>

Underwater scene featuring a white sofa with coral plants and ocean waves, surrounded by coral-covered rocks (no text or symbols)
</details>

User: Do you think the image is unusual or not?

# Show-o:

The image is unusual because it depicts a living room with a couch and a table in the middle of the ocean, surrounded by a large body of water. This is not a typical setting for a living room, as living rooms are usually indoors and designed for relaxation and entertainment. The presence of the couch and table in the ocean adds a unique and unexpected element to the scene, making it stand out from typical living room setups.

# Chameleon:

The image is unusual because of the transparent and colorful fish and coral in the water. The image is also unusual because of the clear blue sky with white clouds. The unusual objects in the image are the fish and coral in the water. The unusual objects in the image are the fish and coral in the water. The unusual objects in the image are the fish and coral in the water. The unusual objects in the image are the fish and coral in the water.

Figure 6: Comparisons of VQA capabilities between Chameleon and Show-o.

RefinedWeb (Penedo et al., 2023) is adopted as text corpora to maintain the language modeling capability. Appendix E provides more details about these datasets.

Evaluation Details. Following LLaVA (Liu et al., 2024b), we evaluate the multimodal understanding capabilities of Show-o on POPE, MME, Flickr30k, VQAv2, GQA, and MMMU benchmarks. Besides, we adopt Fréchet Inception Distance (FID) on MSCOCO dataset to evaluate the generation fidelity of Show-o. Further, we follow SD3 (Esser et al., 2024) to evaluate the text-to-image generation capabilities of Show-o on the GenEval (Ghosh et al., 2023) benchmark.

Implementation details. Current version of Show-o is based on Phi-1.5 (1.3B) (Li et al., 2023). In the following, the default Show-o employs discrete image tokens as input for both multimodal understanding and generation. Show-o $^{\dagger}$ and Show-o $^{\ddagger}$ indicate the use of continuous image representations from the pre-trained MAGVIT-v2 and CLIP-ViT (corresponding to options (b) and (c) in Fig. 3), respectively, for multimodal understanding. Training details can be found in Appendix F.

# 4.2 MULTIMODAL UNDERSTANDING

Quantitative Evaluation. Table 1 presents the multimodal understanding capability of Show-o on public benchmarks, such as image captioning and visual question-answering tasks. i) The current version of Show-o is built upon Phi-1.5 and thus we follow LLaVA to train Show-o's understanding only counterpart as our direct baseline, namely LLaVA-v1.5-Phi-1.5. The proposed Show-o exhibits comparable performance in all evaluation metrics to the baseline LLaVA-v1.5-Phi-1.5, which is dedicated and optimized to only multimodal understanding. This demonstrates the great potential of our framework to unify multimodal understanding and generation in one single transformer. ii) When comparing with understanding only models including InstructBLIP, Qwen-VL-Chat, and mPLUG-Owl2 on multimodal understanding, our model with a much smaller model size also achieves competitive performance on POPE, MME, Flickr30k and VQAv2 benchmarks and performs better on GQA benchmark. iii) Compared with unified models with a much larger number of parameters, such as NExT-GPT-13B and Chameleon-34B, our model also achieves decent performance on Flickr30k benchmark and performs much better on VQAv2 benchmark.

Qualitative Results. We present Show-o's visual question-answering capability and make comparisons with Chameleon in Fig. 6. It is evident that when presented with a query image, Show-o can respond to commonly asked questions, even addressing the unusual aspects within the image. In the example of Fig. 6, when asked, “Do you think the image is unusual or not”, Chameleon fails to correctly identify the unusual aspect. In contrast, Show-o's response, “as living rooms are usually indoors and designed for relaxation and entertainment”, is more accurate.

# 4.3 VISUAL GENERATION

Results on MSCOCO 30K. We present zero-shot FID of Show-o on MSCOCO 30K in Table 2. It can be observed that, compared to generation models trained with larger numbers of parameters and training images such as GLIDE and DALL·E 2, Show-o achieves a better FID, i.e., 9.24, with only 1.3B parameters and 35M training data. Though Giga-GAN, Imagen, and RAPHAEL obtain a relatively better performance

Table 2: MSCOCO zero-shot FID. Und. and Gen. denote “understanding” and “generation”, respectively. 

<table><tr><td>Type</td><td>Method</td><td># Params</td><td># Images</td><td>FID-30K↓</td></tr><tr><td rowspan="9">Gen. Only</td><td>DALL·E (Ramesh et al., 2021)</td><td>12B</td><td>250M</td><td>27.50</td></tr><tr><td>GLIDE (Nichol et al., 2021)</td><td>5B</td><td>250M</td><td>12.24</td></tr><tr><td>LDM (Rombach et al., 2022)</td><td>1.4B</td><td>400M</td><td>12.64</td></tr><tr><td>DALL·E 2 (Ramesh et al., 2022a)</td><td>6.5B</td><td>650M</td><td>10.39</td></tr><tr><td>SDv1.5 (Rombach et al., 2022)</td><td>0.9B</td><td>2000M</td><td>9.62</td></tr><tr><td>GigaGAN (Kang et al., 2023)</td><td>0.9B</td><td>2700M</td><td>9.09</td></tr><tr><td>PixArt (Chen et al., 2024)</td><td>0.6B</td><td>25M</td><td>7.32</td></tr><tr><td>Imagen (Saharia et al., 2022)</td><td>3B</td><td>860M</td><td>7.27</td></tr><tr><td>RAPHAEL (Xue et al., 2024)</td><td>3B</td><td>5000M+</td><td>6.61</td></tr><tr><td rowspan="5">Und. and Gen.</td><td>CoDI (Tang et al., 2024)</td><td>-</td><td>400M</td><td>22.26</td></tr><tr><td>LWM (Liu et al., 2024a)</td><td>7B</td><td>-</td><td>12.68</td></tr><tr><td>SEED-X (Ge et al., 2024)</td><td>17B</td><td>-</td><td>14.99</td></tr><tr><td>DreamLLM (Dong et al., 2024)</td><td>7B</td><td>-</td><td>8.76</td></tr><tr><td>Show-o (Ours)</td><td>1.3B</td><td>35M</td><td>9.24</td></tr></table>

Table 3: Evaluation on the GenEval (Ghosh et al., 2023) benchmark. Und. and Gen. denote “understanding” and “generation”, respectively. We highlight the model size of Show-o in green, and we use blue to highlight the larger model size than ours. Obj.: Object. Attri.: Attribute. 

<table><tr><td>Type</td><td>Method</td><td># Params</td><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Color Attri.</td><td>Overall↑</td></tr><tr><td rowspan="8">Gen. Only</td><td>LlamaGen (Sun et al., 2024)</td><td>0.8B</td><td>0.71</td><td>0.34</td><td>0.21</td><td>0.58</td><td>0.07</td><td>0.04</td><td>0.32</td></tr><tr><td>LDM (Rombach et al., 2022)</td><td>1.4B</td><td>0.92</td><td>0.29</td><td>0.23</td><td>0.70</td><td>0.02</td><td>0.05</td><td>0.37</td></tr><tr><td>SDv1.5 (Rombach et al., 2022)</td><td>0.9B</td><td>0.97</td><td>0.38</td><td>0.35</td><td>0.76</td><td>0.04</td><td>0.06</td><td>0.43</td></tr><tr><td>PixArt-alpha (Chen et al., 2024)</td><td>0.6B</td><td>0.98</td><td>0.50</td><td>0.44</td><td>0.80</td><td>0.08</td><td>0.07</td><td>0.48</td></tr><tr><td>SDv2.1 (Rombach et al., 2022)</td><td>0.9B</td><td>0.98</td><td>0.51</td><td>0.44</td><td>0.85</td><td>0.07</td><td>0.17</td><td>0.50</td></tr><tr><td>DALL-E 2 (Ramesh et al., 2022a)</td><td>6.5B</td><td>0.94</td><td>0.66</td><td>0.49</td><td>0.77</td><td>0.10</td><td>0.19</td><td>0.52</td></tr><tr><td>SDXL (Podell et al., 2023)</td><td>2.6B</td><td>0.98</td><td>0.74</td><td>0.39</td><td>0.85</td><td>0.15</td><td>0.23</td><td>0.55</td></tr><tr><td>SD3 (d=24) (Esser et al., 2024)</td><td>2B</td><td>0.98</td><td>0.74</td><td>0.63</td><td>0.67</td><td>0.34</td><td>0.36</td><td>0.62</td></tr><tr><td rowspan="8">Und. and Gen.</td><td>CoDI (Tang et al., 2024)</td><td>-</td><td>0.89</td><td>0.16</td><td>0.16</td><td>0.65</td><td>0.02</td><td>0.01</td><td>0.31</td></tr><tr><td>LWM (Liu et al., 2024a)</td><td>7B</td><td>0.93</td><td>0.41</td><td>0.46</td><td>0.79</td><td>0.09</td><td>0.15</td><td>0.47</td></tr><tr><td>SEED-X (Ge et al., 2024)</td><td>17B</td><td>0.97</td><td>0.58</td><td>0.26</td><td>0.80</td><td>0.19</td><td>0.14</td><td>0.49</td></tr><tr><td>Emu3 Wang et al. (2024)</td><td>8B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.66</td></tr><tr><td>Transfusion Zhou et al. (2024)</td><td>7.3B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.63</td></tr><tr><td>Chameleon (Team, 2024)</td><td>7B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.39</td></tr><tr><td>Show-o (Ours)</td><td>1.3B</td><td>0.98</td><td>0.80</td><td>0.66</td><td>0.84</td><td>0.31</td><td>0.50</td><td>0.68</td></tr><tr><td>Show-o $^{\ddagger}$  (Ours)</td><td>1.3B</td><td>0.98</td><td>0.85</td><td>0.67</td><td>0.81</td><td>0.28</td><td>0.55</td><td>0.69</td></tr></table>

than Show-o, they are much larger in model size (3B v.s. 1.3B) and trained with much more data. In comparison to unified models, Show-o also exhibits improvement. However, it is worth noting that FID on MSCOCO 30K may not be a comprehensively accurate assessment of generation fidelity. The reason lies in the fact that existing generation models are commonly fine-tuned with high-quality and aesthetic images that do not align with the distribution of the MSCOCO dataset.

Results on GenEval. One can observe in Table 3 that when comparing to the model in a similar size such as LDM (1.4B), Show-o obtains significantly better performance in all six metrics, with an improvement of around 0.24 overall. Besides, Show-o achieves a better performance than DALL·E 2, which is 5 times larger in model size, and SDXL. Further, Show-o, with only 1.3B parameters, achieves comparable performance to models with around two times larger number of parameters such as SD3 (2B). It indicates that our unified model's generative capabilities are comparable to or even surpass those of specialized generation models. In comparison to unified models such as CoDI, SEED-X, and Chameleon, Show-o also demonstrates significant improvements.

Qualitative Results. We show image samples generated by Show-o in Fig. 7. One can observe that Show-o is capable of generating diverse, interesting, and realistic visual content in a resolution of $512 \times 512$ . For example, Show-o can generate a futuristic style of car, a highly detailed face, cute objects, and vivid scenery with vibrant contrast.

Text-guided Inpainting and Extrapolation. As mentioned, Show-o naturally supports text-based inpainting and extrapolation without requiring any fine-tuning. We illustrate examples in Fig. 8 (a). As shown on the top of the figure, given an input image and inpainting mask, Show-o can inpaint the original red trolley car to a blue sports car with sleek curves and tinted windows based on the user-provided text prompt. Specifically, we first tokenize the original image, mask those tokens to be inpainted, and then Show-o will gradually replace the masked tokens with predicted image tokens. Besides, Show-o is capable of extrapolating the original image horizontally/vertically based on the given text prompt. These cases significantly demonstrate the inherent advantages of Show-o over those autoregressive models for downstream applications.

# 4.4 MIXED-MODALITY GENERATION OF VIDEO KEYFRAMES AND CAPTIONS

Here, we explore the mixed-modality generation ability of Show-o based on the text descriptions and video keyframes in the GenHowTo dataset. Given a sequence of interleaved text descriptions and video keyframes (as shown at the bottom of Fig. 4), Show-o is trained to predict the next text tokens or keyframe tokens conditioning on all preceding tokens. Thus, Show-o can generate mixed-modality of text descriptions and video keyframes. Examining a single frame, these tokens are generated in a diffusion manner. When considering the modeling of long sequences, as subsequent keyframes are produced based on all preceding text and image information, this can also be viewed as a form of temporal auto-regressive modeling.

We have tried to train Show-o using instructional examples and present qualitative examples in Fig. 8 (b). For example, given a question “Can you guide me through making Avocado and Apple Juice”, Show-o exhibits the capability to generate video keyframes with text descriptions related to the question. It is apparent that the generated keyframes are temporally consistent.

![](images/5a0d36fa7e6483e395e247dace30148d753f67e345dce7f9c1953e5212c7f8ac.jpg)

<details>
<summary>natural_image</summary>

Futuristic concept car with glowing blue design, surrounded by futuristic metallic cylinders and city skyline (no text or symbols)
</details>

![](images/f6d1a732ef7537c6a6571bb1306fff00d397d6ea54cc61e67cffb8a512a75acc.jpg)

<details>
<summary>natural_image</summary>

Illustration of a colorful tiger with striped fur and eye-like eyes against a colorful abstract background (no text or symbols)
</details>

![](images/231edd41c71e16275b16eff78713d056d9e847d3029396952bc707f7284a178a.jpg)

<details>
<summary>natural_image</summary>

Stylized illustration of a hummingbird surrounded by flowers and a sun (no text or symbols)
</details>

![](images/b1692d345908faa862988d91e23502c5b67c5eba40934f6ad9f7b9f6a3852196.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a woman with green facial features and makeup (no text or symbols visible)
</details>

![](images/7a8072af9d6459c267b5b88d6ea2789429db04ed970f89e2d4e53c8d0438c231.jpg)

<details>
<summary>natural_image</summary>

Fantasy beach scene with a glowing pink cloud, a colorful smokestack, and a figure in a flowing dress (no text or symbols)
</details>

![](images/465d76ba91c09a262607d7b4723d021185e11e95fb8adf7352f412bec3ea3603.jpg)

<details>
<summary>natural_image</summary>

Cute cartoon character resembling a white dumpling with yellow eyes and a red ribbon, set against a gradient pink-to-blue background (no text or symbols)
</details>

Figure 7: Images generated by Show-o. Text prompts are provided in Appendix G.

# 4.5 VIDEO UNDERSTANDING AND GENERATION

Beyond image understanding and generation, we have explored Show-o to support video understanding and generation. We inflate the existing pre-trained MAGVIT-v2 to support encoding videos into discrete tokens, compressing an 8 FPS video tensor of $3 \times 17 \times 256 \times 256$ into $5 \times 16 \times 16$ . In this way, following the sequence format and omni-attention of multimodal understanding and generation introduced in Section 3.2, it is convenient to fine-tune the existing Show-o to involve video understanding and generation. One can observe visual examples in Figs. 8 (c) and (d) that Show-o can accurately comprehend and describe the variations in the video and generate consistent video frames with “a jeep car is approaching from a distance”. More examples can be found in Appendix H.

# 4.6 ABLATION STUDIES

Impact of dataset scale and image resolution on multimodal understanding. As only discrete image tokens are extracted from the vision tokenizer, it is required to learn image token

Table 4: Impact of dataset scale and image resolution on the learning of discrete image token embeddings for multimodal understanding. 

<table><tr><td># Image-text</td><td>Resolution</td><td>POPE</td><td>MME</td><td>Flickr30k</td><td>VQAv2(test)</td><td>GQA</td><td>MMMU</td></tr><tr><td>35M</td><td> $256^{2}$ </td><td>73.8</td><td>948.4</td><td>36.2</td><td>59.3</td><td>48.7</td><td>25.1</td></tr><tr><td>2.0B</td><td> $256^{2}$ </td><td>76.2</td><td>1014.9</td><td>48.9</td><td>64.7</td><td>54.2</td><td>25.0</td></tr><tr><td>2.0B</td><td> $512^{2}$ </td><td>80.0</td><td>1097.2</td><td>62.5</td><td>69.4</td><td>58.0</td><td>26.7</td></tr></table>

embeddings in Show-o from scratch. Unlike aligned image representations from the CLIP well-trained on a large-scale image-text dataset, Show-o necessitates the multimodal alignment between image and text embeddings during the pre-training stages. Here, we study the impact of the dataset scale and image resolution on the learning of discrete image token embeddings for multimodal understanding in Table 4. One can observe that the multimodal understanding capabilities of Show-o are consistently improved when increasing the data scale and image resolution. This reveals that it is required to involve more image-text pairs for multimodal alignment and more image tokens to represent an image for better comprehending the image information.

As illustrated in Fig. 3(a), the default Show-o adopts the pre-trained MAGVIT-v2 to tokenize input image to discrete tokens, which are then passed to the embedding layer to obtain embeddings as input for multimodal understanding. Beyond, we provide a systematic exploration of different design choices for the input of Show-o to enhance multimodal understanding. Specifically, as shown in Fig. 3(b) and (c), instead of discrete image tokens, we extract the continuous image representations

Original Image   
"A blue sports car with sleek curves and tinted windows." "A serene natural landscape featuring a clear, blue lake surrounded by lush green trees."   
![](images/d8ae82ea563eb678aaaf539e4aa660377df80688b32eb6113cb185d72757b98a.jpg)

![](images/b597bfb7cea99c57d9ede962cf5bb8f8cb1cd68d438b692831ab915aa4dd8f77.jpg)

![](images/23d0cba1d24a7fc9f8ae366d81fbf350bdc350afa7b6dc8d55bf77b859f8d49e.jpg)

![](images/43d7aa97240faee7c8e6658405d2565f382a0cf7b910b890ddcbf9997a7076d6.jpg)

![](images/65aee81aafba55c34ce227f823bdad922478c1793b24d7511e9e1b956ca2b2dc.jpg)

![](images/f15275c18e5be4de09f8b11c090ed674b79f96fbd76ac274b95817b3e23a0cf5.jpg)  
(a) Text-guided Inpainting and Extrapolation

User: Can you guide me through making Avocado and Apple Juice?   
![](images/cc5acaca94e5f263d29859bb534788fbcf7e01a054eb929eb4b34f97502c7feb.jpg)  
(1) Apple Peeling/Cutting.

![](images/1d966f33e5957b1b04dd4a24cf6355b2d92e462a503b6e08e08f6c054afcb570.jpg)  
(2) a person is cutting an apple.

![](images/640f74ad8ec4ecbb593ccf251c9537a0d1f5facd3dfed4085d6ec35580bd3cd4.jpg)  
(3) a person slicing an apple on a plate.

![](images/02622cb3390f021d1faeb45900f6cd5ccbb946c32be2c8dc5f6da85621c1adfb.jpg)  
(4) Avocado Peeling/Slicing.

![](images/3a783f0ad429e3673de151dc49a8475af9fef198715245f07139669070151a83.jpg)  
(5) a woman is cutting an avocado on a plate.

![](images/e7414971c3495964ca2024a2eba98174a9e8c97ed88740d77e3c9a61bed44db2.jpg)  
(6) a food processor with avocado in it.   
Generated Text Descriptions with Corresponding Video Keyframes

(b) Mixed-modality Generation   
![](images/74cbcf8c9f01ed7d03b2fe2dba674b565e0c24ca85aaacf85805a223114e899d.jpg)  
User: Describe the video content.

![](images/0c1d159361b10df52cc79566e6791e7d085f9dabbfc337007710419bdc5ec40b.jpg)  
content.

![](images/6b220bf4f74a516c3297fb926efa5fa4e9ace6287a7f61016a0841c9bd435d9a.jpg)  
[Non-Text]

![](images/40dee6844367233d65231ab1ff978568ecb55b1dafd997fbfeded221209c1155.jpg)  
[Non-Text]

![](images/83079ccc4a5de074526680d1928b31ee7767006cd61ea944b16ed63e07fea4d8.jpg)  
[Non-Text]

![](images/6a6d7811cf0059538394952dea9c6899a61f0df2e08ac9d58bc29fa6661a5746.jpg)

Show-o: The video shows a little girl sitting in a car seat, holding an ice cream cone in her hand. She appears to be enjoying her treat, and her eyes are closed, suggesting she is either asleep or simply taking a moment to relax. The car seat she is sitting on is black, and the background of the video is blurred, indicating that the focus is on the girl and her ice cream. The lighting in the car is bright, and the car seat is positioned in front of a window, providing a view of the outside environment.   
(c) Video Understanding   
![](images/d20df9571224bde12e790475fa62bb9e6ae58ca087126f36c630eb4c85a5d038.jpg)

![](images/b95019f654d2f01fb39e371c4cfd25612f4fb0f0cae877b3d152f0de2f34d9be.jpg)

![](images/228558d94d692df2587ea41265bc9c04c17ea1179510d7b323041bf2d4ffd52f.jpg)

![](images/93c3485c01aeee17dc3684c0fee7f05591416b65de1e22b6880fe544fa1856b6.jpg)

![](images/99282c96fba6a4fa88d4ff56dde6f6d6c5872b06c4337cd4c83150cb9e1dd16e.jpg)

![](images/58ff02a560ce103e0656f2c2f4f8346059aa7658348df0e9448a173575b59e48.jpg)  
"A jeep car is approaching from a distance."   
(d) Video Generation   
Figure 8: Examples of text-guided inpainting/extrapolation, mixed-modality generation, video understanding and generation. 6 frames are sampled from the (generated) video for illustration.

from the pre-trained MAGVIT-v2 and CLIP-ViT, respectively, as input for Show-o when dealing with multimodal understanding. We provide experimental results and insights in Appendix I.

Additionally, we present qualitative examples to illustrate the impact of sampling steps and classifier-free guidance for text-to-image generation in Appendix I. In observation, increasing the sampling steps can allow the synthesis of an image that closely adheres to the prompt and improve fidelity. Besides, the classifier-free guidance can significantly make the colors and contents more diverse and consistent with the given text prompt.

We also discuss the failure modes of Show-o in Appendix K.

# 5 CONCLUSION

This paper proposed a unified transformer, i.e., Show-o, to unify multimodal understanding and generation. Show-o for the first time unified autoregressive and (discrete) diffusion modeling that can handle different modalities in distinct ways. Extensive experimental results demonstrated that Show-o is comparable to even better than individual expert models across a wide range of vision-language tasks. This highlighted its potential as a next-generation foundation model.

# 6 ACKNOWLEDGMENTS

This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No:AISG3-RP-2022-030).

We would like to express our sincere gratitude to Henry Hengyuan Zhao for his valuable discussions and insightful feedback on multimodal understanding, Mingrui Wang for his patient assistance in helping us set up the development environment, and Han Yao for providing preprocessed video datasets.

# REFERENCES

Emanuele Aiello, LILI YU, Yixin Nie, Armen Aghajanyan, and Barlas Oguz. Jointly training large autoregressive multimodal models. In ICLR, 2024.   
Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 1, 2023.   
Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. NeurIPS, pp. 17981–17993, 2021.   
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. CoRR, abs/2308.12966, 2023.   
Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.   
Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023.   
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020.   
Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.   
Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. NeurIPS, pp. 28266–28279, 2022.   
Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, pp. 11315–11325, 2022.   
Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.   
Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pp. 3558–3568, 2021.   
Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Zhongdao Wang, James T. Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart- $\alpha$ : Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR. OpenReview.net, 2024.

Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.   
Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, pp. 1691–1703, 2020.   
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barrett Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1–240:113, 2023.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.   
Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme Ruiz, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin Fathy Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Collier, Alexey A. Gritsenko, Vighnesh Birodkar, Cristina Nader Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Pavetic, Dustin Tran, Thomas Kipf, Mario Lucic, Xiaohua Zhai, Daniel Keysers, Jeremiah J. Harmsen, and Neil Houlsby. Scaling vision transformers to 22 billion parameters. In ICML, pp. 7480–7512, 2023.   
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pp. 248–255, 2009.   
Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. DreamLLM: Synergistic multimodal comprehension and creation. In ICLR, 2024.   
Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pp. 12873–12883, 2021.   
Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.   
Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. NeurIPS, 2024.   
Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.   
Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In EMNLP, pp. 6111–6120, 2019.   
Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In NeurIPS, 2023.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 2014.   
Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, pp. 10696–10706, 2022.   
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, pp. 6840–6851, 2020a.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, pp. 6840–6851, 2020b.   
Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 2022.   
Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In ICLR. OpenReview.net, 2022.   
Minghui Hu, Chuanxia Zheng, Zuopeng Yang, Tat-Jen Cham, Heliang Zheng, Chaoyue Wang, Dacheng Tao, and Ponnuthurai N. Suganthan. Unified discrete diffusion for simultaneous vision-language generation. In ICLR, 2023.   
Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, pp. 10124–10134. IEEE, 2023.   
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, pp. 4015–4026, 2023.   
Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.   
Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 16(1-2):1–214, 2024.   
Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.   
Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint, 2024a.   
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pp. 26296–26306, 2024b.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2024c.   
Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26439–26455, 2024.   
Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.   
Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In ICML, pp. 4055–4064, 2018.   
William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pp. 4195-4205, 2023.   
Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refined-web dataset for falcon LLM: outperforming curated corpora with web data only. In NeurIPS, 2023.   
Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763, 2021.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.   
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pp. 8821–8831. Pmlr, 2021.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. CoRR, abs/2204.06125, 2022a.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022b.   
Suman Ravuri and Oriol Vinyals. Classification accuracy score for conditional generative models. NeurIPS, 32, 2019.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35:36479–36494, 2022.   
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pp. 2256–2265, 2015.   
Tomáš Souček, Dima Damen, Michael Wray, Ivan Laptev, and Josef Sivic. Genhowto: Learning to generate actions and state transformations from instructional videos. In CVPR, pp. 6561–6571, 2024.   
Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.   
Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. CoRR, abs/2312.13286, 2023a.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. CoRR, abs/2307.05222, 2023b.   
Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In ICLR, 2023c.   
Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. NeurIPS, 36, 2024.   
Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.   
Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 30, 2017a.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017b.   
Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.   
Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for large-scale transformer training instabilities. arXiv preprint arXiv:2309.14322, 2023.   
Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023a.   
Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multi-modal llm. arXiv preprint arXiv:2309.05519, 2023b.   
Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.   
Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In ICCV, pp. 7452–7461, 2023.   
Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. NeurIPS, 36, 2024.   
Hanrong Ye, De-An Huang, Yao Lu, Zhiding Yu, Wei Ping, Andrew Tao, Jan Kautz, Song Han, Dan Xu, Pavlo Molchanov, et al. X-vila: Cross-modality alignment for large language model. arXiv preprint arXiv:2405.19335, 2024a.   
Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In CVPR, pp. 13040–13051, 2024b.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.   
Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion-tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.   
Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Copilot4d: Learning unsupervised world models for autonomous driving via discrete diffusion. In ICLR, 2024.   
Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. 2024. URL https://api.semanticscholar.org/CorpusID:271909855.   
Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. CoRR, abs/2304.10592, 2023a.   
Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, and Ying Shan. VL-GPT: A generative pre-trained transformer for vision and language understanding and generation. CoRR, abs/2312.09251, 2023b.

Image corruption by adding different level mask (noise) tokens   
![](images/c4231b8ff6e60e0bf14662c76f5a40adde3012cf60fccabe3bb31bd1e58321ea.jpg)

<details>
<summary>text_image</summary>

Image generation by iteratively removing mask (noise) tokens
</details>

Figure 9: Illustration of image corruption by adding different level mask (noise) tokens and image generation by iteratively removing mask (noise) tokens in the absorbing discrete diffusion paradigm.

# APPENDIX

# A PRELIMINARIES

In recent years, denoising diffusion probabilistic models (DDPMs) (Ho et al., 2020a) have demonstrated unprecedented performance in text-to-image/video generation in continuous state spaces, particularly exemplified by the popular Stable Diffusion series (Podell et al., 2023; Esser et al., 2024). Concurrently, discrete denoising diffusion probabilistic models (D3PMs) (Austin et al., 2021) have also shown impressive capabilities in modeling data in discrete form, featuring models like VQ-Diffusion (Gu et al., 2022) and Copilot4D (Zhang et al., 2024). Further, MaskGIT (Chang et al., 2022) and Muse (Chang et al., 2023) have demonstrated a simplified discrete diffusion that can effectively model discrete image tokens. Our Show-o model is built upon MaskGIT so that both such discrete visual and textual tokens can share a unified learning objective format. In the following, we provide preliminaries for diffusion models and draw the connection between discrete diffusion and mask token prediction employed in MaskGIT.

In diffusion models, the forward process $q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})$ corrupts the image data $x_0 \sim q(\mathbf{x}_0)$ into latent variables $x_1, \cdots, x_T$ in different noise level. The reverse Markov process is learned to iteratively remove the noises added to the latent variables towards the real image distribution $q(\mathbf{x}_0)$ . In the continuous scenario, the transition distribution $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ is commonly characterized by a Gaussian distribution:

$$
q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) = \mathcal {N} (\mathbf {x} _ {t} | \sqrt {1 - \beta_ {t}} \mathbf {x} _ {t - 1}, \beta_ {t} \mathbf {I}), \tag {4}
$$

where the mean is $\sqrt{1-\beta_{t}}x_{t-1}$ and the variance is $\beta_{t}$ . For images tokenized into K (i.e., the codebook size) categorical random variables $x_{t}, x_{t-1} \in \{1, \cdots, K\}$ and given a [MASK] state, the transition distribution is instead formulated by a stochastic transition matrix $\mathbf{Q}_{t} \in \mathbb{R}^{(K+1) \times (K+1)}$ :

$$
q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t - 1}\right) = \operatorname{Cat} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t - 1} \mathbf {Q} _ {t}\right), \tag {5}
$$

where $[\mathbf{Q}_{t}]_{ij} = q(\mathbf{x}_{t} = j|\mathbf{x}_{t-1} = i)$ , $x_{t-1}Q_{t}$ indicates the row vector-matrix product, and $\text{Cat}(\mathbf{x}_{t}|\mathbf{x}_{t-1}\mathbf{Q}_{t})$ is a categorical distribution over the one-hot row vector $x_{t}$ given by $x_{t-1}Q_{t}$ . When the transition matrix $Q_{t}$ is applied to each image token in a sequence, the marginal and posterior at time step t and t-1, respectively, are formulated as:

$$
q (\mathbf {x} _ {t} | \mathbf {x} _ {0}) = \operatorname{Cat} \left(\mathbf {x} _ {t} | \mathbf {x} _ {0} \overline {{\mathbf {Q}}} _ {t}\right), \quad \text { where } \quad \overline {{\mathbf {Q}}} _ {t} = \mathbf {Q} _ {1} \mathbf {Q} _ {2} \dots \mathbf {Q} _ {t},
$$

$$
q \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t}, \mathbf {x} _ {0}\right) = \frac {q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t - 1} , \overline {{{\mathbf {x}}}} _ {0}\right) q \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {0}\right)}{q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right)} = \operatorname{Cat} \left(\mathbf {x} _ {t - 1} \mid \frac {\mathbf {x} _ {t} \mathbf {Q} _ {t} ^ {\top} \odot \mathbf {x} _ {0} \overline {{{\mathbf {Q}}}} _ {t - 1}}{\mathbf {x} _ {0} \overline {{{\mathbf {Q}}}} _ {t} \mathbf {x} _ {t} ^ {\top}}\right), \tag {6}
$$

where $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ because of the Markov property.

In the following, we introduce the Absorbing-Uniform Discrete Diffusion by defining the stochastic transition matrix $\mathbf{Q}_t$ as follows:

$$
\mathbf {Q} _ {t} = \mathbf {Q} _ {t} ^ {a} \mathbf {Q} _ {t} ^ {u}, \tag {7}
$$

where $e_{m}$ is a one-hot vector with a value of 1 at the index of [MASK] token, $\mathbf{Q}_{t}^{a} = (1 - \alpha_{t})\mathbf{I} + \alpha_{t}\mathbf{1}\mathbf{e}_{m}^{\top}$ , and $\mathbf{Q}_{t}^{u} = \mathbf{I} - \beta_{t}(\mathbf{I} - \mathbf{e}_{m}\mathbf{e}_{m}^{\top}) + \frac{\beta_{t}}{(K+1)}(\mathbf{1} - \mathbf{e}_{m})(\mathbf{1} - \mathbf{e}_{m})^{\top}$ . Here, $\alpha_{t}$ and $\beta_{t}$ represent the

probabilities of an image token transforming into the [MASK] token and non-[MASK] token at time step t, respectively. Specifically, the matrix form of $Q_{t}$ can be written as:

$$
\mathbf {Q} _ {t} = \left[ \begin{array}{c c c c c c} \omega_ {t} + \nu_ {t} & \nu_ {t} & \nu_ {t} & \dots & \nu_ {t} & \alpha_ {t} \\ \nu_ {t} & \omega_ {t} + \nu_ {t} & \nu_ {t} & \dots & \nu_ {t} & \alpha_ {t} \\ \nu_ {t} & \nu_ {t} & \omega_ {t} + \nu_ {t} & \dots & \nu_ {t} & \alpha_ {t} \\ \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ \nu_ {t} & \nu_ {t} & \nu_ {t} & \dots & \omega_ {t} + \nu_ {t} & \alpha_ {t} \\ 0 & 0 & 0 & \dots & 0 & 1 \end{array} \right], \tag {8}
$$

where $\omega_{t}=(1-\alpha_{t}-\beta_{t})$ and $\nu_{t}=\frac{\beta_{t}}{(K+1)}$ . Intuitively, during the corruption process, each image token in the sequence has a probability of $\alpha_{t}$ to be replaced by the [MASK] token, a chance of $\nu_{t}$ to be uniformly diffused, and a probability of $\omega_{k}+\nu_{t}$ remain unchanged. Besides, if a token turns into a [MASK] token, it will stay in the same [MASK] state during the following corruption process. Likewise, $\overline{Q}_{t}=\overline{Q}_{t}^{a}\overline{Q}_{t}^{u}$ can be accordingly derived. An illustration of the image corruption process using [MASK] token is provided in Fig. 9.

The evidence-lower bound (ELBO) for the variational diffusion models is:

$$
\begin{array}{l} - \mathcal {L} _ {\mathrm{ELBO}} (\mathbf {x} _ {0}, \theta) = \mathbb {E} _ {q (\mathbf {x} _ {1: T} | \mathbf {x} _ {0})} \left[ - \underbrace {D _ {\mathrm{KL}} [ q (\mathbf {x} _ {T} | \mathbf {x} _ {0}) \| p (\mathbf {x} _ {T}) ]} _ {\mathcal {L} _ {T}} + \underbrace {\log p _ {\theta} (\mathbf {x} _ {0} | \mathbf {x} _ {1})} _ {\mathcal {L} _ {0}} \right. \\ \left. - \sum_ {t = 2} ^ {T} \underbrace {D _ {\mathrm{KL}} \left[ q \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t} , \mathbf {x} _ {0}\right) \| p _ {\theta} \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t}\right) \right]} _ {\mathcal {L} _ {t - 1}} \right]. \tag {9} \\ \end{array}
$$

Considering the proposition in (Campbell et al., 2022) and the following parameterization of the reverse process:

$$
p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}) = \sum_ {\mathbf {x} _ {0}} q (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}, \mathbf {x} _ {0}) p _ {\theta} (\mathbf {x} _ {0} | \mathbf {x} _ {t}), \tag {10}
$$

the variational lower bound can be further expressed under the image distribution $q(\mathbf{x}_{0})$ (referring to the proof provided by Zhang et al. (2024) as detailed in the Appendix B):

$$
\mathbb {E} _ {q (\mathbf {x} _ {0})} [ \log p _ {\theta} (\mathbf {x} _ {0}) ] \geq \mathbb {E} _ {q (\mathbf {x} _ {0})} [ - \mathcal {L} _ {\mathrm{ELBO}} (\mathbf {x} _ {0}, \theta) ] \geq \sum_ {t = 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {0}) q (\mathbf {x} _ {t} | \mathbf {x} _ {0})} [ \log p _ {\theta} (\mathbf {x} _ {0} | \mathbf {x} _ {t}) ] + C. \tag {11}
$$

When deriving this lower bound, the discrete diffusion paradigm can be further simplified by restricting each image token to be either unchanged or replaced with the [MASK] token, with no possibility of becoming other categorical variables. The resulting lower bound is effectively the Cross-Entropy loss used in MaskGIT (Chang et al., 2022), which is the mask token prediction to learn a neural network $p_{\theta}$ to reconstruct masked regions of $\mathbf{x}_0$ from the noised $\mathbf{x}_t$ . In this work, we follow MaskGIT to integrate this simplified discrete diffusion paradigm into Show-o because of its simplicity. Further, Muse (Chang et al., 2023) has successfully scaled up such a paradigm for text-to-image models of 3B parameters using 460M image-text pairs.

# B ALTERNATIVE LOWER BOUND FOR THE VARIATIONAL DIFFUSION

$$
\begin{array}{l} \mathbb {E} _ {q (\mathbf {x} _ {0})} [ \log p _ {\theta} (\mathbf {x} _ {0}) ] \\ = \mathbb {E} _ {q (\mathbf {x} _ {0})} [ \log \int p _ {\theta} (\mathbf {x} _ {0}, \mathbf {x} _ {1} \dots \mathbf {x} _ {T}) \mathrm{d} \mathbf {x} _ {1} \dots \mathbf {x} _ {T} ] \\ = \mathbb {E} _ {q (\mathbf {x} _ {0})} \left\{\log \mathbb {E} _ {q (\mathbf {x} _ {1: T} | \mathbf {x} _ {0})} \left[ \frac {p _ {\theta} (\mathbf {x} _ {0 : T - 1} | \mathbf {x} _ {T})}{q (\mathbf {x} _ {1 : T} | \mathbf {x} _ {0})} p (\mathbf {x} _ {T}) \right] \right\} \\ \geq \mathbb {E} _ {q (\mathbf {x} _ {0}) q (\mathbf {x} _ {1: T} | \mathbf {x} _ {0})} \left[ \log \frac {p _ {\theta} (\mathbf {x} _ {0 : T - 1} | \mathbf {x} _ {T})}{q (\mathbf {x} _ {1 : T} | \mathbf {x} _ {0})} + \log p (\mathbf {x} _ {T}) \right] \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ \sum_ {t \geq 1} ^ {T} \log \frac {p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})}{q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1})} + \log p (\mathbf {x} _ {T}) \bigg ] \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \left[ \sum_ {t \geq 1} ^ {T} \log p _ {\theta} \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t}\right) + \log p \left(\mathbf {x} _ {T}\right) - \sum_ {t \geq 1} ^ {T} \log q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t - 1}\right) \right] \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ \sum_ {t \geq 1} ^ {T} \log \sum_ {\tilde {\mathbf {x}} _ {0}} q (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}, \tilde {\mathbf {x}} _ {0}) \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) \bigg ] + \underbrace {\mathbb {E} _ {q (\mathbf {x} _ {0 : T})} \bigg [ \log p (\mathbf {x} _ {T}) - \sum_ {t \geq 1} ^ {T} \log q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) \bigg ]} _ {C _ {1}} \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \left[ \sum_ {t \geq 1} ^ {T} \log \sum_ {\tilde {\mathbf {x}} _ {0}} \frac {q (\mathbf {x} _ {t - 1} , \tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t})}{q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t})} \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) \right] + C _ {1} \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \left[ \sum_ {t \geq 1} ^ {T} \log \sum_ {\tilde {\mathbf {x}} _ {0}} \frac {q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1})}{q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t})} \overbrace {q (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})} ^ {q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) q (\mathbf {x} _ {t - 1}) / q (\mathbf {x} _ {t})} \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) \right] + C _ {1} \\ \geq \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \Big [ \sum_ {t \geq 1} ^ {T} \sum_ {\tilde {\mathbf {x}} _ {0}} q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1}) \log \left(\frac {q (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})}{q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t})} \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t})\right) \Big ] + C _ {1} \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ \sum_ {t \geq 1} ^ {T} \sum_ {\tilde {\mathbf {x}} _ {0}} q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1}) \log \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) \bigg ] + C _ {1} + \underbrace {\mathbb {E} _ {q (\mathbf {x} _ {0 : T})} \bigg [ \sum_ {t \geq 1} ^ {T} \sum_ {\tilde {\mathbf {x}} _ {0}} q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1}) \log \frac {q (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})}{q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t})} \bigg ]} _ {C _ {2}} \\ = \sum_ {t \geq 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {0}, \mathbf {x} _ {t - 1}, \mathbf {x} _ {t})} \left[ \sum_ {\tilde {\mathbf {x}} _ {0}} q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1}) \log \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) \right] + C _ {1} + C _ {2} \\ = \sum_ {t \geq 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {0}, \mathbf {x} _ {t - 1}, \mathbf {x} _ {t}) q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1})} [ \log \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) ] + C _ {1} + C _ {2} \\ = \sum_ {t \geq 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {0} | \mathbf {x} _ {t - 1}) q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) q (\mathbf {x} _ {t - 1}) q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1})} [ \log \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) ] + C _ {1} + C _ {2} \\ = \sum_ {t \geq 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) q (\mathbf {x} _ {t - 1}, \tilde {\mathbf {x}} _ {0})} [ \log \tilde {p} _ {\theta} (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) ] + C _ {1} + C _ {2} \\ = \sum_ {t \geq 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {t}, \mathbf {x} _ {0})} [ \log \tilde {p} _ {\theta} (\mathbf {x} _ {0} | \mathbf {x} _ {t}) ] + C _ {1} + C _ {2} \\ \end{array}
$$

The constants $C_1$ and $C_2$ are:

$$
\begin{array}{l} C _ {1} = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ - \sum_ {t = 1} ^ {T} \log q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) + \underbrace {\log p (\mathbf {x} _ {T})} _ {\text { Note   that } p (\mathbf {x} _ {T}) = q (\mathbf {x} _ {T})} \bigg ] \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ - \sum_ {t = 1} ^ {T} \log q (\mathbf {x} _ {t}, \mathbf {x} _ {t - 1}) + \sum_ {t = 0} ^ {T} \log q (\mathbf {x} _ {t}) \bigg ] \\ C _ {2} = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ \sum_ {t = 1} ^ {T} \log q (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}) \bigg ] - \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ \sum_ {t = 1} ^ {T} \sum_ {\tilde {\mathbf {x}} _ {0}} q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1}) \log q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) \bigg ] \\ = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} \bigg [ \sum_ {t = 1} ^ {T} \log q (\mathbf {x} _ {t}, \mathbf {x} _ {t - 1}) - \sum_ {t = 1} ^ {T} \log q (\mathbf {x} _ {t}) \bigg ] - \sum_ {t = 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {0: T}) q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t - 1})} [ \log q (\tilde {\mathbf {x}} _ {0} | \mathbf {x} _ {t}) ] \\ \end{array}
$$

$$
C _ {1} + C _ {2} = \mathbb {E} _ {q (\mathbf {x} _ {0: T})} [ \log q (\mathbf {x} _ {0}) - \sum_ {t = 1} ^ {T} \log q (\mathbf {x} _ {0} | \mathbf {x} _ {t}) ]
$$

The alternative lower bound can be derived as:

$$
\begin{array}{l} \mathbb {E} _ {q (\mathbf {x} _ {0})} [ \log p _ {\theta} (\mathbf {x} _ {0}) ] \geq \sum_ {t = 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {t}, \mathbf {x} _ {0})} [ \log p _ {\theta} (\mathbf {x} _ {0} | \mathbf {x} _ {t}) ] + \mathbb {E} _ {q (\mathbf {x} _ {0: T})} [ \log q (\mathbf {x} _ {0}) - \sum_ {t = 1} ^ {T} \log q (\mathbf {x} _ {0} | \mathbf {x} _ {t}) ] \\ = \sum_ {t = 1} ^ {T} \mathbb {E} _ {q (\mathbf {x} _ {0}) q (\mathbf {x} _ {t} | \mathbf {x} _ {0})} [ \log p _ {\theta} (\mathbf {x} _ {0} | \mathbf {x} _ {t}) ] + C. \\ \end{array}
$$

This proof is provided by Zhang et al. (2024).

# C TRAINING PIPELINE

Given that the embedding of image tokens is newly initialized, it necessitates large-scale pre-training to align for multimodal understanding and generation. Besides, Show-o eliminates the text encoder to extract text embeddings for text-to-image generation, which poses a significant challenge for achieving effective alignment between text and image content within one single transformer. To this end, we employ a three-stage approach to progressively and effectively train Show-o:

i) Image Token Embedding and Pixel Dependency Learning: We employ RefinedWeb (Penedo et al., 2023) dataset to train Show-o to maintain the language modeling ability. Meanwhile, ImageNet-1K dataset (Deng et al., 2009) and large-scale image-text pairs are adopted to train Show-o for class-conditional image generation and image captioning, respectively. Here, we directly leverage the class names from ImageNet-1K as textual inputs for learning class-conditional image generation. This stage primarily involves the learning of new learnable embeddings for discrete image tokens, pixel dependency for image generation, and alignment between image and text for image captioning.

ii) Image-Text Alignment for Multimodal Understanding and Generation: Building upon the pre-trained weights, we proceed to involve training of text-to-image generation on the image-text data instead of the ImageNet-1K. This stage mainly focuses on image and text alignment for both image captioning and text-to-image generation.

iii) High-Quality Data Fine-tuning: Lastly, we further refine the pre-trained Show-o model by incorporating filtered high-quality image-text pairs for text-to-image generation and instructional data for multimodal understanding and mixed-modality generation.

# D INFERENCE DETAILS

In inference, two types of predictions, i.e., text and image tokens, are involved in Show-o. In multimodal understanding, given the conditional image and questions, text tokens are auto-regressively sampled from the predicted tokens with higher confidence. In visual generation, given N text tokens and M [MASK] tokens as initial input, Show-o predict M logits $\ell^{t} = \{\ell_{i}^{t}\}_{i=1}^{M}$ in parallel, where

$\ell_{i}^{t}\in\mathbb{R}^{1\times(K+1)}$ and t is the time step. Following the work (Chang et al., 2023), we compute both the conditional logit $\ell_{c}^{t}$ and unconditional logit $\ell_{u}^{t}$ for masked tokens. The final logit $\ell^{t}$ of each [MASK] token is obtained by the following equation with a guidance scale w:

$$
\ell^ {t} = (1 + w) \ell_ {c} ^ {t} - w \ell_ {u} ^ {t}. \tag {12}
$$

For each [MASK] token $u_*$ at location $i$ , we sample an image token $u_i^t$ from the codebook based on the predicted probability $p_i^t = \text{softmax}(\ell_i^t)$ and indicate its predicted score $s_i \in \mathbb{R}$ as the confidence of this sampled token. The confidence of the unmasked token is set as 1.0. Next, we compute the number of image tokens $m$ that should be re-masked based on the mask scheduling function $\gamma$ , where $m = \lceil \gamma(\frac{t}{T})M \rceil$ . More details of mask scheduling functions can be found in MaskGIT (Chang et al., 2022). Subsequently, we replace the predicted image tokens with [MASK] token $u_*$ based on the following metric:

$$
u _ {i} ^ {(t + 1)} = \left\{ \begin{array}{l l} u _ {*}, & \text { if   } s _ {i} <   \operatorname{sorted} _ {j} (s _ {j}) [ m ]. \\ u _ {i} ^ {t}, & \text { otherwise }. \end{array} \right. \tag {13}
$$

The resulting sequence, consisting of the remaining image tokens and $[MASK]$ tokens, will be fed back to Show-o for the subsequent round of prediction until reaching the final time step T. The finalized image tokens are decoded by the image tokenizer into an image.

# E DATASET DETAILS

Three types of data are adopted for training Show-o: i) Text-only Data: We employ the publicly available RefinedWeb dataset (Penedo et al., 2023) to preserve the text reasoning capabilities of the pre-trained LLM. This dataset comprises approximately 1 billion instances (equivalent to 968 million individual web pages) and totals 2.8 terabytes of curated text data. ii) Image Data with Class Names: Show-o learns pixel dependencies using 1.28M images sourced from the ImageNet-1K (Deng et al., 2009) dataset. iii) Image-Text Data: For pre-training tasks correspond to multimodal understanding and generation, we assemble roughly 35M image-text pairs from the publicly available datasets including CC12M (Changpinyo et al., 2021), SA1B (Kirillov et al., 2023), and LAION-aesthetics-12M $^{\dagger}$ . Additionally, we further increase the data scale to around 2.0B by incorporating DataComp (Gadre et al., 2024) and COYO700M (Byeon et al., 2022) with some filtering strategies. Note that, we employ ShareGPT4V (Chen et al., 2023) to re-caption these datasets. Additionally, around 1M internal image-text pairs serve as high-quality text-to-image generation datasets for the final fine-tuning. Following LLaVA-v1.5 (Liu et al., 2024b), we incorporate LLaVA-Pretrain-558K and LLaVA-v1.5-mix-665K for instruction tuning. Moreover, the GenHowTo dataset (Souček et al., 2024) is utilized for mixed-modality generation.

# F IMPLEMENTATION DETAILS

We initially conduct joint training of Show-o using the RefinedWeb, a collection of image-text pairs, and the ImageNet-1K for language modeling, image captioning, and class-conditional image generation, respectively, over 500K steps. Subsequently, we replace the class-conditional generation with the training for text-to-image generation using the around 35M image-text pairs for an additional 1,000K steps. The base model is trained on 48 A100 (80GB) GPUs with a total batch size of 1,152. We employ the AdamW optimizer with a weight decay of 0.01, 5,000 steps of warm-up, and an initial learning rate of 1e-4 with a cosine scheduling. Finally, we fine-tune Show-o with around 1M internal high-quality image-text pairs and adhere to the configuration of LLaVA-v1.5 for instruction data tuning. Note that, the current version of Show-o is based on Phi-1.5 (Li et al., 2023). In the following experiment sections, the default Show-o employs discrete image tokens as input for both multimodal understanding and generation. Show-o $^{\dagger}$ and Show-o $^{\ddagger}$ indicate the use of continuous image representations from the pre-trained MAGVIT-v2 and CLIP-ViT (corresponding to options (b) and (c) in Fig. 3), respectively, for multimodal understanding and we discuss this exploration in Section 4.6.

Based on the pre-trained Show-o, we continue to train it on the 2.0B image-text pairs for 500K steps and then we increase the image resolution to $512 \times 512$ and train Show-o for an additional

500K steps. Finally, we fine-tune Show-o with around 1M internal high-quality image-text pairs and adhere to the configuration of LLaVA-v1.5 for instruction data tuning.

# G TEXT PROMPTS

“A 3D render of a futuristic car made of glass, driving through a city of mirrors.”

“A colorful cartoon of a tiger camouflaged in an abstract art painting, its stripes merging with the wild brushstrokes.”

"The image features a stylized stained glass illustration of a hummingbird with vibrant colors, set against a backdrop of swirling patterns and a large sun. The composition includes floral elements and intricate details, creating a vivid and dynamic scene that emphasizes the beauty of the bird. The colors range from greens to reds, enhancing the lively and artistic aesthetic of the piece."

“A 3D render of a surreal explosion scene on the shore of a beautiful white sand beach with crystal clear water. The explosion has a spatter of oil paint with pastel colors and a thick consistency. The explosion is in a quiet and serene environment. A beautiful Japanese woman with a dress compacted to the sea is seen. There are butterfly petals and flowers with an ethereal glow and bioluminescence. There are pink and blue roses, and the overall image has a surreal and dreamlike quality.”

“A hyper-realistic close-up photograph of a woman’s face, focusing on the left side. The image is highly detailed and realistic, showing voluminous glossy lips slightly parted, a well-defined nose, and open eyes with long eyelashes that cast shadows on the skin. The eye color is crystal clear almond green. The skin texture is crisp, with incredible detail of natural, lush skin and pores and freckles, with subtle highlights and shadows that give a realistic, close-up appearance.”

"A 3D render of a cute, round rice ball character named Mochi, with big, sparkling eyes that convey curiosity and joy. Its body is a soft, fluffy white with a slight sheen, resembling freshly cooked rice. Mochi has small, rosy cheeks that give it a warm, friendly expression. A tiny smile brightens its face, and it often sports a colorful ribbon tied around its "waist," adding a playful touch. Mochi's arms and feet are cartoonishly short, allowing it to bounce adorably around its surroundings. This time, Mochi is placed against a background that is a vibrant explosion of colors, with bright hues of fuchsia, turquoise, lemon yellow, and emerald green creating a canvas of vibrant contrasts and playful energy. The clashing colors make Mochi's soft white body and rosy cheeks stand out even more, inviting viewers into a world of cheerful exuberance and visual delight."

# H MORE EXAMPLES OF VIDEO

We provide more examples of video understanding and generation in Fig. 10.

# I ABLATION STUDIES

Impact of Vision Encoder for Multimodal Understanding. The default Show-o employs MAGVIT-v2 to encode images into discrete tokens for both multimodal understanding and generation. Inspired by the literature (Liu et al., 2024b), we investigate the impact of the most popular design choice of vision encoder, i.e., the pre-trained CLIP ViT (Radford et al., 2021), for multimodal understanding. We first compare the two settings using our Show-o model. In Table 5, the comparison between Exp 2 and Exp 4, Exp 3 and Exp 5 clearly demonstrates that continuous representations from CLIP-ViT have significantly better performance on multimodal understanding than that of MAGVIT-v2. This mainly attributes to: i) The CLIP-ViT is pre-trained on a much larger dataset (400M) than that of our pre-trained MAGVIT-v2 (35M); ii) In contrast to image reconstruction learning objective in MAGVIT-v2, the discriminative loss, i.e., image-text matching, in CLIP-ViT makes the extracted representations easier to be adapted for multimodal understanding.

Impact of Various Representations for Multimodal Understanding. In typical multimodal understanding models like LLaVA, the image representation extraction and cross-modal alignment usually happen in the continuous space. However, image tokenizers such as MAGVIT-v2 naturally yield discrete image tokens. As shown in Table 5, we compare the two types of input, i.e., continuous representations and discrete tokens, in the multimodal understanding scenario. In Exp 6

![](images/2392e00d9d5f58d64abbc137b92f86c3833c91048f2119ffd4b9b11f6990d42e.jpg)

![](images/97c2f660b0ff6cbb3a0caaefcc40e13ef1964bf7623b80ccdf94f4107dae6ae0.jpg)

![](images/4ffed8755abcf5f481f6de21c4e631f072e2a35854396e13aa7fef2709d42b62.jpg)

![](images/020bde95277a4ce3620908d10f56deded6ae96050f32e9ca48918a46c70a4b16.jpg)

![](images/fcea59f3df84ee7e8848ef6db63625c87ba01ad4374cf6b148b3fbb4e0735b4c.jpg)  
Describe the video content.

The video features a man in a red shirt who is making a drink by adding various ingredients to a glass. He pours the mixture into a strainer and then adds a lemon slice to the glass. The man then drinks the drink from the glass.

![](images/0296dec4a6b00a1d34ae5cc65f184b6126659f943ff3adfe70a0bce655cc1b71.jpg)

![](images/a39bc7eb470685fdc3270f2f7eee516f3ec44eaa2072a885f381976330567bf9.jpg)

![](images/5b69201e325ba2bb4f6345da2a89ee1f59960660ceb1314a4c8d2e7e189834c8.jpg)

![](images/35cf8c081d2025951d19e2582d5d9ab765af316792f1f8651d728ce39c9f0371.jpg)

![](images/a8f86d7bfec287695a8d548d0f5e456d9d27457014998bcf5c69319fd2bed37b.jpg)

![](images/6655c7d0d574d0b57ee78bf3fc8de4fee10f7f299549122a0c3c1c7a9cf5b7c4.jpg)  
Describe the video content.

The video features a woman who is sitting in front of a computer and talking to the camera. She is wearing a blue shirt and has her hands in her lap. Throughout the video, she is seen making various gestures and speaking to the camera. The background of the video is filled with various items, including a clock, a mirror, and a shelf with clothes. The woman's expressions and gestures vary throughout the video, but they are all focused on her conversation.

![](images/68be63ea8b407c3c757ad933fbcfc9b3f06e1b5d18fdaf9545ce9dcee6e33c79.jpg)

![](images/698c157014a95f9db2172ae8e5f9982fc6e302c465fcc82e5d7299b82de1eb39.jpg)

![](images/d65984c52f63bba63abddc266a498de4694ec25f514a2f0efebdf44f89277669.jpg)

![](images/e4da9d2ddf93fd3dd40529e86a10de91c533f161fa152d2d9d6c7ca3df2d1d13.jpg)

![](images/ad37b73bd40973311e897a62ad4c0ca5f9425731e1216ad8469f0e4f029236c5.jpg)  
Describe the video content.

The video features a woman in a kitchen who is cooking an omelette. She is shown stirring the eggs in a bowl, adding butter and grated cheese, and then flipping the omelette. The video also includes a cartoon character who is shown cooking an omelette.

![](images/938aee8ec270b243d66b77b60f27101e581b7b177c74726c75adbb193a427d8f.jpg)

![](images/bca2d990b8892bae0447f3d80f82b5348a97d4f1c239c6695395791968ab1b53.jpg)

![](images/cfff1c4aeef45c400cf98d4cc6bc3b4b75535314fe35e3bba869d0b35f318545.jpg)

![](images/d45f5d376b0035214d1564406636c68c0365f433d9a2c0975b0e719bf69902fd.jpg)

![](images/ecbda5911afca91027d4f507d44653d0a208ce498b62a6b3e48963ac78527ee2.jpg)

![](images/5cb604b403bf0899df744a9f9dbf01ac2f558afda837b2f3364725add1969975.jpg)  
"A blue car drives past a white picket fence on a sunny day"

![](images/99520725323d833478c2f29b2270de4dd70924473bb41f18544399b711afe6e7.jpg)

![](images/370872019be03fd9c23c5edeadc6434960ef5df56f2eaf3d48c76738a946c151.jpg)

![](images/f6697ed7b6d28d2d6f5707dfa18dfbc990cd790bd51c12acd9f027e34b0ba08a.jpg)

![](images/e9803205e74cd75c1a3a9f0f5e69561479dc00108fbef37728dc25b4b0c833f3.jpg)

![](images/7b81b1f299e0d79d4331bc84ecf3643c015c133e791f4e3223ac465f4cf0681a.jpg)

![](images/70e9a0a089ef8a5d9ddd0265e05e385b3217a3bc7da748bcd1dd00a84f3155c2.jpg)  
"There are two dogs, one brown and one black, playing together in a fenced-in area."   
Figure 10: More examples on video understanding and generation. Only some key frames are sampled for illustration.

Table 5: Ablation studies of various vision encoders and kinds input representations for multimodal understanding. Note that this experiment is based on the Show-o pre-trained on 35M image-text data in a resolution of $256 \times 256$ . 

<table><tr><td># Exp</td><td>Method</td><td>Vision Encoder</td><td>Unified Pretrain</td><td>Feature type</td><td>POPE</td><td>MME</td><td>Flickr30k</td><td>VQAv2(val)</td><td>GQA</td><td>MMMU</td></tr><tr><td>1</td><td>LLaVA</td><td>CLIP-ViT</td><td> $\mathcal{X}$ </td><td>Continuous</td><td>84.1</td><td>1128.0</td><td>69.6</td><td>73.0</td><td>56.5</td><td>30.67</td></tr><tr><td>2</td><td>Show-o $^{\ddagger}$ </td><td>CLIP-ViT</td><td> $\checkmark$ </td><td>Continuous</td><td>84.5</td><td>1182.7</td><td>64.3</td><td>71.9</td><td>57.5</td><td>27.4</td></tr><tr><td>3</td><td>Show-o $^{\ddagger}$ </td><td>CLIP-ViT</td><td> $\mathcal{X}$ </td><td>Continuous</td><td>84.5</td><td>1161.6</td><td>68.5</td><td>73.5</td><td>58.7</td><td>29.2</td></tr><tr><td>4</td><td>Show-o $^{\dagger}$ </td><td>MAGVIT-v2</td><td> $\checkmark$ </td><td>Continuous</td><td>74.3</td><td>947.8</td><td>33.9</td><td>59.4</td><td>51.0</td><td>26.7</td></tr><tr><td>5</td><td>Show-o $^{\dagger}$ </td><td>MAGVIT-v2</td><td> $\mathcal{X}$ </td><td>Continuous</td><td>65.1</td><td>800.0</td><td>12.3</td><td>50.8</td><td>43.9</td><td>24.6</td></tr><tr><td>6</td><td>Show-o</td><td>MAGVIT-v2</td><td> $\checkmark$ </td><td>Discrete</td><td>73.8</td><td>948.4</td><td>36.2</td><td>57.8</td><td>48.7</td><td>25.1</td></tr><tr><td>7</td><td>Show-o</td><td>MAGVIT-v2</td><td> $\mathcal{X}$ </td><td>Discrete</td><td>63.8</td><td>689.1</td><td>4.5</td><td>46.1</td><td>40.5</td><td>28.1</td></tr></table>

and 7, we use the pre-trained MAGVIT-v2 to extract discrete tokens and train an embedding layer to embed the tokens into the continuous embedding space of the LLM. In Exp 4 and 5, we modify MAGVIT-v2 to output continuous representations without quantization. The cross-modal projection layer follows the setting of LLaVA. The comparison between Exp 5 and Exp 7 reveals that discrete

"A colorful cartoon of a tiger camouflaged in an abstract art painting, its stripes merging with the wild brushstrokes."

![](images/08d70de746e6af3a110ccb86245a25f95ec9466ffb3769081596da18601832b3.jpg)

Figure 11: Illustration of generated samples using different sampling steps and classifier-free guidance scale w. This experiment is based on the Show-o in a resolution of $512 \times 512$ .   
(a) Failure Cases in Multimodal Understanding   
![](images/424790af369afaa429a7c045bc3e4c196d7700e3acba9dfece315d8a9e3a3c3a.jpg)

<details>
<summary>text_image</summary>

CLOSING
DOWN!
ALL
STOCK
MUST
GO!
</details>

What are the words in this image?

![](images/b0067e27304a69fcb7ddf2cb5fd2314b247167a2cefa9d794fb1f4fb3ac908bc.jpg)

<details>
<summary>text_image</summary>

SPAGHETTIHOUSE
ROADSTER
</details>

How many people in this picture?

(b) Failure Cases in Visual Generation   
![](images/9129223a6504247ac449040fa34be18030bbb9aab27aad2322a0bda130da8b62.jpg)

<details>
<summary>natural_image</summary>

White coffee mug with 'MREW' branding placed on a surface with scattered coffee beans (no text or symbols on the mug itself)
</details>

"The word 'mardefly' on a coffee mug"

![](images/476157199f827d78d90b4bcd9a77707e333a15cfb27b5813642493daba3eab60.jpg)

<details>
<summary>natural_image</summary>

Group photo of six skiers in colorful gear posing on a snowy slope under clear sky (no text or symbols visible)
</details>

"A group of seven people standing on a snow-covered slope, all wearing skis and posing for a picture."   
Show-o: The words in this image are "closing stock must go down." Show-o: There are three people in the picture.   
Figure 12: Illustration of failure cases of Show-o in multimodal understanding and generation.

tokens show much worse performance on most benchmarks. We attribute the performance gap to that popular multimodal understanding datasets, e.g., LLaVA-Pretrain-558K, are not sufficient to align discrete image tokens into the language space, leading to an unsatisfactory cross-modal understanding. In contrast, continuous representations, already lying in a well-shaped embedding space, are much easier to align.

Impact of Unified Pre-training for Multimodal Understanding. Our training pipeline involves two-stage unified pre-training to learn image token embedding and image-text alignment for multimodal understanding and generation (as described in Section C). Here we elaborate on the impact of the unified per-training with different vision encoders and types of representations:

- CLIP-ViT with Continuous Representations. The comparison between Exp 2 and Exp 3 shows that the unified pre-training has a small negative effect on the CLIP ViT-based understanding, as the performance on most benchmarks has marginal degradations. We hypothesize that the MAGVIT-v2 token-based pre-training and the CLIP ViT-based tuning happen in nearly orthogonal dimensions, and the capability of the backbone has been spared to maintain the compatibility of the two tasks.   
- MAGVIT-v2 with Continuous Representations. In the comparison between Exp 4 and Exp 5, we also notice a performance improvement brought by the unified pre-training, even though the pre-training uses discrete tokens while the experiments here use continuous features. This comparison further validates the hypothesis that unified pre-training enhances the multimodal understanding and reasoning capabilities of the backbone by diverse multimodal interactions during pre-training.   
- MAGVIT-v2 with Discrete Tokens. The comparison between Exp 6 and Exp 7 shows that the unified pre-training has significantly boosted the multimodal understanding performance. This is intuitive since the pre-training also adopts MAGVIT-v2 discrete tokens as image representation. Specifically, we attribute the performance gain to that unified pre-training learns a better cross-modal alignment with large-scale data and enhances the multimodal understanding capabilities of the backbone.

Impact of Sampling Steps. We present generated results at $512 \times 512$ resolution with varying sampling steps on the left of Fig. 11. With just five steps, Show-o can produce an image that is roughly related to the given prompt. Increasing the sampling steps to 25 allows the synthesis of an image that closely adheres to the prompt. When the sampling step is set as 50, the generated image becomes more detailed and realistic. In contrast, auto-regressive models Team (2024); Sun

![](images/82209d2f5daafa6245262bd4410c3808f3c1f5c00c087318d262a1551f5fc15c.jpg)  
Original image

![](images/2734c7b1c3a66ad3703fa593b3715b335762c55440bd33ff1433be25f5771447.jpg)  
A green apple on the desk

![](images/d35ad0464b1d7925235268d579ec7d1ddb90400b229cd252b74929cba583bb90.jpg)  
A small mouse on the desk

![](images/f33cce047b8516d85532e6608ff955298ed20645e27c76fdd3a6f5869085a169.jpg)  
Original image

![](images/41543293bc0d9d1053e6e44fbe7d08253c0d9574b29df358f237606c879113ac.jpg)  
A blue car in the cartoon style

![](images/a0447c28f85a8b7dc6703e671519ce47e544540d742c44a4cf3ded5f15719600.jpg)  
A blue car in the oil painting style   
Figure 13: Mask-free image editing.

et al. (2024) require 1024 sampling steps to generate an image of the same resolution when the downsampling rate is 16, which is around 20 times more steps than our approach.

Impact of Classifier-free Guidance. The visual variations of generated images with different classifier-free guidance scales w are illustrated on the right of Fig. 11. It can be observed that the object in the generated images lacks detail without classifier-free guidance. As the classifier-free guidance scale w is gradually increased to 3 and 5, the colors and contents become more diverse and consistent with the given text prompt.

# J MASK-FREE IMAGE EDITING

Given an image, it can be converted into discrete image tokens, which can then be subject to iterative random masking for sampling purposes. This process allows for image editing without the need for predefined masks. For instance, as illustrated in Fig. 13, Show-o enables local-region modifications such as "changing the red apple to green" or "replacing the apple with a mouse." Furthermore, Show-o facilitates global style adjustments like "transforming the original image into a cartoon or oil painting style."

# K FAILURE CASES

We provide failure cases of Show-o in multimodal understanding and generation in Fig. 12. The current version of Show-o sometimes cannot accurately recognize the text and count the object instances and exhibits challenges in generating correct belongings such as skis for each instance. One can observe that Show-o fails to identify the phrase "closing down" in the left of Fig. 12(a) and is unable to generate the term "mardefly" (as shown left of Fig. 12(b)). This limitation is mainly attributed to the insufficiency of specific data tailored to these scenarios, as our model relies on a limited set of image-text pairs sourced from publicly available datasets and utilizes automatically generated captions. Enriching such kind of data holds promise for addressing these failure modes in Show-o, an aspect that will be explored in the future.