# Learning to Prompt with Text Only Supervision for Vision-Language Models

Muhammad Uzair Khattak $^{1}$ Muhammad Ferjad Naeem $^{2}$ Muzammal Naseer $^{1}$ Luc Van Gool $^{2}$ Federico Tombari $^{3,4}$

$^{1}$ Mohamed bin Zayed University of AI $^{2}$ ETH Zurich $^{3}$ TU Munich $^{4}$ Google

# Abstract

Foundational vision-language models such as CLIP are becoming a new paradigm in vision, due to their excellent generalization abilities. However, adapting these models for downstream tasks while maintaining their generalization remains a challenge. In literature, one branch of methods adapts CLIP by learning prompts using visual information. While effective, most of these works require labeled data which is not practical, and often struggle to generalize towards new datasets due to over-fitting on the source data. An alternative approach resorts to training-free methods by generating class descriptions from large language models (LLMs) and perform prompt ensembling. However, these methods often generate class specific prompts that cannot be transferred to other classes, which incur higher costs by generating LLM descriptions for each class separately. In this work, we propose to combine the strengths of these both streams of methods by learning prompts using only text data derived from LLMs. As supervised training of prompts is not trivial due to absence of images, we develop a training approach that allows prompts to extract rich contextual knowledge from LLM data. Moreover, with LLM contextual data mapped within the learned prompts, it enables zero-shot transfer of prompts to new classes and datasets potentially cutting the LLM prompt engineering cost. To the best of our knowledge, this is the first work that learns generalized prompts using text only data. We perform extensive evaluations on 4 benchmarks where our method improves over prior ensembling works while being competitive to those utilizing labeled images. Our code and pre-trained models are available at https://github.com/muzairkhattak/ProText.

# 1. Introduction

The Vision field is experiencing a new paradigm in its model-building approach with the emergence of foundational models $[18, 23, 37, 47]$ , which are large DNNs pretrained on web-scale data. Among these, Vision-Language models (VLMs) such as CLIP $[37]$ stand out as the latest

<table><tr><td></td><td>Method</td><td>Do not require images</td><td>Transfer to unseen datasets</td></tr><tr><td rowspan="4">Prompt learning methods</td><td>CoOp [50]</td><td>✕</td><td>√</td></tr><tr><td>CoCoOp [49]</td><td>✕</td><td>√</td></tr><tr><td>MaPLe [20]</td><td>✕</td><td>√</td></tr><tr><td>PromptSRC [21]</td><td>✕</td><td>√</td></tr><tr><td rowspan="3">Prompt ensembling methods (LLM)</td><td>DCLIP [29]</td><td>√</td><td>✕</td></tr><tr><td>WaffleCLIP-Concept [39]</td><td>√</td><td>✕</td></tr><tr><td>CuPL [36]</td><td>√</td><td>✕</td></tr><tr><td></td><td>ProText (Ours)</td><td>√</td><td>√</td></tr></table>

Table 1. Existing methods improve CLIP's generalization by learning prompts with image supervision or using non-transferable prompt ensembling with LLM knowledge. In contrast, our approach, ProText, effectively learns prompts with text-only supervision which are transferable to new datasets and classes.

highlights which leverage contrastive pre-training on massive image-text pairs from the internet. During pre-training, CLIP learns to align image-text samples in a shared feature space. This allows CLIP to encode open-vocabulary concepts and generalize well to zero-shot recognition tasks.

CLIP consists of two encoders to encode image and text inputs respectively. At inference, a hand-crafted prompt such as 'a photo of a CLS' is used as the text input. Text features of classes are compared with visual feature and class with highest similarity is assigned as predicted label. Improving the quality of text templates such as adding attributes [1], or class-specific details [19, 36] has shown to improve CLIP performance. However, designing high-quality prompts that can best describe test image remains a key challenge, as image content is not known in advance.

In literature, numerous techniques have been proposed to adapt CLIP for downstream recognition tasks. One branch of methods $[6, 17, 27, 41, 49, 50]$ treat text prompts as learnable vectors and optimize them using task-specific objectives such as cross-entropy. As prompts are learned in the embedding space, this allows them to be used with classes and datasets beyond those on which they were trained on. While effective over the baseline CLIP, most of these methods require annotated image labels to optimize the prompts which is often impractical, especially in real-world scenarios such as medical imaging, remote sensing, security,

![](images/8ae834a57827490f786bb22e886c9c974526c3a9b84cd686f23d86457ad4fab4.jpg)

<details>
<summary>bar</summary>

| Method       | Performance (%) |
| ------------ | --------------- |
| CLIP         | 65.15           |
| CuPL         | 65.15           |
| CoOp         | 63.88           |
| CoCoOp       | 65.74           |
| PromptSRC    | 65.81           |
| MaPLe        | 66.3            |
| ProText (Ours)| 67.23           |
</details>

Figure 1. Without using any images for supervision, ProText with text-only training improves over CLIP, CuPL, and prior 16-shot image-supervised methods in challenging cross-dataset transfer settings. Prompt ensembling based CuPL performs same as CLIP as it cannot transfer class specific LLM templates to cross-datasets.

surveillance, etc. Moreover, these methods tend to overfit on few-shot source samples and struggle to retain CLIP's generalization, especially in cross-dataset settings.

Alternatively, several methods $[29, 36]$ have adopted the training-free approach of prompt ensembling by leveraging the capabilities of Large Language Models (LLMs). Instead of using hand-crafted templates, these methods mine dataset or class specific descriptors and captions from LLMs to enrich text features. These enriched features aim to better represent content that could possibly occur in test images, leading to improvements over baseline CLIP. Although these methods do not require image information, the knowledge acquired from LLMs is mostly specific to each class and not directly transferable across unseen classes and datasets since no optimization is performed. Additionally, generating LLM descriptions for each concept separately incurs additional LLM serving and prompt engineering costs.

In this work, we present a new paradigm to improve CLIP's generalization. Our motivation comes from combining the strengths of prompt learning and prompt ensembling approaches while effectively addressing their limitations. To this end, we introduce ProText: Prompt Learning with Text-Only Supervision. In contrast to previous methods, our approach instead proposes to learn prompts using text only data obtained from LLMs. As supervised training of prompts is not trivial due to image-free setting, we develop a novel training framework that allows prompts to learn and extract rich contextual knowledge from LLM data. Moreover, as LLM contextual knowledge is mapped within the learned prompts, it enables zero-shot transfer of prompts to new classes and datasets, potentially leading to a substantial reduction in LLM serving and prompt engineering cost.

As shown in Tab. 1, our approach is different from prior methods as it does not require image samples to learn prompts, in addition the adapted CLIP transfers well to unseen classes and datasets, therefore addressing a key limitation of LLM-based prompt ensembling techniques. We demonstrate the effectiveness of ProText by performing extensive evaluations on 4 benchmarks. On challenging crossdataset transfer setting, ProText without using any visual information achieves an average gain of +2.08% over CLIP while surpassing the performance of previous best image-supervised prompt learning method MaPLe [20] by +0.93% (Fig. 1). Further, ProText with text-only supervision performs competitively against prior methods in domain generalization, base-to-novel class, and text-only supervised setting. Our main contributions are summarized as follows:

- We present a new approach for prompt learning in CLIP using text-only supervision. Our method harmonically combines the strengths of prompt learning and prompt ensembling methods to improve CLIP's generalization.   
- To optimize prompts with text-only data, we develop a training approach that allows prompts to learn a mapping by extracting rich contextual information from LLM data.   
- As LLM contextual knowledge is mapped within the learned prompts, this enables prompts to be directly used with new classes and datasets potentially cutting the additional LLM serving and prompt engineering cost.   
- We validate the effectiveness of our method through extensive experiments across four benchmarks. Our TextPro approach improves the generalization of CLIP across various settings and fares competitive to approaches that explicitly use labeled image samples for training.

# 2. Related Work

Foundational Vision-Language models (VLMs). VLMs [18, 33, 37, 46–48] leverage joint image-text pretraining using internet-scale data in a self-supervised fashion. Representative VLMs like CLIP [37] and ALIGN [18] have utilized around 400M and 1B image-text pairs during their pre-training. Using the contrastive learning objective, VLMs learn rich multi-modal features by attracting together the features of paired images and texts while repelling un-paired image-text features in a joint feature space. The resulting model learns open-vocabulary concepts interpretable through natural language suitable for various downstream discriminative vision tasks such as open-vocabulary image classification [6, 20, 27, 31, 32, 50], detection [3, 10, 26, 30, 51], and segmentation [13, 24, 25]. Although promising, adapting VLMs effectively while maintaining their original generalization remains a crucial challenge. In this work, we propose a novel method to adapt CLIP with prompt learning through text modality supervision to improve its performance on vision modality tasks.

Prompt Learning for VLMs. Prompt Learning [6, 9, 27, 40, 41, 49, 50] has emerged as an effective fine-tuning strategy to adapt large-scale models. This approach adds a small number of learnable embeddings along with model inputs which are optimized during training while the rest of the model is kept frozen. As the pre-trained model is unchanged during prompt learning, it has become particularly effective for VLMs such as CLIP, where maintaining the model's

original generalizability is crucial. CoOp [50] is the pioneering prompt learning method for CLIP which learns text prompt embeddings to fine-tune CLIP. CoCoOp [49] improves CoOp's generalization by conditioning text prompts on visual features. MaPLe [20] proposes a multi-modal prompting framework to adapt both vision and language branches of CLIP. UPL [17] adopts an unsupervised prompt learning approach to finetune CLIP. PromptSRC [21] improves prompt learning from a regularization perspective by making use of additional loss functions during training. While these methods improve baseline CLIP performance, most of them require image samples with labels, which is less practical, and generating pseudo-labels is often less effective. In contrast, we present a novel prompt learning approach that improves CLIP generalization without relying on any visual samples during training.

Training-Free Text Prompt Enhancement. With the emergence of LLMs such as GPT-3 $[5]$ , several approaches $[29, 36, 39]$ have demonstrated their potential for improving zero-shot generalization of CLIP. Instead of using handcrafted templates for generating class features, these methods leverage LLMs to generate high-level concepts, class descriptions, and/or attributes which are used in one form or another to produce enriched text features. DCLIP $[29]$ generates fine-grained per-class language descriptors and ensemble its similarity with image to produce classification scores. WaffleCLIP $[39]$ matches DCLIP performance with random descriptors and show further gains by data-specific concepts generated via LLMs. CuPL $[36]$ query LLMs to generate class-specific prompt descriptions for text prompt ensembling. Although effective, most of these approaches generate class-specific text data from LLMs which are not directly transferable to unseen classes and new datasets since no training is performed. On the other hand, we aim to leverage the same LLM data via novel text-only prompt learning technique which seamlessly allows the transfer of learned prompts toward unseen classes and new datasets.

# 3. Method

Given the language interpretable nature of foundational VLMs such as CLIP [37], they are naturally suited for zero-shot recognition tasks. However, to achieve full potential of CLIP's generalization for downstream tasks, adaptation still appears to be necessary. Numerous approaches have since been proposed to adapt general knowledge of CLIP for user-specific downstream tasks. One line of methods adopts prompt learning [20, 27, 49, 50] to re-purpose CLIP features for downstream data. While effective, most of them require image samples with labels to learn the prompts, which is a hard requirement to meet. Another line of methods adopts training-free prompt ensembling techniques [29, 36, 39] with the help of LLMs. Although ensembling-based approaches do not require image information, the majority of these works generate class-specific LLM prompts that are not directly transferable to new classes and datasets.

To this end, we present a new paradigm for learning generalized transferable prompts for VLMs using text-only supervision. Our proposed adaptation framework, ProText: Prompt Learning with Text only supervision aims to address the challenges of existing approaches by learning transferable prompts without relying on images. Fig. 2 shows our ProText framework. First, we curate text-only LLM template data using class names of a given dataset and a LLM such as GPT-3 [5]. As a text-supervised approach, ProText only requires CLIP text encoders during training. Specifically, we employ one frozen encoder with learnable prompts and a second frozen encoder without learnable prompts. Learnable prompts with class-name templates are input to the prompted text encoder to obtain the class-name template feature, and a frozen text encoder generates LLM template feature from its description obtained from LLM data. Next, we employ a contextual mapping training objective which maps class-name template feature to the LLM template feature. Contextual mapping allows the prompts to learn a mapping function that embeds rich contextual knowledge from LLM data within the prompt vectors. As prompts are learned in the embedding space, they are directly compatible with new classes and datasets. At inference, the learned prompts are shipped with CLIP model for standard zero-shot CLIP inference for visual recognition.

Below we explain our proposed approach in detail. We first revisit CLIP and previous methods including Prompt Learning and Prompt Ensembling via LLMs in Sec. 3.1 and then we present our ProText approach in Sec. 3.2.

# 3.1. Preliminaries

Contrastive Language-Image Pre-training (CLIP). CLIP consist of an image encoder f and a text encoder g which maps image and text input into visual and textual feature respectively. We denote CLIP parameters as $\theta_{CLIP} = \{\theta_f, \theta_g\}$ where $\theta_f$ and $\theta_g$ refer to the image and text encoder parameters, respectively. Input image X is divided into M patches which are linearly projected to produce patch tokens and a learnable class token CLS is prepended resulting in the final sequence as $\tilde{X} = \{CLS, e_1, e_2, \cdots, e_M\}$ . The image encoder f encodes the input patches via multiple transformer blocks to produce a latent visual feature representation $\tilde{f} = f(\tilde{X}, \theta_f)$ , where $\tilde{f} \in R^d$ . Next, the corresponding class label y is embedded in a text template, such as ‘a photo of a [CLASS]’ which can be formulated as $\tilde{Y} = \{SOS, t_1, t_2, \cdots, t_L, c_k, EOS\}$ . Here $\{t_l|_{l=1}^L\}$ and $c_k$ are the word embeddings corresponding to the text template and the label y, respectively while SOS and EOS are the learnable start and end token embeddings. The text encoder g encodes $\tilde{Y}$ via multiple transformer blocks to produce the latent text feature as $\tilde{g} = g(\tilde{Y}, \theta_g)$ , where $\tilde{g} \in R^d$ .

![](images/c1c27007216db05ea2345c7ad8b877aadb2adc4c93bb492798ccde45a3d66874.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Class name template"] --> B["a photo of a persian cat"]
    C["Text prompts"] --> D["concat"]
    E["Frozen parameters"] --> F["Text Encoder"]
    G["Learnable parameters"] --> H["Text Encoder"]
    I["Class name feature"] --> J["concat"]
    K["LLM template feature"] --> L["concat"]
    M["Frozen parameters"] --> N["Text Encoder"]
    O["Learnable parameters"] --> P["Text Encoder"]
    Q["Text Encoder"] --> R["Contextual Mapping"]
    S["Text Encoder"] --> T["Contextual Mapping"]
    U["Text Encoder"] --> V["Contextual Mapping"]
    W["Text Encoder"] --> X["Contextual Mapping"]
    Y["Text Encoder"] --> Z["Contextual Mapping"]
    AA["Text Encoder"] --> AB["Contextual Mapping"]
    AC["Text Encoder"] --> AD["Contextual Mapping"]
    AE["Text Encoder"] --> AF["Contextual Mapping"]
    AG["Text Encoder"] --> AH["Contextual Mapping"]
    AI["Text Encoder"] --> AJ["Contextual Mapping"]
    AK["Text Encoder"] --> AL["Contextual Mapping"]
    AM["Text Encoder"] --> AN["Contextual Mapping"]
    AO["Text Encoder"] --> AP["Contextual Mapping"]
    AQ["Text Encoder"] --> AR["Contextual Mapping"]
    AS["Text Encoder"] --> AT["Contextual Mapping"]
    AU["Text Encoder"] --> AV["Contextual Mapping"]
    AW["Text Encoder"] --> AX["Contextual Mapping"]
    AY["LLM GPT-3"] --> AZ["How does a persian cat look like?"]
    AZ --> BA["LLM Data"]
    BA --> BB["A persian cat is a large, long-haired cat with a broad face and round eyes."]
    BB --> BC["Text Encoder"]
    BC --> BD["Contextual Mapping"]
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
    BE --> BE
```
</details>

![](images/8aa63a96b4dfc68a722a921e08f67407b230a95a8024094710d52506653e18e6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["a photo of a persian cat"] --> B["concat"]
    C["Class names"] --> B
    B --> D["Text Encoder"]
    D --> E["Image Encoder"]
    E --> F["Image Encoder"]
    G["Cat Image"] --> H["Image Encoder"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style G fill:#ccf,stroke:#333
    style B fill:#cff,stroke:#333
    style D fill:#ffc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
```
</details>

Figure 2. Overview of ProText framework. (Left) First, diverse captions are generated for training classes using LLM like GPT-3. During training, CLIP text encoders generate prompted class-name feature ( $\tilde{g}_{p}$ ) from class-name templates with learnable prompts and frozen LLM template feature ( $\tilde{g}$ ) from LLM generated templates. Next, we employ contextual mapping loss to guide learnable prompts to learn a mapping from the prompted class-name feature to the LLM template feature containing more information about the class. This allows the learned prompts to exploit internal knowledge of text encoder complemented by LLM descriptions. (Right) At inference, learned prompts are used with class-name templates, and the standard zero-shot CLIP inference protocol is followed. Moreover, rich contextual information from LLM descriptions mapped within the learned prompts enables its transferability to new classes and datasets.

For zero-shot inference, text features of text template with class labels $\{1,2,\cdots,C\}$ are matched with image feature $\tilde{f}$ as $\frac{\exp(\text{sim}(\tilde{g}\cdot\tilde{f})\tau)}{\sum_{i=1}^{C}\exp(\text{sim}(\tilde{g}_{i}\cdot\tilde{f})\tau)}$ , where $\text{sim}()$ denotes the cosine similarity and $\tau$ is the temperature.

Prompt Learning with CLIP. Being a parameter efficient tuning method, prompt learning has emerged as a popular technique to adapt vision-language models like CLIP. Since most of the model is kept frozen during adaptation, prompt learning aims to reduce overfitting. Learnable prompts are appended either at the image side $[2]$ , text encoder side $[49, 50]$ , or both sides. In this work, we learn hierarchical prompts at the text encoder named Deep Language Prompting (DLP) $[20]$ formulated as follows.

T learnable language prompts $P_{t} = \{p_{t}^{1}, p_{t}^{2}, \cdots, p_{t}^{T}\}$ are appended with text input tokens, resulting in $Y_{p} = \{SOS, P_{t}, t_{1}, t_{2}, \cdots, t_{L}, c_{k}, EOS\}$ . The text encoder processes $\tilde{Y}_{p}$ and prompted text feature is obtained as $\tilde{g}_{p} = g(\tilde{Y}_{p}, \theta_{g})$ . We use deep prompting which learns hierarchical prompts at subsequent transformer blocks of text encoder. Visual feature $\tilde{f}$ is obtained without utilizing learnable prompts. To adapt CLIP on image classification task on dataset D, prompts $P_{t}$ are optimized in a supervised fashion using labeled image samples with cross-entropy loss, $L_{CE}$ .

$$
\mathcal {L} _ {\mathrm{CE}} = \arg \min _ {\boldsymbol {P} _ {t}} \mathbb {E} _ {(\boldsymbol {X}, y) \sim \mathcal {D}} \mathcal {L} (\text { sim } (\tilde {\boldsymbol {f}}, \tilde {\boldsymbol {g}} _ {\boldsymbol {p}}), y). \tag {1}
$$

Prompt Ensembling with LLM descriptions. Several methods have recently proposed to adapt CLIP via training-free prompt ensembling techniques. The majority of these approaches leverage the capabilities of LLMs to mine rich descriptions, attributes, or high-level concepts of class names. The corresponding text features are either averaged

[36] or the similarity score of each attribute with the image is calculated to obtain classification scores [39] [29].

In this work, we focus our comparison with a strong ensembling baseline CuPL [36]. Specifically, a Large Language Model F such as GPT-3 [5] is used to generate class-specific descriptions for class labels $\{1,2,\cdots,C\}$ using queries such as 'How does a CLASS look like'. Text features of the same class description are averaged together, which serves as the ensembled text features. Finally, zero-shot inference is performed with those ensembled text features.

# 3.2. Prompt Learning with Text-Only Supervision

While image-supervised prompt learning and LLM-based prompt ensembling methods have proven effective in adapting CLIP, they face notable challenges as outlined below.

Visual data dependency. Existing prompt learning methods require visual samples with labels to optimize prompts using Eq. 1. However, collecting samples and labels is difficult in critical scenarios like medical images, remote sensing, and surveillance. Pseudo-labels alleviate label dependency but they are often less effective. Furthermore, these methods tend to overfit CLIP to source data distributions and compromise generalization across cross-datasets. For instance, CoOp utilizing labeled source samples reduces average CLIP performance by 1.27% on 10 cross-datasets.

LLM Prompts transferability limitation. LLM-based prompt ensembling approaches like CuPL $[36]$ generate class-specific LLM descriptions that cannot be directly transferred to unseen classes and datasets. While open-source LLMs exhibit lower performance, proprietary ones such as GPT-3 are required for generating data for new classes and datasets leading to additional serving costs.

Our work aims to address the aforementioned limitations within a unified framework. Below we detail our strategy for curating text-to-text data via LLMs for training, followed by our text-only prompt learning framework.

# 3.2.1 Text-Only LLM data for Prompt Learning

As discussed in Sec. 3.1, optimizing prompts for downstream datasets typically requires image-labels pairs. Since we explicitly aim to bypass this requirement, we first leverage LLMs to curate text data for prompt learning which consists of text inputs and text outputs. Given a set of classes $\{c_i\}_{i=1}^C$ , we prepare text inputs $\{L_{inputs}^i\}_{i=1}^C$ by wrapping each class name in a standard hand-written text template,

$$
L _ {\text { inputs }} ^ {i} = \text { `a   photo   of   a   } c _ {i}.
$$

Next, we prepare text outputs corresponding to the $L_{inputs}$ . Specifically, we query GPT-3 model to generate detailed descriptions for each class name $c_{i}$ . Similar to CuPL [36], we prompt GPT-3 with different queries Q conditioned on class names such as 'How does a $c_{i}$ look like?' and 'How can you identify a $c_{i}$ ?' to obtain text outputs,

$$
L _ {\mathrm{outputs}} ^ {i} = \mathcal {F} (Q | c _ {i}).
$$

Similar to [36], we generate M text outputs per query Q and use N different queries, resulting in $M \times N$ text outputs per class category. We associate all $L_{inputs}$ with the corresponding single $L_{inputs}$ for each class $c_{i}$ . As LLMs are pre-trained on internet-scale text corpora, they possess the capability of generating very diverse and high-quality descriptions and captions for different class categories which results in high-quality text outputs. Finally we combine $L_{inputs}$ and $L_{outputs}$ to create LLM based text-to-text data for text only prompt learning, $D_{PROMPT} = \{L_{inputs}^{i}, L_{outputs}^{i}\}_{i=1}^{M \times N \times C}$ . We refer the readers to supplementary for additional details on the choice of LLM prompts and examples of $D_{PROMPT}$ .

# 3.2.2 Contextual mapping with Prompt Learning

To leverage LLM text-to-text data $D_{PROMPT}$ for learning generalized transferable prompts, we propose a contextual mapping strategy that effectively learns a mapping function that maps standard class name templates such as ‘a photo of a $c_{i}$ ’ to the text feature generated from a LLM description which contains more information about the class $c_{i}$ . In other words, contextual mapping allows learnable prompts to map $L_{inputs}$ to $L_{outputs}$ in the text feature space of CLIP. The mapping function is realized in the form of learnable prompt vectors, which we found to be more effective in our ablations as compared to other techniques such as adapters via linear projection and MLP. For an $i_{th}$ training sample from $D_{PROMPT}$ consisting of a text-to-text pair $\{L_{inputs}, L_{outputs}\}_{i}$ , we obtain prompted class-name feature $\tilde{g}_{p}$ for $L_{inputs}^{i}$ using learnable prompts and frozen LLM feature $\tilde{g}$ for $L_{outputs}^{i}$ without the prompt vectors within the pre-trained latent space of CLIP. We then impose a contextual mapping constraint between $\tilde{g}_{p}$ and $\tilde{g}$ text features as follows,

$$
\mathcal {L} _ {\text { mapping }} = \frac {1}{d} \sum_ {i = 1} ^ {d} | | \tilde {\boldsymbol {g}} _ {\boldsymbol {p}} - \tilde {\boldsymbol {g}} | | _ {2} ^ {2}. \tag {2}
$$

As shown above, we utilize MSE loss objective to enforce contextual mapping from $L_{inputs}^{i}$ to $L_{outputs}^{i}$ . We study other choices of consistency objectives in our ablations (Sec. 4.7).

Motivation for $L_{mapping}$ . Contextual mapping objective allows learnable prompts to exploit internal knowledge of text encoder of CLIP to generate rich contextual features aligned with the LLM descriptions ( $L_{outputs}^{i}$ ) for a given class. This strategy effectively learns prompts without using any visual information and when trained using all training classes together, it enables prompts to capture versatile and generalized context from the LLM descriptions. These context-aware prompts become adaptable for use with any dataset and effectively enable the transferability of class-specific LLM descriptions to unseen classes and datasets. Consequently, this substantially reduces the per-dataset overhead associated with LLM serving and prompt engineering.

Inference. Once text prompt vectors are optimized through our TextPro framework in the text domain, they become ready to be shipped with CLIP for downstream visual domain inference with a standard zero-shot CLIP inference setup. As shown in Fig. 2 (right), the learned prompts $P_{t}$ are fused with each given class name to produce prompted text features $\{\tilde{g}_{p}\}_{i=1}^{C}$ . Finally, zero-shot inference is performed with the prompted text features and the input image feature $\tilde{f}$ to produce classification scores on test images.

# 4. Experiments

# 4.1. Evaluation settings

We perform evaluations in 4 benchmark settings. Prompt ensembling methods and ProText utilize text-only LLM data for adapting CLIP while image-supervised prompt learning methods use image-label pairs for training.

Base-to-Novel Generalization. This setting evaluates the generalization of methods within a dataset. Following previous methods $[49, 50]$ , we split each dataset into base and novel classes. Models are trained on base classes and evaluated on the test set of base and novel classes respectively.

Cross-dataset transfer. This setting evaluates the generalization ability of models trained on ImageNet-1k [8] source dataset by directly transferring it on cross-datasets.

Domain Generalization. We evaluate the robustness of different methods on out-of-distribution datasets. We train

<table><tr><td>Method</td><td>ImageNet Acc.</td></tr><tr><td>1: CLIP (ICML&#x27;21)</td><td>66.72</td></tr><tr><td>2: CLIP-Attribute</td><td>67.60</td></tr><tr><td>3: CLIP-80</td><td>68.32</td></tr><tr><td>4: DCLIP (ICLR&#x27;23)</td><td>68.03</td></tr><tr><td>5: Waffle CLIP (ICCV&#x27;23)</td><td>68.34</td></tr><tr><td>6: CuPL (ICCV&#x27;23)</td><td>69.62</td></tr><tr><td>7: ProText-Attribute</td><td>68.05</td></tr><tr><td>8: ProText-80</td><td>68.48</td></tr><tr><td>9: ProText-CuPL</td><td>70.22</td></tr></table>

Table 2. With the same amount of text data, learning contextual prompts with text-only supervision improves CLIP performance in comparison to the prompt ensembling techniques.

<table><tr><td rowspan="2">Dataset</td><td colspan="3">CLIP [37]</td><td colspan="3">CuPL [50]</td><td colspan="3">ProText (Ours)</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>ImageNet</td><td>72.43</td><td>68.14</td><td>70.22</td><td>74.30</td><td>68.14</td><td>71.09</td><td>75.00</td><td>71.38</td><td>73.14</td></tr><tr><td>Caltech101</td><td>96.84</td><td>94.00</td><td>95.40</td><td>97.22</td><td>94.00</td><td>95.58</td><td>98.06</td><td>95.63</td><td>96.83</td></tr><tr><td>OxfordPets</td><td>91.17</td><td>97.26</td><td>94.12</td><td>94.42</td><td>97.26</td><td>95.82</td><td>94.95</td><td>98.00</td><td>96.45</td></tr><tr><td>StanfordCars</td><td>63.37</td><td>74.89</td><td>68.65</td><td>63.54</td><td>74.89</td><td>68.75</td><td>64.54</td><td>76.08</td><td>69.84</td></tr><tr><td>Flowers102</td><td>72.08</td><td>77.80</td><td>74.83</td><td>74.36</td><td>77.80</td><td>76.04</td><td>74.36</td><td>78.44</td><td>76.35</td></tr><tr><td>Food101</td><td>90.10</td><td>91.22</td><td>90.66</td><td>89.93</td><td>91.22</td><td>90.57</td><td>90.20</td><td>91.98</td><td>91.08</td></tr><tr><td>Aircraft</td><td>27.19</td><td>36.29</td><td>31.09</td><td>30.61</td><td>36.29</td><td>33.21</td><td>30.91</td><td>34.13</td><td>32.44</td></tr><tr><td>SUN397</td><td>69.36</td><td>75.35</td><td>72.23</td><td>76.02</td><td>75.35</td><td>75.68</td><td>76.14</td><td>79.14</td><td>77.61</td></tr><tr><td>DTD</td><td>53.24</td><td>59.90</td><td>56.37</td><td>62.85</td><td>59.90</td><td>61.34</td><td>63.08</td><td>61.59</td><td>62.33</td></tr><tr><td>EuroSAT</td><td>56.48</td><td>64.05</td><td>60.03</td><td>59.64</td><td>64.05</td><td>61.77</td><td>59.71</td><td>80.97</td><td>68.73</td></tr><tr><td>UCF101</td><td>70.53</td><td>77.50</td><td>73.85</td><td>75.28</td><td>77.50</td><td>76.37</td><td>75.54</td><td>79.50</td><td>77.47</td></tr><tr><td>Average</td><td>69.34</td><td>74.22</td><td>71.70</td><td>72.56</td><td>74.22</td><td>73.38</td><td>72.95</td><td>76.98</td><td>74.91</td></tr></table>

Table 3. Base-to-novel setting. ProText enables the transferability of learned prompts to new classes and improves over CuPL [36].

models on the ImageNet-1k source dataset and evaluate its performance on four ImageNet variants with domain shifts. Supervised setting. We provide performance comparison of ProText with CuPL[36] with text-only data per dataset.

Datasets. For the aforementioned benchmarks, we use same datasets as followed by previous works $[20, 21, 49, 50]$ . For cross-dataset transfer, domain generalization, and base-to-novel generalization settings, we use 11 image datasets that cover multiple recognition tasks. These include ImageNet $[8]$ and Caltech101 $[11]$ which contains generic objects; OxfordPets $[35]$ , StanfordCars $[22]$ , Flowers102 $[34]$ , Food101 $[4]$ , and FGVCAircraft $[28]$ for fine-grained classification, SUN397 $[45]$ for scene recognition, UCF101 $[42]$ for action recognition, DTD $[7]$ for texture classification, and EuroSAT $[14]$ for satellite images categorization. For domain generalization setting, we train models on ImageNet $[8]$ as a source dataset and use ImageNet-A $[16]$ , ImageNet-R $[15]$ , ImageNet-Sketch $[44]$ and ImageNetV2 $[38]$ for out of distribution dataset evaluation.

Implementation details. We use a publicly available pretrained ViT-B/16 CLIP model from OpenAI $[37]$ . We train ProText with Deep Language Prompting in the first 9 transformer blocks of the CLIP text encoder. For cross-dataset transfer and domain generalization setting, we train ProText using T = 4 and T = 16 language prompts with 10 and 200 epochs respectively. Similar to $[44]$ , ProText and zero-shot CLIP use additional concepts where available with its prompts such as ‘a photo of a CLS, a type of flower’ for OxfordFlowers $[34]$ . For base-to-novel and supervised text-only settings, ProText uses optimal prompt length and epoch configuration for each dataset. Optimal training configuration is obtained through hyper-parameter search on validation split of datasets. To generate text-only data, we utilize GPT-3 DaVinci-002 model $[5]$ and generate class-specific descriptions using the LLM prompts provided by CuPL $[36]$ . We use publicly available CuPL data and generate descriptions for datasets not provided by CuPL. AdamW optimizer is used with 5 warm-up epochs for training. We use a single 16-GB V100 to train our models. Refer to supplementary material for additional implementation details.

# 4.2. Effectiveness of Text-Only Supervision

We first present an ablation to motivate our approach of learning prompts with text-only supervision. We train Pro-Text with 3 types of text data and evaluate performance on ImageNet-1k [8]. ProText-Attribute uses 46 templates from [1] which corresponds to common image attributes such as rotation, blurriness, etc. ProText-80 is trained on standard 80 templates provided by CLIP [37] and ProText-CuPL is trained on class-specific LLM data employed by our main baseline CuPL [36] for its ensembling approach.

In Tab. 2, we compare ProText with CLIP and recent LLM-based ensembling methods. Prompt ensembling with attribute templates and 80 templates improves over CLIP single template result. Among the LLM-based ensembling methods, CuPL provide highest performance of 69.62%. In contrast, ProText uses a learning-based approach and shows competitive performance against prompt ensembling methods using the same text data. ProText-Attribute provides gain of 0.45% over CLIP-Attribute while roughly maintaining its performance against CLIP-80. When equipped with CuPL LLM text-data, ProText surpasses CuPL by 0.60% leading to highest performance against all methods. These results motivate our approach that instead of prompt ensembling, one can achieve competitive results by utilizing the same available text data to learn prompts. Next, we demonstrate the generalization of ProText such that the learned prompts transfer well across new classes and datasets.

# 4.3. Base to novel class generalization

We now present results in base-to-novel class generalization setting where training data for only base classes are available and the model is evaluated on both base and novel classes. For CuPL $[36]$ , we use base-class LLM templates for base classes and zero-shot CLIP results for its novel classes. For ProText, we use base-class LLM templates for training and transfer the learned prompts for novel classes.

Results are shown in Tab. 3. CuPL outperforms zero-shot CLIP on base classes while maintaining its performance on novel classes as LLM prompts for new classes are not available. ProText shows consistent improvements over CuPL on base classes for 11 datasets. Furthermore,

<table><tr><td rowspan="2"></td><td>Source</td><td colspan="11">Target</td></tr><tr><td>ImageNet</td><td>Caltech101</td><td>OxfordPets</td><td>StanfordCars</td><td>Flowers102</td><td>Food101</td><td>Aircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td><td>Average</td></tr><tr><td colspan="13">Methods utilizing labeled visual samples</td></tr><tr><td>CoOp</td><td>71.51</td><td>93.70</td><td>89.14</td><td>64.51</td><td>68.71</td><td>85.30</td><td>18.47</td><td>64.15</td><td>41.92</td><td>46.39</td><td>66.55</td><td>63.88</td></tr><tr><td>Co-CoOp</td><td>71.02</td><td>94.43</td><td>90.14</td><td>65.32</td><td>71.88</td><td>86.06</td><td>22.94</td><td>67.36</td><td>45.73</td><td>45.37</td><td>68.21</td><td>65.74</td></tr><tr><td>MaPLe</td><td>70.72</td><td>93.53</td><td>90.49</td><td>65.57</td><td>72.23</td><td>86.20</td><td>24.74</td><td>67.01</td><td>46.49</td><td>48.06</td><td>68.69</td><td>66.30</td></tr><tr><td>PromptSRC</td><td>71.27</td><td>93.60</td><td>90.25</td><td>65.70</td><td>70.25</td><td>86.15</td><td>23.90</td><td>67.10</td><td>46.87</td><td>45.50</td><td>68.75</td><td>65.81</td></tr><tr><td colspan="13">Zero-shot &amp; Prompt ensembling methods</td></tr><tr><td>CLIP</td><td>66.72</td><td>92.98</td><td>89.13</td><td>65.29</td><td>71.30</td><td>86.11</td><td>24.90</td><td>62.59</td><td>44.56</td><td>47.84</td><td>66.83</td><td>65.15</td></tr><tr><td>CuPL</td><td>69.62</td><td>92.98</td><td>89.13</td><td>65.29</td><td>71.30</td><td>86.11</td><td>24.90</td><td>62.59</td><td>44.56</td><td>47.84</td><td>66.83</td><td>65.15</td></tr><tr><td colspan="13">Prompt learning with text-only supervision</td></tr><tr><td>ProText (Ours)</td><td>69.80</td><td>94.81</td><td>91.01</td><td>66.00</td><td>72.35</td><td>86.66</td><td>24.72</td><td>67.34</td><td>47.93</td><td>51.86</td><td>69.60</td><td>67.23</td></tr></table>

Table 4. Cross-dataset transfer setting. CuPL and CLIP perform same for cross-datasets as CuPL source data cannot transfer to cross-datasets. Image-based models are trained on 16-shot ImageNet samples. ProText employ same ImageNet data as CuPL for prompt learning.

<table><tr><td rowspan="2"></td><td>Source</td><td colspan="5">Target</td></tr><tr><td>ImageNet</td><td>-V2</td><td>-S</td><td>-A</td><td>-R</td><td>Avg.</td></tr><tr><td colspan="7">Methods utilizing labeled visual samples</td></tr><tr><td>CoOp</td><td>71.51</td><td>64.20</td><td>47.99</td><td>49.71</td><td>75.21</td><td>59.28</td></tr><tr><td>CoCoOp</td><td>71.02</td><td>64.07</td><td>48.75</td><td>50.63</td><td>76.18</td><td>59.91</td></tr><tr><td>MaPLe</td><td>70.72</td><td>64.07</td><td>49.15</td><td>50.90</td><td>76.98</td><td>60.27</td></tr><tr><td colspan="7">Zero-shot &amp; Prompt ensembling methods</td></tr><tr><td>CLIP</td><td>66.72</td><td>60.83</td><td>46.15</td><td>47.77</td><td>73.96</td><td>57.18</td></tr><tr><td>CuPL</td><td>69.62</td><td>63.27</td><td>49.02</td><td>50.72</td><td>77.05</td><td>60.01</td></tr><tr><td colspan="7">Prompt learning with text-only supervision</td></tr><tr><td>ProText (Ours)</td><td>70.22</td><td>63.54</td><td>49.45</td><td>51.47</td><td>77.35</td><td>60.45</td></tr></table>

Table 5. Domain generalization. Prompt learning methods are trained on imageNet and evaluated on datasets with domain shifts.

<table><tr><td>Dataset</td><td>CLIP</td><td>CuPL</td><td>ProText</td><td>Δ</td></tr><tr><td>ImageNet</td><td>66.72</td><td>69.60</td><td>70.22</td><td>+0.62</td></tr><tr><td>Caltech101</td><td>92.98</td><td>94.32</td><td>95.29</td><td>+0.97</td></tr><tr><td>DTD</td><td>44.56</td><td>53.96</td><td>54.02</td><td>+0.06</td></tr><tr><td>EuroSAT</td><td>47.84</td><td>60.27</td><td>58.53</td><td>-1.74</td></tr><tr><td>StanfordCars</td><td>65.29</td><td>65.95</td><td>66.77</td><td>+0.82</td></tr><tr><td>Flowers102</td><td>71.30</td><td>73.85</td><td>74.42</td><td>+0.57</td></tr><tr><td>Aircraft</td><td>24.90</td><td>27.66</td><td>29.01</td><td>+1.35</td></tr><tr><td>SUN397</td><td>62.59</td><td>69.00</td><td>69.76</td><td>+0.76</td></tr><tr><td>OxfordPets</td><td>89.13</td><td>91.11</td><td>92.72</td><td>+1.61</td></tr><tr><td>UCF101</td><td>66.83</td><td>70.63</td><td>71.45</td><td>+0.82</td></tr><tr><td>Food101</td><td>86.11</td><td>86.11</td><td>86.68</td><td>+0.57</td></tr><tr><td>Average</td><td>65.15</td><td>69.31</td><td>69.90</td><td>+0.59</td></tr></table>

Table 6. Pro-Text results with text supervision on each dataset. We compare ProText with CLIP and CuPL. Gains of ProText over CuPL are shown in blue.

with the same LLM base-class data as CuPL, ProText effectively transfers learned prompts towards novel classes and improves CLIP and CuPL novel class performance by 2.76% averaged across 11 datasets. This shows the advantage of ProText prompts to benefit unseen class performance potentially reducing the LLM prompt serving cost by half.

# 4.4. Cross-dataset transfer

In cross-dataset transfer setting, we compare ProText with CLIP [37], CuPL [36], and image-supervised prompt learning methods. Since class-specific ImageNet LLM prompts limit its transfer to other datasets in CuPL, we assign CLIP results to CuPL for cross-datasets. Image-supervised methods [20, 21, 49, 50] are trained with 16-shot ImageNet data.

We show our main comparison results in Tab. 4. CuPL improves ImageNet performance of CLIP by ensembling ImageNet LLM prompts, while its cross-dataset results remain the same as CLIP. In contrast, ProText effectively addresses the transferability challenges of CuPL using generalized prompts trained with the same ImageNet LLM data. Since ProText allows generalization to unseen datasets, these learned prompts can directly be used with CLIP for cross-datasets leading to absolute average gains of +2.1% against CLIP and CuPL. With ProText, one can notably reduce proprietary LLM serving and prompt engineering costs as prompts learned on one dataset are effectively transferable to other datasets. We next compare ProText with strong 16-shot image-supervised methods. Without using any visual samples, ProText demonstrates effective generalization on cross-datasets and consistently surpasses previous state-of-the-art MaPLe on 9/10 datasets leading to the highest average accuracy of 67.23%. This highlights that text-only methods like ProText can lead to better generalization of CLIP as compared to image-supervised methods which tend to overfit on the source sample distributions.

# 4.5. Domain generalization experiments

We present the results for domain generalization task in Table 5. As the domain shift variants of ImageNet share class names with ImageNet, CuPL employs prompt ensembling for each dataset and provides an average gain of +2.84% over CLIP. In contrast, ProText with learned prompts shows an additional gain of +0.44% against CuPL averaged over 4 datasets. Moreover, ProText fairs competitively with image-supervised methods by showing consistent improvements over CoOp, CoCoOp, and MaPLe. These results suggest that text-only supervision methods like ProText can serve as an effective alternative to improve the robustness of VLMs when no visual information is available for training.

![](images/4c7f4c559f2f545ae8e7c157d025409e0481463fc8dce224068001a9185a6814.jpg)

<details>
<summary>line</summary>

| Number of prompts | ImageNet Top-1 Acc. | Depth of language prompting | ImageNet Top-1 Acc. |
| ----------------- | ------------------- | --------------------------- | ------------------- |
| 4                 | 70.16               | 3                           | 70.09               |
| 8                 | 70.17               | 5                           | 70.10               |
| 12                | 70.17               | 7                           | 70.14               |
| 16                | 70.22               | 9                           | 70.22               |
| 32                | 70.10               | 12                          | 70.24               |
</details>

Figure 3. Ablation: Prompt length (left) and prompt depth (right). 

<table><tr><td>Method</td><td>ImageNet Top1.</td></tr><tr><td>1: ProText-contrastive loss</td><td>68.12</td></tr><tr><td>2: ProText- L1 loss</td><td>69.96</td></tr><tr><td>3: ProText-MSE loss</td><td>70.22</td></tr></table>

Table 7. Ablation of choice of loss for contextual mapping. MSE loss provides highest results.

<table><tr><td>Method</td><td>ImageNet Top1</td></tr><tr><td>1: ProText-80 templates</td><td>68.48</td></tr><tr><td>2: ProText-Alpaca</td><td>67.10</td></tr><tr><td>3: ProText-GPT-3</td><td>70.22</td></tr></table>

Table 8. Effect on performance with different text data for training. GPT-3 text data show highest results.

<table><tr><td>Method</td><td>ImageNet Top1.</td></tr><tr><td>1: Linear Adaptor</td><td>69.36</td></tr><tr><td>2: MLP Adaptor</td><td>69.24</td></tr><tr><td>3: Prompt Learning</td><td>70.22</td></tr></table>

Table 9. Ablation on the choice of mapping network. Prompt Learning shows optimal performance.

<table><tr><td rowspan="2">Method</td><td colspan="4">Correct class confidence (%) ↑</td><td colspan="4">Incorrect class confidence (%) ↓</td></tr><tr><td>DTD</td><td>SUN</td><td>Caltech</td><td>UFC</td><td>DTD</td><td>SUN</td><td>Caltech</td><td>UFC</td></tr><tr><td>CLIP</td><td>30.5</td><td>49.3</td><td>84.5</td><td>56.4</td><td>1.51</td><td>0.13</td><td>0.16</td><td>0.44</td></tr><tr><td>ProText</td><td>33.1</td><td>54.2</td><td>89.1</td><td>59.5</td><td>1.45</td><td>0.12</td><td>0.11</td><td>0.40</td></tr></table>

Table 10. Confidence score analysis: ProText trained on ImageNet improves its logit confidence for correct classes in unseen datasets.

# 4.6. Supervised text-only training

In this setting, we compare ProText with CuPL for each dataset trained on LLM template data and the results are shown in Tab. 6. While utilizing the same LLM data, ProText achieves consistent improvements over CuPL on 10/11 datasets with an average gain of +0.59%. This reflects the generalization of the ProText approach across various diverse image datasets where it better utilizes LLM data within the learned prompts. We also compare ProText with image-supervised methods and observe that ProText fares competitively with approaches utilizing up to 2-shot samples for training. This shows ProText as a potential alternative to image-supervised methods in extremely low-data regimes. Refer to supplementary for additional results.

# 4.7. Ablative analysis

On understanding ProText prompts. In Table. 10, we present average confidence scores obtained from ProText logits trained on ImageNet-1k text data when applied to cross-datasets. Compared to CLIP, ProText exhibits increased confidence scores for correct classes across various datasets, while marginally decreasing confidence scores for incorrect classes. This suggests that the prompts learned on ImageNet-1k provide complementary and transferable contextual cues, leading to improved results. We conjecture that ProText prompts potentially improve the classification of test samples situated near the decision boundary due to higher confidence for correct classes. Refer to the supplementary section for qualitative and additional analysis.

![](images/39ff6daaad733a88fe64e20211504e2fdfff55b6303d37042f01a4cbb6a0971a.jpg)

<details>
<summary>bar_line</summary>

| Model | ImageNet Top-1 Acc. (GPT descriptions per class) | ImageNet Top-1 Acc. (Ensembled) |
| :--- | :--- | :--- |
| CuPL | 69.3 | 69.6 |
| ProText | 69.5 | 70.0 |
| ProText | 70.2 | 70.3 |
</details>

Figure 4. (Left) Effect of LLM data size on performance. (Right) Ablation on ensembling LLM descriptions for training ProText.

Loss metric in contextual mapping. We ablate on choice of loss used for the contextual mapping module in Tab. 7. Distance-based losses improve over contrastive loss. We conjecture that contrastive loss treats samples of same class labels in a same batch as negatives leading to noisy training.

Choice of LLM for generating text data. ProText by default uses GPT-3 [5] LLM to obtain text templates for training. Here we ablate on an open-source Alpaca [43] model as an alternative choice. As shown in Tab. 8, ProText with Alpaca templates performs worse than ProText-80 template and ProText-GPT-3. We observed that Alpaca templates are often noisy while GPT-3 descriptions contain more enriched class details which results in better performance.

Prompt learning verses adapter. While ProText employs prompt learning to learn contextual mapping from LLM templates, here ablations on adapters in Tab. 9. Similar to [12], we attach adapter at the output of CLIP text encoder. Adapters perform lower as compared to prompting. We conjecture that adapter completely transforms text features and loses CLIP generalization. In contrast, prompt learning append learnable vectors with CLIP text input without significant replacement and learns effective mapping function.

Training data size for text-supervision. To assess the effect of LLM template data size on ProText, we ablate on the number of descriptions per class in Fig. 4 (left). Increasing descriptions for each class consistently improves the results. This suggests that we could further boost ProText performance as quality and size of text data increases.

Ensembling in ProText training. ProText uses multiple descriptions per class and enforce mapping of class-name template feature to feature of each LLM description for that class. We conduct an alternative experiment by ensembling

a single feature from multiple LLM descriptions per class and enforce mapping on ensembled LLM feature. As shown in Fig. 4 (right), ProText-ensemblied performs lower than ProText with individual samples. We conjecture that learning on each description allows the model to utilize additional context present in each description. Ensembling can potentially mask out less frequent details available in text.

Prompt length and prompt depth. Fig. 3 (left) shows the effect of prompt length for training ProText. Setting prompt length to 16 leads to optimal performance. Fig. 3 (right) shows the effect of prompt depth on final performance where prompt depth of 9 shows optimal results.

# 5. Conclusion

Prompt learning and LLM-based ensembling are effective techniques to improve CLIP's generalization. However, prompt learning often requires labeled images, which is less practical, while LLM-based ensembling methods are dominantly class-specific and not directly transferable to new classes. To address these challenges, we propose a new direction to adapt CLIP by learning generalized prompts with text-only supervision, without relying on visual data. We introduce a training strategy for prompts to learn a mapping function that embeds rich contextual knowledge from LLM text data within the prompts. The context learned by these prompts transfers well to unseen classes and datasets, potentially reducing the LLM prompt engineering and serving cost. We perform extensive evaluations on four benchmarks where our text-only approach performs favorably well over previous methods, including those utilizing labeled images.

Acknowledgements: We would like to thank Hanan Ghani and Jameel Hassan for their help in downloading datasets. We also thank Muhammad Jehanzeb Mirza for providing Alpaca LLM prompt data for ablation experiments.

# References

[1] Bang An, Sicheng Zhu, Michael-Andrei Panaitescu-Liess, Chaithanya Kumar Mummadi, and Furong Huang. More context, less distraction: Improving zero-shot inference of clip by inferring and describing spurious features. In Workshop on Efficient Systems for Foundation Models@ICML2023, 2023. 1, 6   
[2] Hyojin Bahng, Ali Jahanian, Swami Sankaranarayanan, and Phillip Isola. Visual prompting: Modifying pixel space to adapt pre-trained models. arXiv preprint arXiv:2203.17274, 2022. 4   
[3] Hanoona Bangalath, Muhammad Maaz, Muhammad Uzair Khattak, Salman H Khan, and Fahad Shahbaz Khan. Bridging the gap between object and image-level representations for open-vocabulary detection. NeurIPS, 35:33781–33794, 2022. 2   
[4] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.

Food-101–mining discriminative components with random forests. In ECCV, pages 446–461. Springer, 2014. 6   
[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877–1901, 2020. 3, 4, 6, 8   
[6] Guangyi Chen, Weiran Yao, Xiangchen Song, Xinyue Li, Yongming Rao, and Kun Zhang. Plot: Prompt learning with optimal transport for vision-language models. In ICLR, 2022. 1, 2   
[7] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In CVPR, pages 3606-3613, 2014. 6   
[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. Ieee, 2009. 5, 6   
[9] Mohammad Mahdi Derakhshani, Enrique Sanchez, Adrian Bulat, Victor G Turrisi da Costa, Cees GM Snoek, Georgios Tzimiropoulos, and Brais Martinez. Bayesian prompt learning for image-language model generalization. In CVPR, pages 15237-15246, 2023. 2   
[10] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In CVPR, pages 14084–14093, 2022. 2   
[11] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPR Workshop, pages 178–178. IEEE, 2004. 6   
[12] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. IJCV, pages 1–15, 2023. 8   
[13] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In ECCV, pages 540–557. Springer, 2022. 2   
[14] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. J-STARS, 12(7):2217–2226, 2019. 6   
[15] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kada-vath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, pages 8340–8349, 2021. 6   
[16] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In CVPR, pages 15262-15271, 2021. 6   
[17] Tony Huang, Jack Chu, and Fangyun Wei. Unsupervised prompt learning for vision-language models. arXiv preprint arXiv:2204.03649, 2022. 1, 3   
[18] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICLR, pages 4904-4916. PMLR, 2021. 1, 2

[19] Woojeong Jin, Yu Cheng, Yelong Shen, Weizhu Chen, and Xiang Ren. A good prompt is worth millions of parameters? low-resource prompt-based learning for vision-language models. arXiv preprint arXiv:2110.08484, 2021.1   
[20] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. In CVPR, pages 19113-19122, 2023. 1, 2, 3, 4, 6, 7   
[21] Muhammad Uzair Khattak, Syed Talal Wasim, Muzammal Naseer, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Self-regulating prompts: Foundational model adaptation without forgetting. In ICCV, pages 15190–15200, 2023. 1, 3, 6, 7   
[22] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In ICCV, pages 554–561, 2013. 6   
[23] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692, 2023. 1   
[24] Boyi Li, Kilian Q. Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Language-driven semantic segmentation, 2022. 2   
[25] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In CVPR, pages 7061–7070, 2023. 2   
[26] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 2   
[27] Yuning Lu, Jianzhuang Liu, Yonggang Zhang, Yajing Liu, and Xinmei Tian. Prompt distribution learning. In CVPR, pages 5206-5215, 2022. 1, 2, 3   
[28] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013. 6   
[29] Sachit Menon and Carl Vondrick. Visual classification via description from large language models. In ICLR, 2023. 1, 2, 3, 4   
[30] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In ECCV, pages 728–755. Springer, 2022. 2   
[31] Muhammad Ferjad Naeem, Yongqin Xian, Luc V Gool, and Federico Tombari. I2dformer: Learning image to document attention for zero-shot image classification. NeurIPS, 2022. 2   
[32] Muhammad Ferjad Naeem, Muhammad Gul Zain Ali Khan, Yongqin Xian, Muhammad Zeshan Afzal, Didier Stricker, Luc Van Gool, and Federico Tombari. I2mvformer: Large language model generated multi-view document supervision for zero-shot image classification. In CVPR, 2023. 2

[33] Muhammad Ferjad Naeem, Yongqin Xian, Xiaohua Zhai, Lukas Hoyer, Luc Van Gool, and Federico Tombari. Silc: Improving vision language pretraining with self-distillation. arXiv preprint arXiv:2310.13355, 2023. 2   
[34] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In ICVGIP, pages 722-729. IEEE, 2008. 6, 2   
[35] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In CVPR, pages 3498–3505. IEEE, 2012. 6   
[36] Sarah Pratt, Ian Covert, Rosanne Liu, and Ali Farhadi. What does a platypus look like? generating customized prompts for zero-shot image classification. In ICCV, pages 15691-15701, 2023. 1, 2, 3, 4, 5, 6, 7   
[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 1, 2, 3, 6, 7, 4   
[38] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In International Conference on Machine Learning, pages 5389–5400. PMLR, 2019. 6   
[39] Karsten Roth, Jae Myung Kim, A Koepke, Oriol Vinyals, Cordelia Schmid, and Zeynep Akata. Waffling around for performance: Visual classification with random words and broad concepts. 2023. 1, 3, 4, 2   
[40] Jameel Hassan Abdul Samadh, Hanan Gani, Noor Hazim Hussein, Muhammad Uzair Khattak, Muzammal Naseer, Fahad Khan, and Salman Khan. Align your prompts: Test-time prompting with distribution alignment for zero-shot generalization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2   
[41] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models. NeurIPS, 35:14274–14289, 2022. 1, 2   
[42] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 6   
[43] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023. 8   
[44] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. In NeurIPS, 2019. 6   
[45] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In CVPR, pages 3485–3492. IEEE, 2010. 6   
[46] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. In ICLR, 2021. 2   
[47] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive

captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 1   
[48] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 2   
[49] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In CVPR, pages 16816-16825, 2022. 1, 2, 3, 4, 5, 6, 7   
[50] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. IJCV, 130(9):2337–2348, 2022. 1, 2, 3, 4, 5, 6, 7   
[51] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In ECCV, pages 350-368. Springer, 2022. 2

# Learning to Prompt with Text Only Supervision for Vision-Language Models Supplementary Material

The following sections provide supplementary material for our main paper. This includes additional analysis and comparison experiments, implementation details, and specifics of our text-to-text data used for training. The contents are organized as follows:

• Additional analysis and comparison experiments (Sec. A)   
• Additional Implementation details (Sec. B)   
• Details on Text-Only Data (Sec. C)

# A. Additional Experiments

# A.1. Additional Analysis.

Here we provide additional analysis experiments for our ProText technique.

Qualitative Analysis. In order to understand the transferability of ProText prompts across new datasets, we visualize attention maps in Fig. 5. Specifically, we employ ProText prompts learned on ImageNet-1k text-only dataset and transfer it to cross-datasets. We observe that ProText tends to focus to relevant image features while reducing its attention towards spurious features as shown in Oxford Pets and Caltech-101 images. In case of texture image from DTD, ProText shows more global attention on the texture portion of the image which is crucial in recognizing the correct texture due to the fine-grained nature of texture classes. This suggests that ProText can learn complementary contextual features, which steers CLIP for better transferability towards new datasets without relying on visual samples.

![](images/014c41d1260c201009c46753a6175171970abd153e8b6d55b45f0be77ded460d.jpg)

<details>
<summary>text_image</summary>

Oxford Pets
DTD
Caltech-101
Input Image
CLIP
ProText
</details>

Figure 5. Attention map visualizations for CLIP and ProText for cross-datasets. ProText is trained on ImageNet-1k text-only data.

<table><tr><td>Layer #</td><td>CTX 1</td><td>CTX 2</td><td>CTX 3</td><td>CTX 4</td></tr><tr><td>1</td><td>a</td><td>a</td><td>for</td><td>onto</td></tr><tr><td>2</td><td>bi</td><td>paper</td><td>erup</td><td>believes</td></tr><tr><td>3</td><td>ilwx</td><td>ered</td><td>emon</td><td>enclosure</td></tr><tr><td>4</td><td>devoted</td><td>fly</td><td>ced</td><td>hair</td></tr><tr><td>5</td><td>sin</td><td>tous</td><td>cona</td><td>emor</td></tr><tr><td>6</td><td>foto</td><td>unwanted</td><td>swagg</td><td>curfew</td></tr><tr><td>7</td><td>banan</td><td>lift</td><td>knob</td><td>maz</td></tr><tr><td>8</td><td>slow</td><td>commuter</td><td>helene</td><td>nuff</td></tr><tr><td>9</td><td>chevron</td><td>rear</td><td>crepe</td><td>opi</td></tr></table>

Table 11. Illustration of nearest words in CLIP word vocabulary against ProText prompts in different transformer layers. ProText prompts are trained on ImageNet-1k LLM prompt data.

Towards interpreting ProText prompts. Our main experiments in Sec. 4.7 demonstrated that ProText trained on ImageNet-1k text dataset performs favorably well across cross-datasets. Here we are interested in studying how the ProText prompt vectors are interpreted in natural language. Specifically, we searched for words in CLIP vocabulary that are closest to the learned prompts using Euclidean distance in the embedding space. The results in Table 11 show the nearest (valid) word for ProText prompts across different transformer layers. Note that these words may note concretely correspond to the learned prompts as we could only select nearest ones. We observe that the represented words are diverse containing connecting words that are common in web captions such as "a," "for," and "onto". Additionally, since CLIP uses a BPE representation for tokenization, several subwords appear among the nearest words, such as "sin," "ced," and "banan." These subwords can collectively contribute to strong context priors, such as deriving "banana" from "banan," "Mercedes" from "ced," and "casino" from "sin," which may be potentially relevant for downstream datasets like SUN397 and Stanford Cars. At the same time, some words do not appear to contribute much for context enhancement such as "ilwx", "curfew" etc. In summary, similar to the findings in [50], the learned vectors may encompass word representations not explicitly present in the existing vocabulary.

# A.2. Additional comparisons with WaffleCLIP.

We present additional comparisons between ProText and WaffleCLIP [39] approach. WaffleCLIP employs prompt ensembling by introducing random descriptors and characters alongside class names. Specifically, we perform a comparison with a WaffleCLIP-Concept variant, which incorporates high-level dataset concepts in its text prompts,

such as ‘a photo of a flower: a CLS’ for OxfordFlowers [34]. Further details on the WaffleCLIP framework and its variants can be found in [39].

Cross-dataset transfer. For cross-dataset transfer settings, all methods only utilize ImageNet source dataset LLM prompt information. The results are shown in Tab. 12. CuPL shows the same performance as CLIP for cross-datasets as class-specific descriptions for new datasets are not available in this setting. Overall, WaffleCLIP uses random descriptors which leads to improvements over CLIP and CuPL. In contrast, ProText with text-only training with ImageNet-1k LLM templates shows consistent improvements over WaffleCLIP by surpassing on 9/10 cross-datasets and leads to the averaged accuracy of 67.23% in the challenging cross-dataset transfer setting.

Text-only supervised setting. We additionally compare WaffleCLIP in text-only supervised setting. As shown in Tab. 13, WaffleCLIP improves over CLIP but lags behind CuPL as it only relies on high-level dataset concepts and random descriptors. CuPL uses class-specific LLM descriptions for prompt ensembling and shows improved results. In contrast to these approaches, ProText adopts a learning-based approach using text data and shows the highest performance by surpassing both WaffleCLIP and CuPL in 10/11 datasets. This suggests that text-only prompt learning can serve as a better alternative to training-free prompt ensembling methods.

# A.3. Comparison with image-supervised methods.

We show additional comparisons of ProText with image-supervised methods in terms of generalization performance. In base to novel class generalization setting, we include prompt learning methods utilizing 16-shot image data where we mainly focus on novel class performance for comparison. In text-only supervised setting, we compare ProText with few-shot image supervised methods including CLIP Linear Probe, CoOp, and CoCoOp, which are trained up to 2-shot data.

Unseen class generalization. All methods are trained on seen classes of each dataset and we specifically analyze their performance on unseen classes to study generalization. Results are shown in Tab. 14. Image-supervised prompt learning methods utilize 16-shot base-class labeled data and demonstrate improved accuracy for novel classes. For example, the previous state-of-the-art method, PromptSRC, achieves a substantial accuracy of 70.73% on ImageNet for novel classes. In comparison, ProText, leveraging text-only data, shows an improvement of +0.65% against PromptSRC for novel classes on ImageNet. In summary, ProText consistently outperforms PromptSRC on 9 out of 11 datasets for novel classes, leading to the highest novel class accuracy of 76.98% averaged over 11 datasets.

Supervised setting. In Tab. 15, compare ProText with few-shot image-supervised methods including CLIP Linear Probe, CoOp, and CoCoOp. ProText shows improved averaged performance over 1 & 2 shot Linear Probe. Similarly, ProText without using any images for training improves on most datasets against CoOp and CoCoOp trained with 1 and 2 shots. ProText, without using any images for training, outperforms CoOp and CoCoOp trained with 1 and 2 shots on most datasets. This suggests that text-only training can be considered an effective alternative approach to image-supervised methods under extreme low-data regimes.

# A.4. Additional ablation studies.

We present additional ablation experiments conducted on ProText as outlined below.

Combining prompt ensembling and prompt learning. In our ProText approach, learnable prompts for inference are trained on text data. Here, we explore an alternative experiment by averaging the text features with ProText-learned prompts and text features of LLM templates obtained via prompt ensembling. Specifically, we average the LLM prompt features (e.g., CuPL features) and ProText features for the same classes to study if prompt learning and prompt ensembling could be complementary. The results are shown in Table 16. Combining ProText and CuPL features leads to marginal improvement compared to ProText alone. We conjecture that since ProText uses the same LLM template data to learn prompts, the LLM template features and ProText features might not be strongly complementary.

# B. Additional Implementation details

Training details. For training ProText, we use a publicly available CLIP ViT-B/16 model from OpenAI [37]. Language prompts for each training are initialized with 'a photo of a' for the first layer and randomly initialized for the remaining transformer layers of the text encoder of CLIP. All models are trained using the AdamW optimizer on a single 16-GB V100 GPU. For cross-dataset and domain generalization benchmarks, we train ProText using $T = 4$ and $T = 16$ language prompts, respectively, for 10 and 200 epochs, respectively. The warm-up epochs are set to 5 during training.

As text data from LLMs varies in quality and size across datasets, we have observed that training ProText on each dataset requires custom training configurations to achieve the best performance. Therefore, ProText employs optimal prompt length and epoch configuration for each dataset. The optimal training configurations are obtained through the validation splits of each dataset.

Base-to-novel generalization setting. In Tab. 17, we show the hyperparameters used for training models in base-to-novel generalization settings. We use a learning rate of 0.03 for all datasets except UCF101, FOOD101, and Oxford-Flowers where learning rate of 0.0025 is used.

<table><tr><td rowspan="2"></td><td>Source</td><td colspan="11">Target</td></tr><tr><td>ImageNet</td><td>Caltech101</td><td>OxfordPets</td><td>StanfordCars</td><td>Flowers102</td><td>Food101</td><td>Aircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td><td>Average</td></tr><tr><td colspan="13">Zero-shot &amp; Prompt ensembling methods</td></tr><tr><td>CLIP</td><td>66.72</td><td>92.98</td><td>89.13</td><td>65.29</td><td>71.30</td><td>86.11</td><td>24.90</td><td>62.59</td><td>44.56</td><td>47.84</td><td>66.83</td><td>65.15</td></tr><tr><td>CuPL</td><td>69.62</td><td>92.98</td><td>89.13</td><td>65.29</td><td>71.30</td><td>86.11</td><td>24.90</td><td>62.59</td><td>44.56</td><td>47.84</td><td>66.83</td><td>65.15</td></tr><tr><td>WaffleCLIP-Concept</td><td>68.34</td><td>94.01</td><td>89.57</td><td>63.42</td><td>72.00</td><td>86.84</td><td>24.49</td><td>66.17</td><td>45.15</td><td>47.74</td><td>67.96</td><td>65.74</td></tr><tr><td colspan="13">Prompt learning with text-only supervision</td></tr><tr><td>ProText (Ours)</td><td>69.80</td><td>94.81</td><td>91.01</td><td>66.00</td><td>72.35</td><td>86.66</td><td>24.72</td><td>67.34</td><td>47.93</td><td>51.86</td><td>69.60</td><td>67.23</td></tr></table>

Table 12. Cross-dataset transfer setting. Results comparison of ProText with CLIP, CuPL, and Waffle-CLIP. ProText overall shows consistent improvements over LLM-based prompt ensembling methods.

<table><tr><td>Dataset</td><td>CLIP</td><td>CuPL</td><td>WaffleCLIP-C</td><td>ProText</td><td> $\Delta$ </td></tr><tr><td>ImageNet</td><td>66.72</td><td>69.60</td><td>68.34</td><td>70.22</td><td>+0.62</td></tr><tr><td>Caltech101</td><td>92.98</td><td>94.32</td><td>94.01</td><td>95.29</td><td>+0.97</td></tr><tr><td>DTD</td><td>44.56</td><td>53.96</td><td>45.15</td><td>54.04</td><td>+0.06</td></tr><tr><td>EuroSAT</td><td>47.84</td><td>60.27</td><td>47.74</td><td>58.53</td><td>-1.74</td></tr><tr><td>StanfordCars</td><td>65.29</td><td>65.95</td><td>63.42</td><td>66.77</td><td>+0.82</td></tr><tr><td>Flowers102</td><td>71.30</td><td>73.85</td><td>72.00</td><td>74.42</td><td>+0.57</td></tr><tr><td>Aircraft</td><td>24.90</td><td>27.66</td><td>24.49</td><td>29.01</td><td>+1.35</td></tr><tr><td>SUN397</td><td>62.59</td><td>69.00</td><td>66.17</td><td>69.76</td><td>+0.76</td></tr><tr><td>OxfordPets</td><td>89.13</td><td>91.11</td><td>89.57</td><td>92.72</td><td>+1.61</td></tr><tr><td>UCF101</td><td>66.83</td><td>70.63</td><td>67.96</td><td>71.45</td><td>+0.82</td></tr><tr><td>Food101</td><td>86.11</td><td>86.11</td><td>86.84</td><td>86.68</td><td>+0.57</td></tr><tr><td>Average</td><td>65.15</td><td>69.31</td><td>65.97</td><td>69.90</td><td>+0.59</td></tr></table>

Table 13. ProText results with text supervision on each dataset. We compare ProText with CLIP and CuPL and WaffleCLIP-Concept. Gains of ProText over CuPL are shown in blue.

<table><tr><td>Dataset</td><td>CuPL [36]</td><td>ProText Ours</td><td>CoOp [50]</td><td>CoCoOp [49]</td><td>MaPLe [20]</td><td>PromptSRC [21]</td><td> $\Delta$ </td></tr><tr><td>ImageNet</td><td>68.14</td><td>71.38</td><td>67.88</td><td>70.43</td><td>70.54</td><td>70.73</td><td>+3.2</td></tr><tr><td>Caltech101</td><td>94.00</td><td>95.63</td><td>89.81</td><td>93.81</td><td>94.36</td><td>94.03</td><td>+1.6</td></tr><tr><td>DTD</td><td>59.90</td><td>61.59</td><td>41.18</td><td>56.00</td><td>59.18</td><td>62.97</td><td>+1.7</td></tr><tr><td>EuroSAT</td><td>64.05</td><td>80.97</td><td>54.74</td><td>60.04</td><td>73.23</td><td>73.90</td><td>+17</td></tr><tr><td>StanfordCars</td><td>74.89</td><td>76.08</td><td>60.40</td><td>73.59</td><td>74.00</td><td>74.97</td><td>+1.2</td></tr><tr><td>Flowers102</td><td>77.80</td><td>78.44</td><td>59.67</td><td>71.75</td><td>72.46</td><td>76.50</td><td>+0.6</td></tr><tr><td>Aircraft</td><td>36.29</td><td>34.13</td><td>22.30</td><td>23.71</td><td>35.61</td><td>37.87</td><td>-2.2</td></tr><tr><td>SUN397</td><td>75.35</td><td>79.14</td><td>65.89</td><td>76.86</td><td>78.70</td><td>78.47</td><td>+3.8</td></tr><tr><td>OxfordPets</td><td>97.26</td><td>98.00</td><td>95.29</td><td>97.69</td><td>97.76</td><td>97.30</td><td>+0.7</td></tr><tr><td>UCF101</td><td>77.50</td><td>79.50</td><td>56.05</td><td>73.45</td><td>78.66</td><td>78.80</td><td>+2.0</td></tr><tr><td>Food101</td><td>91.22</td><td>91.98</td><td>82.26</td><td>91.29</td><td>92.05</td><td>91.53</td><td>+0.8</td></tr><tr><td>Average</td><td>74.22</td><td>76.98</td><td>63.22</td><td>71.69</td><td>75.14</td><td>76.10</td><td>+2.8</td></tr></table>

Table 14. Novel-class generalization comparison. We compare ProText with prompt ensembling and image-supervised methods on unseen class performance in base-to-novel class generalization setting. Gains of ProText over CuPL are shown in blue.

Text-only supervised setting. For our comparison with CuPL [36] in Table 15, ProText models are trained using the same LLM text data as utilized by CuPL. Hyperparameter values are shown in Table 18. All models are trained using a learning rate of 0.03, except for UCF101, EuroSAT, and Oxford-Flowers, where a learning rate of 0.0025 is used.

# C. Details on Text-Only Data

As discussed in Sec. 3.2.1, our ProText approach relies on text-only data (DPROMPT) curated from Language Models (LLMs) for training its language prompts. Here, we provide additional details on the curation of text-only data. Specifically, we first provide information on the text queries used as input to LLMs for generating prompts, followed by qualitative examples of $D_{PROMPT}$ .

# C.1. Queries to LLMs to curate Text-Only Data

Following [36], we obtain class descriptions from LLMs by providing various queries as inputs. Specifically, we utilize queries termed as Full prompts by CuPL [36]. For instance, to generate class descriptions of ImageNet-1k classes, we prompt GPT-3 with the following 5 queries:

- 'Describe what a(n) CLS looks like.'   
- 'How can you identify a(n) CLS?'   
• 'What does a(n) look like?'   
- 'Describe an image from the internet of a(n) CLS.'   
- 'A caption of an image of a(n) CLS.'

Here, CLS denotes the class names present in the dataset. After generating LLM class descriptions, we associate all descriptions of the same class with its class-name template given as 'A photo of a CLS'. This results in our text-only training data $D_{PROMPT}$ with text-to-text mapping pairs used to train ProText. Refer to [36] for LLM queries of other datasets used to generate class-specific descriptions. For standardized comparisons, we use publicly available CuPL data and generate descriptions for datasets not provided by CuPL.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">CLIP</td><td rowspan="2">CuPL</td><td rowspan="2">ProText</td><td colspan="2">Linear Probe</td><td colspan="2">CoOp</td><td colspan="2">CoCoOp</td></tr><tr><td>K=1</td><td>K=2</td><td>K=1</td><td>K=2</td><td>K=1</td><td>K=2</td></tr><tr><td>ImageNet</td><td>66.70</td><td>69.62</td><td>70.22</td><td>32.13</td><td>44.88</td><td>66.33</td><td>67.07</td><td>69.43</td><td>69.78</td></tr><tr><td>Caltech101</td><td>92.98</td><td>94.32</td><td>95.29</td><td>79.88</td><td>89.01</td><td>92.60</td><td>93.07</td><td>93.83</td><td>94.82</td></tr><tr><td>DTD</td><td>44.56</td><td>53.96</td><td>54.04</td><td>34.59</td><td>40.76</td><td>50.23</td><td>53.60</td><td>48.54</td><td>52.17</td></tr><tr><td>EuroSAT</td><td>47.84</td><td>60.27</td><td>58.53</td><td>49.23</td><td>61.98</td><td>54.93</td><td>65.17</td><td>55.33</td><td>46.74</td></tr><tr><td>StanfordCars</td><td>65.29</td><td>65.95</td><td>66.77</td><td>35.66</td><td>50.28</td><td>67.43</td><td>70.50</td><td>67.22</td><td>68.37</td></tr><tr><td>Flowers102</td><td>71.30</td><td>73.85</td><td>74.42</td><td>69.74</td><td>85.07</td><td>77.53</td><td>87.33</td><td>72.08</td><td>75.79</td></tr><tr><td>Aircraft</td><td>24.90</td><td>27.66</td><td>29.01</td><td>19.61</td><td>26.41</td><td>21.37</td><td>26.20</td><td>12.68</td><td>15.06</td></tr><tr><td>SUN397</td><td>62.59</td><td>69.00</td><td>69.76</td><td>41.58</td><td>53.70</td><td>66.77</td><td>66.53</td><td>68.33</td><td>69.03</td></tr><tr><td>OxfordPets</td><td>89.13</td><td>91.11</td><td>92.72</td><td>44.06</td><td>58.37</td><td>90.37</td><td>89.80</td><td>91.27</td><td>92.64</td></tr><tr><td>UCF101</td><td>66.83</td><td>70.63</td><td>71.45</td><td>53.66</td><td>65.78</td><td>71.23</td><td>73.43</td><td>70.30</td><td>73.51</td></tr><tr><td>Food101</td><td>86.11</td><td>86.11</td><td>86.68</td><td>43.96</td><td>61.51</td><td>84.33</td><td>84.40</td><td>85.65</td><td>86.22</td></tr><tr><td>Average</td><td>65.15</td><td>69.31</td><td>69.90</td><td>45.83</td><td>57.98</td><td>67.56</td><td>70.65</td><td>66.79</td><td>67.65</td></tr></table>

Table 15. ProText results with text supervision on each dataset. We compare ProText with CLIP [37], CuPL [36] and image supervised Linear Probe [37], CoOp [50] and CoCoOp [49] methods.

<table><tr><td>Method</td><td>ImageNet Top1.</td></tr><tr><td>1: CuPL</td><td>69.62</td></tr><tr><td>2: ProText</td><td>70.22</td></tr><tr><td>3: Ensembling: ProText + CuPL</td><td>70.28</td></tr></table>

Table 16. Ablation on combining CuPL and ProText text features.

<table><tr><td>H.parameter</td><td>ImageNet</td><td>Caltech101</td><td>OxfordPets</td><td>StanfordCars</td><td>Flowers102</td><td>Food101</td><td>Aircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td></tr><tr><td>Epochs</td><td>30</td><td>30</td><td>50</td><td>30</td><td>150</td><td>50</td><td>200</td><td>30</td><td>200</td><td>30</td><td>20</td></tr><tr><td># Prompts (T)</td><td>4</td><td>8</td><td>4</td><td>8</td><td>4</td><td>8</td><td>4</td><td>8</td><td>4</td><td>16</td><td>16</td></tr></table>

Table 17. Hyper-parameters setting used for base-to-novel generalization setting. Optimal configuration is set using validation splits of each dataset. 

<table><tr><td>H.parameter</td><td>ImageNet</td><td>Caltech101</td><td>OxfordPets</td><td>StanfordCars</td><td>Flowers102</td><td>Food101</td><td>Aircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td></tr><tr><td>Epochs</td><td>200</td><td>30</td><td>50</td><td>20</td><td>300</td><td>30</td><td>200</td><td>200</td><td>200</td><td>300</td><td>100</td></tr><tr><td># Prompts (T)</td><td>16</td><td>16</td><td>4</td><td>8</td><td>4</td><td>16</td><td>4</td><td>16</td><td>16</td><td>4</td><td>8</td></tr></table>

Table 18. Hyper-parameters used for text-only supervised setting.

# C.2. Qualitative examples

As LLMs are pre-trained on internet-scale text corpora, they possess the capability of generating diverse and high-quality descriptions and captions for different class categories, resulting in high-quality text outputs. Below we show some examples of $D_{PROMPT}$ text-to-text pairs for the ImageNet-1k dataset.

# Class: Tench

Class-name template: 'A photo of a Tench'

Associated LLM descriptions:

- 'A tench is a freshwater fish with a dark green back and light-colored sides.'   
- 'A tench looks like a freshwater fish with a dark olive-green back, fading to yellowish-brown on the sides.'   
- 'Tench are a freshwater fish that can grow up to 70cm long! They have olive-brown skin with dark spots, and their meat is white and firm.'   
- 'This image shows a large, dark green tench swimming in a pond.'

# Class: bath towel

Class-name template: 'A photo of a bath towel'

Associated LLM descriptions:

- 'A bath towel typically has a loops on one side and a smooth surface on the other.'   
- 'A bath towel is a rectangular piece of fabric, usually Cotton, that is used to dry oneself after a bath or shower.'   
• ‘The image is of a white bath towel with a blue and green stripes.’   
- 'A fluffy white bath towel draped over a towel rack.'

# Class: sandal

Class-name template: 'A photo of a sandal'

Associated LLM descriptions:

- 'A sandal is a shoe typically made of leather or synthetic material that has an open toe and a strap or straps that go around the foot or up the ankle.'   
- 'A sandal is usually a flat shoe with a strap that goes around the foot or ankle.'   
- 'This sandal is from the ancient Egyptian city of Thebes.'   
- 'When you are looking to identify a sandal, the first place to start is by looking at the features of the shoe.'