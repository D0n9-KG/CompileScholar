# CAST: Cross-Attention in Space and Time for Video Action Recognition

Dongho Lee\*

Jongseo Lee\*

Jinwoo Choi $^{†}$

Kyung Hee University, Republic of Korea

{kide004, jong980812, jinwoochoi}@khu.ac.kr

# Abstract

Recognizing human actions in videos requires spatial and temporal understanding. Most existing action recognition models lack a balanced spatio-temporal understanding of videos. In this work, we propose a novel two-stream architecture, called Cross-Attention in Space and Time (CAST), that achieves a balanced spatio-temporal understanding of videos using only RGB input. Our proposed bottleneck cross-attention mechanism enables the spatial and temporal expert models to exchange information and make synergistic predictions, leading to improved performance. We validate the proposed method with extensive experiments on public benchmarks with different characteristics: EPIC-KITCHENS-100, Something-Something-V2, and Kinetics-400. Our method consistently shows favorable performance across these datasets, while the performance of existing methods fluctuates depending on the dataset characteristics. The code is available at https://github.com/KHU-VLL/CAST.

# 1 Introduction

To accurately recognize human actions in videos, a model must understand both the spatial and temporal contexts. A model that lacks fine-grained spatial understanding is likely to fail in predicting the correct action. For example, as shown in Figure 1 (a), a model that understands temporal context such as hand motion across frames but not the fine-grained spatial context may confuse whether an object in the hand a ketchup, or a cheese, or a milk carton. Consequently, the model fails to predict the correct action, Put down a cheese. Similarly, a model that lacks temporal context understanding may also fail to predict the correct action. In Figure 1 (b), let us suppose a model understands spatial context but does not understand temporal context, e.g., the model is confused about whether the hand is moving from outside the fridge to the inside or vice versa. Then the model fails to predict the correct action of Take out a sauce. Therefore, for accurate action recognition, models need to comprehend both the spatial and temporal contexts of videos.

Despite the recent progress in action recognition through the use of Transformers $[60, 11, 3]$ , achieving a balanced spatio-temporal understanding remains a challenging problem. Compared to images, the additional temporal dimension in videos makes spatio-temporal representation learning computationally intensive and requires a significant amount of training data $[3]$ . Consequently, most action recognition models lack a balanced spatio-temporal understanding of videos. Notably, models that perform well on static-biased $[32, 8, 50]$ datasets, such as Kinetics-400, may not perform as well on temporal-biased $[3, 28]$ datasets, such as Something-Something-V2, and vice versa. For instance, as shown in Figure 4 (a), on the EPIC-KITCHENS-100 dataset, VideoMAE $[56]$ outperforms ST-Adapter $[42]$ on the verb prediction task, while ST-Adapter outperforms VideoMAE on the noun prediction task. Similarly, BEVT $[62]$ outperforms AIM $[72]$ on the Something-Something-V2

![](images/c634f9a7d4e8d119d5372e2aa0a18cf35cedde188d94bdc78d53812ceb0c1124.jpg)

<details>
<summary>text_image</summary>

Ground truth: Put down a cheese
Ketchup?
√ Cheese?
× Milk carton?
(a) Importance of spatial understanding
</details>

![](images/7c75295a5e272cfc6545775a2a1bb95ef22bc5d433478316672c7ab73f52d2bd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Ground truth: Take out a sauce"] --> B["Action 1"]
    A --> C["Action 2"]
    A --> D["Action 3"]
    A --> E["Action 4"]
    F["Put in?"] --> G["Action 1"]
    F --> H["Action 2"]
    I["Take out?"] --> J["Action 3"]
    I --> K["Action 4"]
```
</details>

Figure 1: The importance of spatio-temporal understanding. If a model lacks fine-grained spatial understanding, the model may predict an incorrect action. E.g., the model fails to predict Put down a cheese in (a) due to subtle appearance differences between the objects. On the other hand, if a model lacks temporal context understanding, the model may predict an incorrect action. E.g., the model fails to predict Take out a sauce in (b) due to the ambiguity of the action. Therefore, both spatial and temporal understanding are crucial in action recognition. Best viewed with zoom and color.

dataset, while BEVT underperforms AIM on the Kinetics-400 dataset. We observe a similar trend for other methods as reported in Table 2.

One possible solution to the challenge of balanced spatio-temporal understanding is to use multi-modal learning. For example, two-stream networks $[51, 14]$ employ both RGB and optical flow streams to learn both spatial and temporal contexts. However, this approach can be computationally expensive due to optical flow estimation.

In this work, we introduce a two-stream architecture, Cross-Attention in Space and Time (CAST), to address the challenge of balanced spatio-temporal understanding using only RGB input. In Figure 2, we show a high-level illustration of the proposed method. Our architecture employs two expert models - a spatial expert model and a temporal expert model - which exchange information to make a synergistic collective prediction. We realize the information exchange by cross-attention between the two experts. We empirically validate that placing cross-attention in a bottleneck architecture facilitates more effective learning. To validate the effectiveness of the proposed method, we conduct extensive experiments on multiple datasets with distinct characteristics, including the temporal-biased Something-Something-V2, static-biased Kinetics-400, and fine-grained EPIC-KITCHENS-100. Our results demonstrate that CAST achieves balanced spatio-temporal understanding and shows favorable performance across these different datasets.

In this work, we make the following significant contributions.

- We introduce a two-stream architecture, CAST, which addresses the challenge of balanced spatio-temporal understanding that has been largely overlooked by previous works.   
- We conduct extensive experiments on multiple datasets with distinct characteristics to demonstrate the effectiveness of CAST. In terms of balanced spatio-temporal understanding, CAST shows favorable performance, while existing methods show more imbalanced performance.   
- We conduct an extensive ablation study and analysis to validate the design choices of the proposed method. We show that employing spatial expert and temporal expert and placing cross-attention in a bottleneck architecture is crucial for achieving effective spatio-temporal representation learning.

# 2 Related Work

Video Action Recognition. CNN-based approaches have been widely used for action recognition, including 2D CNNs $[61, 74, 33, 52, 27]$ , 3D CNNs $[58, 5, 59, 64, 13]$ , 2D and 1D separable CNNs $[59, 70]$ , or two-stream CNNs $[14, 15]$ . These methods have achieved great progress thanks to the strong inductive biases. Recently, Transformer-based approaches $[1, 3, 21, 43, 68, 12, 71]$ become popular in the community due to the long-term context modeling capabilities. Similar to the two-stream CNNs, we propose a two-stream transformer architecture consisting of two expert models: a spatial expert and a temporal expert. However, unlike traditional two-stream CNNs, we use RGB input only, instead of RGB and flow.

![](images/96ec947bb06499b9f154be87d380f3cf2744ca5c3186faf27e287ad59a977bad.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["This is a hand holding a fork."] --> B["Spatial expert"]
    C["There are hand, dishes, and utensil holder."] --> D["Spatial expert"]
    E["Collective prediction: Pick up a fork"] --> F["Next conversation"]
    G["The actor is picking up a cutlery."] --> H["Temporal expert"]
    I["The actor is reaching out and bringing something."] --> J["Temporal expert"]
    K["Ground Truth: Pick up a fork"] --> L["Next conversation"]
    M["Ground Truth: Pick up a fork"] --> N["Next conversation"]
```
</details>

Figure 2: High-level illustration of the proposed method. In this work, we employ spatial and temporal expert models. The two experts exchange information with each other using cross-attention. Initially, the experts may predict incorrect actions due to the lack of information. For example, the temporal expert may predict reach out to something while the ground truth is Pick up a fork. Similarly, the spatial expert may predict utensil holder instead of fork in the shallower layers. However, after using cross-attention to exchange information multiple times, the proposed method can collectively predict the correct action Pick up a fork. Best viewed with zoom and color.

Cross-attention. Cross-attention has been widely utilized in multi-modal learning to facilitate information exchange between different modalities such as audio, visual, and text $[34, 67, 40, 30, 18]$ . Recently, cross-attention between different views of the same video has shown impressive results $[71, 75, 6, 26]$ . Similar to these, we propose a cross-attention method using a single RGB input, but with two distinct expert models: a spatial expert and a temporal expert. The two experts attend to each other through cross-attention to achieve a balanced spatio-temporal understanding.

Foundation model. Trained on web-scale datasets using self-supervised learning, foundation models $[25, 4, 48, 45, 44]$ are highly adaptable and versatile. Foundation models show impressive performance on various tasks in computer vision $[65, 62]$ , natural language processing $[49, 57]$ , and audio recognition $[17]$ . In this work, we employ CLIP $[44]$ as our spatial expert as it shows impressive performance on more than 30 computer vision tasks.

Parameter-efficient transfer learning. Although the “pre-training and fine-tuning” paradigm with strong foundation models has demonstrated impressive performance on several computer vision tasks, it is computationally expensive and often unnecessary to fine-tune the full model $[72]$ . Several works have demonstrated that learning only a small subset of parameters and keeping the remaining parameters frozen is effective for NLP tasks $[23, 29]$ and computer vision tasks $[36, 66, 54, 46, 47]$ . Extending image foundation models by adding adapter architectures has shown favorable performance on action recognition $[35, 72, 42]$ . The proposed method also employs adapter architecture with cross-attention between two experts. We empirically demonstrate that the proposed method outperforms existing adapter-based video models in terms of achieving balanced spatio-temporal understanding.

# 3 Method: Cross-Attention in Space and Time

We introduce CAST, a method for balanced spatio-temporal representation learning for action recognition, as shown in Figure 3. We employ frozen spatial and temporal expert models that can be any vision transformer, consisting of 12 transformer blocks each. To facilitate information exchange between the experts, we introduce the bottleneck cross-attention in space and time (B-CAST) module on top of the frozen layers. This module enables the experts to exchange information and learn more balanced spatio-temporal contexts than separate experts. To improve adaptation to downstream tasks, we use adapter layers with a small number of learnable parameters, following AIM [72]. In the following subsections, we provide a detailed description of each component of our proposed CAST.

# 3.1 Input embeddings

CAST takes only RGB videos as inputs. The input is a mini-batch of videos, $I \in R^{B \times 2T \times H \times W \times C}$ , consisting of B videos of 2T frames, $H \times W$ spatial dimensions, and C channels. We apply patch tokenization to the input videos for the spatial expert and the temporal expert. For the spatial expert, we decompose every even frame of each video in I into N non-overlapping patches of $p \times p$ pixels [11].

![](images/e2ef05a678659d54e8289fc1b9ccf8bf16a358dc328e03cddf1a054a0d5bc76d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph "Proposed Architecture Overview"
        A["Input Video"] --> B["Patch & Pos Embed"]
        B --> C["LayerNorm"]
        C --> D["Multi-Head Self-Attention"]
        D --> E["Adapter"]
        E --> F["B-CAST"]
        F --> G["Adapter"]
        G --> H["LayerNorm"]
        H --> I["Feed Forward"]
        I --> J["Adapter"]
        J --> K["CLS token"]
        K --> L["Classification Head"]
        L --> M["Adapter"]
        M --> N["GAP token"]
        N --> O["Information Exchange"]
        O --> P["B-CAST"]
        P --> Q["Adapter"]
        Q --> R["LayerNorm"]
        R --> S["Feed Forward"]
        S --> T["Adapter"]
        T --> U["B-CAST"]
    end

    subgraph "Cross-attention Window Shape"
        V["T2S Cross-attention (Window shape: Time)"]
        W["S2T Cross-attention (Window shape: Space)"]
    end

    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style P fill:#f9f,stroke:#333
    style Q fill:#f9f,stroke:#333
    style R fill:#f9f,stroke:#333
    style S fill:#f9f,stroke:#333
    style T fill:#f9f,stroke:#333
    style U fill:#f9f,stroke:#333
    style V fill:#f9f,stroke:#333
    style W fill:#f9f,stroke:#333
    style X fill:#f9f,stroke:#333
    style Y fill:#f9f,stroke:#333
    style Z fill:#f9f,stroke:#333
```
</details>

Figure 3: Overview of CAST. (a) CAST employs frozen spatial and temporal expert models. On top of the experts, we add a cross-attention module B-CAST to enable the exchange of information between the two experts. Additionally, we employ adapters with a small number of learnable parameters to the experts for better adaptation. (b) The proposed B-CAST consists of temporal-to-spatial (T2S) and spatial-to-temporal (S2T) cross-attentions to allow for a better understanding of the spatio-temporal features in the video data. For efficient and effective learning, we incorporate cross-attention into the bottleneck adapter. We employ separate position embedding for each expert. (c) We visualize T2S and S2T cross-attentions. Given a query, the model attends along the temporal axis only in T2S while the model attends along the spatial axes only in S2T.

Then we pass the patches through a frozen linear layer and add position embeddings to obtain spatial embeddings, $X_{s} \in R^{BT \times N \times D}$ . For the temporal expert, we decompose every two frames of each video in I into $2 \times p \times p$ pixels non-overlapping tubes [1]. Then we pass the tubes through a frozen linear layer and add position embeddings to obtain temporal embeddings, $X_{t} \in R^{B \times TN \times D}$ .

# 3.2 CAST architecture

The model architecture of each expert is the same as the ViT except for adapters and the B-CAST module. All the other parameters are frozen, while the adapter and B-CAST parameters are learnable.

For completeness, we first define the operations used and then describe the entire model architecture. Given an input X, we define Multi-Head Self Attention (MHSA) operation as follows:

$$
\operatorname{MHSA} (\mathbf {X}) = \operatorname{Softmax} \left(\left(\mathbf {X W} _ {Q}\right) \left(\mathbf {X W} _ {K}\right) ^ {\top}\right) \left(\mathbf {X W} _ {V}\right), \tag {1}
$$

$\mathbf{W}_Q, \mathbf{W}_K,$ and $\mathbf{W}_V$ are the query, key, and value projection matrices, respectively. We also define the adapter operation with linear down and up projection matrices $\mathbf{W}_D$ and $\mathbf{W}_U$ as follows:

$$
\mathrm{ADAP} (\mathbf {X}) = \boldsymbol {\sigma} (\mathbf {X} \mathbf {W} _ {D}) \mathbf {W} _ {U}, \tag {2}
$$

where $\sigma (\cdot)$ is the GELU activation function [20].

For each attention block l, we apply independent Multi-Head Self Attention (MHSA) for each expert along with a skip connection as follows:

$$
\mathbf {Y} ^ {(l)} = \mathbf {X} ^ {(l)} + \operatorname{ADAP} (\operatorname{MHSA} (\operatorname{LN} (\mathbf {X} ^ {(l)}))) + \operatorname{MHSA} (\operatorname{LN} (\mathbf {X} ^ {(l)})), \tag {3}
$$

where $\mathrm{LN}(\cdot)$ denotes the Layer Normalization operation. The spatial path undergoes spatial attention, while the temporal path undergoes space-time attention following TimeSformer [3].

As shown in Figure 3 (b), to exchange information between the two experts, we apply the B-CAST operation $\Phi (\cdot)$ to $\mathbf{Y}_{e_1}$ and $\mathbf{Y}_{e_2}$ from the expert $e_1$ and $e_2$ as follows along with a skip connection:

$$
\mathbf {B} ^ {(l)} = \mathbf {Y} ^ {(l)} + \Phi (\mathbf {Y} _ {e _ {1}} ^ {(l)}, \mathbf {Y} _ {e _ {2}} ^ {(l)}). \tag {4}
$$

We describe the B-CAST operation $\Phi(\cdot)$ in detail in Section 3.3.

Finally, we pass the output, denoted as $\mathbf{B}^{(l)}$ , through a two-layer feed forward network (FFN) [11] with the GELU activation function in between the layers and another adapter to obtain the next layer input $\mathbf{X}^{(l+1)}$ as follows along with a skip connection:

$$
\mathbf {X} ^ {(l + 1)} = \mathbf {B} ^ {(l)} + \operatorname{FFN} \left(\mathrm{LN} \left(\mathbf {B} ^ {(l)}\right)\right) + \operatorname{ADAP} \left(\mathrm{LN} \left(\mathbf {B} ^ {(l)}\right)\right). \tag {5}
$$

Classification head. To produce the final prediction, we need to aggregate the outputs of both spatial and temporal experts. For the spatial expert, we average the frame-level class tokens from the last attention block, $\mathbf{X}_{s}^{(12)}$ , to obtain a single class token. We denote this operation as $\mathrm{CLS}(\cdot)$ . To obtain temporal expert features, we aggregate all the tokens from the last attention block of the temporal expert, $\mathbf{X}_{t}^{(12)}$ , using the global average pooling $\mathrm{GAP}(\cdot)$ operation. Then we add the adapter output of the CLS token and the adapter output of the GAP token to produce a fused token Z:

$$
\mathbf {Z} = \mathrm{ADAP} (\mathrm{CLS} (\mathbf {X} _ {s} ^ {(1 2)})) + \mathrm{ADAP} (\mathrm{GAP} (\mathbf {X} _ {t} ^ {(1 2)})). \tag {6}
$$

Finally, we feed the fused token Z a classification layer followed by a softmax function to obtain the predicted class probabilities. We train the model using the standard cross-entropy loss.

# 3.3 B-CAST module architecture

Multi-Head Cross-Attention. Multi-Head Cross-Attention (MHCA) is a variant of the MHSA operation (1), where query tokens come from one expert ( $e_{1}$ ) and key and value tokens come from another expert ( $e_{2}$ ). This allows the experts to exchange information and benefit from the strengths of each other. We define the MHCA operation as follows:

$$
\operatorname{MHCA} \left(\mathbf {Y} _ {e _ {1}}, \mathbf {Y} _ {e _ {2}}\right) = \operatorname{Softmax} \left(\left(\mathbf {Y} _ {e _ {1}} \mathbf {W} _ {Q}\right) \left(\mathbf {Y} _ {e _ {2}} \mathbf {W} _ {K}\right) ^ {\top}\right) \left(\mathbf {Y} _ {e _ {2}} \mathbf {W} _ {V}\right), \tag {7}
$$

where $W_{Q}$ , $W_{K}$ , and $W_{V}$ are learnable query, key, and value parameter matrices respectively.

Temporal-to-Spatial Cross-Attention. In Temporal-to-Spatial (T2S) cross-attention, query tokens come from the spatial expert s, and key and value tokens come from the temporal expert t: MHCA( $Y_{s}^{(l)}, Y_{t}^{(l)}$ ). We depict the attention window in Figure 3 (c). Given a query, the model attends along the temporal dimension only. By using T2S cross-attention, the spatial expert can learn to attend to temporal features from the temporal expert. T2S MHCA leads to capturing spatio-temporal dependencies and improves the model performance in action recognition.

Spatial-to-Temporal Cross-Attention. In Spatial-to-Temporal (S2T) cross-attention, query tokens come from the temporal expert t, and key and value tokens come from the spatial expert s: $\mathrm{MHCA}(\mathbf{Y}_{t}^{(l)},\mathbf{Y}_{s}^{(l)})$ . We illustrate the attention window in Figure 3 (c). Given a query, the model attends along the spatial dimension only. By using S2T cross-attention, the temporal expert can attend to fine-grained spatial features from the spatial expert. S2T MHCA leads to a more balanced spatio-temporal understanding and improves the performance in fine-grained action recognition.

Bottleneck Cross-Attention in Space and Time. To achieve efficient and effective learning, we incorporate the T2S and S2T MHCA into bottleneck-shaped adapters. We illustrate B-CAST architecture in Figure 3 (b). We plug the MHCA modules into adapters and add new learnable positional embeddings for each MHCA. We define the B-CAST operation for T2S $\Phi_S(\cdot)$ as follows:

$$
\Phi_ {S} \left(\mathbf {Y} _ {s} ^ {(l)}, \mathbf {Y} _ {t} ^ {(l)}\right) = \sigma \left(\operatorname{MHCA} \left(\mathbf {E} _ {s} + \operatorname{LN} \left(\mathbf {Y} _ {s} ^ {(l)} \mathbf {W} _ {D, s}\right), \mathbf {E} _ {t} + \operatorname{LN} \left(\mathbf {Y} _ {t} ^ {(l)} \mathbf {W} _ {D, t}\right)\right)\right) \mathbf {W} _ {U, s}, \tag {8}
$$

where $W_{D,s}$ and $W_{U,s}$ are linear down- and up-projection matrices for the spatial expert, and $E_{s}$ and $E_{t}$ are new positional embeddings for the spatial and temporal experts, $\sigma(\cdot)$ is the GELU activation function, respectively. We can define the B-CAST operation for S2T, $\Phi_{T}(\cdot)$ in a similar manner. The output of B-CAST goes into a feed forward network using (5). Our empirical validation shows that the B-CAST architecture is efficient and effective. (See Table 1.)

![](images/9b2f01e0e74170ab61301384726dca3aaa7978caaa1156fe6afe679c8a99e597.jpg)

<details>
<summary>bar</summary>

| Metric | ST-Adapter | VideoMAE |
| :--- | :--- | :--- |
| EK100 Verb | 67.6 | 70.5 |
| EK100 Noun | 55.0 | 51.4 |
</details>

![](images/8581d132e53d81f05dc3c8e800b16cb7cac3c1a6cd66981f6d73f20dc1a48853.jpg)

<details>
<summary>bar</summary>

| Category | SSV2 | K400 |
|---|---|---|
| AIM | 68.1 | 84.5 |
| BEVT | 70.6 | 80.6 |
</details>

![](images/66d595636dcc53e1a87dde51314047e679200a057af4df639bc76ad0c93e9fd3.jpg)

<details>
<summary>bar</summary>

Harmonic Mean of EK100 Verb, Noun, SSV2, K400
| Method | Harmonic Mean |
|---|---|
| CLIP | 56.5 |
| VideoMAE | 66.6 |
| CAST | 71.6 |
</details>

Figure 4: Balanced spatio-temporal understanding performance. We visualize the action recognition accuracies of existing methods and the proposed method. (a) We show the Top-1 accuracies of ST-Adapter and VideoMAE on the EK100 verb and noun prediction tasks. (b) We show the Top-1 accuracies of AIM and BEVT on the SSV2, and K400. (c) For each method, we show the harmonic mean of Top-1 accuracies on the EK100 noun, EK100 verb, SSV2, and K400. CAST shows a more balanced spatio-temporal understanding capability compared to the existing methods. Best viewed with zoom and color.

# 4 Experimental Results

In this section, we present the experimental results that answer the following research questions: (1) Do existing methods show a balanced spatio-temporal understanding of videos? (Section 4.3) (2) What are the ingredients for a balanced spatio-temporal understanding? (Section 4.3) (3) Is the proposed method effective? (Section 4.3, Section 4.4) (4) How can we effectively combine spatial and temporal models to achieve such balance? (Section 4.5) (5) Does the proposed method outperform state-of-the-art methods in terms of balanced spatio-temporal understanding? (Section 4.6) To this end, we first provide details about the datasets and implementation in Section 4.1 and Section 4.2, respectively.

# 4.1 Datasets

Action recognition. We evaluate the CAST on two public datasets for conventional action recognition: Something-Something-V2 (SSV2) [19] and Kinetics-400 (K400) [24]. The SSV2 requires more temporal reasoning [3, 28] while the K400 is relatively static biased [32, 8, 50].

Fine-grained action recognition. We evaluate the CAST on the fine-grained action recognition task: EPIC-KITCHENS-100 (EK100) [10]. In contrast to conventional action recognition, EK100 defines an action as a combination of a verb and a noun. Therefore, we refer to the action recognition in EK100 as fine-grained action recognition. Since fine-grained action recognition requires correctly predicting both the verb and the noun to recognize an action it is more challenging than conventional action recognition, which requires predicting a single action label: e.g., K400 or SSV2.

# 4.2 Implementation details

In this section, we briefly provide our experimental setup and implementation details. Please refer to the Appendix § B for complete implementation details. We conduct all the experiments with 16 NVIDIA GeForce RTX 3090 GPUs. We implement CAST using PyTorch and build upon the existing codebase of VideoMAE [56].

Training. We sample 16 frames from each video to construct an input clip. For the K400 dataset, we apply dense sampling [15], while for SSV2 and EK100, we use uniform sampling [61]. We then perform random cropping and resizing every frame into $224 \times 224$ pixels. We use the AdamW [39] optimizer with momentum betas of (0.9, 0.999) [7] and a weight decay of 0.05. By default, we train the model for 50 epochs, with the cosine annealing learning rate scheduling [38] and a warm-up period of 5 epochs. The default base learning rate, layer decay [2], and drop path are set to 0.001, 0.8, and 0.2, respectively. We freeze all the parameters of each expert, except for the B-CAST layer, adapters, and the last layer normalization. We set the batch size per GPU as 6 with update frequency of 2.

Inference. Given an input video, we randomly sample frames multiple times to construct input clips with multiple temporal views with multiple spatial crops. After the temporal frame sampling, we resize every frame so that the shorter side has 224 pixels. Then we perform spatial cropping to get multiple $224 \times 224$ crops for each clip. We get the final prediction by averaging the predictions on (temporal views) $\times$ (spatial crops). For the K400 dataset, we use (5 clips) $\times$ (3 crops) views, while for the other datasets, we use (2 clips) $\times$ (3 crops) views for the inference.

![](images/a7188a3467f16b736b5461efb5465fc1cb2f913f86550369de14e8c0cc45f24d.jpg)

<details>
<summary>bar</summary>

EK100 Noun, overall improvement over CLIP: 8.3%
| Category | F1 Score improvement |
|---|---|
| materials | 0.155 |
| cutlery | 0.152 |
| utensils | 0.137 |
| vegetables | 0.132 |
| baked goods and grains | 0.126 |
| nuts and nuts | 0.124 |
| dairy and eggs | 0.122 |
| dairy | 0.080 |
| drinks | 0.075 |
| appliances | 0.060 |
| crockers | 0.055 |
| prepared food | 0.052 |
| cowware | 0.050 |
| spices and farms and sauces | 0.048 |
| furniture | 0.045 |
| hand | 0.042 |
| other | 0.038 |
| containers | 0.035 |
| cleaning equipment and material storage | 0.032 |
| meat and substitute | -0.005 |
| Meat and substitute | -0.025 |
</details>

![](images/6596d4f84ea93ff4e1c6579246933a8a7be59975fd222509613376252cb8c044.jpg)

<details>
<summary>bar</summary>

EK100 Noun, overall improvement over VideoMAE : 9.2%
| Category | F1 Score improvement |
|---|---|
| vegetables | 0.24 |
| other fruits and nuts | 0.235 |
| materials | 0.19 |
| cutlery, dairy and eggs | 0.18 |
| baked goods and grains | 0.165 |
| utensils | 0.15 |
| drinks, containers | 0.125 |
| spices and herbs and sauces | 0.115 |
| crookery | 0.09 |
| roxbish hand | 0.085 |
| cleaning equipment and material | 0.075 |
| appliances storage | 0.065 |
| meat and prepare | 0.055 |
| furniture | 0.045 |
| prepared food | 0.035 |
| Prepared food | 0.01 |
</details>

Figure 5: Improvements of CAST over each expert on EK100 noun classes. We show the super-category-wise weighted average F1 score improvement of CAST over each expert. (Left) Improvement over CLIP. CAST outperforms CLIP for every super-category except meat and substitute. (Right) Improvement over VideoMAE. CAST outperforms VideoMAE for every super-category except furniture and prepared food. Best viewed with zoom and color.

# 4.3 Balanced spatio-temporal understanding

In Figure 4 (a), we present the top-1 accuracies of several existing models. In the EK100 verb prediction task, VideoMAE outperforms ST-Adapter with a margin of 2.9 points (70.5% vs. 67.6%), while in the EK100 noun prediction task, ST-Adapter [42] outperforms VideoMAE [56] with a margin of 3.6 points (55.0% vs. 51.4%). As shown in Figure 4 (b), BEVT [62] outperforms AIM [72] with a margin of 2.5 points (70.6% vs. 68.1%) on the SSV2 dataset, while on the K400 dataset, AIM outperforms BEVT with a margin of 3.9 points (84.5% vs. 80.6%). We observe similar trends for other methods as well. Please refer to Section 4.6 for a detailed comparison. Our findings indicate that the performance of many existing models is significantly imbalanced toward either spatial or temporal understanding.

Ingredients for balanced spatio-temporal understanding. To achieve a more balanced spatio-temporal understanding, we can employ two expert models: a spatial expert and a temporal expert. For the spatial expert, we use CLIP $[44]$ , which has demonstrated impressive performance on various computer vision tasks. For the temporal expert, we use VideoMAE $[56]$ , which has shown favorable performance on temporal-biased tasks such as SSV2 and EK100 verb prediction tasks. (Please refer to Section 4.6 for the accuracy details.) While each expert is highly specialized in its own domain, we aim to create synergy between them by exchanging information to improve the balanced spatio-temporal understanding performance.

Effect of CAST. In Figure 4 (c), we gauge the balanced spatio-temporal understanding performance of our spatial expert, temporal expert, and CAST. For each method, we calculate the harmonic mean of top-1 accuracies for EK100 noun, EK100 verb, SSV2, and K400. The harmonic mean is an effective metric for gauging balanced performance because it gives more weight to lower-performing tasks. A higher harmonic mean value indicates that the performance over the different tasks is more balanced. Our spatial expert achieves an accuracy of 56.5%, while the temporal expert achieves an accuracy of 66.6%, and our CAST achieves an accuracy of 71.6%. These results validate the effectiveness of our proposed method, CAST, which allows our spatial and temporal experts to make synergistic predictions by exchanging information with each other through cross-attention.

# 4.4 Analysis on fine-grained action recognition

In this section, we provide a detailed analysis of how the proposed CAST improves the balanced spatio-temporal understanding in the fine-grained action recognition task: EK100.

Category-level performance analysis. In Figure 5, We present the EK100 noun super-category-wise weighted average F1 score improvement of CAST over our spatial expert (CLIP) and temporal expert (VideoMAE). In Figure 5 left, we observe that CAST significantly improves upon the spatial expert, CLIP, in several super-categories such as cutlery, utensils, and vegetables. These results indicate that the spatial expert achieves a more accurate understanding of fine-grained small objects interacting with the actors by leveraging the temporal context from the temporal expert. Similarly, in Figure 5 right, we observe that CAST significantly improves upon the temporal expert, VideoMAE, in several categories such as vegetables and cutlery. The trend is similar to the comparison with the spatial expert: CAST achieves more accurate understanding of fine-grained small objects by leveraging the fine-grained spatial context from CLIP.

Qualitative analysis. To better understand the effectiveness of CAST, we provide qualitative analysis on a few sample frames from the EK100 dataset in Figure 6. We show the predictions of CLIP, VideoMAE, and CAST. As expected, each expert model provides more accurate prediction in their respective tasks of expertise but shows weaker performance in the other task. In contrast,

Table 1: Ablation study. To validate the effect of each component, we show experimental results on the EPIC-Kitchens-100 dataset. In every experiment, we use the ViT-B/16 backbone for every expert. The best numbers are highlighted in gray.   
(a) Effect of information exchange. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>Indep. experts w/o adapter</td><td>70.7</td><td>50.1</td><td>40.0</td></tr><tr><td>Indep. experts w/ adapter</td><td>68.1</td><td>54.2</td><td>41.7</td></tr><tr><td>Ensemble of experts w/ adapters</td><td>68.2</td><td>55.3</td><td>42.9</td></tr><tr><td>CAST</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

(b) Different information exchange methods. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Late</td><td rowspan="2">Layer-wise</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>Add</td><td>√</td><td></td><td>68.9</td><td>56.6</td><td>44.2</td></tr><tr><td>Concat</td><td>√</td><td></td><td>69.2</td><td>56.4</td><td>44.5</td></tr><tr><td>Lateral</td><td></td><td>√</td><td>68.9</td><td>49.1</td><td>39.0</td></tr><tr><td>CAST</td><td></td><td>√</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

(c) B-CAST architecture. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Tune Param(M)</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>Identity</td><td>18.1</td><td>68.1</td><td>54.2</td><td>41.7</td></tr><tr><td>w/o adapter</td><td>85.9</td><td>69.3</td><td>49.4</td><td>39.4</td></tr><tr><td>X-attn.→adapter</td><td>93.0</td><td>71.3</td><td>60.1</td><td>47.9</td></tr><tr><td>B-CAST</td><td>44.8</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

(d) Effect of projection ratio. 

<table><tr><td rowspan="2">Ratio</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>1/8</td><td>70.7</td><td>59.9</td><td>47.4</td></tr><tr><td>1/4</td><td>71.3</td><td>59.8</td><td>47.4</td></tr><tr><td>1/2</td><td>72.5</td><td>60.3</td><td>48.7</td></tr><tr><td>1</td><td>72.1</td><td>59.8</td><td>48.6</td></tr></table>

(e) Effect of cross-attention window shape. 

<table><tr><td colspan="2">Window shape</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>T2S</td><td>S2T</td><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>space-time</td><td>space-time</td><td>71.0</td><td>59.3</td><td>47.2</td></tr><tr><td>space-time</td><td>space</td><td>71.9</td><td>60.3</td><td>48.4</td></tr><tr><td>space</td><td>space</td><td>72.3</td><td>60.2</td><td>48.5</td></tr><tr><td>time</td><td>space</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

(f) Effect of the number of cross-attention layers. 

<table><tr><td colspan="4">X-attention layer</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>1-3</td><td>4-6</td><td>7-9</td><td>10-12</td><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td></td><td></td><td></td><td>√</td><td>71.2</td><td>59.4</td><td>47.4</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td>71.3</td><td>59.9</td><td>47.9</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>71.8</td><td>60.0</td><td>48.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

(g) Effect of bi-directional cross-attention. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>Indep. experts w/ adapter</td><td>68.1</td><td>54.2</td><td>41.7</td></tr><tr><td>S2T only</td><td>71.2</td><td>55.0</td><td>43.7</td></tr><tr><td>T2S only</td><td>68.7</td><td>60.5</td><td>46.7</td></tr><tr><td>CAST</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

(h) Role of each expert. 

<table><tr><td colspan="2">Expert</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Spatial</td><td>Temporal</td><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>CLIP</td><td>CLIP</td><td>69.3</td><td>58.8</td><td>46.0</td></tr><tr><td>VideoMAE</td><td>CLIP</td><td>72.2</td><td>58.8</td><td>47.8</td></tr><tr><td>VideoMAE</td><td>VideoMAE</td><td>69.8</td><td>49.9</td><td>40.3</td></tr><tr><td>CLIP</td><td>VideoMAE</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

CAST consistently shows correct predictions for both noun and verb prediction tasks, such as spoon and open. The qualitative examples demonstrate the effectiveness of CAST in achieving balanced spatio-temporal understanding, which is essential for fine-grained action recognition.

# 4.5 Ablation study on CAST architecture

We conduct comprehensive ablation studies to examine the design choices for the proposed CAST architecture. Here we conduct all experiments on the EK100 $[10]$ dataset with 16-frame input videos and report the top-1 accuracy on the validation set. We employ CLIP $[44]$ as a spatial expert model and VideoMAE $[56]$ as a temporal expert model. For a fair ablation study, we use the same hyperparameters for each experiment unless explicitly mentioned.

Effect of information exchange. We investigate whether CAST effectively achieves a synergistic effect by exchanging information between the two expert models. In Table 1 (a), we compare CAST with three baselines. i) A baseline using two independent expert models without any information exchange (fully fine-tuned). ii) The same baseline as i), but we add adapters and fine-tune the adapters and head only, iii) A test-time ensemble of two independent experts (with adapters and heads fine-tuning only). The baselines predict nouns using the spatial model and verbs using the temporal model. We observe that the two expert models using ensembling achieve an improvement in Action accuracy by at least 1.2 points compared to the baselines without any information exchange. Furthermore, CAST achieves a best Action accuracy of 48.7%. These results suggest that information exchange is crucial for achieving balanced spatio-temporal understanding.

Comparison with simple information exchange baselines. We compare CAST with simple information exchange baselines: i) late fusion with addition, ii) late fusion with concatenation, iii) layer-wise fusion using the bidirectional lateral connection (element-wise addition) with linear projection. We fully fine-tune the two expert models without adapters in all three baselines, using our training recipe in Section 4.2. For the details of baseline fusion methods, please see Figure 7. We show the results in Table 1 (b). It is worth noting that both the late fusion and layer-wise lateral connection baselines result in a significant performance drop. Furthermore, we observe that layer-wise fusion without cross-attention yields inferior performance compared to the simple late fusion baselines. The results indicate that cross-attention in the bottleneck architecture is crucial for effective information exchange between spatial and temporal experts.

Design of B-CAST module. To explore the most effective and efficient way to integrate adapters for information exchange between the two expert models, we conduct an ablation study and present the results in Table 1 (c). For the details of the baselines, please see Figure 8. The first row of the table represents a baseline without the B-CAST module, which is equivalent to the identity function. Compared to this baseline, B-CAST achieves a significant improvement of 7.0 points in Action accuracy. The second row shows the performance of a baseline with cross-attention but without the

![](images/69e0545a673009911d5cc844b31a8503c7fcb0d1130d1072913c4f99f2e86e60.jpg)

<details>
<summary>text_image</summary>

Noun Classes
CLIP: Spoon ✓
VideoMAE: Fork ✗
CAST: Spoon ✓
CLIP: Cloth ✗
VideoMAE: Lid ✗
CAST: Fork ✓
CLIP: Pan ✗
VideoMAE: Pan ✗
CAST: Broccoli ✓
Verb Classes
CLIP: Trun-on ✗
VideoMAE: Open ✓
CAST: Open ✓
CLIP: Insert ✗
VideoMAE: Insert ✗
CAST: Scoop ✓
CLIP: Take ✗
VideoMAE: Sort ✗
CAST: Mix ✓
</details>

Figure 6: Qualitative examples from EK100 comparing CLIP, VideoMAE, and the proposed CAST. Each expert model shows more accurate predictions in their expertise but shows weaker performance on the other task. However, CAST consistently shows correct predictions for both tasks, demonstrating the effectiveness of the proposed spatio-temporal cross-attention mechanism.

bottleneck adapters. The 9.3-point gap between this baseline and B-CAST highlights the importance of bottleneck adapters for effective information exchange between the two expert models. The third row (X-attn.→adapter) is a baseline with the adapters after cross-attention. Compared to B-CAST, this baseline shows a 0.8 points drop in Action accuracy while having more than double the number of learnable parameters (44.9M vs. 93.0M). The results indicate that cross-attention in bottleneck is more effective and more efficient than the baseline. In summary, by placing cross-attention in the middle of the bottleneck adapter, B-CAST facilitates effective information exchange between the two experts and achieves a synergistic effect.

Effect of projection ratio in bottleneck. In this study, we investigate the impact of the down projection ratio in the bottleneck architecture presented in Table 1 (d). The results demonstrate that a ratio of 1/2 yields the best performance. Notably, a ratio of 1 results in inferior performance, which we attribute to overfitting caused by the addition of more parameters.

Effect of cross-attention window shape. We investigate the impact of the window shape in the cross-attention mechanism in the T2S and S2T modules in Table 1 (e). Please refer to Figure 3 (c) for the details of the window size. We maintain the same model capacity across different methods. Using space-time attention for both T2S and S2T modules results in the worst performance. We conjecture that learning joint space-time attention is challenging with the given model capacity [3]. On the other hand, using time attention in T2S and space attention in S2T yields the best performance. Consequently, we adopt this configuration throughout the paper.

Effect of the number of cross-attention layers. We investigate the impact of the number of cross-attention layers used. To this end, we gradually increase the number of cross-attention layers starting from the top three layers, and report the results in Table 1 (f). As we increase the number of cross-attention layers, we observe a corresponding improvement in performance, as expected.

Effect of bi-directional cross-attention. To validate the effectiveness of bi-directional information exchange, we ablate each cross attention at a time. We compare CAST with unidirectional information exchange baselines equipped with S2T or T2S cross-attention only. Each unidirectional information exchange baseline still has both experts. In Table 1 (g), compared to our CAST (48.7%), the S2T only baseline shows 5.0 points drop (43.7%) and the T2S only baseline shows 2.0 points drop (46.7%) in accuracy. The results validate the effectiveness of the proposed bi-directional cross-attention.

Role of each expert. In Table 1 (h), we investigate the role of experts within CAST by controlling the assignment of models to each expert. We observe that we can achieve the best performance of 48.7% when we employ CLIP as our spatial expert and VideoMAE as our temporal expert. When we employ one VideoMAE as our spatial expert and another VideoMAE as our temporal expert, we obtain 40.3% accuracy. When we employ one CLIP as our spatial expert and another CLIP as our temporal expert, we obtain 46.0% accuracy.

Interestingly, when we revert the role of CLIP and VideoMAE, i.e., we employ VideoMAE as the spatial and CLIP as the temporal expert, we achieve a good performance of 47.8%. The results demonstrate that the B-CAST architecture facilitates effective information exchange between the two experts. Through the stacked B-CAST, the experts can learn high-quality spatio-temporal representations by exchanging information, even when the roles are reverted.

In summary, these findings suggest that CAST achieves optimal performance when models are assigned to expert roles that align with their strengths. CLIP serves as an effective spatial expert, whereas VideoMAE is more effective as a temporal expert. The B-CAST architecture encourages these experts to leverage their respective strengths through information exchange, resulting in enhanced spatio-temporal balanced understanding.

Table 2: Comparison with the state-of-the-arts on the EK100, SSV2 and K400 datasets. We show the Top-1 accuracy on each dataset and the harmonic mean (H.M.) of the Top-1 accuracies. The best performance is in bold and the second best is underscored. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">GFLOPs/View</td><td colspan="3">EK100 Top-1</td><td colspan="3">SSV2 &amp; K400 Top-1</td><td>All</td></tr><tr><td>Verb</td><td>Noun</td><td>Act.</td><td>SSV2</td><td>K400</td><td>H.M.</td><td>H.M.</td></tr><tr><td>CLIP* [44]</td><td>140</td><td>54.9</td><td>52.7</td><td>33.8</td><td>47.8</td><td>78.9</td><td>59.5</td><td>56.5</td></tr><tr><td>EVL [35]</td><td>592</td><td>-</td><td>-</td><td>-</td><td>62.4</td><td>82.9</td><td>71.2</td><td>-</td></tr><tr><td>ST-Adapter [42]</td><td>607</td><td>67.6</td><td>55.0</td><td>-</td><td>69.5</td><td>82.7</td><td>75.5</td><td>67.3</td></tr><tr><td>AIM [72]</td><td>404</td><td>64.8</td><td>55.5</td><td>41.3*</td><td>68.1</td><td>84.5</td><td>75.4</td><td>66.7</td></tr><tr><td>MBT [40]</td><td>936</td><td>64.8</td><td>58.0</td><td>43.4</td><td>-</td><td>80.8</td><td>-</td><td>-</td></tr><tr><td>ViViT FE [1]</td><td>990</td><td>66.4</td><td>56.8</td><td>44.0</td><td>65.9</td><td>81.7</td><td>73.0</td><td>66.6</td></tr><tr><td>TimeSformer [3]</td><td>2380</td><td>-</td><td>-</td><td>-</td><td>62.4</td><td>80.7</td><td>70.4</td><td>-</td></tr><tr><td>MViT [12]</td><td>170</td><td>-</td><td>-</td><td>-</td><td>67.7</td><td>80.2</td><td>73.4</td><td>-</td></tr><tr><td>MFormer [43]</td><td>1185</td><td>67.1</td><td>57.6</td><td>44.1</td><td>68.1</td><td>80.2</td><td>73.7</td><td>67.3</td></tr><tr><td>ORViT MF [21]</td><td>-</td><td>68.4</td><td>58.7</td><td>45.7</td><td>67.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Video Swin [37]</td><td>282</td><td>-</td><td>-</td><td>-</td><td>69.6</td><td>82.7</td><td>75.8</td><td>-</td></tr><tr><td>BEVT [62]</td><td>282</td><td>-</td><td>-</td><td>-</td><td>70.6</td><td>80.6</td><td>75.3</td><td>-</td></tr><tr><td>VideoMAE [56]</td><td>180</td><td>70.5</td><td>51.4</td><td>41.7*</td><td>70.8</td><td>81.5</td><td>75.8</td><td>66.6</td></tr><tr><td>MeMViT [68]</td><td>59</td><td>70.6</td><td>58.5</td><td>46.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OMNIVORE [16]</td><td>-</td><td>69.5</td><td>61.7</td><td>49.9</td><td>71.4</td><td>84.0</td><td>77.2</td><td>70.8</td></tr><tr><td>MTV-HR [71]</td><td>930</td><td>68.0</td><td>63.1</td><td>48.6</td><td>68.5</td><td>82.4</td><td>74.8</td><td>69.8</td></tr><tr><td>CAST</td><td>391</td><td>72.5</td><td>60.9</td><td>49.3</td><td>71.6</td><td>85.3</td><td>77.9</td><td>71.6</td></tr></table>

\*We conduct experiments with our own implementation.

# 4.6 Comparison with state-of-the-art

In this section, we evaluate the performance of CAST and state-of-the-art methods in terms of balanced spatio-temporal understanding on multiple datasets, as shown in Table 2. For each method, in addition to reporting the top-1 accuracy of each task, we report the harmonic mean of top-1 accuracies for i) SSV2, and K400, and ii) EK100 verb, EK100 noun, SSV2, and K400. For comparison with state-of-the-art models, we have set different hyperparameters than those used in our ablation study. Please refer to Table 4 for the details. For fair comparisons of the computation complexity, we show the GFLOPs/View. In cases where a compared method shows various GFLOPs/View depending on the dataset, we specifically note the lowest GFLOPs/View value for reference. For more detailed comparison of computation complexity, please refer to the Appendix § E.

We observe that among the CLIP-based methods (the second group in Table 2), AIM [72] achieves favorable performance on the static-biased K400 dataset, with $84.5\%$ accuracy. However, AIM shows a relatively lower performance of $68.1\%$ on the temporal-biased SSV2. On the other hand, VideoMAE [56], one of the state-of-the-art methods, shows $70.8\%$ accuracy on the SSV2 dataset, which is more competitive than AIM. However, VideoMAE shows a lower accuracy of $81.5\%$ on the K400 dataset, less competitive than AIM. Our proposed method, CAST, demonstrates favorable performance on both the SSV2 $(71.6\%)$ and K400 $(85.3\%)$ datasets, resulting in a harmonic mean of $77.9\%$ , which is higher than that of AIM $(75.4\%)$ and VideoMAE $(75.8\%)$ . CAST shows a more balanced spatio-temporal understanding than the existing methods. Additionally, CAST shows favorable performance in fine-grained action recognition on the EK100 dataset. CAST achieves a competitive Action accuracy of $49.3\%$ , which is the second best among the compared methods.

In terms of the overall harmonic mean of EK100 verb, EK100 noun, SSV2, and K400 accuracies, CAST shows the best performance of 71.6%. The results highlight the effectiveness of CAST. By exchanging information between spatial and temporal experts, our CAST shows a favorable balanced spatio-temporal understanding performance.

# 5 Conclusions

In this paper, we present a solution to the problem of action recognition models lacking a balanced spatio-temporal understanding of videos. The proposed method, CAST, incorporates a spatial expert and a temporal expert that exchange information through cross-attention to achieve synergistic predictions. Our extensive experiments on datasets with varying characteristics demonstrate that CAST outperforms both individual expert models and existing methods in terms of a balanced spatio-temporal understanding measure: the harmonic mean of accuracies on the datasets. The results highlight the effectiveness of CAST in achieving a balanced spatio-temporal understanding of videos, and suggest that CAST could have broad applicability in the field of video understanding.

Acknowledgment. This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University)); by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea Government (MSIT) (Artificial Intelligence Innovation Hub) under Grant 2021-0-02068; by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. 2022R1F1A1070997).

# References

[1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video vision transformer. In ICCV, 2021.   
[2] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.   
[3] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, 2021.   
[4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.   
[5] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.   
[6] Chun-Fu Richard Chen, Quanfu Fan, and Rameswar Panda. Crossvit: Cross-attention multi-scale vision transformer for image classification. In ICCV, 2021.   
[7] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020.   
[8] Jinwoo Choi, Chen Gao, Joseph CE Messou, and Jia-Bin Huang. Why can't i dance in the mall? learning to mitigate scene bias in action recognition. In NeurIPS, 2019.   
[9] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In CVPR workshops, 2020.   
[10] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, , Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision. IJCV, 130(1):33–55, 2022.   
[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.   
[12] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In ICCV, 2021.   
[13] Christoph Feichtenhofer. X3d: Expanding architectures for efficient video recognition. In CVPR, 2020.   
[14] Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman. Convolutional two-stream network fusion for video action recognition. In CVPR, 2016.   
[15] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, 2019.   
[16] Rohit Girdhar, Mannat Singh, Nikhila Ravi, Laurens van der Maaten, Armand Joulin, and Ishan Misra. Omnivore: A Single Model for Many Visual Modalities. In CVPR, 2022.   
[17] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. arXiv preprint arXiv:2305.05665, 2023.   
[18] Satya Krishna Gorti, Noël Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, and Guangwei Yu. X-pool: Cross-modal language-video attention for text-video retrieval. In CVPR, 2022.   
[19] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The "something something" video database for learning and evaluating visual common sense. In ICCV, 2017.

[20] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.   
[21] Roei Herzig, Elad Ben-Avraham, Karttikeya Mangalam, Amir Bar, Gal Chechik, Anna Rohrbach, Trevor Darrell, and Amir Globerson. Object-region video transformers. In CVPR, 2022.   
[22] Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment your batch: Improving generalization through instance repetition. In CVPR, 2020.   
[23] Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder. Compacter: Efficient low-rank hypercomplex adapter layers. In NeurIPS, 2021.   
[24] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.   
[25] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2019.   
[26] Hannah Halin Kim, Shuzhi Yu, Shuai Yuan, and Carlo Tomasi. Cross-attention transformer for video interpolation. In ACCV, 2022.   
[27] Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Mingxing Tan, Matthew Brown, and Boqing Gong. Movinets: Mobile video networks for efficient video recognition. In CVPR, 2021.   
[28] Matthew Kowal, Mennatullah Siam, Md Amirul Islam, Neil DB Bruce, Richard P Wildes, and Konstantinos G Derpanis. A deeper dive into what deep spatiotemporal networks encode: Quantifying static vs. dynamic information. In CVPR, 2022.   
[29] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In EMNLP, 2021.   
[30] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. NeurIPS, 2021.   
[31] Kunchang Li, Yali Wang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatiotemporal representation learning. In ICLR, 2022.   
[32] Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In ECCV, 2018.   
[33] Ji Lin, Chuang Gan, and Song Han. Tsm: Temporal shift module for efficient video understanding. In ICCV, 2019.   
[34] Yan-Bo Lin, Jie Lei, Mohit Bansal, and Gedas Bertasius. Eclipse: Efficient long-range video retrieval using sight and sound. In ECCV, 2022.   
[35] Ziyi Lin, Shijie Geng, Renrui Zhang, Peng Gao, Gerard de Melo, Xiaogang Wang, Jifeng Dai, Yu Qiao, and Hongsheng Li. Frozen clip models are efficient video learners. In ECCV, 2022.   
[36] Yen-Cheng Liu, Chih-Yao Ma, Junjiao Tian, Zijian He, and Zsolt Kira. Polyhistor: Parameter-efficient multi-task adaptation for dense vision tasks. In NeurIPS, 2022.   
[37] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In CVPR, 2022.   
[38] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.   
[39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.   
[40] Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, and Chen Sun. Attention bottlenecks for multimodal fusion. In NeurIPS, 2021.   
[41] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. In ECCV, 2022.   
[42] Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, and Hongsheng Li. St-adapter: Parameter-efficient image-to-video transfer learning. In NeurIPS, 2022.

[43] Mandela Patrick, Dylan Campbell, Yuki M. Asano, Ishan Misra Florian Metze, Christoph Feichtenhofer, Andrea Vedaldi, and João F. Henriques. Keeping your eye on the ball: Trajectory attention in video transformers. In NeurIPS, 2021.   
[44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.   
[45] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.   
[46] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. In NeurIPS, 2017.   
[47] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Efficient parametrization of multi-domain deep neural networks. In CVPR, 2018.   
[48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.   
[49] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.   
[50] Laura Sevilla-Lara, Shengxin Zha, Zhicheng Yan, Vedanuj Goswami, Matt Feiszli, and Lorenzo Torresani. Only time can tell: Discovering temporal data for temporal modeling. In WACV, 2021.   
[51] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. In NeurIPS, 2014.   
[52] Swathikiran Sudhakaran, Sergio Escalera, and Oswald Lanz. Gate-shift-fuse for video action recognition. arXiv:2203.08897, 2022.   
[53] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.   
[54] Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. Lst: Ladder side-tuning for parameter and memory efficient transfer learning. In NeurIPS, 2022.   
[55] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.   
[56] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS, 2022.   
[57] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
[58] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In ICCV, 2015.   
[59] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In CVPR, 2018.   
[60] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.   
[61] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks for action recognition in videos. TPAMI, 41(11):2740–2755, 2018.   
[62] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, and Lu Yuan. Bevt: Bert pretraining of video transformers. In CVPR, 2022.   
[63] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, and Yu-Gang Jiang. Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning. In CVPR, 2023.   
[64] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In CVPR, 2018.

[65] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.   
[66] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In CVPR, 2022.   
[67] Xi Wei, Tianzhu Zhang, Yan Li, Yongdong Zhang, and Feng Wu. Multi-modality cross attention network for image and sentence matching. In CVPR, 2020.   
[68] Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Memvit: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In CVPR, 2022.   
[69] Wenhao Wu, Zhun Sun, and Wanli Ouyang. Revisiting classifier: Transferring vision-language models for video recognition. In AAAI, 2023.   
[70] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning for video understanding. In ECCV, 2018.   
[71] Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, and Cordelia Schmid. Multiview transformers for video recognition. In CVPR, 2022.   
[72] Taojiannan Yang, Yi Zhu, Yusheng Xie, Aston Zhang, Chen Chen, and Mu Li. Aim: Adapting image models for efficient video understanding. In ICLR, 2023.   
[73] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In ICLR, 2017.   
[74] Bolei Zhou, Alex Andonian, Aude Oliva, and Antonio Torralba. Temporal relational reasoning in videos. In ECCV, 2018.   
[75] Haowei Zhu, Wenjing Ke, Dong Li, Ji Liu, Lu Tian, and Yi Shan. Dual cross-attention learning for fine-grained visual categorization and object re-identification. In CVPR, 2022.

# Appendix

In this appendix, we provide additional architecture/implementation/dataset details, experimental settings, quantitative/qualitative results, limitations of our method, and broader impact of our method to complement the main paper. We organize the appendix as follows:

A. Architecture details of our framework   
B. Implementation details and experimental settings   
C. Details of datasets in our experiments   
D. Additional quantitative and qualitative results   
E. Comparison with State-of-the-Art with additional information   
F. Class-wise F1 score on EK100 verb and noun classes   
G. Additional qualitative analysis on EK100   
H. Limitations   
I. Broader impacts

# A Architecture Details

In this section, we provide details of our B-CAST architecture. Let us assume we employ CLIP [44] as a spatial expert and VideoMAE [56] as a temporal expert.

Table 3: Stage-wise details of the two experts in B-CAST. We provide a detailed description of each operation performed in each expert. The input for this example is a video consisting of 16 frames. MHCA represents multi-head cross-attention applied with a specific window shape: either time-attention or space-attention. In the description, B, D, T, and N represent the batch size, embedding dimension, temporal sequence length, and spatial sequence length, respectively. We omit the Layer Normalization and activation functions for simplicity. 

<table><tr><td></td><td colspan="2">Spatial Expert</td><td colspan="2">Temporal Expert</td></tr><tr><td>B-CAST Stage</td><td>Remark</td><td>Output Tensor Shape</td><td>Remark</td><td>Output Tensor Shape</td></tr><tr><td>Up Projection</td><td>Linear projection with ratio = 2.0</td><td> $Y_s:(196+1)×B·8×768$ </td><td>Linear projection with ratio = 2.0</td><td> $Y_t:B×8·196×768$ </td></tr><tr><td>Post Processing</td><td>Attach CLS token of  $Y_s$  Reshape: N×B·T×D</td><td> $Y_s:(196+1)×B·8×384$ </td><td>Reshape: B×T·N×D</td><td> $Y_t:B×8·196×384$ </td></tr><tr><td>Cross-Attention</td><td>T2S MHCA( $Y_s$ , $Y_t$ ) Window shape: time</td><td> $Y_s:B·196×8×384$ </td><td>S2T MHCA( $Y_t$ , $Y_s$ ) Window shape: space</td><td> $Y_t:B·8×196×384$ </td></tr><tr><td>Positional Embeddings</td><td># parameters: 8×384</td><td> $Y_s:B·196×8×384$   $Y_t:B·196×8×384$ </td><td># parameters: 196×384</td><td> $Y_t:B·8×196×384$   $Y_s:B·8×196×384$ </td></tr><tr><td>Pre processing</td><td>Detach CLS token of  $Y_s$  Reshape: B·N×T×D</td><td> $Y_s:B·196×8×384$   $Y_t:B·196×8×384$ </td><td>Detach CLS token of  $Y_s$  Reshape: B·T×N×D</td><td> $Y_t:B·8×196×384$   $Y_s:B·8×196×384$ </td></tr><tr><td>Gather Features</td><td>Gather  $Y_t$  from Temporal Expert</td><td> $Y_s:(196+1)×B·8×384$   $Y_t:B×8·196×384$ </td><td>Gather  $Y_s$  from Spatial Expert</td><td> $Y_t:B×8·196×384$   $Y_s:(196+1)×B·8×384$ </td></tr><tr><td>Down Projection</td><td>Linear projection with ratio = 0.5</td><td> $Y_s:(196+1)×B·8×384$ </td><td>Linear projection with ratio = 0.5</td><td> $Y_t:B×8·196×384$ </td></tr><tr><td>Input of B-CAST</td><td>-</td><td> $Y_s:(196+1)×B·8×768$ </td><td>-</td><td> $Y_t:B×8·196×768$ </td></tr></table>

B-CAST architecture. In Table 3, we provide stage-wise details of the two experts in B-CAST. Given an input from multi-head self-attention (MHSA) layer, we first pass it through the linear projection layer of each expert. Subsequently, the two experts exchanges their features each other. The multi-head cross-attention (MHCA) layer of each expert enables the effectively exchange of information between the two experts. Afterward, we pass the output tensors from the B-CAST

through the Feed-Forward Network (FFN). We repeat this process in a stacked manner, consisting of 12 blocks, each comprising the MHSA module along with adapters, B-CAST, and FFN module along with adapters. Finally, we feed the resulting tensors to a classification head to predict action.

Architecture details of different information exchange methods. In Figure 7, we visualize the architectures of the simple information exchange baselines presented in Table 1 (b) of the main paper. These baselines involve fully fine-tuning the two expert models without the use of adapters, following the training recipe outlined in Appendix § B. In Figure 7 (a), we present the add baseline, which facilitates information exchange between the spatial and temporal experts through late fusion using element-wise addition. In Figure 7 (b), we present the concat baseline, which exchanges the information between the spatial and the temporal experts through late fusion using concatenation. In Figure 7 (c), we show the lateral connection baseline, which enables information exchange between the spatial and temporal experts through layer-wise lateral connections, incorporating linear projections.

![](images/afe3d96b589ffab3d37dfb16d64ec7772c5f8e37bb5debdfd666bd00ffe4c180.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Classification Head"] --> B["Add (Late fusion)"]
    C["Classification Head"] --> B
    D["Spatial Expert"] --> B
    E["Temporal Expert"] --> B
    B --> F["Noun"]
    B --> G["Verb"]
    H["CLS Token"] --> I["Red dot"]
    J["GAP Token"] --> K["Blue dot"]
    L["(a) Add"] --> M["Output"]
```
</details>

![](images/b12f67afcf703d00f6fcf8a26e101283c96ddb9b84f2eab044963856ac237993.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Classification Head"] --> B["Down Projection"]
    C["Classification Head"] --> B
    D["Verb"] --> B
    B --> E["Concatenate (Late fusion)"]
    E --> F["Spatial Expert"]
    E --> G["Temporal Expert"]
    F --> H["(b) Concat"]
    G --> H
    I["Noun"] --> A
    J["Verb"] --> C
    K["CLS Token R^D"] --> F
    L["GAP Token R^D"] --> G
```
</details>

![](images/4c13845235c39d9666f4c1d5a41d0d7912cfe39496f10b480a47020124d788ae.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph "Noun"
        A["Classification Head"] --> B["CLS Token"]
        B --> C["Spatial Expert x12 Blocks"]
        C --> D["Feed Forward Network"]
        D --> E["Multi-Head Self-Attention"]
        E --> F["+"]
    end

    subgraph "Verb"
        G["Classification Head"] --> H["GAP Token"]
        H --> I["Temporal Expert x12 Blocks"]
        I --> J["Feed Forward Network"]
        J --> K["Multi-Head Self-Attention"]
        K --> L["+"]
    end

    C --> M["Linear"]
    M --> N["Linear"]
    N --> O["+"]
    O --> P["Multi-Head Self-Attention"]
    P --> Q["Output"]
    style A fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#ccf,stroke:#333
    style N fill:#ccf,stroke:#333
    style O fill:#ccf,stroke:#333
    style P fill:#ccf,stroke:#333
```
</details>

Figure 7: Architecture visualization of the different information exchange baselines.

Architecture details of the baselines in the B-CAST architecture ablation study. In Figure 8, we visualize the architectures of the baselines used in the B-CAST architecture ablation study, as presented in Table 1 (c) of the main paper. In Figure 8 (a), we illustrate the identity baseline, which serves as a baseline without the B-CAST module. In Figure 8 (b), we present the w/o adapter baseline, which includes cross-attention but does not incorporate the bottleneck adapters. In Figure 8 (c), we show the X-attn.→adapter baseline, which uses adapters positioned after the cross-attention stage. For reference, we include our B-CAST architecture in Figure 8 (d), which represents the final model used in our study.

![](images/66b3e3c9273985052b21b0ef60c7b5d0e897390075a7a4888222611f71e5cdd1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph "(a) Identity"
        A1["Identity"] --> A2["Identity"]
        A2 --> A3["Temporal to Spatial T2S Cross-Attention"]
        A3 --> A4["Q K V"]
        A4 --> A5["LayerNorm"]
        A5 --> A6["Q K V"]
        A6 --> A7["Up Projection"]
        A7 --> A8["Down Projection"]
        A8 --> A9["Temporal to Spatial T2S Cross-Attention"]
        A9 --> A10["Q K V"]
        A10 --> A11["LayerNorm"]
        A11 --> A12["Q K V"]
        A12 --> A13["Up Projection"]
        A13 --> A14["Down Projection"]
        A14 --> A15["Spatial to Temporal S2T Cross-Attention"]
        A15 --> A16["Q K V"]
        A16 --> A17["LayerNorm"]
        A17 --> A18["Q K V"]
        A18 --> A19["Up Projection"]
        A19 --> A20["Down Projection"]
        A20 --> A21["Spatial to Temporal S2T Cross-Attention"]
        A21 --> A22["Q K V"]
        A22 --> A23["LayerNorm"]
        A23 --> A24["Down Projection"]
    end

    subgraph "(b) W/o adapter"
        B1["W/o adapter"] --> B2["W/o adapter"]
        B2 --> B3["W/o adapter"]
        B3 --> B4["W/o adapter"]
        B4 --> B5["W/o adapter"]
        B5 --> B6["W/o adapter"]
        B6 --> B7["W/o adapter"]
        B7 --> B8["W/o adapter"]
        B8 --> B9["W/o adapter"]
        B9 --> B10["W/o adapter"]
        B10 --> B11["W/o adapter"]
        B11 --> B12["W/o adapter"]
        B12 --> B13["W/o adapter"]
        B13 --> B14["W/o adapter"]
        B14 --> B15["W/o adapter"]
        B15 --> B16["W/o adapter"]
        B16 --> B17["W/o adapter"]
        B17 --> B18["W/o adapter"]
        B18 --> B19["W/o adapter"]
        B19 --> B20["W/o adapter"]
    end

    subgraph "(c) X-attn.→adapter"
        C1["X-attn.→adapter"] --> C2["X-attn.→adapter"]
        C2 --> C3["X-attn.→adapter"]
        C3 --> C4["X-attn.→adapter"]
        C4 --> C5["X-attn.→adapter"]
        C5 --> C6["X-attn.→adapter"]
        C6 --> C7["X-attn.→adapter"]
        C7 --> C8["X-attn.→adapter"]
        C8 --> C9["X-attn.→adapter"]
        C9 --> C10["X-attn.→adapter"]
        C10 --> C11["X-attn.→adapter"]
        C11 --> C12["X-attn.→adapter"]
        C12 --> C13["X-attn.→adapter"]
        C13 --> C14["X-attn.→adapter"]
        C14 --> C15["X-attn.→adapter"]
        C15 --> C16["X-attn.→adapter"]
        C16 --> C17["X-attn.→adapter"]
        C17 --> C18["X-attn.→adapter"]
        C18 --> C19["X-attn.→adapter"]
        C19 --> C20["X-attn.→adapter"]
    end

    subgraph "(d) B-CAST"
        D1["B-CAST"] --> D2["B-CAST"] --> D3["B-CAST"] --> D4["B-CAST"] --> D5["B-CAST"]
    end
```
</details>

Figure 8: Architecture visualization of the baselines used in the B-CAST ablation study.

Classification head. We use different classification strategies for conventional action recognition and fine-grained action recognition as shown in Figure 9. i) For conventional action recognition datasets, i.e., Kinetics-400 (K400) and Something-Something-V2 (SSV2), CAST combines the CLS and GAP tokens from the two experts to predict actions, as depicted in Figure 9 (a). The spatial expert, CLIP [44], generates one CLS token for each frame. To obtain a single CLS token,

![](images/950dc82c66f4a6e24d5a6ddff52660ca034cbe8dc8d4d6e736d2c691e664f630.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Action"] --> B["Classification Head"]
    B --> C["Adapter"]
    B --> D["Adapter"]
    C --> E["CLS Token R^D"]
    D --> F["GAP Token R^D"]
    E --> G["Average"]
    F --> H["Average"]
    G --> I["Spatial Path ×12 Blocks"]
    H --> J["Temporal Path ×12 Blocks"]
    I --> K["R^T×(N+1)×D"]
    J --> L["R^N-T×D"]
    K --> M["+"]
    L --> M
```
</details>

(a) Conventional action recognition

![](images/fe433b09e0b80c89354abf37d33ffd36de7a882fa87cc2903eb7feb434810b7d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Action"] --> B["Noun"]
    B --> C["Classification Head"]
    C --> D["CLS Token"]
    D --> E["Average"]
    E --> F["Spatial Path ×12 Blocks"]
    F --> G["Temporal Path ×12 Blocks"]
    G --> H["Classification Head"]
    H --> I["GAP Token"]
    I --> J["Average"]
    J --> K["Temporal Path ×12 Blocks"]
    K --> L["Classification Head"]
    L --> M["CLS Token"]
    M --> N["Average"]
    N --> O["Temporal Path ×12 Blocks"]
    O --> P["Classification Head"]
    P --> Q["GAP Token"]
    Q --> R["Average"]
    R --> S["Temporal Path ×12 Blocks"]
    S --> T["Classification Head"]
    T --> U["CLS Token"]
    U --> V["Average"]
    V --> W["Temporal Path ×12 Blocks"]
    W --> X["Classification Head"]
    X --> Y["GAP Token"]
    Y --> Z["Average"]
    Z --> AA["Temporal Path ×12 Blocks"]
    AA --> AB["Classification Head"]
    AB --> AC["GAP Token"]
    AC --> AD["Average"]
    AD --> AE["Temporal Path ×12 Blocks"]
    AE --> AF["Classification Head"]
    AF --> AG["GAP Token"]
    AG --> AH["Average"]
    AH --> AI["Temporal Path ×12 Blocks"]
    AI --> AJ["Classification Head"]
    AJ --> AK["GAP Token"]
    AK --> AL["Average"]
    AL --> AM["Temporal Path ×12 Blocks"]
    AM --> AN["GAP Token"]
    AN --> AO["Average"]
    AO --> AP["Temporal Path ×12 Blocks"]
    AP --> AQ["GAP Token"]
    AQ --> AR["Average"]
    AR --> AS["Temporal Path ×12 Blocks"]
    AS --> AT["GAP Token"]
    AT --> AU["Average"]
    AU --> AV["Temporal Path ×12 Blocks"]
    AV --> AW["GAP Token"]
    AW --> AX["Average"]
    AX --> AY["GAP Token"]
    AY --> AZ["Average"]
    AZ --> BA["GAP Token"]
    BA --> BB["Average"]
    BB --> BC["GAP Token"]
    BC --> BD["Average"]
    BD --> BE["GAP Token"]
    BE --> BF["Average"]
    BF --> BG["GAP Token"]
    BG --> BH["Average"]
    BH --> BI["GAP Token"]
    BI --> BJ["Average"]
    BJ --> BK["GAP Token"]
    BK --> BL["GAP Token"]
```
</details>

(b) Fine-grained action recognition   
Figure 9: Classification head architecture. We use different classification strategies for conventional action recognition shown in (a) and fine-grained action recognition shown in (b). T, N, and D denote the temporal sequence length, spatial sequence length, and embedding dimension respectively. The output from the spatial expert consists of frame-level feature vectors. Each frame-level feature vector consists of N patches and one CLS token. The output of the temporal expert is a video-level feature vector, consisting of $N \cdot T$ patches.

we take the average of the CLS tokens from all frames of the input video. The temporal expert, VideoMAE $[56]$ , performs global average pooling on all output features to obtain a single GAP token. After passing through adapters, we add the CLS and GAP tokens together and feed the token into a linear classification head. ii) In a fine-grained action recognition dataset, i.e., EPIC-KITCHENS-100 (EK100), where a model needs to predict both verb and noun, we use two separate classification heads instead of a single head shown in Figure 9 (b). Specifically, we feed the CLS token from the spatial expert into a linear classification head for noun prediction and the GAP token from the temporal expert into another linear classification head for verb prediction.

# B Implementation Details

In this section, we provide more details of our experimental setup and implementation of each dataset. We conduct the experiments with 16 NVIDIA GeForce RTX 3090 GPUs. We implement CAST using PyTorch and build upon the existing codebase of VideoMAE [56].

Data preprocessing. After sampling the videos to 16 frames, We randomly crop each frame of the video and resize it to $224 \times 224$ . We also apply data augmentation techniques, including mixup [73], label smoothing [55], horizontal flip, color jitter, and randaugment [9], repeated augmentation [22] to diversify the training data. We do not use horizontal flip on SSV2. Note that if the patch embedding layer of the temporal expert has a time stride value of 2, e.g., VideoMAE [56], we only take even frames for the spatial pathway. After the patch embedding layers, both experts take an input of 3 channels $\times$ 8 frames $\times$ 224 width $\times$ 224 height. We use the same data preprocessing protocol in all the experiments.

Model training. We conduct experiments using 2 nodes, each equipped with 8 GPUs. To ensure efficient multi-node training, we utilize the DeepSpeed $^{3}$ library. Additionally, we increase the effective batch size by implementing gradient accumulation to update the model weights. For the EK100 dataset, we set the update frequency to 4 iterations $^{4}$ , resulting in a total batch size of 24

Table 4: Hyperparameters used and the fine-tuning configuration for each dataset. 

<table><tr><td>Config</td><td>EK100</td><td>K400</td><td>SSV2</td></tr><tr><td>Optimizer</td><td>AdamW [39]</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Base learning rate</td><td>1e-3</td><td>1e-3</td><td>1e-3</td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Optimizer momentum [7]</td><td> $\beta_1, \beta_2=0.9, 0.999$ </td><td> $\beta_1, \beta_2=0.9, 0.999$ </td><td> $\beta_1, \beta_2=0.9, 0.999$ </td></tr><tr><td>Gpu per batch size</td><td>6</td><td>6</td><td>6</td></tr><tr><td>Update frequency</td><td>4</td><td>6</td><td>4</td></tr><tr><td>Learning rate schedule</td><td>cosine decay [38]</td><td>cosine decay</td><td>Cosine decay</td></tr><tr><td>Warmup epochs</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Training epochs</td><td>50</td><td>70</td><td>50</td></tr><tr><td>Flip augmentation</td><td>yes</td><td>yes</td><td>no</td></tr><tr><td>Color jitter</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td>RandAug [9]</td><td>(9, 0.5)</td><td>(9, 0.5)</td><td>(9, 0.5)</td></tr><tr><td>Label smoothing [55]</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Mixup [73]</td><td>0.8</td><td>0.8</td><td>0.8</td></tr><tr><td>Drop path</td><td>0.2</td><td>0.2</td><td>0.3</td></tr><tr><td>Layer-wise lr decay [2]</td><td>0.75</td><td>0.75</td><td>0.75</td></tr></table>

per GPU. We linearly scale the base learning rate, then actual lr = base lr × total batch size/256. In Table 4, we summarize the fine-tuning configuration and the hyperparameters used in Table 2, Table 9, Table 10, and Table 11.

Pre-trained weights. We take the off-the-shelf pre-trained weights of the two expert models. For our main temporal expert, we take the VideoMAE $[56]$ weights pre-trained on the K400, and SSV2 datasets from the official repository $^{5}$ . Since VideoMAE does not provide pre-trained weights specifically for the EK100, we pre-train VideoMAE on the EK100 without incorporating extra video datasets. The pre-training process follows the recipe described in the VideoMAE paper $[56]$ . For all other experiments, we make use of the pre-trained weights provided by the respective model repositories to ensure consistency and reliability in our results.

# C Datasets.

Action recognition We evaluate our B-CAST module on two video datasets: Kinetics400 (K400) [24], Something-Something-V2 (SSV2) [19]. i) K400 is a large-scale third-person video dataset for action recognition that contains around 300K video clips and 400 human action classes. The dataset is split into train/val/test, with 240K/20K/40K video clips. The videos are all trimmed to around 10 seconds from different YouTube video. ii) SSV2 contains over 220K short video clips labeled video clips of humans performing pre-defined, basic actions with everyday objects. The dataset is split into train/val/test, with 168K/24K/27K and have 174 human-objects interaction categories.

Fine-grained action recognition We evaluate our B-CAST module on a Compositional Action dataset: EPIC-KITCHENS-100 (EK100) [10]. EK100 is a large-scale(100hours) egocentric video dataset that records several days of kitchen unscripted activities. It consists of 90K action segments, which are split into train/val/test sets of 67K/10K/13K. Differ from preceding two datasets, EK100 define an action as a combination of a verb and a noun. Because it requires matching both verbs and nouns to recognize an action, it is more challenging than recognizing actions in a dataset where actions are represented by a single label e.g., Kinetics-400, Something-Something-V2.

# D Additional Quantitative Analysis

In this section, we provide additional results to complement the main paper. We demonstrate (1) the generality of CAST with different ViT architectures and pre-trained weights in Appendix § D.1, and (2) the effect of B-CAST-specific positional embeddings in Appendix § D.2.

# D.1 Generality

In this section, we showcase the generality of the proposed method. We demonstrate that CAST works well with any ViT backbone and pre-trained weights. We conduct all the experiments on the EK100 dataset.

CAST is pre-training dataset agnostic. CAST is pre-training dataset agnostic. In Table 5, we compare CAST to a CAST variant where we replace CLIP with a ViT-B model pre-trained on the ImageNet-21K (IN21K) dataset. As a reference, we also show the performance of the independent experts using two separate expert models without any information exchange. When equipped with IN21K pre-trained ViT-B, CAST still achieves reasonable performance with an Action accuracy of 45.5%, while the baseline of independent experts achieves 40.4% accuracy.

Table 5: Effect of CLIP pre-trained weights. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Pre-training dataset</td><td rowspan="2">Information Exchange</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Spatial expert</td><td>Temporal expert</td><td>Verb</td><td>Noun</td><td>Action</td></tr><tr><td>Independent experts</td><td>IN21K</td><td>EK100</td><td>✕</td><td>69.7</td><td>50.9</td><td>40.4</td></tr><tr><td>CAST</td><td>IN21K</td><td>EK100</td><td>✕</td><td>70.9</td><td>56.8</td><td>45.5</td></tr><tr><td>CAST</td><td>CLIP</td><td>EK100</td><td>✕</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

In Table 6, we analyze the effect of pre-training datasets on the temporal expert, VideoMAE, in CAST. We show the results of using EK100, SSV2, and K400 pre-trained VideoMAE weights. We also investigate the impact of pre-training datasets on the spatial expert, CLIP. We present the results of using IN21K and CLIP pre-trained weights. When using CLIP pre-trained weights for the spatial expert, we observe stable Action accuracy ranging from a minimum of 48.7% to a maximum of 49.4%. We observe a similar trend when we use IN21K pre-trained CLIP weights. These results demonstrate that the proposed method is agnostic to the pre-training datasets.

Table 6: Effect of pre-trained weights. 

<table><tr><td colspan="2">Pre-training dataset</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Spatial expert</td><td>Temporal expert</td><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>IN-21K</td><td>EK100</td><td>70.9</td><td>56.8</td><td>45.5</td></tr><tr><td>IN-21K</td><td>SSV2</td><td>71.6</td><td>56.1</td><td>45.3</td></tr><tr><td>IN-21K</td><td>K400</td><td>72.2</td><td>56.4</td><td>45.9</td></tr><tr><td>CLIP</td><td>EK100</td><td>72.5</td><td>60.3</td><td>48.7</td></tr><tr><td>CLIP</td><td>SSV2</td><td>73.3</td><td>60.0</td><td>49.0</td></tr><tr><td>CLIP</td><td>K400</td><td>72.9</td><td>60.4</td><td>49.4</td></tr></table>

CAST is model-agnostic. In Table 7, we demonstrate that CAST is model-agnostic. In this experiment, we replace our spatial and temporal expert models with other existing models. We employ EVA [53], an extension of CLIP that has shown excellent performance in various vision tasks, as our spatial expert. We employ MVD [63] as our temporal expert. The results show that CAST achieves similar performance when we employ different models as the experts. For example, EVA + MVD achieves an Action accuracy of 49.2%, while CLIP + VideoMAE achieves 48.7% Action accuracy.

Table 7: Effect of employing different models as experts. 

<table><tr><td colspan="2">Model architecture</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Spatial expert</td><td>Temporal expert</td><td>Verb</td><td>Noun</td><td>Act.</td></tr><tr><td>CLIP</td><td>VideoMAE</td><td>72.5</td><td>60.3</td><td>48.7</td></tr><tr><td>CLIP</td><td>MVD</td><td>73.1</td><td>60.1</td><td>49.3</td></tr><tr><td>EVA</td><td>VideoMAE</td><td>73.1</td><td>59.8</td><td>49.1</td></tr><tr><td>EVA</td><td>MVD</td><td>73.7</td><td>60.1</td><td>49.2</td></tr></table>

![](images/30f05dfd01213913ddc90a6d0939aa0e5308db4bcbb4a2bca2678687fe073668.jpg)  
Figure 10: Visualization of attention window shape.

# D.2 B-CAST-specific positional embeddings

We investigate the impact of B-CAST-specific positional embeddings, as depicted in Figure 8 (d). In the temporal-to-spatial (T2S) cross-attention, the spatial expert attends along the temporal axis, as shown in Figure 10 (a), while in the self-attention stage during pre-training, the spatial expert attends along the spatial axes, as depicted in Figure 10 (b). Consequently, the spatial expert lacks information about temporal patch sequences. Similarly, in the spatial-to-temporal (S2T) cross-attention, the temporal expert attends along the spatial axis, as illustrated in Figure 10 (b), while during self-attention stage during pre-training, the temporal expert attends along the spatio-temporal axes, as shown in Figure 10 (c). Consequently, the temporal expert lacks information about spatial-only patch sequences. To address this limitation, we introduce new learnable positional embeddings that are specific to T2S and S2T cross-attention.

As shown in Table 8, adding the B-CAST-specific positional embeddings boost the performance by 1.2 points compared to without using the B-CAST-specific positional embeddings. The results indicate the effectiveness of the B-CAST-specific positional embeddings.

Table 8: Effect of B-CAST-specific positional embeddings. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Verb</td><td>Noun</td><td>Action</td></tr><tr><td>CAST w/o B-CAST-specific positional embeddings</td><td>71.3</td><td>59.7</td><td>47.5</td></tr><tr><td>CAST w/ B-CAST-specific positional embeddings</td><td>72.5</td><td>60.3</td><td>48.7</td></tr></table>

# E Comparison with State-of-the-Art

To provide more comprehensive information, we augment the tables for comparison with state-of-the-art in the main paper. Table 9, Table 10, and Table 11, corresponding to the respective datasets, include additional details such as the number of frames per clip, the number of temporal and spatial views used for inference, the computation complexity, and the number of learnable parameters for each model. For the details of the hyperparameters used, please refer to Table 4.

# F Class-Wise Performance Comparison

We provide class-wise F1 score improvement of CAST over our spatial expert (CLIP) and our temporal expert (VideoMAE). We show the verb-class-wise F1 score improvement over CLIP in Figure 11 and the improvement over VideoMAE in Figure 12. We show the noun-class-wise F1 score improvement over CLIP in Figure 13 and the improvement over VideoMAE in Figure 14.

Table 9: Comparison with the state-of-the-arts on the EPIC-Kitchens-100 dataset. We show the Top-1 accuracy for Action, Noun, and Verb prediction tasks as well as the number of frames per clip, the number of temporal and spatial views used for inference, the computation complexity, and the number of learnable parameters for each model. The best performance is in bold and the second best is underscored. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">Frames</td><td rowspan="2">Views</td><td rowspan="2">TFLOPs</td><td rowspan="2">LearnableParam (M)</td><td colspan="3">Top-1 Acc.</td></tr><tr><td>Verb</td><td>Noun</td><td>Action</td></tr><tr><td>SlowFast [15]</td><td>ResNet50</td><td>-</td><td>-</td><td>-</td><td>-</td><td>54.9</td><td>50.0</td><td>38.5</td></tr><tr><td>GSF [52]</td><td>ResNet50</td><td>16</td><td> $3 \times 2$ </td><td>0.4</td><td>-</td><td>68.8</td><td>52.7</td><td>44.0</td></tr><tr><td>MoViNet [27]</td><td>MoViNet-A6</td><td>-</td><td>-</td><td>-</td><td>31</td><td>72.2</td><td>57.3</td><td>47.7</td></tr><tr><td>CLIP* [44]</td><td>ViT-B</td><td>8</td><td> $2 \times 3$ </td><td>0.84</td><td>86</td><td>55.5</td><td>52.3</td><td>33.9</td></tr><tr><td>AIM* [72]</td><td>ViT-B</td><td>16</td><td> $2 \times 3$ </td><td>2.42</td><td>14</td><td>64.8</td><td>55.5</td><td>41.3</td></tr><tr><td>ST-Adapter [42]</td><td>ViT-B</td><td>8</td><td> $3 \times 1$ </td><td>-</td><td>-</td><td>67.6</td><td>55.0</td><td>-</td></tr><tr><td>MBT [40]</td><td>ViT-B</td><td>32</td><td>-</td><td>-</td><td>-</td><td>64.8</td><td>58.0</td><td>43.4</td></tr><tr><td>ViViT FE [1]</td><td>ViT-L</td><td>32</td><td> $4 \times 1$ </td><td>15.92</td><td>311</td><td>66.4</td><td>56.8</td><td>44.0</td></tr><tr><td>MFormer [43]</td><td>ViT-L</td><td>32</td><td> $3 \times 1$ </td><td>3.56</td><td>-</td><td>67.1</td><td>57.6</td><td>44.1</td></tr><tr><td>ORViT-MF-HR [21]</td><td>ViT-B</td><td>16</td><td> $10 \times 3$ </td><td>-</td><td>-</td><td>68.4</td><td>58.7</td><td>45.7</td></tr><tr><td>VideoSwin [37]</td><td>Swin-B</td><td>-</td><td>-</td><td>-</td><td>89</td><td>67.8</td><td>57.0</td><td>46.1</td></tr><tr><td>VideoMAE* [56]</td><td>ViT-B</td><td>16</td><td> $2 \times 3$ </td><td>1.08</td><td>87</td><td>70.5</td><td>51.4</td><td>41.7</td></tr><tr><td>MeMViT [68]</td><td>ViT-B</td><td>16</td><td> $1 \times 1$ </td><td>0.06</td><td>-</td><td>70.6</td><td>58.5</td><td>46.2</td></tr><tr><td>OMNIVORE [16]</td><td>Swin-B</td><td>32</td><td>-</td><td>-</td><td>-</td><td>69.5</td><td>61.7</td><td>49.9</td></tr><tr><td>MTV-HR [71]</td><td>MTV-B</td><td>32</td><td> $4 \times 1$ </td><td>3.72</td><td>310</td><td>68.0</td><td>63.1</td><td>48.6</td></tr><tr><td>CAST w/ CLIP &amp; VideoMAE pretrained on EPIC-KITCHENS-100</td><td>CAST-B</td><td>16</td><td> $2 \times 3$ </td><td>2.35</td><td>45</td><td>72.5</td><td>60.9</td><td>49.3</td></tr></table>

\*We conduct experiments with our own implementation.

Table 10: Comparison with the state-of-the-arts on the Something-Something-V2 dataset. We show the Top-1 accuracy as well as the number of frames per clip, the number of temporal and spatial views used for inference, the computation complexity, and the number of learnable parameters for each model. The best performance is in bold and the second best is underscored. 

<table><tr><td>Method</td><td>Backbone</td><td>Frames</td><td>Views</td><td>TFLOPs</td><td>Learnable Param (M)</td><td>Top-1 Acc.</td></tr><tr><td>SlowFast [15]</td><td>ResNet101</td><td>8+32</td><td>1×3</td><td>0.32</td><td>-</td><td>63.1</td></tr><tr><td>CLIP* [44]</td><td>ViT-B</td><td>8</td><td>2×3</td><td>0.84</td><td>-</td><td>43.2</td></tr><tr><td>EVL [35]</td><td>ViT-B</td><td>32</td><td>1×3</td><td>2.05</td><td>29</td><td>62.4</td></tr><tr><td>ST-Adapter [42]</td><td>ViT-B</td><td>32</td><td>3×1</td><td>1.96</td><td>-</td><td>69.5</td></tr><tr><td>AIM [72]</td><td>ViT-B</td><td>32</td><td>1×3</td><td>2.50</td><td>14</td><td>69.1</td></tr><tr><td>ViViT FE [1]</td><td>ViT-L</td><td>32</td><td>4×3</td><td>11.89</td><td>311</td><td>65.9</td></tr><tr><td>TimeSformer [3]</td><td>ViT-L</td><td>64</td><td>1×3</td><td>7.14</td><td>-</td><td>62.4</td></tr><tr><td>MViT [12]</td><td>ViT-B</td><td>64</td><td>1×3</td><td>1.37</td><td>37</td><td>67.7</td></tr><tr><td>MFormer [43]</td><td>ViT-L</td><td>32</td><td>1×3</td><td>3.56</td><td>-</td><td>68.1</td></tr><tr><td>Video Swin [37]</td><td>Swin-B</td><td>32</td><td>1×3</td><td>0.96</td><td>89</td><td>69.6</td></tr><tr><td>BEVT [62]</td><td>Swin-B</td><td>32</td><td>1×3</td><td>0.96</td><td>-</td><td>70.6</td></tr><tr><td>VideoMAE [56]</td><td>ViT-B</td><td>16</td><td>2×3</td><td>1.08</td><td>87</td><td>70.8</td></tr><tr><td>ORViT-MF-L [21]</td><td>ViT-L</td><td>32</td><td>1×3</td><td>-</td><td>-</td><td>69.5</td></tr><tr><td>OMNIVORE [16]</td><td>Swin-B</td><td>32</td><td>-</td><td>-</td><td>-</td><td>71.4</td></tr><tr><td>MTV-HR [71]</td><td>MTV-B</td><td>32</td><td>4×3</td><td>11.16</td><td>310</td><td>68.5</td></tr><tr><td>CAST w/ VideoMAE pretrained on Something-Something-V2</td><td>CAST-B</td><td>16</td><td>2×3</td><td>2.35</td><td>45</td><td>71.6</td></tr></table>

\*We conduct experiments with our own implementation.

# G Qualitative Analysis

To better understand the effectiveness of CAST, we provide qualitative analysis on more sample frames from the EK100 dataset in Figure 15. Each expert model provides more accurate prediction in their respective tasks of expertise but shows weaker performance in the other task. In contrast, CAST consistently shows correct predictions for both noun and verb prediction tasks. The qualitative examples demonstrate the effectiveness of CAST in achieving balanced spatio-temporal understanding, which is essential for fine-grained action recognition.

# H Limitations

Despite achieving a good balanced spatio-temporal understanding performance, CAST has a few limitations as well. CAST has a small number of learnable parameters since we freeze the expert models except for the adapters. However, the computational complexity of CAST is not negligible due to the utilization of two expert models. Due to resource limitations, we are unable to conduct experiments on various input video lengths and model sizes. Lastly, cross-attention layers require features of the same dimension for attention operation. Therefore, if the two model have significantly different architectures, it might be challenging for CAST to employ the two models.

Table 11: Comparison with the state-of-the-arts on the Kinetics400 dataset. We show the Top-1 accuracy as well as the number of frames per clip, the number of temporal and spatial views used for inference, the computation complexity, and the number of learnable parameters for each model. The best performance is in bold and the second best is underscored. 

<table><tr><td>Method</td><td>Backbone</td><td>Frames</td><td>Views</td><td>TFLOPs</td><td>Learnable Param (M)</td><td>Top-1 Acc.</td></tr><tr><td>SlowFast [15]</td><td>ResNet101</td><td>80</td><td> $10 \times 3$ </td><td>7.02</td><td>-</td><td>79.8</td></tr><tr><td>X3D [13]</td><td>X3D-XL</td><td>16</td><td> $10 \times 3$ </td><td>1.45</td><td>-</td><td>79.1</td></tr><tr><td>MoViNet [27]</td><td>MoViNet-A6</td><td>120</td><td> $1 \times 1$ </td><td>0.39</td><td>-</td><td>81.5</td></tr><tr><td>UniFormer [31]</td><td>Hybrid-B</td><td>32</td><td> $4 \times 3$ </td><td>3.12</td><td>50</td><td>83.0</td></tr><tr><td>CLIP* [44]</td><td>ViT-B</td><td>8</td><td> $5 \times 3$ </td><td>2.2</td><td>86</td><td>77.3</td></tr><tr><td>EVL [35]</td><td>ViT-B</td><td>32</td><td> $3 \times 1$ </td><td>1.78</td><td>29</td><td>84.2</td></tr><tr><td>ST-Adapter [42]</td><td>ViT-B</td><td>32</td><td> $3 \times 1$ </td><td>1.82</td><td>-</td><td>82.7</td></tr><tr><td>Text4Vis [69]</td><td>ViT-B</td><td>16</td><td> $4 \times 3$ </td><td>-</td><td>-</td><td>83.6</td></tr><tr><td>AIM [72]</td><td>ViT-B</td><td>32</td><td> $3 \times 1$ </td><td>2.43</td><td>11</td><td>84.7</td></tr><tr><td>X-CLIP [41]</td><td>ViT-B</td><td>16</td><td> $4 \times 3$ </td><td>3.44</td><td>-</td><td>84.7</td></tr><tr><td>ViViT FE [1]</td><td>ViT-L</td><td>128</td><td> $1 \times 3$ </td><td>11.94</td><td>311</td><td>81.7</td></tr><tr><td>TimeSformer [3]</td><td>ViT-L</td><td>96</td><td> $1 \times 3$ </td><td>25.06</td><td>430</td><td>80.7</td></tr><tr><td>MViT [12]</td><td>ViT-B</td><td>32</td><td> $5 \times 1$ </td><td>0.85</td><td>37</td><td>80.2</td></tr><tr><td>BEVT [62]</td><td>Swin-B</td><td>32</td><td> $4 \times 3$ </td><td>3.38</td><td>88</td><td>80.6</td></tr><tr><td>MFormer [43]</td><td>ViT-L</td><td>32</td><td> $10 \times 3$ </td><td>35.55</td><td>-</td><td>80.2</td></tr><tr><td>Video Swin [37]</td><td>Swin-L</td><td>32</td><td> $4 \times 3$ </td><td>7.25</td><td>197</td><td>83.1</td></tr><tr><td>VideoMAE [56]</td><td>ViT-B</td><td>16</td><td> $5 \times 3$ </td><td>2.7</td><td>87</td><td>81.5</td></tr><tr><td>OMNIVORE [16]</td><td>Swin-B</td><td>32</td><td>-</td><td>-</td><td>-</td><td>84.0</td></tr><tr><td>MTV-HR [71]</td><td>MTV-B</td><td>32</td><td> $4 \times 3$ </td><td>11.16</td><td>310</td><td>82.4</td></tr><tr><td>CAST w/ VideoMAE pretrained on Kinetics-400</td><td>CAST-B</td><td>16</td><td> $5 \times 3$ </td><td>5.87</td><td>45</td><td>85.3</td></tr></table>

\*We conduct experiments with our own implementation.

# I Broader Impacts

Our work is on the task of human action recognition from videos. Surveillance could be one application, which might have privacy related concerns when the technology is deployed. Other consumer applications like personal or internet video search and tagging is expected to benefit individuals and organizations alike by helping them more efficiently maintain human centered data.

![](images/57345f7d8da2d56c9caa33db61b065160ea3af1e998c0664e93e3c55cd89f830.jpg)

<details>
<summary>bar</summary>

| Word | F1 score improvement |
|---|---|
| brush | 0.92 |
| slide | 0.78 |
| eat | 0.62 |
| lower | 0.45 |
| unroll | 0.43 |
| water | 0.40 |
| stretch | 0.39 |
| apply | 0.33 |
| knead | 0.31 |
| feel | 0.29 |
| close | 0.27 |
| gather | 0.26 |
| divide | 0.25 |
| shake | 0.25 |
| move | 0.25 |
| take | 0.24 |
| put | 0.23 |
| unscrew | 0.21 |
| sprinkle | 0.15 |
| pull | 0.14 |
| turn-off | 0.14 |
| look | 0.13 |
| hold | 0.12 |
| open | 0.11 |
| spray | 0.10 |
| insert | 0.09 |
| empty | 0.08 |
| wrap | 0.07 |
| scoop | 0.06 |
| filter | 0.06 |
| mix | 0.06 |
| turn-on | 0.06 |
| cook | 0.05 |
| remove | 0.05 |
| fold | 0.05 |
| break | 0.05 |
| adjust | 0.05 |
| scrub | 0.04 |
| hang | 0.04 |
| pour | 0.03 |
| dry | 0.02 |
| squeeze | 0.02 |
| wash | 0.01 |
| fill | 0.01 |
| throw | 0.01 |
| press | -0.01 |
| wait | -0.01 |
| use | -0.01 |
| turn-down | -0.01 |
| sort | -0.01 |
| soak | -0.01 |
| smell | -0.01 |
| sharpen | -0.01 |
| search | -0.01 |
| rub | -0.01 |
| roll | -0.01 |
| rip | -0.01 |
| pat | -0.01 |
| measure | -0.01 |
| lift | -0.01 |
| increase | -0.01 |
| grate | -0.01 |
| form | -0.01 |
| flip | -0.01 |
| drop | -0.01 |
| coat | -0.01 |
| attach | -0.01 |
| turn | -0.01 |
| cut | -0.01 |
| peel | -0.01 |
| check | -0.12 |
| add | -0.13 |
| crush | -0.14 |
| scrape | -0.22 |
| wear | -0.48 |
| set | -0.52 |
| drink | -0.58 |
| transition | -0.68 |
</details>

Figure 11: Improvements of CAST over CLIP on EK100 verb classes. We show the class-wise F1 score improvement of CAST over CLIP. CAST achieves an improvement of 17.8 points on average. Best viewed with zoom and color.

![](images/01dec598b0df7ea0dbb40ef04a84e85540bc03ab11315887dfab977278a77063.jpg)

<details>
<summary>bar</summary>

| Word | F1 score improvement |
|---|---|
| brush | 0.92 |
| slide | 0.78 |
| lower | 0.45 |
| unroll | 0.44 |
| divide | 0.43 |
| water | 0.39 |
| stretch | 0.38 |
| spray | 0.36 |
| feel | 0.31 |
| look | 0.29 |
| gather | 0.27 |
| apply | 0.26 |
| unscrew | 0.22 |
| fold | 0.18 |
| wrap | 0.16 |
| sprinkle | 0.15 |
| pull | 0.14 |
| filter | 0.12 |
| break | 0.11 |
| remove | 0.09 |
| empty | 0.08 |
| scoop | 0.07 |
| eat | 0.06 |
| press | 0.05 |
| insert | 0.03 |
| turn-off | 0.02 |
| dry | 0.01 |
| move | 0.01 |
| shake | 0.01 |
| cut | 0.01 |
| take | 0.01 |
| close | 0.01 |
| throw | 0.01 |
| squeeze | 0.01 |
| pour | 0.01 |
| open | 0.01 |
| put | 0.01 |
| peel | 0.01 |
| turn | 0.01 |
| adjust | 0.01 |
| mix | 0.01 |
| turn-on | 0.01 |
| wash | 0.01 |
| wait | 0.01 |
| use | 0.01 |
| sort | 0.01 |
| smell | 0.01 |
| sharpen | 0.01 |
| scrape | 0.01 |
| rub | 0.01 |
| roll | 0.01 |
| rip | 0.01 |
| pat | 0.01 |
| measure | 0.01 |
| lift | 0.01 |
| increase | 0.01 |
| hold | 0.01 |
| grate | 0.01 |
| form | 0.01 |
| flip | 0.01 |
| drop | 0.01 |
| coat | 0.01 |
| attach | -0.02 |
| fill | -0.03 |
| check | -0.04 |
| add | -0.05 |
| knead | -0.12 |
| crush | -0.14 |
| scrub | -0.15 |
| search | -0.17 |
| hang | -0.18 |
| cook | -0.22 |
| drink | -0.26 |
| soak | -0.32 |
| wear | -0.42 |
| set | -0.62 |
| turn-down | -0.72 |
| transition | -0.88 |
</details>

Figure 12: Improvements of CAST over VideoMAE on EK100 verb classes. We show the class-wise F1 score improvement of CAST over VideoMAE. CAST achieves an improvement of 2.2 points in on average. Best viewed with zoom and color.

![](images/dffb43a7bc30c41631826f18e0ae175b333bfbc304103e7e24a4af3ad47802bf.jpg)  
Figure 13: Improvements of CAST over CLIP on EK100 noun classes. We show the class-wise F1 score improvement of CAST over CLIP. CAST achieves an improvement of 7.9 points on average. Best viewed with zoom and color.

![](images/b124cdafd150d48c2b678bb532bd58cb8e9f6714f8aff718b324af01682cefa0.jpg)

<details>
<summary>bar</summary>

| Category | F1 score improvement |
|---|---|
| foodder | 0.95 |
| blender | 0.88 |
| thermometer | 0.85 |
| pepper/salted | 0.82 |
| heater | 0.79 |
| carrot | 0.76 |
| book | 0.73 |
| mango | 0.70 |
| time | 0.67 |
| flour | 0.64 |
| chocolate | 0.61 |
| onion/bottle | 0.58 |
| omicette | 0.55 |
| cake | 0.52 |
| bacon | 0.49 |
| stock | 0.46 |
| napkin | 0.43 |
| pumpkin | 0.40 |
| grater | 0.37 |
| kiwup | 0.34 |
| syrup | 0.31 |
| oregano | 0.28 |
| noodle | 0.25 |
| recipe | 0.22 |
| pork | 0.19 |
| pepper | 0.16 |
| orange | 0.13 |
| cucumber | 0.10 |
| banana | 0.07 |
| red-cut-ring | 0.04 |
| spinach | 0.01 |
| spinach | -0.02 |
| potato | -0.05 |
| lighter | -0.08 |
| heccor | -0.11 |
| maltarpea | -0.14 |
| onion/spring | -0.17 |
| kiwarts | -0.20 |
| asubgrine | -0.23 |
| hummose | -0.26 |
| mel | -0.29 |
| nut | -0.32 |
| apple | -0.35 |
| tisa | -0.38 |
| chicken | -0.41 |
| biscuit | -0.44 |
| olive | -0.47 |
| oatmeal | -0.50 |
| blueberry | -0.53 |
| basket | -0.56 |
| herb | -0.59 |
| egg | -0.62 |
| light | -0.65 |
| carrot | -0.68 |
| riceon | -0.71 |
| wrap/plastics | -0.74 |
| pharma | -0.77 |
| butter | -0.80 |
| olive | -0.83 |
| pepper | -0.86 |
| bread | -0.89 |
| crisp | -0.92 |
| lettuce | -0.95 |
| jzig | -0.98 |
| rice | -1.01 |
| onion/krupp | -1.04 |
| peanut | -1.07 |
| tomato | -1.10 |
| knife | -1.13 |
| spice | -1.16 |
| saucer | -1.19 |
| potato | -1.22 |
| pickle | -1.25 |
| pickle/plastic | -1.28 |
| garlic | -1.31 |
| pumpkin (pasta) | -1.34 |
| green (pasta) | -1.37 |
| beer (pasta) | -1.40 |
| milk (pasta) | -1.43 |
| tamarate (pasta) | -1.46 |
| microwave (pasta) | -1.49 |
| bowl (pasta) | -1.52 |
| barley (pasta) | -1.55 |
| mushroom (pasta) | -1.58 |
| pasta (pasta) | -1.61 |
| cantal (pasta) | -1.64 |
| lettuce (pasta) | -1.67 |
| salt (pasta) | -1.70 |
| pear (pasta) | -1.73 |
| burger (pasta) | -1.76 |
| chibi (pasta) | -1.79 |
| grass (pasta) | -1.82 |
| bean (pasta) | -1.85 |
| boat chopsticks (pasta) | -1.88 |
| jar (pasta) | -1.91 |
| liquid washing (pasta) | -1.94 |
| liquid washing (pasta) with maker coffee (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta), milk (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta) with cream (pasta). The chart displays a descending trend from top to bottom, indicating that most items have positive F1 score improvements, while few items have negative or near-zero improvements, suggesting a general decline in performance across the food chain from top to bottom of the scale.
</details>

Figure 14: Improvements of CAST over VideoMAE on EK100 noun classes. We show the class-wise F1 score improvement of CAST over VideoMAE. CAST achieves an improvement of 9.2 points on average. Best viewed with zoom and color.

![](images/3b41a85e1874c3a86c1dde3949980e41bc6cd3a37bb54a0a4b3f6a1555fa3285.jpg)  
CLIP: Carrot √
VideoMAE: Box ✗
CAST: Carrot √   
CLIP: Pan ✗
VideoMAE: Pasta ✗
CAST: Mushroom √   
CLIP: Take ✗
VideoMAE: Turn-on ✓
CAST: Turn-on ✓   
CLIP: Pour ✓
VideoMAE: Put ✗
CAST: Pour ✓   
CLIP: Knife ✓
VideoMAE: Bag ✗
CAST: Knife ✓   
CLIP: Potato ✓
VideoMAE: Pepper ✗
CAST: Potato ✓   
CLIP: Turn-on √
VideoMAE: Turn-on √
CAST: Turn-on √   
CLIP: Take ✗
VideoMAE: Take ✗
CAST: Cut √   
CLIP: Fork ✗
VideoMAE: Galic ✗
CAST: Been:green ✗
GT: Asparagus   
CLIP: Peach ✓
VideoMAE: Knife ✗
CAST: Peach ✓   
CLIP: Hang ✗
VideoMAE: Throw ✓
CAST: Throw ✓   
CLIP: Put ✗
VideoMAE: Move ✗
CAST: Shake √   
CLIP: Package ✗
VideoMAE: Package ✗
CAST: Blueberry √   
CLIP: Nuts √
VideoMAE: Bag ✗
CAST: Nuts √   
CLIP: Wash √
VideoMAE: Dry ✗
CAST: Wash √   
CLIP: Take ✗
VideoMAE: Mix ✓
CAST: Mix ✓   
CLIP: Recipe ✓
VideoMAE: Box ✗
CAST: Recipe ✓   
CLIP: Sponge ✗
VideoMAE: Sponge ✗
CAST: Liquid √   
CLIP: Scoop ✓
VideoMAE: Insert ✗
CAST: Scoop ✓   
CLIP: Take ✗
VideoMAE: Dry ✓
CAST: Dry ✓

Figure 15: Qualitative examples from EK100 comparing CLIP, VideoMAE, and the proposed CAST. Each expert model shows more accurate predictions in their expertise but shows weaker performance on the other task. However, the proposed CAST consistently shows correct predictions for both noun and verb classes, demonstrating the effectiveness of the proposed spatio-temporal cross-attention mechanism. Best viewed with zoom and color.