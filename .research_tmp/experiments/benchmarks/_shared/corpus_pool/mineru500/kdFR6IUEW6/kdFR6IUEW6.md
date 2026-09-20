# Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition

Shuhuai Ren $^{\ddagger\dagger}$ , Aston Zhang $^{\dagger,*}$ , Yi Zhu $^{\dagger}$ , Shuai Zhang $^{\dagger}$ , Shuai Zheng $^{\dagger}$ , Mu Li $^{\dagger}$ , Alex Smola $^{\dagger}$ , Xu Sun $^{\ddagger}$

$^{\ddagger}$ National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University

$^{\dagger}$ Amazon Web Services

# Abstract

This work proposes POMP, a prompt pre-training method for vision-language models. Being memory and computation efficient, POMP enables the learned prompt to condense semantic information for a rich set of visual concepts with over twenty-thousand classes. Once pre-trained, the prompt with a strong transferable ability can be directly plugged into a variety of visual recognition tasks including image classification, semantic segmentation, and object detection, to boost recognition performances in a zero-shot manner. Empirical evaluation shows that POMP achieves state-of-the-art performances on 21 datasets, e.g., 67.0% average accuracy on 10 classification datasets (+3.1% compared to CoOp) and 84.4 hIoU on open-vocabulary Pascal VOC segmentation (+6.9 compared to ZSSeg). Our code is available at https://github.com/amazon-science/prompt-pretraining.

# 1 Introduction

It has been a new norm to formulate visual recognition tasks (e.g., image classification, object detection, and semantic segmentation) as language-guided visual recognition or vision-and-language problems $[42, 19, 61]$ . In language-guided visual recognition, categories of images are represented by natural language rather than discrete label IDs, and the semantics between images and their corresponding textual descriptions are often aligned via a constrative loss during training $[42]$ . Model inference also becomes an image-to-text matching problem, where text prompts like “a photo of a [CLASSNAME]” are curated as text descriptions of images. By varying the [CLASSNAME] placeholder and computing the similarity between text descriptions and images, we can identify the most suitable class name and consider it as the predicted target class. A significant benefit of this language-guided paradigm is that it supports open-vocabulary inference, that is, zero-shot recognition for arbitrary categories that may not even have been seen during training, thanks to the flexibility in modifying the class names in the textual prompt $[42, 45]$ .

As the context for class names, the text prompt plays a critical role in language-guided visual recognition models. A good prompt should holistically express the semantics of visual categories to better elicit the knowledge learned by vision-language models (VLMs) during pre-training. There are two popular types of prompts: hard prompts (e.g., a photo of a [CLASSNAME]) and soft prompts. Soft prompts are learnable token embeddings that can be fine-tuned given some input data, and have been demonstrated to be more effective and stable on downstream tasks than hard prompts $[65, 35, 63]$ . However, traditional prompt tuning methods usually fine-tune the soft prompt on task-specific datasets with a limited number of class labels, making it difficult to generalize to novel classes and across tasks $[45, 2]$ . For example, when transferred between the two downstream datasets of Flowers102 $[40]$

and DTD [9], the soft prompt fine-tuned on DTD only achieves 33.4% accuracy on Flowers102, significantly lower than the 61.8% accuracy of a hand-crafted prompt on Flowers102, demonstrating a severe overfitting issue [49].

In this work, we aim to learn a universal soft prompt that covers a broad set of visual concepts while being task-agnostic. Specifically, we propose PrOMpt Pre-training (POMP), a method for scaling up prompt learning on the ImageNet-21K dataset, which has over twenty-thousand classes organized by the WordNet [37] hierarchy. The set of classes in ImageNet-21K includes general and long-tail visual categories of various semantic granularities, which have been proven to provide better downstream results for large models [30, 14]. Pre-training on this large-scale dataset helps condense semantic information into the soft prompt for universal visual discrimination. Once pre-trained, this universal prompt (i) can be easily applied to downstream datasets to improve model performance in zero-shot settings; (ii) is compatible with both region-level and pixel-level visual patterns, making it useful for various vision tasks such as object detection and semantic segmentation.

![](images/e3fa878364a83826dac5ed9ee13552669c904da2391ae24aad030ad83e8e726c.jpg)

<details>
<summary>radar</summary>

| Category | Previous SOTA | POMP (Ours) |
| -------- | ------------- | ----------- |
| Image Classification (ImageNet-21k) | 24.8 | 25.2 |
| Image Classification (Cross-dataset) | 26.4 | 66.8 |
| Image Classification (Cross-domain) | 26.8 | 60.8 |
| Object Detection (Open-vocab LVIS) | 39.2 | 39.8 |
| Object Detection (LVIS to COCO) | 15.8 | 40.4 |
| Object Detection (LVIS to Object365) | 16.6 | 16.2 |
| Instance Segmentation (Open-vocab LVIS) | 24.8 | 25.2 |
| Semantic Segmentation (Open-vocab COCO Stuff) | 38.2 | 39.4 |
| Semantic Segmentation (Open-vocab PASCAL VOC) | 83.4 | 81.8 |
| Semantic Segmentation (COCO Stuff to ADE20k) | 78.6 | 80.2 |
| Semantic Segmentation (COCO Stuff to PASCAL Context) | 50.8 | 51.6 |
</details>

Figure 1: POMP outperforms previous state-of-the-art models on a broad range of visual recognition tasks and datasets.

However, pre-training prompt with such a massive class set is challenging due to generally prohibitive computational costs. During prompt pre-training, the activation of the whole text encoder needs to be kept independently for every class and the memory consumption increases proportionally to the number of classes. In short, pre-training prompts on ImageNet-21K requires over 300 GB GPU memory with traditional methods like CoOp [65]. In POMP, we solve this issue with a simple class sampling strategy, local contrast, which reduces the GPU memory requirement dramatically to less than 16 GB. Moreover, we propose a local correction strategy to reduce the bias caused by class sampling and improve the generalization of the pre-trained prompt.

Experimental results in Figure 1 show that POMP outperforms previous state-of-the-art (SOTA) models on a broad range of visual recognition tasks and datasets. Specifically, compared to zeroshot CLIP [42], POMP improves the accuracy on ImageNet-21K by a gain of +2.9%. It also achieves an average accuracy of 67.0% when transferred to 10 downstream image classification datasets, which is 3.1% higher than CoOp [65]. For semantic segmentation, POMP achieves 39.1 hIoU on open-vocab COCO Stuff and 84.4 hIoU on open-vocab Pascal VOC, outperforming ZSSeg [61] by +1.3 and +6.9 hIoU, respectively. For object detection, POMP achieves 57.9 and 22.9 AP $_{50}$ when transferred from LVIS to COCO and Object365, surpassing Detic [67] by +1.9 and +0.8 AP $_{50}$ , respectively.

# 2 Related Work

# 2.1 Language-Guided Visual Recognition

Language-guided visual recognition usually leverages VLMs as foundation models. Representative VLMs like CLIP $[42]$ consist of an image encoder and a text encoder, which are used to encode image-text pairs into a joint feature space for learning the semantic alignment between vision and language $[7, 46]$ . After being pre-trained on large-scale image-text pairs, CLIP-like models $[28, 62, 33]$ are able to map images to their corresponding language descriptions, allowing visual recognition to generalize in the wild. This language-driven modeling paradigm also facilitates other vision tasks, including semantic segmentation $[61, 32, 43, 13]$ and object detection $[19, 15, 67]$ . These works typically designed a two-stage framework: it first leverages the pre-trained proposal network to extract features of specific visual patterns (e.g., segment mask and region) and then conducts classification in the same matching style as CLIP. The class descriptions for matching are synthesized using prompts, and in this work, we pre-train a soft prompt for VLMs on ImageNet-21K to further enhance their zero-shot generalization ability.

# 2.2 Prompt Tuning

In order to adapt VLMs to downstream tasks, recent research proposed a parameter-efficient tuning method named prompt tuning. CoOp [65] first proposed to replace the hand-crafted prompt with learnable vectors (also known as a soft prompt) for fine-tuning while freezing the entire pre-trained parameters. VPT [12], on the contrary, moved the learnable vectors from the text side to the image side, and proposed to concatenate the “visual soft prompt” and the patch sequence of an image as the input for fine-tuning. Prompt tuning was also used in other visual recognition tasks, such as object detection [15], semantic segmentation [43], and video recognition [39]. Over manual prompt engineering, the soft prompt optimized with few-shot data has achieved significant performance improvements, but only fitting one specific downstream dataset.

To enhance the generalization of the soft prompt to a wider range of unseen classes and datasets, CoCoOp [66] modeled the context condition on input images. A recent work MaPLe [29] appended the soft prompt to the hidden representations at each layer in both the text and image encoder. In sharp contrast to previous approaches, we propose to pre-train a universal soft prompt on large-scale datasets with massive visual categories. Such a pre-trained prompt is task-agnostic, allowing for direct transfer to various downstream datasets without fine-tuning.

# 3 Method

We first review the process of classical prompt tuning for VLMs in § 3.1. To address the training efficiency issue of previous methods, in § 3.2 we introduce our method of prompt pre-training (POMP) that includes two key components: local contrast and local correction. Our pre-trained prompt can then be transferred to downstream datasets and tasks in a zero-shot manner as discussed in § 3.3.

# 3.1 Preliminaries

Language-guided visual recognition models like CLIP formulate image classification as an image-text matching problem, where the goal is to select the correct textual class name from a predefined class set for the image query. Following [65, 29], we consider CLIP as our vision-language foundation model, for its simplicity in design and wide applicability. CLIP consists of an image encoder $f_I$ and a text encoder $f_T$ . Given a visual recognition dataset $\mathcal{D}$ with a class set of $N$ class names $\mathcal{C} = \{c_i\}_{i=1}^N$ , CLIP manually devises a hard prompt to synthesize textual descriptions $t_i$ for each class name $c_i$ , e.g., $t_i =$ "a photo of a $[c_i]$ ". Then each class description is fed into the text encoder to generate the normalized class feature $\mathbf{w}_i = f_T(t_i) / \|f_T(t_i)\|_2 \in \mathbb{R}^d$ , where $d$ is the dimension of the feature. The concatenation of $N$ class features $[\mathbf{w}_1, \cdots, \mathbf{w}_N] \in \mathbb{R}^{N \times d}$ can be considered as the class weight of a linear classifier for classifying an image. Given an input image $x$ , the image encoder is used to extract its visual feature $\mathbf{x} = f_I(x) / \|f_I(x)\|_2 \in \mathbb{R}^d$ . Finally, CLIP calculates the similarity between $\mathbf{x}$ and all the class features, then predicts the class with the highest similarity as the target class.

To address the inefficient expressiveness of the manual prompt, previous research like CoOp [65] proposed to parameterize the manual prompt as a soft prompt $\Theta$ , and fine-tune it to fit downstream datasets. The soft prompt is made up of a sequence of learnable token embeddings $\Theta = [\theta_{1}, \theta_{2}, \cdots, \theta_{M}] \in R^{M \times e}$ , where M is a hyperparameter specifying the length of the soft prompt and e is the dimension of the token embedding. The token embeddings of each class name $c_{i}$ are further appended to the soft prompt $\Theta$ to generate the class feature $\mathbf{w}_{i}^{(\Theta)}$ , and the prediction probability for the ground-truth class y is denotes as

$$
P (y \mid \mathbf {x}; \boldsymbol {\Theta}) = \frac {\exp (\mathbf {x} ^ {\top} \mathbf {w} _ {y} ^ {(\boldsymbol {\Theta})} / \tau)}{\sum_ {i = 1} ^ {N} \exp (\mathbf {x} ^ {\top} \mathbf {w} _ {i} ^ {(\boldsymbol {\Theta})} / \tau)}, \tag {1}
$$

where $\mathbf{x}^{\top}\mathbf{w}_i$ represents the similarity score and $\tau$ is a temperature parameter. The parameters of the soft prompt are updated by minimizing the cross-entropy loss [65]:

$$
\mathcal {L} (\boldsymbol {\Theta}) = \underset {(\mathbf {x}, y) \in \mathcal {D}} {\mathbb {E}} \left[ - \log P (y \mid \mathbf {x}; \boldsymbol {\Theta}) \right]. \tag {2}
$$

The gradient of $\mathcal{L}(\Theta)$ is represented as:

$$
\nabla_ {\boldsymbol {\Theta}} \left(- \log P (y \mid \mathbf {x}; \boldsymbol {\Theta})\right) = \frac {1}{\tau} \left[ - \nabla_ {\boldsymbol {\Theta}} \left(\mathbf {x} ^ {\top} \mathbf {w} _ {y} ^ {(\boldsymbol {\Theta})}\right) + \sum_ {i = 1} ^ {N} P \left(y _ {i} \mid \mathbf {x}; \boldsymbol {\Theta}\right) \nabla_ {\boldsymbol {\Theta}} \left(\mathbf {x} ^ {\top} \mathbf {w} _ {i} ^ {(\boldsymbol {\Theta})}\right) \right]. \tag {3}
$$

![](images/e019bc33ed75c9fa518d7d3790c2077da127381100d95779d0dd19df5c98e4fc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Prompt"] --> B["Massive Class Names"]
    B --> C["Text Encoder"]
    C --> D["Class Features"]
    D --> E["Contrastive Loss"]
    E --> F["Image Feature"]
    F --> G["Classification"]
    
    H["Zero-Shot Transfer (Stage 2)"] --> I["Target Seg"]
    I --> J["CLS Classes"]
    I --> K["DET Classes"]
    I --> L["Cat...lion"]
    
    M["Prompt Pre-Training (Stage 1)"] --> N["Text Encoder"]
    N --> O["Class Features"]
    O --> P["Contrastive Loss"]
    P --> Q["Image Feature"]
    Q --> R["Classification"]
    
    S["Prompt"] --> T["bus...train"]
    T --> U["Text Encoder"]
    U --> V["Class Features"]
    V --> W["Class Features"]
    W --> X["Mask Feature"]
    X --> Y["Segmentation"]
    
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style M fill:#ccf,stroke:#333
```
</details>

Figure 2: Overview of POMP. POMP pre-trains a soft prompt ( $\text{fl/fl}$ :learnable) on the ImgaNet-21K dataset with massive classes, and then directly transfers the learned prompt ( $\text{fl/fl}$ :frozen) to downstream datasets of image classification (CLS), object detection (DET), and semantic segmentation (SEG) tasks. For DET and SEG, the region and mask proposal networks require pre-training with POMP prompt on detection and segmentation source data, respectively (See Appendix B).

![](images/3215f42ffc8d35833c1b7b15d6eab7bb882df4e9b361a656b6ed900bbc05a4f2.jpg)

<details>
<summary>bar</summary>

| Dataset (#class) | CoOp (GB) | POMP (Ours, K=1000) (GB) |
|---|---|---|
| CIFAR10 (10) | 0.3 | |
| DTD (47) | 0.9 | |
| FGVC Aircraft (100) | 1.6 | |
| Stanford Cars (196) | 3.0 | |
| SUN397 (397) | 5.9 | |
| ImageNet-1K (1000) | 15.7 | |
| ImageNet-21K (21K) | 316.4 | 15.7 |
</details>

Figure 3: GPU memory overhead (GB) required for prompt tuning on datasets with varying numbers of classes. The memory cost of CoOp on ImageNet-21K is 316.4 GB, which is generally prohibitive. POMP reduces the cost dramatically to 15.7 GB with local contrast among the 1000 sampled classes for optimization.

It can be decomposed into positive reinforcement for the ground-truth class and negative reinforcement for every class, which are the first and the second terms inside the square brackets of (3), respectively.

Note that both the image encoder and the text encoder are frozen during the prompt tuning process, which allows adapting the soft prompt efficiently to downstream data with very few learnable parameters. Methods along this direction of soft prompt learning include CoCoOp [66] and MaPLe [29]. However, most previous works fine-tune task-specific prompts, which limits their versatility and generalization [49].

# 3.2 POMP: Prompt Pre-Training

Now we present our task-agnostic prompt pre-training method: POMP. Once pre-trained, the learned prompt can be directly used for downstream tasks without fine-tuning (see Figure 2). As introduced in § 1, we propose to pre-train the soft prompt on the ImageNet-21K dataset for universal visual discrimination. Although the prompt tuning methods such as COOP and CoCoOp are parameter-efficient, they still incur computationally prohibitive training costs when applied to large-scale datasets with massive number of classes. Recall that the learnable parameters $\Theta$ are embedded in the text input, while the loss is calculated at the output layer of the text encoder. For every class description, we need to allocate nearly $15\mathrm{MB}$ of GPU memory to preserve the state of the entire frozen encoder (Transformer-base [53] with 12 layers), and propagate the gradient back through the last layer to the first layer, to update the soft prompt $\Theta$ . Accordingly, the computational and caching cost of prompt tuning is proportional to the number of classes $N$ . As shown in Figure 3, for general large-scale datasets like ImageNet-21K with more than twenty-thousand classes, traditional tuning methods will allocate $21\mathrm{K} \times 15\mathrm{MB}$ (more than $300\mathrm{GB}$ ) of GPU memory, which is generally prohibitive.

To enable prompt tuning on massive classes and acquire the capability of global visual discrimination, we introduce a training-efficient algorithm called POMP, which reduces the GPU memory and training time of prompt tuning dramatically. POMP has two major components: local contrast and local correction. The former decreases the number of classes for contrastive learning through negative class sampling, and the latter reduces the bias caused by local contrast by adjusting the similarity scores of negative classes. We detail these two components in the following.

# 3.2.1 Local Contrast

Discriminating all classes during contrastive learning is the source of training inefficiency. In order to alleviate this problem, we propose to narrow the scope of contrastive learning from global to local, and only require the model to identify the ground-truth class of the input image from a subset of the full class set. The class subset is sampled at each training step, allowing the model to discriminate within an ever-changing set of categories, and gradually restoring the relationship among all categories.

Specifically, given an input image, we sample K classes (K is much smaller than the total number of classes, N), including the ground-truth class y and K - 1 negative classes. We use a straightforward yet effective proposal distribution of uniform distribution for negative class sampling, where every negative class has an equal probability, i.e., $p = 1/(N - 1)$ , of being sampled. We also explore alternative types of proposal distribution, such as frequency-based and similarity-based distributions. However, our experiments reveal that POMP with the simple uniform distribution considers both common and rare classes, as well as easy and difficult classes, resulting to the best performance. Please refer to Appendix E.1 for details.

After sampling, we denote the set of the negative classes as $\mathcal{N}$ , with $|\mathcal{N}| = K - 1$ . By using the local contrast, we can significantly reduce the training overhead to a fraction of $K / N$ of the original one. Upon completion of training, we can use the full class set to compute the prediction probability of each image. Overall, the motivation behind our local contrast is analogous to that in the NCE-based contrastive learning frameworks [52, 57]. In these frameworks, the contrast is performed by sampling a batch of instances due to computational limitations and computing the loss within the batch as an empirical estimation for the expected contrastive loss [25, 1, 22, 51, 68].

# 3.2.2 Local Correction

Given that the local contrast component necessitates a reduced number of negative classes, the negative reinforcement in the vanilla gradient in (3) is diminished to $K / N$ . As a result, the prompt optimization direction is inevitably biased due to the absence of other negative classes. To mitigate this bias and enhance the model performance, we add a local correction term $m$ to the logits of the sampled negative classes $\mathbf{x}^{\top}\mathbf{w}_i^{(\Theta)} / \tau (i\neq y)$ , which serves as a margin [60, 45, 69] between the positive and the negative logits. Accordingly, the final prediction probability of POMP is denoted as:

$$
\tilde {P} (y \mid \mathbf {x}; \boldsymbol {\Theta}) = \frac {\exp (\mathbf {x} ^ {\top} \mathbf {w} _ {y} ^ {(\boldsymbol {\Theta})} / \tau)}{\exp (\mathbf {x} ^ {\top} \mathbf {w} _ {y} ^ {(\boldsymbol {\Theta})} / \tau) + \sum_ {i \sim \mathcal {N}} \exp (\mathbf {x} ^ {\top} \mathbf {w} _ {i} ^ {(\boldsymbol {\Theta})} / \tau + m)}. \tag {4}
$$

The local correction term m encourages the positive logit to be larger than the negative logits by a certain margin, resulting in a more stringent decision boundary:

$$
C _ {+}: \mathbf {x} ^ {\top} \mathbf {w} _ {y} ^ {(\boldsymbol {\Theta})} / \tau \geq \mathbf {x} ^ {\top} \mathbf {w} _ {i} ^ {(\boldsymbol {\Theta})} / \tau + m, i \neq y.
$$

Therefore, compared to the prediction probability without local correction, (4) makes the decision boundary more robust against the unsampled negative classes and enforces the learning of more discriminative class features [60]. This will improve model regularization and robustness across datasets and domains (to be shown in § 4.4). Different from other margin-based losses that use a fixed margin [11, 54], our margin $m$ is designed to adaptively adjust itself based on the value of $K$ :

$$
m = - \log \big ((K - 1) / (N - 1) \big). \tag {5}
$$

It is worth noting that m in (5) is a positive scalar. When K = N, all classes are included during optimization, and m equals zero. In this case, (4) degenerates to the standard prediction probability in (1). As the value of K decreases and the number of visible classes is reduced, the margin m increases to create space for potential class features in the representation space. Our adaptive margin outperforms the fixed margins (which are specified as hyper-parameters) for various K, allowing models to maintain optimal performance under different computing budgets (to be shown in § 4.4).

# 3.3 Zero-Shot Transfer Learning

As shown in Figure 2, after pre-training, our POMP prompt can be used to synthesize class features for classification with an arbitrary class set, supporting zero-shot inference on downstream datasets

and tasks. In order to plug the POMP prompt into other visual tasks like semantic segmentation and object detection, we adopt a two-stage framework. In stage one, we use a pre-trained proposal network to generate a set of mask or region proposals. In stage two, we classify each proposal with the class features generated by our POMP prompt. Experiments in § 4.3 will show that the POMP prompt can handle both pixel-level and region-level visual patterns, leading to improved performance in segmentation and detection tasks.

# 4 Experiments

# 4.1 POMP Prompt Pre-Training

We take CLIP (ViT/B-16) [42] as the backbone and conduct prompt pre-training on the ImageNet-21K dataset. The number of training samples for each class is 16 (16 shots), and the prompt length is 16. We sample 1,000 classes at each training step, i.e., K = 1000 in (4). See Appendix A for details.

# 4.2 End Task Setups and Implementation Details

To evaluate the generalization of the pre-trained prompt, we directly transfer it to downstream tasks and datasets. Appendix C lists the details of all the datasets. We follow previous works $[58, 61, 19, 67]$ to designate data belonging to two class sets as source data and target data, respectively. The proposal networks are pre-trained on the source data with the source class set, while conducting zero-shot evaluation on the target data with the target class set. There are two protocols for the source-target data split. The first is the open-vocabulary protocol, where the class set of one dataset is divided into two disjoint groups for the source and target data. The second protocol is the cross-dataset protocol, in which the source and target data are from two independent datasets with potentially overlapping class sets. See Appendix B for implementation details.

# 4.3 Results and Analysis

# 4.3.1 Prompt Pre-Training on ImageNet-21K

Table 1 shows the results of POMP prompt on the ImageNet-21K test set after pre-training. The traditional prompt learning methods (e.g., CoOp and MaPLe) are trained on the ImageNet-1K dataset due to their prohibitive computational cost if trained on ImageNet-21K (more than 300 GB of GPU memory). On the contrary, our POMP prompt, pre-trained on ImageNet-21K using less than 16 GB of GPU memory, achieves the highest accuracy of $25.3\%$ based on the CLIP (ViT-B/16) backbone, which surpasses Zeroshot-CLIP by $3.5\%$ and Linear Probe by $4.4\%$ . The VPT method, which uses visual prompts on the image side, does not require training overhead proportional to the number of classes, making it applicable to the ImageNet-21K dataset. VPT prepends independent learnable vectors to the hidden states of each layer in the visual backbone, surpassing linear probing and the previous prompt tuning methods. However, its performance based on ViT-B/16 is still 0.5% worse than ours, demonstrating that our POMP prompt can better distinguish a large number of general visual categories. In addition, our method is agnostic to the backbone architectures like ResNet and ViT, and the improvement is consistent.

Table 1: Performance on the ImageNet-21K test set. ZeroshotCLIP and Prompt Ensemble in the top block conduct zero-shot inference. CoOp and MaPLe, indicated in gray in the middle block, are trained on the ImageNet-1K dataset due to prohibitive GPU memory consumption if trained on ImageNet-21K. The remaining methods in the bottom block are trained on ImageNet-21K. 

<table><tr><td>Method</td><td>ResNet50</td><td>ViT-B/32</td><td>ViT-B/16</td></tr><tr><td>ZeroshotCLIP [42]</td><td>17.5</td><td>19.8</td><td>21.8</td></tr><tr><td>Prompt Ensemble [42]</td><td>18.8</td><td>20.9</td><td>23.5</td></tr><tr><td>CoOp [65]</td><td>16.6</td><td>18.1</td><td>20.8</td></tr><tr><td>MaPLe [29]</td><td>-</td><td>21.6</td><td>24.2</td></tr><tr><td>Linear Probing [42]</td><td>6.5</td><td>18.2</td><td>20.9</td></tr><tr><td>VPT [12]</td><td>-</td><td>21.8</td><td>24.8</td></tr><tr><td>POMP (Ours)</td><td>20.2</td><td>22.2</td><td>25.3</td></tr></table>

Cross-dataset and Cross-domain Image Classification. Our POMP prompt, which has been pre-trained on a large number of classes, demonstrates a strong generalization ability. As shown in Table 2, POMP achieves the highest average accuracy of $67.0\%$ when transferred to 10 downstream image classification datasets, outperforming CoOp by $3.1\%$ and surpassing the previous SOTA in

Table 2: Cross-dataset and cross-domain evaluation for image classification. The backbone is ViT/B-16. Overall, POMP achieves the highest average accuracy, indicating better generalization. 

<table><tr><td rowspan="2"></td><td colspan="11">Target (cross-dataset)</td><td colspan="5">Target (cross-domain)</td></tr><tr><td>Caltech101</td><td>OxfordPets</td><td>StanfordCars</td><td>Flowers102</td><td>Food101</td><td>Aircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td><td>Average</td><td>ImageNetV2</td><td>ImageNet-S</td><td>ImageNet-A</td><td>ImageNet-R</td><td>Average</td></tr><tr><td>hard prompt</td><td>93.3</td><td>88.2</td><td>65.6</td><td>67.4</td><td>85.3</td><td>23.7</td><td>62.6</td><td>44.3</td><td>42.0</td><td>65.1</td><td>63.7</td><td>60.9</td><td>46.1</td><td>47.8</td><td>74.0</td><td>57.2</td></tr><tr><td>CoOp [65]</td><td>93.7</td><td>89.1</td><td>64.5</td><td>68.7</td><td>85.3</td><td>18.5</td><td>64.2</td><td>41.9</td><td>46.4</td><td>66.6</td><td>63.9</td><td>64.2</td><td>48.0</td><td>49.7</td><td>75.2</td><td>59.3</td></tr><tr><td>CoCoOp [66]</td><td>94.4</td><td>90.1</td><td>65.3</td><td>71.9</td><td>86.1</td><td>22.9</td><td>67.4</td><td>45.7</td><td>45.4</td><td>68.2</td><td>65.7</td><td>64.1</td><td>48.8</td><td>50.6</td><td>76.2</td><td>59.9</td></tr><tr><td>LASP [5]</td><td>94.5</td><td>89.4</td><td>64.8</td><td>70.5</td><td>86.3</td><td>23.0</td><td>67.0</td><td>45.5</td><td>48.3</td><td>68.2</td><td>65.8</td><td>63.8</td><td>49.0</td><td>50.7</td><td>77.1</td><td>60.1</td></tr><tr><td>VPT [12]</td><td>93.7</td><td>90.6</td><td>65.0</td><td>70.9</td><td>86.3</td><td>24.9</td><td>67.5</td><td>46.1</td><td>45.9</td><td>68.7</td><td>66.0</td><td>64.2</td><td>49.2</td><td>51.3</td><td>77.0</td><td>60.4</td></tr><tr><td>MaPLe [29]</td><td>93.5</td><td>90.5</td><td>65.6</td><td>72.2</td><td>86.2</td><td>24.7</td><td>67.0</td><td>46.5</td><td>48.1</td><td>68.7</td><td>66.3</td><td>64.1</td><td>49.2</td><td>50.9</td><td>77.0</td><td>60.3</td></tr><tr><td>POMP (Ours)</td><td>95.0</td><td>89.5</td><td>66.8</td><td>72.4</td><td>86.3</td><td>25.6</td><td>67.7</td><td>46.2</td><td>52.1</td><td>68.5</td><td>67.0</td><td>63.8</td><td>49.8</td><td>51.6</td><td>77.9</td><td>60.8</td></tr></table>

7/10 datasets. This is due to the fact that, after learning with enormous long-tail categories, POMP can provide a more expressive context for fine-grained visual concepts such as specific objects and scenes, resulting in improved performance on datasets like StanfordCars (+1.2%) and Aircraft (+0.9%), as well as SUN397 (+0.7%) and EuroSAT (+4%). Furthermore, POMP is more robust to domain shift and achieves a new SOTA with 60.8% accuracy on 4 out-of-domain variants of the ImageNet dataset.

Training Efficiency. POMP also achieves comparable accuracy to the classical prompt tuning methods when fine-tuning on specific downstream datasets, but significantly reduces the training cost. Table 3 shows the performance of prompt tuning on ImageNet-1K, using a visual backbone of ViT-B/16 and 16 shots. The epoch is 50 for CoOp and POMP, and 10 for CoCoOp. CoOp generates all the 1000 class features at each training step, which consumes 28 GB of memory and takes 5.9 hours to finish the fine-tuning. The training time of CoCoOp is

Table 3: Prompt tuning on ImageNet-1K. POMP (K = 128) achieves comparable accuracy with CoOp and CoCoOp, but using less than 19% GPU memory and 50% training time. 

<table><tr><td>Method</td><td>Acc. (%)</td><td>GPU Mem. (GB)</td><td>Training Time (h)</td></tr><tr><td>CoOp</td><td>71.9</td><td>28.2</td><td>5.9</td></tr><tr><td>CoCoOp</td><td>70.1</td><td>28.3</td><td>27.5</td></tr><tr><td>POMP ( $K = 128$ )</td><td>71.2</td><td>5.3</td><td>2.7</td></tr><tr><td>POMP ( $K = 256$ )</td><td>71.4</td><td>8.8</td><td>3.3</td></tr><tr><td>POMP ( $K = 512$ )</td><td>71.6</td><td>15.9</td><td>4.2</td></tr></table>

even longer because it devises instance-specific prompts that require an independent forward pass for each image. Compared to these baselines, POMP (K = 128) achieves competitive accuracy on ImageNet-1K while using less than 19% of GPU memory and 50% of training time, demonstrating its superiority.

# 4.3.2 Open-Vocabulary Semantic Segmentation

Table 4 shows the results of our method on open-vocabulary COCO Stuff and Pascal VOC. POMP outperforms the previous state-of-the-art method, ZSSeg [61], with a higher hIoU of 39.1 and mIoU-unseen of 38.2 on COCO Stuff. On Pascal VOC, the improvement of POMP is more significant with +6.9 hIoU and +4.3 mIoU-unseen. Figure 4 illustrates qualitative results on open-vocabulary COCO-Stuff, where POMP demonstrates a stronger ability to distinguish background categories compared to ZSSeg. For example, in case (1), ZSSeg misclassifies the classes of playingfield as dirt, while the POMP prompt with richer contextual semantics better expresses the difference between regular land and the playingfield with specific textures, thus facilitating the matching of the visual region with the ground-truth class.

POMP also demonstrates its generalization ability in cross-dataset settings. Taking standard COCO Stuff as the source dataset for mask proposal network pre-training, POMP achieves 20.7 mIoU and 51.1 mIoU when transferred to the target datasets of ADE20K and PASCAL Context, respectively, outperforming ZSSeg by +1.3 mIoU and +0.3 mIoU. Overall, POMP obtains remarkable gains over previous works in all settings.

# 4.3.3 Open-Vocabulary Object Detection

Table 4: Comparison with state-of-the-art methods on COCO Stuff dataset and Pascal VOC dataset. POMP and ZSSeg share the same mask proposal network and training strategy. 

<table><tr><td rowspan="3">Method</td><td colspan="3">Open-Vocab COCO Stuff</td><td colspan="3">Open-Vocab Pascal VOC</td></tr><tr><td rowspan="2">hIoU</td><td colspan="2">mIoU</td><td rowspan="2">hIoU</td><td colspan="2">mIoU</td></tr><tr><td>seen</td><td>unseen</td><td>seen</td><td>unseen</td></tr><tr><td>SPNet [58]</td><td>16.8</td><td>20.5</td><td>14.3</td><td>21.8</td><td>73.3</td><td>15.0</td></tr><tr><td>ZS3 [4]</td><td>15.0</td><td>34.7</td><td>9.5</td><td>28.7</td><td>77.3</td><td>17.7</td></tr><tr><td>CaGNet [20]</td><td>18.2</td><td>35.5</td><td>12.2</td><td>39.7</td><td>78.4</td><td>25.6</td></tr><tr><td>ZegFormer [13]</td><td>34.8</td><td>36.6</td><td>33.2</td><td>73.3</td><td>86.4</td><td>63.6</td></tr><tr><td>ZSSeg [61]</td><td>37.8</td><td>39.3</td><td>36.3</td><td>77.5</td><td>83.5</td><td>72.5</td></tr><tr><td>POMP (Ours)</td><td>39.1</td><td>39.9</td><td>38.2</td><td>84.4</td><td>93.6</td><td>76.8</td></tr></table>

Table 5: Cross-dataset evaluation for semantic segmentation. The mask proposal network is pre-trained on standard COCO Stuff. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Source Dataset:Standard COCO Stuff</td><td colspan="3">Target Dataset:ADE20K</td><td colspan="3">Target Dataset:PASCAL Context</td></tr><tr><td>mIoU</td><td>fwIoU</td><td>pACC</td><td>mIoU</td><td>fwIoU</td><td>pACC</td><td>mIoU</td><td>fwIoU</td><td>pACC</td></tr><tr><td>ZSSeg [61]</td><td>40.8</td><td>49.0</td><td>62.7</td><td>19.5</td><td>48.7</td><td>60.0</td><td>50.8</td><td>64.1</td><td>75.7</td></tr><tr><td>POMP (Ours)</td><td>41.1</td><td>49.2</td><td>62.9</td><td>20.7</td><td>51.5</td><td>63.7</td><td>51.1</td><td>65.4</td><td>76.1</td></tr></table>

![](images/0fc3982ca4d18ecd0ee5305181bd2c0d439250960a3bf9c3fb97a3ed677849d5.jpg)

<details>
<summary>text_image</summary>

ZSSeg
(Baseline)
person
zirrland
person
piston racket
playingfield
POMP
(Ours)
person
piston racket
playingfield
Ground
-truth
person
piston racket
ground
(1)
</details>

![](images/438c0cd705c24290e3c9126760cd35347c0ae0672852c0935c1f3189556fe108.jpg)

<details>
<summary>text_image</summary>

grass
water-other
sheep
grass
plant-other
river
sheep
grass
plant-other
river
sheep
grass
(2)
</details>

Figure 4: Qualitative results on open-vocabulary COCO-Stuff. Compared to ZSSeg, POMP correctly identifies the background category of playingfield (left) and river (right).

Table 6: Cross-dataset evaluation for object detection. The region proposal network is pre-trained on standard LVIS. POMP and Detic share the same region proposal network and training strategy. 

<table><tr><td rowspan="2">Method</td><td colspan="6">Source Dataset: Standard LVIS</td><td colspan="6">Target Dataset: COCO</td><td colspan="6">Target Dataset: Objects365</td></tr><tr><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_s$ </td><td> $AP_m$ </td><td> $AP_l$ </td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_s$ </td><td> $AP_m$ </td><td> $AP_l$ </td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_s$ </td><td> $AP_m$ </td><td> $AP_l$ </td></tr><tr><td>ViLD* [19]</td><td>27.5</td><td>41.8</td><td>29.3</td><td>20.6</td><td>35.9</td><td>43.4</td><td>34.1</td><td>52.3</td><td>36.5</td><td>21.6</td><td>38.9</td><td>46.1</td><td>11.5</td><td>17.8</td><td>12.3</td><td>4.2</td><td>11.1</td><td>17.8</td></tr><tr><td>DetPro [15]</td><td>28.4</td><td>42.9</td><td>30.3</td><td>21.0</td><td>36.7</td><td>44.1</td><td>34.9</td><td>53.8</td><td>37.4</td><td>22.5</td><td>39.6</td><td>46.3</td><td>12.1</td><td>18.8</td><td>12.9</td><td>4.5</td><td>11.5</td><td>18.6</td></tr><tr><td>Detic [67]</td><td>36.8</td><td>50.7</td><td>38.6</td><td>26.1</td><td>46.7</td><td>51.7</td><td>38.8</td><td>56.0</td><td>41.9</td><td>25.6</td><td>42.2</td><td>50.0</td><td>15.6</td><td>22.1</td><td>16.8</td><td>6.1</td><td>15.6</td><td>23.8</td></tr><tr><td>POMP (Ours)</td><td>37.2</td><td>51.1</td><td>39.3</td><td>26.5</td><td>47.2</td><td>52.6</td><td>40.3</td><td>57.9</td><td>43.6</td><td>28.3</td><td>43.9</td><td>50.6</td><td>16.1</td><td>22.9</td><td>17.3</td><td>6.2</td><td>16.3</td><td>24.7</td></tr></table>

We compare POMP with state-of-the-art methods on the open-vocabulary LVIS benchmarks and report results in Table 7. POMP achieves $AP_{r}$ of 26.8 for object detection and 25.2 for instance segmentation. See Appendix D for qualitative results. Under the cross-dataset setting, we pre-train the visual backbone on the source dataset of standard LVIS, and evaluate the recognition ability on COCO and Object365. As shown in Table 6, compared to Detic, POMP provides a gain of $1.9 \, AP_{50}$ on COCO and $0.8 \, AP_{50}$ on Object365, respectively.

Table 7: Comparison with previous SOTA on LVIS dataset. $AP_{r}$ is the main evaluation metric for open-vocabulary object detection. 

<table><tr><td rowspan="2">Method</td><td colspan="4">Detection</td><td colspan="4">Instance segmentation</td></tr><tr><td> $AP_r$ </td><td> $AP_c$ </td><td> $AP_f$ </td><td>AP</td><td> $AP_r$ </td><td> $AP_c$ </td><td> $AP_f$ </td><td>AP</td></tr><tr><td>ViLD [19]</td><td>16.7</td><td>26.5</td><td>34.2</td><td>27.8</td><td>16.6</td><td>24.6</td><td>30.3</td><td>25.5</td></tr><tr><td>DetPro [15]</td><td>20.8</td><td>27.8</td><td>32.4</td><td>28.4</td><td>19.8</td><td>25.6</td><td>28.9</td><td>25.9</td></tr><tr><td>PromptDet [18]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>21.4</td><td>23.3</td><td>29.3</td><td>25.3</td></tr><tr><td>Detic [67]</td><td>26.7</td><td>36.4</td><td>40.3</td><td>36.3</td><td>24.9</td><td>32.5</td><td>35.6</td><td>32.4</td></tr><tr><td>POMP (Ours)</td><td>26.8</td><td>36.4</td><td>40.4</td><td>36.2</td><td>25.2</td><td>33.0</td><td>35.6</td><td>32.7</td></tr></table>

# 4.4 Ablation Study

We decouple the two components of local contrast and local correction in POMP, and conduct an ablation study to examine their individual contributions. Since removing the local contrast component will lead to prohibitive training cost, we investigate the impact of this component by varying the number of sampled classes K. As shown in Table 8, the performance of POMP improves as K increases. As discussed in § 4.3.1 and Table 3, the local contrast component balances accuracy and cost by adjusting K.

Table 8: Ablation on the local contrast and local correction in POMP based on the CLIP (ViT/B-16). 

<table><tr><td>Method</td><td>ImageNet-21K</td><td>Cross-dataset (10 Avg.)</td><td>Cross-domain (4 Avg.)</td></tr><tr><td>POMP ( $K = 100$ )</td><td>24.1</td><td>65.5</td><td>59.5</td></tr><tr><td>POMP ( $K = 500$ )</td><td>24.9</td><td>66.5</td><td>60.0</td></tr><tr><td>POMP ( $K = 1000$ )</td><td>25.3</td><td>67.0</td><td>60.8</td></tr><tr><td>- local correction</td><td>25.0 (-0.3)</td><td>65.8 (-1.2)</td><td>59.8 (-1.0)</td></tr></table>

On the other hand, as shown in Table 8, removing the local correction component from POMP (K = 1000) results in a decline of 1.2 and 1.0 in the average accuracy of cross-dataset and cross-domain transfer, respectively. This indicates that local correction significantly improves the

![](images/521ce4e2bc0f35a4f863dfcaee718867d081b4cbffd9cdd12da0c76a37f2590f.jpg)

<details>
<summary>scatter</summary>

| Method | ℓ_align | ℓ_uniform | Value |
|---|---|---|---|
| POMP (w/o local correction) | 1.36 | -1.05 | 65.8 |
| POMP | 1.38 | -1.35 | 67.0 |
| MaPLe | 1.42 | -1.30 | 66.3 |
</details>

Figure 5: $\ell_{align}$ and $\ell_{uniform}$ of POMP. For both measures, lower numbers are better. The color of circles and the numbers in the boxes denote the average cross-dataset accuracy over 10 datasets (higher is better).

![](images/dc4248333e02bb473a3e9f248694e22b3a7fdeba9ec09c290cbae8e9f61f7914.jpg)

<details>
<summary>scatter</summary>

| Group | X Coordinate | Y Coordinate |
|-------|--------------|--------------|
| Red   | 0.1          | 0.8          |
| Red   | 0.2          | 0.75         |
| Red   | 0.3          | 0.7          |
| Red   | 0.4          | 0.65         |
| Red   | 0.5          | 0.6          |
| Red   | 0.6          | 0.55         |
| Red   | 0.7          | 0.5          |
| Red   | 0.8          | 0.45         |
| Red   | 0.9          | 0.4          |
| Red   | 1.0          | 0.35         |
| Yellow| 0.1          | 0.85         |
| Yellow| 0.2          | 0.8          |
| Yellow| 0.3          | 0.75         |
| Yellow| 0.4          | 0.7          |
| Yellow| 0.5          | 0.65         |
| Yellow| 0.6          | 0.6          |
| Yellow| 0.7          | 0.55         |
| Yellow| 0.8          | 0.5          |
| Yellow| 0.9          | 0.45         |
| Yellow| 1.0          | 0.4          |
| Green | 0.1          | 0.8          |
| Green | 0.2          | 0.75         |
| Green | 0.3          | 0.7          |
| Green | 0.4          | 0.65         |
| Green | 0.5          | 0.6          |
| Green | 0.6          | 0.55         |
| Green | 0.7          | 0.5          |
| Green | 0.8          | 0.45         |
| Green | 0.9          | 0.4          |
| Green | 1.0          | 0.35         |
</details>

(a) Aircraft.

![](images/882beb0a727320ee6dc99d39d7e0facb8e0d1005a19f5255c1cd0293be6de9d5.jpg)  
(b) UCF101.   
Figure 6: Projection of image features (points), class features of POMP (intersections of solid lines and sphere) and class features of CoOp (intersections of light dash-dot lines and sphere). Each color represents a class. Class features of POMP have better alignment with centroids of the corresponding images, and are distributed with better uniformity.

generalization of the pre-trained prompt. Furthermore, we analyze the impact of the adaptive margin (5) in the local correction. We pre-train prompts on ImageNet-21K with varying $m$ values (0, 0.5, 1, 1.5) and report the cross-dataset accuracy. Notably, our local correction method dynamically sets $m$ to 1.5 when $K = 319$ , and $m$ to 1 when $K = 1000$ . The results are shown in Table 9. When the number of sampled classes is relatively small $(K = 319)$ , increasing the margin creates more space for potential negative classes, thereby improves cross-dataset accuracy. Conversely, for large K values (e.g., 1000), imposing a very large margin $(m = 1.5)$ disrupts the natural class distribution and diminishes generalization ability. Overall, compared to the fixed margins [11, 54], our adaptive margin decreases as K increases, achieving optimal performance across different computing budgets (controlled by K) and sparing the time for extensive hyper-parameter search. See Appendix E for more ablation studies on the number of shots and prompt length.

Table 9: Ablation on the adaptive margin m in the local correction. 

<table><tr><td>m</td><td>K=319</td><td>K=1000</td></tr><tr><td>0</td><td>65.2</td><td>65.8</td></tr><tr><td>0.5</td><td>65.7</td><td>66.1</td></tr><tr><td>1</td><td>66.2</td><td>67.0 (Ours)</td></tr><tr><td>1.5</td><td>66.5 (Ours)</td><td>66.4</td></tr></table>

# 4.5 Understanding the Pre-trained Prompt

To better understand the pre-trained prompt, we analyze the feature space of POMP through the properties of alignment and uniformity [56]. Intuitively, the image feature and its ground-truth class feature are supposed to stay closed (alignment). Besides, all the class features should be uniformly distributed to preserve maximal information and make the categories more distinguishable (uniformity). We use the alignment and uniformity loss in the vision-and-language field [45, 63] for representation probing. The alignment loss calculates the expected distance between features of an image x and its ground truth class $\mathbf{w}_y^{(\Theta)}$ :

$$
\ell_ {\text { align }} \triangleq \underset {(\mathbf {x}, y) \in \mathcal {D}} {\mathbb {E}} \left\| \mathbf {x} - \mathbf {w} _ {y} ^ {(\boldsymbol {\Theta})} \right\| ^ {2}, \tag {6}
$$

while the uniformity loss measures how well the class features $\mathbf{w}^{(\Theta)}$ are uniformly distributed:

$$
\ell_ {\text { uniform }} \triangleq \log_ {\substack {1 \leqslant i, j \leqslant N, \\ i \neq j}} \mathbb {E} \exp (- 2 \| \mathbf {w} _ {i} ^ {(\boldsymbol {\Theta})} - \mathbf {w} _ {j} ^ {(\boldsymbol {\Theta})} \| ^ {2}). \tag{7}
$$

We visualize the alignment and uniformity measures of POMP and the previous SOTA, MaPLe, in Figure 5. For both measures, lower numbers are better. The circle of POMP in the figure is located in the lower left with the lightest color, indicating relatively smaller losses and the best performance under the cross-dataset setting. Compared with the method without local correction, POMP significantly reduces the uniformity loss at only a slight expense of alignment. In other words, our pre-trained prompt not only ensures the alignment of the image and the ground-truth class, but

also disperses the class features in the representation space, thereby improving the generalization and robustness of the model. The visualization of the feature space in Figure 6 also verifies our superiority. The endpoints of the POMP class features are closer to the centroids of the image features, indicating better alignment and reduced $\ell_{align}$ loss (from 1.39 to 1.36 on Aircraft and from 1.41 to 1.36 on UCF101). Furthermore, the larger angles between the POMP class features demonstrate better feature uniformity and reduced $\ell_{uniform}$ loss (from -0.66 to -0.81 on Aircraft and from -0.95 to -1.23 on UCF101) compared to CoOp.

# 5 Conclusion

We present POMP to pre-train a general soft prompt on ImageNet-21K for universal visual discrimination. The learned prompt can be easily plugged into various visual recognition datasets and tasks for zero-shot inference. Experiments on open-vocabulary image classification, semantic segmentation, and object detection show that POMP surpasses previous methods by a considerable margin.

# Limitations

To facilitate future research, we analyze the limitations in our work and propose potential solutions. (1) We present the local contrast and use the loss within a subsampled class set as an empirical estimation for the expected contrastive loss within the full class set. However, the theoretical risk of such an estimation is urged to be investigated. (2) ImageNet-21K comprises a vast number of classes that are organized based on a semantic structure. By leveraging the hyponym and hypernym relations provided by WordNet synsets, we can derive the parent class and a list of child classes for each class. We believe that utilizing the semantic information holds the potential to further enhance performance. (3) Despite the excellent performance exhibited by our pre-trained prompt, its interpretability poses a significant challenge because the context vectors are optimized in a continuous space. We leave it as future work.

# References

[1] Philip Bachman, R. Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In Neural Information Processing Systems, 2019.   
[2] M Saiful Bari, Aston Zhang, Shuai Zheng, Xingjian Shi, Yi Zhu, Shafiq R. Joty, and Mu Li. Spt: Semi-parametric prompt tuning for multitask prompted learning. 2022.   
[3] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 - mining discriminative components with random forests. In ECCV, 2014.   
[4] Max Bucher, Tuan-Hung Vu, Matthieu Cord, and Patrick Pérez. Zero-shot semantic segmentation. ArXiv, abs/1906.00817, 2019.   
[5] Adrian Bulat and Georgios Tzimiropoulos. Language-aware soft prompting for vision & language foundation models. ArXiv, abs/2210.01115, 2022.   
[6] Holger Caesar, Jasper R. R. Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1209–1218, 2016.   
[7] Jize Cao, Zhe Gan, Yu Cheng, Licheng Yu, Yen-Chun Chen, and Jingjing Liu. Behind the scene: Revealing the secrets of pre-trained vision-and-language models. In European Conference on Computer Vision, 2020.   
[8] Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In Neural Information Processing Systems, 2021.   
[9] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 3606–3613, 2014.

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, K. Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.   
[11] Jiankang Deng, J. Guo, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4685–4694, 2018.   
[12] Mohammad Mahdi Derakhshani, Enrique Sanchez, Adrian Bulat, Victor Costa, Cees G. M. Snoek, Georgios Tzimiropoulos, and Brais Martínez. Variational prompt tuning improves generalization of vision-language models. ArXiv, abs/2210.02390, 2022.   
[13] Jian Ding, Nan Xue, Guisong Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11573-11582, 2021.   
[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ArXiv, abs/2010.11929, 2020.   
[15] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guo Chun Li. Learning to prompt for open-vocabulary object detection with vision-language model. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14064–14073, 2022.   
[16] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John M. Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International Journal of Computer Vision, 88:303–338, 2010.   
[17] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. 2004 Conference on Computer Vision and Pattern Recognition Workshop, pages 178–178, 2004.   
[18] Chengjian Feng, Yujie Zhong, Zequn Jie, Xiangxiang Chu, Haibing Ren, Xiaolin Wei, Weidi Xie, and Lin Ma. Promptdet: Towards open-vocabulary detection using uncurated images. In European Conference on Computer Vision, 2022.   
[19] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Zero-shot detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921, 2021.   
[20] Zhangxuan Gu, Siyuan Zhou, Li Niu, Zihan Zhao, and Liqing Zhang. Context-aware feature generation for zero-shot semantic segmentation. Proceedings of the 28th ACM International Conference on Multimedia, 2020.   
[21] Agrim Gupta, Piotr Dollár, and Ross B. Girshick. Lvis: A dataset for large vocabulary instance segmentation. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5351–5359, 2019.   
[22] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9726–9735, 2019.   
[23] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2015.   
[24] Patrick Helber, Benjamin Bischke, Andreas R. Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12:2217–2226, 2019.   
[25] Olivier J. Hénaff, A. Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aäron van den Oord. Data-efficient image recognition with contrastive predictive coding. ArXiv, abs/1905.09272, 2019.

[26] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Lixuan Zhu, Samyak Parajuli, Mike Guo, Dawn Xiaodong Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8320–8329, 2020.   
[27] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Xiaodong Song. Natural adversarial examples. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15257–15266, 2019.   
[28] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, 2021.   
[29] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. ArXiv, abs/2210.03117, 2022.   
[30] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. In European Conference on Computer Vision, 2019.   
[31] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. 2013 IEEE International Conference on Computer Vision Workshops, pages 554–561, 2013.   
[32] Boyi Li, Kilian Q. Weinberger, Serge J. Belongie, Vladlen Koltun, and René Ranftl. Language-driven semantic segmentation. ArXiv, abs/2201.03546, 2022.   
[33] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. ArXiv, abs/2110.05208, 2021.   
[34] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, 2014.   
[35] Yuning Lu, Jianzhuang Liu, Yonggang Zhang, Yajing Liu, and Xinmei Tian. Prompt distribution learning. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5196–5205, 2022.   
[36] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew B. Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. ArXiv, abs/1306.5151, 2013.   
[37] George A. Miller. Wordnet: A lexical database for english. Commun. ACM, 38:39–41, 1995.   
[38] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Loddon Yuille. The role of context for object detection and semantic segmentation in the wild. 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 891–898, 2014.   
[39] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. In European Conference on Computer Vision, 2022.   
[40] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, pages 722-729, 2008.   
[41] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI, USA, June 16-21, 2012, pages 3498–3505. IEEE Computer Society, 2012.   
[42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

[43] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. Denseclip: Language-guided dense prediction with context-aware prompting. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18061–18070, 2021.   
[44] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In International Conference on Machine Learning, 2019.   
[45] Shuhuai Ren, Lei Li, Xuancheng Ren, Guangxiang Zhao, and Xu Sun. Delving into the openness of CLIP. In Findings of the Association for Computational Linguistics: ACL 2023. Association for Computational Linguistics, jul 2023.   
[46] Shuhuai Ren, Junyang Lin, Guangxiang Zhao, Rui Men, An Yang, Jingren Zhou, Xu Sun, and Hongxia Yang. Learning relation alignment for calibrated cross-modal retrieval. In Annual Meeting of the Association for Computational Linguistics, 2021.   
[47] T. Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses. ArXiv, abs/2104.10972, 2021.   
[48] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8429–8438, 2019.   
[49] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models. ArXiv, abs/2209.07511, 2022.   
[50] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. ArXiv, abs/1212.0402, 2012.   
[51] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In European Conference on Computer Vision, 2019.   
[52] Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. ArXiv, abs/1807.03748, 2018.   
[53] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. ArXiv, abs/1706.03762, 2017.   
[54] H. Wang, Yitong Wang, Zheng Zhou, Xing Ji, Zhifeng Li, Dihong Gong, Jin Zhou, and Wei Liu. Cosface: Large margin cosine loss for deep face recognition. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5265–5274, 2018.   
[55] Haohan Wang, Songwei Ge, Eric P. Xing, and Zachary Chase Lipton. Learning robust global representations by penalizing local predictive power. In Neural Information Processing Systems, 2019.   
[56] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. ArXiv, abs/2005.10242, 2020.   
[57] Zhirong Wu, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3733–3742, 2018.   
[58] Yongqin Xian, Subhabrata Choudhury, Yang He, Bernt Schiele, and Zeynep Akata. Semantic projection network for zero- and few-label semantic segmentation. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8248–8257, 2019.   
[59] Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 3485–3492, 2010.   
[60] Jiahao Xie, Xiaohang Zhan, Ziwei Liu, Yew Soon Ong, and Chen Change Loy. Delving into inter-image invariance for unsupervised visual representations. International Journal of Computer Vision, 130:2994 – 3013, 2020.

[61] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for zero-shot semantic segmentation with pre-trained vision-language model. ArXiv, abs/2112.14757, 2021.   
[62] Lewei Yao, Runhu Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. ArXiv, abs/2111.07783, 2021.   
[63] Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, and Chen Change Loy. Unified vision and language prompt learning. ArXiv, abs/2210.07225, 2022.   
[64] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5122–5130, 2017.   
[65] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130:2337 - 2348, 2021.   
[66] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16795–16804, 2022.   
[67] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Phillip Krahenbuhl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision, 2022.   
[68] Xingyi Zhou, Vladlen Koltun, and Philipp Krähenbühl. Probabilistic two-stage detection. ArXiv, abs/2103.07461, 2021.   
[69] Benjin Zhu, Junqiang Huang, Zeming Li, Xiangyu Zhang, and Jian Sun. Eqco: Equivalent rules for self-supervised contrastive learning. ArXiv, abs/2010.01929, 2020.

# A Pre-training Details

We conduct prompt pre-training on the ImageNet-21K dataset (official winter 2021 released version $^{2}$ ). We follow the processing methods in [47], which involves cleaning invalid classes, allocating 50 images per class for a validation split, and crop-resizing all the images to 224 resolution. We conduct all the experiments on 8×Nvidia V100 GPUs. For pre-training, the learnable vector is randomly initialized by drawing from a zero-mean Gaussian distribution with a standard deviation equal to 0.02. We use the SGD optimizer with an initial learning rate of 0.002, decayed by the cosine annealing rule. The batch size is 32, and the maximum epoch is 20.

For the mask proposal network and region proposal network pre-training, we strictly follow the settings of ZSSeg [61] and Detic [67], respectively. Specifically, we take MaskFormer [8] with ResNet-101 [23] as the mask proposal network. We use an AdamW optimizer with the initial learning rate of 1e-4, weight decay of 1e-4, a backbone multiplier of 0.1, and a poly learning rate policy with a power of 0.9. Besides, we take CenterNet2 [68] detector with ImageNet-21k pre-trained ResNet-50 [47] as the region proposal network. We use an Adam optimizer with learning rate 2e-4. Other tricks like Federated Loss, repeat factor sampling, and large scale jittering are incorporated to further improve the performance. As with Detic, we leverage both region-level and image-level supervision. We always first train a converged base-class-only model (4× schedule) and fine-tune it with additional image-labeled data for another 4× schedule.

# B Setting for Segmentation and Detection

Table 10 outlines the settings for semantic segmentation and object detection. We further introduce the settings in detail from three perspectives: backbone, data processing, and prompt.

Backbone. In general, we adopt a two-stage framework for these two tasks. At stage one, we use a pre-trained proposal network to generate a set of mask or region proposals. At stage two, we classify each proposal with the class features generated by our POMP prompt. For semantic segmentation, our POMP shares the same visual backbone as ZSSeg [61], which uses a pre-trained MaskFormer [8] with ResNet-101 [23] as default backbone to extract a set of binary masks. For object detection, our POMP shares the same visual backbone with Detic [67], which takes CenterNet2 [68] detector with ResNet-50 as its backbone, and leverages both region-level and image-level supervision.

Data Processing. We follow previous work $[58, 61, 19, 67]$ to designate data belonging to two class sets as source data and target data, respectively. The proposal networks are pre-trained on the source data with the source class set, while conducting zero-shot evaluation on the target data with the target class set. There are two protocols for the source-target data split. The first is the open-vocabulary protocol, where the class set of one dataset is divided into two disjoint groups for the source and target data, respectively. The second protocol is the cross-dataset protocol, in which the source and target data are from two independent datasets with potentially overlapping class sets.

We introduce the details of class set splitting in the open-vocabulary protocol. COCO Stuff and Pascal VOC 2012 are the two semantic segmentation datasets using the open-vocabulary protocol. Following previous settings $[58, 61]$ , a total of 171 annotated classes in COCO Stuff are divided into 156 seen classes and 15 unseen classes. For Pascal VOC 2012, a total of 20 classes are divided into 15 seen classes and 5 unseen classes, and the provided augmented annotations are used. LVIS is the object detection dataset using the open-vocabulary protocol. The standard LVIS dataset contains object detection and instance segmentation labels for 1203 classes. The classes are divided into three groups: frequent, common, and rare, based on the number of training images. According to previous work $[19]$ , the data from the 866 frequent and common classes are considered the source data, while those from the remaining 337 rare classes are the target data in testing. We note that Detic utilizes both box-supervised data from LVIS as well as image-supervised data from ImageNet-21K that overlaps with LVIS (997 classes, 277 of which are novel classes). This allows Detic to demonstrate transfer not only from base to novel classes, but also from image-level to box-level recognition. Since Detic is the closest existing method to ours that leverages ImageNet-21K, we chose it as a strong baseline and followed its setup for fair comparison.

Table 10: Settings for semantic segmentation and object detection. 

<table><tr><td>Task</td><td>Proposal Network</td><td>Setting</td><td>Source Data and Class Set (for proposal network pre-training)</td><td>Target Data and Class Set (for zero-shot evaluation)</td></tr><tr><td rowspan="3">Semantic Segmentation</td><td rowspan="3">MaskFormer (Mask Proposal Network)</td><td>Open-vocab COCO Stuff</td><td>COCO Stuff (seen)</td><td>COCO Stuff (unseen)</td></tr><tr><td>Open-vocab PASCAL VOC</td><td>PASCAL VOC (seen)</td><td>PASCAL VOC (unseen)</td></tr><tr><td>Cross-dataset</td><td>COCO Stuff</td><td>ADE20K / PASCAL Context</td></tr><tr><td rowspan="2">Object Detection</td><td rowspan="2">CenterNet2 detector (Region Proposal Network)</td><td>Open-vocab LVIS</td><td>LVIS (frequent+common)+ ImageNet-21K (overlaps with LVIS)</td><td>LVIS (rare)</td></tr><tr><td>Cross-dataset</td><td>LVIS+ ImageNet-21K (overlaps with LVIS)</td><td>COCO / Object365</td></tr></table>

Prompt. ZSSeg provides two kinds of prompts: hand-crafted prompts and learning-based prompts. Hand-crafted prompts include single prompt, i.e., “a sculpture of a [CLASSNAME]”, as well as ImageNet prompts $[42]$ and ViLD prompts $[19]$ , which are used for prompt ensemble and consist of 80 and 14 hard prompts, respectively. The learning-based prompt is obtained by fine-tuning a randomly initialized soft prompt on the source data. Accordingly, for a fair comparison, we conducted two sets of experiments based on whether to use the source data for prompt fine-tuning. (1) The results of ZSSeg with various hard-crafted prompts and the pre-trained POMP prompt without access to the source data can be found in Table 13 in Appendix E.3. (2) The results of ZSSeg with learning-based prompts initialized from random vectors and our pre-trained POMP prompt, both using source data for further fine-tuning, can be found in Table 4 and Table 5 in § 4.3.2. Detic has also extensively delved into intricate prompts, such as “a photo of a [CLASS] in the scene”. Moreover, it has made endeavors to employ synonyms for each category. Nevertheless, its ultimate recommendation is to use a simple yet effective prompt, i.e., “a [CLASSNAME]”, and all its released checkpoints are based on this prompt. We strictly adhere to Detic’s best practice, the evaluation of Detic and POMP in § 4.3.3 are both conducted without any further prompt tuning on the source data.

# C Datasets

The details of the downstream datasets for image classification, semantic segmentation, and object detection are shown in Table 11.

Image Classification. For cross-dataset image classification, we evaluate the performance of POMP on 10 downstream datasets, including Caltech-101 [17], Oxford-Pets [41], Stanford Cars [31], Oxford-Flowers102 [40], Food-101 [3], FGVC Aircraft [36], EuroSAT [24], SUN-397 [59], Describable Textures (DTD) [9], UCF-101 [50]. We also conduct zero-shot evaluation on 4 out-of-domain datasets derived from ImageNet [10], including ImageNetV2 [44], ImageNet-S [55], ImageNet-A [27], and ImageNet-R [26], to evaluate the domain generalization capability of our method.

Semantic Segmentation. We perform open-vocab semantic segmentation on COCO Stuff $[6]$ and Pascal VOC 2012 $[16]$ . Following previous notation and settings $[58, 61]$ , we split the class set into seen and unseen classes, where data for seen classes is considered the source data and data for unseen classes is considered the target data. The major measures for evaluation include mIoU and the harmonic mean IoU (hIoU) among both seen and unseen classes $[61]$ . The hIoU is defined as:

$$
\mathrm{hIoU} = \frac {2 \times \mathrm{mIoU} _ {\text { seen }} \times \mathrm{mIoU} _ {\text { unseen }}}{\mathrm{mIoU} _ {\text { seen }} + \mathrm{mIoU} _ {\text { unseen }}}
$$

We also conduct cross-dataset evaluation, which takes the standard COCO Stuff dataset as the source dataset for pre-training a mask proposal network, and then conducts zero-shot inference on ADE20K [64] and PASCAL Context [38].

Object Detection. We evaluate the performance of POMP on the object detection dataset LVIS [21] under the open-vocabulary setting proposed by [19]. The source data consists of box-level data from LVIS's 866 frequent and common classes, as well as image-level data from ImageNet-21K that overlaps with LVIS. The target data for testing comprises the remaining 337 rare classes in LVIS. We take $\mathrm{AP}_r$ , i.e., AP on rare classes, as the major measure. $\mathrm{AP}_f$ and $\mathrm{AP}_c$ , i.e., AP on frequent and common classes, are also reported. In the cross-dataset setting, the region proposal network is

Table 11: Datasets in our experiments. 

<table><tr><td>Dataset</td><td>Classes</td><td>Train Size</td><td>Test Size</td><td>Metric</td></tr><tr><td colspan="5">Datasets of Image Classification</td></tr><tr><td>Caltech-101 [17]</td><td>102</td><td>3,060</td><td>6,086</td><td>mean per-class accuracy</td></tr><tr><td>Oxford-IIIT Pets [41]</td><td>37</td><td>3,680</td><td>3,669</td><td>mean per-class accuracy</td></tr><tr><td>Stanford Cars [31]</td><td>196</td><td>8,144</td><td>8,041</td><td>accuracy</td></tr><tr><td>Oxford Flowers-102 [40]</td><td>102</td><td>2,040</td><td>6,149</td><td>mean per-class accuracy</td></tr><tr><td>Food-101 [3]</td><td>101</td><td>75,750</td><td>25,250</td><td>accuracy</td></tr><tr><td>FGVC Aircraft [36]</td><td>100</td><td>6,667</td><td>3,333</td><td>mean per-class accuracy</td></tr><tr><td>SUN-397 [59]</td><td>397</td><td>15,880</td><td>19,850</td><td>accuracy</td></tr><tr><td>Describable Textures (DTD) [9]</td><td>47</td><td>3,760</td><td>1,880</td><td>accuracy</td></tr><tr><td>EuroSAT [24]</td><td>10</td><td>10,000</td><td>5,000</td><td>accuracy</td></tr><tr><td>UCF-101 [50]</td><td>101</td><td>7,639</td><td>3,783</td><td>accuracy</td></tr><tr><td>ImageNetV2 [44]</td><td>1,000</td><td>10,000</td><td>10,000</td><td>accuracy</td></tr><tr><td>ImageNet-S [55]</td><td>1,000</td><td>50,889</td><td>50,889</td><td>accuracy</td></tr><tr><td>ImageNet-A [27]</td><td>200</td><td>7,500</td><td>7,500</td><td>accuracy</td></tr><tr><td>ImageNet-R [26]</td><td>200</td><td>30,000</td><td>30,000</td><td>accuracy</td></tr><tr><td colspan="5">Datasets of Semantic Segmentation</td></tr><tr><td>COCO Stuff [6]</td><td>171</td><td>117K</td><td>5K</td><td>mIoU (seen/unseen), hIoU</td></tr><tr><td>PASCAL VOC [16]</td><td>20</td><td>11,185</td><td>1,449</td><td>mIoU (seen/unseen), hIoU</td></tr><tr><td>ADE20K [64]</td><td>150</td><td>20K</td><td>3K</td><td>mIoU, fwIoU, pACC</td></tr><tr><td>PASCAL Context [38]</td><td>59</td><td>10,103</td><td>9,637</td><td>mIoU, fwIoU, pACC</td></tr><tr><td colspan="5">Datasets of Object Detection</td></tr><tr><td>LVIS [21]</td><td>1,203</td><td>100,170</td><td>19,822</td><td> $AP_r$ ,  $AP_c$ ,  $AP_f$ , AP</td></tr><tr><td>COCO [34]</td><td>80</td><td>118K</td><td>5K</td><td>AP,  $AP_{50}$ ,  $AP_{75}$ ,  $AP_s$ ,  $AP_m$ ,  $AP_l$ </td></tr><tr><td>Object365 [48]</td><td>365</td><td>600K</td><td>38K</td><td>AP,  $AP_{50}$ ,  $AP_{75}$ ,  $AP_s$ ,  $AP_m$ ,  $AP_l$ </td></tr></table>

pre-trained on the source dataset, which includes standard LVIS and ImageNet-21K (overlapping with LVIS). It is then directly used for inference on two target datasets: COCO [34] and Object365 [48]. We use AP, $AP_{50}$ , $AP_{75}$ , $AP_{s}$ , $AP_{m}$ , and $AP_{l}$ the evaluation metrics.

# D Qualitative Results for Semantic Segmentation and Object Detection

In this section, we provide more qualitative results of our POMP for semantic segmentation and object detection. Figure 7 shows another three cases on open-vocabulary COCO-Stuff segmentation. POMP demonstrates a stronger ability than ZSSeg in the recognition of background classes. In case (1), POMP correctly identified the dirt and plant-other in the scene, instead of marking all these areas as grass. In case (2) and (3), POMP recognizes the classes of clouds and tree, respectively, while ZSSeg misclassifies them as sky-other and bush. However, POMP misses some objects of sheep located at the edge in case (2) and neglects the object of branch in case (3), indicating it still has insufficient recognition of small objects. For object detection, Figure 8 illustrates qualitative results on LVIS images. Base and novel categories are shown in purple and green, respectively. POMP identifies regions from the novel class without using the corresponding 1.2K detection annotations, demonstrating its generalization in the wild.

# E More Ablation Study

# E.1 Ablation on Proposal Distribution

As introduced in § 3.2, we also investigate other types of proposal distribution for local contrast and negative class sampling. The first is the frequency distribution $Q^{(f)}$ , which samples the negative class $i$ based on the number of training samples belonging to this class. Note that the original ImageNet-21K is class-imbalanced, i.e., the number of training samples belonging to common classes is larger than those belonging to rare classes, which can roughly reflect the long-tail distribution of object categories in nature. The frequency distribution will allow for more sampling of common classes while suppressing the exposure of rare classes in prompt tuning. Let $M_i$ be the number of

(1)   
![](images/b943c6abbae7b0aa450dc0d59ed1449258de5b227641c3e27c9b8d5cfafa6592.jpg)

<details>
<summary>text_image</summary>

grass
zebra
</details>

![](images/228cdde65a01b869c8f0b5b8790fdf7de32875b8219cd1c76e2f5af0e9418b91.jpg)

<details>
<summary>text_image</summary>

plant-other
dirt
grass
zebra
</details>

![](images/0309997db65325ab4b53a3dd7dc43207f3e888e154f5b1eece66a23964c32f74.jpg)

<details>
<summary>text_image</summary>

plant-other
dirt
grass
zebra
</details>

(2)   
![](images/1435209d5e143f57573d072cb44ed8f0f2f49ae1502069dfd8c31b9a24c58d40.jpg)

<details>
<summary>text_image</summary>

tree
sheep
sky-other
grass
</details>

![](images/b55f42de2633d21e07b354cee13fe091f0f7170ff786063e7ae2a294c11d7820.jpg)

<details>
<summary>text_image</summary>

tree
clouds
sheep
grass
</details>

![](images/b313aab96291754ca971ca40650d1ba7259769d32e56c9d1a497fee7d6e19d14.jpg)

<details>
<summary>text_image</summary>

tree
clouds
sheep
grass
</details>

(3)   
![](images/81ef5feb05fe7a2ec16531a487064b0b8e9508c98d3d38bf1629f0c81e38b0b4.jpg)

<details>
<summary>text_image</summary>

bush
tree
giraffe
plant-
other
</details>

ZSSeg (Baseline)

![](images/89bbf5ae90637b0c6823a21bb4a8f496bfa5f880baf80c878d7d34e57545d97e.jpg)

<details>
<summary>natural_image</summary>

Illustration of giraffes feeding from a tree, labeled 'giraffe' (no text on diagram itself)
</details>

POMP (Ours)

![](images/20632fe332286bcc57e9b471d19a83b976a312ca72d922991a095a4a5b61c78b.jpg)

<details>
<summary>text_image</summary>

tree
branch
giraffe
</details>

Ground-truth   
Figure 7: More qualitative results on open-vocabulary COCO-Stuff segmentation.

![](images/5727f3023ef4585a433579680dfd42599a13c355115c67c23fb1c1e451fb2fbe.jpg)

<details>
<summary>pie</summary>

| Item | Percentage (%) |
|---|---|
| boiled_egg | 78 |
| dish_4% / 69% | egg 81 |
| egg_80% / 59% | egg 52 |
| egg 61% | egg 52 |
| gravy_boat | 88 |
| gravy_boat | 75 |
| bowl | 81 |
| candy_bar_81% / 66% | bar 74% |
| candy_berry_66% / 8% | ke 52 |
| candy_candy_bar_87% / 63% | bar 52 |
| candy_berry_66% / 8% | ke 6 |
| candy_candy_bar_87% / 63% | ke 6 |
| hudge_10% (cigarette) | 66 |
| hudge_10% (cigarette) | 66 |
| hudge_10% (cigarette) | 66 |
| hudge_10% (cigarette) | 66 |
| hudge_10% (cigarette) | 66 |
| hudge_10% (cigarette) | 66 |
| hudge_10% (cigarette) - chocolate_cake | 61.5 |
| hudge_10% (cigarette) - chocolate_mouse | 64 |
The chart displays a single data point for the total sum of all items. The values are expressed as percentages relative to a whole. The labels above the chart indicate the food items and their corresponding percentages.
</details>

![](images/fb91c706a5c88fe229a4f819d403bb4b817cec72a8ce4b67099f0fcd243e49eb.jpg)

<details>
<summary>text_image</summary>

BPG 25%
BPG 14%
BPG 71%
Saua plate 30 fl.
BPG 2%
BPG 1%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 6%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 5%
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
BPG 4
</details>

![](images/58ac55aec227c00b7fd1d2635c40b4ef0b794bdc06bff1c9a48a28dfbe560e48.jpg)

<details>
<summary>text_image</summary>

eg 81%g 79%4%g 68%34%
egg 79% 50%
clementine 75% 71%
dome legume 70%
dome (fruit) 58%
dome (bowl) 60%
dome (pump) 52%
pruffle (chocolate) 53%
dome (fruit) 73%
luffle
lemons 79%
</details>

![](images/c7ca21b2e64da4a504e272e0c0d5812b18be6ec55257b30253a8d4e7df383f56.jpg)

<details>
<summary>text_image</summary>

monitor(computer.equipment) computer_monitor
motor_vehicle 60-2%
driver's hardware 91%
driver's equipment 70%-92%
speaker_lstero_ecc
speaker_lstero.ecc
speaker_lstero.ecc
userTelephone 71%-92%
computer_keyboard 87%
computer_keyboard 90%-92%
chair 53%
dresser
</details>

![](images/0e91156c0bbda0ec332ff13aecab3cdd43f6316d2c93f379a79922bc31fe165e.jpg)

<details>
<summary>text_image</summary>

Cooking utensil 54%
pot 51%
coozerpan 56%
cauzerpan 56%
cucurran 56%
cucurran 56%
myrna
bakes 54%
pork 50%
chee
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
corn
cucuranspenders 37%
litter 37%
meat
over 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
meat 50%
</details>

![](images/3fa245ff9294a79cb8947433754c83ced6302b72b307a0e772d906bd13ee9464.jpg)

<details>
<summary>text_image</summary>

stepladder 58%
macht truck 50%
motorcycle 26% 2%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
bent
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
13%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
macht truck 50%
motorcycle 26% 2%
</details>

Figure 8: Qualitative results on LVIS images. Base and novel categories are shown in purple and green colors respectively. We use a score threshold of 0.5 and show the most confident class for each box.

Table 12: Ablation on the sampling distribution in POMP based on CLIP (ViT/B-16) backbone. 

<table><tr><td>Method</td><td>ImageNet-21K</td><td>Cross-dataset (10 Avg.)</td><td>Cross-domain (4 Avg.)</td></tr><tr><td>POMP (uniform distribution)</td><td>25.3</td><td>67.0</td><td>60.8</td></tr><tr><td>POMP (frequency distribution)</td><td>24.9 (-0.4)</td><td>66.2 (-0.8)</td><td>60.1 (-0.7)</td></tr><tr><td>POMP (similarity distribution)</td><td>23.6 (-1.7)</td><td>64.2 (-2.8)</td><td>59.2 (-1.6)</td></tr></table>

training samples belonging to the negative class i, the frequency distribution is defined as:

$$
Q _ {i} ^ {(f)} = \frac {M _ {i}}{\sum_ {j = 1} ^ {N} M _ {j}}. \tag {8}
$$

The second is the similarity distribution $Q^{(s)}$ , which aims to sample more hard negative classes. Hard negative classes are those that have a higher similarity between their features and the features of the input images, and are more likely to be confused with the positive class. Accordingly, in the similarity distribution, the likelihood of a negative class being sampled increases as the similarity between its feature and the image feature increases. To achieve this, we pre-encode features of all classes represented by a hand-crafted prompt (i.e., “a photo of a [CLASSNAME]”). The feature of class i is denoted as $w_{i}$ . The likelihood of sampling a negative class is determined by the similarity between the class feature $w_{i}$ and the image feature x:

$$
Q _ {i} ^ {(s)} (\mathbf {x}) = \frac {\exp (\mathbf {x} ^ {\top} \mathbf {w} _ {i} / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathbf {x} ^ {\top} \mathbf {w} _ {j} / \tau)}. \tag {9}
$$

Table 12 illustrates the performance of different proposal distributions. Compared to the uniform distribution, using the frequency distribution for sampling leads to degraded performance, particularly in cross-dataset and cross-domain settings, due to reduced sampling of rare categories. This highlights the importance of a large number of long-tail categories in the ImageNet-21K dataset for the generalization of the soft prompt. Additionally, the performance of the similarity distribution is also not as strong as that of the uniform distribution. The reason for this may be that as the soft prompt evolves, the features of hard negative classes change. However, the negative features used in (9) are obtained from the hard prompt, creating a fixed proposal distribution that is unable to adapt to these changes, potentially causing the soft prompt to converge to a local optimum. In contrast, POMP with the simple uniform distribution considers both common and rare classes, as well as easy and difficult classes, leading to the best performance for both the soft prompt and class features.

# E.2 Ablation on #shot and Prompt Length

We further conduct ablation on the number of pre-training instances per class (#shot) and the prompt length to analyze their influence on the generalization ability of POMP. The left panel in Figure 9 illustrates the results of #shot. The green curve represents the average accuracy of 10 datasets under the cross-dataset evaluation, while the purple curve represents the averaged accuracy of 4 datasets under the cross-domain evaluation. Overall, the performance of POMP improves as #shot increases. We find that POMP can achieve decent cross-dataset and cross-domain accuracy even with #shot=1. This is due to the huge number of classes in ImageNet-21K. Even if there are only one instance per class, the overall amount of data (21K instances for 21K classes) is enough for training a soft prompt with only 0.012 M learnable parameters.

The right panel in the figure shows the results of the prompt length. The soft prompt of length 16 achieves $65.0\%$ accuracy across datasets, which is lower than the soft prompt of length 4 with $67.0\%$ cross-dataset accuracy. It indicates that the prompt with too large lengths impairs its generalization, which consistent with the findings from previous work [66, 29].

# E.3 Ablation on Prompt Types for Semantic Segmentation

We perform an ablation study on prompt types for cross-dataset semantic segmentation to further demonstrate the superior generalization ability of our prompt on downstream tasks. Specifically, we

![](images/9c2b44a18a4da21b026d88530b35147cb5c51624f0fd8e5e398d63b8c46efd70.jpg)

![](images/fbede21bfa464ab84109900fae8bd44c606eb2c7005fc7166ef6db85a5e11be9.jpg)

![](images/b0087e477928e41d3a248924e5e1f1c3ef65fb5892e88cc688d1a9a1f8252f20.jpg)

<details>
<summary>line</summary>

| prompt length | Cross-domain (4 Avg.) |
| ------------- | --------------------- |
| 4             | 60.7                  |
| 8             | 60.2                  |
| 16            | 59.8                  |
</details>

Figure 9: Ablation study on #shot and prompt length. When varying #shot, the prompt length is 4, and when varying the prompt length, #shot is 16.

Table 13: Cross-dataset evaluation for semantic segmentation. All methods share the same visual backbone with ZSSeg, but use different prompts. 

<table><tr><td rowspan="2">Method</td><td colspan="4">Source Dataset: Standard COCO Stuff</td><td colspan="4">Target Dataset: ADE20K</td><td colspan="4">Target Dataset: PASCAL Context</td></tr><tr><td>mIoU</td><td>fwIoU</td><td>mACC</td><td>pACC</td><td>mIoU</td><td>fwIoU</td><td>mACC</td><td>pACC</td><td>mIoU</td><td>fwIoU</td><td>mACC</td><td>pACC</td></tr><tr><td>ZSSeg (single prompt)</td><td>40.5</td><td>47.8</td><td>53.5</td><td>61.7</td><td>17.8</td><td>44.0</td><td>31.0</td><td>52.9</td><td>51.8</td><td>64.6</td><td>69.9</td><td>74.3</td></tr><tr><td>ZSSeg (ImageNet prompts)</td><td>40.9</td><td>48.4</td><td>54.7</td><td>62.3</td><td>17.7</td><td>46.5</td><td>31.8</td><td>57.1</td><td>52.0</td><td>64.7</td><td>70.3</td><td>75.4</td></tr><tr><td>ZSSeg (ViLD prompts)</td><td>40.9</td><td>48.6</td><td>54.2</td><td>62.3</td><td>20.2</td><td>49.1</td><td>33.4</td><td>60.7</td><td>51.8</td><td>63.8</td><td>69.6</td><td>73.8</td></tr><tr><td>ZSSeg (POMP prompt, ours)</td><td>41.2</td><td>49.0</td><td>54.7</td><td>62.6</td><td>20.6</td><td>49.3</td><td>35.0</td><td>61.7</td><td>52.4</td><td>65.3</td><td>70.6</td><td>76.4</td></tr></table>

take ZSSeg as the backbone and evaluate the performance of four types of prompts, as described in Appendix B. As shown in Table 13, ZSSeg with our POMP prompt achieves the highest performance on the three datasets. It is noteworthy that, despite using 80 hard prompts for ImageNet prompts and 14 for ViLD prompts for prompt ensemble, their performance was consistently worse than our POMP with just one soft prompt, highlighting the effectiveness of our method.