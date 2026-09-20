# KOSMOS-2: Grounding Multimodal Large Language Models to the World

Zhiliang Peng, $^{*}$ Wenhui Wang, $^{*}$ Li Dong, $^{*}$ Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei $^{\dagger}$ Microsoft Research
https://aka.ms/GeneralAI

# Abstract

We introduce KOSMOS-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., “[text span] (bounding boxes)”, where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GRIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), KOSMOS-2 integrates the grounding capability into downstream applications. We evaluate KOSMOS-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence. Code and pretrained models are available at https://aka.ms/kosmos-2.

![](images/526d899874854d0ff0030266e15fb9c718d261c168179bb51063b65ac39321d7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Grounding"] --> B["a campfire(<loc₄><loc₁₀₀₇>)"]
    C["Kosmos-2: Multimodal Large Language Model"] --> D["[It"]<loc₄₄><loc₈₆₃>) sits next to Referring]
```
</details>

Figure 1: KOSMOS-2 is a multimodal large language model that has new capabilities of multimodal grounding and referring. KOSMOS-2 can understand multimodal input, follow instructions, perceive object descriptions (e.g., bounding boxes), and ground language to the visual world.

![](images/0ee4ef6426aa0ec9c4148a1e0ddc4bbb19d34ef9c9f4297a9b03909e6b5c77ea.jpg)

This animal is unique because it is a turtle. Turtles are known for their slow and steady pace, whereas rabbits are known to be agile and quick. It is not common for a turtle to be faster than a rabbit, especially in terms of speed.

a buoy attached to the boat.

The biggest difference between bottle-1 and bottle-2 is the label.

A snowman sits next to a campfire in the snow. He is wearing a hat, scarf, and mittens. There are several pots nearby, likely containing a hot meal for the snowman. The scene is serene and peaceful, with the snowman looking at the fire with a warm smile.

Figure 2: Selected examples generated from KOSMOS-2. The examples include (1) visual grounding, (2)-(3) grounded question answering, (4)-(6) multimodal referring via bounding boxes, and (7) grounded image captioning.

# 1 Introduction

Multimodal Large Language Models (MLLMs) $[HSD^{+}22, ADL^{+}22, HDW^{+}23, DXS^{+}23, Ope23]$ have successfully played a role as a general-purpose interface across a wide range of tasks, such as language, vision, and vision-language tasks. MLLMs can perceive general modalities, including texts, images, and audio, and generate responses using free-form texts under zero-shot and few-shot settings.

In this work, we unlock the grounding capability for multimodal large language models. Grounding capability can provide a more convenient and efficient human-AI interaction for vision-language tasks. It enables the user to point to the object or region in the image directly rather than input detailed text descriptions to refer to it, the model can understand that image region with its spatial locations. Grounding capability also enables the model to respond with visual answers (i.e., bounding boxes), which can support more vision-language tasks such as referring expression comprehension. Visual answers are more accurate and resolve the coreference ambiguity compared with text-only responses. In addition, grounding capability can link noun phrases and referring expressions in the generated free-form text response to the image regions, providing more accurate, informational, and comprehensive answers.

We introduce KOSMOS-2, a multimodal large language model with grounding capability built upon KOSMOS-1. KOSMOS-2 is a Transformer-based causal language model and is trained using the next-word prediction task. In order to unlock the grounding capability, we construct a web-scale dataset of grounded image-text pairs, and combine it with the multimodal corpora in KOSMOS-1 to train the model. The grounded image-text pairs are built upon a subset of image-text pairs from LAION-2B $[SBV^{+}22]$ and COYO-700M $[BPK^{+}22]$ . We construct a pipeline to extract and link the text spans (i.e., noun phrases and referring expressions) in the caption to the spatial locations (e.g., bounding boxes) of its corresponding objects or regions in the image. We convert the spatial coordinates of the bounding boxes to a sequence of location tokens, which is then appended after its respective text spans. The data format serves as a “hyperlink” to connect the objects or regions of the image to the caption.

Experimental results demonstrate that KOSMOS-2 not only achieves competitive performance on language and vision-language tasks evaluated in KOSMOS-1, but also achieves impressive performance on grounding tasks (phrase grounding and referring expression comprehension) and referring tasks (referring expression generation). As shown in Figure 2, integrating the grounding capability enables KOSMOS-2 to be used for more downstream tasks, such as grounded image captioning, and grounded visual question answering.

# 2 Construction of Web-Scale Grounded Image-Text Pairs (GRIT)

We introduce GRIT $^{2}$ , a large-scale dataset of Grounded Image-Text pairs, which is created based on image-text pairs from a subset of COYO-700M $[BPK^{+}22]$ and LAION-2B $[SBV^{+}22]$ . We construct a pipeline to extract and link text spans (i.e., noun phrases and referring expressions) in the caption to their corresponding image regions. The pipeline mainly consists of two steps: generating noun-chunk-bounding-box pairs and producing referring-expression-bounding-box pairs. We describe these steps in detail below:

Step-1: Generating noun-chunk-bounding-box pairs Given an image-text pair, we first extract noun chunks from the caption and associate them with image regions using a pretrained detector. As illustrated in Figure 3, we use spaCy [HMVLB20] to parse the caption ("a dog in a field of flowers") and extract all noun chunks ("a dog", "a field" and "flowers"). We eliminate certain abstract noun phrases that are challenging to recognize in the image, such as "time", "love", and "freedom", to reduce potential noise. Subsequently, we input the image and noun chunks extracted from the caption into a pretrained grounding model (e.g., GLIP $\left[\mathrm{LZZ}^{+}22\right]$ ) to obtain the associated bounding boxes. Non-maximum suppression algorithm is applied to remove bounding boxes that have a high overlap with others, even if they are not for the same noun chunk. We keep noun-chunk-bounding-box pairs with predicted confidence scores higher than 0.65. If no bounding boxes are retained, we discard the corresponding image-caption pair.

![](images/8d89123444b37c9907fd794619579180bfd60b8a2baa58de76ce7aa632f5f535.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["a dog in a field of flowers"] --> B["Identify noun chunks"]
    B --> C["a dog in a field of flowers"]
    B --> D["a field"]
    B --> E["flowers"]
    C --> F["Expand noun chunks"]
    D --> F
    E --> F
    F --> G["Drop substrings"]
    G --> H["Keep &quot;a dog in a field of flowers&quot;"]
    G --> I["Drop &quot;a field of flowers&quot;"]
    G --> J["Drop &quot;flowers&quot;"]
    B --> K["Detection & Post-process"]
    K --> L["a dog: [290,371,605,750"]
    L --> M["a field: [0,264,919,921"]]
    K --> N["Compose"]
    N --> O["a dog in a field of flowers: [290,371,605,750"]]
    O --> P["Step-2: Producing referring expression - bounding box pairs"]
```
</details>

Figure 3: The pipeline of constructing web-scale grounded image-text pairs.

<table><tr><td>Dataset</td><td>Images</td><td>Objects</td><td>Text Spans</td><td>Avg Expression Length</td></tr><tr><td>Flickr Entities [PWC+15]</td><td>31,783</td><td>275,775</td><td>513,644</td><td>-</td></tr><tr><td>RefCOCOg [MHT+15]</td><td>26,711</td><td>54,822</td><td>85,474</td><td>8.43</td></tr><tr><td>RefCOCO [YPY+16]</td><td>19,994</td><td>50,000</td><td>142,209</td><td>3.61</td></tr><tr><td>RefCOCO+ [YPY+16]</td><td>19,992</td><td>49,856</td><td>141,564</td><td>3.53</td></tr><tr><td>Visual Genome [KZG+16]</td><td>108,077</td><td>4,102,818</td><td>-</td><td>-</td></tr><tr><td>GRIT (Ours)</td><td>90,614,680</td><td>137,349,210</td><td>114,978,233</td><td>4.7</td></tr></table>

Table 1: Comparison GRIT with existing visual grounding datasets.

Step-2: Producing referring-expression-bounding-box pairs In order to endow the model with the ability to ground complex linguistic descriptions, we expand noun chunks to referring expressions. Specifically, we use spaCy to obtain dependency relations of the sentence. We then expand a noun chunk into a referring expression by recursively traversing its children in the dependency tree and concatenating children tokens with the noun chunk. We do not expand noun chunks with conjuncts. For noun chunks without children tokens, we keep them for the next process. In the example shown in Figure 3, the noun chunk ‘a dog’ can be expanded to “a dog in a field of flowers”, and the noun chunk ‘a field’ can be expanded to “a field of flowers”.

Furthermore, we only retain referring expressions or noun chunks that are not contained by others. As shown in Figure 3, we keep the referring expression “a dog in a field of flowers” and drop “a field of flowers” (as it is entailed by “a dog in a field of flowers”) and ‘flowers’. We assign the bounding box of the noun chunk (‘a dog’) to the corresponding generated referring expression (“a dog in a field of flowers”).

In the end, we obtain approximately 91M images, 115M text spans, and 137M associated bounding boxes. We compare GRIT with existing publicly accessible visual grounding datasets in Table 1. Data samples of GRIT are shown in the Appendix.

# 3 KOSMOS-2: A Grounded Multimodal Large Language Model

KOSMOS-2 is a grounded multimodal large language model, which integrates grounding and referring capabilities compared with KOSMOS-1. The model can accept image regions selected by the user using bounding boxes as input, provide visual answers (i.e., bounding boxes), and ground the text output to the visual world. KOSMOS-2 adopts the same model architecture and training objective as KOSMOS-1. We add grounded image-text pairs into the training data to endow the model with grounding and referring capabilities. For a text span (such as noun phrase and referring expression) and its corresponding bounding boxes in a grounded image-text pair, We discretize continuous coordinates of bounding boxes into a sequence of location tokens to encode with text tokens in a unified way. Then we link the location tokens and their corresponding text span via a “hyperlink” data

format. The model is trained to establish a mapping between image regions and their corresponding location tokens and connect the image regions with their associated text spans.

# 3.1 Grounded Input Representations

Given a text span and its associated bounding boxes in a grounded image-text pair, we first convert the continuous coordinates of bounding boxes into a sequence of discrete location tokens $[CSL^{+}21]$ . For an image with width W and height H, we evenly divide both the width and height into P segments each. $P \times P$ bins are obtained and each bin consists of $(W/P) \times (H/P)$ pixels. For each bin, we use a location token to represent the coordinates within that bin. We use the coordinates of the center pixel of each bin to determine bounding boxes on the image. In total, $P \times P$ location tokens are introduced, and these tokens are added to word vocabulary to enable unified modeling with texts.

The bounding box can be represented using its top-left point $(x_{1}, y_{1})$ and bottom-right point $(x_{2}, y_{2})$ . We discretize the top-left and bottom-right corner points to location tokens, respectively. We concatenate the top-left location token $<loc_{1}>$ , the bottom-right location token $<loc_{2}>$ , and special boundary tokens <box> and </box>, to represent a single bounding box: "<box><loc\_{1}><loc\_{2}></box>" . If the text span is associated with multiple bounding boxes, we use a special token <delim> to concatenate the location tokens of these bounding boxes: "<box><loc\_{1}^{i}><loc\_{2}^{i}><delim>...<loc\_{1}^{j}><loc\_{2}^{j}></box>" .

Then we arrange the text span and its associated location tokens in a format resembling a “hyperlink” in markdown. For the text span with a single bounding box, the resulted sequence is “<p> text span </p><box><loc1><loc2></box>”, where <p> and </p> are special tokens indicating the beginning and end of the text span. The data format tells the model that image regions within the bounding box are associated with the text span.

For the example shown in Figure 1, the input representation is:

```html
<s> <image> Image Embedding </image> <grounding> <p> It </p><box><loc44><loc863></box> seats next to <p> a campfire </p><box><loc4><loc1007></box></s> 
```

where <s> and </s> indicate start- and end-of-sequence, and <image> and </image> represent the beginning and end of encoded image embeddings. <grounding> is a special token to tell the model ground the text output to the visual world. We map input text tokens and location tokens to embeddings via a lookup table. Following KOSMOS-1, a vision encoder and a resampler module are used to obtain image embeddings for input images.

For language-only data, cross-modal paired data (i.e., image-text pairs), and interleaved multimodal data, we use the same input representations as of KOSMOS-1.

# 3.2 Grounded Multimodal Large Language Models

Based on KOSMOS-1, KOSMOS-2 enhances multimodal large language models by incorporating grounding and referring capabilities. KOSMOS-2 also uses a Transformer-based causal language model as the backbone and is trained with the next-token prediction task.

In addition to multimodal corpora used in KOSMOS-1 (including text corpora, image-caption pairs, and interleaved image-text data), we add grounded image-text pairs into training. The training loss only considers discrete tokens, such as text tokens and location tokens. The model can learn to locate and understand image regions by their location tokens and the whole image, associate text spans to image regions, and output bounding boxes of the image region using location tokens.

KOSMOS-2 shows new capabilities of grounding and referring. The referring capability enables us to point out image regions with bounding boxes. KOSMOS-2 can understand the image regions users refer to by the coordinates of bounding boxes. The referring capability provides a new interaction method. Different from previous MLLMs $[ADL^{+}22, HSD^{+}22, HDW^{+}23]$ , which can only provide text output, KOSMOS-2 can provide visual answers (i.e., bounding boxes) and ground text output to the image. The grounding capability enables the model to provide more accurate, informative, and comprehensive responses. In addition to vision, language, and vision-language tasks evaluated in

KOSMOS-1, the model can be used for more downstream tasks, such as grounded image-captioning, grounded VQA, referring expression comprehension and generation.

# 3.3 Model Training

Training Setup We train the model on newly added grounded image-text pairs, monomodal text corpora, image-caption pairs, and interleaved image-text data. Our training process involves a batch size of 419K tokens, consisting of 185K tokens from text corpora, 215K tokens from original and grounded image-caption pairs, and 19K tokens from interleaved data. We train KOSMOS-2 for 60k steps, equivalent to around 25 billion tokens. The AdamW optimizer is employed with $\beta = (0.9, 0.98)$ . We set the weight decay to 0.01 and the dropout rate to 0.1. The learning rate increases to 2e-4 during the first 375 warm-up steps and linearly decays to zero. We train the model on 256 V100 GPUs and the training takes approximately one day to complete. In order to tell the model when to ground text output to the visual world, we prepend the ‘<grounding>’ token to the grounded caption during training.

Following KOSMOS-1, the vision encoder has 24 layers with 1,024 hidden size and 4,096 FFN intermediate size. The multimodal large language model component is a 24-layer MAGNETO Transformer $[WMH^{+}22, MWH^{+}22]$ with 2,048 hidden dimensions, 32 attention heads, and 8,192 FFN intermediate size. The total number of trainable parameters amounts to approximately 1.6B. The image resolution is set to $224 \times 224$ and the patch size is $14 \times 14$ . We divide the width and height of the image into 32 bins, with each bin consisting of $7 \times 7$ pixels. A total of $32 \times 32$ location tokens are added to the vocabulary. KOSMOS-2 uses the weights of KOSMOS-1 for initialization, the newly added word embeddings of location tokens are initialized randomly. We update all the parameters during training and instruction tuning.

Instruction Tuning After the model is trained, we perform instruct tuning to better align KOSMOS-2 with human instructions. we combine vision-language instruction dataset (i.e., LLaVA-Instruct [LLWL23]) and language-only instruction datasets (i.e., Unnatural Instructions [HSLS22] and FLANv2 $\left[\mathrm{LHV}^{+}23\right]$ ) with the training data to tune the model. In addition, we construct grounded instruction data by utilizing the pairs of bounding boxes and expressions (i.e., noun phrases, and referring expressions) in GRIT. Given an expression-bounding-box pair, we use "<p> expression </p>" as the input instruction, and prompt the model to generate the corresponding location tokens of the bounding boxes. We also use the prompt like "<p> It </p><box><loc1><loc2></box> is" to ask the model to generate expressions according to its bounding boxes. Table B in Appendix presents more templates.

# 4 Evaluation

We first evaluate KOSMOS-2 on multimodal grounding and multimodal referring tasks to assess the new capabilities, and then test the model on language and perception-language tasks evaluated in KOSMOS-1.

- Multimodal grounding   
- Phrase grounding   
– Referring expression comprehension   
- Multimodal referring   
– Referring expression generation   
• Perception-language tasks   
- Image captioning   
- Visual question answering   
- Language tasks   
- Language understanding   
- Language generation

![](images/a30353f7af6ce2fa4489a6f7dbe019ac7fb7d8d12811aff70a6f4ebeebc697bc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["<box> <loc165> <loc360> </box>"] --> B["Grounded MLLM"]
    C["A man in a blue hard hat and <p> orange safety vest </p>"] --> B
    B --> D["Street photo of worker"]
```
</details>

(1) Phrase grounding

![](images/a25f4e8f73ba5882612ded521046dc57d5f6112d39dab442c06dfc2f68eb7dec.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["<box> <loc₆₈> <loc₄₂₅> </box>"] --> B["Grounded MLLM"]
    C["<p> A man in a blue hard hat and orange safety vest </p>"] --> B
    B --> D["Street photo of worker"]
```
</details>

(2) Referring expression comprehension   
Figure 4: Input format of evaluation on (1) phrase grounding and (2) referring expression comprehension.

# 4.1 Multimodal Grounding

In order to evaluate the ability of multimodal grounding, we test KOSMOS-2 on widely used phrase grounding and referring expression comprehension tasks in a generation manner. Phrase grounding task requires the model to predict a set of bounding boxes based on one or more given phrases that maybe interrelated within a single caption. Referring expression comprehension task encourages the model to locate the object described in a text referring expression within a given image.

By testing KOSMOS-2 on these two tasks, we can assess how well the model performs in grounding text descriptions to the visual world, which is crucial for developing advanced AI systems capable of handling complex multimodal tasks.

For both phrase grounding and referring expression comprehension tasks, KOSMOS-2 is required to generate location tokens which are then converted to bounding boxes for evaluation. The input format is “<s><image> Image Embedding </image><grounding>...”, where “<grounding>” is used to prompt the model to generate locations tokens.

# 4.1.1 Phrase Grounding

We evaluate phrase grounding task on Flickr30k Entities $[PWC^{+}15]$ val and test splits. In order to reduce ambiguity, we do not prompt the model with individual phrases; instead, we use the current phrase along with the preceding words as input where preceding words serve as context: “... <p> {phrase} </p>”. For the example shown in Figure 4(1), the model needs to predict the locations of phrases “A man”, “a blue hard hat”, “orange safety vest” and “an intersection” in the caption “A man in a blue hard hat and orange safety vest stands in an intersection.”. To generate the location tokens for the phrase “A man” that is the beginning of the caption, the prompt is “<p>A man</p>”. For the phrase “orange safety vest”, the prompt is “A man in a blue hard hat and <p>orange safety vest</p>”. When multiple men are in the image, the context “A man in a blue hard hat and” explicitly helps the model locate the object to reduce ambiguity.

We obtain the location tokens in “<box>...</box>” from the model response and then covert it into bounding boxes. The generated bounding box is correct if its intersection over union (IoU) with the ground-truth bounding box is greater than 0.5. If KOSMOS-2 generates a location sequence that can not be converted correctly (e.g., “<box><loc1></box>”), we treat it as a negative sample. We use ANY-BOX protocol in MDETR [KSL+21]. We report the R@1, R@5, and R@10 metrics, where R@1/5/10 means calculating the recall using the top 1/5/10 generated bounding boxes. If there are fewer than 5 or 10 bounding boxes generated by KOSMOS-2, we use all available bounding boxes for the calculation.

Results Table 2 presents results on Flickr30k Entities $[PWC^{+}15]$ val and test splits. KOSMOS-2 achieves impressive zero-shot performance and outperforms GRILL $[JMC^{+}23]$ , which relies on an attached detector, by a large margin. Moreover, our model outperforms traditional finetuned

<table><tr><td rowspan="2">Model</td><td rowspan="2">Zero-shot</td><td colspan="3">Val Split</td><td colspan="3">Test Split</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>VisualBert [LYY+19]</td><td>✘</td><td>70.4</td><td>84.5</td><td>86.3</td><td>71.3</td><td>85.0</td><td>86.5</td></tr><tr><td>MDETR [KSL+21]</td><td>✘</td><td>83.6</td><td>93.4</td><td>95.1</td><td>84.3</td><td>93.9</td><td>95.8</td></tr><tr><td>GLIP [LZZ+22]</td><td>✘</td><td>86.7</td><td>96.4</td><td>97.9</td><td>87.1</td><td>96.9</td><td>98.1</td></tr><tr><td>FIBER [DKG+22]</td><td>✘</td><td>87.1</td><td>96.1</td><td>97.4</td><td>87.4</td><td>96.4</td><td>97.6</td></tr><tr><td>GRILL [JMC+23]</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>18.9</td><td>53.4</td><td>70.3</td></tr><tr><td>KOSMOS-2</td><td>✓</td><td>77.8</td><td>79.2</td><td>79.3</td><td>78.7</td><td>80.1</td><td>80.1</td></tr></table>

Table 2: Phrase grounding results on Flickr30k Entities. We report the R@1, R@5, and R@10 metrics, where R@1/5/10 means calculating the recall using the top 1/5/10 generated bounding boxes.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Zero-shot</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td></tr><tr><td>UNITER [CLY+19]</td><td>✘</td><td>81.41</td><td>87.04</td><td>74.17</td><td>75.90</td><td>81.45</td><td>66.70</td><td>74.86</td><td>75.77</td></tr><tr><td>MDETR [KSL+21]</td><td>✘</td><td>87.51</td><td>90.40</td><td>82.67</td><td>81.13</td><td>85.52</td><td>72.96</td><td>83.35</td><td>83.31</td></tr><tr><td>OFA [WYM+22]</td><td>✘</td><td>90.05</td><td>92.93</td><td>85.26</td><td>84.49</td><td>90.10</td><td>77.77</td><td>84.54</td><td>85.20</td></tr><tr><td>FIBER [DKG+22]</td><td>✘</td><td>90.68</td><td>92.59</td><td>87.26</td><td>85.74</td><td>90.13</td><td>79.38</td><td>87.11</td><td>87.32</td></tr><tr><td>VisionLLM [WCC+23]</td><td>✘</td><td>86.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GRILL [JMC+23]</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>47.5</td></tr><tr><td>KOSMOS-2</td><td>✓</td><td>52.32</td><td>57.42</td><td>47.26</td><td>45.48</td><td>50.73</td><td>42.24</td><td>60.57</td><td>61.65</td></tr></table>

Table 3: Referring expression comprehension results on RefCOCO, RefCOCO+ and RefCOCOg. We report the accuracy metric for all methods.

VisualBert $[LYY^{+}19]$ model by 7.4% R@1 on both val and test splits. In contrast to other models, KOSMOS-2 does not involve prior designs (e.g., object queries or proposals), leading to similar results among R@1, R@5, and R@10. These results demonstrate that KOSMOS-2 can generate high-quality locations without the need for post-processing redundant locations. This capability highlights the effectiveness of our model in handling phrase grounding tasks.

# 4.1.2 Referring Expression Comprehension

We assess the referring expression comprehension task using three well-established datasets: RefCOCO $[YPY^{+}16]$ , RefCOCO+ $[YPY^{+}16]$ and RefCOCOg $[MHT^{+}15]$ . Both RefCOCO and RefCOCO+ were generated through a two-player game, with RefCOCO+ specifically designed to exclude spatial relations, such as “on the left”. RefCOCOg incorporates spatial relations and features longer expressions on average. Different from phrase grounding on Flickr30k entities, we measure this task by using referring expression as the input: “<p> referring expression </p>”. For the example shown in Figure 4(2), the input sequence is “<p>A man in a blue hard hat and orange safety vest</p>”. Similarly, the predicted bounding box is considered correct only if its IOU with the ground-truth bounding box is greater than 0.5. The failed decoded sequence is also treated as a negative sample. We use the first generated bounding box for the query expression to measure the accuracy.

Results Table 3 reports referring comprehension results on RefCOCO $[YPY^{+}16]$ , Ref-COCO+ $[YPY^{+}16]$ and RefCOCOg $[MHT^{+}15]$ . Kosmos-2 also obtains promising zero-shot performance on the comprehension task, significantly outperforming previous zero-shot models on RefCOCOg benchmark. However, compared to previous finetuned works, Kosmos-2 achieves slightly lower performance on RefCOCO and RefCOCO+ than on RefCOCOg. This discrepancy can be attributed to the data distribution present in RefCOCO and RefCOCO+, where they tend to use a shorter referring expression (e.g., “left bottom”) during the two-player game. Hence, one of our future goals is to enhance MLLMs’ ability to accurately understand more types of human expressions.

![](images/d3a94743920b7fc2bed1b69a697e301ef0ce15fa15f7cd4249db1be410b93a62.jpg)  
Figure 5: The input format of referring expression generation evaluation under (1) zero-shot and (2) few-shot settings. The bounding boxes shown in the image are for visualization purposes.

# 4.2 Multimodal Referring

In addition to multimodal grounding tasks, we evaluate the model's ability to understand image regions or objects users refer to via inputting bounding boxes. Compared with previous multimodal LLMs that can only refer image regions or objects to the model via detailed text descriptions, directly referring to image regions using its bounding boxes is more effective and reduces ambiguity.

We evaluate the model on the referring expression generation task, which aims to generate unambiguous text descriptions of specific objects or regions within the bounding box. We employ the widely used RefCOCOg dataset $[MHT^{+}15]$ to evaluate the model's performance under both zero-shot and few-shot settings, showcasing its adaptability in different scenarios.

# 4.2.1 Evaluation Setup

The model is tasked with generating an associated text description for an object or region given its location tokens of the bounding boxes (e.g., “<box><loc1><loc2></box>”). Benefiting from the unified input format, we use “<p>It</p><box><loc1><loc2></box>is” as prompt to encourage the model to predict its text description. Figure 5 (1) and (2) demonstrate the input format for zero-shot and few-shot referring expression generation, respectively. Following previous works, we report results using METEOR and CIDEr metrics. The image resolution is $224 \times 224$ . Greedy search is used for decoding.

# 4.2.2 Results

Table 4 presents the zero-shot and few-shot results of referring expression generation on RefCOCOg. We compare KOSMOS-2 with a finetuned listener-speaker model, which introduces an added reward-based module (SLR). Our model obtains impressive zero-shot performance on referring expression generation, and even outperforms finetuned SLR by 1.1 CIDEr scores. Moreover, when prompted with fewshot demonstrations, KOSMOS-2 shows further improvements, highlighting its in-context learning ability.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="2">RefCOCOg</td></tr><tr><td>Meteor</td><td>CIDEr</td></tr><tr><td>SLR[YTBB17]</td><td>Finetuning</td><td>15.4</td><td>59.2</td></tr><tr><td>SLR+Rerank[YTBB17]</td><td>Finetuning</td><td>15.9</td><td>66.2</td></tr><tr><td rowspan="3">KOSMOS-2</td><td>Zero-shot</td><td>12.2</td><td>60.3</td></tr><tr><td>Few-shot (k=2)</td><td>13.8</td><td>62.2</td></tr><tr><td>Few-shot (k=4)</td><td>14.1</td><td>62.3</td></tr></table>

Table 4: Results of referring expression generation on RefCOCOg.

# 4.3 Perception-Language Tasks

In addition to multimodal grounding and referring tasks, we also evaluate KOSMOS-2 on the vision-language tasks following KOSMOS-1. In particular, we perform zero-shot evaluations on two popular tasks, including image captioning and visual question answering. Image captioning requires the model to generate a text description of the given image, whereas visual question answering seeks to answer a natural language question based on an image. In order to have a fair comparison with KOSMOS-1, we report results without instruction tuning.

# 4.3.1 Evaluation Setup

For image captioning, we evaluate the model on the widely used Flickr30k Karpathy split test set. We employ beam search for caption generation, with a beam size of 5. We report results using CIDEr [VLZP15] metrics evaluated by COCOEvalCap $^{3}$ . We use the prompt “An image of” to generate the image description.

For visual question-answering, we evaluate zero-shot performance on the test-dev set of VQAv2. Greedy search is used for decoding. We report VQA scores obtained from VQAv2 evaluation server $^{4}$ . “Question: {question} Answer: {answer}” is used as the prompt for the dataset. The image resolution is $224 \times 224$ for both two tasks.

# 4.3.2 Results

We present the zero-shot performance on Flickr30k and VQAv2 in Table 5. KOSMOS-2 exhibits comparable overall performance to the KOSMOS-1, showing a slight improvement on Flickr30k while experiencing a marginal decrease on VQA. While KOSMOS-2 introduces new capabilities of grounding and referring, the model still achieves competitive performance on perception-language tasks.

<table><tr><td rowspan="2">Model</td><td>Flickr30k</td><td>VQAv2</td></tr><tr><td>CIDEr</td><td>VQA acc.</td></tr><tr><td>FewVLM [JCS+22]</td><td>31.0</td><td>-</td></tr><tr><td>METALM [HSD+22]</td><td>43.4</td><td>41.1</td></tr><tr><td>Flamingo-3B [ADL+22]</td><td>60.6</td><td>49.2</td></tr><tr><td>Flamingo-9B [ADL+22]</td><td>61.5</td><td>51.8</td></tr><tr><td>KOSMOS-1</td><td>65.2</td><td>46.7</td></tr><tr><td>KOSMOS-2</td><td>66.7</td><td>45.6</td></tr></table>

Table 5: Zero-shot image captioning results on Flickr30k test set and zero-shot visual question answering results on VQAv2 test-dev set. We report results of KOSMOS-2 and KOSMOS-1 without instruction tuning.

# 4.4 Language Tasks

We evaluate KOSMOS-2 on eight language tasks, such as cloze and completion tasks (StoryCloze, HellaSwag), Winograd-style tasks (Winograd, Winogrande), commonsense reasoning (PIQA), and three SuperGLUE benchmark $[WPN^{+}19]$ datasets (BoolQ, CB, and COPA). We report the zero-shot results in Table 6. Compared with KOSMOS-1, KOSMOS-2 achieves similar performance on StoryCloze, HellaSwag, Winograd, Winogrande, and PIQA, experiences a decrease in performance on CB, but shows improvement on BoolQ and COPA. In summary, KOSMOS-2 demonstrates the acquisition of new capabilities while experiencing comparable performance on language tasks. This illustrates the potential of the model in balancing and expanding its skills across different domains.

<table><tr><td>Model</td><td>Story Cloze</td><td>Hella Swag</td><td>Winograd</td><td>Winogrande</td><td>PIQA</td><td>BoolQ</td><td>CB</td><td>COPA</td></tr><tr><td>LLM</td><td>72.9</td><td>50.4</td><td>71.6</td><td>56.7</td><td>73.2</td><td>56.4</td><td>39.3</td><td>68.0</td></tr><tr><td>KOSMOS-1</td><td>72.1</td><td>50.0</td><td>69.8</td><td>54.8</td><td>72.9</td><td>56.4</td><td>44.6</td><td>63.0</td></tr><tr><td>KOSMOS-2</td><td>72.0</td><td>49.4</td><td>69.1</td><td>55.6</td><td>72.9</td><td>62.0</td><td>30.4</td><td>67.0</td></tr></table>

Table 6: Zero-shot performance comparisons of language tasks between KOSMOS-2, KOSMOS-1 and LLM. LLM uses the same text data and training setup to reimplement a language model as KOSMOS-1. We report results of KOSMOS-2 and KOSMOS-1 without instruction tuning. Results of KOSMOS-1 and the LLM baseline are from $[HDW^{+}23]$ .

# 5 Conclusion

We present KOSMOS-2, a multimodal large language modal, that can ground to the visual world. Specifically, we pre-train KOSMOS-2 by augmenting the multimodal corpora used in KOSMOS-1 with GRIT, a large-scale dataset of Grounded Image-Text pairs, which is created by extracting and associating noun phrases and referring expressions in the caption to the objects or regions in the scene. KOSMOS-2 enables new capabilities of perceiving image regions and grounding text output to the visual world, which makes grounding as a foundation capability of MLLMs in many downstream applications. Experimental results demonstrate that KOSMOS-2 achieves impressive results on language and vision-language tasks evaluated in KOSMOS-1, grounding tasks including phrase grounding and referring expression comprehension, and referring tasks such as referring expression generation.

# Acknowledgement

Some examples (such as Figure 1) are taken from the WHOOPS corpus [BGBH $^{+}$ 23].

# Ethics Statement

The model presented in this paper is intended for academic and research purposes. The utilization of the model to create unsuitable material is strictly forbidden and not endorsed by this work. The accountability for any improper or unacceptable application of the model rests exclusively with the individuals who generated such content. We also put Microsoft AI Principles $^{5}$ into practice when developing the models.

# References

[ADL $^{+}$ 22] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022.   
[AHR $^{+}$ 22] Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, and Luke Zettlemoyer. CM3: A causal masked multimodal model of the Internet. ArXiv, abs/2201.07520, 2022.   
[BGBH $^{+}$ 23] Nitzan Bitton-Guetta, Yonatan Bitton, Jack Hessel, Ludwig Schmidt, Yuval Elovici, Gabriel Stanovsky, and Roy Schwartz. Breaking common sense: WHOOPS! a vision-and-language benchmark of synthetic and compositional images. ArXiv, abs/2303.07274, 2023.

[BPK $^{+}$ 22] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset, 2022.   
[CLY $^{+}$ 19] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In European Conference on Computer Vision, 2019.   
[CSL $^{+}$ 21] Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, and Geo rey E. Hinton. Pix2seq: A language modeling framework for object detection. ArXiv, abs/2109.10852, 2021.   
[DKG $^{+}$ 22] Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, Jianfeng Gao, and Lijuan Wang. Coarse-to-fine vision-language pre-training with fusion in the backbone. ArXiv, abs/2206.07643, 2022.   
[DXS $^{+}$ 23] Danny Driess, F. Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Ho Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Peter R. Florence. Palm-e: An embodied multimodal language model. ArXiv, abs/2303.03378, 2023.   
[HDW $^{+}$ 23] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, and Furu Wei. Language is not all you need: Aligning perception with language models. ArXiv, abs/2302.14045, 2023.   
[HMVLB20] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spaCy: Industrial-strength Natural Language Processing in Python. 2020.   
[HSD $^{+}$ 22] Yaru Hao, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shuming Ma, and Furu Wei. Language models are general-purpose interfaces. ArXiv, abs/2206.06336, 2022.   
[HSLS22] Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural instructions: Tuning language models with (almost) no human labor, 2022.   
[JCS $^{+}$ 22] Woojeong Jin, Yu Cheng, Yelong Shen, Weizhu Chen, and Xiang Ren. A good prompt is worth millions of parameters: Low-resource prompt-based learning for vision-language models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2763–2775, Dublin, Ireland, May 2022. Association for Computational Linguistics.   
[JMC $^{+}$ 23] Woojeong Jin, Subhabrata Mukherjee, Yu Cheng, Yelong Shen, Weizhu Chen, Ahmed Hassan Awadallah, Damien Jose, and Xiang Ren. Grill: Grounded vision-language pre-training via aligning text and image regions. ArXiv, abs/2305.14676, 2023.   
[KSL $^{+}$ 21] Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, and Nicolas Carion. Mdetr - modulated detection for end-to-end multi-modal understanding. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 1760–1770, 2021.   
[KZG $^{+}$ 16] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision, 123:32–73, 2016.   
[LHV $^{+}$ 23] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.

[LLSH23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ArXiv, abs/2301.12597, 2023.   
[LLWL23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.   
[LYY $^{+}$ 19] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. ArXiv, abs/1908.03557, 2019.   
[LZZ $^{+}$ 22] Liunian Harold Li\*, Pengchuan Zhang\*, Haotian Zhang\*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In CVPR, 2022.   
[MHT $^{+}$ 15] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana-Maria Camburu, Alan Loddon Yuille, and Kevin P. Murphy. Generation and comprehension of unambiguous object descriptions. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 11–20, 2015.   
[MWH $^{+}$ 22] Shuming Ma, Hongyu Wang, Shaohan Huang, Wenhui Wang, Zewen Chi, Li Dong, Alon Benhaim, Barun Patra, Vishrav Chaudhary, Xia Song, and Furu Wei. TorchScale: Transformers at scale. CoRR, abs/2211.13184, 2022.   
[Ope23] OpenAI. Gpt-4 technical report. 2023.   
[PWC $^{+}$ 15] Bryan A. Plummer, Liwei Wang, Christopher M. Cervantes, Juan C. Caicedo, J. Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. International Journal of Computer Vision, 123:74–93, 2015.   
[SBV $^{+}$ 22] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.   
[VLZP15] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, pages 4566-4575, 2015.   
[WCC $^{+}$ 23] Wen Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Y. Qiao, and Jifeng Dai. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. ArXiv, abs/2305.11175, 2023.   
[WMH $^{+}$ 22] Hongyu Wang, Shuming Ma, Shaohan Huang, Li Dong, Wenhui Wang, Zhiliang Peng, Yu Wu, Payal Bajaj, Saksham Singhal, Alon Benhaim, Barun Patra, Zhun Liu, Vishrav Chaudhary, Xia Song, and Furu Wei. Foundation transformers. CoRR, abs/2210.06423, 2022.   
[WPN $^{+}$ 19] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. arXiv preprint arXiv:1905.00537, 2019.   
[WYM $^{+}$ 22] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, 2022.   
[YPY $^{+}$ 16] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions. ArXiv, abs/1608.00272, 2016.   
[YTBB17] Licheng Yu, Hao Tan, Mohit Bansal, and Tamara L. Berg. A joint speaker-listener-reinforcer model for referring expressions. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 3521–3529. IEEE Computer Society, 2017.

# A Hyperparameters

The training hyperparameters of KOSMOS-2 are listed in Table 7.

<table><tr><td colspan="2">Hyperparameters</td></tr><tr><td>Image embedding number</td><td>64</td></tr><tr><td>Location tokens</td><td>1,024</td></tr><tr><td>Training steps</td><td>60,000</td></tr><tr><td>Warmup steps</td><td>375</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>2e-4</td></tr><tr><td>Learning rate decay</td><td>Linear</td></tr><tr><td>Adam β</td><td>(0.9, 0.98)</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Batch size of text corpora</td><td>93</td></tr><tr><td>Batch size of original image-caption pairs</td><td>1,117</td></tr><tr><td>Batch size of grounded image-text pairs</td><td>1,117</td></tr><tr><td>Batch size of interleaved data</td><td>47</td></tr></table>

Table 7: Training hyperparameters of Kosmos-2

The instruction tuning hyperparameters are listed in Table 8.

<table><tr><td colspan="2">Hyperparameters</td></tr><tr><td>Training steps</td><td>10,000</td></tr><tr><td>Warmup steps</td><td>375</td></tr><tr><td>Learning rate</td><td>1e-5</td></tr><tr><td>Batch size of language instruction data</td><td>117</td></tr><tr><td>Batch size of vision-language instruction data</td><td>351</td></tr><tr><td>Batch size of grounded image-text pairs &amp; grounded instruction data</td><td>1404</td></tr><tr><td>Batch size of text corpora</td><td>30</td></tr><tr><td>Batch size of interleaved data</td><td>15</td></tr></table>

Table 8: Instruction tuning hyperparameters of KOSMOS-2

# B Templates for Grounded Instruction Data

Table 9 presents the instruction templates of expression generation based on its associated bounding boxes during instruction tuning.

- "What is <p> it </p><box><loc1><loc2></box>? It is {expression}."   
- "What is <p> this </p><box><loc1><loc2></box>? This is {expression}."   
- "Describe <p> this object </p><box><loc1><loc2></box>. This object is {expression}.   
- "<p> It </p><box><loc1><loc2></box> is {expression}."   
- "<p> This </p><box><loc1><loc2></box> is {expression}."   
- "<p> The object </p><box><loc1><loc2></box> is {expression}."

Table 9: Instruction templates used for expression generation.

# C Examples of GRIT

We present some examples of the GRIT corpus in Figures 6–9. The grounded image-text pairs span over various domains and contain different numbers of objects.

![](images/5e95d31d874fb55cb70090ddd5609ded63bcc17f3236afd4c221cab0b3300982.jpg)

<details>
<summary>text_image</summary>

a small white bowl filled with creamy light green
an aluminium tray.
roasted vegetable salad
</details>

Figure 6: Example from GRIT. Caption: “A serving of kale and roasted vegetable salad on an aluminium tray served with a small white bowl filed with creamy light green avocado Caesar dressing”.

![](images/405e7788134c46e3c28c6ff0cc6bac5b0a1b84a4701bfd8d52ec7d58293bb96d.jpg)

<details>
<summary>text_image</summary>

A Keto Chicken Nugget
a bowl of keto honey mustard
</details>

Figure 7: Example from GRIT. Caption: “A Keto Chicken Nugget being dipped into a bowl of keto honey mustard.”

![](images/bd4594801dd24d1efd833b98532dea187b96cf2efc46852fad3c987299578a62.jpg)

<details>
<summary>text_image</summary>

The Sydney skyline
Solar cells on a red roof
</details>

Figure 8: Example from GRIT. Caption: “Solar cells on a red roof are in the foreground. The Sydney skyline is in the background.”

![](images/9e0fe98adf14754c8315de83aceca4525d1af2ae7e6860f77202a02feeefdd50.jpg)

<details>
<summary>text_image</summary>

The background
Woman standing outdoors in a city landscape and wearing a hijab
a hijab
Her arm
a young girl who is hugging her side
</details>

Figure 9: Example from GRIT. Caption: “Woman standing outdoors in a city landscape and wearing a hijab. Her arm is around a young girl who is hugging her side. The background is blurred.”

# D More Examples of KOSMOS-2

As illustrated in Figure 10, multimodal referring capability used for visual dialogue can unlock potential in human-AI interaction. In Figure 11, our approach demonstrates its in-context learning ability for fine-grained object detection using both text and image descriptions. Figure 12 and Figure 13 showcase more selected examples, including grounded visual question answering, grounded image captioning, and multimodal referring.

![](images/80047b0bff47e31e5b6ea012e22b01c04ade1639f46629e91e7cccf5e6c2f4d9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["What is the name of this building?"] --> B["Sydney Opera House."]
    B --> C["When it was completed?"]
    C --> D["In 1973."]
    D --> E["What is the name of this?"]
    E --> F["Sydney Harbour Bridge."]
    
    G["What he is doing?"] --> H["The man is riding a horse."]
    H --> I["How about this man?"]
    I --> J["He is holding a flag."]
    J --> K["Which person is this one facing to?"]
    K --> L["The person on the right side of the fence."]
```
</details>

Figure 10: Examples of visual dialogue generated from KOSMOS-2.

![](images/d200ca026104e3fa0e79b6fde8ea5f3640058b57b345dda296c211e3764196fb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["<p>a downy woodpecker</p>"] --> B["<box><loc97><loc942></box>"]
    B --> C["With text description"]
    C --> D["<p>a downy woodpecker</p> <box><loc115><loc958></box>"]
    D --> E["With image description"]
    E --> F["This is a downy woodpecker"]
    F --> G["<p>a downy woodpecker</p> <box><loc115><loc958></box>"]
    G --> H["With text description"]
```
</details>

Figure 11: Examples of object detection with multimodal descriptions from KOSMOS-2.

Input Prompt   
![](images/d1e377286080271ff36ab4a29ef1b7eefb64386e447ff5ecd01c36d0211c7b42.jpg)

<details>
<summary>natural_image</summary>

Cricketing match in progress on a grass field, players in white uniforms and one in white shirt observing (no visible text or signage)
</details>

Question: Where is the coach?
Answer: <p>The coach</p>

![](images/66634e5a9031193b2bd9dcba25bd9b59fd40551347b20874d1cd3a546720fea0.jpg)

<details>
<summary>text_image</summary>

de suis CHARIE!
###
</details>

Question: What does it say?
Answer:

![](images/0ce0cdf81bd6e538be9d7663643081693e0154ac921a6e225cdde1eb3a44b413.jpg)

<details>
<summary>natural_image</summary>

Nighttime scene of a snowman sitting on fire with a campfire and scattered debris (no text or symbols visible)
</details>

Question: What makes this image weird? Answer:   
The image is weird because the snowman is sitting next to a campfire in the snow.

Completion   
The coach is standing in the middle of the field.   
"Je suis Charlie!" is written in French on a chalkboard.   
![](images/72dd65a6401a8e340332c88da0fb2b77abad66a70511fca01b139e6cdb281966.jpg)

<details>
<summary>natural_image</summary>

Group of cricket players in action on a grass field, one player standing near the ground (no visible text or symbols)
</details>

![](images/eae3b1ce1c3b0855f2966d856ba31d2098bdbc7cf7be67c66ce6877171c99eb9.jpg)

<details>
<summary>text_image</summary>

de suis CHARLIE!
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
</details>

![](images/46b300c524b26d123d26e24a8851d47d8dfce6a3beeb570ec3f32da700abcbcb.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a campfire, a snowman in a hat, and scattered rocks (no text or symbols)
</details>

(1) Grounded question answering   
Input Prompt   
![](images/2d84fc3cff5bac24855e9108e966236f322b9dd39e9b60b014f3ce7a12ac9881.jpg)

<details>
<summary>natural_image</summary>

White microwave oven with a yellow bowl containing a creamy substance inside (no visible text or symbols)
</details>

Question: Where is it? Answer:

![](images/66793d4e7d29e36d485300e85c09d428db695a1cf7614964f669d07e57272a3d.jpg)

<details>
<summary>text_image</summary>

Street photo with visible traffic lights and a red car detection bounding box, including a green bounding box highlighting the vehicle's focus.
</details>

You are driving this car. To pick up that man, you need to

![](images/3ba2b65c0274a5cf92353ecc17b2b417459c38efba5c06c13dbbfc19d76bf35b.jpg)

<details>
<summary>natural_image</summary>

Night scene with a campfire, a snowman figure sitting on fire, and scattered debris (no text or symbols visible)
</details>

Question: What will happen to it?
Answer:

Completion   
In a microwave oven.   
![](images/c1b18a6ceb56def1ed0a7c87eadb86ab2860d1623f33502373a349364210f1d5.jpg)

<details>
<summary>natural_image</summary>

White microwave oven with whipped cream inside a bowl, open door open (no visible text or symbols)
</details>

go through the intersection and turn right.   
It will melt away as it warms up.   
(2) Multimodal referring via bounding boxes   
Figure 12: Examples generated from KOSMOS-2.

Input Prompt   
![](images/474f8e756049f20ff6c136ba3935ed320dfc538d5c72da0316bff0b5cadb0328.jpg)

<details>
<summary>natural_image</summary>

A bird standing on a wooden post with its wings spread, surrounded by blurred natural background (no text or symbols visible)
</details>

Describe this bird in detail:   
Completion

The eagle is a large bird with a black and white body and a yellow beak. It has large, dark wings and a pointed tail. It stands on a stump and spreads its wings, displaying its magnificent plumage.

![](images/9952b93d0abf1cba2dcfe48ac78a8fb44c528681a79bdb2b036eb63eeaa72a0b.jpg)

<details>
<summary>natural_image</summary>

A bird perched on a tree stump, with a close-up of its beak and its yellow beak (no text or symbols visible)
</details>

Input Prompt   
![](images/be7b4a3546eff531bc1e2042e6e99ed26f9e8b6fe22c7df06eff5e5e135c99d5.jpg)

<details>
<summary>natural_image</summary>

A formation of six jets flying in formation against a blue sky with white clouds (no visible text or symbols)
</details>

Describe this image in detail:   
Completion

Six fighter jets fly in formation in the sky, creating a beautiful cloud trail.

![](images/eacf8f908f27adcdb9212ec89551c4dc3f48f5a14502b6f442c1010cd89223bb.jpg)

<details>
<summary>natural_image</summary>

Aerial photo of multiple jets flying in formation against a blue sky with white clouds, leaving white smoke trails (no text or symbols visible)
</details>

Input Prompt   
![](images/4fef01e9db1f393263eb02a7d6590ce21e8863065d1fa7ee9e28571d5ab0463d.jpg)

<details>
<summary>natural_image</summary>

Person in red shirt blowing a large flame against palm trees and sky (no text or symbols visible)
</details>

Describe this image in detail:   
Completion

A man is standing in front of a fire, blowing the flames out of his mouth. He is in the middle of a tropical setting with palm trees surrounding him.

![](images/0f0df25d9b3c06926ab17a02735e7ed8c7c1750407b90d0ce949ad7c76ac356e.jpg)

<details>
<summary>natural_image</summary>

Person in red dress standing outdoors with palm trees and a large explosion display (no text or symbols visible)
</details>

Figure 13: Examples of grounded image captioning generated from KOSMOS-2.