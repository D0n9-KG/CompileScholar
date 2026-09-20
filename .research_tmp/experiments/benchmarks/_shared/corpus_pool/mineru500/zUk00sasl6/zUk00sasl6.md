# QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval

Jaehyun Kwak $^{1}$ Ramahdani Muhammad Izaaz Inhar $^{1}$ Se-Young Yun $^{1}$ Sung-Ju Lee $^{1}$

# Abstract

Composed Image Retrieval (CIR) retrieves relevant images based on a reference image and accompanying text describing desired modifications. However, existing CIR methods only focus on retrieving the target image and disregard the relevance of other images. This limitation arises because most methods employing contrastive learning—which treats the target image as positive and all other images in the batch as negatives—can inadvertently include false negatives. This may result in retrieving irrelevant images, reducing user satisfaction even when the target image is retrieved. To address this issue, we propose Query-Relevant Retrieval through Hard Negative Sampling (QURE), which optimizes a reward model objective to reduce false negatives. Additionally, we introduce a hard negative sampling strategy that selects images positioned between two steep drops in relevance scores following the target image, to effectively filter false negatives. In order to evaluate CIR models on their alignment with human satisfaction, we create Human-Preference FashionIQ (HP-FashionIQ), a new dataset that explicitly captures user preferences beyond target retrieval. Extensive experiments demonstrate that QURE achieves state-of-the-art performance on FashionIQ and CIRR datasets while exhibiting the strongest alignment with human preferences on the HP-FashionIQ dataset. The source code is available at https://github.com/jackwaky/QuRe.

# 1. Introduction

Composed Image Retrieval (CIR) retrieves images from a large corpus using text and image inputs, enabling precise

$^{1}$ KAIST. Correspondence to: Sung-Ju Lee <profsj@kaist.ac.kr>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](images/e32fcb216e73d530d1729195e8d356f6abb87c0259fab8f45b9cc7dcac5c557c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Blue t-shirt with short sleeves"] --> B["Existing CIR methods"]
    B --> C["QuRe"]
    C --> D["Hard Negative Set"]
    D --> E["Preference Optimization"]
    F["Image Corpus"] --> G["Contrastive Learning"]
    H["Target"] --> I["Irrelevant"]
    J["Target"] --> K["Relevant"]
    L["✓"] --> M["Preference Optimization"]
    N["Sort"] --> C
```
</details>

Figure 1. Comparison of existing CIR methods and QURE. Traditional CIR approaches treat all non-target images as negatives in contrastive learning. In contrast, QURE ranks the image corpus using a learned relevance score to identify a hard negative set. It then applies preference-based optimization to distinguish not only the target image but also other relevant images from hard negatives, leading to improved retrieval performance.

search capabilities in scenarios where textual descriptions alone are insufficient. This task is critical for applications such as e-commerce and Internet search, where users often seek results that match complex, visually nuanced specifications. For instance, as shown in Figure 1, a query combining an image of a shirt with the text ‘blue t-shirt with short sleeves’ requires CIR to retrieve matching images, including variations in style or color.

Existing CIR methods focus on retrieving the target image and overlook the broader relevance of other images. This limitation arises from the structure of CIR datasets, which typically annotate only a single target image per query and lack annotations for false negatives—relevant images not marked as targets. Furthermore, most CIR methods (Bai et al., 2024; Li et al.; Zhang et al., 2024) adopt contrastive learning, treating the target image as positive and all other images in the batch as negatives, inevitably including false negatives. While effective at ranking the target image within the top-k results, this approach frequently retrieves irrelevant images, as illustrated in Figure 1. Such irrelevance can reduce user satisfaction, as retrieval experience largely depends on the proportion of relevant items in the retrieved set (Al-Maskari & Sanderson, 2010).

We propose Query-Relevant Retrieval through Hard Negative Sampling (QURE), which aims to retrieve not only

the target image but also other relevant images with high ranks, thereby improving user satisfaction. To mitigate the inclusion of multiple false negatives, QURE adopts a reward model training objective (Ouyang et al., 2022), optimizing the likelihood of ranking a positive image above a single negative image, with the target image designated as positive for each query. A key challenge lies in sampling appropriate negatives, excluding false negatives while incorporating hard negative.

Hard negatives are generally defined as samples that (1) belong to a different class than the anchor and (2) have embeddings close to the anchor (Robinson et al., 2020; Ma et al., 2020; Tabassum et al., 2022; Huynh et al., 2022). Traditional hard negative selection relies on class labels, but in CIR, each query has a unique target, making class-based distinctions impractical. Moreover, randomly selecting negatives from the entire corpus or choosing those too similar to the target often leads to suboptimal training (Figure 4). To address these challenges, we redefine the first condition as 'less relevant to the query than the target,' ensuring that hard negatives differ from the query in at least one key attribute, such as color or shape.

To balance both conditions for selecting proper hard negatives in CIR, QURE periodically sorts the images in the corpus based on their relevance scores, calculated using the training model. During training, with the target image marked as positive, visually similar images (false negatives) tend to rank near the top of the sorted list. The sorted images are then divided into three groups: (1) false negatives, including the target image, (2) hard negatives, and (3) easy negatives. Hard negatives are defined as images that fall between two sharp declines in relevance scores, which occur after the target image. These steep drops indicate significant changes in relevance (Xia et al., 2024), ensuring that hard negatives differ from the query in at least one key attribute (Figure 6). This distinction makes hard negatives particularly valuable as challenging examples for training.

Our approach achieves state-of-the-art performance on the FashionIQ and CIRR datasets. To further evaluate alignment with human preferences, we created the Human-Preference FashionIQ (HP-FashionIQ) dataset, where human annotations indicate preferences between two retrieved sets for a given query. Experiments on HP-FashionIQ demonstrate that QURE achieves the best alignment with human preferences compared to baseline methods.

Our contributions are as follows:

- We propose QURE, a CIR algorithm that retrieves not only the target image but also other relevant images with high ranks to enhance user satisfaction.   
• We introduce a novel hard negative sampling strategy that

identifies images between two sharp declines in relevance scores after the target image. It optimizes model training by leveraging highly challenging samples while ensuring they are less relevant to the query than the target.

\- We achieve state-of-the-art performance on the FashionIQ and CIRR datasets, and demonstrate superior alignment with human preferences on the newly introduced Human-Preference FashionIQ (HP-FashionIQ) dataset.

# 2. Related Work

Vision-language foundation model. Vision-Language Models (VLMs) have gained attention for their ability to integrate multimodal data. Transformer-based architectures effectively handle both visual and language inputs (Li et al., 2019; Lu et al., 2019). Contrastive learning methods, which align visual and language modalities, have significantly improved performance in VLMs (Jia et al., 2021; Radford et al., 2021). New architectures combine features from both modalities. For instance, Flamingo (Alayrac et al., 2022) and BLIP (Li et al., 2022) use cross-attention, where visual hidden states from the vision encoder are inserted into cross-attention layers within the text encoder layers. BLIP-2 (Li et al., 2023) and QWEN (Bai et al., 2023) utilize pre-trained image and text encoders with learnable networks that bridge the gap between modalities.

QURE fine-tunes BLIP-2, using its image and text encoders with the Q-former module to handle modality gaps. We chose BLIP-2 for its efficient combination of image and text processing, requiring minimal training of the Q-former module.

Composed image retrieval. The CIR task fetches images using multimodal input features. A common approach is feature fusion, where the reference image and text are jointly embedded and compared against embeddings of candidate images (Vo et al., 2019; Dodds et al., 2020; Liu et al., 2021; Baldrati et al., 2023). Bi-BLIP4CIR (Liu et al., 2024) trains the text encoder using bi-directional training to capture both text directions of a given relation. CASE (Levy et al., 2024) leverages BLIP (Li et al., 2022) cross-attention architecture to perform an early fusion between the modalities. Other approaches transform images into pseudo-word embeddings or sentence-level prompts for text-to-image retrieval (Liu et al., 2023; Saito et al., 2023; Bai et al., 2024). MGUR (Chen et al., 2024) introduces an uncertainty loss for coarse-grained retrieval, and SPN4CIR (Feng et al., 2024) proposes a data generation method to scale positive and negative samples using multimodal LLMs.

However, all previous works trained models using contrastive loss, treating the target image as positive and all other images in the batch as negatives. This approach risks

![](images/814c2ab2f0f48be4148fbc6cde3b95d860a22d3bfd1c438409b951179d616aaf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Reference Image"] --> B["Image Encoder"]
    B --> C["Q-Former"]
    C --> D["Query Tokens"]
    D --> E["Query Features"]
    E --> F["Positive Features"]
    F --> G["Positive Score"]
    G --> H["KL-Div Loss"]
    I["Positive Image"] --> J["Image Encoder"]
    J --> K["Q-Former"]
    K --> L["Positive Features"]
    L --> M["Positive Score"]
    M --> N["Negative Features"]
    N --> O["Negative Image"]
    O --> P["Negative Score"]
    P --> Q["KL-Div Loss"]
    
    R["Training Objective"] --> S["“Two dogs of the same breed on the floor”"]
    S --> T["Q-Former"]
    T --> U["Query Tokens"]
    U --> V["Query Features"]
    V --> W["Positive Features"]
    W --> X["Positive Score"]
    X --> Y["KL-Div Loss"]
    
    Z["Hard Negative Sampling"] --> AA["Target Image: Steep"]
    AA --> AB["After target, Between Steeps (Hard Negatives)"]
    AB --> AC["Steep"]
```
</details>

Figure 2. Overview of QURE. During training, QURE periodically ranks corpus images by relevance score using the current model. Hard negatives are selected from the range between two sharp drops in relevance scores following the target image. A KL divergence loss is then used to train the model to assign higher relevance to the positive image than to a randomly chosen hard negative.

including false negatives as negatives, which may lead to irrelevant retrieval results. QURE overcomes this issue by employing the reward model objective, pairing each positive image with a sampled single hard negative.

Hard negative sampling. Hard negative sampling is a common technique in contrastive learning that selects more informative negatives rather than treating all images in the batch equally. HCL (Robinson et al., 2020) suggests that hard negatives should (1) belong to a different class than the anchor, and (2) have embeddings closer to it. However, in unsupervised settings like the CIR dataset, the lack of annotations makes it challenging to identify appropriate hard negatives. Embedding similarity alone can lead to including false negatives, prompting some approaches to incorporate model uncertainty, as higher uncertainty often indicates proximity to class boundaries (Ma et al., 2020). UnReMix (Tabassum et al., 2022) addresses this by combining embedding similarity with model uncertainty to identify suitable hard negatives. Meanwhile, FNC (Huynh et al., 2022) uses a predefined threshold to filter out false negatives and treats such samples as positives.

To address these challenges, QURE introduces a novel hard negative sampling strategy. It selects hard negatives from images positioned between two sharp drops in relevance scores following the target image. The steep drop after the target ensures that the selected image differs from the query in at least one key attribute (e.g., color or shape), making it a suitable hard negative for training.

# 3. Methodology

We denote a CIR dataset as $D = \{d_i \mid i = 1, \ldots, N_{data}\}$ , where each data point consists of a reference image, relative text, and a target image, i.e., $d_i = \{x_{I_i}, x_{T_i}, y_{I_i}\}$ . The goal of CIR is to retrieve a set of images from the image corpus $I = \{I_j \mid j = 1, \ldots, N_{img}\}$ , including the target image $y_I$ , where the retrieved images reflect the specified relative text $x_T$ while preserving the visual properties of the reference image $x_I$ . The training algorithm of QURE is provided in Appendix A.

# 3.1. QURE: Query-Relevant Retrieval through Hard Negative Sampling

Relevance score. For each image in corpus $I \in I$ , we define the relevance score to the query as the inner product of the bi-modal query and image embeddings:

$$
s (x _ {I}, x _ {T}, I) = \frac {Q (E _ {i m g} (x _ {I}) , x _ {T}) \cdot Q (E _ {i m g} (I))}{\tau}. \tag {1}
$$

Here, $E_{img}$ is the BLIP-2 image encoder, Q denotes the Q-Former, and $\tau$ is the learned BLIP-2 temperature. Q-Former processes the reference image embedding and relative text to align image and text modalities through cross-attention.

Training objective. QURE is trained to maximize the probability of preferring the highly relevant image, positive, over the less relevant image, negative. We model the latent preference distribution $p^{*}$ using the Bradley-Terry model (Bradley & Terry, 1952), where $I_{p}$ and $I_{n}$ denote the positive and negative images, respectively.

$$
p ^ {*} (I _ {p} \succ I _ {n} \mid x _ {I}, x _ {T}) = \sigma (s (x _ {I}, x _ {T}, I _ {p}) - s (x _ {I}, x _ {T}, I _ {n})). \tag {2}
$$

We set the target image as positive $I_{p} = y_{I}$ . To include the negative image $I_{n}$ , we construct the dataset $\mathbb{D}^{*} = \{(x_{I}, x_{T}, I_{p}, I_{n}) \mid (x_{I}, x_{T}, I_{p}) \in \mathbb{D}, I_{n} \in \mathbb{H}\}$ , where $I_{n}$ is drawn from the hard negative set H, defined in Section 3.2. The model is optimized by minimizing the negative log-likelihood (NLL) loss:

$$
\mathcal {L} = - \mathbb {E} _ {(x _ {I}, x _ {T}, I _ {p}, I _ {n}) \sim \mathbb {D} ^ {*}} [ \log (p ^ {*} (I _ {p} \succ I _ {n} \mid x _ {I}, x _ {T})) ]. (3)
$$

This objective is equivalent to minimizing the KL divergence between $p^{*}$ and a target distribution $p = [1, 0]$ , ensuring the model prefers the positive image over the negative.

# 3.2. Hard Negative Set Sampling

Defining hard negative set. We defined the conditions for an appropriate negative image $I_{n}$ in the CIR setting as follows:

C1. The negative image $I_{n}$ should be less relevant to the query than the target image $I_{p}$ .   
C2. The relevance score of the negative image $I_{n}$ should be similar to that of the target image $I_{p}$ .

However, identifying false negatives is practically infeasible as the CIR dataset annotates only the target image for each query. Additionally, selecting images that are overly dissimilar from the target may satisfy C1 but fail to meet C2, or vice versa, further complicating the selection process.

To balance these conditions, we sort the images in corpus $\mathbb{I}$ by their relevance scores obtained from the training model, forming an ordered set:

$$
\mathbb {S} _ {i} = \left\{s _ {i, 1}, \dots , s _ {i, N _ {i m g}} \right\}, \quad \text { where } s _ {i, 1} \geq \dots \geq s _ {i, N _ {i m g}} \tag {4}
$$

where $\mathbb{S}_i$ represents the relevance scores sorted in descending order for the $i$ -th query.

Based on the ordered set, we categorize images into three groups: (1) false negatives, which include the target image, (2) hard negatives, and (3) easy negatives. To identify hard negatives, we select images positioned between two steep relevance score drops occurring after the target image. The steep drop in relevance score indicates a noticeable shift in semantic similarity (Xia et al., 2024), helping to exclude false negatives while selecting negatives that are still challenging for the model. By focusing on this transition zone, we ensure that the selected negatives maintain a balance between similarity and distinction from the target image.

The subset of scores lower than the target score is defined as:

$$
\mathbb {S} _ {i} ^ {<   t a r g} = \{s _ {i, j} \mid s _ {i, j} <   s (x _ {I _ {i}}, x _ {T _ {i}}, y _ {i}) \}. \tag {5}
$$

From this subset, the indices of the top two largest degradations are identified as:

$$
k _ {1}, k _ {2} = \arg \operatorname{top-2} _ {j} (s _ {i, j} - s _ {i, j + 1} \mid s _ {i, j} \in \mathbb {S} _ {i} ^ {<   t a r g}). \tag {6}
$$

Finally, the hard negative set for the $i$ -th query is defined as:

$$
\mathbb {H} _ {i} = \left\{I _ {j} \mid j \in \left[ \min \left(k _ {1}, k _ {2}\right) + 1, \max \left(k _ {1}, k _ {2}\right) \right], \right.
$$

$$
s _ {i, j} <   s (x _ {I _ {i}}, x _ {T _ {i}}, y _ {i}) \}. \tag {7}
$$

Sampling hard negatives. To ensure diverse and informative negatives, a single image is sampled from the defined hard negative set for every epoch based on a uniform distribution.

# 4. HP-FashionIQ Dataset

The commonly used evaluation metric, Recall@k, fails to capture user satisfaction. While user satisfaction increases with the number of relevant items retrieved (Al-Maskari & Sanderson, 2010), Recall@k only checks whether the target image is retrieved, disregarding the relevance of other images in the fetched result. However, assessing the relevance of retrieved images is challenging, as it requires evaluating how well the images align with both the text and image inputs, which in CIR involves considering numerous complex attributes.

Human evaluation remains the most reliable way to measure image relevance, as humans can accurately assess how well an image matches a multi-modal query. To facilitate such evaluation, we created the Human-Preference FashionIQ (HP-FashionIQ) dataset, using the validation set of the FashionIQ dataset (Wu et al., 2021) with 61 participants. We selected the FashionIQ dataset for its high relevance and broad applicability, mirroring the search functionalities of e-commerce platforms.

Table 1. Statistics of HP-FashionIQ Dataset: Each query has two sets of retrieved images from different CIR models, annotated based on their preferences. 

<table><tr><td># Total Queries</td><td># Shirts Queries</td><td># Toptee Queries</td><td># Valid Queries</td></tr><tr><td>3,050</td><td>1,800</td><td>1,250</td><td>2,715</td></tr></table>

Data collection setting. Each question consisted of two retrieved image sets, each with the top 5 results from different CIR models. For every question, two CIR models were randomly selected from the following four: CLIP4CIR (Baldrati et al., 2023), Bi-BLIP4CIR (Liu et al., 2024), CoVR-BLIP (Ventura et al., 2024b), and SPRC (Bai et al., 2024). We provided queries and retrieved images from the 'shirts' or 'top tees' categories of the FashionIQ dataset to participants. Each participant was given 50 questions with a total of 100 sets of retrieved images, covering 3,050 queries in the FashionIQ validation set.

Annotation methodology. For each question in the survey, participants chose the preferred set between the two provided sets, assessing the alignment with human preferences.

Table 2. Performance comparison on the FashionIQ validation dataset across different methods. The best results are highlighted in bold, and the second-best are underlined. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Dress</td><td colspan="2">Shirt</td><td colspan="2">Toptee</td><td colspan="3">Average</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>Avg.</td></tr><tr><td>CoSMo (Lee et al., 2021)</td><td>23.60</td><td>49.18</td><td>18.11</td><td>43.18</td><td>24.63</td><td>54.31</td><td>22.11</td><td>48.89</td><td>35.50</td></tr><tr><td>MGUR (Chen et al., 2024)</td><td>23.15</td><td>48.74</td><td>18.99</td><td>43.47</td><td>25.55</td><td>52.83</td><td>22.56</td><td>48.35</td><td>35.46</td></tr><tr><td>CLIP4CIR (Baldrati et al., 2023)</td><td>38.32</td><td>63.90</td><td>44.31</td><td>65.41</td><td>47.27</td><td>70.98</td><td>43.30</td><td>66.76</td><td>55.03</td></tr><tr><td>Bi-BLIP4CIR (Liu et al., 2024)</td><td>39.12</td><td>62.92</td><td>39.21</td><td>62.81</td><td>44.37</td><td>67.06</td><td>40.90</td><td>64.26</td><td>52.58</td></tr><tr><td>CoVR-BLIP (Ventura et al., 2024b)</td><td>44.55</td><td>69.03</td><td>48.43</td><td>67.42</td><td>52.60</td><td>74.31</td><td>48.53</td><td>70.25</td><td>60.24</td></tr><tr><td>SPRC (Bai et al., 2024)</td><td>45.71</td><td>70.00</td><td>51.37</td><td>72.77</td><td>55.48</td><td>77.46</td><td>50.86</td><td>73.41</td><td>62.13</td></tr><tr><td>QURE</td><td>46.80</td><td>69.81</td><td>53.53</td><td>72.87</td><td>57.47</td><td>77.77</td><td>52.60</td><td>73.48</td><td>63.04</td></tr></table>

To our knowledge, this is the first CIR dataset with human preference-annotated retrieved images. An example of data from HP-FashionIQ is shown in Figure 3, with a detailed explanation of the data collection process in Appendix C.

![](images/24f18beb515510355d12c8f62cbdbc3d65068c3c3e4764bbebad98f2529c726a.jpg)

<details>
<summary>text_image</summary>

Reference Image
+
Relative Text
"Has long sleeves and brighter in color,
is white and long-sleeve"
Retrieved Set 1
Retrieved Set 2
</details>

Figure 3. An example from the HP-FashionIQ dataset: Given a query, two retrieved image sets are presented. In this example, the user preferred Set 2, as it better preserves the visual properties of the reference image.

Modality redundancy check. While CIR should consider both image and text, some examples focus solely on the image or the text. CASE (Levy et al., 2024) highlighted modality redundancy in FashionIQ, indicating that text is sometimes more influential than the image. We instructed participants to consider both input modalities equally. We asked them to flag instances where one modality seemed irrelevant to the retrieved images to exclude data that was unclear for human evaluation. A total of 307 queries were treated as irrelevant and excluded.

Sanity check. To identify instances of decreased user concentration during the annotation period, users rated the relevance of the retrieved sets before annotating for their preferred choice. Users rated each set on a 5-point Likert scale (Likert, 1932), with a score of 5 indicating a strong match with the query. Queries were discarded if the user did not prefer the set with the higher relevance score. As a result, 28 queries were excluded, leaving 2,715 valid queries. The total number of queries in the HP-FashionIQ dataset is shown in Table 1.

# 5. Experiments

# 5.1. Experimental Setup

Datasets. We evaluate the models on widely used CIR datasets, FashionIQ (Wu et al., 2021) and CIRR (Suhr et al., 2018), to assess their ability to retrieve the target image. Additionally, we evaluate them on the HP-FashionIQ dataset to assess their alignment with human preferences.

Implementation. We used BLIP-2 (Li et al., 2023) with a ViT-L image encoder. Following previous work (Baldrati et al., 2023), we resized images to $224 \times 224$ with a 1.25 padding ratio. QURE is trained using the AdamW optimizer (Loshchilov, 2017) for 50 epochs on CIRR and 30 epochs on FashionIQ. The hard negative set H was defined $n_{def}$ times, starting with a warm-up phase where H initially included the entire corpus except for the target during the first $\lfloor n_{epoch}/n_{def} \rfloor$ epochs. The hard negative set H is updated every $\lfloor n_{epoch}/n_{def} \rfloor$ epochs. We set $n_{def}$ to six for both FashionIQ and CIRR. All experiments were conducted using a single Nvidia RTX 3090 GPU.

# 5.2. Comparison with State-of-the-art CIR Models.

Table 7 presents the evaluation of CIR models on the FashionIQ dataset. QURE consistently achieves the best or second-best performance across all categories, attaining the highest overall average. Notably, QURE demonstrates

Table 3. Performance comparison on the CIRR test dataset across different methods, where Recall $_{s}$ @K represents Recallsubset@K. The best results are highlighted in bold, and the second-best are underlined. 

<table><tr><td rowspan="2">Method</td><td colspan="4">Recall@K</td><td colspan="3"> $Recall_s @K$ </td><td>Average</td></tr><tr><td>K=1</td><td>K=5</td><td>K=10</td><td>K=50</td><td>K=1</td><td>K=2</td><td>K=3</td><td>R@5 +  $R_s @1$ </td></tr><tr><td>CosMo (Lee et al., 2021)</td><td>6.48</td><td>23.11</td><td>34.63</td><td>67.33</td><td>20.29</td><td>40.22</td><td>60.80</td><td>43.55</td></tr><tr><td>MGUR (Chen et al., 2024)</td><td>5.78</td><td>21.45</td><td>33.42</td><td>67.06</td><td>20.29</td><td>40.22</td><td>60.80</td><td>42.91</td></tr><tr><td>CLIP4CIR (Baldrati et al., 2023)</td><td>44.12</td><td>77.23</td><td>86.51</td><td>97.95</td><td>73.11</td><td>89.11</td><td>95.42</td><td>75.17</td></tr><tr><td>Bi-BLIP4CIR (Liu et al., 2024)</td><td>32.55</td><td>64.36</td><td>76.53</td><td>91.61</td><td>63.54</td><td>82.46</td><td>92.48</td><td>63.95</td></tr><tr><td>CoVR-BLIP (Ventura et al., 2024b)</td><td>39.76</td><td>70.15</td><td>80.89</td><td>95.01</td><td>72.46</td><td>87.86</td><td>94.77</td><td>71.30</td></tr><tr><td>SPRC (Bai et al., 2024)</td><td>50.75</td><td>80.58</td><td>88.72</td><td>97.59</td><td>79.57</td><td>91.76</td><td>96.70</td><td>80.07</td></tr><tr><td>QURE</td><td>52.22</td><td>82.53</td><td>90.31</td><td>98.17</td><td>78.51</td><td>91.28</td><td>96.48</td><td>80.52</td></tr></table>

significant improvements over SPRC (Bai et al., 2024) when the retrieved set size is small, such as in Recall@10, with gains of 1.09%, 2.16%, and 1.99% for the dress, shirt, and toptee categories, respectively. Previous work (Levy et al., 2024) has identified high modality redundancy in the FashionIQ dataset, where text dominates the retrieval process, favoring text-based methods such as Bi-BLIP4CIR (Liu et al., 2024) and SPRC (Bai et al., 2024). Despite this bias, QURE achieves state-of-the-art performance, improving the overall average recall by 10.46% and 0.91% compared with Bi-BLIP4CIR and SPRC, respectively. These results highlight QURE's effectiveness in accurately retrieving the target image, even in the presence of modality redundancy.

Table 3 shows the evaluation results on CIRR, a general-domain dataset. QURE achieves the highest performance across all Recall@k metrics, particularly excelling in Recall@1 and Recall@5, surpassing the current state-of-the-art method, SPRC (Bai et al., 2024), by 1.47% and 1.95%, respectively. Regarding Recall\_s@k, which measures retrieval performance from a subset containing relevant images and the target, QURE achieves the second-best results. This result is attributed to the design of QURE, where even false negative images can receive higher scores than the target as they closely match the query. This behavior arises from our hard negative set definition, which excludes false negatives, ensuring that relevant images are not treated as negatives and can be ranked higher than the target. Notably, QURE achieves state-of-the-art performance on the combined Recall@5 + Recall\_s@1 average.

# 5.3. Evaluation with the HP-FashionIQ Dataset

We evaluate the alignment of CIR models with human preferences. Given a query $\{x_{I}, x_{T}\}$ , each participant was presented with two different sets of retrieved images, Set 1 and Set 2, and annotated their preferences between them. To calculate the overall relevance score of a CIR model for each set, we averaged the relevance scores of the five retrieved images within the set:

$$
s _ {r e l} (S e t i) = \frac {1}{5} \sum_ {I \in S e t i} s _ {r e l} (x _ {I}, x _ {T}, I). \tag {8}
$$

where $s_{rel}$ denotes the relevance score (e.g., cosine similarity) computed by CIR models.

The alignment with human preferences is measured through the preference rate, which represents the conditional probability that Set 1 is preferred when its relevance score is greater than that of Set 2. Formally, we define the preference rate as:

$$
\mathbb {P} (S e t 1 \succ S e t 2 \mid s _ {r e l} (S e t 1) > s _ {r e l} (S e t 2)). \tag {9}
$$

Table 4 shows that the ranking of CIR models based on their alignment with human preferences on the HP-FashionIQ dataset differs from their performance rankings on the FashionIQ dataset (Table 7) using the Recall@k metric. For instance, MGUR achieves performance comparable to CosMo in retrieving the target image but aligns more closely with human preferences. This discrepancy stems from MGUR's additional coarse-grained loss, which considers both the target image and visually similar alternatives as positives. Moreover, while SPRC, CoVR-BLIP, and Bi-BLIP4CIR surpass CLIP4CIR in terms of Recall@k on the FashionIQ dataset, CLIP4CIR aligns better with human preferences on the HP-FashionIQ dataset, despite its lower accuracy in retrieving the exact target image.

QURE achieves the best alignment with human preferences, which shows that Set 1 is preferred 74.55% of the time when its relevance score exceeds that of Set 2. This result

Table 4. Preference rate comparison on HP-FashionIQ dataset across different methods. The best results are highlighted in bold, and the second-best are underlined. 

<table><tr><td>Method</td><td>Preference Rate (%)</td></tr><tr><td>CosMo (Lee et al., 2021)</td><td>72.96</td></tr><tr><td>MGUR (Chen et al., 2024)</td><td>73.99</td></tr><tr><td>CLIP4CIR (Baldrati et al., 2023)</td><td>74.45</td></tr><tr><td>Bi-BLIP4CIR (Liu et al., 2024)</td><td>67.33</td></tr><tr><td>CoVR-BLIP (Ventura et al., 2024b)</td><td>73.15</td></tr><tr><td>SPRC (Bai et al., 2024)</td><td>73.82</td></tr><tr><td>QURE</td><td>74.55</td></tr></table>

![](images/1aca81f2be8991b6d397ae441f8d3695293b2c23aa6ab1d75e0851ae511a897c.jpg)

<details>
<summary>line</summary>

| Epoch | Top-k | All corpus | QuRe | After target, top-k |
|-------|-------|------------|------|---------------------|
| 0     | 57.0  | 57.0       | 53.0 | 57.0                |
| 5     | 54.0  | 57.5       | 59.0 | 60.0                |
| 10    | 60.0  | 58.0       | 61.0 | 61.5                |
| 15    | 57.0  | 58.5       | 62.0 | 61.5                |
| 20    | 61.0  | 59.0       | 62.5 | 61.5                |
| 25    | 60.5  | 59.0       | 62.5 | 61.5                |
</details>

Figure 4. Average recall on the FashionIQ validation set using four hard negative set definitions. After the initial hard negative set is established (e.g., at epoch 4), the choice of definition significantly influences average recall throughout training.

demonstrates QURE's ability not only to retrieve the correct target image but also relevant images that best align with human preferences.

# 5.4. Ablation Studies

We present ablation results under various scenarios, with additional experiments included in Appendix B.

Zero-shot performance comparison. We evaluated the zero-shot performance of the models on the CIRCO dataset using those pre-trained on the CIRR dataset. Although QURE and baseline methods are not explicitly designed for zero-shot tasks, models that effectively retrieve relevant images are expected to perform well in such scenarios. Furthermore, CIRCO is the first CIR dataset to include multiple ground truths, addressing the issue of false negatives in existing datasets. Thus, evaluating the mean average precision at k (mAP@k) on this dataset provides a reliable measure of the model's ability to retrieve relevant items.

Table 5 reveals that QURE achieves the best performance among all baselines, outperforming the second-best method, CoVR-BLIP, by an average margin of 5.13 mAP. While Table 3 indicates SPRC (Bai et al., 2024) significantly outperforms CoVR-BLIP (Ventura et al., 2024b) on the CIRR dataset, CoVR-BLIP shows better performance in the zero-shot setting. This suggests that CoVR-BLIP generalizes unseen tasks better. The results highlight that QURE achieves the highest generalizability, significantly improving over other baselines.

Table 5. Zero-shot performance comparison on the CIRCO dataset across different methods. The best results are highlighted in bold, and the second-best are underlined. 

<table><tr><td>Method</td><td>mAP@5</td><td>mAP@10</td><td>mAP@25</td><td>mAP@50</td></tr><tr><td>CosMo</td><td>0.31</td><td>0.40</td><td>0.47</td><td>0.53</td></tr><tr><td>MGUR</td><td>0.14</td><td>0.17</td><td>0.25</td><td>0.30</td></tr><tr><td>CLIP4CIR</td><td>10.58</td><td>11.18</td><td>12.32</td><td>12.96</td></tr><tr><td>Bi-BLIP4CIR</td><td>4.74</td><td>4.97</td><td>5.69</td><td>6.10</td></tr><tr><td>CoVR-BLIP</td><td>18.35</td><td>19.25</td><td>21.02</td><td>21.88</td></tr><tr><td>SPRC</td><td>17.57</td><td>18.48</td><td>20.14</td><td>20.98</td></tr><tr><td>QURE</td><td>23.22</td><td>24.23</td><td>26.26</td><td>27.24</td></tr></table>

![](images/4e4744650dacb302a3f234b559017910738257c6446ac7b37fd1f62a8b36e314.jpg)

<details>
<summary>line</summary>

| Epoch | FashionIQ | CIRR  |
|-------|-----------|-------|
| 5     | 800       | 820   |
| 10    | 170       | 430   |
| 15    | 120       | 180   |
| 20    | 100       | 120   |
| 25    | 90        | 100   |
| 30    | 80        | 90    |
| 40    | 70        | 85    |
| 50    | 60        | 80    |
</details>

Figure 5. Average size of the hard negative set per epoch on FashionIQ and CIRR. This plot shows how the average number of hard negatives per query evolves across epochs in which the set is defined.

Effect of hard negative set definition strategy. To assess the effectiveness of the hard negative set definition approach, we compare four strategies on the FashionIQ dataset. The hard negative set is defined as follows: (1) the entire image corpus (All corpus), (2) the top-k images based on relevance scores (Top-k), (3) the top-k images with relevance scores lower than the target (After target, top-k), and (4) images between sharp drops in relevance scores, occurring after the target (QURE).

Figure 4 presents the average recall of each strategy across training epochs. The results indicate that the All corpus approach shows consistent improvement; however, it ul-

![](images/597b885e525119ae236133bc4496fa8cd2478ffb6a18b01effbf7ae9ab165efc.jpg)

<details>
<summary>text_image</summary>

Reference Image
Relative Text
Target Image
"is short sleeved and has a collar,
is grey with shorter sleeves"
Reference Image
Relative Text
Target Image
"has vertical and horizontal stripes,
is darker and has a plaid pattern"
Hard Negative Set
Hard Negative Set
</details>

Figure 6. Hard negative set examples for two shirt-category queries in FashionIQ. Each set contains images that are semantically similar to the target but deemed less relevant by the model.

timately leads to suboptimal performance as the selected negatives from the entire corpus may be too irrelevant. Inspired by HCL (Robinson et al., 2020), we define a hard negative set by selecting the top-100 images based on relevance scores (Top-k). This approach occasionally enhances model performance (e.g., at epochs 10 and 20), but it often degrades due to the inclusion of false negatives. As training progresses, since the target image is treated as positive, visually similar images (false negatives) tend to rank higher. To mitigate the issue in the Top-k approach, we analyze the performance of the After target, top-k strategy, which excludes false negatives by selecting the top-k images following the target. Unlike Top-k, this method provides consistent improvements without fluctuations but gradually leads to slight performance degradation over time. This decline results from the heuristic assumption that all images right after the target are reliable hard negatives. In contrast, QURE identifies points where relevance drops sharply and selects images between two such points, resulting in stable performance improvements throughout training. These results show that, with a novel hard negative sampling method, QURE effectively balances the conditions C1 and C2 of selecting proper hard negatives.

Size of hard negative set. The number of hard negative images varies significantly depending on the complexity of bi-modal queries and the corpus. For instance, when the input text contains fewer attributes, such as 'blue shirt with short sleeves', the hard negative set may include images that match either 'blue' or 'short sleeves', resulting in a larger set. Therefore, it is crucial to define a query-specific hard negative set size (Xia et al., 2024).

Figure 5 shows the average size of the hard negative set across all queries in the FashionIQ and CIRR datasets. While QURE does not explicitly define the size of the hard negative set, the results indicate a consistent decrease throughout training. Initially, the model identifies a broader range of images as hard negatives due to lower confidence. As training progresses, it refines this selection, yielding a smaller yet more challenging hard negative set. This dynamic resembles curriculum learning, where increasingly difficult samples accelerate model convergence.

Visualization of hard negative set. To evaluate our hard negative set sampling strategy, we present qualitative examples from the FashionIQ dataset in Figure 6, illustrating two queries. In the first query, the hard negatives either lack a collar or text from the reference image 'ECK' or depict shirts instead of t-shirts, differing from the input image. In the second query, all retrieved shirts contain only vertical stripes while successfully retrieving darker shirts. These examples demonstrate that the hard negative sets defined by our method consist of images that are less relevant than the target (C1) while remaining semantically similar (C2), thereby satisfying both conditions outlined in Section 3.2.

# 6. Conclusion

We present QURE, a CIR model designed to retrieve both target and relevant images to enhance user satisfaction. Existing CIR datasets typically annotate only the target image per query, leading prior methods to rely on contrastive learning that treats all non-target images as negatives. QURE mitigates false negatives by leveraging reward model objectives and introduces a novel hard negative sampling strategy, selecting images between two sharp relevance score drops after the target. To evaluate the alignment of CIR models with human preferences, we introduce the Human-Preference FashionIQ (HP-FashionIQ) dataset. QURE achieves state-of-the-art performance on both the FashionIQ and CIRR datasets and demonstrates the highest alignment with human preferences on the HP-FashionIQ dataset.

# Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT)(RS-2024-00337007, 50%), Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST), 5%), and the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration, 45%)

# Impact Statement

This paper introduces Query-Relevant Retrieval through Hard Negative Sampling (QURE), an approach to Composed Image Retrieval (CIR) that improves retrieval quality by retrieving both the target and other relevant images, improving user satisfaction. It addresses the false negative problem in CIR with a novel hard negative sampling strategy, advancing Machine Learning and Information Retrieval. QURE contributes to improving search efficiency in e-commerce and visual search platforms, which benefits businesses and consumers by delivering more relevant search results. Additionally, the human preference-based evaluation (HP-FashionIQ dataset) aligns CIR models with human expectations, making AI-powered retrieval more user-centric.

# References

Al-Maskari, A. and Sanderson, M. A review of factors influencing user satisfaction in information retrieval. Journal of the American Society for Information Science and Technology, 61(5):859–868, 2010.   
Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.   
Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.   
Bai, Y., Xu, X., Liu, Y., Khan, S., Khan, F., Zuo, W., Goh, R. S. M., and Feng, C.-M. Sentence-level prompts benefit composed image retrieval. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=m3ch3kJL7q.   
Baldrati, A., Bertini, M., Uricchio, T., and Del Bimbo, A.

Composed image retrieval using contrastive learning and task-oriented clip-based features. ACM Transactions on Multimedia Computing, Communications and Applications, 20(3):1–24, 2023.

Bradley, R. A. and Terry, M. E. Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons. Biometrika, 39(3-4):324–345, 12 1952. ISSN 0006-3444. doi: 10.1093/biomet/39.3-4.324. URL https://doi.org/10.1093/biomet/39.3-4.324.

Chen, Y., Zheng, Z., Ji, W., Qu, L., and Chua, T.-S. Composed image retrieval with text feedback via multi-grained uncertainty regularization. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=Yb5KvPkKQg.

Dodds, E., Culpepper, J., Herdade, S., Zhang, Y., and Boakye, K. Modality-agnostic attention fusion for visual search with text feedback. arXiv preprint arXiv:2007.00145, 2020.

Feng, Z., Zhang, R., and Nie, Z. Improving composed image retrieval via contrastive learning with scaling positives and negatives. arXiv preprint arXiv:2404.11317, 2024.

Huynh, T., Kornblith, S., Walter, M. R., Maire, M., and Khademi, M. Boosting contrastive self-supervised learning with false negative cancellation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2785–2795, 2022.

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pp. 4904–4916. PMLR, 2021.

Lee, S., Kim, D., and Han, B. Cosmo: Content-style modulation for image retrieval with text feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 802–812, 2021.

Levy, M., Ben-Ari, R., Darshan, N., and Lischinski, D. Data roaming and quality assessment for composed image retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 2991–2999, 2024.

Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023.

Li, L. H., Yatskar, M., Yin, D., Hsieh, C.-J., and Chang, K.-W. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019.   
Li, W., Fan, H., Wong, Y., Yang, Y., and Kankanhalli, M. Improving context understanding in multimodal large language models via multimodal composition learning. In Forty-first International Conference on Machine Learning.   
Likert, R. A technique for the measurement of attitudes. Archives of psychology, 1932.   
Liu, Z., Rodriguez-Opazo, C., Teney, D., and Gould, S. Image retrieval on real-life images with pre-trained vision-and-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2125–2134, 2021.   
Liu, Z., Sun, W., Teney, D., and Gould, S. Candidate set re-ranking for composed image retrieval with dual multimodal encoder. arXiv preprint arXiv:2305.16304, 2023.   
Liu, Z., Sun, W., Hong, Y., Teney, D., and Gould, S. Bidirectional training for composed image retrieval via text prompt learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5753–5762, 2024.   
Loshchilov, I. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
Lu, J., Batra, D., Parikh, D., and Lee, S. Vilbert: Pre-training task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019.   
Ma, S., Zeng, Z., McDuff, D., and Song, Y. Active contrastive learning of audio-visual video representations. arXiv preprint arXiv:2009.09805, 2020.   
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
Robinson, J., Chuang, C.-Y., Sra, S., and Jegelka, S. Contrastive learning with hard negative samples. arXiv preprint arXiv:2010.04592, 2020.

Saito, K., Sohn, K., Zhang, X., Li, C.-L., Lee, C.-Y., Saenko, K., and Pfister, T. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19305–19314, 2023.   
Suhr, A., Zhou, S., Zhang, A., Zhang, I., Bai, H., and Artzi, Y. A corpus for reasoning about natural language grounded in photographs. arXiv preprint arXiv:1811.00491, 2018.   
Tabassum, A., Wahed, M., Eldardiry, H., and Lourentzou, I. Hard negative sampling strategies for contrastive representation learning. arXiv preprint arXiv:2206.01197, 2022.   
Ventura, L., Yang, A., Schmid, C., and Varol, G. Covr-2: Automatic data construction for composed video retrieval. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024a.   
Ventura, L., Yang, A., Schmid, C., and Varol, G. CoVR: Learning composed video retrieval from web video captions. AAAI, 2024b.   
Vo, N., Jiang, L., Sun, C., Murphy, K., Li, L.-J., Fei-Fei, L., and Hays, J. Composing text and image for image retrieval—an empirical odyssey. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6439–6448, 2019.   
Wu, H., Gao, Y., Guo, X., Al-Halah, Z., Rennie, S., Grauman, K., and Feris, R. Fashion iq: A new dataset towards retrieving images by natural language feedback. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pp. 11307–11317, 2021.   
Xia, P., Zhu, K., Li, H., Wang, T., Shi, W., Wang, S., Zhang, L., Zou, J., and Yao, H. Mmed-rag: Versatile multimodal rag system for medical vision language models. arXiv preprint arXiv:2410.13085, 2024.   
Zhang, K., Luan, Y., Hu, H., Lee, K., Qiao, S., Chen, W., Su, Y., and Chang, M.-W. Magiclens: Self-supervised image retrieval with open-ended instructions. Forty-first International Conference on Machine Learning, 2024.

# A. Algorithm

Algorithm 1 Training Flow of QURE   
1: Input: Parameters $\theta$ , CIR dataset $\mathbb{D}$ , Image corpus $\mathbb{I}$ , Number of defining negative set $n_{\mathrm{def}}$ , Total epochs $n_{\mathrm{epoch}}$ 2: for each epoch $e$ do  
3: if $e == 0$ then  
4: $\mathbb{H} \leftarrow \mathbb{I} \setminus y_I$ // Warmup: using all candidates as the negative set  
5: else if $e > 0$ and $e \bmod \lfloor n_{\mathrm{epoch}} / n_{\mathrm{def}} \rfloor == 0$ then  
6: $\mathbb{H} \leftarrow \{I \mid s(x_I, x_T, y_I) < s(x_I, x_T, I), I \in \text{range between two largest score degradations}\}$ // Equation (7)  
7: end if  
8: for each batch $b$ do  
9: $I_{n_b} \leftarrow \text{Sample}(\mathbb{H}_b)$ // Sample one negative for every query  
10: $\mathcal{L} \leftarrow -\log(\sigma(s(x_{I_b}, x_{T_b}, I_{p_b}) - s(x_{I_b}, x_{T_b}, I_{n_b}))$ // Equation (3)  
11: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ 12: end for  
13: end for

# B. Additional Experiments

# B.1. Ablations with Consistent Model Backbone

In Section 5, we compare QURE with existing baselines on FashionIQ, CIRR, HP-FashionIQ, and CIRCO. However, model backbones are not unified, as we follow the comparison settings used in prior CIR methods (Baldrati et al., 2023; Liu et al., 2024; Bai et al., 2024). To ensure a fairer comparison, we conduct experiments using both BLIP and BLIP-2 backbones. Specifically, we train QURE with the BLIP backbone to compare against Bi-BLIP4CIR. We also identify CoVR-2 (Ventura et al., 2024a), the latest version of CoVR-BLIP that adopts BLIP-2, and compare it with our original QURE model.

Table 6, Table 7, and Table 8 present results on the CIRR, FashionIQ, HP-FashionIQ, and CIRCO datasets. QURE with a BLIP backbone consistently outperforms Bi-BLIP4CIR. Moreover, CoVR-2, which adopts a BLIP-2 backbone, still underperforms compared to our original QURE model with the same backbone.

Table 6. Performance comparison on the CIRR test dataset with consistent model backbones, where Recall $_{s}$ @K represents Recallsub-set@K. The best results are highlighted in bold. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">Recall@K</td><td colspan="3"> $Recall_s@K$ </td><td>Average</td></tr><tr><td>K=1</td><td>K=5</td><td>K=10</td><td>K=50</td><td>K=1</td><td>K=2</td><td>K=3</td><td>R@5 +  $R_s@1$ </td></tr><tr><td>Bi-BLIP4CIR</td><td>BLIP</td><td>32.55</td><td>64.36</td><td>76.53</td><td>91.61</td><td>63.54</td><td>82.46</td><td>92.48</td><td>63.95</td></tr><tr><td>QURE</td><td>BLIP</td><td>51.52</td><td>80.29</td><td>88.89</td><td>97.74</td><td>78.02</td><td>91.23</td><td>96.55</td><td>79.16</td></tr><tr><td>CoVR-2</td><td>BLIP-2</td><td>42.80</td><td>74.60</td><td>83.90</td><td>96.22</td><td>69.49</td><td>86.22</td><td>93.98</td><td>72.05</td></tr><tr><td>QURE</td><td>BLIP-2</td><td>52.22</td><td>82.53</td><td>90.31</td><td>98.17</td><td>78.51</td><td>91.28</td><td>96.48</td><td>80.52</td></tr></table>

# B.2. Visualization of Score Steepness

QURE defines a query-specific hard negative set by excluding false and easy negatives, aiming to enhance training effectiveness. Specifically, it selects images between the two largest drops in relevance scores following the target image. To analyze this steepness, we visualize the sorted relevance scores.

Figure 7 illustrates the relevance scores for a sample query from the FashionIQ dataset at two stages: before training and after the warm-up phase. QURE identifies hard negatives as the images between the red and green lines, determined by the top two largest drops in relevance scores after the target. These steep declines-visible immediately before the red line and after the green line-suggest substantial drops in relevance (Xia et al., 2024), effectively separating false negatives, hard negatives, and easy negatives.

Table 7. Performance comparison on the FashionIQ validation dataset with consistent model backbones. The best results are highlighted in bold. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="2">Dress</td><td colspan="2">Shirt</td><td colspan="2">Toptee</td><td colspan="3">Average</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>Avg.</td></tr><tr><td>Bi-BLIP4CIR</td><td>BLIP</td><td>39.12</td><td>62.92</td><td>39.21</td><td>62.81</td><td>44.37</td><td>67.06</td><td>40.90</td><td>64.26</td><td>52.58</td></tr><tr><td>QURE</td><td>BLIP</td><td>40.80</td><td>64.90</td><td>45.93</td><td>65.90</td><td>52.07</td><td>72.87</td><td>46.27</td><td>67.89</td><td>57.08</td></tr><tr><td>CoVR-2</td><td>BLIP-2</td><td>46.41</td><td>69.51</td><td>49.75</td><td>67.76</td><td>51.86</td><td>72.46</td><td>49.34</td><td>69.91</td><td>59.63</td></tr><tr><td>QURE</td><td>BLIP-2</td><td>46.80</td><td>69.81</td><td>53.53</td><td>72.87</td><td>57.47</td><td>77.77</td><td>52.60</td><td>73.48</td><td>63.04</td></tr></table>

Table 8. Comparison of HP-FashionIQ Preference Rate and CIRCO zero-shot performance with consistent model backbones. The best results are highlighted in bold. 

<table><tr><td colspan="3">HP-FashionIQ</td></tr><tr><td>Method</td><td>Backbone</td><td>Preference Rate (%)</td></tr><tr><td>Bi-BLIP4CIR</td><td>BLIP</td><td>67.33</td></tr><tr><td>QURE</td><td>BLIP</td><td>75.28</td></tr><tr><td>CoVR-2</td><td>BLIP-2</td><td>71.99</td></tr><tr><td>QURE</td><td>BLIP-2</td><td>74.55</td></tr></table>

<table><tr><td colspan="6">CIRCO (Zero-shot)</td></tr><tr><td>Method</td><td>Backbone</td><td>m@5</td><td>m@10</td><td>m@25</td><td>m@50</td></tr><tr><td>Bi-BLIP4CIR</td><td>BLIP</td><td>4.74</td><td>4.97</td><td>5.69</td><td>6.1</td></tr><tr><td>QURE</td><td>BLIP</td><td>20.85</td><td>21.48</td><td>23.35</td><td>24.31</td></tr><tr><td>CoVR-2</td><td>BLIP-2</td><td>23.18</td><td>23.59</td><td>25.57</td><td>26.49</td></tr><tr><td>QURE</td><td>BLIP-2</td><td>23.22</td><td>24.23</td><td>26.26</td><td>27.24</td></tr></table>

Since Figure 7 shows only a single query, we also present aggregated results across all queries in Figure 8. This aggregated visualization shows that, after the warm-up phase, hard negatives shift toward higher-ranked positions, likely capturing more true hard negatives. This supports our design choice of introducing a warm-up stage prior to hard negative selection. Without this stage, the selected negatives tend to include many easy examples, as observed in the left panel of the figure.

Sorted Score Plot after the Target   
![](images/9a99ab3caa9948a4851e829ad01c9766bd785bc23e3214024a4608b2503d59a3.jpg)

<details>
<summary>line</summary>

| Index | Relevance Score |
|-------|-----------------|
| 80    | 0.1750          |
| 120   | 0.1700          |
| 200   | 0.1630          |
| 300   | 0.1560          |
</details>

![](images/e6b28132709f73eae02ba8a387579b9697837b212218953feec57d82a915eedd.jpg)

<details>
<summary>line</summary>

| Index | Relevance Score |
|-------|-----------------|
| 0     | 0.50            |
| 10    | 0.48            |
| 20    | 0.45            |
| 30    | 0.43            |
| 40    | 0.42            |
| 50    | 0.41            |
| 60    | 0.40            |
| 70    | 0.39            |
| 80    | 0.38            |
| 90    | 0.37            |
| 100   | 0.36            |
</details>

Figure 7. Visualization of sorted relevance scores for a FashionIQ query at two stages: prior to training and after completing the warm-up phase.

# C. HP-FashionIQ Dataset

# C.1. Data Annotation Examples

We conducted a data collection via Google Forms. Each form consisted of instructions and 25 questions, with each question including a query and two different sets of retrieved images. Each participant completed two forms, covering 50 queries from the FashionIQ validation dataset, with no overlap between participants.

Aggregated Sorted Score Plot after the Target   
![](images/7026cae253816e0a644f4dfaaae290d3e127ce3149391fc5a9e0ea9caa45c13f.jpg)

<details>
<summary>line</summary>

| Index | Score | Average target index | Average hard negatives start index | Average hard negatives end index |
|-------|-------|----------------------|------------------------------------|----------------------------------|
| 0     | 0.36  | -                    | -                                  | -                                |
| 1000  | 0.22  | 0.22                 | -                                  | -                                |
| 2000  | 0.18  | -                    | -                                  | -                                |
| 3000  | 0.15  | -                    | -                                  | -                                |
| 4000  | 0.13  | -                    | -                                  | -                                |
| 5000  | 0.11  | -                    | -                                  | -                                |
| 6000  | 0.05  | -                    | -                                  | -                                |
</details>

![](images/7de4d0c8220e2035925ed41ebca3875c2952a6d19f74cb0d25a2b3bf88967092.jpg)

<details>
<summary>line</summary>

| Index | Relevance Score |
|-------|-----------------|
| 0     | 0.55            |
| 1000  | 0.35            |
| 2000  | 0.28            |
| 3000  | 0.22            |
| 4000  | 0.18            |
| 5000  | 0.12            |
| 6000  | 0.05            |
</details>

Figure 8. Visualization of sorted relevance scores aggregated over all FashionIQ queries at two stages: before training and after the warm-up phase.

Fig 9 shows the guidelines provided to participants, who were asked to score the retrieved image sets based on relevance to the query using a 5-point Likert scale. We gave participants a relatable scenario that required them to evaluate the results from two different online shopping malls based on their input. Figure 10 illustrates an example of a reference image (original image), relative text (user text), and two sets of retrieved images from different CIR models (Shopping Mall 1 and 2). Participants (1) rated the relevance of each set, (2) indicated which set they preferred, and (3) noted whether any results were irrelevant to the reference image or text, as shown in Figure 11.

1. Original Image: The image you have.   
2. User Text: A description of how the image you are looking for differs from the original image.   
3. Two sets of image search results: Each set contains 5 images searched through two different shopping malls.   
Please evaluate how well the two sets of image search results match the original image and the user text. Each set consists of 5 images, and the higher the match with the original image and the user text, the higher the score you should give. You should evaluate each set as a whole, not each individual image.   
1: Does not match at all.   
2: Does not match.   
3: Matches to an average degree.   
4: Matches.   
5: Matches very well.   
- Recommend completing the survey on a PC rather than a mobile device to make it easier to view the images.   
- The order of the 5 images within the image search result sets does not matter.   
- The search results are not generated images but are found from a fixed set of images that best match the original image and user text. Therefore, the quality of the search results may not meet user expectations. Please score as consistently as possible.   
- The user text in this survey is from the dataset as is, so there may be typos or duplicated expressions.   
- There are questions throughout the survey where you will be asked to explain the reason for your score.   
- There will be simple math problems before proceeding to the next question. If you get these problems wrong, you cannot move on to the next page.

# √ Evaluation Method:

# Notes:

Figure 9. Guidelines for user survey.

Original Image / User Text: is more green and has 3/4 sleeves and has more colors and is sportier

![](images/a28a0128b84684bb6b93df69bc7be0b3d5b8a5f8acb7d9c0426ae62ea64f3ee1.jpg)

<details>
<summary>text_image</summary>

Suinness
</details>

Shopping Mall 1

![](images/32aab346b021279c634677c4c62dfc541b2272714debd5850fbf6322ccc307d3.jpg)

Shopping Mall 2

![](images/605fc7e78266721caecd608db2fdf9936aaa439e28c3bb8899c59957e324c3a7.jpg)  
Figure 10. Example of query and two set of retrieved images in user survey.

Score for shopping mall 1 \*

1: Does not match at all, 2: Does not match, 3: Matches to an average degree, 4: Matches, 5: Matches very well

1

2

3

4

5

![](images/462874649d5b31df6398db25f7371b79dd2de3bd051545b4a4920451d49be34b.jpg)

![](images/79f9aab00629d30388e71a87354e74f5e6a375143b3264ecedf83b2fd725a6b2.jpg)

![](images/49ad37c351d4098c2a34d86343cd380351a664b5628229b553af994f0220bf80.jpg)

![](images/26b2dd77dd48c72d6991f0cec05e239d98a9c135eb724feab79fbc486052133a.jpg)

![](images/0cce58cdad59e2bb677ab072435e215ad0d2fdf1ebff5c3b54840883f4ad9734.jpg)

Score for shopping mall 2 \*

1: Does not match at all, 2: Does not match, 3: Matches to an average degree, 4: Matches, 5: Matches very well

1

2

3

4

5

![](images/2505fdd0c77a6b1d5dc85def77730800b5d352e4c70ed9497ac6b23441b65ef4.jpg)

![](images/d8ec0be8f37c9c7d1837f6d006661e4bebe2bbb674bd91fa73a128af7e4cdbed.jpg)

![](images/41677dd6fd379e9fcf4dd17e7d6f95cf789be16283a880b4109a021473b4e6de.jpg)

![](images/3249e76c3e9bca06db3f55a42743ef576c47fab4ddab881a4634e30de01372d8.jpg)

![](images/4033ec14fd8099b175f9f59ff846a52caaf642395c67008df53c9fc6ef9847b9.jpg)

Preference Question \*

Which shopping mall's results do you prefer?

![](images/a1a8f4d79aa2994b7ba2a0437a10467e60d68dd1e451ea2533a44b0d0ca281a0.jpg)

Shopping Mall 1

![](images/564201c5188ed4b25318c713251435769c22c2526da8f409590bac740d508fbd.jpg)

Shopping Mall 2

(Optional) Irrelevance Check Question

![](images/d9cb040ff5d17323d447c110cc860244eeebbbb7847aa5f7981fd2d20b4a0115.jpg)

Two set of search results are irrelevant to original image or user text

Figure 11. Example of questions in user survey.