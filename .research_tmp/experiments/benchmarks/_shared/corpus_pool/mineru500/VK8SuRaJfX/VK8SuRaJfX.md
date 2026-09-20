# The Four Color Theorem for Cell Instance Segmentation

Ye Zhang $^{12}$ Yu Zhou $^{2}$ Yifeng Wang $^{3}$ Jun Xiao $^{1}$ Ziyue Wang $^{4}$ Yongbing Zhang $^{1}$ Jianxu Chen $^{2}$

# Abstract

Cell instance segmentation is critical to analyzing biomedical images, yet accurately distinguishing tightly touching cells remains a persistent challenge. Existing instance segmentation frameworks, including detection-based, contour-based, and distance mapping-based approaches, have made significant progress, but balancing model performance with computational efficiency remains an open problem. In this paper, we propose a novel cell instance segmentation method inspired by the four-color theorem. By conceptualizing cells as countries and tissues as oceans, we introduce a four-color encoding scheme that ensures adjacent instances receive distinct labels. This reformulation transforms instance segmentation into a constrained semantic segmentation problem with only four predicted classes, substantially simplifying the instance differentiation process. To solve the training instability caused by the non-uniqueness of four-color encoding, we design an asymptotic training strategy and encoding transformation method. Extensive experiments on various modes demonstrate our approach achieves state-of-the-art performance. The code is available at https://github.com/zhangye-zoe/FCIS.

# 1. Introduction

Cell-level analysis tasks hold broad application prospects in the biomedical field. Accurate cell segmentation (Petukhov et al., 2022; Zhang et al., 2025a; Chen et al., 2024) not only provides a necessary foundation for downstream tasks such as cell counting (Falk et al., 2019), cell classification

![](images/7d9f8433598df7a369fa995beb09bc58320aa62ec05b4d32f46a3c0a734f1858.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Microscope"] --> B["Detection"]
    B --> C["Non-maxima Suppression"]
    C --> D["Segmentation"]
    D --> E["GT Label"]
```
</details>

(a) Detection-based Method   
![](images/312d75ffb44a81528c2c1c435091f26fc93f3d8a354fdab93b42b50780bed151.jpg)

<details>
<summary>text_image</summary>

Pathology
true con.
Prob map
1.0
0.0
Probability
Threshold
Contour Prediction
p=0.5
p=0.6
</details>

(b) Contour Prediction Method

![](images/642fc6551a7b0a16be6f4fafa8ecf639bc988f92525c8a7227c588ac5b55da01.jpg)

<details>
<summary>text_image</summary>

Fluorescence Hor. distance
Post Processing
Edge gradient Prediction
</details>

(c) Distance Mapping Method   
Figure 1. Existing cell instance segmentation frameworks. (a) represents the detection-based method, which cannot tackle elongated cells. (b) represents the contour prediction methods influenced by threshold value choice. (c) represents the distance mapping methods, which contains multiple tasks and relies on a post-processing process. The Red “×” indicates the segmentation mistakes.

(Cords et al., 2023; Zhang et al., 2025b), and cell tracking (Merryweather et al., 2021), but also underpins critical applications in clinical diagnostics, such as immune microenvironment analysis (Barkley et al., 2022; Kao et al., 2022) and biomarker discovery (Mann et al., 2021).

At present, cell instance segmentation models can be categorized into three primary approaches: (a) detection-based methods (Jiang et al., 2023), which rely on object detection frameworks (Ren et al., 2016) to localize and delineate individual cells; (b) contour prediction methods (Chen et al., 2016), which explicitly predict cell boundaries to achieve instance differentiation; and (c) distance mapping methods (Graham et al., 2019; He et al., 2021), which encode spatial relationships or distance information to separate adjacent cells. Despite their demonstrated success in cell instance segmentation tasks, existing methods face critical limitations due to the inherent diversity of cell morphologies and image characteristics across different scenarios, which impose stringent demands on model generalization.

For different segmentation methods, their problems include the following aspects, as shown in Figure 1. First, detection-based methods often struggle with complex cases such as

![](images/b30475c456b14639c50d9407cf1857ae4cfa14b19ac52264458e64f32ee5d9ea.jpg)

<details>
<summary>text_image</summary>

Map
Cell Image
Non-unique Encoding
</details>

Figure 2. Our proposed cell encoding method is based on the four-color theorem. In the method, each cell is viewed as a “country,” and the encoding ensures that adjacent cells have different colors.

elongated fibroblasts or overlapping cells, leading to frequently missed detections. Second, contour prediction methods attempt to achieve instance separation by introducing an additional contour category. However, their performance heavily depends on the contour threshold setting. Lastly, distance mapping methods rely on highly intricate network architectures and sophisticated post-processing workflows, which significantly increase computational overhead and model complexity. Therefore, it is imperative to develop an innovative approach that can address the shortcomings of current methods by enhancing robustness, reducing computational complexity, and improving generalization across diverse biomedical image modes.

The four-color theorem (Fritsch et al., 1998) offers a novel perspective on cell instance segmentation. As shown map in Figure 2, the theorem states that only four colors are sufficient to ensure that adjacent regions are assigned distinct colors (Gonthier et al., 2008). By drawing an analogy to cell images, we conceptualize each cell instance as a “country,” while the background corresponds to the “ocean.” This enables the development of a four-color encoding scheme that assigns unique encodings to adjacent cells. Under this framework, the instance segmentation task is reformulated as a four-class semantic segmentation problem, simplifying instance differentiation. However, the inherent non-uniqueness of four-color encodings and the class imbalance introduced by the encoding strategy pose significant challenges for model training. Directly using these encodings as supervision can lead to training instability and hinder optimization. Therefore, a well-designed training strategy is essential to address the training problem effectively.

To address the above challenges, we propose an asymptotic training architecture. Different from traditional semantic segmentation methods (Wang et al., 2018; Zhou et al., 2022), which adopt simultaneous multi-class, multi-channel output strategies, our method adopts a step-by-step approach: first distinguish foreground and background and then make category prediction within the foreground region. Our approach prioritizes high-level spatial information over fine-grained semantic details by imposing orthogonal constraints on adjacent cells, effectively solving the class imbalance problem. To mitigate the training instability caused by the non-uniqueness of the encoding, we introduce an encoding transformation method that maps the output to a minimum color representation, ensuring consistency by removing ambiguity in encoding variations. In addition, we provide a theoretical analysis to prove the reasonability of the designs.

In summary, the contributions of this paper are five folds:

- We propose an innovative cell segmentation method based on the four-color theorem, which transforms instance segmentation into a semantic segmentation task, eliminating complex instance differentiation designs.   
- We design an asymptotic training strategy incorporating a foreground prediction transformation module, greatly enhancing training stability and robustness.   
- We systematically summarize the characteristics of cell distributions in medical images, demonstrating that cell coloring is inherently simpler than map coloring.   
- We provide a rigorous theoretical analysis to justify the rationale and feasibility of the proposed model design.   
- We validate the effectiveness of our method on three distinct types of medical image datasets. The results show that our method successfully balances performance and model complexity.

# 2. Complexity Analysis of Model Training

The advancement of deep learning revolutionizes automated cell segmentation, significantly reducing the time and effort required for manual annotation (Stringer et al., 2021; Pachitariu & Stringer, 2022; Zhang et al., 2025c). Although existing approaches show impressive performance, they are difficult to fit in various segmentation scenes and face challenges regarding training complexity and post-processing requirements. To facilitate a comprehensive comparison of existing methods, we summarize the computational complexity of the above three types of models. At the same time, the Supplementary Material provides more extensive research of related works.

# Detection-Based Methods

The overlapping boundaries of cells remain a critical challenge in cell segmentation. Detection-based methods address this issue through a two-stage strategy: first, a detection network (Ren et al., 2016; Redmon & Farhadi, 2017) localizes cell positions; then, segmentation predictions are generated based on the detection results. Representative methods include IRNet (Zhou et al., 2020), and DoNet (Jiang et al., 2023). To enhance localization accuracy, these methods commonly incorporate non-maximum suppression (NMS) to merge highly overlapping detection boxes, thereby reducing over-prediction. However, this strategy can result in missed detections, particularly for small or irregularly

shaped cells. Additionally, detection-based approaches often exhibit high computational complexity due to the intricate detection and segmentation network designs. As shown in Table 1, these methods typically have higher parameter complexity and computational overhead regarding FLOPs.

# Contour Prediction Methods

Contour prediction-based methods achieve instance differentiation by introducing contour semantic categories into the model's predictions. However, due to the few pixels in the contour, the segmentation for contours is often inferior to that for the background and foreground. To address this issue, two solutions are proposed. The first solution, represented by UNet (Ronneberger et al., 2015; Zhou et al., 2018), increases the loss weight of boundary to guide the model to focus more on contour. With relatively simple structural designs, these methods typically have lower parameter counts and computational complexity, as shown in Table 1. However, their performance remains suboptimal, constrained by the limited effectiveness of the loss weighting strategy. The second solution, represented by Micro-Net (Raza et al., 2019), enhances contextual perception by introducing complex network structures (Zhou et al., 2019), such as multi-scale feature fusion (Srivastava et al., 2021) and attention mechanisms (Prangemeier et al., 2020; Hörst et al., 2024). While this approach significantly improves performance compared to the former, including complex modules substantially increases model complexity, resulting in longer training times and higher computational costs.

# Distance Mapping Methods

Distance-based cell segmentation methods, such as StarDist (Schmidt et al., 2018), CellViT (Hörst et al., 2024), and RepSNet (Xiong et al., 2025), utilize distance maps to enhance instance differentiation, especially in cases involving irregular cell shapes or densely packed regions. While these methods have shown significant success, they typically rely on multiple decoding branches that require post-processing (Graham et al., 2019; Chen et al., 2023; Meng et al., 2024) to merge the results into accurate instance segmentations. This multi-branch design increases model complexity, as the network must simultaneously handle various tasks, including distance map prediction and semantic category classification. As shown in Table 1, distance-based methods generally exhibit higher parameter complexity and computational cost than detection-based and contour prediction methods, limiting their efficiency for large-scale applications.

The four-color-theorem introduces a novel cell instance segmentation paradigm that eliminates the dedicated instance differentiation modules. This method significantly reduces training complexity by reformulating the instance segmentation task as a semantic segmentation problem. Furthermore, experimental results in Table 1 demonstrate that this approach achieves substantial advantages in both parameter efficiency and computational cost compared to detection-based and distance-based segmentation methods.

<table><tr><td>Methods</td><td># Paras</td><td>#FLOPs</td><td>Publication</td></tr><tr><td colspan="4">Detection based methods</td></tr><tr><td>Mask-RCNN (He et al., 2017)</td><td>44.66 M</td><td>411.61 G</td><td>ICCV</td></tr><tr><td>DoNet (Jiang et al., 2023)</td><td>67.71</td><td>221.64 G</td><td>CVPR</td></tr><tr><td colspan="4">Contour prediction methods</td></tr><tr><td>UNet (Ronneberger et al., 2015)</td><td>32.14 M</td><td>64.27 G</td><td>MICCAI</td></tr><tr><td>DCAN (Chen et al., 2016)</td><td>41.16 M</td><td>77.82 G</td><td>CVPR</td></tr><tr><td>CNN3 (Kumar et al., 2017)</td><td>65.46 M</td><td>1.06 G</td><td>TMI</td></tr><tr><td>UNet++ (Zhou et al., 2018)</td><td>9.28 M</td><td>35.61 G</td><td>MICCAI</td></tr><tr><td>FullNet (Qu et al., 2019)</td><td>112.60 M</td><td>116.92 G</td><td>MICCAI</td></tr><tr><td>Micro-Net (Raza et al., 2019)</td><td>89.64 M</td><td>72.96 G</td><td>MIA</td></tr><tr><td>NucleiSegNet (Lal et al., 2021)</td><td>12.40 M</td><td>18.19 G</td><td>CBM</td></tr><tr><td>TSFD-Net (Ilyas et al., 2022)</td><td>21.96 M</td><td>12.10 G</td><td>NN</td></tr><tr><td>GeNSeg-Net (Xu et al., 2024)</td><td>87.11 M</td><td>86.84 G</td><td>MM</td></tr><tr><td colspan="4">Distance mapping methods</td></tr><tr><td>StarDist (Schmidt et al., 2018)</td><td>21.43 M</td><td>92.40 G</td><td>MICCAI</td></tr><tr><td>HoverNet(Graham et al., 2019)</td><td>49.70 M</td><td>192.70 G</td><td>MIA</td></tr><tr><td>CDNet (He et al., 2021)</td><td>70.55 M</td><td>44.87 G</td><td>ICCV</td></tr><tr><td>SONNET (Doan et al., 2022)</td><td>63.87 M</td><td>166.75 G</td><td>JBHI</td></tr><tr><td>TransUNet (He et al., 2023)</td><td>112.21 M</td><td>37.67 G</td><td>MICCAI</td></tr><tr><td>CPP-Net (Chen et al., 2023)</td><td>80.75 M</td><td>163.10 G</td><td>TIP</td></tr><tr><td>SMILE (Pan et al., 2023)</td><td>53.85 M</td><td>68.58 G</td><td>MIA</td></tr><tr><td>NuSEA (Meng et al., 2024)</td><td>55.26 M</td><td>74.20 G</td><td>JBHI</td></tr><tr><td>CellViT (Hörst et al., 2024)</td><td>96.81 M</td><td>124.25 G</td><td>MIA</td></tr><tr><td>RepSNet (Xiong et al., 2025)</td><td>28.20 M</td><td>137.11 G</td><td>IJCV</td></tr><tr><td colspan="4">Our four-color theorem based method</td></tr><tr><td>FCIS (Ours)</td><td>39.75 M</td><td>58.03 G</td><td>-</td></tr></table>

Table 1. The computational complexity and number of parameters comparisons. All the methods are reported for $256 \times 256$ inputs.

# 3. Cell Encoding by Four Color Theorem

# 3.1. Greedy Algorithm for Encoding

The four-color theorem illustrates the minimum color number required to label adjacent regions without overlap, providing a novel approach to the cell instance segmentation problem. Unlike traditional instance segmentation workflows, this method transforms the instance segmentation task into a multi-class semantic segmentation problem. Based on this theory, we designed a greedy algorithm to generate four-class encoded representations for the foreground regions. The encoding process is illustrated in Algorithm 1.

We first preprocess the input image and its corresponding labels $(X,Y)$ to construct a cell graph $G=(V,E)$ , where the node set $V=\{v_{i} \mid i=1,\cdots,N\}$ represents the cells in the image, and the edge set $E=\{e_{i,j}\}$ represents the adjacency relationships between cells. The label Y contains instance-level annotations, with each instance uniquely identified by an identification, while $e_{i,j}$ indicates that cells $v_{i}$ and $v_{j}$ are adjacent. Next, we assign color encodings to

![](images/4b7b4a550c4db54e331cdb0854312fe1a4fa560312d58e7e6ec71e733c7d08d4.jpg)  
Figure 3. The non-uniqueness of four color encoding can be summarized as the following three cases: (a) Encoding substitution; (b) Encoding exchange, and (c) Encoding rule modification.

the nodes $v \in V$ using a greedy algorithm. Specifically, for each node v, we first compute its set of neighboring nodes $N(v) = \{u \mid (v, u) \in E\}$ and collect the colors $C_{used}$ already assigned to these neighboring nodes as follows:

$$
\mathcal {C} _ {\text { used }} = \{C (u) \mid u \in N (v), C (u) \neq 0 \}. \tag {1}
$$

We assign the smallest available color from the four color set $C = \{1, 2, 3, 4\}$ to the current node v, ensuring that it does not conflict with the colors of its neighboring nodes:

$$
C (v) = \min (\mathcal {C} \setminus \mathcal {C} _ {\text { used }}). \tag {2}
$$

This process guarantees that two adjacent nodes $v_{i}$ and $v_{j}$ are assigned different colors, i.e., $C(v_{i}) \neq C(v_{j})$ . After encoding all cells, we generate the final segmentation mask M. For each pixel p in the image, if the pixel belongs to a specific nucleus v, it is assigned the color encoding $C(v)$ corresponding to that nucleus. The final output segmentation mask M provides a four class semantic segmentation representation based on the four color theorem.

# 3.2. Non-uniqueness Property of Encoding

While four-color encoding ensures heterogeneity of adjacent cell colors, its non-uniqueness may lead to convergence issues during model training. To illustrate the potential problems of this encoding more intuitively, Figure 3 presents the differences in cell encoding under various distribution scenarios. The first row shows the spatial relationships between cells, and the second row represents the corresponding cell graph structures, which cover the most common cell distribution scenarios. Combinations of these basic patterns can represent more complex cell distributions. Moreover, the third row illustrates encoding representations. In detail, the cells marked in red represent the initial encoding generated by the greedy algorithm. In contrast, the cells marked in black correspond to the equivalent encoding that satisfies the four-color theorem. The differences between the greedy algorithm's results and the equivalent encoding contain the following three cases:

(a) Substitution: The color of a cell is replaced with another color while maintaining the same number of colors.   
(b) Exchange: The color assignments between two or more cells are swapped, preserving the overall color count.   
(c) Rule Modification: Certain cells are assigned new colors, resulting in an increase in the total number of colors.

# Algorithm 1 Cell Encoding by Greedy Algorithm

1: Input: Cell graph $G = (V, E)$ , where $V$ is the set of cells and $E$ represents adjacency relation.

2: Output: Four-color encoded mask $C(v)$ .

3: Initialize color set $\mathcal{C} = \{1,2,3,4\}$ .

4: Initialize mask $C(v) \leftarrow 0, \forall v \in V$ .

5: for each nucleus $v \in V$ do

6: Get the set of neighbors: $N(v) = \{u \mid (v, u) \in E\}$ .

7: Collect used colors: $C_{used} = \{C(u) \mid u \in N(v), C(u) \neq 0\}$ .

8: Assign the smallest available color:

9: $C(v) \leftarrow \min(\mathcal{C} \setminus \mathcal{C}_{\text{used}}).$

10: end for

11: Generate segmentation mask M:   
12: for each pixel p in the image do   
13: Assign pixel p to the color of the cell it belongs to:   
14: $M(p) \leftarrow C(v)$ , where v is the cell containing p.   
15: end for   
16: Return M (Four-color annotation mask)

Segmentation networks typically learn semantic categories based on the object's morphological and textural features (Jain et al., 2023), while four-color encoding emphasizes the spatial relationships between cells. When faced with the issue of encoding non-uniqueness, conventional networks often struggle to converge stably (Ronneberger et al., 2015). To propose a reasonable solution, we further analyze the characteristics of the greedy algorithm in the next section.

# 3.3. Low-rank Property of Greedy Encoding

Greedy algorithms (GAs), as heuristic methods, are commonly used to generate locally optimal solutions. However, in the cell coloring problem, GAs can achieve globally optimal solutions. The reasons for this are twofold: First, the cells usually exhibit global dispersion and local aggregation, with a relatively small number of cells in each cluster, which differs from the distribution of countries. Second, adjacent cells in the image usually follow a chain-like or rectangular arrangement. Hence, each cell has much fewer neighbors. These structural properties render the cell coloring problem more straightforward than the map coloring problem.

To clarify the above fact, we statistics the number of colors distributed on each image under four-color encoding in Figure 4. The scatter plot on the left corresponds to each sample, and the box plot shows the distribution of encoding

![](images/d51dfaa09ed4c1eef3fced298aec0a9bede9a755330f65ad1d17895274b874fd.jpg)

<details>
<summary>boxplot</summary>

| enc  | Value |
| ---- | ----- |
| enc=1 | 60    |
| enc=2 | 20    |
| enc=3 | 5     |
</details>

(a) DSB2018

![](images/7277b18f75675451f28eaf2d620ff9b53a5b9d2e784b6ff83eef6dee3cddf864.jpg)

<details>
<summary>scatter</summary>

| enc  | Value |
| ---- | ----- |
| enc=1 | 80    |
| enc=2 | 60    |
| enc=3 | 40    |
</details>

(b) PanNuke   
Figure 4. Statistics of the number of cells with different color in the DSB2018 and PanNuke datasets.

numbers. The results demonstrate that only a tiny proportion of images require more than two colors and almost no image requires four colors. Based on these, we will present the global optimal theory of greedy algorithm coloring.

Theorem 1. Global Optimality of Greedy Coloring: Let $G = (V, E)$ be an undirected graph; among them, $V$ is the set of vertices, and $E$ is the set of edges. Suppose $G$ satisfies the following conditions:

(1) $G$ is planar, meaning it can be embedded in the plane without any edges crossing each other.   
(2) The maximum degree of $G$ , denoted $\Delta(G)$ , satisfies:

$$
\Delta (G) \leq k, \quad \text { where } \quad k \leq 4. \tag {3}
$$

(3) The vertex distribution of G follows a specific structure, either a chain structure (vertices are ordered linearly) or a rectangular structure (vertices are arranged in a grid pattern).

Then, the chromatic number with the greedy algorithm is equal to the chromatic number:

$$
\chi_ {\text { greedy }} (G) = \chi (G). \tag {4}
$$

Where $\chi(G)$ denote the chromatic number, which is the minimum number of colors, and $\chi_{\text{greedy}}(G)$ denote the chromatic number obtained by applying the greedy coloring algorithm. Some related definitions and proofs are included in the Supplementary Material.

# 4. Method Designs

# 4.1. Asymptotic Training Strategy

Previous research primarily focused on designing powerful feature extractors (Liu et al., 2022; Yu et al., 2024) or context-aware modules (Liu et al., 2021; Li et al., 2024) to enhance the network classification ability. However, in the scenario of four-color encoding, the model not only requires learning semantic features to distinguish foreground and background but also needs to learn positional information, ensuring adjacent cells are assigned distinct colors. To address the dual requirements, we propose an asymptotic training strategy as illustrated in Figure 5 (a).

# Binary Classification Semantic Prediction

Given an input image $X_{i}$ , an encoder-decoder network is employed to generate a five-channel feature map $\hat{Y}_{i} \in R^{H \times W \times 5}$ , where H and W are the height and width of input. Among these channels, the first represents the background probability, and the remaining four represent the prediction of the four-color encoding. Hence, the probability map of background $\hat{Y}_{b}$ is extracted as follows:

$$
\hat {Y} _ {b} = \hat {Y} _ {i} [:, 0 ], \tag {5}
$$

where $\hat{Y}_{i}[:,0]$ denotes the first channel of the prediction feature map. For obtaining the foreground probability, we use a convolution operation to transform the last four channels into a single-channel foreground probability:

$$
\hat {Y} _ {f} = \operatorname{Conv} \left(\hat {Y} _ {i} [:, 1: 5 ]\right), \tag {6}
$$

where $\operatorname{Conv}(\cdot)$ represents convolutional layers. Combined the probability maps of background $\hat{Y}_{b}$ and foreground $\hat{Y}_{f}$ , the binary semantic prediction can be formulated as:

$$
\hat {Y} _ {b, i} = \text { Concat } (\hat {Y} _ {b}, \hat {Y} _ {f}), \tag {7}
$$

where $\text{Concat}(\cdot,\cdot)$ denotes the concatenation operation along the channel dimension. To optimize the binary semantic predictions, we define the semantic loss as:

$$
\mathcal {L} _ {\text { sem }} = \mathrm{CE} (\hat {Y} _ {b, i}, Y _ {i}) + \mathrm{Dice} (\hat {Y} _ {b, i}, Y _ {i}), \tag {8}
$$

where $\mathrm{CE}(\cdot,\cdot)$ represents the cross-entropy loss, and $\mathrm{Dice}(\cdot,\cdot)$ is the Dice coefficient loss. Where $Y_{i}$ denotes the ground truth labels for binary segmentation.

# Four-Color Category Prediction

To accurately identify foreground regions and ensure distinct encodings for adjacent cells, we propose a negative sampling constraint method, as shown in Figure 5 (b). This method enforces heterogeneity for adjacent cells while preserving the accuracy of four-color encoding.

First, based on cell connectivity relationships, we sample features from the adjacent cell pairs $(v_{i}, v_{j})$ . Meantime, the sampled feature sets can be formulated as follows:

$$
F _ {i} = \{f _ {i} ^ {\alpha} \mid \alpha = 1, \dots , M \}, \tag {9}
$$

$$
F _ {j} = \{f _ {j} ^ {\beta} \mid \beta = 1, \dots , N \}, \tag {10}
$$

where M and N are the number of sampling obtained from cells $v_{i}$ and $v_{j}$ , respectively, and $f_{i}^{\alpha}$ denotes the feature vector of the $\alpha$ -th pixel in cell $v_{i}$ .

To ensure that the feature representations of adjacent cells exhibit sufficient heterogeneity, we impose an orthogonality constraint in the feature space. This constraint is formulated using a cosine similarity loss:

$$
\mathcal {L} _ {\text { ort }} = \frac {1}{| E |} \sum_ {(v _ {i}, v _ {j}) \in E} \mathrm{Cos} (F _ {i}, F _ {j}), \tag {11}
$$

![](images/f56881b42f43bf7384393fa0aba87e5fdfc36c69f5ce9a40e0df7de5613a1f63.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Xi"] --> B["Seg. Network"]
    B --> C["Output Ŷi"]
    C --> D["FC Pred"]
    C --> E["FC GT"]
    C --> F["Sem. Pred"]
    D --> G["Yf"]
    E --> H["Yi"]
    F --> I["Yb,i"]
    F --> J["Ysem"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fff,stroke:#333
    style H fill:#fff,stroke:#333
    style I fill:#fff,stroke:#333
    style J fill:#fff,stroke:#333
```
</details>

Seg. : Segmentation   
Trans.: Transformation   
Sem. : Semantic   
FC: Four-color   
Samp. : Sampling   
0: $0^{st}$ channel   
■ : Sampled feature   
Figure 5. The training framework of proposed FCIS. (a) represents the asymptotic training method, (b) represents the negative sampling learning for adjacent cells, and (c) represents the encoding transformation method.

where $\mathrm{Cos}(\cdot, \cdot)$ denotes the cosine similarity function, $E$ is the set of all edges representing adjacent cell pairs, and $|E|$ is the total number of edges. By minimizing $\mathcal{L}_{\mathrm{ort}}$ , the similarity of feature representations is suppressed, thereby enhancing the model's ability to distinguish cells.

# 4.2. Encoding Transformation

Although the orthogonality constraint ensures heterogeneous encoding, this sampling-based supervision remains weak and may not effectively guide model training. To provide stronger supervision, we introduce four-color encoding as the target label. However, the non-uniqueness of the encoding can lead to inconsistencies in supervision, potentially hindering model convergence. To mitigate this issue, we propose an encoding transformation method, whose mechanism is established in the following theorem.

Theorem 2. Greedy Coloring Compatibility: In the cell instance segmentation task, let the encoding matrix generated by the greedy algorithm be:

$$
\mathbf {C} \in \mathbb {R} ^ {n \times k}, (k \leq 4). \tag {12}
$$

And the encoding matrix predicted by the network is:

$$
\mathbf {P} \in \mathbb {R} ^ {n \times k ^ {\prime}}, \tag {13}
$$

n represents the number of cells, k is the number of colors used in the greedy algorithm, and $k'$ is the number of predicted encodings.

If the predicted encoding matrix $\mathbf{P}$ has one of the relations with the greedy encoding $\mathbf{C}$ , i.e., substitution, exchange, rule modification. Then there exists a mapping function: $f: \mathbf{P} \to \mathbf{C}$ , such that the network's predicted result can be transformed into the four-color encoding result. The detailed proof is shown in Supplementary Material.

Based on the above theory, we propose an encoding transformation method consisting of two convolutional layers, which maps the network's predicted output $\hat{Y}_f$ into the optimal encoding $\hat{Y}_t$ , as shown in Figure 5 (c). This transformation ensures adherence to the four-color encoding rules, improving the model's overall performance and accelerating its convergence during training. Employing the transformed prediction, we compute a classification loss specific to the foreground as follows:

$$
\mathcal {L} _ {\mathrm{cls}} = \mathrm{CE} (\hat {Y} _ {t}, Y _ {f}) + \mathrm{Dice} (\hat {Y} _ {t}, Y _ {f}), \tag {14}
$$

Where $\hat{Y}_{t}$ and $Y_{f}$ represent the predicted and ground truth foreground regions, respectively. In the optimization objective, we only calculate the loss of the foreground region.

# Total Loss Function

The overall loss function integrates the semantic, orthogonality, and classification losses and is formulated as follows:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { sem }} + \lambda_ {1} \mathcal {L} _ {\text { ort }} + \lambda_ {2} \mathcal {L} _ {\text { cls }}. \tag {15}
$$

where $\lambda_{1}$ and $\lambda_{2}$ are hyperparameters that control the importance of the orthogonality and classification losses. In the paper, we set $\lambda_{1}=2$ and $\lambda_{2}=1$ . More experiment comparisons are added in Supplementary Material.

# 5. Experiments

# 5.1. Datasets

We evaluated our proposed method on multiple types of cell images, including pathological images, fluorescence-stained images, bright-field images and phase-contrast images. Specifically, the datasets used include BBBC006v1 (Ljosa et al., 2012), DSB2018 (Caicedo et al., 2019), Pan-Nuke (Gamper et al., 2020) and YeaZ (Dietler et al., 2020).

The BBBC006v1 consists of 768 Hoechst 33342 marker-stained images, each with a resolution of $696 \times 520$ pixels. Following the dataset split used by CPP-Net (Chen et al., 2023), we divide the dataset into 462 training, 153 validation, and 153 testing images.

The DSB2018 source from the Data Science Bowl 2018 competition, contains 670 fluorescence-stained images with resolutions ranging from $256 \times 256$ to $520 \times 696$ pixels using DAPI and Hoechst stains. We split the dataset into 380 training, 67 validation, and 50 testing images.

![](images/4ca9589e66fa4ffb530b262291adff9f9d9500aa49000eee49dd591a7e66eab9.jpg)

<details>
<summary>text_image</summary>

Input HoverNet CPP-Net GeNSeg-Net FC Pred (ours) FC GT Inst Pred (ours) Inst GT
</details>

Figure 6. The visualization comparisons between different methods.

<table><tr><td rowspan="2">Methods</td><td colspan="5">Metrics</td></tr><tr><td>DICE (↑)</td><td>AJI (↑)</td><td>DQ (↑)</td><td>SQ (↑)</td><td>PQ (↑)</td></tr><tr><td>DCAN (Chen et al., 2016)</td><td>0.795</td><td>0.676</td><td>0.743</td><td>0.780</td><td>0.626</td></tr><tr><td>HoverNet (Graham et al., 2019)</td><td>0.898</td><td>0.762</td><td>0.863</td><td>0.877</td><td>0.762</td></tr><tr><td>NucleiSegNet (Lal et al., 2021)</td><td>0.904</td><td>0.671</td><td>0.784</td><td>0.843</td><td>0.682</td></tr><tr><td>DoNet (Jiang et al., 2023)</td><td>0.823</td><td>0.716</td><td>0.787</td><td>0.829</td><td>0.673</td></tr><tr><td>CPP-Net (Chen et al., 2023)</td><td>0.914</td><td>0.813</td><td>0.866</td><td>0.879</td><td>0.758</td></tr><tr><td>GeNSeg (Xu et al., 2024)</td><td>0.856</td><td>0.781</td><td>0.843</td><td>0.791</td><td>0.759</td></tr><tr><td>Un-SAM (Chen et al., 2025)</td><td>0.902</td><td>0.786</td><td>0.826</td><td>0.834</td><td>0.747</td></tr><tr><td>CellPose (Stringer, 2025)</td><td>0.923</td><td>0.824</td><td>0.862</td><td>0.871</td><td>0.764</td></tr><tr><td>FCIS (Ours)</td><td>0.939</td><td>0.828</td><td>0.875</td><td>0.878</td><td>0.770</td></tr></table>

Table 2. The comparison performances on DSB2018 dataset.

<table><tr><td rowspan="2">Methods</td><td colspan="5">Metrics</td></tr><tr><td>DICE (↑)</td><td>AJI (↑)</td><td>DQ (↑)</td><td>SQ (↑)</td><td>PQ (↑)</td></tr><tr><td>DCAN (Chen et al., 2016)</td><td>0.778</td><td>0.587</td><td>0.659</td><td>0.721</td><td>0.506</td></tr><tr><td>HoverNet (Graham et al., 2019)</td><td>0.798</td><td>0.646</td><td>0.718</td><td>0.782</td><td>0.595</td></tr><tr><td>NucleiSegNet (Lal et al., 2021)</td><td>0.752</td><td>0.544</td><td>0.618</td><td>0.689</td><td>0.457</td></tr><tr><td>DoNet (Jiang et al., 2023)</td><td>0.781</td><td>0.612</td><td>0.684</td><td>0.750</td><td>0.544</td></tr><tr><td>CPP-Net (Chen et al., 2023)</td><td>0.814</td><td>0.638</td><td>0.711</td><td>0.776</td><td>0.583</td></tr><tr><td>Un-SAM (Chen et al., 2025)</td><td>0.801</td><td>0.629</td><td>0.704</td><td>0.767</td><td>0.570</td></tr><tr><td>CellPose (Stringer, 2025)</td><td>0.787</td><td>0.626</td><td>0.703</td><td>0.764</td><td>0.591</td></tr><tr><td>FCIS (Ours)</td><td>0.816</td><td>0.653</td><td>0.721</td><td>0.796</td><td>0.610</td></tr></table>

Table 3. The comparison performances on PanNuke dataset.

The PanNuke dataset includes 7901 H&E-stained images, each 256×256 pixels, originating from 19 organs, with a total of 189,744 annotated nuclei. We divide this dataset into 2656 training, 2523 validation, and 2722 testing images.

The YeaZ comprises 306 bright-field (BF) images with resolutions ranging from $301 \times 301$ to $1463 \times 1311$ pixels, and 43 phase-contrast (PC) images with resolutions ranging from $256 \times 256$ to $1988 \times 2000$ pixels. Due to the limited number of PC images, we merge the BF and PC datasets to train a unified model, resulting in 300 training, 20 validation, and 29 testing images.

# 5.2. Implementation Details and Evaluation Metrics

Our all experiments are conducted using PyTorch on an NVIDIA A100 GPU. We employ stochastic gradient descent (SGD) as the optimizer, with a learning rate of 0.01,

<table><tr><td rowspan="2">Methods</td><td colspan="5">Metrics</td></tr><tr><td>DICE (↑)</td><td>AJI (↑)</td><td>DQ (↑)</td><td>SQ (↑)</td><td>PQ (↑)</td></tr><tr><td>DCAN (Chen et al., 2016)</td><td>0.921</td><td>0.816</td><td>0.875</td><td>0.850</td><td>0.773</td></tr><tr><td>HoverNet (Graham et al., 2019)</td><td>0.941</td><td>0.891</td><td>0.924</td><td>0.911</td><td>0.856</td></tr><tr><td>NucleiSegNet (Lal et al., 2021)</td><td>0.939</td><td>0.671</td><td>0.809</td><td>0.844</td><td>0.719</td></tr><tr><td>DoNet (Jiang et al., 2023)</td><td>0.933</td><td>0.836</td><td>0.882</td><td>0.871</td><td>0.794</td></tr><tr><td>CPP-Net (Chen et al., 2023)</td><td>0.944</td><td>0.914</td><td>0.917</td><td>0.914</td><td>0.898</td></tr><tr><td>GeNSeg-Net (Xu et al., 2024)</td><td>0.934</td><td>0.907</td><td>0.913</td><td>0.911</td><td>0.915</td></tr><tr><td>Un-SAM (Chen et al., 2025)</td><td>0.933</td><td>0.912</td><td>0.909</td><td>0.911</td><td>0.904</td></tr><tr><td>CellPose (Stringer, 2025)</td><td>0.949</td><td>0.917</td><td>0.912</td><td>0.922</td><td>0.914</td></tr><tr><td>FCIS (Ours)</td><td>0.954</td><td>0.921</td><td>0.926</td><td>0.945</td><td>0.935</td></tr></table>

Table 4. The comparison performances on BBBC006v1 dataset.

<table><tr><td rowspan="2">Methods</td><td colspan="5">Metrics</td></tr><tr><td>DICE (↑)</td><td>AJI (↑)</td><td>DQ (↑)</td><td>SQ (↑)</td><td>PQ (↑)</td></tr><tr><td>DCAN (Chen et al., 2016)</td><td>0.881</td><td>0.772</td><td>0.571</td><td>0.736</td><td>0.446</td></tr><tr><td>HoverNet (Graham et al., 2019)</td><td>0.907</td><td>0.814</td><td> $\underline{0.602}$ </td><td>0.739</td><td>0.445</td></tr><tr><td>NucleiSegNet (Lal et al., 2021)</td><td>0.874</td><td>0.788</td><td>0.583</td><td>0.734</td><td>0.439</td></tr><tr><td>DoNet (Jiang et al., 2023)</td><td>0.878</td><td>0.754</td><td>0.577</td><td>0.720</td><td>0.431</td></tr><tr><td>GeNSeg-Net (Xu et al., 2024)</td><td>0.869</td><td>0.747</td><td>0.572</td><td>0.722</td><td>0.433</td></tr><tr><td>Un-SAM (Chen et al., 2025)</td><td>0.904</td><td>0.808</td><td>0.597</td><td>0.734</td><td>0.442</td></tr><tr><td>CellPose (Stringer, 2025)</td><td> $\underline{0.911}$ </td><td> $\underline{0.823}$ </td><td> $\underline{0.609}$ </td><td> $\underline{0.740}$ </td><td> $\underline{0.451}$ </td></tr><tr><td>FCIS (Ours)</td><td> $\underline{0.922}$ </td><td> $\underline{0.819}$ </td><td>0.599</td><td> $\underline{0.741}$ </td><td> $\underline{0.456}$ </td></tr></table>

Table 5. The comparison performances on YeaZ dataset.

momentum of 0.9, and weight decay of 0.0005. The network is trained for 200 epochs. Segmentation performance is evaluated using the DICE coefficient, Aggregated Jaccard Index (AJI) (Kumar et al., 2017), Detection Quality (DQ) (Kirillov et al., 2019), Segmentation Quality (SQ), and Panoptic Quality (PQ) metrics. In all tables presented in this paper, the highest performance scores are highlighted in bold, while the second-best scores are underlined.

# 5.3. Main Experiments

We evaluate the performance of our proposed method against eight state-of-the-art models across three benchmark datasets. The compared methods include the detection-based DoNet (Jiang et al., 2023); contour prediction-based approaches such as DCAN (Chen et al., 2016), NucleiSeg-Net (Lal et al., 2021), and GeSegNet (Xu et al., 2024); distance mapping-based methods including HoverNet (Gra

<table><tr><td rowspan="2" colspan="3">Settings</td><td colspan="5">DSB2018</td><td rowspan="2" colspan="3">Settings</td><td colspan="5">PanNuke</td></tr><tr><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td></tr><tr><td colspan="3">Baseline</td><td>0.876</td><td>0.751</td><td>0.847</td><td>0.856</td><td>0.746</td><td colspan="3">Baseline</td><td>0.786</td><td>0.627</td><td>0.705</td><td>0.778</td><td>0.580</td></tr><tr><td colspan="3">w. Four-color</td><td>0.843(-3.3)</td><td>0.725(-2.6)</td><td>0.805(-4.2)</td><td>0.827(-2.9)</td><td>0.674(-7.2)</td><td colspan="3">w. Four-color</td><td>0.766(-2.0)</td><td>0.617(-1.0)</td><td>0.686(-1.9)</td><td>0.752(-2.6)</td><td>0.559(-2.1)</td></tr><tr><td>Asymp.</td><td>Trans.</td><td>Samp.</td><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td><td>Asymp.</td><td>Trans.</td><td>Samp.</td><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td></tr><tr><td>√</td><td></td><td></td><td>0.862</td><td>0.740</td><td>0.812</td><td>0.831</td><td>0.679</td><td>√</td><td></td><td></td><td>0.773</td><td>0.624</td><td>0.691</td><td>0.763</td><td>0.565</td></tr><tr><td>√</td><td>√</td><td></td><td>0.883</td><td>0.756</td><td>0.829</td><td>0.844</td><td>0.701</td><td>√</td><td>√</td><td></td><td>0.787</td><td>0.630</td><td>0.710</td><td>0.774</td><td>0.572</td></tr><tr><td></td><td></td><td>√</td><td>0.910</td><td>0.785</td><td>0.846</td><td>0.863</td><td>0.741</td><td></td><td></td><td>√</td><td>0.803</td><td>0.642</td><td>0.714</td><td>0.776</td><td>0.598</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.939</td><td>0.828</td><td>0.875</td><td>0.878</td><td>0.770</td><td>√</td><td>√</td><td>√</td><td>0.816</td><td>0.653</td><td>0.721</td><td>0.796</td><td>0.610</td></tr></table>

Table 6. Ablation studies on the DSB2018 and PanNuke datasets. Baseline denotes the binary semantic segmentation model based on U-Net (Ronneberger et al., 2015). w. Four-color increases the number of channels from two to five by directly employing four-color encoding in Algorithm 1 as ground truth. Asymp. represents the asymptotic training method, Trans. applies the encoding transformation method, and Samp. introduces a negative sampling constraint for adjacent cells.

![](images/2095487bf817b243336139c716636e5d444f2191de94cf240829512f975c46a7.jpg)

<details>
<summary>line</summary>

| Step | w/o Trans | w Trans |
| ---- | --------- | ------- |
| 0    | 10.0      | 7.0     |
| 2000 | 3.0       | 2.5     |
| 4000 | 2.5       | 2.0     |
| 6000 | 2.0       | 1.8     |
</details>

![](images/e818f44e17ba09df7806caaca3b75e472ce26eee3eea5e0517b307702aa565d9.jpg)

<details>
<summary>line</summary>

| Step | w/o Trans | w Trans |
| ---- | --------- | ------- |
| 0    | 55        | 55      |
| 1000 | 75        | 75      |
| 2000 | 78        | 79      |
| 3000 | 79        | 80      |
| 4000 | 80        | 81      |
| 5000 | 80        | 81      |
| 6000 | 80        | 81      |
| 7000 | 80        | 81      |
</details>

Figure 7. Convergence analysis of the training loss and AJI on validation set before and after applying the encoding transformation.

ham et al., 2019), CPP-Net (Chen et al., 2023), and CellPose (Stringer, 2025); as well as SAM-based foundation model Un-SAM (Chen et al., 2025). Quantitative results are summarized in Tables 2–5. It is worth noting that GeSegNet, which was not designed for pathological image segmentation and performs poorly on the PanNuke dataset, is excluded from comparisons on that dataset.

From the results, we can see that FCIS consistently outperforms existing methods across all datasets. It achieves the highest DICE and AJI scores, demonstrating superior segmentation accuracy and instance-level consistency. In the DSB2018 dataset, our model achieves a DICE score of 0.939, surpassing Un-SAM and CellPose, among the best-performing prior methods. The PQ metric of 0.770 further indicates our model's ability to maintain segmentation quality and object-level distinction. In BBBC006v1, we can observe similar trends. While segmenting in the more challenging PanNuke, FCIS achieves 0.610 on the PQ, outperforming all previous methods and confirming its generalization capabilities. Although HoverNet achieves comparable performance to our method but incurs significantly higher parameter counts and computational complexity, as shown in Table 1. Therefore, considering the trade-off between model performance and computational cost, our method demonstrates a more pronounced overall advantage.

Furthermore, we visually compare different models in Figure 6. First, the results from “FC Pred” demonstrate that our method strictly adheres to the four-color encoding rule,

![](images/3031d8a1325080f8a86643d5eb3ac776748e7e7c025fa693a0036cd804d63fea.jpg)  
Figure 8. The visualization comparisons between different settings. The blue box indicates that the binary semantic prediction cannot distinguish adjacent cells. The red boxes indicate the four-color encoding lacks effective supervision for adjacent cells. The white boxes indicate our FCIS encodes adjacent cells with distinct colors.

ensuring that adjacent cells are assigned distinct colors, enhancing instance differentiation. By comparing the cell morphologies produced by different methods, we also observe that our segmentation results align more closely with the ground truth. This improvement can be attributed to incorporating the negative sampling learning method, effectively enhancing the boundary delineation. Additionally, due to page constraints, more visualization results are provided in the Supplementary Material.

# 5.4. Ablation Studies

# Effectiveness Analysis of the Method Designs

We conduct an ablation study to evaluate the module's performance under various configurations. The ablation methods include employing an asymptotic training strategy, applying encoding transformations to the network's predictions, and introducing a sampling constraint for adjacent cells. The experimental results are presented in Table 6. From the table, we observe the following: (1) When using the four-color encoding as supervision, the model performance decreases significantly, indicating the inherent

challenges of directly employing this encoding as a training signal. (2) Adding the asymptotic training strategy or encoding transformation methods leads to slight performance improvements, suggesting that these techniques provide some regularization benefits to the learning process. (3) Introducing the sampling constraint for adjacent cells results in a substantial performance boost, highlighting the effectiveness of this design in enforcing spatial consistency among predictions. These findings demonstrate that the proposed designs contribute positively to model performance.

# Analysis of Training Convergence

We analyze the model's convergence behavior by comparing the training loss and the validation AJI before and after applying the encoding transformation, as illustrated in Figure 7. The results indicate that incorporating the encoding transformation accelerates the convergence of the training loss, leading to faster stabilization with lower loss. Additionally, the AJI metric shows a significant improvement after applying the transformation, demonstrating the effectiveness of this design in enhancing model performance.

# Visualization of Different Settings

Based on the experimental results in Table 6, we conduct the visualization comparisons as shown in Figure 8. The red annotations represent the binary semantic segmentation ground truth (GT), the four-color encoding GT, and the instance segmentation GT, respectively. First, from the baseline results (b), it is evident that using only dual-channel predictions fails to distinguish adjacent cells effectively. Second, when directly using four-color encoding (b) as a supervision, the model lacks awareness of encoding inconsistency for adjacent cells, resulting in not only indistinguishable instances, but also fragmented predictions. By incorporating the asymptotic training (c) strategy, these issues are partially alleviated; however, distinguishing adjacent cells remains challenging. In contrast, our proposed method (f) demonstrates that the predicted results ensure not only that adjacent cells are encoded with different colors but also that the structural integrity of each instance.

# 6. Conclusions

We present a novel approach to cell instance segmentation by leveraging the four-color theorem, which reformulates the instance segmentation problem as a four-class semantic segmentation task. This transformation significantly reduces computational overhead and simplifies model design. To address challenges arising from the non-uniqueness of color encodings, we propose an asymptotic training strategy and an encoding transformation mechanism that ensure stable optimization. Extensive experiments on diverse biomedical imaging modalities, including fluorescence, H&E, and bright-field microscopy, demonstrate that our method consistently achieves superior segmentation accuracy and efficiency compared to state-of-the-art approaches. Future work will explore adaptive encoding strategies that dynamically respond to varying tissue architectures and cell densities, further improving generalization across datasets. Additionally, extending the proposed framework to downstream tasks such as cell nucleus classification represents a promising research direction.

# Acknowledgments

This research was supported by the National Key R&D Program of China (No. 2023YFC3305600), the Federal Ministry of Education and Research in Germany under funding reference 161L0272, and the Ministry of Culture and Science of the State of North Rhine-Westphalia. The authors would like to thank the anonymous reviewers for their valuable comments and suggestions, which greatly improved the quality of this paper.

# Impact Statement

This paper introduces a novel cell instance segmentation method based on four color theorem that significantly improves the accuracy and computational efficiency. By simplifying the segmentation task, this approach has the potential to enhance automated biomedical image analysis and accelerate clinical diagnostics.

# References

Barkley, D., Moncada, R., Pour, M., Liberman, D. A., Dryg, I., Werba, G., Wang, W., Baron, M., Rao, A., Xia, B., et al. Cancer cell states recur across tumor types and form specific interactions with the tumor microenvironment. Nature genetics, 54(8):1192–1201, 2022.   
Caicedo, J. C., Goodman, A., Karhohs, K. W., Cimini, B. A., Ackerman, J., Haghighi, M., Heng, C., Becker, T., Doan, M., McQuin, C., et al. Nucleus segmentation across imaging experiments: the 2018 data science bowl. Nature methods, 16(12):1247–1253, 2019.   
Chen, H., Qi, X., Yu, L., and Heng, P.-A. Dcan: deep contour-aware networks for accurate gland segmentation. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2487–2496, 2016.   
Chen, J., Mei, J., Li, X., Lu, Y., Yu, Q., Wei, Q., Luo, X., Xie, Y., Adeli, E., Wang, Y., et al. Transunet: Rethinking the u-net architecture design for medical image segmentation through the lens of transformers. Medical Image Analysis, 97:103280, 2024.   
Chen, S., Ding, C., Liu, M., Cheng, J., and Tao, D. Cpp-net: Context-aware polygon proposal network for nucleus segmentation. IEEE Transactions on Image Processing, 32:980–994, 2023.   
Chen, Z., Xu, Q., Liu, X., and Yuan, Y. Un-sam: Domain-adaptive self-prompt segmentation for universal nuclei images. Medical Image Analysis, pp. 103607, 2025.   
Cords, L., Tietscher, S., Anzeneder, T., Langwieder, C., Rees, M., de Souza, N., and Bodenmiller, B. Cancer-associated fibroblast classification in single-cell and spatial proteomics data. Nature communications, 14(1):4294, 2023.   
Dietler, N., Minder, M., Gligorovski, V., Economou, A. M., Joly, D. A. H. L., Sadeghi, A., Chan, C. H. M., Koziński, M., Weigert, M., Bitbol, A.-F., et al. A convolutional neural network segments yeast microscopy images with high accuracy. Nature communications, 11(1):5723, 2020.   
Doan, T. N., Song, B., Vuong, T. T., Kim, K., and Kwak, J. T. Sonnet: A self-guided ordinal regression neural network for segmentation and classification of nuclei in large-scale multi-tissue histology images. IEEE Journal of Biomedical and Health Informatics, 26(7):3218–3228, 2022.   
Falk, T., Mai, D., Bensch, R., Çiçek, Ö., Abdulkadir, A., Marrakchi, Y., Böhm, A., Deubner, J., Jäckel, Z., Seiwald, K., et al. U-net: deep learning for cell counting, detection, and morphometry. Nature methods, 16(1):67–70, 2019.

Fritsch, R., Fritsch, R., Fritsch, G., and Fritsch, G. Four-Color Theorem. Springer, 1998.

Gamper, J., Koohbanani, N. A., Benes, K., Graham, S., Jahanifar, M., Khurram, S. A., Azam, A., Hewitt, K., and Rajpoot, N. Pannuke dataset extension, insights and baselines. arXiv preprint arXiv:2003.10778, 2020.

Gonthier, G. et al. Formal proof—the four-color theorem. Notices of the AMS, 55(11):1382–1393, 2008.

Graham, S., Vu, Q. D., Raza, S. E. A., Azam, A., Tsang, Y. W., Kwak, J. T., and Rajpoot, N. Hover-net: Simultaneous segmentation and classification of nuclei in multitissue histology images. Medical image analysis, 58:101563, 2019.

He, H., Huang, Z., Ding, Y., Song, G., Wang, L., Ren, Q., Wei, P., Gao, Z., and Chen, J. Cdnet: Centripetal direction network for nuclear instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4026–4035, 2021.

He, K., Gkioxari, G., Dollár, P., and Girshick, R. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pp. 2961–2969, 2017.

He, Z., Unberath, M., Ke, J., and Shen, Y. Transnuseg: A lightweight multi-task transformer for nuclei segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 206–215. Springer, 2023.

Hörst, F., Rempe, M., Heine, L., Seibold, C., Keyl, J., Baldini, G., Ugurel, S., Siveke, J., Grünwald, B., Egger, J., et al. Cellvit: Vision transformers for precise cell segmentation and classification. Medical Image Analysis, 94:103143, 2024.

Ilyas, T., Mannan, Z. I., Khan, A., Azam, S., Kim, H., and De Boer, F. Tsfd-net: Tissue specific feature distillation network for nuclei segmentation and classification. Neural Networks, 151:1–15, 2022.

Jain, J., Singh, A., Orlov, N., Huang, Z., Li, J., Walton, S., and Shi, H. Semask: Semantically masked transformers for semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 752–761, 2023.

Jiang, H., Zhang, R., Zhou, Y., Wang, Y., and Chen, H. Donet: Deep de-overlapping network for cytology instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15641–15650, 2023.

Kao, K.-C., Vilbois, S., Tsai, C.-H., and Ho, P.-C. Metabolic communication in the tumour–immune microenvironment. Nature cell biology, 24(11):1574–1583, 2022.

Kirillov, A., He, K., Girshick, R., Rother, C., and Dollár, P. Panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9404–9413, 2019.   
Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., and Sethi, A. A dataset and a technique for generalized nuclear segmentation for computational pathology. IEEE transactions on medical imaging, 36(7):1550–1560, 2017.   
Lal, S., Das, D., Alabhya, K., Kanfade, A., Kumar, A., and Kini, J. Nucleisegnet: Robust deep learning architecture for the nuclei segmentation of liver cancer histopathology images. Computers in Biology and Medicine, 128:104075, 2021.   
Li, H., Zhang, D., Dai, Y., Liu, N., Cheng, L., Li, J., Wang, J., and Han, J. Gp-nerf: Generalized perception nerf for context-aware 3d scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21708–21718, 2024.   
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.   
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11976–11986, 2022.   
Ljosa, V., Sokolnicki, K. L., and Carpenter, A. E. Annotated high-throughput microscopy image sets for validation. Nature methods, 9(7):637–637, 2012.   
Mann, M., Kumar, C., Zeng, W.-F., and Strauss, M. T. Artificial intelligence for proteomics and biomarker discovery. Cell systems, 12(8):759–770, 2021.   
Meng, Z., Dong, J., Zhang, B., Li, S., Wu, R., Su, F., Wang, G., Guo, L., and Zhao, Z. Nusea: Nuclei segmentation with ellipse annotations. IEEE Journal of Biomedical and Health Informatics, 2024.   
Merryweather, A. J., Schnedermann, C., Jacquet, Q., Grey, C. P., and Rao, A. Operando optical tracking of single-particle ion dynamics in batteries. Nature, 594(7864):522–528, 2021.   
Pachitariu, M. and Stringer, C. Cellpose 2.0: how to train your own model. Nature methods, 19(12):1634–1641, 2022.

Pan, X., Cheng, J., Hou, F., Lan, R., Lu, C., Li, L., Feng, Z., Wang, H., Liang, C., Liu, Z., et al. Smile: Cost-sensitive multi-task learning for nuclear segmentation and classification with imbalanced annotations. Medical Image Analysis, 88:102867, 2023.   
Petukhov, V., Xu, R. J., Soldatov, R. A., Cadinu, P., Khodosevich, K., Moffitt, J. R., and Kharchenko, P. V. Cell segmentation in imaging-based spatial transcriptomics. Nature biotechnology, 40(3):345–354, 2022.   
Prangemeier, T., Reich, C., and Koeppl, H. Attention-based transformers for instance segmentation of cells in microstructures. In 2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pp. 700–707. IEEE, 2020.   
Qu, H., Yan, Z., Riedlinger, G. M., De, S., and Metaxas, D. N. Improving nuclei/gland instance segmentation in histopathology images by full resolution neural network and spatial constrained loss. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2019:22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part I 22, pp. 378–386. Springer, 2019.   
Raza, S. E. A., Cheung, L., Shaban, M., Graham, S., Epstein, D., Pelengaris, S., Khan, M., and Rajpoot, N. M. Micronet: A unified model for segmentation of various objects in microscopy images. Medical image analysis, 52:160–173, 2019.   
Redmon, J. and Farhadi, A. Yolo9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7263–7271, 2017.   
Ren, S., He, K., Girshick, R., and Sun, J. Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence, 39(6):1137–1149, 2016.   
Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234–241. Springer, 2015.   
Schmidt, U., Weigert, M., Broaddus, C., and Myers, G. Cell detection with star-convex polygons. In Medical Image Computing and Computer Assisted Intervention-MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II 11, pp. 265–273. Springer, 2018.

Srivastava, A., Jha, D., Chanda, S., Pal, U., Johansen, H. D., Johansen, D., Riegler, M. A., Ali, S., and Halvorsen, P. Msrf-net: a multi-scale residual fusion network for biomedical image segmentation. IEEE Journal of Biomedical and Health Informatics, 26(5):2252–2263, 2021.   
Stringer, C. Cellpose3: one-click image restoration for improved cellular segmentation. Nature Methods, pp. 1–8, 2025.   
Stringer, C., Wang, T., Michaelos, M., and Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1):100–106, 2021.   
Wang, P., Chen, P., Yuan, Y., Liu, D., Huang, Z., Hou, X., and Cottrell, G. Understanding convolution for semantic segmentation. In 2018 IEEE winter conference on applications of computer vision (WACV), pp. 1451–1460. Ieee, 2018.   
Xiong, S., Li, X., Zhong, Y., and Peng, W. Repsnet: A nucleus instance segmentation model based on boundary regression and structural re-parameterization. International Journal of Computer Vision, pp. 1–20, 2025.   
Xu, S., Li, G., Song, H., Wang, J., Wang, Y., and Li, Q. Genseg-net: A general segmentation framework for any nucleus in immunohistochemistry images. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 4475–4484, 2024.   
Yu, W., Zhou, P., Yan, S., and Wang, X. Inceptionnext: When inception meets convnext. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5672–5683, 2024.   
Zhang, Y., Cai, L., Wang, Z., and Zhang, Y. Seine: Structure encoding and interaction network for nuclei instance segmentation. IEEE Journal of Biomedical and Health Informatics, 2025a.   
Zhang, Y., Fang, Z., Wang, Y., Zhang, L., Guan, X., and Zhang, Y. Category prompt mamba network for nuclei segmentation and classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 10284–10292, 2025b.   
Zhang, Y., Wang, Y., Fang, Z., Bian, H., Cai, L., Wang, Z., and Zhang, Y. Dawn: Domain-adaptive weakly supervised nuclei segmentation via cross-task interactions. IEEE Transactions on Circuits and Systems for Video Technology, 35(5):4753–4767, 2025c.   
Zhou, T., Wang, W., Konukoglu, E., and Van Gool, L. Rethinking semantic segmentation: A prototype view. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2582–2593, 2022.

Zhou, Y., Onder, O. F., Dou, Q., Tsougenis, E., Chen, H., and Heng, P.-A. Cia-net: Robust nuclei instance segmentation with contour-aware information aggregation. In Information Processing in Medical Imaging: 26th International Conference, IPMI 2019, Hong Kong, China, June 2–7, 2019, Proceedings 26, pp. 682–693. Springer, 2019.

Zhou, Y., Chen, H., Lin, H., and Heng, P.-A. Deep semi-supervised knowledge distillation for overlapping cervical cell instance segmentation. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020:23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part I 23, pp. 521–531. Springer, 2020.

Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., and Liang, J. Unet++: A nested u-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4, pp. 3–11. Springer, 2018.

# Supplementary Material

# A. Data Splitting

We applied an overlapping cropping method to the DSB2018, BBBC006V1, and YeaZ datasets during the data preprocessing stage. Specifically, we used a sliding window with a stride of 128 to extract $256 \times 256$ patches. Since the original image size in the PanNuke dataset is already $256 \times 256$ , no additional processing was required. The number of samples in the training, validation, and test sets is shown in Table 7.

<table><tr><td>Datasets</td><td>No. Traing</td><td>No. Validation</td><td>No. Testing</td></tr><tr><td>DSB2018</td><td>602</td><td>109</td><td>89</td></tr><tr><td>PanNuke</td><td>2656</td><td>2522</td><td>2722</td></tr><tr><td>BBBC006v1</td><td>1848</td><td>612</td><td>612</td></tr><tr><td>YeaZ</td><td>1000</td><td>140</td><td>200</td></tr></table>

Table 7. The number of samples in the training, validation, and test datasets.

# B. Related Work

# B.1. Detection-Based Cell Segmentation

The challenge of distinguishing individual cells in overlapping regions has long been a critical issue in instance segmentation. With the introduction of Faster R-CNN Ren et al. (2016), Mask R-CNN He et al. (2017) extended this detection framework by incorporating an instance segmentation module. This two-stage approach first generates bounding boxes to locate individual instances and then performs segmentation within these regions. Mask R-CNN's inherent ability to separate instances without requiring complex post-processing has made it a widely adopted framework for semi-supervised cell instance segmentation tasks Zhou et al. (2020).

# B.2. Contour Prediction-Based Cell Segmentation

Contour-based segmentation methods focus on explicitly predicting cell boundaries to achieve instance separation. Early works, such as U-Net Ronneberger et al. (2015), facilitated boundary learning by assigning higher pixel-wise weights to cell edges, followed by post-processing techniques like watershed or contour detection to delineate individual instances. This architecture significantly influenced deep learning-based segmentation, particularly in medical imaging. Subsequent advancements introduced explicit contour prediction to improve instance separation. DCAN Chen et al. (2016) incorporated additional semantic categories for boundary pixels, enabling clearer differentiation between cells and the background. UNet++ Zhou et al. (2018) refined U-Net's performance by employing nested skip connections, while FullNet Qu et al. (2019) and CIA-Net Zhou et al. (2019) leveraged multi-scale

context aggregation to enhance boundary delineation. More recent models, such as TSFD-Net Ilyas et al. (2022) and GeNSeg-Net Xu et al. (2024), continue to advance the field by integrating sophisticated architectures designed to improve boundary prediction accuracy.

# B.3. Distance-Based Cell Segmentation

Distance-based segmentation approaches predict spatial relationships between pixels and their corresponding cell instances, facilitating robust separation of adjacent cells. StarDist Schmidt et al. (2018), one of the pioneering methods in this category, introduced radial distance predictions, which proved effective for segmenting cells with irregular shapes. HoverNet Graham et al. (2019) extended this concept by simultaneously predicting a distance map and a classification map, enabling accurate instance separation in densely packed regions. CDNet He et al. (2021) further improved generalization across datasets by employing multi-task learning. Recent advancements have explored more sophisticated architectures to enhance both segmentation accuracy and computational efficiency. SONNET Doan et al. (2022) introduced a self-organizing network to model complex spatial relationships, while TransUNet He et al. (2023) combined transformer-based architectures with distance prediction to enhance feature representation. CPP-Net Chen et al. (2023) and SMILE Pan et al. (2023) incorporated context-aware modules to improve adaptability to diverse cell morphologies. Emerging models such as CellViT Hörst et al. (2024) and RepSNet Xiong et al. (2025) integrate vision transformers with structural priors, further advancing distance-based segmentation techniques for challenging datasets.

![](images/67eb656584790fbbd3c6e31f8ab5058f9000f23d72bd4ff53ce22cabe27c8aec.jpg)

<details>
<summary>text_image</summary>

BBBC006V1
PanNuke
DSB2018
</details>

Figure 9. Visualization of four-color encoding results.

# C. The Analysis of Four-color Encoding

The four-color encoding method highlights the feasibility of transforming cell instance segmentation into a semantic segmentation task. To better understand the characteristics of four-color encoding, we present visualizations of encoded images from multiple datasets, as shown in Figure 9. In this figure, we randomly selected images from three different datasets and applied four-color encoding. The results reveal the following patterns:

(1) The majority of cells are encoded in red, while a smaller proportion are assigned green;   
(2) Cells encoded in blue are scarce, appearing only in highly dense regions (highlighted by white box), typically with one or two occurrences;   
(3) The fourth encoding category (represented by yellow) does not appear, indicating that cell encoding is more constrained and simplified than the traditional map-coloring problem.

Furthermore, the statistical analysis of the four-color encoding results, illustrated in Figure 10, aligns with the observed distribution of cell color assignments, further validating the characteristics of this encoding approach.

![](images/eb3da65e5d11b8a6d6a26a71a0f2d8b7ed57e22db104a8170782656c3032b05d.jpg)

<details>
<summary>bar</summary>

| enc | Value |
| --- | --- |
| 1   | 25  |
| 2   | 10  |
| 3   | 5   |
</details>

(a) DSB2018

![](images/0d558e8f344ab95a5b3b41388f8b479d578333ad309e5d16b4c8807dacbaae40.jpg)

<details>
<summary>scatter</summary>

| enc | Value |
| --- | --- |
| 1   | 25  |
| 2   | 40  |
| 3   | 10  |
</details>

(b) PanNuke   
Figure 10. Statistics of different color encodings

# D. Preliminaries

We provide essential definitions and concepts to establish the foundation for the proposed method.

Definition 1. Undirected Graph. An undirected graph is represented as $G = (V, E)$ , where V is the set of vertices, and E is the set of edges. An edge $e = (u, v) \in E$ indicates that vertices u and v are adjacent.

Definition 2. Coloring Number. The chromatic number of a graph G, denoted as $\chi(G)$ , is the minimum number of colors required to color the vertices of G such that no two adjacent vertices share the same color.

Definition 3. Maximum Degree. The degree of a vertex $v \in V$ , denoted as $d(v)$ , is the number of vertices adjacent to $v$ . The maximum degree of the graph $G$ is defined as $\Delta(G) = \max_{v \in V} d(v)$ .

Definition 4. Chain Structure: A type of graph where the vertices are arranged in a linear path, formally known as a path graph $P_{n}$ . In the structure, each vertex is connected to at most two adjacent vertices. For instance, in the graph $P_{4}$ with 4 vertices, the coloring sequence can be described as:

$$
v _ {1} \rightarrow \text { color } 1, v _ {2} \rightarrow \text { color } 2, v _ {3} \rightarrow \text { color } 1, v _ {4} \rightarrow \text { color } 2.
$$

Definition 5. Rectangular Structure: A rectangular structure is a graph where vertices are arranged in a regular rectangular grid. Such graphs are a specific type of planar graph, where each vertex typically

has a degree of 2 or 4, satisfying $\Delta(G) \leq 4$ .

Definition 6. Planar Graph. A planar graph is a graph that can be embedded in the plane such that no edges intersect. According to the Four-Color Theorem, the chromatic number of a planar graph satisfies $\chi(G) \leq 4$ .

# E. Theorem and Proof

Theorem 1. Global Optimality of Greedy Coloring: Let $G = (V, E)$ be an undirected graph representing a cell distribution, where V is the set of vertices (cells), and E is the set of edges representing adjacency relationships between cells. Suppose G satisfies the following conditions:

(1) $G$ is a planar graph, meaning it can be embedded in a plane such that no two edges intersect;   
(2) The maximum degree of $G$ , denoted by $\Delta(G)$ , satisfies:

$$
\Delta (G) \leq k, \quad \text { where } k \leq 4;
$$

(3) The vertex distribution of G follows either a chain structure (a path graph $P_{n}$ ) or a rectangular structure (a grid-like planar graph).

Then, the chromatic number of G, defined as the minimum number of colors required to color the vertices such that no two adjacent vertices share the same color, satisfies:

$$
\chi_ {\text { greedy }} (G) = \chi (G).
$$

Where $\chi_{\mathrm{greedy}}(G)$ is the coloring number by applying the greedy algorithm with any arbitrary vertex ordering. This result demonstrates that the greedy algorithm produces a globally optimal solution to the graph coloring problem.

# Proof 1:

Definition and Properties of Greedy Algorithm: The greedy algorithm colors graph G as follows: - Traverse all vertices in the order $v_{1}, v_{2}, \ldots, v_{n}$ ; - For each vertex $v_{i} \in V$ , assign the smallest color that has not been used by any of its adjacent vertices; - Each vertex checks at most $\Delta(G)$ adjacent vertices, and the number of colors needed is at most $\Delta(G) + 1$ .

Thus, the chromatic number generated by the greedy algorithm satisfies:

$$
\chi_ {\mathrm{greedy}} (G) \leq \Delta (G) + 1
$$

# Optimality Analysis under Special Structures:

(a) Chain Structure (Path Graph $P_{n}$ ): For a path graph $P_{n}$ , each vertex has a degree $\Delta(P_{n}) = 2$ . - The chromatic number of a path graph is $\chi(P_{n}) = 2$ ; - When the greedy algorithm colors in any vertex

order, it uses at most two colors:

$$
\chi_ {\mathrm{greedy}} (P _ {n}) = \chi (P _ {n}) = 2
$$

Therefore, the greedy algorithm is optimal for path graphs.

(b) Rectangular Structure: For cells arranged in a rectangular grid, graph $G$ is planar, and $\Delta(G) \leq 4$ . According to the Four Color Theorem:

$$
\chi (G) \leq 4
$$

The greedy algorithm, in each iteration, uses the smallest available color, and each vertex checks at most 4 adjacent vertices. Therefore, the chromatic number generated by the greedy algorithm satisfies:

$$
\chi_ {\mathrm{greedy}} (G) \leq 4 = \chi (G)
$$

Thus, the greedy algorithm is also optimal for rectangular structures. Extending Local Optimality to Global Optimality:

Local Sparsity: Due to the distribution properties of graph G, in locally clustered regions, the number of vertices is limited and the maximum degree is low. Hence, the greedy algorithm is optimal in local regions.

Global Sparsity of Planar Graphs: The global distribution of planar graphs is sparse, and edges connecting different regions are limited, causing little interference with the local optimal solution. As a result, the local optimality of the greedy algorithm extends to global optimality.

Based on the above analysis, the chromatic number of graph G, which satisfies the given conditions, is equal to the minimum chromatic number:

$$
\chi_ {\mathrm{greedy}} (G) = \chi (G)
$$

Thus, the greedy algorithm is an effective method for generating the minimum color coding in this scenario.

Theorem 2. Greedy Coloring Compatibility: In the cell instance segmentation task, let the encoding matrix generated by the greedy algorithm be:

$$
\mathbf {C} \in \mathbb {R} ^ {n \times k}, (k \leq 4). \tag {16}
$$

And the encoding matrix predicted by the network is:

$$
\mathbf {P} \in \mathbb {R} ^ {n \times k ^ {\prime}}, \tag {17}
$$

where n represents the number of cells, k is the number of colors used in the greedy algorithm, and $k'$ is the number of predicted encodings. If the predicted encoding matrix P has one of the relations with

the greedy encoding, i.e., substitution, exchange, modification of rules. Then there exists a mapping function:

$$
f: \mathbf {P} \rightarrow \mathbf {C}, \tag {18}
$$

such that the network's predicted result P can be transformed into the four-color encoding result C.

# Proof 2:

The four-color encoding matrix C generated by the greedy algorithm satisfies the following properties:

(a) Sparsity: Each row has at most one nonzero element $(\mathbf{C}[i,j] \in \{0,1\})$ , representing that the i-th node uses the j-th color;   
(b) Optimality: The number of colors used is minimized, $\text{rank}(\mathbf{C}) = k$ , and $k \leq 4$ ;   
(c) Adjacency constraint: Any two adjacent nodes $(v_{i}, v_{j})$ satisfy $C[i, :] \neq C[j, :]$ (i.e., they cannot use the same color).

These properties can be formally expressed as follows:

(1) Sparsity: $\sum_{j=1}^{k}C[i,j]=1,\forall i.$ (2) Adjacency constraint: If $e_{i,j}=1$ , then $C[i,:]\cdot C[j,:]^{\top}=0$ .

The encoding matrix $P \in R^{n \times k'}$ predicted by the network exhibit non-uniqueness due to the following reasons:

(a) Substitution: Some rows of the encoding are replaced, introducing redundancy;   
(b) Exchange: The order of the columns is changed;   
(c) Rule modification: Additional colors are introduced, resulting in $k' > k$ .

Thus, the column rank of P satisfies:

$$
\operatorname{rank} (\mathbf {P}) \geq k. \tag {19}
$$

Hence, we need to construct a mapping function $f : P \rightarrow C$ to transform the predicted encoding matrix P into the four-color encoding matrix C that satisfies the constraints.

(1) Column Redundancy Elimination

A linear transformation is applied to eliminate redundant columns in P, ensuring that the resulting matrix has rank k. Specifically: Define a column transformation matrix $T \in R^{k' \times k}$ , where

$$
\mathbf {T} = \underset {\mathbf {T}} {\operatorname{argmin}} \| \mathbf {P T} - \mathbf {C} \| _ {F} ^ {2}, \quad \text {s.t.} \operatorname{rank} (\mathbf {P T}) = k.
$$

The transformed matrix is

$$
\mathbf {P} ^ {\prime} = \mathbf {P T},
$$

where $P' \in R^{n \times k}$ , and $\text{rank}(\mathbf{P}') = k$ .

(2) Column Order Adjustment

The columns of $P'$ are reordered to align with the column order of C. Let the column permutation matrix be $S \in R^{k \times k}$ , then

$$
\mathbf {C} = \mathbf {P} ^ {\prime} \mathbf {S}.
$$

The matrix S is a permutation matrix satisfying $S^{\top}S = I$ .

(3) Adjacency Constraint Verification

After the mapping, the adjacency constraint is verified to ensure that the resulting matrix satisfies the four-color encoding rule:

$$
\mathbf {C} [ i,: ] \cdot \mathbf {C} [ j,: ] ^ {\top} = 0, \quad \forall e _ {i, j} = 1.
$$

It can be seen that, for any predicted matrix P, the three-step mapping function f ensures that the transformed matrix C satisfies:

(1) The rank of the transformed matrix is k, i.e., $\text{rank}(\mathbf{P}') = k$ ;   
(2) The column order is aligned with C;   
(3) The adjacency constraint holds, making C a valid four-color encoding result.

Therefore, we design encoding transformation and orthogonal constraints to ensure the rationality of four-color prediction.

# F. Hyper-parameter Ablation Experiments

We conducted an ablation study on hyperparameter selection using the DSB2018 and BBBC006v1 datasets, focusing on the impact of the sampling rate and the weight of the orthogonal constraint loss function. The experimental results are presented in Tables 8 and 9.

The results indicate that increasing the sampling rate generally improves model performance. However, the performance gain from 0.5 to 0.7 is less significant than the improvement observed when increasing the sampling rate from 0.3 to 0.5. We set the sampling rate to 0.5 in the main experiments to balance model performance and computational efficiency.

Furthermore, we examined the effect of the orthogonal constraint loss weight on model performance. A significant performance drop is observed when the weight is set to 1. We hypothesize that this is due to the insufficient enforcement of the orthogonal constraint at lower weights, reducing the model's ability to distinguish adjacent instances and ultimately degrading segmentation performance effectively.

<table><tr><td rowspan="2">Ratio</td><td colspan="5">DSB2018</td><td rowspan="2">Ratio</td><td colspan="5">BBBC006v1</td></tr><tr><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td></tr><tr><td>r=0.3</td><td>0.913</td><td>0.803</td><td>0.854</td><td>0.871</td><td>0.758</td><td>r=0.3</td><td>0.947</td><td>0.917</td><td>0.893</td><td>0.926</td><td>0.899</td></tr><tr><td>r=0.5</td><td>0.939</td><td>0.828</td><td>0.875</td><td>0.878</td><td>0.770</td><td>r=0.5</td><td>0.954</td><td>0.921</td><td>0.926</td><td>0.945</td><td>0.935</td></tr><tr><td>r=0.7</td><td>0.941</td><td>0.832</td><td>0.866</td><td>0.881</td><td>0.779</td><td>r=0.7</td><td>0.946</td><td>0.924</td><td>0.933</td><td>0.951</td><td>0.938</td></tr></table>

Table 8. Ablation studies of sampling ration on DSB2018 and BBBC006v1 datasets. 

<table><tr><td rowspan="2">Weight</td><td colspan="5">DSB2018</td><td rowspan="2">Weight</td><td colspan="5">BBBC006v1</td></tr><tr><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td><td>DICE</td><td>AJI</td><td>DQ</td><td>SQ</td><td>PQ</td></tr><tr><td> $\lambda = 1$ </td><td>0.908</td><td>0.798</td><td>0.832</td><td>0.825</td><td>0.716</td><td> $\lambda = 1$ </td><td>0.922</td><td>0.898</td><td>0.891</td><td>0.904</td><td>0.880</td></tr><tr><td> $\lambda = 2$ </td><td>0.939</td><td>0.828</td><td>0.875</td><td>0.878</td><td>0.770</td><td> $\lambda = 2$ </td><td>0.954</td><td>0.921</td><td>0.926</td><td>0.945</td><td>0.935</td></tr></table>

Table 9. Ablation studies of weight setting on DSB2018 and BBBC006v1 datasets.

# G. More Visualization Results

We present the semantic and instance segmentation results, including error analysis, as shown below. Specifically, the subfigures include the input image, pixel-wise error analysis, four-class semantic ground truth, and instance segmentation labels (the last two subfigures can be ignored). The results in the DSB2018 and BBBC006v1 datasets demonstrate that our method not only achieves accurate instance segmentation but also excels in pixel-wise classification by significantly reducing false positive (FP) and false negative (FN) prediction errors. These results validate the effectiveness of our approach.

![](images/826bf724cd181b9005b08203614834c2e390740207d20b3a3dad5f34aedaa597.jpg)

![](images/dbd21cce75259d985ab488532d2fdbaa7b980a67262ab8e2fbf04c62a15cbb2e.jpg)

Image   
![](images/b5adf472e24c2a2438797fbfc87efdebcf24ce905de8fe0cad12746332d53506.jpg)

<details>
<summary>natural_image</summary>

Microscopic tissue section showing cellular structures with pink and purple staining (no text or labels visible)
</details>

Error Analysis: FN-FP-TP   
![](images/d10947e7a0fa7ebfad5863cee49917d639da686626dfb5fdec6402257541d247.jpg)

<details>
<summary>natural_image</summary>

Fluorescently labeled cells with blue, green, and red markers against a black background (no text or symbols)
</details>

Instance Level Prediction   
![](images/0892cacd04d2f1aef5a3d0013a3fde277fc82b63f2531b5c21f9a655cc371b02.jpg)

<details>
<summary>natural_image</summary>

Colorful abstract shapes on black background, no text or symbols present
</details>

Instance Level Ground Truth   
![](images/7cf2c08ec24ddf11e6e5f0b911f70d4f6f7b1bd659d117310cbf9b20ae12a7c4.jpg)

<details>
<summary>natural_image</summary>

Abstract colorful oval shapes on black background (no text or symbols)
</details>

![](images/95869e12ae938231c696d808215b4b112954e119e2dfa5e961b3a445c8e0ecfd.jpg)

Semantic Level Prediction   
![](images/1c002b7b368935b72bb6744cf474407911f033d1aa4f61977271e96ffacb4abb.jpg)

<details>
<summary>natural_image</summary>

Abstract composition of purple and pink irregular shapes on black background (no text or symbols)
</details>

Semantic Level Ground Truth   
![](images/647330891e14f66fb86262e9075638a69ea42ffda1a6e54d9499bb90c93f4ed6.jpg)

<details>
<summary>natural_image</summary>

Abstract composition of blue and green oval shapes on black background (no text or symbols)
</details>

Three-class Semantic Level Prediction   
![](images/f6d8154c6cda51420b651b2cdc16c6c8c094e23454d2ce76625d35349d823693.jpg)

<details>
<summary>natural_image</summary>

Abstract composition of scattered light blue and orange oval shapes on black background (no text or symbols)
</details>

Three-class Semantic Level Ground Truth   
![](images/7be6ed973e181feb08c338dcbd15786650ecba2ee460959a9e4da4ac553418ae.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of green fluorescent cells against a black background (no text or symbols)
</details>

# H. Others

To better demonstrate the rationality of our model's module design, we plot the convergence curves of various loss functions during the training process on the PanNuke dataset, as shown in Figure 11. The results indicate that our method ensures stable model convergence.

![](images/b9dabc8e6a58ef3f91e51d6c0e320f8f1b922764e62d8eea307edf76197da330.jpg)  
Figure 11. The convergence of loss function and Dice in training process.