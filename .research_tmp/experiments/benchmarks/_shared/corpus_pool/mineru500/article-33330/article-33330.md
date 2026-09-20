# Effective and Efficient Representation Learning for Flight Trajectories

Shuo Liu $^{1,2}$ , Wenbin Li $^{2,3}$ , Di Yao $^{2*}$ , Jingping Bi $^{2*}$

$^{1}$ School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences, China

$^{2}$ Institute of Computing Technology, Chinese Academy of Sciences, China

$^{3}$ University of Chinese Academy of Sciences, China

{liushuo22s, liwenbin20z, yaodi, bjp}@ict.ac.cn

# Abstract

Flight trajectory data plays a vital role in the traffic management community, especially for downstream tasks such as trajectory prediction, flight recognition, and anomaly detection. Existing works often utilize handcrafted features and design models for different tasks individually, which heavily rely on domain expertise and are hard to extend. We argue that different flight analysis tasks share the same useful features of the trajectory. Jointly learning a unified representation for flight trajectories could be beneficial for improving the performance of various tasks. However, flight trajectory representation learning (TRL) faces two primary challenges, i.e. unbalanced behavior density and 3D spatial continuity, which disable recent general TRL methods. In this paper, we propose FLIGHT2VEC, a flight-specific representation learning method to address these challenges. Specifically, a behavior-adaptive patching mechanism is used to inspire the learned representation to pay more attention to behavior-dense segments. Moreover, we introduce a motion trend learning technique that guides the model to memorize not only the precise locations, but also the motion trend to generate better representations. Extensive experimental results demonstrate that FLIGHT2VEC significantly improves performance in downstream tasks such as flight trajectory prediction, flight recognition, and anomaly detection.

# Introduction

With the continual development of air transportation, flights tend to broadcast their current locations for safe and efficient air traffic control (Zhang et al. 2023). The collected flight trajectories are one of the most critical data sources for air traffic management, attracting increasing attention from both academic and industrial fields. Recent works show that deep representation learning has become the dominant technique and achieved significant performance for various flight trajectory-related tasks, such as trajectory prediction (Liu and Hansen 2018; Wu et al. 2022; Zhang et al. 2023; Guo et al. 2024), flight monitoring (Zhang et al. 2016; Fernández et al. 2019), and anomaly detection (Olive and Basora 2020; Memarzadeh, Matthews, and Templin 2022; Memarzadeh, Matthews, and Weckler 2023). However, these works either utilize handcrafted features or design representation models for one specified task. We argue that different flight tasks may share the same useful features of trajectory. Jointly learning a unified representation for flight trajectories could be beneficial for boosting the performance of various tasks, which motivates this work.

![](images/47f239e5b39150a69d5ad91abc88c695b9a548554a225bf2ab857b028d90c184.jpg)

<details>
<summary>text_image</summary>

ALB
Behavior Parts
5764 Points
The total number of points
in the entire flight trajectory
</details>

![](images/345476589c7b09f2b3e161d9c8aed76ac106acafb35e1f3437030c60d398441b.jpg)

<details>
<summary>text_image</summary>

165 Points
</details>

(a) Unbalanced behavior density   
![](images/e990b859d59bdbd6442fabf0c6011e4e807e8181b9638bf1c29822533b4df211.jpg)

<details>
<summary>natural_image</summary>

3D diagram of an airplane in flight with red and blue trajectory arrows indicating movement (no text or symbols)
</details>

(b) 3D spatial continuity   
Figure 1: The motivation of FLIGHT2VEC

Learning unified trajectory representation for different tasks has been well-studied on vehicle and human mobility trajectories (Yao et al. 2017; Li et al. 2018; Chen et al. 2021; Jiang et al. 2023). Deep sequential models, such as Recurrent Neural Networks (Yao et al. 2017; Li et al. 2018) and Transformer (Chen et al. 2021; Yao et al. 2022; Jiang et al. 2023) are employed to encode the spatial-temporal correlations and transform raw trajectories into generic representation vectors. Nevertheless, existing solutions are hard to extend to represent flight trajectories due to the two challenges, i.e. unbalanced behavior density and 3D spatial continuity.

Unbalanced behavior density. Flight trajectories are characterized by high point density but sparse behavior information. As shown in Figure 1(a), the aircraft usually fly in a straight line with a fixed attitude. The representative activities, such as turns, holding patterns, and takeoff/landing phases, only account for a relatively small part, e.g. 5%, of the whole trajectory. Existing TRL methods usually treat every point equally without considering the density of behaviors, leading to inconsequential representations.

3D spatial continuity. Flights move in a three-dimensional space, exhibiting more complex spatial patterns compared to ground trajectories. As shown in Figure 1(b),

the movement of fights follows the spatial proximity in 3D space, i.e. the motion trend of flights are constrained in the forward direction. However, previous works utilize MSE loss to reconstruct location coordinates, which does not effectively capture the spatial continuity and dependencies. Moreover, MSE loss is sensitive to outliers, which significantly affect the representation learning process.

In this paper, we propose a generic flight trajectory representation learning method FLIGHT2VEC which encourages the learned representation paying more attention to behavior parts and memorize the motion trend of flights. Specifically, a novel behavior-based patching mechanism is designed to automatically sample trajectories according to the behavior density and reduce the sequence length of model input. FLIGHT2VEC first employs a criterion to select the behavior parts of trajectories and construct patches for each behavior respectively. The generated patches are feed to the decoder-only Transformer to obtain the representations. To model the 3D spatial continuity, we propose a motion trend learning technique which predicts the moving direction of each record as a 26-classes classification task. Along with the MSE loss, FLIGHT2VEC can not only extract the information of spatial coordinates but also capture the motion trend of flights. In summary, the main contributions of this paper are summarized as follows:

- We propose an effective and efficient representation learning framework FLIGHT2VEC for flight trajectories. To the best of our knowledge, this is the first work that designs a general representation learning framework specifically tailored to the characteristics of flight trajectories.   
- We introduce an behavior-adaptive patching mechanism to achieve effective trajectory representation learning while preserving the information of behavior density. Additionally, we propose a motion trend learning technique that explicitly models the spatial continuity flight trajectories without requiring additional features.   
- Extensive experiments demonstrate that FLIGHT2VEC significantly enhances performance across various downstream tasks, such as anomaly detection, trajectory prediction, and flight monitoring.

# Related Work

Flight Trajectory Analysis Framework. Existing research on flight trajectory data analysis always requires manual feature engineering and specialized models for each task, heavily relying on domain expertise and being difficult to extend. For instance, (Guo et al. 2022a, 2024) proposed a feature representation method based on binary encoding (BE) for trajectory prediction, utilizing Conv1D and Transformer modules to capture spatiotemporal features of trajectory points. (Fang et al. 2021) selected 10 correlation coefficient features, such as Oil Temperature (OilT) and Oil Pressure, for trajectory recognition. (Qin et al. 2022) introduced unsupervised feature engineering methods to map input data into latent feature spaces for anomaly detection. The limitation of these methods is that they require expert knowledge and complex feature engineering to extract useful information. There are also some learning-based methods, but they do not solve the problem of uneven behavior density. They either model the entire trajectory indiscriminately (Guo et al. 2024) or only focus on specific segments such as takeoff and landing phases (Fernández et al. 2019; Memarzadeh, Matthews, and Templin 2022), unable to adaptively identify and model informative segments of the trajectory. In conclusion, although existing methods have achieved success in specific tasks, they rely on complex feature engineering highlights the need for a unified approach.

Trajectory Representation Learning. Trajectory Representation Learning (TRL) has gained significant attention in the data engineering community due to its effectiveness in enhancing various downstream tasks. Existing methods can be broadly divided into two categories: road network-based methods and grid-based methods. Road network-based methods (Fu and Lee 2020; Chen et al. 2021; Jiang et al. 2023; Qian et al. 2024) map trajectories to nodes of the road network and learn representations of nodes to generate trajectory representations. However, flight trajectories can move in three-dimensional space without the constraints of road networks, making road network-based methods inapplicable. Grid-based methods (Yao et al. 2018, 2019; Li et al. 2018; Yang et al. 2021; Jing et al. 2022) divide the geographical space into a grid of cells and map trajectories to these grids to learn representations. These methods are also difficult to apply to flight trajectories since dividing 3D space into grids will result in an exponential increase in the number of grids. This will make the trajectory data on many grids too sparse, thus the model cannot effectively learn trajectory representations. Overall, there is no existing work specifically focused on representation learning for flight trajectories and existing TRL methods are not applicable to flight trajectories.

Patch-based Transformer. In recent years, patch-based approaches for time series analysis have emerged as mainstream. Compared to point-wise processing methods, these models not only improve efficiency but also exhibit notable performance improvements, revealing the importance of enhancing modeling of local semantics through patching. For instance, PatchTST (Nie et al. 2022) segments each time series into patches and employs an Multi-layer Perceptron (MLP) to feed patch embeddings into a Transformer. This method also utilizes mask-based unsupervised pre-training to learn representations that can generalize to various downstream tasks. Following PatchTST, a series of patch-based Transformer models for time series have continually set new benchmarks in various downstream tasks, such as prediction (Wang et al. 2024; Chen et al. 2024), classification (Li, Li, and Yan 2024) and anomaly detection (Yang et al. 2023). Notably, HDMixer (Huang et al. 2024) has revealing the importance of patch division. This research shows that incorrect patch boundaries can obscure local patterns and disrupt the semantic continuity of the sequence. However, these methods do not address the issue of unbalanced behavior density, i.e., they fail to effectively capture sparse but critical behaviors in flight trajectories.

![](images/ae9d02c85ea44535393dea056498c4a88ad4469424109157890fcf141960dc0e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["T"] --> B["Behavior-Based Patching"]
    B --> C["patches p"]
    C --> D["Mask Patch Transformer"]
    D --> E["Trajectory Representation v"]
    F["Input sequence length L"] --> G["Patch size S"]
    G --> H["patching"]
    H --> I["trajectory length n"]
    I --> B
    J["Linear Layer"] --> K["mask p ∈ Pb"]
    K --> L["Transformer Encoder"]
    L --> M["backward L"]
    M --> N["reconstruct"]
    N --> O["Output"]
    P["Motion Trend Learning"] --> Q["xi^mask"]
    P --> R["yi^mask"]
    P --> S["\hat{x}_i^mask"]
    P --> T["\hat{y}_i^mask"]
    Q --> U["L = L_MSE + λ·L_MD"]
    R --> U
    S --> U
```
</details>

Figure 2: Overview of FLIGHT2VEC Preliminary

In this section, we first define the problem and then describe the proposed method FLIGHT2VEC, respectively.

Problem Definition. Given a flight trajectory denoted by $T = \{x_{1}, x_{2}, \ldots, x_{n}\}$ , where each $x_{i}$ represents a recorded point at timestamp i, the objective of Trajectory Representation Learning (TRL) is to generate a general low-dimensional representation, $v \in R^{D}$ , that can be utilized in various downstream tasks.

In this study, each trajectory point $x_{i}$ is described by six key attributes related to the aircraft's status:

$$
x _ {t} = \left[ l o n _ {i}, l a t _ {i}, a l t _ {i}, V _ {l o n _ {i}}, V _ {l a t _ {i}}, V _ {a l t _ {i}} \right],
$$

where $lon_{i}$ , $lat_{i}$ , and $alt_{i}$ represent the longitude, latitude, and altitude of the aircraft, respectively. The attributes $V_{lon_{i}}$ , $V_{lat_{i}}$ , and $V_{alt_{i}}$ denote the velocity components in the longitudinal, latitudinal, and altitude dimensions, respectively.

Overview of FLIGHT2VEC. As shown in Figure 2, FLIGHT2VEC consists of two key components, i.e. behavior-adaptive patching Transformer and model optimization. In the first component, we segment the flight trajectory into a sequence of patches according to the density of behaviors. For non-behavior segments, we down-sample the original records and compress the long segments into patches with equal size. The generated patches are feed to a patch Transformer encoder to obtain the flight representation. To optimize the parameters in FLIGHT2VEC, a motion trend learning approach is proposed along with the traditional MSE loss to reconstruct the masked patches. It encourages the learned representations memorize not only the location coordinates but also the moving directions of each records in the masked patches.

# Methodology

We specify the two key components of FLIGHT2VEC, i.e., the behavior-adaptive patching Transformer, and the optimization of the model, respectively.

# Behavior-Adaptive Patching Transformer

Behaviors such as takeoff and turning only account for a very small part of flight trajectories, but they are informative and crucial for modeling trajectory patterns. Therefore, we propose an behavior-based patching mechanism that extracts more informative features by amplifying behavior-dense segments.

![](images/3152c0fb74bd8820700c7834734d60b80bfd4fc77a12e4df33292dd25063b34c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["non-behavior segments a_j"] --> B["sample"]
    B --> C["behavior segments A"]
    D["p_i"] --> E["S = 4"]
    E --> F["p_{i+1}"]
    F --> G["S = 4"]
    G --> H["behavior segments A"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fcf,stroke:#333
    style H fill:#cff,stroke:#333
```
</details>

Figure 3: Behavior-Based Patching

Behavior-Based Patching We observe that: (i) behavior segments reflect significant trajectory changes, such as turns and holding patterns; (ii) non-behavioral segments show stable movement with dense point distributions, which are less informative. Inspired by the observation, we first identify behavior segments A in a flight trajectory T and then adaptively patch the trajectory based on A.

Specifically, considering a flight trajectory indicated by $T = \{x_{1}, x_{2}, \ldots, x_{n}\}$ , we first apply a threshold filter to remove noisy points with excessive oscillations. Next, we calculate the angle change for each point as: $angle_{i} = \{a_{1}, a_{2}, \ldots, a_{n}\}$ .

Then, points with angle changes exceeding a threshold s are identified as active points, denoted by $A'$ :

$$
\mathcal {A} ^ {\prime} = \left\{a _ {j} \mid \text { angle } _ {j} > s \right\}
$$

After that, we use a behavior-based patching algorithm to create patches as the inputs of Transformer. Since each behavior is composed of multiple consecutive points, for each active point, we compute its index distance to neighboring active points and cluster those with index distance less than a threshold. Then, we select a central active point c from each cluster as a patch center and we can get a patch with a predefined patch size S. As illustrated in Figure 3, $a_{j}$ , $a_{j+1}$ , and $a_{j+2}$ are all active points. Since they are adjacent, we consider them to be in the same cluster and select point $a_{j+1}$ as the center point c. After that, we obtain a behavioral patch set $P_{b}$ consists of g patches, where g is the number of active point clusters. Each patch in $P_{b}$ preserves the information of a behavioral segment, improving the model's ability to learn from local patterns.

For non-behavioral segments, we perform uniform sampling with a step size of $\frac{n-S\cdot g}{N-g}$ to generate additional patches from these segments, where the N denotes the number of patches and S is the patch size. Overall, we get a sequence of patches $P = [p_{1}, \ldots, p_{N}]$ which consists of g behavioral patches and N - g non-behavioral patches.

![](images/8ba8d31c52746c0be67ed96f9a48f207a0f1eb754e7ce57f8bea5b66cc4526af.jpg)

<details>
<summary>text_image</summary>

MSE(xₐ,x₂) = MSE(xₑ,x₂)
x₁	x₂	x₃
x₄
xₘₐₓₖ
Up Z
(-1,1,1)
Forward
Y
(-1,-1,-1)
Right
</details>

Figure 4: Illustration of Motion Trend Learning

Patch Transformer FLIGHT2VEC utilizes a similar approach similar to PatchTST (Nie et al. 2022) to learn the trajectory embeddings.

Specifically, we transform each patch $p_{i}$ into the latent space of dimension $d_{m}$ via a trainable linear projection $W_{p}$ , and add a learnable additive position encoding matrix $W_{pos}$ to encode its temporal order:

$$
\mathbf {Y} = \mathbf {W} _ {p} \mathbf {P} + \mathbf {W} _ {p o s}.
$$

Then, we feed Y into a Transformer to generate the final trajectory embedding $\mathbf{v} = \mathbf{Transformer}(\mathbf{Y})$ , where $v \in R^{N \times S \times d_{m}}$ .

# Optimization of FLIGHT2VEC

To model the complex moving patterns and the spatial continuity in flight trajectories, we optimize FLIGHT2VEC with a mask-based self-supervised learning strategy and a novel motion trend learning.

Mask-based self-supervised learning. Previous methods often adopt random masking to optimize patch Transformer, which is not suitable for flight trajectories where most points are moving uniformly in a straight line. For these points, the model can easily infer the masked values through simple interpolation between adjacent time points, without capturing the complex trajectory patterns. To address this problem, we introduce a motion-based randomization strategy to effectively mask the behavior patches $P_{b}$ and their surrounding areas. Specifically, we mask the patches in $P_{b}$ and their neighboring patches with the probability $\rho_{b}$ , and we mask other patches with the probability $\rho_{n}$ , where $\rho_{n} < \rho_{b}$ . Then, we use a $D \times P$ linear layer to reconstruct the masked patches $x^{mask}$ . The Mean Squared Error (MSE) loss to minimize reconstruction errors:

$$
\mathcal {L} _ {M S E} = \mathbb {E} _ {x} \frac {1}{M} \sum_ {i = 1} ^ {M} \left\| \hat {x} _ {i} ^ {\text {masked}} - x _ {i} ^ {\text {mask}} \right\| _ {2} ^ {2}
$$

where M is the number of points in the masked patches.

Motion Trend Learning. However, MSE loss is not enough to model flight trajectories for two main reasons. First, MSE loss treats all errors equally, regardless of their spatial context, which means it does not effectively capture the spatial continuity and dependencies inherent in flight trajectories. Second, MSE loss does not adequately model the uncertainty and sparsity of significant behavioral points within dense trajectory data. Critical behavioral segments, such as turns and altitude changes, are sparse but vital for accurate representation. Optimizing the model with only MSE loss may result in suboptimal learning, as it might not adequately focus on these sparse yet crucial points.

For example, as illustrated in Figure 4, consider a point $x_{1}$ that needs to be reconstructed or predicted, with two candidate points $x_{a}$ and $x_{b}$ . Both $x_{a}$ and $x_{b}$ are equidistant from $x_{1}$ , resulting in the same MSE loss. But $x_{a}$ is better than $x_{b}$ because it is located on the line where the trajectory is moving forward. To model this preference and inherent uncertainty, we introduce the moving direction loss function.

Specifically, we categorize each point in the trajectory based on its movement direction. For point $x_{i} = (lon_{i}, lat_{i}, alt_{i})$ and $x_{i+1} = (lon_{i+1}, lat_{i+1}, alt_{i+1})$ , we calculate the direction vector $\vec{d}$ from $x_{i}$ to $x_{i+1}$ :

$$
\vec {d} = \left(l o n _ {i + 1} - l o n _ {i}, l a t _ {i + 1} - l a t _ {i}, a l t _ {i + 1} - a l t _ {i}\right).
$$

Each component of $\vec{d}$ can be positive, negative, or zero, corresponding to the direction along each axis. The direction is represented as a triplet $(d_{lon}, d_{lat}, d_{alt})$ based on the direction vector, where $d_{lon} = \text{sign}(lon_{i+1} - lon_i)$ , $d_{lat} = \text{sign}(lat_{i+1} - lat_i)$ , and $d_{alt} = \text{sign}(alt_{i+1} - alt_i)$ . The sign function is defined as:

$$
\operatorname{sign} (u) = \left\{ \begin{array}{l l} 1 & \text { if } u > 0 \\ 0 & \text { if } u = 0 \\ - 1 & \text { if } u <   0 \end{array} \right.
$$

We divided the 3D moving space into 26 categories based on directional movement. Each dimension (longitude, latitude, and altitude) has three possible states: positive, negative, or unchanged. Combining these states yields $3 \times 3 \times 3 - 1 = 26$ unique directions, excluding the case where all dimensions remain unchanged. Each point is then assigned a category corresponding to its movement direction.

We adopt following moving direction loss to optimize the model and to learn these directional preferences:

$$
\mathcal {L} _ {M D} = - \mathbb {E} _ {x} \log \mathbb {P} (y _ {i} ^ {\text { masked }} \mid y _ {1: i - 1} ^ {\text { mask }}, x)
$$

where $y_{i}^{masked}$ represents the label of one of the 26 moving directions.

The final loss is a combination of Mean Squared Error (MSE) loss and the moving direction loss, enabling the model to capture both spatial relationships and precise values. Thus, the final combined loss function is:

$$
\mathcal {L} _ {m} = \mathcal {L} _ {M D} + \lambda \cdot \mathcal {L} _ {M S E}
$$

where $\lambda$ is a weighting factor that balances the contributions of the two parts.

By incorporating the spatial proximity-aware loss function and the moving direction loss, FLIGHT2VEC can effectively encode the complex three-dimensional movement patterns in flight trajectories into the learned representations.

# Complexity Analysis

The complexity of the behavior-based patching is $O(n)$ , where n represents the length of the flight trajectory. Then, the complexity of patch projection and position embedding is $O(N \cdot S \cdot d_{m})$ , where N, S, $d_{m}$ denote the number of

patches, the size of each patch and the embedding dimension, respectively. Then, the backbone, i.e., the Transformer encoder, involves self-attention computation, resulting in a complexity of $O(N^{2} \cdot d_{m})$ . Finally, the complexity of the Linear Layer is $O(N \cdot d_{m})$ . In general, the time complexity of our framework is $O(N^{2} \cdot d_{m})$ , with the Transformer Encoder being the most computationally intensive component. In practice, the number of patches is much less the length of trajectory which makes FLIGHT2VEC efficient.

# Experiment

# Experimental Settings

In this section, we briefly introduce the datasets, experiment protocols, baselines and hyperparameter settings. The code and data are public available at https://github.com/liushuoer/FLIGHT2VEC. More details of the experimental settings are described in the Appendix.

Data Descriptions We conduct extensive experiments on two real-world datasets, the Swedish Civil Air Traffic Control (SCAT) (Nilsson and Unger 2023) and Aircraft Trajectory Classification Data for Air Traffic Management(ATFMTraj) (Phisannupawong, Damanik, and Choi 2024b).

Experimental Protocol We employ three representative tasks, i.e. flight trajectory prediction (FTP), flight recognition (FR), and anomaly detection (AD), to evaluate the representations learned from FLIGHT2VEC. For FTP, we predict the future trajectory in (1, 3, 15, 30, 60) different horizons and utilize Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE) and Mean Distance Error (MDE) as the evaluation metrics. For FR, we directly use the category of flight as ground truth and evaluate the performance with accuracy (ACC), precision (PRE), and recall (REC). For AD, we generate the synthetic anomalies according to the previous work (Guo et al. 2022b) and evaluate the performance of FLIGHT2VEC with AUC and AUPR. Following the settings of the previous work, we use the Mean Time Cost(MTC) metric to evaluate the computational performance.

Baselines As described in the experimental protocol, we employ three tasks to verify the performance of FLIGHT2VEC. For flight trajectory prediction, three representative methods, i.e. FlightBERT++ (Guo et al. 2024), LSTM+Attention (Guo et al. 2022a), and PatchTST (Nie et al. 2022) are compared. For anomaly detection methods, we use DMDN (Lijing, Weili, and Zhao 2021), and DDM (Guo et al. 2022b) for performance comparison. For flight trajectory recognition, we select SPIRAL (Lei et al. 2019) and ATSCC (Phisannupawong, Damanik, and Choi 2024a) as our baselines. Moreover, we also compare the ablations of FLIGHT2VEC to verify the superiority of the proposed techniques and study the parameter sensitivity to provide some insight in the use of FLIGHT2VEC.

Hyperparameters setting Our model utilizes the Transformer configuration from (Nie et al. 2022), which includes 3 layers with a model dimension of 256, and 16 attention heads with a dropout rate of 0.2. The binomial masking probability is set at 0.4. The dimension of the representation $p_{i}$ is set to 256. For training, the batch size is set to 256, and the AdamW optimizer is used with a learning rate of $1 \times 10^{-5}$ . The model is pre-trained for 100 epochs with a patch length of 32. All the experiments are conducted on the $2 \times$ NVIDIA 3090Ti.

# Effectiveness of FLIGHT2VEC

To evaluate the effectiveness of FLIGHT2VEC, we conduct experiments on the aforementioned three tasks and analyze the results of each task respectively.

Results of Flight Trajectory Prediction. We conduct FTP experiment on SCAT dataset and report the quantitative results in Table 1. According to the results, we have four observations. Firstly, the Transformer-based methods, such as PatchTST, FlightBERT++ and FLIGHT2VEC, achieve superior performance compared with the LSTM+Attention. Owing to the large parameter size and self-attention mechanism, Transformer-based methods have higher model capacity to model the long-term dependence for flight prediction. Secondly, FlightBERT++ is the most competitive baseline, which can predict the flight trajectory in a non-autoregressive manner, but it is also inferior to FLIGHT2VEC. This proves the effectiveness of activity density and moving trends modeled in FLIGHT2VEC. Thirdly, with the prediction horizon increasing, the performances of all methods are dropped. For example, the MDE of FlightBERT++ increases by approximately 237% when the horizon increases from 3 to 15. FLIGHT2VEC still beats all compared baselines on all prediction horizons. We attribute this to the effectiveness of moving direction loss which captures the 3D spatial motion trend of flight trajectory. Lastly, due to the fusion of some attributes of the trajectory points, FlightBERT++ achieves better performance than FLIGHT2VEC for short-term horizon predictions (in horizons 1 and 3). We use the same method in FlightBERT++ to integrate the attributes in FLIGHT2VEC and form FLIGHT2VEC +BE. As shown, FLIGHT2VEC +BE achieves the best performance across all prediction horizons, which also proves the scalability of our method.

Results of Anomaly Detection. The flight trajectory anomaly detection experiments are also conducted on SCAT dataset. The results are presented in Table 2. Due to the absence of real ground truth anomalies. We construct four types of anomalies, i.e., Successive Multipoint Anomaly (SMA), Horizontal Deviation (HD), Vertical Deviation (VD) and Go-Around following the existing work (Guo et al. 2022b). We train another MLP layer along with the learned representations of FLIGHT2VEC to achieve anomaly detection. All the methods are trained with the same training dataset. For evaluation metrics, we utilize AUPR in addition to AUC to handle the unbalanced anomaly classes. As shown, the performance of DMDN is inferior to DDM, which proves the density estimation method of DMDN can not learn representative features for detecting anomalies. FLIGHT2VEC achieves significant improvements in all

Table 1: The experimental results of FTP task. 

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Hor.</td><td colspan="3">MAE ↓</td><td colspan="3">MAPE (%) ↓</td><td colspan="3">RMSE ↓</td><td rowspan="2">MDE ↓</td></tr><tr><td>Lon</td><td>Lat</td><td>Alt</td><td>Lon</td><td>Lat</td><td>Alt</td><td>Lon</td><td>Lat</td><td>Alt</td></tr><tr><td rowspan="5">LSTM+Attention</td><td>1</td><td>0.0078</td><td>0.0075</td><td>2.45</td><td>0.0062</td><td>0.0289</td><td>0.38</td><td>0.0133</td><td>0.0148</td><td>7.18</td><td>1.28</td></tr><tr><td>3</td><td>0.0073</td><td>0.0069</td><td>3.09</td><td>0.0058</td><td>0.0241</td><td>0.47</td><td>0.0123</td><td>0.0146</td><td>9.86</td><td>1.23</td></tr><tr><td>15</td><td>0.0142</td><td>0.0152</td><td>9.13</td><td>0.0124</td><td>0.0543</td><td>1.97</td><td>0.0319</td><td>0.0367</td><td>20.48</td><td>1.75</td></tr><tr><td>30</td><td>0.0257</td><td>0.0296</td><td>18.66</td><td>0.0179</td><td>0.0922</td><td>3.79</td><td>0.0592</td><td>0.0695</td><td>40.16</td><td>2.94</td></tr><tr><td>60</td><td>0.0735</td><td>0.0698</td><td>50.25</td><td>0.0699</td><td>0.2314</td><td>13.78</td><td>0.1763</td><td>0.1998</td><td>102.06</td><td>6.79</td></tr><tr><td rowspan="5">PatchTST</td><td>1</td><td>0.0041</td><td>0.0039</td><td>1.47</td><td>0.0032</td><td>0.0136</td><td>0.32</td><td>0.0125</td><td>0.0134</td><td>8.04</td><td>0.79</td></tr><tr><td>3</td><td>0.0073</td><td>0.0072</td><td>2.98</td><td>0.0063</td><td>0.0256</td><td>0.61</td><td>0.0214</td><td>0.0231</td><td>8.72</td><td>1.34</td></tr><tr><td>15</td><td>0.0126</td><td>0.0129</td><td>8.03</td><td>0.0125</td><td>0.0532</td><td>1.77</td><td>0.0305</td><td>0.0322</td><td>19.94</td><td>1.61</td></tr><tr><td>30</td><td>0.0231</td><td>0.0248</td><td>14.99</td><td>0.0217</td><td>0.0962</td><td>3.48</td><td>0.0591</td><td>0.0672</td><td>38.88</td><td>2.70</td></tr><tr><td>60</td><td>0.0555</td><td>0.0572</td><td>24.65</td><td>0.0489</td><td>0.1978</td><td>7.87</td><td>0.1129</td><td>0.1376</td><td>72.15</td><td>6.50</td></tr><tr><td rowspan="5">FlightBERT++</td><td>1</td><td>0.0018</td><td>0.0018</td><td>1.16</td><td>0.0016</td><td>0.0067</td><td>0.21</td><td>0.0037</td><td>0.0116</td><td>13.11</td><td>0.32</td></tr><tr><td>3</td><td>0.0032</td><td>0.0032</td><td>2.33</td><td>0.0031</td><td>0.0119</td><td>0.43</td><td>0.0076</td><td>0.0133</td><td>12.88</td><td>0.59</td></tr><tr><td>15</td><td>0.0125</td><td>0.0118</td><td>7.48</td><td>0.0110</td><td>0.0429</td><td>1.38</td><td>0.0268</td><td>0.0329</td><td>22.91</td><td>1.99</td></tr><tr><td>30</td><td>0.0214</td><td>0.0213</td><td>12.97</td><td>0.0286</td><td>0.0916</td><td>2.46</td><td>0.0576</td><td>0.0671</td><td>48.55</td><td>4.44</td></tr><tr><td>60</td><td>0.0482</td><td>0.0497</td><td>29.87</td><td>0.0606</td><td>0.1666</td><td>6.02</td><td>0.1076</td><td>0.1319</td><td>79.25</td><td>7.24</td></tr><tr><td rowspan="5">FLIGHT2VEC</td><td>1</td><td>0.0019</td><td>0.0018</td><td>1.17</td><td>0.0015</td><td>0.0062</td><td>0.23</td><td>0.0038</td><td>0.0121</td><td>8.02</td><td>0.34</td></tr><tr><td>3</td><td>0.0034</td><td>0.0033</td><td>2.32</td><td>0.0029</td><td>0.0118</td><td>0.42</td><td>0.0075</td><td>0.0136</td><td>8.11</td><td>0.57</td></tr><tr><td>15</td><td>0.0124</td><td>0.0121</td><td>7.58</td><td>0.0109</td><td>0.0431</td><td>1.36</td><td>0.0259</td><td>0.0319</td><td>18.24</td><td>1.98</td></tr><tr><td>30</td><td>0.0196</td><td>0.0213</td><td>12.14</td><td>0.0212</td><td>0.0811</td><td>2.09</td><td>0.0436</td><td>0.0501</td><td>37.52</td><td>3.34</td></tr><tr><td>60</td><td>0.0381</td><td>0.0401</td><td>25.67</td><td>0.0403</td><td>0.1216</td><td>3.72</td><td>0.0836</td><td>0.1019</td><td>56.45</td><td>5.14</td></tr><tr><td rowspan="5">FLIGHT2VEC +BE</td><td>1</td><td>0.0017</td><td>0.0017</td><td>1.15</td><td>0.0015</td><td>0.0061</td><td>0.19</td><td>0.0036</td><td>0.0112</td><td>8.01</td><td>0.29</td></tr><tr><td>3</td><td>0.0029</td><td>0.0029</td><td>1.92</td><td>0.0028</td><td>0.0116</td><td>0.39</td><td>0.0036</td><td>0.0109</td><td>7.83</td><td>0.55</td></tr><tr><td>15</td><td>0.0122</td><td>0.0112</td><td>7.43</td><td>0.0101</td><td>0.0402</td><td>1.31</td><td>0.0243</td><td>0.0312</td><td>19.81</td><td>1.44</td></tr><tr><td>30</td><td>0.0192</td><td>0.0198</td><td>11.89</td><td>0.0203</td><td>0.0699</td><td>2.03</td><td>0.0432</td><td>0.0498</td><td>33.88</td><td>2.24</td></tr><tr><td>60</td><td>0.0371</td><td>0.0392</td><td>25.45</td><td>0.0401</td><td>0.1209</td><td>3.62</td><td>0.0819</td><td>0.0932</td><td>54.77</td><td>5.04</td></tr></table>

Table 2: The experimental results of anomaly detection task. 

<table><tr><td rowspan="2">Method</td><td colspan="4">AUC</td><td colspan="4">AUPR</td></tr><tr><td>SMA</td><td>HD</td><td>VD</td><td>Go-Around</td><td>SMA</td><td>HD</td><td>VD</td><td>Go-Around</td></tr><tr><td>DMDN</td><td>0.7498</td><td>0.7444</td><td>0.7354</td><td>0.7100</td><td>0.7416</td><td>0.7498</td><td>0.7326</td><td>0.7210</td></tr><tr><td>DDM</td><td>0.8509</td><td>0.8411</td><td>0.8324</td><td>0.8200</td><td>0.8122</td><td>0.8112</td><td>0.7923</td><td>0.7890</td></tr><tr><td>FLIGHT2VEC</td><td>0.9201</td><td>0.9178</td><td>0.9127</td><td>0.9050</td><td>0.9072</td><td>0.9059</td><td>0.9017</td><td>0.9005</td></tr></table>

Table 3: The experimental results of flight recognition task. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">RKSla</td><td colspan="3">RKSId</td><td colspan="3">ESSA</td><td colspan="3">LSZH</td></tr><tr><td>ACC</td><td>PRE</td><td>REC</td><td>ACC</td><td>PRE</td><td>REC</td><td>ACC</td><td>PRE</td><td>REC</td><td>ACC</td><td>PRE</td><td>REC</td></tr><tr><td>SPIRAL</td><td>0.8344</td><td>0.8301</td><td>0.8302</td><td>0.9946</td><td>0.9942</td><td>0.9932</td><td>0.8503</td><td>0.8496</td><td>0.8408</td><td>0.9173</td><td>0.9172</td><td>0.9124</td></tr><tr><td>ATSCC</td><td>0.9946</td><td>0.9951</td><td>0.9911</td><td>0.9987</td><td>0.9981</td><td>0.9971</td><td>0.9990</td><td>0.9989</td><td>0.9980</td><td>0.9977</td><td>0.9979</td><td>0.9971</td></tr><tr><td>Flight2Vec</td><td>0.9947</td><td>0.9952</td><td>0.9943</td><td>0.9988</td><td>0.9980</td><td>0.9970</td><td>0.9990</td><td>0.9986</td><td>0.9982</td><td>0.9981</td><td>0.9980</td><td>0.9980</td></tr></table>

metrics, proving that the learned representations can capture the local movement patterns of the flight trajectory. For Go-Around anomaly, FLIGHT2VEC achieves 10.4% and 14.1% improvements of AUC and AUPR, compared with the most competitive baseline DDM. Moreover, the results of FLIGHT2VEC consistently outperform the baselines on four anomaly types, indicating that the proposed behavior-adaptive patching and moving direction loss can encourage the representation model to learn useful features.

Results of Flight Recognition. We conduct the flight recognition experiment on ATFMTraj dataset and present

the results in Table 3. In ATFMTraj, the flight trajectories are categorized into four classes, i.e., RKSla, RKSId, ESSA and LSZH, by the airports of take-off and landing. As illustrated, the performances of all methods are relatively high. The best accuracy of the four classes are over 99%. For the compared baselines, the performance of ATSCC significantly outperforms SPIRAL. The precision increased from 0.8301 to 0.9946 on RKSla. This phenomenon indicates the segment-patch framework in ATSCC can better represent the flight trajectory compared with using original data directly. FLIGHT2VEC utilizes the behavior-adaptive patching

Table 4: Comparison of the Computational Performance. 

<table><tr><td>Task</td><td>Methods</td><td>Parameters (M)</td><td>MTC (ms)</td></tr><tr><td rowspan="4">FTP</td><td>LSTM+Attention</td><td>0.90</td><td>48.96</td></tr><tr><td>PatchTST</td><td>1.61</td><td>2.20</td></tr><tr><td>FlightBERT++</td><td>29.96</td><td>7.21</td></tr><tr><td>FLIGHT2VEC</td><td>1.86</td><td>3.12</td></tr><tr><td rowspan="3">FR</td><td>SPIRAL</td><td>-</td><td>0.76</td></tr><tr><td>ATSCC</td><td>0.90</td><td>43.89</td></tr><tr><td>FLIGHT2VEC</td><td>1.85</td><td>3.11</td></tr><tr><td rowspan="3">AD</td><td>DMDN</td><td>0.61</td><td>47.96</td></tr><tr><td>DDM</td><td>0.11</td><td>6.53</td></tr><tr><td>FLIGHT2VEC</td><td>1.85</td><td>3.11</td></tr></table>

Table 5: The experimental results of the ablation study. 

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Hor.</td><td colspan="3">MAE ↓</td><td colspan="3">MAPE (%) ↓</td><td colspan="3">RMSE ↓</td></tr><tr><td>Lon</td><td>Lat</td><td>Alt</td><td>Lon</td><td>Lat</td><td>Alt</td><td>Lon</td><td>Lat</td><td>Alt</td></tr><tr><td rowspan="2">w/o PD</td><td>1</td><td>0.0032</td><td>0.0032</td><td>1.65</td><td>0.0029</td><td>0.0098</td><td>0.54</td><td>0.0052</td><td>0.0145</td><td>14.55</td></tr><tr><td>15</td><td>0.0119</td><td>0.0117</td><td>7.94</td><td>0.0117</td><td>0.0512</td><td>1.62</td><td>0.0289</td><td>0.0276</td><td>19.12</td></tr><tr><td rowspan="2">w/o MD</td><td>1</td><td>0.0022</td><td>0.0022</td><td>1.39</td><td>0.0019</td><td>0.0072</td><td>0.29</td><td>0.0042</td><td>0.0133</td><td>13.95</td></tr><tr><td>15</td><td>0.0144</td><td>0.0141</td><td>7.88</td><td>0.0162</td><td>0.0438</td><td>1.42</td><td>0.0279</td><td>0.0382</td><td>23.11</td></tr><tr><td rowspan="2">our</td><td>1</td><td>0.0019</td><td>0.0018</td><td>1.17</td><td>0.0015</td><td>0.0062</td><td>0.23</td><td>0.0038</td><td>0.0121</td><td>13.21</td></tr><tr><td>15</td><td>0.0124</td><td>0.0121</td><td>7.58</td><td>0.0109</td><td>0.0431</td><td>1.36</td><td>0.0259</td><td>0.0319</td><td>18.24</td></tr></table>

Transformer and achieves comparable results with the performance of ATSCC.

# Efficiency of FLIGHT2VEC

To verify the efficiency of FLIGHT2VEC, we report the Mean Time Cost (MTC) of representation generation along with the model size in Table 4. As shown, the computational time of FLIGHT2VEC is better than, at least comparable with, all the compared baselines. For the FTP task, FLIGHT2VEC is the fast method expect for PatchTST. However, PatchTST is a light model that is not specifically designed for flight trajectories and performs poorer than FLIGHT2VEC. FlightBERT++ is inferior to FLIGHT2VEC, because FlightBERT++ is a encoder-decoder framework while FLIGHT2VEC is a decoder-only architecture. The parameters in FLIGHT2VEC are much less than FlightBERT++, leading to better computational efficiency. For FR, SPIRAL has minimal computational time. This is because SPIRAL is not a deep learning method. Compared with the SOTA method, FLIGHT2VEC is 10 times faster than ATSCC, i.e., from 43.89 ms (ATSCC) to 3.11 ms (FLIGHT2VEC). For AD task, FLIGHT2VEC demonstrate substantial improvements in computational performance compared to the baselines.

# Ablation Study

We compare FLIGHT2VEC with two ablations to analyze the effectiveness of the proposed components. We remove the proposed behavior-based patching, randomly sample points at fixed intervals and divide the patch to obtain w/o PD. We obtain w/o MD by removing moving direction loss.

Due to the space limit, we only report the experiment on FTP task and the results are shown in Table 5. We observe:

![](images/005c7af729ff18db0541e80f2a1e6cce4ad89820e11a335eb7e545cad73354f9.jpg)  
Figure 5: MDE scores with varying model parameters.

(1) Comparing the results of FLIGHT2VEC with w/o MD, we observe the moving direction loss can model the spatial continuity flight trajectories. For example, the RMSE improves from 18.24 to 23.11 on altitude. (2) From the results of w/o PD and FLIGHT2VEC, we can conclude that the activity patching mechanism capacity to represent flight trajectory is more effective than a fixed patch. (3) FLIGHT2VEC achieves the best performance compared to all ablations, which proves the effectiveness of the proposed techniques.

# Sensitivity Analysis

We analyze the impact of selecting different patch sizes and representation dimensions on different tasks. We ranged the patch size from 8 to 48 and the representation dimension from 64 to 512, and then show the average performance of these configurations on different tasks in Figure 5.

With the increase of patch size, the performance of FLIGHT2VEC first increases and then drops. FLIGHT2VEC achieves the best performance with the patch size of 32 on both FTP and AD tasks. For FR task, the optimal performance is achieved with the largest patch size of 48. This is because the FR task pays more attention to the global movement patterns of the flight trajectory.

For the change of embedding dimension, the performances of the three tasks have similar patterns. FLIGHT2VEC achieves the best performance on the dimension size of 256. This phenomenon indicates FLIGHT2VEC is robust to different embedding dimension sizes.

# Conclusion

In this paper, we present FLIGHT2VEC which is the first unified framework specifically designed for flight trajectory representation learning. By addressing the challenges of unbalanced behavior density and 3D spatial continuity, FLIGHT2VEC enhances the representation of flight trajectories. Our approach introduces an adaptive behavior-based patching mechanism and a direction loss, significantly improving performance across tasks like trajectory prediction, anomaly detection, and flight monitoring. Extensive experiments across multiple task demonstrate that FLIGHT2VEC not only significantly outperforms existing methods but also sets a new benchmark in the field.

# Acknowledgment

This work has been supported by the National Natural Science Foundation of China under Grant No. 62472405. This work is also sponsored by CCF-DiDi GAIA Collaborative Research Funds for Young Scholars.

# References

Chen, P.; Zhang, Y.; Cheng, Y.; Shu, Y.; Wang, Y.; Wen, Q.; Yang, B.; and Guo, C. 2024. Multi-scale transformers with adaptive pathways for time series forecasting. In International Conference on Learning Representations.   
Chen, Y.; Li, X.; Cong, G.; Bao, Z.; Long, C.; Liu, Y.; Chandran, A. K.; and Ellison, R. 2021. Robust road network representation learning: When traffic patterns meet traveling semantics. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 211–220.   
Fang, W.; Wang, Y.; Yan, W.; and Lin, C. 2021. Symbolic flight action recognition based on neural networks. Syst. Eng. Electron, 13: 963–969.   
Fernández, A.; Martínez, D.; Hernández, P.; Cristóbal, S.; Schwaiger, F.; Nunez, J. M.; and Ruiz, J. M. 2019. Flight data monitoring (FDM) unknown hazards detection during approach phase using clustering techniques and AutoEncoders. Proceedings of the Ninth SESAR Innovation Days, Athens, Greece, 2–5.   
Fu, T.-Y.; and Lee, W.-C. 2020. Trembr: Exploring road networks for trajectory representation learning. ACM Transactions on Intelligent Systems and Technology (TIST), 11(1): 1–25.   
Guo, D.; Wu, E. Q.; Wu, Y.; Zhang, J.; Law, R.; and Lin, Y. 2022a. FlightBERT: binary encoding representation for flight trajectory prediction. IEEE Transactions on Intelligent Transportation Systems, 24(2): 1828–1842.   
Guo, D.; Zhang, Z.; Yan, Z.; Zhang, J.; and Lin, Y. 2024. FlightBERT++: A Non-autoregressive Multi-Horizon Flight Trajectory Prediction Framework. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 127–134.   
Guo, Z.; Yin, C.; Zeng, W.; Tan, X.; and Bao, J. 2022b. Data-driven method for detecting flight trajectory deviation anomaly. Journal of Aerospace Information Systems, 19(12): 799–810.   
Huang, Q.; Shen, L.; Zhang, R.; Cheng, J.; Ding, S.; Zhou, Z.; and Wang, Y. 2024. Hdmixer: Hierarchical dependency with extendable patch for multivariate time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 12608–12616.   
Jiang, J.; Pan, D.; Ren, H.; Jiang, X.; Li, C.; and Wang, J. 2023. Self-supervised trajectory representation learning with temporal regularities and travel semantics. In 2023 IEEE 39th international conference on data engineering (ICDE), 843–855. IEEE.   
Jing, Q.; Liu, S.; Fan, X.; Li, J.; Yao, D.; Wang, B.; and Bi, J. 2022. Can Adversarial Training benefit Trajectory Representation? An Investigation on Robustness for Trajectory

Similarity Computation. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 905–914.   
Lei, Q.; Yi, J.; Vaculin, R.; Wu, L.; and Dhillon, I. S. 2019. Similarity preserving representation learning for time series clustering. In Proceedings of the 28th International Joint Conference on Artificial Intelligence, 2845–2851.   
Li, X.; Zhao, K.; Cong, G.; Jensen, C. S.; and Wei, W. 2018. Deep representation learning for trajectory similarity computation. In 2018 IEEE 34th international conference on data engineering (ICDE), 617–628. IEEE.   
Li, Z.; Li, S.; and Yan, X. 2024. Time series as images: Vision transformer for irregularly sampled time series. Advances in Neural Information Processing Systems, 36.   
Lijing, C.; Weili, Z.; and Zhao, Y. 2021. An Aircraft Trajectory Anomaly Detection Method Based on Deep Mixture Density Network. Transactions of Nanjing University of Aeronautics & Astronautics, 38(5).   
Liu, Y.; and Hansen, M. 2018. Predicting aircraft trajectories: A deep generative convolutional recurrent neural networks approach. arXiv preprint arXiv:1812.11670.   
Memarzadeh, M.; Matthews, B.; and Templin, T. 2022. Multiclass Anomaly Detection in Flight Data Using Semi-Supervised Explainable Deep Learning Model. Journal of Aerospace Information Systems, 19(2): 83–97.   
Memarzadeh, M.; Matthews, B. L.; and Weckler, D. I. 2023. Anomaly Detection in Flight Operational Data Using Deep Learning. In System-Wide Safety Technical Challenge 1 Close Out Event.   
Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730.   
Nilsson, J.; and Unger, J. 2023. Swedish civil air traffic control dataset. Data in brief, 48: 109240.   
Olive, X.; and Basora, L. 2020. Detection and identification of significant events in historical aircraft trajectory data. Transportation Research Part C: Emerging Technologies, 119: 102737.   
Phisannupawong, T.; Damanik, J. J.; and Choi, H.-L. 2024a. Aircraft Trajectory Segmentation-based Contrastive Coding: A Framework for Self-supervised Trajectory Representation. arXiv:2407.20028.   
Phisannupawong, T.; Damanik, J. J.; and Choi, H.-L. 2024b. ATFMTraj: Aircraft Trajectory Classification Data for Air Traffic Management. https://huggingface.co/datasets/petchthwr/ATFMTraj.   
Qian, T.; Li, J.; Chen, Y.; Cong, G.; Sun, T.; Wang, F.; and Xu, Y. 2024. Context-Enhanced Multi-View Trajectory Representation Learning: Bridging the Gap through Self-Supervised Models. arXiv preprint arXiv:2410.13196.   
Qin, K.; Wang, Q.; Lu, B.; Sun, H.; and Shu, P. 2022. Flight anomaly detection via a deep hybrid model. Aerospace, 9(6): 329.   
Wang, S.; Wu, H.; Shi, X.; Hu, T.; Luo, H.; Ma, L.; Zhang, J. Y.; and Zhou, J. 2024. Timemixer: Decomposable multiscale mixing for time series forecasting. arXiv preprint arXiv:2405.14616.

Wu, X.; Yang, H.; Chen, H.; Hu, Q.; and Hu, H. 2022. Long-term 4D trajectory prediction using generative adversarial networks. Transportation Research Part C: Emerging Technologies, 136: 103554.   
Yang, P.; Wang, H.; Zhang, Y.; Qin, L.; Zhang, W.; and Lin, X. 2021. T3s: Effective representation learning for trajectory similarity computation. In 2021 IEEE 37th International Conference on Data Engineering (ICDE), 2183–2188. IEEE.   
Yang, Y.; Zhang, C.; Zhou, T.; Wen, Q.; and Sun, L. 2023. Dcdetector: Dual attention contrastive representation learning for time series anomaly detection. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 3033–3045.   
Yao, D.; Cong, G.; Zhang, C.; and Bi, J. 2019. Computing trajectory similarity in linear time: A generic seed-guided neural metric learning approach. In 2019 IEEE 35th international conference on data engineering (ICDE), 1358–1369. IEEE.   
Yao, D.; Hu, H.; Du, L.; Cong, G.; Han, S.; and Bi, J. 2022. TrajGAT: A graph-based long-term dependency modeling approach for trajectory similarity computation. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, 2275–2285.   
Yao, D.; Zhang, C.; Zhu, Z.; Hu, Q.; Wang, Z.; Huang, J.; and Bi, J. 2018. Learning deep representation for trajectory clustering. Expert Systems, 35(2): e12252.   
Yao, D.; Zhang, C.; Zhu, Z.; Huang, J.; and Bi, J. 2017. Trajectory clustering via deep representation learning. In 2017 international joint conference on neural networks (IJCNN), 3880–3887. IEEE.   
Zhang, X.; Yin, Z.; Liu, F.; and Huang, Q. 2016. Data mining method for aircraft maneuvering division. J. Northwest. Polytech. Univ, 34: 33–40.   
Zhang, Z.; Guo, D.; Zhou, S.; Zhang, J.; and Lin, Y. 2023. Flight trajectory prediction enabled by time-frequency wavelet transform. Nature Communications, 14(1): 5258.

# Appendix

# Details of datasets.

- SCAT: The SCAT dataset contains detailed data of almost 170,000 flights from October 2016 to September 2017, as well as weather forecasts and airspace data collected from the air traffic control system in the Swedish flight information region. The dataset is limited to scheduled flights, excluding military and private aircraft. After removing trajectories that only flew at a single altitude, we obtained a final dataset of 100,383 trajectories.   
- ATFMTraj: The dataset has classification labels based on aeronautical publications. The dataset is divided into four sub-datasets based on the airport and flight plan (departing or arriving). The arrival and departure datasets of Incheon International Airport (ICAO code: RKSI) are denoted as RKSIs and RKSIds, respectively. The arrival datasets of Stockholm Arlanda Airport (ICAO code: ESSA) and Zurich Airport (ICAO code: LSZH) are denoted as ESSA and LSZH, respectively.

# Details of Compared Baselines.

We compare FLIGHT2VEC with seven baselines, which are divided into three groups based on downstream tasks: flight trajectory prediction (FTP) methods, flight trajectory recognition methods, and anomaly detection methods. For the first group, we employ three baselines. We compare the State-of-the-Art (SOTA) model FlightBert++(Guo et al. 2024) in the field of flight trajectory prediction and the model LSTM+Attention(Guo et al. 2022a) based on the Seq2Seq architecture according to its experimental settings. We also compare the patch-based Transformer model PatchTST(Nie et al. 2022) and the model FLIGHT2VEC +BE that introduces the additional features proposed by FlightBert++. For flight trajectory recognition methods, we select SPIRAL(Lei et al. 2019) and ATSCC(Phisannupawong, Damanik, and Choi 2024a), as our baselines. For anomaly detection methods, we use DMDN (Lijing, Weili, and Zhao 2021), and DDM (Guo et al. 2022b) for performance comparison.

# Experimental Protocol.

In our experiments, each dataset is divided into two subsets: the first $50\%$ of timestamps is denoted as the training set, while the latter $50\%$ is denoted as the test set. We utilize three downstream tasks to evaluate the performance of FLIGHT2VEC. For trajectory prediction task, we predict the results in a single inference process for multiple time steps on the SCAT data set. The experimental results are divided into five (1, 3, 15, 30, 60) different horizons, corresponding to 20 seconds, 1, 5, 10 and 20 minutes trajectories in the future. For flight trajectory recognition task, we directly used the real label to conduct the experiments. Due to privacy concerns, Existing flight trajectory dataset either have no labeled anomaly edges or only have one anomaly type. To verify the ability of FLIGHT2VEC on various anomaly types, we follow the experiments of (Guo et al. 2022b) and generate three kinds of systematic anomaly

types, i.e., Successive Multipoint Anomaly (SMA), Horizontal Deviation (HD), Vertical Deviation (VD) and Go-Around for SCAT datasets. SMA refers to multiple consecutive trajectory points in a flight that deviate from the planned flight path. This anomaly may indicate a problem with the aircraft's control or navigation. HD refers to the aircraft's deviation from the planned horizontal path during flight. Especially during the approach phase, HD anomalies may affect flight safety. VD refers to the aircraft's deviation from the expected vertical flight trajectory. The aircraft fails to accurately follow the glide path, which may cause the aircraft to be in a dangerous state of being too high or too low during approach or landing. Go-Around refers to aborting the landing and re-entering the route when the aircraft cannot land safely. In the anomaly detection task, in order to simulate the situation with few anomaly in real-world scenarios, we only add 5% anomaly lable data to the test set.