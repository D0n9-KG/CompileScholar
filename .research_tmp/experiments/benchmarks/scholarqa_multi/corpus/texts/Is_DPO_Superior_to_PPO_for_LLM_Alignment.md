# Watch Your Step: Optimal Retrieval for Continual Learning at Scale

Truman Hickok

University of Texas at San Antonio

truman.hickok@utsa.edu

Dhireesha Kudithipudi

University of Texas at San Antonio

dk@utsa.edu

# Abstract

In continual learning, a model learns incrementally over time while minimizing interference between old and new tasks. One of the most widely used approaches in continual learning is referred to as replay. Replay methods support interleaved learning by storing past experiences in a replay buffer. Although there are methods for selectively constructing the buffer and reprocessing its contents, there is limited exploration of the problem of selectively retrieving samples from the buffer. Current solutions have been tested in limited settings and, more importantly, in isolation. Existing work has also not explored the impact of duplicate replays on performance. In this work, we propose a framework for evaluating selective retrieval strategies, categorized by simple, independent class- and sample-selective primitives. We evaluated several combinations of existing strategies for selective retrieval and present their performances. Furthermore, we propose a set of strategies to prevent duplicate replays and explore whether new samples with low loss values can be learned without replay. In an effort to match our problem setting to a realistic continual learning pipeline, we restrict our experiments to a setting involving a large, pre-trained, open vocabulary object detection model, which is fully fine-tuned on a sequence of 15 datasets.

# 1. Introduction

The field of continual learning represents a cornerstone in the evolution of artificial intelligence, as continual learning algorithms open the door to agents that can efficiently self-improve in the face of failure and uncertainty, along with being able to adapt to changing requirements $[14, 46]$ . Fundamentally, continual learning algorithms seek to allow machine learning practitioners to sequentially train models on task-specific datasets following a broad, expensive pre-training phase, without sacrificing performance on pretrained or downstream tasks. This setting differs from the current mainstream paradigm of either superficially adding new capabilities through in-context learning $[5, 11]$ and retrieval augmentation $[26]$ or, much worse, appending newtask data to the pre-training dataset and retraining the model from scratch [6]. Therefore, continual learning algorithms uniquely allow new information to reconfigure the model's existing representations such that performance can be maximized on both novel and existing tasks [4].

Replay-based continual learning algorithms selectively store or generate inputs from previous tasks so that they can be mixed, or "interleaved", with data from new tasks $[17]$ . A key set of algorithms in this ecosystem is selective retrieval algorithms, which retrieve samples from the replay buffer such that they are prototypical and representative of previous tasks $[15]$ or are likely to be interfered with by samples from new tasks $[1, 36]$ . However, up to this point, algorithms for selective retrieval have been evaluated in isolated and limited problem settings, which means that they have not been directly compared to each other and have been tested in settings with questionable applicability to modern continual learning tasks. Moreover, two basic properties of these algorithms have gone unexamined to this point: the manner in which duplicate replays should be avoided and whether each new sample should warrant replay regardless of the magnitude of its loss.

In this paper, we expand on these works; we start with a broad question: what is the best algorithm for selective retrieval? We begin answering this question by organizing and extending existing algorithmic primitives for selective retrieval, which are the building blocks of the algorithms under study and are separated into class-selective and sample-selective primitives. Next, we specify existing and novel combinations of primitives and compare their performance on a unique benchmark meant to match a modern, large scale continual learning setting (Figure 1, bottom). We jointly execute this comparison with a comprehensive sweep of deduplication schedules, which determine how long training takes without allowing duplicate replays. Interestingly, we observe that two of the simplest primitives, implemented as standalone solutions, perform better than all other algorithms, including a combination of these primitives themselves. As expected, deduplication is necessary for almost every algorithm, but only up to a certain shared point.

Motivated by recent work showing abrupt representation drift upon entering a new task $[8, 35]$ , we then experiment with the idea of reducing the impact of replay when new samples' losses are below a threshold, along with reducing the number of replays in these cases. This procedure reflects the hypothesis that reducing the impact of replay for new samples with the smallest gradients can lead to extra plasticity for new tasks, without incurring more forgetting of previous tasks. Here, we find that applying this procedure rapidly leads to forgetting previous tasks, as the case where only 3% of new samples are below the loss threshold leads to significant drops in performance.

Finally, we provide analyses regarding the distributions over samples and classes produced by our algorithms, as well as forgetting dynamics according to properties of our datasets. In summary, we:

1. Compare algorithms for selective retrieval, demonstrating that certain algorithms outperform all others, including combinations of the algorithms themselves   
2. Compare different intervals over which no duplicate replays are allowed, demonstrating that duplicate replays should be prevented for each downstream dataset   
3. Show that only using replay for new samples with high losses quickly leads to an unacceptable drop in overall performance   
4. Analyze distributions produced by the best algorithms and discuss dataset-dependent forgetting dynamics

# 2. Related Work

Continual Learning: The space of solutions in continual learning can be organized into three types of methods: regularization, model expansion, and replay [14]. Regularization-based methods involve adding terms to the model's loss function to prevent the model from overwriting existing information, and can be executed in the space of weights or predictions [3, 41]. Model expansion involves adding new parameters to the network for each new task [34]. Replay-based methods involve sampling from a stored subset of data from previous tasks or from a generative model of previous tasks and interleaving data from new tasks with data from previous tasks [7, 17].

Almost all existing work in continual learning has been executed in the setting where a model is randomly initialized before being trained sequentially on balanced, independent subsets of a single dataset (Figure 1, top) [10, 43]. On the other hand, a more recent line of work has considered the setting where a model is initialized after broad pretraining and then trained in the same fashion as the original setting (Figure 1, middle) [12, 20, 24, 31]. This scenario is quite different from what modern machine learning pipelines require, as they need to continuously evaluate and preserve the extensive capabilities of the model during the learning process. As the field has progressed, several recent

![](images/810c84da1ee8848aedd9f002a83f72c24a222effd5e52b0fc8461129c9d3ac2d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["perform well on all tasks"] --> B["Dataset 1,0"]
    A --> C["Dataset 1,1"]
    A --> D["Dataset 1,2"]
    A --> E["..."]
    A --> F["Dataset 1,N"]
    G["Broad Pre-Training"] --> H["Dataset 1,0"]
    G --> I["Dataset 1,1"]
    G --> J["..."]
    G --> K["Dataset 1,N"]
    L["Broad Pre-Training"] --> M["Dataset 1"]
    L --> N["Dataset 2"]
    L --> O["..."]
    L --> P["Dataset N"]
```
</details>

Figure 1. The problem settings of continual learning research. Top: the original and most common setup, where dataset is divided into N balanced subsets and the model is trained sequentially on each subset. Middle: another common setup, which is the same as above except the model is pre-trained; forgetting on pre-training tasks is ignored. Bottom: our setup, where a pre-trained model is sequentially trained on OOD datasets and forgetting on pre-training tasks is minimized.

works address this issue [13, 19, 28] (Figure 1, bottom).

Replay Methods: Replay algorithms can be organized into four distinct branches, which are not mutually exclusive. The first branch is concerned with how information from past tasks is represented. This class of methods varies along two axes: whether the data from previous tasks is stored $[7]$ or generated $[40, 44]$ and whether inputs $[7]$ or activations $[16, 44]$ are being represented $[17]$ . The second branch is concerned with how replayed samples are reprocessed, and includes regularization terms such as logit distillation $[7]$ and gradient episodic memory (GEM) $[9]$ . The third branch is concerned with selectively storing information from previous tasks and encourages the samples in the replay buffer to be diverse $[2, 42]$ and "easy" $[18]$ . The fourth branch is concerned with selectively retrieving samples from the replay buffer and is the focus of this work.

Selective Retrieval for Replay: While the vast majority of work in continual learning has used balanced or fully random retrieval during replay $[7, 17]$ , some selective retrieval algorithms have recently been shown to outperform these baselines. Adversarial Shapley Experience Replay (ASER) $[39]$ proposes a scoring mechanism over samples in the replay buffer which balances terms representing how adversarial each sample is with respect to the current batch and how representative each sample is of other samples in the replay buffer. ASER was evaluated relative to MIR $[1]$ and random retrieval on Split-CIFAR and Split-Mini-ImageNet benchmarks in the traditional continual learning environment (Figure 1, top). Similarity-Weighted Interleaved Learning (SWIL) $[36]$ proposes a weighted sampling distribution over classes according to how similar

each class's prototype is to samples in new batches. SWIL was only evaluated relative to balanced retrieval and was tested on CIFAR-10 and CIFAR-100 where 90% of the classes are learned, then a single new class is learned with replay. "Gradually Select Less Prototypical" (GRASP) [15] proposes a weighted sampling distribution over samples according to how prototypical each sample is with respect to its corresponding class prototype. Although GRASP was evaluated in both a modern setting and relative to a slew of simple retrieval methods, it was not compared to SWIL or ASER. In addition to being tested in isolation, none of these methods considers the problem of deduplication or the prospect of loss-conditioned retrieval.

These three algorithms are the focus of this work and will be organized and extended (Section 4) then compared (Section 5) and analyzed (Section 6).

# 3. Problem Setting

We consider the general continual learning setting where the learner is initialized as a pre-trained network and is then fine-tuned on a sequence of datasets (Figure 1, bottom). Formally, after initializing parameters $\theta_{\mathrm{pt}}$ using dataset $D_{\mathrm{pt}}$ , we sequentially fine-tune the network on $N$ datasets, which we refer to as $D_{\mathrm{cl}} = \{D_{\mathrm{cl},1}, D_{\mathrm{cl},2}, ..., D_{\mathrm{cl,N}}\}$ . Throughout this process, we aim to maintain (or improve) performance on all pre-training tasks while also improving performance on $D_{\mathrm{cl}}$ as much as possible, which we accomplish by selectively retrieving and interleaving samples from the replay buffer, which is a subset of $D_{\mathrm{pt}}$ . We do not add samples from $D_{\mathrm{cl}}$ to the buffer. When training without loss adaptivity, we append samples from the replay buffer to the current batch such that there is a 1:1 ratio between new samples and replay samples.

# 4. Methods

In this section, we first formalize the principal objects of this paper: the primitives which constitute algorithms for selective retrieval during replay. We then describe our set of deduplication schedules, loss-thresholded replay, and our replay buffer selection algorithm. See Table 2 for an overview of all combinations of primitives evaluated in our main experiments.

All of our experiments used the OWL-ViT foundation model (Figure 7), which is a CLIP model adapted to object detection by removing the final pooling layer of the vision transformer and attaching a lightweight class embedding and box prediction head. For each image, the class embedding head outputs a tensor of size $(T,E)$ , where $T$ is the number of tokens processed by the vision transformer and $E$ is the class embedding size. The box prediction head outputs a tensor of size $(T,4)$ [30]. See Figure 2 for a depiction of the contents of our replay buffer; note that the term "prototype" refers to the average class embedding for a given class.

![](images/82c5277f8442b52d8d31ed339206b3327a587cf7de61272fa1645c19ed3f683b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Replay Buffer"] --> B["Class"]
    B --> C["Prototypes"]
    B --> D["Sample"]
    C --> E["Color Block 1"]
    C --> F["Color Block 2"]
    C --> G["Color Block 3"]
    C --> H["Color Block 4"]
    C --> I["Color Block 5"]
    C --> J["Color Block 6"]
    C --> K["Color Block 7"]
    C --> L["Color Block 8"]
    C --> M["Color Block 9"]
    C --> N["Color Block 10"]
    D --> O["Image: CQ in a video camera icon"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#ffc,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#ffc,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#ffc,stroke:#333
```
</details>

Figure 2. An overview of the contents of our replay buffer. Each image is stored with its top-k class embeddings and their corresponding query embeddings. Note that classes are overlapping in terms of the samples they contain, as most samples contain instances of multiple classes.

# 4.1. Class Selection Primitives

Class selection primitives are those that determine the classes to be sampled from for each new batch. Typically, this means producing a different probability distribution over classes for each image in the batch, then sampling a single class for each. In this work, we evaluate two class-selection primitives.

Balanced: Classes are sampled equally and deterministically. For example, if we are currently retrieving four samples from the replay buffer and the last sampled class was class 10, we would retrieve a sample from classes 11 through 14. If we were to retrieve four more samples for the next batch, we would retrieve a sample from classes 15 through 18.

Similarity-Weighted [36]: Classes are sampled in proportion to their prototypes' similarities with new samples' embeddings (See Section 10 of the Supplementary Material for a full description of the algorithm). In general, we make two important modifications with respect to [36], since we adapt the original formulation to operate fully online (on the level of batches) and also use a hyperparameter to control the entropy of the similarity-weighted distribution over classes.

# 4.2. Sample Selection Primitives

Sample selection primitives are those that produce distributions over samples in the replay buffer. In this work, we evaluate three of them. Note that our first two sample selection primitives assume a single class to be sampled from (Figure 2) while our third sample selection primitive does not make this assumption, only assuming access to a candidate set of samples.

Uniform: Each sample in the selected class has an equal probability of being selected.

Prototype-Weighted [15]: Each sample in the selected class has a probability in inverse proportion to its distance to its corresponding class prototype. We begin computing each sample's "score" by taking the minimum cosine distance between its top-k class embeddings for that class and the class prototype, where k is set as the ground-truth number of instances for the class within the sample's image. We finalize each sample's score by raising its distance to a negative power, which is controlled by a hyperparameter. Thus, the full equation for computing a sample's score is:

$$
s _ {i} = \min _ {j \in k} \left(1 - \frac {\mathbf {e} _ {c} \cdot \mathbf {e} _ {j}}{\| \mathbf {e} _ {c} \| \| \mathbf {e} _ {j} \|}\right) ^ {- w} \tag {1}
$$

where $e_{j}$ is a class embedding from the sample, $e_{c}$ is the class prototype, and w controls the entropy of the distribution of scores across samples.

We then convert each sample's score into a probability using:

$$
p _ {i} = \frac {s _ {i}}{\sum_ {s _ {m} \in N _ {c}} s _ {m}} \tag {2}
$$

where $N_{c}$ represents the number of samples containing an instance of class $c$ .

Adversarial Shapley Value [39]: Each sample in a selected candidate set is scored according to the Adversarial Shapley Value (ASV), which is derived from the KNN Shapley Value (KNN-SV) [21]. The KNN Shapley Value is an efficient approximation of the Shapley Value [38] used in data valuation and can be understood as an algorithm for attributing predictions made by a model for individual samples in an evaluation set (e.g. a validation set or, in this case, a batch of new data) to individual samples in a training set (which, in our case, is instead the candidate set drawn from our replay buffer). When computing a set of KNN-SVs for a candidate set $D_c$ with $N_c$ images and an evaluation set $D_e$ with $N_e$ images, our goal is to compute a $(N_c, N_e)$ tensor of KNN-SVs, where a high KNN-SV at index $(i, j)$ means that the model's prediction for the evaluation image $j$ is highly attributable to candidate image $i$ being present in the training dataset.

Proposed specifically for selective retrieval from a replay buffer, the Adversarial Shapley Value is computed for each sample in the candidate set as:

$$
A S V (i) = \frac {w}{N _ {c}} \sum_ {j \in D _ {c}} s _ {j} (i) - \min _ {k \in D _ {e}} s _ {k} (i) \tag {3}
$$

where $s_{j}(i)$ is the KNN-SV of sample i with respect to sample j and w is a weighting term controlled by a hyperparameter. Note that the left term uses $D_{c}$ as the candidate set and as the evaluation set, and is meant to convey how representative sample i is of the other candidates. The right term uses $D_{c}$ and $D_{e}$ in their usual roles and is subtracted so that it conveys how adversarial sample i is with respect to the evaluation set, which is a batch of new samples in our case. While [39] found that averaging both terms performs best, we found averaging the left term and minimizing the right term outperform all other combinations.

In introducing w, we find it beneficial to dynamically compute its value as opposed to fixing it:

$$
w = c * \frac {\left\| \min _ {s \in S _ {r}} s \right\|}{\left\| \min _ {s \in S _ {l}} s \right\|} \tag {4}
$$

where c is a hyperparameter, $S_{r}$ is the set of KNN-SVs computed for the right term of the ASV and $S_{l}$ is the set for the left term.

See Section 11 of supplementary material for a full description of the algorithm, including the computation of the KNN-SV.

# 4.3. Composing and Modifying Primitives

Our main results (Table 2, 3) consider combinations of class and sample selection primitives. We refer to each composition of primitives using acronyms from the original papers. (e.g. "ASER" refers to combining balanced class selection and ASV sample selection). Besides combining existing primitives (and adding hyperparameters), we extend a few primitives with respect to how they are combined and/or implemented:

1. ASER-PC, where the left term of the ASV is pre-computed with the entire replay buffer as the candidate and evaluation dataset. This allows us to scale the candidate set and potentially get a more accurate estimate of how representative each sample is of the replay buffer.   
2. A-SW-GRASP, where we adaptively determine whether to use GRASP (balanced class selection, prototype-weighted sample selection [15]) or SWIL (similarity-weighted class selection, uniform sample selection [36]) for each sample in the batch. We use GRASP when the normalized entropy of the distribution over classes produced by SWIL is too close to 1.0, where the specific threshold is determined by a hyperparameter.

# 4.4. Deduplication Schedules

We refer to deduplication as the process of reducing duplicate replays for a fixed time period, where after each period, all samples in the buffer are once again eligible for retrieval.

A deduplication schedule is a function that determines the length of this time period; we study three of them in this work: ensuring that there are no duplicate replays occurring within each epoch, within each dataset, and within $\frac{1}{3}$ of the replay buffer that is replayed. In our case, the latter schedule corresponds to about the $\frac{1}{3}$ -mark of training, on average.

# 4.5. Loss Adaptivity

New samples with low losses have smaller gradients and thus may be able to be learned without replay. Selectively limiting the impact of replay in this way could lead to more plasticity for new tasks, without incurring more forgetting. Here, we propose an algorithm for adaptively reducing the number of replayed samples and for reducing the magnitude of replayed samples' losses according to each new sample's loss magnitude.

For each batch of new samples, we begin by computing each sample's loss. We then compute the number of samples that have loss values above a threshold:

$$
r = \left| \{s \in B \mid \text { loss } (s) > l \} \right| \tag {5}
$$

where B is the current batch, s is a sample from the current batch, and l is a hyperparameter. We then use r to determine how many samples to replay. After separately computing the average loss of samples in the new batch and the average loss of samples in the replay batch, we compute the final loss value for the batch as follows:

$$
L _ {T} = L _ {B} + \frac {r}{| B |} * L _ {R} \tag {6}
$$

where $L_{T}$ is the final loss value for the batch, $L_{B}$ is the average loss computed for the batch, and $L_{B}$ is the average loss computed for the replay batch.

# 4.6. Replay Buffer Selection

Before training, we must select samples from the pre-training dataset for our replay buffer. Although there are sophisticated methods, they require access to samples' gradients throughout training $[2, 42]$ , which we do not have. We therefore propose a gradient-free algorithm for selectively constructing a replay buffer.

To satisfy the need for low-loss samples in the replay buffer [18], we begin by removing samples that have loss values greater than a predefined magnitude:

$$
R = \{s \in D _ {p t} \mid \text { loss } (s) <   t \} \tag {7}
$$

where $R$ is the replay buffer, $s$ is an image in $D_{pt}$ , and $t$ is a hyperparameter.

Then, instead of maximizing the diversity of gradients in the replay buffer, we sampled from $D_{pt}$ to ensure that each class had at least 50 images in R.

# 5. Experiments

In this section, we detail the experimental setup used to evaluate all proposed methods. Our experiments begin with a full sweep over loss thresholds for constructing the replay buffer. We then present our main results, which represent a comprehensive and simultaneous sweep over deduplication schedules and hyperparameters for each selective retrieval algorithm. Next, we take a closer look at the effects of different deduplication schedules. Finally, we provide results for loss-adaptive retrieval.

<table><tr><td>Replay Buffer</td><td>O365</td><td>LVIS</td><td>LVIS rare</td><td>ODinW</td></tr><tr><td>Non-selective</td><td>20.6</td><td>19.8</td><td>15.5</td><td>31.6</td></tr><tr><td>Selective</td><td>21.8</td><td>23.8</td><td>20.0</td><td>33.2</td></tr></table>

Table 1. Test mAP of best-performing replay buffer versus including all pre-training samples in the replay buffer. Uniform retrieval. The results are averaged across 3 dataset orderings and recorded at the end of the 15-dataset sequence.

Model: We focus our experiments on the OWL-ViT L/14 open-vocabulary object detection model $[30]$ (Figure 7). This model was CLIP $[32]$ pre-trained on 6 billion image-text pairs, then adapted to object detection on the Visual Genome $[23]$ and Objects365 $[37]$ datasets, which total over 700,000 images. See Section 9.

Datasets: For the continual learning phase, we sequentially fine-tune our model on a carefully selected 15-dataset subset of the Objects Detection in the Wild (ODinW) [27] benchmark. See Section 13 of the supplementary material for more details on our datasets.

Training: Before fine-tuning the model sequentially on each ODinW dataset, we independently perform a hyperparameter sweep on each dataset over learning rates, epochs, and layerwise decay coefficients of learning rate. We also used a cosine learning rate scheduler with a single, linear warmup epoch. Optimization details and hardware can be found in Section 12 of the supplementary material.

Evaluation: Before and after the continual learning phase, we evaluate the model on the Objects365 dataset and the LVIS dataset. The LVIS dataset measures the open vocabulary performance of the model and, consistent with $[30]$ , we also evaluate the model on the "rare" class subset of LVIS, which includes classes for which the model has never seen bounding boxes. These three benchmarks allow us to evaluate the model on classes that are and are not in the replay buffer, which is necessary for this problem setting.

We follow the standard practice of measuring AP at IoU=.50:.05:.95, averaged across classes (termed mAP). We measure the performance of each algorithm according to mAP after training on all datasets, where "ODinW" indicates the average mAP across all 15 datasets.

# 5.1. Replay Buffer Selection

Table 1 compares the performances between continual learners using the entire pre-training dataset as the replay buffer and continual learners using the buffer produced by the best-performing Objects365 and Visual Genome loss thresholds. It is clear that our simple, gradient-free buffer

<table><tr><td rowspan="2"></td><td colspan="2">Class Selection</td><td colspan="3">Sample Selection</td><td colspan="2">Modifications</td></tr><tr><td>Balanced</td><td>Similarity</td><td>Uniform</td><td>Prototype</td><td>ASER</td><td>ASER-PC</td><td>Adaptive</td></tr><tr><td>Uniform</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uniform balanced</td><td>✓</td><td></td><td>✓</td><td></td><td></td><td></td><td></td></tr><tr><td>GRASP [15]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td></tr><tr><td>SWIL [36]</td><td></td><td>✓</td><td>✓</td><td></td><td></td><td></td><td></td></tr><tr><td>SW-GRASP</td><td></td><td>✓</td><td></td><td>✓</td><td></td><td></td><td></td></tr><tr><td>A-SW-GRASP</td><td></td><td>✓</td><td></td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>ASER [39]</td><td>✓</td><td></td><td>✓</td><td></td><td>✓</td><td></td><td></td></tr><tr><td>ASER-PC</td><td>✓</td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td></tr><tr><td>SW-ASER-PC</td><td></td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td><td></td></tr></table>

Table 2. Retrieval algorithms evaluated in this work.

<table><tr><td>Algorithm</td><td>O365</td><td>LVIS</td><td>LVIS rare</td><td>ODinW</td></tr><tr><td>No fine-tuning</td><td>24.2</td><td>29.0</td><td>23.8</td><td>12.3</td></tr><tr><td>Separate models</td><td>-</td><td>-</td><td>-</td><td>45.6</td></tr><tr><td>No replay</td><td>0.0</td><td>0.3</td><td>0.2</td><td>11.1</td></tr><tr><td>Uniform</td><td>21.8</td><td>23.8</td><td>20.0</td><td>33.2</td></tr><tr><td>Uniform balanced</td><td>22.2</td><td>24.5</td><td>20.7</td><td>33.7</td></tr><tr><td>GRASP [15]</td><td>22.5</td><td>24.9</td><td>21.4</td><td>33.6</td></tr><tr><td>SWIL [36]</td><td>22.5</td><td>24.9</td><td>21.6</td><td>33.6</td></tr><tr><td>SW-GRASP</td><td>22.1</td><td>24.4</td><td>20.9</td><td>33.5</td></tr><tr><td>A-SW-GRASP</td><td>22.4</td><td>24.8</td><td>21.5</td><td>33.7</td></tr><tr><td>ASER [39]</td><td>22.2</td><td>24.3</td><td>20.5</td><td>33.6</td></tr><tr><td>ASER-PC</td><td>22.4</td><td>24.6</td><td>20.8</td><td>33.7</td></tr><tr><td>SW-ASER-PC</td><td>22.5</td><td>24.7</td><td>20.9</td><td>33.9</td></tr></table>

Table 3. Test mAP for each retrieval algorithm. The results are shown for the best deduplication algorithm (deduplication for each dataset) averaged across six ODinW dataset orderings. The results are recorded at the end of the 15-dataset sequence. Baselines are highlighted in gray.

selection algorithm provides substantial performance gains relative to the non-selective baseline, especially for classes which are not represented in the replay buffer (LVIS and LVIS rare).

It is also clear that our model shows a strong preference for low-loss samples in the replay buffer, as the best-performing hyperparameter set only included 50,000 images. The best replay buffer also entirely excluded images from Visual Genome, which can be explained by its extremely large concept space and high average loss per image; all VG images have more than 10x the loss of any Objects365 images in our final replay buffer.

Overall, the success of our buffer selection algorithm suggests that it is possible to replicate the success of more sophisticated gradient-dependent techniques with simple loss thresholding and class balancing.

<table><tr><td>Replay %</td><td>O365</td><td>LVIS</td><td>LVIS rare</td><td>ODinW</td></tr><tr><td>100%</td><td>21.8</td><td>24.5</td><td>22.2</td><td>33.2</td></tr><tr><td>97%</td><td>20.9</td><td>23.4</td><td>19.9</td><td>33.3</td></tr><tr><td>85%</td><td>18.7</td><td>21.7</td><td>17.7</td><td>32.5</td></tr></table>

Table 4. Test mAP of each loss threshold when paired with SWIL, averaged across 3 random dataset orderings. The baseline is highlighted in gray.

<table><tr><td>Deduplication</td><td>O365</td><td>LVIS</td><td>LVIS rare</td><td>ODinW</td></tr><tr><td>None</td><td>21.8</td><td>23.9</td><td>20.2</td><td>32.5</td></tr><tr><td>Each epoch</td><td>21.8</td><td>24.2</td><td>20.5</td><td>32.1</td></tr><tr><td>Each dataset</td><td>22.1</td><td>24.5</td><td>21.1</td><td>32.9</td></tr><tr><td>1/3 of buffer</td><td>21.6</td><td>23.7</td><td>20.0</td><td>32.7</td></tr></table>

Table 5. Test mAP of each deduplication algorithm when paired with SWIL, averaged across 3 random dataset orderings. The baseline is highlighted in gray.

# 5.2. Comparing Retrieval Strategies

After building the replay buffer, we perform a hyperparameter sweep for each algorithm (refer to Table 2 for a description of each algorithm), which includes a sweep over deduplication schedules. Table 3 reports the test performance of each algorithm's best-performing hyperparameter sets.

We find that the two top-performing algorithms are also the most simple and, most surprisingly, that combining the two best algorithms results in a significant drop in performance by all measures. The failure of A-SW-GRASP particularly indicates that the best class selection primitive (similarity-weighted) and the best sample selection primitive (prototype-weighted) are fundamentally incompatible despite being completely orthogonal. A potential explanation for this incompatibility is that both SWIL and GRASP can range from weak to strong selectivity in their produced

![](images/9f8da507904de6f2c252a73170c77ca77162a2a3650e71505fc360fe9c920b78.jpg)

<details>
<summary>histogram</summary>

| Normalized Entropy Bin | Number of Classes in Bin |
| ---------------------- | ------------------------ |
| 0.970 - 0.971          | 1                        |
| 0.971 - 0.972          | 2                        |
| 0.972 - 0.973          | 1                        |
| 0.973 - 0.974          | 1                        |
| 0.974 - 0.975          | 1                        |
| 0.975 - 0.976          | 1                        |
| 0.976 - 0.977          | 1                        |
| 0.977 - 0.978          | 1                        |
| 0.978 - 0.979          | 1                        |
| 0.979 - 0.980          | 1                        |
| 0.980 - 0.981          | 1                        |
| 0.981 - 0.982          | 1                        |
| 0.982 - 0.983          | 1                        |
| 0.983 - 0.984          | 1                        |
| 0.984 - 0.985          | 1                        |
| 0.985 - 0.986          | 1                        |
| 0.986 - 0.987          | 1                        |
| 0.987 - 0.988          | 1                        |
| 0.988 - 0.989          | 1                        |
| 0.989 - 0.990          | 1                        |
| 0.990 - 0.991          | 1                        |
| 0.991 - 0.992          | 1                        |
| 0.992 - 0.993          | 1                        |
| 0.993 - 0.994          | 1                        |
| 0.994 - 0.995          | 1                        |
| 0.995 - 0.996          | 1                        |
| 0.996 - 0.997          | 1                        |
| 0.997 - 0.998          | 1                        |
| 0.998 - 0.999          | 1                        |
| 0.999 - 1.000          | 1                        |
</details>

![](images/9be66673ac0a5ccfd128de943b9bc2d26548ee609352d348d4f79a23b3d684c5.jpg)

<details>
<summary>area</summary>

MIN-Entropy GRASP Distribution (ent=0.975)
| Sample (sorted) | Sampling Probability |
| :--- | :--- |
| 0 | 0.0034 |
| 100 | 0.0028 |
| 200 | 0.0024 |
| 300 | 0.0020 |
| 400 | 0.0012 |
| 500 | 0.0008 |
| 600 | 0.0006 |
</details>

![](images/3a592210aaa980637c64d7ed8fd669302d0f599b34d1c92363150d0e30d06548.jpg)

<details>
<summary>bar</summary>

| Sample (sorted) | Sampling Probability |
| --------------- | -------------------- |
| 0               | 0.005                |
| 50              | 0.004                |
| 100             | 0.003                |
| 150             | 0.0025               |
| 200             | 0.002                |
| 250             | 0.0015               |
| 300             | 0.001                |
| 350             | 0.0005               |
</details>

![](images/489cb401d287b779a979cdb96116ae66558b5e98a6ba5c4589c62d11fc22b582.jpg)

<details>
<summary>bar</summary>

| Sample (sorted) | Sampling Probability |
| --------------- | -------------------- |
| 0               | 0.010                |
| 10              | 0.009                |
| 20              | 0.0085               |
| 30              | 0.008                |
| 40              | 0.0075               |
| 50              | 0.007                |
| 60              | 0.0065               |
| 70              | 0.006                |
| 80              | 0.0055               |
| 90              | 0.005                |
| 100             | 0.0045               |
| 110             | 0.004                |
| 120             | 0.0035               |
</details>

Figure 3. Histogram of normalized entropies for distributions produced by GRASP, followed by the distributions with minimum, median, and maximum entropies. Recall that GRASP produces a distribution over samples within each class.

distributions (see Section 6.1) which may, on average, excessively reduce the randomness of the overall algorithm. This effect may be especially prevalent considering the fact that each algorithm operates under the assumption that there is a single class per image, while there may be many classes per image in our setting.

Another surprising result is that using the adversarial shapley value provides little to no benefits over uniform balanced retrieval and is even counterproductive in its original form ("ASER"). A potential explanation for this outcome is that ASER shows a strong preference for the retrieval of samples which have the smallest distances to samples in the current batch (see Section 6.2).

Overall, the failures of SW-GRASP and A-SW-GRASP along with the marginal benefits of ASER-PC and SW-ASER-PC suggest that the problem of selective retrieval involves a delicate tradeoff between selectivity and diversity.

# 5.3. Preventing Duplicate Replays

We now compare deduplication schedules. See Table 5 for SWIL results. All algorithms besides uniform balanced show a preference for dataset-level deduplication, with the worst-performing schedule consistently being the strictest.

This result reflects the fact that each new dataset represents a distinct continual learning problem (see Figure 6) which may call for maximum selectivity in the early epochs since that is when each new sample's gradients are largest and have the highest variance. This selectivity can then be reduced in later epochs in favor of greater coverage of the replay buffer.

# 5.4. Loss Adaptivity

Following our main experiments, we tested whether keeping a 1:1 ratio between new and replayed samples and maintaining the full impact of the replay loss, regardless of new samples' losses, is necessary. As seen in Table 4, the case where as little as 3% of new samples leads to a performance decrease.

Paired with the "No Replay" row from Table 3, this result suggests that fine-tuning a foundation model represents a setting where representations are more fragile than

![](images/62ad103a0413ea9f225d397af5d2da48d490b8dca3bf00ed72db7a0b3fee4a15.jpg)

<details>
<summary>line</summary>

| Class ID | Sampling Probability |
| -------- | -------------------- |
| 0        | 0.002                |
| 50       | 0.003                |
| 100      | 0.004                |
| 150      | 0.003                |
| 200      | 0.005                |
| 250      | 0.004                |
| 300      | 0.003                |
| 350      | 0.002                |
</details>

![](images/f421f0557cf4984422e56a85b3dfc87ab6a312b179cfba92b18361ecf5ccedec.jpg)

<details>
<summary>line</summary>

| Class ID | Sampling Probability |
| -------- | -------------------- |
| 0        | 0.0025               |
| 50       | 0.0026               |
| 100      | 0.0027               |
| 150      | 0.0026               |
| 200      | 0.0025               |
| 250      | 0.0024               |
| 300      | 0.0023               |
| 350      | 0.0022               |
| 400      | 0.0021               |
| 450      | 0.0020               |
| 500      | 0.0019               |
| 550      | 0.0018               |
| 600      | 0.0017               |
| 650      | 0.0016               |
| 700      | 0.0015               |
| 750      | 0.0014               |
| 800      | 0.0013               |
| 850      | 0.0012               |
| 900      | 0.0011               |
| 950      | 0.0010               |
| 1000     | 0.0009               |
</details>

Figure 4. SWIL distributions with minimum (left) and maximum (right) entropies. Remember that SWIL produces a distribution over classes in the replay buffer.

in standard continual learning settings, as previous work has shown baseline training runs with no continual learning methods to catastrophically forget, but not to completely destroy knowledge from previous tasks, especially when using proper training objectives. For example, [3] shows 30% final accuracy on Split-Mini-ImageNet when naively fine-tuning the model with a SupCon [22] loss, while our model quickly drops to an mAP of around 0 when naively fine-tuning on new tasks.

# 6. Analyses

In this section, we examine the distributions produced by SWIL and GRASP then search for characteristics of downstream datasets which are predictive of forgetting.

# 6.1. The Shape of SWIL and GRASP

The leftmost plot of Figure 3 shows a histogram for the normalized entropies of the distributions produced by GRASP, while the three rightmost plots show the minimum, median, and maximum entropy distributions produced by GRASP as bar charts. Each of the three rightmost distributions are over samples within an O365 class and are measured before the continual learning phase.

The distribution of entropies has a long tail, and even the median-entropy distribution produced by GRASP is strongly selective (the highest-probability sample has a 5x higher probability than the lowest-probability sample). These plots therefore show that GRASP prevents forget-

![](images/7353f3955bc996c1ebee7791d62d80cb98cd8b7ecde5127ca52793d5b549976a.jpg)

<details>
<summary>histogram</summary>

| KNN-SV of Candidate for Current Batch | Frequency |
| ------------------------------------- | --------- |
| -0.008 to -0.006                        | 0.000     |
| -0.006 to -0.004                        | 0.000     |
| -0.004 to -0.002                        | 0.000     |
| -0.002 to 0.000                         | 0.100     |
| 0.000 to 0.002                          | 0.080     |
| 0.002 to 0.004                          | 0.040     |
| 0.004 to 0.006                          | 0.020     |
| 0.006 to 0.008                          | 0.010     |
</details>

![](images/21abf382611f9ad82e9be5e5c335bee6f48d031e0c848024f0ae6c5d859c2ec6.jpg)

<details>
<summary>histogram</summary>

| Minimum KNN-SV Candidate's Distance Rank | Frequency |
| ----------------------------------------- | --------- |
| 250                                       | 0.00      |
| 275                                       | 0.01      |
| 300                                       | 0.03      |
| 325                                       | 0.04      |
| 350                                       | 0.04      |
</details>

Figure 5. Histogram of KNN-SVs across an entire training sequence (left) and histogram of distance ranks for lowest-scoring (most likely to be selected) images in a candidate set across the same sequence (right). A higher distance rank means the image is closer to the evaluation image; each candidate set contained 352 images.

ting by showing a strong preference towards prototypical samples throughout training. These plots also explain why omitting deduplication for GRASP leads to a substantial performance decrease.

Figure 4 shows the minimum- and maximum-entropy distributions (over classes) produced by SWIL for a dataset. We took the average probability of selecting each O365 class for an entire continual learning dataset as the SWIL distribution for that dataset.

While the minimum-entropy SWIL distribution is quite selective (probabilities range from 0.0025 to 0.005), the maximum-entropy distribution is nearly uniform.

To determine whether lower-entropy (more selective) SWIL distributions predict SWIL's per-dataset performance increase/decrease relative to GRASP, we computed a correlation between each dataset's SWIL entropy and the difference between SWIL and GRASP's Objects365 mAPs after training on the dataset. This resulted in a correlation coefficient of -0.08 (less entropy correlates with better performance).

This relationship suggests that SWIL's equivalence with GRASP (Table 3) can be attributed to SWIL effectively targeting the most vulnerable circuits within the network in the case of sufficient similarity between new samples and samples in the pre-training dataset; in other cases, SWIL is outperformed by GRASP.

Importantly, this insight leads to the conclusion that SWIL may scale quite well with the number of classes in the replay buffer, as it would then be more likely to produce lower-entropy distributions over classes in the replay buffer. Such an effect could lead SWIL to outperform GRASP in certain scenarios, although the inverse would likely hold as well.

See Section 14 of supplementary material for analysis of the relationship between the distributions produced by SWIL and resulting class-specific forgetting.

![](images/4b97cdd7efa465750b1a5c6c075112f1da56a9e9cbae8594587dc78f61ce7876.jpg)

<details>
<summary>bar</summary>

| Dataset Size | Change in LVIS rare mAP |
| ------------ | ------------------------ |
| 90           | -2.437                   |
| 142          | -2.128                   |
| 202          | -1.800                   |
| 255          | -1.480                   |
| 371          | -1.260                   |
| 407          | -1.040                   |
| 448          | -0.820                   |
| 465          | -0.600                   |
| 516          | -0.400                   |
| 576          | -0.200                   |
| 780          | 0.200                    |
| 800          | 0.400                    |
| 1,000        | 0.600                    |
| 1,218        | 0.800                    |
| 1,437        | 1.000                    |
</details>

Figure 6. Change in LVIS rare mAP for each dataset, sorted by dataset size. Averaged across 3 SWIL training runs.

# 6.2. The Bias of ASER

The left plot of Figure 5 contains a histogram for the KNN-SVs computed between candidate images and batch images for SW-ASER-PC (Table 2). It is clear that the KNN-SV provides a strong signal, as the distribution is long-tailed.

The right plot of Figure 5 contains a histogram depicting the frequency of different distance rankings for the lowest-KNN-SV candidate images, where the "distance ranking" is the ranking of the candidate image in terms of its distance to an image in the current batch. Note that the image with the lowest KNN-SV for the batch is most likely to be retrieved, since the "adversarial" term of the ASV is negated (Equation 12).

From the right plot, it is clear that SW-ASER-PC is biased towards retrieving the from the 25-closest candidates for each sample in the batch. This property leads to redundancy in retrieved samples, which may explain the lesser performance of SW-ASER-PC relative to SWIL and GRASP.

# 6.3. Dataset-Dependent Forgetting Dynamics

In our search for predictors of dataset-dependent forgetting, we correlated forgetting (mAP of pre-training datasets before and after encountering each downstream dataset) with characteristics including datasets' mAP before training, mAP improvement from training, position in the sequence of datasets, and size.

The strongest predictor was datasets' size, which is plotted in Figure 6. The rest of our metrics showed a weak correlation, if any. Note that pre-training mAP both improves and degrades throughout training.

This result shows the destructive effect of reducing the time-to-recurrence of each datapoint, which follows from the intuition that continual learning calls for the prevention of model overfitting on unimportant aspects of new training data $[25]$ . The unreliability of the relationship shown in

Figure 6 also suggests more nuanced forgetting dynamics, which are likely dependent on properties of the model's pre-training distribution.

# 7. Conclusion

Selective retrieval is a key component of any replay algorithm. Existing methods can be organized into class- and sample-selective primitives; however, the best class- and sample-selective primitive cannot be combined to reach a new state of the art. This may be explained by the best selective retrieval algorithms being highly selective at times, leading to an overall insufficient amount of randomness across the learning trajectory.

Any algorithm used for selective retrieval should ensure that no duplicates are retrieved within a single continual learning task (dataset). This leads to retrieval being highly selective during the early epochs for each task (when losses on new samples are highest) and greater coverage of the replay buffer in later epochs.

When using replay, it is essential that the replay loss's magnitude is kept consistent, regardless of new samples' losses. This reflects the necessity of maintaining existing representations in the network, especially in the problem setting of continual learning at scale, since previous representations cover many different classes and are required for forward transfer (optimal learning of new tasks).

Extensions to this work includes measures for reducing the computational overhead associated with replaying full inputs. This can be achieved by developing methods which replay old samples on the level of tokens, with an architecture similar to that of retrieval augmented generation $[26]$ .

One key limitation of this work is that the replay buffer was not updated with samples from downstream tasks. Adding new samples to the replay buffer and ensuring they are replayed could potentially remove the performance gap between the ODinW performance upper bound and ODinW performance achieved when using replay.

Another key limitation of this work is that the continual learning phase only consisted of a sequence of 15 datasets. In practice, this sequence would need to be much longer for sequential learning to make sense, as all 15 datasets could be learned jointly. Some interesting properties of the problem setting also may only be revealed given a task sequence of sufficient length.

Based on the preceding analyses on dataset-dependent forgetting, a higher replay ratio for smaller downstream datasets should also be explored.

# 8. Acknowledgments

This work is partially supported by the NSF EFRI BRAID grant #2317706.

# References

[1] Rahaf Aljundi, Lucas Caccia, Eugene Belilovsky, Massimo Caccia, Min Lin, Laurent Charlin, and Tinne Tuytelaars. Online continual learning with maximally interfered retrieval, 2019. 1, 2   
[2] Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection for online continual learning, 2019. 2, 5   
[3] Nader Asadi, MohammadReza Davari, Sudhir Mudur, Rahaf Aljundi, and Eugene Belilovsky. Prototype-sample relation distillation: Towards replay-free continual learning, 2023. 2, 7   
[4] Angels Balaguer, Vinamra Benara, Renato Luiz de Freitas Cunha, Roberto de M. Estevão Filho, Todd Hendry, Daniel Holstein, Jennifer Marsman, Nick Mecklenburg, Sara Malvar, Leonardo O. Nunes, Rafael Padilha, Morris Sharp, Bruno Silva, Swati Sharma, Vijay Aski, and Ranveer Chandra. Rag vs fine-tuning: Pipelines, tradeoffs, and a case study on agriculture, 2024. 1   
[5] Ivana Balažević, David Steiner, Nikhil Parthasarathy, Relja Arandjelović, and Olivier J. Hénaff. Towards in-context scene understanding, 2023. 1   
[6] Konstantinos Bousmalis, Giulia Vezzani, Dushyant Rao, Coline Devin, Alex X. Lee, Maria Bauza, Todor Davchev, Yuxiang Zhou, Agrim Gupta, Akhil Raju, Antoine Laurens, Claudio Fantacci, Valentin Dalibard, Martina Zambelli, Murilo Martins, Rugile Pevceviciute, Michiel Blokzijl, Misha Denil, Nathan Batchelor, Thomas Lampe, Emilio Parisotto, Konrad Żołna, Scott Reed, Sergio Gómez Colmenarejo, Jon Scholz, Abbas Abdolmaleki, Oliver Groth, Jean-Baptiste Regli, Oleg Sushkov, Tom Rothörl, José Enrique Chen, Yusuf Aytar, Dave Barker, Joy Ortiz, Martin Riedmiller, Jost Tobias Springenberg, Raia Hadsell, Francesco Nori, and Nicolas Heess. Robocat: A self-improving generalist agent for robotic manipulation, 2023. 1   
[7] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline, 2020. 2   
[8] Lucas Caccia, Rahaf Aljundi, Nader Asadi, Tinne Tuytelaars, Joelle Pineau, and Eugene Belilovsky. New insights on reducing abrupt representation change in online continual learning, 2022. 2, 1   
[9] Arslan Chaudhry, Marc' Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem, 2019. 2   
[10] Matthias Delange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Greg Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 1–1, 2021. 2   
[11] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. A survey on in-context learning, 2023. 1   
[12] Alexandre Galashov, Jovana Mitrovic, Dhruva Tirumala, Yee Whye Teh, Timothy Nguyen, Arslan Chaudhry, and Razvan Pascanu. Continually learning representations at

scale. In Proceedings of The 2nd Conference on Lifelong Learning Agents, pages 534–547. PMLR, 2023. 2   
[13] Saurabh Garg, Mehrdad Farajtabar, Hadi Pouransari, Raviteja Vemulapalli, Sachin Mehta, Oncel Tuzel, Vaishaal Shankar, and Fartash Faghri. Tic-clip: Continual training of clip models, 2023. 2   
[14] Raia Hadsell, Dinesh Rao, Andrei A. Rusu, and Razvan Pascanu. Embracing change: Continual learning in deep neural networks. Trends in Cognitive Sciences, 24(12):1028–1040, 2020. 1, 2   
[15] Md Yousuf Harun, Jhair Gallardo, and Christopher Kanan. Grasp: A rehearsal policy for efficient online continual learning, 2023. 1, 3, 4, 6   
[16] Tyler L. Hayes, Kushal Kafle, Robik Shrestha, Manoj Acharya, and Christopher Kanan. Remind your neural network to prevent catastrophic forgetting, 2020. 2   
[17] Tyler L. Hayes, Giri P. Krishnan, Maxim Bazhenov, Hava T. Siegelmann, Terrence J. Sejnowski, and Christopher Kanan. Replay in deep learning: Current approaches and missing biological elements, 2021. 1, 2   
[18] Julio Hurtado, Alain Raymond-Saez, Vladimir Araujo, Vincenzo Lomonaco, Alvaro Soto, and Davide Bacciu. Memory population in continual learning via outlier elimination, 2023. 2, 5   
[19] Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights, 2022. 2   
[20] Paul Janson, Wenxuan Zhang, Rahaf Aljundi, and Mohamed Elhoseiny. A simple baseline that questions the use of pretrained-models in continual learning, 2023. 2   
[21] Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo Li, Ce Zhang, Costas J. Spanos, and Dawn Song. Efficient task-specific data valuation for nearest neighbor algorithms, 2020. 4   
[22] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning, 2021. 7   
[23] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Fei-Fei Li. Visual genome: Connecting language and vision using crowdsourced dense image annotations, 2016. 5   
[24] Kuan-Ying Lee, Yuanyi Zhong, and Yu-Xiong Wang. Do pre-trained models benefit equally in continual learning?, 2022. 2   
[25] Timothée Lesort. Continual feature selection: Spurious features in continual learning, 2022. 8   
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. 1, 9   
[27] Chunyuan Li, Haotian Liu, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, and Jianfeng Gao. Elevater: A benchmark and toolkit for evaluating language-augmented visual models, 2022. 5

[28] Zhiqiu Lin, Jia Shi, Deepak Pathak, and Deva Ramanan. The clear benchmark: Continual learning on real-world imagery, 2022. 2   
[29] Zheda Mai, Ruiwen Li, Hyunwoo Kim, and Scott Sanner. Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning, 2021. 1   
[30] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. Simple open-vocabulary object detection with vision transformers, 2022. 3, 5, 1   
[31] Aristeidis Panos, Yuriko Kobe, Daniel Olmeda Reino, Rahaf Aljundi, and Richard E. Turner. First session adaptation: A strong replay-free baseline for class-incremental learning, 2024. 2   
[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 5   
[33] Vinay V. Ramasesh, Ethan Dyer, and Maithra Raghu. Anatomy of catastrophic forgetting: Hidden representations and task semantics, 2020. 1   
[34] Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks, 2022. 2   
[35] Fahad Sarfraz, Elahe Arani, and Bahram Zonooz. Error sensitivity modulation based experience replay: Mitigating abrupt representation drift in continual learning, 2023. 2   
[36] Rajat Saxena, Justin L. Shobe, and Bruce L. McNaughton. Learning in deep neural networks and brains with similarity-weighted interleaved learning. Proceedings of the National Academy of Sciences, 119(27):e2115229119, 2022. 1, 2, 3, 4, 6   
[37] Shuai Shao, Zeming Li, Tong Zhang, Chao Peng, Gang Yu, Xiaogang Zhang, Jianping Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 8429–8438, 2019. 5   
[38] Lloyd S. Shapley. A value for n-person games. Contributions to the Theory of Games, 2:307-317, 1953. 4   
[39] Dongsub Shim, Zheda Mai, Jihwan Jeong, Scott Sanner, Hyunwoo Kim, and Jongseong Jang. Online class-incremental continual learning with adversarial shapley value, 2021. 2, 4, 6   
[40] Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep generative replay, 2017. 2   
[41] James Seale Smith, Junjiao Tian, Shaunak Halbe, Yen-Chang Hsu, and Zsolt Kira. A closer look at rehearsal-free continual learning, 2023. 2   
[42] Rishabh Tiwari, Krishnateja Killamsetty, Rishabh Iyer, and Pradeep Shenoy. Gcr: Gradient coreset based replay buffer selection for continual learning, 2022. 2, 5   
[43] Gido M. van de Ven and Andreas S. Tolias. Three scenarios for continual learning, 2019. 2

[44] Gido M. van de Ven, Hava T. Siegelmann, and Andreas S. Tolias. Brain-inspired replay for continual learning with artificial neural networks. Nature Communications, 11(1):4069, 2020. 2   
[45] Eli Verwimp, Matthias De Lange, and Tinne Tuytelaars. Rehearsal revealed: The limits and merits of revisiting samples in continual learning, 2021. 1   
[46] Eli Verwimp, Rahaf Aljundi, Shai Ben-David, Matthias Bethge, Andrea Cossu, Alexander Gepperth, Tyler L. Hayes, Eyke Hüllermeier, Christopher Kanan, Dhireesha Kudithipudi, Christoph H. Lampert, Martin Mundt, Razvan Pascanu, Adrian Popescu, Andreas S. Tolias, Joost van de Weijer, Bing Liu, Vincenzo Lomonaco, Tinne Tuytelaars, and Gido M. van de Ven. Continual learning: Applications and the road forward, 2023. 1

# Watch Your Step: Optimal Retrieval for Continual Learning at Scale Supplementary Material

![](images/650e11200ef7143d8a6a7240b7e4df3d03e41318361fbfc9e7c6dad5087cfaab.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["'giraffe' 'tree' 'car'"] --> B["Text Transformer encoder"]
    C["Vision Transformer encoder"] --> D["Linear projection"]
    C --> E["MLP head"]
    B --> F["Query embeddings"]
    D --> G["Predicted classes/queries"]
    E --> H["Predicted boxes"]
    F --> I["'giraffe'"]
    F --> J["'tree'"]
    G --> K["<no object>"]
    H --> L["(x₁, y₁, w₁, h₁)"]
    H --> M["(x₂, y₂, w₂, h₂)"]
    H --> N["(x₃, y₃, w₃, h₃)"]
    H --> O["(x₄, y₄, w₄, h₄)"]
    I --> P["Set prediction loss over objects in an image."]
    J --> P
    K --> P
    L --> P
    M --> P
    N --> P
    O --> P
```
</details>

Figure 7. Depiction of the OWL-ViT model under study, taken from the original paper [30]. Following CLIP pre-training, ViT token pooling is removed and light-weight object classification and localization heads are directly attached to the image encoder output tokens.

# 9. OWL-ViT

Besides broad pre-training and 0.5B parameters, a desirable property of this model is that there are no task-specific parameters, which have complicated continual learning research in the past due to models experiencing abrupt representation drift upon the introduction of new, randomly initialized parameters [8, 29, 33, 45].

# 10. Full Similarity-Weighted Class Selection Primitive

SWIL is executed in five main steps:

1. Select the top-k class embeddings for each image in the current batch; will be of size $(B, k, E)$ where B is the batch size. Note that, here, top-k means the k class embeddings with the highest confidence score for any query embedding.   
2. Compute each class embedding's similarity with each previous-class prototype; $(B,k,C_{rb})$ , where $C_{rb}$ is the number of classes in the replay buffer   
3. Average or maximize similarities over dimension $k$ to get image-level similarities; $(B, C_{rb})$   
4. Transform these similarities into a probability distribution over previous classes; $(B, C_{rb})$   
5. Sample from each distribution to get a class ID for each sample in the batch.

First, we select the top-k class embeddings from each image in the current batch according to the confidence associated with each class embedding's predicted class. Assuming our model outputs a class probability tensor of shape (batch size, # class embeddings, # text queries), the indices of the top-k class embeddings can be defined as the (batch size, k) tensor containing indices corresponding to the $k$ class embeddings which have the most positive inner product with any of the text embeddings for the corresponding image (see Figure 7).

The next step is to compute the similarity between class embeddings of the current batch and a stored set of prototypical class embeddings, which includes one class embedding for each class learned before dataset $d_{cl,i}$ . Assuming $C_{buff}$ classes are represented in the replay buffer, we are given a ( $C_{buff}$ , embedding dim) tensor of previous-class prototypes along with a (batch size, k, embedding dim) tensor of current-batch class embeddings. We then compute a (batch size, k, $C_{buff}$ ) tensor of cosine distances, where cosine distance is defined as:

$$
\mathrm{dist} (\mathbf {e} _ {i}, \mathbf {e} _ {j}) = 1 - \frac {\mathbf {e} _ {i} \cdot \mathbf {e} _ {j}}{\| \mathbf {e} _ {i} \| \| \mathbf {e} _ {j} \|} \tag {8}
$$

To finalize these prototype similarities such that they can be interpreted on the level of images as opposed to instances, we minimize over dimension k to get a (batch size, $C_{buff}$ ) tensor of minimum prototype distances for each image.

The final step is to compute a weighted probability distribution over each class $C_{\mathrm{buff}}$ for each image in the current batch, which will be proportional to the inverse of the image's prototype distance vector $\mathbf{d}_i$ :

$$
\mathbf {p} _ {C _ {\text { buff }}} = \frac {1}{Z} \cdot (\mathbf {d} _ {i}) ^ {w} \tag {9}
$$

Which we then use to sample from each batch image's class distribution to determine which class to sample from for each image in the batch.

# 11. Full ASV Sample Selection Primitive

Assuming a single discrete class label y and a single class embedding per image, given evaluation point $(\mathbf{x}_{j}^{e}, \mathbf{y}_{j}^{e}) \in D_{e}$ we compute the KNN-SV using the following recursion:

$$
s _ {j} (\alpha_ {N _ {c}}) = \frac {\mathbb {1} [ y _ {\alpha_ {N _ {c}}} = y _ {j} ^ {e v} ]}{N _ {c}} \tag {10}
$$

$$
s _ {j} (\alpha_ {m}) = s _ {j} (\alpha_ {m + 1}) +
$$

$$
\frac {\mathbb {1} \left[ y _ {\alpha_ {m}} = y _ {j} ^ {e v} \right] - \mathbb {1} \left[ y _ {\alpha_ {m + 1}} = y _ {j} ^ {e v} \right]}{K} \frac {\min (K , m)}{m} \tag {11}
$$

where $\alpha_{m}$ is the index of candidate sample with the $m^{th}$ -closest embedding to the evaluation point, $s_{j}(i)$ is the KNN-SV for pair $(i,j)$ , K is a pre-defined constant, and 1 is the indicator function. Note that, in our implementation, we

replace values for $y_{\alpha_{m}}$ and $y_{j}^{ev}$ with text embeddings and replace the indicator function with the cosine similarity between embeddings. Similar to our SWIL implementation, we again use the top-k class embeddings as a drop-in replacement for the class embeddings used in previous work. This means our $(N_{c}, N_{e})$ tensor of KNN-SVs instead becomes a $(N_{c}, k, N_{e}, k)$ tensor computed on the level of instances in an all-to-all manner.

Given the above procedure for computing a $(N_{c}, k, N_{e}, k)$ tensor of KNN-SVs for a given candidate and evaluation set, we now go over the computation of the ASV, which was specifically designed for selective retrieval from a replay buffer [39]. Given a candidate set $D_{c}$ containing a subset of the replay buffer and an evaluation set $D_{e}$ containing samples from the current batch, the ASV for the candidate sample at index i is defined as:

$$
A S V (i) = \frac {w}{N _ {c}} \sum_ {j \in D _ {c}} s _ {j} (i) - \min _ {k \in D _ {e}} s _ {k} (i) \tag {12}
$$

where w is a novel hyperparameter controlling the relative contributions of each term to the final score. Note that the left term of the ASV computes KNN-SVs with the candidate set also acting as the evaluation set, while the right term computes KNN-SVs with the candidate set and evaluation set in their usual roles and is subtracted. The left term therefore measures how representative each candidate sample is of the candidate set while the right term measures how adversarial the sample is for the current batch.

Before providing the two KNN-SV tensors to the ASV function, we maximize the KNN-SVs in the $(N_{c}, k, N_{c}, k)$ tensor (the left term of the ASV) over each k dimension to get a $(N_{c}, N_{c})$ image-level tensor. For the $(N_{c}, k, N_{e}, k)$ tensor (right term), we instead minimize over the k dimensions. We found maximizing/minimizing to outperform averaging.

To get the final ASV tensor, we apply Equation 12 and linearly combine the tensors of the left term and right term after averaging and minimizing over each of their second dimensions, respectively; this leaves us with a 1-dimensional ASV tensor with $N_{c}$ entries. Finally, we take the top-n highest-scoring candidates as replay samples. Note that the original paper [39] finds averaging both terms of the ASV to perform best, while we found averaging the left term and minimizing the right term to outperform all other combinations.

# 12. Optimization and Hardware

For superior regularization, we use Dark Experience Replay (DER++) [7]. DER++ offers substantial performance benefits over only using a classification and box-prediction loss on replayed samples thanks to its logit distillation term. Note that we keep our replay buffer's logits and embeddings (including prototypes) fixed throughout the training sequence and that we do not add data from the continual learning sequence to the replay buffer.

![](images/ef50ca709f0d47dfc6127e566048c0eaaaca972243e0270746d9ae58d6a262e5.jpg)

<details>
<summary>line</summary>

| Class | % Change in Class's mAP |
|-------|--------------------------|
| 0     | ~0                       |
| 10    | ~-25                     |
| 20    | ~-10                     |
| 30    | ~20                      |
| 40    | ~-10                     |
| 50    | ~-100                    |
| 60    | ~-5                      |
| 70    | ~0                       |
| 80    | ~0                       |
| 90    | ~0                       |
| 100   | ~0                       |
| 110   | ~-50                     |
| 120   | ~-10                     |
| 130   | ~-5                      |
| 140   | ~25                      |
| 150   | ~-50                     |
| 160   | ~35                      |
| 170   | ~-5                      |
| 180   | ~0                       |
| 190   | ~0                       |
| 200   | ~0                       |
| 210   | ~0                       |
| 220   | ~0                       |
| 230   | ~0                       |
| 240   | ~0                       |
| 250   | ~0                       |
| 260   | ~-5                      |
| 270   | ~45                      |
| 280   | ~-5                      |
| 290   | ~-100                    |
| 300   | ~-75                     |
| 310   | ~-5                      |
| 320   | ~25                      |
| 330   | ~-5                      |
| 340   | ~-5                      |
| 350   | ~-5                      |
</details>

Figure 8. SWIL distributions with minimum and maximum entropies. Remember that SWIL produces a distribution over classes in the replay buffer.

All experiments were executed on a single DGX A100-640GB node; each training run took roughly 4 hours to complete. All selective retrieval algorithms incur negligible training overhead, except for ASER. Since ASER's overhead is controlled by the size of the candidate set and whether the left term of the ASV is pre-computed, we limit the size of the candidate set for each variant such that training-time overhead does not exceed 1% of the training time of the "uniform balanced" algorithm.

# 13. Datasets

We select our subset such that it solely consists of datasets which: have large enough test sets for low test-time variance, have high-quality, comprehensive annotations, and, most importantly, are performed poorly on by our pretrained model (12.3 mAP average).

# 14. Is SWIL Biased?

It remains unclear whether SWIL's performance benefits when producing low-entropy distributions over classes is due to an imbalanced reduction in Objects365 classes' forgetting. Specifically, SWIL could be disproportionately improving performance on classes which it gives high probabilities to while ignoring others.

Figure 8 shows the forgetting observed for each class when training on the dataset for which SWIL produced the lowest-entropy (most selective) distribution over classes. Class IDs (on the x-axis) are sorted in ascending order by their average sampling probabilities across the dataset.

This plot shows that there is not a strong relationship between the forgetting of the model on specific classes and their average sampling probability across the dataset's training. This suggests that the performance benefits of SWIL are highly distributed across classes/subnetworks.

<table><tr><td>Dataset</td><td># Concepts</td><td># Train</td><td># Test</td><td>Pre-Trained OWL-ViT L/14 mAP</td></tr><tr><td>Pothole</td><td>1</td><td>465</td><td>67</td><td>23.0</td></tr><tr><td>WildfireSmoke</td><td>1</td><td>516</td><td>74</td><td>24.4</td></tr><tr><td>ThermalCheetah</td><td>2</td><td>90</td><td>14</td><td>18.7</td></tr><tr><td>ThermalDogsAndPeople</td><td>2</td><td>142</td><td>20</td><td>49.9</td></tr><tr><td>BCCD</td><td>3</td><td>255</td><td>36</td><td>3.6</td></tr><tr><td>ShellfishOpenImages</td><td>4</td><td>407</td><td>58</td><td>19.9</td></tr><tr><td>EgoHands (specific)*</td><td>4</td><td>1000</td><td>480</td><td>2.1</td></tr><tr><td>AerialMaritimeDrone(tiled)</td><td>5</td><td>371</td><td>32</td><td>12.5</td></tr><tr><td>BrackishUnderwater*</td><td>6</td><td>800</td><td>1468</td><td>2.5</td></tr><tr><td>Dice</td><td>6</td><td>576</td><td>71</td><td>0.3</td></tr><tr><td>Aquarium</td><td>7</td><td>448</td><td>63</td><td>21.6</td></tr><tr><td>ChessPieces</td><td>13</td><td>202</td><td>29</td><td>3.5</td></tr><tr><td>AmericanSignLanguageLetters*</td><td>26</td><td>780</td><td>72</td><td>0.6</td></tr><tr><td>Plantdoc</td><td>30</td><td>2128</td><td>239</td><td>1.0</td></tr><tr><td>OxfordPets(breed)</td><td>37</td><td>2437</td><td>345</td><td>0.7</td></tr></table>

Table 6. ODinW datasets used in this work. \* indicates that we use a balanced subset.

# 15. Hyperparameters

<table><tr><td>Algorithm</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="2">DER++ (replay loss)</td><td>α</td><td>2.0</td></tr><tr><td>β</td><td>1.0</td></tr><tr><td>All methods</td><td>k (for top-k tokens)</td><td>8</td></tr><tr><td rowspan="2">Buffer Loss Thresholds</td><td>Objects365</td><td>0.15</td></tr><tr><td>VG</td><td>0.0</td></tr><tr><td>SWIL</td><td>w</td><td>1.0</td></tr><tr><td>GRASP</td><td>w</td><td>1.0</td></tr><tr><td rowspan="3">ASER</td><td>w</td><td>0.15</td></tr><tr><td>K (in recursive KNN-SV)</td><td>20</td></tr><tr><td>N candidates</td><td>168</td></tr><tr><td>ASER-PC</td><td>N candidates</td><td>352</td></tr></table>

Table 7. Hyperparameters used for each retrieval algorithm in this work. All were swept.

<table><tr><td>Dataset</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="5">Pothole</td><td>Epochs</td><td>5</td></tr><tr><td>Linear Warmup Epochs</td><td>1</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">WildfireSmoke</td><td>Epochs</td><td>5</td></tr><tr><td>Linear Warmup Epochs</td><td>1</td></tr><tr><td>ViT LR</td><td> $1.4 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $1.4 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">ThermalCheetah</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $1.4 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $1.4 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">ThermalDogsAndPeople</td><td>Epochs</td><td>5</td></tr><tr><td>Linear Warmup Epochs</td><td>1</td></tr><tr><td>ViT LR</td><td> $1.4 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $1.4 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">BCCD</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">ShellfishOpenImages</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr></table>

Table 8. Hyperparameters used for each dataset in this work. "LR" refers to the learning rate. "LW-LRD Coeff" refers to the LayerWise LR Decay Coefficient, which scales the LR at each layer (starting from the classifier and applied to the vision and text towers independently). All datasets used cosine LR decay with a linear warmup. All HPs were swept. \* indicates that the dataset was subsetted (see Datasets table).

<table><tr><td>Dataset</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="5">EgoHands (specific)*</td><td>Epochs</td><td>5</td></tr><tr><td>Linear Warmup Epochs</td><td>1</td></tr><tr><td>ViT LR</td><td> $1.4 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $1.4 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">AerialMaritimeDrone(tiled)</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $1.4 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $1.4 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.98</td></tr><tr><td rowspan="5">BrackishUnderwater</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">Dice</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">Aquarium</td><td>Epochs</td><td>5</td></tr><tr><td>Linear Warmup Epochs</td><td>1</td></tr><tr><td>ViT LR</td><td> $1.4 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $1.4 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">ChessPieces</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $6.4 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $6.4 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.94</td></tr><tr><td rowspan="5">AmericanSignLanguageLetters*</td><td>Epochs</td><td>7</td></tr><tr><td>Linear Warmup Epochs</td><td>2</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.96</td></tr><tr><td rowspan="5">Plantdoc</td><td>Epochs</td><td>5</td></tr><tr><td>Linear Warmup Epochs</td><td>1</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr><tr><td rowspan="5">OxfordPets(breed)</td><td>Epochs</td><td>5</td></tr><tr><td>Linear Warmup Epochs</td><td>1</td></tr><tr><td>ViT LR</td><td> $2.56 \times 10^{-4}$ </td></tr><tr><td>Text Encoder LR</td><td> $2.56 \times 10^{-5}$ </td></tr><tr><td>LW-LRD Coeff</td><td>0.99</td></tr></table>

Table 9. Continued