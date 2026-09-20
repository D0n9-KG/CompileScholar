# Neural Fine-Tuning Search for Few-Shot Learning

Panagiotis Eustratiadis $^{1}$

Łukasz Dudziak $^{2}$

Da Li $^{1,2}$

Timothy Hospedales $^{1,2}$

$^{1}$ University of Edinburgh

$^{2}$ Samsung AI Center, Cambridge

# Abstract

In few-shot recognition, a classifier that has been trained on one set of classes is required to rapidly adapt and generalize to a disjoint, novel set of classes. To that end, recent studies have shown the efficacy of fine-tuning with carefully crafted adaptation architectures. However this raises the question of: How can one design the optimal adaptation strategy? In this paper, we study this question through the lens of neural architecture search (NAS). Given a pretrained neural network, our algorithm discovers the optimal arrangement of adapters, which layers to keep frozen and which to fine-tune. We demonstrate the generality of our NAS method by applying it to both residual networks and vision transformers and report state-of-the-art performance on Meta-Dataset and Meta-Album.

# 1. Introduction

Few-shot recognition $[23, 33, 51]$ aims to learn novel concepts from few examples, often by rapid adaptation of a model trained on a disjoint set of labels. Many solutions adopt a meta-learning perspective $[16, 25, 38, 41, 43]$ , or train a powerful feature extractor on the source classes $[44, 50]$ – both of which assume that the training and testing classes are drawn from the same underlying distribution e.g., handwritten characters $[24]$ , or ImageNet categories $[48]$ . Later work considers a more realistic and challenging problem variant where a classifier should perform few-shot adaptation not only across visual categories, but also across diverse visual domains $[46, 47]$ . In this cross-domain problem variant, customising the feature extractor to the novel domains is important, and several studies address this through dynamic feature extractors $[2, 40]$ or ensembles of features $[12, 28, 32]$ . Another group of studies employ simple yet effective fine-tuning strategies for adaptation $[10, 21, 29, 53]$ that are predominantly heuristically motivated. Thus, an important question that arises from previous work is: How can one design the optimal adaptation strategy? In this paper, we take a step towards answering this question.

![](images/ed1e7517e38927cb3db16a458ac8daf7853dde35cff073b56d38a0e70efdfe96.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["α1"]
    B --> C["φ1"]
    B --> D["φ'1"]
    E["fθ,α,φ'"] --> F["α2"]
    F --> G["φ2"]
    F --> H["φ'2"]
    I["g1,φ',α"] --> J["αK"]
    K["g2,φ',α"] --> L["αK"]
    M["gK,φ',α"] --> N["αK"]
    O["..."] --> P["..."]
    Q["fθ,α,φ'"] --> R["αK"]
    S["fθ,α,φ'"] --> T["αK"]
```
</details>

Figure 1: Our proposed supernet architecture for few-shot adaptation. The supernet contains all combinations of pre-trained, fine-tuned and adapter parameters. f denotes the feature extractor, which is composed of many layers, g, which are the minimal unit for adaptation in our search space. The dotted lines represent possible paths that can be sampled during SPOS training. Every adaptable layer $g_{i}^{\phi,\phi',\alpha}$ has its own pre-trained parameters ( $\phi_{i} \subset \theta$ ), fine-tuned parameters ( $\phi_{i}'$ ), and adapter parameters ( $\alpha_{i}$ ).

Fine-tuning approaches to few-shot adaptation must manage a trade-off between adapting a large or small number of parameters. The former allows for better adaptation, but risks overfitting on a few-shot training set. The latter reduces the risk of overfitting, but limits the capacity for adaptation to novel categories and domains. The recent PMF $[21]$ manages this trade-off through careful tuning of learning rates while fine-tuning the entire feature extractor. TSA $[29]$ and ETT $[53]$ manage it by freezing the feature extractor weights, and inserting some parameter-efficient

adaptation modules, lightweight enough to be trained in a few-shot manner. FLUTE $[45]$ manages it through selective fine-tuning of a tiny set of FILM $[36]$ parameters, while keeping most of them fixed. Despite this progress, the best way to manage the adaptation/generalisation trade-off in fine-tuning approaches to few-shot learning (FSL) is still an open question. For example, which layers should be fine-tuned? What kind of adapters should be inserted, and where? While PMF, TSA, ETT, FLUTE, and others provide some intuitive recommendations, we propose a more systematic approach to answer these questions.

In this paper, we advance the adaptation-based paradigm for FSL by developing a neural architecture search (NAS) algorithm to find the optimal adaptation architecture. Given an initial pre-trained feature extractor, our NAS determines the subset of the architecture that should be fine-tuned, as well as the subset of layers where adaptation modules should be inserted. We draw inspiration from recent work in NAS $[4, 7, 8, 18, 55]$ that proposes revised versions of the stochastic Single-Path One-Shot (SPOS) $[18]$ weight-sharing strategy. Specifically, given a pre-trained ResNet $[19]$ or Vision Transformer (ViT) $[11]$ , we consider a search space defined by the inclusion or non-inclusion of task-specific adapters per layer, and the freezing or fine-tuning of learnable parameters per layer. Based on this search space, we construct a supernet $[3]$ that we train by sampling a random path in each forward pass $[18]$ . Our supernet architecture is illustrated schematically in Figure 1, where the aforementioned decisions are drawn as decision nodes ( $\diamondsuit$ ), and possible paths are marked in dotted lines.

While the supernet training remains somewhat similar to the standard NAS approaches, the subsequent search poses new challenges due to the inherent characteristics of the FSL setting. Specifically, as cross-domain FSL considers a number of datasets including novel domains at test time, it becomes questionable whether searching for a single model – which is the prevalent paradigm in NAS $[5, 27, 31, 49]$ – is the best choice. On the other hand, per-episode architecture selection is too slow and might overfit to the small support set.

Motivated by these challenges, we propose a novel NAS algorithm that shortlists a small number of architecturally diverse configurations at training time, but defers the final selection until the dataset and episode is known at test time. We empirically show that this is not only computationally efficient, but also improves results noticeably, especially when only a limited amount of domains is available at training time. We term our method Neural Fine-Tuning Search (NFTS).

NFTS defines a generic search space that is relevant to both major architecture families (i.e., convolutional networks and transformers), and the choice of which specific adapter modules to consider is a hyperparameter, rather than a hard constraint. In this paper, we consider using adapter modules that are currently state-of-the-art for ResNets and ViTs (TSA and ETT, respectively), but more adaptation architectures can be added to the search space.

Our contributions are summarised as follows: (i) We provide the first systematic Auto-ML approach to finding the optimal adaptation strategy that trades-off adaptation flexibility and overfitting risk in multi-domain FSL. (ii) Our novel NFTS algorithm automatically determines which layers should be frozen or adapted, and where new adaptation parameters should be inserted for best few-shot adaptation. (iii) We advance the state-of-the-art in the well-established and challenging Meta-Dataset $[46]$ , and the more recent and diverse Meta-Album $[47]$ benchmarks.

# 2. Related Work

# 2.1. Adaptation for Few-shot Learning

Gradient-Based Adaptation Parameter-efficient adaptation modules have been previously applied for multidomain learning, and transfer learning. A seminal example of this are Residual Adapters $[39]$ , which are lightweight 1x1 convolutional filters added to ResNet blocks. They were initially proposed for multi-domain learning, but are also useful for FSL, by providing the ability to update the feature extractor while being lightweight enough to avoid severe overfitting in the few-shot regime. Task-Specific Adapters (TSA) $[29]$ use such adapters together with a URL $[28]$ pre-trained backbone to achieve state of the art results for CNNs on the Meta-Dataset benchmark $[46]$ . Meanwhile, prompt $[22]$ and prefix $[30]$ tuning are established examples of parameter-efficient adaptation for transformer architectures for similar reasons. In FSL, Efficient Transformer Tuning (ETT) $[53]$ apply a similar strategy to few-shot ViT adaptation using a DINO $[6]$ pre-trained backbone.

PMF [21], FLUTE [45] and FT [10] focus on adaptation of existing parameters without inserting new ones. To manage the adaptation/overfitting trade-off in the few-shot regime, PMF fine-tunes the whole ResNet or ViT backbone, but with carefully-managed learning rates. Meanwhile, FLUTE hand-picks a set of FILM parameters with a modified ResNet backbone for few-shot fine-tuning, while keeping the majority of the feature extractor frozen.

All of the methods above make heuristic choices about where to place adapters within the backbone, or for which parameters to allow/disallow fine-tuning. However, as different input layers represent different features $[7, 54]$ , there is scope for making better decisions about which features to update. Furthermore, in the multi-domain setting different target datasets may benefit from different choices about which modules to update. This paper takes an Auto-ML NAS-based approach to systematically address this issue.

Feed-Forward Adaptation The aforementioned meth-

ods all use stochastic gradient descent to update the features during adaptation. We briefly mention CNAPS [40] and derivatives [2] as a competing line of work that use feed-forward networks to modulate the feature extraction process. However, these dynamic feature extractors are less able to generalise to completely novel domains than gradient-based methods [17], as the adaptation module itself suffers from an out of distribution problem.

# 2.2. Neural architecture search

Neural Architecture Search (NAS) is a large and well-studied topic $[14]$ which we do not attempt to review in detail here. Mainstream NAS aims to discover new architectures that achieve high performance when training on a single dataset from scratch in a many-shot regime. To this end, research aims to develop faster search algorithms $[1, 18, 31, 52]$ , and more effective search spaces $[9, 15, 37, 56]$ . We build upon the popular SPOS $[18]$ family of search strategies that encapsulate the entire search space inside a supernet that is trained by sampling paths randomly, and a search algorithm then determines the optimal path.

We develop an instantiation of the SPOS strategy for the multi-domain FSL problem. We construct a search space suited for parameter-efficient adaptation of a prior architecture to a new set of categories, and extend SPOS to learn on a suite of datasets, and efficiently generalise to novel datasets. This is different than the traditional SPOS paradigm of training and evaluating on the same dataset and same set of categories.

While there exist some recent NAS works that try to address a similar “train once, search many times” problem efficiently $[4, 26, 34, 35]$ , naively using these approaches has two serious shortcomings: i) They assume that after the initial supernet training, subsequent searches do not involve any training (e.g., a search is only performed to consider a different FLOPs constraint while accuracy of different configurations is assumed to stay the same) and thus can be done efficiently – this is not true in the FSL setting as explained earlier. ii) Even if naively searching for each dataset at test time were computationally feasible, the few-shot nature of our setting poses a significant risk of overfitting the architecture to the small support set considered in each episode.

# 3. Neural Fine-Tuning Search

# 3.1. Few-Shot Learning Background

Let $\mathcal{D} = \{\mathcal{D}_i\}_{i=1}^D$ be the set of $D$ classification domains, and $\bar{\mathcal{D}} = \{X,Y\} \in \mathcal{D}$ a task containing $n$ samples along with their designated true labels $\{\bar{X},\bar{Y}\} = \{x_j,y_j\}_{j=1}^n$ . Few-shot classification is defined as the problem of learning to correctly classify a query set $\mathcal{Q} = \{X_{\mathcal{Q}},Y_{\mathcal{Q}}\} \sim \bar{\mathcal{D}}$ by training on a support set $\mathcal{S} = \{X_{\mathcal{S}},Y_{\mathcal{S}}\} \sim \bar{\mathcal{D}}$ that contains very few examples. This can be achieved by finding the parameters $\theta$ of a classifier $f_{\theta}$ with the objective

![](images/13f81ad02cc7bc3f19e017365daba5d68174040c682f95b404f0216bb7020327.jpg)

<details>
<summary>heatmap</summary>

|        | g1   | g2   | g3   | g4   | g5   | g6   | g7   | g8   | g9   | g10  | g11  | g12  | g13  | g14  | g15  | g16  |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| α      | +0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | +0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 |
| φ'     | +0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 |
</details>

(a) Correlation between inclusion/non-inclusion of learnable parameters $\alpha$ and $\phi'$ , and validation performance.   
![](images/edfe9c295131450f0996481ff453de84abb316c0be759f75e4aeb3cec3b827a9.jpg)  
(b) Top 3 performing paths subject to diversity constraint.   
Figure 2: Qualitative analysis of our architecture search. Fig. 2a summarises the whole search space by answering the question: How important is to adapt ( $\alpha$ ) or fine-tune ( $\phi'$ ) each block? The color of each square indicates the point-biserial correlation (over all searched architectures) between adapting/fine-tuning layer $g_{i}$ and validation performance. Fig. 2b shows the top 3 performing candidates subject to a diversity constraint, after 15 generations of evolutionary search. Dark blue indicates that the layer is adapted/fine-tuned and light blue that it is not.

$$
\arg \max _ {\theta} \prod_ {\mathcal {D}} p (Y _ {\mathcal {Q}} | f _ {\theta} (\mathcal {S}, X _ {\mathcal {Q}})). \tag {1}
$$

In practice, if $\theta$ is randomly initialised and trained using stochastic gradient descent on a small support set S, it will overfit and fail to generalise to Q. To address this issue, one can exploit knowledge transfer from some seen classes to the novel classes. Formally, each domain $\bar{D}$ is partitioned into two disjoint sets $\bar{D}_{train}$ and $\bar{D}_{test}$ , which are commonly referred to as “meta-train” and “meta-test”, respectively. The labels in these sets are also disjoint, i.e., $Y_{train} \cap Y_{test} = \emptyset$ . In that case, $\theta$ is trained by maximising the objective in Eq. 1 using the meta-train set, but the overall objective is to perform adequately when transferring knowledge to meta-test.

The knowledge transferred from meta-train to meta-test can take various forms $[20]$ . As discussed earlier, we aim to generalise a family of few-shot methods $[21, 29, 53]$ where parameters $\theta$ are transferred before a subset of them $\phi \subset \theta$ are fine-tuned; and possibly extended by attaching additional “adapter” parameters $\alpha$ that are trained for the target task. For meta-test, Eq. 1 can therefore be rewritten as

$$
\underset {\alpha , \phi} {\arg \max} \prod_ {\mathcal {D} _ {\text { test }}} p (Y _ {\mathcal {Q}} | f _ {\alpha , \phi} (\mathcal {S}, X _ {\mathcal {Q}})), \tag {2}
$$

In this paper, we focus on the problem of finding the optimal adaptation strategy in terms of (i) the optimal subset

![](images/0743a06760c305e2c08147225923d04538e1ddf65bbbce06155e01b8d39fb9fa.jpg)

<details>
<summary>bubble</summary>

| x | y | color |
|---|---|---|
| 0.1 | 0.8 | orange |
| 0.2 | 0.75 | green |
| 0.3 | 0.65 | yellow |
| 0.4 | 0.55 | red |
| 0.5 | 0.45 | light green |
| 0.6 | 0.35 | orange |
| 0.7 | 0.25 | green |
| 0.8 | 0.15 | yellow |
| 0.9 | 0.05 | red |
| 1.0 | 0.95 | light green |
| 1.1 | 0.85 | orange |
| 1.2 | 0.75 | green |
| 1.3 | 0.65 | yellow |
| 1.4 | 0.55 | red |
| 1.5 | 0.45 | light green |
| 1.6 | 0.35 | orange |
| 1.7 | 0.25 | green |
| 1.8 | 0.15 | yellow |
| 1.9 | 0.05 | red |
| 2.0 | 0.95 | light green |
| 2.1 | 0.85 | orange |
| 2.2 | 0.75 | green |
| 2.3 | 0.65 | yellow |
| 2.4 | 0.55 | red |
| 2.5 | 0.45 | light green |
| 2.6 | 0.35 | orange |
| 2.7 | 0.25 | green |
| 2.8 | 0.15 | yellow |
| 2.9 | 0.05 | red |
| 3.0 | 0.95 | light green |
| 3.1 | 0.85 | orange |
| 3.2 | 0.75 | green |
| 3.3 | 0.65 | yellow |
| 3.4 | 0.55 | red |
| 3.5 | 0.45 | light green |
| 3.6 | 0.35 | orange |
| 3.7 | 0.25 | green |
| 3.8 | 0.15 | yellow |
| 3.9 | 0.05 | red |
| 4.0 | 0.95 | light green |
| 4.1 | 0.85 | orange |
| 4.2 | 0.75 | green |
| 4.3 | 0.65 | yellow |
| 4.4 | 0.55 | red |
| 4.5 | 0.45 | light green |
| 4.6 | 0.35 | orange |
| 4.7 | 0.25 | green |
| 4.8 | 0.15 | yellow |
| 4.9 | 0.05 | red |
| 5.0 | 0.95 | light green |
| 5.1 | 0.85 | orange |
| 5.2 | 0.75 | green |
| 5.3 | 0.65 | yellow |
| 5.4 | 0.55 | red |
| 5.5 | 0.45 | light green |
| 5.6 | 0.35 | orange |
| 5.7 | 0.25 | green |
| 5.8 | 0.15 | yellow |
| 5.9 | 0.05 | red |
| 6.0 | 0.95 | light green |
| 6.1 | 0.85 | orange |
| 6.2 | 0.75 | green |
| 6.3 | 0.65 | yellow |
| 6.4 | 0.55 | red |
| 6.5 | 0.45 | light green |
| 6.6 | 0.35 | orange |
| 6.7 | 0.25 | green |
| 6.8 | 0.15 | yellow |
| 6.9 | 0.05 | red |
| 7.0 | 0.95 | light green |
| 7.1 | 0.85 | orange |
| 7.2 | 0.75 | green |
| 7.3 | 0.65 | yellow |
| 7.4 | 0.55 | red |
| 7.5 | 0.45 | light green |
| 7.6 | 0.35 | orange |
| 7.7 | 0.25 | green |
| 7.8 | 0.15 | yellow |
| 7.9 | 0.05 | red |
| 8.0 | 0.95 | light green |
| 8.1 | 0.85 | orange |
| 8.2 | 0.75 | green |
| 8.3 | 0.65 | yellow |
| 8.4 | 0.55 | red |
| 8.5 | 0.45 | light green |
| 8.6 | 0.35 | orange |
| 8.7 | 0.25 | green |
| 8.8 | 0.15 | yellow |
| 8.9 | 0.05 | red |
| 9.0 | 0.95 | light green |
| 9.1 | 0.85 | orange |
| 9.2 | 0.75 | green |
| 9.3 | 0.65 | yellow |
| 9.4 | 0.55 | red |
| 9.5 | 0.45 | light green |
| 9.6 | 0.35 | orange |
| 9.7 | 0.25 | green |
| 9.8 | 0.15 | yellow |
| 9.9 | 0.05 | red |
|10<fcel>-<fcel>-<nl>
</details>

Generation 1

![](images/13f79ed7f76feeef7f16d26463fda75bc41709e61e5e57817e2e271307eef43d.jpg)

<details>
<summary>bubble</summary>

| X | Y | Size |
|---|---|---|
| 0.1 | 0.9 | 5 |
| 0.2 | 0.85 | 4 |
| 0.3 | 0.8 | 6 |
| 0.4 | 0.75 | 3 |
| 0.5 | 0.7 | 4 |
| 0.6 | 0.65 | 5 |
| 0.7 | 0.6 | 3 |
| 0.8 | 0.55 | 4 |
| 0.9 | 0.5 | 6 |
| 1.0 | 0.45 | 3 |
| 1.1 | 0.4 | 5 |
| 1.2 | 0.35 | 4 |
| 1.3 | 0.3 | 6 |
| 1.4 | 0.25 | 3 |
| 1.5 | 0.2 | 4 |
| 1.6 | 0.15 | 5 |
| 1.7 | 0.1 | 3 |
| 1.8 | 0.05 | 4 |
| 1.9 | 0.0 | 6 |
| 2.0 | -0.05 | 3 |
| 2.1 | -0.1 | 4 |
| 2.2 | -0.15 | 5 |
| 2.3 | -0.2 | 3 |
| 2.4 | -0.25 | 4 |
| 2.5 | -0.3 | 6 |
| 2.6 | -0.35 | 3 |
| 2.7 | -0.4 | 4 |
| 2.8 | -0.45 | 5 |
| 2.9 | -0.5 | 3 |
| 3.0 | -0.55 | 4 |
| 3.1 | -0.6 | 6 |
| 3.2 | -0.65 | 3 |
| 3.3 | -0.7 | 4 |
| 3.4 | -0.75 | 5 |
| 3.5 | -0.8 | 3 |
| 3.6 | -0.85 | 4 |
| 3.7 | -0.9 | 6 |
| 3.8 | -0.95 | 3 |
| 3.9 | -1.0 | 4 |
| 4.0 | -1.05 | 5 |
| 4.1 | -1.1 | 3 |
| 4.2 | -1.15 | 4 |
| 4.3 | -1.2 | 6 |
| 4.4 | -1.25 | 3 |
| 4.5 | -1.3 | 4 |
| 4.6 | -1.35 | 5 |
| 4.7 | -1.4 | 3 |
| 4.8 | -1.45 | 4 |
| 4.9 | -1.5 | 6 |
| 5.0 | -1.55 | 3 |
| 5.1 | -1.6 | 4 |
| 5.2 | -1.65 | 5 |
| 5.3 | -1.7 | 3 |
| 5.4 | -1.75 | 4 |
| 5.5 | -1.8 | 6 |
| 5.6 | -1.85 | 3 |
| 5.7 | -1.9 | 4 |
| 5.8 | -1.95 | 5 |
| 5.9 | -2.0 | 3 |
| 6.0 | -2.05 | 4 |
| 6.1 | -2.1 | 6 |
| 6.2 | -2.15 | 3 |
| 6.3 | -2.2 | 4 |
| 6.4 | -2.25 | 5 |
| 6.5 | -2.3 | 3 |
| 6.6 | -2.35 | 4 |
| 6.7 | -2.4 | 6 |
| 6.8 | -2.45 | 3 |
| 6.9 | -2.5 | 4 |
| 7.0 | -2.55 | 5 |
| 7.1 | -2.6 | nan |
| ... (or not labeled)<lcel><lcel><nl>
</details>

Generation 5

![](images/085688f5cb30f6fc00cfeb1d86ae45e1072ce7eaf7564894b8a25d1ebe80583e.jpg)

<details>
<summary>bubble</summary>

| x | y | size |
|---|---|------|
| 0.1 | 0.9 | 10 |
| 0.2 | 0.85 | 15 |
| 0.3 | 0.75 | 20 |
| 0.4 | 0.65 | 25 |
| 0.5 | 0.55 | 30 |
| 0.6 | 0.45 | 35 |
| 0.7 | 0.35 | 40 |
| 0.8 | 0.25 | 45 |
| 0.9 | 0.15 | 50 |
| 1.0 | 0.05 | 55 |
| 1.1 | 0.95 | 60 |
| 1.2 | 0.85 | 65 |
| 1.3 | 0.75 | 70 |
| 1.4 | 0.65 | 75 |
| 1.5 | 0.55 | 80 |
| 1.6 | 0.45 | 85 |
| 1.7 | 0.35 | 90 |
| 1.8 | 0.25 | 95 |
| 1.9 | 0.15 | 100 |
| 2.0 | 0.05 | 105 |
| 2.1 | 0.95 | 110 |
| 2.2 | 0.85 | 115 |
| 2.3 | 0.75 | 120 |
| 2.4 | 0.65 | 125 |
| 2.5 | 0.55 | 130 |
| 2.6 | 0.45 | 135 |
| 2.7 | 0.35 | 140 |
| 2.8 | 0.25 | 145 |
| 2.9 | 0.15 | 150 |
| 3.0 | 0.05 | 155 |
| 3.1 | 0.95 | 160 |
| 3.2 | 0.85 | 165 |
| 3.3 | 0.75 | 170 |
| 3.4 | 0.65 | 175 |
| 3.5 | 0.55 | 180 |
| 3.6 | 0.45 | 185 |
| 3.7 | 0.35 | 190 |
| 3.8 | 0.25 | 195 |
| 3.9 | 0.15 | 200 |
| 4.0 | 0.95 | 205 |
| 4.1 | 0.85 | 210 |
| 4.2 | 0.75 | 215 |
| 4.3 | 0.65 | 220 |
| 4.4 | 0.55 | 225 |
| 4.5 | 0.45 | 230 |
| 4.6 | 0.35 | 235 |
| 4.7 | 0.25 | 240 |
| 4.8 | 0.15 | 245 |
| 4.9 | 0.95 | 250 |
| 5.0 | 0.85 | 255 |
| 5.1 | 0.75 | 260 |
| 5.2 | 0.65 | 265 |
| 5.3 | 0.55 | 270 |
| 5.4 | 0.45 | 275 |
| 5.5 | 0.35 | 280 |
| 5.6 | 0.25 | 285 |
| 5.7 | 0.15 | 290 |
| 5.8 | 0.95 | 295 |
| 5.9 | 0.85 | 300 |
| 6.0 | 0.75 | 305 |
| 6.1 | 0.65 | 310 |
| 6.2 | 0.55 | 315 |
| 6.3 | 0.45 | 320 |
| 6.4 | 0.35 | 325 |
| 6.5 | 0.25 | 330 |
| 6.6 | 0.15 | 335 |
| 6.7 | -0.95 | -18 |
| -18: The image contains only a legend entry for the data series.
</details>

Generation 15

![](images/e341be3df853527ea021a96679d53d1f89d65fb4185a0ce4419043db474abfa0.jpg)  
Figure 3: Population of paths(candidate architectures) in the search space after 1, 5, and 15 generations of evolutionary search. Each dot is a 2-d TSNE projection of the binary vector representing an architecture, and its color shows the validation performance for that architecture. The supernet contains a wide variety of models in terms of validation performance, and the search algorithm converges to a well-performing population. The top 3 performing paths that are given in 2b are highlighted in the far right figure (Generation 15) in purple outline.

of parameters $\phi \subset \theta$ that need to be fine-tuned, and (ii) the optimal task-specific parameters $\alpha$ to add.

# 3.2. Defining the search space

Let $g_{\phi_k}$ be the minimal unit for adaptation in an architecture. We consider these to be the repeated units in contemporary deep architectures, e.g., a convolutional layer in a ResNet, or a self-attention block in a ViT. If the feature extractor $f_\theta$ comprises of $K$ such units with learnable parameters $\phi_k$ , then we denote $\theta = \bigcup_{k=1}^{K} \phi_k$ , assuming all other parameters are kept fixed. For brevity in notation we will now omit the indices and refer to every such layer as $g_\phi$ . Following the state-of-the-art [21, 29, 45, 53], let us also assume that task-specific adaptation can be performed either by inserting additional adapter parameters $\alpha$ into $g_\phi$ , or by fine-tuning the layer parameters $\phi$ .

This allows us to define the search space as two independent binary decisions per layer: (i) The inclusion or non-inclusion of an adapter module attached to $g_{\phi}$ , and (ii) the decision of whether to use the pre-trained parameters $\phi$ , or replace them with their fine-tuned counterparts $\phi'$ . The size of the search space is, therefore, $(2^{2})^{K} = 4^{K}$ . For ResNets, we use the proposed adaptation architecture of TSA [29], where a residual adapter $h_{\alpha}$ , parameterised by $\alpha$ , is con-

<table><tr><td></td><td> $g_{\phi,\phi',\alpha}(x)$  (ResNet)</td><td> $g_{\phi,\phi',\alpha}(x)$  (ViT)</td></tr><tr><td> $\phi,-$ </td><td> $g_{\phi}(x)$ </td><td> $z(A_{qkv}[q;g_{\phi}(x)])$ </td></tr><tr><td> $\phi,\alpha$ </td><td> $g_{\phi}(x)+h_{\alpha}(x)$ </td><td> $z(A_{qkv}[q;g_{\phi}(x)]+h_{\alpha1})+h_{\alpha2}$ </td></tr><tr><td> $\phi',-$ </td><td> $g_{\phi'}(x)$ </td><td> $z(A_{qkv}[q;g_{\phi'}(x)])$ </td></tr><tr><td> $\phi',\alpha$ </td><td> $g_{\phi'}(x)+h_{\alpha}(x)$ </td><td> $z(A_{qkv}[q;g_{\phi'}(x)]+h_{\alpha1})+h_{\alpha2}$ </td></tr></table>

Table 1: The search space, as described in Section 3.2. When sampling a layer $g_{\phi, \phi', \alpha}$ , it can be sampled in one of the following variants: (i) $\phi$ : fixed pre-trained parameters, no adaptation, (ii) $\alpha$ : fixed pre-trained parameters, with adaptation, (iii) $\phi'$ : fine-tuned parameters, no adaptation, (iv) $\phi'$ , $\alpha$ fine-tuned-parameters, with adaptation.

# Algorithm 1: Supernet training.

Input: Supernet $f_{\theta, \alpha, \phi'}$ . Datasets $\mathcal{D}$ . Step sizes $\eta_1$ , $\eta_2$ . Path pool $P$ . Prototypical loss $\mathcal{L}$ (Eq. 5).

Output: Trained supernet $f_{\theta, \alpha, \phi'}$ .

# repeat

Sample dataset $\bar{\mathcal{D}}\sim \mathcal{D}$

Sample episode $\mathcal{S}$ , $\mathcal{Q} \sim \bar{\mathcal{D}}$

Sample path $p\sim P$ with learnable parameters

$\alpha_{p},\phi_{p}^{\prime}$ and frozen parameters $\phi_p\subset \theta$

$\alpha_{p}\longleftarrow \alpha_{p} - \eta_{1}\nabla_{\alpha_{p}}\mathcal{L}(f_{\theta ,\alpha ,\phi^{\prime}}^{p},\mathcal{S},\mathcal{Q})$

$\phi_{p}^{\prime}\longleftarrow \phi_{p}^{\prime} - \eta_{2}\nabla_{\phi_{p}^{\prime}}\mathcal{L}(f_{\theta ,\alpha ,\phi^{\prime}}^{p},\mathcal{S},\mathcal{Q})$

until prototypical loss converges

nected to $g_{\phi}$

$$
g _ {\phi , \phi^ {\prime}, \alpha} (x) = g _ {\phi , \phi^ {\prime}} (x) + h _ {\alpha} (x), \tag {3}
$$

where $x \in R^{W,H,C}$ . For ViTs, we use the proposed adaptation architecture of ETT [53], where a tuneable prefix is prepended to the multi-head self-attention module $A_{qkv}$ , and a residual adapter is appended to both $A_{qkv}$ and the feed-forward module z in each decoder block

$$
g _ {\phi , \phi^ {\prime}, \alpha} (x) = z (A _ {q k v} [ q ; g _ {\phi , \phi^ {\prime}} (x) ] + h _ {\alpha 1}) + h _ {\alpha 2}, \tag {4}
$$

where $x \in R^{D}$ and $[\cdot; \cdot]$ denotes the concatenation operation. Note that in the case of ViTs the adapter is not a function of the input features, but simply an added offset.

Irrespective of the architecture, every layer $g_{\phi,\phi',\alpha}$ is parameterised by three sets of parameters, $\phi$ , $\phi'$ , and $\alpha$ , denoting the initial parameters, fine-tuned parameters and adapter parameters respectively. Consequently, when sampling a configuration (i.e., path) from that search space, every such layer can be sampled as one of the variants listed in Table 1.

# 3.3. Training the supernet

Following SPOS [18], our search space is actualised in the form of a supernet $f_{\theta,\alpha,\phi'}$ ; a “super” architecture that

contains all possible architectures derived from the decisions detailed in Section 3.2. It is parameterised by: (i) $\theta$ , the frozen parameters from the backbone architecture $f_{\theta}$ , (ii) $\alpha$ , from the adapters $h_{\alpha}$ , and (iii) $\phi'$ , from the fine-tuned parameters per layer $g_{\phi, \phi', \alpha}$ .

We use a prototypical loss $\mathcal{L}(f,S,Q)$ as the core objective during supernet training and the subsequent search and fine-tuning.

$$
\mathcal {L} (f, \mathcal {S}, \mathcal {Q}) = \frac {1}{| \mathcal {Q} |} \sum_ {i = 1} ^ {| \mathcal {Q} |} \log \frac {e ^ {- d _ {c o s} (C _ {\mathcal {Q} _ {i}} , f (\mathcal {Q} _ {i}))}}{\sum_ {j = 1} ^ {| C |} e ^ {- d _ {c o s} (C _ {j} , f (\mathcal {Q} _ {i}))}}, \tag {5}
$$

where $C_{Q_{i}}$ denotes the embedding of the class centroid that corresponds to the true class of $Q_{i}$ , and $d_{cos}$ denotes the cosine distance. The set of class centroids C is computed as the mean embeddings of support examples that belong to the same class:

$$
C = \left\{\frac {1}{| \mathcal {S} ^ {y = l} |} \sum_ {i = 1} ^ {| \mathcal {S} |} f (\mathcal {S} _ {i} ^ {y = l}) \right\} _ {l = 1} ^ {L}, \tag {6}
$$

where $L$ denotes the number of unique labels in $\mathcal{S}$ .

For supernet training specifically, let P be a set of size $4^{K}$ , enumerating all possible sequences of K layers that can be sampled from the search space. Denoting a path sampled from the supernet as $f_{\theta,\alpha,\phi'}^{p}$ , we minimise an expectation of the loss in Eq. 5 over multiple episodes and paths, so the final objective becomes:

$$
\underset {\alpha , \phi^ {\prime}} {\arg \min} \mathbb {E} _ {p \sim P} \mathbb {E} _ {\mathcal {S}, \mathcal {Q}} \mathcal {L} (f _ {\theta , \alpha , \phi^ {\prime}} ^ {p}, \mathcal {S}, \mathcal {Q}). \tag {7}
$$

Algorithm 1 summarises the supernet training algorithm in pseudocode.

# 3.4. Searching for an optimal path

A supernet $f_{\theta,\alpha,\phi'}$ trained with the method described in Section 3.3 contains $4^{K}$ models, intertwined via weight sharing. As explained in Section 1, our goal is to search for the best-performing one, but the main challenge is related to the fact that we do not know what data is going to be used for adaptation at test time. One extreme approach, would be to search for a single solution at training time and simply use it throughout the entire test, regardless of the potential domain shift. Another, would be to defer the search and perform it from scratch each time a new support set is given to us at test time. However, both have their shortcomings. As such, we propose a generalization of this process where searching is split into two phases – one during training, and a subsequent one during testing.

Meta-training time. The search is responsible for pre-selecting a set of N models from the entire search space. Its main purpose is to mitigate potential overfitting that can happen at test time, when only a small amount of data is

Algorithm 2: Training time evolutionary search.   
Input: Supernet $f_{\theta, \alpha, \phi'}$ . Datasets $\mathcal{D}$ . Step sizes $\eta_1$ , $\eta_2$ . Prototypical loss $\mathcal{L}$ (Eq. 5). NCC accuracy $A$ (Eq. 11).

Output: Optimal path $p^*$ .

Initialise population $P$ randomly

Initialise fitness of $P$ as $\Psi_P \longleftarrow 0$ repeat
    Sample episodes from all datasets $S, Q \sim D$ for each candidate $p \in P$ do
    for a small number of epochs do $\alpha_p \longleftarrow \alpha_p - \eta_1 \nabla_{\alpha_p} \mathcal{L}(f_{\theta, \alpha, \phi'}^p, S, S)$ $\phi_p' \longleftarrow \phi_p' - \eta_2 \nabla_{\phi_p'} \mathcal{L}(f_{\theta, \alpha, \phi'}^p, S, S)$ end $\Psi_p \longleftarrow A(f_{\theta, \alpha, \phi'}^p, S, Q)$ end
    offspring $\longleftarrow$ recombine the $M$ best candidates of $P$ w.r.t. $\Psi_P$ $P \longleftarrow P + \text{offspring}$ eliminate the $M$ worst candidates of $P$ w.r.t. $\Psi_P$ until population fitness converges or max. iterations

available, while providing enough diversity to successfully adjust the architecture to the diverse set of test domains. Formally, we search for a sequence of paths $(p_{1}, p_{2}, ..., p_{N})$ where:

$$
p _ {k} = \underset {p \in P} {\arg \max} \mathbb {E} _ {\mathcal {S}, \mathcal {Q}} A (f _ {\theta , \alpha^ {*}, \phi^ {\prime *}} ^ {p}, \mathcal {S}, \mathcal {Q}), \quad \text { s.t. } \tag {8}
$$

$$
\alpha^ {*}, \phi^ {\prime *} = \underset {\alpha , \phi^ {\prime}} {\arg \min} \mathcal {L} (f _ {\theta , \alpha , \phi^ {\prime}} ^ {p}, \mathcal {S}, \mathcal {S}) \tag {9}
$$

$$
\forall_ {j = 1, \dots , k - 1} d _ {\cos} (p _ {k}, p _ {j}) \geq T, \tag {10}
$$

where T denotes a scalar threshold for the cosine distance between paths $p_{k}$ and $p_{j}$ , and A is the classification accuracy of a nearest centroid classifier (NCC) [43],

$$
A (f, \mathcal {S}, \mathcal {Q}) = \frac {1}{| \mathcal {Q} |} \sum_ {i = 1} ^ {| \mathcal {Q} |} [ \underset {j} {\arg \min} d _ {c o s} (C _ {\mathcal {Q} _ {j}}, f (\mathcal {Q} _ {i})) = Y _ {\mathcal {Q} _ {i}} ]. \tag {11}
$$

Noticeably, we measure accuracy of a solution using a query set, after fine-tuning on a separate support set (Eq. 9), then average across multiple episodes to avoid overfitting to a particular support set (Eq. 8). We also employ a diversity constraint, in the form of cosine distance between binary encodings of selected paths (Eq. 10), to allow for sufficient flexibility in the following test time search.

To efficiently obtain sequence $\{p_{1},\ldots,p_{N}\}$ , we use evolutionary search to find points that maximise Eq. 8, and afterwards select the N best performers from the evolutionary

search history that satisfy the constraint in Eq. 10. Algorithm 2 summarises training-time search.

Meta-testing time. For a given meta-test episode, we decide which one of the pre-selected N models is best suited for adaptation on the given support set data. It acts as a fail-safe to counteract the bias of the initial selection made at training time in cases when the support set might be particularly out-of-domain. Formally, the final path $p^{*}$ to be used in a particular episode is defined as:

$$
p ^ {*} = \underset {p \in \{p _ {1}, \dots , p _ {N} \}} {\arg \min} \mathcal {L} (f _ {\theta , \alpha^ {*}, \phi^ {\prime *}} ^ {p}, \mathcal {S}, \mathcal {S}), \quad \text { s.t. } \tag {12}
$$

$$
\alpha^ {*}, \phi^ {\prime *} = \underset {\alpha , \phi^ {\prime}} {\arg \min} \mathcal {L} (f _ {\theta , \alpha , \phi^ {\prime}} ^ {p}, \mathcal {S}, \mathcal {S}) \tag {13}
$$

Noticeably, we test each of the N models by fine-tuning it on the support set (Eq. 13) and testing its performance on the same support set (Eq. 12). This is because the support set is the only source of data we have at test time and we cannot extract a disjoint validation set from it without risking the quality of the fine-tuning process. It is important to note that, while this step risks overfitting, the pre-selection of models at training time, as described previously, should already limit the subsequent search to only models that are unlikely to overfit. Since N is kept small in our experiments, we use a naive grid search to find $p^{*}$ .

This approach is a generalization of the existing NAS approaches, as it recovers both when N = 1 or $N = 4^{K}$ . Our claim is that intermediate values of N are more likely to give us better results than any of the extremes, due to the reasons mentioned earlier. In particular, we would expect pre-selecting $1 < N \ll 4^{K}$ models to introduce reasonable overhead at test time while improving results, especially in cases when exposure to different domains might be limited at training time. In our evaluation we compare N = 3 and N = 1 to test this hypothesis. We do not include comparison to $N = 4^{K}$ as it is computationally infeasible in our setting (performing equivalent of training time search for each test episode would require us to fine-tune $\approx 14 * 10^{6}$ models in total).

# 4. Experiments

# 4.1. Experimental setup

Evaluation on Meta-Dataset We evaluate NFTS on the extended version of Meta-Dataset $[40, 46]$ , currently the most commonly used benchmark for few-shot classification, consisting of 13 publicly available datasets: FGVC Aircraft, CU Birds, Describable Textures (DTD), FGVCx Fungi, ImageNet, Omniglot, QuickDraw, VGG Flowers, CIFAR-10/100, MNIST, MSCOCO, and Traffic Signs. There are 2 evaluation protocols: single domain learning and multi-domain learning. In the single domain setting, only ImageNet is seen during training and meta-training, while in the multi-domain setting the first eight datasets are seen (FGVC Aircraft to VGG Flower). For meta-testing at least 600 episodes are sampled for each domain.

Evaluation on Meta-Album Further, we evaluate NFTS on the more recently introduced Meta-Album $[47]$ . Meta-Album is more diverse than Meta-Dataset. We use the currently available Sets 0-2, which contain over 1000 unique labels across 30 datasets spanning 10 domains including microscopy, remote sensing, manufacturing, plant disease, character recognition and human action recognition tasks, etc. Unlike Meta-Dataset, where their default evaluation protocol is variable-way variable-shot, Meta-Album evaluation follows a 5-way variable-shot setting, where the number of shots is typically 1, 5, 10 and 20. For meta-testing, results are averaged over 1800 episodes.

Architectures We employ two different backbone architectures, a ResNet-18 [19] and a ViT-small [11]. Following TSA [29], the ResNet-18 backbone is pre-trained on the seen domains with the knowledge-distillation method URL [28] and, following ETT [53], the ViT-small backbone is pre-trained on the seen portion of ImageNet with the self-supervised method DINO [6]. We consider TSA residual adapters [29, 39] for ResNet and Prefix Tuning [30, 53] adapters for ViT. This is mainly to enable direct comparison with prior work on the same base architectures that use exactly these same adapter families, without introducing new confounders in terms of mixing adapter types [29, 53]. However our framework is flexible, meaning it can accept any adapter type, or even multiple types in its search space.

# 4.2. Comparison to state-of-the-art

Meta-Dataset The results on Meta-Dataset are shown in Table 2 and Table 3 for single-domain and multi-domain training setting respectively. We can see that NFTS obtains the best average performance across all the competitor methods for both ResNet and ViT architectures. The margins over prior state-of-the-art are often substantial for this benchmark with +1.9% over TSA in ResNet-18 single domain, +2.3% in multi-domain and +1.6% over ETT in VIT-small single domain. The increased margin in the multidomain case is intuitive, as our framework has more data with which to learn the optimal path(s).

We re-iterate that PMF, ETT, and TSA are special cases of our search space corresponding respectively to: (i) Fine-tune all and include no adapters, (ii) Include ETT adapters at every layer while freezing all backbone weights and (iii) Include TSA adapters at every layer while freezing all backbone weights. We also share initial pre-trained backbones with ETT and TSA (but not PMF, as it uses a stronger pre-trained model with additional data). Thus the margins achieved over these competitors are attributable to our systematic approach to finding suitable architectures in terms of where to fine-tune and where to insert new adapter pa

<table><tr><td></td><td>Method</td><td>Aircrafts</td><td>Birds</td><td>DTD</td><td>Fungi</td><td>ImageNet</td><td>Omniglot</td><td>QuickDraw</td><td>Flowers</td><td>CIFAR-10</td><td>CIFAR-100</td><td>MNIST</td><td>MSCOCO</td><td>Tr. Signs</td><td>Average</td></tr><tr><td rowspan="6">ResNet-18</td><td>FLUTE [32]</td><td>48.5</td><td>47.9</td><td>63.8</td><td>31.8</td><td>46.9</td><td>61.6</td><td>57.5</td><td>80.1</td><td>65.4</td><td>52.7</td><td>80.8</td><td>41.4</td><td>46.5</td><td>52.6</td></tr><tr><td>ProtoNet [43]</td><td>53.1</td><td>68.8</td><td>66.6</td><td>39.7</td><td>50.5</td><td>60.0</td><td>49.0</td><td>85.3</td><td>-</td><td>-</td><td>-</td><td>41.0</td><td>47.1</td><td>56.1</td></tr><tr><td>BOHB [42]</td><td>54.1</td><td>70.7</td><td>68.3</td><td>41.4</td><td>51.9</td><td>67.6</td><td>50.3</td><td>87.3</td><td>-</td><td>-</td><td>-</td><td>48.0</td><td>51.8</td><td>59.2</td></tr><tr><td>FO-MAML [46]</td><td>63.4</td><td>69.8</td><td>70.8</td><td>41.5</td><td>52.8</td><td>61.9</td><td>59.2</td><td>86.0</td><td>-</td><td>-</td><td>-</td><td>48.1</td><td>60.8</td><td>61.4</td></tr><tr><td>TSA [29]</td><td>72.2</td><td>74.9</td><td>77.3</td><td>44.7</td><td>59.5</td><td>78.2</td><td>67.6</td><td>90.9</td><td>82.1</td><td>70.7</td><td>93.9</td><td>59.0</td><td>82.5</td><td>73.3</td></tr><tr><td>NFTS</td><td>74.9</td><td>76.5</td><td>81.6</td><td>50.5</td><td>62.7</td><td>80.2</td><td>67.2</td><td>94.5</td><td>83.0</td><td>71.5</td><td>94.0</td><td>59.7</td><td>81.9</td><td>75.2</td></tr><tr><td rowspan="3">ViT-S</td><td>PMF* [21]</td><td>76.8</td><td>85.0</td><td>86.6</td><td>54.8</td><td>74.7</td><td>80.7</td><td>71.3</td><td>94.6</td><td>-</td><td>-</td><td>-</td><td>62.6</td><td>88.3</td><td>77.5</td></tr><tr><td>ETT [53]</td><td>79.9</td><td>85.9</td><td>87.6</td><td>61.8</td><td>67.4</td><td>78.1</td><td>71.3</td><td>96.6</td><td>-</td><td>-</td><td>-</td><td>62.3</td><td>85.1</td><td>77.6</td></tr><tr><td>NFTS</td><td>83.0</td><td>85.5</td><td>87.6</td><td>62.2</td><td>71.0</td><td>81.9</td><td>74.5</td><td>96.0</td><td>79.4</td><td>72.6</td><td>95.2</td><td>62.6</td><td>87.9</td><td>79.2</td></tr></table>

Table 2: Comparison to the state-of-the-art methods on Meta-Dataset. Single domain setting – only ImageNet is seen during training and search. Reporting mean accuracy over 600 episodes. \* Additional data used for training. 

<table><tr><td></td><td>Method</td><td>Aircrafts</td><td>Birds</td><td>DTD</td><td>Fungi</td><td>ImageNet</td><td>Omniglot</td><td>QuickDraw</td><td>Flowers</td><td>CIFAR-10</td><td>CIFAR-100</td><td>MNIST</td><td>MSCOCO</td><td>Tr. Signs</td><td>Average</td></tr><tr><td rowspan="8">ResNet-18</td><td>CNAPS [40]</td><td>83.7</td><td>73.6</td><td>59.5</td><td>50.2</td><td>50.8</td><td>91.7</td><td>74.7</td><td>88.9</td><td>-</td><td>-</td><td>-</td><td>39.4</td><td>56.5</td><td>66.9</td></tr><tr><td>SCNAPS [50]</td><td>82.0</td><td>74.8</td><td>68.8</td><td>46.6</td><td>58.4</td><td>91.6</td><td>76.5</td><td>90.5</td><td>74.9</td><td>61.3</td><td>94.6</td><td>48.9</td><td>57.2</td><td>69.5</td></tr><tr><td>SUR [13]</td><td>85.5</td><td>71.0</td><td>71.0</td><td>64.3</td><td>56.2</td><td>94.1</td><td>81.8</td><td>82.9</td><td>66.5</td><td>56.9</td><td>94.3</td><td>52.0</td><td>51.0</td><td>71.4</td></tr><tr><td>URT [32]</td><td>85.8</td><td>76.2</td><td>71.6</td><td>64.0</td><td>56.8</td><td>94.2</td><td>82.4</td><td>87.9</td><td>67.0</td><td>57.3</td><td>90.6</td><td>51.5</td><td>48.2</td><td>71.8</td></tr><tr><td>FLUTE [45]</td><td>82.8</td><td>75.3</td><td>71.2</td><td>48.5</td><td>58.6</td><td>92.0</td><td>77.3</td><td>90.5</td><td>75.4</td><td>62.0</td><td>96.2</td><td>52.8</td><td>63.0</td><td>72.7</td></tr><tr><td>URL [28]</td><td>89.4</td><td>80.7</td><td>77.2</td><td>68.1</td><td>58.8</td><td>94.5</td><td>82.5</td><td>92.0</td><td>74.2</td><td>63.5</td><td>94.7</td><td>57.3</td><td>63.3</td><td>76.6</td></tr><tr><td>TSA [29]</td><td>89.9</td><td>81.1</td><td>77.5</td><td>66.3</td><td>59.5</td><td>94.9</td><td>81.7</td><td>92.2</td><td>82.9</td><td>70.4</td><td>96.7</td><td>57.6</td><td>82.8</td><td>78.4</td></tr><tr><td>NFTS</td><td>90.1</td><td>83.8</td><td>82.3</td><td>68.4</td><td>61.4</td><td>94.3</td><td>82.6</td><td>92.2</td><td>83.0</td><td>75.1</td><td>95.4</td><td>58.8</td><td>81.9</td><td>80.7</td></tr><tr><td rowspan="2">ViT-S</td><td>PMF* [21]</td><td>88.3</td><td>91.0</td><td>86.6</td><td>74.2</td><td>74.6</td><td>91.8</td><td>79.2</td><td>94.1</td><td>-</td><td>-</td><td>-</td><td>62.6</td><td>88.9</td><td>83.1</td></tr><tr><td>NFTS</td><td>89.1</td><td>92.5</td><td>86.3</td><td>75.1</td><td>74.6</td><td>92.0</td><td>80.6</td><td>93.5</td><td>75.9</td><td>70.8</td><td>91.3</td><td>62.8</td><td>87.2</td><td>83.4</td></tr></table>

Table 3: Comparison to the state-of-the-art methods on Meta-Dataset. Multi-domain setting – the first 8 datasets are seen during training and search. Reporting mean accuracy over 600 episodes. \* Additional data used for training.

rameters.

Meta-Album The results on Meta-Album are shown in Figure 4 as a function of number of shots within the 5-way setting, following [47]. We can see that across the whole range of support set sizes, our NFTS dominates all of the well-tuned baselines from [47]. The margins are substantial, greater than $5\%$ at 5-way/5-shot operating point, for example. This result confirms that our framework scales to even more diverse datasets and domains than those considered previously in Meta-Dataset.

# 4.3. Ablation study

To analyse more precisely the role that our architecture search plays in few-shot performance, we also conduct an ablation study of our final model against four corners of our search space: (i) Initial model only, using a pre-trained feature extractor and simple NCC classifier, which loosely corresponds to SimpleShot $[50]$ , (ii) Full adaptation only, using a fixed feature extractor, which loosely corresponds to TSA $[29]$ , ETT $[53]$ , FLUTE $[45]$ , and others – depending on base architecture and choice of adapter, (iii) Fully fine-tuned model, which loosely corresponds to PMF $[21]$ , and (iv) Combination of full fine-tuning and adaptation. From the results in Table 4 we can see that both fine-tuning (ii), adapters (iii), and their combination (iv) give improvement on the linear readout baseline (i). However, all of them are worse than the systematically optimised adaptation architecture of NFTS.

The ablation also compares the results using the top-1 adaptation architecture found by SPOS architecture search

<table><tr><td></td><td>Method</td><td>Single Domain</td><td>Multi-Domain</td></tr><tr><td rowspan="6">ResNet-18</td><td> $\phi , -$ </td><td>67.8</td><td>67.8</td></tr><tr><td> $\phi , \alpha$ </td><td>70.4</td><td>76.5</td></tr><tr><td> $\phi', -$ </td><td>70.2</td><td>76.3</td></tr><tr><td> $\phi', \alpha$ </td><td>70.8</td><td>76.9</td></tr><tr><td>NFTS-1</td><td>73.6</td><td>80.1</td></tr><tr><td>NFTS-N</td><td>75.2</td><td>80.7</td></tr><tr><td rowspan="6">ViT-S</td><td> $\phi , -$ </td><td>71.8</td><td>71.8</td></tr><tr><td> $\phi , \alpha$ </td><td>73.8</td><td>77.3</td></tr><tr><td> $\phi', -$ </td><td>74.0</td><td>77.5</td></tr><tr><td> $\phi', \alpha$ </td><td>74.4</td><td>78.9</td></tr><tr><td>NFTS-1</td><td>78.7</td><td>83.1</td></tr><tr><td>NFTS-N</td><td>79.2</td><td>83.4</td></tr></table>

Table 4: Ablation study on Meta-Dataset comparing four special cases of the search space in terms of average accuracy: (i) $\phi, -$ : No adaptation, no fine-tuning, (ii) $\phi, \alpha$ : Adapt all, (iii) $\phi', -$ : Fine-tune all, (iv) $\phi', \alpha$ : Adapt and fine-tune all. NFTS- $1,N$ refer to conventional and deferred episode-wise NAS respectively.

against our novel progressive approach that defers the final architecture selection to an episode-wise decision. Our deferred architecture selection improves on fixing the top-1 architecture from meta-train, demonstrating the value of per-dataset/episode architecture selection (see also Sec 4.4).

# 4.4. Further analysis

The ablation study shows quantitatively the benefit of adaptation architecture search over common fixed adaptation strategies. In this Section, we aim to analyse: What

![](images/0c212ea13d29009ca3aebd21e313fe14518f48ec39e45b6424352960789e65c5.jpg)

<details>
<summary>line</summary>

| Number of shots | TrainFromScratch | FineTuning | MatchingNet | ProtoNet | FO-MAML | NFTS |
| --------------- | ---------------- | ---------- | ----------- | -------- | ------- | ---- |
| 1               | 30.5             | 41.0       | 34.0        | 38.0     | 34.5    | 44.0 |
| 5               | 38.0             | 51.0       | 45.0        | 51.0     | 44.5    | 58.0 |
| 10              | 39.5             | 53.5       | 49.5        | 56.0     | 49.0    | 60.0 |
| 20              | 40.0             | 55.0       | 52.5        | 60.0     | 51.0    | 61.0 |
</details>

Figure 4: Comparison of our method against Meta-Album baselines, as reported in Fig. 2 of their paper [47]. The setting is 5-way [1, 5, 10, 20]-shot, and accuracy scores are averaged over 1800 tasks drawn from Set0, Set1 and Set2.

kind of adaptation architecture is discovered by our NAS strategy, and how it is discovered?

Discovered Architectures We first summarise results of the entire search space in terms of which layers are preferential to fine-tune or not, and which layers are preferential to insert adapters or not in Figure 2a. The blocks indicate layers (columns) and adapters/fine-tuning (rows), with the color indicating whether that architectural decision was positively (green) or negatively (red) correlated with validation performance. We can see that the result is complex, without a simple pattern, as assumed by existing work [21, 29, 53]. That said, our NAS does discover some interpretable trends. For example, adapters should be included at early/late ResNet-18 layers and not at layers 5-9.

We next show the top three performing paths subject to diversity constraint in Figure 2b. We see that these follow the strong trends in the search space from Figure 2a. For example, they always adapt $(\alpha)$ block 14 and never adapt block 9. However, otherwise they do include diverse decisions (such as whether to fine-tune $(\phi')$ block 15) which was not strongly indicated in Figure 2a.

Finally, we analyse how our small set of N = 3 candidate architectures in Figure 2b as used during meta-test. Recall that this small set allows us to perform an efficient minimal episode-wise NAS, including for novel datasets unseen during training. The results in Table 5 show how often each architecture is selected by held out datasets during meta-test (shading), and what is the per-dataset performance using only that architecture. It shows how our approach successfully learns to select the most suitable architecture on a per-dataset basis, even for unseen datasets. This unique capability goes beyond prior work $[21, 29, 53]$ where all domains must rely on the same adaptation strategy despite their diverse adaptation needs.

<table><tr><td></td><td>CIFAR-10</td><td>CIFAR-100</td><td>MNIST</td><td>MSCOCO</td><td>Tr. Signs</td></tr><tr><td>CIFAR-10</td><td>82.0</td><td>81.2</td><td>83.3</td><td></td><td></td></tr><tr><td>CIFAR-100</td><td>75.9</td><td>75.0</td><td>75.1</td><td></td><td></td></tr><tr><td>MNIST</td><td>95.5</td><td>94.4</td><td>95.1</td><td></td><td></td></tr><tr><td>MSCOCO</td><td>58.1</td><td>57.8</td><td>56.4</td><td></td><td></td></tr><tr><td>Tr. Signs</td><td>81.7</td><td>82.2</td><td>81.8</td><td></td><td></td></tr></table>

Table 5: How the diverse selection of architectures from Fig. 2b perform per unseen downstream domain in Meta-Dataset. Shading indicates episode-wise architecture selection frequency, numbers indicate accuracy using the corresponding architecture. The best dataset-wise architecture (bold) is most often selected (shading).

Path Search Process In addition, we illustrate the path search process in Figure 3. This figure shows a 2D t-SNE projection of our 2K-dimensional architecture search space, where the dots are candidate architectures of the evolutionary search process at different iterations. The dots are colored according to their validation accuracy. From the results we can see that: The initial set of candidates is broadly dispersed and generally low performing (left), and gradually converge toward a tighter cluster of high performing candidates (right). The top 3 performing paths subject to a diversity constraint (also illustrated in Fig. 2b) are annotated in purple outline.

Discussion As analysed in Section 4.3, our approach can be used in either top-1 – where each episode is a pure fine-tuning operation given the chosen architecture; or top-N architecture mode as discussed above – where each episode performs a mini architecture selection based on the short listed produced during evolutionary search, as well as fine-tuning. We remark that while the latter imposes a slightly increased cost during testing ( $N = 3 \times$ in practice), this is similar or less than competitors who repeat adaptation with different learning rates during testing [21] ( $4 \times$ cost), or exploit a backbone ensemble ( $8 \times$ cost) [13, 32].

# 5. Conclusions

In this paper we present NFTS, a novel neural architecture-search based approach that discovers the optimal adaptation architecture for gradient-based few-shot learning. NFTS contains several recent strong heuristic adaptation architectures as special cases within its search space, and we show that by systematic architecture search they are all outperformed, leading to a new state-of-the-art on Meta-Dataset and Meta-Album. While in this paper we use a simple and coarse search space for easy and direct comparison to prior work's hand-designed adaptation strategies, in future work we will extend this framework to include a richer range of adaptation strategies, and a finer-granularity of search.

# References

[1] Mohamed S Abdelfattah, Abhinav Mehrotra, Łukasz Dudziak, and Nicholas Donald Lane. Zero-cost proxies for lightweight NAS. In International Conference on Learning Representations, 2021. 3   
[2] Peyman Bateni, Raghav Goyal, Vaden Masrani, Frank Wood, and Leonid Sigal. Improved few-shot visual classification. In CVPR, 2020. 1, 3   
[3] Andrew Brock, Theo Lim, James M. Ritchie, and Nick Weston. SMASH: One-shot model architecture search through hypernetworks. In ICLR, 2018. 2   
[4] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. In ICLR, 2020. 2, 3   
[5] Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In ICLR, 2019. 2   
[6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2, 6   
[7] Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for visual recognition. In ICCV, 2021. 2   
[8] Xiangxiang Chu, Bo Zhang, and Ruijun Xu. Fairnas: Rethinking evaluation fairness of weight sharing neural architecture search. In ICCV, 2021. 2   
[9] Yuanzheng Ci, Chen Lin, Ming Sun, Boyu Chen, Hongwen Zhang, and Wanli Ouyang. Evolving search space for neural architecture search. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6659–6669, 2021. 3   
[10] Guneet Singh Dhillon, Pratik Chaudhari, Avinash Ravichandran, and Stefano Soatto. A baseline for few-shot image classification. In International Conference on Learning Representations, 2020. 1, 2   
[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 2, 6   
[12] Nikita Dvornik, Cordelia Schmid, and Julien Mairal. Selecting relevant features from a multi-domain representation for few-shot classification. In ECCV, 2020. 1   
[13] Nikita Dvornik, Cordelia Schmid, and Julien Mairal. Selecting relevant features from a universal representation for few-shot classification. In ECCV, 2020. 7, 8   
[14] Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Neural architecture search: A survey. Journal of Machine Learning Research, 20(55):1–21, 2019. 3   
[15] Jiemin Fang, Yuzhu Sun, Qian Zhang, Yuan Li, Wenyu Liu, and Xinggang Wang. Densely connected search space for more flexible neural architecture search. In CVPR, 2020. 3   
[16] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In ICML, 2017. 1

[17] Chelsea Finn and Sergey Levine. Meta-learning and universality: Deep representations and gradient descent can approximate any learning algorithm. In ICLR, 2018. 3   
[18] Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling. In ECCV, 2020. 2, 3, 4   
[19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 2, 6   
[20] Timothy M Hospedales, Antreas Antoniou, Paul Micaelli, and Amos J. Storkey. Meta-Learning in Neural Networks: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–1, 2021. 3   
[21] Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim, and Timothy M. Hospedales. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In CVPR, 2022. 1, 2, 3, 4, 7, 8   
[22] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In ECCV, 2022. 2   
[23] Brenden M. Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua B. Tenenbaum. One shot learning of simple visual concepts. In CogSci, 2011. 1   
[24] Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. Human-level concept learning through probabilistic program induction. Science, 2015. 1   
[25] Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In CVPR, 2019. 1   
[26] Changlin Li, Jiefeng Peng, Liuchun Yuan, Guangrun Wang, Xiaodan Liang, Liang Lin, and Xiaojun Chang. Blockwisely supervised neural architecture search with knowledge distillation. In CVPR, 2020. 3   
[27] Guohao Li, Guocheng Qian, Itzel C Delgadillo, Matthias Muller, Ali Thabet, and Bernard Ghanem. Sgas: Sequential greedy architecture search. In CVPR, 2020. 2   
[28] Wei-Hong Li, Xialei Liu, and Hakan Bilen. Universal representation learning from multiple domains for few-shot classification. In ICCV, 2021. 1, 2, 6, 7   
[29] Wei-Hong Li, Xialei Liu, and Hakan Bilen. Cross-domain few-shot learning with task-specific adapters. In CVPR, 2022. 1, 2, 3, 4, 6, 7, 8, 11   
[30] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In ACL, 2021. 2, 6   
[31] Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In International Conference on Learning Representations, 2019. 2, 3   
[32] Lu Liu, William L. Hamilton, Guodong Long, Jing Jiang, and Hugo Larochelle. A universal representation transformer layer for few-shot image classification. In ICLR, 2021. 1, 7, 8   
[33] Erik G. Miller, Nicholas E. Matsakis, and Paul A. Viola. Learning from one example through shared densities on transforms. In CVPR, 2000. 1   
[34] Pavlo Molchanov, Jimmy Hall, Hongxu Yin, Jan Kautz, Nicolò Fusi, and Arash Vahdat. LANA: latency aware network acceleration. In ECCV, 2022. 3

[35] Bert Moons, Parham Noorzad, Andrii Skliar, Giovanni Mariani, Dushyant Mehta, Chris Lott, and Tijmen Blankevoort. Distilling optimal neural networks: Rapid search in diverse spaces. In ICCV, 2021. 3   
[36] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, 2018. 2   
[37] Ilija Radosavovic, Justin Johnson, Saining Xie, Wan-Yen Lo, and Piotr Dollár. On network design spaces for visual recognition. In ICCV, 2019. 3   
[38] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In ICLR, 2017. 1   
[39] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. In NeurIPS, 2017. 2, 6   
[40] James Requeima, Jonathan Gordon, John Bronskill, Sebastian Nowozin, and Richard E. Turner. Fast and flexible multi-task classification using conditional neural adaptive processes. In NeurIPS, 2019. 1, 3, 6, 7   
[41] Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. In ICLR, 2019. 1   
[42] Tonmoy Saikia, Thomas Brox, and Cordelia Schmid. Optimized generic feature learning for few-shot classification across domains. arXiv preprint arXiv:2001.07926, 2020. 7   
[43] Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In NeurIPS, 2017. 1, 5, 7   
[44] Yonglong Tian, Yue Wang, Dilip Krishnan, Joshua B Tenenbaum, and Phillip Isola. Rethinking few-shot image classification: a good embedding is all you need? In ECCV, 2020. 1   
[45] Eleni Triantafillou, Hugo Larochelle, Richard S. Zemel, and Vincent Dumoulin. Learning a universal template for few-shot dataset generalization. In ICML, 2021. 2, 4, 7   
[46] Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset of datasets for learning to learn from few examples. In ICLR, 2020. 1, 2, 6, 7   
[47] Ihsan Ullah, Dustin Carrion, Sergio Escalera, Isabelle M Guyon, Mike Huisman, Felix Mohr, Jan N van Rijn, Haozhe Sun, Joaquin Vanschoren, and Phan Anh Vu. Meta-album: Multi-domain meta-dataset for few-shot image classification. In NeurIPS Datasets and Benchmarks Track, 2022. 1, 2, 6, 7, 8   
[48] Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. In NeurIPS, 2016. 1   
[49] Ruochen Wang, Minhao Cheng, Xiangning Chen, Xiaocheng Tang, and Cho-Jui Hsieh. Rethinking architecture selection in differentiable NAS. In ICLR, 2021. 2   
[50] Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, and Laurens van der Maaten. Simpleshot: Revisiting nearest-neighbor classification for few-shot learning, 2019. 1, 7

[51] Yaqing Wang, Quanming Yao, James T. Kwok, and Lionel M. Ni. Generalizing from a few examples: A survey on few-shot learning. ACM Comput. Surv., 2020. 1   
[52] Lichuan Xiang, Lukasz Dudziak, Mohamed S. Abdelfattah, Thomas Chau, Nicholas D. Lane, and Hongkai Wen. Zero-cost operation scoring in differentiable architecture search. In AAAI, 2023. 3   
[53] Chengming Xu, Siqian Yang, Yabiao Wang, Zhanxiong Wang, Yanwei Fu, and Xiangyang Xue. Exploring efficient few-shot adaptation for vision transformers. TMLR, 2022. 1, 2, 3, 4, 6, 7, 8, 11   
[54] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In ECCV, 2014. 2   
[55] Yuanhan Zhang, Kaiyang Zhou, and Ziwei Liu. Neural prompt search. CoRR, 2022. 2   
[56] Daquan Zhou, Xiaojie Jin, Xiaochen Lian, Linjie Yang, Yujing Xue, Qibin Hou, and Jiashi Feng. Autospace: Neural architecture search with less human interference. In ICCV, 2021. 3

<table><tr><td rowspan="2" colspan="2">Hyperparameter</td><td colspan="3">ResNet-18</td><td colspan="2">ViT-S</td></tr><tr><td>SDL (MD)</td><td>MDL (MD)</td><td>MDL (MA)</td><td>SDL (MD)</td><td>MDL (MD)</td></tr><tr><td rowspan="2"></td><td>Backbone architecture</td><td>URL</td><td>URL</td><td>Supervised</td><td>DINO</td><td>DINO</td></tr><tr><td>Adapter architecture</td><td>TSA</td><td>TSA</td><td>TSA</td><td>ETT</td><td>ETT</td></tr><tr><td rowspan="8">TRAIN</td><td>Number of episodes</td><td>50000</td><td>80000</td><td>20000</td><td>80000</td><td>160000</td></tr><tr><td>Number of epochs</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Optimizer</td><td>adadelta</td><td>adadelta</td><td>adadelta</td><td>adamw</td><td>adamw</td></tr><tr><td>Learning rate</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.00007</td><td>0.00007</td></tr><tr><td>Learning rate schedule</td><td>-</td><td>-</td><td>-</td><td>cosine</td><td>cosine</td></tr><tr><td>Learning rate warmup</td><td>-</td><td>-</td><td>-</td><td>linear</td><td>linear</td></tr><tr><td>Weight decay</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.01</td><td>0.01</td></tr><tr><td>Weight decay schedule</td><td>-</td><td>-</td><td>-</td><td>cosine</td><td>cosine</td></tr><tr><td rowspan="10">SEARCH</td><td>Number of episodes</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Number of epochs</td><td>20</td><td>20</td><td>20</td><td>40</td><td>40</td></tr><tr><td>Optimizer</td><td>adadelta</td><td>adadelta</td><td>adadelta</td><td>adamw</td><td>adamw</td></tr><tr><td>Learning rate</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.000003</td><td>0.000003</td></tr><tr><td>Weight decay</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.1</td><td>0.1</td></tr><tr><td>Initial population size</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Top-K crossover</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Mutation chance</td><td>5%</td><td>5%</td><td>5%</td><td>5%</td><td>5%</td></tr><tr><td>Top-N paths</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Diversity threshold</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.2</td><td>0.2</td></tr><tr><td rowspan="6">TEST</td><td>Number of episodes</td><td>600</td><td>600</td><td>1800</td><td>600</td><td>600</td></tr><tr><td>Number of epochs</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td></tr><tr><td>Optimizer</td><td>adadelta</td><td>adadelta</td><td>adadelta</td><td>adamw</td><td>adamw</td></tr><tr><td>Learning rate</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.000003</td><td>0.000003</td></tr><tr><td>Weight decay</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.1</td><td>0,1</td></tr><tr><td>Regulariser strength</td><td>0.04</td><td>0.04</td><td>0.04</td><td>-</td><td>-</td></tr></table>

Table 6: Hyperparameter setting for all experiments presented in Section 4 of the main paper. The notation is as follows: SDL=Single domain learning, MDL=Multi-domain learning, MD=Meta-Dataset, MA=Meta-Album, TRAIN=Supernet training phase, SEARCH=Evolutionary search phase, TEST=Meta-test phase.

# A. Hyperparameter Setting

Table 6 reports the hyperparameters used for all of our experiments. Note the following clarifications:

- “Number of epochs” refers to multiple forward passes of the same episode, while “Number of episodes” refers to the number of episodes sampled in total.   
- The batch size is not mentioned, because we only conduct episodic learning, where we do not split the episode into batches, i.e., we feed the entire support and query set into our neural network architectures.   
- Learning rate warmup, where applicable, occurs for the first $10\%$ of the episodes.

We further specify something important: While our strongest competitors $[29, 53]$ tune their learning rates for meta-testing (e.g., TSA uses LR=0.1 for seen domains and LR=1.0 for unseen, and ETT uses a different learning rate per downstream Meta-Dataset domain), we treat meta-testing episodes as completely unknown, and use the same hyperparameters we used on the validation set during search.

# B. Detailed Ablation Study

Tables 7 and 8 provide the exact scores per Meta-Dataset domain that are summarised in Table 4 of the main paper, for single domain and multi-domain FSL respectively.

<table><tr><td></td><td>Method</td><td>Aircrafts</td><td>Birds</td><td>DTD</td><td>Fungi</td><td>ImageNet</td><td>Omniglot</td><td>QuickDraw</td><td>Flowers</td><td>CIFAR-10</td><td>CIFAR-100</td><td>MNIST</td><td>MSCOCO</td><td>Tr. Signs</td><td>Average</td></tr><tr><td rowspan="6">ResNet-18</td><td> $\phi , -$ </td><td>64.5</td><td>69.6</td><td>71.1</td><td>41.2</td><td>56.4</td><td>74.8</td><td>64.2</td><td>84.6</td><td>75.0</td><td>63.9</td><td>82.1</td><td>55.9</td><td>77.7</td><td>67.8</td></tr><tr><td> $\phi , \alpha$ </td><td>69.6</td><td>67.7</td><td>75.0</td><td>42.5</td><td>59.5</td><td>71.3</td><td>64.9</td><td>88.8</td><td>77.4</td><td>70.0</td><td>90.2</td><td>58.4</td><td>80.1</td><td>70.4</td></tr><tr><td> $\phi', -$ </td><td>69.9</td><td>74.7</td><td>73.3</td><td>39.5</td><td>57.3</td><td>71.9</td><td>65.4</td><td>89.0</td><td>76.5</td><td>66.3</td><td>93.6</td><td>54.4</td><td>81.4</td><td>70.2</td></tr><tr><td> $\phi', \alpha$ </td><td>67.6</td><td>69.1</td><td>77.0</td><td>39.3</td><td>59.7</td><td>77.8</td><td>66.1</td><td>87.4</td><td>81.7</td><td>69.5</td><td>91.9</td><td>55.1</td><td>78.7</td><td>70.8</td></tr><tr><td>NFTS-1</td><td>73.2</td><td>76.5</td><td>81.6</td><td>42.1</td><td>61.3</td><td>80.2</td><td>66.9</td><td>90.0</td><td>82.9</td><td>68.8</td><td>94.0</td><td>58.4</td><td>80.6</td><td>73.6</td></tr><tr><td>NFTS-K</td><td>74.9</td><td>76.5</td><td>81.6</td><td>50.5</td><td>62.7</td><td>80.2</td><td>67.2</td><td>94.5</td><td>83.0</td><td>71.5</td><td>94.0</td><td>59.7</td><td>81.9</td><td>75.2</td></tr><tr><td rowspan="6">ViT-S</td><td> $\phi , -$ </td><td>73.4</td><td>73.6</td><td>81.6</td><td>56.3</td><td>60.3</td><td>69.4</td><td>70.8</td><td>90.4</td><td>70.4</td><td>61.5</td><td>83.8</td><td>60.5</td><td>81.7</td><td>71.8</td></tr><tr><td> $\phi , \alpha$ </td><td>76.9</td><td>83.2</td><td>86.7</td><td>59.3</td><td>63.7</td><td>75.8</td><td>65.1</td><td>89.5</td><td>70.7</td><td>67.4</td><td>81.1</td><td>54.8</td><td>82.9</td><td>73.8</td></tr><tr><td> $\phi', -$ </td><td>76.8</td><td>80.9</td><td>85.8</td><td>61.4</td><td>65.9</td><td>73.2</td><td>68.5</td><td>91.0</td><td>69.9</td><td>66.1</td><td>82.5</td><td>57.6</td><td>78.8</td><td>74.0</td></tr><tr><td> $\phi', \alpha$ </td><td>77.0</td><td>83.4</td><td>82.4</td><td>58.6</td><td>66.7</td><td>73.1</td><td>65.0</td><td>95.9</td><td>76.7</td><td>66.1</td><td>87.7</td><td>58.7</td><td>82.9</td><td>74.4</td></tr><tr><td>NFTS-1</td><td>83.0</td><td>85.5</td><td>87.3</td><td>62.2</td><td>68.8</td><td>81.9</td><td>72.9</td><td>95.3</td><td>79.4</td><td>72.6</td><td>95.2</td><td>62.6</td><td>87.5</td><td>78.7</td></tr><tr><td>NFTS-N</td><td>83.0</td><td>85.5</td><td>87.6</td><td>62.2</td><td>71.0</td><td>81.9</td><td>74.5</td><td>96.0</td><td>79.4</td><td>72.6</td><td>95.2</td><td>62.6</td><td>87.9</td><td>79.2</td></tr></table>

Table 7: Ablation study on Meta-Dataset comparing four special cases of the search space: (i) $\phi$ , -: No adaptation, no fine-tuning, (ii) $\phi$ , $\alpha$ : Adapt all, (iii) $\phi'$ , -: Fine-tune all, (iv) $\phi'$ , $\alpha$ : Adapt and fine-tune all. NFTS- $1,N$ refer to conventional and deferred episode-wise NAS respectively. Single domain setting: Only ImageNet is seen during training and search. Reporting mean accuracy over 600 episodes. 

<table><tr><td></td><td>Method</td><td>Aircrafts</td><td>Birds</td><td>DTD</td><td>Fungi</td><td>ImageNet</td><td>Omniglot</td><td>QuickDraw</td><td>Flowers</td><td>CIFAR-10</td><td>CIFAR-100</td><td>MNIST</td><td>MSCOCO</td><td>Tr. Signs</td><td>Average</td></tr><tr><td rowspan="6">ResNet-18</td><td> $\phi, -$ </td><td>64.5</td><td>69.6</td><td>71.1</td><td>41.2</td><td>56.4</td><td>74.8</td><td>64.2</td><td>84.6</td><td>75.0</td><td>63.9</td><td>82.1</td><td>55.9</td><td>77.7</td><td>67.8</td></tr><tr><td> $\phi, \alpha$ </td><td>89.3</td><td>78.3</td><td>76.1</td><td>62.7</td><td>57.2</td><td>93.8</td><td>76.0</td><td>90.8</td><td>77.8</td><td>66.1</td><td>90.5</td><td>56.9</td><td>79.5</td><td>76.5</td></tr><tr><td> $\phi', -$ </td><td>90.2</td><td>76.7</td><td>70.6</td><td>63.1</td><td>57.8</td><td>88.2</td><td>79.3</td><td>88.9</td><td>78.2</td><td>68.2</td><td>96.1</td><td>51.7</td><td>82.9</td><td>76.3</td></tr><tr><td> $\phi', \alpha$ </td><td>86.1</td><td>78.9</td><td>77.2</td><td>60.5</td><td>57.6</td><td>94.1</td><td>79.5</td><td>86.5</td><td>81.0</td><td>67.2</td><td>96.1</td><td>52.6</td><td>81.8</td><td>76.9</td></tr><tr><td>NFTS-1</td><td>90.1</td><td>82.1</td><td>79.9</td><td>67.9</td><td>61.4</td><td>94.3</td><td>82.6</td><td>92.2</td><td>82.4</td><td>73.8</td><td>95.4</td><td>58.1</td><td>81.0</td><td>80.1</td></tr><tr><td>NFTS-K</td><td>90.1</td><td>83.8</td><td>82.3</td><td>68.4</td><td>61.4</td><td>94.3</td><td>82.6</td><td>92.2</td><td>83.0</td><td>75.1</td><td>95.4</td><td>58.8</td><td>81.9</td><td>80.7</td></tr><tr><td rowspan="6">ViT-S</td><td> $\phi, -$ </td><td>73.4</td><td>73.6</td><td>81.6</td><td>56.3</td><td>60.3</td><td>69.4</td><td>70.8</td><td>90.4</td><td>70.4</td><td>61.5</td><td>83.8</td><td>60.5</td><td>81.7</td><td>71.8</td></tr><tr><td> $\phi, \alpha$ </td><td>85.7</td><td>84.3</td><td>81.8</td><td>68.7</td><td>70.4</td><td>89.1</td><td>77.0</td><td>90.2</td><td>73.5</td><td>61.4</td><td>82.6</td><td>53.7</td><td>72.4</td><td>77.3</td></tr><tr><td> $\phi', -$ </td><td>83.0</td><td>84.5</td><td>81.1</td><td>70.9</td><td>72.4</td><td>88.6</td><td>74.6</td><td>90.4</td><td>75.1</td><td>63.5</td><td>87.0</td><td>54.0</td><td>75.5</td><td>77.5</td></tr><tr><td> $\phi', \alpha$ </td><td>82.5</td><td>85.9</td><td>82.7</td><td>68.9</td><td>73.7</td><td>90.4</td><td>77.1</td><td>94.0</td><td>73.4</td><td>66.2</td><td>85.9</td><td>55.9</td><td>77.4</td><td>78.9</td></tr><tr><td>NFTS-1</td><td>89.1</td><td>90.3</td><td>86.3</td><td>75.1</td><td>74.6</td><td>92.0</td><td>80.6</td><td>93.5</td><td>75.9</td><td>70.8</td><td>91.3</td><td>62.7</td><td>87.2</td><td>83.1</td></tr><tr><td>NFTS-N</td><td>89.1</td><td>92.5</td><td>86.3</td><td>75.1</td><td>74.6</td><td>92.0</td><td>80.6</td><td>93.5</td><td>75.9</td><td>70.8</td><td>91.3</td><td>62.8</td><td>87.2</td><td>83.4</td></tr></table>

Table 8: Ablation study on Meta-Dataset comparing four special cases of the search space: (i) $\phi$ , -: No adaptation, no fine-tuning, (ii) $\phi$ , $\alpha$ : Adapt all, (iii) $\phi'$ , -: Fine-tune all, (iv) $\phi'$ , $\alpha$ : Adapt and fine-tune all. NFTS- $1,N$ refer to conventional and deferred episode-wise NAS respectively. Multi-domain setting: The first 8 datasets are seen during training and search. Reporting mean accuracy over 600 episodes.

# C. Source code

The source code is available at: https://github.com/peustr/nfts-public.