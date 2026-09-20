# TOWARDS META-PRUNING VIA OPTIMAL TRANSPORT

Alexander Theus & Olin Geimer

{atheus, geimero}@ethz.ch

Department of Computer Science

ETH Zurich, Switzerland

Friedrich Wicke, Thomas Hofmann $^{\dagger}$ , Sotiris Anagnostidis $^{\dagger}$ & Sidak Pal Singh $^{\dagger}$

{friedrich.wicke, thomas.hofmann,

sotirios.anagnostidis, sidak.singh}@inf.ethz.ch

Department of Computer Science

ETH Zurich, Switzerland

# ABSTRACT

Structural pruning of neural networks conventionally relies on identifying and discarding less important neurons, a practice often resulting in significant accuracy loss that necessitates subsequent fine-tuning efforts. This paper introduces a novel approach named Intra-Fusion, challenging this prevailing pruning paradigm. Unlike existing methods that focus on designing meaningful neuron importance metrics, Intra-Fusion redefines the overlying pruning procedure. Through utilizing the concepts of model fusion and Optimal Transport, we leverage an agnostically given importance metric to arrive at a more effective sparse model representation. Notably, our approach achieves substantial accuracy recovery without the need for resource-intensive fine-tuning, making it an efficient and promising tool for neural network compression. Additionally, we explore how fusion can be added to the pruning process to significantly decrease the training time while maintaining competitive performance. We benchmark our results for various networks on commonly used datasets such as CIFAR-10, CIFAR-100, and ImageNet. More broadly, we hope that the proposed Intra-Fusion approach invigorates exploration into a fresh alternative to the predominant compression approaches. Our code is available here $^{1}$ .

# 1 INTRODUCTION

Alongside the massive progress in the past few years, modern over-parameterized neural networks have also brought another thing onto the table. That is, of course, their massive size. Consequently, as part of the community keeps training bigger networks, another community has been working, often in the background, to ensure that these bulky networks can be made compact to actually be deployed (Hassibi et al., 1993). Techniques to reduce the size of these networks and speed-up inference come in various forms, such as pruning — which can itself be unstructured (Han et al., 2015), semi-structured (Zhou et al., 2021a), or structured (Wang et al., 2019; Frantar & Alistarh, 2023); quantization (Dettmers et al., 2022; Yao et al., 2022; Xiao et al., 2022); knowledge distillation (Hinton et al., 2015; Gou et al., 2021); low-rank decomposition (Yu et al., 2017); hardware co-design (Zhu et al., 2019), to list a few.

However, despite the apparent conceptual simplicity of these techniques, compressing neural networks, in practice, is not as straightforward as simply doing one or two traditional post-processing steps (Blalock et al., 2020). The process involves a crucial element—fine-tuning or retraining, on the original dataset or a subset—extending over several additional epochs.

While such additional fine-tuning may not seem too much of an issue for some networks, for others like large language models even a single epoch might be excessively expensive. Hence, this makes the question of investigating the direction of ‘fine-tuning-free’ compression methods or even ‘data-free’ compression methods all the more pertinent.

Besides, since it is almost a given that the training pipeline for taking any interesting network from scratch to deployment will include some form of compression, an overlooked aspect is whether any improvements can be introduced in this joint space of training and pruning in the conventional strategy. For instance, a fine example is the Lottery Ticket Hypothesis (Frankle & Carbin, 2018), which suggests the presence of sparse sub-networks that can be trained from the outset while obviating the need for subsequent pruning. Presently, however, this is more of an existence result, since the sub-networks are obtained retrospectively, i.e., having trained dense networks from scratch. Another prominent example is that of Federated learning (McMahan et al., 2017; Zhang et al., 2021), which has made the community rethink the process of training large models in a distributed manner by carrying local model updates and then aggregating these model parameters directly.

In fact, stemming from the practical interest in federated learning, but also theoretical questions of mode connectivity (Garipov et al., 2018), another novel line of work has lately explored the possibility of fusing (the parameters of) independently trained networks (potentially of different sizes). A notable work in this direction, from which we are heavily inspired, is that of OTFusion (Singh & Jaggi, 2020). More specifically, amongst other things, the authors also demonstrate the idea of fusing a network with a lesser version of itself, say a pruned version, in a bid to help recover the performance drop in the past. However, as their demonstration was merely a proof-of-concept and inherently limited in scope, this exciting idea of self-recovery has remained in a nascent stage.

Our focus in this paper is, therefore, to unite these two lines of work, namely pruning and fusion, in a more cohesive manner. By unifying these two concepts, we sim to expand the horizon of the conventional pruning paradigm — across the trifold axes of:

(i) Intra-Fusion: While most research on pruning has focused on devising more meaningful neuron importance metrics, the overlying procedure has remained largely the same: Keep the most important neurons, discard the rest. In contrast, Intra-Fusion leverages Optimal Transport to additionally inform the process of model compression with the neurons that otherwise would have been discarded (see Section 3).

(ii) Data-Free pruning: Pruning neural networks generally leads to immediate drops in accuracy, hence requiring an extensive fine-tuning step to be usable in a practical setting. Accuracy drops between simpler and more sophisticated neuron importance metrics do not differ substantially. We argue that this is largely an artifact of the underlying pruning procedure and show that, by using Intra-Fusion, a significant amount of accuracy can be recovered without the need for any datapoints.

(iii) Split-Data Training: Models that are to be pruned after training are most often fine-tuned for many epochs to regain sufficient performance. Via the presented ‘PaF’ and ‘FaP’ approaches, we factorize this process through the combination of model pruning and model fusion. By splitting the training dataset into multiple parts, over which models are trained concurrently, we achieve significant training time speedups.

# 2 BACKGROUND

# 2.1 PRUNING

Pruning techniques (LeCun et al., 1989) can broadly be classified into structured (Wang et al., 2019; Fang et al., 2023), and unstructured pruning (Han et al., 2015; Singh & Alistarh, 2020). Unstructured pruning involves zeroing out individual weights and leaves the network structure unaltered. Our work exclusively deals with structured pruning, which aims to remove entire sets of parameters or neurons and thereby directly alters the network structure. The reason being that (a) the structured pruning procedure directly translates into a speedup in the inference time — unlike in unstructured pruning, which requires the aid of specialized hardware accelerators to extract some (typically reduced) levels of speedup; (b) structured pruning eases the storage and memory footprint of the network; while unstructured pruning methods yield no such gains (but, in fact, also necessitate the maintenance of binary masks during the course of training).

Structured pruning. A very common and successful approach to prune networks structurally is to capture a neuron's importance by its $\ell_{p}$ -norm, where p is the order of the norm. Despite its simplicity, $\ell_{p}$ -norm pruning can achieve state-of-the-art performance by cleverly incorporating the dependencies within the network (Fang et al., 2023). However, other works such as (He et al., 2019) have shown that the “smaller-norm-less-important assumption” does not always hold. Instead of removing neurons with a small norm, redundant filters can be found by exploiting relationships between neurons of the same layer. Namely, they consider neurons close to the geometric mean to be redundant as they represent information abundant in the layer. Instead of evaluating importance based on the weight itself, other methods focus on the activations of the neurons. The importance of a neuron is thus measured by evaluating the reconstruction error of the current layer (He et al., 2017) or of the final response layer (Yu et al., 2018).

Evidently, research on structured pruning has largely focused on devising more meaningful importance measures, while the overlying procedure has remained the same, repeating the mantra: Keep the most important neurons, discard the rest. This work challenges the above mantra by recycling or restoring information from all neurons to create more accurate compressed networks.

# 2.2 OPTIMAL TRANSPORT & MODEL FUSION

Optimal Transport (OT). OT (Villani et al., 2009) is a mathematical framework that provides a rigorous and geometrically interpretable way to compare probability distributions. The OT problem aims to find the most economical way to “transport” mass from one distribution defined over a space X to another supported over the space Y, where the cost is determined by a function c. It achieves this by lifting the metric in the ground space (i.e., X,Y) to obtain a metric in the space of distributions. In the case of discrete probability measures, OT reduces to the well-known transportation problem in linear programming, which has the following form:

$$
\operatorname{OT} (\mu , \nu ; C) := \min \langle T, C \rangle_ {F} \quad \text { s.t., } \quad T \mathbf {1} _ {m} = \alpha ,   T ^ {\top} \mathbf {1} _ {n} = \beta \quad \text { and } \quad T \in \mathbb {R} _ {+} ^ {(n \times m)}  .
$$

Here, $\mu := \sum_{i=1}^{n} \alpha_i \cdot \delta(x_i)$ and $\nu := \sum_{j=1}^{m} b_j \cdot \delta(y_j)$ , with $\sum_{i=1}^{n} \alpha_i = \sum_{j=1}^{m} \beta_j = 1$ describe two probability measures, where $\delta(\cdot)$ denotes the dirac delta function. Further, T denotes the transport map whose row and columns should sum to the marginals $\alpha$ and $\beta$ . Besides, C denotes the ground cost, of moving a unit mass from point $x_i$ to $y_j$ , and for instance, in the Euclidean case, it can be $C(x_i, y_j) = \|x_i - y_j\|^2$ . Optimal Transport has found applications in many fields, especially in machine learning (Kusner et al., 2015; Frogner et al., 2015; Arjovsky et al., 2017; Zhou et al., 2021b) and we will consider this in regards to fusion.

Model Fusion. The key idea behind model fusion is to combine the capabilities of two parent networks into a single-child network. Singh & Jaggi (2020) introduces Optimal Transport (OT) for model fusion, which is the fusion technique we are using throughout this paper. We will refer to this method as OTFusion. The main idea behind this work is to combine multiple independently trained neural networks, after accounting for the permutation symmetries that exist within the layers. Finding the permutation symmetries is then framed as an Optimal Transport problem between the neurons of the given networks. An important thing to note is that Singh & Jaggi (2020) primarily use uniform distributions in place of $\alpha$ and $\beta$ ; however as we will see later, we further exploit this in our proposed method.

Besides OTFusion, other works are also inherently based on a similar formulation (Li et al., 2015; Yurochkin et al., 2019; Wang et al., 2020), though not always geared towards the same end. Our focus will nevertheless be on OTFusion, as we directly build atop their exploration of pruning and fusion.

# 3 METHODOLOGY

Intra-Fusion, as a “meta-pruning” approach, is an attempt to take a step back from the search for meaningful importance metrics, and reconsider the way these found importance metrics are leveraged to come up with a sparse model representation. Instead of simply discarding the least important neurons (as done in the “conventional pruning” approach, see Algorithm 1), Intra-Fusion leverages a modified version of OTFusion to incorporate the discarded neurons into the “surviving” ones (see Algorithm 2). We refer to our algorithm as a new ”meta” approach to pruning, since it challenges

the overlying framework that defines how an agnostic importance metric is integrated. It does not compete with any importance metrics. Instead it provides an alternative methodology on how these metrics are used to compress a network.

This section is dedicated to the inner workings of Intra-Fusion. Since Intra-Fusion is an alternative to the conventional pruning procedure, we show both meta-pruning approaches side by side (see Algorithm 1 and 2) to highlight the differences between the two. Section 3.1 serves as a high-level overview of the two algorithms, whereas the subsequent sections are more detailed explanations of the individual parts that make up Intra-Fusion, which are also referenced in Algorithm 2.

# 3.1 META PRUNING: AN OVERVIEW

Data: IMPORTANCE $i_{1 \times n}$ , group $\mathbb{G}$ with group cardinality $n$ , and target group cardinality $m$ .

Result: group $G_{new}$ with group cardinality m

<table><tr><td>Algorithm 1: Conventional Pruning</td><td colspan="2">Algorithm 2: Intra-Fusion</td></tr><tr><td> $t \leftarrow m^{\text{th}}$  highest scalar in  $i$ ; $\mathbb{G}_{new} \leftarrow \emptyset$ ;for layer  $\in \mathbb{G}$  do|layer $_{new} \leftarrow$ layer without neurons  $j$ where  $i[j] < t$ ; $\mathbb{G}_{new} = \mathbb{G}_{new} \cup \{layer_{new}\}$ ;end</td><td> $\mathbb{Y} \leftarrow \text{GETTARGET}(\mathbb{G})$ ; $\mathbb{X} \leftarrow \mathbb{G}$ ; $\boldsymbol{C}_{n \times m} \leftarrow \text{COMPUTECOST}(\mathbb{X}, \mathbb{Y})$ ; $\boldsymbol{\nu}_{1 \times m} \leftarrow \text{GETTARGETDISTR}(m, i)$ ; $\boldsymbol{\mu}_{1 \times n} \leftarrow \text{GETSOURCEDISTR}(n, i)$ ; $\boldsymbol{T}_{n \times m} \leftarrow \text{OT}(\boldsymbol{\mu}, \boldsymbol{\nu}, \boldsymbol{C})$ ; $\boldsymbol{T}_{m \times n} \leftarrow \text{diag}(\frac{1}{\boldsymbol{\mu}}) \times \boldsymbol{T}^{\top}$ ; $\mathbb{G}_{new} \leftarrow \emptyset$ ;for layer  $\in \mathbb{G}$  do|  $\mathbb{G}_{new} = \mathbb{G}_{new} \cup \{T \times layer\}$ ; end</td><td>3.2.13.2.13.2.23.2.33.2.33.2.43.2.43.2.4</td></tr></table>

Structured Pruning: Group-by-Group. As described in Section 2.1, the increasing complexity of neural networks pose a challenge for structured pruning. Removing neurons from layers individually can lead to broken networks, as neurons in a layer might not only be dependent on the previous layer, but also on those that lie even further back.

![](images/50248ce826dbd7c6d8775bcbb4036c3a2dfc4285403a899aa878fb28d46c5e17.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["1."]
    A --> C["2."]
    A --> D["3."]
    B --> E["+"]
    C --> F["+"]
    D --> G["+"]
    style A fill:#fff,stroke:#000
    style B fill:#fff,stroke:#000
    style C fill:#fff,stroke:#000
    style D fill:#fff,stroke:#000
    style E fill:#fff,stroke:#000
    style F fill:#fff,stroke:#000
    style G fill:#fff,stroke:#000
```
</details>

(a) The beginning of an example network

![](images/54c02d7a02616fb058ca5ee2b641116017006b84a5bed75849731e5d90206ccf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Layer 1"]
    A --> C["Layer 2"]
    A --> D["Layer 3"]
    B --> E["Output +"]
    C --> F["Output -"]
    D --> G["Feedback Loop"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
```
</details>

(b) Broken network: individually pruned layers

![](images/fb304905463e8a8610135e90c61e9ca5e1783824cb57f1945bd0b31f3d4a54ec.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["1."]
    A --> C["2."]
    A --> D["3."]
    B --> E["+"]
    C --> F["+"]
    D --> G["+"]
    E --> H["Output"]
    F --> H
    G --> H
    H --> I["Output"]
    style A fill:#fff,stroke:#000
    style B fill:#fff,stroke:#000
    style C fill:#fff,stroke:#000
    style D fill:#fff,stroke:#000
    style E fill:#fff,stroke:#000
    style F fill:#fff,stroke:#000
    style G fill:#fff,stroke:#000
    style H fill:#fff,stroke:#000
    style I fill:#fff,stroke:#000
```
</details>

(c) The network as a set of neuron pairings

![](images/226ab43ef2958da920a9f5c52af1f4016fa54d6d7e6e762da2e8f7d5c5752638.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["1."]
    A --> C["2."]
    A --> D["3."]
    B --> E["+"]
    C --> F["+"]
    D --> G["+"]
    E --> H["Output"]
    F --> I["Output"]
    G --> J["Output"]
```
</details>

(d) Structurally pruning complete group   
Figure 1: Structural pruning by considering groups

In Figure 1, we show an example of this: Assume we are given the beginning of a network in Figure 1a, where layer three represents the arrival of a residual connection. When iterating through the network layer-by-layer, while pruning, the network can easily be broken (see Figure 1b). As multiple layers can be arranged in such a dependency, pruning of multiple layers has to be done jointly (in our example layers one to three). We call a set of layers that have to be pruned in unison a group G. In a group, not all neurons are dependent on one another. We can identify “pairings of neurons” that have to be handled jointly (see the different colored pairings in Figure 1c). The number of neuron pairings in a group we term “group cardinality” (in our example this would be three).

Structurally pruning the network in Figure 1a whilst considering the dependencies could result in the network shown in Figure 1d.

Meta-Pruning Comparison. The starting point for conventional pruning and for our Intra-Fusion methodology is the same. We are given a group G with initial group cardinality n, that we wish to prune to a target group cardinality $m \leq n$ . Moreover, we are given an importance vector $i_{1 \times n}$ that assigns an agnostic importance score (e.g. $\ell_{1}$ -norm) to each independent pairing in the group. In conventional pruning (see Algorithm 1), we keep the m most important neuron pairings according to i, and remove the rest to arrive at our new group $G_{new}$ with group cardinality m.

Most research on structured pruning has focused on devising more meaningful importance measures, i.e. i, whereas the overlying procedure detailed in Algorithm 1 has remained practically unaltered. Inspired by OTFusion, we attempt to develop an alternative to the way pruning is conventionally done. Instead of simply discarding the less important pairings in a group (in Figure 1c this would be the blue pairing), we leverage the computed importance metrics to inform the process of fusing these pairings to end up at a lower group cardinality.

To end up at a lower group cardinality, we fuse the existing n-many pairings to end up at m-many pairings. The matching of the pairings to be fused is found via Optimal Transport (OT), which requires the careful setting of multiple non-trivial hyper-parameter choices. In particular, the discrete “source distribution” of our OT problem is representative of the original network’s group G (group cardinality n), i.e., it is supported on the space of neuron pairings.

To complete the OT problem formulation we additionally need to determine a target of group cardinality m, and a neural similarity measure to quantify the transportation cost. Lastly, we need to ascertain the probabilistic measures encapsulating the mass distribution of both the source and target entities. Solving the just formulated OT problem gives a transportation map T.

# 3.2 COMPONENTS OF INTRA-FUSION

3.2.1 TARGET AND SOURCE SELECTION. Our experiments reveal that a simple, but fruitful option is to use $G_{new}$ derived by Algorithm 1, i.e. the neurons with the highest importance according to some agnostically defined metric. Another promising approach is to cluster the neurons with K-means (Lloyd, 1982), or Gaussian Mixture Models, with m clusters, and use the respective cluster centroids as the target.

3.2.2 TRANSPORTATION COST. Once we have determined the source and the target, the next step is to quantify the cost of moving a unit mass from a source neuron to a target neuron. For this we will take a metric that measures how similar or dissimilar neurons are. In case a group contains multiple layers (as shown in Figure 1), we have to find a joint similarity measure for pairings of neurons.

Given two vectors a and b, each representative of the weights of a different neural pairing, we determine similarity by their normalized $\ell_{1}$ -distance. The weights for neural pairings can be derived via concatenating the weights of the respective neurons and bias terms.

However, architectural components such as Batch Normalization (BN) (Ioffe & Szegedy, 2015) evidently presents themselves as a challenge due to their unique connection with their prior layer. We overcome this challenge by taking advantage of the properties of batch normalization, and simply merge the batchnorm layer into the prior layer whose outputs it acts upon (Jacob et al., 2018),

$$
\boldsymbol {w} _ {\text { new }} = \frac {\boldsymbol {w} \times \gamma}{\sqrt {\sigma}}, \quad b _ {\text { new }} = \frac {(b - \mu) \times \gamma}{\sqrt {\sigma}} + \beta . \tag {1}
$$

In this context, w denotes the weight vector corresponding to the preceding layer. Furthermore, we denote b as its associated bias term. The symbols $\mu$ and $\sigma$ represent the running mean and variance, respectively, of the batch normalization layer, while $\gamma$ and $\beta$ symbolize the learnable parameters.

3.2.3 PROBABILITY DISTRIBUTION. We propose two different ways of quantifying probability mass: uniform or importance-informed. A uniform distribution prescribes equal mass to each neuron pairing. In the importance-informed option, the mass is relative to the importance of the neuron pairing. In order to transform the importance of a neuron pairing to a probability, one can either divide the importance by the sum of all importances, or use softmax. In Appendix C.3 we show that there are no significant differences in accuracy between the choice of source and target distribution.

![](images/4770b088f035787ea2e887a030bc0f0306725a6e039574e5dc655a81df89e267.jpg)  
Figure 2: Options for target/source probability mass distribution with $\ell_{1}$ -norm as importance.

3.2.4 DERIVING FUSED NEURONS. Given the cost matrix C, and our probability distributions $\mu$ , and $\nu$ , we can finally derive the optimal transport map T. Since the columns of this transport map act as the coefficients for the weighted aggregation of corresponding neuron pairings, it is imperative to ensure that these coefficients collectively sum to unity.

Moreover, as a final preparatory step before amalgamating matching neuron pairings, it is necessary to conjoin the batch normalization layer with the layer upon which it operates. This process is elucidated in Equation 1. Subsequently, we establish the values of $\mu$ and $\beta$ as zero, and $\sigma$ and $\gamma$ as one, thereby preserving the unaltered state of the batch normalization layer's activation.

Consequently, we proceed to traverse each layer within the group denoted as G, and rather than removing neurons of diminished significance, we opt to generate fused neurons through a process of matrix multiplication with the transport map T. This methodology culminates in pruned layers that emerge as a product of a nuanced aggregation, where the shared features of these neurons are subject to a weighted summation.

# 4 EMPIRICAL RESULTS

Here, we seek to illustrate the accuracy gains Intra-Fusion can achieve. Most pruning literature ignores pre-finetuning accuracy, largely due to the significant accuracy drops imposed. In Section 4.1, we show that this drop is mainly an artefact of the overlying pruning methodology. Namely, by utilizing importance metrics in a more involved way as done in Intra-Fusion, a significant amount of accuracy can be maintained. Lastly, for the sake of completeness, we show how Intra-Fusion stands up in face of fine-tuning.

Terminology. Since we cut whole neurons and not just individual edges we will distinguish between “neuron sparsity” and “weight sparsity”. We refer to “neuron sparsity” as the number of neuron pairings that are cut out of a group. Accordingly, we will refer to the number of edges removed from the neural network as “weight sparsity”. Where unspecified, we are referring to neuron sparsity when talking about “sparsity”. See Appendix E.1 for a comprehensive comparison.

Lastly, in this Section we use the term “Group”. This refers to G as described in Section 3. Moreover, the group indices are ordered such that small indices are closer to the output of the model, e.g. Group 4 is closer to the end of the model than Group 5.

# 4.1 DATA-FREE: PRUNING WITHOUT FINE-TUNING

In order to compare the data-free performance of the conventional meta-pruning paradigm (see Algorithm 1), and Intra-Fusion (see Algorithm 2), we compare the test accuracy of a VGG11-BN, ResNet18, ResNet50, on CIFAR-10, CIFAR-100, and ImageNet. Furthermore, the pruning is done on the basis of a group, for importance metrics $\ell_{1}$ , and more sophisticated ones such as Taylor (Molchanov et al., 2019), LAMP (Lee et al., 2021) and CHIP (Sui et al., 2021). Given the nature of Taylor importance, additional information about parameter gradients is needed when deploying it as a metric. Due to the extensive set of experiments, we are forced to show only a selection in this section and refer to Appendix E.2 for a more comprehensive list of results.

A snapshot of the results can be seen in Figure 3 and 4. While diverse importance metrics do not seem to affect pre-finetuning accuracy meaningfully, Intra-Fusion can leverage an importance

![](images/0210a9973762915a1423298d1c802e35b11a762401392e92fe0ed73736670010.jpg)

<details>
<summary>line</summary>

| Neuron Sparsity | Group 6 (Intra-Fusion) | Group 6 (Default) | Group 16 (Intra-Fusion) | Group 16 (Default) | Group 19 (Intra-Fusion) | Group 19 (Default) |
| --------------- | ---------------------- | ----------------- | ----------------------- | ------------------ | ----------------------- | ------------------ |
| 10%             | 75.0%                  | 75.0%             | 75.0%                   | 75.0%              | 75.0%                   | 75.0%              |
| 20%             | 74.5%                  | 74.0%             | 74.8%                   | 73.5%              | 74.2%                   | 73.0%              |
| 30%             | 74.0%                  | 73.5%             | 74.5%                   | 72.5%              | 73.8%                   | 72.0%              |
| 40%             | 73.5%                  | 73.0%             | 74.0%                   | 71.5%              | 73.0%                   | 71.0%              |
| 50%             | 72.0%                  | 72.5%             | 73.5%                   | 70.5%              | 71.5%                   | 69.0%              |
| 60%             | 68.0%                  | 69.0%             | 73.0%                   | 69.0%              | 68.0%                   | 66.0%              |
| 70%             | 62.0%                  | 63.0%             | 70.5%                   | 68.0%              | 58.0%                   | 56.0%              |
</details>

![](images/c14521fae719a233bef398072b83969a2ab1c58a3733f1d342ba6c12c0581e77.jpg)

<details>
<summary>line</summary>

| Neuron Sparsity | Group 6 (Intra-Fusion) | Group 6 (Default) | Group 16 (Intra-Fusion) | Group 16 (Default) | Group 19 (Intra-Fusion) | Group 19 (Default) |
| --------------- | ---------------------- | ----------------- | ----------------------- | ------------------ | ----------------------- | ------------------ |
| 10%             | 75.0%                  | 74.0%             | 75.0%                   | 74.0%              | 75.0%                   | 74.0%              |
| 20%             | 74.5%                  | 73.0%             | 74.5%                   | 73.0%              | 74.5%                   | 73.0%              |
| 30%             | 74.0%                  | 72.0%             | 74.0%                   | 72.0%              | 74.0%                   | 72.0%              |
| 40%             | 73.5%                  | 71.0%             | 73.5%                   | 71.0%              | 73.5%                   | 71.0%              |
| 50%             | 73.0%                  | 70.0%             | 73.0%                   | 70.0%              | 73.0%                   | 70.0%              |
| 60%             | 72.5%                  | 69.0%             | 72.5%                   | 69.0%              | 72.5%                   | 69.0%              |
| 70%             | 72.0%                  | 68.0%             | 72.0%                   | 68.0%              | 72.0%                   | 68.0%              |
</details>

Figure 3: Data-Free Pruning: ResNet50 on ImageNet. $\ell_{1}$ (left), Taylor (right).   
![](images/1483dffa777a7a0f225c5092e4e99f705d5ce11665588359b58c5a8272bfd1e7.jpg)

<details>
<summary>line</summary>

| Neuron Sparsity | Group 3 (Intra-Fusion) | Group 3 (Default) | Group 5 (Intra-Fusion) | Group 5 (Default) | Group 6 (Intra-Fusion) | Group 6 (Default) |
| --------------- | ---------------------- | ----------------- | ---------------------- | ----------------- | ---------------------- | ----------------- |
| 10%             | 92.0%                  | 92.0%             | 92.0%                  | 92.0%             | 92.0%                  | 92.0%             |
| 20%             | 91.0%                  | 91.0%             | 91.0%                  | 91.0%             | 91.0%                  | 91.0%             |
| 30%             | 90.0%                  | 90.0%             | 90.0%                  | 90.0%             | 90.0%                  | 90.0%             |
| 40%             | 88.0%                  | 88.0%             | 88.0%                  | 88.0%             | 88.0%                  | 88.0%             |
| 50%             | 85.0%                  | 85.0%             | 85.0%                  | 85.0%             | 85.0%                  | 85.0%             |
| 60%             | 82.0%                  | 82.0%             | 82.0%                  | 82.0%             | 82.0%                  | 82.0%             |
| 70%             | 60.0%                  | 60.0%             | 60.0%                  | 60.0%             | 30.0%                  | 30.0%             |
</details>

![](images/58191bafe94c1023281cb3f9f28b314647d42743688fdb5262f6ee5b9f665c9f.jpg)

<details>
<summary>line</summary>

| Neuron Sparsity | Group 0 (Intra-Fusion) | Group 0 (Default) | Group 1 (Intra-Fusion) | Group 1 (Default) | Group 3 (Intra-Fusion) | Group 3 (Default) |
| --------------- | ---------------------- | ----------------- | ---------------------- | ----------------- | ---------------------- | ----------------- |
| 10%             | 68.0%                  | 67.0%             | 67.5%                  | 66.5%             | 67.5%                  | 66.0%             |
| 20%             | 67.5%                  | 66.5%             | 67.0%                  | 66.0%             | 67.0%                  | 65.5%             |
| 30%             | 67.0%                  | 66.0%             | 66.5%                  | 65.5%             | 66.5%                  | 65.0%             |
| 40%             | 66.5%                  | 65.5%             | 66.0%                  | 65.0%             | 66.0%                  | 64.5%             |
| 50%             | 66.0%                  | 65.0%             | 65.5%                  | 64.5%             | 65.5%                  | 64.0%             |
| 60%             | 65.5%                  | 64.5%             | 65.0%                  | 64.0%             | 65.0%                  | 63.5%             |
| 70%             | 65.0%                  | 64.0%             | 64.5%                  | 63.5%             | 64.5%                  | 63.0%             |
</details>

Figure 4: Data-Free Pruning: ResNet18 on CIFAR-10 and VGG11-BN on CIFAR-100, $\ell_{1}$ .

metric to substantially increase accuracy (by up to +60% in certain cases) without any additional use of data. To further highlight that the improvements of Intra-Fusion are agnostic to the choice of importance metric, we include results assigning random scores drawn from a uniform distribution as an importance metric (see Appendix C.4).

Across different network architectures, there appear to be, in general, two different kinds of groups: volatile and resilient. Volatile groups exhibit strong accuracy losses as the sparsity increases (e.g., see "Group 6" in Figure 4), whereas resilient groups only experience small ones (e.g., see "Group 16" in Figure 3). This pattern seems to be agnostic with respect to the importance metric. Nevertheless, Intra-Fusion manages to increase accuracy substantially for both types.

Take for instance Group 6 from Figure 4, a volatile group. For a neuron sparsity of 40%, the conventionally pruned model has dropped to an accuracy of only 80.2%, whereas the Intra-Fused model is still at a competitive 92.1%. However, even for resilient groups, where the margin of improvement is low, we make improvements (see Group 16 in Fig. 3). These results represent a general trend (see Appendix E.2 for all results). Overall, this shows the benefit of our proposed approach, which can alleviate the performance drop without relying on fine-tuning, or for that matter on any datapoints at all.

# 4.2 DATA-DRIVEN: PRUNING WITH FINE-TUNING

Although fine-tuning may not always be the most convenient in all scenarios, it might be possible in others. In any case, it would be interesting to see whether the performance gains delivered by Intra-Fusion standup in the face of fine-tuning or not. Hence, we carry out a similar experiment as before for both VGG11-BN and ResNet18 trained on CIFAR-10; however, this time, fine-tuning after model compression is available.

Table 1 contains our results (averaged over multiple runs) for this setting. We observe that Intra-Fusion obtains a consistent gain of up to 1% test accuracy (with standard deviation of 0.13% for all sparsities), across all the considered sparsity levels. While the gains might not seem as stark, we must remark that here we allowed for a significantly long fine-tuning schedule, and that Intra-Fusion converges faster due to the large initial accuracy gains. It is also important to note that the focus of this paper is on data-free pruning.

Table 1: Intra-Fusion vs. default pruning with fine-tuning for $\ell_{1}$ importance on CIFAR-10. 

<table><tr><td>Model</td><td>Base (%)</td><td>Sparsity (%)</td><td>Default (%)</td><td>Intra-Fusion (%)</td><td>Gain. (%)</td></tr><tr><td rowspan="4">VGG11-BN</td><td rowspan="4">89.56%</td><td>64</td><td>88.95</td><td>89.19</td><td>+0.24</td></tr><tr><td>84</td><td>88.14</td><td>88.51</td><td>+0.36</td></tr><tr><td>91</td><td>87.43</td><td>88.15</td><td>+0.72</td></tr><tr><td>96</td><td>85.21</td><td>85.89</td><td>+0.68</td></tr><tr><td rowspan="4">ResNet18</td><td rowspan="4">92.17%</td><td>64</td><td>91.53</td><td>91.98</td><td>+0.45</td></tr><tr><td>84</td><td>91.22</td><td>91.62</td><td>+0.40</td></tr><tr><td>91</td><td>90.41</td><td>91.37</td><td>+0.96</td></tr><tr><td>96</td><td>88.82</td><td>89.60</td><td>+0.79</td></tr></table>

To conclude, our consistent gains show that the boost afforded by Intra-Fusion is complementary to that provided via just fine-tuning the pruned model — thereby demonstrating the efficacy of our approach.

# 5 UNDERSTANDING INTRA-FUSION

So far, we have elucidated and juxtaposed Intra-Fusion with the conventional pruning methodology, demonstrating that Intra-Fusion possesses the distinctive capability to enhance accuracy significantly without relying on any data. In this section, we would like to obtain a better understanding of the inner workings of Intra-Fusion. For a more comprehensive analysis, please see Appendix C.

# 5.1 OUTPUT PRESERVATION

A very intuitive explanation for the superior performance of Intra-Fusion is its ability to better preserve the output of the original non-pruned model. Hence, we quantify output divergence by the $\ell_{2}$ -distance to the output of the original model for various groups in the data-free scenario at different sparsities. As can be seen in Figure 5 (and in more detail in Appendix C.1), Intra-Fusion is indeed able to preserve the output better, with the margin growing as the sparsity increases. Thus, it seems that merging akin neurons as we do with Intra-Fusion leads to better output preservation, and subsequent superior performance.

![](images/b5e6c7840469af732901eb9152aea75d290525f6c42c9f9f1727bfb7fc19bbda.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Intra-Fusion | Default | Mean (Default): 2.0 |
|---|---|---|---|
| 0-1 | 3300 | 0 | 0 |
| 1-2 | 3600 | 0 | 0 |
| 2-3 | 1400 | 800 | 2.0 |
| 3-4 | 900 | 500 | 2.0 |
| 4-5 | 400 | 300 | 2.0 |
| 5-6 | 200 | 150 | 2.0 |
| 6-7 | 100 | 100 | 2.0 |
| 7-8 | 50 | 50 | 2.0 |
| 8-9 | 20 | 20 | 2.0 |
| 9-10 | 10 | 10 | 2.0 |
| 10-11 | 5 | 5 | 2.0 |
| 11-12 | 2 | 2 | 2.0 |
</details>

(a) Sparsity: 10%

![](images/2b13790e8e3c6538fb84a1fcd90e399fb4bd52869ed71b74eca45946afe0eafb.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 0.0 - 2.5 | 1500 |
| 2.5 - 5.0 | 3000 |
| 5.0 - 7.5 | 800 |
| 7.5 - 10.0 | 400 |
| 10.0 - 12.5 | 200 |
| 12.5 - 15.0 | 100 |
</details>

(b) Sparsity: 20%

![](images/e95c2b87288723919cc8b9f34654731f0515118a5f69a600f2b46dcddf54b27a.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Intra-Fusion | Default | Mean (Default): 8.3 |
|---|---|---|---|
| 0.0-2.5 | 1500 | 0 | 0 |
| 2.5-5.0 | 1750 | 0 | 0 |
| 5.0-7.5 | 750 | 0 | 0 |
| 7.5-10.0 | 500 | 900 | 0 |
| 10.0-12.5 | 400 | 1100 | 0 |
| 12.5-15.0 | 100 | 600 | 0 |
Mean (Default): 8.3
Mean (IF): 5.8
</details>

(c) Sparsity: 30%   
Figure 5: Output preservation comparison of the original model. Model: ResNet18. Dataset: CIFAR-10. Group: 0. Importance metric: $\ell_{1}$ .

# 5.2 NEURAL LANDSCAPE

To gather further insights as to how Intra-Fusion differs from the conventional pruning methodology when navigating the weight space, we show an example of a pruned network and the corresponding accuracy landscape that surrounds the models. We vectorize all the parameters before carrying out their linear interpolation for this figure and following the procedure in (Garipov et al., 2018). We specify three models in that space (in our case: original model, default pruned model, and the Intra-Fusion model), one of which serves as the origin, and thus a 2D slice is built. In this slice, we sample a grid of possible networks and evaluate their performance on the test set. It is important to note that not all models in the given 2D slice of the parameter space are pruned (e.g. original model).

![](images/d381cf4963d1126b51f2ab6fbc867f628d562363459f0d64a184da1d7fbfd782.jpg)

<details>
<summary>heatmap</summary>

| Category | Value |
| -------- | ----- |
| Original | 95%   |
| Default  | 76%   |
| IF       | 93%   |
</details>

(a) Sparsity: 40%

![](images/c96b74c7b44506199d9325c8d52cba6e4cf76d64d92035dd6ae002fefb6f651a.jpg)

<details>
<summary>heatmap</summary>

| Category | Accuracy |
| -------- | -------- |
| Original | 95%      |
| Default  | 31%      |
| IF       | 77%      |
</details>

(b) Sparsity: 60%   
Figure 6: Resnet18. CIFAR-10, when group 6 is pruned. IF: Intra-Fusion.

In Figure 6 we depict how the accuracy varies for different models in the identified 2D slice of the parameter space. Evidently, the Intra-Fusion model ends up at a more convenient part of the accuracy landscape when compared to the regularly pruned model ("Default") resulting in a superior performance. To further explore how the accuracy landscape develops through different sparsities and when the whole model is pruned, we include additional figures in Appendix C.2. It appears that using the less important neurons to inform the process of model pruning is more effective at identifying a favorable spot in the parameter space than simply discarding them.

# 5.3 ABLATION STUDY ON VARYING TARGET AND SOURCE DISTRIBUTION

As mentioned in Section 3.2, we present two choices for the source and target distribution: uniform and importance-informed. In order to discern which option is advantageous, we prune six ResNet18 models using different combinations of source and target distributions with the $\ell_{1}$ -norm criterion. We employ the Kruskal-Wallis H-test (Kruskal & Wallis, 1952) to assess statistically significant differences between the distribution choices. The test reveals that the p-values for nearly all cases indicate statistical insignificance. In cases where significance is observed ( $p \leq 0.05$ ), the differences in accuracy are marginal. However, it is plausible that a more informative importance metric than the $\ell_{1}$ -norm could give the importance-informed version an advantage. See Appendix C.3 for more details.

# 6 APPLICATION BEYOND PRUNING: FACTORIZING MODEL TRAINING

In an attempt to find further valuable integrations of pruning and fusion, we also explore how fusion can be used to “factorize” and possibly speed up the training process of models that are supposed to be pruned after training. In an increasingly digitalized world the amount of available data points is consistently increasing. We are looking for an approach that manages to leverage the fact that also on subsets of the whole dataset, a competitive performance can be achieved. This factorization provides another angle on enhancing the performance and training time of models. It could for example be especially interesting as an alternative or enhancement to data parallelism during distributed model training.

The model that is created with the standard pruning approach (trained on the whole dataset, then pruned and finally fine-tuned) we call the “whole-data model”. Like in a real-world setting, the result of the pruning is fine-tuned since this most of the time recovers a lot of performance.

# 6.1 THE SPLIT-DATA APPROACH

In the approach we want to propose, we split the model training into smaller phases by utilizing pruning and fusion. For this we first split the data set into two subsets a and b on which we then train two individual models $model_{a}$ , $model_{b}$ in parallel. This leads to a theoretical 2x speedup in the model training time since the convergence on half of the dataset does not take more epochs than on the whole dataset (see Figure 20). These two models are then fused using OT and fine-tuned on the whole data set. To reach the target sparsity we prune the resulting network and fine-tune again. We call this approach FaP (Fuse and Prune, see Figure 7). For completeness we also include the performance of individually pruning and fine-tuning the networks before fusing — this we call PaF (Prune and Fuse, see Figure 21). For the pruning part of doing FaP and PaF, we use the introduced Intra-Fusion.

![](images/96066ab31335a687d2a2b2e9c19282230279cc3dcbb29ff3c94ac3ccf1d6f763.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["untrained network architecture"] --> B["whole data set"]
    C["training on a data subset"] --> D["model a"]
    E["fusing the models"] --> F["model b"]
    G["fine-tuning on the whole data set"] --> H["model c"]
    I["pruning the model"] --> J["model d"]
    K["fine-tuning on the whole data set"] --> L["model e"]
    M["FaP model"] --> N["model f"]
```
</details>

Figure 7: The FaP approach.

To give a concrete insight into the relative runtimes of the different approaches, we give a side-by-side comparison of their timelines in Figure 22. Relative to the model training time of a VGG11-BN (CIFAR-10), we achieve a speedup of 1.81. The overall speedups of the split-data approaches are 1.42 (PaF) and 1.31 (FaP). Importantly, in applications, the time for training $T_{1}$ of a model on large datasets will typically be much greater than the time for fine-tuning $T_{2}$ ( $T_{1} \gg T_{2}$ ), and speedups are expected to be much more significant.

# 6.2 K-FOLD SPLIT-DATA

As an alternative, to simply splitting the dataset into two and using each subset to train a model (that's what we have done so far), we can generalize to a k-fold style approach.

Generating Additional Trainingsets. Here we split the dataset into k equally sized and distinct subsets $s_{p}$ . We now create training datasets $d_{i}$ that consist of k/2-many of these subsets $s_{p}$ . By choosing k mod 2 = 0 we ensure that each $d_{i}$ will end up containing 50% of the original dataset. We then take all possible $\binom{k}{k/2}$ -many $d_{i}$ and individually train models on them.

Extending PaF and FaP. Since the fusion algorithm naturally extends to fusing more than two models (as also presented by (Singh & Jaggi, 2020)), we can now generalize PaF and FaP to combine pruning and fusion of more than two models - leading to a more effective use of the OT-based fusion approach.

Consequences for Model Training Speedup. This approach comes with additional requirements for computational resources to enable the parallel training and pruning of the multiple individually trained models. However, besides fusion taking insignificantly longer, it does not require more time than the previously explored split-data approaches and thus yields the same speedup.

# 6.3 SPLIT-DATA PERFORMANCE

Deploying the Split-Data approach when training Resnet18 on CIFAR-10, we were able to recover and even slightly improve over the "whole-data model" performance, while providing a significant speedup in the training time of the involved models (see Appendix D.2). For VGG11-BN we achieve similar results at the cost of higher resource requirements ("k-Fold" setting, see Appendix D.4).

We delve deeper into the details and performance of this “Split-Data” concept in Appendix D. Specifically see “Performance Comparison: After Convergence” (Appendix D.2) and “Performance Comparison: Varying Fine-Tuning” (Appendix D.3).

# 6.4 SPLIT-DATA AS ALTERNATIVE TO DATA PARALLELISM

Splitting the dataset into different parts that models are trained on individually is not a new idea. Specifically, during distributed model training in cloud infrastructures, this is a common approach called Data-Parallelism. However, the gradients are exchanged among the models after the individual backpropagation steps so all models are effectively updated with gradients computed from the whole dataset. This leads to high network utilization, sensitivity to network latency and wait times between the training steps.

In our Split-Data approaches (like PaF and FaP), we completely separate the model training. Each model has its designated part of the training data it is trained on. Only after the models have individually converged are the edge weights communicated and fused. This yields far less communication overhead and is not sensitive to network latency.

# 7 CONCLUSION

In sum, we perform a detailed investigation of unifying and bridging the paradigms of pruning and fusion through our conceptions of Pruning-and-Fusion as well as Fusion-and-Pruning. Specifically, we showed how our proposed technique of Intra-Fusion provides a consistent gain — with and without fine-tuning, the latter being also privacy-preserving and highly cost-efficient. We also investigated how fusion can be used to factorize the training process, given that it is subsequently accompanied by pruning, to result in non-trivial speedup in training times. The sparsity obtained via our algorithm is also amenable to actual speedup in inference times, without needing special accelerators. Overall, our work shows the compatibility of bringing together pruning and fusion in the form of meta-pruning, and the potential inherent therein. All in all, this raises the question of rethinking the pre-dominant paradigm and perhaps redefining our approach to obtaining compact models via fusion.

# ACKNOWLEDGEMENTS

Sidak Pal Singh would like to acknowledge the financial support from Max Planck ETH Center for Learning Systems.

# REFERENCES

Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan, 2017.   
Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. What is the state of neural network pruning? Proceedings of machine learning and systems, 2:129–146, 2020.   
Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm. int8(): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.   
Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and Xinchao Wang. Depgraph: Towards any structural pruning, 2023.   
Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018.   
Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot, 2023.   
Charlie Frogner, Chiyuan Zhang, Hossein Mobahi, Mauricio Araya, and Tomaso A Poggio. Learning with a wasserstein loss. Advances in neural information processing systems, 28, 2015.   
Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems, 31, 2018.   
Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129:1789–1819, 2021.   
Song Han, Jeff Pool, John Tran, and William J. Dally. Learning both weights and connections for efficient neural networks, 2015.   
Babak Hassibi, David G. Stork, and Gregory J. Wolff. Optimal brain surgeon and general network pruning. IEEE International Conference on Neural Networks, pp. 293–299 vol.1, 1993.   
Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang. Filter pruning via geometric median for deep convolutional neural networks acceleration. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4335–4344, 2019. doi: 10.1109/CVPR.2019.00447.

Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In 2017 IEEE International Conference on Computer Vision (ICCV), pp. 1398–1406, 2017. doi: 10.1109/ICCV.2017.155.   
Richard M. Heiberger and Erich Neuwirth. One-Way ANOVA, pp. 165–191. Springer New York, New York, NY, 2009. ISBN 978-1-4419-0052-4. doi: 10.1007/978-1-4419-0052-4\_7. URL https://doi.org/10.1007/978-1-4419-0052-4\_7.   
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.   
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. CoRR, abs/1502.03167, 2015. URL http://arxiv.org/abs/1502.03167.   
Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.   
William H. Kruskal and Wilson Allen Wallis. Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47:583–621, 1952. URL https://api.semanticscholar.org/CorpusID:51902974.   
Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. From word embeddings to document distances. In International conference on machine learning, pp. 957–966. PMLR, 2015.   
Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. Advances in neural information processing systems, 2, 1989.   
Jaeho Lee, Sejun Park, Sangwoo Mo, Sungsoo Ahn, and Jinwoo Shin. Layer-adaptive sparsity for the magnitude-based pruning, 2021.   
Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft. Convergent learning: Do different neural networks learn the same representations? arXiv preprint arXiv:1511.07543, 2015.   
Stuart P. Lloyd. Least squares quantization in pcm. IEEE Trans. Inf. Theory, 28:129–136, 1982. URL https://api.semanticscholar.org/CorpusID:10833328.   
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273–1282. PMLR, 2017.   
Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. Importance estimation for neural network pruning. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11256–11264, 2019. doi: 10.1109/CVPR.2019.01152.   
Sidak Pal Singh and Dan Alistarh. Woodfisher: Efficient second-order approximation for neural network compression. Advances in Neural Information Processing Systems, 33:18098–18109, 2020.   
Sidak Pal Singh and Martin Jaggi. Model fusion via optimal transport. Advances in Neural Information Processing Systems, 33:22045–22055, 2020.   
Yang Sui, Miao Yin, Yi Xie, Huy Phan, Saman Aliari Zonouz, and Bo Yuan. Chip: Channel independence-based pruning for compact neural networks. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 24604–24616. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/file/ce6babd060aa46c61a5777902cca78af-Paper.pdf.   
Cédric Villani et al. Optimal transport: old and new, volume 338. Springer, 2009.

Chaoqi Wang, Roger B. Grosse, Sanja Fidler, and Guodong Zhang. Eigendamage: Structured pruning in the kronecker-factored eigenbasis. CoRR, abs/1905.05934, 2019. URL http://arxiv.org/abs/1905.05934.   
Hongyi Wang, Mikhail Yurochkin, Yuekai Sun, Dimitris Papailiopoulos, and Yasaman Khazaeni. Federated learning with matched averaging. arXiv preprint arXiv:2002.06440, 2020.   
Guangxuan Xiao, Ji Lin, Mickael Seznec, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. arXiv preprint arXiv:2211.10438, 2022.   
Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. Advances in Neural Information Processing Systems, 35:27168–27183, 2022.   
Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I. Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S. Davis. Nisp: Pruning networks using neuron importance score propagation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9194–9203, 2018. doi: 10.1109/CVPR.2018.00958.   
Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. On compressing deep models by low rank and sparse decomposition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7370–7379, 2017.   
Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Nghia Hoang, and Yasaman Khazaeni. Bayesian nonparametric federated learning of neural networks. In International Conference on Machine Learning, pp. 7252–7261. PMLR, 2019.   
Chen Zhang, Yu Xie, Hang Bai, Bin Yu, Weihong Li, and Yuan Gao. A survey on federated learning. Knowledge-Based Systems, 216:106775, 2021.   
Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, and Hongsheng Li. Learning n: m fine-grained structured sparse neural networks from scratch. arXiv preprint arXiv:2102.04010, 2021a.   
Da-Wei Zhou, Han-Jia Ye, and De-Chuan Zhan. Co-transport for class-incremental learning. In Proceedings of the 29th ACM International Conference on Multimedia, MM '21, pp. 1645–1654, New York, NY, USA, 2021b. Association for Computing Machinery. ISBN 9781450386517. doi:10.1145/3474085.3475306. URL https://doi.org/10.1145/3474085.3475306.   
Maohua Zhu, Tao Zhang, Zhenyu Gu, and Yuan Xie. Sparse tensor core: Algorithm and hardware co-design for vector-wise sparse neural networks on modern gpus. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 359–371, 2019.

# Appendix

# Table of Contents

A Experimental Hyperparameters 15

A.1 Training 15   
A.2 Intra-Fusion Settings 15

B Implementation 15

C Understanding Intra-Fusion 15

C.1 Output Preservation 15   
C.2 Neural Landscape 18   
C.3 Ablation Study on Varying Target and Source Distribution 19   
C.4 Agnosticism to Importance Metrics 21

D Application Beyond Pruning: Factorizing Model Training 23

D.1 Runtime Comparison of Split-Data and Whole-Data approach 23   
D.2 Performance Comparison: After Convergence 24   
D.3 Performance Comparison: Varying Fine-Tuning 25   
D.4 k-Fold Split-Data 25   
D.5 Extensions for Future Work 26   
D.6 Performance of Models Used 26

E Empirical Results 28

E.1 Terminology 28   
E.2 Data-Free Experiments 28

# A EXPERIMENTAL HYPERPARAMETERS

# A.1 TRAINING

During the training of the used VGG11-BN and Resnet18 networks, we deploy the training hyperparameters in Table 2. For the fine-tuning of models after pruning we use the hyperparameters in Table 3.

Table 2: Hyperparameters during model training. 

<table><tr><td>Loss function</td><td>Cross Entropy</td></tr><tr><td>Optimizer</td><td>SGD with momentum = 0.9</td></tr><tr><td>Learning Rate Schedule</td><td> $0.05 \times 0.5^{\lfloor epoch/30 \rfloor}$ </td></tr><tr><td>Training Epochs</td><td>300</td></tr><tr><td>Batch Size</td><td>128</td></tr></table>

Table 3: Hyperparameters during fine-tuning. 

<table><tr><td>Loss function</td><td>Cross Entropy</td></tr><tr><td>Optimizer</td><td>SGD with momentum = 0.9</td></tr><tr><td>Learning Rate Schedule</td><td> $0.01 \times 0.5^{\lfloor epoch/30 \rfloor}$ </td></tr><tr><td>Batch Size</td><td>128</td></tr></table>

# A.2 INTRA-FUSION SETTINGS

For our data-free and data-driven results, we use a homogeneous distribution for both the target and source distribution. Moreover, we use the most important neuron pairings as the target.

# B IMPLEMENTATION

As mentioned in Section 3, structural pruning is inherently complex due to the inderdependencies existing within state-of-the-art neural networks. Researchers and engineers have relied on manually-designed and model-specific schemes to handle these. Evidently, this is intractable and not scalable, particularly for more complex networks.

Recently, (Fang et al., 2023) introduced a fully-automatic and general way to structurally prune neural networks, by extracting a dependency graph from the computational graph derived by backpropagation. Thus, in order for Intra-Fusion to be generally applicable across a wide range of models in an automated fashion, we have extended their library to work with Intra-Fusion. This way, Intra-Fusion can be applied to a wide and diverse set of models without the user having to adapt the code in any way.

# C UNDERSTANDING INTRA-FUSION

In the following sections, we want to further expand on Section 5.1 by providing more extensive results and background information.

# C.1 OUTPUT PRESERVATION

The following figures show how well Intra-Fusion is able to preserve the output of the non-pruned model for VGG11 and ResNet18 on CIFAR-10 and CIFAR-100. As before, Intra-Fusion seems to be able to better preserve the output at low to high sparsities.

![](images/1425e9aab8e808da20ae57de45506c7d00249f36a8195b2f790a24fe0ec7e291.jpg)  
(a) Sparsity: 10%

![](images/d391b1420b541485b0f9c965c9782f75b15323db39412fc63ecfc719579662f6.jpg)

<details>
<summary>histogram</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 0-5                                      | 100       |
| 5-10                                     | 1800      |
| 10-15                                    | 2000      |
| 15-20                                    | 1500      |
| 20-25                                    | 500       |
| 25-30                                    | 100       |
</details>

(b) Sparsity: 20%

![](images/ccbe81602b5d6b49345133781e78c6acbbb8bdd20441414014120cb91b5c5def.jpg)

<details>
<summary>histogram</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 5-7                                      | 100       |
| 7-9                                      | 400       |
| 9-11                                     | 900       |
| 11-13                                    | 1400      |
| 13-15                                    | 1500      |
| 15-17                                    | 1400      |
| 17-19                                    | 1300      |
| 19-21                                    | 1200      |
| 21-23                                    | 1000      |
| 23-25                                    | 800       |
| 25-27                                    | 600       |
| 27-29                                    | 400       |
| 29-31                                    | 200       |
| 31-33                                    | 100       |
| 33-35                                    | 50        |
</details>

(c) Sparsity: 30%   
Figure 8: Output preservation comparison of the original model. Model: VGG11. Dataset: CIFAR-10. Group: 0. Importance metric: $\ell_{1}$ .

![](images/d954f0c358a605152dc43b3b4177594c41985cb922d2c54b03e7fcf4b0c7b90c.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 10                                       | 400       |
| 15                                       | 1600      |
| 20                                       | 2900      |
| 25                                       | 2500      |
| 30                                       | 1700      |
| 35                                       | 1000      |
| 40                                       | 600       |
| 45                                       | 300       |
| 50                                       | 100       |
</details>

(a) Sparsity: 10%

![](images/f4cf1d9b3d15d757f41aa7bfd32188afa5380bd61dab6a8613d10b3f63c5cd69.jpg)  
(b) Sparsity: 20%

![](images/11735edfe402a97059063e58b31a506d7efb20f8d5082754dc0b642f39389887.jpg)  
(c) Sparsity: 30%   
Figure 9: Output preservation comparison of the original model. Model: ResNet18. Dataset: CIFAR-100. Group: 0. Importance metric: $\ell_{1}$ .

![](images/a923114c0ba7951fce6fd8e5e50f807a424393f6bbb4ab6ea0d8aa5c8c71edae.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 10-15 | 1000 |
| 15-20 | 3000 |
| 20-25 | 1740 |
| 25-30 | 600 |
| 30-35 | 1600 |
| 35-40 | 1200 |
| 40-45 | 800 |
| 45-50 | 400 |
| 50-55 | 200 |
| 55-60 | 100 |
</details>

(a) Sparsity: 10%

![](images/22be7b5f0298afab8d95f75f5877f34e7172227d5d4610db0637f8a365022a90.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 20                                       | 2700      |
| 30                                       | 1900      |
| 40                                       | 1600      |
| 50                                       | 1200      |
| 60                                       | 800       |
| 70                                       | 400       |
| 80                                       | 200       |
</details>

(b) Sparsity: 20%

![](images/042a2c8639b156d6860b5ee2952eb982761c6bf422820607bbd7402a7be49a35.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 30 | 2500 |
| 31 | 2200 |
| 32 | 2000 |
| 33 | 1800 |
| 34 | 1600 |
| 35 | 1400 |
| 36 | 1200 |
| 37 | 1000 |
| 38 | 800 |
| 39 | 600 |
| 40 | 400 |
| 41 | 200 |
| 42 | 100 |
| 43 | 50 |
| 44 | 20 |
| 45 | 10 |
| 46 | 5 |
| 47 | 2 |
| 48 | 1 |
| 49 | 0.5 |
| 50 | 0.2 |
| 51 | 0.1 |
| 52 | 0.05 |
| 53 | 0.02 |
| 54 | 0.01 |
| 55 | 0.005 |
| 56 | 0.002 |
| 57 | 0.001 |
| 58 | 0.0005 |
| 59 | 0.0002 |
| 60 | 0.0001 |
| 61 | 0.00005 |
| 62 | 0.00002 |
| 63 | 0.00001 |
| 64 | 0.000005 |
| 65 | 0.000002 |
| 66 | 0.000001 |
| 67 | 0.0000005 |
| 68 | 0.0000002 |
| 69 | 0.0000001 |
| 70 | 0.00000005 |
| 71 | 0.00000002 |
| 72 | 0.00000001 |
| 73 | 0.000000005 |
| 74 | 0.000000002 |
| 75 | 0.000000001 |
| 76 | 0.0000000005 |
| 77 | 0.0000000002 |
| 78 | 0.0000000001 |
| 79 | 0.00000000005 |
| 80 | 0.00000000002 |
</details>

(c) Sparsity: 30%   
Figure 10: Output preservation comparison of the original model. Model: VGG11. Dataset: CIFAR-100. Group: 0. Importance metric: $\ell_{1}$ .

![](images/cf6980652426282f1133d1c698b1ff4476ee9581d63b55ea2f32285b78af4f3c.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 0-1                                      | 3200      |
| 1-2                                      | 3600      |
| 2-3                                      | 1400      |
| 3-4                                      | 800       |
| 4-5                                      | 400       |
| 5-6                                      | 200       |
| 6-7                                      | 100       |
| 7-8                                      | 50        |
| 8-9                                      | 20        |
| 9-10                                     | 10        |
| 10-11                                    | 5         |
| 11-12                                    | 2         |
</details>

(a) Sparsity: 10%

![](images/621fc2271035e43558a072eb1092fbb0466814a3e4d1cea0aebe058addb2fe21.jpg)

<details>
<summary>histogram</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 0.0 - 2.5 | 1500 |
| 2.5 - 5.0 | 3000 |
| 5.0 - 7.5 | 1000 |
| 7.5 - 10.0 | 500 |
| 10.0 - 12.5 | 200 |
| 12.5 - 15.0 | 100 |
</details>

(b) Sparsity: 20%

![](images/86ad07e119df154fd73e85f5f1625ac00ef3e2eb8feff4c3a66d33af629dab80.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 0.0-2.5 | 150 |
| 2.5-5.0 | 1100 |
| 5.0-7.5 | 900 |
| 7.5-10.0 | 800 |
| 10.0-12.5 | 1050 |
| 12.5-15.0 | 600 |
</details>

(c) Sparsity: 30%   
Figure 11: Output preservation comparison of the original model. Model: ResNet18. Dataset: CIFAR-10. Group: 1. Importance metric: $\ell_{1}$ .

![](images/53bcfc610b9cdf1888e5f2225b926b20c763b9f475c213be8aa0605e6037fbd6.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Intra-Fusion | Default | Mean (Default): 7.8 |
|---|---|---|---|
| 0-1 | 100 | 0 | 0 |
| 1-2 | 300 | 0 | 0 |
| 2-3 | 900 | 0 | 0 |
| 3-4 | 1900 | 0 | 0 |
| 4-5 | 2000 | 0 | 0 |
| 5-6 | 1800 | 0 | 0 |
| 6-7 | 1600 | 0 | 0 |
| 7-8 | 1200 | 0 | 0 |
| 8-9 | 750 | 0 | 0 |
| 9-10 | 500 | 0 | 0 |
| 10-11 | 250 | 0 | 0 |
| 11-12 | 150 | 0 | 0 |
| 12-13 | 50 | 0 | 0 |
| 13-14 | 25 | 0 | 0 |
| 14-15 | 15 | 0 | 0 |
| 15-16 | 5 | 0 | 0 |
| 16-17 | 2 | 0 | 0 |
| 17-18 | 1 | 0 | 0 |
| 18-19 | 0 | 0 | 0 |
| 19-20 | 0 | 0 | 0 |
The chart displays frequency distributions for each method. The x-axis represents the distance to outputs of the original model, and the y-axis represents frequency. Legend categories are 'Intra-Fusion', 'Default', and 'Mean (Default)'. The mean (Default) is explicitly labeled as 7.8. The dashed line indicates a mean of the default. The data shows that the mean of the default is approximately 6.6, while the actual mean of the default is approximately 7.8. The bars for the 'Intra-Fusion' and 'Default' categories are not explicitly labeled but are visually present in the legend. The chart is saved as a PNG file named 'chart_footnote.png'.
</details>

(a) Sparsity: 10%

![](images/6cd3ea00bab7225d17af0b6a6a50d0fa57176139b62e7573e855f57ff360963e.jpg)

<details>
<summary>histogram</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 0-5                                      | 100       |
| 5-10                                     | 1800      |
| 10-15                                    | 2000      |
| 15-20                                    | 1500      |
| 20-25                                    | 500       |
| 25-30                                    | 100       |
</details>

(b) Sparsity: 20%

![](images/892c6e3bfa06ff178c51f007d591f65f75648f2f1c36192c9f9e148c32548613.jpg)

<details>
<summary>histogram</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 5-7                                      | 100       |
| 7-9                                      | 400       |
| 9-11                                     | 900       |
| 11-13                                    | 1400      |
| 13-15                                    | 1500      |
| 15-17                                    | 1400      |
| 17-19                                    | 1300      |
| 19-21                                    | 1200      |
| 21-23                                    | 1000      |
| 23-25                                    | 800       |
| 25-27                                    | 600       |
| 27-29                                    | 400       |
| 29-31                                    | 200       |
| 31-33                                    | 100       |
| 33-35                                    | 50        |
</details>

(c) Sparsity: 30%   
Figure 12: Output preservation comparison of the original model. Model: VGG11. Dataset: CIFAR-10. Group: 1. Importance metric: $\ell_{1}$ .

![](images/4aa1bbb611023ff7aa7b9619bd024860975e6c3c7edfb54fe8c05f4b43acca1b.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Intra-Fusion | Default | Mean (IF) | Mean (Default): |
|---|---|---|---|---|
| 10-15 | 400 | 0 | 0 | 0 |
| 15-20 | 1600 | 0 | 0 | 0 |
| 20-25 | 2900 | 1800 | 2500 | 24.5 |
| 25-30 | 2100 | 1700 | 2100 | 18.1 |
| 30-35 | 1600 | 1000 | 1600 | 0 |
| 35-40 | 600 | 200 | 600 | 0 |
| 40-45 | 100 | 100 | 100 | 0 |
| 45-50 | 50 | 50 | 50 | 0 |
</details>

(a) Sparsity: 10%

![](images/68d628a1beced9c6726e7403b023123cc76ed319999a994ad51f4f33ac18ee79.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 10-15 | 600 |
| 15-20 | 1550 |
| 20-25 | 2300 |
| 25-30 | 1400 |
| 30-35 | 900 |
| 35-40 | 1800 |
| 40-45 | 1000 |
| 45-50 | 600 |
| 50-55 | 200 |
| 55-60 | 100 |
| Mean (Default): 31.3, Mean (IF): 23.2
</details>

(b) Sparsity: 20%

![](images/8bfc7b92d485ba5a9629c70ae70c782bbb1eb6d989916f3f670ce3bd9a4df28a.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 20-25 | 1200 |
| 25-30 | 1800 |
| 30-35 | 2000 |
| 35-40 | 1900 |
| 40-45 | 1700 |
| 45-50 | 1400 |
| 50-55 | 600 |
| 55-60 | 200 |
| 60+ | 50 |
</details>

(c) Sparsity: 30%   
Figure 13: Output preservation comparison of the original model. Model: ResNet18. Dataset: CIFAR-100. Group: 1. Importance metric: $\ell_{1}$ .

![](images/071225ab86c3a0ad10f032d7d6d268ea8a54e426d4ade2cec7da99390f881764.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 10-15 | 1000 |
| 15-20 | 3000 |
| 20-25 | 1800 |
| 25-30 | 1600 |
| 30-35 | 1200 |
| 35-40 | 800 |
| 40-45 | 400 |
| 45-50 | 200 |
| 50-55 | 100 |
| 55-60 | 50 |
</details>

(a) Sparsity: 10%

![](images/fc7796b40ee3c4e580ddbedcaec243cc0f79075e4b7d088d2eb2643173deb063.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
| ---------------------------------------- | --------- |
| 20                                       | 2600      |
| 30                                       | 1900      |
| 40                                       | 1700      |
| 50                                       | 1200      |
| 60                                       | 600       |
| 70                                       | 100       |
| 80                                       | 50        |
</details>

(b) Sparsity: 20%

![](images/882189724c52d309547fcdaea5654410077caf21c3aca31d4bec31ca32eec833.jpg)

<details>
<summary>bar</summary>

| Distance to outputs of the original model | Frequency |
|---|---|
| 30 | 2150 |
| 35 | 2500 |
| 40 | 1800 |
| 45 | 1700 |
| 50 | 1400 |
| 55 | 1200 |
| 60 | 700 |
| 65 | 400 |
| 70 | 100 |
| 75 | 50 |
| 80 | 20 |
</details>

(c) Sparsity: 30%   
Figure 14: Output preservation comparison of the original model. Model: VGG11. Dataset: CIFAR-100. Group: 1. Importance metric: $\ell_{1}$ .

# C.2 NEURAL LANDSCAPE

We expand the experiment in Section 5.1 by showing results for more sparsities and model-dataset combinations.

![](images/663554f3e4ac7325520ea0c194b6d4308155f774fa367333f4f1ffe1141d2caf.jpg)

Figure 15: Slice of the model weight space: Resnet18, CIFAR-10, neuron sparsity: 10%-80%, criterion: $\ell_{1}$ , no fine-tuning. IF: Intra-Fusion.   
![](images/a0930860178267c91e1f86d855748c83571c3a453514fe182905c2d0d856e434.jpg)

<details>
<summary>heatmap</summary>

| Sparsity | Condition | Original | IF   | Default |
|----------|-----------|----------|------|---------|
| 10%      | Original  | 20       | 0    | 0       |
| 10%      | Default   | 20       | 0    | 0       |
| 20%      | Original  | 30       | 0    | 0       |
| 20%      | Default   | 30       | 0    | 0       |
| 40%      | Original  | 50       | 0    | 0       |
| 40%      | Default   | 50       | 0    | 0       |
| 60%      | Original  | 50       | 0    | 0       |
| 60%      | Default   | 50       | 0    | 0       |
| 80%      | Original  | 50       | 0    | 0       |
| 80%      | Default   | 50       | 0    | 0       |
</details>

Figure 16: Slice of the model weight space: VGG11-BN, CIFAR-10, neuron sparsity: 10%-80%, criterion: $\ell_{1}$ , no fine-tuning. IF: Intra-Fusion.

![](images/3e208ecb8679406166b2948619d79b3a8341ea04b04485025f28311f0af9aa19.jpg)

<details>
<summary>heatmap</summary>

| Sparsity Level | Condition | Original Accuracy | IF Default Accuracy |
| -------------- | --------- | ----------------- | ------------------- |
| 10%            | Original  | ~100              | ~0                  |
| 10%            | IF        | ~100              | ~0                  |
| 15%            | Original  | ~100              | ~0                  |
| 15%            | IF        | ~100              | ~0                  |
| 15%            | Default   | ~100              | ~0                  |
| 20%            | Original  | ~100              | ~0                  |
| 20%            | IF        | ~100              | ~0                  |
| 20%            | Default   | ~100              | ~0                  |
| 30%            | Original  | ~100              | ~0                  |
| 30%            | IF        | ~100              | ~0                  |
| 30%            | Default   | ~100              | ~0                  |
| 40%            | Original  | ~100              | ~0                  |
| 40%            | IF        | ~100              | ~0                  |
| 40%            | Default   | ~100              | ~0                  |
</details>

Figure 17: Slice of the model weight space: Resnet18, CIFAR-100, neuron sparsity: 10%-40%, criterion: $\ell_{1}$ , no fine-tuning. IF: Intra-Fusion.

![](images/a302406766b8aedb0acccdf378a7e9d90321eb4fce5355a3cc352d13701c6b47.jpg)

<details>
<summary>heatmap</summary>

| Sparsity | Original | IF   | Default |
| -------- | -------- | ---- | ------- |
| 10%      | 50       | 0    | 0       |
| 15%      | 50       | 0    | 0       |
| 20%      | 50       | 0    | 0       |
| 30%      | 50       | 0    | 0       |
| 40%      | 50       | 0    | 0       |
</details>

Figure 18: Slice of the model weight space: VGG11-BN, CIFAR-100, neuron sparsity: 10%-40%, criterion: $\ell_{1}$ , no fine-tuning. IF: Intra-Fusion.

# C.3 ABLATION STUDY ON VARYING TARGET AND SOURCE DISTRIBUTION

In this section, we intend to shed a light on the potential differences between choosing uniform or an importance-informed distribution for the target and source distribution, as explained in Section 3.2.

In our analysis, we trained six ResNet18 models on CIFAR-10, applying $\ell_{1}$ criterion-based pruning across various groups and sparsity levels. The source and target distributions for the optimal transport (OT) setting were chosen as either uniform or importance-informed. To assess the significance of distribution choices, we computed the p-value using the Kruskal-Wallis H-test (Kruskal & Wallis, 1952) for each group-sparsity combination. This test evaluates whether the population medians of all distributions are equal, serving as a non-parametric alternative to ANOVA (Heiberger & Neuwirth, 2009).

The resulting p-values for each group-sparsity pair are visualized in Figure 19. We consider p-values less than or equal to 0.05 as statistically significant. Interestingly, the majority of cases do not exhibit a significant difference. However, for groups 10 and 11, with sparsities ranging from 10% to 50%, notable and statistically significant differences between the distributions emerge.

In Table 4, we focus on Group 10 and 11 to discern nuances in performance. The optimal distribution choices are highlighted in green for the best-performing and in red for the least effective. Notably, the uTuS (uniform target, uniform source) and iTuS (importance-informed target, uniform source) configurations appear superior, while uTiS (uniform target, importance-informed source) performs less favorably. However, it is crucial to emphasize that the variations in accuracy are subtle, seldom exceeding one percentage point.

In light of these findings, we deduce that the selection between uniform or importance-informed distributions for both source and target in the optimal transport (OT) context lacks statistically significant impact in the majority of cases. Furthermore, in instances where statistical significance is observed, the differences remain marginal.

Remark: As mentioned before, the pruning criterion used in this study is based on the $\ell_{1}$ -norm. Evidently, this necessarily affects the importance-informed distribution. Thus, it might be possible that some other, more meaningful importance criterion could yield improvements for the importance-informed distributions. However, in light of the results we expect the differences to be more or less marginal.

![](images/46e60ff1cf3f4dd83e85cb73bffe9679a137a039bbbb35c73bfef68a9f12ec48.jpg)

<details>
<summary>heatmap</summary>

| Group | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|
| 0 | 0.44 | 0.36 | 0.31 | 0.59 | 0.58 | 0.85 | 0.80 |
| 1 | 0.70 | 0.76 | 0.75 | 0.87 | 0.99 | 0.86 | 0.94 |
| 2 | 0.95 | 0.96 | 0.94 | 0.94 | 0.93 | 0.79 | 0.98 |
| 3 | 0.50 | 0.49 | 0.02 | 0.24 | 0.95 | 0.69 | 0.99 |
| 4 | 0.93 | 0.86 | 0.65 | 0.82 | 0.80 | 0.90 | 0.96 |
| 5 | 0.78 | 0.89 | 0.95 | 0.91 | 0.32 | 0.81 | 0.98 |
| 6 | 0.12 | 0.38 | 0.36 | 0.78 | 0.09 | 0.64 | 0.12 |
| 7 | 0.59 | 0.39 | 0.92 | 0.81 | 0.71 | 0.78 | 0.72 |
| 8 | 0.78 | 0.96 | 0.66 | 0.98 | 0.68 | 0.98 | 1.00 |
| 9 | 0.01 | 0.03 | 0.61 | 0.90 | 0.87 | 0.79 | 0.90 |
| 10 | 0.01 | 0.23 | 0.06 | 0.03 | 0.06 | 0.90 | 1.00 |
| 11 | 0.01 | 0.01 | 0.21 | 0.06 | 0.04 | 0.88 | 0.76 |
</details>

Figure 19: Kruskal-Wallis H-test (Kruskal & Wallis, 1952) p-value between uniform and importance-informed source and target distributions, for different group and sparsity pairs. Model: ResNet18. Dataset: CIFAR-10. Pruning criteria: $\ell_{1}$ .

Table 4: Data-free results for different source and target distributions. uTuS: Uniform target, uniform source. uTis: Uniform target, importance-informed source. iTuS: Importance-informed target, uniform source. iTis: importance-informed target, importance-informed source. Model: ResNet18. Dataset: CIFAR-100. Criterion: $\ell_{1}$ . 

<table><tr><td>Group</td><td>Sparsity (%)</td><td>uTuS (%)</td><td>uTiS (%)</td><td>iTuS (%)</td><td>iTiS (%)</td></tr><tr><td rowspan="5">Group 10</td><td>10</td><td>94.84</td><td>94.49</td><td>94.1</td><td>94.84</td></tr><tr><td>20</td><td>94.6</td><td>94.31</td><td>94.49</td><td>94.55</td></tr><tr><td>30</td><td>94.4</td><td>93.98</td><td>94.51</td><td>94.36</td></tr><tr><td>40</td><td>94.17</td><td>93.76</td><td>94.48</td><td>94.15</td></tr><tr><td>50</td><td>93.3</td><td>92.86</td><td>93.81</td><td>93.24</td></tr><tr><td rowspan="5">Group 11</td><td>10</td><td>94.83</td><td>94.48</td><td>91.89</td><td>94.78</td></tr><tr><td>20</td><td>94.72</td><td>94.32</td><td>93.49</td><td>94.62</td></tr><tr><td>30</td><td>94.63</td><td>94.18</td><td>94.29</td><td>94.49</td></tr><tr><td>40</td><td>94.37</td><td>93.97</td><td>94.56</td><td>94.15</td></tr><tr><td>50</td><td>93.39</td><td>92.91</td><td>94.31</td><td>93.41</td></tr></table>

# C.4 AGNOSTICISM TO IMPORTANCE METRICS

As we have alluded to in Section 3, we argue that the performance improvements of Intra-Fusion are agnostic with respect to the choice of the importance metric. That is, regardless of the expressiveness of the importance metric, Intra-Fusion is able to achieve a significant gain in accuracy. In order to further highlight this, we compare how Intra-Fusion performs when it is given random scores drawn from a uniform distribution as an importance metric.

As can be seen Table 5, Intra-Fusion still achieves significant increases in accuracy even when only random scores are available. We argue that the superiority of Intra-Fusion is inherent to the integration of the less important neuron pairings in the compressed network. Hence, making it truly agnostic to the importance metric used.

Table 5: Data-free results for ResNet50 on ImageNet. Group 0-10. Random pruning. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td>Random (%)</td><td>IF (Random)(%)</td><td> $\delta (\%)$ </td></tr><tr><td rowspan="4">Group 0</td><td>10</td><td>72.82</td><td>75.07</td><td>+2.25</td></tr><tr><td>20</td><td>67.97</td><td>73.96</td><td>+5.99</td></tr><tr><td>30</td><td>60.73</td><td>72.35</td><td>+11.62</td></tr><tr><td>40</td><td>53.94</td><td>70.18</td><td>+16.24</td></tr><tr><td rowspan="4">Group 1</td><td>10</td><td>75.2</td><td>75.49</td><td>+0.29</td></tr><tr><td>20</td><td>74.19</td><td>74.62</td><td>+0.43</td></tr><tr><td>30</td><td>72.61</td><td>73.56</td><td>+0.95</td></tr><tr><td>40</td><td>70.41</td><td>72.63</td><td>+2.22</td></tr><tr><td rowspan="4">Group 2</td><td>10</td><td>74.7</td><td>75.41</td><td>+0.71</td></tr><tr><td>20</td><td>73.33</td><td>74.55</td><td>+1.22</td></tr><tr><td>30</td><td>71.41</td><td>73.26</td><td>+1.85</td></tr><tr><td>40</td><td>69.5</td><td>71.46</td><td>+1.96</td></tr><tr><td rowspan="4">Group 3</td><td>10</td><td>75.06</td><td>75.45</td><td>+0.39</td></tr><tr><td>20</td><td>73.98</td><td>75.02</td><td>+1.04</td></tr><tr><td>30</td><td>73.47</td><td>74.21</td><td>+0.74</td></tr><tr><td>40</td><td>71.8</td><td>73.13</td><td>+1.33</td></tr><tr><td rowspan="4">Group 4</td><td>10</td><td>75.02</td><td>75.54</td><td>+0.52</td></tr><tr><td>20</td><td>74.09</td><td>75.09</td><td>+1.0</td></tr><tr><td>30</td><td>73.1</td><td>74.71</td><td>+1.61</td></tr><tr><td>40</td><td>72.49</td><td>73.49</td><td>+1.0</td></tr><tr><td rowspan="4">Group 5</td><td>10</td><td>75.03</td><td>75.42</td><td>+0.39</td></tr><tr><td>20</td><td>73.48</td><td>75.0</td><td>+1.52</td></tr><tr><td>30</td><td>71.39</td><td>74.16</td><td>+2.77</td></tr><tr><td>40</td><td>70.07</td><td>73.18</td><td>+3.11</td></tr><tr><td rowspan="4">Group 6</td><td>10</td><td>74.63</td><td>75.4</td><td>+0.77</td></tr><tr><td>20</td><td>72.44</td><td>75.16</td><td>+2.72</td></tr><tr><td>30</td><td>70.73</td><td>74.0</td><td>+3.27</td></tr><tr><td>40</td><td>69.83</td><td>72.75</td><td>+2.92</td></tr><tr><td rowspan="4">Group 7</td><td>10</td><td>72.92</td><td>74.93</td><td>+2.01</td></tr><tr><td>20</td><td>68.3</td><td>73.14</td><td>+4.84</td></tr><tr><td>30</td><td>58.97</td><td>70.27</td><td>+11.3</td></tr><tr><td>40</td><td>52.51</td><td>65.07</td><td>+12.56</td></tr><tr><td rowspan="4">Group 8</td><td>10</td><td>75.58</td><td>75.89</td><td>+0.31</td></tr><tr><td>20</td><td>74.79</td><td>75.56</td><td>+0.77</td></tr><tr><td>30</td><td>74.82</td><td>75.37</td><td>+0.55</td></tr><tr><td>40</td><td>74.26</td><td>74.81</td><td>+0.55</td></tr><tr><td rowspan="4">Group 9</td><td>10</td><td>75.56</td><td>75.79</td><td>+0.23</td></tr><tr><td>20</td><td>75.27</td><td>75.61</td><td>+0.34</td></tr><tr><td>30</td><td>74.84</td><td>75.53</td><td>+0.69</td></tr><tr><td>40</td><td>74.68</td><td>75.18</td><td>+0.5</td></tr><tr><td rowspan="4">Group 10</td><td>10</td><td>75.68</td><td>75.8</td><td>+0.12</td></tr><tr><td>20</td><td>75.43</td><td>75.61</td><td>+0.18</td></tr><tr><td>30</td><td>74.98</td><td>75.5</td><td>+0.52</td></tr><tr><td>40</td><td>74.57</td><td>74.89</td><td>+0.32</td></tr></table>

# D APPLICATION BEYOND PRUNING: FACTORIZING MODEL TRAINING

# D.1 RUNTIME COMPARISON OF SPLIT-DATA AND WHOLE-DATA APPROACH

![](images/82433c61464aec07753dc30a27e70bdac20f7cbd9e1183a51afcb059929a610f.jpg)

<details>
<summary>line</summary>

| Epochs | whole-data model | model a | model b |
| ------ | ---------------- | ------- | ------- |
| 0      | 40.0%            | 20.0%   | 10.0%   |
| 50     | 85.0%            | 80.0%   | 75.0%   |
| 100    | 88.0%            | 83.0%   | 82.0%   |
| 150    | 89.0%            | 84.0%   | 83.5%   |
| 200    | 89.5%            | 84.5%   | 84.0%   |
| 250    | 89.8%            | 84.8%   | 84.2%   |
| 300    | 90.0%            | 85.0%   | 84.5%   |
</details>

(a) VGG11 on CIFAR-10.

![](images/0b6a4d29b206a99e5beac70acc07ae7d7c01ec1ccf7f83e010f0f450bfcb4d7e.jpg)

<details>
<summary>line</summary>

| Epochs | whole-data model | model a | model b |
| ------ | ---------------- | ------- | ------- |
| 0      | 35.0%            | 35.0%   | 35.0%   |
| 50     | 85.0%            | 82.0%   | 84.0%   |
| 100    | 90.0%            | 86.0%   | 87.0%   |
| 150    | 90.5%            | 87.0%   | 88.0%   |
| 200    | 91.0%            | 87.5%   | 88.5%   |
| 250    | 91.0%            | 87.5%   | 88.5%   |
| 300    | 91.0%            | 87.5%   | 88.5%   |
</details>

(b) Resnet18 on CIFAR-10.

Figure 20: Training convergence speed: comparing whole-data model with the models trained on half the data.   
![](images/66c3c9d7b09cfb2adaa5e66564c03869473fb9a6815b11c6c0f4266314862628.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["untrained network architecture"] --> B["whole data set"]
    C["training on a data subset"] --> D["model a"]
    D --> E["pruning individually"]
    E --> F["fine-tuning on the whole data set"]
    F --> G["fusing the two models"]
    G --> H[" fine-tuning on the whole data set "]
    H --> I["PaF model"]
    
    J["untrained network architecture"] --> K["whole data set"]
    L["training on a data subset"] --> M["model a"]
    M --> N["pruning individually"]
    N --> O["fine-tuning on the whole data set"]
    O --> P["fusing the two models"]
    P --> Q[" fine-tuning on the whole data set "]
    Q --> R["PaF model"]
```
</details>

Figure 21: The PaF approach.

![](images/8ede528f488beab1456480bbe78389ee0ee97037dc1db17af271efb991c753cb.jpg)

<details>
<summary>bar_stacked</summary>

| Method      | Model Train Time | Fuse | Fine-Tune (whole dataset) | Iterative Prune (4x10 epochs) |
| ----------- | ---------------- | ---- | ------------------------- | ----------------------------- |
| Whole-Data  | 25               | 0    | 15                        | 5                             |
| FaP         | 15               | 0    | 8                         | 5                             |
| PaF         | 15               | 0    | 7                         | 5                             |
</details>

Figure 22: VGG11-bn on CIFAR-10. Relative runtimes for different approaches on an Nvidia-Gpu. Fuse time is so short in comparison to other steps that an extra marking was added for clarity. Speedup of the model train time: 1.81. Overall speedups: 1.31 (FaP), 1.42 (PaF). Further speedups are possible by optimizing the number of fine-tuning epochs at different stages. For models and datasets where the time for model training $T_{1}$ is much greater than time for fine-tuning $T_{2}$ ( $T_{1} \gg T_{2}$ ), speedups would show to be much more significant.

# D.2 PERFORMANCE COMPARISON: AFTER CONVERGENCE

To investigate the performance potential of the two approaches, in Figure 23 we show the performance of the PaF and FaP model when the fine-tuning at the intermediate steps is done until convergence. Although there is no clear performance equivalency between the classic whole-data model and the “factorized” model training (FaP and PaF), it is evident that depending on the model architecture and training dynamics the factorization has the potential to even outperform the standard approach. All experiments are done with an iterative pruning approach to recover a stronger performance at higher sparsities. Since data is available for fine-tuning and due to its superior performance, we use activation-based fusion with a sample size of 200.

![](images/1d983e0b348581dd7d66cb146de41749b055c88c5c4c78688f1bda089264476f.jpg)

<details>
<summary>line</summary>

| Percentage of Pruned Model Weights | PaF Model (Intra-Fusion) | Whole-Data Model | FaP Model (Intra-Fusion) |
| ---------------------------------- | ------------------------ | ---------------- | ------------------------ |
| 65%                                | 88.3%                    | 89.0%            | 88.7%                    |
| 70%                                | 88.2%                    | 88.8%            | 88.5%                    |
| 75%                                | 88.1%                    | 88.6%            | 88.3%                    |
| 80%                                | 88.0%                    | 88.4%            | 88.1%                    |
| 85%                                | 87.9%                    | 88.2%            | 87.9%                    |
| 90%                                | 87.5%                    | 87.6%            | 87.4%                    |
| 95%                                | 85.2%                    | 85.3%            | 85.1%                    |
</details>

(a) VGG11 on CIFAR-10.

![](images/a79f07fb1be5389dcd59c2aa9ca9d1ab33662449f381a6a5b51c5543a9118123.jpg)

<details>
<summary>line</summary>

| Percentage of Pruned Model Weights | PaF Model (Intra-Fusion) | Whole-Data Model | FaP Model (Intra-Fusion) |
| ---------------------------------- | ------------------------ | ---------------- | ------------------------ |
| 65%                                | 91.2%                    | 91.1%            | 91.3%                    |
| 70%                                | 91.0%                    | 90.9%            | 91.1%                    |
| 75%                                | 90.8%                    | 90.7%            | 90.9%                    |
| 80%                                | 90.6%                    | 90.5%            | 90.7%                    |
| 85%                                | 90.4%                    | 90.3%            | 90.5%                    |
| 90%                                | 89.8%                    | 89.7%            | 89.9%                    |
| 95%                                | 88.5%                    | 88.4%            | 88.6%                    |
</details>

(b) Resnet18 on CIFAR-10.   
Figure 23: Performance of PaF and FaP compared to the whole-data model across different sparsities. Here we use Intra-Fusion for pruning in PaF and FaP. An iterative pruning approach is chosen for all three models: 4 steps of each 10 epochs. PaF and FaP are fine-tuned for additional 80 epochs after the iterative pruning and after fusion. The whole-data model is fine-tuned for another $2 \times 80 = 160$ epochs after the iterative pruning. Experiments are done across four different seeds.

For reference, we also compare the performance of the PaF and FaP models using regular pruning (instead of Intra-Fusion) in Figure 24.

![](images/e715542e7c84538496f6de025e060fa8d5d6784899ad9a1d4ef459351b0cf3a8.jpg)

<details>
<summary>line</summary>

| Percentage of Pruned Model Weights | PaF Model (Default-Pruning) | Whole-Data Model | FaP Model (Default-Pruning) |
| ---------------------------------- | --------------------------- | ---------------- | --------------------------- |
| 65%                                | 88.2%                       | 89.0%            | 88.3%                       |
| 70%                                | 87.8%                       | 88.8%            | 88.0%                       |
| 75%                                | 87.5%                       | 88.5%            | 87.7%                       |
| 80%                                | 87.2%                       | 88.2%            | 87.4%                       |
| 85%                                | 86.8%                       | 87.9%            | 87.1%                       |
| 90%                                | 86.2%                       | 87.5%            | 86.6%                       |
| 95%                                | 83.0%                       | 85.2%            | 84.5%                       |
</details>

(a) VGG11 on CIFAR-10.

![](images/7be0143bec9b3916a1c0038106a577f66763e41077b3334413ca3fc207e940ee.jpg)

<details>
<summary>line</summary>

| Percentage of Pruned Model Weights | PaF Model (Default-Pruning) | Whole-Data Model | FaP Model (Default-Pruning) |
| ---------------------------------- | --------------------------- | ---------------- | --------------------------- |
| 65%                                | 90.8                        | 91.0             | 90.9                        |
| 70%                                | 90.7                        | 90.9             | 90.8                        |
| 75%                                | 90.6                        | 90.8             | 90.7                        |
| 80%                                | 90.5                        | 90.7             | 90.6                        |
| 85%                                | 90.4                        | 90.6             | 90.5                        |
| 90%                                | 89.8                        | 90.2             | 89.7                        |
| 95%                                | 87.5                        | 88.5             | 87.8                        |
</details>

(b) Resnet18 on CIFAR-10.   
Figure 24: Performance of PaF and FaP compared to the whole-data model across different sparsities. Here we use regular pruning (instead of Intra-Fusion) for PaF and FaP. An iterative pruning approach is chosen for all three models: 4 steps of each 10 epochs. PaF and FaP are trained for additional 80 epochs after the iterative pruning and after fusion. The whole-data model is trained for another $2 \times 80 = 160$ epochs after the iterative pruning.

# D.3 PERFORMANCE COMPARISON: VARYING FINE-TUNING

Since the performance of the PaF approach depends on the amount of fine-tuning that is available at the intermediate steps, we also explore how the performance difference to the whole-data model develops with varying amounts of fine-tuning. In this setting, the PaF and FaP models gain a theoretical 2x speedup in the training process and take the same time in the post-processing as the whole-data model. In Figure 25 (using Intra-Fusion) and Figure 26 (using conventional pruning) for multiple sparsities, we vary the amount of retraining that is available to the models. This means here we do not get the converged performance of PaF and Fap (only converged at 80 fine-tuning epochs).

![](images/1b7c5e56c37e34abac90c2a2110d8df5a34d2c472a458c623dfb13371d6ab3f0.jpg)

<details>
<summary>line</summary>

| Retrained Epochs | Sparsity: 0.4 | Sparsity: 0.5 | Sparsity: 0.6 | Sparsity: 0.7 | Sparsity: 0.8 |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 20.0             | -1.5%         | -1.0%         | -1.0%         | -0.5%         | -1.0%         |
| 30.0             | -1.5%         | -1.0%         | -1.0%         | -0.5%         | -1.0%         |
| 40.0             | -1.0%         | -0.5%         | -0.5%         | 0.5%          | -0.5%         |
| 60.0             | -1.0%         | -0.5%         | -0.5%         | 0.5%          | -0.5%         |
| 80.0             | -1.0%         | -0.5%         | -0.5%         | 0.5%          | -0.5%         |
</details>

(a) VGG11 on CIFAR-10.

![](images/d564298563f684fa4ee34c5a6bc01e91071138818b10a27d9d35e701e25db5e2.jpg)

<details>
<summary>line</summary>

| Retrained Epochs | Sparsity: 0.4 | Sparsity: 0.5 | Sparsity: 0.6 | Sparsity: 0.7 | Sparsity: 0.8 |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 20.0             | -1.0%         | -1.0%         | -1.0%         | -1.0%         | -1.0%         |
| 30.0             | -1.2%         | -1.1%         | -1.1%         | -0.8%         | -1.0%         |
| 40.0             | -0.5%         | -0.4%         | 0.0%          | -0.2%         | -0.3%         |
| 60.0             | 0.0%          | 0.1%          | 0.2%          | 0.1%          | 0.2%          |
| 80.0             | 0.3%          | 0.4%          | 0.5%          | 0.4%          | 0.3%          |
</details>

(b) Resnet18 on CIFAR-10.

Figure 25: Comparing the development of performance difference of PaF and the whole-data model when varying the total amount of retraining that is done. PaF, FaP and the whole data model always get the same amount of retraining. Here Intra-Fusion is used in the context of PaF and FaP. Sparsity here is the node sparsity.   
![](images/085454cee5793714c3eaa3b59807fb46f81dff6ae590a6da70fcd0835130a6cf.jpg)

<details>
<summary>line</summary>

| Retrained Epochs | Sparsity: 0.4 | Sparsity: 0.5 | Sparsity: 0.6 | Sparsity: 0.7 | Sparsity: 0.8 |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 20.0             | -1.2%         | -1.3%         | -1.4%         | -1.5%         | -1.6%         |
| 30.0             | -1.5%         | -1.6%         | -1.7%         | -1.8%         | -2.0%         |
| 40.0             | -1.0%         | -1.1%         | -1.2%         | -0.5%         | -1.3%         |
| 60.0             | -0.8%         | -0.9%         | -1.0%         | -0.4%         | -1.1%         |
| 80.0             | -0.6%         | -0.7%         | -0.8%         | -0.5%         | -0.6%         |
</details>

(a) VGG11 on CIFAR-10.

![](images/5e5a704e039d8e10eb8c4fefd0ac99baa630ab3694d783354e341dad46f2b7ba.jpg)

<details>
<summary>line</summary>

| Retrained Epochs | Sparsity: 0.4 | Sparsity: 0.5 | Sparsity: 0.6 | Sparsity: 0.7 | Sparsity: 0.8 |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 20.0             | -1.3%         | -1.2%         | -1.3%         | -1.4%         | -1.7%         |
| 30.0             | -1.1%         | -1.1%         | -1.1%         | -1.2%         | -1.4%         |
| 40.0             | -0.8%         | -0.9%         | -0.7%         | -0.8%         | -1.2%         |
| 60.0             | -0.5%         | -0.6%         | -0.5%         | -0.7%         | -1.0%         |
| 80.0             | 0.0%          | 0.1%          | 0.2%          | 0.3%          | -0.8%         |
</details>

(b) Resnet18 on CIFAR-10.   
Figure 26: Comparing the development of performance difference of PaF and the whole-data model when varying the total amount of retraining that is done. PaF, FaP and the whole data model always get the same amount of retraining. Here regular pruning (instead of Intra-Fusion) is used in the context of PaF and FaP. Sparsity here is the node sparsity.

# D.4 K-FOLD SPLIT-DATA

It is obvious that the 50/50 split of the dataset that we considered in the regular split-data experiments can also be interpreted as a k-fold style approach with $k = 2$ , yielding $\binom{2}{1} = 2$ different models ("model a" and "model b" in Figure 27a).

To explore the potential of the k-fold approach we also evaluated the performance for the next even choice of $k$ , namely $k = 4$ . This already yields $\binom{4}{2} = 6$ different models that are be combined in PaF and FaP (see models "a" to "f" in Figure 27b).

# D.4.1 PERFORMANCE COMPARISON ACROSS DIFFERENT K

The performance of PaF and FaP based on the $k = 2$ and $k = 4$ can be compared in Figure 28. For the VGG11-BN we observe performance improvements of up to $1\%$ . Here is important to note

![](images/dc28926a1efca6ddcda2bcd7b46ddeffc2f4d86d3fc00bf7e53186cd4cda5806.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["dataset 1"] --> B["model a"]
    C["dataset 2"] --> D["model b"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
```
</details>

(a) 2-Fold

![](images/a7f83a589df7c79d926a3b75a2cb8514da27eb9bace098058d4ff935b607c34e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Dataset"] --> B["subset 1"]
    A --> C["subset 2"]
    A --> D["subset 3"]
    A --> E["subset 4"]
    F["model f"] --> A
    G["model a"] --> A
    H["model b"] --> A
    I["model c"] --> A
    J["model d"] --> A
    K["model e"] --> A
    B <--> C
    C <--> D
    D <--> E
    E <--> F
```
</details>

(b) 4-Fold   
Figure 27: Visualization of k-fold data splits at different k.

that across all measured sparsities the 4-Fold approach always outperforms the 2-Fold approach and yields a very competitive performance when compared to the benchmark “Whole Data Model”. For the Resnet18 we seem to not make any improvements in performance by extending from two to six combined models.

![](images/214349bae17e1f288c3130fce668b51aa3b190946b7a8d7e913d4ef9b264ae12.jpg)

<details>
<summary>line</summary>

| Percentage of Pruned Model Weights | FaP (2-Fold) | Whole Data Model | PaF (2-Fold) | PaF (4-Fold) | PaF (4-Fold) |
| ----------------------------------- | ------------ | ---------------- | ------------ | ------------ | ------------ |
| 65%                                 | 88.5%        | 88.8%            | 88.1%        | 88.9%        | 88.7%        |
| 75%                                 | 88.3%        | 88.6%            | 88.0%        | 88.7%        | 88.5%        |
| 85%                                 | 87.7%        | 88.4%            | 87.9%        | 88.5%        | 88.3%        |
| 90%                                 | 87.5%        | 87.0%            | 87.6%        | 87.8%        | 87.9%        |
| 95%                                 | 85.0%        | 85.5%            | 85.2%        | 85.3%        | 85.6%        |
</details>

(a) VGG11-BN on CIFAR-10.

![](images/db8a07237ae2c25915b3940cf2a1991561b612b097573c33fc86556958913c61.jpg)

<details>
<summary>line</summary>

| Percentage of Pruned Model Weights | FaP (2-Fold) | Whole Data Model | PaF (2-Fold) | PaF (4-Fold) | PaF (4-Fold) |
| ---------------------------------- | ------------ | ---------------- | ------------ | ------------ | ------------ |
| 65%                                | 91.3%        | 91.0%            | 91.4%        | 91.2%        | 91.1%        |
| 70%                                | 91.2%        | 90.7%            | 91.3%        | 91.2%        | 91.1%        |
| 75%                                | 91.1%        | 90.7%            | 91.2%        | 91.2%        | 91.0%        |
| 80%                                | 91.0%        | 90.7%            | 91.0%        | 91.0%        | 90.8%        |
| 85%                                | 91.2%        | 90.7%            | 90.8%        | 90.8%        | 90.7%        |
| 90%                                | 90.5%        | 89.8%            | 89.5%        | 89.5%        | 89.5%        |
| 95%                                | 88.5%        | 88.3%            | 88.7%        | 88.7%        | 88.5%        |
</details>

(b) Resnet18 on CIFAR-10.   
Figure 28: Performance comparison of k-fold data splits at different k.

# D.5 EXTENSIONS FOR FUTURE WORK

We leave it for further research to explore less drastic splits of the dataset. We believe that this will lead to a better fine-tuning/accuracy trade-off - especially at lower sparsities. For example, the dataset could be split into overlapping sets that for example make up 60% or 70% of the original dataset.

# D.6 PERFORMANCE OF MODELS USED

To also get a feeling for how uncompetitive the performance of the individual split-data models is (since they each were only trained on one half of the data) before deploying our PaF and FaP approach and fine-tuning on the whole dataset we include the model performance figures (across different seeds) in Tables 6 and 8. For each seed a different split of the dataset is generated which the split-data models are trained upon

Table 6: Performance of the used 2-Fold split-data models. 

<table><tr><td>Model</td><td>Training Data</td><td>Seed</td><td>Accuracy (%)</td></tr><tr><td rowspan="8">VGG11-BN</td><td rowspan="4">1. data subset</td><td>A</td><td>85.74</td></tr><tr><td>B</td><td>85.54</td></tr><tr><td>C</td><td>85.80</td></tr><tr><td>D</td><td>85.33</td></tr><tr><td rowspan="4">2. data subset</td><td>A</td><td>84.50</td></tr><tr><td>B</td><td>85.32</td></tr><tr><td>C</td><td>84.99</td></tr><tr><td>D</td><td>86.09</td></tr><tr><td rowspan="8">Resnet18</td><td rowspan="4">1. data subset</td><td>A</td><td>88.46</td></tr><tr><td>B</td><td>87.44</td></tr><tr><td>C</td><td>87.75</td></tr><tr><td>D</td><td>88.17</td></tr><tr><td rowspan="4">2. data subset</td><td>A</td><td>87.71</td></tr><tr><td>B</td><td>88.31</td></tr><tr><td>C</td><td>87.65</td></tr><tr><td>D</td><td>88.23</td></tr></table>

Table 7: Performance of the used 4-Fold split-data models. Single seed. 

<table><tr><td>Model</td><td>Training Data</td><td>Accuracy (%)</td></tr><tr><td rowspan="6">VGG11-BN</td><td>1. data subset</td><td>85.36</td></tr><tr><td>2. data subset</td><td>86.00</td></tr><tr><td>3. data subset</td><td>85.56</td></tr><tr><td>4. data subset</td><td>85.22</td></tr><tr><td>5. data subset</td><td>83.96</td></tr><tr><td>6. data subset</td><td>85.69</td></tr><tr><td rowspan="6">Resnet18</td><td>1. data subset</td><td>87.87</td></tr><tr><td>2. data subset</td><td>87.67</td></tr><tr><td>3. data subset</td><td>87.99</td></tr><tr><td>4. data subset</td><td>87.91</td></tr><tr><td>5. data subset</td><td>88.37</td></tr><tr><td>6. data subset</td><td>87.75</td></tr></table>

Table 8: Performance of the used whole-data models. 

<table><tr><td>Model</td><td>Seed</td><td>Accuracy (%)</td></tr><tr><td rowspan="4">VGG11-BN</td><td>A</td><td>89.28</td></tr><tr><td>B</td><td>89.47</td></tr><tr><td>C</td><td>89.04</td></tr><tr><td>D</td><td>89.21</td></tr><tr><td rowspan="4">Resnet18</td><td>A</td><td>91.47</td></tr><tr><td>B</td><td>92.08</td></tr><tr><td>C</td><td>91.36</td></tr><tr><td>D</td><td>91.64</td></tr></table>

# E EMPIRICAL RESULTS

# E.1 TERMINOLOGY

In Table 9, we show how Neuron Sparsity (applied to all groups in the model) translates to Weight Sparsity.

Table 9: Neuron Sparsity to Weight Sparsity Translation. 

<table><tr><td>Neuron Sparsity</td><td>VGG11_bn #Parameters</td><td>Resnet18 #Parameters</td><td>Weight Sparsity</td></tr><tr><td>Original (0%)</td><td>9,753,674</td><td>11,164,352</td><td>-</td></tr><tr><td>40%</td><td>3,505,301</td><td>4,003,668</td><td>~64%</td></tr><tr><td>50%</td><td>2,441,770</td><td>2,792,800</td><td>~75%</td></tr><tr><td>60%</td><td>1,551,390</td><td>1,772,832</td><td>~84%</td></tr><tr><td>70%</td><td>871,988</td><td>994,402</td><td>~91%</td></tr><tr><td>80%</td><td>388,770</td><td>442,308</td><td>~96%</td></tr></table>

# E.2 DATA-FREE EXPERIMENTS

Here, we provide a full list of results for the data-free experiments. The indices again indicate how close a group is to the output of the model, i.e. Group 0 is the last group of the network. Moreover, we color-code every entry where the absolute difference between the default and Intra-Fused model is greater than 0.5%.

Table 10: Data-free results for VGG11-BN on CIFAR-10. Group 0-5. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 0</td><td>10</td><td>89.53</td><td>89.47</td><td>-0.06</td><td>89.47</td><td>89.46</td><td>-0.01</td><td>89.53</td><td>89.52</td><td>-0.01</td></tr><tr><td>20</td><td>89.38</td><td>89.41</td><td>+0.03</td><td>89.40</td><td>89.55</td><td>+0.15</td><td>89.54</td><td>89.49</td><td>-0.05</td></tr><tr><td>30</td><td>89.30</td><td>89.31</td><td>+0.01</td><td>89.27</td><td>89.46</td><td>+0.19</td><td>89.44</td><td>89.56</td><td>+0.12</td></tr><tr><td>40</td><td>88.86</td><td>89.21</td><td>+0.35</td><td>88.67</td><td>89.45</td><td>+0.78</td><td>89.14</td><td>89.50</td><td>+0.36</td></tr><tr><td>50</td><td>87.91</td><td>89.07</td><td>+1.17</td><td>87.55</td><td>89.39</td><td>+1.84</td><td>88.85</td><td>89.46</td><td>+0.60</td></tr><tr><td>60</td><td>87.42</td><td>89.36</td><td>+1.94</td><td>85.40</td><td>89.21</td><td>+3.81</td><td>88.88</td><td>89.41</td><td>+0.52</td></tr><tr><td>70</td><td>86.78</td><td>89.29</td><td>+2.51</td><td>80.82</td><td>88.93</td><td>+8.11</td><td>88.43</td><td>89.25</td><td>+0.82</td></tr><tr><td rowspan="7">Group 1</td><td>10</td><td>89.31</td><td>89.52</td><td>+0.21</td><td>89.45</td><td>89.53</td><td>+0.08</td><td>89.22</td><td>89.44</td><td>+0.22</td></tr><tr><td>20</td><td>88.99</td><td>89.47</td><td>+0.47</td><td>89.16</td><td>89.46</td><td>+0.30</td><td>88.81</td><td>89.41</td><td>+0.60</td></tr><tr><td>30</td><td>88.74</td><td>89.15</td><td>+0.42</td><td>88.76</td><td>89.24</td><td>+0.48</td><td>87.25</td><td>89.28</td><td>+2.03</td></tr><tr><td>40</td><td>87.68</td><td>88.84</td><td>+1.17</td><td>88.51</td><td>88.92</td><td>+0.42</td><td>84.88</td><td>88.84</td><td>+3.97</td></tr><tr><td>50</td><td>85.44</td><td>88.06</td><td>+2.62</td><td>87.49</td><td>88.23</td><td>+0.74</td><td>81.14</td><td>88.67</td><td>+7.53</td></tr><tr><td>60</td><td>77.81</td><td>87.58</td><td>+9.77</td><td>83.55</td><td>87.53</td><td>+3.98</td><td>68.12</td><td>88.25</td><td>+20.13</td></tr><tr><td>70</td><td>60.63</td><td>86.82</td><td>+26.19</td><td>68.88</td><td>86.60</td><td>+17.72</td><td>44.23</td><td>88.02</td><td>+43.79</td></tr><tr><td rowspan="7">Group 2</td><td>10</td><td>87.67</td><td>89.41</td><td>+1.74</td><td>89.44</td><td>89.37</td><td>-0.07</td><td>88.82</td><td>89.50</td><td>+0.68</td></tr><tr><td>20</td><td>85.17</td><td>88.92</td><td>+3.76</td><td>88.99</td><td>89.00</td><td>+0.01</td><td>86.89</td><td>89.21</td><td>+2.32</td></tr><tr><td>30</td><td>80.41</td><td>88.26</td><td>+7.85</td><td>87.34</td><td>88.46</td><td>+1.12</td><td>82.80</td><td>88.65</td><td>+5.84</td></tr><tr><td>40</td><td>76.07</td><td>87.18</td><td>+11.12</td><td>84.61</td><td>87.36</td><td>+2.75</td><td>77.29</td><td>87.98</td><td>+10.69</td></tr><tr><td>50</td><td>68.41</td><td>85.37</td><td>+16.96</td><td>78.94</td><td>85.11</td><td>+6.17</td><td>70.82</td><td>86.55</td><td>+15.73</td></tr><tr><td>60</td><td>60.10</td><td>83.51</td><td>+23.42</td><td>74.10</td><td>83.70</td><td>+9.60</td><td>48.04</td><td>84.21</td><td>+36.16</td></tr><tr><td>70</td><td>37.89</td><td>80.11</td><td>+42.23</td><td>65.62</td><td>80.86</td><td>+15.25</td><td>28.00</td><td>80.80</td><td>+52.81</td></tr><tr><td rowspan="7">Group 3</td><td>10</td><td>89.11</td><td>89.28</td><td>+0.17</td><td>89.30</td><td>89.35</td><td>+0.05</td><td>89.41</td><td>89.42</td><td>+0.01</td></tr><tr><td>20</td><td>88.49</td><td>88.84</td><td>+0.35</td><td>89.06</td><td>88.76</td><td>-0.31</td><td>88.70</td><td>89.30</td><td>+0.60</td></tr><tr><td>30</td><td>86.76</td><td>87.79</td><td>+1.03</td><td>88.50</td><td>88.06</td><td>-0.44</td><td>87.79</td><td>88.99</td><td>+1.21</td></tr><tr><td>40</td><td>85.56</td><td>87.45</td><td>+1.89</td><td>87.47</td><td>87.12</td><td>-0.35</td><td>86.73</td><td>88.47</td><td>+1.74</td></tr><tr><td>50</td><td>79.49</td><td>86.68</td><td>+7.19</td><td>85.52</td><td>86.06</td><td>+0.53</td><td>83.51</td><td>87.93</td><td>+4.41</td></tr><tr><td>60</td><td>72.23</td><td>85.58</td><td>+13.35</td><td>82.58</td><td>86.15</td><td>+3.57</td><td>76.07</td><td>86.94</td><td>+10.87</td></tr><tr><td>70</td><td>54.86</td><td>84.10</td><td>+29.24</td><td>71.07</td><td>85.10</td><td>+14.02</td><td>59.39</td><td>85.58</td><td>+26.19</td></tr><tr><td rowspan="7">Group 4</td><td>10</td><td>89.13</td><td>89.20</td><td>+0.07</td><td>89.16</td><td>89.16</td><td>+0.00</td><td>88.55</td><td>89.38</td><td>+0.83</td></tr><tr><td>20</td><td>88.79</td><td>88.45</td><td>-0.34</td><td>88.29</td><td>87.59</td><td>-0.70</td><td>84.29</td><td>88.40</td><td>+4.11</td></tr><tr><td>30</td><td>87.06</td><td>86.64</td><td>-0.43</td><td>86.31</td><td>86.74</td><td>+0.43</td><td>82.46</td><td>88.04</td><td>+5.59</td></tr><tr><td>40</td><td>82.33</td><td>85.04</td><td>+2.71</td><td>81.69</td><td>84.52</td><td>+2.84</td><td>80.17</td><td>87.10</td><td>+6.93</td></tr><tr><td>50</td><td>65.26</td><td>80.29</td><td>+15.03</td><td>73.31</td><td>82.81</td><td>+9.50</td><td>74.15</td><td>83.98</td><td>+9.83</td></tr><tr><td>60</td><td>51.82</td><td>80.24</td><td>+28.42</td><td>57.40</td><td>80.32</td><td>+22.92</td><td>68.52</td><td>82.74</td><td>+14.22</td></tr><tr><td>70</td><td>47.16</td><td>79.02</td><td>+31.85</td><td>37.24</td><td>76.56</td><td>+39.32</td><td>46.77</td><td>77.39</td><td>+30.63</td></tr><tr><td rowspan="7">Group 5</td><td>10</td><td>89.17</td><td>88.85</td><td>-0.32</td><td>89.27</td><td>89.00</td><td>-0.27</td><td>87.97</td><td>89.25</td><td>+1.28</td></tr><tr><td>20</td><td>87.76</td><td>87.65</td><td>-0.11</td><td>87.86</td><td>88.03</td><td>+0.18</td><td>85.45</td><td>88.59</td><td>+3.13</td></tr><tr><td>30</td><td>85.67</td><td>86.56</td><td>+0.89</td><td>86.48</td><td>87.58</td><td>+1.10</td><td>84.58</td><td>88.18</td><td>+3.60</td></tr><tr><td>40</td><td>83.49</td><td>84.92</td><td>+1.42</td><td>83.72</td><td>86.05</td><td>+2.32</td><td>82.52</td><td>85.87</td><td>+3.35</td></tr><tr><td>50</td><td>79.45</td><td>78.75</td><td>-0.70</td><td>81.26</td><td>80.57</td><td>-0.69</td><td>76.41</td><td>82.30</td><td>+5.88</td></tr><tr><td>60</td><td>74.63</td><td>75.26</td><td>+0.62</td><td>72.66</td><td>77.44</td><td>+4.79</td><td>67.12</td><td>78.68</td><td>+11.56</td></tr><tr><td>70</td><td>62.81</td><td>67.15</td><td>+4.34</td><td>59.04</td><td>68.87</td><td>+9.83</td><td>49.90</td><td>69.93</td><td>+20.03</td></tr></table>

Table 11: Data-free results for VGG11-BN on CIFAR-10. Group 6-7. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 6</td><td>10</td><td>88.59</td><td>88.76</td><td>+0.17</td><td>88.88</td><td>88.39</td><td>-0.49</td><td>87.38</td><td>89.00</td><td>+1.62</td></tr><tr><td>20</td><td>87.05</td><td>87.78</td><td>+0.73</td><td>87.32</td><td>87.69</td><td>+0.37</td><td>83.39</td><td>88.26</td><td>+4.88</td></tr><tr><td>30</td><td>83.59</td><td>86.56</td><td>+2.97</td><td>80.06</td><td>84.90</td><td>+4.84</td><td>79.21</td><td>86.42</td><td>+7.21</td></tr><tr><td>40</td><td>78.80</td><td>82.72</td><td>+3.93</td><td>57.45</td><td>82.03</td><td>+24.58</td><td>59.87</td><td>84.33</td><td>+24.46</td></tr><tr><td>50</td><td>59.99</td><td>71.37</td><td>+11.38</td><td>48.39</td><td>70.92</td><td>+22.53</td><td>40.62</td><td>76.15</td><td>+35.52</td></tr><tr><td>60</td><td>35.62</td><td>64.56</td><td>+28.94</td><td>32.46</td><td>66.00</td><td>+33.54</td><td>28.70</td><td>66.78</td><td>+38.08</td></tr><tr><td>70</td><td>16.66</td><td>51.69</td><td>+35.03</td><td>21.35</td><td>51.91</td><td>+30.56</td><td>20.41</td><td>49.87</td><td>+29.46</td></tr><tr><td rowspan="7">Group 7</td><td>10</td><td>89.10</td><td>88.00</td><td>-1.10</td><td>89.41</td><td>86.48</td><td>-2.93</td><td>84.84</td><td>88.83</td><td>+3.99</td></tr><tr><td>20</td><td>88.71</td><td>86.51</td><td>-2.20</td><td>89.17</td><td>84.48</td><td>-4.69</td><td>57.63</td><td>84.20</td><td>+26.56</td></tr><tr><td>30</td><td>88.22</td><td>79.10</td><td>-9.12</td><td>88.19</td><td>78.35</td><td>-9.84</td><td>50.06</td><td>83.44</td><td>+33.38</td></tr><tr><td>40</td><td>87.19</td><td>68.28</td><td>-18.92</td><td>87.23</td><td>70.19</td><td>-17.04</td><td>41.91</td><td>80.88</td><td>+38.97</td></tr><tr><td>50</td><td>84.23</td><td>63.05</td><td>-21.17</td><td>83.84</td><td>47.18</td><td>-36.66</td><td>34.23</td><td>75.74</td><td>+41.52</td></tr><tr><td>60</td><td>65.06</td><td>57.42</td><td>-7.64</td><td>75.65</td><td>47.05</td><td>-28.60</td><td>28.81</td><td>69.65</td><td>+40.84</td></tr><tr><td>70</td><td>48.07</td><td>49.67</td><td>+1.60</td><td>48.08</td><td>41.42</td><td>-6.67</td><td>23.30</td><td>60.77</td><td>+37.47</td></tr></table>

Table 12: Data-free results for a ResNet18 on CIFAR-10. Group 0-5. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 0</td><td>10</td><td>94.89</td><td>94.91</td><td>+0.02</td><td>94.88</td><td>94.84</td><td>-0.04</td><td>94.71</td><td>94.70</td><td>-0.01</td></tr><tr><td>20</td><td>94.81</td><td>94.82</td><td>+0.01</td><td>94.81</td><td>94.78</td><td>-0.03</td><td>94.46</td><td>94.64</td><td>+0.17</td></tr><tr><td>30</td><td>94.75</td><td>94.79</td><td>+0.04</td><td>94.77</td><td>94.66</td><td>-0.11</td><td>94.23</td><td>94.55</td><td>+0.32</td></tr><tr><td>40</td><td>94.50</td><td>94.68</td><td>+0.18</td><td>94.59</td><td>94.49</td><td>-0.10</td><td>93.56</td><td>94.56</td><td>+1.00</td></tr><tr><td>50</td><td>94.35</td><td>94.56</td><td>+0.21</td><td>94.22</td><td>94.43</td><td>+0.21</td><td>92.83</td><td>94.51</td><td>+1.68</td></tr><tr><td>60</td><td>93.81</td><td>94.39</td><td>+0.58</td><td>92.37</td><td>94.10</td><td>+1.73</td><td>89.54</td><td>94.16</td><td>+4.62</td></tr><tr><td>70</td><td>92.90</td><td>93.98</td><td>+1.08</td><td>87.81</td><td>93.82</td><td>+6.01</td><td>74.56</td><td>93.82</td><td>+19.26</td></tr><tr><td rowspan="7">Group 1</td><td>10</td><td>94.86</td><td>94.93</td><td>+0.07</td><td>94.90</td><td>94.85</td><td>-0.05</td><td>94.81</td><td>94.76</td><td>-0.05</td></tr><tr><td>20</td><td>94.88</td><td>94.88</td><td>+0.00</td><td>94.88</td><td>94.87</td><td>-0.01</td><td>94.80</td><td>94.80</td><td>+0.00</td></tr><tr><td>30</td><td>94.86</td><td>94.89</td><td>+0.04</td><td>94.72</td><td>94.88</td><td>+0.16</td><td>94.70</td><td>94.83</td><td>+0.13</td></tr><tr><td>40</td><td>94.84</td><td>94.82</td><td>-0.02</td><td>94.65</td><td>94.89</td><td>+0.24</td><td>94.64</td><td>94.74</td><td>+0.09</td></tr><tr><td>50</td><td>94.72</td><td>94.86</td><td>+0.14</td><td>94.36</td><td>94.86</td><td>+0.49</td><td>94.38</td><td>94.78</td><td>+0.40</td></tr><tr><td>60</td><td>94.61</td><td>94.85</td><td>+0.24</td><td>94.18</td><td>94.80</td><td>+0.62</td><td>94.05</td><td>94.70</td><td>+0.65</td></tr><tr><td>70</td><td>94.09</td><td>94.83</td><td>+0.74</td><td>93.13</td><td>94.62</td><td>+1.49</td><td>91.54</td><td>94.65</td><td>+3.11</td></tr><tr><td rowspan="7">Group 2</td><td>10</td><td>94.87</td><td>94.91</td><td>+0.04</td><td>94.86</td><td>94.85</td><td>-0.01</td><td>94.71</td><td>94.84</td><td>+0.13</td></tr><tr><td>20</td><td>94.75</td><td>94.87</td><td>+0.12</td><td>94.87</td><td>94.75</td><td>-0.12</td><td>94.67</td><td>94.79</td><td>+0.11</td></tr><tr><td>30</td><td>94.58</td><td>94.86</td><td>+0.28</td><td>94.72</td><td>94.75</td><td>+0.03</td><td>94.49</td><td>94.73</td><td>+0.25</td></tr><tr><td>40</td><td>94.34</td><td>94.82</td><td>+0.48</td><td>94.48</td><td>94.74</td><td>+0.26</td><td>94.19</td><td>94.70</td><td>+0.51</td></tr><tr><td>50</td><td>94.10</td><td>94.76</td><td>+0.66</td><td>94.06</td><td>94.59</td><td>+0.53</td><td>93.77</td><td>94.57</td><td>+0.80</td></tr><tr><td>60</td><td>93.50</td><td>94.61</td><td>+1.11</td><td>93.67</td><td>94.30</td><td>+0.63</td><td>93.35</td><td>94.59</td><td>+1.24</td></tr><tr><td>70</td><td>92.41</td><td>94.41</td><td>+2.00</td><td>92.90</td><td>94.07</td><td>+1.17</td><td>92.64</td><td>94.33</td><td>+1.69</td></tr><tr><td rowspan="7">Group 3</td><td>10</td><td>94.25</td><td>94.56</td><td>+0.31</td><td>94.20</td><td>94.52</td><td>+0.32</td><td>93.96</td><td>94.46</td><td>+0.49</td></tr><tr><td>20</td><td>93.22</td><td>94.09</td><td>+0.87</td><td>92.95</td><td>93.93</td><td>+0.98</td><td>93.17</td><td>93.99</td><td>+0.82</td></tr><tr><td>30</td><td>91.33</td><td>93.59</td><td>+2.26</td><td>91.93</td><td>93.43</td><td>+1.50</td><td>91.85</td><td>93.44</td><td>+1.59</td></tr><tr><td>40</td><td>87.75</td><td>92.46</td><td>+4.71</td><td>89.17</td><td>92.58</td><td>+3.41</td><td>90.06</td><td>92.61</td><td>+2.56</td></tr><tr><td>50</td><td>81.91</td><td>91.25</td><td>+9.34</td><td>83.69</td><td>91.67</td><td>+7.98</td><td>85.43</td><td>90.45</td><td>+5.01</td></tr><tr><td>60</td><td>75.73</td><td>88.32</td><td>+12.59</td><td>76.30</td><td>88.42</td><td>+12.12</td><td>80.52</td><td>88.75</td><td>+8.23</td></tr><tr><td>70</td><td>58.93</td><td>80.70</td><td>+21.77</td><td>57.86</td><td>79.97</td><td>+22.11</td><td>62.23</td><td>80.91</td><td>+18.68</td></tr><tr><td rowspan="7">Group 4</td><td>10</td><td>94.85</td><td>94.88</td><td>+0.03</td><td>94.78</td><td>10.10</td><td>-84.68</td><td>94.60</td><td>94.65</td><td>+0.05</td></tr><tr><td>20</td><td>94.69</td><td>94.75</td><td>+0.06</td><td>94.44</td><td>10.10</td><td>-84.35</td><td>94.27</td><td>94.49</td><td>+0.22</td></tr><tr><td>30</td><td>94.43</td><td>94.54</td><td>+0.10</td><td>94.17</td><td>10.10</td><td>-84.07</td><td>93.90</td><td>94.39</td><td>+0.49</td></tr><tr><td>40</td><td>94.18</td><td>94.30</td><td>+0.12</td><td>93.91</td><td>10.10</td><td>-83.81</td><td>93.43</td><td>94.13</td><td>+0.69</td></tr><tr><td>50</td><td>93.71</td><td>94.02</td><td>+0.31</td><td>93.57</td><td>10.10</td><td>-83.48</td><td>92.87</td><td>93.92</td><td>+1.05</td></tr><tr><td>60</td><td>93.11</td><td>93.76</td><td>+0.65</td><td>92.67</td><td>10.10</td><td>-82.58</td><td>92.38</td><td>93.38</td><td>+1.00</td></tr><tr><td>70</td><td>91.96</td><td>92.70</td><td>+0.74</td><td>91.81</td><td>10.10</td><td>-81.71</td><td>91.65</td><td>92.62</td><td>+0.97</td></tr><tr><td rowspan="7">Group 5</td><td>10</td><td>94.73</td><td>94.84</td><td>+0.11</td><td>94.52</td><td>94.77</td><td>+0.25</td><td>94.38</td><td>94.64</td><td>+0.26</td></tr><tr><td>20</td><td>94.37</td><td>94.67</td><td>+0.31</td><td>94.04</td><td>94.63</td><td>+0.59</td><td>93.95</td><td>94.59</td><td>+0.64</td></tr><tr><td>30</td><td>93.84</td><td>94.58</td><td>+0.74</td><td>93.68</td><td>94.50</td><td>+0.82</td><td>93.61</td><td>94.33</td><td>+0.72</td></tr><tr><td>40</td><td>93.03</td><td>94.40</td><td>+1.37</td><td>92.98</td><td>94.30</td><td>+1.33</td><td>92.77</td><td>94.09</td><td>+1.33</td></tr><tr><td>50</td><td>91.21</td><td>93.66</td><td>+2.45</td><td>91.24</td><td>93.62</td><td>+2.38</td><td>91.40</td><td>93.40</td><td>+2.00</td></tr><tr><td>60</td><td>88.30</td><td>93.10</td><td>+4.80</td><td>88.93</td><td>93.14</td><td>+4.20</td><td>89.34</td><td>92.85</td><td>+3.51</td></tr><tr><td>70</td><td>82.40</td><td>91.05</td><td>+8.65</td><td>84.31</td><td>91.23</td><td>+6.92</td><td>85.76</td><td>91.05</td><td>+5.29</td></tr></table>

Table 13: Data-free results for a ResNet18 on CIFAR-10. Group 6-10. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 6</td><td>10</td><td>93.46</td><td>94.58</td><td>+1.12</td><td>94.20</td><td>94.60</td><td>+0.41</td><td>93.80</td><td>94.54</td><td>+0.74</td></tr><tr><td>20</td><td>89.98</td><td>94.20</td><td>+4.22</td><td>92.44</td><td>94.42</td><td>+1.98</td><td>91.17</td><td>94.04</td><td>+2.87</td></tr><tr><td>30</td><td>85.69</td><td>93.47</td><td>+7.78</td><td>90.64</td><td>93.70</td><td>+3.06</td><td>87.71</td><td>93.69</td><td>+5.98</td></tr><tr><td>40</td><td>78.73</td><td>92.27</td><td>+13.54</td><td>85.92</td><td>93.60</td><td>+7.68</td><td>78.96</td><td>92.48</td><td>+13.52</td></tr><tr><td>50</td><td>67.09</td><td>86.44</td><td>+19.35</td><td>79.16</td><td>90.85</td><td>+11.69</td><td>61.25</td><td>85.41</td><td>+24.16</td></tr><tr><td>60</td><td>50.30</td><td>83.08</td><td>+32.78</td><td>70.92</td><td>87.90</td><td>+16.98</td><td>38.66</td><td>83.69</td><td>+45.03</td></tr><tr><td>70</td><td>29.14</td><td>64.10</td><td>+34.96</td><td>49.26</td><td>61.44</td><td>+12.18</td><td>23.06</td><td>61.76</td><td>+38.70</td></tr><tr><td rowspan="7">Group 7</td><td>10</td><td>94.89</td><td>94.92</td><td>+0.02</td><td>94.89</td><td>10.10</td><td>-84.79</td><td>94.70</td><td>94.78</td><td>+0.08</td></tr><tr><td>20</td><td>94.74</td><td>94.88</td><td>+0.15</td><td>94.73</td><td>10.10</td><td>-84.63</td><td>94.33</td><td>94.64</td><td>+0.31</td></tr><tr><td>30</td><td>94.71</td><td>94.71</td><td>-0.00</td><td>94.75</td><td>10.10</td><td>-84.65</td><td>93.87</td><td>94.50</td><td>+0.62</td></tr><tr><td>40</td><td>94.57</td><td>94.59</td><td>+0.02</td><td>94.54</td><td>10.10</td><td>-84.44</td><td>93.42</td><td>94.55</td><td>+1.13</td></tr><tr><td>50</td><td>94.37</td><td>94.32</td><td>-0.05</td><td>94.33</td><td>10.10</td><td>-84.24</td><td>93.10</td><td>94.29</td><td>+1.19</td></tr><tr><td>60</td><td>94.02</td><td>94.10</td><td>+0.07</td><td>94.11</td><td>10.10</td><td>-84.01</td><td>92.10</td><td>94.32</td><td>+2.23</td></tr><tr><td>70</td><td>93.25</td><td>93.77</td><td>+0.52</td><td>93.23</td><td>10.10</td><td>-83.13</td><td>90.27</td><td>93.77</td><td>+3.50</td></tr><tr><td rowspan="7">Group 8</td><td>10</td><td>94.80</td><td>94.87</td><td>+0.07</td><td>94.71</td><td>94.71</td><td>+0.00</td><td>94.21</td><td>94.74</td><td>+0.53</td></tr><tr><td>20</td><td>94.48</td><td>94.76</td><td>+0.28</td><td>94.47</td><td>94.64</td><td>+0.17</td><td>93.44</td><td>94.49</td><td>+1.04</td></tr><tr><td>30</td><td>93.97</td><td>94.59</td><td>+0.62</td><td>94.28</td><td>94.45</td><td>+0.17</td><td>90.71</td><td>93.82</td><td>+3.11</td></tr><tr><td>40</td><td>93.02</td><td>93.74</td><td>+0.72</td><td>93.32</td><td>93.95</td><td>+0.62</td><td>87.82</td><td>93.04</td><td>+5.23</td></tr><tr><td>50</td><td>91.03</td><td>91.42</td><td>+0.39</td><td>91.47</td><td>92.28</td><td>+0.81</td><td>82.35</td><td>88.71</td><td>+6.36</td></tr><tr><td>60</td><td>86.78</td><td>88.72</td><td>+1.93</td><td>86.46</td><td>91.16</td><td>+4.70</td><td>69.53</td><td>86.15</td><td>+16.62</td></tr><tr><td>70</td><td>77.14</td><td>79.06</td><td>+1.91</td><td>75.05</td><td>86.84</td><td>+11.79</td><td>49.80</td><td>70.09</td><td>+20.29</td></tr><tr><td rowspan="7">Group 9</td><td>10</td><td>94.26</td><td>94.38</td><td>+0.11</td><td>94.56</td><td>94.35</td><td>-0.21</td><td>93.03</td><td>93.89</td><td>+0.86</td></tr><tr><td>20</td><td>91.28</td><td>93.40</td><td>+2.13</td><td>94.04</td><td>93.87</td><td>-0.17</td><td>84.66</td><td>92.62</td><td>+7.97</td></tr><tr><td>30</td><td>82.56</td><td>91.57</td><td>+9.01</td><td>91.48</td><td>92.61</td><td>+1.14</td><td>57.60</td><td>86.02</td><td>+28.42</td></tr><tr><td>40</td><td>70.47</td><td>87.74</td><td>+17.27</td><td>85.47</td><td>93.21</td><td>+7.73</td><td>42.69</td><td>81.29</td><td>+38.61</td></tr><tr><td>50</td><td>57.92</td><td>71.75</td><td>+13.84</td><td>79.20</td><td>89.01</td><td>+9.81</td><td>16.63</td><td>74.11</td><td>+57.49</td></tr><tr><td>60</td><td>40.01</td><td>64.03</td><td>+24.02</td><td>65.89</td><td>79.90</td><td>+14.00</td><td>12.52</td><td>41.70</td><td>+29.18</td></tr><tr><td>70</td><td>23.41</td><td>45.47</td><td>+22.05</td><td>40.43</td><td>69.15</td><td>+28.72</td><td>10.21</td><td>17.16</td><td>+6.96</td></tr><tr><td rowspan="7">Group 10</td><td>10</td><td>94.91</td><td>94.84</td><td>-0.07</td><td>94.88</td><td>94.80</td><td>-0.08</td><td>94.45</td><td>94.73</td><td>+0.28</td></tr><tr><td>20</td><td>94.89</td><td>94.66</td><td>-0.23</td><td>94.82</td><td>94.37</td><td>-0.45</td><td>93.91</td><td>94.47</td><td>+0.55</td></tr><tr><td>30</td><td>94.83</td><td>94.42</td><td>-0.41</td><td>94.87</td><td>94.22</td><td>-0.64</td><td>93.23</td><td>94.45</td><td>+1.22</td></tr><tr><td>40</td><td>94.59</td><td>94.21</td><td>-0.37</td><td>94.85</td><td>93.89</td><td>-0.96</td><td>93.13</td><td>94.46</td><td>+1.33</td></tr><tr><td>50</td><td>94.33</td><td>93.51</td><td>-0.82</td><td>94.73</td><td>93.58</td><td>-1.15</td><td>90.27</td><td>93.62</td><td>+3.35</td></tr><tr><td>60</td><td>93.61</td><td>92.69</td><td>-0.92</td><td>94.22</td><td>93.09</td><td>-1.14</td><td>89.17</td><td>92.41</td><td>+3.23</td></tr><tr><td>70</td><td>91.41</td><td>91.70</td><td>+0.29</td><td>92.28</td><td>90.70</td><td>-1.57</td><td>86.45</td><td>89.01</td><td>+2.56</td></tr></table>

Table 14: Data-free results on VGG11-BN on CIFAR-100. Group 0-5. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 0</td><td>10</td><td>66.96</td><td>67.51</td><td>+0.55</td><td>65.53</td><td>66.16</td><td>+0.63</td><td>65.39</td><td>65.80</td><td>+0.42</td></tr><tr><td>20</td><td>65.75</td><td>67.02</td><td>+1.27</td><td>63.23</td><td>65.92</td><td>+2.69</td><td>64.53</td><td>65.41</td><td>+0.88</td></tr><tr><td>30</td><td>63.94</td><td>66.46</td><td>+2.51</td><td>61.14</td><td>64.96</td><td>+3.83</td><td>62.87</td><td>64.91</td><td>+2.05</td></tr><tr><td>40</td><td>62.12</td><td>65.77</td><td>+3.65</td><td>57.75</td><td>64.48</td><td>+6.72</td><td>60.84</td><td>63.58</td><td>+2.74</td></tr><tr><td>50</td><td>59.88</td><td>65.09</td><td>+5.21</td><td>52.46</td><td>64.02</td><td>+11.56</td><td>56.55</td><td>63.03</td><td>+6.49</td></tr><tr><td>60</td><td>54.46</td><td>62.93</td><td>+8.47</td><td>46.61</td><td>61.76</td><td>+15.15</td><td>51.60</td><td>60.62</td><td>+9.02</td></tr><tr><td>70</td><td>46.92</td><td>60.07</td><td>+13.14</td><td>38.54</td><td>59.21</td><td>+20.67</td><td>44.77</td><td>57.42</td><td>+12.65</td></tr><tr><td rowspan="7">Group 1</td><td>10</td><td>66.42</td><td>67.03</td><td>+0.61</td><td>64.83</td><td>65.99</td><td>+1.16</td><td>65.35</td><td>65.70</td><td>+0.36</td></tr><tr><td>20</td><td>63.43</td><td>66.70</td><td>+3.27</td><td>61.88</td><td>65.38</td><td>+3.50</td><td>63.42</td><td>65.06</td><td>+1.64</td></tr><tr><td>30</td><td>57.36</td><td>65.51</td><td>+8.15</td><td>57.79</td><td>64.20</td><td>+6.41</td><td>61.70</td><td>64.03</td><td>+2.33</td></tr><tr><td>40</td><td>47.87</td><td>63.87</td><td>+16.00</td><td>52.58</td><td>62.78</td><td>+10.20</td><td>58.02</td><td>62.48</td><td>+4.46</td></tr><tr><td>50</td><td>37.97</td><td>62.39</td><td>+24.42</td><td>44.30</td><td>60.96</td><td>+16.65</td><td>51.99</td><td>60.85</td><td>+8.86</td></tr><tr><td>60</td><td>26.97</td><td>59.38</td><td>+32.42</td><td>36.47</td><td>56.50</td><td>+20.03</td><td>40.54</td><td>57.46</td><td>+16.92</td></tr><tr><td>70</td><td>17.01</td><td>52.65</td><td>+35.64</td><td>26.15</td><td>51.09</td><td>+24.94</td><td>25.02</td><td>51.84</td><td>+26.82</td></tr><tr><td rowspan="7">Group 2</td><td>10</td><td>65.92</td><td>67.24</td><td>+1.32</td><td>64.88</td><td>66.01</td><td>+1.13</td><td>65.00</td><td>66.04</td><td>+1.04</td></tr><tr><td>20</td><td>62.49</td><td>66.74</td><td>+4.25</td><td>62.10</td><td>65.22</td><td>+3.12</td><td>63.53</td><td>65.18</td><td>+1.65</td></tr><tr><td>30</td><td>56.68</td><td>65.60</td><td>+8.92</td><td>58.82</td><td>64.56</td><td>+5.74</td><td>59.71</td><td>64.40</td><td>+4.69</td></tr><tr><td>40</td><td>50.29</td><td>64.14</td><td>+13.85</td><td>54.55</td><td>63.03</td><td>+8.48</td><td>54.10</td><td>63.25</td><td>+9.15</td></tr><tr><td>50</td><td>40.53</td><td>62.68</td><td>+22.15</td><td>47.24</td><td>60.31</td><td>+13.07</td><td>46.82</td><td>60.58</td><td>+13.77</td></tr><tr><td>60</td><td>24.31</td><td>60.10</td><td>+35.79</td><td>36.85</td><td>58.17</td><td>+21.32</td><td>32.31</td><td>58.51</td><td>+26.21</td></tr><tr><td>70</td><td>13.53</td><td>54.58</td><td>+41.05</td><td>19.68</td><td>53.26</td><td>+33.58</td><td>17.20</td><td>53.84</td><td>+36.64</td></tr><tr><td rowspan="7">Group 3</td><td>10</td><td>66.35</td><td>67.55</td><td>+1.21</td><td>65.09</td><td>66.13</td><td>+1.04</td><td>65.35</td><td>66.20</td><td>+0.85</td></tr><tr><td>20</td><td>62.89</td><td>67.00</td><td>+4.11</td><td>62.39</td><td>65.48</td><td>+3.09</td><td>63.42</td><td>65.64</td><td>+2.23</td></tr><tr><td>30</td><td>58.79</td><td>66.04</td><td>+7.25</td><td>59.89</td><td>65.07</td><td>+5.18</td><td>61.16</td><td>64.93</td><td>+3.77</td></tr><tr><td>40</td><td>50.44</td><td>65.07</td><td>+14.64</td><td>54.36</td><td>64.17</td><td>+9.81</td><td>57.01</td><td>63.49</td><td>+6.48</td></tr><tr><td>50</td><td>41.80</td><td>62.12</td><td>+20.32</td><td>47.59</td><td>60.57</td><td>+12.98</td><td>48.83</td><td>60.54</td><td>+11.71</td></tr><tr><td>60</td><td>31.21</td><td>60.42</td><td>+29.21</td><td>35.29</td><td>58.70</td><td>+23.41</td><td>37.66</td><td>58.19</td><td>+20.53</td></tr><tr><td>70</td><td>20.74</td><td>53.43</td><td>+32.69</td><td>22.84</td><td>52.06</td><td>+29.21</td><td>20.98</td><td>52.04</td><td>+31.05</td></tr><tr><td rowspan="7">Group 4</td><td>10</td><td>65.00</td><td>67.17</td><td>+2.17</td><td>63.35</td><td>65.76</td><td>+2.41</td><td>63.93</td><td>65.85</td><td>+1.92</td></tr><tr><td>20</td><td>58.71</td><td>66.12</td><td>+7.41</td><td>59.33</td><td>64.94</td><td>+5.62</td><td>59.32</td><td>65.04</td><td>+5.73</td></tr><tr><td>30</td><td>52.00</td><td>65.18</td><td>+13.18</td><td>52.89</td><td>64.11</td><td>+11.22</td><td>54.52</td><td>63.24</td><td>+8.72</td></tr><tr><td>40</td><td>37.99</td><td>63.16</td><td>+25.17</td><td>45.60</td><td>61.75</td><td>+16.15</td><td>46.91</td><td>61.39</td><td>+14.48</td></tr><tr><td>50</td><td>22.63</td><td>58.89</td><td>+36.26</td><td>35.16</td><td>57.87</td><td>+22.72</td><td>37.08</td><td>56.27</td><td>+19.19</td></tr><tr><td>60</td><td>13.11</td><td>54.19</td><td>+41.08</td><td>18.60</td><td>52.74</td><td>+34.14</td><td>18.32</td><td>53.83</td><td>+35.50</td></tr><tr><td>70</td><td>5.10</td><td>45.65</td><td>+40.55</td><td>7.80</td><td>46.37</td><td>+38.57</td><td>7.66</td><td>45.50</td><td>+37.84</td></tr><tr><td rowspan="7">Group 5</td><td>10</td><td>66.20</td><td>67.82</td><td>+1.62</td><td>65.03</td><td>66.32</td><td>+1.29</td><td>63.89</td><td>66.09</td><td>+2.20</td></tr><tr><td>20</td><td>61.23</td><td>67.29</td><td>+6.05</td><td>61.19</td><td>65.58</td><td>+4.38</td><td>59.71</td><td>65.57</td><td>+5.85</td></tr><tr><td>30</td><td>51.81</td><td>65.63</td><td>+13.83</td><td>54.28</td><td>65.12</td><td>+10.84</td><td>53.83</td><td>64.24</td><td>+10.41</td></tr><tr><td>40</td><td>37.58</td><td>63.28</td><td>+25.70</td><td>42.68</td><td>63.01</td><td>+20.33</td><td>44.47</td><td>62.12</td><td>+17.65</td></tr><tr><td>50</td><td>22.12</td><td>57.66</td><td>+35.54</td><td>27.74</td><td>55.85</td><td>+28.12</td><td>29.66</td><td>55.66</td><td>+26.00</td></tr><tr><td>60</td><td>10.84</td><td>52.93</td><td>+42.09</td><td>13.89</td><td>52.49</td><td>+38.60</td><td>15.35</td><td>51.71</td><td>+36.36</td></tr><tr><td>70</td><td>5.30</td><td>41.17</td><td>+35.87</td><td>8.20</td><td>43.46</td><td>+35.27</td><td>8.04</td><td>44.28</td><td>+36.24</td></tr></table>

Table 15: Data-free results on VGG11-BN on CIFAR-100. Group 6-7. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 6</td><td>10</td><td>64.33</td><td>67.01</td><td>+2.68</td><td>60.67</td><td>64.46</td><td>+3.79</td><td>63.09</td><td>66.52</td><td>+3.43</td></tr><tr><td>20</td><td>54.91</td><td>65.08</td><td>+10.18</td><td>48.99</td><td>63.01</td><td>+14.02</td><td>55.63</td><td>64.95</td><td>+9.33</td></tr><tr><td>30</td><td>39.80</td><td>61.02</td><td>+21.21</td><td>36.19</td><td>59.28</td><td>+23.08</td><td>42.50</td><td>62.69</td><td>+20.18</td></tr><tr><td>40</td><td>22.04</td><td>52.87</td><td>+30.82</td><td>18.26</td><td>52.03</td><td>+33.77</td><td>32.98</td><td>56.08</td><td>+23.10</td></tr><tr><td>50</td><td>10.74</td><td>41.23</td><td>+30.49</td><td>12.17</td><td>41.64</td><td>+29.47</td><td>18.80</td><td>45.09</td><td>+26.30</td></tr><tr><td>60</td><td>5.67</td><td>30.70</td><td>+25.03</td><td>8.76</td><td>34.49</td><td>+25.73</td><td>7.53</td><td>38.07</td><td>+30.55</td></tr><tr><td>70</td><td>2.98</td><td>21.34</td><td>+18.36</td><td>5.06</td><td>24.19</td><td>+19.13</td><td>3.33</td><td>27.16</td><td>+23.82</td></tr><tr><td rowspan="7">Group 7</td><td>10</td><td>66.95</td><td>66.43</td><td>-0.52</td><td>65.54</td><td>63.11</td><td>-2.42</td><td>60.49</td><td>63.64</td><td>+3.14</td></tr><tr><td>20</td><td>65.27</td><td>61.24</td><td>-4.02</td><td>64.44</td><td>59.46</td><td>-4.97</td><td>49.83</td><td>60.14</td><td>+10.30</td></tr><tr><td>30</td><td>60.32</td><td>57.44</td><td>-2.89</td><td>58.42</td><td>47.30</td><td>-11.12</td><td>36.83</td><td>54.33</td><td>+17.50</td></tr><tr><td>40</td><td>49.00</td><td>50.27</td><td>+1.27</td><td>53.25</td><td>41.28</td><td>-11.98</td><td>24.64</td><td>49.68</td><td>+25.04</td></tr><tr><td>50</td><td>41.53</td><td>41.59</td><td>+0.06</td><td>46.19</td><td>34.64</td><td>-11.55</td><td>12.67</td><td>42.42</td><td>+29.76</td></tr><tr><td>60</td><td>24.42</td><td>31.29</td><td>+6.87</td><td>18.81</td><td>27.59</td><td>+8.78</td><td>6.59</td><td>33.58</td><td>+27.00</td></tr><tr><td>70</td><td>10.26</td><td>21.77</td><td>+11.51</td><td>11.83</td><td>16.07</td><td>+4.24</td><td>5.71</td><td>18.73</td><td>+13.02</td></tr></table>

Table 16: Data-free results on ResNet18 on CIFAR-100. Group 0-5. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 0</td><td>10</td><td>76.01</td><td>76.16</td><td>+0.15</td><td>70.30</td><td>71.71</td><td>+1.40</td><td>73.63</td><td>74.98</td><td>+1.35</td></tr><tr><td>20</td><td>73.62</td><td>74.87</td><td>+1.26</td><td>66.26</td><td>69.87</td><td>+3.61</td><td>71.36</td><td>73.20</td><td>+1.84</td></tr><tr><td>30</td><td>70.57</td><td>72.10</td><td>+1.53</td><td>60.26</td><td>66.69</td><td>+6.44</td><td>66.45</td><td>71.30</td><td>+4.86</td></tr><tr><td>40</td><td>66.24</td><td>69.31</td><td>+3.08</td><td>50.57</td><td>61.49</td><td>+10.92</td><td>61.44</td><td>66.90</td><td>+5.46</td></tr><tr><td>50</td><td>58.22</td><td>65.88</td><td>+7.66</td><td>38.57</td><td>58.64</td><td>+20.08</td><td>55.14</td><td>64.40</td><td>+9.26</td></tr><tr><td>60</td><td>48.41</td><td>55.55</td><td>+7.14</td><td>28.28</td><td>46.83</td><td>+18.54</td><td>43.33</td><td>54.75</td><td>+11.41</td></tr><tr><td>70</td><td>37.10</td><td>42.58</td><td>+5.48</td><td>17.46</td><td>31.77</td><td>+14.31</td><td>36.69</td><td>40.57</td><td>+3.88</td></tr><tr><td rowspan="7">Group 1</td><td>10</td><td>76.38</td><td>76.85</td><td>+0.46</td><td>71.48</td><td>72.13</td><td>+0.65</td><td>74.89</td><td>75.38</td><td>+0.48</td></tr><tr><td>20</td><td>75.21</td><td>75.82</td><td>+0.61</td><td>68.99</td><td>71.07</td><td>+2.09</td><td>73.71</td><td>74.29</td><td>+0.57</td></tr><tr><td>30</td><td>74.19</td><td>74.91</td><td>+0.72</td><td>66.49</td><td>69.27</td><td>+2.79</td><td>72.17</td><td>72.70</td><td>+0.52</td></tr><tr><td>40</td><td>71.00</td><td>72.69</td><td>+1.68</td><td>63.24</td><td>66.55</td><td>+3.31</td><td>70.01</td><td>71.85</td><td>+1.84</td></tr><tr><td>50</td><td>67.77</td><td>71.53</td><td>+3.76</td><td>58.11</td><td>64.73</td><td>+6.63</td><td>67.24</td><td>70.43</td><td>+3.19</td></tr><tr><td>60</td><td>62.55</td><td>64.88</td><td>+2.33</td><td>48.68</td><td>55.73</td><td>+7.04</td><td>62.66</td><td>63.77</td><td>+1.11</td></tr><tr><td>70</td><td>56.11</td><td>55.59</td><td>-0.52</td><td>41.71</td><td>44.67</td><td>+2.96</td><td>56.22</td><td>52.40</td><td>-3.82</td></tr><tr><td rowspan="7">Group 2</td><td>10</td><td>76.69</td><td>77.22</td><td>+0.52</td><td>72.09</td><td>72.49</td><td>+0.40</td><td>75.03</td><td>75.34</td><td>+0.31</td></tr><tr><td>20</td><td>75.75</td><td>76.70</td><td>+0.95</td><td>71.45</td><td>72.31</td><td>+0.86</td><td>74.12</td><td>75.02</td><td>+0.90</td></tr><tr><td>30</td><td>74.53</td><td>76.29</td><td>+1.76</td><td>70.23</td><td>72.02</td><td>+1.79</td><td>72.60</td><td>74.03</td><td>+1.43</td></tr><tr><td>40</td><td>71.95</td><td>74.84</td><td>+2.89</td><td>68.41</td><td>71.06</td><td>+2.65</td><td>70.97</td><td>73.22</td><td>+2.25</td></tr><tr><td>50</td><td>68.18</td><td>73.14</td><td>+4.96</td><td>65.76</td><td>69.82</td><td>+4.05</td><td>67.15</td><td>71.22</td><td>+4.07</td></tr><tr><td>60</td><td>62.58</td><td>70.91</td><td>+8.33</td><td>62.14</td><td>68.40</td><td>+6.26</td><td>62.10</td><td>68.57</td><td>+6.47</td></tr><tr><td>70</td><td>54.84</td><td>65.63</td><td>+10.80</td><td>54.65</td><td>64.66</td><td>+10.01</td><td>53.48</td><td>62.10</td><td>+8.62</td></tr><tr><td rowspan="7">Group 3</td><td>10</td><td>74.07</td><td>76.45</td><td>+2.38</td><td>71.61</td><td>72.36</td><td>+0.75</td><td>73.36</td><td>75.18</td><td>+1.82</td></tr><tr><td>20</td><td>66.22</td><td>74.77</td><td>+8.55</td><td>68.12</td><td>71.55</td><td>+3.43</td><td>70.57</td><td>74.07</td><td>+3.50</td></tr><tr><td>30</td><td>53.07</td><td>72.54</td><td>+19.47</td><td>62.31</td><td>70.20</td><td>+7.89</td><td>63.92</td><td>71.98</td><td>+8.06</td></tr><tr><td>40</td><td>35.92</td><td>67.97</td><td>+32.05</td><td>52.52</td><td>66.56</td><td>+14.04</td><td>53.54</td><td>68.42</td><td>+14.88</td></tr><tr><td>50</td><td>27.15</td><td>51.21</td><td>+24.06</td><td>41.49</td><td>47.46</td><td>+5.97</td><td>45.52</td><td>50.02</td><td>+4.50</td></tr><tr><td>60</td><td>15.88</td><td>40.85</td><td>+24.97</td><td>32.54</td><td>39.36</td><td>+6.82</td><td>29.02</td><td>39.55</td><td>+10.52</td></tr><tr><td>70</td><td>11.45</td><td>18.74</td><td>+7.29</td><td>18.26</td><td>12.63</td><td>-5.63</td><td>16.01</td><td>9.11</td><td>-6.90</td></tr><tr><td rowspan="7">Group 4</td><td>10</td><td>77.41</td><td>77.43</td><td>+0.02</td><td>72.49</td><td>72.45</td><td>-0.04</td><td>75.84</td><td>75.97</td><td>+0.13</td></tr><tr><td>20</td><td>76.99</td><td>77.26</td><td>+0.28</td><td>72.45</td><td>72.62</td><td>+0.17</td><td>75.79</td><td>75.59</td><td>-0.20</td></tr><tr><td>30</td><td>76.68</td><td>77.01</td><td>+0.33</td><td>72.24</td><td>72.40</td><td>+0.16</td><td>75.62</td><td>75.58</td><td>-0.04</td></tr><tr><td>40</td><td>76.26</td><td>76.67</td><td>+0.42</td><td>71.52</td><td>72.05</td><td>+0.53</td><td>75.08</td><td>75.10</td><td>+0.02</td></tr><tr><td>50</td><td>75.82</td><td>75.88</td><td>+0.06</td><td>70.56</td><td>71.46</td><td>+0.90</td><td>74.24</td><td>74.52</td><td>+0.28</td></tr><tr><td>60</td><td>74.91</td><td>75.18</td><td>+0.27</td><td>69.11</td><td>71.36</td><td>+2.25</td><td>73.06</td><td>73.71</td><td>+0.65</td></tr><tr><td>70</td><td>73.18</td><td>73.97</td><td>+0.79</td><td>67.41</td><td>69.96</td><td>+2.55</td><td>71.25</td><td>72.32</td><td>+1.07</td></tr><tr><td rowspan="7">Group 5</td><td>10</td><td>77.03</td><td>77.45</td><td>+0.43</td><td>72.02</td><td>72.84</td><td>+0.82</td><td>75.17</td><td>76.15</td><td>+0.98</td></tr><tr><td>20</td><td>75.89</td><td>77.22</td><td>+1.33</td><td>70.01</td><td>72.98</td><td>+2.98</td><td>73.70</td><td>75.92</td><td>+2.22</td></tr><tr><td>30</td><td>74.60</td><td>76.51</td><td>+1.91</td><td>67.00</td><td>72.41</td><td>+5.41</td><td>71.67</td><td>75.15</td><td>+3.48</td></tr><tr><td>40</td><td>71.69</td><td>75.14</td><td>+3.45</td><td>62.50</td><td>70.90</td><td>+8.40</td><td>67.04</td><td>74.01</td><td>+6.97</td></tr><tr><td>50</td><td>66.46</td><td>71.03</td><td>+4.58</td><td>58.16</td><td>66.45</td><td>+8.29</td><td>61.54</td><td>69.32</td><td>+7.78</td></tr><tr><td>60</td><td>58.35</td><td>68.89</td><td>+10.54</td><td>45.24</td><td>64.24</td><td>+19.00</td><td>53.68</td><td>68.01</td><td>+14.33</td></tr><tr><td>70</td><td>44.50</td><td>61.99</td><td>+17.48</td><td>34.30</td><td>55.93</td><td>+21.64</td><td>41.11</td><td>61.33</td><td>+20.22</td></tr></table>

Table 17: Data-free results on ResNet18 on CIFAR-100. Group 6-10. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 6</td><td>10</td><td>69.78</td><td>75.80</td><td>+6.02</td><td>67.37</td><td>71.79</td><td>+4.42</td><td>71.67</td><td>75.18</td><td>+3.51</td></tr><tr><td>20</td><td>55.99</td><td>72.57</td><td>+16.57</td><td>63.01</td><td>71.59</td><td>+8.57</td><td>66.87</td><td>73.75</td><td>+6.88</td></tr><tr><td>30</td><td>39.59</td><td>71.20</td><td>+31.62</td><td>51.77</td><td>69.61</td><td>+17.84</td><td>54.26</td><td>71.37</td><td>+17.11</td></tr><tr><td>40</td><td>29.47</td><td>64.98</td><td>+35.51</td><td>34.22</td><td>65.32</td><td>+31.10</td><td>41.69</td><td>67.75</td><td>+26.06</td></tr><tr><td>50</td><td>18.06</td><td>46.48</td><td>+28.42</td><td>23.29</td><td>44.48</td><td>+21.19</td><td>27.90</td><td>43.30</td><td>+15.41</td></tr><tr><td>60</td><td>3.61</td><td>42.87</td><td>+39.26</td><td>16.55</td><td>34.34</td><td>+17.78</td><td>14.76</td><td>46.28</td><td>+31.52</td></tr><tr><td>70</td><td>1.10</td><td>19.20</td><td>+18.11</td><td>9.24</td><td>8.78</td><td>-0.45</td><td>5.72</td><td>21.46</td><td>+15.74</td></tr><tr><td rowspan="7">Group 7</td><td>10</td><td>77.31</td><td>77.47</td><td>+0.16</td><td>72.48</td><td>72.49</td><td>+0.01</td><td>76.02</td><td>76.09</td><td>+0.07</td></tr><tr><td>20</td><td>76.84</td><td>77.38</td><td>+0.54</td><td>72.34</td><td>72.47</td><td>+0.13</td><td>75.23</td><td>76.21</td><td>+0.98</td></tr><tr><td>30</td><td>76.48</td><td>77.23</td><td>+0.75</td><td>71.81</td><td>72.28</td><td>+0.47</td><td>73.93</td><td>75.94</td><td>+2.01</td></tr><tr><td>40</td><td>75.50</td><td>77.12</td><td>+1.61</td><td>71.21</td><td>72.19</td><td>+0.98</td><td>71.25</td><td>75.92</td><td>+4.67</td></tr><tr><td>50</td><td>73.61</td><td>75.56</td><td>+1.96</td><td>69.77</td><td>71.31</td><td>+1.54</td><td>69.59</td><td>73.99</td><td>+4.40</td></tr><tr><td>60</td><td>71.09</td><td>76.02</td><td>+4.92</td><td>67.41</td><td>71.05</td><td>+3.65</td><td>65.50</td><td>74.40</td><td>+8.90</td></tr><tr><td>70</td><td>67.94</td><td>74.64</td><td>+6.70</td><td>66.31</td><td>69.77</td><td>+3.46</td><td>60.58</td><td>72.63</td><td>+12.05</td></tr><tr><td rowspan="7">Group 8</td><td>10</td><td>77.23</td><td>77.70</td><td>+0.46</td><td>72.17</td><td>72.66</td><td>+0.48</td><td>75.42</td><td>76.16</td><td>+0.74</td></tr><tr><td>20</td><td>76.92</td><td>77.23</td><td>+0.31</td><td>70.49</td><td>72.16</td><td>+1.67</td><td>72.48</td><td>75.58</td><td>+3.11</td></tr><tr><td>30</td><td>75.83</td><td>76.13</td><td>+0.30</td><td>65.82</td><td>71.09</td><td>+5.27</td><td>64.99</td><td>73.56</td><td>+8.56</td></tr><tr><td>40</td><td>73.12</td><td>73.97</td><td>+0.85</td><td>59.01</td><td>68.63</td><td>+9.62</td><td>58.02</td><td>69.22</td><td>+11.20</td></tr><tr><td>50</td><td>68.84</td><td>66.72</td><td>-2.12</td><td>51.62</td><td>57.01</td><td>+5.39</td><td>50.76</td><td>49.91</td><td>-0.85</td></tr><tr><td>60</td><td>59.29</td><td>57.90</td><td>-1.38</td><td>41.72</td><td>51.94</td><td>+10.22</td><td>34.89</td><td>36.93</td><td>+2.04</td></tr><tr><td>70</td><td>43.29</td><td>33.75</td><td>-9.53</td><td>30.38</td><td>35.99</td><td>+5.61</td><td>19.31</td><td>31.30</td><td>+11.99</td></tr><tr><td rowspan="7">Group 9</td><td>10</td><td>73.95</td><td>75.43</td><td>+1.47</td><td>70.52</td><td>72.23</td><td>+1.71</td><td>73.37</td><td>75.19</td><td>+1.82</td></tr><tr><td>20</td><td>69.46</td><td>72.05</td><td>+2.59</td><td>68.42</td><td>71.35</td><td>+2.93</td><td>65.49</td><td>72.39</td><td>+6.90</td></tr><tr><td>30</td><td>46.02</td><td>62.52</td><td>+16.50</td><td>63.97</td><td>68.60</td><td>+4.63</td><td>55.27</td><td>68.80</td><td>+13.53</td></tr><tr><td>40</td><td>30.89</td><td>55.22</td><td>+24.33</td><td>52.59</td><td>63.11</td><td>+10.52</td><td>33.60</td><td>63.24</td><td>+29.64</td></tr><tr><td>50</td><td>12.79</td><td>36.13</td><td>+23.34</td><td>44.74</td><td>43.83</td><td>-0.91</td><td>21.64</td><td>33.83</td><td>+12.19</td></tr><tr><td>60</td><td>4.10</td><td>29.84</td><td>+25.73</td><td>28.89</td><td>47.17</td><td>+18.29</td><td>8.03</td><td>20.18</td><td>+12.15</td></tr><tr><td>70</td><td>1.30</td><td>18.36</td><td>+17.07</td><td>12.86</td><td>24.49</td><td>+11.63</td><td>3.00</td><td>7.29</td><td>+4.29</td></tr><tr><td rowspan="7">Group 10</td><td>10</td><td>77.58</td><td>77.46</td><td>-0.12</td><td>72.59</td><td>72.75</td><td>+0.16</td><td>75.95</td><td>76.20</td><td>+0.25</td></tr><tr><td>20</td><td>77.23</td><td>77.47</td><td>+0.24</td><td>72.57</td><td>72.33</td><td>-0.24</td><td>75.64</td><td>75.98</td><td>+0.34</td></tr><tr><td>30</td><td>77.23</td><td>76.96</td><td>-0.27</td><td>72.51</td><td>71.91</td><td>-0.59</td><td>74.52</td><td>75.72</td><td>+1.21</td></tr><tr><td>40</td><td>76.99</td><td>75.82</td><td>-1.17</td><td>72.23</td><td>72.25</td><td>+0.02</td><td>71.33</td><td>74.07</td><td>+2.74</td></tr><tr><td>50</td><td>76.17</td><td>74.90</td><td>-1.27</td><td>72.30</td><td>71.79</td><td>-0.51</td><td>69.52</td><td>69.74</td><td>+0.22</td></tr><tr><td>60</td><td>75.71</td><td>75.51</td><td>-0.20</td><td>71.94</td><td>70.98</td><td>-0.96</td><td>66.93</td><td>67.14</td><td>+0.21</td></tr><tr><td>70</td><td>73.79</td><td>72.51</td><td>-1.29</td><td>70.34</td><td>66.40</td><td>-3.95</td><td>59.09</td><td>65.78</td><td>+6.70</td></tr></table>

Table 18: Data-free results on ResNet50 on ImageNet. Group 0-9. Pruning criterion: CHIP (Sui et al., 2021). 

<table><tr><td>Group</td><td>Sparsity (%)</td><td>CHIP (%)</td><td>IF (CHIP)(%)</td><td> $\delta(\%)$ </td></tr><tr><td rowspan="4">Group 0</td><td>10</td><td>74.74</td><td>75.51</td><td>+0.77</td></tr><tr><td>20</td><td>71.83</td><td>74.79</td><td>+2.96</td></tr><tr><td>30</td><td>66.79</td><td>73.31</td><td>+6.52</td></tr><tr><td>40</td><td>60.13</td><td>70.88</td><td>+10.75</td></tr><tr><td rowspan="4">Group 1</td><td>10</td><td>74.67</td><td>75.36</td><td>+0.69</td></tr><tr><td>20</td><td>73.57</td><td>74.66</td><td>+1.09</td></tr><tr><td>30</td><td>72.01</td><td>73.7</td><td>+1.69</td></tr><tr><td>40</td><td>70.48</td><td>72.42</td><td>+1.94</td></tr><tr><td rowspan="4">Group 2</td><td>10</td><td>74.67</td><td>75.29</td><td>+0.62</td></tr><tr><td>20</td><td>73.11</td><td>74.27</td><td>+1.16</td></tr><tr><td>30</td><td>71.45</td><td>72.88</td><td>+1.43</td></tr><tr><td>40</td><td>68.98</td><td>71.15</td><td>+2.17</td></tr><tr><td rowspan="4">Group 3</td><td>10</td><td>74.95</td><td>75.5</td><td>+0.55</td></tr><tr><td>20</td><td>73.42</td><td>74.9</td><td>+1.48</td></tr><tr><td>30</td><td>72.16</td><td>73.97</td><td>+1.81</td></tr><tr><td>40</td><td>70.19</td><td>72.58</td><td>+2.39</td></tr><tr><td rowspan="4">Group 4</td><td>10</td><td>74.65</td><td>75.24</td><td>+0.59</td></tr><tr><td>20</td><td>73.71</td><td>75.01</td><td>+1.3</td></tr><tr><td>30</td><td>72.62</td><td>74.38</td><td>+1.76</td></tr><tr><td>40</td><td>71.24</td><td>73.54</td><td>+2.3</td></tr></table>

<table><tr><td>Group</td><td>Sparsity (%)</td><td>CHIP (%)</td><td>IF (CHIP)(%)</td><td> $\delta(\%)$ </td></tr><tr><td rowspan="4">Group 5</td><td>10</td><td>74.52</td><td>75.44</td><td>+0.92</td></tr><tr><td>20</td><td>72.98</td><td>74.81</td><td>+1.83</td></tr><tr><td>30</td><td>71.31</td><td>74.07</td><td>+2.76</td></tr><tr><td>40</td><td>69.39</td><td>72.64</td><td>+3.25</td></tr><tr><td rowspan="4">Group 6</td><td>10</td><td>74.27</td><td>75.38</td><td>+1.11</td></tr><tr><td>20</td><td>72.39</td><td>74.68</td><td>+2.29</td></tr><tr><td>30</td><td>69.78</td><td>73.73</td><td>+3.95</td></tr><tr><td>40</td><td>67.51</td><td>72.41</td><td>+4.9</td></tr><tr><td rowspan="4">Group 7</td><td>10</td><td>67.98</td><td>73.11</td><td>+5.13</td></tr><tr><td>20</td><td>58.45</td><td>70.21</td><td>+11.76</td></tr><tr><td>30</td><td>42.37</td><td>65.51</td><td>+23.14</td></tr><tr><td>40</td><td>26.56</td><td>60.62</td><td>+34.06</td></tr><tr><td rowspan="4">Group 8</td><td>10</td><td>75.39</td><td>75.62</td><td>+0.23</td></tr><tr><td>20</td><td>74.83</td><td>75.32</td><td>+0.49</td></tr><tr><td>30</td><td>74.59</td><td>75.12</td><td>+0.53</td></tr><tr><td>40</td><td>73.41</td><td>74.71</td><td>+1.3</td></tr><tr><td rowspan="4">Group 9</td><td>10</td><td>75.43</td><td>75.67</td><td>+0.24</td></tr><tr><td>20</td><td>74.9</td><td>75.41</td><td>+0.51</td></tr><tr><td>30</td><td>74.55</td><td>75.32</td><td>+0.77</td></tr><tr><td>40</td><td>74.08</td><td>75.12</td><td>+1.04</td></tr></table>

Table 19: Data-free results on ResNet50 on ImageNet. Group 0-5. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 0</td><td>10</td><td>74.29</td><td>75.11</td><td>+0.82</td><td>73.58</td><td>75.46</td><td>+1.88</td><td>72.84</td><td>75.29</td><td>+2.45</td></tr><tr><td>20</td><td>71.74</td><td>74.35</td><td>+2.61</td><td>68.36</td><td>74.30</td><td>+5.95</td><td>68.11</td><td>73.84</td><td>+5.73</td></tr><tr><td>30</td><td>68.05</td><td>72.90</td><td>+4.86</td><td>61.10</td><td>72.56</td><td>+11.46</td><td>62.51</td><td>72.45</td><td>+9.94</td></tr><tr><td>40</td><td>61.71</td><td>70.66</td><td>+8.95</td><td>46.87</td><td>69.73</td><td>+22.86</td><td>54.38</td><td>70.23</td><td>+15.85</td></tr><tr><td>50</td><td>52.76</td><td>66.69</td><td>+13.93</td><td>30.23</td><td>64.60</td><td>+34.37</td><td>42.47</td><td>66.12</td><td>+23.65</td></tr><tr><td>60</td><td>41.90</td><td>62.18</td><td>+20.28</td><td>17.64</td><td>59.17</td><td>+41.53</td><td>29.79</td><td>59.71</td><td>+29.93</td></tr><tr><td>70</td><td>28.35</td><td>52.94</td><td>+24.59</td><td>10.81</td><td>48.84</td><td>+38.03</td><td>19.90</td><td>45.74</td><td>+25.84</td></tr><tr><td rowspan="7">Group 1</td><td>10</td><td>74.78</td><td>75.30</td><td>+0.52</td><td>74.99</td><td>75.33</td><td>+0.34</td><td>75.06</td><td>75.50</td><td>+0.44</td></tr><tr><td>20</td><td>73.82</td><td>74.60</td><td>+0.77</td><td>73.92</td><td>74.77</td><td>+0.85</td><td>74.14</td><td>74.75</td><td>+0.61</td></tr><tr><td>30</td><td>72.32</td><td>73.60</td><td>+1.27</td><td>72.41</td><td>73.71</td><td>+1.30</td><td>72.75</td><td>73.77</td><td>+1.02</td></tr><tr><td>40</td><td>70.93</td><td>72.33</td><td>+1.41</td><td>71.01</td><td>72.60</td><td>+1.59</td><td>70.78</td><td>72.69</td><td>+1.91</td></tr><tr><td>50</td><td>68.46</td><td>70.95</td><td>+2.50</td><td>69.02</td><td>71.02</td><td>+1.99</td><td>69.20</td><td>71.04</td><td>+1.83</td></tr><tr><td>60</td><td>65.88</td><td>68.46</td><td>+2.58</td><td>65.97</td><td>68.29</td><td>+2.32</td><td>66.37</td><td>68.86</td><td>+2.50</td></tr><tr><td>70</td><td>63.06</td><td>64.28</td><td>+1.22</td><td>62.92</td><td>64.23</td><td>+1.31</td><td>63.35</td><td>64.81</td><td>+1.46</td></tr><tr><td rowspan="7">Group 2</td><td>10</td><td>74.76</td><td>75.35</td><td>+0.59</td><td>74.55</td><td>75.40</td><td>+0.85</td><td>74.50</td><td>75.30</td><td>+0.80</td></tr><tr><td>20</td><td>73.61</td><td>74.60</td><td>+0.99</td><td>73.08</td><td>74.45</td><td>+1.37</td><td>73.08</td><td>74.45</td><td>+1.37</td></tr><tr><td>30</td><td>72.03</td><td>73.54</td><td>+1.51</td><td>71.23</td><td>73.08</td><td>+1.85</td><td>71.28</td><td>73.32</td><td>+2.04</td></tr><tr><td>40</td><td>70.13</td><td>71.97</td><td>+1.84</td><td>69.19</td><td>71.37</td><td>+2.18</td><td>68.91</td><td>71.80</td><td>+2.89</td></tr><tr><td>50</td><td>66.93</td><td>70.01</td><td>+3.08</td><td>65.99</td><td>69.63</td><td>+3.64</td><td>66.39</td><td>69.80</td><td>+3.41</td></tr><tr><td>60</td><td>63.41</td><td>66.70</td><td>+3.29</td><td>62.15</td><td>65.67</td><td>+3.52</td><td>63.36</td><td>66.14</td><td>+2.77</td></tr><tr><td>70</td><td>59.82</td><td>61.33</td><td>+1.51</td><td>57.53</td><td>60.46</td><td>+2.94</td><td>60.02</td><td>61.38</td><td>+1.36</td></tr><tr><td rowspan="7">Group 3</td><td>10</td><td>74.82</td><td>75.54</td><td>+0.73</td><td>75.06</td><td>75.64</td><td>+0.58</td><td>75.16</td><td>75.48</td><td>+0.32</td></tr><tr><td>20</td><td>73.84</td><td>75.04</td><td>+1.20</td><td>74.34</td><td>75.01</td><td>+0.67</td><td>74.37</td><td>75.16</td><td>+0.79</td></tr><tr><td>30</td><td>73.12</td><td>74.27</td><td>+1.15</td><td>73.31</td><td>74.20</td><td>+0.89</td><td>73.58</td><td>74.17</td><td>+0.58</td></tr><tr><td>40</td><td>72.23</td><td>73.18</td><td>+0.95</td><td>72.21</td><td>73.41</td><td>+1.20</td><td>72.47</td><td>72.99</td><td>+0.52</td></tr><tr><td>50</td><td>70.94</td><td>70.50</td><td>-0.43</td><td>71.16</td><td>70.76</td><td>-0.39</td><td>70.97</td><td>70.23</td><td>-0.74</td></tr><tr><td>60</td><td>69.53</td><td>69.09</td><td>-0.44</td><td>69.76</td><td>69.57</td><td>-0.19</td><td>69.01</td><td>68.87</td><td>-0.14</td></tr><tr><td>70</td><td>67.68</td><td>65.88</td><td>-1.80</td><td>67.85</td><td>66.23</td><td>-1.62</td><td>67.20</td><td>65.49</td><td>-1.71</td></tr><tr><td rowspan="7">Group 4</td><td>10</td><td>75.24</td><td>75.69</td><td>+0.45</td><td>75.26</td><td>75.67</td><td>+0.41</td><td>74.99</td><td>75.61</td><td>+0.62</td></tr><tr><td>20</td><td>74.54</td><td>75.32</td><td>+0.79</td><td>74.28</td><td>75.24</td><td>+0.97</td><td>73.99</td><td>75.15</td><td>+1.16</td></tr><tr><td>30</td><td>73.75</td><td>74.69</td><td>+0.94</td><td>72.77</td><td>74.52</td><td>+1.75</td><td>73.01</td><td>74.48</td><td>+1.47</td></tr><tr><td>40</td><td>72.75</td><td>73.88</td><td>+1.13</td><td>70.47</td><td>73.73</td><td>+3.26</td><td>71.95</td><td>73.53</td><td>+1.58</td></tr><tr><td>50</td><td>71.40</td><td>71.42</td><td>+0.02</td><td>67.87</td><td>71.52</td><td>+3.65</td><td>70.71</td><td>71.15</td><td>+0.44</td></tr><tr><td>60</td><td>69.64</td><td>69.91</td><td>+0.27</td><td>64.26</td><td>69.72</td><td>+5.46</td><td>68.38</td><td>69.54</td><td>+1.16</td></tr><tr><td>70</td><td>66.96</td><td>65.50</td><td>-1.45</td><td>61.86</td><td>66.24</td><td>+4.38</td><td>63.92</td><td>63.93</td><td>+0.01</td></tr><tr><td rowspan="7">Group 5</td><td>10</td><td>74.86</td><td>75.55</td><td>+0.69</td><td>74.58</td><td>75.35</td><td>+0.77</td><td>74.87</td><td>75.53</td><td>+0.67</td></tr><tr><td>20</td><td>73.61</td><td>75.02</td><td>+1.41</td><td>73.08</td><td>74.93</td><td>+1.85</td><td>73.67</td><td>74.97</td><td>+1.29</td></tr><tr><td>30</td><td>71.93</td><td>74.43</td><td>+2.49</td><td>71.45</td><td>74.06</td><td>+2.61</td><td>72.34</td><td>74.31</td><td>+1.97</td></tr><tr><td>40</td><td>70.25</td><td>73.22</td><td>+2.97</td><td>69.40</td><td>72.87</td><td>+3.47</td><td>70.06</td><td>73.06</td><td>+3.00</td></tr><tr><td>50</td><td>68.03</td><td>70.86</td><td>+2.84</td><td>66.64</td><td>71.24</td><td>+4.60</td><td>67.63</td><td>70.86</td><td>+3.23</td></tr><tr><td>60</td><td>64.61</td><td>68.41</td><td>+3.80</td><td>63.67</td><td>68.82</td><td>+5.15</td><td>64.74</td><td>68.07</td><td>+3.33</td></tr><tr><td>70</td><td>61.39</td><td>62.65</td><td>+1.26</td><td>58.06</td><td>63.26</td><td>+5.20</td><td>61.20</td><td>62.28</td><td>+1.08</td></tr></table>

Table 20: Data-free results on ResNet50 on ImageNet. Group 6-11. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 6</td><td>10</td><td>74.56</td><td>75.43</td><td>+0.87</td><td>74.37</td><td>75.30</td><td>+0.93</td><td>74.79</td><td>75.46</td><td>+0.67</td></tr><tr><td>20</td><td>73.08</td><td>74.88</td><td>+1.80</td><td>72.53</td><td>74.61</td><td>+2.08</td><td>73.18</td><td>74.97</td><td>+1.79</td></tr><tr><td>30</td><td>70.98</td><td>74.09</td><td>+3.11</td><td>70.66</td><td>74.02</td><td>+3.36</td><td>71.28</td><td>74.09</td><td>+2.82</td></tr><tr><td>40</td><td>68.92</td><td>72.85</td><td>+3.93</td><td>67.30</td><td>72.68</td><td>+5.38</td><td>69.06</td><td>72.98</td><td>+3.92</td></tr><tr><td>50</td><td>66.28</td><td>70.65</td><td>+4.37</td><td>62.90</td><td>70.82</td><td>+7.91</td><td>66.23</td><td>70.53</td><td>+4.29</td></tr><tr><td>60</td><td>63.25</td><td>67.88</td><td>+4.63</td><td>58.67</td><td>67.86</td><td>+9.19</td><td>64.07</td><td>67.39</td><td>+3.32</td></tr><tr><td>70</td><td>59.09</td><td>61.80</td><td>+2.71</td><td>52.85</td><td>60.48</td><td>+7.63</td><td>58.50</td><td>61.42</td><td>+2.91</td></tr><tr><td rowspan="7">Group 7</td><td>10</td><td>71.18</td><td>74.31</td><td>+3.13</td><td>72.77</td><td>74.68</td><td>+1.91</td><td>72.90</td><td>75.15</td><td>+2.25</td></tr><tr><td>20</td><td>64.11</td><td>72.27</td><td>+8.16</td><td>66.90</td><td>72.42</td><td>+5.52</td><td>67.77</td><td>73.35</td><td>+5.58</td></tr><tr><td>30</td><td>50.33</td><td>69.02</td><td>+18.69</td><td>57.71</td><td>68.86</td><td>+11.14</td><td>59.52</td><td>70.93</td><td>+11.41</td></tr><tr><td>40</td><td>33.79</td><td>62.84</td><td>+29.05</td><td>45.91</td><td>63.56</td><td>+17.65</td><td>45.87</td><td>65.46</td><td>+19.60</td></tr><tr><td>50</td><td>18.54</td><td>51.01</td><td>+32.47</td><td>33.06</td><td>51.04</td><td>+17.98</td><td>29.57</td><td>46.38</td><td>+16.81</td></tr><tr><td>60</td><td>7.86</td><td>34.63</td><td>+26.77</td><td>17.87</td><td>31.53</td><td>+13.66</td><td>9.14</td><td>31.87</td><td>+22.74</td></tr><tr><td>70</td><td>2.49</td><td>7.10</td><td>+4.61</td><td>6.70</td><td>7.40</td><td>+0.70</td><td>2.21</td><td>6.02</td><td>+3.81</td></tr><tr><td rowspan="7">Group 8</td><td>10</td><td>75.51</td><td>75.80</td><td>+0.29</td><td>75.63</td><td>75.79</td><td>+0.17</td><td>75.61</td><td>75.78</td><td>+0.17</td></tr><tr><td>20</td><td>75.03</td><td>75.49</td><td>+0.46</td><td>75.30</td><td>75.58</td><td>+0.28</td><td>75.28</td><td>75.66</td><td>+0.38</td></tr><tr><td>30</td><td>74.43</td><td>75.24</td><td>+0.80</td><td>74.75</td><td>75.28</td><td>+0.54</td><td>74.64</td><td>75.22</td><td>+0.58</td></tr><tr><td>40</td><td>73.88</td><td>74.92</td><td>+1.04</td><td>74.31</td><td>74.70</td><td>+0.39</td><td>74.28</td><td>74.76</td><td>+0.49</td></tr><tr><td>50</td><td>72.86</td><td>73.47</td><td>+0.60</td><td>73.47</td><td>73.37</td><td>-0.10</td><td>73.46</td><td>73.41</td><td>-0.05</td></tr><tr><td>60</td><td>71.82</td><td>72.67</td><td>+0.85</td><td>72.38</td><td>72.58</td><td>+0.20</td><td>72.64</td><td>72.10</td><td>-0.54</td></tr><tr><td>70</td><td>70.69</td><td>70.06</td><td>-0.63</td><td>70.79</td><td>70.58</td><td>-0.21</td><td>71.13</td><td>68.54</td><td>-2.59</td></tr><tr><td rowspan="7">Group 9</td><td>10</td><td>75.74</td><td>75.85</td><td>+0.11</td><td>75.58</td><td>75.72</td><td>+0.14</td><td>75.72</td><td>75.82</td><td>+0.10</td></tr><tr><td>20</td><td>75.39</td><td>75.69</td><td>+0.30</td><td>75.24</td><td>75.61</td><td>+0.38</td><td>75.36</td><td>75.82</td><td>+0.47</td></tr><tr><td>30</td><td>75.00</td><td>75.41</td><td>+0.41</td><td>74.90</td><td>75.36</td><td>+0.46</td><td>74.82</td><td>75.48</td><td>+0.66</td></tr><tr><td>40</td><td>74.40</td><td>75.11</td><td>+0.71</td><td>74.35</td><td>75.00</td><td>+0.66</td><td>74.45</td><td>75.17</td><td>+0.72</td></tr><tr><td>50</td><td>73.95</td><td>74.02</td><td>+0.07</td><td>73.87</td><td>74.14</td><td>+0.27</td><td>73.79</td><td>74.11</td><td>+0.32</td></tr><tr><td>60</td><td>73.43</td><td>73.61</td><td>+0.18</td><td>73.29</td><td>73.69</td><td>+0.40</td><td>73.27</td><td>73.92</td><td>+0.66</td></tr><tr><td>70</td><td>72.86</td><td>71.83</td><td>-1.02</td><td>72.54</td><td>72.33</td><td>-0.21</td><td>72.38</td><td>71.95</td><td>-0.43</td></tr><tr><td rowspan="7">Group 10</td><td>10</td><td>75.56</td><td>75.72</td><td>+0.16</td><td>75.80</td><td>75.75</td><td>-0.04</td><td>75.69</td><td>75.84</td><td>+0.16</td></tr><tr><td>20</td><td>75.22</td><td>75.63</td><td>+0.41</td><td>75.47</td><td>75.64</td><td>+0.17</td><td>75.47</td><td>75.66</td><td>+0.18</td></tr><tr><td>30</td><td>75.01</td><td>75.39</td><td>+0.38</td><td>75.17</td><td>75.46</td><td>+0.29</td><td>75.23</td><td>75.49</td><td>+0.26</td></tr><tr><td>40</td><td>74.58</td><td>75.00</td><td>+0.41</td><td>74.86</td><td>75.11</td><td>+0.25</td><td>74.74</td><td>75.17</td><td>+0.43</td></tr><tr><td>50</td><td>74.27</td><td>74.46</td><td>+0.19</td><td>74.40</td><td>74.13</td><td>-0.27</td><td>74.61</td><td>73.96</td><td>-0.65</td></tr><tr><td>60</td><td>73.86</td><td>73.94</td><td>+0.08</td><td>73.81</td><td>74.00</td><td>+0.19</td><td>74.27</td><td>71.24</td><td>-3.03</td></tr><tr><td>70</td><td>73.42</td><td>72.44</td><td>-0.97</td><td>73.42</td><td>72.45</td><td>-0.96</td><td>73.59</td><td>68.29</td><td>-5.30</td></tr><tr><td rowspan="7">Group 11</td><td>10</td><td>75.75</td><td>75.88</td><td>+0.14</td><td>75.70</td><td>75.77</td><td>+0.07</td><td>75.73</td><td>75.91</td><td>+0.18</td></tr><tr><td>20</td><td>75.57</td><td>75.78</td><td>+0.22</td><td>75.58</td><td>75.70</td><td>+0.12</td><td>75.45</td><td>75.81</td><td>+0.36</td></tr><tr><td>30</td><td>75.25</td><td>75.70</td><td>+0.45</td><td>75.41</td><td>75.50</td><td>+0.09</td><td>75.10</td><td>75.66</td><td>+0.56</td></tr><tr><td>40</td><td>75.05</td><td>75.44</td><td>+0.39</td><td>75.02</td><td>75.47</td><td>+0.44</td><td>74.60</td><td>75.42</td><td>+0.81</td></tr><tr><td>50</td><td>74.62</td><td>74.66</td><td>+0.04</td><td>74.79</td><td>74.76</td><td>-0.03</td><td>74.13</td><td>74.59</td><td>+0.47</td></tr><tr><td>60</td><td>74.11</td><td>74.43</td><td>+0.32</td><td>74.27</td><td>74.56</td><td>+0.28</td><td>73.45</td><td>74.26</td><td>+0.81</td></tr><tr><td>70</td><td>73.32</td><td>73.19</td><td>-0.14</td><td>73.66</td><td>73.54</td><td>-0.12</td><td>72.80</td><td>72.91</td><td>+0.12</td></tr></table>

Table 21: Data-free results on ResNet50 on ImageNet. Group 12-17. 

<table><tr><td>Group</td><td>Sparsity (%)</td><td> $\ell_1(%)$ </td><td>IF  $(\ell_1)(%)$ </td><td> $\delta(%)$ </td><td>Taylor(%)</td><td>IF (Taylor)(%)</td><td> $\delta(%)$ </td><td>LAMP(%)</td><td>IF (LAMP)(%)</td><td> $\delta(%)$ </td></tr><tr><td rowspan="7">Group 12</td><td>10</td><td>75.65</td><td>75.82</td><td>+0.17</td><td>75.71</td><td>75.79</td><td>+0.09</td><td>75.67</td><td>75.78</td><td>+0.11</td></tr><tr><td>20</td><td>75.41</td><td>75.71</td><td>+0.30</td><td>75.36</td><td>75.69</td><td>+0.32</td><td>75.40</td><td>75.59</td><td>+0.19</td></tr><tr><td>30</td><td>74.91</td><td>75.51</td><td>+0.60</td><td>74.89</td><td>75.51</td><td>+0.62</td><td>75.07</td><td>75.53</td><td>+0.47</td></tr><tr><td>40</td><td>74.44</td><td>75.14</td><td>+0.70</td><td>74.50</td><td>75.26</td><td>+0.76</td><td>74.45</td><td>75.05</td><td>+0.60</td></tr><tr><td>50</td><td>73.85</td><td>74.16</td><td>+0.32</td><td>74.26</td><td>74.47</td><td>+0.21</td><td>73.69</td><td>73.76</td><td>+0.06</td></tr><tr><td>60</td><td>72.92</td><td>73.63</td><td>+0.71</td><td>73.38</td><td>74.07</td><td>+0.69</td><td>72.75</td><td>71.93</td><td>-0.82</td></tr><tr><td>70</td><td>72.08</td><td>71.33</td><td>-0.75</td><td>72.65</td><td>72.61</td><td>-0.05</td><td>72.02</td><td>66.49</td><td>-5.53</td></tr><tr><td rowspan="7">Group 13</td><td>10</td><td>75.83</td><td>75.87</td><td>+0.04</td><td>75.75</td><td>75.83</td><td>+0.07</td><td>75.75</td><td>75.85</td><td>+0.10</td></tr><tr><td>20</td><td>75.62</td><td>75.78</td><td>+0.15</td><td>75.62</td><td>75.74</td><td>+0.12</td><td>75.59</td><td>75.81</td><td>+0.22</td></tr><tr><td>30</td><td>75.43</td><td>75.77</td><td>+0.34</td><td>75.53</td><td>75.81</td><td>+0.29</td><td>75.21</td><td>75.73</td><td>+0.51</td></tr><tr><td>40</td><td>74.90</td><td>75.51</td><td>+0.61</td><td>75.11</td><td>75.54</td><td>+0.43</td><td>74.86</td><td>75.58</td><td>+0.72</td></tr><tr><td>50</td><td>74.59</td><td>74.77</td><td>+0.18</td><td>74.87</td><td>74.79</td><td>-0.08</td><td>74.30</td><td>74.55</td><td>+0.24</td></tr><tr><td>60</td><td>73.97</td><td>74.25</td><td>+0.28</td><td>74.25</td><td>74.41</td><td>+0.16</td><td>73.67</td><td>74.19</td><td>+0.52</td></tr><tr><td>70</td><td>73.07</td><td>72.01</td><td>-1.07</td><td>73.36</td><td>72.66</td><td>-0.70</td><td>72.80</td><td>72.22</td><td>-0.59</td></tr><tr><td rowspan="7">Group 14</td><td>10</td><td>75.72</td><td>75.80</td><td>+0.08</td><td>75.80</td><td>75.76</td><td>-0.04</td><td>75.53</td><td>75.77</td><td>+0.24</td></tr><tr><td>20</td><td>75.40</td><td>75.65</td><td>+0.25</td><td>75.56</td><td>75.73</td><td>+0.17</td><td>75.32</td><td>75.67</td><td>+0.35</td></tr><tr><td>30</td><td>75.06</td><td>75.51</td><td>+0.45</td><td>75.34</td><td>75.64</td><td>+0.30</td><td>75.06</td><td>75.49</td><td>+0.44</td></tr><tr><td>40</td><td>74.45</td><td>75.15</td><td>+0.70</td><td>74.97</td><td>75.24</td><td>+0.27</td><td>74.69</td><td>75.22</td><td>+0.53</td></tr><tr><td>50</td><td>73.63</td><td>74.13</td><td>+0.50</td><td>74.67</td><td>74.42</td><td>-0.25</td><td>74.05</td><td>74.18</td><td>+0.12</td></tr><tr><td>60</td><td>72.92</td><td>73.29</td><td>+0.38</td><td>73.90</td><td>73.93</td><td>+0.03</td><td>73.11</td><td>73.46</td><td>+0.34</td></tr><tr><td>70</td><td>72.28</td><td>70.39</td><td>-1.89</td><td>72.95</td><td>72.50</td><td>-0.45</td><td>72.12</td><td>69.76</td><td>-2.36</td></tr><tr><td rowspan="7">Group 15</td><td>10</td><td>75.84</td><td>75.94</td><td>+0.10</td><td>75.68</td><td>75.89</td><td>+0.21</td><td>75.69</td><td>75.81</td><td>+0.13</td></tr><tr><td>20</td><td>75.59</td><td>75.87</td><td>+0.28</td><td>75.60</td><td>75.77</td><td>+0.17</td><td>75.55</td><td>75.76</td><td>+0.22</td></tr><tr><td>30</td><td>75.16</td><td>75.67</td><td>+0.52</td><td>75.41</td><td>75.70</td><td>+0.29</td><td>75.27</td><td>75.75</td><td>+0.48</td></tr><tr><td>40</td><td>74.58</td><td>75.41</td><td>+0.83</td><td>75.15</td><td>75.43</td><td>+0.29</td><td>74.79</td><td>75.42</td><td>+0.63</td></tr><tr><td>50</td><td>73.87</td><td>74.37</td><td>+0.50</td><td>74.80</td><td>74.30</td><td>-0.50</td><td>74.13</td><td>74.37</td><td>+0.25</td></tr><tr><td>60</td><td>72.90</td><td>73.49</td><td>+0.60</td><td>74.30</td><td>73.71</td><td>-0.59</td><td>73.45</td><td>73.82</td><td>+0.37</td></tr><tr><td>70</td><td>71.81</td><td>70.31</td><td>-1.50</td><td>73.70</td><td>71.03</td><td>-2.67</td><td>71.91</td><td>70.89</td><td>-1.02</td></tr><tr><td rowspan="7">Group 16</td><td>10</td><td>75.71</td><td>75.83</td><td>+0.12</td><td>75.75</td><td>75.90</td><td>+0.15</td><td>75.63</td><td>75.87</td><td>+0.24</td></tr><tr><td>20</td><td>75.39</td><td>75.68</td><td>+0.29</td><td>75.51</td><td>75.79</td><td>+0.29</td><td>75.48</td><td>75.83</td><td>+0.35</td></tr><tr><td>30</td><td>75.07</td><td>75.49</td><td>+0.43</td><td>75.12</td><td>75.60</td><td>+0.48</td><td>75.08</td><td>75.54</td><td>+0.46</td></tr><tr><td>40</td><td>74.37</td><td>75.18</td><td>+0.81</td><td>74.21</td><td>75.15</td><td>+0.94</td><td>74.74</td><td>75.16</td><td>+0.42</td></tr><tr><td>50</td><td>73.04</td><td>73.99</td><td>+0.95</td><td>72.51</td><td>73.85</td><td>+1.34</td><td>74.01</td><td>74.04</td><td>+0.03</td></tr><tr><td>60</td><td>70.39</td><td>73.31</td><td>+2.92</td><td>68.92</td><td>73.52</td><td>+4.59</td><td>73.51</td><td>73.03</td><td>-0.47</td></tr><tr><td>70</td><td>67.94</td><td>70.54</td><td>+2.60</td><td>64.13</td><td>70.42</td><td>+6.29</td><td>72.07</td><td>69.05</td><td>-3.02</td></tr><tr><td rowspan="7">Group 17</td><td>10</td><td>75.59</td><td>75.84</td><td>+0.25</td><td>75.74</td><td>75.79</td><td>+0.04</td><td>75.83</td><td>75.94</td><td>+0.12</td></tr><tr><td>20</td><td>75.31</td><td>75.84</td><td>+0.53</td><td>75.57</td><td>75.72</td><td>+0.15</td><td>75.55</td><td>75.86</td><td>+0.31</td></tr><tr><td>30</td><td>74.68</td><td>75.61</td><td>+0.93</td><td>75.28</td><td>75.72</td><td>+0.44</td><td>74.90</td><td>75.67</td><td>+0.77</td></tr><tr><td>40</td><td>73.20</td><td>75.25</td><td>+2.05</td><td>74.81</td><td>75.40</td><td>+0.59</td><td>73.59</td><td>75.07</td><td>+1.48</td></tr><tr><td>50</td><td>70.32</td><td>73.60</td><td>+3.28</td><td>74.25</td><td>74.08</td><td>-0.17</td><td>70.47</td><td>73.10</td><td>+2.63</td></tr><tr><td>60</td><td>65.56</td><td>72.76</td><td>+7.20</td><td>73.45</td><td>72.87</td><td>-0.58</td><td>65.95</td><td>71.47</td><td>+5.52</td></tr><tr><td>70</td><td>56.26</td><td>67.58</td><td>+11.33</td><td>71.67</td><td>68.94</td><td>-2.73</td><td>59.27</td><td>62.87</td><td>+3.60</td></tr></table>