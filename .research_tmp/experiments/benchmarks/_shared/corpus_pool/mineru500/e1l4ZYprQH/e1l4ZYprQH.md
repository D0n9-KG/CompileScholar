# MathNAS: If Blocks Have a Role in Mathematical Architecture Design

Qinsi Wang $^{1*}$ Jinghan Ke $^{1*}$ Zhi Liang $^{2}$ Sihai Zhang $^{3,4}$

$^{1}$ University of Science and Technology of China

$^{2}$ School of Life Sciences, University of Science and Technology of China

$^{3}$ Key Laboratory of Wireless-Optical Communications, Chinese Academy of Sciences

$^{4}$ School of Microelectronics, University of Science and Technology of China {wqs, jinghan}@mail.ustc.edu.cn, {liangzhi, shzhang}@ustc.edu.cn

# Abstract

Neural Architecture Search (NAS) has emerged as a favoured method for unearthing effective neural architectures. Recent development of large models has intensified the demand for faster search speeds and more accurate search results. However, designing large models by NAS is challenging due to the dramatical increase of search space and the associated huge performance evaluation cost. Consider a typical modular search space widely used in NAS, in which a neural architecture consists of m block nodes and a block node has n alternative blocks. Facing the space containing $n^{m}$ candidate networks, existing NAS methods attempt to find the best one by searching and evaluating candidate networks directly. Different from the general strategy that takes architecture search as a whole problem, we propose a novel divide-and-conquer strategy by making use of the modular nature of the search space. Here, we introduce MathNAS, a general NAS framework based on mathematical programming. In MathNAS, the performances of the $m * n$ possible building blocks in the search space are calculated first, and then the performance of a network is directly predicted based on the performances of its building blocks. Although estimating block performances involves network training, just as what happens for network performance evaluation in existing NAS methods, predicting network performance is completely training-free and thus extremely fast. In contrast to the $n^{m}$ candidate networks to evaluate in existing NAS methods, which require training and a formidable computational burden, there are only $m * n$ possible blocks to handle in MathNAS. Therefore, our approach effectively reduces the complexity of network performance evaluation. The superiority of MathNAS is validated on multiple large-scale CV and NLP benchmark datasets. Notably on ImageNet-1k, MathNAS achieves 82.5% top-1 accuracy, 1.2% and 0.96% higher than Swin-T and LeViT-256, respectively. In addition, when deployed on mobile devices, MathNAS achieves real-time search and dynamic network switching within 1s (0.4s on TX2 GPU), surpassing baseline dynamic networks in on-device performance. Our code is available at https://github.com/wangqinsi1/MathNAS.

# 1 Introduction

Neural Architecture Search (NAS) has notably excelled in designing efficient models for Computer Vision (CV) [1, 2, 3, 4] and Natural Language Processing (NLP) [5, 6, 7] tasks. With the growing popularity of the Transformer architecture [8, 9], designers are increasingly drawn to using NAS to

develop powerful large-scale models. Many existing NAS studies focus on designing search spaces for large models and conducting searches within them $[10, 11, 12]$ .

However, designing large models by NAS is challenging due to the dramatical increase of search space and the associated huge performance evaluation cost $[10, 13]$ . Consider a widely used modular search space, in which a neural architecture is treated as a topological organization of m different block nodes and each block node has n different block implementations. Obviously, the number of possible networks or neural architectures, $n^{m}$ , grows explosively with n and m. In addition, candidate networks of large models are larger and require more computation for performance evaluation. Therefore, in order to conduct an effective architecture search, a proper search strategy and a suitable performance evaluation method are extremely important.

It is noteworthy that to improve search strategy, recent researches $[14, 15]$ convert NAS to mathematical programming (MP) problems, which substantially decrease the search cost. MP-NAS provides a promising direction for rapidly designing large models. However, current MP-NAS methods exhibit certain architectural constraints. For example, DeepMAD $[14]$ is solely applicable for architecture design within CNN search spaces, and LayerNAS $[15]$ is exclusively suitable for hierarchically ordered search spaces. These limitations impede the application of MP-NAS methods to advanced search spaces, such as SuperTransformer $[13, 5]$ . Besides, alternative strategies for effective performance evaluation of candidate networks are also expected, despite the improvement brought by parameter sharing $[16, 17]$ , performance prediction based on learning curves $[18, 19]$ and so on.

In this study, we introduce MathNAS, a novel MP-NAS framework for universal network architecture search. In contrast to previous studies which estimate the performance of a network by solving a whole problem, MathNAS adopts an alternative divide-and-conquer approach. In brief, MathNAS improves the performance evaluation of a candidate network by estimating the performance of each block of the network first, and then combining them to predict the overall performance of the network. Although estimating block performance involves network training, predicting network performance is completely training-free. Therefore, this approach reduces the complexity of network performance evaluation. MathNAS achieves further improvement of the search strategy by transforming NAS to a programming problem, reducing the search complexity to polynomial time.

MathNAS contains three key steps:

- Block performance estimation: The performance of each block is estimated by the performance difference between networks having and having not that specific block.   
- Network performance prediction: The performance of a network is predicted based on the performances of its blocks.   
- NAS by ILP: Utilizing the block performances, NAS is solved as an Integer Linear Programming (ILP) problem.

We perform experiments on search spaces with various network architectures, including NAS-Bench-201 (GNN) [20], MobileNetV3 (CNN) [21], SuperTransformer (Transformer) [5] and NASViT (CNN+Trans) [13]. Our experiments demonstrate that predicting network performance based on its blocks' performances is applicable to different network architectures. In particular, the Spearman coefficient between the actual and the predicted top-1 indices on four different search spaces achieve 0.97, 0.92, 0.93, and 0.95, respectively. At the same time, by using the merit of the divide-and-conquer strategy to transform NAS into an ILP problem, MathNAS can find models superior to state-of-the-art (SOTA) models across different search spaces and tasks. In CV tasks, MathNAS achieves $82.5\%$ top-1 accuracy on ImageNet-1k with 1.5G FLOPs and 15M parameters, outperforming AutoFormer-small $(81.7\%)$ and LeViT-256 $(81.6\%)$ . In NLP tasks, MathNAS reaches a Blue Score of 28.8, on par with Transformer (28.4), but requiring only 1/5 of the FLOPs. In summary, our contributions are as follows:

1. We propose a general framework for performance evaluation of candidate networks by estimating block performance first and then combining them to predict network performance, which greatly improves the evaluation efficiency.   
2. By virtue of the established mapping between block performance and network performance, we transform NAS into an ILP problem, which reduces the search complexity to polynomial.   
3. We demonstrate MathNAS by considering three key performance indices for network design, i.e. accuracy, latency and energy, and achieve results superior to SOTA models.

# 2 The Proposed Method

Search Space and Notations. In this paper, we consider a widely used modular search space $S = \{N_{1}, ..., N_{k}, ...\}$ , in which network $N_{k}$ consists of m block nodes $\mathcal{B}_{i}(1 \leq i \leq m)$ . For a block node, there is n alternative blocks, i.e. $B_{i} = \{b_{i,1}, ..., b_{i,j}, ..., b_{i,n}\}$ , where block $b_{i,j}(1 \leq i \leq m, 1 \leq j \leq n)$ represents the j-th implementation of the i-th block node. A network can therefore be denoted as $\mathcal{N} = (b_{1,\mathcal{I}_{1}}, ..., b_{i,\mathcal{I}_{i}}, ..., b_{m,\mathcal{I}_{m}})$ , with its i-th block node implemented by block $b_{i,\mathcal{J}_{i}}$ . Totally, there are $m^{n}$ possible networks in the search space. Here, we focus on three key performance indices for network design, i.e. accuracy, latency and energy consumption. The accuracy, latency and energy consumption of network N are denoted as $Acc(\mathcal{N})$ , $Lat(\mathcal{N})$ , $Eng(\mathcal{N})$ , respectively.

# 2.1 Problem Formulation: Reduce NAS Search Complexity from $\mathcal{O}(n^{m})$ to $\mathcal{O}(m*n)$ .

The objective of hardware-aware NAS is to find the best neural architecture $N^{*}$ with the highest accuracy under limited latency $\hat{L}$ and energy consumption $\hat{E}$ in the search space S:

$$
\mathcal {N} ^ {*} = \underset {\mathcal {N} \in \mathcal {S}} {\arg \max} A c c (\mathcal {N}), s. t. L a t (\mathcal {N}) \leq \hat {L}, E n g (\mathcal {N}) \leq \hat {E} \tag {1}
$$

In order to fulfill the above goal, NAS has to search a huge space, evaluate and compare the performance of candidate networks. Early NAS studies usually fully train the candidate networks to obtain their performance ranking, which is prohibitively time-consuming. Subsequent works introduce various acceleration methods. For instance, the candidate networks can avoid training from scratch by sharing weights $[16, 17]$ . Also, the performance of a candidate network can be predicted based on the learning curve obtained from early termination of an incomplete training $[18, 19]$ . Despite of these improvements, a candidate network has to be trained, no matter fully or partially, to obtain a reasonable performance evaluation. And the huge number of candidate networks in the search space poses a formidable efficiency challenge.

However, the modular nature of the search space may provide us with a novel possibility. Although there are $n^m$ candidate networks in the search space, they are all constructed by the $n * m$ blocks. If we can evaluate the performances of the blocks by training of a limited number of networks, and if we can combine these block performance indices to obtain a reasonable performance evaluation of networks, we can reduce the complexity of network performance evaluation from $\mathcal{O}(n^m)$ to $\mathcal{O}(n * m)$ .

Guided by this idea, we reformulate the search of $N^{*}$ from S as a succession of sub-problems. Each sub-problem corresponds to the task of searching the block $b_{i,J_{i}^{*}}$ with the highest accuracy within the block node $B_{i}$ . This approach notably simplifies the original problem:

$$
\mathcal {N} ^ {*} = \left(b _ {1, \mathcal {J} _ {1} ^ {*}}, b _ {2, \mathcal {J} _ {2} ^ {*}}, \dots , b _ {m, \mathcal {J} _ {m} ^ {*}}\right) = \underset {b _ {1, j} \in \mathcal {B} _ {1}} {\arg \max} \left(b _ {1, j} ^ {A}\right) \oplus \underset {b _ {2, j} \in \mathcal {B} _ {2}} {\arg \max} \left(b _ {2, j} ^ {A}\right) \oplus \dots \oplus \underset {b _ {m, j} \in \mathcal {B} _ {m}} {\arg \max} \left(b _ {m, j} ^ {A}\right), \tag {2}
$$

$$
s. t. \sum_ {i = 1} ^ {m} b _ {i, \mathcal {J} _ {i} ^ {*}} ^ {L} \leq \hat {L}, \sum_ {i = 1} ^ {m} b _ {i, \mathcal {J} _ {i} ^ {*}} ^ {E} \leq \hat {E},
$$

where $b_{i,j}^{A}$ , $b_{i,j}^{L}$ , $b_{i,j}^{E}$ represent the accuracy, latency, and energy of block $b_{i,j}$ respectively. $\oplus$ denotes the operation of adding a block to the network. With this approach, each block $b_{i,j}$ is searched only once and the complexity can be effectively reduced to $\mathcal{O}(n*m)$ . However, due to the mutual influence between blocks, a unified understanding of the relationship between the performance of N and its constituent blocks remains elusive, posing a challenge to the application of this method.

# 2.2 Divide-and-Conquer: Network-Block-Network

Consider the switching process $\mathcal{N}_{(i,1)\to(i,\mathcal{J}_{i})}$ , signifying the selection of the i-th B in network N as it switches from $b_{i,1}$ to $b_{i,\mathcal{J}_{i}}$ , with other selected blocks remaining unchanged. Thus, any network $\mathcal{N}=(b_{1,\mathcal{J}_{1}},b_{2,\mathcal{J}_{2}},...,b_{m,\mathcal{J}_{m}})$ can be viewed as the outcome of the base network $\widetilde{\mathcal{N}}=(b_{1,1},b_{2,1},...,b_{m,1})$ undergoing m sequential switching processes. Guided by this idea, we explore two aspects:

# E1: Can block performance be directly calculated as with network performance?

Considering the entire search space S, let us denote the collection of networks with the selected block $b_{i,1}$ as $\mathcal{N}_{(i,1)}^{\Omega}$ , which comprises $n^{m-1}$ networks. For any network $\mathcal{N}_{(i,1)}$ in $\mathcal{N}_{(i,1)}^{\Omega}$ , the switching process $\mathcal{N}_{(i,1)\to(i,j)}$ signifies the selection of the i-th B in network N as it switches from $b_{i,1}$

![](images/1d194e4b0590e1e57c07294d92f3e90856c8742e4f4faea3b73d8bb9fc71e400.jpg)

![](images/18a818098b96b7c2bac7bcfbffcf00e6f04d8db4f7f4cc50c555000854b776ba.jpg)

![](images/951f4d6864986b9a9cb7d5028ce0bfda64f7558042ccac6893705841e67b3c83.jpg)

![](images/cd9411abe6768910a9e819217cf94b86f9e31773080525152426c41e548dd5cf.jpg)

![](images/0d4161a53a24e478fcbd549628dae60507c61cf2edade4d356e5c108b6d42ef2.jpg)

![](images/19fa5a6c418a6a5a6b569c6a7ee35d39356938428ff23685d5ea6004daa00263.jpg)

(a) $\Delta Acc(\mathcal{N}_{(i,1)\to (i,j)})$ vs $\mathcal{F}(\mathcal{N})$ on NAS-Bench-201 (left) and MobileNetV3 (right).   
![](images/6a67dd0b63680984a9f4be1bc8adbad7bfb43c5a65ef0c7c850206286fc614ad.jpg)

![](images/9a85f83ec84f51290f58b1709846efb6079780a0c46386d93eadfa63326e70ad.jpg)

![](images/f47dc2908da2133625be3bb0774a63e3223a67da526005c76ded2dee7de11322.jpg)

![](images/84533dcc3dca1613cd51f01ac7fbaf7806b6a753b39eb8be7c008bb7684e4cff.jpg)

![](images/c9afe6273ce17e03f5cc93377bec3d065a466b9dff01c0bc8c3e76bbdfaebf77.jpg)

![](images/88d766824e71ccadda9b91d88798f4f6fc741aa0c568a47f0f6ceed5b384b3ae.jpg)  
(b) $\Delta Lat(\mathcal{N}_{(i,1)\to (i,j)})$ (left)/ $\Delta Eng(\mathcal{N}_{(i,1)\to (i,j)})$ (right) vs $\mathcal{F}(\mathcal{N})$ in NAS-Bench-201 on edgegpu.   
Figure 1: The relationship between $\Delta Acc(\mathcal{N}_{(i,1)\to (i,j)})$ , $\Delta Lat(\mathcal{N}_{(i,1)\to (i,j)})$ , $\Delta Eng(\mathcal{N}_{(i,1)\to (i,j)})$ and $\mathcal{F}(\mathcal{N})$ . We conduct experiments on two search spaces: (1) NAS-Bench-201 [20]. Graph Network. We use the accuracy obtained on ImageNet after training each network independently. We count the accuracy of all networks in the search space. (2) MobileNetv3 [21]: sequentially connected CNN network. We use the accuracy obtained by the network with shared weights. We sample 5 blocks per block node and count the accuracies of 3125 subnetworks on the ImageNet validation set.

to $b_{i,j}$ , with other selected blocks remaining unchanged. In this switch, two changes occur. The first change sees the selected block in $B_{i}$ switching from $b_{i,1}$ to $b_{i,j}$ . The second change arises from an internal adjustment in $B_{i}$ , modifying its interactions with other block spaces in the network. These changes lead to a difference in performance between $\mathcal{N}_{(i,1)}$ and $\mathcal{N}_{(i,j)}$ , denote as $\Delta Acc(\mathcal{N}_{(i,1)\to(i,j)})$ , $\Delta Lat(\mathcal{N}_{(i,1)\to(i,j)})$ , $\Delta Eng(\mathcal{N}_{(i,1)\to(i,j)})$ . By averaging performance difference obtained from all $n^{m-1}$ switching processes, $\mathcal{N}_{(i,1)\to(i,j)}^{\Omega}$ , we can derive two key parameters:

1. $\Delta \phi (\mathcal{B}_{(i,1)\to (i,j)})$ , the change in inherent capability of $\mathcal{B}_i$ .   
2. $\Delta \Phi (\mathcal{B}_{(i,1)\to (i,j)})$ , the change in the interactive capability of $\mathcal{B}_i$ within $S$ .

Accordingly, we define the performances of $b_{i,j}$ as:

$$
b _ {i, j} ^ {A} = \overline {{\Delta A c c \left(\mathcal {N} _ {(i , 1) \rightarrow (i , j)} ^ {\Omega}\right)}} = \Delta \phi (\mathcal {B} _ {(i, 1) \rightarrow (i, j)}) + \Delta \Phi (\mathcal {B} _ {(i, 1) \rightarrow (i, j)}) \tag {3}
$$

Similarly, $b_{i,j}^{L}$ and $b_{i,j}^{E}$ can be calculated employing the same methodology. The unbiased empirical validation and theoretical proof supporting this method can be found in the Appendix.

# E2: How can we predict network performance using block performance?

To accurately derive the performance difference stemming from the switching process, we consider the performance difference that a particular block switch brings to different networks. We performed the identical process $\mathcal{N}_{(i,1)\to (i,\mathcal{J}_i)}$ over a range of networks within $\mathcal{N}_{(i,1)}^{\Omega}$ . The outcome is illustrated in Figure 1, which depicts the relationship between the differences of three performances—latency, energy, and accuracy—and the FLOPs of the network. Our findings reveal that, within the same switching operation, latency and energy differences maintain consistency across different networks, while accuracy differences exhibit an inverse proportionality to the network's FLOPs, $\mathcal{F}(\mathcal{N})$ .

This finding is logical, as networks with smaller FLOPs have lower computational complexity, rendering them more susceptible to block alterations. Conversely, networks with larger FLOPs exhibit higher computational complexity, making them less sensitive to individual block switching. In the Appendix, we proceed to fit the inverse relationship between accuracy difference and network FLOPs using various formulas. The results suggest that the accuracy difference is approximately inversely related to the network's FLOPs. Consequently, we can posit $\Delta Acc(\mathcal{N}_{(i,1)\to (i,\mathcal{J}_i)}) = \alpha * 1 / \mathcal{F}(\mathcal{N})$ , where $\alpha$ represents a specific block's coefficient.

Based on these observations, we can deduce that for any network $\mathcal{N}_{(i,1)}$ within the set $\mathcal{N}_{(i,1)}^{\Omega}$ , the performance difference resulting from the switching process can be approximated as follows:

$$
\Delta L a t (\mathcal {N} _ {(i, 1) \rightarrow (i, j)}) \approx b _ {i, j} ^ {L}, \Delta E n g (\mathcal {N} _ {(i, 1) \rightarrow (i, j)}) \approx b _ {i, j} ^ {E}, \Delta A c c (\mathcal {N} _ {(i, 1) \rightarrow (i, j)}) \approx b _ {i, j} ^ {A} * \frac {\overline {{\mathcal {F} \left(\mathcal {N} _ {(i , j)} ^ {\Omega}\right)}}}{\mathcal {F} (\mathcal {N} _ {(i , j)})} (4)
$$

![](images/a8be0a81e83a1b70bd634ef599fc5609eeeb6c5f3a2edbd32c0618a2ba6e83a8.jpg)

<details>
<summary>scatter</summary>

| Actual Latency(ms) | Predict Latency(ms) |
| ------------------ | ------------------- |
| 4000               | 4000                |
| 5000               | 5000                |
| 6000               | 6000                |
| 7000               | 7000                |
| 8000               | 8000                |
</details>

![](images/c8ce9cd2a62c32eb93bbfd2d283842ba9634d4dbb80c29675a0a592c83cb8112.jpg)

<details>
<summary>scatter</summary>

| Actual Energy(ms) | Predict Energy(ms) |
| ----------------- | ------------------ |
| 15000             | 15000              |
| 20000             | 20000              |
| 25000             | 25000              |
| 30000             | 30000              |
| 35000             | 35000              |
</details>

![](images/2e6551699019761504e48fcf1261b1a7c3ae600559fed7cce1ce17977b969a0c.jpg)

![](images/d73f89bdb798cc08717fca25ea06d3b5ea8f7bb5c20c39975d7becc313ba4017.jpg)

<details>
<summary>scatter</summary>

| Actual topk1 | Predict topk1 |
| ------------ | ------------- |
| 10           | 10            |
| 20           | 20            |
| 30           | 30            |
| 40           | 40            |
| 50           | 50            |
</details>

![](images/3f2e635190d018907b2225b8579d9a51526aea654c46a948968f1c51e05278ee.jpg)  
(a)Lat on NB201

![](images/d774a1d4a0ab22130b58403f88b5fd1e211769b2d2afbeeba0c27625ec995713.jpg)  
(b)Energy on NB201

![](images/6c48004e90a6ab10a6a2f8733c2f65c78708f8d88167802de5ccb54c075ec115.jpg)  
(c)Acc on MBV3

![](images/3707dead3b7c8569a57d3c1ba23356aaaff76e85cf6bbe85842049d943535edd.jpg)  
(d)Acc on NB201   
Figure 2: Validation of the predictive performance of Equation 5 and the effect of sampling rate on it. The setup of the search space is the same as in Figure 1. The first row shows the predictive performance of Equation 5 when calculating $b_{i,j}^{A}$ , $b_{i,j}^{L}$ , and $b_{i,j}^{E}$ with Equation 3. The second row shows the predictive performance of Equation 5 when randomly sampling the network and computing the average difference to estimate $b_{i,j}^{A}$ , $b_{i,j}^{L}$ , and $b_{i,j}^{E}$ for the corresponding dataset.

By integrating Equation 3 and 4, we can estimate the performance of $\mathcal{N} = (b_{1,\mathcal{J}_1}, b_{2,\mathcal{J}_2}, ..., b_{m,\mathcal{J}_m})$ :

$$
L a t (\mathcal {N}) = L a t (\widetilde {\mathcal {N}}) - \sum_ {i = 1} ^ {m} b _ {i, \mathcal {J} _ {i}} ^ {L}, E n g (\mathcal {N}) = E n g (\widetilde {\mathcal {N}}) - \sum_ {i = 1} ^ {m} b _ {i, \mathcal {J} _ {i}} ^ {E}
$$

$$
A c c (\mathcal {N}) = A c c (\widetilde {\mathcal {N}}) - \sum_ {i = 1} ^ {m} b _ {i, \mathcal {J} _ {i}} ^ {A} * \frac {\overline {{{\mathcal {F} \left(\mathcal {N} _ {(i , \mathcal {J} _ {i})} ^ {\Omega}\right)}}}}{\mathcal {F} \left(\mathcal {N} _ {(i , \mathcal {J} _ {i})}\right)} \tag {5}
$$

Proof-of-Concept Experiment. In our examination of the predictive efficacy of Equation 5 across diverse network types, as depicted in Figure 2, we observe that it accurately forecasts three performances for both sequentially connected networks and graph networks and both weight-independent and weight-shared networks, all without necessitating network training. To the best of our knowledge, our method is the first work to precisely estimate network performances using a linear formula, and notably, it is theoretically applicable to all architectures.

# 2.3 Simplification: Single-sampling Strategy Instead of Full-sampling

Despite the outstanding predictive performance displayed by Equation 5, its computation of loss averages proves to be costly. In this section, by employing a single-sample sampling strategy, we effectively reduce the time complexity from $O(n^{m})$ to $O(n*m)$ , enhancing efficiency without compromising precision.

Partial Network Sampling Strategies. We begin by investigating the sample size requirements for Equation 5. The second row of Figure 2 demonstrates rapid convergence of Equation 5 with a notably small sample count for performance prediction. Specifically, Figure 2(c)(d) reveals that a mere 5% random sampling of networks is sufficient for the prediction to converge towards optimal performance. This underscores the impressive efficacy of Equation 5, which exhibits rapid convergence even with a limited number of random samples.

Single Network Sampling Strategy. Building upon Equation 4, we can select an average-FLOPs network, denoted as $\mathcal{N}^{avg}$ , in the search

# Algorithm 1 Math Neural Architecture Search

Stage1: Determine the Average-FLOPs Network for i=1,2,...,m do

$$
\begin{array}{l} \begin{array}{l} \text {Calculate the average of the FLOPs of all blocks at} \\ \text {the i - th note,} \end{array} \\ \overline {{{{\mathcal {F} (b _ {i})}}}} = (\mathcal {F} (b _ {i, 1}) + \dots + \mathcal {F} (b _ {i, n})) / n; \\ \text { Select } b _ {i, \mathcal {J} _ {i}} \text { whose   FLOPs   is   closest   to } \overline {{\mathcal {F} (b _ {i})}}. \\ \end{array}
$$

end

Define average-FLOPs net $\mathcal{N}^{avg} = \{b_1, \mathcal{J}_1, \dots, b_m, \mathcal{J}_m\}$ .

Stage2: Calculate Block Performances

for i=1,2,...,m do

for j=1,2,...,n do

$$
\begin{array}{l} \text {Switch the i - th block in \mathcal {N} ^ {avg} from b_{i,1} to b_{i,j} ;} \\ \text {Calculate the performance difference of the} \\ \text {network brought about by switching and use it} \\ \text {as the performance of the block b_{i,j} ^{A} , b_{i,j} ^{L} , b_{i,j} ^{E}}. \end{array}
$$

end

end

Stage3: Prediction and Architecture Search

Calculate three characteristics of the base net $\widetilde{\mathcal{N}} = \{b_{1,1},\dots ,b_{m,1}\}$ as

$Acc(\widetilde{\mathcal{N}}), Lat(\widetilde{\mathcal{N}}), Eng(\widetilde{\mathcal{N}}).$

For network $\mathcal{N} = \{b_{1,\mathcal{J}_1}, b_2,\mathcal{J}_2, ..., b_m,\mathcal{J}_m\}$ , its accuracy, latency and energy can be estimated by Equa. 5.

Set required accuracy/latency/energy limit, and solve the corresponding ILP problem to obtain the architecture.

![](images/6fbbcc7950042c441f44a02c05a72b98ae00c575d2208658e7bf1733a02f025e.jpg)

<details>
<summary>scatter</summary>

| Actual topk1(%) | Predict topk1(%) |
| --------------- | ---------------- |
| 76              | 76               |
| 77              | 77               |
| 78              | 78               |
| 79              | 79               |
</details>

![](images/a7e9e62357a2ddf028167756203391f419b2742560e1a682622791168d53a946.jpg)

<details>
<summary>scatter</summary>

| Actual topk1 | Predict topk1 | Density |
| ------------ | ------------- | ------- |
| 10           | 10            | 0.002   |
| 20           | 20            | 0.004   |
| 30           | 30            | 0.006   |
| 40           | 40            | 0.008   |
| 50           | 50            | 0.010   |
</details>

![](images/b74cd3614f54e9f19d75330470cbe544a38398af4d6c4d6509473d1680e6ac68.jpg)

<details>
<summary>scatter</summary>

| Actual BLUE | Predict BLUE | DLPY |
| ----------- | ------------ | ---- |
| 24.0        | 24.0         | 0.1  |
| 25.0        | 25.0         | 0.3  |
| 26.0        | 26.0         | 0.5  |
| 27.0        | 27.0         | 0.7  |
</details>

![](images/10681172362a1d1b1f75da73fe16a35c030e595f25bd08332393a1d2ce45963c.jpg)

<details>
<summary>scatter</summary>

| Actual topk1(%) | Predict topk1(%) |
| --------------- | ---------------- |
| 80.5            | 80.5             |
| 81.0            | 81.0             |
| 81.5            | 81.5             |
| 82.0            | 82.0             |
| 82.5            | 82.5             |
</details>

![](images/24e8c54089dde825d84bc2b076194a1c988d38ac5282e1d3830cf29a64a5a0a0.jpg)

<details>
<summary>scatter</summary>

| Actual Latency(ms) | Predict Latency(ms) |
| ------------------ | ------------------- |
| 2000               | 2000                |
| 4000               | 4000                |
| 6000               | 6000                |
| 8000               | 8000                |
</details>

![](images/bff62b881022cee73a10d2b811ab3bf14f5c7bca743e69ffa2a16f07bc02dc87.jpg)

<details>
<summary>scatter</summary>

| Actual Latency(s) | Predict Latency(s) |
| ----------------- | ------------------ |
| 4.0               | 4.0                |
| 5.0               | 5.0                |
| 6.0               | 6.0                |
| 7.0               | 7.0                |
| 8.0               | 8.0                |
</details>

![](images/bfabbab256850dffac0cd0077079c8a4193316680e53a67fd586dcb58e69f4f1.jpg)

<details>
<summary>scatter</summary>

| Actual Energy(J) | Predict Energy(J) |
| ---------------- | ----------------- |
| 20               | 20                |
| 25               | 25                |
| 30               | 30                |
| 35               | 35                |
</details>

![](images/520d32252fed9676695a9ab1c5df82b2b6c7de979f395bf6c9a50ea7d958f978.jpg)

<details>
<summary>scatter</summary>

| Actual Latency(ms) | Predict Latency(ms) |
| ------------------ | ------------------- |
| 3000               | 3000                |
| 4000               | 4000                |
| 5000               | 5000                |
| 6000               | 6000                |
| 7000               | 7000                |
| 8000               | 8000                |
</details>

(e)Lat on MBv3   
(f)Lat on NB201   
(g)Ener on NB201   
(h)Lat on SuperTran   
Figure 3: MathNAS algorithm verification. We conduct experiments on 4 search space. (1) MobileNetV3 [21] (2) NAS-Bench-201 [20] (3) SuperTransformer [5] (4) SuperViT [13]. For NAS-Bench-201, we use the accuracy of each network trained individually as $Acc(\mathcal{N})$ . For other spaces, the validate accuracy under shared weights base net is used. We show accuracy predictions on these four networks as well as hardware efficiency predictions on them. The calculation of $b_{i,j}^{A}, b_{i,j}^{L}$ and $b_{i,j}^{E}$ follows Algorithm 1. For NAS-Bench-201, we verify all nets and other spaces, we randomly sample 1000 nets to verify the prediction effect.

space with FLOPs approximating the mean value, ensuring $\overline{\mathcal{F}\left(\mathcal{N}_{(i,j)}^{\Omega}\right)}/\mathcal{F}(\mathcal{N}_{(i,j)}^{avg})\approx1$ . This leads to: $b_{i,j}^{A}\approx\Delta Acc(\mathcal{N}_{(i,1)\to(i,j)}^{avg})$ , $b_{i,j}^{L}\approx\Delta Lat(\mathcal{N}_{(i,1)\to(i,j)}^{avg})$ , $b_{i,j}^{E}\approx\Delta Eng(\mathcal{N}_{(i,1)\to(i,j)}^{avg})$ . By incorporating $N^{avg}$ into Equation 5, we gain the ability to calculate any network performance. Thus, we only need to verify the performance of $N_{(i,1)}^{avg}$ and $N_{(i,j)}^{avg}$ for $b_{i,j}$ , resulting in an $O(n*m)$ search time complexity. The complete MathNAS algorithm is presented in Algorithm 1, which, in theory, can achieve accurate prediction on any network structure with polynomial search complexity.

Validation experiment of single-sample strategy effectiveness. We assess the efficacy of Algorithm 1 across CNN, GCN, Transformer, and CNN+Transformer search spaces, with the outcomes displayed in Figure 3. It is evident that the Spearman correlation coefficient between predicted and actual values exceeds 0.9 for all networks. Remarkably, accuracies of 0.95 and 0.97 are attained on ViT and CNN architectures, respectively, emphasizing the algorithm's robustness.

# 2.4 MathNAS: Converting Architecture Search to ILP.

We denote $b_{i,j}^{F}$ as the FLOPs of block $b_{i,j}$ , and $b_{i,j}^{B} \in \{0,1\}$ as the indicator of whether block $b_{i,j}$ is used in a network. If block $b_{i,j}$ is selected as the implementation of block node $B_{i}$ in a network, $b_{i,j}^{B} = 1$ , otherwise 0. The problem that NAS needs to solve can be formulated as:

$$
\max _ {b ^ {B}} A c c (\widetilde {\mathcal {N}}) - \frac {\sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i , j} ^ {A} * b _ {i , j} ^ {B}}{\sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i , j} ^ {F} * b _ {i , j} ^ {B}} * \overline {{\mathcal {F} (\mathcal {N})}}
$$

$$
s. t. L a t (\widetilde {\mathcal {N}}) - \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i, j} ^ {L} * b _ {i, j} ^ {B} \leq \hat {L}, E n g (\widetilde {\mathcal {N}}) - \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i, j} ^ {E} * b _ {i, j} ^ {B} \leq \hat {E}, \tag {6}
$$

$$
\sum_ {j = 1} ^ {n} b _ {i, j} ^ {B} = 1, b _ {i, j} ^ {B} \in \{0, 1 \}, \forall 1 \leq i \leq m.
$$

The objective is to obtain the maximum accuracy network under two constraints. First, the latency and energy cannot exceed the limit. Second, for any block node, only one block is used. As Equation 9 is a fractional objective function, it can be transformed into an ILP problem by variable substitution.

# 3 Performance Evaluation

Experiments were conducted at three levels to evaluate the efficacy of MathNAS. Firstly, we validated the effectiveness of MathNAS on CV tasks by conducting searches across three different search spaces. Then we employed MathNAS to design efficient architectures for NLP tasks, showcasing its remarkable generalization capabilities. Finally, we leveraged MathNAS to perform real-time searches on edge devices, considering hardware resources, and achieved exceptional on-device performance.

# 3.1 Experimental Setup

Search space. For CV tasks, we validate our method on three search spaces: (1) NAS-Bench-201 [20]: a search space encompasses 15,625 architectures in a DARTS-like configuration. (2) SuperViT [13]: a hybrid search space that combines ViT and CNN, containing approximately $4 \times 10^{10}$ network architectures. (3) MobileNetV3 [21]: a lightweight network search space comprising about $10^{10}$ network architectures. For NLP tasks, we validate our approach on the SuperTransformer search space [5], which includes $10^{15}$ networks within a lightweight Transformer framework.

Search and training settings. For NAS-Bench-201 and MobileNetV3, we adopt the training methodology employed in $[16]$ and $[22]$ to train the base net for 100 epochs. Subsequently, we conducted MathNAS search on the base net. As for SuperTransformer and SuperViT, we adhere to the training algorithm proposed by $[13]$ to train the base net for 100 epochs before conducting MathNAS search. The settings of hyperparameters in the training are consistent with the original paper. We employ the Gurobipy solver to address the ILP problem. In the SuperViT and SuperTransformer search spaces, we impose a search time limit of 10 seconds to expedite the process. For the other search spaces, we do not enforce any time constraints.

The search cost of MathNAS consists of two stages: offline network pre-training that is conducted only once and online real-time search. During the offline network pre-training, MathNAS evaluates block performance once. During online searching, MathNAS is capable of multiple real-time searches based on the current hardware resource constraints. To negate the influence of variations in GPU models and versions on the pre-training time, and to facilitate comparisons by future researchers, we have adopted pre-trained networks provided by existing works. All mentions of search cost in the paper refer solely to the real-time search time on edge devices. We provide a detailed description of the search space, more experimental results, and visualizations of the searched architectures in the Appendix.

# 3.2 MathNAS for Designing Effective CV Networks

MathNAS for NAS-Bench-201. To assess the effectiveness of our method in striking a balance between accuracy and hardware efficiency, we compare networks searched by MathNAS under hardware efficiency constraints to those searched by BRP-NAS [30], which utilizes GNN predictors to estimate network performance. As illustrated in Figure 4, MathNAS consistently locates networks that approach Pareto optimality in the majority of cases, whereas the employment of GNN predictors leads to suboptimal model choices. An extensive comparison between the searched architectures and SOTA models is provided in the Appendix for further insight.

![](images/8e69a32191702e854bbb28711d076a63d2ce8cf03d45162f2b785ac13c56a430.jpg)

<details>
<summary>scatter</summary>

| Latency on FPGA(ms) | Accuracy(%) | Method     |
| ------------------- | ----------- | ---------- |
| 500                 | 65          | oracle     |
| 500                 | 64          | MathNAS    |
| 500                 | 63          | BRP-NAS    |
| 1000                | 68          | oracle     |
| 1000                | 67          | MathNAS    |
| 1000                | 66          | BRP-NAS    |
| 1500                | 70          | oracle     |
| 1500                | 69          | MathNAS    |
| 1500                | 68          | BRP-NAS    |
| 2000                | 72          | oracle     |
| 2000                | 71          | MathNAS    |
| 2000                | 70          | BRP-NAS    |
</details>

![](images/94683d673f1b6ef3de64a73103d35fb0f5eecffaed6f9dd3fac76d4fc6316803.jpg)

<details>
<summary>scatter</summary>

| Energy on FPGA(mJ) | Oracle Accuracy(%) | MathNAS Accuracy(%) | BRP-NAS Accuracy(%) |
| ------------------ | ------------------ | ------------------- | ------------------- |
| 4000               | 65.0               | 65.0                | 65.0                |
| 5000               | 67.0               | 67.0                | 67.0                |
| 6000               | 69.0               | 69.0                | 69.0                |
| 7000               | 71.0               | 71.0                | 71.0                |
| 8000               | 72.0               | 72.0                | 72.0                |
| 9000               | 73.0               | 73.0                | 73.0                |
| 10000              | 74.0               | 74.0                | 74.0                |
| 11000              | 75.0               | 75.0                | 75.0                |
| 12000              | 76.0               | 76.0                | 76.0                |
| 13000              | 77.0               | 77.0                | 77.0                |
| 14000              | 78.0               | 78.0                | 78.0                |
| 15000              | 79.0               | 79.0                | 79.0                |
| 16000              | 80.0               | 80.0                | 80.0                |
</details>

![](images/a67e72645ae4a35d121305cea31f1f734c6340433afb51b5127d2f06e66bf7a3.jpg)

<details>
<summary>scatter</summary>

| Latency on edgegpu(ms) | Accuracy(%) | Method     |
| ---------------------- | ----------- | ---------- |
| 5000                   | 67          | oracle     |
| 5000                   | 68          | MathNAS    |
| 5000                   | 69          | BRP-NAS    |
| 6000                   | 69          | oracle     |
| 6000                   | 70          | MathNAS    |
| 6000                   | 71          | BRP-NAS    |
| 7000                   | 71          | oracle     |
| 7000                   | 72          | MathNAS    |
| 7000                   | 73          | BRP-NAS    |
</details>

![](images/a0078a92f6f079f90e394341bc78c214f36fe5b9a4eada87fc4c57eb6c963948.jpg)

<details>
<summary>scatter</summary>

| Energy on edgegpu(mJ) | Accuracy(%) | Method     |
| --------------------- | ----------- | ---------- |
| 20000                 | 68          | oracle     |
| 22500                 | 69          | oracle     |
| 25000                 | 70          | oracle     |
| 27500                 | 71          | oracle     |
| 30000                 | 72          | oracle     |
| 32500                 | 71          | oracle     |
| 35000                 | 70          | oracle     |
| 25000                 | 69          | MathNAS    |
| 27500                 | 70          | MathNAS    |
| 30000                 | 71          | MathNAS    |
| 32500                 | 72          | MathNAS    |
| 35000                 | 71          | MathNAS    |
| 25000                 | 68          | BRP-NAS    |
| 27500                 | 69          | BRP-NAS    |
| 30000                 | 70          | BRP-NAS    |
| 32500                 | 71          | BRP-NAS    |
| 35000                 | 72          | BRP-NAS    |
</details>

Figure 4: A comparison of networks searched by MathNAS (red circles) versus those searched by BRP-NAS (brown triangles) in the NAS-Bench-201 space. Across different devices, the networks searched by MathNAS demonstrate a closer proximity to the Pareto front (yellow five-pointed stars) as compared to the networks obtained through BRP-NAS.

![](images/ec601d643ffb4395f7b542b907b576b5a0f5ba04df1539d274812df7952523b4.jpg)

<details>
<summary>line</summary>

| Model           | FLOPs(M) | ImageNet Top-1 Accuracy(%) |
| --------------- | -------- | -------------------------- |
| LeViT           | 250      | 76.5                       |
| LeViT           | 500      | 78.5                       |
| LeViT           | 750      | 80.0                       |
| LeViT           | 1000     | 81.3                       |
| LeViT           | 1500     | 82.0                       |
| FBNetV3         | 250      | 79.0                       |
| FBNetV3         | 500      | 80.5                       |
| FBNetV3         | 750      | 81.3                       |
| FBNetV3         | 1000     | 82.0                       |
| FBNetV3         | 1500     | 82.5                       |
| BigNAS          | 250      | 76.5                       |
| BigNAS          | 500      | 79.0                       |
| BigNAS          | 750      | 80.5                       |
| BigNAS          | 1000     | 81.3                       |
| BigNAS          | 1500     | 82.0                       |
| EfficientNet    | 250      | 76.5                       |
| EfficientNet    | 500      | 79.0                       |
| EfficientNet    | 750      | 80.5                       |
| EfficientNet    | 1000     | 81.3                       |
| EfficientNet    | 1500     | 82.0                       |
| Mobile-Former   | 250      | 76.5                       |
| Mobile-Former   | 500      | 79.0                       |
| Mobile-Former   | 750      | 80.5                       |
| Mobile-Former   | 1000     | 81.3                       |
| Mobile-Former   | 1500     | 82.0                       |
| OnceForAll      | 250      | 76.5                       |
| OnceForAll      | 500      | 79.0                       |
| OnceForAll      | 750      | 80.5                       |
| OnceForAll      | 1000     | 81.3                       |
| OnceForAll      | 1500     | 82.0                       |
| ReXNet          | 250      | 76.5                       |
| ReXNet          | 500      | 79.0                       |
| ReXNet          | 750      | 80.5                       |
| ReXNet          | 1000     | 81.3                       |
| ReXNet          | 1500     | 82.0                       |
| DeepMAD         | 250      | 76.5                       |
| DeepMAD         | 500      | 79.0                       |
| DeepMAD         | 750      | 80.5                       |
| DeepMAD         | 1000     | 81.3                       |
| DeepMAD         | 1500     | 82.0                       |
| LayerNAS        | 250      | 76.5                       |
| LayerNAS        | 500      | 79.0                       |
| LayerNAS        | 750      | 80.5                       |
| LayerNAS        | 1000     | 81.3                       |
| LayerNAS        | 1500     | 82.0                       |
| Ours(MathNAS-T) | 250      | 78.5                       |
| Ours(MathNAS-T) | 500      | 81.3                       |
| Ours(MathNAS-T) | 750      | 82.0                       |
| Ours(MathNAS-T) | 1000     | 82.5                       |
| Ours(MathNAS-T) | 1500     | 82.5                       |
</details>

<table><tr><td>Model</td><td>Top-1(%)</td><td>Top-5(%)</td><td> $\mathcal{F}$ (M)</td><td>Param(M)</td></tr><tr><td>ResNet-18[23]</td><td>77.7</td><td>93.2</td><td>1800</td><td>11.7</td></tr><tr><td>ReXNet[24]</td><td>77.9</td><td>93.9</td><td>400</td><td>4.8</td></tr><tr><td>MathNAS-T1</td><td>78.4</td><td>93.5</td><td>200</td><td>8.9</td></tr><tr><td>LeViT-128[25]</td><td>78.6</td><td>94.0</td><td>406</td><td>9.2</td></tr><tr><td>EfficientNet-B1[26]</td><td>79.1</td><td>94.4</td><td>700</td><td>7.8</td></tr><tr><td>ReXNet[24]</td><td>79.5</td><td>94.7</td><td>660</td><td>7.6</td></tr><tr><td>MathNAS-T2</td><td>79.6</td><td>94.3</td><td>325</td><td>9.3</td></tr><tr><td>ResNetY-4G[27]</td><td>80.0</td><td>94.8</td><td>4000</td><td>21</td></tr><tr><td>LeViT-192[25]</td><td>80.0</td><td>94.7</td><td>658</td><td>10.9</td></tr><tr><td>EfficientNet-B2[26]</td><td>80.1</td><td>94.9</td><td>1000</td><td>9.2</td></tr><tr><td>ReXNet[24]</td><td>80.3</td><td>95.2</td><td>900</td><td>9.7</td></tr><tr><td>ResNet-50[23]</td><td>80.6</td><td>95.1</td><td>4100</td><td>25.6</td></tr><tr><td>MathNAS-T3</td><td>81.3</td><td>95.1</td><td>671</td><td>13.6</td></tr><tr><td>Swin-T[28]</td><td>81.3</td><td>95.5</td><td>4500</td><td>29</td></tr><tr><td>EfficientNet-B3[26]</td><td>81.6</td><td>95.7</td><td>1800</td><td>12</td></tr><tr><td>LeViT-256[25]</td><td>81.6</td><td>95.4</td><td>1120</td><td>18.9</td></tr><tr><td>ReXNet[24]</td><td>81.6</td><td>95.7</td><td>1500</td><td>16</td></tr><tr><td>AutoFormer-small[29]</td><td>81.7</td><td>95.7</td><td>5100</td><td>22.9</td></tr><tr><td>MathNAS-T4</td><td>82.0</td><td>95.7</td><td>1101</td><td>14.4</td></tr><tr><td>AutoFormer-base[29]</td><td>82.4</td><td>95.7</td><td>11000</td><td>54</td></tr><tr><td>LeViT-384[25]</td><td>82.6</td><td>96.0</td><td>2353</td><td>39.1</td></tr><tr><td>MathNAS-T5</td><td>82.5</td><td>95.8</td><td>1476</td><td>14.8</td></tr></table>

Figure 5: MathNAS v.s. SOTA ViT and CNN models on ImageNet-1K.

MathNAS for ViT. To assess the performance of MathNAS in designing larger models, we utilize it to create effective ViT models for ImageNet-1K classification. Figure 5 demonstrates that MathNAS surpasses or matches the performance of existing SOTA models. For instance, MathNAS-T5 achieves an accuracy of $82.5\%$ , which is comparable to LeViT-384 [25] and Autoformer-base [29], while consuming only about $50\%$ and $15\%$ of their respective FLOPs. Similarly, MathNAS-T3 achieves comparable accuracy to RexNet [24] but with approximately half the FLOPs. MathNAS also exhibits exceptional performance in the realm of small networks. Particularly, MathNAS-T1 achieves a top-1 accuracy of $78.4\%$ , surpassing ResNet-18 [23] and ReXNet [24] by $0.7\%$ and $0.5\%$ respectively.

MathNAS for Mobile CNNs. We employ MathNAS to design mobile CNN models for further investigation, conducting our search within the MobileNetV3 search space. As demonstrated in Table 1, MathNAS-MB4 achieves a top-1 accuracy of 79.2%, which is on par with EfficientNet-B1 (79.1%). It is important to note that EfficientNet-B1 is derived through a brute-force grid search, necessitating approximately 72,000 GPU hours [4]. Despite this, MathNAS-MB4 offers comparable performance to EfficientNet-B1 while only requiring 0.8 seconds to solve an ILP problem on the GPU and search for a suitable network. MathNAS also excels in the context of smaller networks. Notably, MathNAS-MB1 requires only 257M FLOPs to achieve a top-1 accuracy of 75.9%, surpassing the performance of FBNet-b [31], AtomNAS-A [32], and OFA [22], all of which demand higher computational resources.

Table 1: Performance of mobile networks designed with MathNAS. Top-1 accuracy on ImageNet-1K. 

<table><tr><td>Model</td><td>FLOPs(M)</td><td>Top-1</td><td>Search Time</td><td>Scale Up</td></tr><tr><td>FBNet-b[31]</td><td>295</td><td>74.1</td><td>609h</td><td> $1.9 \times$ </td></tr><tr><td>AtomNAS-A[32]</td><td>258</td><td>74.6</td><td>492h</td><td> $2.3 \times$ </td></tr><tr><td>OFA[22]</td><td>301</td><td>74.6</td><td>120h</td><td> $9.6 \times$ </td></tr><tr><td>MathNAS-MB1</td><td>257</td><td>75.9</td><td>0.9s</td><td> $4.6M \times$ </td></tr><tr><td>MnasNet-A1[33]</td><td>312</td><td>75.2</td><td>40025h</td><td> $1.0 \times$ </td></tr><tr><td>ProxylessNAS-R[34]</td><td>320</td><td>74.6</td><td>520h</td><td> $76.9 \times$ </td></tr><tr><td>AtomNAS-B[32]</td><td>326</td><td>75.5</td><td>492h</td><td> $81.4 \times$ </td></tr><tr><td>FairNAS-C[35]</td><td>321</td><td>71.1</td><td>384h</td><td> $104.2 \times$ </td></tr><tr><td>Single Path One-Shot[36]</td><td>323</td><td>74.4</td><td>288h</td><td> $138.9 \times$ </td></tr><tr><td>OFA[22]</td><td>349</td><td>75.8</td><td>120h</td><td> $333.5 \times$ </td></tr><tr><td>MathNAS-MB2</td><td>289</td><td>76.4</td><td>1.2s</td><td> $144M \times$ </td></tr><tr><td>EfficientNet B0[4]</td><td>390</td><td>76.3</td><td>72000h</td><td> $1.0 \times$ </td></tr><tr><td>FBNet-c[31]</td><td>375</td><td>74.9</td><td>580h</td><td> $124.1 \times$ </td></tr><tr><td>ProxylessNAS-GPU[34]</td><td>465</td><td>75.1</td><td>516h</td><td> $139.5 \times$ </td></tr><tr><td>AtomNAS-C[32]</td><td>363</td><td>76.3</td><td>492h</td><td> $146.3 \times$ </td></tr><tr><td>FairNAS-A[35]</td><td>388</td><td>75.3</td><td>384h</td><td> $104.2 \times$ </td></tr><tr><td>FairNAS-B[35]</td><td>345</td><td>75.1</td><td>384h</td><td> $104.2 \times$ </td></tr><tr><td>MathNAS-MB3</td><td>336</td><td>78.2</td><td>1.5s</td><td> $173M \times$ </td></tr><tr><td>EfficientNet B1[4]</td><td>700</td><td>79.1</td><td>72000</td><td> $1.0 \times$ </td></tr><tr><td>MnasNetA1[33]</td><td>532</td><td>75.4</td><td>40025</td><td> $1.8 \times$ </td></tr><tr><td>BigNAS-M[37]</td><td>418</td><td>78.9</td><td>1152</td><td> $62.5 \times$ </td></tr><tr><td>MathNAS-MB4</td><td>669</td><td>79.2</td><td>0.8s</td><td> $324M \times$ </td></tr></table>

# 3.3 MathNAS for Designing Effective NLP Networks

We perform a comparative evaluation of MathNAS against SOTA NLP models on the WMT'14 En-De task to gauge its effectiveness. Table 2 reveals that MathNAS surpasses all baseline models in terms of BLEU score while also achieving FLOPs reduction across three distinct devices. Specifically, on Intel Xeon CPUs, MathNAS with full precision attains a remarkable $74\%$ reduction in FLOPs compared to Transformer [8] and a $23\%$ reduction compared to HAT [5], while registering improved BLEU scores by 0.4 and 0.3, respectively. Additionally, MathNAS excels in designing lightweight

Table 2: MathNAS vs. SOTA baselines in terms of accuracy and efficiency on NLP tasks. 

<table><tr><td></td><td colspan="3">Raspberry Pi</td><td colspan="3">Intel Xeon CPU</td><td colspan="3">Nvidia TITAN Xp GPU</td><td colspan="2">Search Cost</td></tr><tr><td>Model</td><td>FLOPs</td><td>BLEU</td><td>Latency</td><td>FLOPs</td><td>BLEU</td><td>Latency</td><td>FLOPs</td><td>BLEU</td><td>Latency</td><td>Time</td><td> $CO_2$ </td></tr><tr><td>Transformer[8]</td><td>10.6G</td><td>28.4</td><td>20.5s</td><td>10.6G</td><td>28.4</td><td>808ms</td><td>10.6G</td><td>28.4</td><td>334ms</td><td>184h</td><td>52lbs</td></tr><tr><td>Evolved Trans.[38]</td><td>2.9G</td><td>28.2</td><td>7.6s</td><td>2.9G</td><td>28.2</td><td>300ms</td><td>2.9G</td><td>28.2</td><td>124ms</td><td>219200h</td><td>624000lbs</td></tr><tr><td>HAT[5]</td><td>1.5G</td><td>25.8</td><td>3.5s</td><td>1.9G</td><td>25.8</td><td>138ms</td><td>1.9G</td><td>25.6</td><td>57ms</td><td>200h</td><td>57lbs</td></tr><tr><td>MathNAS</td><td>1.7G</td><td>25.5</td><td>3.2s</td><td>1.8G</td><td>25.9</td><td>136ms</td><td>1.8G</td><td>25.9</td><td>68ms</td><td>10s</td><td>0.0008lbs</td></tr><tr><td>HAT[5]</td><td>2.3G</td><td>27.8</td><td>5.0s</td><td>2.5G</td><td>27.9</td><td>279ms</td><td>2.5G</td><td>27.9</td><td>126ms</td><td>200h</td><td>57lbs</td></tr><tr><td>MathNAS</td><td>2.1G</td><td>28.3</td><td>4.7s</td><td>2.4G</td><td>28.6</td><td>272ms</td><td>2.0G</td><td>28.1</td><td>107ms</td><td>10s</td><td>0.0008lbs</td></tr><tr><td>HAT[5]</td><td>3.0G</td><td>28.4</td><td>7.0s</td><td>3.5G</td><td>28.5</td><td>384ms</td><td>3.1G</td><td>28.5</td><td>208ms</td><td>200h</td><td>57lbs</td></tr><tr><td>MathNAS</td><td>2.8G</td><td>28.6</td><td>6.5s</td><td>2.8G</td><td>28.8</td><td>336ms</td><td>2.6G</td><td>28.6</td><td>189ms</td><td>10s</td><td>0.0008lbs</td></tr></table>

NLP models. On Nvidia TITAN Xp GPUs under latency constraints, MathNAS yields FLOPs comparable to HAT [5], but with a 0.3 higher BLEU score. A noteworthy aspect is that the network search process facilitated by MathNAS requires only 10 seconds, considerably reducing search time. As a result, employing MathNAS leads to a reduction of over $99\%$ in $CO_{2}$ emissions compared to baseline models, underscoring its positive environmental impact.

# 3.4 MathNAS for Designing Dynamic Networks.

Deployed on edge devices (Raspberry Pi 4b, Jetson TX2 CPU, TX2 GPU), MathNAS allows dynamic network switching suited to device conditions. Within the MobileNetV3 search space, we account for memory limitations by calculating performance indices for each block, subsequently deploying five selected Pareto-optimal blocks balancing accuracy and latency in each block node. During runtime, latency is continuously monitored. Should it surpass a preset threshold, MathNAS immediately updates the blocks' latency. Then the device solves the ILP problem to identify the optimal network architecture, comparing and switching blocks with the searched network as required.

In comparison with SOTA dynamic network models, MathNAS demonstrates superior performance as outlined in Table 3 and Figure 6. Impressively, MathNAS solves the ILP problem on-device in a mere 0.4 seconds on the TX2 GPU, enabling real-time search. This notably enhances the number of executable networks on the device, outdoing SlimmableNet [39] and USlimmableNet [40] by factors of 781 and 116 respectively. Additionally, through a block-based approach, MathNAS enables efficient network alterations by replacing only essential blocks. When compared to Dynamic-OFA [41], which shares similar performance, MathNAS significantly reduces the switching time by $80\%$ . The Appendix details the use of Pareto-optimal blocks and related network experiment results.

Table 3: MathNAS vs. SOTA baselines in terms of Dynamic Networks. 

<table><tr><td></td><td colspan="2">Network</td><td colspan="3">Latency (ms)</td><td colspan="6">On Device Performance</td></tr><tr><td>Model</td><td>Top-1 (%)</td><td>FLOPs (M)</td><td>Raspb Pi</td><td>TX2 CPU</td><td>TX2 GPU</td><td>Search Method</td><td>Search Time</td><td>Switch Unit</td><td>Switch Time</td><td>Nets Number</td><td>Scale Up</td></tr><tr><td>S-MbNet-v2[39]</td><td>70.5</td><td>301</td><td>1346</td><td>958</td><td>118</td><td>Manual Design</td><td>-</td><td>Channel</td><td>15ms</td><td>4</td><td>1.0x</td></tr><tr><td>US-MbNet-v2[40]</td><td>71.5</td><td>300</td><td>1358</td><td>959</td><td>158</td><td>Manual Design</td><td>-</td><td>Channel</td><td>18ms</td><td>27</td><td>6.7x</td></tr><tr><td>AS-MNASNet[42]</td><td>75.4</td><td>532</td><td>2097</td><td>1457</td><td>2097</td><td>Greedy Slimming</td><td>4000h</td><td>Channel</td><td>37ms</td><td>4</td><td>1.0x</td></tr><tr><td>Dynamic-OFA[41]</td><td>78.1</td><td>732</td><td>2404</td><td>1485</td><td>80</td><td>Random+Evplution</td><td>35h</td><td>Network</td><td>244ms</td><td>7</td><td>1.7x</td></tr><tr><td rowspan="2">MathNAS</td><td>75.9</td><td>257</td><td>832</td><td>525</td><td>76</td><td rowspan="2">On-Device Search</td><td rowspan="2">0.4-12s</td><td rowspan="2">Block</td><td rowspan="2">61ms</td><td rowspan="2">3125</td><td rowspan="2">781x</td></tr><tr><td>79.2</td><td>669</td><td>2253</td><td>1398</td><td>81</td></tr></table>

![](images/7f09bd0338d11e93999b868fcada1c9ac9db68bdd1e189daa1f27cd45df9bf3b.jpg)

<details>
<summary>line</summary>

| Latency(ms) | Slimmable MBv2 | US-Slim MBv2 | AutoSlim MnasNet | DynamicOFA | Ours(MathNAS-MB) |
| ----------- | -------------- | ------------ | ---------------- | ---------- | ---------------- |
| 500         | 60             | 62           | 63               | 73         | 75               |
| 1000        | 68             | 70           | 72               | 75         | 77               |
| 1500        | 70             | 72           | 74               | 76         | 78               |
| 2000        | 72             | 74           | 75               | 77         | 79               |
| 2500        | 74             | 76           | 76               | 78         | 80               |
</details>

![](images/1c02cbe67e2edc6a3fbe50eab8c197b1cfc3c039f7ba29b3fcb1befb315aab70.jpg)

<details>
<summary>line</summary>

| Latency(ms) | Slimmable MBv2 | US-Slim MBv2 | AutoSlim MnasNet | DynamicOFA | Ours(MathNAS-MB) |
| ----------- | -------------- | ------------ | ---------------- | ---------- | ---------------- |
| 500         | 60             | 65           | 65               | 70         | 75               |
| 1000        | 70             | 75           | 75               | 75         | 80               |
| 1500        | 75             | 80           | 80               | 80         | 80               |
</details>

![](images/0cb4c2dfd22b30a926e769a65d5cff3ec1c6ff36b1224743e0ebb5fab315e3f1.jpg)

<details>
<summary>line</summary>

| Latency(ms) | Slimmable MBv2 | US-Slim MBv2 | AutoSlim MnasNet | DynamicOFA | Ours(MathNAS-MB) |
| ----------- | -------------- | ------------ | ---------------- | ---------- | ----------------- |
| 50          | 60             | 75           | 75               | 75         | 75                |
| 75          | 65             | 75           | 75               | 75         | 75                |
| 100         | 70             | 75           | 75               | 75         | 75                |
| 125         | 70             | 75           | 75               | 75         | 75                |
| 150         | 70             | 75           | 75               | 75         | 75                |
</details>

![](images/6c4f07f7358c7c75d9038f43fb19b807750e8ec816be8bd5622c9fba268dc953.jpg)

<details>
<summary>bar</summary>

| Method    | Networks Search Time (log(hours)) | Subnets Number |
|-----------|------------------------------------|----------------|
| D-OFA     | 10^4                               | ×4             |
| AutoSlim  | 10^4                               | ×7             |
| Ours      | 10^4                               | ×3125          |
</details>

Figure 6: Top-1 vs. Latency of MathNAS over SOTA dynamic baselines on three devices.

# 4 Conclusion

This paper introduces MathNAS, the first architecturally-general MP-NAS. By virtue of the modular nature of the search space, we introduce block performance and establish the mapping from block performance and network performance, which enables subsequent transformation of NAS to an ILP problem. This novel strategy reduces network search complexity from exponential to polynomial levels while maintaining excellent network performance. We propose three transformation formulas to depict this process and support them with theoretical proofs and extensive experiments. MathNAS achieves SOTA performance on various large-scale CV and NLP benchmark datasets. When deployed on mobile devices, MathNAS enables real-time search and dynamic networks, surpassing baseline dynamic networks in on-device performance. Capable of conducting rapid searches on any architecture, MathNAS offers an appealing strategy for expediting the design process of large models, providing a clearer and more effective solution.

# 5 Acknowledgment

We thank the anonymous reviewers for their constructive comments. This work was partially supported by National Key R&D Program of China (2022YFB2902302). Sihai Zhang gratefully acknowledges the support of Innovative Research (CXCCTEC2200001).

# References

[1] Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for visual recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12270–12280, 2021.   
[2] Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Bichen Wu, Zijian He, Zhen Wei, Kan Chen, Yuandong Tian, Matthew Yu, Peter Vajda, et al. Fbnetv3: Joint architecture-recipe search using predictor pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16276–16285, 2021.   
[3] Bichen Wu, Chaojian Li, Hang Zhang, Xiaoliang Dai, Peizhao Zhang, Matthew Yu, Jialiang Wang, Yingyan Lin, and Peter Vajda. Fbnetv5: Neural architecture search for multiple tasks in one run. arXiv preprint arXiv:2111.10007, 2021.   
[4] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.   
[5] Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, and Song Han. Hat: Hardware-aware transformers for efficient natural language processing. arXiv preprint arXiv:2005.14187, 2020.   
[6] Nikita Klyuchnikov, Ilya Trofimov, Ekaterina Artemova, Mikhail Salnikov, Maxim Fedorov, Alexander Filippov, and Evgeny Burnaev. Nas-bench-nlp: neural architecture search benchmark for natural language processing. IEEE Access, 10:45736–45747, 2022.   
[7] Hung-yi Lee, Shang-Wen Li, and Ngoc Thang Vu. Meta learning for natural language processing: A survey. arXiv preprint arXiv:2205.01500, 2022.   
[8] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[9] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM computing surveys (CSUR), 54(10s):1–41, 2022.   
[10] Mojan Javaheripi, Gustavo de Rosa, Subhabrata Mukherjee, Shital Shah, Tomasz Religa, Caio Cesar Teodoro Mendes, Sebastien Bubeck, Farinaz Koushanfar, and Debadeepta Dey. Litetransformersearch: Training-free neural architecture search for efficient language models. Advances in Neural Information Processing Systems, 35:24254–24267, 2022.   
[11] Gresa Shala, Thomas Elsken, Frank Hutter, and Josif Grabocka. Transfer nas with meta-learned bayesian surrogates. In Sixth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems.

[12] Zihao Sun, Yu Hu, Shun Lu, Longxing Yang, Jilin Mei, Yinhe Han, and Xiaowei Li. Agnas: Attention-guided micro and macro-architecture search. In International Conference on Machine Learning, pages 20777–20789. PMLR, 2022.   
[13] Chengyue Gong, Dilin Wang, Meng Li, Xinlei Chen, Zhicheng Yan, Yuandong Tian, Vikas Chandra, et al. Nasvit: Neural architecture search for efficient vision transformers with gradient conflict aware supernet training. In International Conference on Learning Representations, 2021.   
[14] Xuan Shen, Yaohua Wang, Ming Lin, Yilun Huang, Hao Tang, Xiuyu Sun, and Yanzhi Wang. Deepmad: Mathematical architecture design for deep convolutional neural network. arXiv preprint arXiv:2303.02165, 2023.   
[15] Yicheng Fan, Dana Alon, Jingyue Shen, Daiyi Peng, Keshav Kumar, Yun Long, Xin Wang, Fotis Iliopoulos, Da-Cheng Juan, and Erik Vee. Layernas: Neural architecture search in polynomial complexity. arXiv preprint arXiv:2304.11517, 2023.   
[16] Hieu Pham, Melody Guan, Barret Zoph, Quoc Le, and Jeff Dean. Efficient neural architecture search via parameters sharing. In International conference on machine learning, pages 4095-4104. PMLR, 2018.   
[17] Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Smash: one-shot model architecture search through hypernetworks. arXiv preprint arXiv:1708.05344, 2017.   
[18] Bowen Baker, Otkrist Gupta, Ramesh Raskar, and Nikhil Naik. Accelerating neural architecture search using performance prediction. arXiv preprint arXiv:1705.10823, 2017.   
[19] Boyang Deng, Junjie Yan, and Dahua Lin. Peephole: Predicting network performance before training. arXiv preprint arXiv:1712.03351, 2017.   
[20] Xuanyi Dong and Yi Yang. Nas-bench-201: Extending the scope of reproducible neural architecture search. arXiv preprint arXiv:2001.00326, 2020.   
[21] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1314–1324, 2019.   
[22] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. arXiv preprint arXiv:1908.09791, 2019.   
[23] Sasha Targ, Diogo Almeida, and Kevin Lyman. Resnet in resnet: Generalizing residual architectures. arXiv preprint arXiv:1603.08029, 2016.   
[24] Dongyoon Han, Sangdoo Yun, Byeongho Heo, and YoungJoon Yoo. Rethinking channel dimensions for efficient model design. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pages 732–741, 2021.   
[25] Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, and Matthijs Douze. Levit: a vision transformer in convnet's clothing for faster inference. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12259–12269, 2021.   
[26] Mingxing Tan and QuocV. Le. Efficientnet: Rethinking model scaling for convolutional neural networks, May 2019.   
[27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, Dec 2015.   
[28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Mar 2022.   
[29] Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for visual recognition, Jul 2021.   
[30] Lukasz Dudziak, Thomas Chau, Mohamed Abdelfattah, Royson Lee, Hyeji Kim, and Nicholas Lane. Brp-nas: Prediction-based nas using gcns. Advances in Neural Information Processing Systems, 33:10480–10490, 2020.

[31] Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer. Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10734–10742, 2019.   
[32] Jieru Mei, Yingwei Li, Xiaochen Lian, Xiaojie Jin, Linjie Yang, Alan Yuille, and Jianchao Yang. Atomnas: Fine-grained end-to-end neural architecture search. arXiv preprint arXiv:1912.09640, 2019.   
[33] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2820–2828, 2019.   
[34] Han Cai, Ligeng Zhu, and Song Han. Proxylessnas: Direct neural architecture search on target task and hardware. arXiv preprint arXiv:1812.00332, 2018.   
[35] Xiangxiang Chu, Bo Zhang, and Ruijun Xu. Fairnas: Rethinking evaluation fairness of weight sharing neural architecture search. In Proceedings of the IEEE/CVF International Conference on computer vision, pages 12239–12248, 2021.   
[36] Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVI 16, pages 544–560. Springer, 2020.   
[37] Jiahui Yu, Pengchong Jin, Hanxiao Liu, Gabriel Bender, Pieter-Jan Kindermans, Mingxing Tan, Thomas Huang, Xiaodan Song, Ruoming Pang, and Quoc Le. Bignas: Scaling up neural architecture search with big single-stage models. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16, pages 702–717. Springer, 2020.   
[38] David So, Quoc Le, and Chen Liang. The evolved transformer. In International conference on machine learning, pages 5877–5886. PMLR, 2019.   
[39] Changlin Li, Guangrun Wang, Bing Wang, Xiaodan Liang, Zhihui Li, and Xiaojun Chang. Dynamic slimmable network. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 8607–8617, 2021.   
[40] Jiahui Yu and Thomas S Huang. Universally slimmable networks and improved training techniques. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1803–1811, 2019.   
[41] Wei Lou, Lei Xun, Amin Sabet, Jia Bi, Jonathon Hare, and Geoff V Merrett. Dynamic-ofa: Runtime dnn architecture switching for performance scaling on heterogeneous embedded platforms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3110–3118, 2021.   
[42] Jiahui Yu and Thomas Huang. Autoslim: Towards one-shot architecture search for channel numbers. arXiv preprint arXiv:1903.11728, 2019.   
[43] Syed Asif Raza Shah, Wenji Wu, Qiming Lu, Liang Zhang, Sajith Sasidharan, Phil DeMar, Chin Guok, John Macauley, Eric Pouyoul, Jin Kim, et al. Amoebanet: An sdn-enabled network service for big data science. Journal of Network and Computer Applications, 119:70–82, 2018.   
[44] Zhaohui Yang, Yunhe Wang, Xinghao Chen, Boxin Shi, Chao Xu, Chunjing Xu, Qi Tian, and Chang Xu. Cars: Continuous evolution for efficient neural architecture search. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1829–1838, 2020.   
[45] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2820–2828, 2019.   
[46] Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. In Proceedings of the European conference on computer vision (ECCV), pages 19–34, 2018.   
[47] Shoukang Hu, Sirui Xie, Hehui Zheng, Chunxiao Liu, Jianping Shi, Xunying Liu, and Dahua Lin. Dsnas: Direct neural architecture search without parameter retraining. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12084–12092, 2020.   
[48] Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. Snas: stochastic neural architecture search. arXiv preprint arXiv:1812.09926, 2018.   
[49] Han Cai, Ligeng Zhu, and Song Han. Proxylessnas: Direct neural architecture search on target task and hardware. arXiv preprint arXiv:1812.00332, 2018.   
[50] J Pablo Muñoz, Nikolay Lyalyushkin, Yash Akhauri, Anastasia Senina, Alexander Kozlov, and Nilesh Jain. Enabling nas with automated super-network generation. arXiv preprint arXiv:2112.10878, 2021.   
[51] Changlin Li, Jiefeng Peng, Liuchun Yuan, Guangrun Wang, Xiaodan Liang, Liang Lin, and Xiaojun Chang. Block-wisely supervised neural architecture search with knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1989–1998, 2020.   
[52] Bert Moons, Parham Noorzad, Andrii Skliar, Giovanni Mariani, Dushyant Mehta, Chris Lott, and Tijmen Blankevoort. Distilling optimal neural networks: Rapid search in diverse spaces. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12229-12238, 2021.   
[53] Pavlo Molchanov, Jimmy Hall, Hongxu Yin, Jan Kautz, Nicolo Fusi, and Arash Vahdat. Lana: latency aware network acceleration. In European Conference on Computer Vision, pages 137–156. Springer, 2022.   
[54] Thomas Chau, Łukasz Dudziak, Hongkai Wen, Nicholas Lane, and Mohamed Abdelfattah. Blox: Macro neural architecture search benchmark and algorithms. Advances in Neural Information Processing Systems, 35:30851–30864, 2022.

Organization In this supplementary file, we provide in-depth descriptions of the materials that are not covered in the main paper, and report additional experimental results. The document is organized as follows:

• Section A- Related work.   
• Section B- Theoretical Analysis.

- B.1 Unbiased proof.   
- B.2 Fit validation analysis.   
- B.3 Ablation analysis.   
- B.4 Effectiveness analysis.   
- B.5 Space generality analysis.   
- B.6 Searching equation solving details.

• Section C- Experimental setups.

• Section D- Additional experiments.

- D.1 performance comparison on NAS-Bench-201.   
- D.2 performance comparison with block-wise methods.   
- D.3 Additional experiments on dynamic networks.

• Section E- Searched Architecture Visualization.

- E.1 Visualization Architecture on MobileNet-V3.   
- E.2 Visualization Architecture on SuperViT Space.   
- E.3 Visualization Architecture on Dynamic Network.

• Section F- Discussion.

- F.1 Limitation and Future Work.   
- F.2 Potential negative social impact.

# A Related Work

Neural Architecture Search (NAS) was introduced to ease the process of manually designing complex neural networks. Early NAS $[43]$ efforts employed a brute force approach by training candidate architectures and using their accuracy as a proxy for discovering superior designs. Subsequent EA and RL-driven methods significantly enhanced search efficiency by sampling and training multiple candidate architectures $[44, 45, 46]$ . One-shot NAS methods $[47, 48, 49]$ further reduced the cost by training large supernetworks and identifying high-accuracy subnetworks, often generated from pre-trained models. Nevertheless, as search spaces expand with architectural innovations $[50, 10]$ , more efficient methods are necessary to predict neural network accuracy in vast design spaces.

Recent mathematical programming (MP) based NAS methods $[14, 15]$ are noteworthy, as they transform multi-objective NAS problems into mathematical programming solutions. MP-NAS reduces search complexity from exponential to polynomial, presenting a promising avenue for large model architecture search. However, existing MP-NAS methods face architectural limitations. For instance, DeepMAD $[14]$ is designed for convolutional neural networks, while LayerNAS $[15]$ is suited for hierarchically connected networks. These limitations hinder MP-NAS usage in SOTA search spaces, leaving the challenge of swiftly designing effective large models unresolved. To address this, we propose an architecturally generalized MP-NAS, MathNAS. With the capability to perform rapid searches on any architecture, MathNAS presents an enticing approach to accelerate the design process for large models, providing a clearer and more effective solution.

# B Theoretical Analysis

# B.1 $b_{i,j}^{A}$ Unbiased Proof

The main text introduces the definition of $b_{i,j}^{A}$ as follows:

$$
b _ {i, j} ^ {A} = \overline {{\Delta A c c \left(\mathcal {N} _ {(i , 1) \rightarrow (i , j)} ^ {\Omega}\right)}} = \frac {1}{n ^ {m - 1}} \sum_ {k = 1} ^ {n ^ {m - 1}} \left(A c c \left(\mathcal {N} _ {(i, 1)} ^ {k}\right) - A c c \left(\mathcal {N} _ {(i, j)} ^ {k}\right)\right) \tag {7}
$$

In investigating the unbiasedness of $b_{i,j}^{A}$ , it is essential to first examine the distributions of $Acc(\mathcal{N}(i,1))$ and $Acc(\mathcal{N}(i,j))$ . We conducted an experiment on NAS-Bench-201 to discern the distribution of accuracies for all networks that include a specific block. Figure 1 illustrates the experimental results, which suggest that, when the Kolmogorov-Smirnov (KS) test is applied to a beta distribution, all p-values exceed 0.05. This allows us to infer that the network accuracy related to a specific block adheres to a beta distribution, with the parameters $\alpha$ and $\beta$ varying among different blocks.

We can thus hypothesize that $Acc(\mathcal{N}(i,1))$ and $Acc(\mathcal{N}(i,j))$ follow the beta distributions with parameters $\alpha_{1},\beta_{1}$ and $\alpha_{2},\beta_{2}$ , respectively. The expectation of $b_{i,j}^{A}$ is computed as follows:

![](images/55d013d7e703c4794ecd515fdc0104295663db6f62a8c4eb16592c3f5385c4a6.jpg)  
Figure 7: Distribution of accuracy for networks containing specific blocks on NAS-Bench-201. Accuracy is the result of the network training on ImageNet for 200 epochs. It can be seen that the ks verification p value of the beta distribution on all blocks exceeds 0.05, so it can be considered that the accuracy of all networks containing a specific block is in line with the beta distribution.

$$
\begin{array}{l} E \left[ b _ {i, j} ^ {A} \right] = E \left[ \frac {1}{n ^ {m - 1}} \sum_ {k = 1} ^ {n ^ {m - 1}} \left(A c c \left(\mathcal {N} _ {(i, 1)} ^ {k}\right) - A c c \left(\mathcal {N} _ {(i, j)} ^ {k}\right)\right) \right] \\ = \frac {1}{n ^ {m - 1}} \sum_ {k = 1} ^ {n ^ {m - 1}} \left(E [ A c c (\mathcal {N} _ {(i, 1)} ^ {k}) ] - E [ A c c (\mathcal {N} _ {(i, j)} ^ {k}) ]\right) \tag {8} \\ = E \left[ A c c \left(\mathcal {N} _ {(i, 1)}\right) \right] - E \left[ A c c \left(\mathcal {N} _ {(i, j)}\right) \right] \\ = \frac {\alpha_ {1}}{\alpha_ {1} + \beta_ {1}} - \frac {\alpha_ {2}}{\alpha_ {2} + \beta_ {2}} \\ \end{array}
$$

It is evident that the expected value of statistic $b_{i,j}^{A}$ represents the discrepancy between the expected values of $Acc(\mathcal{N}(i,1))$ and $Acc(\mathcal{N}(i,j))$ , and is independent of any specific sample value. In other words, regardless of the sample values observed, the expected value of statistic $b_{i,j}^{A}$ always reflects the difference between the expected values of $Acc(\mathcal{N}(i,1))$ and $Acc(\mathcal{N}(i,j))$ . Thus, we can assert that the statistic $b_{i,j}^{A}$ is unbiased, meaning the estimate of the difference between the expected values of $Acc(\mathcal{N}(i,1))$ and $Acc(\mathcal{N}(i,j))$ remains unskewed.

Similarly, it can be argued that $b_{i,j}^{L}$ and $b_{i,j}^{E}$ are also unbiased. This conclusion adds to our understanding of the general applicability of these statistics in analyzing network accuracy associated with specific blocks.

# B.2 Fit Validation

![](images/be1506265ac65a4944e23aea197797a4e8a0b6e542aa0965dcf559f475f3e8b1.jpg)  
Figure 8: More result of the relationship between $\Delta Acc(\mathcal{N}_{(i,1)\to (i,j)})$ , $\Delta Lat(\mathcal{N}_{(i,1)\to (i,j)})$ , $\Delta Eng(\mathcal{N}_{(i,1)\to (i,j)})$ and $FLOPs(\mathcal{N})$ .

Within the main body of the text, we put forth several rules derived from our empirical observations:

1. For accuracy, $\Delta Acc(\mathcal{N}_{(i,1)\to (i,j_i)})\approx \alpha *1 / \mathcal{F}(\mathcal{N})$   
2. For latency and energy consumption, $\Delta Lat(\mathcal{N}(i,1)\to (i,j_i))$ and $\Delta Eng(\mathcal{N}(i,1)\to (i,j_i))$ remain relatively constant across different networks when referring to a specific block.

We offer additional $\Delta Acc/Lat/Eng(\mathcal{N}(i,1)\to(i,j))-FLOPs(\mathcal{N})$ pairs from NAS-Bench-201 and MobileNetV3 as depicted in Figure 8, which serve to substantiate the universality of these rules. Furthermore, we fit $\Delta Acc(\mathcal{N}(i,1)\to(i,j))-FLOPs(\mathcal{N})$ using different inverse functions. As evidenced in Table 4, a reciprocal function provides the best fit for this relationship.

Table 4: Comparison of the fitting effect of different functions on the relationship between $\mathcal{N}_{(i,1)\to (i,j)}$ and $FLOPs(\mathcal{N})$ . 

<table><tr><td>Net</td><td colspan="3">Linear Function</td><td colspan="3">Quadratic Function</td><td colspan="3">Reciprocal Function</td><td colspan="3">Log Function</td><td colspan="3">Exp Function</td></tr><tr><td></td><td> $R^{2}$ </td><td>MSE</td><td>DC</td><td> $R^{2}$ </td><td>MSE</td><td>DC</td><td> $R^{2}$ </td><td>MSE</td><td>DC</td><td> $R^{2}$ </td><td>MSE</td><td>DC</td><td> $R^{2}$ </td><td>MSE</td><td>DC</td></tr><tr><td>MobileNetV3</td><td>0.47</td><td>0.0061</td><td>0.47</td><td>0.45</td><td>0.0062</td><td>0.46</td><td>0.51</td><td>0.0059</td><td>0.48</td><td>0.48</td><td>0.0061</td><td>0.48</td><td>0.001</td><td>0.011</td><td>0.0011</td></tr><tr><td>NAS-Bench-201</td><td>0.57</td><td>0.018</td><td>0.57</td><td>0.46</td><td>0.022</td><td>0.46</td><td>0.68</td><td>0.013</td><td>0.68</td><td>0.61</td><td>0.015</td><td>0.61</td><td>0.004</td><td>0.04</td><td>0.004</td></tr></table>

# B.3 Ablation Analysis

To elucidate the significance of each element within our predictive methodology, we instituted three sets of control models:

1. An accuracy prediction model that operates without FLOPs information.   
2. A model predicting accuracy/latency where the Baseline network is arbitrarily chosen, instead of selecting a network with average FLOPs.   
3. A model predicting accuracy/latency that uses any arbitrary block as the baseline block as opposed to the first block.

Table 5 illustrates the outcomes of these models. It becomes evident that the omission of any rule adversely affects the model's performance. The second model, in particular, proves to be the most vital. These findings corroborate the necessity for the basenet to be the network of average FLOPs, aligning with the rule between $\Delta Acc(\mathcal{N}_{(i,1)\to (i,j)}) - FLOPs(\mathcal{N})$ we derived earlier.

Table 5: Ablation Study of the proposed prediction method on MobileNetV3 

<table><tr><td>Predict Properties</td><td>Network</td><td>RMSE</td><td>MAE</td><td>Error</td></tr><tr><td rowspan="4">Accuracy</td><td>Complete Model</td><td>0.117</td><td>0.0916</td><td>-</td></tr><tr><td>Without FLOPs info</td><td>0.212</td><td>0.184</td><td>0.0924</td></tr><tr><td>Without Ave-FLOPs BaseNet</td><td>0.513</td><td>0.416</td><td>0.324</td></tr><tr><td>Without Super Baseblock</td><td>0.123</td><td>0.0992</td><td>0.0076</td></tr><tr><td rowspan="3">Latency</td><td>Complete Model</td><td>0.268</td><td>0.189</td><td>-</td></tr><tr><td>Without Ave-FLOPs BaseNet</td><td>0.408</td><td>0.317</td><td>0.128</td></tr><tr><td>Without Super Baseblock</td><td>0.276</td><td>0.194</td><td>0.005</td></tr></table>

# B.4 Effectiveness Analysis

This section provides an exploration into the efficacy of our proposed predictive model, substantiated by experimental validations. We draw comparisons with the classic GNN predictor on NAS-Bench-201 and MobileNetV3 search spaces. The GNN predictor's training process adheres to the Brp-NAS method. Figure 9 demonstrates that our model's MAE and MSE converge as the sample size of the network increases. This convergence is apparent in both weight-shared and weight-independent networks.

# B.5 Space Generality Analysis

In our primary discussion, we reformulated the NAS problem as an Integer Linear Programming (ILP) problem. In this section, we delve further into the application of this formulation within the actual search space.

Flexibility in Number of Block Nodes. Some search spaces provide the flexibility to choose the number of block nodes. For instance, the SuperTransformer allows for the selection of anywhere between 1 and 6 decoders. For this kind of search space, MathNAS can also be applied. Specifically, for block node i that has no block operation, we regard it as a special block of this node and represent

![](images/7bf673d694ed6ce12d0d13d33e1e79c71b0c18aa752946ea0cca3df23df79eed.jpg)

<details>
<summary>line</summary>

| Sample Ratio(%) | GNN Predictor MAE | BAAL Predictor MAE |
| --------------- | ----------------- | ------------------ |
| 0               | 0.55              | 0.10               |
| 10              | 0.45              | 0.10               |
| 20              | 0.30              | 0.10               |
| 30              | 0.25              | 0.10               |
| 40              | 0.23              | 0.10               |
| 50              | 0.22              | 0.10               |
| 60              | 0.21              | 0.10               |
| 70              | 0.22              | 0.10               |
| 80              | 0.23              | 0.10               |
| 90              | 0.22              | 0.10               |
| 100             | 0.21              | 0.10               |
</details>

(a) MobileNetV3

![](images/72084e31649cc7105d583d24a954888e3cc7da5039301d2a28ac61a35a884947.jpg)

<details>
<summary>line</summary>

| Sample Ratio(%) | GNN Predictor MAE | BAAL Predictor MAE |
| --------------- | ----------------- | ------------------ |
| 0               | 8.5               | 2.8                |
| 10              | 7.0               | 2.7                |
| 20              | 5.5               | 2.6                |
| 30              | 4.8               | 2.5                |
| 40              | 4.2               | 2.5                |
| 50              | 3.8               | 2.5                |
| 60              | 3.5               | 2.5                |
| 70              | 3.3               | 2.5                |
| 80              | 3.2               | 2.5                |
| 90              | 3.1               | 2.5                |
| 100             | 3.0               | 2.5                |
</details>

(b) NAS-Bench-201   
Figure 9: Sample effect of our accuracy predictor compared with GNN predictor.

it with $b_{i,0}$ . Under this condition, the ILP formula can be modified accordingly as follows:

$$
\max _ {b ^ {B}} A c c (\widetilde {\mathcal {N}}) - \frac {\sum_ {i = 1} ^ {m} \sum_ {j = 0} ^ {n} b _ {i , j} ^ {A} * b _ {i , j} ^ {B}}{\sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i , j} ^ {F} * b _ {i , j} ^ {B}} * F L O P s (\overline {{\mathcal {N}}})
$$

s.t.

$$
\operatorname{Lat} (\widetilde {\mathcal {N}}) - \sum_ {i = 1} ^ {m} \sum_ {j = 0} ^ {n} b _ {i, j} ^ {L} * b _ {i, j} ^ {B} \leq \hat {L}, \tag {9}
$$

$$
E n g (\widetilde {\mathcal {N}}) - \sum_ {i = 1} ^ {m} \sum_ {j = 0} ^ {n} b _ {i, j} ^ {E} * b _ {i, j} ^ {B} \leq \hat {E},
$$

$$
\sum_ {k = 1} ^ {i} b _ {k, 0} ^ {B} + \sum_ {j = 1} ^ {n} b _ {i, j} ^ {B} = 1, b _ {i, j} ^ {B} \in \{0, 1 \}, \forall 1 \leq i \leq m.
$$

Applicability in Micro Search Space. Unlike other search spaces we use, NAS-Bench-201 is a micro search space. In this paragraph, we explain how we performed our experiments in this search space and why MathNAS is applicable.

Practical Implementation (How): During experiments within the NAS-Bench-201 space, we identify a set of edges in the same position across multiple GNN cells as a single "block". Given that the structure of GNN cells in the network remains consistent, our focus is on a single cell. Hence, any alteration in an edge operation essentially translates to a corresponding change in all cells, while other edge operations remain static.

Theoretical Framework (Why): Regardless of the distinction between macro and micro search spaces, the networks in both are assembled from multiple mutable modules, be it blocks or edges. The capability of the entire network can be represented by the capabilities of these individual modules. In MathNAS, to explore the contribution of module capabilities to the network's performance, we evaluated changes in inherent module capabilities and their interactive capacities during module switches. This module evaluation methodology is applicable to the definition of blocks in NAS-Bench-201 that we proposed above: alterations in edge operations impact not only the specific edges' output data (inherent capability) but also influence the input and output data of other edges within the network (interactive capability). Therefore, MathNAS is theoretically suitable for micro search spaces represented by NAS-Bench-201.

# B.6 Searching Equation Solving Details

In this section, we describe in detail the solution of the fractional objective function programming equation proposed in the paper. The solution is divided into two steps.

1. Convert the original equation into an integer linear programming equation.   
2. Solve the ILP equation.

Equation Transformation. In order to transform the equation into an ILP problem, we first perform variable substitution on the original equation.

$$
l e t \quad b _ {i, j} ^ {\widetilde {B}} = \frac {b _ {i , j} ^ {B}}{\sum_ {i = 1} ^ {m} \sum_ {i = 1} ^ {n} b _ {i , j} ^ {F} * b _ {i , j} ^ {B}}, \quad z = \frac {1}{\sum_ {i = 1} ^ {m} \sum_ {i = 1} ^ {n} b _ {i , j} ^ {F} * b _ {i , j} ^ {B}} \tag {10}
$$

Then the original equation can be transformed into the following integer linear programming problem:

$$
O = \min _ {b ^ {\widetilde {B}}, z} (\sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i, j} ^ {A} * b _ {i, j} ^ {\widetilde {B}} * \overline {{\mathcal {F} (\mathcal {N})}})
$$

s.t.

$$
(L a t (\widetilde {\mathcal {N}}) - \hat {L}) * z \leq \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i, j} ^ {L} * b _ {i, j} ^ {\widetilde {B}}, (E n g (\widetilde {\mathcal {N}}) - \hat {E}) * z \leq \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} b _ {i, j} ^ {E} * b _ {i, j} ^ {\widetilde {B}} \tag {11}
$$

$$
\forall 1 \leq i \leq m, \sum_ {j = 1} ^ {n} b _ {i, j} ^ {\widetilde {B}} = z, b _ {i, j} ^ {\widetilde {B}} \in \{0, z \}.
$$

ILP Solving. To solve the ILP equations, we use the off-the-shelf Linprog Python package and the Gurobipy Python package to find feasible candidate solutions.

- Linprog is a basic integer programming solver that can be used on almost all edge devices, even on the resource-constrained Raspberry Pi. We use it to implement the branch and bound method and solve the ILP problem.   
- Gurobipy is a more powerful solver, which has built-in a variety of advanced solving algorithms such as heuristic algorithms, and can flexibly utilize all available hardware resources on the device. Although Gurobipy is powerful, it requires more hardware resources than Linprog.

Therefore, for devices with limited hardware resources, we use Linprog for searching. For well-resourced devices, we use Gurobipy.

# C Experimental Setups

# C.1 Description of Search Space

This section provides a detailed overview of the search spaces that we employ in our experimental setup. Furthermore, in our experiments, each resolution corresponds to a separate search space.

NAS-Bench-201 search space encompasses cell-based neural architectures, where an architecture is represented as a graph. Each cell in this graph comprises four nodes and six edges, with each edge offering a choice among five operational candidates - zerorize, skip connection, 1-by-1 convolution, 3-by-3 convolution, and 3-by-3 average pooling. This design leads to a total of 15,626 unique architectures. The macro skeleton is constructed with one stem cell, three stages each composed of five repeated cells, residual blocks between the stages, and a final classification layer which incorporates an average pooling layer and a fully connected layer with a softmax function. The stem cell is formed by a 3-by-3 convolution with 16 output channels followed by a batch normalization layer. Each cell within the three stages has 16, 32 and 64 output channels respectively. The intermediate residual blocks contain convolution layers with stride 2 for down-sampling.

MobileNetV3 search space follows a layer-by-layer paradigm, where the building blocks use MB-Convs, squeeze and excite mechanisms, and modified swish nonlinearity to construct efficient neural networks. This space is organized into five stages, each containing a number of building blocks varying from 2 to 4. The kernel size for each block can be chosen from $\{3, 5, 7\}$ and the expansion ratio from $\{1, 4, 6\}$ . The search space encapsulates approximately $10^{19}$ sub-nets, with each block offering 7,371 choices.

SuperViT search space is a tiled space featuring a fixed macroarchitecture. It is composed of three dynamic CNN blocks and four dynamic Transformer blocks connected in sequence. Each block allows for variations in width, depth, kernel size and expansion ratio. The SuperViT search space is displayed in Table 6.

Table 6: An illustration of SuperViT search space. Bold black represents blocks selected for the baseline net. The block in basenet corresponds to the largest of all available options. 

<table><tr><td>Block</td><td>Width</td><td>Depth</td><td>Kernel size</td><td>Expansion ratio</td><td>SE</td><td>Stride</td><td>Number of Windows</td></tr><tr><td>Conv</td><td>{16,24}</td><td>-</td><td>3</td><td>-</td><td>-</td><td>2</td><td>-</td></tr><tr><td>MBConv-1</td><td>{16,24}</td><td>{1,2}</td><td>{3,5}</td><td>1</td><td>N</td><td>1</td><td>-</td></tr><tr><td>MBConv-2</td><td>{24,32}</td><td>{3,4,5}</td><td>{3,5}</td><td>{4,5,6}</td><td>N</td><td>2</td><td>-</td></tr><tr><td>MBConv-3</td><td>{32,40}</td><td>{3,4,5,6}</td><td>{3,5}</td><td>{4,5,6}</td><td>Y</td><td>2</td><td>-</td></tr><tr><td>Transformer-4</td><td>{64,72}</td><td>{3,4,5,6}</td><td>-</td><td>{1,2}</td><td>-</td><td>2</td><td>1</td></tr><tr><td>Transformer-5</td><td>{112,120,128}</td><td>{3,4,5,6,7,8}</td><td>-</td><td>{1,2}</td><td>-</td><td>2</td><td>1</td></tr><tr><td>Transformer-6</td><td>{160,168,176,184}</td><td>{3,4,5,6,7,8}</td><td>-</td><td>{1,2}</td><td>-</td><td>1</td><td>1</td></tr><tr><td>Transformer-7</td><td>{208,216,224}</td><td>{3,4,5,6}</td><td>-</td><td>{1,2}</td><td>-</td><td>2</td><td>1</td></tr><tr><td>MBPool</td><td>{1792,1984}</td><td>-</td><td>1</td><td>6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Input Resolution</td><td colspan="2"></td><td colspan="5">{192,224,256,288}</td></tr></table>

SuperTransformer search space we employ is a large design space, constructed with Arbitrary Encoder-Decoder Attention and Heterogeneous Layers. This space contains an encoder layer and a decoder layer, the embedding dimensions of which can be selected from 512, 640. The encoder comprises six layers, each with an attention module and an FFN layer in series. The number of decoder layers can range from 1 to 6, and each decoder layer consists of two attention modules and one FFN layer in series. Each decoder has the option to focus on the last, second last or third last encoder layer. The hidden dimension of the FFN can be chosen from 1024, 2048, 3072, and the number of heads in attention modules from 4, 8. Every encoder layer has six choices, while each decoder layer provides 36 options. The search space and its corresponding baseline network are illustrated in Table 7.

Table 7: An illustration of SuperTransformer search space and the baseline net. 

<table><tr><td>Module</td><td>Choice</td><td>Baseline Net</td></tr><tr><td>encoder-embed</td><td>[640, 512]</td><td>512</td></tr><tr><td>decoder-embed</td><td>[640, 512]</td><td>512</td></tr><tr><td>encoder-ffn-embed-dim</td><td>[3072, 2048, 1024]</td><td>2048</td></tr><tr><td>decoder-ffn-embed-dim</td><td>[3072, 2048, 1024]</td><td>2048</td></tr><tr><td>encoder-layer-num</td><td>[6]</td><td>6</td></tr><tr><td>decoder-layer-num</td><td>[6, 5, 4, 3, 2, 1]</td><td>6</td></tr><tr><td>encoder-self-attention-heads</td><td>[8, 4]</td><td>4</td></tr><tr><td>decoder-self-attention-heads</td><td>[8, 4]</td><td>4</td></tr><tr><td>decoder-ende-attention-heads</td><td>[8, 4]</td><td>4</td></tr><tr><td>decoder-arbitrary-ende-attn</td><td>[-1, 1, 2]</td><td>1</td></tr></table>

# D Additional Experiments

# D.1 Experiment on NAS-Bench-201 Search Space

In the main body of this paper, we have showcased the competency of MathNAS in attaining accuracy-latency trade-offs within the NAS-Bench-201 search space. In this section, we extend our discussion to compare MathNAS with the current state-of-the-art models.

A comparative analysis with the leading-edge algorithms on NAS-Bench-201 is presented in Table 8. All the algorithms under consideration utilize the CIFAR-10 training and validation sets for architectural search and employ the NAS-bench-201 API to ascertain the ground-truth performance of the searched architecture across the three datasets. The reported experimental results are averaged over four separate searches.

From the results, it is evident that our method outperforms the state-of-the-art methods, with our best results approaching the peak performance. This robust performance serves as a testament to the efficacy of our proposed approach.

Table 8: Comparison results of MathNAS with state-of-the-art NAS methods on NAS-Bench-201. 

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td><td colspan="2">ImageNet-16-120</td></tr><tr><td>validation</td><td>test</td><td>validation</td><td>test</td><td>validation</td><td>test</td></tr><tr><td> $DARTS^{1st}$ </td><td>39.77 ± 0.00</td><td>54.30 ± 0.00</td><td>15.03 ± 0.00</td><td>15.61 ± 0.00</td><td>16.43 ± 0.00</td><td>16.32 ± 0.00</td></tr><tr><td> $DARTS^{2nd}$ </td><td>39.77 ± 0.00</td><td>54.30 ± 0.00</td><td>38.57 ± 0.00</td><td>38.97 ± 0.00</td><td>18.87 ± 0.00</td><td>18.41 ± 0.00</td></tr><tr><td>SETN</td><td>84.04 ± 0.28</td><td>87.64 ± 0.00</td><td>58.86 ± 0.06</td><td>59.05 ± 0.24</td><td>33.06 ± 0.02</td><td>32.52 ± 0.21</td></tr><tr><td>FairNAS</td><td>90.07 ± 0.57</td><td>93.23 ± 0.18</td><td>70.94 ± 0.84</td><td>71.00 ± 1.46</td><td>41.9 ± 1.00</td><td>42.19 ± 0.31</td></tr><tr><td>SGNAS</td><td>90.18 ± 0.31</td><td>93.53 ± 0.12</td><td>70.28 ± 1.20</td><td>70.31 ± 1.09</td><td>44.65 ± 2.32</td><td>44.98 ± 2.10</td></tr><tr><td>DARTS-</td><td>91.03 ± 0.44</td><td>93.80 ± 0.40</td><td>71.36 ± 1.51</td><td>71.53 ± 1.51</td><td>44.87 ± 1.46</td><td>45.12 ± 0.82</td></tr><tr><td>Ours</td><td>90.18±0.00</td><td>93.31±0.00</td><td>71.74±0.00</td><td>70.82±0.00</td><td>46.00±0.00</td><td>46.53±0.00</td></tr><tr><td>Optimal</td><td>91.61</td><td>94.37</td><td>73.49</td><td>73.51</td><td>46.77</td><td>47.31</td></tr></table>

# D.2 Performance Comparison with Block-wise Methods

Existing blockwise methods such as DNA $[51]$ , DONNA $[52]$ , and LANA $[53]$ use block distillation techniques to block the teacher model, obtaining the architecture of blocks to be replaced and their performance evaluation. Following this, they use the performance of each block to guide the algorithm in finding models with superior performance. Recent work $[54]$ has pointed out the limitations of such methods. They depend on an excellent teacher network due to their use of distillation techniques. Furthermore, previous block performance estimation methods are unable to effectively predict actual block performance. Additionally, these methods are only suitable for the macro search space. In this section, we compare MathNAS with these methods and demonstrate that MathNAS overcomes these limitations.

We compare the performance of our method and previous blocking methods on the MobileNetV3 search space and the ImageNet dataset. For DNA and DONNA, we use the data reported in the original DONNA paper, where DONNA selects the data of the predictor after training on 40 samples. In the case of LANA, due to the absence of the official code, we computed the metrics using the

supernet in the search space as the teacher net, determining block evaluation scores from the variation in verification accuracy. The final network performance scores result from a linear addition, in line with LANA's methodology. The results are shown in Table 9. Evidently, MathNAS outperforms prior block-wise methods in both network performance evaluation and accuracy prediction.

Table 9: Comparison of prediction accuracy between our proposed method and block-wise methods on the MobileNetV3 (x 1.2) search space with the ImageNet dataset. 

<table><tr><td>Methods</td><td>Kendall-Tau</td><td>MSE</td><td>Block Evaluation</td><td>Net Evaluation</td></tr><tr><td>DNA</td><td>0.74</td><td>NA</td><td>Block Knowledge Distillation</td><td>Sorting Model</td></tr><tr><td>DONNA</td><td>0.82</td><td>0.08</td><td>Distillation Loss</td><td>Linear Regression</td></tr><tr><td>LANA</td><td>0.77</td><td>0.04</td><td>Change of Validation Accuracy</td><td>Simple Summation</td></tr><tr><td>MathNAS</td><td>0.86</td><td>0.01</td><td>Average of Accuracy Variations</td><td>MathNAS Formula</td></tr></table>

Previous blocking methods were targeted at macrospaces, while MathNAS has excellent performance in both macrospaces and microspaces. Figures 10 and 11 show the performance comparison between MathNAS and LANA on macrospace MobileNetV3 and microspace NAS-Bench-201 respectively. Accuracy refers to ImageNet validation after 200 independent training epochs. For LANA's assessment, the largest FLOPs network in NAS-Bench-201 serves as the teacher net, with validation accuracy changes noted for each block. As shown in Figure 11(b), while LANA is tailored for macrospaces and struggles in micro spaces, MathNAS exhibits consistent prediction accuracy even in the micro-space NAS-Bench-201, underscoring its search space adaptability.

![](images/a8e8c2a98c599e4922a13ddbe7a4130a195bbbcd19b5494ff25e7cca513f358c.jpg)

<details>
<summary>scatter</summary>

| Actual topk1(%) | Predict topk1(%) |
| --------------- | ---------------- |
| 76.0            | 76.0             |
| 77.0            | 77.5             |
| 78.0            | 78.5             |
| 79.0            | 79.5             |
</details>

(a) MathNAS

![](images/2f875a053f4d740c07656ba5fe33093fa501fce28a5102c1242c5e85b86c3b27.jpg)

<details>
<summary>scatter</summary>

| Actual topk1(%) | Predict topk1(%) | Density |
| --------------- | ---------------- | ------- |
| 76.0            | 76.0             | 0.5     |
| 77.0            | 77.5             | 1.0     |
| 78.0            | 78.5             | 1.5     |
| 79.0            | 79.5             | 2.0     |
</details>

(b) LANA

![](images/1c175b1b2ce04d026a962f37aa5cd03fd02320d39fc4a77660fb5c246bb07d93.jpg)

<details>
<summary>bar</summary>

| Metric | LANA | MathNAS |
| :--- | :--- | :--- |
| RMSE | 0.201 | 0.124 |
| MAE | 0.161 | 0.097 |
</details>

(c) Comparison

Figure 10: Evaluation of accuracy prediction capabilities of MathNAS versus LANA in the MobileNetV3 (x 1.2) search space on ImageNet.   
![](images/bc1012f41b38375d8b74b33f744fda171c5a7f975c443b46c7827a009e6423e5.jpg)

<details>
<summary>scatter</summary>

| Actual topk1 | Predict topk1 | Density |
| ------------ | ------------- | ------- |
| 10           | 10            | 0.002   |
| 20           | 20            | 0.004   |
| 30           | 30            | 0.006   |
| 40           | 40            | 0.008   |
| 50           | 50            | 0.010   |
</details>

(a) MathNAS

![](images/be522b3a081a431f1562bdfe4f731036a8250c12cfee09d9ecb05e84477a9878.jpg)

<details>
<summary>scatter</summary>

| Actual topk1 | Predict topk1 | Density |
| ------------ | ------------- | ------- |
| 10           | 32.5          | 0.002   |
| 20           | 35.0          | 0.004   |
| 30           | 37.5          | 0.006   |
| 40           | 40.0          | 0.010   |
| 50           | 42.5          | 0.008   |
</details>

(b) LANA

![](images/73569b8a01af3213e53edc1551b4b243034cde84250c1f7e1fb4921e7652ad72.jpg)

<details>
<summary>bar</summary>

| Metric | LANA | MathNAS |
| :--- | :--- | :--- |
| RMSE | 7.607 | 2.247 |
| MAE | 5.419 | 1.696 |
</details>

(c) Comparison   
Figure 11: Accuracy prediction comparison between MathNAS and LANA within the NAS-Bench-201 search space.

Overall, MathNAS solves the following limitations compared to previous blockwise methods:

\- It does not depend on distillation but proposes a new block evaluation method based on the observed relationship between FLOPs and delta-Accuracy. This method is mathematically efficient and succinct, and it has been theoretically validated across different search spaces.

- It applies to a wider variety of search spaces, beyond the classical ones apart from the macro search space, such as the micro search space of NB201.   
- Its evaluation of block performance and network accuracy prediction is more precise.   
- It can find more superior architectures based on a full-space search, holding true even when compared to non-blockwise NAS methods. Moreover, the time complexity of the search algorithm is at a polynomial level.

# D.3 Experiment on Dynamic Network

In the main body of this paper, we analyze the dynamic performance of MathNAS and state-of-the-art (SOTA) dynamic networks on real devices. Here, we extend this analysis by providing a more comprehensive comparison of the on-device performance of MathNAS and SOTA dynamic networks.

Search Time. We delve into the search time required by MathNAS under various latency constraints, comparing it with other methodologies. Figure 12 (a) reveals that the search time varies under different latency constraints. More specifically, when the latency constraint is either relatively large or small, the search duration decreases. This is primarily due to the pruning process undertaken using the branch and bound method, which improves search efficiency by employing upper and lower bounds. A larger latency constraint results in a higher lower bound, and a smaller latency constraint results in a lower upper bound. This in turn increases the extent of pruning, reducing the search range and thus enhancing the search speed. Moreover, we compare the search times of various NAS methods. As Figure 12 (b) indicates, the search time on Linprog is longer than that on Gurobipy. This is because Gurobipy can utilize all available hardware resources on a device, whereas Linprog relies solely on a single CPU. Despite this, Linprog's search time is about 1/3300 of that of OFA, resulting in a significant reduction in NAS search time. For Gurobipy, the search time is 1/80000 of the OFA search time.

![](images/629440985c2aca8ee974454d359b8041d9a27ebd4f78e605a35e8409151ea050.jpg)

<details>
<summary>bar</summary>

| Framework | Gurobipy (s) | Linprog (s) | Gurobipy (h) | Linprog (h) |
| :--- | :--- | :--- | :--- | :--- |
| MnasNet | 0.5 | 15 | 3000 | 3000 |
| FBNet | 2 | 180 | 200 | 200 |
| OFA | 3 | 160 | 150 | 150 |
| MathNAS(linprog) | 0.4 | 120 | 0.01 | 0.01 |
| MathNAS(gurobipy) | 0.1 | 50 | 0.001 | 0.001 |
</details>

Figure 12: Experiments are performed on OFA search space and Jetson TX2 GPU. (a) Search time of MathNAS under different latency constraints. (b) Comparison of search times of MathNAS with other NAS methods.

Energy. The key objective of dynamic networks is to adaptively switch architectures based on the device environment. As such, the time and energy consumed during architecture switching, as well as the ability to respond to environmental changes, are crucial metrics for dynamic network methods. In this section, we scrutinize the performance of MathNAS in dynamic environments with latency and energy as metrics. We also utilize the UNI-T UT658 power monitor to gauge energy consumption during the switching process. Initial findings, as displayed in Figure 13, indicate that both the switching energy and latency of MathNAS are significantly less than those of Dynamic-OFA. This is primarily due to the fact that MathNAS only switches necessary sub-blocks, while Dynamic-OFA switches entire sub-nets. When deployed on Raspberry Pi, MathNAS requires approximately 80% less energy to switch sub-nets compared to Dynamic-OFA. Furthermore, the latency and energy required for MathNAS to switch sub-nets are only marginally higher than those required by AutoSlim, despite the fact that the size of candidate networks in AutoSlim is much smaller than in MathNAS.

![](images/892e5f681db05bf39c18bb5ccb6a6a975a90cdd012728aa270cd1155c4f9088a.jpg)

<details>
<summary>scatter</summary>

| Actual Latency(ms) | Predict Latency(ms) |
| ------------------ | ------------------- |
| 4000               | 4000                |
| 5000               | 5000                |
| 6000               | 6000                |
| 7000               | 7000                |
| 8000               | 8000                |
</details>

![](images/8ecf85e47d891739ca15456e7ed413e729a39c29e4a173edd085e59a8542eea3.jpg)

<details>
<summary>scatter</summary>

| Actual Energy(ms) | Predict Energy(ms) |
| ----------------- | ------------------ |
| 15000             | 15000              |
| 20000             | 20000              |
| 25000             | 25000              |
| 30000             | 30000              |
| 35000             | 35000              |
</details>

![](images/2c026cbb12ace88db2a5315422120fe7a90ea6636ae222c97b02111a7fe472bf.jpg)

<details>
<summary>scatter</summary>

| Actual topk1 | Predict topk1 |
| ------------ | ------------- |
| 76.5         | 76.5          |
| 77.0         | 77.0          |
| 77.5         | 77.5          |
| 78.0         | 78.0          |
| 78.5         | 78.5          |
| 79.0         | 79.0          |
| 79.5         | 79.5          |
</details>

![](images/caaeb2f1776ec303c8698fc17de0749566dffe084635b82b3920bc398695afd8.jpg)

<details>
<summary>scatter</summary>

| Actual topk1 | Predict topk1 |
| ------------ | ------------- |
| 10           | 10            |
| 20           | 20            |
| 30           | 30            |
| 40           | 40            |
| 50           | 50            |
</details>

![](images/23a9972711ff00e0aa84a3df32aa4cfb4c3b8ff635687124d31294c479d685b0.jpg)

<details>
<summary>line</summary>

| Total number of model switches | MathNAS(Ours) Energy Consumption (J) | DynamicOFA Energy Consumption (J) | AutoSlim MnasNet Energy Consumption (J) |
| :--- | :--- | :--- | :--- |
| 200 | 20 | 40 | 15 |
| 400 | 60 | 120 | 30 |
| 600 | 100 | 200 | 45 |
| 800 | 140 | 280 | 60 |
| 1000 | 170 | 380 | 75 |
</details>

![](images/bcb4313a51a642df99662c0d290c2fa3343e93e20248df588f0d28046362e24a.jpg)

<details>
<summary>line</summary>

| Total number of model switches | MathNAS(Ours) | DynamicOFA | AutoSlim MnasNet |
| ------------------------------ | ------------ | ---------- | ---------------- |
| 0                              | 0            | 0          | 0                |
| 200                            | 50           | 100        | 20               |
| 400                            | 100          | 200        | 40               |
| 600                            | 150          | 300        | 60               |
| 800                            | 200          | 400        | 80               |
| 1000                           | 250          | 500        | 100              |
</details>

Figure 13: Latency (top) and energy consumption (bottom) required for different dynamic networks switching on Raspberry Pi (left), TX2GPU (middle) and TX2CPU (right).

# E Searched Architecture Visualization

In this section, we provide visualizations of our searched architectures, including MathNAS-MB and MathNAS-T, and dynamic networks.

# E.1 Visualization Architecture on MobileNet-V3

Figure 14 shows the visual architecture of the network searched by MathNAS on the MobileNet-V3 search space.

# E.2 Visualization Architecture on SuperViT Space

Table 10 shows the visual architecture of the network searched by MathNAS on the SuperViT search space.

# E.3 Visualization Architecture on Dynamic Network

Figure 15 shows the Pareto-optimal sub-blocks selected on Jetson TX2 CPU/ Raspberry Pi and Jetson TX2 GPU in the MobileNet-v3 search space, respectively. GPU prefers shallow and wide block architectures, thus its sub-blocks have fewer layers but larger kernel size and channel number, while CPU prefers deep and narrow block architectures, thus its sub-blocks have more layers, but the kernel size and channel number are smaller.

# F Discussion

# F.1 Limitations and Future Work

In this section, we discuss the limitations of this work.

- The theoretical explanation of the law of FLOPs and accuracy changes proposed by Math-NAS needs to be strengthened. We intend to continue to explore and try to give a complete theoretical proof from the aspect of network interpretability.   
- Zero-shot NAS algorithms have been proven to be more efficient. Our future goal is to investigate the potential of applying MathNAS to zero-shot NAS algorithms.

![](images/2ac25b1d5cef41e1b8a6f2ad7eb40379c7d272c4cb5483f81ee2b88cd358ea60.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["224x224"] --> B["MB4 3x3"]
    B --> C["MB3 3x3"]
    C --> D["MB6 5x5"]
    D --> E["MB3 3x3"]
    E --> F["MB3 3x3"]
    F --> G["MB3 5x5"]
    G --> H["MB3 5x5"]
    H --> I["MB3 3x3"]
    I --> J["MB3 3x3"]
    J --> K["MB4 3x3"]
    K --> L["MB3 3x3"]
    L --> M["MB6 3x3"]
    M --> N["MB3 3x3"]
```
</details>

(a) Architecture Visualization of MathNAS-MB1

![](images/4d2b070f40c82b68a80dffb3dcecc7126e3a0bfef3b5f1cd2743f07485d502e7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["224x224"] --> B["MB3 3x3"]
    B --> C["MB6 7x7"]
    C --> D["MB6 7x7"]
    D --> E["MB3 7x7"]
    E --> F["MB3 7x7"]
    F --> G["MB3 7x7"]
    G --> H["MB3 7x7"]
    H --> I["MB3 7x7"]
    I --> J["MB6 7x7"]
    J --> K["MB6 3x3"]
```
</details>

(b) Architecture Visualization of MathNAS-MB2

![](images/7037afa30f7f434e770c4393b76fe4461e2f986f9b99740e9b5c5d9586dd8a72.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["224x224"] --> B["MB3 3x3"]
    B --> C["MB6 7x7"]
    C --> D["MB6 7x7"]
    D --> E["MB3 7x7"]
    E --> F["MB6 5x5"]
    F --> G["MB6 5x5"]
    G --> H["MB3 7x7"]
    H --> I["MB3 7x7"]
    I --> J["MB6 7x7"]
    J --> K["MB6 5x5"]
    K --> L["MB6 5x5"]
    L --> M["MB6 7x7"]
    M --> N["MB6 3x3"]
    N --> O["MB6 5x5"]
    O --> P["MB6 5x5"]
```
</details>

(c) Architecture Visualization of MathNAS-MB3

![](images/81956f61cdf1c7f3beac40bb2c305ce71e36e82009863af866762624eca7b68c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["224x224"] --> B["MB6 5x5"]
    B --> C["MB6 5x5"]
    C --> D["MB6 3x3"]
    D --> E["MB6 5x5"]
    E --> F["MB6 7x7"]
    F --> G["MB6 5x5"]
    G --> H["MB3 7x7"]
    H --> I["MB3 7x7"]
    I --> J["MB6 7x7"]
    J --> K["MB6 7x7"]
    K --> L["MB6 7x7"]
    L --> M["MB4 7x7"]
    M --> N["MB6 7x7"]
    N --> O["MB6 5x5"]
    O --> P["MB6 5x5"]
    P --> Q["MB6 7x7"]
    Q --> R["MB6 5x5"]
    R --> S["MB6 3x3"]
    S --> T["MB6 5x5"]
    T --> U["MB4 7x7"]
```
</details>

(d) Architecture Visualization of MathNAS-MB4   
Figure 14: Architecture Visualization of MathNAS-MB Network.

# F.2 Potential negative societal impact

Our proposed technique for rapidly designing network architectures may lead to unemployment of network architecture designers. The technology can also be misused by those who might create evil artificial intelligence.

Table 10: Architecture visualization of MathNAS-T model. ‘c’ denotes the number of output channels, ‘d’ denotes the number of layers, ‘ks’ denotes kernel size, ‘e’ denotes expansion ratio, ‘k’ denotes the number of windows, and ‘s’ denotes stride. 

<table><tr><td></td><td>MathNAS-T1</td><td>MathNAS-T2</td><td>MathNAS-T3</td><td>MathNAS-T4</td><td>MathNAS-T5</td></tr><tr><td rowspan="4">Conv</td><td>c:16</td><td>c:16</td><td>c:24</td><td>c:16</td><td>c:24</td></tr><tr><td>d:1</td><td>d:1</td><td>d:1</td><td>d:1</td><td>d:1</td></tr><tr><td>ks:3</td><td>ks:3</td><td>ks:3</td><td>ks:3</td><td>ks:3</td></tr><tr><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td></tr><tr><td rowspan="5">MBConv-1</td><td>c:16</td><td>c:16</td><td>c:16</td><td>c:16</td><td>c:24</td></tr><tr><td>d:3</td><td>d:3</td><td>d:1</td><td>d:1</td><td>d:2</td></tr><tr><td>ks:3</td><td>ks:3</td><td>ks:</td><td>ks:5</td><td>ks:3</td></tr><tr><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td></tr><tr><td>s:1</td><td>s:1</td><td>s:1</td><td>s:1</td><td>s:1</td></tr><tr><td rowspan="5">MBConv-2</td><td>c:24</td><td>c:24</td><td>c:24</td><td>c:24</td><td>c:24</td></tr><tr><td>d:3</td><td>d:4</td><td>d:3</td><td>d:3</td><td>d:4</td></tr><tr><td>ks:3</td><td>ks:3</td><td>ks:</td><td>ks:3</td><td>ks:3</td></tr><tr><td>e:4</td><td>e:4</td><td>e:5</td><td>e:5</td><td>e:5</td></tr><tr><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td></tr><tr><td rowspan="5">MBConv-3</td><td>c:32</td><td>c:40</td><td>c:40</td><td>c:32</td><td>c:40</td></tr><tr><td>d:3</td><td>d:3</td><td>d:4</td><td>d:6</td><td>d:6</td></tr><tr><td>ks:3</td><td>ks:3</td><td>ks:</td><td>ks:3</td><td>ks:3</td></tr><tr><td>e:4</td><td>e:4</td><td>e:4</td><td>e:4</td><td>e:5</td></tr><tr><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td></tr><tr><td rowspan="5">Transformer-4</td><td>c:64</td><td>c:64</td><td>c:64</td><td>c:72</td><td>c:72</td></tr><tr><td>d:3</td><td>d:4</td><td>d:3</td><td>d:4</td><td>d:6</td></tr><tr><td>k:3</td><td>k:3</td><td>k:3</td><td>k:3</td><td>k:3</td></tr><tr><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td></tr><tr><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td></tr><tr><td rowspan="4">Transformer-5</td><td>c:112</td><td>c:112</td><td>c:120</td><td>c:120</td><td>c:128</td></tr><tr><td>d:3</td><td>d:3</td><td>d:6</td><td>d:8</td><td>d:8</td></tr><tr><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td></tr><tr><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td></tr><tr><td rowspan="4">Transformer-6</td><td>c:176</td><td>c:176</td><td>c:184</td><td>c:176</td><td>c:184</td></tr><tr><td>d:3</td><td>d:3</td><td>d:6</td><td>d:8</td><td>d:8</td></tr><tr><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td></tr><tr><td>s:1</td><td>s:1</td><td>s:1</td><td>s:1</td><td>s:1</td></tr><tr><td rowspan="4">Transformer-7</td><td>c:208</td><td>c:208</td><td>c:216</td><td>c:216</td><td>c:216</td></tr><tr><td>d:3</td><td>d:3</td><td>d:6</td><td>d:5</td><td>d:5</td></tr><tr><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td><td>e:1</td></tr><tr><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td><td>s:2</td></tr><tr><td>MBPool</td><td>1792</td><td>1792</td><td>1792</td><td>1792</td><td>1792</td></tr><tr><td>Resolution</td><td>192</td><td>224</td><td>256</td><td>288</td><td>288</td></tr></table>

![](images/6b2b9a146e5af357fdc16c1467ad532a44cb6de7c4bbb1647e87b91ce108ba41.jpg)  
Figure 15: CPU (right) and GPU (left) Pareto-optimal sub-blocks selected in the MobileNet-V3 super-net. From top to bottom is from block1 to block5. Blocks in the same row are sub-blocks of the same super-block, and for the same device, the size of the sub-blocks increases sequentially from left to right.