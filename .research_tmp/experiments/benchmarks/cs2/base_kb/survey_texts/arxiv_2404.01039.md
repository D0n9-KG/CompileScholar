A Survey on Hypergraph Neural Networks: An In-Depth and Step-by-Step Guide 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.01039v3 [cs.LG] 25 Jul 2024 
 
 

# A Survey on Hypergraph Neural Networks: 
 An In-Depth and Step-by-Step Guide

 CCS:  Computing methodologies Machine learning Conference:  Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining; August 25–29, 2024; Barcelona, Spain Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain DOI:  10.1145/3637528.3671457 ISBN:  979-8-4007-0490-1/24/08 
 
 
 Sunwoo Kim
 
 Note:  Equal contribution
 
 Affiliation:  KAIST , Seoul , Republic of Korea 
 
 email: kswoo97@kaist.ac.kr 
 
 , 
 Soo Yong Lee
 
 Affiliation:  KAIST , Seoul , Republic of Korea 
 
 email: syleetolow@kaist.ac.kr 
 
 , 
 Yue Gao
 
 Affiliation:  Tsinghua University , Beijing , China 
 
 email: gaoyue@tsinghua.edu.cn 
 
 , 
 Alessia Antelmi
 
 Affiliation:  University of Turin , Turin , Italy 
 
 email: alessia.antelmi@unito.it 
 
 , 
 Mirko Polato
 
 Affiliation:  University of Turin , Turin , Italy 
 
 email: mirko.polato@unito.it 
 
 and 
 Kijung Shin
 
 Note:  Corresponding author
 
 Affiliation:  KAIST , Seoul , Republic of Korea 
 
 email: kijungs@kaist.ac.kr 
 
 © acmlicensed 

 Abstract. 
 
 Higher-order interactions (HOIs) are ubiquitous in real-world complex systems and applications. Investigation of deep learning for HOIs, thus, has become a valuable agenda for the data mining and machine learning communities.
As networks of HOIs are expressed mathematically as hypergraphs, hypergraph neural networks (HNNs) have emerged as a powerful tool for representation learning on hypergraphs.
Given the emerging trend, we present the first survey dedicated to HNNs, with an in-depth and step-by-step guide.
Broadly, the present survey overviews HNN architectures, training strategies, and applications.
First, we break existing HNNs down into four design components:
( i ) input features, ( ii ) input structures, ( iii ) message-passing schemes, and ( iv ) training strategies.
Second, we examine how HNNs address and learn HOIs with each of their components.
Third, we overview the recent applications of HNNs in recommendation, bioinformatics and medical science, time series analysis, and computer vision.
Lastly, we conclude with a discussion on limitations and future directions.

 
 
 
 Keywords:  Hypergraph Neural Network, Self-supervised Learning
 
 

## 1. Introduction

 
 
 
 
 (a) Co-authors of publications 
 
 
 (b) Hypergraph 
 
 Figure 1. 
An example hypergraph modeling the co-authorship relationship among five authors across three publications.
Each node represents an author, while each hyperedge includes all co-authors of a publication. 
 
 
 Higher-order interactions (HOIs) are pervasive in real-world complex systems and applications.
These relations describe multi-way or group-wise interactions, occurring from physical systems  ( Battiston and Petri, 2022 ) , microbial communities  ( Morin et al., 2022 ) , brain functions  ( Expert and Petri, 2022 ) , and social networks  ( Iacopini et al., 2022 ) , to name a few. HOIs reveal structural patterns unobserved in their pairwise counterparts and inform network dynamics.
For example, they have been shown to affect or correlate with synchronization in physical systems  ( Battiston et al., 2020 ) , bacteria invasion inhibition in microbial communities  ( Mickalide and Kuehn, 2019 ) , cortical dynamics in brains  ( Yu et al., 2011 ) , and contagion in social networks  ( de Arruda et al., 2020 ) .

 
 
 Hypergraphs mathematically express higher-order networks or networks of HOIs  ( Bianconi, 2021 ) ,
where nodes and hyperedges respectively represent entities and their HOIs.
In contrast to an edge connecting only two nodes in pairwise graphs, a hyperedge can connect any number of nodes, offering hypergraphs advantages in their descriptive power.
For instance, as shown in Fig.  1 , the co-authorship relations among researchers can be represented as a hypergraph.
With their expressiveness and flexibility, hypergraphs have been routinely used to model higher-order networks in various domains  ( de Arruda et al., 2020 ; Hao et al., 2024 ; Battiston et al., 2021 ; Feng et al., 2021 ) to uncover their structural patterns  ( Lee et al., 2024a ; Kim et al., 2023b ; Kim et al., 2023a ; Lee et al., 2020 ; Do et al., 2020 ; Lee et al., 2021 ) .

 
 
 As hypergraphs are extensively utilized, the demand grew to make predictions on them, estimating node properties or identifying missing hyperedges.
Hypergraph neural networks (HNNs) have shown strong promise in solving such problems.
For example, they have shown state-of-the-art performances in industrial and scientific applications,
including missing metabolic reaction prediction  ( Chen et al., 2023 ) ,
brain classification  ( Ji et al., 2022 ) ,
traffic forecast  ( Zhao et al., 2023a ) ,
product recommendation  ( Ji et al., 2020 ) ,
and more  ( Han et al., 2023 ; Xu et al., 2022a ; Ma et al., 2022b ) .

 
 
 The research on HNNs has been exponentially growing.
Simultaneously, further research on deep learning for higher-order networks is an imminent agenda for the data mining and machine learning communities  ( Papamarkou et al., 2024 ) .
Therefore, we provide a timely survey on HNNs that addresses the following questions:

 
 • 
 
 Encoding (Sec.  3 ). 
 How do HNNs effectively capture HOIs? 

 

 • 
 
 Training (Sec.  4 ). 
 How to encode HOIs with training objectives, especially when external labels are scarce or absent? 

 

 • 
 
 Application (Sec.  5 ). What are notable applications of HNNs? 

 

 
 
 
 Our scope is largely confined to HNNs for undirected, static, and homogeneous hypergraphs, with node classification or hyperedge prediction as their downstream tasks.
The survey aims to provide an in-depth and step-by-step guide, with HNNs’ design components (see Fig.  2 ) and their analysis (see Table  2 ).

 
 
 

## 2. Preliminaries

 
 
 
 
 Modeling higher-order interactions 
 
 
 
 Encoding: Input feature 
 
 
 
 External 
 info. 
 
 
 
 Structural 
 info. 
 
 
 
 Identity 
 info. 
 
 
 
 Encoding: Input structure 
 
 
 
 Reductive 
 transformation 
 
 
 
 Non-reductive 
 transformation 
 
 
 
 Encoding: Message passing 
 
 
 
 Target 
 selection 
 
 
 
 Message 
 representation 
 
 
 
 Aggregate 
 function 
 
 
 
 Training: Objective 
 
 
 
 Learning to 
 classify 
 
 
 
 Learning to 
 contrast 
 
 
 
 Learning to 
 generate 
 
 
 
 Feature 
 
 
 
 Label 
 
 
 
 Local 
 
 
 
 Global 
 
 
 
 Random 
 indicator 
 
 
 
 Clique 
 
 
 
 Adaptive 
 
 
 
 Star 
 
 
 
 Line 
 
 
 
 Tensor 
 
 
 
 Node to 
 node 
 
 
 
 Node to 
 hyperedge 
 
 
 
 Hyperedge 
 consistent 
 
 
 
 Hyperedge 
 dependent 
 
 
 
 Fixed 
 pooling 
 
 
 
 Learnable 
 pooling 
 
 
 
 Rule-based 
 neg. sam. 
 
 
 
 Learnable 
 neg. sam. 
 
 
 
 Node 
 level 
 
 
 
 Hyperedge 
 level 
 
 
 
 Membership 
 level 
 
 
 
 Ground 
 truth 
 
 
 
 Latent 
 
 
 Figure 2. Taxonomy on modeling higher-order interactions. The term neg. sam. denotes negative sampling. 
 
 
 In this section, we present definitions of basic concepts related to hypergraphs and HNNs.
See Table  1 for frequently-used symbols.

 
 
 A hypergraph 𝒢 = ( 𝒱 , ℰ ) \mathcal{G}=(\mathcal{V},\mathcal{E}) is defined as a set of nodes 𝒱 = { v 1 , v 2 , ⋯ , v | 𝒱 | } \mathcal{V}=\{v_{1},v_{2},\cdots,v_{|\mathcal{V}|}\} and a set of hyperedges ℰ = { e 1 , e 2 , ⋯ , e | ℰ | } \mathcal{E}=\{e_{1},e_{2},\cdots,e_{|\mathcal{E}|}\} .
Each hyperedge e j e_{j} is a non-empty subset of nodes (i.e., ∅ ≠ e j ⊆ 𝒱 \emptyset\neq e_{j}\subseteq\mathcal{V} ).
Alternatively, ℰ \mathcal{E} can be represented with an incidence matrix 𝐇 ∈ { 0 , 1 } | 𝒱 | × | ℰ | \mathbf{H}\in\{0,1\}^{|\mathcal{V}|\times|\mathcal{E}|} , where 𝐇 i , j = 1 \mathbf{H}_{i,j}=1 if v i ∈ e j v_{i}\in e_{j} and 0 0 otherwise.
The incident hyperedges of a node v i v_{i} , denoted as 𝒩 ℰ ​ ( v i ) \mathcal{N}_{\mathcal{E}}(v_{i}) , is the set of hyperedges that contain v i v_{i} (i.e., 𝒩 ℰ ​ ( v i ) = { e k ∈ ℰ : v i ∈ e k } \mathcal{N}_{\mathcal{E}}(v_{i})=\{e_{k}\in\mathcal{E}:v_{i}\in e_{k}\} ).
We assume that each node v i v_{i} and hyperedge e j e_{j} are equipped with (input) node features 𝒙 i ∈ ℝ d \boldsymbol{x}_{i}\in\mathbb{R}^{d} and hyperedge features 𝒚 j ∈ ℝ d ′ \boldsymbol{y}_{j}\in\mathbb{R}^{d^{\prime}} , respectively. 1 1 
 1 
 
 
 
 Sometimes, (external) node and hyperedge features may not be given. In such cases, one may utilize structural or identity features, as described in Sec.  3.1 . 
Similarly, we denote node and hyperedge feature matrices as 𝐗 ∈ ℝ | 𝒱 | × d \mathbf{X}\in\mathbb{R}^{|\mathcal{V}|\times d} and 𝐘 ∈ ℝ | ℰ | × d ′ \mathbf{Y}\in\mathbb{R}^{|\mathcal{E}|\times d^{\prime}} , respectively, where the i i -th row 𝐗 i \mathbf{X}_{i} corresponds to 𝒙 i \boldsymbol{x}_{i} and j j -th row 𝐘 j \mathbf{Y}_{j} corresponds to 𝒚 i \boldsymbol{y}_{i} .
In Sec.  3.1 , we detail approaches to obtain the features.

 
 
 Table 1. Frequently-used symbols 
 
 
 
 Notation | 
 Definition | 

 
 𝒢 = ( 𝒱 , ℰ ) \mathcal{G}=(\mathcal{V},\mathcal{E}) | 
 Hypergraph with nodes set 𝒱 \mathcal{V} and hyperedges set ℰ \mathcal{E} | 

 
 𝐇 ∈ { 0 , 1 } | 𝒱 | × | ℰ | \mathbf{H}\in\{0,1\}^{|\mathcal{V}|\times|\mathcal{E}|} | 
 Incidence matrix | 

 
 𝐗 ∈ ℝ | 𝒱 | × d \mathbf{X}\in\mathbb{R}^{|\mathcal{V}|\times d} , 𝐘 ∈ ℝ | ℰ | × d ′ \mathbf{Y}\in\mathbb{R}^{|\mathcal{E}|\times d^{\prime}} | 
 Node features ( 𝐗 \mathbf{X} ) and hyperedge features ( 𝐘 \mathbf{Y} ) | 

 
 𝐏 ( ℓ ) ∈ ℝ | 𝒱 | × k \mathbf{P}^{(\ell)}\in\mathbb{R}^{|\mathcal{V}|\times k} , 𝐐 ( ℓ ) ∈ ℝ | ℰ | × k ′ \mathbf{Q}^{(\ell)}\in\mathbb{R}^{|\mathcal{E}|\times k^{\prime}} | 
 ℓ \ell -th layer embeddings of nodes ( 𝐏 ( ℓ ) \mathbf{P}^{(\ell)} ) and hyperedges ( 𝐐 ( ℓ ) \mathbf{Q}^{(\ell)} ) | 

 
 𝒩 ℰ ​ ( v i ) \mathcal{N}_{\mathcal{E}}(v_{i}) | 
 Incident hyperedges of node v i v_{i} | 

 
 𝐈 n \mathbf{I}_{n} | 
 n n -by- n n identity matrix | 

 
 𝕀 ⁡ [ cond ] \mathbb{I}[\texttt{cond}] | 
 Indicator function that returns 1 if cond is True , 0 otherwise | 

 
 σ ⁡ ( ⋅ ) \sigma(\cdot) | 
 Non-linear activation function | 

 
 𝐌 i , : ≔ 𝒎 i \mathbf{M}_{i,:}\coloneqq\boldsymbol{m}_{i} | 
 i i -th row of matrix 𝐌 \mathbf{M} | 

 
 𝐌 i , j ≔ m i ​ j \mathbf{M}_{i,j}\coloneqq m_{ij} | 
 ( i , j ) (i,j) -entry of matrix 𝐌 \mathbf{M} | 

 
 
 
 Hypergraph neural networks (HNNs) are neural functions that transform given nodes, hyperedges, and their features into vector representations (i.e., embeddings).
Typically, their input is represented as either ( 𝐗 , ℰ ) (\mathbf{X},\mathcal{E}) or ( 𝐗 , 𝐘 , ℰ ) (\mathbf{X},\mathbf{Y},\mathcal{E}) .
HNNs first prepare the input hypergraph structure ℰ \mathcal{E} (Sec.  3.2 ).
Then, HNNs perform message passing between nodes (and/or hyperedges) to update their embeddings (Sec.  3.3 ).
A node (or hyperedge) message roughly refers to its vector representation for other nodes (or hyperedges) to aggregate.
The message passing operation is repeated L L times, where each iteration corresponds to one HNN layer.
Here, we denote the ℓ \ell -th layer embedding matrix of nodes and hyperedges as 𝐏 ( ℓ ) ∈ ℝ | 𝒱 | × k \mathbf{P}^{(\ell)}\in\mathbb{R}^{|\mathcal{V}|\times k} and 𝐐 ( ℓ ) ∈ ℝ | ℰ | × k ′ \mathbf{Q}^{(\ell)}\in\mathbb{R}^{|\mathcal{E}|\times k^{\prime}} , respectively.
Unless otherwise stated, we assume 𝐏 ( 0 ) = 𝐗 \mathbf{P}^{(0)}=\mathbf{X} and 𝐐 ( 0 ) = 𝐘 \mathbf{Q}^{(0)}=\mathbf{Y} .
We use 𝐈 n \mathbf{I}_{n} , ∥ \| , ⊙ \odot , and σ ⁡ ( ⋅ ) \sigma(\cdot) to denote the n n -by- n n identity matrix, vector concatenation, elementwise product, and a non-linear activation function, respectively.

 
 
 

## 3. Encoder Design Guidance

 
 In this section, we provide a step-by-step description of how HNNs encode higher-order interactions (HOIs).

 
 

### 3.1. Step 1: Design features to reflect HOIs

 
 First, HNNs require a careful choice of input node features 𝐗 ∈ ℝ | 𝒱 | × d \mathbf{X}\in\mathbb{R}^{|\mathcal{V}|\times d} and/or hyperedge features 𝐘 ∈ ℝ | ℰ | × d ′ \mathbf{Y}\in\mathbb{R}^{|\mathcal{E}|\times d^{\prime}} .
Their quality can be vital for a successful application of HNNs  ( Lee et al., 2024c ; Zhang et al., 2020 ) .
Thus, studies have crafted input features to enhance HNNs in encoding HOIs.
Three primary approaches include the use of ( i ) external features or labels, ( ii ) structural features, and ( iii ) identity features.

 
 

#### 3.1.1. External features or labels 

 
 External features or labels broadly refer to information that is not directly obtained from the hypergraph structure.
Using external features allows HNNs to capture information that may not be transparent in hypergraph structure alone.
When available, using external node features 𝐗 \mathbf{X} and hyperedge features 𝐘 \mathbf{Y} as HNN input is the standard practice.

 
 
 Some examples of node features from widely-used benchmark datasets are bag-of-words vectors  ( Yadati et al., 2019 ) , TF-IDFs  ( Dua et al., 2017 ) , visual object embeddings  ( Feng et al., 2019 ) , or noised label vectors  ( Chien et al., 2022 ) .
Interestingly, as in label propagation, HyperND  ( Prokopchik et al., 2022 ) constructs input node features 𝐗 \mathbf{X} by concatenating external node features with label vectors.
Specifically, one-hot-encoded label vectors and zero vectors are concatenated for nodes with known and unknown labels, respectively.
Since external hyperedge features are typically missing in the benchmark datasets,
in practice, input features of e j e_{j} can be obtained by averaging its constituent nodes (i.e., 𝒚 j = ∑ v k ∈ e j 𝒙 k / | e j | \boldsymbol{y}_{j}=\sum_{v_{k}\in e_{j}}\boldsymbol{x}_{k}/|e_{j}| )  ( Yan et al., 2024b ) .

 
 
 

#### 3.1.2. Structural features 

 
 On top of external features, studies have also utilized structural features as HNN input features.
Structural features are typically derived from the input hypergraph structure ℰ \mathcal{E} to capture structural proximity or similarity between nodes.
While leveraging them in addition to the structure ℰ \mathcal{E} may seem redundant,
several studies have highlighted their theoretical and empirical advantages, particularly for hyperedge prediction  ( Wan et al., 2021 ) and for transformer-based HNNs  ( Choe et al., 2023 ; Saifuddin et al., 2023a ; Liu et al., 2024 ) .

 
 
 Broadly speaking, studies have leveraged either local or global structural features.
To capture local structures around each node, some HNNs use the incidence matrix 𝐇 \mathbf{H} as part of the input features  ( Zhang et al., 2020 ; Wan et al., 2021 ; Liu et al., 2024 ) .
Notably, HyperGT  ( Liu et al., 2024 ) parameterizes its structural node features 𝐗 ′ ∈ ℝ | 𝒱 | × k \mathbf{X}^{\prime}\in\mathbb{R}^{|\mathcal{V}|\times k} and hyperedge features 𝐘 ′ ∈ ℝ | ℰ | × k \mathbf{Y}^{\prime}\in\mathbb{R}^{|\mathcal{E}|\times k} as follows: 𝐗 ′ = 𝐇 ​ 𝚯 \mathbf{X}^{\prime}=\mathbf{H}\mathbf{\Theta} and 𝐘 ′ = 𝐇 T ​ 𝚽 \mathbf{Y}^{\prime}=\mathbf{H}^{T}\mathbf{\Phi} , where 𝚯 ∈ ℝ | ℰ | × k \mathbf{\Theta}\in\mathbb{R}^{|\mathcal{E}|\times k} and 𝚽 ∈ ℝ | 𝒱 | × k \mathbf{\Phi}\in\mathbb{R}^{|\mathcal{V}|\times k} are learnable weight matrices.
Some HNNs leverage structural patterns within each hyperedge.
Intuitively, the importance or role of each node may vary depending on hyperedges.
For instance, WHATsNet  ( Choe et al., 2023 ) uses within-order positional encoding, where node centrality order within each hyperedge serves as edge-dependent node features (detailed in Sec.  3.3.2 ).
Also, a study  ( Moon et al., 2023 ) utilizes the occurrence of each hypergraphlet (i.e., a predefined pattern of local structures describing the overlaps of hyperedges within a few hops) around each node or hyperedge as input features.
Global features based on roles and proximity in the entire hypergraph context have also been adopted.
For example, Hyper-SAGNN  ( Zhang et al., 2020 ) uses a Hyper2Vec  ( Huang et al., 2019 ) variant to incorporate structural features preserving node proximity.
VilLain  ( Lee et al., 2024c ) leverages potential node label distributions inferred from the hypergraph structure.
HyperFeat  ( Do and Shin, 2024 ) aims to capture the structural identity of nodes through random walks.
THTN  ( Saifuddin et al., 2023a ) integrates learnable node centrality, uniqueness, and positional encodings.

 
 
 

#### 3.1.3. Identity features 

 
 Some HNNs use identity features, especially for recommendation applications.
Generally, identity features refer to features uniquely assigned to each node (and hyperedge), enabling HNNs to learn distinct embeddings for each node (and hyperedge)  ( Zhu et al., 2021 ; You et al., 2021 ) .
Prior studies have typically used randomly generated features or separately learnable ones  ( Ji et al., 2020 ; Xia et al., 2021 ; Xia et al., 2022b ; Xia et al., 2022a ) .

 
 
 

#### 3.1.4. Comparison with GNNs 

 
 Graph neural networks (GNNs) also require node and/or edge features for representation learning on pairwise graphs  ( Yang et al., 2016 ; Gong and Cheng, 2019 ; Yoo et al., 2022 ) , while typical structural features for GNNs  ( Grover and Leskovec, 2016 ; Dwivedi et al., 2022 ; Wang et al., 2022 ) do not focus on HOIs.

 
 
 
 

### 3.2. Step 2: Express hypergraphs to reflect HOIs

 
 Some HNNs transform the input hypergraph structure to better capture the underlying HOIs.
They utilize either ( i ) reductive or ( ii ) non-reductive expressions of hypergraph structures (See Fig.  3 ).

 
 
 
 
 
 (a) Hypergraph 
 
 
 (b) Clique-expanded graph with edge weights 
 
 
 (c) Star-expanded graph 
 
 Figure 3. An example hypergraph (a), its clique-expanded graph (b), and its star-expanded graph (c). 
 
 

#### 3.2.1. Reductive transformation 

 
 One way to represent hypergraph structure is through reductive transformation .
In this approach, each node from the original hypergraph is preserved as a node in the graph,
while hyperedges are transformed into pairwise edges (see Fig.  3 (b)).
Reductive transformation enables the direct application of methods developed for graphs, such as spectral filters  ( Feng et al., 2019 ) , to hypergraphs. However, it may result in information loss, and the original hypergraph structure may not be precisely recovered after transformation.
Reductive transformation includes two approaches: clique and adaptive expansion.
Each expansion is represented as τ : ( ℰ , 𝐗 , 𝐘 ) ↦ 𝐀 \tau:(\mathcal{E},\mathbf{X},\mathbf{Y})\mapsto\mathbf{A} , where 𝐀 ∈ ℝ | 𝒱 | × | 𝒱 | \mathbf{A}\in\mathbb{R}^{|\mathcal{V}|\times|\mathcal{V}|} .
We elaborate on the definition of each entry a i ​ j a_{ij} of 𝐀 \mathbf{A} for both expansions.

 
 
 Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. Clique expansion. 
Clique expansion converts each hyperedge e j ∈ ℰ e_{j}\in\mathcal{E} into a clique (i.e., complete subgraph) formed by the set e j e_{j} of nodes (see Fig.  3 (b)).
Consider two distinct hypergraphs: ( e 1 = { v 1 , v 2 , v 3 } e_{1}=\{v_{1},v_{2},v_{3}\} ) and ( e 1 = { v 1 , v 2 , v 3 } e_{1}=\{v_{1},v_{2},v_{3}\} , e 2 = { v 1 , v 3 } e_{2}=\{v_{1},v_{3}\} , and e 3 = { v 2 , v 3 } e_{3}=\{v_{2},v_{3}\} ).
Despite their changes,
both result in identical clique-expanded graph ( e 1 = { v 1 , v 2 } e_{1}=\{v_{1},v_{2}\} , e 2 = { v 1 , v 3 } e_{2}=\{v_{1},v_{3}\} , and e 3 = { v 2 , v 3 } e_{3}=\{v_{2},v_{3}\} ) if edges are unweighted.
This example illustrates that, in clique expansion, assigning proper edge weights is crucial for capturing HOIs.
To weigh the edges, studies  ( Feng et al., 2019 ; Tang et al., 2024 ) have utilized ( i ) the node pair co-occurrence, such that pairs appearing together more frequently in hyperedges are assigned larger weights, or ( ii ) hyperedge sizes, such that that node pairs in larger hyperedges are assigned smaller weights.
An example ( Tang et al., 2024 ) is a i ​ j = ∑ e k ∈ ℰ δ ⁡ ( v i , v j , e k ) | e k | , a_{ij}=\sum\nolimits_{e_{k}\in\mathcal{E}}\frac{\delta(v_{i},v_{j},e_{k})}{|e_{k}|}, 
where δ ⁡ ( v i , v j , e k ) = 𝕀 ⁡ [ ( { v i , v j } ⊆ e k ) ∧ ( i ≠ j ) ] \delta(v_{i},v_{j},e_{k})=\mathbb{I}[(\{v_{i},v_{j}\}\subseteq e_{k})\wedge(i\neq j)] , and 𝕀 ⁡ [ cond ] \mathbb{I}[\texttt{cond}] is an indicator function that returns 1 1 if cond is True and 0 otherwise.

 
 
 Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. Adaptive expansion. 
Within each transformed clique, some edges may be redundant or even unhelpful. Adaptive expansion selectively adds and/or weighs edges within each clique, often tailored to a given downstream task  ( Qian et al., 2023 ; Yadati et al., 2019 ) .
For example, AdE  ( Qian et al., 2023 ) uses a feature-distance-based edge weighting strategy.
It obtains projected node features 𝐗 ′ ∈ ℝ | 𝒱 | × d \mathbf{X}^{\prime}\in\mathbb{R}^{|\mathcal{V}|\times d} by 𝐗 ′ = 𝐗 ⊙ 𝐖 \mathbf{X}^{\prime}=\mathbf{X}\odot\mathbf{W} ,
where all row vectors of 𝐖 \mathbf{W} are sigmoid ​ ( MLP ​ ( ∑ v k ∈ 𝒱 𝒙 k / | 𝒱 | ) ) \texttt{sigmoid}(\texttt{MLP}(\sum_{v_{k}\in\mathcal{V}}\boldsymbol{x}_{k}/|\mathcal{V}|)) .
Then, AdE selects two distant nodes v i , j v_{i,j} and v k , j v_{k,j} within each hyperedge e j e_{j} , i.e., { v i , j , v k , j } = arg ⁡ max { v i , v k } ∈ ( e j 2 ) ​ | ∑ t = 1 d ( 𝐗 i , t ′ − 𝐗 k , t ′ ) | \{v_{i,j},v_{k,j}\}=\arg\max_{\{v_{i},v_{k}\}\in\binom{e_{j}}{2}}|\sum^{d}_{t=1}(\mathbf{X}^{\prime}_{i,t}-\mathbf{X}^{\prime}_{k,t})| .
After that, it connects the all nodes in e j e_{j} with v i , j v_{i,j} and v k , j v_{k,j} , essentially adding ℰ j ′ = { { v i , j , v t } : v t ∈ e j ∖ { v i , j } } ∪ { { v k , j , v t } : v t ∈ e j ∖ { v k , j } } \mathcal{E}^{\prime}_{j}=\{\{v_{i,j},v_{t}\}:v_{t}\in e_{j}\setminus\{v_{i,j}\}\}\cup\{\{v_{k,j},v_{t}\}:v_{t}\in e_{j}\setminus\{v_{k,j}\}\} .
AdE assigns weights to each edge in ℰ j ′ \mathcal{E}^{\prime}_{j} as follows:

 

 
 (1) | 
 | 
 a i ​ k = ∑ e j ∈ ℰ 𝕀 [ { v i , v k } ∈ ℰ ′ j ] ξ ( i , k ) ∑ { v s , v t } ∈ ( e j 2 ) ξ ⁡ ( s , t ) , \displaystyle a_{ik}=\sum_{e_{j}\in\mathcal{E}}\frac{\mathbb{I}[\{v_{i},v_{k}\}\in\mathcal{E}^{\prime}_{j}]\xi(i,k)}{\sum_{\{v_{s},v_{t}\}\in\binom{e_{j}}{2}}\xi(s,t)}, | 
 | 
 

 where ξ ⁡ ( i , k ) = exp ⁡ ( ∥ 𝒙 i − 𝒙 k ∥ 2 ​ ∑ t = 1 d ( 𝐗 i , t ′ − 𝐗 k , t ′ ) 2 / θ t 2 ) \xi(i,k)=\exp\left(\lVert\boldsymbol{x}_{i}-\boldsymbol{x}_{k}\rVert_{2}\sum_{t=1}^{d}{(\mathbf{X}^{\prime}_{i,t}-\mathbf{X}^{\prime}_{k,t})^{2}}/{\theta^{2}_{t}}\right) and θ t \theta_{t} , ∀ t ∈ [ d ] \forall t\in[d] , are learnable scalars.

 
 
 

#### 3.2.2. Non-reductive transformation 

 
 Non-reductive transformation of hypergraph structure includes star expansion  ( Chien et al., 2022 ; Choe et al., 2023 ; Wang et al., 2023c ; Saifuddin et al., 2023b ) , line expansion  ( Yang et al., 2022b ) , and tensor representation  ( Kim et al., 2022 ; Wang et al., 2024a ; Wang et al., 2024d ) .
They express hypergraph structure without information loss. That is, the hyperedges ℰ \mathcal{E} can be exactly recovered after transformation.

 
 
 Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. Star expansion. 
A star-expanded graph of a hypergraph 𝒢 = ( 𝒱 , ℰ ) \mathcal{G}=(\mathcal{V},\mathcal{E}) has two new groups of nodes: the node group , which is the same as the node set 𝒱 \mathcal{V} of 𝒢 \mathcal{G} , and the hyperedge group , consisting of nodes corresponding to the hyperedges ℰ \mathcal{E} (refer to Fig.  3 (c)).
Star expansion captures HOIs by connecting each node (i.e., a node from the node group) with the hyperedges (i.e., nodes from the hyperedge group) it belongs to, resulting a bipartite graph between the two groups.
Star expansion is expressed as
 τ : ( ℰ , 𝐗 , 𝐘 ) ↦ 𝐀 \tau:(\mathcal{E},\mathbf{X},\mathbf{Y})\mapsto\mathbf{A} , where each entry of 𝐀 ∈ ℝ ( | 𝒱 | + | ℰ | ) × ( | 𝒱 | + | ℰ | ) \mathbf{A}\in\mathbb{R}^{(|\mathcal{V}|+|\mathcal{E}|)\times(|\mathcal{V}|+|\mathcal{E}|)} is defined as

 

 
 (2) | 
 | 
 a i ​ j = { 𝕀 [ v i ∈ e j − | 𝒱 | ] , if  ​ 1 ≤ i ≤ | 𝒱 | j ≤ | 𝒱 | + | ℰ | , 𝕀 [ v j ∈ e i − | 𝒱 | ] , if  ​ 1 ≤ j ≤ | 𝒱 | i ≤ | 𝒱 | + | ℰ | , 0 , otherwise. a_{ij}=\begin{cases}\mathbb{I}[v_{i}\in e_{j-|\mathcal{V}|}], \text{if }1\leq i\leq|\mathcal{V}| j\leq|\mathcal{V}|+|\mathcal{E}|,\\
\mathbb{I}[v_{j}\in e_{i-|\mathcal{V}|}], \text{if }1\leq j\leq|\mathcal{V}| i\leq|\mathcal{V}|+|\mathcal{E}|,\\
0, \text{otherwise.}\end{cases}\vskip-2.84526pt | 
 | 
 

 Here, we assume WLOG that the corresponding index of v i ∈ 𝒱 v_{i}\in\mathcal{V} in 𝐀 \mathbf{A} is i i , and the corresponding index of e j ∈ ℰ e_{j}\in\mathcal{E} in 𝐀 \mathbf{A} is | 𝒱 | + j |\mathcal{V}|+j .

 
 
 Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. Line expansion. 
In a line-expanded graph ( Yang et al., 2022b ) of a hypergraph of 𝒢 = ( 𝒱 , ℰ ) \mathcal{G}=(\mathcal{V},\mathcal{E}) , each pair of a node and a hyperedge containing it is represented as a distinct node.
That is, its node set is { ( v i , e j ) : v i ∈ e j , e j ∈ ℰ } \{(v_{i},e_{j}):v_{i}\in e_{j},e_{j}\in\mathcal{E}\} .
Edges are established between these nodes to connect each pair of distinct nodes ( v i , e j ) (v_{i},e_{j}) and ( v k , e l ) (v_{k},e_{l}) , where i = k i=k or j = l j=l .

 
 
 Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. Tensor expression. 
Several recent HNNs represent hypergraphs as tensors  ( Kim et al., 2022 ; Wang et al., 2024d ) .
For example, T-HyperGNNs  ( Wang et al., 2024a ) expresses a k − k- uniform (i.e., | e j | = k , ∀ e j ∈ ℰ |e_{j}|=k,\forall e_{j}\in\mathcal{E} ) hypergraph 𝒢 = ( 𝒱 , ℰ ) \mathcal{G}=(\mathcal{V},\mathcal{E}) with a k − k- order tensor 𝒜 ∈ ℝ | 𝒱 | k \mathbf{\mathcal{A}}\in\mathbb{R}^{|\mathcal{V}|^{k}} .
That is, if k = 3 k=3 , 𝒜 i , j , k = 1 \mathbf{\mathcal{A}}_{i,j,k}=1 if { v i , v j , v k } ∈ ℰ \{v_{i},v_{j},v_{k}\}\in\mathcal{E} , and 𝒜 i , j , k = 0 \mathbf{\mathcal{A}}_{i,j,k}=0 otherwise.

 
 
 

#### 3.2.3. Comparison with GNNs 

 
 GNNs typically use the adjacency matrix  ( Kipf and Welling, 2017 ; Wu et al., 2019 ) , the personalize PageRank matrix  ( Gasteiger et al., 2019 ; Chien et al., 2020 ) , and the Laplacian matrix  ( Luan et al., 2022 ) to represent the graph structure.

 
 
 
 

### 3.3. Step 3: Pass messages to reflect HOIs

 
 With input features (Sec.  3.1 ) and structure (Sec.  3.2 ), HNNs learn node (and hyperedge) embeddings.
They use neural message passing functions for each node (and hyperedge) to aggregate messages, i.e., information, from other nodes (and hyperedges).
Three questions arise:
( i ) whose messages should be aggregated?
( ii ) what messages should be aggregated?
( iii ) how should they be aggregated?

 
 

#### 3.3.1. Whose messages to aggregate (target selection) 

 
 For message passing, we should decide whose message to aggregate, typically based on the structural expression of the input hypergraph (Sec.  3.2 ).
We provide three representative examples: one clique-expansion-based approach and two star-expansion-based ones. 2 2 
 2 
 
 
 
 Regarding target selection,
adaptive-expansion-  ( Qian et al., 2023 ) , line-expansion-  ( Yang et al., 2022a ) and tensor-representation-based  ( Wang et al., 2024d ) are similar to clique-expanded ones ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ). 

 
 
 On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) On clique-expanded graphs ( 𝒱 → 𝒱 \mathcal{V}\rightarrow\mathcal{V} ) .
Similar to typical GNNs,
clique-expansion-based HNNs perform message passing between neighboring nodes  ( Feng et al., 2019 ; Prokopchik et al., 2022 ; Yadati et al., 2019 ; Bai et al., 2021b ; Tang et al., 2024 ; Benko et al., 2024 ; Hayhoe et al., 2023 ) .
They also often incorporate techniques that are effective in applying GNNs.
This is expected since clique expansion transforms a hypergraph into a homogeneous, pairwise graph.
A notable instance is SHNN  ( Tang et al., 2024 ) , which constructs a propagation matrix 𝐖 \mathbf{W} from 𝐀 \mathbf{A} (Sec.  3.2.1 ) using a re-normalization trick  ( Kipf and Welling, 2017 ) as 𝐖 = 𝐃 ~ − 1 2 ​ 𝐀 ~ ​ 𝐃 ~ − 1 2 \mathbf{W}=\tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}} , where 𝐀 ~ = 𝐀 + 𝐈 | 𝒱 | \tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}_{|\mathcal{V}|} and 𝐃 ~ \tilde{\mathbf{D}} is the diagonal degree matrix, i.e., 𝐃 ~ i , i = ∑ k = 1 | 𝒱 | 𝐀 ~ i , k \tilde{\mathbf{D}}_{i,i}=\sum^{|\mathcal{V}|}_{k=1}\tilde{\mathbf{A}}_{i,k} .
Then, node embeddings at each ℓ \ell -th layer are updated using 𝐖 \mathbf{W} as:

 

 
 (3) | 
 | 
 𝐏 ( ℓ ) = σ ⁡ ( ( ( 1 − α ℓ ) ​ 𝐖𝐏 ( ℓ − 1 ) + α ℓ ​ 𝐏 ( 0 ) ) ​ ( ( 1 − β ℓ ) ​ 𝐈 k + β ℓ ​ 𝚯 ( ℓ ) ) ) , \mathbf{P}^{(\ell)}=\sigma\left(((1-\alpha_{\ell})\mathbf{W}\mathbf{P}^{(\ell-1)}+\alpha_{\ell}\mathbf{{P}}^{(0)})((1-\beta_{\ell})\mathbf{I}_{k}+\beta_{\ell}\mathbf{\Theta}^{(\ell)})\right), | 
 | 
 

 where α ℓ , β ℓ ∈ [ 0 , 1 ] \alpha_{\ell},\beta_{\ell}\in[0,1] are hyperparameters, 𝚯 ( ℓ ) ∈ ℝ k × k \mathbf{\Theta}^{(\ell)}\in\mathbb{R}^{k\times k} is a learnable weight matrix, and 𝐏 ( 0 ) = MLP ​ ( 𝐗 ) \mathbf{{P}}^{(0)}=\texttt{MLP}(\mathbf{X}) .

 
 
 On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). On star-expanded graphs ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} and ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} ). 
In HNNs based on star expansion, message passing occurs from the node group to the hyperedge group ( 𝒱 → ℰ \mathcal{V}\rightarrow\mathcal{E} ) and vice versa ( ℰ → 𝒱 \mathcal{E}\rightarrow\mathcal{V} )  ( Wang et al., 2023c ; Chien et al., 2022 ; Choe et al., 2023 ; Dong et al., 2020 ; Yan et al., 2024b ) , either sequentially or simultaneously.

 
 
 First, we illustrate sequential message passing using ED-HNN ( Wang et al., 2023c ) .
Its message passing at each ℓ \ell -th layer for each node v i ∈ 𝒱 v_{i}\in\mathcal{V} is formalized as follows:

 

 
 (4) | 
 | 
 | 
 𝒒 j ( ℓ ) = ∑ v k ∈ e j MLP 1 ​ ( 𝒑 k ( ℓ − 1 ) ) , \displaystyle\boldsymbol{q}^{(\ell)}_{j}=\sum_{v_{k}\in e_{j}}\texttt{MLP}_{1}\left(\boldsymbol{p}^{(\ell-1)}_{k}\right), | 
 | 
 
 
 (5) | 
 | 
 | 
 𝓹 i ( ℓ ) = ∑ e k ∈ 𝒩 ℰ ​ ( v i ) MLP 2 ( [ 𝒑 i ( ℓ − 1 ) ∥ 𝒒 k ( ℓ ) ] ) , \displaystyle\boldsymbol{\mathscr{p}}^{(\ell)}_{i}=\sum_{e_{k}\in\mathcal{N}_{\mathcal{E}}(v_{i})}\texttt{MLP}_{2}\left(\left[\boldsymbol{p}^{(\ell-1)}_{i}\lVert\boldsymbol{q}^{(\ell)}_{k}\right]\right), | 
 | 
 
 
 (6) | 
 | 
 | 
 𝒑 i ( ℓ ) = MLP 3 ( [ 𝒑 i ( ℓ − 1 ) ∥ 𝓹 i ( ℓ ) ∥ 𝒙 i ⊕ | 𝒩 ℰ ( v i ) | ] ) , \displaystyle\boldsymbol{p}^{(\ell)}_{i}=\texttt{MLP}_{3}\left(\left[\boldsymbol{p}_{i}^{(\ell-1)}\lVert\boldsymbol{\mathscr{p}}^{(\ell)}_{i}\lVert\boldsymbol{x}_{i}\oplus|\mathcal{N}_{\mathcal{E}}(v_{i})|\right]\right), | 
 | 
 

 where 𝒙 ⊕ c \boldsymbol{x}\oplus c denotes the concatenation of vector 𝒙 \boldsymbol{x} and scalar c c . MLP 1 \texttt{MLP}_{1} , MLP 2 \texttt{MLP}_{2} , and MLP 3 \texttt{MLP}_{3} are MLPs shared across all layers.
Note that, in Eq. ( 4 ), hyperedge embeddings are updated by aggregating the embeddings of their constituent nodes.
Subsequently, in Eq. ( 5 ) and Eq. ( 6 ), node embeddings are updated by aggregating transformed embeddings of incident hyperedges.
Here, the message passing in each direction (Eq. ( 4 ) and Eq. ( 5 )) occurs sequentially.

 
 
 Second, we present an example of simultaneous message passing with HDS ode   ( Yan et al., 2024b ) .
Its message passing at each ℓ \ell -th layer for node v i ∈ 𝒱 v_{i}\in\mathcal{V} and hyperedge e j ∈ ℰ e_{j}\in\mathcal{E} is formalized as follows:

 

 
 (7) | 
 | 
 𝓹 i ( ℓ ) \displaystyle\boldsymbol{\mathscr{p}}^{(\ell)}_{i} | 
 = 𝒑 i ( ℓ − 1 ) + σ ⁡ ( 𝒑 i ( ℓ − 1 ) ​ 𝚯 ( v ) + 𝒃 ( v ) ) , \displaystyle=\boldsymbol{p}_{i}^{(\ell-1)}+\sigma(\boldsymbol{p}^{(\ell-1)}_{i}\mathbf{\Theta}_{(v)}+\boldsymbol{b}_{(v)}), | 
 | 
 
 
 (8) | 
 | 
 𝓺 j ( ℓ ) \displaystyle\boldsymbol{\mathscr{q}}^{(\ell)}_{j} | 
 = 𝒒 j ( ℓ − 1 ) + σ ⁡ ( 𝒒 j ( ℓ − 1 ) ​ 𝚯 ( e ) + 𝒃 ( e ) ) , \displaystyle=\boldsymbol{q}_{j}^{(\ell-1)}+\sigma(\boldsymbol{q}^{(\ell-1)}_{j}\mathbf{\Theta}_{(e)}+\boldsymbol{b}_{(e)}), | 
 | 
 
 
 (9) | 
 | 
 𝒑 i ( ℓ ) \displaystyle\boldsymbol{p}^{(\ell)}_{i} | 
 = ( 1 − α ( v ) ) ​ 𝓹 i ( ℓ ) + α ( v ) | 𝒩 ℰ ​ ( v i ) | ​ ∑ e l ∈ 𝒩 ℰ ​ ( v i ) 𝓺 l ( ℓ ) , \displaystyle=(1-\alpha_{(v)})\boldsymbol{\mathscr{p}}^{(\ell)}_{i}+\frac{\alpha_{(v)}}{|\mathcal{N}_{\mathcal{E}}(v_{i})|}\sum\nolimits_{e_{l}\in\mathcal{N}_{\mathcal{E}}(v_{i})}\boldsymbol{\mathscr{q}}^{(\ell)}_{l}, | 
 | 
 
 
 (10) | 
 | 
 𝒒 j ( ℓ ) \displaystyle\boldsymbol{q}^{(\ell)}_{j} | 
 = ( 1 − α ( e ) ) ​ 𝓺 i ( ℓ ) + α ( e ) | e j | ​ ∑ v l ∈ e j 𝓹 l ( ℓ ) , \displaystyle=(1-\alpha_{(e)})\boldsymbol{\mathscr{q}}^{(\ell)}_{i}+\frac{\alpha_{(e)}}{|e_{j}|}\sum\nolimits_{v_{l}\in e_{j}}\boldsymbol{\mathscr{p}}^{(\ell)}_{l}, | 
 | 
 

 where α ( v ) , α ( e ) ∈ [ 0 , 1 ] \alpha_{(v)},\alpha_{(e)}\in[0,1] are hyperparameters, 𝚯 ( v ) , 𝚯 ( e ) ∈ ℝ k × k \mathbf{\Theta}_{(v)},\mathbf{\Theta}_{(e)}\in\mathbb{R}^{k\times k} are learnable weight matrices, and 𝒃 ( v ) , 𝒃 ( e ) ∈ ℝ k \boldsymbol{b}_{(v)},\boldsymbol{b}_{(e)}\in\mathbb{R}^{k} are learnable biases.
After projecting node and hyperedge embeddings (Eq. ( 7 ) and Eq. ( 8 )), each node embedding is updated by aggregating the projected embeddings of its incident hyperedge (Eq. ( 9 )), and each hyperedge embedding is updated by aggregating the projected embeddings of its constituent nodes (Eq. ( 10 )).
The message passing in each direction (Eq. ( 9 ) and Eq. ( 10 )) occurs simultaneously.

 
 
 

#### 3.3.2. What messages to aggregate (message representation) 

 
 After choosing message targets, the next step is determining message representations .
HNNs typically use embeddings from the previous layer as messages, which we term hyperedge-consistent messages   ( Dong et al., 2020 ; Huang and Yang, 2021 ) .
In contrast, several recent studies propose adaptive message transformation based on its target, which we refer to as hyperedge-dependent messages   ( Choe et al., 2023 ; Telyatnikov et al., 2023 ; Aponte et al., 2022 ; Zheng and Worring, 2024 ) .

 
 
 Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. Hyperedge-consistent messages. 
In this widely-used approach ( Huang and Yang, 2021 ; Chien et al., 2022 ; Yan et al., 2024b ) , embeddings from the previous layer are directly treated as vector messages.
A notable example is UniGNN  ( Huang and Yang, 2021 ) , a family of HNNs that obtain node (and hyperedge) embeddings by aggregating the embeddings from its incident hyperedges (or constituent nodes).
UniGIN, a special case of UniGNN, is formalized as follows:

 
 
 

 
 | 
 𝒒 j ( ℓ ) = ∑ v l ∈ e j 𝒑 k ( ℓ − 1 ) ; 𝒑 i ( ℓ ) = ( ( 1 + ϵ ) ​ 𝒑 i ( ℓ − 1 ) + ∑ e l ∈ 𝒩 ℰ ​ ( v i ) 𝒒 l ( ℓ ) ) ​ 𝚯 ( ℓ ) , \boldsymbol{q}^{(\ell)}_{j}=\sum_{v_{l}\in e_{j}}\boldsymbol{p}_{k}^{(\ell-1)}\ ;\ \boldsymbol{p}^{(\ell)}_{i}=\left((1+\epsilon)\boldsymbol{p}_{i}^{(\ell-1)}+\sum_{e_{l}\in\mathcal{N}_{\mathcal{E}}(v_{i})}\boldsymbol{q}^{(\ell)}_{l}\right)\mathbf{\Theta}^{(\ell)}, | 
 | 
 

 where ϵ ∈ ℝ \epsilon\in\mathbb{R} and 𝚯 ( ℓ ) ∈ ℝ k × k ′ \mathbf{\Theta}^{(\ell)}\in\mathbb{R}^{k\times k^{\prime}} respectively can either be a learnable or fixed scalar and is a learnable weight matrix.

 
 
 Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. Hyperedge-dependent messages. 
The role or importance of a node may vary across the hyperedges it belongs to  ( Chitra and Raphael, 2019 ; Choe et al., 2023 ) .
Several studies  ( Choe et al., 2023 ; Telyatnikov et al., 2023 ; Aponte et al., 2022 ) have devised hyperedge-dependent node messages, enabling a node to send tailored messages to each hyperedge it belongs to.
For example, MultiSetMixer  ( Telyatnikov et al., 2023 ) learns different node messages for each incident hyperedge to aggregate with the following message passing function:

 

 
 (11) | 
 | 
 𝒒 j ( ℓ ) \displaystyle\boldsymbol{q}^{(\ell)}_{j} | 
 = 1 | e j | ​ ∑ v k ∈ e j 𝒑 k , j ( ℓ − 1 ) + MLP 1 ( ℓ ) ​ ( LN ​ ( 1 | e j | ​ ∑ v k ∈ e j 𝒑 k , j ( ℓ − 1 ) ) ) , \displaystyle=\frac{1}{|e_{j}|}\sum_{v_{k}\in e_{j}}\boldsymbol{p}^{(\ell-1)}_{k,j}+\texttt{MLP}^{(\ell)}_{1}\left(\texttt{LN}\left(\frac{1}{|e_{j}|}\sum_{v_{k}\in e_{j}}\boldsymbol{p}^{(\ell-1)}_{k,j}\right)\right), | 
 | 
 
 
 (12) | 
 | 
 𝒑 i , j ( ℓ ) \displaystyle\boldsymbol{p}^{(\ell)}_{i,j} | 
 = 𝒑 i , j ( ℓ − 1 ) + MLP 2 ( ℓ ) ​ ( LN ​ ( 𝒑 i , j ( ℓ − 1 ) ) ) + 𝒒 j ( ℓ ) , \displaystyle=\boldsymbol{p}^{(\ell-1)}_{i,j}+\texttt{MLP}^{(\ell)}_{2}\left(\texttt{LN}\left(\boldsymbol{p}^{(\ell-1)}_{i,j}\right)\right)+\boldsymbol{q}^{(\ell)}_{j}, | 
 | 
 

 where 𝒑 i , j ( ℓ ) \boldsymbol{p}^{(\ell)}_{i,j} is the ℓ \ell -th layer message of v i v_{i} that is dependent on e j e_{j} , MLP 1 ( ℓ ) \texttt{MLP}^{(\ell)}_{1} and MLP 2 ( ℓ ) \texttt{MLP}^{(\ell)}_{2} are MLPs, and LN is layer normalization  ( Ba et al., 2016 ) .

 
 
 Alternatively, some HNNs update messages based on hyperedge-dependent node features.
WHATsNet  ( Choe et al., 2023 ) introduces within-order positional encoding ( wope ) to adapt node messages for each target.
Within each hyperedge, WHATsNet ranks constituent nodes according to their centralities for positional encoding.
Formally, let 𝐅 ∈ ℝ | 𝒱 | × T \mathbf{F}\in\mathbb{R}^{|\mathcal{V}|\times T} be a node centrality matrix, where T T and 𝐅 i , t \mathbf{F}_{i,t} respectively denote the number of centrality measures (e.g., node degree) and the t t -th centrality measure score of node v i v_{i} .
The order of an element c c in a set 𝒞 \mathcal{C} is defined as Order ( c , 𝒞 ) = ∑ c ′ ∈ 𝒞 𝕀 [ c ′ ≤ c ] \texttt{Order}(c,\mathcal{C})=\sum_{c^{\prime}\in\mathcal{C}}\mathbb{I}[c^{\prime}\leq c] .
Then, wope of a node v i v_{i} at a hyperedge e j e_{j} is defined as follows:

 

 
 (13) | 
 | 
 wope ( v i , e j ) = ∥ t = 1 T 1 | e j | Order ( 𝐅 i , t , { 𝐅 i , t : v i ∈ e j } ) . \texttt{wope}(v_{i},e_{j})={\big\|}_{t=1}^{T}\frac{1}{|e_{j}|}\texttt{Order}(\mathbf{F}_{i,t},\{\mathbf{F}_{i,t}:v_{i}\in e_{j}\}). | 
 | 
 

 Finally, hyperedge-dependent node messages are defined as follows:

 

 
 (14) | 
 | 
 𝒑 i , j ( ℓ ) = 𝒑 i ( ℓ ) + wope ​ ( v i , e j ) ​ 𝚿 ( ℓ ) , \boldsymbol{p}^{(\ell)}_{i,j}=\boldsymbol{p}^{(\ell)}_{i}+\texttt{wope}(v_{i},e_{j})\mathbf{\Psi}^{(\ell)}, | 
 | 
 

 where 𝚿 ( ℓ ) ∈ ℝ T × k \mathbf{\Psi}^{(\ell)}\in\mathbb{R}^{T\times k} is a learnable projection matrix. 3 3 
 3 
 
 
 
 Similarly,
each hyperedge e j e_{j} ’s message to each node v i v_{i} at the ℓ \ell -th layer is defined as
 𝒒 j , i ( ℓ ) = 𝒒 j ( ℓ ) + wope ​ ( v i , e j ) ​ 𝚿 ( ℓ ) \boldsymbol{q}^{(\ell)}_{j,i}=\boldsymbol{q}^{(\ell)}_{j}+\texttt{wope}(v_{i},e_{j})\mathbf{\Psi}^{(\ell)} . WHATsNet aggregates { q k , i ( ℓ ) : e k ∈ 𝒩 ( ℰ ) ( v i ) \{q^{(\ell)}_{k,i}:e_{k}\in\mathcal{N}_{(\mathcal{E})}(v_{i}) } to obtain 𝐩 i ( ℓ ) \mathbf{p}^{(\ell)}_{i} via set attention proposed by Lee et al. (2019) .
We omit the detailed message passing function since we focus on describing how dependent messages are obtained. 

 
 
 

#### 3.3.3. How to aggregate messages (aggregation function) 

 
 The last step is to decide how to aggregate the received messages for each node (and hyperedge).

 
 
 Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. Fixed pooling. 
Many HNNs use fixed pooling functions, including summation  ( Wang et al., 2023c ; Huang and Yang, 2021 ) and average  ( Wang et al., 2024c ; Gao et al., 2022 ) .
For example, ED-HNN  ( Wang et al., 2023c ) uses summation to aggregate the embeddings of constituent nodes (or incident hyperedges), as described in Eq. ( 4 ) and Eq. ( 5 ).
Clique-expansion-based HNNs without adaptive edge weights also fall into this category  ( Tang et al., 2024 ; Qu et al., 2023 ) .
For example, SHNN  ( Tang et al., 2024 ) uses a fixed propagation matrix 𝐖 \mathbf{W} (see Eq. ( 3 )) to aggregate node embeddings. Specifically, 𝒑 i ( ℓ ) = ∑ v k ∈ 𝒱 𝐖 i , j ​ 𝒑 k ( ℓ − 1 ) \boldsymbol{p}^{(\ell)}_{i}=\sum_{v_{k}\in\mathcal{V}}\mathbf{W}_{i,j}\boldsymbol{p}^{(\ell-1)}_{k} .

 
 
 Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. Learnable pooling. 
Several recent HNNs enhance their pooling functions through attention mechanisms, allowing for weighting messages during aggregation.
Two prominent styles are target-agnostic attention   ( Chien et al., 2022 ; Chai et al., 2024 ) and target-aware attention   ( Choe et al., 2023 ; Saifuddin et al., 2023b ) .

 
 
 Target-agnostic attention functions consider the relations among messages themselves.
AllSetTransformer  ( Chien et al., 2022 ) is an example.
Denote the embeddings of the incident hyperedges of v i v_{i} at each ℓ \ell -th layer as 𝒮 ( ℓ ) ​ ( v i ) ≔ { 𝒒 k ( ℓ ) : e k ∈ 𝒩 ℰ ​ ( v i ) } \mathcal{S}^{(\ell)}(v_{i})\coloneqq\{\boldsymbol{q}^{(\ell)}_{k}:e_{k}\in\mathcal{N}_{\mathcal{E}}(v_{i})\} and its matrix expression as 𝐒 ( ℓ , i ) ∈ ℝ | 𝒮 ( ℓ ) ​ ( v i ) | × k \mathbf{S}^{(\ell,i)}\in\mathbb{R}^{|\mathcal{S}^{(\ell)}(v_{i})|\times k} .
Then, 𝒑 i ( ℓ ) \boldsymbol{p}_{i}^{(\ell)} is derived from 𝐒 ( ℓ , i ) \mathbf{S}^{(\ell,i)} as follows:

 

 
 (15) | 
 | 
 MH ( 𝜽 , 𝐒 ) = ∥ t = 1 h ( ω ( θ t ( MLP t , 1 ( ℓ ) ( 𝐒 ) ) T ) MLP t , 2 ( ℓ ) ( 𝐒 ) ) , \displaystyle\texttt{MH}(\boldsymbol{\theta},\mathbf{S})=\|_{t=1}^{h}\left(\omega\left(\theta_{t}\left(\texttt{MLP}^{(\ell)}_{t,1}(\mathbf{S})\right)^{T}\right)\texttt{MLP}^{(\ell)}_{t,2}(\mathbf{S})\right), | 
 | 
 
 
 | 
 𝒑 i ( ℓ ) = LN ​ ( 𝓹 i ( ℓ ) + MLP 3 ( ℓ ) ​ ( 𝓹 i ( ℓ ) ) ) ; 𝓹 i ( ℓ ) = LN ​ ( 𝜽 + MH ​ ( 𝜽 , 𝐒 ( ℓ , i ) ) ) , \displaystyle\boldsymbol{p}^{(\ell)}_{i}=\texttt{LN}\left(\boldsymbol{\mathscr{p}}^{(\ell)}_{i}+\texttt{MLP}^{(\ell)}_{3}\left(\boldsymbol{\mathscr{p}}^{(\ell)}_{i}\right)\right);\boldsymbol{\mathscr{p}}^{(\ell)}_{i}=\texttt{LN}\left(\boldsymbol{\theta}+\texttt{MH}\left(\boldsymbol{\theta},\mathbf{S}^{(\ell,i)}\right)\right), | 
 | 
 

 where LN is layer normalization  ( Ba et al., 2016 ) , ω ⁡ ( ⋅ ) \omega(\cdot) is row-wise softmax, 𝜽 = ∥ t = 1 T θ t \boldsymbol{\theta}=\|_{t=1}^{T}\theta_{t} is a learnable vector, and MLP t , 1 \texttt{MLP}_{t,1} , MLP t , 2 \texttt{MLP}_{t,2} , and MLP 3 \texttt{MLP}_{3} are MLPs.
Note that Eq. ( 15 ) is a widely-used multi-head attention operation  ( Vaswani et al., 2017 ) , where 𝜽 \boldsymbol{\theta} serves as queries, and 𝐒 \mathbf{S} serves as keys and values.
This process is target-agnostic since it considers only the global variables 𝜽 \boldsymbol{\theta} and the embeddings 𝐒 \mathbf{S} of incident hyperedges, without considering the embedding of the target v i v_{i} itself.

 
 
 In target-aware attention approaches, target information is incorporated to compute attention weights.
HyGNN  ( Saifuddin et al., 2023b ) is an example, with the following message passing function:

 

 
 (16) | 
 | 
 𝒑 i ( ℓ ) \displaystyle\boldsymbol{p}_{i}^{(\ell)} | 
 = σ ⁡ ( ∑ e k ∈ 𝒩 ℰ ​ ( v i ) Att ( 𝒱 ) ( ℓ ) ​ ( 𝒒 k ( ℓ − 1 ) , 𝒑 i ( ℓ − 1 ) ) ​ 𝒒 k ( ℓ − 1 ) ​ 𝚯 ( ℓ , 1 ) ∑ e s ∈ 𝒩 ℰ ​ ( v i ) Att ( 𝒱 ) ( ℓ ) ​ ( 𝒒 s ( ℓ − 1 ) , 𝒑 i ( ℓ − 1 ) ) ) , \displaystyle=\sigma\left(\sum_{e_{k}\in\mathcal{N}_{\mathcal{E}}(v_{i})}\frac{\texttt{Att}_{(\mathcal{V})}^{(\ell)}(\boldsymbol{q}^{(\ell-1)}_{k},\boldsymbol{p}^{(\ell-1)}_{i})\boldsymbol{q}^{(\ell-1)}_{k}\mathbf{\Theta}^{(\ell,1)}}{\sum_{e_{s}\in\mathcal{N}_{\mathcal{E}}(v_{i})}\texttt{Att}_{(\mathcal{V})}^{(\ell)}(\boldsymbol{q}^{(\ell-1)}_{s},\boldsymbol{p}^{(\ell-1)}_{i})}\right), | 
 | 
 
 
 (17) | 
 | 
 𝒒 j ( ℓ ) \displaystyle\boldsymbol{q}_{j}^{(\ell)} | 
 = σ ⁡ ( ∑ v k ∈ e j Att ( ℰ ) ( ℓ ) ​ ( 𝒑 k ( ℓ ) , 𝒒 j ( ℓ − 1 ) ) ​ 𝒑 k ( ℓ ) ​ 𝚯 ( ℓ , 2 ) ∑ v s ∈ e j Att ( ℰ ) ( ℓ ) ​ ( 𝒑 s ( ℓ ) , 𝒒 j ( ℓ − 1 ) ) ) . \displaystyle=\sigma\left(\sum_{v_{k}\in e_{j}}\frac{\texttt{Att}_{(\mathcal{E})}^{(\ell)}(\boldsymbol{p}^{(\ell)}_{k},\boldsymbol{q}^{(\ell-1)}_{j})\boldsymbol{p}^{(\ell)}_{k}\mathbf{\Theta}^{(\ell,2)}}{\sum_{v_{s}\in e_{j}}\texttt{Att}_{(\mathcal{E})}^{(\ell)}(\boldsymbol{p}^{(\ell)}_{s},\boldsymbol{q}^{(\ell-1)}_{j})}\right). | 
 | 
 

 Here, Att ( 𝒱 ) ( ℓ ) ​ ( 𝒒 , 𝒑 ) = σ ⁡ ( 𝒒 T ​ 𝝍 1 ( ℓ ) × 𝒑 T ​ 𝝍 2 ( ℓ ) ) ∈ ℝ \texttt{Att}^{(\ell)}_{(\mathcal{V})}(\boldsymbol{q},\boldsymbol{p})=\sigma(\boldsymbol{q}^{T}\boldsymbol{\psi}^{(\ell)}_{1}\times\boldsymbol{p}^{T}\boldsymbol{\psi}^{(\ell)}_{2})\in\mathbb{R} and Att ( ℰ ) ( ℓ ) ​ ( 𝒑 , 𝒒 ) = σ ⁡ ( 𝒑 T ​ 𝝍 3 ( ℓ ) × 𝒒 T ​ 𝝍 4 ( ℓ ) ) ∈ ℝ \texttt{Att}^{(\ell)}_{(\mathcal{E})}(\boldsymbol{p},\boldsymbol{q})=\sigma(\boldsymbol{p}^{T}\boldsymbol{\psi}^{(\ell)}_{3}\times\boldsymbol{q}^{T}\boldsymbol{\psi}^{(\ell)}_{4})\in\mathbb{R} are attention weight functions, where { 𝝍 1 ( ℓ ) , 𝝍 2 ( ℓ ) , 𝝍 3 ( ℓ ) , 𝝍 4 ( ℓ ) } \{\boldsymbol{\psi}^{(\ell)}_{1},\boldsymbol{\psi}^{(\ell)}_{2},\boldsymbol{\psi}^{(\ell)}_{3},\boldsymbol{\psi}^{(\ell)}_{4}\} and { 𝚯 ( ℓ , 1 ) , 𝚯 ( ℓ , 2 ) } \{\mathbf{\Theta}^{(\ell,1)},\mathbf{\Theta}^{(\ell,2)}\} are sets of learnable vectors and matrices, respectively.
Note that the attention weight functions consider messages from both sources and targets.
Target-aware attention has also been incorporated into clique-expansion-based HNNs, with HCHA  ( Bai et al., 2021b ) as a notable example.

 
 
 Table 2. Summary of hypergraph neural networks (HNNs). 
 
 
 
 Name | 
 Year | 
 Venue | 
 
 
 
 (Structure) | 

 
 Reductive? | 

 | 
 
 
 
 (Embedding Type) | 

 
 Edge Dependent? | 

 | 
 
 
 
 (Aggregation) | 

 
 Learnable? | 

 | 

 
 
 
 Yes 
 | 
 
 
 No 
 | 
 
 
 Yes 
 | 
 
 
 No 
 | 
 
 
 Yes 
 | 
 
 
 No 
 | 

 
 HGNN  ( Feng et al., 2019 ) | 
 2019 | 
 AAAI | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HyperGCN  ( Yadati et al., 2019 ) | 
 2019 | 
 NeurIPS | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 HNHN  ( Dong et al., 2020 ) | 
 2020 | 
 ICML | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HCHA  ( Bai et al., 2021b ) | 
 2019 | 
 Pat. Rec. | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 UniGNN  ( Huang and Yang, 2021 ) | 
 2021 | 
 IJCAI | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 HO Transformer  ( Kim et al., 2021 ) | 
 2021 | 
 NeurIPS | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 AllSet  ( Chien et al., 2022 ) | 
 2022 | 
 ICLR | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 HyperND  ( Prokopchik et al., 2022 ) | 
 2022 | 
 ICML | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 H-GNN  ( Zhang et al., 2022c ) | 
 2022 | 
 ICML | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 EHNN  ( Kim et al., 2022 ) | 
 2022 | 
 ECCV | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 LE GCN   ( Yang et al., 2022a ) | 
 2022 | 
 CIKM | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HERALD  ( Zhang et al., 2022c ) | 
 2022 | 
 ICASSP | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 HGNN+  ( Gao et al., 2022 ) | 
 2022 | 
 TPAMI | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 ED-HNN  ( Wang et al., 2023c ) | 
 2023 | 
 ICLR | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 PhenomNN  ( Wang et al., 2023a ) | 
 2023 | 
 ICML | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 WHATsNet  ( Choe et al., 2023 ) | 
 2023 | 
 KDD | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 

 
 SheafHyperGNN  ( Duta et al., 2023 ) | 
 2023 | 
 NeurIPS | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 MeanPooling  ( Lee and Shin, 2023a ) | 
 2023 | 
 AAAI | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HENN  ( Hayhoe et al., 2023 ) | 
 2023 | 
 LoG | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HyGNN  ( Saifuddin et al., 2023b ) | 
 2023 | 
 ICDE | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 HGraphormer  ( Qu et al., 2023 ) | 
 2023 | 
 arXiv | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 MultiSetMixer  ( Telyatnikov et al., 2023 ) | 
 2023 | 
 arXiv | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 

 
 HJRL  ( Yan et al., 2024a ) | 
 2024 | 
 AAAI | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HDE ode   ( Yan et al., 2024b ) | 
 2024 | 
 ICLR | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HyperGT  ( Liu et al., 2024 ) | 
 2024 | 
 ICASSP | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 THNN  ( Wang et al., 2024d ) | 
 2024 | 
 SDM | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 UniG-Encoder  ( Zou et al., 2024 ) | 
 2024 | 
 Pat. Rec. | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 SHNN  ( Tang et al., 2024 ) | 
 2024 | 
 arXiv | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 

 
 HyperMagNet  ( Benko et al., 2024 ) | 
 2024 | 
 arXiv | 
 
 
 ✔ 
 | 
 | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 

 
 CoNHD  ( Zheng and Worring, 2024 ) | 
 2024 | 
 arXiv | 
 | 
 
 
 ✔ 
 | 
 
 
 ✔ 
 | 
 | 
 
 
 ✔ 
 | 
 | 

 
 
 
 

#### 3.3.4. Comparison with GNNs 

 
 GNNs also use neural message passing to aggregate information from other nodes  ( Gilmer et al., 2017 ; Lee et al., 2023 ; Liang et al., 2024 ) .
However, since GNNs typically perform message passing directly between nodes,
they are not ideal for learning hyperedge (i.e., HOI) representations or hyperedge-dependent node representations.

 
 
 
 
 

## 4. Objective Design Guidance

 
 In this section, we outline training objectives for HNNs to capture HOIs effectively, particularly when label supervision is weak or absent.
Below, we review three branches: ( i ) learning to classify, ( ii ) learning to contrast, and ( iii ) learning to generate.

 
 

### 4.1. Learning to classify

 
 HNNs can learn HOIs by classifying hyperedges  ( Yadati et al., 2020 ; Hwang et al., 2022 ; Zhang et al., 2020 ; Wan et al., 2021 ; Ko et al., 2023 ) as positive or negative.
A positive hyperedge is a ground-truth, “true” hyperedge,
and a negative hyperedge often refers to a heuristically generated “fake” hyperedge, considered unlikely to exist.
By learning to classify them, HNNs may capture the distinguishing patterns of the ground-truth HOIs.

 
 

#### 4.1.1. Heuristic negative sampling. 

 
 We discuss popular negative sampling (NS) strategies to obtain negative hyperedges  ( Patil et al., 2020 ) :

 
 • 
 
 Sized NS : each negative hyperedge contains k k random nodes.

 

 • 
 
 Motif NS : each negative hyperedge contains a randomly chosen k k adjacent nodes.

 

 • 
 
 Clique NS : each negative hyperedge is generated by replacing a randomly chosen node in a positive hyperedge with another randomly chosen node adjacent to the remaining nodes.

 

 
 Similarly, many HNNs use rule-based NS for hyperedge classification  ( Yadati et al., 2020 ; Hwang et al., 2022 ; Zhang et al., 2020 ; Wan et al., 2021 ; Ko et al., 2023 ) .
Others leverage domain knowledge to design NS strategies  ( Chen et al., 2023 ; Wang et al., 2024b ) .

 
 
 

#### 4.1.2. Learnable negative sampling. 

 
 Notably, Hwang et al. (2022) show that training HNNs with the aforementioned NS strategies may cause overfitting to negative hyperedges of specific types.
This may be attributable to the vast population of potential negative hyperedges, where the tiny samples may not adequately represent this population.
To mitigate the problem, they employ adversarial training of a generator that samples negative hyperedges.

 
 
 

#### 4.1.3. Comparison with GNNs 

 
 Link prediction on pairwise graphs is a counterpart of the HOI classification task  ( Zhu et al., 2021 ; Zhang and Chen, 2018 ) .
However, the space of possible negative edges significantly differs between them.
In pairwise graphs, the size of the space is O ⁡ ( | 𝒱 | 2 ) O(|\mathcal{V}|^{2}) .
However, in hypergraphs, since a hyperedge can contain an arbitrary number of nodes, the size of the space is O ⁡ ( 2 | 𝒱 | ) O(2^{|\mathcal{V}|}) , which makes finding representative “unlikely” HOIs, or negative hyperedges, more challenging  ( Hwang et al., 2022 ) .
Consequently, learning the distinguishing patterns of HOIs by classifying the positive and negative hyperedges may be more challenging.

 
 
 
 

### 4.2. Learning to contrast

 
 Contrastive learning (CL) aims to maximize agreement between data obtained from different views.
Intuitively, views refer to different versions of the same data, original or augmented.
Training neural networks with CL has shown strong capacity in capturing the input data characteristics  ( Jaiswal et al., 2020 ) .
For HNNs, several CL techniques have been devised to learn HOIs  ( Lee and Shin, 2023a ; Wei et al., 2022 ; Kim et al., 2023c ) .
Here, we describe three steps of CL for HNNs: ( i ) obtaining views, ( ii ) encoding, and ( iii ) computing contrastive loss.

 
 

#### 4.2.1. View creation and encoding. 

 
 First, we obtain views for contrast.
This can be achieved by augmenting the input hypergraph, using rule-based   ( Lee and Shin, 2023a ; Kim et al., 2023c ) or learnable   ( Wei et al., 2022 ) methods.

 
 
 Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. Rule-based augmentation. This approach stochastically corrupts node features and hyperedges.
For nodes, an augmented feature matrix is obtained by either zeroing out certain entries (i.e., feature values) of 𝐗 \mathbf{X}   ( Lee and Shin, 2023a ; Ko et al., 2023 ) or adding Gaussian noise to them  ( Qian et al., 2024 ) .
For hyperedges, augmented hyperedges are obtained by excluding some nodes from hyperedges  ( Lee and Shin, 2023a ) or perturbing hyperedge membership (e.g., changing e i = { v 1 , v 2 , v 3 } e_{i}=\{v_{1},v_{2},v_{3}\} to e i ′ = { v 1 , v 2 , v 4 } e^{\prime}_{i}=\{v_{1},v_{2},v_{4}\} )  ( Liu et al., 2023 ) .

 
 
 Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. Learnable augmentation. This approach utilizes a neural network to generate views  ( Wei et al., 2022 ) .
Specifically, HyperGCL  ( Wei et al., 2022 ) generates synthetic hyperedges ℰ ′ \mathcal{E}^{\prime} using HNN-based VAE  ( Kingma and Welling, 2013 ) .

 
 
 Once an augmentation strategy τ : ( 𝐗 , ℰ ) ↦ ( 𝐗 ′ , ℰ ′ ) \tau:(\mathbf{X},\mathcal{E})\mapsto(\mathbf{X}^{\prime},\mathcal{E}^{\prime}) is decided, a hypergraph-view pair ( 𝒢 ( 1 ) , 𝒢 ( 2 ) ) (\mathcal{G}^{(1)},\mathcal{G}^{(2)}) can be obtained in two ways:

 
 • 
 
 𝒢 ( 1 ) \mathcal{G}^{(1)} is the original hypergraph with ( 𝐗 , ℰ ) (\mathbf{X},\mathcal{E}) , and 𝒢 ( 2 ) \mathcal{G}^{(2)} is an augmented hypergraph with ( 𝐗 ′ , ℰ ′ ) (\mathbf{X}^{\prime},\mathcal{E}^{\prime}) , where ( 𝐗 ′ , ℰ ′ ) = τ ⁡ ( 𝐗 , ℰ ) (\mathbf{X}^{\prime},\mathcal{E}^{\prime})=\tau(\mathbf{X},\mathcal{E})   ( Wei et al., 2022 ) .

 

 • 
 
 Both 𝒢 ( 1 ) \mathcal{G}^{(1)} and 𝒢 ( 2 ) \mathcal{G}^{(2)} are augmented by applying
 τ \tau to ( 𝐗 , ℰ ) (\mathbf{X},\mathcal{E})   ( Lee and Shin, 2023a ) . They likely differ due to the stochastic nature of τ \tau .

 

 
 
 
 Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. Encoding. 
Then, the message passing on the two views (sharing the same parameters) results in two pairs of node and hyperedge embeddings denoted by ( 𝐏 ′ , 𝐐 ′ ) (\mathbf{P}^{\prime},\mathbf{Q}^{\prime}) and ( 𝐏 ′′ , 𝐐 ′′ ) (\mathbf{P}^{\prime\prime},\mathbf{Q}^{\prime\prime})   ( Lee and Shin, 2023a ; Ko et al., 2023 ) .

 
 
 

#### 4.2.2. Contrastive loss. 

 
 Then, we choose a contrastive loss.
Below, we present node -, hyperedge -, and membership -level contrastive losses.
Here, τ x , τ e , τ m ∈ ℝ \tau_{x},\tau_{e},\tau_{m}\in\mathbb{R} are hyperparameters.

 
 
 Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. Node level. A node-level contrastive loss is used to ( i ) maximize the similarity between the same node from two different views and ( ii ) minimize the similarity for different nodes  ( Lee and Shin, 2023a ; Kim et al., 2023c ; Ko et al., 2023 ; Wei et al., 2022 ; Ma et al., 2023 ) :

 

 
 (18) | 
 | 
 ℒ ( v ) ​ ( 𝐏 ′ , 𝐏 ′′ ) = − 1 | 𝒱 | ​ ∑ v i ∈ 𝒱 log ⁡ exp ⁡ ( sim ​ ( 𝒑 i ′ , 𝒑 i ′′ ) / τ v ) ∑ v k ∈ 𝒱 exp ⁡ ( sim ​ ( 𝒑 i ′ , 𝒑 k ′′ ) / τ v ) , \mathcal{L}^{(v)}(\mathbf{P}^{\prime},\mathbf{P}^{\prime\prime})=\frac{-1}{|\mathcal{V}|}\sum_{v_{i}\in\mathcal{V}}\log{\frac{\exp(\texttt{sim}(\boldsymbol{p}^{\prime}_{i},\boldsymbol{p}^{\prime\prime}_{i})/\tau_{v})}{\sum_{v_{k}\in\mathcal{V}}\exp(\texttt{sim}(\boldsymbol{p}^{\prime}_{i},\boldsymbol{p}^{\prime\prime}_{k})/\tau_{v})}}, | 
 | 
 

 where sim ​ ( 𝒙 , 𝒚 ) \texttt{sim}(\boldsymbol{x},\boldsymbol{y}) is the similarity between 𝒙 \boldsymbol{x} and 𝒚 \boldsymbol{y} (e.g., cosine similarity).

 
 
 Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. Hyperedge level. A
hyperedge-level contrastive loss is implemented in a similar manner  ( Lee and Shin, 2023a ; Ko et al., 2023 ; Li et al., 2024 ) :

 
 
 

 
 (19) | 
 | 
 ℒ ( e ) ​ ( 𝐐 ′ , 𝐐 ′′ ) = − 1 | ℰ | ​ ∑ e j ∈ ℰ log ⁡ exp ⁡ ( sim ​ ( 𝒒 j ′ , 𝒒 j ′′ ) / τ e ) ∑ e k ∈ ℰ exp ⁡ ( sim ​ ( 𝒒 i ′ , 𝒒 k ′′ ) / τ e ) . \mathcal{L}^{(e)}(\mathbf{Q}^{\prime},\mathbf{Q}^{\prime\prime})=\frac{-1}{|\mathcal{E}|}\sum_{e_{j}\in\mathcal{E}}\log{\frac{\exp(\texttt{sim}(\boldsymbol{q}^{\prime}_{j},\boldsymbol{q}^{\prime\prime}_{j})/\tau_{e})}{\sum_{e_{k}\in\mathcal{E}}\exp(\texttt{sim}(\boldsymbol{q}^{\prime}_{i},\boldsymbol{q}^{\prime\prime}_{k})/\tau_{e})}}. | 
 | 
 

 
 
 Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. Membership level. 
A membership-level contrastive loss is used to make the embeddings of incident node-hyperedge pairs distinguishable from those of non-incident pairs across two views  ( Lee and Shin, 2023a ) :

 

 
 | 
 ℒ ( m ) ​ ( 𝐏 ′ , 𝐐 ′′ ) \displaystyle\mathcal{L}^{(m)}(\mathbf{P}^{\prime},\mathbf{Q}^{\prime\prime}) | 
 = − 1 K ​ ∑ e j ∈ ℰ ∑ v i ∈ 𝒱 𝟙 i , j ​ log ⁡ exp ⁡ ( 𝒟 ⁡ ( 𝒑 i ′ , 𝒒 j ′′ ) / τ m ) ∑ v k ∈ 𝒱 exp ⁡ ( 𝒟 ⁡ ( 𝒑 k ′ , 𝒒 j ′′ ) / τ m ) ⏟ when  ​ 𝒒 j ′′ ​ is an anchor \displaystyle=\frac{-1}{K}\sum_{e_{j}\in\mathcal{E}}\sum_{v_{i}\in\mathcal{V}}\underbrace{\mathbb{1}_{i,j}\log{\frac{\exp(\mathcal{D}(\boldsymbol{p}^{\prime}_{i},\boldsymbol{q}^{\prime\prime}_{j})/\tau_{m})}{\sum_{v_{k}\in\mathcal{V}}\exp(\mathcal{D}(\boldsymbol{p}^{\prime}_{k},\boldsymbol{q}^{\prime\prime}_{j})/\tau_{m})}}}_{\texttt{when }\boldsymbol{q}^{\prime\prime}_{j}~\texttt{is an anchor}} | 
 | 
 
 
 | 
 | 
 − 1 K ∑ e j ∈ ℰ ∑ v i ∈ 𝒱 𝟙 i , j ​ log ⁡ exp ⁡ ( 𝒟 ⁡ ( 𝒒 j ′′ , 𝒑 i ′ ) / τ m ) ∑ e k ∈ ℰ exp ⁡ ( 𝒟 ⁡ ( 𝒒 k ′′ , 𝒑 i ′ ) / τ m ) , ⏟ when  ​ 𝒑 i ′ ​ is an anchor \displaystyle-\frac{1}{K}\sum_{e_{j}\in\mathcal{E}}\sum_{v_{i}\in\mathcal{V}}\underbrace{\mathbb{1}_{i,j}\log{\frac{\exp(\mathcal{D}(\boldsymbol{q}^{\prime\prime}_{j},\boldsymbol{p}^{\prime}_{i})/\tau_{m})}{\sum_{e_{k}\in\mathcal{E}}\exp(\mathcal{D}(\boldsymbol{q}^{\prime\prime}_{k},\boldsymbol{p}^{\prime}_{i})/\tau_{m}),}}}_{\texttt{when }\boldsymbol{p}^{\prime}_{i}~\texttt{is an anchor}} | 
 | 
 

 where 𝟙 s , j = 𝕀 [ v s ∈ v j ] \mathbb{1}_{s,j}=\mathbb{I}[v_{s}\in v_{j}] ; 𝒟 ⁡ ( 𝒙 , 𝒚 ) ∈ ℝ \mathcal{D}(\boldsymbol{x},\boldsymbol{y})\in\mathbb{R} is a discriminator for assigning higher value to incident pairs than non-incident pairs  ( Veličković et al., 2019 ) .

 
 
 

#### 4.2.3. Comparison with GNNs 

 
 GNNs are also commonly trained with contrastive objectives  ( Veličković et al., 2019 ; Qiu et al., 2020 ; You et al., 2020 ) .
They typically focus on node-level  ( Veličković et al., 2019 ) and/or graph-level contrast  ( Qiu et al., 2020 ) .

 
 
 
 

### 4.3. Learning to generate

 
 HNNs can also be trained by learning to generate hyperedges.
Existing HNNs aim to generate either ( i ) ground-truth hyperedges to capture their characteristics or ( ii ) latent hyperedges potentially beneficial for designated downstream tasks.

 
 

#### 4.3.1. Generating ground-truth HOIs. 

 
 Training neural networks to generate input data has shown strong efficacy in various domains and downstream tasks  ( OpenAI, 2023 ; He et al., 2022 ) .
In two recent studies, HNNs are trained to generate ground-truth hyperedges to learn HOIs  ( Kim et al., 2024 ; Du et al., 2022 ) .
HypeBoy by  Kim et al. (2024) formulates hyperedge generation as a hyperedge filling task , where the objective is to identify the missing node for a given subset of a hyperedge. Overall, HypeBoy involves three steps: ( i ) hypergraph augmentation, ( ii ) node and hyperedge-subset encoding, and ( iii ) loss-function computation.

 
 
 HypeBoy obtains the augmented node feature matrix 𝐗 ′ \mathbf{X}^{\prime} and augmented input topology ℰ ′ \mathcal{E}^{\prime} , respectively by randomly masking some entries of 𝐗 \mathbf{X} and by randomly dropping some hyperedges from ℰ \mathcal{E} .
Hypeboy, then, feeds 𝐗 ′ \mathbf{X}^{\prime} and ℰ ′ \mathcal{E}^{\prime} into an HNN to obtain node embedding matrix 𝐏 \mathbf{P} .
Subsequently, for each node v i ∈ e j v_{i}\in e_{j} and subset q i ​ j = e j ∖ { v i } q_{ij}=e_{j}\setminus\{v_{i}\} , HypeBoy obtains (final) node embedding 𝓹 i = MLP 1 ​ ( 𝒑 i ) \boldsymbol{\mathscr{p}}_{i}=\texttt{MLP}_{1}(\boldsymbol{p}_{i}) 
and subset embedding 𝓺 i ​ j = MLP 2 ​ ( ∑ v k ∈ q i ​ j 𝒑 k ) \boldsymbol{\mathscr{q}}_{ij}=\texttt{MLP}_{2}(\sum_{v_{k}\in q_{ij}}\boldsymbol{p}_{k}) .
Lastly, the HNN is trained to make embeddings of the ‘true’ node-subset pairs similar and of the ‘false’ node-subset pairs dissimilar.
Specifically, it minimizes the following loss:

 

 
 (20) | 
 | 
 ℒ = − ∑ e j ∈ ℰ ∑ v i ∈ e j log exp ⁡ ( sim ​ ( 𝓹 i , 𝓺 i ​ j ) ) ∑ v k ∈ 𝒱 exp ⁡ ( sim ​ ( 𝓹 k , 𝓺 i ​ j ) ) , \mathcal{L}=-\sum_{e_{j}\in\mathcal{E}}\sum_{v_{i}\in e_{j}}\log{\frac{\exp(\texttt{sim}(\boldsymbol{\mathscr{p}}_{i},\boldsymbol{\mathscr{q}}_{ij}))}{\sum_{v_{k}\in\mathcal{V}}\exp(\texttt{sim}(\boldsymbol{\mathscr{p}}_{k},\boldsymbol{\mathscr{q}}_{ij}))}}, | 
 | 
 

 where sim ​ ( 𝒙 , 𝒚 ) \texttt{sim}(\boldsymbol{x},\boldsymbol{y}) is a cosine similarity between 𝒙 \boldsymbol{x} and 𝒚 \boldsymbol{y} .

 
 
 

#### 4.3.2. Generating latent HOIs. 

 
 HNNs can be trained to generate latent hyperedges, especially when ( i ) (semi-)supervised downstream tasks
and ( ii ) suboptimal input hypergraph structures are assumed.
Typically, the training methods let HNNs generate potential, latent hyperedges, which are used for message passing to improve downstream task performance  ( Zhang et al., 2022a ; Zhang et al., 2022b ; Cai et al., 2022a ; Lei et al., 2024 ) .

 
 
 For example, HSL  ( Cai et al., 2022a ) adopts a learnable augmenter to replace unhelpful hyperedges with the generated ones.
HSL prunes hyperedges using a masking matrix 𝐌 ∈ ℝ | 𝒱 | × | ℰ | \mathbf{M}\in\mathbb{R}^{|\mathcal{V}|\times|\mathcal{E}|} .
Each j − j- th column is m j = sigmoid ​ ( ( log ⁡ ( z j 1 − z j ) + ( ϵ 0 − ϵ 1 ) ) / τ ) m_{j}=\texttt{sigmoid}((\log{(\frac{z_{j}}{1-z_{j}})}+(\epsilon_{0}-\epsilon_{1}))/\tau) , where ϵ 0 \epsilon_{0} and ϵ 1 \epsilon_{1} , τ ∈ ℝ \tau\in\mathbb{R} , and z k ∈ [ 0 , 1 ] , ∀ e k ∈ ℰ z_{k}\in[0,1],\forall e_{k}\in\mathcal{E} respectively are random samples from Gumbel(0, 1) , a hyperparameter, and a learnable scalar.
An unhelpful e k e_{k} is expected to have small z k z_{k} to be pruned.

 
 
 After performing pruning by 𝐇 ^ = 𝐇 ⊙ 𝐌 \mathbf{\hat{H}}=\mathbf{H}\odot\mathbf{M} , HSL modifies 𝐇 ^ \mathbf{\hat{H}} by adding generated latent hyperedges 𝚫 ​ 𝐇 \mathbf{\Delta H} .
Specifically, 𝚫 ​ 𝐇 i , j = 1 \mathbf{\Delta H}_{i,j}=1 if ( 𝐇 i , j = 0 ) ∧ ( 𝐒 i , j ∈ top ​ ( 𝐒 , N ) ) (\mathbf{H}_{i,j}=0)\wedge(\mathbf{S}_{i,j}\in\texttt{top}(\mathbf{S},N)) , and 0 otherwise.
 top ​ ( 𝐒 , N ) \texttt{top}(\mathbf{S},N) denotes the set of top-N entries in a learnable score matrix 𝐒 ∈ ℝ | 𝒱 | × | ℰ | \mathbf{S}\in\mathbb{R}^{|\mathcal{V}|\times|\mathcal{E}|} .
Each score in 𝐒 \mathbf{S} is 𝐒 i , j = 1 T ​ ∑ t = 1 T sim ​ ( 𝒘 t ⊙ 𝒑 i , 𝒘 t ⊙ 𝒒 i ) \mathbf{S}_{i,j}=\frac{1}{T}\sum_{t=1}^{T}\texttt{sim}(\boldsymbol{w}_{t}\odot\boldsymbol{p}_{i},\boldsymbol{w}_{t}\odot\boldsymbol{q}_{i}) , where { 𝒘 t } t = 1 T \{\boldsymbol{w}_{t}\}^{T}_{t=1} and sim respectively are learnable vectors and cosine similarity.
To summarize, node and hyperedge similarities learned by an HNN serve to generate latent hyperedges 𝚫 ​ 𝐇 \mathbf{\Delta H} .
Lastly, 𝐇 ^ + 𝚫 ​ 𝐇 \mathbf{\hat{H}}+\mathbf{\Delta H} is fed into another HNN for a target downstream task (e.g., node classification).
All learnable components, including the HNN for augmentation, are trained end-to-end.

 
 
 Note that the HNNs learning to generate latent hyperedges generally implement additional loss functions to encourage the latent hyperedges to be similar to the original ones  ( Zhang et al., 2022a ; Zhang et al., 2022b ) .
Furthermore, some studies have explored generating latent HOIs when input hypergraph structures were not available  ( Jiang et al., 2019 ; Zhou et al., 2023b ; Gao et al., 2020 ) .

 
 
 

#### 4.3.3. Comparison with GNNs 

 
 Various GNNs also target to generate ground-truth pairwise interactions  ( Kipf and Welling, 2016 ; Tan et al., 2023 ) or latent pairwise interactions  ( Fatemi et al., 2021 ) .
In a pairwise graph, the inner product of two node embeddings is widely used to model the likelihood that an edge joins these nodes  ( Kipf and Welling, 2016 ; Fatemi et al., 2021 ) .
However, modeling the likelihood of a hyperedge, which can join any number of nodes, using an inner product is not straightforward.

 
 
 
 
 

## 5. Application Guidance

 
 HNNs have been adopted in various applications, including recommendation, bioinformatics and medical science, time series analysis, and computer vision.
Their central concerns involve hypergraph construction and hypergraph learning task formulation.

 
 

### 5.1. Recommendation

 

#### 5.1.1. Hypergraph construction. 

 
 For recommender system applications, many studies utilized hypergraphs consisting of item nodes (being recommended) and user hyperedges (receiving recommendations).
For instance, all items that a user interacted with were connected by a hyperedge  ( Wang et al., 2020 ) .
When sessions were available, hyperedges connected item nodes by their context window  ( Wang et al., 2021 ; Xia et al., 2021 ; Li et al., 2022a ) .
Some studies leveraged multiple hypergraphs.
For instance,   Zhang et al. (2021) incorporated user- and group-level hypergraphs.
 Ji et al. (2020) constructed a hypergraph with item nodes and a hypergraph with user nodes, where their hyperedges were inferred from heuristic-based algorithms.
In contrast, other studies incorporated learnable hypergraph structure  ( Xia et al., 2022b ; Xia et al., 2022a ) .

 
 
 

#### 5.1.2. Application tasks. 

 
 Hypergraph-based modeling allows natural applications of HNNs for recommendation, typically formulated as a hyperedge prediction problem.
HNNs have been used for
sequential  ( Li et al., 2021 ; Wang et al., 2020 ) ,
session-based ( Wang et al., 2021 ; Li et al., 2022a ; Xia et al., 2021 ) ,
group  ( Zhang et al., 2021 ; Jia et al., 2021 ) ,
conversational  ( Zhao et al., 2023b ) ,
and point-of-interest  ( Lai et al., 2023 ) recommendation.

 
 
 
 

### 5.2. Bioinformatics and medical science

 

#### 5.2.1. Hypergraph construction. 

 
 For bioinformatics applications, molecular-level structures have often been regarded as nodes.
Studies used hyperedges to connect the structures based on their
joint reaction  ( Chen et al., 2023 ) ,
presence within each drug  ( Saifuddin et al., 2023b ; Hu et al., 2024 ) ,
and association with each disease  ( Hu et al., 2024 ) .
Some studies used multiple node types.
A study considered cell line nodes and drug nodes,
with hyperedge connecting those with a synergy relationship  ( Liu et al., 2022 ; Wang et al., 2024b ) .
Drugs and their side effects were also considered as nodes, where a hyperedge connected those with drug-drug interaction  ( Nguyen et al., 2022 ) .
Drug nodes or target protein nodes were also connected by hyperedges based on their similarity in interactions or associations  ( Ruan et al., 2021 ) .
Studies also used kNN or learnable hyperedges to build hypergraphs  ( Peng et al., 2024 ; Li et al., 2023 ) .

 
 
 Some other studies used hypergraphs to model MRI data.
Many of them had a region-of-interest serving as a node, while a hyperedge connected the nodes using interaction strength estimation  ( Wang et al., 2023b ) , k-means  ( Ji et al., 2022 ) , or random-walk-based sampling  ( Cai et al., 2023 ) .
On the other hand, in some studies, study subjects were nodes, and hyperedges connected the neighbors found by kNN  ( Hao et al., 2024 ; Madine et al., 2020 ) .

 
 
 Lastly, electronic health records (EHR) data were often modeled with hypergraphs.
Most often, nodes were either medical codes  ( Cai et al., 2022b ; Wu et al., 2023a ; Xu et al., 2022a ; Xu et al., 2023 ; Cui et al., 2024 ) or clinical events  ( Zhu et al., 2022 ) .
A hyperedge connected the codes or clinical events that were shared by each patient.

 
 
 

#### 5.2.2. Application tasks. 

 
 For bioinformatics applications, HNNs have been applied to predict interactions or associations among molecular-level structures.
Thus, many of the tasks could be naturally formulated as a hyperedge prediction task.
Specifically, the application tasks include predictions of
missing metabolic reactions  ( Yadati et al., 2020 ; Chen et al., 2023 ) ,
drug-drug interactions  ( Liu et al., 2022 ; Wang et al., 2024b ; Saifuddin et al., 2023b ; Nguyen et al., 2022 ) ,
drug-target interactions  ( Ruan et al., 2021 ) ,
drug-gene interactions  ( Tao et al., 2023 ) ,
herb–disease associations  ( Hu et al., 2024 ) ,
and miRNA-disease associations  ( Peng et al., 2024 ) .

 
 
 For MRI analysis, when a region-of-interest served as a node, HNNs have been applied to solve a hypergraph classification problem.
Alzheimer’s disease classification  ( Hao et al., 2024 ) ,
brain connectome analysis  ( Wang et al., 2023b ) ,
autism prediction  ( Madine et al., 2020 ; Ji et al., 2022 ) ,
and brain network dysfunction prediction  ( Cai et al., 2023 ) problems
have been solved with HNNs.

 
 
 In analyzing EHR data, since a hyperedge consisted of medical codes or clinical events of a patient,
HNNs have been applied for hyperedge prediction.
Studies used HNNs to predict
mortality  ( Cai et al., 2022b ) ,
readmission  ( Cai et al., 2022b ) ,
diagnosis  ( Wu et al., 2023a ) ,
medication  ( Wu et al., 2023a ) ,
phenotype  ( Xu et al., 2022a ; Cui et al., 2024 ) ,
clinical outcomes  ( Xu et al., 2022a ; Xu et al., 2023 ; Cui et al., 2024 ) ,
and clinical pathways  ( Zhu et al., 2022 ) .

 
 
 
 

### 5.3. Time series analysis

 

#### 5.3.1. Hypergraph construction. 

 
 A variety of nodes have been used for time series forecast applications.
Depending on the data, nodes were
cities  ( Yi and Park, 2020 ; Wang et al., 2024c ) ,
gas regulators  ( Yi and Park, 2020 ) ,
rail segments  ( Yi and Park, 2020 ) ,
train stations  ( Wang et al., 2024c ) ,
stocks  ( Liao et al., 2024 ; Sawhney et al., 2021 ) ,
or regions  ( Li et al., 2022b ) .
Studies often leveraged similarity- or proximity-based hyperedges  ( Yi and Park, 2020 ; Liao et al., 2024 ; Sawhney et al., 2021 ) or learnable hyperedges  ( Wang et al., 2024c ; Liao et al., 2024 ; Li et al., 2022b ) .

 
 
 

#### 5.3.2. Application tasks. 

 
 When applying HNNs, many time series forecast problems can be formulated as node regression problems.
Specifically, the prior works used HNNs to forecast
taxi demands  ( Yi and Park, 2020 ) ,
gas pressures  ( Yi and Park, 2020 ) ,
vehicle speeds  ( Yi and Park, 2020 ) ,
traffic  ( Luo et al., 2022 ; Shang and Chen, 2024 ; Wang et al., 2024c ; Wu et al., 2023b ; Zhao et al., 2023a ) , electricity consumptions  ( Shang and Chen, 2024 ; Wu et al., 2023b ) ,
meteorological measures  ( Shang and Chen, 2024 ; Wang et al., 2024c ) ,
stocks  ( Liao et al., 2024 ; Sawhney et al., 2021 ) ,
and crimes  ( Li et al., 2022b ) .

 
 
 
 

### 5.4. Computer vision

 

#### 5.4.1. Hypergraph construction. 

 
 Hypergraph-based modeling has also been adopted for computer vision applications.
Studies used nodes to represent
image patches  ( Han et al., 2023 ) ,
features  ( Yan et al., 2020 ) ,
3D shapes  ( Bai et al., 2021a ) ,
joints  ( Zhou et al., 2022 ; Liu et al., 2020 ; Xu et al., 2022b ) ,
and humans  ( Huang et al., 2023 ) .
To connect the nodes by a hyperedge,
kNN  ( Yan et al., 2020 ; Bai et al., 2021a ) ,
Fuzzy C-Means  ( Han et al., 2023 ) ,
and other learnable functions  ( Wadhwa et al., 2021 ; Liu et al., 2020 ; Xu et al., 2022b ) were adopted.

 
 
 

#### 5.4.2. Application tasks. 

 
 For computer vision tasks,
studies used HNNs to solve problems including
image classification  ( Han et al., 2023 ) ,
object detection  ( Han et al., 2023 ) ,
video-based person re-identification  ( Yan et al., 2020 ) ,
image impainting  ( Wadhwa et al., 2021 ) ,
action recognition  ( Zhou et al., 2022 ) ,
pose estimation  ( Liu et al., 2020 ; Xu et al., 2022b ) ,
3D shape retrieval and recogntion  ( Bai et al., 2021a ) ,
and multi-human mesh recovery  ( Huang et al., 2023 ) .
Due to the heterogeneity of the applied tasks, we found no consistent hypergraph learning task formulation.

 
 
 
 
 

## 6. Discussions

 
 In this work, we provide a survey on hypergraph neural networks (HNNs), with a focus on how they address higher-order interactions (HOIs).
We aim for the present survey to be in-depth, covering HNN encoders (Sec.  3 ), training objectives (Sec.  4 ), and applications (Sec.  5 ).
Having reviewed the exponentially growing literature, we close the survey with some future directions.

 
 
 HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. HNN theory. 
Studies have theoretically investigated graph neural networks (GNNs) on their graph isomorphism recognition  ( Xu et al., 2019 ; Vignac et al., 2020 ) , approximation abilities  ( Keriven and Peyré, 2019 ; Maron et al., 2019 ) , and relation to homophily  ( Ma et al., 2022a ; Lee et al., 2024b ) .
However, given the complex nature of hypergraphs, directly applying these theoretical findings to hypergraphs can be non-trivial  ( Feng et al., 2024 ) .
Therefore, many theoretical properties of HNNs remain yet to be unveiled, and some areas have begun to be explored, including their generalization abilities  ( Zhezheng et al., 2023 ) and transferability  ( Hayhoe et al., 2023 ) .

 
 
 Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. Advantages of HNNs. 
Instead of leveraging HNNs, one could use GNNs for a hypergraph by reducing its structure to a pairwise one.
While studies have empirically shown that HNNs outperform these alternatives  ( Feng et al., 2019 ; Chien et al., 2022 ; Dong et al., 2020 ; Wang et al., 2023c ; Kim et al., 2024 ) , the factors that confer HNNs the advantages remain unclear.
While the advantages of using HOIs for a heuristic classifier have been investigated  ( Yoon et al., 2020 ) ,
studies dedicated to HNNs may inspire improved HNNs and their training strategies.

 
 
 Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. Complex hypergraphs. 
Networks of HOIs often exhibit temporal, directional, and heterogeneous properties, which are respectively modeled by
temporal  ( Lee and Shin, 2023b ) , directed  ( Gallo et al., 1993 ) , and heterogeneous  ( Huang et al., 2024 ; Yadati, 2020 ) hypergraphs.
Although their structural patterns have been studied  ( Lee and Shin, 2023b ; Kim et al., 2023b ; Moon et al., 2023 ; Benson et al., 2018 ) ,
developing HNNs to learn such complex HOIs is in the early stages  ( Tran and Tran, 2020 ; Luo et al., 2022 ; Agarwal et al., 2022 ; Huang et al., 2024 ; Yadati, 2020 ; Zhou et al., 2023a ) .
Thus, more benchmark datasets and tasks for complex hypergraphs are necessary.
The proper datasets and tasks will catalyze studies to develop HNNs that better exploit the complex nature of HOIs.

 
 
 

## Acknowledgements

 
 This work was partly supported by Institute of Information Communications Technology Planning Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00157, Robust, Fair, Extensible Data-Centric Continual Learning) (No. RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST)). This work has been partially supported by the spoke “FutureHPC BigData” of the ICSC – Centro Nazionale di Ricerca in High-Performance Computing, Big Data and Quantum Computing funded by European Union – NextGenerationEU.
 

 
 
 

## References

 
 
 Agarwal et al . (2022) 
 
Shivam Agarwal, Ramit Sawhney, Megh Thakkar, Preslav Nakov, Jiawei Han, and Tyler Derr. 2022.

 
 Think: Temporal hypergraph hyperbolic network. In ICDM .

 
 
 

 
 Aponte et al . (2022) 
 
Ryan Aponte, Ryan A Rossi, Shunan Guo, Jane Hoffswell, Nedim Lipka, Chang Xiao, Gromit Chan, Eunyee Koh, and Nesreen Ahmed. 2022.

 
 A hypergraph neural network framework for learning hyperedge-dependent node embeddings.

 
 arXiv preprint arXiv:2212.14077 (2022).

 
 
 

 
 Ba et al . (2016) 
 
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016.

 
 Layer normalization.

 
 arXiv preprint arXiv:1607.06450 (2016).

 
 
 

 
 Bai et al . (2021a) 
 
Junjie Bai, Biao Gong, Yining Zhao, Fuqiang Lei, Chenggang Yan, and Yue Gao. 2021a.

 
 Multi-scale representation learning on hypergraph for 3D shape retrieval and recognition.

 
 IEEE Transactions on Image Processing 30 (2021), 5327–5338.

 
 
 

 
 Bai et al . (2021b) 
 
Song Bai, Feihu Zhang, and Philip HS Torr. 2021b.

 
 Hypergraph convolution and hypergraph attention.

 
 Pattern Recognition 110 (2021), 107637.

 
 
 

 
 Battiston et al . (2021) 
 
Federico Battiston, Enrico Amico, Alain Barrat, Ginestra Bianconi, Guilherme Ferraz de Arruda, Benedetta Franceschiello, Iacopo Iacopini, Sonia Kéfi, Vito Latora, Yamir Moreno, et al . 2021.

 
 The physics of higher-order interactions in complex systems.

 
 Nature Physics 17, 10 (2021), 1093–1098.

 
 
 

 
 Battiston et al . (2020) 
 
Federico Battiston, Giulia Cencetti, Iacopo Iacopini, Vito Latora, Maxime Lucas, Alice Patania, Jean-Gabriel Young, and Giovanni Petri. 2020.

 
 Networks beyond pairwise interactions: Structure and dynamics.

 
 Physics Reports 874 (2020), 1–92.

 
 
 

 
 Battiston and Petri (2022) 
 
Federico Battiston and Giovanni Petri. 2022.

 
 Higher-order systems .

 
 Springer.

 
 
 

 
 Benko et al . (2024) 
 
Tatyana Benko, Martin Buck, Ilya Amburg, Stephen J Young, and Sinan G Aksoy. 2024.

 
 Hypermagnet: A magnetic laplacian based hypergraph neural network.

 
 arXiv preprint arXiv:2402.09676 (2024).

 
 
 

 
 Benson et al . (2018) 
 
Austin R Benson, Ravi Kumar, and Andrew Tomkins. 2018.

 
 Sequences of sets. In KDD .

 
 
 

 
 Bianconi (2021) 
 
Ginestra Bianconi. 2021.

 
 Higher-order networks .

 
 Cambridge University Press.

 
 
 

 
 Cai et al . (2022a) 
 
Derun Cai, Moxian Song, Chenxi Sun, Baofeng Zhang, Shenda Hong, and Hongyan Li. 2022a.

 
 Hypergraph structure learning for hypergraph neural networks. In IJCAI .

 
 
 

 
 Cai et al . (2022b) 
 
Derun Cai, Chenxi Sun, Moxian Song, Baofeng Zhang, Shenda Hong, and Hongyan Li. 2022b.

 
 Hypergraph contrastive learning for electronic health records. In SDM .

 
 
 

 
 Cai et al . (2023) 
 
Hongmin Cai, Zhixuan Zhou, Defu Yang, Guorong Wu, and Jiazhou Chen. 2023.

 
 Discovering Brain Network Dysfunction in Alzheimer’s Disease Using Brain Hypergraph Neural Network. In MICCAI .

 
 
 

 
 Chai et al . (2024) 
 
Lang Chai, Lilan Tu, Xianjia Wang, and Qingqing Su. 2024.

 
 Hypergraph modeling and hypergraph multi-view attention neural network for link prediction.

 
 Pattern Recognition 149 (2024), 110292.

 
 
 

 
 Chen et al . (2023) 
 
Can Chen, Chen Liao, and Yang-Yu Liu. 2023.

 
 Teasing out missing reactions in genome-scale metabolic networks through hypergraph learning.

 
 Nature Communications 14, 1 (2023), 2375.

 
 
 

 
 Chien et al . (2022) 
 
Eli Chien, Chao Pan, Jianhao Peng, and Olgica Milenkovic. 2022.

 
 You are allset: A multiset function framework for hypergraph neural networks. In ICLR .

 
 
 

 
 Chien et al . (2020) 
 
Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. 2020.

 
 Adaptive Universal Generalized PageRank Graph Neural Network. In ICLR .

 
 
 

 
 Chitra and Raphael (2019) 
 
Uthsav Chitra and Benjamin Raphael. 2019.

 
 Random walks on hypergraphs with edge-dependent vertex weights. In ICML .

 
 
 

 
 Choe et al . (2023) 
 
Minyoung Choe, Sunwoo Kim, Jaemin Yoo, and Kijung Shin. 2023.

 
 Classification of edge-dependent labels of nodes in hypergraphs. In KDD .

 
 
 

 
 Cui et al . (2024) 
 
Hejie Cui, Xinyu Fang, Ran Xu, Xuan Kan, Joyce C Ho, and Carl Yang. 2024.

 
 Multimodal fusion of ehr in structures and semantics: Integrating clinical records and notes with hypergraph and llm.

 
 arXiv preprint arXiv:2403.08818 (2024).

 
 
 

 
 de Arruda et al . (2020) 
 
Guilherme Ferraz de Arruda, Giovanni Petri, and Yamir Moreno. 2020.

 
 Social contagion models on hypergraphs.

 
 Physical Review Research 2, 2 (2020), 023032.

 
 
 

 
 Do and Shin (2024) 
 
Manh Tuan Do and Kijung Shin. 2024.

 
 Unsupervised alignmnet of hypergraphs with different scales. In KDD .

 
 
 

 
 Do et al . (2020) 
 
Manh Tuan Do, Se-eun Yoon, Bryan Hooi, and Kijung Shin. 2020.

 
 Structural patterns and generative models of real-world hypergraphs. In KDD .

 
 
 

 
 Dong et al . (2020) 
 
Yihe Dong, Will Sawin, and Yoshua Bengio. 2020.

 
 Hnhn: Hypergraph networks with hyperedge neurons. In ICML Workshop: Graph Representation Learning and Beyond .

 
 
 

 
 Du et al . (2022) 
 
Boxin Du, Changhe Yuan, Robert Barton, Tal Neiman, and Hanghang Tong. 2022.

 
 Self-supervised hypergraph representation learning. In Big Data .

 
 
 

 
 Dua et al . (2017) 
 
Dheeru Dua, Casey Graff, et al . 2017.

 
 Uci machine learning repository.

 
 (2017).

 
 
 

 
 Duta et al . (2023) 
 
Iulia Duta, Giulia Cassarà, Fabrizio Silvestri, and Pietro Liò. 2023.

 
 Sheaf hypergraph networks. In NeurIPS .

 
 
 

 
 Dwivedi et al . (2022) 
 
Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. 2022.

 
 Graph neural networks with learnable structural and positional representations. In ICLR .

 
 
 

 
 Expert and Petri (2022) 
 
Paul Expert and Giovanni Petri. 2022.

 
 Higher-order description of brain function.

 
 In Higher-Order Systems . Springer, 401–415.

 
 
 

 
 Fatemi et al . (2021) 
 
Bahare Fatemi, Layla El Asri, and Seyed Mehran Kazemi. 2021.

 
 Slaps: Self-supervision improves structure learning for graph neural networks. In NeurIPS .

 
 
 

 
 Feng et al . (2021) 
 
Song Feng, Emily Heath, Brett Jefferson, Cliff Joslyn, Henry Kvinge, Hugh D Mitchell, Brenda Praggastis, Amie J Eisfeld, Amy C Sims, Larissa B Thackray, et al . 2021.

 
 Hypergraph models of biological networks to identify genes critical to pathogenic viral response.

 
 BMC bioinformatics 22, 1 (2021), 287.

 
 
 

 
 Feng et al . (2024) 
 
Yifan Feng, Jiashu Han, Shihui Ying, and Yue Gao. 2024.

 
 Hypergraph isomorphism computation.

 
 IEEE Transactions on Pattern Analysis Machine Intelligence 01 (2024), 1–17.

 
 
 

 
 Feng et al . (2019) 
 
Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. 2019.

 
 Hypergraph neural networks. In AAAI .

 
 
 

 
 Gallo et al . (1993) 
 
Giorgio Gallo, Giustino Longo, Stefano Pallottino, and Sang Nguyen. 1993.

 
 Directed hypergraphs and applications.

 
 Discrete applied mathematics 42, 2-3 (1993), 177–201.

 
 
 

 
 Gao et al . (2022) 
 
Yue Gao, Yifan Feng, Shuyi Ji, and Rongrong Ji. 2022.

 
 HGNN+: General hypergraph neural networks.

 
 IEEE Transactions on Pattern Analysis Machine Intelligence 45, 3 (2022), 3181–3199.

 
 
 

 
 Gao et al . (2020) 
 
Yue Gao, Zizhao Zhang, Haojie Lin, Xibin Zhao, Shaoyi Du, and Changqing Zou. 2020.

 
 Hypergraph learning: Methods and practices.

 
 IEEE Transactions on Pattern Analysis Machine Intelligence 44, 5 (2020), 2548–2566.

 
 
 

 
 Gasteiger et al . (2019) 
 
Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Günnemann. 2019.

 
 Predict then propagate: Graph neural networks meet personalized pagerank. In ICLR .

 
 
 

 
 Gilmer et al . (2017) 
 
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. 2017.

 
 Neural message passing for quantum chemistry. In ICML .

 
 
 

 
 Gong and Cheng (2019) 
 
Liyu Gong and Qiang Cheng. 2019.

 
 Exploiting edge features for graph neural networks. In CVPR .

 
 
 

 
 Grover and Leskovec (2016) 
 
Aditya Grover and Jure Leskovec. 2016.

 
 node2vec: Scalable feature learning for networks. In KDD .

 
 
 

 
 Han et al . (2023) 
 
Yan Han, Peihao Wang, Souvik Kundu, Ying Ding, and Zhangyang Wang. 2023.

 
 Vision hgnn: An image is more than a graph of nodes. In ICCV .

 
 
 

 
 Hao et al . (2024) 
 
Xiaoke Hao, Jiawang Li, Mingming Ma, Jing Qin, Daoqiang Zhang, Feng Liu, Alzheimer’s Disease Neuroimaging Initiative, et al . 2024.

 
 Hypergraph convolutional network for longitudinal data analysis in Alzheimer’s disease.

 
 Computers in Biology and Medicine 168 (2024), 107765.

 
 
 

 
 Hayhoe et al . (2023) 
 
Mikhail Hayhoe, Hans Matthew Riess, Michael M Zavlanos, Victor Preciado, and Alejandro Ribeiro. 2023.

 
 Transferable Hypergraph Neural Networks via Spectral Similarity. In LoG .

 
 
 

 
 He et al . (2022) 
 
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022.

 
 Masked autoencoders are scalable vision learners. In CVPR .

 
 
 

 
 Hu et al . (2024) 
 
Lun Hu, Menglong Zhang, Pengwei Hu, Jun Zhang, Chao Niu, Xueying Lu, Xiangrui Jiang, and Yupeng Ma. 2024.

 
 Dual-channel hypergraph convolutional network for predicting herb–disease associations.

 
 Briefings in Bioinformatics 25, 2 (2024), bbae067.

 
 
 

 
 Huang et al . (2023) 
 
Buzhen Huang, Jingyi Ju, Zhihao Li, and Yangang Wang. 2023.

 
 Reconstructing groups of people with hypergraph relational reasoning. In ICCV .

 
 
 

 
 Huang et al . (2019) 
 
Jie Huang, Chuan Chen, Fanghua Ye, Jiajing Wu, Zibin Zheng, and Guohui Ling. 2019.

 
 Hyper2vec: Biased random walk for hyper-network embedding. In DASFAA 2019 International Workshops: BDMS, BDQM, and GDMA .

 
 
 

 
 Huang and Yang (2021) 
 
Jing Huang and Jie Yang. 2021.

 
 Unignn: a unified framework for graph and hypergraph neural networks. In IJCAI .

 
 
 

 
 Huang et al . (2024) 
 
Xingyue Huang, Miguel Romero Orth, Pablo Barceló, Michael M Bronstein, and İsmail İlkan Ceylan. 2024.

 
 Link prediction with relational hypergraphs.

 
 arXiv preprint arXiv:2402.04062 (2024).

 
 
 

 
 Hwang et al . (2022) 
 
Hyunjin Hwang, Seungwoo Lee, Chanyoung Park, and Kijung Shin. 2022.

 
 Ahp: Learning to negative sample for hyperedge prediction. In SIGIR .

 
 
 

 
 Iacopini et al . (2022) 
 
Iacopo Iacopini, Giovanni Petri, Andrea Baronchelli, and Alain Barrat. 2022.

 
 Group interactions modulate critical mass dynamics in social convention.

 
 Communications Physics 5, 1 (2022), 64.

 
 
 

 
 Jaiswal et al . (2020) 
 
Ashish Jaiswal, Ashwin Ramesh Babu, Mohammad Zaki Zadeh, Debapriya Banerjee, and Fillia Makedon. 2020.

 
 A survey on contrastive self-supervised learning.

 
 Technologies 9, 1 (2020), 2.

 
 
 

 
 Ji et al . (2022) 
 
Junzhong Ji, Yating Ren, and Minglong Lei. 2022.

 
 FC–HAT: Hypergraph attention network for functional brain network classification.

 
 Information Sciences 608 (2022), 1301–1316.

 
 
 

 
 Ji et al . (2020) 
 
Shuyi Ji, Yifan Feng, Rongrong Ji, Xibin Zhao, Wanwan Tang, and Yue Gao. 2020.

 
 Dual channel hypergraph collaborative filtering. In KDD .

 
 
 

 
 Jia et al . (2021) 
 
Renqi Jia, Xiaofei Zhou, Linhua Dong, and Shirui Pan. 2021.

 
 Hypergraph convolutional network for group recommendation. In ICDM .

 
 
 

 
 Jiang et al . (2019) 
 
Jianwen Jiang, Yuxuan Wei, Yifan Feng, Jingxuan Cao, and Yue Gao. 2019.

 
 Dynamic hypergraph neural networks.. In IJCAI .

 
 
 

 
 Keriven and Peyré (2019) 
 
Nicolas Keriven and Gabriel Peyré. 2019.

 
 Universal invariant and equivariant graph neural networks. In NeurIPS .

 
 
 

 
 Kim et al . (2022) 
 
Jinwoo Kim, Saeyoon Oh, Sungjun Cho, and Seunghoon Hong. 2022.

 
 Equivariant hypergraph neural networks. In ECCV .

 
 
 

 
 Kim et al . (2021) 
 
Jinwoo Kim, Saeyoon Oh, and Seunghoon Hong. 2021.

 
 Transformers generalize deepsets and can be extended to graphs hypergraphs. In NeurIPS .

 
 
 

 
 Kim et al . (2023a) 
 
Sunwoo Kim, Fanchen Bu, Minyoung Choe, Jaemin Yoo, and Kijung Shin. 2023a.

 
 How transitive are real-world group interactions?-Measurement and reproduction. In KDD .

 
 
 

 
 Kim et al . (2023b) 
 
Sunwoo Kim, Minyoung Choe, Jaemin Yoo, and Kijung Shin. 2023b.

 
 Reciprocity in directed hypergraphs: measures, findings, and generators.

 
 Data Mining and Knowledge Discovery 37, 6 (2023), 2330–2388.

 
 
 

 
 Kim et al . (2024) 
 
Sunwoo Kim, Shinhwan Kang, Fanchen Bu, Soo Yong Lee, Jaemin Yoo, and Kijung Shin. 2024.

 
 HypeBoy: Generative self-supervised representation learning on hypergraphs. In ICLR .

 
 
 

 
 Kim et al . (2023c) 
 
Sunwoo Kim, Dongjin Lee, Yul Kim, Jungho Park, Taeho Hwang, and Kijung Shin. 2023c.

 
 Datasets, tasks, and training methods for large-scale hypergraph learning.

 
 Data Mining and Knowledge Discovery 37, 6 (2023), 2216–2254.

 
 
 

 
 Kingma and Welling (2013) 
 
Diederik P Kingma and Max Welling. 2013.

 
 Auto-encoding variational bayes. In NeurIPS .

 
 
 

 
 Kipf and Welling (2016) 
 
Thomas N Kipf and Max Welling. 2016.

 
 Variational graph auto-encoders. In NeurIPs workshop on bayesian deep learning .

 
 
 

 
 Kipf and Welling (2017) 
 
Thomas N Kipf and Max Welling. 2017.

 
 Semi-supervised classification with graph convolutional networks. In ICLR .

 
 
 

 
 Ko et al . (2023) 
 
Yunyong Ko, Hanghang Tong, and Sang-Wook Kim. 2023.

 
 Enhancing hyperedge prediction with context-aware self-supervised learning.

 
 arXiv preprint arXiv:2309.05798 (2023).

 
 
 

 
 Lai et al . (2023) 
 
Yantong Lai, Yijun Su, Lingwei Wei, Gaode Chen, Tianci Wang, and Daren Zha. 2023.

 
 Multi-view spatial-temporal enhanced hypergraph network for next poi recommendation. In DASFAA .

 
 
 

 
 Lee and Shin (2023a) 
 
Dongjin Lee and Kijung Shin. 2023a.

 
 I’m me, we’re us, and i’m us: Tri-directional contrastive learning on hypergraphs. In AAAI .

 
 
 

 
 Lee et al . (2024a) 
 
Geon Lee, Fanchen Bu, Tina Eliassi-Rad, and Kijung Shin. 2024a.

 
 A survey on hypergraph mining: Patterns, tools, and generators.

 
 arXiv preprint arXiv:2401.08878 (2024).

 
 
 

 
 Lee et al . (2021) 
 
Geon Lee, Minyoung Choe, and Kijung Shin. 2021.

 
 How do hyperedges overlap in real-world hypergraphs?-patterns, measures, and generators. In WWW .

 
 
 

 
 Lee et al . (2020) 
 
Geon Lee, Jihoon Ko, and Kijung Shin. 2020.

 
 Hypergraph motifs: concepts, algorithms, and discoveries.

 
 Proceedings of the VLDB Endowment 13, 11 (2020), 2256–2269.

 
 
 

 
 Lee et al . (2024c) 
 
Geon Lee, Soo Yong Lee, and Kijung Shin. 2024c.

 
 VilLain: Self-supervised learning on homogeneous hypergraphs without features via virtual label propagation. In WWW .

 
 
 

 
 Lee and Shin (2023b) 
 
Geon Lee and Kijung Shin. 2023b.

 
 Temporal hypergraph motifs.

 
 Knowledge and Information Systems 65, 4 (2023), 1549–1586.

 
 
 

 
 Lee et al . (2019) 
 
Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019.

 
 Set transformer: A framework for attention-based permutation-invariant neural networks. In ICML .

 
 
 

 
 Lee et al . (2023) 
 
Soo Yong Lee, Fanchen Bu, Jaemin Yoo, and Kijung Shin. 2023.

 
 Towards deep attention in graph neural networks: Problems and remedies. In ICML .

 
 
 

 
 Lee et al . (2024b) 
 
Soo Yong Lee, Sunwoo Kim, Fanchen Bu, Jaemin Yoo, Jiliang Tang, and Kijung Shin. 2024b.

 
 Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective. In ICML .

 
 
 

 
 Lei et al . (2024) 
 
Fangyuan Lei, Jiahao Huang, Jianjian Jiang, Da Huang, Zhengming Li, and Chang-Dong Wang. 2024.

 
 Unveiling the potential of long-range dependence with mask-guided structure learning for hypergraph.

 
 Knowledge-Based Systems 284 (2024), 111254.

 
 
 

 
 Li et al . (2024) 
 
Fan Li, Xiaoyang Wang, Dawei Cheng, Wenjie Zhang, Ying Zhang, and Xuemin Lin. 2024.

 
 Hypergraph Self-supervised Learning with Sampling-efficient Signals. In IJCAI .

 
 
 

 
 Li et al . (2023) 
 
Wei Li, Bin Xiang, Fan Yang, Yu Rong, Yanbin Yin, Jianhua Yao, and Han Zhang. 2023.

 
 Scmhnn: A novel hypergraph neural network for integrative analysis of single-cell epigenomic, transcriptomic and proteomic data.

 
 Briefings in Bioinformatics 24, 6 (2023), bbad391.

 
 
 

 
 Li et al . (2021) 
 
Yicong Li, Hongxu Chen, Xiangguo Sun, Zhenchao Sun, Lin Li, Lizhen Cui, Philip S Yu, and Guandong Xu. 2021.

 
 Hyperbolic hypergraphs for sequential recommendation. In CIKM .

 
 
 

 
 Li et al . (2022a) 
 
Yinfeng Li, Chen Gao, Hengliang Luo, Depeng Jin, and Yong Li. 2022a.

 
 Enhancing hypergraph neural networks with intent disentanglement for session-based recommendation. In SIGIR .

 
 
 

 
 Li et al . (2022b) 
 
Zhonghang Li, Chao Huang, Lianghao Xia, Yong Xu, and Jian Pei. 2022b.

 
 Spatial-temporal hypergraph self-supervised learning for crime prediction. In ICDE .

 
 
 

 
 Liang et al . (2024) 
 
Langzhang Liang, Sunwoo Kim, Kijung Shin, Zenglin Xu, Shirui Pan, and Yuan Qi. 2024.

 
 Sign is Not a Remedy: Multiset-to-Multiset Message Passing for Learning on Heterophilic Graphs. In ICML .

 
 
 

 
 Liao et al . (2024) 
 
Sihao Liao, Liang Xie, Yuanchuang Du, Shengshuang Chen, Hongyang Wan, and Haijiao Xu. 2024.

 
 Stock trend prediction based on dynamic hypergraph spatio-temporal network.

 
 Applied Soft Computing 154 (2024), 111329.

 
 
 

 
 Liu et al . (2023) 
 
Luotao Liu, Feng Huang, Xuan Liu, Zhankun Xiong, Menglu Li, Congzhi Song, and Wen Zhang. 2023.

 
 Multi-view contrastive learning hypergraph neural network for drug-microbe-disease association prediction. In IJCAI .

 
 
 

 
 Liu et al . (2020) 
 
Shengyuan Liu, Pei Lv, Yuzhen Zhang, Jie Fu, Junjin Cheng, Wanqing Li, Bing Zhou, and Mingliang Xu. 2020.

 
 Semi-dynamic hypergraph neural network for 3d pose estimation.. In IJCAI .

 
 
 

 
 Liu et al . (2022) 
 
Xuan Liu, Congzhi Song, Shichao Liu, Menglu Li, Xionghui Zhou, and Wen Zhang. 2022.

 
 Multi-way relation-enhanced hypergraph representation learning for anti-cancer drug synergy prediction.

 
 Bioinformatics 38, 20 (2022), 4782–4789.

 
 
 

 
 Liu et al . (2024) 
 
Zexi Liu, Bohan Tang, Ziyuan Ye, Xiaowen Dong, Siheng Chen, and Yanfeng Wang. 2024.

 
 Hypergraph transformer for semi-supervised classification.

 
 ICASSP .

 
 
 

 
 Luan et al . (2022) 
 
Sitao Luan, Chenqing Hua, Qincheng Lu, Jiaqi Zhu, Mingde Zhao, Shuyuan Zhang, Xiao-Wen Chang, and Doina Precup. 2022.

 
 Revisiting heterophily for graph neural networks. In NeurIPS .

 
 
 

 
 Luo et al . (2022) 
 
Xiaoyi Luo, Jiaheng Peng, and Jun Liang. 2022.

 
 Directed hypergraph attention network for traffic forecasting.

 
 IET Intelligent Transport Systems 16, 1 (2022), 85–98.

 
 
 

 
 Ma et al . (2022b) 
 
Jing Ma, Mengting Wan, Longqi Yang, Jundong Li, Brent Hecht, and Jaime Teevan. 2022b.

 
 Learning causal effects on hypergraphs. In KDD .

 
 
 

 
 Ma et al . (2023) 
 
Tianyi Ma, Yiyue Qian, Chuxu Zhang, and Yanfang Ye. 2023.

 
 Hypergraph Contrastive Learning for Drug Trafficking Community Detection. In ICDM .

 
 
 

 
 Ma et al . (2022a) 
 
Yao Ma, Xiaorui Liu, Neil Shah, and Jiliang Tang. 2022a.

 
 Is homophily a necessity for graph neural networks?. In ICLR .

 
 
 

 
 Madine et al . (2020) 
 
Mohammad Madine, Islem Rekik, and Naoufel Werghi. 2020.

 
 Diagnosing autism using t1-w mri with multi-kernel learning and hypergraph neural network. In ICIP .

 
 
 

 
 Maron et al . (2019) 
 
Haggai Maron, Ethan Fetaya, Nimrod Segol, and Yaron Lipman. 2019.

 
 On the universality of invariant networks. In ICML .

 
 
 

 
 Mickalide and Kuehn (2019) 
 
Harry Mickalide and Seppe Kuehn. 2019.

 
 Higher-order interaction between species inhibits bacterial invasion of a phototroph-predator microbial community.

 
 Cell systems 9, 6 (2019), 521–533.

 
 
 

 
 Moon et al . (2023) 
 
Heechan Moon, Hyunju Kim, Sunwoo Kim, and Kijung Shin. 2023.

 
 Four-set hypergraphlets for characterization of directed hypergraphs.

 
 arXiv preprint arXiv:2311.14289 (2023).

 
 
 

 
 Morin et al . (2022) 
 
Manon A Morin, Anneliese J Morrison, Michael J Harms, and Rachel J Dutton. 2022.

 
 Higher-order interactions shape microbial interactions as microbial community complexity increases.

 
 Scientific Reports 12, 1 (2022), 22640.

 
 
 

 
 Nguyen et al . (2022) 
 
Duc Anh Nguyen, Canh Hao Nguyen, Peter Petschner, and Hiroshi Mamitsuka. 2022.

 
 Sparse: A sparse hypergraph neural network for learning multiple types of latent combinations to accurately predict drug-drug interactions.

 
 Bioinformatics 38 (2022), i333–i341.

 
 
 

 
 OpenAI (2023) 
 
OpenAI. 2023.

 
 Gpt-4 technical report.

 
 (2023).

 
 
 

 
 Papamarkou et al . (2024) 
 
Theodore Papamarkou, Tolga Birdal, Michael M Bronstein, Gunnar E Carlsson, Justin Curry, Yue Gao, Mustafa Hajij, Roland Kwitt, Pietro Lio, Paolo Di Lorenzo, et al . 2024.

 
 Position: Topological Deep Learning is the New Frontier for Relational Learning. In ICML .

 
 
 

 
 Patil et al . (2020) 
 
Prasanna Patil, Govind Sharma, and M Narasimha Murty. 2020.

 
 Negative sampling for hyperlink prediction in networks. In PAKDD .

 
 
 

 
 Peng et al . (2024) 
 
Wei Peng, Zhichen He, Wei Dai, and Wei Lan. 2024.

 
 Mhclmda: Multihypergraph contrastive learning for mirna–disease association prediction.

 
 Briefings in Bioinformatics 25, 1 (2024), bbad524.

 
 
 

 
 Prokopchik et al . (2022) 
 
Konstantin Prokopchik, Austin R Benson, and Francesco Tudisco. 2022.

 
 Nonlinear feature diffusion on hypergraphs. In ICML .

 
 
 

 
 Qian et al . (2023) 
 
Yiyue Qian, Tianyi Ma, Chuxu Zhang, and Yanfang Ye. 2023.

 
 Adaptive Expansion for Hypergraph Learning.

 
 (2023).

 
 
 

 
 Qian et al . (2024) 
 
Yiyue Qian, Tianyi Ma, Chuxu Zhang, and Yanfang Ye. 2024.

 
 Dual-level Hypergraph Contrastive Learning with Adaptive Temperature Enhancement. In WWW .

 
 
 

 
 Qiu et al . (2020) 
 
Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, and Jie Tang. 2020.

 
 Gcc: Graph contrastive coding for graph neural network pre-training. In KDD .

 
 
 

 
 Qu et al . (2023) 
 
Shilin Qu, Weiqing Wang, Yuan-Fang Li, Xin Zhou, and Fajie Yuan. 2023.

 
 Hypergraph node representation learning with one-stage message passing.

 
 arXiv preprint arXiv:2312.00336 (2023).

 
 
 

 
 Ruan et al . (2021) 
 
Ding Ruan, Shuyi Ji, Chenggang Yan, Junjie Zhu, Xibin Zhao, Yuedong Yang, Yue Gao, Changqing Zou, and Qionghai Dai. 2021.

 
 Exploring complex and heterogeneous correlations on hypergraph for the prediction of drug-target interactions.

 
 Patterns 2, 12 (2021).

 
 
 

 
 Saifuddin et al . (2023a) 
 
Khaled Mohammed Saifuddin, Mehmet Emin Aktas, and Esra Akbas. 2023a.

 
 Topology-guided hypergraph transformer network: Unveiling structural insights for improved representation.

 
 arXiv preprint arXiv:2310.09657 (2023).

 
 
 

 
 Saifuddin et al . (2023b) 
 
Khaled Mohammed Saifuddin, Briana Bumgardner, Farhan Tanvir, and Esra Akbas. 2023b.

 
 Hygnn: Drug-drug interaction prediction via hypergraph neural network. In ICDE .

 
 
 

 
 Sawhney et al . (2021) 
 
Ramit Sawhney, Shivam Agarwal, Arnav Wadhwa, Tyler Derr, and Rajiv Ratn Shah. 2021.

 
 Stock selection via spatiotemporal hypergraph attention network: A learning to rank approach. In AAAI .

 
 
 

 
 Shang and Chen (2024) 
 
Zongjiang Shang and Ling Chen. 2024.

 
 Mshyper: Multi-scale hypergraph transformer for long-range time series forecasting.

 
 arXiv preprint arXiv:2401.09261 (2024).

 
 
 

 
 Tan et al . (2023) 
 
Qiaoyu Tan, Ninghao Liu, Xiao Huang, Soo-Hyun Choi, Li Li, Rui Chen, and Xia Hu. 2023.

 
 S2GAE: self-supervised graph autoencoders are generalizable learners with graph masking. In WSDM .

 
 
 

 
 Tang et al . (2024) 
 
Bohan Tang, Zexi Liu, Keyue Jiang, Siheng Chen, and Xiaowen Dong. 2024.

 
 Hypergraph node classification With graph neural networks.

 
 arXiv preprint arXiv:2402.05569 (2024).

 
 
 

 
 Tao et al . (2023) 
 
Wen Tao, Yuansheng Liu, Xuan Lin, Bosheng Song, and Xiangxiang Zeng. 2023.

 
 Prediction of multi-relational drug-gene interaction via dynamic hypergraph contrastive learning.

 
 Briefings in Bioinformatics 24, 6 (2023), bbad371.

 
 
 

 
 Telyatnikov et al . (2023) 
 
Lev Telyatnikov, Maria Sofia Bucarelli, Guillermo Bernardez, Olga Zaghen, Simone Scardapane, and Pietro Lio. 2023.

 
 Hypergraph neural networks through the lens of message passing: a common perspective to homophily and architecture design.

 
 arXiv preprint arXiv:2310.07684 (2023).

 
 
 

 
 Tran and Tran (2020) 
 
Loc Hoang Tran and Linh Hoang Tran. 2020.

 
 Directed hypergraph neural network.

 
 arXiv preprint arXiv:2008.03626 (2020).

 
 
 

 
 Vaswani et al . (2017) 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.

 
 Attention is all you need. In NeurIPS .

 
 
 

 
 Veličković et al . (2019) 
 
Petar Veličković, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio, and R Devon Hjelm. 2019.

 
 Deep graph infomax.

 
 ICLR .

 
 
 

 
 Vignac et al . (2020) 
 
Clement Vignac, Andreas Loukas, and Pascal Frossard. 2020.

 
 Building powerful and equivariant graph neural networks with structural message-passing. In NeurIPS .

 
 
 

 
 Wadhwa et al . (2021) 
 
Gourav Wadhwa, Abhinav Dhall, Subrahmanyam Murala, and Usman Tariq. 2021.

 
 Hyperrealistic image inpainting with hypergraphs. In WACV .

 
 
 

 
 Wan et al . (2021) 
 
Changlin Wan, Muhan Zhang, Wei Hao, Sha Cao, Pan Li, and Chi Zhang. 2021.

 
 Principled hyperedge prediction with structural spectral features and neural networks.

 
 arXiv preprint arXiv:2106.04292 (2021).

 
 
 

 
 Wang et al . (2024a) 
 
Fuli Wang, Karelia Pena-Pena, Wei Qian, and Gonzalo R Arce. 2024a.

 
 T-hypergnns: Hypergraph neural networks via tensor representations.

 
 IEEE Transactions on Neural Networks and Learning Systems (2024).

 
 
 

 
 Wang et al . (2022) 
 
Haorui Wang, Haoteng Yin, Muhan Zhang, and Pan Li. 2022.

 
 Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks. In ICLR .

 
 
 

 
 Wang et al . (2020) 
 
Jianling Wang, Kaize Ding, Liangjie Hong, Huan Liu, and James Caverlee. 2020.

 
 Next-item recommendation with sequential hypergraphs. In SIGIR .

 
 
 

 
 Wang et al . (2021) 
 
Jianling Wang, Kaize Ding, Ziwei Zhu, and James Caverlee. 2021.

 
 Session-based recommendation with hypergraph attention networks. In SDM .

 
 
 

 
 Wang et al . (2023b) 
 
Junqi Wang, Hailong Li, Gang Qu, Kim M Cecil, Jonathan R Dillman, Nehal A Parikh, and Lili He. 2023b.

 
 Dynamic weighted hypergraph convolutional network for brain functional connectome analysis.

 
 Medical Image Analysis 87 (2023), 102828.

 
 
 

 
 Wang et al . (2024d) 
 
Maolin Wang, Yaoming Zhen, Yu Pan, Zenglin Xu, Ruocheng Guo, and Xiangyu Zhao. 2024d.

 
 Tensorized hypergraph neural networks. In SDM .

 
 
 

 
 Wang et al . (2023c) 
 
Peihao Wang, Shenghao Yang, Yunyu Liu, Zhangyang Wang, and Pan Li. 2023c.

 
 Equivariant hypergraph diffusion neural operators. In ICLR .

 
 
 

 
 Wang et al . (2024c) 
 
Shun Wang, Yong Zhang, Xuanqi Lin, Yongli Hu, Qingming Huang, and Baocai Yin. 2024c.

 
 Dynamic Hypergraph Structure Learning for Multivariate Time Series Forecasting.

 
 IEEE Transactions on Big Data 01 (2024), 1–13.

 
 
 

 
 Wang et al . (2024b) 
 
Wei Wang, Gaolin Yuan, Shitong Wan, Ziwei Zheng, Dong Liu, Hongjun Zhang, Juntao Li, Yun Zhou, and Xianfang Wang. 2024b.

 
 A granularity-level information fusion strategy on hypergraph transformer for predicting synergistic effects of anticancer drugs.

 
 Briefings in Bioinformatics 25, 1 (2024), bbad522.

 
 
 

 
 Wang et al . (2023a) 
 
Yuxin Wang, Quan Gan, Xipeng Qiu, Xuanjing Huang, and David Wipf. 2023a.

 
 From hypergraph energy functions to hypergraph neural networks. In ICML .

 
 
 

 
 Wei et al . (2022) 
 
Tianxin Wei, Yuning You, Tianlong Chen, Yang Shen, Jingrui He, and Zhangyang Wang. 2022.

 
 Augmentations in hypergraph contrastive learning: Fabricated and generative. In NeurIPS .

 
 
 

 
 Wu et al . (2019) 
 
Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. 2019.

 
 Simplifying graph convolutional networks. In ICML .

 
 
 

 
 Wu et al . (2023a) 
 
Jialun Wu, Kai He, Rui Mao, Chen Li, and Erik Cambria. 2023a.

 
 MEGACare: Knowledge-guided multi-view hypergraph predictive framework for healthcare.

 
 Information Fusion 100 (2023), 101939.

 
 
 

 
 Wu et al . (2023b) 
 
Jinming Wu, Qi Qi, Jingyu Wang, Haifeng Sun, Zhikang Wu, Zirui Zhuang, and Jianxin Liao. 2023b.

 
 Not only pairwise relationships: fine-grained relational modeling for multivariate time series forecasting. In IJCAI .

 
 
 

 
 Xia et al . (2022b) 
 
Lianghao Xia, Chao Huang, Yong Xu, Jiashu Zhao, Dawei Yin, and Jimmy Huang. 2022b.

 
 Hypergraph contrastive collaborative filtering. In SIGIR .

 
 
 

 
 Xia et al . (2022a) 
 
Lianghao Xia, Chao Huang, and Chuxu Zhang. 2022a.

 
 Self-supervised hypergraph transformer for recommender systems. In KDD .

 
 
 

 
 Xia et al . (2021) 
 
Xin Xia, Hongzhi Yin, Junliang Yu, Qinyong Wang, Lizhen Cui, and Xiangliang Zhang. 2021.

 
 Self-supervised hypergraph convolutional networks for session-based recommendation. In AAAI .

 
 
 

 
 Xu et al . (2019) 
 
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2019.

 
 How powerful are graph neural networks?. In ICLR .

 
 
 

 
 Xu et al . (2023) 
 
Ran Xu, Mohammed K Ali, Joyce C Ho, and Carl Yang. 2023.

 
 Hypergraph transformers for ehr-based clinical predictions.

 
 AMIA Summits on Translational Science Proceedings 2023 (2023), 582.

 
 
 

 
 Xu et al . (2022a) 
 
Ran Xu, Yue Yu, Chao Zhang, Mohammed K Ali, Joyce C Ho, and Carl Yang. 2022a.

 
 Counterfactual and factual reasoning over hypergraphs for interpretable clinical predictions on ehr. In ML4H .

 
 
 

 
 Xu et al . (2022b) 
 
Xixia Xu, Qi Zou, and Xue Lin. 2022b.

 
 Adaptive hypergraph neural network for multi-person pose estimation. In AAAI .

 
 
 

 
 Yadati (2020) 
 
Naganand Yadati. 2020.

 
 Neural message passing for multi-relational ordered and recursive hypergraphs. In NeurIPS .

 
 
 

 
 Yadati et al . (2019) 
 
Naganand Yadati, Madhav Nimishakavi, Prateek Yadav, Vikram Nitin, Anand Louis, and Partha Talukdar. 2019.

 
 Hypergcn: A new method for training graph convolutional networks on hypergraphs. In NeurIPS .

 
 
 

 
 Yadati et al . (2020) 
 
Naganand Yadati, Vikram Nitin, Madhav Nimishakavi, Prateek Yadav, Anand Louis, and Partha Talukdar. 2020.

 
 NHP: Neural hypergraph link prediction. In CIKM .

 
 
 

 
 Yan et al . (2024b) 
 
Jielong Yan, Yifan Feng, Shihui Ying, and Yue Gao. 2024b.

 
 Hypergraph dynamic system. In ICLR .

 
 
 

 
 Yan et al . (2024a) 
 
Yuguang Yan, Yuanlin Chen, Shibo Wang, Hanrui Wu, and Ruichu Cai. 2024a.

 
 Hypergraph Joint Representation Learning for Hypervertices and Hyperedges via Cross Expansion. In AAAI .

 
 
 

 
 Yan et al . (2020) 
 
Yichao Yan, Jie Qin, Jiaxin Chen, Li Liu, Fan Zhu, Ying Tai, and Ling Shao. 2020.

 
 Learning multi-granular hypergraphs for video-based person re-identification. In CVPR .

 
 
 

 
 Yang et al . (2022a) 
 
Chaoqi Yang, Ruijie Wang, Shuochao Yao, and Tarek Abdelzaher. 2022a.

 
 Hypergraph learning with line expansion. In CIKM .

 
 
 

 
 Yang et al . (2022b) 
 
Chaoqi Yang, Ruijie Wang, Shuochao Yao, and Tarek Abdelzaher. 2022b.

 
 Semi-supervised hypergraph node classification on hypergraph line expansion. In CIKM .

 
 
 

 
 Yang et al . (2016) 
 
Zhilin Yang, William Cohen, and Ruslan Salakhudinov. 2016.

 
 Revisiting semi-supervised learning with graph embeddings. In ICML .

 
 
 

 
 Yi and Park (2020) 
 
Jaehyuk Yi and Jinkyoo Park. 2020.

 
 Hypergraph convolutional recurrent neural network. In KDD .

 
 
 

 
 Yoo et al . (2022) 
 
Jaemin Yoo, Hyunsik Jeon, Jinhong Jung, and U Kang. 2022.

 
 Accurate node feature estimation with structured variational graph autoencoder. In KDD .

 
 
 

 
 Yoon et al . (2020) 
 
Se-eun Yoon, Hyungseok Song, Kijung Shin, and Yung Yi. 2020.

 
 How much and when do we need higher-order information in hypergraphs? a case study on hyperedge prediction. In WWW .

 
 
 

 
 You et al . (2021) 
 
Jiaxuan You, Jonathan M Gomes-Selman, Rex Ying, and Jure Leskovec. 2021.

 
 Identity-aware graph neural networks. In AAAI .

 
 
 

 
 You et al . (2020) 
 
Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. 2020.

 
 Graph contrastive learning with augmentations. In NeurIPS .

 
 
 

 
 Yu et al . (2011) 
 
Shan Yu, Hongdian Yang, Hiroyuki Nakahara, Gustavo S Santos, Danko Nikolić, and Dietmar Plenz. 2011.

 
 Higher-order interactions characterized in cortical activity.

 
 Journal of neuroscience 31, 48 (2011), 17514–17526.

 
 
 

 
 Zhang et al . (2022a) 
 
Jiying Zhang, Yuzhao Chen, Xi Xiao, Runiu Lu, and Shu-Tao Xia. 2022a.

 
 Learnable hypergraph laplacian for hypergraph learning. In ICASSP .

 
 
 

 
 Zhang et al . (2021) 
 
Junwei Zhang, Min Gao, Junliang Yu, Lei Guo, Jundong Li, and Hongzhi Yin. 2021.

 
 Double-scale self-supervised hypergraph learning for group recommendation. In CIKM .

 
 
 

 
 Zhang et al . (2022c) 
 
Jiying Zhang, Fuyang Li, Xi Xiao, Tingyang Xu, Yu Rong, Junzhou Huang, and Yatao Bian. 2022c.

 
 Hypergraph convolutional networks via equivalency between hypergraphs and undirected graphs.

 
 ICML Workshop on Topology, Algebra, and Geometry in Machine Learning .

 
 
 

 
 Zhang and Chen (2018) 
 
Muhan Zhang and Yixin Chen. 2018.

 
 Link prediction based on graph neural networks. In NeurIPS .

 
 
 

 
 Zhang et al . (2020) 
 
Ruochi Zhang, Yuesong Zou, and Jian Ma. 2020.

 
 Hyper-SAGNN: a self-attention based graph neural network for hypergraphs. In ICLR .

 
 
 

 
 Zhang et al . (2022b) 
 
Zizhao Zhang, Yifan Feng, Shihui Ying, and Yue Gao. 2022b.

 
 Deep hypergraph structure learning.

 
 arXiv preprint arXiv:2208.12547 (2022).

 
 
 

 
 Zhao et al . (2023b) 
 
Sen Zhao, Wei Wei, Xian-Ling Mao, Shuai Zhu, Minghui Yang, Zujie Wen, Dangyang Chen, and Feida Zhu. 2023b.

 
 Multi-view hypergraph contrastive policy learning for conversational recommendation. In SIGIR .

 
 
 

 
 Zhao et al . (2023a) 
 
Yusheng Zhao, Xiao Luo, Wei Ju, Chong Chen, Xian-Sheng Hua, and Ming Zhang. 2023a.

 
 Dynamic hypergraph structure learning for traffic flow forecasting. In ICDE .

 
 
 

 
 Zheng and Worring (2024) 
 
Yijia Zheng and Marcel Worring. 2024.

 
 Co-Representation Neural Hypergraph Diffusion for Edge-Dependent Node Classification.

 
 arXiv preprint arXiv:2405.14286 (2024).

 
 
 

 
 Zhezheng et al . (2023) 
 
Luo Zhezheng, Mao Jiayuan, Tenenbaum Joshua B., and Kaelbling Leslie, Pack. 2023.

 
 On the expressiveness and generalization of hypergraph neural networks. In LoG .

 
 
 

 
 Zhou et al . (2023b) 
 
Peng Zhou, Zongqian Wu, Xiangxiang Zeng, Guoqiu Wen, Junbo Ma, and Xiaofeng Zhu. 2023b.

 
 Totally dynamic hypergraph neural network. In IJCAI .

 
 
 

 
 Zhou et al . (2023a) 
 
Xue Zhou, Bei Hui, Ilana Zeira, Hao Wu, and Ling Tian. 2023a.

 
 Dynamic relation learning for link prediction in knowledge hypergraphs.

 
 Applied Intelligence 53, 22 (2023), 26580–26591.

 
 
 

 
 Zhou et al . (2022) 
 
Yuxuan Zhou, Zhi-Qi Cheng, Chao Li, Yanwen Fang, Yifeng Geng, Xuansong Xie, and Margret Keuper. 2022.

 
 Hypergraph transformer for skeleton-based action recognition.

 
 arXiv preprint arXiv:2211.09590 (2022).

 
 
 

 
 Zhu et al . (2022) 
 
Fanglin Zhu, Shunyu Chen, Yonghui Xu, Wei He, Fuqiang Yu, Xu Zhang, and Lizhen Cui. 2022.

 
 Temporal hypergraph for personalized clinical pathway recommendation. In BIBM .

 
 
 

 
 Zhu et al . (2021) 
 
Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. 2021.

 
 Neural bellman-ford networks: A general graph neural network framework for link prediction. In NeurIPS .

 
 
 

 
 Zou et al . (2024) 
 
Minhao Zou, Zhongxue Gan, Yutong Wang, Junheng Zhang, Dongyan Sui, Chun Guan, and Siyang Leng. 2024.

 
 Unig-encoder: A universal feature encoder for graph and hypergraph node classification.

 
 Pattern Recognition 147 (2024), 110115.