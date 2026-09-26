A Survey on Self-Supervised Graph Foundation Models: Knowledge-Based Perspective 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.16137v3 [cs.LG] 06 May 2025 
 
 

# A Survey on Self-Supervised Graph Foundation Models: Knowledge-Based Perspective

 
 
 Ziwen Zhao
 
    
 Yixin Su
 
    
 Yuhua Li
 † † thanks: Y.˜Li and R.˜Zhang are corresponding authors. 
    
 Yixiong Zou
 
    
 Ruixuan Li
 
    
 and Rui Zhang
 † † thanks: 
Z.˜Zhao, Y.˜Su, Y.˜Li, Y.˜Zou, R.˜Li, and R.˜Zhang are with School of Computer Science and Technology, Huazhong University of Science and Technology. E-mail: {zwzhao, idcliyuhua, yixiongz, rxli}@hust.edu.cn, yixin.su@outlook.com, rayteam@yeah.net ( www.ruizhang.info ).
 † † thanks: $ˆ†$ Z.˜Zhao and Y.˜Su are co-first authors. † † thanks: 
This work is supported by the National Key Research and Development Program of China under grant 2024YFC3307900; the National Natural Science Foundation of China under grants 62436003, 62376103, 62206102 and 62302184; the Science and Technology Support Program of Hubei Province under grant 2022BAA046; Hubei science and technology talent service project under grant 2024DJC078; Ant Group through CCF-Ant Research Fund; and the HPC Platform of Huazhong University of Science and Technology.
 † † thanks: Y.˜Li and R.˜Zhang are corresponding authors. 

 Abstract 
 
 The field of graph foundation models (GFMs) has seen a dramatic rise in interest in recent years.
Their powerful generalization ability is believed to be endowed by self-supervised pre-training and downstream tuning techniques.
There is a wide variety of knowledge patterns embedded in the graph data, such as node properties and clusters, which are crucial for learning generalized representations for GFMs.
We present a comprehensive survey of self-supervised GFMs from a novel knowledge-based perspective.
Our main contribution is a knowledge-based taxonomy that categorizes self-supervised graph models by the specific graph knowledge utilized: microscopic (nodes, links, etc.), mesoscopic (context, clusters, etc.), and macroscopic (global structure, manifolds, etc.).
It covers a total of 9 knowledge categories and 300 references for self-supervised pre-training as well as various downstream tuning strategies.
Such a knowledge-based taxonomy allows us to more clearly re-examine potential GFM architectures, including large language models (LLMs), as well as provide deeper insights for constructing future GFMs.

 
 
 
 Index Terms:  Graph foundation models, self-supervised learning, pre-training, graph neural networks, large language models

 
 

## I Introduction 

 
 Graphs are prevalent in various real-world applications. They exhibit diverse knowledge patterns due to the inherent topology  [ 1 , 2 , 3 ] . Moreover, the availability of features and properties associated with nodes and links, such as textual attributes and centrality measures, further enriches the knowledge present in graphs. Over time, deep graph mining techniques have evolved from graph neural networks (GNNs)  [ 4 , 5 , 6 ] to graph Transformers (GTs)  [ 7 , 8 ] and more recent large language model (LLM)-based graph language models  [ 9 , 10 , 11 ] . They are motivated by capturing more comprehensive knowledge patterns within the graph data, from local relationships to the global structure.

 
 
 Fig. 1: 
How self-supervised GFMs are believed to work: pre-training and downstream tuning.
Updating the pre-trained model during downstream tuning is optional depending on the tuning strategy.
 
 
 
 However, when confronted with various downstream task requirements, researchers often encounter graph data that lacks available labels, such as the field of an article in citation networks.
Fortunately, self-supervised learning on graphs has emerged as a powerful approach to uncovering underlying patterns in enormous unannotated data  [ 12 , 13 ] .
SSL methods design unsupervised tasks – pretext tasks – to pre-train a graph model, and adapt the pre-trained model to the specific application scenarios by downstream tuning approaches, as depicted in Fig.  1 .
Researchers have observed powerful generalization ability within graph models pre-trained with self-supervision  [ 14 ] , as they aim to mine the underlying knowledge patterns of graph data instead of solely relying on manual labels that are limited to specific task spaces.
Therefore, self-supervised pre-training and downstream tuning are believed to be the most promising techniques to achieve a graph foundation model (GFM) – a highly generalized model that can handle a wide range of application tasks  [ 15 ] .

 
 
 Previous efforts .
The popularity of self-supervised learning and LLMs on graphs in recent years has given rise to a flood of surveys.
Early efforts  [ 16 , 17 , 3 ] focus on summarizing general self-supervised graph models. [ 18 , 19 , 20 , 21 ] systematically summarize the trending direction of graphs meet LLMs, shortly after the sensational debut of ChatGPT. The success of LLMs has also activated heated discussions towards GFMs  [ 14 , 22 ] , summarizing key techniques and principles of learning generalized graph models and providing outlooks towards the realization of GFMs.
Despite the promising work, we reveal three major shortcomings of the existing surveys:

 
 
 (1) Lack of comprehensiveness :
existing surveys in the field of self-supervised graph learning  [ 16 , 17 , 3 ] do not cover the latest progress in this fast-developing field. For example, none of these surveys have discussed the new achievements of masked graph autoencoders  [ 23 ] and learning graph manifolds  [ 24 ] .
A recent survey  [ 25 ] includes cutting-edge developments in graph contrastive learning, yet it focuses on real-world applications rather than realizing GFMs.

 
 
 (2) Unclear categorization :
existing surveys  [ 14 , 17 , 3 ] broadly categorize graph pre-training methods as “generative – contrastive (predictive)”.
This rough categorization is insufficient to capture the unique characteristics of graphs, which have diverse knowledge patterns embedded in their structure and properties. For instance, predicting links requires local relationships between nodes, whereas predicting clusters requires the node distribution on the entire graph. However, both generative and contrastive (predictive) frameworks can utilize the knowledge of links  [ 12 , 8 ] and clusters  [ 26 , 27 ] , which the aforementioned taxonomy fails to distinguish.
On the other hand, recent surveys of GFMs only give a brief summary of existing pre-training and downstream tuning methods: [ 14 ] puts the emphasis on the architecture design of graph models, while [ 18 , 22 ] are closer to outlooks towards future directions of GFMs.

 
 
 (3) Limited to specific architectures :
the aforementioned graph self-supervised learning surveys are limited to GNNs/GTs only. On the other hand, LLM-based surveys  [ 19 , 20 , 21 ] overemphasize the language model architectures and textual attributes of graphs while overlooking other structural patterns. A recent GFM survey  [ 14 ] categorizes existing studies into three groups of GNN, LLM, and GNN+LLM, still limited by specific backbone architectures instead of an in-depth perspective towards the ultimate goal – mining generalized graph knowledge.
As language models are not designed for mining various types of graph knowledge, it still remains an unanswered question if LLMs are ideal architectures for GFMs. If other promising generalized architectures showed up in the near future (which is happening right now), their architecture-based taxonomy might no longer apply.

 
 
 Our contributions .
Considering the aforementioned issues, it is necessary to provide a comprehensive survey of self-supervised graph models with a clearer categorization and taxonomy, which will offer a better understanding and greater insight into how future GFMs work.
We first propose a knowledge-based taxonomy that categorizes self-supervised graph pre-training based on the types of knowledge utilized:
 microscopic pre-training (Section III ) that focuses on individual nodes and links;
 mesoscopic pre-training (Section IV ) that focuses on local relationships in the graph, such as context and clusters;
and macroscopic pre-training (Section V ) that focuses on the structure and the manifold underlying the entire graph.
Such a knowledge-based taxonomy provides a unified perspective to analyze the pre-training and downstream tuning strategies (Section VI ) of both GNNs/GTs and recent graph language models (Section VII ), providing valuable insights for the future directions of GFMs (Section VIII ).
Our knowledge-based perspective is also architecture-agnostic, compared to existing surveys which are applicable to only certain types of architectures.
Therefore, we provide a more systematic view covering a much wider range of graph models.
As illustrated in Fig.  2 , we analyze 9 knowledge categories and 300 references ranging from the 2010s to 2025, which are to our knowledge the most detailed categorization of self-supervised GFMs.
All papers included are summarized as tables in Appendix  A for better comparison.
We also collate more than 500 relevant papers and list them on GitHub 1 1 
 1 
 
 
 
 https://github.com/Newiz430/Pretext .
We hope this survey will help researchers exploit more powerful GFMs by exploiting graph-specific knowledge.

 
 {forest} 
 
 Fig. 2: 

Our knowledge-based taxonomy of self-supervised graph pre-training with representative literature.

 
 
 
 

## II Preliminary 

 
 This section provides basic concepts related to our topic.

 
 
 Graph. 
Graph is a data structure consisting of a node (vertex) set and an edge (link) set 𝒢 = ( 𝒱 , ℰ ) \mathcal{G}=(\mathcal{V},\mathcal{E}) . The adjacency matrix 𝐀 ∈ { 0 , 1 } n × n \mathbf{A}\in\{0,1\}^{n\times n} indicates if two nodes are connected by a link.
For an attributed graph, each node is associated with a row of the feature matrix 𝐗 ∈ ℝ n × d \mathbf{X}\in\mathbb{R}^{n\times d} .
For every node i ∈ 𝒱 i\in\mathcal{V} , its (undirected) neighborhood is 𝒩 i = { j ∈ 𝒱 | A i , j = 1 } \mathcal{N}_{i}=\{j\in\mathcal{V}|A_{i,j}=1\} .
A graph dataset can contain one graph only or multiple relatively small graphs.

 
 
 Graph model and graph foundation model (GFM). 
A graph model is an encoding function 𝐙 = f ⁡ ( 𝒢 , Θ ) \mathbf{Z}=f(\mathcal{G};\Theta) that can be parameterized by GNNs  [ 4 , 5 , 6 ] , GTs  [ 7 , 8 ] , graph language models  [ 9 , 10 , 11 ] , etc.
A GFM is an (ideal) graph model pre-trained on various kinds of unsupervised graph data to handle various types of graph-related tasks  [ 14 ] .

 
 
 Pre-training task (pretext). 
A pretext ℒ ∈ 𝒯 \mathcal{L}\in\mathcal{T} is a self-supervised task performed during the pre-training phase of a graph model, where 𝒯 \mathcal{T} represents the pretext task set. A pretext should meet two conditions: (1) during the self-supervised pre-training, no manual-labeled data is used;
(2) its goal is to achieve improved performance on one or multiple downstream tasks ℒ ˇ \check{\mathcal{L}} :

 

 
 | 
 
 
 ∑ ℒ ˇ ∈ 𝒯 ˇ min Φ , Θ ∗ ⁡ ℒ ˇ ​ ( f ˇ ⋅ f ∗ , 𝒢 ˇ , 𝒴 ˇ ) , s . t . f ∗ = ∑ ℒ ∈ 𝒯 arg ⁡ min Θ ⁡ ℒ ⁡ ( f , 𝒢 ) \displaystyle\sum_{\check{\mathcal{L}}\in\check{\mathcal{T}}}{\min_{\Phi,\Theta^{*}}\check{\mathcal{L}}(\check{f}\cdot f^{*},\check{\mathcal{G}},\check{\mathcal{Y}})},\ s.t.\ f^{*}={\sum_{\mathcal{L}\in\mathcal{T}}{\arg\min_{\Theta}\mathcal{L}(f,\mathcal{G})}} 

 | 
 | 
 (1) | 
 

 where f ˇ ​ ( 𝒢 ˇ , Φ ) \check{f}(\check{\mathcal{G}};\Phi) denotes some optional downstream branches.
“ Θ ∗ \Theta^{*} ” is optional depending on whether the pre-trained model parameters are tuned for downstream tasks.
“ 𝒴 ˇ \check{\mathcal{Y}} ” is also optional depending on whether task-specific labels are used, also known as supervised fine-tuning (SFT).

 
 
 

## III Microscopic Pre-training Tasks 

 
 Microscopic pre-training tasks treat nodes or edges as individual instances. They extract features, properties, and local relationships between these instances.

 
 

### III-A Node Features 

 
 Node features are a rich source of semantic information in attributed graphs, encoding domain-specific knowledge such as textual content in citation networks or chemical properties in molecular graphs.
The expressiveness and utility of these features heavily depend on their origin and the encoding methods employed.

 
 Feature prediction. 
Feature prediction serves as a fundamental pretext task in graph autoencoding methods like MGAE  [ 28 ] , GALA  [ 29 ] , and Graph-Bert  [ 30 ] . These methods reconstruct low-dimensional node representations and match them with the original feature size, minimizing the reconstruction error such as MSE: ℒ = 𝔼 i ∈ 𝒱 ​ [ ‖ 𝑿 i − 𝑿 ^ i ‖ 2 ] \mathcal{L}=\mathbb{E}_{i\in\mathcal{V}}[\|\boldsymbol{X}_{i}-\hat{\boldsymbol{X}}_{i}\|^{2}] ,
while GMI  [ 31 ] maximizes the mutual information between the original graph and the output representations by a discriminator network.

 
 
 Another kind of prediction task focuses on feature denoising , where noise is first added to the original features 𝐗 ~ = 𝐗 + ϵ \tilde{\mathbf{X}}=\mathbf{X}+\epsilon , and then the model is tasked with recovering the original noise-free features.
The success of masked language/image modeling  [ 130 , 131 ] has led to the rise of masked feature prediction , also known as masked autoencoding or graph completion  [ 33 ] .
Methods like AttrMask  [ 32 , 34 ] , LaGraph  [ 35 ] , and SLAPS  [ 36 ] sample a noise matrix from a Bernoulli distribution 𝐌 ∈ { 0 , 1 } n × d \mathbf{M}\in\{0,1\}^{n\times d} and obtain masked features 𝐗 ~ = 𝐌 ∘ 𝐗 \tilde{\mathbf{X}}=\mathbf{M}\circ\mathbf{X} . Then, the original features are reconstructed by an MSE loss.
GPT-GNN  [ 37 ] adopts an autoregressive masking approach, where the masked node attributes and their corresponding edges are generated one-by-one.
Recent GraphMAE series  [ 23 , 38 ] introduces a scaled cosine error with a focusing parameter λ \lambda to adjust the weight of each sample:

 

 
 | 
 
 
 ℒ = 𝔼 q ⁡ ( ( 1 − 𝐌 ) ∘ 𝐗 | 𝐗 ) ​ [ 1 − ( 𝐗 ⊤ ​ f ​ ( 𝐌 ∘ 𝐗 , 𝐀 , Θ ) ‖ 𝐗 ‖ ​ ‖ f ⁡ ( 𝐌 ∘ 𝐗 , 𝐀 , Θ ) ‖ ) λ ] \displaystyle\mathcal{L}=\mathbb{E}_{q((1-\mathbf{M})\circ\mathbf{X}|\mathbf{X})}\left[1-\left(\frac{\mathbf{X}^{\top}f(\mathbf{M}\circ\mathbf{X},\mathbf{A};\Theta)}{\|\mathbf{X}\|\|f(\mathbf{M}\circ\mathbf{X},\mathbf{A};\Theta)\|}\right)^{\lambda}\right] 

 | 
 | 
 (2) | 
 

 and this has inspired various new-generation masked autoencoder architectures  [ 39 , 40 , 132 ] . DiscoGNN  [ 41 ] first randomly replaces nodes with different ones. Then, it learns to find and reconstruct the replaced nodes.

 

 Node instance discrimination. 
Instance discrimination, also referred to as “contrastive learning”, has achieved significant success in the visual domain  [ 133 , 134 ] and subsequently becomes a fundamental and general task for graph pre-training.
Node instance discrimination aims to perform instance discrimination between node pairs. The workflow involves creating two perturbed versions (views) of an original graph 𝒢 i , 𝒢 ii \mathcal{G}^{\text{i}},\mathcal{G}^{\text{ii}} .
Nodes at the same position across views ( 𝒁 i i , 𝒁 i ii ) (\boldsymbol{Z}^{\text{i}}_{i},\boldsymbol{Z}^{\text{ii}}_{i}) form positive pairs,
while others ( 𝒁 i i , 𝒁 j ii ) (\boldsymbol{Z}^{\text{i}}_{i},\boldsymbol{Z}^{\text{ii}}_{j}) or ( 𝒁 i i , 𝒁 j i ) (\boldsymbol{Z}^{\text{i}}_{i},\boldsymbol{Z}^{\text{i}}_{j}) ) are randomly sampled as negative pairs.
The goal is to maximize the similarity between positive pairs and minimize it between negative ones, allowing for the learning of general and perturbation-invariant representations.

 
 
 The simplest form of node-level instance discrimination is to minimize the MSE between positive pairs ℒ = 𝔼 i ∈ 𝒱 ​ [ ‖ 𝒁 i i − 𝒁 i ii ‖ 2 ] \mathcal{L}=\mathbb{E}_{i\in\mathcal{V}}[\|{\boldsymbol{Z}^{\text{i}}_{i}}-\boldsymbol{Z}^{\text{ii}}_{i}\|^{2}] .
However, it can suffer from representation degeneration, i.e., the output may degenerate to a constant vector regardless of the input. This is often addressed by combining it with other tasks  [ 35 , 38 , 58 , 59 ] .
By contrast, mutual information (MI) provides a more effective criterion by capturing non-linear statistical dependence between node instances  [ 135 ] :

 

 
 | 
 I ( 𝒁 i i ; 𝒁 j ii ) = D KL [ p ( 𝒁 i i , 𝒁 i ii ) ∥ p ( 𝒁 i i ) p ( 𝒁 i ii ) ] I(\boldsymbol{Z}^{\text{i}}_{i};\boldsymbol{Z}^{\text{ii}}_{j})=D_{\text{KL}}[p(\boldsymbol{Z}^{\text{i}}_{i},\boldsymbol{Z}^{\text{ii}}_{i})\|p(\boldsymbol{Z}^{\text{i}}_{i})p(\boldsymbol{Z}^{\text{ii}}_{i})] | 
 | 
 (3) | 
 

 Calculating MI in a high-dimension space is a challenging task  [ 135 ] . Therefore, various techniques have been proposed to estimate MI. These techniques mainly include:

 
 
 (1) Jenson-Shannon (JS) estimator   [ 136 ] : it replaces the KL divergence in ( 3 ) with JS divergence and approximates the distributions usually by a discriminator network 𝒟 \mathcal{D} :

 

 
 | 
 
 
 ℒ = − 𝔼 i ∈ 𝒱 ​ [ σ + ​ ( − 𝒟 ⁡ ( 𝒁 i i , 𝒁 i ii ) ) + 𝔼 j ∈ 𝒱 i − ​ [ σ + ​ ( 𝒟 ⁡ ( 𝒁 i i , 𝒁 ⋅ ; j ) ) ] ] \displaystyle\mathcal{L}=-\mathbb{E}_{i\in\mathcal{V}}[\sigma_{+}(-\mathcal{D}(\boldsymbol{Z}^{\text{i}}_{i},\boldsymbol{Z}^{\text{ii}}_{i}))+\mathbb{E}_{j\in\mathcal{V}_{i}^{-}}[\sigma_{+}(\mathcal{D}(\boldsymbol{Z}^{\text{i}}_{i},\boldsymbol{Z}_{\cdot;j}))]] 

 | 
 | 
 (4) | 
 

 
 
 (2) InfoNCE estimator   [ 137 ] : it is based on the Noise Contrastive Estimation (NCE) loss. Formally:

 

 
 | 
 
 
 ℒ = − 𝔼 i ∈ 𝒱 ​ [ log ⁡ exp ⁡ ( ⟨ 𝒁 i i , 𝒁 i ii ⟩ ) ∑ j ≠ i exp ⁡ ( ⟨ 𝒁 i i , 𝒁 j i ⟩ ) + ∑ j = 1 n exp ⁡ ( ⟨ 𝒁 i i , 𝒁 j ii ⟩ ) ] \displaystyle\mathcal{L}\!=\!-\mathbb{E}_{i\in\mathcal{V}}\!\left[\log\!\frac{\exp(\langle\boldsymbol{Z}^{\text{i}}_{i},\!\boldsymbol{Z}^{\text{ii}}_{i}\rangle)}{\sum_{j\neq i}\!{\exp(\langle\boldsymbol{Z}^{\text{i}}_{i},\!\boldsymbol{Z}^{\text{i}}_{j}\rangle)}\!+\!\sum_{j=1}^{n}\!{\exp(\langle\boldsymbol{Z}^{\text{i}}_{i},\!\boldsymbol{Z}^{\text{ii}}_{j}\rangle)}}\right] 

 | 
 | 
 (5) | 
 

 where ⟨ ⋅ , ⋅ ⟩ \langle\cdot,\cdot\rangle is the relative similarity of two samples.
This estimator is well-known in representative models like GRACE  [ 42 ] , GCA  [ 43 ] , ProGCL  [ 44 ] , and more  [ 45 , 46 , 47 ] .

 
 
 (3) Triplet (margin) estimator   [ 138 ] : some contrastive frameworks like SUGRL  [ 50 ] employ a triplet loss to contrast between the anchor-positive pairs ( 𝐙 , 𝐙 + ) (\mathbf{Z},\mathbf{Z}^{+}) the anchor-negative pairs ( 𝐙 , 𝐙 − ) (\mathbf{Z},\mathbf{Z}^{-}) :

 

 
 | 
 ℒ = 𝔼 i ∈ 𝒱 ​ [ ⟨ 𝒁 i , 𝒁 i + ⟩ − ⟨ 𝒁 i , 𝒁 i − ⟩ + ϵ ] \mathcal{L}=\mathbb{E}_{i\in\mathcal{V}}\left[\langle\boldsymbol{Z}_{i},\boldsymbol{Z}^{+}_{i}\rangle-\langle\boldsymbol{Z}_{i},\boldsymbol{Z}^{-}_{i}\rangle+\epsilon\right] | 
 | 
 (6) | 
 

 where ϵ \epsilon denotes the distance margin, controlling the lower bound of distance between positive and negative samples.

 
 
 There are other instance discrimination objectives that have achieved competitive performance in learning node features.
For example, the bootstrapping loss  [ 139 , 48 , 49 ] generally computes the cosine similarity ℒ = − 𝔼 i ∈ 𝒱 ​ [ 𝒁 i i ​ p ​ ( 𝒁 i ii ) ⊤ / ‖ 𝒁 i i ‖ ​ ‖ p ⁡ ( 𝒁 i ii ) ‖ ] \mathcal{L}=-\mathbb{E}_{i\in\mathcal{V}}[{\boldsymbol{Z}^{\text{i}}_{i}{p(\boldsymbol{Z}^{\text{ii}}_{i})}^{\top}}/{\|\boldsymbol{Z}^{\text{i}}_{i}\|\|p(\boldsymbol{Z}^{\text{ii}}_{i})\|}] between two asymmetric and momentum-updated networks, aligned by a projector p ⁡ ( ⋅ ) p(\cdot) . Other examples include Bayesian Personalized Ranking loss (BPR)  [ 140 , 51 , 52 ] and population spectral contrastive loss  [ 141 , 55 , 56 ] .

 
 
 Fig. 3: 
An illustration of discrimination tasks between node features. The similarity is defined as the dot-product between embeddings 𝐙 \mathbf{Z} .
Node instance discrimination performs row-wise contrast, which computes the similarity between every pair of node embeddings (distinguished by the node shape: circle/triangle/square).
Dimension discrimination performs column-wise contrast, which computes the similarity between every pair of node dimensions (distinguished by the embedding color: orange/red).
 
 
 
 Node instance discrimination is one of the most popular and generalizable tasks beyond graph pre-training.
Recent literature focusing on downstream tuning has unified multiple tasks such as link prediction, node classification, and graph classification into node instance discrimination  [ 142 , 79 , 143 ] , as an impactful self-supervised tuning strategy.

 
 Dimension discrimination. 
Dimension discrimination focuses on distinguishing different dimensions of node representations.
This can be seen as a column-wise discrimination approach, as shown in Fig.  3 .
Similar to node instance discrimination, it begins with two augmented views.
Then, it maximizes the correlation between corresponding dimensions and minimizes it between different ones.
This strategy is initially proposed by Barlow Twins  [ 144 ] and then introduced to the graph domain by G-BT  [ 57 ] , which considers the similarity between dimensions of different instances. On the other hand, CCA-SSG  [ 58 ] 
focuses on the dimensional similarity within a single instance via a covariance regularization term:

 

 
 | 
 
 
 ℒ = ‖ 𝐙 i − 𝐙 ii ‖ F 2 + λ ​ ( ‖ 𝐙 i ⊤ ​ 𝐙 i − 𝐈 d ‖ F 2 + ‖ 𝐙 ii ⊤ ​ 𝐙 ii − 𝐈 d ‖ F 2 ) ⏟ dimension discrimination \displaystyle\mathcal{L}=\|\mathbf{Z}^{\text{i}}-\mathbf{Z}^{\text{ii}}\|^{2}_{F}+\lambda\underbrace{\left(\|{\mathbf{Z}^{\text{i}}}^{\top}\mathbf{Z}^{\text{i}}-\mathbf{I}_{d}\|^{2}_{F}+\|{\mathbf{Z}^{\text{ii}}}^{\top}\mathbf{Z}^{\text{ii}}-\mathbf{I}_{d}\|^{2}_{F}\right)}_{\text{dimension discrimination}} 

 | 
 | 
 (7) | 
 

 Future improvements mainly focus on the effectiveness of the regularization term, such as how to avoid the global or local representation collapse problem  [ 59 , 60 ] .

 
 Discussion. 
For feature prediction, both traditional methods and the latest masked autoencoders tend to preserve shallow features rather than capture deeper semantic information.
Despite that node instance discrimination encourages the model to focus on deeper semantic information,
it mainly focuses on node feature correlations while neglecting high-order structural knowledge, potentially leading to suboptimal performance in structural tasks like link prediction.
Compared to discrimination between instances, dimension discrimination enables augmentation-invariant feature learning by clarifying node dimensions. However, as noted in  [ 58 ] , its benefits may diminish with smaller representation dimensions.

 
 
 Efficiency analysis. 
The computational efficiency of node feature-based tasks primarily depends on the size of the feature matrix 𝐗 ∈ ℝ n × d \mathbf{X}\in\mathbb{R}^{n\times d} and the model architecture.
The time complexity for the most common InfoNCE-based node instance discrimination is 𝒪 ⁡ ( n − ​ n ​ d ) \mathcal{O}(n^{-}nd) , where n − n^{-} is the number of negative samples for each node. The huge demand for negative samples results in a prohibitive worst-case cost 𝒪 ⁡ ( n 2 ​ d ) \mathcal{O}(n^{2}d) .
Similarly, the time complexity for the dimension discrimination term in ( 7 ) is 𝒪 ⁡ ( n ​ d 2 ) \mathcal{O}(nd^{2}) .
In contrast, feature prediction and bootstrapping-based methods scale linearly with both n n and d d ( 𝒪 ⁡ ( n ​ d ) \mathcal{O}(nd) ), making them more advantageous in terms of scalability. The lightweight linear decoder adopted by new-generation masked autoencoders can further reduce the computational cost  [ 131 ] .

 
 
 

### III-B Node Properties 

 
 While Section  III-A discusses pretext tasks that focus on node semantic information, another class of tasks focuses on node properties, which are crucial for understanding their structural roles within the graph.

 
 
 The centrality is a set of node properties that quantify the relative importance between nodes based on local structure.
For example, the degree of a node deg ​ ( ⋅ ) \text{deg}(\cdot) is a common measure of local connectivity, defined as the number of edges incident to that node.
There are various kinds of centrality measures including degree,
closeness, betweenness, eigenvector, and PageRank  [ 145 ] . Predicting centralities is one of the most direct methods for capturing the structural importance of nodes.
Autoencoding methods such as NWR-GAE  [ 62 ] and MaskGAE  [ 63 ] employ an MSE loss to predict the node degree: ℒ = 𝔼 i ∈ 𝒱 ​ [ ‖ deg ​ ( i ) − deg ^ ​ ( i ) ‖ 2 ] \mathcal{L}=\mathbb{E}_{i\in\mathcal{V}}[\|\text{deg}(i)-\hat{\text{deg}}(i)\|^{2}] .
Besides degree prediction, CenPre  [ 65 ] leverages the left singular vector of 𝐀 \mathbf{A} as structural representations, similar to the eigenvector centrality, and computes its similarity with the original features.
Hu et al.   [ 61 ] propose centrality ranking , which predicts if a node has a higher or lower centrality score s s compared to another node.

 
 
 There are also other node properties that capture structural features beyond relative importance.
For instance, (local) clustering coefficient   [ 146 ] measures the gathering tendency of node groups,
and predicting the clustering coefficient highlights the local relationship between nodes and guides the model to preserve them  [ 34 ] .
 Node order is also a special kind of property that holds the permutation-invariant information for more expressive GNNs.
PIGAE  [ 64 ] aligns the order of output node representations with the input node order by adding a learnable permuter into a VGAE  [ 12 ] .

 
 Discussion. 
Node properties help models comprehend node information by encoding structural roles such as relative importance.
However, node properties tend to be more task-specific  [ 34 ] , meaning that they can be difficult to leverage as generalizable knowledge in certain scenarios.
Additionally, node properties may not always provide sufficient discriminative power. Graphs with different topologies can have the same degree distribution, making it difficult to distinguish between them solely by the degree.

 
 
 Efficiency analysis. 
Calculating different node properties necessitates varying degrees of additional computational costs.
For example, the complexity for calculating the degree of all nodes is 𝒪 ⁡ ( | ℰ | ) \mathcal{O}(|\mathcal{E}|) , while the complexity for calculating the eigenvector centrality (via eigenvalue decomposition) is 𝒪 ⁡ ( n 3 ) \mathcal{O}(n^{3}) .
Considering that most of these properties are scalars, the memory required to store them is usually 𝒪 ⁡ ( n ) \mathcal{O}(n) .

 
 
 

### III-C Links 

 
 From chemical bonds in molecular graphs and social connections in social networks to semantic relationships in knowledge graphs,
links play a fundamental role in graphs as they represent basic relationships between nodes.

 
 
 Link prediction is a fundamental task in graph-based SSL aiming to predict the existence or probability of a link between two nodes.
Structure-based autoencoders such as GAE  [ 12 ] feed the learned node representations into a dot-product decoder 𝐀 ^ = σ ⁡ ( 𝐙𝐙 ⊤ ) \hat{\mathbf{A}}=\sigma(\mathbf{Z}\mathbf{Z}^{\top}) to predict the existence probability between a pair of nodes:

 

 
 | 
 ℒ = − 𝔼 ( i , j ) ∈ ℰ ​ [ log ⁡ A ^ i , j ] + 𝔼 ( i , j ) ∈ ℰ − ​ [ log ⁡ ( 1 − A ^ i , j ) ] \mathcal{L}=-\mathbb{E}_{(i,j)\in\mathcal{E}}[\log\hat{A}_{i,j}]+\mathbb{E}_{(i,j)\in\mathcal{E}^{-}}[\log(1-\hat{A}_{i,j})] | 
 | 
 (8) | 
 

 To capture more complicated latent spaces,
variational autoencoders such as VGAE  [ 12 ] and [ 66 , 67 , 74 ] learn a Gaussian model for latent embeddings q ⁡ ( 𝐙 | μ , σ ) = 𝒩 ⁡ ( μ , σ 2 ) q(\mathbf{Z}|\mu,\sigma)=\mathcal{N}(\mu,\sigma^{2}) 
to approximate the real posterior p ⁡ ( 𝐙 | 𝐗 , 𝐀 ) p(\mathbf{Z|\mathbf{X},\mathbf{A}}) .
Representation vectors are then sampled from these distributions to maximize the expected log-likelihood log ⁡ p ⁡ ( 𝐀 ) \log p(\mathbf{A}) bounded by the evidence lower bound (ELBO):

 

 
 | 
 𝒥 = 𝔼 q ⁡ ( 𝐙 | μ , σ ) [ log p ( 𝐀 | 𝐙 ) ] − D KL [ q ( 𝐙 | μ , σ ) ∥ p ( 𝐙 ) ] \displaystyle\mathcal{J}=\mathbb{E}_{q(\mathbf{Z}|\mu,\sigma)}[\log p(\mathbf{A}|\mathbf{Z})]-D_{\text{KL}}[q(\mathbf{Z}|\mu,\sigma)\|p(\mathbf{Z})] | 
 | 
 (9) | 
 

 where p ⁡ ( 𝐙 ) = 𝒩 ⁡ ( 0 , 𝐈 ) p(\mathbf{Z})=\mathcal{N}(0,\mathbf{I}) is the preset Gaussian prior.
Link prediction usually serves as an auxiliary task for training semi-supervised GNNs  [ 72 ] , discimination models  [ 71 , 93 , 94 ] , and feature-based autoencoders  [ 91 , 73 ] . Instead of a learnable decoder, discrimination losses such as InfoNCE are also applicable to link prediction  [ 8 , 147 , 148 ] .

 
 
 Another kind of prediction task, link denoising , involves adding random noises to the adjacency matrix.
For example, Bandana  [ 70 ] samples continuous edge noises and predicts the noise values.
A more widely adopted approach for link denoising is
 masked link prediction , where a portion p p of edges is randomly masked using binary noise M i , j ∼ B ​ e ​ r ​ n ​ o ​ u ​ l ​ l ​ i ​ ( 1 − p ) M_{i,j}\sim Bernoulli(1-p) .
The objective is similar to that of binary link prediction,
with the key difference being that only the masked edges are treated as positive samples during training.
EdgeMask  [ 34 ] and S2GAE  [ 69 ] learn a decoder 𝐀 ^ = σ ⁡ ( g ⁡ ( 𝐙𝐙 ⊤ , Ψ ) ) \hat{\mathbf{A}}=\sigma(g(\mathbf{Z}\mathbf{Z}^{\top};\Psi)) to recover the masked edges.
MaskGAE  [ 63 ] captures long-range relationships by randomly masking out paths obtained from random walks.

 
 
 Edge features in some attributed graphs also provide rich semantics complementing the graph structure, such as the number of co-authored papers or research topics in a co-authorship graph.
Methods for node feature learning, such as auto-encoding in PIGAE  [ 64 ] and ASD-VAE  [ 73 ] , as well as masked feature prediction in AttrMask  [ 32 ] , can be effortlessly applied to edge feature prediction .

 
 Discussion. 
Link prediction has brought significant benefits to structure-based downstream tasks by capturing the structural information of graphs.
It explicitly models the relationships between nodes that are not considered in node feature-based methods.
Despite its widespread use, link prediction has been criticized for overemphasizing local structure  [ 13 , 23 ] . This highlights the need for exploring more compatible link-based pre-training strategies with feature semantics and higher-order structures.

 
 
 Efficiency analysis. 
The memory required for dense and sparse adjacency matrices is 𝒪 ⁡ ( n 2 ) \mathcal{O}(n^{2}) and 𝒪 ⁡ ( | ℰ | ) \mathcal{O}(|\mathcal{E}|) , respectively.
Link prediction involves calculating the node similarity, such as the dot product and cosine similarity, both of which have a time complexity of 𝒪 ⁡ ( d ​ | ℰ | ) \mathcal{O}(d|\mathcal{E}|) .
As the efficiency of link prediction models is primarily bottlenecked by the edge size, pruning methods and masking link prediction can significantly reduce the memory cost for encoding.

 
 
 
 

## IV Mesoscopic Pre-training Tasks 

 
 In contrast to microscopic tasks that focus on individual nodes and links, mesoscopic pre-training tasks aim to capture properties within a local range. These pretexts learn representations that encode higher-order information and long-range dependencies.

 
 

### IV-A Context 

 
 Graph context refers to the neighborhood or a broader subgraph surrounding a node.
Most context-based methods leverage the homophily assumption   [ 149 ] , where adjacent nodes tend to share similar attributes.

 
 
 Context discrimination is a pretext that can be traced back to network embedding algorithms,
e.g., DeepWalk  [ 150 ] . They sample node sequences from the graph using random walks and then iteratively update their embeddings using text embedding methods.
GraphSAGE  [ 6 ] redefines “context” from random walk sequences to the neighborhood subgraphs induced from the graph.
It optimizes a negative sampling-based JS estimator loss ( 4 ) between the central node i i and its contextual nodes j ∈ 𝒩 i j\in\mathcal{N}_{i} :

 

 
 | 
 
 
 ℒ = − 𝔼 i ∈ 𝒱 j ∈ 𝒩 i ​ [ log ⁡ σ ⁡ ( 𝒁 i ⊤ ​ 𝒁 j ) + λ ​ ∑ k ∈ 𝒱 − log ⁡ σ ⁡ ( − 𝒁 i ⊤ ​ 𝒁 k ) ] \displaystyle\mathcal{L}=-\mathbb{E}_{\begin{subarray}{c}i\in\mathcal{V}\\
j\in\mathcal{N}_{i}\end{subarray}}\big[\!\log\sigma(\boldsymbol{Z}_{i}^{\top}\boldsymbol{Z}_{j})+\lambda\sum_{k\in\mathcal{V}^{-}}{\!\log\sigma(-\boldsymbol{Z}_{i}^{\top}\boldsymbol{Z}_{k})}\big] 

 | 
 | 
 (10) | 
 

 Later efforts  [ 75 , 76 ] improve GraphSAGE with a discriminator network to determine whether one node is the neighbor of another node.
COLES  [ 77 ] captures neighborhood similarity by equipping Laplacian Eigenmaps  [ 151 ] with negative sampling,
which is further generalized by GLEN  [ 78 ] as a rank optimization problem of representation scatter matrices.

 
 
 For other MI estimators,
Graph-MLP  [ 82 ] and
N2N  [ 83 ] define their positive sample pairs as every node and its k k -hop neighborhood, and employ an InfoNCE loss.
Subg-Con  [ 84 ] selects k k -nearest neighbors by personalized PageRank scores as positive samples of a triplet loss.
AFGRL  [ 85 ] selects k k -nearest neighbors in the context of both structure and feature as positive samples of a bootstrapping loss.
HGRL  [ 86 ] further leverages the homophily assumption by selecting homophilic neighbors as precise positive samples. BSG  [ 87 ] uses mean pooling to obtain neighborhood embeddings 𝐙 𝒩 \mathbf{Z}_{\mathcal{N}} and maximizes the mutual information I ⁡ ( 𝐙 , 𝐙 𝒩 ) I(\mathbf{Z},\mathbf{Z}_{\mathcal{N}}) and the conditional entropy H ⁡ ( 𝐙 | 𝐙 𝒩 ) H(\mathbf{Z}|\mathbf{Z}_{\mathcal{N}}) through MSE and hinge loss, respectively.

 
 
 Fig. 4: 
An illustration of the contextual knowledge. For the central node   of a 2-hop subgraph (Left),
context discrimination often takes its neighboring nodes  
  or   as positive samples and other nodes  
  as negative samples. Contextual subgraph discrimination samples multiple contextual subgraphs (Right) as positive pairs, while negative ones are sampled from other subgraphs.
 
 
 
 Another task, contextual subgraph discrimination , measures the similarity between two different sampled subgraphs, as shown in Fig.  4 .
ContextPred  [ 32 ] samples a “context graph” from the periphery of the k k -hop subgraph and matches them as a positive pair.
Instead of sampling an additional context graph,
GCC  [ 80 ] directly induces two different subgraphs from the k k -hop neighborhood of each node as a positive pair.
S 3 -CL  [ 81 ] contrasts between intermediate message-passing layers to aggregate neighborhoods of varying scales.

 
 Discussion. 
Compared to individual links, treating node context as instances facilitates a more complete understanding of the local graph structure.
Nonetheless, it is empirically verified that some context learning methods have limited contributions to the performance of message-passing GNNs  [ 34 ] , owing to the inherent capability of message-passing to extract local structural information.
Context learning has the potential to benefit models that put more emphasis on global interactions, e.g., graph Transformers.

 
 
 Efficiency analysis. 
Using graph traversal algorithms, one can obtain the k k -hop subgraph of any node with a time complexity of O ⁡ ( n + | ℰ | ) O(n+|\mathcal{E}|) . A more common approach is to aggregate the embeddings of neighboring nodes by left-multiplying 𝐀 \mathbf{A} , which has a time complexity of O ⁡ ( k ​ | ℰ | ) O(k|\mathcal{E}|) . Since the complexity is independent of n n , this approach is particularly suitable for handling sparse networks with a large number of nodes. Existing GFM researchers  [ 8 , 152 , 153 , 154 ] often reformulate node classification on large networks as predicting subgraph labels around the target node, which is essentially a divide-and-conquer strategy.

 
 
 

### IV-B Long-range Similarities 

 
 Long-range similarities capture relationships between non-neighboring nodes that share semantic relevance. These similarities reveal higher-order dependencies beyond local neighborhoods.
For example, in social networks, the small-world property implies that any two individuals are likely connected through a short chain of acquaintances  [ 146 ] .

 
 Similarity prediction. 
Similarity prediction aims to capture long-range similarities between nodes by directly predicting the similarity matrix 𝐒 ∈ ℝ n × n \mathbf{S}\in\mathbb{R}^{n\times n} .
Depending on whether two nodes are connected by a path, long-range similarities can be divided into topologically accessible similarities and topologically inaccessible ones.
For the former, the shortest path distance measures the minimal distance between two connected nodes, and the Katz index   [ 155 ] measures the total number of paths of every length between two connected nodes.
S 2 GRL  [ 88 ] and PairwiseDistance  [ 34 ] train the graph model to predict these similarities between all pairs of nodes by a negative log-likelihood loss.
For the latter, the PageRank similarity   [ 145 ] quantifies the importance of node pairs in terms of graph structure, while the Jaccard’s coefficient   [ 156 ] measures the overlap between node neighborhoods.
There are also feature-based measures that quantify the degree of similarity between two nodes’ features, regardless of their connectivity, such as Euclidean distance and cosine similarity  [ 89 , 34 , 157 ] . While Graph-Bert  [ 30 ] directly predicts them by a regressive loss,
AGE  [ 89 ] and PairwiseAttrSim  [ 34 ] adopt similarity-based discrimination that selects a subset of node pairs with the highest (resp. lowest) similarity scores and uses them as positive (resp. negative) samples. These similarities can bridge the disconnected components in the graph data which message-passing cannot.

 

 Similarity graph alignment. 
Similarity graphs, derived from the original graph based on the pairwise similarities between nodes, usually serve as an alternative structural view of the graph.
For instance, the kNN graph reconstructs the edge set by connecting the k k -nearest neighbors of each node through various feature-based measures.
It shares both commonalities and differences with the original graph structure (and other similarity graphs); therefore, aligning their semantics becomes a principled approach to combine feature semantics and graph structure.
AM-GCN  [ 90 ] and DLR-GAE  [ 91 ] minimize the discrepancy between original and similarity graph representations by MSE and cross-entropy.
Instance discrimination methods  [ 92 , 93 , 94 ] treat the original and similarity graphs as two views to integrate complementary information from both views.

 
 
 Discussion. 
Long-range similarities play a crucial role in capturing the dependencies between nodes out of reach for local contexts.
It also enables the model to handle sparse graphs or graphs with disconnected components.
However, the discrepancy between feature similarity and structural similarity can lead to semantic conflicts.
Nodes with similar features may not always have similar structural neighborhoods.
Therefore, the choice of similarity measures should depend on the real-world requirements.

 
 
 Efficiency analysis. 
Computing all-pairs shortest paths can be computationally expensive for large graphs.
The running time of the classic Floyd-Warshall algorithm is 𝒪 ⁡ ( n 3 ) \mathcal{O}(n^{3}) and the memory required is 𝒪 ⁡ ( n 2 ) \mathcal{O}(n^{2}) . Using Dijkstra’s algorithm for all nodes results in a complexity of 𝒪 ⁡ ( n ​ | ℰ | ​ log ⁡ n ) \mathcal{O}(n|\mathcal{E}|\log n) .
By contrast, calculating feature-based similarities requires 𝒪 ⁡ ( n 2 ​ d ) \mathcal{O}(n^{2}d) time, which is more efficient for dense networks.

 
 
 

### IV-C Motifs 

 
 Motifs are small subgraphs that frequently appear and carry significant structural and functional information, such as functional groups in molecular graphs, coregulators in regulatory networks, and cliques of people in social networks.

 
 
 Motif prediction tasks aim to learn motif-level representations by predicting the motif pseudo-labels of subgraphs. These pseudo-labels are given by unsupervised motif discovery algorithms, e.g., RDKit  [ 158 ] .
GROVER  [ 7 ] assigns motif pseudo-labels to molecular graphs and trains a GNN for classification, and MoAMa  [ 97 ] extends this idea by conducting motif-wise feature masking and prediction.
DGPM  [ 98 ] performs a binary node-motif matching task to predict if a node belongs to a motif.
Recent literature introduces the concept of “fragment graphs”, whose nodes are aggregated from subgraphs containing specific motifs, shown in Fig.  5 .
The aggregated supernode representations are collected in a motif dictionary. In this way, motif prediction is transformed into a lookup task: the representation vector of each node is associated with an entry in the motif dictionary.
MGSSL  [ 95 ] proposed an autoregressive method to generate and classify the supernodes sequentially, while GraphFP  [ 96 ] performs multi-label classification on the entire graph.

 
 
 Another line of work employs motif-based discrimination that creates contrastive sample pairs for motif-aware representations.
MotifRGC  [ 99 ] designs an adversarial motif generator to generate positive and negative views.
Fragment graphs can also serve as contrastive views:
MICRO-Graph  [ 100 ] and GraphFP  [ 96 ] treat the original graph and its corresponding fragment graph as a positive pair.

 
 
 Fig. 5: 
A molecular graph is converted into a fragment graph by aggregating functional groups into supernodes. For motif prediction, each supernode embedding is matched with a prototype in a motif dictionary.
 
 
 
 Discussion. 
Most motif-based pretexts are designed specifically for molecular graphs, limiting their applicability to larger-scale networks.
The only exception as far as we know is CTAug  [ 101 ] , a contrastive method aiming to preserve cohesive motifs (k-cliques, k-cores, etc.) in social networks.
Future research should focus on developing more general motif learning methods
to reduce the size of motif dictionaries while preserving essential structural and functional information.

 
 
 Efficiency analysis. 
The efficiency of motif learning depends on the algorithm for obtaining motif pseudo-labels. For example, the IFG algorithm  [ 159 ] called by RDKit has a time complexity 𝒪 ⁡ ( n + | ℰ | ) \mathcal{O}(n+|\mathcal{E}|) for every molecule.
Fragmentation methods traverse all nodes within a graph and map them to supernodes with extra 𝒪 ⁡ ( n ) \mathcal{O}(n) time. However, it is worth noting that the diverse range of motifs can lead to large motif dictionaries, incurring extra memory overhead.

 
 
 

### IV-D Clusters 

 
 Cluster-based tasks aim to learn representations that capture the inherent clustering structure of the graph, which can be defined based on either feature similarities or link connectivity.
Clusters often possess a larger scale compared to motifs, providing a higher-level view of the graph structure.

 
 Node clustering. 
Node clustering, a classic unsupervised learning task, is introduced as a pretext task by M3S  [ 26 ] and NodeCluster  [ 33 ] .
They leverage feature-based clustering algorithms (e.g., K K -means  [ 160 ] , DeepCluster  [ 161 ] ) to assign a cluster pseudo-label to every node.
HomoGCL  [ 103 ] and MGSE  [ 104 ] first generate a prototype vector for every cluster by feature aggregation.
Then, they optimize an MSE and a cross-entropy-based divergence loss of the cluster assignment probabilities, respectively.
CARL-G  [ 105 ] predicts “cluster validation indices”, a set of measures indicating the compactness and separation of clusters.
CommDGI  [ 27 ] , S 3 -CL  [ 81 ] and more  [ 103 , 106 ] 
focus on cluster-based discrimination where node embeddings are contrasted with learnable cluster prototypes.
GraphLoG  [ 102 ] models the hierarchical nature of clustering by setting prototypes at different levels and organizing them in a tree structure.

 
 Graph partitioning. 
Graph partitioning is also known as “non-overlapping community detection” in the scenario of social network mining. Unlike node clustering, graph partitioning is based on structural community patterns and thus is available to unattributed graphs, illustrated in Fig.  6 .
Early works  [ 61 , 33 ] leverage unsupervised community detection methods, such as spectral clustering and Louvain  [ 162 ] , to generate partition pseudo-labels and learn a community indicator matrix. Distance2Clusters  [ 34 ] performs a regression task between node representations and community prototypes.
Several works have incorporated graph partitioning into more complex frameworks, e.g., link prediction VGAE  [ 107 ] and masked autoencoders  [ 108 ] .
 Partition-based discrimination has also gained attention in recent works.
gCooL  [ 109 ] enlarges the positive set by intra-community instances between two views, while CSGCL  [ 110 ] uses the modularity-based community strength to weight node samples. StructComp  [ 111 ] takes a different approach by compressing features of nodes in the same community and performs community-wise contrast with compressed features.

 
 
 Fig. 6: 
Node clustering and graph partitioning. The former clusters nodes mainly by feature similarity on 𝐗 \mathbf{X} , marked by node shapes and colors. The latter clusters nodes mainly by connection density on 𝐀 \mathbf{A} , marked by edge colors.
 
 
 
 Discussion. 
Cluster-based pretexts provide graph models with a deeper understanding of the higher-level structural organization.
However, most cluster-based pretexts rely on non-overlapping algorithms, assuming that each node belongs to a single cluster. In real-world scenarios, nodes often belong to multiple overlapping communities, which remains a challenge for existing graph models.

 
 
 Efficiency analysis. 
The computational cost of clustering/partitioning algorithms can become prohibitive for large networks.
Common implementations of K K -means have the time complexity of 𝒪 ⁡ ( K ​ n ​ d ) \mathcal{O}(Knd) for every iteration.
For graph partitioning, vanilla spectral clustering reaches 𝒪 ⁡ ( n 3 ) \mathcal{O}(n^{3}) time complexity, while the modularity-based algorithms such as Louvain  [ 162 ] are believed to run in 𝒪 ⁡ ( | ℰ | ) \mathcal{O}(|\mathcal{E}|) time.
In practice, however, they still spend a lot of time processing large-scale networks, so clustering methods need to be considered carefully.

 
 
 
 

## V Macroscopic Pre-training Tasks 

 
 Unlike mesoscopic tasks that focus on the local graph structure, macroscopic pre-training tasks aim to capture global patterns and structures that span the entire graph. These pretexts are designed for a broader understanding of the overall organization and dynamics of the graph.

 
 

### V-A Global Structure 

 
 The goal of global structure-based tasks is to capture the overall topology and properties of a graph by learning from its global representations.

 
 Graph instance discrimination. 
This task learns to distinguish between graph instances by focusing on graph-level representations. They are obtained by aggregating node embeddings by a simple readout function, such as mean pooling and summation.
GraphCL  [ 112 ] matches positive and negative sample pairs from batches of small graphs with an InfoNCE estimator ( 5 ),
similar to node instance discrimination.
Other MI estimators are also suitable for graph instances, such as triplet loss  [ 40 ] and bootstrapping loss  [ 115 ] . Subsequent works have explored various aspects of graph instance discrimination,
such as adaptive augmentations  [ 113 , 114 ] and negative sample mining  [ 121 ] .

 
 
 Graph representations can also be used to perform node-graph discrimination , also known as cross-scale contrast  [ 3 ] .
Here the global representation can form positive pairs with every node in the graph, and
negative pairs are generated by applying one-sided perturbations to either the node or the graph representation.
The JS estimator ( 4 ) is in widespread use here, pioneered by DGI  [ 13 ] and InfoGraph  [ 116 ] .
MVGRL  [ 117 ] performs cross-view contrast by both node-level and graph-level perturbations.
GGD  [ 119 ] and D-SLA  [ 120 ] propose group discrimination , a simplified binary classification approach predicting whether an instance belongs to the original or the perturbed view.
This simplification greatly improves the efficiency, as calculating similarities between graph instances is no longer needed.
Node-graph discrimination captures the relationships between global and local representations of a graph, making it applicable to both small and large graphs  [ 76 , 122 ] .

 
 Graph similarity prediction. 
This pretext task leverages various kinds of graph-level similarity functions to learn graph-level representations, first envisioned by  [ 32 ] .
KernelPred  [ 123 ] predicts various graph kernels, including the graphlet kernel, random walk kernel, WL subtree kernel, etc. These kernels capture different aspects of graph similarity, such as structural similarity, node proximity, and subgraph patterns.
D-SLA  [ 120 ] generates a perturbed graph by adding and removing edges and predicts the graph edit distance kernel,
the number of edge modifying steps between the original and perturbed graphs.
HTML  [ 124 ] predicts the isomorphic similarity between graphs based on the Jaccard coefficient.

 
 Discussion. 
Global structure-based tasks offer a holistic view of the entire graph, capturing its overall topology and properties.
This is particularly advantageous when dealing with small graphs or scenarios focusing on global properties, including tasks such as graph classification and graph regression.
However, the readout functions used to generate global representations can be coarse-grained, potentially losing important structural information.
Moreover, graph perturbations can have a significant impact on the global semantics of small graphs.

 
 
 Efficiency analysis. 
Although graph instance discrimination methods define readout functions on the entire graph, their simple formulations imply acceptable computational costs (usually less than 𝒪 ⁡ ( n ) \mathcal{O}(n) ).
However, calculating graph kernels on large networks is not an easy task, which makes prediction tasks more constrained by the data size.

 
 
 

### V-B Manifolds 

 
 Manifolds are underlying global topological patterns that Euclidean spaces struggle to represent. Recent work explores embedding graphs into non-Euclidean manifolds, such as hyperbolic or spherical spaces, to better model hierarchical and tree-like structures of graph data.

 
 
 Cross-manifold discrimination creates contrastive views in different manifolds, thereby capturing the unique properties of each manifold and their relationships.
HGCL  [ 125 ] uses a pair of hyperbolic GNNs to encode views of the graph, and DSGC  [ 24 ] uses both Euclidean and hyperbolic GNNs to obtain views in both spaces.
RiemannGFM  [ 129 ] contrasts between a hyperbolic and a spherical space.
SelfMGNN  [ 126 ] embeds graphs into a product space that combines Euclidean, hyperbolic, and spherical spaces. It enables adaptive learning of the most suitable manifold for each graph based on its structural properties.

For prediction tasks,
HDM-GAE  [ 128 ] performs masked feature/link prediction in the hyperbolic space.
Graph-JEPA  [ 127 ] first expresses graph representations as angle vectors in a unit hyperbola and predicts them by a smooth- ℓ 1 \ell_{1} loss.

 
 
 Discussion. 
Manifold-based tasks offer a promising new direction by capturing complex geometric structures and hierarchical relationships that are difficult to represent in Euclidean spaces.
There is ample room for exploration, such as investigating more general and flexible approaches for graph manifold learning and exploring the integration of different manifolds other than a product space.

 
 
 Efficiency analysis. 
The time overhead can arise from the switching between different topological spaces.
For example, the exponential and logarithmic maps in hyperbolic spaces require multiple calculations of vector norms and inverse trigonometric functions. Although these operations do not alter the upper bound of the algorithm complexity, they introduce additional computations that increase the actual wall-clock time.
Furthermore, storing representations in different spaces incurs additional memory costs.

 
 
 
 

## VI Downstream Tuning 

 
 Fig. 7: 
Different downstream tuning strategies.
 : tuned; : frozen.

(a) Graph fine-tuning, where the pre-trained parameters are updated along with downstream branches through the downstream task ℒ ˇ \check{\mathcal{L}} .
While full fine-tuning updates the entire set of pre-trained parameters, parameter-efficient fine-tuning only updates a small amount of it via specifically designed adapter modules.
(b) Graph prompting, where the specifically designed graph prompts are updated, usually carrying downstream task-specific information. Tunable downstream branches are not necessary.

 
 
 {forest} 
 Fig. 8: 

Our taxonomy of downstream tuning strategies with representative literature.

 
 
 
 Downstream tuning in self-supervised graph models focuses on transferring the knowledge learned from self-supervised pretexts to downstream tasks, as formalized in ( 1 ). This section explores two main approaches: graph fine-tuning and graph prompting, illustrated in Fig.  7 and 8 . These approaches offer different ways to leverage the pre-trained graph model for specific applications.

 
 

### VI-A Graph Fine-tuning 

 
 Fine-tuning adapts pre-trained models to downstream tasks by jointly training them with a generally simple task-specific branch.
Traditional GNN fine-tuning methods, known as full fine-tuning , update all pre-trained parameters.
To improve knowledge transfer, advanced techniques have emerged. For instance, L2P-GNN  [ 163 ] uses a meta-learning framework that divides the pre-training data into support and query sets, simulating the adaptation process during pre-training.
S2PGNN  [ 164 ] decomposes fine-tuning into multiple function modules and dynamically identifies the optimal modules for different downstream tasks.
W2PGNN  [ 165 ] and G-Tuning  [ 166 ] focus on cross-domain transferability by representing fine-tuning as finding a combination of graphon bases. They serve as different dimensions of fundamental transferable patterns across data spaces.
GraphControl  [ 167 ] incorporates a conditional control module to utilize downstream task-specific features effectively.
GFT  [ 168 ] rearranges the input graph data as trees with a virtual root node encoding task-specific information. In this way, downstream task-relevant nodes serve as children of the root node, and their information can be aggregated for various prediction tasks.

 
 
 A specific line of work quantifies the generalization gap by the task similarity between pre-training ℒ \mathcal{L} and downstream tasks ℒ ˇ \check{\mathcal{L}} .
GTOT-Tuning  [ 170 ] models graph fine-tuning as an optimal transport problem and minimizes the masked Wasserstein distance between tasks. AUX-TS  [ 169 ] introduces gradient similarity s ​ i ​ m ​ ( ℒ , ℒ ˇ ) = ⟨ ∇ Θ ℒ , ∇ Θ ℒ ˇ ⟩ sim(\mathcal{L},\check{\mathcal{L}})=\langle\nabla_{\Theta}\mathcal{L},\nabla_{\Theta}\check{\mathcal{L}}\rangle which measures the similarity of loss surfaces between two tasks. If the similarity is positive, it indicates that the optimization directions during the gradient descent are non-conflicting, so two tasks are similar; and vice versa.
Bridge-Tune  [ 171 ] defines representation consistency, the similarity between pairwise node label distributions. A binary label is assigned to each pair of nodes determined by whether their pretext pseudo-labels (or downstream labels) are the same.

 
 
 Fine-tuning large-scale models can be computationally expensive, and biases from downstream tasks may compromise generalizability. To address these challenges, recent works adopt parameter-efficient fine-tuning (PEFT) modules such as adapters  [ 195 , 196 ] , which enable models to update only a small subset of pre-trained parameters during fine-tuning.
AdapterGNN  [ 172 ] and G-Adapter  [ 173 ] introduce adapter modules tailored for GNNs and graph Transformers, respectively.
DAGPrompT  [ 176 ] proposes the Graph Low-Rank Adaptation (GLoRA) module. Different from LoRA  [ 196 ] , both the weights 𝐖 \mathbf{W} and the adjacency matrix 𝐀 \mathbf{A} in GLoRA are decomposed into two low-rank projection matrices 𝐏 , 𝐐 ∈ ℝ d × r \mathbf{P},\mathbf{Q}\in\mathbb{R}^{d\times r} and 𝐏 𝐀 , 𝐐 𝐀 ∈ ℝ n × 1 \mathbf{P}_{\mathbf{A}},\mathbf{Q}_{\mathbf{A}}\in\mathbb{R}^{n\times 1} .

 

 Discussion. Considering the gap between general graph knowledge and domain-specific downstream knowledge, pre-training and fine-tuning are currently indispensable for building a general graph model.
However, graph fine-tuning methods often resort to specific designs in terms of tuning processes and model architectures, limiting their universality across different downstream scenarios.
Moreover, fine-tuning the pre-trained parameters may harm the generalization and expressive power of the pre-trained model.
Despite that PEFT methods enable precise fine-tuning with minimal resource requirements, they are less common in fine-tuning pure GNNs as they are relatively small in size.

 
 
 

### VI-B Graph Prompting 

 
 Prompting is an emerging downstream tuning strategy that has gained popularity with the rise of LLMs.
In the graph domain, prompting jointly encodes downstream graph data and corresponding task-specific information as additional learnable components called “prompts”. During downstream training, only the learnable part of the prompts is updated, while the pre-trained model remains frozen.
To this end, graph prompts should be first integrated into the downstream data before downstream training by various means (addition  [ 177 ] , element-wise multiplication  [ 142 ] , concatenation  [ 181 ] , weighted aggregation  [ 153 ] , linear transformation  [ 142 ] , etc.), depending on the form of prompts and specific downstream requirements.
However, unlike prompts in natural language that follow a deterministic form, graph prompts can take various shapes, increasing the difficulty of prompt design.
The following will discuss several graph prompt designs from a knowledge-based perspective.

 
 Node feature prompts . Node feature prompts are learnable vectors 𝒑 ∈ ℝ d \boldsymbol{p}\in\mathbb{R}^{d} that share the same size with the node features. Node feature prompts are simple, effective, and generalizable, serving as an important cornerstone of graph prompting. The fundamentality of node features enables their generalization across different data domains and downstream tasks, such as node classification, link prediction, and graph classification.
GPF  [ 177 ] simply adds the node feature prompt 𝒑 ∈ ℝ d \boldsymbol{p}\in\mathbb{R}^{d} to every row of the downstream feature matrix, formally 𝐗 ˇ ← [ 𝑿 ˇ i + 𝒑 ] i ∈ 𝒱 \check{\mathbf{X}}\leftarrow[\check{\boldsymbol{X}}_{i}+\boldsymbol{p}]_{i\in\mathcal{V}} to perform supervised fine-tuning.
GPF uses a universal feature prompt for every node, while its variant GPF-plus  [ 177 ] assigns an independent prompt 𝒑 i \boldsymbol{p}_{i} to each node generated by a set of learnable prompt bases.
IA-GPL  [ 178 ] generates a feature prompt for every input node from its representation through a vector quantization-based network.
Aside from feature prompts, DeepGPT  [ 179 ] prepends prefix prompts to every embedding fed into a pre-trained graph Transformer.

 
 Class type prompts .
Class prompts are prototype vectors aggregated from pre-trained node representations that share the same class: 𝒑 c = a ​ g ​ g ​ ( 𝒁 i | i ∈ 𝒱 , y i = c ) \boldsymbol{p}_{c}=agg(\boldsymbol{Z}_{i}|i\in\mathcal{V},y_{i}=c) . They are associated with downstream node classes.
Downstream tasks such as node classification can be achieved by matching node representations with these class prompts with cross-entropy  [ 68 , 181 ] or InfoNCE ( 5 )  [ 180 , 182 ] . The selection of objectives often depends on the form of pretexts.

 
 
 GPPT  [ 68 ] first divides the original graph into different clusters using node clustering algorithms (Section  IV-D ), and then defines a set of independent class prompts within each cluster.
Unlike GPPT which constructs independent node-class prompt pairs, PSP  [ 180 ] connects class prototypes to the original graph as virtual class nodes and fine-tunes them by contrastive learning.
VNT  [ 181 ] employs meta-learning on graph data with virtual class nodes, where the class prompts for each target task are aggregated from source tasks through an attention mechanism, aiming to bridge the gap between the source and target domains.
SGL-PT  [ 182 ] leverages masked feature prediction to fine-tune class prompts. The graph classification problem is transformed into the reconstruction of a masked virtual supernode, which serves as the global representation.
DAGPrompT  [ 176 ] concatenates the embeddings as well as class prompts from every GNN layer, and performs InfoNCE-based similarity learning.
Type prompting-based methods, such as HGPROMPT  [ 148 ] and HetGPT  [ 183 ] , assign a prompt to each node type, similar to class prototypes. Unlike class prompts, node types are considered node properties that are common in heterogeneous graphs and typically do not require manual labeling.

 
 Link prompts .
Considering the structural knowledge carried by downstream graph data is largely overlooked by node feature-based prompts, prompts on adjacency matrices begin to thrive.
EdgePrompt  [ 185 ] proposes edge prompts that are learnable attributes on every edge and are integrated into node representations through message propagation. Similar to GPF, edge prompts can be either one shared vector or customized vectors generated by prompt bases.
All in One  [ 153 ] considers pairwise relationships between prompt tokens and constructs a graph prompt 𝒢 𝒑 = ( 𝐗 𝒑 , 𝐀 𝒑 ) \mathcal{G}_{\boldsymbol{p}}=(\mathbf{X}_{\boldsymbol{p}},\mathbf{A}_{\boldsymbol{p}}) , where 𝐗 𝒑 = [ 𝒑 k ] k \mathbf{X}_{\boldsymbol{p}}=[\boldsymbol{p}_{k}]_{k} and 𝐀 𝒑 = [ ⟨ 𝒑 k , 𝒑 l ⟩ ] k , l \mathbf{A}_{\boldsymbol{p}}=[\langle\boldsymbol{p}_{k},\boldsymbol{p}_{l}\rangle]_{k,l} . A meta-learning strategy is developed to adapt All in One to miscellaneous downstream scenarios.

 
 Contextual prompts .
Context information is often considered in prompting methods in the form of aggregated neighborhood features or embeddings.
GPPT  [ 68 ] and GraphPrompt series  [ 142 , 187 ] design structural prompts that encode one-hop aggregated contextual information for downstream tuning.
Self-Pro  [ 79 ] constructs a 2-hop adjacency matrix 𝐀 2 = [ A i , j = 1 ∩ A j , k = 1 ] i , k \mathbf{A}_{2}=[A_{i,j}=1\cap A_{j,k}=1]_{i,k} as the contextual prompt.
SUPT  [ 188 ] builds upon the learnable prompt bases in GPF-plus  [ 177 ] . However, it uses a message-passing approach to aggregate these bases, preserving the semantic similarity among neighboring nodes in the prompts.
ProNoG  [ 189 ] generates contextual prompts by employing a condition-net, where the input representations are aggregated from k k -hop subgraphs.
Building on this, GCoT  [ 190 ] iteratively aggregates representations from every hidden layer of a pre-trained GNN as “thoughts”, simulating the Chain-of-Thought reasoning in large language models.

 
 
 Some prompting methods move away from feature vectors and seek other effective forms.
For example, PRODIGY  [ 152 ] and OFA  [ 154 ] construct a prompt graph 2 2 
 2 
 
 
 
 Unlike the existing survey  [ 197 ] which categorizes both All in One  [ 153 ] and PRODIGY  [ 152 ] as “Prompt as Graphs”, we explicitly distinguish them by different notions: “ graph prompt ”, a graph added on the downstream graph as an entire prompt; and “ prompt graph ”, a new graph comprised of prompt nodes and class nodes. , in which each prompt node (data node) represents a sampled k k -hop subgraph and each class node represents a class to which the central node belongs. Links between prompt nodes and class nodes indicate the task-specific supervision signals.
Learning to predict these links has been demonstrated to be a highly generalizable strategy, applicable both to pre-training and fine-tuning to facilitate both few-shot in-context learning  [ 152 ] and zero-shot learning  [ 154 ] .
Besides, PRODIGY updates the prompt graph with an auxiliary self-supervised context prediction objective: it predicts if one node belongs to the k k -hop subgraph of another target node.

 
 Other prompts .
Some cutting-edge prompting methods explore underlying topological properties of graph data.
IGAP  [ 143 ] designs a spectral prompt to transform the low-dimensional pre-training domain to the fine-tuning domain, as the low-frequency domain describes local smooth patterns of graph signals.
TGPT  [ 194 ] captures graphlet information – small motifs that describe local structure patterns of a node – into its node-level and the graph-level prompts. Specifically, node-level topology-aware prompts are generated from a fast graphlet transform matrix, and graph-level ones are aggregated from them.

 
 Discussion. 
Graph prompting is a novel approach to envisioning graph downstream tuning.
It can be viewed as a form of PEFT on graph data instead of model architecture,
where prompts are considered tunable adapters across the pre-training and downstream data spaces, achieving both effectiveness and efficiency.
Prompting not only bridges data domains but also provides unified fine-tuning templates for various downstream prediction tasks.
Such templates include link prediction  [ 79 , 152 , 154 , 176 ] , graph classification  [ 153 ] , subgraph similarity prediction  [ 142 , 187 ] , and more.
Despite the promising advancements, graph prompting remains a developing area, offering opportunities for further research and improvement.
For instance, many graph prompts remain challenging for humans to comprehend, posing challenges for the explainability of graph prompting.
Additionally, most proposed “unified prompt templates for downstream tasks” primarily focus on discriminative tasks such as link prediction and graph classification, while neglecting generative and other open-ended tasks that also have widespread demand.

 
 
 
 

## VII Self-supervised Graph Language Models 

 
 Previous sections have explored how self-supervised GFMs learn different types of graph knowledge through pre-training and downstream tuning. The emergence of large language models (LLMs) has opened up new avenues for constructing graph language models (GLMs) that leverage knowledge patterns, architectures, and training strategies from the natural language domain to process graph data. While traditional GFMs excel at capturing structural patterns, GLMs aim to bridge the gap between graph topology and semantic understanding by combining the strengths of both GNNs and language models.

 
 
 This section examines self-supervised GLMs from two perspectives shown in Fig.  9 :
(1) pre-training GLMs with self-supervision, which focuses on incorporating graph knowledge into language modeling pre-training and graph pre-training of GLMs; and
(2) tuning GLMs, which focuses on adapting pre-trained language knowledge to graph-specific scenarios through techniques like prompting and fine-tuning.
In what follows, we delve into these two directions, particularly focusing on how GLMs integrate different forms of graph knowledge.

 
 {forest} 
 
 Fig. 9: 

Our taxonomy of self-supervised GLMs with representative literature.
 : network parameter is updated; : parameter frozen.

 
 
 

### VII-A Pre-training GLMs with Self-supervision 

 
 GLM pre-training includes two distinct categories: language modeling pre-training , and graph pre-training .
Note that both pre-training schemes can be applied to graph models such as GNNs and graph transformers, as well as to open-source language models (LMs) like T5  [ 256 ] and BERT  [ 130 ] .

 
 

#### VII-A 1 Language Modeling Pre-training

 
 Language modeling pre-training refers to a set of self-supervised pretext tasks in the language domain, known as “language modeling”.
They mainly include a uto r egressive language modeling (AR) and m asked l anguage m odeling (MLM).
AR, also known as causal language modeling and next-token prediction, predicts the next token in a sequence usually by a maximum likelihood estimation loss.
The well-known AR models include GPT series  [ 257 ] , LLaMA  [ 258 ] , and DeepSeek  [ 259 ] .
MLM, exemplified by BERT  [ 130 ] and RoBERTa  [ 260 ] , predicts masked tokens using bidirectional context, making it ideal for understanding tasks but less suited for generation.
Additionally, hybrid methods such as T5  [ 256 ] and BART  [ 261 ] combine AR and MLM to handle both understanding and generation, though at the cost of higher complexity.

 
 
 Many early GLMs directly utilize open-source or closed-source LLMs pre-trained by the aforementioned tasks.
Although pure LLMs have demonstrated preliminary abilities in handling and reasoning on graphs  [ 208 ] , the modality gap between text and graphs makes graph tasks challenging for LLMs without additional support  [ 209 , 210 ] .
Therefore, several attempts integrate GNNs as components of LLMs and jointly pre-train them by AR and MLM.
UniGraph  [ 132 ] and P2TAG  [ 192 ] concatenate an LM and a GNN together and perform MLM on textual node attributes. THLM  [ 198 ] jointly pre-trains a BERT and a heterogeneous GNN by MLM.
On top of this, graph-oriented language pre-training methods are developed.
Patton  [ 147 ] improves traditional MLM to contextualized MLM: it utilizes both representations of the current node and its neighboring nodes to predict the missing tokens of the current node.
Path-LLM  [ 199 ] first generates textual sequences by interconnecting node text in the same order as the shortest path, where the earlier node text becomes the prefix tokens for AR pre-training.

 
 
 

#### VII-A 2 Graph Pre-training

 
 Grap pre-training features self-supervised pretext tasks in the graph domain, as introduced in Section  III – V . They are adapted to pre-train or assist in pre-training language model architectures.
Cross-entropy-based and InfoNCE-based link prediction (Section  III-C ) become the first choice:
GALM  [ 200 ] concatenates an LM and a GNN for joint pre-training, while GraphFormers  [ 8 ] and Patton  [ 147 ] incorporate GNN and Transformer layers into a hybrid framework.
UniGLM  [ 205 ] employs node instance discrimination (Section  III-A ) to pre-train a language model, where positive node samples are selected from multiple hops of the central node by Personalized PageRank.

 
 
 In contrast to single-modal discrimination, node-text discrimination is a more common approach inspired by existing multimodal pre-training models like CLIP  [ 262 ] . This approach discriminates between representations derived from a GNN and an LM/LLM to align the two modalities.
GraphGPT  [ 204 ] trains a textual and a graph Transformer by minimizing a cross-entropy-based discrimination loss to align their representation spaces for future instruction tuning.
LLaSA  [ 207 ] unifies different types of structured data into hypergraphs and performs contrastive learning between a hypergraph GNN and a hybrid Transformer.

 
 
 To achieve more effective modality alignment, researchers agree on the necessity of incorporating higher-order graph-specific knowledge during the self-supervised pre-training.
Graph context (Section  IV-A ) becomes especially crucial for GLMs,
as it is easy for LMs to understand compared to other structural knowledge, and provides a clearer perspective of the textual semantics of node entities and their adjacency relationships.
G2P2  [ 201 ] jointly pre-trains a Transformer and a GCN by context-aware node-text discrimination: it generates a summary embedding by averaging neighborhood text embeddings and performs discrimination among node, text, and summary.
GRENADE  [ 202 ] employs both node-level and neighborhood-level discrimination within and between GNN and BERT. It minimizes the KL divergence of neighbor similarity distributions.
THLM  [ 198 ] introduces an auxiliary objective (along with MLM) to differentiate contextual and distant nodes.
Another common graph knowledge is long-range similarities (Section  IV-B ).
ConGraT  [ 203 ] jointly pre-trains a sentence Transformer and a GAT by node-text discrimination with a long-range similarity function, namely the number of common neighbors and SimRank.
GSPT  [ 206 ] pre-trains a Transformer by masked feature prediction (Section  III-A ) on generated random walk sequences.
They employ a node feature reconstruction loss rather than predicting the masked token IDs as in MLM.

 
 
 
 

### VII-B Tuning GLMs 

 
 Downstream tuning is the dominant approach for deploying GLMs, given the high computational demands of LLM pre-training.
LLMs to be tuned are usually pre-trained by large-scale language modeling pretexts, endowing them with a certain level of world knowledge.
They can be utilized for either adapting to graph-specific downstream scenarios or guiding the training of other downstream branches.
Some literature categorizes the roles of LLMs as enhancers, predictors, encoders, aligners, etc.  [ 232 , 19 , 263 ] , viewing pre-trained LLMs as auxiliary modules.
Instead, we consider pre-trained LLMs as the core model and treat all subsequent training processes as downstream tuning.
This perspective allows us to more clearly demonstrate how graph-specific knowledge assists LLMs in graph downstream tuning.

 
 
 Note that GLMs discussed in this section do not necessarily employ self-supervised pre-training, and they may be designed for only one specific downstream task.
However, here we focus solely on the tuning strategies employed,
which we believe have the potential to be applied to self-supervised GLMs or to provide valuable insights.

 
 

#### VII-B 1 Prompting for GLMs

 
 With the advent of LLMs, the earliest GLMs attempt to directly utilize closed-source LLMs such as GPT-3.5 to construct graph learners.
Specifically, by designing particular graph-specific prompts, LLMs can get a grasp of key graph knowledge for open-ended tasks such as question answering and reasoning.
Some methods also rely on in-context learning  [ 264 , 215 ] , i.e., to provide a few task-specific examples within the prompts to guide the LLM output.
Many early attempts, including empirical studies on using LLMs for graph tasks  [ 208 , 209 , 210 , 211 , 212 ] , resort to graph-specific prompts rather than fine-tuning strategies.

 
 
 To achieve effective prompting, the first step is to convert graphs into text using natural language or structured language  [ 264 , 11 ] .
However, merely inputting textualized graphs is insufficient for LLMs to fully comprehend graph structural knowledge. Various methods have been proposed to further embed graph knowledge into prompts.
For instance, SNS  [ 213 ] ranks the textual similarity between nodes and selects the top-2 similar neighbors as additional instructions.
Skianis et al.   [ 214 ] textualize the reasoning process of various graph question-answering tasks as pseudo-code functions.
AskGNN  [ 215 ] employs a GNN to select an optimal set of examples for in-context learning.

 
 
 

#### VII-B 2 Fine-tuning for GLMs

 
 The rise of open-source LLMs, coupled with the limitations of prompting models in terms of cross-modal and zero-shot generalization capabilities, has facilitated the research of graph-specific fine-tuning for GLMs.
 Instruction tuning is the primary fine-tuning approach for LLMs, wherein appropriate queries for downstream tasks are constructed to elicit the desired LLM predictions, and model parameters are updated by minimizing an error function.
However, LLMs typically possess a larger parameter scale than GNNs or GTs, making it challenging for GLMs to perform full fine-tuning.
Consequently, parameter-efficient fine-tuning (PEFT) methods have become the predominant approach, where the LLM is frozen and a special fine-tuning module is tuned instead.
Apart from intrinsic adapters such as LoRA  [ 196 ] , GLMs often incorporate tailored fine-tuning modules to (1) maximize the retention of pre-trained LLMs’ powerful semantic encoding ability, and (2) align the semantic spaces of text and graphs.
Types, interfaces, and tuning strategies of fine-tuning modules vary.

 
 Module types: GNNs vs. Non-GNNs. The most commonly used fine-tuning module is GNN, which can capture structural knowledge of graphs, thereby helping to bridge the modality gap.
The training of these modules can rely on both downstream task-relevant information and self-supervision signals.
GraphToken  [ 217 ] and GraphPrompter  [ 218 ] utilize a GNN to generate graph tokens as input for LLMs, and tune 3 3 
 3 
 
 
 
 By “tuning” we mean that the GNN serves as an auxiliary module of a pre-trained LLM.
From the perspective of the GNN, it may be trained from scratch and regarded as pre-training in the original paper. This does not conflict with our statements in this section. 
the GNN by answering graph-related questions.
Conversely, G-Prompt  [ 219 ] uses MLM to tune a GNN following an LM to obtain graph representations through textual prompts.
GraphAdapter  [ 220 ] tunes a GNN by AR as a graph learning branch of a pre-trained LM. Then, it merges the representations with an MLP-based fusion block before passing them to a downstream head.
GOFA  [ 222 ] interleaves GNN layers into a pre-trained LLM encoder as adapters and employs various self-supervised fine-tuning methods, including AR, shortest path distance prediction, and common neighbor prediction.
DGTL  [ 221 ] first generates disentangled graph embeddings by assigning diverse edge weights to each GNN layer. These embeddings are then incorporated into the input tokens of each LLM layer to perform context-aware instruction tuning.
TAGA  [ 223 ] and GraphCLIP  [ 224 ] employ node-text discrimination fine-tuning to align graph and text.
TEA-GLM  [ 227 ] tunes a GNN using both node instance and dimension discrimination (Section  III-A ) between GNN embeddings and PCA-processed LLM embeddings.
Some methods have developed more complex fine-tuning modules based on GNNs.
Pan et al.   [ 225 ] use a pair of GNNs for two-stage tuning. It first trains a GNN “interpreter” supervised by LLM-extracted keyword information, and then distills the interpreter to another downstream GNN.
ENGINE  [ 226 ] develops a bypass fine-tuning architecture for LLMs, “G-Ladders”, which consists of multiple levels of GNNs and projectors. This structure enhances computational efficiency through node embedding caching.

 
 
 Unlike GNNs, non-GNN fine-tuning module types such as MLPs lack the ability to handle local graph structures. To achieve modality alignment, these methods often resort to specialized tuning schemes.
GraphGPT  [ 204 ] tunes a linear adapter through “ graph-instruction matching ” before downstream instruction tuning. The LLM is instructed to reorder the list of node text to match textual embeddings obtained from a parallel GNN+Transformer encoder.
LLaGA  [ 229 ] extracts contextual knowledge from a graph using two tokenizers: node embedding concatenation through level-order traversal on a neighborhood tree, and neighborhood embedding aggregation in different hops.
HIGHT  [ 230 ] develops a hierarchical graph tokenizer to combine node, motif, and graph-level information for instruction tuning.
GraphTranslator  [ 231 ] tunes an attention-based adapter, the “translator”, to project graph embeddings into the LLM space. It is guided by LLM-generated text that describes various knowledge patterns such as summaries of nodes, neighborhoods, and other commonalities.

 
 Interfaces: representations vs. augmentations. 
We have discussed GLMs that leverage LLM-empowered representations for fine-tuning.
LLMs can also generate augmented data components for fine-tuning modules. They include:

 
 
 (1) Text :
TAPE  [ 10 ] instructs an LLM to explain its decisions in classifying nodes, facilitating an in-depth understanding of node-level information. These explanations are then encoded by a smaller LM to enrich textual features for downstream GNN training.
Unlike TAPE which directly uses original adjacency matrices, SFGL  [ 236 ] generates scale-free graphs from the input data to obtain textual features more aligned with real-world edge distributions.
KEA  [ 232 ] instructs an LLM to generate descriptions of terminologies across different fields.
LLM4Mol  [ 233 ] uses ChatGPT to generate descriptions of chemical molecules, including functional groups and other properties, to fine-tune a small-scale RoBERTa.
TANS  [ 234 ] leverages centralities and clustering coefficients (Section  III-B ), along with contextual information, to generate descriptive text for nodes using GPT-4o-mini for non-textual graphs.
GAugLLM  [ 235 ] enhances textual representations with node context summaries, and uses them to guide feature-level and edge-level augmentations in self-supervised fine-tuning tasks such as node instance discrimination and masked feature prediction.

 
 
 (2) Pseudo-labels :
LLM-GNN  [ 237 ] leverages the LLM to generate cluster-aware node pseudo-labels to supervise a downstream GNN, referred to as label-free node classification .
A set of nodes closer to K K -means cluster centers is selected for annotation based on a cluster density metric.
Similarly, Locle  [ 239 ] uses subspace clustering to find the node set. LLM pseudo-labels are then selected based on their information certainty and refined by graph rewiring.

 
 
 (3) Graph structure :
LLM4NG  [ 240 ] and OpenGraph  [ 241 ] both use a pre-trained LLM to generate node samples and associated links to augment the original graph data.
While LLM4NG trains an MLP-based edge predictor via cross-entropy-based link prediction, OpenGraph generates new edges through Gibbs sampling and trains a GT by masked link prediction.
Sun et al.   [ 243 ] utilize GPT-3.5-Turbo to assist in removing unreliable edges and adding reliable ones to create a refined graph for GNN input.
LOGIN  [ 242 ] treats the LLM as a consultant for node classification during GNN fine-tuning. If the LLM prediction matches the ground truth, it updates the original feature; if not, it prunes its associated links based on neighbor similarity.

 
 Backbone parameters: frozen vs. tuned. 
We have discussed GLMs with additional fine-tuning modules, where the LLM backbone remains frozen.
Another line of studies designs special textual and graph domain co-training strategies to fine-tune the LLM backbone.
GLEM  [ 244 ] iteratively tunes a GNN and a DeBERTa model using a variational Expectation-Maximization framework. GraphLLM  [ 245 ] synergistically tunes a GT and a LLaMA 2 model via “prefix-tuning”, i.e., to project graph representations into trainable tokens and prepend them to the keys and values of every Transformer attention layer.
LEADING  [ 246 ] reduces the cost of joint LLM-GNN fine-tuning by decoupling the computation of node embedding from neighborhood embeddings.
Instead of joint tuning, SimTeG  [ 247 ] separates the tuning of LMs and GNNs in a two-stage manner for more distinguishable embedding spaces.

 
 
 Among all types of graph structural knowledge, link information (Section  III-C ) is often explicitly extracted to fine-tune LLMs, as it directly indicates relationships between entities and can be easily extracted by both GNNs and LLMs.
InstructGLM  [ 11 ] introduces link prediction into LLM instruction tuning as an auxiliary objective, similar to the training of SuperGAT  [ 72 ] .
CMRP  [ 184 ] tunes an LLM and a GNN to dynamically select an optimal edge set. The generated edge set is then injected into prompt tokens to iteratively instruct the LLM.
GIANT  [ 9 ] presents neighborhood prediction to fine-tune an XR-Transformer, which constructs hierarchical clusters and predicts the rows of the adjacency matrix as a multi-class classification task.
AuGLM  [ 254 ] selects and summarizes neighboring nodes by Personalized PageRank scoring and GNN-generated class prototypes for instruction tuning.
Another type of knowledge information involves long-range paths (Section  IV-B ).

WalkLM  [ 248 ] incorporates long-range information into MLM to fine-tune a DistilRoBERTa model: it first samples attributed random walk sequences and then textualizes them as a token list.
GUNDAM  [ 249 ] fine-tunes an LLM by reasoning different paths between two nodes, with path labels generated by unsupervised algorithms.
LinguGKD  [ 250 ] develops a distillation architecture for LLM-GNNs: it first leverages node degrees and k-hop neighbors to instruct-tune an LLM. Then, the GNN is fine-tuned by contrasting every intermediate layer of both models.
InstructGraph  [ 251 ] , GraphWiz  [ 252 ] , and GraphInstruct  [ 253 ] conduct multi-task LLM fine-tuning by combining over a dozen instruction tuning tasks. They include self-supervised predictions on various graph structures (cycles, shortest paths, maximum flows, Hamilton paths, etc.) as well as node-level and link-level downstream signals, aiming to provide the LLM with a comprehensive understanding of the graph domain.

 
 Discussion. 
While both GLM pre-training and tuning strategies demonstrate the potential of bridging the modality gap between graph and text,
it remains underexplored whether GLMs have adequately tapped the potential in LLMs with billion-scale parameters.
Some excellent properties of LLMs, e.g., the emergent ability  [ 265 ] , are yet to be discovered on graph model architectures.
The fragile side of LLMs such as
hallucinations  [ 266 ] and intervention of spurious factors  [ 208 ] keeps posing challenges to GLM researchers.
To tackle these challenges, future research should focus on developing more powerful graph-specific architectures as well as pre-training and fine-tuning strategies that effectively extract and leverage various types of graph knowledge.

 
 
 
 
 

## VIII Challenges and Future Directions 

 
 This section discusses potential challenges that graph researchers may encounter and insights for future research directions towards GFMs, as a conclusion to our survey.

 
 

### VIII-A Combining Different Graph Knowledge Patterns 

 
 Despite that a variety of pretext tasks have been proposed for self-supervised graph models,
the effectiveness of these pretexts depends on the application scenarios applied to downstream tasks  [ 267 ] .
Therefore, it is crucial to investigate how to effectively combine different graph knowledge to adapt graph models to more application scenarios.

 
 
 One research direction is to jointly optimize different pretext objectives to obtain different aspects of graph knowledge. This can be formulated as a multi-task pre-training problem.
Traditional methods simply assign hyperparameters to weigh each pretext task, leading to suboptimal performance.
AutoSSL  [ 267 ] and ParetoGNN  [ 268 ] are neural parameter search algorithms in order to dynamically find a set of optimal coefficients to combine different pretexts.
GraphTCM  [ 269 ] models a correlation value for every pair of pretexts and optimizes the correlation matrix to find the best parameters.
AGSSL  [ 270 ] and WAS  [ 271 ] propose a knowledge distillation technique, where the knowledge from different teachers is distilled into a single unified student.
Another promising approach is to design specific multi-task prompting strategies.
ULTRA-DP  [ 157 ] and MultiGPrompt  [ 191 ] are multi-task prompting methods that assign a learnable task prompt for each pretext and pass them to the downstream model.
While ULTRA-DP searches for the best task prompt for the downstream task, MultiGPrompt combines all of them by a linear combination or a parameterized network.

 
 
 

### VIII-B Knowledge Adaptation across Graph Types 

 
 It is important for future GFMs to explore a wider range of graph knowledge, handling various complex data types beyond simple graphs.
Here, we focus on several graph types and their corresponding knowledge patterns. We will show that these patterns and processing methods share commonalities with the graph knowledge discussed in Sections  III – V , implying potential data unification in future GFMs.

 
 Heterogeneous knowledge graphs. 
Nodes and links in heterogeneous graphs possess unique types that exhibit distinct knowledge patterns.
Some existing methods distinguish and handle different node and edge types.
For instance, Heterformer  [ 272 ] and RMR  [ 273 ] are node instance discrimination methods within specific node and relation types, respectively.
HiGPT  [ 228 ] is a GLM that pre-trains an LLM by matching node types within sampled heterogeneous subgraphs.
For downstream tuning, HG-Adapter  [ 175 ] is a PEFT approach that constructs a heterogeneous structure by calculating correlation scores for every neighboring node type to fine-tune an adapter. HGPROMPT  [ 148 ] splits a heterogeneous graph into multiple type-specific subgraphs and designs a learnable prompt vector for each node type.

 
 
 However, these methods do not explicitly mine the higher-order structured knowledge underlying heterogeneous types.
For example, meta-paths are sequences of nodes and links with specific types, representing composite relations between different entities, such as “co-write: author-paper-author” and “chemical reaction: compound-reaction-compound”.
Learning meta-paths is similar to learning links, involving methods like meta-path prediction  [ 274 , 39 , 255 ] and instance discrimination with meta-path-based augmentation  [ 275 ] . Another example is HetGPT  [ 183 ] , a prompting method that aggregates class prompts along meta-paths.
Additionally, the network schema is a motif-like heterogeneous pattern that contains all relationships of a particular node type. Due to the rich local structural information in the network schema and the ease of instance retrieval and extraction, it is used to generate contrastive instances in PT-HGNN  [ 276 ] and HeCo  [ 275 ] .

 
 
 Knowledge graphs are heterogeneous graphs embedded in specific domains and enriched with domain-specific factual knowledge.
In knowledge graphs, relation triples (head entity, relation, tail entity) serve as the fundamental knowledge units.
RotatE  [ 277 ] employs a negative sampling-based margin loss to discriminate between the head and tail entities in each relation triple.
KEPLER  [ 278 ] further incorporates the margin loss into MLM to fine-tune a RoBERTa model.
SelfKG  [ 279 ] and AutoAlign  [ 280 ] utilize an InfoNCE and Triplet estimator, respectively, to capture entity information and align them.
GNP  [ 281 ] proposes masked relation prediction, similar to masked link prediction, within constructed positive and negative relation triples.

 
 Dynamic graphs. 
Real-world graph data often exhibits dynamic characteristics.
The structure or size of a graph may change over time, a phenomenon known as dynamic evolution .
Discrete dynamic evolution manifests as a series of graph snapshots that are subgraphs with evolved features or graph structure,
while continuous dynamic evolution is represented by a sequence of events with timestamps.

 
 
 Dynamic node features are coupled with temporal knowledge.
GPT-ST  [ 282 ] and STGP  [ 283 ] perform masked feature prediction across the time dimension of the feature matrix. DDGCL  [ 284 ] introduces a temporal similarity function to the InfoNCE estimator, weighted by a time gap penalty. STGP  [ 283 ] and DyGPrompt  [ 285 ] design dual prompts that capture both downstream node features and time information. LLM4DyG  [ 216 ] prompts an LLM to sequentially consider node and time information, significantly improving the LLM’s reasoning abilities on dynamic graphs.
For dynamic evolution on graph structures, GraphPro  [ 193 ] constructs a graph prompt that concatenates various graph snapshots with the original structure, and incorporates temporal weights into downstream message passing.

 
 Hypergraphs. 
Hypergraphs involve hyperedges that connect more than two nodes. They are better suitable for capturing higher-order structural relationships beyond simple pairwise interactions.
Hyperedges can be considered as a special type of context, as they represent local commonalities among nodes.
VilLain  [ 286 ] learns a balanced and distinctive pseudo-label distribution by propagating the prototype vectors along the hyperedges.
For instance discrimination models, HyperGCL  [ 287 ] focuses on generating hypergraph augmentations with a variational autoencoder, while TriCL  [ 288 ] focuses on performing contrast within and between nodes and hyperedges based on the connection membership.
HypeBoy  [ 289 ] predicts if a node belongs to a hyperedge formed by another set of nodes by minimizing the similarity between their projected embeddings.

 
 
 

### VIII-C Tackling Potential Biases in Future GFMs 

 
 Potential biases in graph representations can significantly affect the generalization ability of graph models  [ 53 , 290 ] , presenting challenges in building GFMs.
These biases may be inherent in the graph structural data itself or introduced during pre-training or downstream tuning, mainly including the following aspects.

 
 Imbalanced data distributions. 
Potential biases manifest when the feature or structural properties of graphs exhibit uneven distributions.
They mainly include imbalanced class distributions, node degree distributions, and sensitive representation distributions.
To mitigate potential biases, current methods involve constructing balanced training data distributions through structure-aware augmentations.
GRADE  [ 54 ] capitalizes on feature similarity to balance nodes with imbalanced degrees, including removing dissimilar connections of high-degree nodes and aligning the neighbor distribution of similar low-degree nodes.
For long-tailed class distributions, ImGCL  [ 53 ] first divides a graph into several K-means clusters, each one referring to a latent class.
Node representations are then sampled within each cluster based on PageRank scores.
FPrompt  [ 186 ] is a fair graph prompting approach where the structural prompt is obtained by masking links within different sensitive groups.

 
 
 Another way is to design specific training strategies to distinguish or emphasize imbalanced instances.
CM-GCL  [ 291 ] prunes model parameters during contrastive pre-training to capture minority samples and employs a focal loss to emphasize these samples during fine-tuning.
Graphair  [ 292 ] introduces a learnable graph augmentation network and a discriminative network to identify sensitive attributes in the augmented graphs. It utilizes adversarial training to update both networks.
GraphPAR  [ 174 ] is a fair PEFT (Section  VI-A ) approach that generates multiple sensitive attribute vectors for every node and minimizes the distance between them during fine-tuning.

 
 Vulnerability to attacks. 
Graph models that are not robust against perturbations are vulnerable to malicious attacks,
such as injecting noise into features or textual attributes, and inserting nodes and links into the graph structure to disrupt the underlying knowledge patterns  [ 293 , 294 ] .
Structural attacks often become more effective than feature-based ones  [ 290 ] .

 
 
 To mitigate the impact of structural attacks,
GRV  [ 118 ] introduces a robustness quantification metric based on mutual information and uses it to guide the training of a DGI model  [ 13 ] .
RES  [ 290 ] demonstrates the effectiveness of random edge dropping in defense against structural attacks and applies it to node-level and graph-level instance discrimination.
The recent emergence of GLMs has prompted further research in terms of their robustness.
Although LLMs provide a certain degree of robustness compared to GNNs, their performance can still decline considerably under structural attacks  [ 295 , 238 ] .
To improve structural robustness, LLM4RGNN  [ 238 ] tunes a local LLM and a small LM, which are used to identify and remove malicious links, as well as to re-add missing important links in the perturbed graph data.

 
 
 

### VIII-D Building Powerful and Explainable GFMs 

 
 There is a pressing need for better explainability of GFMs.
It not only ensures the model reliability in practical applications but also provides valuable insights into understanding graph knowledge.
A promising and underexplored avenue is Reasoning on Graphs (RoG).
Drawing from the huge success of LLM reasoning, RoG researchers tend to employ natural language instructions, such as Chain-of-Thoughts and Tree-of-Thoughts, to guide GLMs in reasoning over graph structures  [ 245 , 296 , 297 ] .
Reasoning not only enhances the model explainability by reducing hallucinations but also enriches the knowledge provided, thereby boosting the generalization ability of GFMs in various open-ended tasks.
Additionally, recent GLMs leverage the power of external knowledge bases through Retrieval-Augmented Generation (RAG)  [ 298 , 299 , 300 ] .
These models retrieve knowledge relevant to the queries from knowledge graph databases to enhance reasoning explainability.
It is noteworthy that recent RAG approaches have begun to recognize the importance of higher-order structural knowledge, such as long-range paths  [ 301 ] and clusters  [ 302 ] . They can be integrated into our knowledge taxonomy in the future.
By advancing in these directions, we can pave the way for versatile GFMs and graph agents that are capable of handling diverse, complex, and domain-specific graph tasks with unprecedented effectiveness and adaptability.

 
 
 
 

## References

 
 
 [1] 
 
L. Wu, P. Cui et al. , “Graph neural networks: foundation, frontiers and applications,” in KDD , 2022.

 

 
 [2] 
 
Z. Zhang, P. Cui et al. , “Deep learning on graphs: A survey,” TKDE , 2020.

 

 
 [3] 
 
Y. Liu, M. Jin et al. , “Graph self-supervised learning: A survey,” TKDE , 2022.

 

 
 [4] 
 
T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in ICLR , 2017.

 

 
 [5] 
 
P. Veličković, G. Cucurull et al. , “Graph attention networks,” in ICLR , 2018.

 

 
 [6] 
 
W. Hamilton, Z. Ying et al. , “Inductive representation learning on large graphs,” in NIPS , 2017.

 

 
 [7] 
 
Y. Rong, Y. Bian et al. , “Self-supervised graph transformer on large-scale molecular data,” in NeurIPS , 2020.

 

 
 [8] 
 
J. Yang, Z. Liu et al. , “GraphFormers: GNN-nested transformers for representation learning on textual graph,” in NeurIPS , 2021.

 

 
 [9] 
 
E. Chien, W.-C. Chang et al. , “Node feature extraction by self-supervised multi-scale neighborhood prediction,” in ICLR , 2022.

 

 
 [10] 
 
X. He, X. Bresson et al. , “Harnessing explanations: LLM-to-LM interpreter for enhanced text-attributed graph representation learning,” in ICLR , 2024.

 

 
 [11] 
 
R. Ye, C. Zhang et al. , “Language is all a graph needs,” in EACL Findings , 2024.

 

 
 [12] 
 
T. N. Kipf and M. Welling, “Variational graph auto-encoders,” in NIPS Workshop (BDL) , 2016.

 

 
 [13] 
 
P. Veličković, W. Fedus et al. , “Deep graph infomax,” in ICLR , 2019.

 

 
 [14] 
 
J. Liu, C. Yang et al. , “Towards graph foundation models: A survey and beyond,” CoRR , 2023.

 

 
 [15] 
 
R. Bommasani, D. A. Hudson et al. , “On the opportunities and risks of foundation models,” CoRR , 2021.

 

 
 [16] 
 
J. Xia, Y. Zhu et al. , “A survey of pretraining on graphs: Taxonomy, methods, and applications,” CoRR , 2022.

 

 
 [17] 
 
Y. Xie, Z. Xu et al. , “Self-supervised learning of graph neural networks: A unified review,” TPAMI , 2022.

 

 
 [18] 
 
Z. Zhang, H. Li et al. , “Graph meets llms: Towards large graph models,” in NeurIPS Workshop (GLFrontiers) , 2023.

 

 
 [19] 
 
B. Jin, G. Liu et al. , “Large language models on graphs: A comprehensive survey,” TKDE , 2024.

 

 
 [20] 
 
W. Fan, S. Wang et al. , “Graph machine learning in the era of large language models (LLMs),” CoRR , 2024.

 

 
 [21] 
 
X. Ren, J. Tang et al. , “A survey of large language models for graphs,” in KDD , 2024.

 

 
 [22] 
 
H. Mao, Z. Chen et al. , “Position: Graph foundation models are already here,” in ICML , 2024.

 

 
 [23] 
 
Z. Hou, X. Liu et al. , “GraphMAE: Self-supervised masked graph autoencoders,” in KDD , 2022.

 

 
 [24] 
 
H. Yang, H. Chen et al. , “Dual space graph contrastive learning,” in WWW , 2022.

 

 
 [25] 
 
W. Ju, Y. Wang et al. , “Towards graph contrastive learning: A survey and beyond,” CoRR , 2024.

 

 
 [26] 
 
K. Sun, Z. Lin et al. , “Multi-stage self-supervised learning for graph convolutional networks on graphs with few labeled nodes,” in AAAI , 2020.

 

 
 [27] 
 
T. Zhang, Y. Xiong et al. , “CommDGI: Community detection oriented deep graph infomax,” in CIKM , 2020.

 

 
 [28] 
 
C. Wang, S. Pan et al. , “MGAE: Marginalized graph autoencoder for graph clustering,” in CIKM , 2017.

 

 
 [29] 
 
J. Park, M. Lee et al. , “Symmetric graph convolutional autoencoder for unsupervised graph representation learning,” in ICCV , 2019.

 

 
 [30] 
 
J. Zhang, H. Zhang et al. , “Graph-Bert: Only attention is needed for learning graph representations,” CoRR , 2020.

 

 
 [31] 
 
Z. Peng, W. Huang et al. , “Graph representation learning via graphical mutual information maximization,” in WWW , 2020.

 

 
 [32] 
 
W. Hu, B. Liu et al. , “Strategies for pre-training graph neural networks,” in ICLR , 2019.

 

 
 [33] 
 
Y. You, T. Chen et al. , “When does self-supervision help graph convolutional networks?” in ICML , 2020.

 

 
 [34] 
 
W. Jin, T. Derr et al. , “Self-supervised learning on graphs: Deep insights and new direction,” CoRR , 2020.

 

 
 [35] 
 
Y. Xie, Z. Xu et al. , “Self-supervised representation learning via latent graph prediction,” in ICML , 2022.

 

 
 [36] 
 
B. Fatemi, L. El Asri et al. , “SLAPS: Self-supervision improves structure learning for graph neural networks,” in NeurIPS , 2021.

 

 
 [37] 
 
Z. Hu, Y. Dong et al. , “GPT-GNN: Generative pre-training of graph neural networks,” in KDD , 2020.

 

 
 [38] 
 
Z. Hou, Y. He et al. , “GraphMAE2: A decoding-enhanced masked self-supervised graph learner,” in WWW , 2023.

 

 
 [39] 
 
Y. Tian, K. Dong et al. , “Heterogeneous graph masked autoencoders,” in AAAI , 2023.

 

 
 [40] 
 
J. Xia, C. Zhao et al. , “Mole-BERT: Rethinking pre-training graph neural networks for molecules,” in ICLR , 2023.

 

 
 [41] 
 
J. Xia, S. Chen et al. , “DiscoGNN: A sample-efficient framework for self-supervised graph representation learning,” in ICDE , 2024.

 

 
 [42] 
 
Y. Zhu, Y. Xu et al. , “Deep graph contrastive representation learning,” in ICML Workshop (GRL+) , 2020.

 

 
 [43] 
 
——, “Graph contrastive learning with adaptive augmentation,” in WWW , 2021.

 

 
 [44] 
 
J. Xia, L. Wu et al. , “ProGCL: Rethinking hard negative mining in graph contrastive learning,” in ICML , 2022.

 

 
 [45] 
 
M. Jin, Y. Zheng et al. , “Multi-scale contrastive siamese networks for self-supervised graph representation learning,” in IJCAI , 2021.

 

 
 [46] 
 
Y. Zhang, H. Zhu et al. , “COSTA: Covariance-preserving feature augmentation for graph contrastive learning,” in KDD , 2022.

 

 
 [47] 
 
C. Wei, J. Liang et al. , “Contrastive graph structure learning via information bottleneck for recommendation,” in NeurIPS , 2022.

 

 
 [48] 
 
S. Thakoor, C. Tallec et al. , “Large-scale representation learning on graphs via bootstrapping,” in ICLR , 2022.

 

 
 [49] 
 
D. He, L. Shan et al. , “Exploitation of a latent mechanism in graph contrastive learning: Representation scattering,” in NeurIPS , 2024.

 

 
 [50] 
 
Y. Mo, L. Peng et al. , “Simple unsupervised graph representation learning,” in AAAI , 2022.

 

 
 [51] 
 
J. Yu, H. Yin et al. , “Are graph augmentations necessary? simple graph contrastive learning for recommendation,” in SIGIR , 2022.

 

 
 [52] 
 
X. Cai, C. Huang et al. , “LightGCL: Simple yet effective graph contrastive learning for recommendation,” in ICLR , 2023.

 

 
 [53] 
 
L. Zeng, L. Li et al. , “ImGCL: Revisiting graph contrastive learning on imbalanced node classification,” in AAAI , 2023.

 

 
 [54] 
 
R. Wang, X. Wang et al. , “Uncovering the structural fairness in graph contrastive learning,” in NeurIPS , 2022.

 

 
 [55] 
 
H. Wang, J. Zhang et al. , “Single-pass contrastive learning can work for both homophilic and heterophilic graph,” TMLR , 2023.

 

 
 [56] 
 
P. Zhang, C. Li et al. , “High-frequency-aware hierarchical contrastive selective coding for representation learning on text attributed graphs,” in WWW , 2024.

 

 
 [57] 
 
P. Bielak, T. Kajdanowicz et al. , “Graph Barlow Twins: A self-supervised representation learning framework for graphs,” KBS , 2022.

 

 
 [58] 
 
H. Zhang, Q. Wu et al. , “From canonical correlation analysis to self-supervised graph neural networks,” in NeurIPS , 2021.

 

 
 [59] 
 
A. Bardes, J. Ponce et al. , “VICReg: Variance-invariance-covariance regularization for self-supervised learning,” in ICLR , 2022.

 

 
 [60] 
 
Y. Zhang, H. Zhu et al. , “Geometric view of soft decorrelation in self-supervised learning,” in KDD , 2024.

 

 
 [61] 
 
Z. Hu, C. Fan et al. , “Unsupervised pre-training of graph convolutional networks,” in ICLR Workshop (RLGM) , 2019.

 

 
 [62] 
 
M. Tang, C. Yang et al. , “Graph auto-encoder via neighborhood Wasserstein reconstruction,” in ICLR , 2022.

 

 
 [63] 
 
J. Li, R. Wu et al. , “What’s behind the mask: Understanding masked graph modeling for graph autoencoders,” in KDD , 2023.

 

 
 [64] 
 
R. Winter, F. Noé et al. , “Permutation-invariant variational autoencoder for graph-level representation learning,” in NeurIPS , 2021.

 

 
 [65] 
 
B. Liang, S. Chen et al. , “Centrality-guided pre-training for graph,” in ICLR , 2025.

 

 
 [66] 
 
S. Pan, R. Hu et al. , “Adversarially regularized graph autoencoder for graph embedding,” in IJCAI , 2018.

 

 
 [67] 
 
A. Hasanzadeh, E. Hajiramezanali et al. , “Semi-implicit graph variational auto-encoders,” in NeurIPS , 2019.

 

 
 [68] 
 
M. Sun, K. Zhou et al. , “GPPT: Graph pre-training and prompt tuning to generalize graph neural networks,” in KDD , 2022.

 

 
 [69] 
 
Q. Tan, N. Liu et al. , “S2GAE: Self-supervised graph autoencoders are generalizable learners with graph masking,” in WSDM , 2023.

 

 
 [70] 
 
Z. Zhao, Y. Li et al. , “Masked graph autoencoder with non-discrete bandwidths,” in WWW , 2024.

 

 
 [71] 
 
S. Wan, S. Pan et al. , “Contrastive and generative graph convolutional networks for graph-based semi-supervised learning,” in AAAI , 2021.

 

 
 [72] 
 
D. Kim and A. Oh, “How to find your friendly neighborhood: Graph attention design with self-supervision,” in ICLR , 2021.

 

 
 [73] 
 
X. Jiang, Z. Qin et al. , “Incomplete graph learning via attribute-structure decoupled variational auto-encoder,” in WSDM , 2024.

 

 
 [74] 
 
Y.-S. Cho, “Decoupled variational graph autoencoder for link prediction,” in WWW , 2024.

 

 
 [75] 
 
Q. Zhu, C. Yang et al. , “Transfer learning of graph neural networks with ego-graph information maximization,” in NeurIPS , 2021.

 

 
 [76] 
 
W. Zhao, G. Xu et al. , “Deep graph structural infomax,” in AAAI , 2023.

 

 
 [77] 
 
H. Zhu, K. Sun et al. , “Contrastive Laplacian Eigenmaps,” in NeurIPS , 2021.

 

 
 [78] 
 
H. Zhu and P. Koniusz, “Generalized Laplacian Eigenmaps,” in NeurIPS , 2022.

 

 
 [79] 
 
C. Gong, X. Li et al. , “Self-Pro: A self-prompt and tuning framework for graph neural networks,” in ECML-PKDD , 2024.

 

 
 [80] 
 
J. Qiu, Q. Chen et al. , “GCC: Graph contrastive coding for graph neural network pre-training,” in KDD , 2020.

 

 
 [81] 
 
K. Ding, Y. Wang et al. , “Eliciting structural and semantic global knowledge in unsupervised graph contrastive learning,” in AAAI , 2023.

 

 
 [82] 
 
Y. Hu, H. You et al. , “Graph-MLP: Node classification without message passing in graph,” CoRR , 2021.

 

 
 [83] 
 
W. Dong, J. Wu et al. , “Node representation learning in graph via node-to-neighbourhood mutual information maximization,” in CVPR , 2022.

 

 
 [84] 
 
Y. Jiao, Y. Xiong et al. , “Sub-graph contrast for scalable self-supervised graph representation learning,” in ICDM , 2020.

 

 
 [85] 
 
N. Lee, J. Lee et al. , “Augmentation-free self-supervised learning on graphs,” in AAAI , 2022.

 

 
 [86] 
 
J. Chen, G. Zhu et al. , “Towards self-supervised learning on graphs with heterophily,” in CIKM , 2022.

 

 
 [87] 
 
H. Jung and H. Park, “Balancing graph embedding smoothness in self-supervised learning via information-theoretic decomposition,” in WWW , 2025.

 

 
 [88] 
 
Z. Peng, Y. Dong et al. , “A new self-supervised task on graphs: Geodesic distance prediction,” Information Sciences , 2022.

 

 
 [89] 
 
G. Cui, J. Zhou et al. , “Adaptive graph encoder for attributed graph embedding,” in KDD , 2020.

 

 
 [90] 
 
X. Wang, M. Zhu et al. , “AM-GCN: Adaptive multi-channel graph convolutional networks,” in KDD , 2020.

 

 
 [91] 
 
Z. Chen, Z. Wu et al. , “Dual low-rank graph autoencoder for semantic and topological networks,” in AAAI , 2023.

 

 
 [92] 
 
J. Chen and G. Kou, “Attribute and structure preserving graph contrastive learning,” in AAAI , 2023.

 

 
 [93] 
 
X. Fan, M. Gong et al. , “Maximizing mutual information across feature and topology views for representing graphs,” TKDE , 2023.

 

 
 [94] 
 
W.-Z. Li, C.-D. Wang et al. , “Towards effective and robust graph contrastive learning with graph autoencoding,” TKDE , 2023.

 

 
 [95] 
 
Z. Zhang, Q. Liu et al. , “Motif-based graph self-supervised learning for molecular property prediction,” in NeurIPS , 2021.

 

 
 [96] 
 
K.-D. Luong and A. Singh, “Fragment-based pretraining and finetuning on molecular graphs,” in NeurIPS , 2023.

 

 
 [97] 
 
E. Inae, G. Liu et al. , “Motif-aware attribute masking for molecular graph pre-training,” in LoG , 2024.

 

 
 [98] 
 
P. Yan, K. Song et al. , “Empowering dual-level graph self-supervised pretraining with motif discovery,” in AAAI , 2024.

 

 
 [99] 
 
L. Sun, Z. Huang et al. , “Motif-aware Riemannian graph neural network with generative-contrastive learning,” in AAAI , 2024.

 

 
 [100] 
 
S. Zhang, Z. Hu et al. , “Motif-driven contrastive learning of graph representations,” TKDE , 2024.

 

 
 [101] 
 
Y. Wu, L. Wang et al. , “Graph contrastive learning with cohesive subgraph awareness,” in WWW , 2024.

 

 
 [102] 
 
M. Xu, H. Wang et al. , “Self-supervised graph-level representation learning with local and global structure,” in ICML , 2021.

 

 
 [103] 
 
W. Li, C. Wang et al. , “HomoGCL: Rethinking homophily in graph contrastive learning,” in KDD , 2023.

 

 
 [104] 
 
Q. Wen, M. Ju et al. , “From coarse to fine: Enable comprehensive graph self-supervised learning with multi-granular semantic ensemble,” in ICML , 2024.

 

 
 [105] 
 
W. Shiao, U. S. Saini et al. , “CARL-G: Clustering-accelerated representation learning on graphs,” in KDD , 2023.

 

 
 [106] 
 
M. Chen, B. Wang et al. , “Deep contrastive graph learning with clustering-oriented guidance,” in AAAI , 2024.

 

 
 [107] 
 
J. Li, J. Yu et al. , “Dirichlet graph variational autoencoder,” in NeurIPS , 2020.

 

 
 [108] 
 
J. Li, M. Liu et al. , “Mask-GVAE: Blind denoising graphs via partition,” in WWW , 2021.

 

 
 [109] 
 
B. Li, B. Jing et al. , “Graph communal contrastive learning,” in WWW , 2022.

 

 
 [110] 
 
H. Chen, Z. Zhao et al. , “CSGCL: Community-strength-enhanced graph contrastive learning,” in IJCAI , 2023.

 

 
 [111] 
 
S. Zhang, W. Yang et al. , “StructComp: Substituting propagation with structural compression in training graph contrastive learning,” in ICLR , 2024.

 

 
 [112] 
 
Y. You, T. Chen et al. , “Graph contrastive learning with augmentations,” in NeurIPS , 2020.

 

 
 [113] 
 
——, “Graph contrastive learning automated,” in ICML , 2021.

 

 
 [114] 
 
S. Suresh, P. Li et al. , “Adversarial graph augmentation to improve graph contrastive learning,” in NeurIPS , 2021.

 

 
 [115] 
 
J. Xia, L. Wu et al. , “SimGRACE: A simple framework for graph contrastive learning without data augmentation,” in WWW , 2022.

 

 
 [116] 
 
F. Sun, J. Hoffmann et al. , “InfoGraph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization,” in ICLR , 2020.

 

 
 [117] 
 
K. Hassani and A. H. Khasahmadi, “Contrastive multi-view representation learning on graphs,” in ICML , 2020.

 

 
 [118] 
 
J. Xu, Y. Yang et al. , “Unsupervised adversarially robust representation learning on graphs,” in AAAI , 2022.

 

 
 [119] 
 
Y. Zheng, S. Pan et al. , “Rethinking and scaling up graph contrastive learning: An extremely efficient approach with group discrimination,” in NeurIPS , 2022.

 

 
 [120] 
 
D. Kim, J. Baek et al. , “Graph self-supervised learning with accurate discrepancy learning,” in NeurIPS , 2022.

 

 
 [121] 
 
H. Yang, H. Chen et al. , “Generating counterfactual hard negative samples for graph contrastive learning,” in WWW , 2023.

 

 
 [122] 
 
L. Lin, J. Chen et al. , “Spectral augmentation for self-supervised learning on graphs,” in ICLR , 2023.

 

 
 [123] 
 
N. Navarin, D. V. Tran et al. , “Pre-training graph neural networks with kernels,” CoRR , 2018.

 

 
 [124] 
 
J. Li, Y. Jin et al. , “Hierarchical topology isomorphism expertise embedded graph contrastive learning,” in AAAI , 2024.

 

 
 [125] 
 
J. Liu, M. Yang et al. , “Enhancing hyperbolic graph embeddings via contrastive learning,” in NIPS Workshop (SSL) , 2021.

 

 
 [126] 
 
L. Sun, Z. Zhang et al. , “A self-supervised mixed-curvature graph neural network,” in AAAI , 2022.

 

 
 [127] 
 
G. Skenderi, H. Li et al. , “Graph-level representation learning with joint-embedding predictive architectures,” TMLR , 2025.

 

 
 [128] 
 
R. Gong, Z. Jiang et al. , “Graph representation learning in hyperbolic space via dual-masked,” in COLING , 2025.

 

 
 [129] 
 
L. Sun, Z. Huang et al. , “RiemannGFM: Learning a graph foundation model from structural geometry,” in WWW , 2025.

 

 
 [130] 
 
J. Devlin, M.-W. Chang et al. , “BERT: Pre-training of deep bidirectional transformers for language understanding,” in NAACL , 2019.

 

 
 [131] 
 
K. He, X. Chen et al. , “Masked autoencoders are scalable vision learners,” in CVPR , 2022.

 

 
 [132] 
 
Y. He, Y. Sui et al. , “UniGraph: Learning a unified cross-domain foundation model for text-attributed graphs,” CoRR , 2024.

 

 
 [133] 
 
K. He, H. Fan et al. , “Momentum contrast for unsupervised visual representation learning,” in CVPR , 2020.

 

 
 [134] 
 
T. Chen, S. Kornblith et al. , “A simple framework for contrastive learning of visual representations,” in ICML , 2020.

 

 
 [135] 
 
M. I. Belghazi, A. Baratin et al. , “Mutual information neural estimation,” in ICML , 2018.

 

 
 [136] 
 
S. Nowozin, B. Cseke et al. , “f-GAN: Training generative neural samplers using variational divergence minimization,” in NIPS , 2016.

 

 
 [137] 
 
A. v. d. Oord, Y. Li et al. , “Representation learning with contrastive predictive coding,” CoRR , 2018.

 

 
 [138] 
 
F. Schroff, D. Kalenichenko et al. , “FaceNet: A unified embedding for face recognition and clustering,” in CVPR , 2015.

 

 
 [139] 
 
J.-B. Grill, F. Strub et al. , “Bootstrap Your Own Latent - a new approach to self-supervised learning,” in NeurIPS , 2020.

 

 
 [140] 
 
S. Rendle, C. Freudenthaler et al. , “BPR: Bayesian personalized ranking from implicit feedback,” in UAI , 2009.

 

 
 [141] 
 
J. Z. HaoChen, C. Wei et al. , “Provable guarantees for self-supervised deep learning with spectral contrastive loss,” in NeurIPS , 2023.

 

 
 [142] 
 
Z. Liu, X. Yu et al. , “GraphPrompt: Unifying pre-training and downstream tasks for graph neural networks,” in WWW , 2023.

 

 
 [143] 
 
Y. Yan, P. Zhang et al. , “Inductive graph alignment prompt: Bridging the gap between graph pre-training and inductive fine-tuning from spectral perspective,” in WWW , 2024.

 

 
 [144] 
 
J. Zbontar, L. Jing et al. , “Barlow Twins: Self-supervised learning via redundancy reduction,” in ICML , 2021.

 

 
 [145] 
 
L. Page, S. Brin et al. , “The PageRank citation ranking: Bringing order to the web,” Wayback Machine , 1998.

 

 
 [146] 
 
D. J. Watts and S. H. Strogatz, “Collective dynamics of ‘small-world’ networks,” Nature , 1998.

 

 
 [147] 
 
B. Jin, W. Zhang et al. , “Patton: Language model pretraining on text-rich networks,” in ACL , 2023.

 

 
 [148] 
 
X. Yu, Y. Fang et al. , “HGPROMPT: Bridging homogeneous and heterogeneous graphs for few-shot prompt learning,” in AAAI , 2024.

 

 
 [149] 
 
Y. Ma, X. Liu et al. , “Is homophily a necessity for graph neural networks?” in ICLR , 2022.

 

 
 [150] 
 
B. Perozzi, R. Al-Rfou et al. , “DeepWalk: Online learning of social representations,” in KDD , 2014.

 

 
 [151] 
 
M. Belkin and P. Niyogi, “Laplacian eigenmaps and spectral techniques for embedding and clustering,” in NIPS , 2001.

 

 
 [152] 
 
Q. Huang, H. Ren et al. , “PRODIGY: Enabling in-context learning over graphs,” in NeurIPS , 2023.

 

 
 [153] 
 
X. Sun, H. Cheng et al. , “All in One: Multi-task prompting for graph neural networks,” in KDD , 2023.

 

 
 [154] 
 
H. Liu, J. Feng et al. , “One for All: Towards training one graph model for all classification tasks,” in ICLR , 2024.

 

 
 [155] 
 
L. Katz, “A new status index derived from sociometric analysis,” Psychometrika , 1953.

 

 
 [156] 
 
P. Jaccard, “The distribution of the flora in the alpine zone,” New Phytologist , 1912.

 

 
 [157] 
 
M. Chen, Z. Liu et al. , “ULTRA-DP: Unifying graph pre-training with multi-task graph dual prompt,” CoRR , 2023.

 

 
 [158] 
 
N. Brown, In silico medicinal chemistry: computational methods to support drug design . Royal Society of Chemistry, 2015.

 

 
 [159] 
 
P. Ertl, “An algorithm to identify functional groups in organic molecules,” Journal of Cheminformatics , 2017.

 

 
 [160] 
 
A. Coates and A. Y. Ng, “Learning feature representations with K-Means,” Neural Networks: Tricks of the Trade , 2012.

 

 
 [161] 
 
M. Caron, P. Bojanowski et al. , “Deep clustering for unsupervised learning of visual features,” in ECCV , 2018.

 

 
 [162] 
 
V. D. Blondel, J.-L. Guillaume et al. , “Fast unfolding of communities in large networks,” Journal of Statistical Mechanics: Theory and Experiment , 2008.

 

 
 [163] 
 
Y. Lu, X. Jiang et al. , “Learning to pre-train graph neural networks,” in AAAI , 2021.

 

 
 [164] 
 
Z. Wang, S. Di et al. , “Search to fine-tune pre-trained graph neural networks for graph-level tasks,” in ICDE , 2024.

 

 
 [165] 
 
Y. Cao, J. Xu et al. , “When to pre-train graph neural networks? from data generation perspective!” in KDD , 2023.

 

 
 [166] 
 
Y. Sun, Q. Zhu et al. , “Fine-tuning graph neural networks by preserving graph generative patterns,” in AAAI , 2024.

 

 
 [167] 
 
Y. Zhu, Y. Wang et al. , “GraphControl: Adding conditional control to universal graph pre-trained models for graph domain transfer learning,” in WWW , 2024.

 

 
 [168] 
 
Z. Wang, Z. Zhang et al. , “GFT: Graph foundation model with transferable tree vocabulary,” in NeurIPS , 2024.

 

 
 [169] 
 
X. Han, Z. Huang et al. , “Adaptive transfer learning on graph neural networks,” in KDD , 2021.

 

 
 [170] 
 
J. Zhang, X. Xiao et al. , “Fine-tuning graph neural networks via graph topology induced optimal transport,” in IJCAI , 2022.

 

 
 [171] 
 
R. Huang, J. Xu et al. , “Measuring task similarity and its implication in fine-tuning graph neural networks,” in AAAI , 2024.

 

 
 [172] 
 
S. Li, X. Han et al. , “AdapterGNN: Parameter-efficient fine-tuning improves generalization in GNNs,” in AAAI , 2024.

 

 
 [173] 
 
A. Gui, J. Ye et al. , “G-Adapter: Towards structure-aware parameter-efficient transfer learning for graph transformer networks,” in AAAI , 2024.

 

 
 [174] 
 
Z. Zhang, M. Zhang et al. , “Endowing pre-trained graph models with provable fairness,” in WWW , 2024.

 

 
 [175] 
 
Y. Mo, R. Yu et al. , “HG-Adapter: Improving pre-trained heterogeneous graph neural networks with dual adapters,” in ICLR , 2025.

 

 
 [176] 
 
Q. Chen, L. Wang et al. , “DAGPrompT: Pushing the limits of graph prompting with a distribution-aware graph prompt tuning approach,” in WWW , 2025.

 

 
 [177] 
 
T. Fang, Y. Zhang et al. , “Universal prompt tuning for graph neural networks,” in NeurIPS , 2023.

 

 
 [178] 
 
J. Li, J. Li et al. , “Instance-aware graph prompt learning,” TMLR , 2025.

 

 
 [179] 
 
R. Shirkavand and H. Huang, “Deep prompt tuning for graph transformers,” CoRR , 2023.

 

 
 [180] 
 
Q. Ge, Z. Zhao et al. , “PSP: Pre-training and structure prompt tuning for graph neural networks,” in ECML-PKDD , 2024.

 

 
 [181] 
 
Z. Tan, R. Guo et al. , “Virtual node tuning for few-shot node classification,” in KDD , 2023.

 

 
 [182] 
 
Y. Zhu, J. Guo et al. , “SGL-PT: A strong graph learner with graph prompt tuning,” CoRR , 2023.

 

 
 [183] 
 
Y. Ma, N. Yan et al. , “HetGPT: Harnessing the power of prompt tuning in pre-trained heterogeneous graph neural networks,” in WWW , 2024.

 

 
 [184] 
 
W. Jiang, W. Wu et al. , “Killing two birds with one stone: Cross-modal reinforced prompting for graph and language tasks,” in KDD , 2024.

 

 
 [185] 
 
X. Fu, Y. He et al. , “Edge prompt tuning for graph neural networks,” in ICLR , 2025.

 

 
 [186] 
 
Z. Li, M. Lin et al. , “Fairness-aware prompt tuning for graph neural networks,” in WWW , 2025.

 

 
 [187] 
 
X. Yu, Z. Liu et al. , “Generalized graph prompt: Toward a unification of pre-training and downstream tasks on graphs,” TKDE , 2024.

 

 
 [188] 
 
J. Lee, W. Yang et al. , “Subgraph-level universal prompt tuning,” CoRR , 2024.

 

 
 [189] 
 
X. Yu, J. Zhang et al. , “Non-homophilic graph pre-training and prompt learning,” in KDD , 2025.

 

 
 [190] 
 
X. Yu, C. Zhou et al. , “GCoT: Chain-of-thought prompt learning for graphs,” CoRR , 2025.

 

 
 [191] 
 
——, “MultiGPrompt for multi-task pre-training and prompting on graphs,” in WWW , 2024.

 

 
 [192] 
 
H. Zhao, B. Yang et al. , “Pre-training and prompting for few-shot node classification on text-attributed graphs,” in KDD , 2024.

 

 
 [193] 
 
Y. Yang, L. Xia et al. , “GraphPro: Graph pre-training and prompt learning for recommendation,” in WWW , 2024.

 

 
 [194] 
 
J. Wang, Z. Deng et al. , “A novel prompt tuning for graph transformers: Tailoring prompts to graph topologies,” in KDD , 2024.

 

 
 [195] 
 
N. Houlsby, A. Giurgiu et al. , “Parameter-efficient transfer learning for NLP,” in ICML , 2019.

 

 
 [196] 
 
E. Hu, Y. Shen et al. , “LoRA: Low-rank adaptation of large language models,” in ICLR , 2022.

 

 
 [197] 
 
X. Sun, J. Zhang et al. , “Graph prompt learning: A comprehensive survey and beyond,” CoRR , 2023.

 

 
 [198] 
 
T. Zou, L. Yu et al. , “Pretraining language models with text-attributed heterogeneous graphs,” in EMNLP Findings , 2023.

 

 
 [199] 
 
W. Shang, X. Zhu et al. , “Path-LLM: A shortest-path-based llm learning for unified graph representation,” CoRR , 2024.

 

 
 [200] 
 
H. Xie, D. Zheng et al. , “Graph-aware language model pre-training on a large graph corpus can help multiple graph applications,” in KDD , 2023.

 

 
 [201] 
 
Z. Wen and Y. Fang, “Augmenting low-resource text classification with graph-grounded pre-training and prompting,” in SIGIR , 2023.

 

 
 [202] 
 
Y. Li, K. Ding et al. , “GRENADE: Graph-centric language model for self-supervised representation learning on text-attributed graphs,” in EMNLP Findings , 2023.

 

 
 [203] 
 
W. Brannon, S. Fulay et al. , “ConGraT: Self-supervised contrastive pretraining for joint graph and text embeddings,” in ACL Workshop (TextGraphs) , 2024.

 

 
 [204] 
 
J. Tang, Y. Yang et al. , “GraphGPT: Graph instruction tuning for large language models,” in SIGIR , 2024.

 

 
 [205] 
 
Y. Fang, D. Fan et al. , “UniGLM: Training one unified language model for text-attributed graphs,” in WSDM , 2025.

 

 
 [206] 
 
Y. Song, H. Mao et al. , “A pure transformer pretraining framework on text-attributed graphs,” in LoG , 2024.

 

 
 [207] 
 
Y. Xu, S. He et al. , “LLaSA: Large language and structured data assistant,” CoRR , 2024.

 

 
 [208] 
 
H. Wang, S. Feng et al. , “Can language models solve graph problems in natural language?” in NeurIPS , 2023.

 

 
 [209] 
 
Y. Hu, Z. Zhang et al. , “Beyond text: A deep dive into large language models’ ability on understanding graph data,” in NeurIPS Workshop (GLFrontiers) , 2023.

 

 
 [210] 
 
B. Fatemi, J. Halcrow et al. , “Talk like a graph: Encoding graphs for large language models,” in ICLR , 2024.

 

 
 [211] 
 
J. Huang, X. Zhang et al. , “Can LLMs effectively leverage graph structural information through prompts, and why?” TMLR , 2024.

 

 
 [212] 
 
X. Li, W. Chen et al. , “Can large language models analyze graphs like professionals? a benchmark, datasets and models,” in NeurIPS , 2024.

 

 
 [213] 
 
R. Li, J. Li et al. , “Similarity-based neighbor selection for graph llms,” CoRR , 2024.

 

 
 [214] 
 
K. Skianis, G. Nikolentzos et al. , “Graph reasoning with large language models via pseudo-code prompting,” CoRR , 2024.

 

 
 [215] 
 
Z. Hu, Y. Li et al. , “Let’s ask GNN: Empowering large language model for graph in-context learning,” in EMNLP Findings , 2024.

 

 
 [216] 
 
Z. Zhang, X. Wang et al. , “LLM4DyG: Can large language models solve spatial-temporal problems on dynamic graphs?” in KDD , 2024.

 

 
 [217] 
 
B. Perozzi, B. Fatemi et al. , “Let your graph do the talking: Encoding structured data for LLMs,” CoRR , 2024.

 

 
 [218] 
 
Z. Liu, X. He et al. , “Can we soft prompt LLMs for graph learning tasks?” in WWW (Short Papers) , 2024.

 

 
 [219] 
 
X. Huang, K. Han et al. , “Prompt-based node feature extractor for few-shot learning on text-attributed graphs,” CoRR , 2023.

 

 
 [220] 
 
——, “Can gnn be good adapter for llms?” in WWW , 2024.

 

 
 [221] 
 
Y. Qin, X. Wang et al. , “Disentangled representation learning with large language models for text-attributed graphs,” CoRR , 2023.

 

 
 [222] 
 
L. Kong, J. Feng et al. , “GOFA: A generative one-for-all model for joint graph language modeling,” in ICLR , 2025.

 

 
 [223] 
 
Z. Zhang, Y. Hu et al. , “TAGA: Text-attributed graph self-supervised learning by synergizing graph and text mutual transformations,” CoRR , 2024.

 

 
 [224] 
 
Y. Zhu, H. Shi et al. , “GraphCLIP: Enhancing transferability in graph foundation models for text-attributed graphs,” in WWW , 2025.

 

 
 [225] 
 
B. Pan, Z. Zhang et al. , “Distilling large language models for text-attributed graph learning,” in CIKM , 2024.

 

 
 [226] 
 
Y. Zhu, Y. Wang et al. , “Efficient tuning and inference for large language models on textual graphs,” in IJCAI , 2024.

 

 
 [227] 
 
D. Wang, Y. Zuo et al. , “LLMs as zero-shot graph learners: Alignment of GNN representations with LLM token embeddings,” in NeurIPS , 2024.

 

 
 [228] 
 
J. Tang, Y. Yang et al. , “HiGPT: Heterogeneous graph language model,” in KDD , 2024.

 

 
 [229] 
 
R. Chen, T. Zhao et al. , “LLaGA: Large language and graph assistant,” in ICML , 2024.

 

 
 [230] 
 
Y. Chen, Q. Yao et al. , “Improving molecule-language alignment with hierarchical graph tokenization,” CoRR , 2024.

 

 
 [231] 
 
M. Zhang, M. Sun et al. , “GraphTranslator: Aligning graph model to large language model for open-ended tasks,” in WWW , 2024.

 

 
 [232] 
 
Z. Chen, H. Mao et al. , “Exploring the potential of large language models (LLMs) in learning on graphs,” KDD Explorations Newsletter , 2024.

 

 
 [233] 
 
C. Qian, H. Tang et al. , “Can large language models empower molecular property prediction?” CoRR , 2023.

 

 
 [234] 
 
Z. Wang, S. Liu et al. , “Can LLMs convert graphs to text-attributed graphs?” CoRR , 2024.

 

 
 [235] 
 
Y. Fang, D. Fan et al. , “GAugLLM: Improving graph contrastive learning for text-attributed graphs with large language models,” in KDD , 2024.

 

 
 [236] 
 
J. Lu, Y. Liu et al. , “Scale-free graph-language models,” in ICLR , 2025.

 

 
 [237] 
 
Z. Chen, H. Mao et al. , “Label-free node classification on graphs with large language models (LLMs),” in ICLR , 2024.

 

 
 [238] 
 
Z. Zhang, X. Wang et al. , “Can large language models improve the adversarial robustness of graph neural networks?” in KDD , 2025.

 

 
 [239] 
 
T. Zhang, R. Yang et al. , “Leveraging large language models for effective label-free node classification in text-attributed graphs,” in SIGIR , 2025.

 

 
 [240] 
 
J. Yu, Y. Ren et al. , “Leveraging large language models for node generation in few-shot learning on text-attributed graphs,” in AAAI , 2025.

 

 
 [241] 
 
L. Xia, B. Kao et al. , “OpenGraph: Towards open graph foundation models,” in EMNLP Findings , 2024.

 

 
 [242] 
 
Y. Qiao, X. Ao et al. , “LOGIN: A large language model consulted graph neural network training framework,” in WSDM , 2025.

 

 
 [243] 
 
S. Sun, Y. Ren et al. , “Large language models as topological structure enhancers for text-attributed graphs,” CoRR , 2023.

 

 
 [244] 
 
J. Zhao, M. Qu et al. , “Learning on large-scale text-attributed graphs via variational inference,” in ICLR , 2023.

 

 
 [245] 
 
Z. Chai, T. Zhang et al. , “GraphLLM: Boosting graph reasoning ability of large language model,” CoRR , 2023.

 

 
 [246] 
 
R. Xue, X. Shen et al. , “Efficient large language models fine-tuning on graphs,” CoRR , 2023.

 

 
 [247] 
 
K. Duan, Q. Liu et al. , “SimTeG: A frustratingly simple approach improves textual graph learning,” CoRR , 2023.

 

 
 [248] 
 
Y. Tan, Z. Zhou et al. , “WalkLM: A uniform language model fine-tuning framework for attributed graph embedding,” in NeurIPS , 2023.

 

 
 [249] 
 
S. Ouyang, Y. Hu et al. , “GUNDAM: Aligning large language models with graph understanding,” CoRR , 2024.

 

 
 [250] 
 
S. Hu, G. Zou et al. , “Large language model meets graph neural network in knowledge distillation,” in AAAI , 2025.

 

 
 [251] 
 
J. Wang, J. Wu et al. , “InstructGraph: Boosting large language models via graph-centric instruction tuning and preference alignment,” in ACL Findings , 2024.

 

 
 [252] 
 
N. Chen, Y. Li et al. , “GraphWiz: An instruction-following language model for graph problems,” in KDD , 2024.

 

 
 [253] 
 
Z. Luo, X. Song et al. , “GraphInstruct: Empowering large language models with graph understanding and reasoning capability,” CoRR , 2024.

 

 
 [254] 
 
Z. Xu, K. Hassani et al. , “How to make LLMs strong node classifiers?” CoRR , 2024.

 

 
 [255] 
 
Q. Zhu, L. Zhang et al. , “HierPromptLM: A pure PLM-based framework for representation learning on heterogeneous text-rich networks,” CoRR , 2025.

 

 
 [256] 
 
C. Raffel, N. Shazeer et al. , “Exploring the limits of transfer learning with a unified text-to-text transformer,” JMLR , 2020.

 

 
 [257] 
 
A. Radford, K. Narasimhan et al. , “Improving language understanding by generative pre-training,” OpenAI , 2018.

 

 
 [258] 
 
H. Touvron, T. Lavril et al. , “LLaMA: Open and efficient foundation language models,” CoRR , 2023.

 

 
 [259] 
 
DeepSeek-AI, “Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” CoRR , 2025.

 

 
 [260] 
 
Y. Liu, M. Ott et al. , “RoBERTa: A robustly optimized bert pretraining approach,” CoRR , 2019.

 

 
 [261] 
 
M. Lewis, Y. Liu et al. , “BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in ACL , 2020.

 

 
 [262] 
 
A. Radford, J. W. Kim et al. , “Learning transferable visual models from natural language supervision,” in ICML , 2021.

 

 
 [263] 
 
Y. Li, Z. Li et al. , “A survey of graph meets large language model: Progress and future directions,” in IJCAI , 2024.

 

 
 [264] 
 
J. Zhao, L. Zhuo et al. , “GraphText: Graph reasoning in text space,” CoRR , 2023.

 

 
 [265] 
 
J. Wei, Y. Tay et al. , “Emergent abilities of large language models,” TMLR , 2022.

 

 
 [266] 
 
Z. Ji, N. Lee et al. , “Survey of hallucination in natural language generation,” CSUR , 2023.

 

 
 [267] 
 
W. Jin, X. Liu et al. , “Automated self-supervised learning for graphs,” in ICLR , 2022.

 

 
 [268] 
 
M. Ju, T. Zhao et al. , “Multi-task self-supervised graph neural networks enable stronger task generalization,” in ICLR , 2023.

 

 
 [269] 
 
T. Fang, W. Zhou et al. , “Exploring correlations of self-supervised tasks for graphs,” in ICML , 2024.

 

 
 [270] 
 
L. Wu, Y. Huang et al. , “Automated graph self-supervised learning via multi-teacher knowledge distillation,” CoRR , 2022.

 

 
 [271] 
 
T. Fan, L. Wu et al. , “Decoupling weighing and selecting for integrating multiple graph pre-training tasks,” in ICLR , 2024.

 

 
 [272] 
 
B. Jin, Y. Zhang et al. , “Heterformer: Transformer-based deep node representation learning on heterogeneous text-rich networks,” in KDD , 2023.

 

 
 [273] 
 
H. Duan, C. Xie et al. , “Reserving-masking-reconstruction model for self-supervised heterogeneous graph representation,” in KDD , 2024.

 

 
 [274] 
 
D. Hwang, J. Park et al. , “Self-supervised auxiliary learning with meta-paths for heterogeneous graphs,” in NeurIPS , 2020.

 

 
 [275] 
 
X. Wang, N. Liu et al. , “Self-supervised heterogeneous graph neural network with co-contrastive learning,” in KDD , 2021.

 

 
 [276] 
 
X. Jiang, T. Jia et al. , “Pre-training on large-scale heterogeneous graph,” in KDD , 2021.

 

 
 [277] 
 
Z. Sun, Z.-H. Deng et al. , “RotatE: Knowledge graph embedding by relational rotation in complex space,” in ICLR , 2019.

 

 
 [278] 
 
X. Wang, T. Gao et al. , “KEPLER: A unified model for knowledge embedding and pre-trained language representation,” TACL , 2021.

 

 
 [279] 
 
X. Liu, H. Hong et al. , “SelfKG: Self-supervised entity alignment in knowledge graphs,” in WWW , 2022.

 

 
 [280] 
 
R. Zhang, Y. Su et al. , “AutoAlign: fully automatic and effective knowledge graph alignment enabled by large language models,” TKDE , 2024.

 

 
 [281] 
 
Y. Tian, H. Song et al. , “Graph neural prompting with large language models,” in AAAI , 2024.

 

 
 [282] 
 
Z. Li, L. Xia et al. , “GPT-ST: generative pre-training of spatio-temporal graph neural networks,” in NeurIPS , 2023.

 

 
 [283] 
 
J. Hu, X. Liu et al. , “Prompt-based spatio-temporal graph transfer learning,” in CIKM , 2024.

 

 
 [284] 
 
S. Tian, R. Wu et al. , “Self-supervised representation learning on dynamic graphs,” in CIKM , 2021.

 

 
 [285] 
 
X. Yu, Z. Liu et al. , “Node-time conditional prompt learning in dynamic graphs,” in ICLR , 2025.

 

 
 [286] 
 
G. Lee, S. Y. Lee et al. , “VilLain: Self-supervised learning on homogeneous hypergraphs without features via virtual label propagation,” in WWW , 2024.

 

 
 [287] 
 
T. Wei, Y. You et al. , “Augmentations in hypergraph contrastive learning: Fabricated and generative,” in NeurIPS , 2022.

 

 
 [288] 
 
D. Lee and K. Shin, “I’m me, we’re us, and I’m us: Tri-directional contrastive learning on hypergraphs,” in AAAI , 2023.

 

 
 [289] 
 
S. Kim, S. Kang et al. , “HypeBoy: Generative self-supervised representation learning on hypergraphs,” in ICLR , 2024.

 

 
 [290] 
 
M. Lin, T. Xiao et al. , “Certifiably robust graph contrastive learning,” in NeurIPS , 2023.

 

 
 [291] 
 
Y. Qian, C. Zhang et al. , “Co-modality graph contrastive learning for imbalanced node classification,” in NeurIPS , 2022.

 

 
 [292] 
 
H. Ling, Z. Jiang et al. , “Learning fair graph representations via automated data augmentations,” in ICLR , 2023.

 

 
 [293] 
 
H. Zhang, J. Chen et al. , “Graph contrastive backdoor attacks,” in ICML , 2023.

 

 
 [294] 
 
X. Lyu, Y. Han et al. , “Cross-context backdoor attacks against graph prompt learning,” in KDD , 2024.

 

 
 [295] 
 
K. Guo, Z. Liu et al. , “Learning on graphs with large language models (LLMs): A deep dive into model robustness,” CoRR , 2024.

 

 
 [296] 
 
B. Jin, C. Xie et al. , “Graph chain-of-thought: Augmenting large language models by reasoning on graphs,” in ACL Findings , 2024.

 

 
 [297] 
 
L. Luo, Y.-F. Li et al. , “Reasoning on graphs: Faithful and interpretable large language model reasoning,” in ICLR , 2024.

 

 
 [298] 
 
B. Peng, Y. Zhu et al. , “Graph retrieval-augmented generation: A survey,” CoRR , 2024.

 

 
 [299] 
 
X. He, Y. Tian et al. , “G-Retriever: Retrieval-augmented generation for textual graph understanding and question answering,” in NeurIPS , 2024.

 

 
 [300] 
 
X. Jiang, R. Qiu et al. , “RAGraph: A general retrieval-augmented graph learning framework,” in NeurIPS , 2024.

 

 
 [301] 
 
B. Chen, Z. Guo et al. , “PathRAG: Pruning graph-based retrieval augmented generation with relational paths,” CoRR , 2025.

 

 
 [302] 
 
S. Wang, Y. Fang et al. , “ArchRAG: Attributed community-based hierarchical retrieval-augmented generation,” CoRR , 2025.

 

 
 
 
 

## Appendix A Summary

 
 We have listed all references in our survey in Tables  I – III .
Table  I summarizes self-supervised graph pre-training methods,
Table  II summarizes graph downstream tuning methods,
and Table  III summarizes GLMs.
Papers are arranged in strict chronological order determined by their earliest publication or preprinting time, indicated by the “Time” column.
For more information such as paper links and open-source code links, please refer to our GitHub list.

 
 
 TABLE I: 

Summary of self-supervised graph pre-training methods.

 
 
 
 
 

 
 
 
 
 Model 
 
 Time 
 
 
 Venue 
 
 
 
 Pre-training tasks 
 
 
 
 
 
 
 Graph knowledge 
 
 focused on 
 
 
 
 
 Downstream tasks 
 
 
 
 
 GAE; VGAE  [ 12 ] 
 
 Nov 2016 
 
 
 
 
 
 NIPS 
 
 Workshop 
 
 (BDL)’16 
 
 
 
 
 Link prediction 
 
 
 
 Links 
 
 
 
 Link prediction 
 
 
 
 
 GraphSAGE  [ 6 ] 
 
 Jun 2017 
 
 
 NIPS’17 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 Node classification 
 
 
 
 
 MGAE  [ 28 ] 
 
 Nov 2017 
 
 
 CIKM’17 
 
 
 
 Feature prediction 
 
 
 
 Node features 
 
 
 
 Graph partitioning 
 
 
 
 
 ARGA; ARVGA  [ 66 ] 
 
 Feb 2018 
 
 
 IJCAI’18 
 
 
 
 Link prediction 
 
 
 
 Links 
 
 
 
 
 
 
 Link prediction; 
 
 node clustering 
 
 
 
 
 
 DGI  [ 13 ] 
 
 Sept 2018 
 
 
 ICLR’19 
 
 
 
 Node-graph discrimination 
 
 
 
 Global structure 
 
 
 
 Node classification 
 
 
 
 
 KernelPred  [ 123 ] 
 
 Nov 2018 
 
 
 arXiv 
 
 
 
 Graph similarity prediction 
 
 
 
 Global structure 
 
 
 
 Graph classification 
 
 
 
 
 RotatE  [ 277 ] 
 
 Feb 2019 
 
 
 ICLR’19 
 
 
 
 Node instance discrimination 
 
 
 
 Node features; links 
 
 
 
 Link prediction 
 
 
 
 
 M3S  [ 26 ] 
 
 Feb 2019 
 
 
 AAAI’20 
 
 
 
 Node clustering 
 
 
 
 Clusters 
 
 
 
 Node classification 
 
 
 
 
 
 
 
 Hu et al.   [ 61 ] 
 
 (ScoreRank; 
 
 DenoisingRecon; 
 
 ClusterDetect) 
 
 
 May 2019 
 
 
 
 
 
 ICLR 
 
 Workshop 
 
 (RLGM)’19 
 
 
 
 
 
 
 
 Centrality ranking 
 
 / masked link prediction 
 
 / graph partitioning 
 
 
 
 
 
 
 
 Node properties; links; 
 
 clusters 
 
 
 
 
 Node classification 
 
 
 
 
 
 
 
 GNN-Pretrain  [ 32 ] 
 
 (AttrMask; 
 
 ContextPred) 
 
 
 May 2019 
 
 
 ICLR’20 
 
 
 
 
 
 
 Masked feature prediction 
 
 / edge feature prediction 
 
 / contextual subgraph discrimination 
 
 
 
 
 
 
 
 Node features; links; 
 
 context 
 
 
 
 
 
 
 
 Graph classification; 
 
 biological function prediction 
 
 
 
 
 
 InfoGraph  [ 116 ] 
 
 Jul 2019 
 
 
 ICLR’20 
 
 
 
 Node-graph discrimination 
 
 
 
 
 
 
 Context; 
 
 global structure 
 
 
 
 
 Graph classification 
 
 
 
 
 GALA  [ 29 ] 
 
 Aug 2019 
 
 
 ICCV’19 
 
 
 
 Feature prediction 
 
 
 
 Node features 
 
 
 
 
 
 
 Node clustering; 
 
 link prediction; etc. 
 
 
 
 
 
 SIG-VAE  [ 67 ] 
 
 Aug 2019 
 
 
 NeurIPS’19 
 
 
 
 Link prediction 
 
 
 
 Links 
 
 
 
 
 
 
 Node classification; 
 
 link prediction; 
 
 node clustering; 
 
 graph generation 
 
 
 
 
 
 Graph-Bert  [ 30 ] 
 
 Jan 2020 
 
 
 arXiv 
 
 
 
 
 
 
 Feature prediction; 
 
 similarity prediction 
 
 
 
 
 
 
 
 Node features; context; 
 
 long-range similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 GMI  [ 31 ] 
 
 Feb 2020 
 
 
 WWW’20 
 
 
 
 Feature prediction 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 S 2 GRL  [ 88 ] 
 
 Mar 2020 
 
 
 
 
 
 Information 
 
 Sciences’22 
 
 
 
 
 Similarity prediction 
 
 
 
 Long-range similarities 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 link prediction 
 
 
 
 
 
 GRACE  [ 42 ] 
 
 Jun 2020 
 
 
 
 
 
 ICML 
 
 Workshop 
 
 (GRL+)’20 
 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 MVGRL  [ 117 ] 
 
 Jun 2020 
 
 
 ICML’20 
 
 
 
 Node-graph discrimination 
 
 
 
 Global structure 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 
 
 
 
 
 SS-GCN  [ 33 ] 
 
 (GraphComp; 
 
 NodeCluster; 
 
 GraphPar) 
 
 
 Jun 2020 
 
 
 ICML’20 
 
 
 
 
 
 
 Masked feature prediction 
 
 / node clustering 
 
 / graph partitioning 
 
 
 
 
 Node features; clusters 
 
 
 
 Node classification 
 
 
 
 
 GCC  [ 80 ] 
 
 Jun 2020 
 
 
 KDD’20 
 
 
 
 Contextual subgraph discrimination 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 similarity search 
 
 
 
 
 
 
 
 
 SelfTask  [ 34 ] 
 
 (AttributeMask; 
 
 NodeProperty; 
 
 EdgeMask; 
 
 PairwiseDistance; 
 
 PairwiseAttrSim; 
 
 Distance2Clusters) 
 
 
 Jun 2020 
 
 
 arXiv 
 
 
 
 
 
 
 Masked feature prediction 
 
 / property prediction 
 
 / masked link prediction 
 
 / similarity prediction 
 
 / graph partitioning 
 
 
 
 
 
 
 
 Node features; 
 
 node properties; links; 
 
 long-range similarities; 
 
 clusters 
 
 
 
 
 Node classification 
 
 
 
 
 GROVER  [ 7 ] 
 
 Jun 2020 
 
 
 NeurIPS’20 
 
 
 
 
 
 
 Motif prediction; 
 
 contextual property prediction 
 
 
 
 
 Context; motifs 
 
 
 
 
 
 
 Graph classification; 
 
 graph regression 
 
 
 
 
 
 GPT-GNN  [ 37 ] 
 
 Jun 2020 
 
 
 KDD’20 
 
 
 
 
 
 
 Masked feature prediction; 
 
 masked link prediction 
 
 
 
 
 Node features; links 
 
 
 
 
 
 
 Node classification; 
 
 link prediction; 
 
 edge classification 
 
 
 
 
 
 AGE  [ 89 ] 
 
 Jul 2020 
 
 
 KDD’20 
 
 
 
 Similarity prediction 
 
 
 
 Long-range similarities 
 
 
 
 
 
 
 Node clustering; 
 
 link prediction 
 
 
 
 
 
 AM-GCN  [ 90 ] 
 
 Jul 2020 
 
 
 KDD’20 
 
 
 
 Similarity graph alignment 
 
 
 
 Long-range similarities 
 
 
 
 Node classification 
 
 
 
 
 SELAR  [ 274 ] 
 
 Jul 2020 
 
 
 NeurIPS’20 
 
 
 
 Link prediction 
 
 
 
 Links 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 link prediction 
 
 
 
 
 
 EGI  [ 75 ] 
 
 Sept 2020 
 
 
 NeurIPS’21 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 
 
 
 Link prediction; 
 
 structural role identification 
 
 
 
 
 
 Subg-Con  [ 84 ] 
 
 Sept 2020 
 
 
 ICDM’20 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 Node classification 
 
 
 

 
 
 
 TABLE I: 

Summary of self-supervised graph pre-training methods (continued).

 
 
 
 
 

 
 
 
 
 Model 
 
 Time 
 
 
 Venue 
 
 
 
 Pre-training tasks 
 
 
 
 
 
 
 Graph knowledge 
 
 focused on 
 
 
 
 
 Downstream tasks 
 
 
 
 
 CG 3   [ 71 ] 
 
 Sept 2020 
 
 
 AAAI’21 
 
 
 
 
 
 
 Node instance discrimination; 
 
 link prediction 
 
 
 
 
 Node features; links 
 
 
 
 Node classification 
 
 
 
 
 DGVAE  [ 107 ] 
 
 Oct 2020 
 
 
 NeurIPS’20 
 
 
 
 Partition-conditioned link prediction 
 
 
 
 Links; clusters 
 
 
 
 
 
 
 Node clustering; 
 
 graph generation 
 
 
 
 
 
 CommDGI  [ 27 ] 
 
 Oct 2020 
 
 
 CIKM’20 
 
 
 
 
 
 
 Cluster-based discrimination; 
 
 graph partitioning 
 
 
 
 
 Clusters 
 
 
 
 Node clustering 
 
 
 
 
 GraphCL  [ 112 ] 
 
 Oct 2020 
 
 
 NeurIPS’20 
 
 
 
 Graph instance discrimination 
 
 
 
 Global structure 
 
 
 
 Graph classification 
 
 
 
 
 GCA  [ 43 ] 
 
 Oct 2020 
 
 
 WWW’21 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 GRV  [ 118 ] 
 
 Dec 2020 
 
 
 AAAI’22 
 
 
 
 Node-graph discrimination 
 
 
 
 Global structure 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 link prediction 
 
 
 
 
 
 MICRO-Graph  [ 100 ] 
 
 Dec 2020 
 
 
 TKDE’24 
 
 
 
 
 
 
 Motif-based discrimination; 
 
 graph partitioning 
 
 
 
 
 Motifs; clusters 
 
 
 
 Graph classification 
 
 
 
 
 Mask-GVAE  [ 108 ] 
 
 Feb 2021 
 
 
 WWW’21 
 
 
 
 
 
 
 Graph partitioning; 
 
 partition-conditioned link prediction 
 
 
 
 
 Clusters 
 
 
 
 Node clustering; etc. 
 
 
 
 
 SLAPS  [ 36 ] 
 
 Feb 2021 
 
 
 NeurIPS’21 
 
 
 
 Masked feature prediction 
 
 
 
 
 
 
 Node features; 
 
 long-range similarities 
 
 
 
 
 Node classification; etc. 
 
 
 
 
 BGRL  [ 48 ] 
 
 Feb 2021 
 
 
 ICLR’22 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 PIGAE  [ 64 ] 
 
 Apr 2021 
 
 
 NeurIPS’21 
 
 
 
 
 
 
 Node order matching; 
 
 link prediction; 
 
 edge feature prediction 
 
 
 
 
 Node properties; links 
 
 
 
 Graph classification 
 
 
 
 
 VICReg  [ 59 ] 
 
 May 2021 
 
 
 ICLR’22 
 
 
 
 
 
 
 Node instance discrimination; 
 
 dimension discrimination 
 
 
 
 
 Node features 
 
 
 
 Node classification; etc. 
 
 
 
 
 MERIT  [ 45 ] 
 
 May 2021 
 
 
 IJCAI’21 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 MVMI-FT  [ 93 ] 
 
 May 2021 
 
 
 TKDE’23 
 
 
 
 
 
 
 Link prediction; 
 
 node-graph discrimination; 
 
 similarity graph alignment 
 
 
 
 
 
 
 
 Links; 
 
 long-range similarities; 
 
 global structure 
 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 HeCo  [ 275 ] 
 
 May 2021 
 
 
 KDD’21 
 
 
 
 Node instance discrimination 
 
 
 
 Node features; motifs 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 node clustering 
 
 
 
 
 
 G-BT  [ 57 ] 
 
 Jun 2021 
 
 
 KBS’22 
 
 
 
 Dimension discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 Graph-MLP  [ 82 ] 
 
 Jun 2021 
 
 
 arXiv 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 Node classification 
 
 
 
 
 GraphLoG  [ 102 ] 
 
 Jun 2021 
 
 
 ICML’21 
 
 
 
 
 
 
 Contextual subgraph discrimination; 
 
 node clustering; 
 
 graph instance discrimination 
 
 
 
 
 
 
 
 Context; clusters; 
 
 global structure 
 
 
 
 
 
 
 
 Graph classification; 
 
 biological function prediction 
 
 
 
 
 
 AutoSSL  [ 267 ] 
 
 Jun 2021 
 
 
 ICLR’22 
 
 
 
 Miscellaneous 
 
 
 
 – 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 JOAO  [ 113 ] 
 
 Jun 2021 
 
 
 ICML’21 
 
 
 
 Graph instance discrimination 
 
 
 
 Global structure 
 
 
 
 Graph classification 
 
 
 
 
 AD-GCL  [ 114 ] 
 
 Jun 2021 
 
 
 NeurIPS’21 
 
 
 
 Graph instance discrimination 
 
 
 
 Global structure 
 
 
 
 Graph classification 
 
 
 
 
 CCA-SSG  [ 58 ] 
 
 Jun 2021 
 
 
 NeurIPS’21 
 
 
 
 
 
 
 Node instance discrimination; 
 
 dimension discrimination 
 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 PT-HGNN  [ 276 ] 
 
 Aug 2021 
 
 
 KDD’21 
 
 
 
 Context discrimination 
 
 
 
 
 
 
 Node features; context; 
 
 motifs 
 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 link prediction 
 
 
 
 
 
 MGSSL  [ 95 ] 
 
 Oct 2021 
 
 
 NeurIPS’21 
 
 
 
 
 
 
 Masked feature prediction; 
 
 masked edge feature prediction; 
 
 motif prediction 
 
 
 
 
 
 
 
 Node features; links; 
 
 motifs 
 
 
 
 
 Graph classification 
 
 
 
 
 ProGCL  [ 44 ] 
 
 Oct 2021 
 
 
 ICML’22 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 gCooL  [ 109 ] 
 
 Oct 2021 
 
 
 WWW’22 
 
 
 
 Partition-based discrimination 
 
 
 
 Clusters 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 DDGCL  [ 284 ] 
 
 Oct 2021 
 
 
 CIKM’21 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 
 
 
 (Temporal) 
 
 node classification; 
 
 link prediction 
 
 
 
 
 
 AFGRL  [ 85 ] 
 
 Dec 2021 
 
 
 AAAI’22 
 
 
 
 Context discrimination 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 similarity search 
 
 
 
 
 
 SelfMGNN  [ 126 ] 
 
 Dec 2021 
 
 
 AAAI’22 
 
 
 
 Cross-manifold discrimination 
 
 
 
 Manifolds 
 
 
 
 Node classification 
 
 
 
 
 SimGCL  [ 51 ] 
 
 Dec 2021 
 
 
 SIGIR’22 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Recommendation 
 
 
 
 
 S2GAE  [ 69 ] 
 
 Jan 2022 
 
 
 WSDM’23 
 
 
 
 Masked link prediction 
 
 
 
 Links 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 link prediction 
 
 
 
 
 
 COLES  [ 77 ] 
 
 Jan 2022 
 
 
 NeurIPS’21 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 DSGC  [ 24 ] 
 
 Jan 2022 
 
 
 WWW’22 
 
 
 
 Cross-manifold discrimination 
 
 
 
 Manifolds 
 
 
 
 Graph classification 
 
 
 
 
 HGCL  [ 125 ] 
 
 Jan 2022 
 
 
 
 
 
 NeurIPS 
 
 Workshop 
 
 (SSL)’21 
 
 
 
 
 Cross-manifold discrimination 
 
 
 
 Manifolds 
 
 
 
 Node classification 
 
 
 
 
 D-SLA  [ 120 ] 
 
 Feb 2022 
 
 
 NeurIPS’22 
 
 
 
 
 
 
 Group discrimination; 
 
 graph similarity prediction 
 
 
 
 
 Global structure 
 
 
 
 
 
 
 Graph classification; 
 
 link prediction 
 
 
 
 
 
 SimGRACE  [ 115 ] 
 
 Feb 2022 
 
 
 WWW’22 
 
 
 
 Graph instance discrimination 
 
 
 
 Global structure 
 
 
 
 Graph classification 
 
 
 
 
 LaGraph  [ 35 ] 
 
 Feb 2022 
 
 
 ICML’22 
 
 
 
 
 
 
 Masked feature prediction; 
 
 node instance discrimination 
 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 
 
 S 3 -CL  [ 81 ] 
 
 Feb 2022 
 
 
 AAAI’23 
 
 
 
 
 
 
 Contextual subgraph discrimination; 
 
 cluster-based discrimination 
 
 
 
 
 Context; clusters 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 NWR-GAE  [ 62 ] 
 
 Feb 2022 
 
 
 ICLR’22 
 
 
 
 
 
 
 Property prediction; 
 
 context feature prediction 
 
 
 
 
 
 
 
 Node properties; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 structural role identification 
 
 
 
 
 
 N2N  [ 83 ] 
 
 Mar 2022 
 
 
 CVPR’22 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 Node classification 
 
 
 

 
 
 
 TABLE I: 

Summary of self-supervised graph pre-training methods (continued).

 
 
 
 
 

 
 
 
 
 Model 
 
 Time 
 
 
 Venue 
 
 
 
 Pre-training tasks 
 
 
 
 
 
 
 Graph knowledge 
 
 focused on 
 
 
 
 
 Downstream tasks 
 
 
 
 
 SuperGAT  [ 72 ] 
 
 Apr 2022 
 
 
 ICLR’21 
 
 
 
 Link prediction 
 
 
 
 Links 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 MaskGAE  [ 63 ] 
 
 May 2022 
 
 
 KDD’23 
 
 
 
 
 
 
 Property prediction; 
 
 masked link prediction 
 
 
 
 
 
 
 
 Node properties; links; 
 
 long-range similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 Heterformer  [ 272 ] 
 
 May 2022 
 
 
 KDD’23 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 node clustering; 
 
 link prediction 
 
 
 
 
 
 GraphMAE  [ 23 ] 
 
 May 2022 
 
 
 KDD’22 
 
 
 
 Masked feature prediction 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 
 
 ImGCL  [ 53 ] 
 
 May 2022 
 
 
 AAAI’23 
 
 
 
 Node instance discrimination 
 
 
 
 
 
 
 Node features; 
 
 node properties; 
 
 clusters 
 
 
 
 
 Node classification 
 
 
 
 
 GGD  [ 119 ] 
 
 Jun 2022 
 
 
 NeurIPS’22 
 
 
 
 Group discrimination 
 
 
 
 Global structure 
 
 
 
 Node classification 
 
 
 
 
 COSTA  [ 46 ] 
 
 Jun 2022 
 
 
 KDD’22 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 TriCL  [ 288 ] 
 
 Jun 2022 
 
 
 AAAI’23 
 
 
 
 
 
 
 Node instance discrimination; 
 
 context discrimination 
 
 
 
 
 
 
 
 Node features; links; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 SUGRL  [ 50 ] 
 
 Jun 2022 
 
 
 AAAI’22 
 
 
 
 
 
 
 Node instance discrimination; 
 
 context discrimination 
 
 
 
 
 Node features; context 
 
 
 
 Node classification 
 
 
 
 
 CGC  [ 121 ] 
 
 Jul 2022 
 
 
 WWW’23 
 
 
 
 Graph instance discrimination 
 
 
 
 Global structure 
 
 
 
 Graph classification 
 
 
 
 
 HGMAE  [ 39 ] 
 
 Aug 2022 
 
 
 AAAI’23 
 
 
 
 
 
 
 Feature prediction; 
 
 masked feature prediction; 
 
 masked link prediction 
 
 
 
 
 Node features; links 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 node clustering 
 
 
 
 
 
 SPAN  [ 122 ] 
 
 Oct 2022 
 
 
 ICLR’23 
 
 
 
 Node-graph discrimination 
 
 
 
 
 
 
 Global structure; 
 
 spectrum 
 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 graph regression 
 
 
 
 
 
 ParetoGNN  [ 268 ] 
 
 Oct 2022 
 
 
 ICLR’23 
 
 
 
 Miscellaneous 
 
 
 
 – 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 graph partitioning; 
 
 link prediction 
 
 
 
 
 
 AGSSL  [ 270 ] 
 
 Oct 2022 
 
 
 arXiv 
 
 
 
 Miscellaneous 
 
 
 
 – 
 
 
 
 Node classification 
 
 
 
 
 GRADE  [ 54 ] 
 
 Oct 2022 
 
 
 NeurIPS’22 
 
 
 
 Node instance discrimination 
 
 
 
 
 
 
 Node features; 
 
 node properties; 
 
 long-range similarities 
 
 
 
 
 Node classification 
 
 
 
 
 HyperGCL  [ 287 ] 
 
 Oct 2022 
 
 
 NeurIPS’22 
 
 
 
 
 
 
 Feature prediction; 
 
 node instance discrimination; 
 
 link prediction 
 
 
 
 
 Node features; links 
 
 
 
 
 
 
 Node classification; 
 
 (hyper-)link prediction 
 
 
 
 
 
 HGRL  [ 86 ] 
 
 Oct 2022 
 
 
 CIKM’22 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 CGI  [ 47 ] 
 
 Nov 2022 
 
 
 NeurIPS’22 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Recommendation 
 
 
 
 
 GLEN  [ 78 ] 
 
 Nov 2022 
 
 
 NeurIPS’22 
 
 
 
 Context discrimination 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 CM-GCL  [ 291 ] 
 
 Nov 2022 
 
 
 NeurIPS’22 
 
 
 
 
 
 
 Node instance discrimination; 
 
 similarity-based discrimination 
 
 
 
 
 
 
 
 Node features; 
 
 long-range similarities 
 
 
 
 
 Node classification 
 
 
 
 
 SP-GCL  [ 55 ] 
 
 Nov 2022 
 
 
 TMLR’23 
 
 
 
 Node instance discrimination 
 
 
 
 Node features; context 
 
 
 
 Node classification 
 
 
 
 
 Mole-BERT  [ 40 ] 
 
 Feb 2023 
 
 
 ICLR’23 
 
 
 
 
 
 
 Masked feature prediction; 
 
 graph instance discrimination 
 
 
 
 
 
 
 
 Node features; 
 
 global structure 
 
 
 
 
 
 
 
 Graph classification; 
 
 graph regression 
 
 
 
 
 
 Graphair  [ 292 ] 
 
 Feb 2023 
 
 
 ICLR’23 
 
 
 
 
 
 
 Masked feature prediction; 
 
 masked link prediction; 
 
 node instance discrimination 
 
 
 
 
 Node features; links 
 
 
 
 Node classification 
 
 
 
 
 LightGCL  [ 52 ] 
 
 Feb 2023 
 
 
 ICLR’23 
 
 
 
 Node instance discrimination 
 
 
 
 Node features 
 
 
 
 Recommendation 
 
 
 
 
 GraphMAE2  [ 38 ] 
 
 Apr 2023 
 
 
 WWW’23 
 
 
 
 
 
 
 Masked feature prediction; 
 
 node instance discrimination 
 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 CSGCL  [ 110 ] 
 
 May 2023 
 
 
 IJCAI’23 
 
 
 
 Partition-based discrimination 
 
 
 
 Clusters 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 link prediction 
 
 
 
 
 
 CARL-G  [ 105 ] 
 
 Jun 2023 
 
 
 KDD’23 
 
 
 
 Node clustering 
 
 
 
 Clusters 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 similarity search 
 
 
 
 
 
 HomoGCL  [ 103 ] 
 
 Jun 2023 
 
 
 KDD’23 
 
 
 
 
 
 
 Node clustering; 
 
 cluster-based discrimination 
 
 
 
 
 Clusters 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 AEGCL  [ 94 ] 
 
 Jun 2023 
 
 
 TKDE’23 
 
 
 
 
 
 
 Feature prediction; 
 
 link prediction; 
 
 similarity graph alignment 
 
 
 
 
 
 
 
 Node features; links; 
 
 long-range similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 link prediction 
 
 
 
 
 
 DLR-GAE  [ 91 ] 
 
 Jun 2023 
 
 
 AAAI’23 
 
 
 
 
 
 
 Link prediction; 
 
 similarity graph alignment 
 
 
 
 
 
 
 
 Links; 
 
 long-range similarities 
 
 
 
 
 Node classification 
 
 
 
 
 DGSI  [ 76 ] 
 
 Jun 2023 
 
 
 AAAI’23 
 
 
 
 
 
 
 Context discrimination; 
 
 node-graph discrimination 
 
 
 
 
 
 
 
 Context; 
 
 global structure 
 
 
 
 
 Node classification 
 
 
 
 
 ASP  [ 92 ] 
 
 Jun 2023 
 
 
 AAAI’23 
 
 
 
 Similarity graph alignment 
 
 
 
 Long-range similarities 
 
 
 
 Node classification 
 
 
 
 
 MoAMa  [ 97 ] 
 
 Sept 2023 
 
 
 LoG’24 
 
 
 
 Motif-based masked feature prediction 
 
 
 
 Node features; motifs 
 
 
 
 Graph classification 
 
 
 
 
 Graph-JEPA  [ 127 ] 
 
 Sept 2023 
 
 
 TMLR’25 
 
 
 
 Hyperbolic angle prediction 
 
 
 
 Context; manifolds 
 
 
 
 
 
 
 Graph classification; 
 
 graph regression 
 
 
 
 
 
 GraphFP  [ 96 ] 
 
 Oct 2023 
 
 
 NeurIPS’23 
 
 
 
 
 
 
 Motif prediction; 
 
 motif-based discrimination 
 
 
 
 
 Motifs 
 
 
 
 
 
 
 Graph classification; 
 
 graph regression 
 
 
 
 
 
 RES  [ 290 ] 
 
 Oct 2023 
 
 
 NeurIPS’23 
 
 
 
 
 
 
 Node instance discrimination 
 
 / graph instance discrimination 
 
 
 
 
 
 
 
 Node features; 
 
 global structure 
 
 
 
 
 
 
 
 Node classification 
 
 / graph classification 
 
 
 
 
 
 GPT-ST  [ 282 ] 
 
 Nov 2023 
 
 
 arXiv 
 
 
 
 Masked feature prediction 
 
 
 
 Node features; clusters 
 
 
 
 Time series forecasting 
 
 
 

 
 
 
 TABLE I: 

Summary of self-supervised graph pre-training methods (continued).

 
 
 
 
 

 
 
 
 
 Model 
 
 Time 
 
 
 Venue 
 
 
 
 Pre-training tasks 
 
 
 
 
 
 
 Graph knowledge 
 
 focused on 
 
 
 
 
 Downstream tasks 
 
 
 
 
 StructComp  [ 111 ] 
 
 Dec 2023 
 
 
 ICLR’24 
 
 
 
 Partition-based discrimination 
 
 
 
 Clusters 
 
 
 
 Node classification 
 
 
 
 
 DGPM  [ 98 ] 
 
 Dec 2023 
 
 
 AAAI’24 
 
 
 
 
 
 
 Masked feature prediction; 
 
 motif prediction 
 
 
 
 
 Node features; motifs 
 
 
 
 Graph classification 
 
 
 
 
 HTML  [ 124 ] 
 
 Dec 2023 
 
 
 AAAI’24 
 
 
 
 
 
 
 Contextual property prediction; 
 
 graph instance discrimination; 
 
 graph similarity prediction 
 
 
 
 
 
 
 
 Context; 
 
 global structure 
 
 
 
 
 Graph classification 
 
 
 
 
 MotifRGC  [ 99 ] 
 
 Jan 2024 
 
 
 AAAI’24 
 
 
 
 
 
 
 Motif-based discrimination; 
 
 cross-manifold discrimination 
 
 
 
 
 Motifs; manifolds 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 HypeBoy  [ 289 ] 
 
 Jan 2024 
 
 
 ICLR’24 
 
 
 
 Context discrimination 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 (hyper-)link prediciton 
 
 
 
 
 
 CTAug  [ 101 ] 
 
 Jan 2024 
 
 
 WWW’24 
 
 
 
 
 
 
 Node instance discrimination; 
 
 motif-based discrimination 
 
 
 
 
 Node features; motifs 
 
 
 
 Node classification 
 
 
 
 
 Bandana  [ 70 ] 
 
 Feb 2024 
 
 
 WWW’24 
 
 
 
 Link denoising 
 
 
 
 Links 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 DCGL  [ 106 ] 
 
 Feb 2024 
 
 
 AAAI’24 
 
 
 
 
 
 
 Feature prediction; 
 
 similarity graph alignment; 
 
 cluster-based discrimination 
 
 
 
 
 
 
 
 Node features; 
 
 long-range similarities; 
 
 clusters 
 
 
 
 
 Node clustering 
 
 
 
 
 HASH-CODE  [ 56 ] 
 
 Feb 2024 
 
 
 WWW’24 
 
 
 
 
 
 
 Node instance discrimination; 
 
 context discrimination; 
 
 contextual subgraph discrimination 
 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 WAS  [ 271 ] 
 
 Mar 2024 
 
 
 ICLR’24 
 
 
 
 Miscellaneous 
 
 
 
 – 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 
 
 ASD-VAE  [ 73 ] 
 
 Mar 2024 
 
 
 WSDM’24 
 
 
 
 
 
 
 Feature prediction; 
 
 edge feature prediction 
 
 
 
 
 Node features; links 
 
 
 
 Node classification; etc. 
 
 
 
 
 MGSE  [ 104 ] 
 
 May 2024 
 
 
 ICML’24 
 
 
 
 Node clustering 
 
 
 
 Clusters 
 
 
 
 Graph classification 
 
 
 
 
 GraphTCM  [ 269 ] 
 
 May 2024 
 
 
 ICML’24 
 
 
 
 Miscellaneous 
 
 
 
 – 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 D-VGAE  [ 74 ] 
 
 May 2024 
 
 
 WWW’24 
 
 
 
 Link prediction 
 
 
 
 Links; clusters 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 link prediction 
 
 
 
 
 
 VilLain  [ 286 ] 
 
 May 2024 
 
 
 WWW’24 
 
 
 
 Node clustering 
 
 
 
 Links; clusters 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 (hyper-)link prediciton; etc. 
 
 
 
 
 
 STGP  [ 283 ] 
 
 May 2024 
 
 
 CIKM’24 
 
 
 
 Masked feature prediction 
 
 
 
 Node features 
 
 
 
 Time series forcasting 
 
 
 
 
 DiscoGNN  [ 41 ] 
 
 Jul 2024 
 
 
 ICDE’24 
 
 
 
 
 
 
 Masked feature prediction; 
 
 edge feature prediction; 
 
 graph instance discrimination 
 
 
 
 
 
 
 
 Node features;links; 
 
 global structure 
 
 
 
 
 
 
 
 Graph classification; 
 
 similarity search 
 
 
 
 
 
 LogDet  [ 60 ] 
 
 Aug 2024 
 
 
 KDD’24 
 
 
 
 Dimension discrimination 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 RMR  [ 273 ] 
 
 Aug 2024 
 
 
 KDD’24 
 
 
 
 Node instance discrimination 
 
 
 
 Node features; links 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification 
 
 
 
 
 
 SGRL  [ 49 ] 
 
 Sept 2024 
 
 
 NeurIPS’24 
 
 
 
 Node instance discrimination 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 
 
 HDM-GAE  [ 128 ] 
 
 Jan 2025 
 
 
 COLING’25 
 
 
 
 Hyperbolic masked prediction 
 
 
 
 
 
 
 Node features; links; 
 
 manifolds 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 CenPre  [ 65 ] 
 
 Jan 2025 
 
 
 ICLR’25 
 
 
 
 
 
 
 Node instance discrimination; 
 
 property prediction 
 
 
 
 
 
 
 
 Node features; 
 
 node properties 
 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 link prediction 
 
 
 
 
 
 BSG  [ 87 ] 
 
 Jan 2025 
 
 
 WWW’25 
 
 
 
 
 
 
 Node instance discrimination; 
 
 context discrimination 
 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 RiemannGFM  [ 129 ] 
 
 Jan 2025 
 
 
 WWW’25 
 
 
 
 Cross-manifold discrimination 
 
 
 
 Manifolds 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 

 
 
 
 TABLE II: 

Summary of graph downstream tuning methods. “SFT” refers to “supervised fine-tuning”.

 
 
 
 
 

 
 
 Model 
 Time 
 
 
 Venue 
 
 
 
 
 
 
 Tuning 
 
 strategy 
 
 
 
 
 Training/ tuning tasks 
 
 
 
 
 
 
 Graph knowledge 
 
 focused on 
 
 
 
 
 Downstream tasks 
 
 
 L2P-GNN  [ 163 ] 
 May 2021 
 
 
 AAAI’21 
 
 
 
 Fine-tuning 
 
 
 
 
 
 
 Context discrimination; 
 
      graph instance discrimination SFT 
 
 
 
 
 
 
 
 Node features; context; 
 
 global structure 
 
 
 
 
 Graph classification 
 
 
 AUX-TS  [ 169 ] 
 Jul 2021 
 
 
 AAAI’21 
 
 
 
 Fine-tuning 
 
 
 
 
 
 
 Masked feature prediction; 
 
      masked link prediction SFT 
 
 
 
 
 Node features; links 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 GTOT-Tuning  [ 170 ] 
 Mar 2022 
 
 
 IJCAI’22 
 
 
 
 Fine-tuning 
 
 
 
 Miscellaneous SFT 
 
 
 
 – 
 
 
 
 Graph classification 
 
 
 GPPT  [ 68 ] 
 Aug 2022 
 
 
 KDD’22 
 
 
 
 Prompting 
 
 
 
 
 
 
 Masked link prediction SFT 
 
 
 
 
 Links; context; clusters 
 
 
 
 Node classification 
 
 
 GPF  [ 177 ] 
 Sept 2022 
 
 
 NeurIPS’23 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 link prediction 
 
 
 
 GraphPrompt  [ 142 ] 
 Feb 2023 
 
 
 WWW’23 
 
 
 
 Prompting 
 
 
 
 
 
 
 Context discrimination SFT 
 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 SGL-PT  [ 182 ] 
 Feb 2023 
 
 
 arXiv 
 
 
 
 Prompting 
 
 
 
 
 
 
 Masked feature prediction; 
 
      graph instance discrimination 
 
 Masked feature prediction; SFT 
 
 
 
 
 
 
 
 Node features; 
 
 global structure 
 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 W2PGNN  [ 165 ] 
 Mar 2023 
 
 
 KDD’23 
 
 
 
 Fine-tuning 
 
 
 
 Miscellaneous SFT 
 
 
 
 Motifs 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 AdapterGNN  [ 172 ] 
 Apr 2023 
 
 
 AAAI’24 
 
 
 
 
 
 
 Fine-tuning 
 
 (PEFT) 
 
 
 
 
 Miscellaneous SFT 
 
 
 
 – 
 
 
 
 Graph classification 
 
 
 G-Adapter  [ 173 ] 
 May 2023 
 
 
 AAAI’24 
 
 
 
 
 
 
 Fine-tuning 
 
 (PEFT) 
 
 
 
 
 Miscellaneous SFT 
 
 
 
 
 
 
 Node features; 
 
 long-range similarities 
 
 
 
 
 Graph classification 
 
 
 PRODIGY  [ 152 ] 
 May 2023 
 
 
 NeurIPS’23 
 
 
 
 Prompting 
 
 
 
 
 
 
 Miscellaneous 
 
 Context discrimination; SFT 
 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 link prediction 
 
 
 
 VNT  [ 181 ] 
 Jun 2023 
 
 
 KDD’23 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 node clustering 
 
 
 
 All in One  [ 153 ] 
 Jul 2023 
 
 
 KDD’23 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Node features; links 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 link prediction; 
 
 edge regression; 
 
 graph regression 
 
 
 
 S2PGNN  [ 164 ] 
 Aug 2023 
 
 
 ICDE’24 
 
 
 
 Fine-tuning 
 
 
 
 Miscellaneous SFT 
 
 
 
 Global structure 
 
 
 
 
 
 
 Graph classification; 
 
 graph regression 
 
 
 
 DeepGPT  [ 179 ] 
 Sept 2023 
 
 
 arXiv 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Node features 
 
 
 
 
 
 
 Graph classification; 
 
 graph regression 
 
 
 
 GraphControl  [ 167 ] 
 Oct 2023 
 
 
 WWW’24 
 
 
 
 
 
 
 Fine-tuning; 
 
 prompting 
 
 
 
 
 Miscellaneous SFT 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 OFA  [ 154 ] 
 Oct 2023 
 
 
 ICLR’24 
 
 
 
 Prompting 
 
 
 
 SFT 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 link prediction 
 
 
 
 Self-Pro  [ 79 ] 
 Oct 2023 
 
 
 
 
 
 ECML- 
 
 PKDD’24 
 
 
 
 
 Prompting 
 
 
 
 
 
 
 Context discrimination SFT 
 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 ULTRA-DP  [ 157 ] 
 Oct 2023 
 
 
 arXiv 
 
 
 
 Prompting 
 
 
 
 
 
 
 Link prediction; 
 
      context discrimination SFT 
 
 
 
 
 
 
 
 Node features; 
 
 links; context 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 HetGPT  [ 183 ] 
 Oct 2023 
 
 
 WWW’24 
 
 
 
 Prompting 
 
 
 
 
 
 
 Node instance discrimination SFT 
 
 
 
 
 
 
 
 Node features; 
 
 links; context 
 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification 
 
 
 
 PSP  [ 180 ] 
 Oct 2023 
 
 
 
 
 
 ECML- 
 
 PKDD’24 
 
 
 
 
 Prompting 
 
 
 
 
 
 
 Node-text discrimination SFT 
 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 GraphPrompt+  [ 187 ] 
 Nov 2023 
 
 
 TKDE’24 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 
 
 
 Context; 
 
 global structure 
 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 GraphPro  [ 193 ] 
 Nov 2023 
 
 
 WWW’24 
 
 
 
 Prompting 
 
 
 
 
 
 
 Node instance discrimination 
 
 Node instance discrimination 
 
 
 
 
 Node features; links 
 
 
 
 Recommendation 
 
 
 HGPROMPT  [ 148 ] 
 Dec 2023 
 
 
 AAAI’24 
 
 
 
 Prompting 
 
 
 
 
 
 
 Link prediction SFT 
 
 
 
 
 
 
 
 Node features; 
 
 links; context 
 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 graph classification 
 
 
 
 MultiGPrompt  [ 191 ] 
 Dec 2023 
 
 
 WWW’24 
 
 
 
 Prompting 
 
 
 
 
 
 
 Link prediction; 
 
      graph instance discrimination; 
 
      node-graph discrimination SFT 
 
 
 
 
 Links; global structure 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 G-Tuning  [ 166 ] 
 Dec 2023 
 
 
 AAAI’24 
 
 
 
 Fine-tuning 
 
 
 
 Miscellaneous SFT 
 
 
 
 Motifs 
 
 
 
 Graph classification 
 
 
 SUPT  [ 188 ] 
 Feb 2024 
 
 
 arXiv 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Node features; context 
 
 
 
 Graph classification 
 
 
 GraphPAR  [ 174 ] 
 Feb 2024 
 
 
 WWW’24 
 
 
 
 
 
 
 Fine-tuning 
 
 (PEFT) 
 
 
 
 
 
 
 
 Miscellaneous 
 
 Node instance discrimination; SFT 
 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 IGAP  [ 143 ] 
 Feb 2024 
 
 
 WWW’24 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 
 
 
 Node features; 
 
 spectrum 
 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 Bridge-Tune  [ 171 ] 
 Mar 2024 
 
 
 AAAI’24 
 
 
 
 Fine-tuning 
 
 
 
 Miscellaneous SFT 
 
 
 
 – 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 STGP  [ 283 ] 
 May 2024 
 
 
 CIKM’24 
 
 
 
 Prompting 
 
 
 
 Masked feature prediction 
 
 
 
 Node features 
 
 
 
 Time series forcasting 
 
 
 DyGPrompt  [ 285 ] 
 May 2024 
 
 
 ICLR’25 
 
 
 
 Prompting 
 
 
 
 
 
 
 (Temporal) link prediction SFT 
 
 
 
 
 Node features; links 
 
 
 
 
 
 
 (Temporal) 
 
 node classification; 
 
 link prediction 
 
 
 
 P2TAG  [ 192 ] 
 Jul 2024 
 
 
 KDD’24 
 
 
 
 Prompting 
 
 
 
 Masked language modeling SFT 
 
 
 
 
 
 
 Node features; context; 
 
 long-range similarities 
 
 
 
 
 Node classification 
 
 
 TGPT  [ 194 ] 
 Aug 2024 
 
 
 KDD’24 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Motifs; global structure 
 
 
 
 
 
 
 Graph classification; 
 
 graph regression 
 
 
 
 GraphCLIP  [ 224 ] 
 Oct 2024 
 
 
 WWW’25 
 
 
 
 Prompting 
 
 
 
 
 
 
 Node-text discrimination SFT 
 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 HG-Adapter  [ 175 ] 
 Nov 2024 
 
 
 ICLR’25 
 
 
 
 
 
 
 Fine-tuning 
 
 (PEFT) 
 
 
 
 
 
 
 
 Miscellaneous 
 
 Feature prediction; SFT 
 
 
 
 
 Node features; context 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 node clustering 
 
 
 
 IA-GPL  [ 178 ] 
 Nov 2024 
 
 
 TMLR’25 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Node features 
 
 
 
 Graph classification 
 
 
 DAGPrompT  [ 176 ] 
 Jan 2025 
 
 
 WWW’25 
 
 
 
 
 
 
 Fine-tuning 
 
 (PEFT); 
 
 prompting 
 
 
 
 
 
 
 
 Link prediction SFT 
 
 
 
 
 Node features; context 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 EdgePrompt  [ 185 ] 
 Jan 2025 
 
 
 ICLR’25 
 
 
 
 Prompting 
 
 
 
 Miscellaneous SFT 
 
 
 
 Links 
 
 
 
 
 
 
 Node classification; 
 
 graph classification 
 
 
 
 FPrompt  [ 186 ] 
 Jan 2025 
 
 
 WWW’25 
 
 
 
 
 
 
 Fine-tuning 
 
 (PEFT); 
 
 prompting 
 
 
 
 
 
 
 
 Node instance discrimination 
 
      / node-graph discrimination SFT 
 
 
 
 
 Node features; links 
 
 
 
 Node classification 
 
 
 

 
 
 
 TABLE III: 

Summary of graph language models.
 “Adapter” refers to all kinds of non-GNN/GT network modules apart from the LLM backbone, e.g., LoRA  [ 196 ] and linear projectors.
 “AR” and “MLM” refer to autoregressive and masked language modeling, respectively. “SFT” refers to “supervised fine-tuning”.

 
 
 
 
 

 
 
 
 
 Model 
 
 Time 
 
 
 Venue 
 
 Training/ tuning tasks 
 
 
 
 
 
 Graph knowledge 
 
 focused on 
 
 
 
 
 Downstream tasks 
 
 
 
 
 
 
 LM/LLM 
 
 
 
 Adapter 
 
 
 
 GNN/GT 
 
 
 
 
 
 
 KEPLER  [ 278 ] 
 
 Nov 2019 
 
 
 TACL’21 
 
 
 
 
 
 
 MLM; 
 
 node instance discimination 
 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Node features; 
 
 links 
 
 
 
 
 Link prediction 
 
 
 
 
 GraphFormers  [ 8 ] 
 
 May 2021 
 
 
 NeurIPS’21 
 
 
 
 Link prediction 
 
 
 
 – 
 
 
 
 Link prediction 
 
 
 
 Links 
 
 
 
 Link prediction 
 
 
 
 
 GIANT  [ 9 ] 
 
 Nov 2021 
 
 
 ICLR’22 
 
 
 
 Neighborhood prediction 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Context; 
 
 clusters 
 
 
 
 
 Node classification 
 
 
 
 
 GLEM  [ 244 ] 
 
 Oct 2022 
 
 
 ICLR’23 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 Node classification 
 
 
 
 
 G2P2  [ 201 ] 
 
 May 2023 
 
 
 SIGIR’23 
 
 
 
 
 
 
 Node-text discrimination 
 
 context discrimination SFT 
 
 
 
 
 – 
 
 
 
 
 
 
 Node-text discrimination 
 
 context discrimination 
 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 Node classification 
 
 
 
 
 NLGraph  [ 208 ] 
 
 May 2023 
 
 
 NeurIPS’23 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 – 
 
 
 
 – 
 
 
 
 Graph question answering 
 
 
 
 
 Patton  [ 147 ] 
 
 May 2023 
 
 
 ACL’23 
 
 
 
 
 
 
 MLM; link prediction 
 
 
 
 
 – 
 
 
 
 
 
 
 MLM; link prediction 
 
 
 
 
 
 
 
 Node features; 
 
 links; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction; etc. 
 
 
 
 
 
 ConGraT  [ 203 ] 
 
 May 2023 
 
 
 
 
 
 ACL 
 
 Workshop 
 
 (TextGraphs)’24 
 
 
 
 
 Node-text discrimination 
 
 
 
 – 
 
 
 
 Node-text discrimination 
 
 
 
 
 
 
 Node features; 
 
 long-range 
 
 similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 TAPE  [ 10 ] 
 
 May 2023 
 
 
 ICLR’24 
 
 
 
 Frozen (LLM) SFT (LM) 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 GALM  [ 200 ] 
 
 Jun 2023 
 
 
 KDD’23 
 
 
 
 Link prediction SFT 
 
 
 
 – 
 
 
 
 Link prediction SFT 
 
 
 
 Links 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 link prediction; 
 
 edge classification 
 
 
 
 
 
 KEA  [ 232 ] 
 
 May 2023 
 
 
 
 
 
 KDD 
 
 Explorations 
 
 Newsletter’24 
 
 
 
 
 Frozen (LLM) SFT (LM) 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 Node features 
 
 
 
 Node classification 
 
 
 
 
 LLM4Mol  [ 233 ] 
 
 Jul 2023 
 
 
 arXiv 
 
 
 
 Frozen (LLM) SFT (LM) 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Motifs; 
 
 global structure 
 
 
 
 
 Graph classification 
 
 
 
 
 SimTeG  [ 247 ] 
 
 Aug 2023 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 InstructGLM  [ 11 ] 
 
 Aug 2023 
 
 
 
 
 
 EACL 
 
 Findings’24 
 
 
 
 
 Link prediction; SFT 
 
 
 
 – 
 
 
 
 – 
 
 
 
 Links; context 
 
 
 
 Node classification 
 
 
 
 
 G-Prompt  [ 219 ] 
 
 Sept 2023 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 MLM 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 Node classification 
 
 
 
 
 WalkLM  [ 248 ] 
 
 Sept 2023 
 
 
 NeurIPS’23 
 
 
 
 MLM 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Long-range 
 
 similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 GNP  [ 281 ] 
 
 Sept 2023 
 
 
 AAAI’24 
 
 
 
 Frozen 
 
 
 
 
 
 
 Link prediction; 
 
 SFT 
 
 
 
 
 Link prediction; SFT 
 
 
 
 Links 
 
 
 
 Graph question answering 
 
 
 
 
 OFA  [ 154 ] 
 
 Oct 2023 
 
 
 ICLR’24 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 link prediction 
 
 
 
 
 
 GraphText  [ 264 ] 
 
 Oct 2023 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 Node classification 
 
 
 
 
 LLM-GNN  [ 237 ] 
 
 Oct 2023 
 
 
 ICLR’24 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 Label-free node classification 
 
 
 
 Clusters 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 GraphLLM  [ 245 ] 
 
 Oct 2023 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 SFT 
 
 
 
 Node features 
 
 
 
 Graph question answering 
 
 
 
 
 LLM4NG  [ 240 ] 
 
 Oct 2023 
 
 
 AAAI’25 
 
 
 
 Frozen 
 
 
 
 Link prediction 
 
 
 
 Miscellaneous 
 
 
 
 Links 
 
 
 
 Node classification 
 
 
 
 
 GraphGPT  [ 204 ] 
 
 Oct 2023 
 
 
 SIGIR’24 
 
 
 
 
 
 
 Frozen (LLM) 
 
 Node-text discrimination (LM) 
 
 
 
 
 
 
 
 Graph-instruction 
 
 matching; SFT 
 
 
 
 
 Node-text discrimination 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 GRENADE  [ 202 ] 
 
 Oct 2023 
 
 
 
 
 
 EMNLP 
 
 Findings’23 
 
 
 
 
 
 
 
 Node instance discrimination; 
 
 context discrimination 
 
 
 
 
 – 
 
 
 
 
 
 
 Node instance discrimination; 
 
 context discrimination 
 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 node clustering; 
 
 link prediction 
 
 
 
 
 
 LLM4DyG  [ 216 ] 
 
 Oct 2023 
 
 
 KDD’24 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 (Temporal) 
 
 graph question answering 
 
 
 
 
 
 DGTL  [ 221 ] 
 
 Oct 2023 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 Context 
 
 
 
 Node classification 
 
 
 
 
 THLM  [ 198 ] 
 
 Nov 2023 
 
 
 
 
 
 EMNLP 
 
 Findings’23 
 
 
 
 
 
 
 
 MLM; context discrimination 
 
 SFT 
 
 
 
 
 – 
 
 
 
 
 
 
 MLM; context discrimination 
 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 link prediction 
 
 
 
 
 
 Sun et al.   [ 243 ] 
 
 Nov 2023 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 Links 
 
 
 
 Node classification 
 
 
 
 
 LEADING  [ 246 ] 
 
 Dec 2023 
 
 
 arXiv 
 
 
 
 Miscellaneous 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 – 
 
 
 
 Node classification 
 
 
 
 
 ENGINE  [ 226 ] 
 
 Jan 2024 
 
 
 IJCAI’24 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 SNS  [ 213 ] 
 
 Feb 2024 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Long-range 
 
 similarities 
 
 
 
 
 Node classification 
 
 
 
 
 GraphToken  [ 217 ] 
 
 Feb 2024 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 Node features 
 
 
 
 Graph question answering 
 
 
 
 
 LinguGKD  [ 250 ] 
 
 Feb 2024 
 
 
 AAAI’25 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 
 
 
 Node instance discrimination; 
 
 SFT 
 
 
 
 
 
 
 
 Node features; 
 
 node properties; 
 
 context 
 
 
 
 
 Node classification 
 
 
 
 
 GraphTranslator  [ 231 ] 
 
 Feb 2024 
 
 
 WWW’24 
 
 
 
 Frozen 
 
 
 
 
 
 
 Node-text 
 
 discrimination; 
 
 masked language 
 
 modeling 
 
 
 
 
 Link prediction 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 graph question answering 
 
 
 
 
 
 LLaGA  [ 229 ] 
 
 Feb 2024 
 
 
 ICML’24 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 InstructGraph  [ 251 ] 
 
 Feb 2024 
 
 
 
 
 
 ACL 
 
 Findings’24 
 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 
 
 
 Node classification; 
 
 link prediction; 
 
 graph question answering 
 
 
 
 
 
 GraphPrompter  [ 218 ] 
 
 Feb 2024 
 
 
 
 
 
 WWW’24 
 
 (short papers) 
 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 Context 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 Pan et al.   [ 225 ] 
 
 Feb 2024 
 
 
 CIKM’24 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 
 
 
 Node instance discrimination; 
 
 label-free node classification 
 
 
 
 
 
 
 
 Node features; 
 
 context; 
 
 long-range 
 
 similarities 
 
 
 
 
 Node classification 
 
 
 
 
 GraphAdapter  [ 220 ] 
 
 Feb 2024 
 
 
 WWW’24 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 AR; SFT 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 Node classification 
 
 
 
 
 UniGraph  [ 132 ] 
 
 Feb 2024 
 
 
 arXiv 
 
 
 
 
 
 
 Frozen (LLM) 
 
 Node instance discrimination; 
 
 MLM (LM) 
 
 
 
 
 SFT 
 
 
 
 
 
 
 MLM; 
 
 node instance discrimination 
 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 graph classification; 
 
 edge classification 
 
 
 
 
 
 HiGPT  [ 228 ] 
 
 Feb 2024 
 
 
 KDD’24 
 
 
 
 
 
 
 Frozen (LLM) 
 
 Node-text discrimination (LM) 
 
 
 
 
 
 
 
 Graph-instruction 
 
 matching; SFT 
 
 
 
 
 Node-text discrimination 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification 
 
 
 
 
 
 GraphWiz  [ 252 ] 
 
 Feb 2024 
 
 
 KDD’24 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 Graph question answering 
 
 
 
 
 OpenGraph  [ 241 ] 
 
 Mar 2024 
 
 
 
 
 
 EMNLP 
 
 Findings’24 
 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 Masked link prediction 
 
 
 
 
 
 
 Links; 
 
 long-range 
 
 similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 GraphInstruct  [ 253 ] 
 
 Mar 2024 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 Graph question answering 
 
 
 
 
 LOGIN  [ 242 ] 
 
 May 2024 
 
 
 WSDM’25 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 SFT 
 
 
 
 
 
 
 Node features; 
 
 links 
 
 
 
 
 Node classification 
 
 
 
 
 TAGA  [ 223 ] 
 
 May 2024 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 Node-text discrimination 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 Node classification 
 
 
 

 
 
 
 TABLE III: 

Summary of graph language models (continued).
 “Adapter” refers to all kinds of non-GNN/GT network modules apart from the LLM backbone, e.g., LoRA  [ 196 ] and linear projectors.
 “AR” and “MLM” refer to autoregressive and masked language modeling, respectively. “SFT” refers to “supervised fine-tuning”.

 
 
 
 
 

 
 
 
 
 Model 
 
 Time 
 
 
 Venue 
 
 Training/ tuning tasks 
 
 
 
 
 
 Graph knowledge 
 
 focused on 
 
 
 
 
 Downstream tasks 
 
 
 
 
 
 
 LM/LLM 
 
 
 
 Adapter 
 
 
 
 GNN/GT 
 
 
 
 
 
 
 GAugLLM  [ 235 ] 
 
 Jun 2024 
 
 
 KDD’24 
 
 
 
 
 
 
 Frozen (LLM) 
 
 Neighborhood prediction 
 
 
 
 
 – 
 
 
 
 
 
 
 Node instance discrimination 
 
 / masked feature prediction 
 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 Node classification 
 
 
 
 
 UniGLM  [ 205 ] 
 
 Jun 2024 
 
 
 WSDM’25 
 
 
 
 Node instance discrimination 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 GSPT  [ 206 ] 
 
 Jun 2024 
 
 
 LoG’24 
 
 
 
 Masked feature prediction 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Node features; 
 
 context 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 HIGHT  [ 230 ] 
 
 Jun 2024 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 
 
 
 Masked feature 
 
 prediction; SFT 
 
 
 
 
 Frozen 
 
 
 
 
 
 
 Node features; 
 
 motifs 
 
 
 
 
 Graph classification; etc. 
 
 
 
 
 GOFA  [ 222 ] 
 
 Jul 2024 
 
 
 ICLR’25 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 
 
 
 AR; similarity prediction; 
 
 common neighbor prediction; 
 
 SFT 
 
 
 
 
 
 
 
 Node features; 
 
 context; 
 
 long-range 
 
 similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 P2TAG  [ 192 ] 
 
 Jul 2024 
 
 
 KDD’24 
 
 
 
 MLM 
 
 
 
 – 
 
 
 
 MLM SFT 
 
 
 
 
 
 
 Node features; 
 
 context; 
 
 long-range 
 
 similarities 
 
 
 
 
 Node classification 
 
 
 
 
 Path-LLM  [ 199 ] 
 
 Aug 2024 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 AR 
 
 
 
 – 
 
 
 
 
 
 
 Long-range 
 
 similarities 
 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 LLM4RGNN  [ 238 ] 
 
 Aug 2024 
 
 
 KDD’25 
 
 
 
 Frozen 
 
 
 
 Link prediction 
 
 
 
 Miscellaneous 
 
 
 
 Links 
 
 
 
 Node classification 
 
 
 
 
 CMRP  [ 184 ] 
 
 Aug 2024 
 
 
 KDD’24 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 SFT 
 
 
 
 Links 
 
 
 
 
 
 
 Node classification; 
 
 link prediction; 
 
 graph classification; 
 
 graph question answering 
 
 
 
 
 
 TEA-GLM  [ 227 ] 
 
 Aug 2024 
 
 
 NeurIPS’24 
 
 
 
 Frozen 
 
 
 
 SFT 
 
 
 
 
 
 
 Node instance discrimination; 
 
 dimension discrimination 
 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 Skianis et al.   [ 214 ] 
 
 Sept 2024 
 
 
 arXiv 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 – 
 
 
 
 – 
 
 
 
 Graph question answering 
 
 
 
 
 GUNDAM  [ 249 ] 
 
 Sept 2024 
 
 
 arXiv 
 
 
 
 Similarity prediction; SFT 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Long-range 
 
 similarities 
 
 
 
 
 Graph question answering 
 
 
 
 
 AuGLM  [ 254 ] 
 
 Oct 2024 
 
 
 arXiv 
 
 
 
 SFT 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Context; 
 
 long-range 
 
 similarities 
 
 
 
 
 Node classification 
 
 
 
 
 AskGNN  [ 215 ] 
 
 Oct 2024 
 
 
 
 
 
 EMNLP 
 
 Findings’24 
 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 
 
 
 Node instance discrimination; 
 
 SFT 
 
 
 
 
 
 
 
 Node features; 
 
 long-range 
 
 similarities 
 
 
 
 
 Node classification 
 
 
 
 
 GraphCLIP  [ 224 ] 
 
 Oct 2024 
 
 
 WWW’25 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 Node-text discrimination; SFT 
 
 
 
 Node features 
 
 
 
 
 
 
 Node classification; 
 
 link prediction 
 
 
 
 
 
 LLaSA  [ 207 ] 
 
 Nov 2024 
 
 
 arXiv 
 
 
 
 
 
 
 AR; 
 
 node-text discrimination 
 
 
 
 
 SFT 
 
 
 
 
 
 
 AR; 
 
 node-text discrimination 
 
 
 
 
 Node features 
 
 
 
 Graph question answering 
 
 
 
 
 TANS  [ 234 ] 
 
 Dec 2024 
 
 
 NAACL’25 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 
 
 
 Node properties; 
 
 context 
 
 
 
 
 Node classification 
 
 
 
 
 Locle  [ 239 ] 
 
 Dec 2024 
 
 
 SIGIR’25 
 
 
 
 Frozen 
 
 
 
 – 
 
 
 
 Label-free node classification 
 
 
 
 Links; clusters 
 
 
 
 Node classification 
 
 
 
 
 HierPromptLM  [ 255 ] 
 
 Jan 2025 
 
 
 arXiv 
 
 
 
 
 
 
 MLM; 
 
 link prediction 
 
 
 
 
 – 
 
 
 
 – 
 
 
 
 
 
 
 Node features; 
 
 links 
 
 
 
 
 
 
 
 (Heterogeneous) 
 
 node classification; 
 
 link prediction 
 
 
 
 
 
 SFGL  [ 236 ] 
 
 Jan 2025 
 
 
 ICLR’25 
 
 
 
 Frozen (LLM) SFT (LM) 
 
 
 
 – 
 
 
 
 Miscellaneous 
 
 
 
 
 
 
 Long-range 
 
 similarities 
 
 
 
 
 Node classification