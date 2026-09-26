Deep Graph Generators: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2012.15544v1 [cs.LG] 31 Dec 2020 
 
 

# Deep Graph Generators: A Survey

 
 
 FAEZEH FAEZ 1 
 
    
 YASSAMAN OMMI 2 
 
    
 MAHDIEH SOLEYMANI BAGHSHAH 1 
 
    
 AND HAMID R. RABIEE 1 
 

 Abstract 
 
 Deep generative models have achieved great success in areas such as image, speech, and natural language processing in the past few years. Thanks to the advances in graph-based deep learning, and in particular graph representation learning, deep graph generation methods have recently emerged with new applications ranging from discovering novel molecular structures to modeling social networks. This paper conducts a comprehensive survey on deep learning-based graph generation approaches and classifies them into five broad categories, namely, autoregressive, autoencoder-based, RL-based, adversarial, and flow-based graph generators, providing the readers a detailed description of the methods in each class. We also present publicly available source codes, commonly used datasets, and the most widely utilized evaluation metrics. Finally, we highlight the existing challenges and discuss future research directions.

 
 
 
 Index Terms:  Generative Models, Deep Learning, Graph Data, Deep Graph Generators, Molecular Graph Generation.

 † † history: Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000. † † doi: 10.1109/ACCESS.2017.DOI † † address: Department of Computer Engineering, Sharif University of Technology, Tehran, Iran † † address: Department of Mathematics and Computer Science, Amirkabir University of Technology, Tehran, Iran † † corresponding: Corresponding authors: Hamid R. Rabbiee and Mahdieh Soleymani Baghshah (e-mails: rabiee@sharif.edu , soleymani@sharif.edu). 
 

## I Introduction 

 
 
 
 
 
 Category 
 | 
 
 
 Key Characteristic 
 | 
 
 
 Publications 
 | 

 
 
 
 Autoregressive DGGs 
 | 
 
 
 Adopting a sequential generation strategy, either node-by-node or edge-by-edge 
 | 
 
 
 [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 ] 
 | 

 
 
 
 Autoencoder-Based DGGs 
 | 
 
 
 Making the generation process dependent on latent space variables 
 | 
 
 
 [ 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 14 , 35 , 15 , 16 , 17 , 36 , 37 , 38 , 39 , 18 , 19 ] 
 | 

 
 
 
 RL-Based DGGs 
 | 
 
 
 Utilizing reinforcement learning algorithms to induce desired properties in the generated graphs 
 | 
 
 
 [ 3 , 20 , 21 , 22 , 23 , 24 , 26 , 40 , 25 ] 
 | 

 
 
 
 Adversarial DGGs 
 | 
 
 
 Employing generative adversarial networks (GANs) [ 41 ] to generate graph structures 
 | 
 
 
 [ 42 , 43 , 44 , 20 , 22 , 40 , 38 , 45 , 39 , 46 , 47 ] 
 | 

 
 
 
 Flow-based DGGs 
 | 
 
 
 Learning a mapping from the complicated graph distribution into a distribution mostly modeled as a Gaussian for calculating the exact data likelihood 
 | 
 
 
 [ 12 , 13 , 37 , 48 ] 
 | 

 TABLE I: Categorization, Key Characteristic, and Representative Publications among Deep Graph Generators 
 
 
 Recently, with the rapid development of data collection and storage technologies, an increasing amount of data that needs to be processed is available. In many research areas, including biology, chemistry, pharmacy, social networks, and knowledge graphs, there exist some relationships between data entities that, if taken into account, more valuable features can be extracted, yielding more accurate predictions. Using graph data structure is a common way to represent such data, and therefore graph analysis research has attracted considerable attention.

 
 
 In the past few years, graph-related studies have made significant progress, which mainly focus on graph representation learning [ 49 , 50 , 51 , 52 , 53 ] but also include other problems like graph matching [ 54 , 55 ] , adversarial attack and defense on graph-based neural networks [ 56 , 57 ] , and graph attention networks [ 58 , 59 ] . Graph generation is also another research line aiming to generate new graph structures with some desired properties, which dates back to 1960 [ 60 ] and is followed by several other approaches [ 61 , 62 , 63 , 64 ] . However, the early methods generally use hand-engineered processes to create graphs with predefined statistical properties and, despite their simplicity, are not capable enough to capture complicated graph dependencies.

 
 
 Thanks to the recent successes of deep learning techniques and algorithms, deep generative models, which aim to generate novel samples from a similar distribution as the training data, have received a lot of attention in various data domains such as image [ 65 , 66 ] , text [ 67 , 68 ] , and speech [ 69 , 70 ] . Subsequently, studies related to deep learning-based graph generators have started a little later, which, unlike the traditional approaches, can directly learn from data and eliminate the need for using hand-designed procedures. Therefore, there are apparent horizons in this research area, with applications ranging from discovering new molecular structures to modeling social networks.

 
 
 So far, several surveys have reviewed deep graph-related approaches such as those mainly focusing on graph representation learning methods [ 71 , 72 , 73 , 74 , 75 ] , graph attention models [ 76 ] , attack and defense techniques on graph data [ 77 ] , and graph matching approaches [ 78 , 79 ] . Although most of these surveys have made a passing reference to the modern graph generation approaches, which we refer to as Deep Graph Generators (DGGs), this field requires individual attention due to its value and expanding development.

 
 
 In this paper, we conduct a survey on DGGs in order to exclusively review and categorize these methods and their applications. To this end, we first divide the existing approaches into five broad categories, namely, autoregressive DGGs, autoencoder-based DGGs, RL-based DGGs, adversarial DGGs, and flow-based DGGs, providing the readers with detailed descriptions of the methods in each category and comparing them from different aspects. This categorization is either based on the model architectures, adopted generation strategies, or optimization objectives and the categories may sometimes overlap so that a method can belong to more than one category. Table I summarizes the main characteristics of these categories, along with the most prominent approaches belonging to each of them.

 
 
 The rest of this article is organized as follows. Section II briefly summarizes notations used in this survey and formulates the problem of deep graph generation. Sections III to VII provide a detailed review of the existing DGGs in each of the five categories discussed above. Section VIII classifies the current applications and suggest some potential future ones. Section IX goes through implementation details by summarizing commonly used datasets, widely utilized evaluation metrics, and available source codes. Section X discusses future research directions. Finally, section XI concludes the survey.

 
 
 

## II Notations and Problem Formulation 

 
 TABLE II: COMMONLY USED NOTATIONS 
 
 
 
 
 Notations 
 | 
 
 
 Descriptions 
 | 

 
 
 
 G 
 | 
 
 
 A graph. 
 | 

 
 
 
 V 
 | 
 
 
 The node set of a graph. 
 | 

 
 
 
 E 
 | 
 
 
 The edge set of a graph. 
 | 

 
 
 
 p(G) 
 | 
 
 
 The graph data distribution. 
 | 

 
 
 
 n 
 | 
 
 
 The number of nodes, n=—V—. 
 | 

 
 
 
 N 
 | 
 
 
 The largest graph size in the dataset. 
 | 

 
 
 
 m 
 | 
 
 
 The number of edges, m=—E—. 
 | 

 
 
 
 π \pi 
 | 
 
 
 A node ordering for a graph G. 
 | 

 
 
 
 A π A^{\pi} 
 | 
 
 
 The adjacency matrix corresponding to a graph G under a node ordering π \pi . 
 | 

 
 
 
 S π S^{\pi} 
 | 
 
 
 The sequence corresponding to nodes of a graph under a node ordering π \pi . 
 | 

 
 
 
 S e ​ d ​ g ​ e , π S^{edge,\ \pi} 
 | 
 
 
 The sequence corresponding to edges of a graph. 
 | 

 
 
 
 e m b ( . ) emb(.) 
 | 
 
 
 An embedding function. 
 | 

 
 
 
 [ x , y ] [x,y] 
 | 
 
 
 The concatenation of x and y. 
 | 

 
 
 
 𝒩 u \mathcal{N}_{u} 
 | 
 
 
 The neighbor set of node u u . 
 | 

 
 
 This section reviews the notations used in the survey and provides a problem formulation for generating a set of plausible graphs.

 
 

### II-A Notations 

 
 We represent a graph as G = ( V , E ) G=(V,E) , where V V is the graph’s node set, and E E denotes its edge set, with | V | = n |V|=n and | E | = m |E|=m . N N also indicates the largest graph size in the dataset. There are n ! n! possible node orderings for the graph; thus, if we choose an ordering π \pi , the graph can be represented by the corresponding adjacency matrix A π ∈ ℝ n × n A^{\pi}\in\mathbb{R}^{n\times n} . Moreover, we can represent the graph with sequences of its ordered nodes or edges denoted by S n ​ o ​ d ​ e , π S^{node,\ \pi} and S e ​ d ​ g ​ e , π S^{edge,\ \pi} , respectively, where the former is denoted by the shorter form S π S^{\pi} in the following for simplicity. The notations are summarized in Table II .

 
 
 

### II-B Problem Formulation 

 
 Given a dataset of graphs 𝒟 G \mathcal{D}_{G} with the underlying data distribution p ⁡ ( G ) p(G) (i.e., for each graph G G in the dataset, G ∼ p ⁡ ( G ) G\sim p(G) ), a DGG aims to learn how to obtain new samples from the data distribution by employing deep neural networks. Specifically, this can be done by either estimating the p ⁡ ( G ) p(G) first and then sample from the estimated distribution or acquiring an implicit strategy, which only learns how to sample from the distribution without explicitly modeling it.

 
 
 
 

## III Autoregressive Deep Graph Generators 

 
 In this section, we review those approaches generating graph structures sequentially in a step-wise fashion, where the prediction at each time step is affected by the previous outputs. We further divide them into recurrent and non-recurrent approaches, where the former captures the generation history by employing recurrent units, while the latter makes decisions directly based on the latest partially generated graph. The main characteristics of these methods are summarized in Table III .

 
 

### III-A Recurrent DGGs 

 
 Recurrent DGGs are a bunch of autoregressive deep graph generators that use RNNs, namely long short-term memory (LSTM) [ 80 ] or gated recurrent units (GRU) [ 81 ] , to exert the influence of the generation history on the current decision. Here, we provide a detailed review of these methods in two subcategories.

 
 

#### III-A 1 Node-by-Node Generators

 
 Most of the autoregressive methods append one new node at a time into the already generated graph. For example, Li et al. [ 1 ] propose to generate molecular graphs sequentially, where the generation process initiates by adding a node to an empty graph. It then continues by iteratively deciding whether to append a new node to the graph, connect the lastly added node to the previous ones, or terminate the process. To this end, the authors propose two architectures, namely MolMP and MolRNN, to determine probabilities for each of these three actions. More precisely, MolMP decides based on the graph’s current state, modeling the generation as a Markov Decision Process. It first calculates an initial embedding for graph nodes followed by several convolutional layers and an aggregation operation to obtain a graph-level representation. It then passes both the node-level and graph-level embeddings through MLP and softmax layers to compute the probabilities required for action selection. MolRNN, on the other hand, exploits molecule level recurrent units to make the generation history affect the current decision, which improves the model’s performance. It adopts the same approach as MolMP to obtain embeddings and then updates the recurrent units’ hidden state as follows:

 

 
 | 
 h i = f t ​ r ​ a ​ n ​ s ​ ( h i − 1 , h v ∗ , h G i − 1 ) , h_{i}=f_{trans}(h_{i-1},h_{v^{*}},h_{G_{i-1}}), | 
 | 
 (1) | 
 

 where f t ​ r ​ a ​ n ​ s f_{trans} is implemented using GRUs, h v ∗ h_{v^{*}} is the latest appended node embedding, and h G i − 1 h_{G_{i-1}} denotes the representation for the graph generated before the i i -th generation step. Next, the action probabilities are calculated similarly as MolMP, except that MolRNN replaces h i h_{i} by the graph-level representation. Moreover, the authors make the conditional graph generation possible by first converting a given requirement to a conditional code and then modifying the graph convolution to include this code.

 
 
 You et al. [ 2 ] propose GraphRNN, another deep autoregressive model with a hierarchical architecture consisting of a graph-level RNN and an edge-level RNN which learns to sample G ∼ p ⁡ ( G ) G\sim p(G) without explicitly computing p ⁡ ( G ) p(G) . For this purpose, GraphRNN first defines a mapping f S f_{S} from graphs to sequences where for a graph G G with n n nodes under the node ordering π \pi , the mapping is defined as follows:

 

 
 | 
 S π = f S ​ ( G , π ) = ( S 1 π , … , S n π ) , S^{\pi}=f_{S}(G,\pi)=(S_{1}^{\pi},...,S_{n}^{\pi}), | 
 | 
 (2) | 
 

 where each element S i π ∈ { 0 , 1 } i − 1 , i ∈ { 1 , … , n } S_{i}^{\pi}\in\{0,1\}^{i-1},i\in\{1,...,n\} represents the edges between node π ⁡ ( v i ) \pi(v_{i}) 
and the previous nodes. Since for undirected graphs, there exists the mapping function f G ​ ( S π ) = G f_{G}(S^{\pi})=G , it is possible to sample G G at inference time by first sampling S π ∼ p ⁡ ( S π ) S^{\pi}\sim p(S^{\pi}) and then applying f G f_{G} , which obviates the need to compute p ⁡ ( G ) p(G) explicitly. To learn p ⁡ ( S π ) p(S^{\pi}) , due to the sequential nature of S π S^{\pi} ,
 p ⁡ ( S π ) p(S^{\pi}) can be further decomposed as in Eq. ( 3 ), which is modeled by an RNN with state transition and output functions defined in Eq. ( 4 ) and ( 5 ), respectively:

 

 
 | 
 p ⁡ ( S π ) = ∏ i = 1 n + 1 p ⁡ ( S i π | S 1 π , … , S i − 1 π ) = ∏ i = 1 n + 1 p ⁡ ( S i π | S i π ) , p(S^{\pi})=\prod_{i=1}^{n+1}p(S_{i}^{\pi}|S_{1}^{\pi},...,S_{i-1}^{\pi})=\prod_{i=1}^{n+1}p(S_{i}^{\pi}|S_{ i}^{\pi}), | 
 | 
 (3) | 
 

 

 
 | 
 h i n ​ o ​ d ​ e = f t ​ r ​ a ​ n ​ s n ​ o ​ d ​ e ​ ( h i − 1 n ​ o ​ d ​ e , In i n ​ o ​ d ​ e ) , In i n ​ o ​ d ​ e = S i − 1 π , h^{node}_{i}=f_{trans}^{node}(h^{node}_{i-1},\text{In}^{node}_{i}),\ \ \text{In}^{node}_{i}=S_{i-1}^{\pi}, | 
 | 
 (4) | 
 

 

 
 | 
 θ i = f o ​ u ​ t ​ ( h i n ​ o ​ d ​ e ) , \theta_{i}=f_{out}(h_{i}^{node}), | 
 | 
 (5) | 
 

 where f t ​ r ​ a ​ n ​ s n ​ o ​ d ​ e f_{trans}^{node} is implemented using a GRU and serves as the graph-level RNN that maintains the state of the graph generated so far. Furthermore, the authors propose two varients for the implementation of f o ​ u ​ t f_{out} . First, they propose GraphRNN-S, a simple variant that does not consider dependencies between edges and models p ⁡ ( S i π | S i π ) p(S_{i}^{\pi}|S_{ i}^{\pi}) as a multivariate Bernoulli distribution. Next, to fully capture the complex edge dependencies, they propose the full GraphRNN model as illustrated in Figure 1 , which approximates f o ​ u ​ t f_{out} by another RNN (i.e., the edge-level RNN) formulated as follows:

 

 
 | 
 h i , j e ​ d ​ g ​ e = f t ​ r ​ a ​ n ​ s e ​ d ​ g ​ e ​ ( h i , j − 1 e ​ d ​ g ​ e , In j e ​ d ​ g ​ e ) , In j e ​ d ​ g ​ e = S i , j − 1 π , h i , 0 e ​ d ​ g ​ e = h i n ​ o ​ d ​ e . \small h_{i,j}^{edge}=f_{trans}^{edge}(h^{edge}_{i,j-1},\text{In}^{edge}_{j}),\ \ \text{In}^{edge}_{j}=S_{i,j-1}^{\pi},\ \ h^{edge}_{i,0}=h^{node}_{i}. | 
 | 
 (6) | 
 

 Furthermore, GraphRNN introduces a BFS node ordering scheme to improve scalability with two benefits. First, it will suffice to train the model on all possible BFS orderings, rather than all possible node permutations. Second, it reduces the number of edges to be predicted in the edge-level RNN.

 
 
 Fig. 1: An illustration of the graph generation procedure at inference time proposed
in GraphRNN [ 2 ] (reprinted with permission). Green arrows denote the
graph-level RNN, and blue arrows represent the edge-level RNN. 
 
 
 Subsequently, several graph generation methods have been proposed inspired by GraphRNN. For example, Liu et al. [ 82 ] propose two further variants for the implementation of f o ​ u ​ t f_{out} function in Eq. ( 5 ), namely RNN-Transf and GraphRNN+Attn. Specifically, RNN-Transf replaces the edge-level RNN in the full GraphRNN model with a vanilla Transformer [ 83 ] decoder consisting of a self-attention sublayer and a graph-state attention sublayer with the memory from hidden states of the node-level RNN. GraphRNN+Attn, on the other hand, maintains the edge-level RNN. More precisely, it is an additive attention mechanism in the edge-level RNN that computes the attention weights in each step using the last hidden states of the node-level RNN as well as the current hidden state of the edge-level RNN.

 
 
 MolecularRNN [ 3 ] extends GraphRNN to generate realistic molecular graphs with desired chemical properties. As in molecular graphs, both nodes and edges have types, likelihood formulation in Equation ( 3 ) is rewritten as follows:

 

 
 | 
 p ⁡ ( S π , C π ) = ∏ i = 1 n + 1 p ⁡ ( C i π | S i π , C i π ) ​ p ​ ( S i π | C i π , S i π , C i π ) , p(S^{\pi},C^{\pi})=\prod_{i=1}^{n+1}p(C_{i}^{\pi}|S_{ i}^{\pi},C_{ i}^{\pi})p(S_{i}^{\pi}|C_{i}^{\pi},S_{ i}^{\pi},C_{ i}^{\pi}), | 
 | 
 (7) | 
 

 where S i , j π ∈ { 0 , 1 , 2 , 3 } S_{i,j}^{\pi}\in\{0,1,2,3\} is the categorical edge type that corresponds to no, single, double, or triple bonds, and C i π ∈ { 1 , 2 , … , K } C_{i}^{\pi}\in\{1,2,...,K\} determines node (atom) type. Then, MolecularRNN substitutes the graph-level RNN input in Eq. ( 4 ) with the embeddings of categorical inputs as in Eq. ( 8 ):

 

 
 | 
 In i n ​ o ​ d ​ e = [ e ​ m ​ b ​ ( S i − 1 π ) , e ​ m ​ b ​ ( C i − 1 π ) ] . \text{In}_{i}^{node}=[emb(S_{i-1}^{\pi}),emb(C_{i-1}^{\pi})]. | 
 | 
 (8) | 
 

 Furthermore, a two-layer MLP with softmax output activation is added on top of the hidden states of both graph-level and edge-level RNNs to predict node and edge types, respectively.
After likelihood pretraining on the molecular datasets, the model is fine-tuned with the policy gradient algorithm to shift the distribution
of the generated samples to some desired chemical properties, namely, lipophilicity, drug-likeness, and melting point. Thus, the MolecularRNN acts as a policy network to output the probability of the next action given the current state, where the set of states consists of all possible sub-graphs and the possible atom connections to the existing graph, for all the atom types, serve as the action set. Moreover, each valid molecule is considered as a final state s n s_{n} , where its corresponding final reward is denoted by r ⁡ ( s n ) r(s_{n}) . The intermediate rewards r ⁡ ( s i ) , 0 i n r(s_{i}),0 i n are also obtained by discounting r ⁡ ( s n ) r(s_{n}) as in the following loss function formula:

 

 
 | 
 ℒ ( θ ) = − ∑ i = 1 n r ( s n ) . γ i . log p ( s i | s i − 1 ; θ ) , \mathcal{L}(\theta)=-\sum_{i=1}^{n}r(s_{n}).\gamma^{i}.\log p(s_{i}|s_{i-1};\theta), | 
 | 
 (9) | 
 

 where γ \gamma is the discount factor and the transition probabilities p ⁡ ( s i | s i − 1 ; θ ) p(s_{i}|s_{i-1};\theta) are the elements of the product in Eq( 7 ). Furthermore, MolecularRNN introduces the structural penalty for atoms violating valency constraints during training. It also adopts a valency-based rejection sampling method during inference, which guarantees the generated samples’ validity.

 
 
 Sun et al. [ 84 ] learn a mapping from a source to a target graph by adopting an encoder-decoder based approach, where the encoder utilizes recurrent based models to encode the source graph and the decoder generates the target graph in a node-by-node fashion, which makes it necessary to consider an ordering over nodes. Therefore, the authors first introduce a procedure to transform a graph G G into a DAG (Directed Acyclic Graph) to provide the required node ordering. They then obtain embeddings for each of the DAG’s nodes by proposing two encoders: an Energy-Flow encoder and a Topology-Flow encoder, where the former utilizes only the information of adjacent nodes, while the latter exploits both the adjacent and non-adjacent nodes’ information. Afterward, the decoder sequentially generates the target graph conditioned on the source graph by adopting a relatively similar generation strategy as the GraphRNN.

 
 
 So far, we have studied GraphRNN [ 2 ] as one of the most widely used deep graph generators and then reviewed the subsequent graph generation approaches inspired by it; each generates different types of graphs from general to molecular ones. Moreover, there are also other methods that use GraphRNN as a basis for solving some application-specific problems. For example, REIN [ 85 ] proposes to autoregressively generate meshes from input point clouds inspired by GraphRNN so that in each generation step, it predicts edges from the newly introduced point to all the previous ones. The generated mesh can then be used for the task of 3D object reconstruction. DeepNC [ 86 ] is another GraphRNN-based approach that proposes a network completion algorithm to infer the missing parts of a network. Specifically, it first trains GraphRNN to learn a likelihood over the data distribution. The method then formulates an optimization problem to infer the missing parts of a partially observed input graph in such a way that maximizes the learned likelihood.

 
 
 

#### III-A 2 Edge-by-Edge Generators

 
 In addition to the methods discussed so far, there also exist other approaches adopting an edge-based generation strategy. Bacciu et al. [ 5 ] propose to generate a sequence of edges for each graph instead of generating graphs node-by-node. They first convert a graph G G under the node ordering π \pi to an ordered edge sequence S e ​ d ​ g ​ e , π = [ S 1 e ​ d ​ g ​ e , π , … , S m e ​ d ​ g ​ e , π ] S^{edge,\ \pi}=[S^{edge,\ \pi}_{1},...,S^{edge,\ \pi}_{m}] , where S i e ​ d ​ g ​ e , π = ( u i π , v i π ) S^{edge,\ \pi}_{i}=(u_{i}^{\pi},v_{i}^{\pi}) is the i i -th edge in the sequence that connects the source node u i π u_{i}^{\pi} to the destination node v i π v_{i}^{\pi} (here u i π u_{i}^{\pi} and v i π v_{i}^{\pi} are IDs assigend to graph nodes by π \pi ). Note that the sequence is ordered, that is S i e ​ d ​ g ​ e , π ≤ S i + 1 e ​ d ​ g ​ e , π S_{i}^{edge,\ \pi}\leq S_{i+1}^{edge,\ \pi} iff u i π u i + 1 π u_{i}^{\pi} u_{i+1}^{\pi} or ( u i π = u i + 1 π u_{i}^{\pi}=u_{i+1}^{\pi} and v i π v i + 1 π v_{i}^{\pi} v_{i+1}^{\pi} ). Then, the authors define U π = [ u 1 π , … , u m π ] U^{\pi}=[u_{1}^{\pi},...,u_{m}^{\pi}] and V π = [ v 1 π , … , v m π ] V^{\pi}=[v_{1}^{\pi},...,v_{m}^{\pi}] as sequences of the source and destination node IDs, respectively and decompose the edge sequence probability as followes:

 

 
 | 
 S e ​ d ​ g ​ e , π = p ⁡ ( U π ) ​ p ​ ( V π | U π ) , S^{edge,\ \pi}=p(U^{\pi})p(V^{\pi}|U^{\pi}), | 
 | 
 (10) | 
 

 where p ⁡ ( U π ) p(U^{\pi}) and p ⁡ ( V π | U π ) p(V^{\pi}|U^{\pi}) are approximated with two RNNs. Specifically, RNN1 is used to estimate p ⁡ ( U π ) p(U^{\pi}) with the following transition and output functions:

 

 
 | 
 h i RNN1 = f t ​ r ​ a ​ n ​ s ​ ( h i − 1 RNN1 , In i RNN1 ) , In i RNN1 = e ​ m ​ b ​ ( u i − 1 π ) p ⁡ ( u i π | u i − 1 π , h i − 1 RNN1 ) = f o ​ u ​ t ​ ( h i RNN1 ) = Softmax ​ ( L ​ i ​ n ​ ( h i RNN1 ) ) , \begin{split} h_{i}^{\texttt{RNN1}}=f_{trans}(h_{i-1}^{\texttt{RNN1}},\text{In}_{i}^{\texttt{RNN1}}),\ \ \ \text{In}_{i}^{\texttt{RNN1}}=emb(u_{i-1}^{\pi})\\
 p(u_{i}^{\pi}|u_{i-1}^{\pi},h_{i-1}^{\texttt{RNN1}})=f_{out}(h_{i}^{\texttt{RNN1}})=\text{Softmax}(Lin(h_{i}^{\texttt{RNN1}})),\end{split} | 
 | 
 (11) | 
 

 where f t ​ r ​ a ​ n ​ s f_{trans} is implemented as a GRU and L ​ i ​ n Lin is a linear projection to map the recurrent output to the node ID space. Once all of the U π U^{\pi} ’s elements are generated, the last recurrent state of RNN1 is used to initialize the state of RNN2 , and the process moves to RNN2 that is given U π U^{\pi} as input. Thus RNN2 computes the probability distribution of p ⁡ ( V π | U π ) p(V^{\pi}|U^{\pi}) with the same architecture as RNN1 by approximating p ⁡ ( v i | u i , h i − 1 RNN2 ) p(v_{i}|u_{i},h_{i-1}^{\texttt{RNN2}}) each step.

 
 
 Similarly, GraphGen [ 6 ] proposes another edge-based generation strategy that adds a single edge to the already generated graph at each stage. To this end, the method first converts a graph G G to a sequence S e ​ d ​ g ​ e = [ S 1 e ​ d ​ g ​ e , … , S m e ​ d ​ g ​ e ] S^{edge}=[S_{1}^{edge},...,S_{m}^{edge}] using the minimum DFS code [ 87 ] , where each S i e ​ d ​ g ​ e S_{i}^{edge} corresponds to an edge e = ( u , v ) e=(u,v) and is described using a 5-tuple ( t u , t v , L u , L e , L v ) (t_{u},t_{v},L_{u},L_{e},L_{v}) , where t u t_{u} is the timestamp assigned to node u u during the DFS traversal, and L u L_{u} and L e L_{e} denote the node and edge labels, respectively. As the minimum DFS codes are canonical labels, and thus there is a one-to-one mapping between a graph and its corresponding sequence, there is no longer need to deal with multiple representations for the same graph under different node permutations during training, which improves the method scalability. Then, GraphGen takes a similar approach to GraphRNN [ 2 ] to decompose p ⁡ ( S e ​ d ​ g ​ e ) p(S^{edge}) as follows:

 

 
 | 
 p ⁡ ( S e ​ d ​ g ​ e ) = ∏ i = 1 m + 1 p ⁡ ( S i e ​ d ​ g ​ e | S 1 e ​ d ​ g ​ e , … , S i − 1 e ​ d ​ g ​ e ) = ∏ i = 1 m + 1 p ⁡ ( S i e ​ d ​ g ​ e | S i e ​ d ​ g ​ e ) , \small p(S^{edge})=\prod_{i=1}^{m+1}p(S_{i}^{edge}|S_{1}^{edge},...,S_{i-1}^{edge})=\prod_{i=1}^{m+1}p(S_{i}^{edge}|S_{ i}^{edge}), | 
 | 
 (12) | 
 

 where m m is the number of edges, and making the simplifying assumption that t u t_{u} , t v t_{v} , L u L_{u} , L e L_{e} , and L v L_{v} are independent, reduces p ⁡ ( S i e ​ d ​ g ​ e | S i e ​ d ​ g ​ e ) p(S_{i}^{edge}|S_{ i}^{edge}) in Eq. ( 12 ) to:

 

 
 | 
 p ⁡ ( S i e ​ d ​ g ​ e | S i e ​ d ​ g ​ e ) = p ⁡ ( ( t u , t v , L u , L e , L v ) | S i e ​ d ​ g ​ e ) = p ⁡ ( t u | S i e ​ d ​ g ​ e ) × p ⁡ ( t v | S i e ​ d ​ g ​ e ) × p ⁡ ( L u | S i e ​ d ​ g ​ e ) × p ⁡ ( L e | S i e ​ d ​ g ​ e ) × p ⁡ ( L v | S i e ​ d ​ g ​ e ) . \begin{split}p(S_{i}^{edge}|S_{ i}^{edge}) =p((t_{u},t_{v},L_{u},L_{e},L_{v})|S_{ i}^{edge})\\
 =p(t_{u}|S_{ i}^{edge})\times p(t_{v}|S_{ i}^{edge})\times p(L_{u}|S_{ i}^{edge})\\
 \times p(L_{e}|S_{ i}^{edge})\times p(L_{v}|S_{ i}^{edge}).\end{split} | 
 | 
 (13) | 
 

 To capture conditional distributions in Eq. ( 13 ), the authors propose to use a custom LSTM with the transition function f t ​ r ​ a ​ n ​ s f_{trans} in Eq. ( 14 ), and five separate output functions for each component of the 5-tuple. For example, f t u f_{t_{u}} in Eq. ( 15 ) is utilized for predicting t u t_{u} , where ∼ M \sim_{M} represents sampling from a multinomial distribution.

 

 
 | 
 h i = f t ​ r ​ a ​ n ​ s ​ ( h i − 1 , In i ) , In i = e ​ m ​ b ​ ( S i − 1 e ​ d ​ g ​ e ) , h_{i}=f_{trans}(h_{i-1},\text{In}_{i}),\ \ \text{In}_{i}=emb(S_{i-1}^{edge}), | 
 | 
 (14) | 
 

 

 
 | 
 t u ∼ M θ t u = f t u ( h i ) , t_{u}\sim_{M}\ \theta_{t_{u}}=f_{t_{u}}(h_{i}), | 
 | 
 (15) | 
 

 

 
 | 
 S i e ​ d ​ g ​ e = c ​ o ​ n ​ c ​ a ​ t ​ ( t u , t v , L u , L e , L v ) . S_{i}^{edge}=concat(t_{u},t_{v},L_{u},L_{e},L_{v}). | 
 | 
 (16) | 
 

 Figure 2 outlines the proposed pipeline.

 
 
 Fig. 2: Flowchart of GraphGen [ 6 ] . 
 
 
 
 

### III-B Non-Recurrent DGGs 

 
 There are some other autoregressive methods that, unlike the recurrent models, do not consider the entire history, and instead, they only focus on the latest version of the partially generated graph at each time step. To better review these methods, we further divide them into two following subsections.

 
 

#### III-B 1 Attention-Based Methods

 
 Here, we review the methods in which the attention mechanism plays a key role. In this regard, GRAN [ 7 ] proposes to generate one block of nodes and associated edges at each generation step by optimizing the following likelihood:

 

 
 | 
 p ⁡ ( L π ) = ∏ t = 1 T p ⁡ ( L b t π | L b 1 π , … , L b t-1 π ) , p(L^{\pi})=\prod_{t=1}^{T}p(L^{\pi}_{\textbf{b}_{\textbf{t}}}|L^{\pi}_{\textbf{b}_{\textbf{1}}},...,L^{\pi}_{\textbf{b}_{\textbf{t-1}}}), | 
 | 
 (17) | 
 

 where L π L^{\pi} is the lower triangular part of the adjacency matrix A π A^{\pi} , B B denotes the block size, b t = { B ⁡ ( t − 1 ) + 1 , … , B ​ t } \textbf{b}_{\textbf{t}}=\{B(t-1)+1,...,Bt\} is the set of row indices for the t t -th block of L π L^{\pi} , and T = ⌈ N B ⌉ T=\lceil\frac{N}{B}\rceil is the number of graph generation steps. For the t t -th step, GRAN adds B B new nodes to the already-generated subgraph and connects them with each other as well as the previous B ⁡ ( t − 1 ) B(t-1) nodes to acquire an augmented graph as depicted in Figure 3 . The authors then apply the following graph neural network with attentive messages on the augmented graph to get updated node representations:

 

 
 | 
 m i ​ j r = f ⁡ ( h i r − h j r ) ​ , h i r ~ = [ h i r , x i ] ​ , a i ​ j r = σ ⁡ ( g ⁡ ( h ~ i r − h ~ j r ) ) h i r + 1 = GRU ​ ( h i r , ∑ j ∈ 𝒩 ⁡ ( i ) a i ​ j r ​ m i ​ j r ) , \begin{split}m^{r}_{ij}=f(h^{r}_{i}-h^{r}_{j})\textbf{,}\ \ \ \tilde{h^{r}_{i}} =[h^{r}_{i},x_{i}]\textbf{,}\ \ \ a^{r}_{ij}=\sigma\big(g(\tilde{h}^{r}_{i}-\tilde{h}^{r}_{j})\big)\\
h^{r+1}_{i} =\text{GRU}(h^{r}_{i},\sum_{j\in\mathcal{N}(i)}a^{r}_{ij}m^{r}_{ij}),\end{split} | 
 | 
 (18) | 
 

 where h i r h^{r}_{i} is the representation for node i i after round r r , m i ​ j r m^{r}_{ij} is the message vector from node i i to j j , x i x_{i} indicates whether node i i is in the previously generated nodes or the newly added ones, and a i ​ j r a^{r}_{ij} is an attention weight associated with e ​ d ​ g ​ e ​ ( i , j ) edge(i,j) . Both the message function f f and the attention function g g are implemented as 2-layer MLPs with ReLU nonlinearities. After R R rounds of message passing, the final node representation vectors h i R h^{R}_{i} for each node i i is obtained, and then GRAN models the conditional probability in Eq. ( 17 ) with a mixture of Bernoulli distributions to capture edge dependencies via K K latent mixture components:

 

 
 | 
 p ⁡ ( L b t π | L b 1 π , … , L b t-1 π ) = ∑ k = 1 K α k ​ ∏ i ∈ b t ∏ 1 ≤ j ≤ i θ k , i , j α 1 , … , α K = Softmax ​ ( ∑ i ∈ b t , 1 ≤ j ≤ i MLP α ​ ( h i R − h j R ) ) θ 1 , i , j , … , θ K , i , j = σ ⁡ ( MLP θ ​ ( h i R − h j R ) ) . \begin{split} p(L^{\pi}_{\textbf{b}_{\textbf{t}}}|L^{\pi}_{\textbf{b}_{\textbf{1}}},...,L^{\pi}_{\textbf{b}_{\textbf{t-1}}})=\sum_{k=1}^{K}\alpha_{k}\prod_{i\in\textbf{b}_{\textbf{t}}}\prod_{1\leq j\leq i}\theta_{k,i,j}\\
 \alpha_{1},...,\alpha_{K}=\text{Softmax}\ \Big(\sum_{i\in\textbf{b}_{\textbf{t}},1\leq j\leq i}\text{MLP}_{\alpha}(h_{i}^{R}-h_{j}^{R})\Big)\\
 \theta_{1,i,j},...,\theta_{K,i,j}=\sigma\big(\text{MLP}_{\theta}(h^{R}_{i}-h^{R}_{j})\big).\end{split} | 
 | 
 (19) | 
 

 
 
 Fig. 3: An overview of GRAN [ 7 ] (reprinted with permission). Dashed lines are augmented edges. Nodes with the same color belong to the same block (block size = 2). 
 
 
 GRAM [ 8 ] proposes to combine graph convolutional networks with graph attention mechanisms to obtain richer features during the graph generation process, where the proposed graph attention mechanism extends the one in [ 83 ] by introducing bias terms as a function of the shortest path between nodes. In particular, GRAM tries to maximize the following likelihood, which is somewhat similar to Eq. ( 7 ):

 

 
 | 
 p ⁡ ( CLOSE OPEN A π , C π ) = ∏ i = 1 n + 1 p ( C i π | A i , i π , C i π ) ∏ j = 1 i − 1 p ( A j , i π | A j , i π , C i π , A i , i π , C i π ) . \small\begin{split}p( A^{\pi},C^{\pi})=\\
 \prod_{i=1}^{n+1}p(C_{i}^{\pi}|A_{ i, i}^{\pi},C_{ i}^{\pi})\prod_{j=1}^{i-1}p(A_{j,i}^{\pi}|A_{ j,i}^{\pi},C_{i}^{\pi},A_{ i, i}^{\pi},C_{ i}^{\pi}).\end{split} | 
 | 
 (20) | 
 

 To this end, the authors propose an architecture that consists of three networks, namely, feature extractor, node estimator, and edge estimator. Firstly, the feature extractor extracts the local and global information using graph convolution layers and graph attention layers, respectively, where an attention layer employs a self-attention mechanism with the query, key, and value that are set to the node feature vectors. A graph pooling layer then aggregates all node features into a graph feature vector, denoted as h G h^{G} , by summing them up. Next, the node estimator determines a label for the new node based on the feature vector of the graph generated so far. Thereafter, the edge estimator predicts labels for edges between the newly added node and those already exist in the graph one after the other using a source-target attention mechanism as follows:

 

 
 | 
 A j , i π = Softmax ​ ( g E ​ E ​ ( h j v , h G , h i v , h j e ) ) , A_{j,i}^{\pi}=\text{Softmax}(g_{EE}(h^{v}_{j},h^{G},h^{v}_{i},h^{e}_{ j})), | 
 | 
 (21) | 
 

 where g E ​ E g_{EE} is a three-layer feedforward network, h j v h^{v}_{j} and h i v h^{v}_{i} are label embeddings of node v j v_{j} and the new node v i v_{i} , respectively, and h j e h^{e}_{ j} is computed using a source-target attention with Concat ​ ( h j v , h i v ) \text{Concat}(h^{v}_{j},h^{v}_{i}) as its query and { Concat ( h t v , h i v , h t , i e ) | t = 1 , … , j − 1 } \{\text{Concat}(h^{v}_{t},h^{v}_{i},h^{e}_{t,i})|t=1,...,j-1\} as both the key and value.

 
 
 AGE [ 9 ] introduces another attention-based generative model, which is conditioned on some input graphs. In other words, the method takes an existing source graph as input and generates a transformed version of it, modeling its evolution. To this end, the authors propose an encoder-decoder based architecture, where its decoder autoregressively generates the target graph in a node-by-node fashion. More specifically, the encoder first applies the self-attention mechanism to the source graph in order to learn its nodes’ representations. Then, at each generation step, the decoder first adopts a similar self-attention mechanism as the encoder, followed by source-target attention, which discovers the correlations between the nodes in the source graph and the ones in the already generated target graph. This way the decoder computes a representation for the graph generated so far, which will be further used to predict the new node’s label and connections.

 
 
 

#### III-B 2 Other Methods

 
 Fig. 4: An overview of the edge generation procedure in [ 11 ] (reprinted with permission). 
 
 
 
 
 Method | 
 Recurrent | 
 
 
 
 Generation 
 
 Strategy 
 | 
 
 
 
 Attention 
 
 Mechanism 
 | 
 Features | 
 
 
 
 Conditional 
 
 Generation 
 | 

 
 MolMP [ 1 ] | 
 No | 
 Node-by-node | 
 No | 
 Node/Edge | 
 Yes | 

 
 MolRNN [ 1 ] | 
 Yes | 
 Node-by-node | 
 No | 
 Node/Edge | 
 Yes | 

 
 GraphRNN [ 2 ] | 
 Yes | 
 Node-by-node | 
 No | 
 - | 
 No | 

 
 MolecularRNN [ 3 ] | 
 Yes | 
 Node-by-node | 
 No | 
 Node/Edge | 
 No | 

 
 Bacciu et al. [ 4 , 5 ] | 
 Yes | 
 Edge-by-edge | 
 No | 
 - | 
 No | 

 
 GraphGen [ 6 ] | 
 Yes | 
 Edge-by-edge | 
 No | 
 Node/Edge | 
 No | 

 
 GRAN [ 7 ] | 
 No | 
 Block of nodes | 
 Yes | 
 - | 
 No | 

 
 GRAM [ 8 ] | 
 No | 
 Node-by-node | 
 Yes | 
 Node/Edge | 
 No | 

 
 AGE [ 9 ] | 
 No | 
 Node-by-node | 
 Yes | 
 Node | 
 Yes | 

 
 DeepGMG [ 10 ] | 
 No | 
 Node-by-node | 
 Yes | 
 Node/Edge | 
 Yes | 

 
 BiGG [ 11 ] | 
 No | 
 Node-by-node | 
 No | 
 - | 
 No | 

 TABLE III: The Main Characteristics of Autoregressive Deep Graph Generators 
 
 
 Besides the attention-based methods reviewed above, other autoregressive non-recurrent DGGs either do not use attention at all, or the attention mechanism does not play a decisive role in their generation process. For example, DeepGMG [ 10 ] proposes a sequential graph generation process which can be seen as the following sequence of decisions: (1) whether to add a new node of a particular type or not (with probabilities provided by the f a ​ d ​ d ​ n ​ o ​ d ​ e f_{addnode} in Eq. ( 22 ), where h G h_{G} is the graph representation vector, and f a ​ n f_{an} is an MLP that maps h G h_{G} to the action output space), if a node type is selected (2) the model decides whether to continue connecting the newly added node to the existing graph or not (referring to Eq. ( 23 ), where h v ( T ) h_{v}^{(T)} is embedding of the new node v v after T T rounds of propagation in a graph neural
network, and f a ​ e f_{ae} is another MLP), if yes (3) it selects a node already in the graph and connects it to the new node (referring to the Eq. ( 24 ), where f s f_{s} maps pairs h u ( T ) h_{u}^{(T)} and h v ( T ) h_{v}^{(T)} to a score s u s_{u} ). The algorithm goes back to step (2) and repeats until the model decides not to add another edge. Finally, the algorithm goes back to step (1) to add subsequent nodes or terminate the process.

 

 
 | 
 f a ​ d ​ d ​ n ​ o ​ d ​ e ​ ( G ) = Softmax ​ ( f a ​ n ​ ( h G ) ) , f_{addnode}(G)=\text{Softmax}\ (f_{an}(h_{G})), | 
 | 
 (22) | 
 

 

 
 | 
 f a ​ d ​ d ​ e ​ d ​ g ​ e ​ ( G , v ) = σ ⁡ ( f a ​ e ​ ( h G , h v ( T ) ) ) , f_{addedge}(G,v)=\sigma(f_{ae}(h_{G},h_{v}^{(T)})), | 
 | 
 (23) | 
 

 

 
 | 
 s u = f s ​ ( h u ( T ) , h v ( T ) ) , ∀ u ∈ V f n ​ o ​ d ​ e ​ s ​ ( G , v ) = Softmax ​ ( s ) . \begin{split}s_{u}=f_{s}(h_{u}^{(T)},h_{v}^{(T)}),\ \ \forall u\in V\\
f_{nodes}(G,v)=\text{Softmax}\ (s).\end{split} | 
 | 
 (24) | 
 

 
 
 DeepGG [ 88 ] further extends DeepGMG [ 10 ] by adding the idea of finite state machines into the generation process. Furthermore, similar to the GraphRNN [ 2 ] , DeepGG learns the graph distribution from a sequence called construction sequence, which consists of graph evolutionary actions such as node addition, edge addition, and node deletion.

 
 
 Recently, BiGG [ 11 ] proposes an autoregressive model to increase scalability for generating sparse graphs. To learn a generative model, BiGG uses a single canonical ordering π ⁡ ( G ) \pi(G) to model each graph G, as in [ 10 ] , aiming to learn a lower bound on p(G):

 

 
 | 
 p ⁡ ( G ) = p ⁡ ( V ) ​ P ​ ( E | V ) = p ⁡ ( | V | = n ) ​ ∑ π p ⁡ ( A π ) ≈ p ⁡ ( | V | = n ) ​ p ​ ( A π ⁡ ( G ) ) , \small p(G)=p(V)P(E|V)=p(|V|=n)\sum_{\pi}p(A^{\pi})\approx p(|V|=n)p(A^{\pi(G)}), | 
 | 
 (25) | 
 

 where p ⁡ ( | V | = n ) p(|V|=n) can be directly estimated using an empirical distribution over the graph size. Therefore the goal is only to model p ⁡ ( A π ⁡ ( G ) ) p(A^{\pi(G)}) under a default canonical ordering, which will be denoted by p ⁡ ( A ) p(A) in the following. Considering that most real-world graphs are sparse, BiGG generates only the non-zero entries in A A in a row-wise manner to enhance efficiency and scalability; thus, the method adopts a recursive strategy inspired by R-MAT [ 89 ] , for generating each edge as illustrated in the left half of Figure 4 . To further improve efficiency, the authors propose to jointly generate all the connections of an arbitrary node u u (non-zero entries in the u u -th row of A A ) by autoregressively generating an edge-binary tree , as shown in the right half of Figure 4 . Finally, BiGG introduces the full autoregressive model that generates the entire adjacency matrix row by row. The full model utilizes the autoregressive models as building blocks:

 

 
 | 
 p ⁡ ( A ) = p ⁡ ( { 𝒩 u } u ∈ V ) = ∏ u ∈ V p ⁡ ( 𝒩 u | { 𝒩 u ′ : u ′ u } ) , p(A)=p(\{\mathcal{N}_{u}\}_{u\in V})=\prod_{u\in V}p(\mathcal{N}_{u}|\{\mathcal{N}_{u^{\prime}}:u^{\prime} u\}), | 
 | 
 (26) | 
 

 where 𝒩 u \mathcal{N}_{u} denotes the set of neighbors for node u u . More specifically, inspired by Fenwick tree [ 90 ] , the authors propose a data structure called row-binary forest to encode all the edge-binary trees generated so far, which will be used to generate new edge-binary tree for the current step.

 
 
 
 
 

## IV Autoencoder-Based Deep Graph Generators 

 
 This section reviews those approaches that employ whether autoencoders (AEs) or VAEs [ 91 ] to generate graph structures. In particular, a common practice in these methods is first to encode an input graph into a latent space using GNN [ 92 ] , GCN [ 49 ] , or their variants and then start to generate the graph from this latent space embedding. We divide the existing approaches based on their generation granularity level (i.e., adopting an all-at-once generation strategy, using valid substructures as building blocks, or generating graphs in a node-by-node fashion) into the following three subsections. The main characteristics of the most prominent autoencoder-based graph generators are presented in Table IV .

 
 

### IV-A One-Shot Generators 

 
 A series of autoencoder-based DGGs generate the entire graph all at once. VGAE [ 27 ] proposes a graph generation model that primarily aims to perform unsupervised learning on graphs based on the variational autoencoder [ 91 ] . Given a graph G G with adjacency matrix A A and node feature matrix X X , VGAE infers the latent matrix Z by a two-layer GCN [ 49 ] :

 

 
 | 
 q ⁡ ( Z | X , A ) = ∏ i = 1 n q ⁡ ( z i | X , A ) , with ​ q ​ ( z i | X , A ) = 𝒩 ⁡ ( z i | μ i , diag ​ ( σ i 2 ) ) , \small q(\textbf{Z}|X,A)=\prod_{i=1}^{n}q(\textbf{z}_{i}|X,A),\ \text{with}\ \ q(\textbf{z}_{i}|X,A)=\mathcal{N}(\textbf{z}_{i}|\mu_{i},\text{diag}(\sigma_{i}^{2})), | 
 | 
 (27) | 
 

 where μ = GCN μ ​ ( X , A ) \mu=\text{GCN}_{\mu}(X,A) is the matrix of mean vectors μ i \mu_{i} ; similarly log ⁡ σ = GCN σ ​ ( X , A ) \log\sigma=\text{GCN}_{\sigma}(X,A) . Then, the generative model is designed as a simple inner product of latent variables as follows:

 

 
 | 
 p ⁡ ( A | Z ) = ∏ i = 1 n ∏ j = 1 n p ⁡ ( A i ​ j | z i , z j ) , with ​ p ​ ( A i ​ j = 1 | z i , z j ) = σ ⁡ ( z i T ​ z j ) , \small p(A|\textbf{Z})=\prod_{i=1}^{n}\prod_{j=1}^{n}p(A_{ij}|\textbf{z}_{i},\textbf{z}_{j}),\ \text{with}\ \ p(A_{ij}=1|\textbf{z}_{i},\textbf{z}_{j})=\sigma(\textbf{z}_{i}^{T}\textbf{z}_{j}), | 
 | 
 (28) | 
 

 where σ ( . ) \sigma(.) is the logistic sigmoid function. The model parameters are then learned by optimizing the VAE objective. However, the authors also proposed a more straightforward, non-probabilistic, and autoencoder-based version of the method called GAE. The main limitation of VGAE is that it can only learn from a single input graph. GraphVAE [ 28 ] , on the other hand, proposes another VAE-based generative model that learns from a dataset of graphs. The method first embeds the input graph into continuous representation z using a graph convolution network [ 93 ] as the encoder q ϕ ​ ( z | G ) q_{\phi}(\textbf{z}|G) , where the dimensionality of z is relatively small in order to learn a high-level compression of the input data. Then, the decoder outputs a probabilistic fully-connected graph with a fairly small predefined maximum size directly at once denoted by G ~ \tilde{G} . The whole GraphVAE model is trained by minimizing the upper bound on negative log-likelihood as follows:

 

 
 | 
 ℒ θ , ϕ ( G ) = 𝔼 q ϕ ​ ( z | G ) [ − log p θ ( G | z ) ] + K L [ q ϕ ( z | G ) | | p ( z ) ] , \mathcal{L}_{\theta,\phi}(G)=\mathbb{E}_{q_{\phi}(\textbf{z}|G)}[-\log p_{\theta}(G|\textbf{z})]+KL[q_{\phi}(\textbf{z}|G)||p(\textbf{z})], | 
 | 
 (29) | 
 

 where q ϕ ​ ( z | G ) q_{\phi}(\textbf{z}|G) and p θ ​ ( G | z ) p_{\theta}(G|\textbf{z}) are the encoder posterior and the decoder generative
distributions respectively, and ϕ \phi and θ \theta are parameters to be learned. Since no particular ordering of nodes is imposed in neither G nor G ~ \tilde{G} , the authors further adopt an approximate graph matching algorithm for aligning G with G ~ \tilde{G} in order to compute the likelihood p θ ​ ( G | z ) p_{\theta}(G|\textbf{z}) in Eq. ( 29 ). However, the growth of GPU memory requirements, number of parameters, and graph matching complexity for larger graph sizes limit the applicability of GraphVAE only to generate smaller graphs.

 
 
 MPGVAE [ 29 ] further improves GraphVAE by building a message passing neural network (MPNN) into the encoder and decoder of a VAE, eliminating the need for complex graph matching algorithms. In particular, the method first encodes a molecular graph using a variant of MPNNs [ 94 ] combined with a graph attention [ 58 ] to aggregate the information over each node’s neighbors. The encoder then obtains a graph-level representation using the set2set model [ 95 ] and uses this representation to parametrize the posterior distribution q ϕ ​ ( z | G ) q_{\phi}(\textbf{z}|G) . Next, the decoder samples z ∼ q ϕ ​ ( z | G ) \textbf{z}\sim q_{\phi}(\textbf{z}|G) and projects it to a high dimensional space consisting of several vectors and then passes these vectors through an RNN to compute initial states for the graph nodes. Afterward, it uses an identical MPNN as the encoder to obtain the final representation for each edge and node. The decoder then reconstructs the graph by predicting the atom types and bond types based on these final representations.

 
 
 RGVAE [ 30 ] regularizes the framework of variational autoencoders to generate semantically valid graphs. To impose validity constraints in the training of VAEs, RGVAE transforms a constrained optimization problem to a regularized, unconstrained one by adding inequality constraints to the objective function of VAEs, which forms a Lagrangian function. More precisely, RGVAE minimizes the following loss function in each parameter update:

 

 
 | 
 ℒ = ℒ θ , ϕ ​ ( G ) + μ ​ ∑ i g i ​ ( θ , z ) + , where z ∼ p θ ​ ( z ) , \mathcal{L}=\mathcal{L}_{\theta,\phi}(G)+\mu\sum_{i}g_{i}(\theta,\textbf{z})_{+},\ \ \ \ \text{where}\ \ \ \textbf{z}\sim p_{\theta}(\textbf{z}), | 
 | 
 (30) | 
 

 where g i ​ ( θ , z ) ≤ 0 g_{i}(\theta,\textbf{z})\leq 0 denotes the i i -th validity constraint, g + = m ​ a ​ x ​ ( g , 0 ) g_{+}=max(g,0) , and ℒ θ , ϕ ​ ( G ) \mathcal{L}_{\theta,\phi}(G) is the standard VAE loss function as in Eq. ( 29 ). The training of RGVAE is illustrated in Figure 5 , where l l denotes the index of a training example and l ¯ \underline{l} denotes a synthetic example, utilized in the regularization term.

 
 
 Fig. 5: The framework of RGVAE [ 30 ] (reprinted with permission). The top flow corresponds to the standard VAE, while the bottom flow denotes the regularization, where a synthetic z ( l ¯ ) \textbf{z}^{(\underline{l})} is decoded to compute the constraints g i ​ ( θ , z ( l ¯ ) ) + g_{i}(\theta,\textbf{z}^{(\underline{l})})_{+} . 
 
 
 Graphite [ 31 ] proposes a latent variable generative model based on VAE for unsupervised representation learning in large graphs. The method only models graph structure, and any supplementary information such as node features X ∈ ℝ n × k X\in\mathbb{R}^{n\times k} is considered as conditioning evidence. To learn the model parameters θ \theta , Graphite maximizes a lower bound on log-likelihood of the observed adjacency matrix conditioned on X X :

 

 
 | 
 log ⁡ p θ ​ ( A | X ) ≥ 𝔼 q ϕ ​ ( Z | A , X ) ​ [ log ⁡ p θ ​ ( A , Z | X ) q ϕ ​ ( Z | A , X ) ] . \log p_{\theta}(A|X)\geq\mathbb{E}_{q_{\phi}(\textbf{Z}|A,X)}\Big[\log\frac{p_{\theta}(A,\textbf{Z}|X)}{q_{\phi}(\textbf{Z}|A,X)}\Big]. | 
 | 
 (31) | 
 

 In more detail, the authors take an encoding approach based on the mean-field approximation, which represents graph nodes in the latent space using a graph neural network. Next, they propose an iterative two-step approach as the decoding part: it first constructs an intermediate weighted graph A ^ \hat{A} from the latent matrix Z . Then, a parameterized graph neural network updates the latent matrix Z ∗ . The process alternates between these two steps to refine the graph gradually. More formally, given Z and X X , Graphite iterates over the following two operations:

 

 
 | 
 A ^ = Z Z ⊤ ‖ Z ‖ 2 + 11 ⊤ , Z ∗ = GNN θ ​ ( A ^ , [ Z , X ] ) , \hat{A}=\frac{\textbf{Z}\textbf{Z}^{\top}}{||\textbf{Z}||^{2}}+\textbf{11}^{\top},\ \ \ \textbf{Z}^{*}=\text{GNN}_{\theta}(\hat{A},[\textbf{Z},X]), | 
 | 
 (32) | 
 

 where an additional constant of 1 is added into the first operation to ensure entries are non-negative. Finally, it should be noted that similar to VGAE [ 27 ] , Graphite is also limited to learning from a single input graph.

 
 
 In addition to the aforementioned graph generative approaches, some initial steps have taken towards making DGGs interpretable. Stoehr et al. [ 96 ] propose to learn disentangled, interpretable latent variables corresponding to generative parameters of graphs. The main goal of learning such disentangled variables is to make the latent space more interpretable as each latent variable encodes one and only one data property. Therefore, the authors use a GCN as the encoder combined with a deconvolutional neural network as the decoder to minimize the loss function of β \beta -VAE [ 97 ] . In this setting, a higher value of β \beta yields the more orthogonalized latent space. To further enforce disentanglement of latent variables, the model also learns an additional parameter decoder h h , which maps latent variables to generative parameters as illustrated in Figure 6 . Recently, NED-VAE [ 32 ] proposes a more generalized generative approach for disentanglement learning on attributed graphs that uncovers the independent latent factors in both edges and nodes. In particular, NED-VAE aims to develop a model that can learn the joint distribution of the graph G G and three groups of generative independent latent variables, namely, z f , z e \textbf{z}_{f},\textbf{z}_{e} , and z g \textbf{z}_{g} each of which controls the properties of only nodes, only edges, and the joint patterns between them, respectively. Therefore, inspired by the β \beta -VAE [ 97 ] formulation and considering the independence resulted from the disentanglement assumption, the goal is to maximize the following objective function:

 

 
 | 
 ℒ ⁡ ( θ , ϕ , G , Z , β ) = 𝔼 q ϕ ​ ( Z | G ) ⁡ [ log ⁡ p θ ​ ( F | z f , z g ) ​ p θ ​ ( E | z e , z g ) ] − β D K ​ L ( q ϕ ( z f | F ) | | p ( z f ) ) − β D K ​ L ( q ϕ ( z e | E ) | | p ( z e ) ) − β D K ​ L ( q ϕ ( z g | E , F ) | | p ( z g ) ) , \small\begin{split}\mathcal{L}(\theta,\phi,G,\textbf{Z},\beta) =\mathop{\mathbb{E}_{q_{\phi}(\textbf{Z}|G)}}[\log p_{\theta}(F|\textbf{z}_{f},\textbf{z}_{g})p_{\theta}(E|\textbf{z}_{e},\textbf{z}_{g})]\\
 -\beta D_{KL}(q_{\phi}(\textbf{z}_{f}|F)||p(\textbf{z}_{f}))-\beta D_{KL}(q_{\phi}(\textbf{z}_{e}|E)||p(\textbf{z}_{e}))\\
 -\beta D_{KL}(q_{\phi}(\textbf{z}_{g}|E,F)||p(\textbf{z}_{g})),\end{split} | 
 | 
 (33) | 
 

 where E ∈ ℝ n × n × d E\in\mathbb{R}^{n\times n\times d} is the edge attributes tensor, and F ∈ ℝ n × k F\in\mathbb{R}^{n\times k} refers to the node attribute matrix. Based on the above objective, the authors propose an architecture consisting of three sub-encoders, namely, a node encoder, an edge encoder, and a node-edge co-encoder to model the distributions q ϕ ​ ( z f | F ) q_{\phi}(\textbf{z}_{f}|F) , q ϕ ​ ( z e | E ) q_{\phi}(\textbf{z}_{e}|E) , and q ϕ ​ ( z g | E , F ) q_{\phi}(\textbf{z}_{g}|E,F) , respectively, as depicted in Figure 7 . The architecture also consists of two sub-decoders: a node decoder, and an edge decoder to model p θ ​ ( F | z f , z g ) p_{\theta}(F|\textbf{z}_{f},\textbf{z}_{g}) and p θ ​ ( E | z e , z g ) p_{\theta}(E|\textbf{z}_{e},\textbf{z}_{g}) , respectively. The authors further propose multiple variant models to address different issues, including group-wise disentanglement, variable-wise disentanglement, and the trade-off between reconstruction error and disentanglement performance.

 
 
 Fig. 6: Architecture Overview of [ 96 ] (reprinted with permission). 
 
 
 Fig. 7: The architecture of NED-VAE [ 32 ] consisting of three sub-encoders, as well as two sub-decoders (reprinted with permission). 
 
 
 More recently, DGVAE [ 33 ] proposes to replace the commonly used Gaussian distribution in VAE-based graph generation models by the Dirichlet distribution as a prior for the latent variables, causing them to describe the graph cluster memberships, which as a result adds interpretability to the model. For this purpose, the authors adopt the same formulation as VGAE [ 27 ] in Eq. ( 27 ) for the encoding process except that they utilize a GNN variant, named Heatts, proposed by their own and employ the Laplace approximation [ 98 ] to model both q ⁡ ( z i | X , A ) q(\textbf{z}_{i}|X,A) and p ⁡ ( z i ) p(\textbf{z}_{i}) as Dirichlet distributions. Furthermore, DGVAE adopts a somewhat similar decoding strategy as VGAE [ 27 ] and proves that maximizing the reconstruction term of the model is equivalent to minimizing balanced graph cut , which further gives the authors the motivation for designing the Heatts.

 
 
 

### IV-B Substructure-Based Generators 

 
 There exists a number of works using valid chemical substructures as building blocks to generate more plausible molecular graphs. JT-VAE [ 34 ] adopts such a strategy by extending the variational autoencoder framework, which, as a consequence, avoids invalidity of the intermediate subgraphs. Specifically, JT-VAE first decomposes a molecular graph G into a junction tree τ G \tau_{G} to make the graph cycle-free, where each node in the tree represents a substructure of the molecule. Then, both G and τ G \tau_{G} are encoded to latent representations z G \textbf{z}_{G} and z τ \textbf{z}_{\tau} , respectively, using different encoders. More precisely, z τ \textbf{z}_{\tau} encodes the junction tree without capturing the exact mutual connections between substructures, while z G \textbf{z}_{G} encodes the graph to capture the fine-grained connectivities. Afterward, JT-VAE reconstructs the junction tree from its latent representation z τ \textbf{z}_{\tau} using a tree-structured decoder, where a tree is generated node-by-node, in a top-down manner. Next, the authors introduce a graph decoder to reproduce the molecular graph based on the underlying predicted junction tree. Since there are potentially many molecules corresponding to the same junction tree, the graph decoder learns how to assemble the subgraphs (nodes in the tree) to reconstruct the molecular graph. Furthermore, to generate molecules with desired properties, the method performs Bayesian optimization in the latent space.

 
 
 JT-VAE is basically designed to use small substructures as building blocks, which degrades its performance for generating larger molecules such as polymers. To address this issue, HierVAE [ 14 ] recently proposes a motif-based hierarchical graph encoder-decoder that employs significantly larger motifs as basic building blocks. To this end, the authors design an encoder that learns hierarchical representation for a given molecular graph G G in a fine-to-coarse fashion, from atoms to connected motifs. Then, it obtains the latent vector z G \textbf{z}_{G} by sampling from a Gaussian distribution parametrized using the acquired motif representations. The decoder, on the other hand, autoregressively generates a molecular graph in a coarse-to-fine fashion conditioned on z G \textbf{z}_{G} . More specifically, at each generation step, the decoder first predicts the next motif to be attached to the already generated graph. Then, it predicts the attachment points in the new motif, i.e., what atoms belong to the intersection of the new motif and its neighbor motifs. Finally, the decoder decides how the new motif should be attached to the current graph based on its predicted attachment points. The model parameters are learned by minimizing the VAE loss function formulated in Eq. ( 29 ). Furthermore, the authors extend the architecture to graph-to-graph translation [ 39 ] in order to induce desired properties in the generated molecules.

 
 
 MHG-VAE [ 35 ] guides VAE to always generate valid molecular graphs by proposing molecular hypergraph grammar (MHG), a special case of hyperedge replacement grammar (HRG) [ 99 ] for generating molecular hypergraphs , to encode chemical constraints. In particular, the proposed encoder consists of three parts as follows:

 

 
 | 
 E ​ n ​ c = E ​ n ​ c N ∘ E ​ n ​ c G ∘ E ​ n ​ c H , Enc=Enc_{N}\circ Enc_{G}\circ Enc_{H}, | 
 | 
 (34) | 
 

 where E ​ n ​ c H Enc_{H} first encodes a molecular graph into a molecular hypergraph, and E ​ n ​ c G Enc_{G} represents the molecular hypergraph as a parse tree by leveraging MHG. Then, E ​ n ​ c N Enc_{N} encodes the previously generated parse tree into the latent continuous space using a seq2seq GVAE [ 100 ] . The decoder, on the other hand, acts as an inversion to the encoder and applies production rules, including those with chemical substructure terminals. Finally, using Bayesian optimization, MHG-VAE optimizes the latent continuous space (and its corresponding molecules) towards desired properties.

 
 
 MoleculeChef [ 15 ] proposes to generate molecular graphs using a set of common reactant molecules as building blocks to address the synthesizability issue. In particular, the encoder maps from a multiset of reactants to a distribution over latent space. This is done by using GGNNs [ 101 ] to embed each reactant molecule separately, which are further summed to form one embedding for the whole multiset. A feed-forward network is then used to parameterize a Gaussian distribution over the latent space. The decoder, on the other hand, autoregressively maps from the latent space to a multiset of reactants using an RNN, where the latent vector z initializes its hidden layer, and at each generation step, it outputs one reactant or halts the process. Afterward, a reaction predictor [ 102 ] predicts how the previously generated reactants produce a final molecule as illustrated in Figure 8 . To learn the model parameters, MoleculeChef minimizes the following WAE [ 103 ] objective function:

 

 
 | 
 ℒ θ , ϕ ​ ( G ) = 𝔼 G ∼ 𝒟 ​ 𝔼 q ϕ ​ ( z | G ) ​ [ − log ⁡ p θ ​ ( G | z ) ] + λ ​ D ​ ( 𝔼 G ∼ 𝒟 ​ [ q ϕ ​ ( z | G ) ] , p ⁡ ( z ) ) , \begin{split}\mathcal{L}_{\theta,\phi}(G) =\mathbb{E}_{G\sim\mathcal{D}}\mathbb{E}_{q_{\phi}(\textbf{z}|G)}[-\log p_{\theta}(G|\textbf{z})]\\
 +\lambda D(\mathbb{E}_{G\sim\mathcal{D}}[q_{\phi}(\textbf{z}|G)],p(\textbf{z})),\end{split} | 
 | 
 (35) | 
 

 where D D is a divergence measure, namely the maximum mean discrepancy (MMD). Finally, the authors propose to optimize the molecular properties in the continuous latent space in a similar manner to CGVAE [ 16 ] , which we will discuss in the following subsection.

 
 
 Fig. 8: An overview of MoleculeChef [ 15 ] (reprinted with permission). 
 
 
 

### IV-C Node-by-Node Generators 

 
 Fig. 9: The framework of GraphVRNN [ 18 ] (reprinted with permission). 
 
 
 Besides the methods reviewed above, some other autoencoder-based DGGs use graph nodes as building blocks, and their decoders adopt autoregressive generation strategies. CGVAE [ 16 ] is one of these methods whose encoder first samples a latent vector z v \textbf{z}_{v} for each node v v of an input graph from a normal distribution parametrized using GGNNs [ 101 ] . Then, the decoder starts from these vectors and sequentially generates a graph node-by-node with the help of two decision functions, namely, focus and expand . More precisely, the focus function determines the next node to be added into the graph, and the expand function iteratively chooses edges to add from the currently focused node until particular stop criteria met, and once the generated subgraph changes, all node representations get updated. Moreover, CGVAE employs a valency masking mechanism as part of expand function in the case of molecule generation to guarantee chemical validity. The authors also propose to optimize graph properties by minimizing L 2 L_{2} distance between some numerical property Q and a differentiable gated regression score R ⁡ ( G ) R(G) as formulated in Eq. ( 36 ), using gradient ascent in the continuous latent space:

 

 
 | 
 R ⁡ ( G ) = ∑ v σ ⁡ ( g 1 ​ ( z v ) ) . g 2 ​ ( z v ) , R(G)=\sum_{v}\sigma(g_{1}(\textbf{z}_{v})).g_{2}(\textbf{z}_{v}), | 
 | 
 (36) | 
 

 where g 1 g_{1} and g 2 g_{2} are neural networks.

 
 
 DEFactor [ 17 ] proposes an encoder-decoder based architecture with a recurrent autoregressive decoder for conditional graph generation. In this regard, the encoder first applies a GCN [ 49 ] to obtain node embeddings, which are then aggregated by an LSTM to compute the graph-level representation z . Then, the two-step decoder first employs another LSTM in order to autoregressively generate embeddings for each of the graph nodes based on the computed z :

 

 
 | 
 h i = f t ​ r ​ a ​ n ​ s ​ ( g i ​ n ​ ( [ z , s i − 1 ] ) , h i − 1 ) , s i = f e ​ m ​ b ​ e ​ d ​ ( [ h i , z ] ) , h_{i}=f_{trans}(g_{in}([\textbf{z},s_{i-1}]),h_{i-1}),\ \ s_{i}=f_{embed}([h_{i},\textbf{z}]), | 
 | 
 (37) | 
 

 where f t ​ r ​ a ​ n ​ s f_{trans} is implemented by an LSTM [ 80 ] , g i ​ n g_{in} and f e ​ m ​ b ​ e ​ d f_{embed} are MLPs, and s i s_{i} is the node embedding generated at timestep i i . Next, the decoder establishes an edge factorization approach to compute the existence probability of an edge of type k k between nodes u u and v v as follows:

 

 
 | 
 p ⁡ ( E u , v , k | s u , s v ) = σ ⁡ ( s u ⊺ ​ D k ​ s v ) , p(E_{u,v,k}|s_{u},s_{v})=\sigma(s_{u}^{\intercal}D_{k}s_{v}), | 
 | 
 (38) | 
 

 where D k D_{k} is the diagonal matrix of learnable factors for the k k -th edge type. Finally, DEFactor makes the generation process conditional by first concatenating the condition vector C C with z . It then utilizes a pre-trained discriminator to assess the property C C in the graphs generated by the decoder in the training phase.

 
 
 NeVAE [ 36 ] proposes a probabilistic and permutation invariant encoder that is relatively similar to other graph representation learning algorithms, such as GraphSAGE [ 50 ] and GCNs [ 49 ] , except that it uses variational inference to learn the aggregator functions, which is further proved that makes the resulting embeddings well suited for the molecular graph generation task. The authors then introduce a probabilistic decoder that first samples the number of graph nodes from a Poisson distribution. It also samples a latent vector z v \textbf{z}_{v} per node v ∈ V v\in V from 𝒩 ⁡ ( 0 , I ) \mathcal{N}(\textbf{0},\textbf{I}) . Then, for each node v v , the decoder passes z v \textbf{z}_{v} through a neural network followed by a softmax classifier to determine node features, i.e., the atom type. Next, the total number of graph edges is sampled from a Poisson distribution parametrizes by another neural network conditioned on all latent vectors Z . Thereafter, the decoder samples graph edges one by one from a softmax distribution among all potential edges not generated that far, and similar to CGVAE [ 16 ] , uses a set of binary masks to guarantee some local structural and functional properties. Finally, it determines the edge type by sampling from another softmax distribution with different binary masks. These masks get updated every time the decoder generates a new edge. The model’s objective is similar to that of conventional VAE-based methods plus maximizing the Poisson distribution log-likelihood, which models the number of graph nodes. Moreover, similar to JT-VAE [ 34 ] , NeVAE utilizes Bayesian optimization over the continuous latent space to discover molecules with desirable properties.

 
 
 
 
 
 Method | 
 Type | 
 Input | 
 
 
 
 Generation 
 
 Strategy 
 | 
 
 
 
 Attention 
 
 Mechanism 
 | 
 Features | 
 
 
 
 Conditional 
 
 Generation 
 | 

 
 VGAE [ 27 ] | 
 AE/VAE | 
 one single graph | 
 All at Once | 
 No | 
 Node | 
 No | 

 
 GraphVAE [ 28 ] | 
 VAE | 
 dataset of graphs | 
 All at Once | 
 No | 
 Node/Edge | 
 Yes | 

 
 MPGVAE [ 29 ] | 
 VAE | 
 dataset of graphs | 
 All at Once | 
 Yes | 
 Node/Edge | 
 Yes | 

 
 RGVAE [ 30 ] | 
 VAE | 
 dataset of graphs | 
 All at Once | 
 No | 
 Node/Edge | 
 No | 

 
 Graphite [ 31 ] | 
 AE/VAE | 
 one single graph | 
 All at Once | 
 No | 
 Node | 
 No | 

 
 NED-VAE [ 32 ] | 
 β \beta -VAE | 
 dataset of graphs | 
 All at Once | 
 No | 
 Node/Edge | 
 No | 

 
 DGVAE [ 33 ] | 
 VAE | 
 one single graph | 
 All at Once | 
 No | 
 Node | 
 No | 

 
 JT-VAE [ 34 ] | 
 VAE | 
 dataset of graphs | 
 Substructure-Based | 
 No | 
 Node/Edge | 
 No | 

 
 HierVAE [ 14 ] | 
 VAE | 
 dataset of graphs | 
 Substructure-Based | 
 Yes | 
 Node/Edge | 
 Yes | 

 
 MHG-VAE [ 35 ] | 
 VAE | 
 dataset of graphs | 
 Substructure-Based | 
 No | 
 Node/Edge | 
 No | 

 
 MoleculeChef [ 15 ] | 
 WAE | 
 dataset of graphs | 
 Substructure-Based | 
 No | 
 Node/Edge | 
 No | 

 
 CGVAE [ 16 ] | 
 VAE | 
 dataset of graphs | 
 Node-by-Node | 
 No | 
 Node/Edge | 
 No | 

 
 DEFactor [ 17 ] | 
 AE | 
 dataset of graphs | 
 Node-by-Node | 
 No | 
 Node/Edge | 
 Yes | 

 
 NeVAE [ 36 ] | 
 VAE | 
 dataset of graphs | 
 Node-by-Node | 
 No | 
 Node/Edge | 
 No | 

 
 GraphVRNN [ 18 ] | 
 VAE | 
 dataset of graphs | 
 Node-by-Node | 
 No | 
 Node | 
 No | 

 
 Lim et al. [ 19 ] | 
 VAE | 
 dataset of graphs | 
 Node-by-Node | 
 No | 
 Node/Edge | 
 Yes | 

 
 TABLE IV: The Main Characteristics of Autoencoder-Based Deep Graph Generators 
 
 
 GraphVRNN [ 18 ] proposes a VAE-based extension to GraphRNN [ 2 ] to learn the joint probability distributions of graph structure as well as the underlying node attributes by rewriting the likelihood function in Eq. ( 3 ) as follows:

 

 
 | 
 p ( S π , X π ) = ∏ i = 1 n + 1 p ( S i π , X i π | S i π , X i π ) , p(S^{\pi},X^{\pi})=\prod_{i=1}^{n+1}p(S_{i}^{\pi},X_{i}^{\pi}|S_{ i}^{\pi},X_{ i}^{\pi}), | 
 | 
 (39) | 
 

 where X ∈ ℝ n × k X\in\mathbb{R}^{n\times k} is the attribute matrix. Then, the authors adopt an autoregressive variational autoencoder to capture the latent factors over graphs with complicated structural dependencies by optimising the lower bound of the likelihood as follows:

 

 
 | 
 ℒ θ , ϕ , ψ ​ ( S π , X π ) = ∑ i 𝔼 z i ∼ q ψ ( . ) [ log p θ ( S i π , X i π | S i π , X i π , z ≤ i ) ] − β D K ​ L ( q ψ ( z i | S ≤ i π , X ≤ i π ) | | p ϕ ( z i | S i π , X i π ) ) , \begin{split}\mathcal{L}_{\theta,\phi,\psi}(S^{\pi},X^{\pi}) =\sum_{i}\mathop{\mathbb{E}_{\textbf{z}_{i}\sim q_{\psi}(.)}}[\log p_{\theta}(S_{i}^{\pi},X_{i}^{\pi}|S_{ i}^{\pi},X_{ i}^{\pi},\textbf{z}_{\leq i})]\\
 -\beta D_{KL}(q_{\psi}(\textbf{z}_{i}|S_{\leq i}^{\pi},X_{\leq i}^{\pi})||p_{\phi}(\textbf{z}_{i}|S_{ i}^{\pi},X_{ i}^{\pi})),\end{split} | 
 | 
 (40) | 
 

 where q ψ ​ ( z i | S ≤ i π , X ≤ i π ) q_{\psi}(\textbf{z}_{i}|S_{\leq i}^{\pi},X_{\leq i}^{\pi}) and p ϕ ​ ( z i | S i π , X i π ) p_{\phi}(\textbf{z}_{i}|S_{ i}^{\pi},X_{ i}^{\pi}) are the proposal and the prior distributions in conditional VAE (CVAE) [ 104 ] formulation, respectively, and D K ​ L D_{KL} is the Kullback-Leibler (KL) divergence that is tuned by the β \beta hyperparameter. An overview of the GraphVRNN framework is depicted in Figure 9 .

 
 
 Lim et al. [ 19 ] utilize a combination of VAE and DeepGMG [ 10 ] for generating molecular graphs with desired properties containing an arbitrary input scaffold in their structure. To this purpose, the encoder uses a variant of the interaction network [ 105 ] , [ 94 ] to obtain a representation vector h G h_{G} for the input graph, which will be further used to parametrize a normal distribution to sample a latent vector z . The decoder, on the other hand, takes a scaffold S S as input and extends it by making sequential decisions of node and edge additions in the same way as DeepGMG [ 10 ] , except that it also incorporates the latent vector z in the graph propagation process so that updating node and edge features during the generation is directly affected by z . Moreover, the model makes it possible to conduct the process towards generating molecules with desired properties by concatenating the corresponding condition vector C C with z . After the training with the VAE objective finishes, one could give a scaffold S S as well as a condition vector C C concatenated with a z sampled from the standard normal distribution to the decoder in order to get a generated molecule with optimized properties.

 
 
 
 

## V RL-Based Deep Graph Generators 

 
 This section provides a detailed review of generative approaches utilizing reinforcement learning algorithms to induce desired properties in the generated graphs. Table V gives a comparison among these methods from multiple aspects.

 
 
 GCPN [ 20 ] proposes a stepwise approach for molecular graph generation, formulating the problem as a Markov Decision Process to train an RL agent in a chemistry-aware environment. Thus, at each generation step t t , the method first takes the intermediate graph G t G_{t} and the set of scaffolds C C as input and computes the state s t s_{t} by applying a GCN variant that supports multiple edge types. It then samples an action a t a_{t} from the policy π θ \pi_{\theta} based on the obtained node embeddings, which can be either to add a new scaffold subgraph or connect two nodes already in the graph. Next, the action will be further processed by the state transition dynamics, and if it violates chemical rules, it will be rejected so that the state stays unchanged. After that, GCPN utilizes two types of rewards to guide the RL agent, namely, intermediate and final rewards, where the former consists of a stepwise validity reward that encourages the generation process to obey chemical valency rules, and an adversarial reward, which employs the GAN framework [ 41 ] to ensure similarity between real molecules and those to be generated. On the other hand, the final reward includes a domain-specific reward for molecular property optimization and a similar adversarial reward. Finally, the authors adopt Proximal Policy Optimization (PPO) [ 106 ] to optimize the policy network parameters. An overview of the method is depicted in Figure 10 , where each row corresponds to one step in the generation process.

 
 
 Fig. 10: An overview of GCPN [ 20 ] (reprinted with permission). 
 
 
 Subsequently, several methods propose extensions to GCPN. For example, Shi et al. [ 21 ] utilize a combination of general semantic features extracted from the SMILES representations of molecules and their graph representations in order to form more comprehensive states during the generation process. To this end, the authors propose an architecture consisting of a SMILES encoder and an action generator. First, the encoder obtains context vector z z from an input SMILES string, which will be further processed by two attention mechanisms, namely action-attention and graph-attention, to get the enhanced context vector z ~ \tilde{z} . Then, the model concatenates the current graph state s t s_{t} with z ~ \tilde{z} to pass a heterogeneous state into the action generator that has the same generation mechanism as GCPN, except that it does not involve adversarial rewards. Furthermore, the model is trained in two stages. The supervised learning stage learns an initialization for the model parameters to alleviate the instability of an RL agent training by minimizing the following objective function:

 

 
 | 
 J = − 1 M ∑ m = 1 M log 1 N ∑ n = 1 N ∑ t log P ( a t | z ~ , s t ) + D K ​ L ( P z | | P 0 ) , J=-\frac{1}{M}\sum_{m=1}^{M}\log\frac{1}{N}\sum_{n=1}^{N}\sum_{t}\log P(a_{t}|\tilde{z},s_{t})+D_{KL}(P_{z}||P_{0}), | 
 | 
 (41) | 
 

 where M M is the number of molecules in the training dataset, N N denodes the number of sampled trajectories for generating each molecule, and P z P_{z} and P 0 P_{0} are the distribution of the learned context vector and a prior distribution, respectively. Afterwards, the reinforcement learning stage further optimizes the process towards generating molecular graphs with desired properties. Figure 11 provides an overview of this framework.

 
 
 Fig. 11: An overview of the framework proposed by Shi et al. [ 21 ] . In the supervised learning phase, only the part in the gray box is trained. On the other hand, the whole architecture is involved in the reinforcement learning stage. 
 
 
 Karimi et al. [ 22 ] propose another extension to GCPN for drug-combination design, which is a key part of combination therapy. To this end, the authors first develop Hierarchical Variational Graph Auto-Encoders (HVGAE) to embed prior knowledge such as gene-gene, gene-disease, and disease-disease networks to acquire more accurate disease representations. Then, they formulate the problem as generating a set of graphs 𝒢 = { G ( k ) } k = 1 K \mathcal{G}=\{G^{(k)}\}_{k=1}^{K} conditioned on the learned disease representations by employing a similar generation strategy as GCPN, with the difference that in addition to chemical validity and adversarial rewards, they design a reward to encourage generating disease-specific drug combinations.

 
 
 More recently, DeepGraphMolGen [ 23 ] combines GCPN with a molecular property prediction network implemented as a GCN followed by a feedforward layer, which provides GCPN with an extra chemical reward to tilt the process towards generating molecules with additional property, i.e., binding potency to dopamine transporters.

 
 
 
 
 
 
 Method 
 | 
 
 
 
 Generation 
 
 Strategy 
 | 
 
 
 
 Attention 
 
 Mechanism 
 | 
 Features | 
 
 
 
 Conditional 
 
 Generation 
 | 
 
 
 Action 
 | 
 
 
 Reward 
 | 

 
 
 
 GCPN [ 20 ] 
 | 
 Sequential | 
 No | 
 Node/Edge | 
 No | 
 
 
 Link prediction 
 | 
 
 
 Domain-specific + GAN 
 | 

 
 
 
 Shi et al. [ 21 ] 
 | 
 Sequential | 
 Yes | 
 Node/Edge | 
 No | 
 
 
 Link prediction 
 | 
 
 
 Domain-specific 
 | 

 
 
 
 Karimi et al. [ 22 ] 
 | 
 Sequential | 
 Yes | 
 Node/Edge | 
 Yes | 
 
 
 Link prediction 
 | 
 
 
 Domain-specific + GAN + A reward for drug combinations 
 | 

 
 
 
 DeepGraphMolGen [ 23 ] 
 | 
 Sequential | 
 No | 
 Node/Edge | 
 No | 
 
 
 Link prediction 
 | 
 
 
 Domain-specific + GAN + Property reward (by the property prediction network) 
 | 

 
 
 
 GraphOpt [ 24 ] 
 | 
 Sequential | 
 No | 
 Node/Edge | 
 No | 
 
 
 Link prediction 
 | 
 
 
 Learning a reward function via inverse reinforcement learning 
 | 

 
 
 
 MNCE-RL [ 25 ] 
 | 
 Sequential | 
 No | 
 Node/Edge | 
 No | 
 
 
 Production rule selection 
 | 
 
 
 Domain-specific + A reward regarding the number of generation steps 
 | 

 
 
 
 GEGL [ 26 ] 
 | 
 Sequential | 
 No | 
 Node/Edge | 
 No | 
 
 
 Generating a molecule 
 | 
 
 
 Domain-specific 
 | 

 TABLE V: The Main Characteristics of RL-Based Deep Graph Generators 
 
 
 Besides GCPN and its subsequent approaches, there exist other RL-based methods that generate graph structures by taking different strategies. GraphOpt [ 24 ] models graph formation via a Markov Decision Process, aiming to learn both a graph construction procedure Π \Pi and a usually unknown latent objective function ℱ : G → ℝ \mathcal{F}:G\rightarrow\mathbb{R} that reflects the underlying graph formation mechanism. Therefore, inspired by [ 107 ] , the authors formulate the following objective:

 

 
 | 
 Π ∗ = argmin Π ⁡ max ℱ [ ℱ ⁡ ( G ) − ℱ ⁡ ( Π ⁡ ( V ) ) ] , ℱ o ​ p ​ t = argmax ℱ ⁡ [ ℱ ⁡ ( G ) − ℱ ⁡ ( Π ∗ ​ ( V ) ) ] , \begin{split}\Pi^{*} =\mathop{\text{argmin}}_{\Pi}\mathop{\max}_{\mathcal{F}}[\mathcal{F}(G)-\mathcal{F}(\Pi(V))],\\
\mathcal{F}_{opt} =\mathop{\text{argmax}}_{\mathcal{F}}[\mathcal{F}(G)-\mathcal{F}(\Pi^{*}(V))],\end{split} | 
 | 
 (42) | 
 

 where ℱ o ​ p ​ t \mathcal{F}_{opt} assigns the highest score to the observed graphs compared to all other ones, and optimization over ℱ \mathcal{F} is, in fact, a search for the reward function via inverse reinforcement learning (IRL) [ 108 ] . In other words, GraphOpt learns a reward function, which is in contrast to most of the RL frameworks that utilize an existing one. The optimal construction procedure, on the other hand, tries to construct a graph G ′ = Π ∗ ​ ( V ) G^{\prime}=\Pi^{*}(V) in a sequential link formation process given node-set V V , which is expected to be the most similar graph to the observed one using ℱ \mathcal{F} as the similarity measure. To this end, the authors propose a continuous latent action space to infer a link formation action a t a_{t} at each time step t t by first sampling two vectors a ( 1 ) a^{(1)} and a ( 2 ) a^{(2)} from a normal distribution parametrized based on the current graph state s t s_{t} , which is computed by a GNN [ 92 ] . They then choose two graph nodes with the most similar embeddings to the obtained vectors to construct an edge.

 
 
 MNCE-RL [ 25 ] proposes a graph convolutional policy network with a novel GCN architecture for generating molecules with optimized properties, which, similar to MHG-VAE [ 35 ] , utilizes grammar rules to guarantee the validity of molecules. To this end, the authors first extend the NCE graph grammar [ 109 ] to make it applicable for generating molecules. They then infer the production rules of the grammar from a set of input molecules. Next, the RL-based generation process starts whose action space consists of the set of legal production rules, and at each step, the policy samples a rule based on the node features obtained by applying the proposed GCN on the intermediate graph. A domain-specific reward guides the process towards generating desirable molecules. Moreover, MNCE-RL assigns a negative reward when the number of steps exceeds a threshold to avoid prolonging the generating process.

 
 
 GEGL [ 26 ] proposes to incline a deep neural network called neural apprentice policy towards generating molecules with desired properties. In this respect, the apprentice policy first generates a set of molecules by a SMILES-based LSTM and stores them into a fixed-size max-reward priority queue 𝒬 \mathcal{Q} . Then, a genetic expert policy utilizes the content of 𝒬 \mathcal{Q} as seed molecules and applies two genetic operators, namely, the graph-based mutation and crossover [ 110 ] , to them and stores the generated molecules in another priority queue denoted by 𝒬 e ​ x \mathcal{Q}_{ex} . After generating each sample, both 𝒬 \mathcal{Q} and 𝒬 e ​ x \mathcal{Q}_{ex} are updated so that they always contain molecules with the highest rewards. Next, the apprentice policy updates its model’s parameters by learning to imitate the molecules stored in 𝒬 ∪ 𝒬 e ​ x \mathcal{Q}\cup\mathcal{Q}_{ex} , and the whole procedure repeats iteratively. This way, the expert policy guides the apprentice policy to generate molecules with preferred properties.

 
 
 

## VI Adversarial Deep Graph Generators 

 
 This section reviews methods employing generative adversarial networks (GANs) [ 41 ] to generate either molecular or non-molecular graph structures. To conduct a more accurate study, we divide the existing approaches into multiple subsections. Moreover, Table VI provides a multifaceted comparison of them.

 
 

### VI-A Random Walk-Based Methods 

 
 A series of works focus on generating random walks instead of the entire graph, as graph random walks are invariant under node reordering. In this respect, NetGAN [ 42 ] introduces the first implicit generative model for graphs that learns the distribution of biased random walks over a single graph using the WGAN framework [ 111 ] . In particular, NetGAN first samples a collection of random walks using the biased second-order strategy [ 112 ] to prepare the model’s training data. Then, the generator learns to sequentially generate random walks node-by-node using the LSTM [ 80 ] architecture, which is initialized by a latent vector z sampled from a standard normal distribution. Meanwhile, the discriminator decides whether a random walk is real or not after processing its entire node sequence by another LSTM. After training finishes, the authors construct the adjacency matrix of a new graph using multiple generated random walks. Further to this, a number of generative approaches have been proposed inspired by the idea of NetGAN or extending it. For example, STGGAN [ 113 ] adopts a similar generating scheme for spatial-temporal graphs.

 
 
 MMGAN [ 43 ] generalizes NetGAN to capture higher-order connectivity patterns by introducing multiple types of random walks, each biased towards different motif structures. To simplify the process, MMGAN focuses only on 3-node motifs and proposes an architecture consisting of three GANs, namely, NetGAN that considers pairwise relationships, and two other motif-based GANs. The random walks generated by each of the three GANs are then combined to construct the output graph.

 
 
 S HADOW C AST [ 44 ] proposes another extension to NetGAN in order to make the generation process controllable, which can be considered as a step towards generating graphs with more explainable properties. To this end, the authors first define a graph called s ​ h ​ a ​ d ​ o ​ w shadow with the same structure as the original one but with different node labels so that these node-level properties can control the generation process. Then, they expand the architecture of NetGAN by adding a sequence-to-sequence model called shadow caster, which is implemented by an LSTM [ 80 ] . Specifically, the shadow caster takes in sampled walks from the shadow network and generates synthetic shadow walks of preferred distribution to control the generation process. Next, these model-generated shadow walks are fed into both generator and discriminator as conditions, which are finally trained using the conditional GAN [ 114 ] framework.

 
 
 

### VI-B Graph-Based Methods 

 
 
 
 Method | 
 Input | 
 
 
 
 Generation 
 
 Strategy 
 | 
 
 
 
 Attention 
 
 Mechanism 
 | 
 Features | 
 
 
 
 Conditional 
 
 Generation 
 | 

 
 NetGAN [ 42 ] | 
 one single graph | 
 Sequential | 
 No | 
 - | 
 No | 

 
 MMGAN [ 43 ] | 
 one single graph | 
 Sequential | 
 No | 
 - | 
 No | 

 
 S HADOW C AST [ 44 ] | 
 one single graph | 
 Sequential | 
 No | 
 Node | 
 Yes | 

 
 MolGAN [ 40 ] | 
 dataset of graphs | 
 All at Once | 
 No | 
 Node/Edge | 
 No | 

 
 CONDGEN [ 38 ] | 
 dataset of graphs | 
 All at Once | 
 No | 
 - | 
 Yes | 

 
 TSGG-GAN [ 45 ] | 
 dataset of graphs | 
 All at Once | 
 No | 
 Node | 
 Yes | 

 
 VJTNN + GAN [ 39 ] | 
 dataset of graphs | 
 Sequential | 
 Yes | 
 Node/Edge | 
 No | 

 
 Mol-CycleGAN [ 46 ] | 
 dataset of graphs | 
 Sequential | 
 No | 
 Node/Edge | 
 No | 

 
 Misc-GAN [ 47 ] | 
 one single graph | 
 Not mentioned | 
 No | 
 - | 
 No | 

 TABLE VI: The Main Characteristics of Adversarial Deep Graph Generators 
 
 
 Unlike random walk-based approaches, most of the existing adversarial graph generators deal with the entire graph. Here, we study these methods in two categories.

 
 

#### VI-B 1 General Graph-Based Adversarial DGGs

 
 MolGAN [ 40 ] proposes the first implicit generative model for small molecular graphs. In this respect, its generator first takes a latent vector z sampled from 𝒩 ⁡ ( 0 , I ) \mathcal{N}(0,I) . Then, it outputs a probabilistic graph all at once using an MLP in a way similar to GraphVAE [ 28 ] , which, as a consequence, limits the model to generate graphs of a predefined maximum size. However, in contrast to GraphVAE, MolGAN does not need to perform an expensive graph matching algorithm, as it makes the model likelihood-free using the GAN framework. Next, a permutation-invariant discriminator tries to distinguish between generated graphs and real ones using a combination of the Relational-GCN [ 115 ] and an MLP. The authors train the discriminator using the WGAN [ 111 ] objective, while they combine a reinforcement learning objective with that of the WGAN to train the generator, aiming at inclining the process towards generating molecules with desired chemical properties. More precisely, the authors employ a deterministic policy gradient algorithm, namely, DDPG [ 116 ] , to maximize the reward, which is approximated by a reward network with the same architecture as the discriminator. The overall architecture of MolGAN is shown in Figure 12 .

 
 
 LGGAN [ 117 ] adopts a similar generator to that of MolGAN. However, its discriminator uses JK-Net [ 118 ] to compute graph embeddings and outputs both the graph label and the probability of the graph being real. Moreover, to incorporate the class information, the authors utilize the AC-GAN [ 119 ] framework.

 
 
 Fig. 12: An overview of MolGAN [ 40 ] (reprinted with permission). 
 
 
 CONDGEN [ 38 ] proposes a model of graph variational generative adversarial nets for conditional structure generation. It addresses both the challenges of permutation-invariance and context-structure conditioning indicated by the information of attributes or labels in the networks. The method first applies the trick of latent space conjugation to the base VGAE [ 27 ] model in order to convert its node-level encoding into a permutation-invariant graph-level one that allows learning from a dataset of graphs with variable sizes, which is a notable improvement over the VGAE. More precisely, μ \mu and σ \sigma in Eq. ( 27 ) are replaced by the following parameters:

 

 
 | 
 q ⁡ ( z i | X , A ) = 𝒩 ⁡ ( z ¯ | μ ¯ , diag ​ ( σ ¯ 2 ) ) , q(\textbf{z}_{i}|X,A)=\mathcal{N}(\bar{\textbf{z}}|\bar{\mu},\text{diag}(\bar{\sigma}^{2})), | 
 | 
 (43) | 
 

 where μ ¯ = 1 n ​ ∑ i = 1 n g μ ​ ( X , A ) i \bar{\mu}=\frac{1}{n}\sum_{i=1}^{n}g_{\mu}(X,A)_{i} and σ ¯ 2 = 1 n 2 ​ ∑ i = 1 n g σ ​ ( X , A ) i 2 \bar{\sigma}^{2}=\frac{1}{n^{2}}\sum_{i=1}^{n}g_{\sigma}(X,A)_{i}^{2} . However, the process is still not completely permutation-invariant because the reconstruction loss of the VGAE is computed between the generated adjacency matrix A ′ A^{\prime} and the original matrix A A , which may be under different node permutations. Therefore, the authors propose a GCN-based discriminator to enforce the structural similarity between the generated and the true adjacency matrices and learn its parameters by optimizing the GAN objective. Thus, the encodings GCN D ​ ( A ) \text{GCN}_{D}(A) and GCN D ​ ( A ′ ) \text{GCN}_{D}(A^{\prime}) computed by the discriminator, become permutation-invariant and the reconstruction loss can be computed as ‖ GCN D ​ ( A ) − GCN D ​ ( A ′ ) ‖ 2 2 ||\text{GCN}_{D}(A)-\text{GCN}_{D}(A^{\prime})||^{2}_{2} . Moreover, according to [ 114 ] , the authors use the concatenation of condition vector C C and latent variable Z to enable conditional structure generation. Then, motivated by CycleGAN [ 120 ] , they further enforce mapping consistency between the graph context and the structure spaces by sharing the parameters in the two GCN networks, namely, the GCNs in the graph encoder and the discriminator.

 
 
 More recently, TSGG-GAN [ 45 ] adopts a time series conditioned generative model that aims to generate a graph given an input multivariate time series, where each time series acts as context information associated with one of the graph nodes. This is particularly the case when it is straightforward to obtain node-level information, while the underlying network is totally unknown. To this end, using SRU [ 121 ] to extract the information of the time series followed by an MLP, the generator generates the entire graph all at once. At the same time, the discriminator takes a pair of a multivariate time series and a graph as inputs, which are then processed using SRU and GCN, respectively. Thereafter, the discriminator utilizes Neural Tensor Networks (NTN) [ 122 ] to measure the similarity between the time series and the graph and decides whether the graph is real or not.

 
 
 

#### VI-B 2 Graph-to-Graph Translators

 
 In addition to the aforementioned methods, there exist other approaches trying to generate a new graph based on an initial one. VJTNN [ 39 ] proposes a graph-to-graph translation model that learns a mapping from a source molecular graph X X to a target graph Y Y with enhanced chemical properties by utilizing a similar encoder-decoder architecture as JT-VAE [ 34 ] whose tree decoding process is further enriched by adding an attention mechanism. More specifically, VJTNN augments the basic encoder-decoder model with latent code z derived based on the embeddings of both source and target graphs and minimizes the conditional VAE loss function to learn the mapping F : ( X , z ) → Y F:(X,\textbf{z})\rightarrow Y . The authors then propose an adversarial variation called VJTNN + GAN to force generated graphs to follow the distribution of the target ones, which is trained using the WGAN framework [ 111 ] .

 
 
 Mol-CycleGAN [ 46 ] establishes structural similarity between the source and target molecular graphs by adopting a CycleGAN-based [ 120 ] approach. To this end, the method first computes the latent space embeddings for X X and Y Y using JT-VAE [ 34 ] and then learns the transformation F : X → Y F:X\rightarrow Y ( and its reverse, i.e., G : Y → X G:Y\rightarrow X ) in that space. Mol-CycleGAN also introduces the discriminator D X D_{X} (and D Y D_{Y} ) to decide whether a sample is from the distribution of X X (or Y Y ) or it is generated by G G (or F F ). The model parameters are trained by optimizing the following loss function:

 

 
 | 
 ℒ ⁡ ( F , G , D X , D Y ) = ℒ G ​ A ​ N ​ ( F , D Y , X , Y ) + ℒ G ​ A ​ N ​ ( G , D X , Y , X ) + λ 1 ​ ℒ c ​ y ​ c ​ ( F , G ) + λ 2 ​ ℒ i ​ d ​ e ​ n ​ t ​ i ​ t ​ y ​ ( F , G ) , \begin{split}\mathcal{L}(F,G,D_{X},D_{Y})= \mathcal{L}_{GAN}(F,D_{Y},X,Y)+\mathcal{L}_{GAN}(G,D_{X},Y,X)\\
 +\lambda_{1}\mathcal{L}_{cyc}(F,G)+\lambda_{2}\mathcal{L}_{identity}(F,G),\end{split} | 
 | 
 (44) | 
 

 where the authors utilize the adversarial loss of LS-GAN [ 123 ] and the similar ℒ c ​ y ​ c ​ ( F , G ) \mathcal{L}_{cyc}(F,G) and ℒ i ​ d ​ e ​ n ​ t ​ i ​ t ​ y ​ ( F , G ) \mathcal{L}_{identity}(F,G) as CycleGAN, where the former reduces the space of mapping functions and acts as a regularizer, while the latter makes the generated molecule not to be structurally far away from the original one. After the training finishes, Mol-CycleGAN takes a molecule X X as input and calculates its embedding by applying the encoder of the JT-VAE. Then, F ⁡ ( X ) F(X) computes an embedding corresponding to a molecule with desired properties that is also structurally similar to X X . Finally, the model generates the optimized molecular graph Y Y using the JT-VAE’s decoder.

 
 
 Misc-GAN [ 47 ] proposes another translation model inspired by CycleGAN [ 120 ] to learn a mapping function F F from a source graph G s G_{s} to its corresponding target graph G t G_{t} while preserving the hierarchical graph structures (i.e., the community structures) in the target graph in different levels of granularity. The model training consists of three stages: First, it constructs coarser graphs in L L granularity levels based on an input target graph G t G_{t} . Then, at each level l l , it trains an independent CycleGAN-based generative model from G s G_{s} to G t ( l ) G_{t}^{(l)} . Finally, all generated graphs G ~ t ( l ) \tilde{G}_{t}^{(l)} are aggregated together to form the reconstructed target graph G ~ t \tilde{G}_{t} . The framework is trained by minimizing the following loss function:

 

 
 | 
 ℒ = ℒ m ​ s + ℒ F + ℒ G + ℒ c ​ y ​ c , \mathcal{L}=\mathcal{L}_{ms}+\mathcal{L}_{F}+\mathcal{L}_{G}+\mathcal{L}_{cyc}, | 
 | 
 (45) | 
 

 where ℒ m ​ s \mathcal{L}_{ms} is the multi-scale reconstruction loss between the target graph G t G_{t} and the generated graph G ~ t \tilde{G}_{t} , ℒ F \mathcal{L}_{F} is the forward adversarial loss for learning a mapping from the source to the target graph, ℒ G \mathcal{L}_{G} is the backward adversarial loss to learn the reverse mapping, and ℒ c ​ y ​ c \mathcal{L}_{cyc} is the cycle consistency loss [ 120 ] .

 
 
 
 
 

## VII Flow-based Deep Graph Generators 

 
 
 
 Method | 
 Category | 
 
 
 
 Generation 
 
 Strategy 
 | 
 
 
 
 Attention 
 
 Mechanism 
 | 
 Features | 
 
 
 
 Conditional 
 
 Generation 
 | 

 
 GNF [ 37 ] | 
 Autoencoder-based | 
 All at Once | 
 Yes | 
 Node/Edge | 
 No | 

 
 GraphNVP [ 48 ] | 
 - | 
 All at Once | 
 No | 
 Node/Edge | 
 No | 

 
 GraphAF [ 12 ] | 
 Autoregressive | 
 Node-by-node | 
 No | 
 Node/Edge | 
 No | 

 
 GrAD [ 13 ] | 
 Autoregressive | 
 Block of nodes | 
 Yes | 
 - | 
 No | 

 TABLE VII: The Main Characteristics of Flow-Based Deep Graph Generators 
 
 
 In addition to the methods we have discussed so far, a line of research has recently emerged, which employs flow-based approaches in the field of graph generation. For example, GNF [ 37 ] develops a generative model of graphs by combining normalizing flows with a graph auto-encoder. More specifically, the authors first train a permutation invariant graph auto-encoder that encodes an input graph to a set of node features X ∈ ℝ n × k X\in\mathbb{R}^{n\times k} using a standard GNN. Then, a simple decoder outputs a probabilistic adjacency matrix A ^ \hat{A} , in which edge probability between two arbitrary nodes i i and j j with embedding vectors x i x_{i} and x j x_{j} is computed as follows:

 

 
 | 
 A ^ i ​ j = 1 1 + exp ⁡ ( C ⁡ ( ‖ x i − x j ‖ 2 2 − 1 ) ) , \hat{A}_{ij}=\frac{1}{1+\exp(C(||x_{i}-x_{j}||^{2}_{2}-1))}, | 
 | 
 (46) | 
 

 where C is a temperature hyperparameter. After the auto-encoder training completes, the encoder is employed to compute node features X to be used as training input for the GNF. Then, the GNF, which is based on non-volume preserving flows [ 124 ] , learns a mapping from the complicated graph distribution into a latent distribution that is well modelled as a Gaussian. At inference time, GNF generates node features by first sampling Z ∼ 𝒩 ⁡ ( 0 , I ) Z\sim\mathcal{N}(0,I) from the latent space followed by applying the inverse mapping, X = f − 1 ​ ( Z ) X=f^{-1}(Z) which is then fed into the decoder to get the predicted adjacency matrix as illustrated in Figure 13 . GraphNVP [ 48 ] takes a similar approach, but rather than pretraining an auto-encoder to get continuous node features, the authors propose to perform Dequantization [ 124 ] , [ 125 ] by adding uniform noise to the discrete adjacency tensor as well as the node label matrix. More precisely, GraphNVP proposes a two-step generation scheme by learning two latent representations for each graph, one for the adjacency tensor and the other for node labels.

 
 
 Fig. 13: The framework of GNF [ 37 ] for the graph generation (reprinted with permission). 
 
 
 In addition to the flow-based methods we have studied so far that generate the whole graph in one step, there are other approaches adopting autoregressive generation strategies. For example, GraphAF [ 12 ] proposes to generate molecular graphs by combining the advantages of both autoregressive and flow-based models. The method first converts a molecular graph structure G = ( A , C ) G=(A,C) , where both node type matrix C C and the adjacency matrix A A are discrete, into continuous data G ′ = ( A ′ , C ′ ) G^{\prime}=(A^{\prime},C^{\prime}) using Dequantization technique [ 124 ] , [ 125 ] in order to make the data usable for a flow-based model. Then, conditional distributions for the i i -th generation step are defined according to Autoregressive Flows (AF) [ 126 ] as follows:

 

 
 | 
 p ⁡ ( C i ′ | G i ) = 𝒩 ⁡ ( μ i C , ( α i C ) 2 ) p ( A ′ i ​ j | G i , C i , A i , 1 : j − 1 ) = 𝒩 ( μ A i ​ j , ( α i ​ j A ) 2 ) , \begin{split} p(C^{\prime}_{i}|G_{i})=\mathcal{N}(\mu^{C}_{i},(\alpha_{i}^{C})^{2})\\
 p(A^{\prime}_{ij}|G_{i},C_{i},A_{i,1:j-1})=\mathcal{N}(\mu^{A}_{ij},(\alpha_{ij}^{A})^{2}),\end{split} | 
 | 
 (47) | 
 

 where G i G_{i} is the current sub-graph, μ i C \mu^{C}_{i} , α i C \alpha_{i}^{C} , and μ i ​ j A \mu^{A}_{ij} , α i ​ j A \alpha_{ij}^{A} are the means and standard deviations of Gaussian distributions, which are computed by different neural networks based on node embeddings of the sub-graph generated so far. To calculate the exact likelihood, an invertible mapping from the molecule structures G ′ = ( A ′ , C ′ ) G^{\prime}=(A^{\prime},C^{\prime}) to latent Gaussian space z is defined as:

 

 
 | 
 z i = ( C i ′ − μ i C ) ⊙ 1 α i C , z i ​ j = ( A i ​ j ′ − μ i ​ j A ) ⊙ 1 α i ​ j A , z_{i}=(C^{\prime}_{i}-\mu^{C}_{i})\odot\frac{1}{\alpha_{i}^{C}},\ z_{ij}=(A^{\prime}_{ij}-\mu^{A}_{ij})\odot\frac{1}{\alpha_{ij}^{A}}, | 
 | 
 (48) | 
 

 where 1 α i C \frac{1}{\alpha_{i}^{C}} and 1 α i ​ j A \frac{1}{\alpha_{ij}^{A}} denote element-wise reciprocals of α i C \alpha_{i}^{C} and α i ​ j A \alpha_{ij}^{A} , respectively and ⊙ \odot is the element-wise multiplication. At inference time, GraphAF just samples random variables z i z_{i} and z i ​ j z_{ij} from the latent Gaussian space and converts them to the molecule structures as in Eq. ( 49 ) to generate new graphs in an autoregressive manner:

 

 
 | 
 C i ′ = z i ⊙ α i C + μ i C , A i ​ j ′ = z i ​ j ⊙ α i ​ j A + μ i ​ j A . C^{\prime}_{i}=z_{i}\odot\alpha_{i}^{C}+\mu^{C}_{i}\ ,\ A^{\prime}_{ij}=z_{ij}\odot\alpha_{ij}^{A}+\mu^{A}_{ij}. | 
 | 
 (49) | 
 

 GraphAF further proposes a valency-based rejection sampling similar to MolecularRNN [ 3 ] to guarantee the validity of generated molecules. The authors also propose to fine-tune the generation process with reinforcement learning to generate molecules with optimized properties.

 
 
 More recently, GrAD [ 13 ] proposes another autoregressive flow-based approach for graph generation, which can also be considered as a variant of GRAN [ 7 ] . In particular, its training consists of two stages. Firstly, for generating each new block of B B nodes in the t t -th step, the model samples latent codes H b t ∈ ℝ B × k H_{b_{t}}\in\mathbb{R}^{B\times k} to initialize the corresponding node representations. Then, different from GRAN’s formulation in Eq. ( 18 ), the node features get updated using graph attention layers almost according to [ 83 ] , except that self-attention weights are calculated only based on each node’s neighborhood to inject structural information of the currently generated graph into the updating process. After node representations are obtained, the model follows a similar generation strategy as GRAN and jointly optimizes the generator’s parameters and the distribution of latent codes. In the second training stage, GrAD trains a flow-based reversible model to map samples from the optimized distribution of latent codes to a simple Gaussian distribution. Therefore, at the inference time, one can first sample a batch of B B vectors Z ∈ ℝ B × k Z\in\mathbb{R}^{B\times k} from a Gaussian base distribution and apply the inverse mapping H b t = f − 1 ​ ( Z ) H_{b_{t}}=f^{-1}(Z) to obtain initial node representations and then go through the generation process.

 
 
 Table VII summarizes the main characteristics of the flow-based DGGs.

 
 
 

## VIII Applications 

 
 Deep graph generation approaches have a wide range of applications, from discovering new molecular structures and building knowledge graphs to modeling physical, social, and biological networks. Here we review some of the most explored applications and suggest potential future directions.

 
 

### VIII-A Molecular Graph Generation 

 
 The molecule generation approaches aim to downsize the high-dimensional chemical space to expedite drug design and material discovery. Since molecules can be considered as graphs, where the atoms form the graph’s node-set and the chemical bonds determine how those nodes connect, the most widely explored application of modern deep graph generative methods is generating molecular structures, a problem that has been previously mostly formulated as producing SMILES strings [ 127 , 100 , 128 , 129 , 130 ] . Using graph generators instead of SMILES based models results in generating more valid intermediate substructures compared to mostly meaningless partially generated substrings. It also allows better capturing the similarities between molecules as molecules with similar structures can have totally different SMILES encodings.

 
 
 Deep molecular graph generation approaches proposed so far utilize various frameworks, from VAEs [ 29 , 28 , 34 , 30 , 16 , 36 , 15 , 35 ] and GANs [ 40 , 46 , 14 ] to RL-based [ 20 , 21 , 25 ] , autoregressive [ 12 , 8 , 1 , 3 , 10 , 19 ] and flow-based frameworks [ 37 , 48 , 12 ] . They also adopt different generation strategies at varying granularity levels. For example, some of them generate the whole graph all at once [ 28 , 40 ] , while others add one atom at a time [ 1 , 3 , 10 ] or use valid chemical substructures as their building blocks [ 34 , 14 , 19 ] .

 
 
 Moreover, in molecular graph generation, two challenges must be taken into account. First, the generated molecules must satisfy the explicitly specified validity constraints, i.e., an atom’s chemical bonds should not exceed its valence. The graph generator models proposed so far address the issue by employing different mechanisms, including introducing structural penalties during the training [ 3 , 20 ] , adopting valency-based rejection sampling at the inference time [ 12 , 3 ] , utilizing a grammar-based approach [ 35 ] , adding regularization terms to the objective function [ 30 ] , using valid chemical substructures as building blocks [ 15 , 34 , 19 , 14 ] , and employing valency masking mechanism [ 16 , 36 ] . The second challenge to be considered is that new molecular structures should obey some desired properties. This problem has also been addressed by taking various strategies including minimizing a distance [ 16 , 15 ] or utilizing Bayesian optimization [ 35 , 34 , 36 ] in some continuous latent space, maximizing a domain-specific reward in RL-based approaches [ 20 , 21 , 40 ] , or performing the generation given an input molecule with desired properties and try to preserve those properties in the target molecule [ 46 , 14 ] .

 
 
 

### VIII-B Non-Molecular Graph Generation 

 
 Although the most remarkable application of modern graph generation approaches explored so far is generating molecular structures, several other non-application-specific approaches have been proposed [ 2 , 18 , 5 , 7 , 32 , 13 , 38 ] working on more general datasets. While these methods’ ultimate goal is to be used on real-world applications such as generating social network graphs, most of them suffer from scalability issues. Thus, they are currently being applied to synthetic or relatively small real datasets. Despite the steps taken towards making these models more scalable [ 2 , 7 , 11 ] , it should be specifically considered as future work.

 
 
 

### VIII-C Future Applications 

 
 Beyond the discussed applications, some other problems can potentially be solved from a graph generation perspective. This is particularly the case when the output space includes graphs, and so the generated output, on the one hand, must depend on the input and, on the other hand, must obey the distribution of the output space graphs. Below, two practical examples of these problems are mentioned.

 
 

#### VIII-C 1 Language-Based Graph Generation

 
 In natural language processing, several approaches have been proposed to extract rich graph-structured information from textual data. They include methods aiming to extract AMRs (Abstract Meaning Representations) [ 131 , 132 , 133 ] , semantic graphs [ 134 ] , semantic dependency graphs [ 135 , 136 , 137 ] , and even those that transform one graph into another based on some input sentences [ 138 ] . However, most existing methods often propose some domain-specific procedures to produce these graph-structured knowledge representations, which, as a result, are not capable enough to consider various effective factors. Therefore, as a future orientation, the community can solve such problems with a conditional generative approach, making it possible to generate graphs from their corresponding distribution given the specified textual input.

 
 
 

#### VIII-C 2 Scene Graph Generation

 
 Scene graphs are structured representations of images providing higher-level knowledge for scene understanding, where the objects in each image form the node-set of the corresponding scene graph, and the relationships between objects determine how the nodes connect. As this class of graphs has a wide range of applications, including image captioning, visual question answering, image retrieval, and image generation [ 139 ] , scene graph generation becomes a line of research in recent years.

 
 
 Most scene graph generation methods first detect objects from an input image using object detection models like Faster R-CNN [ 140 ] to form the set of graph nodes. Meanwhile, the relationships can be extracted either jointly with the objects [ 141 ] or after all the objects are detected [ 142 , 143 , 144 , 145 ] . Among these approaches, some generate scene graphs solely based on input images [ 146 , 142 , 143 ] , while others benefit from additional text input [ 144 , 147 ] , or some self-generated extra information [ 139 , 145 ] . Moreover, as the number of objects increases, some models [ 148 , 149 , 139 ] propose to first generate multiple subgraphs and then aggregate them to construct the complete scene graph to address the scalability issue.

 
 
 Here, we have briefly reviewed and categorized some of the scene graph generation methods. However, they are more of a relational information extractor from images rather than graph generators, which estimate the underlying data distribution. Therefore, it can be explored in the future.

 
 
 
 
 

## IX Implementations 

 
 In this section, we discuss the implementation details by categorizing and summarizing commonly used datasets and evaluation metrics. We also collect the available source codes in Appendix A.

 
 

### IX-A Datasets 

 
 There are many datasets for learning on graphs that have been investigated by previous studies, including [ 72 ] and [ 150 ] . However, none of these studies has thoroughly and exclusively examined and categorized the datasets used in graph generation approaches. Here, we have summarized the most prominent ones in three general categories according to the main graph generation applications, as shown in Table VIII .
 

 
 
 
 
 Category | 
 
 
 Dataset 
 | 
 
 
 Citation 
 | 

 
 
 
 
 Chemical 
 
 
 
 Bioinformatics 
 | 
 
 
 ZINC 
 | 
 
 
 [ 25 , 3 , 6 , 8 , 12 , 16 , 20 , 21 , 17 , 23 , 26 , 34 , 30 , 28 , 35 , 36 , 39 , 46 , 48 ] 
 | 

 
 
 
 QM9 
 | 
 
 
 [ 12 , 28 , 29 , 30 , 16 , 36 , 37 , 40 , 48 ] 
 | 

 
 
 
 CEPDB 
 | 
 
 
 [ 16 ] 
 | 

 
 
 
 Polymer 
 | 
 
 
 [ 14 ] 
 | 

 
 
 
 USPTO 
 | 
 
 
 [ 15 ] 
 | 

 
 
 
 GuacaMol 
 | 
 
 
 [ 26 , 25 ] 
 | 

 
 
 
 ChEMBL 
 | 
 
 
 [ 1 , 3 , 10 , 46 ] 
 | 

 
 
 
 Protein 
 | 
 
 
 [ 2 , 5 , 7 , 8 , 11 , 13 , 32 ] 
 | 

 
 
 
 MOSES 
 | 
 
 
 [ 3 , 12 ] 
 | 

 
 
 
 NCI-H23 
 | 
 
 
 [ 6 ] 
 | 

 
 
 
 Yeast 
 | 
 
 
 [ 6 ] 
 | 

 
 
 
 MOLT-4 
 | 
 
 
 [ 6 ] 
 | 

 
 
 
 MCF-7 
 | 
 
 
 [ 6 ] 
 | 

 
 
 
 Enzymes 
 | 
 
 
 [ 5 , 6 ] 
 | 

 
 
 
 PPI 
 | 
 
 
 [ 37 ] 
 | 

 
 Social | 
 
 
 Cora 
 | 
 
 
 [ 6 , 27 , 31 , 37 , 42 , 43 , 44 , 24 ] 
 | 

 
 
 
 Citeseer 
 | 
 
 
 [ 6 , 27 , 31 , 42 , 43 , 24 ] 
 | 

 
 
 
 Pubmed 
 | 
 
 
 [ 27 , 31 , 37 , 42 , 24 ] 
 | 

 
 
 
 DBLP 
 | 
 
 
 [ 42 , 38 ] 
 | 

 
 Synthetic | 
 
 
 Barabasi-Albert 
 | 
 
 
 [ 2 , 10 , 8 , 31 , 45 , 24 , 33 ] 
 | 

 
 
 
 Erdos-Renyi 
 | 
 
 
 [ 31 , 32 , 24 , 33 ] 
 | 

 
 
 
 Watts-Strogatz 
 | 
 
 
 [ 32 ] 
 | 

 
 
 
 Community 
 | 
 
 
 [ 2 , 18 , 5 , 8 , 12 , 13 , 37 ] 
 | 

 
 
 
 Grid 
 | 
 
 
 [ 2 , 7 , 8 , 11 , 13 ] 
 | 

 
 
 
 Lobster 
 | 
 
 
 [ 8 , 11 , 13 ] 
 | 

 
 
 
 Cycles 
 | 
 
 
 [ 10 , 13 ] 
 | 

 
 
 
 Ego 
 | 
 
 
 [ 2 , 18 , 5 , 8 , 9 , 12 , 13 , 31 , 37 , 33 ] 
 | 

 TABLE VIII: Summary of the Commonly Used Datasets 
 
 
 

### IX-B Evaluation Metrics 

 
 Depending on the application, graph generation approaches use different evaluation metrics. Specifically, the molecular graph generators adopt two different sets of metrics where the first set contains those evaluating the overall quality of generated samples, including validity , uniqueness , novelty , reconstruction , internal diversity , negative log-likelihood (NLL) , and some structural statistics like nearest neighbor similarity (SNN) or fragment/scaffold similarity . The second set, on the other hand, contains metrics assessing special chemical properties of the molecules, namely, synthetic accessibility score (SA score) [ 151 ] , drug-likeness score (QED) [ 152 ] , molecular weight (MW) , log partition coefficient (logP) , penalized logP , and topological polar surface area (TPSA) . For non-molecular graph generation, on the other side, a considerable number of approaches employ distribution-related metrics such as Kullback-Leibler Divergence (KLD) or Maximum Mean Discrepancy (MMD) on several graph statistics like degrees, clustering coefficients, orbit counts, and the spectra of the graphs from the eigenvalues of the normalized graph Laplacian. NLL , validity , novelty , and uniqueness are also among other metrics adopted to evaluate non-molecular approaches.

 
 
 
 

## X Future Directions 

 
 Although several deep graph generation models have been proposed in the past few years, due to the emergence of this field and its short history, a number of challenges remain, suggesting future directions for research, as follows.

 
 

### X-A Scalability 

 
 Most of the proposed graph generation methods are only applicable to small graphs with a maximum of a few tens of nodes. Therefore, designing molecular graphs, which are mostly small in size, is the most prominent application of DGGs so far. Although some initial steps [ 2 , 7 , 11 ] have been taken towards scalability of generator models, much more effort is necessary to make them applicable in a wider range of real-world applications, such as social network modeling.

 
 
 

### X-B node ordering 

 
 Each graph with n n nodes can be represented under n ! n! different node orderings. Therefore, it becomes intractable for likelihood-based DGGs to calculate the exact likelihood as the graph size increases. To address this issue, some approximate approaches such as using approximate graph matching algorithms [ 28 ] or maximizing a lower bound of the likelihood by only considering subsets of node orderings (i.e., fixed [ 10 , 11 ] , uniform random [ 10 ] , BFS [ 2 ] , or a family of canonical orderings [ 7 ] ), have been proposed. However, it is still necessary to provide more effective solutions for the node ordering problem, as a result of which, the generators can generate larger samples with higher quality.

 
 
 

### X-C Interpretability 

 
 As mentioned earlier, DGGs are utilized in critical applications such as designing drug molecules, which directly affects public health. Therefore, the more transparent the generation procedure, the better control is exercised on the desirability of generated samples, which prevents additional trials and errors by limiting the number of candidate solutions. Hence, it is of great importance to make graph generation methods more interpretable. As of now, deep generative methods in areas such as image [ 153 , 154 , 155 ] and text [ 156 , 157 , 158 ] have slowly moved towards being more interpretable. However, only a few attempts [ 96 , 32 , 33 ] have recently been made in graph generation, making model interpretability a notable future research prospect.

 
 
 

### X-D Dynamic Graphs 

 
 While existing DGGs focus on generating static graphs, most of the graphs are inherently dynamic, meaning that they change over time by earning/losing nodes or connections, or even their attributes may alter. For example, in a social network, some users may join/leave the network, or the relationships between existing users may change over time. Therefore, generating dynamic graphs would play a key role in predicting how networks evolve. However, dynamicity is almost not addressed in the current generative approaches, making it a potentially challenging problem to explore in the further.

 
 
 

### X-E Conditional Graph Generation 

 
 When generating new graphs, in most cases, one aims to discover structures with desired characteristics. While conditional generation is relatively well investigated in image [ 119 , 159 , 66 , 160 ] and text [ 161 , 162 , 163 ] domains, it is comparably less explored in the field of graph generation. For example, in molecular graph generation, the generated molecules must satisfy some validity constraints or hold desired chemical properties. However, only a limited number of proposed methods adopt a conditional approach by whether incorporating conditional codes into the generation process [ 19 , 28 ] , enforcing the existence of favourable substructures in the output graph [ 19 ] or performing the generation conditioned on an input molecule to ensure the structural similarity [ 14 ] . Meanwhile, the majority of methods do not formulate the problem as conditional generation and address the issue by employing other techniques like property optimization in some latent continuous space [ 16 , 15 , 35 , 34 , 36 ] or injecting validity constraints, whether at the training [ 3 , 30 ] or the inference time [ 12 , 3 ] . Nevertheless, this issue has been even less studied in the non-molecular graph generation models, and only a few of them have partially addressed the problem [ 38 , 44 ] . Therefore, focusing more on conditional graph generation problems, especially those that have not yet been explored, such as class conditioned generation, is an important future research direction.

 
 
 
 

## XI Conclusion 

 
 In this article, we surveyed the emerging field of deep learning-based graph generation. For this purpose, we classified the existing methods into five general categories. We then provided a detailed and comparative review of the approaches in each category. We summarized the implementation details, including datasets, evaluation metrics, and available source codes, and discussed the current applications and possible future trends. Finally, we suggested future research directions according to the current challenges. We believe this article provides the readers a comprehensive insight to the field of graph generation research.

 
 
 

## Appendix A Acronym

 
 Table IX summarizes the acronyms and nomenclature used in this survey.

 
 
 
 
 
 
 Acronym 
 | 
 
 
 Model Name 
 | 
 Reference | 

 
 
 
 DGG 
 | 
 
 
 Deep Graph Generator 
 | 
 | 

 
 
 
 RNN 
 | 
 
 
 Recurrent Neural Network 
 | 
 | 

 
 
 
 LSTM 
 | 
 
 
 Long Short-Term Memory 
 | 
 [ 80 ] | 

 
 
 
 GRU 
 | 
 
 
 Gated Recurrent Unit 
 | 
 [ 81 ] | 

 
 
 
 VAE 
 | 
 
 
 Variational Autoencoder 
 | 
 [ 91 ] | 

 
 
 
 GAN 
 | 
 
 
 Generative Adversarial Network 
 | 
 [ 41 ] | 

 
 
 
 GNN 
 | 
 
 
 Graph Neural Network 
 | 
 [ 92 ] | 

 
 
 
 GCN 
 | 
 
 
 Graph Convolutional Network 
 | 
 [ 49 ] | 

 
 
 
 REIN 
 | 
 
 
 Recurrent Edge Inference Network 
 | 
 [ 85 ] | 

 
 
 
 DeepNC 
 | 
 
 
 Deep Generative Network Completion 
 | 
 [ 86 ] | 

 
 
 
 GRAN 
 | 
 
 
 Graph Recurrent Attention Network 
 | 
 [ 7 ] | 

 
 
 
 GRAM 
 | 
 
 
 Graph Generative Model with Graph Attention Mechanism 
 | 
 [ 8 ] | 

 
 
 
 AGE 
 | 
 
 
 Attention-Based Graph Evolution 
 | 
 [ 9 ] | 

 
 
 
 DeepGMG 
 | 
 
 
 Deep Generative Models of Graphs 
 | 
 [ 10 ] | 

 
 
 
 DeepGG 
 | 
 
 
 Deep Graph Generators 
 | 
 [ 88 ] | 

 
 
 
 BiGG 
 | 
 
 
 BIg Graph Generation 
 | 
 [ 11 ] | 

 
 
 
 VGAE 
 | 
 
 
 Variational Graph Autoencoder 
 | 
 [ 27 ] | 

 
 
 
 MPGVAE 
 | 
 
 
 Message Passing Graph VAE 
 | 
 [ 29 ] | 

 
 
 
 NED-VAE 
 | 
 
 
 Node-Edge Disentangled VAE 
 | 
 [ 32 ] | 

 
 
 
 DGVAE 
 | 
 
 
 Dirichlet Graph VAE 
 | 
 [ 33 ] | 

 
 
 
 JT-VAE 
 | 
 
 
 Junction Tree VAE 
 | 
 [ 34 ] | 

 
 
 
 HierVAE 
 | 
 
 
 Hierarchical VAE 
 | 
 [ 14 ] | 

 
 
 
 MHG-VAE 
 | 
 
 
 Molecular Hypergraph Grammar VAE 
 | 
 [ 35 ] | 

 
 
 
 CGVAE 
 | 
 
 
 Constrained Graph VAE 
 | 
 [ 16 ] | 

 
 
 
 DEFactor 
 | 
 
 
 Differentiable Edge
Factorization-based Probabilistic Graph
Generation 
 | 
 [ 17 ] | 

 
 
 
 GraphVRNN 
 | 
 
 
 Graph Variational RNN 
 | 
 [ 18 ] | 

 
 
 
 GCPN 
 | 
 
 
 Graph Convolutional Policy Network 
 | 
 [ 20 ] | 

 
 
 
 MNCE-RL 
 | 
 
 
 Molecular Neighborhood-Controlled Embedding RL 
 | 
 [ 25 ] | 

 
 
 
 GEGL 
 | 
 
 
 Genetic Expert-Guided Learning 
 | 
 [ 26 ] | 

 
 
 
 MMGAN 
 | 
 
 
 Multi-MotifGAN 
 | 
 [ 43 ] | 

 
 
 
 MolGAN 
 | 
 
 
 Molecular GAN 
 | 
 [ 40 ] | 

 
 
 
 LGGAN 
 | 
 
 
 Labeled Graph GAN 
 | 
 [ 117 ] | 

 
 
 
 TSGG-GAN 
 | 
 
 
 Time Series Conditioned Graph Generation-GAN 
 | 
 [ 45 ] | 

 
 
 
 VJTNN 
 | 
 
 
 Variational Junction Tree Encoder-Decoder 
 | 
 [ 39 ] | 

 
 
 
 Misc-GAN 
 | 
 
 
 Multi-Scale GAN 
 | 
 [ 47 ] | 

 
 
 
 GNF 
 | 
 
 
 Graph Normalizing Flow 
 | 
 [ 37 ] | 

 
 
 
 GrAD 
 | 
 
 
 Graph Auto-Decoder 
 | 
 [ 13 ] | 

 TABLE IX: Acronyms with their extended names 
 
 
 

## Appendix B Source Codes

 
 Table X summarizes the set of publicly available source codes for deep learning-based graph generation approaches discussed in the survey.

 
 
 
 
 Category | 
 Method | 
 
 
 URL 
 | 
 Language/Framework | 
 O.A. | 

 
 Autoregressive | 
 MolMP [ 1 ] | 
 
 
 https://github.com/kevinid/molecule_generator 
 | 
 Python/MXNet | 
 Yes | 

 
 MolRNN [ 1 ] | 
 
 
 https://github.com/kevinid/molecule_generator 
 | 
 Python/MXNet | 
 Yes | 

 
 GraphRNN [ 2 ] | 
 
 
 https://github.com/JiaxuanYou/graph-generation 
 | 
 Python/PyTorch | 
 Yes | 

 
 DeepNC [ 86 ] | 
 
 
 https://github.com/congasix/DeepNC 
 | 
 Python/TensorFlow | 
 Yes | 

 
 Bacciu et al. [ 5 ] | 
 
 
 https://github.com/marcopodda/grapher 
 | 
 Python/PyTorch | 
 Yes | 

 
 GraphGen [ 6 ] | 
 
 
 https://github.com/idea-iitd/graphgen 
 | 
 Python/PyTorch | 
 Yes | 

 
 GRAN [ 7 ] | 
 
 
 https://github.com/lrjconan/GRAN 
 | 
 Python/PyTorch | 
 Yes | 

 
 DeepGMG [ 10 ] | 
 
 
 https://github.com/JiaxuanYou/graph-generation/blob/master/main_DeepGMG.py 
 | 
 Python/PyTorch | 
 No | 

 
 BiGG [ 11 ] | 
 
 
 https://github.com/google-research/google-research/tree/master/bigg 
 | 
 Python/PyTorch | 
 Yes | 

 
 Autoencoder-Based | 
 VGAE [ 27 ] | 
 
 
 https://github.com/tkipf/gae 
 | 
 Python/TensorFlow | 
 Yes | 

 
 GraphVAE [ 28 ] | 
 
 
 https://github.com/JiaxuanYou/graph-generation/tree/master/baselines/graphvae 
 | 
 Python/PyTorch | 
 No | 

 
 Graphite [ 31 ] | 
 
 
 https://github.com/ermongroup/graphite 
 | 
 Python/TensorFlow | 
 Yes | 

 
 NED-VAE [ 32 ] | 
 
 
 https://github.com/xguo7/NED-VAE 
 | 
 | 
 Yes | 

 
 DGVAE [ 33 ] | 
 
 
 https://github.com/xiyou3368/DGVAE 
 | 
 Python/TensorFlow | 
 Yes | 

 
 JT-VAE [ 34 ] | 
 
 
 https://github.com/wengong-jin/icml18-jtnn 
 | 
 Python/PyTorch | 
 Yes | 

 
 HierVAE [ 14 ] | 
 
 
 https://github.com/wengong-jin/hgraph2graph 
 | 
 Python/PyTorch | 
 Yes | 

 
 MHG-VAE [ 35 ] | 
 
 
 https://github.com/ibm-research-tokyo/graph_grammar 
 | 
 Python/PyTorch | 
 Yes | 

 
 MoleculeChef [ 15 ] | 
 
 
 https://github.com/john-bradshaw/molecule-chef 
 | 
 Python/PyTorch | 
 Yes | 

 
 CGVAE [ 16 ] | 
 
 
 https://github.com/microsoft/constrained-graph-variational-autoencoder 
 | 
 Python/TensorFlow | 
 Yes | 

 
 NeVAE [ 36 ] | 
 
 
 https://github.com/Networks-Learning/nevae 
 | 
 Python/TensorFlow | 
 Yes | 

 
 Lim et al. [ 19 ] | 
 
 
 https://github.com/jaechanglim/GGM 
 | 
 | 
 Yes | 

 
 RL-Based | 
 GCPN [ 20 ] | 
 
 
 https://github.com/bowenliu16/rl_graph_generation 
 | 
 Python/TensorFlow | 
 Yes | 

 
 Karimi et al. [ 22 ] | 
 
 
 https://github.com/Shen-Lab/Drug-Combo-Generator 
 | 
 Python/TensorFlow | 
 Yes | 

 
 DeepGraphMolGen [ 23 ] | 
 
 
 https://github.com/dbkgroup/prop_gen 
 | 
 Python/PyTorch | 
 Yes | 

 
 MNCE-RL [ 25 ] | 
 
 
 https://github.com/Zoesgithub/MNCE-RL 
 | 
 Python/PyTorch | 
 Yes | 

 
 GEGL [ 26 ] | 
 
 
 https://github.com/sungsoo-ahn/genetic-expert-guided-learning 
 | 
 Python/PyTorch | 
 Yes | 

 
 Adversarial | 
 NetGAN [ 42 ] | 
 
 
 https://github.com/danielzuegner/netgan 
 | 
 Python/TensorFlow | 
 Yes | 

 
 MolGAN [ 40 ] | 
 
 
 https://github.com/nicola-decao/MolGAN 
 | 
 Python/TensorFlow | 
 Yes | 

 
 CONDGEN [ 38 ] | 
 
 
 https://github.com/KelestZ/CondGen 
 | 
 Python/PyTorch | 
 Yes | 

 
 VJTNN + GAN [ 39 ] | 
 
 
 https://github.com/wengong-jin/iclr19-graph2graph 
 | 
 Python/PyTorch | 
 Yes | 

 
 Mol-CycleGAN [ 46 ] | 
 
 
 https://github.com/ardigen/mol-cycle-gan 
 | 
 Python/Keras | 
 Yes | 

 
 Misc-GAN [ 47 ] | 
 
 
 https://github.com/Leo02016/Miscgan 
 | 
 Python/TensorFlow, Matlab | 
 Yes | 

 
 Flow-Based | 
 GNF [ 37 ] | 
 
 
 https://github.com/jliu/graph-normalizing-flows 
 | 
 Python/TensorFlow | 
 Yes | 

 
 GraphNVP [ 48 ] | 
 
 
 https://github.com/Kaushalya/graph-nvp 
 | 
 Python/Chainer-Chemistry | 
 Yes | 

 
 GraphAF [ 12 ] | 
 
 
 https://github.com/DeepGraphLearning/GraphAF 
 | 
 Python/PyTorch | 
 Yes | 

 TABLE X: A set of publicly available source codes. O.A. = Original Authors 
 
 \justify 
 

## References

 
 
 [1] 
 
Yibo Li, Liangren Zhang, and Zhenming Liu.

 
 Multi-objective de novo drug design with conditional graph generative
model.

 
 Journal of cheminformatics , 10(1):33, 2018.

 

 
 [2] 
 
Jiaxuan You, Rex Ying, Xiang Ren, William Hamilton, and Jure Leskovec.

 
 Graphrnn: Generating realistic graphs with deep auto-regressive
models.

 
 In International Conference on Machine Learning , pages
5708–5717, 2018.

 

 
 [3] 
 
Mariya Popova, Mykhailo Shvets, Junier Oliva, and Olexandr Isayev.

 
 Molecularrnn: Generating realistic molecular graphs with optimized
properties.

 
 arXiv preprint arXiv:1905.13372 , 2019.

 

 
 [4] 
 
Davide Bacciu, Alessio Micheli, and Marco Podda.

 
 Graph generation by sequential edge prediction.

 
 In Proc. ESANN , 2019.

 

 
 [5] 
 
Davide Bacciu, Alessio Micheli, and Marco Podda.

 
 Edge-based sequential graph generation with recurrent neural
networks.

 
 Neurocomputing , 2020.

 

 
 [6] 
 
Nikhil Goyal, Harsh Vardhan Jain, and Sayan Ranu.

 
 Graphgen: A scalable approach to domain-agnostic labeled graph
generation.

 
 In Proceedings of The Web Conference 2020 , pages 1253–1263,
2020.

 

 
 [7] 
 
Renjie Liao, Yujia Li, Yang Song, Shenlong Wang, Will Hamilton, David K
Duvenaud, Raquel Urtasun, and Richard Zemel.

 
 Efficient graph generation with graph recurrent attention networks.

 
 In Advances in Neural Information Processing Systems , pages
4257–4267, 2019.

 

 
 [8] 
 
Wataru Kawai, Yusuke Mukuta, and Tatsuya Harada.

 
 Scalable generative models for graphs with graph attention mechanism.

 
 arXiv preprint arXiv:1906.01861 , 2019.

 

 
 [9] 
 
Shuangfei Fan and Bert Huang.

 
 Attention-based graph evolution.

 
 In Pacific-Asia Conference on Knowledge Discovery and Data
Mining , pages 436–447. Springer, 2020.

 

 
 [10] 
 
Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, and Peter Battaglia.

 
 Learning deep generative models of graphs.

 
 arXiv preprint arXiv:1803.03324 , 2018.

 

 
 [11] 
 
Hanjun Dai, Azade Nazi, Yujia Li, Bo Dai, and Dale Schuurmans.

 
 Scalable deep generative modeling for sparse graphs.

 
 In International Conference on Machine Learning , 2020.

 

 
 [12] 
 
Chence Shi, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang.

 
 Graphaf: a flow-based autoregressive model for molecular graph
generation.

 
 In International Conference on Learning Representations , 2020.

 

 
 [13] 
 
Sohil Atul Shah and Vladlen Koltun.

 
 Auto-decoding graphs.

 
 arXiv preprint arXiv:2006.02879 , 2020.

 

 
 [14] 
 
Wengong Jin, Regina Barzilay, and Tommi Jaakkola.

 
 Hierarchical generation of molecular graphs using structural motifs.

 
 In International Conference on Machine Learning , 2020.

 

 
 [15] 
 
John Bradshaw, Brooks Paige, Matt J Kusner, Marwin Segler, and José Miguel
Hernández-Lobato.

 
 A model to search for synthesizable molecules.

 
 In Advances in Neural Information Processing Systems , pages
7935–7947, 2019.

 

 
 [16] 
 
Qi Liu, Miltiadis Allamanis, Marc Brockschmidt, and Alexander Gaunt.

 
 Constrained graph variational autoencoders for molecule design.

 
 In Advances in neural information processing systems , pages
7795–7804, 2018.

 

 
 [17] 
 
Rim Assouel, Mohamed Ahmed, Marwin H Segler, Amir Saffari, and Yoshua Bengio.

 
 Defactor: Differentiable edge factorization-based probabilistic graph
generation.

 
 arXiv preprint arXiv:1811.09766 , 2018.

 

 
 [18] 
 
Shih-Yang Su, Hossein Hajimirsadeghi, and Greg Mori.

 
 Graph generation with variational recurrent neural network.

 
 In NeurIPS Workshop on Graph Representation Learning , 2019.

 

 
 [19] 
 
Jaechang Lim, Sang-Yeon Hwang, Seokhyun Moon, Seungsu Kim, and Woo Youn Kim.

 
 Scaffold-based molecular design with a graph generative model.

 
 Chemical Science , 11(4):1153–1164, 2020.

 

 
 [20] 
 
Jiaxuan You, Bowen Liu, Zhitao Ying, Vijay Pande, and Jure Leskovec.

 
 Graph convolutional policy network for goal-directed molecular graph
generation.

 
 In Advances in neural information processing systems , pages
6410–6421, 2018.

 

 
 [21] 
 
Fangzhou Shi, Shan You, and Chang Xu.

 
 Reinforced molecule generation with heterogeneous states.

 
 In 2019 IEEE International Conference on Data Mining (ICDM) ,
pages 548–557. IEEE, 2019.

 

 
 [22] 
 
Mostafa Karimi, Arman Hasanzadeh, and Yang Shen.

 
 Network-principled deep generative models for designing drug
combinations as graph sets.

 
 Bioinformatics , 36(Supplement_1):i445–i454, 2020.

 

 
 [23] 
 
Yash Khemchandani, Stephen O’Hagan, Soumitra Samanta, Neil Swainston,
Timothy J Roberts, Danushka Bollegala, and Douglas B Kell.

 
 Deepgraphmolgen, a multi-objective, computational strategy for
generating molecules with desirable properties: a graph convolution and
reinforcement learning approach.

 
 Journal of Cheminformatics , 12(1):1–17, 2020.

 

 
 [24] 
 
Rakshit Trivedi, Jiachen Yang, and Hongyuan Zha.

 
 Graphopt: Learning optimization models of graph formation.

 
 In International Conference on Machine Learning , 2020.

 

 
 [25] 
 
Chencheng Xu, Qiao Liu, Minlie Huang, and Tao Jiang.

 
 Reinforced molecular optimization with neighborhood-controlled
grammars.

 
 Advances in Neural Information Processing Systems , 33, 2020.

 

 
 [26] 
 
Sungsoo Ahn, Junsu Kim, Hankook Lee, and Jinwoo Shin.

 
 Guiding deep molecular optimization with genetic exploration.

 
 In Advances in neural information processing systems , 2020.

 

 
 [27] 
 
Thomas N Kipf and Max Welling.

 
 Variational graph auto-encoders.

 
 In NeurIPS Workshop on Bayesian Deep Learning , 2016.

 

 
 [28] 
 
Martin Simonovsky and Nikos Komodakis.

 
 Graphvae: Towards generation of small graphs using variational
autoencoders.

 
 In International Conference on Artificial Neural Networks ,
pages 412–422. Springer, 2018.

 

 
 [29] 
 
Daniel Flam-Shepherd, Tony Wu, and Alan Aspuru-Guzik.

 
 Graph deconvolutional generation.

 
 arXiv preprint arXiv:2002.07087 , 2020.

 

 
 [30] 
 
Tengfei Ma, Jie Chen, and Cao Xiao.

 
 Constrained generation of semantically valid graphs via regularizing
variational autoencoders.

 
 In Advances in Neural Information Processing Systems , pages
7113–7124, 2018.

 

 
 [31] 
 
Aditya Grover, Aaron Zweig, and Stefano Ermon.

 
 Graphite: Iterative generative modeling of graphs.

 
 In International Conference on Machine Learning , pages
2434–2444, 2019.

 

 
 [32] 
 
Xiaojie Guo, Liang Zhao, Zhao Qin, Lingfei Wu, Amarda Shehu, and Yanfang Ye.

 
 Node-edge co-disentangled representation learning for attributed
graph generation.

 
 In Proceedings of the 26th ACM SIGKDD International Conference
on Knowledge Discovery Data Mining , 2020.

 

 
 [33] 
 
Jia Li, Jianwei Yu, Jiajin Li, Honglei Zhang, Kangfei Zhao, Yu Rong, Hong
Cheng, and Junzhou Huang.

 
 Dirichlet graph variational autoencoder.

 
 Advances in Neural Information Processing Systems , 33, 2020.

 

 
 [34] 
 
Wengong Jin, Regina Barzilay, and Tommi Jaakkola.

 
 Junction tree variational autoencoder for molecular graph generation.

 
 In International Conference on Machine Learning , pages
2323–2332, 2018.

 

 
 [35] 
 
Hiroshi Kajino.

 
 Molecular hypergraph grammar with its application to molecular
optimization.

 
 In International Conference on Machine Learning , pages
3183–3191, 2019.

 

 
 [36] 
 
Bidisha Samanta, DE Abir, Gourhari Jana, Pratim Kumar Chattaraj, Niloy Ganguly,
and Manuel Gomez Rodriguez.

 
 Nevae: A deep generative model for molecular graphs.

 
 In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 33, pages 1110–1117, 2019.

 

 
 [37] 
 
Jenny Liu, Aviral Kumar, Jimmy Ba, Jamie Kiros, and Kevin Swersky.

 
 Graph normalizing flows.

 
 In Advances in Neural Information Processing Systems , pages
13578–13588, 2019.

 

 
 [38] 
 
Carl Yang, Peiye Zhuang, Wenhan Shi, Alan Luu, and Pan Li.

 
 Conditional structure generation through graph variational generative
adversarial nets.

 
 In Advances in Neural Information Processing Systems , pages
1340–1351, 2019.

 

 
 [39] 
 
Wengong Jin, Kevin Yang, Regina Barzilay, and Tommi Jaakkola.

 
 Learning multimodal graph-to-graph translation for molecule
optimization.

 
 In International Conference on Learning Representations , 2018.

 

 
 [40] 
 
Nicola De Cao and Thomas Kipf.

 
 Molgan: An implicit generative model for small molecular graphs.

 
 In ICML Workshop on Theoretical Foundations and Applications of
Deep Generative Models , 2018.

 

 
 [41] 
 
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozair, Aaron Courville, and Yoshua Bengio.

 
 Generative adversarial nets.

 
 In Advances in neural information processing systems , pages
2672–2680, 2014.

 

 
 [42] 
 
Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, and Stephan
Günnemann.

 
 Netgan: Generating graphs via random walks.

 
 In International Conference on Machine Learning , pages
609–618, 2018.

 

 
 [43] 
 
Anuththari Gamage, Eli Chien, Jianhao Peng, and Olgica Milenkovic.

 
 Multi-motifgan (mmgan): Motif-targeted graph generation and
prediction.

 
 In ICASSP 2020-2020 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP) , pages 4182–4186. IEEE, 2020.

 

 
 [44] 
 
Wesley Joon-Wie Tann, Ee-Chien Chang, and Bryan Hooi.

 
 Shadowcast: Controlling network properties to explain graph
generation.

 
 arXiv preprint arXiv:2006.03774 , 2020.

 

 
 [45] 
 
Shanchao Yang, Jing Liu, Kai Wu, and Mingming Li.

 
 Learn to generate time series conditioned graphs with generative
adversarial nets.

 
 arXiv preprint arXiv:2003.01436 , 2020.

 

 
 [46] 
 
Łukasz Maziarka, Agnieszka Pocha, Jan Kaczmarczyk, Krzysztof Rataj, Tomasz
Danel, and Michał Warchoł.

 
 Mol-cyclegan: a generative model for molecular optimization.

 
 Journal of Cheminformatics , 12(1):1–18, 2020.

 

 
 [47] 
 
Dawei Zhou, Lecheng Zheng, Jiejun Xu, and Jingrui He.

 
 Misc-gan: A multi-scale generative model for graphs.

 
 Frontiers in Big Data , 2:3, 2019.

 

 
 [48] 
 
Kaushalya Madhawa, Katushiko Ishiguro, Kosuke Nakago, and Motoki Abe.

 
 Graphnvp: An invertible flow model for generating molecular graphs.

 
 arXiv preprint arXiv:1905.11600 , 2019.

 

 
 [49] 
 
Thomas N Kipf and Max Welling.

 
 Semi-supervised classification with graph convolutional networks.

 
 In International Conference on Learning Representations , 2017.

 

 
 [50] 
 
Will Hamilton, Zhitao Ying, and Jure Leskovec.

 
 Inductive representation learning on large graphs.

 
 In Advances in neural information processing systems , pages
1024–1034, 2017.

 

 
 [51] 
 
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.

 
 How powerful are graph neural networks?

 
 In International Conference on Learning Representations , 2019.

 

 
 [52] 
 
Leonardo Cotta, Carlos HC Teixeira, Ananthram Swami, and Bruno Ribeiro.

 
 Unsupervised joint k k -node graph representations with
compositional energy-based models.

 
 In Advances in neural information processing systems , 2020.

 

 
 [53] 
 
Morteza Ramezani, Weilin Cong, Mehrdad Mahdavi, Anand Sivasubramaniam, and
Mahmut Kandemir.

 
 Gcn meets gpu: Decoupling “when to sample” from “how to
sample”.

 
 Advances in Neural Information Processing Systems , 33, 2020.

 

 
 [54] 
 
Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, and Pushmeet Kohli.

 
 Graph matching networks for learning the similarity of graph
structured objects.

 
 In International Conference on Machine Learning , pages
3835–3845, 2019.

 

 
 [55] 
 
Matthias Fey, Jan E Lenssen, Christopher Morris, Jonathan Masci, and Nils M
Kriege.

 
 Deep graph matching consensus.

 
 In International Conference on Learning Representations , 2020.

 

 
 [56] 
 
Daniel Zügner, Amir Akbarnejad, and Stephan Günnemann.

 
 Adversarial attacks on neural networks for graph data.

 
 In Proceedings of the 24th ACM SIGKDD International Conference
on Knowledge Discovery Data Mining , pages 2847–2856, 2018.

 

 
 [57] 
 
Daniel Zügner and Stephan Günnemann.

 
 Adversarial attacks on graph neural networks via meta learning.

 
 In International Conference on Learning Representations , 2019.

 

 
 [58] 
 
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero,
Pietro Lio, and Yoshua Bengio.

 
 Graph attention networks.

 
 In International Conference on Learning Representations , 2018.

 

 
 [59] 
 
Vineet Kosaraju, Amir Sadeghian, Roberto Martín-Martín, Ian Reid,
Hamid Rezatofighi, and Silvio Savarese.

 
 Social-bigat: Multimodal trajectory forecasting using bicycle-gan and
graph attention networks.

 
 In Advances in Neural Information Processing Systems , pages
137–146, 2019.

 

 
 [60] 
 
Paul Erdős and Alfréd Rényi.

 
 On the evolution of random graphs.

 
 Publ. Math. Inst. Hung. Acad. Sci , 5(1):17–60, 1960.

 

 
 [61] 
 
Duncan J Watts and Steven H Strogatz.

 
 Collective dynamics of ‘small-world’networks.

 
 nature , 393(6684):440–442, 1998.

 

 
 [62] 
 
Réka Albert and Albert-László Barabási.

 
 Statistical mechanics of complex networks.

 
 Reviews of modern physics , 74(1):47, 2002.

 

 
 [63] 
 
Paul W Holland, Kathryn Blackmond Laskey, and Samuel Leinhardt.

 
 Stochastic blockmodels: First steps.

 
 Social networks , 5(2):109–137, 1983.

 

 
 [64] 
 
Jure Leskovec, Deepayan Chakrabarti, Jon Kleinberg, Christos Faloutsos, and
Zoubin Ghahramani.

 
 Kronecker graphs: an approach to modeling networks.

 
 Journal of Machine Learning Research , 11(2), 2010.

 

 
 [65] 
 
Alec Radford, Luke Metz, and Soumith Chintala.

 
 Unsupervised representation learning with deep convolutional
generative adversarial networks.

 
 In International Conference on Learning Representations , 2016.

 

 
 [66] 
 
Xinchen Yan, Jimei Yang, Kihyuk Sohn, and Honglak Lee.

 
 Attribute2image: Conditional image generation from visual attributes.

 
 In European Conference on Computer Vision , pages 776–791.
Springer, 2016.

 

 
 [67] 
 
Yizhe Zhang, Zhe Gan, Kai Fan, Zhi Chen, Ricardo Henao, Dinghan Shen, and
Lawrence Carin.

 
 Adversarial feature matching for text generation.

 
 In International Conference on Machine Learning , 2017.

 

 
 [68] 
 
Prince Zizhuang Wang and William Yang Wang.

 
 Riemannian normalizing flow on variational wasserstein autoencoder
for text modeling.

 
 In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume
1 (Long and Short Papers) , pages 284–294. Association for Computational
Linguistics, 2019.

 

 
 [69] 
 
Takuhiro Kaneko, Hirokazu Kameoka, Kaoru Hiramatsu, and Kunio Kashino.

 
 Sequence-to-sequence voice conversion with similarity metric learned
using generative adversarial networks.

 
 In INTERSPEECH , volume 2017, pages 1283–1287, 2017.

 

 
 [70] 
 
Yang Gao, Rita Singh, and Bhiksha Raj.

 
 Voice impersonation using generative adversarial networks.

 
 In 2018 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP) , pages 2506–2510. IEEE, 2018.

 

 
 [71] 
 
Ziwei Zhang, Peng Cui, and Wenwu Zhu.

 
 Deep learning on graphs: A survey.

 
 IEEE Transactions on Knowledge and Data Engineering , 2020.

 

 
 [72] 
 
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu
Philip.

 
 A comprehensive survey on graph neural networks.

 
 IEEE Transactions on Neural Networks and Learning Systems ,
2020.

 

 
 [73] 
 
Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang,
Changcheng Li, and Maosong Sun.

 
 Graph neural networks: A review of methods and applications.

 
 arXiv preprint arXiv:1812.08434 , 2018.

 

 
 [74] 
 
Wenming Cao, Zhiyue Yan, Zhiquan He, and Zhihai He.

 
 A comprehensive survey on geometric deep learning.

 
 IEEE Access , 8:35929–35949, 2020.

 

 
 [75] 
 
Davide Bacciu, Federico Errica, Alessio Micheli, and Marco Podda.

 
 A gentle introduction to deep learning for graphs.

 
 Neural Networks , 2020.

 

 
 [76] 
 
John Boaz Lee, Ryan A Rossi, Sungchul Kim, Nesreen K Ahmed, and Eunyee Koh.

 
 Attention models in graphs: A survey.

 
 ACM Transactions on Knowledge Discovery from Data (TKDD) ,
13(6):1–25, 2019.

 

 
 [77] 
 
Lichao Sun, Yingtong Dou, Carl Yang, Ji Wang, Philip S Yu, and Bo Li.

 
 Adversarial attack and defense on graph data: A survey.

 
 arXiv preprint arXiv:1812.10528 , 2018.

 

 
 [78] 
 
Guixiang Ma, Nesreen K Ahmed, Theodore L Willke, and Philip S Yu.

 
 Deep graph similarity learning: A survey.

 
 arXiv preprint arXiv:1912.11615 , 2019.

 

 
 [79] 
 
Junchi Yan, Shuang Yang, and Edwin R Hancock.

 
 Learning for graph matching and related combinatorial optimization
problems.

 
 In International Joint Conference on Artificial Intelligence .
York, 2020.

 

 
 [80] 
 
Sepp Hochreiter and Jürgen Schmidhuber.

 
 Long short-term memory.

 
 Neural computation , 9(8):1735–1780, 1997.

 

 
 [81] 
 
Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau,
Fethi Bougares, Holger Schwenk, and Yoshua Bengio.

 
 Learning phrase representations using rnn encoder–decoder for
statistical machine translation.

 
 In Proceedings of the 2014 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 1724–1734, 2014.

 

 
 [82] 
 
Chia-Cheng Liu, Harris Chan, Kevin Luk, and AI Borealis.

 
 Auto-regressive graph generation modeling with improved evaluation
methods.

 
 In NeurIPS Workshop on Graph Representation Learning , 2019.

 

 
 [83] 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

 
 Attention is all you need.

 
 In Advances in neural information processing systems , pages
5998–6008, 2017.

 

 
 [84] 
 
Mingming Sun and Ping Li.

 
 Graph to graph: a topology aware approach for graph structures
learning and generation.

 
 In The 22nd International Conference on Artificial Intelligence
and Statistics , pages 2946–2955, 2019.

 

 
 [85] 
 
Rangel Daroya, Rowel Atienza, and Rhandley Cajote.

 
 Rein: Flexible mesh generation from point clouds.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition Workshops , pages 352–353, 2020.

 

 
 [86] 
 
Cong Tran, Won-Yong Shin, Andreas Spitz, and Michael Gertz.

 
 Deepnc: Deep generative network completion.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
2020.

 

 
 [87] 
 
Xifeng Yan and Jiawei Han.

 
 gspan: Graph-based substructure pattern mining.

 
 In 2002 IEEE International Conference on Data Mining, 2002.
Proceedings. , pages 721–724. IEEE, 2002.

 

 
 [88] 
 
Julian Stier and Michael Granitzer.

 
 Deep graph generators.

 
 arXiv preprint arXiv:2006.04159 , 2020.

 

 
 [89] 
 
Deepayan Chakrabarti, Yiping Zhan, and Christos Faloutsos.

 
 R-mat: A recursive model for graph mining.

 
 In Proceedings of the 2004 SIAM International Conference on Data
Mining , pages 442–446. SIAM, 2004.

 

 
 [90] 
 
Peter M Fenwick.

 
 A new data structure for cumulative frequency tables.

 
 Software: Practice and experience , 24(3):327–336, 1994.

 

 
 [91] 
 
Diederik P Kingma and Max Welling.

 
 Auto-encoding variational bayes.

 
 In International Conference on Learning Representations , 2014.

 

 
 [92] 
 
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele
Monfardini.

 
 The graph neural network model.

 
 IEEE Transactions on Neural Networks , 20(1):61–80, 2008.

 

 
 [93] 
 
Martin Simonovsky and Nikos Komodakis.

 
 Dynamic edge-conditioned filters in convolutional neural networks on
graphs.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 3693–3702, 2017.

 

 
 [94] 
 
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and
George E Dahl.

 
 Neural message passing for quantum chemistry.

 
 In Proceedings of the 34th International Conference on Machine
Learning-Volume 70 , pages 1263–1272, 2017.

 

 
 [95] 
 
Oriol Vinyals, Samy Bengio, and Manjunath Kudlur.

 
 Order matters: Sequence to sequence for sets.

 
 arXiv preprint arXiv:1511.06391 , 2015.

 

 
 [96] 
 
Niklas Stoehr, Marc Brockschmidt, Jan Stuehmer, and Emine Yilmaz.

 
 Disentangling interpretable generative parameters of random and
real-world graphs.

 
 In NeurIPS Workshop on Graph Representation Learning , 2019.

 

 
 [97] 
 
Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot,
Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner.

 
 beta-vae: Learning basic visual concepts with a constrained
variational framework.

 
 In International Conference on Learning Representations , 2017.

 

 
 [98] 
 
Philipp Hennig, David Stern, Ralf Herbrich, and Thore Graepel.

 
 Kernel topic models.

 
 In Artificial Intelligence and Statistics , pages 511–519,
2012.

 

 
 [99] 
 
Frank Drewes, H-J Kreowski, and Annegret Habel.

 
 Hyperedge replacement graph grammars.

 
 In Handbook Of Graph Grammars And Computing By Graph
Transformation: Volume 1: Foundations , pages 95–162. World Scientific,
1997.

 

 
 [100] 
 
MJ Kusner, B Paige, and JM Hernández-Lobato.

 
 Grammar variational autoencoder.

 
 In Proceedings of the 34 th International Conference on Machine
Learning, Sydney, Australia, PMLR 70, 2017 , volume 70, pages 1945–1954.
ACM, 2017.

 

 
 [101] 
 
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel.

 
 Gated graph sequence neural networks.

 
 In International Conference on Learning Representations , 2016.

 

 
 [102] 
 
Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar,
Christopher A Hunter, Costas Bekas, and Alpha A Lee.

 
 Molecular transformer: A model for uncertainty-calibrated chemical
reaction prediction.

 
 ACS central science , 5(9):1572–1583, 2019.

 

 
 [103] 
 
I Tolstikhin, O Bousquet, S Gelly, and B Schölkopf.

 
 Wasserstein auto-encoders.

 
 In International Conference on Learning Representations , 2018.

 

 
 [104] 
 
Kihyuk Sohn, Honglak Lee, and Xinchen Yan.

 
 Learning structured output representation using deep conditional
generative models.

 
 In Advances in neural information processing systems , pages
3483–3491, 2015.

 

 
 [105] 
 
Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, et al.

 
 Interaction networks for learning about objects, relations and
physics.

 
 In Advances in neural information processing systems , pages
4502–4510, 2016.

 

 
 [106] 
 
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

 
 Proximal policy optimization algorithms.

 
 arXiv preprint arXiv:1707.06347 , 2017.

 

 
 [107] 
 
Andrew Y Ng, Stuart J Russell, et al.

 
 Algorithms for inverse reinforcement learning.

 
 In Icml , volume 1, page 2, 2000.

 

 
 [108] 
 
Chelsea Finn, Sergey Levine, and Pieter Abbeel.

 
 Guided cost learning: Deep inverse optimal control via policy
optimization.

 
 In International conference on machine learning , pages 49–58,
2016.

 

 
 [109] 
 
Dirk Janssens and Grzegorz Rozenberg.

 
 Graph grammars with neighbourhood-controlled embedding.

 
 Theoretical Computer Science , 21(1):55–74, 1982.

 

 
 [110] 
 
Jan H Jensen.

 
 A graph-based genetic algorithm and generative model/monte carlo tree
search for the exploration of chemical space.

 
 Chemical science , 10(12):3567–3572, 2019.

 

 
 [111] 
 
Martin Arjovsky, Soumith Chintala, and Léon Bottou.

 
 Wasserstein generative adversarial networks.

 
 In Proceedings of the 34th International Conference on Machine
Learning-Volume 70 , pages 214–223, 2017.

 

 
 [112] 
 
Aditya Grover and Jure Leskovec.

 
 node2vec: Scalable feature learning for networks.

 
 In Proceedings of the 22nd ACM SIGKDD international conference
on Knowledge discovery and data mining , pages 855–864, 2016.

 

 
 [113] 
 
Liming Zhang.

 
 Stggan: Spatial-temporal graph generation.

 
 In Proceedings of the 27th ACM SIGSPATIAL International
Conference on Advances in Geographic Information Systems , pages 608–609,
2019.

 

 
 [114] 
 
Mehdi Mirza and Simon Osindero.

 
 Conditional generative adversarial nets.

 
 In NIPS Workshop on Deep Learning and Representation Learning ,
2014.

 

 
 [115] 
 
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan
Titov, and Max Welling.

 
 Modeling relational data with graph convolutional networks.

 
 In European Semantic Web Conference , pages 593–607. Springer,
2018.

 

 
 [116] 
 
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom
Erez, Yuval Tassa, David Silver, and Daan Wierstra.

 
 Continuous control with deep reinforcement learning.

 
 In International Conference on Learning Representations , 2016.

 

 
 [117] 
 
Shuangfei Fan and Bert Huang.

 
 Conditional labeled graph generation with gans.

 
 In ICLR Workshop on Representation Learning on Graphs and
Manifolds , 2019.

 

 
 [118] 
 
Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi,
and Stefanie Jegelka.

 
 Representation learning on graphs with jumping knowledge networks.

 
 In International Conference on Machine Learning , pages
5453–5462, 2018.

 

 
 [119] 
 
Augustus Odena, Christopher Olah, and Jonathon Shlens.

 
 Conditional image synthesis with auxiliary classifier gans.

 
 In International conference on machine learning , pages
2642–2651, 2017.

 

 
 [120] 
 
Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros.

 
 Unpaired image-to-image translation using cycle-consistent
adversarial networks.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 2223–2232, 2017.

 

 
 [121] 
 
Tao Lei, Yu Zhang, Sida I Wang, Hui Dai, and Yoav Artzi.

 
 Simple recurrent units for highly parallelizable recurrence.

 
 In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing , pages 4470–4481, 2018.

 

 
 [122] 
 
Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.

 
 Reasoning with neural tensor networks for knowledge base completion.

 
 In Advances in neural information processing systems , pages
926–934, 2013.

 

 
 [123] 
 
Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen
Paul Smolley.

 
 Least squares generative adversarial networks.

 
 In Proceedings of the IEEE international conference on computer
vision , pages 2794–2802, 2017.

 

 
 [124] 
 
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio.

 
 Density estimation using real nvp.

 
 In International Conference on Learning Representations , 2017.

 

 
 [125] 
 
Durk P Kingma and Prafulla Dhariwal.

 
 Glow: Generative flow with invertible 1x1 convolutions.

 
 In Advances in neural information processing systems , pages
10215–10224, 2018.

 

 
 [126] 
 
George Papamakarios, Theo Pavlakou, and Iain Murray.

 
 Masked autoregressive flow for density estimation.

 
 In Advances in Neural Information Processing Systems , pages
2338–2347, 2017.

 

 
 [127] 
 
Marcus Olivecrona, Thomas Blaschke, Ola Engkvist, and Hongming Chen.

 
 Molecular de-novo design through deep reinforcement learning.

 
 Journal of cheminformatics , 9(1):48, 2017.

 

 
 [128] 
 
Hanjun Dai, Yingtao Tian, Bo Dai, Steven Skiena, and Le Song.

 
 Syntax-directed variational autoencoder for structured data.

 
 In International Conference on Learning Representations , 2018.

 

 
 [129] 
 
Rafael Gómez-Bombarelli, Jennifer N Wei, David Duvenaud, José Miguel
Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla,
Jorge Aguilera-Iparraguirre, Timothy D Hirzel, Ryan P Adams, and Alán
Aspuru-Guzik.

 
 Automatic chemical design using a data-driven continuous
representation of molecules.

 
 ACS central science , 4(2):268–276, 2018.

 

 
 [130] 
 
Mariya Popova, Olexandr Isayev, and Alexander Tropsha.

 
 Deep reinforcement learning for de novo drug design.

 
 Science advances , 4(7):eaap7885, 2018.

 

 
 [131] 
 
Chuan Wang, Nianwen Xue, and Sameer Pradhan.

 
 A transition-based algorithm for amr parsing.

 
 In Proceedings of the 2015 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies , pages 366–375, 2015.

 

 
 [132] 
 
Chunchuan Lyu and Ivan Titov.

 
 Amr parsing as graph prediction with latent alignment.

 
 In Proceedings of the 56th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 397–407, 2018.

 

 
 [133] 
 
Sheng Zhang, Xutai Ma, Kevin Duh, and Benjamin Van Durme.

 
 Amr parsing as sequence-to-graph transduction.

 
 In Proceedings of the 57th Annual Meeting of the Association for
Computational Linguistics , pages 80–94, 2019.

 

 
 [134] 
 
Bo Chen, Le Sun, and Xianpei Han.

 
 Sequence-to-action: End-to-end semantic graph generation for semantic
parsing.

 
 In Proceedings of the 56th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 766–777, 2018.

 

 
 [135] 
 
Timothy Dozat and Christopher D. Manning.

 
 Deep biaffine attention for neural dependency parsing.

 
 In 5th International Conference on Learning Representations,
ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track
Proceedings . OpenReview.net, 2017.

 

 
 [136] 
 
Yuxuan Wang, Wanxiang Che, Jiang Guo, and Ting Liu.

 
 A neural transition-based approach for semantic dependency graph
parsing.

 
 In AAAI , pages 5561–5568, 2018.

 

 
 [137] 
 
Timothy Dozat and Christopher D Manning.

 
 Simpler but more accurate semantic dependency parsing.

 
 In Proceedings of the 56th Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers) , pages 484–490, 2018.

 

 
 [138] 
 
Daniel D. Johnson.

 
 Learning graphical state transitions.

 
 In 5th International Conference on Learning Representations,
ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track
Proceedings , 2017.

 

 
 [139] 
 
Jiuxiang Gu, Handong Zhao, Zhe Lin, Sheng Li, Jianfei Cai, and Mingyang Ling.

 
 Scene graph generation with external knowledge and image
reconstruction.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 1969–1978, 2019.

 

 
 [140] 
 
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.

 
 Faster r-cnn: Towards real-time object detection with region proposal
networks.

 
 In Advances in neural information processing systems , pages
91–99, 2015.

 

 
 [141] 
 
Yikang Li, Wanli Ouyang, Bolei Zhou, Kun Wang, and Xiaogang Wang.

 
 Scene graph generation from objects, phrases and region captions.

 
 In Proceedings of the IEEE International Conference on Computer
Vision , pages 1261–1270, 2017.

 

 
 [142] 
 
Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei.

 
 Scene graph generation by iterative message passing.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 5410–5419, 2017.

 

 
 [143] 
 
Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, and Devi Parikh.

 
 Graph r-cnn for scene graph generation.

 
 In Proceedings of the European conference on computer vision
(ECCV) , pages 670–685, 2018.

 

 
 [144] 
 
Mengshi Qi, Weijian Li, Zhengyuan Yang, Yunhong Wang, and Jiebo Luo.

 
 Attentive relational networks for mapping images to scene graphs.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 3957–3966, 2019.

 

 
 [145] 
 
Tianshui Chen, Weihao Yu, Riquan Chen, and Liang Lin.

 
 Knowledge-embedded routing network for scene graph generation.

 
 In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition , pages 6163–6171, 2019.

 

 
 [146] 
 
Alejandro Newell and Jia Deng.

 
 Pixels to graphs by associative embedding.

 
 In Advances in neural information processing systems , pages
2171–2180, 2017.

 

 
 [147] 
 
Mahmoud Khademi and Oliver Schulte.

 
 Deep generative probabilistic graph neural networks for scene graph
generation.

 
 In AAAI , pages 11237–11245, 2020.

 

 
 [148] 
 
Yikang Li, Wanli Ouyang, Bolei Zhou, Jianping Shi, Chao Zhang, and Xiaogang
Wang.

 
 Factorizable net: an efficient subgraph-based framework for scene
graph generation.

 
 In Proceedings of the European Conference on Computer Vision
(ECCV) , pages 335–351, 2018.

 

 
 [149] 
 
Matthew Klawonn and Eric Heim.

 
 Generating triples with adversarial networks for scene graph
construction.

 
 In AAAI , 2018.

 

 
 [150] 
 
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu,
Michele Catasta, and Jure Leskovec.

 
 Open graph benchmark: Datasets for machine learning on graphs.

 
 In Advances in neural information processing systems , 2020.

 

 
 [151] 
 
Peter Ertl and Ansgar Schuffenhauer.

 
 Estimation of synthetic accessibility score of drug-like molecules
based on molecular complexity and fragment contributions.

 
 Journal of cheminformatics , 1(1):8, 2009.

 

 
 [152] 
 
G Richard Bickerton, Gaia V Paolini, Jérémy Besnard, Sorel Muresan, and
Andrew L Hopkins.

 
 Quantifying the chemical beauty of drugs.

 
 Nature chemistry , 4(2):90–98, 2012.

 

 
 [153] 
 
Carlo Biffi, Ozan Oktay, Giacomo Tarroni, Wenjia Bai, Antonio De Marvao,
Georgia Doumou, Martin Rajchl, Reem Bedair, Sanjay Prasad, Stuart Cook,
et al.

 
 Learning interpretable anatomical features through deep generative
models: Application to cardiac remodeling.

 
 In International conference on medical image computing and
computer-assisted intervention , pages 464–471. Springer, 2018.

 

 
 [154] 
 
Andrey Voynov and Artem Babenko.

 
 Rpgan: Gans interpretability via random routing.

 
 arXiv preprint arXiv:1912.10920 , 2019.

 

 
 [155] 
 
Carlo Biffi, Juan J Cerrolaza, Giacomo Tarroni, Wenjia Bai, Antonio De Marvao,
Ozan Oktay, Christian Ledig, Loic Le Folgoc, Konstantinos Kamnitsas, Georgia
Doumou, et al.

 
 Explainable anatomical shape analysis through deep hierarchical
generative models.

 
 IEEE Transactions on Medical Imaging , 2020.

 

 
 [156] 
 
Tsung-Hsien Wen, Yishu Miao, Phil Blunsom, and Steve Young.

 
 Latent intention dialogue models.

 
 In International Conference on Machine Learning , pages
3732–3741, 2017.

 

 
 [157] 
 
Tiancheng Zhao, Kyusong Lee, and Maxine Eskenazi.

 
 Unsupervised discrete sentence representation learning for
interpretable neural dialog generation.

 
 In Proceedings of the 56th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 1098–1107, 2018.

 

 
 [158] 
 
Wenxian Shi, Hao Zhou, Ning Miao, and Lei Li.

 
 Dispersed exponential family mixture vaes for interpretable text
generation.

 
 In International Conference on Machine Learning , pages
8840–8851, 2020.

 

 
 [159] 
 
Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex
Graves, et al.

 
 Conditional image generation with pixelcnn decoders.

 
 In Advances in neural information processing systems , pages
4790–4798, 2016.

 

 
 [160] 
 
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan
Catanzaro.

 
 High-resolution image synthesis and semantic manipulation with
conditional gans.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 8798–8807, 2018.

 

 
 [161] 
 
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, and Eric P Xing.

 
 Toward controlled generation of text.

 
 In International Conference on Machine Learning , pages
1587–1596, 2017.

 

 
 [162] 
 
Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, and Bing Liu.

 
 Emotional chatting machine: Emotional conversation generation with
internal and external memory.

 
 In AAAI , 2018.

 

 
 [163] 
 
Nitish Shirish Keskar, Bryan McCann, Lav R Varshney, Caiming Xiong, and Richard
Socher.

 
 Ctrl: A conditional transformer language model for controllable
generation.

 
 arXiv preprint arXiv:1909.05858 , 2019.

 

 
 
 
 
 
 
 
 | 
 
 
 Faezeh Faez received her B.Sc. and the M.Sc. degrees in Software Engineering from Sharif University of Technology, Tehran, Iran. She is currently a Ph.D. candidate in Artificial Intelligence in the Department of Computer Engineering at Sharif University of Technology. Her current research interests include machine learning, deep learning, and deep graph generative models. 
 | 

 
 
 
 
 | 
 
 
 Yassaman Ommi is currently a B.Sc. student in computer science at Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran. Her current research interests include graph-based deep learning, pattern recognition, and complex networks. 
 | 

 
 
 
 
 | 
 
 
 Mahdieh Soleymani Baghshah received the B.Sc., M.Sc., and Ph.D. degrees from the Department of Computer Engineering, Sharif University of Technology, Iran, in 2003, 2005, and 2010, respectively. She is an assistant professor with the Computer Engineering Department, Sharif University of Technology, Tehran, Iran. Her research interests include machine learning and deep learning. 
 | 

 
 
 
 
 | 
 
 
 Hamid R. Rabiee (SM’07) received his BS and MS degrees (with Great Distinction) in Electrical Engineering from CSULB, Long Beach, CA (1987, 1989), his EEE degree in Electrical and Computer Engineering from USC, Los Angeles, CA (1993), and his Ph.D. in Electrical and Computer Engineering from Purdue University, West Lafayette, IN, in 1996. From 1993 to 1996 he was a Member of Technical Staff at AT T Bell Laboratories. From 1996 to 1999 he worked as a Senior Software Engineer at Intel Corporation. He was also with PSU, OGI and OSU universities as an adjunct professor of Electrical and Computer Engineering from 1996-2000. Since September 2000, he has joined Sharif University of Technology, Tehran, Iran. He was also a visiting professor at the Imperial College of London for the 2017-2018 academic year. He is the founder of Sharif University Advanced Information and Communication Technology Research Institute (AICT), ICT Innovation Center, Advanced Technologies Incubator (SATI), Digital Media Laboratory (DML), Mobile Value Added Services Laboratory (VASL), Bioinformatics and Computational Biology Laboratory (BCB) and Cognitive Neuroengineering Research Center. He is also a consultant and member of AI in Health Expert Group at WHO. He has been the founder of many successful High-Tech start-up companies in the field of ICT as an entrepreneur. He is currently a Professor of Computer Engineering at Sharif University of Technology, and Director of AICT, DML, and VASL. He has received numerous awards and honors for his Industrial, scientific and academic contributions, and holds three patents. His research interests include statistical machine learning, Bayesian statistics, data analytics and complex networks with applications in social networks, multimedia systems, cloud and IoT privacy, bioinformatics, and brain networks. 
 |