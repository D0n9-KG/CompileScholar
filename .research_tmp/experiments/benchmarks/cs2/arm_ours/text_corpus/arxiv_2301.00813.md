A Survey on Protein Representation Learning: Retrospect and Prospect 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2301.00813v1 [cs.LG] 31 Dec 2022 
 
 

# A Survey on Protein Representation Learning: Retrospect and Prospect

 
 
 Lirong Wu
 † † thanks: Equal contribution, † Corresponding author 
 Affiliation:  AI Lab, Research Center for Industries of the Future, Westlake University
 
 Affiliation:  College of Computer Science and Technology, Zhejiang University
 
 Email:  {wulirong,huangyufei,linhaitao,stan.zq.li}@westlake.edu.cn 
 
    
 Yufei Huang
 
 Affiliation:  AI Lab, Research Center for Industries of the Future, Westlake University
 
    
 Haitao Lin
 
 Affiliation:  AI Lab, Research Center for Industries of the Future, Westlake University
 
 Affiliation:  College of Computer Science and Technology, Zhejiang University
 
    
 Stan Z. Li
 

 Abstract 
 
 Proteins are fundamental biological entities that play a key role in life activities. The amino acid sequences of proteins can be folded into stable 3D structures in the real physicochemical world, forming a special kind of sequence-structure data . With the development of Artificial Intelligence (AI) techniques, Protein Representation Learning (PRL) has recently emerged as a promising research topic for extracting informative knowledge from massive protein sequences or structures. To pave the way for AI researchers with little bioinformatics background, we present a timely and comprehensive review of PRL formulations and existing PRL methods from the perspective of model architectures, pretext tasks, and downstream applications. We first briefly introduce the motivations for protein representation learning and formulate it in a general and unified framework. Next, we divide existing PRL methods into three main categories: sequence-based, structure-based, and sequence-structure co-modeling. Finally, we discuss some technical challenges and potential directions for improving protein representation learning. The latest advances in PRL methods are summarized in a GitHub repository https://github.com/LirongWu/awesome-protein-representation-learning .

 
 
 

## 1 Introduction

 
 Proteins perform specific biological functions that are essential for all living organisms and therefore play a key role when investigating the most fundamental questions in the life sciences. The proteins are composed of one or several chains of amino acids that fold into a stable 3D structure to enable various biological functionalities. Therefore, understanding, predicting, and designing proteins for biological processes are critical for medical, pharmaceutical, and genetic research.

 
 
 Previous approaches on protein modeling are mostly driven by biological or physical priors, and they explore complex sequence-structure-function relationships through energy minimization Rohl et al. (2004) ; Xu and Zhang (2011) , dynamics simulations Hospital et al. (2015) ; Karplus and
Petsko (1990) , etc. With the development of artificial intelligence and low-cost sequencing technologies, data-driven Protein Representation Learning (PRL) Jumper et al. (2021) ; Rao et al. (2019) ; Rives et al. (2021) ; Hermosilla and
Ropinski (2022) ; Jing et al. (2020) has made remarkable progress due to its superior performance in modeling complex nonlinear relationships. The primary goal of protein representation learning is to extract transferable knowledge from protein data with well-designed model architectures and pretext tasks, and then generalize the learned knowledge to various protein-related downstream applications, ranging from structure prediction to sequence design. Despite their great progress, it is still tricky for AI researchers without bioinformatics background to get started with protein representation learning, and one obstacle is the vast amount of physicochemical knowledge involved behind the proteins. Therefore, a survey on PRL methods that is friendly to the AI community is urgently needed.

 
 
 Existing surveys related to PRL Iuchi et al. (2021) ; Unsal et al. (2020) ; Hu et al. (2021) ; Torrisi et al. (2020) are mainly developed from the perspective of biological applications, but do not go deeper into other important aspects, such as model architectures and pretext tasks. Overall, our contributions can be summarized as follows: (1) Comprehensive review. Our survey provides a comprehensive and up-to-date review of existing PRL methods from the perspective of the model architectures and pretext tasks. (2) New taxonomy. We divide existing PRL methods into three categories: sequence-based, structure-based, and sequence-structure co-modeling. (3) Detailed Implementations. We summarize the paper lists and open-source codes in a public GitHub repository, setting the stage for the development of more future works. (4) Future directions. We point out the technical limitations of current research and discuss several promising directions.

 
 
 

## 2 Notation and Problem Statement

 
 The sequence of amino acids can be folded into a stable 3D structure, forming a special kind of sequence-structure data , which determines its properties and functions. Therefore, we can model each protein as a graph 𝒢 = ( 𝒱 , ℰ , 𝒳 , ℱ ) \mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{X},\mathcal{F}) , where 𝒱 \mathcal{V} is the ordered set of N N nodes in the graph representing amino acid residues and ℰ ∈ 𝒱 × 𝒱 \mathcal{E}\in\mathcal{V}\times\mathcal{V} is the set of edges that connects the nodes. Each node u ∈ 𝒱 u\in\mathcal{V} in graph 𝒢 \mathcal{G} can be attributed with a scalar-vector tuple 𝐱 u = ( s u , V u ) \mathbf{x}_{u}=(s_{u},V_{u}) , where s u ∈ ℝ O s_{u}\in\mathbb{R}^{O} and V u ∈ ℝ 3 × P V_{u}\in\mathbb{R}^{3\times P} . Each edge e ∈ ℰ e\in\mathcal{E} can be attributed with a scalar-vector tuple 𝐟 e = ( s e , V e ) \mathbf{f}_{e}=(s_{e},V_{e}) , where s e ∈ ℝ T s_{e}\in\mathbb{R}^{T} and V e ∈ ℝ 3 × D V_{e}\in\mathbb{R}^{3\times D} .

 
 
 Given a model architecture f θ ​ ( ⋅ ) f_{\theta}(\cdot) and a set of K K losses of pretext tasks { ℒ p ​ r ​ e ( 1 ) ​ ( θ , η 1 ) , ℒ p ​ r ​ e ( 2 ) ​ ( θ , η 2 ) , ⋯ , ℒ p ​ r ​ e ( K ) ​ ( θ , η K ) } \{\mathcal{L}_{pre}^{(1)}(\theta,\eta_{1}),\mathcal{L}_{pre}^{(2)}(\theta,\eta_{2}),\cdots,\mathcal{L}_{pre}^{(K)}(\theta,\eta_{K})\} with projection heads { g η k ​ ( ⋅ ) } k = 1 K \{g_{\eta_{k}}(\cdot)\}_{k=1}^{K} , Protein Representation Learning (PRL) usually works in a two-stage manner: (1) Pre-training the model f θ ​ ( ⋅ ) f_{\theta}(\cdot) with pretext tasks; and (2) Fine-tuning the pre-trained model f θ i ​ n ​ i ​ t ​ ( ⋅ ) f_{\theta_{init}}(\cdot) with a projection head g ω ​ ( ⋅ ) g_{\omega}(\cdot) under the supervision of a specific downstream task ℒ t ​ a ​ s ​ k ​ ( θ , ω ) \mathcal{L}_{task}(\theta,\omega) . The learning objective can be formulated as

 
 
 
 | 
 θ ∗ , ω ∗ = \displaystyle\theta^{*},\omega^{*}= | 
 arg ⁡ min ( θ , ω ) ​ ℒ t ​ a ​ s ​ k ​ ( θ i ​ n ​ i ​ t , ω ) , \displaystyle\arg\min_{(\theta,\omega)}\mathcal{L}_{task}(\theta_{init},\omega), | 
 | 
 (1) | 

 
 | 
 s.t.     ​ θ i ​ n ​ i ​ t , { η k ∗ } k = 1 K = \displaystyle\text{s.t.}\text{ }\text{ }\theta_{init},\{\eta_{k}^{*}\}_{k=1}^{K}= | 
 arg ⁡ min θ , { η k } k = 1 K ⁡ ∑ k = 1 K λ k ​ ℒ p ​ r ​ e ( k ) ​ ( θ , η k ) \displaystyle\mathop{\arg\min}_{\theta,\{\eta_{k}\}_{k=1}^{K}}\sum_{k=1}^{K}\lambda_{k}\mathcal{L}_{pre}^{(k)}(\theta,\eta_{k}) | 
 | 
 

 where { λ k } k = 1 K \{\lambda_{k}\}_{k=1}^{K} are trade-off task hyperparameters. A high-level overview of the PRL framework is shown in Fig.  1 . In practice, if we set K = 1 K=1 , ω = η 1 \omega\!=\!\eta_{1} , i.e., ℒ p ​ r ​ e ( 1 ) ​ ( θ , η 1 ) = ℒ t ​ a ​ s ​ k ​ ( θ , ω ) \mathcal{L}_{pre}^{(1)}(\theta,\eta_{1})\!=\!\mathcal{L}_{task}(\theta,\omega) , it is equivalent to learning task-specific representations directly under downstream supervision, which in this survey can be considered as a special case of Eq. ( 1 ).

 
 
 Figure 1: A general framework for protein representation learning. 
 
 
 In this survey, we mainly focus on the model architecture f θ ​ ( ⋅ ) f_{\theta}(\cdot) and pretext tasks { ℒ p ​ r ​ e ( k ) ​ ( θ , η k ) } k = 1 K \{\mathcal{L}_{pre}^{(k)}(\theta,\eta_{k})\}_{k=1}^{K} for protein representation learning, and defer the discussion on downstream applications until Sec.  5 . A high-level overview of this survey with some representative examples is shown in Fig.  2 .

 
 {forest} 
 Figure 2: A high-level overview of this survey with representative examples. 
 
 
 

## 3 Model Architectures

 
 In this section, we summarize some commonly used model architectures for learning protein sequences or structures.

 
 

### 3.1 Sequence-based Encoder

 
 The sequence encoder takes as input ( 𝒱 , 𝒳 ) (\mathcal{V},\mathcal{X}) and then aims to capture the dependencies between amino acids. Wang et al. (2019) treats protein sequences as a special “biological language” and then establishes an analogy between such “biological language” and natural (textual) language. Inspired by this, many classical model architectures developed for natural language processing can be directly extended to handle protein sequences Asgari et al. (2019) . Depending on whether a single sequence or multiple sequences are to be encoded, there are a variety of different sequence-based encoders.

 
 

#### 3.1.1 Single Sequences

 
 The commonly used sequence encoders for modeling single sequences include Variational Auto-Encoder (VAE) Sinai et al. (2017) ; Ding et al. (2019) , Recurrent Neural Networks (RNNs) Armenteros et al. (2020) , Long Short-Term Memory (LSTM) Hochreiter and
Schmidhuber (1997) , BERT Devlin et al. (2018) , Transformer Vaswani et al. (2017) . Based on the vanilla Transformer, Wu et al. (2022) proposes a novel geometry-inspired transformer (Geoformer) to further distill the structural and physical pairwise relationships between amino acids into the learned protein representation. If we do not consider the ordering of amino acids in the sequences, we can also directly apply Convolutional Neural Networks (CNNs) LeCun et al. (1995) or ResNet He et al. (2016) to capture the local dependencies between adjacent amino acids.

 
 
 

#### 3.1.2 MSA Sequences

 
 The long-standing practices in computational biology are to make inferences from a family of evolutionarily related sequences Weigt et al. (2009) ; Thomas et al. (2005) ; Lapedes et al. (1999) . Therefore, there have been several multiple sequences encoders proposed to capture co-evolutionary information by taking as input a set of sequences in the form of multiple sequence alignment (MSA). For example, MSA Transformer Rao et al. (2021) extends the self-attention mechanism to the MSA setting, which interleaves self-attention across rows and columns to capture dependencies between amino acids and between sequences. As a crucial component of AlphaFold2, Evoformer Jumper et al. (2021) alternatively updates MSA and Pair representations in each block, which encode co-evolutionary information in sequences and relations between residues, respectively.

 
 
 
 

### 3.2 Structure-based Encoder

 
 Despite the effectiveness of sequence-based encoders, the power of pre-training with protein structures has been rarely explored, even though protein structures are known to be determinants of protein functions. To better utilize this critical structural information, a large number of structure-based encoders have been proposed to model structural information, which can be mainly divided into three categories: feature map-based, message-passing GNNs, and geometric GNNs.

 
 

#### 3.2.1 Feature map-based Methods

 
 The use of deep learning to model protein 3D structures could be traced back to a decade ago Zhang and Zhang (2010) ; Schaap et al. (2001) . Early methods directly extracted several hand-crafted feature maps from protein structures and then applied 3D CNNs to model the geometric information of proteins Derevyanko et al. (2018) ; Amidi et al. (2018) ; Townshend et al. (2019) . Later work extended 3D CNNs to spherical convolution for identifying interaction patterns on protein surfaces Sverrisson et al. (2021) ; Gainza et al. (2020) .

 
 
 

#### 3.2.2 Message-passing GNNs

 
 To further capture the geometric relationships and biomedical interactions between amino acids, it has been proposed to first construct a graph from the extracted feature maps by thresholding or k k Nearest Neighbors ( k k NN) Preparata and
Shamos (2012) . Then, many existing message-passing Graph Neural Networks (GNNs) can be directly applied to model protein structures, including Graph Convolutional Network (GCN) Kipf and Welling (2016) , Graph Isomorphism Network (GIN) Xu et al. (2018) , and GraphSAGE Hamilton et al. (2017) . However, the edges in the protein graph may have some key properties, such as dihedral angles and directions, which determine the biological function of proteins. With this in mind, there have been several structure-based encoders proposed to simultaneously leverages the node and edge features of the protein graph. For example, Hermosilla et al. (2020) proposes IE convolution (IEconv) to simultaneously capture the primary, secondary and tertiary structures of proteins by incorporating intrinsic and extrinsic distances between nodes. Besides, Hermosilla and
Ropinski (2022) adopts a similar architecture to IEConv, but introduces seven additional edge features to efficiently describe the relative position and orientation of neighboring nodes. Furthermore, GearNet Zhang et al. (2022) proposes a simple structure encoder, which encodes spatial information by adding different types of sequential or structural edges and then performs both node-level and edge-level message passing simultaneously.

 
 
 

#### 3.2.3 Geometric GNNs

 
 The above message-passing GNNs incorporate the 3D geometry of proteins by encoding the vector features V u V_{u} / V e V_{e} into rotation-invariant scalars s u s_{u} / s e s_{e} . However, reducing this vector information directly to scalars may not fully capture complex geometry. Therefore, geometric-aware neural networks are proposed to bake 3D rigid transformations into network operations, leading to SO(3)-invariant and equivariant GNNs. For example, Jing et al. (2020) introduces Geometric Vector Perceptrons (GVPs), which replace standard multi-layer perceptrons (MLPs) in feed-forward layers and operate directly on both scalar and vector features under a global coordinate system. Besides, Aykent and Xia (2022) proposes Geometric Bottleneck
Perceptron (GBPs) to integrate geometric features and capture complex geometric relations in the 3D structure, based on which a new SO(3)-equivariant message passing neural network is proposed to support a variety of geometric representation learning tasks. To achieve more sensitive geometric awareness in both global transformations and local relations, Li et al. (2022) proposes Directed Weight Perceptrons (DWPs) by extending not only the hidden neurons but the weights from scalars to 2D/3D vectors, naturally saturating the network with 3D structures in the Euclidean space.

 
 
 
 

### 3.3 Sequence-structure Encoder

 
 Compared to sequence- and structure-based encoders, comparatively less work has focused on the co-encoding of protein sequences and structures. The mainstream model architecture is to extract amino acid representations as node features by a language model and then capture the dependencies between amino acids using a GNN module. For example, Gligorijević et al. (2021) introduces DeepFRI, a Graph Convolutional Network (GCN) for predicting protein functions by leveraging sequence representations extracted from a protein language model (LSTM) and protein structures. Besides, LM-GVP Wang et al. (2021) is composed of a protein language model (composed of Transformer blocks) and a GVP network, where the protein LM takes protein sequences as input to compute amino acid embeddings and the GVP network is used to make predictions about protein properties on a graph derived from the protein 3D structure. Moreover, You and Shen (2022) applies the hierarchical RNN and GAT to encode both protein sequences and structures and proposes a cross-interaction module to enforce a learned relationship between the encoded embeddings of the two protein modalities.

 
 
 
 

## 4 Pretext Task

 
 The pretext tasks are designed to extract meaningful representations from massive data through optimizing some well-designed objective functions. In this section, we summarize some commonly used pretext tasks for learning on proteins.

 
 

### 4.1 Sequence-based Pretext Task

 
 There have been many pretext tasks proposed for pre-training language models, including Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) Devlin et al. (2018) , which can be naturally extended to pre-train protein sequences. We divide existing sequence-based pretext tasks into two main categories: self-supervised and supervised.

 
 

#### 4.1.1 Self-supervised Pretext Task

 
 The self-supervised pretext tasks utilize the training data itself as supervision signals without the need for additional annotations. If we consider an amino acid in a sequence as a word in a sentence, we can naturally extend masked language modeling to protein sequences. For example, we can statically or dynamically mask out a single or a set of contiguous amino acids and then predict the masked amino acids from the remaining sequences Rao et al. (2019) ; Elnaggar et al. (2020) ; Rives et al. (2021) ; Rao et al. (2021) ; Nambiar et al. (2020) ; Xiao et al. (2021) . Besides, McDermott et al. (2021) combines adversarial training with MLM and proposes to mask amino acids in a learnable manner. Taking into account the dependence between masked amino acids, Pairwise MLM (PMLM) He et al. (2021) proposes to model the probability of a pair of masked amino acids instead of predicting the probability of a single amino acid. Besides, Next Amino acid Prediction (NAP) Alley et al. (2019) ; Elnaggar et al. (2020) ; Strodthoff et al. (2020) aims to predict the type of the next amino acid based on a set of given sequence fragments. Different from the above methods, Contrastive Predictive Coding (CPC) Lu et al. (2020) applies different augmentation transformations on the input sequence to generate different views, and then maximizes the agreement of two jointly sampled pairs against that of two independently sampled pairs.

 
 
 

#### 4.1.2 Supervised Pretext Task

 
 The supervised pretext tasks use additional labels as auxiliary information to guide the model to learn knowledge relevant to downstream tasks. For example, PLUS Min et al. (2021) devises a protein-specific pretext task, namely Same-Family Prediction (SFP), which trains a model to predict whether a given protein pair belongs to the same protein family. The protein family labels provide weak structural information and help the model learn structurally contextualized representations. Besides, Sturmfels et al. (2020) proposes to use HMM profiles derived from MSA as labels and then take Profile Prediction as a pretext task to help the model learn information about protein structures. In addition, to leverage the exponentially growing protein sequences that lack costly structural annotations, Progen Madani et al. (2020) trains a language model with conditioning tags that encode various annotations, such as taxonomic, functional, and locational information.

 
 
 
 

### 4.2 Structure-based Pretext Task

 
 Despite the great progress in the design of structure-based encoders and graph-based pretext tasks Wu et al. (2021) ; Xie et al. (2022) ; Liu et al. (2022b) , there are few efforts focusing on the structure-based pre-training of proteins. Existing structure-based pretext tasks for proteins can be mainly classified into two branches: contrastive and predictive methods.

 
 

#### 4.2.1 Contrastive Pretext Task

 
 The primary goal of contrastive methods is to maximize the agreement of two jointly sampled positive pairs. For example, Multiview Contrast Hermosilla and
Ropinski (2022) proposes to randomly sample two sub-structures from each protein, encoder them into two representations, and finally maximize the similarity between representations from the same protein while minimizing the similarity between representations from different proteins. Besides, Zhang et al. (2022) adopts almost the same architecture as Multiview Contrast, but replaces GearNet with IEConv as the structure encoder.

 
 
 

#### 4.2.2 Predictive Pretext Task

 
 The contrastive methods deal with the inter-data information (data-data pairs). In contrast, the predictive methods aim to self-generate informative labels from the data as supervision and handle the data-label relationships. Categorized by different types of pseudo labels, the predictive methods have different designs that can capture different levels of structural protein information. For example, Chen et al. (2022) proposes two predictive tasks, namely Distance Prediction and Angle Prediction , which take hidden representations of residues as input and aim to predict the relative distance between pairwise residues and the angle between two edges, respectively, which helps to learn structure-aware protein representations. Furthermore, Hermosilla and
Ropinski (2022) propose Residue Type Prediction and Dihedral Prediction based on geometric or biochemical properties. Specifically, Residue Type Prediction randomly masks the node features of some residues and then lets the structure-based encoders predict these masked residue types. Instead, Dihedral Prediction constructs a learning objective by predicting the dihedral angle between three consecutive edges. Besides, You and Shen (2022) proposes graph completion (GraphComp), which takes as input a protein graph with partially masked residues and then makes predictions for those masked tokens.

 
 
 Table 1: Summary of representative protein representation learning methods. 
 
 
 
 
 Method | 
 Category | 
 Architecture | 
 Pretext Task | 
 Year | 

 
 Bio2Vec-CNN Wang et al. (2019) | 
 Sequence-based | 
 CNN | 
 - | 
 2019 | 

 
 TAPE Rao et al. (2019) | 
 Sequence-based | 
 ResNet, LSTM, Transformer | 
 
 
 
 Masked Language Modeling, | 

 
 Next Amino Acid Prediction | 

 | 
 2019 | 

 
 UniRep Alley et al. (2019) | 
 Sequence-based | 
 Multiplicative LSTM | 
 Next Amino Acid Prediction | 
 2019 | 

 
 TripletProt Nourani et al. (2020) | 
 Sequence-based | 
 Siamese Networks | 
 Contrastive Predictive Coding | 
 2020 | 

 
 PLP-CNN Shanehsazzadeh et al. (2020) | 
 Sequence-based | 
 CNN | 
 - | 
 2020 | 

 
 CPCProt Lu et al. (2020) | 
 Sequence-based | 
 GRU, LSTM | 
 Contrastive Predictive Coding | 
 2020 | 

 
 MuPIPR Zhou et al. (2020) | 
 Sequence-based | 
 GRU, LSTM | 
 Next Amino Acid Prediction | 
 2020 | 

 
 ProtTrans Elnaggar et al. (2020) | 
 Sequence-based | 
 Transformer, Bert, XLNet | 
 Masked Language Modeling | 
 2020 | 

 
 DMPfold Kandathil et al. (2020) | 
 Sequence-based | 
 GRU, ResNet | 
 - | 
 2020 | 

 
 Profile Prediction Sturmfels et al. (2020) | 
 Sequence-based | 
 Transformer | 
 HMM Profile Prediction | 
 2020 | 

 
 PRoBERTa Nambiar et al. (2020) | 
 Sequence-based | 
 Transformer | 
 Masked Language Modeling | 
 2020 | 

 
 UDSMProt Strodthoff et al. (2020) | 
 Sequence-based | 
 LSTM | 
 Next Amino Acid Prediction | 
 2020 | 

 
 ESM-1b Rives et al. (2021) | 
 Sequence-based | 
 Transformer | 
 Masked Language Modeling | 
 2021 | 

 
 PMLM He et al. (2021) | 
 Sequence-based | 
 Transformer | 
 Pairwise Masked Language Modeling | 
 2021 | 

 
 MSA Transformer Rao et al. (2021) | 
 Sequence-based | 
 MSA Transformer | 
 Masked Language Modeling | 
 2021 | 

 
 ProteinLM Xiao et al. (2021) | 
 Sequence-based | 
 BERT | 
 Masked Language Modeling | 
 2021 | 

 
 PLUS Min et al. (2021) | 
 Sequence-based | 
 Bidirectional RNN | 
 
 
 
 Masked Language Modeling, | 

 
 Same-Family Prediction | 

 | 
 2021 | 

 
 Adversarial MLM McDermott et al. (2021) | 
 Sequence-based | 
 Transformer | 
 
 
 
 Masked Language Modeling, | 

 
 Adversarial Training | 

 | 
 2021 | 

 
 ProteinBERT Brandes et al. (2022) | 
 Sequence-based | 
 BERT | 
 Masked Language Modeling | 
 2022 | 

 
 CARP Yang et al. (2022a) | 
 Sequence-based | 
 CNN | 
 Masked Language Modeling | 
 2022 | 

 
 3DCNN Derevyanko et al. (2018) | 
 Structure-based | 
 3DCNN | 
 - | 
 2018 | 

 
 IEConv Hermosilla et al. (2020) | 
 Structure-based | 
 IEConv | 
 - | 
 2020 | 

 
 GVP-GNN Jing et al. (2020) | 
 Structure-based | 
 GVP | 
 - | 
 2020 | 

 
 GraphMS Cheng et al. (2021) | 
 Structure-based | 
 GCN | 
 Multiview Contrast | 
 2021 | 

 
 DL-MSFM Gelman et al. (2021) | 
 Structure-based | 
 GCN | 
 - | 
 2021 | 

 
 PG-GNN Xia and Ku (2021) | 
 Structure-based | 
 PG-GNN | 
 - | 
 2021 | 

 
 CRL Hermosilla and
Ropinski (2022) | 
 Structure-based | 
 IEConv | 
 Multiview Contrast | 
 2022 | 

 
 DW-GNN Li et al. (2022) | 
 Structure-based | 
 DWP | 
 - | 
 2022 | 

 
 GBPNet Aykent and Xia (2022) | 
 Structure-based | 
 GBP | 
 - | 
 2022 | 

 
 GearNet Zhang et al. (2022) | 
 Structure-based | 
 GearNet | 
 
 
 
 Multiview Contrast, | 

 
 Distance and Dihedral Prediction, | 

 
 Residue Type Prediction | 

 | 
 2022 | 

 
 ATOMRefine Wu and Cheng (2022) | 
 Structure-based | 
 SE(3) Transformer | 
 - | 
 2022 | 

 
 STEPS Chen et al. (2022) | 
 Structure-based | 
 GIN | 
 Distance and Dihedral Prediction | 
 2022 | 

 
 GraphCPI Quan et al. (2019) | 
 Co-Modeling | 
 CNN, GNN | 
 - | 
 2019 | 

 
 MT-LSTM Bepler and Berger (2019) | 
 Co-Modeling | 
 Bidirectional LSTM | 
 
 
 
 Contact prediction, | 

 
 Pairwise Similarity Prediction | 

 | 
 2019 | 

 
 LM-GVP Wang et al. (2021) | 
 Co-Modeling | 
 Transformer, GVP | 
 - | 
 2021 | 

 
 AlphaFold2 Jumper et al. (2021) | 
 Co-Modeling | 
 Evoformer | 
 
 
 
 Masked Language Modeling, | 

 
 Full-atomic Structure Prediction | 

 | 
 2021 | 

 
 DeepFRI Gligorijević et al. (2021) | 
 Co-Modeling | 
 LSTM, GCN | 
 - | 
 2021 | 

 
 HJRSS Mansoor et al. (2021) | 
 Co-Modeling | 
 SE(3) Transformer | 
 
 
 
 Masked Language Modeling, | 

 
 Graph Completion | 

 | 
 2021 | 

 
 GraSR Xia et al. (2022) | 
 Co-Modeling | 
 LSTM, GCN | 
 Momentum Contrast | 
 2022 | 

 
 CPAC You and Shen (2022) | 
 Co-Modeling | 
 Hierarchical RNN, GAT | 
 
 
 
 Masked Language Modeling, | 

 
 Graph Completion | 

 | 
 2022 | 

 
 MIF-ST Yang et al. (2022b) | 
 Co-Modeling | 
 CNN, GNN | 
 Masked Inverse Folding | 
 2022 | 

 
 OmegaFold Wu et al. (2022) | 
 Co-Modeling | 
 Geoformer | 
 
 
 
 Masked Language Modeling, | 

 
 Full-atomic Structure Prediction | 

 | 
 2022 | 

 

 
 
 
 
 

### 4.3 Sequence-structure Pretext Task

 
 Most of the existing methods design pretext tasks for a single modality but ignore the dependencies between sequences and structures. If we can design the pretext task based on both protein sequences and structures, it should capture richer information than using single modality data. In practice, there is no clear boundary between pretext tasks and downstream tasks. For example, AlphaFold2 Jumper et al. (2021) takes full-atomic structure prediction as a downstream task. However, if we are concerned with protein property prediction, structure prediction can also be considered as a pretext task that enables the learned sequence representations to contain sufficient structural information. It was found by Hu et al. (2022) that the representations from AlphFold2’s Evoformer could work well on various protein-related downstream tasks, including fold classification, stability prediction, etc. Moreover, Yang et al. (2022b) proposes a novel pre-training pretext task, namely Masked Inverse Folding (MIF), which trains a model to reconstruct the original amino acids conditioned on the corrupted sequence and the backbone structure.

 
 
 
 

## 5 Downstream Tasks (Applications)

 
 In the above, we have presented a variety of commonly used model architectures and pretext tasks for protein representation learning, based on which we summarized the surveyed works in Table.  1 , listing their categories, model architectures, pretext tasks, and publication years. In this section, we can divide existing downstream tasks for protein representation learning into the following four main categories: protein property prediction, protein (complex) structure prediction, protein design, and structure-based drug design.

 
 
 It is worth noting that some downstream tasks have labels (i.e., model outputs) that do not change with rigid body transformations of the inputs (if they can, e.g., protein structures). For example, various protein property prediction tasks take a transformable protein structure as input and output a constant prediction, usually modeled as a simple multi-label classification problem or multiple binary classification problem. However, the labels of some downstream tasks will change equivariantly with the inputs, and these tasks are getting more and more attention. Typically, the learning objectives of these tasks are structure-related, and they usually have higher requirements on the model architecture, requiring the model to be SE(3)-equivariant. We believe that from the perspective of protein representation learning, the approaches to different downstream tasks can also learn from each other.

 
 

### 5.1 Protein Property Prediction

 
 The protein property prediction aims to regress or classify some important properties from protein sequences or structures that are closely related to biological functions, such as the types of secondary structure, the strength of connections between amino acids, types of protein folding, fluorescence intensity, protein stability, etc. Rao et al. (2019) . Besides, several protein-specific prediction tasks can also be grouped into this category, including quality evaluation of protein folding Baldassarre et al. (2021) , predicting the effect of mutations on protein function Meier et al. (2021) , and predicting protein-protein interactions Wang et al. (2019) .

 
 
 

### 5.2 Protein (Complex) Structure Prediction

 
 The primary goal of protein structure prediction is to predict the structural coordinates from a given set of amino acid sequences. Some approaches aim to predict only backbone coordinates Baek et al. (2021) ; Si et al. (2020) , while others focus on the more challenging full-atomic coordinate predictions Jumper et al. (2021) ; Wu et al. (2022) ; Rao et al. (2021) . On the other hand, protein structure refinement Hiranuma et al. (2021) ; Wu and Cheng (2022) proposes to update a coarse protein structure to generate a more fine-grained structure in an iterative manner. Besides, the task of protein structure inpainting aims to reconstruct the complete protein structure from a partially given sub-structure McPartlon and
Xu (2022) or distance map Lee and Kim (2022) .

 
 
 

### 5.3 Protein Design

 
 Deep learning-based protein design has made tremendous progress in recent years, and the major works can be divided into three categories. The first one is to pre-train the model with a large number of sequences from the same protein family, and then use it to generate new homologous sequences Smith and Smith (1990) . The structure-based methods aim to directly generate the protein sequences under the condition of a given protein structure Ingraham et al. (2019) . The last and most challenging one is the de novo protein design Huang et al. (2016) ; Korendovych and
DeGrado (2020) ; Koepnick et al. (2019) , which aims to generate both protein sequences and structures conditioned on taxonomic and keyword tags such as molecular function and cellular component.

 
 
 

### 5.4 Structure-Based Drug Design

 
 Structure-Based Drug Design (SBDD) is a promising direction for fast and cost-efficient compound discovery. Specifically, SBDD designs inhibitors or activators (usually small molecules, i.e., drugs) directly against protein targets of interest, which means a high success rate and efficiency Kuntz (1992) ; Drews (2000) . In the past two years, a line of auto-regressive methods have been proposed for SBDD Liu et al. (2022a) ; Peng et al. (2022) ; Masuda et al. (2020) , which generate molecule atoms one by one conditioned on given structure context of protein targets. Recently, there are some works based on Denoising Diffusion Probabilistic Model (DDPM) Lin et al. (2022) ; Schneuing et al. (2022) . Targeting on specific protein pockets, the diffusion-based methods generate molecule atoms as a whole from random gaussian noise.

 
 
 The above methods are all dependent on a proper representation module of protein, especially the protein structure. The early attempt of deep generative models in this field Luo et al. (2021) uses 3D CNN as the protein structure context encoder to get meaningful and roto-translation invariant features. With the development of protein structure representation methods, particularly the geometric-aware models, subsequent methods widely use geometric-(equi/in)variant networks, such as EGNN Gong and Cheng (2019) , GVP Jing et al. (2020) , and IPA Jumper et al. (2021) , as the backbones. It is worth noting that protein representation models are not only common in various protein structure context encoders, but many generative decoders can also adopt its architectural design. From this example, we can see that protein representation is a very fundamental problem and that many downstream tasks involving proteins can benefit from advances of protein representation research in various aspects, including better embeddings and more excellent model architectures.

 
 
 
 

## 6 Deep Insights and Future Outlooks

 

### 6.1 Deeper Insights

 
 On the basis of a detailed review of the model architectures, pretext tasks, and downstream tasks, we would like to provide some deeper insights into protein representation learning.

 
 

#### 6.1.1 Insights 1: PRL is the core of deep protein modeling

 
 With the development of deep learning, deep protein modeling is becoming a popular research topic, and one of its core is how to learn “meaningful” representations for proteins. This involves three key issues: (1) Feature Extraction: model architectures; (2) Pre-training: pretext tasks; and (3) Application: downstream tasks. An in-depth investigation of the above three key issues is of great importance for the development of more deep protein modeling methods.

 
 
 

#### 6.1.2 Insights 2: Task-level convertibility

 
 Throughout this survey, one of the main points we have emphasized is the convertibility between downstream tasks and pretext tasks. We believe we are the first to explain the role of pretext tasks from this perspective, which seems to have been rarely involved in previous work. For example, we directly categorize some well-known downstream tasks, such as full-atomic structure prediction, as a specific kinds of pretext tasks. The motivation behind such an understanding lies in the fact that the definition of a task is itself a relative concept and that different tasks can help the model extract different aspects of information, which may be complementary to each other. For example, full-atomic structure prediction helps the model capture rich structural information, which is also beneficial for various protein property prediction tasks, such as folding prediction, since it is known that protein structure often determines protein function. This suggests that whether a specific task is a downstream task or a pretext task usually depends on what we are concerned about, and the role of a task may keep changing from application to application.

 
 
 

#### 6.1.3 Insights 3: Data-specific criterion for design selections

 
 It is tricky to discuss the advantages and disadvantages of different methods or designs because the effectiveness of different methods depends heavily on the size, format, and complexity of the data. For example, for simple small-scale data, Transformer is not necessarily more effective than traditional LSTM for sequence modeling, and the situation may be completely opposite for large-scale complex data. Therefore, there is no “optimal” architecture or pretext task that works for all data types and downstream tasks, and the criterion for the selection of architecture and pretext task is data-specific.

 
 
 
 

### 6.2 Future Outlooks

 
 Despite the great progress of existing methods, challenges still exist due to the complexity of proteins. In this section, we suggest some promising directions for future work.

 
 

#### 6.2.1 Direction 1: Broader application scenarios

 
 The biological research topics on proteins are diverse, but most of the existing work has delved into only a small subset of them, due to the fact that these topics have been well formalized by some representative works, such as AlphaFlod2 Jumper et al. (2021) for protein structure prediction and TAPE Rao et al. (2019) for protein property prediction. As a result, it is more worthwhile to explore the role of protein representation learning in a wider range of biological application scenarios than to design some overly complex modules for subtle performance gains in a well-formalized application.

 
 
 

#### 6.2.2 Direction 2: Unified evaluation protocols

 
 Research in protein representation learning is now in an era of barbarism. While a great deal of new works are emerging every day, most of them are on unfair comparisons, such as with different datasets, architectures, metrics, etc. For example, some MSA-based works on structure prediction have been blatantly compared with those single-sequence-based works and claimed to be better. To promote the health of the field, there is an urgent need to establish unified evaluation protocols in various downstream tasks to provide fair comparisons.

 
 
 

#### 6.2.3 Direction 3: Protein-specific designs

 
 Previous PRL methods directly take mature architectures and pretext tasks from the natural language processing field to train proteins. For example, modeling protein sequences using LSTM may be a major innovation, but replacing LSTM with Bi-LSTM for stuble performance improvements makes little sense. Now, it is time to step out of this comfort zone of scientific research, and we should no longer be satisfied with simply extending techniques from other domains to the protein domain. PRL is not only a machine learning problem but also a biological problem, so we should consider designing more protein-specific architectures and pretext tasks by incorporating protein-related domain knowledge. In particular, most of the existing work on PRL is based on unimodal protein sequences or structures, and it requires more work exploring sequence-structure co-modeling to fully explore the correspondence between 1D sequences and 3D structures.

 
 
 

#### 6.2.4 Direction 4: Margin from pre-training to fine-tuning

 
 Currently, tremendous efforts are focusing on protein pre-training strategies. However, how to fine-tune these pre-trained models to specific downstream tasks is still under-explored. Though numerous strategies have been proposed to address this problem in the fields of computer vision and natural language processing Zhuang et al. (2020) , they are difficult to be directly applied to proteins. One obstacle to knowledge transfer is the huge variability between different protein datasets, both in terms of sequence length and structural complexity. The second
one is poor generalization of pre-trained models especially for various tasks where collecting labeled data is laborious. Therefore, it is an important issue to design protein-specific techniques to minimize the margin between pre-training and downstream tasks.

 
 
 

#### 6.2.5 Direction 5: Lack of explainability

 
 While existing protein representation learning methods have achieved promising results on a variety of downstream tasks, we still know little about what the model has learned from protein data. Which of the feature patterns, sequence fragments, or sequence-structure relationships has been learned? These are important issues for understanding and interpreting model behavior, especially for those privacy-secure tasks such as drug design, but are missing in current PRL works. Overall, the interpretability of PRL methods remains to be explored further in many respects, which helps us understand how the model works and provides a guide for better usage.

 
 
 
 
 

## 7 Conclusions

 
 A comprehensive survey of the literature on protein representation learning is conducted in this paper. We develop a general unified framework for PRL methods. Moreover, we systematically divide existing PRL methods into three main categories: sequence-based, structure-based, and sequence-structure co-modeling from three different perspectives, including model architectures, pretext tasks, and downstream applications. Finally, we point out the technical limitations of the current research and provide promising directions for future work on PRL. We hope this survey to pave the way for follow-up AI researchers with no bioinformatics background, setting the stage for the development of more future works.

 
 
 

## References

 
 
 Alley et al. [2019] 
 
Ethan C Alley, Grigory Khimulya, Surojit Biswas, Mohammed AlQuraishi, and
George M Church.

 
 Unified rational protein engineering with sequence-based deep
representation learning.

 
 Nature methods , 16(12):1315–1322, 2019.

 

 
 Amidi et al. [2018] 
 
Afshine Amidi, Shervine Amidi, Dimitrios Vlachakis, Vasileios Megalooikonomou,
Nikos Paragios, and Evangelia I Zacharaki.

 
 Enzynet: enzyme classification using 3d convolutional neural networks
on spatial representation.

 
 PeerJ , 6:e4750, 2018.

 

 
 Armenteros et al. [2020] 
 
Jose Juan Almagro Armenteros, Alexander Rosenberg Johansen, Ole Winther, and
Henrik Nielsen.

 
 Language modelling for biological sequences–curated datasets and
baselines.

 
 BioRxiv , 2020.

 

 
 Asgari et al. [2019] 
 
Ehsaneddin Asgari, Nina Poerner, Alice C McHardy, and Mohammad RK Mofrad.

 
 Deepprime2sec: deep learning for protein secondary structure
prediction from the primary sequences.

 
 BioRxiv , page 705426, 2019.

 

 
 Aykent and Xia [2022] 
 
Sarp Aykent and Tian Xia.

 
 Gbpnet: Universal geometric representation learning on protein
structures.

 
 In Proceedings of the 28th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining , pages 4–14, 2022.

 

 
 Baek et al. [2021] 
 
Minkyung Baek, Frank DiMaio, Ivan Anishchenko, Justas Dauparas, Sergey
Ovchinnikov, Gyu Rie Lee, Jue Wang, Qian Cong, Lisa N Kinch, R Dustin
Schaeffer, et al.

 
 Accurate prediction of protein structures and interactions using a
three-track neural network.

 
 Science , 373(6557):871–876, 2021.

 

 
 Baldassarre et al. [2021] 
 
Federico Baldassarre, David Menéndez Hurtado, Arne Elofsson, and Hossein
Azizpour.

 
 Graphqa: protein model quality assessment using graph convolutional
networks.

 
 Bioinformatics , 37(3):360–366, 2021.

 

 
 Bepler and Berger [2019] 
 
Tristan Bepler and Bonnie Berger.

 
 Learning protein sequence embeddings using information from
structure.

 
 arXiv preprint arXiv:1902.08661 , 2019.

 

 
 Brandes et al. [2022] 
 
Nadav Brandes, Dan Ofer, Yam Peleg, Nadav Rappoport, and Michal Linial.

 
 Proteinbert: A universal deep-learning model of protein sequence and
function.

 
 Bioinformatics , 38(8):2102–2110, 2022.

 

 
 Chen et al. [2022] 
 
Can Chen, Jingbo Zhou, Fan Wang, Xue Liu, and Dejing Dou.

 
 Structure-aware protein self-supervised learning.

 
 arXiv preprint arXiv:2204.04213 , 2022.

 

 
 Cheng et al. [2021] 
 
Shicheng Cheng, Liang Zhang, Bo Jin, Qiang Zhang, Xinjiang Lu, Mao You, and
Xueqing Tian.

 
 Graphms: Drug target prediction using graph representation learning
with substructures.

 
 Applied Sciences , 11(7):3239, 2021.

 

 
 Derevyanko et al. [2018] 
 
Georgy Derevyanko, Sergei Grudinin, Yoshua Bengio, and Guillaume Lamoureux.

 
 Deep convolutional networks for quality assessment of protein folds.

 
 Bioinformatics , 34(23):4046–4053, 2018.

 

 
 Devlin et al. [2018] 
 
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

 
 Bert: Pre-training of deep bidirectional transformers for language
understanding.

 
 arXiv preprint arXiv:1810.04805 , 2018.

 

 
 Ding et al. [2019] 
 
Xinqiang Ding, Zhengting Zou, and Charles L Brooks III.

 
 Deciphering protein evolution and fitness landscapes with latent
space models.

 
 Nature communications , 10(1):1–13, 2019.

 

 
 Drews [2000] 
 
Jürgen Drews.

 
 Drug discovery: A historical perspective.

 
 Science , 287(5460):1960–1964, 2000.

 

 
 Elnaggar et al. [2020] 
 
Ahmed Elnaggar, Michael Heinzinger, Christian Dallago, Ghalia Rihawi, Yu Wang,
Llion Jones, Tom Gibbs, Tamas Feher, Christoph Angerer, Martin Steinegger,
et al.

 
 Prottrans: towards cracking the language of life’s code through
self-supervised deep learning and high performance computing.

 
 arXiv preprint arXiv:2007.06225 , 2020.

 

 
 Gainza et al. [2020] 
 
Pablo Gainza, Freyr Sverrisson, Frederico Monti, Emanuele Rodola, D Boscaini,
MM Bronstein, and BE Correia.

 
 Deciphering interaction fingerprints from protein molecular surfaces
using geometric deep learning.

 
 Nature Methods , 17(2):184–192, 2020.

 

 
 Gelman et al. [2021] 
 
Sam Gelman, Sarah A Fahlberg, Pete Heinzelman, Philip A Romero, and Anthony
Gitter.

 
 Neural networks to learn protein sequence–function relationships
from deep mutational scanning data.

 
 Proceedings of the National Academy of Sciences ,
118(48):e2104878118, 2021.

 

 
 Gligorijević et al. [2021] 
 
Vladimir Gligorijević, P Douglas Renfrew, Tomasz Kosciolek, Julia Koehler
Leman, Daniel Berenberg, Tommi Vatanen, Chris Chandler, Bryn C Taylor, Ian M
Fisk, Hera Vlamakis, et al.

 
 Structure-based protein function prediction using graph convolutional
networks.

 
 Nature communications , 12(1):1–14, 2021.

 

 
 Gong and Cheng [2019] 
 
Liyu Gong and Qiang Cheng.

 
 Exploiting edge features for graph neural networks.

 
 In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , pages 9211–9219, 2019.

 

 
 Hamilton et al. [2017] 
 
Will Hamilton, Zhitao Ying, and Jure Leskovec.

 
 Inductive representation learning on large graphs.

 
 In Neural information processing systems , pages 1024–1034,
2017.

 

 
 He et al. [2016] 
 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

 
 Deep residual learning for image recognition.

 
 In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 770–778, 2016.

 

 
 He et al. [2021] 
 
Liang He, Shizhuo Zhang, Lijun Wu, Huanhuan Xia, Fusong Ju, He Zhang, Siyuan
Liu, Yingce Xia, Jianwei Zhu, Pan Deng, et al.

 
 Pre-training co-evolutionary protein representation via a pairwise
masked language model.

 
 arXiv preprint arXiv:2110.15527 , 2021.

 

 
 Hermosilla and
Ropinski [2022] 
 
Pedro Hermosilla and Timo Ropinski.

 
 Contrastive representation learning for 3d protein structures.

 
 arXiv preprint arXiv:2205.15675 , 2022.

 

 
 Hermosilla et al. [2020] 
 
Pedro Hermosilla, Marco Schäfer, Matěj Lang, Gloria Fackelmann,
Pere Pau Vázquez, Barbora Kozlíková, Michael Krone, Tobias
Ritschel, and Timo Ropinski.

 
 Intrinsic-extrinsic convolution and pooling for learning on 3d
protein structures.

 
 arXiv preprint arXiv:2007.06252 , 2020.

 

 
 Hiranuma et al. [2021] 
 
Naozumi Hiranuma, Hahnbeom Park, Minkyung Baek, Ivan Anishchenko, Justas
Dauparas, and David Baker.

 
 Improved protein structure refinement guided by deep learning based
accuracy estimation.

 
 Nature communications , 12(1):1–11, 2021.

 

 
 Hochreiter and
Schmidhuber [1997] 
 
Sepp Hochreiter and Jürgen Schmidhuber.

 
 Long short-term memory.

 
 Neural computation , 9(8):1735–1780, 1997.

 

 
 Hospital et al. [2015] 
 
Adam Hospital, Josep Ramon Goñi, Modesto Orozco, and Josep L Gelpí.

 
 Molecular dynamics simulations: advances and applications.

 
 Advances and applications in bioinformatics and chemistry:
AABC , 8:37, 2015.

 

 
 Hu et al. [2021] 
 
Lun Hu, Xiaojuan Wang, Yu-An Huang, Pengwei Hu, and Zhu-Hong You.

 
 A survey on computational models for predicting protein–protein
interactions.

 
 Briefings in Bioinformatics , 22(5):bbab036, 2021.

 

 
 Hu et al. [2022] 
 
Mingyang Hu, Fajie Yuan, Kevin K Yang, Fusong Ju, Jin Su, Hui Wang, Fei Yang,
and Qiuyang Ding.

 
 Exploring evolution-based -free protein language models as protein
function predictors.

 
 arXiv preprint arXiv:2206.06583 , 2022.

 

 
 Huang et al. [2016] 
 
Po-Ssu Huang, Scott E Boyken, and David Baker.

 
 The coming of age of de novo protein design.

 
 Nature , 537(7620):320–327, 2016.

 

 
 Ingraham et al. [2019] 
 
John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola.

 
 Generative models for graph-based protein design.

 
 Advances in neural information processing systems , 32, 2019.

 

 
 Iuchi et al. [2021] 
 
Hitoshi Iuchi, Taro Matsutani, Keisuke Yamada, Natsuki Iwano, Shunsuke Sumi,
Shion Hosoda, Shitao Zhao, Tsukasa Fukunaga, and Michiaki Hamada.

 
 Representation learning applications in biological sequence analysis.

 
 Computational and Structural Biotechnology Journal ,
19:3198–3208, 2021.

 

 
 Jing et al. [2020] 
 
Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael JL Townshend, and Ron
Dror.

 
 Learning from protein structure with geometric vector perceptrons.

 
 arXiv preprint arXiv:2009.01411 , 2020.

 

 
 Jumper et al. [2021] 
 
John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov,
Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin
Žídek, Anna Potapenko, et al.

 
 Highly accurate protein structure prediction with alphafold.

 
 Nature , 596(7873):583–589, 2021.

 

 
 Kandathil et al. [2020] 
 
Shaun M Kandathil, Joe G Greener, Andy M Lau, and David T Jones.

 
 Deep learning-based prediction of protein structure using learned
representations of multiple sequence alignments.

 
 Biorxiv , pages 2020–11, 2020.

 

 
 Karplus and
Petsko [1990] 
 
Martin Karplus and Gregory A Petsko.

 
 Molecular dynamics simulations in biology.

 
 Nature , 347(6294):631–639, 1990.

 

 
 Kipf and Welling [2016] 
 
Thomas N Kipf and Max Welling.

 
 Semi-supervised classification with graph convolutional networks.

 
 arXiv preprint arXiv:1609.02907 , 2016.

 

 
 Koepnick et al. [2019] 
 
Brian Koepnick, Jeff Flatten, Tamir Husain, Alex Ford, Daniel-Adriano Silva,
Matthew J Bick, Aaron Bauer, Gaohua Liu, Yojiro Ishida, Alexander Boykov,
et al.

 
 De novo protein design by citizen scientists.

 
 Nature , 570(7761):390–394, 2019.

 

 
 Korendovych and
DeGrado [2020] 
 
Ivan V Korendovych and William F DeGrado.

 
 De novo protein design, a retrospective.

 
 Quarterly reviews of biophysics , 53, 2020.

 

 
 Kuntz [1992] 
 
Irwin D. Kuntz.

 
 Structure-based strategies for drug design and discovery.

 
 Science , 257(5073):1078–1082, 1992.

 

 
 Lapedes et al. [1999] 
 
Alan S Lapedes, Bertrand G Giraud, LonChang Liu, and Gary D Stormo.

 
 Correlated mutations in models of protein sequences: phylogenetic and
structural effects.

 
 Lecture Notes-Monograph Series , pages 236–256, 1999.

 

 
 LeCun et al. [1995] 
 
Yann LeCun, Yoshua Bengio, et al.

 
 Convolutional networks for images, speech, and time series.

 
 The handbook of brain theory and neural networks ,
3361(10):1995, 1995.

 

 
 Lee and Kim [2022] 
 
Jin Sub Lee and Philip M Kim.

 
 Proteinsgm: Score-based generative modeling for de novo protein
design.

 
 bioRxiv , 2022.

 

 
 Li et al. [2022] 
 
Jiahan Li, Shitong Luo, Congyue Deng, Chaoran Cheng, Jiaqi Guan, Leonidas
Guibas, Jian Peng, and Jianzhu Ma.

 
 Directed weight neural networks for protein structure representation
learning.

 
 arXiv preprint arXiv:2201.13299 , 2022.

 

 
 Lin et al. [2022] 
 
Haitao Lin, Yufei Huang, Meng Liu, Xuanjing Li, Shuiwang Ji, and Stan Z Li.

 
 Diffbp: Generative diffusion of 3d molecules for target protein
binding.

 
 arXiv preprint arXiv:2211.11214 , 2022.

 

 
 Liu et al. [2022a] 
 
Meng Liu, Youzhi Luo, Kanji Uchino, Koji Maruhashi, and Shuiwang Ji.

 
 Generating 3d molecules for target protein binding.

 
 In International Conference on Machine Learning , 2022.

 

 
 Liu et al. [2022b] 
 
Yixin Liu, Ming Jin, Shirui Pan, Chuan Zhou, Yu Zheng, Feng Xia, and Philip Yu.

 
 Graph self-supervised learning: A survey.

 
 IEEE Transactions on Knowledge and Data Engineering , 2022.

 

 
 Lu et al. [2020] 
 
Amy X Lu, Haoran Zhang, Marzyeh Ghassemi, and Alan Moses.

 
 Self-supervised contrastive learning of protein representations by
mutual information maximization.

 
 BioRxiv , 2020.

 

 
 Luo et al. [2021] 
 
Shitong Luo, Jiaqi Guan, Jianzhu Ma, and Jian Peng.

 
 A 3D generative model for structure-based drug design.

 
 In Thirty-Fifth Conference on Neural Information Processing
Systems , 2021.

 

 
 Madani et al. [2020] 
 
Ali Madani, Bryan McCann, Nikhil Naik, Nitish Shirish Keskar, Namrata Anand,
Raphael R Eguchi, Po-Ssu Huang, and Richard Socher.

 
 Progen: Language modeling for protein generation.

 
 arXiv preprint arXiv:2004.03497 , 2020.

 

 
 Mansoor et al. [2021] 
 
Sanaa Mansoor, Minkyung Baek, Umesh Madan, and Eric Horvitz.

 
 Toward more general embeddings for protein design: Harnessing joint
representations of sequence and structure.

 
 bioRxiv , 2021.

 

 
 Masuda et al. [2020] 
 
Tomohide Masuda, Matthew Ragoza, and David Ryan Koes.

 
 Generating 3d molecular structures conditional on a receptor binding
site with deep generative models.

 
 arXiv preprint arXiv:2010.14442 , 2020.

 

 
 McDermott et al. [2021] 
 
Matthew McDermott, Brendan Yap, Harry Hsu, Di Jin, and Peter Szolovits.

 
 Adversarial contrastive pre-training for protein sequences.

 
 arXiv preprint arXiv:2102.00466 , 2021.

 

 
 McPartlon and
Xu [2022] 
 
Matthew McPartlon and Jinbo Xu.

 
 Attnpacker: An end-to-end deep learning method for rotamer-free
protein side-chain packing.

 
 bioRxiv , 2022.

 

 
 Meier et al. [2021] 
 
Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, and Alex Rives.

 
 Language models enable zero-shot prediction of the effects of
mutations on protein function.

 
 Advances in Neural Information Processing Systems ,
34:29287–29303, 2021.

 

 
 Min et al. [2021] 
 
Seonwoo Min, Seunghyun Park, Siwon Kim, Hyun-Soo Choi, Byunghan Lee, and
Sungroh Yoon.

 
 Pre-training of deep bidirectional protein sequence representations
with structural information.

 
 IEEE Access , 9:123912–123926, 2021.

 

 
 Nambiar et al. [2020] 
 
Ananthan Nambiar, Maeve Heflin, Simon Liu, Sergei Maslov, Mark Hopkins, and
Anna Ritz.

 
 Transforming the language of life: transformer neural networks for
protein prediction tasks.

 
 In Proceedings of the 11th ACM International Conference on
Bioinformatics, Computational Biology and Health Informatics , pages 1–8,
2020.

 

 
 Nourani et al. [2020] 
 
Esmaeil Nourani, Ehsaneddin Asgari, Alice C McHardy, and Mohammad RK Mofrad.

 
 Tripletprot: Deep representation learning of proteins based on
siamese networks.

 
 Biorxiv , 2020.

 

 
 Peng et al. [2022] 
 
Xingang Peng, Shitong Luo, Jiaqi Guan, Qi Xie, Jian Peng, and Jianzhu Ma.

 
 Pocket2mol: Efficient molecular sampling based on 3d protein pockets.

 
 In International Conference on Machine Learning , 2022.

 

 
 Preparata and
Shamos [2012] 
 
Franco P Preparata and Michael I Shamos.

 
 Computational geometry: an introduction .

 
 Springer Science Business Media, 2012.

 

 
 Quan et al. [2019] 
 
Zhe Quan, Yan Guo, Xuan Lin, Zhi-Jie Wang, and Xiangxiang Zeng.

 
 Graphcpi: Graph neural representation learning for compound-protein
interaction.

 
 In 2019 IEEE International Conference on Bioinformatics and
Biomedicine (BIBM) , pages 717–722. IEEE, 2019.

 

 
 Rao et al. [2019] 
 
Roshan Rao, Nicholas Bhattacharya, Neil Thomas, Yan Duan, Peter Chen, John
Canny, Pieter Abbeel, and Yun Song.

 
 Evaluating protein transfer learning with tape.

 
 Advances in neural information processing systems , 32, 2019.

 

 
 Rao et al. [2021] 
 
Roshan M Rao, Jason Liu, Robert Verkuil, Joshua Meier, John Canny, Pieter
Abbeel, Tom Sercu, and Alexander Rives.

 
 Msa transformer.

 
 In International Conference on Machine Learning , pages
8844–8856. PMLR, 2021.

 

 
 Rives et al. [2021] 
 
Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason
Liu, Demi Guo, Myle Ott, C Lawrence Zitnick, Jerry Ma, et al.

 
 Biological structure and function emerge from scaling unsupervised
learning to 250 million protein sequences.

 
 Proceedings of the National Academy of Sciences ,
118(15):e2016239118, 2021.

 

 
 Rohl et al. [2004] 
 
Carol A Rohl, Charlie EM Strauss, Kira MS Misura, and David Baker.

 
 Protein structure prediction using rosetta.

 
 In Methods in enzymology , volume 383, pages 66–93. Elsevier,
2004.

 

 
 Schaap et al. [2001] 
 
Marcel G Schaap, Feike J Leij, and Martinus Th Van Genuchten.

 
 Rosetta: A computer program for estimating soil hydraulic parameters
with hierarchical pedotransfer functions.

 
 Journal of hydrology , 251(3-4):163–176, 2001.

 

 
 Schneuing et al. [2022] 
 
Arne Schneuing, Yuanqi Du, Charles Harris, Arian Jamasb, Ilia Igashov, Weitao
Du, Tom Blundell, Pietro Lió, Carla Gomes, Max Welling, et al.

 
 Structure-based drug design with equivariant diffusion models.

 
 arXiv preprint arXiv:2210.13695 , 2022.

 

 
 Shanehsazzadeh et al. [2020] 
 
Amir Shanehsazzadeh, David Belanger, and David Dohan.

 
 Is transfer learning necessary for protein landscape prediction?

 
 arXiv preprint arXiv:2011.03443 , 2020.

 

 
 Si et al. [2020] 
 
Dong Si, Spencer A Moritz, Jonas Pfab, Jie Hou, Renzhi Cao, Liguo Wang, Tianqi
Wu, and Jianlin Cheng.

 
 Deep learning to predict protein backbone structure from
high-resolution cryo-em density maps.

 
 Scientific reports , 10(1):1–22, 2020.

 

 
 Sinai et al. [2017] 
 
Sam Sinai, Eric Kelsic, George M Church, and Martin A Nowak.

 
 Variational auto-encoding of protein sequences.

 
 arXiv preprint arXiv:1712.03346 , 2017.

 

 
 Smith and Smith [1990] 
 
Randall F Smith and Temple F Smith.

 
 Automatic generation of primary sequence patterns from sets of
related protein sequences.

 
 Proceedings of the National Academy of Sciences ,
87(1):118–122, 1990.

 

 
 Strodthoff et al. [2020] 
 
Nils Strodthoff, Patrick Wagner, Markus Wenzel, and Wojciech Samek.

 
 Udsmprot: universal deep sequence models for protein classification.

 
 Bioinformatics , 36(8):2401–2409, 2020.

 

 
 Sturmfels et al. [2020] 
 
Pascal Sturmfels, Jesse Vig, Ali Madani, and Nazneen Fatema Rajani.

 
 Profile prediction: An alignment-based pre-training task for protein
sequence models.

 
 arXiv preprint arXiv:2012.00195 , 2020.

 

 
 Sverrisson et al. [2021] 
 
Freyr Sverrisson, Jean Feydy, Bruno E Correia, and Michael M Bronstein.

 
 Fast end-to-end learning on protein surfaces.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 15272–15281, 2021.

 

 
 Thomas et al. [2005] 
 
John Thomas, Naren Ramakrishnan, and Chris Bailey-Kellogg.

 
 Graphical models of residue coupling in protein families.

 
 In Proceedings of the 5th international workshop on
Bioinformatics , pages 12–20, 2005.

 

 
 Torrisi et al. [2020] 
 
Mirko Torrisi, Gianluca Pollastri, and Quan Le.

 
 Deep learning methods in protein structure prediction.

 
 Computational and Structural Biotechnology Journal ,
18:1301–1310, 2020.

 

 
 Townshend et al. [2019] 
 
Raphael Townshend, Rishi Bedi, Patricia Suriana, and Ron Dror.

 
 End-to-end learning on 3d protein structure for interface prediction.

 
 Advances in Neural Information Processing Systems , 32, 2019.

 

 
 Unsal et al. [2020] 
 
Serbulent Unsal, Heval Ataş, Muammer Albayrak, Kemal Turhan, Aybar C
Acar, and Tunca Doğan.

 
 Evaluation of methods for protein representation learning: a
quantitative analysis.

 
 bioRxiv , 2020.

 

 
 Vaswani et al. [2017] 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

 
 Attention is all you need.

 
 Advances in neural information processing systems , 30, 2017.

 

 
 Wang et al. [2019] 
 
Yanbin Wang, Zhu-Hong You, Shan Yang, Xiao Li, Tong-Hai Jiang, and Xi Zhou.

 
 A high efficient biological language model for predicting
protein–protein interactions.

 
 Cells , 8(2):122, 2019.

 

 
 Wang et al. [2021] 
 
Zichen Wang, Steven A Combs, Ryan Brand, Miguel Romero Calvo, Panpan Xu, George
Price, Nataliya Golovach, Emannuel O Salawu, Colby J Wise, Sri Priya
Ponnapalli, et al.

 
 Lm-gvp: A generalizable deep learning framework for protein property
prediction from sequence and structure.

 
 bioRxiv , 2021.

 

 
 Weigt et al. [2009] 
 
Martin Weigt, Robert A White, Hendrik Szurmant, James A Hoch, and Terence Hwa.

 
 Identification of direct residue contacts in protein–protein
interaction by message passing.

 
 Proceedings of the National Academy of Sciences , 106(1):67–72,
2009.

 

 
 Wu and Cheng [2022] 
 
Tianqi Wu and Jianlin Cheng.

 
 Atomic protein structure refinement using all-atom graph
representations and se (3)-equivariant graph neural networks.

 
 bioRxiv , 2022.

 

 
 Wu et al. [2021] 
 
Lirong Wu, Haitao Lin, Cheng Tan, Zhangyang Gao, and Stan Z Li.

 
 Self-supervised learning on graphs: Contrastive, generative, or
predictive.

 
 IEEE Transactions on Knowledge and Data Engineering , 2021.

 

 
 Wu et al. [2022] 
 
Ruidong Wu, Fan Ding, Rui Wang, Rui Shen, Xiwen Zhang, Shitong Luo, Chenpeng
Su, Zuofan Wu, Qi Xie, Bonnie Berger, et al.

 
 High-resolution de novo structure prediction from primary sequence.

 
 bioRxiv , 2022.

 

 
 Xia and Ku [2021] 
 
Tian Xia and Wei-Shinn Ku.

 
 Geometric graph representation learning on protein structure
prediction.

 
 In Proceedings of the 27th ACM SIGKDD Conference on Knowledge
Discovery Data Mining , pages 1873–1883, 2021.

 

 
 Xia et al. [2022] 
 
Chunqiu Xia, Shi-Hao Feng, Ying Xia, Xiaoyong Pan, and Hong-Bin Shen.

 
 Fast protein structure comparison through effective representation
learning with contrastive graph neural networks.

 
 PLoS computational biology , 18(3):e1009986, 2022.

 

 
 Xiao et al. [2021] 
 
Yijia Xiao, Jiezhong Qiu, Ziang Li, Chang-Yu Hsieh, and Jie Tang.

 
 Modeling protein using large-scale pretrain language model.

 
 arXiv preprint arXiv:2108.07435 , 2021.

 

 
 Xie et al. [2022] 
 
Yaochen Xie, Zhao Xu, Jingtun Zhang, Zhengyang Wang, and Shuiwang Ji.

 
 Self-supervised learning of graph neural networks: A unified review.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence ,
2022.

 

 
 Xu and Zhang [2011] 
 
Dong Xu and Yang Zhang.

 
 Improving the physical realism and structural accuracy of protein
models by a two-step atomic-level energy minimization.

 
 Biophysical journal , 101(10):2525–2534, 2011.

 

 
 Xu et al. [2018] 
 
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.

 
 How powerful are graph neural networks?

 
 arXiv preprint arXiv:1810.00826 , 2018.

 

 
 Yang et al. [2022a] 
 
Kevin K Yang, Alex X Lu, and Nicolo K Fusi.

 
 Convolutions are competitive with transformers for protein sequence
pretraining.

 
 bioRxiv , 2022.

 

 
 Yang et al. [2022b] 
 
Kevin K Yang, Niccolò Zanichelli, and Hugh Yeh.

 
 Masked inverse folding with sequence transfer for protein
representation learning.

 
 bioRxiv , 2022.

 

 
 You and Shen [2022] 
 
Yuning You and Yang Shen.

 
 Cross-modality and self-supervised protein embedding for
compound–protein affinity and contact prediction.

 
 Bioinformatics , 38(Supplement_2):ii68–ii74, 2022.

 

 
 Zhang and Zhang [2010] 
 
Jian Zhang and Yang Zhang.

 
 A novel side-chain orientation dependent potential derived from
random-walk reference state for protein fold selection and structure
prediction.

 
 PloS one , 5(10):e15386, 2010.

 

 
 Zhang et al. [2022] 
 
Zuobai Zhang, Minghao Xu, Arian Jamasb, Vijil Chenthamarakshan, Aurelie Lozano,
Payel Das, and Jian Tang.

 
 Protein representation learning by geometric structure pretraining.

 
 arXiv preprint arXiv:2203.06125 , 2022.

 

 
 Zhou et al. [2020] 
 
Guangyu Zhou, Muhao Chen, Chelsea JT Ju, Zheng Wang, Jyun-Yu Jiang, and Wei
Wang.

 
 Mutation effect estimation on protein–protein interactions using
deep contextualized representation learning.

 
 NAR genomics and bioinformatics , 2(2):lqaa015, 2020.

 

 
 Zhuang et al. [2020] 
 
Fuzhen Zhuang, Zhiyuan Qi, Keyu Duan, Dongbo Xi, Yongchun Zhu, Hengshu Zhu, Hui
Xiong, and Qing He.

 
 A comprehensive survey on transfer learning.

 
 Proceedings of the IEEE , 109(1):43–76, 2020.