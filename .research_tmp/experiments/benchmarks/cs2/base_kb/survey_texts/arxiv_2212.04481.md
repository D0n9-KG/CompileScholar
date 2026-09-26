A Survey of Graph Neural Networks for Social Recommender Systems 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2212.04481v3 [cs.SI] 01 May 2024 
 
 

# A Survey of Graph Neural Networks for Social Recommender Systems Thanks:  ∗ Equal Contribution Thanks:  † Equal Contribution Thanks:  + Corresponding author 

 DOI:  10.1145/3661821 Journal:  CSUR 4 CCS:  Computing methodologies Neural networks CCS:  Information systems Social networks CCS:  Information systems Recommender systems 
 
 
 Kartik Sharma ∗ 
 
 email: ksartik@gatech.edu 
 
 Affiliation:  Georgia Institute of Technology , USA 
 
 , 
 Yeon-Chang Lee ∗ 
 
 email: yeonchang@unist.ac.kr 
 
 Affiliation:  UNIST , South Korea 
 
 , 
 Sivagami Nambi
 
 email: sivagami.nambi@gatech.edu 
 
 Affiliation:  Georgia Institute of Technology , USA 
 
 , 
 Aditya Salian † 
 
 email: asalian@gatech.edu 
 
 Affiliation:  Georgia Institute of Technology , USA 
 
 , 
 Shlok Shah † 
 
 email: sshah672@gatech.edu 
 
 Affiliation:  Georgia Institute of Technology , USA 
 
 , 
 Sang-Wook Kim
 
 email: wook@hanyang.ac.kr 
 
 Affiliation:  Hanyang University , South Korea 
 
 and 
 Srijan Kumar + 
 
 email: srijan@gatech.edu 
 
 Affiliation:  Georgia Institute of Technology , USA 
 
 2024 

 Abstract. 
 
 Social recommender systems (SocialRS) simultaneously leverage the user-to-item interactions as well as the user-to-user social relations for the task of generating item recommendations to users.
Additionally exploiting social relations is clearly effective in understanding users’ tastes due to the effects of homophily and social influence.
For this reason, SocialRS has increasingly attracted attention.
In particular, with the advance of graph neural networks (GNN), many GNN-based SocialRS methods have been developed recently.
Therefore, we conduct a comprehensive and systematic review of the literature on GNN-based SocialRS.

 
 In this survey, we first identify 84 84 papers on GNN-based SocialRS after annotating 2,151 papers by following the PRISMA framework (preferred reporting items for systematic reviews and meta-analyses).
Then, we comprehensively review them in terms of their inputs and architectures to propose a novel taxonomy: (1) input taxonomy includes 5 groups of input type notations and 7 groups of input representation notations; (2) architecture taxonomy includes 8 groups of GNN encoder notations, 2 groups of decoder notations, and 12 groups of loss function notations.
We classify the GNN-based SocialRS methods into several categories as per the taxonomy and describe their details.
Furthermore, we summarize benchmark datasets and metrics widely used to evaluate the GNN-based SocialRS methods.
Finally, we conclude this survey by presenting some future research directions.
GitHub repository with the curated list of papers are available at https://github.com/claws-lab/awesome-GNN-social-recsys .

 
 
 
 Keywords:  graph neural networks, social network, recommender systems, social recommendation, survey
 
 

## 1. Introduction

 
 With the advent of online social network platforms ( e .g., Facebook, Twitter, Instagram, etc.), there has been a surge of research efforts in developing social recommender systems (SocialRS), which simultaneously utilize user-user social relations along with user-item interactions to recommend relevant items to users.
Exploiting social relations in recommendation works well because of the effects of social homophily   ( McPherson et al., 2001 ) and social influence   ( Marsden and Friedkin, 1993 ) :
(1) social homophily indicates that a user tends to connect herself to other users with similar attributes and preferences, and
(2) social influence indicates that users with direct or indirect relations tend to influence each other to make themselves become more similar.
Accordingly, SocialRS can effectively mitigate the data sparsity problem by exploiting social neighbors to capture the preferences of a sparsely interacting user.

 
 
 Literature has shown that SocialRS can be applied successfully in various recommendation domains ( e .g., product  ( Wu et al., 2020 ; Wu et al., 2019b ) , music  ( Yu et al., 2021a ; Yu et al., 2021b ; Yu et al., 2022a ) , location  ( Wu et al., 2022a ; Seng et al., 2021 ; Li et al., 2022b ) , and image  ( Wu et al., 2019a ; Wu et al., 2022c ; Tao et al., 2022 ) ), thereby improving user satisfaction.
Furthermore, techniques and insights explored from SocialRS can also be exploited in real-world applications other than recommendations.
For instance, García-Sánchez et al.  ( García-Sánchez et al., 2020 ) leveraged SocialRS to design a decision-making system for marketing ( e .g., advertisement), while Gasparetti et al.  ( Gasparetti et al., 2021 ) analyzed SocialRS in terms of community detection.

 
 2019 2020 2021 2022 ⋆ 0 0 20 20 40 40 Years # Papers 
 Figure 1. The number of papers related to GNN-based SocialRS per year. ⋆ For 2022, we count the number of relevant papers published until October. 
 
 
 Motivated by such wide applicability, there has been an increasing interest in research on developing accurate SocialRS models.
In the early days, research focused on matrix factorization (MF) techniques  ( Ma et al., 2008 ; Ma et al., 2009a ; Ma et al., 2009b ; Jamali and Ester, 2010 ; Ma et al., 2011 ; Yang et al., 2013 ; Tang et al., 2013b ) .
However, MF-based methods cannot effectively model the complex ( i .e., non-linear) relationships inherent in user-user social relations and user-item interactions  ( Shokeen and Rana, 2020a ) .
Motivated by this, most recent works have focused on applying deep-learning techniques to SocialRS, e .g., autoencoder  ( Deng et al., 2017 ; Ying et al., 2016 ) , generative adversarial networks (GAN)  ( Krishnan et al., 2019 ) , and graph neural networks (GNN)  ( Fan et al., 2019 ; Wu et al., 2019a ) .

 
 
 In particular, since user-item interactions and user-user social relations can naturally be represented as graph data, GNN-based SocialRS has increasingly attracted attention in the literature.
As a demonstration, Figure  1 shows that the number of papers related to GNN-based SocialRS has increased consistently since 2019.
Given the growing and timely interest in this area, we survey GNN-based SocialRS methods in this survey.

 
 

### 1.1. Challenges

 
 Applying GNN into SocialRS is not trivial and faces the following challenges.

 
 
 Input representation . The input data should be modeled appropriately into a heterogeneous graph structure. Many SocialRS methods build two separate graphs: one where nodes represent users and items, and edges represent user-item interactions; the other where nodes represent users and edges represent user-user social relations.
Thus, GNN methods for SocialRS need to extract knowledge from both the networks simultaneously for accurate inference. This is in contrast with most regular GNNs that consider only a single network.
Additionally, we note that there are valuable input features in the two networks, such as user/item attributes, item knowledge/relation, and group information. Thus, methods fuse features along with network information in GNN-based SocialRS.
In this survey, we discuss the input types used in GNN-based SocialRS methods and the different ways they are represented as graphs.

 
 
 Design of GNN encoder . The performance of GNN-based SocialRS methods relies heavily on their GNN encoders, which aim to represent users and items into low-dimensional embeddings.
For this reason, existing SocialRS methods have explored various design choices regarding GNN encoders and have adopted different architectures according to their goals.
For instance, many SocialRS methods employ the graph attention neural network (GANN)  ( Veličković et al., 2018 ) to differentiate each user’s preference for items or each user’s influence on their social friends.
On the other hand, some methods  ( Sun et al., 2020 ; Yan et al., 2022 ; Gu et al., 2021 ; Narang et al., 2021 ; Niu et al., 2021 ) use the graph recurrent neural networks (GRNN)  ( Peng et al., 2017 ; Zayats and Ostendorf, 2018 ) to model the sequential behaviors of users.
It should be noted that GNN encoders for SocialRS need to simultaneously consider the characteristics of user-item interactions and user-user social relations.
This is in contrast with GNN encoders for non-SocialRS that model only user-item interactions.
In this survey, we discuss different types of GNN encoders used by SocialRS methods.

 
 
 Training . The training of GNN-based SocialRS should be designed to reflect users’ tastes and items’ characteristics in the embeddings for the corresponding users and items. To this end, SocialRS methods employ well-known loss functions, such as mean squared error (MSE), Bayesian personalized ranking (BPR)  ( Rendle et al., 2009 ) , and cross-entropy (CE), to reconstruct user behaviors. Furthermore, to mitigate the data sparsity problem, some works have additionally employed auxiliary loss functions such as self-supervised loss  ( Liu et al., 2021b ) and group-based loss  ( Leng and Yu, 2022 ; Liao et al., 2022a ) .
It is worth mentioning that loss functions used by GNN-based SocialRS are designed so that rich structural information such as motifs and user attributes can be exploited. These are not considered by loss functions for non-SocialRS.
In this survey, we discuss the training remedies of GNN-based SocialRS methods to learn the user and item embeddings.

 
 
 

### 1.2. Related Surveys

 

 Table 1. Comparison with existing surveys. For each survey, we summarize the topics covered, some statistics regarding GNN-based SocialRS papers ( i .e., relevant papers), and the main scope to survey. 
 
 
 
 
 Surveys | 
 Topics | 
 GNN-based SocialRS Papers | 
 Scope | 

 
 SocialRS | 
 GNN | 
 # Papers | 
 Latest Year | 

 
 ( Tang et al., 2013a ; Yang et al., 2014 ; Xu et al., 2015 ; Dou et al., 2016 ; Chen et al., 2018 ) | 
 ✓ | 
 | 
 0 0 | 
 - | 
 Traditional SocialRS | 

 
 ( Gasparetti et al., 2021 ) | 
 ✓ | 
 | 
 0 0 | 
 - | 
 SocialRS for CD | 

 
 ( Shokeen and Rana, 2020b ) | 
 ✓ | 
 | 
 0 0 | 
 - | 
 General SocialRS | 

 
 ( Shokeen and Rana, 2020a ) | 
 ✓ | 
 ✓ | 
 1 1 | 
 2019 | 
 General SocialRS | 

 
 ( Deng, 2022 ) | 
 ✓ | 
 ✓ | 
 2 2 | 
 2019 | 
 Graph-based RS | 

 
 ( Wang et al., 2021b ) | 
 ✓ | 
 ✓ | 
 3 | 
 2020 | 
 Graph-based RS | 

 
 ( Wu et al., 2022b ) | 
 ✓ | 
 ✓ | 
 14 14 | 
 2021 | 
 GNN -based RS | 

 
 ( Gao et al., 2022 ) | 
 ✓ | 
 ✓ | 
 19 19 | 
 2021 | 
 GNN -based RS | 

 
 Ours | 
 ✓ | 
 ✓ | 
 80 80 | 
 Oct, 2022 | 
 GNN -based SocialRS | 

 

 
 
 • 
 
 ✓: fully covered, 
✓
: partially covered. 

 

 • 
 
 # Papers: the number of GNN-based SocialRS papers included in the survey. 

 

 • 
 
 Latest year: the latest publication year of a relevant paper included in the survey. 

 

 
 
 
 Most of the existing surveys, which fully cover SocialRS papers, focus either on traditional methods  ( Papadimitriou et al., 2012 ; Tang et al., 2013a ; Yang et al., 2014 ; Xu et al., 2015 ; Dou et al., 2016 ; Chen et al., 2018 ; Shokeen and Rana, 2018 ) ( e .g., matrix factorization), feature information  ( Shokeen and Rana, 2020b ) ( e .g., context), or a specific application  ( Gasparetti et al., 2021 ) ( e .g., community detection).
On the other hand, the other related surveys  ( Deng, 2022 ; Wang et al., 2021b ; Wu et al., 2022b ; Gao et al., 2022 ) focus on graph-based recommender systems, including GNN-based RS methods, but they partially cover SocialRS papers in their surveys.
A comparison between the current survey and the previous surveys is shown in Table  1 .

 
 
 Figure 2. A timeline of GNN-based SocialRS methods. We categorize methods according to their GNN encoders: graph convolutional network (GCN), lightweight GCN (LightGCN), graph attention neural networks (GANN), heterogeneous GNN (HetGNN), graph recurrent neural networks (GRNN), hypergraph neural networks (HyperGNN), graph autoencoder (GAE), and hyperbolic GNN. It should be noted that some methods employ two or more GNN encoders in their architectures. 
 
 
 Specifically, several survey papers on SocialRS have been published before 2019   ( Papadimitriou et al., 2012 ; Tang et al., 2013a ; Yang et al., 2014 ; Xu et al., 2015 ; Dou et al., 2016 ; Chen et al., 2018 ; Shokeen and Rana, 2018 ) .
However, they only focus on traditional methods such as matrix factorization and collaborative filtering.
These surveys largely ignore methods that use modern-day deep-learning techniques, in particular GNN.

 
 
 More recent surveys discuss the taxonomy of social recommendation,
starting the comparison of deep-learning based techniques  ( Shokeen and Rana, 2020b ; Shokeen and Rana, 2020a ; Gasparetti et al., 2021 ) .
However, Shokeen and Rana  ( Shokeen and Rana, 2020b ) 
only focus on the taxonomy of feature information regarding social relations,
such as context, trust, and group, used in SocialRS methods, while Gasparetti et al.  ( Gasparetti et al., 2021 ) only discuss SocialRS methods using community detection (CD) techniques.
Shokeen and Rana  ( Shokeen and Rana, 2020a ) include just one social recommendation method based on GNNs.

 
 
 With the advent of GNNs in recommender systems, multiple surveys have been conducted on graph-based recommender systems  ( Wang et al., 2021b ; Wu et al., 2022b ; Gao et al., 2022 ; Deng, 2022 ) .
However, their focus is not on SocialRS as they consider different kinds of recommender systems where graph-learning is employed. They cover only a small section of most representative papers on GNN-based SocialRS. Thus, one cannot rely on these surveys to gain insights on the ever-increasing field of using GNNs for SocialRS.

 
 
 As shown in Table  1 , no survey paper exists in the literature that focuses specifically on GNN-based SocialRS methods. In the current work, we aim to fill this gap by providing a comprehensive and systematic survey on GNN-based SocialRS methods.

 
 0 0 2 2 4 4 6 6 8 8 ACM TIST Data Min. Knowl. Discov. IEEE Intell. Syst. IJCNN IEEE BigData IEEE ICDE IEEE ICDM ECML-PKDD ACM KDD Appl. Intell. Information Sciences Knowl. Based Syst. IEEE ICTAI ACM TOIS DASFAA ACM WSDM ACM CIKM Neurocomputing ACM SIGIR WWW IEEE TKDE # Papers Venues 
 Figure 3. The number of GNN-based SocialRS papers published in relevant journals and conferences. We only present statistics with respect to prominent data mining journals (including IEEE TKDE, ACM TOIS, Knowledge-Based Systems, and Information Sciences) and conferences (including WWW, ACM SIGIR, ACM KDD, ACM CIKM, ACM WSDM, IEEE ICDE, and IEEE ICDM). We believe it would help researchers in this field to identify appropriate venues where GNN-based SocialRS papers are published. 
 
 
 

### 1.3. Contributions

 
 The main contribution of this survey paper is summarized as follows:

 
 
 
 • 
 
 The First Survey in GNN-based SocialRS : To the best of our knowledge, we are the first to systematically dedicate ourselves to reviewing GNN-based SocialRS methods. Most of the existing surveys focus either on traditional methods  ( Papadimitriou et al., 2012 ; Tang et al., 2013a ; Yang et al., 2014 ; Xu et al., 2015 ; Dou et al., 2016 ; Chen et al., 2018 ; Shokeen and Rana, 2018 ) ( e .g., matrix factorization), feature information  ( Shokeen and Rana, 2020b ) ( e .g., context), or a specific application  ( Gasparetti et al., 2021 ) ( e .g., community detection).
The other related surveys  ( Deng, 2022 ; Wang et al., 2021b ; Wu et al., 2022b ; Gao et al., 2022 ) focus on graph-based recommender systems, but they partially cover SocialRS.

 

 • 
 
 Comprehensive Survey : We systematically identify the relevant papers on GNN-based SocialRS by following the guidelines of the preferred reporting items for systematic reviews and meta-analyses (PRISMA framework)  ( Moher et al., 2009 ) . Then, we comprehensively review them in terms of their inputs and architectures. Figure  2 provides a brief timeline of GNN-based SocialRS methods. In addition, Figure  3 shows the number of relevant papers published in relevant journals ( e .g., IEEE TKDE and ACM TOIS) and conferences ( e .g., WWW, ACM SIGIR, and ACM CIKM).

 

 • 
 
 Novel Taxonomy of Inputs and Architectures : We provide a novel taxonomy of inputs and architectures in GNN-based SocialRS methods, enabling researchers to capture the research trends in this field easily. An input taxonomy includes 5 groups of input type notations and 7 groups of input representation notations. On the other hand, an architecture taxonomy includes 8 groups of GNN encoder notations, 2 groups of decoder notations, and 12 groups (4 for primary losses and 8 for auxiliary losses) of loss function notations.

 

 • 
 
 Benchmark Datasets : We review 17 benchmark datasets used to evaluate the performance of GNN-based SocialRS methods. We group the datasets into 8 domains ( i .e., product, location, movie, image, music, bookmark, microblog, and miscellaneous). Also, we present some statistics for each dataset and a list of papers using the dataset.

 

 • 
 
 Future Directions : We discuss the limitations of existing GNN-based SocialRS methods and provide several future research directions.

 

 
 
 
 The rest of this survey paper is organized as follows.
In Section  2 , we introduce the survey methodology based on PRISMA  ( Moher et al., 2009 ) that collects the papers on GNN-based SocialRS thoroughly.
In Section  3 , we define the social recommendation problem.
In Sections  4 and  5 , we review 84 GNN-based SocialRS methods in terms of their inputs and architectures, respectively. We summarize
17 benchmark datasets and 8 evaluation metrics, widely-used in GNN-based SocialRS methods, in Section  6 .
Section  7 discusses future research directions.
Finally, we conclude the paper in Section  8 .

 
 
 
 

## 2. Survey Methodology

 
 Following the guidelines set by the PRISMA  ( Moher et al., 2009 ) , the Scopus index was queried to filter for relevant literature. In particular, the following query was run on October 14, 2022, resulting in 2,151 2,151 papers.

 TITLE-ABS-KEY (social AND (recommendation OR recommender) AND graph) AND ( PUBYEAR 2009) AND ( LIMIT-TO ( LANGUAGE , ‘‘English’’)) 

 
 
 To obtain the final list of relevant papers for the current survey, an iterative strategy of manual reviewing and filtering was carried out, following PRISMA guidelines. Four expert annotators were used to select the relevant papers. Before reviewing the papers, a comprehensive and exhaustive discussion was held among the annotators to discuss and agree upon the definitions of the main concepts that a paper is to be examined for before including it in the survey. These included concepts of Graph Neural Networks and Social Recommendation.

 
 
 Based on these guidelines, each annotator labeled one batch of 200 200 papers together. Each paper in this batch was assigned one of the three categories by each annotator: “Yes”, “No”, and “Maybe”. “Yes” represents full confidence of relevance, “Maybe” represents some confidence of relevance, and “No” represents full confidence of irrelevance of the paper for the current survey. A high inter-annotator agreement of 0.845 0.845 among the annotators was reported on this set.

 
 
 The remaining papers were then divided equally among the annotators without any overlap. The annotator assigned each paper a label of “Yes”, “No”, or “Maybe”. Papers marked “Maybe” were reviewed again by the other annotators to reach a consensus. Finally, papers marked “Yes” were collected together and these served as the focus of our survey. Through this comprehensive process of filtering, we finally found 84 84 papers that study GNN-based SocialRS for our survey paper.

 
 
 Table 2. Notations used in this paper.
 
 
 
 
 
 Notation | 
 Description | 

 
 
 
 𝒰 \mathcal{U} , ℐ \mathcal{I} | 
 Sets of users p i p_{i} and items q j q_{j} | 

 
 𝐑 \mathbf{R} , 𝐒 \mathbf{S} | 
 Matrices representing U-I rating and U-U social | 

 
 𝒩 p i \mathcal{N}_{p_{i}} | 
 Set of items rated by p i p_{i} | 

 
 𝐮 i I \mathbf{u}_{i}^{I} | 
 Embedding of p i p_{i} obtained via the user interaction encoder | 

 
 𝐮 i S \mathbf{u}_{i}^{S} | 
 Embedding of p i p_{i} obtained via the user social encoder | 

 
 𝐮 i \mathbf{u}_{i} | 
 Embedding of p i p_{i} obtained by fusing 𝐮 i I \mathbf{u}_{i}^{I} and 𝐮 i S \mathbf{u}_{i}^{S} | 

 
 𝐯 j \mathbf{v}_{j} | 
 Embedding of q j q_{j} via the item encoder | 

 
 r i ​ j r_{ij} | 
 Real rating score of p i p_{i} on q j q_{j} | 

 
 r ^ i ​ j \hat{r}_{ij} | 
 Predicted preference of p i p_{i} on q j q_{j} via the decoder | 

 

 
 
 
 

## 3. Notations and Problem Definition

 
 The social recommendation problem is formulated as follows.
Let 𝒰 = { p 1 , p 2 , ⋯ , p m } \mathcal{U}=\{p_{1},p_{2},\cdots,p_{m}\} and ℐ = { q 1 , q 2 , ⋯ , q n } \mathcal{I}=\{q_{1},q_{2},\cdots,q_{n}\} be sets of m m users and n n items, respectively.
Also, 𝐑 ∈ ℝ m × n \mathbf{R}\in\mathbb{R}^{m\times n} represents a rating matrix that stores user-item ratings (that we call U-I rating).
 𝐒 ∈ ℝ m × m \mathbf{S}\in\mathbb{R}^{m\times m} represents a social matrix that stores user-user social relations (that we call U-U social).
In addition, 𝒩 p i \mathcal{N}_{p_{i}} indicates a set of items rated by a user p i p_{i} .
In this paper, we use bold uppercase letters and bold lowercase letters to denote matrices and vectors, respectively.
Also, we use calligraphic letters to denote sets and graphs.
Table  2 summarizes a list of notations used in this paper.

 
 
 The goal of GNN-based SocialRS methods is to solve the rating prediction and/or top- N N recommendation tasks.
Given 𝐑 \mathbf{R} and 𝐒 \mathbf{S} , both tasks are formally defined as follows:

 
 
 Problem 1 ( Rating Prediction ). 
 
 The goal is to predict the rating values for unrated items (i.e., ℐ \mathcal{I} \ 𝒩 p i \mathcal{N}_{p_{i}} ) in 𝐑 \mathbf{R} as close as possible to the ground truth. 

 
 
 
 Problem 2 ( Top- N N Recommendation ). 
 
 The goal is to recommend the top- N N items that are most likely to be preferred by each user p i p_{i} among p i p_{i} ’s unrated items (i.e., ℐ \mathcal{I} \ 𝒩 p i \mathcal{N}_{p_{i}} ). 

 
 
 
 

## 4. Taxonomy of Inputs

 
 In this section, we present a taxonomy of inputs for GNN-based SocialRS.
Figures  4 and  5 depict the input types and their representations, respectively.
In the subsequent subsections, we describe each of these in detail.

 
 
 Figure 4. Overview of input types used by GNN-based SocialRS methods. 
 
 

### 4.1. Input Types: Types of Inputs to the Models

 
 In this subsection, we group the input types used by GNN-based SocialRS into 5 categories: user-item ratings, user-user social relations, attributes, knowledge graph (KG), and groups.
Table  3 categorizes all papers based on the input data types they use.

 
 
 Table 3. Taxonomy of input types. 
 
 
 
 
 U-I Rating | 
 U-U Social | 
 Additional Features | 
 Models | 

 
 User | 
 Item | 

 
 Static | 
 Homogeneous | 
 - | 
 - | 
 GraphRec  ( Fan et al., 2019 ) , DANSER  ( Wu et al., 2019b ) , DICER  ( Fu et al., 2021 ) , ASR  ( Jiang et al., 2021b ) , GNNTSR  ( Mandal and Maiti, 2021 ) , GAT-NSR  ( Mu et al., 2019 ) , | 

 
 SoRecGAT  ( Vijaikumar et al., 2019 ) , SAGLG  ( Liu et al., 2021a ) , PA-GAN  ( Hou et al., 2021 ) , MGNN  ( Xiao et al., 2020 ) , MutualRec  ( Xiao et al., 2021 ) , | 

 
 GHSCF  ( Bi et al., 2021 ) , HIDM  ( Li and Mu, 2020 ) , SocialLGN  ( Liao et al., 2022b ) , SOAP-VAE  ( Walker et al., 2021 ) , GraphRec+  ( Fan et al., 2022 ) , DSR  ( Sha et al., 2021 ) , | 

 
 SHGCN  ( Zhu et al., 2021 ) , GTN  ( Hoang et al., 2021 ) , PDARec  ( Zheng et al., 2021 ) , MHCN  ( Yu et al., 2021b ) , SEPT  ( Yu et al., 2021a ) , DcRec  ( Wu et al., 2022a ) , | 

 
 GSFR  ( Xiao et al., 2022 ) , APTE  ( Zhen et al., 2022 ) , EAGCN  ( Wu et al., 2022c ) , HOSR  ( Liu et al., 2022a ) , SDCRec  ( Du et al., 2022 ) , SoHRML  ( Liu et al., 2022c ) , DESIGN  ( Tao et al., 2022 ) | 

 
 HyperSoRec  ( Wang et al., 2021c ) , CGL  ( Zhang et al., 2022a ) , DISGCN  ( Li et al., 2022a ) , ME-LGN  ( Miao et al., 2022 ) , GDSRec  ( Chen et al., 2022c ) , SGA  ( Liufu and Shen, 2021 ) | 

 
 SIGA  ( Liu et al., 2022d ) , ESRF  ( Yu et al., 2022a ) , Motif-Res  ( Sun et al., 2022 ) , FeSoG  ( Liu et al., 2022e ) , GDMSR  ( Quan et al., 2023 ) , MADM  ( Ma et al., 2024 ) , DSL  ( Wang et al., 2023 ) | 

 
 KG | 
 KConvGraphRec  ( Tien and Van, 2020 ) , HeteroGraphRec  ( Salamat et al., 2021 ) , Social-RippleNet  ( Jiang and Sun, 2022 ) , SCGRec  ( Yang et al., 2022 ) | 

 
 Attributes | 
 GNN-SOR  ( Guo and Wang, 2020 ) , MPSR  ( Liu et al., 2022f ) , FBNE  ( Chen et al., 2022d ) | 

 
 Attributes | 
 Attributes | 
 Diffnet  ( Wu et al., 2019a ) , Diffnet++  ( Wu et al., 2020 ) , DiffnetLG  ( Song et al., 2021 ) , IDiffNet  ( Li et al., 2022c ) , MEGCN  ( Jin et al., 2020 ) , SAN ( Jiang et al., 2021a ) , | 

 
 SRAN  ( Xie et al., 2022 ) , MrAPR  ( Song et al., 2022 ) , SENGR  ( Shi et al., 2022 ) , TAG  ( Qiao et al., 2022 ) , ATGCN  ( Seng et al., 2021 ) , HSGNN  ( Wei et al., 2022a ) | 

 
 Groups | 
 - | 
 IGRec  ( Chen et al., 2022b ) , GLOW  ( Leng and Yu, 2022 ) | 

 
 Attributes | 
 GMAN  ( Liao et al., 2022a ) | 

 
 Multiple | 
 - | 
 Attributes | 
 DH-HGCN  ( Han et al., 2022 ) | 

 
 Attributes | 
 Attributes | 
 BFHAN  ( Zhao et al., 2021 ) | 

 
 Temporal | 
 Homogeneous | 
 - | 
 - | 
 EGFRec  ( Gu et al., 2021 ) , FuseRec  ( Narang et al., 2021 ) , DGARec-R  ( Sun et al., 2020 ) , MGSR  ( Niu et al., 2021 ) , MOHCN  ( Wang and Zhao, 2022 ) , | 

 
 GNNRec  ( Liu et al., 2022b ) , DCAN  ( Wang et al., 2022b ) , SGHAN  ( Wei et al., 2022b ) , SSRGNN  ( Chen et al., 2022a ) , GNN-DSR  ( Lin et al., 2022 ) , | 

 
 DGRec  ( Song et al., 2019 ) , DREAM  ( Song et al., 2020 ) , SeRec  ( Chen and Wong, 2021 ) , TGRec  ( Bai et al., 2020 ) | 

 
 KG | 
 SSDRec  ( Yan et al., 2022 ) | 

 
 Multiple | 
 Homogeneous | 
 - | 
 - | 
 SR-HGNN  ( Xu et al., 2020 ) | 

 

 
 
 

#### 4.1.1. User-Item Rating 

 
 Users interact with different items as they rate them, thus forming the rating matrix 𝐑 ∈ ℝ m × n \mathbf{R}\in\mathbb{R}^{m\times n} . Therefore, each user has a list of items that he/she has interacted with along with the corresponding rating.

 
 
 The timestamp of the user-item interaction may also be available and can be exploited to recommend items to users at specific points in time.
Each rating can thus also be associated with a timestamp for that rating.
Some models exploit the temporal information to make more-effective recommendations in continuous time  ( Bai et al., 2020 ; Sun et al., 2020 ; Narang et al., 2021 ) or during a user session  ( Yan et al., 2022 ; Gu et al., 2021 ; Song et al., 2019 ; Song et al., 2020 ) .

 
 
 Furthermore, one may also have multi-typed user-item interactions. For example, a user may interact with an item positively (positive rating) or negatively (negative rating). Some models have distinguished among these different interaction types to predict each type more effectively  ( Xu et al., 2020 ) .

 
 
 

#### 4.1.2. User-User Social 

 
 The second essential input to SocialRS is the social adjacency matrix 𝐒 ∈ ℝ m × m \mathbf{S}\in\mathbb{R}^{m\times m} , storing user-user social relations.

 
 
 People may be connected to each other via different kinds of social relations. For example, two users may be related if they are friends or if they may
co-comment on an item or if one follows the other, etc. DH-HGCN  ( Han et al., 2022 ) and BFHAN  ( Zhao et al., 2021 ) consider multifaceted, heterogeneous user-user relations in the social network.

 
 
 

#### 4.1.3. Additional Features 

 
 Attributes. 
Both user and items may have additional attributes that can be encoded by the models to make better social recommendations. User attributes are often features of user profiles on social media, e .g., age, sex, etc., while item attributes are often information about the items such as its price and category. Some models just incorporate user attributes  ( Xiao et al., 2021 ; Seng et al., 2021 ) , some only item attributes  ( Guo and Wang, 2020 ; Chen and Wong, 2021 ) , and others incorporate both  ( Wu et al., 2019a ; Wu et al., 2020 ; Song et al., 2021 ; Jin et al., 2020 ; Jiang et al., 2021a ) .

 
 
 Knowledge Graph (KG). 
Items are structured on a product site in the form of a knowledge graph where items are related with each other if they have some mutual dependency. Models incorporate such dependencies between items as represented by this knowledge graph  ( Tien and Van, 2020 ; Salamat et al., 2021 ) .

 
 
 Groups. 
Users are often grouped together denoting a group structure among them. For instance, multiple users can form an online social group based on similar interests or hobbies. Models incorporate the group membership in addition to the social relations to model the social network more effectively  ( Liao et al., 2022a ; Chen et al., 2022b ; Leng and Yu, 2022 ) . User groups can also be formed based on the businesses that they are part of or are clients of, as in   ( Chen et al., 2022d ) .

 
 
 Figure 5. Overview of input representations used by GNN-based SocialRS methods. 
 
 
 Table 4. Taxonomy of input representations. 
 
 
 
 
 Graph Representations | 
 Models | 

 
 
 
 U-U/U-I | 
 GraphRec  ( Fan et al., 2019 ) , DANSER  ( Wu et al., 2019b ) , DICER  ( Fu et al., 2021 ) , ASR  ( Jiang et al., 2021b ) , GNNTSR  ( Mandal and Maiti, 2021 ) , GAT-NSR  ( Mu et al., 2019 ) , | 

 
 SoRecGAT  ( Vijaikumar et al., 2019 ) , SAGLG  ( Liu et al., 2021a ) , PA-GAN  ( Hou et al., 2021 ) , MGNN  ( Xiao et al., 2020 ) , MutualRec  ( Xiao et al., 2021 ) , GHSCF  ( Bi et al., 2021 ) , | 

 
 HIDM  ( Li and Mu, 2020 ) , SocialLGN  ( Liao et al., 2022b ) , SOAP-VAE  ( Walker et al., 2021 ) , GraphRec+  ( Fan et al., 2022 ) , DSR  ( Sha et al., 2021 ) , GTN  ( Hoang et al., 2021 ) , | 

 
 PDARec  ( Zheng et al., 2021 ) , SEPT  ( Yu et al., 2021a ) , DcRec  ( Wu et al., 2022a ) , GSFR  ( Xiao et al., 2022 ) , APTE  ( Zhen et al., 2022 ) , EAGCN  ( Wu et al., 2022c ) , | 

 
 HOSR  ( Liu et al., 2022a ) , SDCRec  ( Du et al., 2022 ) , SoHRML  ( Liu et al., 2022c ) , DESIGN  ( Tao et al., 2022 ) , HyperSoRec  ( Wang et al., 2021c ) , CGL  ( Zhang et al., 2022a ) , | 

 
 DISGCN  ( Li et al., 2022a ) , GDSRec  ( Chen et al., 2022c ) , SGA  ( Liufu and Shen, 2021 ) , SIGA  ( Liu et al., 2022d ) , ESRF  ( Yu et al., 2022a ) , SR-HGNN  ( Xu et al., 2020 ) , | 

 
 EGFRec  ( Gu et al., 2021 ) , FuseRec  ( Narang et al., 2021 ) , DGARec-R  ( Sun et al., 2020 ) , MGSR  ( Niu et al., 2021 ) , GNNRec  ( Liu et al., 2022b ) , DCAN  ( Wang et al., 2022b ) , SGHAN  ( Wei et al., 2022b ) | 

 
 GNN-DSR  ( Lin et al., 2022 ) , DGRec  ( Song et al., 2019 ) , DREAM  ( Song et al., 2020 ) , SeRec  ( Chen and Wong, 2021 ) , TGRec  ( Bai et al., 2020 ) , MADM  ( Ma et al., 2024 ) , DSL  ( Wang et al., 2023 ) | 

 
 U-U-I | 
 SHGCN  ( Zhu et al., 2021 ) , SSRGNN  ( Chen et al., 2022a ) , ME-LGN  ( Miao et al., 2022 ) , SENGR  ( Shi et al., 2022 ) , IGRec  ( Chen et al., 2022b ) , GLOW  ( Leng and Yu, 2022 ) , GDMSR  ( Quan et al., 2023 ) | 

 
 Attributed | 
 DiffNet  ( Wu et al., 2019a ) , DiffNet++  ( Wu et al., 2020 ) , DiffNet-LG  ( Song et al., 2021 ) , IDiffNet  ( Li et al., 2022c ) , MEGCN  ( Jin et al., 2020 ) , SAN  ( Jiang et al., 2021a ) , | 

 
 SRAN  ( Xie et al., 2022 ) , MrAPR  ( Song et al., 2022 ) , SENGR  ( Shi et al., 2022 ) , TAG  ( Qiao et al., 2022 ) , ATGCN  ( Seng et al., 2021 ) , HSGNN  ( Wei et al., 2022a ) , | 

 
 GMAN  ( Liao et al., 2022a ) , GNN-SOR  ( Guo and Wang, 2020 ) , MPSR  ( Liu et al., 2022f ) , FBNE  ( Chen et al., 2022d ) , DH-HGCN  ( Han et al., 2022 ) , BFHAN  ( Zhao et al., 2021 ) | 

 
 Multiplex | 
 DH-HGCN  ( Han et al., 2022 ) , BFHAN  ( Zhao et al., 2021 ) | 

 
 U-U/U-I/I-I | 
 KConvGraph  ( Tien and Van, 2020 ) , HeteroGraphRec  ( Salamat et al., 2021 ) , Social-RippleNet  ( Jiang and Sun, 2022 ) , SCGRec  ( Yang et al., 2022 ) , | 

 
 SSDRec  ( Yan et al., 2022 ) , DGNN  ( Xia et al., 2023 ) | 

 
 Hypergraph | 
 DH-HGCN  ( Han et al., 2022 ) , MHCN  ( Yu et al., 2021b ) , SHGCN  ( Zhu et al., 2021 ) , Motif-Res  ( Sun et al., 2022 ) , MOHCN  ( Wang and Zhao, 2022 ) | 

 
 Decentralized | 
 FeSoG  ( Liu et al., 2022e ) | 

 

 
 
 
 
 

### 4.2. Input Representations: Representation of Inputs within the Models

 
 In order to effectively use the available inputs with GNN-based models, SocialRS methods represent them as different graphs.
In particular, the input representations employed by GNN-based SocialRS can be grouped into 7 categories: U-U/U-I graphs, U-U-I graph, attributed graph, multiplex graph, U-U/U-I/I-I graphs, hypergraph, and decentralized.
Table  4 categorizes papers based on the input representation they develop using the input data.

 
 

#### 4.2.1. U-U/U-I Graphs 

 
 The simplest representation of the input for social recommendation is to use separate graphs for a user-user social network and a user-item interaction network. The user-item interaction network is represented as a bipartite graph and the user-user social network is represented as a general undirected/directed graph. Information from the two graphs is encoded separately at the common user node and later aggregated. Most works follow this representation to encode users and items  ( Fan et al., 2019 ; Wu et al., 2019a ; Fu et al., 2021 ; Wu et al., 2019b ; Fan et al., 2022 ; Xiao et al., 2021 ; Xiao et al., 2020 ; Li and Mu, 2020 ; Wu et al., 2020 ; Song et al., 2021 ; Tao et al., 2022 ; Wu et al., 2022c ) .

 
 
 

#### 4.2.2. U-U-I Graph 

 
 Both kinds of user-user relations and user-item interactions can be modeled together by a single graph as well. Here, user-user edges and user-item edges in the graph need to be distinguished by the type of the end node. Many works thus merge the social relation edges and interaction edges together in a single graph to obtain node embeddings for both users and items  ( Zhu et al., 2021 ; Chen et al., 2022a ; Miao et al., 2022 ; Shi et al., 2022 ) .

 
 
 

#### 4.2.3. Attributed Graph 

 
 Both user and item nodes may further contain features describing the corresponding entity. For example, users may have profile features while items may have their description features. These features are first encoded numerically and then represented explicitly as node attributes in the U-U/U-I graph or U-U-I graph to make effective recommendations  ( Wu et al., 2019a ; Wu et al., 2020 ; Song et al., 2021 ; Xie et al., 2022 ; Jiang et al., 2021a ; Song et al., 2022 ; Seng et al., 2021 ) . These attributes are either fused with the learned embeddings or are used as initialization for the GNN layers.

 
 
 

#### 4.2.4. Multiplex Graph 

 
 Users may be related to each other via multiple relationships while they may also interact with items in multiple ways. Such relationships are often represented using a multiplex network, i .e., using multiple layers of the U-U/U-I graph, where each layer represents a particular relation type  ( Xu et al., 2020 ; Han et al., 2022 ; Zhao et al., 2021 ) .

 
 
 

#### 4.2.5. U-U/U-I/I-I Graphs 

 
 When information on item-item relations is available, an item-item knowledge graph is considered in addition to the U-U and U-I graphs. Item embeddings are now obtained separately one from the U-I interaction graph and the other from the item-item knowledge graph and then aggregated later to obtain the final item embedding   ( Tien and Van, 2020 ; Salamat et al., 2021 ) .

 
 
 

#### 4.2.6. Hypergraph 

 
 One may want to incorporate higher-order relations among users and items to explicitly establish organizational properties in the input such as (1) constructing a user-only hyperedge if a group of users are connected together in closed motifs, (2) constructing a user-item joint hyperedge if a group of users interacts with the same item, and (3) constructing an item-item hyperedge if one user interacts with a group of items. Models have been developed to include just user-item joint hyperedges  ( Zhu et al., 2021 ) , both user-user and user-item joint hyperedges  ( Yu et al., 2021b ) , and user-user and item-item hyperedges  ( Han et al., 2022 ) .

 
 
 

#### 4.2.7. Decentralized 

 
 Centralized data storage is becoming infeasible in practice due to rising privacy concerns. Thus, instead of storing the complete U-U/U-I graphs together, a decentralized storage of the graphs is often required. Here, the edges for social relations and interactions of each user are stored locally at each user’s local server such that only non-sensitive data is shared with the centralized server  ( Liu et al., 2022e ) 

 
 
 
 
 

## 5. Taxonomy of Architectures

 
 In this section, we present the taxonomy of architectures for GNN-based SocialRS.
Model architectures consist of three key components as shown in Figure  6 :
(C1) encoders; (C2) decoders; (C3) loss functions.
Using the U-U and U-I graphs in (C1), the encoders create low-dimensional vectors ( i .e., embeddings) for users and items by employing different GNN encoders.
Here, some works exploit additional information of users and/or items ( e .g., their attributes and groups; refer to Section  4 ) to construct more-accurate user and item embeddings.
In Figure  6 , the dashed lines show which encoders use each additional piece of information.
In (C2), the decoders predict each user’s preference on each item via different operations on the user and item embeddings obtained from (C1).
Finally, in (C3), different loss functions are optimized to learn the embeddings in an end-to-end manner.
We discuss the advantages and disadvantages of each encoder in Table  5 while the loss functions are discussed in Table  6 .
In the subsequent subsections, we describe each component of GNN-based SocialRS in detail.

 
 
 Figure 6. Overview of architectures for GNN-based SocialRS methods.
The solid lines represent the common procedure of the GNN-based SocialRS methods, while the dashed lines represent the flows for the auxiliary inputs or the losses.
 
 
 
 Table 5. High-level comparison of different encoders. 
 
 
 
 
 Encoder | 
 Complexity | 
 Representativeness | 
 Additional Features | 
 Known Issues | 

 
 
 
 GCN | 
 Low | 
 Low | 
 None | 
 Oversmoothing | 

 
 LightGCN | 
 Very Low | 
 Lower | 
 None | 
 Linear representation | 

 
 GANN | 
 High | 
 High | 
 None | 
 Oversmoothing, more parameters | 

 
 HetGNN | 
 High | 
 Higher | 
 Heterogeneous interactions | 
 Extra annotation, Hard to | 

 
 | 
 | 
 | 
 | 
 generalize over interactions | 

 
 GRNN | 
 Very High | 
 Higher | 
 Temporal interactions | 
 Vanishing and exploding gradients | 

 
 HyperGNN | 
 Higher | 
 Higher | 
 High-order relations | 
 Extra motif annotation | 

 
 GAE | 
 Very High | 
 High | 
 Generative | 
 Harder to optimize | 

 
 Hyperbolic GNN | 
 High | 
 Very High | 
 Hierarchical nature | 
 Less empirical evidence | 

 

 
 
 
 Table 6. High-level comparison of different loss functions. 
 
 
 
 
 Loss function | 
 Complexity | 
 Benefits | 
 Known Issues | 

 
 MSE | 
 Low | 
 Learns continuous ratings | 
 Sensitive to outliers | 

 
 BPR | 
 Low | 
 Learns rankings between items | 
 Cannot handle continuous ratings | 

 
 CE | 
 Low | 
 Suited for classification | 
 Cannot handle continuous ratings | 

 
 Hinge | 
 Low | 
 Faster convergence | 
 Cannot handle continuous ratings | 

 
 Social LP | 
 Low | 
 More suitable social embeddings | 
 Multi-objective trade-off | 

 
 SSL | 
 High | 
 More informed representations | 
 Expensive pre-processing | 

 
 Group | 
 Low | 
 Group-level representations | 
 Groups form for specific items | 

 
 | 
 | 
 | 
 (common interests) | 

 
 Adv | 
 High | 
 More robust representations | 
 Adversarial instability and cost | 

 
 Path | 
 High | 
 Predicts social influence propagation | 
 Complicated auxiliary task | 

 
 KD | 
 High | 
 Less overfitting | 
 Training multiple models | 

 
 Sentiment | 
 High | 
 User sentiment-weighed ratings | 
 Complex sentiment classification task | 

 
 Policy Net. | 
 High | 
 Importance weights to each component | 
 Expensive weight allocation | 

 

 
 
 
 Table 7. Taxonomy of encoder architectures. It should be noted that some methods employ non-GNN encoders ( e .g., RNN, MLP, or just an embedding vector (Emb)) or no encoders ( i .e., − - ) to obtain embeddings. 
 
 
 
 
 User Social | 
 User Interest | 
 Item Encoder | 
 Models | 

 
 GANN | 
 GANN | 
 GANN | 
 GraphRec  ( Fan et al., 2019 ) , DiffNet++  ( Wu et al., 2020 ) , DiffNetLG  ( Song et al., 2021 ) , MutualRec  ( Xiao et al., 2021 ) , SR-HGNN  ( Xu et al., 2020 ) , ASR  ( Jiang et al., 2021b ) , | 

 
 GNNTSR  ( Mandal and Maiti, 2021 ) , GAT-NSR  ( Mu et al., 2019 ) , SoRecGAT  ( Vijaikumar et al., 2019 ) , PA-GAN  ( Hou et al., 2021 ) , SOAP-VAE  ( Walker et al., 2021 ) , GTN  ( Hoang et al., 2021 ) , | 

 
 PDARec  ( Zheng et al., 2021 ) , FeSoG  ( Liu et al., 2022e ) , ESRF  ( Yu et al., 2022a ) , SoHRML  ( Liu et al., 2022c ) , FBNE  ( Chen et al., 2022d ) , DISGCN  ( Li et al., 2022a ) , ME-LGN  ( Miao et al., 2022 ) , | 

 
 SGA  ( Liufu and Shen, 2021 ) , GDSRec  ( Chen et al., 2022c ) , DANSER  ( Wu et al., 2019b ) , GraphRec+ ( Fan et al., 2022 ) , SRAN  ( Xie et al., 2022 ) , TAG  ( Qiao et al., 2022 ) , SDCRec  ( Du et al., 2022 ) , | 

 
 KConvGraph  ( Tien and Van, 2020 ) , HeteroGraphRec  ( Salamat et al., 2021 ) , SocialRippleNet  ( Jiang and Sun, 2022 ) , SCGRec  ( Yang et al., 2022 ) , TGRec  ( Bai et al., 2020 ) , | 

 
 | 
 GSFR  ( Xiao et al., 2022 ) , IGRec  ( Chen et al., 2022b ) , DICER  ( Fu et al., 2021 ) | 

 
 Emb | 
 SAN  ( Jiang et al., 2021a ) , HIDM  ( Li and Mu, 2020 ) , GLOW  ( Leng and Yu, 2022 ) , GMAN  ( Liao et al., 2022a ) , HSGNN  ( Wei et al., 2022a ) , MrAPR  ( Song et al., 2022 ) | 

 
 GCN | 
 BFHAN  ( Zhao et al., 2021 ) , SHGCN  ( Zhu et al., 2021 ) | 

 
 GRNN | 
 GRNN | 
 DGARec-R  ( Sun et al., 2020 ) , SSRGNN  ( Chen et al., 2022a ) , GNN-DSR  ( Lin et al., 2022 ) | 

 
 Emb | 
 DREAM  ( Song et al., 2020 ) | 

 
 RNN | 
 GANN | 
 FuseRec  ( Narang et al., 2021 ) | 

 
 RNN | 
 SGHAN  ( Wei et al., 2022b ) | 

 
 Emb | 
 SSDRec  ( Yan et al., 2022 ) | 

 
 - | 
 - | 
 GHSCF  ( Bi et al., 2021 ) | 

 
 GCN | 
 GCN | 
 Emb | 
 DiffNet  ( Wu et al., 2019a ) , MEGCN  ( Jin et al., 2020 ) , MPSR  ( Liu et al., 2022f ) , HOSR  ( Liu et al., 2022a ) , | 

 
 GCN | 
 ATGCN  ( Seng et al., 2021 ) , SENGR  ( Shi et al., 2022 ) , SAGLG  ( Liu et al., 2021a ) , GNN-SOR  ( Guo and Wang, 2020 ) , GDMSR  ( Quan et al., 2023 ) , MADM  ( Ma et al., 2024 ) , DSL  ( Wang et al., 2023 ) | 

 
 GRNN | 
 GRNN | 
 GNNRec  ( Liu et al., 2022b ) , MGSR  ( Niu et al., 2021 ) | 

 
 GCN | 
 EGFRec  ( Gu et al., 2021 ) | 

 
 Emb | 
 DGRec  ( Song et al., 2019 ) | 

 
 RNN | 
 GANN | 
 MOHCN  ( Wang and Zhao, 2022 ) | 

 
 MLP | 
 Emb | 
 MGNN  ( Xiao et al., 2020 ) | 

 
 LightGCN | 
 LightGCN | 
 LightGCN | 
 SocialLGN  ( Liao et al., 2022b ) , DcRec  ( Wu et al., 2022a ) , APTE  ( Zhen et al., 2022 ) , EAGCN  ( Wu et al., 2022c ) , CGL  ( Zhang et al., 2022a ) | 

 
 SEPT  ( Yu et al., 2021a ) , DSR  ( Sha et al., 2021 ) , DESIGN  ( Tao et al., 2022 ) | 

 
 Emb | 
 IDiffNet  ( Li et al., 2022c ) | 

 
 HetGNN | 
 HetGNN | 
 HetGNN | 
 SeRec  ( Chen and Wong, 2021 ) , DGNN  ( Xia et al., 2023 ) | 

 
 GANN | 
 DCAN  ( Wang et al., 2022b ) | 

 
 HyperGNN | 
 GANN | 
 HyperGNN | 
 MHCN  ( Yu et al., 2021b ) , Motif-Res  ( Sun et al., 2022 ) , DH-HGCN  ( Han et al., 2022 ) | 

 
 GAE | 
 GAE | 
 GAE | 
 SIGA  ( Liu et al., 2022d ) | 

 
 Hyperbolic | 
 Hyperbolic | 
 Hyperbolic | 
 HyperSoRec  ( Wang et al., 2021c ) | 

 

 
 
 

### 5.1. Encoders

 
 We group the encoders of GNN-based SocialRS into 8 categories: graph convolutional network (GCN), lightweight GCN (LightGCN), graph attention neural networks (GANN), heterogeneous GNN (HetGNN), graph recurrent neural networks (GRNN), hypergraph neural networks (HyperGNN), graph autoencoder (GAE), and hyperbolic GNN.
Table  7 shows the taxonomy of encoders used in existing work in detail.
Figures  7 ,  8 , and  9 present the conceptual views for distinct types of GNN encoders.

 
 
 Generally, in (C1) encoders,
most methods represent each user p i p_{i} into two types of low-dimensional vectors ( i .e., embeddings) by employing a GNN encoder: p i p_{i} ’s interaction embedding 𝐮 i I \mathbf{u}_{i}^{I} based on a U-I graph and p i p_{i} ’s social embedding 𝐮 i S \mathbf{u}_{i}^{S} based on a U-U graph.
Then, they aggregate them into one embedding 𝐮 i \mathbf{u}_{i} for the corresponding user p i p_{i} .
In the meantime, they also obtain each item q j q_{j} ’s embedding 𝐯 j \mathbf{v}_{j} via another GNN encoder using a U-I graph.
As mentioned above, some works enhance these embeddings by incorporating additional input representations, such as user/item attributes and hypergraphs.

 
 
 It should be noted that some works employ only a single GNN encoder to obtain the two embeddings.
In contrast, others use different GNN encoders for the embeddings of different node types ( i .e., users or items).
For simplicity, however, we here explain the GNN encoders by generalizing them to any node type in the input graph.

 
 

#### 5.1.1. GCN 

 
 Early works  ( Wu et al., 2019a ; Jin et al., 2020 ; Liu et al., 2022f ; Liu et al., 2022a ; Zhu et al., 2022 ; Seng et al., 2021 ; Shi et al., 2022 ; Liu et al., 2021a ; Guo and Wang, 2020 ; Xiao et al., 2020 ) have focused on representing the user and item embeddings using GCN.
Given a node n i n_{i} ( i .e., a user or an item) in the input graph ( i .e., U-I or U-U graphs), a n i n_{i} ’s embedding 𝐞 i k \mathbf{e}_{i}^{k} in k k -th layer is represented based on the embeddings of n i n_{i} ’s neighbors in ( k − 1 ) (k-1) -th layer as follows:

 

 
 (1) | 
 | 
 𝐞 i ( k ) = σ ⁡ ( ∑ n j ∈ 𝒩 n i 1 | 𝒩 n j | ​ | 𝒩 n i | ​ 𝐞 j ( k − 1 ) ​ 𝐖 ( k ) ) , \mathbf{e}_{i}^{(k)}=\sigma(\sum\nolimits_{n_{j}\in\mathcal{N}_{n_{i}}}\frac{1}{\sqrt{|\mathcal{N}_{n_{j}}||\mathcal{N}_{n_{i}}|}}\mathbf{e}_{j}^{(k-1)}\mathbf{W}^{(k)}), | 
 | 
 

 where σ \sigma and 𝐖 ( k ) ∈ ℝ d × d \mathbf{W}^{(k)}\in\mathbb{R}^{d\times d} denote a non-linear activation function ( e .g., ReLU) and a trainable transformation matrix, respectively. Also, 𝒩 n i \mathcal{N}_{n_{i}} indicates a set of n i n_{i} ’s neighbors in the input graph.
Here, some works take the self-connection of n i n_{i} into consideration by aggregating over the set 𝒩 n i ∪ { n i } \mathcal{N}_{n_{i}}\cup\{n_{i}\} .
Most methods simply consider the n i n_{i} ’s embedding in the last K K -th layer, 𝐞 i ( K ) \mathbf{e}_{i}^{(K)} , as its final embedding 𝐳 i \mathbf{z}_{i} .
Another variant is to aggregate n i n_{i} ’s embeddings from all layers, i .e., 𝐳 i = ∑ k = 1 K 𝐞 i ( k ) \mathbf{z}_{i}=\sum^{K}_{k=1}\mathbf{e}_{i}^{(k)} .
For instance, DiffNet  ( Wu et al., 2019a ) obtains a user p i p_{i} ’s social embedding 𝐮 i S \mathbf{u}_{i}^{S} (resp. interaction embedding 𝐮 i I \mathbf{u}_{i}^{I} ) by performing GCN with k k -layers (resp. 1 1 -layer) based on the U-U graph (resp. U-I graph).
For each item q j q_{j} , it simply obtains q j q_{j} ’s embedding 𝐯 j \mathbf{v}_{j} based on its attributes without using a GNN encoder.

 
 
 Note that different normalization schemes have been proposed in the literature to normalize the weight of each neighbor. The most common strategy is the symmetric normalization as 1 / | 𝒩 n j | ​ | 𝒩 n i | 1/\sqrt{|\mathcal{N}_{n_{j}}||\mathcal{N}_{n_{i}}|} for its simpler symmetric matrix form. One can also just use 1 / | 𝒩 n j | 1/|\mathcal{N}_{n_{j}}| but it gives a lower weight to high-degree nodes as compared to the previous form, which may not be desirable. Finally, just using 1 / | 𝒩 n i | 1/|\mathcal{N}_{n_{i}}| is also not typically desirable as it smooths the neighbor information without considering their degrees. We will thus use symmetric normalization unless otherwise mentioned.

 
 
 

#### 5.1.2. LightGCN 

 
 It is well-known that non-linear activation and feature transformation in GCN encoders make the propagation step very complicated for training and scalability  ( He et al., 2020 ; Mao et al., 2021 ) .
Motivated by this, some works  ( Liao et al., 2022b ; Wu et al., 2022a ; Zhen et al., 2022 ; Wu et al., 2022c ; Zhang et al., 2022a ; Yu et al., 2021a ; Sha et al., 2021 ; Tao et al., 2022 ; Li et al., 2022c ) have attempted to replace their GCN encoders with lightweight GCN  ( He et al., 2020 ) , i .e.,

 

 
 (2) | 
 | 
 𝐞 i ( k ) = ∑ n j ∈ 𝒩 n i 1 | 𝒩 n j | ​ | 𝒩 n i | ​ 𝐞 j ( k − 1 ) . \mathbf{e}_{i}^{(k)}=\sum\nolimits_{n_{j}\in\mathcal{N}_{n_{i}}}\frac{1}{\sqrt{|\mathcal{N}_{n_{j}}||\mathcal{N}_{n_{i}}|}}\mathbf{e}_{j}^{(k-1)}. | 
 | 
 

 It should be noted that LightGCN  ( He et al., 2020 ) has no non-linear activation function, no feature transformation, and no self-connection.

 
 
 For instance, DcRec  ( Wu et al., 2022a ) obtains each user p i p_{i} ’s social embedding 𝐮 i S \mathbf{u}_{i}^{S} via GCN, whereas obtaining the p i p_{i} ’s interaction embedding 𝐮 i I \mathbf{u}_{i}^{I} and each item q j q_{j} ’s embedding 𝐯 j \mathbf{v}_{j} via the LightGCN encoder.

 
 
 Figure 7. Conceptual views of the GCN, LightGCN, GANN, and HetGNN encoders with a single GNN layer. The left side represents user embeddings, while the right side represents item embeddings. 
 
 
 

#### 5.1.3. GANN 

 
 The attention mechanism in graphs originated from the graph attention network (GAT)  ( Veličković et al., 2018 ) and has already been successful in many applications, including recommender systems.
Considering different weights from neighbor nodes in the input graph helps focus on important adjacent nodes while filtering out noises during the propagation process  ( Veličković et al., 2018 ) .
Therefore, almost all existing works on SocialRS have leveraged the attention mechanism in their GNN encoders  ( Fan et al., 2019 ; Wu et al., 2020 ; Song et al., 2021 ; Xiao et al., 2021 ; Xu et al., 2020 ; Jiang et al., 2021b ; Mandal and Maiti, 2021 ; Mu et al., 2019 ; Vijaikumar et al., 2019 ; Hou et al., 2021 ; Walker et al., 2021 ; Hoang et al., 2021 ; Zheng et al., 2021 ; Liu et al., 2022e ; Yu et al., 2022a ; Liu et al., 2022c ; Chen et al., 2022d ; Li et al., 2022a ; Miao et al., 2022 ; Liufu and Shen, 2021 ; Chen et al., 2022c ; Wu et al., 2019b ; Fan et al., 2022 ; Xie et al., 2022 ; Qiao et al., 2022 ; Du et al., 2022 ; Tien and Van, 2020 ; Salamat et al., 2021 ; Jiang and Sun, 2022 ; Bai et al., 2020 ; Yang et al., 2022 ; Xiao et al., 2022 ; Chen et al., 2022b ; Fu et al., 2021 ; Chen et al., 2022b ; Jiang et al., 2021a ; Li and Mu, 2020 ; Leng and Yu, 2022 ; Liao et al., 2022a ; Wei et al., 2022a ; Song et al., 2022 ; Zhao et al., 2021 ; Zhu et al., 2021 ; Sun et al., 2020 ; Chen et al., 2022a ; Lin et al., 2022 ; Song et al., 2020 ; Narang et al., 2021 ; Wei et al., 2022b ; Yan et al., 2022 ; Bi et al., 2021 ) .

 
 
 The common intuitions behind their design of the attention mechanism are:
(1) each user’s preferences for different items may differ, and
(2) each user’s influences on her social friends may differ.
Based on such intuitions, many methods represent a node n i n_{i} ’s embedding in k k -th layer by attentively aggregating the embeddings of n i n_{i} ’s neighbors in ( k − 1 ) (k-1) -th layer as follows:

 

 
 (3) | 
 | 
 𝐞 i ( k ) = σ ⁡ ( ∑ n j ∈ 𝒩 n i ( α i ​ j ⋅ 𝐞 j ( k − 1 ) ) ​ 𝐖 ( k ) ) , \mathbf{e}_{i}^{(k)}=\sigma(\sum\nolimits_{n_{j}\in\mathcal{N}_{n_{i}}}(\alpha_{ij}\cdot\mathbf{e}_{j}^{(k-1)})\mathbf{W}^{(k)}), | 
 | 
 

 where α i ​ j \alpha_{ij} indicates the attention weight of neighbor node n j n_{j} w.r.t n i n_{i} .

 
 
 Now, we discuss how to compute the attention weights in existing works.
Most methods, including DANSER  ( Wu et al., 2019b ) and SCGRec  ( Yang et al., 2022 ) , typically use the concatenation-based graph attention as follows:

 

 
 (4) | 
 | 
 α i ​ j = exp ⁡ ( MLP ​ [ 𝐞 i , 𝐞 j ] ) ∑ n k ∈ 𝒩 n i exp ⁡ ( MLP ​ [ 𝐞 i , 𝐞 j ] ) . \alpha_{ij}=\frac{\exp(\text{MLP}[\mathbf{e}_{i},\mathbf{e}_{j}])}{\sum\nolimits_{n_{k}\in\mathcal{N}_{n_{i}}}\exp(\text{MLP}[\mathbf{e}_{i},\mathbf{e}_{j}])}. | 
 | 
 

 
 
 Also, other methods, including DICER  ( Wu et al., 2019b ) and MEGCN  ( Jin et al., 2020 ) , use the similarity-based graph attention, which is another popular technique, i .e.,

 

 
 (5) | 
 | 
 α i ​ j = exp ⁡ ( sim ​ ( 𝐞 i , 𝐞 j ) ) ∑ n k ∈ 𝒩 n i exp ⁡ ( sim ​ ( 𝐞 i , 𝐞 k ) ) , \alpha_{ij}=\frac{\exp(\text{sim}(\mathbf{e}_{i},\mathbf{e}_{j}))}{\sum\nolimits_{n_{k}\in\mathcal{N}_{n_{i}}}\exp(\text{sim}(\mathbf{e}_{i},\mathbf{e}_{k}))}, | 
 | 
 

 where sim() denotes a similarity function such as cosine similarity and dot product.

 
 
 

#### 5.1.4. HetGNN 

 
 The user-item interactions and user-user social relations can be regarded as the users’ heterogeneous relationships, i .e., a user’s preferences on items and his/her friendship.
In this sense, a few methods  ( Chen and Wong, 2021 ; Wang et al., 2022b ) have attempted to model the inputs as a heterogeneous graph and then design the HetGNN encoders for learning user and item embeddings, i .e.,

 

 
 (6) | 
 | 
 𝐞 i ( k ) = σ ⁡ ( ∑ n j ∈ 𝒩 n i 1 | 𝒩 n j | ​ | 𝒩 n i | ​ 𝐞 j ( k − 1 ) ​ 𝐖 v i ​ j ( k ) ) , \mathbf{e}_{i}^{(k)}=\sigma(\sum\nolimits_{n_{j}\in\mathcal{N}_{n_{i}}}\frac{1}{\sqrt{|\mathcal{N}_{n_{j}}||\mathcal{N}_{n_{i}}|}}\mathbf{e}_{j}^{(k-1)}\mathbf{W}_{v_{ij}}^{(k)}), | 
 | 
 

 where v i ​ j v_{ij} indicates the type of relation between n i n_{i} and n j n_{j} .
As a result, the HetGNN encoder employs different transformation matrices according to the relations between two nodes.

 
 
 For instance, SeRec  ( Chen and Wong, 2021 ) defines four types of directed edges ( i .e., user-user edges, user-item edges, item-user edges, and item-item edges), constructing a heterogeneous graph based on the above edges.
Then, it obtains each user p i p_{i} ’s embedding 𝐮 i \mathbf{u}_{i} and each item q j q_{j} ’s embedding 𝐯 j \mathbf{v}_{j} via the HetGNN encoder.

 
 
 

#### 5.1.5. GRNN 

 
 The sequential behaviors of users when they interact with items reflect the evolution of their preferences of items over time.
For this reason, time-aware recommender systems have attracted increasing attention in recent years  ( Wang et al., 2021a ) .
Such temporal interactions are often divided into multiple user sessions and modeled as session-based SocialRS.
Multiple works  ( Sun et al., 2020 ; Chen et al., 2022a ; Lin et al., 2022 ; Liu et al., 2022b ; Niu et al., 2021 ; Gu et al., 2021 ; Song et al., 2019 ; Wang and Zhao, 2022 ; Xiao et al., 2020 ) have attempted to model dynamic user interests through session-based or temporal SocialRS.
These models leverage the GRNN encoders to capture these time-evolving interests.

 
 
 Figure 8. Conceptual view of the GRNN encoder with a single GNN layer. Note that existing work did not use GRNN encoders to obtain user social embeddings. 
 
 
 Suppose each user p p interacts with items in a given sequence 𝒮 p \mathcal{S}_{p} . Consequently, one can create a sequence of interactions for each item q q as 𝒮 q \mathcal{S}_{q} , consisting of users that rate item q q in a temporal sequence. In general,
the temporal sequence is denoted for node n i n_{i} as 𝒮 n i = { n 1 i , n 2 i , ⋯ , n K i } \mathcal{S}_{n_{i}}=\{n^{i}_{1},n^{i}_{2},\cdots,n^{i}_{K}\} .
Note that session-based encoders would divide 𝒮 n i \mathcal{S}_{n_{i}} into multiple sessions 𝒮 n i t \mathcal{S}^{t}_{n_{i}} and encode each session separately. The GRNN encoder for node n i n_{i} can be then generalized as:

 

 
 (7) | 
 | 
 𝐞 i = GRNN ​ ( 𝒮 n i , 𝒩 n i ) , \mathbf{e}_{i}=\textsc{GRNN}(\mathcal{S}_{n_{i}},\mathcal{N}_{n_{i}}), | 
 | 
 

 where GRNN is a combination of RNN and GNN modules. In particular, one can obtain dynamic user interests and item embeddings through a long short-term memory (LSTM)  ( Peng et al., 2017 ; Zayats and Ostendorf, 2018 ) unit, i .e.,

 
 
 
 (8) | 
 | 
 𝐱 i ( k ) \displaystyle\mathbf{x}_{i}^{(k)} | 
 = σ ⁡ ( 𝐖 x ​ [ 𝐡 i ( k − 1 ) , 𝐧 k i ] + b x ) , \displaystyle=\sigma(\mathbf{W}_{x}[\mathbf{h}_{i}^{(k-1)},\mathbf{n}_{k}^{i}]+b_{x}), | 
 | 

 
 | 
 𝐟 i ( k ) \displaystyle\mathbf{f}_{i}^{(k)} | 
 = σ ⁡ ( 𝐖 s ​ [ 𝐡 i ( k − 1 ) , 𝐧 k i ] + b s ) , \displaystyle=\sigma(\mathbf{W}_{s}[\mathbf{h}_{i}^{(k-1)},\mathbf{n}_{k}^{i}]+b_{s}), | 
 | 

 
 | 
 𝐨 i ( k ) \displaystyle\mathbf{o}_{i}^{(k)} | 
 = σ ⁡ ( 𝐖 o ​ [ 𝐡 i ( k − 1 ) , 𝐧 k i ] + b o ) , \displaystyle=\sigma(\mathbf{W}_{o}[\mathbf{h}_{i}^{(k-1)},\mathbf{n}_{k}^{i}]+b_{o}), | 
 | 

 
 | 
 𝐜 ~ i ( k ) \displaystyle\mathbf{\tilde{c}}_{i}^{(k)} | 
 = tanh ⁡ ( 𝐖 c ​ [ 𝐡 j ( k − 1 ) , 𝐧 k i ] + b c ) , \displaystyle=\tanh(\mathbf{W}_{c}[\mathbf{h}_{j}^{(k-1)},\mathbf{n}_{k}^{i}]+b_{c}), | 
 | 

 
 | 
 𝐜 i ( k ) \displaystyle\mathbf{c}_{i}^{(k)} | 
 = 𝐟 i ( k ) ⊙ 𝐜 i ( k − 1 ) + 𝐱 i ( k ) ⊙ 𝐜 ~ i ( k ) , \displaystyle=\mathbf{f}_{i}^{(k)}\odot\mathbf{c}_{i}^{(k-1)}+\mathbf{x}_{i}^{(k)}\odot\mathbf{\tilde{c}}_{i}^{(k)}, | 
 | 

 
 | 
 𝐡 i ( k ) \displaystyle\mathbf{h}_{i}^{(k)} | 
 = 𝐨 i ( k ) ⊙ tanh ⁡ ( 𝐜 i ( k ) ) . \displaystyle=\mathbf{o}_{i}^{(k)}\odot\tanh(\mathbf{c}_{i}^{(k)}). | 
 | 
 

 
 
 Then, the node embedding 𝐞 i \mathbf{e}_{i} is obtained using a GNN module such as GANN and GCN (as discussed below). In general, one can obtain

 

 
 (9) | 
 | 
 𝐞 i = GNN ​ ( 𝐡 i ( K ) , { 𝐡 j ( K ) : n j ∈ 𝒩 n i ∪ { n i } } ) . \mathbf{e}_{i}=\textsc{GNN}(\mathbf{h}_{i}^{(K)},\{\mathbf{h}_{j}^{(K)}:n_{j}\in\mathcal{N}_{n_{i}}\cup\{n_{i}\}\}). | 
 | 
 

 
 
 For instance, DREAM  ( Song et al., 2020 ) obtains each user p i p_{i} ’s embedding within each session using the GRNN encoder as above. It uses a Relational GAT module for the GNN layer to aggregate information from its social neighbors. Meanwhile, item embeddings 𝐯 j \mathbf{v}_{j} for item q j q_{j} is obtained using a simple embedding layer.

 
 
 Figure 9. Conceptual view of the HyperGNN encoder with a single GNN layer. Note that existing work did not distinguish between user embeddings as interaction and social embeddings, nor did it utilize HyperGNN encoders to generate item embeddings. 
 
 
 

#### 5.1.6. HyperGNN 

 
 Most GNN encoders, as mentioned above, learn pairwise connectivity between two nodes.
However, more-complicated connections can be captured by
jointly using user-item relations with user-user edges and/or using higher-order social relations.
For instance, triangular structures, including two users and their co-rated items, are a common motif.
To leverage such high-order relations, some works  ( Yu et al., 2021b ; Sun et al., 2022 ; Han et al., 2022 ) have attempted to model the inputs as a hypergraph and then design the HyperGNN encoders for learning user and item embeddings.

 
 
 Let 𝒢 = ( 𝒩 , ℰ ) \mathcal{G}=(\mathcal{N},\mathcal{E}) denotes a hypergraph where 𝒩 \mathcal{N} and ℰ \mathcal{E} indicate sets of nodes and hyperedges, respectively.
Each hyperedge e ∈ ℰ e\in\mathcal{E} is a subset of nodes, i.e. , e ∈ 2 𝒩 e\in 2^{\mathcal{N}} . The node degree is thus d i = ∑ e ∈ ℰ : n i ∈ e 1 d_{i}=\sum_{e\in\mathcal{E}:n_{i}\in e}{1} for ∀ n i ∈ 𝒩 \forall n_{i}\in\mathcal{N} . Then, each layer of the HyperGNN encoder learns node embeddings using relations as

 
 
 

 
 (10) | 
 | 
 𝐡 i ( k ) = σ ( ∑ e ∈ ℰ : n i ∈ e ∑ n j ∈ e w e , i , j d i ​ d j ​ | e | 𝐡 j ( k − 1 ) ) , \mathbf{h}_{i}^{(k)}=\sigma(\sum\nolimits_{e\in\mathcal{E}:n_{i}\in e}\sum\nolimits_{n_{j}\in e}\frac{w_{e,i,j}}{\sqrt{d_{i}d_{j}}|e|}\mathbf{h}_{j}^{(k-1)}), | 
 | 
 

 where w e , i , j w_{e,i,j} are learnable parameters and d i = ∑ e ∈ ℰ : n i ∈ e 1 d_{i}=\sum_{e\in\mathcal{E}:n_{i}\in e}{1} . We note that HyperGNN-based SocialRS methods  ( Yu et al., 2021b ; Han et al., 2022 ) remove non-linear activation and feature transformation as in the LightGCN encoder.

 
 
 For instance, MHCN  ( Yu et al., 2021b ) designs three types of triangular motifs, constructing three incidence matrices, each representing a hypergraph induced by each motif.
Then, it obtains each user p i p_{i} ’s embedding 𝐮 i \mathbf{u}_{i} via the multi-type HyperGNN encoders while obtaining each item q j q_{j} ’s embedding 𝐯 j \mathbf{v}_{j} via the GCN encoder.

 
 
 

#### 5.1.7. Others 

 
 Furthermore, we briefly describe the two encoders, GAE and hyperbolic GNN, each of which is employed by only one method.
Liu et al.  ( Liu et al., 2022d ) pointed out that GCN is mainly suitable for semi-supervised learning tasks.
On the other hand, they claimed that the goal of GAE coincides with that of the recommendation task, which is to minimize the reconstruction error of input and output  ( Liu et al., 2022d ) .
For this reason, they proposed a SocialRS method, named SIGA, which employs GAE and is used for the rating prediction task.

 
 
 Meanwhile, Wang et al.  ( Wang et al., 2021c ) pointed out that since existing methods usually learn the user and item embeddings in the Euclidean space, these methods fail to explore the latent hierarchical property in the data.
For this reason, they proposed a SocialRS method, named HyperSoRec, which performs in the hyperbolic space because the exponential expansion of hyperbolic space helps preserve more-complex relationships between users and items  ( Krioukov et al., 2010 ) .

 
 
 Table 8. Taxonomy of decoder architectures 
 
 
 
 
 Decoders | 
 Models | 

 
 
 
 Dot-product | 
 DiffNet  ( Wu et al., 2019a ) , DiffNet++  ( Wu et al., 2020 ) , DiffNetLG  ( Song et al., 2021 ) , MEGCN  ( Jin et al., 2020 ) , ASR  ( Jiang et al., 2021b ) , ATGCN  ( Seng et al., 2021 ) , GNN-SOR  ( Guo and Wang, 2020 ) , | 

 
 DGARec  ( Sun et al., 2020 ) , SAGLG  ( Liu et al., 2021a ) , HIDM  ( Li and Mu, 2020 ) , SocialLGN  ( Liao et al., 2022b ) , GMAN  ( Liao et al., 2022a ) , MPSR  ( Liu et al., 2022f ) , DREAM  ( Song et al., 2020 ) , | 

 
 SHGCN  ( Zhu et al., 2021 ) , SCGRec  ( Yang et al., 2022 ) , PDARec  ( Zheng et al., 2021 ) , MHCN  ( Yu et al., 2021b ) , SEPT  ( Yu et al., 2021a ) , DcRec  ( Wu et al., 2022a ) , | 

 
 DH-HGCN  ( Han et al., 2022 ) , SSDRec  ( Yan et al., 2022 ) , SeRec  ( Chen and Wong, 2021 ) , Motif-Res  ( Sun et al., 2022 ) , APTE  ( Zhen et al., 2022 ) , EAGCN  ( Wu et al., 2022c ) , | 

 
 HOSR  ( Liu et al., 2022a ) , FeSoG  ( Liu et al., 2022e ) , IGRec  ( Chen et al., 2022b ) , ESRF  ( Yu et al., 2022a ) , SDCRec  ( Du et al., 2022 ) , SoHRML  ( Liu et al., 2022c ) , HyperSoRec  ( Wang et al., 2021c ) , DSR  ( Sha et al., 2021 ) | 

 
 DESIGN  ( Tao et al., 2022 ) , SRAN  ( Xie et al., 2022 ) , IDiffNet  ( Li et al., 2022c ) , CGL  ( Zhang et al., 2022a ) , FBNE  ( Chen et al., 2022d ) , MrAPR  ( Song et al., 2022 ) , GNNRec  ( Liu et al., 2022b ) , SGHAN  ( Wei et al., 2022b ) | 

 
 SSRGNN  ( Chen et al., 2022a ) , DISGCN  ( Li et al., 2022a ) , ME-LGN  ( Miao et al., 2022 ) , SGA  ( Liufu and Shen, 2021 ) , SIGA  ( Liu et al., 2022d ) , GDMSR  ( Quan et al., 2023 ) , DGNN  ( Xia et al., 2023 ) , DSL  ( Wang et al., 2023 ) | 

 
 MLP | 
 GraphRec  ( Fan et al., 2019 ) , GraphRec+  ( Fan et al., 2022 ) , DICER  ( Fu et al., 2021 ) , SAN  ( Jiang et al., 2021a ) , KConvGraphRec  ( Tien and Van, 2020 ) , EGFRec  ( Gu et al., 2021 ) , | 

 
 FuseRec  ( Narang et al., 2021 ) , GNNTSR  ( Mandal and Maiti, 2021 ) , GAT-NSR  ( Mu et al., 2019 ) , TGRec  ( Bai et al., 2020 ) , SoRecGAT  ( Vijaikumar et al., 2019 ) , PA-GAN  ( Hou et al., 2021 ) , | 

 
 GHSCF  ( Bi et al., 2021 ) , HeteroGraphRec  ( Salamat et al., 2021 ) , GLOW  ( Leng and Yu, 2022 ) , BFHAN  ( Zhao et al., 2021 ) , MGSR  ( Niu et al., 2021 ) , GTN  ( Hoang et al., 2021 ) , | 

 
 MGNN  ( Xiao et al., 2020 ) , SR-HGNN  ( Xu et al., 2020 ) , DANSER  ( Wu et al., 2019b ) , MutualRec  ( Xiao et al., 2021 ) , GSFR  ( Xiao et al., 2022 ) , SOAP-VAE  ( Walker et al., 2021 ) , | 

 
 SENGR  ( Shi et al., 2022 ) , MOHCN  ( Wang and Zhao, 2022 ) , DCAN  ( Wang et al., 2022b ) , TAG  ( Qiao et al., 2022 ) , HSGNN  ( Wei et al., 2022a ) , Social-RippleNet  ( Jiang and Sun, 2022 ) , | 

 
 GNN-DSR  ( Lin et al., 2022 ) , GDSRec  ( Chen et al., 2022c ) , DGRec  ( Song et al., 2019 ) , DREAM  ( Song et al., 2020 ) | 

 

 
 
 
 
 

### 5.2. Decoders

 
 In this subsection, we group the decoders of GNN-based SocialRS into two categories: dot-product and multi-layer perceptron (MLP).
Table  8 summarizes the taxonomy of these decoders.

 
 
 Table 9. Taxonomy of loss functions 
 
 
 
 
 Loss Functions | 
 Models | 

 
 
 
 
 Primary 
 
 Objectives 
 | 
 MSE | 
 GraphRec  ( Fan et al., 2019 ) , GNNTSR  ( Mandal and Maiti, 2021 ) , GAT-NSR  ( Mu et al., 2019 ) , TGRec  ( Bai et al., 2020 ) , PA-GAN  ( Hou et al., 2021 ) , GHSCF  ( Bi et al., 2021 ) , | 

 
 GraphRec+  ( Fan et al., 2022 ) , GTN  ( Hoang et al., 2021 ) , PDARec  ( Zheng et al., 2021 ) ,
GNN-SOR  ( Guo and Wang, 2020 ) , DANSER  ( Wu et al., 2019b ) , SAN  ( Jiang et al., 2021a ) , | 

 
 KConvGraphRec  ( Tien and Van, 2020 ) , HeteroGraphRec  ( Salamat et al., 2021 ) , GMAN  ( Liao et al., 2022a ) , SR-HGNN  ( Xu et al., 2020 ) , DGARec-R  ( Sun et al., 2020 ) , | 

 
 MGSR  ( Niu et al., 2021 ) , APTE  ( Zhen et al., 2022 ) , EAGCN  ( Wu et al., 2022c ) , FeSoG  ( Liu et al., 2022e ) , SENGR  ( Shi et al., 2022 ) , MOHCN  ( Wang and Zhao, 2022 ) , TAG  ( Qiao et al., 2022 ) , | 

 
 Social-RippleNet  ( Jiang and Sun, 2022 ) , GNN-DSR  ( Lin et al., 2022 ) , GDSRec  ( Chen et al., 2022c ) | 

 
 BPR | 
 ASR  ( Jiang et al., 2021b ) , SAGLG  ( Liu et al., 2021a ) , MGNN  ( Xiao et al., 2020 ) , HIDM  ( Li and Mu, 2020 ) , SocialLGN  ( Liao et al., 2022b ) , MPSR  ( Liu et al., 2022f ) , SHGCN  ( Zhu et al., 2021 ) , | 

 
 MutualRec  ( Xiao et al., 2021 ) , ATGCN  ( Seng et al., 2021 ) , DiffNet  ( Wu et al., 2019a ) , DiffNet++  ( Wu et al., 2020 ) , DiffNetLG  ( Song et al., 2021 ) , MEGCN  ( Jin et al., 2020 ) , | 

 
 GLOW  ( Leng and Yu, 2022 ) , SCGRec  ( Yang et al., 2022 ) , SEPT  ( Yu et al., 2021a ) , DcRec  ( Wu et al., 2022a ) , MHCN  ( Yu et al., 2021b ) , DH-HGCN  ( Han et al., 2022 ) , GSFR  ( Xiao et al., 2022 ) , HOSR  ( Liu et al., 2022a ) | 

 
 IGRec  ( Chen et al., 2022b ) , ESRF  ( Yu et al., 2022a ) , SoHRML  ( Liu et al., 2022c ) , DSR  ( Sha et al., 2021 ) , SRAN  ( Xie et al., 2022 ) , IDiffNet  ( Li et al., 2022c ) , CGL  ( Zhang et al., 2022a ) , MrAPR  ( Song et al., 2022 ) , DISGCN  ( Li et al., 2022a ) | 

 
 HSGNN  ( Wei et al., 2022a ) , ME-LGN  ( Miao et al., 2022 ) , SGA  ( Liufu and Shen, 2021 ) , Motif-Res  ( Sun et al., 2022 ) , GDMSR  ( Quan et al., 2023 ) , MADM  ( Ma et al., 2024 ) , DGNN  ( Xia et al., 2023 ) , DSL  ( Wang et al., 2023 ) | 

 
 CE | 
 DICER  ( Fu et al., 2021 ) , SoRecGAT  ( Vijaikumar et al., 2019 ) , DANSER  ( Wu et al., 2019b ) , BFHAN  ( Zhao et al., 2021 ) , EGFRec  ( Gu et al., 2021 ) , FuseRec  ( Narang et al., 2021 ) , | 

 
 SeRec  ( Chen and Wong, 2021 ) , SSDRec  ( Yan et al., 2022 ) ,
DESIGN  ( Tao et al., 2022 ) , FBNE  ( Chen et al., 2022d ) , GNNRec  ( Liu et al., 2022b ) , DCAN  ( Wang et al., 2022b ) , SOAP-VAE  ( Walker et al., 2021 ) | 

 
 SGHAN  ( Wei et al., 2022b ) , SSRGNN  ( Chen et al., 2022a ) , GDSRec  ( Chen et al., 2022c ) ,
SIGA  ( Liu et al., 2022d ) , DGRec  ( Song et al., 2019 ) , DREAM  ( Song et al., 2020 ) | 

 
 Hinge | 
 HyperSoRec  ( Wang et al., 2021c ) | 

 
 
 
 
 Auxiliary 
 
 Objectives 
 | 
 Social LP | 
 MGNN  ( Xiao et al., 2020 ) , MutualRec  ( Xiao et al., 2021 ) , SR-HGNN  ( Xu et al., 2020 ) , APTE  ( Zhen et al., 2022 ) , SoHRML  ( Liu et al., 2022c ) , FBNE  ( Chen et al., 2022d ) , GDMSR  ( Quan et al., 2023 ) | 

 
 SSL | 
 Social | 
 SEPT  ( Yu et al., 2021a ) , DcRec  ( Wu et al., 2022a ) , MADM  ( Ma et al., 2024 ) , DSL  ( Wang et al., 2023 ) | 

 
 Interaction | 
 SDCRec  ( Du et al., 2022 ) , CGL  ( Zhang et al., 2022a ) , DISGCN  ( Li et al., 2022a ) , DCAN  ( Wang et al., 2022b ) , DcRec  ( Wu et al., 2022a ) | 

 
 Motif | 
 Motif-Res  ( Sun et al., 2022 ) , MHCN  ( Yu et al., 2021b ) | 

 
 Group | 
 GLOW  ( Leng and Yu, 2022 ) , GMAN  ( Liao et al., 2022a ) | 

 
 Adv | 
 ESRF  ( Yu et al., 2022a ) | 

 
 Path | 
 SPEX  ( Li et al., 2021 ) | 

 
 KD | 
 DESIGN  ( Tao et al., 2022 ) | 

 
 Sentiment | 
 SENGR  ( Shi et al., 2022 ) | 

 
 Policy Net. | 
 DANSER  ( Wu et al., 2019b ) | 

 

 
 
 

#### 5.2.1. Dot-product 

 
 Many methods  ( Wu et al., 2019a ; Wu et al., 2020 ; Song et al., 2021 ; Jin et al., 2020 ; Jiang et al., 2021b ; Seng et al., 2021 ; Guo and Wang, 2020 ; Sun et al., 2020 ; Liu et al., 2021a ; Li and Mu, 2020 ; Liao et al., 2022b ; Liao et al., 2022a ; Liu et al., 2022f ; Song et al., 2020 ; Zhu et al., 2021 ; Yang et al., 2022 ; Zheng et al., 2021 ; Yu et al., 2021b ; Yu et al., 2021a ; Wu et al., 2022a ; Han et al., 2022 ; Yan et al., 2022 ; Chen and Wong, 2021 ; Sun et al., 2022 ; Zhen et al., 2022 ; Wu et al., 2022c ; Liu et al., 2022a ; Liu et al., 2022e ; Chen et al., 2022b ; Yu et al., 2022a ; Zhu et al., 2022 ; Du et al., 2022 ; Liu et al., 2022c ; Wang et al., 2021c ; Sha et al., 2021 ; Tao et al., 2022 ; Xie et al., 2022 ; Li et al., 2022c ; Zhang et al., 2022a ; Chen et al., 2022d ; Song et al., 2022 ; Liu et al., 2022b ; Wei et al., 2022b ; Chen et al., 2022a ; Li et al., 2022a ; Miao et al., 2022 ; Liufu and Shen, 2021 ; Liu et al., 2022d ) simply predict a user p i p_{i} ’s preference r ^ i ​ j \hat{r}_{ij} on an item q j q_{j} via a dot product of their corresponding embeddings, i .e.,

 

 
 (11) | 
 | 
 r ^ i ​ j = 𝐮 i ⋅ 𝐯 j ⊤ . \hat{r}_{ij}=\mathbf{u}_{i}\cdot\mathbf{v}_{j}^{\top}. | 
 | 
 

 
 
 

#### 5.2.2. MLP 

 
 More than half of the existing methods  ( Fan et al., 2019 ; Fan et al., 2022 ; Fu et al., 2021 ; Jiang et al., 2021a ; Tien and Van, 2020 ; Gu et al., 2021 ; Narang et al., 2021 ; Mandal and Maiti, 2021 ; Mu et al., 2019 ; Bai et al., 2020 ; Vijaikumar et al., 2019 ; Hou et al., 2021 ; Bi et al., 2021 ; Salamat et al., 2021 ; Leng and Yu, 2022 ; Zhao et al., 2021 ; Niu et al., 2021 ; Hoang et al., 2021 ; Xiao et al., 2020 ; Xu et al., 2020 ; Wu et al., 2019b ; Xiao et al., 2021 ; Xiao et al., 2022 ; Walker et al., 2021 ; Shi et al., 2022 ; Wang and Zhao, 2022 ; Wang et al., 2022b ; Qiao et al., 2022 ; Wei et al., 2022a ; Jiang and Sun, 2022 ; Lin et al., 2022 ; Chen et al., 2022c ; Song et al., 2019 ; Song et al., 2020 ) predict a user p i p_{i} ’s preference r ^ i ​ j \hat{r}_{ij} on an item q j q_{j} by employing MLP as follows:

 

 
 (12) | 
 | 
 r ^ i ​ j = σ L ​ ( 𝐖 L ⊤ ​ ( σ L − 1 ​ ( … ​ σ 2 ​ ( 𝐖 2 ⊤ ​ [ 𝐮 i 𝐯 j ] + 𝐛 2 ) ​ … ) ) + 𝐛 L CLOSE , \hat{r}_{ij}=\sigma_{L}(\mathbf{W}^{\top}_{L}(\sigma_{L-1}(...\sigma_{2}(\mathbf{W}_{2}^{\top}\begin{bmatrix}\mathbf{u}_{i}\\
\mathbf{v}_{j}\end{bmatrix}+\mathbf{b}_{2})...))+\mathbf{b}_{L},\\
 | 
 | 
 

 where 𝐖 i \mathbf{W}_{i} , 𝐛 i \mathbf{b}_{i} , and σ i \sigma_{i} denote the weight matrix, bias vector, and activation function for i i -th layer’s perceptron, respectively.

 
 
 
 

### 5.3. Loss Functions

 
 In this subsection, we first group the primary loss functions of GNN-based SocialRS into 4 categories: Bayesian personalized ranking (BPR)  ( Rendle et al., 2009 ) , mean squared error (MSE), cross-entropy (CE), and hinge loss.
In addition, we found that some works additionally employ auxiliary loss functions.
Thus, we further group these loss functions into 8 categories: social link prediction (LP) loss, self-supervised loss (SSL), group-based loss, adversarial (Adv) loss, path-based loss, knowledge distillation (KD) loss, sentiment-aware loss, and policy-network-based (Policy Net) loss.
Table  9 summarizes the taxonomy of loss functions used in existing work.

 
 

#### 5.3.1. Primary Loss Functions 

 
 Different primary loss functions are employed depending on whether the methods focus on explicit or implicit feedback.

 
 
 MSE Loss. For the methods that focus on explicit feedback ( e .g., star ratings) of users, most of them  ( Fan et al., 2019 ; Mandal and Maiti, 2021 ; Mu et al., 2019 ; Bai et al., 2020 ; Hou et al., 2021 ; Bi et al., 2021 ; Fan et al., 2022 ; Hoang et al., 2021 ; Zheng et al., 2021 ; Guo and Wang, 2020 ; Wu et al., 2019b ; Jiang et al., 2021a ; Tien and Van, 2020 ; Salamat et al., 2021 ; Xu et al., 2020 ; Liao et al., 2022a ; Sun et al., 2020 ; Niu et al., 2021 ; Zhen et al., 2022 ; Wu et al., 2022c ; Liu et al., 2022e ; Shi et al., 2022 ; Wang and Zhao, 2022 ; Qiao et al., 2022 ; Jiang and Sun, 2022 ; Lin et al., 2022 ; Chen et al., 2022c ) learn user and item embeddings via the MSE-based loss function ℒ M ​ S ​ E \mathcal{L}_{MSE} , which is defined as follows:

 

 
 (13) | 
 | 
 ℒ M ​ S ​ E = ∑ p i ∈ 𝒰 ∑ q j ∈ ℐ ( r ^ i ​ j − r i ​ j ) 2 , \mathcal{L}_{MSE}=\sum_{p_{i}\in\mathcal{U}}\sum_{q_{j}\in\mathcal{I}}(\hat{r}_{ij}-{r}_{ij})^{2}, | 
 | 
 

 where r i ​ j {r}_{ij} indicates p i p_{i} ’s real rating score on q j q_{j} . That is, the embeddings of p i p_{i} and q j q_{j} are learned, aiming at minimizing the differences between p i p_{i} ’s real and predicted scores, i .e., r i ​ j {r}_{ij} and r ^ i ​ j \hat{r}_{ij} , for q j q_{j} .

 
 
 BPR Loss. For the methods that focus on implicit feedback ( e .g., click or browsing history) of users, most of them  ( Jiang et al., 2021b ; Liu et al., 2021a ; Xiao et al., 2020 ; Li and Mu, 2020 ; Liao et al., 2022b ; Liu et al., 2022f ; Zhu et al., 2021 ; Xiao et al., 2021 ; Seng et al., 2021 ; Wu et al., 2019a ; Wu et al., 2020 ; Song et al., 2021 ; Jin et al., 2020 ; Leng and Yu, 2022 ; Yang et al., 2022 ; Yu et al., 2021a ; Wu et al., 2022a ; Yu et al., 2021b ; Han et al., 2022 ; Xiao et al., 2022 ; Liu et al., 2022a ; Chen et al., 2022b ; Yu et al., 2022a ; Liu et al., 2022c ; Sha et al., 2021 ; Xie et al., 2022 ; Li et al., 2022c ; Zhang et al., 2022a ; Song et al., 2022 ; Li et al., 2022a ; Wei et al., 2022a ; Miao et al., 2022 ; Liufu and Shen, 2021 ; Sun et al., 2022 ) learn user and item embeddings via the BPR-based loss function ℒ B ​ P ​ R \mathcal{L}_{BPR} , which is defined as follows:

 

 
 (14) | 
 | 
 ℒ B ​ P ​ R = − ∑ p i ∈ 𝒰 ∑ q j ∈ 𝒩 p i ∑ q k ∈ ℐ ​ \ ​ 𝒩 p i log σ ( r ^ i ​ j − r ^ i ​ k ) , \mathcal{L}_{BPR}=-\sum_{p_{i}\in\mathcal{U}}\sum_{q_{j}\in\mathcal{N}_{p_{i}}}\sum_{q_{k}\in\mathcal{I}\text{\textbackslash}\mathcal{N}_{p_{i}}}\text{log}\sigma(\hat{r}_{ij}-\hat{r}_{ik}), | 
 | 
 

 where 𝒰 \mathcal{U} and 𝒩 n i \mathcal{N}_{n_{i}} denote a set of users and a set of items rated by p i p_{i} , respectively. r ^ i ​ j \hat{r}_{ij} and r ^ i ​ k \hat{r}_{ik} indicate p i p_{i} ’s preference on the rated item q j q_{j} and the (randomly-sampled) unrated item q k q_{k} , respectively.
Also, σ \sigma indicates the sigmoid function.
That is, the embeddings of p i p_{i} , q j q_{j} , and q k q_{k} are learned based on the intuition that p i p_{i} ’s preference r ^ i ​ j \hat{r}_{ij} on q j q_{j} is likely to be higher than p i p_{i} ’s preference r ^ i ​ k \hat{r}_{ik} on q k q_{k} .

 
 
 CE Loss. Several methods  ( Fu et al., 2021 ; Vijaikumar et al., 2019 ; Wu et al., 2019b ; Zhao et al., 2021 ; Gu et al., 2021 ; Narang et al., 2021 ; Chen and Wong, 2021 ; Yan et al., 2022 ; Zhu et al., 2022 ; Tao et al., 2022 ; Chen et al., 2022d ; Liu et al., 2022b ; Wang et al., 2022b ; Walker et al., 2021 ; Wei et al., 2022b ; Chen et al., 2022a ; Chen et al., 2022c ; Li et al., 2021 ; Liu et al., 2022d ; Song et al., 2019 ; Song et al., 2020 ) for implicit feedback learn user and item embeddings via the CE-based loss function ℒ C ​ E \mathcal{L}_{CE} , which is defined as follows:

 

 
 (15) | 
 | 
 ℒ C ​ E = − ∑ p i ∈ 𝒰 ∑ q j ∈ ℐ r i ​ j log ( r ^ i ​ j ) + ( 1 − r i ​ j ) log ( 1 − r ^ i ​ j ) , \mathcal{L}_{CE}=-\sum_{p_{i}\in\mathcal{U}}\sum_{q_{j}\in\mathcal{I}}{r}_{ij}\text{log}(\hat{r}_{ij})+(1-{r}_{ij})\text{log}(1-\hat{r}_{ij}), | 
 | 
 

 where ℐ \mathcal{I} indicates a set of items. It should be noted that r i ​ j = 1 {r}_{ij}=1 if q j ∈ 𝒩 p i q_{j}\in\mathcal{N}_{p_{i}} , otherwise r i ​ j = 0 {r}_{ij}=0 .
That is, the embeddings of p i p_{i} and q j q_{j} are learned, aiming at maximizing p i p_{i} ’s preferences on his/her rated items while minimizing p i p_{i} ’s preferences on his/her unrated items.

 
 
 Hinge Loss. A method  ( Wang et al., 2021c ) for implicit feedback learns user and item embeddings via the hinge loss function ℒ H ​ i ​ n ​ g ​ e \mathcal{L}_{Hinge} , which is defined as follows:

 

 
 (16) | 
 | 
 ℒ H ​ i ​ n ​ g ​ e = ∑ p i ∈ 𝒰 ∑ q j ∈ 𝒩 p i ∑ q k ∈ ℐ ​ \ ​ 𝒩 p i max ​ ( 0 , λ + ( r ^ i ​ j ) 2 − ( r ^ i ​ k ) 2 ) , \mathcal{L}_{Hinge}=\sum_{p_{i}\in\mathcal{U}}\sum_{q_{j}\in\mathcal{N}_{p_{i}}}\sum_{q_{k}\in\mathcal{I}\text{\textbackslash}\mathcal{N}_{p_{i}}}\text{max}(0,\lambda+(\hat{r}_{ij})^{2}-(\hat{r}_{ik})^{2}), | 
 | 
 

 where λ \lambda indicates the safety margin size. That is, the embeddings of p i p_{i} , q j q_{j} , and q k q_{k} are learned, aiming at ensuring that p i p_{i} ’s preferences on his/her rated items q j q_{j} are higher than those on his/her unrated items q k q_{k} at least by a margin of λ \lambda .

 
 
 

#### 5.3.2. Auxiliary Loss Functions 

 
 Here, we discuss the auxiliary loss functions used by GNN-based SocialRS methods.

 
 
 Social Link Prediction (LP) Loss. 
It should be noted that the primary objectives of the existing works focus on reconstructing the input U-I rating graph.
Along with this, papers like MGNN  ( Xiao et al., 2020 ) , MutualRec  ( Xiao et al., 2021 ) , and SR-HGNN  ( Xu et al., 2020 ) learn the BPR-based social LP loss that aims at reconstructing the input U-U social graph.
Through this method, user embeddings can be informed further to reconstruct the social relations, which allows them to better capture the social network structure that is essential for the more-effective social recommendation.

 
 
 Self-Supervised Loss (SSL) . SSL originated in image and text domains to address the deficiency of labeled data  ( Liu et al., 2021b ) .
The basic idea of SSL is to assign labels for unlabeled data and exploit them additionally in the training process.
It is well-known that the data sparsity problem significantly affects the performance of recommender systems.
Therefore, there has recently been a surge of interest in SSL for recommender systems  ( Yu et al., 2022b ) .

 
 
 Some GNN-based SocialRS methods  ( Yu et al., 2021a ; Yu et al., 2021b ; Wu et al., 2022a ; Zhang et al., 2022a ; Li et al., 2022a ; Wang et al., 2022b ; Du et al., 2022 ; Wu et al., 2022a ; Sun et al., 2022 ) designed SSL, which is derived from U-U social and/or U-I rating graphs.
In this survey, we categorized them as social SSL and interaction-based SSL depending on the graph type employed to design SSL.
For the social SSL, SEPT  ( Yu et al., 2021a ) augments different views related to users with the U-U social graph and designs two socially-aware encoders that aim at reconstructing the augmented views. It adopts the regime of tri-training  ( Zhou and Li, 2005 ) , which operates on the augmented views above for self-supervised signals.
For the interaction-based SSL, SDCRec  ( Du et al., 2022 ) samples two items among items rated by a user, which have the highest similarities to the user. Then, it additionally utilizes them as self-supervised signals.

 
 
 On the other hand, Motif-Res  ( Sun et al., 2022 ) and MHCN  ( Yu et al., 2021b ) explore the motif information in graph structure so that such information can be utilized as self-supervised signals.
For instance, MHCN  ( Yu et al., 2021b ) constructs multi-type hyperedges, which are instances of a set of triangular relations, and designs SSL by leveraging the hierarchy in the hypergraph structures. It aims at reflecting the user node’s local and global high-order connectivity patterns in different hypergraphs  ( Yu et al., 2021b ) .

 
 
 Group-based Loss. GLOW  ( Leng and Yu, 2022 ) and GMAN  ( Liao et al., 2022a ) make use of the user groups.
Based on the group information, both methods additionally design the group-based loss.
They define the group-item interaction as indicating a set of users that have interacted with an item.
Then, they represent each group’s embedding by attentively aggregating the users’ embeddings within the corresponding group.
Finally, the user and item embeddings are learned via a group-based loss so that each group’s preferences on items rated by users in the corresponding group are likely to be higher than those of their unrated items.

 
 
 Others. We briefly discuss the other loss functions that are employed by only one method.
Yu et al.  ( Yu et al., 2022a ) designed an adversarial mechanism to consider the fact that social relations are very sparse, noisy, and multi-faceted in real-world social networks.
On the other hand, Li et al.  ( Li et al., 2021 ) pointed out that existing SocialRS methods fail to distinguish social influence from social homophily. To address this limitation, they designed an auxiliary loss function that models and captures the rich information conveyed by the formation of social homophily  ( Li et al., 2021 ) .
Furthermore, Tao et al.  ( Tao et al., 2022 ) leveraged the knowledge distillation (KD) technique into the social recommendation to address the overfitting problem of existing methods.
Shi et al.  ( Shi et al., 2022 ) incorporated both sentiment information derived from reviews and interaction information captured by the GNN encoder. To this end, they designed an auxiliary loss function that captures different sentimental aspects of items from reviews  ( Shi et al., 2022 ) .
Lastly, Wu et al.  ( Wu et al., 2019b ) designed a policy-based loss function based on a contextual multi-armed bandit  ( Bubeck and Cesa-Bianchi, 2012 ) , which dynamically weighs different social effects, i .e., social homophily, social influence, item-to-item homophily, and item-to-item influence.

 
 
 Table 10. Comparison of time complexity for GNN-based SocialRS methods. It should be noted that only methods that discuss their complexity in the respective papers are listed. 
 
 
 
 
 Encoders | 
 Models | 
 Time Complexity | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 GANN | 
 DiffNet++  ( Wu et al., 2020 ) | 
 O ⁡ ( m ⁡ ( L s + L i ) ​ D + n ​ L u ​ D ) O(m(L_{s}+L_{i})D+nL_{u}D) | 

 
 SR-HGNN  ( Xu et al., 2020 ) | 
 O ⁡ ( | 𝐑 | ​ d ) O(|\mathbf{R}|d) | 

 
 DISGCN  ( Li et al., 2022a ) | 
 OPEN O ⁡ ( | B | ​ d 2 ​ K + ( | B | + | 𝐒 + | + | 𝐑 | ) ​ d ​ K + | 𝐑 | ​ d ) ) O(|B|d^{2}K+(|B|+|\mathbf{S}^{+}|+|\mathbf{R}|)dK+|\mathbf{R}|d)) | 

 
 ME-LGN  ( Miao et al., 2022 ) | 
 O ⁡ ( m ​ K ​ s ) O(mKs) | 

 
 GDSRec  ( Chen et al., 2022c ) | 
 O ⁡ ( ( ( m + n ) ​ s + Q ) ​ D ) O(((m+n)s+ Q )D) | 

 
 GraphRec+  ( Fan et al., 2022 ) | 
 O ⁡ ( m ⁡ ( L i + L s ) ​ d + n ⁡ ( L u + m c ) ​ d ) O(m(L_{i}+L_{s})d+n(L_{u}+m_{c})d) | 

 
 ESRF  ( Yu et al., 2022a ) | 
 O ⁡ ( | 𝐑 | ​ d + | 𝐒 | ​ d + k ​ m ​ d ) O(|\mathbf{R}|d+|\mathbf{S}|d+kmd) | 

 
 SoHRML  ( Liu et al., 2022c ) | 
 O ( | 𝐑 | + | 𝐒 | ) ( 2 K d 1 d 2 + ∑ k = 1 d k d k − 1 ) ) O(|\mathbf{R}|+|\mathbf{S}|)(2Kd_{1}d_{2}+\sum_{k}=1d_{k}d_{k-1})) | 

 
 Emb | 
 SAN  ( Jiang et al., 2021a ) | 
 O ⁡ ( m ​ L s ​ K ) O(mL_{s}K) | 

 
 HSGNN  ( Wei et al., 2022a ) | 
 O ⁡ ( m ​ K ​ s ) O(mKs) | 

 
 GCN | 
 BFHAN  ( Zhao et al., 2021 ) | 
 O ⁡ ( K ⁡ ( | 𝐑 | ​ d + ( m + n ) ​ ( d ​ 2 + d ) ) ) O(K(|\mathbf{R}|d+(m+n)(d2+d))) | 

 
 SHGCN  ( Zhu et al., 2021 ) | 
 O ⁡ ( ( | 𝐒 | + | 𝐄 | ) ​ d ) O((|\mathbf{S}|+|\mathbf{E}|)d) | 

 
 GRNN | 
 GRNN | 
 GNN-DSR  ( Lin et al., 2022 ) | 
 O ⁡ ( m ⁡ ( 2 ​ L i + L s ) ​ d 2 + n ⁡ ( 2 ​ L u + m c ) ​ d 2 ) O(m(2L_{i}+L_{s})d^{2}+n(2L_{u}+m_{c})d^{2}) | 

 
 RNN | 
 RNN | 
 SGHAN  ( Wei et al., 2022b ) | 
 O ⁡ ( h ​ M ​ d ) + O ⁡ ( h ​ d ​ K ) O(hMd)+O(hdK) | 

 
 GCN | 
 GCN | 
 Emb | 
 DiffNet  ( Wu et al., 2019a ) | 
 O ⁡ ( m ​ K ​ L s ) O(mKL_{s}) | 

 
 MEGCN  ( Jin et al., 2020 ) | 
 O ⁡ ( m ​ k 1 ​ k 2 ) O(mk_{1}k_{2}) | 

 
 HOSR  ( Liu et al., 2022a ) | 
 O ⁡ ( K ​ | 𝐒 | ​ d 2 + | 𝐑 | ​ d ) O(K|\mathbf{S}|d^{2}+|\mathbf{R}|d) | 

 
 MLP | 
 Emb | 
 MGNN  ( Xiao et al., 2020 ) | 
 O ⁡ ( m 2 ​ K ​ d 2 ) O(m^{2}Kd^{2}) | 

 
 LightGCN | 
 LightGCN | 
 LightGCN | 
 SocialLGN  ( Liao et al., 2022b ) | 
 O ⁡ ( m ⁡ ( L s + L i + d ) ​ d + n ​ L u ​ d ) O(m(L_{s}+L_{i}+d)d+nL_{u}d) | 

 
 EAGCN  ( Wu et al., 2022c ) | 
 O ⁡ ( m + n ​ d 2 + ( | 𝐑 | + | 𝐒 | ) ​ d ​ K + | 𝐑 | ​ d ) O(m+nd^{2}+(|\mathbf{R}|+|\mathbf{S}|)dK+|\mathbf{R}|d) | 

 
 CGL  ( Zhang et al., 2022a ) | 
 O ⁡ ( | 𝐑 | ​ d ​ ( K + 1 ) + | 𝐒 | ​ d ​ K + 5 ​ B ​ d + B 2 ​ d ) O(|\mathbf{R}|d(K+1)+|\mathbf{S}|dK+5Bd+B^{2}d) | 

 
 SEPT  ( Yu et al., 2021a ) | 
 O ⁡ ( | 𝐑 | ​ d + m ​ l ​ o ​ g ​ ( K ) ) O(|\mathbf{R}|d+mlog(K)) | 

 
 DSR  ( Sha et al., 2021 ) | 
 O ⁡ ( ( L S ​ | 𝐒 | ​ t ​ d + L R ​ | 𝐑 | ​ d ) + m ​ f ​ d 2 ) O((L_{S}|\mathbf{S}|td+L_{R}|\mathbf{R}|d)+mfd^{2}) | 

 
 Emb | 
 IDiffNet  ( Li et al., 2022c ) | 
 O ⁡ ( m ​ K ​ L i + n ​ K ​ L u ) O(mKL_{i}+nKL_{u}) | 

 
 HyperGNN | 
 GANN | 
 HyperGNN | 
 MHCN  ( Yu et al., 2021b ) | 
 O ⁡ ( | 𝐀 + | ​ d ​ K ) O(|\mathbf{A}+|dK) | 

 
 Hyperbolic | 
 Hyperbolic | 
 Hyperbolic | 
 HyperSoRec  ( Wang et al., 2021c ) | 
 O ⁡ ( c ​ b ​ ∏ i = 1 K | N i | ) O(cb\prod_{i=1}^{K}|N_{i}|) | 

 

 
 
 
 
 

### 5.4. Model Complexity

 
 Finally, we conduct a time complexity analysis for GNN-based SocialRS methods. In Table  10 , we present a summary of the time complexity of the methods, providing values from the corresponding papers. The common notations used for the time complexity are outlined below.

 
 • 
 
 m m and n n : the number of users and items, respectively;

 

 • 
 
 | 𝐑 | |\mathbf{R}| , | 𝐒 | |\mathbf{S}| , and | 𝐄 | |\mathbf{E}| : number of edges in the user-item interaction, user-user social, and hypergraphs, respectively;

 

 • 
 
 | 𝐒 + | |\mathbf{S}^{+}| : number of all the friend pairs with social influence;

 

 • 
 
 K K and d d : number of GNN layers and the embedding size, respectively;

 

 • 
 
 L S L_{S} and L R L_{R} : number of layers for social and rating graphs, respectively;

 

 • 
 
 L s L_{s} and L i L_{i} : average number of social and item neighbors per user, respectively;

 

 • 
 
 L u L_{u} and m c m_{c} : average number of user and item neighbors per item, respectively;

 

 • 
 
 s s and k ​ m km : number of the sampled neighbors and alternative neighbors, respectively;

 

 • 
 
 t t , h h , and B B : number of iterations, number of LSTM hidden state, and the batch size, respectively.

 

 
 For the remaining model-specific notations, including M M , k 1 k_{1} , k 2 k_{2} , f f , c c , and b b , please refer to the corresponding papers. From Table  10 , we observe that methods using encoders such as RNN and HetGNN, which have high complexity, do not offer detailed insights into their computational demands.
On the contrary, most methods using LightGCN discuss their complexity and also substantiate their efficiency, including scalability, through experimental validation.

 
 
 Table 11. Statistics of 17 publicly-available benchmark datasets. Dataset source is hyperlinked to each dataset name: datasets colored blue contain links for both user-item interactions and user-user relations, whereas the ones in red only contain one of the two due to unavailability of the other. 
 
 
 
 
 Domains | 
 Datasets | 
 # Users | 
 # Items | 
 # Ratings | 
 # Social | 
 Papers Used | 

 
 Product | 
 Epinions | 
 18,088 | 
 261,649 | 
 764,352 | 
 355,813 | 
 ( Zheng et al., 2021 ; Guo and Wang, 2020 ; Xiao et al., 2021 ; Fan et al., 2019 ; Fu et al., 2021 ; Mandal and Maiti, 2021 ; Mu et al., 2019 ; Bai et al., 2020 ; Hou et al., 2021 ; Xiao et al., 2020 ; Bi et al., 2021 ; Li and Mu, 2020 ; Walker et al., 2021 ; Fan et al., 2022 ; Hoang et al., 2021 ) , | 

 
 ( Wu et al., 2020 ; Wu et al., 2019b ; Tien and Van, 2020 ; Salamat et al., 2021 ; Xu et al., 2020 ; Zhao et al., 2021 ; Narang et al., 2021 ; Sun et al., 2020 ; Xiao et al., 2022 ; Zhen et al., 2022 ; Liu et al., 2022e ; Du et al., 2022 ; Liu et al., 2022c ) , | 

 
 ( Lin et al., 2022 ; Chen et al., 2022c ; Li et al., 2021 ; Song et al., 2020 ; Sha et al., 2021 ; Tao et al., 2022 ; Wang et al., 2021c ; Wang and Zhao, 2022 ; Liu et al., 2022b ) | 

 
 Ciao | 
 7,317 | 
 104,975 | 
 283,319 | 
 111,781 | 
 ( Mandal and Maiti, 2021 ; Bai et al., 2020 ; Hou et al., 2021 ; Bi et al., 2021 ; Li and Mu, 2020 ; Liao et al., 2022b ; Walker et al., 2021 ; Liu et al., 2022f ; Tien and Van, 2020 ; Fan et al., 2022 ; Salamat et al., 2021 ; Xu et al., 2020 ; Zhao et al., 2021 ; Narang et al., 2021 ; Sun et al., 2020 ) , | 

 
 ( Wang et al., 2021c ; Fan et al., 2019 ; Fu et al., 2021 ; Jiang et al., 2021b ; Hoang et al., 2021 ; Wu et al., 2022a ; Liu et al., 2022d ; Xiao et al., 2022 ; Wu et al., 2022c ; Liu et al., 2022e ; Du et al., 2022 ; Liu et al., 2022c ; Sha et al., 2021 ; Tao et al., 2022 ; Chen et al., 2022c ) , | 

 
 ( Wang and Zhao, 2022 ; Song et al., 2022 ; Lin et al., 2022 ) | 

 
 Beidan | 
 2,841 | 
 2,298 | 
 35,146 | 
 2,367 | 
 ( Li et al., 2022a ) | 

 
 Beibei | 
 24,827 | 
 16,864 | 
 1,667,320 | 
 197,590 | 
 ( Li et al., 2022a ) | 

 
 Location | 
 Yelp 1 1 footnotemark: 
 1 
 
 
 
 | 
 19,539 | 
 21,266 | 
 405,884 | 
 363,672 | 
 ( Jiang et al., 2021b ; Vijaikumar et al., 2019 ; Liu et al., 2022f ; Guo and Wang, 2020 ; Wu et al., 2019a ; Wu et al., 2020 ; Song et al., 2021 ; Jin et al., 2020 ; Jiang et al., 2021a ; Yu et al., 2021a ; Yu et al., 2021b ; Han et al., 2022 ; Sun et al., 2022 ; Zhen et al., 2022 ) , | 

 
 ( Wu et al., 2022c ; Liu et al., 2022a ; Tao et al., 2022 ; Wang et al., 2021c ; Xie et al., 2022 ; Li et al., 2022c ; Zhang et al., 2022a ; Wang and Zhao, 2022 ; Song et al., 2022 ; Wei et al., 2022b ; Miao et al., 2022 ; Song et al., 2019 ; Shi et al., 2022 ; Chen et al., 2022d ; Qiao et al., 2022 ) | 

 
 Dianping | 
 59,426 | 
 10,224 | 
 934,334 | 
 813,331 | 
 ( Wu et al., 2020 ; Wu et al., 2022a ) | 

 
 Gowalla | 
 33,661 | 
 41,229 | 
 1,218,599 | 
 283,778 | 
 ( Seng et al., 2021 ; Chen and Wong, 2021 ; Li et al., 2022b ; Wu et al., 2022c ; Yu et al., 2022a ; Wang et al., 2022b ; Wei et al., 2022b ; Chen et al., 2022a ; Liufu and Shen, 2021 ) | 

 
 Foursquare | 
 39,302 | 
 45,595 | 
 3,627,093 | 
 304,030 | 
 ( Chen and Wong, 2021 ; Li et al., 2022b ; Wang et al., 2022b ; Chen et al., 2022a ) | 

 
 Movie | 
 MovieLens | 
 138,159 | 
 16,954 | 
 1,501,622 | 
 487,184 | 
 ( Liu et al., 2021a ; Tien and Van, 2020 ; Jiang and Sun, 2022 ; Chen et al., 2022d ) | 

 
 Flixster | 
 58,470 | 
 38,076 | 
 3,619,736 | 
 667,313 | 
 ( Xiao et al., 2020 ; Fan et al., 2022 ; Guo and Wang, 2020 ; Xiao et al., 2021 ; Xiao et al., 2022 ; Liu et al., 2022c ; Liu et al., 2022d ) | 

 
 FilmTrust | 
 1,508 | 
 2,071 | 
 35,497 | 
 1,853 | 
 ( Jiang et al., 2021b ; Mu et al., 2019 ; Zheng et al., 2021 ; Sun et al., 2022 ; Liu et al., 2022e ; Liu et al., 2022d ) | 

 
 Image | 
 Flickr | 
 8,358 | 
 82,120 | 
 327,815 | 
 187,273 | 
 ( Wu et al., 2019a ; Wu et al., 2020 ; Jin et al., 2020 ; Jiang et al., 2021a ; Wu et al., 2022c ; Tao et al., 2022 ; Xie et al., 2022 ; Li et al., 2022c ; Zhang et al., 2022a ) | 

 
 Music | 
 Last.fm | 
 1,892 | 
 17,632 | 
 92,834 | 
 25,434 | 
 ( Liao et al., 2022b ; Xiao et al., 2021 ; Seng et al., 2021 ; Tien and Van, 2020 ; Yu et al., 2021a ; Yu et al., 2021b ; Zhang et al., 2021 ; Chen et al., 2022b ; Yu et al., 2022a ; Liufu and Shen, 2021 ; Miao et al., 2022 ) | 

 
 Bookmark | 
 Delicious | 
 1,629 | 
 3,450 | 
 282,482 | 
 12,571 | 
 ( Li and Mu, 2020 ; Gu et al., 2021 ; Chen and Wong, 2021 ; Wang et al., 2022b ; Chen et al., 2022a ; Lin et al., 2022 ; Song et al., 2019 ) | 

 
 Microblog | 
 Weibo | 
 6,812 | 
 19,519 | 
 157,555 | 
 133,712 | 
 ( Li et al., 2021 ) | 

 
 Twitter | 
 8,930 | 
 232,849 | 
 466,259 | 
 96,718 | 
 ( Li et al., 2021 ) | 

 
 Miscellaneous | 
 Douban | 
 2,848 | 
 39,586 | 
 894,887 | 
 35,770 | 
 ( Bai et al., 2020 ; Walker et al., 2021 ; Seng et al., 2021 ; Salamat et al., 2021 ; Xu et al., 2020 ; Gu et al., 2021 ; Niu et al., 2021 ; Han et al., 2022 ; Sun et al., 2022 ; Liu et al., 2022a ; Liu et al., 2022b ; Liu et al., 2022d ; Song et al., 2019 ; Chen et al., 2022b ) , | 

 
 ( Yu et al., 2022a ; Yu et al., 2021a ; Yu et al., 2021b ; Song et al., 2020 ; Miao et al., 2022 ) | 

 

 
 • 
 
 1 Raw dataset is available at https://www.yelp.com/dataset/documentation/main .

 

 
 
 
 
 
 

## 6. Experimental Setup

 
 In this section, we discuss the experimental setup of GNN-based SocialRS methods.
Specifically, we review 17 benchmark datasets and 8 evaluation metrics, that are widely used in GNN-based SocialRS methods.
Furthermore, we compare the recommendation accuracy between GNN-based SocialRS methods across datasets.

 
 

### 6.1. Benchmark Datasets 

 
 We summarize the datasets widely used by existing GNN-based SocialRS methods in Table  11 .
These datasets come from 8 different application domains: product, location, movie, image, music, bookmark, microblog, and miscellaneous.
We present the statistics of each dataset, including the numbers of users, items, ratings, and social relations, and a list of papers using the corresponding dataset.
Since several versions exist per dataset, we chose the version that includes the most significant number of rating information.

 
 

#### 6.1.1. Product-related Datasets 

 
 Epinions. 
This dataset is collected from a now-defunct consumer review site, Epinions.
It contains 355.8K trust relations from 18.0K users and 764.3K ratings from 18.0K users on 261.6K products.
Here, a trust relation between two users indicates that one user trusts a review of a product written by another user.
For each rating, this dataset originally provides the product name, its category, the rating score in the range [1, 5], the timestamp that a user rated on an item, and the helpfulness of this rating.
37 GNN-based SocialRS methods reviewed in this survey used this dataset  ( Zheng et al., 2021 ; Guo and Wang, 2020 ; Xiao et al., 2021 ; Fan et al., 2019 ; Fu et al., 2021 ; Mandal and Maiti, 2021 ; Mu et al., 2019 ; Bai et al., 2020 ; Hou et al., 2021 ; Xiao et al., 2020 ; Bi et al., 2021 ; Li and Mu, 2020 ; Walker et al., 2021 ; Fan et al., 2022 ; Hoang et al., 2021 ; Wu et al., 2020 ; Wu et al., 2019b ; Tien and Van, 2020 ; Salamat et al., 2021 ; Xu et al., 2020 ; Zhao et al., 2021 ; Narang et al., 2021 ; Sun et al., 2020 ; Xiao et al., 2022 ; Zhen et al., 2022 ; Liu et al., 2022e ; Du et al., 2022 ; Liu et al., 2022c ; Lin et al., 2022 ; Chen et al., 2022c ; Li et al., 2021 ; Song et al., 2020 ; Sha et al., 2021 ; Tao et al., 2022 ; Wang et al., 2021c ; Wang and Zhao, 2022 ; Liu et al., 2022b ) , which means the most popular in SocialRS.

 
 
 Ciao. 
This dataset is collected from a consumer review site in the UK, Ciao ( https://www.ciao.co.uk/ ).
It contains 111.7K trust relations from 7.3K users and 283.3K ratings from 7.3K users on 104.9K products. The rating scale is from 1 to 5.
This dataset was used from 34 GNN-based SocialRS methods reviewed in this survey  ( Mandal and Maiti, 2021 ; Bai et al., 2020 ; Hou et al., 2021 ; Bi et al., 2021 ; Li and Mu, 2020 ; Liao et al., 2022b ; Walker et al., 2021 ; Liu et al., 2022f ; Tien and Van, 2020 ; Fan et al., 2022 ; Salamat et al., 2021 ; Xu et al., 2020 ; Zhao et al., 2021 ; Narang et al., 2021 ; Sun et al., 2020 ; Wang et al., 2021c ; Fan et al., 2019 ; Fu et al., 2021 ; Jiang et al., 2021b ; Hoang et al., 2021 ; Wu et al., 2022a ; Liu et al., 2022d ; Xiao et al., 2022 ; Wu et al., 2022c ; Liu et al., 2022e ; Du et al., 2022 ; Liu et al., 2022c ; Sha et al., 2021 ; Tao et al., 2022 ; Chen et al., 2022c ; Wang and Zhao, 2022 ; Song et al., 2022 ; Lin et al., 2022 ) .

 
 
 Beidan. 
This dataset is collected from a social e-commerce platform in China, Beidan ( https://www.beidian.com/ ), which allows users’ sharing behaviors.
It includes 2.3K social relations from 2.8K users and 35.1K ratings from 2.8K users on 2.2K products.
For each social relation, Li et al  ( Li et al., 2022a ) collected when a user’s friend clicks a link shared by the user that points to the information of a specific item.
In this dataset, rating information does not provide explicit preference scores of users, rather containing implicit feedback only.
This dataset was used in only one GNN-based SocialRS method  ( Li et al., 2022a ) .

 
 
 Beibei. 
This dataset is collected from another social e-commerce platform in China, Beibei ( https://www.beibei.com/ ).
It is similar to Beidan but provides larger sizes of social relations and ratings.
This dataset includes 197.5K social relations from 24.8K users and 1.6M ratings from 24.8K users on 16.8K products.
For ratings, this dataset provides users’ implicit feedback.
This dataset was used in  ( Li et al., 2022a ) only.

 
 
 

#### 6.1.2. Location-related Datasets 

 
 Yelp. 
This dataset is collected from a business review site, Yelp ( https://www.yelp.com/ ).
It contains 363.6K social relations from 19.5K users and 405.8K ratings from 19.5K users on 21.2K businesses.
On Yelp, users can share their check-ins about local businesses ( e .g., restaurants and home services) and express their experience through ratings in the range [0, 5].
Also, users can create social relations with other users.
Each check-in contains a user, a timestamp, and a business ( i .e., an item) that the user visited.
29 GNN-based SocialRS methods reviewed in this survey used this dataset  ( Jiang et al., 2021b ; Vijaikumar et al., 2019 ; Liu et al., 2022f ; Guo and Wang, 2020 ; Wu et al., 2019a ; Wu et al., 2020 ; Song et al., 2021 ; Jin et al., 2020 ; Jiang et al., 2021a ; Yu et al., 2021a ; Yu et al., 2021b ; Han et al., 2022 ; Sun et al., 2022 ; Zhen et al., 2022 ; Wu et al., 2022c ; Liu et al., 2022a ; Tao et al., 2022 ; Wang et al., 2021c ; Xie et al., 2022 ; Li et al., 2022c ; Zhang et al., 2022a ; Wang and Zhao, 2022 ; Song et al., 2022 ; Wei et al., 2022b ; Miao et al., 2022 ; Song et al., 2019 ; Shi et al., 2022 ; Chen et al., 2022d ; Qiao et al., 2022 ) .

 
 
 Dianping. 
This dataset is collected from a local restaurant search and review platform in China, Dianping ( https://www.dianping.com/ ).
It contains 813.3K social relations from 59.4K users and 934.3K ratings from 59.4K users on 10.2K restaurants.
For ratings, each user can give scores in the range [1, 5].
This dataset was used in two GNN-based SocialRS methods  ( Wu et al., 2020 ; Wu et al., 2022a ) .

 
 
 Gowalla. 
This dataset is collected from a location-based social networking site, Gowalla ( https://www.gowalla.com/ ).
It contains 283.7K friendship relations from 33.6K users and 1.2M ratings from 33.6K users on 41.2K locations.
On Gowalla, users can share information about their locations by check-in and make friends based on the shared information.
For ratings, this dataset provides users’ implicit feedback.
9 GNN-based SocialRS methods used this dataset  ( Seng et al., 2021 ; Chen and Wong, 2021 ; Li et al., 2022b ; Wu et al., 2022c ; Yu et al., 2022a ; Wang et al., 2022b ; Wei et al., 2022b ; Chen et al., 2022a ; Liufu and Shen, 2021 ) .

 
 
 Foursquare. 
This dataset is collected from another location-based social networking site, Foursquare ( https://foursquare.com/ ).
It is similar to Gowalla but provides larger sizes of social relations and ratings.
It contains 304.0K friendship relations from 39.3K users and 3.6M ratings from 39.3K users on 45.5K locations.
For ratings, this dataset provides users’ implicit feedback.
This dataset was used in 4 GNN-based SocialRS methods  ( Chen and Wong, 2021 ; Li et al., 2022b ; Wang et al., 2022b ; Chen et al., 2022a ) .

 
 
 

#### 6.1.3. Movie-related Datasets 

 
 MovieLens. 
This dataset is collected from GroupLens Research ( https://grouplens.org/ ) for the purpose of recommendation research.
It contains 487.1K social relations from 138.1K users and 1.5M ratings from 138.1K users on 16.9K movies.
It should be noted that this dataset has different versions according to the size of the rating information. For the details, refer to https://grouplens.org/datasets/movielens/ .
Since the original MovieLens datasets do not contain users’ social relations, methods using this dataset built social relations by calculating the similarities between users.
This dataset was used in 4 GNN-based SocialRS methods  ( Liu et al., 2021a ; Tien and Van, 2020 ; Jiang and Sun, 2022 ; Chen et al., 2022d ) .

 
 
 Flixster. 
This dataset is collected from a movie review site, Flixster ( https://www.flixster.com/ ).
It contains 667.3K friendship relations from 58.4K users and 3.6M ratings from 58.4K users on 38.0K movies.
On Flixster, users can add other users to their friend lists and express their preferences for movies.
The rating values are 10 discrete numbers in the range [0.5, 5].
We found that 7 GNN-based SocialRS methods used this dataset  ( Xiao et al., 2020 ; Fan et al., 2022 ; Guo and Wang, 2020 ; Xiao et al., 2021 ; Xiao et al., 2022 ; Liu et al., 2022c ; Liu et al., 2022d ) .

 
 
 FilmTrust. 
This dataset is collected from another (now-defunct) movie review site, FilmTrust.
It is similar to Flixster but provides smaller sizes of social relations and ratings.
It contains 1.8K friendship relations from 1.5K users and 35.4K ratings from 1.5K users on 2.0K movies.
The rating scale is from 1 to 5.
This dataset was used in 6 GNN-based SocialRS methods  ( Jiang et al., 2021b ; Mu et al., 2019 ; Zheng et al., 2021 ; Sun et al., 2022 ; Liu et al., 2022e ; Liu et al., 2022d ) .

 
 
 

#### 6.1.4. Image-related Dataset 

 
 Flickr. 
This dataset is collected from a who-trust-whom online image-based social sharing platform, Flickr ( https://www.flickr.com/ ).
It contains 187.2K follow relations from 8.3K users and 327.8K ratings from 8.3K users on 82.1K images.
On Flickr, users can follow other users and share their preferences for images with their followers.
For ratings, this dataset provides users’ implicit feedback.
Also, we found that 9 GNN-based SocialRS methods used this dataset  ( Wu et al., 2019a ; Wu et al., 2020 ; Jin et al., 2020 ; Jiang et al., 2021a ; Wu et al., 2022c ; Tao et al., 2022 ; Xie et al., 2022 ; Li et al., 2022c ; Zhang et al., 2022a ) .

 
 
 

#### 6.1.5. Music-related Dataset 

 
 Last.fm. 
This dataset is collected from a social music platform, Lat.fm ( https://www.last.fm/ ).
It contains 25.4K social relations from 1.8K users and 92.8K ratings from 1.8K users on 17.6K music artists.
Each rating indicates that one user listened to an artist’s music, i .e., implicit feedback.
On Lat.fm, users can make friend relations based on their preferences for artists.
This dataset was used in 11 GNN-based SocialRS methods  ( Liao et al., 2022b ; Xiao et al., 2021 ; Seng et al., 2021 ; Tien and Van, 2020 ; Yu et al., 2021a ; Yu et al., 2021b ; Zhang et al., 2021 ; Chen et al., 2022b ; Yu et al., 2022a ; Liufu and Shen, 2021 ; Miao et al., 2022 ) .

 
 
 

#### 6.1.6. Bookmark-related Dataset 

 
 Delicious. 
This dataset is collected from a social bookmarking system, Delicious ( https://del.icio.us/ ).
It contains 12.5K social relations from 1.6K users and 282.4K ratings from 1.6K users on 3.4K tags.
On Delicious, users can bookmark URLs ( i .e., implicit feedback) and also assign a variety of semantic tags to bookmarks.
Also, they can have social relations with other users having mutual bookmarks or tags.
This dataset was used in 7 GNN-based SocialRS methods  ( Li and Mu, 2020 ; Gu et al., 2021 ; Chen and Wong, 2021 ; Wang et al., 2022b ; Chen et al., 2022a ; Lin et al., 2022 ; Song et al., 2019 ) .

 
 
 

#### 6.1.7. Microblog-related Datasets 

 
 Weibo. 
This dataset is collected from a social microblog site in China, Weibo ( https://weibo.com/ ).
It contains 133.7K social relations from 6.8K users and 157.5K ratings from 6.8K users on 19.5K blogs.
On Weibo, users can post microblogs ( i .e., implicit feedback) and retweet other users’ blogs.
Based on such retweeting behavior, Li et al.  ( Li et al., 2021 ) collected social relations between users.
Specifically, if a user has retweeted a microblog from another user, a social relation between the two users is created.
This dataset was used in only one GNN-based SocialRS method  ( Li et al., 2021 ) .

 
 
 Twitter. 
This dataset is collected from another social microblog site, Twitter ( https://twitter.com/ ).
It is similar to Weibo and contains 96.7K social relations from 8.3K users and 466.2K ratings from 8.9K users on 232.8K blogs.
Li et al.  ( Li et al., 2021 ) collected social relations between two users if a user retweets or replies to a tweet from another user.
This dataset was used in  ( Li et al., 2021 ) only.

 
 
 

#### 6.1.8. Miscellaneous 

 
 Douban. 
This dataset is collected from a social platform in China, Douban ( https://douban.com/ ).
It contains 35.7K social relations from 2.8K users and 894.8K ratings from 2.8K users on 39.5K items of different categories ( e .g., books, movies, movies, and so on).
For ratings, this dataset provides users’ implicit feedback.
This dataset was used in 19 GNN-based SocialRS methods  ( Bai et al., 2020 ; Walker et al., 2021 ; Seng et al., 2021 ; Salamat et al., 2021 ; Xu et al., 2020 ; Gu et al., 2021 ; Niu et al., 2021 ; Han et al., 2022 ; Sun et al., 2022 ; Liu et al., 2022a ; Liu et al., 2022b ; Liu et al., 2022d ; Song et al., 2019 ; Chen et al., 2022b ; Yu et al., 2022a ; Yu et al., 2021a ; Yu et al., 2021b ; Song et al., 2020 ; Miao et al., 2022 ) .
However, it should be noted that most methods using this dataset split users’ ratings according to the item categories and then use those of some categories only, e .g., Douban-Movie and Douban-Book.

 
 
 
 

### 6.2. Evaluation Metrics

 

#### 6.2.1. Rating Prediction Task 

 
 The methods that focus on explicit feedback aim to minimize the errors of the rating prediction task.
To evaluate the performance of this task, they use the following metrics: root mean squared error (RMSE) and mean absolute error (MAE).
Specifically, MAE calculates the average error, the difference between the predicted and actual ratings, while RMSE emphasizes larger errors.
Both metrics are computed as follows:

 
 
 
 (17) | 
 | 
 M ​ A ​ E \displaystyle MAE | 
 = 1 M ​ ∑ p i ∈ 𝒰 ∑ q j ∈ 𝒩 p i | r ^ i ​ j − r i ​ j | , \displaystyle=\frac{1}{M}\sum_{p_{i}\in\mathcal{U}}\sum_{q_{j}\in\mathcal{N}_{p_{i}}}|\hat{r}_{ij}-{r}_{ij}|, | 
 | 

 
 | 
 R ​ M ​ S ​ E \displaystyle RMSE | 
 = 1 M ​ ∑ p i ∈ 𝒰 ∑ q j ∈ 𝒩 p i ( r ^ i ​ j − r i ​ j ) 2 , \displaystyle=\sqrt{\frac{1}{M}\sum_{p_{i}\in\mathcal{U}}\sum_{q_{j}\in\mathcal{N}_{p_{i}}}(\hat{r}_{ij}-{r}_{ij})^{2}}, | 
 | 
 

 where M M indicates the number of ratings. Also, 𝒰 \mathcal{U} and 𝒩 p i \mathcal{N}_{p_{i}} denote a set of users and a set of items rated by p i p_{i} , respectively.
Lastly, r i ​ j {r}_{ij} and r ^ i ​ j \hat{r}_{ij} indicate a user p i p_{i} ’s actual and predicted ratings on an item q j q_{j} , respectively.

 
 
 

#### 6.2.2. Top- N N Recommendation Task 

 
 The methods for implicit feedback aim to improve the accuracy of the top- N N recommendation task.
To evaluate the performance of this task, they use the following metrics: normalized discounted cumulative gain (NDCG)  ( Järvelin and Kekäläinen, 2000 ) , mean reciprocal rank (MRR)  ( Breese et al., 1998 ) , area under the ROC curve (AUC), F1 score, precision, recall, and hit rate (HR).

 
 
 First, NDCG reflects the importance of ranked positions of items in a set ℛ p i \mathcal{R}_{p_{i}} of N N items that each method recommends to a user p i p_{i} .
Let y k y_{k} represent a binary variable for k k -th item i k i_{k} in ℛ p i \mathcal{R}_{p_{i}} , i .e., y k ∈ { 0 , 1 } y_{k}\in{\{0,1\}} .
 y k y_{k} is set as 1 1 if i k ∈ ℛ p i i_{k}\in\mathcal{R}_{p_{i}} and set as 0 0 otherwise.
 𝒩 p i \mathcal{N}_{p_{i}} denotes a set of items considered relevant to p i p_{i} ( i .e., ground truth).
In this case, NDCG p i ​ @ ​ N \text{NDCG}_{p_{i}}@N is computed by:

 
 
 
 (18) | 
 | 
 NDCG p i ​ @ ​ N \displaystyle\text{NDCG}_{p_{i}}@N | 
 = DCG p i ​ @ ​ N IDCG p i ​ @ ​ N , \displaystyle=\dfrac{\text{DCG}_{p_{i}}@N}{\text{IDCG}_{p_{i}}@N}, | 
 | 

 
 | 
 DCG p i ​ @ ​ N \displaystyle\text{DCG}_{p_{i}}@N | 
 = ∑ k = 1 N 2 y k − 1 log 2 ⁡ ( k + 1 ) , \displaystyle=\sum_{k=1}^{N}\dfrac{2^{y_{k}}-1}{\log_{2}{(k+1)}}, | 
 | 
 

 where IDCG p i ​ @ ​ N \text{IDCG}_{p_{i}}@N is the ideal DCG at N , i .e., for the top-N items i k ∈ 𝒩 p i i_{k}\in\mathcal{N}_{p_{i}} , y k y_{k} is set as 1 1 .

 
 
 Second, MRR reflects the average inversed rankings of the first relevant item i k i_{k} in ℛ p i \mathcal{R}_{p_{i}} . MRR p i ​ @ ​ N \text{MRR}_{p_{i}}@N is computed by:

 

 
 (19) | 
 | 
 MRR p i ​ @ ​ N = 1 rank p i , \text{MRR}_{p_{i}}@N=\dfrac{1}{\text{rank}_{p_{i}}}, | 
 | 
 

 where rank p i \text{rank}_{p_{i}} refers to the rank position of the first relevant item in ℛ p i \mathcal{R}_{p_{i}} .

 
 
 Third, AUC evaluates whether each method ranks a rated item higher than an unrated item.
That is, AUC p i {}_{p_{i}} is computed by:

 

 
 (20) | 
 | 
 AUC p i = ∑ q j ∈ 𝒩 p i ∑ q k ∈ 𝒩 p i ​ \ ​ ℐ I ⁡ ( r ^ i ​ j r ^ i ​ k ) | 𝒩 p i | | 𝒩 p i \ ℐ | , \text{AUC}_{p_{i}}=\dfrac{\sum_{q_{j}\in\mathcal{N}_{p_{i}}}\sum_{q_{k}\in\mathcal{N}_{p_{i}}\text{\textbackslash}{\mathcal{I}}}I(\hat{r}_{ij} \hat{r}_{ik})}{\arrowvert\mathcal{N}_{p_{i}}\arrowvert\arrowvert\mathcal{N}_{p_{i}}\text{\textbackslash}{\mathcal{I}}\arrowvert}, | 
 | 
 

 where I ⁡ ( ⋅ ) I(\cdot) is the indicator function.

 
 
 F1 score measures a harmonic mean of the precision and recall of the predictions as:

 

 
 (21) | 
 | 
 F1 p i ​ @ ​ N = 2 ⋅ Precision p i ​ @ ​ N ⋅ Recall p i ​ @ ​ N Precision p i ​ @ ​ N + Recall p i ​ @ ​ N , \text{F1}_{p_{i}}@N=2\cdot\dfrac{\text{Precision}_{p_{i}}@N\cdot\text{Recall}_{p_{i}}@N}{\text{Precision}_{p_{i}}@N+\text{Recall}_{p_{i}}@N}, | 
 | 
 

 
 
 
 (22) | 
 | 
 Precision p i ​ @ ​ N \displaystyle\text{Precision}_{p_{i}}@N | 
 = | 𝒩 p i ⋂ ℛ p i | | ℛ p i | , \displaystyle=\dfrac{\arrowvert\mathcal{N}_{p_{i}}\bigcap\mathcal{R}_{p_{i}}\arrowvert}{\arrowvert\mathcal{R}_{p_{i}}\arrowvert}, | 
 | 

 
 | 
 Recall p i ​ @ ​ N \displaystyle\text{Recall}_{p_{i}}@N | 
 = | 𝒩 p i ⋂ ℛ p i | | 𝒩 p i | , \displaystyle=\dfrac{\arrowvert\mathcal{N}_{p_{i}}\bigcap\mathcal{R}_{p_{i}}\arrowvert}{\arrowvert\mathcal{N}_{p_{i}}\arrowvert}, | 
 | 
 

 where Precision p i ​ @ ​ N \text{Precision}_{p_{i}}@N and Recall p i ​ @ ​ N \text{Recall}_{p_{i}}@N denote precision and recall at N N , respectively.

 
 
 Finally, HR is simply the fraction of users for which the ground truth is included in each ℛ p i \mathcal{R}_{p_{i}} :

 

 
 (23) | 
 | 
 HR ​ @ ​ N = ∑ p i ∈ 𝒰 h ​ i ​ t p i m , \text{HR}@N=\frac{\sum_{p_{i}\in\mathcal{U}}hit_{p_{i}}}{m}, | 
 | 
 

 where m m indicates the number of users.
Also, h ​ i ​ t p i hit_{p_{i}} is assigned 1 if ℛ p i \mathcal{R}_{p_{i}} contains any of the ground truth of p i p_{i} , and 0 otherwise.

 
 
 Conclusion. These metrics thus give complementary insights. While NDCG measures the rank-discounted cumulative gain of the recommendations relative to the ideal gain, MRR finds the predicted rank of the most relevant item for each user. Thus, MRR is focused on the rank of only the first relevant item but NDCG can incorporate the relevance of all items in the ranked list. AUC, F1, Precision, and Recall metrics are rank-free classification metrics that distinguish the prediction of a ranked and an unranked item. While precision measures the proportion of predicted items that are relevant, recall measures the proportion of relevant items that are predicted. Finally, the hit rate measures how many users got their ground truth items predicted in the ranked list.

 
 
 
 

### 6.3. Experimental Results

 

 Table 12. Comparison of recommendation accuracy for GNN-based SocialRS methods. It should be noted that only methods with the same settings on each dataset are listed. 
 
 
 
 
 
 (a) Epinions | 

 
 Encoders | 
 Models | 
 MAE | 
 RMSE | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 GANN | 
 GraphRec  ( Fan et al., 2019 ) | 
 0.8168 | 
 1.0631 | 

 
 DANSER  ( Wu et al., 2019b ) | 
 0.7781 | 
 1.0268 | 

 
 KConvGraph  ( Tien and Van, 2020 ) | 
 0.8057 | 
 1.0104 | 

 
 SR-HGNN  ( Xu et al., 2020 ) | 
 0.7983 | 
 1.0326 | 

 
 GAT-NSR  ( Mu et al., 2019 ) | 
 0.7780 | 
 1.0190 | 

 
 PA-GAN  ( Hou et al., 2021 ) | 
 0.8112 | 
 1.0612 | 

 
 HeteroGraphRec  ( Salamat et al., 2021 ) | 
 0.8104 | 
 1.0483 | 

 
 GTN  ( Hoang et al., 2021 ) | 
 0.8436 | 
 1.0139 | 

 
 GraphRec+ ( Fan et al., 2022 ) | 
 0.8093 | 
 1.0576 | 

 
 GSFR  ( Xiao et al., 2022 ) | 
 0.8018 | 
 1.0501 | 

 
 GCN | 
 BFHAN  ( Zhao et al., 2021 ) | 
 0.8046 | 
 1.0403 | 

 
 GRNN | 
 GRNN | 
 GNN-DSR  ( Lin et al., 2022 ) | 
 0.8016 | 
 1.0579 | 

 
 DGARec-R  ( Sun et al., 2020 ) | 
 0.7818 | 
 1.0261 | 

 
 - | 
 - | 
 GHSCF  ( Bi et al., 2021 ) | 
 0.7968 | 
 0.9731 | 

 
 GCN | 
 RNN | 
 GANN | 
 MOHCN  ( Wang and Zhao, 2022 ) | 
 0.7905 | 
 1.0327 | 

 

 
 
 
 
 
 (b) Yelp | 

 
 
 
 Encoders | 
 Models | 
 HR@5 | 
 HR@10 | 
 HR@15 | 
 NDCG@5 | 
 NDCG@10 | 
 NDCG@15 | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 GANN | 
 DiffNet++  ( Wu et al., 2020 ) | 
 0.2602 | 
 0.3503 | 
 0.4051 | 
 0.1973 | 
 0.2288 | 
 0.2450 | 

 
 DiffNetLG  ( Song et al., 2021 ) | 
 0.2599 | 
 0.3711 | 
 0.4473 | 
 0.1941 | 
 0.2333 | 
 0.2586 | 

 
 SRAN  ( Xie et al., 2022 ) | 
 0.2639 | 
 0.3837 | 
 0.4611 | 
 0.1953 | 
 0.2382 | 
 0.2614 | 

 
 Emb | 
 SAN  ( Jiang et al., 2021a ) | 
 0.2348 | 
 0.3484 | 
 0.4257 | 
 0.1726 | 
 0.2125 | 
 0.2353 | 

 
 MrAPR  ( Song et al., 2022 ) | 
 - | 
 0.3624 | 
 - | 
 - | 
 0.2871 | 
 - | 

 
 GCN | 
 GCN | 
 Emb | 
 DiffNet  ( Wu et al., 2019a ) | 
 0.2276 | 
 0.3477 | 
 0.4232 | 
 0.1679 | 
 0.2121 | 
 0.2331 | 

 
 MEGCN  ( Jin et al., 2020 ) | 
 0.2590 | 
 0.3685 | 
 0.4332 | 
 0.2012 | 
 0.2394 | 
 0.2590 | 

 
 LightGCN | 
 LightGCN | 
 Emb | 
 IDiffNet  ( Li et al., 2022c ) | 
 0.2625 | 
 0.3840 | 
 0.4640 | 
 0.1928 | 
 0.2360 | 
 0.2604 | 

 

 
 
 
 
 
 (c) Flixster | 

 
 Encoders | 
 Models | 
 MAE | 
 RMSE | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 GANN | 
 GraphRec+ ( Fan et al., 2022 ) | 
 0.7047 | 
 0.9303 | 

 
 GSFR  ( Xiao et al., 2022 ) | 
 0.6871 | 
 0.9176 | 

 
 GCN | 
 GCN | 
 GCN | 
 GNN-SOR  ( Guo and Wang, 2020 ) | 
 0.865 | 
 0.8710 | 

 
 GAE | 
 GAE | 
 GAE | 
 SIGA  ( Liu et al., 2022d ) | 
 - | 
 0.9050 | 

 

 
 
 
 
 
 (d) Flickr | 

 
 
 
 Encoders | 
 Models | 
 HR@5 | 
 HR@10 | 
 HR@15 | 
 NDCG@5 | 
 NDCG@10 | 
 NDCG@15 | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 GANN | 
 DiffNet++  ( Wu et al., 2020 ) | 
 0.1412 | 
 0.1832 | 
 0.2203 | 
 0.1269 | 
 0.1420 | 
 0.1544 | 

 
 SRAN  ( Xie et al., 2022 ) | 
 0.1540 | 
 0.1970 | 
 0.2329 | 
 0.1395 | 
 0.1539 | 
 0.1653 | 

 
 Emb | 
 SAN  ( Jiang et al., 2021a ) | 
 0.1267 | 
 0.1653 | 
 0.1977 | 
 0.1151 | 
 0.1290 | 
 0.1393 | 

 
 GCN | 
 GCN | 
 Emb | 
 DiffNet  ( Wu et al., 2019a ) | 
 0.1210 | 
 0.1641 | 
 0.1952 | 
 0.1142 | 
 0.1273 | 
 0.1384 | 

 
 MEGCN  ( Jin et al., 2020 ) | 
 0.1302 | 
 0.1688 | 
 0.2053 | 
 0.1208 | 
 0.1344 | 
 0.1460 | 

 
 LightGCN | 
 LightGCN | 
 LightGCN | 
 DESIGN  ( Tao et al., 2022 ) | 
 - | 
 0.2517 | 
 - | 
 - | 
 0.2590 | 
 - | 

 
 Emb | 
 IDiffNet  ( Li et al., 2022c ) | 
 0.1706 | 
 0.2269 | 
 0.2684 | 
 0.1499 | 
 0.1694 | 
 0.1833 | 

 

 
 
 
 
 
 (e) Last.fm | 

 
 
 
 Encoders | 
 Models | 
 Precision@10 | 
 Precision@20 | 
 Recall@10 | 
 Recall@20 | 
 NDCG@10 | 
 NDCG@20 | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 GANN | 
 SGA  ( Liufu and Shen, 2021 ) | 
 - | 
 0.0712 | 
 - | 
 0.2497 | 
 - | 
 0.2723 | 

 
 GCN | 
 GCN | 
 Emb | 
 ATGCN  ( Seng et al., 2021 ) | 
 0.1253 | 
 - | 
 - | 
 - | 
 0.1607 | 
 - | 

 
 LightGCN | 
 LightGCN | 
 LightGCN | 
 SEPT  ( Yu et al., 2021a ) | 
 0.2019 | 
 - | 
 0.2048 | 
 - | 
 0.2450 | 
 - | 

 
 SocialLGN  ( Liao et al., 2022b ) | 
 0.1972 | 
 0.1368 | 
 0.2026 | 
 0.2794 | 
 0.2566 | 
 0.2883 | 

 
 HyperGNN | 
 GANN | 
 HyperGNN | 
 MHCN  ( Yu et al., 2021b ) | 
 0.2005 | 
 - | 
 0.2037 | 
 - | 
 0.2439 | 
 - | 

 

 
 
 
 
 
 (f) Delicious | 

 
 
 
 Encoders | 
 Models | 
 Recall@10 | 
 Recall@20 | 
 Recall@50 | 
 NDCG@10 | 
 NDCG@20 | 
 NDCG@50 | 
 MRR@20 | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 Emb | 
 HIDM  ( Li and Mu, 2020 ) | 
 0.1730 | 
 0.2255 | 
 0.3025 | 
 0.1145 | 
 0.1290 | 
 0.1453 | 
 - | 

 
 GRNN | 
 GRNN | 
 GNN-DSR  ( Lin et al., 2022 ) | 
 - | 
 - | 
 - | 
 0.2805 | 
 0.3164 | 
 - | 
 0.2254 | 

 
 GCN | 
 GRNN | 
 GCN | 
 EGFRec  ( Gu et al., 2021 ) | 
 - | 
 0.4181 | 
 - | 
 - | 
 0.3000 | 
 - | 
 0.1573 | 

 
 HetGNN | 
 HetGNN | 
 GANN | 
 DCAN  ( Wang et al., 2022b ) | 
 - | 
 - | 
 - | 
 0.2805 | 
 0.3045 | 
 - | 
 0.2198 | 

 

 
 
 
 
 
 (g) Douban | 

 
 Encoders | 
 Models | 
 Precision@10 | 
 Recall@10 | 
 NDCG@10 | 

 
 User Social | 
 User Interest | 
 Item Encoder | 

 
 GANN | 
 GANN | 
 GANN | 
 TGRec  ( Bai et al., 2020 ) | 
 - | 
 - | 
 0.2793 | 

 
 ESRF  ( Yu et al., 2022a ) | 
 0.1823 | 
 0.0654 | 
 0.2103 | 

 
 IGRec  ( Chen et al., 2022b ) | 
 - | 
 0.4806 | 
 0.3921 | 

 
 GCN | 
 GCN | 
 GCN | 
 ATGCN  ( Seng et al., 2021 ) | 
 0.1918 | 
 - | 
 0.2084 | 

 
 GRNN | 
 GRNN | 
 GNNRec  ( Liu et al., 2022b ) | 
 - | 
 0.2350 | 
 - | 

 
 GCN | 
 EGFRec  ( Gu et al., 2021 ) | 
 - | 
 - | 
 0.2008 | 

 
 HyperGNN | 
 GANN | 
 HyperGNN | 
 MHCN  ( Yu et al., 2021b ) | 
 0.1850 | 
 0.0668 | 
 0.2103 | 

 
 Motif-Res  ( Sun et al., 2022 ) | 
 0.2118 | 
 0.0510 | 
 0.2413 | 

 
 DH-HGCN  ( Han et al., 2022 ) | 
 - | 
 0.1081 | 
 0.0891 | 

 

 
 
 
 It should be noted that GNN-based SocialRS methods used different sets of datasets for experimentation and had different experimental settings, such as training/test ratio, top- k k values, and metrics.
For a fair comparison, we selected one dataset for each domain and compared the accuracy values of methods with the same settings on that dataset.

 
 
 Table  12 shows the results on Epinions (Product), Yelp (Location), Flixster (Movie), Flickr (Image), Last.fm (Music), Delicious (Bookmark), and Douban (Miscellaneous).
To summarize, we did not find evidence of GNN encoders being optimized for a specific domain. Therefore, the best performer will vary depending on the domain and metric.
However, it is worth noting that on the Douban dataset, many SocialRS methods are based on HyperGNN. This trend may be due to the fact that the Douban dataset contains different types of user behavior for different types of items.
By employing a HyperGNN encoder, the authors might attempt to accurately capture various motifs.

 
 
 
 

## 7. Future Directions

 
 In this section, we discuss the limitations of GNN-based SocialRS methods and present several future research directions.

 
 

### 7.1. Graph Augmentation in GNN-based SocialRS

 
 An intrinsic challenge of GNN-based SocialRS methods lies in the sparsity of the input data ( i .e., user-item interactions and user-user relations).
To mitigate this problem, some GNN-based SocialRS methods  ( Yu et al., 2021a ; Wu et al., 2022a ; Du et al., 2022 ; Wu et al., 2022a ; Zhang et al., 2022a ; Li et al., 2022a ; Wang et al., 2022b ; Sun et al., 2022 ; Yu et al., 2021b ) have explored more supervision signals from the input data so that such signals can be utilized as different views from the original graph structure.
Although many graph augmentation techniques  ( Ding et al., 2022 ; Zhao et al., 2022 ) such as node/edge deletion and graph rewiring have been proposed recently in a machine learning area, existing GNN-based SocialRS methods only focus on adding edges between two users or between a user and an item  ( Yu et al., 2021a ; Yu et al., 2021b ; Wu et al., 2022a ; Zhang et al., 2022a ; Li et al., 2022a ; Wang et al., 2022b ; Du et al., 2022 ; Wu et al., 2022a ; Sun et al., 2022 ) .
Therefore, it is a promising direction to leverage extra self-supervision signals based on various augmentation techniques to learn user and item embeddings more efficiently and effectively.

 
 
 

### 7.2. Trustworthy GNN-based SocialRS

 
 Existing GNN-based SocialRS methods have focused on improving their accuracy by only relying on users’ past feedback.
However, it is worth mentioning that there are other important ‘‘beyond accuracy’’ metrics, which we call trustworthiness 2 2 
 2 
 
 
 
 “Trustworthy” is defined in the Oxford Dictionary as follows: an object or a person that you can rely on to be good, honest, sincere, etc  ( Zhang et al., 2022b ) . 
according to  ( Zhang et al., 2022b ) .
Motivated by the importance of such metrics, various trustworthy GNN architectures have been proposed to incorporate core aspects of trustworthiness, including robustness, explainability, privacy, and fairness, in the context of GNN encoders  ( Zhang et al., 2022b ) .
One GNN-based SocialRS method is proposed in this direction to specifically address the privacy issue  ( Liu et al., 2022e ) .
In particular, Liu et al.  ( Liu et al., 2022e ) devised a framework that stores user privacy data only in local devices individually and analyzes them together via federated learning.
Thus, developing trustworthy GNN-based SocialRS is a wide open for research.
For example, consider robustness: bad actors may want to target certain products to certain users in a SocialRS setting; how robust would existing GNN-based SocialRS be against such attackers is an unanswered question and opens opportunities to create accurate as well as robust models.

 
 
 

### 7.3. Heterogeneity

 
 In real-world graphs, nodes and their interactions are often multi-typed.
Such graphs, which are called heterogeneous graphs, convey rich information such as heterogeneous attributes, meta-path structures, and temporal properties.
Although HetGNN encoders have recently attracted attention in many domains ( e .g., healthcare and cybersecurity)  ( Wang et al., 2022a ) , there have been only a few attempts to leverage such heterogeneity in SocialRS  ( Chen and Wong, 2021 ; Wang et al., 2022b ) .
Therefore, designing a HetGNN-based SocialRS method remains an open question for the future.

 
 
 

### 7.4. Efficiency and Scalability

 
 Most real-world graphs are too large and also grow rapidly.
However, most GNN-based SocialRS methods are too complicated, thus facing difficulty scaling to such large-scale graphs.
Some works have attempted to make more scalable versions of models, including SocialLGN  ( Liao et al., 2022b ) , SEPT  ( Yu et al., 2021a ) , and DcRec  ( Wu et al., 2022a ) , have attempted to remove the non-linear activation function, feature transformation, and self-connection, whereas Tao et al.  ( Tao et al., 2022 ) leveraged the knowledge distillation (KD) technique into SocialRS.
However, designing a highly scalable GNN architecture is an important problem that remains challenging to date.

 
 
 
 

## 8. Conclusions

 
 Although there has been a surge of papers on developing GNN-based social recommendation methods, no survey paper existed that reviewed them thoroughly.
Our work is the first systematic and comprehensive survey that studies 84 84 papers on GNN-based SocialRS, collected by following the PRISMA guidelines.
We present a novel taxonomy of inputs and architectures for GNN-based SocialRS, thus, categorizing different methods developed over the years in this important topic.
Through this survey, we hope to enable the researchers of this field to better position their works in the recent trend while forming a gateway for the new researchers to get introduced to this important and hot topic.
We hope this survey helps readers to grasp recent trends in SocialRS and develop novel GNN-based SocialRS methods.

 
 
 Acknowledgements. 
The work of Srijan Kumar is supported in part by NSF grants CNS-2154118, IIS-2027689, ITE-2137724, ITE-2230692, CNS-2239879, Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00112290102 (subcontract No. PO70745), and funding from Microsoft, Google, and The Home Depot.
The work of Sang-Wook Kim was supported by the Institute of Information communications Technology Planning Evaluation (IITP) grant funded by the Korean government (MSIT) (No.RS-2022-00155586, A High-Performance Big-Hypergraph Mining Platform for Real-World Downstream Tasks; No. 2020-0-01373, Artificial Intelligence Graduate School Program (Hanyang University)). The work of Yeon-Chang Lee was supported by the Institute of Information communications Technology Planning Evaluation (IITP) grant funded by the Korean government (MSIT)
(No.2020-0-01336, Artificial Intelligence Graduate School Program (UNIST)).

 
 
 

## References

 
 
 Bai et al . (2020) 
 
Ting Bai, Youjie Zhang,
Bin Wu, and Jian-Yun Nie.
2020.

 
 Temporal Graph Neural Networks for Social
Recommendation. In Proceedings of IEEE
International Conference on Big Data (BigData) . 898–903.

 
 
 

 
 Bi et al . (2021) 
 
Zhongqin Bi, Lina Jing,
Meijing Shan, Shuming Dou, and
Shiyang Wang. 2021.

 
 Hierarchical social recommendation model based on a
graph neural network.

 
 Wireless Communications and Mobile
Computing 2021 (2021).

 
 
 

 
 Breese et al . (1998) 
 
John S. Breese, David
Heckerman, and Carl Myers Kadie.
1998.

 
 Empirical Analysis of Predictive Algorithms for
Collaborative Filtering. In Proceedings of
Conference on Uncertainty in Artificial Intelligence (UAI) .
43–52.

 
 
 

 
 Bubeck and Cesa-Bianchi (2012) 
 
Sébastien Bubeck and
Nicolò Cesa-Bianchi.
2012.

 
 Regret Analysis of Stochastic and Nonstochastic
Multi-armed Bandit Problems.

 
 Foundations and Trends in Machine Learning 
5, 1 (2012),
1–122.

 
 
 

 
 Chen et al . (2022d) 
 
Hongxu Chen, Hongzhi Yin,
Tong Chen, Weiqing Wang,
Xue Li, and Xia Hu.
2022d.

 
 Social Boosted Recommendation With Folded Bipartite
Network Embedding.

 
 IEEE Transactions on Knowledge and Data
Engineering 34, 2
(2022), 914–926.

 
 
 https://doi.org/10.1109/TKDE.2020.2982878 

 

 
 Chen et al . (2022c) 
 
Jiajia Chen, Xin Xin,
Xianfeng Liang, Xiangnan He, and
Jun Liu. 2022c.

 
 GDSRec: Graph-Based DecentralizedCollaborative
Filtering for SocialRecommendation.

 
 IEEE Transactions on Knowledge and Data
Engineering (TKDE) (2022).

 
 
 

 
 Chen et al . (2018) 
 
Rui Chen, Qingyi Hua,
Yan-shuo Chang, Bo Wang,
Lei Zhang, and Xiangjie Kong.
2018.

 
 A Survey of Collaborative Filtering-Based
Recommender Systems: From Traditional Methods to Hybrid Methods Based on
Social Networks.

 
 IEEE Access 6
(2018), 64301–64320.

 
 
 

 
 Chen and Wong (2021) 
 
Tianwen Chen and Raymond
Chi-Wing Wong. 2021.

 
 An efficient and effective framework for
session-based social recommendation. In
 Proceedings of ACM International Conference on Web
Search and Data Mining (WSDM) . 400–408.

 
 
 

 
 Chen et al . (2022a) 
 
Yan Chen, Wanhui Qian,
Dongqin Liu, Mengdi Zhou,
Yipeng Su, Jizhong Han, and
Ruixuan Li. 2022a.

 
 Your Social Circle Affects Your Interests: Social
Influence Enhanced Session-Based Recommendation. In
 Proceedings of International Conference on
Computational Science (ICCS) . 549–562.

 
 
 

 
 Chen et al . (2022b) 
 
Yujin Chen, Jing Wang,
Zhihao Wu, and Youfang Lin.
2022b.

 
 Integrating user-Group relationships under interest
similarity constraints for social recommendation.

 
 Knowledge-Based Systems 
(2022), 108921.

 
 
 

 
 Deng et al . (2017) 
 
ShuiGuang Deng, Longtao
Huang, Guandong Xu, Xindong Wu, and
Zhaohui Wu. 2017.

 
 On Deep Learning for Trust-Aware Recommendations in
Social Networks.

 
 IEEE Transactions on Neural Networks and
Learning Systems 28, 5
(2017), 1164–1177.

 
 
 

 
 Deng (2022) 
 
Yue Deng. 2022.

 
 Recommender Systems Based on Graph Embedding
Techniques: A Review.

 
 IEEE Access 10
(2022), 51587–51633.

 
 
 

 
 Ding et al . (2022) 
 
Kaize Ding, Zhe Xu,
Hanghang Tong, and Huan Liu.
2022.

 
 Data Augmentation for Deep Graph Learning: A
Survey.

 
 ACM SIGKDD Explorations 
(2022).

 
 
 

 
 Dou et al . (2016) 
 
Yingtong Dou, Hao Yang,
and Xiaolong Deng. 2016.

 
 A Survey of Collaborative Filtering Algorithms for
Social Recommender Systems. In Proceedings of
International Conference on Semantics, Knowledge and Grids (SKG) .
40–46.

 
 
 

 
 Du et al . (2022) 
 
Jing Du, Zesheng Ye,
Lina Yao, Bin Guo, and
Zhiwen Yu. 2022.

 
 Socially-aware Dual Contrastive Learning for
Cold-Start Recommendation. In Proceedings of
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR) . 1927–1932.

 
 
 

 
 Fan et al . (2019) 
 
Wenqi Fan, Yao Ma,
Qing Li, Yuan He, Eric
Zhao, Jiliang Tang, and Dawei Yin.
2019.

 
 Graph neural networks for social recommendation.
In Proceedings of the ACM Web Conference (WWW) .
417–426.

 
 
 

 
 Fan et al . (2022) 
 
Wenqi Fan, Yao Ma,
Qing Li, Jianping Wang,
Guoyong Cai, Jiliang Tang, and
Dawei Yin. 2022.

 
 A Graph Neural Network Framework for Social
Recommendations.

 
 IEEE Transactions on Knowledge and Data
Engineering 34, 5
(2022), 2033–2047.

 
 
 https://doi.org/10.1109/TKDE.2020.3008732 

 

 
 Fu et al . (2021) 
 
Bairan Fu, Wenming Zhang,
Guangneng Hu, Xinyu Dai,
Shujian Huang, and Jiajun Chen.
2021.

 
 Dual side deep context-aware modulation for social
recommendation. In Proceedings of the ACM Web
Conference (WWW) . 2524–2534.

 
 
 

 
 Gao et al . (2022) 
 
Chen Gao, Yu Zheng,
Nian Li, Yinfeng Li,
Yingrong Qin, Jinghua Piao,
Yuhan Quan, Jianxin Chang,
Depeng Jin, Xiangnan He, and
Yong Li. 2022.

 
 A Survey of Graph Neural Networks for Recommender
Systems: Challenges, Methods, and Directions.

 
 ACM Transactions on Recommender Systems
(TORS) (2022).

 
 
 

 
 García-Sánchez et al . (2020) 
 
Francisco García-Sánchez,
Ricardo Colomo Palacios, and Rafael
Valencia-García. 2020.

 
 A social-semantic recommender system for
advertisements.

 
 Information Processing and Management 
57, 2 (2020),
102153.

 
 
 

 
 Gasparetti et al . (2021) 
 
Fabio Gasparetti, Giuseppe
Sansonetti, and Alessandro Micarelli.
2021.

 
 Community detection in social recommender systems:
a survey.

 
 Applied Intelligence 51,
6 (2021), 3975–3995.

 
 
 

 
 Gu et al . (2021) 
 
Pan Gu, Yuqiang Han,
Wei Gao, Guandong Xu, and
Jian Wu. 2021.

 
 Enhancing session-based social recommendation
through item graph embedding and contextual friendship modeling.

 
 Neurocomputing 419
(2021), 190–202.

 
 
 

 
 Guo and Wang (2020) 
 
Zhiwei Guo and Heng
Wang. 2020.

 
 A deep graph neural network-based mechanism for
social recommendations.

 
 IEEE Transactions on Industrial Informatics 
17, 4 (2020),
2776–2783.

 
 
 

 
 Han et al . (2022) 
 
Jiadi Han, Qian Tao,
Yufei Tang, and Yuhan Xia.
2022.

 
 DH-HGCN: Dual Homogeneity Hypergraph Convolutional
Network for Multiple Social Recommendations. In
 Proceedings of the International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR) .
2190–2194.

 
 
 

 
 He et al . (2020) 
 
Xiangnan He, Kuan Deng,
Xiang Wang, Yan Li,
Yong-Dong Zhang, and Meng Wang.
2020.

 
 LightGCN: Simplifying and Powering Graph
Convolution Network for Recommendation. In
 Proceedings of International ACM SIGIR Conference
on Research and Development in Information Retrieval (SIGIR) .
639–648.

 
 
 

 
 Hoang et al . (2021) 
 
Thi Linh Hoang, Tuan Dung
Pham, and Viet Cuong Ta.
2021.

 
 Improving Graph Convolutional Networks with
Transformer Layer in social-based items recommendation. In
 Proceedings of International Conference on
Knowledge and Systems Engineering (KSE) . 1–6.

 
 
 

 
 Hou et al . (2021) 
 
Liyang Hou, Wenping Kong,
Yali Gao, Yang Chen, and
Xiaoyong Li. 2021.

 
 PA-GAN: Graph Attention Network for
Preference-Aware Social Recommendation. In Journal
of Physics: Conference Series , Vol. 1848.
012141.

 
 
 

 
 Jamali and Ester (2010) 
 
Mohsen Jamali and Martin
Ester. 2010.

 
 A matrix factorization technique with trust
propagation for recommendation in social networks. In
 Proceedings of ACM Conference on Recommender
Systems (RecSys) . 135–142.

 
 
 

 
 Järvelin and Kekäläinen (2000) 
 
Kalervo Järvelin and
Jaana Kekäläinen.
2000.

 
 IR evaluation methods for retrieving highly
relevant documents. In Proceedings of
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR) . 41–48.

 
 
 

 
 Jiang et al . (2021a) 
 
Nan Jiang, Li Gao,
Fuxian Duan, Jie Wen,
Tao Wan, and Honglong Chen.
2021a.

 
 SAN: Attention-based social aggregation neural
networks for recommendation system.

 
 International Journal of Intelligent
Systems (2021).

 
 
 

 
 Jiang and Sun (2022) 
 
Wenbo Jiang and Yanrui
Sun. 2022.

 
 Social-RippleNet: Jointly modeling of ripple net
and social information for recommendation.

 
 Applied Intelligence 
(2022), 1–16.

 
 
 

 
 Jiang et al . (2021b) 
 
Yanbin Jiang, Huifang Ma,
Yuhang Liu, Zhixin Li, and
Liang Chang. 2021b.

 
 Enhancing social recommendation via two-level graph
attentional networks.

 
 Neurocomputing 449
(2021), 71–84.

 
 
 

 
 Jin et al . (2020) 
 
Bo Jin, Ke Cheng,
Liang Zhang, Yanjie Fu,
Minghao Yin, and Lu Jiang.
2020.

 
 Partial relationship aware influence diffusion via
a multi-channel encoding scheme for social recommendation. In
 Proceedings of ACM International Conference on
Information Knowledge Management (CIKM) . 585–594.

 
 
 

 
 Krioukov et al . (2010) 
 
Dmitri V. Krioukov,
Fragkiskos Papadopoulos, Maksim Kitsak,
Amin Vahdat, and Marián
Boguñá. 2010.

 
 Hyperbolic Geometry of Complex Networks.

 
 Physical Review E 82
(2010).

 
 
 

 
 Krishnan et al . (2019) 
 
Adit Krishnan, Hari
Cheruvu, Cheng Tao, and Hari
Sundaram. 2019.

 
 A modular adversarial approach to social
recommendation. In Proceedings of ACM
International Conference on Information Knowledge Management (CIKM) .
1753–1762.

 
 
 

 
 Leng and Yu (2022) 
 
Youfang Leng and Li
Yu. 2022.

 
 Incorporating global and local social networks for
group recommendations.

 
 Pattern Recognition 127
(2022), 108601.

 
 
 

 
 Li et al . (2021) 
 
Hui Li, Lianyun Li,
Guipeng Xv, Chen Lin, Ke
Li, and Bingchuan Jiang.
2021.

 
 SPEX: A Generic Framework for Enhancing Neural
Social Recommendation.

 
 ACM Transactions on Information Systems
(TOIS) 40, 2 (2021),
1–33.

 
 
 

 
 Li et al . (2022a) 
 
Nian Li, Chen Gao,
Depeng Jin, and Qingmin Liao.
2022a.

 
 Disentangled Modeling of Social Homophily and
Influence for Social Recommendation.

 
 IEEE Transactions on Knowledge and Data
Engineering (TKDE) (2022).

 
 
 

 
 Li et al . (2022b) 
 
Quan Li, Xinhua Xu,
Xinghong Liu, and CHEN Qi.
2022b.

 
 An Attention-Based Spatiotemporal GGNN for Next POI
Recommendation.

 
 IEEE Access (2022).

 
 
 

 
 Li and Mu (2020) 
 
Yuan Li and Kedian Mu.
2020.

 
 Heterogeneous Information Diffusion Model for
Social Recommendation. In Proceedings of IEEE
International Conference on Tools with Artificial Intelligence (ICTAI) .
184–191.

 
 
 

 
 Li et al . (2022c) 
 
Yuqiang Li, Zhilong Zhan,
Huan Li, and Chun Liu.
2022c.

 
 Interest-aware influence diffusion model for social
recommendation.

 
 Journal of Intelligent Information Systems 
58, 2 (2022),
363–377.

 
 
 

 
 Liao et al . (2022a) 
 
Guoqiong Liao, Xiaobin
Deng, Changxuan Wan, and Xiping Liu.
2022a.

 
 Group event recommendation based on graph
multi-head attention network combining explicit and implicit information.

 
 Information Processing Management 
59, 2 (2022),
102797.

 
 
 

 
 Liao et al . (2022b) 
 
Jie Liao, Wei Zhou,
Fengji Luo, Junhao Wen,
Min Gao, Xiuhua Li, and
Jun Zeng. 2022b.

 
 SocialLGN: Light Graph Convolution Network for
Social Recommendation.

 
 Information Sciences 
(2022).

 
 
 

 
 Lin et al . (2022) 
 
Junfa Lin, Siyuan Chen,
and Jiahai Wang. 2022.

 
 Graph neural networks with dynamic and static
representations for social recommendation. In
 Proceedings of International Conference on Database
Systems for Advanced Applications (DASFAA) . 264–271.

 
 
 

 
 Liu et al . (2022b) 
 
Chun Liu, Yuxiang Li,
Hong Lin, and Chaojie Zhang.
2022b.

 
 GNNRec: Gated graph neural network for
session-based social recommendation model.

 
 Journal of Intelligent Information Systems 
(2022), 1–20.

 
 
 

 
 Liu et al . (2022f) 
 
Hai Liu, Chao Zheng,
Duantengchuan Li, Zhaoli Zhang,
Ke Lin, Xiaoxuan Shen,
Neal N Xiong, and Jiazhang Wang.
2022f.

 
 Multi-perspective social recommendation method with
graph representation learning.

 
 Neurocomputing 468
(2022), 469–481.

 
 
 

 
 Liu et al . (2022d) 
 
Jinxin Liu, Yingyuan
Xiao, Wenguang Zheng, and Ching-Hsien
Hsu. 2022d.

 
 SIGA: social influence modeling integrating graph
autoencoder for rating prediction.

 
 Applied Intelligence 
(2022), 1–16.

 
 
 

 
 Liu et al . (2021a) 
 
Shenghao Liu, Bang Wang,
Xianjun Deng, and Laurence T Yang.
2021a.

 
 Self-Attentive Graph Convolution Network with
Latent Group Mining and Collaborative Filtering for Personalized
Recommendation.

 
 IEEE Transactions on Network Science and
Engineering (TNSE) (2021).

 
 
 

 
 Liu et al . (2021b) 
 
Xiao Liu, Fanjin Zhang,
Zhenyu Hou, Zhaoyu Wang,
Li Mian, Jing Zhang, and
Jie Tang. 2021b.

 
 Self-supervised Learning: Generative or
Contrastive.

 
 IEEE Transactions on Knowledge and Data
Engineering (TKDE) (2021).

 
 
 

 
 Liu et al . (2022a) 
 
Yang Liu, Liang Chen,
Xiangnan He, Jiaying Peng,
Zibin Zheng, and Jie Tang.
2022a.

 
 Modelling High-Order Social Relations for Item
Recommendation.

 
 IEEE Transactions on Knowledge and Data
Engineering 34, 9
(2022), 4385–4397.

 
 
 https://doi.org/10.1109/TKDE.2020.3039463 

 

 
 Liu et al . (2022c) 
 
Zhen Liu, Xiaodong Wang,
Ying Ma, and Xinxin Yang.
2022c.

 
 Relational metric learning with high-order
neighborhood interactions for social recommendation.

 
 Knowledge and Information Systems (KAIS) 
64, 6 (2022),
1525–1547.

 
 
 

 
 Liu et al . (2022e) 
 
Zhiwei Liu, Liangwei
Yang, Ziwei Fan, Hao Peng, and
Philip S Yu. 2022e.

 
 Federated social recommendation with graph neural
network.

 
 ACM Transactions on Intelligent Systems and
Technology (TIST) 13, 4
(2022), 1–24.

 
 
 

 
 Liufu and Shen (2021) 
 
Yuanwei Liufu and Hong
Shen. 2021.

 
 Social Recommendation via Graph Attentive
Aggregation. In Proceedings of International
Conference on Parallel and Distributed Computing: Applications and
Technologies (PDCAT) . 369–382.

 
 
 

 
 Ma et al . (2009a) 
 
Hao Ma, Irwin King, and
Michael R. Lyu. 2009a.

 
 Learning to recommend with social trust ensemble.
In Proceedings of the International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR) .
203–210.

 
 
 

 
 Ma et al . (2009b) 
 
Hao Ma, Michael R. Lyu,
and Irwin King. 2009b.

 
 In Proceedings of ACM Conference on
Recommender Systems (RecSys) . 189–196.

 
 
 

 
 Ma et al . (2008) 
 
Hao Ma, Haixuan Yang,
Michael R. Lyu, and Irwin King.
2008.

 
 SoRec: social recommendation using probabilistic
matrix factorization. In Proceedings of ACM
International Conference on Information Knowledge Management (CIKM) .
931–940.

 
 
 

 
 Ma et al . (2011) 
 
Hao Ma, Dengyong Zhou,
Chao Liu, Michael R. Lyu, and
Irwin King. 2011.

 
 Recommender systems with social regularization. In
 Proceedings of ACM International Conference on Web
Search and Data Mining (WSDM) . 287–296.

 
 
 

 
 Ma et al . (2024) 
 
Wenze Ma, Yuexian Wang,
Yanmin Zhu, Zhaobo Wang,
Mengyuan Jing, Xuhao Zhao,
Jiadi Yu, and Feilong Tang.
2024.

 
 MADM: A Model-agnostic Denoising Module for
Graph-based Social Recommendation. In Proceedings
of the 17th ACM International Conference on Web Search and Data Mining,
WSDM 2024, Merida, Mexico, March 4-8, 2024 . 501–509.

 
 
 

 
 Mandal and Maiti (2021) 
 
Supriyo Mandal and
Abyayananda Maiti. 2021.

 
 Graph Neural Networks for Heterogeneous Trust based
Social Recommendation. In Proceedings of IEEE
International Joint Conference on Neural Networks (IJCNN) .
1–8.

 
 
 

 
 Mao et al . (2021) 
 
Kelong Mao, Jieming Zhu,
Xi Xiao, Biao Lu,
Zhaowei Wang, and Xiuqiang He.
2021.

 
 UltraGCN: Ultra Simplification of Graph
Convolutional Networks for Recommendation. In
 Proceedings of ACM International Conference on
Information Knowledge Management (CIKM) . 1253–1262.

 
 
 

 
 Marsden and Friedkin (1993) 
 
Peter V Marsden and
Noah E Friedkin. 1993.

 
 Network studies of social influence.

 
 Sociological Methods Research 
22, 1 (1993),
127–151.

 
 
 

 
 McPherson et al . (2001) 
 
Miller McPherson, Lynn
Smith-Lovin, and James M Cook.
2001.

 
 Birds of a feather: Homophily in social networks.

 
 Annual Review of Sociology 
(2001), 415–444.

 
 
 

 
 Miao et al . (2022) 
 
Hang Miao, Anchen Li,
and Bo Yang. 2022.

 
 Meta-path Enhanced Lightweight Graph Neural Network
for Social Recommendation. In Proceedings of
International Conference on Database Systems for Advanced Applications
(DASFAA) . 134–149.

 
 
 

 
 Moher et al . (2009) 
 
David Moher, Alessandro
Liberati, Jennifer Tetzlaff, Douglas G
Altman, and PRISMA Group*.
2009.

 
 Preferred reporting items for systematic reviews
and meta-analyses: the PRISMA statement.

 
 Annals of Internal Medicine 
151, 4 (2009),
264–269.

 
 
 

 
 Mu et al . (2019) 
 
Nan Mu, Daren Zha,
Yuanye He, and Zhihao Tang.
2019.

 
 Graph attention networks for neural social
recommendation. In Proceedings of IEEE
International Conference on Tools with Artificial Intelligence (ICTAI) .
1320–1327.

 
 
 

 
 Narang et al . (2021) 
 
Kanika Narang, Yitong
Song, Alexander Schwing, and Hari
Sundaram. 2021.

 
 FuseRec: fusing user and item homophily modeling
with temporal recommender systems.

 
 Data Mining and Knowledge Discovery (DMKD) 
35, 3 (2021),
837–862.

 
 
 

 
 Niu et al . (2021) 
 
Yong Niu, Xing Xing,
Mindong Xin, Qiuyang Han, and
Zhichun Jia. 2021.

 
 Multi-preference Social Recommendation of Users
Based on Graph Neural Network. In Proceedings of
International Conference on Intelligent Computing, Automation and
Applications (ICAA) . 190–194.

 
 
 

 
 Papadimitriou et al . (2012) 
 
Alexis Papadimitriou,
Panagiotis Symeonidis, and Yannis
Manolopoulos. 2012.

 
 A generalized taxonomy of explanations styles for
traditional and social recommender systems.

 
 Data Mining and Knowledge Discovery (DMKD) 
24, 3 (2012),
555–583.

 
 
 

 
 Peng et al . (2017) 
 
Nanyun Peng, Hoifung
Poon, Chris Quirk, Kristina Toutanova,
and Wen-tau Yih. 2017.

 
 Cross-sentence n-ary relation extraction with graph
lstms.

 
 Transactions of the Association for
Computational Linguistics (TACL) 5
(2017), 101–115.

 
 
 

 
 Qiao et al . (2022) 
 
Pengpeng Qiao, Zhiwei
Zhang, Zhetao Li, Yuanxing Zhang,
Kaigui Bian, Yanzhou Li, and
Guoren Wang. 2022.

 
 TAG: Joint Triple-hierarchical Attention and GCN
for Review-based Social Recommender System.

 
 IEEE Transactions on Knowledge and Data
Engineering (TKDE) (2022).

 
 
 

 
 Quan et al . (2023) 
 
Yuhan Quan, Jingtao Ding,
Chen Gao, Lingling Yi,
Depeng Jin, and Yong Li.
2023.

 
 Robust Preference-Guided Denoising for Graph based
Social Recommendation. In Proceedings of the ACM
Web Conference 2023, WWW 2023, Austin, TX, USA, 30 April 2023 - 4 May
2023 . 1097–1108.

 
 
 

 
 Rendle et al . (2009) 
 
Steffen Rendle, Christoph
Freudenthaler, Zeno Gantner, and Lars
Schmidt-Thieme. 2009.

 
 BPR: Bayesian Personalized Ranking from Implicit
Feedback. In Proceedings of Conference on
Uncertainty in Artificial Intelligence (UAI) . 452–461.

 
 
 

 
 Salamat et al . (2021) 
 
Amirreza Salamat, Xiao
Luo, and Ali Jafari. 2021.

 
 HeteroGraphRec: A heterogeneous graph-based neural
networks for social recommendations.

 
 Knowledge-Based Systems 
217 (2021), 106817.

 
 
 

 
 Seng et al . (2021) 
 
Dewen Seng, Binquan Li,
Chenxuan Lai, and Jiayi Wang.
2021.

 
 Adaptive Learning User Implicit Trust Behavior
Based on Graph Convolution Network.

 
 IEEE Access 9
(2021), 108363–108372.

 
 
 

 
 Sha et al . (2021) 
 
Xiao Sha, Zhu Sun, and
Jie Zhang. 2021.

 
 Disentangling Multi-Facet Social Relations for
Recommendation.

 
 IEEE Transactions on Computational Social
Systems (2021).

 
 
 

 
 Shi et al . (2022) 
 
Liye Shi, Wen Wu,
Wang Guo, Wenxin Hu,
Jiayi Chen, Wei Zheng, and
Liang He. 2022.

 
 SENGR: Sentiment-Enhanced Neural Graph
Recommender.

 
 Information Sciences 589
(2022), 655–669.

 
 
 

 
 Shokeen and Rana (2018) 
 
Jyoti Shokeen and Chhavi
Rana. 2018.

 
 A review on the dynamics of social recommender
systems.

 
 International Journal of Web Engineering and
Technology 13, 3
(2018), 255–276.

 
 
 

 
 Shokeen and Rana (2020a) 
 
Jyoti Shokeen and Chhavi
Rana. 2020a.

 
 Social recommender systems: techniques, domains,
metrics, datasets and future scope.

 
 Journal of Intelligent Information Systems 
54, 3 (2020),
633–667.

 
 
 

 
 Shokeen and Rana (2020b) 
 
Jyoti Shokeen and Chhavi
Rana. 2020b.

 
 A study on features of social recommender systems.

 
 Artificial Intelligence Review 
53, 2 (2020),
965–988.

 
 
 

 
 Song et al . (2021) 
 
Changhao Song, Bo Wang,
Qinxue Jiang, Yehua Zhang,
Ruifang He, and Yuexian Hou.
2021.

 
 Social Recommendation with Implicit Social
Influence. In Proceedings of the International ACM
SIGIR Conference on Research and Development in Information Retrieval
(SIGIR) . 1788–1792.

 
 
 

 
 Song et al . (2022) 
 
Hongtao Song, Feng Wang,
Zhiqiang Ma, and Qilong Han.
2022.

 
 Multirelationship Aware Personalized Recommendation
Model. In Proceedings of International Conference
of Pioneering Computer Scientists, Engineers and Educators (ICPCSEE) .
123–136.

 
 
 

 
 Song et al . (2020) 
 
Liqiang Song, Ye Bi,
Mengqiu Yao, Zhenyu Wu,
Jianming Wang, and Jing Xiao.
2020.

 
 Dream: A dynamic relation-aware model for social
recommendation. In Proceedings of ACM
International Conference on Information Knowledge Management (CIKM) .
2225–2228.

 
 
 

 
 Song et al . (2019) 
 
Weiping Song, Zhiping
Xiao, Yifan Wang, Laurent Charlin,
Ming Zhang, and Jian Tang.
2019.

 
 Session-based social recommendation via dynamic
graph attention networks. In Proceedings of ACM
International Conference on Web Search and Data Mining (WSDM) .
555–563.

 
 
 

 
 Sun et al . (2020) 
 
Hongji Sun, Lili Lin,
and Riqing Chen. 2020.

 
 Social Recommendation based on Graph Neural
Networks. In Proceedings of IEEE International
Conference on Parallel Distributed Processing with Applications, Big Data
 Cloud Computing, Sustainable Computing Communications, Social Computing
 Networking (ISPA/BDCloud/SocialCom/SustainCom) .
489–496.

 
 
 

 
 Sun et al . (2022) 
 
Yundong Sun, Dongjie Zhu,
Haiwen Du, and Zhaoshuo Tian.
2022.

 
 Motifs-based Recommender System via Hypergraph
Convolution and Contrastive Learning.

 
 Neurocomputing (2022).

 
 
 

 
 Tang et al . (2013b) 
 
Jiliang Tang, Xia Hu,
Huiji Gao, and Huan Liu.
2013b.

 
 Exploiting Local and Global Social Context for
Recommendation. In Proceedings of International
Joint Conference on Artificial Intelligence (IJCAI) .
2712–2718.

 
 
 

 
 Tang et al . (2013a) 
 
Jiliang Tang, Xia Hu,
and Huan Liu. 2013a.

 
 Social recommendation: a review.

 
 Social Network Analysis and Mining 
3, 4 (2013),
1113–1133.

 
 
 

 
 Tao et al . (2022) 
 
Ye Tao, Ying Li,
Su Zhang, Zhirong Hou, and
Zhonghai Wu. 2022.

 
 Revisiting Graph based Social Recommendation: A
Distillation Enhanced Social Graph Network. In
 Proceedings of ACM Web Conference (WWW) .
2830–2838.

 
 
 

 
 Tien and Van (2020) 
 
Dong Nguyen Tien and
Hai Pham Van. 2020.

 
 Graph Neural Network Combined Knowledge Graph for
Recommendation System. In Proceedings of
International Conference on Computational Data and Social Networks
(CSoNet) . 59–70.

 
 
 

 
 Veličković et al . (2018) 
 
Petar Veličković,
Guillem Cucurull, Arantxa Casanova,
Adriana Romero, Pietro Lio, and
Yoshua Bengio. 2018.

 
 Graph attention networks. In
 Proceedings of International Conference on Learning
Representations (ICLR) .

 
 
 

 
 Vijaikumar et al . (2019) 
 
M Vijaikumar, Shirish
Shevade, and M Narasimha Murty.
2019.

 
 SoRecGAT: Leveraging graph attention mechanism for
top-N social recommendation. In Proceedings of
Joint European Conference on Machine Learning and Knowledge Discovery in
Databases (ECML-PKDD) . 430–446.

 
 
 

 
 Walker et al . (2021) 
 
Joojo Walker, Fengli
Zhang, Fan Zhou, and Ting Zhong.
2021.

 
 Social-trust-aware variational recommendation.

 
 International Journal of Intelligent
Systems (2021).

 
 
 

 
 Wang et al . (2021c) 
 
Hao Wang, Defu Lian,
Hanghang Tong, Qi Liu,
Zhenya Huang, and Enhong Chen.
2021c.

 
 Hypersorec: Exploiting hyperbolic user and item
representations with multiple aspects for social-aware recommendation.

 
 ACM Transactions on Information Systems
(TOIS) 40, 2 (2021),
1–28.

 
 
 

 
 Wang et al . (2022b) 
 
Liuyin Wang, Xianghong
Xu, Kai Ouyang, Huanzhong Duan,
Yanxiong Lu, and Hai-Tao Zheng.
2022b.

 
 Self-Supervised Dual-Channel Attentive Network for
Session-based Social Recommendation. In
 Proceedings of IEEE International Conference on
Data Engineering (ICDE) . 2034–2045.

 
 
 

 
 Wang et al . (2021a) 
 
Shoujin Wang, Longbing
Cao, Yan Wang, Quan Z Sheng,
Mehmet A Orgun, and Defu Lian.
2021a.

 
 A survey on session-based recommender systems.

 
 ACM Computing Surveys (CSUR) 
54, 7 (2021),
1–38.

 
 
 

 
 Wang et al . (2021b) 
 
Shoujin Wang, Liang Hu,
Yan Wang, Xiangnan He,
Quan Z. Sheng, Mehmet A. Orgun,
Longbing Cao, Francesco Ricci, and
Philip S. Yu. 2021b.

 
 Graph Learning based Recommender Systems: A
Review. In Proceedings of International Joint
Conference on Artificial Intelligence (IJCAI) . 4644–4652.

 
 
 

 
 Wang et al . (2023) 
 
Tianle Wang, Lianghao
Xia, and Chao Huang. 2023.

 
 Denoised Self-Augmented Learning for Social
Recommendation. In Proceedings of the
Thirty-Second International Joint Conference on Artificial Intelligence,
IJCAI 2023, 19th-25th August 2023, Macao, SAR, China .
2324–2331.

 
 
 

 
 Wang et al . (2022a) 
 
Xiao Wang, Deyu Bo,
Chuan Shi, Shaohua Fan,
Yanfang Ye, and Philip S. Yu.
2022a.

 
 A Survey on Heterogeneous Graph Embedding: Methods,
Techniques, Applications and Sources.

 
 IEEE Transactions on Big Data 
(2022).

 
 
 

 
 Wang and Zhao (2022) 
 
Yu Wang and Qilong
Zhao. 2022.

 
 Multi-Order Hypergraph Convolutional Neural Network
for Dynamic Social Recommendation System.

 
 IEEE Access 10
(2022), 87639–87649.

 
 
 

 
 Wei et al . (2022a) 
 
Chunyu Wei, Yushun Fan,
and Jia Zhang. 2022a.

 
 High-order Social Graph Neural Network for Service
Recommendation.

 
 IEEE Transactions on Network and Service
Management (TNSM) (2022).

 
 
 

 
 Wei et al . (2022b) 
 
Chunyu Wei, Yushun Fan,
and Jia Zhang. 2022b.

 
 Time-aware Service Recommendation with
Social-powered Graph Hierarchical Attention Network.

 
 IEEE Transactions on Services Computing 
(2022).

 
 
 

 
 Wu et al . (2022c) 
 
Bin Wu, Lihong Zhong,
Lina Yao, and Yangdong Ye.
2022c.

 
 EAGCN: An Efficient Adaptive Graph Convolutional
Network for Item Recommendation in Social Internet of Things.

 
 IEEE Internet of Things Journal 
(2022).

 
 
 

 
 Wu et al . (2022a) 
 
Jiahao Wu, Wenqi Fan,
Jingfan Chen, Shengcai Liu,
Qing Li, and Ke Tang.
2022a.

 
 Disentangled Contrastive Learning for Social
Recommendation. In Proceedings of ACM
International Conference on Information Knowledge Management (CIKM) .
4570–4574.

 
 
 

 
 Wu et al . (2020) 
 
Le Wu, Junwei Li,
Peijie Sun, Richang Hong,
Yong Ge, and Meng Wang.
2020.

 
 Diffnet++: A neural influence and interest
diffusion network for social recommendation.

 
 IEEE Transactions on Knowledge and Data
Engineering (TKDE) (2020).

 
 
 

 
 Wu et al . (2019a) 
 
Le Wu, Peijie Sun,
Yanjie Fu, Richang Hong,
Xiting Wang, and Meng Wang.
2019a.

 
 A neural influence diffusion model for social
recommendation. In Proceedings of the
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR) . 235–244.

 
 
 

 
 Wu et al . (2019b) 
 
Qitian Wu, Hengrui Zhang,
Xiaofeng Gao, Peng He,
Paul Weng, Han Gao, and
Guihai Chen. 2019b.

 
 Dual graph attention networks for deep latent
representation of multifaceted social effects in recommender systems. In
 Proceedings of the ACM Web Conference (WWW) .
2091–2102.

 
 
 

 
 Wu et al . (2022b) 
 
Shiwen Wu, Wentao Zhang,
Fei Sun, and Bin Cui.
2022b.

 
 Graph Neural Networks in Recommender Systems: A
Survey.

 
 ACM Computing Surveys (CSUR) 
37, 4 (2022).

 
 
 

 
 Xia et al . (2023) 
 
Lianghao Xia, Yizhen
Shao, Chao Huang, Yong Xu,
Huance Xu, and Jian Pei.
2023.

 
 Disentangled Graph Social Recommendation. In
 39th IEEE International Conference on Data
Engineering, ICDE 2023, Anaheim, CA, USA, April 3-7, 2023 .
2332–2344.

 
 
 

 
 Xiao et al . (2022) 
 
Xinyu Xiao, Junhao Wen,
Wei Zhou, Fengji Luo,
Min Gao, and Jun Zeng.
2022.

 
 Multi-interaction fusion collaborative filtering
for social recommendation.

 
 Expert Systems with Applications 
(2022), 117610.

 
 
 

 
 Xiao et al . (2021) 
 
Yang Xiao, Qingqi Pei,
Tingting Xiao, Lina Yao, and
Huan Liu. 2021.

 
 MutualRec: joint friend and item recommendations
with mutualistic attentional graph neural networks.

 
 Journal of Network and Computer
Applications 177 (2021),
102954.

 
 
 

 
 Xiao et al . (2020) 
 
Yang Xiao, Lina Yao,
Qingqi Pei, Xianzhi Wang,
Jian Yang, and Quan Z Sheng.
2020.

 
 MGNN: Mutualistic graph neural network for joint
friend and item recommendation.

 
 IEEE Intelligent Systems 
35, 5 (2020),
7–17.

 
 
 

 
 Xie et al . (2022) 
 
Xiaojun Xie, Xihuang
Zhang, Honggang Luo, and Tao Zhang.
2022.

 
 Similarity-based Multi-Relational Attention Network
for Social Recommendation. In Proceedings of
International Conference on Computing and Artificial Intelligence (ICCAI) .
307–317.

 
 
 

 
 Xu et al . (2015) 
 
Guandong Xu, Zhiang Wu,
Yanchun Zhang, and Jie Cao.
2015.

 
 Social networking meets recommender systems:
survey.

 
 International Journal Social Network Mining 
2, 1 (2015),
64–100.

 
 
 

 
 Xu et al . (2020) 
 
Huance Xu, Chao Huang,
Yong Xu, Lianghao Xia,
Hao Xing, and Dawei Yin.
2020.

 
 Global context enhanced social recommendation with
hierarchical graph neural networks. In Proceedings
of IEEE International Conference on Data Mining (ICDM) .
701–710.

 
 
 

 
 Yan et al . (2022) 
 
Dengcheng Yan, Tianyi
Tang, Wenxin Xie, Yiwen Zhang, and
Qiang He. 2022.

 
 Session-based social and dependency-aware software
recommendation.

 
 Applied Soft Computing 
118 (2022), 108463.

 
 
 

 
 Yang et al . (2013) 
 
Bo Yang, Yu Lei,
Dayou Liu, and Jiming Liu.
2013.

 
 Social Collaborative Filtering by Trust. In
 Proceedings of International Joint Conference on
Artificial Intelligence (IJCAI) . 2747–2753.

 
 
 

 
 Yang et al . (2022) 
 
Liangwei Yang, Zhiwei
Liu, Yu Wang, Chen Wang,
Ziwei Fan, and Philip S. Yu.
2022.

 
 Large-scale Personalized Video Game Recommendation
via Social-aware Contextualized Graph Neural Network. In
 Proceedings of the ACM Web Conference (WWW) .
3376–3386.

 
 
 

 
 Yang et al . (2014) 
 
Xiwang Yang, Yang Guo,
Yong Liu, and Harald Steck.
2014.

 
 A survey of collaborative filtering based social
recommender systems.

 
 Computer Communications 
41 (2014), 1–10.

 
 
 

 
 Ying et al . (2016) 
 
Haochao Ying, Liang Chen,
Yuwen Xiong, and Jian Wu.
2016.

 
 Collaborative Deep Ranking: A Hybrid Pair-Wise
Recommendation Algorithm with Implicit Feedback. In
 Proceedings of Pacific-Asia Conference on Knowledge
Discovery and Data Mining (PAKDD) , Vol. 9652.
555–567.

 
 
 

 
 Yu et al . (2021a) 
 
Junliang Yu, Hongzhi Yin,
Min Gao, Xin Xia,
Xiangliang Zhang, and Nguyen Quoc
Viet Hung. 2021a.

 
 Socially-aware self-supervised tri-training for
recommendation. In Proceedings of ACM SIGKDD
Conference on Knowledge Discovery Data Mining (KDD) .
2084–2092.

 
 
 

 
 Yu et al . (2022a) 
 
Junliang Yu, Hongzhi Yin,
Jundong Li, Min Gao, Zi
Huang, and Lizhen Cui.
2022a.

 
 Enhancing Social Recommendation With Adversarial
Graph Convolutional Networks.

 
 IEEE Transactions on Knowledge and Data
Engineering 34, 8
(2022), 3727–3739.

 
 
 https://doi.org/10.1109/TKDE.2020.3033673 

 

 
 Yu et al . (2021b) 
 
Junliang Yu, Hongzhi Yin,
Jundong Li, Qinyong Wang,
Nguyen Quoc Viet Hung, and Xiangliang
Zhang. 2021b.

 
 Self-supervised multi-channel hypergraph
convolutional network for social recommendation. In
 Proceedings of the ACM Web Conference (WWW) .
413–424.

 
 
 

 
 Yu et al . (2022b) 
 
Junliang Yu, Hongzhi Yin,
Xin Xia, Tong Chen,
Jundong Li, and Zi Huang.
2022b.

 
 Self-Supervised Learning for Recommender Systems:
A Survey.

 
 arXiv:2203.15876 (2022).

 
 
 

 
 Zayats and Ostendorf (2018) 
 
Victoria Zayats and Mari
Ostendorf. 2018.

 
 Conversation modeling on Reddit using a
graph-structured LSTM.

 
 Transactions of the Association for
Computational Linguistics (TACL) 6
(2018), 121–132.

 
 
 

 
 Zhang et al . (2022b) 
 
He Zhang, Bang Wu,
Xingliang Yuan, Shirui Pan,
Hanghang Tong, and Jian Pei.
2022b.

 
 Trustworthy Graph Neural Networks: Aspects, Methods
and Trends.

 
 arXiv:2205.07424 (2022).

 
 
 

 
 Zhang et al . (2021) 
 
Lisa Zhang, Zhe Kang,
Xiaoxin Sun, Hong Sun,
Bangzuo Zhang, and Dongbing Pu.
2021.

 
 KCRec: Knowledge-aware representation Graph
Convolutional Network for Recommendation.

 
 Knowledge-Based Systems 
230 (2021), 107399.

 
 
 

 
 Zhang et al . (2022a) 
 
Yongshuai Zhang, Jiajin
Huang, Mi Li, and Jian Yang.
2022a.

 
 Contrastive Graph Learning for Social
Recommendation.

 
 Frontiers in Physics 
(2022), 35.

 
 
 

 
 Zhao et al . (2021) 
 
Minghao Zhao, Qilin Deng,
Kai Wang, Runze Wu,
Jianrong Tao, Changjie Fan,
Liang Chen, and Peng Cui.
2021.

 
 Bilateral filtering graph convolutional network for
multi-relational social recommendation in the power-law networks.

 
 ACM Transactions on Information Systems
(TOIS) 40, 2 (2021),
1–24.

 
 
 

 
 Zhao et al . (2022) 
 
Tong Zhao, Gang Liu,
Stephan Günnemann, and Meng
Jiang. 2022.

 
 Graph Data Augmentation for Graph Machine Learning:
A Survey.

 
 arXiv:2202.08871 (2022).

 
 
 

 
 Zhen et al . (2022) 
 
Yan Zhen, Huan Liu,
Meiyu Sun, Boran Yang, and
Puning Zhang. 2022.

 
 Adaptive preference transfer for personalized IoT
entity recommendation.

 
 Pattern Recognition Letters 
162 (2022), 40–46.

 
 
 

 
 Zheng et al . (2021) 
 
Li Zheng, Qun Liu, and
Youmin Zhang. 2021.

 
 Social Recommendation Based on Preference
Disentangle Aggregation. In Proceedings of
International Conference on Big Data and Information Analytics (BigDIA) .
1–8.

 
 
 

 
 Zhou and Li (2005) 
 
Zhi-Hua Zhou and Ming
Li. 2005.

 
 Tri-Training: Exploiting Unlabeled Data Using Three
Classifiers.

 
 IEEE Transactions on Knowledge and Data
Engineering (TKDE) 17, 11
(2005), 1529–1541.

 
 
 

 
 Zhu et al . (2022) 
 
Peng Zhu, Dawei Cheng,
Siqiang Luo, Fangzhou Yang,
Yifeng Luo, Weining Qian, and
Aoying Zhou. 2022.

 
 SI-News: Integrating social information for news
recommendation with attention-based graph convolutional network.

 
 Neurocomputing 494
(2022), 33–42.

 
 
 

 
 Zhu et al . (2021) 
 
Zirui Zhu, Chen Gao,
Xu Chen, Nian Li, Depeng
Jin, and Yong Li. 2021.

 
 Inhomogeneous Social Recommendation with Hypergraph
Convolutional Networks. In Proceedings of IEEE
International Conference on Data Engineering (ICDE) .