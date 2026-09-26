A Review of Graph Neural Networks in Epidemic Modeling 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.19852v4 [cs.LG] 09 Sep 2024 
 
 

# A Review of Graph Neural Networks in Epidemic Modeling

 Conference:  Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining; August 25–29, 2024; Barcelona, Spain Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain DOI:  10.1145/3637528.3671455 ISBN:  979-8-4007-0490-1/24/08 
 
 
 Zewen Liu
 
 
 
 Note:  Equal Contribution
 
 Affiliation:  Department of Computer Science , Emory University 
 
 email: zewen.liu@emory.edu 
 
 , 
 Guancheng Wan
 
 
 
 Affiliation:  Department of Computer Science , Emory University 
 
 email: gwan4@emory.edu 
 
 , 
 B. Aditya Prakash
 
 
 
 Affiliation:  College of Computing , Georgia Institute of Technology 
 
 email: badityap@cc.gatech.edu 
 
 , 
 Max S. Y. Lau
 
 
 
 Affiliation:  Department of Biostatistics and Bioinformatics , Emory University 
 
 email: msy.lau@emory.edu 
 
 and 
 Wei Jin
 
 
 
 Affiliation:  Department of Computer Science , Emory University 
 
 email: wei.jin@emory.edu 
 
 © acmlicensed 

 Abstract. 
 
 Since the onset of the COVID-19 pandemic, there has been a growing interest in studying epidemiological models. Traditional mechanistic models mathematically describe the transmission mechanisms of infectious diseases. However, they often suffer from limitations of oversimplified or fixed assumptions, which could cause sub-optimal predictive power and inefficiency in capturing complex relation information. Consequently, Graph Neural Networks (GNNs) have emerged as a progressively popular tool in epidemic research. In this paper, we endeavor to furnish a comprehensive review of GNNs in epidemic tasks and highlight potential future directions. To accomplish this objective, we introduce hierarchical taxonomies for both epidemic tasks and methodologies, offering a trajectory of development within this domain. For epidemic tasks, we establish a taxonomy akin to those typically employed within the epidemic domain. For methodology, we categorize existing work into Neural Models and Hybrid Models . Following this, we perform an exhaustive and systematic examination of the methodologies, encompassing both the tasks and their technical details. Furthermore, we discuss the limitations of existing methods from diverse perspectives and systematically propose future research directions. This survey aims to bridge literature gaps and promote the progression of this promising field, with a list of relevant papers at https://github.com/Emory-Melody/awesome-epidemic-modeling-papers . We hope that it will facilitate synergies between the communities of GNNs and epidemiology, and contribute to their collective progress.

 
 
 
 Keywords:  Epidemiology; Graph Neural Networks; AI for Science; Spatial-Temporal Graphs
 
 

## 1. Introduction

 
 Figure 1. 
 Overview of the survey . Best viewed in color.
 
 
 
 Epidemiology has long been a critical field, with its origins tracing back to ancient societies that observed patterns of disease spread  Bramanti, 2012 ; Bruce-Chwatt, 1977 . Although the conceptualization of epidemiology has evolved over time, the terms health and control have been predominantly associated with it since 1978  Frérot et al., 2018 . Currently, the World Health Organization (WHO) describes epidemiology as the investigation into the distribution and determinants of health-related states or events, encompassing a broad spectrum of issues including disease transmission, vaccination efforts  Fine, 2015 ; Terris, 1993 , cancer and diabetes treatment, etc. This definition underscores the field’s emphasis on controlling health-related issues and making informed decisions. A pertinent illustration of this is the COVID-19 pandemic, which rapidly infected millions worldwide, placing immense strain on the production and distribution of medical resources  Cm, 2020 ; Bergrath et al., 2022 . Decision-making processes like allocation of resources, are crucial in mitigating the impact of diseases and saving lives, highlighting the importance of advancements in epidemiology  Emanuel et al., 2020 .

 
 
 To address a range of health-related challenges, there is an indispensable need for epidemic modeling, and researchers have devised various mechanistic models  Funk et al., 2018 ; Kondratyev, 2013 . These models, grounded in mathematical formulations, simulate the dissemination of infectious diseases by incorporating biological and behavioral underpinnings. By considering factors such as population, they yield insights into patterns of disease transmission and the efficacy of intervention strategies, thereby playing a pivotal role in shaping public health policies  Louz et al., 2010 ; Mikolajczyk et al., 2009 . However, these knowledge-driven methods often depend on oversimplified or fixed assumptions that can lead to biases in modeling. Consequently, this compromises both the accuracy of predictions and their ability to generalize across different contexts.

 
 
 To overcome the limitations of mechanistic models, there is an emerging trend to adopt data-driven approaches in epidemic forecasting tasks, with a particular emphasis on machine learning and deep learning models  Shorten et al., 2021 . Specifically, Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) have demonstrated great success in various epidemiological predictive tasks, including forecasting daily new case counts, estimating virus reproduction and doubling times, and determining disease-related factors  Wu et al., 2018 ; Saleem et al., 2022 ; Baldo et al., 2021 . Despite their effectiveness in these tasks, these models often fall short in incorporating relational information from critical epidemiological data sources such as human mobility, geographic connections, and contact tracing. This deficiency restricts their utility in broader epidemiological applications.

 
 
 Recently, the advances in Graph Neural Networks (GNNs)  Wu et al., 2020 ; Kipf Welling, 2017 ; Veličković et al., 2017 ; Brody et al., 2022 have set the stage for overcoming the aforementioned challenges in epidemic modeling. Specifically, GNNs stand out for their ability to aggregate diverse information through a message-passing mechanism, making them particularly suited for capturing relational dynamics within graphs  Liu et al., 2023 ; Maskey et al., 2022 . Thus, by representing interactions between entities as graphs, researchers can leverage GNNs to harness relational data effectively and facilitate epidemiology tasks  Deng et al., 2020 ; Wang et al., 2022 .
For instance, GNNs are often utilized to model spatial interactions  Gao et al., 2021 and other complex interactions  Cao et al., 2022 , enhancing the analysis of graph data and yielding more precise predictions. In addition, GNNs bring a certain level of interpretability by quantifying the influence of individual nodes (or entities) on final prediction  Feng et al., 2023 . Moreover, the flexible design of GNNs facilitates their integration with traditional mechanistic and probabilistic models to leverage expert knowledge and offer measures of uncertainty. As a result, GNNs have found extensive applications in various tasks within the field including infection prediction   Liu et al., 2023a , outbreak source detection  Ru et al., 2023 , intervention modeling  Song et al., 2020 , etc., facilitating advancements in epidemiology research.

 
 
 Considering the critical role of epidemic modeling and the widespread adoption of GNNs in this area, a systematic review of these algorithms is essential for advancing our understanding of the field. We seek to bridge this knowledge gap by providing a thorough overview and categorization of how GNNs are applied in epidemiological studies. Our goal extends beyond merely highlighting current research directions; we aim to uncover new directions for future exploration and enrich both the GNN (or graph machine learning) and epidemiology communities.

 
 
 Table 1. 
 Overview of related surveys .

 
 
 
 
 
   Work    | 
 Tasks Taxonomy | 
 GNNs | 
 Mechanistic Model | 
 Future Work | 

 
 
 
 [ArXiv’21] Baldo et al., 2021     | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 [JBD’21] Shorten et al., 2021     | 
 ✓ | 
 ✓ | 
 ✓ | 
 | 

 
 [RBME’22] Clement et al., 2022     | 
 | 
 | 
 ✓ | 
 ✓ | 

 
 [NC’22] Kamalov et al., 2022     | 
 | 
 ✓ | 
 ✓ | 
 | 

 
 [RBE’22] 1, 1     | 
 ✓ | 
 | 
 | 
 | 

 
 [IJERPH’22] Saleem et al., 2022     | 
 ✓ | 
 | 
 ✓ | 
 | 

 
 [ArXiv’22] Rodríguez et al., 2022     | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 Ours    | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 

 

 
 
 
 Contributions. This paper presents a comprehensive review of GNNs in epidemic modeling. We focus on task categorization, summarizing the latest methodologies, and outlining future directions. We aspire for this work to become a valuable asset for researchers keen on exploring this interdisciplinary research direction. Our contributions are summarized as follows:

 
 
 
 (1) 
 
 Our work offers a comprehensive and pioneering review of GNNs in the context of epidemic modeling. It encompasses a detailed categorization of various tasks, data resources, and graph construction techniques in the field, as elaborated from Sections  3.1 to 3.3 .

 

 (2) 
 
 We offer an in-depth classification of existing methodologies, complemented by a meticulous review in Section  4 .

 

 (3) 
 
 We point out current methods’ limitations and provide prospective directions in Section 5 , thereby facilitating the ongoing progression of the community.

 

 
 
 
 Connections to Existing Surveys. In contrast to previous surveys that explore the intersection of epidemiology and deep learning models, our paper offers a detailed overview specifically of GNNs in epidemic modeling. While preceding surveys predominantly concentrate on predicting COVID-19 outcomes and often overlook the inclusion of GNN-based methodologies Shorten et al., 2021 ; Clement et al., 2022 ; 1, 1 ; Saleem et al., 2022 , a few studies have indeed integrated GNNs Kamalov et al., 2022 ; Baldo et al., 2021 . However, the scope of such works remains relatively narrow, predominantly confined to virus transmission tasks. Certain studies concentrate exclusively on a singular virus  Saleem et al., 2022 or are dedicated to a specific task like forecasting  Rodríguez et al., 2022 . Distinctively, our research is tailored towards GNN-based approaches, covering a broader spectrum of epidemic modeling tasks. Furthermore, our survey presents the latest review of GNN applications in epidemic modeling, offering deeper insights when compared to existing literature. The comparison can be found in Table  1 . Figure  1 shows the structure of this survey.

 
 
 

## 2. preliminaries and definitions

 

### 2.1. Learning on Graph Data

 
 In this paper, we define the graph data as G = ( V , ℰ ) G=(V,\mathcal{E}) ,
with V V representing the node set comprising | V | = N |V|=N nodes. The edge set ℰ ⊆ V × V \mathcal{E}\subseteq V\times V represents the connections between nodes. The feature matrix X = { x 1 , x 2 , … , x N } ⊤ ∈ ℝ N × D \textbf{X}=\{\textbf{x}_{1},\textbf{x}_{2},\ldots,\textbf{x}_{N}\}^{\top}\in\mathbb{R}^{N\times D} includes feature vectors x i \textbf{x}_{i} associated with node v i v_{i} , where D D denotes the feature dimension. The adjacency matrix of G G , denoted by A ∈ ℝ N × N \textbf{A}\in\mathbb{R}^{N\times N} , sets A i ​ j = 1 \textbf{A}_{ij}=1 for any existing edge e i , j ∈ ℰ e_{i,j}\in\mathcal{E} and A i ​ j = 0 \textbf{A}_{ij}=0 otherwise. The normalized adjacency matrix is given by A ^ = D − 1 / 2 A D − 1 / 2 \hat{\textbf{A}}=\textbf{D}^{-1/2}\textbf{A}\textbf{D}^{-1/2} . The degree matrix D , being a diagonal matrix, is characterized by D i , i = ∑ j A i , j \textbf{D}_{i,i}=\sum_{j}\textbf{A}_{i,j} .

 
 
 In the domain of graph learning, the node-level task stands out as a significant area of focus. The objective of this task is to forecast the properties ( i.e . , numerical value or probability) or class of the individual nodes. This process entails training a neural network model that utilizes a subset of nodes with known properties, denoted as 𝒱 L \mathcal{V}_{L} , to infer the properties of other unknown nodes. The essence of this training is encapsulated by optimizing the function:

 

 
 (1) | 
 | 
 min θ ⁡ ℒ ⁡ ( f θ ​ ( G ) ) = ∑ v i ∈ 𝒱 L ℓ ⁡ ( f θ ​ ( X , A ) i , y i ) , \min_{\theta}\mathcal{L}(f_{\theta}(G))=\sum_{v_{i}\in\mathcal{V}_{L}}\ell(f_{\theta}(\textbf{X},\textbf{A})_{i};y_{i}), | 
 | 
 

 Here, the function f θ ​ ( X , A ) f_{\theta}(\textbf{X},\textbf{A}) aims to forecast the property for each node, with y i y_{i} representing the actual state of node v i v_{i} . The discrepancy between the predicted and true properties is quantified using a loss function ℓ ⁡ ( ⋅ , ⋅ ) \ell(\cdot,\cdot) , such as RMSE (Root Mean Square Error).

 
 
 

### 2.2. Graph Neural Networks

 
 Over recent years, GNNs have garnered increasing interest and have been deployed across diverse fields, including bioinformatics, material science, chemistry, and neuroscience  Reiser et al., 2022 ; Wen et al., 2022 ; Wieder et al., 2020 ; Bessadok et al., 2022 . Among them, Graph Convolutional Networks (GCN)  Kipf Welling, 2017 and Graph Attention Networks (GAT)  Veličković et al., 2017 ; Brody et al., 2022 , have advanced the frontier of research on graph-structured data with their sophisticated and effective designs  Wu et al., 2020 ; Dai et al., 2022 ; Liu et al., 2022 . Typically, GNNs aim to learn graph representations, including node embeddings h i ∈ ℝ d \textbf{h}_{i}\in\mathbb{R}^{d} , by utilizing both the structural and feature information of a graph G G . The process within a GNN involves two key operations: message passing and aggregation of neighborhood information. This involves each node in the graph repeatedly collecting and integrating information from its neighbors as well as its own attributes to enhance its representation. The operation of a GNN over L L layers can be described by the following expression:

 

 
 (2) | 
 | 
 h i ( l + 1 ) = σ ⁡ ( h i ( l ) , 𝐴𝐺𝐺 ⁡ ( h j ( l ) , j ∈ A i ) ) , ∀ l ∈ [ L ] , \textbf{h}^{(l+1)}_{i}=\sigma(\textbf{h}^{(l)}_{i},\mathit{AGG}({\textbf{h}^{(l)}_{j};j\in\textbf{A}_{i}})),\forall l\in[L], | 
 | 
 

 where h i ( l ) \textbf{h}^{(l)}_{i} is the representation of node v i v_{i} at layer l l , with h i ( 0 ) = x i \textbf{h}^{(0)}_{i}=\textbf{x}_{i} being the initial node features. Here, A i \textbf{A}_{i} represents the set of neighbors for node v i v_{i} , 𝐴𝐺𝐺 ⁡ ( ⋅ ) \mathit{AGG(\cdot)} denotes a variant-specific aggregation function, and σ \sigma represents an activation function. Following the completion of L L layers of message passing, the resultant node embedding h i h_{i} is passed through a projection function F ⁡ ( h i ) F(\textbf{h}_{i}) to produce the output prediction y ^ i \hat{y}_{i} .

 
 
 

### 2.3. Mechanistic Models

 
 Empirical models  Aleta et al., 2020 ; Chang et al., 2020 in epidemic forecasting utilize historical data to discern patterns and forecast the future spread of diseases. In contrast, mechanistic models  Jiang et al., 2021 ; Yang et al., 2023 provide a detailed framework that explores the biological and social complexities underlying the transmission of infectious diseases, thus exceeding the reliance on historical data inherent to empirical models. Among mechanistic approaches, classic compartmental models  2, 2 ( e.g . , SIR) are particularly notable. These models adeptly simplify the intricate dynamics of disease transmission into digestible components. This simplification facilitates a clearer understanding of infection spread, serving as a valuable tool for both researchers and policymakers.

 
 
 SIR Compartmental Model .
In the domain of epidemiology  Danon et al., 2011 ; Caals et al., 2017 , it is widely hypothesized that the rate at which networks evolve is significantly slower compared to the propagation speed of diseases. This fundamental assumption underpins the adoption of a SIR model  2, 2 ; Dehning et al., 2020 , which is instrumental in accurately capturing the dynamics of epidemic spread. The SIR model categorizes the population into three distinct groups based on their disease status: susceptible (S) to infection, currently infectious (I), and recovered (R), with the latter group being immune to both contraction and transmission of the disease. The SIR model, formulated using ordinary differential equations (ODEs)  Grassly Fraser, 2008 , are as follows:

 

 
 (3) | 
 | 
 d ​ S ​ ( t ) d ​ t = − β ​ S ⁡ ( t ) ​ I ​ ( t ) N , d ​ I ​ ( t ) d ​ t = β S ⁡ ( t ) ​ I ​ ( t ) N − γ I ( t ) , d ​ R ​ ( t ) d ​ t = γ I ( t ) . \begin{gathered}\frac{dS(t)}{dt}=-\beta\frac{S(t)I(t)}{N},\\
\frac{dI(t)}{dt}=\beta\frac{S(t)I(t)}{N}-\gamma I(t),\quad\frac{dR(t)}{dt}=\gamma I(t).\end{gathered} | 
 | 
 

 These equations distribute the total population N N across the aforementioned categories, with the transitions between states regulated by two pivotal parameters: the transmission rate β \beta ( S → I S\to I ) and the recovery rate γ \gamma ( I → R I\to R ). The model posits a quadratic relationship for disease transmission via interactions between susceptible and infectious individuals ( β ​ S ​ ( t ) ​ I ​ ( t ) \beta S(t)I(t) ), alongside a linear recovery mechanism ( γ ​ I ​ ( t ) \gamma I(t) ). By fine-tuning the parameters of the SIR model, it is possible to compute the basic reproduction number R 0 = β / γ R_{0}=\beta/\gamma , serving as a metric for the disease transmission potentials  Han et al., 2023 ; Shah et al., 2020 .

 
 
 Furthermore, utilizing the SIR model within the context of graph-based structures leads to the development of what is referred to as the Network SIR model  Sha et al., 2021 ; Brede, 2012 ; Balcan et al., 2009 ; Venkatramanan et al., 2017 . Within this framework, an infectious node j j has the potential to transmit the infection to another node i i , provided that i i is susceptible and located adjacently to j j ( i.e . , i ∈ A ⁡ ( j ) i\in A(j) ). The probabilities of the node i i being in a susceptible S i ​ ( t ) S_{i}(t) , infected I i ​ ( t ) I_{i}(t) , or recovered R i ​ ( t ) R_{i}(t) state at any given time t t , they are recalculated as follows:

 

 
 (4) | 
 | 
 d ​ S i ​ ( t ) d ​ t = − β ∑ j A i ​ j S i ( t ) I j ( t ) , d ​ I i ​ ( t ) d ​ t = β ∑ j A i ​ j S i ( t ) I j ( t ) − γ I i ( t ) , d ​ R i ​ ( t ) d ​ t = γ I i ( t ) . \begin{gathered}\frac{dS_{i}(t)}{dt}=-\beta\sum_{j}A_{ij}S_{i}(t)I_{j}(t),\\
\frac{dI_{i}(t)}{dt}=\beta\sum_{j}A_{ij}S_{i}(t)I_{j}(t)-\gamma I_{i}(t),\quad\frac{dR_{i}(t)}{dt}=\gamma I_{i}(t).\end{gathered} | 
 | 
 

 Considering the initial phase where S i ​ ( t ) S_{i}(t) is nearly one, the model facilitates the calculation of the basic reproduction number R 0 R_{0} as R 0 = β ​ λ 1 / γ R_{0}=\beta\lambda_{1}/\gamma , with λ 1 \lambda_{1} representing the leading eigenvalue of the adjacency matrix A A .

 
 
 SIR Variants. 
While the SIR model provides a powerful framework for analyzing disease dynamics, its simplicity can sometimes neglect critical factors such as incubation periods, non-permanent immunity, and heterogeneous population interactions. This limitation has spurred the development of SIR variants, which offer a more comprehensive and nuanced understanding of the spread and control of infectious diseases. We briefly outline some of the most commonly used variants:
 i) SEIR Dixon et al., 2018 : The SEIR model extends the basic SIR framework by incorporating an ’Exposed’ (E) compartment. This compartment represents individuals who have been exposed to an infectious disease but are not yet infectious themselves  2, 2 ; Van, 2017 . The model details the transition through the stages according to the sequence: S → E → I → R S\rightarrow E\rightarrow I\rightarrow R .
 ii) SIRD: Enhancing the traditional SIR model, the SIRD variant includes a ’Dead’ (D) compartment, thus adapting the progression to: S → I → R → D S\rightarrow I\rightarrow R\rightarrow D . This modification accounts for individuals who succumb to the disease, providing a more accurate depiction of its mortality impact  Tomy et al., 2022 ; Wang et al., 2022a ; Loli Zama, 2020 .

 
 
 Bridging the Gap. To bridge the gaps between mechanistic models and neural models like GNNs, a Python library called EpiLearn   Liu et al., 2024 is also developed at https://github.com/Emory-Melody/EpiLearn . This library serves as a toolkit for epidemic modeling and analysis, empowering data mining on epidemiology data with machine learning models.

 
 
 Table 2. 
 A brief description of epidemic tasks we categorized .
 
 
 
 
 
   Tasks | 
 Time Interval | 
 Objective | 

 
 
 
 Detection | 
 History-Present | 
 Incident Back-tracing | 

 
 Surveillance | 
 Present | 
 Event Monitoring | 

 
 Prediction | 
 Future | 
 Future Incident Prediction | 

 
 Projection | 
 Future | 
 Change Simulation and Prediction | 

 

 
 
 
 Table 3. 
 Summary of epidemiological tasks and representative GNN-based methods .
 
 
 
 
 
   Task | 
 Paper | 
 Methodology | 
 Hybrid | 
 Graph Construction | 

 
 Detection | 
 SD-STGCN   Sha et al.,2021 | 
 GAT + GRU + SEIR | 
 ✓ | 
 Spatial-Temporal Graph; Static Graph Structure | 

 
 Surveillance | 
 WDCIP   Wang et al.,2023 | 
 GAE | 
 | 
 Spatial Graph; Static Graph Structure | 

 
 GraphDNA  Yang et al.,2022 | 
 GCN + LSTM | 
 | 
 Spatial-Temporal Graph; Dynamic Graph Structure | 

 
 Projection | 
 MMCA-GNNA  Jhun,2021 | 
 GNN + SIR + RL | 
 ✓ | 
 Spatial-Temporal Graph; Static Graph Structure | 

 
 DURLECA   Song et al.,2020 | 
 GNN + RL | 
 | 

 
 IDRLECA   Feng et al.,2023a | 
 GNN + RL | 
 | 
 Spatial-Temporal Graph; Dynamic Graph Structure | 

 
 Prediction | 
 DGDI   Liu et al.,2023b | 
 GCN + Self-Attention | 
 | 
 Spatial Graph; Static Graph Structure | 

 
 DVGSN   Zhang et al.,2023 | 
 GNN | 
 | 
 Temporal-Only Graph; Static Graph Structure | 

 
 STAN   Gao et al.,2021 | 
 GAT + GRU | 
 | 
 Spatial-Temporal Graph; Static Graph Structure | 

 
 MSDNet   Tang et al.,2023 | 
 GAT + GRU + SIS | 
 ✓ | 

 
 SMPNN   Lin et al.,2023 | 
 MPNN + Autoregression | 
 | 

 
 ATMGNN   Nguyen et al.,2023 | 
 MPNN/MGNN + LSTM/Transformer | 
 | 

 
 DASTGN   Pu et al.,2023 | 
 GNN + Attention + GRU | 
 | 

 
 MSGNN   Qiu et al.,2023 | 
 GCN + N-Beats | 
 | 

 
 STEP   Yu et al.,2023 | 
 GCN + Attention + GRU | 
 | 

 
 GSRNN   Li et al.,2019 | 
 GNN + RNN | 
 | 

 
 Mepo GNN  Cao et al.,2023 ; Cao et al.,2022 | 
 (TCN + GCN) + Modified SIR | 
 ✓ | 
 Spatial-Temporal Graph; Dynamic Graph Structure | 

 
 Epi-Cola-GNN   Liu et al.,2023a | 
 Cola-GNN + Modified SIR | 
 ✓ | 

 
 HiSTGNN   Ma et al.,2022 | 
 Hierarchical GNN + Transformer | 
 | 

 
 CausalGNN   Wang et al.,2022 | 
 GNN + SIRD | 
 ✓ | 

 
 ATGCN   Wang et al.,2022b | 
 GNN + LSTM | 
 | 

 
 HierST   Zheng et al.,2021 | 
 GNN + LSTM | 
 | 

 
 RESEAT   Moon et al.,2023 | 
 GNN + Self-Attention | 
 | 

 
 SAIFlu-Net   Jung et al.,2022 | 
 GNN + LSTM | 
 | 

 
 Epi-GNN   Xie et al.,2022 | 
 GCN + Attention + RNN | 
 | 

 
 | 
 Cola-GNN   Deng et al.,2020 | 
 GCN + Attention + RNN | 
 | 

 

 
 
 
 
 

## 3. Taxonomies

 
 In this section, we provide taxonomies for GNNs in epidemic modeling. These methods can be categorized into different types based on their epidemiological tasks, datasets, graph construction, and methodological distinctions. A comprehensive categorization is shown in Appendix 1 and due to page limitation, we provide part of it in Table  3 .

 
 

### 3.1. Epidemiological Tasks

 
 For epidemiological tasks, we provide a taxonomy from the perspective of epidemiologists and categorize the work we investigated into four categories based on researchers’ goals: Detection , Surveillance , Prediction , and Projection . A brief comparison of these tasks is shown in Table  2 ; the detailed explanations and definitions are introduced as follows:

 
 

#### 3.1.1. Detection

 
 The goal of detection tasks is to identify health states, disease spread, or other related incidents that happened at a specific time . In this survey, we incorporate two different detection tasks from the view of graph data: source detection and transmission detection . To formulate a mathematical definition, the temporal network, which consists of sequenced graphs from different time points, is represented as G = { G 0 , G 1 , … , G T } G=\{G_{0},G_{1},\ldots,G_{T}\} . Within a graph G t G_{t} , the states of nodes and edges are represented by S t V S_{t}^{V} and S t ℰ S_{t}^{\mathcal{E}} respectively. Then, the detection task can be expressed as predicting S t V S_{t}^{V} or S t ℰ S_{t}^{\mathcal{E}} given graph G T G_{T} and time point t t .

 
 
 For example, finding patient-zero   Ru et al., 2023a ; Sha et al., 2021 ; Shah et al., 2020 , as a source detection task, is important for identifying the source of disease outbreaks and aims to find a set of nodes V V on graph G 0 G_{0} . In this setting, the problem can also be seen as identifying the state of each node at the initial time point, which is S 0 V S_{0}^{V} .

 
 
 

#### 3.1.2. Surveillance

 
 Surveillance tasks aim at providing timely and accurate information to support decision-making and disease prevention. Since a prompt response is needed, real-time processing ability has been the most important requirement during modeling. Here, we provide a formal definition: given a temporal graph G = { G 0 , G 1 , … , G T } G=\{G_{0},G_{1},\ldots,G_{T}\} , the goal is to identify a target statistic y on graph G T G_{T} at the present moment or in the short term.

 
 
 To illustrate, tasks like detecting infected individuals promptly Song et al., 2023 and estimating infection risks in different locations in real-time  Wang et al., 2023 ; Yang et al., 2022 ; Gouareb et al., 2023 ; Han et al., 2023 can be seen as surveillance tasks, as they their prediction targets lie in present or near future.

 
 
 

#### 3.1.3. Prediction

 
 Similar to surveillance tasks, prediction tasks also aim to forecast epidemic events using historical data. However, unlike surveillance tasks, prediction tasks typically involve longer time spans and do not require real-time processing . Therefore, prediction tasks are more interested in predicting the target at the longer time ahead like T + 1 T+1 instead of at time T T . Due to the large amount of work, we further classify prediction tasks into two categories based on the type of prediction target:

 
 
 
 (1) 
 
 Incidence Prediction. The target of incidence prediction is to provide quantitative results. In epidemic forecasting, incidences can include the number of infections or deaths in the future   Siji et al., 2023 ; Yu et al., 2023a ; Nguyen et al., 2023a ; Croft et al., 2023 ; Pu et al., 2023 ; Qiu et al., 2023 ; Moon et al., 2023a ; Cao et al., 2023a ; Tang et al., 2023a , influenza activity level   Liu et al., 2023c , Influenza-Like Illness (ILI) rates   Zhang et al., 2023a , vaccine hesitancy   Moon et al., 2023b , etc. The prediction of these incidences is important to decision-making, proactive public health planning, and the effective management of infectious diseases and other health challenges.

 

 (2) 
 
 Trend Prediction. Different from incidence prediction task, which focuses on quantitative targets, the target of trend prediction tasks is to identify a higher-level epidemic spreading pattern. For transmission among locations   Liu et al., 2023b , prediction of infection trend can be described as an information retrieving problem and the goal is to predict the next region to be infected given a historic spreading route. For transmission among individuals or groups, the goal usually includes identifying transmission dynamics in emerging high-risk groups   Sun et al., 2023 .

 

 
 
 
 

#### 3.1.4. Projection

 
 In epidemic forecasting, projection tasks are similar to prediction, but with an additional intention to understand epidemic outcomes . These tasks usually require models with the ability to incorporate changes during the evolving of epidemics, such as external interventions and changing of initial states. Most of the projection tasks we collected involve finding the optimal interventions or maximizing influence to achieve targets like curbing the spread of diseases. Influence maximization   Kempe et al., 2003 aims to identify a subset of nodes so that the influence spreads most effectively across the graph, and there have been several early studies in epidemiological tasks, e.g., node importance ranking   Bucur Holme, 2020 ; Holme, 2017 .

 
 
 In this paper, we extend the traditional setting of influence maximization and combine it with intervention strategy tasks to form a more general definition as follows: Given a temporal graph G = { V ⁡ ( t ) , ℰ ⁡ ( t ) } G=\{V(t),\mathcal{E}(t)\} , the states of nodes ∈ V \in V and edges ∈ ℰ \in\mathcal{E} are influenced by strategies defined as P v ​ ( t ) P_{v}(t) and P ℰ ​ ( t ) P_{\mathcal{E}}(t) , which represents strategies on nodes and edges respectively. The goal of the task is to find optimal strategies so that the target is maximized or minimized.

 
 
 For traditional influence maximization tasks and vaccine strategy tasks   Jhun, 2021 , which aim to vaccinate the optimal set of nodes to minimize epidemic damage, strategies are limited to nodes at the starting time point, which is P v ​ ( 0 ) P_{v}(0) . For interventions throughout the period, strategies can include applying quarantine level to nodes at each step   Feng et al., 2023a , which denotes P v ​ ( t ) P_{v}(t) or restricting mobility on edges   Meirom et al., 2021 ; Song et al., 2020 , which denotes P ℰ ​ ( t ) P_{\mathcal{E}}(t) .

 
 
 

#### 3.1.5. Perspectives from Data Scientists.

 
 The above general taxonomy for epidemiological tasks comes from the perspective of epidemiologists. However, it is also feasible to categorize these works from the perspective of data scientists, who focus more on the computational pipeline. Here we provide a further taxonomy from the perspective of model inputs and outputs :

 
 
 
 (1) 
 
 For inputs of all models, they all consist mainly of two parts: node features and the graph structure. Based on the temporality of nodes, we can further categorize these work into spatial-only tasks, temporal-only    Zhang et al., 2023a , and spatial-temporal tasks. In addition, based on the temporality and learnability of graph structure, we can also use static or dynamic features to distinguish these works. A detailed introduction is presented below in Section  3.2 .

 

 (2) 
 
 In terms of model outputs, there are also three categories to summarize these works: scalar , graph , and action sequence . Scalar outputs are usually used in prediction tasks which provide indicators of the epidemic like infected cases. There are also some works that focus on epidemic graph construction and the outputs of their designed models are graphs   Shan et al., 2023 ; Wang et al., 2023 . Finally, the projection tasks we collected usually adopt Reinforcement Learning (RL), which outputs the actions taken at each time step, forming a consecutive action sequence   Jhun, 2021 ; Feng et al., 2023a ; Meirom et al., 2021 ; Song et al., 2020 .

 

 
 
 
 
 

### 3.2. Data Sources

 
 Figure 2. Countries involved in sampling epidemiology data.
 
 
 
 The investigated works in this survey cover a wide range of datasets from different parts of the world, as shown in Figure   2 . However, the majority of research focuses only on COVID-19 while only a few study Influenza-Like Illnesses (ILI) or bacteria   Gouareb et al., 2023 . We categorize these datasets based on their sources:

 
 (1) 
 
 Demographic and Health Records. Epidemic data can be accessed through public databases released by universities, governments, or other organizations. These data usually include demographic information like populations, number of infections, and health records of individuals or groups. During epidemic graph construction, these data are usually used directly as node features or used in the construction of graph structures.

 

 (2) 
 
 Mobility Information. Mobility information can be acquired through websites that record transportation information, maps, or contact records of individuals. This information is usually used to construct the graph structure.

 

 (3) 
 
 Online Search and Social Media. Epidemic information can also be acquired through social media and online search records. The massive search of disease-related questions in a region can indicate potential outbreaks  Lin et al., 2023 , which can then be utilized as node features.

 

 (4) 
 
 Sensors. Multi-modal data can be acquired through sensors like cameras, satellites, radios, etc. These data can also help epidemic tasks like exposure risk prediction using images   Han et al., 2023 . Unlike conventional data sources, sensor data often requires additional preprocessing using specialized models, such as encoding images with techniques like ResNet  He et al., 2016 , before integration as node features.

 

 (5) 
 
 Simulated Data. Besides real-world data, some research   Tang et al., 2023 ; Sun et al., 2023 ; Mežnar et al., 2021 also utilized simulated data for model training and testing. These data often require simulation models like TimeGEO   Jiang et al., 2016 , Independent Contagion Model (ICM)   Murphy et al., 2021 , and also SIR models to generate temporal graphs.

 

 
 
 
 

### 3.3. Graph Construction

 
 For graph construction, we provide a taxonomy based on the dynamicity of nodes and edges as follows.

 
 

#### 3.3.1. Static Node Features.

 
 Static node features typically refer to characteristics that do not change with time. The shape of static features can be represented as ℝ N × h \mathbb{R}^{N\times h} , where h h refers to the number of different features. Besides tasks involving time series, most GNN tasks are using static features. For example, in a contact graph in which individuals are modeled as nodes and contact information represents edges, personal characteristics like infection order, gender, age, and symptom-related information can be used as static features during training and prediction   Song et al., 2023 . However, tasks involving time series can also use static features as additional information.

 
 
 

#### 3.3.2. Dynamic Node Features.

 
 Contrary to static features, dynamic features represent characteristics that change through time. This type of data is commonly seen in time-series forecasting tasks and the models usually require inputs at each time point. Therefore, the shape of the dynamic features can be represented as ℝ N × T × h \mathbb{R}^{N\times T\times h} , where T T refers to the number of time points given. As an example, the number of daily confirmed cases in a region can be seen as dynamic features   Xie et al., 2022 . Although most models take in a single slice of dynamic features at each time point, some models use the entire dynamic features across time T T in a single input   Ru et al., 2023a .

 
 
 

#### 3.3.3. Static Graph Structure.

 
 The construction of a static graph structure typically entails the use of a predefined approach to generate the graph from available data. Once the graph is established, its structure remains unchanged throughout the training iterations or over different time points. For instance, in tasks involving multiple regions, the geographical adjacency A A , is often employed to connect different regions, which are represented as nodes in the graph G G   Pu et al., 2023a ; Yu et al., 2023a . The distance between regions or other features can be considered as the edge weights. Another strategy focuses on exploring human mobility or transitions, e.g . , linking nodes through nearest neighbors in the case of COVID-19 transmission. This method takes into consideration the distribution of the population and individual movements between various locations   Liu et al., 2023b ; Wang et al., 2023 . In research where the node represents an individual, connections between two nodes often utilize contact information, e.g . , identifying contacts at risk of spreading the disease as links between individuals   Gouareb et al., 2023a ; Tomy et al., 2022a .

 
 
 

#### 3.3.4. Dynamic Graph Structure.

 
 Determining the structure of a dynamic graph commonly involves one of two methodologies. One approach is the modification of adjacency relations over time or throughout the virus propagation process. For example,   Meirom et al., 2021 utilize ℰ ​ ( t ) = { e u ​ v ​ ( t ) } \mathcal{E}(t)=\{e_{uv}(t)\} to represent the set of edges at time step t t , which connect individuals based on transmission probability. Another strategy entails the learning of adaptive edges or edge weights during the training phase. Given the dynamic nature of disease transmission, which evolves at each time step, traditional geographical adjacency matrices fall short of accurately representing true connectivity. Recent studies   Wang et al., 2022a ; Deng et al., 2020 ; Cao et al., 2022 have aimed for models to learn an adaptive relationship between nodes. This typically involves initially generating node features via a neural network, followed by the computation of an attention matrix to depict dynamic connectivity, often expressed as 𝐀 t = a i , j ∈ ℝ N × N {\bf A}_{t}=a_{i,j}\in\mathbb{R}^{N\times N} , where a i , j a_{i,j} indicates the influence of node v j v_{j} on node v i v_{i} .

 
 
 
 

### 3.4. Methodological Distinctions

 
 The methodologies of the GNNs in epidemic modeling can be broadly classified into two categories: Neural Models and Hybrid Models .
This classification illuminates the extent of methods that combine computational techniques with epidemiological insights. Both categories employ neural networks, yet they diverge in their underlying principles. (a) Neural Models primarily focus on a data-driven approach and leverage the power of deep learning (i.e., GNNs in our paper) to uncover complex patterns in disease dynamics without explicit encoding of the underlying epidemiological processes. (b) On the other hand, Hybrid Models represent a synergistic fusion of mechanistic epidemiological models with neural networks. This integration allows for the structured, theory-informed insights of mechanistic models to complement the flexible, data-driven nature of GNNs, aiming to deliver predictions that are interpretable, accurate, and grounded in theoretical knowledge.

 
 
 
 

## 4. Methodology

 
 In this section, we provide a detailed illustration of the methods in epidemic modeling, which are divided into the two categories discussed in Section  3.4 : Neural Models and Hybrid Models . While both categories utilize GNNs as the backbone model as illustrated in Figure  3 , they differ in the adoption of the mechanistic models.

 
 
 Figure 3. GNNs aggregate information from neighborhoods. After aggregation, the node/graph representation can be further utilized in the temporal module, employed to predict parameters of mechanistic and probabilistic models, or directly output prediction targets.
 
 
 

### 4.1. Neural Models

 
 When utilizing GNNs for epidemic modeling, numerous studies have exclusively employed GNNs without incorporating mechanistic models into their tasks, which we term Neural Models. These data-driven models can automatically learn features from raw data and capture intricate patterns across diverse inputs. This inherent capability significantly enhances their performance across various tasks.
In this subsection, we delve into the (GNN-based) Neural Models in epidemic modeling, dissected through three perspectives: (a) Spatial Dynamics Modeling , (b) Temporal Dynamics Modeling , and (c) Intervention Modeling .
This categorization is designed to specifically tackle the challenges of modeling the spatial spread, temporal evolution, and the impact of intervention strategies through the advanced capabilities of GNNs.

 
 

#### 4.1.1. Spatial Dynamics Modeling

 
 One advantage of GNNs, e.g., GCN or GAT, is their ability to capture spatial relationships through various aggregation processes, which can analyze and capture the spatial dimensions of disease propagation. Numerous studies represent the inherent structure of geographical data as graph data, denoted as 𝐀 {\bf A} , where nodes depict regions (e.g., cities, neighborhoods, or countries), and edges describe connections between these regions (e.g., roads, flights, or potential vectors for disease transmission).
Subsequently, GNNs are applied to the graph data to uncover complex relationships and dependencies at the regional level, facilitating predictions regarding disease spread across different areas  Song et al., 2023a ; 3, 3 ; Nguyen et al., 2023 ; Gouareb et al., 2023a ; La et al., 2021 .

 
 
 In the context of GNN modeling, the significance of edge weights is paramount, as they encapsulate the intensity and nature of interactions. Within epidemiological studies, these weights are often derived from the mobility or social connectedness between regions   Lin et al., 2023 ; Panagopoulos et al., 2021 . For instance, studies such as   Song et al., 2020 ; Cao et al., 2023 utilize Origin-Destination (OD) flows to quantify inter-regional mobility, thereby dynamically capturing the intensity of transmission. To further enhance the spatial context of each node within the graph, some research advocates for the implementation of positional encoding techniques   Wang et al., 2022c ; 4, 4 . These techniques are designed to augment the nodes’ spatial awareness. For example, Liu et al.   Liu et al., 2023b introduced a unique encoding for each location, denoted as P ​ E ​ ( k ) PE(k) , with even and odd elements represented by sin ⁡ ( k / 10000 i / L ) \sin(k/10000^{i/L}) and cos ⁡ ( k / 10000 i − 1 / L ) \cos(k/10000^{i-1/L}) respectively, where L L denotes the dimension of the encoding.

 
 
 While GNNs have shown success in modeling spatial relations, challenges arise when dealing with varying input data. Specifically, the absence of direct structural information and the introduction of more complex structural information pose additional difficulties during modeling. To tackle these challenges, several studies have attempted solutions, as outlined below.

 
 
 Adaptive Structure Learning. Although GNNs possess the inherent capability to learn the spatial characteristics of disease dissemination, there are occasions when adjacency relationship information is not available in the real world, often due to data scarcity. To overcome this challenge, several studies highlight the importance of learning an adaptive structure throughout the training process Zheng et al., 2021 ; Wang et al., 2022b . For instance, Wang et al.   Wang et al., 2022b introduced a graph structure learning module, denoted as f θ f_{\theta} . This module is designed to calculate node similarities, thereby representing spatial relationships as follows:

 

 
 (5) | 
 | 
 𝐌 1 = tanh ( f θ 1 ( α 𝐗 1 ) ) , 𝐌 2 = tanh ( f θ 2 ( α 𝐗 2 ) ) , 𝐀 = ReLU ​ ( tanh ⁡ ( α ⁡ ( 𝐌 2 ​ 𝐌 2 ⊤ − 𝐌 1 ​ 𝐌 1 ⊤ ) ) ) , \begin{gathered}\small\mathbf{M}_{1}=\tanh(f_{\theta_{1}}(\alpha\mathbf{X}_{1})),\hskip 9.24994pt\mathbf{M}_{2}=\tanh(f_{\theta_{2}}(\alpha\mathbf{X}_{2})),\\
\mathbf{A}=\text{ReLU}(\tanh(\alpha(\mathbf{M}_{2}\mathbf{M}_{2}^{\top}-\mathbf{M}_{1}\mathbf{M}_{1}^{\top}))),\end{gathered} | 
 | 
 

 where 𝐗 1 \mathbf{X}_{1} and 𝐗 2 \mathbf{X}_{2} are randomly initialized, learnable node embeddings, while α \alpha represents a hyper-parameter. Shan et al.   Shan et al., 2023 employed a method to estimate the graph Laplacian from COVID-19 data through convex optimization of derived eigenvectors. This approach aims to identify dynamic patterns of pandemic spread among countries by analyzing their structural relationships. Additionally, inspired by recent advancements in attention-based mechanisms Vaswani et al., 2023 ; Turner, 2024 ; Soydaner, 2022 , a considerable portion of research suggests the use of an attention matrix to illustrate the relationships between nodes Deng et al., 2020 ; Wang et al., 2022a ; Cui et al., 2021 . Notably, Cola-GNN Deng et al., 2020 pioneers the application of additive attention in learning the adaptive structure, which is defined as follows:

 

 
 (6) | 
 | 
 a i , j = 𝐯 T ​ g ​ ( 𝐖 s ​ 𝐡 i + 𝐖 t ​ 𝐡 j + 𝐛 s ) + b v , a_{i,j}=\mathbf{v}^{T}g(\mathbf{W}^{s}\mathbf{h}_{i}+\mathbf{W}^{t}\mathbf{h}_{j}+\mathbf{b}^{s})+b^{v}, | 
 | 
 

 where g g is an activation function, 𝐖 s , 𝐖 t ∈ ℝ d × D \mathbf{W}^{s},\mathbf{W}^{t}\in\mathbb{R}^{d\times D} , 𝐛 s ∈ ℝ d \mathbf{b}^{s}\in\mathbb{R}^{d} , and b v ∈ ℝ b^{v}\in\mathbb{R} are trainable parameters, with d d as a hyperparameter controlling the dimensions of these parameters. a i , j a_{i,j} reflects the impact of location j j on location i i . This approach allows for dynamic adaptation to changes in graph structure, effectively capturing asymmetric and complex viral transmission patterns.

 
 
 Multi-Scale Modeling. Previous approaches typically operate at a singular level, overlooking the multifaceted nature of real-world epidemiological data, which encompasses multiple scales such as country, state, and community levels. For epidemiological tasks, multi-scale modeling is imperative for capturing the dynamics of disease spread across these varied levels, from individual behaviors to global dissemination, ensuring a more comprehensive analysis. HierST Zheng et al., 2021 leverages multi-scale modeling to effectively capture the spread of COVID-19 across different administrative levels by constructing a unified graph, which encapsulates the spatial correlation dynamics both within and between these levels. To further address both local interactions and long-range dependencies, MSGNN Qiu et al., 2023 is meticulously designed to integrate influences from both immediate and broader regions on disease transmission, enhancing its effectiveness in cross-scale epidemiological dynamics.

 
 
 

#### 4.1.2. Temporal Dynamics Modeling

 
 The temporal dynamics in epidemiological models are pivotal for capturing the evolution of disease spread, reflecting changes in infection rates, recovery rates, and other critical parameters over time. These models typically conceptualize graphs as spatio-temporal networks, underscoring the significance of temporal data in comprehending disease dynamics, and forecasting future trends  Zhang et al., 2023 ; Siji et al., 2023a . A particular strand of research utilizes RNN-based models ( e.g . , LSTM or GRU), as mechanisms to extract node features. These features are then incorporated into the graph convolution process Liu et al., 2021 ; Zheng et al., 2021 . A simple way Kapoor et al., 2020 to achieve this by executing the concat operator:

 

 
 (7) | 
 | 
 h = MLP ​ ( x t | x t − 1 ​ | … | ​ x t − d ) \textbf{h}=\text{MLP}(\textbf{x}_{t}|\textbf{x}_{t-1}|...|\textbf{x}_{t-d}) | 
 | 
 

 where h is simply the output of an MLP (Multilayer Perceptron) over the node temporal features x at time t t reaching back d d days.
Another surge of approaches first executes graph spatial convolution in each time step separately, and then leverages all outputs of the GNNs as the input of the temporal module and utilizes them for final downstream tasks like the prediction Gao et al., 2021 ; Yu et al., 2023 ; Moon et al., 2023c ; Duarte et al., 2023 . STEP Yu et al., 2023 execute the multi-layers graph convolution operation to get all node embedding h , and then leverage the GRU to
get the final output:

 

 
 (8) | 
 | 
 h t = z t ∘ h t − 1 + ( 1 − z t ) ∘ h t ′ , \textbf{h}_{t}=\textbf{z}_{t}\circ\textbf{h}_{t-1}+(1-\textbf{z}_{t})\circ\textbf{h}^{\prime}_{t}, | 
 | 
 

 where h t \textbf{h}_{t} is the final result, and z t \textbf{z}_{t} is the result of the update gate, which controls the inflow of information in the form of gating. The Hadamard product of z t \textbf{z}_{t} and h t − 1 \textbf{h}_{t-1} represents the information retained to the final memory at the previous timestep.

 
 
 In contrast to the initial two methodologies, numerous studies achieve their final output by iteratively layering GNN and temporal models Sesti et al., 2021 . Some work  Guo Yang, 2021 ; Sha et al., 2021 advocate for the employment of Spatio-Temporal Graph Neural Networks (STGNNs)  Hu et al., 2022 ; Zhou et al., 2021 ; Wu et al., 2021 to extract insights from multivariate spatiotemporal epidemic graphs. An STGNN integrates many ST-Conv blocks, which comprise a spatial layer flanked by two temporal layers. Each temporal layer features a 1-D CNN operating along the time axis, followed by a Gated Linear Unit (GLU), to delineate the temporal dynamics. The spatial layer, on the other hand, utilizes a GCN based on the Chebyshev polynomials approximation Defferrard et al., 2016 ; He et al., 2022 for spatial analysis.
To further refine the understanding of spatial dynamics during disease evolution, RESEAT   Moon et al., 2023 proposes the continuous maintenance and adaptive updating of an attention matrix. This process aims to capture regional interrelationships throughout the entirety of the input data period:

 

 
 (9) | 
 | 
 t ​ p i , j t ​ ( A ) = A i t ⋅ A j t , A i , j t + 1 = softmax ​ ( a i , j t + 1 + t ​ p i , j t ​ ( A ) ) , A ​ t ​ t i = ∑ j = 1 N A i , j t × x j . \begin{gathered}tp^{t}_{i,j}(\textbf{A})=\textbf{A}^{t}_{i}\cdot\textbf{A}^{t}_{j},\\
\textbf{A}^{t+1}_{i,j}=\text{softmax}(a^{t+1}_{i,j}+tp^{t}_{i,j}(\textbf{A})),\\
Att_{i}=\sum_{j=1}^{N}\textbf{A}^{t}_{i,j}\times\textbf{x}_{j}.\end{gathered} | 
 | 
 

 The A i , j t ∈ ℝ \textbf{A}^{t}_{i,j}\in\mathbb{R} denotes the attention weight between regions i i and j j at time step t t , and A ​ t ​ t i Att_{i} is employed as the final feature for the node v i v_{i} . Through this mechanism, RESEAT adeptly captures not only temporal patterns but also the dynamically evolving regional interrelationships. To integrate explicit observations with implicit factors over time, Cui et al.   Cui et al., 2021 introduced a new case prediction methodology within an encoder-decoder framework. They contend that relying solely on observed case data, which can be inaccurate, may impair prediction performance. Accordingly, their proposed decoder is designed to incorporate inputs of new cases and deaths, thereby dynamically reflecting temporal changes.

 
 
 

#### 4.1.3. Intervention Modeling

 
 Intervention modeling offers a detailed perspective on epidemic spread by simulating the behaviors and interactions of individuals within a network based on intervention strategies.
This method provides an intricate view of individual actions, mobility patterns, and the likelihood of disease transmission. When combined with GNN, this approach enhances the model’s capability to represent the diversity and complexity inherent in real-world social networks, augmenting the efficacy of intervention strategies. Song et al.   Song et al., 2020 introduced a reinforcement learning framework that dynamically optimizes public health interventions to strike a balance between controlling the epidemic and minimizing economic impacts, to reduce infection rates while maintaining economic activities.
To delve deeper into the individual underlying dynamics, Meirom et al.   Meirom et al., 2021 proposed a dual GNN module strategy. One module updates the node representations according to dynamic processes, while the other manages the propagation of long-range information. Subsequently, they employ RL to modulate the dynamics of social interaction graphs and perform intervention actions on them. This approach aims to indirectly curb epidemic spread by strategically altering network structures, thus avoiding direct intervention in the disease process.

 
 
 IDRLECA Feng et al., 2023 embodies a novel integration, combining an infection probability model with an innovative GNN design. The infection probability model calculates the current likelihood of each individual’s infection status. This information, along with personal health and movement data, is utilized to forecast virus transmission through human contacts using the GNN:

 

 
 (10) | 
 | 
 p i , infected = 1 − p ^ i , healthy = 1 − p i , healthy , T × ( 1 − p c ) contacts , \begin{gathered}p_{i,\text{infected}}=1-\hat{p}_{i,\text{healthy}}=1-p_{i,\text{healthy},T}\times(1-p_{c})^{\text{contacts}},\end{gathered} | 
 | 
 

 here p i , infected p_{i,\text{infected}} represents the probability that individual i i is infected, while p i , healthy , T p_{i,\text{healthy},T} denotes the baseline probability of individual i i being healthy at time T T , before accounting for contact-related risks. p c p_{c} refers to the probability of infection from a single contact. Additionally, a custom reward function is designed to simultaneously minimize the spread of infections and the associated costs, striking a balance between health objectives and economic considerations:

 

 
 (11) | 
 | 
 r = − ( exp ⁡ ( Δ ​ I θ I ) + exp ⁡ ( Δ ​ Q θ Q ) ) . \begin{gathered}r=-\left(\exp\left(\frac{\Delta I}{\theta_{I}}\right)+\exp\left(\frac{\Delta Q}{\theta_{Q}}\right)\right).\end{gathered} | 
 | 
 

 This function considers the change in the number of infections ( Δ ​ I \Delta I ) and the cost of mobility interventions ( Δ ​ Q \Delta Q ), with θ I \theta_{I} and θ Q \theta_{Q} acting as soft thresholds for these changes.

 
 
 
 

### 4.2. Hybrid Models

 
 Unlike Neural Models described above, Hybrid Models effectively combine the predictive capabilities of neural networks with the foundational principles of mechanistic models, thereby enhancing both the accuracy and interpretability of disease forecasting. This integration can be further classified into two categories: Parameter Estimation for Mechanistic Model and Mechanistic Informed Neural Model . The former approach allows these hybrid systems to adapt to evolving epidemic patterns by dynamically estimating parameters of mechanistic models using neural networks, ensuring that simulations remain closely aligned with current trends. Conversely, the latter approach involves incorporating priors from mechanistic models into neural networks, enriching these models with domain-specific knowledge, and directing the learning process to more accurately reflect plausible disease dynamics. This synergistic methodology not only capitalizes on the data-driven strengths of neural models but also firmly anchors predictions within the framework of epidemiological theory, presenting a comprehensive and informed strategy for predicting epidemic spread.

 
 

#### 4.2.1. Parameter Estimation for Mechanistic Models

 
 This line of research highlights that hybrid models, which integrate neural networks, dynamically adjust the parameters of mechanistic models. This combination enables the analysis of real-time data, thus informing and refining mechanistic models to ensure their simulations accurately mirror the dynamics of actual epidemics Jhun, 2021 ; Tang et al., 2023 ; La et al., 2021 . Notably, studies like La et al., 2021 ; Xie et al., 2022a employ GNNs to estimate the contact (transmission) rate, β \beta , and to monitor the epidemic evolution through the implementation of the SIR model. Further, research  Gao et al., 2021 ; Tang et al., 2023 estimates both the transmission rates β \beta and recovery rates γ \gamma by leveraging outputs from the GNN. This methodology initiates with the utilization of GRU to derive node embeddings h , which subsequently facilitate the calculation of parameters:

 

 
 (12) | 
 | 
 β , γ = MLP 1 ​ ( h ) Δ ​ I , Δ ​ R = MLP 2 ​ ( h ) , \beta,\gamma=\text{MLP}_{1}(\textbf{h})\quad\quad\Delta I,\Delta R=\text{MLP}_{2}(\textbf{h}), | 
 | 
 

 where Δ ​ I \Delta I and Δ ​ R \Delta R denote the daily increases in the number of infected and recovered cases, respectively. To enhance the model’s ability to leverage the dynamics of the pandemic for regulating longer-term progressions, the researchers utilize the predicted transmission and recovery rates to calculate predictions based on the dynamics of the disease spread:

 

 
 (13) | 
 | 
 Δ ​ I ^ d = [ Δ ​ I ^ t + 1 d , Δ ​ I ^ t + 2 d , … , Δ ​ I ^ t + L p d ] ,  each  ​ Δ ​ I ^ i d = β ​ S i − 1 − γ ​ I i − 1 = β ⁡ ( N p − I ^ i − 1 d − R ^ i − 1 d ) − γ ​ I ^ i − 1 d , Δ ​ R ^ d = [ Δ ​ R ^ d t + 1 , Δ ​ R ^ d t + 2 , … , Δ ​ R ^ d t + L p ] ,  each  Δ ​ R ^ d i = γ I d i − 1 , \begin{gathered}\hat{\Delta I}^{d}=\left[\hat{\Delta I}^{d}_{t+1},\hat{\Delta I}^{d}_{t+2},...,\hat{\Delta I}^{d}_{t+L_{p}}\right],\\
\text{ each }\hat{\Delta I}^{d}_{i}=\beta S_{i-1}-\gamma I_{i-1}=\beta(N_{p}-\hat{I}^{d}_{i-1}-\hat{R}^{d}_{i-1})-\gamma\hat{I}^{d}_{i-1},\\
\hat{\Delta R}^{d}=\left[\hat{\Delta R}^{d}_{t+1},\hat{\Delta R}^{d}_{t+2},...,\hat{\Delta R}^{d}_{t+L_{p}}\right],\text{ each }\hat{\Delta R}^{d}_{i}=\gamma I^{d}_{i-1},\\
\end{gathered} | 
 | 
 

 where I i − 1 d I^{d}_{i-1} and R i − 1 d R^{d}_{i-1} are iteratively calculated using the ground truth of the infected and recovered cases from the day preceding the current prediction window. N p N_{p} represents the population size of the current location, t t denotes the time steps, and L p L_{p} refers to the number of days into the future for which predictions are made. Ultimately, the researchers propose two loss functions to consider both the short-term and long-term progression of the pandemic.

 
 
 To go beyond single-region recognition, MepoGNN Cao et al., 2023 ; Cao et al., 2023b extends the SIR model to the metapopulation variant Rodríguez et al., 2022 ; Balcan et al., 2009 ; Mousavi et al., 2012 , accommodating heterogeneity within populations and incorporating human mobility to model the spread between sub-populations:

 

 
 (14) | 
 | 
 d ​ S i ​ ( t ) d ​ t = − β i ( t ) ⋅ S i ( t ) ∑ j = 1 N ( h j ​ i ​ ( t ) P j + h i ​ j ​ ( t ) P i ) I j ( t ) , d ​ I i ​ ( t ) d ​ t = β i ( t ) ⋅ S i ( t ) ∑ j = 1 N ( h j ​ i ​ ( t ) P j + h i ​ j ​ ( t ) P i ) I j ( t ) − γ i ( t ) ⋅ I i ( t ) , d ​ R i ​ ( t ) d ​ t = γ i ​ ( t ) ⋅ I i ​ ( t ) . \begin{gathered}\frac{dS_{i}(t)}{dt}=-\beta_{i}(t)\cdot S_{i}(t)\sum_{j=1}^{N}\left(\frac{h_{ji}(t)}{P_{j}}+\frac{h_{ij}(t)}{P_{i}}\right)I_{j}(t),\\
\frac{dI_{i}(t)}{dt}=\beta_{i}(t)\cdot S_{i}(t)\sum_{j=1}^{N}\left(\frac{h_{ji}(t)}{P_{j}}+\frac{h_{ij}(t)}{P_{i}}\right)I_{j}(t)-\gamma_{i}(t)\cdot I_{i}(t),\\
\frac{dR_{i}(t)}{dt}=\gamma_{i}(t)\cdot I_{i}(t).\end{gathered} | 
 | 
 

 MepoGNN incorporates a spatio-temporal GNN designed to learn three dynamic parameters: β i ​ ( t + 1 ) \beta_{i}(t+1) , γ i ​ ( t + 1 ) \gamma_{i}(t+1) , and H ​ ( t ) \textbf{H}(t) , throughout the evolving timeframe. Here, H ​ ( t ) \textbf{H}(t) signifies the epidemic propagation matrix, capturing human mobility between regions, represented by { h ( t ) i ​ j | i , j ∈ { 1 , 2 , … , N } } \{h(t)_{ij}|i,j\in\{1,2,...,N\}\} . The model thereby generates its final prediction of daily confirmed cases as follows:

 

 
 (15) | 
 | 
 y i ​ ( t ) = β i ​ ( t ) ​ ∑ j = 1 N ( h j ​ i ​ ( t ) P j + h i ​ j ​ ( t ) P i ) ​ I j ​ ( t ) , \begin{gathered}y_{i}(t)=\beta_{i}(t)\sum_{j=1}^{N}\left(\frac{h_{ji}(t)}{P_{j}}+\frac{h_{ij}(t)}{P_{i}}\right)I_{j}(t),\end{gathered} | 
 | 
 

 Recent work Liu et al., 2023a integrates the Cola-GNN Deng et al., 2020 framework with the SIR model through the development of Epi-Cola-GNN, introducing a mobility matrix Π \Pi to capture the dynamics of infectious disease spread across different locations. Within this matrix, π i ​ j \pi_{ij} quantifies the intensity of human mobility from location i i to location j j , offering a nuanced perspective on the spatial transmission of diseases. This incorporation leads to a modification in the representation of infectious cases within the SIR model framework:

 

 
 (16) | 
 | 
 d ​ I i d ​ t = β i ​ I i − γ i ​ I i − ∑ j = 1 , j ≠ i N π i , j ​ I i + ∑ j = 1 , j ≠ i N π j , i ​ I j . \begin{gathered}\frac{dI_{i}}{dt}=\beta_{i}I_{i}-\gamma_{i}I_{i}-\sum_{j=1,j\neq i}^{N}\pi_{i,j}I_{i}+\sum_{j=1,j\neq i}^{N}\pi_{j,i}I_{j}.\end{gathered} | 
 | 
 

 Furthermore, they introduce the concept of the Next-Generation Matrix (NGM) Diekmann et al., 2010 , which provides a clearer epidemiological interpretation and more effectively supports both intra-location spread and inter-location transmission influenced by human mobility.

 
 
 Instead of simply estimating the rate indicator, EpiGCN Han et al., 2023a innovatively employs three distinct linear layers to transform the node feature into the SIR state, enhancing the model awareness of the available data:

 

 
 (17) | 
 | 
 S v = σ ( W s ⋅ h v + b s ) , I v = σ ( W i ⋅ h v + b i ) , R v = σ ( W r ⋅ h v + b r ) . \begin{gathered}S_{v}=\sigma(\textbf{W}_{s}\cdot\textbf{h}_{v}+b_{s}),I_{v}=\sigma(\textbf{W}_{i}\cdot\textbf{h}_{v}+b_{i}),R_{v}=\sigma(\textbf{W}_{r}\cdot\textbf{h}_{v}+b_{r}).\end{gathered} | 
 | 
 

 Subsequently, they refine the process of updating the SIR Eq.   3 model and introduce a novel SIR message-passing mechanism that aggregates information from neighboring nodes. This approach modifies the conventional SIR update equation to incorporate spatial dependencies and interactions within the network:

 

 
 (18) | 
 | 
 S v = S v − W t ​ r ​ a ​ n ⋅ concat ​ ( S v , ∑ w ∈ A v e w ​ I w ) , I v = I v + W t ​ r ​ a ​ n ⋅ concat ​ ( S v , ∑ w ∈ A v e w ​ I w ) − W r ​ e ​ c ​ o ​ v ​ I v , R v = R v + W r ​ e ​ c ​ o ​ v ⋅ I v , \begin{gathered}S_{v}=S_{v}-\textbf{W}_{tran}\cdot\text{concat}\left(S_{v},\sum_{\textbf{w}\in\textbf{A}_{v}}e_{w}I_{w}\right),\\
I_{v}=I_{v}+\textbf{W}_{tran}\cdot\text{concat}\left(S_{v},\sum_{\textbf{w}\in\textbf{A}_{v}}e_{w}I_{w}\right)-\textbf{W}_{recov}I_{v},\\
R_{v}=R_{v}+\textbf{W}_{recov}\cdot I_{v},\end{gathered} | 
 | 
 

 here W t ​ r ​ a ​ n ∈ ℝ 2 ​ D × D \textbf{W}_{tran}\in\mathbb{R}^{2D\times D} and W r ​ e ​ c ​ o ​ v ∈ ℝ D × D \textbf{W}_{recov}\in\mathbb{R}^{D\times D} denote the matrices for linear transformations corresponding to the transmission and recovery processes, respectively. Ultimately, the SIR states are concatenated and transformed to align with the prediction objectives:

 

 
 (19) | 
 | 
 y v = softmax ​ ( W o ​ u ​ t ​ p ​ u ​ t ⋅ concat ​ ( S v , I v , R v ) ) . \begin{gathered}y_{v}=\text{softmax}\left(\textbf{W}_{output}\cdot\text{concat}(S_{v},I_{v},R_{v})\right).\end{gathered} | 
 | 
 

 
 
 

#### 4.2.2. Mechanistic-Informed Neural Models

 
 Unlike previous methods wherein neural networks dynamically adjust the parameters of mechanistic models based on data inputs, mechanistic-informed neural models utilize domain knowledge from mechanistic models to inform the architecture and learning processes of GNNs. This strategy flexibility allows for a swift adaptation to changing conditions, markedly improving the accuracy of forecasts and the effectiveness of interventions. Certain studies Mežnar et al., 2021 ; Wang et al., 2023 ; Tomy et al., 2022a utilize the SIR model to generate target data by simulating epidemic spreads from individual nodes, which are then employed to train GNNs for downstream tasks. In the context of source detection tasks, such as those discussed in Ru et al., 2023 ; Shah et al., 2020 ; Sha et al., 2021 , one-hot encoded node states x i t ∈ { 0 , 1 } M x^{t}_{i}\in\{0,1\}^{M} , with M M representing the number of possible states, are used as inputs for the GNN, where the states are defined as either {S, E, I, R} or {S, I, R}.
 Song et al.   Song et al., 2020 integrated SIHR (a variant of SIR)  Kermack McKendrick, 1927 simulation environment with the RL framework, providing a dynamic model of epidemic progression for the RL agent. This capability allows the agent to account for individuals who are hospitalized, enabling the dynamic modification of mobility control policies.

 
 
 To explicitly capture causal dynamics, CausalGNN  Wang et al., 2022 introduces a novel approach to causal modeling by leveraging causal features Q t = ( q i , t ) ∈ ℝ N × 4 \textbf{Q}_{t}=(q_{i,t})\in\mathbb{R}^{N\times 4} , with q i , t : S i ​ ( t ) , I i ​ ( t ) , R i ​ ( t ) , D i ​ ( t ) q_{i,t}:S_{i}(t),I_{i}(t),R_{i}(t),D_{i}(t) representing the cumulative number of individuals in each state of the SIRD model. A causal encoder is then designed to transform these causal features into node embeddings, operating as follows:

 

 
 | 
 H c t = tanh ⁡ ( Q t ​ W e t + b e t ) ∈ ℝ N × D , \textbf{H}^{t}_{c}=\tanh(\textbf{Q}_{t}\textbf{W}^{t}_{e}+b^{t}_{e})\in\mathbb{R}^{N\times D}, | 
 | 
 

 where W e t ∈ ℝ 4 × D \textbf{W}^{t}_{e}\in\mathbb{R}^{4\times D} and b e t ∈ ℝ D b^{t}_{e}\in\mathbb{R}^{D} denote model parameters, and these causal features are intended to be concatenated with other node embeddings. The spatial GNN architecture also infers the SIRD rates β i ​ ( t ) , γ i ​ ( t ) , ρ i ​ ( t ) \beta_{i}(t),\gamma_{i}(t),\rho_{i}(t) by providing P t = ( p i , t ) ∈ ℝ N × 3 \textbf{P}_{t}=(p_{i,t})\in\mathbb{R}^{N\times 3} , which are subsequently utilized for SIRD model updates.

 
 
 
 
 

## 5. Future Work

 
 While many challenges have been addressed in the application of GNNs within epidemic modeling, this field continues to confront various difficulties, both explored and unexplored. In this section, we will examine these challenges and highlight potential avenues for future research.

 
 

### 5.1. Epidemic at Scales

 
 Multi-scale data are crucial in epidemiology because they offer comprehensive insights into both intra-region and inter-region relationships, thus aiding in the more accurate modeling of disease spread. Presently, several studies have acknowledged this importance and initiated the integration of multi-scale data into their frameworks   Tang et al., 2023a ; Wang et al., 2023a ; Qiu et al., 2023a ; Zheng et al., 2021 . Although these efforts have yielded models capable of accommodating multi-scale data, existing approaches are limited to processing only two predefined scales, such as county-level and state-level data. Looking ahead, there is growing anticipation for the development of novel models capable of incorporating data across multiple dynamic scales and adaptable to diverse epidemiological tasks.

 
 
 Meanwhile, scalability must also be considered for numerous reasons: (1) A smaller granularity results in the expansion of graph data. (2) Some tasks require real-time processing   Wang et al., 2023a . While the number of countries or provinces can be small, the graph for individuals or other necessary parts in epidemic models can be extremely large, e.g., contact information graphs in metropolises, which could make the current methods very time-consuming. Furthermore, the use of multi-scale data and the requirements for real-time processing make the problem even harder.

 
 
 

### 5.2. Cross-Modality in Epidemiology

 
 The integration of multi-modal data in epidemiological tasks offers a powerful approach for enhancing our understanding of disease transmission dynamics, improving predictive accuracy, enabling early detection and intervention, conducting comprehensive risk assessments, and fostering interdisciplinary collaboration to address public health challenges more effectively. Data from different modalities can not only serve as augmentations for each other but also compensate for noise from single-modality data. In recent years, some works have successfully incorporated multi-modality in GNNs. Although GNNs are very suitable for information aggregation and handling multi-modality data, there has not been much work exploring the multi-modality of GNNs in an epidemiology setting. Some related works   Lin et al., 2023a ; Han et al., 2023 have utilized unstructured data like textual or image data to construct node features. However, there is no cross-modality in their work in terms of node features.

 
 
 

### 5.3. Epidemic Diffusion Process

 
 The diffusion process, which is the key component in epidemiological tasks, can be both spatial and temporal. All GNN-based methods discussed above involve information aggregation at one or several time points in a discrete manner. Nevertheless, in the real world, disease spreading is a continuous process, which is incompatible with current methods. To address this problem, Continuous GNNs   Xhonneux et al., 2020 ; Chamberlain et al., 2021 ; Poli et al., 2019 , inspired by Neural ODE   Chen et al., 2018 , can be applied to model the continuous spreading process.

 
 
 Another problem lies in that both disease spreading and infection take time, and they can happen asynchronously. One related work   Pu et al., 2023 considers different time-space effects and models the effects using the attention mechanism. However, it is still done in a discrete manner, creating gaps in the real-world transmission process.

 
 
 

### 5.4. Interventions for Epidemics

 
 In epidemiology, control measures are vital for controlling disease spread and safeguarding public health. They include intervention strategies like vaccination, quarantine, and public health education to limit transmission and minimize outbreaks, ultimately saving lives and reducing the burden on healthcare systems   Bhattacharyya Agarwalla, 2023 ; Jamison, 2007 ; Luo et al., 2021 .
Among the methods mentioned in this paper, most of the research incorporates intervention strategy in agent-based models   Feng et al., 2023a ; Meirom et al., 2021 ; Song et al., 2020 or other neural models   Pu et al., 2023a . Generally, the interventions in these methods include deleting nodes, altering nodes, and altering edge weights. However, each method only includes one type of intervention, either node-level or edge-level. In the real world, however, interventions can happen at different graph levels and also at different scales. To better model the real situation, multi-level and multi-scale interventions need to be introduced.

 
 
 

### 5.5. Generating Explainable Predictions

 
 The study of epidemic modeling not only aims for accurate predictions but also emphasizes interpretability. Ideally, experts will rely on epidemic models’ predictions to make informed decisions. However, relying on a model’s predictions becomes risky if the model cannot provide confidence in its forecasts, given the significant consequences of these decisions. Therefore, interpretability is essential and offers various benefits, including understanding disease dynamics, identifying risk factors, and providing measures of uncertainty. Despite the crucial role of interpretability, neural models investigated thus far have not placed significant emphasis on this aspect, with hybrid models primarily relying on mechanistic models to provide explanations. In recent years, there have been some approaches aimed at providing interpretability for general Graph Neural Networks  Dai Wang, 2021 ; Ying et al., 2019 ; Yuan et al., 2022 ; Nian et al., 2024 , which also hold potential usefulness in epidemiological settings.

 
 
 

### 5.6. Handling Challenges from Epidemic Data

 
 The idea of Data-Centric AI (DCAI) has grown more and more important in recent years   Zha et al., 2023 , which inspired people to pay attention not only to modeling but also to processing data itself. In epidemiological tasks, there are also many challenges originating from data, e.g., noise, incomplete data, privacy, etc. Although there have not been many works addressing these challenges in GNN-related epidemiological tasks, we expect future works will try to tackle this problem by proposing model-centric and data-centric methods.

 
 
 Noisy Data .
 Rodríguez et al.   Rodríguez et al., 2022a introduced several sources of data in epidemiology and noise naturally exists in these data. For example, while social media can provide information on epidemic progress, it may also create significant noise by spreading rumors and misinformation. For GNNs in epidemiology, noise can exist both at node-level and edge-level. So far, the denoising mechanisms for GNNs in epidemiology have remained to be studied. Fortunately, there has been a wide range of studies on the robustness of GNNs  Jin et al., 2020 ; Jin et al., 2021 ; Zhu et al., 2021 ; Liu et al., 2021a , which may be adapted to epidemic data.

 
 
 Incomplete Data . In addition to noises, epidemic data can also be incomplete. The data-gathering process is not always perfect and can not guarantee the accurate collection of features for every node. This problem may be mitigated during modeling because the blank features can be replaced by the aggregated neighbor features   Tomy et al., 2022a and simulation or interpolation can also be used to infer missing features at some time points. Nevertheless, there is not much work studying the influence of incomplete data in an epidemiology setting while using GNN models.

 
 
 Privacy Protection . In the real world, epidemic data typically includes sensitive information such as individual health status, location, and potentially identifiable details, which, if mishandled, can lead to severe privacy violations and undermine public trust in health systems   Selgelid, 2016 ; Caals et al., 2017 ; Liu Yang et al., 2022 . In contemporary research, methods commonly utilize all available epidemic data that span countries or regions. However, this reliance on large-scale data may heighten awareness among government entities and departments regarding data privacy. Owing to the growing focus on data sensitivity, stringent legislations   May Sell, 2006 ; Investigators, 2016 have been introduced to regulate data collection and utilization. Therefore, future work should take more privacy problems into consideration.
Federated Graph Learning (FGL)   Yao et al., 2022 ; Huang et al., 2022 ; Huang et al., 2023 emerges as a potent solution to these privacy concerns by leveraging the distributed nature of data without necessitating its central aggregation. This approach aligns with the stringent requirements of data privacy regulations by enabling data to remain at its source, thereby minimizing the risks associated with centralized data storage and processing   Huang et al., 2023a ; Wan et al., 2024 .

 
 
 
 

## 6. Conclusion

 
 In this survey, we present a comprehensive overview of graph neural networks in epidemic modeling. First, we provide introductions and definitions not only for epidemiology and epidemic modeling but also for Graph Neural Networks (GNNs). Then, clear and structured taxonomies for epidemiological tasks and methodology are proposed. In terms of epidemiological tasks, we present a categorization of tasks consisting of four parts: Detection, Surveillance, Prediction, and Projection. In terms of methodology, we focus on GNN-based methods and separate them into Neural Models and Hybrid Models. At the end of our survey, we not only point out the challenges and drawbacks of the current methodology but also offer a number of promising directions for interested researchers to work on. The aim of this survey is to bridge the gaps between Graph Neural Networks (GNNs) and epidemiology, inspiring both epidemiologists and data scientists to pursue advancements in this burgeoning field.

 
 
 

## References

 
 
 Bramanti (2012) 
 Barbara Bramanti
 
 “Ancient Epidemic Diseases in a New Light”
 
 In German Research 34.2 , 2012, pp. 22–27
 

 
 Bruce-Chwatt (1977) 
 L.. Bruce-Chwatt
 
 “Plagues and Peoples. By William H. McNeill. Pp. 369. (Basil Blackwell, Oxford, 1977.)”
 
 In Journal of Biosocial Science 9.4 , 1977, pp. 501–503
 

 
 Frérot et al. (2018) 
 Mathilde Frérot et al.
 
 “What is epidemiology? Changing definitions of epidemiology 1978-2017”
 
 In PloS one 13.12 
 
 Public Library of Science San Francisco, CA USA, 2018, pp. e0208442
 

 
 Fine (2015) 
 Paul Fine
 
 “Another Defining Moment for Epidemiology”
 
 In The Lancet 385.9965 , 2015, pp. 319–320
 

 
 Terris (1993) 
 Milton Terris
 
 “The Society for Epidemiologic Research and the Future of Epidemiology”
 
 In Journal of Public Health Policy 14.2 , 1993, pp. 137
 
 JSTOR:3342960
 

 
 Cm (2020) 
 Jayadevan Cm
 
 “Does the Inadequate Health Resources Aggravate Covid-19 Pandemic?”
 
 In Scholars Journal of Applied Medical Sciences 8.7 , 2020, pp. 1646–1650
 

 
 Bergrath et al. (2022) 
 Sebastian Bergrath et al.
 
 “Impact of the COVID-19 Pandemic on Emergency Medical Resources: An Observational Multicenter Study Including All Hospitals in a Major Urban Center of the Rhein-Ruhr Metropolitan Region”
 
 In Die Anaesthesiologie 71.S2 , 2022, pp. 171–179
 

 
 Emanuel et al. (2020) 
 Ezekiel Emanuel et al.
 
 “Fair allocation of scarce medical resources in the time of Covid-19”
 
 In New England Journal of Medicine 382.21 
 
 Mass Medical Soc, 2020, pp. 2049–2055
 

 
 Funk et al. (2018) 
 Sebastian Funk et al.
 
 “Real-Time Forecasting of Infectious Disease Dynamics with a Stochastic Semi-Mechanistic Model”
 
 In Epidemics 22 , 2018, pp. 56–61
 

 
 Kondratyev (2013) 
 Mikhail Kondratyev
 
 “Forecasting Methods and Models of Disease Spread”
 
 In Computer Research and Modeling 5.5 , 2013, pp. 863–882
 

 
 Louz et al. (2010) 
 Derrick Louz, Hans. Bergmans, Birgit. Loos and Rob. Hoeben
 
 “Emergence of Viral Diseases: Mathematical Modeling as a Tool for Infection Control, Policy and Decision Making”
 
 In Critical Reviews in Microbiology 36.3 , 2010, pp. 195–211
 

 
 Mikolajczyk et al. (2009) 
 Rafael Mikolajczyk et al.
 
 “Influenza”
 
 In Deutsches Ärzteblatt international , 2009
 

 
 Shorten et al. (2021) 
 Connor Shorten, Taghi Khoshgoftaar and Borko Furht
 
 “Deep Learning applications for COVID-19”
 
 In Journal of big Data 8.1 
 
 Springer, 2021, pp. 1–54
 

 
 Wu et al. (2018) 
 Yuexin Wu, Yiming Yang, Hiroshi Nishiura and Masaya Saitoh
 
 “Deep learning for epidemiological predictions”
 
 In The 41st International ACM SIGIR Conference on Research Development in Information Retrieval , 2018, pp. 1085–1088
 

 
 Saleem et al. (2022) 
 Farrukh Saleem, Abdullah-Malaise Al-Ghamdi, Madini Alassafi and Saad AlGhamdi
 
 “Machine learning, deep learning, and mathematical models to analyze forecasting and epidemiology of COVID-19: A systematic literature review”
 
 In International journal of environmental research and public health 19.9 
 
 MDPI, 2022, pp. 5099
 

 
 Baldo et al. (2021) 
 Federico Baldo et al.
 
 “Deep learning for virus-spreading forecasting: A brief survey”
 
 In arXiv:2103.02346 , 2021
 

 
 Wu et al. (2020) 
 Zonghan Wu et al.
 
 “A comprehensive survey on graph neural networks”
 
 In IEEE TNNLS , 2020, pp. 4–24
 

 
 Kipf Welling (2017) 
 Thomas Kipf and Max Welling
 
 “Semi-supervised classification with graph convolutional networks”
 
 In ICLR , 2017
 

 
 Veličković et al. (2017) 
 Petar Veličković et al.
 
 “Graph attention networks”
 
 In arXiv preprint arXiv:1710.10903 , 2017
 

 
 Brody et al. (2022) 
 Shaked Brody, Uri Alon and Eran Yahav
 
 “How attentive are graph attention networks?”
 
 In ICLR , 2022
 

 
 Liu et al. (2023) 
 Xiao Liu, Lijun Zhang and Hui Guan
 
 “Uplifting Message Passing Neural Network with Graph Original Information”
 
 arXiv, 2023
 
 eprint:2210.05382
 

 
 Maskey et al. (2022) 
 Sohir Maskey, Ron Levie, Yunseok Lee and Gitta Kutyniok
 
 “Generalization Analysis of Message Passing Neural Networks on Large Random Graphs”
 
 arXiv, 2022
 
 eprint:2202.00645
 

 
 Deng et al. (2020) 
 Songgaojun Deng et al.
 
 “Cola-GNN: Cross-location Attention Based Graph Neural Networks for Long-term ILI Prediction”
 
 In Proceedings of the 29th ACM International Conference on Information Knowledge Management 
 
 ACM, 2020, pp. 245–254
 

 
 Wang et al. (2022) 
 Lijing Wang et al.
 
 “CausalGNN: Causal-Based Graph Neural Networks for Spatio-Temporal Epidemic Forecasting”
 
 In Proceedings of the AAAI Conference on Artificial Intelligence 36.11 , 2022, pp. 12191–12199
 

 
 Gao et al. (2021) 
 Junyi Gao et al.
 
 “STAN: Spatio-Temporal Attention Network for Pandemic Prediction Using Real-World Evidence”
 
 In Journal of the American Medical Informatics Association 28.4 , 2021, pp. 733–743
 

 
 Cao et al. (2022) 
 Qi Cao et al.
 
 “MepoGNN: Metapopulation Epidemic Forecasting with Graph Neural Networks”, 2022
 

 
 Feng et al. (2023) 
 Tao Feng, Sirui Song, Tong Xia and Yong Li
 
 “Contact Tracing and Epidemic Intervention via Deep Reinforcement Learning”
 
 In ACM Transactions on Knowledge Discovery from Data 17.3 , 2023, pp. 1–24
 

 
 Liu et al. (2023a) 
 Mutong Liu, Yang Liu and Jiming Liu
 
 “Epidemiology-Aware Deep Learning for Infectious Disease Dynamics Prediction”
 
 In International Conference on Information and Knowledge Management, Proceedings 
 
 Association for Computing Machinery, 2023, pp. 4084–4088
 

 
 Ru et al. (2023) 
 Xiaolei Ru et al.
 
 “Inferring Patient Zero on Temporal Networks via Graph Neural Networks”
 
 In Proceedings of the AAAI Conference on Artificial Intelligence 37.8 , 2023, pp. 9632–9640
 

 
 Song et al. (2020) 
 Sirui Song et al.
 
 “Reinforced Epidemic Control: Saving Both Lives and Economy”
 
 arXiv, 2020
 
 eprint:2008.01257
 

 
 Clement et al. (2022) 
 J. Clement, VijayaKumar Ponnusamy, K.C. Sriharipriya and R. Nandakumar
 
 “A Survey on Mathematical, Machine Learning and Deep Learning Models for COVID-19 Transmission and Diagnosis”
 
 In IEEE Reviews in Biomedical Engineering 15 , 2022, pp. 325–340
 

 
 Kamalov et al. (2022) 
 Firuz Kamalov et al.
 
 “Deep Learning for Covid-19 Forecasting: State-of-the-art Review.”
 
 In Neurocomputing 511 , 2022, pp. 142–154
 

 
 (1) 
 J Dinesh and K Pelusi
 
 “Significance of deep learning for COVID-19: state-of-the-art review”
 
 In Research Biomedical Engineering, doi 10 
 

 
 Rodríguez et al. (2022) 
 Alexander Rodríguez et al.
 
 “Data-centric epidemic forecasting: A survey”
 
 In arXiv preprint arXiv:2207.09370 , 2022
 

 
 Reiser et al. (2022) 
 Patrick Reiser et al.
 
 “Graph neural networks for materials science and chemistry”
 
 In Communications Materials 3.1 
 
 Nature Publishing Group UK London, 2022, pp. 93
 

 
 Wen et al. (2022) 
 Hongzhi Wen et al.
 
 “Graph neural networks for multimodal single-cell data integration”
 
 In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining , 2022, pp. 4153–4163
 

 
 Wieder et al. (2020) 
 Oliver Wieder et al.
 
 “A compact review of molecular property prediction with graph neural networks”
 
 In Drug Discovery Today: Technologies 37 
 
 Elsevier, 2020, pp. 1–12
 

 
 Bessadok et al. (2022) 
 Alaa Bessadok, Mohamed Mahjoub and Islem Rekik
 
 “Graph neural networks in network neuroscience”
 
 In IEEE Transactions on Pattern Analysis and Machine Intelligence 45.5 
 
 IEEE, 2022, pp. 5833–5848
 

 
 Dai et al. (2022) 
 Enyan Dai et al.
 
 “A comprehensive survey on trustworthy graph neural networks: Privacy, robustness, fairness, and explainability”
 
 In arXiv preprint arXiv:2204.08570 , 2022
 

 
 Liu et al. (2022) 
 Yue Liu et al.
 
 “A Survey of Deep Graph Clustering: Taxonomy, Challenge, and Application”
 
 In arXiv preprint arXiv:2211.12875 , 2022
 

 
 Aleta et al. (2020) 
 Alberto Aleta et al.
 
 “Modelling the Impact of Testing, Contact Tracing and Household Quarantine on Second Waves of COVID-19”
 
 In Nature Human Behaviour 4.9 , 2020, pp. 964–971
 

 
 Chang et al. (2020) 
 Sheryl. Chang et al.
 
 “Modelling Transmission and Control of the COVID-19 Pandemic in Australia”
 
 In Nature Communications 11.1 , 2020, pp. 5710
 

 
 Jiang et al. (2021) 
 Renhe Jiang et al.
 
 “Countrywide Origin-Destination Matrix Prediction and Its Application for COVID-19”
 
 In Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track 12978 
 
 Springer International Publishing, 2021, pp. 319–334
 

 
 Yang et al. (2023) 
 Chuang Yang et al.
 
 “EpiMob: Interactive Visual Analytics of Citywide Human Mobility Restrictions for Epidemic Control”
 
 In IEEE Transactions on Visualization and Computer Graphics 29.8 , 2023, pp. 3586–3601
 

 
 (2) 
 “A Contribution to the Mathematical Theory of Epidemics”
 
 In Proceedings of the Royal Society of London. Series A, Containing Papers of a Mathematical and Physical Character 115.772 , 1927, pp. 700–721
 

 
 Danon et al. (2011) 
 Leon Danon et al.
 
 “Networks and the epidemiology of infectious disease”
 
 In Interdisciplinary perspectives on infectious diseases 2011 
 
 Hindawi, 2011
 

 
 Caals et al. (2017) 
 Karel Caals, Abha Saxena and Calvin-Loon Ho
 
 “Ethics of Epidemics, Research and Surveillance: A WHO Workshop Report”
 
 In Asian Bioethics Review 9.3 , 2017, pp. 265–271
 

 
 Dehning et al. (2020) 
 Jonas Dehning et al.
 
 “Inferring Change Points in the Spread of COVID-19 Reveals the Effectiveness of Interventions”
 
 In Science 369.6500 , 2020, pp. eabb9789
 

 
 Grassly Fraser (2008) 
 Nicholas. Grassly and Christophe Fraser
 
 “Mathematical Models of Infectious Disease Transmission”
 
 In Nature Reviews Microbiology 6.6 , 2008, pp. 477–487
 

 
 Han et al. (2023) 
 Zhenyu Han et al.
 
 “Devil in the Landscapes: Inferring Epidemic Exposure Risks from Street View Imagery”
 
 In Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems 
 
 ACM, 2023, pp. 1–4
 

 
 Shah et al. (2020) 
 Chintan Shah et al.
 
 “Finding Patient Zero: Learning Contagion Source with Graph Neural Networks”
 
 arXiv, 2020
 
 eprint:2006.11913
 

 
 Sha et al. (2021) 
 Hao Sha, Mohammad Al and George Mohler
 
 “Source Detection on Networks Using Spatial Temporal Graph Convolutional Networks”
 
 In 2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA) 
 
 IEEE, 2021, pp. 1–11
 

 
 Brede (2012) 
 Markus Brede
 
 “ Networks— An Introduction . Mark E. J. Newman. (2010, Oxford University Press.) $65.38, £35.96 (Hardcover), 772 Pages. ISBN-978-0-19-920665-0.”
 
 In Artificial Life 18.2 , 2012, pp. 241–242
 

 
 Balcan et al. (2009) 
 Duygu Balcan et al.
 
 “Multiscale Mobility Networks and the Spatial Spreading of Infectious Diseases”
 
 In Proceedings of the National Academy of Sciences 106.51 , 2009, pp. 21484–21489
 

 
 Venkatramanan et al. (2017) 
 Srinivasan Venkatramanan et al.
 
 “Spatio-Temporal Optimization of Seasonal Vaccination Using a Metapopulation Model of Influenza”
 
 In 2017 IEEE International Conference on Healthcare Informatics (ICHI) 
 
 IEEE, 2017, pp. 134–143
 

 
 Dixon et al. (2018) 
 Brian Dixon, George Lecakes, Paul. Moon and John Schmalzel
 
 “SEDS: Expanding TEDS to Include Physical Structures”
 
 In 2018 IEEE Sensors Applications Symposium (SAS) , 2018, pp. 1–6
 

 
 Van (2017) 
 Pauline Van
 
 “Reproduction Numbers of Infectious Disease Models”
 
 In Infectious Disease Modelling 2.3 , 2017, pp. 288–303
 

 
 Tomy et al. (2022) 
 Abhishek Tomy et al.
 
 “Estimating the State of Epidemics Spreading with Graph Neural Networks”
 
 In Nonlinear Dynamics 109.1 
 
 Springer Science and Business Media B.V., 2022, pp. 249–263
 
 eprint:2105.05060
 

 
 Wang et al. (2022a) 
 Lijing Wang et al.
 
 “CausalGNN: Causal-Based Graph Neural Networks for Spatio-Temporal Epidemic Forecasting”, 2022
 

 
 Loli Zama (2020) 
 Elena Loli and Fabiana Zama
 
 “Monitoring Italian COVID-19 Spread by a Forced SEIRD Model”
 
 In PLOS ONE 15.8 , 2020, pp. e0237417
 

 
 Liu et al. (2024) 
 Zewen Liu et al.
 
 “EpiLearn: A Python Library for Machine Learning in Epidemic Modeling”
 
 In arXiv e-prints , 2024, pp. arXiv–2406
 

 
 Wang et al. (2023) 
 Siqi Wang et al.
 
 “WDCIP: Spatio-Temporal AI-driven Disease Control Intelligent Platform for Combating COVID-19 Pandemic”
 
 In Geo-spatial Information Science 0.0 
 
 Taylor Francis, 2023, pp. 1–25
 

 
 Yang et al. (2022) 
 Carl Yang et al.
 
 “Dynamic network anomaly modeling of cell-phone call detail records for infectious disease surveillance”
 
 In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , 2022, pp. 4733–4742
 

 
 Jhun (2021) 
 Bukyoung Jhun
 
 “Effective Vaccination Strategy Using Graph Neural Network Ansatz”
 
 arXiv, 2021
 
 eprint:2111.00920
 

 
 Feng et al. (2023a) 
 Tao Feng, Sirui Song, Tong Xia and Yong Li
 
 “Contact Tracing and Epidemic Intervention via Deep Reinforcement Learning”
 
 In ACM Transactions on Knowledge Discovery from Data 17.3 , 2023, pp. 1–24
 

 
 Liu et al. (2023b) 
 Yang Liu et al.
 
 “Human Mobility Modeling during the COVID-19 Pandemic via Deep Graph Diffusion Infomax”
 
 In Proceedings of the AAAI Conference on Artificial Intelligence 37.12 , 2023, pp. 14347–14355
 

 
 Zhang et al. (2023) 
 Jie Zhang, Pengfei Zhou, Yijia Zheng and Hongyan Wu
 
 “Predicting Influenza with Pandemic-Awareness via Dynamic Virtual Graph Significance Networks”
 
 In Computers in Biology and Medicine 158 , 2023, pp. 106807
 

 
 Tang et al. (2023) 
 Yinzhou Tang, Huandong Wang and Yong Li
 
 “Enhancing Spatial Spread Prediction of Infectious Diseases through Integrating Multi-scale Human Mobility Dynamics”
 
 In Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems 
 
 ACM, 2023, pp. 1–12
 

 
 Lin et al. (2023) 
 Chen Lin et al.
 
 “Graph Neural Network Modeling of Web Search Activity for Real-time Pandemic Forecasting”
 
 In 2023 IEEE 11th International Conference on Healthcare Informatics (ICHI) 
 
 IEEE, 2023, pp. 128–137
 

 
 Nguyen et al. (2023) 
 Viet Nguyen, Truong Hy, Long Tran-Thanh and Nhung Nghiem
 
 “Predicting COVID-19 Pandemic by Spatio-Temporal Graph Neural Networks: A New Zealand’s Study”, 2023
 
 eprint:2305.07731
 

 
 Pu et al. (2023) 
 Xiaojun Pu et al.
 
 “Dynamic Adaptive Spatio–Temporal Graph Network for COVID-19 Forecasting”
 
 In CAAI Transactions on Intelligence Technology 
 
 John Wiley and Sons Inc, 2023
 

 
 Qiu et al. (2023) 
 Mingjie Qiu, Zhiyi Tan and Bing-kun Bao
 
 “MSGNN: Multi-scale Spatio-temporal Graph Neural Network for Epidemic Forecasting”
 
 arXiv, 2023
 
 eprint:2308.15840
 

 
 Yu et al. (2023) 
 Shuo Yu et al.
 
 “Spatio-Temporal Graph Learning for Epidemic Prediction”
 
 In ACM Transactions on Intelligent Systems and Technology 14.2 
 
 Association for Computing Machinery, 2023
 

 
 Li et al. (2019) 
 Zhijian Li et al.
 
 “A Study on Graph-Structured Recurrent Neural Networks and Sparsification with Application to Epidemic Forecasting”
 
 arXiv, 2019
 
 eprint:1902.05113
 

 
 Cao et al. (2023) 
 Qi Cao et al.
 
 “Metapopulation Graph Neural Networks: Deep Metapopulation Epidemic Modeling with Human Mobility”, 2023, pp. 453–468
 
 eprint:2306.14857
 

 
 Ma et al. (2022) 
 Yihong Ma et al.
 
 “Hierarchical spatio-temporal graph neural networks for pandemic forecasting”
 
 In Proceedings of the 31st ACM International Conference on Information Knowledge Management , 2022, pp. 1481–1490
 

 
 Wang et al. (2022b) 
 Yuejiao Wang et al.
 
 “Adaptively Temporal Graph Convolution Model for Epidemic Prediction of Multiple Age Groups”
 
 In Fundamental Research 2.2 , 2022, pp. 311–320
 

 
 Zheng et al. (2021) 
 Shun Zheng et al.
 
 “HierST: A Unified Hierarchical Spatial-temporal Framework for COVID-19 Trend Forecasting”
 
 In Proceedings of the 30th ACM International Conference on Information Knowledge Management 
 
 ACM, 2021, pp. 4383–4392
 

 
 Moon et al. (2023) 
 Jaeuk Moon, Seungwon Jung, Sungwoo Park and Eenjun Hwang
 
 “RESEAT: Recurrent Self-Attention Network for Multi-Regional Influenza Forecasting”
 
 In IEEE Journal of Biomedical and Health Informatics 27.5 , 2023, pp. 2585–2596
 

 
 Jung et al. (2022) 
 Seungwon Jung, Jaeuk Moon, Sungwoo Park and Eenjun Hwang
 
 “Self-Attention-Based Deep Learning Network for Regional Influenza Forecasting”
 
 In IEEE Journal of Biomedical and Health Informatics 26.2 , 2022, pp. 922–933
 

 
 Xie et al. (2022) 
 Feng Xie, Zhong Zhang, Liang Li and Yusong Tan
 
 “EpiGNN: Exploring Spatial Transmission with Graph Neural Network for Regional Epidemic Forecasting”, 2022
 

 
 Ru et al. (2023a) 
 Xiaolei Ru et al.
 
 “Inferring Patient Zero on Temporal Networks via Graph Neural Networks”
 
 In Proceedings of the AAAI Conference on Artificial Intelligence 37.8 , 2023, pp. 9632–9640
 

 
 Song et al. (2023) 
 Kyungwoo Song et al.
 
 “COVID-19 Infection Inference with Graph Neural Networks”
 
 In Scientific Reports 13.1 
 
 Nature Research, 2023
 

 
 Gouareb et al. (2023) 
 Racha Gouareb et al.
 
 “Detection of Patients at Risk of Enterobacteriaceae Infection Using Graph Neural Networks: A Retrospective Study”
 
 medRxiv, 2023, pp. 2023.06.01.23290386
 

 
 Siji et al. (2023) 
 S. Siji et al.
 
 “Spatio-Temporal Prediction in Epidemiology Using Graph Convolution Network”
 
 In Lecture Notes in Networks and Systems 720 LNNS 
 
 Springer Science and Business Media Deutschland GmbH, 2023, pp. 367–378
 

 
 Yu et al. (2023a) 
 Shuo Yu et al.
 
 “Spatio-Temporal Graph Learning for Epidemic Prediction”
 
 In ACM Transactions on Intelligent Systems and Technology 14.2 
 
 Association for Computing Machinery, 2023
 

 
 Nguyen et al. (2023a) 
 Viet Nguyen, Truong Hy, Long Tran-Thanh and Nhung Nghiem
 
 “Predicting COVID-19 Pandemic by Spatio-Temporal Graph Neural Networks: A New Zealand’s Study”, 2023
 
 eprint:2305.07731
 

 
 Croft et al. (2023) 
 V. Croft, Senna… van Iersel and Cosimo Della
 
 “Forecasting Infections with Spatio-Temporal Graph Neural Networks: A Case Study of the Dutch SARS-CoV-2 Spread”
 
 In Frontiers in Physics 11 
 
 Frontiers Media SA, 2023
 

 
 Moon et al. (2023a) 
 Jaeuk Moon, Seungwon Jung, Sungwoo Park and Eenjun Hwang
 
 “RESEAT: Recurrent Self-Attention Network for Multi-Regional Influenza Forecasting”
 
 In IEEE Journal of Biomedical and Health Informatics 27.5 , 2023, pp. 2585–2596
 

 
 Cao et al. (2023a) 
 Qi Cao et al.
 
 “Metapopulation Graph Neural Networks: Deep Metapopulation Epidemic Modeling with Human Mobility”, 2023, pp. 453–468
 
 eprint:2306.14857
 

 
 Tang et al. (2023a) 
 Yinzhou Tang, Huandong Wang and Yong Li
 
 “Enhancing Spatial Spread Prediction of Infectious Diseases through Integrating Multi-scale Human Mobility Dynamics”
 
 In Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems , SIGSPATIAL ’23
 
 Association for Computing Machinery, 2023, pp. 1–12
 

 
 Liu et al. (2023c) 
 Mutong Liu, Yang Liu and Jiming Liu
 
 “Epidemiology-Aware Deep Learning for Infectious Disease Dynamics Prediction”
 
 In International Conference on Information and Knowledge Management, Proceedings 
 
 Association for Computing Machinery, 2023, pp. 4084–4088
 

 
 Zhang et al. (2023a) 
 Jie Zhang, Pengfei Zhou, Yijia Zheng and Hongyan Wu
 
 “Predicting Influenza with Pandemic-Awareness via Dynamic Virtual Graph Significance Networks”
 
 In Computers in Biology and Medicine 158 , 2023, pp. 106807
 

 
 Moon et al. (2023b) 
 Sifat Moon et al.
 
 “A Graph Based Deep Learning Framework for Predicting Spatio-Temporal Vaccine Hesitancy”
 
 medRxiv, 2023, pp. 2023.10.24.23297488
 

 
 Sun et al. (2023) 
 Chaoyue Sun et al.
 
 “DeepDynaForecast: Phylogenetic-informed Graph Deep Learning for Epidemic Transmission Dynamic Prediction”
 
 bioRxiv, 2023, pp. 2023.07.17.549268
 

 
 Kempe et al. (2003) 
 David Kempe, Jon Kleinberg and Éva Tardos
 
 “Maximizing the spread of influence through a social network”
 
 In Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining , 2003, pp. 137–146
 

 
 Bucur Holme (2020) 
 Doina Bucur and Petter Holme
 
 “Beyond ranking nodes: Predicting epidemic outbreak sizes by network centralities”
 
 In PLoS computational biology 16.7 
 
 Public Library of Science San Francisco, CA USA, 2020, pp. e1008052
 

 
 Holme (2017) 
 Petter Holme
 
 “Three faces of node importance in network epidemiology: Exact results for small graphs”
 
 In Physical Review E 96.6 
 
 APS, 2017, pp. 062305
 

 
 Meirom et al. (2021) 
 Eli. Meirom, Haggai Maron, Shie Mannor and Gal Chechik
 
 “Controlling Graph Dynamics with Reinforcement Learning and Graph Neural Networks”
 
 arXiv, 2021
 
 eprint:2010.05313
 

 
 Shan et al. (2023) 
 Baoling Shan et al.
 
 “Novel Graph Topology Learning for Spatio-Temporal Analysis of COVID-19 Spread”
 
 In IEEE Journal of Biomedical and Health Informatics 27.6 , 2023, pp. 2693–2704
 

 
 He et al. (2016) 
 Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun
 
 “Deep residual learning for image recognition”
 
 In Proceedings of the IEEE conference on computer vision and pattern recognition , 2016, pp. 770–778
 

 
 Mežnar et al. (2021) 
 Sebastian Mežnar, Nada Lavrač and Blaž Škrlj
 
 “Prediction of the Effects of Epidemic Spreading with Graph Neural Networks”
 
 In Complex Networks Their Applications IX , Studies in Computational Intelligence
 
 Springer International Publishing, 2021, pp. 420–431
 

 
 Jiang et al. (2016) 
 Shan Jiang et al.
 
 “The TimeGeo modeling framework for urban mobility without travel surveys”
 
 In Proceedings of the National Academy of Sciences 113.37 
 
 National Acad Sciences, 2016, pp. E5370–E5378
 

 
 Murphy et al. (2021) 
 Charles Murphy, Edward Laurence and Antoine Allard
 
 “Deep Learning of Contagion Dynamics on Complex Networks”
 
 In Nature Communications 12.1 
 
 Nature Publishing Group, 2021, pp. 4720
 

 
 Pu et al. (2023a) 
 Xiaojun Pu et al.
 
 “Dynamic Adaptive Spatio–Temporal Graph Network for COVID-19 Forecasting”
 
 In CAAI Transactions on Intelligence Technology 
 
 John Wiley and Sons Inc, 2023
 

 
 Gouareb et al. (2023a) 
 Racha Gouareb et al.
 
 “Detection of Patients at Risk of Enterobacteriaceae Infection Using Graph Neural Networks: A Retrospective Study”
 
 medRxiv, 2023, pp. 2023.06.01.23290386
 

 
 Tomy et al. (2022a) 
 Abhishek Tomy et al.
 
 “Estimating the State of Epidemics Spreading with Graph Neural Networks”
 
 In Nonlinear Dynamics 109.1 , 2022, pp. 249–263
 

 
 Song et al. (2023a) 
 Kyungwoo Song et al.
 
 “COVID-19 Infection Inference with Graph Neural Networks”
 
 In Scientific Reports 13.1 
 
 Nature Research, 2023
 

 
 (3) 
 Xiaojun Pu et al.
 
 “Dynamic Adaptive Spatio–Temporal Graph Network for COVID-19 Forecasting”
 
 In CAAI Transactions on Intelligence Technology n/a.n/a 
 

 
 La et al. (2021) 
 Valerio La, Vincenzo Moscato, Marco Postiglione and Giancarlo Sperli
 
 “An Epidemiological Neural Network Exploiting Dynamic Graph Structured Data Applied to the COVID-19 Outbreak”
 
 In IEEE Transactions on Big Data 7.1 , 2021, pp. 45–55
 

 
 Panagopoulos et al. (2021) 
 George Panagopoulos, Giannis Nikolentzos and Michalis Vazirgiannis
 
 “Transfer Graph Neural Networks for Pandemic Forecasting”
 
 In Proceedings of the AAAI Conference on Artificial Intelligence 35.6 , 2021, pp. 4838–4845
 

 
 Wang et al. (2022c) 
 Haorui Wang, Haoteng Yin, Muhan Zhang and Pan Li
 
 “Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks”
 
 arXiv, 2022
 
 eprint:2203.00199
 

 
 (4) 
 Deyu Bo, Yuan Fang, Yang Liu and Chuan Shi
 
 “Graph Contrastive Learning with Stable and Scalable Spectral Encoding”
 

 
 Vaswani et al. (2023) 
 Ashish Vaswani et al.
 
 “Attention Is All You Need”
 
 arXiv, 2023
 
 eprint:1706.03762
 

 
 Turner (2024) 
 Richard. Turner
 
 “An Introduction to Transformers”
 
 arXiv, 2024
 
 eprint:2304.10557
 

 
 Soydaner (2022) 
 Derya Soydaner
 
 “Attention Mechanism in Neural Networks: Where It Comes and Where It Goes”
 
 In Neural Computing and Applications 34.16 , 2022, pp. 13371–13385
 
 eprint:2204.13154
 

 
 Cui et al. (2021) 
 Yue Cui et al.
 
 “Into the Unobservables: A Multi-range Encoder-decoder Framework for COVID-19 Prediction”
 
 In Proceedings of the 30th ACM International Conference on Information Knowledge Management 
 
 ACM, 2021, pp. 292–301
 

 
 Siji et al. (2023a) 
 S. Siji et al.
 
 “Spatio-Temporal Prediction in Epidemiology Using Graph Convolution Network”
 
 In IOT with Smart Systems 
 
 Springer Nature, 2023, pp. 367–378
 

 
 Liu et al. (2021) 
 Yixin Liu et al.
 
 “Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning”
 
 In IEEE TNNLS 
 
 IEEE, 2021
 

 
 Kapoor et al. (2020) 
 Amol Kapoor et al.
 
 “Examining COVID-19 Forecasting Using Spatio-Temporal Graph Neural Networks”
 
 arXiv, 2020
 
 eprint:2007.03113
 

 
 Moon et al. (2023c) 
 Sifat Moon et al.
 
 “A Graph Based Deep Learning Framework for Predicting Spatio-Temporal Vaccine Hesitancy”
 
 medRxiv, 2023, pp. 2023.10.24.23297488
 

 
 Duarte et al. (2023) 
 Fernando Duarte et al.
 
 “Time Series Forecasting of COVID-19 Cases in Brazil with GNN and Mobility Networks”, 2023, pp. 361–375
 

 
 Sesti et al. (2021) 
 Nathan Sesti, Juan Garau-Luis, Edward Crawley and Bruce Cameron
 
 “Integrating LSTMs and GNNs for COVID-19 Forecasting”
 
 arXiv, 2021
 
 eprint:2108.10052
 

 
 Guo Yang (2021) 
 Lihao Guo and Yuxin Yang
 
 “Research on the Forecast of the Spread of COVID-19”
 
 In 2021 11th International Conference on Biomedical Engineering and Technology 
 
 ACM, 2021, pp. 47–51
 

 
 Hu et al. (2022) 
 Na Hu et al.
 
 “Graph Learning-Based Spatial-Temporal Graph Convolutional Neural Networks for Traffic Forecasting”
 
 In Connection Science 34.1 , 2022, pp. 429–448
 

 
 Zhou et al. (2021) 
 Jie Zhou et al.
 
 “Graph Neural Networks: A Review of Methods and Applications”
 
 arXiv, 2021
 
 eprint:1812.08434
 

 
 Wu et al. (2021) 
 Zonghan Wu et al.
 
 “A Comprehensive Survey on Graph Neural Networks”
 
 In IEEE Transactions on Neural Networks and Learning Systems 32.1 , 2021, pp. 4–24
 
 eprint:1901.00596
 

 
 Defferrard et al. (2016) 
 Michaël Defferrard, Xavier Bresson and Pierre Vandergheynst
 
 “Convolutional neural networks on graphs with fast localized spectral filtering”
 
 In NeurIPS 29 , 2016
 

 
 He et al. (2022) 
 Mingguo He, Zhewei Wei and Ji-Rong Wen
 
 “Convolutional neural networks on graphs with chebyshev approximation, revisited”
 
 In NeurIPS 35 , 2022, pp. 7264–7276
 

 
 Xie et al. (2022a) 
 Huaze Xie, Da Li, Yuanyuan Wang and Yukiko Kawai
 
 “Visualization Method for the Spreading Curve of COVID-19 in Universities Using GNN”
 
 In 2022 IEEE International Conference on Big Data and Smart Computing (BigComp) 
 
 IEEE, 2022, pp. 121–128
 

 
 Cao et al. (2023b) 
 Qi Cao et al.
 
 “MepoGNN: Metapopulation Epidemic Forecasting with Graph Neural Networks”
 
 In Machine Learning and Knowledge Discovery in Databases 13718 
 
 Springer Nature Switzerland, 2023, pp. 453–468
 

 
 Mousavi et al. (2012) 
 Sayyed Mousavi, Fateme Bahri and Farzaneh Tabataba
 
 “An enhanced beam search algorithm for the shortest common supersequence problem”
 
 In Engineering Applications of Artificial Intelligence 25.3 
 
 Elsevier, 2012, pp. 457–467
 

 
 Diekmann et al. (2010) 
 O. Diekmann, J… Heesterbeek and M.. Roberts
 
 “The Construction of Next-Generation Matrices for Compartmental Epidemic Models”
 
 In Journal of The Royal Society Interface 7.47 , 2010, pp. 873–885
 

 
 Han et al. (2023a) 
 Zhenyu Han et al.
 
 “Devil in the Landscapes: Inferring Epidemic Exposure Risks from Street View Imagery”
 
 In Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems 
 
 ACM, 2023, pp. 1–4
 

 
 Kermack McKendrick (1927) 
 William Kermack and Anderson McKendrick
 
 “A contribution to the mathematical theory of epidemics”
 
 In Proceedings of the royal society of london. Series A, Containing papers of a mathematical and physical character 115.772 
 
 The Royal Society London, 1927, pp. 700–721
 

 
 Wang et al. (2023a) 
 Siqi Wang et al.
 
 “WDCIP: Spatio-Temporal AI-driven Disease Control Intelligent Platform for Combating COVID-19 Pandemic”
 
 In Geo-spatial Information Science 0.0 
 
 Taylor Francis, 2023, pp. 1–25
 

 
 Qiu et al. (2023a) 
 Mingjie Qiu, Zhiyi Tan and Bing-kun Bao
 
 “MSGNN: Multi-scale Spatio-temporal Graph Neural Network for Epidemic Forecasting”
 
 arXiv, 2023
 
 eprint:2308.15840
 

 
 Lin et al. (2023a) 
 Chen Lin et al.
 
 “Graph Neural Network Modeling of Web Search Activity for Real-time Pandemic Forecasting”
 
 In 2023 IEEE 11th International Conference on Healthcare Informatics (ICHI) , 2023, pp. 128–137
 

 
 Xhonneux et al. (2020) 
 Louis-Pascal Xhonneux, Meng Qu and Jian Tang
 
 “Continuous graph neural networks”
 
 In International conference on machine learning , 2020, pp. 10432–10441
 
 PMLR
 

 
 Chamberlain et al. (2021) 
 Ben Chamberlain et al.
 
 “Grand: Graph neural diffusion”
 
 In International Conference on Machine Learning , 2021, pp. 1407–1418
 
 PMLR
 

 
 Poli et al. (2019) 
 Michael Poli et al.
 
 “Graph neural ordinary differential equations”
 
 In arXiv preprint arXiv:1911.07532 , 2019
 

 
 Chen et al. (2018) 
 Ricky Chen, Yulia Rubanova, Jesse Bettencourt and David Duvenaud
 
 “Neural ordinary differential equations”
 
 In Advances in neural information processing systems 31 , 2018
 

 
 Bhattacharyya Agarwalla (2023) 
 Himashree Bhattacharyya and Rashmi Agarwalla
 
 “Public Health Interventions in the Control of Emerging Diseases”
 
 In International Journal Of Community Medicine And Public Health 10.9 , 2023, pp. 3398–3402
 

 
 Jamison (2007) 
 Dean. Jamison
 
 “Disease Control”
 
 In Solutions for the World’s Biggest Problems: Costs and Benefits 
 
 Cambridge University Press, 2007, pp. 295–344
 

 
 Luo et al. (2021) 
 Mingyu Luo, Jimin Sun, Zhenyu Gong and Zhen Wang
 
 “What Is Always Necessary throughout Efforts to Prevent and Control COVID-19 and Other Infectious Diseases? A Physical Containment Strategy and Public Mobilization and Management”
 
 In BioScience Trends 15.3 , 2021, pp. 188–191
 

 
 Dai Wang (2021) 
 Enyan Dai and Suhang Wang
 
 “Towards self-explainable graph neural network”
 
 In Proceedings of the 30th ACM International Conference on Information Knowledge Management , 2021, pp. 302–311
 

 
 Ying et al. (2019) 
 Zhitao Ying et al.
 
 “Gnnexplainer: Generating explanations for graph neural networks”
 
 In Advances in neural information processing systems 32 , 2019
 

 
 Yuan et al. (2022) 
 Hao Yuan, Haiyang Yu, Shurui Gui and Shuiwang Ji
 
 “Explainability in graph neural networks: A taxonomic survey”
 
 In IEEE transactions on pattern analysis and machine intelligence 45.5 
 
 IEEE, 2022, pp. 5782–5799
 

 
 Nian et al. (2024) 
 Yi Nian, Yurui Chang, Wei Jin and Lu Lin
 
 “Globally Interpretable Graph Learning via Distribution Matching”
 
 In WWW , 2024
 

 
 Zha et al. (2023) 
 Daochen Zha et al.
 
 “Data-centric ai: Perspectives and challenges”
 
 In Proceedings of the 2023 SIAM International Conference on Data Mining (SDM) , 2023, pp. 945–948
 
 SIAM
 

 
 Rodríguez et al. (2022a) 
 Alexander Rodríguez et al.
 
 “Data-Centric Epidemic Forecasting: A Survey”
 
 arXiv, 2022
 
 eprint:2207.09370
 

 
 Jin et al. (2020) 
 Wei Jin et al.
 
 “Graph structure learning for robust graph neural networks”
 
 In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery data mining , 2020, pp. 66–74
 

 
 Jin et al. (2021) 
 Wei Jin et al.
 
 “Adversarial attacks and defenses on graphs”
 
 In ACM SIGKDD Explorations Newsletter 22.2 
 
 ACM New York, NY, USA, 2021, pp. 19–34
 

 
 Zhu et al. (2021) 
 Yanqiao Zhu et al.
 
 “Deep graph structure learning for robust representations: A survey”
 
 In arXiv preprint arXiv:2103.03036 14 , 2021, pp. 1–1
 

 
 Liu et al. (2021a) 
 Xiaorui Liu et al.
 
 “Elastic graph neural networks”
 
 In International Conference on Machine Learning , 2021, pp. 6837–6849
 
 PMLR
 

 
 Selgelid (2016) 
 Michael. Selgelid
 
 “Ethics and Security Aspects of Infectious Disease Control: Interdisciplinary Perspectives”
 
 Routledge, 2016
 

 
 Liu Yang et al. (2022) 
 Liu Yang, Zhang Jiahui and Sun Kaiyang
 
 “Interpretation of Information Security and Data Privacy Protection According to the Data Use During the Epidemic”
 
 In Journal of Communication and Computer 19.1 , 2022
 

 
 May Sell (2006) 
 Christopher May and Susan Sell
 
 “Intellectual property rights: A critical history”
 
 Lynne Rienner Publishers Boulder, 2006
 

 
 Investigators (2016) 
 International of Investigators
 
 “Toward fairness in data sharing”
 
 In New England Journal of Medicine 375.5 
 
 Mass Medical Soc, 2016, pp. 405–407
 

 
 Yao et al. (2022) 
 Yuhang Yao, Weizhao Jin, Srivatsan Ravi and Carlee Joe-Wong
 
 “Fedgcn: Convergence and communication tradeoffs in federated training of graph convolutional networks”
 
 In arXiv preprint arXiv:2201.12433 , 2022
 

 
 Huang et al. (2022) 
 Wenke Huang, Mang Ye and Bo Du
 
 “Learn from others and be yourself in heterogeneous federated learning”
 
 In CVPR , 2022
 

 
 Huang et al. (2023) 
 Wenke Huang et al.
 
 “A Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark”
 
 In arXiv , 2023
 

 
 Huang et al. (2023a) 
 Wenke Huang, Guancheng Wan, Mang Ye and Bo Du
 
 “Federated Graph Semantic and Structural Learning”
 
 In IJCAI , 2023
 

 
 Wan et al. (2024) 
 Guancheng Wan, Wenke Huang and Mang Ye
 
 “Federated Graph Learning under Domain Shift with Generalizable Prototypes”
 
 In Proceedings of the AAAI Conference on Artificial Intelligence 38.14 , 2024, pp. 15429–15437
 

 
 Fritz et al. (2022) 
 Cornelius Fritz, Emilio Dorigatti and David Rügamer
 
 “Combining Graph Neural Networks and Spatio-Temporal Disease Models to Improve the Prediction of Weekly COVID-19 Cases in Germany”
 
 In Scientific Reports 12.1 
 
 Nature Research, 2022
 

 
 Davahli et al. (2021) 
 Mohammad Davahli et al.
 
 “Predicting the Dynamics of the COVID-19 Pandemic in the United States Using Graph Theory-Based Neural Networks”
 
 In International Journal of Environmental Research and Public Health 18.7 
 
 Multidisciplinary Digital Publishing Institute, 2021, pp. 3834
 

 
 Mahmud et al. (2021) 
 Shohaib Mahmud, Haiying Shen, Ying Foutz and Joshua Anton
 
 “A Human Mobility Data Driven Hybrid GNN+RNN Based Model For Epidemic Prediction”
 
 In 2021 IEEE International Conference on Big Data (Big Data) 
 
 IEEE, 2021, pp. 857–866
 

 
 
 
 
 

## Complete Taxonomy

 
 Table 1. 
 Summary of epidemiological tasks and Methodology .
 
 
 
 
 
   Task | 
 Paper | 
 Methodology | 
 Hybrid | 
 Graph Construction | 

 
 Detection | 
 SD-STGCN   Sha et al.,2021 | 
 GAT + GRU + SEIR | 
 ✓ | 
 Spatial-Temporal Graph; Static Graph Structure | 

 
   Ru et al.,2023a | 
 GCN + SIR | 
 ✓ | 
 Spatial-Temporal Graph; Dynamic Graph Structure | 

 
   Shah et al.,2020 | 
 GNN + SIR | 
 ✓ | 

 
 Surveillance | 
 WDCIP   Wang et al.,2023 | 
 GAE | 
 | 
 Spatial Graph; Static Graph Structure | 

 
   Song et al.,2023 | 
 GAT/GCN | 
 | 

 
   Gouareb et al.,2023 | 
 GCN | 
 | 

 
   Han et al.,2023 | 
 GCN + Modified SIR | 
 ✓ | 

 
 | 
   Shan et al.,2023 | 
 Graph Fourier Transform | 
 | 

 
 Projection | 
 MMCA-GNNA  Jhun,2021 | 
 GNN + SIR + RL | 
 ✓ | 
 Spatial-Temporal Graph; Static Graph Structure | 

 
 DURLECA   Song et al.,2020 | 
 GNN + RL | 
 | 

 
 IDRLECA   Feng et al.,2023a | 
 GNN + RL | 
 | 
 Spatial-Temporal Graph; Dynamic Graph Structure | 

 
   Meirom et al.,2021 | 
 GNN + RL | 
 | 

 
 Prediction | 
 DGDI   Liu et al.,2023b | 
 GCN + Self-Attention | 
 | 
 Spatial Graph; Static Graph Structure | 

 
   Sun et al.,2023 | 
 GNN + LSTM | 
 | 

 
 DVGSN   Zhang et al.,2023 | 
 GNN | 
 | 
 Temporal-Only Graph; Static Graph Structure | 

 
 STAN   Gao et al.,2021 | 
 GAT + GRU | 
 | 
 Spatial-Temporal Graph; Static Graph Structure | 

 
 MSDNet   Tang et al.,2023 | 
 GAT + GRU + SIS | 
 ✓ | 

 
 SMPNN   Lin et al.,2023 | 
 MPNN + Autoregression | 
 ✓ | 

 
 ATMGNN   Nguyen et al.,2023 | 
 MPNN/MGNN + LSTM/Transformer | 
 | 

 
 DASTGN   Pu et al.,2023 | 
 GNN + Attention + GRU | 
 | 

 
 MSGNN   Qiu et al.,2023 | 
 GCN + N-Beats | 
 | 

 
 STEP   Yu et al.,2023 | 
 GCN + Attention + GRU | 
 | 

 
   Panagopoulos et al.,2021 | 
 GNN + LSTM | 
 | 

 
   Mežnar et al.,2021 | 
 Network Centrality + XGBoost | 
 | 

 
   Tomy et al.,2022a | 
 GNN + SEIRD | 
 ✓ | 

 
   La et al.,2021 | 
 GNN + LSTM + SIRD | 
 ✓ | 

 
 GSRNN   Li et al.,2019 | 
 GNN + RNN | 
 | 
 Spatial-Temporal Graph; Static Graph Structure | 

 
   Fritz et al.,2022 | 
 GNN + SDDR | 
 ✓ | 

 
   Xie et al.,2022a | 
 GCN + Modified SIR | 
 ✓ | 

 
   Moon et al.,2023b | 
 K-GNN + GRU | 
 | 

 
   Sesti et al.,2021 | 
 GNN + LSTM | 
 | 

 
   Davahli et al.,2021 | 
 GNN + LSTM | 
 | 

 
   Murphy et al.,2021 | 
 GNN | 
 | 

 
   Kapoor et al.,2020 | 
 GNN + MLP | 
 | 

 
   Croft et al.,2023 | 
 GAT + GRU | 
 | 

 
 Mepo GNN  Cao et al.,2023 ; Cao et al.,2022 | 
 (TCN + GCN) + Modified SIR | 
 ✓ | 
 Spatial-Temporal Graph; Dynamic Graph Structure | 

 
 Epi-Cola-GNN   Liu et al.,2023a | 
 Cola-GNN + Modified SIR | 
 ✓ | 

 
 HiSTGNN   Ma et al.,2022 | 
 Hierarchical GNN + Transformer | 
 | 

 
 CausalGNN   Wang et al.,2022 | 
 GNN + SIRD | 
 ✓ | 

 
 ATGCN   Wang et al.,2022b | 
 GNN + LSTM | 
 | 

 
 HierST   Zheng et al.,2021 | 
 GNN + LSTM | 
 | 

 
 RESEAT   Moon et al.,2023 | 
 GNN + Self-Attention | 
 | 

 
 SAIFlu-Net   Jung et al.,2022 | 
 GNN + LSTM | 
 | 

 
 Epi-GNN   Xie et al.,2022 | 
 GCN + Attention + RNN | 
 | 

 
 Cola-GNN   Deng et al.,2020 | 
 GCN + Attention + RNN | 
 | 

 
   Cui et al.,2021 | 
 GNN + Attention + LSTM | 
 | 

 
   Mahmud et al.,2021 | 
 GNN + RNN | 
 | 

 
   Guo Yang,2021 | 
 STGCN | 
 | 

 
   Duarte et al.,2023 | 
 GCRN/GCLSTM | 
 |