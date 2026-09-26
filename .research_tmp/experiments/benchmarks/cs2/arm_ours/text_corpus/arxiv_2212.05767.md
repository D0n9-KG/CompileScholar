A Survey of Knowledge Graph Reasoning on Graph Types: Static, Dynamic, and Multi-Modal 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.05767v7 [cs.AI] 22 Jul 2023 
 
 

# A Survey of Knowledge Graph Reasoning on Graph Types: Static, Dynamic, and Multi-Modal

 
 
 Ke Liang,
Lingyuan Meng,
Meng Liu,
Yue Liu,
Wenxuan Tu, Siwei Wang, Sihang Zhou
 
    
 Xinwang Liu
 
    
 Fuchun Sun
 † † thanks: 
$ˆ†$ Corresponding Author.
Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, and Xinwang Liu are with the School of Computer, National University of Defense Technology, Changsha, 410073, China. E-mail: xinwangliu@nudt.edu.cn.
Sihang Zhou is with the College of Intelligence Science and Technology, National University of Defense Technology, Changsha, 410073, China.
Fuchun Sun is with the Department of Computer Science and Technology, Tsinghua University, Beijing, 100084, China. † † thanks: This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. 

 Abstract 
 
 Knowledge graph reasoning (KGR), aiming to deduce new facts from existing facts based on mined logic rules underlying knowledge graphs (KGs), has become a fast-growing research direction. It has been proven to significantly benefit the usage of KGs in many AI applications, such as question answering, recommendation systems, and etc. According to the graph types, existing KGR models can be roughly divided into three categories, i.e., static models, temporal models, and multi-modal models. Early works in this domain mainly focus on static KGR, and recent works try to leverage the temporal and multi-modal information, which are more practical and closer to real-world. However, no survey papers and open-source repositories comprehensively summarize and discuss models in this important direction. To fill the gap, we conduct a first survey for knowledge graph reasoning tracing from static to temporal and then to multi-modal KGs. Concretely, the models are reviewed based on bi-level taxonomy, i.e., top-level (graph types) and base-level (techniques and scenarios). Besides, the performances, as well as datasets, are summarized and presented. Moreover, we point out the challenges and potential opportunities to enlighten the readers. The corresponding open-source repository is shared on GitHub https://github.com/LIANGKE23/Awesome-Knowledge-Graph-Reasoning.

 
 
 
 Index Terms: Knowledge Graph Reasoning, Knowledge Graph, Temporal Knowledge Graph, Multi-Modal Knowledge Graph.

 
 

## I Introduction 

 
 Humans learn skills from two main sources, i.e., specialized books and working experiences. For example, a good doctor needs to get knowledge from school and practice experiences from the hospital. However, most existing artificial intelligence (AI) models only imitate the learning procedure from experiences while ignoring the former [ 1 , 2 , 3 , 4 ] , thus making them less explainable and worse performances. Knowledge graphs (KGs) [ 5 ] , which store the human knowledge facts in intuitive graph structures [ 6 ] , are treated as potential solutions these years. While, the construction of KGs is a dynamic and continuous procedure, thus most KGs suffer from incomplete issues, hindering their effectiveness in KG-assisted applications, such as question answering [ 7 ] , recommendation system [ 8 ] , etc. To alleviate the problem, knowledge graph reasoning (KGR) has drawn increasing attention these years. It aims to infer missing facts from existing ones in KGs. Taking Figure 1 (a) as the target KG, KGR models are expected to derive the logic rules (A, father of, B) ∧ \wedge (A, husband of, C) → \rightarrow (C, mother of, B) , and then further infer the missing fact (Savannah, mother of, Bronny) .

 
 
 According to the information types in KGs, current KGs can be roughly divided into three categories, i.e., static KGs, temporal KGs, and multi-modal KGs, as shown in Figure 1 . Traditional KGs only contain static uni-modal facts, which is simple but effective for developing general basic KGR models. However, they still cannot fully describe real-world scenarios, which consist of information from various sources. Thus, recent KGs ( i.e., temporal KGs and multi-modal KGs) are constructed by integrating extra temporal and multi-modal information based on static KGs, which is more practical and closer to the real world. However, no matter which types of KGs they are, the incomplete problem still exists. Therefore, various advanced KGR models have been continuously developed and studied these years for better reasoning performance. Of course, it is worth noting that the core issues of KGR models for different KG types are different. Specifically, static KGR models focus on the general representation learning capacity. While, how to fuse extra information well is the key to the temporal and multi-modal KGR models. Moreover, for a more comprehensive and systematical review, two sub-taxonomies, i.e., reasoning techniques and reasoning scenarios, are further discussed within each graph type.

 
 
 Figure 1: Examples of three categories of the knowledge graphs, i.e., static, temporal, and multi-modal knowledge graph. 
 
 
 Figure 2: Overview framework of the survey. 
 
 
 Table I: Comparison between different KGR surveys. 
 
 
 
 
 Survey | 
 [ 9 ] | 
 [ 10 ] | 
 [ 11 ] | 
 [ 12 ] | 
 [ 13 ] | 
 [ 14 ] | 
 [ 15 ] | 
 Ours | 

 
 Static KGR | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 | 
 | 
 ✓ \checkmark | 

 
 Embedding-based Model | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 | 
 | 
 ✓ | 

 
 Path-based Model | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 | 
 | 
 | 
 ✓ | 

 
 Rule-based Model | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 | 
 | 
 | 
 ✓ | 

 
 Transductive Scenario | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 ✓ | 

 
 Inductive Scenario | 
 | 
 | 
 | 
 | 
 ✓ \checkmark | 
 | 
 | 
 ✓ | 

 
 Temporal KGR | 
 | 
 | 
 | 
 | 
 | 
 ✓ \checkmark | 
 | 
 ✓ | 

 
 RNN-based Model | 
 | 
 | 
 | 
 | 
 | 
 ✓ \checkmark | 
 | 
 ✓ | 

 
 RNN-agnostic Model | 
 | 
 | 
 | 
 | 
 | 
 ✓ \checkmark | 
 | 
 ✓ | 

 
 Interpolation Scenario | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 ✓ | 

 
 Extrapolation Scenario | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 ✓ | 

 
 Multi-Modal KGR | 
 | 
 | 
 | 
 | 
 | 
 | 
 ✓ \checkmark | 
 ✓ | 

 
 Non-Transformer Model | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 ✓ | 

 
 Transformer Model | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 ✓ | 

 

 
 
 
 There are several survey papers for KGR. Most of them only focus on static KGR but omit the recent progress in other KGs, i.e., temporal KGs and multi-modal KGs. [ 9 ] first categorizes KGR tasks into symbolic and statistical reasoning. Besides, [ 10 ] divides KGR models into three types, i.e., symbolic, neural, and hybrid. After that, [ 11 ] and [ 12 ] propose more fine-grained categorizations for logic-based and embedding-based KGR models. More recently, [ 13 ] analyzes the generalization ability of KGR models to unseen elements. As for temporal KGR, [ 14 ] reviews existing models based on how fact timestamps capture the temporal dynamics. But it does not clearly distinguish the interpolation and extrapolation scenarios for temporal KGR. As the most influential multi-modal KG survey, [ 15 ] focuses more on construction and applications than reasoning over them. Compared to those existing surveys, we conduct a more comprehensive survey for knowledge graph reasoning (See Table I ), tracing from static to temporal and then to multi-modal KGs. More specifically, a bi-level taxonomy is leveraged for review, i.e., top level (graph types) and base level (techniques and scenarios). In particular, we carefully discuss reasoning scenarios for the reviewed models, i.e., transductive and inductive scenario for static KGR, and interpolation and extrapolation scenario for temporal KGR.

 
 
 To summarize, we are the first to thoroughly survey the existing KGR models over different graph types, including traditional KGs (static KGs) and KGs with extra information (temporal KGs and multi-modal KGs). Specifically, we first introduce the preliminary (Sec. 2). Next, we systematically review the recent state-of-the-art KGR models from Sec. 3 to Sec.5 based on the bi-level taxonomy and performances. Later on, we organize and collect typical KGR datasets in Sec. 6. Then, the challenges and potential opportunities are pointed out in Sec. 7. Finally, Sec. 8 concludes the paper. To enhance the value of this survey, we summarize the main contributions as follows:

 
 • 
 
 Comprehensive Review . We comprehensively investigate typical KGR models based on a bi-level taxonomy, i.e., top-level (graph types), and base-level (techniques, scenarios). Three graph types ( i.e., static, temporal, multi-modal KGs), fourteen techniques, and four reasoning scenarios are included, which provides systematical reviews for KGR.

 

 • 
 
 Insightful Analysis . We analyze the strengths and weaknesses of the existing KGR models and their suitable scope, which will provide the readers with useful guidance to select the baselines for their research.

 

 • 
 
 Potential Opportunity . We summarize the challenges of knowledge graph reasoning and point out some potential opportunities which will enlighten the readers.

 

 • 
 
 Open-source Resource . We share the collection of 180 state-of-the-art KGR models ( i.e., papers and codes) and 67 typical datasets on GitHub 1 1 
 1 
 
 
 https://github.com/LIANGKE23/Awesome-Knowledge-Graph-Reasoning .

 

 
 
 
 

## II Preliminary 

 
 In this section, we first formally define static, temporal, and multi-modal knowledge graphs. Then, the reasoning tasks over the different types of KGs and scenarios are formulated. At last, we introduce the taxonomy criterion of KGR models.

 
 
 Table II: Notation summary 
 
 
 
 
 Notation | 
 Explanation | 

 
 
 
 𝒮 ​ 𝒦 ​ 𝒢 \mathcal{SKG} | 
 Static knowledge graph | 

 
 𝒯 ​ 𝒦 ​ 𝒢 \mathcal{TKG} | 
 Temporal knowledge graph | 

 
 ℳ ​ 𝒦 ​ 𝒢 \mathcal{MKG} | 
 Multi-modal knowledge graph | 

 
 ℰ \mathcal{E} | 
 Entity set | 

 
 ℛ \mathcal{R} | 
 Relation set | 

 
 ℱ \mathcal{F} | 
 The set of facts, i.e., edges | 

 
 𝒯 \mathcal{T} | 
 The set of the time stamps | 

 
 ℱ t \mathcal{F}_{t} | 
 The set of facts at time t t | 

 
 ( e h , r , e t ) (e_{h},r,e_{t}) | 
 Fact triplet of the head, relation, tail. | 

 
 ( e h , r , e t , t ) (e_{h},r,e_{t},t) | 
 Fact quadruple of the head, relation, tail, timestamp | 

 
 ( e h q , r q , e t q ) (e_{h}^{q},r^{q},e_{t}^{q}) | 
 Queried fact triplet of head, relation, tail | 

 
 e | 
 Embedding of entity | 

 
 r | 
 Embedding of relation | 

 
 t | 
 Embedding of timestamp | 

 

 
 
 

### II-A Definition and Notation 

 
 Knowledge graphs (KGs) can be viewed as graphical knowledge bases, thus inheriting most functions of traditional knowledge bases [ 16 ] , such as storing, indexing, etc, but in a more intuitive manner. Existing KGs can be roughly divided into three types, i.e., static, temporal, multi-modal KGs. Following previous literature, the definitions of them are declared below and the notations are summarized in Table II .

 
 
 Definition 1 . 
 
 Static Knowledge Graph .
 Static knowledge graph (KG) is defined as 𝒮 ​ 𝒦 ​ 𝒢 \mathcal{SKG} = { ℰ \{\mathcal{E} , ℛ \mathcal{R} , ℱ } \mathcal{F}\} , where ℰ , ℛ \mathcal{E},\ \mathcal{R} , and ℱ \mathcal{F} represent the sets of entities, relations, and facts. The fact is in a triplet format ( e h , r , e t ) ∈ ℱ (e_{h},r,e_{t})\in\mathcal{F} , where e h , e t ∈ ℰ e_{h},e_{t}\in\mathcal{E} , and r ∈ ℛ r\in\mathcal{R} between them. Note that static KGs are known as traditional KGs in [ 9 ] . Phrase ”static” is to distinguish it from other KG types. 

 
 
 
 Definition 2 . 
 
 Temporal Knowledge Graph .
 Temporal knowledge graph (KG) is defined as a sequence of static KGs at different timestamps 𝒯 ​ 𝒦 ​ 𝒢 \mathcal{TKG} = { 𝒮 ​ 𝒦 ​ 𝒢 1 , 𝒮 ​ 𝒦 ​ 𝒢 2 , 𝒮 ​ 𝒦 ​ 𝒢 3 , ⋯ , 𝒮 ​ 𝒦 ​ 𝒢 t } \{\mathcal{SKG}_{1},\mathcal{SKG}_{2},\mathcal{SKG}_{3},\cdots,\mathcal{SKG}_{t}\} . The KG snapshot at timestamp t t is defined as 𝒮 ​ 𝒦 ​ 𝒢 t \mathcal{SKG}_{t} = { ℰ \{\mathcal{E} , ℛ \mathcal{R} , ℱ t } \mathcal{F}_{t}\} , where ℰ , ℛ \mathcal{E},\mathcal{R} are the sets of entities and relations, ℱ t \mathcal{F}_{t} is the set of facts at timestamp t ∈ 𝒯 t\in\mathcal{T} . The quadruple fact ( e h , r , e t , t ) (e_{h},r,e_{t},t) represents that relation r r exists between head e h e_{h} and tail e t e_{t} at timestamp t t . 

 
 
 
 Definition 3 . 
 
 Multi-Modal Knowledge Graph .
 Multi-modal knowledge graph (KG) ℳ ​ 𝒦 ​ 𝒢 \mathcal{MKG} is composed of knowledge facts where more than one modalities exist. According to the representation mode of other modal data, there are two multi-modal KG [ 17 ] , i.e., N-MMKG and A-MMKG (See Figure 3 ). 

 
 
 
 Figure 3: Comparison between two types of multi-modal knowledge graphs. N-MMKG represents the multi-modal data as entities, while A-MMKG represents multi-modal data as new attributes. 
 
 
 

### II-B Task Formulation 

 
 Knowledge graph reasoning (KGR) aims to deduce new facts from existing facts based on the derived underlying logic rules. According to graph types, KGR can be categorized into three tasks, i.e., static, temporal, and multi-modal KGR. Among them, since extra time and visual information are integrated into temporal and multi-modal KGs, respectively, there exist slight differences in task formulation compared to static KGR. Besides, two groups of terms of the reasoning scenarios are also introduced, i.e., transductive \ inductive scenarios and interpolation \ extrapolation scenarios, for a better understanding of our taxonomy.

 
 
 Static Knowledge Graph Reasoning 

 
 Given a static KG 𝒮 ​ 𝒦 ​ 𝒢 \mathcal{SKG} = { ℰ \{\mathcal{E} , ℛ \mathcal{R} , ℱ } \mathcal{F}\} , KGR aims to exploit the existing facts to infer queried fact ( e h q , r q , e t q ) (e_{h}^{q},r_{q},e_{t}^{q}) based on the likelihood calculated by scoring functions. According to the type of missing elements, there are three sub-tasks, i.e., head reasoning ( ? , r q , e t q ) (?,r^{q},e_{t}^{q}) , tail inferring ( e h q , r q , ? ) (e_{h}^{q},r^{q},?) , and relation inferring ( e h q , ? , e t q ) (e_{h}^{q},?,e_{t}^{q}) .

 
 
 
 Temporal Knowledge Graph Reasoning 

 
 Given a temporal KG 𝒯 ​ 𝒦 ​ 𝒢 \mathcal{TKG} = { 𝒮 ​ 𝒦 ​ 𝒢 1 , 𝒮 ​ 𝒦 ​ 𝒢 2 , ⋯ , 𝒮 ​ 𝒦 ​ 𝒢 t } \{\mathcal{SKG}_{1},\mathcal{SKG}_{2},\cdots,\mathcal{SKG}_{t}\} , where 𝒮 ​ 𝒦 ​ 𝒢 t \mathcal{SKG}_{t} = { ℰ \{\mathcal{E} , ℛ \mathcal{R} , ℱ t } \mathcal{F}_{t}\} and timestamp t ∈ 𝒯 t\in\mathcal{T} , KGR aims to infer the quadruple fact ( e h q , r q , e t q , t q ) (e_{h}^{q},r^{q},e_{t}^{q},t^{q}) . Similar to static KGR, there also exist three sub-tasks at specific timestamp t q t^{q} .

 
 
 
 Multi-Modal Knowledge Graph Reasoning 

 
 Multi-modal KGR is similar to the other two KGR types, i.e., inferring missing facts. Besides, extra fusion modules are usually required to leverage the multi-modal information for better reasoning performance.

 
 
 
 Transductive and Inductive Reasoning Scenarios 

 
 According to the visibility of queried entities and relations during training, there are two types of reasoning scenarios (See Figure 4 ), i.e., transductive and inductive scenarios. Within transductive scenarios, entities and relationships in the queried fact are all seen in the given KG, i.e., e h q , e t q ∈ ℰ e_{h}^{q},e_{t}^{q}\in\mathcal{E} and r q ∈ ℛ r^{q}\in\mathcal{R} . As for inductive scenarios, the candidates for e h q , e t q e_{h}^{q},e_{t}^{q} and r q r^{q} may be beyond the given KG. These two scenarios are usually discussed in static KGR.

 
 
 Figure 4: Illustration of transductive and inductive reasoning. In the transductive scenario, entities in test graphs are all seen during the training procedure. While as for the inductive scenario, unseen entities may exist in test graphs. 
 
 
 Figure 5: Illustration of interpolation and extrapolation reasoning. The timestamp t t for knowledge graph reasoning in the interpolation scenario is seen in the past ( 0 ≤ t ≤ T 0\leq t\leq T ). While the queried facts in the future ( t ≥ T t\geq T ) for the extrapolation scenario. 
 
 
 
 Interpolation and Extrapolation Reasoning Scenarios 

 
 According to the occurrence time of the queried fact t q t^{q} , temporal KGR can be divided into two categories (See Figure 5 ), i.e., interpolation and extrapolation scenarios. Concretely, given a temporal KG with the timestamps ranging from time 0 0 to time T T , the interpolation reasoning aims to infer the queried facts for time t t , where 0 ≤ t ≤ T 0\leq t\leq T ; besides, the extrapolation reasoning aims to infer the queried facts for time t t , where t ≥ T t\geq T . These two scenarios are usually discussed in temporal KGR.

 
 
 Figure 6: The bi-level taxonomy of reviewed KGR models. 
 
 
 
 

### II-C Taxonomy Design 

 
 We design a bi-level taxonomy to review existing KGR models for different KGs systematically. Specifically, three taxonomy criterion is adopted to classify the reviewed KGR models, i.e., graph types, techniques, and scenarios. As shown in Figure 6 , Graph types are the top-level taxonomy, which contains three KG types, i.e., static, temporal, and multi-modal KGs. Additionally, the base-level taxonomy consists of techniques (fourteen types) and scenarios (four types). Given a KGR model, we will first categorize it based on the top-level taxonomy. Then, we will further classify it according to its specific techniques and scenarios. As mentioned before, KGR, over different graph types, focus on different techniques and scenarios. Concretely, (1) as for techniques, embedding-based (five sub-types), path-based, and rule-based models are typical techniques for static KGR (See Figure 7 ). Besides, RNN-based (three sub-types) and RNN-agnostic (two sub-types) models are the designed technique criterion for temporal KGR models (See Figure 9 ). Moreover, Transformer-based and Transformer-agnostic are the adopted technique criterion for multi-modal KGR models (See Figure 10 ). (2) As for reasoning scenarios, we only discuss this taxonomy in static and temporal KGR, i.e., transductive and inductive scenarios for static KGR, interpolation and extrapolation scenarios for temporal KGR.

 
 
 
 

## III Static KGR Model 

 
 We systematically introduce 90 static KGR models based on techniques and scenarios (See Table III ).

 
 

### III-A Review on Reasoning Techniques 

 
 Static KGR models can be categorized into embedding-based, path-based, and rule-based models. Details are described below.

 
 

#### III-A 1 Embedding-based Model

 
 Embedding-based models learn the embedding vectors based on existing fact triplets and then rank top k k candidate facts based on the likelihood calculated by scoring functions. In general, there are three types, i.e., translational, tensor decompositional, and neural network models. Due to the majority quantity, the timeline of embedding-based models is present in Figure 8 for clear presentation.

 
 
 Translational Model 

 
 Translational models regard relation r r as translational transformation to project entity e into the latent space.

 
 
 TransE [ 18 ] , as the first translational model, regards the relation as a simple translation operation, e h e_{h} + r r ≈ \approx e t e_{t} . Although proven effective, it cannot handle some specific relations, such as one-to-many, many-to-one, symmetric and transitive relations. To address these limitations, lots of translational models are developed. The entities are encoded into the relation-specific hyperplane in TransH [ 19 ] , which achieves better reasoning performance on one-to-many, many-to-one relations. Besides, TransR [ 20 ] leverages distinct latent spaces for entities and relations and gets better expressive ability on transitive relation reasoning. Moreover, TransD [ 21 ] first considers the scalability issues by leveraging independent projection vectors for entities and relations for large KGs. Afterward, probabilistic principles are integrated to model the uncertainty in KGR. For example, KG2E [ 22 ] leverages the Gaussian distribution covariance, and TransG [ 23 ] makes use of the Bayesian technique for one-to-many relational facts. Meanwhile, to alleviate heterogeneity and imbalance issues, TranSparse [ 24 ] provides an efficient solution by designing adaptive transfer sparser matrices, thus leading to better expressive ability. After that, TorusE [ 25 ] projects embeddings in a compact Lie group torus, and MuRP [ 26 ] designs a Möbius matrix-vector multiplication and Möbius addition for entity embedding projection, which all show better accuracy and scalability. Meanwhile, TransW [ 27 ] first has enriched the entity and relation embedding with the word embeddings, achieving better performance in inferring facts with unseen entities or relations. Moreover, RotatE [ 28 ] proposes a rotation-based translational method with complex-valued embeddings to better infer the symmetry, anti-symmetry, inversion, and composition facts. Besides, HAKE [ 29 ] models the semantic hierarchy rather than relation patterns based on the polar coordinate space. Then, TransRHS [ 30 ] first considers the Relation Hierarchical Structure (RHS) by incorporating RHS seamlessly into the embeddings. Besides, to handle the complex relational facts with a unified model, PairRE [ 31 ] models each relation representation with paired vectors to adaptive adjustment for complex relations, and HousE [ 32 ] involves a novel parameterization based on the designed Householder transformations for rotation and projection. Nowadays, there are also some interesting attempts for translational models for more sufficient interactions, such as TripleRE [ 33 ] and InterHT [ 34 ] . TripleRE creatively divides the relationship vector into three parts, takes advantage of the concept of residual, and achieves better performance. While, InterHT enhances the information interactions between the tail and head, improving the model capacity.

 
 
 Figure 7: Taxonomy of the static KGR models. 
 
 
 
 Tensor Decompositional Model 

 
 Tensor decompositional models encode the KGs as three-way tensors, decomposed into a combination of low-dimensional vectors for entities and relations.

 
 
 Figure 8: Timeline of the embedding-based models for static KGR. 
 
 
 As the first tensor decompositional model, RESCAL [ 35 ] captures the latent semantics of each entity with vectors and further leverages the matrix to model the pairwise interactions among latent factors as a matrix. However, the model is complex with O ⁡ ( d 2 ) O(d^{2}) parameters. To simplify it, DistMult [ 36 ] uses bi-linear diagonal matrices to reduce parameters to O ⁡ ( d ) O(d) per relation. Then, ComplEx [ 37 ] generalizes DistMult by using complex-valued embeddings, which improve asymmetric relations modeling. Meanwhile, HolE [ 38 ] models the holographic reduced representations and circular correlation, and Analogy [ 39 ] designs the bi-linear scoring function with analogical structural constraints for analogical reasoning, which both try to capture rich interactions between entities. Then, some models start to substitute the decompositional operations. SimplE [ 40 ] enhances the Canonical Polyadic (CP) decompositional for two independent entity embeddings, and Tucker Decomposition is first used by Tucker [ 41 ] . Meanwhile, CrossE [ 42 ] considers crossover interactions between entities via a relation-specific interaction matrix. QuatE [ 43 ] empowers the semantic matching between head and tail based on relational rotation quaternion representations. Inspired by it, DualE [ 44 ] projects the embeddings in dual quaternion space to achieve a unified framework for both translation and rotation operations. Besides, HopfE [ 45 ] makes use of both structural and semantic attributes in 4D hyper-sphere space without losing interpretability. Besides expressive ability, efficiency also draws increasing attention these years. A factorized bi-linear pooling model is proposed based on Tucker decomposition, termed LowFER [ 46 ] , which is more efficient and lightweight. Moreover, QuatRE [ 47 ] learns the quaternion embeddings with two novel operations, i.e., enhancing the correlations between entities with Hamilton product for entity embeddings and reducing computation by simplifying the translation matrices.

 
 
 
 Neural Network Model 

 
 Neural network (NN) models have yielded remarkable performance for KG reasoning these years. Three sub-types are defined based on the NN techniques, i.e., traditional neural network (NN), convolutional neural network (CNN), and graph neural network (GNN) models.

 
 
 1) Traditional NN Model : SME [ 48 ] first encodes the entities and relations into the latent space using neural networks. Meanwhile, neural tensor networks are used in NTN [ 49 ] for relation reasoning in KGs. Then, NAM [ 50 ] proposes the relational-modulated neural network (RMNN), and NN models shares variables in ProjE [ 51 ] , which jointly learns embeddings of the entities and relations via the standard loss function. These traditional NN models show great potential on static KGR while they suffer from learning shallow and less expressive features.

 
 
 2) CNN Model: To learn deeper features, convolutional neural networks (CNNs) are integrated with KGR models. ConvE [ 52 ] first leverages 2D convolutional layers for KGR. ConvKB [ 53 ] extends ConvE by removing the reshaping operation and captures global and transitional characteristics within facts for informative expression. Later on, HypER [ 54 ] uses fully connected layers and relation-specific convolutional filters for better performance. Besides, ConvR [ 55 ] designs an adaptive convolutional network designed to maximize entity-relation interactions by constructing convolution filters across the entity and relation representations. Moreover, novel operations, i.e., feature reshaping, feature permutation, and circular convolution, are designed in InteractE [ 56 ] to handle complex interactions. Meanwhile, ConEx [ 57 ] integrates the affine transformation and a Hermitian inner product on complex-valued embeddings with the convolutional operation, which shows good expressiveness. CNN models generally perform better than traditional NN models. However, the information underlying graph structures cannot be well learned.

 
 
 3) GNN Model: Graph neural networks, which are widely used for graph tasks, are also rapidly applied to KG reasoning. RGCN [ 58 ] uses the relation-specific transformation to aggregate neighborhood information. Then, each entity is encoded into a vector, and the decoder i.e., scoring function, reconstructs the facts based on entity representations. While RGCN omits the variances of entities, which hinders expressive ability. To alleviate it, attention mechanisms are integrated into lots of models, such as M-GNN [ 59 ] , KBGAT [ 60 ] , and [ 61 ] . In particular, KBGAT [ 60 ] leverages attention-based feature embeddings for better reasoning performance. Meanwhile, SACN [ 62 ] leverages the weighted graph convolutional network (WGCN) as the encoder and a convolutional network called Conv-TransE as the decoder, which is effective. Afterward, TransGCN [ 63 ] trains both relation and entity embeddings simultaneously with the transformation operator for relations. Later on, DPMPN [ 64 ] and RGHAT [ 65 ] designs a two-GNN framework to simultaneously encode information in different levels separately, i.e., global \ local information for DPMPN and relation \ entity information for RGHAT. After that, KE-GCN [ 66 ] jointly propagates and updates the embedding of both entities and edges. Similarly, COMPGCN [ 67 ] also jointly learns the representations with various entity-relation composition operations. Recently, more and more researchers have tried to handle out-of-knowledge-graph scenarios. GEN [ 68 ] and HRFN [ 69 ] learn entity embeddings based on meta-learning for both seen-to-unseen and unseen-to-unseen facts. INDIGO [ 70 ] is then proposed based on a GNN using pair-wise encoding. Besides, the GraIL-based model is a group of typical GNN models for inductive scenarios. The prototype GraIL [ 71 ] , as the landmark GNN-based model, first leverages RGCN to perform the reasoning based on the local enclosing subgraph. Based on it, many incremental works are developed, including TACT [ 72 ] , CoMPILE [ 73 ] , Meta-iKG [ 74 ] , SNRI [ 75 ] , RPC-IR [ 76 ] , and etc. These GraIL-based models all achieve promising inductive performances. Among them, TACT [ 72 ] and CoMPILE [ 73 ] both raise the importance of relation embeddings in the task. Concretely, TACT [ 72 ] uses topology-aware correlations between relations to generate representations for triplet scoring, which also inspires RMPI [ 77 ] and TEMP [ 78 ] . Besides, CoMPILE enhances the message interactions between relations and entities with a novel mechanism. After that, motivated by the great success of contrastive mechanisms [ 79 , 6 ] , contrastive learning models have been increasingly proposed, e.g., RPC-IR [ 76 ] , SNRI [ 75 ] etc. Besides, Meta-iKG [ 74 ] verifies the effectiveness of meta-learning in the KGR task. After that, researchers try to make the reasoning more efficient. NBF-net [ 80 ] and RED-GNN [ 81 ] achieve better efficiency by leveraging the traditional algorithm i.e., bellman-ford algorithm and dynamic programming to optimize the propagation strategy in previous GNN models. Besides, pGAT [ 82 ] leverages the EM algorithm for efficient learning. Moreover, BERTRL [ 83 ] and ConGLR [ 84 ] integrates the context for each entity to enhance the reasoning on the KGs. In particular, BERTRL [ 83 ] can handle unseen relational facts. Regarding it, CSR [ 85 ] deeply mines the logic rules underlying the structure patterns instead of the paths.

 
 
 Table III: Summary of the static knowledge graph reasoning models. 
 
 
 
 
 Year | 
 Model | 
 Scenario | 
 Technique | 
 | 
 Year | 
 Model | 
 Scenario | 
 Technique | 

 
 
 
 2022 | 
 LogCo [ 86 ] | 
 Inductive | 
 GNN | 
 | 
 2019 | 
 M-GNN [ 59 ] | 
 Transductive | 
 GNN | 

 
 2022 | 
 REPORT [ 87 ] | 
 Inductive | 
 GNN | 
 | 
 2019 | 
 SACN [ 62 ] | 
 Transductive | 
 GNN | 

 
 2022 | 
 RED-GNN [ 81 ] | 
 Inductive | 
 GNN | 
 | 
 2019 | 
 KBGAT [ 60 ] | 
 Transductive | 
 GNN | 

 
 2022 | 
 ConGLR [ 84 ] | 
 Inductive | 
 GNN | 
 | 
 2019 | 
 LAN [ 61 ] | 
 Inductive | 
 GNN | 

 
 2022 | 
 TripleRE [ 33 ] | 
 Transductive | 
 Translational | 
 | 
 2019 | 
 CPL [ 88 ] | 
 Transductive | 
 Relation Path | 

 
 2022 | 
 InterHT [ 34 ] | 
 Transductive | 
 Translational | 
 | 
 2019 | 
 IterE [ 89 ] | 
 Inductive | 
 Logic Rule | 

 
 2022 | 
 HousE [ 32 ] | 
 Transductive | 
 Translational | 
 | 
 2019 | 
 pLogicNet [ 90 ] | 
 Inductive | 
 Logic Rule | 

 
 2022 | 
 BERTRL [ 83 ] | 
 Inductive | 
 GNN | 
 | 
 2019 | 
 DRUM [ 91 ] | 
 Inductive | 
 Logic Rule | 

 
 2022 | 
 SNRI [ 75 ] | 
 Inductive | 
 GNN | 
 | 
 2019 | 
 RLvLR [ 92 ] | 
 Inductive | 
 Logic Rule | 

 
 2022 | 
 TEMP [ 78 ] | 
 Inductive | 
 GNN | 
 | 
 2019 | 
 Neural-Num-LP [ 93 ] | 
 Inductive | 
 Logic Rule | 

 
 2022 | 
 RMPI [ 77 ] | 
 Inductive | 
 GNN | 
 | 
 2018 | 
 SimplE [ 40 ] | 
 Transductive | 
 Tensor Decompositional | 

 
 2022 | 
 Meta-iKG [ 74 ] | 
 Inductive | 
 GNN | 
 | 
 2018 | 
 ConvKB [ 53 ] | 
 Transductive | 
 CNN | 

 
 2022 | 
 CSR [ 85 ] | 
 Inductive | 
 GNN | 
 | 
 2018 | 
 ConvE [ 52 ] | 
 Transductive | 
 CNN | 

 
 2022 | 
 CURL [ 94 ] | 
 Transductive | 
 Relation Path | 
 | 
 2018 | 
 RGCN [ 58 ] | 
 Transductive | 
 GNN | 

 
 2022 | 
 GCR [ 95 ] | 
 Inductive | 
 Logic Rule | 
 | 
 2018 | 
 M-walk [ 96 ] | 
 Transductive | 
 Relation Path | 

 
 2021 | 
 PairRE [ 31 ] | 
 Transductive | 
 Translational | 
 | 
 2018 | 
 MultiHop [ 97 ] | 
 Transductive | 
 Relation Path | 

 
 2021 | 
 HopfE [ 45 ] | 
 Transductive | 
 Tensor Decompositional | 
 | 
 2018 | 
 DIVA [ 98 ] | 
 Transductive | 
 Logic Rule | 

 
 2021 | 
 DualE [ 44 ] | 
 Transductive | 
 Tensor Decompositional | 
 | 
 2018 | 
 RuleN [ 99 ] | 
 Inductive | 
 Logic Rule | 

 
 2021 | 
 ConEx [ 57 ] | 
 Transductive | 
 CNN | 
 | 
 2018 | 
 RUGE [ 100 ] | 
 Inductive | 
 Logic Rule | 

 
 2021 | 
 KE-GCN [ 66 ] | 
 Transductive | 
 GNN | 
 | 
 2017 | 
 ANALOGY [ 39 ] | 
 Transductive | 
 Tensor Decompositional | 

 
 2021 | 
 HRFN [ 69 ] | 
 Inductive | 
 GNN | 
 | 
 2017 | 
 ProjE [ 51 ] | 
 Transductive | 
 Traditional NN | 

 
 2021 | 
 GEN [ 68 ] | 
 Inductive | 
 GNN | 
 | 
 2017 | 
 MINERVA [ 101 ] | 
 Transductive | 
 Relation Path | 

 
 2021 | 
 INDIGO [ 70 ] | 
 Inductive | 
 GNN | 
 | 
 2017 | 
 DeepPath [ 102 ] | 
 Transductive | 
 Relation Path | 

 
 2021 | 
 NBF-Net [ 80 ] | 
 Inductive | 
 GNN | 
 | 
 2017 | 
 NTP [ 103 ] | 
 Inductive | 
 Logic Rule | 

 
 2021 | 
 CoMPILE [ 73 ] | 
 Inductive | 
 GNN | 
 | 
 2017 | 
 NeuralLP [ 104 ] | 
 Inductive | 
 Logic Rule | 

 
 2021 | 
 TACT [ 72 ] | 
 Inductive | 
 GNN | 
 | 
 2016 | 
 TranSparse [ 24 ] | 
 Transductive | 
 Translational | 

 
 2021 | 
 RPC-IR [ 76 ] | 
 Inductive | 
 GNN | 
 | 
 2016 | 
 TransG [ 23 ] | 
 Transductive | 
 Translational | 

 
 2020 | 
 HAKE [ 29 ] | 
 Transductive | 
 Translational | 
 | 
 2016 | 
 HolE [ 38 ] | 
 Transductive | 
 Tensor Decompositional | 

 
 2020 | 
 TransRHS [ 30 ] | 
 Transductive | 
 Translational | 
 | 
 2016 | 
 ComplEx [ 37 ] | 
 Transductive | 
 Tensor Decompositional | 

 
 2020 | 
 LowFER [ 46 ] | 
 Transductive | 
 Tensor Decompositional | 
 | 
 2016 | 
 NAM [ 50 ] | 
 Transductive | 
 Traditional NN | 

 
 2020 | 
 InteractE [ 56 ] | 
 Transductive | 
 CNN | 
 | 
 2016 | 
 LogSumExp [ 105 ] | 
 Transductive | 
 Relation Path | 

 
 2020 | 
 DPMPN [ 64 ] | 
 Transductive | 
 GNN | 
 | 
 2016 | 
 KALE [ 106 ] | 
 Inductive | 
 Logic Rule | 

 
 2020 | 
 RGHAT [ 65 ] | 
 Transductive | 
 GNN | 
 | 
 2015 | 
 TransD [ 21 ] | 
 Transductive | 
 Translational | 

 
 2020 | 
 COMPGCN [ 67 ] | 
 Transductive | 
 GNN | 
 | 
 2015 | 
 TransR [ 20 ] | 
 Transductive | 
 Translational | 

 
 2020 | 
 GraIL [ 71 ] | 
 Inductive | 
 GNN | 
 | 
 2015 | 
 KG2E [ 22 ] | 
 Transductive | 
 Translational | 

 
 2020 | 
 ExpressGNN [ 107 ] | 
 Inductive | 
 Logic Rule | 
 | 
 2015 | 
 DISTMULT [ 36 ] | 
 Transductive | 
 Tensor Decompositional | 

 
 2020 | 
 pGAT [ 82 ] | 
 Inductive | 
 GNN | 
 | 
 2015 | 
 RNNPRA [ 108 ] | 
 Transductive | 
 Relation Path | 

 
 2019 | 
 RotatE [ 28 ] | 
 Transductive | 
 Translational | 
 | 
 2014 | 
 TransH [ 19 ] | 
 Transductive | 
 Translational | 

 
 2019 | 
 TransW [ 27 ] | 
 Inductive | 
 Translational | 
 | 
 2014 | 
 ProPPR [ 109 ] | 
 Transductive | 
 Relation Path | 

 
 2019 | 
 MuRP [ 26 ] | 
 Transductive | 
 Translational | 
 | 
 2013 | 
 AMIE [ 110 ] | 
 Inductive | 
 Logic Rule | 

 
 2019 | 
 QuatE [ 43 ] | 
 Transductive | 
 Tensor Decompositional | 
 | 
 2013 | 
 SME [ 48 ] | 
 Transductive | 
 Traditional NN | 

 
 2019 | 
 TuckER [ 41 ] | 
 Transductive | 
 Tensor Decompositional | 
 | 
 2013 | 
 NTN [ 49 ] | 
 Transductive | 
 Traditional NN | 

 
 2019 | 
 CrossE [ 42 ] | 
 Transductive | 
 Tensor Decompositional | 
 | 
 2013 | 
 TransE [ 18 ] | 
 Transductive | 
 Translational | 

 
 2019 | 
 ConvR [ 55 ] | 
 Transductive | 
 CNN | 
 | 
 2011 | 
 RESCAL [ 35 ] | 
 Transductive | 
 Tensor Decompositional | 

 
 2019 | 
 HypER [ 54 ] | 
 Transductive | 
 CNN | 
 | 
 2010 | 
 PRA [ 111 ] | 
 Transductive | 
 Relation Path | 

 

 
 
 
 Table IV: Performance comparison of static KGR models on WN18RR, FB15k-237 in the transductive scenario. Best results are marked as boldfaced. ”H” is short for ”Hits”. 
 
 
 
 
 Methods | 
 WN18RR | 
 FB15K-237 | 

 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 

 
 
 
 TransE [ 18 ] | 
 0.231 | 
 0.021 | 
 0.409 | 
 0.533 | 
 0.289 | 
 0.193 | 
 0.326 | 
 0.478 | 

 
 RotatE [ 28 ] | 
 0.476 | 
 0.428 | 
 0.492 | 
 0.571 | 
 0.338 | 
 0.241 | 
 0.375 | 
 0.533 | 

 
 QuatE [ 43 ] | 
 0.481 | 
 0.436 | 
 0.500 | 
 0.564 | 
 0.311 | 
 0.221 | 
 0.342 | 
 0.495 | 

 
 InteractE [ 56 ] | 
 0.463 | 
 0.430 | 
 – | 
 0.528 | 
 0.354 | 
 0.263 | 
 – | 
 0.535 | 

 
 DualE [ 44 ] | 
 0.482 | 
 0.440 | 
 0.500 | 
 0.561 | 
 0.330 | 
 0.237 | 
 0.363 | 
 0.518 | 

 
 HAKE [ 29 ] | 
 0.497 | 
 0.453 | 
 0.515 | 
 0.582 | 
 0.335 | 
 0.237 | 
 0.371 | 
 0.530 | 

 
 MuRP [ 26 ] | 
 0.481 | 
 0.440 | 
 0.495 | 
 0.566 | 
 0.335 | 
 0.243 | 
 0.367 | 
 0.518 | 

 
 ConEx [ 57 ] | 
 0.481 | 
 0.448 | 
 0.493 | 
 0.550 | 
 0.366 | 
 0.271 | 
 0.403 | 
 0.555 | 

 
 HousE [ 32 ] | 
 0.511 | 
 0.465 | 
 0.528 | 
 0.602 | 
 0.361 | 
 0.266 | 
 0.399 | 
 0.551 | 

 
 RESCAL [ 35 ] | 
 0.455 | 
 0.419 | 
 0.461 | 
 0.493 | 
 0.353 | 
 0.264 | 
 0.385 | 
 0.528 | 

 
 DisMult [ 36 ] | 
 0.420 | 
 0.370 | 
 0.439 | 
 0.521 | 
 0.243 | 
 0.191 | 
 0.271 | 
 0.328 | 

 
 ComplEX [ 37 ] | 
 0.440 | 
 0.410 | 
 0.460 | 
 0.510 | 
 0.346 | 
 0.256 | 
 0.386 | 
 0.525 | 

 
 TuckER [ 41 ] | 
 0.470 | 
 0.443 | 
 0.482 | 
 0.526 | 
 0.358 | 
 0.266 | 
 0.394 | 
 0.544 | 

 
 ConvE [ 52 ] | 
 0.430 | 
 0.400 | 
 0.440 | 
 0.520 | 
 0.325 | 
 0.237 | 
 0.356 | 
 0.501 | 

 
 HypER [ 54 ] | 
 0.465 | 
 0.443 | 
 0.477 | 
 0.522 | 
 0.341 | 
 0.252 | 
 0.376 | 
 0.520 | 

 
 ConvKB [ 53 ] | 
 0.265 | 
 0.058 | 
 0.445 | 
 0.558 | 
 0.289 | 
 0.198 | 
 0.324 | 
 0.471 | 

 
 ConvR [ 55 ] | 
 0.475 | 
 0.443 | 
 0.489 | 
 0.537 | 
 0.350 | 
 0.261 | 
 0.385 | 
 0.528 | 

 
 ComplEX-DURA [ 37 ] | 
 0.489 | 
 0.445 | 
 0.503 | 
 0.574 | 
 0.370 | 
 0.275 | 
 0.409 | 
 0.562 | 

 
 LowFER [ 46 ] | 
 0.465 | 
 0.434 | 
 0.479 | 
 0.526 | 
 0.359 | 
 0.359 | 
 0.266 | 
 0.396 | 

 
 RGCN [ 58 ] | 
 0.427 | 
 0.382 | 
 0.446 | 
 0.510 | 
 0.248 | 
 0.153 | 
 0.258 | 
 0.414 | 

 
 SACN [ 62 ] | 
 0.470 | 
 0.430 | 
 0.480 | 
 0.540 | 
 0.350 | 
 0.260 | 
 0.390 | 
 0.540 | 

 
 KBGAT [ 60 ] | 
 0.464 | 
 0.426 | 
 0.479 | 
 0.539 | 
 0.350 | 
 0.260 | 
 0.385 | 
 0.531 | 

 
 COMPGCN [ 67 ] | 
 0.469 | 
 0.434 | 
 0.482 | 
 0.537 | 
 0.352 | 
 0.261 | 
 0.387 | 
 0.534 | 

 
 DPMPN [ 64 ] | 
 0.482 | 
 0.444 | 
 0.497 | 
 0.558 | 
 0.369 | 
 0.286 | 
 0.403 | 
 0.530 | 

 
 RGHAT [ 65 ] | 
 0.483 | 
 0.425 | 
 0.499 | 
 0.588 | 
 0.522 | 
 0.462 | 
 0.546 | 
 0.631 | 

 
 RED-GNN [ 81 ] | 
 0.533 | 
 0.485 | 
 – | 
 0.624 | 
 0.374 | 
 0.283 | 
 – | 
 0.558 | 

 
 NeuralLP [ 104 ] | 
 0.459 | 
 0.376 | 
 0.468 | 
 0.657 | 
 0.227 | 
 0.166 | 
 0.248 | 
 0.348 | 

 
 MINERVA [ 101 ] | 
 0.448 | 
 0.413 | 
 0.456 | 
 0.513 | 
 0.271 | 
 0.192 | 
 0.307 | 
 0.426 | 

 
 M-walk [ 96 ] | 
 0.437 | 
 0.415 | 
 0.447 | 
 0.543 | 
 0.234 | 
 0.168 | 
 0.245 | 
 0.403 | 

 
 pLogicNet [ 90 ] | 
 0.441 | 
 0.398 | 
 0.446 | 
 0.537 | 
 0.332 | 
 0.237 | 
 0.367 | 
 0.524 | 

 
 CURL [ 94 ] | 
 0.460 | 
 0.429 | 
 0.471 | 
 0.523 | 
 0.306 | 
 0.224 | 
 0.341 | 
 0.470 | 

 

 
 
 
 Table V: Hits@10 Performance comparison (in percentage) of static KGR models on WN18RR, FB15k-237 and NELL-995 in the inductive scenario. Best results are marked as boldfaced. 
 
 
 
 
 Methods | 
 WN18RR | 
 FB15K-237 | 
 NELL-995 | 

 
 v1 | 
 v2 | 
 v3 | 
 v4 | 
 v1 | 
 v2 | 
 v3 | 
 v4 | 
 v1 | 
 v2 | 
 v3 | 
 v4 | 

 
 
 
 Neural-LP [ 104 ] | 
 74.37 | 
 68.93 | 
 46.18 | 
 67.13 | 
 52.92 | 
 58.94 | 
 52.90 | 
 55.88 | 
 40.78 | 
 78.73 | 
 82.71 | 
 80.58 | 

 
 DRUM [ 91 ] | 
 74.37 | 
 68.93 | 
 46.18 | 
 67.13 | 
 52.92 | 
 58.73 | 
 52.90 | 
 55.88 | 
 19.42 | 
 78.55 | 
 82.71 | 
 80.58 | 

 
 RuleN [ 99 ] | 
 80.85 | 
 78.23 | 
 53.39 | 
 71.59 | 
 49.76 | 
 77.82 | 
 87.69 | 
 85.60 | 
 53.50 | 
 81.75 | 
 77.26 | 
 61.35 | 

 
 GraIL [ 71 ] | 
 82.45 | 
 78.68 | 
 58.43 | 
 73.41 | 
 64.15 | 
 81.80 | 
 82.83 | 
 89.29 | 
 59.50 | 
 93.25 | 
 91.41 | 
 73.19 | 

 
 TACT [ 72 ] | 
 83.24 | 
 81.63 | 
 62.73 | 
 76.27 | 
 65.61 | 
 83.05 | 
 87.28 | 
 91.33 | 
 55.00 | 
 88.97 | 
 91.35 | 
 74.69 | 

 
 CoMPILE [ 73 ] | 
 83.60 | 
 79.82 | 
 60.69 | 
 75.49 | 
 67.64 | 
 82.98 | 
 84.67 | 
 87.44 | 
 58.38 | 
 93.87 | 
 92.77 | 
 75.19 | 

 
 Meta-iKG [ 74 ] | 
 – | 
 – | 
 – | 
 – | 
 66.52 | 
 72.37 | 
 68.81 | 
 74.32 | 
 60.49 | 
 74.07 | 
 77.99 | 
 71.63 | 

 
 RPC-IR [ 76 ] | 
 85.11 | 
 81.63 | 
 62.40 | 
 76.35 | 
 67.56 | 
 82.53 | 
 84.39 | 
 89.22 | 
 59.75 | 
 93.28 | 
 94.01 | 
 71.82 | 

 
 SNRI [ 75 ] | 
 87.23 | 
 83.10 | 
 67.31 | 
 83.32 | 
 71.79 | 
 86.50 | 
 89.59 | 
 89.39 | 
 – | 
 – | 
 – | 
 – | 

 
 RMPI [ 77 ] | 
 82.45 | 
 78.68 | 
 58.68 | 
 73.41 | 
 65.37 | 
 81.80 | 
 81.10 | 
 87.25 | 
 59.50 | 
 92.23 | 
 93.57 | 
 87.62 | 

 
 REPORT [ 87 ] | 
 88.03 | 
 85.83 | 
 72.31 | 
 81.46 | 
 71.69 | 
 88.91 | 
 91.62 | 
 92.28 | 
 – | 
 – | 
 – | 
 – | 

 
 ConGLR [ 112 ] | 
 85.64 | 
 92.93 | 
 70.74 | 
 92.90 | 
 68.29 | 
 85.98 | 
 88.61 | 
 89.31 | 
 – | 
 – | 
 – | 
 – | 

 
 LogCo [ 86 ] | 
 90.16 | 
 84.69 | 
 68.68 | 
 79.08 | 
 73.90 | 
 81.91 | 
 80.64 | 
 84.20 | 
 61.75 | 
 93.48 | 
 94.19 | 
 80.82 | 

 

 
 
 
 
 

#### III-A 2 Path-based Model

 
 Path-based models mine the logical knowledge underlying the paths between the queried head and tail to achieve reasoning.

 
 
 Random walk [ 113 ] inferences have been widely investigated. For instance, the Path-Ranking Algorithm (PRA) [ 111 ] derives the path-based logic rules under path constraints. ProPPR [ 109 ] further introduces space similarity heuristics by incorporating textual content to alleviate the feature sparsity issue in PRA. Meanwhile, Neural multi-hop path-based models are also studied for better expressive ability. By iteratively using compositionality, RNNPRA [ 108 ] leverages RNN to compose the implications of relational paths for reasoning. LogSumExp [ 105 ] designs a logical composition method across all the elements with attention mechanisms for multiple reasoning. Then, a unified variational inference framework is proposed by DIVA [ 98 ] , which separates multi-hop reasoning into two steps, i.e., path-finding and path-reasoning.
Deep reinforcement learning (DRL) techniques, such as the Markov decision process (MDP), have recently been used to reformulate path-finding between entities as a sequential decision-making task. The designed reinforcement learning agent learns to find the reasoning paths according to entity interactions, and the corresponding policy gradient is utilized for training. Concretely, different fine-grained manners are employed by different models.
For example, DeepPath [ 102 ] applies DRL for relational path learning via the novel rewards and relational action spaces, improving both the models’ performance and efficiency. Meanwhile, MINERVA [ 101 ] takes path finding between entities as a sequential optimization problem by maximizing the expected reward [ 7 ] , which excludes the target answer entity for more capable reasoning. After that, MultiHop [ 97 ] designs a soft reward mechanism instead of only relying on binary rewards, as well as the dropout action, which enables more effective path exploration. Besides, Monte Carlo Tree Search (MCTS) is used by M-Walk [ 96 ] to generate the path, and CPL [ 88 ] proposes collaborative policy learning for path-finding and fact extraction by leveraging the text corpus corresponding to the entities. Moreover, two agents in different levels, i.e., DWARF AGENT at the entity level and GIANT AGENT at the cluster level are proposed in CURL [ 94 ] , which collaborate to achieve optimal reasoning performance.

 
 
 

#### III-A 3 Rule-based Model

 
 Rule-based models aim to make better use of such symbolic features underlying the logic rule, which is generally defined in the form of B → \rightarrow A, where A is a fact, and B can be a set of facts.

 
 
 Logical rules can be extracted from KG for reasoning by rule mining tools, e.g., AMIE [ 110 ] , RuleN [ 99 ] , etc. Then, a more scalable rule mining approach via the techniques of rule searching and pruning is designed by RLvLR [ 92 ] . After that,
how to inject logical rules into embeddings for better reasoning performance has drawn increasing research attention [ 7 ] . In general, there are two ways for it, i.e., joint learning and iterative training.
For instance, KALE [ 106 ] is a unified joint model by leveraging the t-norm fuzzy logical connectives between compatible facts and rule embedding.
Besides, RUGE [ 100 ] is an iterative model utilizing the soft rules for embedding rectification. Inspired by it, the iterative training strategy, composed of embedding learning, axiom induction, and axiom injection, is designed by IterE [ 89 ] . After that, researchers integrate neural network techniques into the rule-based models to alleviate the issues of limited expressive ability and huge space consumption of the previous rule-based models. Neural Theorem Provers (NTP) [ 103 ] mines logical rules with the designed radial kernel. Besides, NeuralLP [ 104 ] leverages attention mechanisms and auxiliary memory to optimize the gradients for mining the rules, and Neural-Num-LP [ 93 ] further integrates the cumulative sum operations and dynamic programming with NeuralLP to learn numerical rules. Meanwhile, an end-to-end differentiable rule-based model is proposed in DRUM [ 91 ] . Then, the probabilistic logic neural network is designed in pLogicNet [ 90 ] , which shows great performances for first-order logic mining. ExpressGNN [ 107 ] further generalizes it by finetuning GNN models for more efficient reasoning. Moreover, GCR [ 95 ] achieves promising performance for both reasoning and recommendation by mining the neighborhood information around the queried facts.

 
 
 
 

### III-B Review on Reasoning Scenarios 

 
 Based on the observation, there are 56 transductive models and 34 inductive models (See Table III ). Among them, 56.25% of the inductive models belong to GNN models, and 37.5% belongs to rule-based models. In particular, none of the path-based models has shown an incredible inductive ability for reasoning. Still, most of the rule-based models are good at inductive scenarios, which is reasonable. Since path-based models are developed based on searching for specific paths, the trained models in this manner are hardly applied when unseen elements occur. While most rule-based models can derive entity-agnostic logical rules from the KGs, and the invariance brought by the rules can be easily applied to inductive scenarios. Moreover, we present a fair performance comparison of the typical SOTA static KGR models for transductive and inductive scenarios separately (See Table IV and Table V ). The results support the analysis above. Additionally, embedding-based models, especially for the GNN models, have shown good capacity and compatibility for both scenarios and also most of the recent research lies in this type.

 
 
 

### III-C Observation and Discussion 

 
 Based on the above reviews, we can get the following observations, which may indicate the scope for different static KGR models and reveal the future trend in static KGR. (1) Embedding-based models generally have the better expressive ability but lack explainability. Meanwhile, more attention is currently focused on developing GNN-based models since the KGR models require a high-quality representation of the relational facts and the graph structure, which is most suitable for GNN models. (2) Path-based and Rule-based models are more explainable than embedding-based models, but they usually suffer from limited expressive ability and huge complexity of time and space. (3) Most of the path-based models are more suitable for transductive reasoning due to the path-searching schemes, while Rule-based models naturally inherit the inductive ability due to the generalization of the rule paradigms. (4) For a long time, transductive reasoning models have kept appearing and greatly impacted both academic research and industrial applications. However, due to the issues of scalability and expressive ability issues, researchers have recently focused on developing inductive reasoning models. As a brief conclusion, we would encourage to study more on GNN-based models, considered the most promising model type for static KGR.

 
 
 
 

## IV Temporal KGR Model 

 
 We systematically introduce 58 temporal KGR models according to techniques, i.e., how they integrate time information and scenarios (See Table VI ).

 
 

### IV-A Review on Reasoning Techniques 

 
 Temporal KGR models can be categorized into RNN-based, and RNN-agnostic models. Details are described below.

 
 
 Table VI: Summary of the temporal knowledge graph reasoning models. 
 
 
 
 
 Year | 
 Model | 
 Scenario | 
 Technique | 
 | 
 Year | 
 Model | 
 Scenario | 
 Technique | 

 
 
 
 2023 | 
 RETIA [ 114 ] | 
 Extrapolation | 
 GRU | 
 | 
 2021 | 
 xERTE [ 115 ] | 
 Extrapolation | 
 Time-Vector | 

 
 2023 | 
 RPC [ 116 ] | 
 Extrapolation | 
 GRU | 
 | 
 2021 | 
 CyGNet [ 117 ] | 
 Extrapolation | 
 Time-Vector | 

 
 2022 | 
 CENET [ 118 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2021 | 
 TIE [ 119 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 DA-Net [ 120 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2021 | 
 TeLM [ 121 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 HiSMatch [ 122 ] | 
 Extrapolation | 
 GRU | 
 | 
 2021 | 
 ChronoR [ 123 ] | 
 Interpolation | 
 Time-Vector | 

 
 2022 | 
 rGalT [ 124 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2021 | 
 RE-GCN [ 125 ] | 
 Extrapolation | 
 GRU | 

 
 2022 | 
 MetaTKGR [ 126 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2021 | 
 RTFE [ 127 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 FILT [ 128 ] | 
 Interpolation | 
 Time-Operation | 
 | 
 2021 | 
 HIP [ 129 ] | 
 Extrapolation | 
 GRU | 

 
 2022 | 
 TKGC-AGP [ 130 ] | 
 Interpolation | 
 Time-Operation | 
 | 
 2021 | 
 Tpath [ 131 ] | 
 Interpolation | 
 LSTM | 

 
 2022 | 
 Tlogic [ 132 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2020 | 
 TIMEPLEX [ 133 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 TLT-KGE [ 134 ] | 
 Interpolation | 
 Time-Vector | 
 | 
 2020 | 
 DyERNIE [ 135 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 CEN [ 136 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2020 | 
 DacKGR [ 137 ] | 
 Interpolation | 
 RNN | 

 
 2022 | 
 BoxTE [ 138 ] | 
 Interpolation | 
 Time-Vector | 
 | 
 2020 | 
 TNTComplEx [ 139 ] | 
 Interpolation | 
 Time-Vector | 

 
 2022 | 
 TempoQR [ 140 ] | 
 Interpolation | 
 Time-Vector | 
 | 
 2020 | 
 TComplEx [ 139 ] | 
 Interpolation | 
 Time-Vector | 

 
 2022 | 
 TuckERTNT [ 141 ] | 
 Interpolation | 
 Time-Vector | 
 | 
 2020 | 
 TDGNN [ 142 ] | 
 Extrapolation | 
 Time-Operation | 

 
 2022 | 
 GHT [ 143 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2020 | 
 ATiSE [ 144 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 DKGE [ 145 ] | 
 Interpolation | 
 Time-Operation | 
 | 
 2020 | 
 Diachronic [ 146 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 TiRGN [ 147 ] | 
 Extrapolation | 
 GRU | 
 | 
 2020 | 
 DE-Simple [ 146 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 RotateQVS [ 148 ] | 
 Interpolation | 
 Time-Vector | 
 | 
 2020 | 
 TeRo [ 149 ] | 
 Interpolation | 
 Time-Operation | 

 
 2022 | 
 ExKGR [ 150 ] | 
 Interpolation | 
 LSTM | 
 | 
 2020 | 
 EvolveGCN [ 151 ] | 
 Extrapolation | 
 LSTM+GRU | 

 
 2022 | 
 TRHyTE [ 152 ] | 
 Interpolation | 
 GRU | 
 | 
 2020 | 
 TeMP [ 153 ] | 
 Interpolation | 
 GRU | 

 
 2022 | 
 EvoKG [ 154 ] | 
 Extrapolation | 
 RNN | 
 | 
 2020 | 
 RE-NET [ 155 ] | 
 Extrapolation | 
 RNN | 

 
 2021 | 
 TPmod [ 156 ] | 
 Interpolation | 
 GRU | 
 | 
 2019 | 
 DyRep [ 157 ] | 
 Extrapolation | 
 Time-Operation | 

 
 2021 | 
 TimeTraveler [ 158 ] | 
 Extrapolation | 
 LSTM | 
 | 
 2018 | 
 TTransE [ 159 ] | 
 Interpolation | 
 LSTM | 

 
 2021 | 
 CluSTeR [ 160 ] | 
 Extrapolation | 
 LSTM+GRU | 
 | 
 2018 | 
 HyTE [ 161 ] | 
 Interpolation | 
 Time-Operation | 

 
 2021 | 
 TPRec [ 162 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2018 | 
 ChronoTranslate [ 163 ] | 
 Interpolation | 
 Time-Operation | 

 
 2021 | 
 DBKGE [ 164 ] | 
 Interpolation | 
 Time-Vector | 
 | 
 2018 | 
 TA-DISTMULT [ 165 ] | 
 Interpolation | 
 LSTM | 

 
 2021 | 
 TANGO [ 166 ] | 
 Extrapolation | 
 Time-Operation | 
 | 
 2018 | 
 TA-TransE [ 165 ] | 
 Interpolation | 
 LSTM | 

 
 2021 | 
 T-GAP [ 167 ] | 
 Interpolation | 
 Time-Vector | 
 | 
 2017 | 
 Know-Evolve [ 168 ] | 
 Extrapolation | 
 RNN | 

 

 
 
 
 Figure 9: Taxonomy of the temporal KGR models. 
 
 

#### IV-A 1 RNN-based Model

 
 Recurrent Neural Networks (RNNs) are suitable for mining the changes over time. Thereby many temporal KGR models use RNNs to directly model the temporal information, termed RNN-based models. According to different variants of RNN, the models can be divided into three types, i.e., basic RNN enhanced models, LSTM enhanced models, and GRU enhanced models.

 
 
 Basic RNN Enhanced Models 

 
 Some temporal KGR models can effectively model temporal information using basic RNN models. To name a few, Know-Evolve [ 168 ] is a classical temporal KGR model that generates non-linearly entity embeddings over time. RE-NET [ 155 ] applies the GCN and RNN models to capture the evolutional dynamics in temporal KGs to the query over time. EvoKG [ 154 ] introduces the RNN model to mine the dynamic evolving structural information and models entity interactions by combining the neighborhood information.

 
 
 Table VII: Performance comparison (in percentage) of temporal KGR models on ICEWS14, ICEWS05-15 for interpolation scenario. Best results are marked as boldfaced. ”H” is short for ”Hits”. 
 
 
 
 
 Model | 
 ICEWS05-15 | 
 ICEWS14 | 

 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 

 
 
 
 TA-TransE [ 165 ] | 
 29.90 | 
 9.60 | 
 – | 
 66.80 | 
 27.50 | 
 9.50 | 
 – | 
 62.50 | 

 
 TA-DISTMULT [ 165 ] | 
 47.40 | 
 34.60 | 
 – | 
 72.80 | 
 47.70 | 
 36.30 | 
 – | 
 68.60 | 

 
 HyTE [ 161 ] | 
 31.60 | 
 11.60 | 
 44.50 | 
 68.10 | 
 29.70 | 
 10.80 | 
 41.60 | 
 65.50 | 

 
 TTransE [ 159 ] | 
 27.10 | 
 8.40 | 
 – | 
 61.60 | 
 25.50 | 
 7.40 | 
 – | 
 60.10 | 

 
 RE-NET [ 155 ] | 
 43.70 | 
 33.60 | 
 48.80 | 
 62.72 | 
 39.86 | 
 30.11 | 
 44.02 | 
 58.21 | 

 
 TeMP [ 78 ] | 
 69.10 | 
 56.60 | 
 78.20 | 
 91.70 | 
 60.10 | 
 47.80 | 
 68.10 | 
 82.80 | 

 
 TeRo [ 149 ] | 
 58.60 | 
 46.90 | 
 66.80 | 
 79.50 | 
 56.20 | 
 46.80 | 
 62.10 | 
 73.20 | 

 
 DE-SimplE [ 146 ] | 
 51.30 | 
 39.20 | 
 57.80 | 
 74.80 | 
 52.60 | 
 41.80 | 
 59.20 | 
 72.50 | 

 
 Diachronic [ 146 ] | 
 51.30 | 
 39.20 | 
 57.80 | 
 74.80 | 
 52.60 | 
 41.80 | 
 59.20 | 
 72.50 | 

 
 ATiSE [ 144 ] | 
 51.90 | 
 37.80 | 
 60.60 | 
 79.40 | 
 55.00 | 
 43.60 | 
 62.90 | 
 75.00 | 

 
 TComplEx [ 139 ] | 
 66.40 | 
 58.30 | 
 71.60 | 
 81.10 | 
 61.90 | 
 54.20 | 
 66.10 | 
 76.70 | 

 
 TNTComplEx [ 139 ] | 
 67.00 | 
 59.00 | 
 71.00 | 
 81.00 | 
 62.00 | 
 52.00 | 
 66.00 | 
 76.00 | 

 
 DyERNIE [ 135 ] | 
 73.90 | 
 67.90 | 
 77.30 | 
 85.50 | 
 66.90 | 
 59.90 | 
 71.40 | 
 79.70 | 

 
 T-GAP [ 167 ] | 
 67.00 | 
 56.80 | 
 74.30 | 
 84.50 | 
 61.00 | 
 50.90 | 
 67.70 | 
 79.00 | 

 
 TIMEPLEX [ 133 ] | 
 64.00 | 
 54.50 | 
 – | 
 81.80 | 
 60.40 | 
 51.50 | 
 – | 
 77.10 | 

 
 RTFE [ 127 ] | 
 64.50 | 
 55.30 | 
 70.60 | 
 81.10 | 
 59.20 | 
 50.30 | 
 64.60 | 
 75.80 | 

 
 ChronoR [ 123 ] | 
 68.41 | 
 61.06 | 
 73.01 | 
 82.13 | 
 62.53 | 
 54.67 | 
 66.88 | 
 77.31 | 

 
 TeLM [ 121 ] | 
 67.80 | 
 59.90 | 
 72.80 | 
 82.30 | 
 62.50 | 
 54.50 | 
 67.30 | 
 77.40 | 

 
 RotateQVS [ 148 ] | 
 63.30 | 
 52.90 | 
 70.90 | 
 81.30 | 
 59.10 | 
 50.70 | 
 64.20 | 
 75.40 | 

 
 TuckERTNT [ 141 ] | 
 67.50 | 
 59.30 | 
 72.50 | 
 81.90 | 
 – | 
 – | 
 – | 
 – | 

 
 TLT-KGE [ 134 ] | 
 60.90 | 
 60.90 | 
 74.10 | 
 83.50 | 
 63.40 | 
 55.10 | 
 68.40 | 
 78.60 | 

 
 BoxTE [ 138 ] | 
 66.70 | 
 58.20 | 
 71.90 | 
 82.00 | 
 61.30 | 
 52.80 | 
 66.40 | 
 76.30 | 

 
 TKGC-AGP [ 130 ] | 
 53.20 | 
 39.80 | 
 62.10 | 
 79.70 | 
 56.10 | 
 45.80 | 
 63.10 | 
 73.80 | 

 

 
 
 
 Table VIII: Performance comparison (in percentage) of temporal KGR models on GDELT, ICEWS14, ICEWS05-15, ICEWS18, WIKI, and YAGO for extrapolation scenario. Best results are marked as boldfaced. ”H” is short for ”Hits”. 
 
 
 
 
 | 
 GDELT | 
 ICEWS14 | 
 ICEWS05-15 | 
 ICEWS18 | 
 WIKI | 
 YAGO | 

 
 Model | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 
 MRR | 
 H@1 | 
 H@3 | 
 H@10 | 

 
 RGCRN [ 169 ] | 
 19.37 | 
 12.24 | 
 20.57 | 
 33.32 | 
 38.48 | 
 28.52 | 
 42.85 | 
 58.10 | 
 44.56 | 
 34.16 | 
 50.06 | 
 64.51 | 
 28.02 | 
 18.62 | 
 31.59 | 
 46.44 | 
 65.79 | 
 61.66 | 
 68.17 | 
 72.99 | 
 65.76 | 
 62.25 | 
 67.56 | 
 71.69 | 

 
 RE-NET [ 155 ] | 
 19.55 | 
 12.38 | 
 20.80 | 
 34.00 | 
 39.86 | 
 30.11 | 
 44.02 | 
 58.21 | 
 43.67 | 
 33.55 | 
 48.83 | 
 62.72 | 
 29.78 | 
 19.73 | 
 32.55 | 
 48.46 | 
 58.32 | 
 50.01 | 
 61.23 | 
 73.57 | 
 66.93 | 
 58.59 | 
 71.48 | 
 86.84 | 

 
 CyGNet [ 117 ] | 
 20.22 | 
 12.35 | 
 21.66 | 
 35.82 | 
 37.65 | 
 27.43 | 
 42.63 | 
 57.90 | 
 40.42 | 
 29.44 | 
 46.06 | 
 61.60 | 
 27.12 | 
 17.21 | 
 30.97 | 
 46.85 | 
 58.78 | 
 47.89 | 
 66.44 | 
 78.70 | 
 68.98 | 
 58.97 | 
 76.80 | 
 86.98 | 

 
 TANGO [ 166 ] | 
 19.66 | 
 12.50 | 
 20.93 | 
 33.55 | 
 – | 
 – | 
 – | 
 – | 
 42.86 | 
 32.72 | 
 48.14 | 
 62.34 | 
 28.97 | 
 19.51 | 
 32.61 | 
 47.51 | 
 53.04 | 
 51.52 | 
 53.84 | 
 55.46 | 
 63.34 | 
 60.04 | 
 65.19 | 
 68.79 | 

 
 xERTE [ 115 ] | 
 19.45 | 
 11.92 | 
 20.84 | 
 34.18 | 
 40.79 | 
 32.70 | 
 45.67 | 
 57.30 | 
 46.62 | 
 37.84 | 
 52.31 | 
 63.92 | 
 29.31 | 
 21.03 | 
 33.51 | 
 46.48 | 
 73.60 | 
 69.05 | 
 78.03 | 
 79.73 | 
 84.19 | 
 80.09 | 
 88.02 | 
 89.78 | 

 
 RE-GCN [ 125 ] | 
 19.69 | 
 12.46 | 
 20.93 | 
 33.81 | 
 42.00 | 
 31.63 | 
 47.20 | 
 61.65 | 
 48.03 | 
 37.33 | 
 53.90 | 
 68.51 | 
 32.62 | 
 22.39 | 
 36.79 | 
 52.68 | 
 78.53 | 
 74.50 | 
 81.59 | 
 84.70 | 
 82.30 | 
 78.83 | 
 84.27 | 
 88.58 | 

 
 TLogic [ 132 ] | 
 – | 
 – | 
 – | 
 – | 
 41.80 | 
 31.93 | 
 47.23 | 
 60.53 | 
 45.99 | 
 34.49 | 
 52.89 | 
 67.39 | 
 28.41 | 
 18.74 | 
 32.71 | 
 47.97 | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 CEN [ 136 ] | 
 – | 
 – | 
 – | 
 – | 
 41.64 | 
 31.22 | 
 46.55 | 
 61.59 | 
 49.57 | 
 37.86 | 
 56.42 | 
 71.32 | 
 29.70 | 
 19.38 | 
 33.91 | 
 49.90 | 
 63.39 | 
 – | 
 71.68 | 
 83.16 | 
 51.98 | 
 – | 
 58.96 | 
 70.61 | 

 
 TITer [ 158 ] | 
 18.19 | 
 11.52 | 
 19.20 | 
 31.00 | 
 41.73 | 
 32.74 | 
 46.46 | 
 58.44 | 
 47.60 | 
 38.29 | 
 52.74 | 
 64.86 | 
 29.98 | 
 22.05 | 
 33.46 | 
 44.83 | 
 73.91 | 
 71.70 | 
 75.41 | 
 76.96 | 
 87.47 | 
 80.09 | 
 89.96 | 
 90.27 | 

 
 HisMatch [ 122 ] | 
 22.01 | 
 14.45 | 
 23.80 | 
 36.61 | 
 46.42 | 
 35.91 | 
 51.63 | 
 66.84 | 
 52.85 | 
 42.01 | 
 59.05 | 
 73.28 | 
 33.99 | 
 23.91 | 
 37.90 | 
 53.94 | 
 78.07 | 
 73.89 | 
 81.32 | 
 84.65 | 
 – | 
 – | 
 – | 
 – | 

 
 EvoKG [ 154 ] | 
 19.28 | 
 – | 
 20.55 | 
 34.44 | 
 27.18 | 
 – | 
 30.84 | 
 47.67 | 
 – | 
 – | 
 – | 
 – | 
 29.28 | 
 – | 
 33.94 | 
 50.09 | 
 68.03 | 
 – | 
 79.60 | 
 85.91 | 
 68.59 | 
 - | 
 81.13 | 
 92.73 | 

 
 TiRGN [ 147 ] | 
 21.67 | 
 13.63 | 
 23.27 | 
 37.60 | 
 43.81 | 
 33.49 | 
 48.90 | 
 63.50 | 
 49.84 | 
 39.07 | 
 55.75 | 
 70.11 | 
 33.58 | 
 23.10 | 
 37.90 | 
 54.20 | 
 80.05 | 
 75.15 | 
 84.35 | 
 87.56 | 
 87.95 | 
 84.34 | 
 91.37 | 
 92.92 | 

 
 RPC [ 116 ] | 
 22.41 | 
 14.42 | 
 24.36 | 
 38.33 | 
 44.55 | 
 34.87 | 
 49.90 | 
 65.08 | 
 51.14 | 
 39.47 | 
 57.11 | 
 71.75 | 
 34.91 | 
 24.34 | 
 38.74 | 
 55.89 | 
 81.18 | 
 76.28 | 
 85.43 | 
 88.71 | 
 88.87 | 
 85.10 | 
 92.57 | 
 94.04 | 

 
 CluSTeR [ 160 ] | 
 – | 
 – | 
 – | 
 – | 
 46.00 | 
 33.80 | 
 - | 
 71.20 | 
 44.60 | 
 34.90 | 
 – | 
 63.00 | 
 32.30 | 
 20.60 | 
 – | 
 55.90 | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 RETIA [ 114 ] | 
 – | 
 – | 
 – | 
 – | 
 45.29 | 
 34.60 | 
 50.88 | 
 66.06 | 
 52.17 | 
 40.21 | 
 59.42 | 
 73.98 | 
 34.16 | 
 22.97 | 
 39.27 | 
 55.96 | 
 67.58 | 
 – | 
 78.42 | 
 88.06 | 
 70.11 | 
 – | 
 78.30 | 
 84.77 | 

 

 
 
 
 
 LSTM Enhanced Models 

 
 Long Short-Term Memory (LSTM) network is also widely used to mine temporal features in temporal KGR models. For instance, TTransE [ 159 ] extends TransE by adding the temporal constraints and encodes time information as translations similar to relationships with an RNN so that these translations move the header representation in the embedded space. TA-TransE and TA-DistMult [ 165 ] are also two respectively extended versions of TransE and DistMult that incorporate the temporal embeddings. Furthermore, EvolveGCN [ 151 ] adopts the graph convolutional networks (GCNs) to model the graph structure in each static snapshot and utilizes the LSTM model (also can utilize GRU model) to evolve the GCN parameters over time. CluSTeR [ 160 ] adopts reinforcement learning to discover evolutional patterns with both LSTM and GRU models in Temporal KGs over time. DacKGR [ 137 ] performs multi-hop path-based reasoning on sparse temporal KGs by using time information for dynamic prediction. To capture the timespan information and guide the model learning, TimeTraveler [ 158 ] proposes a novel relative time encoding module and a time-shaped reward module based on Dirichlet distribution. TPath [ 131 ] also introduces LSTM model to mine the current environment information and then generates relation embeddings and temporal embeddings to the environment through activation functions. ExKGR [ 150 ] introduces LSTM for reasoning in temporal KGs and provides the reasoning paths.

 
 
 
 GRU Enhanced Models 

 
 GRU-based models have got a lot of attention these years. More recently, TeMP [ 153 ] is proposed, which leverages message-passing graph neural networks (MPNNs) to learn structure-based entity representations at each timestamp, and then combines representations from all timestamps using an encoder. RE-GCN [ 125 ] focuses on the evolutional dynamics in temporal KGs and generates entity embeddings by modeling the KG sequence of a fixed length at the latest a few timestamps. TPmod [ 156 ] aggregates the attributes of entities and relations and learns dynamic weights to different events. HIP network [ 129 ] passes information from temporal, structural, and repetitive perspectives, which are used to mine the graph’s dynamic evolution, the interactions of events at the same time step, and the known events respectively. TRHyTE [ 152 ] uses GRU first to transform entities into latent space and then encode facts into temporal-relational hyper-planes for time relation-aware representation generation. TiRGN [ 147 ] uses two encoders to mine the information at both local and global levels. HiSMatch [ 122 ] proposes different encoders to mine the semantic information of the historical query structures and candidate entities, respectively.

 
 
 
 

#### IV-A 2 RNN-agnostic Model

 
 RNN-agnostic models extend the original static KGR models by incorporating temporal information without using RNN frameworks. According to how the time information guides the models, they can be roughly divided into two types, i.e., time-vector guided and time-operation guided models.

 
 
 Time-Vector Guided Model 

 
 Time-vector guided models directly generate the additional temporal embedding t as this additional temporal information and fuse them with fact embeddings.

 
 
 TComplEx and TNTComplEx [ 139 ] both come from ComplEx, where the fourth-order tensor space with additional consideration of time information is modeled by them. In the process of constructing subgraphs, the time embedding is used to calculate weighted probabilities in xERTE [ 115 ] . Then, T-GAP [ 167 ] encodes the query-specific structure patterns of Temporal KG and performs path-based reasoning based on it. CyGNet [ 117 ] attempts to solve the entity prediction task by encoding the historical facts related to the subject entity in each query and the time-indexing vector is generated. ChronoR [ 123 ] builds on the basis of RotatE, which connects relation and time embeddings to obtain the overall rotation embedding applied to the final entity embedding. Furthermore, DBKGE [ 164 ] proposed an online inference algorithm that smoothed the representation vector of nodes over time. BoxTE [ 138 ] introduces a novel box representation method for temporal KGR based on the static KGR method BoxE. TuckERTNT [ 141 ] proposes a novel tensor decomposition model for Temporal KGs inspired by the Tucker decomposition of a 4-order tensor with the extra time embedding. TempoQR [ 140 ] generates question-specific time vectors and exploits these vectors to aggregate specific entities and their timestamps. TLT-KGE [ 134 ] captures semantic and time information as different axes of complex space. RotateQVS [ 148 ] aims to consider the time information changes with rotation operation in the latent space.

 
 
 
 Time-Operation Guided Model 

 
 Time-operation guided models leverage some specific operations, such as encoding facts into designed time-specific hyper-planes and generating time-related rewards, to use the temporal information instead of directly fusion based on generating the temporal embeddings t .

 
 
 ChronoTranslate [ 163 ] learns a universal representation of entities and time-specific representations of the Temporal KGs, respectively. HyTE [ 161 ] represents each timestamp as a learnable hyper-plane in the embedding space, then projects entity and relation embeddings into the hyper-plane and utilizes the TransE scoring function on the projections. As a model for graph learning and KGR, DyRep [ 157 ] captures the interleaved dynamics within history, which is further parameterized by a temporal-attentive representation network. Inspired by RotatE, TeRo [ 149 ] introduces a novel temporal guided rotation operation between head and tail entities to evaluate the given fact’s semantic scores. Diachronic embeddings [ 146 ] map entity and relation embeddings, paired with temporal information, into a KGR model space, thus defining a framework yielding specific models such as DE-TransE and DE-SimplE. Due to the uncertainty of temporal information in the graph evolution over time, ATiSE [ 144 ] maps the entity and relation embeddings of Temporal KGs into the Gaussian spaces according to the time stamps. TDGNN [ 142 ] introduces a novel temporal aggregator to combine the neighborhood features and the temporal information from edges to calculate the final representations. To mine the dynamic graph evolution of temporal KGs, DyERNIE [ 135 ] defines the velocity vector in the tangent space over time and encourages entity embeddings to evolve according to it. TPRec [ 162 ] is an interest recommendation method that presents an efficient time-aware interaction relation extraction component to construct a collaborative KG with time-aware interactions and also utilizes a time-aware path module for reasoning. TeLM [ 121 ] leverages a linear temporal regularizer and multi-vector encoders to realize the 4th-order tensor factorization for reasoning. RTFE [ 127 ] treats the Temporal KGs as a Markov chain, which transitions from the previous state to the next state, and then recursively tracks the state transition of Temporal KG by passing updated parameters/features between timestamps. Besides, TIE [ 119 ] combines the experience replay and time regularization into KGR to learn the time-aware incremental embedding. A length-aware CNN is leveraged in CEN [ 136 ] to handle historical facts via an easy-to-difficult curriculum learning strategy over time. DKGE [ 145 ] introduces two different representations for each entity and each relationship (including temporal information). TLogic [ 132 ] performs the temporal random walks and extracts the temporal logical rules based on them, leading to better explainability. TKGC-AGP [ 130 ] leverages the approximations of multivariate Gaussian processes (MGPs) for fact encoding. Besides, FILT [ 128 ] makes use of the meta-learning framework for inferring the facts with unseen entities in temporal KGR. After that, rGalT [ 124 ] first design the attention mechanism in both intra-graph and inter-graph levels to leverage the historical semantics. Similarly to it, DA-Net [ 120 ] also tries to learn attention weights on repetitive facts at different historical timestamps. MetaTKGR [ 126 ] dynamically adjusts the strategies of sampling and aggregating neighbors from recent facts for new entities through temporally supervised signals on future facts as instant feedback. Moreover, CENET [ 118 ] learns both the historical and non-historical dependency for inferring the most potential facts.

 
 
 
 
 

### IV-B Review on Reasoning Scenarios 

 
 Based on the observation, there are 34 interpolation models and 24 extrapolation models (See Table VI ). Among them, 42.57% of the extrapolation models belong to time-operation models, and 47.62% belongs to RNN-based models. In particular, we observe that among the RNN-based models, the ratio of interpolation models and extrapolation models is really close (10:8). Meanwhile, similar observations can also be found among Time-Operation models (12:9). It reveals that RNN-based models and time-Operation models have shown good compatibility with both scenarios. Also, most of the recent research lies in this type. Compared to them, time-vector models can only perform better in interpolation scenarios since they cannot sufficiently the temporal information in such a primitive manner. Moreover, we present a fair performance comparison of the typical SOTA temporal KGR models (See Table VII and Table VIII ). Similar conclusions can be drawn based on the results, which support the analysis above.

 
 
 

### IV-C Observation and Discussion 

 
 Based on the above reviews, we can further get the following observations, which may indicate the scope for different temporal KGR models and reveal the future trend in temporal KGR. (1) RNN-agnostic models, i.e., time-vector guided models and time-operation guided models, generally treat the temporal information as additional attributes and integrate them into the previous static KGR models with different techniques. Such a manner is more flexible compared to RNN-based models. (2) Time-vector guided models encode the time information as an additional time vector t . Although these models are simple, their performances mostly rely on whether the time encoder and embedding fusion module are suitable. Unlike these models, time-operation guided models design specific time operations, which are task-specific. (3) RNN-based models can generally model the time information better than other models and can be more easily adopted to extrapolation scenarios. (4) The extrapolation reasoning is still at an early stage, occupying only around 30 % 30\% of the temporal KGR models, which leaves space for further exploration. As a brief conclusion, we would encourage to study more on RNN-based models, which are considered as the most promising model type for temporal KGR.

 
 
 
 

## V Multi-Modal KGR Model 

 
 We systematically introduce 32 multi-modal KGR models according to techniques (See Figure 11 ).

 
 
 Figure 10: Taxonomy of the multi-modal KGR models. 
 
 

### V-A Review on Reasoning Techniques 

 
 Directly applying static KGR models to multi-modal scenarios generally results in sub-optimal performance because of the lack of fusion modules for extra multi-modal information, e.g., texts, images, etc. Based on techniques to fuse such multi-modal information, we roughly divided the multi-modal KGR models into two types, i.e., transformer-based and transformer-agnostic models.

 
 

#### V-A 1 Transformer-based Models

 
 Transformer-based models are usually adopted as the unified paradigm for multi-modal problems due to their promising capacity when scaling to different modalities.

 
 
 Although some general multi-modal pretrained transformer models, such as VisualBERT [ 170 ] , ViLBERT [ 171 ] , can also be adopted for multi-modal KGR. Due to the variance between the multi-modal KGs and other multi-modal data, directly applying the above general MPT models to multi-modal knowledge graph reasoning (MKGR) may not lead to good reasoning performance. Inspired by it, researchers have attempted to develop transformer-based multi-modal KGR models these two years. VBKGC [ 172 ] leverages the pretrained transformer to encode multi-modal features and designs a multi-modal scoring function for optimization. Then, Knowledge-CLIP [ 173 ] leverages the CLIP [ 174 ] model for a better pre-trained model considering the semantic connections between multi-modal concepts. Meanwhile, MarT [ 175 ] first proposes a model-agnostic reasoning framework with a transformer for analogical reasoning. Additionally, a hybrid transformer with multi-level fusion is designed in MKGformer [ 176 ] , which unifies learning paradigms for different downstream tasks in a uniform framework. Later on, HRGAT [ 177 ] constructs a hyper-node graph to aggregate the multi-modal features generated by transformers. Besides, MSNEA [ 178 ] and IMF [ 179 ] makes use of contrastive learning for multi-modal alignment. Moreover, MuKEA [ 180 ] integrates KGR in vision understanding and reasoning, and DRAGON [ 181 ] also provides a method for pre-training in a self-supervised manner for text and KG. However, transformer-based multi-modal KGR is still at an early stage.

 
 
 Figure 11: Timeline of the multi-modal KGR models. 
 
 
 

#### V-A 2 Transformer-agnostic Models

 
 Most multi-modal KGR models generate and fuse features without using transformer frameworks but design different mechanisms to encode the extra modal information by extending the original unimodal KGR models, such as TransE [ 18 ] . These models are named transformer-agnostic models.

 
 
 CKE [ 182 ] is the first model to perform reasoning and collaborative filtering jointly, which enables it to generate representations and capture the implicit rules in KGs simultaneously. Then, DKRL [ 183 ] takes advantage of entity descriptions in KGs with language neural networks and gets more expressive semantics for reasoning. Inspired by it, IKRL [ 184 ] first designs an attention-based neural network to consider visual information in entity images. Such attention mechanism is also leveraged by TransAE [ 185 ] . KBLRN [ 186 ] first proposes an end-to-end reasoning framework, which combines neural network techniques with expert models for latent, relational, and numerical features. Afterward, KR-AMD [ 187 ] and MKRL [ 188 ] leverage textual data as part of auxiliary data to improve reasoning performance. Besides, inspired by the translation-based static KGR models, MTRL [ 189 ] is a translation-based model with three energy functions corresponding to visual, linguistic, and structural information. Moreover, MKBE [ 190 ] and MRCGN [ 191 ] integrate different neural encoders and decoders with relational models for embedding learning and multi-modal data for reasoning. MMKGR [ 192 ] first investigates the problem of how to effectively leverage multi-modal auxiliary features to conduct multi-hop reasoning in the KG area with a unified gate-attention network. MKGAT [ 193 ] better enhances recommendation systems with a multi-modal graph attention technique to conduct information propagation over multi-modal KGs. KB-VQA [ 194 ] and VSUA [ 195 ] performs reasoning on the image and external knowledge, which provides an intuitive way to explain the generated answers. Similar to it, KVQA [ 196 ] integrates commonsense knowledge with images for reasoning. MMEA [ 197 ] designs a joint loss for multi-modal alignment. Moreover, MoSE [ 198 ] exploits three ensemble inference techniques to combine the modality-split predictions by assessing modality importance. Recently, RSME [ 199 ] designed a forget gate with an MRP metric to select valuable images for multi-modal KGR, which tries to avoid the influence caused by the noise from irrelevant images corresponding to entities. HMEA [ 200 ] projects multi-modal features into a hyperbolic space via GCN. While, OTKGE [ 201 ] models the multi-modal fusion procedure as a transport plan moving different modal embeddings to a unified space by minimizing the Wasserstein distance. Besides, MM-RNS [ 202 ] and CKGC [ 203 ] leverage contrastive learning strategies. MMKRL [ 204 ] leverages reinforcement learning for multi-modal KGR.

 
 
 Table IX: Performance comparison (in percentage) of multi-modal KGR models on FB15K-237-IMG and WN18-IMG. The best results are in boldface. ”H” is short for ”Hits”. 
 
 
 
 
 Model | 
 FB15k-237-IMG | 
 WN18-IMG | 

 
 MR | 
 H@1 | 
 H@10 | 
 MR | 
 H@1 | 
 H@10 | 

 
 
 
 IKRL [ 184 ] | 
 298 | 
 19.4 | 
 45.8 | 
 596 | 
 12.7 | 
 92.8 | 

 
 TransAE [ 185 ] | 
 431 | 
 19.9 | 
 46.3 | 
 352 | 
 32.3 | 
 93.4 | 

 
 MTRL [ 189 ] | 
 187 | 
 22.9 | 
 49.4 | 
 – | 
 – | 
 – | 

 
 MKBE [ 190 ] | 
 158 | 
 25.8 | 
 53.2 | 
 – | 
 – | 
 – | 

 
 RSME [ 199 ] | 
 417 | 
 24.2 | 
 46.7 | 
 223 | 
 94.3 | 
 95.7 | 

 
 MoSE [ 198 ] | 
 117 | 
 28.1 | 
 56.5 | 
 7 | 
 94.8 | 
 97.4 | 

 
 KBLRN [ 186 ] | 
 209 | 
 21.9 | 
 49.3 | 
 – | 
 – | 
 – | 

 
 VisualBERT [ 170 ] | 
 592 | 
 21.7 | 
 43.9 | 
 122 | 
 17.9 | 
 65.4 | 

 
 ViLBERT [ 171 ] | 
 483 | 
 23.3 | 
 45.7 | 
 131 | 
 22.3 | 
 76.1 | 

 
 VBKGC [ 172 ] | 
 - | 
 21.3 | 
 47.8 | 
 – | 
 – | 
 – | 

 
 HRGAT [ 177 ] | 
 156 | 
 27.1 | 
 54.2 | 
 – | 
 – | 
 – | 

 
 MKGformer [ 176 ] | 
 252 | 
 24.3 | 
 49.9 | 
 25 | 
 93.5 | 
 97.0 | 

 
 IMF [ 179 ] | 
 134 | 
 28.7 | 
 59.3 | 
 – | 
 – | 
 – | 

 

 
 
 
 Figure 12: Statistic comparison of models over various KG types. 
 
 
 
 

### V-B Observation and Discussion 

 
 Based on the above reviews and performance comparison (See Table IX ), we can further get the following observations. (1) Initially, most multi-modal KGR models are developed based on embedding-based static KGR models instead of path-based or rule-based ones. It is mainly because most existing multi-modal KGR models leverage the extra multi-modal information via feature fusion in latent space. Specifically, different encoders are designed for features in different modalities. (2) Recently, researchers have tended to study uniform learning frameworks, such as pre-trained transformer-based models, for multi-modal features, especially after Large Language Models (LLM) flourished. These models meet the requirements of artificial general intelligence, which are more practical and scalable in this era. (3) Compared to the other two types of KGR, the research on multi-modal KGR is still at an early stage, i.e., only 18% models are for multi-modal scenarios. In conclusion, there are lots of spaces for deep exploration, further described in Sec. 7.4 as a challenge and opportunity.

 
 
 
 

## VI Datasets 

 
 We comprehensively summarize typical KGR datasets, especially for temporal and multi-modal KGs, and provide their description and statistic as follows. Besides, the datasets are collected in our GitHub repository to better convenience the community.

 
 

### VI-A Static KGR Datasets 

 
 Typical static KGR datasets, i.e., 38 transductive datasets, and 15 inductive datasets are summarized. The statistics are presented in Table X and Table XI , and the descriptions are listed below.

 
 
 Table X: Typical benchmark datasets for static transductive knowledge graph reasoning. 
 
 
 
 
 Dataset | 
 # Ent. | 
 # Rel. | 
 # Train Facts | 
 # Val. Facts | 
 # Test Facts | 

 
 ATOMIC [ 205 ] | 
 304,388 | 
 9 | 
 610,536 | 
 87,700 | 
 87,701 | 

 
 Countries [ 206 ] | 
 271 | 
 2 | 
 1,110 | 
 24 | 
 24 | 

 
 CoDEX-S [ 207 ] | 
 2,034 | 
 42 | 
 32,888 | 
 3,654 | 
 3656 | 

 
 CoDEX-M [ 207 ] | 
 17,050 | 
 51 | 
 185,584 | 
 20620 | 
 20622 | 

 
 CoDEX-L [ 207 ] | 
 77,951 | 
 69 | 
 551,193 | 
 30,622 | 
 30622 | 

 
 ConceptNet [ 208 ] | 
 28,370,083 | 
 50 | 
 27,259,933 | 
 3,407,492 | 
 3,407,492 | 

 
 ConceptNet100K [ 209 ] | 
 78,334 | 
 34 | 
 100,000 | 
 1,200 | 
 1,200 | 

 
 DBpedia50 [ 210 ] | 
 49,900 | 
 654 | 
 32,388 | 
 399 | 
 10,969 | 

 
 DBpedia500 [ 210 ] | 
 517,475 | 
 654 | 
 3,102,677 | 
 10,000 | 
 1,155,937 | 

 
 DB100K [ 211 ] | 
 99,604 | 
 470 | 
 597,482 | 
 49,997 | 
 50,000 | 

 
 FAMILY [ 212 ] | 
 3,007 | 
 12 | 
 23,483 | 
 2,038 | 
 2,835 | 

 
 FB13 [ 213 ] | 
 75,043 | 
 13 | 
 316,232 | 
 11,816 | 
 47,464 | 

 
 FB122 [ 214 ] | 
 9,738 | 
 122 | 
 91,638 | 
 9,595 | 
 11,243 | 

 
 FB15k [ 215 ] | 
 14,951 | 
 1,345 | 
 483,142 | 
 50,000 | 
 59,071 | 

 
 FB20k [ 210 ] | 
 19,923 | 
 1,345 | 
 472,860 | 
 48,991 | 
 90,149 | 

 
 FB24k [ 216 ] | 
 23,634 | 
 673 | 
 402,493 | 
 - | 
 21,067 | 

 
 FB5M [ 19 ] | 
 5,385,322 | 
 1,192 | 
 19,193,556 | 
 50,000 | 
 59,071 | 

 
 FB15k-237 [ 217 ] | 
 14,505 | 
 237 | 
 272,115 | 
 17,535 | 
 20,466 | 

 
 FB60k-NYT10 [ 218 ] | 
 69,514 | 
 1,327 | 
 268,280 | 
 8,765 | 
 8,918 | 

 
 Hetionet [ 219 ] | 
 45,158 | 
 24 | 
 1,800,157 | 
 225,020 | 
 225,020 | 

 
 Kinship [ 212 ] | 
 104 | 
 25 | 
 8,544 | 
 1,068 | 
 1,074 | 

 
 Location [ 220 ] | 
 445 | 
 5 | 
 384 | 
 65 | 
 65 | 

 
 Nation [ 221 ] | 
 14 | 
 55 | 
 1,592 | 
 199 | 
 201 | 

 
 NELL23k [ 222 ] | 
 22,925 | 
 200 | 
 25,445 | 
 4,961 | 
 4,952 | 

 
 NELL-995 [ 223 ] | 
 75,492 | 
 200 | 
 126,176 | 
 5,000 | 
 5,000 | 

 
 OpenBioLink [ 224 ] | 
 180,992 | 
 28 | 
 4,192,002 | 
 188,394 | 
 183,011 | 

 
 Sport [ 220 ] | 
 1,039 | 
 4 | 
 1,349 | 
 358 | 
 358 | 

 
 Toy [ 225 ] | 
 280 | 
 112 | 
 4,565 | 
 109 | 
 152 | 

 
 UMLS [ 226 ] | 
 135 | 
 46 | 
 5,216 | 
 652 | 
 661 | 

 
 UMLS-PubMed [ 218 ] | 
 59,226 | 
 443 | 
 2,030,841 | 
 8,756 | 
 8,689 | 

 
 WD-singer [ 222 ] | 
 10,282 | 
 135 | 
 16,142 | 
 2,163 | 
 2,203 | 

 
 WN11 [ 213 ] | 
 38,588 | 
 11 | 
 110,361 | 
 5,212 | 
 21,035 | 

 
 WN18 [ 227 ] | 
 40,943 | 
 18 | 
 141,442 | 
 5,000 | 
 5,000 | 

 
 WN18RR [ 217 ] | 
 40,559 | 
 11 | 
 86,835 | 
 2,924 | 
 2,824 | 

 
 wikidata5m [ 228 ] | 
 4,594,485 | 
 822 | 
 20,614,279 | 
 5,163 | 
 5,163 | 

 
 YAGO3-10 [ 229 ] | 
 123,143 | 
 37 | 
 1,079,040 | 
 4,978 | 
 4,982 | 

 
 YAGO37 [ 230 ] | 
 123,189 | 
 37 | 
 420,623 | 
 50,000 | 
 50,000 | 

 
 M-/YAGO39k [ 231 ] | 
 85,484 | 
 39 | 
 354,997 | 
 9,341 | 
 9,364 | 

 

 
 
 
 Table XI: Typical benchmark datasets for static inductive knowledge graph reasoning. 
 
 
 
 
 Dataset | 
 # Ent. | 
 # Rel. | 
 # Train Facts | 
 # Val. Facts | 
 # Test Facts | 

 
 
 
 WN18RR v1 [ 71 ] | 
 train-graph | 
 2,746 | 
 9 | 
 5,410 | 
 626 | 
 638 | 

 
 ind-test-graph | 
 922 | 
 9 | 
 1,618 | 
 181 | 
 184 | 

 
 WN18RR v2 [ 71 ] | 
 train-graph | 
 6,954 | 
 10 | 
 15,262 | 
 1,837 | 
 1,868 | 

 
 ind-test-graph | 
 2,923 | 
 10 | 
 4,011 | 
 407 | 
 437 | 

 
 WN18RR v3 [ 71 ] | 
 train-graph | 
 12,078 | 
 11 | 
 25,901 | 
 3,097 | 
 3,152 | 

 
 ind-test-graph | 
 5,084 | 
 11 | 
 6,327 | 
 534 | 
 601 | 

 
 WN18RR v4 [ 71 ] | 
 train-graph | 
 3,861 | 
 9 | 
 7,940 | 
 934 | 
 968 | 

 
 ind-test-graph | 
 7,208 | 
 9 | 
 12,334 | 
 1,394 | 
 1,429 | 

 
 FB15k237 v1 [ 71 ] | 
 train-graph | 
 2,000 | 
 183 | 
 4,245 | 
 485 | 
 492 | 

 
 ind-test-graph | 
 1,500 | 
 146 | 
 1,993 | 
 202 | 
 201 | 

 
 FB15k237 v2 [ 71 ] | 
 train-graph | 
 3,000 | 
 203 | 
 9,739 | 
 1,166 | 
 1,180 | 

 
 ind-test-graph | 
 2,000 | 
 176 | 
 4,145 | 
 469 | 
 478 | 

 
 FB15k237 v3 [ 71 ] | 
 train-graph | 
 4,000 | 
 218 | 
 17,986 | 
 2,194 | 
 2,214 | 

 
 ind-test-graph | 
 3,000 | 
 187 | 
 7,406 | 
 866 | 
 865 | 

 
 FB15k237 v4 [ 71 ] | 
 train-graph | 
 5,000 | 
 222 | 
 27,203 | 
 3,352 | 
 3,361 | 

 
 ind-test-graph | 
 3,500 | 
 204 | 
 11,714 | 
 1,416 | 
 1,424 | 

 
 NELL995 v1 [ 71 ] | 
 train-graph | 
 10,915 | 
 14 | 
 4,687 | 
 414 | 
 435 | 

 
 ind-test-graph | 
 225 | 
 14 | 
 833 | 
 97 | 
 96 | 

 
 NELL995 v2 [ 71 ] | 
 train-graph | 
 2,564 | 
 88 | 
 8,219 | 
 922 | 
 968 | 

 
 ind-test-graph | 
 4,937 | 
 79 | 
 4,586 | 
 455 | 
 476 | 

 
 NELL995 v3 [ 71 ] | 
 train-graph | 
 4647 | 
 142 | 
 16,393 | 
 1,851 | 
 1,873 | 

 
 ind-test-graph | 
 4,921 | 
 122 | 
 8,048 | 
 811 | 
 809 | 

 
 NELL995 v4 [ 71 ] | 
 train-graph | 
 2,092 | 
 77 | 
 7,546 | 
 876 | 
 867 | 

 
 ind-test-graph | 
 3,294 | 
 61 | 
 7,073 | 
 716 | 
 731 | 

 
 WN-MBE [ 232 ] | 
 train-graph | 
 19,361 | 
 11 | 
 35,426 | 
 8,858 | 
 - | 

 
 ind-test-graph-1 | 
 3,723 | 
 11 | 
 5,678 | 
 - | 
 1,352 | 

 
 ind-test-graph-2 | 
 4,122 | 
 11 | 
 6,730 | 
 - | 
 1,874 | 

 
 ind-test-graph-3 | 
 4,300 | 
 11 | 
 7,545 | 
 - | 
 2,054 | 

 
 ind-test-graph-4 | 
 4467 | 
 11 | 
 8,623 | 
 - | 
 2,493 | 

 
 ind-test-graph-5 | 
 4,514 | 
 11 | 
 9,608 | 
 - | 
 2,762 | 

 
 FB-MBE [ 232 ] | 
 train-graph | 
 7,203 | 
 237 | 
 125,769 | 
 31,442 | 
 - | 

 
 ind-test-graph-1 | 
 1,458 | 
 237 | 
 18,394 | 
 - | 
 9,240 | 

 
 ind-test-graph-2 | 
 1,461 | 
 237 | 
 19,120 | 
 - | 
 9,669 | 

 
 ind-test-graph-3 | 
 1,467 | 
 237 | 
 19,740 | 
 - | 
 9,887 | 

 
 ind-test-graph-4 | 
 1,467 | 
 237 | 
 22,455 | 
 - | 
 11,127 | 

 
 ind-test-graph-5 | 
 1,471 | 
 237 | 
 22,214 | 
 - | 
 11,059 | 

 
 NELL-MBE [ 232 ] | 
 train-graph | 
 33,348 | 
 200 | 
 88,814 | 
 22,203 | 
 - | 

 
 ind-test-graph-1 | 
 34,488 | 
 3,200 | 
 34,496 | 
 - | 
 3,853 | 

 
 ind-test-graph-2 | 
 36031 | 
 3,200 | 
 35,411 | 
 - | 
 31,059 | 

 
 ind-test-graph-3 | 
 37,660 | 
 3,200 | 
 36,543 | 
 - | 
 31,277 | 

 
 ind-test-graph-4 | 
 39,056 | 
 3,200 | 
 37,667 | 
 - | 
 31,427 | 

 
 ind-test-graph-5 | 
 310,616 | 
 3,200 | 
 38,876 | 
 - | 
 31,595 | 

 

 
 
 
 
 • 
 
 ATOMIC [ 205 ] is an KGs for everyday commonsense reasoning. It is composed of the reactions, effects, and intents of human behaviors and descriptions of each entity.

 

 • 
 
 Countries [ 206 ] consists of relations among countries based on public geographical data.

 

 • 
 
 CoDEX [ 207 ] is a set of COmpletion Datasets EXtracted from Wikidata and Wikipedia, which contains three different sizes of sub-KGs, i.e., CoDEX-S, CoDEX-M, CoDEX-L.

 

 • 
 
 Conceptnet [ 208 ] connects words and phrases with labeled edges to enhance AI APPs to understand word meanings better. Conceptnet100K [ 209 ] contains 100k training triplets.

 

 • 
 
 DBpedia [ 233 ] consists of structured content from the information created in various Wikimedia projects. According to the entity set size, we can derive several subsets from it, i.e., DBpedia50 [ 210 ] , DBpedia500 [ 210 ] and DB100K [ 211 ] .

 

 • 
 
 FAMILY [ 212 ] consists of relations among family members.

 

 • 
 
 FreeBASE [ 234 ] is a large knowledge base generated from multiple sources, such as Wikipedia, NNDB, Fashion Model Directory, etc. According to the entity set size, we can derive several subsets from it, including FB13 [ 213 ] , FB122 [ 214 ] , FB15k [ 215 ] , FB20k [ 210 ] , FB24k [ 216 ] , FB5M [ 19 ] , FB15k-237 [ 217 ] , FB60k-NYT10 [ 218 ] .

 

 • 
 
 Hetionet [ 219 ] is a knowledge graph derived from biomedical studies based on public resources. It describes relations among compounds, diseases, genes, anatomies, pathways, biological processes, molecular functions, cellular components, pharmacologic classes, side effects, and symptoms.

 

 • 
 
 Kinship [ 212 ] describe kinships in Alyawarra tribes [ 235 ] .

 

 • 
 
 Nation [ 212 ] contains relations among nations [ 221 ] .

 

 • 
 
 NELL [ 236 ] is the knowledge base built based on Never-Ending Language Learner, which attempts to learn to read the web over time. According to the entity set size, we can derive several subsets from it, e.g., Location [ 220 ] , sports [ 220 ] , NELL23k [ 222 ] , NELL-995 [ 223 ] .

 

 • 
 
 OpenBioLink [ 224 ] is a large-scale, high-quality, and highly challenging biomedical KG.

 

 • 
 
 Toy [ 225 ] is a small KG used for testing and debugging.

 

 • 
 
 UMLS [ 237 ] is the KG of the Unified Medical Language System. By cooperating with the PubMed corpus, it is extended to UMLS-PubMed [ 218 ] .

 

 • 
 
 WordNet [ 238 ] is a lexical database of semantic relations, e.g., synonyms, hyponyms, and meronyms, between words. According to the entity set size, we can derive several subsets from it, e.g., WN11 [ 213 ] , WN18 [ 227 ] , WN18RR [ 217 ] .

 

 • 
 
 Wikidata [ 239 ] provides common sources for Wikipedia, where WD-singer [ 222 ] and wikidata5m [ 228 ] are subsets.

 

 • 
 
 YAGO [ 240 ] , as a lightweight and extensible ontology, is built from Wikidata and unified with WordNet. According to the sizes of relations, YAGO3-10 [ 229 ] , YAGO37 [ 230 ] and YAGO39k [ 231 ] can be derived.

 

 
 
 
 

### VI-B Temporal KGR Datasets 

 
 Eighteen Typical temporal KGR datasets are summarized. The statistic is presented in Table XII , and descriptions are listed below.

 
 
 Table XII: Typical benchmark datasets for temporal knowledge graph reasoning. 
 
 
 
 
 Dataset | 
 # Ent. | 
 # Rel. | 
 # Timestamps | 
 # Train Facts | 
 # Val. Facts | 
 # Test Facts | 

 
 
 
 DBpedia-3SP [ 241 ] | 
 66,967 | 
 968 | 
 3 | 
 103,211 | 
 3,000 | 
 - | 

 
 GDELT [ 134 ] | 
 7,691 | 
 240 | 
 8,925 | 
 1,033,270 | 
 238,765 | 
 305,241 | 

 
 GDELT-small [ 119 ] | 
 500 | 
 20 | 
 366 | 
 2,735,685 | 
 341,961 | 
 341,961 | 

 
 GDELT-m10 [ 242 ] | 
 50 | 
 20 | 
 30 | 
 221,132 | 
 27,608 | 
 27,926 | 

 
 IMDB-13-3SP [ 134 ] | 
 3,244,455 | 
 14 | 
 3 | 
 7,913,773 | 
 10,000 | 
 - | 

 
 IMDB-30SP [ 134 ] | 
 243,148 | 
 14 | 
 30 | 
 621,096 | 
 3,000 | 
 3,000 | 

 
 ICEWS05-15 [ 243 ] | 
 10,488 | 
 251 | 
 4,017 | 
 386,962 | 
 46,092 | 
 46275 | 

 
 ICEWS11-14 [ 243 ] | 
 6,738 | 
 235 | 
 1,461 | 
 118,766 | 
 14,859 | 
 14,756 | 

 
 ICEWS14 [ 115 ] | 
 7,128 | 
 230 | 
 365 | 
 63,685 | 
 13,823 | 
 13,222 | 

 
 ICEWS14-Plus [ 242 ] | 
 7,128 | 
 230 | 
 365 | 
 72,826 | 
 8,941 | 
 8,963 | 

 
 ICEWS18 [ 115 ] | 
 23,033 | 
 256 | 
 7,272 | 
 373,018 | 
 45,995 | 
 49,545 | 

 
 YOGA11k/YOGA [ 161 ] | 
 10,623 | 
 10 | 
 189 | 
 161,540 | 
 19,523 | 
 20,026 | 

 
 YOGA-3SP [ 241 ] | 
 27,009 | 
 37 | 
 3 | 
 124,757 | 
 3,000 | 
 3,000 | 

 
 YOGA15k [ 243 ] | 
 15,403 | 
 34 | 
 198 | 
 110,441 | 
 13,815 | 
 13,800 | 

 
 YOGA1830 [ 115 ] | 
 10,038 | 
 10 | 
 205 | 
 51,205 | 
 10,973 | 
 10,973 | 

 
 WIKI/Wikidata12k [ 161 ] | 
 12,554 | 
 24 | 
 232 | 
 2,735,685 | 
 341,961 | 
 341,961 | 

 
 Wikidata11k [ 167 ] | 
 11,134 | 
 95 | 
 328 | 
 242,844 | 
 28,748 | 
 14,283 | 

 
 Wikidata-big [ 140 ] | 
 125,726 | 
 203 | 
 1,700 | 
 323,635 | 
 5,000 | 
 5,000 | 

 

 
 
 
 
 • 
 
 DBpedia-3SP [ 241 ] is extracted subsets from DBpedia in three different timestamps.

 

 • 
 
 GDELT [ 134 ] is a dense KG derived from the Global Database of Events, Language, and Tone. GDELT-m10 [ 242 ] and GDELT-small [ 119 ] are extracted from it.

 

 • 
 
 IMDB [ 244 ] is a KG consisting of the entities of movies, TV series, actors, and directors, which is also known as the Internet Movie Database. IMDB-30SP [ 134 ] and IMDB-13-3SP [ 134 ] are extracted from the dataset in different timestamps.

 

 • 
 
 ICEWS [ 245 ] , short for Integrated Crisis Early Warning System, is a database that contains political events with specific timestamps. Some typical temporal KGs are created out of it, i.e., ICEWS05-15 [ 243 ] , ICEWS11-14 [ 243 ] , ICEWS14 [ 115 ] , ICEWS14-Plus [ 242 ] , ICEWS18 [ 115 ] .

 

 • 
 
 Wikidata [ 239 ] for temporal KGR contains extra time information than the static Wikidata dataset. WIKI/Wikidata12k [ 161 ] , Wikidata11k [ 167 ] and Wikidata-big [ 140 ] are generated based on different periods.

 

 • 
 
 YAGO [ 240 ] for temporal KGR contains extra time information. YOGA11k/YOGA [ 161 ] , YOGA15k [ 243 ] , YOGA-3SP [ 241 ] and YOGA1830 [ 115 ] are generated from it according to different periods.

 

 
 
 
 

### VI-C Multi-Modal KGR Datasets 

 
 Eleven typical multi-modal KGR datasets are summarized. The statistic is presented in Table XIII , and descriptions are listed below.

 
 • 
 
 FB-IMG-TXT [ 189 ] is the KG combined with textual descriptions and images. The triple part is the subset of a classical KG dataset FB15k [ 215 ] , and the images are extracted from ImageNet [ 246 ] . Compared to it, FB15K-237-IMG [ 176 ] changes the scope of triplets to FB15k-237 [ 217 ] .

 

 • 
 
 IMGpedia [ 247 ] is the KG, which incorporates visual information of the images from the Wikimedia Commons dataset.

 

 • 
 
 MKG [ 248 ] consists of two subsets, i.e., MKG-Wikipedia and MKG-YAGO . They both contain visual entities generated by web search engines. But, their triplet parts are extracted from Wikipedia and YAGO, respectively.

 

 • 
 
 MMKG [ 249 ] offers three subsets, including MMKG-FB15k-IMG , MMKG-DB15k , Yago15k-IMG-TXT , which integrates specific KGs with numeric literals and images.

 

 • 
 
 Richpedia [ 250 ] is composed of the triplets, textual descriptions, and images. The textual descriptions are derived from Wikidata, and the corresponding visual resources are crawled from the website.

 

 • 
 
 WN9-IMG-TXT [ 184 ] is the KG combined with textual descriptions and images. The triple part is the subset of a classical KG dataset WN18 [ 227 ] , and the images are extracted from ImageNet [ 246 ] . Compared to it, WN18-IMG [ 176 ] changes the scope of triplets to the whole WN18.

 

 
 
 
 Table XIII: Typical benchmark datasets for multi-modal knowledge graph reasoning. 
 
 
 
 
 Dataset | 
 Modality | 
 # Ent. | 
 # Rel. | 
 # Train Facts | 
 # Val. Facts | 
 # Test. Facts | 

 
 
 
 FB-IMG-TXT [ 189 ] | 
 KG | 
 11,757 | 
 1,231 | 
 285,850 | 
 34,863 | 
 29,580 | 

 
 TXT | 
 11,757 | 

 
 IMG | 
 1,175,700 | 

 
 FB15k-237-IMG [ 176 ] | 
 KG | 
 14,541 | 
 237 | 
 272,115 | 
 17,535 | 
 20,466 | 

 
 IMG | 
 145,410 | 

 
 IMGpedia [ 247 ] | 
 KG | 
 14,765,300 | 
 442,959,000 | 
 3,119,207,705 | 
 - | 
 - | 

 
 IMG | 
 44,295,900 | 

 
 MMKG-FB15k [ 249 ] | 
 KG | 
 14,951 | 
 1,345 | 
 592,213 | 
 - | 
 - | 

 
 Numeric | 
 29,395 | 
 29,395 | 
 - | 
 - | 

 
 IMG | 
 13,444 | 
 13,444 | 
 - | 
 - | 

 
 MMKG-DB15k [ 249 ] | 
 KG | 
 14,777 | 
 279 | 
 99,028 | 
 - | 
 - | 

 
 Numeric | 
 46,121 | 
 46,121 | 
 - | 
 - | 

 
 IMG | 
 12,841 | 
 12,841 | 
 - | 
 - | 

 
 MMKG-Yago15k [ 249 ] | 
 KG | 
 15,283 | 
 32 | 
 122,886 | 
 - | 
 - | 

 
 Numeric | 
 48,405 | 
 48,405 | 
 - | 
 - | 

 
 IMG | 
 11,194 | 
 11,194 | 
 - | 
 - | 

 
 MKG-Wikipedia [ 202 ] | 
 KG | 
 15,000 | 
 169 | 
 34,196 | 
 4,274 | 
 4,276 | 

 
 TXT | 
 14,123 | 

 
 IMG | 
 14,463 | 

 
 MKG-YAGO [ 202 ] | 
 KG | 
 15,000 | 
 28 | 
 21,310 | 
 2,663 | 
 2,665 | 

 
 TXT | 
 12,305 | 

 
 IMG | 
 14,244 | 

 
 RichPedia [ 250 ] | 
 KG | 
 29,985 | 
 3 | 
 119,669,570 | 
 - | 
 - | 

 
 IMG | 
 2,914,770 | 

 
 WN9-IMG-TXT [ 189 ] | 
 KG | 
 6,555 | 
 9 | 
 11,741 | 
 1,319 | 
 1,337 | 

 
 TXT | 
 6,555 | 

 
 IMG | 
 63,225 | 

 
 WN18-IMG [ 176 ] | 
 KG | 
 14,541 | 
 18 | 
 141,442 | 
 5,000 | 
 5,000 | 

 
 IMG | 
 145,410 | 

 

 
 
 
 
 

## VII Challenge and Opportunity 

 
 According to previous analyses of the existing KGR models, we point out several promising directions for future works.

 
 

### VII-A Out-of-distribution Reasoning 

 
 In the real-world scenario, new entities and relations are continuously emerging in the KGs, which are under-explored in the original KGs. Reasoning on the facts with these under-explored elements is called out-of-distribution reasoning, which raises higher requirements for the KGR model design.
Some recent attempts provide potential solutions for inferring unseen entities, which are known as inductive reasoning models, such as [ 71 , 72 , 73 , 75 ] . These models mine the logic rules underlying the graph structure without considering the specific meaning of entities, which achieve promising performances. As for the unseen relation inference, few-shot KGR models [ 85 , 126 , 74 ] tend to improve the generalization ability of models so that the trained model can scale well to the unseen relations with a small amount of facts.
In other words, these few-shot KGR models can quickly learn new tasks according to the previously learned similar knowledge. Besides, BERTRL [ 83 ] tries to handle this case based on their textual semantics calculated by language models. While the performance of these models would drop drastically when language models are not finely trained. In conclusion, the KGR models for out-of-distribution reasoning tasks are still in an early stage, which is worth exploring in-depth in the future.

 
 
 

### VII-B Large-scale Reasoning 

 
 The industrial KGs are generally large-scale, which requires more efficient KGR models. To this end, some existing works try to optimize the propagation procedures in a progressive manner [ 251 ] . For instance, NBF-net [ 80 ] integrates the bellman-ford algorithm to substitute the original DFS-based aggregation procedure in GNN-based KGR models. Moreover, A ∗ Star [ 251 ] Net further optimizes the aggregation procedure with the greedy algorithm. Besides, the idea of graph clustering [ 252 , 253 , 254 ] is also used for it. For example, CURL [ 94 ] first separates the KGs into different clusters according to the entity semantics and then fine-grains the path-finding procedure into two-level, i.e., the intra-cluster level and the inter-cluster level. It reduces the unnecessary searching for the whole graphs. Similarly, many works perform reasoning on sub-graphs instead of complete graphs, such as GraIL [ 71 ] , CSR [ 85 ] etc. But most of them sacrifice the precision of inference, which may still be explored for more all-around models.

 
 
 

### VII-C Multi-relational Reasoning 

 
 The situation that multi-relational facts exist between two entities is common in KGs as shown in Figure 13 (a). However, they are more diverse in structure and more complex in semantics compared to uni-relational and bi-relational facts as shown in Figure 13 (b) and (c).
Thus, the existing KGR models mainly focus on uni-relational and bi-relational facts and even usually treat multi-relational facts as uni-relational and bi-relational facts by omitting some of the facts. The KGR models in such a manner cannot accurately model real situations and lose lots of meaningful semantic information, leading to insufficient expressive ability. In the future, it is necessary to study how to leverage multi-relational facts to enhance reasoning ability.

 
 
 Figure 13: Comparison of multi-relational, bi-relational and uni-relational facts. 
 
 
 

### VII-D Multi-modal Reasoning 

 
 Knowledge reasoning based on the fusion of multi-source information can reduce the disconnectedness and sparsity of knowledge graphs by combining a text corpus or additional information in other modalities. Knowledge reasoning based on the fusion of data in multiple modalities can complement each other’s advantages and improve reasoning performance. However, existing multi-modal KGR models are still at an early stage. They still tend to directly concat the embeddings in different modalities together for final score calculation. Such simple fusion modes have shown their promising performances while developing more fine-grained and scalable modes is still worthwhile. For instance, an adaptive fusion mode, which weighs the importance of different modalities, is worthwhile exploring.

 
 
 

### VII-E Explainable Reasoning 

 
 Explainability is a common and important issue for deep learning models in various fields. Although KGR models generally are more explainable, it is still worthwhile exploring more in this topic, especially for embedding-based KGR models. Nowadays, more and more KGR models are developed based on neural networks, such as GNN [ 255 ] . Most of them have the great expressive ability but suffer from explainability. Compared to them, rule-based and path-based KGR models are more explainable but computation-consuming and less expressive [ 256 ] . To achieve a good trade-off between expressive ability and explainability, there exist some attempts to integrate the embedding-based models with rule-based and path-based models, such as ARGCN [ 232 ] . It builds the reward function based on the embeddings generated by the RGCN [ 58 ] , which makes those path-based models more explainable. However, most of these attempts are still rough.

 
 
 

### VII-F Knowledge Graph Reasoning Application 

 
 Although a large amount of KGR methods have been proposed in recent years, demonstrating the great potential of KGR in theoretical fields, the applications of KGR still need to be studied more [ 257 ] . Nowadays, knowledge graphs are commonly used in many downstream applications, such as medicine, finance, plagiarism detection, etc. Medical knowledge reasoning models aim to assist doctors in diagnosing diseases from electronic medical records. For example, [ 258 ] and [ 259 ] both perform reasoning on the KG constructed from the electronic medical database. The pre-trained language models, such as Bert, are leveraged to generate textual embedding of entities, which is proven effective in existing multi-modal KGR models. Besides, KGR models can also help with Anti-fraud detection, which is an important task in the finance field. For instance, [ 260 ] proposes a case-based reasoning method to assist people in verifying the information to discriminate against fraud in advance. Additionally, [ 261 ] executes plagiarism detection by conducting the KGR approach in a continuous learning manner.

 
 
 

### VII-G Knowledge Graph and Large Language Model 

 
 Large language models (LLMs) [ 262 ] , i.e., ChatGPT, GPT-4, are very popular this year, which have huge impacts due to their promising reasoning capacity and generalizability [ 263 ] . However, these models still suffer from two issues, i.e., (1) the poor explainability, and (2) poor scalability when handling new data, which may be solved when if well cooperated with KGR models. For example, QA-GNN [ 264 ] first attempts to use LLMs for text preprocessing and further guides the reasoning step on the KGs. Besides, DRAGON [ 181 ] is an LLM-guided logical reasoning method for multi-modal KGR. Meanwhile, since more and more data are being trained, some researchers suggest that LLM is a more general KG in the future, which can also achieve functions, like indexing, reasoning, storing, etc. Various hypotheses about the connections between LLM and KG are all reasonable, and we cannot say which one is more valuable. However, we can be sure that exploring and researching between LLM and KG will also be one of the hot spots in the future.

 
 
 
 

## VIII Conclusion 

 
 Our survey thoroughly reviews the existing KGR models based on a bi-level taxonomy, i.e., top level (graph types), and base level (techniques, scenarios). Three graph types ( i.e., static, temporal, multi-modal KGs), fourteen techniques, and four reasoning scenarios are included, which provides systematical reviews for KGR. Besides, we summarize the challenges of knowledge graph reasoning and point out some potential opportunities which will enlighten the readers. The corresponding open-source repository for the collection of 180 state-of-the-art KGR models ( i.e., papers, and codes) and 67 typical datasets are shared on GitHub to convenient the community.

 
 
 

## References

 
 
 [1] 
 
H. Yuan, H. Yu, S. Gui, and S. Ji, “Explainability in graph neural networks: A
taxonomic survey,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2022.

 

 
 [2] 
 
F.-A. Croitoru, V. Hondru, R. T. Ionescu, and M. Shah, “Diffusion models in
vision: A survey,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2023.

 

 
 [3] 
 
Y. Zhang, B. Kang, B. Hooi, S. Yan, and J. Feng, “Deep long-tailed learning: A
survey,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2023.

 

 
 [4] 
 
P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A
survey,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2023.

 

 
 [5] 
 
M. Ali, M. Berrendorf, C. T. Hoyt, L. Vermue, M. Galkin, S. Sharifzadeh,
A. Fischer, V. Tresp, and J. Lehmann, “Bringing light into the dark: A
large-scale evaluation of knowledge graph embedding models under a unified
framework,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , vol. 44, no. 12, pp. 8825–8845, 2021.

 

 
 [6] 
 
K. Liang, Y. Liu, S. Zhou, X. Liu, and W. Tu, “Relational symmetry based
knowledge graph contrastive learning,” 2022.

 

 
 [7] 
 
S. Ji, S. Pan, E. Cambria, P. Marttinen, and S. Y. Philip, “A survey on
knowledge graphs: Representation, acquisition, and applications,” IEEE
Transactions on Neural Networks and Learning Systems , 2021.

 

 
 [8] 
 
C.-M. Wong, F. Feng, W. Zhang, C.-M. Vong, H. Chen, Y. Zhang, P. He, H. Chen,
K. Zhao, and H. Chen, “Improving conversational recommender system by
pretraining billion-scale knowledge graph,” in 2021 IEEE 37th
International Conference on Data Engineering (ICDE) , 2021.

 

 
 [9] 
 
P. Hitzler, K. Janowicz, W. Li, G. Qi, and Q. Ji, “Hybrid reasoning in
knowledge graphs: Combing symbolic reasoning and statistical reasoning,”
 Semant. Web , 2020.

 

 
 [10] 
 
J. Zhang, B. Chen, L. Zhang, X. Ke, and H. Ding, “Neural, symbolic and
neural-symbolic reasoning on knowledge graphs,” AI Open , 2021.

 

 
 [11] 
 
W. Zhang, J. Chen, J. Li, Z. Xu, J. Z. Pan, and H. Chen, “Knowledge graph
reasoning with logics and embeddings: Survey and perspective,” arXiv
preprint arXiv:2202.07412 , 2022.

 

 
 [12] 
 
Y. Chen, H. Li, H. Li, W. Liu, Y. Wu, Q. Huang, and S. Wan, “An overview of
knowledge graph reasoning: Key technologies and applications,” Journal
of Sensor and Actuator Networks , 2022.

 

 
 [13] 
 
M. Chen, W. Zhang, Y. Geng, Z. Xu, J. Z. Pan, and H. Chen, “Generalizing to
unseen elements: A survey on knowledge extrapolation for knowledge graphs,”
 arXiv preprint arXiv:2302.01859 , 2023.

 

 
 [14] 
 
B. Cai, Y. Xiang, L. Gao, H. Zhang, Y. Li, and J. Li, “Temporal knowledge
graph completion: A survey,” arXiv preprint arXiv:2201.08236 , 2022.

 

 
 [15] 
 
X. Zhu, Z. Li, X. Wang, X. Jiang, P. Sun, X. Wang, Y. Xiao, and N. J. Yuan,
“Multi-modal knowledge graph construction and application: A survey,”
 arXiv preprint arXiv:2202.05786 , 2022.

 

 
 [16] 
 
R. H. Richens, “Preprogramming for mechanical translation.” Mech.
Transl. Comput. Linguistics , 1956.

 

 
 [17] 
 
X. Zhu, Z. Li, X. Wang, X. Jiang, P. Sun, X. Wang, Y. Xiao, and N. J. Yuan,
“Multi-modal knowledge graph construction and application: A survey,”
 ArXiv , 2022.

 

 
 [18] 
 
A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko,
“Translating embeddings for modeling multi-relational data,” Proc. of
NeurIPS , 2013.

 

 
 [19] 
 
Z. Wang, J. Zhang, J. Feng, and Z. Chen, “Knowledge graph embedding by
translating on hyperplanes,” in Proc. of AAAI , 2014.

 

 
 [20] 
 
Y. Lin, Z. Liu, M. Sun, Y. Liu, and X. Zhu, “Learning entity and relation
embeddings for knowledge graph completion,” in Proc. of AAAI , 2015.

 

 
 [21] 
 
G. Ji, S. He, L. Xu, K. Liu, and J. Zhao, “Knowledge graph embedding via
dynamic mapping matrix,” in Proc. of ACL , 2015.

 

 
 [22] 
 
S. He, K. Liu, G. Ji, and J. Zhao, “Learning to represent knowledge graphs
with gaussian embedding,” in Proceedings of the 24th ACM international
on conference on information and knowledge management , 2015.

 

 
 [23] 
 
H. Xiao, M. Huang, Y. Hao, and X. Zhu, “Transg: A generative mixture model for
knowledge graph embedding,” arXiv preprint arXiv:1509.05488 , 2015.

 

 
 [24] 
 
G. Ji, K. Liu, S. He, and J. Zhao, “Knowledge graph completion with adaptive
sparse transfer matrix,” in Proc. of AAAI , 2016.

 

 
 [25] 
 
T. Ebisu and R. Ichise, “Toruse: Knowledge graph embedding on a lie group,”
in Proc. of AAAI , 2018.

 

 
 [26] 
 
I. Balazevic, C. Allen, and T. Hospedales, “Multi-relational poincaré
graph embeddings,” Proc. of NeurIPS , 2019.

 

 
 [27] 
 
L. Ma, P. Sun, Z. Lin, and H. Wang, “Composing knowledge graph embeddings via
word embeddings,” arXiv preprint arXiv:1909.03794 , 2019.

 

 
 [28] 
 
Z. Sun, Z.-H. Deng, J.-Y. Nie, and J. Tang, “Rotate: Knowledge graph embedding
by relational rotation in complex space,” arXiv preprint
arXiv:1902.10197 , 2019.

 

 
 [29] 
 
Z. Zhang, J. Cai, Y. Zhang, and J. Wang, “Learning hierarchy-aware knowledge
graph embeddings for link prediction,” in Proc. of AAAI , 2020.

 

 
 [30] 
 
F. Zhang, X. Wang, Z. Li, and J. Li, “Transrhs: A representation learning
method for knowledge graphs with relation hierarchical structure,” in
 Proc. of IJCAI , 2021.

 

 
 [31] 
 
L. Chao, J. He, T. Wang, and W. Chu, “Pairre: Knowledge graph embeddings via
paired relation vectors,” in ACL , 2021.

 

 
 [32] 
 
R. Li, J. Zhao, C. Li, D. He, Y. Wang, Y. Liu, H. Sun, S. Wang, W. Deng,
Y. Shen et al. , “House: Knowledge graph embedding with householder
parameterization,” arXiv preprint arXiv:2202.07919 , 2022.

 

 
 [33] 
 
L. Yu, Z. Luo, H. Liu, D. Lin, H. Li, and Y. Deng, “Triplere: Knowledge graph
embeddings via tripled relation vectors,” arXiv preprint
arXiv:2209.08271 , 2022.

 

 
 [34] 
 
B. Wang, Q. Meng, Z. Wang, D. Wu, W. Che, S. Wang, Z. Chen, and C. Liu,
“Interht: Knowledge graph embeddings by interaction between head and tail
entities,” ArXiv preprint , vol. abs/2202.04897, 2022. [Online].
Available: https://arxiv.org/abs/2202.04897

 

 
 [35] 
 
M. Nickel, V. Tresp, and H.-P. Kriegel, “A three-way model for collective
learning on multi-relational data,” in ICML , 2011.

 

 
 [36] 
 
B. Yang, W.-t. Yih, X. He, J. Gao, and L. Deng, “Embedding entities and
relations for learning and inference in knowledge bases,” arXiv
preprint arXiv:1412.6575 , 2014.

 

 
 [37] 
 
T. Trouillon, J. Welbl, S. Riedel, É. Gaussier, and G. Bouchard, “Complex
embeddings for simple link prediction,” in Proc. of ICML , 2016.

 

 
 [38] 
 
M. Nickel, L. Rosasco, and T. Poggio, “Holographic embeddings of knowledge
graphs,” in Proc. of AAAI , 2016.

 

 
 [39] 
 
H. Liu, Y. Wu, and Y. Yang, “Analogical inference for multi-relational
embeddings,” in Proc. of ICML , 2017.

 

 
 [40] 
 
S. M. Kazemi and D. Poole, “Simple embedding for link prediction in knowledge
graphs,” Proc. of NeurIPS , 2018.

 

 
 [41] 
 
I. Balažević, C. Allen, and T. M. Hospedales, “Tucker: Tensor
factorization for knowledge graph completion,” arXiv preprint
arXiv:1901.09590 , 2019.

 

 
 [42] 
 
W. Zhang, B. Paudel, W. Zhang, A. Bernstein, and H. Chen, “Interaction
embeddings for prediction and explanation in knowledge graphs,” in
 Proc. of WSDM , 2019.

 

 
 [43] 
 
S. Zhang, Y. Tay, L. Yao, and Q. Liu, “Quaternion knowledge graph
embeddings,” Proc. of NeurIPS , 2019.

 

 
 [44] 
 
Z. Cao, Q. Xu, Z. Yang, X. Cao, and Q. Huang, “Dual quaternion knowledge graph
embeddings,” in Proc. of AAAI , 2021.

 

 
 [45] 
 
A. Bastos, K. Singh, A. Nadgeri, S. Shekarpour, I. O. Mulang, and J. Hoffart,
“Hopfe: Knowledge graph representation learning using inverse hopf
fibrations,” in Proc. of CIKM , 2021.

 

 
 [46] 
 
S. Amin, S. Varanasi, K. A. Dunfield, and G. Neumann, “Lowfer: Low-rank
bilinear pooling for link prediction,” in Proc. of ICML , 2020.

 

 
 [47] 
 
D. Q. Nguyen, T. Vu, T. D. Nguyen, and D. Phung, “Quatre: Relation-aware
quaternions for knowledge graph embeddings,” in Proc. of WWW , 2022.

 

 
 [48] 
 
A. Bordes, X. Glorot, J. Weston, and Y. Bengio, “A semantic matching energy
function for learning with multi-relational data,” Machine Learning ,
2014.

 

 
 [49] 
 
R. Socher, D. Chen, C. D. Manning, and A. Ng, “Reasoning with neural tensor
networks for knowledge base completion,” Proc. of NeurIPS , 2013.

 

 
 [50] 
 
Q. Liu, H. Jiang, A. Evdokimov, Z.-H. Ling, X. Zhu, S. Wei, and Y. Hu,
“Probabilistic reasoning via deep learning: Neural association models,”
 arXiv preprint arXiv:1603.07704 , 2016.

 

 
 [51] 
 
B. Shi and T. Weninger, “Proje: Embedding projection for knowledge graph
completion,” in Proc. of AAAI , 2017.

 

 
 [52] 
 
T. Dettmers, P. Minervini, P. Stenetorp, and S. Riedel, “Convolutional 2d
knowledge graph embeddings,” in Proc. of AAAI , 2018.

 

 
 [53] 
 
D. Q. Nguyen, T. D. Nguyen, D. Q. Nguyen, and D. Phung, “A novel embedding
model for knowledge base completion based on convolutional neural network,”
 arXiv preprint arXiv:1712.02121 , 2017.

 

 
 [54] 
 
I. Balažević, C. Allen, and T. M. Hospedales, “Hypernetwork
knowledge graph embeddings,” in Proc. of ICANN , 2019.

 

 
 [55] 
 
X. Jiang, Q. Wang, and B. Wang, “Adaptive convolution for multi-relational
learning,” in Proc. of AACL , 2019.

 

 
 [56] 
 
S. Vashishth, S. Sanyal, V. Nitin, N. Agrawal, and P. Talukdar, “Interacte:
Improving convolution-based knowledge graph embeddings by increasing feature
interactions,” in Proc. of AAAI , 2020.

 

 
 [57] 
 
C. Demir and A.-C. N. Ngomo, “Convolutional complex knowledge graph
embeddings,” in European Semantic Web Conference , 2021.

 

 
 [58] 
 
M. Schlichtkrull, T. N. Kipf, P. Bloem, R. v. d. Berg, I. Titov, and
M. Welling, “Modeling relational data with graph convolutional networks,”
in European semantic web conference , 2018.

 

 
 [59] 
 
Z. Wang, Z. Ren, C. He, P. Zhang, and Y. Hu, “Robust embedding with
multi-level structures for link prediction.” in Proc. of IJCAI , 2019.

 

 
 [60] 
 
D. Nathani, J. Chauhan, C. Sharma, and M. Kaul, “Learning attention-based
embeddings for relation prediction in knowledge graphs,” arXiv
preprint arXiv:1906.01195 , 2019.

 

 
 [61] 
 
P. Wang, J. Han, C. Li, and R. Pan, “Logic attention based neighborhood
aggregation for inductive knowledge graph embedding,” in Proc. of
AAAI , 2019.

 

 
 [62] 
 
C. Shang, Y. Tang, J. Huang, J. Bi, X. He, and B. Zhou, “End-to-end
structure-aware convolutional networks for knowledge base completion,” in
 Proc. of AAAI , 2019.

 

 
 [63] 
 
L. Cai, B. Yan, G. Mai, K. Janowicz, and R. Zhu, “Transgcn: Coupling
transformation assumptions with graph convolutional networks for link
prediction,” in Proceedings of the 10th International Conference on
Knowledge Capture , 2019.

 

 
 [64] 
 
X. Xu, W. Feng, Y. Jiang, X. Xie, Z. Sun, and Z.-H. Deng, “Dynamically pruned
message passing networks for large-scale knowledge graph reasoning,”
 arXiv preprint arXiv:1909.11334 , 2019.

 

 
 [65] 
 
Z. Zhang, F. Zhuang, H. Zhu, Z. Shi, H. Xiong, and Q. He, “Relational graph
neural network with hierarchical attention for knowledge graph completion,”
in Proc. of AAAI , 2020.

 

 
 [66] 
 
D. Yu, Y. Yang, R. Zhang, and Y. Wu, “Knowledge embedding based graph
convolutional network,” in Proc. of WWW , 2021.

 

 
 [67] 
 
S. Vashishth, S. Sanyal, V. Nitin, and P. Talukdar, “Composition-based
multi-relational graph convolutional networks,” arXiv preprint
arXiv:1911.03082 , 2019.

 

 
 [68] 
 
J. Baek, D. B. Lee, and S. J. Hwang, “Learning to extrapolate knowledge:
Transductive few-shot out-of-graph link prediction,” Proc. of
NeurIPS , 2020.

 

 
 [69] 
 
Y. Zhang, W. Wang, W. Chen, J. Xu, A. Liu, and L. Zhao, “Meta-learning based
hyper-relation feature modeling for out-of-knowledge-base embedding,” in
 Proc. of CIKM , 2021.

 

 
 [70] 
 
S. Liu, B. Grau, I. Horrocks, and E. Kostylev, “Indigo: Gnn-based inductive
knowledge graph completion using pair-wise encoding,” Proc. of
NeurIPS , 2021.

 

 
 [71] 
 
K. Teru, E. Denis, and W. Hamilton, “Inductive relation prediction by subgraph
reasoning,” in Proc. of ICML , 2020.

 

 
 [72] 
 
J. Chen, H. He, F. Wu, and J. Wang, “Topology-aware correlations between
relations for inductive link prediction in knowledge graphs,” in Proc.
of AAAI , 2021.

 

 
 [73] 
 
S. Mai, S. Zheng, Y. Yang, and H. Hu, “Communicative message passing for
inductive relation reasoning.” in Proc. of AAAI , 2021.

 

 
 [74] 
 
S. Zheng, S. Mai, Y. Sun, H. Hu, and Y. Yang, “Subgraph-aware few-shot
inductive link prediction via meta-learning,” IEEE Transactions on
Knowledge and Data Engineering , 2022.

 

 
 [75] 
 
X. Xu, P. Zhang, Y. He, C. Chao, and C. Yan, “Subgraph neighboring relations
infomax for inductive link prediction on knowledge graphs,” arXiv
preprint arXiv:2208.00850 , 2022.

 

 
 [76] 
 
Y. Pan, J. Liu, L. Zhang, X. Hu, T. Zhao, and Q. Lin, “Learning first-order
rules with relational path contrast for inductive relation reasoning,”
 arXiv preprint arXiv:2110.08810 , 2021.

 

 
 [77] 
 
Y. Geng, J. Chen, W. Zhang, J. Z. Pan, M. Chen, H. Chen, and S. Jiang,
“Relational message passing for fully inductive knowledge graph
completion,” arXiv preprint arXiv:2210.03994 , 2022.

 

 
 [78] 
 
Z. Hu, V. Gutiérrez-Basulto, Z. Xiang, X. Li, R. Li, and J. Z. Pan,
“Type-aware embeddings for multi-hop reasoning over knowledge graphs,”
 arXiv preprint arXiv:2205.00782 , 2022.

 

 
 [79] 
 
T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for
contrastive learning of visual representations,” in Proc. of ICML ,
2020.

 

 
 [80] 
 
Z. Zhu, Z. Zhang, L.-P. Xhonneux, and J. Tang, “Neural bellman-ford networks:
A general graph neural network framework for link prediction,” Proc.
of NeurIPS , 2021.

 

 
 [81] 
 
Y. Zhang and Q. Yao, “Knowledge graph reasoning with relational digraph,” in
 Proceedings of the ACM Web Conference 2022 , 2022.

 

 
 [82] 
 
L. V. Harsha Vardhan, G. Jia, and S. Kok, “Probabilistic logic graph attention
networks for reasoning,” in Proc. of WWW , 2020.

 

 
 [83] 
 
H. Zha, Z. Chen, and X. Yan, “Inductive relation prediction by bert,” in
 Proc. of AAAI , 2022.

 

 
 [84] 
 
Q. Lin, J. Liu, F. Xu, Y. Pan, Y. Zhu, L. Zhang, and T. Zhao, “Incorporating
context graph with logical reasoning for inductive relation prediction,” in
 Proc. of SIGIR , 2022.

 

 
 [85] 
 
Q. Huang, H. Ren, and J. Leskovec, “Few-shot relational reasoning via
connection subgraph pretraining,” arXiv preprint arXiv:2210.06722 ,
2022.

 

 
 [86] 
 
Y. Pan, J. Liu, L. Zhang, T. Zhao, Q. Lin, X. Hu, and Q. Wang, “Inductive
relation prediction with logical reasoning using contrastive
representations,” in Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing . Abu Dhabi, United Arab Emirates: Association for Computational
Linguistics, Dec. 2022, pp. 4261–4274. [Online]. Available:
https://aclanthology.org/2022.emnlp-main.286

 

 
 [87] 
 
J. Li, Q. Wang, and Z. Mao, “Inductive relation prediction from relational
paths and context with hierarchical transformers,” in ICASSP 2023 -
2023 IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP) , 2023, pp. 1–5.

 

 
 [88] 
 
C. Fu, T. Chen, M. Qu, W. Jin, and X. Ren, “Collaborative policy learning for
open knowledge graph reasoning,” arXiv preprint arXiv:1909.00230 ,
2019.

 

 
 [89] 
 
W. Zhang, B. Paudel, L. Wang, J. Chen, H. Zhu, W. Zhang, A. Bernstein, and
H. Chen, “Iteratively learning embeddings and rules for knowledge graph
reasoning,” in Proc. of WWW , 2019.

 

 
 [90] 
 
M. Qu and J. Tang, “Probabilistic logic neural networks for reasoning,”
 Proc. of NeurIPS , 2019.

 

 
 [91] 
 
A. Sadeghian, M. Armandpour, P. Ding, and D. Z. Wang, “Drum: End-to-end
differentiable rule mining on knowledge graphs,” Proc. of NeurIPS ,
2019.

 

 
 [92] 
 
P. G. Omran, K. Wang, and Z. Wang, “An embedding-based approach to rule
learning in knowledge graphs,” IEEE Transactions on Knowledge and Data
Engineering , 2019.

 

 
 [93] 
 
P.-W. Wang, D. Stepanova, C. Domokos, and J. Z. Kolter, “Differentiable
learning of numerical rules in knowledge graphs,” in Proc. of ICLR ,
2019.

 

 
 [94] 
 
D. Zhang, Z. Yuan, H. Liu, H. Xiong et al. , “Learning to walk with dual
agents for knowledge graph reasoning,” in Proc. of AAAI , 2022.

 

 
 [95] 
 
H. Chen, Y. Li, S. Shi, S. Liu, H. Zhu, and Y. Zhang, “Graph collaborative
reasoning,” in Proc. of WSDM , 2022.

 

 
 [96] 
 
Y. Shen, J. Chen, P.-S. Huang, Y. Guo, and J. Gao, “M-walk: Learning to walk
in graph with monte carlo tree search,” in NIPS 2018 , 2018.

 

 
 [97] 
 
X. V. Lin, R. Socher, and C. Xiong, “Multi-hop knowledge graph reasoning with
reward shaping,” arXiv preprint arXiv:1808.10568 , 2018.

 

 
 [98] 
 
W. Chen, W. Xiong, X. Yan, and W. Wang, “Variational knowledge graph
reasoning,” arXiv preprint arXiv:1803.06581 , 2018.

 

 
 [99] 
 
C. Meilicke, M. Fink, Y. Wang, D. Ruffinelli, R. Gemulla, and
H. Stuckenschmidt, “Fine-grained evaluation of rule-and embedding-based
systems for knowledge graph completion,” in Proc. of ISWC , 2018.

 

 
 [100] 
 
S. Guo, Q. Wang, L. Wang, B. Wang, and L. Guo, “Knowledge graph embedding with
iterative guidance from soft rules,” in Proc. of AAAI , 2018.

 

 
 [101] 
 
R. Das, S. Dhuliawala, M. Zaheer, L. Vilnis, I. Durugkar, A. Krishnamurthy,
A. Smola, and A. McCallum, “Go for a walk and arrive at the answer:
Reasoning over paths in knowledge bases using reinforcement learning,”
 arXiv preprint arXiv:1711.05851 , 2017.

 

 
 [102] 
 
W. Xiong, T. Hoang, and W. Y. Wang, “Deeppath: A reinforcement learning method
for knowledge graph reasoning,” arXiv preprint arXiv:1707.06690 ,
2017.

 

 
 [103] 
 
T. Rocktäschel and S. Riedel, “End-to-end differentiable proving,”
 Proc. of NeurIPS , 2017.

 

 
 [104] 
 
F. Yang, Z. Yang, and W. W. Cohen, “Differentiable learning of logical rules
for knowledge base reasoning,” Proc. of NeurIPS , 2017.

 

 
 [105] 
 
R. Das, A. Neelakantan, D. Belanger, and A. McCallum, “Chains of reasoning
over entities, relations, and text using recurrent neural networks,”
 arXiv preprint arXiv:1607.01426 , 2016.

 

 
 [106] 
 
S. Guo, Q. Wang, L. Wang, B. Wang, and L. Guo, “Jointly embedding knowledge
graphs and logical rules,” in Proc. of EMNLP , 2016.

 

 
 [107] 
 
Y. Zhang, X. Chen, Y. Yang, A. Ramamurthy, B. Li, Y. Qi, and L. Song,
“Efficient probabilistic logic reasoning with graph neural networks,”
 arXiv preprint arXiv:2001.11850 , 2020.

 

 
 [108] 
 
A. Neelakantan, B. Roth, and A. McCallum, “Compositional vector space models
for knowledge base completion,” arXiv preprint arXiv:1504.06662 ,
2015.

 

 
 [109] 
 
M. Gardner, P. Talukdar, J. Krishnamurthy, and T. Mitchell, “Incorporating
vector space similarity in random walk inference over knowledge bases,” in
 Proc. of EMNLP , 2014.

 

 
 [110] 
 
L. A. Galárraga, C. Teflioudi, K. Hose, and F. Suchanek, “Amie:
association rule mining under incomplete evidence in ontological knowledge
bases,” in Proc. of WWW , 2013.

 

 
 [111] 
 
N. Lao and W. W. Cohen, “Relational retrieval using a combination of
path-constrained random walks,” Machine learning , 2010.

 

 
 [112] 
 
Q. Lin, J. Liu, F. Xu, Y. Pan, Y. Zhu, L. Zhang, and T. Zhao, “Incorporating
context graph with logical reasoning for inductive relation prediction,”
ser. SIGIR ’22. New York, NY, USA:
Association for Computing Machinery, 2022, p. 893–903. [Online]. Available:
https://doi-org-s.libyc.nudt.edu.cn:443/10.1145/3477495.3531996

 

 
 [113] 
 
G. F. Lawler and V. Limic, Random walk: a modern introduction . Cambridge University Press, 2010.

 

 
 [114] 
 
K. Liu, F. Zhao, G. Xu, X. Wang, and H. Jin, “Retia: relation-entity
twin-interact aggregation for temporal knowledge graph extrapolation,” in
 IEEE International Conference on Data Engineering . IEEE, 2023.

 

 
 [115] 
 
Z. Han, P. Chen, Y. Ma, and V. Tresp, “Explainable subgraph reasoning for
forecasting on temporal knowledge graphs,” in Proc. of ICLR , 2020.

 

 
 [116] 
 
K. Liang, L. Meng, M. Liu, Y. Liu, W. Tu, S. Wang, S. Zhou, and X. Liu, “Learn
from relational correlations and periodic events for temporal knowledge graph
reasoning,” in Proceedings of the 46th International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR
’23) , 2023.

 

 
 [117] 
 
C. Zhu, M. Chen, C. Fan, G. Cheng, and Y. Zhang, “Learning from history:
Modeling temporal knowledge graphs with sequential copy-generation
networks,” in Proc. of AAAI , 2021.

 

 
 [118] 
 
Y. Xu, J. Ou, H. Xu, and L. Fu, “Temporal knowledge graph reasoning with
historical contrastive learning,” in Proc. of AAAI , 2022.

 

 
 [119] 
 
J. Wu, Y. Xu, Y. Zhang, C. Ma, M. Coates, and J. C. K. Cheung, “Tie: A
framework for embedding-based incremental temporal knowledge graph
completion,” in Proc. of SIGIR , 2021.

 

 
 [120] 
 
K. Liu, F. Zhao, H. Chen, Y. Li, G. Xu, and H. Jin, “Da-net: Distributed
attention network for temporal knowledge graph reasoning,” in Proc. of
CIKM , 2022.

 

 
 [121] 
 
C. Xu, Y.-Y. Chen, M. Nayyeri, and J. Lehmann, “Temporal knowledge graph
completion using a linear temporal regularizer and multivector embeddings,”
in Proc. of AACL , 2021.

 

 
 [122] 
 
Z. Li, Z. Hou, S. Guan, X. Jin, W. Peng, L. Bai, Y. Lyu, W. Li, J. Guo, and
X. Cheng, “Hismatch: Historical structure matching based temporal knowledge
graph reasoning,” arXiv preprint arXiv:2210.09708 , 2022.

 

 
 [123] 
 
A. Sadeghian, M. Armandpour, A. Colas, and D. Z. Wang, “Chronor: rotation
based temporal knowledge graph embedding,” in Proc. of AAAI , 2021.

 

 
 [124] 
 
Y. Gao, L. Feng, Z. Kan, Y. Han, L. Qiao, and D. Li, “Modeling precursors for
temporal knowledge graph reasoning via auto-encoder structure,” in
 Proc. of IJCAI , 2022.

 

 
 [125] 
 
Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y. Wang, and X. Cheng,
“Temporal knowledge graph reasoning based on evolutional representation
learning,” in Proc. of SIGIR , 2021.

 

 
 [126] 
 
R. Wang, Z. Li, D. Sun, S. Liu, J. Li, B. Yin, and T. Abdelzaher, “Learning to
sample and aggregate: Few-shot reasoning over temporal knowledge graphs,”
 arXiv preprint arXiv:2210.08654 , 2022.

 

 
 [127] 
 
Y. Xu, E. Haihong, M. Song, W. Song, X. Lv, W. Haotian, and Y. Jinrui, “Rtfe:
A recursive temporal fact embedding framework for temporal knowledge graph
completion,” in Proc. of AACL , 2021.

 

 
 [128] 
 
Z. Ding, J. Wu, B. He, Y. Ma, Z. Han, and V. Tresp, “Few-shot inductive
learning on temporal knowledge graphs using concept-aware information,”
 AKBC , 2022.

 

 
 [129] 
 
Y. He, P. Zhang, L. Liu, Q. Liang, W. Zhang, and C. Zhang, “Hip network:
Historical information passing network for extrapolation reasoning on
temporal knowledge graph.” in Proc. of IJCAI , 2021.

 

 
 [130] 
 
L. Zhang and D. Zhou, “Temporal knowledge graph completion with approximated
gaussian process embedding,” in Proc. of COLING , 2022.

 

 
 [131] 
 
L. Bai, W. Yu, M. Chen, and X. Ma, “Multi-hop reasoning over paths in temporal
knowledge graphs using reinforcement learning,” Applied Soft
Computing , 2021.

 

 
 [132] 
 
Y. Liu, Y. Ma, M. Hildebrandt, M. Joblin, and V. Tresp, “Tlogic: Temporal
logical rules for explainable link forecasting on temporal knowledge
graphs,” in Proc. of AAAI , 2022.

 

 
 [133] 
 
P. Jain, S. Rathi, S. Chakrabarti et al. , “Temporal knowledge base
completion: New algorithms and evaluation protocols,” arXiv preprint
arXiv:2005.05035 , 2020.

 

 
 [134] 
 
F. Zhang, Z. Zhang, X. Ao, F. Zhuang, Y. Xu, and Q. He, “Along the time:
Timeline-traced embedding for temporal knowledge graph completion,” in
 Proc. of CIKM , 2022.

 

 
 [135] 
 
Z. Han, P. Chen, Y. Ma, and V. Tresp, “Dyernie: Dynamic evolution of
riemannian manifold embeddings for temporal knowledge graph completion,” in
 Proc. of EMNLP , 2020.

 

 
 [136] 
 
Z. Li, S. Guan, X. Jin, W. Peng, Y. Lyu, Y. Zhu, L. Bai, W. Li, J. Guo, and
X. Cheng, “Complex evolutional pattern learning for temporal knowledge graph
reasoning,” in Proc. of ACL , 2022.

 

 
 [137] 
 
X. Lv, X. Han, L. Hou, J. Li, Z. Liu, W. Zhang, Y. Zhang, H. Kong, and S. Wu,
“Dynamic anticipation and completion for multi-hop reasoning over sparse
knowledge graph,” in Proc. of EMNLP , 2020.

 

 
 [138] 
 
J. Messner, R. Abboud, and I. I. Ceylan, “Temporal knowledge graph completion
using box embeddings,” in Proc. of AAAI , 2022.

 

 
 [139] 
 
T. Lacroix, G. Obozinski, and N. Usunier, “Tensor decompositions for temporal
knowledge base completion,” in Proc. of ICLR , 2019.

 

 
 [140] 
 
C. Mavromatis, P. L. Subramanyam, V. N. Ioannidis, A. Adeshina, P. R. Howard,
T. Grinberg, N. Hakim, and G. Karypis, “Tempoqr: temporal question reasoning
over knowledge graphs,” in Proc. of AAAI , 2022.

 

 
 [141] 
 
P. Shao, D. Zhang, G. Yang, J. Tao, F. Che, and T. Liu, “Tucker
decomposition-based temporal knowledge graph completion,”
 Knowledge-Based Systems , 2022.

 

 
 [142] 
 
L. Qu, H. Zhu, Q. Duan, and Y. Shi, “Continuous-time link prediction via
temporal dependent graph neural network,” in Proc. of WWW , 2020.

 

 
 [143] 
 
H. Sun, S. Geng, J. Zhong, H. Hu, and K. He, “Graph Hawkes transformer for
extrapolated reasoning on temporal knowledge graphs,” in Proceedings
of the 2022 Conference on Empirical Methods in Natural Language
Processing . Abu Dhabi, United Arab
Emirates: Association for Computational Linguistics, Dec. 2022, pp.
7481–7493. [Online]. Available:
https://aclanthology.org/2022.emnlp-main.507

 

 
 [144] 
 
C. Xu, M. Nayyeri, F. Alkhoury, H. Yazdi, and J. Lehmann, “Temporal knowledge
graph completion based on time series gaussian embedding,” in Proc. of
ISWC , 2020.

 

 
 [145] 
 
T. Wu, A. Khan, M. Yong, G. Qi, and M. Wang, “Efficiently embedding dynamic
knowledge graphs,” Knowledge-Based Systems , 2022.

 

 
 [146] 
 
R. Goel, S. M. Kazemi, M. Brubaker, and P. Poupart, “Diachronic embedding for
temporal knowledge graph completion,” in Proc. of AAAI , 2020.

 

 
 [147] 
 
“Tirgn: Time-guided recurrent graph network with local-global historical
patterns for temporal knowledge graph reasoning.”

 

 
 [148] 
 
K. Chen, Y. Wang, Y. Li, and A. Li, “Rotateqvs: Representing temporal
information as rotations in quaternion vector space for temporal knowledge
graph completion,” in Proc. of ACL , 2022.

 

 
 [149] 
 
C. Xu, M. Nayyeri, F. Alkhoury, H. S. Yazdi, and J. Lehmann, “Tero: A
time-aware knowledge graph embedding via temporal rotation,” in Proc.
of COLING , 2020.

 

 
 [150] 
 
C. Yan, F. Zhao, and H. Jin, “Exkgr: Explainable multi-hop reasoning for
evolving knowledge graph,” in Proc. of DASFAA , 2022.

 

 
 [151] 
 
A. Pareja, G. Domeniconi, J. Chen, T. Ma, T. Suzumura, H. Kanezashi, T. Kaler,
T. Schardl, and C. Leiserson, “Evolvegcn: Evolving graph convolutional
networks for dynamic graphs,” in Proc. of AAAI , 2020.

 

 
 [152] 
 
L. Yuan, Z. Li, J. Qu, T. Zhang, A. Liu, L. Zhao, and Z. Chen, “Trhyte:
Temporal knowledge graph embedding based on temporal-relational
hyperplanes,” in Proc. of DASFAA , 2022.

 

 
 [153] 
 
J. Wu, M. Cao, J. C. K. Cheung, and W. L. Hamilton, “Temp: Temporal message
passing for temporal knowledge graph completion,” in Proc. of EMNLP ,
2020.

 

 
 [154] 
 
N. Park, F. Liu, P. Mehta, D. Cristofor, C. Faloutsos, and Y. Dong, “Evokg:
Jointly modeling event time and network structure for reasoning over temporal
knowledge graphs,” in Proc. of WSDM , 2022.

 

 
 [155] 
 
W. Jin, M. Qu, X. Jin, and X. Ren, “Recurrent event network: Autoregressive
structure inferenceover temporal knowledge graphs,” in Proc. of
EMNLP , 2020.

 

 
 [156] 
 
L. Bai, X. Ma, M. Zhang, and W. Yu, “Tpmod: A tendency-guided prediction model
for temporal knowledge graph completion,” ACM Transactions on
Knowledge Discovery from Data , 2021.

 

 
 [157] 
 
R. Trivedi, M. Farajtabar, P. Biswal, and H. Zha, “Dyrep: Learning
representations over dynamic graphs,” in Proc. of ICLR , 2019.

 

 
 [158] 
 
H. Sun, J. Zhong, Y. Ma, Z. Han, and K. He, “Timetraveler: Reinforcement
learning for temporal knowledge graph forecasting,” in Proc. of
EMNLP , 2021.

 

 
 [159] 
 
A. Garcia-Duran, S. Dumancic, and M. Niepert, “Learning sequence encoders for
temporal knowledge graph completion,” in EMNLP , 2018.

 

 
 [160] 
 
Z. Li, X. Jin, S. Guan, W. Li, J. Guo, Y. Wang, and X. Cheng, “Search from
history and reason for future: Two-stage reasoning on temporal knowledge
graphs,” in ACL , 2021.

 

 
 [161] 
 
S. S. Dasgupta, S. N. Ray, and P. Talukdar, “Hyte: Hyperplane-based temporally
aware knowledge graph embedding,” in Proc. of EMNLP , 2018.

 

 
 [162] 
 
Y. Zhao, X. Wang, J. Chen, Y. Wang, W. Tang, X. He, and H. Xie, “Time-aware
path reasoning on knowledge graph for recommendation,” ACM
Transactions on Information Systems (TOIS) , 2021.

 

 
 [163] 
 
A. Sadeghian, M. Rodriguez, D. Z. Wang, and A. Colas, “Temporal reasoning over
event knowledge graphs,” in Workshop on Knowledge Base Construction,
Reasoning and Mining , 2016.

 

 
 [164] 
 
S. Liao, S. Liang, Z. Meng, and Q. Zhang, “Learning dynamic embeddings for
temporal knowledge graphs,” in Proc. of WSDM , 2021.

 

 
 [165] 
 
A. Garca-Duran, S. Dumancic, and M. Niepert, “Learning sequence encoders for
temporal knowledge graph completion,” in EMNLP , 2018.

 

 
 [166] 
 
Z. Han, Z. Ding, Y. Ma, Y. Gu, and V. Tresp, “Learning neural ordinary
equations for forecasting future links on temporal knowledge graphs,” in
 Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing , 2021, pp. 8352–8364.

 

 
 [167] 
 
J. Jung, J. Jung, and U. Kang, “Learning to walk across time for interpretable
temporal knowledge graph completion,” in Proc. of KDD , 2021.

 

 
 [168] 
 
R. Trivedi, H. Dai, Y. Wang, and L. Song, “Know-evolve: Deep temporal
reasoning for dynamic knowledge graphs,” in Proc. of ICML , 2017.

 

 
 [169] 
 
Y. Seo, M. Defferrard, P. Vandergheynst, and X. Bresson, “Structured sequence
modeling with graph convolutional recurrent networks,” in
 International conference on neural information processing . Springer, 2018, pp. 362–373.

 

 
 [170] 
 
L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang, “Visualbert: A
simple and performant baseline for vision and language,” arXiv
preprint arXiv:1908.03557 , 2019.

 

 
 [171] 
 
W. Su, X. Zhu, Y. Cao, B. Li, L. Lu, F. Wei, and J. Dai, “Vl-bert:
Pre-training of generic visual-linguistic representations,” arXiv
preprint arXiv:1908.08530 , 2019.

 

 
 [172] 
 
Y. Zhang and W. Zhang, “Knowledge graph completion with pre-trained multimodal
transformer and twins negative sampling,” arXiv preprint
arXiv:2209.07084 , 2022.

 

 
 [173] 
 
X. Pan, T. Ye, D. Han, S. Song, and G. Huang, “Contrastive language-image
pre-training with knowledge graphs,” in Proc. of NeurIPS .

 

 
 [174] 
 
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry,
A. Askell, P. Mishkin, J. Clark et al. , “Learning transferable visual
models from natural language supervision,” in Proc. of ICML , 2021.

 

 
 [175] 
 
N. Zhang, L. Li, X. Chen, X. Liang, S. Deng, and H. Chen, “Multimodal
analogical reasoning over knowledge graphs,” arXiv preprint
arXiv:2210.00312 , 2022.

 

 
 [176] 
 
X. Chen, N. Zhang, L. Li, S. Deng, C. Tan, C. Xu, F. Huang, L. Si, and H. Chen,
“Hybrid transformer with multi-level fusion for multimodal knowledge graph
completion,” in Proc. of SIGIR , 2022.

 

 
 [177] 
 
S. Liang, A. Zhu, J. Zhang, and J. Shao, “Hyper-node relational graph
attention network for multi-modal knowledge graph completion,” vol. 19,
no. 2, feb 2023. [Online]. Available: https://doi.org/10.1145/3545573

 

 
 [178] 
 
L. Chen, Z. Li, T. Xu, H. Wu, Z. Wang, N. J. Yuan, and E. Chen, “Multi-modal
siamese network for entity alignment,” in Proceedings of the 28th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining , ser. KDD
’22. New York, NY, USA: Association
for Computing Machinery, 2022, p. 118–126. [Online]. Available:
https://doi.org/10.1145/3534678.3539244

 

 
 [179] 
 
X. Li, X. Zhao, J. Xu, Y. Zhang, and C. Xing, “Imf: Interactive multimodal
fusion model for link prediction,” in Proceedings of the ACM Web
Conference 2023 , ser. WWW ’23. New
York, NY, USA: Association for Computing Machinery, 2023, p. 2572–2580.
[Online]. Available: https://doi.org/10.1145/3543507.3583554

 

 
 [180] 
 
Y. Ding, J. Yu, B. Liu, Y. Hu, M. Cui, and Q. Wug, “Mukea: Multimodal
knowledge extraction and accumulation for knowledge-based visual question
answering,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) , 2022.

 

 
 [181] 
 
M. Yasunaga, A. Bosselut, H. Ren, X. Zhang, C. D. Manning, P. Liang, and
J. Leskovec, “Deep bidirectional language-knowledge graph pretraining,” in
 Proc. of NeurIPS .

 

 
 [182] 
 
F. Zhang, N. J. Yuan, D. Lian, X. Xie, and W.-Y. Ma, “Collaborative knowledge
base embedding for recommender systems,” in Proc. of KDD , 2016.

 

 
 [183] 
 
R. Xie, Z. Liu, J. Jia, H. Luan, and M. Sun, “Representation learning of
knowledge graphs with entity descriptions,” in Proc. of AAAI , 2016.

 

 
 [184] 
 
R. Xie, Z. Liu, H. Luan, and M. Sun, “Image-embodied knowledge representation
learning,” in Proc. of IJCAI , 2017.

 

 
 [185] 
 
Z. Wang, L. Li, Q. Li, and D. Zeng, “Multimodal data enhanced representation
learning for knowledge graphs,” in Proc. of IJCNN , 2019.

 

 
 [186] 
 
A. Garcia-Duran and M. Niepert, “Kblrn: End-to-end learning of knowledge base
representations with latent, relational, and numerical features,”
 arXiv e-prints , 2017.

 

 
 [187] 
 
Y. Zuo, Q. Fang, S. Qian, X. Zhang, and C. Xu, “Representation learning of
knowledge graphs with entity attributes and multimedia descriptions,” in
 2018 IEEE Fourth International Conference on Multimedia Big Data
(BigMM) , 2018.

 

 
 [188] 
 
X. Tang, L. Chen, J. Cui, and B. Wei, “Knowledge representation learning with
entity descriptions, hierarchical types, and textual relations,”
 Information Processing Management , 2019.

 

 
 [189] 
 
H. Mousselly-Sergieh, T. Botschen, I. Gurevych, and S. Roth, “A multimodal
translation-based approach for knowledge graph representation learning,” in
 Proceedings of the Seventh Joint Conference on Lexical and
Computational Semantics , 2018.

 

 
 [190] 
 
P. Pezeshkpour, L. Chen, and S. Singh, “Embedding multimodal relational data
for knowledge base completion,” in Proc. of EMNLP , 2018.

 

 
 [191] 
 
W. Wilcke, P. Bloem, V. de Boer, R. van t Veer, and F. van Harmelen,
“End-to-end entity classification on multimodal knowledge graphs,”
 arXiv preprint arXiv:2003.12383 , 2020.

 

 
 [192] 
 
S. Zheng, W. Wang, J. Qu, H. Yin, W. Chen, and L. Zhao, “Mmkgr: Multi-hop
multi-modal knowledge graph reasoning,” arXiv preprint
arXiv:2209.01416 , 2022.

 

 
 [193] 
 
R. Sun, X. Cao, Y. Zhao, J. Wan, K. Zhou, F. Zhang, Z. Wang, and K. Zheng,
“Multi-modal knowledge graphs for recommender systems,” in Proc. of
CIKM , 2020.

 

 
 [194] 
 
P. Wang, Q. Wu, C. Shen, A. v. d. Hengel, and A. Dick, “Explicit
knowledge-based reasoning for visual question answering,” arXiv
preprint arXiv:1511.02570 , 2015.

 

 
 [195] 
 
L. Guo, J. Liu, J. Tang, J. Li, W. Luo, and H. Lu, “Aligning linguistic words
and visual semantic units for image captioning,” in Proceedings of the
27th ACM international conference on multimedia , 2019, pp. 765–773.

 

 
 [196] 
 
S. Shah, A. Mishra, N. Yadati, and P. P. Talukdar, “Kvqa: Knowledge-aware
visual question answering,” in Proc. of AAAI , 2019.

 

 
 [197] 
 
L. Chen, Z. Li, Y. Wang, T. Xu, Z. Wang, and E. Chen, “Mmea: Entity alignment
for multi-modal knowledge graph,” in International Conference on
Knowledge Science, Engineering and Management . Springer, 2020, pp. 134–147.

 

 
 [198] 
 
Y. Zhao, X. Cai, Y. Wu, H. Zhang, Y. Zhang, G. Zhao, and N. Jiang, “MoSE:
Modality split and ensemble for multimodal knowledge graph completion,” in
 Proceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing . Abu Dhabi,
United Arab Emirates: Association for Computational Linguistics, Dec. 2022,
pp. 10 527–10 536. [Online]. Available:
https://aclanthology.org/2022.emnlp-main.719

 

 
 [199] 
 
M. Wang, S. Wang, H. Yang, Z. Zhang, X. Chen, and G. Qi, “Is visual context
really helpful for knowledge graph? a representation learning perspective,”
in Proceedings of the 29th ACM International Conference on Multimedia ,
ser. MM ’21. New York, NY, USA:
Association for Computing Machinery, 2021, p. 2735–2743. [Online].
Available: https://doi.org/10.1145/3474085.3475470

 

 
 [200] 
 
H. Guo, J. Tang, W. Zeng, X. Zhao, and L. Liu, “Multi-modal entity alignment
in hyperbolic space,” Neurocomputing , vol. 461, pp. 598–607, 2021.

 

 
 [201] 
 
Z. Cao, Q. Xu, Z. Yang, Y. He, X. Cao, and Q. Huang, “Otkge: Multi-modal
knowledge graph embeddings via optimal transport,” Advances in Neural
Information Processing Systems , vol. 35, pp. 39 090–39 102, 2022.

 

 
 [202] 
 
D. Xu, T. Xu, S. Wu, J. Zhou, and E. Chen, “Relation-enhanced negative
sampling for multimodal knowledge graph completion,” in Proc. of ACM
MM , 2022.

 

 
 [203] 
 
X. Cao, Y. Shi, J. Wang, H. Yu, X. Wang, and Z. Yan, “Cross-modal knowledge
graph contrastive learning for machine learning method recommendation,” in
 Proc. of ACM MM , 2022.

 

 
 [204] 
 
X. Lu, L. Wang, Z. Jiang, S. He, and S. Liu, “Mmkrl: A robust embedding
approach for multi-modal knowledge graph representation learning,” vol. 52,
no. 7, p. 7480–7497, may 2022. [Online]. Available:
https://doi.org/10.1007/s10489-021-02693-9

 

 
 [205] 
 
M. Sap, R. Le Bras, E. Allaway, C. Bhagavatula, N. Lourie, H. Rashkin, B. Roof,
N. A. Smith, and Y. Choi, “Atomic: An atlas of machine commonsense for
if-then reasoning,” in Proc. of AAAI , 2019.

 

 
 [206] 
 
G. Bouchard, S. Singh, and T. Trouillon, “On approximate reasoning
capabilities of low-rank vector spaces,” in Proc. of AAAI , 2015.

 

 
 [207] 
 
T. Safavi and D. Koutra, “Codex: A comprehensive knowledge graph completion
benchmark,” in Proc. of EMNLP , 2020.

 

 
 [208] 
 
R. Speer, J. Chin, and C. Havasi, “Conceptnet 5.5: An open multilingual graph
of general knowledge,” in Proc. of AAAI , 2017.

 

 
 [209] 
 
J. Guo and S. Kok, “Bique: Biquaternionic embeddings of knowledge graphs,” in
 Proc. of EMNLP , 2021.

 

 
 [210] 
 
B. Shi and T. Weninger, “Open-world knowledge graph completion,” in
 Proc. of AAAI , 2018.

 

 
 [211] 
 
B. Ding, Q. Wang, B. Wang, and L. Guo, “Improving knowledge graph embedding
using simple constraints,” in Proc. of ACL , 2018.

 

 
 [212] 
 
S. Kok and P. Domingos, “Statistical predicate invention,” in Proc. of
ICML , 2007.

 

 
 [213] 
 
R. Socher, D. Chen, C. D. Manning, and A. Ng, “Reasoning with neural tensor
networks for knowledge base completion,” Proc. of NeurIPS , 2013.

 

 
 [214] 
 
S. Guo, Q. Wang, L. Wang, B. Wang, and L. Guo, “Jointly embedding knowledge
graphs and logical rules,” in Proc. of EMNLP , 2016.

 

 
 [215] 
 
A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko,
“Translating embeddings for modeling multi-relational data,” Proc. of
NeurIPS , 2013.

 

 
 [216] 
 
Y. Lin, Z. Liu, and M. Sun, “Knowledge representation learning with entities,
attributes and relations,” in Proc. of IJCAI , 2016.

 

 
 [217] 
 
K. Toutanova and D. Chen, “Observed versus latent features for knowledge base
and text inference,” in Proceedings of the 3rd workshop on continuous
vector space models and their compositionality , 2015.

 

 
 [218] 
 
C. Fu, T. Chen, M. Qu, W. Jin, and X. Ren, “Collaborative policy learning for
open knowledge graph reasoning,” in Proc. of EMNLP , 2019.

 

 
 [219] 
 
D. S. Himmelstein, A. Lizee, C. Hessler, L. Brueggeman, S. L. Chen, D. Hadley,
A. Green, P. Khankhanian, and S. E. Baranzini, “Systematic integration of
biomedical knowledge prioritizes drugs for repurposing,” eLife , 2017.

 

 
 [220] 
 
S. Guo, Q. Wang, B. Wang, L. Wang, and L. Guo, “Semantically smooth knowledge
graph embedding,” in Proc. of ACL , 2015.

 

 
 [221] 
 
R. J. Rummel, “Dimensionality of nations project,” Tech. Rep., 1968.

 

 
 [222] 
 
X. Lv, X. Han, L. Hou, J. Li, Z. Liu, W. Zhang, Y. Zhang, H. Kong, and S. Wu,
“Dynamic anticipation and completion for multi-hop reasoning over sparse
knowledge graph,” in Proc. of EMNLP , 2020.

 

 
 [223] 
 
W. Xiong, T. Hoang, and W. Y. Wang, “Deeppath: A reinforcement learning method
for knowledge graph reasoning,” in EMNLP , 2017.

 

 
 [224] 
 
A. Breit, S. Ott, A. Agibetov, and M. Samwald, “OpenBioLink: a benchmarking
framework for large-scale biomedical link prediction,”
 Bioinformatics , 2020.

 

 
 [225] 
 
S. Broscheit, D. Ruffinelli, A. Kochsiek, P. Betz, and R. Gemulla, “LibKGE
- A knowledge graph embedding library for reproducible research,” in
 Proc. of EMNLP , 2020.

 

 
 [226] 
 
A. T. McCray, “An upper-level ontology for the biomedical domain,”
 Comparative and functional genomics , 2003.

 

 
 [227] 
 
K. Toutanova and D. Chen, “Observed versus latent features for knowledge base
and text inference,” in Proceedings of the 3rd workshop on continuous
vector space models and their compositionality , 2015.

 

 
 [228] 
 
X. Wang, T. Gao, Z. Zhu, Z. Zhang, Z. Liu, J. Li, and J. Tang, “Kepler: A
unified model for knowledge embedding and pre-trained language
representation,” Transactions of the Association for Computational
Linguistics , 2021.

 

 
 [229] 
 
F. Mahdisoltani, J. Biega, and F. M. Suchanek, “A knowledge base from
multilingual wikipedias–yago3,” Tech. Rep., 2014.

 

 
 [230] 
 
S. Guo, Q. Wang, L. Wang, B. Wang, and L. Guo, “Knowledge graph embedding with
iterative guidance from soft rules,” in Proc. of AAAI , 2018.

 

 
 [231] 
 
X. Lv, L. Hou, J. Li, and Z. Liu, “Differentiating concepts and instances for
knowledge graph embedding,” in Proc. of EMNLP , 2018.

 

 
 [232] 
 
Y. Cui, Y. Wang, Z. Sun, W. Liu, Y. Jiang, K. Han, and W. Hu, “Inductive
knowledge graph reasoning for multi-batch emerging entities,” in Proc.
of CIKM , 2022.

 

 
 [233] 
 
S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, and Z. G. Ives,
“Dbpedia: A nucleus for a web of open data,” in ISWC/ASWC , 2007.

 

 
 [234] 
 
K. D. Bollacker, C. Evans, P. K. Paritosh, T. Sturge, and J. Taylor,
“Freebase: a collaboratively created graph database for structuring human
knowledge,” in SIGMOD Conference , 2008.

 

 
 [235] 
 
W. W. Denham, “The detection of patterns in alyawara nonverbal behavior,”
Ph.D. dissertation, 1973.

 

 
 [236] 
 
A. Carlson, J. Betteridge, B. Kisiel, B. Settles, E. R. Hruschka, and T. M.
Mitchell, “Toward an architecture for never-ending language learning,” in
 Proc. of AAAI , 2010.

 

 
 [237] 
 
O. Bodenreider, “The unified medical language system (umls): integrating
biomedical terminology,” Nucleic acids research , 2004.

 

 
 [238] 
 
G. A. Miller, WordNet: An electronic lexical database . MIT press, 1998.

 

 
 [239] 
 
D. Vrandečić and M. Krötzsch, “Wikidata: a free collaborative
knowledgebase,” Communications of the ACM , 2014.

 

 
 [240] 
 
M. Fabian, K. Gjergji, W. Gerhard et al. , “Yago: A core of semantic
knowledge unifying wordnet and wikipedia,” in Proc. of WWW , 2007.

 

 
 [241] 
 
T. Wu, A. Khan, M. Yong, G. Qi, and M. Wang, “Efficiently embedding dynamic
knowledge graphs,” Knowledge-Based Systems , 2022.

 

 
 [242] 
 
Z. Han, G. Zhang, Y. Ma, and V. Tresp, “Time-dependent entity embedding is not
all you need: A re-evaluation of temporal knowledge graph completion models
under a unified framework,” in Proc. of EMNLP , 2021.

 

 
 [243] 
 
A. García-Durán, S. Dumancic, and M. Niepert, “Learning sequence
encoders for temporal knowledge graph completion,” in EMNLP , 2018.

 

 
 [244] 
 
[Online]. Available: https://www.imdb.com/interfaces/

 

 
 [245] 
 
E. Boschee, J. Lautenschlager, S. O’Brien, S. Shellman, J. Starz, and M. Ward,
“ICEWS Coded Event Data,” 2015. [Online]. Available:
https://doi.org/10.7910/DVN/28075

 

 
 [246] 
 
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A
large-scale hierarchical image database,” in Proc. of CVPR , 2009.

 

 
 [247] 
 
S. Ferrada, B. Bustos, and A. Hogan, “Imgpedia: a linked dataset with
content-based analysis of wikimedia images,” in Proc. of ISWC , 2017.

 

 
 [248] 
 
Z. Sun, Q. Zhang, W. Hu, C. Wang, M. Chen, F. Akrami, and C. Li, “A
benchmarking study of embedding-based entity alignment for knowledge
graphs,” Proceedings of the VLDB Endowment , 2020.

 

 
 [249] 
 
Y. Liu, H. Li, A. Garcia-Duran, M. Niepert, D. Onoro-Rubio, and D. S.
Rosenblum, “Mmkg: multi-modal knowledge graphs,” in European Semantic
Web Conference , 2019.

 

 
 [250] 
 
M. Wang, G. Qi, H. Wang, and Q. Zheng, “Richpedia a comprehensive multi-modal
knowledge graph,” in Joint International Semantic Technology
Conference , 2020.

 

 
 [251] 
 
Y. Zhang, Z. Zhou, Q. Yao, X. Chu, and B. Han, “Learning adaptive propagation
for knowledge graph reasoning,” 2022.

 

 
 [252] 
 
Y. Liu, J. Xia, S. Zhou, S. Wang, X. Guo, X. Yang, K. Liang, W. Tu, Z. S. Li,
and X. Liu, “A survey of deep graph clustering: Taxonomy, challenge, and
application,” arXiv preprint arXiv:2211.12875 , 2022.

 

 
 [253] 
 
Y. Yang, Z. Guan, Z. Wang, W. Zhao, C. Xu, W. Lu, and J. Huang,
“Self-supervised heterogeneous graph pre-training based on structural
clustering,” arXiv preprint arXiv:2210.10462 , 2022.

 

 
 [254] 
 
X. Yang, Y. Liu, S. Zhou, S. Wang, X. Liu, and E. Zhu, “Contrastive deep graph
clustering with learnable augmentation,” arXiv preprint
arXiv:2212.03559 , 2022.

 

 
 [255] 
 
F. Xia, K. Sun, S. Yu, A. Aziz, L. Wan, S. Pan, and H. Liu, “Graph learning: A
survey,” IEEE Transactions on Artificial Intelligence , vol. 2, no. 2,
pp. 109–127, 2021.

 

 
 [256] 
 
J. Dong, Y. Cong, G. Sun, Z. Fang, and Z. Ding, “Where and how to transfer:
knowledge aggregation-induced transferability perception for unsupervised
domain adaptation,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , 2021.

 

 
 [257] 
 
T. Chen, L. Lin, R. Chen, X. Hui, and H. Wu, “Knowledge-guided multi-label
few-shot learning for general image recognition,” IEEE Transactions on
Pattern Analysis and Machine Intelligence , vol. 44, no. 3, pp. 1371–1384,
2020.

 

 
 [258] 
 
Y. Lan, S. He, K. Liu, X. Zeng, S. Liu, and J. Zhao, “Path-based knowledge
reasoning with textual semantic information for medical knowledge graph
completion,” BMC Medical Informatics and Decision Making , 2021.

 

 
 [259] 
 
M. Rotmensch, Y. Halpern, A. Tlimat, S. Horng, and D. Sontag, “Learning a
health knowledge graph from electronic medical records,” Scientific
reports , 2017.

 

 
 [260] 
 
S. Kapetanakis, G. Samakovitis, B. Gunasekara, and M. Petridis, “Monitoring
financial transaction fraud with the use of case-based reasoning,” 2012.

 

 
 [261] 
 
M. Franco-Salvador, P. Gupta, P. Rosso, and R. E. Banchs, “Cross-language
plagiarism detection over continuous-space- and knowledge graph-based
representations of language,” Knowledge-Based Systems , 2016.

 

 
 [262] 
 
Y. Liu, T. Han, S. Ma, J. Zhang, Y. Yang, J. Tian, H. He, A. Li, M. He, Z. Liu
 et al. , “Summary of chatgpt/gpt-4 research and perspective towards
the future of large language models,” arXiv preprint
arXiv:2304.01852 , 2023.

 

 
 [263] 
 
S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, “Unifying large language
models and knowledge graphs: A roadmap,” arXiv preprint
arXiv:2306.08302 , 2023.

 

 
 [264] 
 
M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec, “Qa-gnn:
Reasoning with language models and knowledge graphs for question answering,”
 arXiv preprint arXiv:2104.06378 , 2021.