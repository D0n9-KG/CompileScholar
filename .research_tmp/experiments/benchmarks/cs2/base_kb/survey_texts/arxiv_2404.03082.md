Machine Learning and Data Analysis Using Posets: A Survey 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2404.03082v3 [cs.LG] 10 Aug 2026 
 
 

# Machine Learning and Data Analysis Using Posets: A Survey Thanks: Corresponding author: Arnauld M. Mwafise (arnauld.mesinga@gmail.com) 

 
 
 Arnauld Mesinga Mwafise
 
 Affiliation: arnauld.mesinga@gmail.com 
 

 Abstract 
 
 Partially ordered sets (posets) are discrete mathematical structures that formalize the notion of comparison without forcing every pair of objects to be comparable. This makes them a natural representation for the many machine learning and data-analysis settings in which objects are related by dominance, containment, priority, or refinement relations rather than by a single scalar score. Over the past two decades, a substantial and fragmented literature has connected posets and lattice theory to ranking, clustering, formal concept analysis, multidimensional and multi-criteria data analysis, structured and safe learning, graph and topological deep learning, and explainable artificial intelligence, spanning a wide range of application domains. Despite this activity, the field has lacked (i) an organizing taxonomy that relates these disparate strands of work, and (ii) an up-to-date account extending through 2025–2026 that incorporates recent developments in poset-structured learning methods, including poset-structured safety layers for reinforcement learning, poset-valued neural pooling operators, functor-calculus approaches to multiparameter persistent homology, and order-theoretic formulations of abstract dynamic programming. This survey addresses both gaps. We propose a four-axis taxonomy of poset-based methods (representation, learning paradigm, data modality, and task), provide a comprehensive and comparative review of representative models and algorithms organized along this taxonomy, curate an extensive collection of datasets, software packages, and algorithmic resources, and close with a critical discussion of open theoretical and practical problems – including model depth and expressivity, scalability–fidelity trade-offs, heterogeneity of order-structured data, and dynamicity of posets over time – that we argue define a research agenda for the next generation of order-aware machine learning. 

 
 
 
 Keywords— partially ordered sets, machine learning, data analysis, formal concept analysis, lattice theory, reinforcement learning, deep learning, explainable AI 

 
 

## 1 Introduction

 
 Data analysis plays a pivotal role in today’s data-driven world. Machine learning is used to automate the data analysis process, and enables the extraction of valuable insights by making accurate predictions from large complex datasets. Data often have complex structure which can usually be endowed with a natural order. Such complex data can be derived from text, social and behavioral sciences, chemical structures, biological structures, images, videos, governmental and business settings, and so forth. Mathematical techniques and concepts serve as powerful tools in the development of new algorithms and methods for machine learning and data analysis. For some time now, many studies based on the use of partially ordered sets (posets) for machine learning and data analysis have emerged. Furthermore, over this time the literature on these topics has significantly expanded in volume, scope, and the range of its applications in various fields. Yet this literature is scattered across statistics, discrete mathematics, environmental science, topology, and machine learning venues, using inconsistent terminology and with no organizing framework that connects the underlying methods. This impedes the ability of researchers and machine learning scientists alike to build on prior work or to identify which poset-based tool is appropriate for a given task. In order to address this gap, we present an up-to-date overview with an emphasis on the most prominent and currently relevant works on data analysis and machine learning methods in which the role of posets has been established, including a substantial body of work published in 2025 and 2026 that has not previously been surveyed. By covering earlier works as well as these recent advances, this survey aims to provide researchers, applied mathematicians, data scientists, statisticians, social scientists, and practitioners new to the field, as well as all those aiming to keep pace with innovations in this growing and promising research area, with a solid understanding of the different conceptual and methodological approaches that incorporate posets into data analytics or machine learning applications. 

 
 

### 1.1 Contributions

 
 This survey makes the following contributions. 

 
 
 
 • 
 
 A new taxonomy. No existing survey organizes poset-based machine learning and data-analysis methods along a common set of axes. We propose a four-axis taxonomy (Section 2 ) that classifies work by (i) the mathematical representation of the poset used (order-relational, matrix-based, metric/distance-based, algebraic-categorical), (ii) the learning paradigm it belongs to (supervised, unsupervised, semi-supervised, reinforcement/control, or purely descriptive/statistical), (iii) the data modality it targets (tabular/multi-indicator, text, vision, graph/hypergraph, time series, or topological), and (iv) the task it solves (ranking, classification, clustering, model selection, metric computation, or explanation). This taxonomy is used throughout the survey to situate individual papers relative to one another, including papers that have not previously been discussed in a poset-and-machine-learning context. 

 

 • 
 
 A comprehensive and comparative review. We provide what is, to our knowledge, the most comprehensive review to date of posets in machine learning and data analysis, giving detailed descriptions of representative models, drawing explicit comparisons between competing approaches to the same problem (e.g., alternative ranking, clustering, and metric-learning schemes for partially ordered data), and summarizing the corresponding algorithms and their complexity where this information is available in the primary sources. 

 

 • 
 
 Abundant resources. We collect an extensive set of resources – datasets, open-source software packages, and algorithms – relevant to posets in data analysis and machine learning. The survey is intended to serve as a hands-on reference for understanding, using, and developing data-analysis, machine-learning, and deep-learning approaches for real-world applications involving order-structured data. 

 

 • 
 
 A research agenda. We discuss the theoretical foundations of posets in data analysis and machine learning, analyze the limitations of existing methods, and lay out a concrete research agenda organized around model depth and expressivity, scalability–fidelity trade-offs, heterogeneity of order-structured data, dynamicity of evolving posets, safety and constrained learning, and the emerging connections between posets, category theory, and topological data analysis. 

 

 
 
 
 The rest of this survey is organized as follows. Section 2 introduces the taxonomy that organizes the remainder of the paper. Section 3 introduces and outlines the key concepts of poset theory and lattice theory. Section 4 presents a summary of research studies on machine learning and deep learning with respect to poset theory, lattice theory, and formal concept analysis, including safe and constrained learning, reinforcement learning and dynamic programming, and explainable AI. Section 5 focuses on cluster analysis using posets and formal concept analysis methods. Section 6 presents multidimensional data analysis from a poset-theoretic perspective, and Section 7 gives a unified treatment of its exploratory and descriptive aspects. Section 8 presents a collection of applications across various domains, including software packages, datasets, and algorithms selected for this purpose. Section 9 discusses possible future research directions. Section 10 summarizes the paper. 

 
 
 
 

## 2 A Taxonomy of Poset-Based Methods

 
 To organize the heterogeneous body of work reviewed in this survey, we introduce a taxonomy along four largely orthogonal axes: the mathematical representation used for the poset, the learning paradigm the method belongs to, the data modality it is applied to, and the task it addresses. Table 1 summarizes the taxonomy together with representative sections of this survey and representative references; most methods can be located at the intersection of one entry from each axis. We use this taxonomy in the remainder of the survey to situate each method discussed. 

 
 
 Table 1: A four-axis taxonomy of poset-based methods in machine learning and data analysis. 
 
 
 
 
 
 Axis 
 | 
 
 
 Categories 
 | 
 
 
 Representative sections 
 | 

 
 
 
 
 
 Representation 
 | 
 
 
 Order-relational (Hasse diagrams, cover relations); matrix-based (incidence, cover, mutual-ranking-probability, poset matrices); metric/distance-based (poset metrics, model-oriented graph distances); algebraic/categorical (operads of posets, operads of poset matrices, poset cocalculus, order polytopes) 
 | 
 
 
 § 3 , § 4 , § 9 
 | 

 
 
 
 Learning paradigm 
 | 
 
 
 Unsupervised (clustering, FCA, concept lattices); supervised (classification, learning to rank); semi-supervised; reinforcement learning and control (safe learning, abstract dynamic programming, poset/lattice generation); purely descriptive/statistical (depth functions, multi-indicator ranking) 
 | 
 
 
 § 4 , § 5 , § 6 
 | 

 
 
 
 Data modality 
 | 
 
 
 Tabular/multi-indicator (socio-economic, environmental); text/NLP (semantic parsing, dependency order); vision (pose estimation, image classification); graph/hypergraph (GNNs, partial-order hypergraphs); time series (event sequences); topological (multiparameter persistence) 
 | 
 
 
 § 4 , § 8 
 | 

 
 
 
 Task 
 | 
 
 
 Ranking and rank aggregation; classification and model selection; clustering and concept discovery; metric/distance computation and isomorphism testing; generation and enumeration; explanation and auditing (XAI) 
 | 
 
 
 § 4 , § 6 , § 7 , § 9 
 | 

 

 
 
 The taxonomy is intentionally coarse-grained: individual papers frequently combine categories from more than one axis (for example, a poset-pooling convolutional filter is simultaneously an algebraic/categorical representation, a supervised-learning paradigm, a vision-data modality, and a classification task). We use it as an organizing device rather than a strict partition, and we return to it in Section 9 when discussing gaps in the current literature – most notably the comparative scarcity of methods addressing heterogeneous or dynamically evolving posets. We also note that the representation axis is not strictly partitioned in practice: poset matrices are simultaneously a matrix-based computational representation and, once equipped with composition operations, an algebraic/categorical one [ 285 ] , and generation and isomorphism testing – added to the task axis to accommodate recent work discussed in Section 9 – are dual problems in the sense that a scalable generator is only as useful as its ability to cheaply determine whether a newly generated candidate duplicates one already found [ 283 , 284 ] . 

 
 
 

## 3 Basics of Partial Order Theory

 
 Order theory is a fundamental branch of mathematics which focuses on the arrangement of elements within various structures based on certain rules that can intuitively be captured using binary relations. It plays a crucial role in understanding mathematical sequences, hierarchies, and the organisation of data in fields such as computer science, economics, social sciences, environmental sciences, biomedical sciences etc. Generally, order theory deals with the structure and properties of partial orders which occurs naturally in subset relations and integer relations. The analysis of partially ordered sets is a well studied topic in discrete mathematics particularly in combinatorics
 [ 259 ] . The key concept in partial order theory is the ‘concept of comparison’. Objects are mutually compared, and transitivity is assumed [ 84 ] . That is if object x is better than object y, and object y is better than object z then object x is better than object z. However, comparison can be independently possible without the requirement of transitivity in some scenarios. For instance, in the theory of tournaments in sports, it is possible for team A to beat team B and team B beats team C without necessarily implying that team A beats team C.
Mathematically, we can summarize the definition of partial order sets as follows. A nonempty set X X of size n n which is endowed with the order relation “ ≼ \preccurlyeq ”(comparison operator) is called a partially ordered set or simply a poset,
denoted by P = ( X , ≼ ) P=(X,\preccurlyeq) , if the relation ≼ \preccurlyeq satisfies the following conditions: 

 
 1. 
 
 reflexive i.e. x ∈ X , x ≼ x ; x\in X,x\preccurlyeq x; 

 

 2. 
 
 antisymmetric i.e. x , y ∈ X x,y\in X if x ≼ y x\preccurlyeq y and y ≼ x ⟹ x = y ; y\preccurlyeq x\implies x=y; 

 

 3. 
 
 transitive i.e. x , y , z ∈ X x,y,z\in X if x ≼ y x\preccurlyeq y and y ≼ z ⟹ x ≼ z . y\preccurlyeq z\implies x\preccurlyeq z. 

 

 
 Moreover, reflexivity means that a given object can be compared with itself. Anti-symmetry means that if both comparisons are valid, i.e., y is better than x and at the same time, x is better than y, then this axiom requires that x is identical to y. Transitivity means that if the objects are characterized by properties which are at least ordinal scaled, then any measurable quantity like height, length, price, tournament team rankings, order of product quality, order of agreement or satisfaction etc., implies transitivity [ 85 ] .
If either x ≼ y x\preccurlyeq y or y ≼ x y\preccurlyeq x then x x and y y are comparable. For partially ordered sets it is not required that all pairs x , y ∈ X x,y\in X are comparable either as x ≼ y x\preccurlyeq y or y ≼ x . y\preccurlyeq x. In the case where all pairs x , y ∈ X x,y\in X are comparable, then the set X X is referred to as a totally ordered set or an ordered set. The numerical comparison operator ≤ \leq is used to denote the relationship between any two pairs of elements(objects) within an ordered set. With an ordered set, ranking is always possible and any sorting algorithm may be applied using ≤ \leq as the comparison operator for choosing whether to swap the objects in an ordered list. Classical problems of sorting and searching assume an underlying linear ordering of the objects being compared. An important result in order theory is that every partial order can be extended to a linear ordering or a total order. In order theory, a linear extension of a partial order is a total order (or linear order) that is compatible with the partial order. More formally, a linear extension [ 212 , 262 ] of a partially ordered set P is a permutation of the elements p 1 , p 2 , p 3 , … p_{1},p_{2},p_{3},... of P such that p i p j p_{i} p_{j} implies i j i j . For example, the linear extensions of the partially ordered set ( ( 1 , 2 ) , ( 3 , 4 ) ) ((1,2),(3,4)) are 1234 , 1324 , 1342 , 3124 , 3142 , and ​  3412 , 1234,1324,1342,3124,3142,\text{and}\;3412, all of which have 1 1 before 2 2 and 3 3 before 4 4 .
Some key concepts on partially ordered sets P = ( X , ≼ ) P=(X,\preccurlyeq) are described as follows: 

 
 
 
 • 
 
 Maximal elements (or objects) of the poset P P are the set of elements x ∈ X x\in X in which no other element y ∈ X y\in X satisfies the relation y x . y x. If x x is the only maximal element then it is referred to as the “greatest” element. In a totally ordered set, the terms maximal element and greatest element coincide. 

 

 • 
 
 Minimal elements (or objects) of the poset P P are the set of elements x ∈ X x\in X in which no other element y ∈ X y\in X exist such that y x y x . If x x is the only minimal element, then x x is referred to as the “least” element. In a totally ordered set, the terms minimal element and least element coincide. 

 

 • 
 
 Chain is a subset C of X, in which any object (or element) is mutually comparable with all other elements of C. That is, a chain C is a poset such that no element can be added between any two of its elements without losing the property of being totally ordered. 

 

 • 
 
 Antichain is a subset C ′ C^{\prime} of X, in which each object (or element) of C ′ C^{\prime} is mutually incomparable with all
other elements in C ′ C^{\prime} . That is, all the elements of the set C ′ C^{\prime} are pairwise incomparable and as such they can never have the property of being totally ordered. 

 

 • 
 
 A cover relation for 𝒫 = ( X , ≼ ) {\mathcal{P}}=(X,\preccurlyeq) is the set of pairs ( x , y ) (x,y) such that x , y ∈ X , x,y\in X, and y y covers x x whenever there exist no element z ∈ X z\in X such that x ≺ z ≺ y . x\prec z\prec y. 

 

 
 
 
 Beyond this relational description, a poset also admits an equivalent algebraic representation as a binary matrix, which is the representation underlying some of the algorithmic and machine-learning literature surveyed in this paper. Following the formulation in [ 284 , 314 ] , a poset matrix P ​ M ∈ { 0 , 1 } n × n PM\in\{0,1\}^{n\times n} encodes a poset P = ( X , ≼ ) P=(X,\preccurlyeq) on n n elements via P ​ M i ​ i = 1 PM_{ii}=1 (reflexivity), P ​ M i ​ j = 1 ⇒ P ​ M j ​ i = 0 PM_{ij}=1\Rightarrow PM_{ji}=0 for i ≠ j i\neq j (antisymmetry), and P ​ M i ​ j = 1 ∧ P ​ M j ​ k = 1 ⇒ P ​ M i ​ k = 1 PM_{ij}=1\wedge PM_{jk}=1\Rightarrow PM_{ik}=1 (transitivity), the last of which can be written compactly in Boolean arithmetic as P ​ M 2 ≼ P ​ M PM^{2}\preccurlyeq PM . If the elements of P P are topologically ordered, P ​ M PM takes a unit lower-triangular form, though this representation is not unique: simultaneous permutation of rows and columns yields another valid poset matrix for the same underlying poset, which is precisely why testing whether two poset matrices represent isomorphic posets is a nontrivial algorithmic problem in its own right (Section 9 ). Posets also admit a dual matrix representation P ​ M ∗ = [ p n + 1 − j , n + 1 − i ] PM^{*}=[p_{n+1-j,\,n+1-i}] , with a poset called self-dual when it is isomorphic to its own dual. This matrix-theoretic view of posets underlies not only the classical incidence- and rank-matrix methods used throughout Sections 6 and 7 , but also a growing body of purely combinatorial and algorithmic work that treats poset matrices as first-class objects of study: algebraically, via operad structures defined directly on the collection of poset matrices [ 285 ] , and algorithmically, via hierarchical matrix-decomposition methods for poset isomorphism testing [ 284 ] and reinforcement-learning-based generation of posets and lattices [ 283 ] , both discussed further in Section 9 . 

 
 
 Partially ordered sets are graphically visualized using Hasse diagrams. A Hasse diagram can be derived from a directed acyclic graph, where the vertices are representing the objects and
a line relates object x x with y y whenever x ≺ y x\prec y . In the case of a transitivity relation whenever x ≺ y x\prec y and y ≺ z y\prec z then it suffices to draw a line only for x ≺ y x\prec y and y ≺ z y\prec z and not for x ≺ z x\prec z . Hasse diagrams can be viewed as a powerful tool for unifying ideas and concepts. Hasse diagrams reveal an intricate network of comparabilities and incomparabilities, maximal and minimal elements. In addition, Hasse diagrams are also characterized with a structure that comprises levels, chains and antichains. 

 
 
 Hasse diagram: 
 2 4 3 1 
 

 
 
 Linear Extensions: 
 2 3 1 4 3 2 1 4 3 4 1 2 
 

 
 
 Finite posets can also be represented by square matrices which comprise: incidence [ 99 , 227 ] , cover [ 7 ] ,
and mutual ranking probability matrices [ 133 ] . The mutual ranking probability matrices are a class of matrices which comprise and convey information on the dominance among statistical units [ 133 ] . All the three main types of poset matrices are useful in conveying essential information on the structure of the order relation from different perspectives.
Posets can also be subdivided into variour classes such as series-parallel posets, semantic posets, and ranking posets etc. Series-parallel posets are characterized by the N-free partial order structure. Series-parallel partial orders have been applied to machine learning of event sequencing in time series data [ 216 ] . Semantic posets are used for compositional generalization in semantic parsing [ 152 ] .
The ranking poset 𝔐 n \mathfrak{M}_{n} [ 208 ] is the poset in which the elements are all possible cosets 𝔊 λ ​ π \mathfrak{G}_{\lambda}\pi ,
where λ \lambda is an ordered partition of n n and π ∈ 𝔊 n . \pi\in\mathfrak{G}_{n}. The ranking poset is useful in model selection and validation in machine learning.
More recently, Fueyo et al. [ 119 ] introduced leveled partially ordered sets , a class of posets equipped with a level (height) function generalizing Birkhoff’s classical notion of element height, originally motivated by the problem of ordering the layers of points generated during additive manufacturing (3D printing). A leveled poset satisfying a Jordan–Dedekind-type chain condition can always be completed into a bounded lattice by adding links to its existing order relation rather than by introducing new elements, which makes this class attractive for applications, discussed further in Section 9 , in which data arrives incrementally in layers or batches. From an algebraic and categorical perspective, posets can also be studied as algebras over the operad of posets : Arciniega-Nevárez, Berghoff, and Dolores-Cuenca [ 15 ] use the language of operads to formalize composition of posets and study a nontrivial suboperad, the Wixárika posets, together with its associated algebras, illustrating how order-theoretic combinatorics can be organized using the same categorical toolkit used elsewhere in machine learning (e.g., for compositional model architectures). 

 
 
 Posets are natural models for many statistical applications [ 39 , 287 , 247 , 282 ] . Partial orders are also the natural mathematical structure for comparing multivariate data that lacks a natural order [ 246 ] . For instance, multivariate data on colours has no natural ordering and therefore the rank features in colour spaces can be modelled as partial orderings, and this approach is generalizable in similar contexts [ 269 ] . It also plays a key role in decision making in environmental informatics and computational chemistry [ 53 , 160 , 196 , 198 , 197 , 56 , 64 ] , social sciences [ 29 , 68 ] etc. Moreover, in data analysis a typical binary relation is defined for a set of objects and a set of attributes. For instance, objects are transactions in a supermarket, attributes are items in the supermarket, and the relation consists of pairs (transaction, item) such that the item occurs in the transaction [ 207 ] . 

 
 
 The study of partially ordered sets and lattices has been covered by a significant number of mathematical publications. Birkhoffs (1967) [ 38 ] and
Gr a ¨ \ddot{\text{a}} tzer’s (1978) [ 151 ] books on lattice theory are considered to be classics. Lattices [ 312 ] are a broad structured class of posets. Lattice theory has shown its ability to contribute to solving problems in a wide variety of applications. Formally, a lattice [ 149 ] is a poset, in which every pair of elements has both a least upper bound and a greatest lower bound. In other words, it is a structure with two binary operations: join and meet. The following sub-definitions summarize a lattice structure.
A meet semilattice is a poset for which any two elements a and b have a greatest lower bound denoted a ∧ b a\wedge b . The greatest lower bound of a and b is the largest element that is still less than both of them. In a lattice, the greatest lower bound must be unique. The greatest lower bound of a and b is also called the meet or infimum of a a and b b . A join semilattice is a poset for which any two elements a a and b b have a least upper bound, denoted a ∨ b a\vee b . The least upper bound of a a and b b is the smallest element that is still greater than both. In a lattice, the least upper bound must be unique.
The least upper bound of a a and b b is also called the join or supremum of a a and b b . If a poset is both a meet semilattice and a join semilattice, then the poset is also a lattice. 

 
 
 Example of a lattice : 
 2 1 3 4 
 

 
 
 In the example above: The maximal elemnt is 4 4 and the greatest element is 4 . 4. Similarly, the minimal element is 1 1 and the least element is 1 . 1. 

 
 
 

## 4 Machine learning using posets and formal concept analysis

 

### 4.1 Overview

 
 In general, machine learning involves the creation of knowledge from a training dataset by using models and algorithms to learn from data in order to make predictions, find patterns, classify data and automate decision making processes. Machine learning has progressed significantly over the past two decades, and this has been driven by the development of new learning algorithms, theory, and high performance computing architecture [ 147 ] . There are many different kinds of machine learning algorithms. The most well-known ones are supervised, unsupervised, semi-supervised, and reinforcement learning. Supervised learning is useful in scenarios for which some relationship between input and output labelled data has been established, unsupervised learning is effective for uncovering hidden patterns in unlabelled data, and semi-supervised learning utilizes a combination of labelled and unlabelled data to train models. The goal of any machine learning algorithm is to optimize the performance of a system when handling new instances of data through user defined programming logic in a given environment [ 30 ] . In order to build more efficient machine learning algorithms or models so as to improve on their performance and accuracy in most predictive task, various ensemble learning methods have been developed. Ensemble learning [ 258 ] is an approach which aggregates the predictions of two or more models fitted to the same data in order to minimize error. In the area of supervised learning, the most popular ensemble machine learning techniques are bagging, boosting, and stacking, which use multiple learning algorithms or models to produce one optimal predictive model. Ensemble methods can broadly be categorized into sequential ensemble techniques and parallel ensemble techniques. Bagging(bootstrap aggregating) aims to reduce variance by adopting parallel ensemble learning on homogenous weak learners, boosting aims to reduce bias by adopting sequencial ensemble learning on homogenous weak learners, and stacking aims to improve prediction accuracy by adopting parallel ensemble learning on heterogeneous weak learners. Sequential ensemble techniques generate base learners in a sequence that are characterized by the dependence between the base learners. On the other hand the parallel ensemble technique base learners are generated in parallel, and in such a way that independence is enforced between the base learners. Dagging(Disjoint samples aggregating) is another useful type of parallel ensemble technique that was introduced in [ 295 ] . Dagging is similar to bagging, but instead of bootstrapping, it uses stratified sampling by creating a number of disjoint groups and stratified data from the original learning data set, with each considered as a subset of learning. In unsupervised learning, ensemble clustering [ 322 , 20 ] aims at combining the outputs of several clustering algorithms to form a single clustering structure (crisp or fuzzy partition, hierarchy). 

 
 
 In recent years, deep learning [ 115 ] has been the most popular computational approach in the field of machine learning and has unique advantages when dealing with high-dimensional nonlinear problems. The performance of machine learning algorithms relies heavily on the representation, and finding a good representation can facilitate the discovery of structure in the input data by the learning algorithm. Deep learning as a model is a specific type of representation learning [ 34 ] that uses layered algorithms known as artificial neural networks, which attempts to mimic the human brain through a combination of data inputs, weights, and biases, to learn representations of data. It has succesfully been used in a wide range of disciplines such as cybersecurity, natural language processing, visual recognition, machine translation, robotics, recommendation systems etc. There are several types of deep learning algorithms. These are: Convolutional Neural Network(CNN), Long Short Term Memory, Graph Neural Networks etc. Traditionally, machine-learning and deep learning algorithms have been used on data represented in Euclidean space such as image data which can be represented as a regular grid of pixel values. On the other hand, graph data cannot be represented on Euclidean space [ 21 ] . Graph Neural Networks (GNNs) [ 315 ] are a class of deep learning methods designed to perform inference on data described by graphs which are non-Euclidean (non-metric) structured. They are highly influenced by CNN which learns features by inspecting neighboring pixels(nodes in GNN) in three dimensional image data for classification and object recognition purposes. Posets are special classes of directed graphs. There is also a growing use of graph neural networks for the modeling and analysis of partially ordered data [ 134 , 117 , 152 , 310 , 320 ] . More recently, the problem of machine learning on meet/join lattices and posets was extensively studied in [ 311 ] . The main focus of the study was on developing methods that are based on generalized convolutions and sparse Fourier transforms algorithms which are capable of learning set functions. A complementary, more architecture-level use of order theory in deep learning is given by Dolores-Cuenca et al. [ 239 ] , who observe that every four-point poset corresponds to an order polytope and, via the correspondence between integer-valued neural networks and tropical rational functions, to a 2 × 2 2\times 2 convolutional filter; the resulting poset pooling filters can be inserted into any neural network and are reported to update weights during backpropagation with greater precision than average, max, or mixed pooling, without adding trainable parameters, while the composition of such poset-neural architectures can itself be organized using the operad of posets discussed above [ 15 ] .
The first lattice-based machine learning models were introduced by V.K. Finn [ 139 , 138 , 137 ] . It was based on a closure system which uses the JSM(John Stuart Mille)-method of automated hypothesis generation. In this model, positive hypotheses are searched among intersections(similarity as meet operations) of positive example descriptions (object intents), likewise for negative hypotheses [ 22 ] . The JSM method also identifies data patterns by means of induction and it can also be used as a logical rule-based classification method formulated in terms of formal concepts [ 204 ] .
Machine learning has been studied from many perspectives in the context of poset theory and lattice theory. These include : machine learning performnce comparison [ 40 , 208 ] , multilabel classification [ 110 ] , ensemble classification [ 208 ] , sequential clasification [ 216 , 288 , 287 , 293 ] ; natural language processing [ 117 , 214 , 152 ] , deep unsupervised learning [ 32 ] , semi-supervised learning [ 134 ] , learning to rank [ 97 ] , time series modelling [ 232 , 216 ] , ensemble clustering [ 110 ] , model selection [ 282 ] , topological deep learning [ 159 ] , attention-based neural networks [ 158 ] . 

 
 
 

### 4.2 Safe Reinforcement Learning, Control, and Order-Theoretic Dynamic Programming

 
 Two closely related bodies of work use posets not merely to represent data but to represent the constraints or the mathematical machinery of the learning problem itself . The first concerns safety-constrained control and reinforcement learning. Deploying learning-based controllers in safety-critical robotic systems typically requires enforcing multiple safety constraints simultaneously, and existing approaches usually do so either uniformly or via a fixed priority order, which can be infeasible or brittle when safety requirements are genuinely heterogeneous. Wong, Xiao, and Rus [ 238 ] instead formalize this setting as poset-structured safety : safety constraints are modeled as a partially ordered set in which some constraints are mutually comparable (one strictly takes precedence) while others are inherently incomparable, and safety composition is treated as a structural property of the policy class rather than as an externally imposed schedule. Building on this formulation, they propose PoSafeNet, a differentiable neural safety layer that enforces safety via sequential closed-form projection under poset-consistent constraint orderings, allowing the controller to adaptively select or mix valid safety executions while provably preserving priority semantics. This is a direct, and to our knowledge novel, instance of a neural architecture whose layer-level computation is explicitly organized by a partial order supplied by the problem specification, complementing the poset-pooling architectures of Section 4 in which the poset instead organizes the receptive field of a convolutional filter. 

 
 
 The second body of work uses partial orders as the foundational structure of dynamic programming itself, rather than of the state or action space in the usual sense. Sargent and Stachurski [ 236 ] represent a dynamic program abstractly as a family of policy operators acting on a partially ordered set, and derive an optimality theory – covering value function iteration and policy iteration – from purely order-theoretic assumptions; because the framework does not presuppose a real-valued Bellman equation, it uniformly covers standard Markov decision processes together with nonlinear recursive-preference models, robust-control objectives, and distributional dynamic programs whose value “functions” take values in a space of distributions. Peng, Stachurski, and Yang [ 234 ] strengthen this framework by pairing the order-theoretic assumptions with topological and metric stability conditions (global stability and contractivity of the policy operators), from which they recover convergence of value function iteration, Howard policy iteration, and optimistic policy iteration, together with a proof that stationary policies dominate nonstationary policy plans under weak assumptions; applications include optimal stopping without discounting and Bayesian sequential analysis, for which their results weaken existing assumptions. These two papers do not analyze machine-learned models directly, but they establish exactly the order-theoretic convergence theory that any future poset-structured reinforcement learning architecture – such as PoSafeNet – would need to invoke to obtain formal optimality or convergence guarantees, and we return to this connection in Section 9 . 

 
 
 A third and distinct role for RL in this space is as a generator of poset-structured objects rather than a consumer of a fixed poset-structured problem specification. Mwafise [ 283 ] trains a policy-gradient agent to construct finite lattices, join-semilattices, and meet-semilattices directly, motivated by the combinatorial explosion of exact lattice enumeration: the number of non-isomorphic lattices on n n elements grows from 5,994 5{,}994 at n = 10 n=10 to over 23 23 trillion at n = 20 n=20 (OEIS A006966), which confines exhaustive, orderly-generation algorithms to n ≲ 24 n\lesssim 24 . Rather than visiting graph nodes sequentially, as in GraphRNN-style autoregressive generators, the agent visits element pairs in a freshly randomized order at every rollout and chooses among “no relation”, u → v u\!\to\!v , or v → u v\!\to\!u under a hard pre-softmax legality mask that enforces the poset axioms structurally; an independent, exact tensor-based verification oracle then certifies the target closure property (lattice, join-, or meet-semilattice) for every candidate, so that no confirmed structure can be a false positive regardless of policy quality. The central empirical finding is that this order-agnostic construction alone – independent of any other architectural choice – improves the lattice-discovery rate at n = 20 n=20 by a factor of 4.7 4.7 (from 10.5 % 10.5\% to 49.7 % 49.7\% ) relative to fixed sequential visitation, because sequential, node-by-node construction structurally starves early-visited elements of the admissible relations a valid join or meet requires, regardless of how well the policy is trained. Combined with property-aware reward shaping that discourages the easily-reached distributive lattice class while rewarding the rarer modular non-distributive ( M 3 M_{3} ) and non-modular semimodular classes, an entropy-augmented Generalized Advantage Estimation variant, and GPU-vectorized batch generation, the framework sustains mean discovery rates of 60 60 – 80 % 80\% (peaking at 80 80 – 100 % 100\% ) across n ∈ { 30 , 40 , 50 } n\in\{30,40,50\} , a regime roughly double the practical ceiling of exact enumeration. The verified-oracle design is methodologically significant beyond this specific application: because the sparse, non-differentiable validity predicate is checked exactly rather than learned, training instability in the generator can depress discovery rates but can never manufacture a false positive, a separation-of-concerns principle directly analogous to the exact verification used elsewhere in poset-based safe learning (PoSafeNet’s projection step, above) and one we return to as a broader design pattern in Section 9 . 

 
 
 

### 4.3 Posets in Explainable Artificial Intelligence

 
 Posets are increasingly used in explainable artificial intelligence (XAI) to manage complexity, provide transparency, and represent hierarchical structure in model behavior and in data, precisely because they allow a model to represent that two objects, features, or explanations are genuinely incomparable rather than forcing an arbitrary total order between them. This is a natural fit for interpretability, where forcing a single linear ranking (as in a scalar composite score or a single feature-importance ranking) can hide the fact that different explanations are valid along different, mutually incompatible dimensions. Several strands of the material already reviewed in this survey can be read specifically through this lens. 

 
 
 
 • 
 
 Transparent decision-making. Rather than reducing complex, multi-dimensional data to a single score, the multi-indicator and multidimensional poset methods of Section 6 preserve the original comparability structure of the data and make explicit which objects dominate which others and along which criteria, which is the central requirement of a transparent, auditable ranking or scoring procedure [ 122 , 2 ] . 

 

 • 
 
 Formal concept analysis (FCA) and concept lattices. As discussed in Section 4.12 , FCA organizes a formal context into a concept lattice whose Hasse diagram directly exposes the hierarchy of object–attribute concepts, providing a visual and mathematically grounded explanation of how instances group together [ 143 , 312 ] . Where full concept lattices are too large to interpret directly, reduced skeleton structures known as AOC-posets (posets of attribute- and object-concepts) retain the information relevant for interpretation while substantially reducing the number of nodes that must be inspected. 

 

 • 
 
 Interpretable neural architectures. The poset-pooling filters of Dolores-Cuenca et al. [ 239 ] (Section 4 ) are explicitly motivated by interpretability: because each filter corresponds to an explicit poset and an associated order polytope, the effect of the filter on backpropagation can be analyzed geometrically rather than treated as a black box, in contrast to standard max or average pooling. 

 

 • 
 
 Model selection as a poset. As discussed in Section 3 and Section 7 , model-selection problems can themselves be organized as a poset in which a “least” element represents a null or baseline model and the order relation encodes model refinement [ 282 ] ; this makes explicit, and auditable, which comparisons between candidate models are actually licensed by the data rather than being an artifact of an arbitrary total ranking. 

 

 • 
 
 Handling incomparability as a consistency constraint. Across the applications reviewed in Sections 8 , incomparability is repeatedly identified not as a shortcoming to be resolved but as information: two objects, cities, chemicals, or models may simply not be comparable given the available criteria [ 231 , 84 ] . Explicitly representing this, rather than silently resolving it via aggregation, is a recurring argument in the poset literature for why posets support more honest, trustworthy explanations than composite-index or single-ranking approaches. 

 

 
 
 
 Taken together, these threads suggest that posets are already, if implicitly, a recurring representational choice in interpretable and auditable machine learning, even where the original works are not framed using XAI terminology; we discuss the resulting research opportunities further in Section 9 . 

 
 
 

### 4.4 Performance Comparison and Model Selection

 
 Comparing the performance of competing learning algorithms is one of the most common tasks in machine learning, yet it is typically reduced to a single scalar criterion (e.g., accuracy), which discards information whenever algorithms trade off differently across multiple performance measures. Blocher and Schollmeyer [ 40 ] address this by recasting algorithm comparison as a problem over partially ordered data. They extend classical statistical depth functions [ 328 ] – originally developed to generalize the univariate notions of median, quantiles, and order statistics to multivariate data – to the space of partial orders itself, via an adaptation of simplicial depth that they call the union-free generic (ufg) depth. A key property of the ufg depth is that it treats a partial order holistically rather than decomposing it into pairwise comparisons, which allows samples of poset-valued random variables (for example, the partial order induced by several performance measures over a set of algorithms) to be analyzed descriptively without collapsing them onto a single criterion such as balanced accuracy. This depth-function perspective is complementary to the multidimensional and descriptive poset-analysis methods discussed in Sections 6 – 7 , applied here specifically to the problem of algorithm and model comparison. 

 
 
 A related but distinct use of posets is in model selection itself, where the object of interest is not a ranking of performance scores but the space of candidate models. Taeb et al. [ 282 ] organize classes of models as partially ordered sets in order to address model structures that lack an underlying Boolean logical structure – a property that is otherwise a prerequisite for controlling the false-positive error rate of standard model-selection procedures. This poset-structured view of model selection recurs later in the survey, both as a component of formal concept analysis (Section 4.12 ) and as one of the recurring examples of order-theoretic explainability discussed in Section 4.3 . 

 
 
 

### 4.5 Natural Language Processing

 
 Compositionality [ 150 ] – the ability to recombine familiar units such as words into novel phrases and sentences – is a long-standing goal in natural language understanding, and existing neural sequence models are known to generalize poorly along this dimension [ 106 ] . To improve the compositional generalization of neural encoder–decoder architectures, Guo et al. [ 152 ] introduce a hierarchical poset decoding paradigm that explicitly accounts for the partial permutation invariance of semantic representations, treating the decoding target itself as a poset rather than a linear sequence. Compositional generalization is particularly relevant to tasks involving complex utterances with multiple equivalent meaning representations, such as semantic dependency parsing, in which a sentence is mapped to a directed graph capturing the semantic relations between its words [ 203 ] . In a closely related direction, Dyer [ 117 ] converts dependency trees into surface word orders using syntactic word embeddings combined with edge-weighted posets, training a graph neural network to learn the poset’s edge weights on the Universal Dependencies (UD) corpora (Section 8 ). Together, these results illustrate that posets can serve not only as a representation of the output of a language model, as in the decoding case, but also as the learned structure that mediates between a semantic representation and its surface realization. 

 
 
 

### 4.6 Classification

 
 Classification – the prediction of a discrete target label – has long been a central concern of machine learning, and many classification tasks are naturally reframed as ranking problems once the target admits a partial order. In the simplest case, a single label y ∈ Y y\in Y is associated with a covariate x x ; conditional ranking generalizes this by instead assigning x x a full or partial ranking of the items in Y Y . Lebanon and Lafferty [ 208 ] propose a unifying algebraic framework for ensemble classification and conditional ranking based on the ranking poset introduced in Section 3 : the collection of all full and partial rankings of a fixed set of items, partially ordered by refinement. The structure of this poset induces natural, order-invariant distance functions that generalize both Kendall’s Tau [ 192 ] and the Hamming distance, from which a generative probabilistic model can be derived and subsequently used for model selection and validation [ 208 ] . 

 
 
 A second strand of work derives classification rules directly from lattice structure. Sahami [ 266 ] introduces Ruleamer, an inductive algorithm that takes as input a lattice L L , a set of instance labelings C C corresponding to nodes of L L , and a noise parameter N N bounding the fraction of training instances a rule is permitted to misclassify, and outputs a set of symbolic classification rules; this line of work has since been extended to sequential data [ 103 , 102 ] . A related problem arises when classification states themselves form a lattice and response distributions vary by experiment: Tatsuoka [ 293 ] proposes a Bayesian framework for sequential classification under this assumption, building on an earlier data-analytic framework for fitting and validating latent, complex classification models [ 287 ] and on the theory of asymptotically optimal sequential experiment selection [ 288 , 289 ] , which applies whenever the parameter space is finitely partially ordered. These lattice-based classification models are a natural fit whenever classification states are more plausibly partially than totally ordered, as in cognitive and neuropsychological assessment, educational testing, and group-testing data (Section 8 ). 

 
 
 A third connection between posets and classification arises through multi-label classification and reasoning under uncertainty. Multi-label classification [ 37 ] allows each instance to be associated with a set of class labels, but becomes computationally expensive once classes overlap in feature space or the label space grows large, since the number of admissible label combinations grows exponentially. Dempster–Shafer theory (DST) [ 272 , 109 ] offers a generalized framework for reasoning under such uncertainty by blending probability with set-theoretic logic, but its computational cost likewise grows sharply with the number of events under consideration. Denoeux [ 110 ] shows that when the frame of discernment carries a lattice structure, the set of events under consideration can be restricted to intervals of that lattice, making DST computationally tractable for demanding tasks such as multi-label classification and the ensemble-clustering methods discussed in Section 5 . 

 
 
 

### 4.7 Deep Learning

 
 This subsection reviews three interconnected uses of posets in deep learning: unsupervised representation learning, attention-based architectures, and topological deep learning, extending the poset-pooling, safety-layer, and generative (reinforcement-learning) architectures already introduced in Sections 4 and 4.2 . Deep unsupervised learning [ 190 ] is attractive precisely because unlabelled data is far cheaper to obtain than labelled data, which matters for tasks such as visual similarity learning [ 96 ] that would otherwise require millions of labelled training pairs when addressed with standard convolutional neural networks (CNNs) [ 309 ] . Bautista et al. [ 32 ] propose an unsupervised alternative that frames visual similarity learning as a combination of surrogate (artificially constructed) classification tasks and a partial ordering over samples. This combination of classification with poset-structured ordering directly addresses three known limitations of purely surrogate-class approaches: many training samples cannot be confidently assigned to any surrogate class because their mutual similarity is not easily established; joint optimization across surrogate tasks is hindered by mutually conflicting relationships once transitivity fails to hold; and obtaining sufficiently many labelled samples for training remains costly. 

 
 
 Attention-based architectures constitute a second point of contact between deep learning and order theory. Neural attention models [ 226 ] restrict computation to informative regions of an image rather than processing it uniformly, originally motivated by both computational efficiency and an analogy to human visual attention, and have since become central to a wide range of vision and language tasks [ 268 , 42 ] . Hajij et al. [ 158 ] generalize this idea with higher-order attention networks (HOANs), defined over a combinatorial complex (CC) – a structure that unifies simplicial and cell complexes with hypergraphs, and that is itself a poset ordered by set inclusion. They show that any combinatorial complex reduces to a Hasse graph, which both characterizes the computational structure of HOANs in purely graph-theoretic terms and supplies a natural notion of equivariance for such networks. 

 
 
 This Hasse-graph reduction subsequently underpinned the broader development of topological deep learning (TDL) [ 327 ] , a framework that combines topological data analysis (TDA) with deep learning by using topological features to inform model design. TDL traces its origins to topological signal processing (TSP) [ 27 ] , which first demonstrated the value of modeling higher-order, multi-way relationships among data points that pairwise graph structures cannot capture. Hajij et al. [ 159 ] formalize this direction with combinatorial complex neural networks (CCNNs), an abstract architecture class that generalizes convolutional and attention-based networks and whose neighborhood structure is fully determined by the incidence, adjacency, and coadjacency matrices of the underlying combinatorial complex. Because every CC reduces to a Hasse graph describing the poset structure between its cells, Hajij et al. [ 159 ] show that (i) any CCNN-based computation can be realized as a message-passing scheme over a subgraph of the augmented Hasse graph of the CC, and (ii) the resulting tensor-diagram representation of a CCNN is likewise fully realizable on augmented Hasse graphs – reinforcing, at the architectural level, the connection between posets and multiparameter topological structure that we revisit in Section 9 . 

 
 
 

### 4.8 Semi-Supervised Learning

 
 Semi-supervised learning [ 301 ] bridges supervised and unsupervised paradigms by training on a combination of labelled and unlabelled data, and is particularly valuable when labelled data is scarce or expensive to obtain. Graph-based semi-supervised methods [ 276 ] have proven effective across many domains due to their structural flexibility and scalability, but they are typically restricted to ordinary graphs, in which each edge connects exactly two vertices. Hypergraphs generalize this by allowing an edge to join an arbitrary number of vertices, yet conventional hypergraph representations still treat each hyperedge as an unordered set, discarding any ordering relationship among its member vertices – relationships that are present in much real-world relational data. To address this gap, Feng et al. [ 134 ] introduce the Partial-Order Hypergraph, a data structure that explicitly injects partial-ordering relations among vertices into a hyperedge, together with a regularization-based learning theory that generalizes conventional hypergraph learning by incorporating logical rules encoding these partial-order relations. Feng et al. further apply a graph convolutional network (GCN) [ 195 ] in a semi-supervised setting over the resulting partial-order hypergraphs, demonstrating that the order-aware hypergraph construction is compatible with standard graph-neural-network training pipelines. 

 
 
 

### 4.9 Time Series Modeling

 
 Many machine learning and statistical models exist for modeling sequential data [ 318 ] , one important source of which is a series of discrete events occurring over time – as arises, for instance, in web browsing, e-commerce, and process monitoring. A central problem in mining such event sequences is recovering an overview of the ordering relationships they encode. Mannila and Meek [ 216 ] address this by treating a partial order as a generative model for event sequences, describing an observed set of sequences via a mixture of partial-order models; they show that the likelihood of a given partial order is inversely proportional to the number of total orders (linear extensions) compatible with it, and restrict the computation of this quantity to series-parallel posets, for which it can be evaluated efficiently. A greedy search algorithm over the space of partial orders on the set of possible events is then used for learning, and is able to test compatibility between a candidate partial order and an observed sequence in linear time. Nicholls et al. [ 232 ] apply a complementary Bayesian approach, inferring partial orders from random linear extensions of time-series data describing social hierarchy across the eleventh and twelfth centuries. Their model represents actors listed in order of precedence as realizations of a queue whose position is constrained by an underlying social-hierarchy partial order, providing a rank-order time-series framework for modeling how that hierarchy evolves over the observed period. 

 
 
 

### 4.10 Learning to rank

 
 Learning to rank (LTR) [ 71 , 324 ] describes a class of algorithmic techniques that apply supervised machine learning to solve ranking problems from a wide range of domains such as information retrieval, recommendation systems, search engine optimization etc. It arises from the need to obtain prediction results based on the best order of their relevance to a machine learning classification problem. Several methods have been proposed for LTR and they can be categorized or grouped into three main approaches: pointwise, pairwise , listwise. Pointwise approach involves scoring items independently and then ranking them based on their scores. In contrast the pairwise or listwise ranking methods, consider the relative positions of items in pairs or lists respectively. The goal for the ranker is to minimize the number of inversions in ranking i.e. cases in which the pair of results are in the wrong order relative to the ground truth. In particular, the listwise learning approach addresses the ranking problem, by analyzing ranked lists of objects as input instances and then trains a ranking function through the minimization of a listwise loss function defined on the predicted list and the ground truth list.
In [ 97 ] , it is noted that not all rankings can follow a strict total order in the context of learning to rank. As such a proposed method of relaxing the conventional setting such that predictions are given in terms of partial instead of total orders is outlined. The key idea is that if a model is uncertain about the relative order of two alternatives, in which case it is unable to clearly determine whether the former should precede the latter and vice-versa, it may decline or pospone the making of a conclusive decision and instead declare such pair of alternatives as incomparable.
In general, learning to rank problems as presented in [ 98 ] can be formulated as follows :
Given a set of training instances { x 1 , … , x n } ⊆ 𝒳 \{x_{1},...,x_{n}\}\subseteq\mathcal{X} and a set of labels 𝒴 = { y 1 , … , y k } \mathcal{Y}=\{y_{1},...,y_{k}\} endowed with an order y 1 y 2 , … , y k y_{1} y_{2} ,..., y_{k} such that for each training instance x l x_{l} it can be associated a label y l . y_{l}. The problem is to determine a ranking function that orders a new set of instances { x j ′ } j = 1 t \{x^{\prime}_{j}\}^{t}_{j=1} according to their (unknown) preference degrees. The performance meaures for this are: AUC( k = 2 k=2 , for the bipartite ranking) and C-index in the polytomous case ( k 2 k 2 , for k k -partite ranking). Since the ranker has the ability to reject predictions, there is a trade-off between correctness and completeness. Correctness is measured by gamma rank correlation [ 261 ] is a measure of rank correlation, i.e., the similarity of the orderings of the data when ranked by each of the quantities. Completeness measure penalizes the abstention from comparisons that should actually be made. Furthermore, a preference relation P : A × A → [ 0 , 1 ] P:A\times A\rightarrow\left[0,1\right] provides a
measure of support for the pairwise preference a ≻ b a\succ b with P ⁡ ( a , b ) = 1 − P ⁡ ( b , a ) P(a,b)=1-P(b,a) for all
 a , b ∈ A , a,b\in A, which can be considered an application of a generic approach [ 98 ] to transform every ranker into a partial ranker via ensembling such that: 

 
 
 
 1. 
 
 for any ranker L L , train k k ranking models M 1 ​ … ​ M k M_{1}…M_{k} by resampling from the original data set, i.e., by k k bootstrap samples. By
querying these models, k k rankings ≻ 1 … ≻ k \succ_{1}...\succ_{k} are generated; 

 

 2. 
 
 for each pair of alternatives a a and b b , the degree of
preference is defined as: P ( a , b ) = 1 k | { i ∣ a ≻ i b } | . P(a,b)=\frac{1}{k}\lvert\{i\mid a\succ_{i}b\}\rvert. 

 

 
 
 
 

### 4.11 Metrics and Distances over Posets

 
 A prerequisite for applying nearest-neighbor methods, kernel machines, or clustering algorithms (Section 5 ) to poset-valued data is a well-behaved notion of distance between posets, or between elements of a poset. Taeb, Guo, and Henckel [ 237 ] address the specific case of distances between graphs that represent statistical models, motivated by the observation that generic graph distances, such as the structural Hamming distance, ignore the structure of the underlying model space. They organize the graphs of interest – probabilistic undirected graphs, causal directed acyclic graphs, and several partially directed generalizations – into a poset ordered by model inclusion, which induces a neighborhood structure, and define the model-oriented distance as the length of a shortest path through this neighborhood structure; the resulting metric is shown to behave more consistently than existing alternatives when used to evaluate graph-structure-learning estimators. Olave [ 233 ] takes a more purely order-theoretic approach, introducing a family of extended metrics defined directly on path-connected and fence-connected posets without requiring any additional valuation structure; these metrics arise as a shortest-path distance that accounts for both path length and the number of order-direction alternations along the path, converge to a shortest-fence metric on discrete posets, characterize most discrete path-connected posets up to isomorphism, and coincide with interleaving distances when the poset is viewed as a thin category – connecting this line of work to the poset-cocalculus and multiparameter-persistence material discussed in Section 9 . Together, these two contributions suggest that poset-valued distance learning is beginning to mature from bespoke, domain-specific rank-correlation measures (Section 6 ) toward general-purpose metrics with formal characterization results, though, as discussed in Section 9 , their computational scalability to large posets remains to be systematically evaluated. 

 
 
 A closely related, and in a precise sense prerequisite, problem is poset isomorphism testing : any distance measure that assigns a poset to itself distance zero only up to relabeling requires an efficient way to decide whether two poset matrices (Section 3 ) represent the same underlying order relation. Because a poset on n n elements admits up to n ! n! distinct labeled matrix representations, naively comparing poset-valued data at scale – for example, deduplicating a large collection of posets before clustering, or indexing them for nearest-neighbor lookup – inherits the same factorial search space as the general graph isomorphism problem, of which poset isomorphism is a structured instance. Mwafise [ 284 ] addresses this directly with a tiered hierarchical matrix-decomposition framework that maps a poset matrix to a Hierarchical Poset Matrix Tree (HPMT) by recursively stripping universal bounds and partitioning disconnected cores, reducing isomorphism testing to a sequence of structured tree comparisons rather than an explicit search over permutations; a first tier resolves the common case of matrices with nontrivial universal bounds in empirically near-quadratic time (a 150 × 150\times speedup over the VF2 baseline at n = 100 n=100 , with VF2 timing out entirely by n = 500 n=500 ), a second tier handles reducible posets via direct-sum decomposition, and a third tier – required for highly symmetric, non-reducible structures such as crown posets and N N -dimensional hypercube (Boolean lattice) posets, where degree-based invariants collapse – extracts maximal disconnected principal submatrices in O ⁡ ( n 4 ) O(n^{4}) time, remaining tractable (e.g., 62 62 seconds for a 512 512 -element hypercube Q 9 Q_{9} ) at sizes where VF2-based backtracking times out. The framework is accompanied by formal invariance theorems establishing that the decomposition commutes with relabeling, so that isomorphic posets provably yield identical hierarchical fingerprints. For the metric-learning methods surveyed above, fast canonicalization of this kind is a natural and currently missing preprocessing step: it is what would make pairwise poset-distance computation tractable at the scale needed for the nearest-neighbor and kernel-based learning tasks motivating this subsection in the first place, a connection we return to in Section 9 . 

 
 
 

### 4.12 Machine Learning Using Formal Concept Analysis

 
 Formal concept analysis (FCA) [ 11 , 143 ] originates from partial order and lattice theory and provides a mathematically grounded method for conceptual knowledge representation and data analysis, introduced in the seminal work of Wille [ 312 ] as an attempt to restructure classical order and lattice theory around the notion of a concept . The basic data format in FCA [ 260 ] is a cross-table given by a triple ( O , A , I ) (O,A,I) called a formal context , where O O is a set of formal objects, A A is a set of formal attributes, and I ⊆ O × A I\subseteq O\times A is the incidence relation between them; a pair ( X , Y ) (X,Y) , where X X is a maximal set of objects (the extent ) sharing a maximal set of common attributes Y Y (the intent ), constitutes a formal concept . The set of all formal concepts of a context is partially ordered by extent inclusion and forms a complete lattice, whose representability by ordered sets of meet- and join-irreducibles [ 28 ] is the fundamental result underlying FCA. Because a formal context is naturally represented by a binary object–attribute matrix – rows indexed by objects, columns by attributes, with a 1 1 entry wherever an object possesses an attribute [ 204 ] – properties of lattice theory can be applied directly to the analysis of such matrices [ 205 ] . A range of discretization and Booleanization procedures further allow diverse datasets to be converted into formal contexts or concept lattices [ 9 , 8 , 10 ] , so that FCA can be viewed, in effect, as an unsupervised machine learning technique that takes a binary relation as input and returns the natural concepts of the data, organized as a Hasse diagram [ 92 , 145 , 168 ] . This generality has driven a growing adoption of FCA across data science tasks [ 206 ] , with several FCA-based methods reported to be competitive with classical machine learning approaches [ 5 ] . In an application to image classification, Khatri [ 200 ] identifies two specific advantages of FCA over convolutional neural networks: (i) FCA provides interpretability by construction, since the classification hierarchy is directly visualized as a lattice, and (ii) data can be added to or removed from the lattice without retraining the model, unlike a trained CNN – with the resulting FCA-based classifier reported to outperform the random forest classifier, itself considered a relatively interpretable baseline. A further advantage of FCA-based classification more generally is that it makes no distributional assumptions about the underlying data [ 249 ] , in contrast to many classical statistical classifiers. 

 
 
 This lattice-theoretic view of classification extends naturally to learning from positive and negative examples , in which a learning system constructs a generalization of positive examples that excludes (does not “cover”) negative examples [ 205 ] ; several FCA-based models address this problem directly [ 206 , 241 , 205 ] . Jabin [ 174 ] applies a genetic algorithm to automatically learn object-oriented hierarchies closely related to lattice-based concept hierarchies from large datasets, while Ikeda and Yamamoto [ 169 ] classify data by constructing a concept lattice and then selecting the formal concepts within it that are most informative for a given classification task, improving feature-selection robustness by retaining both selected and redundant concepts. FCA has also been integrated with ensemble-learning strategies – Dagging [ 222 , 5 ] , Boosting [ 221 ] , Bagging [ 188 ] , and recommendation-based multiple classifier systems [ 191 ] – extending the ensemble-clustering connections already discussed in Section 5 . Xie [ 317 ] embeds simple base classifiers (Naive Bayes and k k -nearest-neighbor) into each node of a concept lattice, producing composite Concept Lattice Naive Bayes (CLNB) and Concept Lattice Nearest Neighbor (CLNN) classifiers that outperform their respective base classifiers – and, in the case of CLNB, several state-of-the-art classifiers – across 26 benchmark datasets. 

 
 
 FCA’s inherently visual, lattice-based structure also makes it a natural tool for the explainability problem introduced in Section 4.3 : because many machine-learning-based AI systems are designed as black boxes, achieving interpretability in the face of increasingly complex model architectures remains a central challenge. Sangroya et al. [ 267 ] propose a general concept-lattice-based framework for explaining deep learning outcomes, in which a model prediction and a domain ontology are combined to identify the explanation that points a user to the most salient feature set underlying that prediction. 

 
 
 Finally, FCA supports the automated construction of domain-specific ontologies directly from textual descriptions of domain entities [ 26 , 298 , 326 ] . Ontology learning [ 229 ] is the broader process of automatically extracting knowledge structures – typically annotated taxonomies, concept hierarchies, or domain ontologies – from unstructured or semi-structured sources such as text, speech, images, or sensor measurements; ontologies of this kind are a key building block of the semantic web, capturing domain knowledge in a machine-processable form. Relational Concept Analysis (RCA) [ 167 ] extends the FCA framework to multi-relational datasets by generating a family of concept lattices, one per object category, and both FCA and RCA underpin a range of approaches to ontology learning and extraction [ 163 , 183 , 157 , 242 , 101 ] . 

 
 
 
 

## 5 Clustering Partially Ordered Data

 
 Cluster analysis is a multivariate technique for identifying structural patterns in complex data by grouping similar objects together, typically by evaluating similarity or dissimilarity across a set of shared attributes. Under the unsupervised-learning branch of the taxonomy in Table 1 , several distinct lines of work have brought poset and lattice structure to bear on this problem: an ordinal model of clustering built directly from order theory, hierarchical clustering methods that operate on the Hasse graph of a poset, ontology -driven clustering based on the subset relation, and conceptual clustering derived from formal concept analysis (Section 4.12 ). 

 
 
 The ordinal model for clustering with posets, introduced by Janowitz [ 178 ] , shows that characterizing flat cluster methods [ 177 ] reduces to a universal mapping problem in the theory of partially ordered sets, with generalized notions of adjoints of order-preserving mappings between posets recurring as a persistent underlying theme [ 179 ] . Building on this, Janowitz [ 179 ] develops a clustering scheme based directly on dissimilarities measured over posets, in keeping with the broader observation that most classificatory clustering methods operate on a dissimilarity coefficient defined over a set of objects [ 36 ] . 

 
 
 Cluster analysis is traditionally divided into hierarchical and non-hierarchical methods: hierarchical clustering produces a nested sequence of partitions, while non-hierarchical clustering produces a single partition. Hierarchical clustering further splits into divisive methods, which proceed top-down by successively dividing a single cluster as inter-object distance increases, and agglomerative methods, which proceed bottom-up by successively merging clusters as inter-object distance decreases; agglomerative clustering is commonly further categorized into single linkage (minimum distance between clusters) and complete linkage (maximum distance between clusters) [ 252 ] . Several authors have combined these classical hierarchical schemes with poset structure. Sabara et al. [ 263 ] apply both single- and complete-linkage agglomerative clustering directly to the Hasse graph of a poset in a multidimensional data-analysis setting (Section 6 ), while Wu et al. [ 316 ] and Kardaetz et al. [ 189 ] likewise combine poset methodology with hierarchical clustering from a multidimensional perspective; Wu et al. [ 316 ] in particular develop a hierarchical stratification method, interpretable via the partial-order Hasse graph, that divides multi-dimensional indicators into layers with distinct properties. Janowitz [ 180 ] provides a comprehensive review of this broader area, including the use of lattices to generalize tree-based clustering. 

 
 
 A related strand of work uses posets to formalize the relationship between clustering and ontologies , which represent data as hierarchies of possibly overlapping classes and are thus closely related to clustering hierarchies in their own right. Liu et al. [ 213 ] show that modeling ontologies as posets over the subset relation allows classical dissimilarity-matrix-based clustering algorithms to incorporate all available ontological information without loss, and use this observation as the basis for a clustering algorithm that produces a partially ordered set of clusters directly from a dissimilarity matrix. Because a dissimilarity matrix has size 𝒪 ⁡ ( N 2 ) \mathcal{O}(N^{2}) in the number of objects N N , independent of the dimensionality of the objects themselves, this poset-of-clusters construction sidesteps some of the difficulties otherwise associated with clustering high-dimensional data. 

 
 
 A fourth strand connects clustering directly to formal concept analysis (FCA), which – beyond its role in classification (Section 4.12 ) – is itself a form of conceptual clustering [ 180 ] , a branch of machine learning introduced by Michalski [ 224 ] that groups unlabelled objects into classes. Carpineto [ 91 ] identifies three defining features of conceptual clustering methods: (i) each output class is characterized both extensionally, by the objects it covers, and intensionally, by the concept it represents; (ii) output classes are arranged into a hierarchy ordered by generality; and (iii) class formation proceeds incrementally, so that processing the n n th object does not require reprocessing the preceding n − 1 n-1 objects. The theory of concept (Galois) lattices offers a natural formalization of this process: the Galois lattice generated from a binary relation [ 148 ] is itself a concept hierarchy, and Carpineto’s GALOIS algorithm [ 90 ] computes the concept lattice for a given set of objects with an update-time complexity ranging from O ⁡ ( n ) O(n) to O ⁡ ( n 2 ) O(n^{2}) in the number of concepts n n , and has been shown useful for both class discovery and class prediction. Related conceptual-clustering perspectives include Fisher’s treatment of conceptual clustering as an extension of numerical taxonomy [ 135 ] ; Restrepo et al.’s use of hierarchical cluster analysis (HCA) to reduce the number of elements in a poset to a set of representatives, thereby improving the interpretability of the resulting Hasse diagrams [ 254 ] ; and Zhang et al.’s distance-function-based hierarchical conceptual clustering, which mitigates the NP-hardness of exhaustively determining all concepts of a formal context by subsetting the feature set used to define the clusters [ 325 ] . Markov [ 218 ] instead induces a lattice structure over the clusters via a generalization operator, maximizing overall clustering quality by evaluating the hierarchy level by level in a bottom-up fashion, while Yoneda et al. [ 321 ] use the algebraic closedness property of FCA to learn a graph-structured representation of multivariate data in which each node is a cluster and each edge encodes a subset–superset relationship between clusters; related graph-based hierarchical conceptual clustering methods applicable to partially ordered data are studied in [ 184 , 185 ] . 

 
 
 Posets also arise naturally within a single clustering run, independent of any external ontology or concept lattice, through the space of partitions itself: the frame of discernment in clustering is the set of all partitions of a finite set E E , denoted 𝒫 ⁡ ( E ) \mathcal{P}(E) , and this set carries a natural partial order in which a partition p p is finer than a partition p ′ p^{\prime} (written p ≼ p ′ p\preccurlyeq p^{\prime} ) whenever the clusters of p p are obtained by splitting those of p ′ p^{\prime} , yielding the poset ( 𝒫 ⁡ ( E ) , ≼ ) (\mathcal{P}(E),\preccurlyeq) . This structure underlies ensemble clustering , which combines the outputs of several clusterers into a single clustering structure. Using evidential reasoning grounded in Dempster–Shafer theory [ 110 ] – already introduced in the classification context of Section 4 – one can assume the existence of a “true” partition p ∗ p^{*} , of which each clusterer supplies partial evidence; the evidence from multiple clusterers can then be combined within the partition poset to draw plausible, well-supported conclusions about p ∗ p^{*} even in situations where no single clusterer’s output is decisive. 

 
 
 

## 6 Multidimensional Data Analysis

 
 Multidimensional data analysis (MDA) originated with the development of relational databases and On-Line Analytical Processing (OLAP) by Codd in 1993 [ 248 , 165 , 141 , 210 ] , whose central idea – organizing data along dimensions – provides a flexible way to interpret the same dataset from several angles at once. In statistics, econometrics, and related fields, MDA accordingly groups data into dimensions (the axes along which analysis is carried out) and measurements (the values associated with different dimensional entities), with the resulting hierarchy, sequencing, and dependency relationships among dimensions typically organized into a meaningful structure [ 165 , 43 , 248 ] . 

 
 
 Multi-indicator models [ 281 ] – in which two or more “alternative” measures are used for the same underlying concept – are especially useful in this setting; a familiar example is measuring individual satisfaction with a service via several differently worded survey questions, and this kind of data is common in public-opinion surveys of many thousands of individuals [ 72 , 172 ] and in environmental and infrastructure monitoring, where a battery of indicator values assesses several features of a site’s condition [ 231 ] . In both settings, different indicators frequently convey conflicting comparative signals about the same object. Multi-indicator systems are similarly central to modeling business processes that span multiple geographic regions, channels, and products, but quantifying such systems is difficult and time-consuming because of their inherent complexity [ 58 , 66 ] : nominal or ordinal multivariate data is often first scaled to a quantitative form, a step that can itself introduce inconsistency [ 120 ] . The conventional solution – aggregating indicators via a weighted average or another aggregation scheme (additive [ 173 ] , hierarchical [ 244 ] , etc.) into a single composite indicator [ 230 ] , which is then used for ranking [ 60 ] – is a widely used method in the social sciences for measuring multi-dimensional phenomena such as well-being (which might combine income, employment, health, and education into one score) [ 23 ] . However, the validity and robustness of composite indices constructed this way has been repeatedly challenged, largely because of the unavoidable subjectivity involved in choosing aggregation weights: aggregative methods tend to oversimplify the underlying phenomenon and can produce identical scores for genuinely different situations [ 3 , 2 ] , and ordinal data of this kind often lacks a clear ordering criterion in the first place, making individual-level comparisons ambiguous [ 120 ] . A substantial body of work (e.g., [ 23 , 51 , 245 , 122 , 58 , 72 , 131 , 12 , 19 , 70 , 126 , 83 ] ) therefore advocates replacing aggregation with partially ordered set theory, which restricts subjectivity to the choice of which properties to consider in the first place, rather than to how they are numerically combined. The underlying motivation is that classical aggregative methods can fail to express the true complexity of a phenomenon, whereas partial orders make explicit why an object occupies a given ranking position, and how sensitive that position is to changes in the underlying indicators [ 58 ] – directly complementing the explainability motivation for posets discussed in Section 4.3 . The principal advantages of treating multivariate data as a poset for ordering purposes are summarized below. 

 
 
 
 • 
 
 Partial order theory yields rankings that can accommodate ties without requiring indicator weighting [ 84 ] : the horizontal arrangement of objects within a Hasse diagram, organized by level, already gives a first approximation to a weak order in which tied ranks are not excluded. 

 

 • 
 
 Partial orders impose no requirement that indicators be mapped onto a single common scale. 

 

 • 
 
 Phenomena such as subjective well-being can be evaluated consistently and effectively using partial order theory, overcoming the limitations of both composite and simple counting paradigms [ 125 ] . 

 

 • 
 
 Partial order theory handles multidimensional systems of ordinal data without variable aggregation into composite indicators, so there is no need to convert ordinal scores into numerical values – a conversion that can itself introduce inconsistency into the evaluation of the underlying phenomenon [ 122 , 12 ] . 

 

 • 
 
 No weighting of evaluation dimensions is required to account for their differing relevance [ 122 ] . 

 

 • 
 
 Objects are compared simultaneously across all indicators, without any need for prior aggregation [ 85 ] . 

 

 • 
 
 Partial order ranking is a non-parametric method, requiring no assumptions of linearity or of any particular statistical distribution over the attributes [ 75 ] . 

 

 • 
 
 The graphical representation of posets as Hasse diagrams requires no additional information to sort objects; because a Hasse diagram represents a well-defined mathematical structure, further conclusions can be drawn from it beyond a simple sorted order [ 51 , 46 ] . 

 

 • 
 
 Posets define the structure of comparabilities underlying a multi-indicator system [ 2 ] , permitting mathematical analysis of the phenomenon under study and enabling a reduction in its dimensionality without loss of information, since no difference or ratio of indicator values is ever computed to derive distances or proportions. 

 

 • 
 
 Because different indicators often convey different comparative messages, there is frequently no single, unambiguous way to rank a set of objects while honoring all of the available indicator information [ 231 ] : object A A may dominate object B B on one indicator while B B dominates A A on another. This complicates prioritization based on raw indicator scores, but the properties of posets and their associated rank matrices provide additional information useful for decision-making beyond a single overall ordering. 

 

 • 
 
 Representing a phenomenon by a Hasse diagram avoids the arbitrariness inherent in constructing a single ranking index [ 51 ] , and offers a holistic view of all objects to be ranked without introducing artificial ranking indices [ 47 ] . 

 

 
 
 
 We now summarize how a poset structure is obtained from a subjective dataset in practice. Let k k denote the number of subjective indicators used to study a phenomenon of interest, and let n n denote the number of individuals in the study, so that A = { q 1 , q 2 , … , q k } A=\{q_{1},q_{2},\ldots,q_{k}\} represents the set of subjective-indicator variables. Each indicator is scored on an m m -degree ordinal scale ( y 1 , y 2 , … , y m ) (y_{1},y_{2},\ldots,y_{m}) with y 1 y 2 ⋯ y m y_{1} y_{2} \cdots y_{m} , and each of the n n individual profiles is determined by the m m ordinal scores assigned across the k k indicators in A A . Pairwise comparison of all n n individual profiles then yields a poset structure whenever the resulting comparisons are not totally ordered, which can be depicted as a Hasse graph of size n n . For a four-degree scale, for example, the profile ( y 2 , y 3 , y 2 , y 3 ) (y_{2},y_{3},y_{2},y_{3}) dominates ( y 4 , y 1 , y 3 , y 1 ) (y_{4},y_{1},y_{3},y_{1}) , whereas ( y 3 , y 3 , y 4 , y 2 ) (y_{3},y_{3},y_{4},y_{2}) and ( y 4 , y 3 , y 3 , y 2 ) (y_{4},y_{3},y_{3},y_{2}) are incomparable. Several complementary methodologies have been developed for analyzing multivariate datasets of this kind, based respectively on (i) a fuzzy set approach [ 121 , 120 ] , (ii) partial order linear extensions [ 122 ] , (iii) average rank [ 72 ] , (iv) poset theory combined with the Adjusted Mazziotta–Pareto Index [ 1 ] , (v) interpersonal comparability [ 270 ] , and (vi) fuzzy first-order dominance [ 131 ] , summarized in turn below. 

 
 
 
 • 
 
 The fuzzy set approach, originating in Zadeh’s seminal work [ 323 ] , models uncertainty over vague concepts by allowing partial set membership. Combined with posets, it has been used to statistically evaluate ordinal data on socio-economic phenomena while overcoming the limitations of classical composite-indicator aggregation [ 121 , 120 , 17 ] , and its main advantage is that the measurement level of the data is fully respected, avoiding any improper rescaling. 

 

 • 
 
 Fattore et al. [ 122 ] rank a finite collection of objects – each represented by a suite of indicator values, i.e., a point in a multidimensional cloud – by treating their relative positions as defining a partial order, avoiding the arbitrary assignment and aggregation of a composite numerical score; incomparable pairs are simply left incomparable. Each object’s possible rank interval is then determined from the Hasse diagrams of all linear extensions compatible with the partial order. Since exhaustively enumerating linear extensions becomes computationally intractable for large datasets, Bruggemann et al. [ 245 ] instead propose Discrete Markov Chain Monte Carlo (MCMC) sampling of linear extensions [ 69 ] , followed by cumulative-frequency ranking based on the sampled extensions, while Lerche [ 142 ] develops ranking-probability estimates from random linear extensions – both approaches make it practical to predict ranking probabilities and average rank for posets too large to enumerate exhaustively. 

 

 • 
 
 Caperna and Boccuzzo [ 72 ] address big, complex datasets in which some attributes require measuring complex concepts on ordinal or dichotomous scales, by sampling units from the population using a simple criterion and then applying the central limit theorem to compare group-level results via standard statistical tests on the means; their Height of Groups by Sampling method then compares average rank across groups defined by one or more socio-demographic variables. 

 

 • 
 
 Patil et al. [ 246 ] review procedures for extracting ordering properties embodied in multivariate datasets, applicable to configuring sets of indicators more generally. 

 

 • 
 
 Alaimo et al. [ 1 ] synthesize the evolution of multi-indicator systems over time by combining partial order theory with the Adjusted Mazziotta–Pareto Index, applied to one of the fifteen Sustainable Development Goals; posets define the comparability structure underlying the multi-indicator system, scores are then evaluated to reduce the phenomenon’s dimensionality, and temporal posets are obtained by merging posets across time periods, with an embedded five-level scale (minimum, maximum, and the first, second, and third quartiles of the indicator values) used to further refine the measurement. 

 

 • 
 
 Sen [ 270 ] develops a framework for interpersonal comparability in aggregating individual welfare measures, considering an aggregation relation between social states that reduces to the Pareto quasi-ordering under noncomparability and becomes a complete ordering under unit or full comparability. 

 

 • 
 
 Fattore and Maggino [ 125 ] outline a comprehensive evaluation procedure for multidimensional ordinal settings, comprising identification of evaluation dimensions, assignment of attribute relevance, threshold selection, and computation of evaluation scores at both the individual and population level. 

 

 • 
 
 Fattore [ 127 ] proposes a partial-order-based procedure for assessing multidimensional deprivation with ordinal data, focusing on achievement profiles and their comparabilities and incomparabilities; the resulting synthetic indicators account for both the vagueness and the intensity of multidimensional deprivation, incorporate attribute-importance information, and handle missing data. 

 

 • 
 
 Fattore [ 131 ] introduces fuzzy first-order dominance (F-FOD), which combines partially ordered set theory with fuzzy relational calculus to overcome limitations of earlier first-order-dominance algorithms, producing full pairwise comparison matrices from which partial orderings and rankings of statistical units can be derived. 

 

 • 
 
 Bruggemann et al. [ 52 ] propose an average-rank method based on the structure of the Hasse diagram associated with the order relations encoded in the data matrix. 

 

 
 
 
 

## 7 Exploratory and Descriptive Data Analysis

 
 Exploratory and descriptive data analysis are closely related stages of the data-analysis pipeline, and posets have been used extensively in both: exploratory data analysis (EDA) is the preliminary investigation of a dataset to uncover patterns, detect anomalies, test hypotheses, and check assumptions, while descriptive analysis summarizes a dataset’s primary properties and uses historical trends and relationships between variables to inform further analysis and decision-making. We treat both together in this section, since the poset-based techniques that support them – handling noise and missing data on one hand, and visualizing the order structure of a dataset on the other – form a continuous methodological pipeline. 

 
 

### 7.1 Noise, Trend, and Missing-Data Handling

 
 A major challenge in the exploratory phase is identifying noisy data, characterized by a degree of variability that the rest of the dataset cannot explain, since noise can alter the ordering relations between objects and thus introduce genuine data uncertainty into a partial ordering [ 277 ] . Bruggemann and Carlsen [ 65 ] address this by constructing a probability scheme that specifies an explicit noise model, from which the distribution of noisy values can be derived analytically; this supports a priori estimation of how a given noise level affects the order relation between any pair of objects, so that the expected magnitude of noise-induced perturbation can be quantified in advance. Carlsen and Bruggemann [ 80 ] extend this analysis with a fuzzy approach in which the partially ordered set itself is treated as a function of the noise level, making it possible to identify the range of noise over which the original partial order remains stable. 

 
 
 Trend analysis – tracking how a phenomenon evolves over time – is a natural fit for poset-based exploratory analysis. Alaimo et al. [ 1 ] show that analyzing the evolution of Hasse diagrams over time yields a preliminary depiction of a phenomenon’s temporal trend, though missing data remains a well-known obstacle to nonlinear trend analysis more generally [ 271 ] . Fattore [ 127 ] addresses this obstacle directly with a poset methodology for handling missing values, applied to a case study of deprivation data; the underlying idea is that pairwise comparison of poset elements is inherently robust to missing information, since the absence of sufficient information to compare one object to a particular other object does not preclude using the remaining, available comparisons elsewhere in the dataset [ 23 ] . In the specific context of formal concept analysis (Section 4.12 ), this same robustness – together with FCA’s human-centered, visually grounded representation of concepts – makes it particularly well suited to exploratory data analysis [ 33 , 168 ] . 

 
 
 

### 7.2 Visualization via the Hasse Diagram Technique

 
 Turning from exploration to description, poset theory has been used extensively for data summarization and visualization, most centrally through the Hasse diagram, which conveys a considerable amount of information about a dataset’s partial order structure in a single graphical representation. Hasse diagrams have accordingly been used for visualization tasks spanning causality [ 297 ] , socioeconomic analysis [ 123 ] , linear models and analysis of variance [ 25 ] , experimental design [ 187 ] , learning analytics [ 193 ] , image analytics [ 105 ] , medical data analysis [ 112 ] , molecular structure prediction [ 199 , 223 ] , spatial analysis [ 186 , 49 ] , multidimensional analysis [ 104 , 87 , 58 ] , biomonitoring [ 300 ] , multivariate analysis [ 68 ] , sustainability analysis [ 89 ] , multi-criteria decision analysis [ 88 ] , environmental data analysis [ 299 ] , and decision support [ 313 , 231 ] , among other domains reviewed further in Section 8 . The overarching goal of such a visualization tool, for multidimensional and partially ordered datasets alike, is to represent the data structure directly, reducing its complexity while retaining its essential patterns [ 123 ] . 

 
 
 Among the formal methods built around the Hasse diagram, the most established is the Hasse Diagram Technique (HDT), introduced by Bruggemann and Voigt [ 55 ] as an application of partial order theory to a data matrix, used to analyze the structure of multivariate datasets whose objects are characterized by multiple attributes (indicators). The HDT sorting rule [ 52 ] can be stated concisely: given a dataset of objects to be sorted with respect to some criterion, let m ⁡ ( i , x ) m(i,x) denote the value of the i i th attribute of object x x , for i = 1 , … , n i=1,\ldots,n ; then x ≥ y x\geq y (i.e., x x is evaluated as at least as good as y y ) if and only if m ⁡ ( i , x ) ≥ m ⁡ ( i , y ) m(i,x)\geq m(i,y) for every i = 1 , … , n i=1,\ldots,n . The HDT is especially useful for multi-criteria ranking problems in which each object must be assessed relative to all others [ 46 ] , for several reasons: it makes the ranking process transparent; it requires no normative constraints on how the ranking is performed; it enables extraction of structural information from, and visualization of, the resulting Hasse diagrams [ 51 ] ; and it supports visualization of sensitivity-analysis parameters [ 45 ] . 

 
 
 Bruggemann et al. [ 52 ] apply the HDT to sort chemicals by potential environmental hazard, establishing dominance of one chemical over another whenever every attribute simultaneously supports that dominance. One limitation of the basic HDT is its lack of weighted aggregating functions, which would otherwise allow conflicting descriptor values to be resolved; Simon et al. [ 274 ] address this with METEOR (METhod of Evaluation by ORder), an extension of the HDT procedure that, unlike the base method, combines transparent decision support with the ability to incorporate stakeholder preferences [ 275 ] , resolves the problem of obtaining a single top-ranked object, and avoids costly trial-and-error selection of descriptor priorities by directly computing the probability of a given linear order under a specified descriptor prioritization [ 255 ] . METEOR has subsequently been applied to the computational evaluation of chemicals and environmental hazards [ 306 , 56 ] . In a related vein, Halfon et al. [ 160 ] propose a vectorial ranking procedure for partial ordering that applies broadly across problems in environmental toxicology, motivated by the poset’s ability to represent both comparable and genuinely incomparable chemicals with respect to environmental hazard; the resulting rankings are visualized via computer-generated Hasse diagrams applicable to datasets of any size, with the chain and antichain structure of the diagram corresponding respectively to its vertical and horizontal components. Bruggemann [ 61 ] shows that tripartite graphs are useful for interpreting complex datasets of this kind by clarifying which indicators drive the incomparabilities reflected in a Hasse diagram’s horizontal components, and Carlsen et al. [ 79 ] demonstrate, in an application to analytic-performance evaluation, that such incomparabilities – an inherent feature of partial order ranking rather than a defect – can themselves reveal important characteristics of the data, with summary statistics (absolute z z -score, absolute skewness, standard deviation) visualized directly on the Hasse diagram to provide a holistic view of the results. A related graphical technique, POSAC [ 253 , 70 ] , maps the rows of a data matrix (e.g., geographic regions) into a two-dimensional space that maximizes preservation of their partial order, placing similar objects in close proximity; POSAC has proven useful for tracking the evolution of crime analytics across large geographical areas over time. 

 
 
 Formal concept analysis offers a complementary route to visualization: its concept lattice, together with the associated Hasse graph, provides a conceptual framework for structuring, analyzing, and visualizing data in a more concise and comprehensible form [ 140 ] , distinguished from a plain Hasse diagram by its “symmetric” treatment of objects and properties rather than objects alone [ 61 ] . Large datasets nonetheless remain computationally challenging for FCA, since the number of formal concepts derived from a dataset is the key factor determining whether the resulting concept lattice is usable for visualization. Andrews and Orphanides [ 9 ] show that interpretable results can still be obtained from data sources that would otherwise be intractable to visualize, by first focusing on the information of interest and then reducing noise in the formal context, thereby revealing readable lattices that faithfully represent the conceptual structure of large datasets. 

 
 
 A general limitation of the Hasse diagram as a representation, independent of FCA, is that it becomes difficult to interpret as the number of vertices and edges grows, and it provides no metric information even when such information is available in the original data. Cluster analysis (Section 5 ) addresses the complexity problem but is not designed to preserve comparability and incomparability information. Bruggemann and Carlsen [ 62 ] combine the two, producing a visual output that lets end users jointly grasp both the partial order and the metric structure of the data via a three-step process detailed by Fattore et al. [ 123 ] : (i) reduce dataset complexity through clustering based on a self-organizing map (SOM); (ii) build a classical Hasse diagram over the resulting population of clusters, associated with the SOM weight vectors; and (iii) visually annotate the diagram with information on statistical units, clusters, and covariate values. Alternatively, combining cluster analysis and principal component analysis [ 51 ] with the HDT can help obtain a statistically relevant data representation while avoiding insignificant numerical differences between attributes that would otherwise generate spurious comparabilities and incomparabilities, and correspondingly overcomplicated Hasse diagrams. 

 
 
 
 

## 8 Applications

 
 This section surveys the datasets, software, and algorithmic resources that support the practical analysis of partially ordered data. Since most machine-learning applications of posets have already been discussed in Section 4 , the focus here is on data-analysis applications across application domains, following the data modality and task axes of the taxonomy in Table 1 . Partially ordered data and structures are ubiquitous, and a significant share of their applications concern data-driven decisions in sustainable development [ 166 , 2 , 164 , 1 ] , where the central use case is analyzing complex, multidimensional systems of ordinal data for multi-criteria decision-making in the socio-economic and environmental sciences [ 129 ] . Partial order methodology is sometimes used as an interim step before other analytical tools are applied [ 85 ] , but posets and other lattice-theoretic methods are more generally useful throughout comparative evaluation processes [ 46 ] : binary comparisons based on a given criterion are naturally order-theoretic, so poset methodology is often better suited to comparative evaluation than purely statistical tools. Because posets can be visualized directly, additional structural results can be derived from the underlying mathematical theory of relations between objects – order relations support identifying and evaluating the most relevant objects in a study, and deriving the relative importance of criteria used in an ordinal ranking process – and the Hasse Diagram Technique (Section 7 ) offers a particularly powerful tool for such comparative evaluations [ 49 ] . Newlin and Patil [ 231 ] outline a procedure for identifying the minimal and maximal elements of a poset from its Hasse graph in complex scenarios, which is of central importance whenever a priority-setting procedure must be carried out. More generally, partially ordered sets encode a substantial amount of information about the degree of dominance among their elements, and the tools reviewed throughout this survey exist precisely to extract that information and convert it into rankings for the wide range of data-analysis purposes surveyed below. 

 
 

### 8.1 Environmental Data Analysis

 
 The concept of partially ordered sets and their visualisation by Hasse diagrams turns out to be very useful in many applications of environmental pollution data studies where evaluative considerations and assessments are important [ 160 , 130 , 74 , 77 , 76 , 78 , 278 , 302 , 303 , 304 , 50 , 44 ] . Environmental data have been extensively studied by
Hasse Diagram Technique in [ 48 , 49 , 51 , 53 ] with the goal to identify significant relationships among sediment samples (objects) and degradation indices (attributes). Further investigations using formal concept analysis [ 11 ] with the aim to highlight interaction among hygienic compounds and a synergism between toxicity tests applied to the sediment samples from surface water sources. Hasse diagram technique [ 49 ] supports the visualization of more than two-dimensional problems to identify
pollution patterns by characterizing geographical regions using their chemical pollution levels, and then suggesting priority regions for further examination. The pollution pattern which determines the location of the regions within a Hasse diagram is useful in the search for remediation strategies. In an alternative approach outlined by Restrepo Bruggemann [ 254 ] , the pollution pattern from regional studies was examined using a methodology that combined hierarchical clustering analysis(HCA) and Hasse Diagram Technique. The HCA was used for classification of the objects in the set in order to find similarity classes resulting to reduced set of object repreentatives, thereby making the Hasse diagrams for analysis depicting the network structure more easily understandable.
Posets can also be very useful when ranking environmental chemicals by multicriteria analysis [ 56 , 107 , 13 ] . Posets in this case provides a solid formal framework for the ranking of objects without assigning a common scale or weights to the criteria. In the prediction of toxicity levels of chlorobenzene in environmental chemical polychlorinated hydrocarbons, a scheme is developed by Ivanciuc et al. [ 171 ] based on poset theory and embedded within an overall reaction network. 

 
 
 

### 8.2 Socio-Economic Data Analysis

 
 Partially ordered data are prevalent in many branches of the social and behavioral sciences. This has been driven by the fact that partial order theory employed in the applied sciences overcomes the intrinsic disadvantage hidden in aggregation if a multiple attribute system is available [ 12 ] . A typical example is in response categories: “Agree”, “Neutral”, “Disagree”, and “Don’t Know”, of which the first three can be ordered and the last forms a category of its own. This type of data can be derived from a wide range of applications covering different topics such as multidimensional poverty, economic development, inequality measurement etc. Partially ordered set theory has been shown to be particularly suitable to address multi-criteria decision problems since it shows where multidimensional indicator data values are expressing a conflict which can then be identified and resolved by appropriate corrective action [ 87 ] . The use of posets with quantitative data reduces the set of operations and choices to be made in order to synthesize indicators (normalization, aggregation), even if they are the natural representation of multidimensional ordinal data [ 127 , 128 ] . Using this approach, it is possible not only to investigate the nature of the phenomenon in more detail, but also to help policy-makers in their assessments. Infact, it can be considered an effective policy tool aimed at promoting the identification of disadvantaged and under-developed profiles and contexts [ 2 ] . The application of the poset-based method provides an understanding of the complexity of the evolution of a social phenomena in terms of temporal trends and comparisons between regional structures in a countrywide basis using a multidimensional data source for the process [ 1 , 72 ] . Incomparability captures the existence of intrinsically different forms of it, thereby providing a more realistic picture of the phenomenon under investigation. In this respect, the impracticability to compare the profiles of different cities, due to the existence of dimensions where they perform in conflicting ways, reveals the irreducible complexity of sustainability. In census data on deprivation within regions of a country, poset theory has been applied because they can account for incomparabilities which are at the basis of deprivation complexity [ 17 , 120 , 121 ] . Partial order theory has been extensively applied to the construction of synthetic indicators in various socio-economic contexts [ 1 , 2 , 14 , 3 , 24 , 111 , 60 , 81 , 82 , 108 , 126 , 128 , 132 , 256 , 257 ] .
In the study of crime analytics [ 70 ] , POSAC(Partially Ordered Scalogram Analysis with Coordinates) is a graphical representation, providing dimensional reduction procedures that preserve partial order. It has been used to study the structure of criminal networks in terms of their longevity, biographical data the, network size and so forth. Levy [ 211 ] performs data analysis by employing the technique of partial order analysis of crime indicators characterizing cities. The results were visualized with a scalogram based on the method of Partial Order Structuple Analysis which was developed for non-metric data analysis. Furthermore, it is noted that the approach described in Levy [ 211 ] , can be useful in a broad range of problems on the stratification of cities, individuals, as well as several varieties of social indicators for classification. 

 
 
 

### 8.3 Neurocognition Modelling

 
 Finite partially ordered sets are natural models for cognition as it is reasonable to assume that some cognitive states have higher levels of functionality
than others [ 287 , 288 , 286 , 290 , 291 ] . Additionally, finite partially ordered classification models are useful for many statistical applications
including cognitive modelling. In particular, a data analytic framework for implementing latent finite partially ordered classification models introduced by Tatsuoka [ 287 ] , provides useful methods for evaluating cognitive applications that are latent and complex. Poset models are flexible and can become quite rich and complex, enabling them to
be effective models for describing response phenomena from educational test data or neuropsychological assessment data [ 287 , 289 , 294 ] . An objective of neuropsychological assessment is to determine differences in cognitive functioning in clinical settings. Carr et al. [ 93 ] used poset classification models of neuropsychological test data to classify samples into detailed cognitive profiles using ADNI2(Alzheimer’s Disease Neuroimaging Initiative) and AIBL(Australian Imaging, Biomarker Lifestyle ) datasets. In the risk of disease progression of Alzheimer disease for individuals with mild cognitive impairment
(MCI), Tatsuoka et al. [ 292 ] suggest that poset-based modeling methods may be useful in providing more precise classification of cognitive subgroups among MCI for imaging and genetics studies, and for developing more efficient and focused cognitive test batteries. An order structure arises naturally with skill profiles. The flexibility to not necessarily assume that one state is greater than another is an appealing feature of posets [ 290 ] . In addition, posets are comprised of states, into which cases are classified, that are associated with distinct patterns of attribute strengths and weaknesses [ 176 ] . Posets have several advantages over conventional
statistical methods for handling large numbers of polyfactorial neuropsychological test variables such as the ability to mimic the expert judgment of a clinical neuropsychologist for each case in a large sample. Posets are efficient in these tasks since as valid conclusions can be drawn based on relatively few measures, and classification of large samples can be accomplished rapidly [ 175 ] .
On the other hand, Jaeger et al. [ 175 ] noted that the primary limitation in poset modeliing of large conventional neuropsychological test datasets is based on its restriction to a selected set of attributes, while additional important distinctions remain to be tested. 

 
 
 

### 8.4 Other

 
 The application of posets and lattice theory are not limited to the aforementioned domains and tasks for data analysis. Posets have also been used in: 1) hypothesis management in large scale research projects [ 149 ] ; 2) longitudinal data analysis [ 114 ] ; 3) social macroeconomics analysis [ 94 ] ; 4) adaptive testing for cognitive assessment [ 293 ] ; 5) psychometrics [ 170 ] ; 6) medical statistics [ 273 ] ; 7) machine learning of reviewer’s paper preferences [ 97 ] ; 8) visual similarity learning using pose estimations [ 32 ] ; 9) gender analysis [ 86 ] ; 10) Pattern Mining [ 280 ] ; 11) large-scale data mining, where AI-integrated lattice and poset structures are combined with the algebraic features of these structures to process complex relational data, discover hidden patterns, and scale to large datasets [ 240 ] . A further, weaker connection to poset structure appears in compiler optimization: pass-ordering search spaces are naturally partially ordered by which optimization passes must precede others, and recent reinforcement-learning-and-graph-neural-network frameworks for multi-objective compiler phase ordering [ 235 ] implicitly search this space, although the poset structure is not the explicit focus of that line of work and we flag it here only as an adjacent application rather than a core contribution to poset theory. 

 
 
 

### 8.5 Some Selected Datasets

 
 
 1. 
 
 Computer Vision Datasets 

 
 
 
 • 
 
 The Olympic Sports dataset [ 333 ] , The Leeds Sports Pose(LSP) [ 332 ] , MPII Pose [ 331 ] . They are employed in the study of visual similarity learning from pose estimation using posets. [ 32 ] . 

 

 • 
 
 Micro-video dataset [ 100 ] . It is used for the learning problem on hypergraph partial order [ 134 ] . 

 

 
 

 2. 
 
 Natural Language Datasets : 

 
 • 
 
 The Compositional Freebase Questions (CFQ)(Keysers et al., 2020) [ 334 ] is a dataset that is specifically designed to measure compositional generalization. It is used for the study of hierarchical poset decoding for compositional generalization in language [ 152 ] . 

 

 • 
 
 Universal Dependencies (UD) corpora [ 335 ] . It is used in the learning algorithm for generating a surface word order for a sentence given
its dependency tree [ 117 ] . 

 

 
 

 3. 
 
 Sustainanble Development Data : 

 
 
 The sustainable datasets are categorized into socio-economic and environmetal datasets. 

 
 
 Socio-economic data 

 
 
 
 • 
 
 Equitable and sustainable well-being data [ 336 ] . Used in [ 1 , 3 ] for multidimensional data analysis in the context of sustainable development. 

 

 • 
 
 Service Performance Dataset [ 337 ] . Used for multidimensional data analysis in [ 12 ] . 

 

 • 
 
 Data on crime rates and their ranking for sixteen American cities is described in Levy [ 211 ] . 

 

 
 

 4. 
 
 Environmental data 

 
 
 
 • 
 
 A battery of biochemical, microbiological and bioassay tests were used to identify degraded or degrading sediments in waters [ 48 , 51 , 11 ] . The dataset is available in [ 116 ] . 

 

 • 
 
 Collection of toxicity data [ 171 ] . 

 

 • 
 
 Multiple indicator data for stream channel stability at bridge crossing is described in [ 231 ] . 

 

 
 

 5. 
 
 Other datasets : 

 
 
 
 • 
 
 (i)Data covering a time period up to the end of 2022 on the historical head-to-head matches of six professional tennis players.(ii) Data on educational testing from 15 OECD countries in.reading comprehension based on test performance in 2015 from the Programme for International Student Assessment (PISA). Both of these are described in [ 282 ] for the purpose of partial ranking of tennis players and total ranking of educational systems respectively. 

 

 • 
 
 A mass spectroscopy dataset consisting of 11 phosphoproteins and phospholipids containing approximately 854 measurements of abundance levels in an observational setting described in the supplementary material in [ 264 ] . It is used in Taeb et al. [ 282 ] for the purpose of learning causal relations and structures in proteins. 

 

 • 
 
 UCI machine learning repository [ 329 ] , used in [ 221 ] . 

 

 • 
 
 Poset/Hypergraph Currvature Datasets [ 340 ] , used in [ 319 ] to perform an empirical study involving computation and analysis of the Forman–Ricci curvature of hyperedges in 12 real-world hypergraphs. 

 

 
 

 
 
 
 

### 8.6 Selected Algorithms and Software Packages

 
 
 1. 
 
 Machine learning and Deep learning 

 
 
 
 • 
 
 Python implementation of deep visual similarity learning using posets [ 338 ] . Used in [ 32 ] . 

 

 • 
 
 Given as input an edge-weighted poset, the algorithm in [ 117 ] constructs a total order such that nodes with smallest weights are adjacent. The algorithm works by attempting to order a set of words as closely as possible to their original surface realization in the Universal Dependencies(UD) corpus. Due to the fact that words may repeat in the sentence, each order is instead represented by a list of integers, and it is these lists of integers which are compared [ 117 ] . For example, assuming a target reference order of [1,2,3] for the red horse, the generated order of red the horse would be [2,1,3]. 

 

 • 
 
 Causal Structure Discovery Algorithm-Greedy Sparsest Poset (GSPo) , python implementation in [ 339 ] . Used for causal structure learning in the presence of latent variables [ 35 ] . 

 

 • 
 
 A learning algorithm for discovering partial orders from sequences of events is described in [ 216 ] . 

 

 • 
 
 The learning algorithm of pertinent concept described in [ 221 ] uses the Adaboost algorithm with formal concept analysis on a learning dataset to discover lattice concepts used for classification rules. 

 

 • 
 
 An algorithm framework for integrating base classifier into concept node of concept lattice is described in [ 317 ] . 

 

 • 
 
 Greedy sequential algorithm for model selection described in [ 282 ] . It is used in partial ranking of tennis players, total ranking in educational systems, and causal structure learning on proteins dataset. 

 

 
 

 2. 
 
 Multidimensional data analyis : 

 
 
 The Hasse Diagram Technique (HDT) described in section 7 provides a huge collection of methods that are simple but tend to be complicated if the number of elements in the poset increases. To address this chalenge, special software packages have been developed to support the HDT. Some of these are described as follows. 

 
 
 
 • 
 
 The PARSEC [ 16 , 124 ] is an R package for poset-based evaluation of socio-economic data. Its main goal is to provide socio-economic scholars with an integrated set of elementary functions for multidimensional poverty evaluation based on ordinal information. The package is organized in four main parts 

 
 – 
 
 
 i) Data management..
 
 ii) Basic poset analysis.
 
 iii) Poset-based evaluation.
 
 iv) OPHI counting approach.
 
 
 

 
 

 • 
 
 An algorithm based on posets for multidimensional data analysis is described in [ 18 ] . 

 

 • 
 
 Self-organizing map algorithm(SOM) [ 330 ] is a popular tool for non-linear dimensionality reduction and pattern recognition. It is used for multidimensional data analysis described in [ 17 ] . 

 

 • 
 
 A non-aggregative partial order algorithm for the construction of sustainability synthetic indicators on multi-indicators systems is described by Arcagni et al. [ 18 ] . 

 

 • 
 
 An application of partial order theory for object rank correlation analysis of multiple variables is implemented by the software package PO Correlation [ 279 ] . The
design is made transparent for rank correlation analysis by a detailed mapping of the rank relations between all objects. 

 

 • 
 
 The software WHASSE is applied for chemical monitoring data analysis [ 52 , 54 , 161 ] , medical data analysis [ 112 ] . 

 

 • 
 
 PLMIX [ 228 ] an R package for modeling and clustering partially ranked data. 

 

 • 
 
 PyHasse software [ 63 , 67 , 202 ] is used for the purpose of ordinal analysis in data matrices, identification and analysis of partial order relations as well as in computing ranks. Furthermore, it has been used in [ 308 , 56 , 305 ] for the analysis and evaluation of environmental data and in [ 88 ] for multi-criteria decision analyses. 

 

 • 
 
 ProRank [ 250 ] software for partial order ranking. It is used in the evaluation of environmental databases [ 307 ] . 

 

 
 

 
 
 
 
 

## 9 Limitations, Open Problems, and Future Directions

 
 Posets have proven their usefulness in a broad range of applications. However, several theoretical and practical challenges limit the extent to which poset-based methods have been adopted in mainstream machine learning, and a number of very recent results (2025–2026) suggest concrete directions in which these limitations can be addressed. We organize this discussion around six themes, using the taxonomy of Section 2 to relate each theme to the representation, paradigm, modality, and task it primarily affects. 

 
 

### 9.1 Model Depth and Expressivity

 
 A major challenge in data science is the identification of geometric structure in high-dimensional data. The structural understanding of data is very relevant for designing efficient algorithms for optimization and machine learning. Classically, the structure of data has been studied under Euclidean assumptions: the fundamental representation of the features for any machine learning model is the vector, and its multidimensional generalization, the tensor. Many machine learning algorithms and data-analysis pipelines have accordingly been developed with the aim of computing vectors or matrices of real numbers, and a wide range of well-studied tools and algorithms assume such structure. However, many scientific fields study data with an underlying structure that can only be represented in non-Euclidean space, such as graphs and manifolds, and posets – as a class of directed acyclic graphs – have only recently begun to receive attention with regard to developing appropriate deep learning architectures rather than shallow order-theoretic statistics. Wendler [ 311 ] introduced methods for Fourier-sparse learning on data indexed by lattices and posets, and the graph neural network class of algorithms applies to posets as discussed in Section 4 , but there is not yet widespread use of graph neural networks relative to the number of applications based on posets and lattices. A concrete recent step toward genuinely deep, multi-layer poset-structured architectures is the poset-pooling construction of Dolores-Cuenca et al. [ 239 ] , who show that convolutional filters derived from four-point posets and their associated order polytopes update network weights during backpropagation with greater precision than average, max, or mixed pooling, without introducing additional trainable parameters, and who formalize the composition of such poset-neural architectures using the algebra of the operad of posets. This raises an open question of central importance for model depth: how does the expressivity of a poset-structured network (in the sense of, e.g., function classes realizable by stacking poset-pooling or poset-projection layers) scale with network depth, and what is lost or gained relative to unconstrained architectures when the intermediate representations are themselves required to respect a partial order? The success of machine learning algorithms generally depends on data representation, since different representations encode different explanatory factors of variation behind the data [ 34 , 113 ] . There is accordingly enormous scope to develop deeper, more expressive architectures for learning the structural characteristics of data represented as Hasse graphs of lattices and posets, together with model-evaluation metrics designed specifically for poset-valued outputs. 

 
 
 

### 9.2 Scalability–Fidelity Trade-offs

 
 Many of the classical poset-analytic techniques surveyed in Sections 6 – 7 – enumeration of linear extensions, exact computation of mutual ranking probabilities, exhaustive Hasse-diagram construction – are worst-case exponential in the number of incomparable elements, which is precisely the situation in which poset methods are most informative relative to a forced total order. Sampling-based approximations such as Markov Chain Monte Carlo over linear extensions [ 245 , 69 ] trade exactness for scalability, but the bias–variance behavior of these approximations for downstream machine learning tasks (as opposed to descriptive ranking) is not well characterized. A related, and largely open, scalability question concerns distance and metric computation over posets, which underlies nearest-neighbor, clustering, and kernel-based learning on order-structured data. Two 2026 contributions are directly relevant here. Taeb, Guo, and Henckel [ 237 ] propose a model-oriented framework in which each graph is treated as a statistical model organized in a poset by inclusion, and define a distance as the length of a shortest path through poset neighbors; this framework applies to probabilistic undirected graphs and to several classes of causal DAGs, but its computational cost scales with the size of the neighborhood structure of the poset, which is not addressed in closed form for large model spaces. Olave [ 233 ] introduces a family of extended metrics on path-connected and fence-connected posets that do not require additional algebraic structure (e.g., a valuation), showing that these metrics characterize discrete path-connected posets up to isomorphism and coincide with interleaving distances when posets are viewed as thin categories; this is a promising building block for scalable metric learning on posets, but its practical computational complexity for large, densely comparable posets has not yet been benchmarked. Understanding when a cheaper, approximate poset metric preserves the statistical or predictive guarantees of the exact one – analogous to Johnson–Lindenstrauss-type results for Euclidean embeddings – remains an open problem. 

 
 
 The same exponential-versus-polynomial tension recurs, in sharper form, in two further problems that have not previously been connected to this scalability discussion: generation and isomorphism testing of posets and lattices at scale. On the generation side, exact enumeration algorithms produce a complete, duplicate-free catalog of all lattices of a given size, but scale in the size of the output itself – which grows from 5,994 5{,}994 non-isomorphic lattices at n = 10 n=10 to over 23 23 trillion at n = 20 n=20 (OEIS A006966) – confining exhaustive approaches to n ≲ 24 n\lesssim 24 in practice. Mwafise’s reinforcement-learning generator [ 283 ] (Section 4.2 ) instead trades the completeness guarantee of exact enumeration for a sampling procedure that scales in the size of a single candidate rather than the size of the target class, extending practical lattice discovery to n ∈ { 30 , 40 , 50 } n\in\{30,40,50\} with mean discovery rates of 60 60 – 80 % 80\% ; the two approaches are explicitly complementary rather than competing, with exhaustive enumeration remaining preferable whenever a complete catalog at moderate n n is the actual goal. On the isomorphism-testing side, the general graph isomorphism problem is GI-complete and not known to admit a polynomial-time solution, and the classical poset-isomorphism solvers used throughout the descriptive and clustering literature of this survey (VF2-style backtracking) inherit this worst-case factorial cost. Mwafise’s hierarchical matrix-decomposition framework [ 284 ] (Section 4.11 ) instead achieves empirically polynomial ( O ⁡ ( n 2 ) O(n^{2}) to O ⁡ ( n 4 ) O(n^{4}) , depending on structural regularity) isomorphism testing for a broad class of posets by recursively decomposing the poset matrix rather than searching over relabelings, with formally proved invariance under permutation. These two results are best read together: a scalable generator is only as useful as its ability to cheaply determine that a newly generated candidate is not a duplicate of one already found, which is exactly the isomorphism-testing bottleneck the decomposition framework targets, and we return to this generate-and-canonicalize relationship as a concrete open research direction in Section 9.7 . 

 
 
 

### 9.3 Heterogeneity of Order-Structured Data

 
 Most poset-based methods surveyed in this paper assume a single, homogeneous order relation over a single population of objects. Real applications, however, frequently combine several heterogeneous and only partially compatible order relations: for example, multiple stakeholders may impose different priority orders over the same set of safety constraints, or a multi-indicator system may need to merge orders derived from measurements of different types and reliabilities (Section 6 ). Wong, Xiao, and Rus [ 238 ] address a version of this problem directly in the setting of safe reinforcement learning: rather than imposing a single fixed priority order or enforcing all safety constraints uniformly, they formalize safety requirements as a poset in which some constraints are comparable and others are genuinely incomparable, and introduce PoSafeNet, a differentiable neural safety layer that performs sequential closed-form projection consistent with the poset structure, enabling adaptive selection or mixing of valid safety executions while preserving priority semantics by construction. This is, to our knowledge, the first neural architecture designed explicitly around the possibility that constraints are only partially ordered rather than either freely combinable or strictly ranked, and it suggests a broader research direction: extending poset-structured layers to settings with heterogeneous, possibly conflicting orders supplied by different sources (annotators, sensors, stakeholders), and developing principled methods for reconciling or completing such orders, in the spirit of the completion results for leveled posets discussed below. 

 
 
 

### 9.4 Dynamicity of Evolving Posets

 
 The vast majority of the theory and algorithms surveyed in this paper – Hasse diagram construction, linear extension counting, formal concept lattices, poset-based ranking – is formulated for a static poset. Many of the application domains reviewed in Sections 6 and 8 , however, are inherently temporal: multi-indicator systems evolve year over year [ 1 ] , safety constraints in control systems can activate or deactivate along a trajectory [ 238 ] , and streaming or additive-manufacturing data is generated incrementally layer by layer. Fueyo et al. [ 119 ] introduce leveled partially ordered sets to manage exactly this kind of layer-indexed data, motivated by an industrial 3D-printing application in which points must be ordered layer by layer; they generalize Birkhoff’s notion of element height to the leveled setting and prove that, under conditions related to the Jordan–Dedekind chain condition, a leveled poset can always be completed into a bounded lattice purely by adding links to the existing order relation, without introducing new elements. This is a rare example of a completion result that could plausibly be adapted to online or streaming settings, where new layers (or new time steps) arrive incrementally and the poset – and any downstream lattice used for analysis – must be updated rather than recomputed from scratch. Formalizing and analyzing such incremental poset-update algorithms, together with their approximation guarantees relative to full recomputation, remains largely open. 

 
 
 

### 9.5 Posets, Category Theory, and Multiparameter Topological Data Analysis

 
 Topological data analysis (TDA) [ 95 ] is a fast-growing field providing topological and geometric tools to infer relevant features from complex data. In (single-parameter) persistent homology, the shape of a dataset is often encoded into a system of vector spaces and linear maps over a totally ordered set, and a growing body of literature connects posets more generally to topological data analysis [ 181 , 73 , 41 , 296 ] . The genuinely hard case is multiparameter persistence, where the indexing structure is a poset (typically a lattice) rather than a chain, and modules over this poset generally fail to decompose into simple interval summands, unlike in the single-parameter case. Hem [ 154 ] introduces poset cocalculus , a variant of functor calculus defined for functors from a poset (or, more specifically, a distributive lattice) to a model category, and shows that it produces stable degree- n n approximations of a functor under an appropriate interleaving distance. In a companion paper, Hem [ 155 ] applies poset cocalculus to prove that a pointwise finite-dimensional bipersistence module is middle-exact if and only if it is isomorphic to the homology of a homotopy-degree-1 functor, yielding a new, more synthetic proof of the interval decomposability of middle-exact bipersistence modules, and gives a decomposition theorem expressing a middle-exact multipersistence module as a direct sum of a projective, an injective, and a bidegree-1 module. These results connect the purely combinatorial poset structures used throughout this survey to the algebraic and homotopy-theoretic machinery of functor calculus, and, together with the algebraic/operadic perspective on posets developed by Arciniega-Nevárez, Berghoff, and Dolores-Cuenca [ 15 ] (who study posets as algebras over an operad and use this framework to analyze the combinatorics of a nontrivial suboperad, the Wixárika posets), point toward a category-theoretic reformulation of several of the ranking and clustering constructions in Sections 5 – 7 . There is more generally a need to study multiway interactions within data endowed with a poset structure, for example by leveraging concepts such as Latent Topology Inference (LTI) [ 31 ] , and to connect the resulting topological deep learning models [ 159 ] to the functor-calculus formalism above. 

 
 
 A second, complementary operadic perspective targets the matrix representation of posets directly rather than the poset as an abstract order relation. Cheon, Choi, Giraudo, and Mwafise [ 285 ] define a family of partial composition operations on poset matrices that build larger poset matrices from smaller ones, prove that three of these operations satisfy the operad axioms – giving the collection of poset matrices itself a genuine operad structure, distinct from and complementary to the Wixárika suboperad of [ 15 ] – characterize the corresponding dual operations, and use the resulting framework to address open questions in poset enumeration. Notably, the same operadic vocabulary reappears independently, and in an entirely algorithmic register, in the isomorphism-testing framework of Section 4.11 : Mwafise [ 284 ] explicitly frames the Hierarchical Poset Matrix Tree(HPMT) as a combinatorial invariant developed for the efficient resolution of the poset isomorphism problem. The HPMT Tree can also be viewed within the broader framework of machine learning over combinatorial tree structures, a growing area spanning structural learning and geometric deep learning. Its recursive boundary-stripping and core-extraction procedure exploits the “operadic properties of poset matrices [ 314 ] ” by decomposing a poset into hierarchically composed substructures. This suggests that its algorithmic tree decomposition may provide a natural bridge between operadic composition and learnable representations of hierarchical poset structure– is both plausible and currently unestablished. 

 
 
 

### 9.6 Safe and Constrained Learning, and Order-Theoretic Reinforcement Learning

 
 A further theme that has not previously been surveyed in connection with posets is the use of order-theoretic formalisms in reinforcement learning and control. In addition to the safety-layer work of Wong, Xiao, and Rus discussed above [ 238 ] , Sargent and Stachurski [ 236 ] represent a dynamic program as a family of operators acting on a partially ordered set and give an optimality theory based purely on order-theoretic assumptions, covering applications ranging from traditional Markov decision processes to nonlinear recursive preferences, robustness objectives, and distributional dynamic programming. Peng, Stachurski, and Yang [ 234 ] extend this line of work by pairing the order-theoretic approach with topological and metric foundations, showing that readily verifiable forms of topological stability (global stability and contractivity of the policy operators) deliver both the fundamental optimality properties of dynamic programming and convergence of value function iteration, Howard policy iteration, and optimistic policy iteration, with applications including optimal stopping and Bayesian sequential analysis. Together with the classical machine-learning-and-formal-concept-analysis material of Section 4.12 , this suggests that order theory can serve not only as a data-analytic and representational tool, as in most of this survey, but also as a foundational framework for the algorithmic and convergence theory of reinforcement learning itself – an angle that, to our knowledge, has not previously been connected to the broader poset-and-machine-learning literature surveyed here. A more indepth review of existing and new models for machine learning using formal concept approaches, and of graph-based learning-to-rank methods [ 144 , 118 ] applied to datasets with an underlying partial order (Sections 6 and 7 ), would also be a useful complement to the material collected in this survey. 

 
 
 

### 9.7 Toward a Generate–Canonicalize–Compose Pipeline for Large-Scale Poset Discovery

 
 Collectively, the reinforcement-learning generator [ 283 ] , the hierarchical isomorphism-decomposition framework [ 284 ] , and the operad of poset matrices [ 285 ] discussed across Sections 4.2 , 4.11 , and 9 above chain naturally into a three-stage pipeline that, to our knowledge, has not been proposed as such in the existing literature: generate candidate posets or lattices at scale beyond the reach of exact enumeration; canonicalize them cheaply via polynomial-time isomorphism testing, deduplicating discoveries and enabling large-scale indexing; and compose validated primitives into larger structures, or decompose known large lattices into their generating components, using a formal operad. Each stage already exists in isolation, but their combination suggests concrete extensions beyond what any one paper claims. First, canonicalization is not merely a downstream convenience for a generator: the discovery rates reported by the RL framework are defined over labeled structures, precisely because no fast isomorphism-reduction step was integrated into its training loop; wiring the hierarchical decomposition framework’s near-linear Tier-1/Tier-2 checks (Section 4.11 ) directly into the novelty bonus of the generator’s reward function would convert a labeled discovery rate into an isomorphism-aware coverage measure – a gap the RL paper’s own stated future work identifies as open, alongside its separately stated interest in “integration of graph-based machine learning techniques for enhanced structural analysis.” Second, the operad of poset matrices offers a principled, order-theoretically legal action space: rather than the free pairwise edge choices of the current generator, an RL agent could instead be restricted to operadic composition moves, which would guarantee every intermediate and terminal candidate is a well-formed poset matrix by construction rather than by post hoc verification, potentially easing the combinatorial sparsity that motivates the current framework’s exact-verification-oracle design (Section 4.2 ). Third, beyond generation and testing, this same toolchain is a plausible basis for constructing the first large-scale, deduplicated benchmark corpus of labeled and canonicalized posets and lattices, of the kind needed to systematically evaluate the poset-pooling architectures, poset-valued metrics, and safety layers surveyed throughout Sections 4 – 4.11 – none of which currently have access to anything resembling a standardized, large-scale evaluation dataset. We flag all three directions as concrete, currently unrealized extensions of existing work rather than results established by the papers cited. 

 
 
 

### 9.8 Summary of Open Problems

 
 Table 2 summarizes the open problems discussed above together with the taxonomy axis (Table 1 ) each one primarily stresses, to make explicit that the current literature is comparatively dense along the representation and task axes but comparatively sparse along the heterogeneity and dynamicity dimensions of the learning-paradigm and data-modality axes. 

 
 
 Table 2: Open problems in poset-based machine learning and the taxonomy axis each primarily stresses. 
 
 
 
 
 
 Open problem 
 | 
 
 
 Description 
 | 
 
 
 Taxonomy axis stressed 
 | 

 
 
 
 
 
 Model depth / expressivity 
 | 
 
 
 Scaling poset-structured layers (e.g. poset pooling) to deep, multi-layer architectures 
 | 
 
 
 Representation 
 | 

 
 
 
 Scalability–fidelity trade-off 
 | 
 
 
 Approximate linear-extension sampling, poset metrics, and RL-based lattice generation at scale 
 | 
 
 
 Task (metric computation, generation) 
 | 

 
 
 
 Heterogeneity 
 | 
 
 
 Reconciling multiple, only partially compatible order relations (safety, multi-stakeholder) 
 | 
 
 
 Learning paradigm 
 | 

 
 
 
 Dynamicity 
 | 
 
 
 Online/incremental posets and lattice completions for streaming or layered data 
 | 
 
 
 Data modality 
 | 

 
 
 
 Category-theoretic / topological integration 
 | 
 
 
 Poset cocalculus and operadic structure (of posets and of poset matrices) for multiparameter TDA 
 | 
 
 
 Representation 
 | 

 
 
 
 Order-theoretic RL 
 | 
 
 
 Convergence theory for dynamic programming, and generation, posed directly over partial orders 
 | 
 
 
 Learning paradigm 
 | 

 
 
 
 Generate–canonicalize–compose pipeline 
 | 
 
 
 Coupling RL-based generation, polynomial-time isomorphism testing, and operadic composition into a unified, benchmark-ready toolchain 
 | 
 
 
 Task (generation, isomorphism testing) 
 | 

 

 
 
 
 

## 10 Conclusion

 
 This survey has provided a comprehensive, taxonomy-organized review of the use of partially ordered sets in machine learning and data analysis. The breadth of application of poset theory across machine learning and data analysis – from ranking and clustering to formal concept analysis, safe reinforcement learning, and multiparameter topological data analysis – demonstrates its continued relevance. Building on an extensive account of the classical literature on descriptive, multidimensional, and exploratory poset-based data analysis, we incorporated a substantial body of methods published in 2025–2026 that had not previously been connected to this literature, including poset-structured safety layers for reinforcement learning [ 238 ] , order-theoretic formulations of abstract dynamic programming [ 236 , 234 ] , new families of metrics over posets [ 237 , 233 ] , poset cocalculus for multiparameter persistence [ 154 , 155 ] , two complementary operadic/algebraic perspectives on poset combinatorics [ 15 , 285 ] , leveled posets for layer-indexed and streaming data [ 119 ] , poset-pooling neural architectures for interpretable deep learning [ 239 ] , reinforcement-learning-based generation of finite lattices and semilattices [ 283 ] , and a polynomial-time hierarchical matrix-decomposition framework for poset isomorphism testing [ 284 ] . This last pair – a generator that samples posets at scale beyond the reach of exact enumeration, and a canonicalization procedure that tests isomorphism far faster than classical backtracking – suggested a concrete, currently unrealized generate–canonicalize–compose pipeline (Section 9.7 ) that we believe is a promising direction for building the large-scale, deduplicated poset benchmarks this field currently lacks. Organizing this material along the four-axis taxonomy introduced in Section 2 makes clear that the field is comparatively mature along the representation and task axes but still comparatively underdeveloped with respect to heterogeneous and dynamically evolving posets, and with respect to the expressivity of deep, multi-layer poset-structured architectures – gaps that we outlined as a concrete research agenda in Section 9 . We aim for this survey to provide both a foundational reference for researchers newly entering this area and a practical map of open problems for those already working at the intersection of order theory and machine learning. 

 
 
 

## References

 
 
 [1] 
 Alaimo, L.S., Arcagni, A., Fattore, M. et al. Synthesis of Multi-indicator System Over Time: A Poset-based Approach. Soc Indic Res 157, 77–99 (2021).
 
 

 
 [2] 
 Alaimo, L.S., Ciacci, A. Ivaldi, E. Measuring Sustainable Development by Non-aggregative Approach. Soc Indic Res 157, 101–122 (2021).
 
 

 
 [3] 
 L. S. Alaimo F. Maggino, 2020. “Sustainable Development Goals Indicators at Territorial Level: Conceptual and Methodological Issues–The Italian Perspective,” Social Indicators Research: An International and Interdisciplinary Journal for Quality-of-Life Measurement, Springer, vol. 147(2), pages 383-419.
 
 

 
 [4] 
 A. Albuquerque, T. Amador, R. Ferreira, A. Veloso and N. Ziviani, “Learning to Rank with Deep Autoencoder Features,” 2018 International Joint Conference on Neural Networks (IJCNN), Rio de Janeiro, Brazil, 2018, pp. 1-8, doi: 10.1109/IJCNN.2018.8489646 .
 
 

 
 [5] 
 M. A. Ali, A. Jaoua and S. A. Al-Maadeed, “A novel Conceptual Machine Learning Method using Random Conceptual Decomposition,” 2020 IEEE International Conference on Informatics, IoT, and Enabling Technologies (ICIoT), Doha, Qatar, 2020, pp. 18-22.
 
 

 
 [6] 
 A An., Learning Classification Rules from Data, (2003) http://www.cs.yorku.ca/~aan/research/paper/cam03.pdf .
 
 

 
 [7] 
 Anđelić, M., da Fonseca, C.M. Cover matrices of posets and their spectra. Czech Math J 59, 1077–1085 (2009). https://doi.org/10.1007/s10587-009-0075-6 
 
 

 
 [8] 
 Andrews, S.: Data Conversion and Interoperability for FCA. In: CS-TIW 2009, pp. 42-49, http://www.kde.cs.uni-kassel.de/ws/cs-tiw2009/proceedings_final_15July.pdf 
 
 

 
 [9] 
 S. Andrews and C. Orphanides, “Analysis of large data sets using
formal concept lattices,” In: M. Kryszkiewicz and S. Obiedkov, (eds.) Proceedings of the 7th International Conference on Concept Lattices and Their Applications . Seville, University of Seville, 2010, 104-115.
 
 

 
 [10] 
 Andrews, S., Polovina, S., Visualising computational intelligence through converting data into formal concepts(2011), https://shura.shu.ac.uk/2720/ .
 
 

 
 [11] 
 P. Annoni, R. Bruggemann, The dualistic approach of FCA: A further insight into Ontario Lake sediments, Chemosphere 70 (2008) 2025–2031.
 
 

 
 [12] 
 Annoni, P., Brüggemann, R. Exploring Partial Order of European Countries. Soc Indic Res 92, 471–487 (2009).
 
 

 
 [13] 
 P. Annoni, R. Brüggemann, A. Saltelli, Partial order investigation of multiple indicator systems using variance-based sensitivity analysis,
 Environmental Modelling Software , Vol. 26; 7, 2011, pp. 950-958, https://doi.org/10.1016/j.envsoft.2011.01.008 
 
 

 
 [14] 
 Annoni, P., Fattore, M. Bruggemann R. (2011). A Multi-Criteria Fuzzy Approach for Analyzing Poverty structure. Statistica Applicazioni, Special
Issue, 7-30.
 
 

 
 [15] 
 Arciniega-Nevárez, J.A., Berghoff, M., Dolores-Cuenca, E. (2026). Operad of posets 101: The Wixárika posets. Journal of Prime Research in Mathematics , 22(1), 29–43. arXiv:2406.07370.
 
 

 
 [16] 
 Arcagni, A., Fattore, M. (2014). PARSEC: An R package for poset-based evaluation of multidimen-sional poverty. In R. Bruggemann, L. Carlsen, J. Wittmann (Eds.), Multi-indicator systems andmodelling in partial order . Berlin: Springer.
 
 

 
 [17] 
 Arcagni, A., Barbiano di Belgiojoso, E., Fattore, M. et al. Multidimensional Analysis of Deprivation and Fragility Patterns of Migrants in Lombardy, Using Partially Ordered Sets and Self-Organizing Maps. Soc Indic Res 141, 551–579 (2019).
 
 

 
 [18] 
 Arcagni A., Cavalli L., and Fattore, M., Partial Order Algorithms for the Assessment of Italian Cities Sustainability (February 3, 2021). FEEM Working Paper No. 1.2021, Available at https://ssrn.com/abstract=3778559orhttp://dx.doi.org/10.2139/ssrn.3778559 
 
 

 
 [19] 
 Arcagni, A., A. Avellone, and M. Fattore (2022). Complexity reduction and approximation of multidomain systems of partially ordered data. Computational Statistics Data Analysis 173 , 107520.
 
 

 
 [20] 
 Avogadri, R., Valentini, G. (2008). Ensemble Clustering with a Fuzzy Approach. In: Okun, O., Valentini, G. (eds) Supervised and Unsupervised Ensemble Methods and their Applications. Studies in Computational Intelligence, vol 126. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-78981-9_3 
 
 

 
 [21] 
 N. A. Asif et al.,“Graph Neural Network: A Comprehensive Review on Non-Euclidean Space,” in IEEE Access , vol. 9, pp. 60588-60606, 2021, doi: 10.1109/ACCESS.2021.3071274 .
 
 

 
 [22] 
 M. A. Babin and S. O. Kuznetsov, Enumerating Minimal Hypotheses and Dualizing Monotone Boolean Functions on Lattices
 
 

 
 [23] 
 J. Bachtrögler, H. Badinger, A. F. de Clairfontaine, and W. H.Reuter, “Summarizing data using partially ordered set theory: anapplication to fiscal frameworks in 97 countries,” Stat. J. IAOS , vol.32, no. 3, pp. 383–402
 
 

 
 [24] 
 Badinger H., Reuter W. H. (2015). Measurement of Fiscal Rules: Introducing the Application of Partially Ordered Set (POSET) Theory. Journal of Macroeconomics, 43, 108–23.
 
 

 
 [25] 
 R. A. Bailey (2021) Hasse diagrams as a visual aid for linear models and analysis of variance, Communications in Statistics - Theory and Methods, 50:21, 5034-5067, DOI: 10.1080/03610926.2019.1676443
 
 

 
 [26] 
 Bain, M. (2003). Inductive Construction of Ontologies from Formal Concept Analysis. In: Gedeon, T.(.D., Fung, L.C.C. (eds) AI 2003: Advances in Artificial Intelligence. AI 2003. Lecture Notes in Computer Science(), vol 2903. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-24581-0_8 .
 
 

 
 [27] 
 S. Barbarossa and S. Sardellitti, “Topological signal processing over simplicial complexes,” IEEE Transactions on Signal Processing, 2020. 
 
 

 
 [28] 
 Barbut, M. and Monjardet, B., Ordre et classification, II, Paris: Hachette, 1970.
 
 

 
 [29] 
 Barthélemy, J.P., Flament, C., Monjardet, B. (1982). Ordered Sets and Social Sciences. In: Rival, I. (eds) Ordered Sets. NATO Advanced Study Institutes Series, vol 83. Springer, Dordrecht.
 
 

 
 [30] 
 S. M. Basha, D. S. Rajput, Survey on Evaluating the Performance of Machine Learning Algorithms: Past Contributions and Future Roadmap, https://doi.org/10.1016/B978-0-12-816718-2.00016-6 .
 
 

 
 [31] 
 Battiloro C. et al, From Latent Graph to Latent Topology Inference: Differentiable Cell Complex Module, arXiv:2305.16174v2.
 
 

 
 [32] 
 M.A. Bautista, A. Sanakoyeu, B. Ommer, Deep Unsupervised Similarity Learning using Partially Ordered Sets, 2017 IEEE Conference on Computer Vision and Pattern Recognition(CVPR) , Honolulu HI USA 2017,pp. 1923–1932, doi: 10.1109/CVPR.2017.208.
 
 

 
 [33] 
 Belohlavek, R., Sklenar, V., Zacpal, J. (2004). Formal concept analysis with hierarchically ordered attributes. International Journal of General Systems, 33(4), 383–394. https://doi.org/10.1080/03081070410001679715 
 
 

 
 [34] 
 Y. Bengio, A. Courville, P. Vincent†,(2014) Representation Learning: A Review and New Perspectives: arXiv:1206.5538v3.
 
 

 
 [35] 
 Bernstein, D.I.; Saeed, B.; Squires, C.;Uhler, C.; Ordering-Based Causal Structure Learning in the Presence of Latent Variables, Proceedings of the 23 rd {}^{\text{rd}} International Conference on Artificial Intelligence and Statistics (AISTATS) 2020, Palermo, Italy. PMLR: Volume 108.
 
 

 
 [36] 
 Bertrand, M.F. Janowitz, Pyramids and weak hierarchies in the ordinal model for clustering, Discrete Applied Mathematics, Vol. 122; 1–3, 2002, pp. 55-81,
 https://doi.org/10.1016/S0166-218X(01)00354-7 .
 
 

 
 [37] 
 Bi, W. ; Kwok, J.. (2013). Efficient Multi-label Classification with Many Labels. Proceedings of the 30th International Conference on Machine Learning , in Proceedings of Machine Learning Research 28(3):405-413 Available from https://proceedings.mlr.press/v28/bi13.html. 
 
 

 
 [38] 
 G. Birkhoff, Lattice Theory, American Mathematical Society, Vol. 25; 3 1967.
 
 

 
 [39] 
 Blocher, H., Schollmeyer, G., Jansen, C. (2022). Statistical Models for Partial Orders Based on Data Depth and Formal Concept Analysis. In: Ciucci, D., et al. Information Processing and Management of Uncertainty in Knowledge-Based Systems. IPMU 2022. Communications in Computer and Information Science, vol 1602. Springer, Cham. https://doi.org/10.1007/978-3-031-08974-9_2 .
 
 

 
 [40] 
 H. Blocher, G. Schollmeyer, C. Jasen, M. Nalenz, Depth Functions for Partial Orders with a Descriptive Analysis of Machine Learning Algorithms, Proceedings of Machine learning Research 215: 59–71, 2023.
 
 

 
 [41] 
 MB Botnan, Topological Data Analysis, Lecture Notes, https://www.few.vu.nl/~botnan/lecture_notes.pdf 
 
 

 
 [42] 
 G. Brauwers and F. Frasincar, ”A General Survey on Attention Mechanisms in Deep Learning” in IEEE Transactions on Knowledge Data Engineering , vol. 35, no. 04, pp. 3279-3298, 2023. doi: 10.1109/TKDE.2021.3126456 .
 
 

 
 [43] 
 Brisaboa, N.R., Cerdeira-Pena, A., López-López, N., Navarro, G., Penabad, M.R., Silva-Coira, F. (2016). Efficient Representation of Multidimensional Data over Hierarchical Domains. In: Inenaga, S., Sadakane, K., Sakai, T. (eds) String Processing and Information Retrieval. SPIRE 2016. Lecture Notes in Computer Science, vol 9954. Springer, Cham. https://doi.org/10.1007/978-3-319-46049-9_19 .
 
 

 
 [44] 
 Brüggemann, R., Münzer, B., Halfon, E. (1994). An algebraic/graphical tool to compare ecosystems with respect to their pollution. The German River Elbe as an example. I : Hasse-diagrams. Chemosphere , 28, 863-872.
 
 

 
 [45] 
 R. Bruggemann, J. Schwaiger, R. D. Negele, Applying Hasse diagram technique for the evaluation of toxicological fish tests, Chemosphere 30 (1995) 1767–1780.
 
 

 
 [46] 
 Bruggemann, R. and K. Voigt (1995). “An Evaluation of Online Databases by Methods of Lattice Theory.” Chemosphere 31: 3585-3594.
 
 

 
 [47] 
 Bruggemann, R.; Oberemm, A.; Steinberg, C. Ranking of Aquatic Effect Tests Using Hasse Diagrams. Toxicol. EnViron. Chem. 1997, 63, 125-139.
 
 

 
 [48] 
 Bruggemann, R.; Halfon, E. Comparative Analysis of Nearshore
Contaminated Sites in Lake Ontario: Ranking for Environmental Hazard. J. EnViron. Sci. Healt h 1997, A32(1), 277-292.
 
 

 
 [49] 
 R. Bruggemann, S. Pudenz, K. Voigt, A. Kaune, K. Kreimes, An algebraic/graphical tool to compare ecosystems with respect to their pollution. IV: Comparative regionalanalysis by Boolean arithmetics, Chemosphere 38 (1999) 2263–2279.
 
 

 
 [50] 
 Brüggemann, R. and H.-G. Bartel (1999) A Theoretical Concept to Rank Environmentally Significant Chemicals. J.Chem.Inf.Comp.Sc. 39, 211-217 .
 
 

 
 [51] 
 R. Bruggemann, E. Halfon, G. Welzl, K. Voigt, C. Steinberg, Applying the concept of partially ordered sets on the ranking of near-shore sediments by a battery of tests, J. Chem. Inf. Comp. Sci. 41 (2001) 918–925.
 
 

 
 [52] 
 R. Bruggemann, U. Simon, S. Mey, Estimation of averaged ranks by extended local partial order models, MATCH Commun. Math. Comput. Chem. 54 (2005) 489–518.
 
 

 
 [53] 
 Brüggemann, R.; Carlsen, L. Partial Order in Environmental Sciences and Chemistry; Springer, 2006.
 
 

 
 [54] 
 R. Bruggemann, G. Restrepo, K. Voigt, Structure–fate relationships of organic chemicals derived from the software packages E4CHEM and WHASSE, J. Chem. Inf.Model . 46 (2006) 894–902.
 
 

 
 [55] 
 R. Bruggemann, K. Voigt, Basic principles of Hasse diagram technique in chemistry, Comb. Chem. High Throughput Screen . 11 (2008) 756–769.
 
 

 
 [56] 
 R. Bruggemann, K. Voigt, G. Restrepo, U. Simon, The concept of stability fields and hot spots in ranking of environmental chemicals, J. Environ. Model. Soft. 23 (2008) 1000–1012.
 
 

 
 [57] 
 R. Bruggemann, K. Voigt, Analysis of partial orders in environmental systems apply-ing the new software PyHasse, in: J. Wittmann, M. Flechsig (Eds), Simulation inUmwelt- und Geowissenschaften- Workshop Potsdam 2009, Shaker–Verlag, Aachen, 2009, pp. 43–55.
 
 

 
 [58] 
 Bruggemann, R., Patil, G.P. Multicriteria prioritization and partial order in environmental sciences. Environ Ecol Stat 17, 383–410 (2010). https://doi.org/10.1007/s10651-010-0167-3 
 
 

 
 [59] 
 R. Bruggemann, L. Carlsen, An improved estimation of averaged ranks of partial orders, MATCH Commun. Math. Comput. Chem . 65 (2011) 383–414.
 
 

 
 [60] 
 Bruggemann R, Patil G.P. Ranking and prioritization for multi-indicator systems—Introduction to partial order applications.Springer; 2011.
 
 

 
 [61] 
 Brüggemann, R. and K. Voigt (2011). “A New Tool to Analyze Partially Ordered Sets. Application: Ranking of Polychlorinated Biphenyls and Alkanes/Alkenes in River Main, Germany.” MATCH: Communications in Mathematical and in Computer Chemistry 66: 231–251.
 
 

 
 [62] 
 R. Bruggemann, L. Carlsen, Incomparable: What now II? Absorption of incomparabilities by a cluster method, Quality Quantity 49(4), (2014).
 
 

 
 [63] 
 Brüggemann, R.; Carlsen, L.; Voigt, K.; Wieland, R. PyHasse Software for Partial Order Analysis: Scientific Background andDescription of Selected Modules. In Multi-Indicator Systems and Modelling in Partial Order ; Springer: New York, NY, USA, 2014; pp. 389–423.
 
 

 
 [64] 
 Bruggemann, R.; Carlsen, L. Incomparable—What now? MATCH Commun. Math. Comput. Chem . 2014 , 71, 694–716.
 
 

 
 [65] 
 Bruggemann, R. and Carlsen, L. An attempt to Understand Noisy Posets. MATCH Commun.Math.Comput.Chem., 2016 , 75, 485-510.
 
 

 
 [66] 
 Bruggemann, R. et al.(Eds.)(2021). Measuring and Understanding Complex Phenomena: Indicators and Their Analysis in Different Scientific Fields . London, UK, Springer Nature.
 
 

 
 [67] 
 Bruggemann, R., Kerber, A., Koppatz, P., Pratz, V. (2021). PyHasse, a Software Package for Applicational Studies of Partial Orderings. In: Bruggemann, R., Carlsen, L., Beycan, T., Suter, C., Maggino, F. (eds) Measuring and Understanding Complex Phenomena. Springer, Cham. https://doi.org/10.1007/978-3-030-59683-5_18 .
 
 

 
 [68] 
 Brunsdon, C. Rank Inadequacy: A Partially Ordered Set Approach For Multivariate Data Analysis. https://huckg.is/gisruk2017/GISRUK_2017_paper_31.pdf .
 
 

 
 [69] 
 Bubley R, Dyer M (1999) Faster random generation of linear extensions. Discrete Math 201(1–3):81–88.
 
 

 
 [70] 
 Canter, D. A Partial Order Scalogram Analysis of Criminal Network Structures. Behaviormetrika 31, 131–152 (2004). https://doi.org/10.2333/bhmk.31.131 
 
 

 
 [71] 
 Cao et al., Learning to Rank: From Pairwise Approach to Listwise Approach, ICML ’07: Proceedings of the 24th international conference on Machine learningJune 2007Pages 129–136 https://doi.org/10.1145/1273496.1273513 .
 
 

 
 [72] 
 Caperna, G., Boccuzzo, G. (2018). Use of poset theory with big datasets: A new proposal applied to the analysis of life satisfaction in Italy. Social Indicators Research , 136, 1071–1088.
 
 

 
 [73] 
 Caputi, L., Collari, C. Di Trani, S. Combinatorial and topological aspects of path posets, and multipath cohomology. J Algebr Comb 57, 617–658 (2023). https://doi.org/10.1007/s10801-022-01180-9 .
 
 

 
 [74] 
 L. Carlsen, Partial order ranking of organophosphates with special emphasis on nerve agents, MATCH Commun. Math. Comput. Chem . 54 (2005) 519–534.
 
 

 
 [75] 
 Carlsen L. Assessment of chemicals applying partial order ranking techniques. Comb Chem High Throughput Screen. 2008 Dec;11(10):794-805. doi: 10.2174/138620708786734280. PMID: 19075601
 
 

 
 [76] 
 Carlsen, R. Bruggemann, Partial order ranking as a tool in environmental impact assessment. PAH and PCB pollution of the river Main as an illustrative example, in:G. T. Halley, Y. T. Fridian (Eds.), Environmental Impact Assessment , Nova Science Publishers, 2009, pp. 335–354.
 
 

 
 [77] 
 Carlsen, B. N. Kenessov, S. B. Batyrbekova, A QSAR/QSTR study on the human health impact of the rocket fuel 1,1-dimethylhydrazine and its transformation products. Multicriteria hazard ranking based on partial order methodologies, Environ. Tox. Pharm . 27 (2009) 415–423.
 
 

 
 [78] 
 L. Carlsen,The interplay between QSAR/QSPR studies and partial order ranking and formal concept analyses, Int. J. Mol. Sci. 10 (2009) 1628–1657.
 
 

 
 [79] 
 Carlsen L.; Bruggemann R.; Kenessova O.; Erzhigitov E. Evaluation of analytical performance based on partial order methodology. Talanta 2015, 132, 285-293.
 
 

 
 [80] 
 Carlsen, L.; Bruggemann, R. On the influence of data noise and uncertainty on ordering of objects, described by a multi-indicatorsystem. A set of pesticides as an exemplary case. J. Chemom . 2016 , 30, 22–29 .
 
 

 
 [81] 
 Carlsen L., Brueggemann R. (2017). Fragile State Index: Trends and Developments. A Partial Order Data Analysis. Social Indicators Research 133, 1-14).
 
 

 
 [82] 
 Carlsen L. (2017). An Alternative View on Distribution Keys for the Possible
Relocation of Refugees in the European Union. Social Indicators Research.,
130, 1147-1163.
 
 

 
 [83] 
 Carlsen, L. (2018). Happiness as a sustainability factor.The world happiness index: A posetic-based dataanalysis. Sustainability Science , 13(2), 549–571
 
 

 
 [84] 
 Carlsen, L., Bruggemann, R. (2019). An analysis of the ‘Failed States Index’ by partial order methodology. J ournal of Social Structure , 14(1), 1–31.
 
 

 
 [85] 
 Carlsen, L.; Bruggermann, R. Inequalities in the European Union—A Partial Order Analysis of the Main Indicators, Sustainability (2021), 13, 62–78.
 
 

 
 [86] 
 Carlsen, L., Bruggemann, R. (2021). Gender equality in Europe: The development of the sustainabledevelopment goal No. 5 illustrated by exemplary cases. Social Indicators Research, 158(3), 1127-1151.
 
 

 
 [87] 
 Carlsen, L.; Bruggemann, R. Partial Order as Decision Support Between Statistics and Multi-criteria Decision Analyses. Standards 2022 , 2, 22.
 
 

 
 [88] 
 Carlsen, L.; Bruggemann, R. Combining different stakeholders’ opinions in multi-criteria decision analyses applying partial ordermethodology. Standards 2022 , 2, 35.
 
 

 
 [89] 
 Carlsen, L. Food Waste: The Good, the Bad, and (Maybe) the Ugly, Standards 3(1):43-56, 2023.
 
 

 
 [90] 
 Carpineto C. and Giovanni R., “GALOIS: An Order-Theoretic Approach to Conceptual Clustering.” International Conference on Machine Learning (1993).
 
 

 
 [91] 
 Carpineto, C., Romano, G. (1996). A Lattice Conceptual Clustering System and Its Application to Browsing Retrieval. Machine Learning , 24, 95-122.
 
 

 
 [92] 
 C. Carpineto, G. Romano, Concept Data Analysis , Wiley, Chichester, 2004.
 
 

 
 [93] 
 Carr et al., Associating Cognition With Amyloid Status Using Partially Ordered Set Analysis. Front Neurol. 2019;10:976.
 
 

 
 [94] 
 Cavalletti, B., Corsi, M. “Beyond GDP” Effects on National Subjective Well-Being of OECD Countries. Soc Indic Res 136, 931–966 (2018). https://doi.org/10.1007/s11205-016-1477-0 
 
 

 
 [95] 
 F. Chanza B. Michel, l An Introduction to Topological Data Analysis: Fundamental and Practical Aspects for Data Scientists Front.Artif.Intell. , Vol. 4 (2021) https://doi.org/10.3389/frai.2021.667963 
 
 

 
 [96] 
 Y. Chen, M. Mancini, X. Zhu and Z. Akata, ”Semi-Supervised and Unsupervised Deep Visual Learning: A Survey,” in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 46, no. 3, pp. 1327-1347, March 2024, doi: 10.1109/TPAMI.2022.3201576.
 
 

 
 [97] 
 Cheng, W., Rademaker, M., De Baets, B., Hüllermeier, E. (2010). Predicting Partial Orders: Ranking with Abstention. In: Balcázar, J.L., Bonchi, F., Gionis, A., Sebag, M. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2010. Lecture Notes in Computer Science(), vol 6321. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-15880-3_20 .
 
 

 
 [98] 
 Cheng et al., Partial Orders: Ranking with Abstention, https://www.weiweicheng.com/research/slidesposters/cheng-ecml10aslides.pdf 
 
 

 
 [99] 
 G.-S Cheon, B. Curtis, G. Kwon and A.M. Mwafise, Riordan posets and associated incidence matrices, Linear Algebra Appl. , 632: 308–331 (2022).
 
 

 
 [100] 
 J. Chen, X. Song, Liqiang Nie, X. Wang, H. Zhang, and Tat-Seng Chua. 2016. Micro tells macro: predicting the popularity of micro-videos via a transductive model. In MM. 898–907.
 
 

 
 [101] 
 Cimiano, P., Hotho, A., Staab, S. (2005). Learning Concept Hierarchies from Text Corpora using Formal Concept Analysis. Journal of Artificial Intelligence Research 2005 DOI: 10.1613/jair.1648 
 
 

 
 [102] 
 M Collery(2022), Learning binary classification rules for sequential data, https://strl2022.github.io/files/short1.pdf .
 
 

 
 [103] 
 M. Collery et al.(2023), Neural-based classification rule learning for sequential data, arXiv:2302.11286.
 
 

 
 [104] 
 Comim, F. A Poset-Generalizability Method for Human Development Indicators. Soc. Indic. Res. 2021 , 158, 1179–1198.
 
 

 
 [105] 
 M. Crampes, J. Oliveira-Kumar, S. Ranwez, J. Villerd. Visualizing Social Photos on a Hasse Diagram for Eliciting Relations and Indexing New Photos. IEEE Computer Graphics and Applications, 2009, 15 (6), pp.985-992.
 
 

 
 [106] 
 V. Dankers, E. Bruni, and D. Hupkes. 2022. The Paradox of the Compositionality of Natural Language: A Neural Machine Translation Case Study. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 4154–4175, Dublin, Ireland. Association for Computational Linguistics.
 
 

 
 [107] 
 De Loof K, De Baets B, De Meyer H, Brüggemann R. A hitchhiker’s guide to poset ranking. Comb Chem High Throughput Screen. 2008 ;11(9):734-44. doi: 10.2174/138620708786306032.PMID:18991576 .
 
 

 
 [108] 
 della Queva, S. (2017). Analysis of Social Participation: A Multidimensional Approach Based on the Theory of Partial Ordering. In M. Fattore, R. Bruggemann Partial Order Concepts in Applied Sciences, Springer.
 
 

 
 [109] 
 Dempster, A. P. (1967). Upper and lower probabilities induced by a multivalued mapping. The Annals of Mathematical Statistics . 38 (2): 325–339. doi: 10.1214/aoms/1177698950 .
 
 

 
 [110] 
 T. Denoeux, M-H Masson (2010), Dempster-Shafer Reasoning in large Partially Ordered Sets: Applications in Machine Learning, Advances in Intelligent and Soft Computing, book series (AINSC) volume 68. Springer, Berlin, Heidelberg.
 
 

 
 [111] 
 Di Brisco, A., Farina, P. (2018). Measuring gender gap from a poset perspective. Social Indicators Research , 136, 1109–1124.
 
 

 
 [112] 
 Diaz et al., Predicting Proteome-Early Drug Induced Cardiac Toxicity Relationships (Pro-EDICToRs) with Node Overlapping Parameters (NOPs) of a new class of Blood Mass-Spectra graphs ,The 11th International Electronic Conference on Synthetic Organic Chemistry session Computational Chemistry https://sciforum.net/paper/view/1371 
 
 

 
 [113] 
 Diego Colombo, Marloes H Maathuis, Markus Kalisch, and Thomas S Richardson. Learning high-dimensional directed acyclic graphs with latent and selection variables. The Annals of Statistics , pages 294–321, 2012.
 
 

 
 [114] 
 di Bella, E., Corsi, M., Leporatti, L. (2017). POSET Analysis of Panel Data with POSAC. In: Fattore, M., Bruggemann, R. (eds) Partial Order Concepts in Applied Sciences. Springer, Cham. https://doi.org/10.1007/978-3-319-45421-4_11 .
 
 

 
 [115] 
 S. Dong, P. Wang, and K. Abbas. 2021. A survey on deep learning and its applications. Comput. Sci. Rev. 40, C (May 2021).
 
 

 
 [116] 
 B.J. Dutka, 1, K. Jones, 1, K.K. Kwan, 1, H. Bailey , 2, R. McInnis 1. Use of microbial and toxicant screening tests for priority site selection of degraded areas in water bodies, Water Research , Vol. 22; 4, 1988, 503-510. https://doi.org/10.1016/0043-1354(88)90047-4 
 
 

 
 [117] 
 W. Dyer, Weighted Posets: Learning surface order from dependency trees. Proceedings of the 18th International Workshop on Treebanks and Linguistic Theories (TLT, SyntaxFest 2019) https://api.semanticscholar.org/CorpusID:204885369 
 
 

 
 [118] 
 U. Ergashev, E. C. Dragut, W. Meng, Learning To Rank Resources with GNN, WWW ’23: Proceedings of the ACM Web Conference 2023, pp. 3247–3256, https://doi.org/10.1145/3543507.3583360 
 
 

 
 [119] 
 Fueyo, F., Abascal, P., Jiménez, J., Palacio, A., Serrano, M.L., Tepavčević, A. (2025). Leveled partially ordered sets. Computational and Applied Mathematics , 44, article 416. https://doi.org/10.1007/s40314-025-03368-8 
 
 

 
 [120] 
 Fattore, M. (2008). Hasse diagrams, poset theory and fuzzy poverty measures. Rivista Internazionale Di Scienze Sociali, Anno , 116(1), 63–75.
 
 

 
 [121] 
 Fattore, M., Brueggemann, R., OWSIŃSKI, J., (2011). Using poset theory to compare fuzzy multidimensional material deprivation across regions. In: Ingrassia S., Rocci R. and Vichi, M. New Perspectives in Statistical Modeling and Data Analysis. Berlin: Springer-Verlag.
 
 

 
 [122] 
 Fattore, M., Maggino, F. and Colombo, E. (2012) “From Composite Indicators to Partial Orders: evalu-ating socio-economic phenomena through ordinal data”. In: Maggino F, Nuvolati G (eds.) Quality of Life in Italy: research and reflections. Social Indicators Research Series, 48, 41–68.
 
 

 
 [123] 
 Fattore, M., Arcagni, A., Barberis, S., (2014). Visualizing Partially Ordered Sets for Socioeconomic analysis. Revista Colombiana de Estadistica, 34(2), pp. 437–450.
 
 

 
 [124] 
 Fattore, M., Arcagni, A., (2014). PARSEC: an R package for poset-based
evaluation of multidimensional poverty. In: Bruggemann R., Carlsen L. and Wittmann J. Multi-Indicator Systems and Modelling in Partial Order. Berlin: Springer.
 
 

 
 [125] 
 Fattore M.; Maggino, F.; Arcagni, A.(2015)Exploiting Ordinal Data For Subjective Well-Being Evaluation The Measurement of Subjective Well-Being in Survey Research Vol. 16, No. 3, pp. 409–428.
 
 

 
 [126] 
 Fattore M., Maggino F., (2015). A new method for measuring and analyzing suffering - Comparing suffering patterns in Italian society. In Anderson R. E.
(Eds.). World Suffering and the Quality of Life. New York: Springer.
 
 

 
 [127] 
 Fattore, M. (2016). Partially ordered sets and the measurement of multidimensional ordinal deprivation. Social Indicators Research , 128(2), 835–858.
 
 

 
 [128] 
 Fattore M., Maggino F., Arcagni A. (2016) “Non-aggregative assessment ofsubjective well-being”, In G. Alleva, A. Giommi (eds.), Topics in Theoretical and Applied Statistics, Springer International Publishing Switzerland,
 
 

 
 [129] 
 M. Fattore, R. Bruggemann. Partial Order Concepts in Applied Sciences , Springer, Cham, 2017, pp. 71–86. DOI10.1007/978-3-319-27274-0_20 .
 
 

 
 [130] 
 Fattore M. (2018) “Non-aggregated indicators of environmental sustainability”, Silesian Statistical Review, 16(22), 7-22.
 
 

 
 [131] 
 Fattore, M., Arcagni, A. (2018). F-FOD: Fuzzy first order dominance analysis and populations ranking over ordinal multi-indicators system. Social Indicators Research , 144(1), 1–29.
 
 

 
 [132] 
 Fattore M., Zenga Ma. (2019) “New posetic tools for the evaluation of financialliteracy”, In Bianco A., Gnaldi M., Conigliaro. P. (eds.), Italian Studies on
Quality of Life, Springer, 978-3-030-06021-3.
 
 

 
 [133] 
 Fattore, M., Arcagni, A., Maggino, F. (2019). Optimal scoring of partially ordered data,with an application to the ranking of smart cities. https://research.uniroma1.it/pubblicazioni/49725 .
 
 

 
 [134] 
 F. Feng, X. He, Y. Liu, L. Nie, and T-S Chua. 2018. Learning on Partial-Order Hypergraphs. In WWW 2018: The 2018 Web Conference , April 23–27, 2018, Lyon, France. ACM, New York, NY, USA 10 Pages.
 
 

 
 [135] 
 Fisher, D.H., Langley, P. (1985). Approaches to Conceptual Clustering. International Joint Conference on Artificial Intelligence.
 
 

 
 [136] 
 H. Fisher. Knowledge Acquisition Via Incremental Conceptual Clustering. Machine Learning 2:139–172, 1987.
 
 

 
 [137] 
 Finn, V.K., and others (1982). Many valued 1ogics as fragments formalize semantics. Acta phi- losophica Fennica, vol.35.
 
 

 
 [138] 
 V. K. Finn, M. I. Zabezhailo and O. M. Anshakov On a Computer-Oriented Formalization of Plausible Reasoning In F. Bacon-J. S. Mill’S Style (Main Principles and Computer Experiments) IFAC Artificial Intelligence. Leningrad. USSR 1983.
. 
 

 
 [139] 
 Finn, V.K.: Plausible Reasoning in Systems of JSM Type. Itogi Nauki i Tekhniki, Seriya Informatika 15, 54–101 (1991).
 
 

 
 [140] 
 A. Formica, Ontology-based concept similarity in Formal Concept Analysis, Information Sciences, Volume 176, Issue 18, 2006, Pages 2624-2641, https://doi.org/10.1016/j.ins.2005.11.014.
 
 

 
 [141] 
 Francés, O.; Abreu-Salas, J.; Fernández, J.; Gutiérrez, Y.; Palomar, M. Multidimensional Data Analysis for Enhancing In-Depth Knowledge on the Characteristics of Science and Technology Parks. Appl. Sci. 2023, 13, 12595. https://doi.org/10.3390/app132312595 
 
 

 
 [142] 
 Lerche, D. and P. Sorensen (2003) Evaluation of the ranking probabilities for partial orders based on random linear extensions. Chemosphere 53, 981-992.
 
 

 
 [143] 
 Ganter, B., Wille, R.: Formal Concept Analysis, Mathematical Foundations. Springer-Verlag (1999).
 
 

 
 [144] 
 H. Gao et al. “Graph-augmented Learning to Rank for Querying Large-scale Knowledge Graph.” ArXiv abs/2111.10541 (2021):
 
 

 
 [145] 
 Garriga, G.C. (2011). Formal Concept Analysis. In: Sammut, C., Webb, G.I. (eds) Encyclopedia of Machine Learning. Springer, Boston, MA.
 
 

 
 [146] 
 Gennari, J., Langley, P., Fisher, D. (1989). Models of incremental concept formation. Artificial Intelligence,
40:12–61.
 
 

 
 [147] 
 P. Gepner, Machine Learning and High-Performance Computing Hybrid Systems, a New Way of Performance Acceleration in Engineering and Scientific Applications, Proceedings of the 16th Conference on Computer Science and Intelligence Systems,ACSIS , Vol. 25, pages 27–36 (2021).
 
 

 
 [148] 
 R. Godin, R. Missaoui, H. Alaoui. Incremental concept formation algorithms based on Galois(concept) Lattice, Computational IntelligenceVolume 11: 2 ,pp. 246–267.
 
 

 
 [149] 
 Goncalves, B. and Porto, F. Research lattices: towards a scientific hypothesis data model. In Conference on Scientific and Statistical Database Management, SSDBM ’13, Baltimore, MD, USA, July 29 - 31, 2013. 
 
 

 
 [150] 
 E. Goodwin et al., Compositional Generalization in Dependency Parsing. arXiv:2110.06843 (cs).
 
 

 
 [151] 
 G. Grätzer, General Lattice Theory: Academic Press, Inc., New York. 1978.
 
 

 
 [152] 
 Y. Guo, Z. Lin, J-G Lou, D. Zhang, Hierarchical Poset Decoding for Compositional Generalization in Language, 34 t ​ h 34^{th} conference on Neural Processing Systems(NeurlPS 2020) Vancouver Canada.
 
 

 
 [153] 
 Guo et al. RankDNN: Learning to Rank for Few-Shot Learning,(2023) The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23)
 
 

 
 [154] 
 Hem, B.G. (2025). Poset functor cocalculus and applications to topological data analysis. arXiv:2501.05996.
 
 

 
 [155] 
 Hem, B.G. (2025). Decomposing multipersistence modules using functor calculus. arXiv:2510.06178.
 
 

 
 [156] 
 Hadzikadic, M., Yun, D. (1989). Concept formation by incremental conceptual clustering. Proceedings of
the Eleventh International Joint Conference on Artificial Intelligence (pp. 831–836). Detroit, MI: Morgan Kaufmann.
 
 

 
 [157] 
 M. R. Hacene, A. Napoli, P. Valtchev, Y. Toussaint and R. Bendaoud, ”Ontology Learning from Text Using Relational Concept Analysis,” 2008 International MCETECH Conference on e-Technologies (mcetech 2008), Montreal, QC, Canada, 2008, pp. 154-163, doi: 10.1109/MCETECH.2008.29.
 
 

 
 [158] 
 M. Hajij et al., Higher-Order Attention Networks (2022), https://arxiv.org/abs/2206.00606v1 .
 
 

 
 [159] 
 M. Hajij et al., Topological Deep Learning: Going Beyond Graph Data (2023), arXiv: arXiv:2206.00606v3.
 
 

 
 [160] 
 E. Halfon, M.G. Reggiani, On ranking chemicals for environmental hazard Environ. Sci. Technol. 1986, 20, 11, 1173–1179.
 
 

 
 [161] 
 Halfon, E. (2006). Hasse Diagrams and Software Development. In: Brüggemann, R., Carlsen, L. (eds) Partial Order in Environmental Sciences and Chemistry. Springer, Berlin, Heidelberg.
 
 

 
 [162] 
 Hanson, S., Bauer, M. (1989). Conceptual clustering, categorization, and polymorphy. Machine Learning ,
3:343–372.
 
 

 
 [163] 
 B. A. Hassan, Ontology Learning Using Formal Concept Analysis and WordNet, arXiv:2311.14699, 2023.
 
 

 
 [164] 
 Hilckmann, A., Bach, V., Bruggemann, R., Ackermann, R., Finkbeiner, M. (2017). Partial order analysis of the government dependence of the sustainable development performance in Germany’s federal states. In M. Fattore R. Bruggemann (Eds.), (2017) Partial Order Concepts in Applied Sciences . Springer.
 
 

 
 [165] 
 S. Hira, P.S. Deshpande, Data Analysis using Multidimensional Modeling, Statistical Analysis and Data Mining on Agriculture Parameters, Procedia Computer Science, Vol. 54, 2015, pp. 431-439, https://doi.org/10.1016/j.procs.2015.06.050 .
 
 

 
 [166] 
 T. Hirai a, F. Comim, Measuring the sustainable development goals: A poset analysis, Ecological Indicators , Vol. 145, 2022.
 
 

 
 [167] 
 Huchard, M., Rouane-Hacène, M., Roume, C., Valtchev, P.: Relational concept discovery in structured datasets. Ann. Math. Artif. Intell. 49 (1-4) (2007) 39-76.
 
 

 
 [168] 
 D. I. Ignatov, Introduction to Formal Concept Analysis and Its Applications in Information Retrieval and Related Fields(2017), arXiv:1703.02819
 
 

 
 [169] 
 Ikeda, M., Yamamoto, A. (2013). Classification by Selecting Plausible Formal Concepts in a Concept Lattice.
 
 

 
 [170] 
 Ip, E., Chen, S.-H., Quandt, S. (2016). Analysis of multiple partially ordered responses to belief items with don’t know option. Psychometrika , 81(2), 483–505.
 
 

 
 [171] 
 T. Ivanciuc, O. Ivanciuc, D. J. Klein, Posetic Quantitative Superstructure/Activity Relationships (QSSARs) for Chlorobenzenes, J. Chem. Inf. Model. 2005, 45, 4, 870–879. https://doi.org/10.1021/ci0501342 
 
 

 
 [172] 
 Ivaldi, E., Ciacci, A., Soliani, R. (2020). Urban deprivation in Argentina: A poset analysis. Papers in Regional Science , 99(6), 1723-1747.
 
 

 
 [173] 
 S. D. Langhans, P. Reichert, N. Schuwirth, The method matters: A guide for indicator aggregation in ecological assessments, Ecological Indicators, Volume 45,
2014, pp. 494-507, https://doi.org/10.1016/j.ecolind.2014.05.014 .
 
 

 
 [174] 
 S. Jabin, Machine Learning methods and applications using Formal Concept Analysis, International Journal of New Technologies in Science and Engineering Vol. 2: 3, 2015,
 
 

 
 [175] 
 Jaeger J, Tatsuoka C, Berns S, Varadi F, Czobor P, Uzelac S: Associating functional recovery with neurocognitive profiles identified using partially
ordered classification models. Schizophr Res 2006, 85:40-48.
 
 

 
 [176] 
 Jaeger J, Tatsuoka C, Berns SM, Varadi F: Distinguishing neurocognitive functions in schizophrenia using partially ordered classification models. Schizophr Bull 2006, 32:679-691
 
 

 
 [177] 
 M.F. Janowitz., Semi-Flat Cluster Methods, Discrete Mathematics 21 (1978) , 47-60.
 
 

 
 [178] 
 M.F. Janowitz., An order theoretic model for cluster analysis. SIAM Journal on Applied Mathematics ,34:55–72, 1978.
 
 

 
 [179] 
 Janowitz, M.F. (2007). Cluster Analysis Based on Posets. In: Brito, P., Cucumel, G., Bertrand, P., de Carvalho, F. (eds) Selected Contributions in Data Analysis and Classification. Studies in Classification, Data Analysis, and Knowledge Organization. Springer, Berlin, Heidelberg.
 
 

 
 [180] 
 M. F. Janowitz: Ordinal and Relational Clustering. Interdisciplinary Mathematical Sciences 10, World Scientific 2010, pp. 1-200.
 
 

 
 [181] 
 R. Jardine, Posets, Metric Spaces, Topological Data Analysis(2020), http://www.sci.brooklyn.cuny.edu/~noson/CTslides/Jardine20.pdf 
 
 

 
 [182] 
 W. Kim, F. Memoli, Persistence Over Posets, Notices of American Mathemtical Society, Vol 20, No. 8, (2023).
 
 

 
 [183] 
 Jia, H., Newman, J., Tianfield, H. (2009). A new Formal Concept Analysis based learning approach to Ontology building. In: Sicilia, MA., Lytras, M.D. (eds) Metadata and Semantics. Springer, Boston, MA. https://doi.org/10.1007/978-0-387-77745-0_42 .
 
 

 
 [184] 
 I. Jonyer, L. B. Holder, and D. J. Cook. Graph-Based Hierarchical Conceptual Clustering. Proceedings of the Florida Artificial Intelligence Research Symposium , pp. 91-95, 2000.
 
 

 
 [185] 
 I. Jonyer, D. J. Cook , L. B. Holder, Graph-Based Hierarchical Conceptual Clustering, Journal of Machine Learning Research 2 (2001) 19-43.
 
 

 
 [186] 
 Kainz, W., Egenhofer, M. J., and Greasley, I. (1993). Modelling spatial relations and operations with partially ordered sets. International Journal of Geographical Information Systems , 7(3):215–229.
 
 

 
 [187] 
 H-M. Kaltenbach, Teaching Design of Experiments using Hasse diagrams arXiv:1912.08567v1
 
 

 
 [188] 
 Kang, X., Li, D., Wang, S. (2011). A multi-instance ensemble learning model based on concept lattice. Knowl. Based Syst. , 24, 1203-1213.
 
 

 
 [189] 
 S. Kardaetz, T. Strube, R. Bruggemann, G. Nützmann, Ecological scenarios analyzed and evaluated by a shallow lake model, J. Environ. Manag . 88 (2008) 120–135.
 
 

 
 [190] 
 J. Karhunen, T. Raiko, K. Cho, Unsupervised deep learning: A short review, Advances in Independent Component Analysis and Learning Machines , Academic Press, 2015, Pages 125-142, https://doi.org/10.1016/B978-0-12-802806-3.00007-5 .
 
 

 
 [191] 
 Y. Kashnitsky and D. I. Ignatov, “Can FCA-based Recommender System Suggest a Proper Classi er ? 2 Multiple Classi er Systems,” FCA do Artif. Intell. 2014), p. 17, 2014.
 
 

 
 [192] 
 M. Kendall, A new measure of rank correlation, Biometrika, Volume 30, Issue 1-2, June 1938, Pages 81–93 (doi: 10.1093/biomet/30.1-2.81 )
 
 

 
 [193] 
 Kickmeier-Rust, M.D., Albert, D. (2013). Using Hasse Diagrams for Competence-Oriented Learning Analytics. In: Holzinger, A., Pasi, G. (eds) Human-Computer Interaction and Knowledge Discovery in Complex, Unstructured, Big Data. HCI-KDD 2013. Lecture Notes in Computer Science, vol 7947. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-39146-0_6 
 
 

 
 [194] 
 Kim, W., Memoli, F.: Persistence over posets. Notices of theAmerican Mathematical Society 2761, 1214–1224 (September 2023).
 
 

 
 [195] 
 T. N Kipf and M. Welling. 2017. Semi-supervised classification with graph convolutional networks. ICLR (2017).
 
 

 
 [196] 
 Klein, D. J. Similarity and Dissimilarity in Posets. J. Math. Chem. 1995 , 18, 321-348.
 
 

 
 [197] 
 Klein, D. J.; Babic, D. Partial Orderings in Chemistry. J. Chem. Inf. Comput. Sc. 1997 , 37, 656–671.
 
 

 
 [198] 
 D. J. Klein, Prolegomenon on partial orderings in chemistry, MATCH Commun. Math.Comput. Chem. 42 (2000) 1–290.
 
 

 
 [199] 
 Klein, D. J., Ivanciuc, T. (2006). Directed reaction graphs as posets. In R. Bruggemann L. Carlsen (Eds.), Partial order in environmental sciences and chemistry (pp. 35–57). Berlin: Springer.
 
 

 
 [200] 
 Khatri, Minal, ”Formal Concept Analysis for Image Classification and Machine Learning Models for Anti-CRISPR Protein Discovery in Bioinformatics” (2023). Dissertations and Doctoral Documents from University of Nebraska-Lincoln, 2024–. 47.
 
 

 
 [201] 
 Kotsianti, S.B., Kanellopoulos, D. (2007). Combining Bagging, Boosting and Dagging for Classification Problems. In: Apolloni, B., Howlett, R.J., Jain, L. (eds) Knowledge-Based Intelligent Information and Engineering Systems. KES 2007. Lecture Notes in Computer Science(), vol 4693. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-74827-4_62 
 
 

 
 [202] 
 V. Kristina Et Al. , “Features of PyHasse software used for the evaluation of chemicals in human breast milk samples,” Simulation in Umwelt- und Geowissenschaften , vol.141, Hamburg, Germany, pp.169-180, 2012.
 
 

 
 [203] 
 R. Kurtz(2020), Contributions to Semantic Dependency Parsing.
 https://www.diva-portal.org/smash/get/diva2:1457198/FULLTEXT02 .
 
 

 
 [204] 
 S. O. Kuznetsov, Mathematical aspects of concept analysis, Journal of Mathematical Sciences 80(2),(1996).
 
 

 
 [205] 
 Kuznetsov, S.O. Machine Learning on the Basis of Formal Concept Analysis. Automation and Remote Control 62, 1543–1564 (2001). https://doi.org/10.1023/A:1012435612567 
 
 

 
 [206] 
 Kuznetsov, S.O, Machine Learning and Formal Concept Analysis Conference: Concept Lattices, Second International Conference on Formal Concept Analysis, ICFCA 2004, Sydney, Australia, February 23-26, 2004.
 
 

 
 [207] 
 S. Kuznetsov, Ordered Sets for Data Analysis(2019), arXiv:1908.11341 .
 
 

 
 [208] 
 G. Lebanon, J. Lafferty, Conditional models on the ranking poset, NIPS’02: Proceedings of the 15th International Conference on Neural Information Processing Systems, January 2002, pp. 431–438.
 
 

 
 [209] 
 Lebowitz, M. (1986). Concept learning in a rich input domain: generalization-based memory. In R.S. Michalski,
J.G. Carbonell, T.M. Mitchell (Eds.), Machine Learning: An Artificial Intelligence Approach (Vol. 2). Morgan Kaufmann, San Mateo, CA.
 
 

 
 [210] 
 C. Lei et al., The Application of Multidimensional Data Analysis in the EIA Database of Electric Industry, 2011 3rd International Conference on Environmental Science and Information Application Technology (ESIAT 2011), In: Procedia Environmental Sciences 10 ( 2011 ) 1210 – 1215.
 
 

 
 [211] 
 Levy, S. (1985). Partial order analysis of crime indicators. Social Indicators Research , 16(2), 195–199.
 
 

 
 [212] 
 Linear Extensions https://mathworld.wolfram.com/LinearExtension.html .
 
 

 
 [213] 
 J. Liu, Q. Zhang, W. Wang, L. McMillan, J. Prins, Clustering pair-wise dissimilarity data into partially ordered sets. In Proceedings of the Twelfth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. (2006) 637–642.
 
 

 
 [214] 
 Liu, Q., An, S., Lou, J.-G., Chen, B., Lin, Z., Gao, Y., Zhou, B., Zheng, N., and Zhang, D. Compositional generalization by learning analytical expressions. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY, USA, 2020.
 
 

 
 [215] 
 Maarek, Y.S. (1990). An Incremental Conceptual Clustering Algorithm that Reduces Input-Ordering Bias. In: Golumbic, M.C. (eds) Advances in Artificial Intelligence. Springer, New York, NY. https://doi.org/10.1007/978-1-4613-9052-7_7 
 
 

 
 [216] 
 H. Mannila, C. Meek, Global partial orders from sequential data, KDD’00: Proceedings of the sixth ACM SIGKDD international conference on knowledge discovery and data mining, August 2000, pages 161–168.
 
 

 
 [217] 
 L. Markowsky and G. Markowsky, “Lattice Data Analytics and an Exploratory Analysis of the Carver2 Dataset,” to appear in IDAACS 2019.
 
 

 
 [218] 
 Markov, A lattice-based approach to hierarchical clustering. Proceedings of the Florida Artificial Intelligence Research Symposium , pp. 389–393, 2001.
 
 

 
 [219] 
 Martin, J., Billman, D. (1994). Acquiring and combining overlapping concepts. Machine Learning , 16(1–
2):121–155.
 
 

 
 [220] 
 McKusick, K., Langley, P. (1991). Constraints on tree structure in concept formation. Proceedings of the
Thirteenth International Joint Conference on Artificial Intelligence (pp. 810–816). Sydney, Australia: Morgan
Kaufmann.
 
 

 
 [221] 
 Meddouri, N., Maddouri, M. (2009). Boosting Formal Concepts to Discover Classification Rules. In: Chien, BC., Hong, TP., Chen, SM., Ali, M. (eds) Next-Generation Applied Intelligence. IEA/AIE 2009. Lecture Notes in Computer Science(), vol 5579. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-02568-6_51 .
 
 

 
 [222] 
 N. Meddouri , H. Khoufi , M. Maddouri, Parallel Learning and Classification for Rules based on Formal Concepts Procedia Computer Science
Volume 35, 2014, pp. 358-367.
 
 

 
 [223] 
 J-P. M e ´ \acute{e} tivier et al., Discovering Structural Alerts for Mutagenicity Using Stable Emerging Molecular Patterns, J. Chem. Inf. Model. 2015, 55, 5, 925–940, 2015 https://doi.org/10.1021/ci500611v .
 
 

 
 [224] 
 R.S. Michalski, Knowledge Acquisition Through Conceptual Clustering: A Theoretical Framework and an Algorithm for Partitioning Data into Conjunctive Concepts. International Journal for Policy Analysis and Information Systems . Vol. 4, No. 3, 1980.
 
 

 
 [225] 
 Michalski and R. E. Stepp. Learning From Observation: Conceptual Clustering. In R.S.Michalski, J.G. Carbonell, and T.M. Mitchell (Eds.), Machine Learning: An Artificial Intelligence Approach, Volume 1, Tioga Publishing Company, pp. 331-363, 1983.
 
 

 
 [226] 
 V. Mnih, N. Heess, A. Graves, and k. kavukcuoglu, “Recurrent models of visual attention,” in 27th Annual Conference on Neural
Information Processing Systems (NIPS 2014). Curran Associates,
Inc., 2014, pp. 2204–2212.
 
 

 
 [227] 
 A. M. Mwafise, G-S Cheon, H. J. Choi, S. Giraudo, Operad Structure of Poset Matrices (2024), arXiv:2401.06814.
 
 

 
 [228] 
 Mollica, C., Tardella, L. (2020). PLMIX: an R package for modelling and clustering partially ranked data. Journal of Statistical Computation and Simulation, 90(5), 925–959. https://doi.org/10.1080/00949655.2020.1711909 .
 
 

 
 [229] 
 M. Nabeel et al., A survey of ontology learning techniques and applications, Database, Volume 2018, 2018, bay101, https://doi.org/10.1093/database/bay101 
 
 

 
 [230] 
 Nardo, M., M. Saisana, A. Saltelli and S. Tarantola: 2005b, Tools for Composite IndicatorsBuilding. European Commission, report EUR 21682 EN (Joint Research Centre, Ispra,Italy).
 
 

 
 [231] 
 Newlin, J.T., Patil, G.P. Application of partial order to stream channel assessment at bridge infrastructure for mitigation management. Environ Ecol Stat 17, 437–454 (2010). https://doi.org/10.1007/s10651-010-0162-8 .
 
 

 
 [232] 
 Nicholls GK, Lee JE, Karn N, Johnson D, Huang R, Muir-Watt A, Bayesian inference for partial orders from random linear extensions: power relations from 12th Century Royal Acta, , 2022. arXiv: 2212.05524.
 
 

 
 [233] 
 Olave, A.A. (2026). A new family of distances over partially ordered sets. arXiv:2606.06377.
 
 

 
 [234] 
 Peng, C., Stachurski, J., Yang, J. (2026). Abstract Dynamic Programming on Partially Ordered Spaces. arXiv:2606.12777.
 
 

 
 [235] 
 Sadr, A., Alidoost Nia, M. (2026). MileStone: A Multi-Objective Compiler Phase Ordering Framework for Graph-based IR-Level Optimization. arXiv:2605.23435.
 
 

 
 [236] 
 Sargent, T.J., Stachurski, J. (2025). Dynamic Programs on Partially Ordered Sets. SIAM Journal on Control and Optimization . arXiv:2308.02148.
 
 

 
 [237] 
 Taeb, A., Guo, F.R., Henckel, L. (2026). Model-oriented Graph Distances via Partially Ordered Sets. arXiv:2511.10625.
 
 

 
 [238] 
 Wong, K., Xiao, W., Rus, D. (2026). PoSafeNet: Safe Learning with Poset-Structured Neural Nets. arXiv:2601.22356.
 
 

 
 [239] 
 Dolores-Cuenca, E., Guzmán-Sáenz, A., Kim, S., López-Moreno, S., Mendoza-Cortes, J. (2025). Order Theory in the Context of Machine Learning. arXiv:2412.06097.
 
 

 
 [240] 
 P.C. Golar et al. Mining large-scale data using AI-integrated lattice and poset structures (2026). Journal of Discrete Mathematical Sciences and Cryptography , 29(2-A), 671–680. https://doi.org/10.47974/JDMSC-2510 .
 
 

 
 [241] 
 Onishchenko, A.A., Gurov, S.I. Classification based on formal concept analysis and biclustering: possibilities of the approach. Comput Math Model 23, 329–336 (2012). https://doi.org/10.1007/s10598-012-9141-2 .
 
 

 
 [242] 
 Ouyang, C.; Liu, Y. , Formal Concept Analysis Supporting Ontology Learning From Database, Advanced Science Letters, Volume 7 2012, pp. 473-477(5).
 
 

 
 [243] 
 V. Padhye, K. Lakshmanan, A deep actor critic reinforcement learning framework for learning to rank, Neurocomputing, Volume 547, (2023.) https://doi.org/10.1016/j.neucom .
 
 

 
 [244] 
 Pakkar, M.S. A Hierarchical Aggregation Approach for Indicators Based on Data Envelopment Analysis and Analytic Hierarchy Process. Systems 2016, 4, 6. https://doi.org/10.3390/systems4010006
 
 

 
 [245] 
 G. P. Patil, C. Taillie, Multiple indicators, partially ordered sets, and linear extensions: Multi-criterion ranking and prioritization, Environ. Ecol. Stat. 11 (2004) 199–228.
 
 

 
 [246] 
 G. Patil, W. L. Myers, R. Bruggemann, Multivariate datasets for inference of order: some considerations and explorations, in: R. Bruggemann, L. Carlsen, J. Wittmann (Eds.), Multi-Indicator Systems and Modelling in Partial Order , Springer, New York, 2014,pp.13–45.
 
 

 
 [247] 
 L. Pellegrina, C. Cousins, F. Vandin, M. Riondato. MCRapper: Monte-Carlo Rademacher Averages for Poset Families and Approximate Pattern Mining. Extended version. https://arxiv.org/abs/2006.09085 .
 
 

 
 [248] 
 Popescu-Spineni, S. (1998). Hierarchy Techniques of Multidimensional Data Analysis (MDA) in Social Medicine Research. In: Rizzi, A., Vichi, M., Bock, HH. (eds) Advances in Data Science and Classification. Studies in Classification, Data Analysis, and Knowledge Organization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-72253-0_87 
 
 

 
 [249] 
 O. Prokasheva, A. Onishchenko, S. Gurov, Classification Methods Based on Formal Concept Analysis, Conference: FCAIR 2013 – Formal Concept Analysis Meets Information Retrieval. Workshop co-located with the 35th European Conference on Information Retrieval (ECIR 2013)At: Moscow.
 
 

 
 [250] 
 S. Pudenz, ProRank – software for partial order ranking, MATCH Commun. Math. Comput. Chem. 54 (2005) 611–622.
 
 

 
 [251] 
 Qiu, J., Wu, Q., Ding, G. et al. A survey of machine learning for big data processing. EURASIP J. Adv. Signal Process. 2016, 67 (2016). https://doi.org/10.1186/s13634-016-0355-x .
 
 

 
 [252] 
 N.R Rashidah, A Comparison Between Single Linkage and Complete Linkage in Agglomerative Hierarchical Cluster Analysis for Identifying Tourists Segments, IIUM Engineering Journal, Vol. 12, No. 6, 2011: Special Issue in Science and Ethics.
 
 

 
 [253] 
 A. Raveh and S. F. Landau, Partial Order Scalogram Analysis with Base Coordinates (POSAC): Its Application to Crime Patterns in All the States in the United States, Journal of Quantitative Criminology Vol. 9, No. 1., pp. 83-99 (1993).
 
 

 
 [254] 
 G. Restrepo R. Brüggemann, Ranking regions using cluster analysis, Hasse diagram technique and topology , 3 r ​ d 3^{rd} International Congress on Environmental Modelling and Software, Burlington, Vermont, USA 2006.
 
 

 
 [255] 
 G. Restrepo, R. Bruggemann, M. Weckert, S. Gerstmann, H. Frank, Ranking patterns, an application to refrigerants, MATCH Commun. Math. Comput. Chem. 59 (2008) 555–584.
 
 

 
 [256] 
 Rimoldi S.M.L., Arcagni A., Fattore M., Barbiano di Belgiojoso E.. (2020)
“Targeting policies for multidimensional poverty and social fragility relief
among migrants in Italy, using F-FOD analysis”, Social Indicators Research
- doi10.1007/s11205-020-02485-7 
 
 

 
 [257] 
 Rimoldi S.M.L., Arcagni A., Fattore M., Terzera T. (2020) “Social and
material vulnerability of the Italian municipalities: comparing alternative ap-
proaches”, Social Indicators Research- doi:10.1007/s11205-020-02330-x .
 
 

 
 [258] 
 Sagi O, Rokach L. 2018. Ensemble learning: a survey . Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 8(4):e1249
 
 

 
 [259] 
 G.-C. Rota, On the foundations of combinatorial theory I. Theory of Mobius functions, Z. Wahrsch. Verw. Gebiete 2 (2964) 340-368.
 
 

 
 [260] 
 Rouane-Hacene, M., Valtchev, P., Nkambou, R.: Supporting Ontology Design through Large-Scale FCA-Based Ontology Restructuring. In: ICCS 2011. (2011) 257–269.
 
 

 
 [261] 
 M. D. Ruiz, E. Hüllermeier, A formal and empirical analysis of the fuzzy gamma rank correlation coefficient, Information Sciences,Vol. 206, 2012, pp. 1-17, https://doi.org/10.1016/j.ins.2012.04.006 .
 
 

 
 [262] 
 F. Ruskey, Generating linear extensions of posets by transpositions, Journal of Combinatorial Theory, Series B, Volume 54, Issue 1, 1992, Pages 77-101, https://doi.org/10.1016/0095-8956(92)90067-8 .
 
 

 
 [263] 
 I.M. Sabara, F. Rozi, M. N. Jauhari, Agglomerative Hierarchical Clustering Analysis Based on Partially-Ordered Hasse Graph of Poverty Indicators in East Java,
Proceedings of the 12th International Conference on Green Technology (ICGT 2022).
 
 

 
 [264] 
 K. Sachs, O. Perez, D. Lauffenburger, G. Nolan, Causal protein-signaling networks derived from multiparameter single-cell data. Science. https://www.science.org/doi/10.1126/science.1105809 #supplementary-materials.
 
 

 
 [265] 
 Salman, H.E. Leveraging a combination of machine learning and formal concept analysis to locate the implementation of features in software variants 2023, Information and Software Technology, vol. 164.
 
 

 
 [266] 
 Sahami, M. (1995). Learning classification rules using lattices (Extended abstract). In: Lavrac, N., Wrobel, S. (eds) Machine Learning: ECML-95. ECML 1995. Lecture Notes in Computer Science, vol 912. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-59286-5_83 .
 
 

 
 [267] 
 Sangroya, A., Anantaram, C., Rawat, M., Rastogi, M. (2019). Using Formal Concept Analysis to Explain Black Box Deep Learning Classification Models. FCA4AI@IJCAI. 
 
 

 
 [268] 
 A. Santana, E. Colombini, Neural Attention Models in Deep Learning: Survey and Taxonomy(2021), arXiv:2112.05909v1.
 
 

 
 [269] 
 Semeraldi et al., Partial Order Rank Features in Colour Space,Appl. Sci. 2020, 10(2), 499; https://doi.org/10.3390/app10020499 
 
 

 
 [270] 
 Sen, A. (1970b). Interpersonal aggregation and partial comparability. Econometrica , 38(3), 393–409.
 
 

 
 [271] 
 Seymour, R. B. (1960). Missing Data in Non-Linear Trend Analysis of Repeated Measurements on the Same Individuals. The Journal of Educational Research, 54(4), 141–144. http://www.jstor.org/stable/27530403 .
 
 

 
 [272] 
 Shafer, G.: A mathematical theory of evidence . Princeton University Press, Princeton (1976).
 
 

 
 [273] 
 Silan M, Boccuzzo G, Arpino B. Matching on poset‐based average rank for multiple treatments to compare many unbalanced groups. Statistics in Medicine . 2021;40(28):6443–6458. doi: 10.1002/sim.9192 .
 
 

 
 [274] 
 U. Simon, R. Bruggemann, S. Mey, S. Pudenz, METEOR- application of a decision support tool based on discrete mathematics, MATCH Commun. Math. Comput. Chem. 54 (2005) 623–642.
 
 

 
 [275] 
 Simon, U.; Brüggemann, R.; Behrendt, H.; Shulenberger, E.; Pudenz, S. METEOR: a step-by-step procedure to explore effects of indicator aggregation in multi criteria decision aiding – application to water management in Berlin, Germany. Acta hydrochim. Hydrobiol. 2006, 34, 126-136.
 
 

 
 [276] 
 Song et al., Graph-based Semi-supervised Learning: A Comprehensive Review 2021.
 
 

 
 [277] 
 Sorensen, P. B.; Mogensen, B. B.; Carlsen, L.; Thomsen, M. The
Influence on Partial Order Ranking from Input Parameter Uncertainty Definition of a Robustness Parameter. Chemosphere 2000 , 595–600.
 
 

 
 [278] 
 P. B. Sorensen, R. Bruggemann, L. Carlsen, B. B. Mogensen, J. Kreuger, S. Pudenz, Analysis of monitoring data of pesticide residues in surface waters using partial order ranking theory, Envir. Tox. Chem. 22 (2003) 661–670.
 
 

 
 [279] 
 Sorensen, P. B. ; Brüggemann; M. Thomsen; D. B. Lerche, Application of multidimensional rank-correlation. MATCH Communications in Mathematical and in Computer Chemistry, 54(3), 643-670.
 
 

 
 [280] 
 M. Sugiyama, Machine Learning and Information Geometry II, https://mahito.info/files/Sugiyama_NII_IISS_2018_02.pdf .
 
 

 
 [281] 
 J.L. Sullivan S. Feldman Multiple Indicators - An Introduction (1979), https://www.ojp.gov/ncjrs/virtual-library/abstracts/multiple-indicators-introduction .
 
 

 
 [282] 
 Taeb, A. ; Buhlmann, P. ; Chandrasekaran, V. Model Selection over Partially Ordered Sets(2024), ArXiV: 2308.10375 .
 
 

 
 [283] 
 Mwafise, A.M. (2026). Order-Agnostic Generation of Lattices via Reinforcement Learning. ScienceOpen Preprints . DOI: 10.14293/PR2199.004125.v2.
 
 

 
 [284] 
 Mwafise, A.M. (2026). Matrix Decomposition Algorithms for Accelerated Poset Isomorphism. ScienceOpen Preprints . DOI: 10.14293/PR2199.003522.v2.
 
 

 
 [285] 
 Cheon, G.-S., Choi, H.J., Giraudo, S., Mwafise, A.M. (2026). Operads of Poset Matrices. The Electronic Journal of Combinatorics , 33(2), #P2.37. DOI: 10.37236/14396.
 
 

 
 [286] 
 Tatsuoka C: Sequential classification on partially ordered sets. PhD thesis Cornell University, Statistics Department; 1996.
 
 

 
 [287] 
 Tatsuoka C. Data analytic methods for latent partially ordered classification models. Applied Statistics (Journal of the Royal Statistical Society Series C ). 2002;51:337–350.
 
 

 
 [288] 
 Tatsuoka C, Ferguson T: Sequential classification on partially ordered sets. Journal of the Royal Statistical Society , Series B 2003, 65 :143-157.
 
 

 
 [289] 
 Tatsuoka C, Corrigendum:Data analytic methods for latent partiallyordered classification models, Appl.Statist.(2005) 54 ,Part2,pp.465–467
 
 

 
 [290] 
 Tatsuoka C. Diagnostic models as partially ordered sets. Measurement . (2009) 7:49–53.
 
 

 
 [291] 
 Tatsuoka C, Varadi F, Jaeger J. Latent partially ordered classification models and normal mixtures. J Edu Behav Stat. (2013) 38:267–94.
 
 

 
 [292] 
 Tatsuoka C, Tseng H, Jaeger J, Varadi F, Smith MA, Yamada T, Smyth KA,Lerner AJ. com AsDNInn: Modeling the heterogeneity in risk of progres-sion to Alzheimer’s disease acrosscognitive profiles in mild cognitiveimpairment. Alzheimer’s Res Ther. 2013;5:1–19.
 
 

 
 [293] 
 Tatsuoka C. Sequential classification on lattices with experiment-specific response distributions. Sequent Anal Design Methods Appl. (2014) 33:400–20.
 
 

 
 [294] 
 Tatsuoka C, McGowan B, Yamada T, Espy KA, Minich N, Taylor HG. Effects of extreme prematurity on numerical skills and executive function in kindergarten children: an application of partially ordered classification modeling. Learn Individ Differ. (2016) 49:332–40.
 
 

 
 [295] 
 K. M. Ting, I. Witten, Stacking Bagged and Dagged Models (1997), https://www.researchgate.net/publication/2516354_Stacking_Bagged_and_Dagged_Models 
 
 

 
 [296] 
 F Tombari (2023), Tame representations in Topological Data Analysis, https://www.diva-portal.org/smash/get/diva2:1759244/FULLTEXT01.pdf .
 
 

 
 [297] 
 F. J. Torres-Rojas Castro-Mora, Partially Ordered Sets and Logical Clocks for Distributed Systems
 
 

 
 [298] 
 Tovar, M., Pinto, D., Rendón, A.M., Serna, J.G., Ayala, D.V. (2014). Identification of Ontological Relations Using Formal Concept Analysis. LANMR .
 
 

 
 [299] 
 Tsakovski, V. Simeonov, Hasse diagrams as explanatory tool in environmental data mining: A case study, in: J. Owsinski, R. Bruggemann (Eds.) Multicriteria Ordering and Ranking: Partial Orders , Ambiguities and Applied Issues, Sys. Res. Inst. Polish Acad. Sci., Warsaw, 2007, pp. 50–68.
 
 

 
 [300] 
 Pirintsos, S., Bariotakis, M., Kalogrias, V., Katsogianni, S., Brüggemann, R. (2014). Hasse Diagram Technique Can Further Improve the Interpretation of Results in Multielemental Large-Scale Biomonitoring Studies of Atmospheric Metal Pollution. In: Brüggemann, R., Carlsen, L., Wittmann, J. (eds) Multi-indicator Systems and Modelling in Partial Order. Springer, New York, NY. https://doi.org/10.1007/978-1-4614-8223-9_11 
 
 

 
 [301] 
 van Engelen, J.E., Hoos, H.H. A survey on semi-supervised learning. Mach Learn 109, 373–440 (2020). https://doi.org/10.1007/s10994-019-05855-6 
 
 

 
 [302] 
 K. Voigt, G. Welzl, R. Bruggemann, Data analysis of environmental air pollutant monitoring systems in Europe, Environmetrics 15 (2004) 577–596.
 
 

 
 [303] 
 K. Voigt, R. Bruggemann, Ranking of pharmaceuticals detected in the environment: Aggregation and weighting procedures, Combin. Chem. High Through. Screen. 11 (2008) 770–782.
 
 

 
 [304] 
 K. Voigt, R. Bruggemann, M. Kirchner, K.-W. Schramm, Influence of altitude concerning the contamination of humus soils in the German Alps: a data evaluation
approach using PyHasse, Environ. Sci. Pollut. Res. 17 (2010) 429–440.
 
 

 
 [305] 
 K. Voigt, R. Bruggemann, H. Scherb, H. Shen, K. H. Schramm, Evaluating therelationship between chemical exposure and cryptorchidism by discrete mathematical method using PyHasse software, J. Environ. Model. Soft. 25 (2010) 1801–1812.
 
 

 
 [306] 
 Voigt, K.; Brüggemann, R. Water contamination with pharmaceuticals: data availability and evaluation approach with Hasse diagram technique and METEOR. MATCH Commun. Math. Comput. Chem. 2005, 54, 671-689.
 
 

 
 [307] 
 Voigt, Kristina et al. “A multi-criteria evaluation of environmental databases using the Hasse Diagram Technique (ProRank) software.” Environ. Model. Softw. 21 (2006): 1587-1597.
 
 

 
 [308] 
 K. Voigt et al., Application of the PYHASSE Program Features: Sensitivity, Similarity, and Separability for Environmental Health Data, Statistica Applicazioni - Special Issue, 2011, pp. 155-168.
 
 

 
 [309] 
 X. Wang and A. Gupta. Unsupervised learning of visual representations using videos. In ICCV, 2015.
 
 

 
 [310] 
 X. Wang, P. Li, M. Zhang, Improving Graph Neural Networks on Multi-node Tasks with Labeling Tricks, (2023) arXiv:2304.10074. Available in: https://arxiv.org/abs/2304.10074 
 
 

 
 [311] 
 C. Wendler, Machine Learning on non-Euclidean domains: powersets, lattices, posets, (2023) ETH Zurich.
 
 

 
 [312] 
 R. Wille, Restructuring lattice theory: An approach based on hierarchies of concepts, in: I. Rival (Ed.), Ordered Sets , D. Reidel Publishing, Dordrecht, 1982, pp. 445–470.
 
 

 
 [313] 
 J. Wittmann, J. Markert, S. Plura, R. Brüggemann, A Software Platform Towards a Comparison of Cars-A Case Study for Handling Ratio-Based Decisions, EnviroInfo 2011: Innovations in Sharing Environmental Observations and Information, Shaker Verlag Aachen. Available in: http://enviroinfo.eu/sites/default/files/pdfs/vol6919/0467.pdf .
 
 

 
 [314] 
 
A. M. Mwafise,
 Poset Matrix Structure Via Partial Composition Operations ,
arXiv preprint arXiv:2212.11644, 2022.
 
 

 
 [315] 
 Wu et al., A Comprehensive Survey on Graph Neural Networks,(2019) arXiv:1901.00596.
 
 

 
 [316] 
 Wu, C., Li, H., Ren, J. (2021). Research on hierarchical clustering method based on partially-ordered
Hasse graph. Future Generation Computer Systems. Doi: https://doi.org/10.1016/j.future.2021.07.025 
 
 

 
 [317] 
 Z. Xie, W. Hsu, Z. Liu, M. L. Lee: Concept Lattice based Composite Classifiers for high Predictability. Artificial Intelligence, vol. 139, pp. 253–267, Wollongong, Australia (2002).
 
 

 
 [318] 
 Xiao, S., Yan, J., Farajtabar, M., Song, L., Yang, X., Zha, H. (2017). Joint Modeling of Event Sequence and Time Series with Attentional Twin Recurrent Neural Networks. ArXiv, abs/1703.08524.
 
 

 
 [319] 
 Y. Yadav, A. Samal, E. Saucan, A Poset-Based Approach to Curvature of Hypergraphs, Symmetry , 14(2):420, 2022.
 
 

 
 [320] 
 M. Yang, R. Wang, Y. Shen, H. Qi and B. Yin, “ Breaking the Expression Bottleneck of Graph Neural Networks,” in IEEE Transactions on Knowledge and Data Engineering , vol. 35, no. 6, pp. 5652-5664, 1 June 2023, doi: 10.1109/TKDE.2022.3168070 .
 
 

 
 [321] 
 Yoneda, Y., Sugiyama, M., Washio, T. (2018). Learning Graph Representation via Formal Concept Analysis. ArXiv, abs/1812.03395 .
 
 

 
 [322] 
 H. Yu, Y. Chen, P. Lingras, G. Wang, A three-way cluster ensemble approach for large-scale data, International Journal of Approximate Reasoning, Volume 115, 2019, Pages 32-49, https://doi.org/10.1016/j.ijar.2019.09.001 .
 
 

 
 [323] 
 L. A. Zadeh, Fuzzy Sets, Information and Control , 8 , 338–353 (1965).
 
 

 
 [324] 
 J. Zhang , Learning to Rank Cases with Classification Rules(2008),
 http://www.ecmlpkdd2008.org/files/pdf/workshops/pl/8.pdf .
 
 

 
 [325] 
 L. Zhang, D. Wang , X. Liu, Computing Concept Lattices with Clustering Approaches.
 
 

 
 [326] 
 Zhao, M., Zhang, S., Li, W. et al. Matching biomedical ontologies based on formal concept analysis. J Biomed Semant 9, 11 (2018). https://doi.org/10.1186/s13326-018-0178-9 .
 
 

 
 [327] 
 Zia, A., Khamis, A., Nichols, J. et al. Topological deep learning: a review of an emerging paradigm. Artif Intell Rev 57, 77 (2024). https://doi.org/10.1007/s10462-024-10710-9 .
 
 

 
 [328] 
 Y. Zuo and R. Serfling, General Notions of Statistical Depth Function, The Annals of Statistics , Vol. 28, No. 2 (Apr., 2000), pp. 461-482.
 
 

 
 [329] 
 UCI machine learning repository, http://www.ics.uci.edu/~mlearn/MLRepository.html. 
 
 

 
 [330] 
 Kohonen, Teuvo (January 2013). ”Essentials of the self-organizing map”. Neural Networks. 37: 52–65. doi:1 0.1016/j.neunet.2012.09.018 .
 
 

 
 [331] 
 MPII Human Pose Dataset, http://human-pose.mpi-inf.mpg.de/ .
 
 

 
 [332] 
 Leeds Sports Poset Dataset https://datasets.activeloop.ai/docs/ml/datasets/lsp-dataset/ .
 
 

 
 [333] 
 Olympic Sports Dataset http://vision.stanford.edu/Datasets/OlympicSports/ .
 
 

 
 [334] 
 Compositional Freebase Questions (CFQ) Dataset, https://github.com/google-research/google-research/tree/master/cfq 
 
 

 
 [335] 
 Universal Dependencies Corpora https://www.tensorflow.org/datasets/catalog/universal_dependencies 
 
 

 
 [336] 
 Istat. (2018a). Indicators for the measurement of equitable and sustainable well-being. https://www.istat.it/en/well-being-and-sustainability/the-measurement-of-well-being/indicators .
 
 

 
 [337] 
 Eurobarometer dataset for service performance https://www.gesis.org/en/eurobarometer-data-service/ 
 
 

 
 [338] 
 Python implementaion of deep supervised visual similarity learning(2017) In : https://github.com/asanakoy/deep_unsupervised_posets .
 
 

 
 [339] 
 C. Squires, causaldag (2018), https://github.com/uhlerlab/causaldag .
 
 

 
 [340] 
 Poset-Hypergraph-Curvature Datasets, https://github.com/asamallab/Poset-Hypergraph-Curvature .