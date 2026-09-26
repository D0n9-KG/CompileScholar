Confidential Machine Learning on Untrusted Platforms: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2012.08156v2 [cs.LG] 12 Jun 2021 
 
 

# Confidential Machine Learning on Untrusted Platforms: A Survey

 
 
 SSSagar Sharma
 
 Address:  
HP Inc.,
Vancouver,
 \cny USA

 
    
 JRSKeke Chen
 
 Address:  
Department of Computer Science, Marquette University,
Milwaukee,
 \cny USA

 

 Abstract 
 
 With the ever-growing data and the need for developing powerful machine learning models, data owners increasingly depend on various untrusted platforms (e.g., public clouds, edges, and machine learning service providers) for scalable processing or collaborative learning. Thus, sensitive data and models are in danger of unauthorized access, misuse, and privacy compromises. A relatively new body of research confidentially trains machine learning models on protected data to address these concerns. In this survey, we summarize notable studies in this emerging area of research. With a unified framework, we highlight the critical challenges and innovations in outsourcing machine learning confidentially. We focus on the cryptographic approaches for confidential machine learning (CML), primarily on model training, while also covering other directions such as perturbation-based approaches and CML in the hardware-assisted computing environment. The discussion will take a holistic way to consider a rich context of the related threat models, security assumptions, design principles, and associated trade-offs amongst data utility, cost, and confidentiality.

 
 
 
 Keywords:  Machine Learning, 
 
 keywords 

 
 {fmbox} \dochead 
 Survey

 
 {abstractbox} 
 
 
 

## 1 Introduction

 
 Data-driven methods, e.g., machine learning and data mining, have become essential tools for numerous research and application domains. With abundant data, data owners can build complex analytic models for areas ranging from social networking, healthcare informatics, entertainment, and advanced science and technology. However, limited in-house resources, inadequate expertise, or collaborative/distributed processing needs force data owners (e.g., parties that collect and analyze user-generated data) to depend on somewhat untrusted platforms (e.g., cloud/edge service providers) for elastic storage and data processing. As a result, cloud services for data analytics, such as machine-learning-as-a-service (MLaaS), have been rapidly growing during the past few years. While untrusted platforms refer to all non-in-house resources not directly owned by the data owner, we will use cloud services to represent them here forth.

 
 
 When outsourcing sensitive data (e.g., proprietary, human-related, or confidential data), data owners have raised concerns in privacy, confidentiality, and ownership [ 1 , 2 ] . On the one hand, cloud users cannot verifiably prevent the cloud provider from accessing their data; i.e., in practice, using public clouds often means one must fully trust the cloud provider. On the other hand, public cloud providers are not immune to security attacks leading to sensitive data breaches. Recent security incidents, including insider attacks [ 3 , 2 ] and external security breaches at the service providers [ 4 , 5 ] , show the risks are aggravating by day. Researchers and practitioners have developed solutions to protect the confidentiality of cloud data at rest. For example, Google Cloud Platform has allowed users to include an external key manager to store encrypted data on the cloud with a third party (e.g., Fortanix) stores and manages keys off the cloud. However, it remains a critical challenge for data owners and cloud providers to protect confidentiality in computing, i.e., learning models on the cloud, while protecting the confidentiality of both the training data and the learned models.

 
 
 In the past few years, researchers have made some progress in developing novel confidential machine learning (CML) approaches for model training with encrypted data. A successful CML approach is not straightforward. Unlike traditional machine learning approaches, a practical CML framework wrestles in balancing security (confidentiality) guarantees, costs, and model quality, while allocating appropriate workload distributions between cloud and client. Direct application of cryptographic and privacy-protection methods such as fully homomorphic encryption (FHE) [ 6 ] and garbled circuits (GC) [ 7 ] in a homogeneous fashion do not usually meet the criteria for practical CML approaches. Most efficient approaches have been using hybrid methods that combine multiple primitives instead of a homogeneous translation. Recent studies [ 8 , 9 , 10 , 11 , 12 , 13 ] have followed this direction to effectively reduce performance bottlenecks and other practicality issues in developing CML solutions. However, the underlying techniques in these studies scatter among several papers making the basic principles are unclear. The purpose of this survey is to uncover these basic principles and accurately organize the existing techniques under a unified framework so that researchers and practitioners can quickly grasp the development and challenges in this new area of research.

 
 
 Contributions and Organization Overview. 
Capturing a comprehensive view of a complex and new topic like confidential machine learning is challenging. We primarily focus on frameworks for model training using cryptographic techniques that guarantee strong (semantic) security with practical cost overburden. A complete machine learning service usually includes a model application (or model inference) component that applies the learned model to generate a prediction for new input data, equivalent to secure function evaluation. The confidential model inference is much simpler and in a more mature state than confidential model training, therefore, not covered in this survey. Interested readers may refer to the related studies about confidential inference with pre-trained models, such as Gilad-Bachrach et al. [ 14 ] , Bost et al. [ 15 ] , Hesamifard et al. [ 16 ] , and Rouhani et al. [ 17 ] .

 
 
 This survey paper presents a unified perspective on designing and implementing different CML model learning methods with state-of-the-art cryptographic approaches. Despite numerous machine learning methods [ 18 ] , the studies on CML methods have focused on only a few specific machine learning methods. On the other hand, researchers have applied several cryptographic methods to realize CML frameworks. We observe that many clever CML techniques apply to specific machine learning algorithms without clearly establishing the basic principles for extending these techniques to broader machine learning algorithms. To systematically understand the set of developed techniques in CML, we summarize them under a general framework, the decomposition-mapping-composition (DMC) procedure + design and selection of crypto-friendly algorithms. The DMC procedure involves: decomposing the target machine learning algorithm into several components, mapping these components to their cryptographic constructions, and finally composing the CML solution with the confidential component counterparts. Moreover, several CML approaches adopting the DMC process development have a unique additional feature: they use “crypto-friendly” alternative machine learning algorithms or components to achieve more efficient protocols. Keeping these observations in mind, we develop a systemization framework to summarize the design principles, strategies, cryptographic techniques, and optimization measures, which have been applied to solve the challenging problems in confidentially learning models over encrypted data.

 
 
 We organize the survey based on underlying design principles of CML rather than any specific machine learning problems. As part of the survey, we summarize the experiences and learnings in each category of CML topics as insights and gaps . This work promotes practical aspects of applying cryptographic primitives in CML at their current level of maturity. Focuses will be on how different frameworks balance the associated trade-offs amongst cost, confidentiality, and data utility or model quality in different threat models and privacy settings. The survey, however, does not cover the orthogonal line of research that aims to optimize fully expressive primitives such as FHE and GC schemes. This survey will be a great resource for researchers to adopt and advance privacy-enhancing technologies in solving novel research questions and for practitioners to learn the best practices and avoid common pitfalls.

 
 
 In the following sections, first, we will include the necessary background knowledge, notations, definitions, and the targeted threat model in Section 3 . Then, in Section 4 , we present the systematization framework along with the basic principles and methodologies in the CML development. After that, we briefly discuss the homogeneous approaches that aim to translate any machine learning algorithm into a confidential one with a single cryptographic primitive (Section 5 ). Next, we move to the main theme: the compositional hybrid approaches (Section 6 ), which have resulted in more efficient protocols for complex machine learning models. We will also cover several topics, such as security proofs and common evaluation methods for cryptographic protocols in Sections 7 and 8. Finally, we briefly review other non-cryptographic-protocol approaches, including the perturbation methods and hardware-assisted (e.g., SGX) methods.

 
 
 

## 2 Related Work

 
 A few survey papers are related to the topic of this paper. Shan et al. [ 19 ] focus on techniques for practical secure outsourced computation, using machine learning as a sample application. However, it does not comprehensively cover the major approaches as we do.
Attacks on the integrity of machine learning models have also raised serious concerns due to the wide applications of machine learning in real-life scenarios such as self-driving cars [ 20 ] . Different from our survey focusing on the confidentiality of the model learning process, Papernot et al. [ 21 ] focus on the integrity of training data, learning process, models, and model application.

 
 
 There are also several survey papers on a specific category of cryptographic primitives. Since the first fully homomorphic encryption scheme was published in 2009 [ 6 ] , it has been an active research area during the past decade. Acar et al. [ 22 ] have a comprehensive review about the current development of homomorphic encryption schemes. Secure multi-party computation methods, including the garbled circuits and secret sharing methods, have been actively developed for the past two decades. Readers may find more information from other sources [ 23 , 24 ] .

 
 
 Differentially private machine learning frameworks are somewhat related to CML but hold a distinct thread model that aims to share data and models. They assume that the data consumer (i.e., model developer or model users) is not trusted, who may try to reveal private information in the training data shared by data owners or data contributors. It does not protect the ownership of data and models as the purpose is to share them without breaching individuals’ privacy in the training data. Along with recent developments on differentially private deep learning such as Abadi et al.   [ 25 ] and Shokri et al. [ 26 ] , Ji et al. [ 27 ] and Sarwate and Chaudhari [ 28 ] also provide excellent surveys on this topic. Other studies in privacy-preserving data mining (PPDM) [ 29 , 30 , 31 , 32 ] also aim to share the data (and the models) while preserving individual’s privacy, thus excluded from our survey.

 
 
 

## 3 Preliminaries of CML Approaches

 
 In this section, we review the terms and concepts used in the literature. First, we look at the representative system architectures considered in the published confidential machine learning (CML) approaches based on cryptographic protocols. Then, we examine how different threat models, associated confidential assets, and considered attacks affect CML designs. Finally, we briefly describe prevailing cryptographic and privacy primitives that serve as the skeleton of most CML approaches.

 
 

### 3.1 System Architectures

 
 The CML research is motivated by the cloud computing paradigm and then extended to more scenarios, such as edge computing and services computing. Thus, we use “Cloud” as the representative of untrusted platforms in CML system architectures henceforth. Such a system may involve cloud providers, optional cryptographic service providers, data owners or application service providers, and data and model consumers.
Figure 1 shows an architecture with a data owner outsourcing its data and computation to a single cloud provider. The data owner must ensure the cloud provider does not compromise any proprietary and privacy-sensitive data. A few homomorphic-encryption-based frameworks, e.g., Graepel et al. [ 33 ] and Lu et al. [ 34 ] , present protocols for training machine learning models over encrypted data outsourced to a cloud provider without almost any engagement of the data owner. However, the associated cost makes these protocols unrealistic in real-life scenarios. An alternate strategy would involve the data owner in minimal tasks intermediately to simplify the single cloud architecture framework [ 12 ] . As long as the cloud takes the majority of the workload and the client’s cost is practical, e.g., linear or sublinear to the number of records, more efficient protocols can be possible.

 
 
 As some protocols become too expensive for the data owner to assist cloud-centric learning, the architecture was evolved to a multi-server(cloud) setting. A data owner may choose to rely on two or more cloud providers to reduce the overall expense of learning. The second party may be as equally capable as the first party [ 11 ] , or in the case of a cryptographic service provider (CSP), which manages keys and assists the cloud with intermediate decryption operations and light-weight computations [ 8 , 9 , 13 ] . The two non-client parties in such an architecture carry out secure multi-party computations without any of the parties learning the training data and the trained model. This setting also assumes that the two parties do not collude with each other, thus slightly more vulnerable than the client-cloud two-party setting. Figure 2 shows such a framework that uses a garbled circuit.

 
 
 

### 3.2 Threat Models

 
 In this section, we examine the widely accepted threat models in the context of CML. We focus on the following aspects: the assumptions on the adversaries and the related confidential assets in CML.

 
 
 Assumptions on Adversaries. 
Most CML approaches [ 13 , 11 , 8 , 9 , 33 ] adopt the honest-but-curious (or semi-honest) adversary model to describe the untrusted cloud provider. Honest-but-curious parties, by definition, perform their share of tasks obediently, i.e., guarantee data and model integrity and follow the pre-defined protocols exactly. However, they might clandestinely snoop the storage, interactions, and computations to learn private information. Data owners and data contributors’ concerns about data and model leakages, even when the infrastructure platforms are reputed, are alleviated by preserving the confidentiality of data and models. Many CML approaches also use an honest-but-curious cryptographic service provider to design more efficient protocols.

 
 
 Some CML approaches additionally address an adversary which actively seeks to compromise data and model confidentiality by performing additional probing tasks, e.g., by inserting crafted records or secretively running the algorithms on a selected record set offline. Sharma and Chen [ 13 ] address the possibility of an adversary who may actively track identifiable training records to the datasets and follow the computations to infer the information about other training records. Nikolaenko et al. [ 9 ] consider an adversary that selectively runs the machine learning protocol over an individual’s data to draw personal inferences from the learned models.

 
 
 Nevertheless, with either passive or active adversaries, CML approaches assume that the data and model integrity not be compromised at the end of the training. This assumption distinguishes CML from other studies such as attacks on machine learning by polluting training data or modifying learned models [ 35 ] .

 
 
 Moreover, the CML approaches often assume non-collusion between the involved parties, for example, between the cloud provider and the CSP [ 8 , 9 , 13 ] or the two cloud providers [ 11 ] in the two-server architecture. Collusion between the two parties in these frameworks directly compromises the privacy of the training data and learned models.

 
 
 Most CML approaches assume that data and model consumers are trusted, which is orthogonal to the applications of differential privacy [ 26 , 25 ] that specifically targets untrusted data and model consumers. Furthermore, CML approaches assume properly secured infrastructures and communication channels to exclude external attacks and focus on the CML-specific challenges.

 
 
 Confidential Assets at Risk. 
An adversarial party may be interested in the confidentiality of sensitive data and the generated models . All CML methods protect the training data feature vectors. Some methods designed for supervised learning [ 33 , 9 ] expose the training data labels to simplify their secure modeling algorithms with the assumption that knowing the labels will not bring significantly more information to adversaries, which might be false for some applications. Some CML studies also expose unprotected models [ 33 , 9 , 34 ] . However, recent studies [ 36 , 37 , 38 , 39 , 40 ] have shown that an adversary may use crafted data to infer sensitive training data or use the advanced features in deep learning models to breach data privacy. Furthermore, the intermediate results of outsourcing computations in the setting of federated learning, for example, the intermediate representation in a convolutional neural network learning, may reveal information about the private training data [ 26 ] . Thus, CML must protect both data and model confidentiality.

 
 
 

### 3.3 Cryptographic Primitives 

 
 The cryptographic primitives are the fundamental building blocks for CML approaches. Some of these primitives are more expressive – meaning they can implement more types of functions or higher-level functions. On the other hand, some primitives are more cost-efficient than others. To make this survey self-contained, in this section, we briefly cover the most frequently-used primitives in existing CML approaches.

 
 
 Additive Homomorphic Encryption (AHE). AHE schemes (e.g., Paillier encryption [ 41 ] ) allow the additive operation over encrypted messages without decryption. For any two integers α \alpha and β \beta , an AHE scheme allows the additive homomorphic operation: E ⁡ ( α + β ) = f ⁡ ( E ⁡ ( α ) , E ⁡ ( β ) ) E(\alpha+\beta)=f(E(\alpha),E(\beta)) where the function f f works on encrypted values. Conceptually, with one of the operands unencrypted, a ‘‘pseudo-homomorphic’’ multiplication between two messages can be expressed as a series of additions 1 1 
 1 
 
 
 Some methods like Paillier encryption [ 41 ] allow more efficient pseudo-homomorphic multiplication. , i.e., E ⁡ ( α ​ β ) = E ⁡ ( ∑ i = 1 β α ) E(\alpha\beta)=E(\sum_{i=1}^{\beta}\alpha) . With homomorphic addition and pseudo-homomorphic multiplication, one can derive pseudo-homomorphic dot-product of vectors, matrix-vector multiplication, and matrix-matrix multiplication. However, the unencrypted operands in these operations either need to be non-sensitive information or protected with some masking and de-masking mechanism [ 12 , 13 ] . ElGamal, Goldwasser-Micali, Benaloh, and Okamoto-Uchiyama cryptosystems are some additional examples of AHE schemes [ 22 ] .

 
 
 Somewhat Homomorphic Encryption (SHE). There are many encryption schemes in this category (e.g., BV, BGV, NTRU, GSW, BFV, and BGN [ 22 ] and their variations such as TFHE [ 42 ] and CKK [ 43 ] ). SHE schemes allow both homomorphic additions and multiplications over encrypted messages, while the number of consecutive multiplications is limited to a few. A popular SHE scheme used in CML is the ring learning-with-error (RLWE) scheme that relies on the intractability of the learning-with-errors (LWE) problem on polynomial rings [ 44 ] . Theoretically, RLWE supports arbitrary levels of multiplications. Therefore, it is considered to be fully homomorphic. However, due to the associated high cost for deeper levels of multiplications, RLWE is more suitable as a SHE scheme only (i.e., 1-3 levels of multiplications). A ciphertext in RLWE is represented as a two-tuple ( c 0 , c 1 ) (c_{0},c_{1}) , where c 0 c_{0} and c 1 c_{1} are polynomials. Let C i = ( c 0 , i , c 1 , i ) C_{i}=(c_{0,i},c_{1,i}) and C j C_{j} be the ciphertext of any two values. The encrypted addition of the two values is simply ( c 0 , i + c 0 , j , c 1 , i + c 1 , j ) (c_{0,i}+c_{0,j},c_{1,i}+c_{1,j}) . The encrypted multiplication is translated to a series of polynomial operations on the ciphertext elements. RLWE allows multiple levels of multiplication at a certain cost. For details, please refer to the paper [ 44 ] . Message packing [ 44 ] enables packing multiple ciphertexts into one polynomial, which considerably reduces RLWE’s ciphertext size and optimizes linear algebra operations [ 45 ] . HELib library [ 45 ] is a popular implementation of the RLWE scheme.

 
 
 Garbled Circuits (GC). Garbled Circuits (GC) [ 7 ] allow two parties, each holding an input to a function, to securely evaluate a function without revealing any information about the respective inputs. GC can express arbitrary functions using several basic gates such as AND and XOR gates in a secure two-party computation setting (usually with a Cryptographic Service Provider (CSP)). One party constructs the circuit, whereas the other evaluates it. Despite several GC cost optimization techniques, such as Free XOR gates [ 46 ] , Half AND gates [ 47 ] , and OTExtensions [ 48 ] , GC still incurs high communication costs. Therefore, one must carefully examine its use in composing CML frameworks. FastGC [ 49 ] and ObliVM [ 50 ] are two popular GC libraries.

 
 
 Randomized Secret Sharing (SecSh). The randomized secret sharing method [ 10 ] protects data by splitting it into two (or multiple) random additive shares outsourced to two (or more) non-colluding untrusted parties. The two parties compute on the respective shares and return the results also as random shares. Addition is straightforward as α + β = ( α 0 + β 0 ) + ( α 1 + β 1 ) \alpha+\beta=(\alpha_{0}+\beta_{0})+(\alpha_{1}+\beta_{1}) with α \alpha and β \beta distributed between two parties 0 and 1. Multiplication, however, is expensive as it depends on the beaver triplet generation method [ 10 , 11 ] , which further depends on expensive AHE or Oblivious Transfer (OT) schemes to exchange the intermediate results securely.

 
 
 Random Additive Masking. 
A data owner may generate a random mask to hide the sensitive data, which will be stripped off at a certain step in the CML protocol to recover the desired result. Due to its low cost, it frequently serves as an auxiliary tool for a complex protocol, for instance, in CML for spectral clustering [ 12 ] , boosting [ 13 ] , and matrix factorization [ 8 ] .

 
 
 
 

## 4 Systematization Framework

 
 It is challenging to have a clear understanding of the whole body of CML model training methods due to the following reasons. First, the number of machine learning models is huge [ 18 ] and even the most used ones are around tens [ 51 ] . They are so different that no unified framework can be used to describe them. Second, security researchers are often more interested in a specific utility-preserving cryptographic primitive method and pick the machine learning algorithms they are most familiar with. As a result, the results are scattered with focuses on either a specific machine learning model or the application of a novel cryptographic primitive. There is no thorough understanding of which primitive method (or framework) is best for a specific machine learning method or whether a CML method can be extended to other machine learning models. The fundamental principles are missing for solving all (or most) CML model training methods.

 
 
 Categories of CML approaches. We believe this survey is the first effort to systematically organize and analyze the whole body of most representative CML approaches. We focus on the major category of methods: the pure software-based cryptographic protocols , while also briefly reviewing the perturbation-based approaches and the hardware-assisted approaches. Figure 3 shows the systematization framework. The fundamental features of the three categories are as follows.

 
 • 
 
 The cryptographic protocols are the focus of this survey, which can be further divided into two categories: those using one cryptographic primitive homogeneously and those employing novel hybrid compositions of multiple primitives. The homogeneous approaches take one of the homomorphic encryption (HE) schemes or garbled circuits to develop the solution. The hybrid approaches involve multiple primitives and often a clever composition strategy to achieve lower overall costs. We will analyze them in more detail.

 

 • 
 
 The perturbation-based CML approaches depend on novel data transformations to preserve a certain type of data utility, e.g., Euclidean distance, that is critical to one or multiple machine learning methods. Their security mainly depends on secret transformation parameters and random noise addition, holding a different and somewhat weaker security notion compared to cryptographic protocols. However, they are often much more efficient and thus appealing for many applications that seek better protection than plaintext-based approaches while not taking significantly more overhead.

 

 • 
 
 The third category depends on trusted execution environment , such as Intel SGX [ 52 ] , which demands hardware-level supports and are thus distinct from the former two categories of pure software approaches. The hardware-level features enforce secure enclaves , in which the adversaries cannot observe the running programs and data.

 

 
 
 
 Common CML Development Strategies. We look into a unified framework to analyze both the homogeneous and hybrid approaches. Fundamentally, most approaches aim to design an efficient and secure transformation of the specific (or a class of) machine learning algorithms for the setting of two or three distributed parties (see Section 3.1 ). To make the transformation easier, researchers often implicitly use the Decomposition-Mapping-Composition (DMC) procedure: decomposing the target algorithm into different subcomponents, mapping the sub-components to crypto-primitives, and composing the CML framework with the confidential sub-components. Many approaches skip the description of this whole procedure and only present the final composition, which creates difficulties for newcomers to fully appreciate the fundamental ideas scattered in several approaches.

 
 
 Beyond the straightforward DMC procedure, we have also noticed a unique feature [ 13 ] specific to the CML development: finding “crypto-friendly” alternative machine learning algorithms or components. This feature is unique to machine learning algorithms because all machine learning algorithms essentially try to find an approximate model fitting the training data, and there is no unique model for a specific problem, only better or worse ones. In general, machine learning methods can be roughly categorized into two types: supervised learning that depends on labeled datasets and unsupervised learning [ 18 ] . For each type, there are numerous algorithms working under the same setting but performing differently for specific applications or datasets. Even for the same algorithm, there are many variants. For example, different base classifiers can be used to make ensemble classifiers [ 53 ] , and different activation functions can be used for neural networks [ 54 ] . Among so many machine learning algorithms, some are more crypto-friendly, i.e., they can be converted to more efficient CML solutions.

 
 
 With all these features in mind, we reassemble the common development framework behind most CML approaches (Procedure 1 ).

 
 
 Procedure 1 A common procedure for developing CML methods 
 
 
 1: 
 
 procedure Generalized procedure for CML development ( A A ) 
 
 
 2: 
 
    A A : the target algorithm

 
 
 3: 
 
   Identify the desired architecture and involved parties.

 
 
 4: 
 
   Identify a list of alternative algorithms of A A 

 
 
 5: 
 
    for Each candidate algorithm do 
 
 
 6: 
 
    decompose the algorithm to basic components

 
 
 7: 
 
     for Each component do 

 
 
 8: 
 
      identify possible approximate/equivalent solutions

 
 
 9: 
 
       for Each solution do 

 
 
 10: 
 
       identify candidate crypto-primitive mappings

 
 
 11: 
 
       end for 
 
 
 12: 
 
      select the best solution and mapping.

 
 
 13: 
 
     end for 
 
 
 14: 
 
    find the best composition method.

 
 
 15: 
 
    end for 
 
 
 16: 
 
   evaluate the candidate alternative algorithms and identify the best one.

 
 
 17: 
 
 end procedure 
 
 
 
 
 Note that most of the steps in this procedure cannot be automated, and thus each specific approach represents a result of enormous efforts behind the scene. Next, we analyze the homogeneous and hybrid approaches under this unified procedure.

 
 
 

## 5 Homogeneous Cryptographic Approaches

 
 Homogeneous approaches rely on a single primitive to construct the framework protocols. The primitives used in the homogeneous composition of CML are broadly in two categories: (1) Fully Homomorphic Encryption (FHE) and Garbled Circuits (GC) and (2) Additively Homomorphic Encryption (AHE) and Somewhat Homomorphic Encryption (SHE). Since FHE implements arbitrary levels of homomorphic addition and multiplication and GC implements the boolean gates, in theory, they can individually construct all CML algorithms. FHE and GC are, therefore, the most expressive privacy primitives. However, both FHE and GC are too expensive to be practical when mapped to for training complex CML models. Oppositely, AHE and SHE schemes provide limited support for encrypted operations, therefore, less expressive and can only enable relatively simple algorithms. Most approaches we discuss next are relatively simple, and thus AHE or SHE scheme is sufficient. The decomposition and mapping steps of the DMC procedure described in the last section are still at play in the homogeneous approaches, but the composition step is trivial.

 
 
 AHE and SHE are widely used to construct homogeneous solutions for applications involving only one or a few multiplications, including the elementary statistical aggregation functions, such as average, sum, and variance. Graepel et al. [ 33 ] present a SHE-based framework for learning Fisher’s linear discriminant analysis and Linear Means Classifier models on encrypted data. However, the implemented models are limited to linearly separable datasets. Lu et al. [ 34 ] apply SHE for more sophisticated principal component analysis, and linear regression training [ 18 ] . However, due to the limited message space of the selected SHE implementation (60-bits in HELib) and the limited number of possible multiplications, only low data dimensionality (about 20) and a few training iterations were used in their evaluation. Such restrictions, however, resulted in only sub-optimal models.

 
 
 More sophisticated machine learning algorithms often result in expensive homogeneous solutions. Phong et al. [ 55 ] employ LWE and Paillier encryption in encrypting the gradients in their privacy-preserving deep learning framework. The framework, however, takes over 2.5 hours to complete one iteration of a simple neural network training for 20,000 MNIST images. Researchers also aim to provide libraries for homogenous learning based on Garbled Circuits (GC). However, their uses are limited in practicality due to huge costs [ 50 ] . Liu et al. [ 50 ] present a GC-based KMeans learning framework that involves two untrusted servers. The associated cost overburden, however, is far from efficient in real-world settings. For example, the KMeans implementation required over 2,000 million AND gates and more than 200 GB communication for clustering just 6,000 data points. Rouhani et al. [ 56 ] propose a deep learning model inference frameworks using garbled circuits to protect both the model’s parameters and test data samples. Similarly, the costs are staggeringly high.

 
 
 Insight. Homogeneous solutions are often limited to simple functions involving only additions (for AHE), a few multiplications (SHE), or a few comparisons (GC). Individually, these crypto primitives are not practical to construct complex CML algorithms. However, they can be valuable components for hybrid solutions, as we will see later. 

 
 
 

## 6 Hybrid Composition

 
 As discussed above, depending on a single cryptographic primitive to compose a sophisticated CML algorithm is impractical. However, each primitive has its unique strengths and shortcomings (e.g., performance, storage, bandwidth advantage) in attaining certain operations. This realization leads to an interesting strategy: can we combine different primitives in such a manner to compose secure yet more optimized protocols? The idea of hybrid composition is thus, mixing and switching amongst several privacy primitives to avoid the associated cost bottlenecks and restrictions of any individual primitive.

 
 
 This section will look into the details of specific steps of the DMC procedure. First, we dissect the common sub-components and underlying operations in machine learning algorithms. We examine the various ways to implement these sub-components and operations confidentially. Then, we explore the different switching and mixing strategies, including some recent automated ones, essential to hybrid CML frameworks in practice. Finally, we discuss the unique feature or desired requirement of CML development: designing crypto-friendly machine learning algorithms or sub-components for cost-efficient and practical CML solutions.

 
 

### 6.1 Basic Operations

 
 We devote this subsection to inspecting the mapping of the foundational sub-components of the target machine learning algorithms to their confidential versions. We observe that some of these mappings are practical or crypto-friendly, whereas others may face cost bottlenecks and limitations. The understanding of the different implementations of basic operations will affect the composition strategies.

 
 
 Simple Arithmetic Operations 
With AHE or a SHE encryption scheme, one can conveniently add two encrypted integers. Adding two b b bit integers with the Paillier cryptosystem involves modular multiplication with O( b 2 b^{2} ) complexity. Additions with an RLWE-like scheme involve polynomial additions linear to the number of bits for the given polynomial degree [ 57 ] . With a specific integer encoding, subtraction becomes trivial expressed as encrypted additions. SHE schemes allow homomorphic multiplications over encrypted integers. RLWE-like crypto-systems allow several rounds of multiplications and additions. However, with each additional multiplications, the ciphertext noise, cipher size, and cost increase. Generally, multiplying two b b bit integers with RLWE-like crypto-systems involves homomorphically computing O( b 2 b^{2} ) AND circuits [ 57 ] . On the other hand, the AHE scheme requires one of the operands to be unencrypted to realize multiplication expressed as summations. With Paillier encryption, multiplication is modular exponentiation of encrypted b b -bit message by the unencrypted b b -bit operand with a cost complexity of O( b 3 b^{3} ). The only caveat of using AHE-multiplication is that if the unencrypted operand is privacy-sensitive, a mechanism to mask it needs to be augmented, the masking recoverable after the multiplication is complete [ 12 , 8 ] .

 
 
 Additions and subtractions are trivial with randomized secret sharing in the multi-party setting with constant time complexity. Each party performs additions and subtractions on respective shares of data and shares the results for recovery. A GC protocol for addition requires two parties to construct O ⁡ ( b ) O(b) many AND gates and carry out O ⁡ ( b ) O(b) communication, encryptions, and decryptions along with O ⁡ ( b ) O(b) oblivious transfers when adding two b b bit integers. Multiplication with randomized secret sharing involves a costly multiplicative triplet generation scheme that relies on Oblivious transfer or AHE [ 10 , 11 ] . For example, the AHE-based scheme incurs transmission of two encrypted integers between the parties and performing two homomorphic encryptions, multiplications, additions, and decryptions by each party. Multiplying two integers of b b bits with GC, on the other hand, requires construction and evaluation of O ⁡ ( b 2 ) O(b^{2}) AND gates involving O ⁡ ( b 2 ) O(b^{2}) communication, encryption, and decryption.

 
 
 Comparison. 
Comparison is essential in many operations, such as sorting vectors and applying activation functions in training neural networks. Unfortunately, comparing two encrypted or protected integers is not trivial. Graepel et al. [ 33 ] pose the complexity of comparison as the reason to avoid algorithms like perceptrons and logistic regression in their SHE-based confidential ML framework. Veugen [ 58 ] presents a client-server interactive comparison protocol for two encrypted integers based on the AHE scheme, which involves computation and transfer of b b many AHE encrypted bits. Each comparison incurs O ⁡ ( b ) O(b) homomorphic multiplications for both client and server. Lu et al. [ 34 ] use the technique of “greater than” protocol [ 59 ] optimized with the message packing of the RLWE scheme for comparing two encrypted messages in a two-party setting. However, the associated complexity is an astonishing O ⁡ ( 2 b / h ) O(2^{b}/h) of homomorphic additions when comparing two b b -bit integers while packing h h messages in a ciphertext. With GC, a comparison between two b b -bit integers is possible with O ⁡ ( b ) O(b) AND gates and O ⁡ ( b ) O(b) communication, encryption, and decryption by two parties. Since GC-based comparison for full integers is expensive, one may use an efficient one-bit sign checking protocol [ 11 , 13 ] by encoding negative integers as two’s complement, making the comparison cost is constant to the number of bits. Note that the GMW protocol of Goldreich, Micali, and Wigderson [ 60 ] can perform comparisons just as garbled circuits but with O ⁡ ( b ) O(b) rounds. A similar sign-checking protocol is possible with GMW. However, the GC-based comparison seems the popular choice in current solutions.

 
 
 Division. 
Division can be essential to many analytics algorithms, e.g., from the computation of mean to the implementation of complex algorithms such as K-means [ 61 ] and Levenshtein distance [ 62 ] . Despite its prevalence and importance, translating division to its confidential version is expensive and often results in a performance bottleneck [ 63 ] . Veugen [ 58 ] presents a protocol for exact division in a client-server scenario, using the AHE scheme and additive noise masking. However, the protocol requires the divisor to be public knowledge. On top of that, the protocol requires O ⁡ ( b ) O(b) homomorphic comparisons and O ⁡ ( b ) O(b) encrypted communication for division between two b b -bit integers. Dahl, Chao, and Tomas [ 64 ] present two AHE-based division schemes that rely on Taylor approximation in a secure multi-party setting. The schemes brought expensive O ⁡ ( b ) O(b) encrypted communication. It is possible to perform integer divisions with GC when the two parties hold the numerator and denominator respectively in a 2-party setting [ 63 , 9 ] . However, even with several optimizations, a division between two b b -bit integers involves the construction and evaluation of a circuit with O ⁡ ( b ) O(b) non-XOR gates [ 63 ] . A more practical solution would be to decrypt the operands at a crypto-service provider and conduct division on plaintext before finally encrypting the result.

 
 
 Linear Algebra Operations. 
Linear algebra operations, such as vector dot products, matrix-vector multiplication, and matrix-matrix multiplications, are the core operations for many machine learning algorithms. They are commonly implemented with the cryptographic versions of additions and multiplications with some tricks in RLWE-based SHE for improved efficiency. Among all available methods, the AHE and SHE-based implementations are the most efficient ones.

 
 
 A dot product x k T ​ y k x_{k}^{T}y_{k} involves O ⁡ ( k ) O(k) element-wise homomorphic multiplications and additions. Similarly, a matrix-vector multiplication A n × k ​ x k A_{n\times k}x_{k} involves O ⁡ ( n ​ k ) O(nk) homomorphic multiplications and additions, and a matrix-matrix multiplication A n × k ​ B k × m A_{n\times k}B_{k\times m} involves O ⁡ ( n ​ k ​ m ) O(nkm) multiplications and additions. With the AHE scheme, one of the operands must remain unencrypted for these multiplicative operations. Therefore, the unencrypted operand needs some level of protection, e.g., novel randomized masking [ 12 ] with a minimized cost. With the message packing feature for the RLWE-like SHE scheme, one can easily vectorize the vector and matrix operations with message packing to gain more efficiency [ 45 ] . With such facilities, Jiang et al. [ 65 ] can optimize matrix-matrix multiplication with only O ⁡ ( k ) O(k) complexity for symmetric matrices of k k dimensions.

 
 
 Randomized secret sharing enables linear algebraic operations with the multiplicative triplet generation approach in a multi-party setting. However, this involves the expensive AHE or OT-based multiplicative triplet generation schemes as used in [ 11 , 10 ] . In computing a matrix-vector multiplication A ​ b Ab , each party is responsible for O ⁡ ( n + k ) O(n+k) encryptions and upload, O ⁡ ( n ​ k ) O(nk) homomorphic multiplications, O ⁡ ( n ​ k + n ) O(nk+n) homomorphic additions, and O ⁡ ( n ) O(n) decryptions.

 
 
 One can easily map linear algebra operations to garbled circuits. GC-based vector and matrix addition/subtraction require O ⁡ ( k ​ b ) O(kb) and O ⁡ ( n ​ k ​ b ) O(nkb) AND gates where b b is the number of bits in the vector and matrix elements. They also result in O ⁡ ( k ​ b ) O(kb) and O ⁡ ( n ​ k ​ b ) O(nkb) communication, encryption, and decryption operations, respectively. GC-based dot product for two b b bit vectors with k k dimensions is a collection of sub-circuits for multiplication and additions, which consist of O ⁡ ( k ​ b 2 ) O(kb^{2}) AND gates. The cost also involves O ⁡ ( b 2 ) O(b^{2}) encryption and decryption, and O ⁡ ( b 2 ) O(b^{2}) encrypted communication. The GC-based dot product can easily extend to matrix-vector and matrix-matrix multiplication. However, GC-based linear algebra solutions are more expensive than HE-based ones.

 
 

#### 6.1.1 Empirical Cost Comparison

 
 We have formally analyzed different crypto implementations for each of the major operations. However, some of them look close in terms of bigO complexity levels. To have a better idea how the cost differences look like for the different implementations of the same operator, we also prepare Table 1 . Since this comparison rests on a specific hardware configuration and software implementation, readers should only focus on the relative differences rather than the actual numbers. After a careful study of available AHE and SHE implementations, we choose the most efficient one for each category: we use the HELib library [ 66 ] for the RLWE encryption scheme and implement the Paillier cryptosystem [ 41 ] for the AHE encryption scheme. We adopt the ObliVM (oblivm.com) library for the garbled circuits. We also take the AHE scheme for the multiplicative triplet generation when using the randomized secret sharing (SecSh) method. We pick cryptographic parameters 2 2 
 2 
 
 
 The Paillier cryptosystem uses a 2048-bit key size. We set the degree of the corresponding cyclotomic polynomial in the RLWE scheme to ϕ ⁡ ( m ) \phi(m) = 12, 000 and c = 7 modulus switching matrices, which gives us h = 600 slots for message packing. corresponding to 112 112 -bit security. All schemes allow at least 32-bit messages-space overall. The RLWE parameters allow one full vector replication and at least two levels of multiplication. Note that the GC and SecSh costs are for the two-party setting, which has to involve communication costs between the two parties. Thus, we also include the bytes of exchanged messages for these methods. We run the experiments on an Intel i7-4790K CPU running at 4.0 GHz using 32 GB RAM with Ubuntu 18.04.

 
 
 Table   1 compares the related costs of arithmetic operations over integers. We have observed that the AHE scheme has the most efficient arithmetic additions and multiplications. However, for comparison and division, the 2-party garbled circuits are the only viable option. The table also shows the costs for the linear algebraic operations. The observation is consistent with the simpler arithmetic operation of additions and multiplications. As we can fit multiple messages in a ciphertext when using the RLWE scheme, the vectorized additions and multiplications are much more efficient than the non-vectorized additions and multiplications. The RLWE with message packing realizes homomorphic additions more efficiently when compared to the Paillier scheme. The RLWE costs for dot product and matrix-vector multiplication involve the ciphertext replication costs. Although better than without message packing, the RLWE scheme with the vectorized linear algebraic operation is still slower than the Paillier solutions. Randomized secret sharing is almost free for vector addition but involves higher computation and communication costs for the dot product and matrix-vector multiplication. Garbled circuits appear to be the worst solution for the confidential versions of the linear algebraic operation with higher computation and communication costs between the two parties. Although the Paillier implementation shows performance advantages over RLWE on arithmetic operations, it requires one operand to be plaintext. Paillier’s encryption and decryption costs, however, are higher than that of RLWE [ 12 ] . When CSP is involved in a solution, encryption and decryption costs will become a critical performance factor. These cost comparisons on the basic operations will be useful for readers to analyze and compare a pair of CML protocols, especially when not all CML methods are open-source.

 
 
 We do not experimentally compare complete CML approaches because 1) different approaches often solve different ML problems, which makes the comparison difficult, and 2) not all approaches have open-sourced their implementation or shared executable binaries. However, we hope the empirical comparison between different implementations for basic operators gives an intuitive understanding of the rationales behind different CML design strategies and optimization methods. We refer readers to the papers describing CML approaches that often contain detailed performance comparisons between selected CML approaches.

 
 
 Insight. 
Based on most studies, the most efficient constructions for confidential comparison are GC-based, while SHE and AHE are better candidates for linear algebra operations. Since most division schemes are too expensive, one should consider transforming the functions/algorithms with divisions to the equivalent (often approximately) ones that involve no division.
 

 
 
 
 

### 6.2 Switching and Composing Strategies

 
 When composing the confidential versions of operations implemented with different primitives, there is an important step: switching computation flows between the primitives. This switching often requires a second party in the CML frameworks, i.e., either the data owner, the second non-colluding cloud, or a CSP to achieve better performance.

 
 
 HE to/from GC. Switching from a HE component to a GC component involves a second server (e.g., a CSP) in the framework. A straightforward approach would be including a data decryption circuit inside a garbled circuit to be evaluated by the two parties. However, such an approach is super-expensive [ 8 ] . A more practical strategy [ 8 , 9 , 13 ] is to have the party holding the encrypted data, denoted P A P_{A} , mask it homomorphically before sending it to the second party, P B P_{B} for decryption. The second party constructs the desired garbled circuit, where the first step of the garbled circuit is de-masking the data with inputs: the decrypted masked data from P B P_{B} and the mask from P A P_{A} .

 
 
 SecSh to/from GC. Switching from a SecSh component to a GC component is straightforward in a two-party architecture. The two random shares in possession of the two parties can be their respective private inputs to the desired garbled circuits [ 11 , 13 , 67 ] . Similarly, switching from GC to SecSh involves evaluating the GC and randomly distributing the output to two parties [ 67 ] .

 
 
 SecSh to/from HE. A switch from randomized secret sharing to a HE component needs two involved parties to encrypt their respective shares. Then, one of the parties homomorphically reconstructs the protected value from the shares. Similarly, a switch from a HE component to a randomized secret sharing protocol includes a masking mechanism (homomorphic noise addition) similar to the HE-to-GC switch discussed above. These two switches are relevant in the AHE-based multiplicative triplet generation protocol for randomized secret sharing [ 11 , 10 ] .

 
 
 Table 2 provides some examples of switching between cryptographic primitives in well-known CML approaches. These switchings lead to simplification of the CML framework and cost optimizations, as explained in the “Justification” column of the table.
The ABY framework [ 10 ] covers different adapter-like switching protocols for the multi-party computation settings, where two servers hold the training data as arithmetic, boolean, or Yao’s garbled shares. The ABY3 [ 68 ] and BLAZE [ 69 ] framework extend the switches to 3-party scenarios. These works, however, do not cover the switching from and to the homomorphic encryption schemes.

 
 
 Manual vs. Automated Composition. 
Most existing CML approaches using the hybrid composition strategy [ 11 , 13 , 12 , 9 ] are manually composed as there are myriads of problem-specific details to address. A line of research explores the possibility of automatically composing the CML frameworks [ 70 , 71 ] . Although promising, the automatic composition strategy of Dreier and Kerschbaum [ 70 ] depends on the availability of an extensive performance matrix for the different confidential versions of the target algorithms’ components. Henecka et al. [ 71 ] propose the TASTY compiler that automatically compiles a given machine learning problem as a mixture of garbled circuits and homomorphic encryption in a secure two-party computation framework. However, the process is still not fully automated - it requires a privacy expert to design and specify the components as well as the recommended mappings.

 
 
 Gap. Due to the high complexity of formulating the component-wise costs and profiling the switching costs, the automated composition approaches are not yet fully mature. More importantly, as we will see in the next section, the construction of a practical CML solution involves one more crucial step that automated composition methods cannot help much. One must establish an in-depth understanding and analysis of the target ML algorithm to redesign a “crypto-friendly” algorithm.
 

 
 
 

### 6.3 Crypto-friendly ML Algorithms

 
 So far, the DMC framework seems straightforward: one decomposes the target machine-learning algorithm to its sub-components and maps them to cryptographic constructions, and the final composition becomes almost trivial except that the primitive switching requires some clever steps. With enough experimentation, one can find an optimal set of confidential components for the target ML algorithm. However, this straightforward strategy may only work for some problems. Despite the best optimization of mapping and composition, one may still end up with an impractical protocol, although better than the homogeneous or other suboptimal compositions. The fundamental reason is that the original machine learning algorithms do not account for confidential computation. They are optimized to achieve the best model prediction power rather than to be crypto-friendly. On the other hand, a less-known slightly-under-performing ML algorithm that attains the same learning goal might be more cost-effective to translate to its confidential version. Thus, an advanced design step critical to the DMC procedure is replacing or redesigning some of the underlying ML components or even the entire ML algorithm to find the most efficient CML protocols.
Table 3 summarizes some example CML frameworks that incorporate strategies to make their protocols crypto-friendly and hence more cost-effective. Mohassel et al. [ 11 ] , in their SecureML work, substitute the expensive softmax operation involving inverses with a ReLU-based function involving only one division. This way, the framework significantly reduces the cost bottlenecks in their protocol. Graepel et al. [ 33 ] cleverly avoid division of encrypted data in the framework for confidential linear means classifier and Fisher’s linear discriminant analysis by replacing divisions with a multiplicative factor. Nikolaenko et al. [ 9 ] use the more efficient Cholesky’s decomposition instead of the expensive LU decomposition in solving a system of linear equations in their linear regression framework. Similarly, Nikolaenko et al. [ 8 ] adopt the sorting-based matrix factorization solution to reduce the overall complexity of computing gradient descent with Cholesky’s decomposition-based matrix factorization. Sharma and Chen [ 13 ] propose to train a boosting classifier over encrypted data with an ensemble of random linear classifiers (RLC) instead of decision stumps. An RLC takes mere N N encrypted comparisons, whereas a decision stump takes far too many comparisons. Naehrig et al. [ 72 ] replace the exponential function (the sigmoid) in their logistic regression protocol with the Taylor approximation of exponentiation. Computing the exact exponential function would have led to the computation of many levels of multiplications over the encrypted message – which would have been intolerably expensive with SHE schemes. Similarly, Sharma et al. [ 12 ] replace the inherently expensive eigendecomposition O ⁡ ( N 3 ) O(N^{3}) with cheaper O ⁡ ( N 2 ) O(N^{2}) approximation algorithms of Lancozs and Nystrom in their spectral clustering framework.

 
 
 Data reduction techniques such as subsampling and preserving the sparsity of matrix are also critical to performance. Nikolaenko et al. [ 8 ] , in their matrix factorization framework, use a sorting network that optimizes the garbled circuit-based gradient descent algorithm by only updating it for the user ratings that are present in the training dataset. Similarly, Sharma et al. [ 12 ] propose a differential privacy-based graph submission mechanism that reduced total storage by over 15 times and costs involving encryptions and the associated homomorphic operations by over 20 times on the graph drastically when running the secure Nystrom method for spectral clustering. To sum up, although the approximate algorithms introduce some degradation to the learned models, they deliver desired cost practicality justifying the tolerable quality sacrifice.

 
 
 Insight. 
For the same learning problem, there are numerous algorithms. Even for the same learning algorithm, there are many variants [ 18 ] . The search space for optimal composition can be quite large. More difficultly, most well-known ML algorithms are best known for model quality or learning efficiency and none specifically designed with optimal CML in mind. Even worse, some crypto-friendly alternatives might have been forgotten or become obsolete due to their suboptimal quality or efficiency. The design of a good CML solution heavily depends on the designer’s deep understanding of the ML algorithms and even the history of ML algorithm development.
 

 
 
 Gap. There is no systematic way to explore crypto-friendly alternative ML algorithms. The current practice is to design a problem/algorithm-specific crypto-friendly solution. Although the problem-specific design experiences and learnings can extend to a new solution design, there are no well-known rules or general frameworks for exploring such alternative ML algorithms yet.
 

 
 
 
 

## 7 Security Proofs, Attacks, and Correctness

 
 In this section, we summarize the three aspects: security proofs, attack analysis, and correctness for existing CML approaches, which are commonly discussed in other cryptographic protocols.

 
 
 Security Proofs. Homogeneous approaches do not use complex protocols other than the cryptographic primitive they use. For example, homomorphic encryption-based approaches involve only simple interactions between the client and the cloud - the client submitting the data and the cloud computes and returns the result; the GC-based methods have two involved parties following the fundamental GC protocols. Thus, most such approaches simply skip the security proof step, fully depending on the proven security and privacy guarantees provided by the primitives.

 
 
 For hybrid approaches, it’s more sophisticated to prove their security, as they may include complex interactions among parties. We have observed two security proof frameworks are in prevalence. SecureML [ 11 ] utilizes the Universally Composable Security (UC) framework [ 73 ] . The UC security framework defines security-preserving universal composition operation and allows for modular design and analysis of complex cryptographic protocols from simpler building blocks. PrivateGraph [ 12 ] , SecureBoost [ 13 ] , and Lu et al. [ 34 ] adopt the simulation-based security proof [ 74 ] . The simulation approach needs to show the existence of a simulator in the ideal scenario that corresponds to the adversary in the real scenario, such that it is impossible to distinguish the interactions in the ideal scenario from those in the real scenario. The assumption of semi-honest parties held by most CML approaches makes the security proofs much easier [ 74 , 73 ] . As a result, many CML approaches ignore the steps of security proof.

 
 
 Attacks. To our knowledge, attacks on the confidentiality of cryptographic CML approaches have not been fully explored. Most works we covered in this category did not mention any potential attacks on their approaches, partially due to the well-known security guarantees provided by the underlying primitives or formal security proofs provided by a few approaches. While all approaches want to fully protect feature vectors in the training data, some approaches require the labels (in supervised learning) to be exposed for easier modeling [ 33 ] , and some even expose the final learned models [ 9 , 34 ] . However, recent studies have shown that exposed models may lead to serious attacks, such as model inversion attacks [ 37 , 75 ] , and membership inference attacks [ 39 ] .

 
 
 Correctness. Contrary to some cryptographic protocols and encryption systems that need to prove their correctness (e.g., encrypted values can be correctly decrypted), the correctness of CML protocols is attached to the correctness of the original machine learning algorithms. The DMC procedure honestly reassembles the original learning algorithm with the cryptographic components. Thus, as long as the primitives preserve the correctness and the composition strategy does not change the correctness (see Section 6.2 ), the correctness property is guaranteed. However, when researchers adopt a crypto-friendly alternative algorithm or component, they must justify whether the alternative methods warrant/attain the desired learning objective. SecureBoost [ 13 ] depends on the basic boosting theory [ 53 ] that states any weak base classifier, including random weak linear classifiers, can be used for the boosting framework. Naehrig et al. [ 72 ] utilize the Taylor approximation of exponentiation to approximate the sigmoid function, which is a well-accepted mathematical method. While these alternative methods may affect the model quality, implying a potential trade-off between model quality and costs, they are all considered correct algorithms.

 
 
 Gap. Security proofs are missing for some existing CML approaches, which raise a concern that they may contain flaws leading to significant information leaks. Further studies are needed to rigorously analyze these approaches.
 

 
 
 

## 8 Evaluation Methods

 
 Researchers evaluate their proposed CML methods primarily based on costs and model quality. Some CML methods also involve trade-offs between these two aspects.

 
 
 Costs. CML researchers primarily concern about the costs of protocol, striving to find the most efficient secure protocols. Since multiple parties are involved, the costs for each party, i.e., the cloud provider, the client, and possibly the crypto-service provider or the second cloud provider, are all essential to the design of CML protocols. For a given CML method, each party’s costs are the outcome of the cost for comparing the encryption/
decryption, data transmission, and other computation overhead. Because of the original motivation of outsourcing large-scale
computation, a skewed cost distribution between the client and the cloud is fundamental, i.e., the client should take much lower overheads compared to the cloud [ 12 , 13 , 11 ] . However, the client may still take much higher costs when running CML protocols when compared to running the original non-secure ML solution. The cost of external storage and related I/O operations are also critical to the cloud-side components as they are responsible for storing the encrypted data, which often is much larger than the plaintext version and cannot reside in memory. It is also highly desired that the cloud-side computation can be done parallelly with a popular processing framework such as MapReduce [ 76 , 12 ] . Besides, when GC is adopted as a primitive to implement some components, additional communication cost related to the GC protocol is also significant, including the cost of transmitting the circuit and one-party’s input data obliviously to the other party [ 50 , 49 ] . As a result, the use of GC is limited to a few operations, such as comparison [ 10 ] . The overall computation and communication costs of different approaches are frequently compared and used as a measure to show the novelty of a new method. For example, Mohassel et al. [ 11 ] show their work is more computation efficient than the GC-based framework considered by [ 9 ] by about two orders of magnitude. Similarly, Sharma et al. [ 13 ] show their boosting solution is about three times faster than the neural network CML in [ 11 ] .

 
 
 Model Quality. Model quality, a unique feature of CML evaluation, is often tightly related to the cost of model training. Many machine learning algorithms are iterative, such as logistic regression, neural networks, and many clustering algorithms. As a result, model quality increases with the number of iterations until the process converges. However, a large number of iterations implies the increased overall costs. Some CML methods, e.g., Lu et al. [ 34 ] , may only report the overall costs for one/few iterations of a specific learning algorithm, which is insufficient unless the number of iterations necessary for optimal results is specified. More precisely, many works miss the requirement that model evaluation should be tied to the cost evaluation, i.e., how much cost is needed to reach a certain model accuracy [ 11 , 13 ] . The discussion on crypto-friendly alternative algorithms also holds the assumption that model quality can be possibly traded off with costs, with the expectation that the crypto-friendly alternative may perform comparably or slightly worse than the original machine learning algorithm [ 13 , 12 , 11 , 33 ] .

 
 
 

## 9 Other CML Approaches

 
 So far, we have focused on cryptographic methods based on well-known primitives. To cover a panoramic view of development in the growing area of confidential machine learning, we briefly discuss two closely related approaches, the perturbation-based approach and the hardware-assisted approach.

 
 

### 9.1 Perturbation Methods

 
 Most practical CML solutions that carefully follow the DMC process with some innovative uses of crypto-friendly ML algorithms still cost magnitudes more than the original plaintext algorithms. Especially if the learning algorithm is intrinsically expensive or relies on a massive-scale training dataset, the cryptographic primitives that provide semantic security may become impractically expensive, discouraging users from adopting the outsourcing paradigm. Another category of work: the perturbation-based approach offers much more efficient solutions with some weaker security notions. Often, they do not guarantee semantic security and may only be resilient to ciphertext-only attacks. Nevertheless, they can be interesting for users who are willing to make a practical trade-off between efficiency and the level of protection. We briefly discuss this body of work to extend readers’ interests to this unique domain.

 
 
 The basic idea of perturbation is injecting random noises into the outsourced data while (approximately) preserving some specific properties machine learning models rely upon. The most well-known properties are geometric and topological structures in the multidimensional space. Therefore, one can still train a model from the perturbed data on the untrusted platform with preserved confidentiality of both data and model. Typical perturbation methods include randomized response [ 77 , 78 ] , additive perturbation [ 79 ] , geometric perturbation [ 80 ] , random projection perturbation [ 81 ] , and random space perturbation [ 82 ] . They have been applied to decision tree learning [ 78 , 79 ] , clustering [ 80 , 81 ] , kNN classifier [ 80 ] , support vector machines [ 80 ] , linear classifier [ 80 , 83 ] , and boosting [ 83 ] . The perturbation mechanisms can also disguise the training images in deep learning frameworks [ 84 ] to achieve much lower training costs than cryptographic protocols [ 11 ] . Furthermore, the perturbation methods often do not involve expensive cryptographic primitives. Consequentially, one can observe significant cost savings in the entire life cycle of data analytics, including data submission, computation, and communication amongst the involved parties.

 
 
 Insight. The key idea of perturbation approaches is to identify a certain high-level utility and preserve it in secure randomized transformations. Similar ideas have also been explored in the cryptographic domain, such as order-preserving encryption [ 85 , 86 , 87 ] and encrypted keyword search [ 88 , 89 ] .

 
 
 Gap. 
Despite their efficiency, perturbation approaches face two critical weaknesses. First, perturbation methods may cause significant degradation to the data quality and introduce significant trade-offs between utility and confidentiality. Second, there is no systematic framework for analyzing the protection level guaranteed by a perturbation method. Some of them are known not to provide provable semantic security [ 80 , 82 ] . However, under a clear, rigorous threat model definition and thorough analysis, these methods will have high practical values in the venues where users can accept the specific threat model.
 

 
 
 

### 9.2 Hardware-Assisted Approaches

 
 During the past few years, hardware-assisted trustworthy computing has made a significant breakthrough. In particular, several CPU manufactures have implemented the trusted execution environment (TEE) platforms, among which the most popular one is Intel’s Software Guard Extensions (SGX) [ 52 ] . We will take SGX as an example in the following. SGX defines a specific memory area (e.g., the enclave ). Only the authorized owner can run programs and access data in the enclave via special instructions. Owners and users gain access rights via an attestation protocol. SGX minimized the trust boundary to the enclave, which means even though the entire operating system is compromised, adversaries cannot access the enclave. The physical enclave memory is limited (less than 100MB are usable by users). When the enclave memory pages are swapped out/in by the virtual memory management subsystem of the OS 3 3 
 3 
 
 
 The enclave virtual memory management is only enabled on the Linux system for early versions of SGX, which might be changed in newer versions of SGX , they are encrypted/decrypted by the SGX library functions implicitly. SGX uses AES encryption (is this always true?), and thus the encryption and decryption costs are much lower than the primitives we have discussed so far. Besides, since the enclave program works on decrypted data, there is no need to develop special CML algorithms for running inside the enclave, making SGX an appealing platform for developing CML solutions for complex algorithms working with large data.

 
 
 However, there are a few challenges for migrating algorithms to the SGX environment. First, users need to learn the whole SGX working mechanism and learn to use special instructions and APIs, which can be inconvenient. A few efforts have simplified the migration of applications to SGX, among which the Graphene-SGX library OS [ 90 ] , SCONE [ 91 ] , and Panoply [ 92 ] are the most well-known. With a tool like Graphene-SGX, developing CML solutions becomes more straightforward. Lee et al. [ 93 ] have tried to migrate machine learning algorithms to SGX based on Graphene-SGX. However, these methods do not address side-channel attacks.

 
 
 Second, side-channel attacks are considered the primary threat to SGX-based applications. As TEEs have prevented many traditional attacks and the assumption is now changed to adversary-controlled OS, side-channel attacks are active research areas. Memory side channels and cache side channels are the two types that researchers mostly examined. Memory side-channel attacks are primarily access pattern attacks [ 94 , 95 , 96 ] . As the encrypted data have to be loaded from the file to the untrusted area first and then accessed by the enclave, the access pattern attacks seem inevitable for data-intensive applications like CML. The well-known approach addressing this problem is the Oblivious RAM technique [ 97 ] , which has been applied to SGX by ZeroTrace [ 94 ] and Obliviate [ 95 ] . Ohrimenko et al. [ 98 ] also used oblivious access techniques for multi-party machine learning with SGX. Branching attacks [ 96 ] utilize the branching statements and manipulate page faults to extract information, often addressable with oblivious branching instructions such as CMOV [ 96 , 94 , 99 ] . Cache side-channel attacks such as cache timing and transient execution state [ 100 , 101 , 102 , 103 ] utilize the unique CPU architectural features and thus depend on the manufacturers’ firmware and software patches to fix. More studies are necessary to explore the full potential and unique problems with SGX-based CML.

 
 
 Insight. The TEE, e.g., SGX, techniques can significantly boost CML’s performance on untrusted platforms, as the solutions do not involve expensive crypto primitives or protocols. We consider the SGX based CML as a promising direction because it achieves a strong confidentiality guarantee with significant performance benefits compared to other approaches.
 

 
 
 Gap. The most critical challenge TEEs face is side-channel attacks, especially the access pattern attacks. Also, machine learning algorithms have unique features (e.g., data access, batching, etc.) that may lead to specific attacks that have not been fully explored yet. Another practical concern is that most recent Intel server CPUs still have not had SGX enabled. A few cloud platforms such as Microsoft Azure and IBM Cloud have started offering SGX-enabled instances, and thus we consider this gap of missing public SGX resources will be filled up soon. 

 
 
 
 

## 10 Conclusion

 
 Despite the potential risk of data and model leakages, many resource-constrained data owners use untrusted platforms (e.g., clouds and edges) for training machine learning models. Researchers have been designing and developing confidential machine learning (CML) approaches for outsourced data using cryptographic primitives and various composition strategies. CML’s overall goal is to protect the confidentiality of data, model, and intermediate results from the untrusted platforms while also preserving the trained model quality with acceptable costs.

 
 
 We have reviewed the recent significant developments on CML under a systemization framework, focusing on the cryptographic approaches. We have included the cryptographic primitives that are the backbone of the CML approaches and compared the costs for basic operations. While the homogeneous methods that rely on a single cryptographic primitive are straightforward, their solutions are too expensive to be practical. Thus, we focus on the primary design trend of the hybrid composition of multiple primitives under the decomposition-mapping-composition (DMC) procedure and the selection of crypto-friendly alternative learning algorithms. We describe the critical issues such as the switching between primitives and the principles of identifying crypto-friendly machine learning algorithms. Finally, we also include a brief discussion of related approaches and new directions, including the perturbation and hardware-assisted methods. At the end of most sections, we have also included a concise summary area labeled with Insight and Gap for readers to get the gist conveniently. We believe this survey can be valuable to both researchers and practitioners to build more complex and practical CML solutions in the future.

 
 
 

## Acknowledgements

 
 Not applicable

 
 
 

## Funding

 
 This work is partially supported by the National Science Foundation under grant no. 1245847 and the National Institute of Health under grant no. 1R43AI136357-01A1.

 
 
 

## Availability of data and materials

 
 Not applicable

 
 
 

## References

 
 
 [1] 
 
Sharma, S.,
Chen, K.,
Sheth, A.:
Toward practical privacy-preserving analytics for iot and cloud-based
healthcare systems.
IEEE Internet Computing
 22 (2),
42–51
(2018).
doi: 10.1109/MIC.2018.112102519 

 

 
 [2] 
 
Duncan, A.J.,
Creese, S.,
Goldsmith, M.:
Insider attacks in cloud computing.
In: 2012 IEEE 11th International Conference on Trust, Security and
Privacy in Computing and Communications
(2012)

 

 
 [3] 
 
Chen, A.:
Gcreep: Google engineer stalked teens, spied on chats.
Gawker, http://gawker.com/5637234/
(2010)

 

 
 [4] 
 
Mansfield-Devine, S.:
The Ashley Madison affair.
Network Security
 2015 (9),
8–16
(2015)

 

 
 [5] 
 
Unger, L.:
Breaches to customer account data.
Computer and Internet Lawyer
 32 (2),
14–20
(2015)

 

 
 [6] 
 
Gentry, C.:
Fully homomorphic encryption using ideal lattices.
In: Annual ACM Symposium on Theory of Computing,
pp. 169–178.
ACM,
New York, NY, USA
(2009)

 

 
 [7] 
 
Yao, A.C.:
How to generate and exhange secrets.
In: IEEE Symposium on Foundations of Computer Science,
pp. 162–167
(1986)

 

 
 [8] 
 
Nikolaenko, V.,
Ioannidis, S.,
Weinsberg, U.,
Joye, M.,
Taft, N.,
Boneh, D.:
Privacy-preserving matrix factorization.
In: ACM SIGSAC Conference on Computer and Communications Security,
pp. 801–812
(2013)

 

 
 [9] 
 
Nikolaenko, V.,
Weinsberg, U.,
Ioannidis, S.,
Joye, M.,
Boneh, D.,
Taft, N.:
Privacy-preserving ridge regression on hundreds of millions of
records.
In: IEEE Symposium on Security and Privacy,
pp. 334–348
(2013)

 

 
 [10] 
 
Demmler, D.,
Schneider, T.,
Zohner, M.:
ABY - A framework for efficient mixed-protocol secure two-party
computation.
In: 22nd Annual Network and Distributed System Security Symposium,
NDSS 2015, San Diego, California, USA, February 8-11, 2015
(2015).
 https://www.ndss-symposium.org/ndss2015/aby—framework-efficient-mixed-protocol-secure-two-party-computation 

 

 
 [11] 
 
Mohassel, P.,
Zhang, Y.:
Secureml: A system for scalable privacy-preserving machine learning.
In: 2017 IEEE Symposium on Security and Privacy (SP),
pp. 19–38
(2017)

 

 
 [12] 
 
Sharma, S.,
Powers, J.,
Chen, K.:
Privategraph: Privacy-preserving spectral analysis of encrypted graphs
in the cloud.
IEEE Transactions on Knowledge and Data Engineering
 31 (5),
981–995
(2019).
doi: 10.1109/TKDE.2018.2847662 

 

 
 [13] 
 
Sharma, S.,
Chen, K.:
Confidential boosting with random linear classifiers for outsourced
user-generated data.
In: Computer Security - ESORICS 2019 - 24th European Symposium on
Research in Computer Security, Luxembourg, September 23-27, 2019,
Proceedings, Part I,
pp. 41–65
(2019)

 

 
 [14] 
 
Gilad-Bachrach, R.,
Dowlin, N.,
Laine, K.,
Lauter, K.,
Naehrig, M.,
Wernsing, J.:
Cryptonets: Applying neural networks to encrypted data with high
throughput and accuracy.
In: Balcan, M.F.,
Weinberger, K.Q. (eds.)
Proceedings of The 33rd International Conference on Machine Learning.
Proceedings of Machine Learning Research,
vol. 48,
pp. 201–210
(2016)

 

 
 [15] 
 
Bost, R.,
Popa, R.A.,
Tu, S.,
Goldwasser, S.:
Machine learning classification over encrypted data.
In: Annual Network and Distributed System Security Symposium (NDSS)
(2015)

 

 
 [16] 
 
Hesamifard, E.,
Takabi, H.,
Ghasemi, M.:
Cryptodl: Deep neural networks over encrypted data.
CoRR
 abs/1711.05189 
(2017).
 1711.05189 

 

 
 [17] 
 
Rouhani, B.,
Hussain, S.U.,
Lauter, K.,
Koushanfar, F.:
Redcrypt: Realtime privacy preserving deep learning using fpgas.
ACM Transactions on Reconfigurable Technology and Systems (TRETS)
(2018)

 

 
 [18] 
 
Hastie, T.,
Tibshirani, R.,
Friedman, J.:
The Elements of Statistical Learning.
Springer,
New York City, New York
(2001)

 

 
 [19] 
 
Shan, Z.,
Ren, K.,
Blanton, M.,
Wang, C.:
Practical secure computation outsourcing: A survey.
ACM COMPUTING SURVEYS
 51 (2)
(2018)

 

 
 [20] 
 
Grigorescu, S.,
Trasnea, B.,
Cocias, T.,
Macesanu, G.:
A survey of deep learning techniques for autonomous driving.
Journal of Field Robotics
 37 (3)
(2019)

 

 
 [21] 
 
Papernot, N.,
McDaniel, P.,
Sinha, A.,
Wellman, M.P.:
Sok: Security and privacy in machine learning.
In: 2018 IEEE European Symposium on Security and Privacy (EuroS P),
pp. 399–414
(2018)

 

 
 [22] 
 
Acar, A.,
Aksu, H.,
Uluagac, A.S.,
Conti, M.:
A survey on homomorphic encryption schemes: Theory and implementation.
ACM Computing Surveys
 51 (4)
(2018).
doi: 10.1145/3214303 

 

 
 [23] 
 
Lindell, Y.:
Secure Multiparty Computation (MPC).
Cryptology ePrint Archive, Report 2020/300.
 https://eprint.iacr.org/2020/300 
(2020)

 

 
 [24] 
 
Evans, D.,
Kolesnikov, V.,
Rosulek, M.,
(2018).
doi: 10.1561/3300000019 

 

 
 [25] 
 
Abadi, M.,
Chu, A.,
Goodfellow, I.,
McMahan, H.B.,
Mironov, I.,
Talwar, K.,
Zhang, L.:
Deep learning with differential privacy.
In: Proceedings of the 2016 ACM SIGSAC Conference on Computer and
Communications Security.
CCS ’16,
pp. 308–318.
ACM,
New York, NY, USA
(2016).
doi: 10.1145/2976749.2978318 .
 http://doi.acm.org/10.1145/2976749.2978318 

 

 
 [26] 
 
Shokri, R.,
Shmatikov, V.:
Privacy-preserving deep learning.
In: Proceedings of the 22nd ACM SIGSAC Conference on Computer and
Communications Security
(2015)

 

 
 [27] 
 
Ji, Z.,
Lipton, Z.C.,
Elkan, C.:
Differential privacy and machine learning: a survey and review.
CoRR
 abs/1412.7584 
(2014)

 

 
 [28] 
 
Sarwate, A.D.,
Chaudhuri, K.:
Signal processing and machine learning with differential privacy:
Algorithms and challenges for continuous data.
IEEE Signal Processing Magazine
 30 (5),
86–94
(2013).
doi: 10.1109/MSP.2013.2259911 

 

 
 [29] 
 
Aggarwal, C.C.,
Yu, P.S.:
Privacy-Preserving Data Mining: Models and Algorithms.
Springer,
New York City, NY
(2010)

 

 
 [30] 
 
Matwin, S.:
In: Custers, B.,
Calders, T.,
Schermer, B.,
Zarsky, T. (eds.)
Privacy-Preserving Data Mining Techniques: Survey and Challenges,
pp. 209–221.
Springer,
Berlin, Heidelberg
(2013)

 

 
 [31] 
 
Aldeen, Y.A.A.S.,
Salleh, M.,
Razzaque, M.A.:
A comprehensive review on privacy preserving data mining.
SpringerPlus
 4 (1),
694
(2015).
doi: 10.1186/s40064-015-1481-x 

 

 
 [32] 
 
Sachan, A.,
Roy, D.,
Arun, P.V.:
An analysis of privacy preservation techniques in data mining.
In: Meghanathan, N.,
Nagamalai, D.,
Chaki, N. (eds.)
Advances in Computing and Information Technology,
pp. 119–128.
Springer,
Berlin, Heidelberg
(2013)

 

 
 [33] 
 
Graepel, T.,
Lauter, K.,
Naehrig, M.:
Ml confidential: Machine learning on encrypted data.
In: International Conference on Information Security and Cryptology,
pp. 1–21
(2013)

 

 
 [34] 
 
Lu, W.,
Kawasaki, S.,
Sakuma, J.:
Using fully homomorphic encryption for statistical analysis of
categorical, ordinal and numerical data.
In: The Network and Distributed System Security Symposium
(2017)

 

 
 [35] 
 
Liu, Q.,
Li, P.,
Zhao, W.,
Cai, W.,
Yu, S.,
Leung, V.C.M.:
A survey on security threats and defensive techniques of machine
learning: A data driven view.
IEEE Access
 6 ,
12103–12117
(2018).
doi: 10.1109/ACCESS.2018.2805680 

 

 
 [36] 
 
Fredrikson, M.,
Lantz, E.,
Jha, S.,
Lin, S.,
Page, D.,
Ristenpart, T.:
Privacy in pharmacogenetics: An end-to-end case study of personalized
warfarin dosing.
In: 23rd USENIX Security Symposium USENIX Security,
pp. 17–32.
USENIX Association,
San Diego, CA
(2014)

 

 
 [37] 
 
Fredrikson, M.,
Jha, S.,
Ristenpart, T.:
Model inversion attacks that exploit confidence information and basic
countermeasures.
In: ACM Conference on Computer and Communications Security
(2015)

 

 
 [38] 
 
Hitaj, B.,
Ateniese, G.,
Perez-Cruz, F.:
Deep models under the gan: Information leakage from collaborative deep
learning.
In: Proceedings of the 2017 ACM SIGSAC Conference on Computer and
Communications Security.
CCS ’17,
pp. 603–618.
ACM,
New York, NY, USA
(2017).
doi: 10.1145/3133956.3134012 .
 http://doi.acm.org/10.1145/3133956.3134012 

 

 
 [39] 
 
Shokri, R.,
Stronati, M.,
Song, C.,
Shmatikov, V.:
Membership inference attacks against machine learning models.
In: 2017 IEEE Symposium on Security and Privacy, SP 2017, San
Jose, CA, USA, May 22-26, 2017,
pp. 3–18
(2017)

 

 
 [40] 
 
Song, C.,
Shmatikov, V.:
Overlearning reveals sensitive attributes.
In: International Conference on Learning Representations
(2020).
 https://openreview.net/forum?id=SJeNz04tDS 

 

 
 [41] 
 
Paillier, P.:
Public-key cryptosystems based on composite degree residuosity
classes.
In: The Proceedings of EUROCRYPT,
pp. 223–238
(1999)

 

 
 [42] 
 
Chillotti, I.,
Gama, N.,
Georgieva, M.,
Izabachène, M.:
TFHE: fast fully homomorphic encryption over the torus.
J. Cryptology
 33 (1),
34–91
(2020).
doi: 10.1007/s00145-019-09319-x 

 

 
 [43] 
 
Cheon, J.H.,
Kim, A.,
Kim, M.,
Song, Y.:
Homomorphic encryption for arithmetic of approximate numbers.
In: Takagi, T.,
Peyrin, T. (eds.)
Advances in Cryptology – ASIACRYPT 2017,
pp. 409–437.
Springer,
Cham
(2017)

 

 
 [44] 
 
Brakerski, Z.,
Gentry, C.,
Vaikuntanathan, V.:
(leveled) fully homomorphic encryption without bootstrapping.
In: Innovations in Theoretical Computer Science Conference (ITSC),
pp. 309–325
(2012)

 

 
 [45] 
 
Halevi, S.,
Shoup, V.:
Algorithms in HELib.
In: International Cryptology Conference,
pp. 554–571
(2014).
Springer

 

 
 [46] 
 
Kolesnikov, V.,
Schneider, T.:
Improved garbled circuit: Free xor gates and applications.
In: Proceedings of the 35th International Colloquium on Automata,
Languages and Programming, Part II,
pp. 486–498.
Springer,
Berlin, Heidelberg
(2008)

 

 
 [47] 
 
Zahur, S.,
Rosulek, M.,
Evans, D.:
Two Halves Make a Whole,
pp. 220–250.
Springer,
Berlin, Heidelberg
(2015)

 

 
 [48] 
 
Asharov, G.,
Lindell, Y.,
Schneider, T.,
Zohner, M.:
More efficient oblivious transfer and extensions for faster secure
computation.
In: 2013 ACM SIGSAC Conference on Computer and Communications
Security, CCS’13, Berlin, Germany, November 4-8, 2013,
pp. 535–548
(2013).
doi: 10.1145/2508859.2516738 .
 http://doi.acm.org/10.1145/2508859.2516738 

 

 
 [49] 
 
Huang, Y.,
Evans, D.,
Katz, J.,
Malka, L.:
Faster secure two-party computation using garbled circuits.
In: USENIX Conference on Security,
pp. 35–35
(2011)

 

 
 [50] 
 
Liu, C.,
Wang, X.S.,
Nayak, K.,
Huang, Y.,
Shi, E.:
Oblivm: A programming framework for secure computation.
In: 2015 IEEE Symposium on Security and Privacy,
pp. 359–376
(2015).
doi: 10.1109/SP.2015.29 

 

 
 [51] 
 
Wu, X.,
Kumar, V.,
Ross Quinlan, J.,
Ghosh, J.,
Yang, Q.,
Motoda, H.,
McLachlan, G.J.,
Ng, A.,
Liu, B.,
Yu, P.S.,
Zhou, Z.-H.,
Steinbach, M.,
Hand, D.J.,
Steinberg, D.:
Top 10 algorithms in data mining.
Knowledge and Information Systems
 14 (1),
1–37
(2007)

 

 
 [52] 
 
Costan, V.,
Devadas, S.:
Intel sgx explained.
IACR Cryptology ePrint Archive
 2016 ,
86
(2016)

 

 
 [53] 
 
Schapire, R.E.:
A brief introduction to boosting.
In: Proceedings of the 16th International Joint Conference on
Artificial Intelligence - Volume 2.
IJCAI’99,
pp. 1401–1406.
Morgan Kaufmann Publishers Inc.,
San Francisco, CA, USA
(1999)

 

 
 [54] 
 
LeCun, Y.,
Bengio, Y.,
Hinton, G.:
Deep learning.
Nature
 521 ,
436–444
(2015)

 

 
 [55] 
 
Phong, L.T.,
Aono, Y.,
Hayashi, T.,
Wang, L.,
Moriai, S.:
Privacy-preserving deep learning via additively homomorphic
encryption.
IEEE Transactions on Information Forensics and Security
 13 (5),
1333–1345
(2018).
doi: 10.1109/TIFS.2017.2787987 

 

 
 [56] 
 
Rouhani, B.D.,
Riazi, M.S.,
Koushanfar, F.:
Deepsecure: Scalable provably-secure deep learning.
In: Proceedings of the 55th Annual Design Automation Conference.
DAC ’18.
Association for Computing Machinery,
New York, NY, USA
(2018).
doi: 10.1145/3195970.3196023 .
 https://doi.org/10.1145/3195970.3196023 

 

 
 [57] 
 
Chakarov, D.,
Papazov, Y.:
Evaluation of the complexity of fully homomorphic encryption schemes
in implementations of programs.
In: Proceedings of the 20th International Conference on Computer
Systems and Technologies.
CompSysTech ’19,
pp. 62–67.
Association for Computing Machinery,
New York, NY, USA
(2019).
doi: 10.1145/3345252.3345292 .
 https://doi.org/10.1145/3345252.3345292 

 

 
 [58] 
 
Veugen, T.:
Encrypted integer division and secure comparison.
International Journal of Applied Cryptography
 3 (2),
166
(2014)

 

 
 [59] 
 
Golle, P.:
A private stable matching algorithm.
In: International Conference on Financial Cryptography and Data
Security,
pp. 65–80
(2006)

 

 
 [60] 
 
Goldreich, O.,
Micali, S.,
Wigderson, A.:
How to play any mental game.
In: Proceedings of the Nineteenth Annual ACM Symposium on Theory of
Computing.
STOC ’87,
pp. 218–229.
ACM,
New York, NY, USA
(1987).
doi: 10.1145/28395.28420 .
 http://doi.acm.org/10.1145/28395.28420 

 

 
 [61] 
 
Bunn, P.,
Ostrovsky, R.:
Secure two-party k-means clustering.
In: Proceedings of the 14th ACM Conference on Computer and
Communications Security.
CCS ’07,
pp. 486–497.
ACM,
New York, NY, USA
(2007).
doi: 10.1145/1315245.1315306 .
 http://doi.acm.org/10.1145/1315245.1315306 

 

 
 [62] 
 
Rane, S.,
Sun, W.:
Privacy preserving string comparisons based on levenshtein distance.
In: 2010 IEEE International Workshop on Information Forensics and
Security,
pp. 1–6
(2010).
doi: 10.1109/WIFS.2010.5711449 

 

 
 [63] 
 
Lazzeretti, R.,
Barni, M.:
Division between encrypted integers by means of garbled circuits.
In: 2011 IEEE International Workshop on Information Forensics and
Security,
pp. 1–6
(2011).
doi: 10.1109/WIFS.2011.6123132 

 

 
 [64] 
 
Dahl, M.,
Ning, C.,
Toft, T.:
On secure two-party integer division.
In: Keromytis, A.D. (ed.)
Financial Cryptography and Data Security,
pp. 164–178.
Springer,
Berlin, Heidelberg
(2012)

 

 
 [65] 
 
Jiang, X.,
Kim, M.,
Lauter, K.,
Song, Y.:
Secure outsourced matrix computation and application to neural
networks.
In: Proceedings of the 2018 ACM SIGSAC Conference on Computer and
Communications Security.
CCS ’18,
pp. 1209–1222.
ACM,
New York, NY, USA
(2018).
doi: 10.1145/3243734.3243837 .
 http://doi.acm.org/10.1145/3243734.3243837 

 

 
 [66] 
 
Halevi, S.,
Shoup, V.:
Design and implementation of a homomorphic-encryption library
(2013)

 

 
 [67] 
 
Riazi, M.S.,
Weinert, C.,
Tkachenko, O.,
Songhori, E.M.,
Schneider, T.,
Koushanfar, F.:
Chameleon: A hybrid secure computation framework for machine learning
applications.
In: Proceedings of the 2018 on Asia Conference on Computer and
Communications Security.
ASIACCS ’18,
pp. 707–721.
Association for Computing Machinery,
New York, NY, USA
(2018).
doi: 10.1145/3196494.3196522 .
 https://doi.org/10.1145/3196494.3196522 

 

 
 [68] 
 
Mohassel, P.,
Rindal, P.:
ABY3: A Mixed Protocol Framework for Machine Learning.
Cryptology ePrint Archive, Report 2018/403.
 https://eprint.iacr.org/2018/403 
(2018)

 

 
 [69] 
 
Patra, A.,
Suresh, A.:
BLAZE: Blazing Fast Privacy-Preserving Machine Learning.
Cryptology ePrint Archive, Report 2020/042.
 https://eprint.iacr.org/2020/042 
(2020)

 

 
 [70] 
 
Dreier, J.,
Kerschbaum, F.:
Practical privacy-preserving multiparty linear programming based on
problem transformation.
In: Proceedings of the Third IEEE Conference on Social Computing,
pp. 916–924
(2011)

 

 
 [71] 
 
Henecka, W.,
K ögl, S.,
Sadeghi, A.-R.,
Schneider, T.,
Wehrenberg, I.:
Tasty: Tool for automating secure two-party computations.
In: Proceedings of the 17th ACM Conference on Computer and
Communications Security.
CCS ’10,
pp. 451–462.
ACM,
New York, NY, USA
(2010).
doi: 10.1145/1866307.1866358 .
 http://doi.acm.org/10.1145/1866307.1866358 

 

 
 [72] 
 
Naehrig, M.,
Lauter, K.,
Vaikuntanathan, V.:
Can homomorphic encryption be practical?
In: Proceedings of Cloud Computing Security Workshop,
pp. 113–124.
ACM,
New York, NY, USA
(2011)

 

 
 [73] 
 
Canetti, R.:
Universally composable security.
J. ACM
 67 (5)
(2020)

 

 
 [74] 
 
Lindell, Y.:
In: Lindell, Y. (ed.)
How to Simulate It – A Tutorial on the Simulation Proof Technique,
pp. 277–346.
Springer,
Cham
(2017)

 

 
 [75] 
 
Tramèr, F.,
Zhang, F.,
Juels, A.,
Reiter, M.K.,
Ristenpart, T.:
Stealing machine learning models via prediction apis.
In: Proceedings of the 25th USENIX Conference on Security Symposium.
SEC’16,
pp. 601–618.
USENIX Association,
USA
(2016)

 

 
 [76] 
 
Dean, J.,
Ghemawat, S.:
Mapreduce: Simplified data processing on large clusters.
In: OSDI,
pp. 137–150
(2004)

 

 
 [77] 
 
Erlingsson, U.,
Pihur, V.,
Korolova, A.:
Rappor: Randomized aggregatable privacy-preserving ordinal response.
In: Proceedings of the 2014 ACM SIGSAC Conference on Computer and
Communications Security.
CCS ’14,
pp. 1054–1067.
ACM,
New York, NY, USA
(2014).
doi: 10.1145/2660267.2660348 .
 http://doi.acm.org/10.1145/2660267.2660348 

 

 
 [78] 
 
Du, W.,
Zhan, Z.:
Using randomized response techniques for privacy-preserving data
mining.
In: Proceedings of the Ninth ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining,
pp. 505–510.
ACM, ???
(2003)

 

 
 [79] 
 
Agrawal, R.,
Srikant, R.:
Privacy-preserving data mining.
In: Proceedings of ACM SIGMOD Conference,
pp. 439–450.
ACM,
Dallas, Texas
(2000)

 

 
 [80] 
 
Chen, K.,
Liu, L.:
Geometric data perturbation for outsourced data mining.
Knowledge and Information Systems
 29 (3)
(2011)

 

 
 [81] 
 
Liu, K.,
Kargupta, H.,
Ryan, J.:
Random projection-based multiplicative data perturbation for privacy
preserving distributed data mining.
IEEE Transactions on Knowledge and Data Engineering (TKDE)
 18 (1),
92–106
(2006)

 

 
 [82] 
 
Xu, H.,
Guo, S.,
Chen, K.:
Building confidential and efficient query services in the cloud with rasp data
perturbation.
IEEE Transactions on Knowledge and Data Engineering
 26 (2)
(2014)

 

 
 [83] 
 
Chen, G.,
Chen, S.,
Xiao, Y.,
Zhang, Y.,
Lin, Z.,
Lai, T.H.:
Sgxpectre attacks: Leaking enclave secrets via speculative execution.
CoRR
 abs/1802.09085 
(2018)

 

 
 [84] 
 
Sharma, S.,
Chen, K.:
Image disguising for privacy-preserving outsourced deep learning.
In: Poster Session of ACM CCS
(2018)

 

 
 [85] 
 
Boldyreva, A.,
Chenette, N.,
O’Neill, A.:
Order-preserving encryption revisited:improved security analysisand
alternative solutions.
In: CRYPTO
(2011)

 

 
 [86] 
 
Boldyreva, A.,
Chenette, N.,
Lee, Y.,
O’Neill, A.:
Order preserving symmetric encryption.
In: Proceedings of EUROCRYPT Conference
(2009)

 

 
 [87] 
 
Kerschbaum, F.:
Frequency-hiding order-preserving encryption.
In: Proceedings of ACM Conference on Computer and Communication
Security
(2015)

 

 
 [88] 
 
Golle, P.,
Staddon, J.,
Waters, B.:
Secure conjunctive keyword search over encrypted data.
In: ACNS 04: 2nd International Conference on Applied Cryptography and
Network Security,
pp. 31–45.
Springer, ???
(2004)

 

 
 [89] 
 
Curtmola, R.,
Garay, J.,
Kamara, S.,
Ostrovsky, R.:
Searchable symmetric encryption: improved definitions and efficient
constructions.
In: ACM CCS,
pp. 79–88
(2006)

 

 
 [90] 
 
Tsai, C.,
Porter, D.E.,
Vij, M.:
Graphene-sgx: A practical library OS for unmodified applications
on SGX.
In: Silva, D.D.,
Ford, B. (eds.)
2017 USENIX Annual Technical Conference, USENIX ATC 2017, Santa
Clara, CA, USA, July 12-14, 2017,
pp. 645–658
(2017)

 

 
 [91] 
 
Arnautov, S.,
Trach, B.,
Gregor, F.,
Knauth, T.,
Martin, A.,
Priebe, C.,
Lind, J.,
Muthukumaran, D.,
O’Keeffe, D.,
Stillwell, M.L.,
Goltzsche, D.,
Eyers, D.,
Kapitza, R.,
Pietzuch, P.,
Fetzer, C.:
Scone: Secure linux containers with intel sgx.
In: Proceedings of the 12th USENIX Conference on Operating Systems
Design and Implementation.
OSDI’16,
pp. 689–703.
USENIX Association,
Berkeley, CA, USA
(2016)

 

 
 [92] 
 
Shinde, S.,
Tien, D.L.,
Tople, S.,
Saxena, P.:
Panoply: Low-tcb linux applications with sgx enclaves.
In: Proceedings of NDSS
(2017)

 

 
 [93] 
 
Lee, D.,
Kuvaiskii, D.,
Vahldiek-Oberwagner, A.,
Vij, M.:
Privacy-preserving machine learning in untrusted clouds made simple.
CoRR
 abs/2009.04390 
(2020).
 2009.04390 

 

 
 [94] 
 
Sasy, S.,
Gorbunov, S.,
Fletcher, C.W.:
Zerotrace : Oblivious memory primitives from intel SGX.
In: 25th Annual Network and Distributed System Security Symposium,
NDSS 2018, San Diego, California, USA, February 18-21, 2018
(2018)

 

 
 [95] 
 
Ahmad, A.,
Kim, K.,
Sarfaraz, M.I.,
Lee, B.:
Obliviate: A data oblivious file system for intel sgx.
In: the Network and Distributed System Security Symposium
(2018)

 

 
 [96] 
 
Shinde, S.,
Chua, Z.L.,
Narayanan, V.,
Saxena, P.:
Preventing page faults from telling your secrets.
In: Proceedings of the 11th ACM on Asia Conference on Computer and
Communications Security.
ASIACCS16,
pp. 317–328.
Association for Computing Machinery,
New York, NY, USA
(2016).
doi: 10.1145/2897845.2897885 .
 https://doi.org/10.1145/2897845.2897885 

 

 
 [97] 
 
Goldreich, O.,
Ostrovsky, R.:
Software protection and simulation on oblivious ram.
Journal of the ACM
 43 ,
431–473
(1996)

 

 
 [98] 
 
Ohrimenko, O.,
Schuster, F.,
Fournet, C.,
Mehta, A.,
Nowozin, S.,
Vaswani, K.,
Costa, M.:
Oblivious multi-party machine learning on trusted processors.
In: Holz, T.,
Savage, S. (eds.)
25th USENIX Security Symposium, USENIX Security 16, Austin, TX,
USA, August 10-12, 2016,
pp. 619–636.
USENIX Association, ???
(2016).
 https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/ohrimenko 

 

 
 [99] 
 
Alam, A.K.M.M.,
Sharma, S.,
Chen, K.:
Sgx-mr: Regulating dataflows for protecting access patterns of
data-intensive sgx applications.
Proceedings on Privacy Enhancing Technologies
 2021 (1),
5–20
(01 Jan. 2021).
doi: 10.2478/popets-2021-0002 

 

 
 [100] 
 
Bulck, J.V.,
Minkin, M.,
Weisse, O.,
Genkin, D.,
Kasikci, B.,
Piessens, F.,
Silberstein, M.,
Wenisch, T.F.,
Yarom, Y.,
Strackx, R.:
Foreshadow: Extracting the keys to the intel SGX kingdom with
transient out-of-order execution.
In: 27th USENIX Security Symposium (USENIX Security 18),
pp. 991–1008.
USENIX Association,
Baltimore, MD
(2018).
 https://www.usenix.org/conference/usenixsecurity18/presentation/bulck 

 

 
 [101] 
 
Ristenpart, T.,
Tromer, E.,
Shacham, H.,
Savage, S.:
Hey, you, get off of my cloud: exploring information leakage in
third-party compute clouds.
In: Proceedings of the 16th ACM Conference on Computer and
Communications Security,
New York, NY, USA,
pp. 199–212
(2009)

 

 
 [102] 
 
Kocher, P.,
Horn, J.,
Fogh, A.,
Genkin, D.,
Gruss, D.,
Haas, W.,
Hamburg, M.,
Lipp, M.,
Mangard, S.,
Prescher, T.,
Schwarz, M.,
Yarom, Y.:
Spectre attacks: Exploiting speculative execution.
In: 2019 IEEE Symposium on Security and Privacy (SP),
pp. 1–19
(2019).
doi: 10.1109/SP.2019.00002 

 

 
 [103] 
 
Lipp, M.,
Schwarz, M.,
Gruss, D.,
Prescher, T.,
Haas, W.,
Fogh, A.,
Horn, J.,
Mangard, S.,
Kocher, P.,
Genkin, D.,
Yarom, Y.,
Hamburg, M.:
Meltdown: Reading kernel memory from user space.
In: 27th USENIX Security Symposium (USENIX Security 18)
(2018)

 

 
 
 
 
 

## Figures

 
 Figure 1: A data owner outsourcing to an untrusted cloud provider for learning a model. The data contributors directly submit their encrypted data to the cloud. The cloud carries out the major expensive computations over the encrypted data and data owner can assist with some lightweight work. 
 
 
 Figure 2: A data owner outsources data storage and machine learning tasks to the Cloud. The Cryptographic Service Provider (CSP) manages the keys, decrypts intermediate results, and assists the Cloud with other relatively lightweight computations. 
 
 
 Figure 3: The systematization framework for confidential machine learning (CML) approaches. 
 
 
 Figure 4: The decomposition-mapping-composition (DMC) process for constructing hybrid CML solutions. 
 
 
 
 

## Tables

 
 Table 1: Real cost comparison for confidential arithmetic and linear algebra operations at 112-bit security, v 100 × 1 v_{100\times 1} and M 100 × 100 M_{100\times 100} . 
 
 
 
 | 
 AHE (Paillier) | 
 SHE (RLWE) | 
 Garbled Circuits | 
 Secret Sharing | 

 
 | 
 Comp | 
 Comp | 
 Comp | 
 Comm | 
 Comp | 
 Comm | 

 
 Addition/Subtraction | 
 0.01 ms | 
 0.2 ms | 
 37 ms | 
 2 KB | 
 0.0 ms | 
 0.0 KB | 

 
 Multiplication | 
 0.05 ms | 
 39 ms | 
 138 ms | 
 40 KB | 
 1 s | 
 2 KB | 

 
 Comparison | 
 429 h | 
 10 5 ​ h 10^{5}h | 
 37 ms | 
 2 KB | 
 - | 
 - | 

 
 Division | 
 - | 
 - | 
 208 ms | 
 46 KB | 
 - | 
 - | 

 
 Vector Addition | 
 0.6 ms | 
 0.2 ms | 
 36 ms | 
 192 KB | 
 0.0 ms | 
 0.0 KB | 

 
 Dot Product | 
 6 ms | 
 39 ms | 
 5 s | 
 4 MB | 
 7 s | 
 195 KB | 

 
 Matrix-vector Multiplication | 
 1 s | 
 3 m | 
 8 m | 
 396 MB | 
 7 s | 
 290 KB | 

 

 
 
 Table 2: Examples for primitive switching strategies in hybrid composition of CML frameworks. 
 
 
 
 Framework | 
 Primitive Switch | 
 
 
 Operation Switch 
 | 
 
 
 Justification 
 | 

 
 | 
 | 

 
 Sharma and Chen [ 13 ] | 
 SHE → \rightarrow GC | 
 
 
 Matrix vector multiplication → \rightarrow Sign Check 
 | 
 
 
 Sign checking is impractically expensive with SHE whereas tolerable with GC. 
 | 

 
 Nikolaenko et al. [ 9 ] | 
 AHE → \rightarrow GC | 
 
 
 Matrix Additions → \rightarrow Cholesky’s decomposition 
 | 
 
 
 The operations of division and square root in Cholesky’s decomposition were not feasible with the AHE scheme. 
 | 

 
 Nikolaenko et al. [ 8 ] | 
 AHE → \rightarrow GC | 
 
 
 Matrix Additions → \rightarrow Gradient Descent 
 | 
 
 
 Gradient descent involved multiplications, additions, and subtractions not entirely feasible with the AHE scheme. 
 | 

 
 Mohassel et al. [ 11 ] | 
 SecSh → \rightarrow GC | 
 
 
 Matrix-vector multiplication → \rightarrow Comparison 
 | 
 
 
 Comparison is impossible over randomly shared secrets leading the switch to the garbled circuits. 
 | 

 
 Mohassel et al. [ 11 ] | 
 GC → \rightarrow SecSh | 
 
 
 Comparison → \rightarrow Vector Subtraction 
 | 
 
 
 Use of garbled circuits for comparison was unavoidable however continuing GC on to vector subtraction would result in excessive cost overhead. 
 | 

 
 Demmler et al. [ 10 ] | 
 SecSh → \rightarrow AHE/OT | 
 
 
 Data at rest → \rightarrow Multiplication 
 | 
 
 
 Multiplication with random shares required switching to either AHE or OT protocol involving the two parties in the frameworks. 
 | 

 
 Riazi et al. [ 67 ] | 
 SecSh → \rightarrow GC | 
 
 
 Matrix matrix multiplication → \rightarrow ReLu computation 
 | 
 
 
 Sign checking is impossible over randomly shared secrets leading the switch to garbled circuits 
 | 

 
 Riazi et al. [ 67 ] | 
 GC → \rightarrow SecSh | 
 
 
 ReLu → \rightarrow Matrix vector multiplication 
 | 
 
 
 Use of garbled circuits for matrix vector multiplication is impractical. 
 | 

 

 
 
 Table 3: Example CML methods that replace the expensive algorithmic components with their crypto-friendly versions. 
 
 
 
 Framework | 
 
 
 ML Algorithm 
 | 
 
 
 Original Component 
 | 
 
 
 Crypto-friendly Component 
 | 
 
 
 Benefits 
 | 

 
 Mohassel et al. [ 11 ] | 
 
 
 Logistic Regression, Neural Networks 
 | 
 
 
 Sigmoid, Softmax 
 | 
 
 
 ReLu 
 | 
 
 
 Avoids inversion and limits expensive confidential divisions to one. 
 | 

 
 Graepel et al. [ 33 ] | 
 
 
 LMC, Fisher’s LDA 
 | 
 
 
 Divisions 
 | 
 
 
 Multiplications with incorporated division factors 
 | 
 
 
 Avoids division costs and simplifies the protocol. 
 | 

 
 Nikolaenko et al. [ 9 ] | 
 
 
 Ridge Linear Regression 
 | 
 
 
 LU decomposition 
 | 
 
 
 Cholesky’s decomposition 
 | 
 
 
 Reduces the cost complexity by half. 
 | 

 
 Nikolaenko et al. [ 8 ] | 
 
 
 Matrix Factorization 
 | 
 
 
 Cholesky’s Decomposition 
 | 
 
 
 Sorting based matrix factorization 
 | 
 
 
 Reduces the overall complexity from quadratic to within a polylogarithmic factor of the complexity in the plaintext 
 | 

 
 Sharma and Chen [ 13 ] | 
 
 
 Boosting 
 | 
 
 
 Decision Stumps 
 | 
 
 
 Random Linear Classifiers 
 | 
 
 
 Reduced number of comparisons and simplicity in learning. 
 | 

 
 Naehrig et al. [ 72 ] | 
 
 
 Logistic Regression 
 | 
 
 
 Exponentiation 
 | 
 
 
 Taylor Expansion 
 | 
 
 
 Avoids costs involved in multiple levels of multiplications. 
 | 

 
 Sharma et al. [ 12 ] | 
 
 
 Spectral Clustering 
 | 
 
 
 Eigen decomposition 
 | 
 
 
 Eigen-approximation by Lanczos and Nystrom 
 | 
 
 
 Reduces complexity of the problem from O ⁡ ( N 3 ) O(N^{3}) to O ⁡ ( N 2 ) O(N^{2}) . 
 |