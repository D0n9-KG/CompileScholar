A Survey on Federated Recommendation Systems 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2301.00767v2 [cs.IR] 09 Mar 2023 
 
 

# A Survey on Federated Recommendation Systems

 
 
 Zehua Sun 1 ,
Yonghui Xu 1 ,
Yong Liu,
Wei He,
Lanju Kong,
Fangzhao Wu,
Yali Jiang 2 ,
Lizhen Cui
 † † thanks: Zehua Sun, Yonghui Xu, Wei He, Lanju Kong, Yali Jiang and Lizhen Cui are with Joint SDU-NTU Centre for Artificial Intelligence Research (C-FAIR) Software School, Shandong University. Yonghui Xu are also with Sino-Singapore International Joint Research Institute. † † thanks: Yong Liu are with Alibaba-NTU Singapore Joint Research Institute, Nanyang Technological University, Singapore. † † thanks: Fangzhao Wu are with Microsoft Research Asia, China. † † thanks: 1 Zehua Sun and Yonghui Xu are Co-First authors. † † thanks: 2 Corresponding author: jiang.yl@sdu.edu.cn. 
 Affiliation:  
 

 Abstract 
 
 Federated learning has recently been applied to recommendation systems to protect user privacy. In federated learning settings, recommendation systems can train recommendation models by collecting the intermediate parameters instead of the real user data, which greatly enhances user privacy. Besides, federated recommendation systems can cooperate with other data platforms to improve recommendation performance while meeting the regulation and privacy constraints. However, federated recommendation systems face many new challenges such as privacy, security, heterogeneity and communication costs. While significant research has been conducted in these areas, gaps in the surveying literature still exist. In this survey, we—(1) summarize some common privacy mechanisms used in federated recommendation systems and discuss the advantages and limitations of each mechanism; (2) review several novel attacks and defenses against security; (3) summarize some approaches to address heterogeneity and communication costs problems; (4) introduce some realistic applications and public benchmark datasets for federated recommendation systems; (5) present some prospective research directions in the future. This survey can guide researchers and practitioners understand the research progress in these areas.

 
 
 
 Index Terms:  Recommendation Systems, Federated Learning, Privacy, Security, Heterogeneity, Communication Costs.

 
 

## I Introduction 

 
 In recent years, recommendation systems have been widely used to model user interests so as to solve information overload problems in many real-world fields, e.g., e-commerce [ 1 ] [ 2 ] , news [ 3 ] [ 4 ] and healthcare [ 5 ] [ 6 ] . To further improve the recommendation performance, such systems usually collect as much data as possible, including a lot of private information about users, such as user attributes, user behaviors, social relations, and context information.

 
 
 Although these recommendation systems have achieved remarkable results in accuracy, most of them require a central server to store collected user data, which exists potential privacy leakage risks because user data could be sold to a third party without user consent, or stolen by motivated attackers. In addition, due to privacy concerns and regulatory restrictions, it becomes more difficult to integrate data from other platforms to improve recommendation performance. For example, regulations such as General Data Protection Regulation (GDPR) [ 7 ] set strict rules on collecting user data and sharing data between different platforms, which may lead to insufficient data for recommendation systems and further affects recommendation performance.

 
 
 Federated learning is a privacy-preserving distributed learning scheme proposed by Google [ 8 ] , which enables participants to collaboratively train a machine learning model by sharing intermediate parameters (e.g., model parameters, gradients) instead of their real data. Therefore, combining federated learning with recommendation systems becomes a promising solution for privacy-preserving recommendation systems. In this paper, we term it federated recommendation system (FedRS).

 
 

### I-A Challenges 

 
 While FedRS avoids direct exposure of real user data and provides a privacy-aware paradigm for model training, there are still some core challenges that need to be addressed.

 
 
 Challenge 1: Privacy concerns for users. Privacy protection is often the major goal of FedRS. In FedRS, each participant jointly trains a global recommendation model by sharing intermediate parameters instead of their real user-item interaction data, which makes an important step towards privacy-preserving recommendation systems. However, a curious server can still infer user ratings and user interaction behaviors from the uploaded intermediate parameters [ 9 ] [ 10 ] . Besides, FedRS also faces the risk of privacy leakage when integrating auxiliary information (e.g., social features) to improve recommendation performance.

 
 
 Challenge 2: Security attacks on FedRS. In federated recommendation scenarios, participants may be malicious, and they can poison their local training samples or uploaded intermediate parameters to attack the security of FedRS. They can increase the exposure of specific products for profit [ 11 ] , or destroy the overall recommendation performance of competing companies [ 12 ] . To ensure the fairness and performance of recommendations, FedRS must have the ability to detect and defend against poison attacks from participants.

 
 
 Challenge 3: Heterogeneity in FedRS. FedRS also faces the problem of system heterogeneity, statistical heterogeneity and privacy heterogeneity during the collaborative training by multiple clients. When training recommendation model locally, due to the difference in storage, computing and communication capabilities, the clients with limited capabilities may become stragglers and further affect the training efficiency. Besides, data (e.g., user attributes, ratings and interaction behavior) in different clients is usually not independent and identically distributed (Non-IID), and training a consistent global recommendation model for all users can’t achieve the personalization of recommendation results. Moreover, in realistic applications, users often have different privacy needs and adopt different privacy settings. So simply using the same privacy budgets for users will bring unnecessary loss of recommendation accuracy and efficiency.

 
 
 Challenge 4: Communication costs during FedRS model training and inference. 
To achieve satisfactory recommendation performance, clients need to communicate with the central server for multiple rounds. However, real-world recommendation systems are usually built on complex deep learning models and millions of intermediate parameters need to be communicated [ 13 ] .
In addition, clients must receive a large amount of item data from the server to generate recommendation results locally. Therefore, clients may be hard to afford severe communication costs, which greatly limits the application of FedRS in large-scale recommendation scenarios.

 
 
 

### I-B Related Surveys 

 
 There are many surveys that have focused on recommendation systems or federated learning. For examples, Adomavicius e ​ t ​ a ​ l . et\ al. [ 14 ] provide a detailed categorization of recommendation methods and introduce various limitations of each method. Yang e ​ t ​ a ​ l . et\ al. [ 15 ] give the definition of federated learning and discuss its architectures and applications. And Li e ​ t ​ a ​ l . et\ al. [ 16 ] summarize the unique characteristics and challenges of federated learning. Besides, there are also some surveys on the privacy and security of federated learning. For examples, Viraaji e ​ t ​ a ​ l . et\ al. [ 17 ] identify and evaluate the privacy threats and security vulnerabilities in federated learning. And Lyu e ​ t ​ a ​ l . et\ al. [ 18 ] comprehensively explore the assumptions, reasons, principles and differences of the current attacks and defenses in the privacy and robustness fields of federated learning. However, the existing surveys usually treat recommendation systems and federated learning separately, and few work surveyed specific problems in FedRS [ 19 ] . Yang e ​ t ​ a ​ l . et\ al. [ 19 ] categorize FedRS from the aspect of the federated learning and discuss the algorithm-level and system-level challenges for FedRS. However, they do not provide comprehensive methods to address privacy, security, heterogeneity, and communication costs challenges.

 
 
 Fig. 1: Communication architecture of FedRS. 
 
 
 

### I-C Our Contribution 

 
 Compared with the previous surveys, this paper makes the following contributions: Firstly, we provide a comprehensive overview of FedRS from the perspectives of definition, communication architectures and categorization. Secondly, we summarize the state-of-the-art studies of FedRS in terms of privacy, security, heterogeneity and communication costs areas. Thirdly, we introduce some applications and public benchmark datasets for FedRS. Fourthly, we discuss the promising future directions for FedRS.

 
 
 The rest of the paper is organized as follows: Section  II discusses the overview of FedRS. Section  III -Section  VI summarize the state-of-the-art studies of FedRS from the aspects of privacy, security, heterogeneity and communication costs. Section  VII introduces the applications and public benchmark datasets for FedRS. Section  VIII presents some prospective research directions. Finally, Section  IX concludes this survey.

 
 
 
 

## II Overview of Federated Recommendation Systems 

 

### II-A Definition 

 
 FedRS is a technology that provides recommendation services in a privacy-preserving way. To protect user privacy, the participants in FedRS collaboratively train the recommendation model by exchanging intermediate parameters instead of sharing their own real data. In the ideal case, the performance of recommendation model trained in FedRS should be close to the performance of the recommendation model trained in the data- centralized setting, which can be formalized as:

 

 
 | 
 | V F ​ E ​ D − V S ​ U ​ M | δ . |V_{FED}-V_{SUM}| \delta. | 
 | 
 (1) | 
 

 where V F ​ E ​ D V_{FED} is the recommendation model performance in FedRS , V S ​ U ​ M V_{SUM} is the recommendation model performance in traditional recommendation systems for centralized data storage, and δ \delta is a small positive number.

 
 
 

### II-B Communication Architecture 

 
 In FedRS, the data of participants is stored locally, and the intermediate parameters are communicated between the server and participants. There are two major communication architectures used in the study of FedRS, including client-server architecture and peer-peer architecture.

 
 
 Client-Server Architecture . Client-server architecture is the most common communication architecture used in FedRS, as shown in Fig. 1 (a), which relies on a trusted central server to perform initialization and model aggregation tasks. In each round, the server distributes the current global recommendation model to some selected clients. Then the selected clients use the received model and their own data for local training, and send the updated intermediate parameters (e.g., model parameters, gradients) to the server for global aggregation. The client-server architecture requires a central server to aggregate the intermediate parameters uploaded by the clients. Thus, once the server has a single point of failure, the entire training process will be seriously affected [ 20 ] . In addition, the curious server may infer the clients’ privacy information through the intermediate parameters, leaving potential privacy concerns [ 9 ] .

 
 
 Peer-Peer Architecture . Considering the single point of failure problem for client-server architecture in FedRS, Hegeds e ​ t ​ a ​ l . et\ al. [ 21 ] design a peer-peer communication architecture with no central server involved in the communication process, which is shown in Fig. 1 (b). During each communication round, each participant broadcasts the updated intermediate parameters to some random online neighbors in the peer to peer network, and aggregates received parameters into its own global model. In this architecture, the single point of failure and privacy issues associated with a central server can be avoided. However, the aggregation process occurs on each client, which greatly increases the communication and computation overhead for clients [ 22 ] .

 
 
 

### II-C Categorization 

 
 In FedRS, the participants are responsible for the local training process as the data owners. They can be different mobile devices or data platforms. Considering the unique properties of different participant types, FedRS usually have different application scenarios and designs. Besides, there are also some differences between different recommendation models in the federation process. Thus, we summarize the current FedRS and categorize them from the perspectives of participant type and recommendation model. Fig. 2 shows the summary of the categorization of FedRS.

 
 
 Fig. 2: Categorization of federated recommendation systems. 
 
 

#### II-C 1 Participant Type

 
 Based on the type of participants, FedRS can be categorized into cross-device FedRS and cross-platform FedRS.

 
 
 Cross-device FedRS . In cross-device FedRS, different mobile devices are usually treated as participants [ 23 ] [ 10 ] . The typical application of cross-device FedRS is to build a personal recommendation model for users without collecting their local data. In this way, users can enjoy recommend services while protecting their private information. The number of participants in cross-device FedRS is relatively large and each participant keeps a small amount of data. Considering the limited computation and communication abilities of mobile devices, cross-device FedRS cannot handle very complex training tasks. Besides, due to the power and the network status, mobile devices may drop out of the training process. Thus, the major challenges for cross-device FedRS are how to improve the efficiency and deal with the straggler problem of devices during the training process.

 
 
 Cross-platform FedRS . In cross-platform FedRS, different data platforms are usually treated as participants who want to collaborate to improve recommendation performance while meeting regulation and privacy constraints [ 24 ] [ 25 ] [ 26 ] . For example, to improve the recommendation performance, recommendation systems often integrate data from multiple platforms (e.g., e-commercial platforms, social platforms) [ 27 ] . However, due to privacy and regulation concerns, the different data platforms are often unable to directly share their data with each other. In this scenario, cross-platform FedRS can be used to collaboratively train recommendation models between different data platforms without directly exchanging their users’ data. Compared to cross-device FedRS, the number of participants in cross-platform FedRS is relatively small, and each participant owns relative large amount of data. An important challenge for cross-platform FedRS is how to design a fair incentive mechanism to measure contributions and benefits of different data platforms. Besides, it is hard to find a trusted server to manage training process in cross-platform FedRS, so a peer to peer communication architecture can be a good choice in this case.

 
 
 

#### II-C 2 Recommendation Model

 
 According to the different recommendation models used in FedRS, FedRS can be categorized into matrix factorization based FedRS, deep learning based FedRS and meta learning based FedRS.

 
 
 Matrix factorization based FedRS . Matrix factorization [ 28 ] is the most common model used in FedRS, which formulates the user-item interaction or rating matrix R ∈ ℝ N × M R\in\mathbb{R}^{N\times M} as a linear combination of user profile matrix U ∈ ℝ N × K U\in\mathbb{R}^{N\times K} and item profile matrix V ∈ ℝ M × K V\in\mathbb{R}^{M\times K} :

 

 
 | 
 R = U ​ V T . R=UV^{T}. | 
 | 
 (2) | 
 

 then uses the learned model to recommend new items to the user according to the predicted value. In matrix factorization model based FedRS, the user factor vectors are stored and updated locally on the clients, and only the item factor vectors [ 29 ] or the gradients of item factor vectors [ 23 ] [ 10 ] [ 9 ] [ 30 ] are uploaded to the server for aggregation. Matrix factorization model based FedRS can simply and effectively capture user tastes with the interaction and rating information between users and items. However, it still has many limitations such as sparsity (the number of ratings to be predicted is much smaller than the known ratings) and cold-start (new users and new items lack ratings) problems [ 14 ] .

 
 
 Deep learning based FedRS . To learn more complex representations of users and items and improve recommendation performance, deep learning technology has been widely used in recommendation systems. However, as privacy regulations get stricter, it becomes more difficult for recommendation systems to collect enough user data to build a high performance deep learning model. To make full use of user data while meeting privacy regulations, many effective deep learning model based FedRS have been proposed [ 31 ] [ 32 ] [ 33 ] . Considering different model structures, deep learning model based FedRS usually adopt different model update and intermediate parameter transmit processes. For examples, Perifanis e ​ t ​ a ​ l . et\ al. [ 31 ] propose a federated neural collaborative filtering (FedNCF) framework based on NCF [ 34 ] . In FedNCF, the clients locally update the network weights as well as the user and item profiles, then upload the item profile and network weights after masking to the server for aggregation. Wu e ​ t ​ a ​ l . et\ al. [ 32 ] propose a federated graph neural network (FedGNN) framework based on GNN. In FedGNN, the clients locally train GNN models and update the user/item embeddings from their local sub-graph, then send the perturbed gradients of GNN model and item embedding to the central server for aggregation. Besides, Huang e ​ t ​ a ​ l . et\ al. [ 35 ] propose a federated multi-view recommendation framework based on Deep Structured Semantic Model (DSSM [ 36 ] ). In FL-MV-DSSM, each view i i locally trains the user and item sub-models based on their own user data and local shared item data, then send the perturbed gradients of both user and item sub-models to server for aggregation. Although deep learning model based FedRS achieve outstanding performance in terms of accuracy, the massive model parameters of deep learning models bring huge computation and communication overhead to the clients, which presents a serious challenge for real industrial recommendation scenarios.

 
 
 Meta learning based FedRS . The most of existing federated recommendation studies are built on the assumption that data distributed on each client is independent and identically (IID). However, learning a unified federated recommendation model often performs poorly when handling Non-IID and highly personalized data on clients. Meta learning model can quickly adapt to new tasks while maintaining good generalization ability [ 37 ] , which makes it particularly suitable for FedRS. In meta learning model based FedRS, the server aggregates the intermediate parameters uploaded by clients to learn a model parameter initialization, and the clients fine-tune the initialed model parameters in the local training phase to fit their local data [ 38 ] [ 39 ] . In this way, meta learning model based FedRS can adapt the clients’ local data to provide more personalized recommendations. Although the performance of meta learning model based FedRS are generally better than learning a unified global model, the private information leakage can still occur during the learning process of model parameter initialization [ 38 ] .

 
 
 
 
 

## III Privacy of Federated recommendation Systems 

 
 In the model training process of FedRS, the user data is stored locally and only the intermediate parameters are uploaded to a server, which can further protect user privacy while keeping recommendation performance. Nevertheless, several research works show that the central server can still infer some sensitive information based on intermediate parameters. For examples, a curious server can identify items the user has interacted with according to the non-zero gradients sent by the client [ 32 ] . Besides, the server can also infer the user ratings as long as obtaining the user uploaded gradients in two consecutive rounds [ 9 ] . To further protect the privacy of FedRS, many studies have incorporated other privacy protection mechanisms into the FedRS, including pseudo items, homomorphic encryption, secret sharing and differential privacy. This section introduces the application of each privacy mechanism used in FedRS, and compare their advantages and limitations.

 
 

### III-A Pseudo Items 

 
 To prevent the server from inferring the set of items that users have interacted with based on non-zero gradients, some studies utilize pseudo items to protect user interaction behaviors in FedRS. The key idea of pseudo items is that the clients not only upload gradients of items that have been interacted with but also upload gradients of some sampled items that have not been with.

 
 
 For example, Lin e ​ t ​ a ​ t . et\ at. [ 10 ] propose a federated recommendation framework for explicit feedback scenario named FedRec, in which they design an effective hybrid filling strategy to generate virtual ratings of unrated items by the following equation:

 

 
 | 
 r u ​ i ′ = { ∑ k = 1 m y u ​ k ​ r u ​ k ∑ k = 1 m y u ​ k , t T p ​ r ​ e ​ d ​ i ​ c ​ t r ^ u ​ i , t ≥ T p ​ r ​ e ​ d ​ i ​ c ​ t r_{ui}^{{}^{\prime}}=\left\{\begin{aligned} \frac{\sum_{k=1}^{m}y_{uk}r_{uk}}{\sum_{k=1}^{m}y_{uk}},\ t T_{predict}\\
\hat{r}_{ui},\ t\geq T_{predict}\end{aligned}\right. | 
 | 
 (3) | 
 

 where t t denotes the number of current training iteration, and T p ​ r ​ e ​ d ​ i ​ c ​ t T_{predict} denotes the iteration number when choosing the average value or predict value as virtual rating value to a sampled item i i . However, the hybrid filling strategy in FedRec introduces extra noise to the recommendation model, which inevitably affects the model performance. To tackle this problem, Feng e ​ t ​ a ​ t . et\ at. [ 40 ] design a lossless version of FedRec named FedRec++. FedRec++ divides clients into ordinary clients and denoising clients. The denoising clients collect noisy gradients from ordinary clients and send the summation of the noisy gradients to the server to eliminate the gradient noise.

 
 
 Although pseudo items can effectively protect user interaction behaviors in FedRS, it does not modify the gradients of rated items. The curious server can still infer user ratings on the gradients uploaded by users [ 9 ] .

 
 
 

### III-B Homomorphic Encryption 

 
 To further protect the user ratings in FedRS, many studies attempt to encrypt intermediate parameters before uploading them to the server. Homomorphic encryption mechanism allows mathematical operation on encrypted data [ 41 ] , so it is well suited for the intermediate parameters upload and aggregation processes in FedRS.

 
 
 For example, Chai e ​ t ​ a ​ t . et\ at. [ 9 ] propose a secure federated matrix factorization framework named FedMF, in which clients use Paillier homomorphic encryption mechanism [ 42 ] to encrypt the gradients of item embedding matrix before uploading them to the server, and the server aggregates gradients on the cipher-text. Due to the characteristics of homomorphic encryption, FedMF can achieve the same recommendation accuracy as traditional matrix factorization. However, FedMF causes serious computation overheads since all computation operations are performed on the ciphertext and most of system’s time is spent on server updates. Besides, FedMF assumes that all participants are honest and will not leak the secret key to the server, which is hard to guarantee in reality. Moreover, Zhang e ​ t ​ a ​ t . et\ at. [ 43 ] propose a federated recommendation method (CLFM-VFL) for vertical federated learning scenarios where participants have more overlapping users but fewer overlapping features of users. CLFM-VFL uses homomorphic encryption to protect the gradients of user hidden vectors for each participant and cluster the users to improve recommendation accuracy and reduce matrix dimension.

 
 
 Besides, many studies also utilize homomorphic encryption mechanism to integrate private information from other participants to improve recommendation accuracy [ 32 ] [ 44 ] . For examples, Wu e ​ t ​ a ​ l . et\ al. [ 32 ] use homomorphic encryption mechanism to find the anonymous neighbors of users to expand the local user-item graph. Perifanis e ​ t ​ a ​ l . et\ al. [ 44 ] use Cheon-Kim-Kim-Song (CKKS) fully homomorphic encryption mechanism [ 45 ] to incorporate learned parameters between user’s friends after the global model is generated.

 
 
 Homomorphic encryption mechanism based FedRS can effectively protect user ratings while maintaining recommendation accuracy. Besides, it can prevent privacy leaks when integrating information from other participants. However, homomorphic encryption brings huge computation costs during operation process. And it is also a serious challenge to keep the secret key not be obtained by the server or other malicious participants.

 
 
 TABLE I: Comparison between different privacy mechanism. 
 
 
 Privacy Mechanisms | 
 Ref | 
 Main Protect Object | 
 Accuracy Loss | 
 Communication/Computation Costs | 

 
 Pseudo Items | 
 [ 10 ] [ 40 ] [ 32 ] [ 46 ] [ 47 ] | 
 Interaction Behaviors | 
 ✓ | 
 Low Costs | 

 
 Homomorphic Encryption | 
 [ 9 ] | 
 Ratings | 
 ✗ | 
 High Computation Costs | 

 
 [ 32 ] | 
 High-order Graph | 

 
 [ 44 ] | 
 Social Features | 

 
 Secret Sharing | 
 [ 30 ] [ 47 ] | 
 Ratings | 
 ✗ | 
 High Communication Costs | 

 
 Local Differential Privacy | 
 [ 29 ] [ 32 ] [ 46 ] | 
 Ratings | 
 ✓ | 
 Low Costs | 

 
 
 

### III-C Secret Sharing 

 
 As another encryption mechanism used in FedRS, secret sharing mechanism breaks intermediate parameters up into multiple pieces, and distributes the pieces among participants, so that only when all pieces are collected can reconstruct the intermediate parameters.

 
 
 For example, Ying [ 30 ] proposes a secret sharing based federated matrix factorization framework named ShareMF. The participants divide the item matrix gradients g p ​ l ​ a ​ i ​ n g^{plain} into several random numbers that meet:

 

 
 | 
 g p ​ l ​ a ​ i ​ n = g s ​ u ​ b ​ 1 + g s ​ u ​ b ​ 2 + … + g s ​ u ​ b ​ t . g^{plain}=g^{sub1}+g^{sub2}+...+g^{subt}. | 
 | 
 (4) | 
 

 Each participant keeps one of the random numbers and sends the rest to t − 1 t-1 sampled participants, then uploads the sum of received and kept numbers as hybrid gradients to the server for aggregation. ShareMF protects the user ratings and interaction behaviors from being inferred by the server, but the rated items can still be leaked to other participants who received the split numbers. To tackle this problem, Lin e ​ t ​ a ​ l . et\ al. [ 47 ] combine secret sharing and pseudo items mechanisms to provide a stronger privacy guarantee.

 
 
 Secret sharing mechanism based FedRS can protect user ratings while maintaining recommendation accuracy, and have lower computation costs compared to homomorphic encryption based FedRS. But the exchange process of pieces between participants greatly increases the communication costs.

 
 
 

### III-D Local Differential Privacy 

 
 Considering the huge computation or communication costs caused by encryption based mechanisms, many studies try to use perturbation based mechanisms to adapt to large-scale FedRS for industrial scenarios. Local differential privacy (LDP) mechanism allows statistical computations while guaranteeing each individual participant’s privacy [ 48 ] , which can be used to perturb the intermediate parameters in FedRS.

 
 
 For example, Dolui e ​ t ​ a ​ l . et\ al. [ 29 ] propose a federated matrix factorization framework, which applies differential privacy on item embedding matrix before sending it to the server for weighted average. However, the server can still infer which items the user has rated just by comparing the changes in item embedding matrix.

 
 
 In order to achieve more comprehensive privacy protection during model training process, Wu e ​ t ​ a ​ l . et\ al. [ 32 ] combine pseudo items and LDP mechanisms to protect both user interaction behaviors and ratings in FedGNN. Firstly, to protect user interaction behaviors in FedGNN, the clients randomly sample N N items that they have not interacted with, then generate the virtual gradients of item embeddings by using the same Gaussian distribution as the real embedding gradients. Secondly, to protect user ratings in FedGNN, the clients apply a LDP module to clip the gradients according to their L2-norm with a threshold δ \delta and perturb the gradients by adding zero-mean Laplacian noise. The LDP module of FedGNN can be formulated as follow:

 

 
 | 
 g i = c ​ l ​ i ​ p ​ ( g i , δ ) + L ​ a ​ p ​ l ​ a ​ c ​ e ​ ( 0 , λ ) . g_{i}=clip(g_{i},\delta)+Laplace(0,\lambda). | 
 | 
 (5) | 
 

 where λ \lambda is the Laplacian noise strength. However, the gradient magnitude of different parameters varies during training process, thus it is usually not appropriate to perturb gradients at different magnitudes with a constant noise strength. So Liu e ​ t ​ a ​ l . et\ al. [ 46 ] propose to add dynamic noise according to the gradients, which can be formulated as follow:

 

 
 | 
 g i = c ​ l ​ i ​ p ​ ( g i , δ ) + L ​ a ​ p ​ l ​ a ​ c ​ e ​ ( 0 , λ ⋅ m ​ e ​ a ​ n ​ ( g i ) ) . g_{i}=clip(g_{i},\delta)+Laplace(0,\lambda\cdot mean(g_{i})). | 
 | 
 (6) | 
 

 
 
 Local differential privacy mechanism doesn’t bring heavy computation and communication overhead to FedRS, but the additional noise inevitably affects the performance of the recommendation model. Thus, in the actual application scenario, we must consider the trade-off between privacy and recommendation accuracy.

 
 
 

### III-E Comparison 

 
 To provide a stronger privacy guarantee, many privacy mechanisms (i.e., pseudo items, homomorphic encryption, differential privacy and secret sharing) have been widely used in FedRS, and the comparison between these mechanisms is shown in Table I . Firstly, the main protect objects of these mechanisms are different: pseudo items mechanism is to protect user interaction behaviors, and the rest mechanisms are to protect user ratings. Besides, homomorphic encryption can also integrate data from other participants in a privacy-preserving way. Secondly, homomorphic encryption and secret sharing are both encryption-based mechanisms, and they can protect privacy while keeping accuracy. However, the high computation cost of homomorphic encryption limits it’s application in large-scale industrial scenarios. Although the secret sharing mechanism reduces the computation costs, the communication costs increase greatly. Pseudo items and differential privacy mechanisms protect privacy by adding random noise, which has low computation costs and don’t bring additional communication costs. But the addition of random noise will inevitably affect model performance to a certain extent.

 
 
 
 

## IV Security of Federated recommendation Systems 

 
 Apart from privacy leakage problems, traditional recommendation systems for centralized data storage are also vulnerable to poisoning attacks (shilling attacks) [ 49 ] [ 50 ] [ 51 ] . Attackers can poison recommendation systems and make recommendations as their desire by injecting well-crafted data into the training dataset. But most of these poisoning attacks assume that the attackers have full prior knowledge of entire training datasets. Such an assumption may be not valid for FedRS since the data in FedRS is distributed and stored locally for each participant. Thus, FedRS provides a stronger security guarantee than traditional recommendation systems. However, the latest studies indicate that attackers can still conduct poisoning attacks on FedRS with limited prior knowledge [ 11 ] [ 12 ] [ 52 ] [ 53 ] . In this section, we summarize some novel poisoning attacks against FedRS and provide some defense methods.

 
 
 TABLE II: Representative works on the security of FedRS. RA refers to robust aggregation and AD refers to anomaly detection. 
 
 
 Works | 
 Ref | 
 Attack Type | 
 Poison Object | 
 Defense Type | 
 Goal | 

 
 Target | 
 Untarget | 
 Model | 
 Data | 
 RA | 
 AD | 

 
 PipAttack | 
 [ 11 ] | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 Increase/decrease popularity of target items. | 

 
 FedRecAttack | 
 [ 52 ] | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 Increase/decrease popularity of target items. | 

 
 A-ra/A-hum | 
 [ 53 ] | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 Increase/decrease popularity of target items. | 

 
 FedAttack | 
 [ 12 ] | 
 | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 Degrade the overall performance of FedRS. | 

 
 Median | 
 [ 54 ] | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 Guarantee global model convergence. | 

 
 Trimmed-Mean | 
 [ 54 ] | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 Guarantee global model convergence. | 

 
 (Multi-)Krum | 
 [ 55 ] | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 Guarantee global model convergence. | 

 
 Bulyan | 
 [ 56 ] | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 Guarantee global model convergence. | 

 
 Norm-Bounding | 
 [ 57 ] | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 Guarantee global model convergence. | 

 
 A-FRS | 
 [ 58 ] | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 Guarantee global model convergence. | 

 
 FSAD | 
 [ 59 ] | 
 | 
 | 
 | 
 | 
 | 
 ✓ | 
 Identify and filter poisoned parameters. | 

 
 

### IV-A Poisoning Attacks 

 
 According to the goal of attacks, the poisoning attacks against FedRS can be categorized into targeted attacks and untargeted attacks as shown in Table II .

 
 

#### IV-A 1 Target Poisoning Attacks

 
 The goal of target attacks on FedRS is to increase or decrease the exposure chance of specific items, which are usually driven by financial profit. For example, Zhang e ​ t ​ a ​ l . et\ al. [ 11 ] propose a poisoning attack for item promotion (PipAttack) against FedRS by utilizing popularity bias. To boost the rank score of target items, PipAttack uses popularity bias to align target items with popular items in the embedding space. Besides, to avoid damaging recommendation accuracy and being detected, PipAttack designs a distance constraint to keep modified gradients uploaded by malicious clients close to normal ones.

 
 
 In order to further reduce the degradation of recommendation accuracy caused by targeted poisoning attacks, and the proportion of malicious clients needed to ensure the attack effectiveness, Rong [ 52 ] propose a model poisoning attack against FedRS (FedRecAttack), which makes use of a small proportion of public interactions to approximate the user feature matrix, then uses it to generate poisoned gradients.

 
 
 Both PipAttack and FedRecAttack rely on some prior knowledge. For example, PipAttack assumes the attack can access popularity information, and FedRecAttack assumes the attacker can get public interactions. So the attack effectiveness is greatly reduced in the absence of prior knowledge, which makes both attacks not generic in all FedRS. To make attackers conduct effective poisoning attacks to FedRS without prior knowledge, Rong e ​ t ​ a ​ l . et\ al. [ 53 ] design two methods (i.e., random approximation and hard user mining) for malicious clients to generate poisoned gradients. In particular, random approximation (A-ra) uses Gaussian distribution to approximate normal users’ embedding vectors, and hard user mining (A-hum) uses gradient descent to optimize users’ embedding vectors obtained by A-ra to mine hard users. In this way, A-hum can still effectively attack FedRS with extremely small proportion of malicious users.

 
 
 

#### IV-A 2 Untarget Poisoning Attacks

 
 The goal of untarget attacks on FedRS is to degrade the overall performance of the recommendation model, which are usually conducted by competing companies. For example, Wu e ​ t ​ a ​ l . et\ al. [ 12 ] propose an untargeted poisoning attack to FedRS named FedAttack, which uses a globally hard sampling technique [ 60 ] to subvert model training process. More specifically, after inferring the user’s interest from local user profiles, the malicious clients select candidate items that best match the user’s interest as negative samples, and select candidate items that least match the user’s interest as positive samples. FedAttack only modifies training samples, and the malicious clients are also similar to normal clients with different interests, thus FedAttack can effectively damage the performance of FedRS even under defense.

 
 
 
 

### IV-B Defense Methods 

 
 To reduce the influence of poisoning attacks on FedRS, many defense methods have been proposed in the literature, which can be classified into robust aggregation and anomaly detection.

 
 

#### IV-B 1 Robust Aggregation

 
 The goal of robust aggregation is to guarantee global model convergence when up to 50% of participants are malicious [ 18 ] , which selects statistically more robust values rather than the mean values of uploaded intermediate parameters for aggregation.

 
 
 Median [ 54 ] . Median selects the median value of each updated model parameter independently as aggregated global model parameter, which can represent the center of the distribution better. Specifically, the server ranks each i − t ​ h i-th parameter of n n local model update, and uses the median value as i − t ​ h i-th parameter of the global model.

 
 
 Trimmed-Mean [ 54 ] . Trimmed-Mean removes the maximum and minimum values of each updated model parameter independently, and then takes the mean value as aggregated global model parameter. Specifically, the server ranks each i − t ​ h i-th parameter of n n local model update, removes β \beta smallest and β \beta largest values, and uses the mean value of remaining n − 2 ​ β n-2\beta as i − t ​ h i-th parameter of global model. In this way, Trimmed-Mean can effectively reduce the impact of outliers.

 
 
 Krum and Multi-Krum [ 55 ] . Krum selects a local model that is the closest to the others as the global model. Multi-Krum selects multiple local models by using Krum, then aggregates them into a global model. In this way, even if the selected parameter vectors are uploaded by malicious clients, their impact is still limited because they are similar to other local parameters uploaded by normal clients.

 
 
 Bulyan [ 56 ] . Bulyan is a combination of Krum and Trimmed-Mean, which iteratively selects m m local model parameter vectors through Krum, and then performs Trimmed-Mean on these m m parameter vectors for aggregation. With high dimensional and highly non-convex loss functions, Bulyan can still converge to effectual models.

 
 
 Norm-Bounding [ 57 ] . Norm-Bounding clips the received local parameters to a fixed threshold, then aggregates them to update the global model. Norm-Bounding can limit the contribution of each local model updates so as to mitigate the affect of poisoned parameters on the aggregated model.

 
 
 A-FRS [ 58 ] . A-FRS utilizes gradient-based Krum instead of model parameter-based Krum to filter malicious clients in momentum-based FedRS. A-FRS theoretically guarantees that if the selected gradient is close to the normal gradient, the momentum and model parameters will also be close to the normal momentum and model parameters.

 
 
 Although these robust aggregation strategies provide convergence guarantees to some extent, most of them (i.e., Bulyan, Krum, Median and Trimmed-mean) greatly degrade the performance of FedRS. Besides, some noval attacks(i.e., PipAttack, FedAttack) [ 11 ] [ 12 ] utilize well-designed constraints to approximate the patterns of normal users and circumvent defenses, which further increases the difficulty of defense.

 
 
 

#### IV-B 2 Anomaly Detection

 
 The purpose of anomaly detection strategy is to identify the poisoned model parameters uploaded by malicious clients and filter them during the global model aggregation process. For example, Jiang e ​ t ​ a ​ l . et\ al. [ 59 ] propose an anomaly detection strategy named federated shilling attack detector (FSAD) to detect poisoned gradients in federated collaborative filtering scenarios. FSAD extracts 4 novel features according to the gradients uploaded by clients, then uses the gradient-based features to train a semi-supervised bayes classifier so as to identify and filter the poisoned gradients. However, in FedRS, the interests of different users vary widely, thus the parameters they uploaded are usually quite different, which increases the difficulty of anomaly detection [ 52 ] .

 
 
 Fig. 3: Heterogeneity of federated recommendation systems. 
 
 
 
 
 

## V Heterogeneity of Federated Recommendation Systems 

 
 Compared with traditional recommendation systems, FedRS face more severe challenges in terms of heterogeneity, which are mainly reflected in system heterogeneity, statistical heterogeneity and model heterogeneity, as shown in Fig. 3 . System heterogeneity refers to client devices have significantly different storage, computation, and communication capabilities. Devices with limited capabilities greatly affect training efficiency, and further reduce the accuracy of the global recommendation model. [ 61 ] ; Statistical heterogeneity refers to the data collected by different clients is usually not independent and identically distributed (non-IID). As a result, simply training a single global model is difficult to generalize to all clients, which affects the personalization of recommendations [ 62 ] ; Privacy heterogeneity means that the privacy constraints of different users and information vary greatly, so simply treating them with the same privacy budgets will carry unnecessary costs [ 63 ] . This section introduces some effective approaches to address the heterogeneity of FedRS.

 
 

### V-A System Heterogeneity 

 
 In FedRS, the hardware configuration, network bandwidth and battery capacity of participating clients vary greatly, which results in diverse computing capability, communication speed, and storage capability [ 16 ] . During the training process, the clients with limited capacity could become stragglers, and even drop out of current training due to network failure, low battery and other problems [ 20 ] . The system heterogeneity significantly delays the training process of FedRS, further reducing the recommendation accuracy of the global model. To make the training process compatible with different hardware structures and tolerate the straggling and exit issues of clients, the most common methods are asynchronous communication [ 64 ] [ 20 ] and clients selection [ 65 ] .

 
 
 Asynchronous communication. Considering the synchronous communication based federated learning must wait for straggler devices during the aggregation process, many asynchronous communication strategies are presented to improve training efficiency. For examples, FedSA [ 64 ] proposes a semi-asynchronous communication method, where the server aggregates the local models based on their arrival order of each round. FedAsync [ 20 ] uses a weighted average strategy to aggregate the local models based on staleness, which assigns less weight to delayed feedback in the update process.

 
 
 Clients selection . Client selection approach selects clients for updates based on resource constraints so that the server can aggregate as many local updates as possible at the same time. For example, in FedCS [ 65 ] , the server sends a resource request to each client so as to get their resource information, then estimates the required time of model distribution, updating and uploading processes based on the resource information. According to the estimated time, the server determines which clients can participant in the training process.

 
 
 

### V-B Statistical Heterogeneity 

 
 Most of the existing federated recommendation studies are built on the assumption that data in each participant is independent and identically distributed (IID). However, the data distribution of each client usually varies greatly, hence training a consistent global model is difficult to generalize to all clients under non-IID data and inevitably neglects the personalization of clients [ 63 ] . To address the statistical heterogeneity problem of FedRS, many effective strategies have been proposed, which are mainly based on meta learning [ 39 ] [ 66 ] and clustering [ 67 ] [ 68 ] .

 
 
 Meta learning. As known as “learning to learn”, meta learning technology aims to quickly adapt the global model learned by other tasks to a new task by using only a few samples [ 37 ] . The rapid adaptation and good generalization abilities make it particularly well-suited for building personalized federated recommendation models. For examples, FedMeta [ 39 ] uses Model-Agnostic Meta-Learning (MAML) [ 69 ] algorithm to learn a well-initialized model that can be quickly adapted to clients, and effectively improve the personalization and convergence of FedRS. However, FedMeta needs to compute the second-order gradients, which greatly increases computation costs. Besides, the data split process also brings a huger challenge for clients with limited samples. Based on FedMeta, Wang e ​ t ​ a ​ l . et\ al. [ 66 ] propose a new meta learning algorithm called Reptile which applies the approximate first-order derivatives for the meta-learning updates, which greatly reduces the computation overloads of clients. Moreover, Reptile doesn’t need a data split process, which makes it also suitable for clients with limited samples.

 
 
 Clustering. The core idea of clustering is training personalized models jointly with the same group of homogeneous clients. For examples, Jie e ​ t ​ a ​ l . et\ al. [ 67 ] use historical parameter clustering technology to realize personalized federated recommendation, in which the server aggregates local parameters to generate global model parameters and clusters the local parameters to generate clustering parameters for different client groups. Then the clients combine the clustering parameters with the global parameters to learn personalized models. Luo e ​ t ​ a ​ l . et\ al. [ 68 ] propose a personalized federated recommendation framework named PerFedRec, which constructs a collaborative graph and integrates attribute information so as to jointly learn the user representations by federated GNN. Based on the learned user representations, clients are clustered into different groups. And each cluster learns a cluster-level recommendation model. At last, each client can obtain a personalized model by merging the global recommendation model, the cluster-level recommendation model, and the fine-tuned local recommendation model. Although clustering based approaches can alleviate statistical heterogeneity, the clustering and combination process greatly increase the computation costs.

 
 
 

### V-C Privacy Heterogeneity 

 
 In reality, the privacy restrictions of different participants and information vary greatly, thereby using the same high level of privacy budget for all participants and information is unnecessary, which even increases the computation/communication costs and degrades the model performance.

 
 
 Heterogeneous user privacy . In order to adapt to the privacy needs of different users, Anelli e ​ t ​ a ​ l . et\ al. [ 70 ] present a user controlled federated recommendation framework named FedeRank. FedeRank introduces a probability factor π ∈ [ 0 , 1 ] \pi\in[0,1] to control the proportion of interacted item updates and masks the remain interacted item update by setting them to zero. In this way, FedeRank allows users to decide the proportion of data they want to share by themselves, which addresses the heterogeneity of user privacy.

 
 
 Heterogeneous information privacy . In order to adapt to the privacy needs of different information components, HPFL [ 63 ] designs a differentiated component aggregation strategy. To obtain the global public information components, the server directly weighted aggregates the local public components with the same properties. And to obtain the global privacy information components, the user and item representations are kept locally, and the server only aggregates the local drafts without the need to align the presentations. With the differentiated component aggregation strategy, HPFL can safely aggregate components with heterogeneous privacy constraints in user modeling scenarios.

 
 
 
 

## VI Communication Costs of Federated recommendation Systems 

 
 To achieve satisfactory recommendation performance, FedRS requires multiple communications between the server and clients. However, the real-world recommendation systems are usually conducted by complexity deep learning models with large model size [ 71 ] , and millions of parameters need to be updated and communicated [ 13 ] , which brings severe communication overload to resource limited clients and further affects the application of FedRS in large-scale industrial scenarios. This section summarizes some optimization methods to reduce communication costs of FedRS, which can be classified into importance-based updating [ 72 ] [ 22 ] [ 73 ] [ 74 ] , model compression [ 75 ] [ 76 ] , active sampling [ 77 ] and one shot learning [ 78 ] .

 
 

### VI-A Importance-based Model Updating 

 
 Importance-based model updating selects important parts of the global model instead of the whole model to update and communicate, which can effectively reduce the communicated parameter size in each round.

 
 
 For examples, Qin e ​ t ​ a ​ l . et\ al. [ 72 ] propose a federated framework named PPRSF, which uses 4-layers hierarchical structure for reducing communication costs, including the recall layer, ranking layer, re-ranking layer and service layer. In the recall layer, the server roughly sorts the large inventory by using public user data, and recalls a relatively small number of items for each client. In this way, the clients only need to update and communicate the candidate item embeddings, which greatly reduces the communication costs between the server and clients, and the computation costs in the local model training and inference phases. However, the recall layer of PPRSF needs to get some public information about users, which raises certain difficulty and privacy concerns.

 
 
 Yi e ​ t ​ a ​ l . et\ al. [ 22 ] propose an efficient federated news recommendation framework called Efficient-FedRec, which breaks the news recommendation model into a small user model and a big news model. Each client only requests the user model and a few news representations involved in their local click history for local training, which greatly reduces the communication and computation overhead. To further protect specific user click history against the server, they transmit the union news representations set involved in a group of user click history by using a secure aggregation protocol [ 79 ] .

 
 
 Besides, Khan e ​ t ​ a ​ l . et\ al. [ 73 ] propose a multi-arm bandit method (FCF-BTS) to select part of the global model that contains a smaller payload for all clients. The rewards of the selection process are guided by Bayesian Thompson Sampling (BTS) [ 80 ] approach with Gaussian priors. Experiments show that FCF-BTS can reduce 90% model payload for highly sparse datasets. Besides, the selection process occurs on the server side, thus avoiding additional computation costs for the clients. But FCF-BTS causes 4% - 8% loss in recommendation accuracy.

 
 
 To achieve a better balance between recommendation accuracy and efficiency, Ai e ​ t ​ a ​ l . et\ al. [ 74 ] propose an all-MLP network that uses a Fourier sub-layer to replace the self-attention sub-layer in a Transformer encoder so as to filter noise data components unrelated to the user’s real interests, and adapts an adaptive model pruning technique to discard the noise model components that doesn’t contribute to model performance. Experiments show that all-MLP network can significantly reduce communication and computation costs, and accelerates the model convergence.

 
 
 Importance-based model updating strategies can greatly reduce communication and computation costs at the same time, but only selecting the important parts for updating inevitably reduces the recommendation performance.

 
 
 

### VI-B Model Compression 

 
 Model Compression is a well-known technology in distributed learning [ 81 ] , which compresses the communicated parameters per round to be more compact.

 
 
 For examples, Konen e ​ t ​ a ​ l . et\ al. [ 75 ] propose two methods (i.e., structured updates and sketched updates) to decrease the uplink communication costs under federated learning settings. Structured updates method directly learns updates from a pre-specified structure parameterized using fewer variables. Sketched updates method compresses the full local update using a lossy compression way before sending it to the server. These two strategies can reduce communication costs by 2 orders of magnitude.

 
 
 To reduce the uplink communication costs in deep learning based FedRS, JointRec [ 76 ] combines low-rank matrix factorization [ 82 ] and 8-bit probabilistic quantization [ 83 ] methods to compress weight update. Supposing the weight update matrix of client n is H n a × b H_{n}^{a\times b} , a ≤ b a\leq b , low-rank matrix factorization decomposes H n a × b H_{n}^{a\times b} into two matrices: H n a × b = U n a × k ​ V n k × b H^{a\times b}_{n}=U^{a\times k}_{n}V^{k\times b}_{n} , where k = b / N k=b/N and N is a positive number that influences the compression performance. And 8-bit probabilistic quantization method transforms the position of matrix value into 8-bit value before sending it to the server. Experiments demonstrate that JointRec can realize 12.83 × 12.83\times larger compression ratio while maintaining recommendation performance.

 
 
 Model compression methods achieve significant results in reducing uplink communication costs. However, the reduction of communication costs sacrifices the computation resources of the clients, so it’s necessary to consider the trade-off between computation and communication costs when using model compression.

 
 
 

### VI-C Client Sampling 

 
 In traditional federated learning frameworks [ 8 ] , the server randomly selects clients to participate in the training process and simply aggregates the local models by average, which requires a large number of communications to realize satisfactory accuracy. Client sampling utilizes efficient sampling strategies so as to improve training efficiency and reduce the communication rounds.

 
 
 For example, Muhammad e ​ t ​ a ​ l . et\ al. [ 77 ] propose an effective sampling strategy named FedFast to speed up the training efficiency of federated recommendation models while keeping more accuracy. FadFast consists of two efficient components: ActvSAMP and ActvAGG. ActvSAMP uses K-means algorithm to cluster users based on their profiles, and samples clients in equal proportions from each cluster. And ActvAGG propagates local updates to the other clients in the same cluster. In this way, the learning process for these similar users is greatly accelerated and the overall efficiency of the FedRS is consequently improved. Experiments show that FedFast reduces communication rounds by 94% compared to FedAvg [ 8 ] . However, FedFast is faced with the cold start problem because it requires a number of users and items for training. Besides, FedFast needs to retrain the model to support new users and items.

 
 
 

### VI-D One Shot Federated Learning 

 
 The goal of one shot federated learning mechanism is to reduce communication rounds of FedRS [ 84 ] [ 85 ] , which limits communication to a single round to aggregate knowledge of local models. For example, Eren e ​ t ​ a ​ l . et\ al. [ 78 ] implement an one-shot federated learning framework for cross-platform FedRS named FedSPLIT. FedSPLIT aggregates model through knowledge distillation [ 86 ] , which can generate client specific recommendation results with just a single pair of communication rounds between the server and clients after a small initial communication. Experiments show that FedSPLIT realizes similar root-mean-square error (RMSE) compared with multi-round communication scenarios, but it is not applicable to the scenario where the participants are individual users.

 
 
 
 

## VII Applications And Public Benchmark Datasets 

 
 This section introduces the typical applications and public benchmark datasets for FedRS.

 
 

### VII-A Applications 

 
 Online services. 
Currently, online services have been involved in various fields of our life such as news, movie and music. A large amount of private information of users is collected and stored centrally by service providers, which faces a serious risk of privacy leakage. User data may be sold to third parties by service providers or stolen by external hackers. FedRS can help users enjoy personalized recommendation services while keeping personal privacy, make the service providers more trusted by users, and ensure the recommendation service complies with the regulations. For example, Tan e ​ t ​ a ​ l . et\ al. [ 87 ] design a federated recommendation system that implements various popular recommendation algorithms to support lots of online recommendation services, and deploys it on a real-world content recommendation application.

 
 
 Healthcare. 
Healthcare recommendation enables patients to enjoy medical service from mobile applications instead of going to the hospital in person when obtaining satisfactory recommendations. Medical data is quite private and sensitive, which means it is hard to fuse user information from different hospitals or other organizations to improve recommendation quality. In this scenario, FedRS can break down the data silos and utilize these data without compromising patients’ privacy. For example, Song e ​ t ​ a ​ l . et\ al. [ 88 ] develop a telecommunication-joint federated healthcare recommendation platform based on Federated AI Technology Enabler (FATE), which helps healthcare providers improve recommendation performance by complementing common user data (e.g., demographic information, user behaviors and geographic information) from mobile network operators. Besides, this platform designs a federated gradient boosting decision tree (FGDBT) model, improving 9.71% of precision and 4% of F1 score for healthcare recommendation. The platform has been deployed on both organizations and applied to online operation.

 
 
 Advertisement. Advertisement is another significant application of FedRS. Platforms that display advertisements often face the problem of insufficient user data and low click-through rates (CTR) for advertisements. FedRS is able to exploit user data across different platforms in a privacy-preserving way, which can better infer user interest and push advertisements more accurately. For example, Wu e ​ t ​ a ​ l . et\ al. [ 25 ] propose a native advertisement CTR prediction method named FedCTR, which can integrate multi-platform user behaviors (e.g., advertisements click behavior, search behavior and browsing behavior) for user interest modeling with no need for centralized storage.

 
 
 E-commerce. 
Currently, recommendation system plays a significant role in e-commerce platforms (e.g., Alibaba, Amazon). To provide users with more precise recommendation services, such systems try to integrate more auxiliary information (e.g., user purchasing power, social information). However, these data are usually distributed on different platforms and difficult to access directly due to regulations and privacy concerns. FedRS can address this problem effectively while meeting regulations and privacy.

 
 
 Point-of-Interest. 
Point-of-Interest (POI) recommendation exploits the user’s historical check-in data and other modal information (e.g., POI attributes and social information) to recommend suitable POI sets for the user. However, the user’s check-in data is very sensitive and sparse, and users are often reluctant to share their context information due to privacy concerns. FedRS can effectively address the data sparsity problem in a privacy-preserving way, which is quite beneficial for POI recommendation [ 89 ] .

 
 
 

### VII-B Public Benchmark Datasets 

 
 MovieLens [ 90 ] . 
MovieLens rating datasets were published by GroupLens, which consist of user, movie, rating and timestamp information. MovieLens-100K contains 100,000 ratings from 943 users for 1682 movies, and MovieLens-1M contains 1,000,209 ratings from 6,040 users for 3,952 movies.

 
 
 FilmTrust [ 91 ] . 
FilmTrust is a movie rating dataset crawled from the FilmTrust website. The dataset contains 35,497 ratings from 1,508 users for 2,071 films.

 
 
 Foursquare [ 92 ] . 
Foursquare dataset is a famous benchmark dataset to evaluate POI recommendation models collected from Foursquare. The dataset contains 22,809,624 global-scale check-ins by 114,324 users on 3,820,891 POIs with 363,704 social relationships.

 
 
 Epinions [ 93 ] . 
Epinions dataset is an online social network built from a consumer review site Epinions.com, which consists of user ratings and trust social network information. The dataset contains 188,478 ratings from 116,260 users for 41,269 items.

 
 
 Mind [ 94 ] . 
Mind is a large-scale dataset for news recommendation collected from anonymous behavior logs of Microsoft News website, which contains about 160,000 English news articles and more than 15 million impression logs generated by 1 million users.

 
 
 LastFM [ 95 ] . 
LastFM dataset was collected from Last.fm online music system, which consists of tagging, music artist listening, and social relationship information. The dataset contains 92,834 listening counts of 17,632 music artists by 1,892 users.

 
 
 Book-Crossing [ 96 ] . 
Book-Crossing dataset is a 4-week crawl dataset from the Book-Crossing community. It contains 1,149,780 ratings (explicit / implicit) for 271,379 books by 278,858 anonymous users with demographic information.

 
 
 
 

## VIII Future Directions 

 
 This section presents and discusses many prospective research directions in the future. Although some directions have been covered in the above sections, we believe they are necessary for FedRS, and need to be further researched.

 
 
 Decentralized FedRS. Most current FedRS are based on client-server communication architecture, which faces single-point-of-failure and privacy issues caused by the central server [ 97 ] . While much work has been devoted to decentralized federated learning [ 98 ] [ 99 ] , few decentralized FedRS have been studied. A feasible solution is to replace client-server communication architecture with peer-peer communication architecture to achieve fully decentralized federated recommendation. For example, Hegeds e ​ t ​ a ​ l . et\ al. [ 21 ] propose a fully decentralized matrix factorization framework based on gossip learning [ 100 ] , where each participant sends their copy of the global recommendation model to random online neighbors in the peer to peer network. In addition, swarm learning [ 101 ] , a decentralized machine learning framework that combines edge computing, blockchain based peer-peer networks and coordination, can keep confidentiality without the need for a central server. Therefore, it is also a promising way to implement decentralized recommendation systems.

 
 
 Incentive mechanisms in FedRS. 
FedRS collaborate with multiple participants to train a global recommendation model, and the recommendation performance of the global model is highly dependent on the quantity and quality of data provided by the participants. Therefore, it is significant to design an appropriate incentive mechanism to inspire participants to contribute their own data and participate in collaborative training, especially in the cross-organization federated recommendation scenarios. The incentive mechanisms must be able to measure the clients’ contribution to the global model fairly and efficiently.

 
 
 Architecture design for FedRS. 
The recommendation systems in industrial scenarios usually consist of the recall layer and ranking layer, which generate recommendation results on the server side. Considering the privacy of users, FedRS must adopt different designs. A feasible solution is local recalling and ranking, where the server sends the entire set of candidate items to clients, and clients generate recommendation results locally. However, such design brings enormous communication, computation and memory costs for clients since there are usually millions of items in real-world recommendation systems. Another effective approach is to place the recall layer on the server side and the ranking layer on the client side, where clients send encrypted or noised user embedding to the server to recall top-N candidate items, then clients generate personalized recommendation results based one these candidate items via ranking layer [ 102 ] . Nevertheless, there is a risk of privacy leakage associated with this approach, because recalled items are known to the server.

 
 
 Cold start problem in FedRS. The cold start problem means that recommendation systems cannot generate satisfactory recommendation results for new users with little history interactions [ 103 ] . In federated settings, the user data is stored locally, so it is more difficult to integrate other auxiliary information (e.g., social relationships) to alleviate the cold start problem. Therefore, it is a challenging and prospective research direction to address the cold start problem while ensuring user privacy.

 
 
 Secure FedRS. 
In the real world, the participants in the FedRS are likely to be untrustworthy. Therefore, participants may upload poisoned intermediate parameters to affect recommendation results or destroy recommendation performance. Although some robust aggregation strategies [ 55 ] and detection methods [ 59 ] have been proposed to defend against poisoning attacks in federated learning settings, most of them don’t work well in FedRS. On one hand, some strategies such as Krum, Median and Trimmed-mean degrade the recommendation performance to a certain extent. On the other hand, some novel attacks [ 12 ] use well-designed constraints to mimic the patterns of normal users, extremely increasing the difficulty of detection and defense. Currently, there are still no effective defense methods against these poisoning attacks while maintaining recommendation accuracy.

 
 
 

## IX Conclusion 

 
 A lot of effort has been devoted to federated recommendation systems. A comprehensive survey is significant and meaningful. This survey summarizes the latest studies on aspects of privacy, security, heterogeneity and communication costs. Based on these aspects, we also make a detailed comparison among the existing designs and solutions. Moreover, we present many prospective research directions to promote development in this field. FedRS will be a promising field with huge potential opportunities, which requires more effort to develop.

 
 
 

## Acknowledgments

 
 This research is partially supported by the National Key R D Program of China 2021YFF0900800, NSFC No.62202279, the Shandong Provincial Key Research and Development Program (Major Scientific and Technological Innovation Project) (No.2021CXGC010108), the Shandong Provincial Natural Science Foundation (No.ZR2022QF018), Shandong Province Outstanding Youth Science Foundation, the Fundamental Research Funds of Shandong University, CCF-Huawei Populus Grove Fund, and the Special Fund for Science and Technology of Guangdong Province under Grant (2021S0053). Sino-Singapore International Joint Research Project (No. 206-A021002).

 
 
 

## References

 
 
 [1] 
 
B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, “Analysis of recommendation
algorithms for e-commerce,” in Proceedings of the 2nd ACM Conference
on Electronic Commerce , 2000, pp. 158–167.

 

 
 [2] 
 
J. B. Schafer, J. A. Konstan, and J. Riedl, “E-commerce recommendation
applications,” Data mining and knowledge discovery , vol. 5, no. 1,
pp. 115–153, 2001.

 

 
 [3] 
 
G. Zheng, F. Zhang, Z. Zheng, Y. Xiang, N. J. Yuan, X. Xie, and Z. Li, “Drn: A
deep reinforcement learning framework for news recommendation,” in
 Proceedings of the 2018 world wide web conference , 2018, pp. 167–176.

 

 
 [4] 
 
J. Liu, P. Dolan, and E. R. Pedersen, “Personalized news recommendation based
on click behavior,” in Proceedings of the 15th international
conference on Intelligent user interfaces , 2010, pp. 31–40.

 

 
 [5] 
 
W. Yue, Z. Wang, J. Zhang, and X. Liu, “An overview of recommendation
techniques and their applications in healthcare,” IEEE/CAA Journal of
Automatica Sinica , vol. 8, no. 4, pp. 701–717, 2021.

 

 
 [6] 
 
J. Kim, D. Lee, and K.-Y. Chung, “Item recommendation based on context-aware
model for personalized u-healthcare service,” Multimedia Tools and
Applications , vol. 71, no. 2, pp. 855–872, 2014.

 

 
 [7] 
 
J. P. Albrecht, “How the gdpr will change the world,” Eur. Data Prot.
L. Rev. , vol. 2, p. 287, 2016.

 

 
 [8] 
 
B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
“Communication-efficient learning of deep networks from decentralized
data,” in Artificial intelligence and statistics . PMLR, 2017, pp. 1273–1282.

 

 
 [9] 
 
D. Chai, L. Wang, K. Chen, and Q. Yang, “Secure federated matrix
factorization,” Intelligent Systems, IEEE , vol. PP, no. 99, pp. 1–1,
2020.

 

 
 [10] 
 
G. Lin, F. Liang, W. Pan, and Z. Ming, “Fedrec: Federated recommendation with
explicit feedback,” Intelligent Systems, IEEE , vol. PP, no. 99, pp.
1–1, 2020.

 

 
 [11] 
 
S. Zhang, H. Yin, T. Chen, Z. Huang, Q. V. H. Nguyen, and L. Cui, “Pipattack:
Poisoning federated recommender systems for manipulating item promotion,” in
 Proceedings of the Fifteenth ACM International Conference on Web Search
and Data Mining , 2022, pp. 1415–1423.

 

 
 [12] 
 
C. Wu, F. Wu, T. Qi, Y. Huang, and X. Xie, “Fedattack: Effective and covert
poisoning attack on federated recommendation via hard sampling,” in
 Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining , ser. KDD ’22. New
York, NY, USA: Association for Computing Machinery, 2022, p. 4164–4172.
[Online]. Available: https://doi.org/10.1145/3534678.3539119 

 

 
 [13] 
 
C.-L. Liao and S.-J. Lee, “A clustering based approach to improving the
efficiency of collaborative filtering recommendation,” Electronic
Commerce Research and Applications , vol. 18, pp. 1–9, 2016.

 

 
 [14] 
 
G. Adomavicius and A. Tuzhilin, “Toward the next generation of recommender
systems: a survey of the state-of-the-art and possible extensions,”
 IEEE Transactions on Knowledge and Data Engineering , vol. 17, no. 6,
pp. 734–749, 2005.

 

 
 [15] 
 
Q. Yang, Y. Liu, T. Chen, and Y. Tong, “Federated machine learning: Concept
and applications,” ACM Transactions on Intelligent Systems and
Technology , vol. 10, no. 2, pp. 1–19, 2019.

 

 
 [16] 
 
T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning:
Challenges, methods, and future directions,” IEEE Signal Processing
Magazine , vol. 37, no. 3, pp. 50–60, 2020.

 

 
 [17] 
 
V. Mothukuri, R. M. Parizi, S. Pouriyeh, Y. Huang, A. Dehghantanha, and
G. Srivastava, “A survey on security and privacy of federated learning,”
 Future Generation Computer Systems , vol. 115, pp. 619–640, 2021.
[Online]. Available:
 https://www.sciencedirect.com/science/article/pii/S0167739X20329848 

 

 
 [18] 
 
L. Lyu, H. Yu, X. Ma, C. Chen, L. Sun, J. Zhao, Q. Yang, and S. Y. Philip,
“Privacy and robustness in federated learning: Attacks and defenses,”
 IEEE Transactions on Neural Networks and Learning Systems , 2022.

 

 
 [19] 
 
L. Yang, B. Tan, V. W. Zheng, K. Chen, and Q. Yang, “Federated recommendation
systems,” in Federated Learning . Springer, 2020, pp. 225–239.

 

 
 [20] 
 
C. Xie, S. Koyejo, and I. Gupta, “Asynchronous federated optimization,”
 arXiv preprint arXiv:1903.03934 , 2019.

 

 
 [21] 
 
I. Hegedűs, G. Danner, and M. Jelasity, “Decentralized recommendation
based on matrix factorization: a comparison of gossip and federated
learning,” in Joint European Conference on Machine Learning and
Knowledge Discovery in Databases . Springer, 2019, pp. 317–332.

 

 
 [22] 
 
J. Yi, F. Wu, C. Wu, R. Liu, G. Sun, and X. Xie, “Efficient-fedrec: Efficient
federated learning framework for privacy-preserving news recommendation,”
 arXiv preprint arXiv:2109.05446 , 2021.

 

 
 [23] 
 
M. Ammad-Ud-Din, E. Ivannikova, S. A. Khan, W. Oyomno, Q. Fu, K. E. Tan, and
A. Flanagan, “Federated collaborative filtering for privacy-preserving
personalized recommendation system,” arXiv preprint arXiv:1901.09888 ,
2019.

 

 
 [24] 
 
C. Chen, L. Li, B. Wu, C. Hong, L. Wang, and J. Zhou, “Secure social
recommendation based on secret sharing,” arXiv preprint
arXiv:2002.02088 , 2020.

 

 
 [25] 
 
WuChuhan, WuFangzhao, LyuLingjuan, HuangYongfeng, and XieXing, “Fedctr:
Federated native ad ctr prediction with cross platform user behavior data,”
 ACM Transactions on Intelligent Systems and Technology (TIST) , 2021.

 

 
 [26] 
 
S. Kalloori and S. Klingler, “Horizontal cross-silo federated recommender
systems,” in Fifteenth ACM Conference on Recommender Systems , 2021,
pp. 680–684.

 

 
 [27] 
 
L. Yang, Y. Yu, and Y. Wei, “Data-driven artificial intelligence
recommendation mechanism in online learning resources,” International
Journal of Crowd Science , vol. 6, no. 3, pp. 150–157, 2022.

 

 
 [28] 
 
Koren, Yehuda, Bell, Robert, Volinsky, and Chris, “Matrix factorization
techniques for recommender systems.” Computer , vol. 42, no. 8, pp.
30–37, 2009.

 

 
 [29] 
 
K. Dolui, I. Cuba Gyllensten, D. Lowet, S. Michiels, H. Hallez, and D. Hughes,
“Towards privacy-preserving mobile applications with federated learning: The
case of matrix factorization (poster),” in Proceedings of the 17th
Annual International Conference on Mobile Systems, Applications, and
Services , 2019, pp. 624–625.

 

 
 [30] 
 
S. Ying, “Shared mf: A privacy-preserving recommendation system,” arXiv
preprint arXiv:2008.07759 , 2020.

 

 
 [31] 
 
V. Perifanis and P. S. Efraimidis, “Federated neural collaborative
filtering,” Knowledge-Based Systems , vol. 242, p. 108441, 2022.

 

 
 [32] 
 
C. Wu, F. Wu, Y. Cao, Y. Huang, and X. Xie, “Fedgnn: Federated graph neural
network for privacy-preserving recommendation,” arXiv preprint
arXiv:2102.04925 , 2021.

 

 
 [33] 
 
M. Imran, H. Yin, T. Chen, N. Q. V. Hung, A. Zhou, and K. Zheng, “Refrs:
Resource-efficient federated recommender system for dynamic and diversified
user preferences,” ACM Trans. Inf. Syst. , aug 2022, just Accepted.
[Online]. Available: https://doi.org/10.1145/3560486 

 

 
 [34] 
 
X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural collaborative
filtering,” in Proceedings of the 26th international conference on
world wide web , 2017, pp. 173–182.

 

 
 [35] 
 
M. Huang, H. Li, B. Bai, C. Wang, K. Bai, and F. Wang, “A federated multi-view
deep learning framework for privacy-preserving recommendations,” arXiv
preprint arXiv:2008.10808 , 2020.

 

 
 [36] 
 
P.-S. Huang, X. He, J. Gao, L. Deng, A. Acero, and L. Heck, “Learning deep
structured semantic models for web search using clickthrough data,” in
 Proceedings of the 22nd ACM international conference on Information 
Knowledge Management , 2013, pp. 2333–2338.

 

 
 [37] 
 
W.-Y. Chen, Y.-C. Liu, Z. Kira, Y.-C. F. Wang, and J.-B. Huang, “A closer look
at few-shot classification,” arXiv preprint arXiv:1904.04232 , 2019.

 

 
 [38] 
 
A. Nichol, J. Achiam, and J. Schulman, “On first-order meta-learning
algorithms,” arXiv preprint arXiv:1803.02999 , 2018.

 

 
 [39] 
 
F. Chen, M. Luo, Z. Dong, Z. Li, and X. He, “Federated meta-learning with fast
convergence and efficient communication,” arXiv preprint
arXiv:1802.07876 , 2018.

 

 
 [40] 
 
F. Liang, W. Pan, and Z. Ming, “Fedrec++: Lossless federated recommendation
with explicit feedback,” in Proceedings of the AAAI conference on
artificial intelligence , vol. 35, no. 5, 2021, pp. 4224–4231.

 

 
 [41] 
 
A. Abbas, A. Hidayet, U. A. Selcuk, and C. Mauro, “A survey on homomorphic
encryption schemes: Theory and implementation,” Acm Computing
Surveys , vol. 51, no. 4, pp. 1–35, 2017.

 

 
 [42] 
 
P. Paillier, “Public-key cryptosystems based on composite degree residuosity
classes,” in International conference on the theory and applications
of cryptographic techniques . Springer, 1999, pp. 223–238.

 

 
 [43] 
 
J. Zhang and Y. Jiang, “A vertical federation recommendation method based on
clustering and latent factor model,” in 2021 International Conference
on Electronic Information Engineering and Computer Science (EIECS) . IEEE, 2021, pp. 362–366.

 

 
 [44] 
 
V. Perifanis, G. Drosatos, G. Stamatelatos, and P. S. Efraimidis, “Fedpoirec:
Privacy preserving federated poi recommendation with social influence,”
 arXiv preprint arXiv:2112.11134 , 2021.

 

 
 [45] 
 
J. H. Cheon, A. Kim, M. Kim, and Y. Song, “Homomorphic encryption for
arithmetic of approximate numbers,” in International conference on the
theory and application of cryptology and information security . Springer, 2017, pp. 409–437.

 

 
 [46] 
 
Z. Liu, L. Yang, Z. Fan, H. Peng, and P. S. Yu, “Federated social
recommendation with graph neural network,” ACM Transactions on
Intelligent Systems and Technology (TIST) , vol. 13, no. 4, pp. 1–24, 2022.

 

 
 [47] 
 
Z. Lin, W. Pan, and Z. Ming, “Fr-fmss: federated recommendation via fake marks
and secret sharing,” in Fifteenth ACM Conference on Recommender
Systems , 2021, pp. 668–673.

 

 
 [48] 
 
C. Dwork, “Calibrating noise to sensitivity in private data analysis,”
 Lecture Notes in Computer Science , vol. 3876, no. 8, pp. 265–284,
2012.

 

 
 [49] 
 
M. Fang, G. Yang, N. Z. Gong, and J. Liu, “Poisoning attacks to graph-based
recommender systems,” in Proceedings of the 34th annual computer
security applications conference , 2018, pp. 381–392.

 

 
 [50] 
 
H. Huang, J. Mu, N. Z. Gong, Q. Li, B. Liu, and M. Xu, “Data poisoning attacks
to deep learning based recommender systems,” arXiv preprint
arXiv:2101.02644 , 2021.

 

 
 [51] 
 
H. Zhang, Y. Li, B. Ding, and J. Gao, “Practical data poisoning attack against
next-item recommendation,” in Proceedings of The Web Conference 2020 ,
2020, pp. 2458–2464.

 

 
 [52] 
 
D. Rong, S. Ye, R. Zhao, H. N. Yuen, J. Chen, and Q. He, “Fedrecattack: Model
poisoning attack to federated recommendation,” arXiv preprint
arXiv:2204.01499 , 2022.

 

 
 [53] 
 
D. Rong, Q. He, and J. Chen, “Poisoning deep learning based recommender model
in federated learning scenarios,” arXiv preprint arXiv:2204.13594 ,
2022.

 

 
 [54] 
 
D. Yin, Y. Chen, R. Kannan, and P. Bartlett, “Byzantine-robust distributed
learning: Towards optimal statistical rates,” in International
Conference on Machine Learning . PMLR,
2018, pp. 5650–5659.

 

 
 [55] 
 
P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, “Machine learning
with adversaries: Byzantine tolerant gradient descent,” Advances in
Neural Information Processing Systems , vol. 30, 2017.

 

 
 [56] 
 
R. Guerraoui, S. Rouault et al. , “The hidden vulnerability of
distributed learning in byzantium,” in International Conference on
Machine Learning . PMLR, 2018, pp.
3521–3530.

 

 
 [57] 
 
Z. Sun, P. Kairouz, A. T. Suresh, and H. B. McMahan, “Can you really backdoor
federated learning?” arXiv preprint arXiv:1911.07963 , 2019.

 

 
 [58] 
 
C. Chen, J. Zhang, A. K. Tung, M. Kankanhalli, and G. Chen, “Robust federated
recommendation system,” arXiv preprint arXiv:2006.08259 , 2020.

 

 
 [59] 
 
Y. Jiang, Y. Zhou, D. Wu, C. Li, and Y. Wang, “On the detection of shilling
attacks in federated collaborative filtering,” in 2020 International
Symposium on Reliable Distributed Systems (SRDS) , 2020, pp. 185–194.

 

 
 [60] 
 
Y. Kalantidis, M. B. Sariyildiz, N. Pion, P. Weinzaepfel, and D. Larlus, “Hard
negative mixing for contrastive learning,” Advances in Neural
Information Processing Systems , vol. 33, pp. 21 798–21 809, 2020.

 

 
 [61] 
 
P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji,
K. Bonawitz, Z. Charles, G. Cormode, R. Cummings et al. , “Advances
and open problems in federated learning,” Foundations and
Trends® in Machine Learning , vol. 14, no. 1–2, pp. 1–210,
2021.

 

 
 [62] 
 
Z. Jie, S. Chen, J. Lai, M. Arif, and Z. He, “Personalized federated
recommendation system with historical parameter clustering,” Journal
of Ambient Intelligence and Humanized Computing , pp. 1–11.

 

 
 [63] 
 
J. Wu, Q. Liu, Z. Huang, Y. Ning, H. Wang, E. Chen, J. Yi, and B. Zhou,
“Hierarchical personalized federated learning for user modeling,” in
 Proceedings of the Web Conference 2021 , 2021, pp. 957–968.

 

 
 [64] 
 
Q. Ma, Y. Xu, H. Xu, Z. Jiang, L. Huang, and H. Huang, “Fedsa: A
semi-asynchronous federated learning mechanism in heterogeneous edge
computing,” IEEE Journal on Selected Areas in Communications ,
vol. 39, no. 12, pp. 3654–3672, 2021.

 

 
 [65] 
 
T. Nishio and R. Yonetani, “Client selection for federated learning with
heterogeneous resources in mobile edge,” in ICC 2019-2019 IEEE
international conference on communications (ICC) . IEEE, 2019, pp. 1–7.

 

 
 [66] 
 
Q. Wang, H. Yin, T. Chen, J. Yu, A. Zhou, and X. Zhang, “Fast-adapting and
privacy-preserving federated recommender system,” The VLDB Journal ,
vol. 31, no. 5, pp. 877–896, 2022.

 

 
 [67] 
 
Z. Jie, S. Chen, J. Lai, M. Arif, and Z. He, “Personalized federated
recommendation system with historical parameter clustering,” Journal
of Ambient Intelligence and Humanized Computing , pp. 1–11, 2022.

 

 
 [68] 
 
S. Luo, Y. Xiao, and L. Song, “Personalized federated recommendation via joint
representation learning, user clustering, and model adaptation,” in
 Proceedings of the 31st ACM International Conference on Information and
Knowledge Management . New York, NY,
USA: Association for Computing Machinery, 2022, p. 4289–4293. [Online].
Available: https://doi.org/10.1145/3511808.3557668 

 

 
 [69] 
 
C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast
adaptation of deep networks,” in International conference on machine
learning . PMLR, 2017, pp. 1126–1135.

 

 
 [70] 
 
V. W. Anelli, Y. Deldjoo, T. D. Noia, A. Ferrara, and F. Narducci, “Federank:
User controlled feedback with federated recommender systems,” in
 European Conference on Information Retrieval . Springer, 2021, pp. 32–47.

 

 
 [71] 
 
B. Acun, M. Murphy, X. Wang, J. Nie, C.-J. Wu, and K. Hazelwood,
“Understanding training efficiency of deep learning recommendation models at
scale,” in 2021 IEEE International Symposium on High-Performance
Computer Architecture (HPCA) . IEEE,
2021, pp. 802–814.

 

 
 [72] 
 
J. Qin, B. Liu, and J. Qian, “A novel privacy-preserved recommender system
framework based on federated learning,” in 2021 The 4th International
Conference on Software Engineering and Information Management , 2021, pp.
82–88.

 

 
 [73] 
 
F. K. Khan, A. Flanagan, K. E. Tan, Z. Alamgir, and M. Ammad-Ud-Din, “A
payload optimization method for federated recommender systems,” in
 Fifteenth ACM Conference on Recommender Systems , 2021, pp. 432–442.

 

 
 [74] 
 
Z. Ai, G. Wu, B. Li, Y. Wang, and C. Chen, “Fourier enhanced mlp with adaptive
model pruning for efficient federated recommendation,” in
 International Conference on Knowledge Science, Engineering and
Management . Springer, 2022, pp.
356–368.

 

 
 [75] 
 
J. Konečnỳ, H. B. McMahan, F. X. Yu, P. Richtárik, A. T. Suresh,
and D. Bacon, “Federated learning: Strategies for improving communication
efficiency,” arXiv preprint arXiv:1610.05492 , 2016.

 

 
 [76] 
 
Sijing, Duan, Deyu, Zhang, Yanbo, Wang, Lingxiang, Li, Yaoxue, and Zhang,
“Jointrec: A deep-learning-based joint cloud video recommendation framework
for mobile iot,” IEEE Internet of Things Journal , vol. PP, no. 99,
pp. 1–1, 2019.

 

 
 [77] 
 
K. Muhammad, Q. Wang, D. O’Reilly-Morgan, E. Tragos, B. Smyth, N. Hurley,
J. Geraci, and A. Lawlor, “Fedfast: Going beyond average for faster training
of federated recommender systems,” in Proceedings of the 26th ACM
SIGKDD International Conference on Knowledge Discovery Data Mining , 2020,
pp. 1234–1242.

 

 
 [78] 
 
M. E. Eren, L. E. Richards, M. Bhattarai, R. Yus, C. Nicholas, and B. S.
Alexandrov, “Fedsplit: One-shot federated recommendation system based on
non-negative joint matrix factorization and knowledge distillation,”
 arXiv preprint arXiv:2205.02359 , 2022.

 

 
 [79] 
 
K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel,
D. Ramage, A. Segal, and K. Seth, “Practical secure aggregation for
privacy-preserving machine learning,” in proceedings of the 2017 ACM
SIGSAC Conference on Computer and Communications Security , 2017, pp.
1175–1191.

 

 
 [80] 
 
W. R. Thompson, “On the likelihood that one unknown probability exceeds
another in view of the evidence of two samples,” Biometrika , vol. 25,
no. 3-4, pp. 285–294, 1933.

 

 
 [81] 
 
H. Wang, S. Sievert, S. Liu, Z. Charles, D. Papailiopoulos, and S. Wright,
“Atomo: Communication-efficient learning via atomic sparsification,”
 Advances in Neural Information Processing Systems , vol. 31, 2018.

 

 
 [82] 
 
Y. Gong, L. Liu, M. Yang, and L. Bourdev, “Compressing deep convolutional
networks using vector quantization,” arXiv preprint arXiv:1412.6115 ,
2014.

 

 
 [83] 
 
Y. Lin, S. Han, H. Mao, Y. Wang, and W. J. Dally, “Deep gradient compression:
Reducing the communication bandwidth for distributed training,” arXiv
preprint arXiv:1712.01887 , 2017.

 

 
 [84] 
 
N. Guha, A. Talwalkar, and V. Smith, “One-shot federated learning,”
 arXiv preprint arXiv:1902.11175 , 2019.

 

 
 [85] 
 
A. Kasturi, A. R. Ellore, and C. Hota, “Fusion learning: A one shot federated
learning,” in International Conference on Computational
Science . Springer, 2020, pp.
424–436.

 

 
 [86] 
 
Q. Li, B. He, and D. Song, “Practical one-shot federated learning for
cross-silo setting,” arXiv preprint arXiv:2010.01017 , 2020.

 

 
 [87] 
 
B. Tan, B. Liu, V. Zheng, and Q. Yang, “A federated recommender system for
online services,” in Proceedings of the 14th ACM Conference on
Recommender Systems , ser. RecSys ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 579–581.
[Online]. Available: https://doi.org/10.1145/3383313.3411528 

 

 
 [88] 
 
Y. Song, Y. Xie, H. Zhang, Y. Liang, X. Ye, A. Yang, and Y. Ouyang, “Federated
learning application on telecommunication-joint healthcare recommendation,”
in 2021 IEEE 21st International Conference on Communication Technology
(ICCT) , 2021, pp. 1443–1448.

 

 
 [89] 
 
L.-e. Wang, Y. Wang, Y. Bai, P. Liu, and X. Li, “Poi recommendation with
federated learning and privacy preserving in cross domain recommendation,”
in IEEE INFOCOM 2021 - IEEE Conference on Computer Communications
Workshops (INFOCOM WKSHPS) , 2021, pp. 1–6.

 

 
 [90] 
 
F. M. Harper and J. A. Konstan, “The movielens datasets: History and
context,” ACM Trans. Interact. Intell. Syst. , vol. 5, no. 4, dec
2015. [Online]. Available: https://doi.org/10.1145/2827872 

 

 
 [91] 
 
G. Guo, J. Zhang, and N. Yorke-Smith, “A novel bayesian similarity measure for
recommender systems.” in IJCAI , vol. 13, 2013, pp. 2619–2625.

 

 
 [92] 
 
D. Yang, D. Zhang, and B. Qu, “Participatory cultural mapping based on
collective behavior data in location-based social networks,” ACM
Trans. Intell. Syst. Technol. , vol. 7, no. 3, jan 2016. [Online]. Available:
 https://doi.org/10.1145/2814575 

 

 
 [93] 
 
J. Tang, H. Gao, and H. Liu, “Mtrust: Discerning multi-faceted trust in a
connected world,” in Proceedings of the Fifth ACM International
Conference on Web Search and Data Mining , ser. WSDM ’12. New York, NY, USA: Association for Computing
Machinery, 2012, p. 93–102. [Online]. Available:
 https://doi.org/10.1145/2124295.2124309 

 

 
 [94] 
 
F. Wu, Y. Qiao, J.-H. Chen, C. Wu, T. Qi, J. Lian, D. Liu, X. Xie, J. Gao,
W. Wu et al. , “Mind: A large-scale dataset for news recommendation,”
in Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics , 2020, pp. 3597–3606.

 

 
 [95] 
 
 HetRec ’11: Proceedings of the 2nd International Workshop on Information
Heterogeneity and Fusion in Recommender Systems . New York, NY, USA: Association for Computing Machinery, 2011.

 

 
 [96] 
 
C.-N. Ziegler, S. M. McNee, J. A. Konstan, and G. Lausen, “Improving
recommendation lists through topic diversification,” in Proceedings of
the 14th international conference on World Wide Web , 2005, pp. 22–32.

 

 
 [97] 
 
L. Lyu, J. Yu, K. Nandakumar, Y. Li, X. Ma, J. Jin, H. Yu, and K. S. Ng,
“Towards fair and privacy-preserving federated deep models,” IEEE
Transactions on Parallel and Distributed Systems , vol. 31, no. 11, pp.
2524–2541, 2020.

 

 
 [98] 
 
L. Lyu, Y. Li, K. Nandakumar, J. Yu, and X. Ma, “How to democratise and
protect ai: Fair and differentially private decentralised deep learning,”
 IEEE Transactions on Dependable and Secure Computing , 2020.

 

 
 [99] 
 
Z. Zhang, T. Yang, and Y. Liu, “Sablockfl: a blockchain-based smart agent
system architecture and its application in federated learning,”
 International Journal of Crowd Science , vol. 4, no. 2, pp. 133–147,
2020.

 

 
 [100] 
 
R. Ormándi, I. Hegedűs, and M. Jelasity, “Gossip learning with
linear models on fully distributed data,” Concurrency and Computation:
Practice and Experience , vol. 25, no. 4, pp. 556–571, 2013.

 

 
 [101] 
 
S. Warnat-Herresthal, H. Schultze, K. L. Shastry, S. Manamohan, S. Mukherjee,
V. Garg, R. Sarveswara, K. Händler, P. Pickkers, N. A. Aziz
 et al. , “Swarm learning for decentralized and confidential clinical
machine learning,” Nature , vol. 594, no. 7862, pp. 265–270, 2021.

 

 
 [102] 
 
T. Qi, F. Wu, C. Wu, Y. Huang, and X. Xie, “Uni-FedRec: A unified
privacy-preserving news recommendation framework for model training and
online serving,” in Findings of the Association for Computational
Linguistics: EMNLP 2021 . Punta Cana,
Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp.
1438–1448. [Online]. Available:
 https://aclanthology.org/2021.findings-emnlp.124 

 

 
 [103] 
 
R. Qiu and W. Ji, “An embedded bandit algorithm based on agent evolution for
cold-start problem,” International Journal of Crowd Science , vol. 5,
no. 3, pp. 228–238, 2021.

 

 
 
 
 
 
 
 | 
 
 
 Zehua Sun is currently pursuing his master’s degree in the School of Software of Shandong University. He received his bachelor’s degree in software engineering from the School of Software of Shandong University in 2017. His research interests include federated learning, recommendation systems and data mining. 
 | 

 
 
 
 
 | 
 
 
 Yonghui Xu 
is a professor at Joint SDU-NTU Centre for Artificial Intelligence Research (C-FAIR), Shandong University, and a research fellow in the Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY), Nanyang Technological University, Singapore. He received his Ph.D. from the School of Computer Science and Engineering at South China University of Technology in 2017 and BS from the Department of Mathematics and Information Science Engineering at Henan University of China in 2011. His research areas include various topics in Trustworthy AI, knowledge graphs, expert systems and their applications in e-commerce and healthcare. He has been invited as reviewer of top journals and leading international conferences, such as, TKDE, TNNLS, IEEE Transactions on Cybernetics, Knowledge-Based System, TKDD, IJCAI and AAAI. 
 | 

 
 
 
 
 | 
 
 
 Yong Liu is a Senior Research Scientist at Alibaba-NTU Singapore Joint Research Institute, Nanyang Technological University (NTU). He was a Data Scientist at NTUC Enterprise, and a Research Scientist at Institute for Infocomm Research (I2R), A*STAR, Singapore. He received his Ph.D. degree in Computer Engineering from NTU in 2016 and B.S. degree in Electronic Science and Technology from University of Science and Technology of China (USTC) in 2008. His research interests include recommendation systems, natural language processing, and knowledge graph. He has been invited as a PC member of major conferences such as KDD, SIGIR, ACL, IJCAI, AAAI, and reviewer for IEEE/ACM transactions. 
 | 

 
 
 
 
 | 
 
 
 Wei He is a associate professor at Shandong university. He received bachelor and master degrees from computer science department of shandong university in 1994 and 1999 respectively, and received Ph.d. from engineering of shandong university in 2009. He won the progress first prize in science and technology of shandong province and the progress second prize in science and technology of shandong province, and excellent achievement in computer application. He has published more than 20 papers in the computer journal, journal of software of domestic and international journals conference. More papers were recorded by SCI, EI. 
 | 

 
 
 
 
 | 
 
 
 Lanju Kong is an Associate Professor at Shandong University in Jinan China. she received her
bachelor degree, master degree and PH.D from Shandong university in 1999,2002 and 2011respectively. In 2015,she worked in UCSB as visiting scholar for one year.Her research interests include blockchain consensus, multi-chain architecture, large - scale data management, and so on(klj@sdu.edu.cn). 
 | 

 
 
 
 
 | 
 
 
 Fangzhao Wu is a Principal Researcher at Microsoft Research Asia, President of AAAI2022 and senior member of China Computer Society. He received the Ph.D. and B.S. degrees both from Electronic Engineering Department of Tsinghua University in 2017 and 2012 respectively. He published more than 100 academic papers and was cited nearly 3000 times He has won NLPCC2019 Excellent Paper Award, WSDM 2019 Outstanding PC and AAAI 2021 Best SPC. His research mainly focuses on responsible AI, privacy protection, natural language processing, and recommender systems. The research results have been applied in Microsoft News, Bing Ads and other Microsoft products. 
 | 

 
 
 
 
 | 
 
 
 Yali Jiang is currently a Lecturer in the School of Software, Shandong University. She received her B.Sc., M.Sc. and Ph.D. degrees from Shandong University in 1999, 2002 and 2011, respectively. She is engaged in information security and cryptography research, her main research areas are public key security authentication system and lattice based cryptographic algorithm design and analysis, including cloud computing security, big data privacy protection, IoT security, etc. She has participated in the National 863 Program, Shandong Provincial Excellent Young and Middle-aged Research Award Fund, Shandong Provincial Natural Science Foundation and joint research projects of enterprises. 
 | 

 
 
 
 
 | 
 
 
 LiZhen Cui (IET Fellow, IEEE Senior Member) is the Dean at School of Software, Shandong University. He is the Co-Director of Joint SDU-NTU Centre for Artificial Intelligence Research (C-FAIR) and Research Center of Software Data Engineering, Shandong University. He is the Associate Director of National Engineering Laboratory for E-Commerce Technologies. He is a Professor with the School of Software and the Joint SDU-NTU Centre for Artificial Intelligence Research (C-FAIR), Shandong University, and also a Visiting Professor with Nanyang Technological University, Singapore. He was a Visiting Scholar with Georgia Tech, Atlanta, GA, USA. He received his bachelor’s, M.Sc., and Ph.D. degrees from Shandong University, Jinan, China, in 1999, 2002 and 2005, respectively. He has authored or coauthored over 200 articles in journals and refereed conference proceedings. His research interests include big data management and analysis and AI theory and application. 
 |