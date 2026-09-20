# Rethinking the Starting Point: Collaborative Pre-Training for Federated Downstream Tasks

Yun-Wei Chu $^{1}$ , Dong-Jun Han $^{2}$ , Seyyedali Hosseinalipour $^{3}$ , Christopher G. Brinton $^{1}$

$^{1}$ Purdue University, $^{2}$ Yonsei University, $^{3}$ University at Buffalo-SUNY chu198@purdue.edu, djh@yonsei.ac.kr, alipour@buffalo.edu, cgb@purdue.edu

# Abstract

A few recent studies have shown the benefits of using centrally pre-trained models for initializing federated learning (FL). However, existing pre-training methods do not generalize well when faced with an arbitrary set of downstream FL tasks. Specifically, they often (i) achieve limited average accuracy, particularly when there are unseen downstream labels, and (ii) result in significant accuracy variance, failing to provide a balanced performance across clients. To address these challenges, we propose CoPreFL, a collaborative/distributed pre-training approach which robustly initializes for downstream FL tasks. The key idea of CoPreFL is a model-agnostic meta-learning (MAML) procedure that tailors the global model to closely mimic heterogeneous and unseen FL scenarios, resulting in a pre-trained model that is rapidly adaptable to any FL task. Our MAML procedure integrates performance variance into the meta-objective function, balancing performance across clients rather than solely optimizing for accuracy. Extensive experiments show that CoPreFL significantly enhances both average accuracy and reduces variance in arbitrary downstream FL tasks with unseen/seen labels, outperforming various pre-training baselines. Additionally, CoPreFL proves compatible with different well-known FL algorithms applied by the downstream tasks, boosting performance in each case.

# 1 Introduction

Federated learning (FL) has gained prominence as a distributed machine learning framework, enabling collaborative training among clients by periodic aggregations of local models on a server (McMahan et al. 2017; Konecný et al. 2016). Recent research has extensively explored various aspects of FL, such as aggregation schemes (Ji et al. 2019; Wang et al. 2020) or local training techniques (Reddi et al. 2021; Sahu et al. 2018). One aspect that remains understudied, however, is the impact of model initialization in FL. While pre-training boosts performance in centralized AI/ML (Radford et al. 2019; Devlin et al. 2019; Dosovitskiy et al. 2021), most FL works still rely on random weight initialization instead of well pre-trained models.

Motivation. Recently, a few works (Nguyen et al. 2023; Chen et al. 2023) have shown that initializing FL with centrally pre-trained models can enhance the average performance across clients. Yet, existing centralized pre-training methods face significant challenges, particularly when handling newly emerging and/or heterogeneous downstream FL tasks unanticipated during pre-training. These include: (i) limited average accuracy (despite outperforming random initialization), due to newly encountered data and labels, and (ii) large performance variance, resulting in unbalanced accuracy across clients. The histograms in Figure 1 show the performance of various pre-trained models in multiple downstream image classification tasks (see Section 4 for details), illustrating these limitations. While using centrally pre-trained models improves average accuracy over random initialization, it introduces significant performance variance across clients, a well-cited concern in distributed AI/ML (Li et al. 2020; Cho et al. 2022). Additionally, the achievable average accuracy of centralized pre-training remains suboptimal, indicating that such models struggle to mimic data heterogeneity and diversity in downstream FL tasks.

Goals. Motivated by these limitations, we aim to develop a robust FL pre-training methodology that provides an initialization which achieves two main objectives: (i) improved average accuracy, and (ii) reduced performance variance for balanced accuracy across clients, in each downstream FL task. This is challenging as it must work across any arbitrary set of downstream FL tasks, which may include data and labels previously unseen due to factors like time-varying environments or new clients joining the system. Thus, the pretrained model must handle unfamiliar classes and data heterogeneity during downstream FL tasks, a challenge overlooked by existing methods (Nguyen et al. 2023; Chen et al. 2023). We summarize our research question as follows:

How can we design a pre-training strategy that can simultaneously (i) enhance average accuracy and (ii) reduce performance variance across clients, for an arbitrary set of downstream FL tasks which possess heterogeneity in their data statistics as well as unseen labels?

Contributions. We propose CoPreFL, a Collaborative Pre-training approach for handling an arbitrary set of downstream FL tasks, to address the above question. We make the following key contributions:

\- Distributed pre-training infused with meta-learning: The core of CoPreFL is an FL-inspired pre-training procedure which employs model-agnostic meta-learning (MAML)-based updates on the collaboratively-bulit global model. Through our developed MAML procedure, CoPreFL ensures robust initializations, enabling the pre-trained model to easily adapt to

Objectives of CoPreFL Pre-Trained Model $\Phi^{*}$ targeting downstream FL

- Conduct distributed pre-training to mimic downstream FL   
• Balance performance across clients   
• Meta-Learning for unseen scenario

![](images/5ded375a050828af513f71221851001a613bad571f44f3fbfd2a8cfdfa159eb4.jpg)

![](images/aa769446a4626c64bc26547ad47fed7de48e420cf8028eaf4a0a39649f7ef68e.jpg)

![](images/7b95837d71879ca37f34f833a78159edccf879c530639ec71eee2622496ba50e.jpg)

Addressing downstream FL challenges via initialization   
![](images/cabd5656c82ae6b79eb81e46e7e6d8291433894037b4e5ff1cddac87202dd971.jpg)  
Robust initialization for any arbitrary set of FL tasks with possible new classes

Downstream FL Task   
![](images/cefa788ef03be6b3d34135c920c51fc6f7ec20b468562aab436682412eac5409.jpg)

Downstream FL Task 2   
![](images/586409f68d93fdd4695319a51a63aa8dc2a97c4bc6ee0e346f07b748dbc44191.jpg)

Downstream FL Task X   
![](images/034eb408aeb72e85c4a673d80c09f1714f73daabc3d02c1e54b09038594b7e3c.jpg)

![](images/adceb87e0b54772627c0f12d661452eca4fc1337512bae1f678c79f289c91b38.jpg)

![](images/b66e5d9115bd8dbebcafccdb5265b512cb7aeeb09af0322dd128403378ff3d77.jpg)

![](images/2e622702ecfdf5d9026a3f15ca7858514e4e4ed6b89ad1b888893fdc00193292.jpg)

the higher the better   
![](images/03738b02fba68b5f35168f08ee54efdd008364f2d14db502613a49a3ed307752.jpg)

the lower the better   
![](images/4345c39b953979fa89050c57bbce00eb828abf58708dd1b31a240ae2e9da8f72.jpg)  
Downstream FL initialized by different pre-trained methods:
Random Centralized FedAvg CoPreFL (Ours)   
Figure 1: (Left): Overview of CoPreFL, aiming to provide a robust initialization for an arbitrary set of downstream FL tasks. (Right): Average accuracy and variance achieved by FL tasks (from Section 4) initialized by various pre-trained models. Centralized pre-training achieves limited performance as it is not able to capture the heterogeneous characteristics of unforeseen FL settings. CoPreFL demonstrates improved performance in terms of both average accuracy and variance by strategically mimicking downstream FL scenarios during pre-training.

unseen labels and various data distributions in any downstream FL tasks. This approach differs in purpose and method from prior works that use meta-learning for personalization in FL (Chen et al. 2018; Jiang et al. 2019; Fallah, Mokhtari, and Ozdaglar 2020): since our downstream tasks aim to construct a global model rather than client-specific personalized models, we conduct meta-updates based on the global model instead of directly using local models.

- Meta-objective function incorporating variance: To enhance average accuracy while improving performance balance among clients in the downstream FL tasks, we explicitly incorporate both expected loss and performance variance into the meta-objective function during pre-training in CoPreFL. In doing so, we introduce a first-order approximation for efficiently computing the gradient of the proposed meta-objective function.   
- Relaxing the assumption of centrally stored pre-training data: CoPreFL relaxes the assumption made by existing works that all pre-training data is stored centrally. We develop our pre-training algorithm for various hybrid client-server data storage settings, where (i) data is exclusively held by distributed clients, or (ii) the server also maintains partial data. This is crucial for distributed/FL applications with data privacy limitations. Our approach is applicable even when all pre-training data is centrally stored, as confirmed by our experiments.   
- Extensive experiments across various downstream FL tasks and settings: We compare CoPreFL against various baselines across diverse downstream FL tasks with different data distributions, seen/unseen labels, and client-server data allocations. Results show notable improvements in accuracy and performance variance when downstream tasks are initialized with CoPreFL. We also show CoPreFL's compatibility with various popular FL algorithms used downstream and its resilience to distributional shifts.

Our work is among the first to consider FL in both the pre-training and downstream stages of distributed learning tasks. We introduce several unique features tailored to FL, including meta-updating the global model during distributed pre-training, hybrid client-server learning, and balancing between average performance and variance across the clients.

# 2 Related Work

Pre-training for FL. While pre-training has been extensively studied in centralized AI/ML applications (Radford et al. 2019; Brown et al. 2020; Devlin et al. 2019; Dosovitskiy et al. 2021), its effects on downstream FL tasks remains less explored. A few recent works have shown that starting FL with centrally pre-trained models can improve performance over random initializations (Nguyen et al. 2023; Chen et al. 2023). However, as observed in Figure 1, such initialization strategies often lead to large performance variance and even limited average accuracy, since they are not able to mimic multiple downstream FL settings. To address this, we develop a MAML-based pre-training strategy, CoPreFL, tailored for distributed downstream settings, so that initializing FL with this pre-trained model can improve both average accuracy and performance variance while addressing the challenges of heterogeneous/unseen data encountered in downstream FL tasks.

Meta-learning in FL. CoPreFL utilizes meta-learning to construct a global model adaptable to any downstream FL tasks, optimizing both performance and variance. This distinguishes it from other meta-learning-based FL, like personalized FL (Chen et al. 2018; Jiang et al. 2019; Fallah, Mokhtari, and Ozdaglar 2020; Chu et al. 2022) and few-round FL (Park et al. 2021). Specifically, Park et al. (2021) uses meta-learning-based episodic training for quick FL adaptation, but it overlooks performance imbalance across clients and fails to consider scenarios where the server holds a proxy dataset for hybrid pre-training. In personalized FL, meta-update target individual client model performance, aiming for better personalization to clients in downstream tasks. In contrast, our focus is on developing a pretrained model that ensures high accuracy and fairness for global models in downstream FL tasks.

Performance imbalance in FL. Several works in FL address performance imbalance (Mohri, Sivek, and Suresh 2019; Li et al. 2020; Cho et al. 2022) typically by creating a global model that satisfies as many clients as possible (e.g., achieving a uniform accuracy across clients). Such models are more likely to satisfy new clients joining the FL system. These methods can be applied downstream from our pre-training methodology, whose primary objective is to build a robust initial model that will lead to higher average and more balanced performance across clients after FL training.

# 3 Proposed CoPreFL Methodology

# 3.1 Problem Setup and Pre-Training Objectives

Federated downstream tasks. Referring to Figure 1, CoPreFL aims to provide a robust initialization for any downstream FL tasks. In each of these downstream tasks, we assume a central server is connected to a set of clients G. Starting from the initialized model $w^{0}$ , each FL task iterates between (i) local training at the clients and (ii) global aggregations at the server, across multiple communication rounds. In each downstream round r, every client $g \in G$ downloads the previous global model $w^{r-1}$ from the server and subsequently updates it through multiple iterations of stochastic gradient descent (SGD) using its local dataset, denoted $D_{g}$ . After completing their local updates, clients upload their updated models, denoted $w_{g}^{r}$ , to the server for aggregation. This aggregation results in a new global model $w^{r} = \sum_{g \in G} \frac{|D_{g}|}{|D|} w_{g}^{r}$ assuming FedAvg (McMahan et al. 2017) is employed, where $|D|$ is the total data samples across all clients. This entire process is iterated over $r = 1, ..., R$ communication rounds for each task.

Pre-training scenarios. One of our contributions is relaxing the assumption that all pre-training data is stored centrally. To this end, we consider two distributed pre-training scenarios for CoPreFL:

- Scenario I: Pre-training datasets are exclusively available at distributed clients.   
- Scenario II: A hybrid scenario where the server also holds a small amount of pre-training data.

Scenario I simulates downstream FL tasks where pre-training labels and data may differ from those in downstream tasks. Scenario II represents settings where the server holds data reflecting the broader population distribution (e.g., a self-driving car manufacturer with database of roadway images). Such hybrid FL settings that combine client data with a relatively small portion of server data are becoming popular (Yang, Chen, and Shen 2023; Bian et al. 2023), but remain underexplored for pre-training. Further, as we will discuss in Remark 2, our method is still applicable even when all pre-training data is centralized.

Pre-training objectives. Our goal is to design a pre-trained model $\Phi^{*}$ that serves as a robust starting point $w^{0} = \Phi^{*}$ for any arbitrary downstream FL task. More precisely, letting G represents possible client sets comprising a downstream FL task, the goal of designing $\Phi^{*}$ is to minimize the following objective function:

$$
A (\Phi) = \mathbb {E} _ {G \sim p (\mathcal {G})} \left[ \frac {1}{| G |} \sum_ {g \in G} f (w ^ {R} (\Phi , G), D _ {g}) \right], \tag {1}
$$

where $p(\mathcal{G})$ represents the probability distribution over G, G is a specific client group (i.e., a specific task) drawn from $p(\mathcal{G})$ , $f(\cdot)$ is the per-client loss function for downstream training, $w^{R}(\Phi, G)$ symbolizes the final R-th round global model derived from client set G when initialized by $\Phi$ , and $D_{g}$ represents the local dataset of client g. $A(\Phi)$ denotes the average FL performance across all clients fir downstream tasks, with each group weighted by likelihood of occurrence.

On the other hand, FL settings can lead to significant performance variations among clients, especially when the aggregated models are biased towards those with larger datasets. This performance variation can be measured by the variance in testing accuracy across participants (Li et al. 2020). Thus, besides improving performance for any FL task, we aim for the final global model $w^{R}(\Phi^{*}, G)$ initialized from our pre-trained model $\Phi^{*}$ to achieve balanced testing performance across client set G. Specifically, our second objective for $\Phi^{*}$ is to minimize the variance of the loss distribution across participants in downstream FL tasks, i.e.,

$$
\begin{array}{l} F (\Phi) = \mathbb {E} _ {G \sim p (\mathcal {G})} \left[ \frac {1}{| G |} \sum_ {g \in G} f ^ {2} (w ^ {R} (\Phi^ {*}, G), D _ {g}) \right. \\ \left. - \left(\frac {1}{| G |} \sum_ {g \in G} f \left(w ^ {R} \left(\Phi^ {*}, G\right), D _ {g}\right)\right) ^ {2} \right]. \tag {2} \\ \end{array}
$$

Overview of approach. One of our key contributions is balancing (1) and (2). The challenge arises as $D_{g}$ , G, and $p(\mathcal{G})$ are unknown during pre-training, preventing us from directly optimizing $A(\Phi)$ and $F(\Phi)$ . To address this, we develop a model-agnostic meta-learning (MAML) approach in CoPreFL to mimic statistical heterogeneity of downstream FL tasks. This method yields pre-trained models that offer robust initialization for unseen downstream tasks, considering (1) and (2). Detailed in Sections 3.2 and 3.3, we construct a pre-training environment for scenarios I and II that mirrors downstream federated setups, enabling the pretrained model to handle data heterogeneity across clients and tasks. Our meta-learning-based CoPreFL updates the pretrained model iteratively over federated rounds using a support set, with a concluding adjustment (meta-update) using a query set treated as unseen knowledge. This enables our pre-trained model to the effectively handle unforeseen FL scenarios downstream while balancing between (1) and (2).

# 3.2 CoPreFL in Scenario I (Pre-training with Distributed Clients)

We first consider the scenario where pre-training data is distributed across M distributed clients, and no data is stored on the server. The detailed procedure of CoPreFL for this case is given in Algorithm 1. To start, in each round $t = 1, \ldots, T$ of pre-training, a set of clients $m \subset M$ is randomly selected to participate in the current round. Further, each participating client $j \in m$ splits its local pre-training dataset $D_{j}^{p}$ into a support set $S_{j}$ and query set $Q_{j}$ , which are disjoint. These steps mimic downstream task variations within and across each pre-training round (i.e., by changing the participating clients across rounds, and holding out the query sets in each round), allowing our meta-learning to improve generalization in unseen downstream scenarios.

Temporary pre-training model construction. In each round $t$ , participating clients $j \in m$ download $\Phi^{t-1}$ from the server. Subsequently, clients perform local training using their support sets $S_j$ , yielding a local support loss $\ell_{S_j}^e(\Phi^t)$ per epoch $e$ , defined in line 8 of Algorithm 1, where $\ell(\cdot)$ is the per-datum loss function (e.g., cross-entropy for classification). After all participants finish $E$ epochs, we obtain the updated local model $\Phi_j^{t,E}$ . Clients then send their updated models to the server for aggregation, resulting in $\overline{\Phi^t}$ (defined in line 12). This model can be viewed as the temporary

Algorithm 1: Our Pre-training Method CoPreFL (Pre-training Phase in Scenario I)   
1: Input: A set of clients M in the pre-training phase, with each client i holding its pre-training dataset $D_{i}^{p}$ .
2: for Each pre-training round $t = 1, 2, ..., T$ do
3: Randomly select a set of clients $m \subset M$ to participate
4: Each participant $j \in m$ partitions its own dataset $D_{j}^{p}$ into support set $S_{j}$ and query set $Q_{j}$ 5: for Each participant j in parallel do
6: Download $\Phi^{t-1}$ from the server
7: for local epoch $e = 1, 2, ..., E$ do
8: $\ell_{S_{j}}^{e}(\Phi^{t}) \leftarrow \frac{1}{|S_{j}|} \sum_{(x,y) \in S_{j}} \ell(\Phi_{j}^{t,e}(x), y)$ { Compute local support loss at each epoch}
9: $\Phi_{j}^{t,e} \leftarrow \Phi_{j}^{t,e-1} - \eta \nabla \ell_{S_{j}}^{e}(\Phi^{t})$ {Perform SGD local update using support loss}
10: end for
11: end for
12: $\overline{\Phi^{t}} \leftarrow \sum_{j \in m} \frac{|S_{j}|}{\sum_{i \in m} |S_{i}|} \Phi_{j}^{t,E}$ {Model aggregation to construct temporary global model}
13: for Each participant j in parallel do
14: Download $\overline{\Phi^{t}}$ from the server
15: $\mathcal{L}_{Q_{j}}(\overline{\Phi^{t}}) \leftarrow \frac{1}{|Q_{j}|} \sum_{(x,y) \in Q_{j}} \ell(\overline{\Phi^{t}}(x), y)$ {Compute local loss (and gradient) using query set $Q_{j}$ }
16: end for
17: Server computes overall meta-loss $\mathcal{L}_{Q}(\overline{\Phi^{t}})$ and variance across meta-losses $\sigma_{Q}^{2}(\overline{\Phi^{t}})$ according to (3)
18: $\mathcal{L}_{meta}(\overline{\Phi^{t}}) = \gamma \mathcal{L}_{Q}(\overline{\Phi^{t}}) + (1 - \gamma) \sigma_{Q}^{2}(\overline{\Phi^{t}})$ {Customized query meta-loss}
19: $\Phi^{t} \leftarrow \overline{\Phi^{t}} - \zeta \nabla \mathcal{L}_{meta}(\overline{\Phi^{t}})$ {Meta-learning model update using customized loss}
20: end for
21: Output: A pre-trained model for downstream FL tasks: $\Phi^{T}$

pre-training model that will be further refined using query sets, with the objective of obtaining robust global models at the conclusion of downstream task training.

Measuring average performance and variance. Next, the query sets are used to evaluate the performance of the temporary pre-training model on each client, mimicking the scenario where the pre-trained model encounters unseen data, and to conduct meta-updates to promote downstream generalization. CoPreFL aims to strike a balance between the following objectives during pre-training:

$$
\begin{array}{l} \mathcal {L} _ {Q} (\overline {{\Phi^ {t}}}) = \sum_ {j \in m} \mathcal {L} _ {Q _ {j}} (\overline {{\Phi^ {t}}}) \text {   and   } \\ \sigma_ {Q} ^ {2} \left(\overline {{{{\Phi^ {t}}}}}\right) = \frac {1}{| m |} \sum_ {j \in m} \left(\mathcal {L} _ {Q _ {j}} \left(\overline {{{{\Phi^ {t}}}}}\right) - \frac {1}{| m |} \mathcal {L} _ {Q} \left(\overline {{{{\Phi^ {t}}}}}\right)\right) ^ {2}, \tag {3} \\ \end{array}
$$

where $L_{Q_{j}}$ represents the loss evaluated using query sets $Q_{j}$ of participants, $L_{Q}$ denotes the overall query loss (characterized by aggregating $L_{Q_{j}}$ across all participants), and $\sigma_{Q}^{2}$ represents the performance variance evaluated using clients' query losses. To balance the performance-variance trade-off, we construct a customized query meta-loss function $\mathcal{L}_{meta}(\overline{\Phi^{t}})$ to minimize both the overall query loss $\mathcal{L}_{Q}(\overline{\Phi^{t}})$ when encountering unseen data and the variance $\sigma_{Q}^{2}(\overline{\Phi^{t}})$ of query losses across participants. Formally, we aim to solve:

$$
\min _ {\Phi} \mathcal {L} _ {m e t a} (\overline {{\Phi^ {t}}}) = \min _ {\Phi} \left[ \gamma \mathcal {L} _ {Q} (\overline {{\Phi^ {t}}}) + (1 - \gamma) \sigma_ {Q} ^ {2} (\overline {{\Phi^ {t}}}) \right], \tag {4}
$$

where $\gamma \in [0,1]$ represents a balancer between the average performance and variance. Setting $\gamma = 0$ encourages a more uniform accuracy distribution, aligning with $\sigma_Q^2$ , but may sacrifice average performance. A larger $\gamma$ emphasizes the average performance with less consideration for uniformity, optimizing the pre-trained model more towards $\mathcal{L}_O$ .

Model-agnostic meta update. Considering the objective function in (4), each participant $j$ downloads the temporary global model $\overline{\Phi^t}$ and employs its query set $Q_j$ to compute its local query loss $\mathcal{L}_{Q_j}(\overline{\Phi^t})$ , as in line 15 in Algorithm 1. The gradients are also computed locally and sent back to the server, as both are necessary to conduct the meta-update. On the server-side, the overall query meta-loss $\mathcal{L}_Q(\overline{\Phi^t})$ and the performance variance $\sigma_Q^2 (\overline{\Phi^t})$ are computed, according to (3). Then, CoPreFL updates the temporary pre-training model $\overline{\Phi^t}$ through a gradient step with the customized query meta-loss $\mathcal{L}_{meta}$ and the aggregated received gradients, to align it with (4). To derive the meta-loss $\nabla_{\Phi^{t - 1}}\mathcal{L}_{meta}(\overline{\Phi^t})$ , we express it through the chain rule as $\nabla_{\overline{\Phi^t}}\mathcal{L}_{meta}(\overline{\Phi^t})\times \frac{\partial\overline{\Phi^t}}{\partial\Phi^{t - 1}}$ . Writing $\overline{\Phi^t} = \sum_{j\in m}\frac{|S_j|}{\sum_{i\in m}|S_i|}\Phi_j^{t,E} = \sum_{j\in m}\frac{|S_j|}{\sum_{i\in m}|S_i|} (\Phi^{t,E - 1} - \eta \nabla \ell_{S_j}^E (\Phi^t))$ , it follows that

$$
\begin{array}{l} \nabla_ {\Phi^ {t - 1}} \mathcal {L} _ {m e t a} (\overline {{\Phi^ {t}}}) = \nabla_ {\overline {{\Phi^ {t}}}} \mathcal {L} _ {m e t a} (\overline {{\Phi^ {t}}}) \times \\ \left(1 - \eta \sum_ {j \in m} \frac {\left| S _ {j} \right|}{\sum_ {i \in m} \left| S _ {i} \right|} \frac {\partial}{\partial \Phi^ {t - 1}} \nabla \ell_ {S _ {j}} ^ {E} (\Phi^ {t})\right). \tag {5} \\ \end{array}
$$

If we ignore the second derivative term, the meta-loss gradient can be approximated as $\nabla_{\overline{\Phi^{t}}}\mathcal{L}_{meta}(\overline{\Phi^{t}})$ . This is similar to making a first-order approximation to a meta-update, a common practice in the implementation of MAML variants to reduce complexity (Finn, Abbeel, and Levine 2017).

The server then sends the meta-updated global model $\Phi^{t}$ to a new set of participants to begin the next round of pre-training. After T rounds, the final global model $\Phi^{T}$ serves as the pre-trained model for initializing FL in the downstream tasks, i.e., in Figure 1, clients in any downstream task conduct FL starting from the pre-trained model $w^{0} = \Phi^{T}$ .

Remark 1 (Key characteristics of CoPreFL meta-update). Using the query datasets, CoPreFL applies a meta-update to the temporary pre-training model—a global model developed through FL on the support datasets. This differs significantly from existing meta-learning based FL methods, which update client models for personalization, as discussed in Section 2. Our method aims to tailor the pre-training model to adapt to any downstream FL tasks, addressing robustness against unseen and heterogeneous data, unlike existing personalization methods. As we will see in Section 4, this leads to notable improvements of CoPreFL over employing these prior methods for pre-training.

# 3.3 CoPreFL in Scenario II (Hybrid Client-Server Pre-Training)

We next explore a pre-training scenario where the server holds a small dataset $D^{s}$ drawn from the broader population distribution, alongside client-held data. Unlike scenario I, where client data was split into support and query sets, in

scenario II, all client samples are used as support data, while the server's data serves as the query set.

The procedure of CoPreFL for scenario II is detailed in Algorithm 2 in Appendix B. Here, we highlight the key differences from Algorithm 1. First, the temporary global model $\overline{\Phi^t}$ is aggregated from local models, each trained on a participant's full local dataset $D_j^p$ . Second, the meta-update of the temporary global model $\overline{\Phi^t}$ utilizes the server's data. To help mimic the distributed nature of downstream FL tasks, we randomly split the dataset $D^s$ into $|m|$ equally to derive the average loss and variance objectives similar to (3). The temporary global model $\overline{\Phi^t}$ is then updated based on meta-loss $\mathcal{L}_{meta}(\overline{\Phi^t})$ , calculated similarly to (4) but through meta-updates using the server's partitioned data.

Note that unequal and/or non-uniformly random partitionings of the server-side dataset for meta-updating could be considered as alternatives. However, due to the server's lack of prior knowledge about future downstream FL tasks during pre-training, including their dataset sizes and distributions, meta-updating the model with randomly allocated query sets is the most viable solution. We show in Section 4 that this partitioning provides significant performance improvements over other pre-training strategies.

Remark 2 (Applications to centralized datasets). Although we present CoPreFL for two distributed scenarios, it is applicable even when all pre-training data is stored at the server (e.g., public datasets). The server can intentionally split the dataset to mimic scenarios I or II and directly apply CoPreFL. We will show in Section 4 that CoPreFL surpasses standard centralized pre-training even in this setup, offering initializations better prepared for downstream data heterogeneity in FL setups.

Remark 3. The theoretical link between pre-training strategies and downstream task performance remains an open problem. This challenge has existed both in centralized-to-centralized (Chang et al. 2020; Dong et al. 2023; Zhang et al. 2022; Hu et al. 2019; Yuan et al. 2024) and centralized-to-federated (Nguyen et al. 2023; Chen et al. 2023) transfers from pre-training to downstream, and persists in the distributed-to-federated case we consider here. We thus leave theoretical analysis of CoPreFL to future work, and instead validate its effectiveness through extensive experiments. The success of CoPreFL can largely be credited to advantage from meta-learning, which enhances robustness across diverse sets of downstream tasks.

# 4 Experiments

# 4.1 Experimental Setup

Datasets and model. For evaluation, we use CIFAR-100 (Krizhevsky 2009), Tiny-ImageNet (Le and Yang 2015), FEMNIST (Caldas et al. 2018), and PACS (Li et al. 2017), following data splits provided in (Park et al. 2021), and adopt ResNet-18 (He et al. 2015). To model scenarios where downstream task labels are unknown during pre-training, we divide CIFAR-100 into 80 classes for pre-training and 20 for downstream tasks, and Tiny-ImageNet into 160 and 40 classes, respectively. We also explore mixed scenarios involving overlapping classes between pre-training and downstream tasks. Following (Yang, Chen, and Shen 2023; Zhang et al. 2020), we allocate 95% of the samples from the pre-training dataset for clients, and the remaining 5% form the server dataset. For PACS, we adopt a one-domain-leave-out setup (Li et al. 2022; Zhou et al. 2021) using different data domains for pre-training and downstream tasks. Detailed dataset information is available in Appendix C.1.

Pre-training phase. We distribute the pre-training dataset to $|M| = 100$ clients following non-IID data partitions according to a Dirichlet distribution (Morafah et al. 2022; Li, He, and Song 2021), and select $|m| = 20$ participants out of the $|M|$ clients for each FL round. Results with different $|m|$ and IID setups are reported throughout Appendix D. We adopt a standard approach commonly used in meta-learning-based research (Jamal et al. 2020; Shu et al. 2019; Park et al. 2021) for support/query splitting, where we randomly partition each client's data into $80\%$ support and $20\%$ query sets. See Appendix C.1-C.2 for more details.

Downstream FL task and evaluation metrics. To generate each downstream FL task, we randomly select 5 of the 20 classes from the CIFAR-100 dataset and 40 classes from the Tiny-ImageNet dataset, and distribute the corresponding data samples to a set of $|G| = 10$ clients following non-IID Dirichlet data distributions (see Appendix D for IID results). Each participant in the downstream phase utilizes 80% of its local data as training samples, while the remaining 20% is reserved for testing samples. We keep the training procedure consistent for each downstream task (see Appendix C.2 for detailed settings). For each task, we evaluate the final global model using test samples from each client $g \in G$ , reporting the accuracy and variance of the accuracy distribution across the clients. We consider a total of X = 10 downstream tasks, and the evaluation metrics are reported as averages (with standard deviations) across the tasks.

Data distribution. Data samples are distributed to $|M| = 100$ clients for pre-training and $|G| = 10$ clients for downstream FL tasks using the corresponding dataset based on a Dirichlet( $\alpha$ ) distribution with $\alpha = 0.5$ , as done in the literature (Morafah et al. 2022; Li, He, and Song 2021).

Baselines for pre-training. We compare CoPreFL with several established FL algorithms, including (i) standard FedAvg (McMahan et al. 2017), (ii) FedMeta (Chen et al. 2018), which employs meta-learning for unseen scenarios, and (iii) q-FFL(q > 0) (Li et al. 2020), designed to balance performance across clients. When applying these baselines in scenario II, in each pre-training round t, after the global model $\Phi^{t}$ has been constructed, we further train with 5 additional iterations on the server dataset. This extended training follows the approach in (Yang, Chen, and Shen 2023; Bian et al. 2023), where the server's data is used to further refine the global model. Similarly, we introduce a baseline called CoPreFL-SGD, which first constructs a global model according to CoPreFL and then further performs SGD iterations using server data on the global model. Finally, we also consider initializations based on (iv) conventional centralized pre-training (Nguyen et al. 2023), and popular FL algorithms like (v) SCAFFOLD (Karimireddy et al. 2019), (vi) FedDyn (Acar et al. 2021), and (vii) PerFedAvg (Fallah, Mokhtari, and Ozdaglar 2020) (see Table 3a).

<table><tr><td>Pre-training</td><td colspan="4">Downstream: Non-IID FedAvg (CIFAR-100)</td><td colspan="4">Downstream: Non-IID FedAvg (Tiny-ImageNet)</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td></tr><tr><td>FedAvg</td><td> $78.96 \pm 2.98$ </td><td> $64.80 \pm 3.01$ </td><td> $62.70 \pm 3.35$ </td><td> $67.00 \pm 2.95$ </td><td> $82.94 \pm 2.59$ </td><td> $37.21 \pm 2.81$ </td><td> $68.99 \pm 2.43$ </td><td> $72.29 \pm 2.61$ </td></tr><tr><td>FedMeta</td><td> $82.45 \pm 3.07$ </td><td> $48.72 \pm 2.84$ </td><td> $68.97 \pm 3.04$ </td><td> $72.41 \pm 3.06$ </td><td> $81.03 \pm 2.86$ </td><td> $37.58 \pm 3.00$ </td><td> $69.44 \pm 2.61$ </td><td> $71.55 \pm 2.93$ </td></tr><tr><td>q-FFL</td><td> $80.01 \pm 2.67$ </td><td> $88.92 \pm 3.31$ </td><td> $64.39 \pm 2.95$ </td><td> $67.48 \pm 2.67$ </td><td> $84.11 \pm 2.49$ </td><td> $43.96 \pm 2.71$ </td><td> $73.87 \pm 2.79$ </td><td> $76.05 \pm 2.61$ </td></tr><tr><td>CoPreFL</td><td> $83.29 \pm 2.61$ </td><td> $34.69 \pm 3.17$ </td><td> $71.58 \pm 3.00$ </td><td> $73.20 \pm 2.98$ </td><td> $85.23 \pm 2.43$ </td><td> $35.40 \pm 2.75$ </td><td> $76.77 \pm 2.58$ </td><td> $78.46 \pm 2.47$ </td></tr></table>

(a) Results in scenario I using CIFAR-100 and Tiny-ImageNet datasets. 

<table><tr><td>Pre-training</td><td colspan="4">Downstream: Non-IID FedAvg (CIFAR-100)</td><td colspan="4">Downstream: Non-IID FedAvg (Tiny-ImageNet)</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td></tr><tr><td>FedAvg</td><td>82.82 ± 3.17</td><td>49.00 ± 3.41</td><td>69.71 ± 3.25</td><td>72.54 ± 3.30</td><td>82.87 ± 3.19</td><td>48.16 ± 2.94</td><td>68.94 ± 3.38</td><td>72.91 ± 3.49</td></tr><tr><td>FedMeta</td><td>82.69 ± 3.05</td><td>48.44 ± 2.99</td><td>68.84 ± 3.14</td><td>71.82 ± 3.27</td><td>84.19 ± 2.93</td><td>49.70 ± 2.74</td><td>70.41 ± 3.16</td><td>72.63 ± 3.00</td></tr><tr><td>q-FFL</td><td>82.14 ± 2.76</td><td>73.10 ± 3.08</td><td>68.22 ± 3.00</td><td>70.64 ± 2.85</td><td>83.51 ± 3.05</td><td>44.22 ± 3.22</td><td>69.91 ± 2.94</td><td>73.71 ± 3.14</td></tr><tr><td>CoPreFL-SGD</td><td>83.63 ± 3.00</td><td>41.73 ± 2.85</td><td>69.76 ± 2.94</td><td>73.46 ± 3.09</td><td>84.30 ± 2.77</td><td>36.24 ± 3.04</td><td>72.83 ± 2.99</td><td>75.64 ± 3.18</td></tr><tr><td>CoPreFL</td><td>86.63 ± 2.93</td><td>31.58 ± 2.64</td><td>73.05 ± 2.51</td><td>75.82 ± 2.88</td><td>84.72 ± 2.51</td><td>24.80 ± 3.00</td><td>75.84 ± 2.87</td><td>77.31 ± 3.13</td></tr></table>

(b) Results in scenario II using CIFAR-100 and Tiny-ImageNet datasets.

Table 1: Performance on 10 non-IID downstream FL tasks, initialized with various non-IID FL pre-training methods. Lowest X% shows the average accuracy of clients with the lowest X% accuracy. CoPreFL provides the best initialization across scenario, metric, and dataset.   
![](images/8be5e148fd671805b9fcedbaef6fdcbd3fa73b88632852d0ea622a49891dbd7c.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedMeta(Acc+Var runner-up) | CoPreFL |
| --------------------- | --------------------------- | ------- |
| 50                    | 1                           | 0       |
| 60                    | 1                           | 1       |
| 70                    | 4                           | 2       |
| 80                    | 13                          | 12      |
| 90                    | 12                          | 11      |
| 100                   | 0                           | 0       |
</details>

(a) Dataset: CIFAR-100; Scenario: I

![](images/40ef40dbcd54321c0c492fad440b3ec6e710a2ecd76b6477ad8643a48302b343.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 60                   | 0                      | 0                       | 0       |
| 70                   | 5                      | 3                       | 0       |
| 80                   | 10                     | 12                      | 10      |
| 90                   | 15                     | 14                      | 20      |
| 100                  | 0                      | 0                       | 0       |
</details>

(b) Dataset: CIFAR-100; Scenario: II

![](images/2c65b8c1e4b943708bf8dcd781980de82828b22f5806fc72096be95871b5dc2f.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Var runner-up) | q-FFL(Acc runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 50                   | 1                      | 0                    | 0       |
| 60                   | 2                      | 0                    | 0       |
| 70                   | 3                      | 5                    | 0       |
| 80                   | 5                      | 10                   | 5       |
| 90                   | 15                     | 12                   | 15      |
| 100                  | 2                      | 0                    | 0       |
</details>

(c) Dataset: Tiny-ImageNet; Scenario: I

![](images/64971bf07fd623e02d117829adcbee47b192a1b6c55f79d1923d2ffa23a036f5.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta(Acc runner-up) | q-FFL(Var runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 50                   | 1                      | 0                    | 0       |
| 60                   | 2                      | 1                    | 0       |
| 70                   | 3                      | 2                    | 1       |
| 80                   | 4                      | 3                    | 2       |
| 90                   | 5                      | 4                    | 3       |
| 100                  | 6                      | 5                    | 4       |
</details>

(d) Dataset: Tiny-ImageNet; Scenario: II   
Figure 2: Testing accuracy distributions in various non-IID FL tasks. CoPreFL achieves the best average accuracy (i.e., right-leaning distribution) and smaller performance variance (i.e., narrower distribution) while also improving worst-performing clients' accuracies.

# 4.2 Experimental Results

Main results for scenarios I & II. Tables 1a and 1b present the test accuracies for CoPreFL across scenarios I and II on the CIFAR-100 and Tiny-ImageNet datasets. In scenario I, CoPreFL shows robust initializations for downstream FL tasks, obtaining a higher average accuracy and reduced performance variance across clients, and significantly improving accuracies for the worst-performing clients (Lowest 10-20% metrics). This highlights the benefits of balancing the competing objectives in (3) during meta-updates. See Appendix D.1 for supplemental analyses and results, including varying the number of participants in pre-training, and different data distributions in downstream FL tasks. In scenario II, CoPreFL also consistently outperforms baselines in this case by effectively utilizing server data in addition to balancing our objectives (3). The benefit of a small server-side dataset, when available, can be seen through the performance gains from Table 1a to 1b. Further, the improvement over CoPreFL-SGD suggests that conducting SGD with server data in a centralized manner after meta-updating the global model may divert the pre-trained model away from our designed objectives. This emphasizes the importance of performing meta-learning on the server data following the partitioning outlined in Algorithm 2. For more results on various configurations, including impacts of varying server dataset sizes, see Appendix D.2.

Performance distribution comparison. Figure 2 show the testing accuracy distributions of the final global model on each client in downstream tasks. We visualize results for our method and those with the second-best average accuracy and the second-lowest variance from Table 1. CoPreFL shows consistently narrower distributions, re-

<table><tr><td> $\gamma$ </td><td>Acc  $\uparrow$ </td><td>Variance  $\downarrow$ </td></tr><tr><td>0.0</td><td>83.11  $\pm$  2.17</td><td>24.70  $\pm$  1.95</td></tr><tr><td>0.25</td><td>84.04  $\pm$  2.00</td><td>35.88  $\pm$  2.59</td></tr><tr><td>0.5</td><td>85.23  $\pm$  2.43</td><td>35.40  $\pm$  2.75</td></tr><tr><td>0.75</td><td>85.19  $\pm$  2.38</td><td>39.31  $\pm$  2.64</td></tr><tr><td>1.0</td><td>86.33  $\pm$  1.92</td><td>39.81  $\pm$  2.30</td></tr></table>

Table 2: Effect of balancer $\gamma$ in scenario I for Tiny-ImageNet.

flecting reduced performance variance. Distribution shifts to the right can also be found, indicating higher average accuracy. Importantly, CoPreFL effectively shifts nearly all low-performing clients from the left tail of the baselines towards the right, enhancing accuracy for these clients. More results for various scenarios are available in Appendix D.3.

Effect of balancer $\gamma$ in CoPreFL. Table 2 presents the performance of CoPreFL on Tiny-ImageNet using different balancers $\gamma$ . A larger $\gamma$ implies that the pre-trained model prioritizes the devices' average performance, whereas a smaller $\gamma$ emphasizes performance balance. We see that increasing $\gamma$ lead to higher average accuracy in downstream FL tasks but also greater variance, indicating performance imbalance. This trend shows that CoPreFL allows control over the relative importance between accuracy and balanced performance during pre-training.

Comparison with other initialization methods. In addition to the baselines in Table 1b which consider performance balance or meta-learning, Table 3a also evaluates popular algorithms like SCAFFOLD (Karimireddy et al. 2019), FedDyn (Acar et al. 2021), and PerFedAvg (Fallah, Mokhtari, and Ozdaglar 2020) providing pre-training for scenario I. We see that CoPreFL also outperforms these baselines, validating our meta-learning approach based on (3). These popular FL algorithms struggle with unseen task heterogeneity and performance balance. We also explore initializing

<table><tr><td>Pre-training (Scenario I)</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>Random Initialization</td><td>75.32 ± 1.68</td><td>41.39 ± 3.35</td></tr><tr><td>Centralized (Nguyen et al. 2023)</td><td>81.30 ± 2.92</td><td>69.44 ± 2.33</td></tr><tr><td>SCAFFOLD (Karimireddy et al. 2019)</td><td>79.15 ± 3.08</td><td>57.84 ± 1.95</td></tr><tr><td>FedDyn (Acar et al. 2021)</td><td>81.23 ± 2.96</td><td>53.17 ± 2.85</td></tr><tr><td>PerFedAvg (Fallah, Mokhtari, and Ozdaglar 2020)</td><td>81.58 ± 1.83</td><td>49.73 ± 2.65</td></tr><tr><td>CoPreFL</td><td>83.29 ± 2.61</td><td>34.69 ± 3.17</td></tr></table>

(a) Comparison with other initializations.

<table><tr><td rowspan="2">Pre-training (Scenario I)</td><td colspan="4">Downstream: Non-IID FL</td></tr><tr><td colspan="2">FedProx ( $\mu = 1$ )</td><td colspan="2">q-FFL (q = 2)</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>Centralized</td><td>82.39 ± 3.17</td><td>51.46 ± 2.59</td><td>79.26 ± 2.33</td><td>47.10 ± 3.05</td></tr><tr><td>FedAvg</td><td>79.53 ± 2.69</td><td>46.15 ± 3.04</td><td>79.53 ± 2.38</td><td>44.59 ± 2.95</td></tr><tr><td>FedMeta</td><td>81.77 ± 3.29</td><td>63.12 ± 3.62</td><td>79.30 ± 3.02</td><td>39.63 ± 3.17</td></tr><tr><td>q-FFL</td><td>83.19 ± 3.03</td><td>52.12 ± 2.97</td><td>81.38 ± 2.67</td><td>37.27 ± 2.85</td></tr><tr><td>CoPreFL</td><td>84.31 ± 3.01</td><td>30.55 ± 2.61</td><td>82.71 ± 2.45</td><td>25.39 ± 2.87</td></tr></table>

(b) Integration with other downstream FL algorithms.

Table 3: Results with (a) other initializations and (b) other downstream FL algorithms on CIFAR-100. 

<table><tr><td>Pre-training</td><td colspan="4">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Var ↓</td><td>Lowest 10%</td><td>Lowest 20%</td></tr><tr><td>Centralized</td><td>82.63 ± 2.63</td><td>63.57 ± 3.08</td><td>67.35 ± 2.57</td><td>69.22 ± 3.01</td></tr><tr><td>FedAvg</td><td>80.19 ± 1.19</td><td>51.35 ± 2.44</td><td>68.72 ± 1.63</td><td>70.15 ± 1.45</td></tr><tr><td>FedMeta</td><td>83.14 ± 2.07</td><td>39.85 ± 1.38</td><td>67.29 ± 2.22</td><td>71.35 ± 2.53</td></tr><tr><td>q-FFL</td><td>81.34 ± 1.91</td><td>47.98 ± 2.00</td><td>69.22 ± 1.85</td><td>70.35 ± 2.39</td></tr><tr><td>CoPreFL</td><td>84.79 ± 1.25</td><td>30.51 ± 1.72</td><td>70.83 ± 1.59</td><td>72.66 ± 1.61</td></tr></table>

Table 4: Results with both seen/unseen classes during downstream FL, using the CIFAR-100 dataset in scenario I.

downstream FedAvg with random weights or a centrally pretrained model, a concept introduced in (Nguyen et al. 2023). While the centralized method boosts downstream FL accuracy over random initialization, it introduces significant performance variance across clients, as it fails to mimic downstream FL characteristics. More details along with additional results can be found in Appendix D.5.

Compatibility with other downstream FL algorithms. We next explore the ability of our pre-training method to enhance the performance of downstream FL algorithms other than FedAvg. For this, we consider FedProx (Sahu et al. 2018) and q-FFL (Li et al. 2020), more advanced FL algorithms that addresses heterogeneity and performance balance, in each federated downstream task. Table 3b shows the results. Overall, we see that CoPreFL consistently achieves superiority in accuracy and variance compared to other pre-training baselines, when combined with different downstream FL algorithms. Details on the implementation and further discussions are provided in Appendix D.6.

Both unseen/seen classes in downstream FL tasks. In addition to the setting without overlapping classes between pre-training and downstream tasks, we explore a mixed scenario where clients in the downstream FL also hold “seen classes.” To implement this, we use CIFAR-100 and randomly sampled 10 classes from the pre-training dataset and 10 classes from the downstream dataset, resulting in 10 seen and 10 unseen downstream classes. From this, we constructed 10 downstream tasks by randomly selecting 5 classes in each case, to conduct non-IID downstream FedAvg. The pre-trained models were solely trained on the original 80 classes of CIFAR-100. Table 4 shows the results. We see that the accuracies are generally higher compared to those in Table 1a, as the downstream tasks involve classes seen during pre-training. Once again, the improvements in each metric confirm the advantage of CoPreFL.

Application to centrally stored public dataset. We also explore the applicability of CoPreFL with centrally stored pre-training data, as detailed in Remark 2. We conducted pre-training using the ImageNet\_1K dataset and use FedAvg with CIFAR-100 as the downstream task. We intentionally split the dataset according to scenario I to mimic the

<table><tr><td>Pre-training</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>Centralized</td><td>86.75 ± 2.89</td><td>67.34 ± 2.17</td></tr><tr><td>CoPreFL</td><td>87.96 ± 1.95</td><td>30.79 ± 2.79</td></tr></table>

Table 5: Results with centrally stored dataset. ImageNet is used for pre-training, while CIFAR-100 is used for downstream FL.

<table><tr><td>Pre-training</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>FedAvg</td><td>62.23 ± 2.65</td><td>51.29 ± 3.11</td></tr><tr><td>FedMeta</td><td>64.35 ± 3.07</td><td>44.38 ± 2.98</td></tr><tr><td>q-FFL</td><td>60.79 ± 3.15</td><td>27.96 ± 3.03</td></tr><tr><td>CoPreFL</td><td>66.83 ± 2.85</td><td>24.31 ± 2.83</td></tr></table>

Table 6: Results in the domain shift scenario using PACS dataset.

distributed nature of downstream FL. Results in Table 5 show that CoPreFL outperforms standard centralized pre-training in both average accuracy and balanced performance, showing CoPreFL's advantage even when a large public dataset is used for pre-training. Further details and additional results are available in Appendix D.7.

Comparison under domain shifts. Our experiments so far have focused on the robustness of the pre-trained model to unseen labels. We now explore CoPreFL's ability to handle unseen data domains during downstream tasks using the PACS dataset. In Table 6, we use 3 domains (Art, Cartoon, and Photo) for pre-training in scenario I and conduct downstream FedAvg using the remaining Sketch domain, distributing samples across clients as in Table 1. Results show that CoPreFL effectively handles domain distributions in downstream FL tasks that differ from the pre-training phase. This highlights CoPreFL's ability to provide robust initializations amid various types of downstream data heterogeneity. Additional settings and results are available in Appendix D.1.

# 5 Conclusion

We presented CoPreFL, a collaborative pre-training method that provides a robust model initialization for an arbitrary set of downstream FL tasks. CoPreFL leverages meta-learning to equip the pre-trained model with the ability to handle different forms of data heterogeneity that manifest in downstream FL, while balancing between average performance and variance across clients. We developed CoPreFL for different distributed pre-training scenarios, and showed its benefit even for centrally stored public data. Extensive experiments demonstrated the advantages of CoPreFL compared with several baselines methods, for a multitude of settings capturing statistical heterogeneity in downstream FL.

# References

Acar, D. A. E.; Zhao, Y.; Navarro, R. M.; Mattina, M.; Whatmough, P. N.; and Saligrama, V. 2021. Federated Learning Based on Dynamic Regularization. abs/2111.04263.

Bian, J.; Wang, L.; Yang, K.; Shen, C.; and Xu, J. 2023. Accelerating Hybrid Federated Learning Convergence under Partial Participation. ArXiv, abs/2304.05397.

Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T. J.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. Conference on Neural Information Processing Systems, abs/2005.14165.

Caldas, S.; Wu, P.; Li, T.; Konecný, J.; McMahan, H. B.; Smith, V.; and Talwalkar, A. 2018. LEAF: A Benchmark for Federated Settings. ArXiv, abs/1812.01097.

Chang, W.-C.; Yu, F. X.; Chang, Y.-W.; Yang, Y.; and Kumar, S. 2020. Pre-training Tasks for Embedding-based Large-scale Retrieval. ArXiv, abs/2002.03932.

Chen, F.; Luo, M.; Dong, Z.; Li, Z.; and He, X. 2018. Federated Meta-Learning with Fast Convergence and Efficient Communication. arXiv: Learning.

Chen, H.-Y.; Tu, C.-H.; Li, Z.; Shen, H. W.; and Chao, W.-L. 2023. On the Importance and Applicability of Pre-Training for Federated Learning. In The Eleventh International Conference on Learning Representations.

Cho, Y. J.; Jhunjhunwala, D.; Li, T.; Smith, V.; and Joshi, G. 2022. Maximizing Global Model Appeal in Federated Learning.

Chu, Y.-W.; Hosseinalipour, S.; Tenorio, E.; Cruz, L.; Douglas, K. A.; Lan, A. S.; and Brinton, C. G. 2022. Mitigating Biases in Student Performance Prediction via Attention-Based Personalized Federated Learning. Proceedings of the 31st ACM International Conference on Information & Knowledge Management.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-Fei, L. 2009. ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248–255.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, abs/1810.04805.

Diao, Y.; Li, Q.; and He, B. 2023. Exploiting Label Skews in Federated Learning with Model Concatenation. In AAAI Conference on Artificial Intelligence.

Dong, J.; Wu, H.; Zhang, H.; Zhang, L.; Wang, J.; and Long, M. 2023. SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling. ArXiv, abs/2302.00861.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.;

Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. The Ninth International Conference on Learning Representations, abs/2010.11929.   
Fallah, A.; Mokhtari, A.; and Ozdaglar, A. E. 2020. Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach. In Neural Information Processing Systems.   
Finn, C.; Abbeel, P.; and Levine, S. 2017. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In International Conference on Machine Learning.   
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2015. Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.   
Hu, W.; Liu, B.; Gomes, J.; Zitnik, M.; Liang, P.; Pande, V. S.; and Leskovec, J. 2019. Strategies for Pre-training Graph Neural Networks. arXiv: Learning.   
Jamal, M. A.; Brown, M. A.; Yang, M.-H.; Wang, L.; and Gong, B. 2020. Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition From a Domain Adaptation Perspective. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7607–7616.   
Ji, S.; Pan, S.; Long, G.; Li, X.; Jiang, J.; and Huang, Z. 2019. Learning Private Neural Language Modeling with Attentive Aggregation. The International Joint Conference on Neural Networks, 1–8.   
Jiang, Y.; Konecný, J.; Rush, K.; and Kannan, S. 2019. Improving Federated Learning Personalization via Model Agnostic Meta Learning. ArXiv, abs/1909.12488.   
Karimireddy, S. P.; Kale, S.; Mohri, M.; Reddi, S. J.; Stich, S. U.; and Suresh, A. T. 2019. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. In International Conference on Machine Learning.   
Konecný, J.; McMahan, H. B.; Yu, F. X.; Richtárik, P.; Suresh, A. T.; and Bacon, D. 2016. Federated Learning: Strategies for Improving Communication Efficiency. ArXiv, abs/1610.05492.   
Krizhevsky, A. 2009. Learning Multiple Layers of Features from Tiny Images.   
Le, Y.; and Yang, X. S. 2015. Tiny ImageNet Visual Recognition Challenge.   
Li, D.; Yang, Y.; Song, Y.-Z.; and Hospedales, T. M. 2017. Deeper, Broader and Artier Domain Generalization. 2017 IEEE International Conference on Computer Vision (ICCV), 5543–5551.   
Li, Q.; He, B.; and Song, D. X. 2021. Model-Contrastive Federated Learning. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10708–10717.   
Li, T.; Sanjabi, M.; Beirami, A.; and Smith, V. 2020. Fair Resource Allocation in Federated Learning. In International Conference on Learning Representations.   
Li, X.; Dai, Y.; Ge, Y.; Liu, J.; Shan, Y.; and Duan, L.-Y. 2022. Uncertainty Modeling for Out-of-Distribution Generalization. ArXiv, abs/2202.03958.

McMahan, H. B.; Moore, E.; Ramage, D.; Hampson, S.; and y Arcas, B. A. 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics.   
Mohri, M.; Sivek, G.; and Suresh, A. T. 2019. Agnostic Federated Learning. International Conference on Machine Learning, abs/1902.00146.   
Morafah, M.; Vahidian, S.; Chen, C.; Shah, M.; and Lin, B. 2022. Rethinking Data Heterogeneity in Federated Learning: Introducing a New Notion and Standard Benchmarks. NeurIPS 2022 workshop on Federated Learning, abs/2209.15595.   
Nguyen, J.; Wang, J.; Malik, K.; Sanjabi, M.; and Rabbat, M. 2023. Where to Begin? On the Impact of Pre-Training and Initialization in Federated Learning. In The Eleventh International Conference on Learning Representations.   
Park, Y.; Han, D.-J.; Kim, D.-Y.; Seo, J.; and Moon, J. 2021. Few-Round Learning for Federated Learning. In Neural Information Processing Systems.   
Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; and Sutskever, I. 2019. Language Models are Unsupervised Multitask Learners.   
Reddi, S. J.; Charles, Z. B.; Zaheer, M.; Garrett, Z.; Rush, K.; Konecný, J.; Kumar, S.; and McMahan, H. B. 2021. Adaptive Federated Optimization. The Ninth International Conference on Learning Representations, abs/2003.00295.   
Sahu, A. K.; Li, T.; Sanjabi, M.; Zaheer, M.; Talwalkar, A.; and Smith, V. 2018. Federated Optimization in Heterogeneous Networks.   
Shu, J.; Xie, Q.; Yi, L.; Zhao, Q.; Zhou, S.; Xu, Z.; and Meng, D. 2019. Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting. In Neural Information Processing Systems.   
Wang, H.; Yurochkin, M.; Sun, Y.; Papailiopoulos, D.; and Khazaeni, Y. 2020. Federated Learning with Matched Averaging. The Eighth International Conference on Learning Representations, abs/2002.06440.   
Yang, K.; Chen, S.; and Shen, C. 2023. On the Convergence of Hybrid Server-Clients Collaborative Training. IEEE Journal on Selected Areas in Communications, 41: 802–819.   
Yuan, H.; Mu, Z.; Xie, F.; and Lu, Z. 2024. Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning. In The Twelfth International Conference on Learning Representations.   
Zhang, X.; Yin, W.; Hong, M.; and Chen, T. 2020. Hybrid Federated Learning: Algorithms and Implementation. ArXiv, abs/2012.12420.   
Zhang, X.; Zhao, Z.; Tsiligkaridis, T.; and Zitnik, M. 2022. Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency. ArXiv, abs/2206.08496.   
Zhou, K.; Yang, Y.; Qiao, Y.; and Xiang, T. 2021. Domain Generalization with MixStyle. ArXiv, abs/2104.02008.

# A Key Applications

Consider a healthcare application where each client, such as a hospital or an individual patient, aims to build a comprehensive global model capable of classifying a wide range of diseases. However, individual clients may possess limited types of diseases in their local datasets – for instance, one client may have data on diseases A and B but lacks information on diseases C and D. In this context, federated learning becomes essential. Clients need to collaborate to construct a global model that not only reflects the diseases available locally but also incorporates information about diseases not present in their individual datasets, ensuring a more robust and universally applicable healthcare model. Similarly, in the domain of autonomous vehicles, each self-driving car may strive to develop a global model for scenario detection in various weather conditions. However, individual cars might encounter limited weather scenarios locally – one car might navigate through a desert environment, while another faces challenges in a snowy storm. Through federated learning, these cars can collectively construct a global model that accounts for a broad spectrum of weather conditions, ensuring robust scenario detection capabilities for all vehicles involved.

As noted in Remark 2, the server can intentionally partition the centralized dataset and implement our scheme, utilizing multiple computing units available at the server, to obtain a pre-trained model. The advantage of this approach, compared to simple centralized training, lies in mitigating side effects such as performance biases and the substantial variance associated with centralized training. This phenomenon stems from the lack of generalizability in the model's design. When a model undergoes pre-training in a centralized manner based on SGD, it becomes rigidly bound to the knowledge in the pre-training dataset. This fixation presents a challenge in adapting the model to the diverse clients that may possess new or unseen data in downstream tasks. Such variations can arise from factors like the time-varying environment or new clients joining the system, as exemplified in the aforementioned applications: classifying different scenarios based on the self-driving car's environment, identifying diverse diseases based on patient interests, or enabling face/speech recognition for new phone users.

In our experimental comparison, we consider a FL baseline, FedMeta (Chen et al. 2018), which also incorporates meta-learning. We would like to emphasize that the distinction between our CoPreFL and FedMeta lies in the application, which requires us to conduct meta-update in a totally different way. FedMeta is designed to offer personalization to individual clients, e.g., when a specific client is interested in predicting only diseases A and B, or when a specific self-driving car is interested in the model tailored to a specific weather. In contrast, our emphasis is on creating an initial model that can construct a good “global model” during downstream instead of “personalized models”, targeting the aforementioned applications. This is the reason why we need to update the temporary global model instead of the local models, which is the key technical difference with FedMeta. Another technical difference is the consideration of performance balance in our method. These two key techniques enables CoPreFL to construct a robust initial model that can quickly adapt to “any group of clients” (instead of individual clients) to construct a global model during downstream tasks.

# B Detailed Procedure for CoPreFL in Scenario II

This section provides a detailed introduction to our CoPreFL in scenario II, as discussed in Section 3.3. Similar to the goals of CoPreFL in scenario II, we still aim to balance between the objective functions in (3), but in this scenario, the data used to perform meta-updates and control our objectives is different. During each federated round $t$ in the pre-training phase, participants download the global model $\Phi^{t-1}$ from the previous round (line 6 in Algorithm 2). Subsequently, they perform few local training iterations utilizing their respective local datasets $D_j^p$ (line 8 in Algorithm 2). This process leads to a training loss $\ell_{D_j^p}^e(\Phi^t)$ for each local epoch $e$ , defined as $\frac{1}{|D_j^p|}\sum_{(x,y)\in D_j^p}\ell(\Phi_j^{t,e}(x),y)$ , where $x$ represents the input (e.g., images), $y$ denotes the true label, and $\ell(\cdot)$ denotes the loss function (e.g., cross-entropy loss). After all participants finish $E$ epochs, we obtain the local model $\Phi_j^{t,E}$ . Upon the completion of local training by all participants, participants' local models are transmitted to the server (line 12 in Algorithm 2), and the server aggregates these models into a temporary global model $\overline{\Phi^t} = \sum_{j\in m}\mu_j\Phi_j^{t,E}$ , which is weighted by relative dataset sizes $\mu_j = \frac{|D_j^p|}{\sum_{i\in m}|D_i^p|}$ .

We then perform meta-updates on the temporary global model $\overline{\Phi^{t}}$ using server's dataset $D^{s}$ . To start, we first randomly divide the server's dataset $D^{s}$ into $|m|$ equal partitions. Instead of equal partitioning, one can also divide the server-side dataset into partitions with unequal sizes and use them for meta-update. However, during pre-training, the server does not know the dataset sizes of clients in future downstream tasks. In this case, one intuitive way is to treat all clients equally/fairly, by meta-updating the model with the same query set sizes. We show that this equal partitioning provides significant performance improvements as can be seen in our experiments. Subsequently, the server evaluates the temporary global model $\overline{\Phi^{t}}$ using each subset $D_{j}^{s}$ (line 14 in Algorithm 2), resulting in the corresponding gradient and loss $\mathcal{L}_{D_{j}^{s}}(\overline{\Phi^{t}})=\frac{1}{|D_{j}^{s}|}\sum_{(x,y)\in D_{j}^{s}}\ell(\overline{\Phi^{t}}(x),y)$ . The collective server's loss, denoted as $\mathcal{L}_{D^{s}}(\overline{\Phi^{t}})$ in line 16 of Algorithm 2, is determined by aggregating all the collected loss values obtained from $D_{j}^{s}$ , and we also calculate the variance $\sigma_{D^{s}}^{2}=\frac{1}{m}\sum_{i\in m}(\mathcal{L}_{D_{i}^{s}}(\overline{\Phi^{t}})-\frac{1}{m}\mathcal{L}_{D^{s}}(\overline{\Phi^{t}}))$ across server's losses to examine the performance distribution. We then tailor a customized server meta-loss $\mathcal{L}_{meta}(\overline{\Phi^{t}})=\gamma\mathcal{L}_{D^{s}}(\overline{\Phi^{t}})+(1-\gamma)\sigma_{D^{s}}^{2}(\overline{\Phi^{t}})$ to achieve a balance between optimizing for performance and performance balance. Finally, in line 19 of Algorithm 2, we employ the customized server meta-loss $\mathcal{L}_{meta}(\overline{\Phi^{t}})$ and the aggregated gradient gathered from the server's subsets to update the temporary

global model $\overline{\Phi^{t}}$ , aligning it with our controlled objective. The server then sends this meta-updated global model $\Phi^{t}$ to the participants in the next round for initialization. After completing T federated rounds, we regard the final global model $\Phi^{T}$ as the pre-trained model in scenario II, which serves as the initialization for the downstream FL tasks.

# C Detailed Settings for Datasets and Hyperparameters

# C.1 Dataset Details

In a practical scenario where labels for downstream tasks are unknown during pre-training, we split the dataset based on classes. For CIFAR-100 (100 classes with 600 images per class), 80 classes are used for pre-training and 20 classes for downstream FL tasks, resulting in 48,000 images for pre-training and 12,000 images for downstream tasks. Similarly, for Tiny-ImageNet (200 classes with 600 images per class), 160 classes are employed for pre-training and 40 classes for downstream FL tasks, providing 96,000 images for pre-training and 24,000 images for downstream tasks. We randomly select 95% of the pre-training dataset samples for clients, while the remaining 5% of samples for the server. Specifically, for CIFAR-100, this results in 45,600 images for clients and 2,400 images for the server. For Tiny-ImageNet, we allocate 91,200 images for clients and 4,800 images for the server. We will distribute 45,600 and 91,200 images to $|M| = 100$ clients based on IID or non-IID data distribution for CIFAR-100 and Tiny-ImageNet, respectively. Each client further divides its local data into 80% support samples and 20% query samples.

For the downstream phase samples (12,000 images for CIFAR-100 and 24,000 images for Tiny-ImageNet), we randomly select 5 classes from the available downstream classes (20 classes for CIFAR-100 and 40 classes for Tiny-ImageNet) to form a single FL task. This results in 3,000 images (5 classes with 600 images per class) for each FL task in both CIFAR-100 and Tiny-ImageNet datasets. These 3,000 images are distributed to $|G| = 10$ clients based on either IID or non-IID data distribution. Within each client, $80\%$ of local data is used for FL training, while the remaining $20\%$ is reserved to evaluate the final global model.

In the IID setup, data samples from each class are distributed equally to $|M| = 100$ clients for pre-training and $|G| = 10$ clients for downstream FL task. Taking the CIFAR-100 dataset in the IID pre-training phase as an example, there are $|M| = 100$ clients, each holding 456 images. We will randomly select $|m|$ clients to participate in pre-training in each round. In the IID downstream phase of CIFAR-100, there are $|G| = 10$ clients, each holding 300 images. In the non-IID setup, samples within each class are partitioned among $|M|$ and $|G|$ clients using a Dirichlet( $\alpha$ ) distribution for pre-training and downstream task, respectively, with $\alpha = 0.5$ selected as is in the literature (Morafah et al. 2022; Li, He, and Song 2021).

In addition to the label distributional shifts discussed in our manuscript, we also explore our method's capability to handle domain shifts. To assess this, we conducted an additional experiment using the PACS dataset (Li et al. 2017). As PACS dataset consists of four domains (Art, Cartoon, Photo, Sketch), we separate them into three domains for pre-training and one domain for downstream FL, following the one-domain-leave-out setup (Li et al. 2022; Zhou et al. 2021).

# C.2 Hyperparameters and Compute Settings

We use ResNet-18 as the model structure for image classification, following the setting of (Nguyen et al. 2023; Chen et al. 2023). For our method, the SGD optimizer with a learning rate of $\eta = 10^{-3}$ and $\zeta = 10^{-3}$ is adopted for both local and meta updates. Both local and meta learning rates are searched within the range of [1e-2, 5e-3, 1e-3, 5e-4]. We searched for learning rates within the range of [1e-2, 5e-3, 1e-3, 5e-4] for local training of all FL pre-training baselines and selected 1e-3 as the optimal learning rate for them. In scenario II, each FL baseline will continue to conduct a few SGD iterations using the server's data after constructing their global model. We searched for learning rates in the range of [1e-2, 1e-3] for this additional training and selected 1e-3 as the optimal learning rate for the server. Regarding hyperparameters in the q-FFL baseline, we conducted experiments with q-values of 1, 3, and 5 and reported the corresponding best statistics. We select a learning rate $\eta$ from the range [1e-2, 5e-3, 1e-3, 5e-4] for local updates in our CoPreFL and determined that 1e-3 provides the best results. Additionally, for meta-updates in both scenarios, we search for the learning rate $\zeta$ within the range [1e-2, 1e-3] and find that 1e-3 is the optimal value. In the case of the centralized baseline mentioned in Section 4.2, we searched for the optimal learning rate within the range [1e-2, 5e-3, 1e-3, 5e-4, 1e-4], ultimately selecting 1e-3. We utilized the SGD optimizer for all updates across all methods, and the batch size is set to be 32 for all experiments.

For the pre-training phase, we set the number of rounds to T = 50, and each round of local training takes E = 5 epochs for each client. For downstream FL tasks, we employ the widely used FedAvg algorithm to isolate the effects of different FL approaches and focus specifically on the impact of different initializations. We consider R = 50 FL rounds using the training set, involving 5 iterations per round for local training using the SGD optimizer with a learning rate of $10^{-3}$ . For the settings of other downstream FL algorithms, see Appendix D.6. In our simulations of CoPreFL, we assessed various balancer values $\gamma$ from the range [0.0, 0.25, 0.5, 0.75, 1.0] in all scenarios during pre-training. For evaluation of downstream FL, we report the best-performing (highest average accuracy) value in our paper. For fair selection/comparison, we also report the results of other baselines with their own best accuracy when searching for hyperparameters. We run all experiments on a 3-GPU cluster of Tesla V100 GPUs, with each GPU having 32GB of memory.

Algorithm 2: Our Pre-training Method CoPreFL (Pre-training Phase in Scenario II)   
1: Input: M clients in the pre-training phase, with each client i holding their own dataset $D_{i}^{p}$ ; the server also holds a dataset $D^{s}$ .
2: for Each communication round $t = 1, 2, ..., T$ do
3: Randomly select a set of client $m \subset M$ to participate in learning
4: Randomly split server's dataset $D^{s}$ into $|m|$ subsets
5: for Each participant j in parallel do
6: Downloads $\Phi^{t-1}$ from the server
7: for local epoch $e = 1, 2, ...E$ do
8: $\ell_{D_{j}^{p}}^{e}(\Phi^{t}) \leftarrow \frac{1}{|D_{j}^{p}|} \sum_{(x,y) \in D_{j}^{p}} \ell(\Phi_{j}^{t,e}(x), y)$ { Compute local support loss $\ell_{D_{j}^{p}}^{e}\}$ 9: $\Phi_{j}^{t,e} \leftarrow \Phi_{j}^{t,e-1} - \eta \nabla \ell_{D_{j}^{p}}^{e}(\Phi^{t})$ {Perform SGD local update using support loss}
10: end for
11: end for
12: $\overline{\Phi^{t}} \leftarrow \sum_{j \in m} \frac{|D_{j}^{p}|}{\sum_{i \in m} |D_{i}^{p}|} \Phi_{j}^{t,E}$ {Model aggregation to construct temporary global model}
13: for Each split server's dataset $D_{j}^{s}$ in parallel, Server do
14: $\mathcal{L}_{D_{j}^{s}}(\overline{\Phi^{t}}) \leftarrow \frac{1}{|D_{j}^{s}|} \sum_{(x,y) \in D_{j}^{s}} \ell(\overline{\Phi^{t}}(x), y)$ {Server's loss corresponding to each partition}
15: end for
16: Overall meta-loss on server: $\mathcal{L}_{D^{s}}(\overline{\Phi^{t}}) = \sum_{j \in m} \mathcal{L}_{D_{j}^{s}}(\overline{\Phi^{t}})$ 17: Variance across server meta-losses: $\sigma_{D_{s}}^{2}(\overline{\Phi^{t}}) = \frac{1}{|m|} \sum_{j \in m} (\mathcal{L}_{D_{j}^{s}}(\overline{\Phi^{t}}) - \frac{1}{|m|} \mathcal{L}_{D^{s}}(\overline{\Phi^{t}}))^{2}$ 18: Customized server meta-loss: $L_{meta}(\overline{\Phi^{t}}) = \gamma L_{D^{s}}(\overline{\Phi^{t}}) + (1 - \gamma) \sigma_{D^{s}}^{2}(\overline{\Phi^{t}})$ 19: $\Phi^{t} \leftarrow \overline{\Phi^{t}} - \zeta \nabla L_{meta}(\overline{\Phi^{t}})$ {Model meta-updates using customized loss}
20: end for
21: Output: A pre-trained model for downstream FL tasks: $\Phi^{T}$

# D Additional Experiments and Analyses

# D.1 Downstream FL Results with Scenario I Pre-training

Additional results with varying degrees of IIDness and numbers of participants. This section provides supplementary results for pre-training scenario I, as discussed in Section 4.2. We train pre-trained models using both IID and non-IID distributions, varying the number of participants in each federated round during the pre-training phase. To be more specific, we specify the number of participants $|m|$ as 15, 20, 25, and 30 out of 100 clients to participate in FL during the pre-training phase. Subsequently, we evaluate these pre-trained models by initializing them for IID and non-IID downstream FL tasks. Tables 7, 8, 9, and 10 display the average performance across 10 IID FL downstream tasks and Tables 11, 12, 13, and 14 show the average performance across 10 non-IID FL downstream tasks. In both cases, the downstream FL were initialized by pretrained models trained on 15, 20, 25, and 30 participants out of 100 clients, respectively, on the CIFAR-100 dataset. For the Tiny-ImageNet dataset, Tables 15, 16, 17, and 18 show the average performance across 10 IID FL downstream tasks and Tables 19, 20, 21, and 22 display the average performance across 10 non-IID FL downstream tasks. In both cases, the downstream FL were also initialized by pretrained models trained on 15, 20, 25, and 30 participants out of 100 clients, respectively. Across these experimental results, considering different data distribution setup during the pre-training phase and different datasets, our CoPreFL con sistently demonstrates superiority over the baseline when used as an initialization for various downstream FL tasks. By creating an environment that mimics downstream FL tasks and specifically addressing the challenges encountered in these tasks, our designed pre-training objectives in (3) establish an ideal pre-trained model for FL. As initialization for various unseen FL tasks, our CoPreFL provide downstream FL tasks with both better average performance and balanced predictions across clients.

Comparison of IID and non-IID downstream tasks. In some cases, when comparing IID and non-IID downstream tasks, we see that data heterogeneity has only a slight impact on the proposed approach. To be more specific, the performance of non-IID downstream FL experiences only a slight performance drop compared to IID downstream FL. In an IID setting, the model must classify all classes present in the system, as each client will have most classes in its local dataset. However, in a non-IID downstream scenario, the model may only need to classify a few classes since each client typically has a limited subset of classes. This simplification of local tasks in the non-IID setting reduces the complexity of evaluation for each client, minimizing the impact of data heterogeneity on the overall federated learning process. This phenomenon aligns with findings from other FL studies, where global model performance is assessed using local test sets from individual clients (Li et al. 2020; Diao, Li, and He 2023).

Varying the number of downstream tasks. To model various unseen downstream scenarios, we conduct 5-way classification during downstream FL (i.e., sampling 5

classes from 20 in CIFAR-100 downstream dataset to conduct one task.) The goal is to design a “generalized initial model” that can adapt to arbitrary downstream tasks that potentially contain unseen classes and to evaluate the versatility of the pre-trained model in providing a robust starting point for various scenarios. We also consider a “single downstream scenario” with 20-way classification using all downstream classes (that have not appeared during pre-training). To be more specific, we distribute the entire downstream dataset of CIFAR-100 among $|G|=10$ clients based on non-IID distribution and performed FedAvg. The results are provided in Table 23, indicating that CoPreFL still performs better than other initialization baselines.

Domain shifts. Each column in Table 24 represents the domain for downstream FL in a one-domain-leave-out setting. For example, in the first column, we used the art painting, cartoon, and photo data domains for pre-training and performed downstream non-IID FedAvg training using the sketch data domain. We followed the same experimental settings as described in Table 1a, considering scenario I, with pre-training data distributed to 100 clients based on a non-IID ( $\alpha = 0.5$ ) distribution, and selected 20 participants for each pre-training round. Table 24 presents the results for each pre-trained method using the PACS dataset. We observed that our pre-trained method CoPreFL outperforms other pre-training methods in scenarios involving domain shifts, both in terms of performance accuracy and performance balance.

# D.2 Downstream FL Results with Scenario II Pre-training

Additional results with varying degrees of IIDness and numbers of participants. This section provides supplementary results for scenario II, where the server holds a small portion of the dataset during the pre-training phase. We also consider varying numbers of participants $|m|$ , specifically 15, 20, 25, and 30 out of 100 clients, during the pre-training phase for these models. Tables 25, 26, 27, and 28 display the average performance across 10 IID FL downstream tasks and Tables 29, 30, 31, and 32 show the average performance across 10 non-IID FL downstream tasks. In both cases, the downstream FL were initialized by pretrained models trained on 15, 20, 25, and 30 participants out of 100 clients, respectively, on the CIFAR-100 dataset. For the Tiny-ImageNet dataset, Tables 33, 34, 35, and 36 show the average performance across 10 IID FL downstream tasks and Tables 37, 38, 39, and 40 display the average performance across 10 non-IID FL downstream tasks. In both cases, the downstream FL were also initialized by pre-trained models trained on 15, 20, 25, and 30 participants out of 100 clients, respectively. It is important to note that in this scenario, FedAvg, FedMeta, and q-FFL undergo further training using server data through the SGD optimizer after each method completes its local iterations and obtains its respective global model in each round (Yang, Chen, and Shen 2023; Bian et al. 2023). Similarly, CoPreFL-SGD is trained using server data with the SGD optimizer on $\Phi^{t}$ in line 17 of Algorithm 1 in each round. This process involves conducting meta-updates and balancing performance and variance using clients' data first, followed by updating the aggregate model again using the server's dataset. Finally, CoPreFL follows Algorithm 2, utilizing server data for meta-updates. By incorporating meta-updates using server data to align with our objectives in (3), our pre-training method consistently outperforms other baselines, leading to improved average accuracy and reduced variance. Comparing CoPreFL with CoPreFL-SGD strongly suggests that, rather than conducting a few SGD iterations using server data, which may dilute our objectives, we recommend building pre-training objectives upon server data using meta-updates.

Alternative implementation for baselines in scenario II. In addition to the hybrid training approach introduced in (Yang, Chen, and Shen 2023; Bian et al. 2023), which utilizes clients' data $(\bigcup_{i\in M}D_i^p)$ to train local models and then refines the aggregated global model using the server's data $(D^{S})$ , we explore an alternative implementation for other FL baselines in scenario II. In this case, we distribute the entire training dataset $(\bigcup_{i\in M}D_i^p +D^S)$ to $|M| = 100$ clients and select $|m| = 20$ participants in each round for federated learning without a further refining step since there is no server's data in this case. Therefore, each client holds more samples compared to their previous scheme. Table 41 shows the average performance of downstream non-IID FedAvg using non-IID federated methods as initialization. It is important to note that we maintain a fixed data splitting setup for our method, meaning we use $\bigcup_{i\in M}D_i^p$ for local training and meta-update the temporary model using $D^{S}$ . These comparisons also show the superiority of our method.

Varying the amount of the server's dataset used for our method. We conduct an experiment to see how the quantity of the server's dataset, $|D^S|$ , impacts the performance. For the same experimental settings shown in Table 1b of the manuscript, while keeping the configurations for clients unchanged, we vary the amount of samples used in the server-side data, reducing it from $5\%$ (default) to $1\%$ or $2\%$ . Table 42 shows the result of our method using different amount of server's data for meta-updating temporary global model. We can see that though $D^S$ is small, it effectively contributes to obtaining a well-pretrained model when server data is available. able to benefit in terms of performance accuracy from the amount of pre-training data available at the server. Nevertheless, comparing with the baselines shown in Table 1b, our CoPreFL achieves a better accuracy with a smaller variance, even with just $1\%$ server-side data, further confirming its effectiveness.

# D.3 Testing Accuracy Distribution of Downstream FL tasks

This section presents supplementary distribution results to evaluate the performance balance of the pre-trained models discussed in Section 4.2. For pre-trained models trained in scenario I, Figures 3 and 4 show the testing accuracy distribution of IID and non-IID FL tasks on CIFAR-100 dataset, and Figures 5 and 6 display the respective distribution on Tiny-ImageNet dataset. Figures 7 and 8 present the testing accuracy distribution of IID and non-IID FL tasks initialized by pre-trained models trained in scenario II

on CIFAR-100 dataset, and Figures 9 and 10 show the respective distribution on Tiny-ImageNet dataset. Across our experimental results, which encompass different data distribution setups and scenarios during the pre-training phase and various datasets, our CoPreFL consistently enhances the performance balance of testing accuracy distributions for diverse downstream FL tasks. In general, distributions of FL tasks initialized by our CoPreFL tend to shift towards the right, indicating improved prediction performance. Moreover, when analyzing clients positioned at the left end of the distribution in each pre-training method, our approach effectively elevates underperforming clients towards the right end, resulting in enhanced predictive accuracy for these clients.

# D.4 Details and Additional Results for FEMNIST Dataset

We also consider the FEMNIST dataset, widely used in FL research, following the data partition provided in (Park et al. 2021). We divide the 62 classes into 52 alphabet classes for the pre-training phase, reserving the remaining 10 digit classes for downstream FL tasks. Instead of using a ResNet-18 model, we employ a model consisting of two $3 \times 3$ convolutional layers followed by two linear layers. We fixed the total number of clients as $|M| = 100$ for pre-training and $|G| = 10$ for downstream FL tasks. During the pre-training phase, we set the number of participants $|m| = 20$ and the federated round $T = 50$ for each federated pre-trained method. We use the SGD optimizer with a learning rate of $10^{-3}$ and batch size 32 for baselines and our method.

For downstream tasks, we randomly select 5 classes from a pool of 10 classes to conduct each FL task using FedAvg. We perform a total of X = 10 FL tasks and report the average evaluations across these tasks. Each task executes FedAvg for R = 10 rounds using the SGD optimizer with a learning rate of $10^{-3}$ . Tables 43 and 44 display the averaged performance of 10 IID and 10 non-IID FL downstream tasks, initialized by various pre-training methods trained in scenario I, on the FEMNIST dataset. For scenario II, Tables 45 and 46 show the performance of IID and non-IID downstream FL tasks. The results also demonstrate that our proposed CoPreFL serves as a robust initialization for various FL setups, benefiting both averaged accuracy and performance balance.

# D.5 Implementation Details and Additional Results for Different Initialization Methods

This section presents supplementary details and results with different initialization methods discussed in Section 4.2, including random initialization, centralized model initialization, and other FL algorithms used for initializing downstream FL. For scenario I, the centralized model is trained on a dataset collected from all $|M| = 100$ clients during the pre-training phase. In scenario II, the centralized model is trained on a dataset obtained from both $|M| = 100$ clients and the server. This centralized training is conducted using the SGD optimizer with a learning rate of $10^{-3}$ chosen from the range [1e-2, 5e-3, 1e-3, 5e-4], with a batch size of 64 and 50 epochs.

For FL baselines, we additionally consider SCAF-FOLD (Karimireddy et al. 2019), which addresses partial client sampling, FedDyn (Acar et al. 2021), designed to tackle non-IID issues, and PerFedAvg (Fallah, Mokhtari, and Ozdaglar 2020), aiming to provide an adaptable personalized model, for a detailed comparison. We train all FL algorithms for 50 rounds with $|m| = 20$ participants selected from $|M| = 100$ clients under a non-IID setting (Dirichlet $\alpha = 0.5$ ), and the final global model is used as initialization for downstream FedAvg. In the case of non-IID related FL, FedDyn, we set the parameter $\alpha$ to 0.01. For personalized FL, PerFedAvg, we employ a two-step gradient descent for local client training introduced in their paper. We use the SGD optimizer with a learning rate of $10^{-3}$ , a batch size of 32, and 50 federated rounds for these FL-based baselines.

Tables 47 and 48 display the average performance of 10 FL downstream tasks initialized by different pre-training methods trained in two scenarios on the CIFAR-100 dataset and Tiny-ImageNet dataset, respectively. Comparing centralized and random initialization, we observe that the centralized method generally improves the average accuracy of downstream FL but at the cost of higher variance in most cases. However, our CoPreFL consistently enhances both average accuracy and performance balance in various downstream FL tasks, demonstrating that with proper FL designs as pre-trained model, FL can be improved through initialization. Comparing with other FL designs as initialization, the results demonstrate the superiority of our method due to the considerations for unseen adaptation and performance balance during pre-training phase.

# D.6 Implementation Details and Additional Results for Different Downstream FL Tasks

In addition to the general downstream FL tasks built by FedAvg, we consider FedProx (Sahu et al. 2018) and q-FFL (Li et al. 2020), more advanced FL algorithms that addresses heterogeneity and performance balance compared to FedAvg, to examine the robustness and generalizability of our pre-trained method. The experiments are conducted using the CIFAR-100 dataset under non-IID pre-training scenario I. Two additional FL algorithms, non-IID Fed-Prox and non-IID q-FFL, are considered for downstream phase. We randomly sample 5 classes from the 20 available in our CIFAR-100 downstream dataset for each downstream task. The sampled data is then distributed to 10 clients, and the training consists of 50 rounds with 5 local iterations per round, utilizing an SGD optimizer with a learning rate of $10^{-3}$ . We set the parameters $\mu = 1$ for the proximal term coefficient in FedProx and $q = 2$ for the loss-reweighting coefficient in q-FFL, following the optimal values reported by the authors for the CIFAR-100 dataset.

In Tables 49 and 50, the results demonstrate that our pre-trained method maintains superiority in different downstream FL algorithms compared to other pre-training methods. It is important to note that the choice of FedAvg as our downstream task is made to minimize the varying impact introduced by other FL algorithms. Comparing the pre-training + downstream pairs, the improvement of CoPreFL + FedAvg (in Table 1a) over Centralized + FedProx/q-FFL

(in Table 49 and 50) shows that a better initialization, which considers the distributed scenario and balances performance in the pre-training phase, could potentially benefit the inferior downstream FL algorithm.

# D.7 Implementation Details when A Public Large-Scale Dataset is Available

In addition to utilizing CIFAR-100 and Tiny-ImageNet datasets, where we partition the datasets for pre-training and downstream tasks, we also explore a scenario where public large datasets are available for pre-training phase. We conducted experiments using pre-trained models with the ImageNet dataset (Deng et al. 2009), a widely used large public dataset for image classification. We sampled 200 images for each of the 1,000 classes in ImageNet\_1K as a pre-training dataset. We conducted pre-training using both the centralized method and our proposed CoPreFL with ImageNet\_1K. Subsequently, we conducted 10 non-IID FedAvg tasks using the CIFAR-100 dataset and initialized the models with these pre-trained models. For the centralized model in pre-training phase, we trained the model with the SGD optimizer and a learning rate of 1e-3, training the model for 50 epochs. For our proposed method during pre-training, we distributed all the sampled data across $|M| = 100$ clients based on non-IID distribution (Dirichlet $\alpha = 0.5$ ), sampling $|m| = 20$ clients in each round, and conducted CoPreFL for 50 rounds. Since the goal of this experiment is to demonstrate that even with a centrally-stored public large dataset, we can intentionally distribute the dataset and apply our method, we only consider conducting our method under scenario I. For the downstream FL phase, we apply 10 IID and non-IID FedAvg tasks to the 20-class and 40-class downstream datasets we used in CIFAR-100 and Tiny-ImageNet, respectively. Each task executes FedAvg for $R = 10$ rounds using the SGD optimizer with a learning rate of $10^{-3}$ . Note that all classes observed during downstream tasks are the seen classes that have already appeared during pre-training, given that CIFAR-100 and Tiny-ImageNet are the subsets of ImageNet.

Table 51 shows the performance of FL downstream tasks on CIFAR-100 and Tiny-ImageNet, where the downstream tasks are initialized by different methods trained on ImageNet\_1K dataset. As we can expect, models pre-trained on ImageNet\_1K provide downstream FL tasks with higher accuracy compared to those pre-trained on CIFAR-100 or Tiny-ImageNet (in Table 1a) since there is no unseen classes when pre-trained on ImageNet. More importantly, as mentioned, we can still apply our method by intentionally splitting the dataset and mimicking the distributed nature of downstream FL to achieve further performance improvements: The centrally pre-trained model on ImageNet achieves lower accuracy and higher variance compared to our CoPreFL. This advantage of CoPreFL is achieved by initializing the model to get higher accuracy and balanced performance in federated settings based on meta-learning. The overall results further confirm the advantage and applicability of our approach.

# E Limitations and Impact Statements

This paper presents a pre-training method with potential applications in various AI domains, including natural language processing and computer vision. It is essential to recognize and address potential ethical and privacy concerns associated with the pre-training dataset. For instance, considerations should be made for privacy issues related to images containing human faces and ethical concerns regarding toxic texts. By acknowledging and mitigating any such issues that arise, we can further promote responsible and ethical advancement of AI/ML technologies.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 15)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>87.01 ± 1.07</td><td>15.44 ± 1.43</td><td>78.91 ± 1.95</td><td>80.55 ± 2.38</td><td>81.41 ± 1.94</td></tr><tr><td>FedMeta</td><td>87.09 ± 1.42</td><td>14.67 ± 1.95</td><td>81.45 ± 2.07</td><td>82.42 ± 1.98</td><td>83.15 ± 2.05</td></tr><tr><td>q-FFL</td><td>87.25 ± 1.64</td><td>13.25 ± 1.38</td><td>80.85 ± 2.35</td><td>81.52 ± 1.66</td><td>82.26 ± 2.31</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>87.84 ± 1.05</td><td>11.49 ± 1.44</td><td>82.61 ± 1.84</td><td>83.52 ± 1.73</td><td>84.44 ± 2.03</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>85.85 ± 1.35</td><td>15.37 ± 1.71</td><td>78.91 ± 2.03</td><td>80.55 ± 1.88</td><td>81.41 ± 2.15</td></tr><tr><td>FedMeta</td><td>86.84 ± 1.65</td><td>12.25 ± 1.44</td><td>81.45 ± 2.07</td><td>82.42 ± 1.68</td><td>83.15 ± 1.95</td></tr><tr><td>q-FFL</td><td>86.37 ± 2.03</td><td>13.54 ± 1.51</td><td>80.85 ± 1.17</td><td>81.52 ± 2.04</td><td>82.26 ± 2.16</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>86.90 ± 1.41</td><td>8.70 ± 1.55</td><td>81.52 ± 1.30</td><td>82.58 ± 1.79</td><td>83.21 ± 2.07</td></tr></table>

Table 7: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario I, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>87.34 ± 0.95</td><td>12.46 ± 1.23</td><td>80.48 ± 2.33</td><td>81.64 ± 1.97</td><td>82.51 ± 1.65</td></tr><tr><td>FedMeta</td><td>86.70 ± 1.33</td><td>14.52 ± 1.29</td><td>81.33 ± 1.87</td><td>82.06 ± 2.09</td><td>82.75 ± 2.15</td></tr><tr><td>q-FFL</td><td>86.95 ± 0.68</td><td>11.97 ± 1.48</td><td>80.48 ± 1.95</td><td>81.58 ± 2.33</td><td>82.51 ± 2.07</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>87.54 ± 0.79</td><td>10.18 ± 1.25</td><td>81.94 ± 1.61</td><td>82.97 ± 2.05</td><td>83.84 ± 1.94</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>86.04 ± 1.31</td><td>14.36 ± 2.07</td><td>80.85 ± 1.98</td><td>81.39 ± 3.07</td><td>82.02 ± 2.55</td></tr><tr><td>FedMeta</td><td>86.15 ± 1.11</td><td>16.16 ± 2.09</td><td>79.52 ± 1.98</td><td>81.27 ± 2.71</td><td>82.18 ± 1.67</td></tr><tr><td>q-FFL</td><td>86.30 ± 1.35</td><td>17.14 ± 1.93</td><td>80.24 ± 1.66</td><td>81.58 ± 2.35</td><td>82.46 ± 1.79</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>86.32 ± 1.04</td><td>14.14 ± 1.58</td><td>81.45 ± 1.77</td><td>82.20 ± 2.05</td><td>82.75 ± 1.88</td></tr></table>

Table 8: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario I, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 25)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>87.68 ± 1.04</td><td>11.36 ± 1.88</td><td>81.58 ± 2.30</td><td>82.79 ± 1.92</td><td>83.68 ± 1.63</td></tr><tr><td>FedMeta</td><td>87.10 ± 1.28</td><td>13.62 ± 2.39</td><td>81.45 ± 1.94</td><td>82.61 ± 2.05</td><td>83.23 ± 1.99</td></tr><tr><td>q-FFL</td><td>87.07 ± 2.01</td><td>17.89 ± 1.95</td><td>80.48 ± 2.05</td><td>81.82 ± 1.83</td><td>82.59 ± 2.30</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>88.13 ± 1.16</td><td>9.30 ± 1.95</td><td>82.85 ± 2.00</td><td>83.94 ± 1.98</td><td>84.75 ± 2.00</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>86.78 ± 1.67</td><td>11.90 ± 2.03</td><td>81.09 ± 1.89</td><td>81.82 ± 2.37</td><td>82.55 ± 2.51</td></tr><tr><td>FedMeta</td><td>85.41 ± 1.88</td><td>15.05 ± 2.35</td><td>79.15 ± 2.04</td><td>79.94 ± 1.96</td><td>80.81 ± 2.21</td></tr><tr><td>q-FFL</td><td>85.92 ± 2.05</td><td>12.11 ± 1.97</td><td>79.03 ± 1.95</td><td>80.55 ± 2.71</td><td>81.49 ± 2.33</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>86.84 ± 1.53</td><td>11.16 ± 2.03</td><td>82.06 ± 1.91</td><td>82.85 ± 2.33</td><td>83.43 ± 2.07</td></tr></table>

Table 9: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario I, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 30)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>86.52 ± 1.55</td><td>13.54 ± 2.07</td><td>80.85 ± 1.35</td><td>81.76 ± 1.69</td><td>82.55 ± 1.93</td></tr><tr><td>FedMeta</td><td>87.65 ± 2.01</td><td>13.47 ± 2.25</td><td>81.58 ± 1.94</td><td>82.91 ± 2.71</td><td>83.64 ± 1.33</td></tr><tr><td>q-FFL</td><td>86.40 ± 1.39</td><td>15.68 ± 1.87</td><td>79.27 ± 1.85</td><td>80.12 ± 1.97</td><td>81.45 ± 1.64</td></tr><tr><td>CoPreFL (γ = 0.0 )</td><td>87.90 ± 1.39</td><td>11.16 ± 1.96</td><td>82.67 ± 1.81</td><td>83.58 ± 1.73</td><td>84.36 ± 1.87</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>86.78 ± 1.72</td><td>12.32 ± 1.94</td><td>80.85 ± 2.03</td><td>81.76 ± 1.88</td><td>82.55 ± 1.65</td></tr><tr><td>FedMeta</td><td>85.87 ± 1.49</td><td>17.22 ± 2.22</td><td>81.58 ± 2.31</td><td>82.09 ± 2.25</td><td>82.64 ± 2.07</td></tr><tr><td>q-FFL</td><td>85.77 ± 1.61</td><td>13.40 ± 1.88</td><td>79.27 ± 1.74</td><td>80.12 ± 2.35</td><td>81.45 ± 2.41</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>87.04 ± 1.52</td><td>9.18 ± 1.64</td><td>81.70 ± 1.93</td><td>82.12 ± 1.99</td><td>82.91 ± 2.07</td></tr><tr><td colspan="2">Pre-training (Scenario I, |m| = 15)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>81.46 ± 2.39</td><td>62.09 ± 2.95</td><td>68.87 ± 3.01</td><td>71.12 ± 2.79</td><td>72.76 ± 2.95</td></tr><tr><td>FedMeta</td><td>81.20 ± 3.07</td><td>63.84 ± 3.35</td><td>69.39 ± 2.38</td><td>71.52 ± 2.74</td><td>73.14 ± 3.06</td></tr><tr><td>q-FFL</td><td>83.45 ± 2.88</td><td>39.94 ± 3.05</td><td>69.95 ± 2.96</td><td>73.66 ± 2.57</td><td>75.43 ± 3.15</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>84.79 ± 2.61</td><td>37.09 ± 2.83</td><td>72.71 ± 2.54</td><td>74.80 ± 2.61</td><td>76.75 ± 2.66</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>83.76 ± 2.54</td><td>51.84 ± 3.08</td><td>69.50 ± 2.69</td><td>72.80 ± 3.33</td><td>74.30 ± 3.01</td></tr><tr><td>FedMeta</td><td>82.65 ± 3.37</td><td>39.19 ± 3.52</td><td>69.39 ± 3.07</td><td>72.76 ± 2.94</td><td>74.87 ± 2.85</td></tr><tr><td>q-FFL</td><td>82.00 ± 2.65</td><td>53.00 ± 3.19</td><td>70.78 ± 2.77</td><td>73.03 ± 2.93</td><td>74.22 ± 2.85</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>84.55 ± 2.39</td><td>38.07 ± 3.45</td><td>71.47 ± 2.63</td><td>73.40 ± 3.05</td><td>75.20 ± 2.91</td></tr></table>

Table 10: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario I, on the CIFAR-100 dataset.

Table 11: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario I, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>84.20 ± 2.61</td><td>57.15 ± 2.38</td><td>68.43 ± 2.95</td><td>71.83 ± 3.22</td><td>74.38 ± 3.07</td></tr><tr><td>FedMeta</td><td>83.80 ± 2.95</td><td>42.64 ± 2.66</td><td>72.30 ± 3.07</td><td>73.79 ± 2.95</td><td>75.47 ± 3.14</td></tr><tr><td>q-FFL</td><td>82.60 ± 3.05</td><td>45.56 ± 2.26</td><td>70.46 ± 3.15</td><td>73.41 ± 3.07</td><td>75.14 ± 3.33</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>84.36 ± 2.35</td><td>38.56 ± 2.33</td><td>73.66 ± 2.89</td><td>75.63 ± 3.07</td><td>77.40 ± 3.11</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>78.96 ± 2.98</td><td>64.80 ± 3.01</td><td>62.70 ± 3.35</td><td>67.00 ± 2.95</td><td>69.80 ± 3.09</td></tr><tr><td>FedMeta</td><td>82.45 ± 3.07</td><td>48.72 ± 2.84</td><td>68.97 ± 3.04</td><td>72.41 ± 3.06</td><td>74.35 ± 2.98</td></tr><tr><td>q-FFL</td><td>80.01 ± 2.67</td><td>88.92 ± 3.31</td><td>64.39 ± 2.95</td><td>67.48 ± 2.67</td><td>70.30 ± 3.08</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>83.29 ± 2.61</td><td>34.69 ± 3.17</td><td>71.58 ± 3.00</td><td>73.20 ± 2.98</td><td>74.59 ± 2.85</td></tr></table>

Table 12: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario I, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 25)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>84.02 ± 3.17</td><td>51.98 ± 2.98</td><td>71.26 ± 3.05</td><td>73.21 ± 2.88</td><td>75.57 ± 2.54</td></tr><tr><td>FedMeta</td><td>82.44 ± 3.05</td><td>55.06 ± 3.51</td><td>68.53 ± 3.29</td><td>71.73 ± 3.74</td><td>73.95 ± 3.85</td></tr><tr><td>q-FFL</td><td>82.63 ± 2.94</td><td>47.20 ± 3.08</td><td>70.52 ± 3.37</td><td>72.42 ± 3.49</td><td>74.01 ± 3.75</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>85.60 ± 2.87</td><td>37.45 ± 3.04</td><td>74.42 ± 3.11</td><td>76.53 ± 3.29</td><td>78.43 ± 3.57</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>82.01 ± 3.32</td><td>39.82 ± 3.09</td><td>70.75 ± 3.46</td><td>73.02 ± 3.81</td><td>74.63 ± 3.68</td></tr><tr><td>FedMeta</td><td>84.02 ± 2.77</td><td>39.56 ± 2.94</td><td>71.86 ± 2.61</td><td>75.17 ± 2.98</td><td>76.80 ± 2.74</td></tr><tr><td>q-FFL</td><td>82.18 ± 3.04</td><td>46.79 ± 2.85</td><td>70.53 ± 2.79</td><td>72.43 ± 3.34</td><td>73.61 ± 3.49</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>85.72 ± 3.00</td><td>29.38 ± 2.67</td><td>75.81 ± 2.83</td><td>77.24 ± 2.71</td><td>78.54 ± 3.05</td></tr></table>

Table 13: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario I, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 30)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>79.60 ± 3.00</td><td>83.36 ± 3.41</td><td>62.68 ± 3.82</td><td>65.87 ± 3.59</td><td>68.57 ± 3.33</td></tr><tr><td>FedMeta</td><td>79.90 ± 2.98</td><td>48.02 ± 3.61</td><td>67.01 ± 2.65</td><td>69.69 ± 3.01</td><td>71.46 ± 2.85</td></tr><tr><td>q-FFL</td><td>83.02 ± 2.65</td><td>52.27 ± 3.04</td><td>70.64 ± 2.88</td><td>72.71 ± 2.73</td><td>74.50 ± 2.71</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>83.48 ± 2.39</td><td>45.16 ± 3.45</td><td>70.80 ± 3.01</td><td>72.72 ± 2.87</td><td>74.59 ± 2.72</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>81.79 ± 3.41</td><td>50.84 ± 3.85</td><td>69.70 ± 2.93</td><td>72.08 ± 2.88</td><td>74.20 ± 3.07</td></tr><tr><td>FedMeta</td><td>82.61 ± 3.54</td><td>43.43 ± 3.95</td><td>71.84 ± 3.39</td><td>73.30 ± 3.86</td><td>74.68 ± 3.51</td></tr><tr><td>q-FFL</td><td>82.68 ± 2.94</td><td>54.17 ± 3.33</td><td>68.68 ± 3.06</td><td>72.08 ± 3.17</td><td>74.06 ± 2.84</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>83.48 ± 2.87</td><td>40.20 ± 3.07</td><td>72.83 ± 3.16</td><td>74.29 ± 2.94</td><td>75.80 ± 2.98</td></tr><tr><td colspan="2">Pre-training (Scenario I, |m| = 15)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>84.27 ± 1.37</td><td>16.48 ± 1.61</td><td>77.20 ± 1.19</td><td>78.43 ± 1.53</td><td>79.27 ± 1.55</td></tr><tr><td>FedMeta</td><td>84.15 ± 1.28</td><td>19.27 ± 2.07</td><td>78.50 ± 1.39</td><td>79.65 ± 1.21</td><td>80.66 ± 1.09</td></tr><tr><td>q-FFL</td><td>84.24 ± 0.95</td><td>17.64 ± 1.23</td><td>77.20 ± 1.09</td><td>78.86 ± 1.10</td><td>79.99 ± 1.38</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>85.05 ± 1.07</td><td>15.21 ± 1.06</td><td>79.08 ± 1.08</td><td>80.38 ± 1.13</td><td>81.19 ± 1.19</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>85.19 ± 1.52</td><td>15.13 ± 1.83</td><td>77.78 ± 1.66</td><td>79.29 ± 1.59</td><td>80.38 ± 1.58</td></tr><tr><td>FedMeta</td><td>85.35 ± 1.39</td><td>15.60 ± 1.94</td><td>78.50 ± 1.43</td><td>80.01 ± 1.37</td><td>81.00 ± 1.57</td></tr><tr><td>q-FFL</td><td>85.91 ± 1.44</td><td>15.76 ± 2.03</td><td>78.22 ± 1.65</td><td>80.38 ± 1.37</td><td>81.12 ± 1.59</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>86.39 ± 1.27</td><td>10.63 ± 1.89</td><td>79.08 ± 1.33</td><td>80.45 ± 1.45</td><td>81.24 ± 1.49</td></tr></table>

Table 14: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario I, on the CIFAR-100 dataset.

Table 15: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario I, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>85.74 ± 1.69</td><td>17.39 ± 2.05</td><td>77.20 ± 1.57</td><td>78.79 ± 1.77</td><td>79.80 ± 1.83</td></tr><tr><td>FedMeta</td><td>85.56 ± 1.93</td><td>17.64 ± 1.33</td><td>77.92 ± 2.19</td><td>79.51 ± 1.84</td><td>80.57 ± 1.74</td></tr><tr><td>q-FFL</td><td>84.64 ± 2.07</td><td>21.07 ± 1.35</td><td>78.79 ± 1.66</td><td>79.74 ± 1.94</td><td>80.91 ± 2.19</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>86.03 ± 1.32</td><td>13.99 ± 1.14</td><td>79.08 ± 1.65</td><td>80.09 ± 1.33</td><td>81.24 ± 1.46</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>85.43 ± 1.38</td><td>17.31 ± 2.55</td><td>77.49 ± 1.54</td><td>79.65 ± 1.69</td><td>80.76 ± 1.83</td></tr><tr><td>FedMeta</td><td>84.16 ± 2.00</td><td>16.89 ± 2.85</td><td>77.20 ± 2.38</td><td>79.73 ± 2.19</td><td>81.05 ± 2.22</td></tr><tr><td>q-FFL</td><td>85.83 ± 1.10</td><td>19.18 ± 1.95</td><td>78.07 ± 1.77</td><td>79.37 ± 1.54</td><td>80.62 ± 1.39</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>86.00 ± 1.13</td><td>16.16 ± 2.00</td><td>79.37 ± 1.38</td><td>80.30 ± 1.60</td><td>81.19 ± 1.54</td></tr></table>

Table 16: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario I, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 25)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>85.24 ± 1.19</td><td>19.71 ± 2.07</td><td>78.35 ± 1.38</td><td>79.87 ± 1.74</td><td>81.05 ± 1.23</td></tr><tr><td>FedMeta</td><td>85.19 ± 1.44</td><td>22.00 ± 2.35</td><td>78.21 ± 1.68</td><td>79.73 ± 1.54</td><td>80.71 ± 1.70</td></tr><tr><td>q-FFL</td><td>85.26 ± 2.04</td><td>16.89 ± 2.85</td><td>78.50 ± 2.39</td><td>79.94 ± 2.17</td><td>81.00 ± 2.43</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>85.74 ± 1.13</td><td>12.81 ± 1.96</td><td>79.84 ± 1.30</td><td>80.68 ± 1.63</td><td>81.49 ± 1.69</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>85.47 ± 1.38</td><td>14.36 ± 2.41</td><td>77.63 ± 1.69</td><td>79.29 ± 1.35</td><td>80.33 ± 1.58</td></tr><tr><td>FedMeta</td><td>85.74 ± 1.63</td><td>17.64 ± 1.94</td><td>77.92 ± 1.47</td><td>79.80 ± 1.83</td><td>81.10 ± 1.79</td></tr><tr><td>q-FFL</td><td>85.82 ± 2.06</td><td>17.64 ± 2.58</td><td>79.08 ± 1.99</td><td>80.52 ± 1.87</td><td>81.58 ± 2.11</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>86.25 ± 1.47</td><td>12.96 ± 1.33</td><td>79.87 ± 1.54</td><td>80.99 ± 1.35</td><td>81.68 ± 1.41</td></tr></table>

Table 17: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario I, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 30)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>85.71 ± 1.92</td><td>14.90 ± 1.61</td><td>79.08 ± 1.94</td><td>80.66 ± 2.38</td><td>81.53 ± 1.77</td></tr><tr><td>FedMeta</td><td>85.48 ± 2.07</td><td>15.92 ± 2.22</td><td>79.51 ± 1.85</td><td>80.16 ± 2.03</td><td>81.19 ± 2.25</td></tr><tr><td>q-FFL</td><td>85.95 ± 2.17</td><td>21.25 ± 1.54</td><td>77.34 ± 2.08</td><td>78.93 ± 2.34</td><td>80.13 ± 2.01</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>86.05 ± 1.85</td><td>14.90 ± 1.56</td><td>80.31 ± 1.73</td><td>81.49 ± 1.99</td><td>82.27 ± 1.86</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>85.64 ± 1.37</td><td>21.07 ± 1.94</td><td>75.90 ± 1.53</td><td>77.99 ± 1.49</td><td>79.32 ± 1.55</td></tr><tr><td>FedMeta</td><td>85.90 ± 1.55</td><td>17.89 ± 2.37</td><td>80.23 ± 1.69</td><td>81.10 ± 1.43</td><td>81.87 ± 1.79</td></tr><tr><td>q-FFL</td><td>86.49 ± 1.45</td><td>14.75 ± 2.00</td><td>78.79 ± 1.67</td><td>80.16 ± 1.32</td><td>81.24 ± 1.45</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>86.51 ± 1.30</td><td>14.06 ± 1.73</td><td>80.74 ± 1.35</td><td>81.35 ± 1.47</td><td>82.16 ± 1.60</td></tr><tr><td colspan="2">Pre-training (Scenario I, |m| = 15)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>78.88 ± 2.66</td><td>64.16 ± 2.05</td><td>67.03 ± 2.39</td><td>68.42 ± 3.01</td><td>69.95 ± 2.85</td></tr><tr><td>FedMeta</td><td>82.62 ± 2.74</td><td>43.16 ± 2.71</td><td>70.76 ± 2.55</td><td>73.48 ± 2.38</td><td>74.48 ± 2.96</td></tr><tr><td>q-FFL</td><td>83.58 ± 3.00</td><td>49.70 ± 2.89</td><td>67.11 ± 2.93</td><td>71.38 ± 2.81</td><td>73.78 ± 3.05</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>83.83 ± 2.54</td><td>41.22 ± 2.17</td><td>73.28 ± 2.41</td><td>74.39 ± 2.30</td><td>75.50 ± 2.88</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>82.19 ± 2.71</td><td>38.32 ± 1.94</td><td>72.64 ± 2.10</td><td>73.90 ± 1.98</td><td>75.38 ± 2.53</td></tr><tr><td>FedMeta</td><td>81.45 ± 2.39</td><td>53.73 ± 2.61</td><td>68.42 ± 2.55</td><td>71.17 ± 2.08</td><td>72.98 ± 2.34</td></tr><tr><td>q-FFL</td><td>82.85 ± 2.83</td><td>32.26 ± 3.01</td><td>73.89 ± 3.19</td><td>76.14 ± 2.95</td><td>77.28 ± 2.91</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>83.65 ± 2.39</td><td>25.81 ± 2.33</td><td>75.41 ± 2.37</td><td>76.45 ± 2.10</td><td>77.73 ± 2.03</td></tr></table>

Table 18: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario I, on the Tiny-ImageNet dataset.

Table 19: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario I, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>79.45 ± 2.33</td><td>35.40 ± 2.79</td><td>64.86 ± 2.51</td><td>68.61 ± 2.49</td><td>70.78 ± 2.55</td></tr><tr><td>FedMeta</td><td>81.68 ± 2.57</td><td>65.61 ± 2.38</td><td>65.96 ± 2.61</td><td>69.17 ± 2.55</td><td>71.59 ± 2.73</td></tr><tr><td>q-FFL</td><td>82.65 ± 2.65</td><td>39.69 ± 2.59</td><td>70.65 ± 2.47</td><td>74.14 ± 2.54</td><td>76.27 ± 2.89</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>83.79 ± 2.30</td><td>34.93 ± 2.44</td><td>72.59 ± 2.49</td><td>75.05 ± 2.38</td><td>76.76 ± 2.41</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>82.94 ± 2.59</td><td>37.21 ± 2.81</td><td>68.99 ± 2.43</td><td>72.29 ± 2.61</td><td>74.40 ± 2.69</td></tr><tr><td>FedMeta</td><td>81.03 ± 2.86</td><td>37.58 ± 3.00</td><td>69.44 ± 2.61</td><td>71.55 ± 2.93</td><td>72.93 ± 2.78</td></tr><tr><td>q-FFL</td><td>84.11 ± 2.49</td><td>43.96 ± 2.71</td><td>73.87 ± 2.79</td><td>76.05 ± 2.61</td><td>77.37 ± 2.45</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>85.23 ± 2.43</td><td>35.40 ± 2.75</td><td>76.77 ± 2.58</td><td>78.46 ± 2.47</td><td>79.86 ± 2.53</td></tr></table>

Table 20: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario I, on the Tiny-ImageNet dataset. 

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 25)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>83.71 ± 2.89</td><td>50.41 ± 2.94</td><td>69.91 ± 2.81</td><td>73.50 ± 2.97</td><td>75.40 ± 3.09</td></tr><tr><td>FedMeta</td><td>84.19 ± 2.61</td><td>42.90 ± 2.65</td><td>73.77 ± 2.53</td><td>76.22 ± 2.88</td><td>77.77 ± 2.97</td></tr><tr><td>q-FFL</td><td>80.11 ± 3.19</td><td>55.20 ± 3.07</td><td>65.45 ± 2.95</td><td>68.54 ± 3.28</td><td>70.72 ± 2.97</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>84.29 ± 2.66</td><td>36.60 ± 2.83</td><td>76.02 ± 2.39</td><td>77.56 ± 3.01</td><td>78.95 ± 2.88</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>79.08 ± 2.54</td><td>55.80 ± 3.05</td><td>66.80 ± 2.79</td><td>69.06 ± 2.81</td><td>71.38 ± 2.53</td></tr><tr><td>FedMeta</td><td>81.58 ± 2.90</td><td>38.07 ± 2.88</td><td>70.86 ± 3.17</td><td>72.83 ± 2.84</td><td>74.39 ± 3.05</td></tr><tr><td>q-FFL</td><td>83.16 ± 3.03</td><td>45.56 ± 3.15</td><td>72.39 ± 2.98</td><td>75.29 ± 3.07</td><td>77.09 ± 2.88</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>83.87 ± 2.61</td><td>25.60 ± 2.94</td><td>75.16 ± 2.89</td><td>76.87 ± 2.74</td><td>78.05 ± 2.83</td></tr></table>

Table 21: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario I, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 30)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>80.37 ± 3.07</td><td>43.56 ± 2.93</td><td>69.27 ± 2.95</td><td>70.91 ± 2.88</td><td>72.44 ± 3.19</td></tr><tr><td>FedMeta</td><td>80.51 ± 2.87</td><td>44.09 ± 3.11</td><td>68.05 ± 2.79</td><td>70.74 ± 2.95</td><td>72.18 ± 2.97</td></tr><tr><td>q-FFL</td><td>81.89 ± 3.10</td><td>45.97 ± 2.87</td><td>68.85 ± 3.31</td><td>72.07 ± 3.19</td><td>73.99 ± 2.93</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>83.17 ± 2.65</td><td>31.81 ± 2.88</td><td>71.16 ± 2.85</td><td>73.64 ± 2.79</td><td>75.49 ± 2.80</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>82.73 ± 2.74</td><td>42.51 ± 3.11</td><td>72.90 ± 2.58</td><td>74.84 ± 2.99</td><td>76.50 ± 2.73</td></tr><tr><td>FedMeta</td><td>82.58 ± 2.38</td><td>34.81 ± 3.00</td><td>71.67 ± 2.59</td><td>74.39 ± 2.41</td><td>75.85 ± 2.51</td></tr><tr><td>q-FFL</td><td>83.39 ± 2.66</td><td>38.07 ± 2.95</td><td>72.60 ± 2.98</td><td>74.97 ± 2.73</td><td>76.57 ± 2.87</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>84.25 ± 2.35</td><td>30.11 ± 2.47</td><td>76.18 ± 2.31</td><td>77.54 ± 2.60</td><td>78.73 ± 2.47</td></tr></table>

Table 22: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario I, on the Tiny-ImageNet dataset.

<table><tr><td>Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td>FedAvg</td><td>81.35</td><td>59.69</td><td>70.35</td><td>70.91</td><td>71.63</td></tr><tr><td>FedMeta</td><td>82.69</td><td>46.38</td><td>72.74</td><td>73.51</td><td>74.39</td></tr><tr><td>q-FFL</td><td>83.27</td><td>51.48</td><td>73.01</td><td>73.82</td><td>75.02</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>84.71</td><td>39.26</td><td>73.28</td><td>74.99</td><td>75.35</td></tr></table>

Table 23: Performance of 20-way classification downstream FedAvg, initialized with various non-IID FL pre-trained methods in scenario I, on the CIFAR-100 dataset.

![](images/fa24f1a3ec3eacd8268d83c6f938a331613cdd4dac2ba1f74600345bdec3d750.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | q-FFL(Var runner-up) | CoPreFL |
|----------------------|------------------------|-----------------------|---------|
| 75                   | 1                      | 0                     | 0       |
| 80                   | 5                      | 2                     | 3       |
| 85                   | 15                     | 10                    | 12      |
| 90                   | 20                     | 15                    | 35      |
| 95                   | 10                     | 5                     | 10      |
| 100                  | 0                      | 0                     | 0       |
</details>

(a) Pre-training: 20 clients IID FL

![](images/d4f0faa50602f6443c7f3eef54b4d389bb40d995c8dd556709dbc072d0066b46.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Var runner-up) | q-FFL(Acc runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 75                   | 0                      | 0                    | 0       |
| 80                   | 5                      | 2                    | 1       |
| 85                   | 15                     | 10                   | 20      |
| 90                   | 10                     | 5                    | 35      |
| 95                   | 0                      | 0                    | 20      |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/448c3ac58a5bd68a05b8b8fc734cfaa9029ee39a60c4f7fc19a14b97475d5889.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta(Acc+Var runner-up) | CoPrefL |
| --------------------- | --------------------------- | ------- |
| 75                    | 0                           | 0       |
| 80                    | 5                           | 2       |
| 85                    | 15                          | 30      |
| 90                    | 20                          | 30      |
| 95                    | 10                          | 0       |
| 100                   | 0                           | 1       |
</details>

(c) Pre-training: 30 clients IID FL

![](images/5b44fd2c6aec3e0021bb7ffda59b6f8793ffbaea2809a438c8492ccf0841a06c.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg (Acc+Var runner-up) | CoPreFL |
| --------------------- | -------------------------- | ------- |
| 75                    | 1                          | 0       |
| 80                    | 10                         | 5       |
| 85                    | 25                         | 15      |
| 90                    | 25                         | 20      |
| 95                    | 5                          | 3       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 3: The distributions of testing accuracy in IID FL downstream tasks under various pre-training setups in scenario I on the CIFAR-100 dataset.

![](images/a160b81d2cc067d160c875cf33379bd2dc9af20fe3aa05da85c76ab87b8ffa26.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 50                   | 0                      | 0                       | 0       |
| 60                   | 2                      | 1                       | 0       |
| 70                   | 5                      | 3                       | 1       |
| 80                   | 10                     | 8                       | 5       |
| 90                   | 17                     | 15                      | 12      |
| 100                  | 0                      | 0                       | 0       |
</details>

(a) Pre-training: 20 clients IID FL

![](images/1b33ebfadea55658b5e5d3309e71cdf0d86170745182360966ae18288dd0a47f.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedMeta(Acc+Var runner-up) | CoPreFL |
| --------------------- | --------------------------- | ------- |
| 50                    | 1                           | 0       |
| 60                    | 2                           | 1       |
| 70                    | 4                           | 3       |
| 80                    | 12                          | 8       |
| 90                    | 14                          | 12      |
| 100                   | 0                           | 0       |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/8b34895832bae46262119d1d54a57f50e311421cdfa4c2b1aa8015d9a9fb9643.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta Var (runner-up) | q-FFL (Acc runner-up) | CoPreFL |
|----------------------|--------------------------|------------------------|---------|
| 30                   | 0                        | 0                      | 0       |
| 40                   | 0                        | 0                      | 0       |
| 50                   | 0                        | 0                      | 0       |
| 60                   | 2                        | 0                      | 0       |
| 70                   | 5                        | 0                      | 0       |
| 80                   | 10                       | 0                      | 15      |
| 90                   | 15                       | 12                     | 15      |
| 100                  | 0                        | 0                      | 0       |
</details>

(c) Pre-training: 30 clients IID FL

![](images/f9a63bd8c85c938477129f11dd048a8de1fb4df8f9beaae5bf45ea42a44a5f1f.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta(Var runner-up) | q-FFL(Acc runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 40                   | 0                      | 0                    | 0       |
| 50                   | 1                      | 2                    | 0       |
| 60                   | 0                      | 0                    | 0       |
| 70                   | 0                      | 0                    | 0       |
| 80                   | 1                      | 1                    | 1       |
| 90                   | 14                     | 12                   | 10      |
| 100                  | 0                      | 0                    | 0       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 4: The distributions of testing accuracy in non-IID FL downstream tasks under various pre-training setups in scenario I on the CIFAR-100 dataset.

![](images/d3c3a3aa0079fd247a55856814ce3c8ff489d2e5ce1b33e027a38764c3f262ff.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc+Var runner-up) | CoPreFL |
|----------------------|---------------------------|---------|
| 70                   | 0                         | 0       |
| 75                   | 0                         | 0       |
| 80                   | 20                        | 10      |
| 85                   | 20                        | 30      |
| 90                   | 20                        | 15      |
| 95                   | 0                         | 5       |
| 100                  | 0                         | 0       |
</details>

(a) Pre-training: 20 clients IID FL

![](images/93049afba990cbecf047bf878a73ee1346aaa9f252ed00fb60fcfd87a3918d3b.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedMeta(Var runner-up) | q-FFL(Acc runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 65                   | 0                      | 0                    | 0       |
| 70                   | 0                      | 5                    | 0       |
| 75                   | 0                      | 10                   | 5       |
| 80                   | 0                      | 20                   | 10      |
| 85                   | 25                     | 30                   | 20      |
| 90                   | 25                     | 25                   | 35      |
| 95                   | 0                      | 0                    | 0       |
| 100                  | 0                      | 0                    | 0       |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/40bbf5572abb0723e3d47ea22dcde6325ada9997d4edf78fe2f5f271f2389cce.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Var runner-up) | q-FFL(Acc runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 65                   | 0                      | 0                    | 0       |
| 70                   | 1                      | 0                    | 0       |
| 75                   | 3                      | 2                    | 1       |
| 80                   | 14                     | 8                    | 15      |
| 85                   | 22                     | 15                   | 22      |
| 90                   | 15                     | 10                   | 15      |
| 95                   | 3                      | 0                    | 3       |
</details>

(c) Pre-training: 30 clients IID FL

![](images/fa146f8e942b71d8866f246faee74d552f9ec9baf011e7283360277699c1ac06.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | q-FFL(Acc+Var runner-up) | CoPreFL |
|----------------------|--------------------------|---------|
| 70                   | 0                        | 0       |
| 75                   | 2                        | 3       |
| 80                   | 5                        | 8       |
| 85                   | 24                       | 30      |
| 90                   | 21                       | 16      |
| 95                   | 10                       | 5       |
| 100                  | 0                        | 0       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 5: The distributions of testing accuracy in IID FL downstream tasks under various pre-training setups in scenario I on the Tiny-ImageNet dataset.

![](images/d576b8da85e31e9a255f9bbda8f2bf56a5757255522585845604b6e409962357.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg Var (runner-up) | q-FFL (Acc runner-up) | CoPreFL |
|----------------------|------------------------|-----------------------|---------|
| 60                   | 1                      | 1                     | 1       |
| 70                   | 5                      | 3                     | 2       |
| 80                   | 8                      | 12                    | 10      |
| 90                   | 10                     | 14                    | 20      |
| 100                  | 1                      | 1                     | 1       |
</details>

(a) Pre-training: 20 clients IID FL

![](images/ab693fd3aed1c5b3afd9cb4a8d0b19777fc519c39a425b20bbe0ef0d1e42b572.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Var runner-up) | q-FFL(Acc runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 50                   | 1                      | 0                    | 0       |
| 60                   | 2                      | 0                    | 0       |
| 70                   | 3                      | 1                    | 0       |
| 80                   | 5                      | 5                    | 1       |
| 90                   | 15                     | 10                   | 15      |
| 100                  | 2                      | 0                    | 2       |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/a42b5a111267b567497c4ae8f0ce2cae405b91152d476b9f000df391f4a7e15c.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Var runner-up) | q-FFL(Acc runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 50                   | 0                      | 0                    | 0       |
| 60                   | 2                      | 1                    | 0       |
| 70                   | 4                      | 3                    | 2       |
| 80                   | 6                      | 5                    | 4       |
| 90                   | 8                      | 7                    | 6       |
| 100                  | 10                     | 9                    | 8       |
</details>

(c) Pre-training: 30 clients IID FL

![](images/3fc8576d43ee231c44e1da1eb2c1e1adc715505f9bfa4fec58dd1f09adfa0ec5.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedMetal Var runner-up | q-FFL(Acc) runner-up | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 50                   | 0                      | 0                    | 0       |
| 60                   | 0                      | 0                    | 0       |
| 70                   | 5                      | 5                    | 5       |
| 80                   | 15                     | 15                   | 15      |
| 90                   | 10                     | 20                   | 10      |
| 100                  | 0                      | 0                    | 0       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 6: The distributions of testing accuracy in non-IID FL downstream tasks under various pre-training setups in scenario I on the Tiny-ImageNet dataset

<table><tr><td rowspan="2">Pre-training (PACS)</td><td colspan="8">Downstream: Non-IID FedAvg</td></tr><tr><td colspan="2">Sketch</td><td colspan="2">Art</td><td colspan="2">Cartoon</td><td colspan="2">Photo</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>FedAvg</td><td>62.23 ± 2.65</td><td>51.29 ± 3.11</td><td>70.99 ± 3.05</td><td>49.32 ± 3.33</td><td>64.39 ± 2.96</td><td>69.31 ± 2.84</td><td>77.24 ± 2.75</td><td>61.35 ± 3.06</td></tr><tr><td>FedMeta</td><td>64.35 ± 3.07</td><td>44.38 ± 2.98</td><td>73.26 ± 2.64</td><td>44.61 ± 3.00</td><td>62.17 ± 3.16</td><td>47.95 ± 3.08</td><td>76.59 ± 2.37</td><td>70.38 ± 3.11</td></tr><tr><td>q-FFL</td><td>60.79 ± 3.15</td><td>27.96 ± 3.03</td><td>73.79 ± 2.99</td><td>30.11 ± 2.95</td><td>66.94 ± 2.63</td><td>53.22 ± 2.99</td><td>80.63 ± 3.00</td><td>54.44 ± 2.98</td></tr><tr><td>CoPreFL</td><td>66.83 ± 2.85</td><td>24.31 ± 2.83</td><td>75.19 ± 2.71</td><td>26.36 ± 3.02</td><td>68.33 ± 2.51</td><td>39.26 ± 3.13</td><td>82.19 ± 2.80</td><td>46.32 ± 3.05</td></tr></table>

Table 24: The results of domain shifts scenario using the PACS dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 15)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>87.02 ± 1.17</td><td>12.82 ± 1.35</td><td>82.42 ± 1.44</td><td>83.33 ± 1.30</td><td>84.04 ± 1.26</td></tr><tr><td>FedMeta</td><td>87.07 ± 1.28</td><td>10.76 ± 1.95</td><td>81.94 ± 1.36</td><td>82.73 ± 1.20</td><td>83.43 ± 1.49</td></tr><tr><td>q-FFL</td><td>87.27 ± 1.33</td><td>13.69 ± 1.61</td><td>81.21 ± 1.29</td><td>82.30 ± 1.37</td><td>82.95 ± 1.45</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>87.87 ± 1.31</td><td>13.32 ± 1.11</td><td>82.73 ± 1.35</td><td>83.58 ± 1.16</td><td>84.06 ± 1.19</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>88.58 ± 1.19</td><td>8.70 ± 1.15</td><td>83.39 ± 1.23</td><td>83.88 ± 1.25</td><td>84.69 ± 1.29</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>86.22 ± 1.28</td><td>15.44 ± 1.55</td><td>79.15 ± 1.41</td><td>80.36 ± 1.39</td><td>81.54 ± 1.46</td></tr><tr><td>FedMeta</td><td>86.09 ± 1.39</td><td>11.42 ± 1.57</td><td>80.12 ± 1.44</td><td>81.03 ± 1.58</td><td>81.98 ± 1.37</td></tr><tr><td>q-FFL</td><td>86.56 ± 1.33</td><td>15.29 ± 1.94</td><td>78.42 ± 1.61</td><td>80.00 ± 1.52</td><td>81.37 ± 1.39</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>86.73 ± 1.17</td><td>12.46 ± 1.39</td><td>80.61 ± 1.08</td><td>81.76 ± 1.23</td><td>82.63 ± 1.19</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>87.42 ± 1.23</td><td>9.06 ± 1.46</td><td>81.21 ± 1.11</td><td>82.30 ± 1.25</td><td>82.95 ± 1.27</td></tr></table>

Table 25: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario II, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 20)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>87.28 ± 1.33</td><td>15.21 ± 2.06</td><td>80.00 ± 1.49</td><td>81.21 ± 1.58</td><td>82.10 ± 1.37</td></tr><tr><td>FedMeta</td><td>87.27 ± 1.46</td><td>12.46 ± 1.77</td><td>81.45 ± 1.38</td><td>82.12 ± 1.63</td><td>82.95 ± 1.52</td></tr><tr><td>q-FFL</td><td>86.84 ± 1.29</td><td>12.74 ± 1.86</td><td>80.73 ± 1.57</td><td>82.12 ± 1.34</td><td>82.87 ± 1.39</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>87.67 ± 1.27</td><td>12.32 ± 1.49</td><td>81.82 ± 1.30</td><td>83.09 ± 1.38</td><td>83.92 ± 1.40</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>88.10 ± 1.30</td><td>9.30 ± 1.35</td><td>83.52 ± 1.42</td><td>84.30 ± 1.33</td><td>85.05 ± 1.43</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>86.39 ± 1.28</td><td>17.31 ± 1.64</td><td>79.64 ± 1.77</td><td>80.79 ± 1.65</td><td>81.78 ± 1.52</td></tr><tr><td>FedMeta</td><td>86.32 ± 1.60</td><td>12.46 ± 2.00</td><td>80.61 ± 1.83</td><td>81.45 ± 1.49</td><td>82.26 ± 1.77</td></tr><tr><td>q-FFL</td><td>86.17 ± 1.93</td><td>16.24 ± 2.31</td><td>79.27 ± 2.07</td><td>81.09 ± 1.89</td><td>82.02 ± 1.95</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>86.63 ± 1.54</td><td>11.76 ± 1.66</td><td>81.21 ± 1.65</td><td>82.00 ± 1.73</td><td>82.46 ± 1.88</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>87.02 ± 1.39</td><td>10.50 ± 1.47</td><td>81.70 ± 1.73</td><td>82.42 ± 1.52</td><td>83.23 ± 1.66</td></tr></table>

Table 26: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario II, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 25)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>87.31 ± 1.09</td><td>14.90 ± 1.93</td><td>79.39 ± 1.17</td><td>81.64 ± 1.25</td><td>82.75 ± 1.22</td></tr><tr><td>FedMeta</td><td>86.81 ± 0.94</td><td>10.89 ± 2.08</td><td>80.97 ± 1.35</td><td>82.30 ± 0.89</td><td>83.07 ± 1.15</td></tr><tr><td>q-FFL</td><td>87.36 ± 1.32</td><td>17.47 ± 1.99</td><td>80.73 ± 1.45</td><td>81.76 ± 1.39</td><td>82.87 ± 1.17</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>87.98 ± 1.00</td><td>11.22 ± 1.45</td><td>82.55 ± 1.28</td><td>83.15 ± 1.10</td><td>83.92 ± 1.04</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>88.67 ± 1.03</td><td>9.98 ± 1.35</td><td>83.88 ± 1.22</td><td>84.55 ± 1.03</td><td>85.29 ± 0.95</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>86.37 ± 1.23</td><td>15.44 ± 1.54</td><td>78.79 ± 1.37</td><td>80.12 ± 1.12</td><td>81.25 ± 1.00</td></tr><tr><td>FedMeta</td><td>85.49 ± 1.08</td><td>16.89 ± 1.25</td><td>79.27 ± 1.14</td><td>80.36 ± 1.03</td><td>81.25 ± 1.11</td></tr><tr><td>q-FFL</td><td>85.67 ± 1.39</td><td>17.06 ± 1.08</td><td>80.61 ± 1.53</td><td>81.45 ± 1.47</td><td>81.98 ± 1.62</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>86.40 ± 1.05</td><td>13.10 ± 1.30</td><td>80.62 ± 1.23</td><td>81.45 ± 1.10</td><td>82.34 ± 1.18</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>87.32 ± 1.08</td><td>11.22 ± 1.00</td><td>82.42 ± 1.35</td><td>83.27 ± 1.18</td><td>83.84 ± 1.23</td></tr></table>

Table 27: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario II, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 30)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>87.51 ± 0.95</td><td>13.76 ± 2.03</td><td>82.55 ± 1.14</td><td>83.58 ± 1.25</td><td>84.20 ± 1.08</td></tr><tr><td>FedMeta</td><td>87.25 ± 1.20</td><td>12.39 ± 1.94</td><td>81.70 ± 1.33</td><td>82.55 ± 1.45</td><td>83.03 ± 1.11</td></tr><tr><td>q-FFL</td><td>86.78 ± 1.29</td><td>13.76 ± 1.53</td><td>81.21 ± 1.47</td><td>82.06 ± 1.35</td><td>82.79 ± 1.53</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>87.75 ± 1.17</td><td>13.40 ± 1.64</td><td>81.52 ± 1.25</td><td>82.94 ± 1.08</td><td>83.70 ± 1.27</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>88.27 ± 1.04</td><td>9.06 ± 1.66</td><td>84.06 ± 1.25</td><td>84.55 ± 1.17</td><td>85.05 ± 1.08</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>86.07 ± 1.33</td><td>11.09 ± 1.94</td><td>80.61 ± 1.58</td><td>81.64 ± 1.49</td><td>82.38 ± 1.76</td></tr><tr><td>FedMeta</td><td>86.25 ± 1.07</td><td>12.96 ± 2.15</td><td>79.03 ± 1.35</td><td>80.36 ± 1.24</td><td>81.66 ± 1.19</td></tr><tr><td>q-FFL</td><td>85.50 ± 1.25</td><td>15.29 ± 2.07</td><td>77.58 ± 1.54</td><td>79.39 ± 1.39</td><td>80.57 ± 1.33</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>86.47 ± 1.16</td><td>10.96 ± 1.85</td><td>80.36 ± 1.37</td><td>81.36 ± 1.40</td><td>82.00 ± 1.29</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>87.54 ± 1.14</td><td>10.96 ± 1.80</td><td>81.09 ± 1.10</td><td>81.67 ± 1.26</td><td>82.40 ± 1.33</td></tr><tr><td colspan="2">Pre-training (Scenario II, |m| = 15)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>84.02 ± 2.73</td><td>46.79 ± 3.61</td><td>71.01 ± 2.95</td><td>74.53 ± 3.08</td><td>76.81 ± 3.21</td></tr><tr><td>FedMeta</td><td>83.47 ± 3.18</td><td>34.11 ± 3.29</td><td>73.68 ± 3.44</td><td>75.20 ± 3.71</td><td>76.35 ± 3.25</td></tr><tr><td>q-FFL</td><td>85.03 ± 2.95</td><td>35.64 ± 2.79</td><td>74.04 ± 3.07</td><td>76.39 ± 3.11</td><td>78.12 ± 2.94</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>85.04 ± 2.80</td><td>35.64 ± 2.45</td><td>74.61 ± 2.88</td><td>76.40 ± 2.98</td><td>78.34 ± 3.01</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>85.08 ± 2.62</td><td>31.70 ± 2.45</td><td>74.61 ± 2.76</td><td>76.87 ± 3.00</td><td>78.63 ± 3.04</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>82.91 ± 3.25</td><td>41.99 ± 3.68</td><td>71.90 ± 3.51</td><td>75.23 ± 3.73</td><td>76.81 ± 3.49</td></tr><tr><td>FedMeta</td><td>78.77 ± 2.99</td><td>70.39 ± 3.37</td><td>65.13 ± 3.41</td><td>67.47 ± 3.29</td><td>69.28 ± 2.95</td></tr><tr><td>q-FFL</td><td>80.94 ± 3.07</td><td>49.42 ± 2.84</td><td>69.57 ± 3.39</td><td>71.46 ± 2.95</td><td>72.86 ± 3.37</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>83.42 ± 3.03</td><td>40.20 ± 2.97</td><td>73.09 ± 3.19</td><td>74.54 ± 3.07</td><td>76.29 ± 2.88</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>83.83 ± 3.00</td><td>39.31 ± 3.04</td><td>74.26 ± 2.92</td><td>76.42 ± 3.10</td><td>78.10 ± 3.04</td></tr></table>

Table 28: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario II, on the CIFAR-100 dataset.

Table 29: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario II, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 20)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>81.79 ± 3.55</td><td>41.73 ± 2.89</td><td>69.84 ± 3.19</td><td>73.47 ± 3.24</td><td>75.11 ± 3.88</td></tr><tr><td>FedMeta</td><td>82.29 ± 3.19</td><td>47.75 ± 2.74</td><td>71.69 ± 3.07</td><td>74.17 ± 2.99</td><td>75.71 ± 2.81</td></tr><tr><td>q-FFL</td><td>82.40 ± 2.99</td><td>40.32 ± 3.15</td><td>73.96 ± 3.08</td><td>75.30 ± 2.87</td><td>76.59 ± 3.12</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>82.90 ± 2.79</td><td>38.94 ± 2.35</td><td>73.02 ± 2.89</td><td>75.60 ± 3.05</td><td>77.18 ± 2.94</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>85.68 ± 2.51</td><td>27.14 ± 2.43</td><td>75.36 ± 2.66</td><td>77.25 ± 2.97</td><td>78.49 ± 2.69</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>82.82 ± 3.17</td><td>49.00 ± 3.41</td><td>69.71 ± 3.25</td><td>72.54 ± 3.30</td><td>74.58 ± 3.21</td></tr><tr><td>FedMeta</td><td>82.69 ± 3.05</td><td>48.44 ± 2.99</td><td>68.84 ± 3.14</td><td>71.82 ± 3.27</td><td>74.14 ± 3.09</td></tr><tr><td>q-FFL</td><td>82.14 ± 2.76</td><td>73.10 ± 3.08</td><td>68.22 ± 3.00</td><td>70.64 ± 2.85</td><td>73.77 ± 3.14</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>83.63 ± 3.00</td><td>41.73 ± 2.85</td><td>69.76 ± 2.94</td><td>73.46 ± 3.09</td><td>75.64 ± 3.15</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>86.63 ± 2.93</td><td>31.58 ± 2.64</td><td>73.05 ± 2.51</td><td>75.82 ± 2.88</td><td>77.58 ± 2.99</td></tr></table>

Table 30: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario II, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 25)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>80.53 ± 2.95</td><td>62.57 ± 3.11</td><td>66.51 ± 3.05</td><td>68.54 ± 3.19</td><td>70.78 ± 2.98</td></tr><tr><td>FedMeta</td><td>82.37 ± 3.00</td><td>45.97 ± 2.54</td><td>70.68 ± 2.98</td><td>73.40 ± 3.01</td><td>75.21 ± 3.12</td></tr><tr><td>q-FFL</td><td>82.06 ± 2.77</td><td>48.44 ± 3.05</td><td>71.08 ± 3.16</td><td>73.03 ± 2.85</td><td>74.71 ± 3.01</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>82.62 ± 2.43</td><td>75.86 ± 2.95</td><td>68.12 ± 2.83</td><td>70.73 ± 2.64</td><td>72.51 ± 2.51</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>85.05 ± 2.61</td><td>33.99 ± 2.98</td><td>75.12 ± 2.79</td><td>76.74 ± 2.90</td><td>77.79 ± 2.64</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>84.06 ± 3.11</td><td>40.07 ± 2.95</td><td>71.11 ± 3.44</td><td>73.36 ± 3.29</td><td>75.67 ± 3.36</td></tr><tr><td>FedMeta</td><td>81.40 ± 3.08</td><td>47.33 ± 3.45</td><td>67.41 ± 3.19</td><td>70.87 ± 2.88</td><td>72.49 ± 3.17</td></tr><tr><td>q-FFL</td><td>82.30 ± 2.84</td><td>55.06 ± 3.33</td><td>67.82 ± 3.15</td><td>71.53 ± 3.29</td><td>73.70 ± 2.95</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>84.25 ± 3.01</td><td>53.88 ± 2.38</td><td>71.62 ± 3.11</td><td>73.48 ± 3.17</td><td>75.92 ± 3.09</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>84.92 ± 2.88</td><td>39.82 ± 2.54</td><td>75.04 ± 3.09</td><td>77.45 ± 2.94</td><td>78.93 ± 3.04</td></tr></table>

Table 31: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario II, on the CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 30)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>82.70 ± 3.15</td><td>62.09 ± 2.44</td><td>66.99 ± 3.53</td><td>71.18 ± 3.29</td><td>73.18 ± 3.47</td></tr><tr><td>FedMeta</td><td>83.00 ± 3.47</td><td>39.94 ± 3.08</td><td>71.16 ± 3.69</td><td>73.43 ± 3.54</td><td>75.52 ± 3.66</td></tr><tr><td>q-FFL</td><td>82.81 ± 3.21</td><td>44.09 ± 3.35</td><td>71.82 ± 3.47</td><td>73.68 ± 3.33</td><td>75.31 ± 3.58</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>85.05 ± 3.07</td><td>37.33 ± 2.94</td><td>75.16 ± 2.99</td><td>76.79 ± 3.30</td><td>78.21 ± 3.41</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>85.78 ± 2.95</td><td>35.88 ± 3.00</td><td>75.26 ± 3.17</td><td>78.60 ± 3.04</td><td>80.55 ± 3.29</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>81.14 ± 3.02</td><td>71.23 ± 3.33</td><td>65.42 ± 3.19</td><td>69.17 ± 3.35</td><td>70.99 ± 3.29</td></tr><tr><td>FedMeta</td><td>78.98 ± 3.17</td><td>64.48 ± 2.98</td><td>63.97 ± 3.05</td><td>66.89 ± 3.32</td><td>69.06 ± 3.19</td></tr><tr><td>q-FFL</td><td>79.87 ± 2.94</td><td>70.06 ± 3.10</td><td>63.96 ± 2.75</td><td>67.47 ± 3.47</td><td>70.16 ± 3.96</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>83.21 ± 3.06</td><td>37.94 ± 3.28</td><td>72.75 ± 2.96</td><td>74.53 ± 3.27</td><td>76.01 ± 3.48</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>85.11 ± 2.93</td><td>36.84 ± 3.07</td><td>72.66 ± 2.88</td><td>75.63 ± 3.15</td><td>77.47 ± 3.22</td></tr><tr><td colspan="2">Pre-training (Scenario II, |m| = 15)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>85.79 ± 1.47</td><td>16.16 ± 2.35</td><td>77.34 ± 1.69</td><td>78.37 ± 1.71</td><td>80.17 ± 1.53</td></tr><tr><td>FedMeta</td><td>85.88 ± 1.15</td><td>17.47 ± 1.98</td><td>77.49 ± 1.33</td><td>78.93 ± 1.28</td><td>80.33 ± 1.20</td></tr><tr><td>q-FFL</td><td>85.24 ± 1.29</td><td>15.60 ± 2.04</td><td>77.38 ± 1.47</td><td>78.37 ± 1.35</td><td>80.23 ± 1.56</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>85.37 ± 1.02</td><td>14.82 ± 1.74</td><td>77.49 ± 1.28</td><td>79.00 ± 1.33</td><td>80.33 ± 1.26</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>86.64 ± 1.14</td><td>14.59 ± 1.61</td><td>80.23 ± 1.30</td><td>81.17 ± 1.25</td><td>82.06 ± 1.22</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>85.17 ± 1.33</td><td>16.56 ± 2.04</td><td>78.21 ± 1.49</td><td>79.73 ± 1.62</td><td>80.52 ± 1.41</td></tr><tr><td>FedMeta</td><td>85.76 ± 1.37</td><td>18.40 ± 1.95</td><td>78.93 ± 1.29</td><td>80.52 ± 1.47</td><td>81.39 ± 1.62</td></tr><tr><td>q-FFL</td><td>86.29 ± 2.00</td><td>18.06 ± 1.92</td><td>78.79 ± 1.91</td><td>80.38 ± 2.17</td><td>81.58 ± 2.35</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>85.49 ± 1.61</td><td>13.84 ± 2.03</td><td>79.65 ± 1.38</td><td>80.66 ± 1.40</td><td>82.07 ± 1.35</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>86.68 ± 1.35</td><td>12.67 ± 1.78</td><td>80.09 ± 1.52</td><td>81.02 ± 1.39</td><td>82.36 ± 1.44</td></tr></table>

Table 32: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario II, on the CIFAR-100 dataset.

Table 33: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 20)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>85.08 ± 1.62</td><td>14.29 ± 2.05</td><td>78.21 ± 1.89</td><td>79.44 ± 1.91</td><td>80.52 ± 1.99</td></tr><tr><td>FedMeta</td><td>85.39 ± 1.27</td><td>19.89 ± 1.98</td><td>78.33 ± 1.54</td><td>79.30 ± 1.66</td><td>80.48 ± 1.64</td></tr><tr><td>q-FFL</td><td>85.41 ± 1.39</td><td>20.70 ± 2.17</td><td>77.63 ± 1.64</td><td>79.22 ± 1.38</td><td>80.28 ± 1.72</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>85.57 ± 1.33</td><td>17.89 ± 2.00</td><td>78.64 ± 1.53</td><td>80.01 ± 1.49</td><td>80.95 ± 1.21</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>86.77 ± 1.11</td><td>12.25 ± 2.00</td><td>80.52 ± 1.39</td><td>81.17 ± 1.53</td><td>81.96 ± 1.17</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>85.15 ± 1.99</td><td>20.98 ± 1.48</td><td>79.04 ± 1.68</td><td>80.45 ± 2.07</td><td>81.34 ± 2.11</td></tr><tr><td>FedMeta</td><td>85.38 ± 2.05</td><td>14.82 ± 1.99</td><td>78.79 ± 2.44</td><td>80.59 ± 2.31</td><td>81.58 ± 2.25</td></tr><tr><td>q-FFL</td><td>85.46 ± 1.65</td><td>19.71 ± 2.07</td><td>78.81 ± 1.49</td><td>80.11 ± 2.23</td><td>81.97 ± 1.84</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>85.57 ± 2.00</td><td>18.75 ± 1.84</td><td>79.65 ± 2.37</td><td>81.10 ± 2.15</td><td>82.06 ± 1.99</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>86.74 ± 1.63</td><td>12.82 ± 1.55</td><td>80.66 ± 1.27</td><td>81.60 ± 2.19</td><td>82.49 ± 1.88</td></tr></table>

Table 34: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 25)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>85.99 ± 0.97</td><td>16.65 ± 2.34</td><td>78.50 ± 1.33</td><td>79.73 ± 1.15</td><td>80.66 ± 1.39</td></tr><tr><td>FedMeta</td><td>85.58 ± 1.15</td><td>19.89 ± 1.46</td><td>78.21 ± 0.99</td><td>79.80 ± 1.37</td><td>80.81 ± 1.33</td></tr><tr><td>q-FFL</td><td>85.66 ± 1.32</td><td>17.22 ± 2.08</td><td>78.07 ± 1.59</td><td>79.73 ± 1.27</td><td>80.71 ± 1.55</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>86.33 ± 1.09</td><td>15.44 ± 1.28</td><td>80.63 ± 1.17</td><td>81.35 ± 1.30</td><td>82.28 ± 1.42</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>86.72 ± 1.03</td><td>15.29 ± 1.17</td><td>80.94 ± 1.10</td><td>81.59 ± 1.19</td><td>82.20 ± 1.25</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>85.50 ± 1.30</td><td>16.48 ± 2.11</td><td>78.35 ± 1.67</td><td>79.80 ± 1.41</td><td>80.86 ± 1.29</td></tr><tr><td>FedMeta</td><td>86.57 ± 1.62</td><td>17.81 ± 1.59</td><td>78.93 ± 2.07</td><td>79.80 ± 1.89</td><td>80.86 ± 1.74</td></tr><tr><td>q-FFL</td><td>86.45 ± 1.17</td><td>14.82 ± 2.22</td><td>79.08 ± 1.58</td><td>80.74 ± 1.31</td><td>81.96 ± 1.25</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>86.61 ± 1.44</td><td>13.62 ± 1.87</td><td>79.37 ± 1.71</td><td>80.45 ± 1.39</td><td>81.19 ± 1.25</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>87.16 ± 1.22</td><td>10.43 ± 1.33</td><td>80.38 ± 1.48</td><td>81.39 ± 1.45</td><td>82.20 ± 1.20</td></tr></table>

Table 35: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 30)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>85.27 ± 1.14</td><td>14.90 ± 2.88</td><td>77.92 ± 1.52</td><td>79.15 ± 1.39</td><td>80.04 ± 1.40</td></tr><tr><td>FedMeta</td><td>85.61 ± 1.33</td><td>13.54 ± 2.05</td><td>78.07 ± 1.62</td><td>79.87 ± 1.27</td><td>81.14 ± 1.36</td></tr><tr><td>q-FFL</td><td>85.34 ± 1.73</td><td>17.22 ± 1.98</td><td>80.37 ± 1.89</td><td>81.39 ± 1.61</td><td>82.15 ± 1.52</td></tr><tr><td>CoPreFL-SGD (γ = 1.0)</td><td>85.61 ± 1.24</td><td>12.25 ± 2.00</td><td>79.94 ± 1.45</td><td>80.74 ± 1.30</td><td>81.29 ± 1.33</td></tr><tr><td>CoPreFL (γ = 1.0)</td><td>86.62 ± 1.20</td><td>11.69 ± 1.86</td><td>81.24 ± 1.47</td><td>81.89 ± 1.42</td><td>82.64 ± 1.38</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>85.70 ± 1.49</td><td>20.61 ± 2.37</td><td>78.64 ± 1.66</td><td>80.09 ± 1.58</td><td>81.19 ± 1.62</td></tr><tr><td>FedMeta</td><td>85.44 ± 1.33</td><td>16.56 ± 2.01</td><td>79.08 ± 1.63</td><td>80.38 ± 1.24</td><td>81.34 ± 1.59</td></tr><tr><td>q-FFL</td><td>85.54 ± 1.18</td><td>17.56 ± 2.56</td><td>79.22 ± 1.44</td><td>80.66 ± 1.39</td><td>81.67 ± 1.83</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>85.79 ± 1.37</td><td>14.21 ± 2.15</td><td>79.65 ± 1.89</td><td>80.59 ± 1.55</td><td>81.43 ± 1.64</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>86.15 ± 1.20</td><td>13.25 ± 1.95</td><td>80.66 ± 1.38</td><td>81.89 ± 1.40</td><td>82.68 ± 1.61</td></tr><tr><td colspan="2">Pre-training (Scenario II, |m| = 15)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>82.50 ± 2.37</td><td>39.06 ± 3.69</td><td>70.33 ± 2.55</td><td>73.19 ± 2.68</td><td>75.15 ± 2.49</td></tr><tr><td>FedMeta</td><td>81.57 ± 2.52</td><td>62.57 ± 2.91</td><td>67.65 ± 3.01</td><td>70.95 ± 2.87</td><td>72.83 ± 2.64</td></tr><tr><td>q-FFL</td><td>82.31 ± 2.07</td><td>48.16 ± 2.54</td><td>70.00 ± 2.29</td><td>72.07 ± 2.37</td><td>73.99 ± 2.17</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>83.11 ± 2.33</td><td>38.94 ± 2.58</td><td>72.06 ± 2.61</td><td>73.63 ± 2.34</td><td>75.29 ± 2.28</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>84.68 ± 1.92</td><td>33.76 ± 2.60</td><td>73.84 ± 2.33</td><td>75.50 ± 2.59</td><td>77.35 ± 2.31</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>80.19 ± 2.61</td><td>53.00 ± 3.04</td><td>67.13 ± 2.95</td><td>69.67 ± 2.66</td><td>71.36 ± 2.87</td></tr><tr><td>FedMeta</td><td>81.94 ± 2.94</td><td>56.40 ± 3.15</td><td>67.15 ± 3.01</td><td>71.39 ± 3.14</td><td>73.39 ± 3.00</td></tr><tr><td>q-FFL</td><td>81.64 ± 2.38</td><td>54.46 ± 2.94</td><td>69.58 ± 2.67</td><td>71.65 ± 3.05</td><td>73.02 ± 2.14</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>83.57 ± 2.66</td><td>41.22 ± 3.33</td><td>71.46 ± 2.79</td><td>73.45 ± 3.05</td><td>75.41 ± 2.93</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>84.26 ± 2.41</td><td>28.52 ± 3.15</td><td>73.61 ± 2.96</td><td>75.55 ± 3.01</td><td>76.79 ± 2.90</td></tr></table>

Table 36: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

Table 37: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 15 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 20)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>81.91 ± 2.91</td><td>77.97 ± 3.00</td><td>64.37 ± 3.19</td><td>71.67 ± 3.38</td><td>74.51 ± 3.04</td></tr><tr><td>FedMeta</td><td>81.58 ± 3.14</td><td>38.56 ± 2.93</td><td>70.94 ± 3.61</td><td>71.82 ± 3.24</td><td>73.09 ± 3.11</td></tr><tr><td>q-FFL</td><td>82.17 ± 3.35</td><td>48.58 ± 3.17</td><td>70.22 ± 3.92</td><td>72.66 ± 3.48</td><td>74.24 ± 3.69</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>82.32 ± 3.07</td><td>42.25 ± 3.25</td><td>71.61 ± 3.19</td><td>73.31 ± 3.27</td><td>74.62 ± 3.38</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>84.48 ± 3.00</td><td>35.64 ± 2.73</td><td>73.66 ± 3.11</td><td>74.75 ± 3.30</td><td>76.21 ± 3.15</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>82.87 ± 3.19</td><td>48.16 ± 2.94</td><td>68.94 ± 3.38</td><td>72.91 ± 3.49</td><td>75.28 ± 3.26</td></tr><tr><td>FedMeta</td><td>84.19 ± 2.93</td><td>49.70 ± 2.74</td><td>70.41 ± 3.16</td><td>72.63 ± 3.00</td><td>74.74 ± 3.38</td></tr><tr><td>q-FFL</td><td>83.51 ± 3.05</td><td>44.22 ± 3.22</td><td>69.91 ± 2.94</td><td>73.71 ± 3.14</td><td>76.01 ± 3.29</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>84.30 ± 2.77</td><td>36.24 ± 3.04</td><td>72.83 ± 2.99</td><td>75.64 ± 3.18</td><td>77.37 ± 3.11</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>84.72 ± 2.51</td><td>24.80 ± 3.00</td><td>75.84 ± 2.87</td><td>77.31 ± 3.13</td><td>78.50 ± 2.99</td></tr></table>

Table 38: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 25)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>83.42 ± 2.62</td><td>44.36 ± 3.09</td><td>70.04 ± 2.93</td><td>72.88 ± 2.77</td><td>75.10 ± 2.95</td></tr><tr><td>FedMeta</td><td>80.66 ± 2.38</td><td>45.29 ± 3.19</td><td>68.38 ± 2.95</td><td>70.89 ± 2.88</td><td>73.05 ± 2.49</td></tr><tr><td>q-FFL</td><td>83.60 ± 2.56</td><td>34.93 ± 2.94</td><td>74.41 ± 3.01</td><td>75.86 ± 2.88</td><td>77.29 ± 2.79</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>84.49 ± 3.07</td><td>46.51 ± 2.89</td><td>72.66 ± 3.35</td><td>74.48 ± 3.18</td><td>75.97 ± 2.99</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>84.81 ± 2.40</td><td>32.15 ± 3.04</td><td>76.65 ± 2.83</td><td>77.97 ± 2.91</td><td>79.12 ± 2.60</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>79.44 ± 3.04</td><td>64.80 ± 2.79</td><td>65.35 ± 3.19</td><td>68.25 ± 3.33</td><td>70.19 ± 3.46</td></tr><tr><td>FedMeta</td><td>81.22 ± 2.99</td><td>48.58 ± 3.07</td><td>69.53 ± 3.30</td><td>71.87 ± 3.18</td><td>73.52 ± 3.29</td></tr><tr><td>q-FFL</td><td>82.14 ± 3.17</td><td>41.60 ± 3.66</td><td>72.32 ± 3.40</td><td>74.56 ± 3.52</td><td>76.31 ± 3.40</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>82.88 ± 2.68</td><td>35.64 ± 3.11</td><td>71.25 ± 3.29</td><td>73.00 ± 2.94</td><td>74.32 ± 3.00</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>84.02 ± 2.87</td><td>24.01 ± 3.04</td><td>75.49 ± 3.08</td><td>77.02 ± 2.95</td><td>78.17 ± 3.14</td></tr></table>

Table 39: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 25 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 30)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>83.63 ± 2.85</td><td>42.25 ± 3.32</td><td>71.27 ± 3.09</td><td>73.56 ± 2.73</td><td>75.00 ± 3.11</td></tr><tr><td>FedMeta</td><td>83.41 ± 3.17</td><td>39.82 ± 2.98</td><td>72.34 ± 3.05</td><td>72.81 ± 3.34</td><td>73.15 ± 3.29</td></tr><tr><td>q-FFL</td><td>82.98 ± 2.66</td><td>63.68 ± 1.48</td><td>62.50 ± 2.97</td><td>65.62 ± 3.19</td><td>68.31 ± 3.31</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>83.66 ± 2.57</td><td>42.90 ± 2.66</td><td>71.84 ± 2.86</td><td>74.30 ± 3.07</td><td>75.70 ± 3.33</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>84.26 ± 2.53</td><td>39.31 ± 2.83</td><td>73.77 ± 2.91</td><td>75.91 ± 2.70</td><td>77.58 ± 2.89</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>83.24 ± 3.01</td><td>42.25 ± 2.77</td><td>69.06 ± 2.98</td><td>73.39 ± 3.45</td><td>75.31 ± 3.33</td></tr><tr><td>FedMeta</td><td>81.61 ± 2.64</td><td>47.89 ± 3.11</td><td>69.38 ± 2.98</td><td>72.47 ± 3.01</td><td>74.39 ± 2.67</td></tr><tr><td>q-FFL</td><td>81.92 ± 3.14</td><td>51.55 ± 2.95</td><td>69.73 ± 2.83</td><td>72.17 ± 3.04</td><td>74.38 ± 3.15</td></tr><tr><td>CoPreFL-SGD (γ = 0.0)</td><td>83.37 ± 2.99</td><td>46.38 ± 3.04</td><td>72.17 ± 2.65</td><td>74.15 ± 3.27</td><td>75.53 ± 3.08</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>85.45 ± 2.83</td><td>38.32 ± 3.10</td><td>74.43 ± 2.91</td><td>75.90 ± 3.05</td><td>77.45 ± 3.00</td></tr></table>

Table 40: Average performance across 10 non-IID downstream FL tasks, initialized with various FL pre-trained methods using 30 out of 100 participants in scenario II, on the Tiny-ImageNet dataset.

<table><tr><td>Pre-training (Scenario II)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td>FedAvg</td><td>83.15 ± 3.10</td><td>51.19 ± 3.64</td><td>70.33 ± 2.97</td><td>72.17 ± 3.05</td><td>73.49 ± 3.17</td></tr><tr><td>FedMeta</td><td>81.77 ± 2.33</td><td>39.52 ± 2.58</td><td>69.71 ± 2.64</td><td>71.55 ± 2.97</td><td>73.68 ± 2.66</td></tr><tr><td>q-FFL</td><td>83.39 ± 2.71</td><td>51.77 ± 2.39</td><td>70.63 ± 2.95</td><td>72.91 ± 2.73</td><td>74.28 ± 3.08</td></tr><tr><td>CoPreFL-SGD ( $\gamma = 0.25$ )</td><td>83.63 ± 3.00</td><td>41.73 ± 2.85</td><td>69.76 ± 2.94</td><td>73.46 ± 3.09</td><td>75.64 ± 3.15</td></tr><tr><td>CoPreFL ( $\gamma = 0.25$ )</td><td>86.63 ± 2.93</td><td>31.58 ± 2.64</td><td>73.05 ± 2.51</td><td>75.82 ± 2.88</td><td>77.58 ± 2.99</td></tr></table>

Table 41: Average performance across 10 non-IID downstream FL tasks, initialized with various non-IID FL pre-trained models. Note that, for FedAvg, FedMeta, and q-FFL baselines, the dataset is distributed from all clients' and server's data without a further refining step, while our method is trained on clients' data and refined using the server's dataset.

<table><tr><td>Pre-training</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>CoPreFL (default: 5%)</td><td>86.63</td><td>31.58</td></tr><tr><td>CoPreFL (2%)</td><td>84.18</td><td>37.97</td></tr><tr><td>CoPreFL (1%)</td><td>83.69</td><td>34.61</td></tr></table>

Table 42: The results of varying the amount of the server's data $|D^S|$ for our method using the CIFAR-100 dataset.

![](images/3338ab783d3fc67ce021eba604fb4c5b8e8091968f9815b89d4dd56be4babac8.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 70                   | 0                      | 0                       | 0       |
| 75                   | 0                      | 0                       | 0       |
| 80                   | 5                      | 5                       | 5       |
| 85                   | 15                     | 15                      | 15      |
| 90                   | 20                     | 20                      | 20      |
| 95                   | 10                     | 10                      | 10      |
| 100                  | 0                      | 0                       | 0       |
</details>

(a) Pre-training: 20 clients IID FL

![](images/de6b90fbc366c1caf60510b0289157e0e604490323e59d1e6cbc7d59221bf9a4.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 75                   | 1                      | 1                       | 1       |
| 80                   | 5                      | 3                       | 2       |
| 85                   | 10                     | 8                       | 10      |
| 90                   | 25                     | 15                      | 25      |
| 95                   | 10                     | 5                       | 10      |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/c6b203e2cc79885ac149e1cd8acf522e3e8d112b45401bd3c8b2078997eca04a.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 75                   | 0                      | 0                       | 0       |
| 80                   | 15                     | 10                      | 5       |
| 85                   | 20                     | 25                      | 15      |
| 90                   | 30                     | 20                      | 20      |
| 95                   | 10                     | 5                       | 5       |
| 100                  | 0                      | 0                       | 0       |
</details>

(c) Pre-training: 30 clients IID FL

![](images/5cfe1ae02ee486995172d2943e60201e19ebe6fef4eddc2d0ebe5430642f8fad.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedAvg(Var runner-up) | FedMeta(Acc runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 75                   | 1                      | 3                       | 1       |
| 80                   | 12                     | 8                       | 10      |
| 85                   | 18                     | 15                      | 20      |
| 90                   | 16                     | 14                      | 18      |
| 95                   | 1                      | 1                       | 1       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 7: The distributions of testing accuracy in IID FL downstream tasks under various pre-training setups in scenario II on the CIFAR-100 dataset.

![](images/c3742d9c6059109e4bbeabca6b8084d006605721ccc91d67274a5c7b28fdd124.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | q-FFL(Acc+Var runner-up) | CoPreFL |
|----------------------|--------------------------|---------|
| 65                   | 2.5                      | 1.0     |
| 70                   | 3.0                      | 1.5     |
| 75                   | 4.0                      | 2.0     |
| 80                   | 6.0                      | 5.0     |
| 85                   | 10.0                     | 12.0    |
| 90                   | 8.0                      | 15.0    |
| 95                   | 2.0                      | 5.0     |
</details>

(a) Pre-training: 20 clients IID FL

![](images/935bee23e3de07343b4199b3f31577add1bdeb287b84cfadbb5357b1871c9203.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 60                   | 0                      | 0                       | 0       |
| 70                   | 5                      | 3                       | 2       |
| 80                   | 10                     | 15                      | 12      |
| 90                   | 12                     | 10                      | 22      |
| 100                  | 0                      | 0                       | 0       |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/89df910ada1e50ff426ccf036e910320a75e4f6075374546bbf44a2808dad081.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta(Acc+Var runner-up) | CoPreFL |
| --------------------- | --------------------------- | ------- |
| 50                    | 0.5                         | 0.2     |
| 60                    | 1.0                         | 0.5     |
| 70                    | 3.0                         | 4.0     |
| 80                    | 8.0                         | 12.0    |
| 90                    | 17.0                        | 15.0    |
| 100                   | 0.5                         | 0.5     |
</details>

(c) Pre-training: 30 clients IID FL

![](images/bd5816fc407c9cc0f4ab8f210dda5803a9ff8c740a29783382b5b0087dbe2e6e.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|--------------------------|---------|
| 50                   | 0                      | 0                        | 0       |
| 60                   | 2                      | 3                        | 1       |
| 70                   | 4                      | 6                        | 2       |
| 80                   | 8                      | 10                       | 5       |
| 90                   | 12                     | 14                       | 15      |
| 100                  | 2                      | 1                        | 0       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 8: The distributions of testing accuracy in non-IID FL downstream tasks under various pre-training setups in scenario II on the CIFAR-100 dataset

![](images/7eeaf89c50c10f5c7d6d69d02842292a27fa47a376bc492185921051f04210bc.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Var runner-up) | FedMeta(Acc runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 70                   | 0                      | 0                       | 0       |
| 75                   | 0                      | 0                       | 0       |
| 80                   | 15                     | 5                       | 10      |
| 85                   | 25                     | 15                      | 20      |
| 90                   | 15                     | 10                      | 15      |
| 95                   | 0                      | 0                       | 0       |
</details>

(a) Pre-training: 20 clients IID FL

![](images/933e0977e65b8db1b1cc307c1b337a87c6f471712f1c3d5dee404e167a3b4db1.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta(Acc+Var runner-up) | CoPeFL |
|----------------------|----------------------------|---------|
| 70                   | 1                          | 1       |
| 75                   | 3                          | 2       |
| 80                   | 10                         | 5       |
| 85                   | 15                         | 25      |
| 90                   | 12                         | 10      |
| 95                   | 2                          | 1       |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/36dea6c095603efedb95bdb468c6915058d7f1ee8a61fd4a66dd46341679c2fe.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta(Acc+Var runner-up) | CoPreFL |
| --------------------- | -------------------------- | ------- |
| 75                    | 2                          | 1       |
| 80                    | 10                         | 5       |
| 85                    | 35                         | 20      |
| 90                    | 20                         | 25      |
| 95                    | 5                          | 5       |
| 100                   | 1                          | 1       |
</details>

(c) Pre-training: 30 clients IID FL

![](images/a479a8e183e7eed6be445d58046175456b376fc2ab32f69db8eff8b24caf1f04.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMetaVar runner-up) | CoPreFL |
|----------------------|------------------------|------------------------|---------|
| 70                   | 1                      | 1                      | 1       |
| 75                   | 2                      | 2                      | 3       |
| 80                   | 5                      | 5                      | 8       |
| 85                   | 30                     | 25                     | 15      |
| 90                   | 10                     | 10                     | 18      |
| 95                   | 2                      | 2                      | 2       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 9: The distributions of testing accuracy in IID FL downstream tasks under various pre-training setups in scenario II on the Tiny-ImageNet dataset.

![](images/b17806724e56fc5d7403ef49665c2be4c8325c9a25b5bfb3e443cb5480faf12b.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc. runner-up) | FedMeta(Var. runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 40                   | 0                      | 0                       | 0       |
| 50                   | 0                      | 0                       | 0       |
| 60                   | 0                      | 2                       | 0       |
| 70                   | 0                      | 5                       | 0       |
| 80                   | 15                     | 10                      | 15      |
| 90                   | 10                     | 5                       | 15      |
| 100                  | 0                      | 0                       | 0       |
</details>

(a) Pre-training: 20 clients IID FL

![](images/ecec707f9d2a5d2d05b67f6bf097c4ef6dec73d8934564c0a0f1a891d9375215.jpg)

<details>
<summary>bar_line</summary>

| Testing Accuracy (%) | FedMeta(Acc runner-up) | q-FFL(Var runner-up) | CoPreFL |
|----------------------|------------------------|----------------------|---------|
| 50                   | 1                      | 0                    | 0       |
| 60                   | 2                      | 1                    | 0       |
| 70                   | 3                      | 2                    | 1       |
| 80                   | 5                      | 4                    | 3       |
| 90                   | 8                      | 7                    | 12      |
| 100                  | 10                     | 11                   | 14      |
</details>

(b) Pre-training: 20 clients Non-IID FL

![](images/9742b590971d8ccf99463d64cb52da3149b2a4abd00b69cf630e71d85dbc5b91.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc runner-up) | FedMeta(Var runner-up) | CoPreFL |
|----------------------|------------------------|-------------------------|---------|
| 50                   | 1                      | 0                       | 0       |
| 60                   | 2                      | 0                       | 0       |
| 70                   | 4                      | 0                       | 0       |
| 80                   | 6                      | 0                       | 0       |
| 90                   | 12                     | 10                      | 14      |
| 100                  | 1                      | 0                       | 0       |
</details>

(c) Pre-training: 30 clients IID FL

![](images/b1513cf851c8f60c7cee970c990bceb5e9210453c20a7cc8ee2a2abfe47f3d94.jpg)

<details>
<summary>bar</summary>

| Testing Accuracy (%) | FedAvg(Acc+Var runner-up) | CoPreFL |
|----------------------|---------------------------|---------|
| 50                   | 2                         | 0       |
| 60                   | 0                         | 0       |
| 70                   | 5                         | 5       |
| 80                   | 10                        | 10      |
| 90                   | 12                        | 15      |
| 100                  | 2                         | 2       |
</details>

(d) Pre-training: 30 clients Non-IID FL   
Figure 10: The distributions of testing accuracy in non-IID FL downstream tasks under various pre-training setups in scenario II on the Tiny-ImageNet dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>86.84 ± 1.77</td><td>2.43 ± 2.49</td><td>76.88 ± 1.58</td><td>79.87 ± 2.33</td><td>81.47 ± 2.05</td></tr><tr><td>FedMeta</td><td>82.16 ± 1.27</td><td>2.31 ± 2.05</td><td>75.32 ± 1.35</td><td>75.97 ± 1.08</td><td>76.62 ± 1.49</td></tr><tr><td>q-FFL</td><td>79.91 ± 2.03</td><td>3.09 ± 1.77</td><td>76.62 ± 2.94</td><td>77.40 ± 2.33</td><td>77.83 ± 2.69</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>91.59 ± 2.44</td><td>1.61 ± 1.96</td><td>86.75 ± 3.01</td><td>87.40 ± 2.95</td><td>88.31 ± 2.59</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>83.84 ± 2.55</td><td>2.13 ± 1.88</td><td>75.97 ± 2.07</td><td>77.88 ± 2.27</td><td>80.79 ± 2.88</td></tr><tr><td>FedMeta</td><td>86.71 ± 1.99</td><td>1.61 ± 2.01</td><td>74.69 ± 1.89</td><td>77.82 ± 2.33</td><td>78.21 ± 2.07</td></tr><tr><td>q-FFL</td><td>79.85 ± 2.00</td><td>2.53 ± 1.95</td><td>69.22 ± 1.84</td><td>71.95 ± 2.23</td><td>74.94 ± 1.95</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>89.01 ± 2.34</td><td>1.46 ± 1.99</td><td>81.29 ± 2.29</td><td>83.55 ± 2.44</td><td>84.24 ± 2.21</td></tr></table>

Table 43: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario I, on the FEMNIST dataset.

<table><tr><td colspan="2">Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="4">IID</td><td>FedAvg</td><td>75.04 ± 2.47</td><td>16.16 ± 2.11</td><td>70.75 ± 2.95</td><td>71.65 ± 2.71</td><td>72.19 ± 2.53</td></tr><tr><td>FedMeta</td><td>69.61 ± 2.58</td><td>6.81 ± 2.35</td><td>59.19 ± 2.69</td><td>62.35 ± 2.77</td><td>64.54 ± 2.98</td></tr><tr><td>q-FFL</td><td>70.47 ± 1.96</td><td>19.30 ± 2.03</td><td>58.31 ± 2.27</td><td>61.66 ± 2.15</td><td>63.67 ± 2.00</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>78.38 ± 2.44</td><td>6.70 ± 2.01</td><td>72.77 ± 2.88</td><td>74.54 ± 3.05</td><td>75.35 ± 3.14</td></tr><tr><td rowspan="4">Non-IID</td><td>FedAvg</td><td>70.74 ± 2.93</td><td>28.58 ± 2.66</td><td>65.06 ± 3.05</td><td>66.20 ± 2.89</td><td>66.99 ± 3.19</td></tr><tr><td>FedMeta</td><td>64.02 ± 2.34</td><td>33.29 ± 1.62</td><td>60.91 ± 2.99</td><td>61.57 ± 3.15</td><td>62.07 ± 2.83</td></tr><tr><td>q-FFL</td><td>68.04 ± 3.85</td><td>31.55 ± 3.13</td><td>58.09 ± 3.25</td><td>60.57 ± 2.95</td><td>61.72 ± 3.30</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>72.65 ± 2.88</td><td>24.89 ± 3.01</td><td>67.49 ± 2.79</td><td>68.40 ± 3.22</td><td>69.32 ± 3.27</td></tr></table>

Table 44: Average performance across 10 Non-IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario I, on the FEMNIST dataset.

<table><tr><td colspan="2">Pre-training (Scenario II, |m| = 20)</td><td colspan="5">Downstream: IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>86.70 ± 2.00</td><td>6.86 ± 2.33</td><td>75.24 ± 1.92</td><td>77.94 ± 1.99</td><td>80.58 ± 1.85</td></tr><tr><td>FedMeta</td><td>81.29 ± 1.90</td><td>3.42 ± 2.35</td><td>70.82 ± 2.27</td><td>72.10 ± 1.93</td><td>74.87 ± 2.00</td></tr><tr><td>q-FFL</td><td>87.07 ± 1.77</td><td>2.85 ± 2.62</td><td>79.16 ± 2.03</td><td>83.52 ± 1.99</td><td>83.72 ± 2.24</td></tr><tr><td>CoPreFL-SGD (γ = 0.5)</td><td>86.31 ± 1.38</td><td>5.33 ± 1.24</td><td>77.16 ± 1.75</td><td>79.72 ± 1.68</td><td>81.23 ± 2.05</td></tr><tr><td>CoPreFL (γ = 0.5)</td><td>90.33 ± 1.94</td><td>2.22 ± 2.39</td><td>82.03 ± 2.47</td><td>83.54 ± 2.22</td><td>84.09 ± 2.15</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>85.24 ± 2.68</td><td>7.78 ± 3.45</td><td>77.01 ± 2.93</td><td>79.25 ± 2.47</td><td>80.52 ± 2.85</td></tr><tr><td>FedMeta</td><td>83.52 ± 2.37</td><td>5.76 ± 3.05</td><td>71.44 ± 3.30</td><td>75.37 ± 2.88</td><td>77.76 ± 2.95</td></tr><tr><td>q-FFL</td><td>87.11 ± 1.44</td><td>10.24 ± 3.48</td><td>74.83 ± 1.83</td><td>75.52 ± 1.65</td><td>76.44 ± 1.59</td></tr><tr><td>CoPreFL-SGD (γ = 0.25)</td><td>87.05 ± 2.98</td><td>11.22 ± 2.37</td><td>73.30 ± 3.01</td><td>76.42 ± 2.59</td><td>79.76 ± 3.02</td></tr><tr><td>CoPreFL (γ = 0.25)</td><td>89.01 ± 2.61</td><td>5.47 ± 2.97</td><td>79.63 ± 3.15</td><td>81.22 ± 2.66</td><td>82.71 ± 3.17</td></tr><tr><td colspan="2">Pre-training (Scenario I, |m| = 20)</td><td colspan="5">Downstream: Non-IID FedAvg</td></tr><tr><td>Distribution</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td rowspan="5">IID</td><td>FedAvg</td><td>71.25 ± 3.09</td><td>15.28 ± 2.83</td><td>54.99 ± 3.22</td><td>58.71 ± 3.71</td><td>60.31 ± 3.55</td></tr><tr><td>FedMeta</td><td>73.29 ± 2.61</td><td>16.89 ± 2.13</td><td>61.39 ± 2.97</td><td>65.22 ± 3.25</td><td>66.17 ± 2.61</td></tr><tr><td>q-FFL</td><td>77.93 ± 3.19</td><td>8.88 ± 2.88</td><td>65.91 ± 3.41</td><td>66.74 ± 3.15</td><td>68.03 ± 2.98</td></tr><tr><td>CoPreFL-SGD (γ = 0.0)</td><td>76.19 ± 2.83</td><td>9.30 ± 3.01</td><td>66.07 ± 3.39</td><td>66.62 ± 3.54</td><td>67.80 ± 2.99</td></tr><tr><td>CoPreFL (γ = 0.0)</td><td>82.33 ± 3.10</td><td>7.95 ± 2.95</td><td>68.31 ± 3.53</td><td>70.37 ± 3.20</td><td>72.19 ± 3.37</td></tr><tr><td rowspan="5">Non-IID</td><td>FedAvg</td><td>66.31 ± 3.57</td><td>21.06 ± 3.31</td><td>44.79 ± 2.99</td><td>50.93 ± 3.48</td><td>53.29 ± 3.69</td></tr><tr><td>FedMeta</td><td>71.49 ± 3.31</td><td>13.10 ± 2.81</td><td>58.31 ± 3.54</td><td>59.27 ± 2.92</td><td>61.33 ± 2.67</td></tr><tr><td>q-FFL</td><td>74.99 ± 3.17</td><td>29.26 ± 3.05</td><td>61.20 ± 3.08</td><td>63.98 ± 3.31</td><td>65.01 ± 2.96</td></tr><tr><td>CoPreFL-SGD (γ = 0.75)</td><td>72.66 ± 3.09</td><td>29.05 ± 2.99</td><td>58.71 ± 3.31</td><td>61.29 ± 2.77</td><td>63.32 ± 3.06</td></tr><tr><td>CoPreFL (γ = 0.75)</td><td>79.31 ± 3.29</td><td>9.55 ± 3.28</td><td>63.29 ± 3.71</td><td>65.33 ± 3.05</td><td>66.92 ± 3.59</td></tr></table>

Table 45: Average performance across 10 IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario II, on the FEMNIST dataset.

Table 46: Average performance across 10 Non-IID downstream FL tasks, initialized with various FL pre-trained methods using 20 out of 100 participants in scenario II, on the FEMNIST dataset.

<table><tr><td colspan="2">Pre-training</td><td colspan="2">Downstream: IID FedAvg</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Scenario</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td rowspan="6">I</td><td>Random</td><td>78.03 ± 0.74</td><td>16.17 ± 2.03</td><td>75.32 ± 1.68</td><td>41.39 ± 3.35</td></tr><tr><td>Centralized</td><td>83.17 ± 1.39</td><td>17.93 ± 1.45</td><td>81.30 ± 2.92</td><td>69.44 ± 2.33</td></tr><tr><td>SCAFFOLD</td><td>82.19 ± 0.95</td><td>33.26 ± 2.39</td><td>79.15 ± 3.08</td><td>57.84 ± 1.95</td></tr><tr><td>FedDyn</td><td>83.61 ± 1.11</td><td>20.31 ± 3.45</td><td>81.23 ± 2.96</td><td>53.17 ± 2.85</td></tr><tr><td>PerFedAvg</td><td>82.69 ± 1.17</td><td>22.38 ± 1.33</td><td>81.58 ± 1.83</td><td>49.73 ± 2.65</td></tr><tr><td>CoPreFL</td><td>86.32 ± 1.04</td><td>14.14 ± 1.58</td><td>83.29 ± 2.61</td><td>34.69 ± 3.17</td></tr><tr><td rowspan="6">II</td><td>Random</td><td>78.21 ± 1.06</td><td>16.44 ± 1.28</td><td>77.50 ± 2.71</td><td>53.00 ± 3.89</td></tr><tr><td>Centralized</td><td>84.39 ± 2.44</td><td>15.92 ± 1.54</td><td>82.07 ± 3.01</td><td>70.90 ± 4.65</td></tr><tr><td>SCAFFOLD</td><td>83.44 ± 1.17</td><td>17.39 ± 1.05</td><td>82.11 ± 1.94</td><td>63.41 ± 2.38</td></tr><tr><td>FedDyn</td><td>85.16 ± 1.53</td><td>19.22 ± 2.37</td><td>83.92 ± 2.19</td><td>56.70 ± 2.04</td></tr><tr><td>PerFedAvg</td><td>84.79 ± 0.96</td><td>13.28 ± 1.44</td><td>84.19 ± 2.00</td><td>54.32 ± 2.37</td></tr><tr><td>CoPreFL</td><td>87.02 ± 1.39</td><td>10.50 ± 1.47</td><td>86.63 ± 2.93</td><td>31.58 ± 2.64</td></tr></table>

Table 47: Average performance of 10 downstream FL tasks with various initializations on CIFAR-100 dataset.

<table><tr><td colspan="2">Pre-training</td><td colspan="2">Downstream: IID FedAvg</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Scenario</td><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td rowspan="6">I</td><td>Random</td><td>81.29 ± 2.39</td><td>18.33 ± 1.44</td><td>75.50 ± 3.58</td><td>54.88 ± 2.45</td></tr><tr><td>Centralized</td><td>83.81 ± 1.07</td><td>19.03 ± 1.39</td><td>83.19 ± 2.05</td><td>42.69 ± 2.61</td></tr><tr><td>SCAFFOLD</td><td>83.66 ± 0.99</td><td>20.29 ± 1.28</td><td>81.49 ± 2.65</td><td>44.76 ± 2.32</td></tr><tr><td>FedDyn</td><td>84.79 ± 1.00</td><td>31.00 ± 2.33</td><td>83.76 ± 3.01</td><td>39.25 ± 2.66</td></tr><tr><td>PerFedAvg</td><td>83.97 ± 1.39</td><td>26.33 ± 1.68</td><td>82.93 ± 2.49</td><td>38.93 ± 2.89</td></tr><tr><td>CoPreFL</td><td>86.00 ± 1.13</td><td>16.16 ± 2.00</td><td>85.23 ± 2.43</td><td>35.40 ± 2.75</td></tr><tr><td rowspan="6">II</td><td>Random</td><td>83.16 ± 0.79</td><td>16.08 ± 1.03</td><td>76.23 ± 1.93</td><td>61.62 ± 2.07</td></tr><tr><td>Centralized</td><td>84.36 ± 1.33</td><td>17.89 ± 2.01</td><td>82.39 ± 2.87</td><td>39.31 ± 3.15</td></tr><tr><td>SCAFFOLD</td><td>85.01 ± 0.95</td><td>19.67 ± 1.82</td><td>82.73 ± 2.04</td><td>41.69 ± 3.31</td></tr><tr><td>FedDyn</td><td>84.33 ± 1.62</td><td>21.30 ± 2.00</td><td>83.17 ± 1.91</td><td>30.66 ± 2.39</td></tr><tr><td>PerFedAvg</td><td>84.61 ± 1.99</td><td>22.71 ± 1.37</td><td>83.27 ± 2.08</td><td>33.79 ± 2.73</td></tr><tr><td>CoPreFL</td><td>86.74 ± 1.63</td><td>12.82 ± 1.55</td><td>84.72 ± 2.51</td><td>24.80 ± 3.00</td></tr></table>

Table 48: Average performance of 10 downstream FL tasks with various initializations on Tiny-ImageNet dataset.

<table><tr><td>Pre-training (Scenario I)</td><td colspan="5">Downstream: Non-IID FedProx ( $\mu = 1$ )</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td>Centralized</td><td>82.39 ± 3.17</td><td>51.46 ± 2.59</td><td>70.33 ± 3.00</td><td>71.28 ± 3.38</td><td>73.52 ± 3.30</td></tr><tr><td>FedAvg</td><td>79.53 ± 2.69</td><td>46.15 ± 3.04</td><td>63.17 ± 2.52</td><td>69.74 ± 2.94</td><td>71.59 ± 2.94</td></tr><tr><td>FedMeta</td><td>81.77 ± 3.29</td><td>63.12 ± 3.62</td><td>63.58 ± 3.55</td><td>68.19 ± 3.73</td><td>70.28 ± 3.22</td></tr><tr><td>q-FFL</td><td>83.19 ± 3.03</td><td>52.12 ± 2.97</td><td>67.41 ± 3.54</td><td>70.59 ± 3.30</td><td>72.33 ± 3.01</td></tr><tr><td>CoPreFL ( $\gamma = 0.25$ )</td><td>84.31 ± 3.01</td><td>30.55 ± 2.61</td><td>70.19 ± 2.95</td><td>73.88 ± 3.04</td><td>75.13 ± 3.10</td></tr><tr><td>Pre-training (Scenario I)</td><td colspan="5">Downstream: Non-IID q-FFL ( $q = 2$ )</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Lowest 10% ↑</td><td>Lowest 20% ↑</td><td>Lowest 30% ↑</td></tr><tr><td>Centralized</td><td> $79.26 \pm 2.33$ </td><td> $47.10 \pm 3.05$ </td><td> $68.31 \pm 2.95$ </td><td> $70.22 \pm 2.69$ </td><td> $71.39 \pm 3.02$ </td></tr><tr><td>FedAvg</td><td> $79.53 \pm 2.38$ </td><td> $44.59 \pm 2.95$ </td><td> $64.52 \pm 2.87$ </td><td> $68.93 \pm 3.02$ </td><td> $72.93 \pm 2.96$ </td></tr><tr><td>FedMeta</td><td> $79.30 \pm 3.02$ </td><td> $39.63 \pm 3.17$ </td><td> $65.63 \pm 3.35$ </td><td> $67.33 \pm 2.94$ </td><td> $71.53 \pm 3.18$ </td></tr><tr><td>q-FFL</td><td> $81.38 \pm 2.67$ </td><td> $37.27 \pm 2.85$ </td><td> $69.35 \pm 3.00$ </td><td> $71.63 \pm 2.98$ </td><td> $73.15 \pm 2.91$ </td></tr><tr><td>CoPreFL ( $\gamma = 0.25$ )</td><td> $\mathbf{82.71} \pm 2.45$ </td><td> $\mathbf{25.39} \pm 2.87$ </td><td> $\mathbf{71.66} \pm 2.90$ </td><td> $\mathbf{73.94} \pm 2.61$ </td><td> $\mathbf{76.29} \pm 2.88$ </td></tr></table>

Table 49: Average performance across 10 non-IID downstream FedProx tasks, initialized with centralized model and various non-IID FL pre-trained models.

Table 50: Average performance across 10 non-IID downstream q-FFL tasks, initialized with centralized model and various non-IID FL pre-trained models.

<table><tr><td rowspan="2">Pre-training (ImageNet)</td><td colspan="4">Dataset: CIFAR-100</td></tr><tr><td colspan="2">Downstream: IID FedAvg</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>Centralized</td><td>87.91 ± 0.99</td><td>13.96 ± 2.35</td><td>86.75 ± 2.89</td><td>67.34 ± 2.17</td></tr><tr><td>CoPreFL</td><td>88.39 ± 1.15</td><td>11.37 ± 2.03</td><td>87.96 ± 1.95</td><td>30.79 ± 2.79</td></tr></table>

(a) Results of downstream FL using CIFAR-100 dataset, initialized with a model pre-trained on ImageNet. 

<table><tr><td rowspan="2">Pre-training (ImageNet)</td><td colspan="4">Dataset: Tiny-ImageNet</td></tr><tr><td colspan="2">Downstream: IID FedAvg</td><td colspan="2">Downstream: Non-IID FedAvg</td></tr><tr><td>Method</td><td>Acc ↑</td><td>Variance ↓</td><td>Acc ↑</td><td>Variance ↓</td></tr><tr><td>Centralized</td><td>87.02 ± 1.33</td><td>15.92 ± 2.54</td><td>85.58 ± 1.95</td><td>50.93 ± 2.06</td></tr><tr><td>CoPreFL</td><td>88.94 ± 1.45</td><td>13.21 ± 1.98</td><td>86.79 ± 1.37</td><td>31.44 ± 3.11</td></tr></table>

(b) Results of downstream FL using Tiny-ImageNet dataset, initialized with a model pre-trained on ImageNet.   
Table 51: Results with pre-training on a centrally stored public dataset. ImageNet is used for pre-training, while CIFAR-100 and Tiny-ImageNet are used for downstream FL.